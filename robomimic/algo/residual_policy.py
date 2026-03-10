from typing import Callable, Union, Dict, List, Tuple
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import random
import numpy as np
import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from robomimic.config.config import Config
import robomimic.models.base_nets as BaseNets
import robomimic.models.value_nets as ValueNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.algo import register_algo_factory_func, PolicyAlgo, algo_factory, ResidualAlgo
from robomimic.models.policy_nets import ResidualGaussianActorNetwork, ResidualScaleNetwork

@register_algo_factory_func("residual_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the Residual Policy algo class.
    """
    return ResidualPolicy, {}

class ResidualPolicy(ResidualAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        Structure:
            - base_policy: The frozen Diffusion Policy
            - actor: Residual actor (outputs delta_a)
            - critic: Double Q-networks (outputs Q-value)
            - critic_target: Target Q-networks
        """
        # set up different observation groups for @MIMO_MLP
        print(f"ResidualPolicy: Loading base policy from {self.algo_config.base_policy.ckpt_path}")

        # algo name and config from model dict
        self.base_policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=self.algo_config.base_policy.ckpt_path, device=self.device, verbose=False)
        self.base_policy = self.base_policy.policy
        self.base_policy.set_eval()

        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        self.nets = nn.ModuleDict()
        base_obs_shapes = OrderedDict(self.obs_shapes)
        actor_obs_shapes = OrderedDict(base_obs_shapes)
        actor_obs_shapes["action"] = (self.ac_dim,)
        observation_horizon = self.base_policy.algo_config.horizon.observation_horizon

        # Create Residual Actor (SAC Style)
        self.nets["res_policy"] = ResidualGaussianActorNetwork(
            obs_shapes=actor_obs_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.residual.layer_dims,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=encoder_kwargs,
            observation_horizon=observation_horizon,
            use_tanh=True,
            low_noise_eval=False
        )

        self.learn_scale = bool(self.algo_config.residual.learn_scale)
        self.residual_scale = 1.0 if self.learn_scale else float(self.algo_config.residual.scale_factor)
        if self.learn_scale:
            self.nets["res_scale"] = ResidualScaleNetwork(
                obs_shapes=base_obs_shapes,
                mlp_layer_dims=self.algo_config.residual.layer_dims,
                goal_shapes=self.goal_shapes,
                encoder_kwargs=encoder_kwargs,
                observation_horizon=observation_horizon,
            )

        # Create Critic (Double Q)
        def create_critic_net():
            return ValueNets.ResidualActionValueNetwork(
                obs_shapes=base_obs_shapes,
                ac_dim=self.ac_dim,
                mlp_layer_dims=self.algo_config.critic.layer_dims,
                observation_horizon=observation_horizon,
                goal_shapes=self.goal_shapes,
                encoder_kwargs=encoder_kwargs,
            )
        
        self.nets["critic"] = nn.ModuleList([create_critic_net(), create_critic_net()])
        self.nets["critic_target"] = copy.deepcopy(self.nets["critic"])

        self.nets = self.nets.float().to(self.device)

        # Sync target networks initially
        for i in range(len(self.nets["critic"])):
            self._soft_update_target_network(self.nets["critic"][i], self.nets["critic_target"][i], 1.0)

        # Parameters for SAC
        self.gamma = self.algo_config.rl.gamma
        self.tau = self.algo_config.rl.tau
        self.alpha = self.algo_config.rl.alpha
        self.residual_reg_weight = float(self.algo_config.residual.get("regularization_weight", 1e-3))

        # Queues for inference (inherited from PolicyAlgo but we explicitly manage them)
        self.reset()

    def _create_optimizers(self):
        super(ResidualPolicy, self)._create_optimizers()
        if self.learn_scale:
            self.optimizers["res_scale"] = TorchUtils.optimizer_from_optim_params(
                net_optim_params=self.optim_params["res_policy"],
                net=self.nets["res_scale"],
            )
            self.lr_schedulers["res_scale"] = TorchUtils.lr_scheduler_from_optim_params(
                net_optim_params=self.optim_params["res_policy"],
                net=self.nets["res_scale"],
                optimizer=self.optimizers["res_scale"],
            )
            self.step_lr_schedulers_every_batch["res_scale"] = self.step_lr_schedulers_every_batch["res_policy"]

    def process_batch_for_training(self, batch):
        """
        Standard RL batch processing.
        Expected batch keys: 'obs', 'actions', 'rewards', 'next_obs', 'dones'
        """
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k] for k in batch["obs"]}
        input_batch["next_obs"] = {k: batch["next_obs"][k] for k in batch["next_obs"]}
        input_batch["actions"] = batch["actions"]
        input_batch["rewards"] = batch["rewards"]
        input_batch["dones"] = batch["dones"]

        if "base_actions" in batch:
            input_batch["base_actions"] = batch["base_actions"]

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def _get_base_action_tensor(self, obs_dict):
        """
        Compute base-policy actions for a batch of observations.
        """
        with torch.no_grad():
            if hasattr(self.base_policy, "_get_action_trajectory"):
                # Diffusion-style base policy: ensure temporal horizon is satisfied.
                To = self.base_policy.algo_config.horizon.observation_horizon
                base_obs = dict()
                for k, v in obs_dict.items():
                    target_rank = len(self.obs_key_shapes[k])
                    if v.ndim == target_rank + 1:
                        v = v.unsqueeze(1)
                    if v.ndim == target_rank + 2:
                        if v.shape[1] < To:
                            pad = v[:, -1:].repeat(1, To - v.shape[1], *([1] * (v.ndim - 2)))
                            v = torch.cat((v, pad), dim=1)
                        elif v.shape[1] > To:
                            v = v[:, -To:, ...]
                    base_obs[k] = v

                traj = self.base_policy._get_action_trajectory(obs_dict=base_obs)
                if traj.ndim == 3:
                    return traj[:, 0, :]
                return traj
            
            return self.base_policy.get_action(obs_dict=obs_dict)

    def _obs_with_action(self, obs_dict, action):
        """
        Create critic / residual-policy inputs by attaching action to observation dict.
        """
        out = {k: v for k, v in obs_dict.items()}
        ref = next(iter(out.values()))

        if ref.ndim >= 3:
            T = ref.shape[1]
            if action.ndim == 2:
                action = action.unsqueeze(1).expand(-1, T, -1)
            elif action.ndim == 3:
                if action.shape[1] < T:
                    pad = action[:, -1:].expand(-1, T - action.shape[1], -1)
                    action = torch.cat((action, pad), dim=1)
                elif action.shape[1] > T:
                    action = action[:, -T:, :]
        elif action.ndim == 3:
            action = action[:, -1, :]

        out["action"] = action
        return out

    def _sample_residual_and_log_prob(self, obs_with_action):
        dist = self.nets["res_policy"].forward_train(obs_dict=obs_with_action)
        if self.nets["res_policy"].use_tanh:
            residual, pre_tanh = dist.rsample(return_pretanh_value=True)
            log_prob = dist.log_prob(residual, pre_tanh_value=pre_tanh).unsqueeze(-1)
        else:
            residual = dist.rsample()
            log_prob = dist.log_prob(residual).unsqueeze(-1)
        return residual, log_prob

    def _get_residual_scale(self, obs_dict, goal_dict=None):
        if not self.learn_scale:
            return self.residual_scale
        return self.residual_scale * self.nets["res_scale"](obs_dict=obs_dict, goal_dict=goal_dict)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        SAC Update Logic with Residuals.
        """
        info = PolicyAlgo.train_on_batch(self, batch, epoch, validate=validate)

        obs = batch["obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"].reshape(-1, 1)
        dones = batch["dones"].reshape(-1, 1)

        with torch.no_grad():
            base_next_action = self._get_base_action_tensor(next_obs)
            next_res_obs = self._obs_with_action(next_obs, base_next_action)
            next_residual, next_log_prob = self._sample_residual_and_log_prob(next_res_obs)
            next_scale = self._get_residual_scale(next_obs)
            next_action = torch.clamp(base_next_action + next_scale * next_residual, -1.0, 1.0)

            q1_next = self.nets["critic_target"][0](obs_dict=next_obs, acts=next_action)
            q2_next = self.nets["critic_target"][1](obs_dict=next_obs, acts=next_action)
            q_min_next = torch.min(q1_next, q2_next)
            target_q = rewards + (1.0 - dones) * self.gamma * (q_min_next - self.alpha * next_log_prob)

        q1 = self.nets["critic"][0](obs_dict=obs, acts=actions)
        q2 = self.nets["critic"][1](obs_dict=obs, acts=actions)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        critic_loss = critic1_loss + critic2_loss

        if not validate:
            for optimizer in self.optimizers["critic"]:
                optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            for optimizer in self.optimizers["critic"]:
                optimizer.step()

        with torch.no_grad():
            if "base_actions" in batch:
                base_action = batch["base_actions"]
            else:
                base_action = self._get_base_action_tensor(obs)

        for critic in self.nets["critic"]:
            for p in critic.parameters():
                p.requires_grad_(False)

        res_obs = self._obs_with_action(obs, base_action)
        residual, log_prob = self._sample_residual_and_log_prob(res_obs)
        residual_scale = self._get_residual_scale(obs)
        scaled_residual = residual_scale * residual
        new_action = torch.clamp(base_action + scaled_residual, -1.0, 1.0)
        q1_pi = self.nets["critic"][0](obs_dict=obs, acts=new_action)
        q2_pi = self.nets["critic"][1](obs_dict=obs, acts=new_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        reg_loss = self.residual_reg_weight * (scaled_residual ** 2).mean()
        actor_loss = (self.alpha * log_prob - min_q_pi).mean() + reg_loss

        if not validate:
            self.optimizers["res_policy"].zero_grad(set_to_none=True)
            if self.learn_scale:
                self.optimizers["res_scale"].zero_grad(set_to_none=True)
            actor_loss.backward()
            self.optimizers["res_policy"].step()
            if self.learn_scale:
                self.optimizers["res_scale"].step()

        for critic in self.nets["critic"]:
            for p in critic.parameters():
                p.requires_grad_(True)

        if not validate:
            for critic, critic_target in zip(self.nets["critic"], self.nets["critic_target"]):
                self._soft_update_target_network(critic, critic_target, self.tau)

        info["critic/loss"] = critic_loss.item()
        info["critic/critic1_loss"] = critic1_loss.item()
        info["critic/critic2_loss"] = critic2_loss.item()
        info["critic/q1"] = q1.mean().item()
        info["critic/q2"] = q2.mean().item()
        info["critic/target_q"] = target_q.mean().item()
        info["actor/loss"] = actor_loss.item()
        info["actor/log_prob"] = log_prob.mean().item()
        info["actor/residual_l2"] = (scaled_residual ** 2).mean().item()
        if torch.is_tensor(residual_scale):
            info["actor/residual_scale"] = residual_scale.mean().item()
        else:
            info["actor/residual_scale"] = float(residual_scale)
        return info
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(ResidualPolicy, self).log_info(info)
        if "critic/loss" in info and "actor/loss" in info:
            log["Loss"] = info["critic/loss"] + info["actor/loss"]
        elif "critic/loss" in info:
            log["Loss"] = info["critic/loss"]
        elif "actor/loss" in info:
            log["Loss"] = info["actor/loss"]

        if "critic/loss" in info:
            log["Critic/Loss"] = info["critic/loss"]
        if "critic/critic1_loss" in info:
            log["Critic/Critic1_Loss"] = info["critic/critic1_loss"]
        if "critic/critic2_loss" in info:
            log["Critic/Critic2_Loss"] = info["critic/critic2_loss"]
        if "critic/q1" in info:
            log["Critic/Q1"] = info["critic/q1"]
        if "critic/q2" in info:
            log["Critic/Q2"] = info["critic/q2"]
        if "critic/target_q" in info:
            log["Critic/Target_Q"] = info["critic/target_q"]

        if "actor/loss" in info:
            log["Actor/Loss"] = info["actor/loss"]
        if "actor/log_prob" in info:
            log["Actor/Log_Prob"] = info["actor/log_prob"]
        if "actor/residual_l2" in info:
            log["Actor/Residual_L2"] = info["actor/residual_l2"]
        if "actor/residual_scale" in info:
            log["Actor/Residual_Scale"] = info["actor/residual_scale"]
        return log
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self.base_policy.reset() # Reset internal queues of base policy
        

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs (Inference).
        returns a = a_base + a_residual
        """
        total_action, _, _ = self.get_action_with_base(obs_dict, goal_dict)
        return total_action

    def get_action_with_base(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs (Inference) with components.
        returns (total_action, base_action, residual_action)
        """
        To = self.base_policy.algo_config.horizon.observation_horizon

        with torch.no_grad():
            # Get Base Action
            base_action = self.base_policy.get_action(obs_dict=obs_dict, goal_dict=goal_dict)
            
            # Calculate Residual action
            actor_obs = {k: v for k, v in obs_dict.items()}
            actor_obs["action"] = base_action.unsqueeze(1).expand(-1, To, -1)
            res_action = self.nets["res_policy"](obs_dict=actor_obs, goal_dict=goal_dict)
            # res_action = self._get_residual_action(obs_dict=obs_dict, goal_dict=goal_dict)

        total_action = self.mix_actions(base_action, res_action, obs_dict=obs_dict, goal_dict=goal_dict)
        total_action = torch.clamp(total_action, -1.0, 1.0)
        return total_action, base_action, res_action
    
    def _get_residual_action(self, obs_dict, goal_dict=None):
        """
        """
        assert not self.nets.training

        nets = self.nets

        inputs = {
            "obs": obs_dict,
            "goal": goal_dict
        }
        for k in self.obs_shapes:
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                # adding time dimension if not present -- this is required as
                # frame stacking is not invoked when sequence length is 1
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
        residual_action = TensorUtils.time_distributed(
            inputs, nets["res_policy"], inputs_as_kwargs=True
        )
        # if multiple frames are provided, use the last timestep's action
        if residual_action.ndim == 3:
            residual_action = residual_action[:, -1]

        return residual_action


    def mix_actions(self, base_action, residual_action, obs_dict=None, goal_dict=None):
        """
        Mix base action and residual action to get the final action.

        Args:
            base_action (torch.Tensor): base action tensor
            residual_action (torch.Tensor): residual action tensor

        Returns:
            action (torch.Tensor): total action tensor
        """
        residual_scale = self._get_residual_scale(obs_dict, goal_dict) if self.learn_scale and obs_dict is not None else self.residual_scale
        action = base_action + residual_action * residual_scale
        return action
    
    def _soft_update_target_network(self, source_network, target_network, tau):
        for target_param, param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def serialize(self):
        """
        Get dictionary of current model parameters.
        We save the residual weights. Base policy is loaded from config/ckpt path.
        """
        nets_state = {
            k: v for k, v in self.nets.state_dict().items()
            if not k.startswith("base_policy.")
        }
        return {
            "nets": nets_state,
            "optimizers": TorchUtils.get_state_dict(self.optimizers),
            "lr_schedulers": TorchUtils.get_state_dict(self.lr_schedulers),
        }

    def deserialize(self, model_dict, load_optimizers=False):
        """
        Load model from a checkpoint.
        """
        if "nets" not in model_dict:
            # backwards compatibility with checkpoints that store raw nets state_dict
            model_dict = {"nets": model_dict}

        nets_state = {
            k: v for k, v in model_dict["nets"].items()
            if not k.startswith("base_policy.")
        }
        self.nets.load_state_dict(nets_state, strict=False)

        if load_optimizers:
            optimizer_state = model_dict.get("optimizers", {})
            if len(optimizer_state) > 0:
                optimizer_subset = {k: self.optimizers[k] for k in self.optimizers if k in optimizer_state}
                if len(optimizer_subset) > 0:
                    TorchUtils.load_state_dict(
                        optimizer_subset,
                        {k: optimizer_state[k] for k in optimizer_subset},
                    )

            scheduler_state = model_dict.get("lr_schedulers", {})
            if len(scheduler_state) > 0:
                scheduler_subset = {
                    k: self.lr_schedulers[k]
                    for k in self.lr_schedulers
                    if k in scheduler_state and self.lr_schedulers[k] is not None and scheduler_state[k] is not None
                }
                if len(scheduler_subset) > 0:
                    TorchUtils.load_state_dict(
                        scheduler_subset,
                        {k: scheduler_state[k] for k in scheduler_subset},
                    )

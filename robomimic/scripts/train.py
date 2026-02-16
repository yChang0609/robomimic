"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes    
"""

import argparse
import json
import numpy as np
import time
import os
import shutil
import psutil
import sys
import socket
import traceback

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.time_utils as TimeUtils
from robomimic.utils.save_utils import SaveManager
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy, ResidualAlgo, ResidualRolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
from robomimic.utils.python_utils import deep_update
from robomimic.utils.replybuffer import ReplayBuffer

def set_seed(config):
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

def _progressive_residual_action_prob(global_steps, warmup_steps=1500, full_residual_steps=10000):
    if global_steps <= warmup_steps:
        return 0.0
    if global_steps >= full_residual_steps:
        return 1.0
    return float(global_steps - warmup_steps) / float(full_residual_steps - warmup_steps)


class ProgressiveResidualRolloutPolicy(ResidualRolloutPolicy):
    """
    Residual rollout policy with stochastic residual gating.
    """
    def __init__(self, policy, obs_normalization_stats=None, action_normalization_stats=None, residual_action_prob=1.0):
        super().__init__(
            policy=policy,
            obs_normalization_stats=obs_normalization_stats,
            action_normalization_stats=action_normalization_stats,
        )
        self.residual_action_prob = float(np.clip(residual_action_prob, 0.0, 1.0))

    def __call__(self, ob, goal=None, batched_ob=False):
        if self.residual_action_prob >= 1.0:
            return super().__call__(ob=ob, goal=goal, batched_ob=batched_ob)

        original_scale = self.policy.residual_scale
        if np.random.rand() >= self.residual_action_prob:
            self.policy.residual_scale = 0.0
        try:
            return super().__call__(ob=ob, goal=goal, batched_ob=batched_ob)
        finally:
            self.policy.residual_scale = original_scale

def print_config(config):
    print("\n============= New Training Run with Config =============")
    print(config)
    print("")

def create_envs_from_dataset(config):
    envs = OrderedDict()
    env_meta_list = []
    shape_meta_list = []

    if isinstance(config.train.data, str):
        with config.values_unlocked():
            config.train.data = [{"path": config.train.data}]
            
    if config.train.data is not None:
        for dataset_cfg in config.train.data:
            dataset_path = os.path.expanduser(dataset_cfg["path"])
            if not os.path.exists(dataset_path):
                raise Exception("Dataset at provided path {} not found!".format(dataset_path))
            env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
            env_meta["lang"] = dataset_cfg.get("lang", "dummy")
            deep_update(env_meta, config.experiment.env_meta_update_dict)
            env_meta_list.append(env_meta)

            shape_meta = FileUtils.get_shape_metadata_from_dataset(
                dataset_config=dataset_cfg,
                action_keys=config.train.action_keys,
                all_obs_keys=config.all_obs_keys
            )
            shape_meta_list.append(shape_meta)

    for env_i in range(len(env_meta_list)):
        dataset_cfg = config.train.data[env_i]
        env_meta = env_meta_list[env_i]
        shape_meta = shape_meta_list[env_i]
        env_names = [env_meta["env_name"]]
        if (env_i == 0) and (config.experiment.additional_envs is not None):
            for name in config.experiment.additional_envs:
                env_names.append(name)
        for env_name in env_names:
            env = create_env(config, env_name, env_meta, shape_meta)
            env_key = os.path.splitext(os.path.basename(dataset_cfg["path"]))[0] if not dataset_cfg.get("key", None) else dataset_cfg["key"]
            envs[env_key] = env
            print(env)
    return envs, env_meta_list, shape_meta_list

def _dataset_cfg_to_env_key(dataset_cfg):
    return os.path.splitext(os.path.basename(dataset_cfg["path"]))[0] if not dataset_cfg.get("key", None) else dataset_cfg["key"]


def _extract_demo_init_state(demo_group):
    if "states" not in demo_group:
        return None

    states = demo_group["states"]
    try:
        first_state = states[0]
    except Exception:
        # Some datasets store states as an HDF5 group of per-field arrays.
        try:
            first_state = {k: states[k][0] for k in states.keys()}
        except Exception:
            return None

    init_state = {"states": first_state}
    if "model_file" in demo_group.attrs:
        init_state["model"] = demo_group.attrs["model_file"]
    if "ep_meta" in demo_group.attrs:
        init_state["ep_meta"] = demo_group.attrs["ep_meta"]
    return init_state


def build_rollout_init_states_from_trainset(config, trainset):
    """
    Build rollout init-state pools per env key from the train dataset(s).
    """
    rollout_init_states = OrderedDict()
    if config.train.data is None:
        return rollout_init_states

    datasets = trainset.datasets if hasattr(trainset, "datasets") else [trainset]
    if len(datasets) != len(config.train.data):
        print(
            "WARNING: dataset count mismatch between trainset ({}) and config.train.data ({}). "
            "Using the first {} entries.".format(
                len(datasets), len(config.train.data), min(len(datasets), len(config.train.data))
            )
        )

    num_entries = min(len(datasets), len(config.train.data))
    for ds_idx in range(num_entries):
        dataset_cfg = config.train.data[ds_idx]
        env_key = _dataset_cfg_to_env_key(dataset_cfg)
        dataset = datasets[ds_idx]
        rollout_init_states[env_key] = []

        if (not hasattr(dataset, "demos")) or (not hasattr(dataset, "hdf5_file_opened")):
            print(
                "WARNING: dataset for env_key '{}' does not expose demos / hdf5 access. "
                "Skipping init-state extraction.".format(env_key)
            )
            continue

        with dataset.hdf5_file_opened() as hdf5_file:
            for demo_id in dataset.demos:
                demo_path = "data/{}".format(demo_id)
                if demo_path not in hdf5_file:
                    print("WARNING: missing '{}' in dataset for env_key '{}', skipping.".format(demo_path, env_key))
                    continue

                init_state = _extract_demo_init_state(hdf5_file[demo_path])
                if init_state is None:
                    print(
                        "WARNING: demo '{}' for env_key '{}' has no valid states[0], skipping.".format(
                            demo_id, env_key
                        )
                    )
                    continue
                rollout_init_states[env_key].append(init_state)

        if len(rollout_init_states[env_key]) == 0:
            print(
                "WARNING: env_key '{}' has no valid init states. Rollouts will fallback to env.reset().".format(
                    env_key
                )
            )
        else:
            print(
                "Loaded {} rollout init states for env_key '{}'.".format(
                    len(rollout_init_states[env_key]), env_key
                )
            )

    return rollout_init_states

# create environment for each env_name
def create_env(config, env_name, env_meta, shape_meta):
    env_kwargs = dict(
        env_meta=env_meta,
        env_name=env_name,
        render=False,
        render_offscreen=config.experiment.render_video,
        use_image_obs=shape_meta["use_images"] or shape_meta["use_depths"],
    )
    env = EnvUtils.create_env_from_metadata(**env_kwargs)
    # handle environment wrappers
    env = EnvUtils.wrap_env_from_config(env, config=config)  # apply environment warpper, if applicable
    return env

def create_model(config, shape_meta, device, resume=None):
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device
    )
    
    if resume is not None:
        # load ckpt dict
        print("*" * 50)
        print("resuming from ckpt at {}".format(resume["latest_model_path"]))
        try:
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=resume["latest_model_path"])
        except Exception as e:
            print("got error: {} when loading from {}".format(e, resume["latest_model_path"]))
            print("trying backup path {}".format(resume["latest_model_backup_path"]))
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=resume["latest_model_backup_path"])
        # load model weights and optimizer state
        model.deserialize(ckpt_dict["model"], load_optimizers=True)
        print("*" * 50)

    # if checkpoint is specified, load in model weights;
    # will not use ckpt_path if resuming training
    ckpt_path = config.experiment.ckpt_path
    if (ckpt_path is not None) and (not resume):
        print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        model.deserialize(ckpt_dict["model"])

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    return model, ckpt_dict["variable_state"] if resume is not None else None

# == Training code == #
def rl_train(config, device, resume=False, auto_remove_exp_dir=False):
    # first set seeds
    set_seed(config)
    torch.set_num_threads(2)
    
    print_config(config)
    log_dir, ckpt_dir, video_dir, time_dir = TrainUtils.get_exp_dir(config, resume=resume, auto_remove_exp_dir=auto_remove_exp_dir)

    # path for latest model and backup (to support @resume functionality)
    latest_model_path = os.path.join(time_dir, "last.pth")
    latest_model_backup_path = os.path.join(time_dir, "last_bak.pth")

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)  

    # extract the metadata and shape metadata across all datasets
    envs, env_meta_list, shape_meta_list = create_envs_from_dataset(config=config)
    env_name = env_meta_list[0]["env_name"]
    if "square" not in env_name.lower():
        raise NotImplementedError(
            "Dense-reward env wrapping is only implemented for square task, got env '{}'".format(env_name)
        )
    for env in envs.values():
        base_env = env.env
        while True:
            if hasattr(base_env, "_init_kwargs"):
                break
            next_env = getattr(base_env, "env", None)
            if (next_env is None) or (next_env is base_env):
                break
            base_env = next_env
        if not hasattr(base_env, "_init_kwargs"):
            raise RuntimeError("Failed to locate base env wrapper with init kwargs for dense reward setup")
        base_env._init_kwargs["reward_shaping"] = True
        rs_env = getattr(base_env, "env", None)
        if (rs_env is None) or (not hasattr(rs_env, "reward_shaping")):
            raise RuntimeError("Failed to enable dense reward for env '{}'".format(base_env))
        rs_env.reward_shaping = True
    
    # TODO [priority: Low] if give mutli dataset need change this rule
    env_meta = env_meta_list[0]
    shape_meta = shape_meta_list[0]
    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # preload failed-demo init states once (for fast per-epoch rollout sampling)
    rollout_init_states = build_rollout_init_states_from_trainset(config=config, trainset=trainset)

    # optional override for rollout episode count when sampling failed init states
    rollout_init_state_sample_size = config.train.get("rollout_init_state_sample_size", None)
    if rollout_init_state_sample_size is None:
        rollout_num_episodes = config.experiment.rollout.n
    else:
        rollout_num_episodes = int(rollout_init_state_sample_size)
        if rollout_num_episodes <= 0:
            raise ValueError(
                "config.train.rollout_init_state_sample_size must be > 0, got {}".format(
                    rollout_init_state_sample_size
                )
            )
        print(
            "Using train.rollout_init_state_sample_size={} as rollout num_episodes "
            "(overrides experiment.rollout.n={}).".format(
                rollout_num_episodes, config.experiment.rollout.n
            )
        )

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.repalybuffer_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # maybe retreve statistics for normalizing actions
    action_normalization_stats = trainset.get_action_normalization_stats()

    # create model saver
    model_saver = SaveManager()

    # create tensorboard logger
    data_logger = DataLogger(log_dir, config, log_tb=config.experiment.logging.log_tb)

    # create replaybuffer for RL training
    replaybuffer = ReplayBuffer(
        capacity=int(config.train.replaybuffer_capacity), 
        obs_keys=config.all_obs_keys) 
    
    # add info to optim_params
    train_num_steps = config.experiment.epoch_every_n_steps
    with config.values_unlocked():
        if "optim_params" in config.algo:
            # add info to optim_params of each net
            for k in config.algo.optim_params:
                config.algo.optim_params[k]["num_train_batches"] = train_num_steps
                config.algo.optim_params[k]["num_epochs"] = config.train.num_epochs
                
    # create mdoel form config 
    model, variable_state = create_model(
        config=config,
        shape_meta=shape_meta,
        device=device,
        resume={
            "latest_model_path":latest_model_path,
            "latest_model_backup_path":latest_model_backup_path
        } if resume else None)
    
    if isinstance(model, ResidualAlgo):
        base_policy_ckpt_path = getattr(getattr(model.algo_config, "base_policy", None), "ckpt_path", None)
        if base_policy_ckpt_path is not None:
            base_policy_ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=base_policy_ckpt_path)

            if config.train.repalybuffer_normalize_obs:
                base_obs_normalization_stats = base_policy_ckpt_dict.get("obs_normalization_stats", None)
                if base_obs_normalization_stats is not None:
                    for obs_key in base_obs_normalization_stats:
                        for stat_key in base_obs_normalization_stats[obs_key]:
                            base_obs_normalization_stats[obs_key][stat_key] = np.array(
                                base_obs_normalization_stats[obs_key][stat_key]
                            )
                    obs_normalization_stats = base_obs_normalization_stats

            base_action_normalization_stats = base_policy_ckpt_dict.get("action_normalization_stats", None)
            if base_action_normalization_stats is not None:
                for action_key in base_action_normalization_stats:
                    for stat_key in base_action_normalization_stats[action_key]:
                        base_action_normalization_stats[action_key][stat_key] = np.array(
                            base_action_normalization_stats[action_key][stat_key]
                        )
                action_normalization_stats = base_action_normalization_stats

    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    # main training loop
    best_return = {k: -np.inf for k in envs}
    best_success_rate = {k: -1. for k in envs}
    last_ckpt_time = time.time()    

    start_epoch = 1 # epoch numbers start at 1
    if resume:
        # load variable state needed for train loop
        start_epoch = variable_state["epoch"] + 1
        best_return = variable_state["best_return"]
        best_success_rate = variable_state["best_success_rate"]
        print("*" * 50)
        print("resuming training from epoch {}".format(start_epoch))
        print("*" * 50)
    
    training_timer = TimeUtils.TrainingTimer()
    for epoch in range(start_epoch, config.train.num_epochs + 1): 
        if isinstance(model, ResidualAlgo):
            residual_action_prob = _progressive_residual_action_prob(len(replaybuffer))
            rollout_model = ProgressiveResidualRolloutPolicy(
                model,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
                residual_action_prob=residual_action_prob,
            )
        else:
            rollout_model = RolloutPolicy(
                model,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

        rollout_log, _ = TrainUtils.rollout_with_stats(
            policy=rollout_model, 
            envs=envs,
            horizon=config.experiment.rollout.horizon,
            num_episodes=rollout_num_episodes,
            use_goals=config.use_goals,
            render=False,
            video_dir=video_dir if config.experiment.render_video else None,
            terminate_on_success=config.experiment.rollout.terminate_on_success,
            epoch=epoch,
            replaybuffer=replaybuffer,
            init_states=rollout_init_states,
        )

        if len(replaybuffer) < config.train.batch_size: 
            continue

        replay_loader = DataLoader(
            replaybuffer,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=replay_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
        )
        model.on_epoch_end(epoch)

        print("Train Epoch {}".format(epoch))
        print(json.dumps(step_log, sort_keys=True, indent=4))
        # TODO: Log the following metrics to monitor model training performance:
        # - rollout success rate
        # - rollout horizon_mean
        # - replaybuffer length
        # - progressive residual action prob
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Train/{}".format(k), v, epoch)
        for env_name, env_rollout_log in rollout_log.items():
            print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, env_rollout_log["time"]))
            print("Env: {}".format(env_name))
            print(json.dumps(env_rollout_log, sort_keys=True, indent=4))
        
        should_save_ckpt = False
        epoch_ckpt_name = f"model_epoch_{epoch}"
        for env_name in envs:
            current_return = rollout_log[env_name]["Return"]
            if current_return > best_return[env_name]:
                best_return[env_name] = current_return
                if config.experiment.save.on_best_rollout_return:
                    should_save_ckpt = True
                    epoch_ckpt_name += "_best_return"

        if config.experiment.save.every_n_epochs and (epoch % config.experiment.save.every_n_epochs == 0):
            should_save_ckpt = True

        variable_state = dict(
            epoch=epoch,
            best_return=best_return,
            best_success_rate=best_success_rate,
            best_valid_loss=None,
        )
        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                variable_state=variable_state,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=None, 
                action_normalization_stats=None,
                saver=model_saver,
                is_temp=False
            )
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1024 / 1024)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
    print(f"RL Training Complete. Total Time: {training_timer.get_elapsed_time()}")

    for env in envs.values():
        base_env = env
        while True:
            next_env = getattr(base_env, "env", None)
            if (next_env is None) or (next_env is base_env):
                break
            base_env = next_env
        close_fn = getattr(base_env, "close", None)
        if callable(close_fn):
            close_fn()

    model_saver.stop()
    data_logger.close()

def il_train(config, device, resume=False, auto_remove_exp_dir=False):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    set_seed(config)
    torch.set_num_threads(2)

    print_config(config)
    log_dir, ckpt_dir, video_dir, time_dir = TrainUtils.get_exp_dir(config, resume=resume, auto_remove_exp_dir=auto_remove_exp_dir)

    # path for latest model and backup (to support @resume functionality)
    latest_model_path = os.path.join(time_dir, "last.pth")
    latest_model_backup_path = os.path.join(time_dir, "last_bak.pth")

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # extract the metadata and shape metadata across all datasets
    env_meta_list = []
    shape_meta_list = []
    if isinstance(config.train.data, str):
        # if only a single dataset is provided, convert to list
        with config.values_unlocked():
            config.train.data = [{"path": config.train.data}]
    for dataset_cfg in config.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        # load basic metadata from training file
        print("\n============= Loaded Environment Metadata =============")
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)

        # populate language instruction for env in env_meta
        env_meta["lang"] = dataset_cfg.get("lang", "dummy")

        # update env meta if applicable
        deep_update(env_meta, config.experiment.env_meta_update_dict)
        env_meta_list.append(env_meta)

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_config=dataset_cfg,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            verbose=True
        )
        shape_meta_list.append(shape_meta)

    if config.experiment.env is not None:
        # if an environment name is specified, just use this env using the first dataset's metadata
        # and ignore envs from all datasets
        env_meta = env_meta_list[0].copy()
        env_meta["env_name"] = config.experiment.env
        env_meta_list = [env_meta]
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        for env_i in range(len(env_meta_list)):
            # check if this env should be evaluated
            dataset_cfg = config.train.data[env_i]
            do_eval = dataset_cfg.get("eval", True)
            if not do_eval:
                continue

            env_meta = env_meta_list[env_i]
            shape_meta = shape_meta_list[env_i]

            env_names = [env_meta["env_name"]]
            if (env_i == 0) and (config.experiment.additional_envs is not None):
                # if additional environments are specified, add them to the list
                # all additional environments use env_meta from the first dataset
                for name in config.experiment.additional_envs:
                    env_names.append(name)


            for env_name in env_names:
                env = create_env(config, env_name, env_meta, shape_meta)
                env_key = os.path.splitext(os.path.basename(dataset_cfg["path"]))[0] if not dataset_cfg.get("key", None) else dataset_cfg["key"]
                envs[env_key] = env
                print(env)

    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # maybe retreve statistics for normalizing actions
    action_normalization_stats = trainset.get_action_normalization_stats()

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True
        )
    else:
        valid_loader = None

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    # add info to optim_params
    with config.values_unlocked():
        if "optim_params" in config.algo:
            # add info to optim_params of each net
            for k in config.algo.optim_params:
                config.algo.optim_params[k]["num_train_batches"] = len(trainset) if train_num_steps is None else train_num_steps
                config.algo.optim_params[k]["num_epochs"] = config.train.num_epochs
        # handling for "hbc" and "iris" algorithms
        if config.algo_name == "hbc":
            for sub_algo in ["planner", "actor"]:
                # add info to optim_params of each net
                for k in config.algo[sub_algo].optim_params:
                    config.algo[sub_algo].optim_params[k]["num_train_batches"] = len(trainset) if train_num_steps is None else train_num_steps
                    config.algo[sub_algo].optim_params[k]["num_epochs"] = config.train.num_epochs
        if config.algo_name == "iris":
            for sub_algo in ["planner", "value"]:
                # add info to optim_params of each net
                for k in config.algo["value_planner"][sub_algo].optim_params:
                    config.algo["value_planner"][sub_algo].optim_params[k]["num_train_batches"] = len(trainset) if train_num_steps is None else train_num_steps
                    config.algo["value_planner"][sub_algo].optim_params[k]["num_epochs"] = config.train.num_epochs
    # Model saver
    model_saver = SaveManager()

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta_list[0]["all_shapes"],
        ac_dim=shape_meta_list[0]["ac_dim"],
        device=device
    )

    if resume:
        # load ckpt dict
        print("*" * 50)
        print("resuming from ckpt at {}".format(latest_model_path))
        try:
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=latest_model_path)
        except Exception as e:
            print("got error: {} when loading from {}".format(e, latest_model_path))
            print("trying backup path {}".format(latest_model_backup_path))
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=latest_model_backup_path)
        # load model weights and optimizer state
        model.deserialize(ckpt_dict["model"], load_optimizers=True)
        print("*" * 50)
    
    # if checkpoint is specified, load in model weights;
    # will not use ckpt_path if resuming training
    ckpt_path = config.experiment.ckpt_path
    if (ckpt_path is not None) and (not resume):
        print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        model.deserialize(ckpt_dict["model"])

    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    # main training loop
    best_valid_loss = None
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1. for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    start_epoch = 1 # epoch numbers start at 1
    if resume:
        # load variable state needed for train loop
        variable_state = ckpt_dict["variable_state"]
        start_epoch = variable_state["epoch"] + 1 # start at next epoch, since this recorded the last epoch of training completed
        best_valid_loss = variable_state["best_valid_loss"]
        best_return = variable_state["best_return"]
        best_success_rate = variable_state["best_success_rate"]
        print("*" * 50)
        print("resuming training from epoch {}".format(start_epoch))
        print("*" * 50)

    training_timer = TimeUtils.TrainingTimer()
    for epoch in range(start_epoch, config.train.num_epochs + 1):
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
        )
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and \
                (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
            epoch_check = (config.experiment.save.every_n_epochs is not None) and \
                (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
            epoch_list_check = (epoch in config.experiment.save.epochs)
            should_save_ckpt = (time_check or epoch_check or epoch_list_check)
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print("Train Epoch {}".format(epoch))
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Train/{}".format(k), v, epoch)

        # Evaluate the model on validation set
        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model,
                    data_loader=valid_loader,
                    epoch=epoch,
                    validate=True,
                    num_steps=valid_num_steps,
                    obs_normalization_stats=obs_normalization_stats,
                )
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Valid/{}".format(k), v, epoch)

            print("Validation Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Evaluate the model by by running rollouts

        # do rollouts at fixed rate or if it's time to save a new ckpt
        video_paths = None
        rollout_check = (epoch % config.experiment.rollout.rate == 0) or (should_save_ckpt and ckpt_reason == "time")
        if config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:
            # wrap model as a RolloutPolicy to prepare for rollouts
            rollout_model = RolloutPolicy(
                model,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

            num_episodes = config.experiment.rollout.n
            all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=num_episodes,
                render=False,
                video_dir=video_dir if config.experiment.render_video else None,
                epoch=epoch,
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )

            # summarize results from rollouts to tensorboard and terminal
            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    if k.startswith("Time_"):
                        data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                    else:
                        data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

                print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
                print('Env: {}'.format(env_name))
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            # checkpoint and video saving logic
            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (config.experiment.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]

        # get variable state for saving model
        variable_state = dict(
            epoch=epoch,
            best_valid_loss=best_valid_loss,
            best_return=best_return,
            best_success_rate=best_success_rate,
        )

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:    
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta_list[0] if len(env_meta_list)==1 else env_meta_list,
                shape_meta=shape_meta_list[0] if len(shape_meta_list)==1 else shape_meta_list,
                variable_state=variable_state,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
                saver=model_saver,
                is_temp=False
            )
        # always save latest model for resume functionality
        print("\nsaving latest model at {}...\n".format(latest_model_path))

        TrainUtils.save_model(
            model=model,
            config=config,
            env_meta=env_meta_list[0] if len(env_meta_list)==1 else env_meta_list,
            shape_meta=shape_meta_list[0] if len(shape_meta_list)==1 else shape_meta_list,
            variable_state=variable_state,
            ckpt_path=latest_model_path,
            obs_normalization_stats=obs_normalization_stats,
            action_normalization_stats=action_normalization_stats,
            saver=model_saver,
            is_temp=True
        )

        # with timer("copyfile"):
        #     # keep a backup model in case last.pth is malformed (e.g. job died last time during saving)
        #     shutil.copyfile(latest_model_path, latest_model_backup_path)
        #     print("\nsaved backup of latest model at {}\n".format(latest_model_backup_path))

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # terminate logging
    print(f"Training time({config.train.num_epochs + 1 - start_epoch} epoch): {training_timer.get_elapsed_time()}")
    model_saver.stop()
    data_logger.close()


def main(args):

    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = [{"path": args.dataset}]

    if args.name is not None:
        config.experiment.name = args.name

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()
    
    if "training_mode" not in config.train:
        with config.values_unlocked():
            config.train.training_mode = "IL"

    train_func = il_train if config.train.training_mode == "IL" else rl_train        
    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train_func(config, device=device, resume=args.resume, auto_remove_exp_dir=args.auto_remove_exp)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    # resume training from latest checkpoint
    parser.add_argument(
        "--resume",
        action='store_true',
        help="set this flag to resume training from latest checkpoint",
    )

    # 
    parser.add_argument(
        "--auto-remove-exp",
        action='store_true',
        help="force delete the experiment folder if it exists"
    )

    args = parser.parse_args()
    main(args)

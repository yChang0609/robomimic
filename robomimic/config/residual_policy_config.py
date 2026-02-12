"""
Config for Residual Policy algorithm.
"""

from robomimic.config.base_config import BaseConfig

class ResidualPolicyConfig(BaseConfig):
    ALGO_NAME = "residual_policy"

    def train_config(self):
        """
        Setting up training parameters for Residual Policy.

        - don't need "next_obs" from hdf5 - so save on storage and compute by disabling it
        - set compatible data loading parameters
        """
        super(ResidualPolicyConfig, self).train_config()
        
        # disable next_obs loading from hdf5
        self.train.hdf5_load_next_obs = False
    
    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        
        # Training mode
        self.train.training_mode = "RL"

        # optimization parameters
        # self.algo.optim_params.res_policy.optimizer_type = "adamw"
        # self.algo.optim_params.res_policy.learning_rate.initial = 1e-4      # policy learning rate
        # self.algo.optim_params.res_policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        # self.algo.optim_params.res_policy.learning_rate.step_every_batch = True
        # self.algo.optim_params.res_policy.learning_rate.scheduler_type = "cosine"
        # self.algo.optim_params.res_policy.learning_rate.num_cycles = 0.5 # number of cosine cycles (used by "cosine" scheduler)
        # self.algo.optim_params.res_policy.learning_rate.warmup_steps = 500 # number of warmup steps (used by "cosine" scheduler)
        # self.algo.optim_params.res_policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs (used by "linear" and "multistep" schedulers)
        # self.algo.optim_params.res_policy.learning_rate.do_not_lock_keys()
        # self.algo.optim_params.res_policy.regularization.L2 = 1e-6          # L2 regularization strength

        # EMA parameters
        self.algo.ema.enabled = True
        self.algo.ema.power = 0.75
        
        # Base policy
        self.algo.base_policy.algo_name = "diffusion_policy"
        self.algo.base_policy.ckpt_path = ""
        self.algo.base_policy.frozen = True
        
        # Residual parameters
        self.algo.residual.mode = "additive"
        self.algo.residual.scale_factor = 0.1
        self.algo.residual.learn_scale = False
        self.algo.residual.temporal_alignment = "sequence"

        # Residual network parameters
        self.algo.residual.layer_dims = [256,512]

        # RL parameters
        self.algo.critic.enabled = True
        self.algo.critic.ensemble_size = 2
        self.algo.critic.layer_dims = [256, 256]
        self.algo.critic.learning_rate = 1e-4
        
        self.algo.rl.gamma = 0.99
        self.algo.rl.tau = 0.005
        self.algo.rl.alpha = 0.2

        
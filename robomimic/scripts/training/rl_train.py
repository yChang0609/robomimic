import json
import os
import psutil
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.time_utils as TimeUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.algo import (
    ProgressiveResidualRolloutPolicy,
    ResidualAlgo,
    RolloutPolicy,
    progressive_residual_action_prob,
)
from robomimic.utils.log_utils import DataLogger, PrintLogger
from robomimic.utils.replybuffer import ReplayBuffer
from robomimic.utils.save_utils import SaveManager

from robomimic.scripts.training.utils import (
    build_rollout_init_states_from_trainset,
    create_envs_from_dataset,
    create_model,
    print_config,
    set_seed,
)

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

            if config.train.get("use_base_action_normalization_stats", False):
                base_action_normalization_stats = base_policy_ckpt_dict.get("action_normalization_stats", None)
                if base_action_normalization_stats is None:
                    print(
                        "\nWARNING: train.use_base_action_normalization_stats is True, but base checkpoint has no action_normalization_stats. "
                        "Falling back to dataset action stats."
                    )
                else:
                    # convert stats from checkpoint to numpy arrays
                    for action_key in base_action_normalization_stats:
                        for stat_key in base_action_normalization_stats[action_key]:
                            base_action_normalization_stats[action_key][stat_key] = np.array(
                                base_action_normalization_stats[action_key][stat_key]
                            )

                    # print out the max difference in offset and scale for each action key 
                    # between the base policy stats and the dataset stats (if available) 
                    # before overriding with the base policy stats
                    if action_normalization_stats is not None:
                        for action_key in base_action_normalization_stats:
                            if action_key not in action_normalization_stats:
                                continue
                            base_stats = base_action_normalization_stats[action_key]
                            dataset_stats = action_normalization_stats[action_key]
                            offset_diff = float(np.max(np.abs(base_stats["offset"] - dataset_stats["offset"])))
                            scale_diff = float(np.max(np.abs(base_stats["scale"] - dataset_stats["scale"])))
                            print(
                                f"Overriding action normalization stats from base policy for key='{action_key}': "
                                f"max|offset diff|={offset_diff:.6f}, max|scale diff|={scale_diff:.6f}"
                            )
                    action_normalization_stats = base_action_normalization_stats

    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    # main training loop
    best_return = {k: -np.inf for k in envs}
    best_success_rate = {k: -1. for k in envs}

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
            residual_action_prob = progressive_residual_action_prob(len(replaybuffer))
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
        rollout_success_rate = np.mean([env_rollout_log["Success_Rate"] for env_rollout_log in rollout_log.values()])
        rollout_horizon_mean = np.mean([env_rollout_log["Horizon"] for env_rollout_log in rollout_log.values()])
        data_logger.record("Rollout/Success_Rate", rollout_success_rate, epoch)
        data_logger.record("Rollout/Horizon_Mean", rollout_horizon_mean, epoch)
        data_logger.record("ReplayBuffer/Length", len(replaybuffer), epoch)
        if isinstance(model, ResidualAlgo):
            data_logger.record("Train/Progressive_Residual_Action_Prob", residual_action_prob, epoch)

        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Train/{}".format(k), v, epoch)
        for env_name, env_rollout_log in rollout_log.items():
            print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, env_rollout_log["time"]))
            print("Env: {}".format(env_name))
            print(json.dumps(env_rollout_log, sort_keys=True, indent=4))
        
        updated_stats = TrainUtils.should_save_from_rollout_logs(
            all_rollout_logs=rollout_log,
            best_return=best_return,
            best_success_rate=best_success_rate,
            epoch_ckpt_name=f"model_epoch_{epoch}",
            save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
            save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
        )
        best_return = updated_stats["best_return"]
        best_success_rate = updated_stats["best_success_rate"]
        epoch_ckpt_name = updated_stats["epoch_ckpt_name"]

        should_save_ckpt = False
        if config.experiment.save.enabled:
            should_save_ckpt = updated_stats["should_save_ckpt"]
        if config.experiment.save.enabled and config.experiment.save.every_n_epochs and (epoch % config.experiment.save.every_n_epochs == 0):
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
                obs_normalization_stats=obs_normalization_stats, 
                action_normalization_stats=action_normalization_stats,
                saver=model_saver,
                is_temp=False
            )
        print("\nsaving latest model at {}...\n".format(latest_model_path))
        TrainUtils.save_model(
            model=model,
            config=config,
            env_meta=env_meta,
            shape_meta=shape_meta,
            variable_state=variable_state,
            ckpt_path=latest_model_path,
            obs_normalization_stats=obs_normalization_stats,
            action_normalization_stats=action_normalization_stats,
            saver=model_saver,
            is_temp=True
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

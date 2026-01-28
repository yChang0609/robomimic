import os
import sys
import argparse

import json
import imageio

import numpy as np
import torch

from collections import deque

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils

from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from robomimic.utils.log_utils import PrintLogger

def env_rollout(solvers, env, horizon, render=False, video_writer=None, video_skip=5, camera_names=None):
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    for key in solvers.keys():
        assert isinstance(solvers[key][0], RolloutPolicy)
    
    num_subtasks = len(solvers)
    current_subtask_id = 0
    for key in solvers.keys():
        solvers[key][0].start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)
    
    try:
        obs_horizon = solvers["task_0"][0].policy.algo_config.horizon.observation_horizon
    except AttributeError:
        obs_horizon = 1
    
    obs_deque = deque([obs] * obs_horizon, maxlen=obs_horizon)
    results = {}
    video_count = 0
    total_reward = 0.
    success = False
    try:
        for step_i in range(horizon):
            current_policy = solvers[f"task_{current_subtask_id}"][0]
            # current_policy = solvers[f"task_0"][0]# for debug
            act = current_policy(ob=obs)
            obs, r, done, info = env.step(act)
            total_reward += r

            subtask_success = check_subtask_finished(env, current_subtask_id)
            if subtask_success:
                print(f"Subtask {current_subtask_id} finished at step {step_i}")
                if current_subtask_id < num_subtasks - 1:
                    current_subtask_id += 1
                else:
                    success = env.is_success()["task"]
                    if success: break
            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1
            if done: break
    except Exception as e:
        print(f"WARNING: rollout exception {e}")

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))
    return stats, None

def check_subtask_finished(env, subtask_id):
    base_env = env.env.env
    if subtask_id == 0:
        nut_idx = base_env.nut_id
        active_nut = base_env.nuts[nut_idx]
        nut_name = active_nut.name

        is_grasped = base_env._check_grasp(
            gripper=base_env.robots[0].gripper,
            object_geoms=base_env.nuts[nut_idx]
        )

        nut_body_id = base_env.obj_body_id[nut_name]
        nut_pos = base_env.sim.data.body_xpos[nut_body_id]

        table_height = base_env.table_offset[2]

        is_lifted = nut_pos[2] > (table_height + 0.1)

        return is_grasped and is_lifted
    
    elif subtask_id == 1:
        return env.is_success()["task"]
    
    return False

def failed_case_rollout():
    pass

def inference(args):

    # Inferences from failed cases or new environments
    assert (args.env is not None) or (args.failed_hdf5 is not None), "Need give failed case or env"

    if args.output_to_txt_path is not None:
        # log stdout and stderr to a text file
        log_file_name = 'test_log.txt' if args.env else 'failed_test_log.txt'
        logger = PrintLogger(os.path.join(args.output_to_txt_path, log_file_name))
        sys.stdout = logger
        sys.stderr = logger

    write_video = (args.video_path is not None)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True) 

    # Load models for each
    task_solver = dict()
    for i, agent_path in enumerate(args.agents):
        print(f"Loading Subtask {i} Solver from: {agent_path}")
        task_solver[f"task_{i}"] = FileUtils.policy_from_checkpoint(ckpt_path=agent_path, device=device, verbose=True) # policy, ckpt_dict
        
    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20) 

    rollout_stats = []
    
    # Env testing
    if args.env:
        # create environment from saved checkpoint
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=task_solver[f"task_0"][1], 
            env_name=args.env, 
            render=args.render, 
            render_offscreen=(args.video_path is not None), 
            verbose=True,
        )
        for i in LogUtils.custom_tqdm(range(rollout_num_episodes)):
            stats, traj = env_rollout(
                solvers=task_solver, 
                env=env, 
                horizon=rollout_horizon, 
                render=args.render, 
                video_writer=video_writer, 
                video_skip=args.video_skip, 
                camera_names=args.camera_names,
            )
            rollout_stats.append(stats)

        rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
        avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
        avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
        print("Average Rollout Stats")
        print(json.dumps(avg_rollout_stats, indent=4))
        env.env.env.close()
    else:
        assert args.failed_hdf5 is not None, "If not selcet env plase give the failed hdf5 file."
        

    if write_video:
        video_writer.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained policy on failed MimicGen environments.")

    parser.add_argument("--agents", type=str, nargs='+', required=True, help="Path to your trained .pth checkpoint")
    parser.add_argument("--failed_hdf5", type=str, default=None, help="Path to the failed_demo.hdf5 file")
    parser.add_argument("--env", type=str, default=None, help="Testing env name")
    parser.add_argument("--n_rollouts", type=int, default=None, help="Number of failed cases to test (default: all)")
    parser.add_argument("--horizon", type=int, required=True, help="Maximum horizon")
    parser.add_argument("--render", action='store_true', help="On-screen rendering")
    parser.add_argument("--video_path", type=str, default=None, help="Path to save output video")
    parser.add_argument("--video_skip", type=int, default=5, help="Render every n steps")
    parser.add_argument("--camera_names", type=str, nargs='+', default=["agentview"], help="Cameras for rendering")
    parser.add_argument("--output_to_txt_path", type=str, default=None, help="Output the log into txt")
    parser.add_argument("--seed",type=int,default=None,help="(optional) set seed for rollouts")

    args = parser.parse_args()
    inference(args)
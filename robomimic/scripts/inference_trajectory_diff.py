import os
import sys
import argparse
import h5py
import json
import imageio
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils
from robomimic.algo import ResidualRolloutPolicy
from robomimic.utils.log_utils import PrintLogger
from robomimic.scripts.training.utils import env_close


ROLLOUT_MODES = (
    ("with_residual", True),
    ("without_residual", False),
)
EEF_OBS_KEYS = ("robot0_eef_pos", "eef_pos")


def extract_latest_eef_pos(obs=None, env=None):
    """
    Extract the most recent end-effector position from observation dicts. This
    handles both unstacked observations and frame-stacked observations.
    """
    candidates = []
    if obs is not None:
        candidates.append(obs)
    if env is not None:
        if hasattr(env, "unwrapped"):
            candidates.append(env.unwrapped.get_observation())
        else:
            candidates.append(env.get_observation())

    for candidate_obs in candidates:
        for key in EEF_OBS_KEYS:
            if key in candidate_obs:
                value = np.asarray(candidate_obs[key], dtype=np.float32)
                if value.ndim == 1:
                    if value.shape[0] >= 3:
                        return value[:3].copy()
                elif value.shape[-1] >= 3:
                    return value.reshape(-1, value.shape[-1])[-1, :3].copy()

    raise KeyError(
        "Could not find end-effector position in observation. "
        f"Tried keys: {EEF_OBS_KEYS}"
    )


def clone_initial_state(initial_state):
    if isinstance(initial_state, dict):
        return {
            key: (np.array(value, copy=True) if isinstance(value, np.ndarray) else value)
            for key, value in initial_state.items()
        }
    return {"states": np.array(initial_state, copy=True)}


def prepare_fixed_initial_state(env, initial_state=None):
    """
    Always reset the environment to a fixed state so both rollout variants
    start from the exact same simulator configuration.
    """
    if initial_state is None:
        env.reset()
        initial_state = env.get_state()

    state_dict = clone_initial_state(initial_state)
    obs = env.reset_to(state_dict)
    return obs, state_dict


def action_to_env_space(policy, action):
    action = np.asarray(action, dtype=np.float32)
    if policy.action_normalization_stats is None:
        return action.copy()
    return np.asarray(policy.unnormalization(action), dtype=np.float32)


def stack_action_buffer(items, action_dim=None):
    if len(items) == 0:
        if action_dim is None:
            return np.zeros((0, 0), dtype=np.float32)
        return np.zeros((0, action_dim), dtype=np.float32)
    return np.asarray(items, dtype=np.float32).reshape(len(items), -1)


def build_absolute_trajectory(initial_position, actions):
    initial_position = np.asarray(initial_position, dtype=np.float32).reshape(3)
    if actions.shape[0] == 0:
        return initial_position[None]
    xyz_deltas = actions[:, :3]
    return np.concatenate(
        (
            initial_position[None],
            initial_position[None] + np.cumsum(xyz_deltas, axis=0),
        ),
        axis=0,
    ).astype(np.float32)


def build_relative_trajectory(actions):
    if actions.shape[0] == 0:
        return np.zeros((1, 3), dtype=np.float32)
    xyz_deltas = actions[:, :3]
    return np.concatenate(
        (
            np.zeros((1, 3), dtype=np.float32),
            np.cumsum(xyz_deltas, axis=0).astype(np.float32),
        ),
        axis=0,
    )


def capture_render_frame_stack(env, camera_names, height, width):
    frame_list = [
        env.render(mode="rgb_array", height=height, width=width, camera_name=cam)
        for cam in camera_names
    ]
    return np.stack(frame_list, axis=0).astype(np.uint8)


def save_numeric_dataset(group, name, array):
    array = np.asarray(array)
    if array.shape == ():
        group.create_dataset(name, data=array)
    else:
        group.create_dataset(name, data=array, compression="gzip", chunks=True)


def write_case_to_hdf5(
    case_group,
    stats_by_mode,
    rollout_by_mode,
    initial_state_dict,
    rollout_meta_by_mode=None,
):
    if "states" in initial_state_dict:
        save_numeric_dataset(case_group, "initial_state", initial_state_dict["states"])
    if "model" in initial_state_dict:
        case_group.attrs["initial_state_has_model_xml"] = True
    if "ep_meta" in initial_state_dict:
        case_group.attrs["initial_state_has_ep_meta"] = True

    for mode_name, stats in stats_by_mode.items():
        mode_group = case_group.create_group(mode_name)
        for stat_key, stat_value in stats.items():
            mode_group.attrs[stat_key] = stat_value
        if rollout_meta_by_mode is not None:
            for meta_key, meta_value in rollout_meta_by_mode.get(mode_name, {}).items():
                mode_group.attrs[meta_key] = meta_value
        for dataset_name, dataset_value in rollout_by_mode[mode_name].items():
            save_numeric_dataset(mode_group, dataset_name, dataset_value)


def summarize_mode_results(result_list):
    if len(result_list) == 0:
        return {
            "Total_Cases": 0,
            "Success_Count": 0,
            "Avg_Success_Rate": 0.0,
            "Avg_Return": 0.0,
            "Avg_Horizon": 0.0,
        }

    final_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(result_list)
    return {
        "Total_Cases": len(result_list),
        "Success_Count": int(np.sum(final_stats["Success_Rate"])),
        "Avg_Success_Rate": float(np.mean(final_stats["Success_Rate"])),
        "Avg_Return": float(np.mean(final_stats["Return"])),
        "Avg_Horizon": float(np.mean(final_stats["Horizon"])),
    }


def paired_summary(with_residual_results, without_residual_results):
    if len(with_residual_results) == 0 or len(without_residual_results) == 0:
        return {
            "Mean_Return_Delta": 0.0,
            "Mean_Success_Delta": 0.0,
            "Mean_Horizon_Delta": 0.0,
            "With_Residual_Better_Return_Count": 0,
            "With_Residual_Better_Success_Count": 0,
        }

    with_returns = np.array([result["Return"] for result in with_residual_results], dtype=np.float32)
    without_returns = np.array([result["Return"] for result in without_residual_results], dtype=np.float32)
    with_success = np.array([result["Success_Rate"] for result in with_residual_results], dtype=np.float32)
    without_success = np.array([result["Success_Rate"] for result in without_residual_results], dtype=np.float32)
    with_horizon = np.array([result["Horizon"] for result in with_residual_results], dtype=np.float32)
    without_horizon = np.array([result["Horizon"] for result in without_residual_results], dtype=np.float32)

    return {
        "Mean_Return_Delta": float(np.mean(with_returns - without_returns)),
        "Mean_Success_Delta": float(np.mean(with_success - without_success)),
        "Mean_Horizon_Delta": float(np.mean(with_horizon - without_horizon)),
        "With_Residual_Better_Return_Count": int(np.sum(with_returns > without_returns)),
        "With_Residual_Better_Success_Count": int(np.sum(with_success > without_success)),
    }


def check_subtask_finished(env, subtask_id):
    base_env = env.env.env
    if subtask_id == 0:
        nut_idx = base_env.nut_id
        is_grasped = base_env._check_grasp(
            base_env.robots[0].gripper, base_env.nuts[nut_idx]
        )
        nut_pos = base_env.sim.data.body_xpos[
            base_env.obj_body_id[base_env.nuts[nut_idx].name]
        ]
        is_lifted = nut_pos[2] > (base_env.table_offset[2] + 0.1)
        return is_grasped and is_lifted
    if subtask_id == 1:
        return env.is_success()["task"]
    return False


def run_rollout(
    solvers,
    env,
    horizon,
    initial_state=None,
    render=False,
    video_writer=None,
    video_skip=5,
    camera_names=None,
    use_residual=True,
    save_render_frames=False,
    frame_height=512,
    frame_width=512,
):
    action_dim = env.action_dimension
    buffer_dict = {
        "final_actions": [],
        "base_actions": [],
        "residual_actions_raw": [],
        "residual_actions_env": [],
        "eef_pos": [],
        "rewards": [],
        "render_frames": [],
    }

    num_subtasks = len(solvers)
    is_multi_task = num_subtasks > 1
    original_residual_scales = {}

    for key in solvers.keys():
        current_solver = solvers[key][0]
        current_solver.start_episode()
        original_residual_scales[key] = current_solver.policy.residual_scale
        current_solver.policy.residual_scale = (
            original_residual_scales[key] if use_residual else 0.0
        )

    obs, initial_state_dict = prepare_fixed_initial_state(env, initial_state=initial_state)
    initial_eef_pos = extract_latest_eef_pos(obs=obs, env=env)
    buffer_dict["eef_pos"].append(initial_eef_pos)
    if save_render_frames:
        buffer_dict["render_frames"].append(
            capture_render_frame_stack(
                env=env,
                camera_names=camera_names,
                height=frame_height,
                width=frame_width,
            )
        )

    video_count = 0
    total_reward = 0.0
    success = False
    current_subtask_id = 0
    steps_taken = 0

    if not is_multi_task:
        single_policy = solvers["task_0"][0]

    try:
        for step_i in range(horizon):
            current_policy = (
                solvers[f"task_{current_subtask_id}"][0]
                if is_multi_task
                else single_policy
            )
            act = np.asarray(current_policy(ob=obs), dtype=np.float32)
            base_action = action_to_env_space(current_policy, current_policy.last_base_action)
            residual_action_raw = np.asarray(
                current_policy.last_residual_action, dtype=np.float32
            )

            obs, r, done, info = env.step(act)
            steps_taken = step_i + 1
            total_reward += r

            buffer_dict["final_actions"].append(act)
            buffer_dict["base_actions"].append(base_action)
            buffer_dict["residual_actions_raw"].append(residual_action_raw)
            buffer_dict["residual_actions_env"].append(act - base_action)
            buffer_dict["eef_pos"].append(extract_latest_eef_pos(obs=obs, env=env))
            buffer_dict["rewards"].append(r)
            if save_render_frames:
                buffer_dict["render_frames"].append(
                    capture_render_frame_stack(
                        env=env,
                        camera_names=camera_names,
                        height=frame_height,
                        width=frame_width,
                    )
                )

            if is_multi_task:
                if check_subtask_finished(env, current_subtask_id):
                    if current_subtask_id < num_subtasks - 1:
                        current_subtask_id += 1
                    else:
                        success = env.is_success()["task"]
                        if success:
                            break
            else:
                success = env.is_success()["task"]
                if success:
                    break

            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = [
                        env.render(
                            mode="rgb_array", height=512, width=512, camera_name=cam
                        )
                        for cam in camera_names
                    ]
                    video_img = np.concatenate(video_img, axis=1)
                    video_writer.append_data(video_img)
                video_count += 1

            if done:
                break
    except Exception as error:
        print(f"WARNING: rollout exception {error}")
    finally:
        for key in solvers.keys():
            solvers[key][0].policy.residual_scale = original_residual_scales[key]

    final_actions = stack_action_buffer(buffer_dict["final_actions"], action_dim)
    base_actions = stack_action_buffer(buffer_dict["base_actions"], action_dim)
    residual_actions_raw = stack_action_buffer(buffer_dict["residual_actions_raw"])
    residual_actions_env = stack_action_buffer(
        buffer_dict["residual_actions_env"], action_dim
    )
    eef_traj = np.asarray(buffer_dict["eef_pos"], dtype=np.float32).reshape(-1, 3)
    rewards = np.asarray(buffer_dict["rewards"], dtype=np.float32)
    render_frames = np.asarray(buffer_dict["render_frames"], dtype=np.uint8)

    rollout_data = {
        "initial_eef_pos": initial_eef_pos.astype(np.float32),
        "actual_eef_traj": eef_traj,
        "final_actions": final_actions,
        "base_actions": base_actions,
        "residual_actions_raw": residual_actions_raw,
        "residual_actions_env": residual_actions_env,
        "rewards": rewards,
        "final_action_traj": build_absolute_trajectory(initial_eef_pos, final_actions),
        "base_action_traj": build_absolute_trajectory(initial_eef_pos, base_actions),
        "residual_delta_traj": build_relative_trajectory(residual_actions_env),
    }
    rollout_metadata = {}
    if save_render_frames:
        rollout_data["render_frames"] = render_frames
        rollout_metadata = {
            "render_frames_camera_names": json.dumps(list(camera_names)),
            "render_frames_height": int(frame_height),
            "render_frames_width": int(frame_width),
            "render_frames_num_cameras": int(len(camera_names)),
        }

    return {
        "Return": total_reward,
        "Horizon": steps_taken,
        "Success_Rate": float(success),
    }, rollout_data, initial_state_dict, rollout_metadata


def get_case_initial_state(env, failed_file, key):
    if failed_file is not None:
        return {"states": np.array(failed_file[f"data/{key}/states"][0], copy=True)}

    env.reset()
    return env.get_state()


def build_video_path(video_path, mode_name):
    base, ext = os.path.splitext(video_path)
    ext = ext if ext else ".mp4"
    return f"{base}_{mode_name}{ext}"


def inference(args):
    if args.save_frames_to_hdf5:
        assert args.trajectory_output_dir is not None, (
            "--save_frames_to_hdf5 requires --trajectory_output_dir"
        )

    assert (args.env is not None) or (args.failed_hdf5 is not None), (
        "Need give failed case or env"
    )

    json_path = None
    if args.output_to_txt_path is not None:
        os.makedirs(args.output_to_txt_path, exist_ok=True)
        log_name = "env_test" if args.env else "failed_case_test"
        json_path = os.path.join(args.output_to_txt_path, log_name + ".json")
        log_path = os.path.join(args.output_to_txt_path, log_name + ".txt")
        sys.stdout = sys.stderr = PrintLogger(log_path)

    trajectory_hdf5_path = None
    trajectory_summary_path = None
    if args.trajectory_output_dir is not None:
        os.makedirs(args.trajectory_output_dir, exist_ok=True)
        trajectory_hdf5_path = os.path.join(
            args.trajectory_output_dir, "paired_rollout_trajectories.hdf5"
        )
        trajectory_summary_path = os.path.join(
            args.trajectory_output_dir, "paired_rollout_summary.json"
        )

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    task_solver = {}
    for i, ckpt_path in enumerate(args.agents):
        ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=ckpt_path)
        assert (
            ckpt_dict["algo_name"] == "residual_policy"
        ), "Only support residual policy"
        task_solver[f"task_{i}"] = FileUtils.policy_from_checkpoint(
            ckpt_dict=ckpt_dict,
            device=device,
            verbose=True,
            rollout_wrapper=ResidualRolloutPolicy,
        )

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    video_writers = {}
    if args.video_path is not None:
        for mode_name, _ in ROLLOUT_MODES:
            mode_video_path = build_video_path(args.video_path, mode_name)
            video_dir = os.path.dirname(mode_video_path)
            if video_dir:
                os.makedirs(video_dir, exist_ok=True)
            video_writers[mode_name] = imageio.get_writer(mode_video_path, fps=20)

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=task_solver["task_0"][1],
        env_name=args.env,
        render=args.render,
        render_offscreen=(
            (args.video_path is not None) or args.save_frames_to_hdf5
        ),
        verbose=True,
    )

    if args.env:
        print(f"Starting standard env rollout: {args.env}")
        n_rollouts = args.n_rollouts if args.n_rollouts is not None else 50
        test_keys = range(n_rollouts)
        failed_file = None
    else:
        print(f"Evaluating failed cases from: {args.failed_hdf5}")
        failed_file = h5py.File(args.failed_hdf5, "r")
        test_keys = list(failed_file["data"].keys())
        if args.n_rollouts:
            test_keys = test_keys[: args.n_rollouts]

    trajectory_hdf5 = None
    mode_results = {mode_name: [] for mode_name, _ in ROLLOUT_MODES}

    try:
        if trajectory_hdf5_path is not None:
            trajectory_hdf5 = h5py.File(trajectory_hdf5_path, "w")
            trajectory_hdf5.attrs["env_name"] = args.env if args.env else "failed_cases"
            trajectory_hdf5.attrs["agents"] = json.dumps(args.agents)
            trajectory_hdf5.attrs["horizon"] = args.horizon
            trajectory_hdf5.attrs["seed"] = -1 if args.seed is None else args.seed

        pbar = LogUtils.custom_tqdm(test_keys)
        for case_index, key in enumerate(pbar):
            fixed_initial_state = get_case_initial_state(env, failed_file, key)
            fixed_initial_state = clone_initial_state(fixed_initial_state)

            stats_by_mode = {}
            rollout_by_mode = {}
            rollout_meta_by_mode = {}
            initial_state_dict = None

            for mode_name, use_residual in ROLLOUT_MODES:
                stats, rollout_data, current_initial_state, rollout_metadata = run_rollout(
                    task_solver,
                    env,
                    args.horizon,
                    initial_state=fixed_initial_state,
                    render=args.render,
                    video_writer=video_writers.get(mode_name),
                    video_skip=args.video_skip,
                    camera_names=args.camera_names,
                    use_residual=use_residual,
                    save_render_frames=args.save_frames_to_hdf5,
                    frame_height=args.frame_height,
                    frame_width=args.frame_width,
                )
                stats_by_mode[mode_name] = stats
                rollout_by_mode[mode_name] = rollout_data
                rollout_meta_by_mode[mode_name] = rollout_metadata
                mode_results[mode_name].append(stats)
                if initial_state_dict is None:
                    initial_state_dict = current_initial_state

            if trajectory_hdf5 is not None:
                case_group = trajectory_hdf5.create_group(f"cases/case_{case_index:05d}")
                case_group.attrs["case_key"] = str(key)
                case_group.attrs["source_type"] = (
                    "failed_case" if failed_file is not None else "env"
                )
                write_case_to_hdf5(
                    case_group=case_group,
                    stats_by_mode=stats_by_mode,
                    rollout_by_mode=rollout_by_mode,
                    initial_state_dict=initial_state_dict,
                    rollout_meta_by_mode=rollout_meta_by_mode,
                )

            with_success_count = sum(
                result["Success_Rate"] for result in mode_results["with_residual"]
            )
            without_success_count = sum(
                result["Success_Rate"] for result in mode_results["without_residual"]
            )
            processed_cases = len(mode_results["with_residual"])
            pbar.set_postfix(
                With=f"{int(with_success_count)}/{processed_cases}",
                Without=f"{int(without_success_count)}/{processed_cases}",
            )

        summary = {
            "Exp_Metadata": {
                "Type": "Standard_Env" if args.env else "Failed_Cases",
                "Model_Count": len(task_solver),
                "Model_Paths": args.agents,
                "Trajectory_Output_Dir": args.trajectory_output_dir,
                "Save_Render_Frames_To_HDF5": args.save_frames_to_hdf5,
                "Frame_Height": args.frame_height,
                "Frame_Width": args.frame_width,
                "Frame_Cameras": args.camera_names,
            },
            "Statistics": {
                "with_residual": summarize_mode_results(mode_results["with_residual"]),
                "without_residual": summarize_mode_results(
                    mode_results["without_residual"]
                ),
                "paired_comparison": paired_summary(
                    mode_results["with_residual"],
                    mode_results["without_residual"],
                ),
            },
        }

        print("\n" + "=" * 30 + "\nFinal Experiment Summary\n" + "=" * 30)
        print(json.dumps(summary, indent=4))

        if json_path is not None:
            with open(json_path, "w") as json_file:
                json.dump(summary, json_file, indent=4)

        if trajectory_summary_path is not None:
            with open(trajectory_summary_path, "w") as json_file:
                json.dump(summary, json_file, indent=4)

    finally:
        if failed_file:
            failed_file.close()
        if trajectory_hdf5 is not None:
            trajectory_hdf5.close()
        for writer in video_writers.values():
            writer.close()
        env_close(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare residual and non-residual rollout trajectories."
    )

    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        required=True,
        help="Path to your trained .pth checkpoint",
    )
    parser.add_argument(
        "--failed_hdf5",
        type=str,
        default=None,
        help="Path to the failed_demo.hdf5 file",
    )
    parser.add_argument("--env", type=str, default=None, help="Testing env name")
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=None,
        help="Number of failed cases to test (default: all)",
    )
    parser.add_argument("--horizon", type=int, required=True, help="Maximum horizon")
    parser.add_argument("--render", action="store_true", help="On-screen rendering")
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Base path for rollout videos. The script saves one file per mode.",
    )
    parser.add_argument(
        "--video_skip", type=int, default=5, help="Render every n steps"
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["agentview"],
        help="Cameras for rendering",
    )
    parser.add_argument(
        "--output_to_txt_path", type=str, default=None, help="Output the log into txt"
    )
    parser.add_argument(
        "--trajectory_output_dir",
        type=str,
        default=None,
        help="Directory to save paired rollout trajectories and summary",
    )
    parser.add_argument(
        "--save_frames_to_hdf5",
        action="store_true",
        help="Save initial and per-step rendered frames into the trajectory HDF5",
    )
    parser.add_argument(
        "--frame_height",
        type=int,
        default=512,
        help="Rendered frame height for HDF5 storage",
    )
    parser.add_argument(
        "--frame_width",
        type=int,
        default=512,
        help="Rendered frame width for HDF5 storage",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="(optional) set seed for rollouts"
    )

    args = parser.parse_args()
    inference(args)

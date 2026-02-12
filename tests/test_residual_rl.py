import argparse
from collections import OrderedDict

import robomimic
import robomimic.utils.test_utils as TestUtils
from robomimic.utils.log_utils import silence_stdout
from robomimic.utils.torch_utils import dummy_context_mgr

def get_algo_base_config():
    """
    Base config for testing BCQ algorithms.
    """

    # config with basic settings for quick training run
    config = TestUtils.get_base_config(algo_name="residual_policy")

    # low-level obs (note that we define it here because @observation structure might vary per algorithm, 
    # for example HBC)
    config.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    config.observation.modalities.obs.rgb = ["agentview_image", "robot0_eye_in_hand_image"]


    return config

MODIFIERS = OrderedDict()
def register_mod(test_name):
    def decorator(config_modifier):
        MODIFIERS[test_name] = config_modifier
    return decorator


@register_mod("residual_rl")
def noop_modifier(config):
    # no-op
    return config


def test_residual_rl(silence=True, base_policy_path=""):
    for test_name in MODIFIERS:
        context = silence_stdout() if silence else dummy_context_mgr() 
        with context:
            base_config = get_algo_base_config()
            base_config.algo.base_policy.ckpt_path = base_policy_path
            res_str = TestUtils.test_run(base_config=base_config, config_modifier=MODIFIERS[test_name])
        print("{}: {}".format(test_name, res_str))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="don't suppress stdout during tests",
    )
    parser.add_argument(
        "--base_policy_path",
        type=str,
        required=True,
        help="don't suppress stdout during tests",
    )
    args = parser.parse_args()

    test_residual_rl(silence=(not args.verbose), base_policy_path=args.base_policy_path)


"""SONIC v1.1 heading-frame pelvis orientation goal input (heading_ori_obs)."""

import math
from pathlib import Path

import pytest
import torch

from supermimic.kinematics.rotations import quat_from_angle_axis, quat_mul, rot6d_from_quat
from supermimic.policy.nets import PolicyCfg, _goal_scale_vector
from supermimic.world_model.heads import heading_relative_ori

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"
Z = torch.tensor([0.0, 0.0, 1.0])
X = torch.tensor([1.0, 0.0, 0.0])
Y = torch.tensor([0.0, 1.0, 0.0])


def rot(axis: torch.Tensor, angle: float) -> torch.Tensor:
    return quat_from_angle_axis(torch.tensor([angle]), axis[None])  # (1,4)


def test_same_yaw_gives_reference_tilt_only():
    tilt = quat_mul(rot(Y, -0.2), rot(X, 0.3))  # body x-axis stays in the xz-plane: zero heading
    q = quat_mul(rot(Z, 1.1), tilt)
    out = heading_relative_ori(q, q[:, None, :])
    torch.testing.assert_close(out, rot6d_from_quat(tilt), atol=1e-6, rtol=0)


def test_yaw_error_and_invariance():
    robot = rot(Z, 0.4)
    ref = rot(Z, 0.4 + 0.7)
    out = heading_relative_ori(robot, ref[:, None, :])
    torch.testing.assert_close(out, rot6d_from_quat(rot(Z, 0.7)), atol=1e-6, rtol=0)
    world = rot(Z, -2.5)
    out2 = heading_relative_ori(quat_mul(world, robot), quat_mul(world, ref)[:, None, :])
    torch.testing.assert_close(out2, out, atol=1e-6, rtol=0)


def test_robot_tilt_is_ignored():
    ref = rot(Z, 0.9)[:, None, :].expand(1, 3, 4)
    upright = heading_relative_ori(rot(Z, 0.2), ref)
    tilted = heading_relative_ori(quat_mul(rot(Z, 0.2), rot(X, 0.5)), ref)
    torch.testing.assert_close(tilted, upright, atol=1e-6, rtol=0)
    assert upright.shape == (1, 18)


def test_goal_scale_with_heading_and_anchor():
    cfg = PolicyCfg(goal_frames=10)
    s = _goal_scale_vector(cfg, 10 * 68 + 60 + 9)
    assert s.shape == (749,)
    assert torch.equal(s[680:], torch.ones(69))
    torch.testing.assert_close(s[:68], s[68:136])


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA and LAFAN1 npz")
def test_env_matches_imagination_formula():
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.envs.state_io import StateIO

    files = discover_npz(NPZ, names=["walk1_subject1", "dance1_subject1"])
    args = SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=8, replay_capacity=64, seed=1, obs_noise=False, heading_ori_obs=True)
    env = make_env(args, device="cuda:0")
    try:
        obs, _ = env.reset()
        assert obs["goal"].shape == (8, 740)
        for _ in range(20):
            obs, *_ = env.step(0.3 * torch.randn(8, 29, device=env.device))
        cmd = env.command_manager.get_term("motion")
        s = StateIO(env).read()  # (8, 71), root quat of the free joint = pelvis
        expect = heading_relative_ori(s[:, 3:7], cmd.library.goal_root_quat(cmd.frame, 10))
        torch.testing.assert_close(obs["goal"][:, 680:], expect, atol=1e-4, rtol=0)
        torch.testing.assert_close(obs["goal"][:, :680], cmd.goal, atol=1e-6, rtol=0)
        assert not math.isclose(float(expect.std()), 0.0)
    finally:
        env.close()

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def end_effector_lin_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize end-effector linear velocity using L2 squared kernel.

    The function computes the L2 squared norm of the end-effector's linear velocity in world frame.
    This encourages the end-effector to move smoothly and avoid excessive velocities.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # get linear velocity of the end-effector body in world frame
    ee_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.sum(torch.square(ee_lin_vel_w), dim=1)


def end_effector_ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize end-effector angular velocity using L2 squared kernel.

    The function computes the L2 squared norm of the end-effector's angular velocity in world frame.
    This encourages the end-effector to rotate smoothly and avoid excessive angular velocities.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # get angular velocity of the end-effector body in world frame
    ee_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.sum(torch.square(ee_ang_vel_w), dim=1)


def randomize_end_effector_payload(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_range: tuple[float, float] = (0.1, 0.5),
    operation: str = "add",
    distribution: str = "uniform",
):
    """Randomize the mass of the end-effector to simulate random payload.
    
    This function adds random mass to the end-effector body to simulate carrying
    different payloads, which helps improve the robustness of the control policy.
    
    Args:
        env: The environment instance
        env_ids: Environment indices to randomize (None for all)
        asset_cfg: Scene entity configuration for the robot
        mass_range: Range of mass to add (min, max) in kg
        operation: Operation type ("add", "scale", "abs")
        distribution: Distribution type ("uniform", "log_uniform", "gaussian")
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    
    # get end-effector body index
    ee_body_id = asset_cfg.body_ids[0]
    
    # get current masses
    masses = asset.root_physx_view.get_masses()
    
    # reset to default mass first
    masses[env_ids, ee_body_id] = asset.data.default_mass[env_ids, ee_body_id].clone()
    
    # sample random mass values
    if distribution == "uniform":
        from isaaclab.utils.math import sample_uniform
        mass_samples = sample_uniform(
            torch.tensor(mass_range[0], device="cpu"),
            torch.tensor(mass_range[1], device="cpu"),
            (len(env_ids),),
            device="cpu"
        )
    elif distribution == "log_uniform":
        from isaaclab.utils.math import sample_log_uniform
        mass_samples = sample_log_uniform(
            torch.tensor(mass_range[0], device="cpu"),
            torch.tensor(mass_range[1], device="cpu"),
            (len(env_ids),),
            device="cpu"
        )
    elif distribution == "gaussian":
        from isaaclab.utils.math import sample_gaussian
        mean = (mass_range[0] + mass_range[1]) / 2
        std = (mass_range[1] - mass_range[0]) / 6  # 3-sigma rule
        mass_samples = sample_gaussian(
            torch.tensor(mean, device="cpu"),
            torch.tensor(std, device="cpu"),
            (len(env_ids),),
            device="cpu"
        )
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # apply operation
    if operation == "add":
        masses[env_ids, ee_body_id] += mass_samples
    elif operation == "scale":
        masses[env_ids, ee_body_id] *= mass_samples
    elif operation == "abs":
        masses[env_ids, ee_body_id] = mass_samples
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    # set the new masses
    asset.root_physx_view.set_masses(masses, env_ids)
    
    # recompute inertia tensors
    ratios = masses[env_ids, ee_body_id] / asset.data.default_mass[env_ids, ee_body_id]
    inertias = asset.root_physx_view.get_inertias()
    inertias[env_ids, ee_body_id] = asset.data.default_inertia[env_ids, ee_body_id] * ratios.unsqueeze(-1)
    asset.root_physx_view.set_inertias(inertias, env_ids)

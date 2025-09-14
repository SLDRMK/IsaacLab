# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.modifiers import DigitalFilterCfg

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=MISSING,  # depends on end-effector axis
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None

randomization = True
filter = False
single_step = True

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        if not randomization:
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
            actions = ObsTerm(func=mdp.last_action)
        elif not filter:
            if single_step:
                joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
                joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
                pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
                actions = ObsTerm(func=mdp.last_action)
            else:
                history_length = 5
                joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=history_length)
                joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=history_length)
                pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"}, history_length=history_length)
                actions = ObsTerm(func=mdp.last_action, history_length=history_length)
        else:
            history_length = 5
            # observation terms (order preserved)
            # 添加历史观察以减少噪声影响
            joint_pos = ObsTerm(
                func=mdp.joint_pos_rel, 
                noise=Unoise(n_min=-0.01, n_max=0.01),
                history_length=history_length,  # 新增：包含当前和前(history_length-1)步的观察
                modifiers=[
                    DigitalFilterCfg(
                        A=[0.8],  # 低通滤波：y[i] = 0.8*y[i-1] + 0.2*x[i]，平滑高频噪声
                        B=[0.2]
                    )
                ]
            )
            joint_vel = ObsTerm(
                func=mdp.joint_vel_rel, 
                noise=Unoise(n_min=-0.01, n_max=0.01),
                history_length=history_length,  # 新增：包含当前和前(history_length-1)步的观察
                modifiers=[
                    DigitalFilterCfg(
                        A=[0.0],           # 移动平均滤波：非递归
                        B=[0.2, 0.2, 0.2, 0.2, 0.2]  # 5步移动平均，进一步平滑速度信号
                    )
                ]
            )
            # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            pose_command = ObsTerm(
                func=mdp.generated_commands, 
                params={"command_name": "ee_pose"},
                history_length=history_length,  # 新增：包含当前和前(history_length-1)步的命令
                modifiers=[
                    DigitalFilterCfg(
                        A=[0.0],           # 移动平均滤波
                        B=[0.5, 0.5]      # 2步移动平均，平滑命令变化
                    )
                ]
            )
            actions = ObsTerm(
                func=mdp.last_action,
                history_length=history_length,  # 新增：包含当前和前(history_length-1)步的动作
                modifiers=[
                    DigitalFilterCfg(
                        A=[0.0],           # 移动平均滤波
                        B=[0.5, 0.5]      # 2步移动平均，平滑动作输出
                    )
                ]
            )

        def __post_init__(self):
            self.enable_corruption = True and randomization
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_end_effector_payload = EventTerm(
        func=mdp.randomize_end_effector_payload,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_range": (0.1, 0.5),  # 0.1 to 0.5 kg payload
            "operation": "add",
            "distribution": "uniform",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "std": 0.1, "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # # end-effector velocity penalties (initially disabled, enabled via curriculum)
    # end_effector_lin_vel = RewTerm(
    #     func=mdp.end_effector_lin_vel_l2,
    #     weight=0.0,  # Start with 0 weight, will be increased via curriculum
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING)},
    # )
    # end_effector_ang_vel = RewTerm(
    #     func=mdp.end_effector_ang_vel_l2,
    #     weight=0.0,  # Start with 0 weight, will be increased via curriculum
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING)},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    )

    # # end-effector velocity penalties curriculum
    # end_effector_lin_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "end_effector_lin_vel", "weight": -0.01, "num_steps": 20}
    # )
    # end_effector_ang_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "end_effector_ang_vel", "weight": -0.01, "num_steps": 20}
    # )


##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0

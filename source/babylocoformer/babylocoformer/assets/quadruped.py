# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import glob
import os

##
# Configuration
##
from pathlib import Path

import isaaclab.sim as sim_utils
from babylocoformer.assets.unitree import unitree_actuators
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

current_file = Path(__file__).resolve()
project_root = current_file.parents[4]
base_dir = project_root / "model" / "usds"
# base_dir = project_root / "model" / "test_usds"

print(base_dir)
usd_path = [
    path for path in glob.glob(os.path.join(base_dir, "**", "*.usd"), recursive=True) if "configuration" not in path
]
# print(usd_path)
QUADRUPED_CFG = ArticulationCfg(
    soft_joint_pos_limit_factor=0.9,
    spawn=sim_utils.MultiUsdFileCfg(
        usd_path=[
            path
            for path in glob.glob(os.path.join(base_dir, "**", "*.usd"), recursive=True)
            if "configuration" not in path
        ],
        random_choice=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            # sleep_threshold=0.005,
            # stabilization_threshold=0.001,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "GO2HV": unitree_actuators.UnitreeActuatorCfg_Go2HV(
            joint_names_expr=[".*"],
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
        ),
    },
)

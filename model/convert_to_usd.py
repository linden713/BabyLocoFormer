# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a URDF into USD format.

Unified Robot Description Format (URDF) is an XML file format used in ROS to describe all elements of
a robot. For more information, see: http://wiki.ros.org/urdf

This script uses the URDF importer extension from Isaac Sim (``isaacsim.asset.importer.urdf``) to convert a
URDF asset into USD format. It is designed as a convenience script for command-line use. For more
information on the URDF importer, see the documentation for the extension:
https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/ext_isaacsim_asset_importer_urdf.html


positional arguments:
  input               The path to the input URDF file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                Show this help message and exit
  --merge-joints            Consolidate links that are connected by fixed joints. (default: False)
  --fix-base                Fix the base to where it is imported. (default: False)
  --joint-stiffness         The stiffness of the joint drive. (default: 100.0)
  --joint-damping           The damping of the joint drive. (default: 1.0)
  --joint-target-type       The type of control to use for the joint drive. (default: "position")

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import glob
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to batch convert URDF files into USD format.")
parser.add_argument(
    "--input_dir",
    type=str,
    default=None,
    help="The path to the directory containing URDF files.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="The path to the directory to store the USD files.",
)
parser.add_argument(
    "--merge-joints",
    action="store_true",
    default=True,
    help="Consolidate links that are connected by fixed joints.",
)
parser.add_argument("--fix-base", action="store_true", default=False, help="Fix the base to where it is imported.")
parser.add_argument(
    "--joint-stiffness",
    type=float,
    default=100.0,
    help="The stiffness of the joint drive.",
)
parser.add_argument(
    "--joint-damping",
    type=float,
    default=1.0,
    help="The damping of the joint drive.",
)
parser.add_argument(
    "--joint-target-type",
    type=str,
    default="position",
    choices=["position", "velocity", "none"],
    help="The type of control to use for the joint drive.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import warnings

import omni.kit.app
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

from tqdm import tqdm


def main():
    warnings.filterwarnings("ignore")
    # Get input and output directories
    input_dir = os.path.abspath(args_cli.input_dir)
    output_dir = os.path.abspath(args_cli.output_dir)

    # Find all URDF files
    urdf_files = glob.glob(os.path.join(input_dir, "*.urdf"))
    if not urdf_files:
        print(f"No URDF files found in {input_dir}")
        return

    print(f"Found {len(urdf_files)} URDF files to convert.")

    # Process each file sequentially
    for urdf_path in tqdm(urdf_files):
        file_name = os.path.basename(urdf_path)
        model_name = os.path.splitext(file_name)[0]

        # Create destination path: /output_dir/model_name/model_name.usd
        dest_dir = os.path.join(output_dir, model_name)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, f"{model_name}.usd")

        # Create Urdf converter config
        urdf_converter_cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=os.path.dirname(dest_path),
            usd_file_name=os.path.basename(dest_path),
            fix_base=args_cli.fix_base,
            merge_fixed_joints=args_cli.merge_joints,
            force_usd_conversion=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=args_cli.joint_stiffness,
                    damping=args_cli.joint_damping,
                ),
                target_type=args_cli.joint_target_type,
            ),
        )

        # Create Urdf converter and import the file
        try:
            urdf_converter = UrdfConverter(urdf_converter_cfg)
            # print(f"Successfully generated USD file: {urdf_converter.usd_path}")
        except Exception as e:
            print(f"Failed to convert {urdf_path}. Error: {e}")

    print("\nBatch conversion complete.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

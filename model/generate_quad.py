import multiprocessing
import os
import subprocess
from itertools import product

import numpy as np


def generate_urdf(params):
    """
    Calls the xacro command to generate a URDF file.
    """
    base_len, base_width, leg_len = params
    output_dir = "model/generated_urdfs"
    filename = f"quad_bl{base_len:.3f}_bw{base_width:.3f}_ll{leg_len:.3f}.urdf"
    output_path = os.path.join(output_dir, filename)

    command = [
        "xacro",
        "model/quad.urdf.xacro",
        f"base_len:={base_len}",
        f"base_width:={base_width}",
        f"leg_len:={leg_len}",
        "-o",
        output_path,
    ]

    try:
        # Using capture_output=True to hide the large output of xacro unless there is an error
        subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate {output_path}: {e}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
    except Exception as e:
        print(f"An unexpected error occurred while generating {output_path}: {e}")


if __name__ == "__main__":
    # Create a directory to store the generated URDF files
    output_dir = "model/generated_urdfs"
    os.makedirs(output_dir, exist_ok=True)

    # Define the range for each parameter
    base_len_range = np.arange(0.25, 0.45, 0.02)
    base_width_range = np.arange(0.15, 0.35, 0.02)
    leg_len_range = np.arange(0.15, 0.25, 0.01)

    # Create a list of all parameter combinations
    param_combinations = list(product(base_len_range, base_width_range, leg_len_range))
    total_files = len(param_combinations)

    print(f"Generating {total_files} URDF files in parallel...")

    # Use a process pool to generate URDF files in parallel
    with multiprocessing.Pool() as pool:
        # Use imap_unordered to get progress
        for i, _ in enumerate(pool.imap_unordered(generate_urdf, param_combinations), 1):
            print(f"Progress: {i}/{total_files} ({i/total_files*100:.2f}%)", end="\r")

    print("\nFinished generating all URDF files.")

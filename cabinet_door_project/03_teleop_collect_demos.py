"""
Step 3: Teleoperate the Robot to Collect Demonstrations
========================================================
Opens an interactive window where you control the PandaOmron robot
with your keyboard (or SpaceMouse) to open cabinet doors.

This gives you hands-on intuition for the task. Note: this script does
NOT save demonstration data to disk. To get training data, run
04_download_dataset.py to download the pre-collected dataset.

Usage:
    # Mac users MUST use mjpython for the rendering window
    mjpython 03_teleop_collect_demos.py

    # Linux users
    python 03_teleop_collect_demos.py

    # Use spacemouse instead of keyboard
    python 03_teleop_collect_demos.py --device spacemouse

Keyboard Controls:
    Movement (hold to move):
        W/S     - Move arm forward / backward
        A/D     - Move arm left / right
        R/F     - Move arm up / down
        Z/X     - Rotate arm (yaw)
        T/G     - Rotate arm (pitch)
        C/V     - Rotate arm (roll)

    Actions:
        E       - Toggle gripper (open/close)
        B       - Toggle between arm control and base control

    Recording:
        Q       - Discard the current episode
"""

import os
import sys

# ── WSLg / XWayland GL setup — re-exec approach ─────────────────────────────
# On WSLg (Windows 11) the .bashrc often sets DISPLAY to a stale VcXsrv-style
# "IP:0" address, and Mesa's D3D12 GPU path fails with gladLoadGL on XWayland.
#
# Setting os.environ inside a running Python process is NOT reliable: C
# libraries (Mesa, GLFW) may read the env at dlopen() time which happens
# during import, before our code runs.  The only guaranteed fix is to restart
# the process with the correct vars already in the OS-level environment.
#
# We use os.execve() to atomically replace this process with an identical one
# that has the correct vars from the very start.  A sentinel env var prevents
# the new process from re-execing again.
if sys.platform == "linux" and "__TELEOP_DISPLAY_OK" not in os.environ:
    _env = dict(os.environ)
    _changed = False

    if _env.get("WAYLAND_DISPLAY"):
        # WSLg: force XWayland socket display and Mesa software renderer.
        if not _env.get("DISPLAY", "").startswith(":"):
            _env["DISPLAY"] = ":0"
            _changed = True
        if _env.get("GALLIUM_DRIVER") != "llvmpipe":
            _env["GALLIUM_DRIVER"] = "llvmpipe"
            _changed = True
        if _env.get("MESA_GL_VERSION_OVERRIDE") != "4.5":
            _env["MESA_GL_VERSION_OVERRIDE"] = "4.5"
            _changed = True

    if _changed:
        _env["__TELEOP_DISPLAY_OK"] = "1"
        os.execve(sys.executable, [sys.executable] + sys.argv, _env)
    else:
        os.environ["__TELEOP_DISPLAY_OK"] = "1"
# ────────────────────────────────────────────────────────────────────────────

import argparse
import time
from copy import deepcopy

import numpy as np
import robocasa  # noqa: F401 - registers environments including OpenCabinet
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper


def collect_trajectory(env, device, mirror_actions=True, max_fr=30):
    """
    Collect a single teleoperation trajectory.

    This is a simplified version of RoboCasa's collect_human_trajectory
    that avoids the circular import in robocasa.scripts.collect_demos.

    Returns:
        success (bool): Whether the cabinet was opened during the episode.
    """
    env.reset()

    ep_meta = env.get_ep_meta()
    lang = ep_meta.get("lang", None)
    if lang is not None:
        print(f"  Task: {lang}")

    # Counter: task must be successful for 15 consecutive timesteps.
    # Counts down from 14 to 0, then breaks on the next check (= 15 steps total).
    task_completion_hold_count = -1
    device.start_control()
    nonzero_ac_seen = False

    # Track gripper state
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # Do a dummy step to initialize
    zero_action = np.zeros(env.action_dim)
    env.step(zero_action)

    discard_traj = False

    while True:
        start = time.time()

        active_robot = env.robots[device.active_robot]

        # Get action from input device
        input_ac_dict = device.input2action(mirror_actions=mirror_actions)

        # None means the user pressed Q (reset signal)
        if input_ac_dict is None:
            discard_traj = True
            break

        action_dict = deepcopy(input_ac_dict)

        # Set arm actions based on controller type
        for arm in active_robot.arms:
            controller_input_type = active_robot.part_controllers[arm].input_type
            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]

        # Skip if no meaningful input yet (spacemouse idle)
        if not nonzero_ac_seen:
            is_empty = np.all(action_dict.get("right_delta", np.array([1])) == 0)
            if is_empty:
                continue
            nonzero_ac_seen = True

        # Build full action vector
        env_action = [
            robot.create_action_vector(all_prev_gripper_actions[i])
            for i, robot in enumerate(env.robots)
        ]
        env_action[device.active_robot] = active_robot.create_action_vector(
            action_dict
        )
        env_action = np.concatenate(env_action)

        # Step the environment
        obs, _, _, _ = env.step(env_action)

        # Check for task completion (15 consecutive success steps)
        if task_completion_hold_count == 0:
            break

        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 14
        else:
            task_completion_hold_count = -1

        # Frame rate limiting
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    success = not discard_traj
    return success


def _check_display():
    """Exit early with helpful instructions if no X display is available."""
    display = os.environ.get("DISPLAY", "")
    wayland = os.environ.get("WAYLAND_DISPLAY", "")

    if wayland:
        # WSLg is running. It provides XWayland at :0, which both pynput
        # (X11-based) and MuJoCo's GLFW viewer need.
        if not display or not display.startswith(":"):
            os.environ["DISPLAY"] = ":0"
        # Force Mesa software rendering (llvmpipe) for the GLFW window.
        # WSLg's Mesa D3D12 GPU path often fails with gladLoadGL; llvmpipe
        # is slower but reliably provides OpenGL 4.5 for the viewer.
        os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
        os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.5")
        return

    if display:
        # Some X display is claimed — let MuJoCo's own error handling deal
        # with actual render failures.
        return

    # Nothing set at all.
    print("ERROR: This script requires a display (X server) for the MuJoCo viewer")
    print("       and keyboard input. No display environment variable is set.")
    print()
    print("Windows 11 (WSLg — recommended, no extra software needed):")
    print("  WSLg provides a built-in display. If DISPLAY is not set, try:")
    print("  export DISPLAY=:0")
    print()
    print("Windows 10 / VcXsrv:")
    print("  1. Launch XLaunch, on 'Extra settings' uncheck 'Native opengl'")
    print("     and check 'Disable access control'")
    print("  2. export DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0")
    print("  3. export LIBGL_ALWAYS_INDIRECT=0")
    print()
    print("Note: steps 01, 02, 04-07 do NOT require a display.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Teleoperate robot for OpenCabinet")
    parser.add_argument(
        "--layout", type=int, default=None, help="Kitchen layout ID (1-60)"
    )
    parser.add_argument(
        "--style", type=int, default=None, help="Kitchen style ID (1-60)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="keyboard",
        choices=["keyboard", "spacemouse"],
        help="Input device",
    )
    args = parser.parse_args()

    _check_display()

    print("=" * 60)
    print("  OpenCabinet - Teleoperation Demo Collection")
    print("=" * 60)
    print()
    print("Controls:")
    print("  W/S/A/D/R/F  - Move arm (forward/back/left/right/up/down)")
    print("  Z/X/T/G/C/V  - Rotate arm")
    print("  E             - Toggle gripper")
    print("  B             - Toggle arm/base control mode")
    print("  Q             - Discard the current episode")
    print()

    # Create the environment
    config = {
        "env_name": "OpenCabinet",
        "robots": "PandaOmron",
        "controller_configs": load_composite_controller_config(robot="PandaOmron"),
        "layout_ids": args.layout,
        "style_ids": args.style,
        "translucent_robot": True,
    }

    print("Initializing environment (this may take a moment)...")
    env = robosuite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="robot0_frontview",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mjviewer",
    )

    env = VisualizationWrapper(env)

    # Initialize input device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(env=env, pos_sensitivity=4.0, rot_sensitivity=4.0)
    elif args.device == "spacemouse":
        import robocasa.macros as macros
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=4.0,
            rot_sensitivity=4.0,
            vendor_id=macros.SPACEMOUSE_VENDOR_ID,
            product_id=macros.SPACEMOUSE_PRODUCT_ID,
        )

    print("\nReady! Move the robot to open the cabinet door.")
    print("Press Q when done with each episode.\n")

    # Collect demonstrations in a loop
    episode = 0
    try:
        while True:
            episode += 1
            print(f"--- Episode {episode} ---")
            success = collect_trajectory(env, device, mirror_actions=True, max_fr=30)
            status = "SUCCESS" if success else "Discarded"
            print(f"  Result: {status}\n")
    except KeyboardInterrupt:
        print("\nTeleoperation ended.")
    finally:
        env.close()


if __name__ == "__main__":
    main()

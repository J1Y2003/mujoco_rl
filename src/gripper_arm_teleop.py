import os
import mujoco
import mujoco.viewer
import time
import json
from datetime import datetime
import numpy as np
from pynput import keyboard
from robot_descriptions import panda_mj_description

os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

active_keys = set()

def on_press(key):
    try:
        active_keys.add(key.char.upper())
    except AttributeError:
        active_keys.add(key) # Captures special keys like arrows

def on_release(key):
    try:
        k = key.char.upper()
        if k in active_keys: active_keys.remove(k)
    except AttributeError:
        if key in active_keys: active_keys.remove(key)

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def main():
    data_dir = "rl_training_data"
    save_dir = os.path.join(data_dir, datetime.now().strftime("%Y-%m%d-%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)

    is_recording = False
    current_trajectory = []
    episode_counter = 0

    record_button_was_pressed = False
    discard_button_was_pressed = False

    RECORD_HZ = 60

    # Calculate how many physics steps (0.001s) make up one recording frame
    record_interval = max(1, int((1.0 / RECORD_HZ) / 0.001))
    physics_step_counter = 0

    # --- EEF TARGET BOUNDING BOX ---
    # Pulled in from the raw workspace edges for a safety margin, and visualized
    # in the scene (ik_bounds_viz) below so the operator has a reference to teleop against.
    X_BOUNDS = [-0.35, 0.35]   # Left / Right
    Y_BOUNDS = [-1.05, 0.05]   # Forward / Backward
    Z_BOUNDS = [0.97, 1.35]

    bounds_center = [(b[0] + b[1]) / 2 for b in (X_BOUNDS, Y_BOUNDS, Z_BOUNDS)]
    bounds_half_size = [(b[1] - b[0]) / 2 for b in (X_BOUNDS, Y_BOUNDS, Z_BOUNDS)]

    # 1. define environment xml with drill asset and all 15 collision hulls
    scene_xml = f"""
    <mujoco>
        <option timestep="0.001" impratio="10" iterations="50"/>

        <visual>
            <global offwidth="1920" offheight="1080"/>
            <quality shadowsize="4096" numslices="28"/>
            <map fogstart="3" fogend="10" shadowclip="0.1"/>

            <headlight ambient="0.3 0.3 0.3" diffuse="0.2 0.2 0.2" specular="0 0 0"/>
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0 0 0" width="512" height="512"/>

            <texture name="tex_wood" type="2d" file="textures/table.png"/>
            <texture name="tex_floor_img" type="2d" file="textures/floor.png"/>
            <texture name="tex_wall_img" type="2d" file="textures/wall.png"/>
            <texture name="tex_metal_img" type="2d" file="textures/metal.png"/>

            <material name="mat_table" texture="tex_wood" reflectance="0.15" specular="0.4" shininess="0.5"/>
            <material name="mat_floor" texture="tex_floor_img" texrepeat="5 5" reflectance="0.1"/>
            <material name="mat_wall" texture="tex_wall_img" texrepeat="3 3" reflectance="0.02"/>
            <material name="mat_metal" texture="tex_metal_img" reflectance="0.5" specular="0.8" shininess="0.8"/>
            <material name="mat_cube" rgba="0.0 0.3 0.9 1" reflectance="0.4"/>

            <texture name="drill_tex" type="2d" file="obj/drill/drill_diffuse.png"/>
            <material name="drill_mat" texture="drill_tex" specular="0.5" shininess="0.5"/>

            <mesh name="drill_visual" file="obj/drill/drill.obj" scale="0.014 0.014 0.014"/>

            <include file="drill_assets.xml"/>
        </asset>

        <worldbody>
            <light name="studio_key" pos="0 0 3" dir="0 -0.5 -1" diffuse="0.4 0.4 0.4" specular="0.3 0.3 0.3" directional="true" castshadow="true"/>

            <geom name="floor" type="plane" size="4 4 0.1" material="mat_floor"/>
            <geom name="wall_back" type="box" size="4 0.1 2.5" pos="0 -2 2.5" material="mat_wall"/>
            <geom name="wall_left" type="box" size="0.1 4 2.5" pos="-3 0 2.5" material="mat_wall"/>
            <geom name="wall_right" type="box" size="0.1 4 2.5" pos="3 0 2.5" material="mat_wall"/>

            <body name="table" pos="0 -0.9 0">
                <geom name="tabletop" type="box" size="0.6 0.35 0.02" pos="0 0 0.78" material="mat_table"/>

                <geom name="leg_fl" type="box" size="0.02 0.02 0.38" pos="-0.55 -0.30 0.38" material="mat_metal"/>
                <geom name="leg_fr" type="box" size="0.02 0.02 0.38" pos=" 0.55 -0.30 0.38" material="mat_metal"/>
                <geom name="leg_bl" type="box" size="0.02 0.02 0.38" pos="-0.55  0.30 0.38" material="mat_metal"/>
                <geom name="leg_br" type="box" size="0.02 0.02 0.38" pos=" 0.55  0.30 0.38" material="mat_metal"/>
            </body>

            <body name="pedestal_base" pos="0 0 0">

                <geom name="wheel_fl" type="cylinder" size="0.04 0.02" pos="-0.2 -0.4 0.04" euler="90 0 0" material="mat_metal"/>
                <geom name="wheel_fr" type="cylinder" size="0.04 0.02" pos=" 0.2 -0.4 0.04" euler="90 0 0" material="mat_metal"/>
                <geom name="wheel_bl" type="cylinder" size="0.04 0.02" pos="-0.2  0.0 0.04" euler="90 0 0" material="mat_metal"/>
                <geom name="wheel_br" type="cylinder" size="0.04 0.02" pos=" 0.2  0.0 0.04" euler="90 0 0" material="mat_metal"/>

                <geom name="cabinet_body" type="box" size="0.25 0.25 0.21" pos="0 -0.2 0.30" material="mat_metal"/>
                <geom name="drawer_line" type="box" size="0.255 0.255 0.005" pos="0 -0.2 0.40" rgba="0.1 0.1 0.1 1"/>

                <body name="platform_head" pos="0 -0.2 0.51" euler="-5 0 0">
                    <geom name="mount_spacer" type="box" size="0.14 0.14 0.14" pos="0 0 0.14" material="mat_metal"/>
                    <geom name="mount_plate" type="cylinder" size="0.12 0.01" pos="0 0 0.29" material="mat_metal"/>
                    <site name="robot_mount" pos="0 0 0.3" euler="0 0 -90"/>
                </body>
            </body>

            <body name="cube" pos="0.0 -0.9 0.85">
                <joint type="free"/>
                <geom type="box" size="0.03 0.03 0.03" material="mat_cube"
                        mass="0.01" friction="1.5 0.5 0.1"/>
            </body>

            <body name="drill" pos="0.2 -0.9 0.85" euler="0 0 90">
                <freejoint/>

                <geom type="mesh" mesh="drill_visual" material="drill_mat" contype="0" conaffinity="0" group="1" mass="1.5"/>

                <include file="drill_collisions.xml"/>
            </body>

            <geom name="back_wall" type="box" size="2 0.05 1" pos="0 1.5 1" rgba="0.7 0.7 0.7 1"/>

            <geom name="ik_bounds_viz" type="box"
                  pos="{bounds_center[0]} {bounds_center[1]} {bounds_center[2]}"
                  size="{bounds_half_size[0]} {bounds_half_size[1]} {bounds_half_size[2]}"
                  rgba="0.3 1 0.3 0.05" contype="0" conaffinity="0" group="2"/>

            <body name="ik_target" mocap="true" pos="0 0 0">
                <geom type="sphere" size="0.04" rgba="0 1 0 0.4" contype="0" conaffinity="0"/>
            </body>

        </worldbody>

        <equality>
            <weld body1="ik_target" body2="franka_hand" relpose="0 0 0 1 0 0 0" solref="0.005 1" solimp="0.99 0.99 0.01"/>
        </equality>

    </mujoco>
    """

    arm_dir = os.path.dirname(panda_mj_description.MJCF_PATH)
    arm_path = os.path.join(arm_dir, "panda.xml")

    scene_spec = mujoco.MjSpec.from_string(scene_xml)
    arm_spec = mujoco.MjSpec.from_file(arm_path)

    scene_spec.attach(arm_spec, site="robot_mount", prefix="franka_")
    model = scene_spec.compile()
    data = mujoco.MjData(model)


    # --- MINIMAL FIX 1: GRAVITY COMPENSATION ---
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if body_name.startswith("franka_link"):
            model.body_gravcomp[i] = 1.0

    # --- NEW: TURN OFF ARM MOTORS (Let the Weld Constraint do the work) ---
    for i in range(7):
        model.actuator_gainprm[i, 0] = 0.0

    # Add damping to the gripper fingers so they don't jitter or explode
    for jname in ["franka_finger_joint1", "franka_finger_joint2"]:
        dof_adr = model.jnt_dofadr[model.joint(jname).id]
        model.dof_damping[dof_adr] = 10.0

    robot_qpos_adr = model.jnt_qposadr[model.joint("franka_joint1").id]

    print(f"Total Actuators to Control: {model.nu}")

    franka_home = [0.0, -0.785, 0.0, -2.356, 0.0, 2.2, 0.785]

    # Gripper: 2 finger qpos slots (open = 0.04 each), 1 tendon-coupled actuator (ctrlrange 0-255)
    GRIP_OPEN_CTRL = 255.0
    GRIP_CLOSED_CTRL = 0.0
    finger_home = [0.04, 0.04]

    data.qpos[robot_qpos_adr : robot_qpos_adr + 9] = franka_home + finger_home
    data.ctrl[:8] = franka_home + [GRIP_OPEN_CTRL]

    mujoco.mj_forward(model, data)

    # Get the ID of the wrist to track its position
    eef_body_name = "franka_hand"
    eef_body_id = model.body(eef_body_name).id

    # Initialize the "Drone" target coordinate
    target_pos = np.array([0.0, -0.6, 1.3])
    # Note: We use data.xmat for bodies, not data.site_xmat
    target_mat = data.xmat[eef_body_id].copy().reshape(3, 3)

    # State tracking
    grip_ratio = 0.0 # 0.0 = Open, 1.0 = Closed

    # Movement Speeds
    IK_SPEED = 0.0002
    HAND_SPEED = 0.001

    def reset_env():
        nonlocal grip_ratio

        mujoco.mj_resetData(model, data)

        target_pos[:] = [0.0, -0.6, 1.3]
        grip_ratio = 0.0

        # Snap the Franka arm AND gripper back to their home postures
        data.qpos[robot_qpos_adr : robot_qpos_adr + 9] = franka_home + finger_home
        data.ctrl[:8] = franka_home + [GRIP_OPEN_CTRL]

        # Update the kinematic trees with the new positions
        mujoco.mj_forward(model, data)

    # --- UNIFIED MAC JOY-CON SETUP ---
    pygame.init()
    pygame.joystick.init()

    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"\n🎮 GAMEPAD CONNECTED: {joystick.get_name()}")
    else:
        print("\n⚠️ No Gamepad detected. Falling back to keyboard.")

    print("\n--- INVERSE KINEMATICS ACTIVE ---")
    print("Click this terminal window to avoid MuJoCo shortcuts!")
    print("Fly X/Y : Arrow Keys | Fly Z: W/S | Hand: A/D")

    mocap_id = model.body("ik_target").mocapid[0]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.3
        viewer.cam.azimuth = 225
        viewer.cam.elevation = -30
        viewer.cam.lookat = [0, -0.9, 0.9]

        while viewer.is_running():
            step_start = time.time()

            # Read all new inputs from the controller
            pygame.event.pump()

            if joystick:
                # 1. Left Stick (Move X and Y)
                left_x = joystick.get_axis(0)
                left_y = joystick.get_axis(1)

                if abs(left_x) > 0.25: target_pos[0] -= left_x * IK_SPEED
                if abs(left_y) > 0.25: target_pos[1] += left_y * IK_SPEED

                # 2. L / R Bumpers (Lower and Lift Z)
                if joystick.get_button(9): # L Bumper (Lift)
                    target_pos[2] += IK_SPEED
                if joystick.get_button(10): # R Bumper (Lower)
                    target_pos[2] -= IK_SPEED

                # 3. A / B Buttons (Grip Control)
                if joystick.get_button(0): # Usually 'B' (or 'A' on Nintendo layout)
                    grip_ratio = np.clip(grip_ratio + HAND_SPEED, 0.0, 1.0)
                if joystick.get_button(1): # Usually 'A' (or 'B' on Nintendo layout)
                    grip_ratio = np.clip(grip_ratio - HAND_SPEED, 0.0, 1.0)

                # 4. X Button (Reset Environment)
                # Depending on the Mac driver, X is usually 2 or 3
                if joystick.get_button(2):
                    reset_env()

                # --- NEW: RECORDING AND DISCARD LOGIC ---
                record_button_is_pressed = joystick.get_button(3)  # 'X' Button (Start / Stop & Save)
                discard_button_is_pressed = joystick.get_button(2) # 'Y' Button (Trash current run)

                # 1. Start or Save a run
                if record_button_is_pressed and not record_button_was_pressed:
                    if not is_recording:
                        # START RECORDING
                        is_recording = True
                        current_trajectory = []
                        print(f"\n🔴 RECORDING STARTED: Episode {episode_counter}")
                    else:
                        # STOP AND SAVE
                        is_recording = False
                        print(f"⏹️ RECORDING STOPPED: Saved {len(current_trajectory)} steps to Episode {episode_counter}.")

                        # --- NEW: SAVE AS JSON ---
                        filename = os.path.join(save_dir, f"episode_{episode_counter}.json")

                        # Convert all NumPy arrays to standard lists so JSON can read them
                        json_ready_trajectory = []
                        for step in current_trajectory:
                            json_step = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in step.items()}
                            json_ready_trajectory.append(json_step)

                        # Save with indent=4 so it is beautifully formatted and easy to read!
                        with open(filename, 'w') as f:
                            json.dump(json_ready_trajectory, f, indent=4)
                        # -------------------------

                        episode_counter += 1

                # 2. Discard a run
                if discard_button_is_pressed and not discard_button_was_pressed:
                    if is_recording:
                        # TRASH IT
                        is_recording = False
                        print(f"\n🗑️ RECORDING DISCARDED: Threw away {len(current_trajectory)} steps.")
                        current_trajectory = [] # Empty the buffer without saving
                    else:
                        print("\n⚠️ Not currently recording, nothing to discard.")

                # Update debounce states
                record_button_was_pressed = record_button_is_pressed
                discard_button_was_pressed = discard_button_is_pressed
                # ----------------------------------------


            # --- 1. MOVE THE TARGET COORDINATE ---
            if keyboard.Key.down in active_keys:    target_pos[1] += IK_SPEED
            if keyboard.Key.up in active_keys:      target_pos[1] -= IK_SPEED
            if keyboard.Key.right in active_keys:   target_pos[0] -= IK_SPEED
            if keyboard.Key.left in active_keys:    target_pos[0] += IK_SPEED
            if 'W' in active_keys:                  target_pos[2] += IK_SPEED
            if 'S' in active_keys:                  target_pos[2] -= IK_SPEED
            if 'R' in active_keys:                  reset_env()

            # --> NEW: ENFORCE THE BOUNDING BOX <--
            target_pos[0] = np.clip(target_pos[0], X_BOUNDS[0], X_BOUNDS[1])
            target_pos[1] = np.clip(target_pos[1], Y_BOUNDS[0], Y_BOUNDS[1])
            target_pos[2] = np.clip(target_pos[2], Z_BOUNDS[0], Z_BOUNDS[1])

            # Render the ghost sphere
            data.mocap_pos[mocap_id] = target_pos

            mat = target_mat.flatten()
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, mat)
            data.mocap_quat[mocap_id] = quat

            # --- 3. HAND CONTROL (Parallel Gripper) ---

            # 1. Update the overall grip ratio (0.0 = Open, 1.0 = Closed)
            if 'A' in active_keys: # Close
                grip_ratio = np.clip(grip_ratio + HAND_SPEED, 0.0, 1.0)

            if 'D' in active_keys: # Open
                grip_ratio = np.clip(grip_ratio - HAND_SPEED, 0.0, 1.0)

            # 2. Interpolate between the open ctrl value and the closed ctrl value
            current_gripper_ctrl = GRIP_OPEN_CTRL + grip_ratio * (GRIP_CLOSED_CTRL - GRIP_OPEN_CTRL)

            # 3. Apply to MuJoCo control (single tendon-coupled actuator)
            data.ctrl[7] = current_gripper_ctrl

            mujoco.mj_step(model, data)

            # --- NEW: RECORD DATA IF ACTIVE ---
            if is_recording and (physics_step_counter % record_interval == 0):
                cube_id = model.body("cube").id

                # Bundle the State and Action into a dictionary
                step_data = {
                    # --- ACTIONS (What you told it to do) ---
                    "action_target_pos": target_pos.copy(),
                    "action_grip_ratio": np.array([grip_ratio]),

                    # --- STATES (How the physics reacted) ---
                    # Robot joints
                    "robot_qpos": data.qpos[robot_qpos_adr : robot_qpos_adr + 9].copy(),
                    "robot_qvel": data.qvel[robot_qpos_adr : robot_qpos_adr + 9].copy(),

                    # End Effector (Wrist) coordinates
                    "eef_pos": data.xpos[eef_body_id].copy(),
                    "eef_quat": data.xquat[eef_body_id].copy(),

                    # Object coordinates
                    "cube_pos": data.xpos[cube_id].copy(),
                    "cube_quat": data.xquat[cube_id].copy(),
                }
                current_trajectory.append(step_data)

            physics_step_counter += 1
            # ----------------------------------

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()

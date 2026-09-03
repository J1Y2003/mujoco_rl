import os
import json
import mujoco
import mujoco.viewer
import time
import numpy as np
from pynput import keyboard
from robot_descriptions import panda_mj_description
from robot_descriptions import robotiq_2f85_mj_description

os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

active_keys = set()

# Space/backtick are edge-triggered (act once per physical press, not once per frame
# held), unlike the movement keys above which are read as held state every frame.
# Backtick (`, above Tab) is used for delete-last-recording rather than Backspace since
# Backspace gets pressed constantly for other reasons and was deleting good episodes.
_record_toggle_pending = False
_delete_last_pending = False
_space_held = False
_backtick_held = False

def on_press(key):
    global _record_toggle_pending, _delete_last_pending, _space_held, _backtick_held
    if key == keyboard.Key.space:
        if not _space_held:
            _record_toggle_pending = True
        _space_held = True
        return
    if getattr(key, "char", None) == "`":
        if not _backtick_held:
            _delete_last_pending = True
        _backtick_held = True
        return
    try:
        active_keys.add(key.char.upper())
    except AttributeError:
        active_keys.add(key) # Captures special keys like arrows

def on_release(key):
    global _space_held, _backtick_held
    if key == keyboard.Key.space:
        _space_held = False
        return
    if getattr(key, "char", None) == "`":
        _backtick_held = False
        return
    try:
        k = key.char.upper()
        if k in active_keys: active_keys.remove(k)
    except AttributeError:
        if key in active_keys: active_keys.remove(key)

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def main():
    global _record_toggle_pending, _delete_last_pending

    DATA_DIR = "rl_training_data"
    ROBOT_KEY = "franka_robotiq_2f85"
    OBJECT_KEY = "alphabet_soup_can"
    DEFAULT_FPS = 15.0
    RESET_JITTER_RANGE = 0.15  # meters, +/- around the object's default spawn XY

    # --- EEF TARGET BOUNDING BOX ---
    # Pulled in from the raw workspace edges for a safety margin, and visualized
    # in the scene (ik_bounds_viz) below so the operator has a reference to teleop against.
    X_BOUNDS = [-0.28, 0.28]   # Left / Right
    Y_BOUNDS = [-0.95, -0.55]  # Forward / Backward (limits how far the arm reaches into the table)
    Z_BOUNDS = [0.99, 1.30]    # mount-frame height; fingertips sit ~0.116m below this

    # 1. define environment xml with a single graspable object (alphabet_soup_can)
    scene_xml = """
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

            <texture name="tex_soup_can" type="2d" file="obj/alphabet_soup_can/alphabet_soup_can_diffuse.png"/>
            <material name="mat_soup_can" texture="tex_soup_can" specular="0.1" shininess="0.2"/>

            <mesh name="soup_can_visual" file="obj/alphabet_soup_can/alphabet_soup_can_visual.obj"/>
            <mesh name="soup_can_collision" file="obj/alphabet_soup_can/alphabet_soup_can_collision.obj"/>
        </asset>

        <worldbody>
            <light name="studio_key" pos="0 0 3" dir="0 -0.5 -1" diffuse="0.4 0.4 0.4" specular="0.3 0.3 0.3" directional="true" castshadow="true"/>

            <geom name="floor" type="plane" size="4 4 0.1" material="mat_floor"/>
            <geom name="wall_back" type="box" size="4 0.1 2.5" pos="0 -2 2.5" material="mat_wall"/>
            <geom name="wall_left" type="box" size="0.1 4 2.5" pos="-3 0 2.5" material="mat_wall"/>
            <geom name="wall_right" type="box" size="0.1 4 2.5" pos="3 0 2.5" material="mat_wall"/>

            <body name="table" pos="0 -0.9 0">
                <geom name="tabletop" type="box" size="0.6 0.35 0.02" pos="0 0 0.78" material="mat_table" friction="0.4 0.005 0.0001"/>

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

                <body name="platform_head" pos="0 -0.2 0.39">
                    <geom name="mount_spacer" type="box" size="0.14 0.14 0.14" pos="0 0 0.14" material="mat_metal"/>
                    <geom name="mount_plate" type="cylinder" size="0.12 0.01" pos="0 0 0.29" material="mat_metal"/>
                    <site name="robot_mount" pos="0 0 0.3" euler="0 0 -90"/>
                </body>
            </body>

            <body name="soup_can" pos="0.0 -0.9 0.80">
                <freejoint/>
                <inertial pos="0.000168 0.001860 0.039739" mass="0.116152"
                          fullinertia="8.110456e-05 8.117612e-05 6.183146e-05 4.720034e-08 -3.390473e-08 -1.691195e-07"/>
                <geom type="mesh" mesh="soup_can_visual" material="mat_soup_can" contype="0" conaffinity="0" group="1"/>
                <geom type="mesh" mesh="soup_can_collision" friction="1.5 0.005 0.0001" rgba="1 1 1 0"/>
            </body>

            <geom name="back_wall" type="box" size="2 0.05 1" pos="0 1.5 1" rgba="0.7 0.7 0.7 1"/>

            <body name="ik_target" mocap="true" pos="0 0 0">
                <geom type="sphere" size="0.02" rgba="0 1 0 0.4" contype="0" conaffinity="0"/>
            </body>

        </worldbody>

        <equality>
            <weld body1="ik_target" body2="franka_hand_base_mount" relpose="0 0 0 1 0 0 0" solref="0.005 1" solimp="0.99 0.99 0.01"/>
        </equality>

    </mujoco>
    """

    arm_dir = os.path.dirname(panda_mj_description.MJCF_PATH)
    arm_path = os.path.join(arm_dir, "panda_nohand.xml")

    gripper_dir = os.path.dirname(robotiq_2f85_mj_description.MJCF_PATH)
    gripper_path = os.path.join(gripper_dir, "2f85.xml")

    scene_spec = mujoco.MjSpec.from_string(scene_xml)
    arm_spec = mujoco.MjSpec.from_file(arm_path)
    gripper_spec = mujoco.MjSpec.from_file(gripper_path)

    attached_frame = arm_spec.attach(gripper_spec, site="attachment_site", prefix="hand_")
    attached_frame.pos = [0.0, 0.0, 0.0]
    attached_frame.quat = [1.0, 0.0, 0.0, 0.0]

    scene_spec.attach(arm_spec, site="robot_mount", prefix="franka_")
    model = scene_spec.compile()
    data = mujoco.MjData(model)

    # --- GRAVITY COMPENSATION (arm links only, gripper stays under normal gravity) ---
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if body_name.startswith("franka_link"):
            model.body_gravcomp[i] = 1.0

    # --- TURN OFF ARM MOTORS (Let the Weld Constraint do the work) ---
    for i in range(7):
        model.actuator_gainprm[i, 0] = 0.0

    robot_qpos_adr = model.jnt_qposadr[model.joint("franka_joint1").id]

    print(f"Total Actuators to Control: {model.nu}")

    # Solved (position+orientation IK) so the gripper starts centered over the table,
    # hovering just above the object, with fingers pointing straight down.
    franka_home = [0.0, 0.4225, 0.0, -1.7889, 0.0, 2.2114, -0.7854]

    # Robotiq 2F-85: 8 passive/coupled finger joints driven by a single tendon actuator
    # (ctrlrange 0-255: 0 = fully open, 255 = fully closed).
    GRIP_OPEN_CTRL = 0.0
    GRIP_CLOSED_CTRL = 255.0
    gripper_home_qpos = [0.0] * 8
    robot_qpos_len = 7 + len(gripper_home_qpos)

    data.qpos[robot_qpos_adr : robot_qpos_adr + robot_qpos_len] = franka_home + gripper_home_qpos
    data.ctrl[:8] = franka_home + [GRIP_OPEN_CTRL]

    mujoco.mj_forward(model, data)

    # --- RECORDER SETUP ---
    # "robot root" = the arm's fixed base link. It never moves (welded to the mount),
    # so its pose is captured once and used to express every recorded position/orientation
    # (objects, table) in the robot's own local frame instead of raw world coordinates.
    root_body_id = model.body("franka_link0").id
    root_pos = data.xpos[root_body_id].copy()
    root_mat = data.xmat[root_body_id].copy().reshape(3, 3)
    root_quat_conj = np.zeros(4)
    mujoco.mju_negQuat(root_quat_conj, data.xquat[root_body_id])

    can_body_id = model.body("soup_can").id
    can_joint_id = model.body_jntadr[can_body_id]
    can_qpos_adr = model.jnt_qposadr[can_joint_id]
    can_default_xy = data.qpos[can_qpos_adr : can_qpos_adr + 2].copy()

    table_world_pos = np.array([0.0, -0.9, 0.80])
    table_surface_z_rel = float((root_mat.T @ (table_world_pos - root_pos))[2])

    # Every joint belonging to the robot subtree (arm + gripper), in qpos order, with the
    # attach()-generated "franka_"/"franka_hand_" prefixes stripped for a clean output key.
    robot_joint_ids = sorted(
        (jid for jid in range(model.njnt)
         if robot_qpos_adr <= model.jnt_qposadr[jid] < robot_qpos_adr + robot_qpos_len),
        key=lambda jid: model.jnt_qposadr[jid],
    )
    robot_joint_names = [model.joint(jid).name.replace("franka_hand_", "").replace("franka_", "")
                          for jid in robot_joint_ids]
    robot_joint_qpos_adrs = [model.jnt_qposadr[jid] for jid in robot_joint_ids]

    def object_pose_relative_to_root():
        obj_pos_world = data.xpos[can_body_id]
        rel_pos = root_mat.T @ (obj_pos_world - root_pos)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, root_quat_conj, data.xquat[can_body_id])
        return rel_pos.tolist(), rel_quat.tolist()

    def capture_frame(frame_index, fps):
        obj_pos, obj_quat = object_pose_relative_to_root()
        joint_values = {name: float(data.qpos[adr])
                         for name, adr in zip(robot_joint_names, robot_joint_qpos_adrs)}
        return {
            "frame_index": frame_index,
            "timestamp": frame_index / fps,
            "phase": "play",
            "result_index": 0,
            "objects": {
                OBJECT_KEY: {"pos": obj_pos, "rotation": obj_quat},
            },
            "robot": {
                "pos": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "joint_values": joint_values,
            },
        }

    def save_episode(frames, fps, task_label):
        episode_dir = os.path.join(DATA_DIR, task_label)
        os.makedirs(episode_dir, exist_ok=True)
        existing_indices = []
        for fname in os.listdir(episode_dir):
            if fname.startswith("episode_") and fname.endswith(".json"):
                try:
                    existing_indices.append(int(fname[len("episode_"):-len(".json")]))
                except ValueError:
                    pass
        next_idx = max(existing_indices, default=-1) + 1
        filepath = os.path.join(episode_dir, f"episode_{next_idx}.json")

        payload = {
            "scene": {
                "rotation": "quat_wxyz",
                "robot_key": ROBOT_KEY,
                "object_keys": [OBJECT_KEY],
                "table_surface_z": table_surface_z_rel,
            },
            "episode": {
                "result_index": 0,
                "task": task_label,
                "metadata": {},
                "success": True,
                "failure_reason": None,
                "fps": fps,
                "frame_count": len(frames),
            },
            "frames": frames,
        }
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)
        return filepath

    # Get the ID of the gripper mount to track its position
    eef_body_name = "franka_hand_base_mount"
    eef_body_id = model.body(eef_body_name).id

    # Initialize the "Drone" target coordinate to the eef's current (home) position
    target_pos = data.xpos[eef_body_id].copy()
    target_mat = data.xmat[eef_body_id].copy().reshape(3, 3)

    # State tracking
    grip_ratio = 0.0 # 0.0 = Open, 1.0 = Closed

    # Movement Speeds
    IK_SPEED = 0.0002
    HAND_SPEED = 0.001

    def reset_env():
        nonlocal grip_ratio

        mujoco.mj_resetData(model, data)

        grip_ratio = 0.0

        # Snap the Franka arm AND gripper back to their home postures
        data.qpos[robot_qpos_adr : robot_qpos_adr + robot_qpos_len] = franka_home + gripper_home_qpos
        data.ctrl[:8] = franka_home + [GRIP_OPEN_CTRL]

        # Randomize the object's spawn position a bit so recordings aren't all identical
        jitter = np.random.uniform(-RESET_JITTER_RANGE, RESET_JITTER_RANGE, size=2)
        data.qpos[can_qpos_adr : can_qpos_adr + 2] = can_default_xy + jitter

        # Update the kinematic trees with the new positions
        mujoco.mj_forward(model, data)

        target_pos[:] = data.xpos[eef_body_id]

    # --- RECORDER STATE ---
    fps_input = input(f"Recording FPS (default {DEFAULT_FPS:g}): ").strip()
    try:
        record_fps = float(fps_input) if fps_input else DEFAULT_FPS
        if record_fps <= 0:
            raise ValueError
    except ValueError:
        print(f"Invalid FPS, using default {DEFAULT_FPS:g}.")
        record_fps = DEFAULT_FPS
    record_interval = max(1, round((1.0 / record_fps) / model.opt.timestep))
    physics_step_counter = 0

    is_recording = False
    current_frames = []
    frame_index = 0
    record_start_time = 0.0
    last_status_print = 0.0
    last_saved_path = None

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
    print("Fly X/Y : Arrow Keys | Fly Z: W/S | Hand: A/D | Reset: R")
    print(f"Record: Space (start/stop, {record_fps:g} fps) | Delete last saved: ` (backtick)")

    mocap_id = model.body("ik_target").mocapid[0]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = 225
        viewer.cam.elevation = -25
        viewer.cam.lookat = [0, -0.9, 0.85]

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
                if joystick.get_button(2):
                    reset_env()

            # --- 1. MOVE THE TARGET COORDINATE ---
            if keyboard.Key.down in active_keys:    target_pos[1] += IK_SPEED
            if keyboard.Key.up in active_keys:      target_pos[1] -= IK_SPEED
            if keyboard.Key.right in active_keys:   target_pos[0] -= IK_SPEED
            if keyboard.Key.left in active_keys:    target_pos[0] += IK_SPEED
            if 'W' in active_keys:                  target_pos[2] += IK_SPEED
            if 'S' in active_keys:                  target_pos[2] -= IK_SPEED
            if 'R' in active_keys:                  reset_env()

            # --- RECORDER: SPACE toggles start/stop, BACKSPACE deletes the last save ---
            if _record_toggle_pending:
                _record_toggle_pending = False
                if not is_recording:
                    is_recording = True
                    current_frames = []
                    frame_index = 0
                    physics_step_counter = 0
                    record_start_time = time.time()
                    print(f"\n🔴 RECORDING STARTED ({record_fps:g} fps)")
                else:
                    is_recording = False
                    duration = time.time() - record_start_time
                    print(f"\n⏹  RECORDING STOPPED — {len(current_frames)} frames, {duration:.1f}s")
                    if len(current_frames) == 0:
                        print("   Nothing captured, not saving.")
                    else:
                        task_label = input("   Save to which category/subdirectory? (e.g. roll, push, drop): ").strip()
                        if not task_label:
                            task_label = "uncategorized"
                        last_saved_path = save_episode(current_frames, record_fps, task_label)
                        print(f"   Saved to {last_saved_path}")
                    current_frames = []

            if _delete_last_pending:
                _delete_last_pending = False
                if last_saved_path and os.path.exists(last_saved_path):
                    os.remove(last_saved_path)
                    print(f"\n🗑  Deleted {last_saved_path}")
                    last_saved_path = None
                else:
                    print("\n⚠️  No previously saved recording to delete.")

            # --> ENFORCE THE BOUNDING BOX <--
            target_pos[0] = np.clip(target_pos[0], X_BOUNDS[0], X_BOUNDS[1])
            target_pos[1] = np.clip(target_pos[1], Y_BOUNDS[0], Y_BOUNDS[1])
            target_pos[2] = np.clip(target_pos[2], Z_BOUNDS[0], Z_BOUNDS[1])

            # Render the ghost sphere
            data.mocap_pos[mocap_id] = target_pos

            mat = target_mat.flatten()
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, mat)
            data.mocap_quat[mocap_id] = quat

            # --- 3. HAND CONTROL (Robotiq 2F-85 Parallel Gripper) ---

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

            # --- RECORDER: sample a frame every record_interval physics steps ---
            if is_recording:
                if physics_step_counter % record_interval == 0:
                    current_frames.append(capture_frame(frame_index, record_fps))
                    frame_index += 1
                now = time.time()
                if now - last_status_print > 0.2:
                    elapsed = now - record_start_time
                    print(f"\r🔴 REC  {elapsed:5.1f}s | {len(current_frames)} frames", end="", flush=True)
                    last_status_print = now
                physics_step_counter += 1

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()

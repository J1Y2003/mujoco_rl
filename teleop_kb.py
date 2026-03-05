import os
import mujoco
import mujoco.viewer
import time
import numpy as np
from pynput import keyboard
from robot_descriptions import panda_mj_description
from robot_descriptions import allegro_hand_mj_description

# The joints that actually curl the fingers (Flexion)
FINGER_Grip = [8, 9, 10, 12, 13, 14, 16, 17, 18] # Index, Middle, Ring (1-3)
THUMB_Grip = [20, 21, 22]                       # Thumb (1-3)

# Combine them for the main grasp
ALL_GRIP = FINGER_Grip + THUMB_Grip

# The rotation/spread joints you want to keep still (Neutral)
UNUSED_JOINTS = [7, 11, 15, 19] # ffa0, mfa0, rfa0, tha0

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
    scene_xml = """
    <mujoco>
        <worldbody>
            <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
            <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.9 0.8 1"/>
            
            <body name="table" pos="0 -0.6 0.4">
                <geom type="box" size="0.6 0.35 0.4" rgba="0.6 0.5 0.4 1"/>
            </body>
            
            <body name="pedestal_base" pos="0 0 0">
                <body name="platform_head" pos="0 -0.2 0.51" euler="-35 0 0">
                    <geom type="cylinder" size="0.18 0.18" pos="0 0 0" euler="0 90 0" rgba="0.1 0.1 0.1 1"/>
                    
                    <geom type="box" size="0.18 0.18 0.15" pos="0 0 0.15" rgba="0.25 0.25 0.25 1"/>
                    
                    <site name="robot_mount" pos="0 0 0.3" euler="0 0 -90"/>
                </body>
            </body>
            <geom name="back_wall" type="box" size="2 0.05 1" pos="0 1.5 1" rgba="0.7 0.7 0.7 1"/>

            <body name="ik_target" mocap="true" pos="0 0 0">
                <geom type="sphere" size="0.05" rgba="0 1 0 0.4" contype="0" conaffinity="0"/>
            </body>

        </worldbody>
    </mujoco>
    """

    arm_dir = os.path.dirname(panda_mj_description.MJCF_PATH)
    arm_path = os.path.join(arm_dir, "panda_nohand.xml")

    hand_dir = os.path.dirname(allegro_hand_mj_description.MJCF_PATH)
    all_files = os.listdir(hand_dir)
    right_hand_files = [f for f in all_files if 'right' in f and 'scene' not in f and f.endswith('.xml')]
    hand_path = os.path.join(hand_dir, right_hand_files[0])

    scene_spec = mujoco.MjSpec.from_string(scene_xml)
    arm_spec = mujoco.MjSpec.from_file(arm_path)
    hand_spec = mujoco.MjSpec.from_file(hand_path)

    attached_frame = arm_spec.attach(hand_spec, site="attachment_site", prefix="hand_")
    attached_frame.pos = [0.0, 0.0, 0.095]
    attached_frame.quat = [0.0, 0.7071, 0.0, 0.7071]

    scene_spec.attach(arm_spec, site="robot_mount", prefix="franka_")
    model = scene_spec.compile()
    data = mujoco.MjData(model)

    print(f"Total Actuators to Control: {model.nu}")

    franka_home = [0.0, -0.785, 0.0, -2.356, 0.0, 3, 0.785]

    allegro_home = [0.0] * 16

    default_qpos = franka_home + allegro_home

    data.qpos[:] = default_qpos

    data.ctrl[:] = default_qpos

    mujoco.mj_forward(model, data)
    
    mujoco.mj_step(model, data)

    # FIX 1: Find the exact Jacobian columns for the 7 arm actuators
    arm_dof_indices = []
    for i in range(7):
        # Find which joint this actuator controls
        joint_id = model.actuator(i).trnid[0]
        # Get its column index in the Jacobian matrix
        dof_adr = model.jnt_dofadr[joint_id]
        arm_dof_indices.append(dof_adr)
        
        # FIX 2: Initialize the motor targets to their current resting position
        qpos_adr = model.jnt_qposadr[joint_id]
        data.ctrl[i] = data.qpos[qpos_adr]

    # Get the ID of the wrist to track its position
    site_name = "franka_attachment_site" # Update if your site name is different!
    site_id = model.site(site_name).id
    
    # Initialize the "Drone" target coordinate
    target_pos = np.array([0.0, -0.3, 1.3])
    target_mat = data.site_xmat[site_id].copy().reshape(3, 3)

    # Movement Speeds
    # Movement Speeds
    IK_SPEED = 0.0007
    HAND_SPEED = 0.008

    X_BOUNDS = [-0.3, 0.3]   # Left / Right
    Y_BOUNDS = [-0.50, 0.1]    # Forward / Backward (assuming table is in +Y)
    Z_BOUNDS = [0.93, 1.2]

    print("\n--- INVERSE KINEMATICS ACTIVE ---")
    print("Click this terminal window to avoid MuJoCo shortcuts!")
    print("Fly X/Y : Arrow Keys | Fly Z: W/S | Hand: C/O")

    mocap_id = model.body("ik_target").mocapid[0]
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.5  # How far away (meters)
        viewer.cam.azimuth = 45     # Rotation around the robot (degrees)
        viewer.cam.elevation = -20  # Looking down angle (degrees)
        viewer.cam.lookat = [0, -0.6, 0.8] # Center the view on the table

        while viewer.is_running():
            step_start = time.time()

            # --- 1. MOVE THE TARGET COORDINATE ---
            if keyboard.Key.up in active_keys:    target_pos[1] += IK_SPEED 
            if keyboard.Key.down in active_keys:  target_pos[1] -= IK_SPEED 
            if keyboard.Key.left in active_keys:  target_pos[0] -= IK_SPEED 
            if keyboard.Key.right in active_keys: target_pos[0] += IK_SPEED 
            if 'W' in active_keys:                target_pos[2] += IK_SPEED 
            if 'S' in active_keys:                target_pos[2] -= IK_SPEED 

            # --> NEW: ENFORCE THE BOUNDING BOX <--
            target_pos[0] = np.clip(target_pos[0], X_BOUNDS[0], X_BOUNDS[1])
            target_pos[1] = np.clip(target_pos[1], Y_BOUNDS[0], Y_BOUNDS[1])
            target_pos[2] = np.clip(target_pos[2], Z_BOUNDS[0], Z_BOUNDS[1])

            # Render the ghost sphere
            data.mocap_pos[mocap_id] = target_pos

            # --- 2. THE 6-DOF JACOBIAN MATH ---
            current_pos = data.site_xpos[site_id]
            current_mat = data.site_xmat[site_id].reshape(3, 3)

            # A. Position Error (3D)
            err_pos = target_pos - current_pos

            # B. Orientation Error (3D)
            # We use the cross product of the X, Y, and Z axes of the rotation matrices
            # to calculate the rotational difference between 'current' and 'target'
            err_rot = 0.5 * (
                np.cross(current_mat[:, 0], target_mat[:, 0]) +
                np.cross(current_mat[:, 1], target_mat[:, 1]) +
                np.cross(current_mat[:, 2], target_mat[:, 2])
            )

            # Combine Position (3) and Rotation (3) into a single 6D error vector
            error_6d = np.concatenate([err_pos, err_rot])
            
            # Limit the max speed to prevent math explosions
            error_6d = np.clip(error_6d, -0.05, 0.05)

            # C. Extract both Position and Rotational Jacobians
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            
            # mj_jacSite calculates BOTH jacp (position) and jacr (rotation)
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            
            # Extract only the 7 arm columns
            J_arm_pos = jacp[:, arm_dof_indices]
            J_arm_rot = jacr[:, arm_dof_indices]
            
            # Stack them vertically into a 6x7 matrix
            J_arm_6d = np.vstack([J_arm_pos, J_arm_rot])

            # D. Solve for required joint angles
            # We are solving 6 equations using 7 joints
            dq = np.linalg.pinv(J_arm_6d) @ error_6d

            # ---> THE KINEMATIC OVERRIDE <---
            data.qpos[:7] += dq * 0.1
            data.ctrl[:7] = data.qpos[:7]
            mujoco.mj_kinematics(model, data)

            # --- 3. HAND CONTROL (Selective Grip) ---
            data.ctrl[UNUSED_JOINTS] = 0.0

            if 'A' in active_keys: # Close
                data.ctrl[ALL_GRIP] = np.clip(data.ctrl[ALL_GRIP] + HAND_SPEED, 0.0, 1.5)
            
            if 'D' in active_keys: # Open
                data.ctrl[ALL_GRIP] = np.clip(data.ctrl[ALL_GRIP] - HAND_SPEED, 0.0, 1.5)

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
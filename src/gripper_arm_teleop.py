import os
import mujoco
import mujoco.viewer
import time
import numpy as np
from pynput import keyboard
from robot_descriptions import panda_mj_description

active_keys = set()

def on_press(key):
    try:
        active_keys.add(key.char.upper())
    except AttributeError:
        active_keys.add(key) 

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
            
            <body name="table" pos="0 -0.9 0.4">
                <geom type="box" size="0.6 0.35 0.4" rgba="0.6 0.5 0.4 1"/>
            </body>
            
            <body name="tote_handle_object" pos="0.2 -0.9 0.85">
                <joint type="free"/>
                <geom type="box" size="0.1 0.05 0.07" rgba="0 0.4 0.8 1" mass="0.2"/>
                <geom type="box" size="0.01 0.01 0.05" pos="-0.08 0 0.12" rgba="1 1 1 1"/>
                <geom type="box" size="0.01 0.01 0.05" pos="0.08 0 0.12" rgba="1 1 1 1"/>
                <geom type="box" size="0.09 0.01 0.01" pos="0 0 0.16" rgba="1 1 1 1"/>
            </body>

            <body name="cube" pos="-0.2 -0.9 0.85">
                <joint type="free"/>
                <geom type="box" size="0.03 0.03 0.03" rgba="0 0.1 0.8 1" mass="0.1"/>
            </body>

            <body name="pedestal_base" pos="0 0 0">
                <body name="platform_head" pos="0 -0.2 0.51" euler="-5 0 0">
                    <geom type="cylinder" size="0.18 0.18" pos="0 0 0" euler="0 90 0" rgba="0.1 0.1 0.1 1"/>
                    <geom type="box" size="0.18 0.18 0.15" pos="0 0 0.15" rgba="0.25 0.25 0.25 1"/>
                    <site name="robot_mount" pos="0 0 0.3" euler="0 0 -90"/>
                </body>
            </body>

            <geom name="back_wall" type="box" size="2 0.05 1" pos="0 1.5 1" rgba="0.7 0.7 0.7 1"/>

            <body name="ik_target" mocap="true" pos="0 0 0">
                <geom type="sphere" size="0.04" rgba="0 1 0 0.4" contype="0" conaffinity="0"/>
            </body>

        </worldbody>
    </mujoco>
    """

    arm_dir = os.path.dirname(panda_mj_description.MJCF_PATH)
    arm_path = os.path.join(arm_dir, "panda.xml") 

    scene_spec = mujoco.MjSpec.from_string(scene_xml)
    arm_spec = mujoco.MjSpec.from_file(arm_path)

    scene_spec.attach(arm_spec, site="robot_mount", prefix="franka_")
    model = scene_spec.compile()
    data = mujoco.MjData(model)

    # --- DEBUGGING: DAMPING + ARMATURE ---
    for i in range(7, model.nu):
        joint_id = model.actuator(i).trnid[0]
        dof_adr = model.jnt_dofadr[joint_id]
        
        # Keep the damping we added
        model.dof_damping[dof_adr] = 100.0 
        
        # NEW: Add virtual motor inertia. 
        # This prevents the finger from changing direction instantly.
        model.dof_armature[dof_adr] = 1.0

    robot_qpos_adr = model.jnt_qposadr[model.joint("franka_joint1").id]

    print(f"Total Actuators to Control: {model.nu}")

    franka_home = [0.0, -0.785, 0.0, -2.356, 0.0, 2.2, 0.785]
    
    data.qpos[robot_qpos_adr : robot_qpos_adr + 7] = franka_home
    data.ctrl[:7] = franka_home
    data.ctrl[7:] = 0.04 

    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)

    arm_dof_indices = []
    for i in range(7):
        joint_id = model.actuator(i).trnid[0]
        dof_adr = model.jnt_dofadr[joint_id]
        arm_dof_indices.append(dof_adr)
        
        qpos_adr = model.jnt_qposadr[joint_id]
        data.ctrl[i] = data.qpos[qpos_adr]

    eef_body_name = "franka_hand"
    eef_body_id = model.body(eef_body_name).id
    
    target_pos = np.array([0.0, -0.6, 1.3])
    target_mat = data.xmat[eef_body_id].copy().reshape(3, 3)

    gripper_width = 0.04
    IK_SPEED = 0.0007
    HAND_SPEED = 1

    X_BOUNDS = [-0.4, 0.4]   
    Y_BOUNDS = [-1.1, 0.1]    
    Z_BOUNDS = [0.85, 1.4]

    def reset_env():
        nonlocal gripper_width # Required to modify the float variable from the outer scope
        
        # 1. Clear all physics state (resets objects, velocities, and forces to XML defaults)
        mujoco.mj_resetData(model, data)
        
        # 2. Reset IK tracking variables to their starting values
        target_pos[:] = [0.0, -0.6, 1.3]
        gripper_width = 0.04
        
        # 3. Snap the Franka arm back to its home posture
        data.qpos[robot_qpos_adr : robot_qpos_adr + 7] = franka_home
        data.ctrl[:7] = franka_home
        data.ctrl[7:] = gripper_width
        
        # 4. Update the kinematic trees with the new positions
        mujoco.mj_forward(model, data)

    print("\n--- INVERSE KINEMATICS ACTIVE ---")
    print("Click this terminal window to avoid MuJoCo shortcuts!")
    print("Fly X/Y : Arrow Keys | Fly Z: W/S | Hand: A/D")

    mocap_id = model.body("ik_target").mocapid[0]
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.0  
        viewer.cam.azimuth = 225     
        viewer.cam.elevation = -45  
        viewer.cam.lookat = [0, -0.6, 0.8] 

        while viewer.is_running():
            step_start = time.time()

            # --- 1. MOVE THE TARGET COORDINATE ---
            if keyboard.Key.down in active_keys:    target_pos[1] += IK_SPEED 
            if keyboard.Key.up in active_keys:      target_pos[1] -= IK_SPEED 
            if keyboard.Key.right in active_keys:   target_pos[0] -= IK_SPEED 
            if keyboard.Key.left in active_keys:    target_pos[0] += IK_SPEED 
            if 'W' in active_keys:                  target_pos[2] += IK_SPEED 
            if 'S' in active_keys:                  target_pos[2] -= IK_SPEED
            if 'R' in active_keys:                  reset_env()

            target_pos[0] = np.clip(target_pos[0], X_BOUNDS[0], X_BOUNDS[1])
            target_pos[1] = np.clip(target_pos[1], Y_BOUNDS[0], Y_BOUNDS[1])
            target_pos[2] = np.clip(target_pos[2], Z_BOUNDS[0], Z_BOUNDS[1])

            data.mocap_pos[mocap_id] = target_pos

            # --- 2. EXACT ORIGINAL 6-DOF JACOBIAN MATH ---
            curr_pos = data.xpos[eef_body_id]
            curr_mat = data.xmat[eef_body_id].reshape(3, 3)

            err_p = target_pos - curr_pos
            err_r = 0.5 * (np.cross(curr_mat[:, 0], target_mat[:, 0]) + np.cross(curr_mat[:, 1], target_mat[:, 1]) + np.cross(curr_mat[:, 2], target_mat[:, 2]))

            error_6d = np.clip(np.concatenate([err_p, err_r]), -0.05, 0.05)

            jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, eef_body_id)
            J = np.vstack([jacp[:, arm_dof_indices], jacr[:, arm_dof_indices]])

            J_pinv = np.linalg.pinv(J)
            null_space = np.eye(7) - J_pinv @ J
            
            posture_error = np.array(franka_home) - data.qpos[robot_qpos_adr : robot_qpos_adr + 7]
    
            dq = J_pinv @ error_6d + (null_space @ (posture_error * 0.5))

            data.qpos[robot_qpos_adr : robot_qpos_adr + 7] += dq * 0.1
            data.ctrl[:7] = data.qpos[robot_qpos_adr : robot_qpos_adr + 7]

            mujoco.mj_kinematics(model, data)

            # --- 3. HAND CONTROL ---
            if 'A' in active_keys: # Close
                # Subtracting to close down to 0.0
                gripper_width = np.clip(gripper_width - HAND_SPEED, 0.0, 300.0)

            if 'D' in active_keys: # Open
                # Adding to open up to 300.0
                gripper_width = np.clip(gripper_width + HAND_SPEED, 0.0, 300.0)

            # Apply to the gripper actuators
            data.ctrl[7:] = gripper_width

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
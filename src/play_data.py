import os
import time
import json
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from robot_descriptions import panda_mj_description
from robot_descriptions import shadow_hand_mj_description

def main():
    # --- 1. SETUP COMMAND LINE ARGUMENTS ---
    parser = argparse.ArgumentParser(description="Replay a recorded MuJoCo trajectory.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the .json trajectory file.")
    args = parser.parse_args()

    print(f"Loading trajectory from: {args.data_path}")
    with open(args.data_path, 'r') as f:
        trajectory = json.load(f)
    print(f"Loaded {len(trajectory)} frames.")

    # --- 2. REBUILD THE EXACT ENVIRONMENT ---
    # We must load the exact same XML and assets so the indices match perfectly.
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
            
            <material name="mat_cube" rgba="0.0 0.3 0.9 1" reflectance="0.4"/> 
        </asset>

        <worldbody>
            <light name="studio_key" pos="0 0 3" dir="0 -0.5 -1" diffuse="0.4 0.4 0.4" specular="0.3 0.3 0.3" directional="true" castshadow="true"/>

            <geom name="floor" type="plane" size="4 4 0.1" material="mat_floor"/>
            <geom name="wall_back" type="box" size="4 0.1 2.5" pos="0 -2 2.5" material="mat_wall"/>
            <geom name="wall_left" type="box" size="0.1 4 2.5" pos="-3 0 2.5" material="mat_wall"/>
            <geom name="wall_right" type="box" size="0.1 4 2.5" pos="3 0 2.5" material="mat_wall"/>
            
            <body name="table" pos="0 -0.9 0">
                <geom name="tabletop" type="box" size="0.6 0.35 0.02" pos="0 0 0.78" material="mat_table"
                    solimp="0.99 0.99 0.01" solref="0.01 1" margin="0"/>
                
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
                        mass="0.01" friction="1.5 0.5 0.1" solimp="0.99 0.99 0.01" solref="0.002 1"/>
            </body>

            <geom name="back_wall" type="box" size="2 0.05 1" pos="0 1.5 1" rgba="0.7 0.7 0.7 1"/>

            <body name="ik_target" mocap="true" pos="0 0 0">
                <geom type="sphere" size="0.04" rgba="0 1 0 0.4" contype="0" conaffinity="0"/>
            </body>

        </worldbody>

        <equality>
            <weld body1="ik_target" body2="franka_hand_rh_palm" relpose="0 0 0 1 0 0 0" solref="0.005 1" solimp="0.99 0.99 0.01"/>
        </equality>

    </mujoco>
    """

    arm_dir = os.path.dirname(panda_mj_description.MJCF_PATH)
    arm_path = os.path.join(arm_dir, "panda_nohand.xml")

    hand_dir = os.path.dirname(shadow_hand_mj_description.MJCF_PATH)
    right_hand_files = [f for f in os.listdir(hand_dir) if 'right' in f and 'scene' not in f and f.endswith('.xml')]
    hand_path = os.path.join(hand_dir, right_hand_files[0])

    scene_spec = mujoco.MjSpec.from_string(scene_xml)
    arm_spec = mujoco.MjSpec.from_file(arm_path)
    hand_spec = mujoco.MjSpec.from_file(hand_path)

    attached_frame = arm_spec.attach(hand_spec, site="attachment_site", prefix="hand_")
    attached_frame.pos = [0.0, 0.0, 0.00]
    attached_frame.quat = [0.0, 0.7071, 0.0, 0.7071]
    scene_spec.attach(arm_spec, site="robot_mount", prefix="franka_")
    
    model = scene_spec.compile()

    franka_light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "franka_top")
    if franka_light_id != -1:
        model.light_active[franka_light_id] = 0 # 0 = Off, 1 = On

    data = mujoco.MjData(model)

    # Calculate memory addresses once
    robot_qpos_adr = model.jnt_qposadr[model.joint("franka_joint1").id]
    
    # Get the memory address for the cube's free joint (which takes up 7 slots: 3 pos, 4 quat)
    cube_joint_id = model.body("cube").jntadr[0]
    cube_qpos_adr = model.jnt_qposadr[cube_joint_id]
    
    # Same for the tote, though we didn't track it in the teleop script. 
    # If it gets bumped, it won't move in the replay unless we start recording it!

    # Calculate playback speed (60Hz)
    playback_dt = 1.0 / 60.0

    print("\n--- LAUNCHING REPLAY ---")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.3
        viewer.cam.azimuth = 225
        viewer.cam.elevation = -30
        viewer.cam.lookat = [0, -0.9, 0.9]

        # Loop the trajectory infinitely
        while viewer.is_running():
            for frame in trajectory:
                if not viewer.is_running():
                    break
                
                step_start = time.time()

                # 1. Teleport the robot joints to their exact recorded angles
                data.qpos[robot_qpos_adr : robot_qpos_adr + 31] = frame["robot_qpos"]
                
                # 2. Teleport the cube to its exact recorded position and rotation
                data.qpos[cube_qpos_adr : cube_qpos_adr + 3] = frame["cube_pos"]
                data.qpos[cube_qpos_adr + 3 : cube_qpos_adr + 7] = frame["cube_quat"]
                
                # 3. Teleport the green target sphere (mocap body) to show where the user was aiming
                mocap_id = model.body("ik_target").mocapid[0]
                data.mocap_pos[mocap_id] = frame["action_target_pos"]

                # 4. Update the visual geometry (NO mj_step!)
                mujoco.mj_forward(model, data)
                viewer.sync()

                # 5. Lock the framerate to 60Hz
                time_until_next = playback_dt - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
            
            # Pause briefly at the end of the trajectory before looping
            time.sleep(1.0)

if __name__ == "__main__":
    main()
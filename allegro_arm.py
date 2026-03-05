import os
import mujoco
import mujoco.viewer
import time
import numpy as np
from robot_descriptions import panda_mj_description
from robot_descriptions import allegro_hand_mj_description

scene_xml = """
<mujoco>
    <worldbody>
        <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.9 0.8 1"/>
        
        <body name="table" pos="0 -0.7 0.4">
            <geom type="box" size="0.6 0.35 0.4" rgba="0.6 0.5 0.4 1"/>
        </body>
        
        <body name="platform" pos="0 0 0.4">
            <geom type="box" size="0.2 0.2 0.4" pos="0 0 0.0" rgba="0.3 0.3 0.3 1"/>
            
            <site name="robot_mount" pos="0 0 0.4" quat="1 0 0 -1"/>
        </body>
        
        <geom name="back_wall" type="box" size="2 0.05 1" pos="0 1.5 1" rgba="0.7 0.7 0.7 1"/>
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

franka_home = [0.0, -0.785, 0.0, -2.356, 0.0, 1.871, 0.785]

allegro_home = [0.0] * 16

default_qpos = franka_home + allegro_home

data.qpos[:] = default_qpos

data.ctrl[:] = default_qpos

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        
        mujoco.mj_step(model, data)
        
        viewer.sync()
        time.sleep(model.opt.timestep)
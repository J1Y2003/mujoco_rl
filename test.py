import os
import mujoco
from robot_descriptions import shadow_hand_mj_description

hand_dir = os.path.dirname(shadow_hand_mj_description.MJCF_PATH)
all_files = os.listdir(hand_dir)
right_hand_files = [f for f in all_files if 'right' in f and 'scene' not in f and f.endswith('.xml')]
hand_path = os.path.join(hand_dir, right_hand_files[0])
hand_spec = mujoco.MjSpec.from_file(hand_path)

model = hand_spec.compile()

print(f"{'Index':<5} | {'Actuator Name':<30} | {'Min Limit':<10} | {'Max Limit':<10}")
print("-" * 65)

# Iterating through all actuators, focusing on indices 7 and up
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    # Check if the actuator actually has a limited control range
    has_limit = model.actuator_ctrllimited[i] 
    
    if has_limit:
        ctrl_min, ctrl_max = model.actuator_ctrlrange[i]
    else:
        ctrl_min, ctrl_max = ("No Limit", "No Limit")
        
    print(f"{i:<5} | {str(name):<30} | {ctrl_min:<10} | {ctrl_max:<10}")
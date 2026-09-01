import mujoco
import mujoco.viewer
import time

xml_string = """
<mujoco>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
        <body pos="0 0 1">
        <joint type="free"/>
        <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
        </body>
    </worldbody>
    </mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml_string)

data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
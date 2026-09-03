"""
Packages the recorded teleop rollouts (rl_training_data/<task>/episode_N.json) into a single
.tar for import into an external rendering machine. The tar bundles:

  data/<task>/episode_N.json  - the recorded trajectories, verbatim
  robot/robot.xml             - a standalone MJCF for the Franka Panda + Robotiq 2F-85
                                 ("franka_robotiq_2f85"), with joint names matching the
                                 "joint_values" keys used in every recorded frame
  robot/assets/*.stl,*.obj    - the meshes robot.xml references (meshdir="assets/")
  metadata.json               - robot/object keys, table corners relative to the robot root,
                                 and the frame/coordinate conventions needed to render a rollout

All recorded episodes currently share one robot ("franka_robotiq_2f85"), one object
("alphabet_soup_can"), and one table height -- see src/robotiq_arm_teleop.py, which is the
teleop script these episodes were recorded with. Re-run this script if new task folders are
added under rl_training_data/.
"""
import glob
import json
import os
import shutil
import tarfile
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
from robot_descriptions import panda_mj_description, robotiq_2f85_mj_description

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "rl_training_data")
EXPORT_DIR = os.path.join(REPO_ROOT, "export")
BUILD_DIR = os.path.join(EXPORT_DIR, "_build")
TAR_PATH = os.path.join(EXPORT_DIR, "rollout_export.tar")

ROBOT_KEY = "franka_robotiq_2f85"


def _panda_gripper_paths():
    arm_dir = os.path.dirname(panda_mj_description.MJCF_PATH)
    arm_path = os.path.join(arm_dir, "panda_nohand.xml")
    gripper_dir = os.path.dirname(robotiq_2f85_mj_description.MJCF_PATH)
    gripper_path = os.path.join(gripper_dir, "2f85.xml")
    return arm_dir, arm_path, gripper_dir, gripper_path


def _fix_nested_anonymous_defaults(root):
    """MjSpec.to_xml() on an attach()ed spec can emit a <default> (no class attribute)
    nested inside another <default>, which MuJoCo's parser rejects ("empty class name").
    Only the outermost <default> is allowed to omit a class; splice any inner ones into
    their parent in place."""
    def fix(parent):
        for child in list(parent):
            if child.tag == "default":
                fix(child)
                if "class" not in child.attrib and parent.tag == "default":
                    idx = list(parent).index(child)
                    parent.remove(child)
                    for i, grandchild in enumerate(list(child)):
                        parent.insert(idx + i, grandchild)
            else:
                fix(child)
    fix(root)


def build_standalone_robot_xml(out_dir):
    """Attach the Robotiq 2F-85 to the Panda arm (no name prefixes, so joint names come out
    as "joint1".."joint7", "right_driver_joint", ... exactly matching the "joint_values" keys
    recorded in every episode's frames), and write it out as one MJCF plus its mesh assets."""
    arm_dir, arm_path, gripper_dir, gripper_path = _panda_gripper_paths()

    arm_spec = mujoco.MjSpec.from_file(arm_path)
    gripper_spec = mujoco.MjSpec.from_file(gripper_path)

    attached_frame = arm_spec.attach(gripper_spec, site="attachment_site", prefix="")
    attached_frame.pos = [0.0, 0.0, 0.0]
    attached_frame.quat = [1.0, 0.0, 0.0, 0.0]

    # Sanity check: this must compile before we bother writing/fixing the XML.
    arm_spec.compile()

    xml_str = arm_spec.to_xml()
    root = ET.fromstring(xml_str)
    _fix_nested_anonymous_defaults(root)

    os.makedirs(out_dir, exist_ok=True)
    assets_out = os.path.join(out_dir, "assets")
    os.makedirs(assets_out, exist_ok=True)
    for src_dir in (os.path.join(arm_dir, "assets"), os.path.join(gripper_dir, "assets")):
        for fname in os.listdir(src_dir):
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(assets_out, fname))

    robot_xml_path = os.path.join(out_dir, "robot.xml")
    ET.ElementTree(root).write(robot_xml_path)

    # Verify the exported file actually loads standalone from its own directory.
    model = mujoco.MjModel.from_xml_path(robot_xml_path)
    joint_names = [model.joint(j).name for j in range(model.njnt)]
    return robot_xml_path, joint_names


def compute_table_corners_relative_to_root():
    """Rebuilds just enough of the scene from src/robotiq_arm_teleop.py (pedestal + mount +
    table) to get the robot root's (franka_link0) world pose, then expresses the tabletop's
    four top-surface corners in that root's local frame -- the same frame every recorded
    episode's object poses are already expressed in (see object_pose_relative_to_root() in
    robotiq_arm_teleop.py)."""
    scene_xml = """
    <mujoco>
        <worldbody>
            <body name="table" pos="0 -0.9 0">
                <geom name="tabletop" type="box" size="0.6 0.35 0.02" pos="0 0 0.78"/>
            </body>
            <body name="pedestal_base" pos="0 0 0">
                <body name="platform_head" pos="0 -0.2 0.39">
                    <geom name="mount_spacer" type="box" size="0.14 0.14 0.14" pos="0 0 0.14"/>
                    <geom name="mount_plate" type="cylinder" size="0.12 0.01" pos="0 0 0.29"/>
                    <site name="robot_mount" pos="0 0 0.3" euler="0 0 -90"/>
                </body>
            </body>
        </worldbody>
    </mujoco>
    """
    _, arm_path, _, gripper_path = _panda_gripper_paths()

    scene_spec = mujoco.MjSpec.from_string(scene_xml)
    arm_spec = mujoco.MjSpec.from_file(arm_path)
    gripper_spec = mujoco.MjSpec.from_file(gripper_path)

    attached_frame = arm_spec.attach(gripper_spec, site="attachment_site", prefix="hand_")
    attached_frame.pos = [0.0, 0.0, 0.0]
    attached_frame.quat = [1.0, 0.0, 0.0, 0.0]
    scene_spec.attach(arm_spec, site="robot_mount", prefix="franka_")

    model = scene_spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    root_id = model.body("franka_link0").id
    root_pos = data.xpos[root_id].copy()
    root_mat = data.xmat[root_id].copy().reshape(3, 3)

    table_id = model.body("table").id
    table_pos = data.xpos[table_id].copy()
    table_mat = data.xmat[table_id].copy().reshape(3, 3)

    half = np.array([0.6, 0.35, 0.02])
    top_z_local = 0.78 + half[2]

    # NOTE: the mount rotates the robot root ~90deg about z relative to the table/world frame
    # (see root_mat above), so a corner's sign in the *table's* local frame does not match its
    # sign in the *root* frame -- label corners by their actual root-frame coordinates instead
    # of by (sx, sy), which would be misleading here.
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            local = np.array([sx * half[0], sy * half[1], top_z_local])
            world = table_pos + table_mat @ local
            rel = root_mat.T @ (world - root_pos)
            corners.append(rel.tolist())

    # The mount rotation is axis-aligned (a pure permutation + sign flip), so the table stays
    # an axis-aligned rectangle in the root frame too -- min/max over the corners is exact.
    corners_arr = np.array(corners)
    bounds = {
        "x": [float(corners_arr[:, 0].min()), float(corners_arr[:, 0].max())],
        "y": [float(corners_arr[:, 1].min()), float(corners_arr[:, 1].max())],
    }

    surface_center_rel = root_mat.T @ (table_pos + table_mat @ np.array([0, 0, top_z_local]) - root_pos)
    return corners, bounds, float(surface_center_rel[2])


def collect_task_dirs():
    tasks = {}
    for entry in sorted(os.listdir(DATA_DIR)):
        task_dir = os.path.join(DATA_DIR, entry)
        if not os.path.isdir(task_dir):
            continue
        episodes = sorted(glob.glob(os.path.join(task_dir, "episode_*.json")))
        if episodes:
            tasks[entry] = episodes
    return tasks


def main():
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(BUILD_DIR)

    tasks = collect_task_dirs()
    print(f"Found {sum(len(v) for v in tasks.values())} episodes across tasks: "
          f"{ {k: len(v) for k, v in tasks.items()} }")

    # 1. Copy the recorded data, preserving task/episode filenames.
    data_out = os.path.join(BUILD_DIR, "data")
    scene_infos = []
    for task, episodes in tasks.items():
        task_out = os.path.join(data_out, task)
        os.makedirs(task_out, exist_ok=True)
        for ep_path in episodes:
            shutil.copy2(ep_path, os.path.join(task_out, os.path.basename(ep_path)))
            with open(ep_path) as f:
                scene_infos.append(json.load(f)["scene"])

    robot_keys = {s["robot_key"] for s in scene_infos}
    object_key_sets = {tuple(s["object_keys"]) for s in scene_infos}
    table_zs = {s["table_surface_z"] for s in scene_infos}
    if len(robot_keys) != 1 or robot_keys != {ROBOT_KEY}:
        raise RuntimeError(f"Expected every episode to use robot_key={ROBOT_KEY!r}, found {robot_keys}")
    if len(object_key_sets) != 1:
        raise RuntimeError(f"Episodes disagree on object_keys: {object_key_sets}")
    if len(table_zs) != 1:
        raise RuntimeError(f"Episodes disagree on table_surface_z: {table_zs}")
    object_keys = list(next(iter(object_key_sets)))
    table_surface_z_recorded = next(iter(table_zs))

    # 2. Build the standalone robot MJCF + meshes.
    robot_out = os.path.join(BUILD_DIR, "robot")
    robot_xml_path, joint_names = build_standalone_robot_xml(robot_out)
    print(f"Wrote standalone robot MJCF: {robot_xml_path}")
    print(f"Robot joint order: {joint_names}")

    # 3. Table corners relative to the robot root.
    table_corners, table_bounds, table_surface_z_computed = compute_table_corners_relative_to_root()
    if abs(table_surface_z_computed - table_surface_z_recorded) > 1e-6:
        raise RuntimeError(
            f"Computed table_surface_z ({table_surface_z_computed}) doesn't match the value "
            f"recorded in the episodes ({table_surface_z_recorded}) -- scene geometry drifted "
            f"from src/robotiq_arm_teleop.py, update compute_table_corners_relative_to_root()."
        )

    # 4. Metadata for the rendering machine.
    metadata = {
        "robot_key": ROBOT_KEY,
        "robot_file": "robot/robot.xml",
        "robot_joint_order": joint_names,
        "object_keys": object_keys,
        "rotation_convention": "quat_wxyz",
        "frame_convention": (
            "All positions/rotations in the episode data (frames[i].objects.<key>.pos/rotation) "
            "and the table_corners below are expressed relative to the robot root: the pose of "
            "robot.xml's own base link ('link0') when robot.xml is loaded standalone with its "
            "root at world position [0,0,0] and identity orientation [1,0,0,0]. Every recorded "
            "frame also reports frames[i].robot.pos/rotation, which is always [0,0,0]/[1,0,0,0] "
            "since the robot base is fixed -- only frames[i].robot.joint_values (keyed by the "
            "names in robot_joint_order) moves."
        ),
        "table": {
            "surface_z": table_surface_z_recorded,
            "corners_relative_to_root": table_corners,
            "corners_note": (
                "The 4 top-surface corners, each [x, y, z] in the root frame, in no particular "
                "winding order. The table is axis-aligned in this frame (the mount rotation is "
                "a pure 90deg permutation), so x_range/y_range below are exact bounds."
            ),
            "x_range_relative_to_root": table_bounds["x"],
            "y_range_relative_to_root": table_bounds["y"],
        },
        "tasks": {task: len(eps) for task, eps in tasks.items()},
        "episode_fps": 15.0,
        "source": "src/robotiq_arm_teleop.py",
    }
    metadata_path = os.path.join(BUILD_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata: {metadata_path}")

    # 5. Tar it all up.
    os.makedirs(EXPORT_DIR, exist_ok=True)
    if os.path.exists(TAR_PATH):
        os.remove(TAR_PATH)
    with tarfile.open(TAR_PATH, "w") as tar:
        for name in ("data", "robot", "metadata.json"):
            tar.add(os.path.join(BUILD_DIR, name), arcname=name)

    shutil.rmtree(BUILD_DIR)
    size_mb = os.path.getsize(TAR_PATH) / (1024 * 1024)
    print(f"\nWrote {TAR_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

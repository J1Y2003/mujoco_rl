# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A MuJoCo simulation sandbox for teleoperating a Franka Panda arm (optionally fitted with a
Shadow Hand or a parallel gripper) to manipulate objects (a cube, a power drill) on a tabletop
scene, recording the resulting trajectories as JSON for later RL training/replay. There is no
package structure, build step, or test suite — scripts are run directly and iterated on in place.

## Environment setup

```bash
python3 -m venv mujoco_env
source mujoco_env/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is missing a few packages that scripts actually import — install these too if
you touch the affected files:
- `pynput` — used by every `*_teleop.py` script for keyboard input
- `trimesh` and `vhacdx` — used by `asdf.py` for convex decomposition

## Running scripts

There is no test suite, linter, or build. Everything is run directly with `python3`:

```bash
python3 src/shadow_arm_teleop.py       # Franka + Shadow Hand teleop, records episodes
python3 src/gripper_arm_teleop.py      # Franka + parallel gripper teleop
python3 src/shadow_arm.py              # Franka + Shadow Hand, static/no-input viewer
python3 src/intro.py                   # minimal MuJoCo viewer smoke test
python3 src/play_data.py --data_path rl_training_data/<run_dir>/episode_0.json
python3 asdf.py                        # regenerates drill_assets.xml / drill_collisions.xml
```

Teleop scripts open an interactive `mujoco.viewer` window and read live keyboard/gamepad input in
a loop — they are not scriptable/headless and can't be validated by just checking they import
cleanly. When editing one, actually run it and drive the arm/hand to confirm the change works.

## Architecture

### Scene assembly via `MjSpec`, not static XML files

Every entry point builds its scene the same way, using `mujoco.MjSpec` to compose a small inline
XML string (table, pedestal, walls, objects, camera-friendly lighting) with external robot
descriptions pulled from the `robot_descriptions` package at runtime:

```python
arm_path  = <panda_mj_description dir>/panda_nohand.xml
hand_path = <shadow_hand_mj_description dir>/<the "right", non-"scene" xml file>

scene_spec = mujoco.MjSpec.from_string(scene_xml)
arm_spec   = mujoco.MjSpec.from_file(arm_path)
hand_spec  = mujoco.MjSpec.from_file(hand_path)

attached_frame = arm_spec.attach(hand_spec, site="attachment_site", prefix="hand_")
attached_frame.pos  = [...]
attached_frame.quat = [...]        # rotates the hand into the arm's frame

scene_spec.attach(arm_spec, site="robot_mount", prefix="franka_")
model = scene_spec.compile()
```

This two-stage `attach()` (hand → arm, then arm → scene) is why body/joint names end up prefixed
`franka_` and `franka_hand_rh_...` — e.g. the end-effector body is `franka_hand_rh_palm`. When
adding a new robot combination, follow this same attach-then-compile pattern rather than hand
editing a monolithic XML file.

### IK via a mocap weld, not a solver call

There is no explicit IK solver. A mocap body (`ik_target`) is welded to the end-effector body via
an `<equality><weld .../></equality>` constraint with a soft `solref`/`solimp`. Moving
`data.mocap_pos[mocap_id]` each frame drags the arm along through the physics solver. Keyboard/
gamepad input only ever updates `target_pos`, which is clamped to `X_BOUNDS`/`Y_BOUNDS`/`Z_BOUNDS`
before being written to the mocap body.

### Actuator retuning after `compile()`

The teleop scripts disable the arm's own position actuators (`actuator_gainprm[i] = 0` for the
first 7 actuators) so the weld constraint — not the actuator — drives the arm, then boost the
hand actuators' force range/gain/bias (`STRENGTH_MULTIPLIER`) and joint damping so the fingers are
stiff enough to grip without the simulation exploding. `UNUSED_JOINTS` holds indices of
abduction/rotation joints (`ffa0`, `mfa0`, `rfa0`, `tha0`) that are pinned to 0 so the hand closes
straight instead of splaying. If you change hand pose targets (`pose_open`, `pose_grip`), keep the
per-finger index math (`for i in [10, 13, 17]: ...`) in sync with which joints those offsets
actually address in the Shadow Hand's actuator ordering.

### Recording format

`shadow_arm_teleop.py` and `gripper_arm_teleop.py` record at a fixed `RECORD_HZ` (decoupled from
the `0.001s` physics timestep via `record_interval`) into `rl_training_data/<timestamp>/episode_N.json`.
Each frame is a flat dict of NumPy arrays converted to lists:
`action_target_pos`, `action_grip_ratio`, `robot_qpos`, `robot_qvel`, `eef_pos`, `eef_quat`,
`cube_pos`, `cube_quat`. `play_data.py` reconstructs the *exact same* scene XML/attach sequence
before replaying, because qpos slice offsets (e.g. `robot_qpos_adr : robot_qpos_adr + 31`) are
only valid if the compiled model's joint ordering matches what was recorded — if you change the
scene assembly in the teleop script, mirror the change in `play_data.py` or old recordings will
desync silently (wrong joints move).

### Drill collision mesh pipeline

`obj/drill/drill.obj` is a single high-poly visual mesh with no usable convex collision geometry.
`asdf.py` runs V-HACD (`vhacdx.compute_vhacd`) to decompose it into up to 30 convex hulls, writing
each as `obj/drill/drill_col_N.obj` plus two generated MuJoCo include files at the repo root:
`drill_assets.xml` (mesh asset declarations) and `drill_collisions.xml` (invisible/`rgba="... 0"`
collision geoms). The teleop scenes `<include>` both files rather than declaring the drill's
collision geometry inline. Re-run `asdf.py` after replacing `drill.obj`, and don't hand-edit the
two generated XML files — regenerate them instead.

### Gravity compensation

`model.body_gravcomp` is set to `1.0` for every body whose name contains `franka_` but not `hand`
— this cancels gravity on the arm links (so the weld constraint only has to fight friction/inertia,
not the arm's own weight) while leaving the hand and manipulated objects subject to normal gravity.
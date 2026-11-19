# G1 Carry Box: 166.560M Real-Robot Deployment

[English](README.md) | [Chinese deployment guide](README.zh-CN.md)

This package contains the G1 29-DoF Carry Box deployment for checkpoint
**166,559,744**, its exported sphere-log actor, and its reward-inferred
`solve_close` latents. The actor accepts 1,252 values and outputs 29 actions.
It runs on a Linux NVIDIA GPU workstation connected to the G1 DDS network.
LuMo motion capture supplies the robot root and box poses; motor encoders
and the onboard IMU alone are insufficient for this policy.

## Install

From the repository root, download only this deployment's LFS model:

```bash
# For a fresh clone, skip the training data/checkpoint downloads:
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Roboparty/I-BFM-Code.git
cd I-BFM-Code
git lfs install
git lfs pull --include="deploy/g1_carry_box_166m/model/**/*.onnx" --exclude=""
cd deploy/g1_carry_box_166m
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/unitree_workstation.txt
```

The pinned GPU runtime requires a compatible NVIDIA driver and CUDA/cuDNN
libraries. CPU ONNX Runtime must not be installed alongside `onnxruntime-gpu`.
The official Unitree SDK requires CycloneDDS 0.10.2. If pip cannot locate its
native library, install that native runtime first and set `CYCLONEDDS_HOME`
to its installation prefix before running pip.
The training environment and the full training checkpoint are not needed.
Keep this directory inside the repository: the FK XML uses the existing
`ibfm/humanoidverse/data/robots/g1_mjlab/meshes/` assets.

Install the vendor-provided LuMo B002 Python protobuf module at
`external/lumo/LusterFrameStruct_pb2.py`, or set `lumo.protobuf_module` in
your site configuration to its absolute path. The vendor SDK is not bundled.

## Configure The Site

```bash
mkdir -p site
cp config/exp/carry/g1_carry_intact_logmap166559744_solve_close_gpu_lumo.yaml site/carry.yaml
export TASK_CONFIG="$PWD/site/carry.yaml"
export G1_INTERFACE=enp131s0
export G1_ORIN_HOST=192.168.123.164
export UFO_GPU_ID=0
export PYTHON_BIN="$PWD/.venv/bin/python"
```

Edit `site/carry.yaml` for the actual LuMo endpoint, rigid-body names,
world/body transforms, and target. The supplied transforms and IP addresses
are examples from the source installation. The configured box is a 0.35 m
cube, and placement success uses a 0.40 m horizontal tolerance. Preserve
the supplied policy, latents, input schema, PD gains, and calibration gate.

## Run

All commands below run from this directory. Preflight validates artifact
hashes, FK, observation dimensions, latents, and GPU inference without
opening DDS or sending motor commands:

```bash
bash run_g1_carry_intact_logmap166559744_solve_close_gpu_preflight.sh
bash run_g1_carry_intact_logmap166559744_solve_close_gpu_mocap_check.sh
```

After motion capture is working, subscribe to the robot and record inference
without creating a LowCmd publisher:

```bash
export UFO_REAL_ROBOT_OK=1
export UFO_INTACT_LOGMAP166559744_SOLVE_CLOSE_GPU_EXPERIMENTAL_OK=1
unset UFO_ALLOW_LOW_LEVEL_COMMANDS
bash run_g1_carry_intact_logmap166559744_solve_close_gpu_readonly.sh
```

For physical operation, use suspension protection and a lightweight box;
keep the area clear and have an operator ready to press R2. Follow the
[three-stage calibration procedure](README.zh-CN.md#首次标定按键顺序).
The following command opens the controller; policy execution still requires
the gamepad calibration/start sequence:

```bash
export UFO_ALLOW_LOW_LEVEL_COMMANDS=1
unset UFO_RELEASE_MOTION_SERVICE
bash run_g1_carry_intact_logmap166559744_solve_close_gpu_workstation.sh
```

| Button | Function |
| --- | --- |
| A | Toggle the default pose while policy execution is off |
| B | Capture pelvis/box-height samples, or finish contact calibration |
| X / Y | Label one no-contact/contact frame during calibration |
| R1 | Start Carry after calibration and A arming |
| L1 + R1 | Validate saved calibration and start Carry |
| R2 | Latch a safe stop |
| R1 + B | Clear the stop latch after releasing the buttons; does not start Carry |

Logs and calibration are stored under `logs/` and excluded from Git.
Saved calibration is specific to the robot, box, marker installation, and
LuMo world frame. Recalibrate when any of these change.

## Tests And Provenance

```bash
python -m unittest discover -s tests -v
python tests/test_ufo_policy_safety.py
```

For a CPU-only artifact/inference check without DDS or motion capture:

```bash
PYTHONPATH=. python tools/preflight_g1_carry.py \
  --require-runtime-mode phased \
  --require-controller-profile intact_logmap150544384_strong \
  --require-deployment-calibration
```

This CPU check does not establish GPU timing or physical task success. The
deployment launchers require an active CUDA provider and retain that check.

Source: the corresponding Carry deployment in
[Roboparty/UFO](https://github.com/Roboparty/UFO). The manifest records the
checkpoint identity, ONNX and latent checksums, and packaged source hashes.
Shared base class/profile names containing older step counts are retained
for compatibility; this package contains only the 166.560M actor and selected
Carry latents. Other task entry points and model versions are not packaged.
The repository's [license](../../LICENSE) applies.

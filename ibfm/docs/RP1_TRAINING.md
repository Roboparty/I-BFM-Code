# RP1 Training

This package organizes the RP1 official-PD training inputs used by the
249,839,616-step model shown in the README video. It does not replace the G1
166.560M checkpoint or its task scripts. The launcher starts a new RP1 policy;
the video checkpoint and inferred latents are not installed in this release.
Reproducing the exact historical weights is not guaranteed by these inputs:
the training source includes later memory optimizations and GPU numerics vary.

## Files

| File | Purpose |
| --- | --- |
| `configs/robots/rp1_28dof_official_pd.yaml` | Frozen robot training configuration from the video model |
| `configs/tasks/carry_box_rp1_full_dr.yaml` | RP1 object geometry, contact observations and full domain randomization |
| `configs/data/rp1_lafan_carry_kick_30_40_30.yaml` | Prepared LAFAN/carry/kick train and full caches, weighted 0.3/0.4/0.3 |
| `configs/data/rp1_assets.json` | Six cache SHA256 values and reference model/PD provenance |
| `humanoidverse/data/robots/rp1_3_2/` | Complete MuJoCo model and its STL meshes |
| `scripts/check_rp1_training.py` | Data installation, checksum, schema and robot checks without a GPU |
| `scripts/train_rp1.sh` | RP1 launcher with dry-run and resume handling |

Paths in this table are relative to `ibfm/`. Older `rp1_28dof.yaml` and the
70/30 data mix remain available; the new launcher explicitly selects the
official-PD configuration and 30/40/30 mix.

## KP/KD

Source: [Roboparty/robolab](https://github.com/Roboparty/robolab/blob/45cdc13a3465449315552af3dae09fa23f7083f5/robolab/assets/robots/roboparty.py),
pinned by `roboparty_train` revision `53b411a0790fe172f717df773122bbc4ce190757`.
This is the source revision used for the recorded model, not a claim about
the latest upstream defaults. Official 24-DoF gains are mapped to our 28-DoF
robot by applying the official wrist gain to all three wrist axes on each side.

| Joint group | KP | KD |
| --- | ---: | ---: |
| Waist roll/yaw, hip pitch/roll, knee | 120 | 7.5 |
| Hip yaw, shoulder pitch | 35 | 2.5 |
| Ankle pitch/roll | 25 | 1.5 |
| Shoulder roll/yaw, elbow, all wrist axes | 12.5 | 0.75 |

## Data Preparation

The six prepared caches are under
`cache/motion_data/rp1_lafan_carry_kick_contact_hyst45_60_30_40_30/`.
Training files use approximately 10-second clips; full files retain complete
motions. Carry/kick contact labels use 4.5/6.0 cm hysteresis. The lock file
records the exact six inputs, totaling about 1.1 GiB. Do not substitute G1
caches or the older RP1 70/30 raw-data manifest.

After preparing the caches, validate them from `ibfm/`:

```bash
PYTHONPATH=. .venv/bin/python scripts/check_rp1_training.py
```

On another machine, obtain the same six trusted caches, then install them:

```bash
PYTHONPATH=. .venv/bin/python scripts/check_rp1_training.py --source-dir /path/to/rp1_caches
```

Only load pickle caches from a trusted source. The checker verifies SHA256
before deserialization and checks every training motion's schema and RP1 pose
dimension. It refuses to overwrite mismatched existing destination files.
The six prepared motion caches are not bundled with this configuration update.
Obtain matching caches separately; `configs/data/rp1_assets.json` is the
manifest for verifying a copy on another machine. A clean checkout needs
these caches installed before training can start.

## Train

Linux, Python 3.10, the repository's `uv sync` environment and CUDA GPUs are
required. From the repository root, `cd ibfm && uv sync` is enough to prepare
the training environment; restoring the G1 checkpoint is not needed for RP1
training. Select eight idle GPUs on a training server:

```bash
cd ibfm
PYTHONPATH=. .venv/bin/python scripts/check_rp1_training.py
bash scripts/train_rp1.sh rp1_official_pd_250m --dry-run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_rp1.sh rp1_official_pd_250m
```

`CUDA_VISIBLE_DEVICES` selects physical devices; `--gpu-ids all` inside the
launcher uses all visible GPUs. Defaults are per-rank, except total steps:

| Setting | Value |
| --- | --- |
| Environments | 512 per GPU |
| Replay capacity | 2,560,000 transitions per GPU |
| Agent updates per trigger | 8 |
| Global environment steps | 250,000,000 |
| Latent update interval | 100 |
| Expert latent / horizon | mean / 8 |
| LOGO local / goal / residual weights | 4/7 / 3/7 / 0.005 |
| Sphere-log mode | actor |
| Seed | 4728 |
| Rolling checkpoint interval | 3,200,000 steps |
| Replay storage | auto (CUDA for CUDA training) |

The task config includes PD gain scaling 0.75-1.25, control delay 0-1 step,
torque RFI 0.1, foot friction 0.4-1.25 and object pose observation error
2 cm / 2 degrees. Nominal training box mass is 1.25 kg with 0.8-1.2 scaling.

Use a new run name. `NUM_ENVS`, `BUFFER_SIZE`, `NUM_AGENT_UPDATES`,
`NUM_ENV_STEPS` and `BUFFER_DEVICE` override defaults; changing them or the
GPU count is a new experiment. The launcher runs input checks before creating
a run. `--dry-run` prints the command and does not initialize GPUs or train.
The full command and data report are saved in `runs/RUN_NAME/`.

## CPU Replay For 48 GB GPUs

For GPUs such as the 4090 48 GB, use host RAM for online replay and expert
buffers to reduce VRAM pressure. From `ibfm/`:

```bash
BUFFER_DEVICE=cpu bash scripts/train_rp1.sh rp1_cpu_replay --dry-run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 BUFFER_DEVICE=cpu \
  bash scripts/train_rp1.sh rp1_cpu_replay
```

This passes `--buffer-device cpu` to the training CLI; model parameters,
rollout simulation and sampled training batches remain on GPU. Each rank has
its own replay/expert allocation in host RAM, so memory usage grows with GPU
count. Allow host memory for all ranks, expert data and the rest of training;
CPU replay also adds host-to-device transfer overhead. It does not move the
entire training process to CPU or guarantee a particular batch size fits.

The CLI and launcher still default to `auto`, which selects CUDA buffers
for CUDA training. Use the same buffer placement when resuming, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 BUFFER_DEVICE=cpu RESUME=1 \
  bash scripts/train_rp1.sh rp1_cpu_replay
```

The CPU-replay command has been checked through argument parsing and config
construction; a new full eight-GPU RP1 training run with this release and
CPU replay has not been performed.

## Checkpoints And Resume

Periodic rolling checkpoints and the normal-completion checkpoint include
model, optimizer and rank-local replay.
This matches the RP1 run's periodic replay behavior, and differs from the G1
launcher's final-only replay saving. Replay writes are large; reserve about
300 GiB for a run including checkpoint writes and headroom.

The CLI's `--checkpoint-buffer-final-only` enables a final save; omitting
`--disable-checkpoint-buffer` also keeps periodic replay saves enabled. Both
are intentional in this RP1 launcher.

`policies/latest` and `policies/best` retain weights/configuration separately
without optimizer or replay. Best uses mean tracking EMD over the actual
number of training motions in this RP1 mix; it does not reuse G1's 2,719-row
count or mean carry success. The release retention helper keeps latest/best,
not every historical periodic snapshot.

To resume a rolling checkpoint using the same GPU count and settings:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RESUME=1 \
  bash scripts/train_rp1.sh rp1_official_pd_250m
```

An interrupted write or missing/stale replay is not equivalent to an
uninterrupted run. The launcher does not initialize RP1 from G1 weights.

## README Video

The README links `assets/rp1/carry-box.mp4` (repository-root relative), the
44-second near-forward scene from the 2026-11-18 standing refinement. Initial
robot-box distance is 0.9 m and goal offset from the initial box is (1, 0) m.
The full video includes approach, pickup, carry, release and standing. It uses
the official-PD RP1 model at step 249,839,616 and solve-close latents, with
post-release crouch/stand action mixing disabled and latent/target filters
retained. It uses a 0.35 m / 0.7 kg box and a 0.40 m success radius.

Training retains its original task success tolerance of 0.15 m; the demonstration
settings must not be presented as training evaluation settings. Five deterministic
simulations passed after the controller refinement, which is not a statistical
success-rate estimate or hardware validation. The existing public G1 interactive
scripts do not load this RP1 model. The full RP1 inference bundle is separate
from this training-input update.

## Validation

Run the configuration and CPU replay regression checks from `ibfm/`:

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -p test_rp1_release_inputs.py
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -p test_buffer_device.py
```

The CPU replay tests cover buffer placement, aligned trajectory sampling,
expert prompt transfer, and gradient equivalence of sequential LOGO backward.
The CUDA optimizer integration test is skipped when CUDA is unavailable.

The prepared inputs were also checked locally on 2026-11-19:

- All six cache SHA256 values and both frozen configuration hashes passed.
  Total cache size: 1,171,419,979 bytes.
- Training motions: LAFAN 1,692; Carry 1,358; Kick 499; total 3,549. The policy
  retention helper uses this total rather than the G1 count of 2,719.
- RP1 MuJoCo model loading and training config construction passed. Actual
  training CLI parsing resolved all three sources with weights 0.3/0.4/0.3.
- Four RP1 release-input tests passed, covering launcher selection/replay
  flags, run-name validation, all 28 PD mappings, mesh/foot geometry references
  and cache-manifest consistency. Shell syntax and diff whitespace checks passed.
- Full video decoded as 1,320 frames at 30 fps; preview is 220 frames at 5 fps.
- No new full distributed training or hardware validation was run for this
  configuration update.

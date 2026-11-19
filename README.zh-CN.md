<h1 align="center">I-BFM: Reward-Conditioned Robust Humanoid Interaction via Unsupervised Reinforcement Learning</h1>

<p align="center">
  <a href="https://iamhardworking.github.io/I-BFM/">[项目网站]</a>
  <a href="https://arxiv.org/abs/2610.06129">[论文]</a>
  <a href="https://github.com/LeCAR-Lab/BFM-Zero">[BFM-Zero 上游代码]</a>
</p>

<p align="center"><img src="assets/roboparty-lab.svg" alt="RoboParty Lab" width="420"></p>

[English](README.md) · [简体中文](README.zh-CN.md)

本仓库提供 G1 的 H200 训练源码、选定的 **166.560M 步权重**，以及 **Carry Box（搬箱）**、**Push Box（推箱）** 和 **Kick Box（踢箱）** 的交互配置。G1 的三个任务使用同一份权重，同时提供 RP1 训练配置和仿真视频。

## 项目视频

点击预览可在项目网站观看完整视频。

| Carry Box（搬箱） | Push Box（推箱） | Kick Box（踢箱） |
| :---: | :---: | :---: |
| [![搬箱视频预览](assets/carry-box-preview.gif)](https://iamhardworking.github.io/I-BFM/#robust-carry-box) | [![推箱视频预览](assets/push-box-preview.gif)](https://iamhardworking.github.io/I-BFM/#robust-push-box) | [![踢箱视频预览](assets/kick-box-preview.gif)](https://iamhardworking.github.io/I-BFM/#robust-kick-box) |

### RP1 搬箱

[![RP1 搬箱仿真预览](assets/rp1/carry-box-preview.gif)](assets/rp1/carry-box.mp4)

[观看 RP1 搬箱完整仿真视频（44 秒）](assets/rp1/carry-box.mp4)，使用官方 KP/KD
重训的第 249,839,616 步模型及修正后的放箱起身控制。模型与场景参数见
[RP1 说明](ibfm/docs/RP1_TRAINING.md)。

## 方法概览

![I-BFM 预训练与奖励推理概览](assets/ibfm-overview.jpg)

## 仓库内容

| 路径 | 内容 |
| --- | --- |
| `ibfm/` | 训练源码和任务运行依赖 |
| `ibfm/tasks/carry_box/` | 搬箱交互脚本、控制器和 latent 向量 |
| `ibfm/tasks/push_box/` | 推箱交互脚本、控制器和 latent 向量 |
| `ibfm/tasks/kick_box/` | 踢箱交互脚本、控制器和 latent 向量 |
| `checkpoints/ibfm-166m/` | 选定的权重及其元数据 |
| [`deploy/g1_carry_box_166m/`](deploy/g1_carry_box_166m/README.zh-CN.md) | G1 搬箱真机部署、166.560M ONNX、solve-close latent、标定与手柄控制 |
| [`ibfm/docs/RP1_TRAINING.md`](ibfm/docs/RP1_TRAINING.md) | RP1 官方 PD 配置、动作缓存清单、训练命令和 CPU replay |

## 安装

需要 Linux、Python 3.10、`uv`、Git LFS 和 MuJoCo。克隆仓库、下载文件并还原权重：

```bash
git clone https://github.com/Roboparty/I-BFM-Code.git
cd I-BFM-Code
git lfs install
git lfs pull
bash restore_checkpoint.sh
cd ibfm
uv sync
```

## 训练配置

完成安装后，在 `ibfm/` 目录使用 8 张 H200 训练：

```bash
./scripts/train_ibfm_h200.sh ibfm_reproduction
```

启动脚本使用发布版的数据配比、随机种子和训练参数。`runs/ibfm_reproduction/checkpoint/` 是可恢复训练的滚动检查点；`policies/latest` 指向最新策略，`policies/best` 指向 2,719 条评估动作的平均 EMD 最低的策略。这两份策略只包含权重与配置，不包含优化器和 replay buffer。训练正常结束时，replay buffer 只写入最终滚动检查点一次。

若训练中断，可从滚动模型与优化器续训：

```bash
RESUME=1 ./scripts/train_ibfm_h200.sh ibfm_reproduction
```

因缺少同期 replay buffer，续训轨迹会与不中断的训练不同。

### 48 GB 显卡的 CPU Replay

训练 CLI 支持 `--buffer-device cpu`，将在线 replay 和 expert buffer 放入主存，
模型及训练批次仍在 GPU 上。这能降低 4090 48 GB 等显卡的显存需求，但会增加
主存占用和数据传输开销。RP1 启动脚本可通过 `BUFFER_DEVICE=cpu` 启用，见下方
命令；CLI 默认仍为 `auto`。详细说明见 [CPU replay](ibfm/docs/RP1_TRAINING.md#cpu-replay-for-48-gb-gpus)。

## RP1 训练

RP1 使用官方 KP/KD、完整域随机化，以及 LAFAN/Carry/Kick 为 30/40/30 的数据配比。
准备好六份 RP1 缓存后，在 `ibfm/` 目录执行：

```bash
PYTHONPATH=. .venv/bin/python scripts/check_rp1_training.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 BUFFER_DEVICE=cpu \
  bash scripts/train_rp1.sh rp1_official_pd_250m
```

默认 8 卡、每卡 512 个环境、总计 250M 步。数据安装、完整参数、checkpoint/replay
保存、磁盘需求及续训命令，见 [RP1 训练说明](ibfm/docs/RP1_TRAINING.md)。

## 交互任务

完成安装后，在 `ibfm/` 目录运行以下命令。脚本会自行创建目录，保存视频、轨迹和运行摘要。

### Carry Box（搬箱）

```bash
./tasks/carry_box/run_interactive.sh
```

<img src="assets/simulations/carry-box.gif" alt="搬箱仿真视频" width="270">

### Push Box（推箱）

```bash
./tasks/push_box/run_interactive.sh
```

<img src="assets/simulations/push-box.gif" alt="推箱仿真视频" width="270">

### Kick Box（踢箱）

```bash
./tasks/kick_box/run_interactive.sh
```

<img src="assets/simulations/kick-box.gif" alt="踢箱仿真视频" width="270">

## Replay buffer

[166.560M 步的 replay buffer](https://huggingface.co/datasets/HanSj77/I-BFM-166560M-replay-buffer) 已上传至 Hugging Face，包含 8 个 rank 的数据，总计约 210 GiB。在 `ibfm/` 目录下载：

```bash
uvx --from huggingface_hub hf download HanSj77/I-BFM-166560M-replay-buffer --repo-type dataset \
  --local-dir runs/ibfm_166560m_replay
```

下载后，`runs/ibfm_166560m_replay/checkpoint/buffers/train_rank_0/` 至 `train_rank_7/` 每个目录都包含 `buffer.hdf5` 和 `config.json`。现有的 `humanoidverse/agents/buffers/trajectory.py` 中的 `TrajectoryDictBufferMultiDim.load()` 可以读取单个 rank：

```bash
./.venv/bin/python - <<'PY'
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim

path = "runs/ibfm_166560m_replay/checkpoint/buffers/train_rank_0"
buffer = TrajectoryDictBufferMultiDim.load(path, device="cpu")
print(buffer.size())
PY
```

这会将选定的整个文件读入内存（单个文件约 28 GB）。该 buffer 提供用于进行 reward inference 中 latent 的选取；可在 `humanoidverse/reward_inference.py` 中通过 `--buffer-path` 指定单个 rank 的目录。

## 社区交流

欢迎添加作者微信，交流 I-BFM、LOGO、复现与人形机器人交互。扫描下方二维码添加好友。

<p align="center"><img src="assets/community/author-wechat.png" alt="作者个人微信二维码" width="360"></p>

## 许可

训练源码沿用上游的 CC BY-NC 4.0 许可，详见 [LICENSE](LICENSE)。

## 引用

如果本项目对你的研究有帮助，请引用 [I-BFM 论文](https://arxiv.org/abs/2610.06129)：

```bibtex
@misc{han2026ibfm,
  title         = {{I-BFM}: Reward-Conditioned Robust Humanoid Interaction via Unsupervised Reinforcement Learning},
  author        = {Ziqi Han and Yitang Li and Junhan Sun and Fanrong Dong and Yaojie Shen and Lei Ye and Zetong Jing and Yongqi Zhang and Yiming Zhang and Xue Wang and Hao Zhao},
  year          = {2026},
  eprint        = {2610.06129},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2610.06129}
}
```

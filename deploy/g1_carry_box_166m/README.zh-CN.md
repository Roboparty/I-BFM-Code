# G1 Carry 166.560M solve_close GPU 真机部署

[English](README.md) | [简体中文](README.zh-CN.md)

该部署使用 Scheme4 `166,559,744` checkpoint 和
`166.560M-reward_inference-solve_close` latent。它复用原部署的真机观测、LuMo、
标定和安全控制实现，包含这一版 Carry Box 的 ONNX、latent、控制器和手柄入口。
需要 Linux x86_64 GPU 工作站连接 G1 DDS 网络，以及 LuMo 动捕提供机器人根部和
箱子的世界位姿；仅电机编码器与 IMU 不能满足该策略的观测输入。

## 安装和现场配置

从仓库根目录下载本部署的 ONNX 即可，不需要恢复完整训练权重或安装训练环境：

```bash
# 首次克隆时跳过训练数据和完整 checkpoint 的下载：
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Roboparty/I-BFM-Code.git
cd I-BFM-Code
git lfs install
git lfs pull --include="deploy/g1_carry_box_166m/model/**/*.onnx" --exclude=""
cd deploy/g1_carry_box_166m
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/unitree_workstation.txt
```

需要与 ONNX Runtime GPU 版兼容的 NVIDIA 驱动、CUDA/cuDNN 库。不要同时安装
CPU 版 `onnxruntime`。官方 Unitree SDK 依赖 CycloneDDS 0.10.2；若 pip 找不到其
原生库，请先安装对应原生运行库，并在 pip 安装前将 `CYCLONEDDS_HOME` 指向其安装目录。
FK XML 复用仓库已有的 `ibfm/humanoidverse/data/robots/g1_mjlab/meshes/`
资源，因此请保持目录位于当前仓库内。LuMo 厂商 SDK 不随包发布，请将 B002 版本的
`LusterFrameStruct_pb2.py` 放到 `external/lumo/`，或在现场配置中指定其绝对路径。

```bash
mkdir -p site
cp config/exp/carry/g1_carry_intact_logmap166559744_solve_close_gpu_lumo.yaml site/carry.yaml
export TASK_CONFIG="$PWD/site/carry.yaml"
export PYTHON_BIN="$PWD/.venv/bin/python"
```

修改 `site/carry.yaml` 中的 LuMo 地址、刚体名称、世界/刚体坐标变换和目标点。
提供的地址和变换来自原现场，必须核对本机安装。保留策略输入、PD、latent 和三阶段
标定配置。`site/`、`external/`、运行日志和现场标定结果均不上传到 Git。

共享基类及 `phase_controller_profile` 中的旧步数是兼容名称，不代表加载旧模型。
本包只包含 166.560M actor 和对应的搬箱 latent。

## 固化内容

- checkpoint/actor：166,559,744 step 的 sphere-log/direct8 actor，ONNX 输入仍为
  `state[98] + last_action[29] + history_actor[372] + privileged_state[497] + latent[256]`
  共 1252 维；
- lift：solve-close 选中的 replay `topk2000 mid-face z_lift`；
- transport：使用 166.560M 专属 transport/turn 与闭环筛选 pivot；
- close alignment：箱子距目标不大于 `0.80 m` 且目标方位绝对值不小于 `30°`
  连续 3 帧后进入；`0.20 s` 混入 pivot latent；
- pivot：左右各 4 个候选，0.5 s 闭环检查转向方向、幅度和平移，停滞或箱高/
  upright 越界时切换候选；单次最多 5 s；
- place：方位绝对值不大于 `15°` 连续 5 帧或已进入 `0.40 m` 且速度不大于
  `0.15 m/s` 后退出 pivot，再允许进入放置；
- lower：使用闭环验证后的 166.560M low-height direct latent，替代会在约
  `0.6 m` 卡住的旧 blend；
- 目标点：LuMo 世界坐标 `(1.5, 0, 0)`；箱体尺寸 `0.35 m × 0.35 m × 0.35 m`。

## LuMo 与接触估计

读取和使用方式与 strong 完全一致：

- 地址：`tcp://10.43.16.129:6868`；
- 刚体：`G1_WAIST`、`CARRY_BOX`、`LEFT_HAND`、`RIGHT_HAND`；
- 使用相同世界/刚体标定变换、质量等级检查、位置差分速度和 `0.35` 速度滤波；
- 左右手只使用手部刚体原点；某只手丢点、未跟踪或距 FK 超过 `0.20 m` 时，
  该手单独回退到 FK capsule；
- 强制保留 pelvis、箱高和手箱接触三阶段标定。

动捕尚未调通时，只运行离线 preflight。不要启动只读或带电入口。

## 运行流程

以下命令都在工作站执行。

### 1. 离线预检（无 DDS、无 LowCmd）

```bash
# 在 deploy/g1_carry_box_166m 目录执行

unset UFO_ALLOW_LOW_LEVEL_COMMANDS
G1_INTERFACE=enp131s0 \
G1_ORIN_HOST=192.168.123.164 \
UFO_GPU_ID=0 \
./run_g1_carry_intact_logmap166559744_solve_close_gpu_preflight.sh
```

必须看到 solve-close controller、CUDA provider 和最终 `PASS`。

### 2. 动捕调通后检查 LuMo（不连接机器人 DDS）

```bash
# 在 deploy/g1_carry_box_166m 目录执行

MOCAP_CHECK_SECONDS=10 \
./run_g1_carry_intact_logmap166559744_solve_close_gpu_mocap_check.sh
```

确认四个刚体名称正确、持续 tracked、位置/姿态有限且没有 stale/rejected。

### 3. 只读联机

只订阅机器人与动捕、执行 GPU 推理和记录，不创建 LowCmd publisher：

```bash
# 在 deploy/g1_carry_box_166m 目录执行

export UFO_REAL_ROBOT_OK=1
export UFO_INTACT_LOGMAP166559744_SOLVE_CLOSE_GPU_EXPERIMENTAL_OK=1
unset UFO_ALLOW_LOW_LEVEL_COMMANDS

G1_INTERFACE=enp131s0 \
G1_ORIN_HOST=192.168.123.164 \
UFO_GPU_ID=0 \
./run_g1_carry_intact_logmap166559744_solve_close_gpu_readonly.sh
```

运行 30–60 s 后 `Ctrl+C`。确认日志中 active provider 包含 CUDA、LuMo 无 stale，
并检查控制周期 P95 小于 20 ms。

### 4. 带电部署

首次必须吊装保护、使用轻质箱、人员持有 R2，且清空机器人周围区域：

```bash
# 在 deploy/g1_carry_box_166m 目录执行

export UFO_REAL_ROBOT_OK=1
export UFO_ALLOW_LOW_LEVEL_COMMANDS=1
export UFO_INTACT_LOGMAP166559744_SOLVE_CLOSE_GPU_EXPERIMENTAL_OK=1
unset UFO_RELEASE_MOTION_SERVICE

G1_INTERFACE=enp131s0 \
G1_ORIN_HOST=192.168.123.164 \
UFO_GPU_ID=0 \
./run_g1_carry_intact_logmap166559744_solve_close_gpu_workstation.sh
```

首轮不要启用 `UFO_RELEASE_MOTION_SERVICE`。进程内会先做 20 次 GPU warm-up，
输出被检查后丢弃，不会下发给机器人。

## 首次标定按键顺序

进程刚启动时策略关闭。`A` 只负责打开/释放默认姿态，绝不会开始采样；pelvis
和 box-height 都由 `B` 显式启动连续 50 Hz 采样。X/Y 每次只标注当前一帧。

1. **Pelvis**：机器人双脚落地直立，按 `A` 进入默认姿态，等待约 10 s 稳定；
   按一次 `B`，自动采集至少 50 个有效帧并进入 box-height 阶段。
2. **箱高**：保持默认姿态，把 0.35 m 箱体直立、静止放在地面；按一次 `B`，
   自动采集至少 50 个有效帧并进入 contact 阶段。
3. **释放手臂**：确认机器人可靠吊装后按一次 `A`，关闭默认姿态。此时策略仍
   关闭，关节跟随当前位置并保留阻尼，可缓慢人工移动手臂。
4. **未接触**：双手明确未接触箱体；改变若干位置，每个位置按一次 `X`。
   每次 X 只记录紧接着的一个有效控制帧。
5. **接触**：双手都明确接触箱体；改变接触位置，每个位置按一次 `Y`。
   每次 Y 也只记录紧接着的一帧。不要用一手接触、一手悬空的状态标 Y。
6. 每只手的接触/未接触各至少需要 25 个有效样本；丢点时该手本次不计数。
   样本足够后按一次 `B` 完成 contact 标定并保存。
7. 停止人工操作，人员撤离；按 `A` 重新进入默认姿态并等待稳定，再按 `R1`
   启动策略。R1 会从 approach 开始。
8. 任意异常立即按 `R2`，进入锁存停止。松开 R1/B 后按 `R1+B` 只解除锁存，
   不会自动启动策略。

## 复用 solve-close 已有标定

本版本首次成功标定后保存到：

```text
logs/g1_carry_intact_logmap166559744_solve_close_gpu/latest_calibration.json
```

下次若刚体安装、LuMo 世界坐标、箱体和地面均未改变：

- 普通站立启动：建议先按 `A` 等待稳定，然后同时按 `L1+R1`；
- 吊装倒地恢复测试：可以不按 A，直接按 `L1+R1`；该组合会校验保存的标定并
  立即启动策略，但不会绕过 R2 锁存或标定文件完整性检查。

166.560M 使用独立标定文件。即使设备地址和机器人 IP 相同，首次仍需完成三阶段标定；只有随后确认
刚体安装、LuMo 世界坐标、箱体和地面均未改变时才使用 `L1+R1` 复用本版本标定。

## 运行期常用按键

| 按键 | 作用 |
|---|---|
| `A` | 策略关闭时切换默认姿态；标定完成后用于普通站立武装 |
| `B` | pelvis/箱高阶段开始 50 Hz 采集；contact 阶段提交标定 |
| `X` | contact 标定中记录当前一帧“未接触” |
| `Y` | contact 标定中记录当前一帧“接触” |
| `R1` | 标定完成且 A 已武装后启动 Carry |
| `L1+R1` | 校验并复用 solve-close 保存标定，随后直接启动 |
| `R2` | 立即进入锁存安全停止 |
| `R1+B` | R2 后解除锁存，不自动启动 |

数据独立写入：

```text
logs/g1_carry_intact_logmap166559744_solve_close_gpu/<timestamp>/
```

仍记录 actor 输出、1252D 输入、state/history/privileged-state/latent、关节状态、
力矩信息、动捕、接触估计、phase/mode、目标关节位置和控制周期耗时。solve-close
还在 metrics 中记录 bearing、pivot 候选编号、闭环 yaw 累积量和退出原因。

## 离线测试

```bash
python -m unittest discover -s tests -v
python tests/test_ufo_policy_safety.py
```

无 GPU 时，可单独检查文件哈希、FK、latent 和 CPU ONNX 推理，不连接机器人或动捕：

```bash
PYTHONPATH=. python tools/preflight_g1_carry.py \
  --require-runtime-mode phased \
  --require-controller-profile intact_logmap150544384_strong \
  --require-deployment-calibration
```

CPU 检查不能证明 GPU 控制周期或真机搬运成功。正式启动脚本仍强制验证 CUDA provider。

源码来自 [Roboparty/UFO](https://github.com/Roboparty/UFO) 中对应的 G1 Carry 部署。
manifest 固定 checkpoint 身份、ONNX/latent 校验和，以及打包后的源码哈希；
本目录遵循仓库根目录的 [LICENSE](../../LICENSE)。

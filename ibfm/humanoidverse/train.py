"""UFO training entrypoint.

UFO provides FB and TeCH unsupervised RL presets for humanoid control.
Defaults are kept in this file; command-line arguments can override them.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def _ensure_compile_cache(cache_root: str | Path | None = None) -> None:
    cache_dir = os.environ.get("UFO_CACHE_DIR") or os.environ.get("BFMZERO_MJLAB_CACHE_DIR")
    root = Path(cache_dir or cache_root or Path.cwd() / "cache").expanduser()
    for key, subdir in {
        "TMPDIR": "tmp",
        "TEMP": "tmp",
        "TMP": "tmp",
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
        "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda",
        "WARP_CACHE_PATH": "warp",
    }.items():
        path = root / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(path))


_ensure_compile_cache()

DEFAULT_AGENT = "fb"
DEFAULT_NUM_ENVS = 1024
DEFAULT_NUM_ENV_STEPS = 192000000
DEFAULT_CHECKPOINT_EVERY_STEPS = 3200000
DEFAULT_DATA_PATH = "humanoidverse/data/lafan_29dof_10s-clipped.pkl"
DEFAULT_WORK_DIR = "runs/ufo"
DEFAULT_BUFFER_SIZE = 5120000
DEFAULT_FB_UPDATE_Z_EVERY_STEP = 100
DEFAULT_TECH_UPDATE_Z_EVERY_STEP = 10
DEFAULT_UPDATE_Z_EVERY_STEP = DEFAULT_FB_UPDATE_Z_EVERY_STEP
DEFAULT_WANDB_PROJECT = "ufo-humanoid"
DEFAULT_ROBOT_CONFIG = "configs/robots/g1_29dof.yaml"
DEFAULT_CARRY_CONFIG = "configs/tasks/carry_box.yaml"

AGENT_ALIASES = {
    "fb": "fb",
    "tech": "tech",
    "tldr": "tech",
}

from humanoidverse.agents.envs.humanoidverse_mjlab import HumanoidVerseMjlabConfig
from humanoidverse.agents.evaluations.humanoidverse_mjlab import HumanoidVerseMjlabTrackingEvaluationConfig
from humanoidverse.agents.presets import build_agent_preset
from humanoidverse.training.workspace import TrainConfig
from humanoidverse.utils.motion_data import prepare_motion_manifest
from humanoidverse.utils.robot_spec import assert_robot_configs_compatible, load_robot_training_spec, resolve_robot_config_path


def _resolve_training_robot_config(
    cli_robot_config: str | Path | None,
    manifest_robot_config: str | Path | None,
) -> Path:
    if cli_robot_config is not None and manifest_robot_config is not None:
        return assert_robot_configs_compatible(cli_robot_config, manifest_robot_config)
    if cli_robot_config is not None:
        return resolve_robot_config_path(cli_robot_config)
    if manifest_robot_config is not None:
        return resolve_robot_config_path(manifest_robot_config)
    return resolve_robot_config_path(DEFAULT_ROBOT_CONFIG)


def canonical_agent_name(agent: str) -> str:
    try:
        return AGENT_ALIASES[agent]
    except KeyError as exc:
        supported = ", ".join(sorted(AGENT_ALIASES))
        raise ValueError(f"Unsupported agent preset: {agent}. Supported presets: {supported}") from exc


def _default_update_z_every_step(agent: str) -> int:
    canonical = canonical_agent_name(agent)
    return DEFAULT_TECH_UPDATE_Z_EVERY_STEP if canonical == "tech" else DEFAULT_FB_UPDATE_Z_EVERY_STEP


def build_ufo_mjlab_config(
    *,
    device: str,
    work_dir: str,
    num_envs: int,
    num_env_steps: int,
    seed: int,
    use_wandb: bool,
    wandb_run_name: str | None,
    checkpoint_every_steps: int = 9600000,
    checkpoint_buffer: bool = True,
    checkpoint_buffer_final_only: bool = False,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
    disable_eval_prioritization: bool = False,
    smoke: bool = False,
    agent: str = DEFAULT_AGENT,
    data_path: str | list[str] | None = None,
    data_mix_weights: list[float] | None = None,
    update_z_every_step: int | None = None,
    expert_z_mode: str = "mean",
    expert_z_horizon: int = 8,
    intact_native_goal_coef: float = 0.0,
    intact_native_local_coef: float = 0.0,
    intact_native_residual_coef: float = 0.0,
    intact_goal_horizon: int = 8,
    intact_grad_diagnostic_samples: int = 0,
    sphere_log_mode: str = "off",
    carry_aux_terms: bool = False,
    task_bn_normalizer: bool = False,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    lr_scale: float = 1.0,
    clip_grad_norm: float = 0.0,
    cartwheel_aux_safe: bool = False,
    num_agent_updates: int | None = None,
    robot_config: str | Path | None = None,
    task: str = "motion",
    carry_config: str | Path = DEFAULT_CARRY_CONFIG,
    init_model_checkpoint: str | Path | None = None,
    disable_compile: bool = False,
) -> TrainConfig:
    agent = canonical_agent_name(agent)
    robot_training = load_robot_training_spec(robot_config or DEFAULT_ROBOT_CONFIG)
    try:
        raw_robot_config = OmegaConf.to_container(OmegaConf.load(robot_training.config_path), resolve=True)
        metadata = raw_robot_config.get("metadata") if isinstance(raw_robot_config, dict) else None
        if isinstance(metadata, dict) and metadata.get("review_status") == "draft":
            print(
                "WARNING: Robot config is auto-generated draft. Review semantics, default pose, PD gains, "
                "actuator parameters, contact bodies, and reward/termination-related fields before formal training.",
                flush=True,
            )
    except Exception as exc:
        print(f"WARNING: Could not inspect robot config metadata for draft status: {exc}", flush=True)
    evaluations = []
    run_eval_and_prioritization = not smoke and not disable_eval_prioritization
    distributed_sync = distributed_world_size > 1
    if run_eval_and_prioritization:
        evaluations = [
            HumanoidVerseMjlabTrackingEvaluationConfig(
                name="HumanoidVerseMjlabTrackingEvaluationConfig",
                generate_videos=False,
                videos_dir="videos",
                video_name_prefix="unknown_agent",
                name_in_logs="humanoidverse_tracking_eval",
                env=None,
                num_envs=num_envs,
                n_episodes_per_motion=1,
            )
        ]
    agent_device = "cuda" if device.startswith("cuda") else "cpu"
    resolved_update_z_every_step = (
        _default_update_z_every_step(agent) if update_z_every_step is None else int(update_z_every_step)
    )
    selected = build_agent_preset(
        agent=agent,
        device=agent_device,
        compile=not distributed_sync and not smoke and not disable_compile,
        update_z_every_step=resolved_update_z_every_step,
        expert_z_mode=expert_z_mode,
        expert_z_horizon=expert_z_horizon,
        intact_native_goal_coef=intact_native_goal_coef,
        intact_native_local_coef=intact_native_local_coef,
        intact_native_residual_coef=intact_native_residual_coef,
        intact_goal_horizon=intact_goal_horizon,
        intact_grad_diagnostic_samples=intact_grad_diagnostic_samples,
        sphere_log_mode=sphere_log_mode,
        carry_aux_terms=carry_aux_terms,
        task_bn_normalizer=task_bn_normalizer,
        lr_scale=lr_scale,
        clip_grad_norm=clip_grad_norm,
        cartwheel_aux_safe=cartwheel_aux_safe,
        carry_box=task == "carry_box",
        wandb_project=DEFAULT_WANDB_PROJECT,
    )
    agent_cfg = selected["agent_cfg"]
    wandb_group = selected["wandb_group"]
    wandb_project = selected["wandb_project"]
    train_runtime = dict(selected["train_runtime"])
    train_runtime["checkpoint_buffer"] = bool(checkpoint_buffer)
    train_runtime["checkpoint_buffer_final_only"] = bool(checkpoint_buffer_final_only)
    if smoke:
        # Exercise an optimizer update instead of ending inside replay seeding.
        train_runtime["num_seed_steps"] = 0
        train_runtime["update_agent_every"] = min(int(train_runtime["update_agent_every"]), 1024)
        train_runtime["num_agent_updates"] = 1
    if num_agent_updates is not None:
        if num_agent_updates <= 0:
            raise ValueError("num_agent_updates must be positive")
        train_runtime["num_agent_updates"] = int(num_agent_updates)
    hydra_overrides = [
        f"robot={robot_training.hydra_robot}",
        f"robot.control.action_scale={robot_training.action_scale}",
        f"robot.control.action_clip_value={robot_training.action_clip_value}",
        f"robot.control.normalize_action_to={robot_training.normalize_action_to}",
        *robot_training.hydra_overrides,
    ]
    if task == "carry_box":
        # Carry samples are excluded from lie-down resets in the environment, while
        # object-free samples retain the original LAFAN recovery curriculum.
        hydra_overrides.append("domain_rand.push_robots=False")
    if cartwheel_aux_safe:
        hydra_overrides.extend(
            [
                "rewards.reward_scales.penalty_undesired_contact=0.0",
                "rewards.reward_scales.penalty_feet_ori=0.0",
                "rewards.reward_scales.feet_heading_alignment=0.0",
                "rewards.reward_scales.penalty_slippage=0.0",
                "rewards.reward_scales.penalty_ankle_roll=0.0",
                "rewards.reward_scales.penalty_action_rate=-0.1",
            ]
        )

    return TrainConfig(
        name="TrainConfig",
        agent=agent_cfg,
        init_model_checkpoint=None if init_model_checkpoint is None else str(init_model_checkpoint),
        motions="",
        motions_root="",
        env=HumanoidVerseMjlabConfig(
            name="humanoidverse_mjlab",
            device=device,
            lafan_tail_path=data_path or DEFAULT_DATA_PATH,
            data_mix_weights=data_mix_weights,
            mjcf_path=robot_training.robot.xml_path,
            robot_config_path=str(robot_training.config_path),
            robot_training=robot_training.to_env_dict(),
            max_episode_length_s=None,
            disable_obs_noise=disable_obs_noise,
            disable_domain_randomization=disable_dr,
            relative_config_path="exp/bfm_zero/bfm_zero",
            include_last_action=True,
            hydra_overrides=hydra_overrides,
            context_length=None,
            include_history_actor=True,
            include_history_noaction=False,
            root_height_obs=True,
            auto_reset=False,
            seed=seed,
            carry_box=(
                OmegaConf.to_container(OmegaConf.load(Path(carry_config).expanduser().resolve()), resolve=True)
                if task == "carry_box"
                else None
            ),
        ),
        work_dir=work_dir,
        seed=seed,
        online_parallel_envs=num_envs,
        log_every_updates=train_runtime["log_every_updates"],
        num_env_steps=num_env_steps,
        update_agent_every=train_runtime["update_agent_every"],
        num_seed_steps=train_runtime["num_seed_steps"],
        num_agent_updates=train_runtime["num_agent_updates"],
        checkpoint_every_steps=checkpoint_every_steps,
        checkpoint_buffer=train_runtime["checkpoint_buffer"],
        checkpoint_buffer_final_only=train_runtime["checkpoint_buffer_final_only"],
        prioritization=run_eval_and_prioritization,
        prioritization_min_val=0.5,
        prioritization_max_val=2.0,
        prioritization_scale=2.0,
        prioritization_mode="exp",
        use_trajectory_buffer=train_runtime["use_trajectory_buffer"],
        buffer_size=int(buffer_size),
        use_wandb=use_wandb,
        wandb_ename=os.environ.get("WANDB_ENTITY"),
        wandb_gname=wandb_group,
        wandb_pname=wandb_project,
        wandb_run_name=wandb_run_name or f"ufo_{agent}",
        load_expert_data_from_motion_lib=True,
        buffer_device="cuda" if device.startswith("cuda") else "cpu",
        disable_tqdm=True,
        evaluations=evaluations,
        eval_every_steps=train_runtime["eval_every_steps"],
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
        rank0_only_writes=True,
        checkpoint_rank_buffers=True,
        distributed_sync=distributed_sync,
        distributed_global_steps=True,
        distributed_average_metrics=True,
        tags={"backend": "mjlab", "agent": agent, "distributed_rank": distributed_rank, "distributed_world_size": distributed_world_size},
    )


def _select_device_and_rank(seed: int) -> tuple[str, int, int, int]:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        try:
            import torch

            if torch.cuda.is_available():
                os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
                return "cuda:0", 0, 0, 1
        except Exception:
            pass
        return "cpu", 0, 0, 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    return f"cuda:{local_rank}", local_rank, rank, world_size


def _parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for item in value.strip().split(","):
        if not item:
            continue
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(item))
    return cpus


def _pin_process_to_gpu_numa(local_rank: int) -> None:
    if os.environ.get("UFO_AUTO_NUMA_AFFINITY", "0") != "1" or not hasattr(os, "sched_setaffinity"):
        return
    try:
        visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
        gpu_token = visible[local_rank] if visible else str(local_rank)
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        gpu_rows = [tuple(field.strip() for field in row.split(",")) for row in query.stdout.splitlines()]
        match = next((row for row in gpu_rows if gpu_token in row[:2]), None)
        if match is None:
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES token {gpu_token!r} was not reported by nvidia-smi")
        domain, bus, device = match[2].lower().split(":")
        pci_bus_id = f"{domain[-4:]}:{bus}:{device}"
        local_cpulist = Path(f"/sys/bus/pci/devices/{pci_bus_id}/local_cpulist").read_text().strip()
        target_cpus = _parse_cpu_list(local_cpulist) & set(os.sched_getaffinity(0))
        if not target_cpus:
            raise RuntimeError(f"GPU {gpu_token} has no CPUs in the process's allowed affinity")
        os.sched_setaffinity(0, target_cpus)
        print(
            f"[INFO] NUMA affinity: local_rank={local_rank}, gpu={gpu_token}, "
            f"pci_bus_id={pci_bus_id}, cpus={local_cpulist}",
            flush=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to pin local rank {local_rank} to its GPU NUMA node: {exc}") from exc


def _init_distributed(local_rank: int, world_size: int) -> None:
    if world_size <= 1:
        return
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        init_kwargs = {
            "backend": "nccl",
            "init_method": "env://",
            "timeout": timedelta(hours=2),
        }
        try:
            init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
            dist.init_process_group(**init_kwargs)
        except TypeError:
            init_kwargs.pop("device_id", None)
            dist.init_process_group(**init_kwargs)


def run_train(args: argparse.Namespace, log_dir: Path) -> None:
    device, _local_rank, rank, world_size = _select_device_and_rank(args.seed)
    _pin_process_to_gpu_numa(_local_rank)
    _init_distributed(_local_rank, world_size)
    seed = args.seed + rank
    cfg = build_ufo_mjlab_config(
        device=device,
        work_dir=str(log_dir),
        num_envs=args.num_envs,
        num_env_steps=args.num_env_steps,
        seed=seed,
        use_wandb=bool(args.use_wandb and rank == 0),
        wandb_run_name=args.wandb_run_name,
        checkpoint_every_steps=args.checkpoint_every_steps,
        checkpoint_buffer=not args.disable_checkpoint_buffer,
        checkpoint_buffer_final_only=bool(args.checkpoint_buffer_final_only),
        distributed_rank=rank,
        distributed_world_size=world_size,
        disable_eval_prioritization=bool(args.disable_eval_prioritization),
        smoke=bool(args.smoke),
        agent=args.agent,
        data_path=args.data_path,
        data_mix_weights=args.data_mix_weights,
        update_z_every_step=args.update_z_every_step,
        expert_z_mode=args.expert_z_mode,
        expert_z_horizon=args.expert_z_horizon,
        intact_native_goal_coef=args.intact_native_goal_coef,
        intact_native_local_coef=args.intact_native_local_coef,
        intact_native_residual_coef=args.intact_native_residual_coef,
        intact_goal_horizon=args.intact_goal_horizon,
        intact_grad_diagnostic_samples=args.intact_grad_diagnostic_samples,
        sphere_log_mode=args.sphere_log_mode,
        carry_aux_terms=bool(args.carry_aux_terms),
        task_bn_normalizer=bool(args.task_bn_normalizer),
        buffer_size=args.buffer_size,
        disable_dr=bool(args.disable_dr),
        disable_obs_noise=bool(args.disable_obs_noise),
        lr_scale=args.lr_scale,
        clip_grad_norm=args.clip_grad_norm,
        cartwheel_aux_safe=bool(args.cartwheel_aux_safe),
        num_agent_updates=args.num_agent_updates,
        robot_config=args.robot_config,
        task=args.task,
        carry_config=args.carry_config,
        init_model_checkpoint=args.init_model_checkpoint,
        disable_compile=bool(args.disable_compile),
    )
    print(
        "[INFO] UFO train: "
        f"agent={args.agent}, task={args.task}, device={device}, rank={rank}/{world_size}, seed={seed}, work_dir={log_dir}, "
        f"robot_config={cfg.env.robot_config_path}, mjcf_path={cfg.env.mjcf_path}, "
        f"data_path={cfg.env.lafan_tail_path}, data_mix_weights={cfg.env.data_mix_weights}, "
        f"num_envs_per_rank={args.num_envs}, global_parallel_envs={args.num_envs * world_size}, "
        f"num_env_steps_global={args.num_env_steps}, buffer_size_per_rank={cfg.buffer_size}, "
        f"num_agent_updates={cfg.num_agent_updates}, update_agent_every_local={cfg.update_agent_every}, "
        f"expert_z={args.expert_z_mode}{args.expert_z_horizon}, "
        f"intact_native_goal=direct{args.intact_goal_horizon}:{args.intact_native_goal_coef}, "
        f"intact_native_local=direct1:{args.intact_native_local_coef}, "
        f"intact_native_residual={args.intact_native_residual_coef}, "
        f"sphere_log_mode={args.sphere_log_mode}, "
        f"carry_aux_terms={args.carry_aux_terms}, task_bn={args.task_bn_normalizer}, "
        f"cartwheel_aux_safe={args.cartwheel_aux_safe}, lr_scale={args.lr_scale}, clip_grad_norm={args.clip_grad_norm}, "
        f"disable_dr={cfg.env.disable_domain_randomization}, disable_obs_noise={cfg.env.disable_obs_noise}, "
        f"init_model_checkpoint={cfg.init_model_checkpoint}, compile={cfg.agent.compile}",
        flush=True,
    )
    try:
        workspace = cfg.build()
        workspace.train()
    finally:
        if world_size > 1:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()


def launch(args: argparse.Namespace) -> None:
    log_dir = Path(args.work_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    _ensure_compile_cache()
    if args.gpu_ids in (None, "single"):
        run_train(args, log_dir)
        return

    existing_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.gpu_ids == "all":
        import torch

        num_gpus = torch.cuda.device_count()
        selected_gpus = None
    else:
        requested = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
        if existing_visible:
            visible = [x.strip() for x in existing_visible.split(",") if x.strip()]
            selected_gpus = [visible[i] for i in requested]
        else:
            selected_gpus = [str(i) for i in requested]
        num_gpus = len(selected_gpus)
    if num_gpus <= 1:
        if selected_gpus is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)
        run_train(args, log_dir)
        return

    import torchrunx

    logging.basicConfig(level=logging.INFO)
    if selected_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)
    os.environ.setdefault("TORCHRUNX_LOG_DIR", str(log_dir / "torchrunx"))
    # Keep MJLab/CUDA imports out of the worker bootstrap so each rank can
    # isolate its physical GPU before the training stack creates a context.
    from humanoidverse.train_worker import run_train_worker

    torchrunx.Launcher(
        hostnames=["localhost"],
        workers_per_host=num_gpus,
        backend=None,
        copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY
        + (
            "MUJOCO*",
            "UFO_CACHE_DIR",
            "BFMZERO_MJLAB_CACHE_DIR",
            "UV_CACHE_DIR",
            "PYTHONPYCACHEPREFIX",
            "TMPDIR",
            "TEMP",
            "TMP",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "CUDA_CACHE_PATH",
            "WARP_CACHE_PATH",
            "UFO_AUTO_NUMA_AFFINITY",
        ),
    ).run(run_train_worker, args, log_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UFO.")
    parser.add_argument(
        "--agent",
        default=DEFAULT_AGENT,
        choices=["fb", "tech", "tldr"],
        help="Training agent preset: fb or tech. tldr is a deprecated alias for tech.",
    )
    parser.add_argument("--task", choices=["motion", "carry_box"], default="motion")
    parser.add_argument("--carry-config", type=Path, default=Path(DEFAULT_CARRY_CONFIG))
    parser.add_argument(
        "--init-model-checkpoint",
        type=Path,
        default=None,
        help="Initialize a new run from model weights in a compatible UFO run/checkpoint; optimizer and replay start fresh.",
    )
    parser.add_argument("--gpu-ids", default="single", help="'single', 'all', or a comma-separated GPU id list relative to CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=None,
        help=(
            "Robot YAML used for training metadata. Defaults to configs/robots/g1_29dof.yaml. "
            "If omitted and --data-manifest declares robot_config, the manifest robot config is used."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--num-env-steps", type=int, default=DEFAULT_NUM_ENV_STEPS)
    parser.add_argument("--checkpoint-every-steps", type=int, default=DEFAULT_CHECKPOINT_EVERY_STEPS)
    parser.add_argument(
        "--disable-checkpoint-buffer",
        action="store_true",
        help="Do not write replay-buffer shards during periodic checkpoints; model and optimizer checkpoints are still saved.",
    )
    parser.add_argument(
        "--checkpoint-buffer-final-only",
        action="store_true",
        help="Save replay-buffer shards only in the final checkpoint; use with --disable-checkpoint-buffer.",
    )
    parser.add_argument(
        "--data-path",
        nargs="+",
        default=None,
        help="One or more motion data pickle files. Multiple files require --data-mix-weights to fix source ratios.",
    )
    parser.add_argument(
        "--data-mix-weights",
        type=float,
        nargs="+",
        default=None,
        help="Source-level sampling weights for multiple --data-path entries, e.g. 0.95 0.05.",
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=None,
        help="YAML manifest describing weighted motion data sources. Cannot be combined with --data-path.",
    )
    parser.add_argument(
        "--rebuild-motion-cache",
        action="store_true",
        help="Rebuild manifest-generated motion pkl caches instead of reusing existing cache files.",
    )
    parser.add_argument(
        "--update-z-every-step",
        type=int,
        default=None,
        help="Override latent update interval. Defaults to 100 for FB and 10 for TeCH.",
    )
    parser.add_argument(
        "--expert-z-mode",
        choices=["mean", "direct"],
        default="mean",
        help="Encode expert tracking latents as the sequence mean or a fixed future goal.",
    )
    parser.add_argument(
        "--expert-z-horizon",
        type=int,
        default=8,
        help="One-indexed future horizon used by --expert-z-mode direct.",
    )
    parser.add_argument(
        "--intact-native-goal-coef",
        type=float,
        default=0.0,
        help="Weight of the native BFM actor objective at the direct K=H hindsight endpoint (0 disables).",
    )
    parser.add_argument(
        "--intact-native-local-coef",
        type=float,
        default=0.0,
        help="Weight of the native BFM actor objective at the direct K=1 hindsight endpoint (0 disables).",
    )
    parser.add_argument(
        "--intact-native-residual-coef",
        type=float,
        default=0.0,
        help=(
            "Scheme 4 residual scale beta in L_actor=L_base+beta*(w_local*L_local+w_goal*L_goal). "
            "Disabled by default so non-Scheme-4 commands retain their original objective. "
            "The formal Scheme 4 launcher explicitly uses the calibrated beta."
        ),
    )
    parser.add_argument(
        "--intact-goal-horizon",
        type=int,
        default=8,
        help="Direct hindsight goal horizon H; endpoint clamps to the real trajectory end.",
    )
    parser.add_argument(
        "--intact-grad-diagnostic-samples",
        type=int,
        default=0,
        help="Log unweighted base/goal/local actor gradient norms and cosines for the first N updates.",
    )
    parser.add_argument(
        "--sphere-log-mode",
        choices=("off", "actor", "all"),
        default="off",
        help=(
            "Retain q as the canonical target and map it with Log_qt(qtarget) for actor only "
            "or for actor/forward/critic/discriminator together. Log-vector norms retain sqrt(d)*theta."
        ),
    )
    parser.add_argument(
        "--task-bn-normalizer",
        action="store_true",
        help="Use task-specific BatchNorm running stats (shared affine) on state/privileged_state.",
    )
    parser.add_argument(
        "--carry-aux-terms",
        action="store_true",
        help="Weight closed-loop carry object/contact and body/root terms into the auxiliary reward objective.",
    )
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE, help="Replay capacity per rank/GPU.")
    parser.add_argument(
        "--num-agent-updates",
        type=int,
        default=None,
        help=(
            "Override optimizer updates per update trigger. For fair env-scaling ablations, use 32 with "
            "2048 envs/GPU and 64 with 4096 envs/GPU to match the 1024 envs/GPU update density."
        ),
    )
    parser.add_argument("--disable-dr", action="store_true", help="Disable domain randomization for training.")
    parser.add_argument("--disable-obs-noise", action="store_true", help="Disable observation noise for training.")
    parser.add_argument("--disable-compile", action="store_true", help="Disable torch.compile for this run.")
    parser.add_argument("--lr-scale", type=float, default=1.0, help="Scale FB learning rates. TeCH preset ignores this value.")
    parser.add_argument("--clip-grad-norm", type=float, default=0.0, help="Enable FB actor/FB gradient clipping when > 0.")
    parser.add_argument(
        "--cartwheel-aux-safe",
        action="store_true",
        help="Use a cartwheel-safe FB auxiliary reward set: remove locomotion contact/foot-shape penalties and reduce action-rate penalty.",
    )
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--disable-eval-prioritization",
        action="store_true",
        help="Validation/debug only: skip tracking eval and expert prioritization without changing default training behavior.",
    )
    parser.add_argument("--smoke", action="store_true", help="Short local smoke settings: 16 envs, 2048 env steps, no W&B.")
    args = parser.parse_args()
    raw_agent = args.agent
    args.agent = canonical_agent_name(args.agent)
    if raw_agent == "tldr":
        print("WARNING: agent=tldr is deprecated; use agent=tech instead.", file=sys.stderr, flush=True)
    if args.update_z_every_step is None:
        args.update_z_every_step = _default_update_z_every_step(args.agent)
    if args.smoke:
        args.num_envs = min(args.num_envs, 16)
        args.num_env_steps = min(args.num_env_steps, 2048)
        args.use_wandb = False
    manifest_robot_config = None
    if args.data_manifest is not None:
        if args.data_path is not None:
            parser.error("--data-manifest and --data-path cannot be used together")
        manifest_data = prepare_motion_manifest(args.data_manifest, rebuild_cache=bool(args.rebuild_motion_cache))
        args.data_path = manifest_data.train_data_paths
        args.data_mix_weights = manifest_data.train_data_weights
        manifest_robot_config = manifest_data.robot_config_path
    elif args.data_path is not None:
        data_path_count = len(args.data_path)
        if args.data_mix_weights is not None:
            if len(args.data_mix_weights) != data_path_count:
                raise ValueError("--data-mix-weights length must match --data-path length")
            if any(w < 0 for w in args.data_mix_weights) or sum(args.data_mix_weights) <= 0:
                raise ValueError("--data-mix-weights must be non-negative and sum to a positive value")
            weight_sum = float(sum(args.data_mix_weights))
            args.data_mix_weights = [float(w) / weight_sum for w in args.data_mix_weights]
        elif data_path_count > 1:
            args.data_mix_weights = [1.0 / data_path_count] * data_path_count
        if data_path_count == 1:
            args.data_path = args.data_path[0]
            args.data_mix_weights = None

    args.robot_config = _resolve_training_robot_config(args.robot_config, manifest_robot_config)

    if args.update_z_every_step <= 0:
        raise ValueError("--update-z-every-step must be positive")
    if args.expert_z_horizon <= 0:
        raise ValueError("--expert-z-horizon must be positive")
    if args.intact_goal_horizon <= 0:
        raise ValueError("--intact-goal-horizon must be positive")
    if args.intact_native_residual_coef < 0:
        raise ValueError("--intact-native-residual-coef must be non-negative")
    if args.intact_native_goal_coef < 0 or args.intact_native_local_coef < 0:
        raise ValueError("INTACT native coefficients must be non-negative")
    if args.intact_native_residual_coef > 0 and args.intact_native_goal_coef + args.intact_native_local_coef <= 0:
        raise ValueError("A positive Scheme 4 residual requires a positive local or goal coefficient")
    if args.intact_grad_diagnostic_samples < 0:
        raise ValueError("--intact-grad-diagnostic-samples must be non-negative")
    if args.agent != "fb" and (
        args.intact_native_goal_coef > 0
        or args.intact_native_local_coef > 0
        or args.intact_native_residual_coef > 0
        or args.intact_grad_diagnostic_samples > 0
        or args.sphere_log_mode != "off"
    ):
        raise ValueError("INTACT native objectives are only supported with --agent fb")
    if args.agent != "fb" and (args.expert_z_mode != "mean" or args.expert_z_horizon != 8):
        raise ValueError("Expert-z overrides are only supported with --agent fb")
    if args.buffer_size <= 0:
        raise ValueError("--buffer-size must be positive")
    if args.num_agent_updates is not None and args.num_agent_updates <= 0:
        raise ValueError("--num-agent-updates must be positive")
    if args.lr_scale <= 0:
        raise ValueError("--lr-scale must be positive")
    if args.clip_grad_norm < 0:
        raise ValueError("--clip-grad-norm must be non-negative")
    if args.cartwheel_aux_safe and args.agent != "fb":
        raise ValueError("--cartwheel-aux-safe is only supported with --agent fb")
    if args.task == "carry_box" and args.agent != "fb":
        raise ValueError("--task carry_box currently requires --agent fb")
    return args


def main() -> None:
    launch(parse_args())


if __name__ == "__main__":
    main()

"""Offline integrity, schema, FK, and ONNX checks for the packaged Carry policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import yaml

from rl_policy.carry_runtime import (
    BestCarry140935168StrongController,
    CarryActorObservation,
    CarryMocapFrame,
    CarryPhaseController,
    G1HandForwardKinematics,
    H20Reward736PhaseController,
    InitialCarryScene,
    HandBoxContactEstimator,
    NewIntactHybridDirectPlaceController,
    ONNX_INPUT_DIM,
    OBJECT_CONTEXT_DIM,
    SPHERE_LOG_ONNX_INPUT_DIM,
    TrackingReferenceAnchor,
    load_fixed_goal_latent,
    load_get_up_latent,
    load_hybrid_direct_place_latents,
    load_phase_latents,
    load_tracking_latents,
)
from rl_policy.lumo_mocap import LuMoCarrySource
from utils.math import quat_mul, quat_rotate_numpy


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_onnx_cpu(path: Path, actor: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        import onnxruntime

        session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        output = session.run(
            None,
            {
                session.get_inputs()[0].name:
                actor.reshape(1, -1).astype(np.float32)
            },
        )[0]
        return output, "onnxruntime"
    except ImportError:
        import onnx
        from onnx.reference import ReferenceEvaluator

        model = onnx.load(path)
        onnx.checker.check_model(model)
        session = ReferenceEvaluator(model)
        output = session.run(
            None,
            {session.input_names[0]: actor.reshape(1, -1).astype(np.float32)},
        )[0]
        return output, "onnx-reference (install requirements/runtime.txt before deployment)"


def run_onnx_provider(
    path: Path, actor: np.ndarray, provider: str
) -> tuple[np.ndarray, list[str]]:
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=[provider])
    active = list(session.get_providers())
    if provider not in active:
        raise RuntimeError(
            f"required ONNX provider {provider!r} is not active: {active}"
        )
    output = session.run(
        None,
        {
            session.get_inputs()[0].name:
            actor.reshape(1, -1).astype(np.float32)
        },
    )[0]
    return output, active


def validate_rigid_source_box_alignment(
    source: LuMoCarrySource,
    exp_cfg: dict,
    scene: InitialCarryScene | None,
    source_config_key: str,
) -> None:
    phase = exp_cfg.get("phase", {})
    contact = exp_cfg.get("contact_estimator", {})
    geom_position = np.asarray(
        phase.get("box_geom_position", contact.get("geom_position", [0.0, 0.0, 0.0])),
        dtype=np.float64,
    )
    geom_quaternion = np.asarray(
        phase.get(
            "box_geom_quaternion_wxyz",
            contact.get("geom_quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]),
        ),
        dtype=np.float64,
    )
    physical_quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
    reference_quaternion = quat_mul(
        physical_quaternion.reshape(1, 4),
        source.object_extrinsic.quaternion_wxyz.reshape(1, 4),
    )[0]
    reconstructed_quaternion = quat_mul(
        reference_quaternion.reshape(1, 4), geom_quaternion.reshape(1, 4)
    )[0]
    quaternion_error = 1.0 - abs(
        float(np.dot(reconstructed_quaternion, physical_quaternion))
    )

    expected_physical_height = float(
        exp_cfg.get("initial_scene", {}).get(
            "expected_box_center_height",
            exp_cfg.get(source_config_key, {}).get(
                "target_box_center_height", phase.get("ground_center_height", 0.0)
            ),
        )
    )
    physical_position = np.asarray(
        [1.2, 0.0, expected_physical_height], dtype=np.float64
    )
    reference_position = physical_position + quat_rotate_numpy(
        physical_quaternion.reshape(1, 4),
        source.object_extrinsic.position.reshape(1, 3),
    )[0]
    reconstructed_position = reference_position + quat_rotate_numpy(
        reference_quaternion.reshape(1, 4), geom_position.reshape(1, 3)
    )[0]
    position_error = float(np.linalg.norm(reconstructed_position - physical_position))
    if position_error > 1.0e-6 or quaternion_error > 1.0e-6:
        raise RuntimeError(
            "mocap physical-box/training-reference transform does not invert the "
            f"MuJoCo box geom: position_error={position_error:.3e}m "
            f"quaternion_error={quaternion_error:.3e}"
        )

    if scene is not None:
        synthetic = CarryMocapFrame(
            sequence=1,
            timestamp_ns=1,
            robot_root_position=np.asarray([0.0, 0.0, 0.8]),
            robot_root_quaternion_wxyz=physical_quaternion,
            robot_root_linear_velocity_world=np.zeros(3),
            object_position=reference_position,
            object_quaternion_wxyz=reference_quaternion,
            object_linear_velocity_world=np.zeros(3),
            object_angular_velocity_world=np.zeros(3),
            hand_contact=np.zeros(2),
            target_position_world=source.target_position_world.copy(),
        )
        scene.apply(synthetic)

    source_cfg = exp_cfg[source_config_key]
    if str(source_cfg.get("target_pose_semantics", "")).lower() == "physical_ground":
        target_center = np.asarray(source_cfg["target_position_world"], dtype=np.float64)
        target_center[2] += float(source_cfg["target_box_center_height"])
        reconstructed_target = source.target_position_world + quat_rotate_numpy(
            reference_quaternion.reshape(1, 4), geom_position.reshape(1, 3)
        )[0]
        if not np.allclose(reconstructed_target, target_center, atol=1.0e-6):
            raise RuntimeError(
                "configured physical-ground target does not round-trip through "
                "the training-reference transform"
            )


def validate_physical_box_contract(
    manifest: dict,
    source: LuMoCarrySource,
    exp_cfg: dict,
    source_config_key: str,
) -> None:
    """Pin real payload dimensions independently from the training XML geom."""
    configured_size = manifest.get("physical_box_size_m")
    if configured_size is None:
        return
    physical_size = np.asarray(configured_size, dtype=np.float64)
    if physical_size.shape != (3,) or not np.all(np.isfinite(physical_size)):
        raise RuntimeError("manifest physical_box_size_m must contain three finite values")
    if np.any(physical_size <= 0.0):
        raise RuntimeError("manifest physical_box_size_m must be positive")

    contact_half_size = np.asarray(
        exp_cfg.get("contact_estimator", {}).get("half_size"), dtype=np.float64
    )
    if contact_half_size.shape != (3,) or not np.allclose(
        2.0 * contact_half_size, physical_size, atol=1.0e-9
    ):
        raise RuntimeError(
            "contact_estimator.half_size does not match manifest physical_box_size_m"
        )

    source_cfg = exp_cfg[source_config_key]
    if str(source_cfg.get("target_pose_semantics", "")).lower() != "physical_ground":
        raise RuntimeError(
            f"real-box contract requires {source_config_key}.target_pose_semantics=physical_ground"
        )
    expected_center_height = 0.5 * float(physical_size[2])
    if not np.isclose(
        float(source_cfg.get("target_box_center_height", np.nan)),
        expected_center_height,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            f"{source_config_key}.target_box_center_height does not match half the physical box height"
        )
    initial_scene = exp_cfg.get("initial_scene")
    if initial_scene is not None and not np.isclose(
        float(initial_scene.get("expected_box_center_height", np.nan)),
        expected_center_height,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            "initial_scene.expected_box_center_height does not match the physical box"
        )
    if (
        initial_scene is None
        and str(exp_cfg.get("phase_controller_profile", "")).lower()
        == "newintact80m_hybrid_direct_place"
        and not bool(exp_cfg.get("phase", {}).get("initial_approach_from_live_pose", False))
    ):
        raise RuntimeError(
            "an unconstrained hybrid initial pose requires "
            "phase.initial_approach_from_live_pose=true"
        )

    expected_target = manifest.get("physical_target_ground_world")
    if expected_target is not None and not np.allclose(
        np.asarray(source_cfg.get("target_position_world"), dtype=np.float64),
        np.asarray(expected_target, dtype=np.float64),
        atol=1.0e-9,
    ):
        raise RuntimeError(
            f"{source_config_key}.target_position_world does not match manifest physical_target_ground_world"
        )
    if not np.isclose(
        float(exp_cfg.get("phase", {}).get("ground_center_height", np.nan)),
        float(source.target_position_world[2]),
        atol=1.0e-8,
    ):
        raise RuntimeError(
            "phase.ground_center_height does not match the calibrated training-reference "
            "height of the grounded physical box"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "model/g1_carry_intact_logmap166559744_solve_close_gpu/manifest.json")
    parser.add_argument(
        "--task-config",
        type=Path,
        default=None,
        help="site/runtime Carry config; validated structurally but not pinned as a model artifact",
    )
    parser.add_argument(
        "--require-runtime-mode",
        choices=("phased", "fixed_goal", "tracking"),
        default=None,
        help="reject a valid Carry config when it selects the wrong controller mode",
    )
    parser.add_argument(
        "--require-controller-profile",
        choices=(
            "legacy_deploy",
            "h20_reward736",
            "bestcarry140935168_strong",
            "intact_logmap150544384_strong",
            "newintact80m_hybrid_direct_place",
        ),
        default=None,
        help="reject a phased Carry config when it selects the wrong controller profile",
    )
    parser.add_argument(
        "--require-onnx-provider",
        default=None,
        help="fail unless this ONNX Runtime execution provider is installed and configured",
    )
    parser.add_argument(
        "--require-deployment-calibration",
        action="store_true",
        help="fail unless the three-stage real-robot calibration gate is enabled",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    mutable_configs = set(manifest.get("mutable_runtime_configs", []))
    checked_hashes = 0
    for relative, expected in manifest["sha256"].items():
        if relative in mutable_configs:
            continue
        path = ROOT / relative
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"hash mismatch for {relative}: {actual} != {expected}")
        checked_hashes += 1

    policy_cfg = yaml.safe_load((ROOT / manifest["policy_config"]).read_text())
    task_path = args.task_config or (ROOT / manifest["experiment_config"])
    if not task_path.is_absolute():
        task_path = ROOT / task_path
    exp_cfg = yaml.safe_load(task_path.read_text())
    if not isinstance(exp_cfg, dict) or exp_cfg.get("type") != "carry":
        raise RuntimeError(f"{task_path} must contain a Carry task mapping")
    calibration_cfg = exp_cfg.get("deployment_calibration", {})
    if not isinstance(calibration_cfg, dict):
        raise RuntimeError("deployment_calibration must be a mapping")
    if args.require_deployment_calibration:
        if not bool(calibration_cfg.get("enabled", False)):
            raise RuntimeError("mandatory deployment calibration is not enabled")
        if not str(calibration_cfg.get("saved_result_path", "")).strip():
            raise RuntimeError(
                "mandatory deployment calibration requires saved_result_path"
            )
        contact_cfg = exp_cfg.get("contact_estimator", {})
        mocap_source_name = str(exp_cfg.get("mocap_source", "json_zmq")).lower()
        mocap_cfg = exp_cfg.get(mocap_source_name, {})
        contact_calibration_enabled = bool(
            calibration_cfg.get("contact_calibration_enabled", True)
        )
        if contact_calibration_enabled:
            if not bool(contact_cfg.get("use_hand_rigid_origins", False)):
                raise RuntimeError(
                    "deployment contact calibration requires hand rigid origins"
                )
            for key in ("left_hand_rigid_body", "right_hand_rigid_body"):
                if not isinstance(mocap_cfg.get(key), dict):
                    raise RuntimeError(
                        f"deployment contact calibration requires {mocap_source_name}.{key}"
                    )
        else:
            if bool(contact_cfg.get("use_hand_rigid_origins", False)):
                raise RuntimeError(
                    "FK-only contact mode must disable use_hand_rigid_origins"
                )
            for key in ("left_hand_rigid_body", "right_hand_rigid_body"):
                if key in mocap_cfg:
                    raise RuntimeError(
                        f"FK-only contact mode must omit {mocap_source_name}.{key}"
                    )
            if str(exp_cfg.get("contact_source", "")) != "box_hysteresis":
                raise RuntimeError(
                    "FK-only contact mode requires contact_source=box_hysteresis"
                )
    sphere_log_actor = bool(exp_cfg.get("sphere_log_actor", False))
    required_input_dim = (
        SPHERE_LOG_ONNX_INPUT_DIM if sphere_log_actor else ONNX_INPUT_DIM
    )
    if int(exp_cfg.get("expected_input_dim", required_input_dim)) != required_input_dim:
        raise RuntimeError(f"{task_path} expected_input_dim must be {required_input_dim}")
    configured_providers = policy_cfg.get("onnx_providers", ["CPUExecutionProvider"])
    if isinstance(configured_providers, str):
        configured_providers = [configured_providers]
    if args.require_onnx_provider is not None:
        if args.require_onnx_provider not in configured_providers:
            raise RuntimeError(
                f"policy does not request required ONNX provider "
                f"{args.require_onnx_provider!r}: {configured_providers}"
            )
        try:
            import onnxruntime
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for provider validation") from exc
        available_providers = onnxruntime.get_available_providers()
        if args.require_onnx_provider not in available_providers:
            raise RuntimeError(
                f"required ONNX provider {args.require_onnx_provider!r} is unavailable; "
                f"available={available_providers}"
            )
    rigid_source = None
    source_config_key = None
    mocap_source_name = str(exp_cfg.get("mocap_source", "json_zmq")).lower()
    if mocap_source_name == "lumo":
        source_config_key = "lumo"
        rigid_source = LuMoCarrySource(
            exp_cfg.get("lumo", {}),
            float(exp_cfg.get("mocap_timeout_ms", 100.0)),
            connect=False,
        )
    else:
        raise RuntimeError("This deployment package requires mocap_source=lumo")
    actor_inertial_source = str(
        exp_cfg.get("actor_inertial_source", "onboard_imu")
    ).lower()
    if actor_inertial_source not in {"onboard_imu", "mocap"}:
        raise RuntimeError(
            "actor_inertial_source must be 'onboard_imu' or 'mocap'"
        )
    if actor_inertial_source == "mocap" and rigid_source is None:
        raise RuntimeError(
            "actor_inertial_source=mocap requires a rigid-body mocap source"
        )
    if str(exp_cfg.get("contact_source", "packet")).lower() == "box_hysteresis":
        HandBoxContactEstimator(exp_cfg.get("contact_estimator", {}))
    joint_names = list(policy_cfg["isaac_joint_names"])
    default = np.asarray([-0.1, 0, 0, 0.3, -0.2, 0] * 2 + [0] * 17, dtype=np.float64)
    fk = G1HandForwardKinematics(
        ROOT / exp_cfg["kinematics_xml"],
        joint_names,
        exp_cfg["hand_contact_offsets"],
        privileged_body_names=exp_cfg.get("privileged_body_names"),
        privileged_torso_body_name=str(
            exp_cfg.get("privileged_torso_body_name", "torso_link")
        ),
    )
    observer = CarryActorObservation(default, fk)
    frame = CarryMocapFrame.from_mapping({
        "sequence": 1,
        "timestamp_ns": 1,
        "robot_root": {
            "position": [0, 0, 0.8],
            "quaternion_wxyz": [1, 0, 0, 0],
            "angular_velocity_world": [0, 0, 0],
        },
        "object": {"position": [0.65, 0, 0.175], "quaternion_wxyz": [1, 0, 0, 0],
                   "linear_velocity_world": [0, 0, 0], "angular_velocity_world": [0, 0, 0]},
        "contacts": {"left": False, "right": False},
        "target_position_world": [1.0, 0.0, 0.175],
    })
    obs = observer.build(frame, default, np.zeros(29), np.asarray([1, 0, 0, 0]), np.zeros(3), np.zeros(29))
    fixed_goal_path = exp_cfg.get("fixed_goal_latent")
    tracking_path = exp_cfg.get("tracking_latent")
    if fixed_goal_path is not None and tracking_path is not None:
        raise RuntimeError(f"{task_path} cannot select fixed_goal_latent and tracking_latent together")
    if tracking_path is not None:
        resolved_tracking_path = Path(str(tracking_path)).expanduser()
        if not resolved_tracking_path.is_absolute():
            resolved_tracking_path = ROOT / resolved_tracking_path
        tracking_latents = load_tracking_latents(resolved_tracking_path)
        TrackingReferenceAnchor(exp_cfg["tracking_reference"])
        latent = tracking_latents[0]
        runtime_mode = "tracking"
    elif fixed_goal_path is None:
        controller_profile = str(
            exp_cfg.get("phase_controller_profile", "legacy_deploy")
        ).lower()
        if controller_profile == "newintact80m_hybrid_direct_place":
            latents = load_hybrid_direct_place_latents(ROOT / exp_cfg["latent_dir"])
            approach_latents = load_phase_latents(
                ROOT / exp_cfg["approach_latent_dir"]
            )
            latents["approach"] = approach_latents["approach"]
            latents["turn"] = approach_latents["turn"]
        else:
            latents = load_phase_latents(ROOT / exp_cfg["latent_dir"])
        get_up_path = exp_cfg.get("get_up_latent")
        if get_up_path is not None:
            resolved_get_up_path = Path(str(get_up_path)).expanduser()
            if not resolved_get_up_path.is_absolute():
                resolved_get_up_path = ROOT / resolved_get_up_path
            latents["get_up"] = load_get_up_latent(resolved_get_up_path)
        controller_types = {
            "legacy_deploy": CarryPhaseController,
            "h20_reward736": H20Reward736PhaseController,
            "bestcarry140935168_strong": BestCarry140935168StrongController,
            "intact_logmap150544384_strong": BestCarry140935168StrongController,
            "newintact80m_hybrid_direct_place": NewIntactHybridDirectPlaceController,
        }
        if controller_profile not in controller_types:
            raise RuntimeError(f"unsupported phase_controller_profile={controller_profile!r}")
        controller_types[controller_profile](exp_cfg.get("phase", {}), latents, 0.02)
        scene = None
        if exp_cfg.get("initial_scene") is not None:
            scene = InitialCarryScene(
                exp_cfg["initial_scene"],
                exp_cfg.get("phase", {}).get(
                    "box_geom_quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]
                ),
                exp_cfg.get("phase", {}).get(
                    "box_geom_position", [0.0, 0.0, 0.0]
                ),
            )
        if rigid_source is not None and source_config_key is not None:
            validate_rigid_source_box_alignment(
                rigid_source, exp_cfg, scene, source_config_key
            )
            validate_physical_box_contract(
                manifest, rigid_source, exp_cfg, source_config_key
            )
        latent = latents["approach"]
        runtime_mode = "phased"
    else:
        goal_path = Path(str(fixed_goal_path)).expanduser()
        if not goal_path.is_absolute():
            goal_path = ROOT / goal_path
        latent = load_fixed_goal_latent(goal_path)
        runtime_mode = "fixed_goal"
        controller_profile = "not_applicable"
    if tracking_path is not None:
        controller_profile = "not_applicable"
    if args.require_runtime_mode is not None and runtime_mode != args.require_runtime_mode:
        raise RuntimeError(
            f"{task_path} selects runtime_mode={runtime_mode}, "
            f"but {args.require_runtime_mode} is required"
        )
    if (
        args.require_controller_profile is not None
        and controller_profile != args.require_controller_profile
    ):
        raise RuntimeError(
            f"{task_path} selects controller_profile={controller_profile}, "
            f"but {args.require_controller_profile} is required"
        )
    actor_parts = [obs["state"], obs["last_action"], obs["history_actor"]]
    if sphere_log_actor:
        fk.hand_geometry(
            frame.robot_root_position,
            frame.robot_root_quaternion_wxyz,
            default,
            root_linear_velocity_world=np.zeros(3),
            root_angular_velocity_body=np.zeros(3),
            joint_velocity=np.zeros(29),
        )
        actor_parts.append(fk.privileged_state(obs["state"][-OBJECT_CONTEXT_DIM:]))
    actor_parts.append(latent)
    actor = np.concatenate(actor_parts)
    if actor.shape != (required_input_dim,) or not np.all(np.isfinite(actor)):
        raise RuntimeError(f"invalid synthetic actor input: {actor.shape}")
    model_path = ROOT / manifest["onnx_model"]
    output, backend = run_onnx_cpu(model_path, actor)
    if output.shape != (1, 29) or not np.all(np.isfinite(output)):
        raise RuntimeError(f"invalid ONNX output: {output.shape}")
    if args.require_onnx_provider is not None:
        provider_output, active_providers = run_onnx_provider(
            model_path, actor, args.require_onnx_provider
        )
        if provider_output.shape != (1, 29) or not np.all(np.isfinite(provider_output)):
            raise RuntimeError(
                f"invalid {args.require_onnx_provider} ONNX output: "
                f"{provider_output.shape}"
            )
        provider_error = float(np.max(np.abs(provider_output - output)))
        if provider_error > 2.0e-4:
            raise RuntimeError(
                f"{args.require_onnx_provider} output disagrees with CPU: "
                f"max_abs_error={provider_error:.8g}"
            )
        print(
            f"PASS required_provider={args.require_onnx_provider} "
            f"active={active_providers} cpu_max_abs_error={provider_error:.8g}"
        )
    approach_model = manifest.get("approach_onnx_model")
    if approach_model is not None:
        configured_approach = exp_cfg.get("approach_model_path")
        if configured_approach != approach_model:
            raise RuntimeError(
                "task approach_model_path does not match manifest "
                f"({configured_approach!r} != {approach_model!r})"
            )
        approach_path = ROOT / approach_model
        approach_output, approach_backend = run_onnx_cpu(approach_path, actor)
        if approach_output.shape != (1, 29) or not np.all(np.isfinite(approach_output)):
            raise RuntimeError(f"invalid approach ONNX output: {approach_output.shape}")
        print(
            f"PASS approach_model input={actor.shape[0]} "
            f"output={approach_output.shape[1]} backend={approach_backend}"
        )
    print(
        f"PASS artifact hashes={checked_hashes} mutable_runtime_configs={len(mutable_configs)} "
        f"input={actor.shape[0]} output={output.shape[1]} backend={backend}"
    )
    print(
        f"PASS task_config={task_path} mocap_source={exp_cfg.get('mocap_source', 'json_zmq')} "
        f"actor_inertial_source={actor_inertial_source} "
        f"runtime_mode={runtime_mode} controller_profile={controller_profile}"
    )
    print(f"PASS hand proxy points={fk.hand_points(frame.robot_root_position, frame.robot_root_quaternion_wxyz, default).tolist()}")


if __name__ == "__main__":
    main()

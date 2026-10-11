"""Shared object observations for simulated and expert carry trajectories."""

from __future__ import annotations

import torch

from humanoidverse.utils.torch_utils import quat_conjugate, quat_mul, quat_rotate, quat_rotate_inverse, quat_to_tan_norm


CARRY_OBJECT_OBSERVATION_DIM = 18
CARRY_HAND_CONTACT_DIM = 2
CARRY_HAND_SPATIAL_DIM = 12
LEGACY_OBJECT_CONTEXT_DIM = CARRY_OBJECT_OBSERVATION_DIM + CARRY_HAND_CONTACT_DIM + 2
OBJECT_PRESENT_INDEX = CARRY_OBJECT_OBSERVATION_DIM + CARRY_HAND_CONTACT_DIM + CARRY_HAND_SPATIAL_DIM
TASK_ID_INDEX = OBJECT_PRESENT_INDEX + 1
OBJECT_CONTEXT_DIM = TASK_ID_INDEX + 1


def select_reference_hand_box_contact(
    phase_contact: torch.Tensor,
    stored_contact: torch.Tensor | None,
    stored_available: torch.Tensor | None,
) -> torch.Tensor:
    """Prefer MuJoCo-generated expert labels and fall back per motion when absent."""

    if stored_contact is None or stored_available is None:
        return phase_contact
    contact = stored_contact.to(device=phase_contact.device, dtype=phase_contact.dtype)
    if contact.shape != phase_contact.shape:
        raise ValueError(f"stored_contact shape {contact.shape} must match phase_contact {phase_contact.shape}")
    available = stored_available.to(device=phase_contact.device, dtype=torch.bool)
    if available.ndim == phase_contact.ndim - 1:
        available = available.unsqueeze(-1)
    if available.shape[:-1] != phase_contact.shape[:-1] or available.shape[-1] not in (1, CARRY_HAND_CONTACT_DIM):
        raise ValueError(f"stored_available shape {available.shape} is incompatible with {phase_contact.shape}")
    return torch.where(available, contact, phase_contact)


def compute_carry_object_observation(
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    object_pos: torch.Tensor,
    object_quat: torch.Tensor,
    object_lin_vel: torch.Tensor,
    object_ang_vel: torch.Tensor,
    target_object_pos: torch.Tensor,
) -> torch.Tensor:
    """Return an 18D root-local object and target observation."""

    object_pos_local = quat_rotate_inverse(base_quat, object_pos - base_pos, w_last=True)
    object_rot_local = quat_mul(quat_conjugate(base_quat, w_last=True), object_quat, w_last=True)
    object_rot_obs = quat_to_tan_norm(object_rot_local, w_last=True)
    object_lin_vel_local = quat_rotate_inverse(base_quat, object_lin_vel, w_last=True)
    object_ang_vel_local = quat_rotate_inverse(base_quat, object_ang_vel, w_last=True)
    target_delta_local = quat_rotate_inverse(base_quat, target_object_pos - object_pos, w_last=True)
    return torch.cat(
        [object_pos_local, object_rot_obs, object_lin_vel_local, object_ang_vel_local, target_delta_local],
        dim=-1,
    )


def compute_hand_spatial_observation(
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    hand_pos: torch.Tensor,
    object_pos: torch.Tensor,
) -> torch.Tensor:
    """Return root-local hand positions and hand-minus-object vectors as 12D."""

    if hand_pos.shape[-2:] != (CARRY_HAND_CONTACT_DIM, 3):
        raise ValueError(f"hand_pos must end in [2, 3], got {hand_pos.shape}")
    batch_shape = hand_pos.shape[:-2]
    if base_pos.shape != batch_shape + (3,):
        raise ValueError(f"base_pos shape {base_pos.shape} is incompatible with hand_pos {hand_pos.shape}")
    if base_quat.shape != batch_shape + (4,):
        raise ValueError(f"base_quat shape {base_quat.shape} is incompatible with hand_pos {hand_pos.shape}")
    if object_pos.shape != batch_shape + (3,):
        raise ValueError(f"object_pos shape {object_pos.shape} is incompatible with hand_pos {hand_pos.shape}")

    expanded_quat = base_quat.unsqueeze(-2).expand(batch_shape + (CARRY_HAND_CONTACT_DIM, 4))
    expanded_base_pos = base_pos.unsqueeze(-2)
    expanded_object_pos = object_pos.unsqueeze(-2)
    hand_pos_local = quat_rotate_inverse(
        expanded_quat.reshape(-1, 4),
        (hand_pos - expanded_base_pos).reshape(-1, 3),
        w_last=True,
    ).reshape(hand_pos.shape)
    hand_minus_object_local = quat_rotate_inverse(
        expanded_quat.reshape(-1, 4),
        (hand_pos - expanded_object_pos).reshape(-1, 3),
        w_last=True,
    ).reshape(hand_pos.shape)
    return torch.cat(
        (
            hand_pos_local.reshape(batch_shape + (CARRY_HAND_CONTACT_DIM * 3,)),
            hand_minus_object_local.reshape(batch_shape + (CARRY_HAND_CONTACT_DIM * 3,)),
        ),
        dim=-1,
    )


def build_object_context(
    object_observation: torch.Tensor,
    object_present: torch.Tensor,
    hand_box_contact: torch.Tensor | None = None,
    hand_spatial_observation: torch.Tensor | None = None,
    task_id: torch.Tensor | None = None,
    *,
    include_hand_spatial: bool = True,
    observe_target_delta: bool = True,
    observe_object_present: bool = True,
    observe_task_id: bool = True,
) -> torch.Tensor:
    """Mask object/contact features and append stable object/task indicators.

    Observation ablations zero their assigned channels instead of changing the
    schema, keeping network sizes and normalizer structure identical across runs.
    """

    if object_observation.shape[-1] != CARRY_OBJECT_OBSERVATION_DIM:
        raise ValueError(
            f"Expected {CARRY_OBJECT_OBSERVATION_DIM} object features, got {object_observation.shape[-1]}"
        )
    present = object_present.to(device=object_observation.device, dtype=object_observation.dtype)
    if present.ndim == object_observation.ndim - 1:
        present = present.unsqueeze(-1)
    if present.shape != object_observation.shape[:-1] + (1,):
        raise ValueError(f"object_present shape {present.shape} is incompatible with {object_observation.shape}")
    if hand_box_contact is None:
        contact = object_observation.new_zeros(object_observation.shape[:-1] + (CARRY_HAND_CONTACT_DIM,))
    else:
        contact = hand_box_contact.to(device=object_observation.device, dtype=object_observation.dtype)
        expected_shape = object_observation.shape[:-1] + (CARRY_HAND_CONTACT_DIM,)
        if contact.shape != expected_shape:
            raise ValueError(f"hand_box_contact shape {contact.shape} must be {expected_shape}")
    if hand_spatial_observation is None:
        hand_spatial = object_observation.new_zeros(object_observation.shape[:-1] + (CARRY_HAND_SPATIAL_DIM,))
    else:
        hand_spatial = hand_spatial_observation.to(device=object_observation.device, dtype=object_observation.dtype)
        expected_shape = object_observation.shape[:-1] + (CARRY_HAND_SPATIAL_DIM,)
        if hand_spatial.shape != expected_shape:
            raise ValueError(f"hand_spatial_observation shape {hand_spatial.shape} must be {expected_shape}")
    if not observe_target_delta:
        object_observation = object_observation.clone()
        object_observation[..., -3:] = 0
    if task_id is None:
        task = present
    else:
        task = task_id.to(device=object_observation.device, dtype=object_observation.dtype)
        if task.ndim == object_observation.ndim - 1:
            task = task.unsqueeze(-1)
        if task.shape != present.shape:
            raise ValueError(f"task_id shape {task.shape} must match object_present shape {present.shape}")
    if not observe_task_id:
        task = torch.zeros_like(task)
    observed_present = present if observe_object_present else torch.zeros_like(present)
    parts = [object_observation * present, contact * present]
    if include_hand_spatial:
        parts.append(hand_spatial * present)
    parts.extend((observed_present, task))
    return torch.cat(parts, dim=-1)


def infer_hand_box_contact(
    hand_pos: torch.Tensor,
    object_pos: torch.Tensor,
    object_quat: torch.Tensor,
    box_half_size: torch.Tensor,
    box_geom_pos: torch.Tensor,
    box_geom_quat: torch.Tensor,
    *,
    hand_contact_radius: float = 0.0,
    enter_distance: float = 0.015,
    exit_distance: float = 0.03,
) -> torch.Tensor:
    """Infer per-hand binary contact from reference poses using box-distance hysteresis.

    The leading dimension is time. Object poses describe the tracked marker, while
    the geometry offset and rotation map that marker to the physical box frame.
    """

    if not 0.0 <= enter_distance <= exit_distance:
        raise ValueError("contact thresholds must satisfy 0 <= enter_distance <= exit_distance")
    surface_distance = compute_hand_box_surface_distance(
        hand_pos,
        object_pos,
        object_quat,
        box_half_size,
        box_geom_pos,
        box_geom_quat,
        hand_contact_radius=hand_contact_radius,
    )

    return infer_hand_box_contact_from_distance(
        surface_distance,
        enter_distance=enter_distance,
        exit_distance=exit_distance,
    ).to(hand_pos.dtype)


def infer_hand_box_contact_from_distance(
    surface_distance: torch.Tensor,
    *,
    enter_distance: float = 0.015,
    exit_distance: float = 0.03,
) -> torch.Tensor:
    """Apply temporal hysteresis to a [T, 2] hand-to-box distance sequence."""

    if surface_distance.ndim != 2 or surface_distance.shape[-1] != CARRY_HAND_CONTACT_DIM:
        raise ValueError(f"surface_distance must have shape [T, 2], got {surface_distance.shape}")
    if not 0.0 <= enter_distance <= exit_distance:
        raise ValueError("contact thresholds must satisfy 0 <= enter_distance <= exit_distance")
    contact = torch.zeros_like(surface_distance, dtype=torch.bool)
    if contact.shape[0] == 0:
        return contact.to(surface_distance.dtype)
    contact[0] = surface_distance[0] < enter_distance
    for frame in range(1, contact.shape[0]):
        contact[frame] = torch.where(
            surface_distance[frame] < enter_distance,
            True,
            torch.where(surface_distance[frame] > exit_distance, False, contact[frame - 1]),
        )
    return contact.to(surface_distance.dtype)


def update_hand_box_contact_hysteresis(
    surface_distance: torch.Tensor,
    previous_contact: torch.Tensor,
    *,
    enter_distance: float = 0.015,
    exit_distance: float = 0.03,
) -> torch.Tensor:
    """Update binary contact for independent online environments."""

    if surface_distance.shape != previous_contact.shape:
        raise ValueError(
            f"surface_distance shape {surface_distance.shape} must match previous_contact {previous_contact.shape}"
        )
    if not 0.0 <= enter_distance <= exit_distance:
        raise ValueError("contact thresholds must satisfy 0 <= enter_distance <= exit_distance")
    previous = previous_contact.to(device=surface_distance.device, dtype=torch.bool)
    return torch.where(
        surface_distance < enter_distance,
        True,
        torch.where(surface_distance > exit_distance, False, previous),
    )


def infer_carry_phase_contact(
    object_pos: torch.Tensor,
    object_quat: torch.Tensor,
    object_present: torch.Tensor,
    box_half_size: torch.Tensor,
    box_geom_pos: torch.Tensor,
    box_geom_quat: torch.Tensor,
    *,
    ground_surface_height: float = 0.0,
    lift_clearance: float = 0.005,
) -> torch.Tensor:
    """Set both labels from the oriented box's bottom clearance above ground."""

    if object_pos.shape[-1] != 3 or object_quat.shape != object_pos.shape[:-1] + (4,):
        raise ValueError(f"Incompatible object pose shapes: {object_pos.shape}, {object_quat.shape}")
    present = object_present.reshape(object_pos.shape[:-1]).to(device=object_pos.device, dtype=torch.bool)
    if lift_clearance < 0.0:
        raise ValueError("lift_clearance must be non-negative")
    geom_pos = box_geom_pos.to(device=object_pos.device, dtype=object_pos.dtype).expand(object_pos.shape)
    geom_quat = box_geom_quat.to(device=object_pos.device, dtype=object_pos.dtype).expand(object_quat.shape)
    geom_center = object_pos + quat_rotate(
        object_quat.reshape(-1, 4), geom_pos.reshape(-1, 3), w_last=True
    ).reshape(object_pos.shape)
    geom_orientation = quat_mul(
        object_quat.reshape(-1, 4), geom_quat.reshape(-1, 4), w_last=True
    ).reshape(object_quat.shape)
    half_size = box_half_size.to(device=object_pos.device, dtype=object_pos.dtype)
    local_axes = torch.eye(3, device=object_pos.device, dtype=object_pos.dtype) * half_size
    expanded_axes = local_axes.expand(object_pos.shape[:-1] + (3, 3))
    expanded_quat = geom_orientation.unsqueeze(-2).expand(object_pos.shape[:-1] + (3, 4))
    world_axes = quat_rotate(
        expanded_quat.reshape(-1, 4), expanded_axes.reshape(-1, 3), w_last=True
    ).reshape(object_pos.shape[:-1] + (3, 3))
    vertical_half_extent = world_axes[..., 2].abs().sum(dim=-1)
    bottom_height = geom_center[..., 2] - vertical_half_extent
    lifted = present & (
        bottom_height - float(ground_surface_height) > float(lift_clearance)
    )
    return lifted.unsqueeze(-1).expand(object_pos.shape[:-1] + (CARRY_HAND_CONTACT_DIM,)).to(object_pos.dtype)


def compute_hand_capsule_box_surface_distance(
    hand_body_pos: torch.Tensor,
    hand_body_quat: torch.Tensor,
    hand_capsule_segments: torch.Tensor,
    object_pos: torch.Tensor,
    object_quat: torch.Tensor,
    box_half_size: torch.Tensor,
    box_geom_pos: torch.Tensor,
    box_geom_quat: torch.Tensor,
    *,
    hand_contact_radius: float,
) -> torch.Tensor:
    """Return exact capsule-to-oriented-box signed gaps for both hands.

    The capsule center lines are transformed from each wrist-yaw body frame to
    the physical box frame. Segment-to-AABB distance is minimized exactly over
    its piecewise-quadratic intervals, then the capsule radius is subtracted.
    """

    if hand_body_pos.shape[-2:] != (CARRY_HAND_CONTACT_DIM, 3):
        raise ValueError(f"hand_body_pos must end in [2, 3], got {hand_body_pos.shape}")
    if hand_body_quat.shape != hand_body_pos.shape[:-1] + (4,):
        raise ValueError(f"hand_body_quat shape {hand_body_quat.shape} is incompatible with {hand_body_pos.shape}")
    if object_pos.shape != hand_body_pos.shape[:-2] + (3,):
        raise ValueError(f"object_pos shape {object_pos.shape} is incompatible with {hand_body_pos.shape}")
    if object_quat.shape != hand_body_pos.shape[:-2] + (4,):
        raise ValueError(f"object_quat shape {object_quat.shape} is incompatible with {hand_body_pos.shape}")
    if hand_contact_radius < 0.0:
        raise ValueError("hand_contact_radius must be non-negative")
    segments = hand_capsule_segments.to(device=hand_body_pos.device, dtype=hand_body_pos.dtype)
    if segments.ndim == 3:
        segments = segments.unsqueeze(1)
    if segments.ndim != 4 or segments.shape[0] != CARRY_HAND_CONTACT_DIM or segments.shape[-2:] != (2, 3):
        raise ValueError(f"hand_capsule_segments must have shape [2, K, 2, 3], got {segments.shape}")

    leading_shape = hand_body_pos.shape[:-2]
    num_capsules = segments.shape[1]
    expanded_segments = segments.expand(leading_shape + segments.shape)
    expanded_quat = hand_body_quat[..., None, None, :].expand(
        leading_shape + (CARRY_HAND_CONTACT_DIM, num_capsules, 2, 4)
    )
    segment_world = hand_body_pos[..., None, None, :] + quat_rotate(
        expanded_quat.reshape(-1, 4),
        expanded_segments.reshape(-1, 3),
        w_last=True,
    ).reshape(leading_shape + (CARRY_HAND_CONTACT_DIM, num_capsules, 2, 3))

    geom_pos = box_geom_pos.to(device=hand_body_pos.device, dtype=hand_body_pos.dtype)
    geom_quat = box_geom_quat.to(device=hand_body_pos.device, dtype=hand_body_pos.dtype)
    expanded_geom_pos = geom_pos.expand(leading_shape + (3,))
    expanded_geom_quat = geom_quat.expand(leading_shape + (4,))
    geom_pos_world = object_pos + quat_rotate(
        object_quat.reshape(-1, 4),
        expanded_geom_pos.reshape(-1, 3),
        w_last=True,
    ).reshape(leading_shape + (3,))
    geom_quat_world = quat_mul(
        object_quat.reshape(-1, 4),
        expanded_geom_quat.reshape(-1, 4),
        w_last=True,
    ).reshape(leading_shape + (4,))
    segment_quat = geom_quat_world[..., None, None, None, :].expand(
        leading_shape + (CARRY_HAND_CONTACT_DIM, num_capsules, 2, 4)
    )
    segment_local = quat_rotate_inverse(
        segment_quat.reshape(-1, 4),
        (segment_world - geom_pos_world[..., None, None, None, :]).reshape(-1, 3),
        w_last=True,
    ).reshape(leading_shape + (CARRY_HAND_CONTACT_DIM, num_capsules, 2, 3))
    half_size = box_half_size.to(device=hand_body_pos.device, dtype=hand_body_pos.dtype)
    centerline_distance = _segment_aabb_distance(
        segment_local[..., 0, :],
        segment_local[..., 1, :],
        half_size,
    )
    return (centerline_distance - hand_contact_radius).amin(dim=-1)


def _segment_aabb_distance(
    segment_start: torch.Tensor,
    segment_end: torch.Tensor,
    half_size: torch.Tensor,
) -> torch.Tensor:
    """Exact Euclidean distance between line segments and a centered AABB."""

    direction = segment_end - segment_start
    half = half_size.to(device=segment_start.device, dtype=segment_start.dtype)
    half = half.expand(segment_start.shape)
    eps = torch.finfo(segment_start.dtype).eps
    safe_direction = torch.where(torch.abs(direction) > eps, direction, torch.ones_like(direction))
    crossings = torch.cat(
        (
            torch.zeros_like(segment_start[..., :1]),
            torch.ones_like(segment_start[..., :1]),
            (-half - segment_start) / safe_direction,
            (half - segment_start) / safe_direction,
        ),
        dim=-1,
    ).clamp(0.0, 1.0)
    crossings, _ = torch.sort(crossings, dim=-1)
    interval_start = crossings[..., :-1]
    interval_end = crossings[..., 1:]
    midpoint = 0.5 * (interval_start + interval_end)
    midpoint_pos = segment_start.unsqueeze(-2) + midpoint.unsqueeze(-1) * direction.unsqueeze(-2)
    lower_active = midpoint_pos < -half.unsqueeze(-2)
    upper_active = midpoint_pos > half.unsqueeze(-2)
    active = lower_active | upper_active
    boundary = torch.where(lower_active, -half.unsqueeze(-2), half.unsqueeze(-2))
    numerator = torch.sum(
        torch.where(active, direction.unsqueeze(-2) * (segment_start.unsqueeze(-2) - boundary), 0.0),
        dim=-1,
    )
    denominator = torch.sum(
        torch.where(active, direction.unsqueeze(-2).square(), 0.0),
        dim=-1,
    )
    stationary = torch.where(denominator > eps, -numerator / denominator, midpoint)
    stationary = torch.maximum(interval_start, torch.minimum(interval_end, stationary))
    candidates = torch.cat((crossings, stationary), dim=-1)
    candidate_pos = segment_start.unsqueeze(-2) + candidates.unsqueeze(-1) * direction.unsqueeze(-2)
    outside = torch.clamp(torch.abs(candidate_pos) - half.unsqueeze(-2), min=0.0)
    return torch.sqrt(torch.min(torch.sum(outside.square(), dim=-1), dim=-1).values)


def compute_hand_box_surface_distance(
    hand_pos: torch.Tensor,
    object_pos: torch.Tensor,
    object_quat: torch.Tensor,
    box_half_size: torch.Tensor,
    box_geom_pos: torch.Tensor,
    box_geom_quat: torch.Tensor,
    *,
    hand_contact_radius: float = 0.0,
) -> torch.Tensor:
    """Return left/right hand collision-surface distance to an oriented box."""

    if hand_pos.ndim != 3 or hand_pos.shape[-2:] != (CARRY_HAND_CONTACT_DIM, 3):
        raise ValueError(f"hand_pos must have shape [T, 2, 3], got {hand_pos.shape}")
    if object_pos.shape != hand_pos.shape[:-2] + (3,):
        raise ValueError(f"object_pos shape {object_pos.shape} is incompatible with hand_pos {hand_pos.shape}")
    if object_quat.shape != hand_pos.shape[:-2] + (4,):
        raise ValueError(f"object_quat shape {object_quat.shape} is incompatible with hand_pos {hand_pos.shape}")
    if hand_contact_radius < 0.0:
        raise ValueError("hand_contact_radius must be non-negative")

    geom_pos = box_geom_pos.to(device=hand_pos.device, dtype=hand_pos.dtype).reshape(1, 3)
    geom_quat = box_geom_quat.to(device=hand_pos.device, dtype=hand_pos.dtype).reshape(1, 4)
    half_size = box_half_size.to(device=hand_pos.device, dtype=hand_pos.dtype).reshape(1, 1, 3)
    marker_count = object_pos.shape[0]
    geom_pos_world = object_pos + quat_rotate(
        object_quat,
        geom_pos.expand(marker_count, -1),
        w_last=True,
    )
    geom_quat_world = quat_mul(object_quat, geom_quat.expand(marker_count, -1), w_last=True)
    expanded_quat = geom_quat_world[:, None, :].expand(-1, CARRY_HAND_CONTACT_DIM, -1)
    hand_local = quat_rotate_inverse(
        expanded_quat.reshape(-1, 4),
        (hand_pos - geom_pos_world[:, None, :]).reshape(-1, 3),
        w_last=True,
    ).reshape_as(hand_pos)
    outside = torch.clamp(torch.abs(hand_local) - half_size, min=0.0)
    return torch.clamp(
        torch.linalg.vector_norm(outside, dim=-1) - hand_contact_radius,
        min=0.0,
    )


def compute_hand_contact_points(
    hand_body_pos: torch.Tensor,
    hand_body_quat: torch.Tensor,
    hand_contact_offsets: torch.Tensor,
) -> torch.Tensor:
    """Transform left/right hand collision-proxy offsets into world positions."""

    if hand_body_pos.shape[-2:] != (CARRY_HAND_CONTACT_DIM, 3):
        raise ValueError(f"hand_body_pos must end in [2, 3], got {hand_body_pos.shape}")
    if hand_body_quat.shape != hand_body_pos.shape[:-1] + (4,):
        raise ValueError(f"hand_body_quat shape {hand_body_quat.shape} is incompatible with {hand_body_pos.shape}")
    offsets = hand_contact_offsets.to(device=hand_body_pos.device, dtype=hand_body_pos.dtype)
    if offsets.shape != (CARRY_HAND_CONTACT_DIM, 3):
        raise ValueError(f"hand_contact_offsets must have shape [2, 3], got {offsets.shape}")
    expanded_offsets = offsets.expand(hand_body_pos.shape[:-2] + offsets.shape)
    rotated_offsets = quat_rotate(
        hand_body_quat.reshape(-1, 4),
        expanded_offsets.reshape(-1, 3),
        w_last=True,
    ).reshape_as(hand_body_pos)
    return hand_body_pos + rotated_offsets


def empty_object_context(reference: torch.Tensor, *, include_hand_spatial: bool = True) -> torch.Tensor:
    """Return the fixed object schema for object-free motion data."""

    context_dim = OBJECT_CONTEXT_DIM if include_hand_spatial else LEGACY_OBJECT_CONTEXT_DIM
    return reference.new_zeros(reference.shape[:-1] + (context_dim,))

from pathlib import Path
from typing import Optional, Sequence, Union

import mujoco
import numpy as np

ROOT_PATH = Path(__file__).parent


def init(
    model: mujoco.MjModel,
    qpos: Optional[np.ndarray] = None,
    qvel: Optional[np.ndarray] = None,
    ctrl: Optional[np.ndarray] = None,
    act: Optional[np.ndarray] = None,
    mocap_pos: Optional[np.ndarray] = None,
    mocap_quat: Optional[np.ndarray] = None,
) -> mujoco.MjData:
    """Initialize Mujoco Data."""
    data = mujoco.MjData(model)
    if qpos is not None:
        data.qpos[:] = qpos
    if qvel is not None:
        data.qvel[:] = qvel
    if ctrl is not None:
        data.ctrl[:] = ctrl
    if act is not None:
        data.act[:] = act
    if mocap_pos is not None:
        data.mocap_pos[:] = mocap_pos.reshape(model.nmocap, -1)
    if mocap_quat is not None:
        data.mocap_quat[:] = mocap_quat.reshape(model.nmocap, -1)
    mujoco.mj_forward(model, data)
    return data


def step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    action: np.ndarray,
    n_substeps: int = 1,
) -> None:
    data.ctrl[:] = action
    mujoco.mj_step(model, data, n_substeps)
    if data.warning.number.any():
        warning_index = np.nonzero(data.warning.number)[0][0]
        warning = mujoco.mjtWarning(warning_index).name
        raise ValueError(f"UNSTABLE MUJOCO. Stopped due to divergence ({warning}).\n")


def get_sensor_adr(model: mujoco.MjModel, sensor_name: str) -> np.ndarray:
    sensor_id = model.sensor(sensor_name).id
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    return np.arange(sensor_adr, sensor_adr + sensor_dim)


def get_sensor_data(model: mujoco.MjModel, data: mujoco.MjData, sensor_name: str) -> np.ndarray:
    """Gets sensor data given sensor name."""
    sensor_id = model.sensor(sensor_name).id
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr : sensor_adr + sensor_dim]


def dof_width(joint_type: Union[int, mujoco.mjtJoint]) -> int:
    """Get the dimensionality of the joint in qvel."""
    if isinstance(joint_type, mujoco.mjtJoint):
        joint_type = joint_type.value
    return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]


def qpos_width(joint_type: Union[int, mujoco.mjtJoint]) -> int:
    """Get the dimensionality of the joint in qpos."""
    if isinstance(joint_type, mujoco.mjtJoint):
        joint_type = joint_type.value
    return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]


def get_qpos_ids(model: mujoco.MjModel, joint_names: Sequence[str]) -> np.ndarray:
    index_list: list[int] = []
    for jnt_name in joint_names:
        jnt = model.joint(jnt_name).id
        jnt_type = model.jnt_type[jnt]
        qadr = model.jnt_dofadr[jnt]
        qdim = qpos_width(jnt_type)
        index_list.extend(range(qadr, qadr + qdim))
    return np.array(index_list)


def get_qvel_ids(model: mujoco.MjModel, joint_names: Sequence[str]) -> np.ndarray:
    index_list: list[int] = []
    for jnt_name in joint_names:
        jnt = model.joint(jnt_name).id
        jnt_type = model.jnt_type[jnt]
        vadr = model.jnt_dofadr[jnt]
        vdim = dof_width(jnt_type)
        index_list.extend(range(vadr, vadr + vdim))
    return np.array(index_list)


__version__ = "0.0.7"
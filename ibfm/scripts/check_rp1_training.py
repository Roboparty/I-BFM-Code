"""Validate RP1 assets and training inputs without allocating a GPU or replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(source_dir=None):
    from omegaconf import OmegaConf
    from humanoidverse.utils.motion_data.adapters import load_ufo_pkl
    from humanoidverse.utils.robot_spec import load_robot_training_spec

    os.chdir(ROOT)
    lock = json.loads((ROOT / "configs/data/rp1_assets.json").read_text())
    for name, expected in lock["config_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen RP1 config changed: {name}. Use a new recipe/lock for experiments.")
    destination = ROOT / lock["cache_directory"]
    if source_dir is not None:
        # Verify every source before installing any data. Existing valid files stay intact.
        for name, expected in lock["sha256"].items():
            source = source_dir / name
            if not source.is_file() or sha256(source) != expected:
                raise ValueError(f"Missing or mismatched RP1 source: {source}")
        destination.mkdir(parents=True, exist_ok=True)
        for name, expected in lock["sha256"].items():
            target = destination / name
            if target.exists():
                if sha256(target) != expected:
                    raise ValueError(f"Refusing to overwrite mismatched asset: {target}")
                continue
            pending = target.with_suffix(f".tmp.{os.getpid()}")
            try:
                shutil.copy2(source_dir / name, pending)
                pending.replace(target)
            finally:
                pending.unlink(missing_ok=True)
    for name, expected in lock["sha256"].items():
        path = destination / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing RP1 data: {path}. Install the six caches with --source-dir.")
        if sha256(path) != expected:
            raise ValueError(f"RP1 data SHA256 mismatch: {path}")

    spec = load_robot_training_spec(ROOT / "configs/robots/rp1_28dof_official_pd.yaml")
    joints = spec.robot.control_joint_names
    if len(joints) != 28:
        raise ValueError(f"RP1 requires 28 actuated joints, got {len(joints)}")
    manifest = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/data/rp1_lafan_carry_kick_30_40_30.yaml"), resolve=True)
    datasets = []
    for item in manifest["datasets"]:
        path = (ROOT / "configs/data" / item["train_path"]).resolve()
        data = load_ufo_pkl(path, source_name=item["name"])
        for key, motion in data.items():
            if motion["pose_aa"].shape[1] != len(joints) + 1:
                raise ValueError(f"Wrong RP1 pose dimension: {path}:{key}")
        datasets.append(dict(name=item["name"], weight=item["weight"], motions=len(data)))
        del data
    return dict(robot=spec.robot.name, joints=len(joints), datasets=datasets,
        expected_eval_rows=sum(d["motions"] for d in datasets),
        total_cache_bytes=sum((destination / name).stat().st_size for name in lock["sha256"]),
        config_sha256=lock["config_sha256"], data_sha256=lock["sha256"],
        data_sha256_verified=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Directory containing the six prepared RP1 caches.")
    args = parser.parse_args()
    print(json.dumps(check(args.source_dir.resolve() if args.source_dir else None), indent=2))

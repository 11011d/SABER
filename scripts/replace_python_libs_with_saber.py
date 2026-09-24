#!/usr/bin/env python3
"""Replace installed CloudVolume/compressed_segmentation with this SABER build."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLOUDFILES_SRC = ROOT / "cloud-files" / "cloudfiles"
CLOUDVOLUME_SRC = ROOT / "cloud-volume" / "cloudvolume"
COMPRESSEDVOXEL_SRC = ROOT / "compressedvoxel.py"


def build_extension(python: str, dry_run: bool) -> None:
    cmd = [python, "setup.py", "build_ext", "--inplace"]
    print("+ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def newest_local_extension(python: str) -> Path:
    candidate_map = {}
    for path in ROOT.glob("compressed_segmentation*.so"):
        candidate_map[path.resolve()] = path
    for path in (ROOT / "build").glob("lib*/compressed_segmentation*.so"):
        candidate_map[path.resolve()] = path

    candidates = sorted(
        candidate_map.values(),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(
            "No local compressed_segmentation extension found after build."
        )

    for candidate in candidates:
        if extension_has_saber_symbols(candidate, python):
            return candidate

    candidate_list = "\n".join(f"  - {path}" for path in candidates)
    raise RuntimeError(
        "Found compressed_segmentation extensions, but none exposes the SABER "
        "symbols PyBlockStore and extract_to_container when imported by the "
        "target Python. Check the build output and target environment ABI.\n"
        f"Checked:\n{candidate_list}"
    )


def extension_has_saber_symbols(path: Path, python: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="saber-cseg-check-") as tmp:
        tmp_dir = Path(tmp)
        shutil.copy2(path, tmp_dir / path.name)
        code = (
            "import compressed_segmentation as cseg; "
            "raise SystemExit(0 if hasattr(cseg, 'PyBlockStore') "
            "and hasattr(cseg, 'extract_to_container') else 1)"
        )
        result = subprocess.run(
            [python, "-c", code],
            cwd=tmp_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0


def copy_path(src: Path, dst: Path, backup_dir: Path | None, dry_run: bool) -> None:
    if dst.exists():
        if backup_dir is None:
            action = f"remove {dst}"
            print(f"+ {action}")
            if not dry_run:
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
        else:
            backup = backup_dir / dst.name
            print(f"+ backup {dst} -> {backup}")
            if not dry_run:
                backup.parent.mkdir(parents=True, exist_ok=True)
                if backup.exists():
                    if backup.is_dir():
                        shutil.rmtree(backup)
                    else:
                        backup.unlink()
                shutil.move(str(dst), str(backup))

    print(f"+ copy {src} -> {dst}")
    if dry_run:
        return
    if src.is_dir():
        shutil.copytree(
            src,
            dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    else:
        shutil.copy2(src, dst)


def remove_old_extensions(python_lib_dir: Path, keep: Path, backup_dir: Path | None, dry_run: bool) -> None:
    for target in python_lib_dir.glob("compressed_segmentation*.so"):
        if target.name == keep.name:
            continue
        if backup_dir is None:
            print(f"+ remove old extension {target}")
            if not dry_run:
                target.unlink()
        else:
            backup = backup_dir / target.name
            print(f"+ backup old extension {target} -> {backup}")
            if not dry_run:
                backup.parent.mkdir(parents=True, exist_ok=True)
                if backup.exists():
                    backup.unlink()
                shutil.move(str(target), str(backup))


def verify_install(python: str, python_lib_dir: Path, dry_run: bool) -> None:
    code = f"""
import cloudvolume
import cloudfiles
import compressed_segmentation as cseg
import compressedvoxel
import inspect
from cloudfiles import CloudFiles
from cloudvolume.saber import SaberOptions
from pathlib import Path
root = Path({str(python_lib_dir)!r}).resolve()
paths = [
    Path(cloudfiles.__file__).resolve(),
    Path(cloudvolume.__file__).resolve(),
    Path(cseg.__file__).resolve(),
    Path(compressedvoxel.__file__).resolve(),
]
for path in paths:
    assert root in path.parents or path.parent == root, path
assert "use_optional_thread" in inspect.signature(CloudFiles.__init__).parameters, inspect.signature(CloudFiles.__init__)
assert hasattr(cseg, "PyBlockStore"), cseg.__file__
assert hasattr(cseg, "extract_to_container"), cseg.__file__
assert SaberOptions.from_params(saber=True).use_compressed_block
print("cloudfiles:", cloudfiles.__file__)
print("cloudvolume:", cloudvolume.__file__)
print("compressed_segmentation:", cseg.__file__)
print("compressedvoxel:", compressedvoxel.__file__)
"""
    cmd = [python, "-c", code]
    print("+ " + " ".join(cmd))
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(python_lib_dir) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env, cwd="/tmp")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy this repository's SABER CloudVolume package and "
            "CloudFiles package plus compressed_segmentation extension into a "
            "concrete Python library directory."
        )
    )
    parser.add_argument(
        "--python-lib-dir",
        required=True,
        help=(
            "Actual Python library directory used by the target environment, "
            "for example /opt/miniconda3/lib/python3.10/site-packages."
        ),
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used for verification.")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Do not rebuild compressed_segmentation before replacement.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Remove existing targets instead of moving them to a timestamped backup directory.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without copying files.")
    args = parser.parse_args()

    python_lib_dir = Path(args.python_lib_dir).expanduser().resolve()
    if not python_lib_dir.exists() or not python_lib_dir.is_dir():
        raise RuntimeError(f"--python-lib-dir is not a directory: {python_lib_dir}")
    if not CLOUDFILES_SRC.exists():
        raise RuntimeError(f"Missing local CloudFiles package: {CLOUDFILES_SRC}")
    if not CLOUDVOLUME_SRC.exists():
        raise RuntimeError(f"Missing local CloudVolume package: {CLOUDVOLUME_SRC}")
    if not COMPRESSEDVOXEL_SRC.exists():
        raise RuntimeError(f"Missing compressedvoxel.py: {COMPRESSEDVOXEL_SRC}")

    if not args.skip_build:
        build_extension(args.python, args.dry_run)

    extension_src = newest_local_extension(args.python)
    backup_dir = None
    if not args.no_backup:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup_dir = python_lib_dir / f"saber-backup-{stamp}"

    extension_dst = python_lib_dir / extension_src.name
    copy_path(CLOUDFILES_SRC, python_lib_dir / "cloudfiles", backup_dir, args.dry_run)
    copy_path(CLOUDVOLUME_SRC, python_lib_dir / "cloudvolume", backup_dir, args.dry_run)
    remove_old_extensions(python_lib_dir, extension_dst, backup_dir, args.dry_run)
    copy_path(extension_src, extension_dst, backup_dir, args.dry_run)
    copy_path(COMPRESSEDVOXEL_SRC, python_lib_dir / "compressedvoxel.py", backup_dir, args.dry_run)
    verify_install(args.python, python_lib_dir, args.dry_run)

    print("SABER library replacement complete.")
    if backup_dir is not None:
        print(f"Backup directory: {backup_dir}")


if __name__ == "__main__":
    main()

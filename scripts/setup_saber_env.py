#!/usr/bin/env python3
"""Build and install the SABER CloudVolume experiment environment."""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLOUD_VOLUME_DIR = ROOT / "cloud-volume"
CLOUD_FILES_DIR = ROOT / "cloud-files"


def run(cmd: list[str], cwd: Path = ROOT, dry_run: bool = False) -> None:
    printable = " ".join(cmd)
    print(f"+ {printable}")
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def find_cloudfiles_py() -> Path:
    spec = importlib.util.find_spec("cloudfiles")
    if spec is None or spec.origin is None:
        raise RuntimeError("cloudfiles is not importable in this Python environment.")

    package_dir = Path(spec.origin).resolve().parent
    cloudfiles_py = package_dir / "cloudfiles.py"
    if not cloudfiles_py.exists():
        raise RuntimeError(f"Could not find cloudfiles.py at {cloudfiles_py}")
    return cloudfiles_py


def patch_cloudfiles(dry_run: bool = False) -> Path:
    """Patch local file:// reads to use the caller thread instead of a pool."""

    target = find_cloudfiles_py()
    text = target.read_text()

    replacements = [
        (
            re.compile(r'(if self\.protocol == "file":\n\s*)num_threads = 1'),
            r"\1num_threads = 0",
        ),
        (
            re.compile(r"(if self\.protocol == 'file':\n\s*)num_threads = 1"),
            r"\1num_threads = 0",
        ),
    ]

    if re.search(r'if self\.protocol == ["\']file["\']:\n\s*num_threads = 0', text):
        print(f"cloudfiles already patched: {target}")
        return target

    patched = text
    for pattern, replacement in replacements:
        patched, count = pattern.subn(replacement, patched, count=1)
        if count:
            break
    else:
        print(
            "cloudfiles patch was not applied because this installed version does not "
            f"contain the old file:// num_threads block: {target}"
        )
        return target

    backup = target.with_suffix(target.suffix + ".saber.bak")
    prefix = "+ would backup" if dry_run else "+ backup"
    print(f"{prefix} {target} -> {backup}")
    if not dry_run:
        shutil.copy2(target, backup)
        target.write_text(patched)
    print(f"{'would patch' if dry_run else 'patched'} cloudfiles: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install local CloudVolume, build compressed_segmentation, and optionally patch cloudfiles."
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--skip-cloudfiles", action="store_true", help="Do not pip install ./cloud-files.")
    parser.add_argument("--skip-cloudvolume", action="store_true", help="Do not pip install ./cloud-volume.")
    parser.add_argument("--skip-build", action="store_true", help="Do not build compressed_segmentation in place.")
    parser.add_argument("--patch-cloudfiles", action="store_true", help="Patch site-packages cloudfiles file:// reads to num_threads=0.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing them.")
    args = parser.parse_args()

    if not args.skip_cloudfiles:
        if not CLOUD_FILES_DIR.exists():
            raise RuntimeError(f"Missing CloudFiles checkout: {CLOUD_FILES_DIR}")
        run([args.python, "-m", "pip", "install", "-e", str(CLOUD_FILES_DIR)], dry_run=args.dry_run)

    if not args.skip_cloudvolume:
        if not CLOUD_VOLUME_DIR.exists():
            raise RuntimeError(f"Missing CloudVolume checkout: {CLOUD_VOLUME_DIR}")
        run([args.python, "-m", "pip", "install", "-e", str(CLOUD_VOLUME_DIR)], dry_run=args.dry_run)

    if not args.skip_build:
        run([args.python, "setup.py", "build_ext", "--inplace"], dry_run=args.dry_run)

    if args.patch_cloudfiles:
        patch_cloudfiles(dry_run=args.dry_run)

    if not args.dry_run:
        run(
            [
                args.python,
                "-c",
                (
                    "import compressed_segmentation as cseg; "
                    "assert hasattr(cseg, 'PyBlockStore'), cseg.__file__; "
                    "assert hasattr(cseg, 'extract_to_container'), cseg.__file__; "
                    "print('compressed_segmentation OK:', cseg.__file__)"
                ),
            ],
            dry_run=False,
        )

    print("SABER environment setup complete.")


if __name__ == "__main__":
    main()

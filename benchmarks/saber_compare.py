#!/usr/bin/env python3
"""Correctness and performance comparison for SABER compressed-block reads."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cc3d
import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
LOCAL_CLOUD_VOLUME = ROOT / "cloud-volume"
sys.path.insert(0, str(ROOT))
if LOCAL_CLOUD_VOLUME.exists():
    sys.path.insert(0, str(LOCAL_CLOUD_VOLUME))

from cloudvolume import CloudVolume  # noqa: E402
import compressed_segmentation as cseg  # noqa: E402


if Path(cseg.__file__).resolve().parent != ROOT:
    raise ImportError(
        "benchmarks/saber_compare.py must use the local compressed_segmentation "
        f"extension from {ROOT}; got {cseg.__file__}"
    )


@dataclass
class StageTiming:
    fetch: float = 0.0
    where: float = 0.0
    connected_component: float = 0.0
    boundary: float = 0.0
    total: float = 0.0


@dataclass
class RunResult:
    mode: str
    points: int
    cases: int
    timing: StageTiming


def nearest_nonzero_idx(a: np.ndarray, x: int, y: int, z: int) -> tuple[np.ndarray, bool]:
    idx = np.argwhere(a)
    idx = idx[~(idx == [x, y, z]).all(1)]
    dists = ((idx - [x, y, z]) ** 2).sum(1)
    min_dist = dists.min()
    is_unique = int((dists == min_dist).sum()) == 1
    return idx[dists.argmin()], is_unique


def parse_coord(value: str) -> tuple[int, int, int]:
    coord = np.array(value.strip("[]").split(","), dtype=float)
    x = int(np.round(coord[0] / 4) * 4)
    y = int(np.round(coord[1] / 4) * 4)
    z = int(np.round(coord[2]))
    return x, y, z


def scan_valid_indices(vol: CloudVolume, df: pd.DataFrame, lx: int, ly: int, lz: int) -> list[int]:
    valid = []
    for ii in tqdm(range(len(df)), desc="scan valid cases"):
        segid1 = df.iloc[ii, 0]
        segid2 = df.iloc[ii, 1]
        x, y, z = parse_coord(df.iloc[ii, 2])
        x0, x1 = int(x / 4 - lx), int(x / 4 + lx)
        y0, y1 = int(y / 4 - ly), int(y / 4 + ly)
        z0, z1 = int(z - lz), int(z + lz)

        vol_cutout = vol[x0:x1, y0:y1, z0:z1]

        vol0 = np.where(vol_cutout == segid1, 255, 0).astype(np.uint8)
        cc0 = cc3d.connected_components(vol0[:, :, :, 0], out_dtype=np.uint64)
        _, u0 = nearest_nonzero_idx(cc0, lx, ly, lz)

        vol1 = np.where(vol_cutout == segid2, 255, 0).astype(np.uint8)
        cc1 = cc3d.connected_components(vol1[:, :, :, 0], out_dtype=np.uint64)
        _, u1 = nearest_nonzero_idx(cc1, lx, ly, lz)

        if u0 and u1:
            valid.append(ii)
    return valid


def extract_boundaries_dense(mask: np.ndarray, origin: tuple[float, float, float], label: int):
    """Dense reference for 3D 6-neighbor boundary voxel point clouds."""
    pc_data = []
    ids_data = []
    foreground = mask != 0
    if not np.any(foreground):
        return pc_data, ids_data

    padded = np.pad(foreground, 1, mode="constant", constant_values=False)
    all_six_neighbors = (
        padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    boundary = padded[1:-1, 1:-1, 1:-1] & ~all_six_neighbors
    coords = np.argwhere(boundary)
    if coords.size == 0:
        return pc_data, ids_data

    points = np.empty((coords.shape[0], 3), dtype=np.float32)
    points[:, 0] = coords[:, 0] + origin[0]
    points[:, 1] = coords[:, 1] + origin[1]
    points[:, 2] = coords[:, 2] + origin[2]
    pc_data.append(points)
    ids_data.extend([label] * coords.shape[0])
    return pc_data, ids_data


def run_mode(vol: CloudVolume, df: pd.DataFrame, indices: list[int], compressed: bool, lx: int, ly: int, lz: int):
    timing = StageTiming()
    all_pc = []
    all_ids = []
    processed = 0
    started = time.perf_counter()

    for ii in tqdm(indices, desc=f"run {'compressed' if compressed else 'dense'}"):
        pc_data = []
        ids_data = []
        segid1 = df.iloc[ii, 0]
        segid2 = df.iloc[ii, 1]
        x, y, z = parse_coord(df.iloc[ii, 2])
        x0, x1 = int(x / 4 - lx), int(x / 4 + lx)
        y0, y1 = int(y / 4 - ly), int(y / 4 + ly)
        z0, z1 = int(z - lz), int(z + lz)
        origin = (float(x0), float(y0), float(z0))

        t0 = time.perf_counter()
        if compressed:
            vol.segid_list = [segid1, segid2]
        cutout = vol[x0:x1, y0:y1, z0:z1]
        timing.fetch += time.perf_counter() - t0

        masks = []
        for segid in (segid1, segid2):
            t0 = time.perf_counter()
            if compressed:
                mask = cutout.where(segid, 255, 0, out_dtype=np.uint8)
            else:
                mask = np.where(cutout == segid, 255, 0).astype(np.uint8)
            timing.where += time.perf_counter() - t0

            t0 = time.perf_counter()
            if compressed:
                mask = mask.keep_nearest_connected_component_optimized(lx, ly, lz)
            else:
                cc = cc3d.connected_components(mask[:, :, :, 0], out_dtype=np.uint64)
                nn_idx, _ = nearest_nonzero_idx(cc, lx, ly, lz)
                relabel = cc[tuple(nn_idx)]
                mask = np.where(cc == relabel, 255, 0).astype(np.uint8)
            timing.connected_component += time.perf_counter() - t0
            masks.append(mask)

        for label, mask in enumerate(masks):
            t0 = time.perf_counter()
            if compressed:
                mask.extract_boundary_voxel_points_3d(origin, pc_data, ids_data, label=label)
            else:
                pcs, ids = extract_boundaries_dense(mask, origin, label)
                pc_data.extend(pcs)
                ids_data.extend(ids)
            timing.boundary += time.perf_counter() - t0

        if pc_data:
            all_pc.append(np.vstack(pc_data))
            all_ids.append(np.array(ids_data))
            processed += 1

    timing.total = time.perf_counter() - started
    pc = np.vstack(all_pc) if all_pc else np.zeros((0, 3))
    ids = np.concatenate(all_ids) if all_ids else np.array([])
    return pc, ids, RunResult("compressed" if compressed else "dense", len(pc), processed, timing)


def compare_results(pc1: np.ndarray, ids1: np.ndarray, pc2: np.ndarray, ids2: np.ndarray) -> bool:
    if len(pc1) != len(pc2):
        return False
    order1 = np.lexsort(pc1[:, ::-1].T)
    order2 = np.lexsort(pc2[:, ::-1].T)
    return bool(np.allclose(pc1[order1], pc2[order2], atol=1e-3) and np.array_equal(ids1[order1], ids2[order2]))


def parse_indices(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    path = Path(value)
    if path.exists():
        return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SABER compressed-block mode against dense CloudVolume mode.")
    parser.add_argument("--cloudpath", default=os.environ.get("SABER_CLOUDPATH"))
    parser.add_argument("--candidate-file", default=str(ROOT / "data" / "candidate0.csv"))
    parser.add_argument("--indices", help="Comma-separated indices, or a file containing one index per line.")
    parser.add_argument("--scan-valid", action="store_true", help="Scan all candidates for unique nearest connected components.")
    parser.add_argument("--lx", type=int, default=80)
    parser.add_argument("--ly", type=int, default=80)
    parser.add_argument("--lz", type=int, default=32)
    parser.add_argument("--cache-bytes", type=int, default=1024 * 1024 * 100)
    parser.add_argument("--json-out", help="Write machine-readable results to this JSON file.")
    args = parser.parse_args()
    if not args.cloudpath:
        parser.error("--cloudpath is required unless SABER_CLOUDPATH is set")

    df = pd.read_csv(args.candidate_file)
    indices = parse_indices(args.indices)
    if indices is None:
        indices = [8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 97, 98, 99, 100, 106, 107, 108, 109, 110, 111, 113, 114]

    if args.scan_valid:
        scan_vol = CloudVolume(args.cloudpath, mip=0, fill_missing=True, cache=True, lru_bytes=args.cache_bytes)
        indices = scan_valid_indices(scan_vol, df, args.lx, args.ly, args.lz)

    compressed_vol = CloudVolume(
        args.cloudpath,
        mip=0,
        fill_missing=True,
        cache=True,
        saber=True,
        cache_thread=0,
        lru_bytes=args.cache_bytes,
    )
    pc_compressed, ids_compressed, compressed_result = run_mode(compressed_vol, df, indices, True, args.lx, args.ly, args.lz)

    dense_vol = CloudVolume(
        args.cloudpath,
        mip=0,
        fill_missing=True,
        cache=True,
        lru_bytes=args.cache_bytes,
    )
    pc_dense, ids_dense, dense_result = run_mode(dense_vol, df, indices, False, args.lx, args.ly, args.lz)

    passed = compare_results(pc_compressed, ids_compressed, pc_dense, ids_dense)
    output = {
        "passed": passed,
        "indices": indices,
        "compressed": asdict(compressed_result),
        "dense": asdict(dense_result),
    }
    print(json.dumps(output, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(output, indent=2) + "\n")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

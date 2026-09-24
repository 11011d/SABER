import os
import sys

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import compressed_segmentation as cseg  # noqa: E402


def test_block_store_storage_stats():
    store = cseg.PyBlockStore(3, np.uint64)
    store.set_block(0, np.array([7], dtype=np.uint64), 0, None)
    store.set_block(
        1,
        np.array([0, 7], dtype=np.uint64),
        1,
        np.array([0xA5A5A5A5, 0x5A5A5A5A], dtype=np.uint32),
    )

    assert store.storage_stats() == {
        "total_block_slots": 3,
        "resident_blocks": 2,
        "palette_bytes": 24,
        "bitstream_bytes": 8,
        "payload_bytes": 32,
    }
    retained = store.retained_storage_stats()
    assert retained["palette_capacity_bytes"] >= 24
    assert retained["bitstream_capacity_bytes"] >= 8
    assert retained["metadata_capacity_bytes"] > 0
    assert retained["retained_native_bytes"] == sum(
        retained[key] for key in (
            "palette_capacity_bytes", "bitstream_capacity_bytes", "metadata_capacity_bytes"
        )
    )


def test_extract_boundary_voxels_3d_store_full_cube():
    block_size = (2, 2, 2)
    grid_size = (2, 2, 2)
    store = cseg.PyBlockStore(8, np.uint8)
    full_palette = np.array([255], dtype=np.uint8)

    for idx in range(8):
        store.set_block(idx, full_palette, 0, None)

    origin = np.asarray((13.25, -7.5, 101.125), dtype=np.float32)
    points, ids = cseg.extract_boundary_voxels_3d_store(
        store,
        grid_size,
        block_size,
        (4, 4, 4),
        (0, 0, 0),
        tuple(origin.tolist()),
        7,
        np.uint8,
    )

    voxel_coords = {
        tuple(np.rint(point - origin).astype(np.int64).tolist())
        for point in points
    }

    expected = {
        (x, y, z)
        for x in range(4)
        for y in range(4)
        for z in range(4)
        if x in (0, 3) or y in (0, 3) or z in (0, 3)
    }

    assert points.dtype == np.float32
    assert ids.dtype == np.int32
    assert voxel_coords == expected
    assert set(ids.tolist()) == {7}


def test_extract_boundary_voxels_3d_store_mixed_block():
    store = cseg.PyBlockStore(1, np.uint8)
    palette = np.array([0, 255], dtype=np.uint8)
    bitstream = np.array([1 << 1], dtype=np.uint32)
    store.set_block(0, palette, 1, bitstream)

    origin = (3.5, -1.25, 7.0)
    points, ids = cseg.extract_boundary_voxels_3d_store(
        store,
        (1, 1, 1),
        (2, 2, 2),
        (2, 2, 2),
        (0, 0, 0),
        origin,
        3,
        np.uint8,
    )

    assert points.tolist() == [[4.5, -1.25, 7.0]]
    assert ids.tolist() == [3]

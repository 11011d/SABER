import os
import sys

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import compressed_segmentation as cseg  # noqa: E402
from compressedvoxel import CompressedVoxelContainer  # noqa: E402


class Box:
    def __init__(self, minpt, maxpt):
        self.minpt = np.asarray(minpt, dtype=np.int64)
        self.maxpt = np.asarray(maxpt, dtype=np.int64)

    def size3(self):
        return self.maxpt - self.minpt


def full_mask_container(origin):
    bbox = Box(origin, np.asarray(origin, dtype=np.int64) + (4, 4, 4))
    container = CompressedVoxelContainer(bbox, bbox, (2, 2, 2), np.uint8)
    palette = np.asarray([255], dtype=np.uint8)
    for index in range(8):
        container.blocks.set_block(index, palette, 0, None)
    return container


def local_point_set(chunks, origin):
    points = np.concatenate(chunks, axis=0)
    return {
        tuple(np.rint(point - origin).astype(np.int64).tolist())
        for point in points
    }


def test_extract_boundary_points_restores_legacy_signature_with_bbox_origin():
    bbox_origin = np.asarray((13, -7, 101), dtype=np.float32)
    container = full_mask_container(bbox_origin.astype(np.int64))

    chunks, ids = container.extract_boundary_points(
        999,
        -999,
        123,
        7,
        11,
        13,
        [],
        [],
        label=5,
    )

    points = np.concatenate(chunks, axis=0)
    local_points = local_point_set(chunks, bbox_origin)
    assert points.dtype == np.float32
    assert set(ids) == {5}
    assert {point[2] for point in local_points} == {0, 1, 2, 3}
    assert all(0 <= point[0] < 4 and 0 <= point[1] < 4 for point in local_points)


def test_extract_boundary_points_uses_explicit_voxel_origin_without_scaling():
    container = full_mask_container((13, -7, 101))
    origin = np.asarray((0.25, 31.5, -9.75), dtype=np.float32)

    chunks, ids = container.extract_boundary_points(
        0,
        0,
        0,
        0,
        0,
        0,
        [],
        [],
        label=8,
        origin=origin,
    )

    points = np.concatenate(chunks, axis=0)
    local_points = local_point_set(chunks, origin)
    assert set(ids) == {8}
    assert {point[2] for point in local_points} == {0, 1, 2, 3}
    assert all(0 <= point[0] < 4 and 0 <= point[1] < 4 for point in local_points)
    assert any(np.isclose(point[0], origin[0]) for point in points)
    assert any(np.isclose(point[1], origin[1]) for point in points)
    assert any(np.isclose(point[2], origin[2]) for point in points)

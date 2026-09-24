import os
import sys

import numpy as np
import pytest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from compressedvoxel import CompressedVoxelContainer  # noqa: E402


class Box:
    def __init__(self, minpt, maxpt):
        self.minpt = np.asarray(minpt, dtype=np.int64)
        self.maxpt = np.asarray(maxpt, dtype=np.int64)

    def size3(self):
        return self.maxpt - self.minpt


def encoded_block(block, dtype):
    values = np.asarray(block, dtype=dtype).reshape(-1, order="F")
    palette, indices = np.unique(values, return_inverse=True)
    if palette.size == 1:
        return palette, 0, None

    bits = max(1, int(np.ceil(np.log2(palette.size))))
    words = np.zeros((values.size * bits + 31) // 32, dtype=np.uint32)
    mask = (1 << bits) - 1
    for voxel_index, palette_index in enumerate(indices):
        bit_position = voxel_index * bits
        word = bit_position // 32
        shift = bit_position % 32
        value = int(palette_index) & mask
        words[word] |= np.uint32(value << shift)
        if shift + bits > 32:
            words[word + 1] |= np.uint32(value >> (32 - shift))
    return palette.astype(dtype), bits, words


def container_from_dense(full_xyz, request_min, request_max, block_size):
    full_xyz = np.asarray(full_xyz)
    full_bbox = Box((0, 0, 0), full_xyz.shape)
    request_bbox = Box(request_min, request_max)
    container = CompressedVoxelContainer(
        request_bbox, full_bbox, block_size, full_xyz.dtype
    )
    grid_size = np.asarray(full_xyz.shape) // np.asarray(block_size)
    for gz in range(grid_size[2]):
        for gy in range(grid_size[1]):
            for gx in range(grid_size[0]):
                lower = np.asarray((gx, gy, gz)) * block_size
                upper = lower + block_size
                block = full_xyz[
                    lower[0] : upper[0],
                    lower[1] : upper[1],
                    lower[2] : upper[2],
                ]
                palette, bits, bitstream = encoded_block(block, full_xyz.dtype)
                block_index = gx + gy * grid_size[0] + gz * grid_size[0] * grid_size[1]
                container.blocks.set_block(int(block_index), palette, bits, bitstream)
    return container


@pytest.mark.parametrize("dtype", (np.uint8, np.uint32, np.uint64))
def test_distinct_labels_in_mask_matches_dense_reference(dtype):
    full = np.zeros((8, 8, 4), dtype=dtype)
    full[0:4, 0:4, 0:2] = 7
    full[4:8, 0:4, 0:2] = 13
    full[0:4, 4:8, 0:2] = 21
    full[4:8, 4:8, 0:2] = 34
    full[1:7, 1:7, 2:4] = 55

    request_min = np.asarray((1, 1, 1))
    request_max = np.asarray((8, 7, 4))
    request = full[
        request_min[0] : request_max[0],
        request_min[1] : request_max[1],
        request_min[2] : request_max[2],
    ]
    selection = np.zeros(request.shape, dtype=bool)
    selection[0:3, 0:3, 0] = True
    selection[3:7, 0:3, 0] = True
    selection[0:2, 3:6, 0] = True
    selection[1:6, 1:5, 1:3] = np.indices((5, 4, 2)).sum(axis=0) % 2 == 0
    selection[1:3, 1:3, 1:3] = True

    container = container_from_dense(full, request_min, request_max, (2, 2, 2))
    labels, metrics = container.distinct_labels_in_mask(selection)

    expected = np.unique(request[selection])
    assert labels.dtype == dtype
    assert np.array_equal(labels, expected)
    assert metrics["selected_voxels"] == int(selection.sum())
    assert metrics["palette_only_blocks"] >= 1
    assert metrics["partially_decoded_blocks"] >= 1
    assert metrics["skipped_blocks"] >= 1


def test_distinct_labels_in_mask_validates_shape_and_dtype():
    full = np.zeros((4, 4, 4), dtype=np.uint64)
    container = container_from_dense(full, (0, 0, 0), (4, 4, 4), (2, 2, 2))

    with pytest.raises(ValueError, match="mask shape"):
        container.distinct_labels_in_mask(np.zeros((4, 4, 3), dtype=bool))
    with pytest.raises(TypeError, match="boolean"):
        container.distinct_labels_in_mask(np.zeros((4, 4, 4), dtype=np.float32))

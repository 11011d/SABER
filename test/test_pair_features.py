import os
import sys

import numpy as np
import pytest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import compressed_segmentation as cseg  # noqa: E402
from compressedvoxel import (  # noqa: E402
    CompressedVoxelContainer,
    extract_pair_features_batch,
)


class Box:
    def __init__(self, minpt, maxpt):
        self.minpt = np.asarray(minpt, dtype=np.int64)
        self.maxpt = np.asarray(maxpt, dtype=np.int64)

    def size3(self):
        return self.maxpt - self.minpt


def _encoded_block(block, dtype):
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


def _container_from_dense(full_xyz, request_min, request_max, block_size):
    full_xyz = np.asarray(full_xyz)
    full_bbox = Box((0, 0, 0), full_xyz.shape)
    request_bbox = Box(request_min, request_max)
    container = CompressedVoxelContainer(request_bbox, full_bbox, block_size, full_xyz.dtype)
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
                palette, bits, bitstream = _encoded_block(block, full_xyz.dtype)
                block_index = gx + gy * grid_size[0] + gz * grid_size[0] * grid_size[1]
                container.blocks.set_block(int(block_index), palette, bits, bitstream)
    return container


def _container_from_encoded(encoded, shape, dtype, block_size, segid_list):
    bbox = Box((0, 0, 0), shape)
    container = CompressedVoxelContainer(bbox, bbox, block_size, dtype)
    cseg.extract_to_container(
        encoded,
        shape,
        dtype,
        (0, 0, 0),
        shape,
        (0, 0, 0),
        shape,
        container,
        (0, 0, 0),
        block_size=block_size,
        segid_list=segid_list,
    )
    return container


def _dense_pair_feature(request_xyz, label_one, label_two, output_shape_zyx):
    request_zyx = np.transpose(request_xyz, (2, 1, 0))
    masks = np.stack((request_zyx == label_one, request_zyx == label_two)).astype(np.uint8)
    source_shape = np.asarray(request_zyx.shape, dtype=np.int64)
    target_shape = np.asarray(output_shape_zyx, dtype=np.int64)
    indices = [
        np.floor(np.arange(target_shape[axis]) * source_shape[axis] / target_shape[axis]).astype(np.int64)
        for axis in range(3)
    ]
    masks = masks[:, indices[0]][:, :, indices[1]][:, :, :, indices[2]]
    return np.concatenate((np.logical_or(masks[0], masks[1])[None], masks)).astype(np.uint8)


def _fixture(dtype, offset=0):
    values = np.zeros((8, 8, 4), dtype=dtype)
    values[1:5, 1:6, 1:3] = 7 + offset
    values[4:7, 2:7, 0:4] = 13 + offset
    values[2:4, 5:8, 2:4] = 21 + offset
    request_min = np.asarray((1, 1, 1))
    request_max = np.asarray((8, 7, 4))
    container = _container_from_dense(values, request_min, request_max, np.asarray((2, 2, 2)))
    request = values[
        request_min[0] : request_max[0],
        request_min[1] : request_max[1],
        request_min[2] : request_max[2],
    ]
    return container, request, 7 + offset, 13 + offset


@pytest.mark.parametrize("dtype", (np.uint8, np.uint32, np.uint64))
@pytest.mark.parametrize("output_shape", ((3, 5, 4), (7, 4, 9)))
def test_extract_pair_feature_matches_dense_floor_index_reference(dtype, output_shape):
    container, request, label_one, label_two = _fixture(dtype)

    actual = container.extract_pair_feature(label_one, label_two, output_shape)
    expected = _dense_pair_feature(request, label_one, label_two, output_shape)

    assert actual.dtype == np.uint8
    assert actual.flags.c_contiguous
    assert actual.shape == (3,) + output_shape
    assert np.array_equal(actual, expected)


def test_extract_pair_feature_handles_equal_labels():
    container, request, label_one, _ = _fixture(np.uint64)

    actual = container.extract_pair_feature(label_one, label_one, (4, 6, 5))
    expected = _dense_pair_feature(request, label_one, label_one, (4, 6, 5))

    assert np.array_equal(actual, expected)
    assert np.array_equal(actual[0], actual[1])
    assert np.array_equal(actual[1], actual[2])


@pytest.mark.parametrize("dtype", (np.uint8, np.uint32, np.uint64))
def test_project_pair_feature_uses_composed_query_operator(dtype):
    container, request, label_one, label_two = _fixture(dtype)
    output_shape = (5, 7, 6)

    actual, physical_plan = container.project_pair_feature(
        label_one, label_two, output_shape, return_plan=True
    )
    expected = _dense_pair_feature(request, label_one, label_two, output_shape)

    assert physical_plan == "pair3-fused"
    assert np.array_equal(actual, expected)


def test_project_label_channels_supports_generic_channel_groups():
    container, request, label_one, label_two = _fixture(np.uint64)
    third_label = 21
    output_shape = tuple(np.transpose(request, (2, 1, 0)).shape)

    actual, physical_plan = container.project_label_channels(
        (label_one, label_two, third_label),
        ((1, 0, 0), (0, 1, 1), (1, 1, 1)),
        output_shape,
        return_plan=True,
    )
    request_zyx = np.transpose(request, (2, 1, 0))
    expected = np.stack(
        (
            request_zyx == label_one,
            np.isin(request_zyx, (label_two, third_label)),
            np.isin(request_zyx, (label_one, label_two, third_label)),
        )
    ).astype(np.uint8)

    assert physical_plan == "generic-label-channel-gather"
    assert np.array_equal(actual, expected)


def test_project_label_channels_zero_fills_below_volume_boundary():
    full = np.zeros((8, 8, 4), dtype=np.uint64)
    full[1:5, 1:6, 0:3] = 7
    full[4:7, 2:7, 0:4] = 13
    full[2:4, 5:8, 1:4] = 21
    request_min = np.asarray((1, 1, -1))
    request_max = np.asarray((8, 7, 3))
    container = _container_from_dense(
        full, request_min, request_max, np.asarray((2, 2, 2))
    )

    actual, physical_plan = container.project_label_channels(
        (7, 13, 21),
        ((1, 0, 0), (0, 1, 1), (1, 1, 1)),
        (4, 6, 7),
        return_plan=True,
    )
    request = np.zeros((7, 6, 4), dtype=full.dtype)
    request[:, :, 1:] = full[1:8, 1:7, 0:3]
    request_zyx = np.transpose(request, (2, 1, 0))
    expected = np.stack(
        (
            request_zyx == 7,
            np.isin(request_zyx, (13, 21)),
            np.isin(request_zyx, (7, 13, 21)),
        )
    ).astype(np.uint8)

    assert physical_plan == "generic-label-channel-gather"
    assert np.array_equal(actual, expected)


def _sorted_points(points):
    points = np.asarray(points, dtype=np.int64)
    if points.size == 0:
        return points.reshape(0, 3)
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    return points[order]


@pytest.mark.parametrize("dtype", (np.uint8, np.uint32, np.uint64))
def test_composed_multilabel_contour_matches_existing_operator_chain(dtype):
    container, _, label_one, label_two = _fixture(dtype)
    labels = (label_one, label_two)
    actual, metrics = container.extract_label_contours_3d(labels)

    expected = []
    for label in labels:
        mask = container.where(label, 255, 0, out_dtype=np.uint8)
        chunks = []
        ids = []
        mask.extract_boundary_voxel_points_3d((0, 0, 0), chunks, ids)
        expected.append(np.vstack(chunks) if chunks else np.empty((0, 3), dtype=np.int64))

    assert metrics["physical_plan"] == "multilabel-boundary6-per-label"
    assert len(actual) == len(expected)
    for actual_points, expected_points in zip(actual, expected):
        assert np.array_equal(_sorted_points(actual_points), _sorted_points(expected_points))


@pytest.mark.parametrize("dtype", (np.uint8, np.uint32, np.uint64))
def test_composed_filtered_contour_matches_existing_operator_chain(dtype):
    container, _, label_one, label_two = _fixture(dtype)
    labels = (label_one, label_two)
    center = np.asarray((3, 3, 1), dtype=np.int64)
    actual, metrics = container.extract_label_contours_3d(labels, centers=center)

    expected = []
    for label in labels:
        mask = container.where(label, 255, 0, out_dtype=np.uint8)
        mask.keep_nearest_connected_component_optimized(*center)
        chunks = []
        ids = []
        mask.extract_boundary_voxel_points_3d((0, 0, 0), chunks, ids)
        expected.append(np.vstack(chunks) if chunks else np.empty((0, 3), dtype=np.int64))

    assert metrics["physical_plan"] == "label-component26-boundary6"
    assert len(actual) == len(expected)
    for actual_points, expected_points in zip(actual, expected):
        assert np.array_equal(_sorted_points(actual_points), _sorted_points(expected_points))


def test_extract_to_container_uses_actual_non_power_of_two_palette_size():
    dense = np.ones((4, 4, 4), dtype=np.uint64)
    dense[1:3, 1:3, 1:3] = 11
    dense[2:, 2:, 2:] = 22
    encoded = cseg.compress(dense, block_size=(4, 4, 4), order="C")

    container = _container_from_encoded(
        encoded, dense.shape, dense.dtype, (4, 4, 4), [11, 22]
    )

    assert np.array_equal(container.get_all_blocks_dense(), dense)
    assert np.array_equal(
        container.extract_pair_feature(11, 22, (4, 4, 4)),
        _dense_pair_feature(dense, 11, 22, (4, 4, 4)),
    )


def test_extract_to_container_rejects_truncated_palette():
    dense = np.ones((4, 4, 4), dtype=np.uint64)
    dense[1:3, 1:3, 1:3] = 11
    dense[2:, 2:, 2:] = 22
    encoded = cseg.compress(dense, block_size=(4, 4, 4), order="C")

    with pytest.raises(RuntimeError, match="palette exceeds input buffer"):
        _container_from_encoded(
            encoded[:-4], dense.shape, dense.dtype, (4, 4, 4), [11, 22]
        )


@pytest.mark.parametrize("parallel", (0, 1, 2, 8))
def test_extract_pair_features_batch_matches_single_query_outputs(parallel):
    first_container, first_request, first_a, first_b = _fixture(np.uint64)
    second_container, second_request, second_a, second_b = _fixture(np.uint64, offset=1000)
    output_shape = (5, 7, 6)

    actual = extract_pair_features_batch(
        (
            (first_container, first_a, first_b),
            (second_container, second_a, second_b),
        ),
        output_shape,
        parallel=parallel,
    )
    expected = np.stack(
        (
            _dense_pair_feature(first_request, first_a, first_b, output_shape),
            _dense_pair_feature(second_request, second_a, second_b, output_shape),
        )
    )

    assert actual.dtype == np.uint8
    assert actual.flags.c_contiguous
    assert actual.shape == (2, 3) + output_shape
    assert np.array_equal(actual, expected)


def test_extract_pair_features_batch_accepts_empty_input():
    actual = extract_pair_features_batch([], (4, 5, 6), parallel=4)
    assert actual.shape == (0, 3, 4, 5, 6)
    assert actual.dtype == np.uint8


def test_extract_pair_feature_rejects_physical_or_invalid_parameters():
    container, _, label_one, label_two = _fixture(np.uint32)
    with pytest.raises(ValueError, match="positive"):
        container.extract_pair_feature(label_one, label_two, (0, 5, 6))
    with pytest.raises(ValueError, match="positive"):
        container.extract_pair_feature(0, label_two, (4, 5, 6))
    with pytest.raises(ValueError, match="nonnegative"):
        extract_pair_features_batch([(container, label_one, label_two)], (4, 5, 6), parallel=-1)
    with pytest.raises(TypeError, match="CompressedVoxelContainer"):
        extract_pair_features_batch([(object(), label_one, label_two)], (4, 5, 6))
    with pytest.raises(ValueError, match="representable"):
        extract_pair_features_batch([(container, -1, label_two)], (4, 5, 6))
    with pytest.raises(ValueError, match="representable"):
        extract_pair_features_batch([(container, 2**32, label_two)], (4, 5, 6))


@pytest.mark.parametrize("dtype", (np.uint8, np.uint32, np.uint64))
def test_composable_binary_pair_feature_matches_dense_reference(dtype):
    container, request, label_one, label_two = _fixture(dtype)
    output_shape = (7, 4, 9)

    mask_one = container.binary_mask(label_one)
    mask_two = container.binary_mask(label_two)
    union = mask_one.binary_union(mask_two)
    actual = union.materialize_binary_channels(mask_one, mask_two, output_shape)
    expected = _dense_pair_feature(request, label_one, label_two, output_shape)

    assert mask_one.dtype == np.dtype(np.uint8)
    assert mask_two.dtype == np.dtype(np.uint8)
    assert union.dtype == np.dtype(np.uint8)
    assert np.array_equal(actual, expected)
    assert np.array_equal(
        mask_one.get_raw_data((0, 0, 0), tuple(mask_one.requested_bbox.size3())),
        (request == label_one).astype(np.uint8),
    )
    assert np.array_equal(
        mask_two.get_raw_data((0, 0, 0), tuple(mask_two.requested_bbox.size3())),
        (request == label_two).astype(np.uint8),
    )


def test_binary_union_copies_disjoint_blocks_and_merges_overlapping_blocks():
    left = np.zeros((8, 8, 4), dtype=np.uint8)
    right = np.zeros_like(left)
    left[0:2, 0:2, 0:2] = 1
    right[1:3, 1:3, 1:3] = 1
    left[4:6, 4:6, 2:4] = 1
    block_size = np.asarray((4, 4, 2))
    lower = np.asarray((0, 0, 0))
    upper = np.asarray(left.shape)
    left_container = _container_from_dense(left, lower, upper, block_size)
    right_container = _container_from_dense(right, lower, upper, block_size)

    union = left_container.binary_union(right_container)
    actual = union.get_raw_data((0, 0, 0), tuple(left.shape))

    assert np.array_equal(actual, np.logical_or(left, right).astype(np.uint8))


def test_binary_union_rejects_nonbinary_or_incompatible_inputs():
    container, _, label_one, label_two = _fixture(np.uint32)
    mask = container.binary_mask(label_one)

    with pytest.raises(TypeError, match="uint8"):
        container.binary_union(container)

    invalid = CompressedVoxelContainer(
        mask.requested_bbox, mask.full_bbox, tuple(mask.block_size), np.uint8
    )
    invalid.blocks.set_block(0, np.asarray([0, 2], dtype=np.uint8), 1, np.zeros(1, np.uint32))
    with pytest.raises(ValueError, match="0/1"):
        mask.binary_union(invalid)

    other, _, _, _ = _fixture(np.uint32, offset=1000)
    other_mask = other.binary_mask(label_two + 1000)
    with pytest.raises(ValueError, match="identical"):
        shifted = CompressedVoxelContainer(
            Box((0, 0, 0), (7, 6, 3)), Box((0, 0, 0), (8, 8, 4)), (2, 2, 2), np.uint8
        )
        mask.binary_union(shifted)

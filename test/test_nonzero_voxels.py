import os
import sys

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import compressed_segmentation as cseg  # noqa: E402


def test_extract_nonzero_voxels_skips_empty_blocks_and_clips_query():
    store = cseg.PyBlockStore(2, np.uint8)
    palette = np.array([0, 1], dtype=np.uint8)
    first_block_bits = (1 << 0) | (1 << 3) | (1 << 7)
    store.set_block(0, palette, 1, np.array([first_block_bits], dtype=np.uint32))
    store.set_block(1, np.array([0], dtype=np.uint8), 0, None)

    points = cseg.extract_nonzero_voxels_3d_store(
        store,
        (2, 1, 1),
        (2, 2, 2),
        (2, 2, 2),
        (1, 0, 0),
        np.uint8,
    )

    assert points.dtype == np.int64
    assert {tuple(point) for point in points.tolist()} == {(0, 1, 0), (0, 1, 1)}


def test_extract_nonzero_voxels_full_constant_block():
    store = cseg.PyBlockStore(1, np.uint64)
    store.set_block(0, np.array([9], dtype=np.uint64), 0, None)

    points = cseg.extract_nonzero_voxels_3d_store(
        store,
        (1, 1, 1),
        (2, 2, 2),
        (2, 2, 2),
        (0, 0, 0),
        np.uint64,
    )

    assert points.shape == (8, 3)
    assert {tuple(point) for point in points.tolist()} == {
        (x, y, z) for x in range(2) for y in range(2) for z in range(2)
    }


def test_extract_matching_voxels_excludes_other_nonzero_labels():
    store = cseg.PyBlockStore(1, np.uint64)
    palette = np.array([0, 7, 9, 13], dtype=np.uint64)
    indices = [1, 2, 3, 0, 2, 1, 3, 2]
    encoded = sum(index << (2 * offset) for offset, index in enumerate(indices))
    store.set_block(0, palette, 2, np.array([encoded], dtype=np.uint32))

    points, values = cseg.extract_matching_voxels_3d_store(
        store,
        (1, 1, 1),
        (2, 2, 2),
        (2, 2, 2),
        (0, 0, 0),
        [7, 13],
        np.uint64,
    )

    actual = {(tuple(point), int(value)) for point, value in zip(points, values)}
    assert actual == {
        ((0, 0, 0), 7),
        ((0, 1, 0), 13),
        ((1, 0, 1), 7),
        ((0, 1, 1), 13),
    }


def test_extract_matching_voxels_clips_unaligned_query():
    store = cseg.PyBlockStore(1, np.uint32)
    store.set_block(0, np.array([5], dtype=np.uint32), 0, None)

    points, values = cseg.extract_matching_voxels_3d_store(
        store,
        (1, 1, 1),
        (2, 2, 2),
        (1, 2, 2),
        (1, 0, 0),
        [5],
        np.uint32,
    )

    assert values.tolist() == [5, 5, 5, 5]
    assert {tuple(point) for point in points.tolist()} == {
        (0, y, z) for y in range(2) for z in range(2)
    }


def test_extract_label_voxels_and_contacts_counts_each_contact_face():
    store = cseg.PyBlockStore(1, np.uint64)
    palette = np.array([0, 7, 9, 13], dtype=np.uint64)
    indices = [1, 2, 1, 3, 0, 0, 0, 0]
    encoded = sum(index << (2 * offset) for offset, index in enumerate(indices))
    store.set_block(0, palette, 2, np.array([encoded], dtype=np.uint32))

    points, contacts = cseg.extract_label_voxels_and_contacts_3d_store(
        store,
        (1, 1, 1),
        (2, 2, 2),
        (2, 2, 2),
        (0, 0, 0),
        7,
        [7, 9, 13],
        np.uint64,
    )

    assert {tuple(point) for point in points.tolist()} == {(0, 0, 0), (0, 1, 0)}
    assert sorted(int(value) for value in contacts) == [9, 13]


def test_extract_label_voxels_and_boundary_keeps_only_outer_shell():
    store = cseg.PyBlockStore(1, np.uint32)
    store.set_block(0, np.array([7], dtype=np.uint32), 0, None)

    solid, boundary = cseg.extract_label_voxels_and_boundary_3d_store(
        store,
        (1, 1, 1),
        (3, 3, 3),
        (3, 3, 3),
        (0, 0, 0),
        7,
        np.uint32,
    )

    assert solid.shape == (27, 3)
    assert boundary.shape == (26, 3)
    assert (1, 1, 1) not in {tuple(point) for point in boundary.tolist()}

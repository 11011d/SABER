import os
import sys

import numpy as np
import pytest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLOUD_VOLUME = os.path.join(ROOT, "cloud-volume")
sys.path.insert(0, ROOT)
sys.path.insert(0, CLOUD_VOLUME)

from cloudvolume import CloudVolume  # noqa: E402
from cloudvolume.lib import Bbox  # noqa: E402
import compressed_segmentation as cseg  # noqa: E402
from compressedvoxel import CompressedVoxelContainer  # noqa: E402


DATASET = os.environ.get("SABER_TEST_DATASET", "")
ROI = (
    slice(30000, 30129),
    slice(16000, 16129),
    slice(3500, 3533),
)
LRU_BYTES = 10 * 1024 * 1024


pytestmark = pytest.mark.skipif(
    not DATASET or not os.path.exists(os.path.join(DATASET, "info")),
    reason="set SABER_TEST_DATASET to a Precomputed volume to run integration tests",
)


def test_uses_local_compressed_segmentation_extension():
    assert os.path.dirname(os.path.abspath(cseg.__file__)) == ROOT


def _cloudvolume(**kwargs):
    return CloudVolume(
        DATASET,
        mip=0,
        bounded=False,
        progress=False,
        cache=True,
        lru_bytes=LRU_BYTES,
        **kwargs,
    )


def _dense_default():
    return _cloudvolume()[ROI]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cache_thread": 0},
        {"saber": 0},
        {"saber": 8},
        {"saber": 0, "cache_thread": 0},
        {"saber": 8, "cache_thread": 0},
    ],
)
def test_saber_dense_reads_match_default_cloudvolume(kwargs):
    expected = _dense_default()
    actual = _cloudvolume(**kwargs)[ROI]

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)


def test_saber_debug_returns_dense_result_and_timing_info():
    expected = _dense_default()
    actual, info = _cloudvolume(saber=0, cache_thread=0, saber_debug=True)[ROI]

    assert np.array_equal(actual, expected)
    assert info["download_sharded_compress_uzip_chunk_num"] == 8
    assert info["download_sharded_get_data_time"] >= 0
    assert info["download_sharded_compress_uzip_decode_fn_time"] >= 0


def test_debug_only_profiles_the_default_dense_path():
    expected = _dense_default()
    actual, info = _cloudvolume(saber_debug=True)[ROI]

    assert np.array_equal(actual, expected)
    assert info["download_sharded_compress_uzip_chunk_num"] == 8
    assert info["download_sharded_get_data_time"] >= 0
    assert info["download_sharded_compress_uzip_decode_fn_time"] >= 0


def test_saber_compressed_block_container_matches_dense_cutout():
    expected = _dense_default()
    container = _cloudvolume(saber=True, cache_thread=0)[ROI]

    assert isinstance(container, CompressedVoxelContainer)

    dense_from_container = container.get_raw_data(
        (0, 0, 0),
        tuple(container.requested_bbox.size3()),
    )

    assert dense_from_container.shape == expected.shape[:3]
    assert np.array_equal(dense_from_container, expected[..., 0])


def test_saber_compressed_block_where_matches_numpy_where():
    expected = _dense_default()[..., 0]
    container = _cloudvolume(saber=True, cache_thread=0)[ROI]

    labels = np.unique(expected)
    labels = labels[labels != 0]
    assert labels.size > 0

    segid = int(labels[0])
    mask = container.where(segid, 255, 0, out_dtype=np.uint8)
    actual = mask.get_raw_data((0, 0, 0), tuple(container.requested_bbox.size3()))
    expected_mask = np.where(expected == segid, 255, 0).astype(np.uint8)

    assert actual.dtype == np.uint8
    assert np.array_equal(actual, expected_mask)


def test_saber_masked_distinct_label_frontend_matches_dense_reference():
    expected = _dense_default()[..., 0]
    selection = np.zeros(expected.shape, dtype=bool)
    selection[::3, 1::4, ::2] = True
    selection[40:80, 35:90, 10:24] = True
    bbox = Bbox.from_slices(ROI)

    labels, stats = _cloudvolume(
        saber=True,
        cache_thread=0,
    ).distinct_labels_in_mask(bbox, selection)

    assert np.array_equal(labels, np.unique(expected[selection]))
    assert stats["selected_voxels"] == int(selection.sum())
    assert stats["bbox_voxels"] == int(expected.size)
    assert stats["fetch_seconds"] >= 0
    assert stats["select_seconds"] >= 0

    array_bbox_labels, _ = _cloudvolume(
        saber=True,
        cache_thread=0,
    ).distinct_labels_in_mask((bbox.minpt.copy(), bbox.maxpt.copy()), selection)
    assert np.array_equal(array_bbox_labels, labels)


def test_saber_masked_distinct_label_frontend_rejects_dense_backend():
    bbox = Bbox.from_slices(ROI)
    selection = np.ones(tuple(bbox.size3()), dtype=bool)
    with pytest.raises(ValueError, match="requires saber=True"):
        _cloudvolume().distinct_labels_in_mask(bbox, selection)

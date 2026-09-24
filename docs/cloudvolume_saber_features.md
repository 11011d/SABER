# SABER CloudVolume Features

Baseline: `ed2cba49ae15333bf602ba8b359cfd55de1bba98`

This branch keeps CloudVolume's default read interface compatible with the
baseline version and adds opt-in SABER paths for sharded
`compressed_segmentation` volumes.

## Feature Selector

SABER optimizations are opt-in. If `saber` and `cache_thread` are omitted,
CloudVolume uses the normal upstream read path.

- `saber=None` or `False`: default CloudVolume behavior, unless `cache_thread`
  or `saber_debug` is set.
- `saber=<int>`: dense partial decompression for sharded
  `compressed_segmentation`. The integer is the decompression parallelism
  (`<=0` single-thread partial, `>0` parallel in-place partial).
- `saber=True`: compressed-block container reads. This returns a
  `CompressedVoxelContainer`, uses the cache read optimization, and does not
  run dense partial decompression.
- `cache_thread=...`: local cache read thread control. Passing `cache_thread`
  without `saber` enables the cache-read-only SABER path.
- `saber_debug=True`: return `(result, debug_info)` from SABER reads.

The two main modes are mutually exclusive: numeric `saber` materializes a dense
`VolumeCutout`, while `saber=True` keeps data in the compressed block container.

## Feature Flags

Use the new behavior by passing explicit options to `CloudVolume`.

```python
from cloudvolume import CloudVolume

vol = CloudVolume(
    "/path/to/precomputed-volume",
    mip=0,
    fill_missing=True,
    cache=True,
    saber=True,
    cache_thread=0,
)
```

## Compressed Block Cutouts

`saber=True` returns a `CompressedVoxelContainer` for sharded
`compressed_segmentation` reads instead of materializing a dense `VolumeCutout`.

```python
vol.segid_list = [segid1, segid2]
origin = (float(x0), float(y0), float(z0))
cutout = vol[x0:x1, y0:y1, z0:z1]

mask = cutout.where(segid1, 255, 0, out_dtype=np.uint8)
mask.keep_nearest_connected_component_optimized(lx, ly, lz)
mask.extract_boundary_voxel_points_3d(origin, pc_data, ids_data)
```

The container keeps compressed blocks in a C++ block store and exposes the
high-level operations used by the SABER point-cloud extraction workflow.
`extract_boundary_voxel_points_3d` emits 3D surface voxel centers in the same
voxel coordinate system as CloudVolume slicing: `point = origin + local_voxel`.
A foreground voxel is included when at least one of its six axis-aligned
neighbors is background or outside the request window.

## Masked Distinct-Label Discovery

Growth-vector candidate discovery can return only the labels selected by a
query-local boolean mask, without materializing a dense label cutout:

```python
vol.segid_list = None
cutout = vol[x0:x1, y0:y1, z0:z1]
labels, stats = cutout.distinct_labels_in_mask(directional_mask)
```

For query paths that need only IDs, use the frontend method so the transient
compressed container does not escape the storage layer:

```python
labels, stats = cv.distinct_labels_in_mask(query_bbox, directional_mask)
```

This API returns only the selected labels and block-classification/timing
statistics. It never returns or materializes a dense voxel cutout.

The native traversal classifies each intersecting block before reading voxel
values. A block disjoint from the selection mask is skipped. A fully selected
block wholly inside the query is answered from its palette. Only a partially
selected block, including a block clipped by the query bbox, reads compressed
palette indices for selected voxels. The return value contains sorted distinct
labels and counters for palette-only, partially decoded, skipped blocks, and
selected voxels. It never returns a dense voxel array.

## Pair Feature Extraction

Node- and Edge-style pair features can be generated directly from the block
store without materializing the requested bbox, intermediate masks, or a
resized source tensor:

```python
vol.segid_list = [label_one, label_two]
cutout = vol[x0:x1, y0:y1, z0:z1]
node_feature = cutout.extract_pair_feature(
    label_one,
    label_two,
    output_shape_zyx=(20, 60, 60),
)
```

The output is a contiguous `uint8` tensor with channels
`[label_one OR label_two, label_one, label_two]`. Sampling follows the floor
index nearest-neighbor rule used by the BiologicalGraphs feature generator.
The API accepts voxel dimensions only. Conversion from physical radius or
resolution to a CloudVolume bbox remains a caller responsibility.

Homogeneous Node or Edge batches use one native batch call and one contiguous
output allocation:

```python
from compressedvoxel import extract_pair_features_batch

features = extract_pair_features_batch(
    [(cutout_a, a1, a2), (cutout_b, b1, b2)],
    output_shape_zyx=(18, 52, 52),
    parallel=2,
)
```

Containers may have different voxel bboxes but must have the same label dtype.
The batch helper parallelizes native feature generation; CloudVolume reads are
still issued by the caller.

### Experimental composable binary-mask path

This path is retained for reusable binary-mask operator evaluation. The stable
BiologicalGraphs workflow uses the fused pair-feature API above.

An alternative single-threaded path keeps the two object masks and their union
as separate compressed values:

```python
vol.segid_list = [label_one]
label_one_cutout = vol[x0:x1, y0:y1, z0:z1]
vol.segid_list = [label_two]
label_two_cutout = vol[x0:x1, y0:y1, z0:z1]

label_one_mask = label_one_cutout.binary_mask(label_one)
label_two_mask = label_two_cutout.binary_mask(label_two)
union_mask = label_one_mask.binary_union(label_two_mask)
feature = union_mask.materialize_binary_channels(
    label_one_mask,
    label_two_mask,
    output_shape_zyx=(20, 60, 60),
)
```

`binary_union` accepts only compatible compressed `uint8` stores whose values
are exactly zero or one. Blocks present in only one input are copied directly;
only positions present in both inputs are decoded, ORed, and recompressed. The
materializer allocates one zero-filled output tensor and traverses only nonempty
union blocks. Physical resolution and bbox construction remain caller concerns.

## Partial Dense Decode

For dense reads, an integer `saber` value enables compressed-segmentation
partial decompression on sharded volumes.

```python
vol = CloudVolume(
    cloudpath,
    mip=0,
    fill_missing=True,
    cache=True,
    saber=8,
)

cutout = vol[x0:x1, y0:y1, z0:z1]
```

Values greater than zero use the parallel in-place decoder. Values less than or
equal to zero use the single-thread partial path. `None` or `False` keeps the
baseline full decode behavior.

## Cache Thread Control

`cache_thread` is passed to local cache reads in the SABER sharded path. This
requires the SABER `cloud-files` dependency, pinned in this repository at
`cloud-files` commit `384945895bb8969e93ee83732ff7e956f4588e51`.

- `None`: keep CloudFiles default behavior.
- `0`: read local cache content in the caller thread.
- positive integer: request that many CloudFiles cache-read threads.

The environment helper installs the local `cloud-files` checkout before
installing CloudVolume:

```bash
python scripts/setup_saber_env.py
```

`--patch-cloudfiles` is still available for older experiments, but clean SABER
environments should use the pinned `cloud-files` checkout instead of relying on
an ad hoc patch of the official package.

For an environment where imports should work without adding this repository to
`PYTHONPATH`, replace the installed libraries directly:

```bash
python scripts/replace_python_libs_with_saber.py \
  --python-lib-dir /path/to/the/actual/python/site-packages \
  --python /path/to/the/python
```

`--python-lib-dir` must be the real library directory used by the Python
interpreter that will run CloudVolume, for example
`/opt/miniconda3/lib/python3.10/site-packages`. The script first rebuilds
`compressed_segmentation` with the target `--python`, then copies the SABER
`cloudfiles`, `cloudvolume`, `compressedvoxel.py`, and rebuilt
`compressed_segmentation*.so` into that directory. It verifies the import paths
and checks that `CloudFiles.__init__` exposes `use_optional_thread`, because
`cache_thread` depends on that parameter. Existing targets are moved into a
timestamped `saber-backup-*` directory unless `--no-backup` is supplied.

The script does not install general runtime dependencies. Make sure the target
environment already has the packages imported by `compressedvoxel.py`, notably
`connected-components-3d` (`cc3d`) and OpenCV (`cv2`). Because
`compressed_segmentation` is rebuilt with the target Python, make sure the
target environment already has the intended NumPy version and build
dependencies installed.

## Debug Info

`saber_debug=True` returns detailed timing and cache/shard counters for SABER
download paths. With no `saber_debug`, CloudVolume returns the same value shape
as the baseline interface.

```python
vol = CloudVolume(cloudpath, saber=8, saber_debug=True)
cutout, debug_info = vol[x0:x1, y0:y1, z0:z1]
```

The debug dictionary includes the existing SABER stage timings, cache hit/miss
counts, shard index timings, and total elapsed time for slicing calls.

## Validation

Run the standardized correctness and performance comparison:

```bash
python benchmarks/saber_compare.py \
  --cloudpath /path/to/precomputed-volume \
  --candidate-file data/candidate0.csv \
  --json-out /tmp/saber_compare.json
```

The benchmark compares compressed-block output against the dense baseline by
point count, sorted coordinates, labels, and stage timings.

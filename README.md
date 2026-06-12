# Compressed Block Optimization for Neuro Tracking

This project provides an optimization solution for memory-bandwidth-intensive operations in neuron data processing. By splitting data into smaller compressed units (blocks) for storage and processing in memory, and by combining the sparsity of neuron data with a core C++ extension, we achieve a substantial reduction in memory bandwidth usage and a significant speedup in runtime.

Concrete usage examples and performance comparisons between the original mode and the compressed-block mode can be reproduced directly by running `test.py`. To fully realize the performance of this repository, a deeply optimized `cloud-volume` implementation is also integrated as a submodule.

## Advantages
* **Reduced memory bandwidth**: Data is stored in physical memory as smaller compressed blocks, which avoids repeated copying and deserialization of large chunks.
* **Acceleration through data sparsity**: By exploiting the sparse characteristics of neuron data together with the optimized C++ path, blocks containing irrelevant data can be skipped, avoiding the loading and processing of large empty regions.
* **Fast C++ method extension**: Key voxel scanning, filtering, and point-cloud reconstruction logic is moved to the C++ layer (see `CompressedVoxelContainer`), which greatly reduces Python call overhead.
* **High-performance read concurrency (CloudVolume optimization)**: By adding parallel decompression and low-level I/O scheduling optimizations to the `cloud-volume` submodule, slice read throughput is significantly improved.

---

## Prerequisites & Usage

### 1. Clone the repository with submodules

This project depends on a customized `cloud-volume` submodule. When cloning the repository, make sure to fetch the full submodule structure:

```bash
git clone --recursive <your-repository-url>
cd <repository-name>

# If you already cloned the repository, you can load the submodule with:
# git submodule update --init --recursive
```

### 2. Build and replace the C++ extension (`compressed-segmentation`)

Build the low-level acceleration library from source. This optimized build is intended to replace the original C++ extension currently installed in the Python environment.

```bash
python setup.py build_ext --inplace
```

After a successful build, a compiled shared library will be generated in the current directory, for example:

```text
compressed_segmentation.cpython-310-x86_64-linux-gnu.so
```

To locate the existing package in your Python environment, run:

```bash
python -c "import compressed_segmentation; print(compressed_segmentation.__file__)"
```

Then copy the newly built shared library and overwrite the installed one at the printed path to activate the optimized version.

### 3. Modify the `cloudfiles` scheduling strategy

To better match the optimized CloudVolume reading path and avoid thread-pool overhead during local disk I/O, modify the `cloudfiles` source in your Python environment so that local reads are forced to use the main process:

1. Locate `cloudfiles/cloudfiles.py` in the same `site-packages` directory.
2. Find the following code:

```python
if self.protocol == "file":
    num_threads = 1
```

3. Change it to:

```python
if self.protocol == "file":
    num_threads = 0
```

### 4. Prepare the data source configuration

This project includes a small batch of test data under the root `data` directory (`data/candidate0.csv`). To run the test successfully:

you need to modify the volume path in `test.py` so that it points to your actual original segmentation data source (CV path), for example:

```python
"/path/to/your/volume"
```

`test.py` will read test coordinates from `data/candidate0.csv` by default.

### 5. Run `test.py`

After replacing the required environment libraries and updating the source data path, run:

```bash
python test.py
```

The script contains the core comparison logic. Before running, check the following key settings and the CloudVolume repository path:

1. `test.py` imports the local `cloud-volume` submodule from the repository root through `LOCAL_CLONE = './cloud-volume'`.
2. Internally, the script first enables the accelerated logic by setting `USE_COMPRESSED_BLOCK = True`, and then switches back to `False` to compare runtime against the native path.

You should then be able to directly compare the runtime reductions for several complex stages such as Fetch, Where, CC, and Boundary in the console.

Example output:

```text
============================================================
[Pre-scan] Initialize the original CloudVolume and scan valid indices...
valid_indices=[8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 97, 98, 99, 100, 106, 107, 108, 109, 110, 111, 113, 114]
============================================================
[Mode 1] Start with use_compressed_block=True
[Timing] Fetch: 0.2461s, Where: 0.1215s, CC: 0.1150s, Boundary: 0.2277s
[Mode 1] Done in 0.73s, total points: 71494, processed cases: 28
============================================================
[Mode 2] Start with use_compressed_block=False
[Timing] Fetch: 1.0638s, Where: 0.3234s, CC: 1.3163s, Boundary: 0.1506s
[Mode 2] Done in 2.88s, total points: 71494, processed cases: 28
============================================================

[Comparison] compressed mode points: 71494, original mode points: 71494
[PASS] The two modes produce identical results.
```

---

## Core Structure

* **`compressedvoxel.py` -> `CompressedVoxelContainer`**
  This is the core data container class on the Python side of the project. It is designed to encapsulate and manage many small compressed data blocks. When users perform voxel-condition filtering and other high-intensity analysis tasks through `CompressedVoxelContainer`, the structure ensures that most of the heavy logic is automatically routed to the underlying per-block processing path designed for performance, significantly reducing traversal time while preserving correctness. It also includes efficient methods for boundary contour analysis and voxel surface extraction.

* **Low-level C++ support (`src/compressed_segmentation.pyx`)**
  The low-level acceleration is implemented through Cython-integrated data structures and logic. The C++ methods written there greatly streamline and avoid slow native Python processing paths. The block mechanism described above is packaged into a dynamically loadable library that can be used directly from Python, which is also the key reason this project achieves a major performance improvement when processing sparse-feature data.

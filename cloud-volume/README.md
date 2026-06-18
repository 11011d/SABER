# LightWeightCloudVolume

**LightWeightCloudVolume** is a high-performance optimized branch based on CloudVolume, designed specifically for voxel-data reading scenarios in the **Compressed-Segmentation** format. By introducing partial decompression, parallel decompression, zero-copy mechanisms, and low-level I/O scheduling optimizations, it significantly improves slice-read throughput for small-task workloads.

## ⚙️ Prerequisites

Before using this branch, you **must** complete the following environment setup and dependency replacement steps. Otherwise, the optimizations will not take effect.

### 1. Build and replace the `compressed-segmentation` backend library

You need to replace the original C++ extension library in your Python environment with the specific optimized version.

1.  **Clone and build the optimized library**:
    ```bash
    git clone git@github.com:P11011/Compressed_Segmentation.git
    cd Compressed_Segmentation
    python setup.py build_ext --inplace
    # After a successful build, a file similar to compressed_segmentation.cpython-310-x86_64-linux-gnu.so will be generated
    # (the 310 in the filename stands for Python 3.10, which may differ in your environment)
    ```

2.  **Locate the original package installation path**:
    Use pip to inspect the package path in the current environment:
    ```bash
    pip show compressed-segmentation
    ```
    *Please note the path shown in the `Location` field.*

3.  **Replace the `.so` file**:
    Copy the `.so` file generated in step 1 and overwrite the file with the same name under the path found in step 2.

### 2. Modify the `cloudfiles` scheduling strategy

To avoid thread-pool overhead during local-disk reads, you need to modify the `cloudfiles` library source code and force it to use the main process for reading.

1.  Locate the `cloudfiles/cloudfiles.py` file (inside `site-packages`).
2.  Search for the following code block:
    ```python
    if self.protocol == "file":
        num_threads = 1
    ```
3.  **Change it to**:
    ```python
    if self.protocol == "file":
        num_threads = 0  # Change to 0 to force main-process reading
    ```
    > **Note**: Setting `num_threads = 0` makes disk-cache reads execute directly in the main process and avoids the context-switch overhead of entering the thread pool.

---

## 🛠 Usage

After the preparation is complete, the usage remains consistent with native `CloudVolume`. By passing specific arguments during initialization, you can enable the optimization mode of `LightWeightCloudVolume`.

### Example code

```python
# Use a local CloudVolume repository
LOCAL_CLONE = 'Path to the CloudVolume repository'
if os.path.exists(LOCAL_CLONE):
    sys.path.insert(0, LOCAL_CLONE)
from cloudvolume import CloudVolume

# Data path
cv_path = "precomputed://file:///path/to/data"

# Initialize LightWeightCloudVolume
#
vol = CloudVolume(
    cv_path,
    mip=0,
    fill_missing=True,
    # It is recommended to enable the second-level cache
    cache=True,
    lru_bytes=1024*1024*10,
    # --- Newly added optimization parameters ---
    partial_decompress_parallel=8,   # Enable optimization: set the number of parallel decompression threads
    log_path="./logs/read_perf.log"  # Optional: set the log path for detailed timing records
)

# Read data (the API remains unchanged)
image = vol[0:256, 0:256, 0:32]

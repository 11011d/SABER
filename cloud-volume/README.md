# LightWeightCloudVolume

**LightWeightCloudVolume** 是基于 CloudVolume 的高性能优化分支，专为 **Compressed-Segmentation** 格式的体素数据读取场景设计。通过引入部分解压、并行解压、零拷贝机制与底层 IO 调度优化，显著提升了小任务场景下切片数据的读取吞吐率。

## ⚙️ 准备工作 (Prerequisites)

在使用本分支之前，**必须**完成以下环境配置和依赖库替换，否则优化功能将无法生效。

### 1. 编译并替换 `compressed-segmentation` 后端库

我们需要使用特定的优化版本替换 Python 环境中原有的 C++ 扩展库。

1.  **克隆并编译优化库**：
    ```bash
    git clone git@github.com:P11011/Compressed_Segmentation.git
    cd Compressed_Segmentation
    python setup.py build_ext --inplace
    # 编译成功后会生成类似 compressed_segmentation.cpython-310-x86_64-linux-gnu.so 的文件
    # (文件名中的 310 代表 Python 3.10，根据您的环境可能有所不同)
    ```

2.  **定位原有包位置**：
    使用 pip 查看当前环境中安装的包路径：
    ```bash
    pip show compressed-segmentation
    ```
    *请记录输出中 `Location` 字段显示的路径。*

3.  **替换 `.so` 文件**：
    将第 1 步编译生成的 `.so` 文件，复制并覆盖第 2 步路径下的同名文件。

### 2. 修改 `cloudfiles` 调度策略

为了避免本地磁盘读取时的线程池开销，需修改 `cloudfiles` 库源码，强制使用主进程读取。

1.  找到 `cloudfiles/cloudfiles.py` 文件（位于 site-packages 中）。
2.  搜索如下代码块：
    ```python
    if self.protocol == "file":
        num_threads = 1
    ```
3.  **将其修改为**：
    ```python
    if self.protocol == "file":
        num_threads = 0  # 修改为 0，强制使用主进程读取
    ```
    > **注意**：设置 `num_threads = 0` 可使得磁盘缓存读取直接在主进程执行，避免了进入线程池的上下文切换开销。

---

## 🛠 使用方法 (Usage)

准备工作完成后，使用方式与原生 `CloudVolume` 保持一致。通过在初始化时传入特定参数，即可激活 `LightWeightCloudVolume` 的优化模式。

### 示例代码

```python
# 使用用本地CloudVolume代码仓
LOCAL_CLONE = 'CloudVolume代码仓地址'
if os.path.exists(LOCAL_CLONE):
    sys.path.insert(0, LOCAL_CLONE)
from cloudvolume import CloudVolume

# 数据路径
cv_path = "precomputed://file:///path/to/data"

# 初始化 LightWeightCloudVolume
# 
vol = CloudVolume(
    cv_path, 
    mip=0, 
    fill_missing=True,
    # 建议开启二级缓存
    cache=True,
    lru_bytes=1024*1024*10,
    # --- 新增优化参数 ---
    partial_decompress_parallel=8,   # 激活优化：设置并行解压线程数
    log_path="./logs/read_perf.log"  # 可选：设置日志路径以记录详细耗时
)

# 读取数据 (API 保持不变)
image = vol[0:256, 0:256, 0:32]
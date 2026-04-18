# Compressed Block Optimization for Neuro Tracking

本项目提供了一个针对神经元数据处理的内存带宽密集型操作的优化方案。通过将数据拆分成更小的压缩单元（block）在内存中存储和处理，并结合神经元数据的稀疏性及核心 C++ 扩展，我们实现了内存带宽占用的大幅降低和任务运行速度的成倍提升。

具体的使用范例和新旧模式（原始模式 vs 压缩块模式）的性能对比数据可在此项目下的 `test.py` 中直接运行并查看。为了在此代码仓中完美发挥性能，我们还将深度优化的 `cloud-volume` 以子模块的形式集成在了项目中。

## 方案优势
* **降低内存带宽**：以更小的压缩块形式在物理内存中存储数据，避免大块数据的重复拷贝和反序列化。
* **利用数据稀疏性加速**：利用神经元的高斯能压缩态与极速 C++ 的稀疏特性，跳过包含无关数据的区块，避免加载和处理大面积空白数据区域。
* **极速 C++ 方法扩展**：将关键的体素扫描、过滤和点云重构逻辑置于 C++ 层（参考 `CompressedVoxelContainer`），大幅减少 Python 的调用开销。
* **高性能读取并发（CloudVolume优化）**：通过在 CloudVolume 的子模块中添加并行解压与底层 IO 调度优化，显著提升切片数据的读取吞吐率。

---

## 🛠 快速使用指南 (Prerequisites & Usage)

### 1. 拉取代码仓及子模块

本项目依赖于特制的 CloudVolume 子模块，拉取代码时请确保获取完整的子模块结构：

```bash
# 拉取主仓库代码并自动包含子模块
git clone --recursive <您的代码仓地址>
cd <代码仓名字>

# 【备用操作】如果您已经提前 clone 了本仓，可以直接运行以下命令来补全加载子模块：
# git submodule update --init --recursive
```

### 2. 编译并替换 C++ 扩展库 (`compressed-segmentation`)

生成基于 C++ 编译的底层加速库，这需要使用特定的优化版本替换 Python 环境中原有的 C++ 扩展库。

```bash
# 在本项目根目录下运行以编译 C++ 扩展：
python setup.py build_ext --inplace
```
编译成功后，将在当前目录下生成一个经过编译的动态链接库（例如：`compressed_segmentation.cpython-310-x86_64-linux-gnu.so`）。

**替换 Python 环境中的 `.so` 库：**
你可以通过以下命令查找环境中现有的包位置：
```bash
python -c "import compressed_segmentation; print(compressed_segmentation.__file__)"
```
（输出内容类似：`.../site-packages/compressed_segmentation.cpython-310-x86_64-linux-gnu.so`）
将当前编译生成的 `.so` 文件复制并覆盖到对应的终端输出位置即可生效。

### 3. 修改 `cloudfiles` 调度策略

为了完美配合 CloudVolume 读取避免本地磁盘 IO 时的线程池开销，需修改 Python 环境中 `cloudfiles` 库源码，强制使用主进程读取。

1. 找到上面库所处同一 site-packages 目录中的 `cloudfiles/cloudfiles.py` 文件。
2. 搜索如下代码块：
   ```python
   if self.protocol == "file":
       num_threads = 1
   ```
3. **将其修改为** (强制使用主进程读取)：
   ```python
   if self.protocol == "file":
       num_threads = 0 
   ```

### 4. 数据源准备配置 

本项目在根目录的 `data` 文件夹下存放了一小批测试数据（`data/candidate0.csv`）。为了顺利运行测试：

你需要将 `test.py` 中的 `'/CX/neuro_tracking/fafb-ffn1'` 字段修改为你目前实际的 **对应的原始分割数据地址 (CV Path)**。`test.py` 内部将会默认读取刚才放置在 `data/candidate0.csv` 的测试坐标文件。

### 5. 执行测试 `test.py`

在确保环境库已经覆盖替换完成并写好原始数据地址后，直接运行即可。代码中包含了运行核心的对比逻辑（使用前请检查以下核心开关及 Cloudvolume 代码仓位置）：

1. 测试脚本 `test.py` 开头的库引入会自动寻址根路径下的 `cloud-volume` (`LOCAL_CLONE = './cloud-volume'`) 子模块资源。
2. 内部通过将 `USE_COMPRESSED_BLOCK` 开关设为 `True` 执行提速后的核心分析逻辑，之后以 `False` 回滚至原生环境进行时间比对。

执行测试：
```bash
python test.py
```
你将能够在控制台直观地对比 Fetch、Where、CC、Boundary 等几个复杂阶段带来的全方位耗时锐减突破。

**示例测试输出参考：**
（需注意：此输出是在开启 `use_compressed_block=True` 和设定 `cache_threads=0` 以代表开启完整 SABER 优化时的代表性打印）：
```text
(base) root@90689ef86a7f:/CX/neuro_tracking/xinr/SABER_code# python test.py 
============================================================
[预扫描] 初始化原始 CloudVolume 并扫描有效下标...
valid_indices=[8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 97, 98, 99, 100, 106, 107, 108, 109, 110, 111, 113, 114]
============================================================
[模式 1] 启动 use_compressed_block=True
[耗时分析] Fetch: 0.2461s, Where: 0.1215s, CC: 0.1150s, Boundary: 0.2277s
[模式 1] 完成，耗时 0.73s，共 71494 个点，处理 case: 28
============================================================
[模式 2] 启动 use_compressed_block=False
[耗时分析] Fetch: 1.0638s, Where: 0.3234s, CC: 1.3163s, Boundary: 0.1506s
[模式 2] 完成，耗时 2.88s，共 71494 个点，处理 case: 28
============================================================

[对比] compressed 模式点数: 71494,  原始模式点数: 71494
[PASS] 两种模式结果完全一致 ✓
```

---

## 核心结构细节说明

* **`compressedvoxel.py` -> `CompressedVoxelContainer` 类**
  作为本项目 Python 层级的核心数据容器类，它被设计来封装和管理多个细小的压缩数据块。当使用者通过 `CompressedVoxelContainer` 进行体素条件筛选分析等高强度任务时，该结构能保证绝大多数负荷逻辑自动地路由到基于性能考虑设计的单个子块处理层上，并在保证正确率的基础上大幅度缩短遍历用时。它也包含如边界轮廓分析与提取体素表面的高效方法。
  
* **底层 C++ 的支持扩展 (`src/compressed_segmentation.pyx`)**
  底层的加速由通过 Cython 集成实现的数据类型定义和逻辑构成，这里面所编写的 C++ 实现方法极大化地精简和规避了那些原生 Python 慢速处理流程。其中承载了上述提到的 Block 机制并封装为 Python 中方便直接引用的动态对象库，这也是项目得以在处理稀疏特征数据的性能大幅飞跃的关键。

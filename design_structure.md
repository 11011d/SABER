# Title: A Sparsity-Aware, Compressed-Domain Processing Framework for Pairwise Point Cloud Extraction

## 1. **Step 4: Pairwise Point Cloud Extraction (成对点云提取)**
在对神经元及其连接信息进行大规模、精准重构的自动校对与连接预测流程中，提取目标连接部的形态学特征（**Step 4: Pairwise Point Cloud Extraction**）起着绝对的主导作用。
然而，由于每个原始神经元断口都需要生成上百个可能的延展探测方向，导致**候选对（Candidate Pairs）的数量极其庞大**（通常激增至原始神经元总数的上百倍）。这些候选对在庞大的脑图谱三维空间内分布极为零散与不规则，需要对每个候选对进行独立的分析。因此，该步骤毫无意外地成了全流程中最为消耗时间且极其重度依赖内存带宽和CPU。为了充分利用异构计算硬件，并避免巨大的中间结果写回问题系统调度通常会将完全依赖 GPU 资源的下游环节（即 **Step 5: Image Embedding Prediction**）与本层 CPU 萃取环节深度糅合成**流水线并行（Pipeline Parallelism）**执行。

在尚未优化的传统提取流程中，每一步核心操作都基于从网络拉取并全量解密的稠密数组（Dense Array）粗暴式地展开，主要包含以下串行步骤：

1. **原始体素数据加载 (Raw Voxel Fetching)**：
   基于坐标区域加载包含目标神经元的原始体素数据块。在这个阶段，数据存储系统需要按 chunk 粒度对齐并全量下载、无差别解压所有数据后，再切片出用户需要的区域。

2. **目标掩码提取 (Target Mask Extraction)**：
   为了剔除其他无关或背景神经元分支的影响，通过传统等式判断函数（如 `np.where(vol_ == segid1, 255, 0)`）在已经完全实例化和解密了的稠密矩阵中逐个进行遍历，剥离并生成仅仅关于当前目标 ID 的二值掩码。

3. **连通域提取操作 (Connected Component Extraction)**：
   利用 `cc3d.connected_components` 对上一步得到的目标掩码场进行全量三维分析，寻找所有孤立块，并通过额外的空间距离度量方法，从庞大的稠密结果场中筛选出最靠近我们关注中心的单一神经元片段的主体并重新赋掩码。

4. **神经元点云轮廓采集 (Neuron Surface Point Cloud Contouring)**：
   通过类似于 `cv2.findContours` 的二维切片扫描算法，逐层遍历 3D NumPy 数组去生成能够表征神经元最表面形态特征的点云（Point Cloud）。

5. **点云采样 (Point Cloud Sampling/Resampling)**：
   将前面采集到的轮廓体素上采样或下采样到指定的标准网络维度序列，用以保障后续下游子任务计算形状和深度特征的一致性。

**传统流程的痛点：**
这一经典流程严重受制于**“内存全量解压”**的机制，具体表现为两个致命问题：

1. **系统内存总线资源争用与带宽墙 (System Memory Bus Contention and Bandwidth Wall)**：在多进程并发环境下，将具备高压缩比的原始神经元数据暴力解压为庞大的稠密数组，会引发极其严重的访存瓶颈。由于大量数据仅用于计算强度极低（Low Arithmetic Intensity）的简单遍历分析并被迅速废弃导致了极高的缓存未命中率（Cache Misses），迫使 CPU 必须直接通过系统总线向主存持续抛出海量的数据装载请求，瞬间轰出了灾难性的内存吞吐（实测引发高达 200GB/s 的冗余访存峰值）。
更具挑战性的是，为实现异构资源的充分利用，系统启用了与高度依赖 GPU 资源的下游环节（即 **Step 5: Image Embedding Prediction**）的流水线并行（Pipeline Parallelism）。但在 Step 5 的推理过程中，即时生成的海量高维特征向量（Feature Embeddings）导致数据规模急剧膨胀，这批庞大的特征矩阵必须经由 PCIe 链路在 CPU 主存与 GPU 显存之间进行密集交替传输。这直接导致了致命的后果：激增的点云解析吞吐与海量的特征传输流在共享的系统总线上发生了剧烈碰撞与带宽争用（Bandwidth Contention）。两路核心计算单元最终彻底榨干了物理带宽上限，双双陷入苦等数据 I/O 的“带宽饥饿（Bandwidth Starvation）”与 Memory-Bound 状态。
2. **海量无效算力与访存的浪费 (Redundant Compute & Fetch Waste)**：目标微观结构往往极度纤细，在截取的 Bounding Box 中通常仅占据不到百分之一的有效空间。传统的稠密计算算子（如全阵列的 `np.where` 扫描或 3D `cc3d`）被迫对背景中海量的零值（Zero Voxels）执行 $O(N^3)$ 的无差别读取与逻辑比对。这不仅占用了算术单元，更致命的是迫使底层硬件浪费极高的吞吐率去加载这些无效背景。

这两个痛点也正是本系统架构中两大基石——**使用“压缩数据（Compressed Domain）”**与运用**“稀疏特征（Sparsity）”**的原始动机。通过在维持原有物理内存开销的框架下转入压缩域操作，直接剪枝掉无用的海量零值读取，我们将系统并发时的内存吞吐需求降低了一半以上（由 200GB/s 降至 90GB/s），从根本上粉碎了“内存墙（Memory Wall）”的枷锁，实现了整体任务速度跨越式的 4x+ 提升。


---

## 2. Our Proposed System Design

### 2.1 Core Ideas of the Framework
为了彻底打破全量解密稠密阵列所带来的性能藩篱，我们提出了 **A Sparsity-Aware, Compressed-Domain Processing Framework (一个稀疏感知的压缩域处理框架)**。该框架的设计基石依赖于两大特质：

*   **Sparsity (稀疏特征)**：神经细胞在此尺度的三维切片上是极为稀疏的。算法有义务感知并利用这种极度的数据倾斜，将计算前锋（Frontier）精确制导至真正非零的拓扑之上，跳过海量背景。
*   **Compressed Domain (压缩数据/压缩域)**：数据从存储侧加载至内存侧后，我们利用自研引擎使其维持在拆分后的小尺寸「压缩块 (Block)」状态，并不将其贸然展开，所有的底层查找、比对与运算全部被特化为在**压缩表示态**下的零拷贝操作。

由这两个基石出发，我们重构并提供了在 Compressed Voxel Container 基础上的一流分析算子，令复杂的分析能够直接根据 Metadata（例如块特征“调色盘/Palette”）以及在压缩结构游走完成。

### 2.2 Re-engineered Pipeline in the Proposed Framework
在这个理论框架支持下，原来的 Pairwise Point Cloud Extraction 工作流被重置并利用下列极速算法得以新生。基于我们在 1732 个有效神经元样本对上的实测结果，端到端的单个任务平均耗时从 **7.74s 骤降至 1.83s**，斩获了总体高达 **4.2x 的系统级加速比**。具体各步骤的底层优化与性能拆解如下：

1. **原始体素数据加载 -> *On-Demand Partial & Palette-Guided Sparse Loading*** **(加速比 10.5x：2.83s $\to$ 0.26s)**
   这一阶段高达 10.5 倍的极速提升，实际上来源于我们底层加载引擎精心设计的**双层解压优化策略（Two-Level Decoding Strategy）**：
   *   **基于 Block 粒度的按需局部解压 (On-Demand Partial Decompression)**：在传统的 Chunk 存取机制下，哪怕用户只请求了极小的候选区域（ROI），底层框架也会粗暴地将请求范围向外延展、对齐至庞大且死板的 Chunk 边界，进而导致巨量的无关区域被连带全量解压。我们通过击穿 Chunk 的内部隔离，细粒度地仅提取并解压那些与用户实际 Bounding Box 拥有空间交集的微小 Block，在空间维度上完美剥离了对齐膨胀造成的冗余负载。
   *   **稀疏感知下的精准拦截 (Sparsity-Guided Conditional Fetching)**：在上述精简出的微小空间集内，系统进一步结合待提取的特征（`segid_list`）触发底层的稀疏感知。取流引擎会极速核对每个 Block 的特征元数据（如 Palette 调色盘），只有当明确感知到内部囊括目标神经元 ID 时，才会真正下发解压指令。
   这两层空间与特征上的交叉剪枝互为表里，直接从内存 I/O 与反序列化链路上“蒸发”掉了超过 90%+ 的无效背景数据，构筑了极其断崖式的数据加载优势。

2. **目标掩码提取 -> *SparsityAwareMasking*** **(加速比 3.0x：0.83s $\to$ 0.28s)**
   对应传统的全量像素遍历 `where` 匹配，新的方法 `where(segid1, ...)` 在维持数据整体压缩特性的同时进行判断。在各个微小分块的调色盘中如果发现没有所求 `segid`，系统将在 $O(1)$ 时间复杂度下将其直接转换为零标记的空区块（Zero-Block），彻底剥离了脱离压缩态后在海量内存中捞针式的遍历负担。

3. **连通域提取操作 -> *SparseAnchorLocalization* & *SparsityAwareCCA*** **(加速比 8.1x：3.42s $\to$ 0.42s)**
   此阶段的优化贡献了全流程中最为显著的绝对耗时下降。我们在连通组件（Connected Component, CC）的提取上实现了极其关键的范式重构：
   *   **传统范式的冗余过计算 (Over-computation)**：传统连通域提取严重依赖于三个开销庞大的串行步骤。首先，系统需要调用 `cc3d.connected_components` 对全局稠密张量进行无插值的全量图扫描，将空间内的所有非零特征划分为各自孤立的独立片段体系；其次，系统需要遍历这些孤立片段集，计算它们与请求空间中心的物理距离，以寻址出距离探测原点最近的目标神经片段节点；最后，算法再次调用掩码算子（如全阵列的 `np.where`），在全局数组中仅单独保留该片段作为主根，而海量前置计算出的其余连通域数据则被当做无效结果直接丢弃。这种方式存在惊人的过度计算与无谓的访存消耗。
   *   **新架构的两步式局域按需萃取 (Seed-Driven In-Situ Isolation)**：我们的设计将其重构为严格去冗的“两步走”串联逻辑。
   *   **第一步：** 通过前置算子 **SparseAnchorLocalization (`nearest_nonzero_idx`)** 完成极速寻点。该操作彻底抛弃了像素级扫描，直接基于底层压缩架构的宏观 Block 树形元数据，实现高跨度的空间跳跃式寻址，以近乎零时钟周期的代价，极速锁定距离空间请求中心最近的非零实体区块起始点（Seed）。
   *   **第二步：** 依托该合法锚点触发核心提取机制 **SparsityAwareCCA (`keep_nearest_connected_component_optimized`)**。该算法的游走指针以 Seed 为中心起步，仅沿着实体解压后真实关联的有效神经分支拓展局部广度优先搜索（Sparsity-Guided BFS）。BFS 的寻路受底层的特征调色盘（Palette）指引，对空间上不连通、成分不匹配的悬浮干扰区块触发拦截免疫，并在搜索扩展期原位重写（In-situ update）背景杂项的内存掩码。这从根本上终结了“先结构全量分析、再欧式过滤剥离”所带来的算力与带宽的结构性空耗。

4. **神经元点云轮廓采集 -> *Compressed-Domain Contouring (extract_boundary_points)***
   即便是部分不便基于 block 粒度直接进行稀疏特性操作的任务（例如高度依赖连续 2D 平面切片的体素轮廓扫描），我们也可以利用稀疏特性进行极速局部解压以满足原始 OpenCV 算法的视角需求。这赋予了系统极强的向后兼容性，且耗时仅从 0.27-> 0.48s 并未引入过高的耗时，表明将压缩的稀疏特征转化为下家所需的常规结构并未带来过多的额外负担（Overhead）。

5. **点云采样 -> GPU Point Cloud Sampling (Retained)** **(耗时表现持平：0.27s $\to$ 0.29s)**
   在经上述高吞吐量的高效前缀计算出高质量点云表面之后，将结果喂入原有的 GPU 采样逻辑。测试结果明确地展现出：这套高度定制的数据压缩域切入方式并未对异构系统的显存搬运或后置算法引入任何明显的额外的拖拽耗时，确保了与既有工作流的完美衔接。

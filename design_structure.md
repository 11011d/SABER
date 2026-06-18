# Title: A Sparsity-Aware, Compressed-Domain Processing Framework for Pairwise Point Cloud Extraction

## 1. **Step 4: Pairwise Point Cloud Extraction**

In the large-scale and precise automatic proofreading and connection-prediction pipeline for neurons and their connectivity, extracting morphological features from the target connection region (**Step 4: Pairwise Point Cloud Extraction**) plays an absolutely dominant role.
However, because each original neuron break point needs to generate hundreds of possible extension directions, the number of **candidate pairs** becomes extremely large, often increasing to hundreds of times the number of original neurons. These candidate pairs are scattered sparsely and irregularly across the massive three-dimensional brain volume, and each candidate pair requires independent analysis. As a result, this step naturally becomes the most time-consuming stage in the entire pipeline and is heavily dependent on memory bandwidth and CPU resources. To fully utilize heterogeneous computing hardware and avoid the problem of writing back huge intermediate results, the scheduler usually deeply couples this CPU-side extraction stage with the downstream stage that depends entirely on GPU resources (**Step 5: Image Embedding Prediction**) and runs them in **pipeline parallelism**.

In the traditional unoptimized extraction workflow, every core operation is carried out directly on dense arrays that are fetched from the network and fully decrypted. The pipeline mainly consists of the following serial stages:

1. **Raw Voxel Fetching**:
   Load raw voxel blocks that contain the target neuron according to the coordinate region. At this stage, the storage system must align the request to chunk boundaries, fully download and decompress all chunk data indiscriminately, and only then slice out the user-requested region.

2. **Target Mask Extraction**:
   To eliminate the influence of other irrelevant or background neuron branches, traditional equality-based operations such as `np.where(vol_ == segid1, 255, 0)` iterate over the fully materialized dense matrix and generate a binary mask that only represents the current target ID.

3. **Connected Component Extraction**:
   Use `cc3d.connected_components` to perform full 3D analysis on the target mask volume, locate all isolated components, and then use additional spatial distance metrics to filter out the single neuron fragment closest to the center of interest before remasking it.

4. **Neuron Surface Point Cloud Contouring**:
   Use a 2D slice-scanning method similar to `cv2.findContours` to traverse the 3D NumPy array layer by layer and produce a point cloud that captures the outermost surface characteristics of the neuron.

5. **Point Cloud Sampling / Resampling**:
   Upsample or downsample the contour voxels collected in the previous steps to a standard network dimension sequence in order to keep the shape and depth features consistent for downstream tasks.

**Pain points of the traditional workflow:**
This classic workflow is severely constrained by the mechanism of **full dense decompression in memory**, which leads to two fatal problems:

1. **System Memory Bus Contention and the Bandwidth Wall**:
   In a multi-process concurrent environment, forcefully decompressing highly compressed raw neuron data into huge dense arrays causes severe memory-access bottlenecks. Since a large amount of data is used only for extremely low-arithmetic-intensity traversal work and is then immediately discarded, the cache miss rate becomes very high. This forces the CPU to continuously issue massive memory-load requests to main memory through the system bus, creating disastrous memory throughput, with measured redundant memory-traffic peaks as high as 200GB/s.
   The challenge becomes even greater because the system also enables pipeline parallelism with the downstream stage that depends heavily on GPU resources (**Step 5: Image Embedding Prediction**). During Step 5 inference, the large number of high-dimensional feature embeddings generated on the fly causes the data volume to grow rapidly. These large feature matrices must be transferred intensively between CPU main memory and GPU memory through the PCIe bus. This creates a fatal consequence: the increased throughput demand from point-cloud parsing collides directly with the heavy feature-transfer stream on the shared system bus, causing severe bandwidth contention. Eventually both the CPU and GPU are starved for data I/O and become memory-bound.

2. **Redundant Compute and Fetch Waste**:
   The target micro-structures are usually extremely thin and often occupy less than one percent of the effective space inside the cropped bounding box. Traditional dense operators, such as full-array `np.where` scans or 3D `cc3d`, are forced to perform indiscriminate $O(N^3)$ reads and logical comparisons over the vast amount of zero-valued background voxels. This wastes arithmetic resources and, more critically, wastes a large amount of memory bandwidth loading useless background data.

These two pain points are exactly the original motivation for the two major pillars of this system architecture: operating on **compressed data (Compressed Domain)** and exploiting **sparsity (Sparsity)**. By switching to compressed-domain operations while keeping the original physical memory footprint, and by pruning out massive useless reads of zero-valued voxels, we reduced the memory-throughput demand during concurrent execution by more than half (from 200GB/s down to 90GB/s), fundamentally breaking through the **memory wall** and achieving an overall task speedup of more than 4x.

---

## 2. Our Proposed System Design

### 2.1 Core Ideas of the Framework

To completely break the performance barrier caused by full decryption into dense arrays, we propose **A Sparsity-Aware, Compressed-Domain Processing Framework**. The design of this framework is built on two core properties:

*   **Sparsity**: Neuron structures are extremely sparse at this scale in three-dimensional slices. The algorithm should be aware of and exploit this extreme skew in the data, directing the computational frontier precisely onto the truly non-zero topology while skipping the massive background.
*   **Compressed Domain**: After the data is loaded from storage into memory, our custom engine keeps it in the form of split, small **compressed blocks (Blocks)** rather than expanding it prematurely. All low-level lookup, comparison, and computation are specialized to operate directly on the **compressed representation** in a zero-copy manner.

Starting from these two pillars, we reconstructed and provided first-class analysis operators based on the Compressed Voxel Container, allowing complex analysis to be performed directly using metadata (such as per-block palettes) and navigation over compressed structures.

### 2.2 Re-engineered Pipeline in the Proposed Framework

Under this theoretical framework, the original Pairwise Point Cloud Extraction workflow is rebuilt using the following high-speed algorithms. Based on our measured results on 1732 valid neuron sample pairs, the average end-to-end runtime for a single task drops sharply from **7.74s to 1.83s**, achieving an overall **4.2x system-level speedup**. The underlying optimizations and performance breakdown for each step are as follows:

1. **Raw Voxel Fetching -> *On-Demand Partial & Palette-Guided Sparse Loading*** **(10.5x speedup: 2.83s $\to$ 0.26s)**
   This 10.5x speedup comes from a carefully designed **two-level decoding strategy** in our low-level loading engine:
   *   **On-Demand Partial Decompression at Block Granularity**: Under the traditional chunk-access mechanism, even if the user requests only a tiny region of interest (ROI), the framework expands the request outward to align it with large and rigid chunk boundaries, causing a large amount of irrelevant space to be fully decompressed. By breaking through the internal isolation of chunks, we finely extract and decompress only those small blocks that spatially intersect the actual user bounding box, perfectly removing the redundant load caused by alignment inflation.
   *   **Sparsity-Guided Conditional Fetching**: Within the reduced spatial set described above, the system further combines the target features (`segid_list`) to trigger sparsity-aware behavior at the lower level. The fetching engine quickly checks the metadata of each block (such as the palette), and only when the target neuron ID is confirmed to exist inside the block does it actually issue a decompression command.
   These two layers of pruning in both space and feature dimensions complement each other and directly eliminate more than 90% of useless background data from memory I/O and deserialization.

2. **Target Mask Extraction -> *SparsityAwareMasking*** **(3.0x speedup: 0.83s $\to$ 0.28s)**
   In contrast to the traditional full-pixel `where` scan, the new `where(segid1, ...)` method performs the check while preserving the global compressed nature of the data. If a target `segid` is absent from the palette of a small block, the system can turn that block directly into a zero-marked empty block (**Zero-Block**) in $O(1)$ time, completely removing the cost of scanning large decompressed memory regions for a needle in a haystack.

3. **Connected Component Extraction -> *SparseAnchorLocalization* & *SparsityAwareCCA*** **(8.1x speedup: 3.42s $\to$ 0.42s)**
   This stage contributes the most significant absolute runtime reduction in the whole pipeline. We introduced a crucial paradigm reconstruction for connected component (CC) extraction:
   *   **Redundant Over-computation in the Traditional Paradigm**: Traditional connected-component extraction depends on three expensive serial stages. First, the system calls `cc3d.connected_components` to perform a full graph scan over the dense tensor and partition every non-zero feature into isolated components. Second, the system traverses all those isolated components and computes their physical distances to the query center in order to find the target neuron fragment closest to the seed point. Third, the algorithm calls a mask operator again (such as full-array `np.where`) to keep only that chosen component while discarding all other computed connected components as useless results. This causes astonishing over-computation and unnecessary memory-access overhead.
   *   **A Two-Step Local On-Demand Extraction in the New Architecture**: Our design reconstructs this into a strictly de-duplicated two-step pipeline.
   *   **Step 1:** The front-end operator **SparseAnchorLocalization (`nearest_nonzero_idx`)** performs a high-speed nearest-point search. This operation completely abandons pixel-level scanning and instead uses coarse block-tree metadata from the compressed backend to perform large-stride spatial jumps, quickly locking onto the nearest non-zero physical block seed point with almost negligible clock cost.
   *   **Step 2:** Based on this valid anchor point, the core operator **SparsityAwareCCA (`keep_nearest_connected_component_optimized`)** is triggered. Starting from the seed, the traversal expands only along truly connected neuron branches that become valid after decompression, using a local breadth-first search guided by sparsity. The BFS path is guided by palette metadata and blocks mismatched floating interference regions that are disconnected in space or composition. During expansion it rewrites background clutter masks in place, fundamentally eliminating the structural waste of the “analyze everything first, then filter by distance” strategy.

4. **Neuron Surface Point Cloud Contouring -> *Compressed-Domain Contouring (extract_boundary_points)***
   Even for tasks that are not convenient to operate directly at block granularity, such as voxel contour extraction that depends heavily on continuous 2D slice scanning, we can still exploit sparsity through fast local decompression to satisfy the perspective required by the original OpenCV algorithm. This gives the system strong backward compatibility, and the runtime increase from 0.27s to 0.48s remains acceptable, showing that converting compressed sparse features into the downstream-friendly dense structure introduces only limited additional overhead.

5. **Point Cloud Sampling -> GPU Point Cloud Sampling (Retained)** **(essentially unchanged runtime: 0.27s $\to$ 0.29s)**
   After the high-throughput prefix stages efficiently generate high-quality point-cloud surfaces, the results are fed into the original GPU sampling logic. The benchmark results clearly show that this highly customized compressed-domain entry strategy does not introduce any obvious extra overhead in heterogeneous GPU-memory transport or downstream algorithms, ensuring smooth integration with the existing workflow.

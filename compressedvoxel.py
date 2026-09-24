import numpy as np
from typing import Tuple, List, Union, Dict
import compressed_segmentation as cseg
import cc3d, time
from collections import deque

class CompressedVoxelContainer:
    def __init__(self, requested_bbox, full_bbox, block_size: Tuple[int, int, int], dtype: np.dtype):
        # 1. 空间属性定义
        self.requested_bbox = requested_bbox
        self.full_bbox = full_bbox
        self.block_size = np.array(block_size)
        self.dtype = np.dtype(dtype)
        
        # 2. 核心坐标变换向量 (Query -> Physical)
        # px = qx + query_to_phys_offset
        self.query_to_phys_offset = np.array(requested_bbox.minpt) - np.array(full_bbox.minpt)
        
        # 3. 物理网格属性
        # grid_size 定义了基于 full_bbox 的网格维度 (nx, ny, nz)
        self.grid_size = (np.array(full_bbox.size3()) // self.block_size).astype(np.int64)
        self.total_blocks = np.prod(self.grid_size)
        
        # 4. 存储容器：改用 C++ 管理的 PyBlockStore
        self.blocks = cseg.PyBlockStore(int(self.total_blocks), self.dtype)

    def _to_phys_coord(self, q_coord: np.ndarray) -> np.ndarray:
        """将用户请求空间的像素坐标转换为物理 full_bbox 空间的像素坐标"""
        return q_coord + self.query_to_phys_offset

    def _get_block_id_from_phys(self, p_coord: np.ndarray) -> int:
        """从物理像素坐标计算 Block ID"""
        grid_idx = p_coord // self.block_size
        # F-order: x + y*nx + z*nx*ny
        return int(grid_idx[0] + grid_idx[1] * self.grid_size[0] + grid_idx[2] * self.grid_size[0] * self.grid_size[1])

    def query_point(self, qx: int, qy: int, qz: int):
        """查询单个点：逻辑坐标 (0,0,0) 代表请求区域起点"""
        p_coord = self._to_phys_coord(np.array([qx, qy, qz]))
        block_id = self._get_block_id_from_phys(p_coord)
        
        if 0 <= block_id < self.total_blocks:
            block = self.blocks[block_id]  # PyBlockStore.__getitem__ 返回 dict
            if block:
                inner_offset = p_coord % self.block_size
                return block, inner_offset
        return None, None

    def query_interval_blocks(self, q_min: Tuple[int, int, int], q_max: Tuple[int, int, int]) -> List[int]:
        """查询一个区间：返回该区间覆盖的所有物理 Block ID"""
        p_min = self._to_phys_coord(np.array(q_min))
        p_max = self._to_phys_coord(np.array(q_max))
        
        s_grid = p_min // self.block_size
        e_grid = (p_max + self.block_size - 1) // self.block_size
        
        ids = []
        for iz in range(s_grid[2], e_grid[2]):
            z_off = iz * self.grid_size[0] * self.grid_size[1]
            for iy in range(s_grid[1], e_grid[1]):
                y_off = iy * self.grid_size[0]
                for ix in range(s_grid[0], e_grid[0]):
                    ids.append(int(ix + y_off + z_off))
        return ids



    def get_raw_data(self, q_min: Tuple[int, int, int], q_max: Tuple[int, int, int]):
        """
        获取逻辑空间指定范围内的原始数据 (NumPy 数组)
        """
        p_min = self._to_phys_coord(np.array(q_min))
        p_max = self._to_phys_coord(np.array(q_max))
        
        s_grid = p_min // self.block_size
        e_grid = (p_max + self.block_size - 1) // self.block_size
        grid_dims = e_grid - s_grid

        block_ids = self.query_interval_blocks(q_min, q_max)
        
        # 非热路径：用旧接口兼容（通过 PyBlockStore.__getitem__ 获取 dict 列表）
        block_data_list = [self.blocks[bid] for bid in block_ids]

        aligned_buffer = cseg.decompress_block_grid(
            block_data_list,
            tuple(self.block_size),
            tuple(grid_dims),
            self.dtype
        )

        buffer_origin = s_grid * self.block_size
        rel_start = p_min - buffer_origin
        rel_end = rel_start + (np.array(q_max) - np.array(q_min))

        return aligned_buffer[
            rel_start[0]:rel_end[0],
            rel_start[1]:rel_end[1],
            rel_start[2]:rel_end[2]
        ]
    
    # 测试使用
    def get_all_blocks_dense(self):
        """
        [测试与诊断专用]
        无视一切请求坐标，直接将当前容器底层的所有 Blocks 强行解压为稠密 NumPy 矩阵。
        """
        return cseg.decompress_block_grid_store(
            self.blocks,
            tuple(self.block_size),
            tuple(self.grid_size),
            self.dtype
        )

    def distinct_labels_in_mask(self, mask):
        """Return labels selected by a query-local boolean mask.

        Fully selected interior blocks are answered from their palettes. Blocks
        intersecting only part of the mask decode selected voxel indices only;
        blocks disjoint from the mask are skipped.
        """
        mask = np.asarray(mask)
        if mask.dtype != np.bool_:
            raise TypeError("mask must be a boolean array")
        expected_shape = tuple(int(value) for value in self.requested_bbox.size3())
        if mask.shape != expected_shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match requested bbox {expected_shape}"
            )
        return cseg.distinct_labels_in_mask_store(
            self.blocks,
            np.ascontiguousarray(mask, dtype=np.uint8),
            tuple(self.grid_size),
            tuple(self.block_size),
            expected_shape,
            tuple(self.query_to_phys_offset),
            self.dtype,
        )

    def discover_labels(self, spatial_selection):
        """Query operator: reduce labels selected by a query-local predicate."""
        return self.distinct_labels_in_mask(spatial_selection)

    def resident_storage_stats(self):
        """Describe the compressed payload established by the Block Loader."""
        return self.blocks.storage_stats()

    def extract_label_contours_3d(self, labels, centers=None, origin=None):
        """Query operator for one or more six-neighbor label contours.

        Without ``centers``, the planner selects either the fixed-cardinality
        per-label fused plan or the generic multi-label block map. With
        centers, each label is restricted to the center-nearest 26-connected
        component before its boundary is emitted. Coordinates remain in voxel
        space; dataset-specific physical scaling belongs to the caller.
        """
        labels_array = np.asarray(tuple(int(value) for value in labels), dtype=self.dtype)
        if labels_array.ndim != 1 or labels_array.size == 0:
            raise ValueError("labels must be a nonempty one-dimensional sequence")
        if np.any(labels_array == 0):
            raise ValueError("labels must be positive")
        if np.unique(labels_array).size != labels_array.size:
            raise ValueError("labels must be unique")

        centers_array = None
        if centers is not None:
            centers_array = np.asarray(centers, dtype=np.int64)
            if centers_array.shape == (3,):
                centers_array = np.broadcast_to(
                    centers_array, (labels_array.size, 3)
                ).copy()
            if centers_array.shape != (labels_array.size, 3):
                raise ValueError("centers must have shape (3,) or (len(labels), 3)")
            request_size = np.asarray(self.requested_bbox.size3(), dtype=np.int64)
            if np.any(centers_array < 0) or np.any(centers_array >= request_size):
                raise ValueError("centers must lie inside the requested voxel bbox")

        contours, metrics = cseg.extract_label_contours_3d_store(
            self.blocks,
            tuple(self.grid_size),
            tuple(self.block_size),
            tuple(self.requested_bbox.size3()),
            tuple(self.query_to_phys_offset),
            labels_array,
            centers_array,
            self.dtype,
        )
        if origin is not None:
            origin_array = np.asarray(origin)
            if origin_array.shape != (3,):
                raise ValueError("origin must contain three voxel coordinates")
            contours = [points + origin_array for points in contours]
        return tuple(contours), metrics
    
    def where(self, segid: int, true_val: int, false_val: int, out_dtype=None):
        """
        高斯能压缩态条件筛选器 (np.where 的高性能替代方案)。
        直接操作 C++ BlockArena，跳过所有 Python dict 操作。
        """
        if out_dtype is None:
            out_dtype = self.dtype
        out_dtype = np.dtype(out_dtype)
        
        # 创建结果容器
        res = CompressedVoxelContainer(
            self.requested_bbox, 
            self.full_bbox, 
            tuple(self.block_size), 
            out_dtype
        )
        
        # 直接操作 C++ BlockArena
        cseg.transform_where_compressed_store(
            self.blocks,
            res.blocks,
            self.dtype,
            segid,
            true_val,
            false_val,
            out_dtype,
            tuple(self.block_size)
        )
        return res

    def binary_mask(self, segid):
        """Return a sparse compressed uint8 mask for one positive segment ID."""
        result = CompressedVoxelContainer(
            self.requested_bbox,
            self.full_bbox,
            tuple(self.block_size),
            np.uint8,
        )
        cseg.binary_mask_compressed_store(
            self.blocks,
            result.blocks,
            segid,
            self.dtype,
            tuple(self.block_size),
        )
        return result

    def _require_same_layout(self, other, operation):
        if not isinstance(other, CompressedVoxelContainer):
            raise TypeError(f"{operation} requires a CompressedVoxelContainer")
        attributes = (
            (self.requested_bbox.minpt, other.requested_bbox.minpt),
            (self.requested_bbox.maxpt, other.requested_bbox.maxpt),
            (self.full_bbox.minpt, other.full_bbox.minpt),
            (self.full_bbox.maxpt, other.full_bbox.maxpt),
            (self.block_size, other.block_size),
            (self.grid_size, other.grid_size),
            (self.query_to_phys_offset, other.query_to_phys_offset),
        )
        if any(not np.array_equal(left, right) for left, right in attributes):
            raise ValueError(f"{operation} requires identical voxel-space layouts")

    def binary_union(self, other):
        """Union two compressed uint8 0/1 masks block by block."""
        self._require_same_layout(other, "binary_union")
        if self.dtype != np.dtype(np.uint8) or other.dtype != np.dtype(np.uint8):
            raise TypeError("binary_union accepts only uint8 0/1 masks")
        result = CompressedVoxelContainer(
            self.requested_bbox,
            self.full_bbox,
            tuple(self.block_size),
            np.uint8,
        )
        cseg.binary_union_compressed_stores(
            self.blocks,
            other.blocks,
            result.blocks,
            tuple(self.block_size),
        )
        return result

    def materialize_binary_channels(self, mask_one, mask_two, output_shape_zyx):
        """Materialize ``[self, mask_one, mask_two]`` from sparse binary blocks."""
        self._require_same_layout(mask_one, "materialize_binary_channels")
        self._require_same_layout(mask_two, "materialize_binary_channels")
        if any(
            container.dtype != np.dtype(np.uint8)
            for container in (self, mask_one, mask_two)
        ):
            raise TypeError("binary materialization accepts only uint8 0/1 masks")
        output_shape_zyx = tuple(int(value) for value in output_shape_zyx)
        if len(output_shape_zyx) != 3 or any(value <= 0 for value in output_shape_zyx):
            raise ValueError("output_shape_zyx must contain three positive dimensions")
        return cseg.materialize_binary_channels_3d_stores(
            self.blocks,
            mask_one.blocks,
            mask_two.blocks,
            tuple(self.grid_size),
            tuple(self.block_size),
            tuple(self.requested_bbox.size3()),
            tuple(self.query_to_phys_offset),
            output_shape_zyx,
        )
    
    def nearest_nonzero_idx(self, x, y, z):
        """极速 C++ 找最近种子点（直接使用 BlockArena CBlock*）"""
        return cseg.find_nearest_seed_fast_store(
            self.blocks, 
            tuple(self.grid_size), 
            tuple(self.block_size), 
            tuple(self.requested_bbox.size3()), 
            tuple(self.query_to_phys_offset),
            (int(x), int(y), int(z)),
            self.dtype
        )
    
    def get_nearest_nonzero_value(self, x, y, z):
        """封装方法：获取最近非零点的标签值"""
        idx = self.nearest_nonzero_idx(x, y, z)
        if idx is not None:
            val = self.get_raw_data(tuple(idx), tuple(idx + 1))
            return int(val.flatten()[0])
        return 0
    
    def keep_nearest_connected_component_optimized(self, center_x, center_y, center_z):
        """
        直接利用已知种子点执行极致的 C++ BFS，结果原地写回 BlockArena。
        """
        seed_arr = self.nearest_nonzero_idx(center_x, center_y, center_z)
        if seed_arr is None:
            # 全部清零：构建单元素零调色板，批量设置
            pal_false = np.array([0], dtype=self.dtype)
            for i in range(int(self.total_blocks)):
                self.blocks.set_block(i, pal_false, 0, None)
            return self

        cseg.extract_cc_fast_store(
            self.blocks, 
            tuple(self.grid_size), 
            tuple(self.block_size), 
            tuple(self.requested_bbox.size3()), 
            tuple(self.query_to_phys_offset),
            tuple(seed_arr), 
            self.dtype
        )
        return self

    def extract_boundary_points(
        self,
        x_off,
        y_off,
        z_off,
        lx,
        ly,
        lz,
        pc_data,
        ids_data,
        label=0,
        origin=None,
    ):
        """Extract legacy per-slice 2D contour points from a filtered mask.

        Args:
            x_off, y_off, z_off: Historical global-center arguments. Retained
                only for source compatibility and not used for coordinates.
            lx, ly, lz: Historical local-center offsets. Retained only for
                source compatibility and not used for coordinates.
            pc_data: Mutable list to which ``float32`` arrays of shape
                ``(N, 3)`` are appended, one array per 2D contour.
            ids_data: Mutable list extended with ``label`` once per emitted
                point.
            label: Integer output label associated with every emitted point.
            origin: Optional three-element voxel-coordinate origin for the
                requested bbox. Defaults to ``requested_bbox.minpt``.

        Returns:
            The same ``(pc_data, ids_data)`` containers after appending points.

        Each non-empty z slice is converted with ``cv2.findContours``. The
        output is in CloudVolume voxel coordinates: ``origin + local_voxel``.
        This method never accepts, stores, or applies physical voxel resolution.
        """
        del x_off, y_off, z_off, lx, ly, lz

        if origin is None:
            origin = self.requested_bbox.minpt
        origin_arr = np.asarray(origin, dtype=np.float32)
        if origin_arr.shape != (3,):
            raise ValueError(f"origin must contain three voxel coordinates, got {origin!r}")

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("extract_boundary_points requires OpenCV (cv2)") from exc

        nx_b, ny_b, nz_b = self.grid_size
        bx, by, bz = self.block_size
        slab_shape = (nx_b * bx, ny_b * by, bz)
        shared_buffer = np.zeros(slab_shape, dtype=self.dtype, order="F")

        req_start_rel = np.asarray(self.requested_bbox.minpt) - np.asarray(self.full_bbox.minpt)
        req_end_rel = req_start_rel + np.asarray(self.requested_bbox.size3())

        for gz in range(nz_b):
            z_start = gz * bz
            if z_start >= req_end_rel[2]:
                break
            if z_start + bz <= req_start_rel[2]:
                continue

            shared_buffer.fill(0)
            cseg.fill_slab_buffer_store(
                self.blocks,
                shared_buffer,
                (bx, by, bz),
                (nx_b, ny_b),
                self.dtype,
                int(gz),
            )

            for i_lz in range(bz):
                abs_z = z_start + i_lz
                if not (req_start_rel[2] <= abs_z < req_end_rel[2]):
                    continue

                data_slice = shared_buffer[:, :, i_lz]
                if not np.any(data_slice):
                    continue

                binary_slice = np.ascontiguousarray(data_slice != 0, dtype=np.uint8)
                contours, _ = cv2.findContours(binary_slice, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    if contour.size == 0:
                        continue
                    boundary = contour.reshape(-1, 2)
                    points = np.empty((boundary.shape[0], 3), dtype=np.float32)
                    points[:, 0] = origin_arr[0] + boundary[:, 1] - req_start_rel[0]
                    points[:, 1] = origin_arr[1] + boundary[:, 0] - req_start_rel[1]
                    points[:, 2] = origin_arr[2] + abs_z - req_start_rel[2]
                    pc_data.append(points)
                    ids_data.extend([label] * boundary.shape[0])

        return pc_data, ids_data

    def extract_boundary_voxel_points_3d(
        self,
        origin,
        pc_data,
        ids_data,
        label=0,
    ):
        """
        Extract 6-neighbor 3D boundary voxel points from a filtered mask.

        Args:
            origin: Three-element requested-bbox origin in CloudVolume voxel
                coordinates.
            pc_data: Mutable list to which one ``float32`` array of shape
                ``(N, 3)`` is appended when points are found.
            ids_data: Mutable list extended with ``label`` once per emitted
                point.
            label: Integer output label associated with every emitted point.

        Returns:
            The same ``(pc_data, ids_data)`` containers after appending points.

        Traversal, block decoding, neighbor checks, and coordinate generation
        run in C++. The coordinate rule is ``point = origin + local_voxel``;
        SABER therefore neither accepts nor applies physical voxel resolution.
        """
        points, ids = cseg.extract_boundary_voxels_3d_store(
            self.blocks,
            tuple(self.grid_size),
            tuple(self.block_size),
            tuple(self.requested_bbox.size3()),
            tuple(self.query_to_phys_offset),
            tuple(float(v) for v in origin),
            int(label),
            self.dtype,
        )
        if points.shape[0] > 0:
            pc_data.append(points)
            ids_data.extend(ids.tolist())
        return pc_data, ids_data

    def extract_nonzero_voxel_coordinates(self, origin=None):
        """Return sparse coordinates without materializing the requested bbox."""
        points = cseg.extract_nonzero_voxels_3d_store(
            self.blocks,
            tuple(self.grid_size),
            tuple(self.block_size),
            tuple(self.requested_bbox.size3()),
            tuple(self.query_to_phys_offset),
            self.dtype,
        )
        if origin is None:
            return points
        origin_arr = np.asarray(origin, dtype=np.int64)
        if origin_arr.shape != (3,):
            raise ValueError(f"origin must contain three voxel coordinates, got {origin!r}")
        return points + origin_arr

    def extract_matching_voxel_coordinates(self, labels, origin=None):
        """Return sparse coordinates and values for an explicit label set."""
        points, values = cseg.extract_matching_voxels_3d_store(
            self.blocks,
            tuple(self.grid_size),
            tuple(self.block_size),
            tuple(self.requested_bbox.size3()),
            tuple(self.query_to_phys_offset),
            labels,
            self.dtype,
        )
        if origin is not None:
            origin_arr = np.asarray(origin, dtype=np.int64)
            if origin_arr.shape != (3,):
                raise ValueError(f"origin must contain three voxel coordinates, got {origin!r}")
            points = points + origin_arr
        return points, values

    def extract_label_voxels_and_contacts(self, primary_label, contact_labels):
        """Return target coordinates and contacted labels without a dense mask."""
        return cseg.extract_label_voxels_and_contacts_3d_store(
            self.blocks,
            tuple(self.grid_size),
            tuple(self.block_size),
            tuple(self.requested_bbox.size3()),
            tuple(self.query_to_phys_offset),
            primary_label,
            contact_labels,
            self.dtype,
        )

    def extract_label_voxels_and_boundary(self, primary_label):
        """Return solid and 6-neighbor boundary coordinates in one block traversal."""
        return cseg.extract_label_voxels_and_boundary_3d_store(
            self.blocks,
            tuple(self.grid_size),
            tuple(self.block_size),
            tuple(self.requested_bbox.size3()),
            tuple(self.query_to_phys_offset),
            primary_label,
            self.dtype,
        )

    def extract_pair_feature(self, label_one, label_two, output_shape_zyx):
        """Return a fixed-size ``[union, label_one, label_two]`` uint8 tensor.

        The container and output shape are expressed only in voxel coordinates.
        Physical resolution and the conversion from a physical radius to the
        requested bbox belong to the caller.
        """
        return cseg.extract_pair_feature_3d_store(
            self.blocks,
            tuple(self.grid_size),
            tuple(self.block_size),
            tuple(self.requested_bbox.size3()),
            tuple(self.query_to_phys_offset),
            label_one,
            label_two,
            tuple(output_shape_zyx),
            self.dtype,
        )

    def project_label_channels(
        self,
        labels,
        channel_membership,
        output_shape_zyx,
        return_plan=False,
    ):
        """Query operator mapping label predicates into fixed tensor channels."""
        labels_array = np.asarray(tuple(int(value) for value in labels), dtype=self.dtype)
        membership = np.asarray(channel_membership, dtype=np.uint8)
        output, physical_plan = cseg.project_label_channels_3d_store(
            self.blocks,
            tuple(self.grid_size),
            tuple(self.block_size),
            tuple(self.requested_bbox.size3()),
            tuple(self.query_to_phys_offset),
            labels_array,
            membership,
            tuple(output_shape_zyx),
            self.dtype,
        )
        if return_plan:
            return output, physical_plan
        return output

    def project_pair_feature(self, label_one, label_two, output_shape_zyx, return_plan=False):
        """Express the P3 pair feature through the generic projection operator."""
        # The planner recognizes this fixed membership contract and dispatches
        # the pair3-fused physical kernel without materializing masks.
        output, physical_plan = self.project_label_channels(
            (label_one, label_two),
            ((1, 1), (1, 0), (0, 1)),
            output_shape_zyx,
            return_plan=True,
        )
        return (output, physical_plan) if return_plan else output


def extract_pair_features_batch(requests, output_shape_zyx, parallel=0):
    """Extract a homogeneous pair-feature batch into one contiguous array.

    ``requests`` contains ``(container, label_one, label_two)`` tuples. All
    containers must use the same label dtype, while their bbox and block-grid
    dimensions may differ. Node and Edge requests should be submitted as
    separate batches because they have different output shapes.
    """
    requests = list(requests)
    output_shape_zyx = tuple(int(value) for value in output_shape_zyx)
    if len(output_shape_zyx) != 3 or any(value <= 0 for value in output_shape_zyx):
        raise ValueError("output_shape_zyx must contain three positive voxel dimensions")
    if int(parallel) < 0:
        raise ValueError("parallel must be nonnegative")
    if not requests:
        return np.empty((0, 3) + output_shape_zyx, dtype=np.uint8)

    count = len(requests)
    first_request = requests[0]
    if len(first_request) != 3:
        raise ValueError("each request must be (container, label_one, label_two)")
    first_container = first_request[0]
    if not isinstance(first_container, CompressedVoxelContainer):
        raise TypeError("each batch request must contain a CompressedVoxelContainer")
    dtype = first_container.dtype
    if dtype not in (np.dtype(np.uint8), np.dtype(np.uint32), np.dtype(np.uint64)):
        raise TypeError("pair features require uint8, uint32, or uint64 labels")
    maximum_label = int(np.iinfo(dtype).max)

    stores = []
    grid_sizes = np.empty((count, 3), dtype=np.int64)
    block_sizes = np.empty((count, 3), dtype=np.int64)
    req_sizes = np.empty((count, 3), dtype=np.int64)
    q2p_offsets = np.empty((count, 3), dtype=np.int64)
    label_pairs = np.empty((count, 2), dtype=np.uint64)

    for index, request in enumerate(requests):
        if len(request) != 3:
            raise ValueError("each request must be (container, label_one, label_two)")
        container, label_one, label_two = request
        if not isinstance(container, CompressedVoxelContainer):
            raise TypeError("each batch request must contain a CompressedVoxelContainer")
        if container.dtype != dtype:
            raise TypeError("all batch containers must have the same dtype")
        label_one = int(label_one)
        label_two = int(label_two)
        if (label_one <= 0 or label_two <= 0 or
                label_one > maximum_label or label_two > maximum_label):
            raise ValueError("pair labels must be positive and representable by the container dtype")
        stores.append(container.blocks)
        grid_sizes[index] = container.grid_size
        block_sizes[index] = container.block_size
        req_sizes[index] = container.requested_bbox.size3()
        q2p_offsets[index] = container.query_to_phys_offset
        label_pairs[index] = (label_one, label_two)

    return cseg.extract_pair_features_batch_3d_stores(
        stores,
        grid_sizes,
        block_sizes,
        req_sizes,
        q2p_offsets,
        label_pairs,
        output_shape_zyx,
        dtype,
        int(parallel),
    )

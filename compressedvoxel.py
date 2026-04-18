import numpy as np
from typing import Tuple, List, Union, Dict
import compressed_segmentation as cseg
import cc3d, time
import cv2
from collections import deque
cv2.setNumThreads(0)

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
    
    def extract_boundary_points(self, x_off, y_off, z_off, lx, ly, lz, pc_data, ids_data, label=0):
        """
        重新设计的提取方法：对齐 Slab 处理，C++ 批量解压。
        （直接使用 fill_slab_buffer_store，传 gz 索引替代 Python 切片）
        """
        nx_b, ny_b, nz_b = self.grid_size
        bx, by, bz = self.block_size
        
        # 1. 预申请全平面（Slab）Buffer，复用内存
        slab_shape = (nx_b * bx, ny_b * by, bz)
        shared_buffer = np.zeros(slab_shape, dtype=self.dtype, order='F')

        req_start_rel = np.array(self.requested_bbox.minpt) - np.array(self.full_bbox.minpt)
        req_end_rel = req_start_rel + np.array(self.requested_bbox.size3())

        for gz in range(nz_b):
            z_start = gz * bz
            # 范围剪枝
            if z_start >= req_end_rel[2]: break
            if z_start + bz <= req_start_rel[2]: continue

            # 2. 极速重置并填充（直接传 gz，避免 Python 切片）
            shared_buffer.fill(0)
            cseg.fill_slab_buffer_store(
                self.blocks, shared_buffer,
                (bx, by, bz), (nx_b, ny_b),
                self.dtype, int(gz)
            )

            # 3. 逐层提取轮廓
            for i_lz in range(bz):
                abs_z = z_start + i_lz
                if not (req_start_rel[2] <= abs_z < req_end_rel[2]):
                    continue

                data_slice = shared_buffer[:, :, i_lz]
                
                if not np.any(data_slice):
                    continue

                contours, _ = cv2.findContours(data_slice, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                
                for cnt in contours:
                    if cnt.size == 0: continue
                    boundary = cnt.getfield(cnt.dtype, 0).reshape(-1, 2)
                    
                    num_pts = boundary.shape[0]
                    
                    res_pc = np.empty((num_pts, 3), dtype=np.float32)
                    res_pc[:, 0] = (boundary[:, 1] - req_start_rel[0]) * 16 + x_off * 4 - lx * 16
                    res_pc[:, 1] = (boundary[:, 0] - req_start_rel[1]) * 16 + y_off * 4 - ly * 16
                    res_pc[:, 2] = (abs_z - req_start_rel[2] + z_off - lz) * 40

                    pc_data.append(res_pc)
                    ids_data.extend([label] * num_pts)

        return pc_data, ids_data
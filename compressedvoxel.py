import numpy as np
from typing import Tuple, List, Union, Dict
import compressed_segmentation as cseg
import cc3d, time
import cv2
from collections import deque
cv2.setNumThreads(0)

class CompressedVoxelContainer:
    def __init__(self, requested_bbox, full_bbox, block_size: Tuple[int, int, int], dtype: np.dtype):
        # 1. Spatial attribute definitions.
        self.requested_bbox = requested_bbox
        self.full_bbox = full_bbox
        self.block_size = np.array(block_size)
        self.dtype = np.dtype(dtype)
        
        # 2. Core coordinate transform vector (Query -> Physical).
        # px = qx + query_to_phys_offset
        self.query_to_phys_offset = np.array(requested_bbox.minpt) - np.array(full_bbox.minpt)
        
        # 3. Physical grid properties.
        # grid_size defines the grid dimensions derived from full_bbox (nx, ny, nz).
        self.grid_size = (np.array(full_bbox.size3()) // self.block_size).astype(np.int64)
        self.total_blocks = np.prod(self.grid_size)
        
        # 4. Storage container: use the C++-managed PyBlockStore.
        self.blocks = cseg.PyBlockStore(int(self.total_blocks), self.dtype)

    def _to_phys_coord(self, q_coord: np.ndarray) -> np.ndarray:
        """Convert voxel coordinates from request space to physical full_bbox coordinates."""
        return q_coord + self.query_to_phys_offset

    def _get_block_id_from_phys(self, p_coord: np.ndarray) -> int:
        """Compute the block ID from a physical voxel coordinate."""
        grid_idx = p_coord // self.block_size
        # F-order: x + y*nx + z*nx*ny
        return int(grid_idx[0] + grid_idx[1] * self.grid_size[0] + grid_idx[2] * self.grid_size[0] * self.grid_size[1])

    def query_point(self, qx: int, qy: int, qz: int):
        """Query a single point: logical coordinate (0,0,0) is the request-region origin."""
        p_coord = self._to_phys_coord(np.array([qx, qy, qz]))
        block_id = self._get_block_id_from_phys(p_coord)
        
        if 0 <= block_id < self.total_blocks:
            block = self.blocks[block_id]  # PyBlockStore.__getitem__ returns a dict.
            if block:
                inner_offset = p_coord % self.block_size
                return block, inner_offset
        return None, None

    def query_interval_blocks(self, q_min: Tuple[int, int, int], q_max: Tuple[int, int, int]) -> List[int]:
        """Query an interval and return all physical block IDs covered by it."""
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
        Fetch raw data for the requested range in logical space as a NumPy array.
        """
        p_min = self._to_phys_coord(np.array(q_min))
        p_max = self._to_phys_coord(np.array(q_max))
        
        s_grid = p_min // self.block_size
        e_grid = (p_max + self.block_size - 1) // self.block_size
        grid_dims = e_grid - s_grid

        block_ids = self.query_interval_blocks(q_min, q_max)
        
        # Non-hot path: keep compatibility with the legacy interface via PyBlockStore.__getitem__.
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
    
    # For testing.
    def get_all_blocks_dense(self):
        """
        [For testing and diagnostics only]
        Ignore request coordinates and decompress all underlying blocks in the container into a dense NumPy array.
        """
        return cseg.decompress_block_grid_store(
            self.blocks,
            tuple(self.block_size),
            tuple(self.grid_size),
            self.dtype
        )
    
    def where(self, segid: int, true_val: int, false_val: int, out_dtype=None):
        """
        A compressed-domain conditional filter, serving as a high-performance alternative to np.where.
        It operates directly on the C++ BlockArena and skips all Python dict operations.
        """
        if out_dtype is None:
            out_dtype = self.dtype
        out_dtype = np.dtype(out_dtype)
        
        # Create the result container.
        res = CompressedVoxelContainer(
            self.requested_bbox, 
            self.full_bbox, 
            tuple(self.block_size), 
            out_dtype
        )
        
        # Operate directly on the C++ BlockArena.
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
        """Find the nearest seed point with the fast C++ path using BlockArena CBlock* directly."""
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
        """Wrapper method: get the label value of the nearest non-zero point."""
        idx = self.nearest_nonzero_idx(x, y, z)
        if idx is not None:
            val = self.get_raw_data(tuple(idx), tuple(idx + 1))
            return int(val.flatten()[0])
        return 0
    
    def keep_nearest_connected_component_optimized(self, center_x, center_y, center_z):
        """
        Use the known seed point to run the optimized C++ BFS, then write the result back to BlockArena in place.
        """
        seed_arr = self.nearest_nonzero_idx(center_x, center_y, center_z)
        if seed_arr is None:
            # Clear everything by constructing a single-element zero palette and applying it in bulk.
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
        Redesigned extraction method: slab-aligned processing with batched C++ decompression.
        It calls fill_slab_buffer_store directly and passes the gz index instead of using Python slicing.
        """
        nx_b, ny_b, nz_b = self.grid_size
        bx, by, bz = self.block_size
        
        # 1. Preallocate the full slab buffer and reuse it.
        slab_shape = (nx_b * bx, ny_b * by, bz)
        shared_buffer = np.zeros(slab_shape, dtype=self.dtype, order='F')

        req_start_rel = np.array(self.requested_bbox.minpt) - np.array(self.full_bbox.minpt)
        req_end_rel = req_start_rel + np.array(self.requested_bbox.size3())

        for gz in range(nz_b):
            z_start = gz * bz
            # Range pruning.
            if z_start >= req_end_rel[2]: break
            if z_start + bz <= req_start_rel[2]: continue

            # 2. Fast reset and fill, passing gz directly to avoid Python slicing.
            shared_buffer.fill(0)
            cseg.fill_slab_buffer_store(
                self.blocks, shared_buffer,
                (bx, by, bz), (nx_b, ny_b),
                self.dtype, int(gz)
            )

            # 3. Extract contours layer by layer.
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

import os
import sys
import time
import numpy as np
import cc3d
import traceback

import random
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- 路径配置 ---
LOCAL_CLONE = './cloud-volume'
if os.path.exists(LOCAL_CLONE):
    sys.path.insert(0, LOCAL_CLONE)

import compressed_segmentation as cseg
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox
from compressedvoxel import CompressedVoxelContainer

LOCAL_EXTENSION_DIR = os.path.abspath(os.path.dirname(__file__))
if os.path.dirname(os.path.abspath(cseg.__file__)) != LOCAL_EXTENSION_DIR:
    raise ImportError(
        "test.py must use the local compressed_segmentation extension. "
        f"Expected it under {LOCAL_EXTENSION_DIR}, got {cseg.__file__}."
    )


USE_COMPRESSED_BLOCK = True
BOUNDARY_MODE = os.environ.get("SABER_BOUNDARY_MODE", "3d").lower()
if BOUNDARY_MODE not in {"3d", "legacy_2d"}:
    raise ValueError("SABER_BOUNDARY_MODE must be '3d' or 'legacy_2d'")

def nearest_nonzero_idx(a, x, y, z):
    idx = np.argwhere(a)

    # If (x,y) itself is also non-zero, we want to avoid those, so delete that
    # But, if we are sure that (x,y) won't be non-zero, skip the next step
    idx = idx[~(idx == [x, y, z]).all(1)]

    dists = ((idx - [x, y, z]) ** 2).sum(1)
    min_dist = dists.min()
    is_unique = int((dists == min_dist).sum()) == 1  # 最近点是否唯一
    return idx[dists.argmin()], is_unique


def extract_boundary_voxel_points_dense(binary_vol, origin, pc_data, ids_data, label=0):
    """
    Dense reference for 3D 6-neighbor boundary voxel point clouds.

    SABER is a storage/query tool and emits coordinates in the same data
    coordinate system as CloudVolume slicing. It is intentionally unaware of
    dataset-specific physical resolution such as FAFB's nm voxel size.
    """
    mask = binary_vol != 0
    if not np.any(mask):
        return pc_data, ids_data

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    all_six_neighbors = (
        padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    boundary = padded[1:-1, 1:-1, 1:-1] & ~all_six_neighbors
    coords = np.argwhere(boundary)
    if coords.size == 0:
        return pc_data, ids_data

    points = np.empty((coords.shape[0], 3), dtype=np.float32)
    points[:, 0] = coords[:, 0] + origin[0]
    points[:, 1] = coords[:, 1] + origin[1]
    points[:, 2] = coords[:, 2] + origin[2]
    pc_data.append(points)
    ids_data.extend([label] * coords.shape[0])
    return pc_data, ids_data


def extract_boundary_points_dense(binary_vol, origin, pc_data, ids_data, label=0):
    """Dense reference for the legacy per-slice 2D contour API.

    ``origin`` is in CloudVolume voxel coordinates. Physical resolution is
    intentionally outside this test path and the SABER implementation.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("legacy_2d boundary mode requires OpenCV (cv2)") from exc

    mask = binary_vol != 0
    for z_index in range(mask.shape[2]):
        binary_slice = np.ascontiguousarray(mask[:, :, z_index], dtype=np.uint8)
        if not np.any(binary_slice):
            continue

        contours, _ = cv2.findContours(binary_slice, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if contour.size == 0:
                continue
            boundary = contour.reshape(-1, 2)
            points = np.empty((boundary.shape[0], 3), dtype=np.float32)
            points[:, 0] = origin[0] + boundary[:, 1]
            points[:, 1] = origin[1] + boundary[:, 0]
            points[:, 2] = origin[2] + z_index
            pc_data.append(points)
            ids_data.extend([label] * boundary.shape[0])
    return pc_data, ids_data


def scan_valid_indices(vol, df, lx=80, ly=80, lz=32):
    """
    用原始模式扫描 df 的每一行，找出 nearest_nonzero_idx 答案唯一的有效 case 下标。
    返回: (valid_indices, skipped_indices)
    """
    valid, skipped = [], []
    N = len(df)
    for ii in tqdm(range(N), desc="扫描有效下标"):
        segid1 = df.iloc[ii, 0]
        segid2 = df.iloc[ii, 1]
        cord = np.array(df.iloc[ii, 2].strip('[]').split(','))
        x, y, z = float(cord[0]), float(cord[1]), float(cord[2])
        x = int(np.round(x / 4) * 4)
        y = int(np.round(y / 4) * 4)
        z = int(np.round(z))
        x0, x1 = int(x / 4 - lx), int(x / 4 + lx)
        y0, y1 = int(y / 4 - ly), int(y / 4 + ly)
        z0, z1 = int(z - lz), int(z + lz)

        vol_ = vol[x0:x1, y0:y1, z0:z1]

        # vol0 唯一性检查
        vol0 = np.where(vol_ == segid1, 255, 0).astype(np.uint8)
        cc0  = cc3d.connected_components(vol0[:, :, :, 0], out_dtype=np.uint64)
        _, u0 = nearest_nonzero_idx(cc0, lx, ly, lz)

        # vol1 唯一性检查
        vol1 = np.where(vol_ == segid2, 255, 0).astype(np.uint8)
        cc1  = cc3d.connected_components(vol1[:, :, :, 0], out_dtype=np.uint64)
        _, u1 = nearest_nonzero_idx(cc1, lx, ly, lz)

        if u0 and u1:
            valid.append(ii)
        else:
            skipped.append(ii)

    print(f"[扫描完成] 有效 case: {len(valid)}, 跳过 case: {len(skipped)}")
    return valid, skipped


def work(vol, valid_indices):
    """only process the rows in valid_indices."""
    global USE_COMPRESSED_BLOCK
    bbox=[[425652., 126020., 150360.],[490488., 170152., 206240.]]
    split= 8
    candidate_file = "./data/candidate0.csv"
    lx, ly, lz, step = 80, 80, 32, 1

    hx,hy,hz = 20,20,32
    df = pd.read_csv(candidate_file)
    lenpc=0
    dir_path = os.path.dirname(candidate_file)
    name = os.path.basename(candidate_file)
    if os.path.exists(dir_path):
        all_pc = []
        all_ids = []
        processed_count = 0  # 正常处理的 case 数
        
        total_time_fetch = 0
        total_time_where = 0
        total_time_cc = 0
        total_time_boundary = 0
        
        from collections import defaultdict
        first = True
        for ii in tqdm(valid_indices, desc="处理", disable=True):
            # if first:
            #     first = False
            # else:
            #     break
            pc_data = []
            ids_data = []
            segid1 = df.iloc[ii, 0]
            segid2 = df.iloc[ii, 1]
            cord =np.array(df.iloc[ii,2].strip('[]').split(','))
            x, y, z = [float(cord[0]), float(cord[1]), float(cord[2])]
            x = int(np.round(x / 4) * 4)  # 四舍五入到最近的4的倍数
            y = int(np.round(y / 4) * 4)  # 四舍五入到最近的4的倍数
            z = int(np.round(z))  # 直接四舍五入
            cord_start_begin = bbox[0] / np.array([4, 4, 40])
            cord_end_end = bbox[1] / np.array([4, 4, 40])
            cord_end_end = (cord_end_end - cord_start_begin) / np.array([split, split, split]) + cord_start_begin
            
            cord_start_begin = np.round(cord_start_begin / 4) * 4
            cord_end_end = np.round(cord_end_end / 4) * 4

            sg1=1
            sg2=1        
            if(sg1 or sg2):
                pc_data = []
                ids_data =[]
                t_start = time.time()
                if USE_COMPRESSED_BLOCK:
                    vol.segid_list = [segid1, segid2]
                x0, x1 = int(x / 4 - lx), int(x / 4 + lx)
                y0, y1 = int(y / 4 - ly), int(y / 4 + ly)
                z0, z1 = int(z - lz), int(z + lz)
                origin = (float(x0), float(y0), float(z0))
                vol_ = vol[x0:x1, y0:y1, z0:z1]
                total_time_fetch += time.time() - t_start


                x = int(np.round(x / 4) * 4)  # 四舍五入到最近的4的倍数
                y = int(np.round(y / 4) * 4)  # 四舍五入到最近的4的倍数
                z = int(np.round(z))  # 直接四舍五入


                t_start = time.time()
                if USE_COMPRESSED_BLOCK:
                    vol0 = vol_.where(segid1, 255, 0, out_dtype=np.uint8)
                else:
                    vol0 = np.where(vol_ == segid1, 255, 0).astype(np.uint8)
                total_time_where += time.time() - t_start

                t_start = time.time()
                if USE_COMPRESSED_BLOCK:
                    vol0 = vol0.keep_nearest_connected_component_optimized(lx,ly,lz)
                else:
                    vol0 = cc3d.connected_components(vol0[:, :, :, 0], out_dtype=np.uint64)
                    nn_idx0, _ = nearest_nonzero_idx(vol0, lx, ly, lz) 
                    relabel0 = vol0[tuple(nn_idx0)]
                    vol0 = np.where(vol0 == relabel0, 255, 0).astype(np.uint8)
                total_time_cc += time.time() - t_start
 
                t_start = time.time()
                if USE_COMPRESSED_BLOCK:
                    vol1 = vol_.where(segid2, 255, 0, out_dtype=np.uint8)
                else:
                    vol1 = np.where(vol_ == segid2, 255, 0).astype(np.uint8)
                total_time_where += time.time() - t_start

                t_start = time.time()
                if USE_COMPRESSED_BLOCK:
                    vol1 = vol1.keep_nearest_connected_component_optimized(lx,ly,lz)
                else:
                    vol1 = cc3d.connected_components(vol1[:, :, :, 0], out_dtype=np.uint64)
                    nn_idx1, _ = nearest_nonzero_idx(vol1, lx, ly, lz)  # valid_indices 已保证唯一
                    relabel1 = vol1[tuple(nn_idx1)]
                    vol1 = np.where(vol1 == relabel1, 255, 0).astype(np.uint8)
                total_time_cc += time.time() - t_start


                t_start = time.time()
                if USE_COMPRESSED_BLOCK:
                    if BOUNDARY_MODE == "legacy_2d":
                        vol0.extract_boundary_points(
                            x, y, z, lx, ly, lz, pc_data, ids_data, label=0, origin=origin
                        )
                    else:
                        vol0.extract_boundary_voxel_points_3d(origin, pc_data, ids_data, label=0)
                else:
                    if BOUNDARY_MODE == "legacy_2d":
                        extract_boundary_points_dense(vol0.astype(np.uint8), origin, pc_data, ids_data, label=0)
                    else:
                        extract_boundary_voxel_points_dense(
                            vol0.astype(np.uint8), origin, pc_data, ids_data, label=0
                        )

                total_time_boundary += time.time() - t_start
                    
                t_start = time.time()
                if USE_COMPRESSED_BLOCK:
                    if BOUNDARY_MODE == "legacy_2d":
                        vol1.extract_boundary_points(
                            x, y, z, lx, ly, lz, pc_data, ids_data, label=1, origin=origin
                        )
                    else:
                        vol1.extract_boundary_voxel_points_3d(origin, pc_data, ids_data, label=1)

                else:
                    if BOUNDARY_MODE == "legacy_2d":
                        extract_boundary_points_dense(vol1.astype(np.uint8), origin, pc_data, ids_data, label=1)
                    else:
                        extract_boundary_voxel_points_dense(
                            vol1.astype(np.uint8), origin, pc_data, ids_data, label=1
                        )

                total_time_boundary += time.time() - t_start

            # 汇总本次 work 的所有点云结果
            if pc_data:
                all_pc.append(np.vstack(pc_data))
                all_ids.append(np.array(ids_data))
                processed_count += 1

        print(f"[耗时分析] Fetch: {total_time_fetch:.4f}s, Where: {total_time_where:.4f}s, CC: {total_time_cc:.4f}s, Boundary: {total_time_boundary:.4f}s")

    result_pc  = np.vstack(all_pc)  if all_pc  else np.zeros((0, 3))
    result_ids = np.concatenate(all_ids) if all_ids else np.array([])
    return result_pc, result_ids, processed_count


def compare_results(pc1, ids1, pc2, ids2):
    """对比两种模式产生的点云结果是否一致。"""
    print(f"\n[对比] compressed 模式点数: {len(pc1)},  原始模式点数: {len(pc2)}")
    if len(pc1) != len(pc2):
        print("[FAIL] 点云数量不一致！")
        return False

    # 按行排序后逐元素比较（两种模式遍历顺序可能不同）
    pc1_sorted  = pc1 [np.lexsort(pc1 [:, ::-1].T)]
    pc2_sorted  = pc2 [np.lexsort(pc2 [:, ::-1].T)]
    ids1_sorted = ids1[np.lexsort(pc1 [:, ::-1].T)]
    ids2_sorted = ids2[np.lexsort(pc2 [:, ::-1].T)]

    if np.allclose(pc1_sorted, pc2_sorted, atol=1e-3) and np.array_equal(ids1_sorted, ids2_sorted):
        print("[PASS] 两种模式结果完全一致 ✓")
        return True
    else:
        diff_mask = ~np.all(np.isclose(pc1_sorted, pc2_sorted, atol=1e-3), axis=1)
        print(f"[FAIL] 有 {diff_mask.sum()} 个点坐标不一致")
        print("前5个不一致行（compressed vs 原始）:")
        for idx in np.where(diff_mask)[0][:5]:
            print(f"  {pc1_sorted[idx]}  vs  {pc2_sorted[idx]}")
        return False


if __name__ == '__main__':
    dataset = os.environ.get("SABER_TEST_DATASET")
    if not dataset:
        raise SystemExit("Set SABER_TEST_DATASET to a Precomputed volume before running test.py")

    candidate_file = "./data/candidate0.csv"
    df = pd.read_csv(candidate_file)

    # ------------------------------------------------------------------ #
    #  预扫描：用原始模式找出所有唯一解的 case 下标                         #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("[预扫描] 初始化原始 CloudVolume 并扫描有效下标...")
    vol_scan = CloudVolume(
        dataset,
        mip=0,
        fill_missing=True,
        cache=True,
        lru_bytes=1024 * 1024 * 100,
    )
    # t_scan0 = time.time()
    # valid_indices, skipped_indices = scan_valid_indices(vol_scan, df)
    valid_indices=[8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 97, 98, 99, 100, 106, 107, 108, 109, 110, 111, 113, 114]
    print(f"valid_indices={valid_indices}")
    print(f"boundary_mode={BOUNDARY_MODE}")
    # t_scan1 = time.time()
    # print(f"[预扫描] 耗时 {t_scan1 - t_scan0:.2f}s，共 {len(valid_indices)} 个有效 case，{len(skipped_indices)} 个跳过 case")

    # ------------------------------------------------------------------ #
    #  模式 1：使用压缩块 (USE_COMPRESSED_BLOCK = True)                    #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("[模式 1] 启动 saber=True")
    USE_COMPRESSED_BLOCK = True
    vol_compressed = CloudVolume(
        dataset,
        mip=0,
        fill_missing=True,
        cache=True,
        saber=True,
        cache_thread=0,
        lru_bytes=1024 * 1024 * 100,
    )
    t0 = time.time()
    pc_compressed, ids_compressed, proc1 = work(vol_compressed, valid_indices)
    t1 = time.time()
    print(f"[模式 1] 完成，耗时 {t1 - t0:.2f}s，共 {len(pc_compressed)} 个点，处理 case: {proc1}")

    # ------------------------------------------------------------------ #
    #  模式 2：不使用压缩块 (USE_COMPRESSED_BLOCK = False)                #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("[模式 2] 启动 dense CloudVolume")
    USE_COMPRESSED_BLOCK = False
    vol_normal = CloudVolume(
        dataset,
        mip=0,
        fill_missing=True,
        cache=True,
        lru_bytes=1024 * 1024 * 100,
    )
    t2 = time.time()
    pc_normal, ids_normal, proc2 = work(vol_normal, valid_indices)
    t3 = time.time()
    print(f"[模式 2] 完成，耗时 {t3 - t2:.2f}s，共 {len(pc_normal)} 个点，处理 case: {proc2}")

    # ------------------------------------------------------------------ #
    #  结果对比                                                            #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    compare_results(pc_compressed, ids_compressed, pc_normal, ids_normal)

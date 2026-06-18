import os
import sys
import time
import numpy as np
import cc3d
import traceback
import cv2
cv2.setNumThreads(0)

import random
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- Path configuration ---
LOCAL_CLONE = './cloud-volume'
if os.path.exists(LOCAL_CLONE):
    sys.path.insert(0, LOCAL_CLONE)

from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox
from compressedvoxel import CompressedVoxelContainer


USE_COMPRESSED_BLOCK = True

def nearest_nonzero_idx(a, x, y, z):
    idx = np.argwhere(a)

    # If (x,y) itself is also non-zero, we want to avoid those, so delete that
    # But, if we are sure that (x,y) won't be non-zero, skip the next step
    idx = idx[~(idx == [x, y, z]).all(1)]

    dists = ((idx - [x, y, z]) ** 2).sum(1)
    min_dist = dists.min()
    is_unique = int((dists == min_dist).sum()) == 1  # Whether the nearest point is unique.
    return idx[dists.argmin()], is_unique

def scan_valid_indices(vol, df, lx=80, ly=80, lz=32):
    """
    Scan each row of df in the baseline mode and find valid case indices
    for which nearest_nonzero_idx has a unique answer.
    Returns: (valid_indices, skipped_indices)
    """
    valid, skipped = [], []
    N = len(df)
    for ii in tqdm(range(N), desc="Scanning valid indices"):
        segid1 = df.iloc[ii, 0]
        segid2 = df.iloc[ii, 1]
        cord = np.array(df.iloc[ii, 2].strip('[]').split(','))
        x, y, z = float(cord[0]), float(cord[1]), float(cord[2])
        x = int(np.round(x / 4) * 4)
        y = int(np.round(y / 4) * 4)
        z = int(np.round(z))

        vol_ = vol[int(x/4 - lx):int(x/4 + lx),
                   int(y/4 - ly):int(y/4 + ly),
                   int(z - lz):int(z + lz)]

        # Uniqueness check for vol0.
        vol0 = np.where(vol_ == segid1, 255, 0).astype(np.uint8)
        cc0  = cc3d.connected_components(vol0[:, :, :, 0], out_dtype=np.uint64)
        _, u0 = nearest_nonzero_idx(cc0, lx, ly, lz)

        # Uniqueness check for vol1.
        vol1 = np.where(vol_ == segid2, 255, 0).astype(np.uint8)
        cc1  = cc3d.connected_components(vol1[:, :, :, 0], out_dtype=np.uint64)
        _, u1 = nearest_nonzero_idx(cc1, lx, ly, lz)

        if u0 and u1:
            valid.append(ii)
        else:
            skipped.append(ii)

    print(f"[Scan complete] Valid cases: {len(valid)}, skipped cases: {len(skipped)}")
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
        processed_count = 0  # Number of normally processed cases.
        
        total_time_fetch = 0
        total_time_where = 0
        total_time_cc = 0
        total_time_boundary = 0
        
        from collections import defaultdict
        first = True
        for ii in tqdm(valid_indices, desc="Processing", disable=True):
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
            x = int(np.round(x / 4) * 4)  # Round to the nearest multiple of 4.
            y = int(np.round(y / 4) * 4)  # Round to the nearest multiple of 4.
            z = int(np.round(z))  # Round directly.
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
                vol_ = vol[int(x / 4 - lx):int(x / 4 + lx), int(y / 4 - ly):int(y / 4 + ly), int(z - lz):int(z + lz)]
                total_time_fetch += time.time() - t_start


                x = int(np.round(x / 4) * 4)  # Round to the nearest multiple of 4.
                y = int(np.round(y / 4) * 4)  # Round to the nearest multiple of 4.
                z = int(np.round(z))  # Round directly.


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
                    nn_idx1, _ = nearest_nonzero_idx(vol1, lx, ly, lz)  # valid_indices already guarantees uniqueness.
                    relabel1 = vol1[tuple(nn_idx1)]
                    vol1 = np.where(vol1 == relabel1, 255, 0).astype(np.uint8)
                total_time_cc += time.time() - t_start


                temp1 = 0
                
                t_start = time.time()
                if USE_COMPRESSED_BLOCK:
                    vol0.extract_boundary_points(x, y, z, lx, ly, lz, pc_data, ids_data, label=0)
                else:
                    vol0=vol0.astype(np.uint8)
                    vol1=vol1.astype(np.uint8)
                    for i in range(2 * lz):
                        data_tmp0 = vol0[:, :, i]
                        flag=False
                        if np.any(data_tmp0 == 255):
                            flag=True
                        if flag:
                            contours0, _ = cv2.findContours(data_tmp0, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                            boundary0 = np.vstack([cnt.squeeze() for cnt in contours0 if cnt.size > 0])

                            if boundary0.ndim == 1:
                                continue
                            num_points = len(boundary0)
                            global_x = boundary0[:, 1] * 16 + x * 4 - lx * 16
                            global_y = boundary0[:, 0] * 16 + y * 4 - ly * 16
                            global_z = np.full(num_points, (i + z - lz) * 40)
                            new_points = np.column_stack((global_x, global_y, global_z))
                            pc_data.append(new_points)
                            ids_data.extend([0] * num_points)
                            temp1 += num_points

                total_time_boundary += time.time() - t_start
                    
                t_start = time.time()
                if USE_COMPRESSED_BLOCK:
                    vol1.extract_boundary_points(x, y, z, lx, ly, lz, pc_data, ids_data, label=1)

                else:
                    for j in range(2 * lz):
                        data_tmp1 = vol1[:, :, j]
                        flag=False
                        if np.any(data_tmp1 == 255):
                            flag=True
                        if flag:
                            contours1, _ = cv2.findContours(data_tmp1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                            boundary1 = np.vstack([cnt.squeeze() for cnt in contours1 if cnt.size > 0])
                            if boundary1.ndim == 1:
                                continue
                            num_points = len(boundary1)
                            global_x = boundary1[:, 1] * 16 + x * 4 - lx * 16
                            global_y = boundary1[:, 0] * 16 + y * 4 - ly * 16
                            global_z = np.full(num_points, (j + z - lz) * 40)
                            new_points = np.column_stack((global_x, global_y, global_z))
                            pc_data.append(new_points)
                            ids_data.extend([1] * num_points)
                            temp1 += num_points 

                total_time_boundary += time.time() - t_start

            # Aggregate all point-cloud results from this work pass.
            if pc_data:
                all_pc.append(np.vstack(pc_data))
                all_ids.append(np.array(ids_data))
                processed_count += 1

        print(f"[Timing] Fetch: {total_time_fetch:.4f}s, Where: {total_time_where:.4f}s, CC: {total_time_cc:.4f}s, Boundary: {total_time_boundary:.4f}s")

    result_pc  = np.vstack(all_pc)  if all_pc  else np.zeros((0, 3))
    result_ids = np.concatenate(all_ids) if all_ids else np.array([])
    return result_pc, result_ids, processed_count


def compare_results(pc1, ids1, pc2, ids2):
    """Compare whether the point-cloud results produced by the two modes are identical."""
    print(f"\n[Compare] compressed mode points: {len(pc1)}, baseline mode points: {len(pc2)}")
    if len(pc1) != len(pc2):
        print("[FAIL] Point counts do not match!")
        return False

    # Sort row-wise and compare element by element, since the traversal order may differ between the two modes.
    pc1_sorted  = pc1 [np.lexsort(pc1 [:, ::-1].T)]
    pc2_sorted  = pc2 [np.lexsort(pc2 [:, ::-1].T)]
    ids1_sorted = ids1[np.lexsort(pc1 [:, ::-1].T)]
    ids2_sorted = ids2[np.lexsort(pc2 [:, ::-1].T)]

    if np.allclose(pc1_sorted, pc2_sorted, atol=1e-3) and np.array_equal(ids1_sorted, ids2_sorted):
        print("[PASS] The two modes produce exactly the same results.")
        return True
    else:
        diff_mask = ~np.all(np.isclose(pc1_sorted, pc2_sorted, atol=1e-3), axis=1)
        print(f"[FAIL] {diff_mask.sum()} point coordinates do not match")
        print("First five mismatching rows (compressed vs baseline):")
        for idx in np.where(diff_mask)[0][:5]:
            print(f"  {pc1_sorted[idx]}  vs  {pc2_sorted[idx]}")
        return False


if __name__ == '__main__':
    candidate_file = "./data/candidate0.csv"
    df = pd.read_csv(candidate_file)

    # ------------------------------------------------------------------ #
    #  Prescan: use the baseline mode to find case indices with unique answers.   #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("[Prescan] Initialize the baseline CloudVolume and scan valid indices...")
    vol_scan = CloudVolume(
        '/CX/neuro_tracking/fafb-ffn1',
        mip=0,
        fill_missing=True,
        cache=True,
        log_path="/dev/shm/scan.log",
        lru_bytes=1024 * 1024 * 100,
    )
    # t_scan0 = time.time()
    # valid_indices, skipped_indices = scan_valid_indices(vol_scan, df)
    valid_indices=[8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 97, 98, 99, 100, 106, 107, 108, 109, 110, 111, 113, 114]
    print(f"valid_indices={valid_indices}")
    # t_scan1 = time.time()
    # print(f"[Prescan] Elapsed {t_scan1 - t_scan0:.2f}s, {len(valid_indices)} valid cases, {len(skipped_indices)} skipped cases")

    # ------------------------------------------------------------------ #
    #  Mode 1: use compressed blocks (USE_COMPRESSED_BLOCK = True)        #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("[Mode 1] Start with use_compressed_block=True")
    USE_COMPRESSED_BLOCK = True
    vol_compressed = CloudVolume(
        '/CX/neuro_tracking/fafb-ffn1',
        mip=0,
        fill_missing=True,
        cache=True,
        use_compressed_block=True,
        log_path="/dev/shm/111.log",
        lru_bytes=1024 * 1024 * 100,
    )
    vol_compressed.cache_threads=0
    t0 = time.time()
    pc_compressed, ids_compressed, proc1 = work(vol_compressed, valid_indices)
    t1 = time.time()
    print(f"[Mode 1] Finished in {t1 - t0:.2f}s, {len(pc_compressed)} total points, processed cases: {proc1}")

    # ------------------------------------------------------------------ #
    #  Mode 2: do not use compressed blocks (USE_COMPRESSED_BLOCK = False) #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("[Mode 2] Start with use_compressed_block=False")
    USE_COMPRESSED_BLOCK = False
    vol_normal = CloudVolume(
        '/CX/neuro_tracking/fafb-ffn1',
        mip=0,
        fill_missing=True,
        cache=True,
        log_path="/dev/shm/222.log",
        lru_bytes=1024 * 1024 * 100,
    )
    t2 = time.time()
    pc_normal, ids_normal, proc2 = work(vol_normal, valid_indices)
    t3 = time.time()
    print(f"[Mode 2] Finished in {t3 - t2:.2f}s, {len(pc_normal)} total points, processed cases: {proc2}")

    # ------------------------------------------------------------------ #
    #  Result comparison                                                   #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    compare_results(pc_compressed, ids_compressed, pc_normal, ids_normal)

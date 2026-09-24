#ifndef FAST_BOUNDARY_HPP
#define FAST_BOUNDARY_HPP

#include <stdint.h>
#include <cstdlib>
#include <vector>
#include "block_query_primitives.hpp"

template <typename T>
inline bool BoundaryGetVoxel(
    const CBlock* blocks,
    std::vector<uint8_t*>& cached_masks,
    int nx, int ny, int nz, int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    int qx, int qy, int qz)
{
    if (qx < 0 || qx >= req_x || qy < 0 || qy >= req_y || qz < 0 || qz >= req_z) {
        return false;
    }

    int full_x = qx + q2p_x;
    int full_y = qy + q2p_y;
    int full_z = qz + q2p_z;
    int gx = full_x / bx;
    int gy = full_y / by;
    int gz = full_z / bz;

    if (gx < 0 || gx >= nx || gy < 0 || gy >= ny || gz < 0 || gz >= nz) {
        return false;
    }

    size_t b_idx = (size_t)gx + (size_t)gy * nx + (size_t)gz * nx * ny;
    if (!blocks[b_idx].has_nonzero) {
        return false;
    }

    int lx = full_x % bx;
    int ly = full_y % by;
    int lz = full_z % bz;
    int v_idx = lx + ly * bx + lz * bx * by;

    if (cached_masks[b_idx] == nullptr) {
        int voxels_per_block = bx * by * bz;
        cached_masks[b_idx] = (uint8_t*)calloc(voxels_per_block, 1);
        DecodeBlockToCache<T>(blocks[b_idx], voxels_per_block, cached_masks[b_idx]);
    }

    return cached_masks[b_idx][v_idx] != 0;
}

template <typename T>
void ExtractBoundaryVoxels3D(
    const CBlock* blocks,
    int nx, int ny, int nz, int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    float origin_x, float origin_y, float origin_z,
    int label,
    std::vector<float>& out_coords,
    std::vector<int>& out_ids)
{
    size_t total_blocks = (size_t)nx * ny * nz;
    int voxels_per_block = bx * by * bz;
    std::vector<uint8_t*> cached_masks(total_blocks, nullptr);

    for (int gz = 0; gz < nz; ++gz) {
        for (int gy = 0; gy < ny; ++gy) {
            for (int gx = 0; gx < nx; ++gx) {
                size_t b_idx = (size_t)gx + (size_t)gy * nx + (size_t)gz * nx * ny;
                if (!blocks[b_idx].has_nonzero) {
                    continue;
                }

                if (cached_masks[b_idx] == nullptr) {
                    cached_masks[b_idx] = (uint8_t*)calloc(voxels_per_block, 1);
                    DecodeBlockToCache<T>(blocks[b_idx], voxels_per_block, cached_masks[b_idx]);
                }

                for (int vl = 0; vl < voxels_per_block; ++vl) {
                    if (cached_masks[b_idx][vl] == 0) {
                        continue;
                    }

                    int local_x = vl % bx;
                    int local_y = (vl / bx) % by;
                    int local_z = vl / (bx * by);
                    int qx = gx * bx + local_x - q2p_x;
                    int qy = gy * by + local_y - q2p_y;
                    int qz = gz * bz + local_z - q2p_z;

                    if (qx < 0 || qx >= req_x || qy < 0 || qy >= req_y || qz < 0 || qz >= req_z) {
                        continue;
                    }

                    bool is_boundary =
                        !BoundaryGetVoxel<T>(blocks, cached_masks, nx, ny, nz, bx, by, bz,
                                             req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
                                             qx - 1, qy, qz) ||
                        !BoundaryGetVoxel<T>(blocks, cached_masks, nx, ny, nz, bx, by, bz,
                                             req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
                                             qx + 1, qy, qz) ||
                        !BoundaryGetVoxel<T>(blocks, cached_masks, nx, ny, nz, bx, by, bz,
                                             req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
                                             qx, qy - 1, qz) ||
                        !BoundaryGetVoxel<T>(blocks, cached_masks, nx, ny, nz, bx, by, bz,
                                             req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
                                             qx, qy + 1, qz) ||
                        !BoundaryGetVoxel<T>(blocks, cached_masks, nx, ny, nz, bx, by, bz,
                                             req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
                                             qx, qy, qz - 1) ||
                        !BoundaryGetVoxel<T>(blocks, cached_masks, nx, ny, nz, bx, by, bz,
                                             req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
                                             qx, qy, qz + 1);

                    if (!is_boundary) {
                        continue;
                    }

                    out_coords.push_back(origin_x + (float)qx);
                    out_coords.push_back(origin_y + (float)qy);
                    out_coords.push_back(origin_z + (float)qz);
                    out_ids.push_back(label);
                }
            }
        }
    }

    for (size_t i = 0; i < total_blocks; ++i) {
        if (cached_masks[i] != nullptr) {
            free(cached_masks[i]);
        }
    }
}

template <typename T>
void ExtractNonzeroVoxels3D(
    const CBlock* blocks,
    int nx, int ny, int nz, int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    std::vector<int64_t>& out_coords)
{
    int voxels_per_block = bx * by * bz;

    for (int gz = 0; gz < nz; ++gz) {
        for (int gy = 0; gy < ny; ++gy) {
            for (int gx = 0; gx < nx; ++gx) {
                size_t b_idx = (size_t)gx + (size_t)gy * nx + (size_t)gz * nx * ny;
                if (!blocks[b_idx].has_nonzero) {
                    continue;
                }

                uint8_t* mask = (uint8_t*)calloc(voxels_per_block, 1);
                DecodeBlockToCache<T>(blocks[b_idx], voxels_per_block, mask);

                for (int vl = 0; vl < voxels_per_block; ++vl) {
                    if (mask[vl] == 0) {
                        continue;
                    }

                    int local_x = vl % bx;
                    int local_y = (vl / bx) % by;
                    int local_z = vl / (bx * by);
                    int qx = gx * bx + local_x - q2p_x;
                    int qy = gy * by + local_y - q2p_y;
                    int qz = gz * bz + local_z - q2p_z;

                    if (qx < 0 || qx >= req_x || qy < 0 || qy >= req_y || qz < 0 || qz >= req_z) {
                        continue;
                    }

                    out_coords.push_back((int64_t)qx);
                    out_coords.push_back((int64_t)qy);
                    out_coords.push_back((int64_t)qz);
                }

                free(mask);
            }
        }
    }
}

template <typename T>
inline T ReadBlockValue(const CBlock& block, int voxel_index)
{
    return ReadCompressedValue<T>(block, voxel_index);
}

template <typename T>
inline bool ReadQueryValue(
    const CBlock* blocks,
    int nx, int ny, int nz, int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    int qx, int qy, int qz,
    T& value)
{
    return ReadQueryValuePrimitive<T>(
        blocks, nx, ny, nz, bx, by, bz,
        req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
        qx, qy, qz, value);
}

template <typename T>
inline bool ValueIsRequested(T value, const T* requested_values, size_t requested_value_count)
{
    for (size_t index = 0; index < requested_value_count; ++index) {
        if (value == requested_values[index]) {
            return true;
        }
    }
    return false;
}

template <typename T>
void ExtractMatchingVoxels3D(
    const CBlock* blocks,
    int nx, int ny, int nz, int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    const T* requested_labels, size_t requested_label_count,
    std::vector<int64_t>& out_coords,
    std::vector<uint64_t>& out_values)
{
    int voxels_per_block = bx * by * bz;

    for (int gz = 0; gz < nz; ++gz) {
        for (int gy = 0; gy < ny; ++gy) {
            for (int gx = 0; gx < nx; ++gx) {
                size_t block_index = (size_t)gx + (size_t)gy * nx + (size_t)gz * nx * ny;
                if (!blocks[block_index].has_nonzero) {
                    continue;
                }

                for (int voxel_index = 0; voxel_index < voxels_per_block; ++voxel_index) {
                    T value = ReadBlockValue<T>(blocks[block_index], voxel_index);
                    bool matches = false;
                    for (size_t label_index = 0; label_index < requested_label_count; ++label_index) {
                        if (value == requested_labels[label_index]) {
                            matches = true;
                            break;
                        }
                    }
                    if (!matches) {
                        continue;
                    }

                    int local_x = voxel_index % bx;
                    int local_y = (voxel_index / bx) % by;
                    int local_z = voxel_index / (bx * by);
                    int qx = gx * bx + local_x - q2p_x;
                    int qy = gy * by + local_y - q2p_y;
                    int qz = gz * bz + local_z - q2p_z;
                    if (qx < 0 || qx >= req_x || qy < 0 || qy >= req_y || qz < 0 || qz >= req_z) {
                        continue;
                    }

                    out_coords.push_back((int64_t)qx);
                    out_coords.push_back((int64_t)qy);
                    out_coords.push_back((int64_t)qz);
                    out_values.push_back((uint64_t)value);
                }
            }
        }
    }
}

template <typename T>
void ExtractLabelVoxelsAndContacts6(
    const CBlock* blocks,
    int nx, int ny, int nz, int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    T primary_label,
    const T* contact_labels, size_t contact_label_count,
    std::vector<int64_t>& out_coords,
    std::vector<uint64_t>& out_contact_values)
{
    static const int neighbor_offsets[6][3] = {
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
        {0, 1, 0}, {0, 0, -1}, {0, 0, 1}
    };
    int voxels_per_block = bx * by * bz;
    for (int gz = 0; gz < nz; ++gz) {
        for (int gy = 0; gy < ny; ++gy) {
            for (int gx = 0; gx < nx; ++gx) {
                size_t block_index = (size_t)gx + (size_t)gy * nx + (size_t)gz * nx * ny;
                if (!blocks[block_index].has_nonzero) {
                    continue;
                }
                for (int voxel_index = 0; voxel_index < voxels_per_block; ++voxel_index) {
                    if (ReadBlockValue<T>(blocks[block_index], voxel_index) != primary_label) {
                        continue;
                    }
                    int local_x = voxel_index % bx;
                    int local_y = (voxel_index / bx) % by;
                    int local_z = voxel_index / (bx * by);
                    int qx = gx * bx + local_x - q2p_x;
                    int qy = gy * by + local_y - q2p_y;
                    int qz = gz * bz + local_z - q2p_z;
                    if (qx < 0 || qx >= req_x || qy < 0 || qy >= req_y || qz < 0 || qz >= req_z) {
                        continue;
                    }
                    out_coords.push_back((int64_t)qx);
                    out_coords.push_back((int64_t)qy);
                    out_coords.push_back((int64_t)qz);
                    for (int neighbor_index = 0; neighbor_index < 6; ++neighbor_index) {
                        T neighbor_value = 0;
                        if (!ReadQueryValue<T>(
                                blocks, nx, ny, nz, bx, by, bz,
                                req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
                                qx + neighbor_offsets[neighbor_index][0],
                                qy + neighbor_offsets[neighbor_index][1],
                                qz + neighbor_offsets[neighbor_index][2],
                                neighbor_value)) {
                            continue;
                        }
                        if (neighbor_value != 0 && neighbor_value != primary_label &&
                            ValueIsRequested<T>(neighbor_value, contact_labels, contact_label_count)) {
                            out_contact_values.push_back((uint64_t)neighbor_value);
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void ExtractLabelVoxelsAndBoundary3D(
    const CBlock* blocks,
    int nx, int ny, int nz, int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    T primary_label,
    std::vector<int64_t>& out_solid_coords,
    std::vector<int64_t>& out_boundary_coords)
{
    static const int neighbor_offsets[6][3] = {
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
        {0, 1, 0}, {0, 0, -1}, {0, 0, 1}
    };
    int voxels_per_block = bx * by * bz;
    for (int gz = 0; gz < nz; ++gz) {
        for (int gy = 0; gy < ny; ++gy) {
            for (int gx = 0; gx < nx; ++gx) {
                size_t block_index = (size_t)gx + (size_t)gy * nx + (size_t)gz * nx * ny;
                if (!blocks[block_index].has_nonzero) {
                    continue;
                }
                for (int voxel_index = 0; voxel_index < voxels_per_block; ++voxel_index) {
                    if (ReadBlockValue<T>(blocks[block_index], voxel_index) != primary_label) {
                        continue;
                    }
                    int local_x = voxel_index % bx;
                    int local_y = (voxel_index / bx) % by;
                    int local_z = voxel_index / (bx * by);
                    int qx = gx * bx + local_x - q2p_x;
                    int qy = gy * by + local_y - q2p_y;
                    int qz = gz * bz + local_z - q2p_z;
                    if (qx < 0 || qx >= req_x || qy < 0 || qy >= req_y || qz < 0 || qz >= req_z) {
                        continue;
                    }
                    out_solid_coords.push_back((int64_t)qx);
                    out_solid_coords.push_back((int64_t)qy);
                    out_solid_coords.push_back((int64_t)qz);

                    bool is_boundary = false;
                    for (int neighbor_index = 0; neighbor_index < 6; ++neighbor_index) {
                        T neighbor_value = 0;
                        bool inside = ReadQueryValue<T>(
                            blocks, nx, ny, nz, bx, by, bz,
                            req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
                            qx + neighbor_offsets[neighbor_index][0],
                            qy + neighbor_offsets[neighbor_index][1],
                            qz + neighbor_offsets[neighbor_index][2],
                            neighbor_value);
                        if (!inside || neighbor_value != primary_label) {
                            is_boundary = true;
                            break;
                        }
                    }
                    if (is_boundary) {
                        out_boundary_coords.push_back((int64_t)qx);
                        out_boundary_coords.push_back((int64_t)qy);
                        out_boundary_coords.push_back((int64_t)qz);
                    }
                }
            }
        }
    }
}

#endif

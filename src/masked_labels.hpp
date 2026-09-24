#ifndef MASKED_LABELS_HPP
#define MASKED_LABELS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "block_arena.hpp"
#include "block_query_primitives.hpp"

struct MaskedDistinctLabelStats {
    size_t palette_only_blocks = 0;
    size_t partially_decoded_blocks = 0;
    size_t skipped_blocks = 0;
    size_t selected_voxels = 0;
};

template <typename T>
inline void DistinctLabelsInMask(
    const BlockArena& arena,
    int nx,
    int ny,
    int nz,
    int bx,
    int by,
    int bz,
    int req_x,
    int req_y,
    int req_z,
    int q2p_x,
    int q2p_y,
    int q2p_z,
    const uint8_t* selection,
    std::vector<uint64_t>& output,
    MaskedDistinctLabelStats& stats)
{
    std::unordered_set<uint64_t> labels;
    const CBlock* blocks = arena.get_cblocks();

    for (int gz = 0; gz < nz; ++gz) {
        const int physical_z0 = gz * bz;
        const int physical_z1 = physical_z0 + bz;
        const int query_z0 = std::max(0, physical_z0 - q2p_z);
        const int query_z1 = std::min(req_z, physical_z1 - q2p_z);
        if (query_z0 >= query_z1) {
            continue;
        }
        for (int gy = 0; gy < ny; ++gy) {
            const int physical_y0 = gy * by;
            const int physical_y1 = physical_y0 + by;
            const int query_y0 = std::max(0, physical_y0 - q2p_y);
            const int query_y1 = std::min(req_y, physical_y1 - q2p_y);
            if (query_y0 >= query_y1) {
                continue;
            }
            for (int gx = 0; gx < nx; ++gx) {
                const QueryBlockIntersection intersection = IntersectQueryBlock(
                    gx, gy, gz, bx, by, bz,
                    req_x, req_y, req_z, q2p_x, q2p_y, q2p_z);
                if (!intersection.intersects) {
                    continue;
                }

                bool any_selected = false;
                bool all_selected = true;
                size_t block_selected_voxels = 0;
                for (int qx = intersection.x0; qx < intersection.x1; ++qx) {
                    for (int qy = intersection.y0; qy < intersection.y1; ++qy) {
                        const size_t row =
                            (static_cast<size_t>(qx) * req_y + qy) * req_z;
                        for (int qz = intersection.z0; qz < intersection.z1; ++qz) {
                            const bool selected = selection[row + qz] != 0;
                            any_selected |= selected;
                            all_selected &= selected;
                            block_selected_voxels += selected ? 1 : 0;
                        }
                    }
                }
                if (!any_selected) {
                    ++stats.skipped_blocks;
                    continue;
                }
                stats.selected_voxels += block_selected_voxels;

                const size_t block_index =
                    static_cast<size_t>(gx) + static_cast<size_t>(gy) * nx +
                    static_cast<size_t>(gz) * nx * ny;
                const StoredBlock& stored = arena.blocks[block_index];
                const CBlock& block = blocks[block_index];
                if (all_selected && intersection.fully_inside && !stored.is_null &&
                    block.palette != nullptr) {
                    const size_t palette_size =
                        stored.palette_bytes.size() / sizeof(T);
                    const T* palette = reinterpret_cast<const T*>(block.palette);
                    for (size_t index = 0; index < palette_size; ++index) {
                        labels.insert(static_cast<uint64_t>(palette[index]));
                    }
                    ++stats.palette_only_blocks;
                    continue;
                }

                ++stats.partially_decoded_blocks;
                if (stored.is_null || block.palette == nullptr) {
                    labels.insert(0);
                    continue;
                }
                for (int qx = intersection.x0; qx < intersection.x1; ++qx) {
                    const int local_x = qx + q2p_x - gx * bx;
                    for (int qy = intersection.y0; qy < intersection.y1; ++qy) {
                        const int local_y = qy + q2p_y - gy * by;
                        const size_t selection_row =
                            (static_cast<size_t>(qx) * req_y + qy) * req_z;
                        for (int qz = intersection.z0; qz < intersection.z1; ++qz) {
                            if (selection[selection_row + qz] == 0) {
                                continue;
                            }
                            const int local_z = qz + q2p_z - gz * bz;
                            const int voxel_index =
                                local_x + local_y * bx + local_z * bx * by;
                            labels.insert(static_cast<uint64_t>(
                                ReadCompressedValue<T>(block, voxel_index)));
                        }
                    }
                }
            }
        }
    }

    output.assign(labels.begin(), labels.end());
    std::sort(output.begin(), output.end());
}

#endif

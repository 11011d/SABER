#ifndef QUERY_OPERATORS_HPP
#define QUERY_OPERATORS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "block_arena.hpp"
#include "block_query_primitives.hpp"

struct LabelContourOperatorStats {
    size_t scanned_blocks = 0;
    size_t palette_skipped_blocks = 0;
    size_t predicate_decoded_blocks = 0;
    size_t traversed_voxels = 0;
    size_t emitted_points = 0;
};

template <typename T>
inline size_t ArenaPaletteSize(const BlockArena& arena, size_t block_index)
{
    return arena.blocks[block_index].palette_bytes.size() / sizeof(T);
}

template <typename T>
inline bool FindNearestLabelSeed(
    const BlockArena& arena,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    T label,
    int cx, int cy, int cz,
    int& seed_x, int& seed_y, int& seed_z,
    LabelContourOperatorStats& stats)
{
    const CBlock* blocks = arena.get_cblocks();
    long long minimum_distance = -1;
    bool found = false;
    const int voxels_per_block = bx * by * bz;

    for (int gz = 0; gz < nz; ++gz) {
        for (int gy = 0; gy < ny; ++gy) {
            for (int gx = 0; gx < nx; ++gx) {
                const size_t block_index = static_cast<size_t>(gx) +
                    static_cast<size_t>(gy) * nx + static_cast<size_t>(gz) * nx * ny;
                ++stats.scanned_blocks;
                const CBlock& block = blocks[block_index];
                const size_t palette_size = ArenaPaletteSize<T>(arena, block_index);
                if (!block.has_nonzero ||
                    !PaletteContains<T>(block, palette_size, label)) {
                    ++stats.palette_skipped_blocks;
                    continue;
                }
                for (int voxel_index = 0; voxel_index < voxels_per_block; ++voxel_index) {
                    if (ReadCompressedValue<T>(block, voxel_index) != label) {
                        continue;
                    }
                    const int local_x = voxel_index % bx;
                    const int local_y = (voxel_index / bx) % by;
                    const int local_z = voxel_index / (bx * by);
                    const int qx = gx * bx + local_x - q2p_x;
                    const int qy = gy * by + local_y - q2p_y;
                    const int qz = gz * bz + local_z - q2p_z;
                    if (qx < 0 || qx >= req_x || qy < 0 || qy >= req_y ||
                        qz < 0 || qz >= req_z) {
                        continue;
                    }
                    const long long dx = static_cast<long long>(qx) - cx;
                    const long long dy = static_cast<long long>(qy) - cy;
                    const long long dz = static_cast<long long>(qz) - cz;
                    const long long distance = dx * dx + dy * dy + dz * dz;
                    if (distance == 0) {
                        continue;
                    }
                    if (minimum_distance < 0 || distance < minimum_distance) {
                        minimum_distance = distance;
                        seed_x = qx;
                        seed_y = qy;
                        seed_z = qz;
                        found = true;
                    }
                }
            }
        }
    }
    return found;
}

template <typename T>
inline std::vector<uint8_t*> TraverseLabelComponent26(
    const BlockArena& arena,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    T label,
    int seed_x, int seed_y, int seed_z,
    LabelContourOperatorStats& stats)
{
    const size_t total_blocks = static_cast<size_t>(nx) * ny * nz;
    const int voxels_per_block = bx * by * bz;
    const CBlock* blocks = arena.get_cblocks();
    std::vector<uint8_t*> component_masks(total_blocks, nullptr);
    std::vector<uint8_t*> predicate_masks(total_blocks, nullptr);
    std::vector<Point3D> frontier;
    frontier.reserve(10000);

    size_t seed_block = 0;
    int seed_voxel = 0;
    if (!QueryCoordinateToBlockVoxel(
            seed_x, seed_y, seed_z, nx, ny, nz, bx, by, bz,
            req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
            seed_block, seed_voxel)) {
        return component_masks;
    }
    const size_t seed_palette_size = ArenaPaletteSize<T>(arena, seed_block);
    if (!PaletteContains<T>(blocks[seed_block], seed_palette_size, label)) {
        return component_masks;
    }
    predicate_masks[seed_block] = static_cast<uint8_t*>(
        std::calloc(static_cast<size_t>(voxels_per_block), 1));
    DecodeEqualityPredicate<T>(
        blocks[seed_block], seed_palette_size, label,
        voxels_per_block, predicate_masks[seed_block]);
    ++stats.predicate_decoded_blocks;
    if (predicate_masks[seed_block][seed_voxel] == 0) {
        std::free(predicate_masks[seed_block]);
        return component_masks;
    }

    component_masks[seed_block] = static_cast<uint8_t*>(
        std::calloc(static_cast<size_t>(voxels_per_block), 1));
    component_masks[seed_block][seed_voxel] = 1;
    frontier.push_back({seed_x, seed_y, seed_z});

    size_t head = 0;
    while (head < frontier.size()) {
        const Point3D point = frontier[head++];
        ++stats.traversed_voxels;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    const int qx = point.x + dx;
                    const int qy = point.y + dy;
                    const int qz = point.z + dz;
                    size_t block_index = 0;
                    int voxel_index = 0;
                    if (!QueryCoordinateToBlockVoxel(
                            qx, qy, qz, nx, ny, nz, bx, by, bz,
                            req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
                            block_index, voxel_index)) {
                        continue;
                    }
                    if (component_masks[block_index] != nullptr &&
                        component_masks[block_index][voxel_index] != 0) {
                        continue;
                    }
                    if (predicate_masks[block_index] == nullptr) {
                        const size_t palette_size = ArenaPaletteSize<T>(arena, block_index);
                        if (!PaletteContains<T>(blocks[block_index], palette_size, label)) {
                            continue;
                        }
                        predicate_masks[block_index] = static_cast<uint8_t*>(
                            std::calloc(static_cast<size_t>(voxels_per_block), 1));
                        DecodeEqualityPredicate<T>(
                            blocks[block_index], palette_size, label,
                            voxels_per_block, predicate_masks[block_index]);
                        ++stats.predicate_decoded_blocks;
                    }
                    if (predicate_masks[block_index][voxel_index] == 0) {
                        continue;
                    }
                    if (component_masks[block_index] == nullptr) {
                        component_masks[block_index] = static_cast<uint8_t*>(
                            std::calloc(static_cast<size_t>(voxels_per_block), 1));
                    }
                    component_masks[block_index][voxel_index] = 1;
                    frontier.push_back({qx, qy, qz});
                }
            }
        }
    }

    for (size_t index = 0; index < total_blocks; ++index) {
        std::free(predicate_masks[index]);
    }
    return component_masks;
}

inline bool SparseMaskContains(
    const std::vector<uint8_t*>& masks,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    int qx, int qy, int qz)
{
    size_t block_index = 0;
    int voxel_index = 0;
    return QueryCoordinateToBlockVoxel(
               qx, qy, qz, nx, ny, nz, bx, by, bz,
               req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
               block_index, voxel_index) &&
        masks[block_index] != nullptr && masks[block_index][voxel_index] != 0;
}

inline void EmitSparseMaskBoundary6(
    std::vector<uint8_t*>& masks,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    std::vector<int64_t>& output,
    LabelContourOperatorStats& stats)
{
    static const int neighbors[6][3] = {
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
        {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
    };
    const size_t total_blocks = static_cast<size_t>(nx) * ny * nz;
    const int voxels_per_block = bx * by * bz;
    for (int gz = 0; gz < nz; ++gz) {
        for (int gy = 0; gy < ny; ++gy) {
            for (int gx = 0; gx < nx; ++gx) {
                const size_t block_index = static_cast<size_t>(gx) +
                    static_cast<size_t>(gy) * nx + static_cast<size_t>(gz) * nx * ny;
                if (masks[block_index] == nullptr) {
                    continue;
                }
                for (int voxel_index = 0; voxel_index < voxels_per_block; ++voxel_index) {
                    if (masks[block_index][voxel_index] == 0) {
                        continue;
                    }
                    const int local_x = voxel_index % bx;
                    const int local_y = (voxel_index / bx) % by;
                    const int local_z = voxel_index / (bx * by);
                    const int qx = gx * bx + local_x - q2p_x;
                    const int qy = gy * by + local_y - q2p_y;
                    const int qz = gz * bz + local_z - q2p_z;
                    if (qx < 0 || qx >= req_x || qy < 0 || qy >= req_y ||
                        qz < 0 || qz >= req_z) {
                        continue;
                    }
                    bool boundary = false;
                    for (int neighbor = 0; neighbor < 6; ++neighbor) {
                        if (!SparseMaskContains(
                                masks, nx, ny, nz, bx, by, bz,
                                req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
                                qx + neighbors[neighbor][0],
                                qy + neighbors[neighbor][1],
                                qz + neighbors[neighbor][2])) {
                            boundary = true;
                            break;
                        }
                    }
                    if (boundary) {
                        output.push_back(qx);
                        output.push_back(qy);
                        output.push_back(qz);
                        ++stats.emitted_points;
                    }
                }
            }
        }
    }
    for (size_t index = 0; index < total_blocks; ++index) {
        std::free(masks[index]);
        masks[index] = nullptr;
    }
}

using QueryLabelIndex = uint16_t;
static constexpr QueryLabelIndex kNoRequestedLabel = UINT16_MAX;

template <typename T>
inline QueryLabelIndex* DecodeRequestedLabelIndexMap(
    const CBlock& block,
    size_t palette_size,
    const T* labels,
    size_t label_count,
    int voxel_count)
{
    QueryLabelIndex* label_map = static_cast<QueryLabelIndex*>(
        std::malloc(static_cast<size_t>(voxel_count) * sizeof(QueryLabelIndex)));
    if (label_map == nullptr) {
        return nullptr;
    }
    std::fill(label_map, label_map + voxel_count, kNoRequestedLabel);
    const T* palette = reinterpret_cast<const T*>(block.palette);
    if (block.bits == 0) {
        const int label_index = MatchRequestedLabel<T>(
            palette[0], labels, label_count);
        if (label_index >= 0) {
            std::fill(
                label_map, label_map + voxel_count,
                static_cast<QueryLabelIndex>(label_index));
        }
        return label_map;
    }
    const uint32_t mask = block.bits == 32
        ? UINT32_MAX
        : (uint32_t(1) << block.bits) - 1;
    size_t bit_position = 0;
    for (int voxel = 0; voxel < voxel_count; ++voxel) {
        const size_t word_index = bit_position / 32;
        const int shift = static_cast<int>(bit_position % 32);
        uint32_t palette_index = (block.bitstream[word_index] >> shift) & mask;
        if (shift + block.bits > 32) {
            palette_index |=
                (block.bitstream[word_index + 1] << (32 - shift)) & mask;
        }
        const T value = palette[palette_index];
        const int label_index = MatchRequestedLabel<T>(value, labels, label_count);
        label_map[voxel] = label_index < 0
            ? kNoRequestedLabel
            : static_cast<QueryLabelIndex>(label_index);
        bit_position += block.bits;
    }
    return label_map;
}

inline QueryLabelIndex ReadRequestedLabelIndex(
    const std::vector<QueryLabelIndex*>& label_maps,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    int qx, int qy, int qz)
{
    size_t block_index = 0;
    int voxel_index = 0;
    if (!QueryCoordinateToBlockVoxel(
            qx, qy, qz, nx, ny, nz, bx, by, bz,
            req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
            block_index, voxel_index) ||
        label_maps[block_index] == nullptr) {
        return kNoRequestedLabel;
    }
    return label_maps[block_index][voxel_index];
}

inline QueryLabelIndex ReadContourNeighborIndex(
    const QueryLabelIndex* current_map,
    int voxel_index,
    int local_x, int local_y, int local_z,
    int qx, int qy, int qz,
    int direction,
    const std::vector<QueryLabelIndex*>& label_maps,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z)
{
    // Most contour neighbors stay in the current block. Avoid the coordinate
    // transform and three integer divisions in that common case.
    if (direction == 0 && qx > 0 && local_x > 0) {
        return current_map[voxel_index - 1];
    }
    if (direction == 1 && qx + 1 < req_x && local_x + 1 < bx) {
        return current_map[voxel_index + 1];
    }
    if (direction == 2 && qy > 0 && local_y > 0) {
        return current_map[voxel_index - bx];
    }
    if (direction == 3 && qy + 1 < req_y && local_y + 1 < by) {
        return current_map[voxel_index + bx];
    }
    if (direction == 4 && qz > 0 && local_z > 0) {
        return current_map[voxel_index - bx * by];
    }
    if (direction == 5 && qz + 1 < req_z && local_z + 1 < bz) {
        return current_map[voxel_index + bx * by];
    }
    static const int offsets[6][3] = {
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
        {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
    };
    return ReadRequestedLabelIndex(
        label_maps, nx, ny, nz, bx, by, bz,
        req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
        qx + offsets[direction][0],
        qy + offsets[direction][1],
        qz + offsets[direction][2]);
}

inline bool ReadSparseMaskValue(
    const std::vector<uint8_t*>& masks,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    int qx, int qy, int qz)
{
    size_t block_index = 0;
    int voxel_index = 0;
    return QueryCoordinateToBlockVoxel(
               qx, qy, qz, nx, ny, nz, bx, by, bz,
               req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
               block_index, voxel_index) &&
        masks[block_index] != nullptr && masks[block_index][voxel_index] != 0;
}

template <typename T>
inline void ExtractSingleLabelBoundary6(
    const BlockArena& arena,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    T label,
    std::vector<int64_t>& output,
    LabelContourOperatorStats& stats)
{
    static const int neighbors[6][3] = {
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
        {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
    };
    const CBlock* blocks = arena.get_cblocks();
    const int voxels_per_block = bx * by * bz;
    const size_t total_blocks = static_cast<size_t>(nx) * ny * nz;
    std::vector<uint8_t*> masks(total_blocks, nullptr);

    // Materialize predicate masks for every query-intersecting block before
    // emitting boundaries. A one-pass implementation would inspect a
    // neighboring block before that block is visited in grid order and would
    // incorrectly classify a cross-block neighbor as background.
    for (int gz = 0; gz < nz; ++gz) {
        for (int gy = 0; gy < ny; ++gy) {
            for (int gx = 0; gx < nx; ++gx) {
                const size_t block_index = static_cast<size_t>(gx) +
                    static_cast<size_t>(gy) * nx + static_cast<size_t>(gz) * nx * ny;
                ++stats.scanned_blocks;
                const QueryBlockIntersection intersection = IntersectQueryBlock(
                    gx, gy, gz, bx, by, bz,
                    req_x, req_y, req_z, q2p_x, q2p_y, q2p_z);
                if (!intersection.intersects) {
                    ++stats.palette_skipped_blocks;
                    continue;
                }
                const CBlock& block = blocks[block_index];
                const size_t palette_size = ArenaPaletteSize<T>(arena, block_index);
                if (!block.has_nonzero ||
                    ClassifyEqualityPredicate<T>(block, palette_size, label) ==
                        BlockPredicateClass::Skip) {
                    ++stats.palette_skipped_blocks;
                    continue;
                }
                masks[block_index] = static_cast<uint8_t*>(
                    std::calloc(static_cast<size_t>(voxels_per_block), 1));
                if (masks[block_index] == nullptr) {
                    std::abort();
                }
                DecodeEqualityPredicate<T>(
                    block, palette_size, label, voxels_per_block,
                    masks[block_index]);
                ++stats.predicate_decoded_blocks;
            }
        }
    }

    for (int gz = 0; gz < nz; ++gz) {
        for (int gy = 0; gy < ny; ++gy) {
            for (int gx = 0; gx < nx; ++gx) {
                const size_t block_index = static_cast<size_t>(gx) +
                    static_cast<size_t>(gy) * nx + static_cast<size_t>(gz) * nx * ny;
                if (masks[block_index] == nullptr) {
                    continue;
                }
                const QueryBlockIntersection intersection = IntersectQueryBlock(
                    gx, gy, gz, bx, by, bz,
                    req_x, req_y, req_z, q2p_x, q2p_y, q2p_z);
                for (int voxel_index = 0; voxel_index < voxels_per_block; ++voxel_index) {
                    if (masks[block_index][voxel_index] == 0) {
                        continue;
                    }
                    const int local_x = voxel_index % bx;
                    const int local_y = (voxel_index / bx) % by;
                    const int local_z = voxel_index / (bx * by);
                    const int qx = gx * bx + local_x - q2p_x;
                    const int qy = gy * by + local_y - q2p_y;
                    const int qz = gz * bz + local_z - q2p_z;
                    if (qx < intersection.x0 || qx >= intersection.x1 ||
                        qy < intersection.y0 || qy >= intersection.y1 ||
                        qz < intersection.z0 || qz >= intersection.z1) {
                        continue;
                    }
                    bool boundary = false;
                    for (int neighbor = 0; neighbor < 6; ++neighbor) {
                        if (!ReadSparseMaskValue(
                                masks, nx, ny, nz, bx, by, bz,
                                req_x, req_y, req_z,
                                q2p_x, q2p_y, q2p_z,
                                qx + neighbors[neighbor][0],
                                qy + neighbors[neighbor][1],
                                qz + neighbors[neighbor][2])) {
                            boundary = true;
                            break;
                        }
                    }
                    if (boundary) {
                        output.push_back(qx);
                        output.push_back(qy);
                        output.push_back(qz);
                        ++stats.emitted_points;
                    }
                }
            }
        }
    }
    for (size_t index = 0; index < total_blocks; ++index) {
        std::free(masks[index]);
    }
}

template <typename T>
inline void ExtractUnfilteredLabelBoundaries6(
    const BlockArena& arena,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    const T* labels,
    size_t label_count,
    std::vector<std::vector<int64_t>>& outputs,
    LabelContourOperatorStats& stats)
{
    static const int neighbors[6][3] = {
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
        {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
    };
    const CBlock* blocks = arena.get_cblocks();
    const int voxels_per_block = bx * by * bz;
    const size_t total_blocks = static_cast<size_t>(nx) * ny * nz;
    // Keep one compact label-index map per relevant block. This is the
    // physical realization of SelectMany + EmitMany: each block is decoded
    // once, while unrelated blocks remain compressed and untouched.
    if (label_count >= static_cast<size_t>(kNoRequestedLabel)) {
        std::abort();
    }
    std::vector<QueryLabelIndex*> label_maps(total_blocks, nullptr);
    for (int gz = 0; gz < nz; ++gz) {
        for (int gy = 0; gy < ny; ++gy) {
            for (int gx = 0; gx < nx; ++gx) {
                const size_t block_index = static_cast<size_t>(gx) +
                    static_cast<size_t>(gy) * nx + static_cast<size_t>(gz) * nx * ny;
                ++stats.scanned_blocks;
                const QueryBlockIntersection intersection = IntersectQueryBlock(
                    gx, gy, gz, bx, by, bz,
                    req_x, req_y, req_z, q2p_x, q2p_y, q2p_z);
                if (!intersection.intersects) {
                    ++stats.palette_skipped_blocks;
                    continue;
                }
                const CBlock& block = blocks[block_index];
                const size_t palette_size = ArenaPaletteSize<T>(arena, block_index);
                bool palette_matches = false;
                for (size_t label_index = 0; label_index < label_count; ++label_index) {
                    if (PaletteContains<T>(block, palette_size, labels[label_index])) {
                        palette_matches = true;
                        break;
                    }
                }
                if (!block.has_nonzero || !palette_matches) {
                    ++stats.palette_skipped_blocks;
                    continue;
                }
                label_maps[block_index] = DecodeRequestedLabelIndexMap<T>(
                    block, palette_size, labels, label_count,
                    voxels_per_block);
                if (label_maps[block_index] == nullptr) {
                    std::abort();
                }
                ++stats.predicate_decoded_blocks;
                for (int voxel_index = 0; voxel_index < voxels_per_block; ++voxel_index) {
                    const QueryLabelIndex label_index = label_maps[block_index][voxel_index];
                    if (label_index == kNoRequestedLabel) {
                        continue;
                    }
                    const int local_x = voxel_index % bx;
                    const int local_y = (voxel_index / bx) % by;
                    const int local_z = voxel_index / (bx * by);
                    const int qx = gx * bx + local_x - q2p_x;
                    const int qy = gy * by + local_y - q2p_y;
                    const int qz = gz * bz + local_z - q2p_z;
                    if (qx < intersection.x0 || qx >= intersection.x1 ||
                        qy < intersection.y0 || qy >= intersection.y1 ||
                        qz < intersection.z0 || qz >= intersection.z1) {
                        continue;
                    }
                    bool boundary = false;
                    for (int neighbor = 0; neighbor < 6; ++neighbor) {
                        if (ReadContourNeighborIndex(
                                label_maps[block_index], voxel_index,
                                local_x, local_y, local_z, qx, qy, qz,
                                neighbor, label_maps,
                                nx, ny, nz, bx, by, bz,
                                req_x, req_y, req_z,
                                q2p_x, q2p_y, q2p_z) != label_index) {
                            boundary = true;
                            break;
                        }
                    }
                    if (boundary) {
                        outputs[label_index].push_back(qx);
                        outputs[label_index].push_back(qy);
                        outputs[label_index].push_back(qz);
                        ++stats.emitted_points;
                    }
                }
            }
        }
    }
    for (size_t index = 0; index < total_blocks; ++index) {
        std::free(label_maps[index]);
    }
}

template <typename T>
inline void ExtractLabelContours3D(
    const BlockArena& arena,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    const T* labels,
    size_t label_count,
    const int64_t* centers,
    bool filter_components,
    std::vector<std::vector<int64_t>>& outputs,
    LabelContourOperatorStats& stats)
{
    outputs.clear();
    outputs.resize(label_count);
    if (!filter_components) {
        // The planner keeps a shared-map implementation for generic
        // SelectMany/EmitMany. For the small fixed-cardinality P1/P2 plans,
        // the per-label fused kernel avoids materializing an intermediate
        // multi-label map and is generally faster on sparse blocks.
        for (size_t index = 0; index < label_count; ++index) {
            ExtractSingleLabelBoundary6<T>(
                arena, nx, ny, nz, bx, by, bz,
                req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
                labels[index], outputs[index], stats);
        }
        return;
    }
    for (size_t index = 0; index < label_count; ++index) {
        int seed_x = -1;
        int seed_y = -1;
        int seed_z = -1;
        if (!FindNearestLabelSeed<T>(
                arena, nx, ny, nz, bx, by, bz,
                req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
                labels[index],
                static_cast<int>(centers[index * 3]),
                static_cast<int>(centers[index * 3 + 1]),
                static_cast<int>(centers[index * 3 + 2]),
                seed_x, seed_y, seed_z, stats)) {
            continue;
        }
        std::vector<uint8_t*> masks = TraverseLabelComponent26<T>(
            arena, nx, ny, nz, bx, by, bz,
            req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
            labels[index], seed_x, seed_y, seed_z, stats);
        EmitSparseMaskBoundary6(
            masks, nx, ny, nz, bx, by, bz,
            req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
            outputs[index], stats);
    }
}

template <typename T>
inline void ProjectLabelChannels3D(
    const CBlock* blocks,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    const T* labels,
    size_t label_count,
    const uint8_t* channel_membership,
    size_t channel_count,
    int out_z, int out_y, int out_x,
    uint8_t* output)
{
    const size_t channel_voxels =
        static_cast<size_t>(out_z) * out_y * out_x;
    std::memset(output, 0, channel_count * channel_voxels);
    for (int oz = 0; oz < out_z; ++oz) {
        const int qz = static_cast<int>(
            static_cast<int64_t>(oz) * req_z / out_z);
        for (int oy = 0; oy < out_y; ++oy) {
            const int qy = static_cast<int>(
                static_cast<int64_t>(oy) * req_y / out_y);
            for (int ox = 0; ox < out_x; ++ox) {
                const int qx = static_cast<int>(
                    static_cast<int64_t>(ox) * req_x / out_x);
                size_t block_index = 0;
                int voxel_index = 0;
                if (!QueryCoordinateToBlockVoxel(
                        qx, qy, qz, nx, ny, nz, bx, by, bz,
                        req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
                        block_index, voxel_index)) {
                    continue;
                }
                const CBlock& block = blocks[block_index];
                if (!block.has_nonzero || block.palette == nullptr) {
                    continue;
                }
                const T value = ReadCompressedValue<T>(block, voxel_index);
                const size_t output_index =
                    (static_cast<size_t>(oz) * out_y + oy) * out_x + ox;
                for (size_t label_index = 0; label_index < label_count; ++label_index) {
                    if (value != labels[label_index]) {
                        continue;
                    }
                    for (size_t channel = 0; channel < channel_count; ++channel) {
                        if (channel_membership[channel * label_count + label_index] != 0) {
                            output[channel * channel_voxels + output_index] = 1;
                        }
                    }
                }
            }
        }
    }
}

#endif

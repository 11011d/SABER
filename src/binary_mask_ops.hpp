#ifndef BINARY_MASK_OPS_HPP
#define BINARY_MASK_OPS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "block_arena.hpp"
#include "block_query_primitives.hpp"
#include "../include/extract_blocks.h"

inline bool ValidateBinaryArena(const BlockArena& arena)
{
    if (arena.itemsize != 1) {
        return false;
    }
    for (int index = 0; index < arena.total; ++index) {
        const StoredBlock& block = arena.blocks[index];
        if (block.is_null) {
            continue;
        }
        if (block.bits > 1) {
            return false;
        }
        const size_t expected_palette_size = block.bits == 0 ? 1 : 2;
        if (block.palette_bytes.size() != expected_palette_size) {
            return false;
        }
        for (uint8_t value : block.palette_bytes) {
            if (value > 1) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
inline void BuildBinaryMaskStore(
    const BlockArena& source,
    BlockArena& destination,
    T label,
    int bx,
    int by,
    int bz)
{
    const CBlock* blocks = source.get_cblocks();
    const ptrdiff_t block_size[3] = {bx, by, bz};
    const int voxels_per_block = bx * by * bz;
    const int words_per_block = (voxels_per_block + 31) / 32;
    const uint8_t one_palette[1] = {1};
    const uint8_t mixed_palette[2] = {0, 1};
    std::vector<uint32_t> bitstream(words_per_block);

    for (int index = 0; index < source.total; ++index) {
        const CBlock& block = blocks[index];
        if (!block.has_nonzero || block.palette == nullptr) {
            continue;
        }
        const size_t palette_size =
            source.blocks[index].palette_bytes.size() / sizeof(T);
        if (palette_size == 0) {
            continue;
        }
        const BlockPredicateClass block_type = ClassifyEqualityPredicate<T>(
            block, palette_size, label);
        if (block_type == BlockPredicateClass::Skip) {
            continue;
        }
        if (block_type == BlockPredicateClass::UniformMatch) {
            destination.set_block(index, one_palette, 1, 0, nullptr, 0, true);
            continue;
        }

        compress_segmentation::CreateGenericBinaryBitstream<T>(
            block.bitstream,
            reinterpret_cast<const T*>(block.palette),
            block.bits,
            label,
            bitstream.data(),
            block_size);
        destination.set_block(
            index,
            mixed_palette,
            2,
            1,
            bitstream.data(),
            words_per_block,
            true);
    }
}

inline void CopyBinaryBlock(
    BlockArena& destination,
    int destination_index,
    const BlockArena& source,
    int source_index)
{
    const StoredBlock& block = source.blocks[source_index];
    destination.set_block(
        destination_index,
        block.palette_bytes.data(),
        static_cast<int>(block.palette_bytes.size()),
        block.bits,
        block.bitstream.empty() ? nullptr : block.bitstream.data(),
        static_cast<int>(block.bitstream.size()),
        block.has_nonzero);
}

inline bool BinaryUnionStores(
    const BlockArena& left,
    const BlockArena& right,
    BlockArena& destination,
    int voxels_per_block)
{
    if (left.total != right.total || left.total != destination.total ||
        !ValidateBinaryArena(left) || !ValidateBinaryArena(right)) {
        return false;
    }

    const CBlock* left_blocks = left.get_cblocks();
    const CBlock* right_blocks = right.get_cblocks();
    const uint8_t zero_palette[1] = {0};
    const uint8_t one_palette[1] = {1};
    const uint8_t mixed_palette[2] = {0, 1};
    std::vector<uint8_t> left_dense(voxels_per_block);
    std::vector<uint8_t> right_dense(voxels_per_block);
    std::vector<uint8_t> merged(voxels_per_block);
    std::vector<uint32_t> bitstream;

    for (int index = 0; index < left.total; ++index) {
        const bool left_nonzero = left_blocks[index].has_nonzero;
        const bool right_nonzero = right_blocks[index].has_nonzero;
        if (!left_nonzero && !right_nonzero) {
            continue;
        }
        if (left_nonzero && !right_nonzero) {
            CopyBinaryBlock(destination, index, left, index);
            continue;
        }
        if (!left_nonzero && right_nonzero) {
            CopyBinaryBlock(destination, index, right, index);
            continue;
        }

        DecodeBlockToCache<uint8_t>(
            left_blocks[index], voxels_per_block, left_dense.data());
        DecodeBlockToCache<uint8_t>(
            right_blocks[index], voxels_per_block, right_dense.data());
        for (int voxel = 0; voxel < voxels_per_block; ++voxel) {
            merged[voxel] = left_dense[voxel] | right_dense[voxel];
        }

        const int status = CompressMaskFast(
            merged.data(), voxels_per_block, bitstream);
        if (status == 0) {
            destination.set_block(index, zero_palette, 1, 0, nullptr, 0, false);
        } else if (status == 1) {
            destination.set_block(index, one_palette, 1, 0, nullptr, 0, true);
        } else {
            destination.set_block(
                index,
                mixed_palette,
                2,
                1,
                bitstream.data(),
                static_cast<int>(bitstream.size()),
                true);
        }
    }
    return true;
}

inline int CeilRatio(int value, int numerator, int denominator)
{
    return static_cast<int>(
        (static_cast<int64_t>(value) * numerator + denominator - 1) /
        denominator);
}

inline bool MaterializeBinaryChannels3D(
    const BlockArena& union_store,
    const BlockArena& label_one_store,
    const BlockArena& label_two_store,
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
    int out_z,
    int out_y,
    int out_x,
    uint8_t* output)
{
    const int expected_blocks = nx * ny * nz;
    if (union_store.total != expected_blocks ||
        label_one_store.total != expected_blocks ||
        label_two_store.total != expected_blocks ||
        !ValidateBinaryArena(union_store) ||
        !ValidateBinaryArena(label_one_store) ||
        !ValidateBinaryArena(label_two_store)) {
        return false;
    }

    const CBlock* union_blocks = union_store.get_cblocks();
    const CBlock* label_one_blocks = label_one_store.get_cblocks();
    const CBlock* label_two_blocks = label_two_store.get_cblocks();
    const size_t channel_voxels =
        static_cast<size_t>(out_z) * out_y * out_x;
    uint8_t* union_output = output;
    uint8_t* label_one_output = output + channel_voxels;
    uint8_t* label_two_output = output + 2 * channel_voxels;
    const int voxels_per_block = bx * by * bz;
    std::vector<uint8_t> union_dense(voxels_per_block);
    std::vector<uint8_t> label_one_dense(voxels_per_block);
    std::vector<uint8_t> label_two_dense(voxels_per_block);

    for (int gz = 0; gz < nz; ++gz) {
        const int query_z0 = std::max(0, gz * bz - q2p_z);
        const int query_z1 = std::min(req_z, (gz + 1) * bz - q2p_z);
        if (query_z0 >= query_z1) {
            continue;
        }
        const int output_z0 = CeilRatio(query_z0, out_z, req_z);
        const int output_z1 = CeilRatio(query_z1, out_z, req_z);
        for (int gy = 0; gy < ny; ++gy) {
            const int query_y0 = std::max(0, gy * by - q2p_y);
            const int query_y1 = std::min(req_y, (gy + 1) * by - q2p_y);
            if (query_y0 >= query_y1) {
                continue;
            }
            const int output_y0 = CeilRatio(query_y0, out_y, req_y);
            const int output_y1 = CeilRatio(query_y1, out_y, req_y);
            for (int gx = 0; gx < nx; ++gx) {
                const size_t block_index =
                    static_cast<size_t>(gx) + static_cast<size_t>(gy) * nx +
                    static_cast<size_t>(gz) * nx * ny;
                if (!union_blocks[block_index].has_nonzero) {
                    continue;
                }

                const int query_x0 = std::max(0, gx * bx - q2p_x);
                const int query_x1 = std::min(req_x, (gx + 1) * bx - q2p_x);
                if (query_x0 >= query_x1) {
                    continue;
                }
                const int output_x0 = CeilRatio(query_x0, out_x, req_x);
                const int output_x1 = CeilRatio(query_x1, out_x, req_x);

                DecodeBlockToCache<uint8_t>(
                    union_blocks[block_index], voxels_per_block, union_dense.data());
                if (label_one_blocks[block_index].has_nonzero) {
                    DecodeBlockToCache<uint8_t>(
                        label_one_blocks[block_index],
                        voxels_per_block,
                        label_one_dense.data());
                } else {
                    std::memset(label_one_dense.data(), 0, voxels_per_block);
                }
                if (label_two_blocks[block_index].has_nonzero) {
                    DecodeBlockToCache<uint8_t>(
                        label_two_blocks[block_index],
                        voxels_per_block,
                        label_two_dense.data());
                } else {
                    std::memset(label_two_dense.data(), 0, voxels_per_block);
                }

                for (int oz = output_z0; oz < output_z1; ++oz) {
                    const int source_z = (oz * req_z) / out_z;
                    const int local_z = source_z + q2p_z - gz * bz;
                    for (int oy = output_y0; oy < output_y1; ++oy) {
                        const int source_y = (oy * req_y) / out_y;
                        const int local_y = source_y + q2p_y - gy * by;
                        const size_t output_row =
                            (static_cast<size_t>(oz) * out_y + oy) * out_x;
                        for (int ox = output_x0; ox < output_x1; ++ox) {
                            const int source_x = (ox * req_x) / out_x;
                            const int local_x = source_x + q2p_x - gx * bx;
                            const int voxel_index =
                                local_x + local_y * bx + local_z * bx * by;
                            const size_t output_index = output_row + ox;
                            union_output[output_index] = union_dense[voxel_index];
                            label_one_output[output_index] =
                                label_one_dense[voxel_index];
                            label_two_output[output_index] =
                                label_two_dense[voxel_index];
                        }
                    }
                }
            }
        }
    }
    return true;
}

#endif

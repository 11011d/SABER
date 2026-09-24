#ifndef BLOCK_QUERY_PRIMITIVES_HPP
#define BLOCK_QUERY_PRIMITIVES_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "fast_cc.hpp"

enum class BlockPredicateClass : uint8_t {
    Skip = 0,
    UniformMatch = 1,
    Mixed = 2,
};

struct QueryBlockIntersection {
    int x0;
    int x1;
    int y0;
    int y1;
    int z0;
    int z1;
    bool intersects;
    bool fully_inside;
};

inline QueryBlockIntersection IntersectQueryBlock(
    int gx, int gy, int gz,
    int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z)
{
    const int physical_x0 = gx * bx;
    const int physical_y0 = gy * by;
    const int physical_z0 = gz * bz;
    QueryBlockIntersection result;
    result.x0 = std::max(0, physical_x0 - q2p_x);
    result.x1 = std::min(req_x, physical_x0 + bx - q2p_x);
    result.y0 = std::max(0, physical_y0 - q2p_y);
    result.y1 = std::min(req_y, physical_y0 + by - q2p_y);
    result.z0 = std::max(0, physical_z0 - q2p_z);
    result.z1 = std::min(req_z, physical_z0 + bz - q2p_z);
    result.intersects =
        result.x0 < result.x1 && result.y0 < result.y1 && result.z0 < result.z1;
    result.fully_inside = result.intersects &&
        physical_x0 >= q2p_x && physical_x0 + bx <= q2p_x + req_x &&
        physical_y0 >= q2p_y && physical_y0 + by <= q2p_y + req_y &&
        physical_z0 >= q2p_z && physical_z0 + bz <= q2p_z + req_z;
    return result;
}

inline bool QueryCoordinateToBlockVoxel(
    int qx, int qy, int qz,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    size_t& block_index,
    int& voxel_index)
{
    if (qx < 0 || qx >= req_x || qy < 0 || qy >= req_y ||
        qz < 0 || qz >= req_z) {
        return false;
    }
    const int full_x = qx + q2p_x;
    const int full_y = qy + q2p_y;
    const int full_z = qz + q2p_z;
    // C++ integer division truncates toward zero, so coordinates in
    // (-block_size, 0) would otherwise be misclassified as block zero and
    // produce a negative voxel offset. Out-of-volume query coordinates have
    // fill_missing semantics and must be skipped before block arithmetic.
    if (full_x < 0 || full_x >= nx * bx ||
        full_y < 0 || full_y >= ny * by ||
        full_z < 0 || full_z >= nz * bz) {
        return false;
    }
    const int gx = full_x / bx;
    const int gy = full_y / by;
    const int gz = full_z / bz;
    block_index = static_cast<size_t>(gx) + static_cast<size_t>(gy) * nx +
        static_cast<size_t>(gz) * nx * ny;
    const int local_x = full_x % bx;
    const int local_y = full_y % by;
    const int local_z = full_z % bz;
    voxel_index = local_x + local_y * bx + local_z * bx * by;
    return true;
}

template <typename T>
inline T ReadCompressedValue(const CBlock& block, int voxel_index)
{
    const T* palette = reinterpret_cast<const T*>(block.palette);
    if (block.bits == 0) {
        return palette[0];
    }
    const uint32_t mask = block.bits == 32
        ? UINT32_MAX
        : (uint32_t(1) << block.bits) - 1;
    const size_t bit_position = static_cast<size_t>(voxel_index) * block.bits;
    const size_t word_index = bit_position / 32;
    const int shift = static_cast<int>(bit_position % 32);
    uint32_t palette_index = (block.bitstream[word_index] >> shift) & mask;
    if (shift + block.bits > 32) {
        palette_index |=
            (block.bitstream[word_index + 1] << (32 - shift)) & mask;
    }
    return palette[palette_index];
}

template <typename T>
inline void DecodeCompressedValues(
    const CBlock& block,
    int voxel_count,
    T* output)
{
    const T* palette = reinterpret_cast<const T*>(block.palette);
    if (block.bits == 0) {
        std::fill(output, output + voxel_count, palette[0]);
        return;
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
        output[voxel] = palette[palette_index];
        bit_position += block.bits;
    }
}

template <typename T>
inline bool PaletteContains(
    const CBlock& block,
    size_t palette_size,
    T label)
{
    if (block.palette == nullptr || palette_size == 0) {
        return false;
    }
    const T* palette = reinterpret_cast<const T*>(block.palette);
    for (size_t index = 0; index < palette_size; ++index) {
        if (palette[index] == label) {
            return true;
        }
    }
    return false;
}

template <typename T>
inline BlockPredicateClass ClassifyEqualityPredicate(
    const CBlock& block,
    size_t palette_size,
    T label)
{
    if (!PaletteContains<T>(block, palette_size, label)) {
        return BlockPredicateClass::Skip;
    }
    return block.bits == 0
        ? BlockPredicateClass::UniformMatch
        : BlockPredicateClass::Mixed;
}

template <typename T>
inline void DecodeEqualityPredicate(
    const CBlock& block,
    size_t palette_size,
    T label,
    int voxel_count,
    uint8_t* output)
{
    const BlockPredicateClass classification =
        ClassifyEqualityPredicate<T>(block, palette_size, label);
    if (classification == BlockPredicateClass::Skip) {
        std::memset(output, 0, static_cast<size_t>(voxel_count));
        return;
    }
    if (classification == BlockPredicateClass::UniformMatch) {
        std::memset(output, 1, static_cast<size_t>(voxel_count));
        return;
    }
    const T* palette = reinterpret_cast<const T*>(block.palette);
    const uint32_t mask = block.bits == 32
        ? UINT32_MAX
        : (uint32_t(1) << block.bits) - 1;
    size_t bit_position = 0;
    for (int index = 0; index < voxel_count; ++index) {
        const size_t word_index = bit_position / 32;
        const int shift = static_cast<int>(bit_position % 32);
        uint32_t palette_index = (block.bitstream[word_index] >> shift) & mask;
        if (shift + block.bits > 32) {
            palette_index |=
                (block.bitstream[word_index + 1] << (32 - shift)) & mask;
        }
        output[index] = palette[palette_index] == label ? 1 : 0;
        bit_position += block.bits;
    }
}

template <typename T>
inline bool ReadQueryValuePrimitive(
    const CBlock* blocks,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    int qx, int qy, int qz,
    T& value)
{
    size_t block_index = 0;
    int voxel_index = 0;
    if (!QueryCoordinateToBlockVoxel(
            qx, qy, qz, nx, ny, nz, bx, by, bz,
            req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
            block_index, voxel_index)) {
        return false;
    }
    const CBlock& block = blocks[block_index];
    if (!block.has_nonzero || block.palette == nullptr) {
        value = 0;
        return true;
    }
    value = ReadCompressedValue<T>(block, voxel_index);
    return true;
}

template <typename T>
inline int MatchRequestedLabel(T value, const T* labels, size_t label_count)
{
    for (size_t index = 0; index < label_count; ++index) {
        if (value == labels[index]) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

#endif

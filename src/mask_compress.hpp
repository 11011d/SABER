#ifndef MASK_COMPRESS_HPP
#define MASK_COMPRESS_HPP

#include <vector>
#include <algorithm>
#include <stdint.h>
#include <stddef.h>

template <typename T>
struct MaskedBlockResult {
    std::vector<T> palette;
    uint8_t bits;
    std::vector<uint32_t> bitstream;
};

template <typename T>
MaskedBlockResult<T> MaskAndRecompressBlock(
    const T* src_palette, uint8_t src_bits, const uint32_t* src_bitstream,
    const ptrdiff_t blksize[3], const ptrdiff_t b_min[3],
    const ptrdiff_t req_min[3], const ptrdiff_t req_max[3]) 
{
    MaskedBlockResult<T> result;
    size_t num_voxels = blksize[0] * blksize[1] * blksize[2];
    
    // 1. 申请密集矩阵，默认全 0 (背景色)
    std::vector<T> dense(num_voxels, 0); 

    ptrdiff_t x0 = std::max((ptrdiff_t)0, req_min[0] - b_min[0]);
    ptrdiff_t x1 = std::min(blksize[0], req_max[0] - b_min[0]);
    ptrdiff_t y0 = std::max((ptrdiff_t)0, req_min[1] - b_min[1]);
    ptrdiff_t y1 = std::min(blksize[1], req_max[1] - b_min[1]);
    ptrdiff_t z0 = std::max((ptrdiff_t)0, req_min[2] - b_min[2]);
    ptrdiff_t z1 = std::min(blksize[2], req_max[2] - b_min[2]);

    uint32_t src_mask = (src_bits == 0) ? 0 : ((1 << src_bits) - 1);

    // 2. 局部解压：只读取请求框内部的有效像素
    for (ptrdiff_t dz = z0; dz < z1; ++dz) {
        for (ptrdiff_t dy = y0; dy < y1; ++dy) {
            for (ptrdiff_t dx = x0; dx < x1; ++dx) {
                size_t i = dx + dy * blksize[0] + dz * blksize[0] * blksize[1];
                if (src_bits == 0) {
                    dense[i] = src_palette[0];
                } else {
                    size_t src_bitpos = i * src_bits;
                    uint32_t val_idx = (src_bitstream[src_bitpos / 32] >> (src_bitpos % 32)) & src_mask;
                    if ((src_bitpos % 32) + src_bits > 32) {
                        val_idx |= (src_bitstream[src_bitpos / 32 + 1] << (32 - (src_bitpos % 32))) & src_mask;
                    }
                    dense[i] = src_palette[val_idx];
                }
            }
        }
    }

    // 3. 构建新的调色盘 (找出独立元素)
    std::vector<T> unique_vals;
    unique_vals.reserve(8);
    for (size_t i = 0; i < num_voxels; ++i) {
        T val = dense[i];
        bool found = false;
        for (T v : unique_vals) {
            if (v == val) { found = true; break; }
        }
        if (!found) unique_vals.push_back(val);
    }
    result.palette = unique_vals;
    size_t pal_size = unique_vals.size();

    // 4. 确定新的位深 (bits)
    if (pal_size <= 1) {
        result.bits = 0;
        if (pal_size == 0) result.palette = {0}; // 彻底被裁成全 0 了
        return result;
    } else if (pal_size <= 2) result.bits = 1;
    else if (pal_size <= 4) result.bits = 2;
    else if (pal_size <= 16) result.bits = 4;
    else if (pal_size <= 256) result.bits = 8;
    else {
        result.bits = 1;
        while ((1ULL << result.bits) < pal_size) result.bits *= 2;
    }

    // 5. 重组新位流 (极速原位位运算)
    size_t words_needed = (num_voxels * result.bits + 31) / 32;
    result.bitstream.assign(words_needed, 0);

    for (size_t i = 0; i < num_voxels; ++i) {
        T val = dense[i];
        uint32_t pal_idx = 0;
        for (size_t j = 0; j < pal_size; ++j) {
            if (unique_vals[j] == val) { pal_idx = (uint32_t)j; break; }
        }

        size_t dst_bitpos = i * result.bits;
        size_t word_idx = dst_bitpos / 32;
        size_t bit_offset = dst_bitpos % 32;

        result.bitstream[word_idx] |= (pal_idx << bit_offset);
        if (bit_offset + result.bits > 32) {
            result.bitstream[word_idx + 1] |= (pal_idx >> (32 - bit_offset));
        }
    }

    return result;
}
#endif // MASK_COMPRESS_HPP
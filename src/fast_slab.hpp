#ifndef FAST_SLAB_HPP
#define FAST_SLAB_HPP

#include <stdint.h>
#include <vector>
#include <cstdlib>

struct SlabBlock {
    const uint32_t* bitstream;
    const void* palette;
    uint8_t bits;
    bool has_nonzero;
    int gx, gy; 
};

template <typename T>
void DecompressSlabToBuffer(
    const SlabBlock* blocks, 
    size_t num_blocks,
    int bx, int by, int bz, 
    int nx_slab, int ny_slab, // 💥 确保这里有两个参数
    T* buffer) 
{
    int stride_x = nx_slab * bx;
    // 💥 修正 stride_xy，代表一整个 Z 平面的像素量
    size_t stride_xy = (size_t)stride_x * (ny_slab * by);

    for (size_t i = 0; i < num_blocks; ++i) {
        const auto& b = blocks[i];
        if (!b.has_nonzero) continue;

        int slab_x_origin = b.gx * bx;
        int slab_y_origin = b.gy * by;
        uint32_t mask = (b.bits == 0) ? 0 : (1U << b.bits) - 1;
        const T* pal = reinterpret_cast<const T*>(b.palette);

        for (int v_idx = 0; v_idx < bx * by * bz; ++v_idx) {
            int lx = v_idx % bx;
            int ly = (v_idx / bx) % by;
            int lz = v_idx / (bx * by);

            T value;
            if (b.bits == 0) {
                value = pal[0];
            } else {
                size_t bitpos = (size_t)v_idx * b.bits;
                uint32_t val_idx = (b.bitstream[bitpos / 32] >> (bitpos % 32)) & mask;
                if ((bitpos % 32) + b.bits > 32) {
                    val_idx |= (b.bitstream[bitpos / 32 + 1] << (32 - (bitpos % 32))) & mask;
                }
                value = pal[val_idx];
            }

            if (value != 0) {
                // 💥 索引计算：x + y*stride_x + z*stride_xy
                size_t buf_idx = (size_t)(slab_x_origin + lx) + 
                                 (size_t)(slab_y_origin + ly) * stride_x + 
                                 (size_t)lz * stride_xy;
                buffer[buf_idx] = value;
            }
        }
    }
}
#endif
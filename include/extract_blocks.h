#ifndef EXTRACT_BLOCKS_H_
#define EXTRACT_BLOCKS_H_

#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace compress_segmentation {

// 每个 Block 的元数据信息
template <class Label>
struct BlockInfo {
    size_t block_id;
    uint8_t encoding_bits;
    const Label* palette_ptr;
    size_t palette_size;
    const uint32_t* bitstream_ptr;
    size_t bitstream_word_count;
};

// 遍历指定的 intersection 区域，提取所有 Block 的指针
template <class Label>
void ExtractBlockMetadata(
    const uint32_t* input,
    const ptrdiff_t volume_size[3],
    const ptrdiff_t block_size[3],
    const ptrdiff_t intersection_start[3],
    const ptrdiff_t intersection_end[3],
    std::vector<BlockInfo<Label>>& out_blocks);

// 解压单个 Block 到目标数组的指定位置
template <class Label>
void DecompressSingleBlock(
    const uint32_t* bitstream, 
    const Label* palette, 
    uint8_t bits,
    Label* output_full,       // 目标大数组指针
    ptrdiff_t out_sx,         // 大数组的 X 步长 (通常为 1)
    ptrdiff_t out_sy,         // 大数组的 Y 步长
    ptrdiff_t out_sz,         // 大数组的 Z 步长
    int off_x, int off_y, int off_z, // 在大数组中的偏移坐标
    const ptrdiff_t block_size[3]    // Block 的尺寸 (通常 8,8,8)
);

template <class Label>
void CreateGenericBinaryBitstream(
    const uint32_t* src_bitstream, 
    const Label* src_palette, 
    uint8_t src_bits,
    Label segid, 
    uint32_t* dst_bitstream, 
    const ptrdiff_t block_size[3]
);


template <class Label>
int CheckBlockType(const Label* palette, size_t size, Label segid);

template <class Label>
void CompressSingleBlock(
    const Label* input,
    const ptrdiff_t input_strides[3],
    const ptrdiff_t block_size[3],
    std::vector<Label>* palette,
    std::vector<uint32_t>* bitstream,
    uint8_t* bits);

template <class Label>
void CreateGenericBinaryBitstreamWithMask(
    const uint32_t* src_bitstream, 
    const Label* src_palette, 
    uint8_t src_bits,
    Label segid, 
    uint32_t* dst_bitstream, 
    const ptrdiff_t block_size[3], 
    const ptrdiff_t b_origin[3],  // 新增：块的世界坐标起点
    const ptrdiff_t req_min[3],   // 新增：Query BBox 的最小边界
    const ptrdiff_t req_max[3]    // 新增：Query BBox 的最大边界
);

} // namespace compress_segmentation

#endif
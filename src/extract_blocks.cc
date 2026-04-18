#include "extract_blocks.h"
#include <unordered_map>

namespace compress_segmentation {

template <class Label>
void ExtractBlockMetadata(
    const uint32_t* input,
    const ptrdiff_t volume_size[3],
    const ptrdiff_t block_size[3],
    const ptrdiff_t intersection_start[3],
    const ptrdiff_t intersection_end[3],
    std::vector<BlockInfo<Label>>& out_blocks) {

    ptrdiff_t grid_size[3];
    ptrdiff_t block_start[3], block_end[3];
    size_t voxels_per_block = block_size[0] * block_size[1] * block_size[2];

    for (size_t i = 0; i < 3; ++i) {
        grid_size[i] = (volume_size[i] + block_size[i] - 1) / block_size[i];
        block_start[i] = intersection_start[i] / block_size[i];
        block_end[i] = (intersection_end[i] + block_size[i] - 1) / block_size[i];
    }

    ptrdiff_t block[3];
    for (block[2] = block_start[2]; block[2] < block_end[2]; ++block[2]) {
        for (block[1] = block_start[1]; block[1] < block_end[1]; ++block[1]) {
            for (block[0] = block_start[0]; block[0] < block_end[0]; ++block[0]) {
                
                // 计算 F-order 的 block 索引
                const size_t block_offset = block[0] + grid_size[0] * (block[1] + grid_size[1] * block[2]);

                // 参照 neuroglancer 格式解析 Header
                // Header 0: table_offset (24bit) | encoding_bits (8bit)
                // Header 1: encoded_value_offset (32bit)
                const uint32_t* header = input + (block_offset * 2);
                uint32_t table_offset = header[0] & 0xffffff;
                uint8_t bits = (header[0] >> 24) & 0xff;
                uint32_t encoded_value_start = header[1];

                BlockInfo<Label> info;
                info.block_id = block_offset;
                info.encoding_bits = bits;
                info.palette_ptr = reinterpret_cast<const Label*>(input + table_offset);
                info.palette_size = (bits == 0) ? 1 : (1 << bits);
                info.bitstream_ptr = input + encoded_value_start;
                // 位流长度（uint32个数）= ceil(像素数 * 位宽 / 32)
                info.bitstream_word_count = (voxels_per_block * bits + 31) / 32;

                out_blocks.push_back(info);
            }
        }
    }
}

template <class Label>
void DecompressSingleBlock(
    const uint32_t* bitstream, const Label* palette, uint8_t bits,
    Label* output_full, ptrdiff_t out_sx, ptrdiff_t out_sy, ptrdiff_t out_sz,
    int off_x, int off_y, int off_z,
    const ptrdiff_t block_size[3]) 
{
    // 情况 1：bits 为 0，说明块内所有像素值相同
    if (bits == 0) {
        Label val = palette[0];
        for (int z = 0; z < block_size[2]; ++z) {
            for (int y = 0; y < block_size[1]; ++y) {
                // 计算当前行在大数组中的起始位置
                Label* write_ptr = output_full + (off_x * out_sx) + ((y + off_y) * out_sy) + ((z + off_z) * out_sz);
                std::fill(write_ptr, write_ptr + block_size[0], val);
            }
        }
        return;
    }

    // 情况 2：存在位压缩流
    uint32_t bitmask = (1 << bits) - 1;
    size_t table_entry_size = sizeof(Label) / 4;

    for (int z = 0; z < block_size[2]; ++z) {
        for (int y = 0; y < block_size[1]; ++y) {
            // 计算当前行在位流中的起始位位置 (F-order)
            size_t bitpos = (z * block_size[1] + y) * block_size[0] * bits;
            
            for (int x = 0; x < block_size[0]; ++x) {
                size_t arraypos = bitpos / 32;
                size_t bitshift = bitpos % 32;
                
                // 提取索引值
                uint32_t bitval = (bitstream[arraypos] >> bitshift) & bitmask;
                // 处理位跨越 uint32 边界的情况
                if (bitshift + bits > 32) {
                    bitval |= (bitstream[arraypos + 1] << (32 - bitshift)) & bitmask;
                }

                // 计算大数组中的线性索引并写入
                size_t out_idx = ((off_x + x) * out_sx) + ((off_y + y) * out_sy) + ((off_z + z) * out_sz);
                output_full[out_idx] = palette[bitval];
                
                bitpos += bits;
            }
        }
    }
}


template <class Label>
void CreateGenericBinaryBitstream(
    const uint32_t* src_bitstream, const Label* src_palette, uint8_t src_bits,
    Label segid, uint32_t* dst_bitstream, const ptrdiff_t block_size[3]) 
{
    // 1. 预计算：在原调色盘中寻找目标 ID 的索引
    int target_idx = -1;
    size_t palette_size = (1 << src_bits);
    for (size_t i = 0; i < palette_size; ++i) {
        if (src_palette[i] == segid) {
            target_idx = (int)i;
            break;
        }
    }

    // 2. 初始化目标位流（1-bit 模式：每个 uint32 存 32 个像素的开关状态）
    // 对于 8x8x8，总共 512 像素，需要 16 个 uint32
    size_t total_voxels = block_size[0] * block_size[1] * block_size[2];
    size_t words_needed = (total_voxels + 31) / 32;
    std::fill(dst_bitstream, dst_bitstream + words_needed, 0);

    // 如果调色盘里根本没有这个 ID，位流保持全 0 即可
    if (target_idx == -1) return;

    uint32_t src_mask = (1 << src_bits) - 1;

    // 3. 核心转换循环
    for (size_t i = 0; i < total_voxels; ++i) {
        size_t src_bitpos = i * src_bits;
        uint32_t val_idx = (src_bitstream[src_bitpos / 32] >> (src_bitpos % 32)) & src_mask;
        
        // 处理位跨越 uint32 边界的情况
        if ((src_bitpos % 32) + src_bits > 32) {
            val_idx |= (src_bitstream[src_bitpos / 32 + 1] << (32 - (src_bitpos % 32))) & src_mask;
        }

        // 如果原始索引匹配，在目标 1-bit 位流的对应位置 1
        if (val_idx == (uint32_t)target_idx) {
            dst_bitstream[i / 32] |= (1u << (i % 32));
        }
    }
}


template <class Label>
int CheckBlockType(const Label* palette, size_t size, Label segid) {
    bool has_true = false;
    bool has_false = false;
    for (size_t i = 0; i < size; ++i) {
        if (palette[i] == segid) {
            has_true = true;
        } else {
            has_false = true;
        }
        // 一旦发现是混合块，立刻提前终止循环，极速返回
        if (has_true && has_false) {
            return 2; 
        }
    }
    return has_true ? 1 : 0;
}

template <class Label>
void CompressSingleBlock(
    const Label* input,
    const ptrdiff_t input_strides[3],
    const ptrdiff_t block_size[3],
    std::vector<Label>* palette,
    std::vector<uint32_t>* bitstream,
    uint8_t* bits) {

    // 1. 提取 Unique 值并构建调色板
    std::unordered_map<Label, uint32_t> value_to_index;
    palette->clear();
    for (ptrdiff_t z = 0; z < block_size[2]; ++z) {
        for (ptrdiff_t y = 0; y < block_size[1]; ++y) {
            for (ptrdiff_t x = 0; x < block_size[0]; ++x) {
                Label val = input[x * input_strides[0] + y * input_strides[1] + z * input_strides[2]];
                if (value_to_index.find(val) == value_to_index.end()) {
                    value_to_index[val] = palette->size();
                    palette->push_back(val);
                }
            }
        }
    }

    // (可选) 排序调色板，保持生成的 Bitstream 严格稳定
    // std::sort(palette->begin(), palette->end());
    // value_to_index.clear();
    // for(size_t i = 0; i < palette->size(); ++i) {
    //     value_to_index[(*palette)[i]] = i;
    // }

    // 2. 根据 Unique 数量确定需要的比特数 (Neuroglancer 规范: 0, 1, 2, 4, 8, 16, 32)
    size_t num_unique = palette->size();
    if (num_unique <= 1) {
        *bits = 0;
        return; // 纯色块无需生成 Bitstream
    } else if (num_unique <= 2) {
        *bits = 1;
    } else if (num_unique <= 4) {
        *bits = 2;
    } else if (num_unique <= 16) {
        *bits = 4;
    } else if (num_unique <= 256) {
        *bits = 8;
    } else if (num_unique <= 65536) {
        *bits = 16;
    } else {
        *bits = 32;
    }

    // 3. 将 3D Block 压缩为一维的 uint32_t 位流
    size_t elements_per_32bit = 32 / (*bits);
    size_t total_elements = block_size[0] * block_size[1] * block_size[2];
    size_t bitstream_words = (total_elements + elements_per_32bit - 1) / elements_per_32bit;
    bitstream->assign(bitstream_words, 0);

    size_t encoded_count = 0;
    // 严格按 F-order (X -> Y -> Z) 遍历，以匹配底层线性化协议
    for (ptrdiff_t z = 0; z < block_size[2]; ++z) {
        for (ptrdiff_t y = 0; y < block_size[1]; ++y) {
            for (ptrdiff_t x = 0; x < block_size[0]; ++x) {
                Label val = input[x * input_strides[0] + y * input_strides[1] + z * input_strides[2]];
                uint32_t encoded_val = value_to_index[val];
                
                size_t word_idx = (encoded_count * (*bits)) / 32;
                size_t bit_shift = (encoded_count * (*bits)) % 32;
                (*bitstream)[word_idx] |= (encoded_val << bit_shift);
                
                encoded_count++;
            }
        }
    }
}


// 模板显式实例化
template void ExtractBlockMetadata<uint8_t>(const uint32_t*, const ptrdiff_t[3], const ptrdiff_t[3], const ptrdiff_t[3], const ptrdiff_t[3], std::vector<BlockInfo<uint8_t>>&);
template void ExtractBlockMetadata<uint32_t>(const uint32_t*, const ptrdiff_t[3], const ptrdiff_t[3], const ptrdiff_t[3], const ptrdiff_t[3], std::vector<BlockInfo<uint32_t>>&);
template void ExtractBlockMetadata<uint64_t>(const uint32_t*, const ptrdiff_t[3], const ptrdiff_t[3], const ptrdiff_t[3], const ptrdiff_t[3], std::vector<BlockInfo<uint64_t>>&);

template void DecompressSingleBlock<uint8_t>(const uint32_t*, const uint8_t*, uint8_t, uint8_t*, ptrdiff_t, ptrdiff_t, ptrdiff_t, int, int, int, const ptrdiff_t[3]);
template void DecompressSingleBlock<uint32_t>(const uint32_t*, const uint32_t*, uint8_t, uint32_t*, ptrdiff_t, ptrdiff_t, ptrdiff_t, int, int, int, const ptrdiff_t[3]);
template void DecompressSingleBlock<uint64_t>(const uint32_t*, const uint64_t*, uint8_t, uint64_t*, ptrdiff_t, ptrdiff_t, ptrdiff_t, int, int, int, const ptrdiff_t[3]);

template void CreateGenericBinaryBitstream<uint8_t>(const uint32_t*, const uint8_t*, uint8_t, uint8_t, uint32_t*, const ptrdiff_t[3]);
template void CreateGenericBinaryBitstream<uint32_t>(const uint32_t*, const uint32_t*, uint8_t, uint32_t, uint32_t*, const ptrdiff_t[3]);
template void CreateGenericBinaryBitstream<uint64_t>(const uint32_t*, const uint64_t*, uint8_t, uint64_t, uint32_t*, const ptrdiff_t[3]);

template int CheckBlockType<uint8_t>(const uint8_t*, size_t, uint8_t);
template int CheckBlockType<uint32_t>(const uint32_t*, size_t, uint32_t);
template int CheckBlockType<uint64_t>(const uint64_t*, size_t, uint64_t);

template void CompressSingleBlock<uint8_t>(const uint8_t*, const ptrdiff_t[3], const ptrdiff_t[3], std::vector<uint8_t>*, std::vector<uint32_t>*, uint8_t*);
template void CompressSingleBlock<uint32_t>(const uint32_t*, const ptrdiff_t[3], const ptrdiff_t[3], std::vector<uint32_t>*, std::vector<uint32_t>*, uint8_t*);
template void CompressSingleBlock<uint64_t>(const uint64_t*, const ptrdiff_t[3], const ptrdiff_t[3], std::vector<uint64_t>*, std::vector<uint32_t>*, uint8_t*);


} // namespace compress_segmentation
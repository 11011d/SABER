#ifndef EXTRACT_BLOCKS_H_
#define EXTRACT_BLOCKS_H_

#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace compress_segmentation {

// Metadata for each block.
template <class Label>
struct BlockInfo {
    size_t block_id;
    uint8_t encoding_bits;
    const Label* palette_ptr;
    size_t palette_size;
    const uint32_t* bitstream_ptr;
    size_t bitstream_word_count;
};

// Traverse the specified intersection region and extract metadata for all blocks.
template <class Label>
void ExtractBlockMetadata(
    const uint32_t* input,
    const ptrdiff_t volume_size[3],
    const ptrdiff_t block_size[3],
    const ptrdiff_t intersection_start[3],
    const ptrdiff_t intersection_end[3],
    std::vector<BlockInfo<Label>>& out_blocks);

// Decompress a single block into the specified location of the target array.
template <class Label>
void DecompressSingleBlock(
    const uint32_t* bitstream, 
    const Label* palette, 
    uint8_t bits,
    Label* output_full,       // Pointer to the full output array.
    ptrdiff_t out_sx,         // X stride of the full array (typically 1).
    ptrdiff_t out_sy,         // Y stride of the full array.
    ptrdiff_t out_sz,         // Z stride of the full array.
    int off_x, int off_y, int off_z, // Offset coordinates within the full array.
    const ptrdiff_t block_size[3]    // Block dimensions (typically 8,8,8).
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
    const ptrdiff_t b_origin[3],  // Added: world-coordinate origin of the block.
    const ptrdiff_t req_min[3],   // Added: minimum boundary of the query bounding box.
    const ptrdiff_t req_max[3]    // Added: maximum boundary of the query bounding box.
);

} // namespace compress_segmentation

#endif

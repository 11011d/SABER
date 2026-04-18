/**
 * @license LICENSE_JANELIA.txt
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/
 

#include "decompress_segmentation.h"

#include <algorithm>
#include <iostream>
#if defined(__GNUC__) || defined(__clang__)
  #define PREFETCH_READ_L2(ptr) __builtin_prefetch((const void*)(ptr), 0, 2)
  #define PREFETCH_WRITE_L2(ptr) __builtin_prefetch((const void*)(ptr), 1, 2)
#elif defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_IX86))
  #include <xmmintrin.h>
  #define PREFETCH_READ_L2(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T1)
  #define PREFETCH_WRITE_L2(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T1)
#else
  #define PREFETCH_READ_L2(ptr)
  #define PREFETCH_WRITE_L2(ptr)
#endif
namespace compress_segmentation {

constexpr size_t dec_kBlockHeaderSize = 2;

template <class Label>
void DecompressChannel(
	const uint32_t* input,
	const ptrdiff_t volume_size[3],
	const ptrdiff_t block_size[3],
	const ptrdiff_t strides[4],
	Label* output,
	const ptrdiff_t channel
) {
	const size_t table_entry_size = (sizeof(Label) + sizeof(uint32_t) - 1) / sizeof(uint32_t);

	// determine number of grids for volume specified and block size
	// (must match what was encoded) 
	ptrdiff_t grid_size[3];
	for (size_t i = 0; i < 3; ++i) {
		grid_size[i] = (volume_size[i] + block_size[i] - 1) / block_size[i];
	}

	ptrdiff_t block[3];
	for (block[2] = 0; block[2] < grid_size[2]; ++block[2]) {
		for (block[1] = 0; block[1] < grid_size[1]; ++block[1]) {
			for (block[0] = 0; block[0] < grid_size[0]; ++block[0]) {
				const size_t block_offset =
					block[0] + grid_size[0] * (block[1] + grid_size[1] * block[2]);
			
				size_t encoded_bits, tableoffset, encoded_value_start;
				tableoffset = input[block_offset * dec_kBlockHeaderSize] & 0xffffff;
				encoded_bits = (input[block_offset * dec_kBlockHeaderSize] >> 24) & 0xff;
				encoded_value_start = input[block_offset * dec_kBlockHeaderSize + 1];

				// find absolute positions in output array (+ base_offset)
				size_t xmin = block[0]*block_size[0];
				size_t xmax = std::min(xmin + block_size[0], size_t(volume_size[0]));

				size_t ymin = block[1]*block_size[1];
				size_t ymax = std::min(ymin + block_size[1], size_t(volume_size[1]));

				size_t zmin = block[2]*block_size[2];
				size_t zmax = std::min(zmin + block_size[2], size_t(volume_size[2]));

				uint64_t bitmask = (1 << encoded_bits) - 1;
				for (size_t z = zmin; z < zmax; ++z) {
					for (size_t y = ymin; y < ymax; ++y) {
						size_t base_outindex = strides[1] * y + strides[2] * z + strides[3] * channel;
						size_t bitpos = (
							block_size[0] * ((z-zmin) * (block_size[1]) 
							+ (y-ymin)) * encoded_bits
						);

						for (size_t x = xmin; x < xmax; ++x) {
							size_t outindex = base_outindex + strides[0] * x;
							size_t bitshift = bitpos % 32;
							size_t arraypos = bitpos / (32);
							size_t bitval = 0;
							if (encoded_bits > 0) {
								bitval = (input[encoded_value_start + arraypos] >> bitshift) & bitmask; 
							}
							Label val = input[tableoffset + bitval*table_entry_size];
							if (table_entry_size == 2) {
								val |= uint64_t(input[tableoffset + bitval*table_entry_size+1]) << 32;
							}
							output[outindex] = val;
							bitpos += encoded_bits; 
						}
					}
				}
		  }
		}
  }
}

template <class Label>
void DecompressPartialIntersection(const uint32_t* input,
                                   const ptrdiff_t volume_size[3],
                                   const ptrdiff_t block_size[3],
                                   const ptrdiff_t chunk_start[3],
                                   const ptrdiff_t intersection_start[3],
                                   const ptrdiff_t intersection_end[3],
								                   const ptrdiff_t strides[4],
                                   Label* output,
                                   ptrdiff_t output_shape[3],
								                   const ptrdiff_t channel) {
  // Calculate output shape based on intersection
  output_shape[0] = intersection_end[0] - intersection_start[0];
  output_shape[1] = intersection_end[1] - intersection_start[1];
  output_shape[2] = intersection_end[2] - intersection_start[2];

  ptrdiff_t grid_size[3];
  ptrdiff_t block_start[3];
  ptrdiff_t block_end[3];
  for (size_t i = 0; i < 3; ++i) {
    grid_size[i] = (volume_size[i] + block_size[i] - 1) / block_size[i];
    block_start[i] = intersection_start[i] / block_size[i];
    block_end[i] = (intersection_end[i] + block_size[i] - 1) / block_size[i];
  }
  // Iterate over the intersecting blocks
  ptrdiff_t block[3];
  size_t block_xmin;
  size_t block_ymin;
  size_t block_zmin;

  size_t x_loop_end;
  size_t y_loop_end;
  size_t z_loop_end;

  size_t x_loop_start ;
  size_t y_loop_start ;
  size_t z_loop_start ;
  for (block[2] = block_start[2]; block[2] < block_end[2]; ++block[2]) {
	block_zmin = block[2] * block_size[2];
	z_loop_end = std::min(block_zmin + block_size[2], static_cast<size_t>(intersection_end[2]));
	z_loop_start = std::max(block_zmin, static_cast<size_t>(intersection_start[2]));
    for (block[1] = block_start[1]; block[1] < block_end[1]; ++block[1]) {
	  block_ymin = block[1] * block_size[1];
	  y_loop_end = std::min(block_ymin + block_size[1], static_cast<size_t>(intersection_end[1]));
	  y_loop_start = std::max(block_ymin, static_cast<size_t>(intersection_start[1]));
      for (block[0] = block_start[0]; block[0] < block_end[0]; ++block[0]) {
		block_xmin = block[0] * block_size[0];
		x_loop_end = std::min(block_xmin + block_size[0], static_cast<size_t>(intersection_end[0]));
		x_loop_start = std::max(block_xmin, static_cast<size_t>(intersection_start[0]));
        const size_t block_offset =
          block[0] + grid_size[0] * (block[1] + grid_size[1] * block[2]);

        size_t encoded_bits, tableoffset, encoded_value_start;
		tableoffset = input[block_offset * dec_kBlockHeaderSize] & 0xffffff;
		encoded_bits = (input[block_offset * dec_kBlockHeaderSize] >> 24) & 0xff;
		encoded_value_start = input[block_offset * dec_kBlockHeaderSize + 1];
        size_t table_entry_size = sizeof(Label)/4;
        uint64_t bitmask = (1 << encoded_bits) - 1;
        for (size_t z = z_loop_start; z < z_loop_end; ++z) {
          for (size_t y = y_loop_start; y < y_loop_end; ++y) {
            size_t base_outindex = strides[1] * (y-intersection_start[1]) + strides[2] * (z-intersection_start[2]) + strides[3] * channel;
			size_t bitpos = (
				(((z-block_zmin) * block_size[1] + y - block_ymin)* block_size[0] + x_loop_start - block_xmin) * encoded_bits 
			);

            for (size_t x = x_loop_start; x < x_loop_end; ++x) {
              	size_t outindex = base_outindex + strides[0] * (x-intersection_start[0]);
				size_t bitshift = bitpos % 32;
				size_t arraypos = bitpos / (32);
				size_t bitval = 0;
				if (encoded_bits > 0) {
					bitval = (input[encoded_value_start + arraypos] >> bitshift) & bitmask; 
				}
				Label val = input[tableoffset + bitval*table_entry_size];
				if (table_entry_size == 2) {
					val |= uint64_t(input[tableoffset + bitval*table_entry_size+1]) << 32;
				}
				output[outindex] = val;
				bitpos += encoded_bits; 
            }
          }
        }
      }
    }
  }
}

// template <class Label>
// void DecompressPartialIntersectionInPlace(const uint32_t* input,
//                                           const ptrdiff_t volume_size[3],
//                                           const ptrdiff_t block_size[3],
//                                           const ptrdiff_t chunk_start[3],
//                                           const ptrdiff_t intersection_start[3],
//                                           const ptrdiff_t intersection_end[3],
//                                           const ptrdiff_t request_start[3],
//                                           const ptrdiff_t strides[4],       
//                                           Label* output) {
//   ptrdiff_t grid_size[3];
//   ptrdiff_t block_start[3];
//   ptrdiff_t block_end[3];
//   ptrdiff_t offset[3];
//   for (size_t i = 0; i < 3; ++i) {
// 	  offset[i] = chunk_start[i] - request_start[i];
//     grid_size[i] = (volume_size[i] + block_size[i] - 1) / block_size[i];
//     block_start[i] = intersection_start[i] / block_size[i];
//     block_end[i] = (intersection_end[i] + block_size[i] - 1) / block_size[i];
//   }
//   // Iterate over the intersecting blocks
//   ptrdiff_t block[3];
//   size_t block_xmin;
//   size_t block_ymin;
//   size_t block_zmin;

//   size_t x_loop_end;
//   size_t y_loop_end;
//   size_t z_loop_end;

//   size_t x_loop_start ;
//   size_t y_loop_start ;
//   size_t z_loop_start ;
//   for (block[2] = block_start[2]; block[2] < block_end[2]; ++block[2]) {
//     block_zmin = block[2] * block_size[2];
//     z_loop_end = std::min(block_zmin + block_size[2], static_cast<size_t>(intersection_end[2]));
//     z_loop_start = std::max(block_zmin, static_cast<size_t>(intersection_start[2]));
//     for (block[1] = block_start[1]; block[1] < block_end[1]; ++block[1]) {
//       block_ymin = block[1] * block_size[1];
//       y_loop_end = std::min(block_ymin + block_size[1], static_cast<size_t>(intersection_end[1]));
//       y_loop_start = std::max(block_ymin, static_cast<size_t>(intersection_start[1]));
//       for (block[0] = block_start[0]; block[0] < block_end[0]; ++block[0]) {
//         block_xmin = block[0] * block_size[0];
//         x_loop_end = std::min(block_xmin + block_size[0], static_cast<size_t>(intersection_end[0]));
//         x_loop_start = std::max(block_xmin, static_cast<size_t>(intersection_start[0]));
//         const size_t block_offset =
//           block[0] + grid_size[0] * (block[1] + grid_size[1] * block[2]);

//         size_t encoded_bits, tableoffset, encoded_value_start;
//         tableoffset = input[block_offset * dec_kBlockHeaderSize] & 0xffffff;
//         encoded_bits = (input[block_offset * dec_kBlockHeaderSize] >> 24) & 0xff;
//         encoded_value_start = input[block_offset * dec_kBlockHeaderSize + 1];
//         size_t table_entry_size = sizeof(Label)/4;
//         uint64_t bitmask = (1 << encoded_bits) - 1;
//         for (size_t z = z_loop_start; z < z_loop_end; ++z) {
//           for (size_t y = y_loop_start; y < y_loop_end; ++y) {
//             size_t base_outindex = strides[1] * (y+offset[1]) + strides[2] * (z+offset[2]);
//             size_t bitpos = (
//               (((z-block_zmin) * block_size[1] + y - block_ymin)* block_size[0] + x_loop_start - block_xmin) * encoded_bits 
//             );
//             for (size_t x = x_loop_start; x < x_loop_end; ++x) {
//               size_t outindex = base_outindex + strides[0] * (x+offset[0]);
//               size_t bitshift = bitpos % 32;
//               size_t arraypos = bitpos / (32);
//               size_t bitval = 0;
//               if (encoded_bits > 0) {
//                 bitval = (input[encoded_value_start + arraypos] >> bitshift) & bitmask; 
//               }
//               Label val = input[tableoffset + bitval*table_entry_size];
//               if (table_entry_size == 2) {
//                 val |= uint64_t(input[tableoffset + bitval*table_entry_size+1]) << 32;
//               }
//               output[outindex] = val;
//               bitpos += encoded_bits; 
//             }
//           }
//         }
//       }
//     }
//   }
// }
template <class Label>
void DecompressPartialIntersectionInPlace(const uint32_t* input,
                                          const ptrdiff_t volume_size[3],
                                          const ptrdiff_t block_size[3],
                                          const ptrdiff_t chunk_start[3],
                                          const ptrdiff_t intersection_start[3],
                                          const ptrdiff_t intersection_end[3],
                                          const ptrdiff_t request_start[3],
                                          const ptrdiff_t strides[4],       
                                          Label* output,
                                          size_t l2cache_size) { // [新增参数]
  ptrdiff_t grid_size[3];
  ptrdiff_t block_start[3];
  ptrdiff_t block_end[3];
  ptrdiff_t offset[3];
  
  for (size_t i = 0; i < 3; ++i) {
    offset[i] = chunk_start[i] - request_start[i];
    grid_size[i] = (volume_size[i] + block_size[i] - 1) / block_size[i];
    block_start[i] = intersection_start[i] / block_size[i];
    block_end[i] = (intersection_end[i] + block_size[i] - 1) / block_size[i];
  }

  // 缓存行大小假设为 64 字节，计算每次预取步进的元素个数
  const size_t CACHE_LINE_BYTES = 64;
  const size_t elements_per_cache_line_out = CACHE_LINE_BYTES / sizeof(Label) > 0 ? CACHE_LINE_BYTES / sizeof(Label) : 1;
  const size_t uint32_per_cache_line = CACHE_LINE_BYTES / sizeof(uint32_t);

  ptrdiff_t block[3];
  size_t block_xmin, block_ymin, block_zmin;
  size_t x_loop_end, y_loop_end, z_loop_end;
  size_t x_loop_start, y_loop_start, z_loop_start;

  for (block[2] = block_start[2]; block[2] < block_end[2]; ++block[2]) {
    block_zmin = block[2] * block_size[2];
    z_loop_end = std::min(block_zmin + block_size[2], static_cast<size_t>(intersection_end[2]));
    z_loop_start = std::max(block_zmin, static_cast<size_t>(intersection_start[2]));
    
    for (block[1] = block_start[1]; block[1] < block_end[1]; ++block[1]) {
      block_ymin = block[1] * block_size[1];
      y_loop_end = std::min(block_ymin + block_size[1], static_cast<size_t>(intersection_end[1]));
      y_loop_start = std::max(block_ymin, static_cast<size_t>(intersection_start[1]));
      
      for (block[0] = block_start[0]; block[0] < block_end[0]; ++block[0]) {
        block_xmin = block[0] * block_size[0];
        x_loop_end = std::min(block_xmin + block_size[0], static_cast<size_t>(intersection_end[0]));
        x_loop_start = std::max(block_xmin, static_cast<size_t>(intersection_start[0]));
        
        const size_t block_offset = block[0] + grid_size[0] * (block[1] + grid_size[1] * block[2]);

        // ==========================================
        // L2 Cache 预取：提前加载下一个 Block 的数据
        // ==========================================
        if (l2cache_size > 0) {
          ptrdiff_t nb_x = block[0] + 1;
          ptrdiff_t nb_y = block[1];
          ptrdiff_t nb_z = block[2];
          if (nb_x == block_end[0]) { nb_x = block_start[0]; nb_y++; }
          if (nb_y == block_end[1]) { nb_y = block_start[1]; nb_z++; }

          if (nb_z < block_end[2]) {
            size_t next_block_offset = nb_x + grid_size[0] * (nb_y + grid_size[1] * nb_z);
            
            uint32_t next_h0 = input[next_block_offset * 2]; // dec_kBlockHeaderSize = 2
            uint32_t next_h1 = input[next_block_offset * 2 + 1];
            size_t next_tableoffset = next_h0 & 0xffffff;
            size_t next_encoded_bits = (next_h0 >> 24) & 0xff;
            size_t next_encoded_value_start = next_h1;

            size_t table_entry_size = sizeof(Label) / 4;
            size_t num_elements = block_size[0] * block_size[1] * block_size[2];
            size_t next_encoded_words = (num_elements * next_encoded_bits + 31) / 32;
            size_t next_table_words = (next_encoded_bits > 0) ? ((1ull << next_encoded_bits) * table_entry_size) : table_entry_size;

            size_t total_bytes = (next_encoded_words + next_table_words + 2) * sizeof(uint32_t);

            if (total_bytes < l2cache_size) {
              // 预取 Lookup Table (按 Cache Line 步进)
              for (size_t p = 0; p < next_table_words; p += uint32_per_cache_line) {
                PREFETCH_READ_L2(&input[next_tableoffset + p]);
              }
              // 预取压缩位数据
              for (size_t p = 0; p < next_encoded_words; p += uint32_per_cache_line) {
                PREFETCH_READ_L2(&input[next_encoded_value_start + p]);
              }
            }
          }
        }
        // ==========================================

        size_t encoded_bits, tableoffset, encoded_value_start;
        tableoffset = input[block_offset * 2] & 0xffffff; // dec_kBlockHeaderSize = 2
        encoded_bits = (input[block_offset * 2] >> 24) & 0xff;
        encoded_value_start = input[block_offset * 2 + 1];
        size_t table_entry_size = sizeof(Label)/4;
        uint64_t bitmask = (1ull << encoded_bits) - 1;

        for (size_t z = z_loop_start; z < z_loop_end; ++z) {
          for (size_t y = y_loop_start; y < y_loop_end; ++y) {
            size_t base_outindex = strides[1] * (y+offset[1]) + strides[2] * (z+offset[2]);
            
            // ==========================================
            // L2 Cache 预取：预热连续写入的 Output 内存
            // ==========================================
            if (l2cache_size > 0 && strides[0] == 1) {
              size_t outindex_start = base_outindex + (x_loop_start + offset[0]);
              size_t out_elements = x_loop_end - x_loop_start;
              // 按 Cache Line 步进发出写预取信号，避免 Write-Allocate 造成的停顿
              for (size_t p = 0; p < out_elements; p += elements_per_cache_line_out) {
                PREFETCH_WRITE_L2(&output[outindex_start + p]);
              }
            }
            // ==========================================

            size_t bitpos = (
              (((z-block_zmin) * block_size[1] + y - block_ymin)* block_size[0] + x_loop_start - block_xmin) * encoded_bits 
            );
            
            for (size_t x = x_loop_start; x < x_loop_end; ++x) {
              size_t outindex = base_outindex + strides[0] * (x+offset[0]);
              size_t bitshift = bitpos % 32;
              size_t arraypos = bitpos / 32;
              size_t bitval = 0;
              
              if (encoded_bits > 0) {
                bitval = (input[encoded_value_start + arraypos] >> bitshift) & bitmask; 
              }
              
              Label val = input[tableoffset + bitval*table_entry_size];
              if (table_entry_size == 2) {
                val |= uint64_t(input[tableoffset + bitval*table_entry_size+1]) << 32;
              }
              
              output[outindex] = val;
              bitpos += encoded_bits; 
            }
          }
        }
      }
    }
  }
}

template <class Label>
void DecompressChannels(
	const uint32_t* input,
	const ptrdiff_t volume_size[4],
	const ptrdiff_t block_size[3],
	const ptrdiff_t strides[4],
	Label* output
) {

  /*
  A simple encoding is used to store multiple channels of compressed segmentation data 
  (assumed to have the same x, y, and z dimensions and compression block size) together. 
  The number of channels, num_channels, is assumed to be known.

  The header consists of num_channels little-endian 32-bit unsigned integers specifying 
  the offset, in 32-bit units from the start of the file, at which the data for each 
  channel begins. The channels should be packed in order, and without any padding. 
  The offset specified in the header for the first channel must be equal to num_channels.

  In the special case that this format is used to encode just a single compressed 
  segmentation channel, the compressed segmentation data is simply prefixed with a 
  single 1 value (encoded as a little-endian 32-bit unsigned integer).
  */
  for (size_t channel_i = 0; channel_i < static_cast<size_t>(volume_size[3]); ++channel_i) {
		DecompressChannel(
			input + input[channel_i], volume_size,
			block_size, strides, output, channel_i
		);
  }
}

template <class Label>
void DecompressPartialChannelsIntersection(const uint32_t* input,
                                           const ptrdiff_t volume_size[4],
                                           const ptrdiff_t block_size[3],
                                           const ptrdiff_t chunk_start[3],
                                           const ptrdiff_t intersection_start[3],
                                           const ptrdiff_t intersection_end[3],
										                       const ptrdiff_t strides[4],
                                           Label* output) {
  ptrdiff_t local_start[3];
  ptrdiff_t local_end[3];
  ptrdiff_t output_shape[3];

  for (size_t i = 0; i < 3; ++i) {
    local_start[i] = intersection_start[i] - chunk_start[i];
    local_end[i]   = intersection_end[i]   - chunk_start[i];
  }
  for (size_t channel_i = 0; channel_i < static_cast<size_t>(volume_size[3]); ++channel_i) {
    DecompressPartialIntersection(input + input[channel_i], volume_size, block_size,
                                  chunk_start, local_start, local_end, strides,
                                  output, output_shape, channel_i);
  }
}

template <class Label>
void DecompressPartialChannelsIntersectionInPlace(const uint32_t* input,
                                                  const ptrdiff_t volume_size[4],
                                                  const ptrdiff_t block_size[3],
                                                  const ptrdiff_t chunk_start[3],
                                                  const ptrdiff_t intersection_start[3],
                                                  const ptrdiff_t intersection_end[3],
                                                  const ptrdiff_t request_start[3], 
                                                  const ptrdiff_t strides[4],       
                                                  Label* output,
                                                  size_t l2cache_size) {
	ptrdiff_t local_start[3];
  ptrdiff_t local_end[3];
  for (size_t i = 0; i < 3; ++i) {
    local_start[i] = intersection_start[i] - chunk_start[i];
    local_end[i]   = intersection_end[i]   - chunk_start[i];
  }
  for (size_t channel_i = 0; channel_i < static_cast<size_t>(volume_size[3]); ++channel_i) {
    Label* channel_output = output + strides[3] * channel_i; 
    DecompressPartialIntersectionInPlace(input + input[channel_i], 
                                         volume_size, 
                                         block_size, 
                                         chunk_start, 
                                         local_start, 
                                         local_end, 
                                         request_start, 
                                         strides, 
                                         channel_output,
                                         l2cache_size); 
  }
}



void DecompressPartialChannelsIntersectionParallel(std::vector<Request>& requests, int parallel, size_t l2cache_size) {
  if (parallel > requests.size()){
    parallel = requests.size();
  }
    if (parallel <= 1) {
        for (auto& req : requests) {
            ptrdiff_t intersection_start[3], intersection_end[3];
            bool empty = false;
            for (size_t i = 0; i < 3; ++i) {
                intersection_start[i] = std::max(req.chunk_start[i], req.request_start[i]);
                intersection_end[i] = std::min(req.chunk_end[i], req.request_end[i]);
                if (intersection_start[i] >= intersection_end[i]) {
                    empty = true;
                    break;
                }
            }
            if (empty) return;

            if (req.is_uint64) {
                // 4D
                if (req.ndim == 4) {
                    DecompressPartialChannelsIntersectionInPlace<uint64_t>(
                        req.encoded_ptr, req.volume_size, req.block_size, req.chunk_start, 
                        intersection_start, intersection_end, req.request_start, 
                        req.strides, static_cast<uint64_t*>(req.output_array_ptr), l2cache_size);
                }
                // 3D
                else {
                    ptrdiff_t temp_volume_size[4] = {req.volume_size[0], req.volume_size[1], req.volume_size[2], 1};
                    ptrdiff_t temp_strides[4] = {req.strides[0], req.strides[1], req.strides[2], 1};
                    DecompressPartialChannelsIntersectionInPlace<uint64_t>(
                        req.encoded_ptr, temp_volume_size, req.block_size, req.chunk_start, 
                        intersection_start, intersection_end, req.request_start, 
                        temp_strides, static_cast<uint64_t*>(req.output_array_ptr), l2cache_size);
                }
            } else { // uint32
                // 4D
                if (req.ndim == 4) {
                    DecompressPartialChannelsIntersectionInPlace<uint32_t>(
                        req.encoded_ptr, req.volume_size, req.block_size, req.chunk_start, 
                        intersection_start, intersection_end, req.request_start, 
                        req.strides, static_cast<uint32_t*>(req.output_array_ptr), l2cache_size);
                }
                // 3D
                else {
                    ptrdiff_t temp_volume_size[4] = {req.volume_size[0], req.volume_size[1], req.volume_size[2], 1};
                    ptrdiff_t temp_strides[4] = {req.strides[0], req.strides[1], req.strides[2], 1};
                    DecompressPartialChannelsIntersectionInPlace<uint32_t>(
                        req.encoded_ptr, temp_volume_size, req.block_size, req.chunk_start, 
                        intersection_start, intersection_end, req.request_start, 
                        temp_strides, static_cast<uint32_t*>(req.output_array_ptr), l2cache_size);
                }
            }
        }
    } else {
        // 多线程执行
        ThreadPool pool(parallel);
        for (auto& req : requests) {
            pool.enqueue([&]() {
                ptrdiff_t intersection_start[3], intersection_end[3];
                bool empty = false;
                for (size_t i = 0; i < 3; ++i) {
                    intersection_start[i] = std::max(req.chunk_start[i], req.request_start[i]);
                    intersection_end[i] = std::min(req.chunk_end[i], req.request_end[i]);
                    if (intersection_start[i] >= intersection_end[i]) {
                        empty = true;
                        break;
                    }
                }
                if (empty) return;

                if (req.is_uint64) {
                    if (req.ndim == 4) {
                        DecompressPartialChannelsIntersectionInPlace<uint64_t>(
                            req.encoded_ptr, req.volume_size, req.block_size, req.chunk_start, 
                            intersection_start, intersection_end, req.request_start, 
                            req.strides, static_cast<uint64_t*>(req.output_array_ptr), l2cache_size);
                    } else {
                        ptrdiff_t temp_volume_size[4] = {req.volume_size[0], req.volume_size[1], req.volume_size[2], 1};
                        ptrdiff_t temp_strides[4] = {req.strides[0], req.strides[1], req.strides[2], 1};
                        DecompressPartialChannelsIntersectionInPlace<uint64_t>(
                            req.encoded_ptr, temp_volume_size, req.block_size, req.chunk_start, 
                            intersection_start, intersection_end, req.request_start, 
                            temp_strides, static_cast<uint64_t*>(req.output_array_ptr), l2cache_size);
                    }
                } else { // uint32
                    if (req.ndim == 4) {
                        DecompressPartialChannelsIntersectionInPlace<uint32_t>(
                            req.encoded_ptr, req.volume_size, req.block_size, req.chunk_start, 
                            intersection_start, intersection_end, req.request_start, 
                            req.strides, static_cast<uint32_t*>(req.output_array_ptr), l2cache_size);
                    } else {
                        ptrdiff_t temp_volume_size[4] = {req.volume_size[0], req.volume_size[1], req.volume_size[2], 1};
                        ptrdiff_t temp_strides[4] = {req.strides[0], req.strides[1], req.strides[2], 1};
                        DecompressPartialChannelsIntersectionInPlace<uint32_t>(
                            req.encoded_ptr, temp_volume_size, req.block_size, req.chunk_start, 
                            intersection_start, intersection_end, req.request_start, 
                            temp_strides, static_cast<uint32_t*>(req.output_array_ptr), l2cache_size);
                    }
                }
            });
        }
    }
}



#define DO_INSTANTIATE(Label)                                        \
  template void DecompressChannel<Label>(                              \
	  const uint32_t* input, const ptrdiff_t volume_size[3],       \
	  const ptrdiff_t block_size[3], \
	  const ptrdiff_t strides[4], \
	  Label* output, \
	  const ptrdiff_t channel);                                \
  template void DecompressChannels<Label>(                             \
	  const uint32_t* input, const ptrdiff_t volume_size[4],            \
	  const ptrdiff_t block_size[3], \
	  const ptrdiff_t strides[4], \
	  Label* output);                                \
  template void DecompressPartialChannelsIntersection<Label>(const uint32_t* input,\
                                           const ptrdiff_t volume_size[4],\
                                           const ptrdiff_t block_size[3],\
                                           const ptrdiff_t chunk_start[3],\
                                           const ptrdiff_t intersection_start[3],\
                                           const ptrdiff_t intersection_end[3],\
										                       const ptrdiff_t strides[4],\
                                           Label* output);\
  template void DecompressPartialIntersection<Label>(const uint32_t* input,\
                                   const ptrdiff_t volume_size[3],\
                                   const ptrdiff_t block_size[3],\
                                   const ptrdiff_t chunk_start[3],\
                                   const ptrdiff_t intersection_start[3],\
                                   const ptrdiff_t intersection_end[3],\
								                   const ptrdiff_t strides[4],\
                                   Label* output,\
                                   ptrdiff_t output_shape[3],\
								                   const ptrdiff_t channel);\
  template void DecompressPartialIntersectionInPlace<Label>(const uint32_t* input, const ptrdiff_t volume_size[3], const ptrdiff_t block_size[3], const ptrdiff_t chunk_start[3], const ptrdiff_t intersection_start[3], const ptrdiff_t intersection_end[3], const ptrdiff_t request_start[3], const ptrdiff_t strides[4], Label* output, size_t l2cache_size);\
  template void DecompressPartialChannelsIntersectionInPlace<Label>(const uint32_t* input, const ptrdiff_t volume_size[4], const ptrdiff_t block_size[3], const ptrdiff_t chunk_start[3], const ptrdiff_t intersection_start[3], const ptrdiff_t intersection_end[3], const ptrdiff_t request_start[3], const ptrdiff_t strides[4], Label* output, size_t l2cache_size);
DO_INSTANTIATE(uint32_t)
DO_INSTANTIATE(uint64_t)

#undef DO_INSTANTIATE

}  // namespace compress_segmentation

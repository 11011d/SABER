#ifndef BLOCK_ARENA_HPP
#define BLOCK_ARENA_HPP

#include <vector>
#include <cstdint>
#include <cstring>
#include "fast_cc.hpp"  // CBlock struct
#include "mask_compress.hpp"
#include "../include/extract_blocks.h"

// Ownership storage for each block's data.
struct StoredBlock {
    std::vector<uint8_t>  palette_bytes;  // Raw palette bytes.
    std::vector<uint32_t> bitstream;       // Bitstream data.
    uint8_t bits;
    bool    has_nonzero;
    bool    is_null;

    StoredBlock() : bits(0), has_nonzero(false), is_null(true) {}
};

// C++-managed block storage.
// - Persistently maintains the CBlock pointer array to avoid repeated Python-side conversions.
class BlockArena {
public:
    int total;
    int itemsize;  // Number of bytes per palette element (1/4/8).
    std::vector<StoredBlock> blocks;
    std::vector<CBlock>      cblocks;

    BlockArena(int n, int item_bytes)
        : total(n), itemsize(item_bytes), blocks(n), cblocks(n)
    {
        for (int i = 0; i < n; ++i) {
            cblocks[i].has_nonzero = false;
            cblocks[i].bits        = 0;
            cblocks[i].palette     = nullptr;
            cblocks[i].bitstream   = nullptr;
        }
    }

    // Set the data for one block (palette byte array + bitstream).
    void set_block(int idx,
                   const uint8_t* pal, int pal_bytes,
                   uint8_t bits_val,
                   const uint32_t* bs, int bs_words,
                   bool hnz)
    {
        if (idx < 0 || idx >= total) return;
        auto& b = blocks[idx];
        b.palette_bytes.assign(pal, pal + pal_bytes);
        b.bits        = bits_val;
        b.has_nonzero = hnz;
        b.is_null     = false;
        if (bits_val > 0 && bs != nullptr && bs_words > 0)
            b.bitstream.assign(bs, bs + bs_words);
        else
            b.bitstream.clear();
        _rebuild_cblock(idx);
    }

    // Set the specified block to empty (all zeros).
    void set_null(int idx) {
        if (idx < 0 || idx >= total) return;
        auto& b = blocks[idx];
        b.is_null     = true;
        b.has_nonzero = false;
        b.palette_bytes.clear();
        b.bitstream.clear();
        b.bits = 0;
        auto& cb = cblocks[idx];
        cb.has_nonzero = false;
        cb.bits        = 0;
        cb.palette     = nullptr;
        cb.bitstream   = nullptr;
    }

    // Set all blocks to a single-element zero palette in bulk (for fast clearing).
    void set_all_false(const uint8_t* false_bytes, int false_nbytes) {
        for (int i = 0; i < total; ++i) {
            auto& b = blocks[i];
            b.palette_bytes.assign(false_bytes, false_bytes + false_nbytes);
            b.bits        = 0;
            b.has_nonzero = false;
            b.is_null     = false;
            b.bitstream.clear();
            auto& cb = cblocks[i];
            cb.has_nonzero = false;
            cb.bits        = 0;
            cb.palette     = b.palette_bytes.data();
            cb.bitstream   = nullptr;
        }
    }

    // Get the precomputed read-only CBlock array.
    const CBlock* get_cblocks() const { return cblocks.data(); }

    // The accessors below are used by Cython __getitem__ (non-hot path).
    int     get_bits(int idx)          const { return (idx>=0&&idx<total) ? blocks[idx].bits : 0; }
    bool    get_is_null(int idx)       const { return (idx>=0&&idx<total) ? blocks[idx].is_null : true; }
    bool    get_has_nonzero(int idx)   const { return (idx>=0&&idx<total) ? blocks[idx].has_nonzero : false; }
    int     get_pal_byte_size(int idx) const { return (idx>=0&&idx<total) ? (int)blocks[idx].palette_bytes.size() : 0; }
    int     get_bs_size(int idx)       const { return (idx>=0&&idx<total) ? (int)blocks[idx].bitstream.size() : 0; }

    const uint8_t*  get_pal_bytes(int idx) const {
        if (idx<0||idx>=total||blocks[idx].palette_bytes.empty()) return nullptr;
        return blocks[idx].palette_bytes.data();
    }
    const uint32_t* get_bs_data(int idx)  const {
        if (idx<0||idx>=total||blocks[idx].bitstream.empty()) return nullptr;
        return blocks[idx].bitstream.data();
    }

private:
    void _rebuild_cblock(int idx) {
        auto& b  = blocks[idx];
        auto& cb = cblocks[idx];
        cb.has_nonzero = b.has_nonzero;
        cb.bits        = b.bits;
        if (!b.is_null && !b.palette_bytes.empty()) {
            cb.palette   = b.palette_bytes.data();
            cb.bitstream = (b.bits > 0 && !b.bitstream.empty())
                           ? b.bitstream.data() : nullptr;
        } else {
            cb.palette   = nullptr;
            cb.bitstream = nullptr;
        }
    }
};

// C++ implementation of ProcessSingleBlockGlobal to bypass Cython overhead
template <class UINT>
inline void ProcessSingleBlockGlobal(
    const compress_segmentation::BlockInfo<UINT>& b,
    const ptrdiff_t rel_grid[3],
    size_t nx, size_t ny,
    const ptrdiff_t local_start[3],
    const ptrdiff_t local_end[3],
    const ptrdiff_t blksize[3],
    BlockArena* arena,
    int arena_nx, int arena_nxy, int itemsize)
{
    size_t bx = b.block_id % nx;
    size_t by = (b.block_id / nx) % ny;
    size_t bz = b.block_id / (nx * ny);

    ptrdiff_t b_min[3];
    ptrdiff_t b_max[3];
    b_min[0] = bx * blksize[0]; b_max[0] = b_min[0] + blksize[0];
    b_min[1] = by * blksize[1]; b_max[1] = b_min[1] + blksize[1];
    b_min[2] = bz * blksize[2]; b_max[2] = b_min[2] + blksize[2];

    bool is_partial = false;
    bool is_outside = false;

    for (int i = 0; i < 3; ++i) {
        if (b_max[i] <= local_start[i] || b_min[i] >= local_end[i]) {
            is_outside = true;
            break;
        }
        if (b_min[i] < local_start[i] || b_max[i] > local_end[i]) {
            is_partial = true;
        }
    }

    if (is_outside) {
        return;
    }

    ptrdiff_t global_grid_coord[3];
    global_grid_coord[0] = rel_grid[0] + bx;
    global_grid_coord[1] = rel_grid[1] + by;
    global_grid_coord[2] = rel_grid[2] + bz;

    int global_block_id = global_grid_coord[0] + global_grid_coord[1] * arena_nx + global_grid_coord[2] * arena_nxy;

    if (global_block_id < 0 || global_block_id >= arena->total) {
        return;
    }

    if (!is_partial) {
        bool has_nz = false;
        int pal_bytes = b.palette_size * itemsize;
        const uint8_t* pal_ptr8 = reinterpret_cast<const uint8_t*>(b.palette_ptr);
        for (int i = 0; i < pal_bytes; ++i) {
            if (pal_ptr8[i] != 0) {
                has_nz = true;
                break;
            }
        }

        arena->set_block(
            global_block_id,
            pal_ptr8, pal_bytes,
            b.encoding_bits,
            b.bitstream_ptr, b.bitstream_word_count,
            has_nz
        );
        return;
    }

    // Handle partially intersecting blocks (Partial Block).
    MaskedBlockResult<UINT> res = MaskAndRecompressBlock<UINT>(
        b.palette_ptr, b.encoding_bits, b.bitstream_ptr,
        blksize, b_min, local_start, local_end
    );

    if (res.bits == 0 && (res.palette.empty() || res.palette[0] == 0)) {
        return;
    }

    bool has_nz = false;
    int pal_bytes = res.palette.size() * itemsize;
    const uint8_t* res_pal_ptr8 = reinterpret_cast<const uint8_t*>(res.palette.data());
    for (int i = 0; i < pal_bytes; ++i) {
        if (res_pal_ptr8[i] != 0) {
            has_nz = true;
            break;
        }
    }

    arena->set_block(
        global_block_id,
        res_pal_ptr8, pal_bytes,
        res.bits,
        res.bits > 0 ? res.bitstream.data() : nullptr, res.bitstream.size(),
        has_nz
    );
}

#endif  // BLOCK_ARENA_HPP

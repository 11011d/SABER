#ifndef FAST_CC_HPP
#define FAST_CC_HPP

#include <stdint.h>
#include <vector>
#include <cstdlib>
#include <cstring>

struct Point3D { int x, y, z; };

struct CBlock {
    const uint32_t* bitstream;
    const void* palette;
    uint8_t bits;
    bool has_nonzero;
};

// 极速全块解码器
template<typename T>
void DecodeBlockToCache(const CBlock& b, int voxels_per_block, uint8_t* cache) {
    if (b.bits == 0) {
        memset(cache, 1, voxels_per_block);
        return;
    }
    uint32_t mask = (1U << b.bits) - 1;
    const T* pal = reinterpret_cast<const T*>(b.palette);

    for (int i = 0; i < voxels_per_block; ++i) {
        size_t bitpos = i * b.bits;
        uint32_t val_idx = (b.bitstream[bitpos / 32] >> (bitpos % 32)) & mask;
        if ((bitpos % 32) + b.bits > 32) {
            val_idx |= (b.bitstream[bitpos / 32 + 1] << (32 - (bitpos % 32))) & mask;
        }
        // 💥 变更点：只要不是 0，就在缓存中标记为 1
        cache[i] = (pal[val_idx] != 0) ? 1 : 0; 
    }
}

// =========================================================================
// 💥 新增：C++ 极速寻找最近种子点 (彻底替换 Python 循环)
// =========================================================================
template<typename T>
bool FastNearestSeed(
    const CBlock* blocks, int nx, int ny, int nz, int bx, int by, int bz,
    int req_x, int req_y, int req_z, int q2p_x, int q2p_y, int q2p_z,
    int cx, int cy, int cz, int& out_sx, int& out_sy, int& out_sz, 
    bool include_self = false) // 💥 新增参数，默认 false
{
    long long min_dist_sq = -1;
    bool found = false;

    for(int gz=0; gz<nz; ++gz) {
        for(int gy=0; gy<ny; ++gy) {
            for(int gx=0; gx<nx; ++gx) {
                size_t b_idx = gx + gy*nx + gz*nx*ny;
                
                if (!blocks[b_idx].has_nonzero) continue; 

                uint32_t mask = (blocks[b_idx].bits == 0) ? 0 : (1U << blocks[b_idx].bits) - 1;
                const T* pal = reinterpret_cast<const T*>(blocks[b_idx].palette);

                for(int lz=0; lz<bz; ++lz) {
                    for(int ly=0; ly<by; ++ly) {
                        for(int lx=0; lx<bx; ++lx) {
                            int v_idx = lx + ly*bx + lz*bx*by;
                            bool is_nonzero = false; 
                            
                            if (blocks[b_idx].bits == 0) {
                                is_nonzero = (pal[0] != 0); 
                            } else {
                                size_t bitpos = v_idx * blocks[b_idx].bits;
                                uint32_t val_idx = (blocks[b_idx].bitstream[bitpos / 32] >> (bitpos % 32)) & mask;
                                if ((bitpos % 32) + blocks[b_idx].bits > 32) {
                                    val_idx |= (blocks[b_idx].bitstream[bitpos / 32 + 1] << (32 - (bitpos % 32))) & mask;
                                }
                                is_nonzero = (pal[val_idx] != 0); 
                            }

                            if (is_nonzero) {
                                int px_req = gx * bx + lx - q2p_x;
                                int py_req = gy * by + ly - q2p_y;
                                int pz_req = gz * bz + lz - q2p_z;

                                if (px_req >= 0 && px_req < req_x && 
                                    py_req >= 0 && py_req < req_y && 
                                    pz_req >= 0 && pz_req < req_z) {
                                    
                                    long long dist_sq = (long long)(px_req - cx)*(px_req - cx) + 
                                                        (long long)(py_req - cy)*(py_req - cy) + 
                                                        (long long)(pz_req - cz)*(pz_req - cz);
                                    
                                    // 💥 核心逻辑：如果排除了自身，且距离为0，直接跳过该点
                                    if (!include_self && dist_sq == 0) continue;
                                                        
                                    if (min_dist_sq == -1 || dist_sq < min_dist_sq) {
                                        min_dist_sq = dist_sq;
                                        out_sx = px_req; 
                                        out_sy = py_req; 
                                        out_sz = pz_req;
                                        found = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return found;
}

// =========================================================================
// 💥 修复：注入 Offset 偏移量，校准寻路坐标系
// =========================================================================
template<typename T>
std::vector<uint8_t*> SparseBFS26(
    const CBlock* blocks, int nx, int ny, int nz, int bx, int by, int bz,
    int req_x, int req_y, int req_z, int q2p_x, int q2p_y, int q2p_z, 
    int sx, int sy, int sz) 
{
    size_t total_blocks = (size_t)nx * ny * nz;
    int voxels_per_block = bx * by * bz;

    std::vector<uint8_t*> cc_masks(total_blocks, nullptr);
    std::vector<uint8_t*> cached_masks(total_blocks, nullptr);

    // 修复暗坑：通过 q2p_offset 完美桥接 Query 坐标与 Physical 网格
    auto get_indices = [&](int px_req, int py_req, int pz_req, size_t& b_idx, int& v_idx) {
        int full_x = px_req + q2p_x, full_y = py_req + q2p_y, full_z = pz_req + q2p_z;
        int gx = full_x / bx, gy = full_y / by, gz = full_z / bz;
        b_idx = gx + gy * nx + gz * nx * ny;
        int lx = full_x % bx, ly = full_y % by, lz = full_z % bz;
        v_idx = lx + ly * bx + lz * bx * by; 
    };

    size_t seed_b_idx;
    int seed_v_idx;
    get_indices(sx, sy, sz, seed_b_idx, seed_v_idx);

    if (!blocks[seed_b_idx].has_nonzero) return cc_masks;

    cached_masks[seed_b_idx] = (uint8_t*)calloc(voxels_per_block, 1);
    DecodeBlockToCache<T>(blocks[seed_b_idx], voxels_per_block, cached_masks[seed_b_idx]);

    if (cached_masks[seed_b_idx][seed_v_idx] == 0) {
        free(cached_masks[seed_b_idx]); 
        return cc_masks; 
    }

    cc_masks[seed_b_idx] = (uint8_t*)calloc(voxels_per_block, 1);
    cc_masks[seed_b_idx][seed_v_idx] = 1;

    std::vector<Point3D> q;
    q.reserve(10000); 
    q.push_back({sx, sy, sz});

    size_t head = 0;
    while (head < q.size()) {
        Point3D p = q[head++];
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    int nx_req = p.x + dx, ny_req = p.y + dy, nz_req = p.z + dz;

                    if (nx_req >= 0 && nx_req < req_x && ny_req >= 0 && ny_req < req_y && nz_req >= 0 && nz_req < req_z) {
                        size_t b_idx;
                        int v_idx;
                        get_indices(nx_req, ny_req, nz_req, b_idx, v_idx);

                        if (cc_masks[b_idx] && cc_masks[b_idx][v_idx]) continue;

                        if (blocks[b_idx].has_nonzero) {
                            if (!cached_masks[b_idx]) {
                                cached_masks[b_idx] = (uint8_t*)calloc(voxels_per_block, 1);
                                DecodeBlockToCache<T>(blocks[b_idx], voxels_per_block, cached_masks[b_idx]);
                            }
                            if (cached_masks[b_idx][v_idx]) {
                                if (!cc_masks[b_idx]) cc_masks[b_idx] = (uint8_t*)calloc(voxels_per_block, 1);
                                cc_masks[b_idx][v_idx] = 1;
                                q.push_back({nx_req, ny_req, nz_req});
                            }
                        }
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < total_blocks; ++i) {
        if (cached_masks[i]) free(cached_masks[i]);
    }
    return cc_masks;
}

inline int CompressMaskFast(const uint8_t* mask, int total_voxels, std::vector<uint32_t>& out_bitstream) {
    bool has_0 = false, has_1 = false;
    int words = (total_voxels + 31) / 32;
    out_bitstream.assign(words, 0);

    for (int i = 0; i < total_voxels; ++i) {
        if (mask[i]) { has_1 = true; out_bitstream[i / 32] |= (1U << (i % 32)); }
        else { has_0 = true; }
    }
    if (has_1 && !has_0) return 1;
    if (!has_1) return 0;
    return 2;
}
#endif
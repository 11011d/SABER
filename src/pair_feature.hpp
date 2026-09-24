#ifndef PAIR_FEATURE_HPP
#define PAIR_FEATURE_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "fast_boundary.hpp"

struct PairFeatureRequest {
    const CBlock* blocks;
    int nx;
    int ny;
    int nz;
    int bx;
    int by;
    int bz;
    int req_x;
    int req_y;
    int req_z;
    int q2p_x;
    int q2p_y;
    int q2p_z;
    uint64_t label_one;
    uint64_t label_two;
    uint8_t* output;
};

template <typename T>
inline void ExtractPairFeature3D(
    const CBlock* blocks,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int req_x, int req_y, int req_z,
    int q2p_x, int q2p_y, int q2p_z,
    T label_one, T label_two,
    int out_z, int out_y, int out_x,
    uint8_t* output)
{
    const size_t channel_voxels =
        static_cast<size_t>(out_z) * static_cast<size_t>(out_y) * static_cast<size_t>(out_x);
    std::memset(output, 0, 3 * channel_voxels * sizeof(uint8_t));

    std::vector<int> source_x(out_x);
    std::vector<int> source_y(out_y);
    std::vector<int> source_z(out_z);
    for (int ox = 0; ox < out_x; ++ox) {
        source_x[ox] = static_cast<int>(
            (static_cast<int64_t>(ox) * static_cast<int64_t>(req_x)) / out_x);
    }
    for (int oy = 0; oy < out_y; ++oy) {
        source_y[oy] = static_cast<int>(
            (static_cast<int64_t>(oy) * static_cast<int64_t>(req_y)) / out_y);
    }
    for (int oz = 0; oz < out_z; ++oz) {
        source_z[oz] = static_cast<int>(
            (static_cast<int64_t>(oz) * static_cast<int64_t>(req_z)) / out_z);
    }

    uint8_t* union_channel = output;
    uint8_t* label_one_channel = output + channel_voxels;
    uint8_t* label_two_channel = output + 2 * channel_voxels;

    for (int oz = 0; oz < out_z; ++oz) {
        const int full_z = source_z[oz] + q2p_z;
        if (full_z < 0 || full_z >= nz * bz) {
            continue;
        }
        const int gz = full_z / bz;
        const int local_z = full_z % bz;
        const size_t output_z_offset = static_cast<size_t>(oz) * out_y * out_x;

        for (int oy = 0; oy < out_y; ++oy) {
            const int full_y = source_y[oy] + q2p_y;
            if (full_y < 0 || full_y >= ny * by) {
                continue;
            }
            const int gy = full_y / by;
            const int local_y = full_y % by;
            const size_t output_y_offset = output_z_offset + static_cast<size_t>(oy) * out_x;

            for (int ox = 0; ox < out_x; ++ox) {
                const int full_x = source_x[ox] + q2p_x;
                if (full_x < 0 || full_x >= nx * bx) {
                    continue;
                }
                const int gx = full_x / bx;

                const size_t block_index =
                    static_cast<size_t>(gx) + static_cast<size_t>(gy) * nx +
                    static_cast<size_t>(gz) * nx * ny;
                const CBlock& block = blocks[block_index];
                if (!block.has_nonzero) {
                    continue;
                }

                const int local_x = full_x % bx;
                const int voxel_index = local_x + local_y * bx + local_z * bx * by;
                const T value = ReadBlockValue<T>(block, voxel_index);
                const size_t output_index = output_y_offset + ox;
                bool matched = false;
                if (value == label_one) {
                    label_one_channel[output_index] = 1;
                    matched = true;
                }
                if (value == label_two) {
                    label_two_channel[output_index] = 1;
                    matched = true;
                }
                if (matched) {
                    union_channel[output_index] = 1;
                }
            }
        }
    }
}

template <typename T>
inline void ExtractPairFeatureRequest3D(
    const PairFeatureRequest& request,
    int out_z, int out_y, int out_x)
{
    ExtractPairFeature3D<T>(
        request.blocks,
        request.nx, request.ny, request.nz,
        request.bx, request.by, request.bz,
        request.req_x, request.req_y, request.req_z,
        request.q2p_x, request.q2p_y, request.q2p_z,
        static_cast<T>(request.label_one),
        static_cast<T>(request.label_two),
        out_z, out_y, out_x,
        request.output);
}

template <typename T>
inline void ExtractPairFeaturesBatch3D(
    const PairFeatureRequest* requests,
    size_t request_count,
    int out_z, int out_y, int out_x,
    int parallel)
{
    if (request_count == 0) {
        return;
    }
    const int worker_count = std::min<int>(
        parallel > 0 ? parallel : 1,
        static_cast<int>(request_count));
    if (worker_count <= 1) {
        for (size_t index = 0; index < request_count; ++index) {
            ExtractPairFeatureRequest3D<T>(requests[index], out_z, out_y, out_x);
        }
        return;
    }

    std::atomic<size_t> next_request(0);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (int worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&]() {
            while (true) {
                const size_t index = next_request.fetch_add(1, std::memory_order_relaxed);
                if (index >= request_count) {
                    return;
                }
                ExtractPairFeatureRequest3D<T>(requests[index], out_z, out_y, out_x);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
}

#endif

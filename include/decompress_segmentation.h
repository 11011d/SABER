/**
 * @license LICENSE_JANELIA.txt
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/
 
/*!
 * Decompresses segmentation encoded using the format described at
 * https://github.com/google/neuroglancer/tree/master/src/neuroglancer/sliceview/compressed_segmentation.
 *
 * User must know the block size for their compressed data and the final
 * volume dimensions.
*/

#ifndef DECOMPRESS_SEGMENTATION_H_
#define DECOMPRESS_SEGMENTATION_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace compress_segmentation {

struct Request {
    const uint32_t* encoded_ptr;
    ptrdiff_t volume_size[4];
    ptrdiff_t block_size[3];
    ptrdiff_t chunk_start[3];
    ptrdiff_t chunk_end[3];
    ptrdiff_t intersection_start[3];
    ptrdiff_t intersection_end[3];
    ptrdiff_t request_start[3];
    ptrdiff_t request_end[3];
    ptrdiff_t strides[4];
    void* output_array_ptr;
    int ndim;
    bool is_uint64;
    size_t element_size;
    const char* order; 
};

class ThreadPool {
public:
  ThreadPool(size_t threads) : stop(false) {
    for (size_t i = 0; i < threads; ++i)
      workers.emplace_back(
        [this] {
          while (true) {
            std::function<void()> task;
            {
              std::unique_lock<std::mutex> lock(this->queue_mutex);
              this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
              if (this->stop && this->tasks.empty())
                return;
              task = std::move(this->tasks.front());
              this->tasks.pop();
            }
            task();
          }
        }
      );
    }

  template<class F>
  void enqueue(F&& f) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      tasks.emplace(std::forward<F>(f));
    }
    condition.notify_one();
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for (std::thread &worker: workers)
      worker.join();
  }

private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};
// Decodes a single channel.
//
// Args:
//   input: Pointer to compressed data.
//
//   volume_size: Extent of the x, y, and z dimensions.
//
//   block_size: Extent of the x, y, and z dimensions of the block.
//
//   output: Vector to which output will be appended.
//
//   returns input pointer location
template <class Label>
void DecompressChannel(const uint32_t* input,
                     const ptrdiff_t volume_size[3],
                     const ptrdiff_t block_size[3],
                     const ptrdiff_t strides[4],
                     Label* output,
                     const ptrdiff_t channel);

// Encodes multiple channels.
//
// Each channel is decoded independently.
//
// The output starts with num_channels (=volume_size[3]) uint32 values
// specifying the starting offset of the encoding of each channel (the first
// offset will always equal num_channels).
//
// Args:
//
//   input: Pointer to compressed data.
//
//   volume_size: Extent of the x, y, z, and channel dimensions.
//
//   block_size: Extent of the x, y, and z dimensions of the block.
//
//   output: Vector where output will be appended.
template <class Label>
void DecompressChannels(const uint32_t* input,
                      const ptrdiff_t volume_size[4],
                      const ptrdiff_t block_size[3],
                      const ptrdiff_t strides[4],
                      Label* output);

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
                                   const ptrdiff_t channel);

// Partially decodes multiple channels.
template <class Label>
void DecompressPartialChannelsIntersection(const uint32_t* input,
                                           const ptrdiff_t volume_size[4],
                                           const ptrdiff_t block_size[3],
                                           const ptrdiff_t chunk_start[3],
                                           const ptrdiff_t intersection_start[3],
                                           const ptrdiff_t intersection_end[3],
                                           const ptrdiff_t strides[4],
                                           Label* output);
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
                                                  size_t l2cache_size = 0);
template <class Label>
void DecompressPartialIntersectionInPlace(const uint32_t* input,
                                          const ptrdiff_t volume_size[3],
                                          const ptrdiff_t block_size[3],
                                          const ptrdiff_t chunk_start[3],
                                          const ptrdiff_t intersection_start[3],
                                          const ptrdiff_t intersection_end[3],
                                          const ptrdiff_t request_start[3],
                                          const ptrdiff_t strides[4],       
                                          Label* output);       

void DecompressPartialChannelsIntersectionParallel(std::vector<Request>& requests, int parallel, size_t l2cache_size = 0);
}  // namespace compress_segmentation

#endif  // DECOMPRESS_SEGMENTATION_H_

# cython: language_level=3
"""
Cython binding for the C++ compressed_segmentation
library by Jeremy Maitin-Shepard and Stephen Plaza.

Image label encoding algorithm binding. Compatible with
neuroglancer.

Key methods: compress, decompress

License: BSD 3-Clause

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: July 2018 - March 2022
"""

from libc.stdio cimport FILE, fopen, fwrite, fclose
from libc.stdlib cimport calloc, free
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int64_t
from cpython cimport array
import array
import sys
import operator
from functools import reduce
from libc.stdio cimport printf

from libcpp.vector cimport vector
from libc.string cimport memcpy
import cython
cimport numpy as np
import numpy as np
import ctypes
ctypedef fused UINT:
  uint32_t
  uint64_t

np.import_array()


cdef extern from "Python.h":
    object PyLong_AsVoidPtr(object o)
cdef extern from "compress_segmentation.h" namespace "compress_segmentation":
  cdef int CompressChannels[Label](
    Label* input, 
    const ptrdiff_t input_strides[4],
    const ptrdiff_t volume_size[4],
    const ptrdiff_t block_size[3],
    vector[uint32_t]* output
  )

cdef extern from "decompress_segmentation.h" namespace "compress_segmentation":
  cdef void DecompressChannels[Label](
    const uint32_t* input,
    const ptrdiff_t volume_size[4],
    const ptrdiff_t block_size[3],
    const ptrdiff_t strides[4],
    Label* output
  )
  cdef void DecompressPartialChannelsIntersection[Label](const uint32_t* input,
                                           const ptrdiff_t volume_size[4],
                                           const ptrdiff_t block_size[3],
                                           const ptrdiff_t chunk_start[3],
                                           const ptrdiff_t intersection_start[3],
                                           const ptrdiff_t intersection_end[3],
                                           const ptrdiff_t strides[4],
                                           Label* output) 
  cdef void DecompressPartialChannelsIntersectionInPlace[Label](const uint32_t* input,
                                                  const ptrdiff_t volume_size[4],
                                                  const ptrdiff_t block_size[3],
                                                  const ptrdiff_t chunk_start[3],
                                                  const ptrdiff_t intersection_start[3],
                                                  const ptrdiff_t intersection_end[3],
                                                  const ptrdiff_t request_start[3], 
                                                  const ptrdiff_t strides[4],       
                                                  Label* output,
                                                  size_t l2cache_size)
  ctypedef struct Request:
        const uint32_t* encoded_ptr
        ptrdiff_t volume_size[4]
        ptrdiff_t block_size[3]
        ptrdiff_t chunk_start[3]
        ptrdiff_t chunk_end[3];
        ptrdiff_t intersection_start[3]
        ptrdiff_t intersection_end[3]
        ptrdiff_t request_start[3]
        ptrdiff_t request_end[3]
        ptrdiff_t strides[4]
        void* output_array_ptr
        int ndim
        bint is_uint64
        size_t element_size
        const char* order;

  # 声明新的并行解压函数
  cdef void DecompressPartialChannelsIntersectionParallel(vector[Request]& requests, int parallel, size_t l2cache_size)

cdef extern from "extract_blocks.h" namespace "compress_segmentation":
  cdef cppclass BlockInfo[Label]:
      size_t block_id
      uint8_t encoding_bits
      const Label* palette_ptr
      size_t palette_size
      const uint32_t* bitstream_ptr
      size_t bitstream_word_count

  void ExtractBlockMetadata[Label](
      const uint32_t* input,
      const ptrdiff_t* volume_size,
      const ptrdiff_t* block_size,
      const ptrdiff_t* start,
      const ptrdiff_t* end,
      vector[BlockInfo[Label]]& blocks
  )
  void DecompressSingleBlock[Label](
      const uint32_t* bitstream, 
      const Label* palette, 
      uint8_t bits,
      Label* output_full, 
      ptrdiff_t out_sx, ptrdiff_t out_sy, ptrdiff_t out_sz,
      int off_x, int off_y, int off_z,
      const ptrdiff_t* block_size
  )
  void CreateGenericBinaryBitstream[Label](
      const uint32_t* src_bitstream, 
      const Label* src_palette, 
      uint8_t src_bits,
      Label segid, 
      uint32_t* dst_bitstream, 
      const ptrdiff_t* block_size
  )
  int CheckBlockType[Label](const Label* palette, size_t size, Label segid)
  void CreateGenericBinaryBitstream[Label](
      const uint32_t* src_bitstream, 
      const Label* src_palette, 
      uint8_t src_bits,
      Label segid, 
      uint32_t* dst_bitstream, 
      const ptrdiff_t* block_size
  )

  cdef void CompressSingleBlock[Label](
        const Label* input,
        const ptrdiff_t input_strides[3],
        const ptrdiff_t block_size[3],
        vector[Label]* palette,
        vector[uint32_t]* bitstream,
        uint8_t* bits
    )

  void CreateGenericBinaryBitstreamWithMask[T](
        const uint32_t* src_bitstream, 
        const T* src_palette, 
        uint8_t src_bits,
        T segid, 
        uint32_t* dst_bitstream,
        const ptrdiff_t block_size[3], 
        const ptrdiff_t b_origin[3],
        const ptrdiff_t req_min[3], 
        const ptrdiff_t req_max[3]
    ) nogil

cdef extern from "mask_compress.hpp":
    cdef cppclass MaskedBlockResult[T]:
        vector[T] palette
        uint8_t bits
        vector[uint32_t] bitstream

    MaskedBlockResult[T] MaskAndRecompressBlock[T](
        const T* src_palette, uint8_t src_bits, const uint32_t* src_bitstream,
        const ptrdiff_t blksize[3], const ptrdiff_t b_min[3],
        const ptrdiff_t req_min[3], const ptrdiff_t req_max[3]
    ) nogil


cdef extern from "fast_cc.hpp":
    cdef struct CBlock:
        const uint32_t* bitstream
        const void* palette
        uint8_t bits
        bint has_nonzero

    bint FastNearestSeed[T](
        const CBlock* blocks, int nx, int ny, int nz, int bx, int by, int bz,
        int req_x, int req_y, int req_z, int q2p_x, int q2p_y, int q2p_z,
        int cx, int cy, int cz, int& out_sx, int& out_sy, int& out_sz, 
        bint include_self) nogil

    vector[uint8_t*] SparseBFS26[T](
        const CBlock* blocks, int nx, int ny, int nz, int bx, int by, int bz,
        int req_x, int req_y, int req_z, int q2p_x, int q2p_y, int q2p_z, 
        int sx, int sy, int sz) nogil

    int CompressMaskFast(const uint8_t* mask, int total_voxels, vector[uint32_t]& out_bitstream) nogil

cdef extern from "fast_slab.hpp":
    cdef struct SlabBlock:
        const uint32_t* bitstream
        const void* palette
        uint8_t bits
        bint has_nonzero
        int gx, gy

    void DecompressSlabToBuffer[T](
        const SlabBlock* blocks, size_t num_blocks,
        int bx, int by, int bz,
        int nx_slab, int ny_slab,
        T* buffer) nogil

DEFAULT_BLOCK_SIZE = (8,8,8)

class DecodeError(Exception):
  pass

@cython.binding(True)
def compress(data, block_size=DEFAULT_BLOCK_SIZE, order='C'):
  """
  compress(data, block_size=DEFAULT_BLOCK_SIZE, order='C')

  Compress a uint32 or uint64 3D or 4D numpy array using the
  compressed_segmentation technique.

  data: the numpy array
  block_size: typically (8,8,8). Small enough to be considered
    random access on a GPU, large enough to achieve compression.
  order: 'C' (row-major, 'C', XYZ) or 'F' (column-major, fortran, ZYX)
    memory layout.

  Returns: byte string representing the encoded file
  """
  if len(data.shape) < 4:
    data = data[ :, :, :, np.newaxis ]

  cdef ptrdiff_t volume_size[4] 
  volume_size[:] = data.shape[:4]

  cdef ptrdiff_t block_sizeptr[3]
  block_sizeptr[:] = block_size[:3]

  cdef ptrdiff_t input_strides[3]

  if order == 'F':
    input_strides[:] = [ 
      1,
      volume_size[0],
      volume_size[0] * volume_size[1]
    ]
  else:
    input_strides[:] = [ 
      volume_size[1] * volume_size[2],
      volume_size[2], 
      1
    ]

  cdef uint32_t[:,:,:,:] arr_memview32
  cdef uint64_t[:,:,:,:] arr_memview64

  cdef vector[uint32_t] *output = new vector[uint32_t]()
  cdef int error = 0

  if data.dtype == np.uint32:
    if data.size == 0:
      arr_memview32 = np.zeros((1,1,1,1), dtype=np.uint32)
    else:
      arr_memview32 = data
    error = CompressChannels[uint32_t](
      <uint32_t*>&arr_memview32[0,0,0,0],
      <ptrdiff_t*>input_strides,
      <ptrdiff_t*>volume_size,
      <ptrdiff_t*>block_sizeptr,
      output
    )
  else:
    if data.size == 0:
      arr_memview64 = np.zeros((1,1,1,1), dtype=np.uint64)
    else:
      arr_memview64 = data

    error = CompressChannels[uint64_t](
      <uint64_t*>&arr_memview64[0,0,0,0],
      <ptrdiff_t*>input_strides,
      <ptrdiff_t*>volume_size,
      <ptrdiff_t*>block_sizeptr,
      output
    )

  if error:
    raise OverflowError(
      "The input data were too large and varied and generated a table offset larger than 24-bits.\n"
      "See lookupTableOffset: https://github.com/google/neuroglancer/blob/c9a6b9948dd416997c91e655ec3d67bf6b7e771b/src/neuroglancer/sliceview/compressed_segmentation/README.md#format-specification"
    )

  cdef uint32_t* output_ptr = <uint32_t *>&output[0][0]
  cdef uint32_t[:] vec_view = <uint32_t[:output.size()]>output_ptr

  bytestrout = bytes(vec_view[:])
  del output
  return bytestrout

cdef decompress_helper(
    bytes encoded, volume_size, order, 
    block_size=DEFAULT_BLOCK_SIZE, UINT dummy_dtype = 0
  ):
  
  dtype = np.uint32 if sizeof(UINT) == 4 else np.uint64
  if any([ sz == 0 for sz in volume_size ]):
    return np.zeros(volume_size, dtype=dtype, order=order)
  
  decode_shape = volume_size
  if len(decode_shape) == 3:
    decode_shape = (volume_size[0], volume_size[1], volume_size[2], 1)

  cdef unsigned char *encodedptr = <unsigned char*>encoded
  cdef uint32_t* uintencodedptr = <uint32_t*>encodedptr;
  cdef ptrdiff_t[4] volsize = decode_shape
  cdef ptrdiff_t[3] blksize = block_size
  cdef ptrdiff_t[4] strides = [ 
    1, 
    volsize[0], 
    volsize[0] * volsize[1], 
    volsize[0] * volsize[1] * volsize[2] 
  ]

  if order == 'C':
    strides[0] = volsize[1] * volsize[2] * volsize[3]
    strides[1] = volsize[2] * volsize[3]
    strides[2] = volsize[3]
    strides[3] = 1

  voxels = reduce(operator.mul, volume_size)

  cdef np.ndarray[UINT] output = np.zeros([voxels], dtype=dtype)

  if sizeof(UINT) == 4:
    DecompressChannels[uint32_t](
      uintencodedptr,
      volsize,
      blksize,
      strides,
      <uint32_t*>&output[0]
    )
  else:
    DecompressChannels[uint64_t](
      uintencodedptr,
      volsize,
      blksize,
      strides,
      <uint64_t*>&output[0]
    )

  return output.reshape( volume_size, order=order )

@cython.binding(True)
def decompress(
    bytes encoded, volume_size, dtype, 
    block_size=DEFAULT_BLOCK_SIZE, order='C'
  ):
  """
  decompress(
    bytes encoded, volume_size, dtype, 
    block_size=DEFAULT_BLOCK_SIZE, order='C'
  )

  Decode a compressed_segmentation file into a numpy array.

  encoded: the file as a byte string
  volume_size: tuple with x,y,z dimensions
  dtype: np.uint32 or np.uint64
  block_size: typically (8,8,8), the block size the file was encoded with.
  order: 'C' (XYZ) or 'F' (ZYX)

  Returns: 3D or 4D numpy array
  """
  dtype = np.dtype(dtype)
  if dtype == np.uint32:
    return decompress_helper(encoded, volume_size, order, block_size, <uint32_t>0)
  elif dtype == np.uint64:
    return decompress_helper(encoded, volume_size, order, block_size, <uint64_t>0)
  else:
    raise TypeError("dtype ({}) must be one of uint32 or uint64.".format(dtype))

@cython.binding(True)
def labels(
  bytes encoded, shape, dtype, 
  block_size=DEFAULT_BLOCK_SIZE
):
  """Extract labels without decompressing."""

  if len(encoded) == 0:
    raise DecodeError("Empty data stream.")

  if encoded[0] != 1:
    raise DecodeError("This function only handles single channel images.")

  shape = np.array(shape)
  if any(shape == 0):
    return np.zeros((0,), dtype=dtype)

  index = _compute_label_offsets(encoded, shape, dtype, block_size)
  index = np.unique(index, axis=0)
  cdef size_t num_headers = index.shape[0]

  encoded = encoded[4:] # skip the channel length
  cdef np.ndarray[uint32_t] data = np.frombuffer(encoded, dtype=np.uint32)

  labels = np.concatenate([ 
    data[index[idx,0]:index[idx,1]]
    for idx in range(num_headers) 
  ]).view(dtype)

  return np.unique(labels)

@cython.binding(True)
def remap(
  bytes encoded, shape, dtype, 
  mapping, preserve_missing_labels=False,
  block_size=DEFAULT_BLOCK_SIZE
):
  """Extract labels without decompressing."""

  if len(encoded) == 0:
    raise DecodeError("Empty data stream.")

  if encoded[0] != 1:
    raise DecodeError("This function only handles single channel images.")

  shape = np.array(shape)
  if np.any(shape == 0):
    return encoded

  index = _compute_label_offsets(encoded, shape, dtype, block_size)
  index = np.unique(index, axis=0)
  cdef size_t num_headers = index.shape[0]

  channel_length = encoded[:4]
  cdef np.ndarray[uint32_t] data = np.copy(np.frombuffer(encoded[4:], dtype=np.uint32))

  for idx in range(num_headers):
    labels = data[index[idx,0]:index[idx,1]].view(dtype)

    if preserve_missing_labels:
      labels = np.array([ mapping.get(label, label) for label in labels ], dtype=dtype)
    else:
      labels = np.array([ mapping[label] for label in labels ], dtype=dtype)

    data[index[idx,0]:index[idx,1]] = labels.view(np.uint32)

  return channel_length + data.tobytes()

def _compute_label_offsets(
  bytes encoded, shape, dtype, block_size
) -> np.ndarray:
  shape = np.array(shape)
  block_size = np.array(block_size)

  grid_size = np.ceil(shape / block_size).astype(np.uint64)
  cdef size_t num_headers = reduce(operator.mul, grid_size)
  cdef size_t header_bytes = 8 * num_headers

  encoded = encoded[4:] # skip the channel length
  cdef np.ndarray[uint64_t] headers = np.frombuffer(encoded[:header_bytes], dtype=np.uint64)
  cdef np.ndarray[uint32_t] data = np.frombuffer(encoded, dtype=np.uint32)

  cdef np.ndarray[uint32_t] offsets = np.zeros((2*num_headers,), dtype=np.uint32)

  cdef size_t i = 0
  cdef size_t lookup_table_offset = 0
  cdef size_t encoded_values_offset = 0
  for i in range(num_headers):
    lookup_table_offset = headers[i] & 0xffffff
    encoded_values_offset = headers[i] >> 32
    offsets[2 * i] = lookup_table_offset
    offsets[2 * i + 1] = encoded_values_offset

  # use unique rather than simply sort b/c
  # label offsets can be reused.
  offsets = np.unique(offsets)

  labels = np.zeros((0,), dtype=dtype)

  cdef size_t dtype_bytes = np.dtype(dtype).itemsize
  cdef size_t start = 0
  cdef size_t end = 0

  cdef int64_t idx = 0
  cdef int64_t size = offsets.size - 1

  cdef np.ndarray[uint32_t, ndim=2] index = np.zeros((num_headers, 2), dtype=np.uint32)

  for i in range(num_headers):
    lookup_table_offset = headers[i] & 0xffffff
    idx = _search(offsets, lookup_table_offset)
    if idx == -1:
      raise IndexError(f"Unable to locate value: {lookup_table_offset}")
    elif idx == size:
      index[i, 0] = offsets[idx]
      index[i, 1] = data.size
    else:
      index[i, 0] = offsets[idx]
      index[i, 1] = offsets[idx+1]

  return index

cdef int64_t _search(np.ndarray[uint32_t] offsets, uint32_t value):
  cdef size_t first = 0
  cdef size_t last = offsets.size - 1
  cdef size_t middle = (first // 2 + last // 2)

  while (last - first) > 1:
    if offsets[middle] == value:
      return middle
    elif offsets[middle] > value:
      last = middle
    else:
      first = middle

    middle = (first // 2 + last // 2 + ((first & 0b1) + (last & 0b1)) // 2)

  if offsets[first] == value:
    return first

  if offsets[last] == value:
    return last

  return -1

class CompressedSegmentationArray:
  def __init__(
    self, binary, shape, dtype, block_size=DEFAULT_BLOCK_SIZE
  ):
    self.binary = binary
    self.shape = np.array(shape, dtype=np.int64)
    self.dtype = np.dtype(dtype)
    self.block_size = np.array(block_size, dtype=np.int64)
    self._labels = None

  @property
  def grid_size(self):
    return np.ceil(self.shape / self.block_size).astype(np.int64)

  def labels(self):
    if self._labels is None:
      self._labels = labels(
        self.binary, shape=self.shape,
        dtype=self.dtype, block_size=self.block_size
      )
    return self._labels

  def remap(self, mapping, preserve_missing_labels=False):
    return remap(
      self.binary, self.shape, self.dtype,
      mapping, preserve_missing_labels,
      self.block_size
    )

  def numpy(self):
    return decompress(
      self.binary, self.shape, 
      self.dtype, self.block_size
    )

  def get(self, x,y,z):
    if (
      (x < 0 or x >= self.shape[0])
      or (y < 0 or y >= self.shape[1])
      or (z < 0 or z >= self.shape[2])
    ):
      raise IndexError(f"<{x},{y},{z}> must be contained within {self.shape}")

    xyz = np.array([x,y,z], dtype=np.int64)
    gpt = xyz // self.block_size
    grid_size = self.grid_size

    channel_offset = int.from_bytes(self.binary[:4], 'little')

    if channel_offset != 1:
      raise DecodeError(
        "Only single channel is currently supported in this function."
      )

    binary = self.binary[4:]
    num_headers = grid_size[0] * grid_size[1] * grid_size[2]
    header_idx = gpt[0] + grid_size[0] * (gpt[1] + grid_size[1] * gpt[2])
    
    headers = np.frombuffer(binary[:8*num_headers], dtype=np.uint64)
    data = np.frombuffer(binary, dtype=np.uint32)

    cdef uint64_t header = headers[header_idx]
    cdef uint64_t tbl_off = header & 0xffffff
    cdef uint64_t encoded_bits = (header >> 24) & 0xff
    cdef uint64_t packed_off = header >> 32
    
    pt = xyz % self.block_size

    cdef uint64_t bitpos = encoded_bits * (
      pt[0] + self.block_size[0] * (pt[1] + self.block_size[1] * pt[2])
    )

    cdef uint64_t bitshift = bitpos % 32
    cdef uint64_t arraypos = bitpos // 32
    cdef uint64_t bitmask = (1 << encoded_bits) - 1
    cdef uint64_t bitval = 0
    if encoded_bits > 0:
      bitval = (data[packed_off + arraypos] >> bitshift) & bitmask

    cdef uint64_t table_entry_size = np.dtype(self.dtype).itemsize // 4
    cdef uint64_t val = data[tbl_off + bitval * table_entry_size]
    if table_entry_size > 1:
      val = val | (<uint64_t>(data[tbl_off + bitval * table_entry_size + 1]) << 32)

    return val

  def __contains__(self, val):
    return val in self.labels()

  def __getitem__(self, slcs):
    return self.get(*slcs)



@cython.binding(True)
def decompress_partial(
  bytes encoded, volume_size, dtype,
  chunk_start, chunk_end, request_start, request_end,
  block_size=DEFAULT_BLOCK_SIZE, order='F'
):
  """
  Partially decode a compressed_segmentation file into a numpy array,
  only decompressing the requested sub-region within a chunk.

  encoded: compressed data (bytes)
  volume_size: full 3D (or 4D) volume size
  dtype: np.uint32 or np.uint64
  chunk_start, chunk_end: bounds of the chunk in global coordinates
  request_start, request_end: requested region in global coordinates
  block_size: typically (8,8,8)
  order: only 'C' (XYZ) supported

  Returns: 3D numpy array of the requested intersection region
  """

  dtype = np.dtype(dtype)
  if dtype not in (np.uint32, np.uint64):
    raise TypeError(f"dtype ({dtype}) must be one of uint32 or uint64.")

  for i in range(3):
    if chunk_end[i] - chunk_start[i] != volume_size[i]:
     raise ValueError(f"size of chunk_start[{i}]({chunk_end[i]}) - chunk_end[{i}]({chunk_start[i]}) != volume_size[{i}]({volume_size[i]}).")
  # Compute intersection
  cdef ptrdiff_t intersection_start[3]
  cdef ptrdiff_t intersection_end[3]

  empty = False
  for i in range(3):
    intersection_start[i] = max(chunk_start[i], request_start[i])
    intersection_end[i]   = min(chunk_end[i],   request_end[i])
    if intersection_start[i] >= intersection_end[i]:
      empty = True

  full = True
  for i in range(3):
    if intersection_start[i] != chunk_start[i] or intersection_end[i] != chunk_end[i]:
      full = False
  if full:
    return decompress(encoded, volume_size, dtype, block_size, order), intersection_start, intersection_end

  # If intersection is empty or volume has zeros, return zeros
  if empty or any(sz == 0 for sz in volume_size):
    out_shape = (
      max(0, request_end[0] - request_start[0]),
      max(0, request_end[1] - request_start[1]),
      max(0, request_end[2] - request_start[2]),
    )
    return np.zeros(out_shape, dtype=dtype, order=order), intersection_start, intersection_end

  # Dispatch by dtype
  if dtype == np.uint32:
    return _decompress_partial_helper(
      <uint32_t>0, encoded, volume_size, block_size,
      chunk_start, intersection_start, intersection_end, dtype, order
    ), intersection_start, intersection_end
  else:  # np.uint64
    return _decompress_partial_helper(
      <uint64_t>0, encoded, volume_size, block_size,
      chunk_start, intersection_start, intersection_end, dtype, order
    ), intersection_start, intersection_end


cdef _decompress_partial_helper(
    UINT dummy_dtype,
    bytes encoded, volume_size, block_size,
    chunk_start, intersection_start, intersection_end,
    dtype, order='F'
):
  """
  Internal helper for partial decompression.
  Calls C++ DecompressPartialChannelsIntersection with the right template.
  """
  decode_shape = volume_size
  if len(decode_shape) == 3:
    decode_shape = (volume_size[0], volume_size[1], volume_size[2], 1)

  # Cast inputs
  cdef unsigned char *encodedptr = <unsigned char*>encoded
  cdef uint32_t* uintencodedptr = <uint32_t*>encodedptr
  cdef ptrdiff_t[4] volsize = decode_shape
  cdef ptrdiff_t[3] blksize = block_size
  cdef ptrdiff_t[3] chunk_start_c = chunk_start
  cdef ptrdiff_t[3] intersection_start_c = intersection_start
  cdef ptrdiff_t[3] intersection_end_c   = intersection_end

  # Output shape = intersection region
  cdef ptrdiff_t out_shape[3]
  out_shape[0] = intersection_end_c[0] - intersection_start_c[0]
  out_shape[1] = intersection_end_c[1] - intersection_start_c[1]
  out_shape[2] = intersection_end_c[2] - intersection_start_c[2]
  
  cdef ptrdiff_t[4] strides = [
    out_shape[1] * out_shape[2] * volsize[3],
    out_shape[2] * volsize[3],
    volsize[3],
    1,
  ]
  if order == 'F':
    strides[0] = 1
    strides[1] = out_shape[0]
    strides[2] = out_shape[0] * out_shape[1]
    strides[3] = out_shape[0] * out_shape[1] * out_shape[2]
  voxels = reduce(operator.mul, out_shape)

  cdef np.ndarray[UINT] output = np.zeros([voxels], dtype=dtype)

  if sizeof(UINT) == 4:
    DecompressPartialChannelsIntersection[uint32_t](
      uintencodedptr, volsize, blksize,
      chunk_start_c, intersection_start_c, intersection_end_c,
      strides, <uint32_t*>&output[0]
    )
  else:
    DecompressPartialChannelsIntersection[uint64_t](
      uintencodedptr, volsize, blksize,
      chunk_start_c, intersection_start_c, intersection_end_c,
      strides, <uint64_t*>&output[0]
    )
  if len(volume_size) == 3:
    return output.reshape(
      (out_shape[0], out_shape[1], out_shape[2]), order=order
    )
  else:
    return output.reshape(
      (out_shape[0], out_shape[1], out_shape[2], volume_size[3]), order=order
    )

@cython.binding(True)
def decompress_partial_in_place(
    bytes encoded, volume_size, dtype,
    chunk_start, chunk_end, request_start, request_end,
    output_array,  
    block_size=DEFAULT_BLOCK_SIZE, order='F'
):
  """
  Partially decode a compressed_segmentation file directly into a provided output_array,
  only decompressing the requested sub-region within a chunk.

  encoded: compressed data (bytes)
  volume_size: full 3D (or 4D) volume size
  dtype: np.uint32 or np.uint64 (must match output_array.dtype)
  chunk_start, chunk_end: bounds of the chunk in global coordinates
  request_start, request_end: requested region in global coordinates
  output_array: A pre-allocated 3D or 4D numpy array where the decompressed data will be written.
                Its shape must match (request_end - request_start).
  block_size: typically (8,8,8)
  order: only 'C' (XYZ) supported (must match output_array.order)

  Returns: Tuple of (intersection_start, intersection_end) or raises an error if shapes don't match.
  """

  dtype = np.dtype(dtype)
  if dtype != output_array.dtype:
    raise TypeError(f"dtype ({dtype}) must match output_array.dtype ({output_array.dtype}).")
  if dtype not in (np.uint32, np.uint64):
    raise TypeError(f"dtype ({dtype}) must be one of uint32 or uint64.")
  if output_array.ndim != 3 and output_array.ndim != 4:
    raise TypeError(f"output_array must be 3D or 4D.")

  expected_shape = tuple(max(0, request_end[i] - request_start[i]) for i in range(3))
  if output_array.shape[:3] != expected_shape:
    raise ValueError(f"output_array shape {output_array.shape[:3]} does not match expected shape {expected_shape} derived from request BBOX.")

  for i in range(3):
    if chunk_end[i] - chunk_start[i] != volume_size[i]:
      raise ValueError(f"size of chunk_start[{i}]({chunk_end[i]}) - chunk_end[{i}]({chunk_start[i]}) != volume_size[{i}]({volume_size[i]}).")
  
  # Compute intersection
  cdef ptrdiff_t intersection_start[3]
  cdef ptrdiff_t intersection_end[3]

  empty = False
  for i in range(3):
    intersection_start[i] = max(chunk_start[i], request_start[i])
    intersection_end[i]    = min(chunk_end[i],    request_end[i])
    if intersection_start[i] >= intersection_end[i]:
      empty = True
  if empty or any(sz == 0 for sz in volume_size):
    return 
  
  # Dispatch by dtype
  if dtype ==  np.uint32:
    if  output_array.ndim == 3:
      _decompress_partial_helper_in_place_3D[uint32_t](
        encoded, volume_size, block_size,
        chunk_start, intersection_start, intersection_end, 
        request_start,
        output_array, order
      )
    else:
      _decompress_partial_helper_in_place_4D[uint32_t](
        encoded, volume_size, block_size,
        chunk_start, intersection_start, intersection_end, 
        request_start,
        output_array, order
      )
  else:
    if  output_array.ndim == 3:
      _decompress_partial_helper_in_place_3D[uint64_t](
        encoded, volume_size, block_size,
        chunk_start, intersection_start, intersection_end, 
        request_start,
        output_array, order
      )
    else:
      _decompress_partial_helper_in_place_4D[uint64_t](
        encoded, volume_size, block_size,
        chunk_start, intersection_start, intersection_end, 
        request_start,
        output_array, order
      )
  return 

cdef _decompress_partial_helper_in_place_3D(
    bytes encoded, volume_size, block_size,
    chunk_start, intersection_start, intersection_end,
    request_start,
    np.ndarray[UINT, ndim=3, cast=True] output_array, 
    order='C'
):
  """
  Internal helper for partial decompression (in-place).
  Calls C++ DecompressPartialChannelsIntersectionInPlace with the right template.
  """
  decode_shape = volume_size
  if len(decode_shape) == 3:
    # 扩展到 4D 形状，通道数为 1
    decode_shape = (volume_size[0], volume_size[1], volume_size[2], 1)

  # Cast inputs
  cdef unsigned char *encodedptr = <unsigned char*>encoded
  cdef uint32_t* uintencodedptr = <uint32_t*>encodedptr
  cdef ptrdiff_t[4] volsize = decode_shape
  cdef ptrdiff_t[3] blksize = block_size
  cdef ptrdiff_t[3] chunk_start_c = chunk_start
  cdef ptrdiff_t[3] intersection_start_c = intersection_start
  cdef ptrdiff_t[3] intersection_end_c    = intersection_end
  cdef ptrdiff_t[3] request_start_c = request_start 

  cdef ptrdiff_t element_size = sizeof(UINT)
  cdef ptrdiff_t out_strides[4]
  out_strides[3] = 1

  # 将 NumPy 字节步长转换为元素数量步长
  # 假设 output_array 的维度是 (X, Y, Z, C)
  for i in range(output_array.ndim):
    out_strides[i] = <ptrdiff_t> (output_array.strides[i] / element_size)

  # 获取底层数据指针
  cdef void* out_data_ptr = <void*>&output_array[0, 0, 0]

  # C++ 函数调用
  if sizeof(UINT) == 4:
    DecompressPartialChannelsIntersectionInPlace[uint32_t](
      uintencodedptr, volsize, blksize,
      chunk_start_c, intersection_start_c, intersection_end_c,
      request_start_c, 
      out_strides, <uint32_t*>out_data_ptr, 0
    )
  else:
    DecompressPartialChannelsIntersectionInPlace[uint64_t](
      uintencodedptr, volsize, blksize,
      chunk_start_c, intersection_start_c, intersection_end_c,
      request_start_c,
      out_strides, <uint64_t*>out_data_ptr, 0
    )
  return

cdef _decompress_partial_helper_in_place_4D(
    bytes encoded, volume_size, block_size,
    chunk_start, intersection_start, intersection_end,
    request_start,
    np.ndarray[UINT, ndim=4, cast=True] output_array, 
    order='F'
):
  """
  Internal helper for partial decompression (in-place).
  Calls C++ DecompressPartialChannelsIntersectionInPlace with the right template.
  """
  decode_shape = volume_size
  if len(decode_shape) == 3:
    # 扩展到 4D 形状，通道数为 1
    decode_shape = (volume_size[0], volume_size[1], volume_size[2], 1)

  # Cast inputs
  cdef unsigned char *encodedptr = <unsigned char*>encoded
  cdef uint32_t* uintencodedptr = <uint32_t*>encodedptr
  cdef ptrdiff_t[4] volsize = decode_shape
  cdef ptrdiff_t[3] blksize = block_size
  cdef ptrdiff_t[3] chunk_start_c = chunk_start
  cdef ptrdiff_t[3] intersection_start_c = intersection_start
  cdef ptrdiff_t[3] intersection_end_c = intersection_end
  cdef ptrdiff_t[3] request_start_c = request_start 

  cdef ptrdiff_t element_size = sizeof(UINT)
  cdef ptrdiff_t out_strides[4]
  out_strides[3] = 1

  # 将 NumPy 字节步长转换为元素数量步长
  # 假设 output_array 的维度是 (X, Y, Z, C)
  for i in range(output_array.ndim):
    out_strides[i] = <ptrdiff_t> (output_array.strides[i] / element_size)

  # 获取底层数据指针
  cdef void* out_data_ptr = <void*>&output_array[0, 0, 0, 0]

  # C++ 函数调用
  if sizeof(UINT) == 4:
    DecompressPartialChannelsIntersectionInPlace[uint32_t](
      uintencodedptr, volsize, blksize,
      chunk_start_c, intersection_start_c, intersection_end_c,
      request_start_c, 
      out_strides, <uint32_t*>out_data_ptr, 0
    )
  else:
    DecompressPartialChannelsIntersectionInPlace[uint64_t](
      uintencodedptr, volsize, blksize,
      chunk_start_c, intersection_start_c, intersection_end_c,
      request_start_c,
      out_strides, <uint64_t*>out_data_ptr, 0
    )
  return

@cython.binding(True)
def decompress_partial_in_place_parallel(
    list requests,
    int parallel=1,
    order='C',
    size_t l2cache_size=0
):
  """
  Partially decode a list of compressed_segmentation files directly into provided output arrays,
  decompressing the requested sub-regions within chunks in parallel.

  requests: A list of dictionaries, where each dictionary contains the parameters
            for a single decompression request:
            {
                'encoded': bytes,
                'volume_size': tuple,
                'dtype': np.uint32 or np.uint64,
                'chunk_start': tuple,
                'chunk_end': tuple,
                'request_start': tuple,
                'request_end': tuple,
                'block_size': tuple,
                'output_array_ptr': Py_ssize_t,
                'output_array_ndim': ptrdiff_t,
                'output_array_shape': tuple,
                'output_array_strides': tuple,
            }
  parallel: The number of parallel threads to use. Defaults to 1 (sequential).
  """
  if parallel < 1:
      raise ValueError("parallel must be a positive integer.")
  cdef unsigned char *encodedptr
  cdef vector[Request] c_requests
  cdef Request c_req
  cdef bytes order_b = order.encode('ascii')
  cdef const char* order_c = order_b
  cdef Py_ssize_t output_array_ptr_value
  cdef int output_array_ndim
  cdef np.dtype dtype
  cdef tuple output_array_strides
  for req in requests:
      # 类型和形状校验
      dtype = np.dtype(req['dtype'])
      if dtype not in (np.uint32, np.uint64):
          raise TypeError(f"dtype ({dtype}) must be one of uint32 or uint64.")
      output_array_ndim = req['output_array_ndim']
      if output_array_ndim != 3 and output_array_ndim != 4:
          raise TypeError(f"output_array must be 3D or 4D.")
      expected_shape = tuple(max(0, req['request_end'][i] - req['request_start'][i]) for i in range(3))
      output_array_shape = req['output_array_shape']
      if output_array_shape[:3] != expected_shape:
          raise ValueError(f"output_array shape {output_array_shape[:3]} does not match expected shape {expected_shape} derived from request BBOX.")
      encodedptr = <unsigned char*>req['encoded']
      c_req.encoded_ptr = <uint32_t*>encodedptr;
      c_req.volume_size = req['volume_size']
      c_req.chunk_start = req['chunk_start']
      c_req.chunk_end = req['chunk_end']
      c_req.request_start = req['request_start']
      c_req.request_end = req['request_end']
      
      output_array_ptr_value = req['output_array_ptr']
      c_req.output_array_ptr = <void*>output_array_ptr_value
      # printf("DsssssEBUG (Cy): c_req.output_array_ptr address = %p\n", c_req.output_array_ptr)
  
      c_req.block_size = req['block_size']
      c_req.order = order_c
      c_req.ndim = output_array_ndim
      c_req.is_uint64 = (dtype == np.uint64)
      c_req.element_size = dtype.itemsize
      output_array_strides = req['output_array_strides']
      for i in range(output_array_ndim):
          c_req.strides[i] = <ptrdiff_t> (output_array_strides[i] / c_req.element_size)

      c_requests.push_back(c_req)

  # 调用C++并行解压函数
  _decompress_parallel_helper(c_requests, parallel, l2cache_size)

cdef _decompress_parallel_helper(vector[Request]& requests, int parallel, size_t l2cache_size):
    """
    Internal helper to dispatch requests to C++ for parallel processing.
    """
    DecompressPartialChannelsIntersectionParallel(requests, parallel, l2cache_size)
    return


# --- 1. 针对 uint32 的分发实现 ---
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fill_container_u32_global(
    uint32_t* data, ptrdiff_t[3] volsize, ptrdiff_t[3] blksize, 
    ptrdiff_t[3] local_start, ptrdiff_t[3] local_end, 
    ptrdiff_t[3] rel_grid, BlockArena* arena, int arena_nx, int arena_nxy, object segid_list
):
    cdef vector[BlockInfo[uint32_t]] blocks
    ExtractBlockMetadata[uint32_t](data, volsize, blksize, local_start, local_end, blocks)
    
    cdef size_t nx = (volsize[0] + blksize[0] - 1) // blksize[0]
    cdef size_t ny = (volsize[1] + blksize[1] - 1) // blksize[1]
    
    cdef size_t i, j, k
    cdef vector[uint32_t] c_segs
    cdef bint has_segid
    cdef bint check_segs = False
    
    if segid_list is not None:
        check_segs = True
        for s in segid_list:
            c_segs.push_back(s)

    for i in range(blocks.size()):
        if check_segs:
            has_segid = False
            for j in range(blocks[i].palette_size):
                for k in range(c_segs.size()):
                    if blocks[i].palette_ptr[j] == c_segs[k]:
                        has_segid = True
                        break
                if has_segid:
                    break
            if not has_segid:
                continue

        # 用 nogil 直接转入原生 C++ 代码
        with nogil:
            ProcessSingleBlockGlobal[uint32_t](
                blocks[i], rel_grid, nx, ny, local_start, local_end, blksize, arena, arena_nx, arena_nxy, 4
            )

# --- 2. 针对 uint64 的分发实现 ---
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fill_container_u64_global(
    uint64_t* data, ptrdiff_t[3] volsize, ptrdiff_t[3] blksize, 
    ptrdiff_t[3] local_start, ptrdiff_t[3] local_end, 
    ptrdiff_t[3] rel_grid, BlockArena* arena, int arena_nx, int arena_nxy, object segid_list
):
    cdef vector[BlockInfo[uint64_t]] blocks
    # 强制转换指针类型以匹配 C++ 签名 (依赖于你的 ExtractBlockMetadata 具体实现)
    ExtractBlockMetadata[uint64_t](<const uint32_t*>data, volsize, blksize, local_start, local_end, blocks)
    
    cdef size_t nx = (volsize[0] + blksize[0] - 1) // blksize[0]
    cdef size_t ny = (volsize[1] + blksize[1] - 1) // blksize[1]
    
    cdef size_t i, j, k
    cdef vector[uint64_t] c_segs
    cdef bint has_segid
    cdef bint check_segs = False
    
    if segid_list is not None:
        check_segs = True
        for s in segid_list:
            c_segs.push_back(s)

    for i in range(blocks.size()):
        if check_segs:
            has_segid = False
            for j in range(blocks[i].palette_size):
                for k in range(c_segs.size()):
                    if blocks[i].palette_ptr[j] == c_segs[k]:
                        has_segid = True
                        break
                if has_segid:
                    break
            if not has_segid:
                continue

        # 用 nogil 直接转入原生 C++ 代码
        with nogil:
            ProcessSingleBlockGlobal[uint64_t](
                blocks[i], rel_grid, nx, ny, local_start, local_end, blksize, arena, arena_nx, arena_nxy, 8
            )

# --- 3. 核心入口函数：将 chunk 粒度数据解压并裁剪后填入 container ---
@cython.binding(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def extract_to_container(
    bytes encoded, volume_size, object dtype,
    chunk_start, chunk_end, request_start, request_end,
    object container, 
    chunk_relative_grid, # 例如 [0, 8, 0]
    block_size=(8, 8, 8), # 替换为你的 DEFAULT_BLOCK_SIZE
    segid_list=None
):
    dtype = np.dtype(dtype)
    cdef ptrdiff_t intersection_start[3]
    cdef ptrdiff_t intersection_end[3]
    cdef int i
    
    # 1. 计算 Chunk 与 Request BBox 的全局物理坐标交集
    for i in range(3):
        intersection_start[i] = max(chunk_start[i], request_start[i])
        intersection_end[i]   = min(chunk_end[i],   request_end[i])
        # 如果当前 Chunk 和请求区域在某个维度上毫无交集，直接跳过整个 Chunk
        if intersection_start[i] >= intersection_end[i]:
            return 

    # 解析编码数据的头部偏移量
    cdef uint32_t* uintencodedptr = <uint32_t*>((<unsigned char*>encoded))
    cdef uint32_t channel_data_offset = uintencodedptr[0]
    cdef uint32_t* channel_data = uintencodedptr + channel_data_offset

    cdef ptrdiff_t[3] volsize = [volume_size[0], volume_size[1], volume_size[2]]
    cdef ptrdiff_t[3] blksize = [block_size[0], block_size[1], block_size[2]]
    cdef ptrdiff_t[3] rel_grid = [chunk_relative_grid[0], chunk_relative_grid[1], chunk_relative_grid[2]]
    
    # 提取 PyBlockStore 底层的 BlockArena 指针和网格信息
    # 假设 container 是 Python 侧使用 CompressedVoxelContainer 封装的，其有个 self.blocks （PyBlockStore）
    # PyBlockStore 内部有个 _arena 指针。我们可以用 Cython 强制转型。
    cdef PyBlockStore pystore = <PyBlockStore>(container.blocks)
    cdef BlockArena* arena = pystore._arena
    cdef int arena_nx = container.grid_size[0]
    cdef int arena_nxy = container.grid_size[0] * container.grid_size[1]
    
    # 2. 将全局交集坐标映射为 Chunk 内部的相对坐标 (局部 BBox)
    cdef ptrdiff_t local_start[3]
    cdef ptrdiff_t local_end[3]
    for i in range(3):
        local_start[i] = intersection_start[i] - chunk_start[i]
        local_end[i]   = intersection_end[i]   - chunk_start[i]

    # 3. 按照数据类型分发给底层引擎
    if dtype == np.uint32:
        _fill_container_u32_global(
            channel_data, volsize, blksize, local_start, local_end, rel_grid, arena, arena_nx, arena_nxy, segid_list
        )
    else:
        _fill_container_u64_global(
            <uint64_t*>channel_data, volsize, blksize, local_start, local_end, rel_grid, arena, arena_nx, arena_nxy, segid_list
        )



@cython.boundscheck(False)
@cython.wraparound(False)
def decompress_block_grid(list block_list, tuple block_size, tuple grid_dims, dtype):
    cdef int nx_b = grid_dims[0]
    cdef int ny_b = grid_dims[1]
    
    cdef np.ndarray out = np.zeros(
        (nx_b * block_size[0], ny_b * block_size[1], grid_dims[2] * block_size[2]), 
        dtype=dtype, order='F'
    )
    
    cdef ptrdiff_t[3] blk_sz = [block_size[0], block_size[1], block_size[2]]
    cdef ptrdiff_t out_sx = 1
    cdef ptrdiff_t out_sy = out.shape[0]
    cdef ptrdiff_t out_sz = out.shape[0] * out.shape[1]

    cdef size_t i, ix_b, iy_b, iz_b
    cdef uint32_t[:] bstream_view
    cdef const uint32_t* b_ptr
    
    # 增加 uint8 视图
    cdef uint8_t[:] pal8
    cdef uint32_t[:] pal32
    cdef uint64_t[:] pal64
    cdef dict b_dict
    cdef object target_dtype = np.dtype(dtype)

    for i in range(len(block_list)):
        b_entry = block_list[i]
        if b_entry is None: continue
        
        b_dict = <dict>b_entry
        ix_b = i % nx_b
        iy_b = (i // nx_b) % ny_b
        iz_b = i // (nx_b * ny_b)

        if b_dict['bits'] > 0:
            bstream_view = b_dict['bitstream']
            b_ptr = &bstream_view[0]
        else:
            b_ptr = NULL
        
        # 分配前必须强制转换 dtype，否则引发 Buffer mismatch
        if target_dtype == np.uint8:
            pal8 = np.ascontiguousarray(b_dict['palette'], dtype=np.uint8)
            DecompressSingleBlock[uint8_t](
                b_ptr, &pal8[0], b_dict['bits'],
                <uint8_t*>out.data, out_sx, out_sy, out_sz,
                ix_b * blk_sz[0], iy_b * blk_sz[1], iz_b * blk_sz[2], blk_sz
            )
        elif target_dtype == np.uint32:
            pal32 = np.ascontiguousarray(b_dict['palette'], dtype=np.uint32)
            DecompressSingleBlock[uint32_t](
                b_ptr, &pal32[0], b_dict['bits'],
                <uint32_t*>out.data, out_sx, out_sy, out_sz,
                ix_b * blk_sz[0], iy_b * blk_sz[1], iz_b * blk_sz[2], blk_sz
            )
        else:
            pal64 = np.ascontiguousarray(b_dict['palette'], dtype=np.uint64)
            DecompressSingleBlock[uint64_t](
                b_ptr, &pal64[0], b_dict['bits'],
                <uint64_t*>out.data, out_sx, out_sy, out_sz,
                ix_b * blk_sz[0], iy_b * blk_sz[1], iz_b * blk_sz[2], blk_sz
            )
            
    return out



cdef inline int check_pal_u8(const uint8_t* pal, size_t size, uint8_t segid) nogil:
    cdef bint ht = False, hf = False
    cdef size_t i
    for i in range(size):
        if pal[i] == segid: ht = True
        else: hf = True
        if ht and hf: return 2
    return 1 if ht else 0

cdef inline int check_pal_u32(const uint32_t* pal, size_t size, uint32_t segid) nogil:
    cdef bint ht = False, hf = False
    cdef size_t i
    for i in range(size):
        if pal[i] == segid: ht = True
        else: hf = True
        if ht and hf: return 2
    return 1 if ht else 0

cdef inline int check_pal_u64(const uint64_t* pal, size_t size, uint64_t segid) nogil:
    cdef bint ht = False, hf = False
    cdef size_t i
    for i in range(size):
        if pal[i] == segid: ht = True
        else: hf = True
        if ht and hf: return 2
    return 1 if ht else 0


@cython.boundscheck(False)
@cython.wraparound(False)
def transform_where_compressed(list block_list, dtype, size_t segid, 
                              object true_val, object false_val, 
                              object out_dtype, tuple block_size):
    
    cdef list new_blocks = []
    cdef ptrdiff_t[3] blk_sz = [block_size[0], block_size[1], block_size[2]]
    cdef int voxels_per_block = blk_sz[0] * blk_sz[1] * blk_sz[2]
    cdef int words_needed = (voxels_per_block + 31) // 32

    # ==========================================================
    # 神经元数据是稀疏的，预先分配常数块，消除昂贵的python调用
    # ==========================================================
    cdef np.ndarray pal_true = np.array([true_val], dtype=out_dtype)
    cdef np.ndarray pal_false = np.array([false_val], dtype=out_dtype)
    cdef np.ndarray pal_mixed = np.array([false_val, true_val], dtype=out_dtype)
    
    cdef dict MISS_BLOCK = {"palette": pal_false, "bits": 0, "bitstream": None}
    cdef dict HIT_BLOCK = {"palette": pal_true, "bits": 0, "bitstream": None}

    cdef dict b_dict
    cdef np.ndarray pal_arr
    cdef np.ndarray bitstream_arr
    cdef object bitstream_obj
    cdef size_t pal_size
    cdef int itemsize
    cdef int b_type
    cdef const uint32_t* b_ptr = NULL

    for b_entry in block_list:
        if b_entry is None:
            new_blocks.append(None)
            continue

        b_dict = <dict>b_entry
        pal_arr = b_dict['palette']
        pal_size = pal_arr.size
        itemsize = pal_arr.itemsize  # 瞬间获取底层类型大小，不掉用昂贵的 Numpy 方法
        # ==========================================================
        # 零拷贝：直接根据 itemsize 强制取底层指针
        # 如果寻找的 ID 比调色盘类型能存的最大值还大，直接判 Miss
        # ==========================================================
        if itemsize == 1:
            if segid > 255: 
                b_type = 0
            else: 
                b_type = CheckBlockType[uint8_t](<uint8_t*>np.PyArray_DATA(pal_arr), pal_size, <uint8_t>segid)
        elif itemsize == 4:
            if segid > 4294967295: 
                b_type = 0
            else: 
                b_type = CheckBlockType[uint32_t](<uint32_t*>np.PyArray_DATA(pal_arr), pal_size, <uint32_t>segid)
        else:
            b_type = CheckBlockType[uint64_t](<uint64_t*>np.PyArray_DATA(pal_arr), pal_size, <uint64_t>segid)

        # 逻辑分发：追加全局常量的引用，内存开销为 0！
        if b_type == 0:
            new_blocks.append(MISS_BLOCK)
        elif b_type == 1:
            new_blocks.append(HIT_BLOCK)
        else:
            # 只有这极少数的混合块，才付出分配内存的代价
            new_bitstream_arr = np.empty(words_needed, dtype=np.uint32)
            
            if b_dict['bits'] > 0:
                bitstream_obj = b_dict['bitstream']
                if type(bitstream_obj) is np.ndarray:
                    b_ptr = <uint32_t*>np.PyArray_DATA(bitstream_obj)
                else:
                    bitstream_arr = np.frombuffer(bitstream_obj, dtype=np.uint32)
                    b_ptr = <uint32_t*>np.PyArray_DATA(bitstream_arr)
            else:
                b_ptr = NULL
            
            # 位流映射
            if itemsize == 1:
                CreateGenericBinaryBitstream[uint8_t](
                    b_ptr, <uint8_t*>np.PyArray_DATA(pal_arr), b_dict['bits'], 
                    <uint8_t>segid, <uint32_t*>np.PyArray_DATA(new_bitstream_arr), blk_sz
                )
            elif itemsize == 4:
                CreateGenericBinaryBitstream[uint32_t](
                    b_ptr, <uint32_t*>np.PyArray_DATA(pal_arr), b_dict['bits'], 
                    <uint32_t>segid, <uint32_t*>np.PyArray_DATA(new_bitstream_arr), blk_sz
                )
            else:
                CreateGenericBinaryBitstream[uint64_t](
                    b_ptr, <uint64_t*>np.PyArray_DATA(pal_arr), b_dict['bits'], 
                    <uint64_t>segid, <uint32_t*>np.PyArray_DATA(new_bitstream_arr), blk_sz
                )

            new_blocks.append({
                "palette": pal_mixed,
                "bits": 1,
                "bitstream": new_bitstream_arr
            })
            
    return new_blocks


@cython.boundscheck(False)
@cython.wraparound(False)
def compress_single_block(np.ndarray dense_data):
    """
    纯净的单块压缩包装器：直接调用 C++ CompressSingleBlock
    返回: (bitstream_arr, palette_arr, bits)
    """
    cdef ptrdiff_t blk_sz[3]
    cdef ptrdiff_t strides[3]
    for i in range(3):
        blk_sz[i] = dense_data.shape[i]
        strides[i] = dense_data.strides[i] // dense_data.itemsize
        
    cdef vector[uint32_t] bitstream_vec
    cdef uint8_t bits = 0
    cdef object target_dtype = dense_data.dtype
    
    # 预留不同类型的 Palette Vector
    cdef vector[uint8_t] pal_8
    cdef vector[uint32_t] pal_32
    cdef vector[uint64_t] pal_64
    
    # 根据类型调用 C++ 模板
    if target_dtype == np.uint8:
        CompressSingleBlock[uint8_t](
            <uint8_t*>dense_data.data, strides, blk_sz, 
            &pal_8, &bitstream_vec, &bits)
    elif target_dtype == np.uint32:
        CompressSingleBlock[uint32_t](
            <uint32_t*>dense_data.data, strides, blk_sz, 
            &pal_32, &bitstream_vec, &bits)
    elif target_dtype == np.uint64:
        CompressSingleBlock[uint64_t](
            <uint64_t*>dense_data.data, strides, blk_sz, 
            &pal_64, &bitstream_vec, &bits)
    else:
        raise ValueError(f"Unsupported dtype: {target_dtype}")

    # --- 将 C++ vector 高效且安全地转换为 Numpy 数组 ---
    cdef np.ndarray bitstream_arr = np.zeros(bitstream_vec.size(), dtype=np.uint32)
    if bitstream_vec.size() > 0:
        memcpy(np.PyArray_DATA(bitstream_arr), bitstream_vec.data(), bitstream_vec.size() * sizeof(uint32_t))
        
    cdef np.ndarray pal_arr
    if target_dtype == np.uint8:
        pal_arr = np.zeros(pal_8.size(), dtype=np.uint8)
        if pal_8.size() > 0:
            memcpy(np.PyArray_DATA(pal_arr), pal_8.data(), pal_8.size() * sizeof(uint8_t))
    elif target_dtype == np.uint32:
        pal_arr = np.zeros(pal_32.size(), dtype=np.uint32)
        if pal_32.size() > 0:
            memcpy(np.PyArray_DATA(pal_arr), pal_32.data(), pal_32.size() * sizeof(uint32_t))
    else:
        pal_arr = np.zeros(pal_64.size(), dtype=np.uint64)
        if pal_64.size() > 0:
            memcpy(np.PyArray_DATA(pal_arr), pal_64.data(), pal_64.size() * sizeof(uint64_t))
            
    return bitstream_arr, pal_arr, bits


cdef inline void _build_c_blocks(list block_list, vector[CBlock]& c_blocks, int itemsize):
    cdef int i, p_idx
    cdef bint has_nonzero # 💥 正确的变量名
    cdef dict b_dict
    cdef np.ndarray pal_arr
    cdef uint8_t[:] pal_8
    cdef uint32_t[:] pal_32
    cdef uint64_t[:] pal_64

    for i in range(len(block_list)):
        if block_list[i] is None:
            c_blocks[i].has_nonzero = False
            continue

        b_dict = <dict>block_list[i]
        pal_arr = b_dict['palette']
        has_nonzero = False
        
        # 极速内存视图检查：找非零值！
        if itemsize == 1:
            pal_8 = pal_arr
            for p_idx in range(pal_8.shape[0]):
                if pal_8[p_idx] != 0:  # 💥 注意这里是 != 0
                    has_nonzero = True
                    break 
        elif itemsize == 4:
            pal_32 = pal_arr
            for p_idx in range(pal_32.shape[0]):
                if pal_32[p_idx] != 0: # 💥 注意这里是 != 0
                    has_nonzero = True
                    break 
        else:
            pal_64 = pal_arr
            for p_idx in range(pal_64.shape[0]):
                if pal_64[p_idx] != 0: # 💥 注意这里是 != 0
                    has_nonzero = True
                    break 

        c_blocks[i].has_nonzero = has_nonzero
        if has_nonzero:
            c_blocks[i].bits = b_dict['bits']
            c_blocks[i].palette = pal_arr.data
            if c_blocks[i].bits > 0:
                c_blocks[i].bitstream = <uint32_t*>(<np.ndarray>b_dict['bitstream']).data
            else:
                c_blocks[i].bitstream = NULL




@cython.boundscheck(False)
@cython.wraparound(False)
def find_nearest_seed_fast(
    list block_list, tuple grid_size, tuple block_size, tuple req_size, 
    tuple q2p_offset, tuple center, object out_dtype, 
    bint include_self=False
):
    cdef int nx_b = grid_size[0], ny_b = grid_size[1], nz_b = grid_size[2]
    cdef int bx = block_size[0], by = block_size[1], bz = block_size[2]
    cdef int req_x = req_size[0], req_y = req_size[1], req_z = req_size[2]
    cdef int q2p_x = q2p_offset[0], q2p_y = q2p_offset[1], q2p_z = q2p_offset[2]
    cdef int cx = center[0], cy = center[1], cz = center[2]
    
    cdef vector[CBlock] c_blocks
    c_blocks.resize(nx_b * ny_b * nz_b)
    cdef int itemsize = np.dtype(out_dtype).itemsize
    _build_c_blocks(block_list, c_blocks, itemsize)

    cdef int seed_x = -1, seed_y = -1, seed_z = -1
    cdef bint found = False

    with nogil:
        # 💥 将 include_self 透传给 C++
        if itemsize == 1:
            found = FastNearestSeed[uint8_t](c_blocks.data(), nx_b, ny_b, nz_b, bx, by, bz, req_x, req_y, req_z, q2p_x, q2p_y, q2p_z, cx, cy, cz, seed_x, seed_y, seed_z, include_self)
        elif itemsize == 4:
            found = FastNearestSeed[uint32_t](c_blocks.data(), nx_b, ny_b, nz_b, bx, by, bz, req_x, req_y, req_z, q2p_x, q2p_y, q2p_z, cx, cy, cz, seed_x, seed_y, seed_z, include_self)
        else:
            found = FastNearestSeed[uint64_t](c_blocks.data(), nx_b, ny_b, nz_b, bx, by, bz, req_x, req_y, req_z, q2p_x, q2p_y, q2p_z, cx, cy, cz, seed_x, seed_y, seed_z, include_self)

    if found: return np.array([seed_x, seed_y, seed_z], dtype=int)
    return None

@cython.boundscheck(False)
@cython.wraparound(False)
def extract_cc_fast(
    list block_list, tuple grid_size, tuple block_size, tuple req_size, 
    tuple q2p_offset, tuple seed_point, object out_dtype
):
    cdef int nx_b = grid_size[0], ny_b = grid_size[1], nz_b = grid_size[2]
    cdef int bx = block_size[0], by = block_size[1], bz = block_size[2]
    cdef int req_x = req_size[0], req_y = req_size[1], req_z = req_size[2]
    cdef int q2p_x = q2p_offset[0], q2p_y = q2p_offset[1], q2p_z = q2p_offset[2]
    cdef int sx = seed_point[0], sy = seed_point[1], sz = seed_point[2]
    cdef int total_blocks = nx_b * ny_b * nz_b

    cdef vector[CBlock] c_blocks
    c_blocks.resize(total_blocks)
    cdef int itemsize = np.dtype(out_dtype).itemsize
    _build_c_blocks(block_list, c_blocks, itemsize)

    cdef vector[uint8_t*] cc_masks
    with nogil:
        if itemsize == 1:
            cc_masks = SparseBFS26[uint8_t](c_blocks.data(), nx_b, ny_b, nz_b, bx, by, bz, req_x, req_y, req_z, q2p_x, q2p_y, q2p_z, sx, sy, sz)
        elif itemsize == 4:
            cc_masks = SparseBFS26[uint32_t](c_blocks.data(), nx_b, ny_b, nz_b, bx, by, bz, req_x, req_y, req_z, q2p_x, q2p_y, q2p_z, sx, sy, sz)
        else:
            cc_masks = SparseBFS26[uint64_t](c_blocks.data(), nx_b, ny_b, nz_b, bx, by, bz, req_x, req_y, req_z, q2p_x, q2p_y, q2p_z, sx, sy, sz)

    cdef np.ndarray pal_false = np.array([0], dtype=out_dtype)
    cdef np.ndarray pal_true = np.array([255], dtype=out_dtype)
    cdef np.ndarray pal_mixed = np.array([0, 255], dtype=out_dtype)
    cdef dict MISS_BLOCK = {"palette": pal_false, "bits": 0, "bitstream": None}
    cdef dict HIT_BLOCK = {"palette": pal_true, "bits": 0, "bitstream": None}
    
    cdef int compress_status
    cdef vector[uint32_t] new_bitstream_vec
    cdef np.ndarray new_bitstream_arr

    for i in range(total_blocks):
        if cc_masks[i] == NULL:
            block_list[i] = MISS_BLOCK
        else:
            with nogil:
                compress_status = CompressMaskFast(cc_masks[i], bx * by * bz, new_bitstream_vec)
                free(cc_masks[i])
            
            if compress_status == 0: block_list[i] = MISS_BLOCK
            elif compress_status == 1: block_list[i] = HIT_BLOCK
            else:
                new_bitstream_arr = np.zeros(new_bitstream_vec.size(), dtype=np.uint32)
                memcpy(np.PyArray_DATA(new_bitstream_arr), new_bitstream_vec.data(), new_bitstream_vec.size() * 4)
                block_list[i] = {"palette": pal_mixed, "bits": 1, "bitstream": new_bitstream_arr}

    return block_list








@cython.boundscheck(False)
@cython.wraparound(False)
def fill_slab_buffer_c(list py_blocks, np.ndarray buffer, tuple block_size, tuple slab_grid_size, object dtype):
    cdef int bx = block_size[0], by = block_size[1], bz = block_size[2]
    cdef int nx_slab = slab_grid_size[0], ny_slab = slab_grid_size[1]
    cdef int num_blocks = len(py_blocks)
    
    cdef vector[SlabBlock] c_blocks
    c_blocks.resize(num_blocks)
    
    cdef dict b_dict
    cdef int i
    cdef bint block_has_nonzero
    cdef np.ndarray pal_arr
    cdef int itemsize = np.dtype(dtype).itemsize

    # 1. 解构数据
    for i in range(num_blocks):
        item = py_blocks[i]
        if item is None:
            c_blocks[i].has_nonzero = False
            continue
            
        b_dict = <dict>item
        pal_arr = b_dict['palette']
        
        block_has_nonzero = False
        if itemsize == 1:
            for p_val in (<uint8_t*>pal_arr.data)[:pal_arr.size]:
                if p_val != 0: block_has_nonzero = True; break
        elif itemsize == 4:
            for p_val in (<uint32_t*>pal_arr.data)[:pal_arr.size]:
                if p_val != 0: block_has_nonzero = True; break
        else:
            for p_val in (<uint64_t*>pal_arr.data)[:pal_arr.size]:
                if p_val != 0: block_has_nonzero = True; break
        
        c_blocks[i].has_nonzero = block_has_nonzero
        if block_has_nonzero:
            c_blocks[i].bits = b_dict['bits']
            c_blocks[i].palette = pal_arr.data
            c_blocks[i].gx = i % nx_slab
            c_blocks[i].gy = i // nx_slab
            if c_blocks[i].bits > 0:
                c_blocks[i].bitstream = <uint32_t*>(<np.ndarray>b_dict['bitstream']).data
            else:
                c_blocks[i].bitstream = NULL

    # 2. 调用 C++ (确保传 8 个参数)
    with nogil:
        if itemsize == 1:
            DecompressSlabToBuffer[uint8_t](c_blocks.data(), num_blocks, bx, by, bz, nx_slab, ny_slab, <uint8_t*>buffer.data)
        elif itemsize == 4:
            DecompressSlabToBuffer[uint32_t](c_blocks.data(), num_blocks, bx, by, bz, nx_slab, ny_slab, <uint32_t*>buffer.data)
        else:
            DecompressSlabToBuffer[uint64_t](c_blocks.data(), num_blocks, bx, by, bz, nx_slab, ny_slab, <uint64_t*>buffer.data)

# ============================================================
#  BlockArena C++ wrapper (via block_arena.hpp)
# ============================================================
from libc.string cimport memset
cdef extern from "block_arena.hpp":

    cdef cppclass BlockArena:
        int total
        int itemsize
        BlockArena(int n, int item_bytes) except +
        void set_block(int idx, const uint8_t* pal, int pal_bytes,
                       uint8_t bits_val, const uint32_t* bs, int bs_words,
                       bint hnz) nogil
        void set_null(int idx) nogil
        void set_all_false(const uint8_t* false_bytes, int false_nbytes) nogil
        const CBlock* get_cblocks() nogil
        int     get_bits(int idx) nogil
        bint    get_is_null(int idx) nogil
        bint    get_has_nonzero(int idx) nogil
        int     get_pal_byte_size(int idx) nogil
        int     get_bs_size(int idx) nogil
        const uint8_t*  get_pal_bytes(int idx) nogil
        const uint32_t* get_bs_data(int idx) nogil

    void ProcessSingleBlockGlobal[UINT](
        const BlockInfo[UINT]& b,
        const ptrdiff_t rel_grid[3],
        size_t nx, size_t ny,
        const ptrdiff_t local_start[3],
        const ptrdiff_t local_end[3],
        const ptrdiff_t blksize[3],
        BlockArena* arena,
        int arena_nx, int arena_nxy, int itemsize) nogil


# ============================================================
#  PyBlockStore：Cython 扩展类型，封装 BlockArena*
# ============================================================
cdef class PyBlockStore:
    """
    将所有压缩 block 数据直接存储在 C++ 管理的内存中，
    持久维护 CBlock 指针数组，消除每次 C++ 调用前的 Python 层转换。
    """
    cdef BlockArena* _arena
    cdef object _dtype_obj
    cdef int _itemsize

    def __cinit__(self, int total, object dtype):
        self._dtype_obj = np.dtype(dtype)
        self._itemsize  = self._dtype_obj.itemsize
        self._arena = new BlockArena(total, self._itemsize)

    def __dealloc__(self):
        if self._arena != NULL:
            del self._arena
            self._arena = NULL

    @property
    def total(self):
        return self._arena.total

    @property
    def dtype(self):
        return self._dtype_obj

    def __len__(self):
        return self._arena.total

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def set_block(self, int idx, np.ndarray palette, uint8_t bits, object bitstream):
        """从 NumPy 数组设置一个 Block。"""
        cdef np.ndarray bs_arr
        cdef const uint32_t* bs_ptr = NULL
        cdef int bs_words = 0
        cdef const uint8_t* pal_raw = <const uint8_t*>palette.data
        cdef int pal_bytes = palette.size * self._itemsize
        # 字节级扫描判断是否含非零值
        cdef bint has_nz = False
        cdef int j
        for j in range(pal_bytes):
            if pal_raw[j] != 0:
                has_nz = True
                break
        if bitstream is not None and bits > 0:
            bs_arr  = np.ascontiguousarray(bitstream, dtype=np.uint32)
            bs_ptr  = <const uint32_t*>bs_arr.data
            bs_words = bs_arr.size
        self._arena.set_block(idx, pal_raw, pal_bytes, bits, bs_ptr, bs_words, has_nz)

    def set_null(self, int idx):
        self._arena.set_null(idx)

    def __getitem__(self, int idx):
        """返回 dict（兼容旧接口，非热路径）。"""
        if idx < 0 or idx >= self._arena.total:
            raise IndexError(f"index {idx} out of range")
        if self._arena.get_is_null(idx) and not self._arena.get_has_nonzero(idx):
            return None
        cdef int pal_bytes = self._arena.get_pal_byte_size(idx)
        cdef int bs_sz     = self._arena.get_bs_size(idx)
        cdef uint8_t bits  = self._arena.get_bits(idx)
        cdef int pal_elems = pal_bytes // self._itemsize if self._itemsize > 0 else 0

        pal_arr = np.empty(pal_elems, dtype=self._dtype_obj)
        if pal_bytes > 0 and self._arena.get_pal_bytes(idx) != NULL:
            memcpy(np.PyArray_DATA(pal_arr),
                   self._arena.get_pal_bytes(idx), pal_bytes)

        bs_arr = None
        if bits > 0 and bs_sz > 0 and self._arena.get_bs_data(idx) != NULL:
            bs_arr = np.empty(bs_sz, dtype=np.uint32)
            memcpy(np.PyArray_DATA(bs_arr),
                   self._arena.get_bs_data(idx), bs_sz * 4)

        return {"palette": pal_arr, "bits": int(bits), "bitstream": bs_arr}


# ============================================================
#  新版热路径函数：直接操作 PyBlockStore
# ============================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def transform_where_compressed_store(
        PyBlockStore src,
        PyBlockStore dst,
        dtype, size_t segid,
        object true_val, object false_val,
        object out_dtype,
        tuple block_size):
    """
    in-place 写入 dst：压缩态 where 映射（src → dst）。
    直接使用 BlockArena CBlock*，跳过所有 Python dict 操作。
    """
    cdef int n = src._arena.total
    cdef const CBlock* src_cb = src._arena.get_cblocks()
    cdef int src_itemsize = src._arena.itemsize
    cdef int out_itemsize = np.dtype(out_dtype).itemsize

    cdef ptrdiff_t[3] blk_sz = [block_size[0], block_size[1], block_size[2]]
    cdef int words_needed = (blk_sz[0] * blk_sz[1] * blk_sz[2] + 31) // 32

    # 预构建常量 palette 字节（最大 8 字节 = uint64 两元素）
    cdef uint8_t pal_false_bytes[8]
    cdef uint8_t pal_true_bytes[8]
    cdef uint8_t pal_mixed_bytes[16]
    memset(pal_false_bytes, 0, 8)
    memset(pal_true_bytes,  0, 8)
    memset(pal_mixed_bytes, 0, 16)

    # 将 true_val / false_val 写成小端字节
    cdef uint64_t tv64 = <uint64_t>int(true_val)
    cdef uint64_t fv64 = <uint64_t>int(false_val)
    memcpy(pal_true_bytes,               <uint8_t*>&tv64, out_itemsize)
    memcpy(pal_false_bytes,              <uint8_t*>&fv64, out_itemsize)
    memcpy(pal_mixed_bytes,              <uint8_t*>&fv64, out_itemsize)
    memcpy(pal_mixed_bytes + out_itemsize, <uint8_t*>&tv64, out_itemsize)

    cdef bint tv_nonzero = (tv64 != 0)
    cdef bint fv_nonzero = (fv64 != 0)

    cdef np.ndarray new_bs_arr
    cdef vector[uint32_t] new_bs_vec
    cdef int b_type
    cdef size_t pal_size
    cdef uint8_t cb_bits
    cdef bint cb_hnz
    cdef const void* cb_palette
    cdef const uint32_t* cb_bitstream

    for i in range(n):
        cb_hnz      = src_cb[i].has_nonzero
        cb_bits     = src_cb[i].bits
        cb_palette  = src_cb[i].palette
        cb_bitstream = src_cb[i].bitstream

        if not cb_hnz:
            # 全零块 → 必为 miss
            dst._arena.set_block(i, pal_false_bytes, out_itemsize, 0, NULL, 0, fv_nonzero)
            continue

        pal_size = <size_t>1 if cb_bits == 0 else (<size_t>1 << cb_bits)

        if src_itemsize == 1:
            b_type = CheckBlockType[uint8_t](<uint8_t*>cb_palette, pal_size, <uint8_t>segid)
        elif src_itemsize == 4:
            b_type = CheckBlockType[uint32_t](<uint32_t*>cb_palette, pal_size, <uint32_t>segid)
        else:
            b_type = CheckBlockType[uint64_t](<uint64_t*>cb_palette, pal_size, <uint64_t>segid)

        if b_type == 0:
            dst._arena.set_block(i, pal_false_bytes, out_itemsize, 0, NULL, 0, fv_nonzero)
        elif b_type == 1:
            dst._arena.set_block(i, pal_true_bytes, out_itemsize, 0, NULL, 0, tv_nonzero)
        else:
            new_bs_arr = np.empty(words_needed, dtype=np.uint32)
            if src_itemsize == 1:
                CreateGenericBinaryBitstream[uint8_t](
                    cb_bitstream, <uint8_t*>cb_palette, cb_bits,
                    <uint8_t>segid, <uint32_t*>np.PyArray_DATA(new_bs_arr), blk_sz)
            elif src_itemsize == 4:
                CreateGenericBinaryBitstream[uint32_t](
                    cb_bitstream, <uint32_t*>cb_palette, cb_bits,
                    <uint32_t>segid, <uint32_t*>np.PyArray_DATA(new_bs_arr), blk_sz)
            else:
                CreateGenericBinaryBitstream[uint64_t](
                    cb_bitstream, <uint64_t*>cb_palette, cb_bits,
                    <uint64_t>segid, <uint32_t*>np.PyArray_DATA(new_bs_arr), blk_sz)
            dst._arena.set_block(i, pal_mixed_bytes, 2 * out_itemsize, 1,
                                  <uint32_t*>np.PyArray_DATA(new_bs_arr), words_needed,
                                  tv_nonzero or fv_nonzero)


@cython.boundscheck(False)
@cython.wraparound(False)
def find_nearest_seed_fast_store(
        PyBlockStore store,
        tuple grid_size, tuple block_size, tuple req_size,
        tuple q2p_offset, tuple center, object out_dtype,
        bint include_self=False):
    """直接使用 BlockArena CBlock*，无需重建临时 vector。"""
    cdef int nx_b = grid_size[0], ny_b = grid_size[1], nz_b = grid_size[2]
    cdef int bx = block_size[0], by = block_size[1], bz = block_size[2]
    cdef int req_x = req_size[0], req_y = req_size[1], req_z = req_size[2]
    cdef int q2p_x = q2p_offset[0], q2p_y = q2p_offset[1], q2p_z = q2p_offset[2]
    cdef int cx = center[0], cy = center[1], cz = center[2]
    cdef int itemsize = np.dtype(out_dtype).itemsize

    cdef const CBlock* c_blocks = store._arena.get_cblocks()
    cdef int seed_x = -1, seed_y = -1, seed_z = -1
    cdef bint found = False

    with nogil:
        if itemsize == 1:
            found = FastNearestSeed[uint8_t](c_blocks, nx_b, ny_b, nz_b, bx, by, bz,
                req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
                cx, cy, cz, seed_x, seed_y, seed_z, include_self)
        elif itemsize == 4:
            found = FastNearestSeed[uint32_t](c_blocks, nx_b, ny_b, nz_b, bx, by, bz,
                req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
                cx, cy, cz, seed_x, seed_y, seed_z, include_self)
        else:
            found = FastNearestSeed[uint64_t](c_blocks, nx_b, ny_b, nz_b, bx, by, bz,
                req_x, req_y, req_z, q2p_x, q2p_y, q2p_z,
                cx, cy, cz, seed_x, seed_y, seed_z, include_self)

    if found:
        return np.array([seed_x, seed_y, seed_z], dtype=int)
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
def extract_cc_fast_store(
        PyBlockStore store,
        tuple grid_size, tuple block_size, tuple req_size,
        tuple q2p_offset, tuple seed_point, object out_dtype):
    """BFS 连通域提取，结果原地写回 PyBlockStore。"""
    cdef int nx_b = grid_size[0], ny_b = grid_size[1], nz_b = grid_size[2]
    cdef int bx = block_size[0], by = block_size[1], bz = block_size[2]
    cdef int req_x = req_size[0], req_y = req_size[1], req_z = req_size[2]
    cdef int q2p_x = q2p_offset[0], q2p_y = q2p_offset[1], q2p_z = q2p_offset[2]
    cdef int sx = seed_point[0], sy = seed_point[1], sz = seed_point[2]
    cdef int total_blocks = nx_b * ny_b * nz_b
    cdef int itemsize = np.dtype(out_dtype).itemsize

    cdef const CBlock* c_blocks = store._arena.get_cblocks()
    cdef vector[uint8_t*] cc_masks

    with nogil:
        if itemsize == 1:
            cc_masks = SparseBFS26[uint8_t](c_blocks, nx_b, ny_b, nz_b, bx, by, bz,
                req_x, req_y, req_z, q2p_x, q2p_y, q2p_z, sx, sy, sz)
        elif itemsize == 4:
            cc_masks = SparseBFS26[uint32_t](c_blocks, nx_b, ny_b, nz_b, bx, by, bz,
                req_x, req_y, req_z, q2p_x, q2p_y, q2p_z, sx, sy, sz)
        else:
            cc_masks = SparseBFS26[uint64_t](c_blocks, nx_b, ny_b, nz_b, bx, by, bz,
                req_x, req_y, req_z, q2p_x, q2p_y, q2p_z, sx, sy, sz)

    # 常量调色板字节（miss=[0], hit=[255], mixed=[0,255]）
    cdef uint8_t pal_false_bytes[8]
    cdef uint8_t pal_true_bytes[8]
    cdef uint8_t pal_mixed_bytes[16]
    memset(pal_false_bytes, 0, 8)
    memset(pal_true_bytes,  0, 8)
    memset(pal_mixed_bytes, 0, 16)
    # true_val = 255
    cdef uint64_t tv = 255
    memcpy(pal_true_bytes,               <uint8_t*>&tv, itemsize)
    memcpy(pal_mixed_bytes + itemsize,   <uint8_t*>&tv, itemsize)

    cdef vector[uint32_t] new_bs_vec
    cdef int compress_status
    cdef int i
    for i in range(total_blocks):
        if cc_masks[i] == NULL:
            store._arena.set_block(i, pal_false_bytes, itemsize, 0, NULL, 0, False)
        else:
            with nogil:
                compress_status = CompressMaskFast(cc_masks[i], bx * by * bz, new_bs_vec)
                free(cc_masks[i])
            if compress_status == 0:
                store._arena.set_block(i, pal_false_bytes, itemsize, 0, NULL, 0, False)
            elif compress_status == 1:
                store._arena.set_block(i, pal_true_bytes, itemsize, 0, NULL, 0, True)
            else:
                store._arena.set_block(i, pal_mixed_bytes, 2 * itemsize, 1,
                                        new_bs_vec.data(), new_bs_vec.size(), True)


@cython.boundscheck(False)
@cython.wraparound(False)
def fill_slab_buffer_store(
        PyBlockStore store,
        np.ndarray buffer,
        tuple block_size, tuple slab_grid_size,
        object dtype, int gz):
    """
    解压第 gz 层 slab 到 buffer（直接使用 BlockArena CBlock*）。
    """
    cdef int nx_slab = slab_grid_size[0], ny_slab = slab_grid_size[1]
    cdef int bx = block_size[0], by = block_size[1], bz = block_size[2]
    cdef int num_per_slab = nx_slab * ny_slab
    cdef int slab_start   = gz * num_per_slab
    cdef int itemsize = np.dtype(dtype).itemsize

    cdef const CBlock* arena_cb = store._arena.get_cblocks()
    cdef vector[SlabBlock] slab_blocks
    slab_blocks.resize(num_per_slab)

    cdef int i
    for i in range(num_per_slab):
        slab_blocks[i].has_nonzero = arena_cb[slab_start + i].has_nonzero
        slab_blocks[i].bits        = arena_cb[slab_start + i].bits
        slab_blocks[i].palette     = arena_cb[slab_start + i].palette
        slab_blocks[i].bitstream   = arena_cb[slab_start + i].bitstream
        slab_blocks[i].gx = i % nx_slab
        slab_blocks[i].gy = i // nx_slab

    with nogil:
        if itemsize == 1:
            DecompressSlabToBuffer[uint8_t](slab_blocks.data(), num_per_slab,
                bx, by, bz, nx_slab, ny_slab, <uint8_t*>buffer.data)
        elif itemsize == 4:
            DecompressSlabToBuffer[uint32_t](slab_blocks.data(), num_per_slab,
                bx, by, bz, nx_slab, ny_slab, <uint32_t*>buffer.data)
        else:
            DecompressSlabToBuffer[uint64_t](slab_blocks.data(), num_per_slab,
                bx, by, bz, nx_slab, ny_slab, <uint64_t*>buffer.data)


@cython.boundscheck(False)
@cython.wraparound(False)
def decompress_block_grid_store(PyBlockStore store, tuple block_size, tuple grid_dims, dtype):
    """从 PyBlockStore 解压完整网格为稠密数组（供 get_raw_data 使用）。"""
    cdef int nx_b = grid_dims[0], ny_b = grid_dims[1]
    cdef np.ndarray out = np.zeros(
        (nx_b * block_size[0], ny_b * block_size[1], grid_dims[2] * block_size[2]),
        dtype=dtype, order='F'
    )
    cdef ptrdiff_t[3] blk_sz = [block_size[0], block_size[1], block_size[2]]
    cdef ptrdiff_t out_sx = 1
    cdef ptrdiff_t out_sy = out.shape[0]
    cdef ptrdiff_t out_sz = out.shape[0] * out.shape[1]
    cdef int n = store._arena.total
    cdef const CBlock* c_blocks = store._arena.get_cblocks()
    cdef object target_dtype = np.dtype(dtype)
    cdef int itemsize = target_dtype.itemsize

    cdef uint8_t  pal8_buf[256]
    cdef uint32_t pal32_buf[256]
    cdef uint64_t pal64_buf[256]
    cdef int i, ix_b, iy_b, iz_b
    cdef size_t pal_size

    for i in range(n):
        if not c_blocks[i].has_nonzero:
            continue
        ix_b = i % nx_b
        iy_b = (i // nx_b) % ny_b
        iz_b = i // (nx_b * ny_b)
        pal_size = <size_t>1 if c_blocks[i].bits == 0 else (<size_t>1 << c_blocks[i].bits)
        if itemsize == 1:
            memcpy(pal8_buf, c_blocks[i].palette, pal_size * 1)
            DecompressSingleBlock[uint8_t](
                c_blocks[i].bitstream, pal8_buf, c_blocks[i].bits,
                <uint8_t*>out.data, out_sx, out_sy, out_sz,
                ix_b * blk_sz[0], iy_b * blk_sz[1], iz_b * blk_sz[2], blk_sz)
        elif itemsize == 4:
            memcpy(pal32_buf, c_blocks[i].palette, pal_size * 4)
            DecompressSingleBlock[uint32_t](
                c_blocks[i].bitstream, pal32_buf, c_blocks[i].bits,
                <uint32_t*>out.data, out_sx, out_sy, out_sz,
                ix_b * blk_sz[0], iy_b * blk_sz[1], iz_b * blk_sz[2], blk_sz)
        else:
            memcpy(pal64_buf, c_blocks[i].palette, pal_size * 8)
            DecompressSingleBlock[uint64_t](
                c_blocks[i].bitstream, pal64_buf, c_blocks[i].bits,
                <uint64_t*>out.data, out_sx, out_sy, out_sz,
                ix_b * blk_sz[0], iy_b * blk_sz[1], iz_b * blk_sz[2], blk_sz)
    return out
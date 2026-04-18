from functools import partial
import itertools
import math
import multiprocessing as mp
import os
import threading

import numpy as np
from tqdm import tqdm

from cloudfiles import reset_connection_pools, CloudFiles, compression
import fastremap

from ....exceptions import EmptyVolumeException, EmptyFileException
from ....lib import (  
  mkdir, clamp, xyzrange, Vec, 
  Bbox, min2, max2, 
  jsonify, red, sip, first
)
from .... import chunks

from cloudvolume.scheduler import schedule_jobs
from cloudvolume.threaded_queue import DEFAULT_THREADS
from cloudvolume.volumecutout import VolumeCutout

import cloudvolume.sharedmemory as shm

from ..common import should_compress, content_type
from .common import (
  parallel_execution, 
  chunknames, shade, gridpoints,
  compressed_morton_code
)
import itertools
import numpy as np
from functools import partial
from .. import sharding
import time
from tqdm import tqdm
from operator import itemgetter
from os.path import basename
import itertools
progress_queue = None # defined in common.initialize_synchronization
fs_lock = None # defined in common.initialize_synchronization

import math
import time
import itertools
from functools import partial
import numpy as np
import sys
sys.path.append("/CX/neuro_tracking/xinr/cloudvolume_test/fast_cloudvolume")
from compressedvoxel import CompressedVoxelContainer

def download_sharded_compressed_block(
  requested_bbox, mip,
  meta, cache, lru, lru_encoding, spec,
  compress, progress,
  fill_missing, 
  order, background_color,
  label, partial_decompress_parallel, cache_thread, segid_list
):
  total_start_time = time.time()
  download_sharded_info = {}
  download_sharded_info['download_sharded_requested_bbox_size'] = requested_bbox.size3()
  download_sharded_info['download_sharded_requested_bbox_volume'] = requested_bbox.volume()

  full_bbox = requested_bbox.expand_to_chunk_size(
    meta.chunk_size(mip), offset=meta.voxel_offset(mip)
  )
  full_bbox = Bbox.clamp(full_bbox, meta.bounds(mip))
  shape = list(requested_bbox.size3()) + [ meta.num_channels ]
  download_sharded_info['download_sharded_full_bbox_size'] = full_bbox.size3()
  download_sharded_info['download_sharded_full_bbox_volume'] = full_bbox.volume() 
  start_time = time.time()

  download_sharded_info['download_sharded_fill_buffer'] = time.time() - start_time
  if not meta.overlaps_roi(requested_bbox, mip):
    return None

  chunk_size = meta.chunk_size(mip)
  grid_size = np.ceil(meta.bounds(mip).size3() / chunk_size).astype(np.uint32)

  reader = sharding.ShardReader(meta, cache, spec)
  bounds = meta.bounds(mip)
  download_sharded_info['download_sharded_expand_bbox_and_get_chunk_initial_time'] = time.time() - start_time
  start_time = time.time()
  gpts = list(gridpoints(full_bbox, bounds, chunk_size))

  code_map = {}
  morton_codes = compressed_morton_code(gpts, grid_size)
  for gridpoint, morton_code in zip(gpts, morton_codes):
    cutout_bbox = Bbox(
      bounds.minpt + gridpoint * chunk_size,
      min2(bounds.minpt + (gridpoint + 1) * chunk_size, bounds.maxpt)
    )
    code_map[morton_code] = cutout_bbox
  download_sharded_info['download_sharded_info_calculate_morton_code'] = time.time() - start_time
  start_time = time.time()
  single_voxel = requested_bbox.volume() == 1
  if label is not None:
    decode_fn = partial(decode_binary_image, label)
  elif single_voxel:
    decode_fn = partial(decode_single_voxel, requested_bbox.minpt - full_bbox.minpt)
  else:
      decode_fn = decode_partial_compressed_block
  download_sharded_info['download_sharded_decode_fn_init'] = time.time() - start_time
  start_time = time.time()
  all_keys = set(code_map.keys())
  download_sharded_info['download_sharded_full_bbox_chunk_num'] = len(all_keys)
  lru_keys = set([ key for key in all_keys if key in lru ])
  io_keys = all_keys - lru_keys
  del all_keys
  download_sharded_info['download_sharded_get_io_task'] = time.time() - start_time
  download_sharded_info['download_sharded_cache_hit_num'] = len(lru_keys)
  download_sharded_info['download_sharded_cache_miss'] = len(io_keys)
  start_time_lru = time.time()
  lru_chunkdata = [ (zcode, lru[zcode]) for zcode in lru_keys ]
  download_sharded_info['download_sharded_get_data_lru_time'] = time.time() - start_time_lru

  start_time_io = time.time()
  io_chunkdata, get_data_info = reader.get_data(io_keys, meta.key(mip), cache_thread = cache_thread, progress=progress)
  io_chunkdata = { k: (meta.encoding(mip), v) for k,v in io_chunkdata.items() } 
  download_sharded_info['download_sharded_get_data_time'] = time.time() - start_time_io

  # 将新获取的数据更新到LRU缓存
  start_time = time.time()
  for zcode, (data_encoding, chunkdata) in io_chunkdata.items():
      lru[zcode] = (data_encoding, chunkdata)
  download_sharded_info['download_sharded_update_lru_time'] = time.time() - start_time

  start_time = time.time()
  chunk_num_total = 0
  time_decode = 0.0
  container = CompressedVoxelContainer(requested_bbox, full_bbox, meta.compressed_segmentation_block_size(mip), meta.dtype)
  for zcode, (data_encoding, chunkdata) in itertools.chain(io_chunkdata.items(), lru_chunkdata):
    chunk_num_total = chunk_num_total +1
    cutout_bbox = code_map[zcode]
    decode_start = time.time()
    chunk_relative_grid = (cutout_bbox.minpt - full_bbox.minpt) // chunk_size
    decode_fn(
      meta, requested_bbox, cutout_bbox, 
      chunkdata, fill_missing, mip,
      background_color=background_color,
      encoding=data_encoding,
      chunk_relative_grid=chunk_relative_grid,
      container=container,
      segid_list = segid_list
    )
    
    time_decode = time_decode + time.time() - decode_start

  download_sharded_info['download_sharded_compress_uzip_time'] = time.time() - start_time
  download_sharded_info['download_sharded_compress_uzip_decode_fn_time'] = time_decode
  download_sharded_info['download_sharded_compress_uzip_chunk_num'] = chunk_num_total
  download_sharded_info['download_sharded_total_time'] = time.time() - total_start_time
  download_sharded_info['requested_bbox'] = [requested_bbox.minpt, requested_bbox.maxpt]
  download_sharded_info.update(get_data_info)
  return container,download_sharded_info

def download_sharded(
  requested_bbox, mip,
  meta, cache, lru, lru_encoding, spec,
  compress, progress,
  fill_missing, 
  order, background_color,
  label, partial_decompress_parallel,
  renderbuffer, cache_thread, l2cache_size = 0
):
  total_start_time = time.time()
  download_sharded_info = {}
  download_sharded_info['download_sharded_requested_bbox_size'] = requested_bbox.size3()
  download_sharded_info['download_sharded_requested_bbox_volume'] = requested_bbox.volume()

  full_bbox = requested_bbox.expand_to_chunk_size(
    meta.chunk_size(mip), offset=meta.voxel_offset(mip)
  )
  full_bbox = Bbox.clamp(full_bbox, meta.bounds(mip))
  shape = list(requested_bbox.size3()) + [ meta.num_channels ]
  download_sharded_info['download_sharded_full_bbox_size'] = full_bbox.size3()
  download_sharded_info['download_sharded_full_bbox_volume'] = full_bbox.volume() 
  start_time = time.time()
  if renderbuffer is None:
    if label is None:
      renderbuffer = np.full(
        shape=shape, fill_value=background_color,
        dtype=meta.dtype, order=order
      )
    else:
      renderbuffer = np.zeros(shape, dtype=bool, order="F")
  else:
    renderbuffer = np.ndarray(shape=shape, dtype=meta.dtype, buffer=renderbuffer, order=order)
  download_sharded_info['download_sharded_fill_buffer'] = time.time() - start_time
  if not meta.overlaps_roi(requested_bbox, mip):
    return renderbuffer

  chunk_size = meta.chunk_size(mip)
  grid_size = np.ceil(meta.bounds(mip).size3() / chunk_size).astype(np.uint32)

  reader = sharding.ShardReader(meta, cache, spec)
  bounds = meta.bounds(mip)
  download_sharded_info['download_sharded_expand_bbox_and_get_chunk_initial_time'] = time.time() - start_time
  start_time = time.time()
  gpts = list(gridpoints(full_bbox, bounds, chunk_size))

  code_map = {}
  morton_codes = compressed_morton_code(gpts, grid_size)
  for gridpoint, morton_code in zip(gpts, morton_codes):
    cutout_bbox = Bbox(
      bounds.minpt + gridpoint * chunk_size,
      min2(bounds.minpt + (gridpoint + 1) * chunk_size, bounds.maxpt)
    )
    code_map[morton_code] = cutout_bbox
  download_sharded_info['download_sharded_info_calculate_morton_code'] = time.time() - start_time
  start_time = time.time()
  single_voxel = requested_bbox.volume() == 1
  use_decode_partial_parallel = False
  use_decode_partial = False
  if label is not None:
    decode_fn = partial(decode_binary_image, label)
  elif single_voxel:
    decode_fn = partial(decode_single_voxel, requested_bbox.minpt - full_bbox.minpt)
  else:
    if partial_decompress_parallel is None:
      decode_fn = decode
    elif partial_decompress_parallel <= 0:
      decode_fn = decode_partial
      use_decode_partial = True
    else:
      decode_fn = decode_partial_parallel
      use_decode_partial_parallel = True
  download_sharded_info['download_sharded_decode_fn_init'] = time.time() - start_time
  start_time = time.time()
  all_keys = set(code_map.keys())
  download_sharded_info['download_sharded_full_bbox_chunk_num'] = len(all_keys)
  lru_keys = set([ key for key in all_keys if key in lru ])
  io_keys = all_keys - lru_keys
  del all_keys
  download_sharded_info['download_sharded_get_io_task'] = time.time() - start_time
  download_sharded_info['download_sharded_cache_hit_num'] = len(lru_keys)
  download_sharded_info['download_sharded_cache_miss'] = len(io_keys)
  start_time_lru = time.time()
  lru_chunkdata = [ (zcode, lru[zcode]) for zcode in lru_keys ]
  download_sharded_info['download_sharded_get_data_lru_time'] = time.time() - start_time_lru

  start_time_io = time.time()
  io_chunkdata, get_data_info = reader.get_data(io_keys, meta.key(mip), cache_thread = cache_thread, progress=progress)
  io_chunkdata = { k: (meta.encoding(mip), v) for k,v in io_chunkdata.items() } 
  download_sharded_info['download_sharded_get_data_time'] = time.time() - start_time_io

  # 将新获取的数据更新到LRU缓存
  start_time = time.time()
  for zcode, (data_encoding, chunkdata) in io_chunkdata.items():
      lru[zcode] = (data_encoding, chunkdata)
  download_sharded_info['download_sharded_update_lru_time'] = time.time() - start_time

  start_time = time.time()
  chunk_num_total = 0
  time_decode = 0.0
  if use_decode_partial_parallel:
    tasks = []
    for zcode, (data_encoding, chunkdata) in itertools.chain(io_chunkdata.items(), lru_chunkdata):
      chunk_num_total = chunk_num_total +1
      cutout_bbox = code_map[zcode]
      job={
        'meta':meta,
        'cutout_bbox':cutout_bbox,
        'output_array_ptr':renderbuffer.ctypes.data,
        'output_array_ndim':renderbuffer.ndim,
        'output_array_shape':renderbuffer.shape,
        'output_array_strides':renderbuffer.strides,
        'content':chunkdata,
        'encoding':data_encoding,
      }
      tasks.append(job)
    decode_start = time.time()
    decode_fn(
      tasks,
      requested_bbox,
      fill_missing, mip,
      background_color=background_color,
      partial_decompress_parallel=partial_decompress_parallel,
      allow_none=True, l2cache_size = l2cache_size
    )
    time_decode = time_decode + time.time() - decode_start
  else:
    for zcode, (data_encoding, chunkdata) in itertools.chain(io_chunkdata.items(), lru_chunkdata):
      chunk_num_total = chunk_num_total +1
      cutout_bbox = code_map[zcode]
      decode_start = time.time()
      if use_decode_partial:
        img3d, cutout_bbox = decode_fn(
          meta, requested_bbox, cutout_bbox, 
          chunkdata, fill_missing, mip,
          background_color=background_color,
          encoding=data_encoding,
          )
      else:
        img3d = decode_fn(
          meta, cutout_bbox, 
          chunkdata, fill_missing, mip,
          background_color=background_color,
          encoding=data_encoding,
        )
      time_decode = time_decode + time.time() - decode_start
      if single_voxel:
        renderbuffer[:] = img3d
      elif single_voxel and label is not None:
        renderbuffer[:] = img3d == label
      else:
        shade(renderbuffer, requested_bbox, img3d, cutout_bbox)
  download_sharded_info['download_sharded_compress_uzip_time'] = time.time() - start_time
  download_sharded_info['download_sharded_compress_uzip_decode_fn_time'] = time_decode
  download_sharded_info['download_sharded_compress_uzip_chunk_num'] = chunk_num_total
  download_sharded_info['download_sharded_total_time'] = time.time() - total_start_time
  download_sharded_info['requested_bbox'] = [requested_bbox.minpt, requested_bbox.maxpt]
  download_sharded_info.update(get_data_info)
  return VolumeCutout.from_volume(
      meta, mip, renderbuffer, 
      requested_bbox
  ), download_sharded_info

def download_raw_unsharded(
  requested_bbox, mip, 
  meta, cache, 
  decompress, 
  progress, parallel, 
  secrets, green, fill_missing,
  compress_type, background_color,
  cache_only
):
  """
  Download all the chunks without rendering.

  decompress: strip bytestream compression like gzip or br
    leaving the image encoding untouched.
  """
  full_bbox = requested_bbox.expand_to_chunk_size(
    meta.chunk_size(mip), offset=meta.voxel_offset(mip)
  )
  full_bbox = Bbox.clamp(full_bbox, meta.bounds(mip))
  cloudpaths = chunknames(
    full_bbox, meta.bounds(mip), 
    meta.key(mip), meta.chunk_size(mip), 
    protocol=meta.path.protocol
  )

  results = {}
  def store_result(binary, bbox):
    nonlocal results
    if cache_only:
      return
    key = meta.join(meta.key(mip), bbox.to_filename())
    results[key] = binary

  def noop_decode(
    meta, input_bbox, 
    content, fill_missing, 
    mip, background_color=0,
    encoding=None,
  ):
    return content

  compress_cache = should_compress(meta.encoding(mip), compress_type, cache, iscache=True)

  download_chunks_threaded(
    meta, cache, 
    lru=None, lru_encoding="same", mip=mip, cloudpaths=cloudpaths, 
    fn=store_result, decode_fn=noop_decode, fill_missing=fill_missing,
    progress=progress, compress_cache=compress_cache, 
    green=green, secrets=secrets, background_color=background_color,
    full_decode=True,
  )

  return results

def download(
  requested_bbox, mip, 
  meta, cache, lru, lru_encoding,
  fill_missing, progress,
  parallel, location, 
  retain, use_shared_memory, 
  use_file, compress, order='F',
  green=False, secrets=None,
  renumber=False, background_color=0,
  label=None
):
  """Cutout a requested bounding box from storage and return it as a numpy array."""
  
  full_bbox = requested_bbox.expand_to_chunk_size(
    meta.chunk_size(mip), offset=meta.voxel_offset(mip)
  )
  full_bbox = Bbox.clamp(full_bbox, meta.bounds(mip))
  cloudpaths = chunknames(
    full_bbox, meta.bounds(mip), 
    meta.key(mip), meta.chunk_size(mip), 
    protocol=meta.path.protocol
  )
  shape = list(requested_bbox.size3()) + [ meta.num_channels ]

  compress_cache = should_compress(meta.encoding(mip), compress, cache, iscache=True)

  handle = None

  if renumber and (parallel != 1):
    raise ValueError("renumber is not supported for parallel operation.")

  if use_shared_memory and use_file:
    raise ValueError("use_shared_memory and use_file are mutually exclusive arguments.")

  if label is not None:
    dtype = bool
    background_color = 0
    decode_fn = partial(decode_binary_image, label)
  else:
    decode_fn = decode
    dtype = np.uint16 if renumber else meta.dtype

  if not meta.overlaps_roi(requested_bbox, mip):
    return np.full(
      shape=shape, fill_value=background_color,
      dtype=dtype, order=order
    )

  if requested_bbox.volume() == 1:
    return download_single_voxel_unsharded(
      meta, cache, lru, lru_encoding,
      requested_bbox, first(cloudpaths), 
      mip, fill_missing, compress_cache,
      secrets, renumber, background_color,
      label
    )
  elif parallel == 1:
    if use_shared_memory: # write to shared memory
      handle, renderbuffer = shm.ndarray(
        shape, dtype=dtype, order=order,
        location=location, lock=fs_lock
      )
      if not retain:
        shm.unlink(location)
    elif use_file: # write to ordinary file
      handle, renderbuffer = shm.ndarray_fs(
        shape, dtype=dtype, order=order,
        location=location, lock=fs_lock,
        emulate_shm=False
      )
      if not retain:
        os.unlink(location)
    elif background_color == 0:
      renderbuffer = np.zeros(shape, dtype=dtype, order=order)
    else:
      renderbuffer = np.full(shape=shape, fill_value=background_color,
                             dtype=dtype, order=order)

    def process(img3d, bbox):
      shade(renderbuffer, requested_bbox, img3d, bbox)

    remap = { background_color: background_color }
    lock = threading.Lock()
    N = 1
    def process_renumber(img3d, bbox):
      nonlocal N
      nonlocal lock 
      nonlocal remap
      nonlocal renderbuffer
      if img3d is None:
        img_labels = [ background_color ]
      else:
        img_labels = fastremap.unique(img3d)
      with lock:
        for lbl in img_labels:
          if lbl not in remap:
            remap[lbl] = N
            N += 1
        if N > np.iinfo(renderbuffer.dtype).max:
          renderbuffer = fastremap.refit(renderbuffer, value=N, increase_only=True)

        fastremap.remap(img3d, remap, in_place=True)
        shade(renderbuffer, requested_bbox, img3d, bbox)

    fn = process
    if renumber and not (use_file or use_shared_memory):
      fn = process_renumber  

    download_chunks_threaded(
      meta, cache, lru, lru_encoding, mip, cloudpaths, 
      fn=fn, decode_fn=decode_fn, fill_missing=fill_missing,
      progress=progress, compress_cache=compress_cache, 
      green=green, secrets=secrets, background_color=background_color,
      full_decode=True,
    )
  else:
    handle, renderbuffer = multiprocess_download(
      requested_bbox, mip, cloudpaths,
      meta, cache, lru, lru_encoding, compress_cache,
      fill_missing, progress,
      parallel, location, retain, 
      use_shared_memory=(use_file == False),
      order=order,
      green=green,
      secrets=secrets,
      background_color=background_color
    )
  
  out = VolumeCutout.from_volume(
    meta, mip, renderbuffer, 
    requested_bbox, handle=handle
  )
  if renumber:
    return (out, remap)
  return out

def download_single_voxel_unsharded(
  meta, cache, lru, lru_encoding,
  requested_bbox, filename, 
  mip, fill_missing, compress_cache,
  secrets, renumber, background_color,
  segid
):
  """Specialized function for rapidly extracting a single voxel."""
  locations = cache.compute_data_locations([ filename ])
  cachedir = 'file://' + cache.path

  if locations["local"]:
    cloudpath = cachedir
    cache_enabled = False
    locking = cache.config.cache_locking
  else:
    cloudpath = meta.cloudpath
    cache_enabled = cache.enabled
    locking = False

  if filename is None:
    if fill_missing:
      label = np.zeros((1,1,1,1), dtype=meta.dtype)
    else:
      raise EmptyVolumeException(requested_bbox)
  else:
    chunk_bbx = Bbox.from_filename(filename)
    label, _ = download_chunk(
      meta, cache, lru, lru_encoding,
      cloudpath, mip,
      filename, fill_missing,
      cache_enabled, compress_cache,
      secrets, background_color,
      partial(decode_single_voxel, requested_bbox.minpt - chunk_bbx.minpt),
      decompress=True, locking=locking, full_decode=False,
    )

  if segid is not None:
    label = label == segid

  if renumber:
    lbl = label[0,0,0,0]
    if lbl == background_color:
      return label, { lbl: lbl }
    else:
      remap = { lbl: 1 }
      label[0,0,0,0] = 1
      return label, remap

  return label

def multiprocess_download(
    requested_bbox, mip, cloudpaths,
    meta, cache, lru, lru_encoding, compress_cache,
    fill_missing, progress,
    parallel, location, 
    retain, use_shared_memory, order,
    green, secrets=None, background_color=0,
  ):
  cpd = partial(child_process_download, 
    meta, cache, 
    mip, compress_cache, requested_bbox, 
    fill_missing, progress,
    location, use_shared_memory,
    green, secrets, background_color
  )

  if lru.size > 0:
    for path in cloudpaths:
      lru.pop(path, None)

  parallel_execution(
    cpd, cloudpaths, parallel, 
    progress=progress, 
    desc="Download",
    cleanup_shm=location,
    block_size=750,
  )

  shape = list(requested_bbox.size3()) + [ meta.num_channels ]

  if use_shared_memory:
    mmap_handle, renderbuffer = shm.ndarray(
      shape, dtype=meta.dtype, order=order, 
      location=location, lock=fs_lock
    )
  else:
    handle, renderbuffer = shm.ndarray_fs(
      shape, dtype=meta.dtype, order=order,
      location=location, lock=fs_lock,
      emulate_shm=False
    )

  repopulate_lru_from_shm(meta, mip, lru, lru_encoding, renderbuffer, requested_bbox)

  if not retain:
    if use_shared_memory:
      shm.unlink(location)
    else:
      os.unlink(location)

  return mmap_handle, renderbuffer

def repopulate_lru_from_shm(
  meta, mip, lru, lru_encoding,
  renderbuffer, requested_bbox
):
  """
  Used for repopulating the LRU from the shared memory buffer
  after a multiprocess download. This can't be done in process
  due to the communication overhead between processes.
  """
  if lru.size == 0:
    return

  retracted_bbox = requested_bbox.shrink_to_chunk_size(
    meta.chunk_size(mip), offset=meta.voxel_offset(mip)
  )
  retracted_bbox = Bbox.clamp(retracted_bbox, meta.bounds(mip))
  delta = retracted_bbox.minpt - requested_bbox.minpt
  core_chunks = list(chunknames(
    retracted_bbox, 
    meta.bounds(mip), meta.key(mip), meta.chunk_size(mip),
    protocol=meta.path.protocol,
  ))

  encoding = lru_encoding
  if encoding == "same":
    # Since the parallel version populates the LRU via an image and
    # you don't get the benefit of accessing the raw downloaded bytes,
    # there will be a performance regression for "same" since e.g.
    # jpeg -> img -> jpeg will instead of decode -> img,lru you'll
    # have decode -> img -> encode -> lru. Therefore, this is hacky,
    # but backwards compatible and strictly expands the capabilities
    # of the LRU.
    encoding = "raw" # would ordinarily be: meta.encoding(mip)

  for chunkname in core_chunks[-lru.size:]:
    bbx = Bbox.from_filename(chunkname)
    bbx -= requested_bbox.minpt
    img3d = renderbuffer[ bbx.to_slices() ]
    binary = chunks.encode(
      img3d, encoding,
      meta.compressed_segmentation_block_size(mip),
      compression_params=meta.compression_params(mip),
    )
    lru[chunkname] = (encoding, binary)

def child_process_download(
    meta, cache, 
    mip, compress_cache, dest_bbox, 
    fill_missing, progress,
    location, use_shared_memory, green,
    secrets, background_color, cloudpaths
  ):
  reset_connection_pools() # otherwise multi-process hangs

  shape = list(dest_bbox.size3()) + [ meta.num_channels ]

  if use_shared_memory:
    array_like, dest_img = shm.ndarray(
      shape, dtype=meta.dtype, 
      location=location, lock=fs_lock
    )
  else:
    array_like, dest_img = shm.ndarray_fs(
      shape, dtype=meta.dtype, 
      location=location, emulate_shm=False, 
      lock=fs_lock
    )

  if background_color != 0:
      dest_img[dest_bbox.to_slices()] = background_color

  def process(src_img, src_bbox):
    shade(dest_img, dest_bbox, src_img, src_bbox)
    if progress:
      # This is not good programming practice, but
      # I could not find a clean way to do this that
      # did not result in warnings about leaked semaphores.
      # progress_queue is created in common.py:initialize_progress_queue
      # as a global for this module.
      progress_queue.put(1)

  download_chunks_threaded(
    meta, cache, None, "same", mip, cloudpaths,
    fn=process, decode_fn=decode, fill_missing=fill_missing,
    progress=False, compress_cache=compress_cache,
    green=green, secrets=secrets, background_color=background_color
  )

  array_like.close()

  return len(cloudpaths)

def download_chunk(
  meta, cache, lru, lru_encoding,
  cloudpath, mip,
  filename, fill_missing,
  enable_cache, compress_cache,
  secrets, background_color,
  decode_fn, decompress=True, locking=False,
  full_decode=True,
):
  content = None
  encoding = meta.encoding(mip)
  bbox = Bbox.from_filename(filename) # possible off by one error w/ exclusive bounds

  try:
    encoding, content = lru[filename]
  except (TypeError, KeyError):
    (file,) = CloudFiles(
      cloudpath, secrets=secrets, locking=locking
    ).get([ filename ], raw=True)
    content = file['content']

    if enable_cache:
      cache_content = next(
        compression.transcode(
          file, compress_cache, 
          in_place=(compress_cache == False)
        )
      )['content'] 
      cache.cloudfiles().put(
        path=filename, 
        content=(cache_content or b''), 
        content_type=content_type(encoding), 
        compress=compress_cache,
        raw=bool(cache_content),
      )
      if compress_cache == False:
        content = cache_content
        decompress = False
      del cache_content

    if content is not None and decompress:
      content = compression.decompress(content, file['compress'])

    if lru is not None:
      lru[filename] = (encoding, content)

  img3d = decode_fn(
    meta, filename, content, 
    fill_missing, mip, 
    background_color=background_color,
    encoding=encoding,
  )

  if lru is not None and full_decode: 
    if lru_encoding not in [ "same", encoding ]:
      content = None
      if img3d is not None:
        block_size = meta.compressed_segmentation_block_size(mip)
        if block_size is None:
          block_size = (8,8,8)

        content = chunks.encode(
          img3d, lru_encoding, 
          block_size,
          compression_params=meta.compression_params(mip),
        )
        
      lru[filename] = (lru_encoding, content)

  return img3d, bbox

def download_chunks_threaded(
    meta, cache, lru, lru_encoding, mip, cloudpaths, fn, decode_fn,
    fill_missing, progress, compress_cache,
    green=False, secrets=None, background_color=0,
    decompress=True, full_decode=True,
  ):
  """fn is the postprocess callback. decode_fn is a decode fn."""
  locations = cache.compute_data_locations(cloudpaths)
  cachedir = 'file://' + cache.path

  def process(cloudpath, filename, enable_cache, locking):
    labels, bbox = download_chunk(
      meta, cache, lru, lru_encoding, cloudpath, mip,
      filename, fill_missing,
      enable_cache, compress_cache,
      secrets, background_color,
      decode_fn, decompress, locking, full_decode,
    )
    fn(labels, bbox)

  # If there's an LRU sort the fetches so that the LRU ones are first
  # otherwise the new downloads can kick out the cached ones and make the
  # lru useless.
  are_all_lru_hits = False
  if lru is not None and lru.size > 0:
    if not isinstance(locations['remote'], list):
      locations['remote'] = list(locations['remote'])  
    locations['local'].sort(key=lambda fname: fname in lru, reverse=True)  
    locations['remote'].sort(key=lambda fname: fname in lru, reverse=True)
    are_all_lru_hits = all(( fname in lru for fname in locations['remote'] ))

  qualify = lambda fname: os.path.join(meta.key(mip), os.path.basename(fname))

  local_downloads = ( 
    partial(process, cachedir, qualify(filename), False, cache.config.cache_locking) 
    for filename in locations['local'] 
  )
  remote_downloads = ( 
    partial(process, meta.cloudpath, filename, cache.enabled, False) 
    for filename in locations['remote'] 
  )

  if progress and not isinstance(progress, str):
    progress = "Downloading"

  total = len(locations["local"]) + len(locations["remote"])
  n_threads = DEFAULT_THREADS
  if meta.path.protocol == "file" or are_all_lru_hits:
    n_threads = 0

  with tqdm(desc=progress, total=total, disable=(not progress)) as pbar:
    schedule_jobs(
      fns=local_downloads, 
      concurrency=0, 
      progress=pbar,
      total=len(locations['local']),
      green=green,
    )

    schedule_jobs(
      fns=remote_downloads,
      concurrency=n_threads, 
      progress=pbar,
      total=len(locations['remote']),
      green=green,
    )

def decode(
  meta, input_bbox, 
  content, fill_missing, 
  mip, background_color=0,
  allow_none=True,
  encoding=None,
):
  """
  Decode content from bytes into a numpy array using the 
  dataset metadata.

  If fill_missing is True, return an array filled with background_color
  if content is empty. Otherwise, raise an EmptyVolumeException
  in that case.

  Returns: ndarray
  """
  return _decode_helper(  
    chunks.decode, 
    meta, input_bbox, 
    content, fill_missing, 
    mip, background_color,
    allow_none, encoding,
  )

def decode_partial(
  meta, requested_bbox, input_bbox, 
  content, fill_missing, 
  mip, background_color=0,
  allow_none=True,
  encoding=None,
):
  """
  Decode content from a list of chunks in parallel directly into a numpy array.
  """
  return _decode_partial_helper(
    chunks.partial_decompress, 
    meta, requested_bbox, input_bbox, 
    content, fill_missing, 
    mip, background_color,
    allow_none, encoding,
  )

def decode_partial_compressed_block(
  meta, requested_bbox, input_bbox, 
  content, fill_missing, 
  mip, background_color=0,
  allow_none=True,
  encoding=None,
  chunk_relative_grid=None,
  container=None, segid_list = None
):
  """
  Decode content from a list of chunks in parallel directly into a numpy array.
  """
  _decode_partial_compressed_block_helper(
    chunks.decode_partial_compressed_block, 
    meta, requested_bbox, input_bbox, 
    content, fill_missing, 
    mip, background_color,
    allow_none, encoding,
    chunk_relative_grid,
    container,segid_list
  )

def decode_partial_parallel(
  tasks, requested_bbox, fill_missing,
  mip, background_color=0, partial_decompress_parallel=1,
  allow_none=True, l2cache_size = 0
):
  """
  Decode content from a list of chunks in parallel directly into a numpy array.
  """
  return decode_partial_parallel_helper(
    chunks.partial_decompress_in_place_parallel,
    tasks, requested_bbox, fill_missing, mip,
    background_color,
    partial_decompress_parallel=partial_decompress_parallel,
    allow_none=allow_none, l2cache_size = l2cache_size
  )

def decode_partial_parallel_helper(
  fn, tasks, requested_bbox ,fill_missing, mip, background_color, partial_decompress_parallel=1, allow_none=True, l2cache_size = 0
):
  """
  Parallel partial decode helper that handles a list of tasks.
  """
  requests = []
  
  # Collect all necessary parameters for the parallel decompression
  for job in tasks:
    meta = job['meta']
    cutout_bbox = job['cutout_bbox']
    content = job['content']
    
    content_len = len(content) if content is not None else 0
    if content_len == 0:
      if fill_missing:
        content = b''
      else:
        raise EmptyVolumeException(cutout_bbox)
    if content_len == 0 and allow_none:
      continue

    encoding = job.get('encoding')
    if encoding is None:
      encoding = meta.encoding(mip)
    
    if encoding != "compressed_segmentation":
      raise NotImplementedError(f"Parallel decompression not supported for encoding: {encoding}")

    block_size = meta.compressed_segmentation_block_size(mip)
    if block_size is None:
      block_size = (8, 8, 8)

    cutout_bbox_obj = Bbox.create(cutout_bbox)
    requested_bbox_obj = Bbox.create(requested_bbox)
    intersection = Bbox.intersection(requested_bbox_obj, cutout_bbox_obj)
    
    if intersection.empty():
      continue

    volume_size = list(cutout_bbox_obj.size3())
    volume_size += [meta.num_channels]

    requests.append({
      'encoded': content,
      'volume_size': tuple(volume_size),
      'dtype': meta.dtype,
      'chunk_start': tuple(cutout_bbox_obj.minpt),
      'chunk_end': tuple(cutout_bbox_obj.maxpt),
      'request_start': tuple(requested_bbox_obj.minpt),
      'request_end': tuple(requested_bbox_obj.maxpt),
      'block_size': tuple(block_size),
      'output_array_ptr': job['output_array_ptr'],
      'output_array_ndim': job['output_array_ndim'],
      'output_array_shape': job['output_array_shape'],
      'output_array_strides': job['output_array_strides'],
    })
  try:
    if requests:
      fn(requests, partial_decompress_parallel=partial_decompress_parallel, l2cache_size = l2cache_size)
  except Exception as error:
    print(red('Parallel decompress error: {}'.format(error)))
    raise
  return None 

def decode_binary_image(
  label, meta, input_bbox, 
  content, fill_missing, 
  mip, background_color=0, 
  allow_none=True,
  encoding=None,
):
  bbox = Bbox.create(input_bbox)
  shape = list(bbox.size3()) + [ meta.num_channels ]

  if encoding is None:
    encoding = meta.encoding(mip)

  if not content:
    if fill_missing:
      if allow_none:
        return None
      elif background_color == label:
        return np.ones(shape, dtype=bool, order="F")
      else:
        return np.zeros(shape, dtype=bool, order="F")
    else:
      raise EmptyVolumeException(input_bbox)

  # raw requires an extra decompression cycle
  # so just do the direct == comparison
  has_label = True
  if encoding != "raw":
    has_label = chunks.contains(
      content, label,
      encoding=encoding, 
      shape=shape, 
      dtype=meta.dtype, 
      block_size=meta.compressed_segmentation_block_size(mip),
    )

  if not has_label:
    if allow_none:
      return None
    else:
      return np.zeros(shape, dtype=bool, order="F")

  return _decode_helper(
    partial(chunks.decode_binary_image, label), 
    meta, input_bbox, 
    content, fill_missing, 
    mip, 
    background_color=background_color,
    allow_none=allow_none,
    encoding=encoding,
  )

def decode_unique(
  meta, input_bbox, 
  content, fill_missing, 
  mip, background_color=0,
  encoding=None,
):
  """Gets the unique labels present in a given chunk."""
  return _decode_helper(  
    chunks.labels, 
    meta, input_bbox, 
    content, fill_missing, 
    mip, background_color,
    encoding=encoding,
  )

def decode_single_voxel(
  xyz, meta, input_bbox, 
  content, fill_missing, 
  mip, background_color=0,
  encoding=None,
):
  """
  Specialized decode that for some file formats
  will be faster than regular decode when fetching
  a single voxel. Single voxel fetches are a common
  operation when e.g. people are querying the identity
  of a synapse or organelle location to build a database.
  """
  if (
    not isinstance(content, np.ndarray) 
    and content in (None, b'') 
    and fill_missing
  ):
    return np.full(
      shape=(1,1,1,1), 
      fill_value=background_color,
      dtype=meta.dtype, 
      order="F",
    )

  return _decode_helper(
    partial(chunks.read_voxel, xyz),
    meta, input_bbox,
    content, fill_missing, 
    mip, background_color,
    encoding=encoding,
  )

def _decode_helper(  
  fn, meta, input_bbox, 
  content, fill_missing, mip, 
  background_color=0,
  allow_none=True, 
  encoding=None,
):
  bbox = Bbox.create(input_bbox)
  content_len = len(content) if content is not None else 0

  if content_len == 0:
    if fill_missing:
      content = b''
    else:
      raise EmptyVolumeException(input_bbox)

  shape = list(bbox.size3()) + [ meta.num_channels ]

  if content_len == 0 and allow_none:
    return None

  if encoding is None:
    encoding = meta.encoding(mip)

  block_size = meta.compressed_segmentation_block_size(mip)
  if block_size is None:
    block_size = (8,8,8)

  try:
    return fn(
      content,
      encoding=encoding,
      shape=shape,
      dtype=meta.dtype,
      block_size=block_size,
      background_color=background_color,
    )
  except Exception as error:
    print(red('File Read Error: {} bytes, {}, {}, errors: {}'.format(
        content_len, bbox, input_bbox, error)))
    raise

def _decode_partial_helper(  
  fn, meta, requested_bbox, input_bbox, 
  content, fill_missing, mip, 
  background_color=0,
  allow_none=True, 
  encoding=None,
):
  bbox = Bbox.create(input_bbox)
  content_len = len(content) if content is not None else 0

  if content_len == 0:
    if fill_missing:
      content = b''
    else:
      raise EmptyVolumeException(input_bbox)

  shape = list(bbox.size3()) + [ meta.num_channels ]

  if content_len == 0 and allow_none:
    return None

  if encoding is None:
    encoding = meta.encoding(mip)

  block_size = meta.compressed_segmentation_block_size(mip)
  if block_size is None:
    block_size = (8,8,8)

  cutout_bbox_obj = Bbox.create(input_bbox)
  requested_bbox_obj = Bbox.create(requested_bbox)
  intersection = Bbox.intersection(requested_bbox_obj, cutout_bbox_obj)
  
  if intersection.empty():
    return None,intersection

  try:
    return fn(
      content,
      encoding=encoding,
      chunk_start=cutout_bbox_obj.minpt,
      chunk_end=cutout_bbox_obj.maxpt,
      request_start=requested_bbox_obj.minpt,
      request_end=requested_bbox_obj.maxpt,
      shape=shape,
      dtype=meta.dtype,
      block_size=block_size,
      background_color=background_color,
    ),Bbox.create(intersection)
  except Exception as error:
    print(red('File Read Error: {} bytes, {}, {}, errors: {}'.format(
        content_len, bbox, input_bbox, error)))
    raise




def _decode_partial_compressed_block_helper(  
  fn, meta, requested_bbox, input_bbox, 
  content, fill_missing, mip, 
  background_color=0,
  allow_none=True, 
  encoding=None,
  chunk_relative_grid=None,
  container=None, segid_list = None
):
  bbox = Bbox.create(input_bbox)
  content_len = len(content) if content is not None else 0

  if content_len == 0:
    if fill_missing:
      return
    else:
      raise EmptyVolumeException(input_bbox)

  shape = list(bbox.size3()) + [ meta.num_channels ]

  if content_len == 0 and allow_none:
    return

  if encoding is None:
    encoding = meta.encoding(mip)

  block_size = meta.compressed_segmentation_block_size(mip)
  if block_size is None:
    block_size = (8,8,8)

  cutout_bbox_obj = Bbox.create(input_bbox)
  requested_bbox_obj = Bbox.create(requested_bbox)
  intersection = Bbox.intersection(requested_bbox_obj, cutout_bbox_obj)
  
  if intersection.empty():
    return

  try:
    fn(
      content,
      encoding=encoding,
      chunk_start=cutout_bbox_obj.minpt,
      chunk_end=cutout_bbox_obj.maxpt,
      request_start=requested_bbox_obj.minpt,
      request_end=requested_bbox_obj.maxpt,
      shape=shape,
      dtype=meta.dtype,
      block_size=block_size,
      background_color=background_color,
      chunk_relative_grid = chunk_relative_grid,
      container=container,
      segid_list = segid_list
    )
  except Exception as error:
    print(red('File Read Error: {} bytes, {}, {}, errors: {}'.format(
        content_len, bbox, input_bbox, error)))
    raise

def unique_unsharded(
  requested_bbox, mip, 
  meta, cache, lru, lru_encoding,
  fill_missing, progress,
  parallel,
  compress, 
  green=False, secrets=None,
  background_color=0
):
  """
  Accumulate all unique labels within the requested
  bounding box.
  """
  full_bbox = requested_bbox.expand_to_chunk_size(
    meta.chunk_size(mip), offset=meta.voxel_offset(mip)
  )
  full_bbox = Bbox.clamp(full_bbox, meta.bounds(mip))
  all_chunks = set(chunknames(
    full_bbox, meta.bounds(mip), 
    meta.key(mip), meta.chunk_size(mip), 
    protocol=meta.path.protocol
  ))
  retracted_bbox = requested_bbox.shrink_to_chunk_size(
    meta.chunk_size(mip), offset=meta.voxel_offset(mip)
  )
  retracted_bbox = Bbox.clamp(retracted_bbox, meta.bounds(mip))
  core_chunks = set(chunknames(
    retracted_bbox, 
    meta.bounds(mip), meta.key(mip), meta.chunk_size(mip),
    protocol=meta.path.protocol,
  ))
  shell_chunks = all_chunks.difference(core_chunks)

  shape = list(requested_bbox.size3()) + [ meta.num_channels ]

  compress_cache = should_compress(meta.encoding(mip), compress, cache, iscache=True)

  all_labels = set()
  def process_core(labels, bbox):
    nonlocal all_labels
    if labels is None:
      all_labels |= set([ background_color ])
    else:
      all_labels |= set(labels)

  def process_shell(labels, bbox):
    nonlocal all_labels
    nonlocal requested_bbox
    if labels is None:
      all_labels |= set([ background_color ])
    else:
      crop_bbox = Bbox.intersection(requested_bbox, bbox)
      crop_bbox -= bbox.minpt
      labels = labels[ crop_bbox.to_slices() ]
      all_labels |= set(fastremap.unique(labels))

  # If there's an LRU sort the fetches so that the LRU ones are first
  # otherwise the new downloads can kick out the cached ones and make the
  # lru useless.
  if lru.size > 0:
    core_chunks = list(core_chunks)
    shell_chunks = list(shell_chunks)
    core_chunks.sort(key=lambda fname: fname in lru, reverse=True)
    shell_chunks.sort(key=lambda fname: fname in lru, reverse=True)

  download_chunks_threaded(
    meta, cache, lru, lru_encoding, mip, core_chunks, 
    fn=process_core, decode_fn=decode_unique, fill_missing=fill_missing,
    progress=progress, compress_cache=compress_cache, 
    green=green, secrets=secrets, background_color=background_color,
    full_decode=False,
  )

  if len(shell_chunks) > 0:
    download_chunks_threaded(
      meta, cache, lru, lru_encoding, mip, shell_chunks, 
      fn=process_shell, decode_fn=decode, fill_missing=fill_missing,
      progress=progress, compress_cache=compress_cache, 
      green=green, secrets=secrets, background_color=background_color,
      full_decode=False,
    )

  return all_labels

def unique_sharded(
  requested_bbox, mip,
  meta, cache, 
  lru, lru_encoding, spec,
  compress, progress,
  fill_missing, background_color
):
  """
  Accumulate all unique labels within the requested
  bounding box.
  """
  full_bbox = requested_bbox.expand_to_chunk_size(
    meta.chunk_size(mip), offset=meta.voxel_offset(mip)
  )
  full_bbox = Bbox.clamp(full_bbox, meta.bounds(mip))
  core_bbox = requested_bbox.shrink_to_chunk_size(
    meta.chunk_size(mip), offset=meta.voxel_offset(mip)
  )
  core_bbox = Bbox.clamp(core_bbox, meta.bounds(mip))

  compress_cache = should_compress(meta.encoding(mip), compress, cache, iscache=True)

  chunk_size = meta.chunk_size(mip)
  grid_size = np.ceil(meta.bounds(mip).size3() / chunk_size).astype(np.uint32)

  reader = sharding.ShardReader(meta, cache, spec)
  bounds = meta.bounds(mip)

  all_gpts = list(gridpoints(full_bbox, bounds, chunk_size))
  core_gpts = list(gridpoints(core_bbox, bounds, chunk_size))

  code_map = {}
  all_morton_codes = compressed_morton_code(all_gpts, grid_size)
  for gridpoint, morton_code in zip(all_gpts, all_morton_codes):
    cutout_bbox = Bbox(
      bounds.minpt + gridpoint * chunk_size,
      min2(bounds.minpt + (gridpoint + 1) * chunk_size, bounds.maxpt)
    )
    code_map[morton_code] = cutout_bbox

  lru_codes = set([ code for code in all_morton_codes if code in lru ])
  lru_chunkdata = { code: lru[code] for code in lru_codes }
  
  core_morton_codes = set(compressed_morton_code(core_gpts, grid_size))
  io_core_morton_codes = core_morton_codes - lru_codes
  lru_core_morton_codes = core_morton_codes.intersection(lru_codes)

  def iterate_core():
    for mcs in sip(io_core_morton_codes, 10000):
      core_chunkdata = reader.get_data(mcs, meta.key(mip), progress=progress)
      for zcode, chunkdata in core_chunkdata.items():
        yield (zcode, meta.encoding(mip), chunkdata)
        lru[zcode] = (meta.encoding(mip), chunkdata)
    for code in lru_core_morton_codes:
      yield (code, *lru_chunkdata[code])
      del lru_chunkdata[code]

  all_labels = set()
  for zcode, data_encoding, chunkdata in iterate_core():
    cutout_bbox = code_map[zcode]
    labels = decode_unique(
      meta, cutout_bbox, 
      chunkdata, fill_missing, mip,
      background_color=background_color,
      encoding=data_encoding,
    )
    all_labels |= set(labels)

  shell_morton_codes = set(all_morton_codes) - set(core_morton_codes)
  io_shell_morton_codes = shell_morton_codes - lru_codes
  lru_shell_morton_codes = shell_morton_codes.intersection(lru_codes)

  def iterate_shell():
    shell_chunkdata = reader.get_data(io_shell_morton_codes, meta.key(mip), progress=progress)
    for zcode, chunkdata in shell_chunkdata.items():
      yield (zcode, meta.encoding(mip), chunkdata)
      lru[zcode] = (meta.encoding(mip), chunkdata)
    for code in lru_shell_morton_codes:
      yield (code, *lru_chunkdata[code])
      del lru_chunkdata[code]    

  for zcode, data_encoding, chunkdata in iterate_shell():
    cutout_bbox = code_map[zcode]
    labels = decode(
      meta, cutout_bbox, 
      chunkdata, fill_missing, mip,
      background_color=background_color,
      encoding=data_encoding,
    )
    if labels is None:
      all_labels |= set([ background_color ])
    else:
      crop_bbox = Bbox.intersection(requested_bbox, cutout_bbox)
      crop_bbox -= cutout_bbox.minpt
      labels = fastremap.unique(labels[ crop_bbox.to_slices() ])
      all_labels |= set(labels)

  return all_labels


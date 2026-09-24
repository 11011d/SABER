from dataclasses import dataclass
import numbers
from typing import Optional


@dataclass(frozen=True)
class SaberOptions:
  """Validated SABER read optimization settings.

  Public selector:
    saber=None/False: do not enable partial/compressed-block reads.
    saber=<int>: dense partial decompression with that parallelism value.
    saber=True: compressed-block container reads.

  Derived levels:
    0: disabled, keep the CloudVolume path unchanged.
    1: optimized local cache reads and optional debug info only.
    2: dense partial decompression.
    3: compressed-block container reads.
  """

  level: int = 0
  cache_thread: Optional[int] = None
  partial_decompress_parallel: Optional[int] = None
  use_compressed_block: bool = False
  debug: bool = False

  @classmethod
  def disabled(cls):
    return cls(level=0)

  @classmethod
  def from_params(cls, saber=None, cache_thread: Optional[int] = None, saber_debug: bool = False):
    level = 0
    partial_decompress_parallel = None
    use_compressed_block = False

    if saber is True:
      level = 3
      use_compressed_block = True
    elif saber is None or saber is False:
      if cache_thread is not None:
        level = 1
    elif isinstance(saber, numbers.Integral):
      level = 2
      partial_decompress_parallel = int(saber)
    else:
      raise ValueError("saber must be True, False, None, or an integer.")

    opts = cls(
      level=level,
      cache_thread=cache_thread,
      partial_decompress_parallel=partial_decompress_parallel,
      use_compressed_block=use_compressed_block,
      debug=bool(saber_debug),
    )
    opts.validate()
    return opts

  @property
  def enabled(self):
    return self.level > 0

  def validate(self):
    if self.level not in (0, 1, 2, 3):
      raise ValueError(f"SABER level must be 0, 1, 2, or 3. Got: {self.level}")

    if self.cache_thread is not None and int(self.cache_thread) < 0:
      raise ValueError("cache_thread must be None or >= 0.")

    if self.level == 0:
      if (
        self.cache_thread is not None
        or self.partial_decompress_parallel is not None
        or self.use_compressed_block
      ):
        raise ValueError("SABER is disabled but SABER-only options were provided.")

    if self.level == 1:
      if self.use_compressed_block or self.partial_decompress_parallel is not None:
        raise ValueError("SABER level 1 only enables cache read optimization/debug info.")

    if self.level == 2 and self.use_compressed_block:
      raise ValueError("SABER partial decompression and compressed blocks are mutually exclusive.")

    if self.level == 3 and self.partial_decompress_parallel is not None:
      raise ValueError("SABER compressed-block reads do not run dense partial decompression.")

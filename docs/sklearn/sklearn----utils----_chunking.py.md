# `D:\src\scipysrc\scikit-learn\sklearn\utils\_chunking.py`

```
import warnings
from itertools import islice
from numbers import Integral

import numpy as np

from .._config import get_config
from ._param_validation import Interval, validate_params


def chunk_generator(gen, chunksize):
    """Chunk generator, ``gen`` into lists of length ``chunksize``. The last
    chunk may have a length less than ``chunksize``."""
    # 无限循环，生成长度为 chunksize 的列表
    while True:
        # 从 gen 中获取长度为 chunksize 的片段
        chunk = list(islice(gen, chunksize))
        # 如果片段非空，生成该片段
        if chunk:
            yield chunk
        else:
            return


@validate_params(
    {
        "n": [Interval(Integral, 1, None, closed="left")],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "min_batch_size": [Interval(Integral, 0, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def gen_batches(n, batch_size, *, min_batch_size=0):
    """Generator to create slices containing `batch_size` elements from 0 to `n`.

    The last slice may contain less than `batch_size` elements, when
    `batch_size` does not divide `n`.

    Parameters
    ----------
    n : int
        Size of the sequence.
    batch_size : int
        Number of elements in each batch.
    min_batch_size : int, default=0
        Minimum number of elements in each batch.

    Yields
    ------
    slice of `batch_size` elements

    See Also
    --------
    gen_even_slices: Generator to create n_packs slices going up to n.

    Examples
    --------
    >>> from sklearn.utils import gen_batches
    >>> list(gen_batches(7, 3))
    [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
    >>> list(gen_batches(6, 3))
    [slice(0, 3, None), slice(3, 6, None)]
    >>> list(gen_batches(2, 3))
    [slice(0, 2, None)]
    >>> list(gen_batches(7, 3, min_batch_size=0))
    [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
    >>> list(gen_batches(7, 3, min_batch_size=2))
    [slice(0, 3, None), slice(3, 7, None)]
    """
    # 初始化起始索引为 0
    start = 0
    # 循环生成切片，直到无法再生成完整的 batch_size 长度的切片为止
    for _ in range(int(n // batch_size)):
        # 计算切片的结束索引
        end = start + batch_size
        # 如果添加 min_batch_size 后的结束索引超过 n，则跳过该切片
        if end + min_batch_size > n:
            continue
        # 生成切片并 yield
        yield slice(start, end)
        # 更新起始索引
        start = end
    # 若剩余元素不足一个 batch_size，则生成最后一个切片
    if start < n:
        yield slice(start, n)


@validate_params(
    {
        "n": [Interval(Integral, 1, None, closed="left")],
        "n_packs": [Interval(Integral, 1, None, closed="left")],
        "n_samples": [Interval(Integral, 1, None, closed="left"), None],
    },
    prefer_skip_nested_validation=True,
)
def gen_even_slices(n, n_packs, *, n_samples=None):
    """Generator to create `n_packs` evenly spaced slices going up to `n`.

    If `n_packs` does not divide `n`, except for the first `n % n_packs`
    slices, remaining slices may contain fewer elements.

    Parameters
    ----------
    n : int
        Size of the sequence.
    n_packs : int
        Number of slices to generate.
    """
    # 确定每个切片的基础长度
    base_length = n // n_packs
    # 剩余部分的长度
    remainder = n % n_packs
    # 初始化起始索引
    start = 0
    # 循环生成切片
    for pack_idx in range(n_packs):
        # 计算当前切片的结束索引
        if pack_idx < remainder:
            end = start + base_length + 1
        else:
            end = start + base_length
        # 生成切片并 yield
        yield slice(start, end)
        # 更新起始索引
        start = end
    start = 0
    # 初始化起始索引为`
# 初始化起始索引为0
start = 0
# 循环生成器所需的包数
for pack_num in range(n_packs):
    # 每个包的预期元素数
    this_n = n // n_packs
    # 如果当前包需要多一个元素（余数部分）
    if pack_num < n % n_packs:
        this_n += 1
    # 如果当前包的元素数大于0
    if this_n > 0:
        # 计算结束索引
        end = start + this_n
        # 如果指定了样本数限制，将结束索引修正为样本数或当前结束索引的较小值
        if n_samples is not None:
            end = min(n_samples, end)
        # 生成一个表示从起始到结束的切片对象，并返回给调用者
        yield slice(start, end, None)
        # 更新起始索引为当前结束索引，以便下一个切片的起始位置
        start = end
# 计算在给定的工作内存中可以处理多少行数据。

def get_chunk_n_rows(row_bytes, *, max_n_rows=None, working_memory=None):
    """Calculate how many rows can be processed within `working_memory`.

    Parameters
    ----------
    row_bytes : int
        The expected number of bytes of memory that will be consumed
        during the processing of each row.
    max_n_rows : int, default=None
        The maximum return value.
    working_memory : int or float, default=None
        The number of rows to fit inside this number of MiB will be
        returned. When None (default), the value of
        ``sklearn.get_config()['working_memory']`` is used.

    Returns
    -------
    int
        The number of rows which can be processed within `working_memory`.

    Warns
    -----
    Issues a UserWarning if `row_bytes exceeds `working_memory` MiB.
    """

    # 如果未指定工作内存，则使用默认的配置中的工作内存值
    if working_memory is None:
        working_memory = get_config()["working_memory"]

    # 计算在工作内存中可以处理的行数，单位为MiB
    chunk_n_rows = int(working_memory * (2**20) // row_bytes)
    
    # 如果指定了最大行数，则取计算结果和最大行数的较小值
    if max_n_rows is not None:
        chunk_n_rows = min(chunk_n_rows, max_n_rows)
    
    # 如果计算出的行数小于1，发出警告并将行数设置为1
    if chunk_n_rows < 1:
        warnings.warn(
            "Could not adhere to working_memory config. "
            "Currently %.0fMiB, %.0fMiB required."
            % (working_memory, np.ceil(row_bytes * 2**-20))
        )
        chunk_n_rows = 1
    
    # 返回可以在工作内存中处理的行数
    return chunk_n_rows
```
# `D:\src\scipysrc\pandas\pandas\_libs\internals.pyi`

```
from typing import (
    Iterator,
    Sequence,
    final,
    overload,
)
import weakref

import numpy as np

from pandas._typing import (
    ArrayLike,
    Self,
    npt,
)

from pandas import Index
from pandas.core.internals.blocks import Block as B

# 定义一个函数slice_len，计算切片对象的长度
def slice_len(slc: slice, objlen: int = ...) -> int: ...

# 定义一个函数get_concat_blkno_indexers，返回索引器列表
def get_concat_blkno_indexers(
    blknos_list: list[npt.NDArray[np.intp]],
) -> list[tuple[npt.NDArray[np.intp], BlockPlacement]]: ...

# 定义一个函数get_blkno_indexers，返回块编号索引器列表
def get_blkno_indexers(
    blknos: np.ndarray,  # int64_t[:]
    group: bool = ...,
) -> list[tuple[int, slice | np.ndarray]]: ...

# 定义一个函数get_blkno_placements，生成块编号和块位置的迭代器
def get_blkno_placements(
    blknos: np.ndarray,
    group: bool = ...,
) -> Iterator[tuple[int, BlockPlacement]]: ...

# 定义一个函数update_blklocs_and_blknos，更新块位置和块编号数组
def update_blklocs_and_blknos(
    blklocs: npt.NDArray[np.intp],
    blknos: npt.NDArray[np.intp],
    loc: int,
    nblocks: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...

# 使用@final修饰的类BlockPlacement，表示其为最终类，不能被继承
@final
class BlockPlacement:
    # 初始化方法，接受int、slice或者np.ndarray作为参数
    def __init__(self, val: int | slice | np.ndarray) -> None: ...

    # 返回索引器，类型为np.ndarray或slice
    @property
    def indexer(self) -> np.ndarray | slice: ...

    # 返回值作为np.ndarray
    @property
    def as_array(self) -> np.ndarray: ...

    # 返回值作为slice类型
    @property
    def as_slice(self) -> slice: ...

    # 判断是否类似于slice类型
    @property
    def is_slice_like(self) -> bool: ...

    # 重载方法，根据loc参数返回BlockPlacement对象
    @overload
    def __getitem__(
        self, loc: slice | Sequence[int] | npt.NDArray[np.intp]
    ) -> BlockPlacement: ...

    # 重载方法，根据loc参数返回int类型的值
    @overload
    def __getitem__(self, loc: int) -> int: ...

    # 迭代器，返回int类型的值
    def __iter__(self) -> Iterator[int]: ...

    # 返回长度，即元素数量
    def __len__(self) -> int: ...

    # 删除指定位置的元素，返回BlockPlacement对象
    def delete(self, loc) -> BlockPlacement: ...

    # 添加其他BlockPlacement对象，返回BlockPlacement对象
    def add(self, other) -> BlockPlacement: ...

    # 添加多个BlockPlacement对象，返回BlockPlacement对象
    def append(self, others: list[BlockPlacement]) -> BlockPlacement: ...

    # 为unstack操作创建块的位置数组，返回npt.NDArray[np.intp]
    def tile_for_unstack(self, factor: int) -> npt.NDArray[np.intp]: ...

# 定义一个类Block
class Block:
    _mgr_locs: BlockPlacement  # 块的位置信息
    ndim: int  # 维度数
    values: ArrayLike  # 类型为ArrayLike的值
    refs: BlockValuesRefs  # 块引用信息

    # 初始化方法，接受ArrayLike、BlockPlacement、int和BlockValuesRefs或None作为参数
    def __init__(
        self,
        values: ArrayLike,
        placement: BlockPlacement,
        ndim: int,
        refs: BlockValuesRefs | None = ...,
    ) -> None: ...

    # 根据切片对象对块的行进行切片，返回自身对象
    def slice_block_rows(self, slicer: slice) -> Self: ...

# 定义一个类BlockManager
class BlockManager:
    blocks: tuple[B, ...]  # 块对象元组
    axes: list[Index]  # 索引列表
    _known_consolidated: bool  # 已知是否已合并的标志
    _is_consolidated: bool  # 是否已合并的标志
    _blknos: np.ndarray  # 块编号数组
    _blklocs: np.ndarray  # 块位置数组

    # 初始化方法，接受块对象元组、索引列表和verify_integrity参数作为参数
    def __init__(
        self, blocks: tuple[B, ...], axes: list[Index], verify_integrity=...
    ) -> None: ...

    # 根据给定的切片对象获取一个新的BlockManager对象，返回自身对象
    def get_slice(self, slobj: slice, axis: int = ...) -> Self: ...

    # 重新构建块编号数组和块位置数组，不返回任何内容
    def _rebuild_blknos_and_blklocs(self) -> None: ...

# 定义一个类BlockValuesRefs
class BlockValuesRefs:
    referenced_blocks: list[weakref.ref]  # 引用块的弱引用列表

    # 初始化方法，接受Block或None作为参数
    def __init__(self, blk: Block | None = ...) -> None: ...

    # 添加块引用，不返回任何内容
    def add_reference(self, blk: Block) -> None: ...

    # 添加索引引用，不返回任何内容
    def add_index_reference(self, index: Index) -> None: ...

    # 判断是否有引用存在，返回布尔值
    def has_reference(self) -> bool: ...
```
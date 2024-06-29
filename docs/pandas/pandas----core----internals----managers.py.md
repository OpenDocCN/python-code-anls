# `D:\src\scipysrc\pandas\pandas\core\internals\managers.py`

```
# 从未来版本中导入注释的特性，用于类型检查
from __future__ import annotations

# 导入抽象基类相关模块
from collections.abc import (
    Callable,     # 可调用对象的抽象基类
    Hashable,     # 可哈希对象的抽象基类
    Sequence,     # 序列对象的抽象基类
)
import itertools   # 迭代器相关工具函数
from typing import (  # 导入类型相关声明
    TYPE_CHECKING,    # 类型检查标志
    Any,              # 任意类型
    Literal,          # 字面量类型
    NoReturn,         # 表示函数没有返回值
    cast,             # 类型强制转换函数
    final,            # 类的最终版本修饰符
)
import warnings     # 警告相关模块

import numpy as np   # 导入 NumPy 库

# 从 Pandas 配置中导入获取选项的函数
from pandas._config.config import get_option

# 从 Pandas 私有库中导入算法和内部对象
from pandas._libs import (
    algos as libalgos,         # 算法函数集合
    internals as libinternals, # 内部实现相关函数
    lib,                       # 通用库函数
)
# 从 Pandas 内部实现模块中导入块的布局和引用
from pandas._libs.internals import (
    BlockPlacement,     # 块的放置方式
    BlockValuesRefs,    # 块的值和引用
)
# 从 Pandas 时间序列库中导入时间戳对象
from pandas._libs.tslibs import Timestamp

# 从 Pandas 错误模块中导入异常类
from pandas.errors import (
    AbstractMethodError,   # 抽象方法错误
    PerformanceWarning,    # 性能警告
)
# 从 Pandas 工具模块中导入只读缓存装饰器
from pandas.util._decorators import cache_readonly
# 从 Pandas 工具模块中导入栈级别查找异常
from pandas.util._exceptions import find_stack_level
# 从 Pandas 工具模块中导入布尔类型参数验证函数
from pandas.util._validators import validate_bool_kwarg

# 从 Pandas 核心数据类型转换模块中导入常用函数
from pandas.core.dtypes.cast import (
    find_common_type,           # 查找公共数据类型
    infer_dtype_from_scalar,    # 从标量推断数据类型
    np_can_hold_element,        # NumPy 是否能容纳元素
)
# 从 Pandas 核心数据类型通用模块中导入函数
from pandas.core.dtypes.common import (
    ensure_platform_int,        # 确保平台整数
    is_1d_only_ea_dtype,       # 是否为一维扩展数组数据类型
    is_list_like,              # 是否为类列表对象
)
# 从 Pandas 核心数据类型模块中导入时间序列、扩展和稀疏数据类型
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,    # 带时区的日期时间数据类型
    ExtensionDtype,     # 扩展数据类型
    SparseDtype,        # 稀疏数据类型
)
# 从 Pandas 核心数据类型通用模块中导入数据框和系列的抽象基类
from pandas.core.dtypes.generic import (
    ABCDataFrame,    # 数据框的抽象基类
    ABCSeries,       # 系列的抽象基类
)
# 从 Pandas 核心数据类型缺失值模块中导入数组比较和缺失值判断函数
from pandas.core.dtypes.missing import (
    array_equals,    # 数组比较函数
    isna,            # 是否为缺失值
)

# 导入 Pandas 核心算法模块的函数
import pandas.core.algorithms as algos
# 从 Pandas 核心数组模块中导入日期时间数组对象
from pandas.core.arrays import DatetimeArray
# 从 Pandas 核心数组混合模块中导入基于 NumPy 的扩展数组混合类
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
# 从 Pandas 核心基础模块中导入 Pandas 对象基类
from pandas.core.base import PandasObject
# 从 Pandas 核心构建模块中导入日期时间包装、数组提取函数
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,   # 如果类似日期时间则确保包装
    extract_array,                    # 提取数组
)
# 从 Pandas 核心索引器模块中导入可能的索引转换函数
from pandas.core.indexers import maybe_convert_indices
# 从 Pandas 核心索引 API 模块中导入索引、默认索引和确保索引函数
from pandas.core.indexes.api import (
    Index,             # 索引对象
    default_index,     # 默认索引
    ensure_index,      # 确保索引
)
# 从 Pandas 核心内部块模块中导入块相关函数和类
from pandas.core.internals.blocks import (
    Block,                 # 块对象
    NumpyBlock,            # NumPy 块对象
    ensure_block_shape,    # 确保块的形状
    extend_blocks,         # 扩展块
    get_block_type,        # 获取块类型
    maybe_coerce_values,   # 可能强制转换值
    new_block,             # 创建新块对象
    new_block_2d,          # 创建新的二维块对象
)
# 从 Pandas 核心内部操作模块中导入块操作函数
from pandas.core.internals.ops import (
    blockwise_all,       # 块级别全部操作
    operate_blockwise,   # 块级别操作
)

if TYPE_CHECKING:
    from collections.abc import Generator   # 导入生成器抽象基类

    # 从 Pandas 类型注解模块中导入类型别名
    from pandas._typing import (
        ArrayLike,             # 类数组对象
        AxisInt,               # 轴整数类型
        DtypeObj,              # 数据类型对象
        QuantileInterpolation, # 分位数插值
        Self,                  # 自引用类型
        Shape,                 # 形状类型
        npt,                   # NumPy 类型别名
    )

    # 从 Pandas 扩展数组接口模块中导入扩展数组
    from pandas.api.extensions import ExtensionArray


def interleaved_dtype(dtypes: list[DtypeObj]) -> DtypeObj | None:
    """
    Find the common dtype for `blocks`.

    Parameters
    ----------
    dtypes : list of DtypeObj
        List of data types.

    Returns
    -------
    dtype : np.dtype, ExtensionDtype, or None
        Common data type found across input `dtypes`. Returns None if `dtypes` is empty.
    """
    if not len(dtypes):
        return None

    return find_common_type(dtypes)


def ensure_np_dtype(dtype: DtypeObj) -> np.dtype:
    """
    Ensure the input dtype is converted to a NumPy dtype.

    Parameters
    ----------
    dtype : DtypeObj
        Input data type object.

    Returns
    -------
    np.dtype
        Converted NumPy dtype.
    """
    # TODO: https://github.com/pandas-dev/pandas/issues/22791
    # Give EAs some input on what happens here. Sparse needs this.
    if isinstance(dtype, SparseDtype):
        dtype = dtype.subtype
        dtype = cast(np.dtype, dtype)
    elif isinstance(dtype, ExtensionDtype):
        dtype = np.dtype("object")
    # 如果传入的数据类型为字符串类型（str），则将数据类型修改为对象类型（"object"）
    elif dtype == np.dtype(str):
        dtype = np.dtype("object")
    # 返回修正后的数据类型
    return dtype
class BaseBlockManager(PandasObject):
    """
    Core internal data structure to implement DataFrame, Series, etc.

    Manage a bunch of labeled 2D mixed-type ndarrays. Essentially it's a
    lightweight blocked set of labeled data to be manipulated by the DataFrame
    public API class

    Attributes
    ----------
    shape
        返回数据结构的形状，即各维度的大小
    ndim
        返回数据结构的维度数
    axes
        返回数据结构的轴标签列表
    values
        返回数据结构中所有数据块的值
    items
        返回数据结构中的标签列表

    Methods
    -------
    set_axis(axis, new_labels)
        设置指定轴的新标签
    copy(deep=True)
        复制当前对象，可选择是否深复制

    get_dtypes
        获取数据结构中所有数据块的数据类型

    apply(func, axes, block_filter_fn)
        对数据结构中的数据块应用函数

    get_bool_data
        获取数据结构中布尔类型数据块

    get_numeric_data
        获取数据结构中数值类型数据块

    get_slice(slice_like, axis)
        根据给定的切片对象和轴返回对应的数据块

    get(label)
        根据标签获取数据结构中的数据块

    iget(loc)
        根据位置索引获取数据结构中的数据块

    take(indexer, axis)
        根据索引数组从指定轴上获取数据块

    reindex_axis(new_labels, axis)
        根据新标签重新索引指定轴

    reindex_indexer(new_labels, indexer, axis)
        根据新的索引标签和索引数组重新索引指定轴

    delete(label)
        删除指定标签对应的数据块

    insert(loc, label, value)
        在指定位置插入新的标签和值

    set(label, value)
        设置指定标签对应的数据块的值

    Parameters
    ----------
    blocks: Sequence of Block
        数据结构中的数据块序列
    axes: Sequence of Index
        数据结构的轴标签序列
    verify_integrity: bool, default True
        是否验证数据结构的完整性

    Notes
    -----
    This is *not* a public API class
    """

    __slots__ = ()

    _blknos: npt.NDArray[np.intp]
    _blklocs: npt.NDArray[np.intp]
    blocks: tuple[Block, ...]
    axes: list[Index]

    @property
    def ndim(self) -> int:
        """
        返回数据结构的维度数（未实现）
        """
        raise NotImplementedError

    _known_consolidated: bool
    _is_consolidated: bool

    def __init__(self, blocks, axes, verify_integrity: bool = True) -> None:
        """
        初始化方法（未实现）
        """
        raise NotImplementedError

    @final
    def __len__(self) -> int:
        """
        返回数据结构的长度，即标签列表的长度
        """
        return len(self.items)

    @property
    def shape(self) -> Shape:
        """
        返回数据结构的形状，即各维度的大小
        """
        return tuple(len(ax) for ax in self.axes)

    @classmethod
    def from_blocks(cls, blocks: list[Block], axes: list[Index]) -> Self:
        """
        从数据块和轴标签列表创建实例（未实现）
        """
        raise NotImplementedError

    @property
    def blknos(self) -> npt.NDArray[np.intp]:
        """
        返回一个数组，指示每列所在的数据块编号

        blknos[i] 表示包含第 i 列的数据块在 self.blocks 中的索引

        blklocs[i] 表示在 self.blocks[self.blknos[i]] 中对应的列索引
        """
        if self._blknos is None:
            # 注意：这些属性可能会被其他 BlockManager 方法修改
            self._rebuild_blknos_and_blklocs()

        return self._blknos

    @property
    def blklocs(self) -> npt.NDArray[np.intp]:
        """
        参见 blknos.__doc__
        """
        if self._blklocs is None:
            # 注意：这些属性可能会被其他 BlockManager 方法修改
            self._rebuild_blknos_and_blklocs()

        return self._blklocs
    def make_empty(self, axes=None) -> Self:
        """返回一个空的 BlockManager，其 items 轴长度为 0"""
        if axes is None:
            axes = [default_index(0)] + self.axes[1:]

        # 如果是一维的情况下，保留 dtype
        if self.ndim == 1:
            assert isinstance(self, SingleBlockManager)  # 用于类型检查
            blk = self.blocks[0]
            arr = blk.values[:0]  # 创建一个空的数组
            bp = BlockPlacement(slice(0, 0))
            nb = blk.make_block_same_class(arr, placement=bp)  # 创建与原块相同类型的新块
            blocks = [nb]
        else:
            blocks = []
        return type(self).from_blocks(blocks, axes)

    def __nonzero__(self) -> bool:
        return True

    # Python3 兼容性
    __bool__ = __nonzero__

    def set_axis(self, axis: AxisInt, new_labels: Index) -> None:
        # 调用者负责确保 new_labels 是 Index 对象
        self._validate_set_axis(axis, new_labels)
        self.axes[axis] = new_labels

    @final
    def _validate_set_axis(self, axis: AxisInt, new_labels: Index) -> None:
        # 调用者负责确保 new_labels 是 Index 对象
        old_len = len(self.axes[axis])
        new_len = len(new_labels)

        if axis == 1 and len(self.items) == 0:
            # 如果在没有列的 DataFrame 上设置索引，可以改变长度
            pass

        elif new_len != old_len:
            raise ValueError(
                f"Length mismatch: Expected axis has {old_len} elements, new "
                f"values have {new_len} elements"
            )

    @property
    def is_single_block(self) -> bool:
        # 假设是二维的；由 SingleBlockManager 覆盖
        return len(self.blocks) == 1

    @property
    def items(self) -> Index:
        return self.axes[0]

    def _has_no_reference(self, i: int) -> bool:
        """
        检查第 `i` 列是否有引用。
        （即它是否引用另一个数组或者自身被引用）
        如果列没有引用，则返回 True。
        """
        blkno = self.blknos[i]
        return self._has_no_reference_block(blkno)

    def _has_no_reference_block(self, blkno: int) -> bool:
        """
        检查第 `blkno` 块是否有引用。
        （即它是否引用另一个数组或者自身被引用）
        如果块没有引用，则返回 True。
        """
        return not self.blocks[blkno].refs.has_reference()

    def add_references(self, mgr: BaseBlockManager) -> None:
        """
        将一个管理器的引用添加到另一个管理器中。
        我们假设两个管理器具有相同的块结构。
        """
        if len(self.blocks) != len(mgr.blocks):
            # 如果块结构发生变化，则做了一次复制
            return
        for i, blk in enumerate(self.blocks):
            blk.refs = mgr.blocks[i].refs
            blk.refs.add_reference(blk)
    # 检查两个不同块管理器中的两个块是否引用相同的基础值
    def references_same_values(self, mgr: BaseBlockManager, blkno: int) -> bool:
        # 获取当前对象中索引为blkno的块
        blk = self.blocks[blkno]
        # 遍历mgr对象中索引为blkno的块引用的所有块，检查是否有引用与blk相同的情况
        return any(blk is ref() for ref in mgr.blocks[blkno].refs.referenced_blocks)

    # 获取所有块的数据类型，并按索引获取特定块的数据类型
    def get_dtypes(self) -> npt.NDArray[np.object_]:
        dtypes = np.array([blk.dtype for blk in self.blocks], dtype=object)
        return dtypes.take(self.blknos)

    # 返回所有块的值组成的列表，用于兼容性测试，不建议在实际代码中使用
    @property
    def arrays(self) -> list[ArrayLike]:
        """
        Quick access to the backing arrays of the Blocks.

        Only for compatibility with ArrayManager for testing convenience.
        Not to be used in actual code, and return value is not the same as the
        ArrayManager method (list of 1D arrays vs iterator of 2D ndarrays / 1D EAs).

        Warning! The returned arrays don't handle Copy-on-Write, so this should
        be used with caution (only in read-mode).
        """
        # TODO: Deprecate, usage in Dask
        # https://github.com/dask/dask/blob/484fc3f1136827308db133cd256ba74df7a38d8c/dask/base.py#L1312
        return [blk.values for blk in self.blocks]

    # 返回对象的字符串表示形式，包括类型名称、各轴信息和每个块的详细信息
    def __repr__(self) -> str:
        output = type(self).__name__
        for i, ax in enumerate(self.axes):
            if i == 0:
                output += f"\nItems: {ax}"
            else:
                output += f"\nAxis {i}: {ax}"

        for block in self.blocks:
            output += f"\n{block}"
        return output

    # 由子类实现的方法，用于检查列值是否相等（假定形状和索引已经被检查过）
    def _equal_values(self, other: Self) -> bool:
        raise AbstractMethodError(self)

    # 判断当前对象是否与另一个对象相等
    @final
    def equals(self, other: object) -> bool:
        # 若other不是当前对象的实例，则返回False
        if not isinstance(other, type(self)):
            return False

        # 检查各对象的轴是否相等
        self_axes, other_axes = self.axes, other.axes
        if len(self_axes) != len(other_axes):
            return False
        if not all(ax1.equals(ax2) for ax1, ax2 in zip(self_axes, other_axes)):
            return False

        # 调用_equal_values方法检查列值是否相等
        return self._equal_values(other)

    # 应用函数f到当前对象的数据块，可选地使用align_keys对齐关键字
    def apply(
        self,
        f,
        align_keys: list[str] | None = None,
        **kwargs,
    ) -> Self:
        """
        Iterate over the blocks, collect and create a new BlockManager.

        Parameters
        ----------
        f : str or callable
            Name of the Block method to apply.
        align_keys: List[str] or None, default None
            List of keys for alignment.
        **kwargs
            Keywords to pass to `f`

        Returns
        -------
        BlockManager
            A new BlockManager object.

        """
        # Ensure that "filter" is not passed as a keyword argument
        assert "filter" not in kwargs

        # Initialize align_keys to an empty list if it's None
        align_keys = align_keys or []
        # Initialize an empty list to store result blocks
        result_blocks: list[Block] = []

        # Loop through each block in self.blocks
        for b in self.blocks:
            # If align_keys is not empty, align the arguments
            if aligned_args := {k: kwargs[k] for k in align_keys}:
                for k, obj in aligned_args.items():
                    # Check if obj is a Series or DataFrame
                    if isinstance(obj, (ABCSeries, ABCDataFrame)):
                        # Ensure alignment of obj.axes[-1] with self.items
                        if obj.ndim == 1:
                            kwargs[k] = obj.iloc[b.mgr_locs.indexer]._values
                        else:
                            kwargs[k] = obj.iloc[:, b.mgr_locs.indexer]._values
                    else:
                        # Use obj as an ndarray for alignment
                        kwargs[k] = obj[b.mgr_locs.indexer]

            # Apply the function f to block b with keyword arguments kwargs
            if callable(f):
                applied = b.apply(f, **kwargs)
            else:
                applied = getattr(b, f)(**kwargs)

            # Extend result_blocks with the applied result
            result_blocks = extend_blocks(applied, result_blocks)

        # Create a new BlockManager object from result_blocks and self.axes
        out = type(self).from_blocks(result_blocks, self.axes)
        return out
    def replace(self, to_replace, value, inplace: bool) -> Self:
        # 将 inplace 参数转换为布尔值，确保其有效性
        inplace = validate_bool_kwarg(inplace, "inplace")
        # 确保 to_replace 不是类列表的对象
        assert not lib.is_list_like(to_replace)
        # 确保 value 不是类列表的对象
        assert not lib.is_list_like(value)
        # 应用 replace 操作到当前对象，并返回结果
        return self.apply(
            "replace",
            to_replace=to_replace,
            value=value,
            inplace=inplace,
        )

    @final
    def replace_regex(self, **kwargs) -> Self:
        # 应用 _replace_regex 操作到当前对象，并返回结果
        return self.apply("_replace_regex", **kwargs)

    @final
    def replace_list(
        self,
        src_list: list[Any],
        dest_list: list[Any],
        inplace: bool = False,
        regex: bool = False,
    ) -> Self:
        """进行列表替换操作"""
        # 将 inplace 参数转换为布尔值，确保其有效性
        inplace = validate_bool_kwarg(inplace, "inplace")

        # 应用 replace_list 操作到当前对象，并返回结果
        bm = self.apply(
            "replace_list",
            src_list=src_list,
            dest_list=dest_list,
            inplace=inplace,
            regex=regex,
        )
        # 在原地整理数据
        bm._consolidate_inplace()
        return bm

    def interpolate(self, inplace: bool, **kwargs) -> Self:
        # 应用 interpolate 操作到当前对象，并返回结果
        return self.apply("interpolate", inplace=inplace, **kwargs)

    def pad_or_backfill(self, inplace: bool, **kwargs) -> Self:
        # 应用 pad_or_backfill 操作到当前对象，并返回结果
        return self.apply("pad_or_backfill", inplace=inplace, **kwargs)

    def shift(self, periods: int, fill_value) -> Self:
        # 如果 fill_value 没有默认值，则设为 None
        if fill_value is lib.no_default:
            fill_value = None

        # 应用 shift 操作到当前对象，并返回结果
        return self.apply("shift", periods=periods, fill_value=fill_value)
    def setitem(self, indexer, value) -> Self:
        """
        Set values with indexer.

        For SingleBlockManager, this backs s[indexer] = value
        """
        if isinstance(indexer, np.ndarray) and indexer.ndim > self.ndim:
            raise ValueError(f"Cannot set values with ndim > {self.ndim}")

        if not self._has_no_reference(0):
            # 如果存在引用，说明不是单一块，需要处理多块情况
            # 如果是二维且索引是元组，则处理块的位置信息
            if self.ndim == 2 and isinstance(indexer, tuple):
                blk_loc = self.blklocs[indexer[1]]
                # 如果块位置是列表且维度为2，则压缩成一维
                if is_list_like(blk_loc) and blk_loc.ndim == 2:
                    blk_loc = np.squeeze(blk_loc, axis=0)
                # 如果块位置不是列表，则转换为列表形式
                elif not is_list_like(blk_loc):
                    blk_loc = [blk_loc]  # type: ignore[assignment]
                # 如果块位置列表为空，则返回当前对象的浅复制
                if len(blk_loc) == 0:
                    return self.copy(deep=False)

                values = self.blocks[0].values
                # 如果块的值维度为2，则根据块位置获取相应的值
                if values.ndim == 2:
                    values = values[blk_loc]
                    # 调用内部方法 _iset_split_block，设置块的分割
                    self._iset_split_block(  # type: ignore[attr-defined]
                        0, blk_loc, values
                    )
                    # 将值设置到第一个块中的指定位置
                    self.blocks[0].setitem((indexer[0], np.arange(len(blk_loc))), value)
                    return self
            # 如果不需要分割块，或者是单块管理器，则复制当前对象
            self = self.copy()

        # 对当前对象应用 "setitem" 操作，传入索引和值
        return self.apply("setitem", indexer=indexer, value=value)

    def diff(self, n: int) -> Self:
        # 只有在 self.ndim == 2 的情况下才会执行到这里
        return self.apply("diff", n=n)

    def astype(self, dtype, errors: str = "raise") -> Self:
        # 将当前对象的数据类型转换为指定的 dtype
        return self.apply("astype", dtype=dtype, errors=errors)

    def convert(self) -> Self:
        # 对当前对象执行 "convert" 操作
        return self.apply("convert")

    def convert_dtypes(self, **kwargs):
        # 对当前对象执行 "convert_dtypes" 操作，传入额外的参数
        return self.apply("convert_dtypes", **kwargs)

    def get_values_for_csv(
        self, *, float_format, date_format, decimal, na_rep: str = "nan", quoting=None
    ) -> Self:
        """
        Convert values to native types (strings / python objects) that are used
        in formatting (repr / csv).
        """
        # 将当前对象的值转换为用于格式化（repr / csv）的本地类型（字符串/Python 对象）
        return self.apply(
            "get_values_for_csv",
            na_rep=na_rep,
            quoting=quoting,
            float_format=float_format,
            date_format=date_format,
            decimal=decimal,
        )

    @property
    def any_extension_types(self) -> bool:
        """Whether any of the blocks in this manager are extension blocks"""
        # 判断当前管理器中是否有任何扩展块
        return any(block.is_extension for block in self.blocks)
    def is_view(self) -> bool:
        """检查当前对象是否只包含一个块且为视图"""
        if len(self.blocks) == 1:
            return self.blocks[0].is_view
        
        # 如果包含多个块，则无法确定是否为视图
        # 可以尝试通过以下方式判断哪些块是视图：
        # [ b.values.base is not None for b in self.blocks ]
        # 但存在一种情况，即部分块可能是视图，部分块不是视图。
        # 理论上，非视图块可能设置为视图，但这会比较复杂

        return False

    def _get_data_subset(self, predicate: Callable) -> Self:
        """根据条件选择符合条件的数据子集"""
        blocks = [blk for blk in self.blocks if predicate(blk.values)]
        return self._combine(blocks)

    def get_bool_data(self) -> Self:
        """
        选择布尔类型数据块以及对象类型数据块中全为布尔值的列。
        """
        new_blocks = []

        for blk in self.blocks:
            if blk.dtype == bool:
                new_blocks.append(blk)

            elif blk.is_object:
                # 拆分对象类型块，选择布尔类型块
                new_blocks.extend(nb for nb in blk._split() if nb.is_bool)

        return self._combine(new_blocks)

    def get_numeric_data(self) -> Self:
        """选择数值类型数据块"""
        numeric_blocks = [blk for blk in self.blocks if blk.is_numeric]
        if len(numeric_blocks) == len(self.blocks):
            # 如果所有块都是数值类型，直接返回当前对象，避免昂贵的_combine操作
            return self
        return self._combine(numeric_blocks)

    def _combine(self, blocks: list[Block], index: Index | None = None) -> Self:
        """合并给定的块，返回一个新的管理器对象"""
        if len(blocks) == 0:
            if self.ndim == 2:
                # 保留当前对象的索引数据类型
                if index is not None:
                    axes = [self.items[:0], index]
                else:
                    axes = [self.items[:0]] + self.axes[1:]
                return self.make_empty(axes)
            return self.make_empty()

        # FIXME: 优化潜力
        # 对索引进行排序和反向索引
        indexer = np.sort(np.concatenate([b.mgr_locs.as_array for b in blocks]))
        inv_indexer = lib.get_reverse_indexer(indexer, self.shape[0])

        new_blocks: list[Block] = []
        for b in blocks:
            nb = b.copy(deep=False)
            nb.mgr_locs = BlockPlacement(inv_indexer[nb.mgr_locs.indexer])
            new_blocks.append(nb)

        axes = list(self.axes)
        if index is not None:
            axes[-1] = index
        axes[0] = self.items.take(indexer)

        # 根据新的块和轴创建一个新的对象
        return type(self).from_blocks(new_blocks, axes)

    @property
    def nblocks(self) -> int:
        """返回当前对象中块的数量"""
        return len(self.blocks)
    def copy(self, deep: bool | Literal["all"] = True) -> Self:
        """
        Make deep or shallow copy of BlockManager

        Parameters
        ----------
        deep : bool, string or None, default True
            If False or None, return a shallow copy (do not copy data)
            If 'all', copy data and a deep copy of the index

        Returns
        -------
        BlockManager
        """
        # 保留轴的视图复制概念
        if deep:
            # 在例如 tests.io.json.test_pandas 中使用

            def copy_func(ax):
                # 如果 deep == "all"，则进行深拷贝，否则返回视图
                return ax.copy(deep=True) if deep == "all" else ax.view()

            # 对所有轴执行复制函数
            new_axes = [copy_func(ax) for ax in self.axes]
        else:
            # 对所有轴执行视图复制
            new_axes = [ax.view() for ax in self.axes]

        # 应用 "copy" 操作，传递深拷贝参数，并更新轴
        res = self.apply("copy", deep=deep)
        res.axes = new_axes

        if self.ndim > 1:
            # 避免需要重新计算这些值
            blknos = self._blknos
            if blknos is not None:
                # 复制块编号和块位置信息
                res._blknos = blknos.copy()
                res._blklocs = self._blklocs.copy()

        if deep:
            # 在 inplace 下进行合并操作
            res._consolidate_inplace()
        return res

    def is_consolidated(self) -> bool:
        # 始终返回 True，表示数据已经合并
        return True

    def consolidate(self) -> Self:
        """
        Join together blocks having same dtype

        Returns
        -------
        y : BlockManager
        """
        if self.is_consolidated():
            # 如果数据已经合并，则直接返回自身
            return self

        # 创建一个新的 BlockManager 实例，用当前的 blocks 和 axes，且不验证完整性
        bm = type(self)(self.blocks, self.axes, verify_integrity=False)
        bm._is_consolidated = False
        # 在原地执行合并操作
        bm._consolidate_inplace()
        return bm

    def _consolidate_inplace(self) -> None:
        # 空函数，用于在 consolidate 方法中原地合并数据块
        return

    @final
    def reindex_axis(
        self,
        new_index: Index,
        axis: AxisInt,
        fill_value=None,
        only_slice: bool = False,
    ) -> Self:
        """
        Conform data manager to new index.
        """
        # 重新索引轴，返回新的索引和索引器
        new_index, indexer = self.axes[axis].reindex(new_index)

        # 使用新索引和索引器重新索引数据
        return self.reindex_indexer(
            new_index,
            indexer,
            axis=axis,
            fill_value=fill_value,
            only_slice=only_slice,
        )

    def reindex_indexer(
        self,
        new_axis: Index,
        indexer: npt.NDArray[np.intp] | None,
        axis: AxisInt,
        fill_value=None,
        allow_dups: bool = False,
        only_slice: bool = False,
        *,
        use_na_proxy: bool = False,
    ) -> Self:
        """
        Parameters
        ----------
        new_axis : Index
            新轴的索引对象，用于指定重新索引后的新轴
        indexer : ndarray[intp] or None
            索引器，用于指定重新排列的索引，可以是整数数组或空值
        axis : int
            轴的索引，指定在哪个轴上进行重新索引操作
        fill_value : object, default None
            填充值，用于指定在重新索引过程中遇到缺失数据时的填充值，默认为None
        allow_dups : bool, default False
            是否允许在重新索引过程中出现重复索引的布尔值，默认为False
        only_slice : bool, default False
            是否仅返回视图而不是复制数据，针对列进行操作时使用
        use_na_proxy : bool, default False
            是否使用 np.void ndarray 作为新引入列的占位符
            是否使用 np.void ndarray 作为新引入列的占位符

        pandas 索引器，仅使用 -1 的情况。
        """
        if indexer is None:
            if new_axis is self.axes[axis]:
                return self

            result = self.copy(deep=False)
            result.axes = list(self.axes)
            result.axes[axis] = new_axis
            return result

        # Should be intp, but in some cases we get int64 on 32bit builds
        assert isinstance(indexer, np.ndarray)

        # some axes don't allow reindexing with dups
        if not allow_dups:
            self.axes[axis]._validate_can_reindex(indexer)

        if axis >= self.ndim:
            raise IndexError("Requested axis not found in manager")

        if axis == 0:
            new_blocks = list(
                self._slice_take_blocks_ax0(
                    indexer,
                    fill_value=fill_value,
                    only_slice=only_slice,
                    use_na_proxy=use_na_proxy,
                )
            )
        else:
            new_blocks = [
                blk.take_nd(
                    indexer,
                    axis=1,
                    fill_value=(
                        fill_value if fill_value is not None else blk.fill_value
                    ),
                )
                for blk in self.blocks
            ]

        new_axes = list(self.axes)
        new_axes[axis] = new_axis

        new_mgr = type(self).from_blocks(new_blocks, new_axes)
        if axis == 1:
            # We can avoid the need to rebuild these
            new_mgr._blknos = self.blknos.copy()
            new_mgr._blklocs = self.blklocs.copy()
        return new_mgr

    def _slice_take_blocks_ax0(
        self,
        slice_or_indexer: slice | np.ndarray,
        fill_value=lib.no_default,
        only_slice: bool = False,
        *,
        use_na_proxy: bool = False,
        ref_inplace_op: bool = False,
    ):
        """
        Parameters
        ----------
        slice_or_indexer : slice | np.ndarray
            切片或索引器，用于指定在轴0上切片或索引块
        fill_value : object, default lib.no_default
            填充值，用于指定在取块时遇到缺失数据时的填充值，默认为 lib.no_default
        only_slice : bool, default False
            是否仅返回切片视图而不是复制数据
        use_na_proxy : bool, default False
            是否使用 np.void ndarray 作为新引入列的占位符
        ref_inplace_op : bool, default False
            是否是原地操作的引用

        返回在轴0上切片或索引后的块列表。
        """

    def _make_na_block(
        self, placement: BlockPlacement, fill_value=None, use_na_proxy: bool = False
    ):
        """
        Parameters
        ----------
        placement : BlockPlacement
            块的放置方式，指定创建块的位置信息
        fill_value : object, optional
            填充值，用于指定块中缺失数据的填充值，默认为None
        use_na_proxy : bool, default False
            是否使用 np.void ndarray 作为新引入列的占位符

        创建一个带有缺失数据的块对象。
        """
    ) -> Block:
        # Note: we only get here with self.ndim == 2
        # 注意：仅当 self.ndim == 2 时才会执行到这里

        if use_na_proxy:
            assert fill_value is None
            # 确保 fill_value 为 None
            shape = (len(placement), self.shape[1])
            # 计算新数组的形状
            vals = np.empty(shape, dtype=np.void)
            # 创建一个空的 numpy 数组 vals，dtype 为 np.void
            nb = NumpyBlock(vals, placement, ndim=2)
            # 使用 vals 和 placement 创建一个 NumpyBlock 对象，ndim 设为 2
            return nb
            # 返回新创建的 NumpyBlock 对象

        if fill_value is None or fill_value is np.nan:
            fill_value = np.nan
            # 如果 fill_value 为 None 或者 np.nan，则设置 fill_value 为 np.nan
            # GH45857 avoid unnecessary upcasting
            # GH45857 避免不必要的类型提升
            dtype = interleaved_dtype([blk.dtype for blk in self.blocks])
            # 推断出合适的 dtype，使用所有块的 dtype 进行交错处理
            if dtype is not None and np.issubdtype(dtype.type, np.floating):
                fill_value = dtype.type(fill_value)
                # 如果 dtype 是浮点类型，则将 fill_value 转换为相应的 dtype

        shape = (len(placement), self.shape[1])
        # 计算新数组的形状

        dtype, fill_value = infer_dtype_from_scalar(fill_value)
        # 从标量 fill_value 推断出 dtype 和 fill_value 的值
        block_values = make_na_array(dtype, shape, fill_value)
        # 创建一个填充了 NA 值的数组 block_values
        return new_block_2d(block_values, placement=placement)
        # 返回一个新的二维块，使用 block_values 和 placement 创建

    def take(
        self,
        indexer: npt.NDArray[np.intp],
        axis: AxisInt = 1,
        verify: bool = True,
    ) -> Self:
        """
        Take items along any axis.

        indexer : np.ndarray[np.intp]
        axis : int, default 1
        verify : bool, default True
            Check that all entries are between 0 and len(self) - 1, inclusive.
            Pass verify=False if this check has been done by the caller.

        Returns
        -------
        BlockManager
        """
        # Caller is responsible for ensuring indexer annotation is accurate
        # 调用者需确保 indexer 的注释准确无误

        n = self.shape[axis]
        # 获取指定轴的长度
        indexer = maybe_convert_indices(indexer, n, verify=verify)
        # 将索引 indexer 可能转换为合适的形式，确保在范围内

        new_labels = self.axes[axis].take(indexer)
        # 获取新的标签，使用索引 indexer 从轴上获取

        return self.reindex_indexer(
            new_axis=new_labels,
            indexer=indexer,
            axis=axis,
            allow_dups=True,
        )
        # 返回根据新索引重新索引的 BlockManager 对象
# 继承自 libinternals.BlockManager 和 BaseBlockManager 的基础块管理器，用于存储二维块数据。
class BlockManager(libinternals.BlockManager, BaseBlockManager):
    """
    BaseBlockManager that holds 2D blocks.
    """

    # 类属性，表示块的维度为二维
    ndim = 2

    # ----------------------------------------------------------------
    # 构造函数

    def __init__(
        self,
        blocks: Sequence[Block],  # 传入的块序列，每个块是 Block 类型
        axes: Sequence[Index],    # 传入的轴序列，每个轴是 Index 类型
        verify_integrity: bool = True,  # 是否验证数据完整性，默认为 True
    ) -> None:
        if verify_integrity:
            # 对性能进行优化而禁用的断言
            # assert all(isinstance(x, Index) for x in axes)

            # 遍历传入的块，检查每个块的维度是否与 ndim 相等
            for block in blocks:
                if self.ndim != block.ndim:
                    raise AssertionError(
                        f"Number of Block dimensions ({block.ndim}) must equal "
                        f"number of axes ({self.ndim})"
                    )
                # 从 2.0 版本开始，调用者需确保 DatetimeTZBlock 的 block.ndim == 2
                # 并且 block.values.ndim == 2；之前有特殊检查以兼容 fastparquet。

            # 调用对象的完整性验证方法
            self._verify_integrity()

    # 验证对象完整性的私有方法
    def _verify_integrity(self) -> None:
        mgr_shape = self.shape  # 获取块管理器的形状
        tot_items = sum(len(x.mgr_locs) for x in self.blocks)  # 计算总共的项目数
        # 遍历每个块，检查其形状是否与块管理器的形状的轴部分相等
        for block in self.blocks:
            if block.shape[1:] != mgr_shape[1:]:
                raise_construction_error(tot_items, block.shape[1:], self.axes)
        # 如果管理器的项目数不等于总项目数，则抛出断言错误
        if len(self.items) != tot_items:
            raise AssertionError(
                "Number of manager items must equal union of "
                f"block items\n# manager items: {len(self.items)}, # "
                f"tot_items: {tot_items}"
            )

    @classmethod
    def from_blocks(cls, blocks: list[Block], axes: list[Index]) -> Self:
        """
        从块列表和轴列表构造 BlockManager 或 SingleBlockManager 的构造函数。
        """
        return cls(blocks, axes, verify_integrity=False)

    # ----------------------------------------------------------------
    # 索引操作
    def fast_xs(self, loc: int) -> SingleBlockManager:
        """
        Return the array corresponding to `frame.iloc[loc]`.

        Parameters
        ----------
        loc : int

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        # 如果只有一个数据块
        if len(self.blocks) == 1:
            # TODO: 如果 blk.mgr_locs 不像 slice(None) 那样，可能是错误的；
            #  在一般情况下这种情况被排除了吗？
            #  返回与索引位置 loc 对应的块数据（可能是 np.ndarray 或 ExtensionArray）
            result: np.ndarray | ExtensionArray = self.blocks[0].iget(
                (slice(None), loc)
            )
            # 对于单个数据块的情况，新的块是一个视图
            bp = BlockPlacement(slice(0, len(result)))
            # 创建新的数据块对象
            block = new_block(
                result,
                placement=bp,
                ndim=1,
                refs=self.blocks[0].refs,
            )
            # 返回单块管理器对象，用于管理这个新创建的数据块
            return SingleBlockManager(block, self.axes[0])

        # 计算交错的数据类型
        dtype = interleaved_dtype([blk.dtype for blk in self.blocks])

        n = len(self)

        if isinstance(dtype, ExtensionDtype):
            # TODO: 使用对象数据类型作为非高效 EA.__setitem__ 方法的解决方法。
            #  主要是 ArrowExtensionArray.__setitem__ 在逐个设置值时
            #  https://github.com/pandas-dev/pandas/pull/54508#issuecomment-1675827918
            # 创建一个对象数据类型的空数组
            result = np.empty(n, dtype=object)
        else:
            # 创建一个具有指定数据类型的空数组
            result = np.empty(n, dtype=dtype)
            # 如果数据类型类似日期时间，确保被包装
            result = ensure_wrapped_if_datetimelike(result)

        for blk in self.blocks:
            # 可能会不正确地将 NaT 强制转换为 None 的赋值
            # 遍历每个数据块的管理位置，设置对应位置的值
            for i, rl in enumerate(blk.mgr_locs):
                result[rl] = blk.iget((i, loc))

        if isinstance(dtype, ExtensionDtype):
            # 构造扩展数据类型的数组
            cls = dtype.construct_array_type()
            result = cls._from_sequence(result, dtype=dtype)

        # 创建新的数据块对象
        bp = BlockPlacement(slice(0, len(result)))
        block = new_block(result, placement=bp, ndim=1)
        # 返回单块管理器对象，用于管理这个新创建的数据块
        return SingleBlockManager(block, self.axes[0])

    def iget(self, i: int, track_ref: bool = True) -> SingleBlockManager:
        """
        Return the data as a SingleBlockManager.
        """
        # 获取索引 i 对应的数据块
        block = self.blocks[self.blknos[i]]
        # 获取数据块中相应位置的值
        values = block.iget(self.blklocs[i])

        # 对于从二维块中选择单个维度的快捷方式
        bp = BlockPlacement(slice(0, len(values)))
        # 创建新的数据块对象
        nb = type(block)(
            values, placement=bp, ndim=1, refs=block.refs if track_ref else None
        )
        # 返回单块管理器对象，用于管理这个新创建的数据块
        return SingleBlockManager(nb, self.axes[1])
    def iget_values(self, i: int) -> ArrayLike:
        """
        Return the data for column i as the values (ndarray or ExtensionArray).

        Warning! The returned array is a view but doesn't handle Copy-on-Write,
        so this should be used with caution.
        """
        # 获取第 i 列的数据块
        block = self.blocks[self.blknos[i]]
        # 从数据块中获取第 i 列的值
        values = block.iget(self.blklocs[i])
        return values

    @property
    def column_arrays(self) -> list[np.ndarray]:
        """
        Used in the JSON C code to access column arrays.
        This optimizes compared to using `iget_values` by converting each

        Warning! This doesn't handle Copy-on-Write, so should be used with
        caution (current use case of consuming this in the JSON code is fine).
        """
        # 以下是一个优化过的等价于
        # result = [self.iget_values(i) for i in range(len(self.items))]
        # 的方法
        result: list[np.ndarray | None] = [None] * len(self.items)

        for blk in self.blocks:
            # 获取块的位置
            mgr_locs = blk._mgr_locs
            # 获取块的数组值用于 JSON
            values = blk.array_values._values_for_json()
            if values.ndim == 1:
                # TODO(EA2D): 对于二维 EAs 不需要特殊处理
                result[mgr_locs[0]] = values
            else:
                # 处理多维块数组
                for i, loc in enumerate(mgr_locs):
                    result[loc] = values[i]

        # 错误: 返回值类型不兼容 (得到 "List[None]", 期望 "List[ndarray[Any, Any]]")
        return result  # type: ignore[return-value]

    def iset(
        self,
        loc: int | slice | np.ndarray,
        value: ArrayLike,
        inplace: bool = False,
        refs: BlockValuesRefs | None = None,
    ):
        ...

    def _iset_split_block(
        self,
        blkno_l: int,
        blk_locs: np.ndarray | list[int],
        value: ArrayLike | None = None,
        refs: BlockValuesRefs | None = None,
    ):
        ...
    ) -> None:
        """
        Removes columns from a block by splitting the block.

        Avoids copying the whole block through slicing and updates the manager
        after determining the new block structure. Optionally adds a new block,
        otherwise has to be done by the caller.

        Parameters
        ----------
        blkno_l: The block number to operate on, relevant for updating the manager
        blk_locs: The locations of our block that should be deleted.
        value: The value to set as a replacement.
        refs: The reference tracking object of the value to set.
        """
        # 获取待操作的数据块
        blk = self.blocks[blkno_l]

        # 如果块位置信息未初始化，则重新构建
        if self._blklocs is None:
            self._rebuild_blknos_and_blklocs()

        # 删除块中指定位置的数据，并返回新块的元组
        nbs_tup = tuple(blk.delete(blk_locs))

        # 如果有替换值，则创建一个新的块对象
        if value is not None:
            locs = blk.mgr_locs.as_array[blk_locs]
            first_nb = new_block_2d(value, BlockPlacement(locs), refs=refs)
        else:
            first_nb = nbs_tup[0]
            nbs_tup = tuple(nbs_tup[1:])

        # 计算当前块数量
        nr_blocks = len(self.blocks)

        # 构建更新后的块元组
        blocks_tup = (
            self.blocks[:blkno_l] + (first_nb,) + self.blocks[blkno_l + 1 :] + nbs_tup
        )
        self.blocks = blocks_tup

        # 如果未发生分割且有替换值，则无需更新任何内容
        if not nbs_tup and value is not None:
            return

        # 更新新块的位置信息
        self._blklocs[first_nb.mgr_locs.indexer] = np.arange(len(first_nb))

        # 更新其他块的位置信息
        for i, nb in enumerate(nbs_tup):
            self._blklocs[nb.mgr_locs.indexer] = np.arange(len(nb))
            self._blknos[nb.mgr_locs.indexer] = i + nr_blocks

    def _iset_single(
        self,
        loc: int,
        value: ArrayLike,
        inplace: bool,
        blkno: int,
        blk: Block,
        refs: BlockValuesRefs | None = None,
    ) -> None:
        """
        Fastpath for iset when we are only setting a single position and
        the Block currently in that position is itself single-column.

        In this case we can swap out the entire Block and blklocs and blknos
        are unaffected.
        """
        # 调用者负责验证值的形状

        # 如果是原地操作且当前块支持存储值
        if inplace and blk.should_store(value):
            # 检查是否需要复制块数据
            copy = not self._has_no_reference_block(blkno)
            iloc = self.blklocs[loc]
            # 在原地设置新值
            blk.set_inplace(slice(iloc, iloc + 1), value, copy=copy)
            return

        # 创建一个新的二维块对象
        nb = new_block_2d(value, placement=blk._mgr_locs, refs=refs)
        old_blocks = self.blocks
        # 替换原块为新块
        new_blocks = old_blocks[:blkno] + (nb,) + old_blocks[blkno + 1 :]
        self.blocks = new_blocks
        return

    def column_setitem(
        self, loc: int, idx: int | slice | np.ndarray, value, inplace_only: bool = False
    ):
        """
        Placeholder for column setitem functionality.
        """
    ) -> None:
        """
        Set values ("setitem") into a single column (not setting the full column).

        This is a method on the BlockManager level, to avoid creating an
        intermediate Series at the DataFrame level (`s = df[loc]; s[idx] = value`)
        """
        # 检查指定位置是否存在引用，如果存在引用，则不能直接修改数据
        if not self._has_no_reference(loc):
            # 获取指定位置的块编号
            blkno = self.blknos[loc]
            # 获取指定位置的块位置
            blk_loc = self.blklocs[loc]
            # 获取块中的数据，并复制以便修改
            values = self.blocks[blkno].values
            if values.ndim == 1:
                values = values.copy()
            else:
                # 使用 [blk_loc] 作为索引器以保持 ndim=2，这已经进行了一次复制操作
                values = values[[blk_loc]]
            # 将复制后的数据块设置回原位置
            self._iset_split_block(blkno, [blk_loc], values)

        # 创建临时的列管理器，用于原地修改值
        col_mgr = self.iget(loc, track_ref=False)
        if inplace_only:
            # 如果只允许原地修改，则在列管理器中原地设置指定索引处的值
            col_mgr.setitem_inplace(idx, value)
        else:
            # 否则，创建一个新的管理器，并设置指定索引处的值
            new_mgr = col_mgr.setitem((idx,), value)
            # 将新的块值设置回原位置
            self.iset(loc, new_mgr._block.values, inplace=True)
    # 在指定位置插入项目。

    # 新建一个包含新位置的轴
    new_axis = self.items.insert(loc, item)

    # 如果值的维度是二维，则转置它
    if value.ndim == 2:
        value = value.T
        # 如果转置后的长度大于1，则引发值错误
        if len(value) > 1:
            raise ValueError(
                f"Expected a 1D array, got an array with shape {value.T.shape}"
            )
    else:
        # 确保值的块形状与当前数据结构的维度兼容
        value = ensure_block_shape(value, ndim=self.ndim)

    # 创建一个新的数据块放置对象
    bp = BlockPlacement(slice(loc, loc + 1))
    # 使用给定的值和放置信息创建一个二维数据块
    block = new_block_2d(values=value, placement=bp, refs=refs)

    # 如果没有块存在，则使用快速路径处理
    if not len(self.blocks):
        self._blklocs = np.array([0], dtype=np.intp)
        self._blknos = np.array([0], dtype=np.intp)
    else:
        # 更新管理位置信息，将新块位置以上的块的位置向上偏移一位
        self._insert_update_mgr_locs(loc)
        # 更新块位置和块编号信息
        self._insert_update_blklocs_and_blknos(loc)

    # 更新第一个轴
    self.axes[0] = new_axis
    # 将新创建的块添加到块列表中
    self.blocks += (block,)

    # 表明数据帧不再是已合并的状态
    self._known_consolidated = False

    # 如果启用了性能警告且块中非扩展块的数量超过100个，发出警告
    if (
        get_option("performance_warnings")
        and sum(not block.is_extension for block in self.blocks) > 100
    ):
        warnings.warn(
            "DataFrame is highly fragmented.  This is usually the result "
            "of calling `frame.insert` many times, which has poor performance.  "
            "Consider joining all columns at once using pd.concat(axis=1) "
            "instead. To get a de-fragmented frame, use `newframe = frame.copy()`",
            PerformanceWarning,
            stacklevel=find_stack_level(),
        )

def _insert_update_mgr_locs(self, loc) -> None:
    """
    当在位置 'loc' 插入新块时，递增所有位于其上方的块的 mgr_locs。
    """
    # 使用 np.bincount 来快速确定需要更新的块编号
    blknos = np.bincount(self.blknos[loc:]).nonzero()[0]
    for blkno in blknos:
        # 获取当前块对象
        blk = self.blocks[blkno]
        # 增加当前块中所有位置信息大于 loc 的位置值
        blk._mgr_locs = blk._mgr_locs.increment_above(loc)
    def _insert_update_blklocs_and_blknos(self, loc) -> None:
        """
        When inserting a new Block at location 'loc', we update our
        _blklocs and _blknos.
        """

        # Accessing public blklocs ensures the public versions are initialized
        # 如果 loc 等于 self.blklocs 的行数，表示在末尾插入新的 Block
        if loc == self.blklocs.shape[0]:
            # np.append 的速度更快，所以我们选择使用它
            self._blklocs = np.append(self._blklocs, 0)
            self._blknos = np.append(self._blknos, len(self.blocks))
        elif loc == 0:
            # 在 numpy 1.26.4 中，np.concatenate 比 np.append 更快
            self._blklocs = np.concatenate([[0], self._blklocs])
            self._blknos = np.concatenate([[len(self.blocks)], self._blknos])
        else:
            # 调用内部函数更新块位置和块编号
            new_blklocs, new_blknos = libinternals.update_blklocs_and_blknos(
                self.blklocs, self.blknos, loc, len(self.blocks)
            )
            self._blklocs = new_blklocs
            self._blknos = new_blknos

    def idelete(self, indexer) -> BlockManager:
        """
        Delete selected locations, returning a new BlockManager.
        """
        # 创建一个布尔数组标记要删除的位置
        is_deleted = np.zeros(self.shape[0], dtype=np.bool_)
        is_deleted[indexer] = True
        # 获取未被删除的索引
        taker = (~is_deleted).nonzero()[0]

        # 使用未被删除的索引对块进行切片和获取操作
        nbs = self._slice_take_blocks_ax0(taker, only_slice=True, ref_inplace_op=True)
        # 创建新的列名称列表，剔除了被删除的列
        new_columns = self.items[~is_deleted]
        # 创建新的块管理器对象并返回
        axes = [new_columns, self.axes[1]]
        return type(self)(tuple(nbs), axes, verify_integrity=False)

    # ----------------------------------------------------------------
    # Block-wise Operation

    def grouped_reduce(self, func: Callable) -> Self:
        """
        Apply grouped reduction function blockwise, returning a new BlockManager.

        Parameters
        ----------
        func : grouped reduction function

        Returns
        -------
        BlockManager
        """
        # 初始化结果块列表
        result_blocks: list[Block] = []

        # 遍历每个块
        for blk in self.blocks:
            if blk.is_object:
                # 对于对象类型的块，根据数据类型拆分块
                for sb in blk._split():
                    # 应用给定函数到每个子块并扩展结果块列表
                    applied = sb.apply(func)
                    result_blocks = extend_blocks(applied, result_blocks)
            else:
                # 对于非对象类型的块，直接应用函数并扩展结果块列表
                applied = blk.apply(func)
                result_blocks = extend_blocks(applied, result_blocks)

        # 如果结果块列表为空，则行数为 0
        if len(result_blocks) == 0:
            nrows = 0
        else:
            # 否则，获取第一个结果块的值的形状中的最后一个维度作为行数
            nrows = result_blocks[0].values.shape[-1]

        # 创建默认的索引
        index = default_index(nrows)

        # 使用结果块列表和原始块管理器的第一轴和新索引创建新的块管理器对象并返回
        return type(self).from_blocks(result_blocks, [self.axes[0], index])
    def reduce(self, func: Callable) -> Self:
        """
        Apply reduction function blockwise, returning a single-row BlockManager.

        Parameters
        ----------
        func : reduction function
            The function to be applied for reduction operations.

        Returns
        -------
        BlockManager
            A new BlockManager instance resulting from the reduction operation.
        """
        # If 2D, we assume that we're operating column-wise
        assert self.ndim == 2  # Ensure the object is 2-dimensional

        res_blocks: list[Block] = []
        for blk in self.blocks:
            # Apply the reduction function to each block
            nbs = blk.reduce(func)
            res_blocks.extend(nbs)

        index = Index([None])  # placeholder
        # Create a new BlockManager from the resulting blocks and current items
        new_mgr = type(self).from_blocks(res_blocks, [self.items, index])
        return new_mgr

    def operate_blockwise(self, other: BlockManager, array_op) -> BlockManager:
        """
        Apply array_op blockwise with another (aligned) BlockManager.
        """
        return operate_blockwise(self, other, array_op)

    def _equal_values(self: BlockManager, other: BlockManager) -> bool:
        """
        Used in .equals defined in base class. Only check the column values
        assuming shape and indexes have already been checked.
        """
        return blockwise_all(self, other, array_equals)

    def quantile(
        self,
        *,
        qs: Index,  # with dtype float 64
        interpolation: QuantileInterpolation = "linear",
    ) -> Self:
        """
        Iterate over blocks applying quantile reduction.
        This routine is intended for reduction type operations and
        will do inference on the generated blocks.

        Parameters
        ----------
        interpolation : type of interpolation, default 'linear'
            The method used to interpolate between quantiles.
        qs : Index
            List of the quantiles to be computed, with dtype np.float64.

        Returns
        -------
        BlockManager
            A new BlockManager instance containing quantiles computed over blocks.
        """
        # Series dispatches to DataFrame for quantile, simplifying code
        assert self.ndim >= 2  # Ensure the object is at least 2-dimensional
        assert is_list_like(qs)  # Caller should ensure qs is list-like

        new_axes = list(self.axes)
        new_axes[1] = Index(qs, dtype=np.float64)

        # Compute quantiles for each block in self.blocks
        blocks = [
            blk.quantile(qs=qs, interpolation=interpolation) for blk in self.blocks
        ]

        return type(self)(blocks, new_axes)

    # ----------------------------------------------------------------
    def unstack(self, unstacker, fill_value) -> BlockManager:
        """
        Return a BlockManager with all blocks unstacked.

        Parameters
        ----------
        unstacker : reshape._Unstacker
            An instance of _Unstacker used to manage the unstacking operation.
        fill_value : Any
            Fill value for newly introduced missing values.

        Returns
        -------
        unstacked : BlockManager
            A BlockManager containing all blocks after unstacking.
        """
        # Get the new columns resulting from unstacking operation
        new_columns = unstacker.get_new_columns(self.items)
        # Get the new index after unstacking
        new_index = unstacker.new_index

        # Determine if fill value should be allowed based on mask_all attribute of unstacker
        allow_fill = not unstacker.mask_all
        if allow_fill:
            # Calculate the full 2D mask for optimization
            new_mask2D = (~unstacker.mask).reshape(*unstacker.full_shape)
            # Identify columns that need masking based on the mask
            needs_masking = new_mask2D.any(axis=0)
        else:
            # If fill value is not allowed, initialize needs_masking as all False
            needs_masking = np.zeros(unstacker.full_shape[1], dtype=bool)

        # Initialize lists to store new blocks and corresponding column masks
        new_blocks: list[Block] = []
        columns_mask: list[np.ndarray] = []

        # Determine factor for tile replication based on existing items
        if len(self.items) == 0:
            factor = 1
        else:
            fac = len(new_columns) / len(self.items)
            assert fac == int(fac)  # Ensure factor is an integer
            factor = int(fac)

        # Iterate over existing blocks and perform unstacking
        for blk in self.blocks:
            # Get manager locations for current block
            mgr_locs = blk.mgr_locs
            # Determine new placement for unstacked block
            new_placement = mgr_locs.tile_for_unstack(factor)

            # Call _unstack method on current block to get unstacked blocks and masks
            blocks, mask = blk._unstack(
                unstacker,
                fill_value,
                new_placement=new_placement,
                needs_masking=needs_masking,
            )

            # Extend lists with new blocks and masks
            new_blocks.extend(blocks)
            columns_mask.extend(mask)

            # Assertion to ensure consistency in mask sum
            assert mask.sum() == sum(len(nb._mgr_locs) for nb in blocks)

        # Filter new columns based on columns_mask
        new_columns = new_columns[columns_mask]

        # Create a new BlockManager with unstacked blocks, new columns, and new index
        bm = BlockManager(new_blocks, [new_columns, new_index], verify_integrity=False)
        return bm

    def to_iter_dict(self) -> Generator[tuple[str, Self], None, None]:
        """
        Yield a tuple of (str(dtype), BlockManager) for each dtype found in blocks.

        Returns
        -------
        values : Generator[tuple[str, Self], None, None]
            A generator yielding tuples of (str(dtype), BlockManager).
        """
        # Define key function to group blocks by dtype
        key = lambda block: str(block.dtype)
        # Group blocks by dtype and iterate over each group
        for dtype, blocks in itertools.groupby(sorted(self.blocks, key=key), key=key):
            # Combine blocks in the current group and yield as (dtype, BlockManager) tuple
            yield dtype, self._combine(list(blocks))

    def as_array(
        self,
        dtype: np.dtype | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
        ):
        """
        Placeholder function. Not fully implemented yet.

        Parameters
        ----------
        dtype : np.dtype or None, optional
            Data type for the array. If None, default dtype is used.
        copy : bool, default False
            Whether to copy the data. Not fully implemented.
        na_value : object, default lib.no_default
            Not fully implemented.

        Returns
        -------
        None
            This function does not return a value as it is not fully implemented.
        """
        # This function is intended to be a placeholder and is not fully implemented yet.
        pass
    ```python`
        ) -> np.ndarray:
            """
            Convert the blockmanager data into an numpy array.
    
            Parameters
            ----------
            dtype : np.dtype or None, default None
                Data type of the return array.
            copy : bool, default False
                If True then guarantee that a copy is returned. A value of
                False does not guarantee that the underlying data is not
                copied.
            na_value : object, default lib.no_default
                Value to be used as the missing value sentinel.
    
            Returns
            -------
            arr : ndarray
            """
            # 检查 na_value 是否为浮点数并且是缺失值（NaN）
            passed_nan = lib.is_float(na_value) and isna(na_value)
    
            # 如果块列表为空，创建一个空的 numpy 数组并返回其转置
            if len(self.blocks) == 0:
                arr = np.empty(self.shape, dtype=float)
                return arr.transpose()
    
            # 如果只有一个块，处理单块情况
            if self.is_single_block:
                blk = self.blocks[0]
    
                # 如果提供了 na_value，则需要确保返回一个副本，避免修改原始对象
                if na_value is not lib.no_default:
                    if lib.is_np_dtype(blk.dtype, "f") and passed_nan:
                        pass  # 如果数据已经是 numpy 浮点类型且 na_value 为 NaN，什么也不做
                    else:
                        copy = True  # 否则设置 copy 为 True
    
                # 如果块是扩展类型，避免隐式转换为对象类型
                if blk.is_extension:
                    arr = blk.values.to_numpy(  # 转换为 numpy 数组，忽略类型检查
                        dtype=dtype,
                        na_value=na_value,
                        copy=copy,
                    ).reshape(blk.shape)
                elif not copy:
                    arr = np.asarray(blk.values, dtype=dtype)  # 不复制，直接转换为 numpy 数组
                else:
                    arr = np.array(blk.values, dtype=dtype, copy=copy)  # 复制并转换为 numpy 数组
    
                # 如果不需要复制，创建只读数组
                if not copy:
                    arr = arr.view()
                    arr.flags.writeable = False
            else:
                # 如果有多个块，调用 _interleave 方法处理
                arr = self._interleave(dtype=dtype, na_value=na_value)
                # _interleave 方法内部已经处理了数据复制，因此不需要额外复制数据
    
            # 如果 na_value 为默认值，什么也不做
            if na_value is lib.no_default:
                pass
            # 如果数据类型是浮点数且 na_value 是 NaN，什么也不做
            elif arr.dtype.kind == "f" and passed_nan:
                pass
            else:
                arr[isna(arr)] = na_value  # 将数组中所有缺失值替换为 na_value
    
            return arr.transpose()  # 返回转置后的数组
    
        def _interleave(
            self,
            dtype: np.dtype | None = None,
            na_value: object = lib.no_default,
    ) -> np.ndarray:
        """
        Return ndarray from blocks with specified item order
        Items must be contained in the blocks
        """
        if not dtype:
            # 如果未指定 dtype，则根据 blocks 中的 dtype 创建混合类型
            dtype = interleaved_dtype(  # type: ignore[assignment]
                [blk.dtype for blk in self.blocks]
            )

        # error: Argument 1 to "ensure_np_dtype" has incompatible type
        # "Optional[dtype[Any]]"; expected "Union[dtype[Any], ExtensionDtype]"
        dtype = ensure_np_dtype(dtype)  # type: ignore[arg-type]
        # 创建一个空的 ndarray，用指定的 dtype
        result = np.empty(self.shape, dtype=dtype)

        # 创建一个长度与 shape[0] 相同的零数组，表示每个项的存在情况
        itemmask = np.zeros(self.shape[0])

        if dtype == np.dtype("object") and na_value is lib.no_default:
            # 比使用 to_numpy 更高效
            # 对于每个数据块 blk，将其值放入 result 中对应的位置，更新 itemmask
            for blk in self.blocks:
                rl = blk.mgr_locs
                arr = blk.get_values(dtype)
                result[rl.indexer] = arr
                itemmask[rl.indexer] = 1
            return result

        for blk in self.blocks:
            rl = blk.mgr_locs
            if blk.is_extension:
                # 避免将扩展块隐式转换为对象类型

                # error: Item "ndarray" of "Union[ndarray, ExtensionArray]" has no
                # attribute "to_numpy"
                # 如果是扩展块，将其值转换为 numpy 数组并放入 result 中对应的位置，更新 itemmask
                arr = blk.values.to_numpy(  # type: ignore[union-attr]
                    dtype=dtype,
                    na_value=na_value,
                )
            else:
                # 否则直接获取块的值并放入 result 中对应的位置，更新 itemmask
                arr = blk.get_values(dtype)
            result[rl.indexer] = arr
            itemmask[rl.indexer] = 1

        if not itemmask.all():
            raise AssertionError("Some items were not contained in blocks")

        return result

    # ----------------------------------------------------------------
    # Consolidation

    def is_consolidated(self) -> bool:
        """
        Return True if more than one block with the same dtype
        """
        if not self._known_consolidated:
            # 如果尚未检查过是否已合并，则进行检查
            self._consolidate_check()
        return self._is_consolidated

    def _consolidate_check(self) -> None:
        if len(self.blocks) == 1:
            # 快速路径：只有一个块时，视为已合并
            self._is_consolidated = True
            self._known_consolidated = True
            return
        # 收集所有可合并块的 dtype
        dtypes = [blk.dtype for blk in self.blocks if blk._can_consolidate]
        # 判断是否所有可合并块的 dtype 都不同，更新合并状态
        self._is_consolidated = len(dtypes) == len(set(dtypes))
        self._known_consolidated = True
    def _consolidate_inplace(self) -> None:
        """
        In general, _consolidate_inplace should only be called via
        DataFrame._consolidate_inplace, otherwise we will fail to invalidate
        the DataFrame's _item_cache. The exception is for newly-created
        BlockManager objects not yet attached to a DataFrame.
        """
        # 如果数据块未被合并，则调用 _consolidate 方法进行合并
        if not self.is_consolidated():
            self.blocks = _consolidate(self.blocks)
            # 将 _is_consolidated 标记为 True，表示数据块已经合并
            self._is_consolidated = True
            # 标记 _known_consolidated 为 True，指示数据块已知已合并
            self._known_consolidated = True
            # 重新构建块编号和块位置
            self._rebuild_blknos_and_blklocs()

    # ----------------------------------------------------------------
    # Concatenation

    @classmethod
    def concat_horizontal(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        """
        Concatenate uniformly-indexed BlockManagers horizontally.
        """
        # 初始化偏移量为 0，用于调整块的位置
        offset = 0
        # 初始化块列表
        blocks: list[Block] = []
        # 遍历每个 BlockManager 对象
        for mgr in mgrs:
            # 遍历当前 BlockManager 对象的每个数据块
            for blk in mgr.blocks:
                """
                We need to do getitem_block here otherwise we would be altering
                blk.mgr_locs in place, which would render it invalid. This is only
                relevant in the copy=False case.
                """
                # 切片当前数据块的列，并且将块的位置偏移量添加到 _mgr_locs 中
                nb = blk.slice_block_columns(slice(None))
                nb._mgr_locs = nb._mgr_locs.add(offset)
                # 将处理后的数据块添加到块列表中
                blocks.append(nb)

            # 增加偏移量，以便下一个 BlockManager 的数据块排在当前数据块之后
            offset += len(mgr.items)

        # 创建新的 BlockManager 对象，并返回
        new_mgr = cls(tuple(blocks), axes)
        return new_mgr

    @classmethod
    def concat_vertical(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        """
        Concatenate uniformly-indexed BlockManagers vertically.
        """
        # 抛出未实现的错误，因为垂直拼接逻辑目前在 internals.concat 中实现
        raise NotImplementedError("This logic lives (for now) in internals.concat")
class SingleBlockManager(BaseBlockManager):
    """管理单个数据块"""

    @property
    def ndim(self) -> Literal[1]:
        # 返回数据块的维度，这里是一维数据
        return 1

    _is_consolidated = True
    _known_consolidated = True
    __slots__ = ()
    is_single_block = True

    def __init__(
        self,
        block: Block,
        axis: Index,
        verify_integrity: bool = False,
    ) -> None:
        # Assertions disabled for performance
        # 断言被禁用以提高性能
        # assert isinstance(block, Block), type(block)
        # assert isinstance(axis, Index), type(axis)

        # 初始化管理器的轴和数据块
        self.axes = [axis]
        self.blocks = (block,)

    @classmethod
    def from_blocks(
        cls,
        blocks: list[Block],
        axes: list[Index],
    ) -> 'SingleBlockManager':
        """
        从块列表和轴列表创建 SingleBlockManager 的构造方法。
        """
        assert len(blocks) == 1
        assert len(axes) == 1
        return cls(blocks[0], axes[0], verify_integrity=False)

    @classmethod
    def from_array(
        cls, array: ArrayLike, index: Index, refs: BlockValuesRefs | None = None
    ) -> 'SingleBlockManager':
        """
        从数组创建 SingleBlockManager 的构造方法，如果数组尚未是 Block 类型。
        """
        # 将数组转换为块数据
        array = maybe_coerce_values(array)
        bp = BlockPlacement(slice(0, len(index)))
        block = new_block(array, placement=bp, ndim=1, refs=refs)
        return cls(block, index)

    def to_2d_mgr(self, columns: Index) -> BlockManager:
        """
        将单块管理器转换为二维块管理器的方法，类似于 Series.to_frame
        """
        # 获取单块管理器的块和数据，并确保其形状为二维
        blk = self.blocks[0]
        arr = ensure_block_shape(blk.values, ndim=2)
        bp = BlockPlacement(0)
        new_blk = type(blk)(arr, placement=bp, ndim=2, refs=blk.refs)
        axes = [columns, self.axes[0]]
        return BlockManager([new_blk], axes=axes, verify_integrity=False)

    def _has_no_reference(self, i: int = 0) -> bool:
        """
        检查第 i 列是否没有引用。
        如果列没有引用（即它没有引用其他数组，也没有被其他引用），返回 True。
        """
        return not self.blocks[0].refs.has_reference()

    def __getstate__(self):
        # 获取当前对象的状态以便序列化
        block_values = [b.values for b in self.blocks]
        block_items = [self.items[b.mgr_locs.indexer] for b in self.blocks]
        axes_array = list(self.axes)

        extra_state = {
            "0.14.1": {
                "axes": axes_array,
                "blocks": [
                    {"values": b.values, "mgr_locs": b.mgr_locs.indexer}
                    for b in self.blocks
                ],
            }
        }

        # 保持与 0.13.1 版本的向前兼容性的前三个元素
        return axes_array, block_values, block_items, extra_state
    def __setstate__(self, state) -> None:
        # 定义一个用于反序列化块的函数，返回一个 Block 对象
        def unpickle_block(values, mgr_locs, ndim: int) -> Block:
            # TODO(EA2D): 在 2D EAs 中，ndim 可能不再需要
            # 旧版本的 pickle 可能存储如 DatetimeIndex 而不是 DatetimeArray
            values = extract_array(values, extract_numpy=True)
            # 如果 mgr_locs 不是 BlockPlacement 对象，则创建一个新的 BlockPlacement 对象
            if not isinstance(mgr_locs, BlockPlacement):
                mgr_locs = BlockPlacement(mgr_locs)

            # 可能需要转换 values 的类型
            values = maybe_coerce_values(values)
            # 创建一个新的块对象，并指定其位置和维度
            return new_block(values, placement=mgr_locs, ndim=ndim)

        # 如果 state 是一个元组且长度至少为 4，并且包含字符串 "0.14.1"
        if isinstance(state, tuple) and len(state) >= 4 and "0.14.1" in state[3]:
            # 取出包含 "0.14.1" 的部分作为新的 state
            state = state[3]["0.14.1"]
            # 确保 self.axes 里面的每个元素都是索引对象
            self.axes = [ensure_index(ax) for ax in state["axes"]]
            # 确定 self.axes 的维度
            ndim = len(self.axes)
            # 对每个块进行反序列化，生成一个块的元组
            self.blocks = tuple(
                unpickle_block(b["values"], b["mgr_locs"], ndim=ndim)
                for b in state["blocks"]
            )
        else:
            # 抛出错误，不再支持 0.14.1 之前的 pickle 格式
            raise NotImplementedError("pre-0.14.1 pickles are no longer supported")

        # 完成反序列化后的后续处理
        self._post_setstate()

    def _post_setstate(self) -> None:
        # 空方法，用于子类覆盖以执行反序列化后的额外操作
        pass

    @cache_readonly
    def _block(self) -> Block:
        # 返回 self.blocks 的第一个块对象
        return self.blocks[0]

    @final
    @property
    def array(self) -> ArrayLike:
        """
        Quick access to the backing array of the Block.
        """
        # 返回 self.blocks 的第一个块对象的值数组
        return self.blocks[0].values

    # error: Cannot override writeable attribute with read-only property
    @property
    def _blknos(self) -> None:  # type: ignore[override]
        """compat with BlockManager"""
        # 兼容 BlockManager，返回 None
        return None

    # error: Cannot override writeable attribute with read-only property
    @property
    def _blklocs(self) -> None:  # type: ignore[override]
        """compat with BlockManager"""
        # 兼容 BlockManager，返回 None
        return None

    def get_rows_with_mask(self, indexer: npt.NDArray[np.bool_]) -> Self:
        # 类似于 get_slice，但不限于切片索引
        blk = self._block
        # 如果 indexer 的长度大于 0 并且所有元素都为 True
        if len(indexer) > 0 and indexer.all():
            # 返回一个新的实例，使用 blk 的副本和当前索引
            return type(self)(blk.copy(deep=False), self.index)
        # 从 blk 中根据 indexer 获取数组
        array = blk.values[indexer]

        # 如果 indexer 是 numpy 数组并且其 dtype 的种类是 'b'（布尔类型）
        if isinstance(indexer, np.ndarray) and indexer.dtype.kind == "b":
            # 布尔索引总是返回一个 numpy 的副本
            refs = None
        else:
            # TODO(CoW) 理论上只需要在 new_array 是视图时跟踪引用
            refs = blk.refs

        # 创建一个新的 BlockPlacement 对象，指定数组的范围作为位置
        bp = BlockPlacement(slice(0, len(array)))
        # 根据类型创建一个新的块对象，使用 array、bp、1 维度和 refs
        block = type(blk)(array, placement=bp, ndim=1, refs=refs)

        # 从 self.index 中获取新的索引
        new_idx = self.index[indexer]
        # 返回一个新的实例，使用新创建的块和索引
        return type(self)(block, new_idx)
    def get_slice(self, slobj: slice, axis: AxisInt = 0) -> SingleBlockManager:
        # 断言被禁用以提高性能
        # assert isinstance(slobj, slice), type(slobj)
        # 如果指定的轴超出了数据的维度范围，抛出索引错误异常
        if axis >= self.ndim:
            raise IndexError("Requested axis not found in manager")

        # 获取当前块
        blk = self._block
        # 从块中取出切片后的数组
        array = blk.values[slobj]
        # 创建一个块放置对象，用于管理切片后数组的位置信息
        bp = BlockPlacement(slice(0, len(array)))
        # TODO: 目前此方法仅在 groupby SeriesSplitter 中使用，
        # 因此传递引用（refs）还未被测试覆盖到
        # 根据块的类型创建一个新的块，保留切片后的数组和相关参数
        block = type(blk)(array, placement=bp, ndim=1, refs=blk.refs)
        # 获取新的索引对象，使用切片对象切出的部分
        new_index = self.index._getitem_slice(slobj)
        # 返回一个新的 Series 对象，用新的块和索引对象构建
        return type(self)(block, new_index)

    @property
    def index(self) -> Index:
        # 返回当前对象的第一个轴的索引
        return self.axes[0]

    @property
    def dtype(self) -> DtypeObj:
        # 返回当前块的数据类型
        return self._block.dtype

    def get_dtypes(self) -> npt.NDArray[np.object_]:
        # 返回一个包含当前块数据类型的 NumPy 数组
        return np.array([self._block.dtype], dtype=object)

    def external_values(self):
        """Series.values 返回的数组"""
        # 返回当前块的外部值数组
        return self._block.external_values()

    def internal_values(self):
        """Series._values 返回的数组"""
        # 返回当前块的内部值数组
        return self._block.values

    def array_values(self) -> ExtensionArray:
        """Series.array 返回的数组"""
        # 返回当前块的扩展数组值
        return self._block.array_values

    def get_numeric_data(self) -> Self:
        # 如果当前块包含数值数据，则返回一个当前对象的副本，否则返回一个空的当前对象
        if self._block.is_numeric:
            return self.copy(deep=False)
        return self.make_empty()

    @property
    def _can_hold_na(self) -> bool:
        # 返回当前块是否可以包含缺失值的布尔值
        return self._block._can_hold_na

    def setitem_inplace(self, indexer, value) -> None:
        """
        使用索引器设置值。

        对于 SingleBlockManager，这将支持 s[indexer] = value

        这是 `setitem()` 的原地版本，直接在原对象（和块）上进行操作，
        不返回新的 Manager（和 Block），因此不会改变数据类型。
        """
        # 如果当前块存在引用，则复制块对象以避免引用影响
        if not self._has_no_reference(0):
            self.blocks = (self._block.copy(),)
            self._cache.clear()

        # 获取当前数组
        arr = self.array

        # 对于 ndarray，执行元素值的有效性验证
        if isinstance(arr, np.ndarray):
            # 注意：检查 ndarray 而不是 np.dtype 意味着我们排除了 dt64/td64，它们会执行自己的验证。
            value = np_can_hold_element(arr.dtype, value)

        # 如果值是 ndarray，并且维度为 1 且长度为 1，则进行 NumPy 1.25 弃用处理
        if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == 1:
            value = value[0, ...]

        # 在数组中使用索引器设置值
        arr[indexer] = value

    def idelete(self, indexer) -> SingleBlockManager:
        """
        从 SingleBlockManager 删除单个位置。

        确保 self.blocks 不会变为空。
        """
        # 删除索引器指定位置的块中的数据，并获取新的块
        nb = self._block.delete(indexer)[0]
        # 更新块列表，确保不为空
        self.blocks = (nb,)
        # 更新第一个轴的索引，删除索引器指定位置的数据
        self.axes[0] = self.axes[0].delete(indexer)
        # 清除缓存
        self._cache.clear()
        # 返回更新后的 SingleBlockManager 对象
        return self
    def fast_xs(self, loc):
        """
        快速获取交叉截面的方法
        返回数据的视图
        """
        raise NotImplementedError("Use series._values[loc] instead")
        # 抛出未实现错误，建议使用 series._values[loc] 替代

    def set_values(self, values: ArrayLike) -> None:
        """
        将单个数据块的数值就地设置。

        使用时需谨慎！不会检查传入值是否符合当前 Block/SingleBlockManager 的要求（长度、数据类型等），
        也不会正确跟踪引用。
        """
        # NOTE(CoW) 目前仅用于 FrameColumnApply.series_generator，该函数会根据需要手动设置引用
        self.blocks[0].values = values
        # 设置块在管理器中的位置
        self.blocks[0]._mgr_locs = BlockPlacement(slice(len(values)))

    def _equal_values(self, other: Self) -> bool:
        """
        在基类定义的 .equals 中使用。只检查列数值，假定形状和索引已经被检查过。
        """
        # 对于 SingleBlockManager（即 Series）
        if other.ndim != 1:
            return False
        left = self.blocks[0].values
        right = other.blocks[0].values
        return array_equals(left, right)
        # 比较左右两边的数值数组是否相等，返回比较结果

    def grouped_reduce(self, func):
        arr = self.array
        # 对数组进行指定函数的聚合操作
        res = func(arr)
        # 创建默认长度索引
        index = default_index(len(res))

        # 从数组创建与当前对象相同类型的管理器
        mgr = type(self).from_array(res, index)
        return mgr
# --------------------------------------------------------------------
# Constructor Helpers

# 从已有的 Block 列表创建 BlockManager 对象
def create_block_manager_from_blocks(
    blocks: list[Block],
    axes: list[Index],
    consolidate: bool = True,
    verify_integrity: bool = True,
) -> BlockManager:
    # 如果 verify_integrity=False，则调用者需确保以下条件成立：
    #  all(x.shape[-1] == len(axes[1]) for x in blocks)
    #  sum(x.shape[0] for x in blocks) == len(axes[0])
    #  set(x for blk in blocks for x in blk.mgr_locs) == set(range(len(axes[0])))
    #  all(blk.ndim == 2 for blk in blocks)
    # 这使得我们可以安全地传递 verify_integrity=False

    try:
        mgr = BlockManager(blocks, axes, verify_integrity=verify_integrity)
    except ValueError as err:
        arrays = [blk.values for blk in blocks]
        tot_items = sum(arr.shape[0] for arr in arrays)
        raise_construction_error(tot_items, arrays[0].shape[1:], axes, err)

    # 如果 consolidate=True，则在原地合并 BlockManager 对象
    if consolidate:
        mgr._consolidate_inplace()
    return mgr


# 从列数组列表创建 BlockManager 对象
def create_block_manager_from_column_arrays(
    arrays: list[ArrayLike],
    axes: list[Index],
    consolidate: bool,
    refs: list,
) -> BlockManager:
    # 为了性能，禁用断言（调用者需负责验证）
    # assert isinstance(axes, list)
    # assert all(isinstance(x, Index) for x in axes)
    # assert all(isinstance(x, (np.ndarray, ExtensionArray)) for x in arrays)
    # assert all(type(x) is not NumpyExtensionArray for x in arrays)
    # assert all(x.ndim == 1 for x in arrays)
    # assert all(len(x) == len(axes[1]) for x in arrays)
    # assert len(arrays) == len(axes[0])
    # 这最后三个条件足以确保我们可以安全地传递 verify_integrity=False

    try:
        # 根据列数组列表和其他参数形成 Blocks，并创建 BlockManager 对象
        blocks = _form_blocks(arrays, consolidate, refs)
        mgr = BlockManager(blocks, axes, verify_integrity=False)
    except ValueError as e:
        raise_construction_error(len(arrays), arrays[0].shape, axes, e)
    
    # 如果 consolidate=True，则在原地合并 BlockManager 对象
    if consolidate:
        mgr._consolidate_inplace()
    return mgr


# 抛出与构造相关的错误信息
def raise_construction_error(
    tot_items: int,
    block_shape: Shape,
    axes: list[Index],
    e: ValueError | None = None,
) -> NoReturn:
    """抛出关于构造的有用信息"""
    passed = tuple(map(int, [tot_items] + list(block_shape)))
    # 在 DataFrame 构造期间修正用户可见的错误消息
    if len(passed) <= 2:
        passed = passed[::-1]

    implied = tuple(len(ax) for ax in axes)
    # 在 DataFrame 构造期间修正用户可见的错误消息
    if len(implied) <= 2:
        implied = implied[::-1]

    # 返回异常对象而不是立即抛出，以便在调用者处抛出；这样 mypy 可以更好地处理
    if passed == implied and e is not None:
        raise e
    if block_shape[0] == 0:
        raise ValueError("传递了空数据但指定了索引。")
    raise ValueError(f"传递值的形状为 {passed}，索引暗示为 {implied}")
# -----------------------------------------------------------------------

def _grouping_func(tup: tuple[int, ArrayLike]) -> tuple[int, DtypeObj]:
    # 获取数组的 dtype
    dtype = tup[1].dtype

    # 如果是只有一维且为 ExtensionArray 的 dtype，则不需要分组
    if is_1d_only_ea_dtype(dtype):
        # 我们知道这些不会被合并，因此不需要对它们进行分组
        # 这避免了对 CategoricalDtype 对象进行昂贵的比较
        sep = id(dtype)
    else:
        sep = 0

    return sep, dtype


def _form_blocks(arrays: list[ArrayLike], consolidate: bool, refs: list) -> list[Block]:
    # 枚举数组，得到 (索引, 数组) 的元组列表
    tuples = enumerate(arrays)

    # 如果不需要合并，则直接返回未合并的块列表
    if not consolidate:
        return _tuples_to_blocks_no_consolidate(tuples, refs)

    # 当需要合并时，可以忽略 refs（无论是堆叠总是复制，还是 EA 已在调用 dict_to_mgr 中复制）

    # 按 dtype 进行分组
    grouper = itertools.groupby(tuples, _grouping_func)

    nbs: list[Block] = []
    for (_, dtype), tup_block in grouper:
        # 根据 dtype 获取块类型
        block_type = get_block_type(dtype)

        # 如果 dtype 是 np.dtype 类型
        if isinstance(dtype, np.dtype):
            # 检查 dtype 是否为日期时间类型
            is_dtlike = dtype.kind in "mM"

            # 如果 dtype 是 str 或 bytes 类型，转换为 np.object 类型
            if issubclass(dtype.type, (str, bytes)):
                dtype = np.dtype(object)

            # 堆叠数组并返回值和位置信息
            values, placement = _stack_arrays(tup_block, dtype)
            if is_dtlike:
                values = ensure_wrapped_if_datetimelike(values)
            # 创建块对象并添加到块列表中
            blk = block_type(values, placement=BlockPlacement(placement), ndim=2)
            nbs.append(blk)

        # 如果是只有一维且为 ExtensionArray 的 dtype
        elif is_1d_only_ea_dtype(dtype):
            # 创建多个块对象并添加到块列表中
            dtype_blocks = [
                block_type(x[1], placement=BlockPlacement(x[0]), ndim=2)
                for x in tup_block
            ]
            nbs.extend(dtype_blocks)

        else:
            # 对于其他情况，确保块的形状为二维并创建块对象
            dtype_blocks = [
                block_type(
                    ensure_block_shape(x[1], 2), placement=BlockPlacement(x[0]), ndim=2
                )
                for x in tup_block
            ]
            nbs.extend(dtype_blocks)
    # 返回块对象列表
    return nbs


def _tuples_to_blocks_no_consolidate(tuples, refs) -> list[Block]:
    # 在 _form_blocks 中生成的元组的形式为 (placement, array)
    return [
        new_block_2d(
            ensure_block_shape(arr, ndim=2), placement=BlockPlacement(i), refs=ref
        )
        for ((i, arr), ref) in zip(tuples, refs)
    ]


def _stack_arrays(tuples, dtype: np.dtype):
    # 分离出 placement 和 arrays
    placement, arrays = zip(*tuples)

    # 取第一个数组，获取其形状
    first = arrays[0]
    shape = (len(arrays),) + first.shape

    # 创建指定 dtype 的空数组
    stacked = np.empty(shape, dtype=dtype)
    # 堆叠数组
    for i, arr in enumerate(arrays):
        stacked[i] = arr

    return stacked, placement


def _consolidate(blocks: tuple[Block, ...]) -> tuple[Block, ...]:
    """
    合并具有相同 dtype 的块，排除不进行合并的块
    """
    # 按 _consolidate_key 排序块
    gkey = lambda x: x._consolidate_key
    grouper = itertools.groupby(sorted(blocks, key=gkey), gkey)

    new_blocks: list[Block] = []
    # 遍历分组结果
    for _, group in grouper:
        # 将每个分组中的块添加到新块列表中
        new_blocks.extend(group)

    return tuple(new_blocks)
    # 对 grouper 中的每个元素进行迭代，元素格式为 ((_can_consolidate, dtype), group_blocks)
    for (_can_consolidate, dtype), group_blocks in grouper:
        # 调用 _merge_blocks 函数，将 group_blocks 列表作为参数传入
        # 返回结果为 merged_blocks（合并后的块列表）和 _（未使用的占位符）
        merged_blocks, _ = _merge_blocks(
            list(group_blocks), dtype=dtype, can_consolidate=_can_consolidate
        )
        # 调用 extend_blocks 函数，将 merged_blocks（合并后的块列表）和 new_blocks 参数传入
        # 更新 new_blocks 的值
        new_blocks = extend_blocks(merged_blocks, new_blocks)
    # 返回 new_blocks 元组作为函数的结果
    return tuple(new_blocks)
# 合并多个块对象成为一个，根据给定的数据类型和可合并标志
def _merge_blocks(
    blocks: list[Block], dtype: DtypeObj, can_consolidate: bool
) -> tuple[list[Block], bool]:
    # 如果只有一个块对象，直接返回，无需合并
    if len(blocks) == 1:
        return blocks, False

    # 如果可以合并块对象
    if can_consolidate:
        # TODO: 在所有块包含切片且这些切片的组合也是切片的情况下，有优化潜力。
        
        # 合并所有块的管理位置成为一个新的数组
        new_mgr_locs = np.concatenate([b.mgr_locs.as_array for b in blocks])

        # 新值的类型定义
        new_values: ArrayLike

        # 如果块对象的数据类型是 np.dtype 类型
        if isinstance(blocks[0].dtype, np.dtype):
            # 错误：列表推导中存在不兼容类型 List[Union[ndarray, ExtensionArray]]；期望类型为 List[Union[complex, generic, Sequence[Union[int, float, complex, str, bytes, generic]], Sequence[Sequence[Any]], SupportsArray]]
            # 创建一个垂直堆叠的新值数组，类型标记被忽略（type: ignore[misc]）
            new_values = np.vstack([b.values for b in blocks])
        else:
            # 提取块对象中的值数组列表
            bvals = [blk.values for blk in blocks]
            # 将块对象的值数组强制转换为 Sequence[NDArrayBackedExtensionArray] 类型
            bvals2 = cast(Sequence[NDArrayBackedExtensionArray], bvals)
            # 在相同类型的情况下拼接值数组
            new_values = bvals2[0]._concat_same_type(bvals2, axis=0)

        # 根据管理位置排序新的值数组
        argsort = np.argsort(new_mgr_locs)
        new_values = new_values[argsort]
        new_mgr_locs = new_mgr_locs[argsort]

        # 创建新的块放置对象
        bp = BlockPlacement(new_mgr_locs)
        return [new_block_2d(new_values, placement=bp)], True

    # 无法合并 --> 不进行合并操作
    return blocks, False


# 预处理切片或索引器，返回相应的类型和处理后的结果
def _preprocess_slice_or_indexer(
    slice_or_indexer: slice | np.ndarray, length: int, allow_fill: bool
):
    # 如果是切片对象
    if isinstance(slice_or_indexer, slice):
        # 返回类型为 "slice" 的字符串，切片对象本身，以及切片的长度
        return (
            "slice",
            slice_or_indexer,
            libinternals.slice_len(slice_or_indexer, length),
        )
    else:
        # 如果不是切片对象且是 np.ndarray 类型且数据类型的种类是整数
        if (
            not isinstance(slice_or_indexer, np.ndarray)
            or slice_or_indexer.dtype.kind != "i"
        ):
            # 获取对象的类型和数据类型，抛出类型错误
            dtype = getattr(slice_or_indexer, "dtype", None)
            raise TypeError(type(slice_or_indexer), dtype)

        # 确保索引器是平台整数类型
        indexer = ensure_platform_int(slice_or_indexer)
        # 如果不允许填充，可能会转换索引器以适应给定长度
        if not allow_fill:
            indexer = maybe_convert_indices(indexer, length)
        # 返回类型为 "fancy" 的字符串，索引器对象，以及索引器的长度
        return "fancy", indexer, len(indexer)


# 创建一个带有缺失值的数组，根据给定的数据类型、形状和填充值
def make_na_array(dtype: DtypeObj, shape: Shape, fill_value) -> ArrayLike:
    # 如果数据类型是 DatetimeTZDtype 类型
    if isinstance(dtype, DatetimeTZDtype):
        # 注意：排除例如 pyarrow[dt64tz] 的 dtypes
        
        # 将填充值转换为时间戳，并根据单位创建相应的 datetime 数组
        ts = Timestamp(fill_value).as_unit(dtype.unit)
        i8values = np.full(shape, ts._value)
        dt64values = i8values.view(f"M8[{dtype.unit}]")
        # 创建并返回新的 DatetimeArray 对象
        return DatetimeArray._simple_new(dt64values, dtype=dtype)

    # 如果是只能用于一维扩展数组的数据类型
    elif is_1d_only_ea_dtype(dtype):
        dtype = cast(ExtensionDtype, dtype)
        # 构造相应的数组类型
        cls = dtype.construct_array_type()

        # 从空序列创建缺失值数组
        missing_arr = cls._from_sequence([], dtype=dtype)
        ncols, nrows = shape
        assert ncols == 1, ncols
        # 创建一个与形状匹配的空数组，填充为 -1
        empty_arr = -1 * np.ones((nrows,), dtype=np.intp)
        # 取出填充的数组，并根据允许填充的标志使用填充值填充
        return missing_arr.take(empty_arr, allow_fill=True, fill_value=fill_value)
    # 如果 `dtype` 是 `ExtensionDtype` 类型
    elif isinstance(dtype, ExtensionDtype):
        # TODO: 当前情况下没有测试覆盖到这里，但如果我们禁用上面的 `dt64tz` 特殊情况，
        #  就会有一些测试例覆盖到这里（这种方式更快）。
        
        # 构建 `dtype` 对应的数组类型
        cls = dtype.construct_array_type()
        # 创建一个空数组 `missing_arr`，形状为 `shape`，数据类型为 `dtype`
        missing_arr = cls._empty(shape=shape, dtype=dtype)
        # 将 `missing_arr` 数组所有元素填充为 `fill_value`
        missing_arr[:] = fill_value
        # 返回填充后的数组
        return missing_arr
    else:
        # 注意：理论上不应该出现 `dtype` 是整数或布尔型的情况；
        #  如果出现这种情况，`missing_arr.fill` 操作可能会导致结果无意义
        
        # 创建一个形状为 `shape`，数据类型为 `dtype` 的空数组 `missing_arr_np`
        missing_arr_np = np.empty(shape, dtype=dtype)
        # 将 `missing_arr_np` 数组所有元素填充为 `fill_value`
        missing_arr_np.fill(fill_value)
        
        # 如果数据类型的类型码在 "mM" 中，即日期时间相关类型
        if dtype.kind in "mM":
            # 对于日期时间相关类型的数组，确保其被适当地包装
            missing_arr_np = ensure_wrapped_if_datetimelike(missing_arr_np)
        
        # 返回填充后的数组 `missing_arr_np`
        return missing_arr_np
```
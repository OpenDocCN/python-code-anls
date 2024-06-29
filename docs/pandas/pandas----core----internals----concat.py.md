# `D:\src\scipysrc\pandas\pandas\core\internals\concat.py`

```
# 引入未来的类型注解允许，用于在类型检查中使用类型本身的引用
from __future__ import annotations

# 引入类型检查所需的模块和函数
from typing import (
    TYPE_CHECKING,
    cast,
)

# 引入常用的数值计算库 NumPy
import numpy as np

# 从 pandas 私有库中引入特定模块
from pandas._libs import (
    NaT,                     # 表示缺失时间戳的特殊值
    algos as libalgos,       # 算法相关的函数集合
    internals as libinternals,  # 内部数据结构相关的函数集合
    lib,                     # pandas 底层库的接口
)
from pandas._libs.missing import NA  # 表示缺失值的特殊常量
from pandas.util._decorators import cache_readonly  # 缓存只读属性的装饰器

# 从 pandas 核心数据类型转换相关模块中引入函数
from pandas.core.dtypes.cast import (
    ensure_dtype_can_hold_na,  # 确保数据类型可以容纳缺失值
    find_common_type,          # 查找一组数据的共同类型
)
# 从 pandas 核心数据类型常用模块中引入函数
from pandas.core.dtypes.common import (
    is_1d_only_ea_dtype,       # 检查是否为仅 1 维的扩展数组数据类型
    needs_i8_conversion,       # 检查是否需要将数据类型转换为 int64
)
# 从 pandas 核心数据类型连接相关模块中引入函数
from pandas.core.dtypes.concat import concat_compat  # 兼容不同类型的数据连接
# 从 pandas 核心数据类型定义模块中引入数据类型扩展相关类
from pandas.core.dtypes.dtypes import ExtensionDtype  # 扩展数据类型的定义
# 从 pandas 核心数据类型缺失值处理模块中引入函数
from pandas.core.dtypes.missing import is_valid_na_for_dtype  # 检查缺失值是否适用于指定数据类型

# 从 pandas 核心数据构建模块中引入函数
from pandas.core.construction import ensure_wrapped_if_datetimelike  # 确保日期时间数据的包装
# 从 pandas 核心数据内部块模块中引入函数和类
from pandas.core.internals.blocks import (
    ensure_block_shape,    # 确保数据块的形状
    new_block_2d,          # 创建新的二维数据块
)
# 从 pandas 核心数据内部管理模块中引入函数和类
from pandas.core.internals.managers import (
    BlockManager,           # 数据块管理器
    make_na_array,          # 创建指定形状和数据类型的缺失值数组
)

# 如果处于类型检查模式，引入额外的类型检查相关模块
if TYPE_CHECKING:
    from collections.abc import (
        Generator,           # 生成器类型
        Sequence,            # 序列类型
    )

    from pandas._typing import (
        ArrayLike,           # 类数组对象类型
        AxisInt,             # 轴的整数标识类型
        DtypeObj,            # 数据类型对象类型
        Shape,               # 数据形状类型
    )

    from pandas import Index  # pandas 索引对象
    from pandas.core.internals.blocks import (
        Block,               # 数据块对象
        BlockPlacement,     # 数据块放置对象
    )


def concatenate_managers(
    mgrs_indexers, axes: list[Index], concat_axis: AxisInt, copy: bool
) -> BlockManager:
    """
    Concatenate block managers into one.

    Parameters
    ----------
    mgrs_indexers : list of (BlockManager, {axis: indexer,...}) tuples
        要连接的数据块管理器和对应的轴索引器字典
    axes : list of Index
        要操作的索引对象列表
    concat_axis : int
        连接操作的轴
    copy : bool
        是否复制数据块

    Returns
    -------
    BlockManager
        返回合并后的数据块管理器
    """

    needs_copy = copy and concat_axis == 0

    # Assertions disabled for performance
    # for tup in mgrs_indexers:
    #    # caller is responsible for ensuring this
    #    indexers = tup[1]
    #    assert concat_axis not in indexers

    if concat_axis == 0:
        # 对于横向连接的情况，可能需要重新索引列和处理 NA 代理
        mgrs = _maybe_reindex_columns_na_proxy(axes, mgrs_indexers, needs_copy)
        return mgrs[0].concat_horizontal(mgrs, axes)

    if len(mgrs_indexers) > 0 and mgrs_indexers[0][0].nblocks > 0:
        first_dtype = mgrs_indexers[0][0].blocks[0].dtype
        if first_dtype in [np.float64, np.float32]:
            # TODO: support more dtypes here.  This will be simpler once
            #  JoinUnit.is_na behavior is deprecated.
            #  (update 2024-04-13 that deprecation has been enforced)
            if (
                all(_is_homogeneous_mgr(mgr, first_dtype) for mgr, _ in mgrs_indexers)
                and len(mgrs_indexers) > 1
            ):
                # Fastpath!
                # Length restriction is just to avoid having to worry about 'copy'
                shape = tuple(len(x) for x in axes)
                nb = _concat_homogeneous_fastpath(mgrs_indexers, shape, first_dtype)
                return BlockManager((nb,), axes)

    # 对于其他情况，可能需要重新索引列和处理 NA 代理
    mgrs = _maybe_reindex_columns_na_proxy(axes, mgrs_indexers, needs_copy)
    # 如果输入的 mgrs 列表长度为 1，则直接取出第一个元素作为 mgr
    if len(mgrs) == 1:
        mgr = mgrs[0]
        # 创建一个浅拷贝 out，并将其轴设置为指定的 axes
        out = mgr.copy(deep=False)
        out.axes = axes
        # 返回拷贝后的对象 out
        return out

    # 创建一个空列表用于存储数据块
    blocks = []
    # 声明一个类型为 ArrayLike 的变量 values

    # 遍历使用 _get_combined_plan(mgrs) 函数得到的 placement 和 join_units 组合
    for placement, join_units in _get_combined_plan(mgrs):
        # 取出 join_units 中的第一个单元 unit
        unit = join_units[0]
        # 获取该单元的数据块 blk
        blk = unit.block

        # 检查 join_units 是否是一致的，即数据块是否具有相同的类型和结构
        if _is_uniform_join_units(join_units):
            # 如果一致，则从每个 join_unit 中获取数据块的值 vals
            vals = [ju.block.values for ju in join_units]

            # 如果数据块不是扩展类型，则使用 np.concatenate 连接数据块的值，axis=1 表示按列连接
            if not blk.is_extension:
                # _is_uniform_join_units 确保了单一的数据类型，因此可以使用 np.concatenate，
                # 它比 concat_compat 更高效
                # 错误: "concatenate" 的第一个参数具有不兼容的类型
                # "List[Union[ndarray[Any, Any], ExtensionArray]]"；
                # 预期为 "Union[_SupportsArray[dtype[Any]], _NestedSequence[_SupportsArray[dtype[Any]]]]"
                values = np.concatenate(vals, axis=1)  # type: ignore[arg-type]
            # 如果数据块是一维扩展数组类型
            elif is_1d_only_ea_dtype(blk.dtype):
                # TODO(EA2D): 在 2D 扩展数组情况下，不需要特别处理
                values = concat_compat(vals, axis=0, ea_compat_axis=True)
                # 确保数据块形状符合 2D 格式
                values = ensure_block_shape(values, ndim=2)
            else:
                # 否则使用 concat_compat 沿着 axis=1 连接数据块的值
                values = concat_compat(vals, axis=1)

            # 确保如果数据块类似于日期时间类型，则进行包装处理
            values = ensure_wrapped_if_datetimelike(values)

            # 检查是否可以使用快速路径，即块的值类型与新值 values 的类型相同
            fastpath = blk.values.dtype == values.dtype
        else:
            # 如果 join_units 不一致，则使用 _concatenate_join_units 函数连接它们的值
            values = _concatenate_join_units(join_units, copy=copy)
            fastpath = False

        # 如果可以使用快速路径，则创建一个与 blk 相同类别的块 b，使用给定的 placement 放置块
        if fastpath:
            b = blk.make_block_same_class(values, placement=placement)
        else:
            # 否则创建一个新的 2D 数据块 b，将值 values 放置在指定位置
            b = new_block_2d(values, placement=placement)

        # 将生成的块 b 添加到 blocks 列表中
        blocks.append(b)

    # 返回一个新的 BlockManager，其中包含所有生成的数据块 blocks，并使用指定的轴 axes
    return BlockManager(tuple(blocks), axes)
# 定义一个内部函数，用于根据给定的索引和索引器，可能对列进行重新索引的操作。
def _maybe_reindex_columns_na_proxy(
    axes: list[Index],  # 列的索引列表
    mgrs_indexers: list[tuple[BlockManager, dict[int, np.ndarray]]],  # 包含BlockManager和索引器字典的列表
    needs_copy: bool,  # 是否需要复制的标志
) -> list[BlockManager]:  # 返回一个BlockManager对象的列表
    """
    Reindex along columns so that all of the BlockManagers being concatenated
    have matching columns.

    Columns added in this reindexing have dtype=np.void, indicating they
    should be ignored when choosing a column's final dtype.
    """
    new_mgrs = []  # 初始化一个新的BlockManager列表

    for mgr, indexers in mgrs_indexers:
        # 对于轴=0（即列），使用use_na_proxy和only_slice，因此这是一个廉价的重新索引。
        for i, indexer in indexers.items():
            mgr = mgr.reindex_indexer(
                axes[i],  # 使用的轴的索引
                indexers[i],  # 索引器数组
                axis=i,  # 轴的索引
                only_slice=True,  # 仅对i==0有效
                allow_dups=True,  # 允许重复项
                use_na_proxy=True,  # 仅对i==0有效，使用NA代理
            )
        if needs_copy and not indexers:
            mgr = mgr.copy()  # 如果需要复制且没有索引器，则复制该BlockManager对象

        new_mgrs.append(mgr)  # 将处理后的BlockManager对象添加到新的管理器列表中
    return new_mgrs  # 返回更新后的BlockManager对象列表


# 内部函数，用于检查是否可以将给定的Manager视为单个ndarray。
def _is_homogeneous_mgr(mgr: BlockManager, first_dtype: DtypeObj) -> bool:
    """
    Check if this Manager can be treated as a single ndarray.
    """
    if mgr.nblocks != 1:  # 如果BlockManager中的块数不为1，则返回False
        return False
    blk = mgr.blocks[0]
    if not (blk.mgr_locs.is_slice_like and blk.mgr_locs.as_slice.step == 1):
        return False  # 如果块的位置不像切片或步长不为1，则返回False

    return blk.dtype == first_dtype  # 返回块的数据类型是否与给定的第一个数据类型相同的结果


# 内部函数，用于处理具有同质dtype的单块管理器的快速路径连接。
def _concat_homogeneous_fastpath(
    mgrs_indexers,  # 包含BlockManager和索引器字典的列表
    shape: Shape,  # 数据的形状
    first_dtype: np.dtype  # 第一个数据类型
) -> Block:  # 返回一个Block对象
    """
    With single-Block managers with homogeneous dtypes (that can already hold nan),
    we avoid [...]
    """
    # 假设
    #  all(_is_homogeneous_mgr(mgr, first_dtype) for mgr, _ in in mgrs_indexers)

    if all(not indexers for _, indexers in mgrs_indexers):
        # 如果所有索引器都为空，则直接拼接块的值并创建新的块对象
        arrs = [mgr.blocks[0].values.T for mgr, _ in mgrs_indexers]
        arr = np.concatenate(arrs).T
        bp = libinternals.BlockPlacement(slice(shape[0]))
        nb = new_block_2d(arr, bp)
        return nb

    arr = np.empty(shape, dtype=first_dtype)  # 创建一个指定形状和数据类型的空数组

    if first_dtype == np.float64:
        take_func = libalgos.take_2d_axis0_float64_float64  # 如果数据类型是np.float64，则选择对应的取值函数
    else:
        take_func = libalgos.take_2d_axis0_float32_float32  # 否则选择对应的取值函数

    start = 0
    for mgr, indexers in mgrs_indexers:
        mgr_len = mgr.shape[1]
        end = start + mgr_len

        if 0 in indexers:
            take_func(
                mgr.blocks[0].values,  # 块的值
                indexers[0],  # 第一个索引器
                arr[:, start:end],  # 目标数组的切片
            )
        else:
            # 不需要重新索引，可以直接复制值
            arr[:, start:end] = mgr.blocks[0].values

        start += mgr_len

    bp = libinternals.BlockPlacement(slice(shape[0]))  # 创建块的放置对象
    nb = new_block_2d(arr, bp)  # 创建新的块对象
    return nb  # 返回新的块对象


# 内部函数，生成一个组合计划的生成器，其中包含块的放置信息和连接单元的列表。
def _get_combined_plan(
    mgrs: list[BlockManager],  # 包含BlockManager的列表
) -> Generator[tuple[BlockPlacement, list[JoinUnit]], None, None]:  # 返回一个生成器对象，每次生成元组
    max_len = mgrs[0].shape[0]  # 第一个BlockManager的行数
    # 创建包含所有管理器的块号列表
    blknos_list = [mgr.blknos for mgr in mgrs]
    # 调用库函数，获取连接块号和索引器的元组列表
    pairs = libinternals.get_concat_blkno_indexers(blknos_list)
    # 遍历每个连接块号和索引器的元组
    for blknos, bp in pairs:
        # 对于每个块计划，生成多个连接单元
        units_for_bp = []
        for k, mgr in enumerate(mgrs):
            blkno = blknos[k]
            # 根据连接计划、块号和最大长度获取块数据
            nb = _get_block_for_concat_plan(mgr, bp, blkno, max_len=max_len)
            # 创建连接单元对象
            unit = JoinUnit(nb)
            units_for_bp.append(unit)

        # 生成每个连接块索引器和对应的连接单元列表
        yield bp, units_for_bp
def _get_block_for_concat_plan(
    mgr: BlockManager, bp: BlockPlacement, blkno: int, *, max_len: int
) -> Block:
    # 获取给定块编号对应的块对象
    blk = mgr.blocks[blkno]

    # 禁用断言以提升性能：
    # assert bp.is_slice_like
    # assert blkno != -1
    # assert (mgr.blknos[bp] == blkno).all()

    # 如果块的长度与管理位置的长度相同，并且块的管理位置类似于切片并且步长为1
    if len(bp) == len(blk.mgr_locs) and (
        blk.mgr_locs.is_slice_like and blk.mgr_locs.as_slice.step == 1
    ):
        # 直接使用当前块对象
        nb = blk
    else:
        # 获取管理块位置的索引
        ax0_blk_indexer = mgr.blklocs[bp.indexer]

        # 将索引转换为切片，不超过最大长度
        slc = lib.maybe_indices_to_slice(ax0_blk_indexer, max_len)

        # TODO: 在所有现有的测试用例中，我们这里有一个切片。
        # 这种情况是否总是成立？
        if isinstance(slc, slice):
            # 如果是切片，则对块执行切片操作
            nb = blk.slice_block_columns(slc)
        else:
            # 否则，取出块的指定列
            nb = blk.take_block_columns(slc)

    # assert nb.shape == (len(bp), mgr.shape[1])
    # 返回处理后的块对象
    return nb


class JoinUnit:
    def __init__(self, block: Block) -> None:
        self.block = block

    def __repr__(self) -> str:
        # 返回对象的字符串表示，包含块对象的信息
        return f"{type(self).__name__}({self.block!r})"

    def _is_valid_na_for(self, dtype: DtypeObj) -> bool:
        """
        检查是否所有值都是特定类型/数据类型的NA值。
        使用dtype参数增强self.is_na，用于对NA值类型进行额外检查。
        """
        if not self.is_na:
            return False

        # 获取当前对象的块
        blk = self.block

        # 如果块的数据类型的种类是"V"（void类型）
        if blk.dtype.kind == "V":
            return True

        # 如果块的数据类型是对象类型
        if blk.dtype == object:
            # 获取块的值数组
            values = blk.values
            # 检查所有值是否符合指定数据类型的NA值
            return all(is_valid_na_for_dtype(x, dtype) for x in values.ravel(order="K"))

        # 获取块的填充值
        na_value = blk.fill_value

        # 如果填充值是NaT并且块的数据类型不等于dtype
        if na_value is NaT and blk.dtype != dtype:
            # 例如，我们是dt64，其他是td64
            # 填充值匹配但我们不应将blk.values转换为dtype
            # TODO: 如果我们有非NaN dt64/td64，则需要更新
            return False

        # 如果填充值是NA并且需要进行i8转换成dtype
        if na_value is NA and needs_i8_conversion(dtype):
            # FIXME: 修正；test_append_empty_frame_with_timedelta64ns_nat
            # 例如，blk.dtype == "Int64"并且dtype是td64，我们不希望将它们视为匹配
            return False

        # TODO: 最好使用can_hold_element？
        # 判断填充值是否为指定数据类型的NA值
        return is_valid_na_for_dtype(na_value, dtype)

    @cache_readonly
    def is_na(self) -> bool:
        # 获取当前对象的块
        blk = self.block

        # 如果块的数据类型的种类是"V"（void类型），则返回True
        if blk.dtype.kind == "V":
            return True
        
        # 否则返回False
        return False
    def get_reindexed_values(self, empty_dtype: DtypeObj, upcasted_na) -> ArrayLike:
        values: ArrayLike

        # 如果不需要向上转型并且数据块的数据类型不是"V"种类，则无需进行处理，直接返回数据块的值
        if upcasted_na is None and self.block.dtype.kind != "V":
            return self.block.values
        else:
            fill_value = upcasted_na

            # 如果空数据类型支持当前的填充 NA 值
            if self._is_valid_na_for(empty_dtype):
                # 注意：当 self.block.dtype.kind == "V" 时，此条件始终成立
                blk_dtype = self.block.dtype

                # 如果数据块的数据类型是 np.dtype("object")
                if blk_dtype == np.dtype("object"):
                    # 如果希望避免使用 np.nan 进行填充，而是使用 None；已知所有值都是 null
                    values = cast(np.ndarray, self.block.values)
                    if values.size and values[0, 0] is None:
                        fill_value = None

                # 返回一个用指定空数据类型、数据块形状和填充值构建的 NA 数组
                return make_na_array(empty_dtype, self.block.shape, fill_value)

            # 否则，直接返回数据块的值
            return self.block.values
def _concatenate_join_units(join_units: list[JoinUnit], copy: bool) -> ArrayLike:
    """
    Concatenate values from several join units along axis=1.
    """
    # 获取一个空的 dtype，用于后续操作
    empty_dtype = _get_empty_dtype(join_units)

    # 检查 join_units 中是否有 block 的 dtype 是 "V" 类型的
    has_none_blocks = any(unit.block.dtype.kind == "V" for unit in join_units)
    # 根据空的 dtype 和是否有 "V" 类型的 block，确定 upcasted_na 的值
    upcasted_na = _dtype_to_na_value(empty_dtype, has_none_blocks)

    # 从每个 join unit 中获取重新索引后的值，用于拼接
    to_concat = [
        ju.get_reindexed_values(empty_dtype=empty_dtype, upcasted_na=upcasted_na)
        for ju in join_units
    ]

    # 如果其中任何一个 to_concat 的 dtype 是 1 维的 EA 类型
    if any(is_1d_only_ea_dtype(t.dtype) for t in to_concat):
        # TODO(EA2D): 如果所有的 EA 都使用 HybridBlocks，则不需要这个特殊情况处理

        # 错误：ExtensionArray 的 "__getitem__" 没有匹配的重载变体
        # 根据是否是 1 维 EA 类型选择处理方式
        to_concat = [
            t if is_1d_only_ea_dtype(t.dtype) else t[0, :]  # type: ignore[call-overload]
            for t in to_concat
        ]
        # 使用 concat_compat 进行拼接，指定 axis=0，并处理 EA 兼容的轴
        concat_values = concat_compat(to_concat, axis=0, ea_compat_axis=True)
        # 确保 concat_values 的形状为二维
        concat_values = ensure_block_shape(concat_values, 2)

    else:
        # 使用 concat_compat 进行拼接，指定 axis=1
        concat_values = concat_compat(to_concat, axis=1)

    # 返回拼接后的值
    return concat_values


def _dtype_to_na_value(dtype: DtypeObj, has_none_blocks: bool):
    """
    Find the NA value to go with this dtype.
    """
    # 根据 dtype 的类型确定对应的 NA 值
    if isinstance(dtype, ExtensionDtype):
        return dtype.na_value
    elif dtype.kind in "mM":
        return dtype.type("NaT")
    elif dtype.kind in "fc":
        return dtype.type("NaN")
    elif dtype.kind == "b":
        # 与 missing.na_value_for_dtype 不同
        return None
    elif dtype.kind in "iu":
        # 如果没有 "V" 类型的 block，则返回 np.nan
        if not has_none_blocks:
            return None
        return np.nan
    elif dtype.kind == "O":
        return np.nan
    # 如果未处理的 dtype 类型，则抛出 NotImplementedError
    raise NotImplementedError


def _get_empty_dtype(join_units: Sequence[JoinUnit]) -> DtypeObj:
    """
    Return dtype and N/A values to use when concatenating specified units.

    Returned N/A value may be None which means there was no casting involved.

    Returns
    -------
    dtype
    """
    # 如果所有 join_units 的 block 的 dtype 都相等，则使用第一个 join unit 的 dtype
    if lib.dtypes_all_equal([ju.block.dtype for ju in join_units]):
        empty_dtype = join_units[0].block.dtype
        return empty_dtype

    # 检查 join_units 中是否有 "V" 类型的 block
    has_none_blocks = any(unit.block.dtype.kind == "V" for unit in join_units)

    # 获取所有非 NA 的 block 的 dtype
    dtypes = [unit.block.dtype for unit in join_units if not unit.is_na]

    # 找到这些 dtype 的公共类型
    dtype = find_common_type(dtypes)
    # 如果有 "V" 类型的 block，则确保 dtype 能够容纳 NA 值
    if has_none_blocks:
        dtype = ensure_dtype_can_hold_na(dtype)

    # 返回用于拼接的 dtype
    return dtype


def _is_uniform_join_units(join_units: list[JoinUnit]) -> bool:
    """
    Check if the join units consist of blocks of uniform type that can
    be concatenated using Block.concat_same_type instead of the generic
    _concatenate_join_units (which uses `concat_compat`).

    """
    # 检查 join units 中第一个 block 的 dtype 是否是 "V" 类型
    first = join_units[0].block
    if first.dtype.kind == "V":
        return False
    # 返回一个布尔值，表示是否满足以下条件：
    return (
        # 排除 ju.block 为 None 或者与 first 不同类型的情况
        all(type(ju.block) is type(first) for ju in join_units)
        and
        # 检查所有 join_units 中的 ju.block 的数据类型是否与 first 的数据类型一致
        # 例如，DatetimeLikeBlock 可能是 dt64 或者 td64，但它们并不是一致的
        all(
            ju.block.dtype == first.dtype
            # 对于非数值块，我们只希望进行 dtype_equal 检查
            # GH#42092 目前只针对非数值块进行 dtype_equal 检查，这可能会发生变化，但需要废弃提醒
            or ju.block.dtype.kind in "iub"
            for ju in join_units
        )
        and
        # 确保没有会导致缺失值的块（可能导致类型提升），除非我们处理的是扩展数据类型
        all(not ju.is_na or ju.block.is_extension for ju in join_units)
    )
```
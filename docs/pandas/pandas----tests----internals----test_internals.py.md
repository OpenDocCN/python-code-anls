# `D:\src\scipysrc\pandas\pandas\tests\internals\test_internals.py`

```
from datetime import (  # 导入日期和日期时间模块
    date,
    datetime,
)
import itertools  # 导入迭代工具模块
import re  # 导入正则表达式模块

import numpy as np  # 导入NumPy库并使用np别名
import pytest  # 导入pytest测试框架

from pandas._libs.internals import BlockPlacement  # 从pandas库内部导入BlockPlacement类
from pandas.compat import IS64  # 从pandas兼容模块导入IS64常量

from pandas.core.dtypes.common import is_scalar  # 从pandas核心数据类型通用模块导入is_scalar函数

import pandas as pd  # 导入pandas库并使用pd别名
from pandas import (  # 从pandas导入多个类和函数
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalIndex,
    Series,
    Timedelta,
    Timestamp,
    period_range,
)
import pandas._testing as tm  # 导入pandas测试模块并使用tm别名
import pandas.core.algorithms as algos  # 从pandas核心算法模块导入algos
from pandas.core.arrays import (  # 从pandas核心数组模块导入多个类
    DatetimeArray,
    SparseArray,
    TimedeltaArray,
)
from pandas.core.internals import (  # 从pandas核心内部模块导入多个类和函数
    BlockManager,
    SingleBlockManager,
    make_block,
)
from pandas.core.internals.blocks import (  # 从pandas核心内部块模块导入多个函数
    ensure_block_shape,
    maybe_coerce_values,
    new_block,
)


@pytest.fixture(params=[new_block, make_block])  # pytest装置，参数为new_block和make_block函数
def block_maker(request):
    """
    Fixture to test both the internal new_block and pseudo-public make_block.
    """
    return request.param  # 返回request参数的值


@pytest.fixture
def mgr():
    return create_mgr(
        "a: f8; b: object; c: f8; d: object; e: f8;"
        "f: bool; g: i8; h: complex; i: datetime-1; j: datetime-2;"
        "k: M8[ns, US/Eastern]; l: M8[ns, CET];"
    )


def assert_block_equal(left, right):
    tm.assert_numpy_array_equal(left.values, right.values)  # 使用tm模块的函数比较left和right的numpy数组
    assert left.dtype == right.dtype  # 断言left和right的数据类型相同
    assert isinstance(left.mgr_locs, BlockPlacement)  # 断言left的mgr_locs属性是BlockPlacement类的实例
    assert isinstance(right.mgr_locs, BlockPlacement)  # 断言right的mgr_locs属性是BlockPlacement类的实例
    tm.assert_numpy_array_equal(left.mgr_locs.as_array, right.mgr_locs.as_array)  # 使用tm模块的函数比较left和right的mgr_locs的as_array属性


def get_numeric_mat(shape):
    arr = np.arange(shape[0])  # 创建一个NumPy数组，包含从0到shape[0]-1的整数
    return np.lib.stride_tricks.as_strided(
        x=arr, shape=shape, strides=(arr.itemsize,) + (0,) * (len(shape) - 1)
    ).copy()  # 返回arr的视图，具有指定的形状和步幅，然后复制该视图


N = 10  # 定义常量N为10


def create_block(typestr, placement, item_shape=None, num_offset=0, maker=new_block):
    """
    Supported typestr:

        * float, f8, f4, f2
        * int, i8, i4, i2, i1
        * uint, u8, u4, u2, u1
        * complex, c16, c8
        * bool
        * object, string, O
        * datetime, dt, M8[ns], M8[ns, tz]
        * timedelta, td, m8[ns]
        * sparse (SparseArray with fill_value=0.0)
        * sparse_na (SparseArray with fill_value=np.nan)
        * category, category2

    """
    placement = BlockPlacement(placement)  # 使用BlockPlacement类初始化placement
    num_items = len(placement)  # 获取placement的长度作为num_items

    if item_shape is None:  # 如果item_shape未指定
        item_shape = (N,)  # 使用全局常量N作为item_shape的默认值

    shape = (num_items,) + item_shape  # 创建shape元组，包含num_items和item_shape的维度信息

    mat = get_numeric_mat(shape)  # 调用get_numeric_mat函数创建数值矩阵mat，形状为shape

    if typestr in (  # 根据typestr的值执行不同的数据类型处理
        "float",
        "f8",
        "f4",
        "f2",
        "int",
        "i8",
        "i4",
        "i2",
        "i1",
        "uint",
        "u8",
        "u4",
        "u2",
        "u1",
    ):
        values = mat.astype(typestr) + num_offset  # 将mat转换为指定数据类型，并添加num_offset
    elif typestr in ("complex", "c16", "c8"):
        values = 1.0j * (mat.astype(typestr) + num_offset)  # 将mat转换为复数类型，并添加num_offset
    elif typestr in ("object", "string", "O"):
        values = np.reshape([f"A{i:d}" for i in mat.ravel() + num_offset], shape)  # 创建包含字符串的数组，每个元素以A开头，索引加上num_offset
    elif typestr in ("b", "bool"):
        # 如果 typestr 是 "b" 或 "bool"，创建一个形状为 shape 的布尔类型的数组，所有元素为 True
        values = np.ones(shape, dtype=np.bool_)
    elif typestr in ("datetime", "dt", "M8[ns]"):
        # 如果 typestr 是 "datetime", "dt" 或 "M8[ns]"，将 mat 数组中的值乘以 1e9 后转换为日期时间类型的数组
        values = (mat * 1e9).astype("M8[ns]")
    elif typestr.startswith("M8[ns"):
        # 如果 typestr 以 "M8[ns" 开头，表示带有时区信息的日期时间
        m = re.search(r"M8\[ns,\s*(\w+\/?\w*)\]", typestr)
        assert m is not None, f"incompatible typestr -> {typestr}"
        tz = m.groups()[0]
        assert num_items == 1, "must have only 1 num items for a tz-aware"
        # 创建带有时区信息的日期时间索引数组，单位为纳秒
        values = DatetimeIndex(np.arange(N) * 10**9, tz=tz)._data
        # 确保数组的形状符合指定的维度
        values = ensure_block_shape(values, ndim=len(shape))
    elif typestr in ("timedelta", "td", "m8[ns]"):
        # 如果 typestr 是 "timedelta", "td" 或 "m8[ns]"，将 mat 数组中的值乘以 1 后转换为时间间隔类型的数组
        values = (mat * 1).astype("m8[ns]")
    elif typestr in ("category",):
        # 如果 typestr 是 "category"，创建一个分类数据类型的数组，包含指定的分类值
        values = Categorical([1, 1, 2, 2, 3, 3, 3, 3, 4, 4])
    elif typestr in ("category2",):
        # 如果 typestr 是 "category2"，创建一个分类数据类型的数组，包含指定的字符串分类值
        values = Categorical(["a", "a", "a", "a", "b", "b", "c", "c", "c", "d"])
    elif typestr in ("sparse", "sparse_na"):
        # 如果 typestr 是 "sparse" 或 "sparse_na"，创建一个稀疏数组
        if shape[-1] != 10:
            # 确保最后一个维度的大小为 10
            raise NotImplementedError

        assert all(s == 1 for s in shape[:-1])
        if typestr.endswith("_na"):
            fill_value = np.nan
        else:
            fill_value = 0.0
        # 创建稀疏数组，填充值根据 typestr 的后缀确定
        values = SparseArray(
            [fill_value, fill_value, 1, 2, 3, fill_value, 4, 5, fill_value, 6],
            fill_value=fill_value,
        )
        # 获取稀疏数组的内部值视图，并对其进行偏移处理
        arr = values.sp_values.view()
        arr += num_offset - 1
    else:
        # 如果 typestr 不属于以上任何一种情况，抛出异常
        raise ValueError(f'Unsupported typestr: "{typestr}"')

    # 可能需要将 values 数组的值进行类型转换处理
    values = maybe_coerce_values(values)
    # 调用 maker 函数，生成最终的结果对象，并指定数据放置位置和数组的维度
    return maker(values, placement=placement, ndim=len(shape))
# 根据给定的 typestr 和可选的 num_rows 创建 SingleBlockManager 对象
def create_single_mgr(typestr, num_rows=None):
    # 如果未提供 num_rows，则使用默认的 N 值
    if num_rows is None:
        num_rows = N

    # 调用 create_block 函数创建单个块，并用 Index 对象包装起来，返回 SingleBlockManager 对象
    return SingleBlockManager(
        create_block(typestr, placement=slice(0, num_rows), item_shape=()),
        Index(np.arange(num_rows)),
    )


def create_mgr(descr, item_shape=None):
    """
    根据描述字符串构建 BlockManager 对象。

    描述字符串的语法类似于 np.matrix 的初始化方式，如下所示::

        a,b,c: f8; d,e,f: i8

    规则相对简单：

    * 支持的数据类型列表详见 `create_block` 方法
    * 组件以分号分隔
    * 每个组件的格式为 `NAME,NAME,NAME: DTYPE_ID`
    * 冒号和分号周围的空格会被移除
    * 具有相同 DTYPE_ID 的组件会合并成单个块
    * 若要强制使用相同 dtype 的多个块，可以使用 '-SUFFIX' 形式::

        "a:f8-1; b:f8-2; c:f8-foobar"

    """
    # 如果未提供 item_shape，则使用默认的 (N,) 形状
    if item_shape is None:
        item_shape = (N,)

    offset = 0
    mgr_items = []
    block_placements = {}

    # 根据分号拆分描述字符串，处理每个组件
    for d in descr.split(";"):
        d = d.strip()
        if not len(d):
            continue
        names, blockstr = d.partition(":")[::2]
        blockstr = blockstr.strip()
        names = names.strip().split(",")

        # 将组件名称添加到 mgr_items 列表中
        mgr_items.extend(names)
        # 根据组件名称的数量创建对应的位置列表
        placement = list(np.arange(len(names)) + offset)
        try:
            # 尝试将位置信息添加到对应的块字符串键中
            block_placements[blockstr].extend(placement)
        except KeyError:
            block_placements[blockstr] = placement
        offset += len(names)

    # 将 mgr_items 转换为 Index 对象
    mgr_items = Index(mgr_items)

    blocks = []
    num_offset = 0

    # 遍历 block_placements 字典，创建块并添加到 blocks 列表中
    for blockstr, placement in block_placements.items():
        typestr = blockstr.split("-")[0]
        blocks.append(
            create_block(
                typestr, placement, item_shape=item_shape, num_offset=num_offset
            )
        )
        num_offset += len(placement)

    # 根据块的 mgr_locs 属性排序块，并创建 BlockManager 对象返回
    sblocks = sorted(blocks, key=lambda b: b.mgr_locs[0])
    return BlockManager(
        tuple(sblocks),
        [mgr_items] + [Index(np.arange(n)) for n in item_shape],
    )


@pytest.fixture
# 返回一个使用 create_block 创建的浮点块对象
def fblock():
    return create_block("float", [0, 2, 4])


class TestBlock:
    # 测试 create_block 函数是否能够正确创建 int32 类型的块对象
    def test_constructor(self):
        int32block = create_block("i4", [0])
        assert int32block.dtype == np.int32

    # 参数化测试，验证不同类型和数据的块对象是否能够正确进行 pickle 往返
    @pytest.mark.parametrize(
        "typ, data",
        [
            ["float", [0, 2, 4]],
            ["complex", [7]],
            ["object", [1, 3]],
            ["bool", [5]],
        ],
    )
    def test_pickle(self, typ, data):
        blk = create_block(typ, data)
        assert_block_equal(tm.round_trip_pickle(blk), blk)

    # 测试 fblock 的 mgr_locs 属性是否为 BlockPlacement 类型，并验证其内容是否正确
    def test_mgr_locs(self, fblock):
        assert isinstance(fblock.mgr_locs, BlockPlacement)
        tm.assert_numpy_array_equal(
            fblock.mgr_locs.as_array, np.array([0, 2, 4], dtype=np.intp)
        )

    # 测试 fblock 的形状、数据类型和长度是否与其值一致
    def test_attrs(self, fblock):
        assert fblock.shape == fblock.values.shape
        assert fblock.dtype == fblock.values.dtype
        assert len(fblock) == len(fblock.values)
    def test_copy(self, fblock):
        # 复制给定的 fblock 对象
        cop = fblock.copy()
        # 确保复制后的对象与原对象不是同一个引用
        assert cop is not fblock
        # 验证复制后的对象与原对象相等
        assert_block_equal(fblock, cop)

    def test_delete(self, fblock):
        # 复制给定的 fblock 对象
        newb = fblock.copy()
        # 记录原始对象的位置索引
        locs = newb.mgr_locs
        # 删除索引为 0 的元素，并返回新的对象 nb
        nb = newb.delete(0)[0]
        # 验证删除操作并未改变原始对象的位置索引
        assert newb.mgr_locs is locs

        # 确保 nb 对象与 newb 不是同一个引用
        assert nb is not newb

        # 验证 nb 对象的位置索引数组是否为 [2, 4]
        tm.assert_numpy_array_equal(
            nb.mgr_locs.as_array, np.array([2, 4], dtype=np.intp)
        )
        # 验证删除操作后原始对象的第一个值不全为 1
        assert not (newb.values[0] == 1).all()
        # 验证 nb 对象的第一个值全为 1
        assert (nb.values[0] == 1).all()

        # 再次复制给定的 fblock 对象
        newb = fblock.copy()
        # 记录原始对象的位置索引
        locs = newb.mgr_locs
        # 删除索引为 1 的元素，并返回新的对象 nb
        nb = newb.delete(1)
        # 验证 nb 的长度为 2
        assert len(nb) == 2
        # 验证删除操作并未改变原始对象的位置索引
        assert newb.mgr_locs is locs

        # 验证 nb 的第一个元素的位置索引数组是否为 [0]
        tm.assert_numpy_array_equal(
            nb[0].mgr_locs.as_array, np.array([0], dtype=np.intp)
        )
        # 验证 nb 的第二个元素的位置索引数组是否为 [4]
        tm.assert_numpy_array_equal(
            nb[1].mgr_locs.as_array, np.array([4], dtype=np.intp)
        )
        # 验证删除操作后原始对象的第一个值不全为 2
        assert not (newb.values[1] == 2).all()
        # 验证 nb 的第二个元素的第一个值全为 2
        assert (nb[1].values[0] == 2).all()

        # 再次复制给定的 fblock 对象
        newb = fblock.copy()
        # 删除索引为 2 的元素，并返回新的对象 nb
        nb = newb.delete(2)
        # 验证 nb 的长度为 1
        assert len(nb) == 1
        # 验证 nb 的第一个元素的位置索引数组是否为 [0, 2]
        tm.assert_numpy_array_equal(
            nb[0].mgr_locs.as_array, np.array([0, 2], dtype=np.intp)
        )
        # 验证 nb 的第一个元素的第二个值全为 1
        assert (nb[0].values[1] == 1).all()

        # 再次复制给定的 fblock 对象
        newb = fblock.copy()

        # 使用 pytest 验证删除索引为 3 的操作会引发 IndexError 异常
        with pytest.raises(IndexError, match=None):
            newb.delete(3)

    def test_delete_datetimelike(self):
        # 不要在值上使用 np.delete，因为这会将 DTA/TDA 强制转换为 ndarray
        arr = np.arange(20, dtype="i8").reshape(5, 4).view("m8[ns]")
        df = DataFrame(arr)
        blk = df._mgr.blocks[0]
        # 验证块的值是否为 TimedeltaArray 类型
        assert isinstance(blk.values, TimedeltaArray)

        # 删除索引为 1 的元素，并返回新的对象 nb
        nb = blk.delete(1)
        # 验证 nb 的长度为 2
        assert len(nb) == 2
        # 验证 nb 的第一个元素的值是否为 TimedeltaArray 类型
        assert isinstance(nb[0].values, TimedeltaArray)
        # 验证 nb 的第二个元素的值是否为 TimedeltaArray 类型
        assert isinstance(nb[1].values, TimedeltaArray)

        # 将数组视图转换为 DatetimeArray 类型的 DataFrame
        df = DataFrame(arr.view("M8[ns]"))
        blk = df._mgr.blocks[0]
        # 验证块的值是否为 DatetimeArray 类型
        assert isinstance(blk.values, DatetimeArray)

        # 删除索引为 [1, 3] 的元素，并返回新的对象 nb
        nb = blk.delete([1, 3])
        # 验证 nb 的长度为 2
        assert len(nb) == 2
        # 验证 nb 的第一个元素的值是否为 DatetimeArray 类型
        assert isinstance(nb[0].values, DatetimeArray)
        # 验证 nb 的第二个元素的值是否为 DatetimeArray 类型
        assert isinstance(nb[1].values, DatetimeArray)

    def test_split(self):
        # GH#37799
        # 创建一个随机的标准正态分布数组
        values = np.random.default_rng(2).standard_normal((3, 4))
        # 创建一个新的块对象
        blk = new_block(values, placement=BlockPlacement([3, 1, 6]), ndim=2)
        # 对块对象执行拆分操作，并将结果存储在 result 中
        result = list(blk._split())

        # 验证确保我们得到视图而不是副本
        values[:] = -9999
        # 验证块对象的值是否全部为 -9999
        assert (blk.values == -9999).all()

        # 验证结果列表的长度为 3
        assert len(result) == 3
        # 预期的拆分后的块对象列表
        expected = [
            new_block(values[[0]], placement=BlockPlacement([3]), ndim=2),
            new_block(values[[1]], placement=BlockPlacement([1]), ndim=2),
            new_block(values[[2]], placement=BlockPlacement([6]), ndim=2),
        ]
        # 逐一验证每个拆分后的块对象与预期是否相等
        for res, exp in zip(result, expected):
            assert_block_equal(res, exp)
class TestBlockManager:
    # 测试块管理器的属性
    def test_attrs(self):
        # 创建一个块管理器并验证块数
        mgr = create_mgr("a,b,c: f8-1; d,e,f: f8-2")
        assert mgr.nblocks == 2
        assert len(mgr) == 6

    # 测试重复引用位置导致的失败情况
    def test_duplicate_ref_loc_failure(self):
        # 创建包含重复引用位置的临时块管理器
        tmp_mgr = create_mgr("a:bool; a: f8")

        axes, blocks = tmp_mgr.axes, tmp_mgr.blocks

        # 设定第一个和第二个块的管理位置为重叠位置
        blocks[0].mgr_locs = BlockPlacement(np.array([0]))
        blocks[1].mgr_locs = BlockPlacement(np.array([0]))

        # 测试尝试使用重叠引用位置创建块管理器的情况
        msg = "Gaps in blk ref_locs"
        mgr = BlockManager(blocks, axes)
        with pytest.raises(AssertionError, match=msg):
            mgr._rebuild_blknos_and_blklocs()

        # 修改第二个块的管理位置，使其不再与第一个块重叠
        blocks[0].mgr_locs = BlockPlacement(np.array([0]))
        blocks[1].mgr_locs = BlockPlacement(np.array([1]))
        mgr = BlockManager(blocks, axes)
        # 获取第1个块的数据
        mgr.iget(1)

    # 测试 pickle 功能
    def test_pickle(self, mgr):
        # 对块管理器进行往返 pickle 操作
        mgr2 = tm.round_trip_pickle(mgr)
        # 验证从原始和 pickle 后的块管理器创建的 DataFrame 是否相等
        tm.assert_frame_equal(
            DataFrame._from_mgr(mgr, axes=mgr.axes),
            DataFrame._from_mgr(mgr2, axes=mgr2.axes),
        )

        # GH2431
        # 验证 mgr2 是否具有特定属性
        assert hasattr(mgr2, "_is_consolidated")
        assert hasattr(mgr2, "_known_consolidated")

        # 在加载时重置为 False
        assert not mgr2._is_consolidated
        assert not mgr2._known_consolidated

    # 参数化测试非唯一 pickle
    @pytest.mark.parametrize("mgr_string", ["a,a,a:f8", "a: f8; a: i8"])
    def test_non_unique_pickle(self, mgr_string):
        # 创建非唯一定义的块管理器
        mgr = create_mgr(mgr_string)
        # 对块管理器进行往返 pickle 操作
        mgr2 = tm.round_trip_pickle(mgr)
        # 验证从原始和 pickle 后的块管理器创建的 DataFrame 是否相等
        tm.assert_frame_equal(
            DataFrame._from_mgr(mgr, axes=mgr.axes),
            DataFrame._from_mgr(mgr2, axes=mgr2.axes),
        )

    # 测试分类块 pickle
    def test_categorical_block_pickle(self):
        # 创建包含分类数据的块管理器
        mgr = create_mgr("a: category")
        # 对块管理器进行往返 pickle 操作
        mgr2 = tm.round_trip_pickle(mgr)
        # 验证从原始和 pickle 后的块管理器创建的 DataFrame 是否相等
        tm.assert_frame_equal(
            DataFrame._from_mgr(mgr, axes=mgr.axes),
            DataFrame._from_mgr(mgr2, axes=mgr2.axes),
        )

        # 创建包含分类数据的单列块管理器
        smgr = create_single_mgr("category")
        smgr2 = tm.round_trip_pickle(smgr)
        # 验证从原始和 pickle 后的块管理器创建的 Series 是否相等
        tm.assert_series_equal(
            Series()._constructor_from_mgr(smgr, axes=smgr.axes),
            Series()._constructor_from_mgr(smgr2, axes=smgr2.axes),
        )

    # 测试根据索引获取数据块
    def test_iget(self):
        # 创建包含随机数据的块和块管理器
        cols = Index(list("abc"))
        values = np.random.default_rng(2).random((3, 3))
        block = new_block(
            values=values.copy(),
            placement=BlockPlacement(np.arange(3, dtype=np.intp)),
            ndim=values.ndim,
        )
        mgr = BlockManager(blocks=(block,), axes=[cols, Index(np.arange(3))])

        # 验证通过索引获取的数据块与预期的数据相等
        tm.assert_almost_equal(mgr.iget(0).internal_values(), values[0])
        tm.assert_almost_equal(mgr.iget(1).internal_values(), values[1])
        tm.assert_almost_equal(mgr.iget(2).internal_values(), values[2])
    # 定义一个测试方法，用于测试数据管理器的设置操作
    def test_set(self):
        # 创建一个数据管理器，指定包含的列及其类型，并设置项目形状为(3,)
        mgr = create_mgr("a,b,c: int", item_shape=(3,))

        # 在数据管理器的末尾插入一个新项目，名称为"d"，数据为长度为3的字符串数组["foo", "foo", "foo"]
        mgr.insert(len(mgr.items), "d", np.array(["foo"] * 3))

        # 设置数据管理器中索引为1的项目的值为长度为3的字符串数组["bar", "bar", "bar"]
        mgr.iset(1, np.array(["bar"] * 3))

        # 断言索引为0的项目的内部值为长度为3的整数数组[0, 0, 0]
        tm.assert_numpy_array_equal(mgr.iget(0).internal_values(), np.array([0] * 3))

        # 断言索引为1的项目的内部值为长度为3的对象数组["bar", "bar", "bar"]，指定数据类型为np.object_
        tm.assert_numpy_array_equal(
            mgr.iget(1).internal_values(), np.array(["bar"] * 3, dtype=np.object_)
        )

        # 断言索引为2的项目的内部值为长度为3的整数数组[2, 2, 2]
        tm.assert_numpy_array_equal(mgr.iget(2).internal_values(), np.array([2] * 3))

        # 断言索引为3的项目的内部值为长度为3的对象数组["foo", "foo", "foo"]，指定数据类型为np.object_
        tm.assert_numpy_array_equal(
            mgr.iget(3).internal_values(), np.array(["foo"] * 3, dtype=np.object_)
        )

    # 定义一个测试方法，用于测试数据管理器的更改数据类型操作
    def test_set_change_dtype(self, mgr):
        # 在数据管理器的末尾插入一个新项目，名称为"baz"，数据为长度为N的布尔值全为False的数组
        mgr.insert(len(mgr.items), "baz", np.zeros(N, dtype=bool))

        # 设置数据管理器中名称为"baz"的项目的值为长度为N的字符串数组，重复值为"foo"
        mgr.iset(mgr.items.get_loc("baz"), np.repeat("foo", N))

        # 获取名称为"baz"的项目在数据管理器中的索引
        idx = mgr.items.get_loc("baz")

        # 断言获取到的项目的数据类型为np.object_
        assert mgr.iget(idx).dtype == np.object_

        # 对数据管理器执行合并操作，并将结果保存到mgr2中
        mgr2 = mgr.consolidate()

        # 设置mgr2中名称为"baz"的项目的值为长度为N的字符串数组，重复值为"foo"
        mgr2.iset(mgr2.items.get_loc("baz"), np.repeat("foo", N))

        # 再次获取名称为"baz"的项目在mgr2中的索引
        idx = mgr2.items.get_loc("baz")

        # 断言获取到的项目的数据类型为np.object_
        assert mgr2.iget(idx).dtype == np.object_

        # 在mgr2的末尾插入一个新项目，名称为"quux"，数据为长度为N的整数数组，值由标准正态分布生成
        mgr2.insert(
            len(mgr2.items),
            "quux",
            np.random.default_rng(2).standard_normal(N).astype(int),
        )

        # 再次获取名称为"quux"的项目在mgr2中的索引
        idx = mgr2.items.get_loc("quux")

        # 断言获取到的项目的数据类型为int
        assert mgr2.iget(idx).dtype == np.dtype(int)

        # 设置mgr2中名称为"quux"的项目的值为长度为N的浮点数数组，值由标准正态分布生成
        mgr2.iset(
            mgr2.items.get_loc("quux"), np.random.default_rng(2).standard_normal(N)
        )

        # 再次断言获取到的项目的数据类型为np.float64
        assert mgr2.iget(idx).dtype == np.float64

    # 定义一个测试方法，用于测试数据管理器的复制操作
    def test_copy(self, mgr):
        # 对数据管理器执行浅复制操作，将结果保存到cp中
        cp = mgr.copy(deep=False)

        # 遍历原数据管理器mgr和复制后的数据管理器cp的数据块，进行视图的断言
        for blk, cp_blk in zip(mgr.blocks, cp.blocks):
            # 断言复制后的数据块的值与原始数据块的值相等
            tm.assert_equal(cp_blk.values, blk.values)

            # 如果数据块的值是numpy数组，则断言复制后的数据块的值的基础(base)是原始数据块的值的基础(base)
            if isinstance(blk.values, np.ndarray):
                assert cp_blk.values.base is blk.values.base
            else:
                # 对于DatetimeTZBlock，其值是DatetimeIndex类型的值
                # 断言复制后的数据块的值的_ndarray属性的基础(base)是原始数据块的值的_ndarray属性的基础(base)
                assert cp_blk.values._ndarray.base is blk.values._ndarray.base

        # 执行深复制操作，对原数据管理器mgr执行合并操作
        mgr._consolidate_inplace()
        cp = mgr.copy(deep=True)

        # 再次遍历合并后的数据管理器mgr和深复制后的数据管理器cp的数据块，进行复制后的数据块与原始数据块的断言
        for blk, cp_blk in zip(mgr.blocks, cp.blocks):
            bvals = blk.values
            cpvals = cp_blk.values

            # 断言复制后的数据块的值与原始数据块的值相等
            tm.assert_equal(cpvals, bvals)

            # 如果复制后的数据块的值是numpy数组
            if isinstance(cpvals, np.ndarray):
                lbase = cpvals.base
                rbase = bvals.base
            else:
                lbase = cpvals._ndarray.base
                rbase = bvals._ndarray.base

            # 根据数据类型进行复制断言，确保其基础(base)是不同的
            # 对于DatetimeArray类型，其基础(base)要么都为None，要么不同
            if isinstance(cpvals, DatetimeArray):
                assert (lbase is None and rbase is None) or (lbase is not rbase)
            elif not isinstance(cpvals, np.ndarray):
                assert lbase is not rbase
            else:
                assert lbase is None and rbase is None
    def test_sparse(self):
        # 创建一个 BlockManager 对象，包含稀疏数据类型 "a: sparse-1; b: sparse-2"
        mgr = create_mgr("a: sparse-1; b: sparse-2")
        # 断言转换为数组后的数据类型为 np.float64
        assert mgr.as_array().dtype == np.float64

    def test_sparse_mixed(self):
        # 创建一个 BlockManager 对象，包含混合类型数据 "a: sparse-1; b: sparse-2; c: f8"
        mgr = create_mgr("a: sparse-1; b: sparse-2; c: f8")
        # 断言 BlockManager 对象包含的块数量为 3
        assert len(mgr.blocks) == 3
        # 断言 mgr 是 BlockManager 类的实例
        assert isinstance(mgr, BlockManager)

    @pytest.mark.parametrize(
        "mgr_string, dtype",
        [("c: f4; d: f2", np.float32), ("c: f4; d: f2; e: f8", np.float64)],
    )
    def test_as_array_float(self, mgr_string, dtype):
        # 使用给定的字符串创建一个 BlockManager 对象
        mgr = create_mgr(mgr_string)
        # 断言转换为数组后的数据类型符合预期的 dtype
        assert mgr.as_array().dtype == dtype

    @pytest.mark.parametrize(
        "mgr_string, dtype",
        [
            ("a: bool-1; b: bool-2", np.bool_),
            ("a: i8-1; b: i8-2; c: i4; d: i2; e: u1", np.int64),
            ("c: i4; d: i2; e: u1", np.int32),
        ],
    )
    def test_as_array_int_bool(self, mgr_string, dtype):
        # 使用给定的字符串创建一个 BlockManager 对象
        mgr = create_mgr(mgr_string)
        # 断言转换为数组后的数据类型符合预期的 dtype
        assert mgr.as_array().dtype == dtype

    def test_as_array_datetime(self):
        # 创建一个 BlockManager 对象，包含日期时间类型 "h: datetime-1; g: datetime-2"
        mgr = create_mgr("h: datetime-1; g: datetime-2")
        # 断言转换为数组后的数据类型为 "M8[ns]"
        assert mgr.as_array().dtype == "M8[ns]"

    def test_as_array_datetime_tz(self):
        # 创建一个 BlockManager 对象，包含带时区的日期时间类型 "h: M8[ns, US/Eastern]; g: M8[ns, CET]"
        mgr = create_mgr("h: M8[ns, US/Eastern]; g: M8[ns, CET]")
        # 断言获取第一个元素后的数据类型为 "datetime64[ns, US/Eastern]"
        assert mgr.iget(0).dtype == "datetime64[ns, US/Eastern]"
        # 断言获取第二个元素后的数据类型为 "datetime64[ns, CET]"
        assert mgr.iget(1).dtype == "datetime64[ns, CET]"
        # 断言转换为数组后的数据类型为 "object"
        assert mgr.as_array().dtype == "object"

    @pytest.mark.parametrize("t", ["float16", "float32", "float64", "int32", "int64"])
    def test_astype(self, t):
        # 创建一个 BlockManager 对象，包含数据类型 "c: f4; d: f2; e: f8"
        mgr = create_mgr("c: f4; d: f2; e: f8")

        # 将 t 转换为 numpy 的数据类型
        t = np.dtype(t)
        # 将 BlockManager 对象转换为指定类型 tmgr
        tmgr = mgr.astype(t)
        # 断言转换后的第一个块的数据类型为 t
        assert tmgr.iget(0).dtype.type == t
        # 断言转换后的第二个块的数据类型为 t
        assert tmgr.iget(1).dtype.type == t
        # 断言转换后的第三个块的数据类型为 t
        assert tmgr.iget(2).dtype.type == t

        # 创建一个混合数据类型的 BlockManager 对象
        mgr = create_mgr("a,b: object; c: bool; d: datetime; e: f4; f: f2; g: f8")
        # 将 t 转换为 numpy 的数据类型
        t = np.dtype(t)
        # 将 BlockManager 对象转换为指定类型 tmgr，忽略错误
        tmgr = mgr.astype(t, errors="ignore")
        # 断言转换后的第三个块的数据类型为 t
        assert tmgr.iget(2).dtype.type == t
        # 断言转换后的第五个块的数据类型为 t
        assert tmgr.iget(4).dtype.type == t
        # 断言转换后的第六个块的数据类型为 t
        assert tmgr.iget(5).dtype.type == t
        # 断言转换后的第七个块的数据类型为 t
        assert tmgr.iget(6).dtype.type == t

        # 断言转换后的第一个块的数据类型为 np.object_
        assert tmgr.iget(0).dtype.type == np.object_
        # 断言转换后的第二个块的数据类型为 np.object_
        assert tmgr.iget(1).dtype.type == np.object_
        # 如果 t 不等于 np.int64，则断言转换后的第四个块的数据类型为 np.datetime64
        if t != np.int64:
            assert tmgr.iget(3).dtype.type == np.datetime64
        else:
            # 否则断言转换后的第四个块的数据类型为 t
            assert tmgr.iget(3).dtype.type == t
    def test_convert(self, using_infer_string):
        # 定义内部函数用于比较两个管理器对象的数据块
        def _compare(old_mgr, new_mgr):
            """compare the blocks, numeric compare ==, object don't"""
            # 将旧管理器和新管理器的数据块转换为集合并比较它们的长度
            old_blocks = set(old_mgr.blocks)
            new_blocks = set(new_mgr.blocks)
            assert len(old_blocks) == len(new_blocks)

            # 比较非数值类型的数据块
            for b in old_blocks:
                found = False
                for nb in new_blocks:
                    # 如果两个数据块的值相等，则标记为找到
                    if (b.values == nb.values).all():
                        found = True
                        break
                assert found

            # 再次遍历新管理器的数据块，确保每个都能在旧管理器中找到对应的数据块
            for b in new_blocks:
                found = False
                for ob in old_blocks:
                    if (b.values == ob.values).all():
                        found = True
                        break
                assert found

        # noops
        # 创建并转换管理器对象，进行数据块比较
        mgr = create_mgr("f: i8; g: f8")
        new_mgr = mgr.convert()
        _compare(mgr, new_mgr)

        # convert
        # 创建包含不同数据类型的管理器对象，设置数据并转换，验证转换后的数据类型
        mgr = create_mgr("a,b,foo: object; f: i8; g: f8")
        mgr.iset(0, np.array(["1"] * N, dtype=np.object_))
        mgr.iset(1, np.array(["2."] * N, dtype=np.object_))
        mgr.iset(2, np.array(["foo."] * N, dtype=np.object_))
        new_mgr = mgr.convert()
        dtype = "string[pyarrow_numpy]" if using_infer_string else np.object_
        assert new_mgr.iget(0).dtype == dtype
        assert new_mgr.iget(1).dtype == dtype
        assert new_mgr.iget(2).dtype == dtype
        assert new_mgr.iget(3).dtype == np.int64
        assert new_mgr.iget(4).dtype == np.float64

        # 创建包含更多数据类型的管理器对象，设置数据并转换，验证转换后的数据类型
        mgr = create_mgr(
            "a,b,foo: object; f: i4; bool: bool; dt: datetime; i: i8; g: f8; h: f2"
        )
        mgr.iset(0, np.array(["1"] * N, dtype=np.object_))
        mgr.iset(1, np.array(["2."] * N, dtype=np.object_))
        mgr.iset(2, np.array(["foo."] * N, dtype=np.object_))
        new_mgr = mgr.convert()
        assert new_mgr.iget(0).dtype == dtype
        assert new_mgr.iget(1).dtype == dtype
        assert new_mgr.iget(2).dtype == dtype
        assert new_mgr.iget(3).dtype == np.int32
        assert new_mgr.iget(4).dtype == np.bool_
        assert new_mgr.iget(5).dtype.type == np.datetime64
        assert new_mgr.iget(6).dtype == np.int64
        assert new_mgr.iget(7).dtype == np.float64
        assert new_mgr.iget(8).dtype == np.float16

    def test_interleave(self):
        # self
        # 针对不同的数据类型创建管理器对象并验证其数组表示的数据类型
        for dtype in ["f8", "i8", "object", "bool", "complex", "M8[ns]", "m8[ns]"]:
            mgr = create_mgr(f"a: {dtype}")
            assert mgr.as_array().dtype == dtype
            mgr = create_mgr(f"a: {dtype}; b: {dtype}")
            assert mgr.as_array().dtype == dtype
    @pytest.mark.parametrize(
        "mgr_string, dtype",
        [  # 定义参数化测试参数：mgr_string 是字符串，dtype 是数据类型
            ("a: category", "i8"),  # 第一个参数化测试用例
            ("a: category; b: category", "i8"),  # 第二个参数化测试用例
            ("a: category; b: category2", "object"),  # 第三个参数化测试用例
            ("a: category2", "object"),  # 第四个参数化测试用例
            ("a: category2; b: category2", "object"),  # 第五个参数化测试用例
            ("a: f8", "f8"),  # 第六个参数化测试用例
            ("a: f8; b: i8", "f8"),  # 第七个参数化测试用例
            ("a: f4; b: i8", "f8"),  # 第八个参数化测试用例
            ("a: f4; b: i8; d: object", "object"),  # 第九个参数化测试用例
            ("a: bool; b: i8", "object"),  # 第十个参数化测试用例
            ("a: complex", "complex"),  # 第十一个参数化测试用例
            ("a: f8; b: category", "object"),  # 第十二个参数化测试用例
            ("a: M8[ns]; b: category", "object"),  # 第十三个参数化测试用例
            ("a: M8[ns]; b: bool", "object"),  # 第十四个参数化测试用例
            ("a: M8[ns]; b: i8", "object"),  # 第十五个参数化测试用例
            ("a: m8[ns]; b: bool", "object"),  # 第十六个参数化测试用例
            ("a: m8[ns]; b: i8", "object"),  # 第十七个参数化测试用例
            ("a: M8[ns]; b: m8[ns]", "object"),  # 第十八个参数化测试用例
        ],
    )
    def test_interleave_dtype(self, mgr_string, dtype):
        # 测试方法：测试不同的 mgr_string 参数对应的数据类型是否符合预期

        # 测试单一类型
        mgr = create_mgr("a: category")
        assert mgr.as_array().dtype == "i8"  # 断言数据类型为 i8

        # 测试多个类型的组合
        mgr = create_mgr("a: category; b: category2")
        assert mgr.as_array().dtype == "object"  # 断言数据类型为 object

        # 其他单一类型的测试
        mgr = create_mgr("a: category2")
        assert mgr.as_array().dtype == "object"  # 断言数据类型为 object

        # 各种组合类型的测试
        mgr = create_mgr("a: f8")
        assert mgr.as_array().dtype == "f8"  # 断言数据类型为 f8

        mgr = create_mgr("a: f8; b: i8")
        assert mgr.as_array().dtype == "f8"  # 断言数据类型为 f8

        mgr = create_mgr("a: f4; b: i8")
        assert mgr.as_array().dtype == "f8"  # 断言数据类型为 f8

        mgr = create_mgr("a: f4; b: i8; d: object")
        assert mgr.as_array().dtype == "object"  # 断言数据类型为 object

        mgr = create_mgr("a: bool; b: i8")
        assert mgr.as_array().dtype == "object"  # 断言数据类型为 object

        mgr = create_mgr("a: complex")
        assert mgr.as_array().dtype == "complex"  # 断言数据类型为 complex

        mgr = create_mgr("a: f8; b: category")
        assert mgr.as_array().dtype == "f8"  # 断言数据类型为 f8

        mgr = create_mgr("a: M8[ns]; b: category")
        assert mgr.as_array().dtype == "object"  # 断言数据类型为 object

        mgr = create_mgr("a: M8[ns]; b: bool")
        assert mgr.as_array().dtype == "object"  # 断言数据类型为 object

        mgr = create_mgr("a: M8[ns]; b: i8")
        assert mgr.as_array().dtype == "object"  # 断言数据类型为 object

        mgr = create_mgr("a: m8[ns]; b: bool")
        assert mgr.as_array().dtype == "object"  # 断言数据类型为 object

        mgr = create_mgr("a: m8[ns]; b: i8")
        assert mgr.as_array().dtype == "object"  # 断言数据类型为 object

        mgr = create_mgr("a: M8[ns]; b: m8[ns]")
        assert mgr.as_array().dtype == "object"  # 断言数据类型为 object
    def test_consolidate_ordering_issues(self, mgr):
        # 设置 "f" 到 "h" 五个项目的随机正态分布数据，使用相同的随机数生成器确保一致性
        mgr.iset(mgr.items.get_loc("f"), np.random.default_rng(2).standard_normal(N))
        mgr.iset(mgr.items.get_loc("d"), np.random.default_rng(2).standard_normal(N))
        mgr.iset(mgr.items.get_loc("b"), np.random.default_rng(2).standard_normal(N))
        mgr.iset(mgr.items.get_loc("g"), np.random.default_rng(2).standard_normal(N))
        mgr.iset(mgr.items.get_loc("h"), np.random.default_rng(2).standard_normal(N))

        # 确保 mgr 中包含 datetime/tz 块
        cons = mgr.consolidate()
        assert cons.nblocks == 4
        # 获取数字数据后，期望只有一个块
        cons = mgr.consolidate().get_numeric_data()
        assert cons.nblocks == 1
        # 确保第一个块的 mgr_locs 是 BlockPlacement 类型
        assert isinstance(cons.blocks[0].mgr_locs, BlockPlacement)
        # 确保 mgr_locs 的 as_array 数组与索引数组相等
        tm.assert_numpy_array_equal(
            cons.blocks[0].mgr_locs.as_array, np.arange(len(cons.items), dtype=np.intp)
        )

    def test_reindex_items(self):
        # 创建未合并的 mgr，包含 "f8" 和 "f8-2" 块
        mgr = create_mgr("a: f8; b: i8; c: f8; d: i8; e: f8; f: bool; g: f8-2")

        # 对轴0重新索引，选择顺序为 ["g", "c", "a", "d"]
        reindexed = mgr.reindex_axis(["g", "c", "a", "d"], axis=0)
        # reindex_axis 不进行就地合并，因为这可能会导致无法使 _item_cache 失效
        assert not reindexed.is_consolidated()

        # 确保重新索引后的项目顺序正确
        tm.assert_index_equal(reindexed.items, Index(["g", "c", "a", "d"]))
        # 检查数据在重新索引前后的一致性
        tm.assert_almost_equal(
            mgr.iget(6).internal_values(), reindexed.iget(0).internal_values()
        )
        tm.assert_almost_equal(
            mgr.iget(2).internal_values(), reindexed.iget(1).internal_values()
        )
        tm.assert_almost_equal(
            mgr.iget(0).internal_values(), reindexed.iget(2).internal_values()
        )
        tm.assert_almost_equal(
            mgr.iget(3).internal_values(), reindexed.iget(3).internal_values()
        )

    def test_get_numeric_data(self):
        # 创建 mgr 包含多种类型的数据，形状为 (3,)
        mgr = create_mgr(
            "int: int; float: float; complex: complex;"
            "str: object; bool: bool; obj: object; dt: datetime",
            item_shape=(3,),
        )
        # 设置第5个项目为数组 [1, 2, 3]，类型为 object
        mgr.iset(5, np.array([1, 2, 3], dtype=np.object_))

        # 获取 mgr 中的数值数据
        numeric = mgr.get_numeric_data()
        # 确保数值数据的项目顺序正确
        tm.assert_index_equal(numeric.items, Index(["int", "float", "complex", "bool"]))
        # 检查 float 数据的共享性
        tm.assert_almost_equal(
            mgr.iget(mgr.items.get_loc("float")).internal_values(),
            numeric.iget(numeric.items.get_loc("float")).internal_values(),
        )

        # 检查共享后的数据
        numeric.iset(
            numeric.items.get_loc("float"),
            np.array([100.0, 200.0, 300.0]),
            inplace=True,
        )
        tm.assert_almost_equal(
            mgr.iget(mgr.items.get_loc("float")).internal_values(),
            np.array([1.0, 1.0, 1.0]),
        )
    def test_get_bool_data(self):
        # 创建一个 BlockManager 对象，定义数据类型和形状
        mgr = create_mgr(
            "int: int; float: float; complex: complex;"
            "str: object; bool: bool; obj: object; dt: datetime",
            item_shape=(3,),
        )
        # 在索引 6 的位置设置一个包含 True、False、True 的布尔数组
        mgr.iset(6, np.array([True, False, True], dtype=np.object_))

        # 获取布尔数据列
        bools = mgr.get_bool_data()
        # 检查返回的布尔数据列的索引是否为 ["bool"]
        tm.assert_index_equal(bools.items, Index(["bool"]))
        # 检查内部值是否准确返回，并与原始数据进行比较
        tm.assert_almost_equal(
            mgr.iget(mgr.items.get_loc("bool")).internal_values(),
            bools.iget(bools.items.get_loc("bool")).internal_values(),
        )

        # 在索引 0 的位置用新的布尔数组替换原有数据，同时修改原数据
        bools.iset(0, np.array([True, False, True]), inplace=True)
        # 检查更新后的布尔数据是否符合预期
        tm.assert_numpy_array_equal(
            mgr.iget(mgr.items.get_loc("bool")).internal_values(),
            np.array([True, True, True]),
        )

    def test_unicode_repr_doesnt_raise(self):
        # 测试创建 BlockManager 对象时是否能处理 Unicode 字符
        repr(create_mgr("b,\u05d0: object"))

    @pytest.mark.parametrize(
        "mgr_string", ["a,b,c: i8-1; d,e,f: i8-2", "a,a,a: i8-1; b,b,b: i8-2"]
    )
    def test_equals(self, mgr_string):
        # 使用给定的字符串创建 BlockManager 对象，确保两个对象相等
        # unique items
        bm1 = create_mgr(mgr_string)
        # 反转块的顺序，创建新的 BlockManager 对象
        bm2 = BlockManager(bm1.blocks[::-1], bm1.axes)
        # 断言两个 BlockManager 对象相等
        assert bm1.equals(bm2)

    @pytest.mark.parametrize(
        "mgr_string",
        [
            "a:i8;b:f8",  # basic case
            "a:i8;b:f8;c:c8;d:b",  # many types
            "a:i8;e:dt;f:td;g:string",  # more types
            "a:i8;b:category;c:category2",  # categories
            "c:sparse;d:sparse_na;b:f8",  # sparse
        ],
    )
    def test_equals_block_order_different_dtypes(self, mgr_string):
        # GH 9330
        # 使用给定的字符串创建 BlockManager 对象，测试在不同数据类型和块顺序下对象是否相等
        bm = create_mgr(mgr_string)
        # 对 BlockManager 对象的块进行全排列
        block_perms = itertools.permutations(bm.blocks)
        for bm_perm in block_perms:
            # 创建新的 BlockManager 对象，使用不同的块顺序
            bm_this = BlockManager(tuple(bm_perm), bm.axes)
            # 断言两个对象相等
            assert bm.equals(bm_this)
            assert bm_this.equals(bm)

    def test_single_mgr_ctor(self):
        # 创建一个单一列的 BlockManager 对象，数据类型为 f8，行数为 5
        mgr = create_single_mgr("f8", num_rows=5)
        # 检查外部值是否与预期列表相同
        assert mgr.external_values().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]

    @pytest.mark.parametrize("value", [1, "True", [1, 2, 3], 5.0])
    def test_validate_bool_args(self, value):
        # 使用给定的字符串创建 BlockManager 对象，测试替换列表功能是否能处理不同类型的值
        bm1 = create_mgr("a,b,c: i8-1; d,e,f: i8-2")

        # 准备错误消息，用于测试异常情况
        msg = (
            'For argument "inplace" expected type bool, '
            f"received type {type(value).__name__}."
        )
        # 使用 pytest 的异常断言，检查替换列表时传入非布尔值是否会抛出 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            bm1.replace_list([1], [2], inplace=value)

    def test_iset_split_block(self):
        # 创建一个 BlockManager 对象，包含多个列和块
        bm = create_mgr("a,b,c: i8; d: f8")
        # 在索引 0 的位置设置一个新的块，使用 np.array([0]) 初始化
        bm._iset_split_block(0, np.array([0]))
        # 检查块位置数组是否更新正确
        tm.assert_numpy_array_equal(
            bm.blklocs, np.array([0, 0, 1, 0], dtype="int64" if IS64 else "int32")
        )
        # 检查块编号数组是否更新正确
        tm.assert_numpy_array_equal(
            bm.blknos, np.array([0, 0, 0, 1], dtype="int64" if IS64 else "int32")
        )
        # 断言 BlockManager 对象现在包含正确数量的块
        assert len(bm.blocks) == 2
    # 定义一个测试方法，用于测试块管理器的分块值设置功能
    def test_iset_split_block_values(self):
        # 创建一个块管理器，指定列及其类型
        bm = create_mgr("a,b,c: i8; d: f8")
        # 在块管理器中的第一个块索引位置0处，设置块位置为0，以及包含的数据为一个长度为10的列表的NumPy数组
        bm._iset_split_block(0, np.array([0]), np.array([list(range(10))]))
        # 使用断言确认块位置数组（blklocs）与期望值相等，这里根据 IS64 变量决定使用int64或int32类型
        tm.assert_numpy_array_equal(
            bm.blklocs, np.array([0, 0, 1, 0], dtype="int64" if IS64 else "int32")
        )
        # 使用断言确认块编号数组（blknos）与期望值相等，这里根据 IS64 变量决定使用int64或int32类型
        # 第一个索引器当前不与任何块关联
        tm.assert_numpy_array_equal(
            bm.blknos, np.array([0, 2, 2, 1], dtype="int64" if IS64 else "int32")
        )
        # 使用断言确认块管理器中的块数量为3
        assert len(bm.blocks) == 3
# 定义一个函数 _as_array，接受一个 mgr 参数
def _as_array(mgr):
    # 如果 mgr 的维度为 1，则调用 external_values 方法返回外部值
    if mgr.ndim == 1:
        return mgr.external_values()
    # 否则调用 as_array 方法并转置返回数组
    return mgr.as_array().T


# 定义一个测试类 TestIndexing
class TestIndexing:
    # Nosetests 风格的数据驱动测试。
    #
    # 此测试将不同的索引例程应用于块管理器，并将结果与同样的操作在 np.ndarray 上的结果进行比较。
    #
    # 注意：稀疏的块（具有 fill_value != np.nan 的 SparseBlock）在很多测试中失败，
    #       因此被禁用了。

    # 定义 MANAGERS 列表，包含不同类型的块管理器，用于测试
    MANAGERS = [
        create_single_mgr("f8", N),  # 创建一个包含 "f8" 类型数据的单一块管理器
        create_single_mgr("i8", N),  # 创建一个包含 "i8" 类型数据的单一块管理器
        # 2 维
        create_mgr("a,b,c,d,e,f: f8", item_shape=(N,)),  # 创建一个二维块管理器，元素类型为 "f8"
        create_mgr("a,b,c,d,e,f: i8", item_shape=(N,)),  # 创建一个二维块管理器，元素类型为 "i8"
        create_mgr("a,b: f8; c,d: i8; e,f: string", item_shape=(N,)),  # 创建一个复合类型二维块管理器
        create_mgr("a,b: f8; c,d: i8; e,f: f8", item_shape=(N,)),  # 创建一个复合类型二维块管理器
    ]

    # 使用 pytest.mark.parametrize 装饰器，参数化测试用例，参数为 MANAGERS 列表中的每个块管理器
    @pytest.mark.parametrize("mgr", MANAGERS)
    # 定义测试方法，用于测试切片操作在给定的数据管理器上
    def test_get_slice(self, mgr):
        # 定义内部方法，用于验证切片操作的正确性
        def assert_slice_ok(mgr, axis, slobj):
            # 将数据管理器转换为 ndarray
            mat = _as_array(mgr)

            # 如果切片对象是 ndarray，则需要根据轴的长度进行调整
            if isinstance(slobj, np.ndarray):
                ax = mgr.axes[axis]
                # 如果轴的长度不为零且切片对象的长度不为零且长度不等于轴的长度，则进行长度补齐
                if len(ax) and len(slobj) and len(slobj) != len(ax):
                    slobj = np.concatenate(
                        [slobj, np.zeros(len(ax) - len(slobj), dtype=bool)]
                    )

            # 如果切片对象是 slice 类型，则调用数据管理器的 get_slice 方法进行切片操作
            if isinstance(slobj, slice):
                sliced = mgr.get_slice(slobj, axis=axis)
            # 如果数据管理器是一维的，并且切片轴是 0，并且切片对象是布尔型的 ndarray，则调用 get_rows_with_mask 方法
            elif (
                mgr.ndim == 1
                and axis == 0
                and isinstance(slobj, np.ndarray)
                and slobj.dtype == bool
            ):
                sliced = mgr.get_rows_with_mask(slobj)
            else:
                # 如果不支持非切片操作或者不支持轴数大于 0 的操作，则抛出类型错误
                # BlockManager 不支持非切片操作，SingleBlockManager 不支持轴数大于 0
                raise TypeError(slobj)

            # 构造出在 ndarray 上应用切片的索引
            mat_slobj = (slice(None),) * axis + (slobj,)
            # 使用测试工具函数验证切片后的数据与预期数据相等
            tm.assert_numpy_array_equal(
                mat[mat_slobj], _as_array(sliced), check_dtype=False
            )
            # 使用测试工具函数验证切片后的轴数据与预期数据相等
            tm.assert_index_equal(mgr.axes[axis][slobj], sliced.axes[axis])

        # 确保数据管理器的维度小于等于 2，否则抛出异常
        assert mgr.ndim <= 2, mgr.ndim
        # 遍历每一个轴进行切片测试
        for ax in range(mgr.ndim):
            # 对每一个轴分别进行以下切片测试
            assert_slice_ok(mgr, ax, slice(None))
            assert_slice_ok(mgr, ax, slice(3))
            assert_slice_ok(mgr, ax, slice(100))
            assert_slice_ok(mgr, ax, slice(1, 4))
            assert_slice_ok(mgr, ax, slice(3, 0, -2))

            # 如果数据管理器的维度小于 2，则继续进行下面的测试
            if mgr.ndim < 2:
                # 仅支持 2 维的数据管理器才能使用布尔型掩码进行切片

                # 布尔型掩码切片测试
                assert_slice_ok(mgr, ax, np.ones(mgr.shape[ax], dtype=np.bool_))
                assert_slice_ok(mgr, ax, np.zeros(mgr.shape[ax], dtype=np.bool_))

                # 如果轴的长度大于等于 3，则继续进行下面的测试
                if mgr.shape[ax] >= 3:
                    # 使用布尔型数组切片测试
                    assert_slice_ok(mgr, ax, np.arange(mgr.shape[ax]) % 3 == 0)
                    assert_slice_ok(
                        mgr, ax, np.array([True, True, False], dtype=np.bool_)
                    )

    # 使用 pytest 参数化标记，对 MANAGERS 列表中的每个数据管理器执行测试
    @pytest.mark.parametrize("mgr", MANAGERS)
    def test_take(self, mgr):
        # 定义内部函数 assert_take_ok，用于测试 take 方法的正确性
        def assert_take_ok(mgr, axis, indexer):
            # 将 mgr 转换为 numpy 数组
            mat = _as_array(mgr)
            # 调用 take 方法，获取按索引器 indexer 取值后的结果
            taken = mgr.take(indexer, axis)
            # 使用 numpy 的 take 方法进行比较，检查结果是否相等，不检查数据类型
            tm.assert_numpy_array_equal(
                np.take(mat, indexer, axis), _as_array(taken), check_dtype=False
            )
            # 使用 assert_index_equal 检查 mgr 在轴 axis 上索引器 indexer 的结果是否与 taken 的轴相等
            tm.assert_index_equal(mgr.axes[axis].take(indexer), taken.axes[axis])

        # 遍历 mgr 的维度数
        for ax in range(mgr.ndim):
            # 对于每一个轴，测试 take 方法的不同情况
            # 1. 空的索引器数组
            assert_take_ok(mgr, ax, indexer=np.array([], dtype=np.intp))
            # 2. 索引器数组中元素全为0
            assert_take_ok(mgr, ax, indexer=np.array([0, 0, 0], dtype=np.intp))
            # 3. 索引器数组为轴形状的范围内的所有索引
            assert_take_ok(
                mgr, ax, indexer=np.array(list(range(mgr.shape[ax])), dtype=np.intp)
            )

            # 当轴的长度大于等于3时，进行额外的测试
            if mgr.shape[ax] >= 3:
                # 4. 索引器数组为 [0, 1, 2]
                assert_take_ok(mgr, ax, indexer=np.array([0, 1, 2], dtype=np.intp))
                # 5. 索引器数组为 [-1, -2, -3]
                assert_take_ok(mgr, ax, indexer=np.array([-1, -2, -3], dtype=np.intp))

    @pytest.mark.parametrize("mgr", MANAGERS)
    @pytest.mark.parametrize("fill_value", [None, np.nan, 100.0])
    def test_reindex_axis(self, fill_value, mgr):
        # 定义内部函数 assert_reindex_axis_is_ok，用于测试 reindex_axis 方法的正确性
        def assert_reindex_axis_is_ok(mgr, axis, new_labels, fill_value):
            # 将 mgr 转换为 numpy 数组
            mat = _as_array(mgr)
            # 获取新标签 new_labels 的索引器
            indexer = mgr.axes[axis].get_indexer_for(new_labels)

            # 调用 reindex_axis 方法，获取重新索引后的结果
            reindexed = mgr.reindex_axis(new_labels, axis, fill_value=fill_value)
            # 使用 algos.take_nd 方法进行重新索引，比较结果是否相等，不检查数据类型
            tm.assert_numpy_array_equal(
                algos.take_nd(mat, indexer, axis, fill_value=fill_value),
                _as_array(reindexed),
                check_dtype=False,
            )
            # 使用 assert_index_equal 检查重新索引后的轴是否与新标签 new_labels 相等
            tm.assert_index_equal(reindexed.axes[axis], new_labels)

        # 遍历 mgr 的维度数
        for ax in range(mgr.ndim):
            # 对于每一个轴，测试 reindex_axis 方法的不同情况
            # 1. 使用空的 Index 作为新标签
            assert_reindex_axis_is_ok(mgr, ax, Index([]), fill_value)
            # 2. 使用当前轴的标签作为新标签
            assert_reindex_axis_is_ok(mgr, ax, mgr.axes[ax], fill_value)
            # 3. 使用当前轴的第一个标签重复三次作为新标签
            assert_reindex_axis_is_ok(mgr, ax, mgr.axes[ax][[0, 0, 0]], fill_value)
            # 4. 使用新的字符串标签 "foo", "bar", "baz" 作为新标签
            assert_reindex_axis_is_ok(mgr, ax, Index(["foo", "bar", "baz"]), fill_value)
            # 5. 使用新的字符串标签，其中包括当前轴的第一个标签 "foo" 和 "baz"
            assert_reindex_axis_is_ok(
                mgr, ax, Index(["foo", mgr.axes[ax][0], "baz"]), fill_value
            )

            # 当轴的长度大于等于3时，进行额外的测试
            if mgr.shape[ax] >= 3:
                # 6. 使用当前轴的除去最后三个标签的部分作为新标签
                assert_reindex_axis_is_ok(mgr, ax, mgr.axes[ax][:-3], fill_value)
                # 7. 使用当前轴的倒序排列去除最后三个标签的部分作为新标签
                assert_reindex_axis_is_ok(mgr, ax, mgr.axes[ax][-3::-1], fill_value)
                # 8. 使用当前轴的前三个标签重复两次作为新标签
                assert_reindex_axis_is_ok(
                    mgr, ax, mgr.axes[ax][[0, 1, 2, 0, 1, 2]], fill_value
                )
    # 定义测试函数 test_reindex_indexer，用于测试 reindex_indexer 方法
    def test_reindex_indexer(self, fill_value, mgr):
        # 定义内部辅助函数 assert_reindex_indexer_is_ok，用于验证 reindex_indexer 方法的正确性
        def assert_reindex_indexer_is_ok(mgr, axis, new_labels, indexer, fill_value):
            # 将 mgr 转换为 numpy 数组 mat
            mat = _as_array(mgr)
            # 使用 algos.take_nd 函数根据 indexer 在指定轴上重新索引 mat 数组，填充值为 fill_value
            reindexed_mat = algos.take_nd(mat, indexer, axis, fill_value=fill_value)
            # 调用 mgr 对象的 reindex_indexer 方法进行重新索引操作，返回结果为 reindexed
            reindexed = mgr.reindex_indexer(
                new_labels, indexer, axis, fill_value=fill_value
            )
            # 使用 tm.assert_numpy_array_equal 函数验证 reindexed_mat 与 reindexed 的 numpy 数组表示是否相等，不检查数据类型
            tm.assert_numpy_array_equal(
                reindexed_mat, _as_array(reindexed), check_dtype=False
            )
            # 使用 tm.assert_index_equal 函数验证 reindexed 在指定轴上的索引与 new_labels 是否相等
            tm.assert_index_equal(reindexed.axes[axis], new_labels)

        # 遍历 mgr 对象的维度数量
        for ax in range(mgr.ndim):
            # 测试空索引情况，new_labels 为 Index([])，indexer 为空数组，fill_value 为指定值 fill_value
            assert_reindex_indexer_is_ok(
                mgr, ax, Index([]), np.array([], dtype=np.intp), fill_value
            )
            # 测试标准整数索引情况，new_labels 为 mgr.axes[ax]，indexer 为 np.arange(mgr.shape[ax])，fill_value 为指定值 fill_value
            assert_reindex_indexer_is_ok(
                mgr, ax, mgr.axes[ax], np.arange(mgr.shape[ax]), fill_value
            )
            # 测试重复标签索引情况，new_labels 为 Index(["foo"] * mgr.shape[ax])，indexer 为 np.arange(mgr.shape[ax])，fill_value 为指定值 fill_value
            assert_reindex_indexer_is_ok(
                mgr,
                ax,
                Index(["foo"] * mgr.shape[ax]),
                np.arange(mgr.shape[ax]),
                fill_value,
            )
            # 测试反向索引情况，new_labels 为 mgr.axes[ax][::-1]，indexer 为 np.arange(mgr.shape[ax])，fill_value 为指定值 fill_value
            assert_reindex_indexer_is_ok(
                mgr, ax, mgr.axes[ax][::-1], np.arange(mgr.shape[ax]), fill_value
            )
            # 测试反向整数索引情况，new_labels 为 mgr.axes[ax]，indexer 为 np.arange(mgr.shape[ax])[::-1]，fill_value 为指定值 fill_value
            assert_reindex_indexer_is_ok(
                mgr, ax, mgr.axes[ax], np.arange(mgr.shape[ax])[::-1], fill_value
            )
            # 测试指定标签列表索引情况，new_labels 为 Index(["foo", "bar", "baz"])，indexer 为 np.array([0, 0, 0])，fill_value 为指定值 fill_value
            assert_reindex_indexer_is_ok(
                mgr, ax, Index(["foo", "bar", "baz"]), np.array([0, 0, 0]), fill_value
            )
            # 测试指定标签列表和负整数索引情况，new_labels 为 Index(["foo", "bar", "baz"])，indexer 为 np.array([-1, 0, -1])，fill_value 为指定值 fill_value
            assert_reindex_indexer_is_ok(
                mgr, ax, Index(["foo", "bar", "baz"]), np.array([-1, 0, -1]), fill_value
            )
            # 测试混合标签列表和首个轴标签索引情况，new_labels 为 Index(["foo", mgr.axes[ax][0], "baz"])，indexer 为 np.array([-1, -1, -1])，fill_value 为指定值 fill_value
            assert_reindex_indexer_is_ok(
                mgr,
                ax,
                Index(["foo", mgr.axes[ax][0], "baz"]),
                np.array([-1, -1, -1]),
                fill_value,
            )

            # 若 mgr 对象的当前轴的形状大于等于 3，进一步测试标准索引情况
            if mgr.shape[ax] >= 3:
                assert_reindex_indexer_is_ok(
                    mgr,
                    ax,
                    Index(["foo", "bar", "baz"]),
                    np.array([0, 1, 2]),
                    fill_value,
                )
class TestBlockPlacement:
    # 使用 pytest 的参数化装饰器，定义多个参数化测试用例，验证 BlockPlacement 对象的行为
    @pytest.mark.parametrize(
        "slc, expected",
        [
            (slice(0, 4), 4),         # 测试切片长度为4的情况
            (slice(0, 4, 2), 2),      # 测试步长为2时切片长度为2的情况
            (slice(0, 3, 2), 2),      # 测试起始0，终止3，步长2时切片长度为2的情况
            (slice(0, 1, 2), 1),      # 测试步长大于终止与起始差值时切片长度为1的情况
            (slice(1, 0, -1), 1),     # 测试逆向切片时切片长度为1的情况
        ],
    )
    def test_slice_len(self, slc, expected):
        assert len(BlockPlacement(slc)) == expected

    # 测试步长为0时是否会引发 ValueError 异常
    @pytest.mark.parametrize("slc", [slice(1, 1, 0), slice(1, 2, 0)])
    def test_zero_step_raises(self, slc):
        msg = "slice step cannot be zero"
        with pytest.raises(ValueError, match=msg):
            BlockPlacement(slc)

    # 测试负终止索引与负步长时的切片规范化
    def test_slice_canonize_negative_stop(self):
        # GH#37524 negative stop is OK with negative step and positive start
        slc = slice(3, -1, -2)

        bp = BlockPlacement(slc)
        assert bp.indexer == slice(3, None, -2)

    # 测试非法的不受限制切片是否会引发 ValueError 异常
    @pytest.mark.parametrize(
        "slc",
        [
            slice(None, None),
            slice(10, None),
            slice(None, None, -1),
            slice(None, 10, -1),
            slice(-1, None),
            slice(None, -1),
            slice(-1, -1),
            slice(-1, None, -1),
            slice(None, -1, -1),
            slice(-1, -1, -1),
        ],
    )
    def test_unbounded_slice_raises(self, slc):
        msg = "unbounded slice"
        with pytest.raises(ValueError, match=msg):
            BlockPlacement(slc)

    # 测试非切片形式的切片对象是否返回 is_slice_like 属性为 False
    @pytest.mark.parametrize(
        "slc",
        [
            slice(0, 0),
            slice(100, 0),
            slice(100, 100),
            slice(100, 100, -1),
            slice(0, 100, -1),
        ],
    )
    def test_not_slice_like_slices(self, slc):
        assert not BlockPlacement(slc).is_slice_like

    # 测试数组到切片的转换是否正确
    @pytest.mark.parametrize(
        "arr, slc",
        [
            ([0], slice(0, 1, 1)),
            ([100], slice(100, 101, 1)),
            ([0, 1, 2], slice(0, 3, 1)),
            ([0, 5, 10], slice(0, 15, 5)),
            ([0, 100], slice(0, 200, 100)),
            ([2, 1], slice(2, 0, -1)),
        ],
    )
    def test_array_to_slice_conversion(self, arr, slc):
        assert BlockPlacement(arr).as_slice == slc

    # 测试非切片形式的数组对象是否返回 is_slice_like 属性为 False
    @pytest.mark.parametrize(
        "arr",
        [
            [],
            [-1],
            [-1, -2, -3],
            [-10],
            [-1, 0, 1, 2],
            [-2, 0, 2, 4],
            [1, 0, -1],
            [1, 1, 1],
        ],
    )
    def test_not_slice_like_arrays(self, arr):
        assert not BlockPlacement(arr).is_slice_like

    # 测试 BlockPlacement 对象的迭代行为是否正确
    @pytest.mark.parametrize(
        "slc, expected",
        [(slice(0, 3), [0, 1, 2]), (slice(0, 0), []), (slice(3, 0), [])],
    )
    def test_slice_iter(self, slc, expected):
        assert list(BlockPlacement(slc)) == expected
    # 使用 pytest 的装饰器 mark.parametrize 来定义多组参数化测试用例
    @pytest.mark.parametrize(
        "slc, arr",
        [
            # 参数化测试用例1：slice(0, 3) 对应 [0, 1, 2]
            (slice(0, 3), [0, 1, 2]),
            # 参数化测试用例2：slice(0, 0) 对应空列表 []
            (slice(0, 0), []),
            # 参数化测试用例3：slice(3, 0) 对应空列表 []
            (slice(3, 0), []),
            # 参数化测试用例4：slice(3, 0, -1) 对应 [3, 2, 1]
            (slice(3, 0, -1), [3, 2, 1]),
        ],
    )
    # 定义测试方法 test_slice_to_array_conversion，验证 BlockPlacement 类的 as_array 方法
    def test_slice_to_array_conversion(self, slc, arr):
        # 使用 assert_numpy_array_equal 方法比较 BlockPlacement(slc).as_array 和 np.asarray(arr, dtype=np.intp) 是否相等
        tm.assert_numpy_array_equal(
            BlockPlacement(slc).as_array, np.asarray(arr, dtype=np.intp)
        )
    
    # 定义测试方法 test_blockplacement_add，验证 BlockPlacement 类的 add 方法
    def test_blockplacement_add(self):
        # 测试在切片 slice(0, 5) 上执行 add(1) 后的结果是否为 slice(1, 6, 1)
        bpl = BlockPlacement(slice(0, 5))
        assert bpl.add(1).as_slice == slice(1, 6, 1)
        # 测试在切片 slice(0, 5) 上执行 add(np.arange(5)) 后的结果是否为 slice(0, 10, 2)
        assert bpl.add(np.arange(5)).as_slice == slice(0, 10, 2)
        # 测试在切片 slice(0, 5) 上执行 add(np.arange(5, 0, -1)) 后的结果是否为 [5, 5, 5, 5, 5]
        assert list(bpl.add(np.arange(5, 0, -1))) == [5, 5, 5, 5, 5]
    
    # 使用 pytest 的装饰器 mark.parametrize 来定义多组参数化测试用例
    @pytest.mark.parametrize(
        "val, inc, expected",
        [
            # 参数化测试用例1：slice(0, 0) 初始值为 [], 增量为 0，预期结果为 []
            (slice(0, 0), 0, []),
            # 参数化测试用例2：slice(1, 4) 初始值为 [1, 2, 3], 增量为 0，预期结果为 [1, 2, 3]
            (slice(1, 4), 0, [1, 2, 3]),
            # 参数化测试用例3：slice(3, 0, -1) 初始值为 [3, 2, 1], 增量为 0，预期结果为 [3, 2, 1]
            (slice(3, 0, -1), 0, [3, 2, 1]),
            # 参数化测试用例4：[1, 2, 4] 初始值为 [1, 2, 4], 增量为 0，预期结果为 [1, 2, 4]
            ([1, 2, 4], 0, [1, 2, 4]),
            # 参数化测试用例5：slice(0, 0) 初始值为 [], 增量为 10，预期结果为 []
            (slice(0, 0), 10, []),
            # 参数化测试用例6：slice(1, 4) 初始值为 [1, 2, 3], 增量为 10，预期结果为 [11, 12, 13]
            (slice(1, 4), 10, [11, 12, 13]),
            # 参数化测试用例7：slice(3, 0, -1) 初始值为 [3, 2, 1], 增量为 10，预期结果为 [13, 12, 11]
            (slice(3, 0, -1), 10, [13, 12, 11]),
            # 参数化测试用例8：[1, 2, 4] 初始值为 [1, 2, 4], 增量为 10，预期结果为 [11, 12, 14]
            ([1, 2, 4], 10, [11, 12, 14]),
            # 参数化测试用例9：slice(0, 0) 初始值为 [], 增量为 -1，预期结果为 []
            (slice(0, 0), -1, []),
            # 参数化测试用例10：slice(1, 4) 初始值为 [1, 2, 3], 增量为 -1，预期结果为 [0, 1, 2]
            (slice(1, 4), -1, [0, 1, 2]),
            # 参数化测试用例11：[1, 2, 4] 初始值为 [1, 2, 4], 增量为 -1，预期结果为 [0, 1, 3]
            ([1, 2, 4], -1, [0, 1, 3]),
        ],
    )
    # 定义测试方法 test_blockplacement_add_int，验证 BlockPlacement 类的 add 方法对不同类型的输入正确处理
    def test_blockplacement_add_int(self, val, inc, expected):
        # 使用 list(BlockPlacement(val).add(inc)) 比较结果是否与 expected 相等
        assert list(BlockPlacement(val).add(inc)) == expected
    
    # 使用 pytest 的装饰器 mark.parametrize 来定义多组参数化测试用例
    @pytest.mark.parametrize("val", [slice(1, 4), [1, 2, 4]])
    # 定义测试方法 test_blockplacement_add_int_raises，验证 BlockPlacement 类的 add 方法在特定条件下是否会引发 ValueError 异常
    def test_blockplacement_add_int_raises(self, val):
        msg = "iadd causes length change"
        # 使用 pytest.raises 检查是否引发 ValueError 异常，并验证异常消息是否匹配
        with pytest.raises(ValueError, match=msg):
            BlockPlacement(val).add(-10)
class TestCanHoldElement:
    @pytest.fixture(
        params=[
            lambda x: x,                   # 参数化测试数据，每个 lambda 表达式返回一个处理函数
            lambda x: x.to_series(),       # 将输入转换为 Series 对象的处理函数
            lambda x: x._data,             # 返回输入对象的 _data 属性的处理函数
            lambda x: list(x),             # 将输入对象转换为列表的处理函数
            lambda x: x.astype(object),    # 将输入对象转换为 object 类型的处理函数
            lambda x: np.asarray(x),       # 将输入对象转换为 NumPy 数组的处理函数
            lambda x: x[0],                # 返回输入对象的第一个元素的处理函数
            lambda x: x[:0],               # 返回输入对象的空切片的处理函数
        ]
    )
    def element(self, request):
        """
        Functions that take an Index and return an element that should have
        blk._can_hold_element(element) for a Block with this index's dtype.
        """
        return request.param  # 返回 request 中的参数化函数作为测试数据处理函数

    def test_datetime_block_can_hold_element(self):
        block = create_block("datetime", [0])

        assert block._can_hold_element([])  # 检查空列表是否能被 block._can_hold_element 处理

        # We will check that block._can_hold_element iff arr.__setitem__ works
        arr = pd.array(block.values.ravel())

        # coerce None
        assert block._can_hold_element(None)  # 检查 None 是否能被 block._can_hold_element 处理
        arr[0] = None
        assert arr[0] is pd.NaT  # 确保 arr[0] 变为 pd.NaT

        # coerce different types of datetime objects
        vals = [np.datetime64("2010-10-10"), datetime(2010, 10, 10)]
        for val in vals:
            assert block._can_hold_element(val)  # 检查不同类型的 datetime 对象是否能被处理
            arr[0] = val

        val = date(2010, 10, 10)
        assert not block._can_hold_element(val)  # 检查 date 对象是否不能被处理

        msg = (
            "value should be a 'Timestamp', 'NaT', "
            "or array of those. Got 'date' instead."
        )
        with pytest.raises(TypeError, match=msg):
            arr[0] = val  # 确保给定的 TypeError 异常信息与预期一致

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
    def test_interval_can_hold_element_emptylist(self, dtype, element):
        arr = np.array([1, 3, 4], dtype=dtype)
        ii = IntervalIndex.from_breaks(arr)
        blk = new_block(ii._data, BlockPlacement([1]), ndim=2)

        assert blk._can_hold_element([])  # 检查空列表是否能被 blk._can_hold_element 处理
        # TODO: check this holds for all blocks

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
    # 测试给定的数据类型和元素是否能够被区间索引容纳
    def test_interval_can_hold_element(self, dtype, element):
        # 创建一个 NumPy 数组，作为区间索引的断点数组
        arr = np.array([1, 3, 4, 9], dtype=dtype)
        # 从断点数组创建区间索引对象
        ii = IntervalIndex.from_breaks(arr)
        # 使用区间索引数据创建一个新的数据块
        blk = new_block(ii._data, BlockPlacement([1]), ndim=2)

        # 获取指定元素在区间索引上的表现
        elem = element(ii)
        # 检查设置元素后的系列操作，应为 True
        self.check_series_setitem(elem, ii, True)
        # 断言数据块能够容纳该元素
        assert blk._can_hold_element(elem)

        # 注意：为了获得预期的就地修改 Series 行为，需要确保 `elem` 的长度与 `arr` 不同
        # 从部分断点数组创建另一个区间索引对象
        ii2 = IntervalIndex.from_breaks(arr[:-1], closed="neither")
        # 获取在新区间索引上的元素
        elem = element(ii2)
        # 在设置不兼容数据类型元素时会产生警告信息
        msg = "Setting an item of incompatible dtype is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 检查设置元素后的系列操作，应为 False
            self.check_series_setitem(elem, ii, False)
        # 断言数据块不能容纳该元素
        assert not blk._can_hold_element(elem)

        # 创建一个由时间戳构成的区间索引对象
        ii3 = IntervalIndex.from_breaks([Timestamp(1), Timestamp(3), Timestamp(4)])
        # 获取在时间戳区间索引上的元素
        elem = element(ii3)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 检查设置元素后的系列操作，应为 False
            self.check_series_setitem(elem, ii, False)
        # 断言数据块不能容纳该元素
        assert not blk._can_hold_element(elem)

        # 创建一个由时间增量构成的区间索引对象
        ii4 = IntervalIndex.from_breaks([Timedelta(1), Timedelta(3), Timedelta(4)])
        # 获取在时间增量区间索引上的元素
        elem = element(ii4)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 检查设置元素后的系列操作，应为 False
            self.check_series_setitem(elem, ii, False)
        # 断言数据块不能容纳该元素

    # 测试区间索引对象为空列表时的元素容纳情况
    def test_period_can_hold_element_emptylist(self):
        # 创建一个包含 3 个年度周期的期间范围对象
        pi = period_range("2016", periods=3, freq="Y")
        # 使用期间范围数据创建一个新的数据块
        blk = new_block(pi._data.reshape(1, 3), BlockPlacement([1]), ndim=2)

        # 断言数据块能够容纳空列表
        assert blk._can_hold_element([])

    # 测试给定元素是否能够被期间范围对象容纳
    def test_period_can_hold_element(self, element):
        # 创建一个包含 3 个年度周期的期间范围对象
        pi = period_range("2016", periods=3, freq="Y")

        # 获取在期间范围对象上的元素
        elem = element(pi)
        # 检查设置元素后的系列操作，应为 True
        self.check_series_setitem(elem, pi, True)

        # 注意：为了获得预期的就地修改 Series 行为，需要确保 `elem` 的长度与 `arr` 不同
        # 将期间范围转换为每日频率并去掉最后一个元素
        pi2 = pi.asfreq("D")[:-1]
        # 获取在修改后的期间范围上的元素
        elem = element(pi2)
        # 在设置不兼容数据类型元素时会产生警告信息
        msg = "Setting an item of incompatible dtype is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 检查设置元素后的系列操作，应为 False
            self.check_series_setitem(elem, pi, False)

        # 将期间范围转换为时间戳并去掉最后一个元素
        dti = pi.to_timestamp("s")[:-1]
        # 获取在时间戳区间索引上的元素
        elem = element(dti)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 检查设置元素后的系列操作，应为 False
            self.check_series_setitem(elem, pi, False)

    # 检查给定对象和元素是否能够被管理器的第一个数据块容纳
    def check_can_hold_element(self, obj, elem, inplace: bool):
        # 获取对象的第一个数据块
        blk = obj._mgr.blocks[0]
        if inplace:
            # 断言数据块能够容纳该元素
            assert blk._can_hold_element(elem)
        else:
            # 断言数据块不能容纳该元素
            assert not blk._can_hold_element(elem)
    `
        # 检查 Series 对象的设置项操作，验证元素是否可以被插入
        def check_series_setitem(self, elem, index: Index, inplace: bool):
            # 复制索引数据以确保不修改原始数据
            arr = index._data.copy()
            # 根据复制的索引数据创建 Series 对象，copy=False 表示数据共享
            ser = Series(arr, copy=False)
    
            # 调用函数检查是否可以将元素 elem 插入到 Series 中
            self.check_can_hold_element(ser, elem, inplace)
    
            # 如果 elem 是标量值，则将其赋值给 ser 的第一个元素
            if is_scalar(elem):
                ser[0] = elem
            else:
                # 否则，将 elem 赋值给 ser 的前 len(elem) 个位置
                ser[: len(elem)] = elem
    
            # 如果 inplace 为 True，确保设置是在原地完成的，即 ser 的数组仍然是 arr
            if inplace:
                assert ser.array is arr  # 即设置是原地完成的
            else:
                # 如果 inplace 为 False，确保 ser 的 dtype 是 object 类型
                assert ser.dtype == object
class TestShouldStore:
    def test_should_store_categorical(self):
        # 创建一个包含三个类别的 Categorical 对象
        cat = Categorical(["A", "B", "C"])
        # 使用 Categorical 对象创建一个 DataFrame
        df = DataFrame(cat)
        # 获取 DataFrame 的第一个数据块
        blk = df._mgr.blocks[0]

        # 检查数据类型匹配
        assert blk.should_store(cat)
        # 检查移除最后一个元素后的 Categorical 对象是否匹配
        assert blk.should_store(cat[:-1])

        # 检查不同数据类型的 Categorical 对象是否不匹配
        assert not blk.should_store(cat.as_ordered())

        # 检查使用 ndarray 而不是 Categorical 对象时是否不匹配
        assert not blk.should_store(np.asarray(cat))


def test_validate_ndim():
    # 创建一个包含两个浮点数的 ndarray
    values = np.array([1.0, 2.0])
    # 创建一个 BlockPlacement 对象
    placement = BlockPlacement(slice(2))
    # 设置错误信息字符串
    msg = r"Wrong number of dimensions. values.ndim != ndim \[1 != 2\]"

    # 设置废弃信息字符串
    depr_msg = "make_block is deprecated"
    # 使用 pytest 检查是否会引发 ValueError，并匹配特定错误信息和废弃警告信息
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(DeprecationWarning, match=depr_msg):
            make_block(values, placement, ndim=2)


def test_block_shape():
    # 创建一个包含整数的 Index 对象
    idx = Index([0, 1, 2, 3, 4])
    # 创建一个 Series 对象，并重新索引为指定 Index
    a = Series([1, 2, 3]).reindex(idx)
    # 创建一个包含类别数据的 Series 对象，并重新索引为指定 Index
    b = Series(Categorical([1, 2, 3])).reindex(idx)

    # 检查两个 Series 的第一个数据块的位置索引是否相同
    assert a._mgr.blocks[0].mgr_locs.indexer == b._mgr.blocks[0].mgr_locs.indexer


def test_make_block_no_pandas_array(block_maker):
    # 创建一个 NumpyExtensionArray 对象
    arr = pd.arrays.NumpyExtensionArray(np.array([1, 2]))

    # 设置废弃信息字符串
    depr_msg = "make_block is deprecated"
    # 根据 block_maker 的不同，确定是否设置 DeprecationWarning
    warn = DeprecationWarning if block_maker is make_block else None

    # 对于 NumpyExtensionArray，没有指定 dtype 的情况
    with tm.assert_produces_warning(warn, match=depr_msg):
        # 使用 block_maker 创建一个数据块，设置 ndim 为数组的维数
        result = block_maker(arr, BlockPlacement(slice(len(arr))), ndim=arr.ndim)
    # 检查结果的数据类型是否为整数或无符号整数
    assert result.dtype.kind in ["i", "u"]

    if block_maker is make_block:
        # 对于 new_block，需要调用者解包 NumpyExtensionArray
        assert result.is_extension is False

        # 对于 NumpyExtensionArray，指定了 NumpyEADtype
        with tm.assert_produces_warning(warn, match=depr_msg):
            # 使用 block_maker 创建数据块，指定 dtype 和 ndim
            result = block_maker(arr, slice(len(arr)), dtype=arr.dtype, ndim=arr.ndim)
        # 检查结果的数据类型是否为整数或无符号整数，且不是扩展类型
        assert result.dtype.kind in ["i", "u"]
        assert result.is_extension is False

        # 对于 new_block 不再接受 dtype 关键字
        # 对于 ndarray，指定了 NumpyEADtype
        with tm.assert_produces_warning(warn, match=depr_msg):
            # 使用 block_maker 创建数据块，转换为 ndarray，指定 dtype 和 ndim
            result = block_maker(
                arr.to_numpy(), slice(len(arr)), dtype=arr.dtype, ndim=arr.ndim
            )
        # 检查结果的数据类型是否为整数或无符号整数，且不是扩展类型
        assert result.dtype.kind in ["i", "u"]
        assert result.is_extension is False
```
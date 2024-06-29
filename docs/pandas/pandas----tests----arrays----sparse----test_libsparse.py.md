# `D:\src\scipysrc\pandas\pandas\tests\arrays\sparse\test_libsparse.py`

```
import operator  # 导入 operator 模块，用于函数式编程中的操作符函数

import numpy as np  # 导入 numpy 库，用于科学计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

import pandas._libs.sparse as splib  # 导入 pandas 稀疏数据结构的底层库
import pandas.util._test_decorators as td  # 导入 pandas 测试装饰器相关模块

from pandas import Series  # 从 pandas 中导入 Series 类
import pandas._testing as tm  # 导入 pandas 测试相关模块
from pandas.core.arrays.sparse import (  # 从 pandas 稀疏数据结构中导入以下类和函数
    BlockIndex,  # 块索引类
    IntIndex,  # 整数索引类
    make_sparse_index,  # 创建稀疏索引的函数
)


@pytest.fixture  # pytest 的测试装置，用于设置测试的前置条件
def test_length():
    return 20  # 返回一个整数 20，作为测试的长度条件


@pytest.fixture(  # pytest 的测试装置，接受参数的形式
    params=[  # 参数化的测试用例列表
        [
            [0, 7, 15],  # 第一个参数的第一个测试数据列表
            [3, 5, 5],  # 第一个参数的第二个测试数据列表
            [2, 9, 14],  # 第一个参数的第三个测试数据列表
            [2, 3, 5],  # 第一个参数的第四个测试数据列表
            [2, 9, 15],  # 第一个参数的第五个测试数据列表
            [1, 3, 4],  # 第一个参数的第六个测试数据列表
        ],
        [
            [0, 5],  # 第二个参数的第一个测试数据列表
            [4, 4],  # 第二个参数的第二个测试数据列表
            [1],  # 第二个参数的第三个测试数据列表
            [4],  # 第二个参数的第四个测试数据列表
            [1],  # 第二个参数的第五个测试数据列表
            [3],  # 第二个参数的第六个测试数据列表
        ],
        [
            [0],  # 第三个参数的第一个测试数据列表
            [10],  # 第三个参数的第二个测试数据列表
            [0, 5],  # 第三个参数的第三个测试数据列表
            [3, 7],  # 第三个参数的第四个测试数据列表
            [0, 5],  # 第三个参数的第五个测试数据列表
            [3, 5],  # 第三个参数的第六个测试数据列表
        ],
        [
            [10],  # 第四个参数的第一个测试数据列表
            [5],  # 第四个参数的第二个测试数据列表
            [0, 12],  # 第四个参数的第三个测试数据列表
            [5, 3],  # 第四个参数的第四个测试数据列表
            [12],  # 第四个参数的第五个测试数据列表
            [3],  # 第四个参数的第六个测试数据列表
        ],
        [
            [0, 10],  # 第五个参数的第一个测试数据列表
            [4, 6],  # 第五个参数的第二个测试数据列表
            [5, 17],  # 第五个参数的第三个测试数据列表
            [4, 2],  # 第五个参数的第四个测试数据列表
            [],  # 第五个参数的第五个测试数据列表
            [],  # 第五个参数的第六个测试数据列表
        ],
        [
            [0],  # 第六个参数的第一个测试数据列表
            [5],  # 第六个参数的第二个测试数据列表
            [],  # 第六个参数的第三个测试数据列表
            [],  # 第六个参数的第四个测试数据列表
            [],  # 第六个参数的第五个测试数据列表
            [],  # 第六个参数的第六个测试数据列表
        ],
    ],
    ids=[  # 参数化测试用例的标识列表
        "plain_case",  # 第一个测试用例的标识
        "delete_blocks",  # 第二个测试用例的标识
        "split_blocks",  # 第三个测试用例的标识
        "skip_block",  # 第四个测试用例的标识
        "no_intersect",  # 第五个测试用例的标识
        "one_empty",  # 第六个测试用例的标识
    ],
)
def cases(request):
    return request.param  # 返回参数化的测试用例数据


class TestSparseIndexUnion:
    @pytest.mark.parametrize(  # 使用 pytest 参数化装饰器标记测试函数
        "xloc, xlen, yloc, ylen, eloc, elen",  # 参数化的参数列表
        [  # 参数化的测试用例列表
            [[0], [5], [5], [4], [0], [9]],  # 第一个测试用例数据
            [[0, 10], [5, 5], [2, 17], [5, 2], [0, 10, 17], [7, 5, 2]],  # 第二个测试用例数据
            [[1], [5], [3], [5], [1], [7]],  # 第三个测试用例数据
            [[2, 10], [4, 4], [4], [8], [2], [12]],  # 第四个测试用例数据
            [[0, 5], [3, 5], [0], [7], [0], [10]],  # 第五个测试用例数据
            [[2, 10], [4, 4], [4, 13], [8, 4], [2], [15]],  # 第六个测试用例数据
            [[2], [15], [4, 9, 14], [3, 2, 2], [2], [15]],  # 第七个测试用例数据
            [[0, 10], [3, 3], [5, 15], [2, 2], [0, 5, 10, 15], [3, 2, 3, 2]],  # 第八个测试用例数据
        ],
    )
    def test_index_make_union(self, xloc, xlen, yloc, ylen, eloc, elen, test_length):
        # 定义一个测试方法，用于测试索引合并功能
        # Case 1
        # x: ----
        # y:     ----
        # r: --------
        # Case 2
        # x: -----     -----
        # y:   -----          --
        # Case 3
        # x: ------
        # y:    -------
        # r: ----------
        # Case 4
        # x: ------  -----
        # y:    -------
        # r: -------------
        # Case 5
        # x: ---  -----
        # y: -------
        # r: -------------
        # Case 6
        # x: ------  -----
        # y:    -------  ---
        # r: -------------
        # Case 7
        # x: ----------------------
        # y:   ----  ----   ---
        # r: ----------------------
        # Case 8
        # x: ----       ---
        # y:       ---       ---
        # 通过给定的位置、长度创建 BlockIndex 对象 xindex 和 yindex
        xindex = BlockIndex(test_length, xloc, xlen)
        yindex = BlockIndex(test_length, yloc, ylen)
        # 调用 make_union 方法进行索引合并
        bresult = xindex.make_union(yindex)
        # 断言 bresult 是 BlockIndex 类型的对象
        assert isinstance(bresult, BlockIndex)
        # 断言 bresult 的块位置和块长度数组与预期的 eloc 和 elen 相等
        tm.assert_numpy_array_equal(bresult.blocs, np.array(eloc, dtype=np.int32))
        tm.assert_numpy_array_equal(bresult.blengths, np.array(elen, dtype=np.int32))

        # 将 xindex 和 yindex 转换为 IntIndex 类型的对象
        ixindex = xindex.to_int_index()
        iyindex = yindex.to_int_index()
        # 调用 make_union 方法进行整数索引合并
        iresult = ixindex.make_union(iyindex)
        # 断言 iresult 是 IntIndex 类型的对象
        assert isinstance(iresult, IntIndex)
        # 断言 iresult 的索引数组与 bresult 转换为整数索引后的 indices 相等
        tm.assert_numpy_array_equal(iresult.indices, bresult.to_int_index().indices)

    def test_int_index_make_union(self):
        # 测试 IntIndex 类的 make_union 方法
        # Case 1: 非空数组合并
        a = IntIndex(5, np.array([0, 3, 4], dtype=np.int32))
        b = IntIndex(5, np.array([0, 2], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([0, 2, 3, 4], np.int32))
        assert res.equals(exp)

        # Case 2: 一个数组为空，另一个非空
        a = IntIndex(5, np.array([], dtype=np.int32))
        b = IntIndex(5, np.array([0, 2], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([0, 2], np.int32))
        assert res.equals(exp)

        # Case 3: 两个数组均为空
        a = IntIndex(5, np.array([], dtype=np.int32))
        b = IntIndex(5, np.array([], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([], np.int32))
        assert res.equals(exp)

        # Case 4: 两个数组相同
        a = IntIndex(5, np.array([0, 1, 2, 3, 4], dtype=np.int32))
        b = IntIndex(5, np.array([0, 1, 2, 3, 4], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([0, 1, 2, 3, 4], np.int32))
        assert res.equals(exp)

        # Case 5: 长度不匹配，预期引发 ValueError 异常
        a = IntIndex(5, np.array([0, 1], dtype=np.int32))
        b = IntIndex(4, np.array([0, 1], dtype=np.int32))
        msg = "Indices must reference same underlying length"
        with pytest.raises(ValueError, match=msg):
            a.make_union(b)
class TestSparseIndexIntersect:
    # 跳过在Windows上运行的测试
    @td.skip_if_windows
    def test_intersect(self, cases, test_length):
        # 解包测试用例参数
        xloc, xlen, yloc, ylen, eloc, elen = cases
        # 创建 BlockIndex 对象
        xindex = BlockIndex(test_length, xloc, xlen)
        yindex = BlockIndex(test_length, yloc, ylen)
        expected = BlockIndex(test_length, eloc, elen)
        longer_index = BlockIndex(test_length + 1, yloc, ylen)

        # 测试两个索引的交集操作
        result = xindex.intersect(yindex)
        assert result.equals(expected)
        result = xindex.to_int_index().intersect(yindex.to_int_index())
        assert result.equals(expected.to_int_index())

        msg = "Indices must reference same underlying length"
        # 测试不同长度索引的交集操作是否引发异常
        with pytest.raises(Exception, match=msg):
            xindex.intersect(longer_index)
        with pytest.raises(Exception, match=msg):
            xindex.to_int_index().intersect(longer_index.to_int_index())

    # 测试空索引的交集操作
    def test_intersect_empty(self):
        xindex = IntIndex(4, np.array([], dtype=np.int32))
        yindex = IntIndex(4, np.array([2, 3], dtype=np.int32))
        assert xindex.intersect(yindex).equals(xindex)
        assert yindex.intersect(xindex).equals(xindex)

        xindex = xindex.to_block_index()
        yindex = yindex.to_block_index()
        assert xindex.intersect(yindex).equals(xindex)
        assert yindex.intersect(xindex).equals(xindex)

    @pytest.mark.parametrize(
        "case",
        [
            # 参数 2 给 "IntIndex" 的类型为 "ndarray[Any, dtype[signedinteger[_32Bit]]]"，期望类型为 "Sequence[int]"
            IntIndex(5, np.array([1, 2], dtype=np.int32)),  # type: ignore[arg-type]
            IntIndex(5, np.array([0, 2, 4], dtype=np.int32)),  # type: ignore[arg-type]
            IntIndex(0, np.array([], dtype=np.int32)),  # type: ignore[arg-type]
            IntIndex(5, np.array([], dtype=np.int32)),  # type: ignore[arg-type]
        ],
    )
    # 测试相同索引的交集操作
    def test_intersect_identical(self, case):
        assert case.intersect(case).equals(case)
        case = case.to_block_index()
        assert case.intersect(case).equals(case)


class TestSparseIndexCommon:
    # 测试整数类型的稀疏索引
    def test_int_internal(self):
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind="integer")
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.indices, np.array([2, 3], dtype=np.int32))

        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind="integer")
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.indices, np.array([], dtype=np.int32))

        idx = make_sparse_index(
            4, np.array([0, 1, 2, 3], dtype=np.int32), kind="integer"
        )
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.indices, np.array([0, 1, 2, 3], dtype=np.int32))
    def test_block_internal(self):
        # 创建一个稀疏索引对象，种类为“block”，包含索引值为2和3的数据点
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind="block")
        # 断言确保返回的索引对象是 BlockIndex 类型
        assert isinstance(idx, BlockIndex)
        # 断言索引对象中的数据点数量为2
        assert idx.npoints == 2
        # 断言索引对象中的块索引为数组 [2]
        tm.assert_numpy_array_equal(idx.blocs, np.array([2], dtype=np.int32))
        # 断言索引对象中的块长度为数组 [2]
        tm.assert_numpy_array_equal(idx.blengths, np.array([2], dtype=np.int32))

        # 创建一个稀疏索引对象，种类为“block”，不包含任何数据点
        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind="block")
        # 断言确保返回的索引对象是 BlockIndex 类型
        assert isinstance(idx, BlockIndex)
        # 断言索引对象中的数据点数量为0
        assert idx.npoints == 0
        # 断言索引对象中的块索引为空数组
        tm.assert_numpy_array_equal(idx.blocs, np.array([], dtype=np.int32))
        # 断言索引对象中的块长度为空数组
        tm.assert_numpy_array_equal(idx.blengths, np.array([], dtype=np.int32))

        # 创建一个稀疏索引对象，种类为“block”，包含索引值为0到3的全部数据点
        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind="block")
        # 断言确保返回的索引对象是 BlockIndex 类型
        assert isinstance(idx, BlockIndex)
        # 断言索引对象中的数据点数量为4
        assert idx.npoints == 4
        # 断言索引对象中的块索引为数组 [0]
        tm.assert_numpy_array_equal(idx.blocs, np.array([0], dtype=np.int32))
        # 断言索引对象中的块长度为数组 [4]
        tm.assert_numpy_array_equal(idx.blengths, np.array([4], dtype=np.int32))

        # 创建一个稀疏索引对象，种类为“block”，包含索引值为0、2和3的数据点
        idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind="block")
        # 断言确保返回的索引对象是 BlockIndex 类型
        assert isinstance(idx, BlockIndex)
        # 断言索引对象中的数据点数量为3
        assert idx.npoints == 3
        # 断言索引对象中的块索引为数组 [0, 2]
        tm.assert_numpy_array_equal(idx.blocs, np.array([0, 2], dtype=np.int32))
        # 断言索引对象中的块长度为数组 [1, 2]
        tm.assert_numpy_array_equal(idx.blengths, np.array([1, 2], dtype=np.int32))
    # 定义测试方法，用于测试稀疏索引对象的查找功能
    def test_lookup_array(self, kind):
        # 创建一个稀疏索引对象，索引长度为4，包含元素为[2, 3]的数组作为稀疏索引，数据类型为kind指定的类型
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind=kind)

        # 测试稀疏索引对象对输入数组[-1, 0, 2]的查找功能
        res = idx.lookup_array(np.array([-1, 0, 2], dtype=np.int32))
        # 预期结果是数组[-1, -1, 0]，数据类型为int32
        exp = np.array([-1, -1, 0], dtype=np.int32)
        # 使用测试框架中的函数验证结果是否符合预期
        tm.assert_numpy_array_equal(res, exp)

        # 继续测试稀疏索引对象对输入数组[4, 2, 1, 3]的查找功能
        res = idx.lookup_array(np.array([4, 2, 1, 3], dtype=np.int32))
        # 预期结果是数组[-1, 0, -1, 1]，数据类型为int32
        exp = np.array([-1, 0, -1, 1], dtype=np.int32)
        # 使用测试框架中的函数验证结果是否符合预期
        tm.assert_numpy_array_equal(res, exp)

        # 创建一个空的稀疏索引对象，长度为4，不包含任何索引元素
        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind=kind)
        # 测试空稀疏索引对象对输入数组[-1, 0, 2, 4]的查找功能
        res = idx.lookup_array(np.array([-1, 0, 2, 4], dtype=np.int32))
        # 预期结果是数组[-1, -1, -1, -1]，数据类型为int32
        exp = np.array([-1, -1, -1, -1], dtype=np.int32)
        # 使用测试框架中的函数验证结果是否符合预期
        tm.assert_numpy_array_equal(res, exp)

        # 创建一个包含完整索引的稀疏索引对象，索引长度为4，包含元素[0, 1, 2, 3]
        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind=kind)
        # 测试完整稀疏索引对象对输入数组[-1, 0, 2]的查找功能
        res = idx.lookup_array(np.array([-1, 0, 2], dtype=np.int32))
        # 预期结果是数组[-1, 0, 2]，数据类型为int32
        exp = np.array([-1, 0, 2], dtype=np.int32)
        # 使用测试框架中的函数验证结果是否符合预期
        tm.assert_numpy_array_equal(res, exp)

        # 继续测试完整稀疏索引对象对输入数组[4, 2, 1, 3]的查找功能
        res = idx.lookup_array(np.array([4, 2, 1, 3], dtype=np.int32))
        # 预期结果是数组[-1, 2, 1, 3]，数据类型为int32
        exp = np.array([-1, 2, 1, 3], dtype=np.int32)
        # 使用测试框架中的函数验证结果是否符合预期
        tm.assert_numpy_array_equal(res, exp)

        # 创建一个包含部分索引的稀疏索引对象，索引长度为4，包含元素[0, 2, 3]
        idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind=kind)
        # 测试部分稀疏索引对象对输入数组[2, 1, 3, 0]的查找功能
        res = idx.lookup_array(np.array([2, 1, 3, 0], dtype=np.int32))
        # 预期结果是数组[1, -1, 2, 0]，数据类型为int32
        exp = np.array([1, -1, 2, 0], dtype=np.int32)
        # 使用测试框架中的函数验证结果是否符合预期
        tm.assert_numpy_array_equal(res, exp)

        # 继续测试部分稀疏索引对象对输入数组[1, 4, 2, 5]的查找功能
        res = idx.lookup_array(np.array([1, 4, 2, 5], dtype=np.int32))
        # 预期结果是数组[-1, -1, 1, -1]，数据类型为int32
        exp = np.array([-1, -1, 1, -1], dtype=np.int32)
        # 使用测试框架中的函数验证结果是否符合预期
        tm.assert_numpy_array_equal(res, exp)

    @pytest.mark.parametrize(
        "idx, expected",
        [
            [0, -1],
            [5, 0],
            [7, 2],
            [8, -1],
            [9, -1],
            [10, -1],
            [11, -1],
            [12, 3],
            [17, 8],
            [18, -1],
        ],
    )
    # 定义基础查找功能的参数化测试方法，验证对块索引和整数索引对象的查找功能
    def test_lookup_basics(self, idx, expected):
        # 创建一个块索引对象，总长度为20，块边界为[5, 12]，块大小为[3, 6]
        bindex = BlockIndex(20, [5, 12], [3, 6])
        # 断言使用块索引对象对idx进行查找的结果等于预期的expected值
        assert bindex.lookup(idx) == expected

        # 将块索引对象转换为整数索引对象
        iindex = bindex.to_int_index()
        # 断言使用整数索引对象对idx进行查找的结果等于预期的expected值
        assert iindex.lookup(idx) == expected
class TestBlockIndex:
    # 测试块索引对象的功能

    def test_block_internal(self):
        # 测试 make_sparse_index 函数生成块索引对象的内部方法
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.blocs, np.array([2], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([2], dtype=np.int32))

        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.blocs, np.array([], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([], dtype=np.int32))

        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.blocs, np.array([0], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([4], dtype=np.int32))

        idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 3
        tm.assert_numpy_array_equal(idx.blocs, np.array([0, 2], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([1, 2], dtype=np.int32))

    @pytest.mark.parametrize("i", [5, 10, 100, 101])
    def test_make_block_boundary(self, i):
        # 测试 make_sparse_index 函数生成块索引对象的边界情况
        idx = make_sparse_index(i, np.arange(0, i, 2, dtype=np.int32), kind="block")

        exp = np.arange(0, i, 2, dtype=np.int32)
        tm.assert_numpy_array_equal(idx.blocs, exp)
        tm.assert_numpy_array_equal(idx.blengths, np.ones(len(exp), dtype=np.int32))

    def test_equals(self):
        # 测试块索引对象的相等性判断
        index = BlockIndex(10, [0, 4], [2, 5])

        assert index.equals(index)
        assert not index.equals(BlockIndex(10, [0, 4], [2, 6]))

    def test_check_integrity(self):
        # 测试块索引对象的完整性检查
        locs = []
        lengths = []

        # 0-length OK
        BlockIndex(0, locs, lengths)

        # also OK even though empty
        BlockIndex(1, locs, lengths)

        msg = "Block 0 extends beyond end"
        with pytest.raises(ValueError, match=msg):
            BlockIndex(10, [5], [10])

        msg = "Block 0 overlaps"
        with pytest.raises(ValueError, match=msg):
            BlockIndex(10, [2, 5], [5, 3])

    def test_to_int_index(self):
        # 测试块索引对象转换为整数索引对象
        locs = [0, 10]
        lengths = [4, 6]
        exp_inds = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15]

        block = BlockIndex(20, locs, lengths)
        dense = block.to_int_index()

        tm.assert_numpy_array_equal(dense.indices, np.array(exp_inds, dtype=np.int32))

    def test_to_block_index(self):
        # 测试块索引对象的自我转换为块索引对象
        index = BlockIndex(10, [0, 5], [4, 5])
        assert index.to_block_index() is index
    def test_check_integrity(self):
        # 测试索引的完整性检查

        # 测试：超出指定长度的索引数量
        msg = "Too many indices"
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=1, indices=[1, 2, 3])

        # 测试：索引不能为负数
        msg = "No index can be less than zero"
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, -2, 3])

        # 再次测试：索引不能为负数
        msg = "No index can be less than zero"
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, -2, 3])

        # 测试：所有索引必须小于长度
        msg = "All indices must be less than the length"
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 2, 5])

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 2, 6])

        # 测试：索引必须严格递增
        msg = "Indices must be strictly increasing"
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 3, 2])

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 3, 3])

    def test_int_internal(self):
        # 测试内部整数索引的生成和验证

        # 测试用例1
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind="integer")
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.indices, np.array([2, 3], dtype=np.int32))

        # 测试用例2
        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind="integer")
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.indices, np.array([], dtype=np.int32))

        # 测试用例3
        idx = make_sparse_index(
            4, np.array([0, 1, 2, 3], dtype=np.int32), kind="integer"
        )
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.indices, np.array([0, 1, 2, 3], dtype=np.int32))

    def test_equals(self):
        # 测试索引对象的相等性判断

        # 创建索引对象
        index = IntIndex(10, [0, 1, 2, 3, 4])

        # 断言索引对象与自身相等
        assert index.equals(index)

        # 断言索引对象与不同的索引对象不相等
        assert not index.equals(IntIndex(10, [0, 1, 2, 3]))

    def test_to_block_index(self, cases, test_length):
        # 测试将整数索引转换为块索引的过程

        # 从测试用例中获取块索引的位置和长度
        xloc, xlen, yloc, ylen, _, _ = cases

        # 创建块索引对象
        xindex = BlockIndex(test_length, xloc, xlen)
        yindex = BlockIndex(test_length, yloc, ylen)

        # 测试：转换为整数索引再转换回块索引后是否保持不变
        xbindex = xindex.to_int_index().to_block_index()
        ybindex = yindex.to_int_index().to_block_index()
        assert isinstance(xbindex, BlockIndex)
        assert xbindex.equals(xindex)
        assert ybindex.equals(yindex)

    def test_to_int_index(self):
        # 测试将块索引转换为整数索引的过程

        # 创建整数索引对象
        index = IntIndex(10, [2, 3, 4, 5, 6])

        # 断言转换后的整数索引对象与原始对象相同
        assert index.to_int_index() is index
`
# 定义一个测试类 TestSparseOperators，用于测试稀疏操作函数
class TestSparseOperators:
    # 使用 pytest 的参数化标记，针对不同的操作名进行参数化测试
    @pytest.mark.parametrize("opname", ["add", "sub", "mul", "truediv", "floordiv"])
    # 定义测试方法 test_op，接受操作名、测试用例和测试长度作为参数
    def test_op(self, opname, cases, test_length):
        # 解包测试用例中的变量
        xloc, xlen, yloc, ylen, _, _ = cases
        # 从 splib 模块中获取对应的稀疏操作函数
        sparse_op = getattr(splib, f"sparse_{opname}_float64")
        # 从 operator 模块中获取对应的 Python 操作函数
        python_op = getattr(operator, opname)

        # 创建 BlockIndex 对象，表示 x 的块索引
        xindex = BlockIndex(test_length, xloc, xlen)
        # 创建 BlockIndex 对象，表示 y 的块索引
        yindex = BlockIndex(test_length, yloc, ylen)

        # 将 xindex 转换为整数索引
        xdindex = xindex.to_int_index()
        # 将 yindex 转换为整数索引
        ydindex = yindex.to_int_index()

        # 创建 x 数组，使用块索引中的点数和填充值
        x = np.arange(xindex.npoints) * 10.0 + 1
        # 创建 y 数组，使用块索引中的点数和填充值
        y = np.arange(yindex.npoints) * 100.0 + 1

        # 设置 x 的填充值
        xfill = 0
        # 设置 y 的填充值
        yfill = 2

        # 调用稀疏操作函数，计算结果块值、块索引和填充值
        result_block_vals, rb_index, bfill = sparse_op(
            x, xindex, xfill, y, yindex, yfill
        )
        # 调用稀疏操作函数，计算结果整数值、整数索引和填充值
        result_int_vals, ri_index, ifill = sparse_op(
            x, xdindex, xfill, y, ydindex, yfill
        )

        # 断言结果的整数索引与整数索引相等
        assert rb_index.to_int_index().equals(ri_index)
        # 使用测试工具函数，断言结果块值与整数值相等
        tm.assert_numpy_array_equal(result_block_vals, result_int_vals)
        # 断言块填充值与整数填充值相等
        assert bfill == ifill

        # 对比与 Series 的结果...
        # 创建 Series 对象 xseries，使用 xdindex 的索引，并对缺失值填充
        xseries = Series(x, xdindex.indices)
        xseries = xseries.reindex(np.arange(test_length)).fillna(xfill)

        # 创建 Series 对象 yseries，使用 ydindex 的索引，并对缺失值填充
        yseries = Series(y, ydindex.indices)
        yseries = yseries.reindex(np.arange(test_length)).fillna(yfill)

        # 使用 Python 操作函数，计算 Series 的结果
        series_result = python_op(xseries, yseries)
        # 根据整数索引重新索引 Series 的结果
        series_result = series_result.reindex(ri_index.indices)

        # 使用测试工具函数，断言结果块值与 Series 的值相等
        tm.assert_numpy_array_equal(result_block_vals, series_result.values)
        # 使用测试工具函数，断言结果整数值与 Series 的值相等
        tm.assert_numpy_array_equal(result_int_vals, series_result.values)
```
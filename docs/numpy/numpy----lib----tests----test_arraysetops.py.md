# `.\numpy\numpy\lib\tests\test_arraysetops.py`

```py
"""Test functions for 1D array set operations.

"""
import numpy as np

from numpy import (
    ediff1d, intersect1d, setxor1d, union1d, setdiff1d, unique, isin
    )
from numpy.exceptions import AxisError
from numpy.testing import (assert_array_equal, assert_equal,
                           assert_raises, assert_raises_regex)
import pytest


class TestSetOps:

    def test_intersect1d(self):
        # unique inputs
        a = np.array([5, 7, 1, 2])
        b = np.array([2, 4, 3, 1, 5])

        ec = np.array([1, 2, 5])
        # Compute intersection of arrays 'a' and 'b', assuming unique values
        c = intersect1d(a, b, assume_unique=True)
        assert_array_equal(c, ec)

        # non-unique inputs
        a = np.array([5, 5, 7, 1, 2])
        b = np.array([2, 1, 4, 3, 3, 1, 5])

        ed = np.array([1, 2, 5])
        # Compute intersection of arrays 'a' and 'b', allowing duplicates
        c = intersect1d(a, b)
        assert_array_equal(c, ed)
        # Check intersection with empty arrays
        assert_array_equal([], intersect1d([], []))

    def test_intersect1d_array_like(self):
        # See gh-11772
        class Test:
            def __array__(self, dtype=None, copy=None):
                return np.arange(3)

        a = Test()
        # Test intersection with array-like objects
        res = intersect1d(a, a)
        assert_array_equal(res, a)
        res = intersect1d([1, 2, 3], [1, 2, 3])
        assert_array_equal(res, [1, 2, 3])

    def test_intersect1d_indices(self):
        # unique inputs
        a = np.array([1, 2, 3, 4])
        b = np.array([2, 1, 4, 6])
        # Compute intersection and return indices along with result
        c, i1, i2 = intersect1d(a, b, assume_unique=True, return_indices=True)
        ee = np.array([1, 2, 4])
        assert_array_equal(c, ee)
        assert_array_equal(a[i1], ee)
        assert_array_equal(b[i2], ee)

        # non-unique inputs
        a = np.array([1, 2, 2, 3, 4, 3, 2])
        b = np.array([1, 8, 4, 2, 2, 3, 2, 3])
        # Compute intersection with non-unique inputs and return indices
        c, i1, i2 = intersect1d(a, b, return_indices=True)
        ef = np.array([1, 2, 3, 4])
        assert_array_equal(c, ef)
        assert_array_equal(a[i1], ef)
        assert_array_equal(b[i2], ef)

        # non1d, unique inputs
        a = np.array([[2, 4, 5, 6], [7, 8, 1, 15]])
        b = np.array([[3, 2, 7, 6], [10, 12, 8, 9]])
        # Compute intersection for multi-dimensional arrays, assuming unique values
        c, i1, i2 = intersect1d(a, b, assume_unique=True, return_indices=True)
        ui1 = np.unravel_index(i1, a.shape)
        ui2 = np.unravel_index(i2, b.shape)
        ea = np.array([2, 6, 7, 8])
        assert_array_equal(ea, a[ui1])
        assert_array_equal(ea, b[ui2])

        # non1d, not assumed to be unique inputs
        a = np.array([[2, 4, 5, 6, 6], [4, 7, 8, 7, 2]])
        b = np.array([[3, 2, 7, 7], [10, 12, 8, 7]])
        # Compute intersection for multi-dimensional arrays, allowing duplicates
        c, i1, i2 = intersect1d(a, b, return_indices=True)
        ui1 = np.unravel_index(i1, a.shape)
        ui2 = np.unravel_index(i2, b.shape)
        ea = np.array([2, 7, 8])
        assert_array_equal(ea, a[ui1])
        assert_array_equal(ea, b[ui2])
    # 定义一个测试函数，用于测试 setxor1d 函数的功能
    def test_setxor1d(self):
        # 创建两个 NumPy 数组 a 和 b
        a = np.array([5, 7, 1, 2])
        b = np.array([2, 4, 3, 1, 5])

        # 预期结果的 NumPy 数组
        ec = np.array([3, 4, 7])
        # 调用 setxor1d 函数，计算 a 和 b 的对称差集
        c = setxor1d(a, b)
        # 断言计算结果与预期结果相等
        assert_array_equal(c, ec)

        # 更新数组 a 和 b
        a = np.array([1, 2, 3])
        b = np.array([6, 5, 4])

        # 更新预期结果的 NumPy 数组
        ec = np.array([1, 2, 3, 4, 5, 6])
        # 再次调用 setxor1d 函数，计算新的 a 和 b 的对称差集
        c = setxor1d(a, b)
        # 断言计算结果与更新后的预期结果相等
        assert_array_equal(c, ec)

        # 再次更新数组 a 和 b
        a = np.array([1, 8, 2, 3])
        b = np.array([6, 5, 4, 8])

        # 更新预期结果的 NumPy 数组
        ec = np.array([1, 2, 3, 4, 5, 6])
        # 第三次调用 setxor1d 函数，计算新的 a 和 b 的对称差集
        c = setxor1d(a, b)
        # 断言计算结果与更新后的预期结果相等
        assert_array_equal(c, ec)

        # 对空数组的情况进行额外的断言，计算空数组的对称差集应为空数组
        assert_array_equal([], setxor1d([], []))

    # 定义另一个测试函数，用于测试 setxor1d 函数在 assume_unique=True 时的功能
    def test_setxor1d_unique(self):
        # 创建两个 NumPy 数组 a 和 b
        a = np.array([1, 8, 2, 3])
        b = np.array([6, 5, 4, 8])

        # 预期结果的 NumPy 数组
        ec = np.array([1, 2, 3, 4, 5, 6])
        # 调用 setxor1d 函数，计算 a 和 b 的假定唯一值的对称差集
        c = setxor1d(a, b, assume_unique=True)
        # 断言计算结果与预期结果相等
        assert_array_equal(c, ec)

        # 更新数组 a 和 b，这次包含嵌套数组
        a = np.array([[1], [8], [2], [3]])
        b = np.array([[6, 5], [4, 8]])

        # 更新预期结果的 NumPy 数组
        ec = np.array([1, 2, 3, 4, 5, 6])
        # 再次调用 setxor1d 函数，计算新的 a 和 b 的假定唯一值的对称差集
        c = setxor1d(a, b, assume_unique=True)
        # 断言计算结果与更新后的预期结果相等
        assert_array_equal(c, ec)

    # 定义测试函数，用于测试 ediff1d 函数的功能
    def test_ediff1d(self):
        # 创建不同长度的 NumPy 数组用于测试 ediff1d 函数
        zero_elem = np.array([])
        one_elem = np.array([1])
        two_elem = np.array([1, 2])

        # 断言计算空数组的差分结果应为空数组
        assert_array_equal([], ediff1d(zero_elem))
        # 断言计算带有 to_begin 参数的空数组的差分结果
        assert_array_equal([0], ediff1d(zero_elem, to_begin=0))
        # 断言计算带有 to_end 参数的空数组的差分结果
        assert_array_equal([0], ediff1d(zero_elem, to_end=0))
        # 断言计算带有 to_begin 和 to_end 参数的空数组的差分结果
        assert_array_equal([-1, 0], ediff1d(zero_elem, to_begin=-1, to_end=0))
        # 断言计算单元素数组的差分结果应为空数组
        assert_array_equal([], ediff1d(one_elem))
        # 断言计算两元素数组的差分结果
        assert_array_equal([1], ediff1d(two_elem))
        # 断言计算带有 to_begin 和 to_end 参数的两元素数组的差分结果
        assert_array_equal([7, 1, 9], ediff1d(two_elem, to_begin=7, to_end=9))
        # 断言计算带有不同类型 to_begin 和 to_end 参数的两元素数组的差分结果
        assert_array_equal([5, 6, 1, 7, 8],
                           ediff1d(two_elem, to_begin=[5, 6], to_end=[7, 8]))
        # 断言计算带有 to_end 参数的两元素数组的差分结果
        assert_array_equal([1, 9], ediff1d(two_elem, to_end=9))
        # 断言计算带有不同类型 to_end 参数的两元素数组的差分结果
        assert_array_equal([1, 7, 8], ediff1d(two_elem, to_end=[7, 8]))
        # 断言计算带有 to_begin 参数的两元素数组的差分结果
        assert_array_equal([7, 1], ediff1d(two_elem, to_begin=7))
        # 断言计算带有不同类型 to_begin 参数的两元素数组的差分结果
        assert_array_equal([5, 6, 1], ediff1d(two_elem, to_begin=[5, 6]))

    # 使用 pytest 的 parametrize 装饰器定义参数化测试
    @pytest.mark.parametrize("ary, prepend, append, expected", [
        # 测试输入包含 np.nan，期望报错指示 np.nan 不能转换为整数数组
        (np.array([1, 2, 3], dtype=np.int64),
         None,
         np.nan,
         'to_end'),
        # 测试输入包含不同类型的数组，期望报错指示无法将浮点数数组 downcast 到整数类型
        (np.array([1, 2, 3], dtype=np.int64),
         np.array([5, 7, 2], dtype=np.float32),
         None,
         'to_begin'),
        # 测试输入包含 np.nan，期望报错指示 np.nan 不能转换为整数数组，这里包含两个 np.nan
        (np.array([1., 3., 9.], dtype=np.int8),
         np.nan,
         np.nan,
         'to_begin'),
         ])
    def test_ediff1d_forbidden_type_casts(self, ary, prepend, append, expected):
        # 验证对 gh-11490 的解决方案

        # 具体来说，在尝试使用不兼容类型进行追加或前置时，抛出适当的异常
        msg = 'dtype of `{}` must be compatible'.format(expected)
        # 使用 assert_raises_regex 断言，验证是否抛出预期的 TypeError 异常，并检查异常消息
        with assert_raises_regex(TypeError, msg):
            # 调用 ediff1d 函数，传入数组 ary，prepend 和 append 作为参数
            ediff1d(ary=ary,
                    to_end=append,
                    to_begin=prepend)

    @pytest.mark.parametrize(
        "ary,prepend,append,expected",
        [
         (np.array([1, 2, 3], dtype=np.int16),
          2**16,  # 根据相同类型规则将被转换为 int16
          2**16 + 4,
          np.array([0, 1, 1, 4], dtype=np.int16)),
         (np.array([1, 2, 3], dtype=np.float32),
          np.array([5], dtype=np.float64),
          None,
          np.array([5, 1, 1], dtype=np.float32)),
         (np.array([1, 2, 3], dtype=np.int32),
          0,
          0,
          np.array([0, 1, 1, 0], dtype=np.int32)),
         (np.array([1, 2, 3], dtype=np.int64),
          3,
          -9,
          np.array([3, 1, 1, -9], dtype=np.int64)),
        ]
    )
    def test_ediff1d_scalar_handling(self,
                                     ary,
                                     prepend,
                                     append,
                                     expected):
        # 在修复 gh-11490 后，保持对标量 prepend / append 行为的向后兼容性
        # 调用 np.ediff1d 函数，传入参数 ary、prepend 和 append，获取实际结果
        actual = np.ediff1d(ary=ary,
                            to_end=append,
                            to_begin=prepend)
        # 使用 assert_equal 断言，验证 actual 是否等于预期结果 expected
        assert_equal(actual, expected)
        # 进一步验证实际结果的数据类型是否与预期结果的数据类型相同
        assert actual.dtype == expected.dtype

    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    # 定义一个测试方法，用于测试isin函数在不同情况下的行为，参数中包含一种特定的kind类型
    def test_isin(self, kind):
        
        # 定义一个内部函数_isin_slow，用于检查元素a是否在数组b中，返回布尔值
        def _isin_slow(a, b):
            # 将数组b转换为NumPy数组，并展平后转换为列表
            b = np.asarray(b).flatten().tolist()
            return a in b
        
        # 使用np.vectorize创建向量化版本的_isin_slow函数，指定输出类型为布尔型，排除第二个参数的向量化
        isin_slow = np.vectorize(_isin_slow, otypes=[bool], excluded={1})

        # 定义一个辅助函数assert_isin_equal，用于比较isin和isin_slow函数的结果是否相等
        def assert_isin_equal(a, b):
            # 使用isin函数计算a在b中的成员关系
            x = isin(a, b, kind=kind)
            # 使用isin_slow函数计算a在b中的成员关系
            y = isin_slow(a, b)
            # 断言两个结果数组是否完全相等
            assert_array_equal(x, y)

        # 多维数组作为参数
        a = np.arange(24).reshape([2, 3, 4])
        b = np.array([[10, 20, 30], [0, 1, 3], [11, 22, 33]])
        # 断言isin和isin_slow在多维数组a和b上的行为是否一致
        assert_isin_equal(a, b)

        # 数组样式的参数
        c = [(9, 8), (7, 6)]
        d = (9, 7)
        # 断言isin和isin_slow在数组样式的c和d上的行为是否一致
        assert_isin_equal(c, d)

        # 零维数组:
        f = np.array(3)
        # 断言isin和isin_slow在零维数组f和b上的行为是否一致
        assert_isin_equal(f, b)
        assert_isin_equal(a, f)
        assert_isin_equal(f, f)

        # 标量:
        # 断言isin和isin_slow在标量值5和数组b上的行为是否一致
        assert_isin_equal(5, b)
        # 断言isin和isin_slow在多维数组a和标量值6上的行为是否一致
        assert_isin_equal(a, 6)
        # 断言isin和isin_slow在标量值5和6上的行为是否一致
        assert_isin_equal(5, 6)

        # 空数组样式:
        if kind != "table":
            # 如果kind不是"table"类型，则执行以下代码块
            # 空列表将被转换为float64类型，对于kind="table"是无效的
            x = []
            # 断言isin和isin_slow在空列表x和数组b上的行为是否一致
            assert_isin_equal(x, b)
            # 断言isin和isin_slow在多维数组a和空列表x上的行为是否一致
            assert_isin_equal(a, x)
            # 断言isin和isin_slow在空列表x上的行为是否一致
            assert_isin_equal(x, x)

        # 空数组样式，包含不同类型的空数组:
        for dtype in [bool, np.int64, np.float64]:
            if kind == "table" and dtype == np.float64:
                continue

            if dtype in {np.int64, np.float64}:
                ar = np.array([10, 20, 30], dtype=dtype)
            elif dtype in {bool}:
                ar = np.array([True, False, False])

            # 创建特定类型的空数组
            empty_array = np.array([], dtype=dtype)

            # 断言isin和isin_slow在空数组empty_array和数组ar上的行为是否一致
            assert_isin_equal(empty_array, ar)
            # 断言isin和isin_slow在数组ar和空数组empty_array上的行为是否一致
            assert_isin_equal(ar, empty_array)
            # 断言isin和isin_slow在空数组empty_array上的行为是否一致
            assert_isin_equal(empty_array, empty_array)

    # 使用pytest的参数化标记对test_isin方法进行参数化测试，参数kind包括None, "sort", "table"
    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    def test_isin(self, kind):
        # we use two different sizes for the b array here to test the
        # two different paths in isin().
        # 在这里使用两种不同大小的 b 数组来测试 isin() 中的两条不同路径

        for mult in (1, 10):
            # One check without np.array to make sure lists are handled correct
            # 进行一次不使用 np.array 的检查，确保列表被正确处理
            a = [5, 7, 1, 2]
            # b 数组的大小随 mult 的不同而变化
            b = [2, 4, 3, 1, 5] * mult
            ec = np.array([True, False, True, True])
            # 调用 isin 函数，使用给定的参数进行计算，并将结果与期望值进行比较
            c = isin(a, b, assume_unique=True, kind=kind)
            assert_array_equal(c, ec)

            # 修改 a 的第一个元素，更新期望结果，并再次调用 isin 函数进行比较
            a[0] = 8
            ec = np.array([False, False, True, True])
            c = isin(a, b, assume_unique=True, kind=kind)
            assert_array_equal(c, ec)

            # 再次修改 a 的元素，并再次调用 isin 函数进行比较
            a[0], a[3] = 4, 8
            ec = np.array([True, False, True, False])
            c = isin(a, b, assume_unique=True, kind=kind)
            assert_array_equal(c, ec)

            # 使用 np.array 创建 a 和 b 数组，进行 isin 函数调用和比较
            a = np.array([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5])
            b = [2, 3, 4] * mult
            ec = [False, True, False, True, True, True, True, True, True,
                  False, True, False, False, False]
            c = isin(a, b, kind=kind)
            assert_array_equal(c, ec)

            # 修改 b 数组并再次调用 isin 函数进行比较
            b = b + [5, 5, 4] * mult
            ec = [True, True, True, True, True, True, True, True, True, True,
                  True, False, True, True]
            c = isin(a, b, kind=kind)
            assert_array_equal(c, ec)

            # 使用 np.array 创建 a 和 b 数组，进行 isin 函数调用和比较
            a = np.array([5, 7, 1, 2])
            b = np.array([2, 4, 3, 1, 5] * mult)
            ec = np.array([True, False, True, True])
            c = isin(a, b, kind=kind)
            assert_array_equal(c, ec)

            # 使用 np.array 创建 a 和 b 数组，进行 isin 函数调用和比较
            a = np.array([5, 7, 1, 1, 2])
            b = np.array([2, 4, 3, 3, 1, 5] * mult)
            ec = np.array([True, False, True, True, True])
            c = isin(a, b, kind=kind)
            assert_array_equal(c, ec)

            # 使用 np.array 创建 a 和 b 数组，进行 isin 函数调用和比较
            a = np.array([5, 5])
            b = np.array([2, 2] * mult)
            ec = np.array([False, False])
            c = isin(a, b, kind=kind)
            assert_array_equal(c, ec)

        # 对于单个元素的情况，进行 isin 函数调用和比较
        a = np.array([5])
        b = np.array([2])
        ec = np.array([False])
        c = isin(a, b, kind=kind)
        assert_array_equal(c, ec)

        # 如果 kind 参数为 None 或 "sort"，则进行额外的测试
        if kind in {None, "sort"}:
            assert_array_equal(isin([], [], kind=kind), [])

    def test_isin_char_array(self):
        # 使用字符数组进行 isin 函数的测试
        a = np.array(['a', 'b', 'c', 'd', 'e', 'c', 'e', 'b'])
        b = np.array(['a', 'c'])

        ec = np.array([True, False, True, False, False, True, False, False])
        c = isin(a, b)

        assert_array_equal(c, ec)

    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    def test_isin_invert(self, kind):
        "Test isin's invert parameter"
        # 使用两种不同大小的 b 数组来测试 isin() 中的两条不同路径
        for mult in (1, 10):
            # 创建整数数组 a
            a = np.array([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5])
            # 根据 mult 值创建 b 数组
            b = [2, 3, 4] * mult
            # 断言 invert 参数为 True 时的结果与 invert 参数为 False 时的结果相反
            assert_array_equal(np.invert(isin(a, b, kind=kind)),
                               isin(a, b, invert=True, kind=kind))

        # 浮点数测试:
        if kind in {None, "sort"}:
            for mult in (1, 10):
                # 创建浮点数数组 a
                a = np.array([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5],
                            dtype=np.float32)
                # 根据 mult 值创建浮点数数组 b
                b = [2, 3, 4] * mult
                b = np.array(b, dtype=np.float32)
                # 断言 invert 参数为 True 时的结果与 invert 参数为 False 时的结果相反
                assert_array_equal(np.invert(isin(a, b, kind=kind)),
                                   isin(a, b, invert=True, kind=kind))

    def test_isin_hit_alternate_algorithm(self):
        """Hit the standard isin code with integers"""
        # 需要极端范围以触发标准 isin 代码路径
        # 这样可以避免使用 kind='table'
        a = np.array([5, 4, 5, 3, 4, 4, 1e9], dtype=np.int64)
        b = np.array([2, 3, 4, 1e9], dtype=np.int64)
        expected = np.array([0, 1, 0, 1, 1, 1, 1], dtype=bool)
        # 断言使用 isin 函数的返回结果与预期结果相等
        assert_array_equal(expected, isin(a, b))
        # 断言使用 invert=True 时的结果与 invert=False 时的结果相反
        assert_array_equal(np.invert(expected), isin(a, b, invert=True))

        a = np.array([5, 7, 1, 2], dtype=np.int64)
        b = np.array([2, 4, 3, 1, 5, 1e9], dtype=np.int64)
        ec = np.array([True, False, True, True])
        # 使用 assume_unique=True 来测试 isin 函数
        c = isin(a, b, assume_unique=True)
        # 断言返回结果 c 与预期结果 ec 相等
        assert_array_equal(c, ec)

    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    def test_isin_boolean(self, kind):
        """Test that isin works for boolean input"""
        # 创建布尔类型的数组 a 和 b
        a = np.array([True, False])
        b = np.array([False, False, False])
        expected = np.array([False, True])
        # 断言 isin 函数对布尔输入的结果与预期结果相等
        assert_array_equal(expected,
                           isin(a, b, kind=kind))
        # 断言 invert=True 时的结果与 invert=False 时的结果相反
        assert_array_equal(np.invert(expected),
                           isin(a, b, invert=True, kind=kind))

    @pytest.mark.parametrize("kind", [None, "sort"])
    def test_isin_timedelta(self, kind):
        """Test that isin works for timedelta input"""
        # 创建随机数种子
        rstate = np.random.RandomState(0)
        # 创建整数数组 a 和 b
        a = rstate.randint(0, 100, size=10)
        b = rstate.randint(0, 100, size=10)
        truth = isin(a, b)
        # 将 a 和 b 转换为 timedelta 类型
        a_timedelta = a.astype("timedelta64[s]")
        b_timedelta = b.astype("timedelta64[s]")
        # 断言 isin 函数对 timedelta 输入的结果与预期结果相等
        assert_array_equal(truth, isin(a_timedelta, b_timedelta, kind=kind))

    def test_isin_table_timedelta_fails(self):
        a = np.array([0, 1, 2], dtype="timedelta64[s]")
        b = a
        # 确保会引发 ValueError
        with pytest.raises(ValueError):
            # 测试 isin 函数对 timedelta 类型使用 kind="table" 时的行为
            isin(a, b, kind="table")
    @pytest.mark.parametrize(
        "dtype1,dtype2",
        [  # 参数化测试用例，分别设置不同的数据类型对
            (np.int8, np.int16),
            (np.int16, np.int8),
            (np.uint8, np.uint16),
            (np.uint16, np.uint8),
            (np.uint8, np.int16),
            (np.int16, np.uint8),
            (np.uint64, np.int64),
        ]
    )
    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    # 参数化测试用例，测试不同的排序方法（None, "sort", "table"）
    def test_isin_mixed_dtype(self, dtype1, dtype2, kind):
        """Test that isin works as expected for mixed dtype input."""
        # 混合数据类型输入情况下，测试 isin 函数的预期行为
        is_dtype2_signed = np.issubdtype(dtype2, np.signedinteger)
        # 检查 dtype2 是否为有符号整数类型
        ar1 = np.array([0, 0, 1, 1], dtype=dtype1)
        # 创建数组 ar1，使用 dtype1 指定的数据类型

        if is_dtype2_signed:
            ar2 = np.array([-128, 0, 127], dtype=dtype2)
        else:
            ar2 = np.array([127, 0, 255], dtype=dtype2)
        # 根据 dtype2 的类型，创建相应的 ar2 数组

        expected = np.array([True, True, False, False])
        # 预期结果数组

        expect_failure = kind == "table" and (
            dtype1 == np.int16 and dtype2 == np.int8)
        # 检查是否预期会失败的情况（kind 为 "table"，并且 dtype1 和 dtype2 符合特定条件）

        if expect_failure:
            with pytest.raises(RuntimeError, match="exceed the maximum"):
                isin(ar1, ar2, kind=kind)
                # 预期出现 RuntimeError 异常，并匹配 "exceed the maximum" 字符串
        else:
            assert_array_equal(isin(ar1, ar2, kind=kind), expected)
            # 否则，断言 isin 函数的结果与预期结果 expected 相等

    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    # 参数化测试用例，测试不同的排序方法（None, "sort", "table"）
    def test_isin_mixed_boolean(self, kind):
        """Test that isin works as expected for bool/int input."""
        # 布尔/整数混合输入情况下，测试 isin 函数的预期行为
        for dtype in np.typecodes["AllInteger"]:
            # 遍历所有整数类型的 dtype
            a = np.array([True, False, False], dtype=bool)
            # 创建布尔类型数组 a
            b = np.array([0, 0, 0, 0], dtype=dtype)
            # 创建整数类型数组 b
            expected = np.array([False, True, True], dtype=bool)
            # 预期结果数组

            assert_array_equal(isin(a, b, kind=kind), expected)
            # 断言 isin 函数的结果与预期结果 expected 相等

            a, b = b, a
            # 交换数组 a 和 b

            expected = np.array([True, True, True, True], dtype=bool)
            # 更新预期结果数组

            assert_array_equal(isin(a, b, kind=kind), expected)
            # 断言 isin 函数的结果与预期结果 expected 相等

    def test_isin_first_array_is_object(self):
        # 测试第一个数组是对象类型的情况
        ar1 = [None]
        # 创建对象数组 ar1
        ar2 = np.array([1]*10)
        # 创建整数数组 ar2
        expected = np.array([False])
        # 预期结果数组
        result = np.isin(ar1, ar2)
        # 调用 isin 函数
        assert_array_equal(result, expected)
        # 断言 isin 函数的结果与预期结果 expected 相等

    def test_isin_second_array_is_object(self):
        # 测试第二个数组是对象类型的情况
        ar1 = 1
        # 创建整数 ar1
        ar2 = np.array([None]*10)
        # 创建对象数组 ar2
        expected = np.array([False])
        # 预期结果数组
        result = np.isin(ar1, ar2)
        # 调用 isin 函数
        assert_array_equal(result, expected)
        # 断言 isin 函数的结果与预期结果 expected 相等

    def test_isin_both_arrays_are_object(self):
        # 测试两个数组都是对象类型的情况
        ar1 = [None]
        # 创建对象数组 ar1
        ar2 = np.array([None]*10)
        # 创建对象数组 ar2
        expected = np.array([True])
        # 预期结果数组
        result = np.isin(ar1, ar2)
        # 调用 isin 函数
        assert_array_equal(result, expected)
        # 断言 isin 函数的结果与预期结果 expected 相等

    def test_isin_both_arrays_have_structured_dtype(self):
        # 测试两个数组都有结构化数据类型的情况，包含一个整数字段和一个允许任意 Python 对象的对象字段
        dt = np.dtype([('field1', int), ('field2', object)])
        # 定义结构化数据类型 dt，包含一个整数字段 'field1' 和一个对象字段 'field2'
        ar1 = np.array([(1, None)], dtype=dt)
        # 创建结构化数组 ar1
        ar2 = np.array([(1, None)]*10, dtype=dt)
        # 创建结构化数组 ar2
        expected = np.array([True])
        # 预期结果数组
        result = np.isin(ar1, ar2)
        # 调用 isin 函数
        assert_array_equal(result, expected)
        # 断言 isin 函数的结果与预期结果 expected 相等
    def test_isin_with_arrays_containing_tuples(self):
        # 创建包含元组的 NumPy 数组 ar1 和 ar2，dtype 为 object
        ar1 = np.array([(1,), 2], dtype=object)
        ar2 = np.array([(1,), 2], dtype=object)
        # 预期结果是一个包含 True 和 True 的 NumPy 数组
        expected = np.array([True, True])
        # 使用 np.isin 检查 ar1 中的元素是否在 ar2 中，结果存储在 result 中
        result = np.isin(ar1, ar2)
        # 断言检查 result 是否与 expected 相等
        assert_array_equal(result, expected)
        # 再次使用 np.isin，但这次使用 invert=True 参数
        result = np.isin(ar1, ar2, invert=True)
        # 断言检查 result 是否与 np.invert(expected) 相等
        assert_array_equal(result, np.invert(expected))

        # 添加一个整数到数组末尾，以确保数组构建器创建包含元组的数组，
        # 然后移除末尾的整数。数组构建器存在一个处理元组不正确的 bug，
        # 添加整数可以修复这个问题。
        ar1 = np.array([(1,), (2, 1), 1], dtype=object)
        ar1 = ar1[:-1]  # 移除末尾的整数
        ar2 = np.array([(1,), (2, 1), 1], dtype=object)
        ar2 = ar2[:-1]  # 移除末尾的整数
        expected = np.array([True, True])
        result = np.isin(ar1, ar2)
        assert_array_equal(result, expected)
        result = np.isin(ar1, ar2, invert=True)
        assert_array_equal(result, np.invert(expected))

        # 创建包含元组的 NumPy 数组 ar1 和 ar2，dtype 为 object
        ar1 = np.array([(1,), (2, 3), 1], dtype=object)
        ar1 = ar1[:-1]  # 移除末尾的整数
        ar2 = np.array([(1,), 2], dtype=object)
        # 预期结果是一个包含 True 和 False 的 NumPy 数组
        expected = np.array([True, False])
        result = np.isin(ar1, ar2)
        assert_array_equal(result, expected)
        result = np.isin(ar1, ar2, invert=True)
        assert_array_equal(result, np.invert(expected))

    def test_isin_errors(self):
        """Test that isin raises expected errors."""

        # 错误 1：`kind` 参数不是 'sort'、'table' 或 None 中的一个。
        ar1 = np.array([1, 2, 3, 4, 5])
        ar2 = np.array([2, 4, 6, 8, 10])
        assert_raises(ValueError, isin, ar1, ar2, kind='quicksort')

        # 错误 2：对非整数数组使用 `kind="table"` 会引发错误。
        obj_ar1 = np.array([1, 'a', 3, 'b', 5], dtype=object)
        obj_ar2 = np.array([1, 'a', 3, 'b', 5], dtype=object)
        assert_raises(ValueError, isin, obj_ar1, obj_ar2, kind='table')

        for dtype in [np.int32, np.int64]:
            ar1 = np.array([-1, 2, 3, 4, 5], dtype=dtype)
            # 此数组的范围将会溢出：
            overflow_ar2 = np.array([-1, np.iinfo(dtype).max], dtype=dtype)

            # 错误 3：当计算 ar2 的范围时，如果存在整数溢出，使用 `kind="table"` 将触发运行时错误。
            assert_raises(
                RuntimeError,
                isin, ar1, overflow_ar2, kind='table'
            )

            # 非错误：使用 `kind=None` 当存在整数溢出时不会触发运行时错误，会切换到 `sort` 算法。
            result = np.isin(ar1, overflow_ar2, kind=None)
            assert_array_equal(result, [True] + [False] * 4)
            result = np.isin(ar1, overflow_ar2, kind='sort')
            assert_array_equal(result, [True] + [False] * 4)
    # 测试 union1d 函数
    def test_union1d(self):
        # 创建数组 a
        a = np.array([5, 4, 7, 1, 2])
        # 创建数组 b
        b = np.array([2, 4, 3, 3, 2, 1, 5])

        # 期望结果数组 ec
        ec = np.array([1, 2, 3, 4, 5, 7])
        # 使用 union1d 函数对数组 a 和 b 进行操作，得到结果数组 c，并进行数组相等性断言
        c = union1d(a, b)
        assert_array_equal(c, ec)

        # 测试特殊情况，参数非一维时，应该展平
        x = np.array([[0, 1, 2], [3, 4, 5]])
        y = np.array([0, 1, 2, 3, 4])
        # 期望结果数组 ez
        ez = np.array([0, 1, 2, 3, 4, 5])
        # 使用 union1d 函数对数组 x 和 y 进行操作，得到结果数组 z，并进行数组相等性断言
        z = union1d(x, y)
        assert_array_equal(z, ez)

        # 对空数组进行操作，进行数组相等性断言
        assert_array_equal([], union1d([], []))

    # 测试 setdiff1d 函数
    def test_setdiff1d(self):
        # 创建数组 a
        a = np.array([6, 5, 4, 7, 1, 2, 7, 4])
        # 创建数组 b
        b = np.array([2, 4, 3, 3, 2, 1, 5])

        # 期望结果数组 ec
        ec = np.array([6, 7])
        # 使用 setdiff1d 函数对数组 a 和 b 进行操作，得到结果数组 c，并进行数组相等性断言
        c = setdiff1d(a, b)
        assert_array_equal(c, ec)

        # 创建一定范围内的数组 a 和 b，并进行操作，得到结果数组 c，并进行数组相等性断言
        a = np.arange(21)
        b = np.arange(19)
        # 期望结果数组 ec
        ec = np.array([19, 20])
        c = setdiff1d(a, b)
        assert_array_equal(c, ec)

        # 对空数组进行操作，进行数组相等性断言
        assert_array_equal([], setdiff1d([], []))
        # 创建数组 a，指定数据类型为 np.uint32，并进行操作，比较结果数组的数据类型
        a = np.array((), np.uint32)
        assert_equal(setdiff1d(a, []).dtype, np.uint32)

    # 测试设置 assume_unique 参数的 setdiff1d 函数
    def test_setdiff1d_unique(self):
        # 创建数组 a
        a = np.array([3, 2, 1])
        # 创建数组 b
        b = np.array([7, 5, 2])
        # 期望结果数组 expected
        expected = np.array([3, 1])
        # 使用 setdiff1d 函数对数组 a 和 b 进行操作，设置 assume_unique 参数为 True，得到结果数组 actual，并进行数组相等性断言
        actual = setdiff1d(a, b, assume_unique=True)
        assert_equal(actual, expected)

    # 测试对字符数组的 setdiff1d 函数
    def test_setdiff1d_char_array(self):
        # 创建字符数组 a
        a = np.array(['a', 'b', 'c'])
        # 创建字符数组 b
        b = np.array(['a', 'b', 's'])
        # 对字符数组 a 和 b 进行操作，进行数组相等性断言
        assert_array_equal(setdiff1d(a, b), np.array(['c']))

    # 测试 setxor1d、intersect1d 和 union1d 函数的联合操作
    def test_manyways(self):
        # 创建数组 a
        a = np.array([5, 7, 1, 2, 8])
        # 创建数组 b
        b = np.array([9, 8, 2, 4, 3, 1, 5])

        # 使用 setxor1d、intersect1d 和 union1d 函数联合操作得到结果数组 c1，aux1 和 aux2
        c1 = setxor1d(a, b)
        aux1 = intersect1d(a, b)
        aux2 = union1d(a, b)
        # 使用 setdiff1d 函数对数组 aux2 和 aux1 进行操作，得到结果数组 c2，并进行数组相等性断言
        c2 = setdiff1d(aux2, aux1)
        assert_array_equal(c1, c2)
# 定义一个名为 TestUnique 的测试类
class TestUnique:

    # 测试在指定轴上的唯一性错误处理
    def test_unique_axis_errors(self):
        # 断言抛出 TypeError 异常，测试 _run_axis_tests 方法处理对象类型错误
        assert_raises(TypeError, self._run_axis_tests, object)
        # 断言抛出 TypeError 异常，测试 _run_axis_tests 方法处理包含元组的错误类型
        assert_raises(TypeError, self._run_axis_tests, [('a', int), ('b', object)])

        # 断言抛出 AxisError 异常，测试在轴数为2时的唯一性函数调用
        assert_raises(AxisError, unique, np.arange(10), axis=2)
        # 断言抛出 AxisError 异常，测试在负轴数为-2时的唯一性函数调用
        assert_raises(AxisError, unique, np.arange(10), axis=-2)

    # 测试在指定轴上的唯一性处理列表
    def test_unique_axis_list(self):
        # 错误消息字符串
        msg = "Unique failed on list of lists"
        # 输入列表
        inp = [[0, 1, 0], [0, 1, 0]]
        # 转换为 NumPy 数组
        inp_arr = np.asarray(inp)
        # 断言按轴0处理的唯一性，比较列表输入和数组输入的结果
        assert_array_equal(unique(inp, axis=0), unique(inp_arr, axis=0), msg)
        # 断言按轴1处理的唯一性，比较列表输入和数组输入的结果
        assert_array_equal(unique(inp, axis=1), unique(inp_arr, axis=1), msg)

    # 测试在指定轴上的唯一性处理
    def test_unique_axis(self):
        # 初始化类型列表
        types = []
        # 扩展整数类型代码
        types.extend(np.typecodes['AllInteger'])
        # 扩展浮点数类型代码
        types.extend(np.typecodes['AllFloat'])
        # 添加日期类型
        types.append('datetime64[D]')
        # 添加时间间隔类型
        types.append('timedelta64[D]')
        # 添加包含元组的整数类型
        types.append([('a', int), ('b', int)])
        # 添加包含元组的整数和浮点数类型
        types.append([('a', int), ('b', float)])

        # 遍历类型列表
        for dtype in types:
            # 调用 _run_axis_tests 方法处理类型测试
            self._run_axis_tests(dtype)

        # 错误消息字符串
        msg = 'Non-bitwise-equal booleans test failed'
        # 创建布尔值数据
        data = np.arange(10, dtype=np.uint8).reshape(-1, 2).view(bool)
        # 预期的结果数组
        result = np.array([[False, True], [True, True]], dtype=bool)
        # 断言按轴0处理的唯一性，比较数据和预期结果
        assert_array_equal(unique(data, axis=0), result, msg)

        # 错误消息字符串
        msg = 'Negative zero equality test failed'
        # 创建包含负零的浮点数数据
        data = np.array([[-0.0, 0.0], [0.0, -0.0], [-0.0, 0.0], [0.0, -0.0]])
        # 预期的结果数组
        result = np.array([[-0.0, 0.0]])
        # 断言按轴0处理的唯一性，比较数据和预期结果
        assert_array_equal(unique(data, axis=0), result, msg)

    # 使用 pytest 标记的参数化测试，测试在指定轴上的唯一性处理
    @pytest.mark.parametrize("axis", [0, -1])
    def test_unique_1d_with_axis(self, axis):
        # 创建一维整数数组
        x = np.array([4, 3, 2, 3, 2, 1, 2, 2])
        # 调用唯一性处理函数，返回唯一值数组
        uniq = unique(x, axis=axis)
        # 断言唯一值数组与预期结果数组相等
        assert_array_equal(uniq, [1, 2, 3, 4])

    # 使用 pytest 标记的参数化测试，测试在指定轴上的唯一性和反向索引处理
    @pytest.mark.parametrize("axis", [None, 0, -1])
    def test_unique_inverse_with_axis(self, axis):
        # 创建二维整数数组
        x = np.array([[4, 4, 3], [2, 2, 1], [2, 2, 1], [4, 4, 3]])
        # 调用唯一性处理函数，并返回唯一值数组和反向索引数组
        uniq, inv = unique(x, return_inverse=True, axis=axis)
        # 断言反向索引数组的维度与原始数组的维度相同
        assert_equal(inv.ndim, x.ndim)
        # 如果轴为 None，使用唯一值数组对反向索引数组进行取值，并与原始数组比较
        if axis is None:
            assert_array_equal(x, np.take(uniq, inv))
        else:
            # 使用唯一值数组和轴对反向索引数组进行取值，并与原始数组比较
            assert_array_equal(x, np.take_along_axis(uniq, inv, axis=axis))
    def test_unique_axis_zeros(self):
        # 定义一个空的二维数组，shape为(2, 0)，元素类型为int8
        single_zero = np.empty(shape=(2, 0), dtype=np.int8)
        # 调用unique函数，沿着axis=0进行唯一化操作，并返回唯一化后的结果、索引、逆向索引和计数
        uniq, idx, inv, cnt = unique(single_zero, axis=0, return_index=True,
                                     return_inverse=True, return_counts=True)

        # 断言唯一化结果的数据类型与single_zero相同
        assert_equal(uniq.dtype, single_zero.dtype)
        # 断言唯一化后的结果是一个形状为(1, 0)的空数组
        assert_array_equal(uniq, np.empty(shape=(1, 0)))
        # 断言索引数组idx包含一个元素0
        assert_array_equal(idx, np.array([0]))
        # 断言逆向索引数组inv为一个形状为(2, 1)的数组，全为0
        assert_array_equal(inv, np.array([[0], [0]]))
        # 断言计数数组cnt包含一个元素2
        assert_array_equal(cnt, np.array([2]))

        # 再次调用unique函数，沿着axis=1进行唯一化操作
        uniq, idx, inv, cnt = unique(single_zero, axis=1, return_index=True,
                                     return_inverse=True, return_counts=True)

        # 断言唯一化结果的数据类型与single_zero相同
        assert_equal(uniq.dtype, single_zero.dtype)
        # 断言唯一化后的结果是一个形状为(2, 0)的空数组
        assert_array_equal(uniq, np.empty(shape=(2, 0)))
        # 断言索引数组idx为空数组
        assert_array_equal(idx, np.array([]))
        # 断言逆向索引数组inv是一个形状为(1, 0)的空数组
        assert_array_equal(inv, np.empty((1, 0)))
        # 断言计数数组cnt为空数组
        assert_array_equal(cnt, np.array([]))

        # 测试一个复杂的形状
        shape = (0, 2, 0, 3, 0, 4, 0)
        # 创建一个根据shape定义的空数组
        multiple_zeros = np.empty(shape=shape)
        # 对于数组的每个轴进行循环
        for axis in range(len(shape)):
            # 创建期望的结果形状列表，如果当前轴上的大小为0，则期望结果的该轴也为0，否则为1
            expected_shape = list(shape)
            if shape[axis] == 0:
                expected_shape[axis] = 0
            else:
                expected_shape[axis] = 1

            # 断言在当前轴上进行唯一化操作后的结果与期望的空数组形状相同
            assert_array_equal(unique(multiple_zeros, axis=axis),
                               np.empty(shape=expected_shape))

    def test_unique_masked(self):
        # issue 8664
        # 创建一个包含有0的uint8类型的数组x
        x = np.array([64, 0, 1, 2, 3, 63, 63, 0, 0, 0, 1, 2, 0, 63, 0],
                     dtype='uint8')
        # 创建一个掩码数组y，将x中所有值为0的元素掩盖
        y = np.ma.masked_equal(x, 0)

        # 调用unique函数对掩码数组y进行唯一化操作
        v = np.unique(y)
        # 再次调用unique函数，同时返回唯一化后的结果、索引和计数
        v2, i, c = np.unique(y, return_index=True, return_counts=True)

        # 断言简单唯一化操作和详细唯一化操作返回的结果一致
        msg = 'Unique returned different results when asked for index'
        assert_array_equal(v.data, v2.data, msg)
        assert_array_equal(v.mask, v2.mask, msg)

    def test_unique_sort_order_with_axis(self):
        # These tests fail if sorting along axis is done by treating subarrays
        # as unsigned byte strings.  See gh-10495.
        # 定义不同整数类型的数组a，形状为(-1, 1)或(0, 1)
        fmt = "sort order incorrect for integer type '%s'"
        for dt in 'bhilq':
            a = np.array([[-1], [0]], dt)
            # 对数组a沿着axis=0进行唯一化操作
            b = np.unique(a, axis=0)
            # 断言唯一化后的结果与原数组a相同，如果排序顺序不正确，则给出错误信息
            assert_array_equal(a, b, fmt % dt)
    def _run_axis_tests(self, dtype):
        # 创建一个二维数组作为测试数据，将其转换为指定的数据类型
        data = np.array([[0, 1, 0, 0],
                         [1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [1, 0, 0, 0]]).astype(dtype)

        # 测试 unique 函数对一维数组应用，并验证结果是否符合预期
        msg = 'Unique with 1d array and axis=0 failed'
        result = np.array([0, 1])
        assert_array_equal(unique(data), result.astype(dtype), msg)

        # 测试 unique 函数对二维数组按指定轴应用，并验证结果是否符合预期
        msg = 'Unique with 2d array and axis=0 failed'
        result = np.array([[0, 1, 0, 0], [1, 0, 0, 0]])
        assert_array_equal(unique(data, axis=0), result.astype(dtype), msg)

        # 测试 unique 函数对二维数组按指定轴应用，并验证结果是否符合预期
        msg = 'Unique with 2d array and axis=1 failed'
        result = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
        assert_array_equal(unique(data, axis=1), result.astype(dtype), msg)

        # 测试 unique 函数对三维数组按指定轴应用，并验证结果是否符合预期
        msg = 'Unique with 3d array and axis=2 failed'
        data3d = np.array([[[1, 1],
                            [1, 0]],
                           [[0, 1],
                            [0, 0]]]).astype(dtype)
        result = np.take(data3d, [1, 0], axis=2)
        assert_array_equal(unique(data3d, axis=2), result, msg)

        # 测试 unique 函数在指定多个参数的情况下的返回结果是否符合预期
        uniq, idx, inv, cnt = unique(data, axis=0, return_index=True,
                                     return_inverse=True, return_counts=True)
        msg = "Unique's return_index=True failed with axis=0"
        assert_array_equal(data[idx], uniq, msg)
        msg = "Unique's return_inverse=True failed with axis=0"
        assert_array_equal(np.take_along_axis(uniq, inv, axis=0), data)
        msg = "Unique's return_counts=True failed with axis=0"
        assert_array_equal(cnt, np.array([2, 2]), msg)

        # 测试 unique 函数在指定多个参数的情况下的返回结果是否符合预期
        uniq, idx, inv, cnt = unique(data, axis=1, return_index=True,
                                     return_inverse=True, return_counts=True)
        msg = "Unique's return_index=True failed with axis=1"
        assert_array_equal(data[:, idx], uniq)
        msg = "Unique's return_inverse=True failed with axis=1"
        assert_array_equal(np.take_along_axis(uniq, inv, axis=1), data)
        msg = "Unique's return_counts=True failed with axis=1"
        assert_array_equal(cnt, np.array([2, 1, 1]), msg)

    def test_unique_nanequals(self):
        # issue 20326 的测试用例
        a = np.array([1, 1, np.nan, np.nan, np.nan])
        unq = np.unique(a)
        not_unq = np.unique(a, equal_nan=False)
        assert_array_equal(unq, np.array([1, np.nan]))
        assert_array_equal(not_unq, np.array([1, np.nan, np.nan, np.nan]))
    # 定义测试函数，用于测试唯一数组 API 的各个功能
    def test_unique_array_api_functions(self):
        # 创建包含 NaN 值的 NumPy 数组
        arr = np.array([np.nan, 1, 4, 1, 3, 4, np.nan, 5, 1])

        # 遍历不同的唯一数组 API 函数调用及其预期结果
        for res_unique_array_api, res_unique in [
            (
                np.unique_values(arr),  # 调用自定义的唯一值函数
                np.unique(arr, equal_nan=False)  # 使用 NumPy 提供的唯一值函数
            ),
            (
                np.unique_counts(arr),  # 调用自定义的唯一值计数函数
                np.unique(arr, return_counts=True, equal_nan=False)  # 使用 NumPy 提供的唯一值计数函数
            ),
            (
                np.unique_inverse(arr),  # 调用自定义的唯一值逆映射函数
                np.unique(arr, return_inverse=True, equal_nan=False)  # 使用 NumPy 提供的唯一值逆映射函数
            ),
            (
                np.unique_all(arr),  # 调用自定义的全功能唯一值函数
                np.unique(
                    arr,
                    return_index=True,
                    return_inverse=True,
                    return_counts=True,
                    equal_nan=False  # 使用 NumPy 提供的全功能唯一值函数
                )
            )
        ]:
            # 断言自定义函数返回的结果长度与 NumPy 函数返回结果长度相等
            assert len(res_unique_array_api) == len(res_unique)
            # 遍历并断言每个元素是否相等
            for actual, expected in zip(res_unique_array_api, res_unique):
                assert_array_equal(actual, expected)

    # 定义测试函数，用于测试唯一值逆映射函数的形状问题
    def test_unique_inverse_shape(self):
        # 创建一个二维 NumPy 数组作为测试数据
        arr = np.array([[1, 2, 3], [2, 3, 1]])

        # 使用 NumPy 提供的唯一值及逆映射函数获取预期值和逆映射结果
        expected_values, expected_inverse = np.unique(arr, return_inverse=True)
        expected_inverse = expected_inverse.reshape(arr.shape)

        # 遍历要测试的函数：唯一值逆映射函数和全功能唯一值函数
        for func in np.unique_inverse, np.unique_all:
            # 调用函数获取实际结果
            result = func(arr)

            # 断言预期值与实际值的数组是否相等
            assert_array_equal(expected_values, result.values)
            # 断言预期逆映射结果与实际逆映射结果的数组是否相等
            assert_array_equal(expected_inverse, result.inverse_indices)
            # 断言原始数组与通过逆映射结果索引后的值数组是否相等
            assert_array_equal(arr, result.values[result.inverse_indices])
```
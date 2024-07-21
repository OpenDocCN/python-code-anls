# `.\pytorch\test\torch_np\numpy_tests\lib\test_arraysetops.py`

```py
# Owner(s): ["module: dynamo"]

"""Test functions for 1D array set operations.

"""
# 引入测试所需的模块和函数
from unittest import expectedFailure as xfail, skipIf

# 引入 NumPy 库
import numpy

# 引入断言相关的函数
from pytest import raises as assert_raises

# 引入 Torch 相关的测试工具和函数
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo,
)

# 如果使用 TorchDynamo 进行测试，则使用 NumPy
# 在 eager 模式下测试，则使用 torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import ediff1d, in1d, intersect1d, setdiff1d, setxor1d, union1d, unique
    from numpy.testing import assert_array_equal, assert_equal, assert_raises_regex

else:
    import torch._numpy as np
    from torch._numpy import unique
    from torch._numpy.testing import assert_array_equal, assert_equal


# 标记测试类，设置跳过条件，NumPy 版本要求大于等于 1.24
@skipIf(numpy.__version__ < "1.24", reason="NP_VER: fails on NumPy 1.23.x")
@skipIf(True, reason="TODO implement these ops")
@instantiate_parametrized_tests
class TestSetOps(TestCase):
    def test_intersect1d(self):
        # 唯一值输入
        a = np.array([5, 7, 1, 2])
        b = np.array([2, 4, 3, 1, 5])

        ec = np.array([1, 2, 5])
        # 计算数组 a 和 b 的交集，假定输入数组已经是唯一的
        c = intersect1d(a, b, assume_unique=True)
        assert_array_equal(c, ec)

        # 非唯一值输入
        a = np.array([5, 5, 7, 1, 2])
        b = np.array([2, 1, 4, 3, 3, 1, 5])

        ed = np.array([1, 2, 5])
        # 计算数组 a 和 b 的交集
        c = intersect1d(a, b)
        assert_array_equal(c, ed)
        assert_array_equal([], intersect1d([], []))

    def test_intersect1d_array_like(self):
        # 见 GitHub issue #11772
        # 定义一个类，实现 __array__ 方法返回一个数组
        class Test:
            def __array__(self):
                return np.arange(3)

        a = Test()
        # 测试自定义类 Test 的数组与自身的交集
        res = intersect1d(a, a)
        assert_array_equal(res, a)
        # 测试普通列表的交集
        res = intersect1d([1, 2, 3], [1, 2, 3])
        assert_array_equal(res, [1, 2, 3])
    # 定义一个测试函数，用于测试 intersect1d 函数的功能
    def test_intersect1d_indices(self):
        # 创建数组 a 和 b，包含唯一元素
        a = np.array([1, 2, 3, 4])
        b = np.array([2, 1, 4, 6])
        # 调用 intersect1d 函数，返回交集 c，以及在 a 和 b 中对应的索引 i1 和 i2
        c, i1, i2 = intersect1d(a, b, assume_unique=True, return_indices=True)
        # 预期的交集数组 ee
        ee = np.array([1, 2, 4])
        # 断言 c 是否等于 ee
        assert_array_equal(c, ee)
        # 断言 a 中 i1 索引对应的元素是否等于 ee
        assert_array_equal(a[i1], ee)
        # 断言 b 中 i2 索引对应的元素是否等于 ee
        assert_array_equal(b[i2], ee)

        # 创建数组 a 和 b，包含非唯一元素
        a = np.array([1, 2, 2, 3, 4, 3, 2])
        b = np.array([1, 8, 4, 2, 2, 3, 2, 3])
        # 调用 intersect1d 函数，返回交集 c，以及在 a 和 b 中对应的索引 i1 和 i2
        c, i1, i2 = intersect1d(a, b, return_indices=True)
        # 预期的交集数组 ef
        ef = np.array([1, 2, 3, 4])
        # 断言 c 是否等于 ef
        assert_array_equal(c, ef)
        # 断言 a 中 i1 索引对应的元素是否等于 ef
        assert_array_equal(a[i1], ef)
        # 断言 b 中 i2 索引对应的元素是否等于 ef
        assert_array_equal(b[i2], ef)

        # 创建二维数组 a 和 b，假定数组元素唯一
        a = np.array([[2, 4, 5, 6], [7, 8, 1, 15]])
        b = np.array([[3, 2, 7, 6], [10, 12, 8, 9]])
        # 调用 intersect1d 函数，返回交集 c，以及在 a 和 b 中对应的索引 i1 和 i2
        c, i1, i2 = intersect1d(a, b, assume_unique=True, return_indices=True)
        # 将一维索引 i1 和 i2 转换为对应的二维索引 ui1 和 ui2
        ui1 = np.unravel_index(i1, a.shape)
        ui2 = np.unravel_index(i2, b.shape)
        # 预期的交集数组 ea
        ea = np.array([2, 6, 7, 8])
        # 断言从数组 a 中提取的元素是否等于 ea
        assert_array_equal(ea, a[ui1])
        # 断言从数组 b 中提取的元素是否等于 ea
        assert_array_equal(ea, b[ui2])

        # 创建二维数组 a 和 b，不假定数组元素唯一
        a = np.array([[2, 4, 5, 6, 6], [4, 7, 8, 7, 2]])
        b = np.array([[3, 2, 7, 7], [10, 12, 8, 7]])
        # 调用 intersect1d 函数，返回交集 c，以及在 a 和 b 中对应的索引 i1 和 i2
        c, i1, i2 = intersect1d(a, b, return_indices=True)
        # 将一维索引 i1 和 i2 转换为对应的二维索引 ui1 和 ui2
        ui1 = np.unravel_index(i1, a.shape)
        ui2 = np.unravel_index(i2, b.shape)
        # 预期的交集数组 ea
        ea = np.array([2, 7, 8])
        # 断言从数组 a 中提取的元素是否等于 ea
        assert_array_equal(ea, a[ui1])
        # 断言从数组 b 中提取的元素是否等于 ea
        assert_array_equal(ea, b[ui2])

    # 定义一个测试函数，用于测试 setxor1d 函数的功能
    def test_setxor1d(self):
        # 创建数组 a 和 b，进行 setxor1d 操作
        a = np.array([5, 7, 1, 2])
        b = np.array([2, 4, 3, 1, 5])
        # 预期的对称差数组 ec
        ec = np.array([3, 4, 7])
        # 调用 setxor1d 函数，计算数组 a 和 b 的对称差
        c = setxor1d(a, b)
        # 断言计算结果是否等于预期的 ec
        assert_array_equal(c, ec)

        # 创建数组 a 和 b，进行 setxor1d 操作
        a = np.array([1, 2, 3])
        b = np.array([6, 5, 4])
        # 预期的对称差数组 ec
        ec = np.array([1, 2, 3, 4, 5, 6])
        # 调用 setxor1d 函数，计算数组 a 和 b 的对称差
        c = setxor1d(a, b)
        # 断言计算结果是否等于预期的 ec
        assert_array_equal(c, ec)

        # 创建数组 a 和 b，进行 setxor1d 操作
        a = np.array([1, 8, 2, 3])
        b = np.array([6, 5, 4, 8])
        # 预期的对称差数组 ec
        ec = np.array([1, 2, 3, 4, 5, 6])
        # 调用 setxor1d 函数，计算数组 a 和 b 的对称差
        c = setxor1d(a, b)
        # 断言计算结果是否等于预期的 ec
        assert_array_equal(c, ec)

        # 对空数组进行 setxor1d 操作
        assert_array_equal([], setxor1d([], []))
    # 定义单元测试方法，测试函数 ediff1d 的各种情况
    def test_ediff1d(self):
        # 创建不同大小的 numpy 数组作为测试数据
        zero_elem = np.array([])
        one_elem = np.array([1])
        two_elem = np.array([1, 2])

        # 断言空数组的差分结果为空数组
        assert_array_equal([], ediff1d(zero_elem))
        # 断言空数组添加起始值为 0 后的差分结果为 [0]
        assert_array_equal([0], ediff1d(zero_elem, to_begin=0))
        # 断言空数组添加末尾值为 0 后的差分结果为 [0]
        assert_array_equal([0], ediff1d(zero_elem, to_end=0))
        # 断言空数组添加起始值为 -1，末尾值为 0 后的差分结果为 [-1, 0]
        assert_array_equal([-1, 0], ediff1d(zero_elem, to_begin=-1, to_end=0))
        # 断言单元素数组的差分结果为空数组
        assert_array_equal([], ediff1d(one_elem))
        # 断言两元素数组的差分结果为 [1]
        assert_array_equal([1], ediff1d(two_elem))
        # 断言两元素数组添加起始值为 7，末尾值为 9 后的差分结果为 [7, 1, 9]
        assert_array_equal([7, 1, 9], ediff1d(two_elem, to_begin=7, to_end=9))
        # 断言两元素数组添加起始数组为 [5, 6]，末尾数组为 [7, 8] 后的差分结果为 [5, 6, 1, 7, 8]
        assert_array_equal(
            [5, 6, 1, 7, 8], ediff1d(two_elem, to_begin=[5, 6], to_end=[7, 8])
        )
        # 断言两元素数组添加末尾值为 9 后的差分结果为 [1, 9]
        assert_array_equal([1, 9], ediff1d(two_elem, to_end=9))
        # 断言两元素数组添加末尾数组为 [7, 8] 后的差分结果为 [1, 7, 8]
        assert_array_equal([1, 7, 8], ediff1d(two_elem, to_end=[7, 8]))
        # 断言两元素数组添加起始值为 7 后的差分结果为 [7, 1]
        assert_array_equal([7, 1], ediff1d(two_elem, to_begin=7))
        # 断言两元素数组添加起始数组为 [5, 6] 后的差分结果为 [5, 6, 1]
        assert_array_equal([5, 6, 1], ediff1d(two_elem, to_begin=[5, 6]))

    # 使用 @parametrize 装饰器定义参数化测试
    @parametrize(
        "ary, prepend, append, expected",
        [
            # 下面的测试用例应该失败，因为尝试将 np.nan 转换为整数数组
            (
                np.array([1, 2, 3], dtype=np.int64),
                None,
                np.nan,
                "to_end"
            ),
            # 下面的测试用例应该失败，因为尝试将浮点数数组 downcast 到整数类型
            subtest(
                (
                    np.array([1, 2, 3], dtype=np.int64),
                    np.array([5, 7, 2], dtype=np.float32),
                    None,
                    "to_begin",
                ),
            ),
            # 下面的测试用例应该失败，因为尝试将特殊的浮点数值转换为整数数组
            (
                np.array([1.0, 3.0, 9.0], dtype=np.int8),
                np.nan,
                np.nan,
                "to_begin"
            ),
        ],
    )
    # 定义测试函数，验证当尝试追加或添加不兼容类型时，是否会引发适当的异常
    def test_ediff1d_forbidden_type_casts(self, ary, prepend, append, expected):
        # 验证解决 gh-11490 的问题

        # 断言在尝试追加或添加不兼容类型时，是否引发了适当的 TypeError 异常，异常消息应包含正确的期望值
        msg = f"dtype of `{expected}` must be compatible"
        with assert_raises_regex(TypeError, msg):
            ediff1d(ary=ary, to_end=append, to_begin=prepend)
    @parametrize(
        "ary,prepend,append,expected",
        [
            (
                np.array([1, 2, 3], dtype=np.int16),
                2**16,  # will be cast to int16 under same kind rule.
                2**16 + 4,
                np.array([0, 1, 1, 4], dtype=np.int16),
            ),
            (
                np.array([1, 2, 3], dtype=np.float32),
                np.array([5], dtype=np.float64),
                None,
                np.array([5, 1, 1], dtype=np.float32),
            ),
            (
                np.array([1, 2, 3], dtype=np.int32),
                0,
                0,
                np.array([0, 1, 1, 0], dtype=np.int32),
            ),
            (
                np.array([1, 2, 3], dtype=np.int64),
                3,
                -9,
                np.array([3, 1, 1, -9], dtype=np.int64),
            ),
        ],
    )


        # 定义参数化测试用例，测试不同输入下的函数行为
        def test_ediff1d_scalar_handling(self, ary, prepend, append, expected):
            # 维持向后兼容性
            # 在修复 gh-11490 后，ediff1d 的标量 prepend / append 行为
            # 调用 numpy 的 ediff1d 函数，计算数组 ary 的差分，添加指定的 prepend 和 append
            actual = np.ediff1d(ary=ary, to_end=append, to_begin=prepend)
            # 断言实际输出与预期输出相等
            assert_equal(actual, expected)
            # 断言实际输出的数据类型与预期输出的数据类型相等
            assert actual.dtype == expected.dtype


        @skipIf(True, reason="NP_VER: fails with NumPy 1.22.x")
        @parametrize("kind", [None, "sort", "table"])
    # 定义测试函数 test_isin，接受参数 self 和 kind
    def test_isin(self, kind):
        # the tests for in1d cover most of isin's behavior
        # if in1d is removed, would need to change those tests to test
        # isin instead.
        # in1d 的测试覆盖了 isin 的大部分行为
        # 如果移除 in1d，需要修改这些测试以测试 isin

        # 定义内部函数 _isin_slow，用于比较 a 是否在 b 中存在的简单版本
        def _isin_slow(a, b):
            # 将 b 转换为 NumPy 数组，并展平为一维后转为列表
            b = np.asarray(b).flatten().tolist()
            return a in b

        # 使用 np.vectorize 创建一个向量化的 _isin_slow 函数，指定输出类型为 bool，排除第二个参数（b）
        isin_slow = np.vectorize(_isin_slow, otypes=[bool], excluded={1})

        # 定义断言函数 assert_isin_equal，比较 np.isin 和 isin_slow 的结果是否相等
        def assert_isin_equal(a, b):
            x = np.isin(a, b, kind=kind)  # 使用 np.isin 检查 a 是否在 b 中，并指定比较类型为 kind
            y = isin_slow(a, b)  # 使用向量化的 _isin_slow 检查 a 是否在 b 中
            assert_array_equal(x, y)  # 断言 x 和 y 的内容是否相等

        # 多维数组作为参数
        a = np.arange(24).reshape([2, 3, 4])
        b = np.array([[10, 20, 30], [0, 1, 3], [11, 22, 33]])
        assert_isin_equal(a, b)

        # 数组样式的参数作为输入
        c = [(9, 8), (7, 6)]
        d = (9, 7)
        assert_isin_equal(c, d)

        # 零维数组作为参数
        f = np.array(3)
        assert_isin_equal(f, b)
        assert_isin_equal(a, f)
        assert_isin_equal(f, f)

        # 标量作为参数
        assert_isin_equal(5, b)
        assert_isin_equal(a, 6)
        assert_isin_equal(5, 6)

        # 空数组作为参数
        if kind != "table":
            # An empty list will become float64,
            # which is invalid for kind="table"
            # 空列表将会变成 float64 类型，对于 kind="table" 是无效的
            x = []
            assert_isin_equal(x, b)
            assert_isin_equal(a, x)
            assert_isin_equal(x, x)

        # 空数组不同类型的测试
        for dtype in [bool, np.int64, np.float64]:
            if kind == "table" and dtype == np.float64:
                continue

            if dtype in {np.int64, np.float64}:
                ar = np.array([10, 20, 30], dtype=dtype)
            elif dtype in {bool}:
                ar = np.array([True, False, False])

            # 创建一个指定类型和空数组的 NumPy 数组
            empty_array = np.array([], dtype=dtype)

            assert_isin_equal(empty_array, ar)
            assert_isin_equal(ar, empty_array)
            assert_isin_equal(empty_array, empty_array)

    # 使用 @parametrize 装饰器为 kind 参数传递多个测试参数
    @parametrize("kind", [None, "sort", "table"])
    # 定义一个测试方法，用于测试 in1d 函数在不同情况下的行为
    def test_in1d(self, kind):
        # 使用两种不同大小的 b 数组来测试 in1d() 函数的两种不同路径
        for mult in (1, 10):
            # 创建一个列表 a，用于测试列表是否正确处理
            a = [5, 7, 1, 2]
            # 创建一个长度为 5*mult 的列表 b
            b = [2, 4, 3, 1, 5] * mult
            # 预期结果数组 ec
            ec = np.array([True, False, True, True])
            # 调用 in1d 函数，传入参数 a, b，并指定 assume_unique=True 和 kind=kind
            c = in1d(a, b, assume_unique=True, kind=kind)
            # 断言 c 与预期结果 ec 相等
            assert_array_equal(c, ec)

            # 修改 a 的第一个元素为 8，更新预期结果 ec
            a[0] = 8
            ec = np.array([False, False, True, True])
            # 再次调用 in1d 函数
            c = in1d(a, b, assume_unique=True, kind=kind)
            assert_array_equal(c, ec)

            # 修改 a 的第一个和第四个元素，更新预期结果 ec
            a[0], a[3] = 4, 8
            ec = np.array([True, False, True, False])
            # 再次调用 in1d 函数
            c = in1d(a, b, assume_unique=True, kind=kind)
            assert_array_equal(c, ec)

            # 创建一个 numpy 数组 a，包含多个重复的元素
            a = np.array([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5])
            # 创建一个长度为 3*mult 的列表 b
            b = [2, 3, 4] * mult
            # 更新预期结果 ec
            ec = [
                False,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                False,
                False,
                False,
            ]
            # 再次调用 in1d 函数
            c = in1d(a, b, kind=kind)
            assert_array_equal(c, ec)

            # 将 b 扩展，加入额外的元素，更新预期结果 ec
            b = b + [5, 5, 4] * mult
            ec = [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
            ]
            # 再次调用 in1d 函数
            c = in1d(a, b, kind=kind)
            assert_array_equal(c, ec)

            # 创建一个 numpy 数组 a，包含不同的元素
            a = np.array([5, 7, 1, 2])
            # 创建一个长度为 5*mult 的 numpy 数组 b
            b = np.array([2, 4, 3, 1, 5] * mult)
            # 更新预期结果 ec
            ec = np.array([True, False, True, True])
            # 再次调用 in1d 函数
            c = in1d(a, b, kind=kind)
            assert_array_equal(c, ec)

            # 创建一个 numpy 数组 a，包含不同的元素
            a = np.array([5, 7, 1, 1, 2])
            # 创建一个长度为 6*mult 的 numpy 数组 b
            b = np.array([2, 4, 3, 3, 1, 5] * mult)
            # 更新预期结果 ec
            ec = np.array([True, False, True, True, True])
            # 再次调用 in1d 函数
            c = in1d(a, b, kind=kind)
            assert_array_equal(c, ec)

            # 创建一个 numpy 数组 a，包含相同的元素
            a = np.array([5, 5])
            # 创建一个长度为 2*mult 的 numpy 数组 b
            b = np.array([2, 2] * mult)
            # 更新预期结果 ec
            ec = np.array([False, False])
            # 再次调用 in1d 函数
            c = in1d(a, b, kind=kind)
            assert_array_equal(c, ec)

        # 创建一个包含单个元素的 numpy 数组 a 和 b
        a = np.array([5])
        b = np.array([2])
        # 更新预期结果 ec
        ec = np.array([False])
        # 再次调用 in1d 函数
        c = in1d(a, b, kind=kind)
        assert_array_equal(c, ec)

        # 如果 kind 是 None 或 "sort"，断言空数组与 in1d([], []) 相等
        if kind in {None, "sort"}:
            assert_array_equal(in1d([], [], kind=kind), [])

    # 定义一个测试方法，测试 in1d 函数对字符数组的处理
    def test_in1d_char_array(self):
        # 创建一个包含多个字符的 numpy 字符数组 a 和 b
        a = np.array(["a", "b", "c", "d", "e", "c", "e", "b"])
        b = np.array(["a", "c"])
        # 更新预期结果 ec
        ec = np.array([True, False, True, False, False, True, False, False])
        # 调用 in1d 函数
        c = in1d(a, b)
        # 断言 c 与预期结果 ec 相等
        assert_array_equal(c, ec)
    @parametrize("kind", [None, "sort", "table"])
    def test_in1d_invert(self, kind):
        "Test in1d's invert parameter"
        # We use two different sizes for the b array here to test the
        # two different paths in in1d().
        # 在这里使用两种不同大小的 b 数组，以测试 in1d() 中的两条不同路径。

        for mult in (1, 10):
            # 创建数组 a，包含整数值，用于测试
            a = np.array([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5])
            # 创建数组 b，包含重复的元素 [2, 3, 4]，倍数由 mult 决定
            b = [2, 3, 4] * mult
            # 断言 np.invert(in1d(a, b, kind=kind)) 的结果等于 in1d(a, b, invert=True, kind=kind)
            assert_array_equal(
                np.invert(in1d(a, b, kind=kind)), in1d(a, b, invert=True, kind=kind)
            )

        # float:
        # 如果 kind 是 None 或者 "sort"
        if kind in {None, "sort"}:
            for mult in (1, 10):
                # 创建浮点数数组 a，使用 np.float32 类型
                a = np.array(
                    [5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5], dtype=np.float32
                )
                # 创建数组 b，包含重复的元素 [2, 3, 4]，倍数由 mult 决定，使用 np.float32 类型
                b = [2, 3, 4] * mult
                b = np.array(b, dtype=np.float32)
                # 断言 np.invert(in1d(a, b, kind=kind)) 的结果等于 in1d(a, b, invert=True, kind=kind)
                assert_array_equal(
                    np.invert(in1d(a, b, kind=kind)), in1d(a, b, invert=True, kind=kind)
                )

    @parametrize("kind", [None, "sort", "table"])
    def test_in1d_ravel(self, kind):
        # Test that in1d ravels its input arrays. This is not documented
        # behavior however. The test is to ensure consistentency.
        # 测试 in1d 是否对其输入数组进行了展平操作。虽然这不是文档化的行为，但测试是为了确保一致性。

        # 创建数组 a 和 b，分别是形状为 (2, 3) 和 (3, 2) 的二维数组
        a = np.arange(6).reshape(2, 3)
        b = np.arange(3, 9).reshape(3, 2)
        long_b = np.arange(3, 63).reshape(30, 2)
        ec = np.array([False, False, False, True, True, True])

        # 断言 in1d(a, b, assume_unique=True, kind=kind) 的结果等于 ec
        assert_array_equal(in1d(a, b, assume_unique=True, kind=kind), ec)
        # 断言 in1d(a, b, assume_unique=False, kind=kind) 的结果等于 ec
        assert_array_equal(in1d(a, b, assume_unique=False, kind=kind), ec)
        # 断言 in1d(a, long_b, assume_unique=True, kind=kind) 的结果等于 ec
        assert_array_equal(in1d(a, long_b, assume_unique=True, kind=kind), ec)
        # 断言 in1d(a, long_b, assume_unique=False, kind=kind) 的结果等于 ec
        assert_array_equal(in1d(a, long_b, assume_unique=False, kind=kind), ec)

    def test_in1d_hit_alternate_algorithm(self):
        """Hit the standard isin code with integers"""
        # Need extreme range to hit standard code
        # This hits it without the use of kind='table'
        # 需要极端范围来触发标准的 isin 代码

        # 创建整数数组 a 和 b，使用 np.int64 类型
        a = np.array([5, 4, 5, 3, 4, 4, 1e9], dtype=np.int64)
        b = np.array([2, 3, 4, 1e9], dtype=np.int64)
        expected = np.array([0, 1, 0, 1, 1, 1, 1], dtype=bool)
        # 断言 np.in1d(a, b) 的结果等于 expected
        assert_array_equal(expected, in1d(a, b))
        # 断言 np.invert(expected) 的结果等于 in1d(a, b, invert=True)
        assert_array_equal(np.invert(expected), in1d(a, b, invert=True))

        a = np.array([5, 7, 1, 2], dtype=np.int64)
        b = np.array([2, 4, 3, 1, 5, 1e9], dtype=np.int64)
        ec = np.array([True, False, True, True])
        c = in1d(a, b, assume_unique=True)
        # 断言 c 的结果等于 ec
        assert_array_equal(c, ec)

    @parametrize("kind", [None, "sort", "table"])
    def test_in1d_boolean(self, kind):
        """Test that in1d works for boolean input"""
        # 测试 in1d 是否适用于布尔输入

        # 创建布尔数组 a 和 b
        a = np.array([True, False])
        b = np.array([False, False, False])
        expected = np.array([False, True])
        # 断言 in1d(a, b, kind=kind) 的结果等于 expected
        assert_array_equal(expected, in1d(a, b, kind=kind))
        # 断言 np.invert(expected) 的结果等于 in1d(a, b, invert=True, kind=kind)
        assert_array_equal(np.invert(expected), in1d(a, b, invert=True, kind=kind))

    @parametrize("kind", [None, "sort"])
    # 测试函数，用于测试 in1d 是否能处理 timedelta 类型的输入
    def test_in1d_timedelta(self, kind):
        """Test that in1d works for timedelta input"""
        # 创建随机状态生成器
        rstate = np.random.RandomState(0)
        # 生成随机整数数组 a 和 b
        a = rstate.randint(0, 100, size=10)
        b = rstate.randint(0, 100, size=10)
        # 使用 in1d 函数计算 a 和 b 的结果
        truth = in1d(a, b)
        # 将数组 a 和 b 转换为 timedelta64 类型
        a_timedelta = a.astype("timedelta64[s]")
        b_timedelta = b.astype("timedelta64[s]")
        # 断言 in1d 函数对 timedelta 类型输入的计算结果
        assert_array_equal(truth, in1d(a_timedelta, b_timedelta, kind=kind))

    # 测试函数，检验当输入的是 timedelta 类型时，in1d 函数是否会失败
    def test_in1d_table_timedelta_fails(self):
        # 创建 timedelta64 类型的数组 a 和 b
        a = np.array([0, 1, 2], dtype="timedelta64[s]")
        b = a
        # 确保 in1d 函数在这种情况下会引发 ValueError 异常
        with assert_raises(ValueError):
            in1d(a, b, kind="table")

    # 参数化测试函数，测试不同混合 dtype 的情况下，in1d 函数的行为
    @parametrize(
        "dtype1,dtype2",
        [
            (np.int8, np.int16),
            (np.int16, np.int8),
        ],
    )
    @parametrize("kind", [None, "sort", "table"])
    def test_in1d_mixed_dtype(self, dtype1, dtype2, kind):
        """Test that in1d works as expected for mixed dtype input."""
        # 检查 dtype2 是否为有符号整数类型
        is_dtype2_signed = np.issubdtype(dtype2, np.signedinteger)
        # 创建 dtype1 类型的数组 ar1
        ar1 = np.array([0, 0, 1, 1], dtype=dtype1)

        # 根据 dtype2 类型不同，创建数组 ar2
        if is_dtype2_signed:
            ar2 = np.array([-128, 0, 127], dtype=dtype2)
        else:
            ar2 = np.array([127, 0, 255], dtype=dtype2)

        # 预期的结果数组
        expected = np.array([True, True, False, False])

        # 如果 kind 是 "table" 并且满足特定条件，则预期测试会失败
        expect_failure = kind == "table" and any(
            (
                dtype1 == np.int8 and dtype2 == np.int16,
                dtype1 == np.int16 and dtype2 == np.int8,
            )
        )

        # 根据预期是否失败来进行断言
        if expect_failure:
            # 确保在这种情况下 in1d 函数会引发 RuntimeError 异常
            with assert_raises(RuntimeError, match="exceed the maximum"):
                in1d(ar1, ar2, kind=kind)
        else:
            # 断言 in1d 函数的计算结果是否符合预期
            assert_array_equal(in1d(ar1, ar2, kind=kind), expected)

    # 参数化测试函数，测试 bool 和 int 混合输入情况下，in1d 函数的行为
    @parametrize("kind", [None, "sort", "table"])
    def test_in1d_mixed_boolean(self, kind):
        """Test that in1d works as expected for bool/int input."""
        # 遍历所有整数类型的 dtype
        for dtype in np.typecodes["AllInteger"]:
            # 创建布尔类型数组 a
            a = np.array([True, False, False], dtype=bool)
            # 创建指定 dtype 类型的数组 b
            b = np.array([0, 0, 0, 0], dtype=dtype)
            # 预期的结果数组
            expected = np.array([False, True, True], dtype=bool)
            # 断言 in1d 函数对于布尔类型和整数类型的输入的计算结果
            assert_array_equal(in1d(a, b, kind=kind), expected)

            # 交换 a 和 b，重新计算预期的结果数组
            a, b = b, a
            expected = np.array([True, True, True, True], dtype=bool)
            # 断言 in1d 函数对于交换后的输入的计算结果
            assert_array_equal(in1d(a, b, kind=kind), expected)

    # 测试函数，检验当第一个数组是对象类型时，in1d 函数的行为
    def test_in1d_first_array_is_object(self):
        # 创建对象类型的数组 ar1 和整数类型的数组 ar2
        ar1 = [None]
        ar2 = np.array([1] * 10)
        # 预期的结果数组
        expected = np.array([False])
        # 计算 in1d 函数的结果
        result = np.in1d(ar1, ar2)
        # 断言 in1d 函数的计算结果是否符合预期
        assert_array_equal(result, expected)

    # 测试函数，检验当第二个数组是对象类型时，in1d 函数的行为
    def test_in1d_second_array_is_object(self):
        # 创建整数类型的 ar1 和对象类型的 ar2
        ar1 = 1
        ar2 = np.array([None] * 10)
        # 预期的结果数组
        expected = np.array([False])
        # 计算 in1d 函数的结果
        result = np.in1d(ar1, ar2)
        # 断言 in1d 函数的计算结果是否符合预期
        assert_array_equal(result, expected)
    # 定义一个测试函数，测试 np.in1d 函数对两个包含对象的数组的操作
    def test_in1d_both_arrays_are_object(self):
        # 创建包含一个 None 元素的列表 ar1
        ar1 = [None]
        # 创建包含 10 个 None 元素的 NumPy 数组 ar2
        ar2 = np.array([None] * 10)
        # 创建期望的结果数组，包含一个 True 元素
        expected = np.array([True])
        # 使用 np.in1d 函数比较 ar1 和 ar2，返回比较结果的数组 result
        result = np.in1d(ar1, ar2)
        # 使用 assert_array_equal 断言函数验证 result 是否等于 expected
        assert_array_equal(result, expected)

    @xfail
    # 定义一个预期失败的测试函数，测试 np.in1d 函数对包含结构化数据类型的数组的操作
    def test_in1d_both_arrays_have_structured_dtype(self):
        # 定义结构化数据类型 dt，包含一个整数字段和一个允许任意 Python 对象的字段
        dt = np.dtype([("field1", int), ("field2", object)])
        # 创建包含一个元素的结构化数组 ar1，使用定义的数据类型 dt
        ar1 = np.array([(1, None)], dtype=dt)
        # 创建包含 10 个元素的结构化数组 ar2，使用定义的数据类型 dt
        ar2 = np.array([(1, None)] * 10, dtype=dt)
        # 创建期望的结果数组，包含一个 True 元素
        expected = np.array([True])
        # 使用 np.in1d 函数比较 ar1 和 ar2，返回比较结果的数组 result
        result = np.in1d(ar1, ar2)
        # 使用 assert_array_equal 断言函数验证 result 是否等于 expected
        assert_array_equal(result, expected)

    # 定义一个测试函数，测试 np.in1d 函数对包含元组的数组的操作
    def test_in1d_with_arrays_containing_tuples(self):
        # 创建包含一个元组和一个整数的对象数组 ar1，数据类型为 object
        ar1 = np.array([(1,), 2], dtype=object)
        # 创建与 ar1 相同的对象数组 ar2，数据类型为 object
        ar2 = np.array([(1,), 2], dtype=object)
        # 创建期望的结果数组，包含两个 True 元素
        expected = np.array([True, True])
        # 使用 np.in1d 函数比较 ar1 和 ar2，返回比较结果的数组 result
        result = np.in1d(ar1, ar2)
        # 使用 assert_array_equal 断言函数验证 result 是否等于 expected
        assert_array_equal(result, expected)
        # 使用 np.invert 函数对 expected 取反，作为反向比较的期望结果
        inverted_expected = np.invert(expected)
        # 使用 np.in1d 函数对 ar1 和 ar2 进行反向比较，返回比较结果的数组 result
        result = np.in1d(ar1, ar2, invert=True)
        # 使用 assert_array_equal 断言函数验证 result 是否等于 inverted_expected
        assert_array_equal(result, inverted_expected)

        # 在数组末尾添加一个整数，以确保数组构建器创建带有元组的数组
        # 构建器存在无法正确处理元组的 bug，添加整数可以修复该问题
        ar1 = np.array([(1,), (2, 1), 1], dtype=object)
        ar1 = ar1[:-1]  # 移除末尾的整数
        ar2 = np.array([(1,), (2, 1), 1], dtype=object)
        ar2 = ar2[:-1]  # 移除末尾的整数
        # 创建期望的结果数组，包含两个 True 元素
        expected = np.array([True, True])
        # 使用 np.in1d 函数比较 ar1 和 ar2，返回比较结果的数组 result
        result = np.in1d(ar1, ar2)
        # 使用 assert_array_equal 断言函数验证 result 是否等于 expected
        assert_array_equal(result, expected)
        # 使用 np.in1d 函数对 ar1 和 ar2 进行反向比较，返回比较结果的数组 result
        result = np.in1d(ar1, ar2, invert=True)
        # 使用 assert_array_equal 断言函数验证 result 是否等于 inverted_expected
        assert_array_equal(result, np.invert(expected))

        # 创建包含一个元组和一个整数的对象数组 ar1，数据类型为 object
        ar1 = np.array([(1,), (2, 3), 1], dtype=object)
        ar1 = ar1[:-1]  # 移除末尾的整数
        # 创建与 ar1 不同的对象数组 ar2，数据类型为 object
        ar2 = np.array([(1,), 2], dtype=object)
        # 创建期望的结果数组，包含一个 True 元素和一个 False 元素
        expected = np.array([True, False])
        # 使用 np.in1d 函数比较 ar1 和 ar2，返回比较结果的数组 result
        result = np.in1d(ar1, ar2)
        # 使用 assert_array_equal 断言函数验证 result 是否等于 expected
        assert_array_equal(result, expected)
        # 使用 np.in1d 函数对 ar1 和 ar2 进行反向比较，返回比较结果的数组 result
        result = np.in1d(ar1, ar2, invert=True)
        # 使用 assert_array_equal 断言函数验证 result 是否等于 inverted_expected
        assert_array_equal(result, np.invert(expected))
    # 测试 `in1d` 函数的错误情况

    # Error 1: 当 `kind` 参数不是 'sort' 'table' 或 None 时，应该引发 ValueError 异常。
    ar1 = np.array([1, 2, 3, 4, 5])
    ar2 = np.array([2, 4, 6, 8, 10])
    assert_raises(ValueError, in1d, ar1, ar2, kind="quicksort")

    # Error 2: 当 `kind="table"` 时，对于非整数数组会引发 ValueError 异常。
    obj_ar1 = np.array([1, "a", 3, "b", 5], dtype=object)
    obj_ar2 = np.array([1, "a", 3, "b", 5], dtype=object)
    assert_raises(ValueError, in1d, obj_ar1, obj_ar2, kind="table")

    for dtype in [np.int32, np.int64]:
        ar1 = np.array([-1, 2, 3, 4, 5], dtype=dtype)
        # 数组范围会溢出的情况:
        overflow_ar2 = np.array([-1, np.iinfo(dtype).max], dtype=dtype)

        # Error 3: 当 `kind="table"` 时，如果计算 ar2 的范围时会发生整数溢出，会引发 RuntimeError 异常。
        assert_raises(RuntimeError, in1d, ar1, overflow_ar2, kind="table")

        # 非错误情况：当 `kind=None` 时，即使可能存在整数溢出，不会引发运行时错误，
        # 而是会切换到 `sort` 算法。
        result = np.in1d(ar1, overflow_ar2, kind=None)
        assert_array_equal(result, [True] + [False] * 4)
        result = np.in1d(ar1, overflow_ar2, kind="sort")
        assert_array_equal(result, [True] + [False] * 4)
    # 定义一个测试方法，测试集合操作函数的不同用法和结果
    def test_manyways(self):
        # 创建两个 NumPy 数组 a 和 b
        a = np.array([5, 7, 1, 2, 8])
        b = np.array([9, 8, 2, 4, 3, 1, 5])

        # 使用 setxor1d 函数计算数组 a 和 b 的对称差集
        c1 = setxor1d(a, b)
        # 使用 intersect1d 函数计算数组 a 和 b 的交集
        aux1 = intersect1d(a, b)
        # 使用 union1d 函数计算数组 a 和 b 的并集
        aux2 = union1d(a, b)
        # 使用 setdiff1d 函数计算 aux2 和 aux1 的差集，得到 c2
        c2 = setdiff1d(aux2, aux1)
        # 断言 c1 和 c2 相等，确保集合操作函数的结果正确
        assert_array_equal(c1, c2)
# 使用装饰器将当前类标记为参数化测试的类
@instantiate_parametrized_tests
class TestUnique(TestCase):
    # 以下两行代码被注释掉，可能是暂时不需要执行的测试或者注释掉的代码
    # assert_equal(a3_idx.dtype, np.intp)
    # assert_equal(a3_inv.dtype, np.intp)

    # 使用装饰器标记当前方法，该方法用于特定情况下跳过测试（例如，针对特定的Torch Dynamo）
    @xpassIfTorchDynamo  # (reason="unique with nans")
    def test_unique_1d_2(self):
        # test for ticket 2111 - float
        a = [2.0, np.nan, 1.0, np.nan]
        ua = [1.0, 2.0, np.nan]
        ua_idx = [2, 0, 1]
        ua_inv = [1, 2, 0, 2]
        ua_cnt = [1, 1, 2]
        # 断言确保 np.unique 函数按预期返回结果
        assert_equal(np.unique(a), ua)
        assert_equal(np.unique(a, return_index=True), (ua, ua_idx))
        assert_equal(np.unique(a, return_inverse=True), (ua, ua_inv))
        assert_equal(np.unique(a, return_counts=True), (ua, ua_cnt))

        # test for ticket 2111 - complex
        a = [2.0 - 1j, np.nan, 1.0 + 1j, complex(0.0, np.nan), complex(1.0, np.nan)]
        ua = [1.0 + 1j, 2.0 - 1j, complex(0.0, np.nan)]
        ua_idx = [2, 0, 3]
        ua_inv = [1, 2, 0, 2, 2]
        ua_cnt = [1, 1, 3]
        # 断言确保 np.unique 函数按预期返回结果
        assert_equal(np.unique(a), ua)
        assert_equal(np.unique(a, return_index=True), (ua, ua_idx))
        assert_equal(np.unique(a, return_inverse=True), (ua, ua_inv))
        assert_equal(np.unique(a, return_counts=True), (ua, ua_cnt))

        # test for gh-19300
        all_nans = [np.nan] * 4
        ua = [np.nan]
        ua_idx = [0]
        ua_inv = [0, 0, 0, 0]
        ua_cnt = [4]
        # 断言确保 np.unique 函数按预期返回结果
        assert_equal(np.unique(all_nans), ua)
        assert_equal(np.unique(all_nans, return_index=True), (ua, ua_idx))
        assert_equal(np.unique(all_nans, return_inverse=True), (ua, ua_inv))
        assert_equal(np.unique(all_nans, return_counts=True), (ua, ua_cnt))

    # 测试在指定轴向上调用 unique 函数时是否引发 AxisError 异常
    def test_unique_axis_errors(self):
        assert_raises(np.AxisError, unique, np.arange(10), axis=2)
        assert_raises(np.AxisError, unique, np.arange(10), axis=-2)

    # 测试在处理嵌套列表时，unique 函数能否正确处理指定轴向上的唯一值
    def test_unique_axis_list(self):
        msg = "Unique failed on list of lists"
        inp = [[0, 1, 0], [0, 1, 0]]
        inp_arr = np.asarray(inp)
        assert_array_equal(unique(inp, axis=0), unique(inp_arr, axis=0), msg)
        assert_array_equal(unique(inp, axis=1), unique(inp_arr, axis=1), msg)

    # 使用装饰器标记当前方法，跳过特定情况下的测试（例如，与 Torch 的 unique 行为不一致）
    @xpassIfTorchDynamo  # _run_axis_tests xfails with the message
    # torch has different unique ordering behaviour"
    def test_unique_axis(self):
        types = []
        types.extend(np.typecodes["AllInteger"])
        types.extend(np.typecodes["AllFloat"])

        # 针对不同数据类型执行 _run_axis_tests 方法
        for dtype in types:
            self._run_axis_tests(dtype)

        # 断言确保 unique 函数在处理特定数据类型时能够按预期返回结果
        msg = "Non-bitwise-equal booleans test failed"
        data = np.arange(10, dtype=np.uint8).reshape(-1, 2).view(bool)
        result = np.array([[False, True], [True, True]], dtype=bool)
        assert_array_equal(unique(data, axis=0), result, msg)

        msg = "Negative zero equality test failed"
        data = np.array([[-0.0, 0.0], [0.0, -0.0], [-0.0, 0.0], [0.0, -0.0]])
        result = np.array([[-0.0, 0.0]])
        assert_array_equal(unique(data, axis=0), result, msg)

    # 使用参数化测试方式测试 unique 函数在不同轴向上的表现
    @parametrize("axis", [0, -1])
    # 定义一个测试函数，用于测试在给定轴上返回唯一值的情况
    def test_unique_1d_with_axis(self, axis):
        # 创建一个包含重复元素的一维 NumPy 数组
        x = np.array([4, 3, 2, 3, 2, 1, 2, 2])
        # 调用 unique 函数，返回指定轴上的唯一值数组
        uniq = unique(x, axis=axis)
        # 断言返回的唯一值数组与期望的数组相等
        assert_array_equal(uniq, [1, 2, 3, 4])

    @xpassIfTorchDynamo  # (reason="unique / return_index")
    # 标记为跳过测试条件的装饰器，原因是涉及 unique 函数的返回索引问题
    def test_unique_axis_zeros(self):
        # 解决 issue 15559
        # 创建一个空的二维数组，用于测试轴上的唯一值处理
        single_zero = np.empty(shape=(2, 0), dtype=np.int8)
        # 调用 unique 函数，返回唯一值、索引、逆向索引和计数
        uniq, idx, inv, cnt = unique(
            single_zero,
            axis=0,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )

        # 断言唯一值的数据类型与原数组相同
        assert_equal(uniq.dtype, single_zero.dtype)
        # 断言返回的唯一值数组是空数组的期望形状
        assert_array_equal(uniq, np.empty(shape=(1, 0)))
        # 断言返回的索引数组是预期的索引数组
        assert_array_equal(idx, np.array([0]))
        # 断言返回的逆向索引数组是预期的逆向索引数组
        assert_array_equal(inv, np.array([0, 0]))
        # 断言返回的计数数组是预期的计数数组
        assert_array_equal(cnt, np.array([2]))

        # 调用 unique 函数，测试在另一个轴上的唯一值处理
        uniq, idx, inv, cnt = unique(
            single_zero,
            axis=1,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )

        # 断言唯一值的数据类型与原数组相同
        assert_equal(uniq.dtype, single_zero.dtype)
        # 断言返回的唯一值数组是空数组的期望形状
        assert_array_equal(uniq, np.empty(shape=(2, 0)))
        # 断言返回的索引数组是空数组，因为轴上没有元素
        assert_array_equal(idx, np.array([]))
        # 断言返回的逆向索引数组是空数组，因为轴上没有元素
        assert_array_equal(inv, np.array([]))
        # 断言返回的计数数组是空数组，因为轴上没有元素
        assert_array_equal(cnt, np.array([]))

        # 测试一个复杂的形状
        shape = (0, 2, 0, 3, 0, 4, 0)
        multiple_zeros = np.empty(shape=shape)
        # 遍历形状的每个轴，测试唯一值处理结果
        for axis in range(len(shape)):
            expected_shape = list(shape)
            # 如果轴上的大小为0，则期望的形状也是0
            if shape[axis] == 0:
                expected_shape[axis] = 0
            else:
                expected_shape[axis] = 1

            # 断言 unique 函数返回的结果与期望的空数组形状相等
            assert_array_equal(
                unique(multiple_zeros, axis=axis), np.empty(shape=expected_shape)
            )

    def test_unique_sort_order_with_axis(self):
        # 当按轴排序时，这些测试将失败，因为子数组被视为无符号字节字符串。参见 gh-10495。
        fmt = "sort order incorrect for integer type '%s'"
        # 对于不同的整数类型进行测试
        for dt in "bhil":
            # 创建一个整数类型的二维数组
            a = np.array([[-1], [0]], dt)
            # 调用 unique 函数，返回唯一值数组
            b = np.unique(a, axis=0)
            # 断言原数组与返回的唯一值数组相等，否则输出格式字符串
            assert_array_equal(a, b, fmt % dt)
    # 定义一个私有方法 `_run_axis_tests`，接受一个参数 `dtype`
    def _run_axis_tests(self, dtype):
        # 创建一个二维 NumPy 数组，内容为特定类型的整数
        data = np.array(
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
        ).astype(dtype)

        # 测试使用默认轴（axis=0）的 `unique` 函数，断言其返回结果与预期结果相同
        msg = "Unique with 1d array and axis=0 failed"
        result = np.array([0, 1])
        assert_array_equal(unique(data), result.astype(dtype), msg)

        # 测试使用轴 0 的 `unique` 函数，断言其返回结果与预期结果相同
        msg = "Unique with 2d array and axis=0 failed"
        result = np.array([[0, 1, 0, 0], [1, 0, 0, 0]])
        assert_array_equal(unique(data, axis=0), result.astype(dtype), msg)

        # 测试使用轴 1 的 `unique` 函数，断言其返回结果与预期结果相同
        msg = "Unique with 2d array and axis=1 failed"
        result = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
        assert_array_equal(unique(data, axis=1), result.astype(dtype), msg)

        # 示例注释，展示使用 `unique` 函数处理三维数组时的预期输出
        #
        #     >>> x = np.array([[[1, 1], [0, 1]], [[1, 0], [0, 0]]])
        #     >>> np.unique(x, axis=2)
        #    [[1, 1], [0, 1]], [[1, 0], [0, 0]]
        #     >>> torch.unique(torch.as_tensor(x), dim=2)
        #    [[1, 1], [1, 0]], [[0, 1], [0, 0]]
        #
        msg = "Unique with 3d array and axis=2 failed"
        data3d = np.array([[[1, 1], [1, 0]], [[0, 1], [0, 0]]]).astype(dtype)
        result = np.take(data3d, [1, 0], axis=2)
        assert_array_equal(unique(data3d, axis=2), result, msg)

        # 测试 `unique` 函数在返回额外参数（return_index, return_inverse, return_counts）时的预期输出
        uniq, idx, inv, cnt = unique(
            data, axis=0, return_index=True, return_inverse=True, return_counts=True
        )
        msg = "Unique's return_index=True failed with axis=0"
        assert_array_equal(data[idx], uniq, msg)
        msg = "Unique's return_inverse=True failed with axis=0"
        assert_array_equal(uniq[inv], data)
        msg = "Unique's return_counts=True failed with axis=0"
        assert_array_equal(cnt, np.array([2, 2]), msg)

        uniq, idx, inv, cnt = unique(
            data, axis=1, return_index=True, return_inverse=True, return_counts=True
        )
        msg = "Unique's return_index=True failed with axis=1"
        assert_array_equal(data[:, idx], uniq)
        msg = "Unique's return_inverse=True failed with axis=1"
        assert_array_equal(uniq[:, inv], data)
        msg = "Unique's return_counts=True failed with axis=1"
        assert_array_equal(cnt, np.array([2, 1, 1]), msg)

    @skipIf(True, reason="NP_VER: fails on CI with older NumPy")
    @xpassIfTorchDynamo  # (reason="unique / return_index / nans")
    # 定义一个测试方法 `test_unique_nanequals`，针对包含 NaN 的数组进行测试
    def test_unique_nanequals(self):
        # issue 20326
        # 创建一个包含 NaN 的 NumPy 数组 `a`
        a = np.array([1, 1, np.nan, np.nan, np.nan])
        # 对数组 `a` 使用 `unique` 函数，断言返回的唯一值与预期结果相同
        unq = np.unique(a)
        # 对数组 `a` 使用 `unique` 函数，并禁用 NaN 的相等性，断言返回的唯一值与预期结果相同
        not_unq = np.unique(a, equal_nan=False)
        assert_array_equal(unq, np.array([1, np.nan]))
        assert_array_equal(not_unq, np.array([1, np.nan, np.nan, np.nan]))
# 如果当前脚本作为主程序执行（而不是被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```
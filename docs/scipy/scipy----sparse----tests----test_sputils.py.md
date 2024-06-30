# `D:\src\scipysrc\scipy\scipy\sparse\tests\test_sputils.py`

```
# 导入所需的库和模块
import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.sparse import _sputils as sputils  # 导入名为_sputils的scipy.sparse模块，并将其重命名为sputils
from scipy.sparse._sputils import matrix  # 从_sputils模块中导入matrix对象

# 定义测试类TestSparseUtils，用于测试稀疏矩阵相关的工具函数
class TestSparseUtils:

    # 定义测试函数test_upcast，用于测试数据类型的提升（upcasting）功能
    def test_upcast(self):
        # 测试upcast函数对'intc'类型的提升是否正确
        assert_equal(sputils.upcast('intc'), np.intc)
        # 测试upcast函数对'int32'和'float32'类型的提升是否正确
        assert_equal(sputils.upcast('int32', 'float32'), np.float64)
        # 测试upcast函数对'bool'、complex和float类型的提升是否正确
        assert_equal(sputils.upcast('bool', complex, float), np.complex128)
        # 测试upcast函数对'i'和'd'类型的提升是否正确
        assert_equal(sputils.upcast('i', 'd'), np.float64)

    # 定义测试函数test_getdtype，用于测试获取数据类型（dtype）的功能
    def test_getdtype(self):
        # 创建一个包含单个元素的int8类型的NumPy数组A
        A = np.array([1], dtype='int8')

        # 测试getdtype函数在未提供dtype时是否返回默认值float
        assert_equal(sputils.getdtype(None, default=float), float)
        # 测试getdtype函数在提供了数组A时是否返回其dtype类型np.int8
        assert_equal(sputils.getdtype(None, a=A), np.int8)

        # 测试getdtype函数在输入dtype为"O"时是否引发ValueError异常，并且异常信息匹配指定的字符串模式
        with assert_raises(
            ValueError,
            match="scipy.sparse does not support dtype object. .*",
        ):
            sputils.getdtype("O")

        # 测试getdtype函数在默认dtype为np.float16时是否引发ValueError异常，并且异常信息匹配指定的字符串模式
        with assert_raises(
            ValueError,
            match="scipy.sparse does not support dtype float16. .*",
        ):
            sputils.getdtype(None, default=np.float16)

    # 定义测试函数test_isscalarlike，用于测试判断对象是否类似标量的功能
    def test_isscalarlike(self):
        # 测试isscalarlike函数对标量数据类型如浮点数、整数、复数、NumPy数组和字符串的判断是否正确
        assert_equal(sputils.isscalarlike(3.0), True)
        assert_equal(sputils.isscalarlike(-4), True)
        assert_equal(sputils.isscalarlike(2.5), True)
        assert_equal(sputils.isscalarlike(1 + 3j), True)
        assert_equal(sputils.isscalarlike(np.array(3)), True)
        assert_equal(sputils.isscalarlike("16"), True)

        # 测试isscalarlike函数对非标量数据类型如NumPy数组、列表和元组的判断是否正确
        assert_equal(sputils.isscalarlike(np.array([3])), False)
        assert_equal(sputils.isscalarlike([[3]]), False)
        assert_equal(sputils.isscalarlike((1,)), False)
        assert_equal(sputils.isscalarlike((1, 2)), False)

    # 定义测试函数test_isintlike，用于测试判断对象是否类似整数的功能
    def test_isintlike(self):
        # 测试isintlike函数对整数、NumPy数组的整数类型的判断是否正确
        assert_equal(sputils.isintlike(-4), True)
        assert_equal(sputils.isintlike(np.array(3)), True)
        assert_equal(sputils.isintlike(np.array([3])), False)

        # 测试isintlike函数在输入为浮点数时是否引发ValueError异常，并且异常信息匹配指定的字符串模式
        with assert_raises(
            ValueError,
            match="Inexact indices into sparse matrices are not allowed"
        ):
            sputils.isintlike(3.0)

        # 测试isintlike函数对浮点数、复数、元组和多元组的判断是否正确
        assert_equal(sputils.isintlike(2.5), False)
        assert_equal(sputils.isintlike(1 + 3j), False)
        assert_equal(sputils.isintlike((1,)), False)
        assert_equal(sputils.isintlike((1, 2)), False)
    # 定义测试方法 test_isshape，用于测试 sputils.isshape 函数的不同参数组合
    def test_isshape(self):
        # 断言 sputils.isshape((1, 2)) 返回 True
        assert_equal(sputils.isshape((1, 2)), True)
        # 断言 sputils.isshape((5, 2)) 返回 True
        assert_equal(sputils.isshape((5, 2)), True)

        # 断言 sputils.isshape((1.5, 2)) 返回 False
        assert_equal(sputils.isshape((1.5, 2)), False)
        # 断言 sputils.isshape((2, 2, 2)) 返回 False
        assert_equal(sputils.isshape((2, 2, 2)), False)
        # 断言 sputils.isshape(([2], 2)) 返回 False
        assert_equal(sputils.isshape(([2], 2)), False)
        # 断言 sputils.isshape((-1, 2), nonneg=False) 返回 True
        assert_equal(sputils.isshape((-1, 2), nonneg=False), True)
        # 断言 sputils.isshape((2, -1), nonneg=False) 返回 True
        assert_equal(sputils.isshape((2, -1), nonneg=False), True)
        # 断言 sputils.isshape((-1, 2), nonneg=True) 返回 False
        assert_equal(sputils.isshape((-1, 2), nonneg=True), False)
        # 断言 sputils.isshape((2, -1), nonneg=True) 返回 False
        assert_equal(sputils.isshape((2, -1), nonneg=True), False)

        # 断言 sputils.isshape((1.5, 2), allow_1d=True) 返回 False
        assert_equal(sputils.isshape((1.5, 2), allow_1d=True), False)
        # 断言 sputils.isshape(([2], 2), allow_1d=True) 返回 False
        assert_equal(sputils.isshape(([2], 2), allow_1d=True), False)
        # 断言 sputils.isshape((2, 2, -2), nonneg=True, allow_1d=True) 返回 False
        assert_equal(sputils.isshape((2, 2, -2), nonneg=True, allow_1d=True), False)
        # 断言 sputils.isshape((2,), allow_1d=True) 返回 True
        assert_equal(sputils.isshape((2,), allow_1d=True), True)
        # 断言 sputils.isshape((2, 2,), allow_1d=True) 返回 True
        assert_equal(sputils.isshape((2, 2,), allow_1d=True), True)
        # 断言 sputils.isshape((2, 2, 2), allow_1d=True) 返回 False
        assert_equal(sputils.isshape((2, 2, 2), allow_1d=True), False)

    # 定义测试方法 test_issequence，用于测试 sputils.issequence 函数的不同参数组合
    def test_issequence(self):
        # 断言 sputils.issequence((1,)) 返回 True
        assert_equal(sputils.issequence((1,)), True)
        # 断言 sputils.issequence((1, 2, 3)) 返回 True
        assert_equal(sputils.issequence((1, 2, 3)), True)
        # 断言 sputils.issequence([1]) 返回 True
        assert_equal(sputils.issequence([1]), True)
        # 断言 sputils.issequence([1, 2, 3]) 返回 True
        assert_equal(sputils.issequence([1, 2, 3]), True)
        # 断言 sputils.issequence(np.array([1, 2, 3])) 返回 True
        assert_equal(sputils.issequence(np.array([1, 2, 3])), True)

        # 断言 sputils.issequence(np.array([[1], [2], [3]])) 返回 False
        assert_equal(sputils.issequence(np.array([[1], [2], [3]])), False)
        # 断言 sputils.issequence(3) 返回 False
        assert_equal(sputils.issequence(3), False)

    # 定义测试方法 test_ismatrix，用于测试 sputils.ismatrix 函数的不同参数组合
    def test_ismatrix(self):
        # 断言 sputils.ismatrix(((),)) 返回 True
        assert_equal(sputils.ismatrix(((),)), True)
        # 断言 sputils.ismatrix([[1], [2]]) 返回 True
        assert_equal(sputils.ismatrix([[1], [2]]), True)
        # 断言 sputils.ismatrix(np.arange(3)[None]) 返回 True
        assert_equal(sputils.ismatrix(np.arange(3)[None]), True)

        # 断言 sputils.ismatrix([1, 2]) 返回 False
        assert_equal(sputils.ismatrix([1, 2]), False)
        # 断言 sputils.ismatrix(np.arange(3)) 返回 False
        assert_equal(sputils.ismatrix(np.arange(3)), False)
        # 断言 sputils.ismatrix([[[1]]]) 返回 False
        assert_equal(sputils.ismatrix([[[1]]]), False)
        # 断言 sputils.ismatrix(3) 返回 False
        assert_equal(sputils.ismatrix(3), False)

    # 定义测试方法 test_isdense，用于测试 sputils.isdense 函数的不同参数组合
    def test_isdense(self):
        # 断言 sputils.isdense(np.array([1])) 返回 True
        assert_equal(sputils.isdense(np.array([1])), True)
        # 断言 sputils.isdense(matrix([1])) 返回 True
        assert_equal(sputils.isdense(matrix([1])), True)

    # 定义测试方法 test_validateaxis，用于测试 sputils.validateaxis 函数的不同参数组合
    def test_validateaxis(self):
        # 断言调用 sputils.validateaxis((0, 1)) 会引发 TypeError 异常
        assert_raises(TypeError, sputils.validateaxis, (0, 1))
        # 断言调用 sputils.validateaxis(1.5) 会引发 TypeError 异常
        assert_raises(TypeError, sputils.validateaxis, 1.5)
        # 断言调用 sputils.validateaxis(3) 会引发 ValueError 异常
        assert_raises(ValueError, sputils.validateaxis, 3)

        # 对于 axis 参数为 (-2, -1, 0, 1, None)，验证调用 sputils.validateaxis 不会引发异常
        for axis in (-2, -1, 0, 1, None):
            sputils.validateaxis(axis)
    # 定义一个测试函数，用于测试获取索引数据类型的函数
    def test_get_index_dtype(self):
        # 计算 np.int32 的最大值，然后转换为 np.int64 类型
        imax = np.int64(np.iinfo(np.int32).max)
        # 计算比 imax 大 1 的值，用于测试超出范围的情况
        too_big = imax + 1

        # 创建两个元素全为 1 的 uint32 类型的数组 a1 和 a2
        a1 = np.ones(90, dtype='uint32')
        a2 = np.ones(90, dtype='uint32')
        # 使用 get_index_dtype 函数检查返回的数据类型是否为 int32
        assert_equal(
            np.dtype(sputils.get_index_dtype((a1, a2), check_contents=True)),
            np.dtype('int32')
        )

        # 修改 a1 的最后一个元素为 imax，检查返回的数据类型是否为 int32
        a1[-1] = imax
        assert_equal(
            np.dtype(sputils.get_index_dtype((a1, a2), check_contents=True)),
            np.dtype('int32')
        )

        # 修改 a1 的最后一个元素为 too_big，检查返回的数据类型是否为 int64
        a1[-1] = too_big
        assert_equal(
            np.dtype(sputils.get_index_dtype((a1, a2), check_contents=True)),
            np.dtype('int64')
        )

        # 创建两个元素全为 1 的 uint32 类型的数组 a1 和 a2
        a1 = np.ones(89, dtype='uint32')
        a2 = np.ones(89, dtype='uint32')
        # 使用 get_index_dtype 函数检查返回的数据类型是否为 int64，没有指定 check_contents
        assert_equal(
            np.dtype(sputils.get_index_dtype((a1, a2))),
            np.dtype('int64')
        )

        # 创建两个元素全为 1 的 uint32 类型的数组 a1 和 a2
        a1 = np.ones(12, dtype='uint32')
        a2 = np.ones(12, dtype='uint32')
        # 使用 get_index_dtype 函数，指定 maxval 参数为 too_big，检查返回的数据类型是否为 int64
        assert_equal(
            np.dtype(sputils.get_index_dtype(
                (a1, a2), maxval=too_big, check_contents=True
            )),
            np.dtype('int64')
        )

        # 修改 a1 的最后一个元素为 too_big，检查返回的数据类型是否为 int64
        a1[-1] = too_big
        assert_equal(
            np.dtype(sputils.get_index_dtype((a1, a2), maxval=too_big)),
            np.dtype('int64')
        )

    # 定义一个测试函数，用于测试检查形状溢出的函数
    def test_check_shape_overflow(self):
        # 调用 check_shape 函数，传入 [(10, -1)] 和 (65535, 131070) 作为参数
        new_shape = sputils.check_shape([(10, -1)], (65535, 131070))
        # 检查返回的 new_shape 是否等于 (10, 858967245)
        assert_equal(new_shape, (10, 858967245))

    # 定义一个测试函数，用于测试创建矩阵的函数
    def test_matrix(self):
        # 创建一个列表 a 和对应的 numpy 数组 b
        a = [[1, 2, 3]]
        b = np.array(a)

        # 检查 sputils.matrix 函数返回的对象是否是 np.matrix 类型
        assert isinstance(sputils.matrix(a), np.matrix)
        assert isinstance(sputils.matrix(b), np.matrix)

        # 将 b 转换为矩阵 c，并将其所有元素设置为 123，然后检查 b 是否与原始列表 a 相等
        c = sputils.matrix(b)
        c[:, :] = 123
        assert_equal(b, a)

        # 将 b 转换为矩阵 c（不进行复制），将其所有元素设置为 123，然后检查 b 是否与预期相等
        c = sputils.matrix(b, copy=False)
        c[:, :] = 123
        assert_equal(b, [[123, 123, 123]])

    # 定义一个测试函数，用于测试创建矩阵的函数
    def test_asmatrix(self):
        # 创建一个列表 a 和对应的 numpy 数组 b
        a = [[1, 2, 3]]
        b = np.array(a)

        # 检查 sputils.asmatrix 函数返回的对象是否是 np.matrix 类型
        assert isinstance(sputils.asmatrix(a), np.matrix)
        assert isinstance(sputils.asmatrix(b), np.matrix)

        # 将 b 转换为矩阵 c，并将其所有元素设置为 123，然后检查 b 是否与预期相等
        c = sputils.asmatrix(b)
        c[:, :] = 123
        assert_equal(b, [[123, 123, 123]])
```
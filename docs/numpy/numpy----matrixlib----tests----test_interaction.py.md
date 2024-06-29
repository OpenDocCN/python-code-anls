# `.\numpy\numpy\matrixlib\tests\test_interaction.py`

```py
"""
Tests of interaction of matrix with other parts of numpy.

Note that tests with MaskedArray and linalg are done in separate files.
"""
import pytest  # 导入 pytest 库

import textwrap  # 导入 textwrap 模块，用于格式化文本
import warnings  # 导入 warnings 模块，处理警告信息

import numpy as np  # 导入 NumPy 库，并用 np 别名表示
from numpy.testing import (assert_, assert_equal, assert_raises,
                           assert_raises_regex, assert_array_equal,
                           assert_almost_equal, assert_array_almost_equal)

def test_fancy_indexing():
    # matrix 类型会影响数组的形状。虽然这很奇怪（不使用 getitem 方法，也没有 setitem 方法，也不了解高级索引），
    # 但这里测试的是 gh-3110
    # 2018-04-29: 从 core.tests.test_index 移动到这里。
    m = np.matrix([[1, 2], [3, 4]])

    assert_(isinstance(m[[0, 1, 0], :], np.matrix))

    # gh-3110。注意目前的转置，因为矩阵不正确支持高级索引的维度固定。
    x = np.asmatrix(np.arange(50).reshape(5, 10))
    assert_equal(x[:2, np.array(-1)], x[:2, -1].T)


def test_polynomial_mapdomain():
    # 测试多项式保留矩阵子类型。
    # 2018-04-29: 从 polynomial.tests.polyutils 移动到这里。
    dom1 = [0, 4]
    dom2 = [1, 3]
    x = np.matrix([dom1, dom1])
    res = np.polynomial.polyutils.mapdomain(x, dom1, dom2)
    assert_(isinstance(res, np.matrix))


def test_sort_matrix_none():
    # 2018-04-29: 从 core.tests.test_multiarray 移动到这里。
    a = np.matrix([[2, 1, 0]])
    actual = np.sort(a, axis=None)
    expected = np.matrix([[0, 1, 2]])
    assert_equal(actual, expected)
    assert_(type(expected) is np.matrix)


def test_partition_matrix_none():
    # gh-4301
    # 2018-04-29: 从 core.tests.test_multiarray 移动到这里。
    a = np.matrix([[2, 1, 0]])
    actual = np.partition(a, 1, axis=None)
    expected = np.matrix([[0, 1, 2]])
    assert_equal(actual, expected)
    assert_(type(expected) is np.matrix)


def test_dot_scalar_and_matrix_of_objects():
    # Ticket #2469
    # 2018-04-29: 从 core.tests.test_multiarray 移动到这里。
    arr = np.matrix([1, 2], dtype=object)
    desired = np.matrix([[3, 6]], dtype=object)
    assert_equal(np.dot(arr, 3), desired)
    assert_equal(np.dot(3, arr), desired)


def test_inner_scalar_and_matrix():
    # 2018-04-29: 从 core.tests.test_multiarray 移动到这里。
    for dt in np.typecodes['AllInteger'] + np.typecodes['AllFloat'] + '?':
        sca = np.array(3, dtype=dt)[()]
        arr = np.matrix([[1, 2], [3, 4]], dtype=dt)
        desired = np.matrix([[3, 6], [9, 12]], dtype=dt)
        assert_equal(np.inner(arr, sca), desired)
        assert_equal(np.inner(sca, arr), desired)


def test_inner_scalar_and_matrix_of_objects():
    # Ticket #4482
    # 2018-04-29: 从 core.tests.test_multiarray 移动到这里。
    arr = np.matrix([1, 2], dtype=object)
    desired = np.matrix([[3, 6]], dtype=object)
    assert_equal(np.inner(arr, 3), desired)
    assert_equal(np.inner(3, arr), desired)


def test_iter_allocate_output_subtype():
    pass  # 该测试函数目前没有实现，保留为占位符
    # 确保优先使用具有优先级的子类型
    # 2018-04-29: 从core.tests.test_nditer移动到此处，考虑矩阵特定形状的测试。

    # 创建一个矩阵和一个 ndarray
    a = np.matrix([[1, 2], [3, 4]])
    b = np.arange(4).reshape(2, 2).T
    
    # 使用 np.nditer 迭代器处理 a, b，并且创建一个写入操作数
    i = np.nditer([a, b, None], [],
                  [['readonly'], ['readonly'], ['writeonly', 'allocate']])
    
    # 断言操作数的类型是 np.matrix
    assert_(type(i.operands[2]) is np.matrix)
    
    # 断言操作数的类型不是 np.ndarray
    assert_(type(i.operands[2]) is not np.ndarray)
    
    # 断言操作数的形状是 (2, 2)
    assert_equal(i.operands[2].shape, (2, 2))

    # 对于矩阵，总是希望数据是二维的
    b = np.arange(4).reshape(1, 2, 2)
    
    # 断言在创建 np.nditer 迭代器时会引发 RuntimeError
    assert_raises(RuntimeError, np.nditer, [a, b, None], [],
                  [['readonly'], ['readonly'], ['writeonly', 'allocate']])
    
    # 如果禁用子类型，结果仍然可以工作
    i = np.nditer([a, b, None], [],
                  [['readonly'], ['readonly'],
                   ['writeonly', 'allocate', 'no_subtype']])
    
    # 断言操作数的类型是 np.ndarray
    assert_(type(i.operands[2]) is np.ndarray)
    
    # 断言操作数的类型不是 np.matrix
    assert_(type(i.operands[2]) is not np.matrix)
    
    # 断言操作数的形状是 (1, 2, 2)
    assert_equal(i.operands[2].shape, (1, 2, 2))
# 2018-04-29: moved here from core.tests.test_numeric
def like_function():
    # 创建一个 2x2 的 numpy 矩阵
    a = np.matrix([[1, 2], [3, 4]])
    
    # 遍历 np.zeros_like, np.ones_like, np.empty_like 这三个函数
    for like_function in np.zeros_like, np.ones_like, np.empty_like:
        # 使用当前遍历到的函数 like_function，对矩阵 a 进行操作并返回结果 b
        b = like_function(a)
        # 断言 b 的类型为 np.matrix
        assert_(type(b) is np.matrix)

        # 使用当前遍历到的函数 like_function，以及 subok=False 参数对矩阵 a 进行操作并返回结果 c
        c = like_function(a, subok=False)
        # 断言 c 的类型不是 np.matrix
        assert_(type(c) is not np.matrix)


# 2018-04-29: copied here from core.tests.test_api
def test_array_astype():
    # 创建一个 dtype='f4' 的 2x3 numpy 矩阵 a
    a = np.matrix([[0, 1, 2], [3, 4, 5]], dtype='f4')
    
    # 使用 subok=True 和 copy=False 对矩阵 a 进行数据类型转换，结果赋给 b
    b = a.astype('f4', subok=True, copy=False)
    # 断言 a 和 b 是同一个对象
    assert_(a is b)

    # 使用默认参数 subok=True 对矩阵 a 进行数据类型转换成 'i4'，结果赋给 b
    b = a.astype('i4', copy=False)
    # 断言 a 和 b 的值相等
    assert_equal(a, b)
    # 断言 b 的类型是 np.matrix
    assert_equal(type(b), np.matrix)

    # 使用 subok=False 对矩阵 a 进行数据类型转换成 'f4'，结果赋给 b
    b = a.astype('f4', subok=False, copy=False)
    # 断言 a 和 b 的值相等
    assert_equal(a, b)
    # 断言 a 和 b 不是同一个对象
    assert_(not (a is b))
    # 断言 b 的类型不是 np.matrix
    assert_(type(b) is not np.matrix)


# 2018-04-29: copied here from core.tests.test_shape_base
def test_stack():
    # 创建一个 2x2 的 numpy 矩阵 m
    m = np.matrix([[1, 2], [3, 4]])
    # 断言在尝试堆叠两个矩阵 m, m 时会引发 ValueError，提示形状过大无法作为矩阵处理
    assert_raises_regex(ValueError, 'shape too large to be a matrix',
                        np.stack, [m, m])


# Tickets #2469 and #4482
# 2018-04-29: moved here from core.tests.test_ufunc
def test_object_scalar_multiply():
    # 创建一个包含对象类型的 numpy 矩阵 arr
    arr = np.matrix([1, 2], dtype=object)
    # 创建一个预期的结果矩阵 desired
    desired = np.matrix([[3, 6]], dtype=object)
    # 断言 np.multiply(arr, 3) 的结果与预期的 desired 相等
    assert_equal(np.multiply(arr, 3), desired)
    # 断言 np.multiply(3, arr) 的结果与预期的 desired 相等
    assert_equal(np.multiply(3, arr), desired)


# Check that it works and that type and
# shape are preserved
# 2018-04-29: moved here from core.tests.test_nanfunctions
def test_nanfunctions_matrices():
    # 创建一个 3x3 的单位矩阵 mat
    mat = np.matrix(np.eye(3))
    
    # 遍历 [np.nanmin, np.nanmax] 这两个函数
    for f in [np.nanmin, np.nanmax]:
        # 对矩阵 mat 沿 axis=0 方向应用函数 f，结果赋给 res
        res = f(mat, axis=0)
        # 断言 res 的类型是 np.matrix
        assert_(isinstance(res, np.matrix))
        # 断言 res 的形状为 (1, 3)
        assert_(res.shape == (1, 3))
        
        # 对矩阵 mat 沿 axis=1 方向应用函数 f，结果赋给 res
        res = f(mat, axis=1)
        # 断言 res 的类型是 np.matrix
        assert_(isinstance(res, np.matrix))
        # 断言 res 的形状为 (3, 1)
        assert_(res.shape == (3, 1))
        
        # 对矩阵 mat 应用函数 f，结果赋给 res
        res = f(mat)
        # 断言 res 是标量
        assert_(np.isscalar(res))
    
    # 检查处理包含 NaN 的行是否正常进行子类化 (#4628)
    mat[1] = np.nan
    # 对于每个函数 `np.nanmin` 和 `np.nanmax` 进行循环处理
    for f in [np.nanmin, np.nanmax]:
        # 捕获警告并记录到 `w` 中
        with warnings.catch_warnings(record=True) as w:
            # 设置警告过滤器为总是显示
            warnings.simplefilter('always')
            # 对矩阵 `mat` 沿着第 0 轴应用函数 `f`
            res = f(mat, axis=0)
            # 断言返回结果是 `np.matrix` 类型
            assert_(isinstance(res, np.matrix))
            # 断言结果中不存在 NaN 值
            assert_(not np.any(np.isnan(res)))
            # 断言警告的数量为 0
            assert_(len(w) == 0)

        # 再次捕获警告并记录到 `w` 中
        with warnings.catch_warnings(record=True) as w:
            # 设置警告过滤器为总是显示
            warnings.simplefilter('always')
            # 对矩阵 `mat` 沿着第 1 轴应用函数 `f`
            res = f(mat, axis=1)
            # 断言返回结果是 `np.matrix` 类型
            assert_(isinstance(res, np.matrix))
            # 断言结果中特定位置存在 NaN 值，其他位置不存在 NaN 值
            assert_(np.isnan(res[1, 0]) and not np.isnan(res[0, 0])
                    and not np.isnan(res[2, 0]))
            # 断言警告的数量为 1，同时验证警告内容为 RuntimeWarning
            assert_(len(w) == 1, 'no warning raised')
            assert_(issubclass(w[0].category, RuntimeWarning))

        # 再次捕获警告并记录到 `w` 中
        with warnings.catch_warnings(record=True) as w:
            # 设置警告过滤器为总是显示
            warnings.simplefilter('always')
            # 对矩阵 `mat` 应用函数 `f`
            res = f(mat)
            # 断言返回结果是标量
            assert_(np.isscalar(res))
            # 断言结果不等于 NaN
            assert_(res != np.nan)
            # 断言警告的数量为 0
            assert_(len(w) == 0)
def test_nanfunctions_matrices_general():
    # 检查函数是否正常工作，并且类型和形状得以保留
    # 2018-04-29: 从core.tests.test_nanfunctions移动到这里

    # 创建一个3x3的单位矩阵
    mat = np.matrix(np.eye(3))

    # 对于每个函数f，分别进行测试：np.nanargmin, np.nanargmax, np.nansum, np.nanprod,
    # np.nanmean, np.nanvar, np.nanstd
    for f in (np.nanargmin, np.nanargmax, np.nansum, np.nanprod,
              np.nanmean, np.nanvar, np.nanstd):
        # 在axis=0上应用函数f
        res = f(mat, axis=0)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (1, 3))

        # 在axis=1上应用函数f
        res = f(mat, axis=1)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (3, 1))

        # 在整个矩阵上应用函数f
        res = f(mat)
        assert_(np.isscalar(res))

    # 对于函数np.nancumsum和np.nancumprod，同样进行测试
    for f in np.nancumsum, np.nancumprod:
        res = f(mat, axis=0)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (3, 3))

        res = f(mat, axis=1)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (3, 3))

        res = f(mat)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (1, 3*3))


def test_average_matrix():
    # 2018-04-29: 从core.tests.test_function_base移动到这里

    # 创建一个5x5的随机矩阵
    y = np.matrix(np.random.rand(5, 5))

    # 检查平均值函数是否等价
    assert_array_equal(y.mean(0), np.average(y, 0))

    # 创建一个2x2的矩阵a和权重矩阵w
    a = np.matrix([[1, 2], [3, 4]])
    w = np.matrix([[1, 2], [3, 4]])

    # 使用权重计算a的平均值，按axis=0计算
    r = np.average(a, axis=0, weights=w)
    assert_equal(type(r), np.matrix)
    assert_equal(r, [[2.5, 10.0/3]])


def test_dot_matrix():
    # 确保矩阵与ndarray得到相同的结果
    # 2018-04-29: 从core.tests.test_function_base移动到这里

    # 创建一个线性空间数组x和y
    x = np.linspace(0, 5)
    y = np.linspace(-5, 0)

    # 将x和y转换为矩阵类型
    mx = np.matrix(x)
    my = np.matrix(y)

    # 计算x和y的点积
    r = np.dot(x, y)

    # 计算矩阵mx和my的点积
    mr = np.dot(mx, my.T)
    assert_almost_equal(mr, r)


def test_ediff1d_matrix():
    # 2018-04-29: 从core.tests.test_arraysetops移动到这里

    # 确保np.ediff1d返回类型为矩阵
    assert(isinstance(np.ediff1d(np.matrix(1)), np.matrix))
    assert(isinstance(np.ediff1d(np.matrix(1), to_begin=1), np.matrix))


def test_apply_along_axis_matrix():
    # 此测试对矩阵特别恶毒，因为矩阵拒绝成为1维
    # 2018-04-29: 从core.tests.test_shape_base移动到这里

    # 定义一个函数，将每行翻倍
    def double(row):
        return row * 2

    # 创建一个2x2的矩阵m和预期的结果矩阵expected
    m = np.matrix([[0, 1], [2, 3]])
    expected = np.matrix([[0, 2], [4, 6]])

    # 沿着axis=0应用函数double
    result = np.apply_along_axis(double, 0, m)
    assert_(isinstance(result, np.matrix))
    assert_array_equal(result, expected)

    # 沿着axis=1应用函数double
    result = np.apply_along_axis(double, 1, m)
    assert_(isinstance(result, np.matrix))
    assert_array_equal(result, expected)


def test_kron_matrix():
    # 2018-04-29: 从core.tests.test_shape_base移动到这里

    # 创建一个2x2的全1矩阵a，并将其转换为矩阵类型m
    a = np.ones([2, 2])
    m = np.asmatrix(a)

    # 检查np.kron函数返回类型的一致性
    assert_equal(type(np.kron(a, a)), np.ndarray)
    assert_equal(type(np.kron(m, m)), np.matrix)
    assert_equal(type(np.kron(a, m)), np.matrix)
    assert_equal(type(np.kron(m, a)), np.matrix)


class TestConcatenatorMatrix:
    # 2018-04-29: 从core.tests.test_index_tricks移动到这里
    # 定义一个测试方法，用于测试 np.r_ 方法的不同用法和结果
    def test_matrix(self):
        # 创建两个简单的数组
        a = [1, 2]
        b = [3, 4]

        # 使用 np.r_ 方法按行连接数组 a 和 b，生成一个行向量
        ab_r = np.r_['r', a, b]
        # 使用 np.r_ 方法按列连接数组 a 和 b，生成一个列向量
        ab_c = np.r_['c', a, b]

        # 断言 ab_r 和 ab_c 的类型为 np.matrix
        assert_equal(type(ab_r), np.matrix)
        assert_equal(type(ab_c), np.matrix)

        # 断言生成的行向量 ab_r 转换为数组后的值为 [[1, 2, 3, 4]]
        assert_equal(np.array(ab_r), [[1, 2, 3, 4]])
        # 断言生成的列向量 ab_c 转换为数组后的值为 [[1], [2], [3], [4]]
        assert_equal(np.array(ab_c), [[1], [2], [3], [4]])

        # 测试使用不支持的连接类型 'rc' 时是否会引发 ValueError 异常
        assert_raises(ValueError, lambda: np.r_['rc', a, b])

    # 定义一个测试方法，用于测试 np.r_ 方法在包含标量的情况下的行连接
    def test_matrix_scalar(self):
        # 使用 np.r_ 方法将包含标量的数组 [1, 2] 和标量 3 进行行连接
        r = np.r_['r', [1, 2], 3]
        # 断言连接结果的类型为 np.matrix
        assert_equal(type(r), np.matrix)
        # 断言连接结果转换为数组后的值为 [[1, 2, 3]]
        assert_equal(np.array(r), [[1, 2, 3]])

    # 定义一个测试方法，用于测试 np.r_ 方法在使用字符串建造器时的结果
    def test_matrix_builder(self):
        # 创建四个简单的数组
        a = np.array([1])
        b = np.array([2])
        c = np.array([3])
        d = np.array([4])

        # 使用字符串建造器 'a, b; c, d' 创建一个 2x2 的矩阵
        actual = np.r_['a, b; c, d']
        # 使用 np.bmat 方法生成一个与 actual 相同的期望矩阵
        expected = np.bmat([[a, b], [c, d]])

        # 断言 actual 和 expected 的值相等
        assert_equal(actual, expected)
        # 断言 actual 和 expected 的类型相同
        assert_equal(type(actual), type(expected))
# 2018-04-29: 从 testing.tests.test_utils 移动到此处。
def test_array_equal_error_message_matrix():
    # 使用 pytest.raises 来捕获 AssertionError 异常，存储异常信息到 exc_info
    with pytest.raises(AssertionError) as exc_info:
        # 断言 np.array([1, 2]) 与 np.matrix([1, 2]) 相等，预期会引发 AssertionError
        assert_equal(np.array([1, 2]), np.matrix([1, 2]))
    # 将异常信息转换为字符串
    msg = str(exc_info.value)
    # 预期的错误信息参考，去除首尾空白和缩进
    msg_reference = textwrap.dedent("""\
    
    Arrays are not equal

    (shapes (2,), (1, 2) mismatch)
     ACTUAL: array([1, 2])
     DESIRED: matrix([[1, 2]])""")
    # 断言捕获的异常信息与预期的错误信息相等
    assert_equal(msg, msg_reference)


def test_array_almost_equal_matrix():
    # 矩阵切片保持二维，而数组不一定。参见 gh-8452。
    # 2018-04-29: 从 testing.tests.test_utils 移动到此处。
    # 创建几个不同的 np.matrix 示例
    m1 = np.matrix([[1., 2.]])
    m2 = np.matrix([[1., np.nan]])
    m3 = np.matrix([[1., -np.inf]])
    m4 = np.matrix([[np.nan, np.inf]])
    m5 = np.matrix([[1., 2.], [np.nan, np.inf]])
    # 对于 assert_array_almost_equal 和 assert_almost_equal 两个函数进行迭代
    for assert_func in assert_array_almost_equal, assert_almost_equal:
        # 对每个矩阵 m 进行测试
        for m in m1, m2, m3, m4, m5:
            # 断言矩阵 m 与自身的近似相等性
            assert_func(m, m)
            # 将矩阵 m 转换为数组 a，并断言数组 a 与矩阵 m 的近似相等性
            a = np.array(m)
            assert_func(a, m)
            # 断言矩阵 m 与数组 a 的近似相等性
            assert_func(m, a)
```
# `.\numpy\numpy\ma\testutils.py`

```py
# 导入Python标准库中的operator模块，用于操作符和函数的标准集合
import operator

# 导入NumPy库，约定使用别名np，提供对多维数组和矩阵的支持
import numpy as np

# 从NumPy中导入ndarray类
from numpy import ndarray

# 导入NumPy核心中的umath模块，提供基本的数学函数
import numpy._core.umath as umath

# 导入NumPy的测试模块
import numpy.testing

# 从NumPy的测试模块中导入指定的函数和类
from numpy.testing import (
    assert_, assert_allclose, assert_array_almost_equal_nulp,
    assert_raises, build_err_msg
    )

# 从当前包的core模块中导入指定的函数和类
from .core import mask_or, getmask, masked_array, nomask, masked, filled

# 定义一个变量，包含一些用于掩码数组和子类测试的杂项函数
__all__masked = [
    'almost', 'approx', 'assert_almost_equal', 'assert_array_almost_equal',
    'assert_array_approx_equal', 'assert_array_compare',
    'assert_array_equal', 'assert_array_less', 'assert_close',
    'assert_equal', 'assert_equal_records', 'assert_mask_equal',
    'assert_not_equal', 'fail_if_array_equal',
    ]

# 包含一些普通的测试函数，以避免破坏错误地包含它们的其他项目，例如SciPy
# 这些函数并非都用于掩码数组，但在此文件中无法区分
from unittest import TestCase

# 定义一个变量，包含一些从测试模块导入的函数和类
__some__from_testing = [
    'TestCase', 'assert_', 'assert_allclose', 'assert_array_almost_equal_nulp',
    'assert_raises'
    ]

# 将所有公开的函数和类组合为__all__变量，以便在包外可见
__all__ = __all__masked + __some__from_testing


# 定义一个函数approx，用于比较两个数组是否在给定的容差范围内相等
def approx(a, b, fill_value=True, rtol=1e-5, atol=1e-8):
    """
    如果a和b的所有元素在给定的容差范围内相等，则返回True。

    如果fill_value为True，则掩码值被视为相等；否则，掩码值被视为不等。
    相对误差rtol应为正数且 << 1.0。绝对误差atol适用于b中非常小或零的元素；它表示a必须多么小。
    """
    # 获取a和b的掩码并合并为一个掩码
    m = mask_or(getmask(a), getmask(b))
    # 获取a和b的实际数据（去除掩码后的数据）
    d1 = filled(a)
    d2 = filled(b)
    # 如果数据类型为对象（O），直接比较是否相等并展平
    if d1.dtype.char == "O" or d2.dtype.char == "O":
        return np.equal(d1, d2).ravel()
    # 将a和b的实际数据填充到掩码数组中，并转换为浮点数类型
    x = filled(
        masked_array(d1, copy=False, mask=m), fill_value
    ).astype(np.float64)
    y = filled(masked_array(d2, copy=False, mask=m), 1).astype(np.float64)
    # 计算x和y之间的相对误差和绝对误差，判断是否在容差范围内
    d = np.less_equal(umath.absolute(x - y), atol + rtol * umath.absolute(y))
    # 返回比较结果并展平
    return d.ravel()


# 定义一个函数almost，用于比较两个数组是否在给定的小数位数内相等
def almost(a, b, decimal=6, fill_value=True):
    """
    如果a和b在小数位数上相等，则返回True。

    如果fill_value为True，则掩码值被视为相等；否则，掩码值被视为不等。
    """
    # 获取a和b的掩码并合并为一个掩码
    m = mask_or(getmask(a), getmask(b))
    # 获取a和b的实际数据（去除掩码后的数据）
    d1 = filled(a)
    d2 = filled(b)
    # 如果数据类型为对象（O），直接比较是否相等并展平
    if d1.dtype.char == "O" or d2.dtype.char == "O":
        return np.equal(d1, d2).ravel()
    # 将a和b的实际数据填充到掩码数组中，并转换为浮点数类型
    x = filled(
        masked_array(d1, copy=False, mask=m), fill_value
    ).astype(np.float64)
    y = filled(masked_array(d2, copy=False, mask=m), 1).astype(np.float64)
    # 比较x和y在小数位数上的精度，判断是否相等
    d = np.around(np.abs(x - y), decimal) <= 10.0 ** (-decimal)
    # 返回比较结果并展平
    return d.ravel()


# 定义一个函数_assert_equal_on_sequences，用于在序列上断言相等
def _assert_equal_on_sequences(actual, desired, err_msg=''):
    """
    在序列上断言actual和desired是否相等。
    """
    Asserts the equality of two non-array sequences.

    """
    # 断言两个非数组序列的长度相等
    assert_equal(len(actual), len(desired), err_msg)
    # 遍历期望结果的长度，逐个断言实际结果和期望结果的每个元素相等
    for k in range(len(desired)):
        assert_equal(actual[k], desired[k], f'item={k!r}\n{err_msg}')
    # 返回
    return
# 检查两个记录是否相等的断言函数
def assert_equal_records(a, b):
    """
    Asserts that two records are equal.

    Pretty crude for now.

    """
    # 断言两个记录的数据类型相等
    assert_equal(a.dtype, b.dtype)
    # 遍历记录中的每个字段
    for f in a.dtype.names:
        # 获取字段 f 的值
        (af, bf) = (operator.getitem(a, f), operator.getitem(b, f))
        # 如果两个字段的值均不为 masked，则断言它们相等
        if not (af is masked) and not (bf is masked):
            assert_equal(operator.getitem(a, f), operator.getitem(b, f))
    return


# 断言两个项是否相等的通用函数
def assert_equal(actual, desired, err_msg=''):
    """
    Asserts that two items are equal.

    """
    # Case #1: 如果 desired 是字典 .....
    if isinstance(desired, dict):
        # 如果 actual 不是字典，抛出断言错误
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        # 断言 actual 和 desired 的长度相等
        assert_equal(len(actual), len(desired), err_msg)
        # 遍历 desired 字典的键值对
        for k, i in desired.items():
            # 如果键 k 不在 actual 中，抛出断言错误
            if k not in actual:
                raise AssertionError(f"{k} not in {actual}")
            # 递归调用 assert_equal 检查 actual[k] 和 desired[k] 是否相等
            assert_equal(actual[k], desired[k], f'key={k!r}\n{err_msg}')
        return
    # Case #2: 如果 desired 是列表或元组 .....
    if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
        # 调用 _assert_equal_on_sequences 函数检查序列是否相等
        return _assert_equal_on_sequences(actual, desired, err_msg='')
    # 如果 actual 和 desired 都不是 ndarray，则构建错误消息并断言它们相等
    if not (isinstance(actual, ndarray) or isinstance(desired, ndarray)):
        msg = build_err_msg([actual, desired], err_msg,)
        if not desired == actual:
            raise AssertionError(msg)
        return
    # Case #4: 如果 actual 或 desired 是 masked，抛出 ValueError
    if ((actual is masked) and not (desired is masked)) or \
            ((desired is masked) and not (actual is masked)):
        msg = build_err_msg([actual, desired],
                            err_msg, header='', names=('x', 'y'))
        raise ValueError(msg)
    # 将 actual 和 desired 转换为 ndarray
    actual = np.asanyarray(actual)
    desired = np.asanyarray(desired)
    # 获取 actual 和 desired 的数据类型
    (actual_dtype, desired_dtype) = (actual.dtype, desired.dtype)
    # 如果 actual 和 desired 都是字符串数组，则调用 _assert_equal_on_sequences 检查序列是否相等
    if actual_dtype.char == "S" and desired_dtype.char == "S":
        return _assert_equal_on_sequences(actual.tolist(),
                                          desired.tolist(),
                                          err_msg='')
    # 否则调用 assert_array_equal 函数断言 actual 和 desired 相等
    return assert_array_equal(actual, desired, err_msg)


# 如果两个项相等，则抛出断言错误的函数
def fail_if_equal(actual, desired, err_msg='',):
    """
    Raises an assertion error if two items are equal.

    """
    # 如果 desired 是字典 .....
    if isinstance(desired, dict):
        # 如果 actual 不是字典，抛出断言错误
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        # 调用 fail_if_equal 检查 actual 和 desired 的长度是否相等
        fail_if_equal(len(actual), len(desired), err_msg)
        # 遍历 desired 字典的键值对
        for k, i in desired.items():
            # 如果键 k 不在 actual 中，抛出断言错误
            if k not in actual:
                raise AssertionError(repr(k))
            # 递归调用 fail_if_equal 检查 actual[k] 和 desired[k] 是否不相等
            fail_if_equal(actual[k], desired[k], f'key={k!r}\n{err_msg}')
        return
    # 如果 desired 是列表或元组 .....
    if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
        # 调用 fail_if_equal 检查 actual 和 desired 的长度是否相等
        fail_if_equal(len(actual), len(desired), err_msg)
        # 遍历列表或元组的每个元素，递归调用 fail_if_equal 检查它们是否不相等
        for k in range(len(desired)):
            fail_if_equal(actual[k], desired[k], f'item={k!r}\n{err_msg}')
        return
    # 如果 actual 或 desired 是 ndarray，则调用 fail_if_array_equal 检查它们是否不相等
    if isinstance(actual, np.ndarray) or isinstance(desired, np.ndarray):
        return fail_if_array_equal(actual, desired, err_msg)
    # 调用 build_err_msg 函数，生成错误信息字符串
    msg = build_err_msg([actual, desired], err_msg)
    # 如果 desired 不等于 actual，抛出 AssertionError 异常，异常信息为 msg
    if not desired != actual:
        raise AssertionError(msg)
# 将 assert_not_equal 定义为 fail_if_equal 的别名
assert_not_equal = fail_if_equal


def assert_almost_equal(actual, desired, decimal=7, err_msg='', verbose=True):
    """
    Asserts that two items are almost equal.

    The test is equivalent to abs(desired-actual) < 0.5 * 10**(-decimal).

    """
    # 如果 actual 或 desired 是 NumPy 数组，则调用 assert_array_almost_equal 函数进行比较
    if isinstance(actual, np.ndarray) or isinstance(desired, np.ndarray):
        return assert_array_almost_equal(actual, desired, decimal=decimal,
                                         err_msg=err_msg, verbose=verbose)
    # 否则，构建错误消息
    msg = build_err_msg([actual, desired],
                        err_msg=err_msg, verbose=verbose)
    # 如果 abs(desired - actual) 不小于 10**(-decimal)，则抛出 AssertionError
    if not round(abs(desired - actual), decimal) == 0:
        raise AssertionError(msg)


# 将 assert_close 定义为 assert_almost_equal 的别名
assert_close = assert_almost_equal


def assert_array_compare(comparison, x, y, err_msg='', verbose=True, header='',
                         fill_value=True):
    """
    Asserts that comparison between two masked arrays is satisfied.

    The comparison is elementwise.

    """
    # 分配一个公共掩码并重新填充
    m = mask_or(getmask(x), getmask(y))
    # 创建掩码数组 x 和 y
    x = masked_array(x, copy=False, mask=m, keep_mask=False, subok=False)
    y = masked_array(y, copy=False, mask=m, keep_mask=False, subok=False)
    # 如果 x 是掩码数组而 y 不是，或者 y 是掩码数组而 x 不是，则抛出 ValueError
    if ((x is masked) and not (y is masked)) or \
            ((y is masked) and not (x is masked)):
        msg = build_err_msg([x, y], err_msg=err_msg, verbose=verbose,
                            header=header, names=('x', 'y'))
        raise ValueError(msg)
    # 现在对填充版本运行基本测试
    return np.testing.assert_array_compare(comparison,
                                           x.filled(fill_value),
                                           y.filled(fill_value),
                                           err_msg=err_msg,
                                           verbose=verbose, header=header)


def assert_array_equal(x, y, err_msg='', verbose=True):
    """
    Checks the elementwise equality of two masked arrays.

    """
    # 检查两个掩码数组的逐元素相等性
    assert_array_compare(operator.__eq__, x, y,
                         err_msg=err_msg, verbose=verbose,
                         header='Arrays are not equal')


def fail_if_array_equal(x, y, err_msg='', verbose=True):
    """
    Raises an assertion error if two masked arrays are not equal elementwise.

    """
    # 定义比较函数，如果 x 和 y 不全等，则返回 True
    def compare(x, y):
        return (not np.all(approx(x, y)))
    # 断言两个掩码数组 x 和 y 在逐元素比较时不全相等
    assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose,
                         header='Arrays are not equal')


def assert_array_approx_equal(x, y, decimal=6, err_msg='', verbose=True):
    """
    Checks the equality of two masked arrays, up to given number odecimals.

    The equality is checked elementwise.

    """
    # 定义比较函数，通过指定的小数位数检查两个掩码数组的近似相等性
    def compare(x, y):
        "Returns the result of the loose comparison between x and y)."
        return approx(x, y, rtol=10. ** -decimal)
    # 断言两个掩码数组 x 和 y 在近似比较时相等
    assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose,
                         header='Arrays are not almost equal')
# 检查两个带掩码的数组在给定小数位数下的几乎相等性
def assert_array_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
    """
    Checks the equality of two masked arrays, up to given number of decimals.

    The equality is checked elementwise.

    """
    # 定义比较函数，比较 x 和 y 的近似程度，精度为 decimal
    def compare(x, y):
        "Returns the result of the loose comparison between x and y)."
        return almost(x, y, decimal)
    # 调用通用的数组比较函数，比较 x 和 y，检查它们是否几乎相等
    assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose,
                         header='Arrays are not almost equal')


# 检查两个数组在元素级别上是否 x 小于 y
def assert_array_less(x, y, err_msg='', verbose=True):
    """
    Checks that x is smaller than y elementwise.

    """
    # 调用通用的数组比较函数，比较 x 和 y，验证 x 是否在元素级别上小于 y
    assert_array_compare(operator.__lt__, x, y,
                         err_msg=err_msg, verbose=verbose,
                         header='Arrays are not less-ordered')


# 断言两个掩码数组的相等性
def assert_mask_equal(m1, m2, err_msg=''):
    """
    Asserts the equality of two masks.

    """
    # 如果 m1 是 nomask，则断言 m2 也是 nomask
    if m1 is nomask:
        assert_(m2 is nomask)
    # 如果 m2 是 nomask，则断言 m1 也是 nomask
    if m2 is nomask:
        assert_(m1 is nomask)
    # 使用通用的数组相等性比较函数，断言 m1 和 m2 在掩码中是相等的
    assert_array_equal(m1, m2, err_msg=err_msg)
```
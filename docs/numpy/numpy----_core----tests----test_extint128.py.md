# `.\numpy\numpy\_core\tests\test_extint128.py`

```
# 引入必要的模块和库
import itertools  # 提供迭代工具的函数
import contextlib  # 上下文管理器的支持
import operator  # 操作符函数的模块
import pytest  # 测试框架

import numpy as np  # 数组和数值计算工具
import numpy._core._multiarray_tests as mt  # NumPy内部的多维数组测试函数

from numpy.testing import assert_raises, assert_equal  # 断言函数，用于测试

# 定义常量
INT64_MAX = np.iinfo(np.int64).max  # int64类型的最大值
INT64_MIN = np.iinfo(np.int64).min  # int64类型的最小值
INT64_MID = 2**32  # int64类型的中间值

# int128不是二进制补码，符号位是单独的
INT128_MAX = 2**128 - 1  # int128类型的最大值
INT128_MIN = -INT128_MAX  # int128类型的最小值
INT128_MID = 2**64  # int128类型的中间值

# 定义不同类型的整数值列表
INT64_VALUES = (
    [INT64_MIN + j for j in range(20)] +  # int64类型的最小值附近的20个值
    [INT64_MAX - j for j in range(20)] +  # int64类型的最大值附近的20个值
    [INT64_MID + j for j in range(-20, 20)] +  # int64类型的中间值附近的40个值
    [2*INT64_MID + j for j in range(-20, 20)] +  # 2倍int64类型的中间值附近的40个值
    [INT64_MID//2 + j for j in range(-20, 20)] +  # int64类型中间值的一半附近的40个值
    list(range(-70, 70))  # -70到69的整数值
)

INT128_VALUES = (
    [INT128_MIN + j for j in range(20)] +  # int128类型的最小值附近的20个值
    [INT128_MAX - j for j in range(20)] +  # int128类型的最大值附近的20个值
    [INT128_MID + j for j in range(-20, 20)] +  # int128类型的中间值附近的40个值
    [2*INT128_MID + j for j in range(-20, 20)] +  # 2倍int128类型的中间值附近的40个值
    [INT128_MID//2 + j for j in range(-20, 20)] +  # int128类型中间值的一半附近的40个值
    list(range(-70, 70)) +  # -70到69的整数值
    [False]  # 表示负零的特殊值
)

INT64_POS_VALUES = [x for x in INT64_VALUES if x > 0]  # int64类型的正整数值列表

@contextlib.contextmanager
def exc_iter(*args):
    """
    创建一个上下文管理器，用于迭代*args的笛卡尔积，如果发生异常，则添加当前迭代的信息。
    """

    value = [None]  # 存储当前迭代的值

    def iterate():
        for v in itertools.product(*args):
            value[0] = v
            yield v

    try:
        yield iterate()  # 返回迭代器对象
    except Exception:
        import traceback
        msg = "At: %r\n%s" % (repr(value[0]),
                              traceback.format_exc())
        raise AssertionError(msg)  # 抛出断言异常，带有追踪信息

def test_safe_binop():
    # 测试安全算术运算函数

    ops = [
        (operator.add, 1),  # 加法操作符和对应的操作码
        (operator.sub, 2),  # 减法操作符和对应的操作码
        (operator.mul, 3)   # 乘法操作符和对应的操作码
    ]

    with exc_iter(ops, INT64_VALUES, INT64_VALUES) as it:
        for xop, a, b in it:
            pyop, op = xop
            c = pyop(a, b)  # 执行Python操作

            if not (INT64_MIN <= c <= INT64_MAX):  # 如果结果超出int64类型范围
                assert_raises(OverflowError, mt.extint_safe_binop, a, b, op)  # 断言应该引发溢出错误
            else:
                d = mt.extint_safe_binop(a, b, op)  # 调用扩展整数安全算术函数
                if c != d:
                    # assert_equal 函数速度较慢
                    assert_equal(d, c)  # 断言结果应该相等

def test_to_128():
    with exc_iter(INT64_VALUES) as it:
        for a, in it:
            b = mt.extint_to_128(a)  # 调用转换为int128的函数
            if a != b:
                assert_equal(b, a)  # 断言结果应该相等

def test_to_64():
    with exc_iter(INT128_VALUES) as it:
        for a, in it:
            if not (INT64_MIN <= a <= INT64_MAX):
                assert_raises(OverflowError, mt.extint_to_64, a)  # 断言应该引发溢出错误
            else:
                b = mt.extint_to_64(a)  # 调用转换为int64的函数
                if a != b:
                    assert_equal(b, a)  # 断言结果应该相等

def test_mul_64_64():
    with exc_iter(INT64_VALUES, INT64_VALUES) as it:
        for a, b in it:
            c = a * b
            d = mt.extint_mul_64_64(a, b)  # 调用int64乘法函数
            if c != d:
                assert_equal(d, c)  # 断言结果应该相等

def test_add_128():
    # 使用异常迭代器处理 INT128_VALUES 和 INT128_VALUES，获取迭代器对象
    with exc_iter(INT128_VALUES, INT128_VALUES) as it:
        # 遍历迭代器，每次迭代获取 a 和 b
        for a, b in it:
            # 计算 a + b 的结果
            c = a + b
            # 检查结果是否在 INT128_MIN 和 INT128_MAX 范围内
            if not (INT128_MIN <= c <= INT128_MAX):
                # 如果结果超出范围，断言抛出 OverflowError 异常，确保函数 mt.extint_add_128 能正确处理溢出
                assert_raises(OverflowError, mt.extint_add_128, a, b)
            else:
                # 如果结果在范围内，调用 mt.extint_add_128 计算结果 d
                d = mt.extint_add_128(a, b)
                # 断言计算结果 d 等于预期结果 c
                if c != d:
                    assert_equal(d, c)
# 测试函数：对给定的 INT128_VALUES 迭代器进行测试
def test_sub_128():
    # 使用 exc_iter 上下文管理器，迭代 INT128_VALUES 中的元素对 (a, b)
    with exc_iter(INT128_VALUES, INT128_VALUES) as it:
        for a, b in it:
            # 计算 a - b
            c = a - b
            # 检查是否 c 在 INT128_MIN 和 INT128_MAX 范围内
            if not (INT128_MIN <= c <= INT128_MAX):
                # 如果不在范围内，预期会引发 OverflowError 异常，调用 mt.extint_sub_128(a, b)
                assert_raises(OverflowError, mt.extint_sub_128, a, b)
            else:
                # 如果在范围内，调用 mt.extint_sub_128(a, b) 计算结果 d
                d = mt.extint_sub_128(a, b)
                # 检查计算结果是否与预期结果 c 相等
                if c != d:
                    assert_equal(d, c)


# 测试函数：对给定的 INT128_VALUES 迭代器进行测试
def test_neg_128():
    # 使用 exc_iter 上下文管理器，迭代 INT128_VALUES 中的元素 a
    with exc_iter(INT128_VALUES) as it:
        for a, in it:
            # 计算 -a
            b = -a
            # 调用 mt.extint_neg_128(a) 计算结果 c
            c = mt.extint_neg_128(a)
            # 检查计算结果是否与预期结果 b 相等
            if b != c:
                assert_equal(c, b)


# 测试函数：对给定的 INT128_VALUES 迭代器进行测试
def test_shl_128():
    # 使用 exc_iter 上下文管理器，迭代 INT128_VALUES 中的元素 a
    with exc_iter(INT128_VALUES) as it:
        for a, in it:
            # 根据 a 的正负情况计算左移结果 b
            if a < 0:
                b = -(((-a) << 1) & (2**128-1))
            else:
                b = (a << 1) & (2**128-1)
            # 调用 mt.extint_shl_128(a) 计算结果 c
            c = mt.extint_shl_128(a)
            # 检查计算结果是否与预期结果 b 相等
            if b != c:
                assert_equal(c, b)


# 测试函数：对给定的 INT128_VALUES 迭代器进行测试
def test_shr_128():
    # 使用 exc_iter 上下文管理器，迭代 INT128_VALUES 中的元素 a
    with exc_iter(INT128_VALUES) as it:
        for a, in it:
            # 根据 a 的正负情况计算右移结果 b
            if a < 0:
                b = -((-a) >> 1)
            else:
                b = a >> 1
            # 调用 mt.extint_shr_128(a) 计算结果 c
            c = mt.extint_shr_128(a)
            # 检查计算结果是否与预期结果 b 相等
            if b != c:
                assert_equal(c, b)


# 测试函数：对给定的 INT128_VALUES 迭代器进行测试
def test_gt_128():
    # 使用 exc_iter 上下文管理器，迭代 INT128_VALUES 中的元素对 (a, b)
    with exc_iter(INT128_VALUES, INT128_VALUES) as it:
        for a, b in it:
            # 计算 a 是否大于 b，结果为布尔值 c
            c = a > b
            # 调用 mt.extint_gt_128(a, b) 计算结果 d
            d = mt.extint_gt_128(a, b)
            # 检查计算结果是否与预期结果 c 相等
            if c != d:
                assert_equal(d, c)


# 测试函数：对给定的 INT128_VALUES 和 INT64_POS_VALUES 迭代器进行测试
@pytest.mark.slow
def test_divmod_128_64():
    with exc_iter(INT128_VALUES, INT64_POS_VALUES) as it:
        for a, b in it:
            # 根据 a 和 b 计算 divmod 结果 c 和 cr
            if a >= 0:
                c, cr = divmod(a, b)
            else:
                c, cr = divmod(-a, b)
                c = -c
                cr = -cr

            # 调用 mt.extint_divmod_128_64(a, b) 计算结果 d, dr
            d, dr = mt.extint_divmod_128_64(a, b)

            # 检查计算结果是否符合预期
            if c != d or d != dr or b * d + dr != a:
                assert_equal(d, c)
                assert_equal(dr, cr)
                assert_equal(b * d + dr, a)


# 测试函数：对给定的 INT128_VALUES 和 INT64_POS_VALUES 迭代器进行测试
def test_floordiv_128_64():
    with exc_iter(INT128_VALUES, INT64_POS_VALUES) as it:
        for a, b in it:
            # 计算 a 除以 b 的整数除法结果 c
            c = a // b
            # 调用 mt.extint_floordiv_128_64(a, b) 计算结果 d
            d = mt.extint_floordiv_128_64(a, b)

            # 检查计算结果是否与预期结果 c 相等
            if c != d:
                assert_equal(d, c)


# 测试函数：对给定的 INT128_VALUES 和 INT64_POS_VALUES 迭代器进行测试
def test_ceildiv_128_64():
    with exc_iter(INT128_VALUES, INT64_POS_VALUES) as it:
        for a, b in it:
            # 计算 a 除以 b 的向上取整除法结果 c
            c = (a + b - 1) // b
            # 调用 mt.extint_ceildiv_128_64(a, b) 计算结果 d
            d = mt.extint_ceildiv_128_64(a, b)

            # 检查计算结果是否与预期结果 c 相等
            if c != d:
                assert_equal(d, c)
```
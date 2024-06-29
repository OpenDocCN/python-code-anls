# `.\numpy\numpy\lib\tests\test_mixins.py`

```
# 导入必要的模块
import numbers           # 导入 numbers 模块，用于处理数字相关操作
import operator          # 导入 operator 模块，提供了各种运算符的函数实现

import numpy as np       # 导入 NumPy 库，用于科学计算
from numpy.testing import assert_, assert_equal, assert_raises  # 导入 NumPy 测试模块中的断言函数

# NOTE: This class should be kept as an exact copy of the example from the
# docstring for NDArrayOperatorsMixin.

# 定义一个类 ArrayLike，继承自 NDArrayOperatorsMixin，用于模拟数组的行为
class ArrayLike(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, value):
        self.value = np.asarray(value)  # 将输入的值转换为 NumPy 数组并存储在 self.value 中

    # One might also consider adding the built-in list type to this
    # list, to support operations like np.add(array_like, list)
    # 定义 _HANDLED_TYPES 元组，指定支持的数据类型，包括 ndarray 和 numbers.Number
    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    # 实现 __array_ufunc__ 方法，处理 NumPy 的通用函数
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())  # 获取关键字参数中的 out 参数
        for x in inputs + out:
            # 只支持 _HANDLED_TYPES 中指定的数据类型或者 ArrayLike 类型的实例
            if not isinstance(x, self._HANDLED_TYPES + (ArrayLike,)):
                return NotImplemented

        # 对输入参数进行处理，如果是 ArrayLike 类型，则取其 value 属性作为操作对象
        inputs = tuple(x.value if isinstance(x, ArrayLike) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x.value if isinstance(x, ArrayLike) else x
                for x in out)
        # 调用 ufunc 的具体方法进行计算
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # 处理多返回值的情况，将每个返回值转换为 ArrayLike 类型
            return tuple(type(self)(x) for x in result)
        elif method == 'at':
            # 对于 'at' 方法，没有返回值
            return None
        else:
            # 处理单返回值的情况，将结果转换为 ArrayLike 类型
            return type(self)(result)

    # 重写 __repr__ 方法，返回对象的字符串表示形式
    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.value)


# 定义一个函数 wrap_array_like，用于将返回值转换为 ArrayLike 类型
def wrap_array_like(result):
    if type(result) is tuple:
        return tuple(ArrayLike(r) for r in result)
    else:
        return ArrayLike(result)


# 定义一个私有函数 _assert_equal_type_and_value，用于断言两个对象的类型和值是否相等
def _assert_equal_type_and_value(result, expected, err_msg=None):
    assert_equal(type(result), type(expected), err_msg=err_msg)  # 断言结果类型与期望类型相等
    if isinstance(result, tuple):
        assert_equal(len(result), len(expected), err_msg=err_msg)  # 断言元组长度相等
        for result_item, expected_item in zip(result, expected):
            _assert_equal_type_and_value(result_item, expected_item, err_msg)  # 递归比较元组内的每个元素
    else:
        assert_equal(result.value, expected.value, err_msg=err_msg)  # 断言值属性相等
        assert_equal(getattr(result.value, 'dtype', None),
                     getattr(expected.value, 'dtype', None), err_msg=err_msg)  # 断言 dtype 相等


# 定义一个列表 _ALL_BINARY_OPERATORS，包含常用的二元运算符函数
_ALL_BINARY_OPERATORS = [
    operator.lt,         # 小于运算符 <
    operator.le,         # 小于等于运算符 <=
    operator.eq,         # 等于运算符 ==
    operator.ne,         # 不等于运算符 !=
    operator.gt,         # 大于运算符 >
    operator.ge,         # 大于等于运算符 >=
    operator.add,        # 加法运算符 +
    operator.sub,        # 减法运算符 -
    operator.mul,        # 乘法运算符 *
    operator.truediv,    # 真除法运算符 /
    operator.floordiv,   # 地板除法运算符 //
    operator.mod,        # 取模运算符 %
    divmod,              # 返回商和余数的元组
    pow,                 # 幂运算符 **
    operator.lshift,     # 左移运算符 <<
    operator.rshift,     # 右移运算符 >>
    operator.and_,       # 按位与运算符 &
    operator.xor,        # 按位异或运算符 ^
    operator.or_,        # 按位或运算符 |
]


# 定义一个测试类 TestNDArrayOperatorsMixin，用于测试 NDArrayOperatorsMixin 类的功能
class TestNDArrayOperatorsMixin:
    def test_array_like_add(self):
        # 定义内部函数check，用于验证结果是否符合预期
        def check(result):
            _assert_equal_type_and_value(result, ArrayLike(0))

        # 测试 ArrayLike 对象与数字的加法
        check(ArrayLike(0) + 0)
        check(0 + ArrayLike(0))

        # 测试 ArrayLike 对象与 NumPy 数组的加法
        check(ArrayLike(0) + np.array(0))
        check(np.array(0) + ArrayLike(0))

        # 测试 ArrayLike 对象封装的 NumPy 数组与数字的加法
        check(ArrayLike(np.array(0)) + 0)
        check(0 + ArrayLike(np.array(0)))

        # 测试 ArrayLike 对象封装的 NumPy 数组与 NumPy 数组的加法
        check(ArrayLike(np.array(0)) + np.array(0))
        check(np.array(0) + ArrayLike(np.array(0)))

    def test_inplace(self):
        # 创建一个封装了 NumPy 数组 [0] 的 ArrayLike 对象
        array_like = ArrayLike(np.array([0]))
        # 对 ArrayLike 对象执行就地操作，加上 1
        array_like += 1
        # 验证执行操作后的类型和值是否与预期一致
        _assert_equal_type_and_value(array_like, ArrayLike(np.array([1])))

        # 创建一个 NumPy 数组 [0]
        array = np.array([0])
        # 对 NumPy 数组执行就地操作，加上 ArrayLike 对象封装的值 1
        array += ArrayLike(1)
        # 验证执行操作后的类型和值是否与预期一致
        _assert_equal_type_and_value(array, ArrayLike(np.array([1])))

    def test_opt_out(self):
        # 定义一个类 OptOut，该类不支持 __array_ufunc__，并定义了加法运算
        class OptOut:
            """Object that opts out of __array_ufunc__."""
            __array_ufunc__ = None

            def __add__(self, other):
                return self

            def __radd__(self, other):
                return self

        # 创建一个封装了值 1 的 ArrayLike 对象
        array_like = ArrayLike(1)
        # 创建一个 OptOut 对象
        opt_out = OptOut()

        # 支持的操作：ArrayLike 对象与 OptOut 对象的加法返回 OptOut 对象
        assert_(array_like + opt_out is opt_out)
        assert_(opt_out + array_like is opt_out)

        # 不支持的操作：预期会抛出 TypeError 异常
        with assert_raises(TypeError):
            # 不要使用默认的 Python 操作，避免执行 array_like = array_like + opt_out
            array_like += opt_out
        with assert_raises(TypeError):
            array_like - opt_out
        with assert_raises(TypeError):
            opt_out - array_like

    def test_subclass(self):
        # 定义一个继承自 ArrayLike 的子类 SubArrayLike
        class SubArrayLike(ArrayLike):
            """Should take precedence over ArrayLike."""

        # 创建 ArrayLike 对象 x 和 SubArrayLike 对象 y
        x = ArrayLike(0)
        y = SubArrayLike(1)
        # 验证 ArrayLike 对象与 SubArrayLike 对象的加法结果与预期一致
        _assert_equal_type_and_value(x + y, y)
        _assert_equal_type_and_value(y + x, y)

    def test_object(self):
        # 创建一个封装了值 0 的 ArrayLike 对象
        x = ArrayLike(0)
        # 创建一个普通对象 obj
        obj = object()
        # 预期会抛出 TypeError 异常，因为 ArrayLike 对象与普通对象不能进行加法操作
        with assert_raises(TypeError):
            x + obj
        with assert_raises(TypeError):
            obj + x
        with assert_raises(TypeError):
            x += obj

    def test_unary_methods(self):
        # 创建一个 NumPy 数组 [-1, 0, 1, 2]
        array = np.array([-1, 0, 1, 2])
        # 创建一个封装了该数组的 ArrayLike 对象
        array_like = ArrayLike(array)
        # 遍历一组一元操作函数，验证其对 ArrayLike 对象的操作结果是否与预期一致
        for op in [operator.neg,
                   operator.pos,
                   abs,
                   operator.invert]:
            _assert_equal_type_and_value(op(array_like), ArrayLike(op(array)))

    def test_forward_binary_methods(self):
        # 创建一个 NumPy 数组 [-1, 0, 1, 2]
        array = np.array([-1, 0, 1, 2])
        # 创建一个封装了该数组的 ArrayLike 对象
        array_like = ArrayLike(array)
        # 遍历一组二元操作函数，验证其对 ArrayLike 对象与标量的操作结果是否与预期一致
        for op in _ALL_BINARY_OPERATORS:
            expected = wrap_array_like(op(array, 1))
            actual = op(array_like, 1)
            err_msg = 'failed for operator {}'.format(op)
            _assert_equal_type_and_value(expected, actual, err_msg=err_msg)
    # 对每个二元运算符进行测试
    def test_reflected_binary_methods(self):
        for op in _ALL_BINARY_OPERATORS:
            # 使用操作符对(2, ArrayLike(1))执行运算，并包装结果为类似数组
            expected = wrap_array_like(op(2, 1))
            # 使用操作符对ArrayLike(1)和2执行运算
            actual = op(2, ArrayLike(1))
            err_msg = 'failed for operator {}'.format(op)
            # 断言期望值和实际值类型与数值都相等
            _assert_equal_type_and_value(expected, actual, err_msg=err_msg)

    # 测试矩阵乘法运算
    def test_matmul(self):
        # 创建浮点64位数组
        array = np.array([1, 2], dtype=np.float64)
        # 将数组转为类似数组对象
        array_like = ArrayLike(array)
        # 预期结果是包装后的浮点64位数
        expected = ArrayLike(np.float64(5))
        # 断言类似数组对象与原数组的矩阵乘法结果类型和数值相等
        _assert_equal_type_and_value(expected, np.matmul(array_like, array))
        # 断言类似数组对象与原数组的矩阵乘法运算结果类型和数值相等
        _assert_equal_type_and_value(
            expected, operator.matmul(array_like, array))
        # 断言原数组与类似数组对象的矩阵乘法运算结果类型和数值相等
        _assert_equal_type_and_value(
            expected, operator.matmul(array, array_like))

    # 测试 ufunc 的 at 方法
    def test_ufunc_at(self):
        # 创建类似数组对象
        array = ArrayLike(np.array([1, 2, 3, 4]))
        # 断言负数 ufunc 在指定索引位置的运算结果为 None
        assert_(np.negative.at(array, np.array([0, 1])) is None)
        # 断言类似数组对象的值为 [-1, -2, 3, 4]
        _assert_equal_type_and_value(array, ArrayLike([-1, -2, 3, 4]))

    # 测试 ufunc 返回两个输出的情况
    def test_ufunc_two_outputs(self):
        # 计算 2 的 -3 次方的尾数和指数
        mantissa, exponent = np.frexp(2 ** -3)
        # 预期结果是封装后的尾数和指数
        expected = (ArrayLike(mantissa), ArrayLike(exponent))
        # 断言类似数组对象计算 2 的 -3 次方的 frexp 结果与预期结果类型和数值相等
        _assert_equal_type_and_value(
            np.frexp(ArrayLike(2 ** -3)), expected)
        # 断言类似数组对象计算 np.array(2 的 -3 次方) 的 frexp 结果与预期结果类型和数值相等
        _assert_equal_type_and_value(
            np.frexp(ArrayLike(np.array(2 ** -3))), expected)
```
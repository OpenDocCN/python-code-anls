# `.\pytorch\test\torch_np\test_ufuncs_basic.py`

```py
"""
Poking around ufunc casting/broadcasting/dtype/out behavior.

The goal is to validate on numpy, and tests should work when replacing
>>> import numpy as no

by
>>> import torch._numpy as np
"""

# 导入标准库的操作符模块
import operator

# 导入 unittest 中的 skipIf 和 SkipTest
from unittest import skipIf as skip, SkipTest

# 导入 pytest 中的 raises
from pytest import raises as assert_raises

# 导入 Torch Dynamo 中的测试相关工具和类
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)

# 根据测试条件选择性地导入 numpy 或 torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_equal

# 参数化装饰器，用于测试一元 ufunc 函数
parametrize_unary_ufuncs = parametrize("ufunc", [np.sin])

# 参数化装饰器，用于测试类型转换相关行为
parametrize_casting = parametrize(
    "casting", ["no", "equiv", "safe", "same_kind", "unsafe"]
)

# 实例化参数化测试类
@instantiate_parametrized_tests
class TestUnaryUfuncs(TestCase):

    # 返回一个 numpy 数组，用于测试一元 ufunc 函数
    def get_x(self, ufunc):
        return np.arange(5, dtype="float64")

    # 测试一元 ufunc 函数接受标量输入，并且结果可转换为标量
    @parametrize_unary_ufuncs
    def test_scalar(self, ufunc):
        x = self.get_x(ufunc)[0]
        float(ufunc(x))

    # 跳过测试，原因是一元 ufunc 函数忽略 dtype 参数
    @skip(True, reason="XXX: unary ufuncs ignore the dtype=... parameter")
    @parametrize_unary_ufuncs
    def test_x_and_dtype(self, ufunc):
        x = self.get_x(ufunc)
        res = ufunc(x, dtype="float")
        assert res.dtype == np.dtype("float")

    # 跳过测试，原因是一元 ufunc 函数忽略 dtype 参数
    @skip(True, reason="XXX: unary ufuncs ignore the dtype=... parameter")
    @parametrize_casting
    @parametrize_unary_ufuncs
    @parametrize("dtype", ["float64", "complex128", "float32"])
    def test_x_and_dtype_casting(self, ufunc, casting, dtype):
        x = self.get_x(ufunc)
        if not np.can_cast(x, dtype, casting=casting):
            with assert_raises(TypeError):
                ufunc(x, dtype=dtype, casting=casting)
        else:
            assert ufunc(x, dtype=dtype, casting=casting).dtype == dtype

    # 测试一元 ufunc 函数在给定输出类型下的行为
    @parametrize_casting
    @parametrize_unary_ufuncs
    @parametrize("out_dtype", ["float64", "complex128", "float32"])
    def test_x_and_out_casting(self, ufunc, casting, out_dtype):
        x = self.get_x(ufunc)
        out = np.empty_like(x, dtype=out_dtype)
        if not np.can_cast(x, out_dtype, casting=casting):
            with assert_raises(TypeError):
                ufunc(x, out=out, casting=casting)
        else:
            result = ufunc(x, out=out, casting=casting)
            assert result.dtype == out_dtype
            assert result is out

    # 参数化装饰器，用于测试一元 ufunc 函数
    @parametrize_unary_ufuncs
    # 定义一个测试方法，用于测试某个ufunc函数的广播行为
    def test_x_and_out_broadcast(self, ufunc):
        # 获取测试用例中的输入数据x
        x = self.get_x(ufunc)
        # 创建一个空数组out，形状为(x的行数, x的行数)
        out = np.empty((x.shape[0], x.shape[0]))

        # 将x广播到和out相同的形状，形成x_b
        x_b = np.broadcast_to(x, out.shape)

        # 使用ufunc函数对x进行操作，指定输出为out，保存结果到res_out
        res_out = ufunc(x, out=out)
        # 使用ufunc函数对x_b进行操作，保存结果到res_bcast
        res_bcast = ufunc(x_b)
        # 断言res_out与out是同一个对象
        assert res_out is out
        # 断言res_out与res_bcast在数值上相等
        assert_equal(res_out, res_bcast)

        # 重新创建一个形状为(1, x的行数)的空数组out
        out = np.empty((1, x.shape[0]))
        # 将x广播到和out相同的形状，形成x_b
        x_b = np.broadcast_to(x, out.shape)

        # 使用ufunc函数对x进行操作，指定输出为out，保存结果到res_out
        res_out = ufunc(x, out=out)
        # 使用ufunc函数对x_b进行操作，保存结果到res_bcast
        res_bcast = ufunc(x_b)
        # 断言res_out与out是同一个对象
        assert res_out is out
        # 断言res_out与res_bcast在数值上相等
        assert_equal(res_out, res_bcast)
# 定义了一组元组，每个元组包含三个元素，分别是 numpy 的函数、对应的 operator 模块中的二进制操作符函数和增强赋值操作符函数
ufunc_op_iop_numeric = [
    (np.add, operator.__add__, operator.__iadd__),
    (np.subtract, operator.__sub__, operator.__isub__),
    (np.multiply, operator.__mul__, operator.__imul__),
]

# 从 ufunc_op_iop_numeric 中提取第一个元素（numpy 函数），组成一个列表
ufuncs_with_dunders = [ufunc for ufunc, _, _ in ufunc_op_iop_numeric]

# 包含一些与数值计算相关的 numpy 函数的列表
numeric_binary_ufuncs = [
    np.float_power,
    np.power,
]

# 不支持复数输入的 numpy 函数列表
no_complex = [
    np.floor_divide,
    np.hypot,
    np.arctan2,
    np.copysign,
    np.fmax,
    np.fmin,
    np.fmod,
    np.heaviside,
    np.logaddexp,
    np.logaddexp2,
    np.maximum,
    np.minimum,
]

# 使用 parametrize 函数，为一组函数生成参数化测试
parametrize_binary_ufuncs = parametrize(
    "ufunc", ufuncs_with_dunders + numeric_binary_ufuncs + no_complex
)

# TODO: 需要特殊处理的一组特定的 numpy 函数，暂未实现具体处理方法
"""
 'bitwise_and',
 'bitwise_or',
 'bitwise_xor',
 'equal',
 'lcm',
 'ldexp',
 'left_shift',
 'less',
 'less_equal',
 'gcd',
 'greater',
 'greater_equal',
 'logical_and',
 'logical_or',
 'logical_xor',
 'matmul',
 'not_equal',
"""

# 使用 instantiate_parametrized_tests 装饰器，实例化参数化测试类
@instantiate_parametrized_tests
class TestBinaryUfuncs(TestCase):

    # 获取参数 ufunc，返回两个 numpy 数组作为参数
    def get_xy(self, ufunc):
        return np.arange(5, dtype="float64"), np.arange(8, 13, dtype="float64")

    # 参数化测试函数，测试 ufunc 是否能接受标量输入并返回标量结果
    @parametrize_binary_ufuncs
    def test_scalar(self, ufunc):
        xy = self.get_xy(ufunc)
        x, y = xy[0][0], xy[1][0]
        float(ufunc(x, y))

    # 参数化测试函数，测试 ufunc 是否能接受向量输入并返回相同长度的向量结果
    @parametrize_binary_ufuncs
    def test_vector_vs_scalar(self, ufunc):
        x, y = self.get_xy(ufunc)
        assert_equal(ufunc(x, y), [ufunc(a, b) for a, b in zip(x, y)])

    # 参数化测试函数，测试 ufunc 是否能接受指定的输入类型和输出类型，并进行类型转换
    @parametrize_casting
    @parametrize_binary_ufuncs
    @parametrize("out_dtype", ["float64", "complex128", "float32"])
    def test_xy_and_out_casting(self, ufunc, casting, out_dtype):
        x, y = self.get_xy(ufunc)
        out = np.empty_like(x, dtype=out_dtype)

        # 如果 ufunc 不支持复数输入，并且输出类型是复数类型，则跳过测试
        if ufunc in no_complex and np.issubdtype(out_dtype, np.complexfloating):
            raise SkipTest(f"{ufunc} does not accept complex.")

        can_cast_x = np.can_cast(x, out_dtype, casting=casting)
        can_cast_y = np.can_cast(y, out_dtype, casting=casting)

        # 如果 x 或者 y 不能被转换成指定的输出类型，则预期会抛出 TypeError 异常
        if not (can_cast_x and can_cast_y):
            with assert_raises(TypeError):
                ufunc(x, out=out, casting=casting)
        else:
            # 否则，计算 ufunc 的结果，并验证其类型和是否为给定的输出数组
            result = ufunc(x, y, out=out, casting=casting)
            assert result.dtype == out_dtype
            assert result is out

    # 参数化测试函数，测试 ufunc 是否能接受广播后的输入，并且能正确地进行广播计算
    @parametrize_binary_ufuncs
    def test_xy_and_out_broadcast(self, ufunc):
        x, y = self.get_xy(ufunc)
        y = y[:, None]  # 扩展 y 以便进行广播
        out = np.empty((2, y.shape[0], x.shape[0]))  # 创建一个空的输出数组

        x_b = np.broadcast_to(x, out.shape)  # 广播 x 到与 out 相同的形状
        y_b = np.broadcast_to(y, out.shape)  # 广播 y 到与 out 相同的形状

        # 使用 ufunc 计算 x, y 的结果，并保存到 out 中
        res_out = ufunc(x, y, out=out)
        res_bcast = ufunc(x_b, y_b)

        # 检查 ufunc 计算结果是否与预期的广播计算结果一致
        # TODO: 交换顺序会导致图形中断，测试失败。参见 test/dynamo/test_misc.py -k test_numpy_graph_break
        assert res_out is out
        assert_equal(res_out, res_bcast)
# 定义一个包含数值型 NumPy 数据类型的列表
dtypes_numeric = [np.int32, np.float32, np.float64, np.complex128]

# 使用装饰器实例化参数化测试，应用于类 TestNdarrayDunderVsUfunc
@instantiate_parametrized_tests
class TestNdarrayDunderVsUfunc(TestCase):
    """Test ndarray dunders which delegate to ufuncs, vs ufuncs."""

    # 参数化测试函数，用于测试基本操作（op）、右操作（rop）、增量操作（iop），仅考虑数值型操作
    @parametrize("ufunc, op, iop", ufunc_op_iop_numeric)
    def test_basic(self, ufunc, op, iop):
        """basic op/rop/iop, no dtypes, no broadcasting"""

        # __add__：测试 ndarray 的 __add__ 方法
        a = np.array([1, 2, 3])
        assert_equal(op(a, 1), ufunc(a, 1))  # 测试标量加法
        assert_equal(op(a, a.tolist()), ufunc(a, a.tolist()))  # 测试与列表的加法
        assert_equal(op(a, a), ufunc(a, a))  # 测试 ndarray 与 ndarray 的加法

        # __radd__：测试 ndarray 的 __radd__ 方法
        a = np.array([1, 2, 3])
        assert_equal(op(1, a), ufunc(1, a))  # 测试标量右加法
        assert_equal(op(a.tolist(), a), ufunc(a, a.tolist()))  # 测试列表右加法

        # __iadd__：测试 ndarray 的 __iadd__ 方法
        a0 = np.array([2, 4, 6])
        a = a0.copy()

        iop(a, 2)  # 原位修改 a
        assert_equal(a, op(a0, 2))  # 验证原位操作后的结果是否正确

        a0 = np.array([2, 4, 6])
        a = a0.copy()
        iop(a, a)
        assert_equal(a, op(a0, a0))  # 验证原位操作 ndarray 与 ndarray 是否正确

    # 参数化测试函数，测试其他标量类型的操作
    @parametrize("ufunc, op, iop", ufunc_op_iop_numeric)
    @parametrize("other_dtype", dtypes_numeric)
    def test_other_scalar(self, ufunc, op, iop, other_dtype):
        """Test op/iop/rop when the other argument is a scalar of a different dtype."""
        a = np.array([1, 2, 3])
        b = other_dtype(3)

        # 如果 ufunc 不支持复数类型而 other_dtype 是复数类型，则跳过测试
        if ufunc in no_complex and issubclass(other_dtype, np.complexfloating):
            raise SkipTest(f"{ufunc} does not accept complex.")

        # __op__：测试 ndarray 的 __op__ 方法
        result = op(a, b)
        assert_equal(result, ufunc(a, b))  # 验证操作结果是否与 ufunc 的结果一致

        # 如果结果的数据类型不等于 a 和 b 的结果数据类型，则验证结果数据类型是否正确
        if result.dtype != np.result_type(a, b):
            assert result.dtype == np.result_type(a, b)

        # __rop__：测试 ndarray 的 __rop__ 方法
        result = op(b, a)
        assert_equal(result, ufunc(b, a))  # 验证右操作的结果是否与 ufunc 的结果一致
        if result.dtype != np.result_type(a, b):
            assert result.dtype == np.result_type(a, b)

        # __iop__：测试 ndarray 的 __iop__ 方法，将结果转换为 self.dtype，如果无法转换则引发异常
        can_cast = np.can_cast(
            np.result_type(a.dtype, other_dtype), a.dtype, casting="same_kind"
        )
        if can_cast:
            a0 = a.copy()
            result = iop(a, b)
            assert_equal(result, ufunc(a0, b))  # 验证原位操作后的结果是否正确
            if result.dtype != np.result_type(a, b):
                assert result.dtype == np.result_type(a0, b)
        else:
            with assert_raises((TypeError, RuntimeError)):  # 如果无法转换则引发异常
                iop(a, b)

    # 参数化测试函数，测试数值型操作以及其他数据类型的标量操作
    @parametrize("ufunc, op, iop", ufunc_op_iop_numeric)
    @parametrize("other_dtype", dtypes_numeric)
    # 定义一个测试函数，用于测试当另一个参数是不同数据类型的数组时的操作
    def test_other_array(self, ufunc, op, iop, other_dtype):
        """Test op/iop/rop when the other argument is an array of a different dtype."""
        # 创建一个整数类型的数组a
        a = np.array([1, 2, 3])
        # 创建一个指定数据类型的数组b
        b = np.array([5, 6, 7], dtype=other_dtype)

        # 如果ufunc在不支持复数运算的列表中，并且other_dtype是复数类型，则抛出跳过测试的异常
        if ufunc in no_complex and issubclass(other_dtype, np.complexfloating):
            raise SkipTest(f"{ufunc} does not accept complex.")

        # __op__：执行操作op(a, b)，并验证结果是否与ufunc(a, b)相等
        result = op(a, b)
        assert_equal(result, ufunc(a, b))
        # 如果结果的数据类型不等于a和b的结果类型，则验证结果的数据类型是否为a和b的结果类型
        if result.dtype != np.result_type(a, b):
            assert result.dtype == np.result_type(a, b)

        # __rop__(other array)：执行操作op(b, a)，并验证结果是否与ufunc(b, a)相等
        result = op(b, a)
        assert_equal(result, ufunc(b, a))
        # 如果结果的数据类型不等于a和b的结果类型，则验证结果的数据类型是否为a和b的结果类型
        if result.dtype != np.result_type(a, b):
            assert result.dtype == np.result_type(a, b)

        # __iop__：检查是否可以在"same_kind"的类型转换下从other_dtype转换到a.dtype
        can_cast = np.can_cast(
            np.result_type(a.dtype, other_dtype), a.dtype, casting="same_kind"
        )
        if can_cast:
            # 如果可以转换，则复制a数组，执行操作iop(a, b)，并验证结果是否与ufunc(a0, b)相等
            a0 = a.copy()
            result = iop(a, b)
            assert_equal(result, ufunc(a0, b))
            # 如果结果的数据类型不等于a和b的结果类型，则验证结果的数据类型是否为a0和b的结果类型
            if result.dtype != np.result_type(a, b):
                assert result.dtype == np.result_type(a0, b)
        else:
            # 如果不能转换，则验证iop(a, b)是否会引发TypeError或RuntimeError异常
            with assert_raises((TypeError, RuntimeError)):  # XXX np.UFuncTypeError
                iop(a, b)

    @parametrize("ufunc, op, iop", ufunc_op_iop_numeric)
    # 定义一个参数化测试，测试操作/反向操作/原地操作的广播情况
    def test_other_array_bcast(self, ufunc, op, iop):
        """Test op/rop/iop with broadcasting"""
        # __op__：创建一个整数类型的数组a，并执行op(a, a[:, None])和ufunc(a, a[:, None])的比较
        a = np.array([1, 2, 3])
        result_op = op(a, a[:, None])
        result_ufunc = ufunc(a, a[:, None])
        assert result_op.shape == result_ufunc.shape
        assert_equal(result_op, result_ufunc)

        # 如果结果的数据类型不等于ufunc的数据类型，则验证结果的数据类型是否为ufunc的数据类型
        if result_op.dtype != result_ufunc.dtype:
            assert result_op.dtype == result_ufunc.dtype

        # __rop__：创建一个整数类型的数组a，并执行op(a[:, None], a)和ufunc(a[:, None], a)的比较
        a = np.array([1, 2, 3])
        result_op = op(a[:, None], a)
        result_ufunc = ufunc(a[:, None], a)
        assert result_op.shape == result_ufunc.shape
        assert_equal(result_op, result_ufunc)

        # 如果结果的数据类型不等于ufunc的数据类型，则验证结果的数据类型是否为ufunc的数据类型
        if result_op.dtype != result_ufunc.dtype:
            assert result_op.dtype == result_ufunc.dtype

        # __iop__ : 原地操作（例如self += other）不会广播self
        # 复制a[:, None]数组，并验证iop(a, b)是否会引发ValueError或RuntimeError异常
        b = a[:, None].copy()
        with assert_raises((ValueError, RuntimeError)):  # XXX ValueError in numpy
            iop(a, b)

        # 然而，`self += other`会广播other
        # 将a广播到(3, 3)的数组aa，并复制aa
        aa = np.broadcast_to(a, (3, 3)).copy()
        aa0 = aa.copy()

        # 执行iop(aa, a)，并验证结果是否与ufunc(aa0, a)相等
        result = iop(aa, a)
        result_ufunc = ufunc(aa0, a)

        assert result.shape == result_ufunc.shape
        assert_equal(result, result_ufunc)

        # 如果结果的数据类型不等于ufunc的数据类型，则验证结果的数据类型是否为ufunc的数据类型
        if result_op.dtype != result_ufunc.dtype:
            assert result_op.dtype == result_ufunc.dtype
class TestUfuncDtypeKwd(TestCase):
    def test_binary_ufunc_dtype(self):
        # default computation uses float64:
        r64 = np.add(1, 1e-15)
        # 检查结果的数据类型是否为 float64
        assert r64.dtype == "float64"
        # 检查计算结果是否大于 0
        assert r64 - 1 > 0

        # force the float32 dtype: loss of precision
        # 强制使用 float32 数据类型，可能导致精度损失
        r32 = np.add(1, 1e-15, dtype="float32")
        assert r32.dtype == "float32"
        assert r32 == 1

        # now force the cast
        # 强制类型转换为布尔值，casting 参数设置为 "unsafe"
        rb = np.add(1.0, 1e-15, dtype=bool, casting="unsafe")
        assert rb.dtype == bool

    def test_binary_ufunc_dtype_and_out(self):
        # all in float64: no precision loss
        # 使用 np.empty 创建指定数据类型的空数组，作为输出数组
        out64 = np.empty(2, dtype=np.float64)
        # 使用 np.add 进行计算，将结果存储到 out64 中
        r64 = np.add([1.0, 2.0], 1.0e-15, out=out64)

        # 检查 r64 数组中的每个元素是否不等于 [1.0, 2.0]
        assert (r64 != [1.0, 2.0]).all()
        # 检查输出数组的数据类型是否为 np.float64
        assert r64.dtype == np.float64

        # all in float32: loss of precision, result is float32
        # 使用 np.empty 创建指定数据类型的空数组，作为输出数组
        out32 = np.empty(2, dtype=np.float32)
        # 使用 np.add 进行计算，将结果存储到 out32 中，并强制使用 np.float32 数据类型
        r32 = np.add([1.0, 2.0], 1.0e-15, dtype=np.float32, out=out32)
        # 检查 r32 数组中的每个元素是否等于 [1, 2]
        assert (r32 == [1, 2]).all()
        # 检查输出数组的数据类型是否为 np.float32
        assert r32.dtype == np.float32

        # dtype is float32, so computation is in float32: precision loss
        # the result is then cast to float64
        # 使用 np.empty 创建指定数据类型的空数组，作为输出数组
        out64 = np.empty(2, dtype=np.float64)
        # 使用 np.add 进行计算，将结果存储到 out64 中，并强制使用 np.float32 数据类型
        r = np.add([1.0, 2.0], 1.0e-15, dtype=np.float32, out=out64)
        # 检查 r 数组中的每个元素是否等于 [1, 2]
        assert (r == [1, 2]).all()
        # 检查输出数组的数据类型是否为 np.float64
        assert r.dtype == np.float64

        # Internal computations are in float64, but the final cast to out.dtype
        # truncates the precision => precision loss.
        # 使用 np.empty 创建指定数据类型的空数组，作为输出数组
        out32 = np.empty(2, dtype=np.float32)
        # 使用 np.add 进行计算，将结果存储到 out32 中，并强制使用 np.float64 数据类型
        r = np.add([1.0, 2.0], 1.0e-15, dtype=np.float64, out=out32)
        # 检查 r 数组中的每个元素是否等于 [1, 2]
        assert (r == [1, 2]).all()
        # 检查输出数组的数据类型是否为 np.float32


if __name__ == "__main__":
    run_tests()
```
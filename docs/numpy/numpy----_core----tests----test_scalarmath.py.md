# `.\numpy\numpy\_core\tests\test_scalarmath.py`

```py
# 导入上下文管理模块
import contextlib
# 导入系统相关模块
import sys
# 导入警告处理模块
import warnings
# 导入迭代工具模块
import itertools
# 导入运算符模块
import operator
# 导入平台信息模块
import platform
# 导入 numpy 中的 PEP 440 版本处理工具
from numpy._utils import _pep440
# 导入 pytest 测试框架
import pytest
# 导入 hypothesis 中的给定测试策略和设置模块
from hypothesis import given, settings
# 导入 hypothesis 中的随机抽样策略模块
from hypothesis.strategies import sampled_from
# 导入 hypothesis.extra 中的 numpy 扩展模块
from hypothesis.extra import numpy as hynp

# 导入 numpy 库并重命名为 np
import numpy as np
# 导入 numpy 中的复数警告
from numpy.exceptions import ComplexWarning
# 导入 numpy 测试模块中的各种断言函数
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_almost_equal,
    assert_array_equal, IS_PYPY, suppress_warnings, _gen_alignment_data,
    assert_warns, _SUPPORTS_SVE,
    )

# 定义常见的数据类型列表
types = [np.bool, np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc,
         np.int_, np.uint, np.longlong, np.ulonglong,
         np.single, np.double, np.longdouble, np.csingle,
         np.cdouble, np.clongdouble]

# 获取所有浮点类型的子类列表
floating_types = np.floating.__subclasses__()
# 获取所有复数浮点类型的子类列表
complex_floating_types = np.complexfloating.__subclasses__()

# 定义包含对象和 None 的列表
objecty_things = [object(), None]

# 合理的标量操作符列表
reasonable_operators_for_scalars = [
    operator.lt, operator.le, operator.eq, operator.ne, operator.ge,
    operator.gt, operator.add, operator.floordiv, operator.mod,
    operator.mul, operator.pow, operator.sub, operator.truediv,
]


# 这个类测试了不同类型的数值运算行为
class TestTypes:
    # 测试各种类型的初始化和比较
    def test_types(self):
        for atype in types:
            a = atype(1)
            assert_(a == 1, "error with %r: got %r" % (atype, a))

    # 测试不同类型的加法运算
    def test_type_add(self):
        # 遍历类型列表
        for k, atype in enumerate(types):
            # 创建标量和数组
            a_scalar = atype(3)
            a_array = np.array([3], dtype=atype)
            # 再次遍历类型列表
            for l, btype in enumerate(types):
                b_scalar = btype(1)
                b_array = np.array([1], dtype=btype)
                # 进行加法运算，比较结果的数据类型
                c_scalar = a_scalar + b_scalar
                c_array = a_array + b_array
                assert_equal(c_scalar.dtype, c_array.dtype,
                           "error with types (%d/'%c' + %d/'%c')" %
                            (k, np.dtype(atype).char, l, np.dtype(btype).char))

    # 测试不同类型的数组创建
    def test_type_create(self):
        for k, atype in enumerate(types):
            # 使用不同类型创建数组，并进行比较
            a = np.array([1, 2, 3], atype)
            b = atype([1, 2, 3])
            assert_equal(a, b)

    # 测试标量对象的内存泄漏
    def test_leak(self):
        # 模拟大量标量对象的创建，观察内存泄漏情况
        for i in range(200000):
            np.add(1, 1)


# 检查 ufunc 和标量等价性的函数
def check_ufunc_scalar_equivalence(op, arr1, arr2):
    scalar1 = arr1[()]
    scalar2 = arr2[()]
    assert isinstance(scalar1, np.generic)
    assert isinstance(scalar2, np.generic)
    # 检查 arr1 或 arr2 是否包含复数类型的数据
    if arr1.dtype.kind == "c" or arr2.dtype.kind == "c":
        # 定义支持复数类型的比较运算符集合
        comp_ops = {operator.ge, operator.gt, operator.le, operator.lt}
        # 如果操作是比较运算符，并且 scalar1 或 scalar2 是 NaN，则跳过测试
        if op in comp_ops and (np.isnan(scalar1) or np.isnan(scalar2)):
            pytest.xfail("complex comp ufuncs use sort-order, scalars do not.")

    # 如果操作是乘方运算符，并且 arr2 的值在 [-1, 0, 0.5, 1, 2] 中
    if op == operator.pow and arr2.item() in [-1, 0, 0.5, 1, 2]:
        # 对于 array**scalar 的特殊情况，可能会有不同的结果数据类型
        # （其他乘方可能也存在问题，但这里不涉及。）
        # TODO: 应该解决这个问题会比较好。
        pytest.skip("array**2 can have incorrect/weird result dtype")

    # 忽略浮点异常，因为对于整数来说，它们可能只是不匹配。
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        # 从 2022 年 3 月开始，将比较警告（DeprecationWarning）转换为错误
        warnings.simplefilter("error", DeprecationWarning)
        try:
            # 尝试应用操作符 op 到 arr1 和 arr2 上
            res = op(arr1, arr2)
        except Exception as e:
            # 如果出现异常，验证应该抛出相同类型的异常
            with pytest.raises(type(e)):
                op(scalar1, scalar2)
        else:
            # 否则，对 scalar1 和 scalar2 应用操作符，并验证结果与 res 是否严格相等
            scalar_res = op(scalar1, scalar2)
            assert_array_equal(scalar_res, res, strict=True)
# 标记此测试为较慢的测试
@pytest.mark.slow
# 设置测试的策略，包括最大样本数和超时时间
@settings(max_examples=10000, deadline=2000)
# 使用 hypothesis 的 given 装饰器定义测试参数
@given(
    # 从 reasonable_operators_for_scalars 中随机选择一个操作符作为参数
    sampled_from(reasonable_operators_for_scalars),
    # 生成一个标量 dtype 的 NumPy 数组，形状为空
    hynp.arrays(dtype=hynp.scalar_dtypes(), shape=()),
    # 生成另一个标量 dtype 的 NumPy 数组，形状为空
    hynp.arrays(dtype=hynp.scalar_dtypes(), shape=())
)
def test_array_scalar_ufunc_equivalence(op, arr1, arr2):
    """
    This is a thorough test attempting to cover important promotion paths
    and ensuring that arrays and scalars stay as aligned as possible.
    However, if it creates troubles, it should maybe just be removed.
    """
    # 调用函数，检查数组和标量之间的等效性
    check_ufunc_scalar_equivalence(op, arr1, arr2)


# 标记此测试为较慢的测试
@pytest.mark.slow
# 使用 hypothesis 的 given 装饰器定义测试参数
@given(
    # 从 reasonable_operators_for_scalars 中随机选择一个操作符作为参数
    sampled_from(reasonable_operators_for_scalars),
    # 生成一个标量 dtype 作为参数
    hynp.scalar_dtypes(),
    # 生成另一个标量 dtype 作为参数
    hynp.scalar_dtypes()
)
def test_array_scalar_ufunc_dtypes(op, dt1, dt2):
    # Same as above, but don't worry about sampling weird values so that we
    # do not have to sample as much
    # 创建一个 NumPy 数组，包含单个元素 2，指定数据类型为 dt1
    arr1 = np.array(2, dtype=dt1)
    # 创建一个 NumPy 数组，包含单个元素 3，指定数据类型为 dt2
    arr2 = np.array(3, dtype=dt2)  # some power do weird things.
    
    # 调用函数，检查数组和标量之间的等效性
    check_ufunc_scalar_equivalence(op, arr1, arr2)


# 使用 pytest 的 parametrize 装饰器，为测试指定不同的 fscalar 参数
@pytest.mark.parametrize("fscalar", [np.float16, np.float32])
def test_int_float_promotion_truediv(fscalar):
    # Promotion for mixed int and float32/float16 must not go to float64
    # 创建一个 np.int8 类型的整数 i
    i = np.int8(1)
    # 创建一个指定类型的浮点数 f，根据参数 fscalar 不同而变化
    f = fscalar(1)
    # 期望的结果数据类型，根据 i 和 f 的类型计算得出
    expected = np.result_type(i, f)
    # 断言整数 i 除以浮点数 f 的结果数据类型符合预期
    assert (i / f).dtype == expected
    # 断言浮点数 f 除以整数 i 的结果数据类型符合预期
    assert (f / i).dtype == expected
    # 但是普通的整数除法（int / int）应该返回 float64：
    assert (i / i).dtype == np.dtype("float64")
    # 对于 int16，结果至少应该是 float32（采用 ufunc 路径）：
    assert (np.int16(1) / f).dtype == np.dtype("float32")


# 创建一个名为 TestBaseMath 的测试类
class TestBaseMath:
    # 使用 pytest 的 mark 装饰器标记此测试为预期失败，并给出原因
    @pytest.mark.xfail(_SUPPORTS_SVE, reason="gh-22982")
    def test_blocked(self):
        # 测试对于 SIMD 指令的对齐偏移
        # 对于 vz + 2 * (vs - 1) + 1 的对齐
        for dt, sz in [(np.float32, 11), (np.float64, 7), (np.int32, 11)]:
            # 生成对齐数据，返回输出、输入1、输入2和消息
            for out, inp1, inp2, msg in _gen_alignment_data(dtype=dt,
                                                            type='binary',
                                                            max_size=sz):
                # 期望 inp1 的元素全为 1
                exp1 = np.ones_like(inp1)
                # 将 inp1 的元素设置为 1
                inp1[...] = np.ones_like(inp1)
                # 将 inp2 的元素设置为 0
                inp2[...] = np.zeros_like(inp2)
                # 断言 inp1 与 inp2 的元素相加结果接近 exp1，如果不是则输出错误消息 msg
                assert_almost_equal(np.add(inp1, inp2), exp1, err_msg=msg)
                # 断言 inp1 与 2 相加结果接近 exp1 + 2，如果不是则输出错误消息 msg
                assert_almost_equal(np.add(inp1, 2), exp1 + 2, err_msg=msg)
                # 断言 1 与 inp2 相加结果接近 exp1，如果不是则输出错误消息 msg
                assert_almost_equal(np.add(1, inp2), exp1, err_msg=msg)

                # 将 inp1 与 inp2 的相加结果存入 out
                np.add(inp1, inp2, out=out)
                # 断言 out 的结果接近 exp1，如果不是则输出错误消息 msg
                assert_almost_equal(out, exp1, err_msg=msg)

                # 将 inp2 的元素逐个加上从 1 开始的序列
                inp2[...] += np.arange(inp2.size, dtype=dt) + 1
                # 断言 inp2 的平方结果接近 inp2 自身与自身相乘的结果，如果不是则输出错误消息 msg
                assert_almost_equal(np.square(inp2),
                                    np.multiply(inp2, inp2),  err_msg=msg)
                # 对于非整数型数据，跳过真实除法的断言
                if dt != np.int32:
                    # 断言 inp2 的倒数接近 1 除以 inp2 的结果，如果不是则输出错误消息 msg
                    assert_almost_equal(np.reciprocal(inp2),
                                        np.divide(1, inp2),  err_msg=msg)

                # 将 inp1 的元素设置为 1
                inp1[...] = np.ones_like(inp1)
                # 将 inp1 与 2 相加结果存入 out
                np.add(inp1, 2, out=out)
                # 断言 out 的结果接近 exp1 + 2，如果不是则输出错误消息 msg
                assert_almost_equal(out, exp1 + 2, err_msg=msg)
                # 将 inp2 的元素设置为 1
                inp2[...] = np.ones_like(inp2)
                # 将 2 与 inp2 相加结果存入 out
                np.add(2, inp2, out=out)
                # 断言 out 的结果接近 exp1 + 2，如果不是则输出错误消息 msg
                assert_almost_equal(out, exp1 + 2, err_msg=msg)

    def test_lower_align(self):
        # 检查未对齐到元素大小的数据
        # 即在 i386 上，双精度浮点数按 4 字节对齐
        d = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)
        o = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)
        # 断言 d 与 d 的和接近 d 的两倍
        assert_almost_equal(d + d, d * 2)
        # 将 d 与 d 的和存入 o
        np.add(d, d, out=o)
        # 将 np.ones_like(d) 与 d 的和存入 o
        np.add(np.ones_like(d), d, out=o)
        # 将 d 与 np.ones_like(d) 的和存入 o
        np.add(d, np.ones_like(d), out=o)
        # 将 np.ones_like(d) 与 d 相加，但不存入任何输出
        np.add(np.ones_like(d), d)
        # 将 d 与 np.ones_like(d) 相加，但不存入任何输出
        np.add(d, np.ones_like(d))
# 定义一个测试类 TestPower，用于测试不同数据类型的幂运算
class TestPower:

    # 测试小数据类型（np.int8, np.int16, np.float16）
    def test_small_types(self):
        # 遍历数据类型列表
        for t in [np.int8, np.int16, np.float16]:
            # 创建一个 t 类型的变量 a，并赋值为 3
            a = t(3)
            # 计算 a 的四次方，赋值给变量 b
            b = a ** 4
            # 断言 b 的值为 81，如果不是则输出错误信息
            assert_(b == 81, "error with %r: got %r" % (t, b))

    # 测试大数据类型（np.int32, np.int64, np.float32, np.float64, np.longdouble）
    def test_large_types(self):
        # 遍历数据类型列表
        for t in [np.int32, np.int64, np.float32, np.float64, np.longdouble]:
            # 创建一个 t 类型的变量 a，并赋值为 51
            a = t(51)
            # 计算 a 的四次方，赋值给变量 b
            b = a ** 4
            # 构造错误信息字符串
            msg = "error with %r: got %r" % (t, b)
            # 如果 t 是整数类型，则断言 b 的值为 6765201，否则使用近似值比较
            if np.issubdtype(t, np.integer):
                assert_(b == 6765201, msg)
            else:
                assert_almost_equal(b, 6765201, err_msg=msg)

    # 测试整数类型取负幂运算
    def test_integers_to_negative_integer_power(self):
        # 构建一个列表，其中包含与各个数据类型的组合可能导致的负幂结果
        exp = [np.array(-1, dt)[()] for dt in 'bhilq']

        # 1 ** -1 特殊情况
        base = [np.array(1, dt)[()] for dt in 'bhilqBHILQ']
        # 遍历基数和负幂结果的组合
        for i1, i2 in itertools.product(base, exp):
            # 如果 i1 的数据类型不是 uint64，则断言会引发 ValueError
            if i1.dtype != np.uint64:
                assert_raises(ValueError, operator.pow, i1, i2)
            else:
                # 计算 i1 的 i2 次幂，返回浮点数结果
                res = operator.pow(i1, i2)
                # 断言 res 的数据类型为 np.float64，并近似等于 1.0
                assert_(res.dtype.type is np.float64)
                assert_almost_equal(res, 1.)

        # -1 ** -1 特殊情况
        base = [np.array(-1, dt)[()] for dt in 'bhilq']
        # 遍历基数和负幂结果的组合
        for i1, i2 in itertools.product(base, exp):
            # 如果 i1 的数据类型不是 uint64，则断言会引发 ValueError
            if i1.dtype != np.uint64:
                assert_raises(ValueError, operator.pow, i1, i2)
            else:
                # 计算 i1 的 i2 次幂，返回浮点数结果
                res = operator.pow(i1, i2)
                # 断言 res 的数据类型为 np.float64，并近似等于 -1.0
                assert_(res.dtype.type is np.float64)
                assert_almost_equal(res, -1.)

        # 2 ** -1 一般情况
        base = [np.array(2, dt)[()] for dt in 'bhilqBHILQ']
        # 遍历基数和负幂结果的组合
        for i1, i2 in itertools.product(base, exp):
            # 如果 i1 的数据类型不是 uint64，则断言会引发 ValueError
            if i1.dtype != np.uint64:
                assert_raises(ValueError, operator.pow, i1, i2)
            else:
                # 计算 i1 的 i2 次幂，返回浮点数结果
                res = operator.pow(i1, i2)
                # 断言 res 的数据类型为 np.float64，并近似等于 0.5
                assert_(res.dtype.type is np.float64)
                assert_almost_equal(res, .5)

    # 测试混合类型的幂运算
    def test_mixed_types(self):
        # 定义一个包含各种数据类型的列表
        typelist = [np.int8, np.int16, np.float16,
                    np.float32, np.float64, np.int8,
                    np.int16, np.int32, np.int64]
        # 遍历两个数据类型的组合
        for t1 in typelist:
            for t2 in typelist:
                # 创建一个 t1 类型的变量 a，并赋值为 3
                a = t1(3)
                # 创建一个 t2 类型的变量 b，并赋值为 2
                b = t2(2)
                # 计算 a 的 b 次幂，结果赋值给 result
                result = a**b
                # 构造错误信息字符串
                msg = ("error with %r and %r:"
                       "got %r, expected %r") % (t1, t2, result, 9)
                # 如果 result 的数据类型是整数类型，则断言 result 等于 9
                if np.issubdtype(np.dtype(result), np.integer):
                    assert_(result == 9, msg)
                else:
                    # 否则使用近似值比较
                    assert_almost_equal(result, 9, err_msg=msg)
    def test_modular_power(self):
        # 模幂功能未实现，因此确保引发错误
        a = 5  # 设置基数为5
        b = 4  # 设置指数为4
        c = 10  # 设置模数为10
        expected = pow(a, b, c)  # 计算 a^b % c 的结果，但此行实际上未使用，用 noqa: F841 标记告知 linter 可以忽略未使用的变量警告
        for t in (np.int32, np.float32, np.complex64):
            # 注意，3元幂运算只根据第一个参数进行分派
            # 对于不同的数据类型 t(a)，验证 pow 函数的调用是否会引发 TypeError
            assert_raises(TypeError, operator.pow, t(a), b, c)
            assert_raises(TypeError, operator.pow, np.array(t(a)), b, c)
# 定义一个函数，实现整数除法和取模操作，并返回商和余数
def floordiv_and_mod(x, y):
    return (x // y, x % y)


# 根据数据类型判断符号并返回一个或两个元素的元组
def _signs(dt):
    if dt in np.typecodes['UnsignedInteger']:
        return (+1,)
    else:
        return (+1, -1)


# 定义一个测试类 TestModulus
class TestModulus:

    # 测试整数除法和取模的基本情况
    def test_modulus_basic(self):
        # 获取所有整数和浮点数的数据类型编码
        dt = np.typecodes['AllInteger'] + np.typecodes['Float']
        # 遍历 floordiv_and_mod 和 divmod 函数
        for op in [floordiv_and_mod, divmod]:
            # 使用 itertools.product 生成数据类型的组合
            for dt1, dt2 in itertools.product(dt, dt):
                # 对于每一对数据类型，获取它们的符号组合
                for sg1, sg2 in itertools.product(_signs(dt1), _signs(dt2)):
                    # 格式化消息字符串，描述当前操作的参数
                    fmt = 'op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s'
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    # 创建 np.array 数组，使用符号和数据类型创建数值
                    a = np.array(sg1*71, dtype=dt1)[()]
                    b = np.array(sg2*19, dtype=dt2)[()]
                    # 调用操作函数进行整数除法和取模运算
                    div, rem = op(a, b)
                    # 使用 assert_equal 断言确保恢复原始值
                    assert_equal(div*b + rem, a, err_msg=msg)
                    # 对于取模结果进行额外的断言，验证其符号和大小关系
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    # 测试浮点数取模的精确性
    def test_float_modulus_exact(self):
        # 测试小整数的浮点数取模结果的精确性，也适用于乘以二的幂的情况
        nlst = list(range(-127, 0))
        plst = list(range(1, 128))
        dividend = nlst + [0] + plst
        divisor = nlst + plst
        arg = list(itertools.product(dividend, divisor))
        tgt = list(divmod(*t) for t in arg)

        a, b = np.array(arg, dtype=int).T
        # 将 Python 的精确整数结果转换为浮点数，以支持使用有符号零进行检查
        tgtdiv, tgtrem = np.array(tgt, dtype=float).T
        tgtdiv = np.where((tgtdiv == 0.0) & ((b < 0) ^ (a < 0)), -0.0, tgtdiv)
        tgtrem = np.where((tgtrem == 0.0) & (b < 0), -0.0, tgtrem)

        # 再次遍历 floordiv_and_mod 和 divmod 函数
        for op in [floordiv_and_mod, divmod]:
            # 遍历所有浮点数数据类型
            for dt in np.typecodes['Float']:
                msg = 'op: %s, dtype: %s' % (op.__name__, dt)
                fa = a.astype(dt)
                fb = b.astype(dt)
                # 使用列表推导式确保 fa 和 fb 是标量
                div, rem = zip(*[op(a_, b_) for  a_, b_ in zip(fa, fb)])
                # 使用 assert_equal 断言检查结果与预期的精确整数结果
                assert_equal(div, tgtdiv, err_msg=msg)
                assert_equal(rem, tgtrem, err_msg=msg)
    def test_float_modulus_roundoff(self):
        # 测试用例函数：测试浮点数取模的舍入误差
        dt = np.typecodes['Float']
        # 循环测试不同操作和数据类型的组合
        for op in [floordiv_and_mod, divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                # 循环测试不同的符号组合
                for sg1, sg2 in itertools.product((+1, -1), (+1, -1)):
                    fmt = 'op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s'
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    # 创建浮点数数组 a 和 b
                    a = np.array(sg1*78*6e-8, dtype=dt1)[()]
                    b = np.array(sg2*6e-8, dtype=dt2)[()]
                    # 执行取整除法和取模操作
                    div, rem = op(a, b)
                    # 使用 fmod 操作进行相等断言
                    assert_equal(div*b + rem, a, err_msg=msg)
                    # 检查余数的符号和大小
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    def test_float_modulus_corner_cases(self):
        # 测试用例函数：测试浮点数取模的边界情况
        # 检查余数的大小
        for dt in np.typecodes['Float']:
            b = np.array(1.0, dtype=dt)
            a = np.nextafter(np.array(0.0, dtype=dt), -b)
            # 执行取模操作
            rem = operator.mod(a, b)
            assert_(rem <= b, 'dt: %s' % dt)
            rem = operator.mod(-a, -b)
            assert_(rem >= -b, 'dt: %s' % dt)

        # 检查 NaN 和 Inf
        with suppress_warnings() as sup:
            # 忽略特定的警告信息
            sup.filter(RuntimeWarning, "invalid value encountered in remainder")
            sup.filter(RuntimeWarning, "divide by zero encountered in remainder")
            sup.filter(RuntimeWarning, "divide by zero encountered in floor_divide")
            sup.filter(RuntimeWarning, "divide by zero encountered in divmod")
            sup.filter(RuntimeWarning, "invalid value encountered in divmod")
            for dt in np.typecodes['Float']:
                fone = np.array(1.0, dtype=dt)
                fzer = np.array(0.0, dtype=dt)
                finf = np.array(np.inf, dtype=dt)
                fnan = np.array(np.nan, dtype=dt)
                # 执行取模操作并进行断言
                rem = operator.mod(fone, fzer)
                assert_(np.isnan(rem), 'dt: %s' % dt)
                # MSVC 2008 在此返回 NaN，因此禁用此检查。
                # rem = operator.mod(fone, finf)
                # assert_(rem == fone, 'dt: %s' % dt)
                rem = operator.mod(fone, fnan)
                assert_(np.isnan(rem), 'dt: %s' % dt)
                rem = operator.mod(finf, fone)
                assert_(np.isnan(rem), 'dt: %s' % dt)
                # 循环测试不同操作的组合
                for op in [floordiv_and_mod, divmod]:
                    div, mod = op(fone, fzer)
                    assert_(np.isinf(div)) and assert_(np.isnan(mod))
    def test_inplace_floordiv_handling(self):
        # 定义一个测试函数，用于测试处理就地整除赋值操作的情况
        # 此测试仅适用于就地整除赋值 //=，因为输出类型会提升为 float，而 uint64 不适用于 float
        # 创建一个包含两个元素的 np.int64 数组 a
        a = np.array([1, 2], np.int64)
        # 创建一个包含两个元素的 np.uint64 数组 b
        b = np.array([1, 2], np.uint64)
        # 使用 pytest 检查是否会抛出 TypeError 异常，并匹配特定错误信息
        with pytest.raises(TypeError,
                match=r"Cannot cast ufunc 'floor_divide' output from"):
            # 对数组 a 执行就地整除赋值操作 //= b
            a //= b
class TestComplexDivision:
    def test_zero_division(self):
        # 忽略所有 NumPy 的错误和警告
        with np.errstate(all="ignore"):
            # 对于每种复数类型进行测试
            for t in [np.complex64, np.complex128]:
                # 创建复数类型对象a为0.0
                a = t(0.0)
                # 创建复数类型对象b为1.0
                b = t(1.0)
                # 断言 b/a 为无穷大
                assert_(np.isinf(b/a))
                # 将b设为复数类型，实部和虚部均为正无穷大
                b = t(complex(np.inf, np.inf))
                # 断言 b/a 为无穷大
                assert_(np.isinf(b/a))
                # 将b设为复数类型，实部为正无穷大，虚部为NaN
                b = t(complex(np.inf, np.nan))
                # 断言 b/a 为无穷大
                assert_(np.isinf(b/a))
                # 将b设为复数类型，实部为NaN，虚部为正无穷大
                b = t(complex(np.nan, np.inf))
                # 断言 b/a 为无穷大
                assert_(np.isinf(b/a))
                # 将b设为复数类型，实部和虚部均为NaN
                b = t(complex(np.nan, np.nan))
                # 断言 b/a 为NaN
                assert_(np.isnan(b/a))
                # 将b设为0.0
                b = t(0.)
                # 断言 b/a 为NaN
                assert_(np.isnan(b/a))

    def test_signed_zeros(self):
        # 忽略所有 NumPy 的错误和警告
        with np.errstate(all="ignore"):
            # 对于每种复数类型进行测试
            for t in [np.complex64, np.complex128]:
                # 定义测试数据：(分子实部, 分子虚部), (分母实部, 分母虚部), (期望结果实部, 期望结果虚部)
                data = (
                    (( 0.0,-1.0), ( 0.0, 1.0), (-1.0,-0.0)),
                    (( 0.0,-1.0), ( 0.0,-1.0), ( 1.0,-0.0)),
                    (( 0.0,-1.0), (-0.0,-1.0), ( 1.0, 0.0)),
                    (( 0.0,-1.0), (-0.0, 1.0), (-1.0, 0.0)),
                    (( 0.0, 1.0), ( 0.0,-1.0), (-1.0, 0.0)),
                    (( 0.0,-1.0), ( 0.0,-1.0), ( 1.0,-0.0)),
                    ((-0.0,-1.0), ( 0.0,-1.0), ( 1.0,-0.0)),
                    ((-0.0, 1.0), ( 0.0,-1.0), (-1.0,-0.0))
                )
                # 对于每组测试数据
                for cases in data:
                    # 取出分子和分母
                    n = cases[0]
                    d = cases[1]
                    # 取出期望的实部和虚部
                    ex = cases[2]
                    # 计算结果
                    result = t(complex(n[0], n[1])) / t(complex(d[0], d[1]))
                    # 分别断言实部和虚部与期望值相等，以避免在数组环境下对符号零的比较问题
                    assert_equal(result.real, ex[0])
                    assert_equal(result.imag, ex[1])
    def test_branches(self):
        # 忽略 NumPy 中的所有错误（errorstate）
        with np.errstate(all="ignore"):
            # 对于每种复数类型进行迭代测试
            for t in [np.complex64, np.complex128]:
                # 定义用于测试的数据列表，每个元素是一个元组 (numerator, denominator, expected)
                data = list()

                # 触发条件分支：real(fabs(denom)) > imag(fabs(denom))
                # 然后执行 else 分支，因为两者都不等于 0
                data.append(((2.0, 1.0), (2.0, 1.0), (1.0, 0.0)))

                # 触发条件分支：real(fabs(denom)) > imag(fabs(denom))
                # 然后执行 if 分支，因为两者都等于 0
                # 这部分的测试被移至 test_zero_division()，因此在此跳过

                # 触发 else if 分支：real(fabs(denom)) < imag(fabs(denom))
                data.append(((1.0, 2.0), (1.0, 2.0), (1.0, 0.0)))

                # 遍历测试数据集合
                for cases in data:
                    # 分别取出 numerator、denominator 和 expected
                    n = cases[0]
                    d = cases[1]
                    ex = cases[2]
                    # 计算结果
                    result = t(complex(n[0], n[1])) / t(complex(d[0], d[1]))
                    # 分别检查实部和虚部，避免在数组上下文中比较，因为它不考虑有符号的零
                    assert_equal(result.real, ex[0])
                    assert_equal(result.imag, ex[1])
class TestConversion:
    # 测试从长浮点数到整数的转换
    def test_int_from_long(self):
        # 创建长浮点数列表
        l = [1e6, 1e12, 1e18, -1e6, -1e12, -1e18]
        # 创建预期整数列表
        li = [10**6, 10**12, 10**18, -10**6, -10**12, -10**18]
        # 对每种数据类型进行测试：None, np.float64, np.int64
        for T in [None, np.float64, np.int64]:
            # 使用指定数据类型创建 NumPy 数组
            a = np.array(l, dtype=T)
            # 断言转换后的每个元素与预期整数相等
            assert_equal([int(_m) for _m in a], li)

        # 使用 np.uint64 类型创建 NumPy 数组
        a = np.array(l[:3], dtype=np.uint64)
        # 断言转换后的每个元素与预期整数相等
        assert_equal([int(_m) for _m in a], li[:3])

    # 测试 np.iinfo 中的长整数值
    def test_iinfo_long_values(self):
        # 对于 'bBhH' 中的每个类型码，测试溢出错误
        for code in 'bBhH':
            with pytest.raises(OverflowError):
                np.array(np.iinfo(code).max + 1, dtype=code)

        # 对于所有整数类型码中的每个类型码，测试最大值与预期最大值相等
        for code in np.typecodes['AllInteger']:
            res = np.array(np.iinfo(code).max, dtype=code)
            tgt = np.iinfo(code).max
            assert_(res == tgt)

        # 对于所有整数类型码中的每个类型码，测试使用 np.dtype 创建的最大值与预期最大值相等
        for code in np.typecodes['AllInteger']:
            res = np.dtype(code).type(np.iinfo(code).max)
            tgt = np.iinfo(code).max
            assert_(res == tgt)

    # 测试整数溢出行为
    def test_int_raise_behaviour(self):
        # 定义溢出错误函数
        def overflow_error_func(dtype):
            dtype(np.iinfo(dtype).max + 1)

        # 对于指定的数据类型，测试是否抛出溢出错误
        for code in [np.int_, np.uint, np.longlong, np.ulonglong]:
            assert_raises(OverflowError, overflow_error_func, code)

    # 测试从无限长双精度浮点数到整数的转换
    def test_int_from_infinite_longdouble(self):
        # gh-627
        # 创建一个无限大的长双精度浮点数，预期抛出溢出错误
        x = np.longdouble(np.inf)
        assert_raises(OverflowError, int, x)
        # 使用抑制警告上下文管理器
        with suppress_warnings() as sup:
            sup.record(ComplexWarning)
            x = np.clongdouble(np.inf)
            assert_raises(OverflowError, int, x)
            assert_equal(len(sup.log), 1)

    # 仅在 PyPy 环境下运行的测试
    @pytest.mark.skipif(not IS_PYPY, reason="Test is PyPy only (gh-9972)")
    def test_int_from_infinite_longdouble___int__(self):
        # 创建一个无限大的长双精度浮点数，预期抛出溢出错误
        x = np.longdouble(np.inf)
        assert_raises(OverflowError, x.__int__)
        # 使用抑制警告上下文管理器
        with suppress_warnings() as sup:
            sup.record(ComplexWarning)
            x = np.clongdouble(np.inf)
            assert_raises(OverflowError, x.__int__)
            assert_equal(len(sup.log), 1)

    # 根据条件跳过测试
    @pytest.mark.skipif(np.finfo(np.double) == np.finfo(np.longdouble),
                        reason="long double is same as double")
    @pytest.mark.skipif(platform.machine().startswith("ppc"),
                        reason="IBM double double")
    def test_int_from_huge_longdouble(self):
        # 生成一个长双精度浮点数，其值将溢出一个双精度浮点数
        # 使用避免 Darwin pow 函数中的错误的指数
        exp = np.finfo(np.double).maxexp - 1
        huge_ld = 2 * 1234 * np.longdouble(2) ** exp
        huge_i = 2 * 1234 * 2 ** exp
        assert_(huge_ld != np.inf)
        assert_equal(int(huge_ld), huge_i)

    # 测试从长双精度浮点数到整数的转换
    def test_int_from_longdouble(self):
        # 创建一个长双精度浮点数，将其转换为整数并断言结果
        x = np.longdouble(1.5)
        assert_equal(int(x), 1)
        x = np.longdouble(-10.5)
        assert_equal(int(x), -10)
    def test_numpy_scalar_relational_operators(self):
        # 对所有整数类型进行测试
        for dt1 in np.typecodes['AllInteger']:
            # 断言 1 大于数组中的零，并给出失败类型信息
            assert_(1 > np.array(0, dtype=dt1)[()], "type %s failed" % (dt1,))
            # 断言 1 不小于数组中的零，并给出失败类型信息
            assert_(not 1 < np.array(0, dtype=dt1)[()], "type %s failed" % (dt1,))

            # 嵌套循环，对第二个整数类型进行测试
            for dt2 in np.typecodes['AllInteger']:
                # 断言以 dt1 类型的数组中的 1 大于以 dt2 类型的数组中的 0，并给出失败类型信息
                assert_(np.array(1, dtype=dt1)[()] > np.array(0, dtype=dt2)[()],
                        "type %s and %s failed" % (dt1, dt2))
                # 断言以 dt1 类型的数组中的 1 不小于以 dt2 类型的数组中的 0，并给出失败类型信息
                assert_(not np.array(1, dtype=dt1)[()] < np.array(0, dtype=dt2)[()],
                        "type %s and %s failed" % (dt1, dt2))

        # 对无符号整数进行测试
        for dt1 in 'BHILQP':
            # 断言 -1 小于以 dt1 类型的数组中的 1，并给出失败类型信息
            assert_(-1 < np.array(1, dtype=dt1)[()], "type %s failed" % (dt1,))
            # 断言 -1 不大于以 dt1 类型的数组中的 1，并给出失败类型信息
            assert_(not -1 > np.array(1, dtype=dt1)[()], "type %s failed" % (dt1,))
            # 断言 -1 不等于以 dt1 类型的数组中的 1，并给出失败类型信息
            assert_(-1 != np.array(1, dtype=dt1)[()], "type %s failed" % (dt1,))

            # 嵌套循环，测试无符号与有符号整数的比较
            for dt2 in 'bhilqp':
                # 断言以 dt1 类型的数组中的 1 大于以 dt2 类型的数组中的 -1，并给出失败类型信息
                assert_(np.array(1, dtype=dt1)[()] > np.array(-1, dtype=dt2)[()],
                        "type %s and %s failed" % (dt1, dt2))
                # 断言以 dt1 类型的数组中的 1 不小于以 dt2 类型的数组中的 -1，并给出失败类型信息
                assert_(not np.array(1, dtype=dt1)[()] < np.array(-1, dtype=dt2)[()],
                        "type %s and %s failed" % (dt1, dt2))
                # 断言以 dt1 类型的数组中的 1 不等于以 dt2 类型的数组中的 -1，并给出失败类型信息
                assert_(np.array(1, dtype=dt1)[()] != np.array(-1, dtype=dt2)[()],
                        "type %s and %s failed" % (dt1, dt2))

        # 对有符号整数和浮点数进行测试
        for dt1 in 'bhlqp' + np.typecodes['Float']:
            # 断言 1 大于以 dt1 类型的数组中的 -1，并给出失败类型信息
            assert_(1 > np.array(-1, dtype=dt1)[()], "type %s failed" % (dt1,))
            # 断言 1 不小于以 dt1 类型的数组中的 -1，并给出失败类型信息
            assert_(not 1 < np.array(-1, dtype=dt1)[()], "type %s failed" % (dt1,))
            # 断言 -1 等于以 dt1 类型的数组中的 -1，并给出失败类型信息
            assert_(-1 == np.array(-1, dtype=dt1)[()], "type %s failed" % (dt1,))

            # 嵌套循环，对第二个有符号整数和浮点数类型进行测试
            for dt2 in 'bhlqp' + np.typecodes['Float']:
                # 断言以 dt1 类型的数组中的 1 大于以 dt2 类型的数组中的 -1，并给出失败类型信息
                assert_(np.array(1, dtype=dt1)[()] > np.array(-1, dtype=dt2)[()],
                        "type %s and %s failed" % (dt1, dt2))
                # 断言以 dt1 类型的数组中的 1 不小于以 dt2 类型的数组中的 -1，并给出失败类型信息
                assert_(not np.array(1, dtype=dt1)[()] < np.array(-1, dtype=dt2)[()],
                        "type %s and %s failed" % (dt1, dt2))
                # 断言以 dt1 类型的数组中的 -1 等于以 dt2 类型的数组中的 -1，并给出失败类型信息
                assert_(np.array(-1, dtype=dt1)[()] == np.array(-1, dtype=dt2)[()],
                        "type %s and %s failed" % (dt1, dt2))
    def test_scalar_comparison_to_none(self):
        # 测试标量与 None 的比较行为
        # 标量应该返回 False 而不是产生警告。
        # 这些比较被 pep8 标记，我们忽略这一点。
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', FutureWarning)
            # 断言：np.float32(1) 不等于 None
            assert_(not np.float32(1) == None)
            # 断言：np.str_('test') 不等于 None
            assert_(not np.str_('test') == None)
            # 这个比较可能不确定 (见下文):
            assert_(not np.datetime64('NaT') == None)

            # 断言：np.float32(1) 等于 None
            assert_(np.float32(1) != None)
            # 断言：np.str_('test') 等于 None
            assert_(np.str_('test') != None)
            # 这个比较可能不确定 (见下文):
            assert_(np.datetime64('NaT') != None)
        # 断言：警告的数量为 0
        assert_(len(w) == 0)

        # 为了文档目的，解释为什么 datetime 是不确定的。
        # 在弃用时没有行为变化，但在弃用时必须考虑这一点。
        assert_(np.equal(np.datetime64('NaT'), None))
# 测试类 TestRepr，用于验证数据类型的字符串表示和反向操作的正确性
class TestRepr:
    # 私有方法，测试给定数据类型 t 的字符串表示和反向操作
    def _test_type_repr(self, t):
        # 获取数据类型 t 的浮点数信息
        finfo = np.finfo(t)
        # 计算最后一个小数位索引
        last_fraction_bit_idx = finfo.nexp + finfo.nmant
        # 计算最后一个指数位索引
        last_exponent_bit_idx = finfo.nexp
        # 计算存储字节数
        storage_bytes = np.dtype(t).itemsize * 8
        
        # 在下面的循环中，处理两种情况：小的非规格化数和小的规格化数
        for which in ['small denorm', 'small norm']:
            # 创建一个存储字节数量的零数组，数据类型为 np.uint8
            constr = np.array([0x00] * storage_bytes, dtype=np.uint8)
            
            # 根据 which 的值设置特定位的值，来模拟特定的浮点数表示
            if which == 'small denorm':
                byte = last_fraction_bit_idx // 8
                bytebit = 7 - (last_fraction_bit_idx % 8)
                constr[byte] = 1 << bytebit
            elif which == 'small norm':
                byte = last_exponent_bit_idx // 8
                bytebit = 7 - (last_exponent_bit_idx % 8)
                constr[byte] = 1 << bytebit
            else:
                raise ValueError('hmm')
            
            # 将构造的数组视图转换为类型 t 的值
            val = constr.view(t)[0]
            # 获取值 val 的字符串表示
            val_repr = repr(val)
            # 通过 eval 将字符串表示转换回类型 t 的值
            val2 = t(eval(val_repr))
            
            # 断言原始值 val 和经过字符串表示转换后的值 val2 相等
            if not (val2 == 0 and val < 1e-100):
                assert_equal(val, val2)
    
    # 测试浮点数类型的字符串表示和反向操作
    def test_float_repr(self):
        # 对每种浮点数类型进行测试：单精度浮点数和双精度浮点数
        for t in [np.float32, np.float64]:
            self._test_type_repr(t)

# 如果不是在 PyPy 环境下执行以下代码
if not IS_PYPY:
    # 测试类 TestSizeOf，用于验证数据类型的字节大小与 sys.getsizeof() 方法的比较
    class TestSizeOf:
        
        # 测试每种数据类型的实际字节大小大于其 nbyte 属性的断言
        def test_equal_nbytes(self):
            # 对于 types 中的每种数据类型
            for type in types:
                # 创建类型为 type 的零值 x
                x = type(0)
                # 断言 x 的实际字节大小大于其 nbyte 属性
                assert_(sys.getsizeof(x) > x.nbytes)
        
        # 测试对非法输入的错误处理
        def test_error(self):
            # 创建一个 np.float32 类型的空对象 d
            d = np.float32()
            # 断言调用 d.__sizeof__("a") 时会引发 TypeError 异常
            assert_raises(TypeError, d.__sizeof__, "a")
    def test_no_seq_repeat_basic_array_like(self):
        # Test that an array-like which does not know how to be multiplied
        # does not attempt sequence repeat (raise TypeError).
        # See also gh-7428.
        class ArrayLike:
            def __init__(self, arr):
                self.arr = arr

            def __array__(self, dtype=None, copy=None):
                return self.arr
        
        # 对于每一个类似数组的对象（如ArrayLike和memoryview），进行测试
        for arr_like in (ArrayLike(np.ones(3)), memoryview(np.ones(3))):
            # 检查乘法操作是否能够正确进行
            assert_array_equal(arr_like * np.float32(3.), np.full(3, 3.))
            assert_array_equal(np.float32(3.) * arr_like, np.full(3, 3.))
            assert_array_equal(arr_like * np.int_(3), np.full(3, 3))
            assert_array_equal(np.int_(3) * arr_like, np.full(3, 3))
class TestNegative:
    def test_exceptions(self):
        # 创建一个布尔类型的 numpy 数组 a，值为 1.0
        a = np.ones((), dtype=np.bool)[()]
        # 断言调用 operator.neg(a) 会抛出 TypeError 异常
        assert_raises(TypeError, operator.neg, a)

    def test_result(self):
        # 获取所有整数和浮点数的 numpy 类型码
        types = np.typecodes['AllInteger'] + np.typecodes['AllFloat']
        # 在抑制特定类型的警告的上下文中进行操作
        with suppress_warnings() as sup:
            # 过滤运行时警告
            sup.filter(RuntimeWarning)
            # 遍历每种数据类型
            for dt in types:
                # 创建一个包含单个元素的 numpy 数组 a，数据类型为 dt
                a = np.ones((), dtype=dt)[()]
                # 如果数据类型 dt 是无符号整数类型
                if dt in np.typecodes['UnsignedInteger']:
                    # 获取该数据类型的最大值
                    st = np.dtype(dt).type
                    max = st(np.iinfo(dt).max)
                    # 断言 operator.neg(a) 的结果等于最大值 max
                    assert_equal(operator.neg(a), max)
                else:
                    # 断言 operator.neg(a) + a 的结果等于 0
                    assert_equal(operator.neg(a) + a, 0)

class TestSubtract:
    def test_exceptions(self):
        # 创建一个布尔类型的 numpy 数组 a，值为 1.0
        a = np.ones((), dtype=np.bool)[()]
        # 断言调用 operator.sub(a, a) 会抛出 TypeError 异常
        assert_raises(TypeError, operator.sub, a, a)

    def test_result(self):
        # 获取所有整数和浮点数的 numpy 类型码
        types = np.typecodes['AllInteger'] + np.typecodes['AllFloat']
        # 在抑制特定类型的警告的上下文中进行操作
        with suppress_warnings() as sup:
            # 过滤运行时警告
            sup.filter(RuntimeWarning)
            # 遍历每种数据类型
            for dt in types:
                # 创建一个包含单个元素的 numpy 数组 a，数据类型为 dt
                a = np.ones((), dtype=dt)[()]
                # 断言 operator.sub(a, a) 的结果等于 0
                assert_equal(operator.sub(a, a), 0)


class TestAbs:
    def _test_abs_func(self, absfunc, test_dtype):
        # 创建一个 test_dtype 类型的对象 x，值为 -1.5
        x = test_dtype(-1.5)
        # 断言调用 absfunc(x) 的结果等于 1.5
        assert_equal(absfunc(x), 1.5)
        # 创建一个 test_dtype 类型的对象 x，值为 0.0
        x = test_dtype(0.0)
        # 调用 absfunc(x) 并将结果赋给变量 res
        res = absfunc(x)
        # 使用 assert_equal() 检查零的符号性
        assert_equal(res, 0.0)
        # 创建一个 test_dtype 类型的对象 x，值为 -0.0
        x = test_dtype(-0.0)
        # 调用 absfunc(x) 并将结果赋给变量 res
        res = absfunc(x)
        # 断言调用 absfunc(x) 的结果等于 0.0
        assert_equal(res, 0.0)

        # 创建一个 test_dtype 类型的对象 x，值为该类型的最大值
        x = test_dtype(np.finfo(test_dtype).max)
        # 断言调用 absfunc(x) 的结果等于 x 的实部
        assert_equal(absfunc(x), x.real)

        # 在抑制特定类型的警告的上下文中进行操作
        with suppress_warnings() as sup:
            # 过滤用户警告
            sup.filter(UserWarning)
            # 创建一个 test_dtype 类型的对象 x，值为该类型的最小非零数
            x = test_dtype(np.finfo(test_dtype).tiny)
            # 断言调用 absfunc(x) 的结果等于 x 的实部
            assert_equal(absfunc(x), x.real)

        # 创建一个 test_dtype 类型的对象 x，值为该类型的最小值
        x = test_dtype(np.finfo(test_dtype).min)
        # 断言调用 absfunc(x) 的结果等于 -x 的实部
        assert_equal(absfunc(x), -x.real)

    @pytest.mark.parametrize("dtype", floating_types + complex_floating_types)
    def test_builtin_abs(self, dtype):
        # 如果在 cygwin 平台上，并且 dtype 是 np.clongdouble 类型，并且运行时版本低于 3.3
        if (
                sys.platform == "cygwin" and dtype == np.clongdouble and
                (
                    _pep440.parse(platform.release().split("-")[0])
                    < _pep440.Version("3.3.0")
                )
        ):
            # 标记此测试为预期失败，原因是在 cygwin 平台上的 double 精度计算小数的情况
            pytest.xfail(
                reason="absl is computed in double precision on cygwin < 3.3"
            )
        # 调用 self._test_abs_func(abs, dtype)
        self._test_abs_func(abs, dtype)

    @pytest.mark.parametrize("dtype", floating_types + complex_floating_types)
    def test_numpy_abs(self, dtype):
        # 如果在 cygwin 平台上，并且 dtype 是 np.clongdouble 类型，并且运行时版本低于 3.3
        if (
                sys.platform == "cygwin" and dtype == np.clongdouble and
                (
                    _pep440.parse(platform.release().split("-")[0])
                    < _pep440.Version("3.3.0")
                )
        ):
            # 标记此测试为预期失败，原因是在 cygwin 平台上的 double 精度计算小数的情况
            pytest.xfail(
                reason="absl is computed in double precision on cygwin < 3.3"
            )
        # 调用 self._test_abs_func(np.abs, dtype)
        self._test_abs_func(np.abs, dtype)

class TestBitShifts:

    @pytest.mark.parametrize('type_code', np.typecodes['AllInteger'])
    # 使用 pytest 的参数化装饰器标记测试函数，对每个操作符进行测试
    @pytest.mark.parametrize('op',
        [operator.rshift, operator.lshift], ids=['>>', '<<'])
    # 定义测试函数，测试位移操作，参数包括类型码和操作符
    def test_shift_all_bits(self, type_code, op):
        """Shifts where the shift amount is the width of the type or wider """
        # 标题注释：测试位移操作，其中位移量等于类型宽度或更宽的情况
        # 从类型码创建 NumPy 数据类型对象
        dt = np.dtype(type_code)
        # 计算数据类型的位数
        nbits = dt.itemsize * 8
        # 对于每个值（5 和 -5）
        for val in [5, -5]:
            # 对于每个位移量（类型宽度和比类型宽度大 4 个单位）
            for shift in [nbits, nbits + 4]:
                # 将值转换为指定的 NumPy 标量类型
                val_scl = np.array(val).astype(dt)[()]
                # 创建指定类型的位移量
                shift_scl = dt.type(shift)
                # 执行位移操作
                res_scl = op(val_scl, shift_scl)
                # 如果值为负且操作为右移位操作
                if val_scl < 0 and op is operator.rshift:
                    # 断言结果保留了符号位
                    assert_equal(res_scl, -1)
                else:
                    # 否则断言结果为 0
                    assert_equal(res_scl, 0)

                # 对数组执行相同的位移操作，验证标量与数组结果一致性
                val_arr = np.array([val_scl]*32, dtype=dt)
                shift_arr = np.array([shift]*32, dtype=dt)
                res_arr = op(val_arr, shift_arr)
                assert_equal(res_arr, res_scl)
class TestHash:
    # 测试整数的哈希值
    @pytest.mark.parametrize("type_code", np.typecodes['AllInteger'])
    def test_integer_hashes(self, type_code):
        # 获取特定类型码的标量类型
        scalar = np.dtype(type_code).type
        # 对于范围内的整数，验证其哈希值与标量的哈希值一致
        for i in range(128):
            assert hash(i) == hash(scalar(i))

    # 测试浮点数和复数的哈希值
    @pytest.mark.parametrize("type_code", np.typecodes['AllFloat'])
    def test_float_and_complex_hashes(self, type_code):
        # 获取特定类型码的标量类型
        scalar = np.dtype(type_code).type
        # 对于给定的值列表，验证其转换后的哈希值与原始值的哈希值一致性
        for val in [np.pi, np.inf, 3, 6.]:
            numpy_val = scalar(val)
            # 如果是复数类型，将其转换为 Python 复数
            if numpy_val.dtype.kind == 'c':
                val = complex(numpy_val)
            else:
                val = float(numpy_val)
            assert val == numpy_val
            assert hash(val) == hash(numpy_val)

        if hash(float(np.nan)) != hash(float(np.nan)):
            # 如果 Python 区分不同的 NaN，则验证 NumPy 也做到了 (gh-18833)
            assert hash(scalar(np.nan)) != hash(scalar(np.nan))

    # 测试复数的哈希值
    @pytest.mark.parametrize("type_code", np.typecodes['Complex'])
    def test_complex_hashes(self, type_code):
        # 获取特定类型码的标量类型
        scalar = np.dtype(type_code).type
        # 验证特定复数值的哈希值
        for val in [np.pi+1j, np.inf-3j, 3j, 6.+1j]:
            numpy_val = scalar(val)
            assert hash(complex(numpy_val)) == hash(numpy_val)


# 管理递归限制的上下文管理器
@contextlib.contextmanager
def recursionlimit(n):
    # 获取当前递归限制
    o = sys.getrecursionlimit()
    try:
        # 设置新的递归限制
        sys.setrecursionlimit(n)
        yield
    finally:
        # 恢复原始的递归限制
        sys.setrecursionlimit(o)


# 使用假设来测试对象作为左操作数的运算符行为
@given(sampled_from(objecty_things),
       sampled_from(reasonable_operators_for_scalars),
       sampled_from(types))
def test_operator_object_left(o, op, type_):
    try:
        # 设置递归限制为200并测试操作
        with recursionlimit(200):
            op(o, type_(1))
    except TypeError:
        pass


# 使用假设来测试对象作为右操作数的运算符行为
@given(sampled_from(objecty_things),
       sampled_from(reasonable_operators_for_scalars),
       sampled_from(types))
def test_operator_object_right(o, op, type_):
    try:
        # 设置递归限制为200并测试操作
        with recursionlimit(200):
            op(type_(1), o)
    except TypeError:
        pass


# 使用假设来测试标量之间的运算符行为
@given(sampled_from(reasonable_operators_for_scalars),
       sampled_from(types),
       sampled_from(types))
def test_operator_scalars(op, type1, type2):
    try:
        # 测试标量操作并捕获可能的类型错误
        op(type1(1), type2(1))
    except TypeError:
        pass


# 使用假设来测试长双精度数与对象之间的运算符行为
@pytest.mark.parametrize("op", reasonable_operators_for_scalars)
@pytest.mark.parametrize("sctype", [np.longdouble, np.clongdouble])
def test_longdouble_operators_with_obj(sctype, op):
    # 这个测试曾经比较棘手，因为 NumPy 通常会通过 `np.asarray()` 使用 ufunc，可能会导致递归问题
    # 这里验证长双精度数与对象之间的运算是否正常
    pass  # 实际测试逻辑在代码注释中有详细说明，但这里不执行具体测试
    # 尝试执行 op(sctype(3), None)，期望没有抛出 TypeError 异常
    try:
        op(sctype(3), None)
    # 如果抛出 TypeError 异常，则捕获并忽略
    except TypeError:
        pass
    
    # 尝试执行 op(None, sctype(3))，期望没有抛出 TypeError 异常
    try:
        op(None, sctype(3))
    # 如果抛出 TypeError 异常，则捕获并忽略
    except TypeError:
        pass
@pytest.mark.parametrize("op", reasonable_operators_for_scalars)
@pytest.mark.parametrize("sctype", [np.longdouble, np.clongdouble])
@np.errstate(all="ignore")
# 定义测试函数，测试长双精度和复长双精度的运算符，忽略所有错误状态
def test_longdouble_operators_with_large_int(sctype, op):
    # (See `test_longdouble_operators_with_obj` for why longdouble is special)
    # 根据 `test_longdouble_operators_with_obj` 查看为何 longdouble 特别
    # NEP 50 means that the result is clearly a (c)longdouble here:
    # NEP 50 意味着这里结果显然是 (复)长双精度
    if sctype == np.clongdouble and op in [operator.mod, operator.floordiv]:
        # The above operators are not support for complex though...
        # 但上述运算符不支持复数类型...
        with pytest.raises(TypeError):
            op(sctype(3), 2**64)
        with pytest.raises(TypeError):
            op(sctype(3), 2**64)
    else:
        assert op(sctype(3), -2**64) == op(sctype(3), sctype(-2**64))
        assert op(2**64, sctype(3)) == op(sctype(2**64), sctype(3))


@pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
@pytest.mark.parametrize("operation", [
        lambda min, max: max + max,
        lambda min, max: min - max,
        lambda min, max: max * max], ids=["+", "-", "*"])
# 测试整数类型的溢出操作，包括加法、减法和乘法
def test_scalar_integer_operation_overflow(dtype, operation):
    st = np.dtype(dtype).type
    min = st(np.iinfo(dtype).min)
    max = st(np.iinfo(dtype).max)

    with pytest.warns(RuntimeWarning, match="overflow encountered"):
        operation(min, max)


@pytest.mark.parametrize("dtype", np.typecodes["Integer"])
@pytest.mark.parametrize("operation", [
        lambda min, neg_1: -min,
        lambda min, neg_1: abs(min),
        lambda min, neg_1: min * neg_1,
        pytest.param(lambda min, neg_1: min // neg_1,
            marks=pytest.mark.skip(reason="broken on some platforms"))],
        ids=["neg", "abs", "*", "//"])
# 测试带符号整数的溢出情况，包括负数、绝对值、乘法和除法
def test_scalar_signed_integer_overflow(dtype, operation):
    # The minimum signed integer can "overflow" for some additional operations
    # 最小的有符号整数可能会在一些额外操作中"溢出"
    st = np.dtype(dtype).type
    min = st(np.iinfo(dtype).min)
    neg_1 = st(-1)

    with pytest.warns(RuntimeWarning, match="overflow encountered"):
        operation(min, neg_1)


@pytest.mark.parametrize("dtype", np.typecodes["UnsignedInteger"])
# 测试无符号整数的溢出情况
def test_scalar_unsigned_integer_overflow(dtype):
    val = np.dtype(dtype).type(8)
    with pytest.warns(RuntimeWarning, match="overflow encountered"):
        -val

    zero = np.dtype(dtype).type(0)
    -zero  # does not warn


@pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
@pytest.mark.parametrize("operation", [
        lambda val, zero: val // zero,
        lambda val, zero: val % zero, ], ids=["//", "%"])
# 测试整数除零操作，包括整数除法和取模
def test_scalar_integer_operation_divbyzero(dtype, operation):
    st = np.dtype(dtype).type
    val = st(100)
    zero = st(0)

    with pytest.warns(RuntimeWarning, match="divide by zero"):
        operation(val, zero)


ops_with_names = [
    ("__lt__", "__gt__", operator.lt, True),
    ("__le__", "__ge__", operator.le, True),
    ("__eq__", "__eq__", operator.eq, True),
    # Note __op__ and __rop__ may be identical here:
    # 注意这里 __op__ 和 __rop__ 可能是相同的：
    ("__ne__", "__ne__", operator.ne, True),
    # 创建元组，每个元组包含两个特殊方法名和对应的操作函数以及一个布尔值
    ("__gt__", "__lt__", operator.gt, True),
    # 创建元组，每个元组包含两个特殊方法名和对应的操作函数以及一个布尔值
    ("__ge__", "__le__", operator.ge, True),
    # 创建元组，每个元组包含两个特殊方法名和对应的操作函数以及一个布尔值
    ("__floordiv__", "__rfloordiv__", operator.floordiv, False),
    # 创建元组，每个元组包含两个特殊方法名和对应的操作函数以及一个布尔值
    ("__truediv__", "__rtruediv__", operator.truediv, False),
    # 创建元组，每个元组包含两个特殊方法名和对应的操作函数以及一个布尔值
    ("__add__", "__radd__", operator.add, False),
    # 创建元组，每个元组包含两个特殊方法名和对应的操作函数以及一个布尔值
    ("__mod__", "__rmod__", operator.mod, False),
    # 创建元组，每个元组包含两个特殊方法名和对应的操作函数以及一个布尔值
    ("__mul__", "__rmul__", operator.mul, False),
    # 创建元组，每个元组包含两个特殊方法名和对应的操作函数以及一个布尔值
    ("__pow__", "__rpow__", operator.pow, False),
    # 创建元组，每个元组包含两个特殊方法名和对应的操作函数以及一个布尔值
    ("__sub__", "__rsub__", operator.sub, False),
@pytest.mark.parametrize(["__op__", "__rop__", "op", "cmp"], ops_with_names)
# 使用pytest的@parametrize装饰器，对参数进行多组化，使用ops_with_names作为参数
@pytest.mark.parametrize("sctype", [np.float32, np.float64, np.longdouble])
# 使用pytest的@parametrize装饰器，对参数进行多组化，测试np.float32、np.float64和np.longdouble类型
def test_subclass_deferral(sctype, __op__, __rop__, op, cmp):
    """
    This test covers scalar subclass deferral.  Note that this is exceedingly
    complicated, especially since it tends to fall back to the array paths and
    these additionally add the "array priority" mechanism.

    The behaviour was modified subtly in 1.22 (to make it closer to how Python
    scalars work).  Due to its complexity and the fact that subclassing NumPy
    scalars is probably a bad idea to begin with.  There is probably room
    for adjustments here.
    """
    class myf_simple1(sctype):
        pass

    class myf_simple2(sctype):
        pass

    def op_func(self, other):
        return __op__
    # 定义一个函数op_func，返回参数__op__

    def rop_func(self, other):
        return __rop__
    # 定义一个函数rop_func，返回参数__rop__

    myf_op = type("myf_op", (sctype,), {__op__: op_func, __rop__: rop_func})
    # 创建一个名为myf_op的类型，继承自sctype，包含属性__op__和__rop__分别对应op_func和rop_func函数

    # inheritance has to override, or this is correctly lost:
    res = op(myf_simple1(1), myf_simple2(2))
    # 使用op函数对myf_simple1和myf_simple2实例进行操作
    assert type(res) == sctype or type(res) == np.bool
    # 断言res的类型为sctype或np.bool
    assert op(myf_simple1(1), myf_simple2(2)) == op(1, 2)  # inherited
    # 断言对myf_simple1和myf_simple2实例使用op函数的结果等于对1和2使用op函数的结果（继承）

    # Two independent subclasses do not really define an order.  This could
    # be attempted, but we do not since Python's `int` does neither:
    assert op(myf_op(1), myf_simple1(2)) == __op__
    # 断言对myf_op(1)和myf_simple1(2)使用op函数的结果等于__op__
    assert op(myf_simple1(1), myf_op(2)) == op(1, 2)  # inherited
    # 断言对myf_simple1(1)和myf_op(2)使用op函数的结果等于对1和2使用op函数的结果（继承）


def test_longdouble_complex():
    # Simple test to check longdouble and complex combinations, since these
    # need to go through promotion, which longdouble needs to be careful about.
    x = np.longdouble(1)
    # 创建一个np.longdouble类型的实例x
    assert x + 1j == 1+1j
    # 断言x加上虚数单位1j等于1加上1j
    assert 1j + x == 1+1j
    # 断言1j加上x等于1加上1j


@pytest.mark.parametrize(["__op__", "__rop__", "op", "cmp"], ops_with_names)
# 使用pytest的@parametrize装饰器，对参数进行多组化，使用ops_with_names作为参数
@pytest.mark.parametrize("subtype", [float, int, complex, np.float16])
# 使用pytest的@parametrize装饰器，对参数进行多组化，测试float、int、complex和np.float16类型
@np._no_nep50_warning()
def test_pyscalar_subclasses(subtype, __op__, __rop__, op, cmp):
    def op_func(self, other):
        return __op__
    # 定义一个函数op_func，返回参数__op__

    def rop_func(self, other):
        return __rop__
    # 定义一个函数rop_func，返回参数__rop__

    # Check that deferring is indicated using `__array_ufunc__`:
    myt = type("myt", (subtype,),
               {__op__: op_func, __rop__: rop_func, "__array_ufunc__": None})
    # 创建一个名为myt的类型，继承自subtype，包含属性__op__和__rop__分别对应op_func和rop_func函数，以及属性"__array_ufunc__"为None

    # Just like normally, we should never presume we can modify the float.
    assert op(myt(1), np.float64(2)) == __op__
    # 断言对myt(1)和np.float64(2)使用op函数的结果等于__op__
    assert op(np.float64(1), myt(2)) == __rop__
    # 断言对np.float64(1)和myt(2)使用op函数的结果等于__rop__

    if op in {operator.mod, operator.floordiv} and subtype == complex:
        return  # module is not support for complex.  Do not test.
    # 如果op函数是operator.mod或operator.floordiv，并且subtype是complex类型，则返回，因为模块不支持complex。不进行测试。

    if __rop__ == __op__:
        return
    # 如果__rop__等于__op__，则返回

    # When no deferring is indicated, subclasses are handled normally.
    myt = type("myt", (subtype,), {__rop__: rop_func})
    # 创建一个名为myt的类型，继承自subtype，包含属性__rop__对应rop_func函数

    # Check for float32, as a float subclass float64 may behave differently
    res = op(myt(1), np.float16(2))
    # 使用op函数对myt(1)和np.float16(2)进行操作
    expected = op(subtype(1), np.float16(2))
    # 计算subtype(1)和np.float16(2)使用op函数的结果
    assert res == expected
    # 断言res等于expected
    assert type(res) == type(expected)
    # 断言res的类型等于expected的类型
    # 使用 op 函数计算 np.float32(2) 和 myt(1) 的结果
    res = op(np.float32(2), myt(1))
    # 使用 op 函数计算 np.float32(2) 和 subtype(1) 的结果（期望结果）
    expected = op(np.float32(2), subtype(1))
    # 断言结果 res 等于期望值 expected
    assert res == expected
    # 断言结果 res 的类型等于 expected 的类型
    assert type(res) == type(expected)

    # 对 np.longdouble 类型进行相同的检查：
    # 使用 op 函数计算 myt(1) 和 np.longdouble(2) 的结果
    res = op(myt(1), np.longdouble(2))
    # 使用 op 函数计算 subtype(1) 和 np.longdouble(2) 的结果（期望结果）
    expected = op(subtype(1), np.longdouble(2))
    # 断言结果 res 等于期望值 expected
    assert res == expected
    # 断言结果 res 的类型等于 expected 的类型
    assert type(res) == type(expected)

    # 再次使用 op 函数进行不同类型的计算：
    # 使用 op 函数计算 np.float32(2) 和 myt(1) 的结果
    res = op(np.float32(2), myt(1))
    # 使用 op 函数计算 np.longdouble(2) 和 subtype(1) 的结果（期望结果）
    expected = op(np.longdouble(2), subtype(1))
    # 断言结果 res 等于期望值 expected
    assert res == expected
# 定义一个测试函数，用于测试整数除法的行为
def test_truediv_int():
    # 这个断言应该成功，因为结果是浮点数：
    assert np.uint8(3) / 123454 == np.float64(3) / 123454


# 将下面的测试函数标记为“slow”，表示它是一个比较慢的测试
@pytest.mark.slow
# 参数化测试，测试各种运算符（不包括幂运算符），这些运算符用于标量
@pytest.mark.parametrize("op", [op for op in reasonable_operators_for_scalars if op is not operator.pow])
# 参数化测试，测试各种数据类型
@pytest.mark.parametrize("sctype", types)
# 参数化测试，测试另一种数据类型，包括 float、int 和 complex
@pytest.mark.parametrize("other_type", [float, int, complex])
# 参数化测试，测试是否翻转操作数
@pytest.mark.parametrize("rop", [True, False])
def test_scalar_matches_array_op_with_pyscalar(op, sctype, other_type, rop):
    # 显式地将值转换为数组，以检查 ufunc 路径是否匹配
    val1 = sctype(2)
    val2 = other_type(2)

    # 如果需要翻转操作数，重新定义操作符函数
    if rop:
        _op = op
        op = lambda x, y: _op(y, x)

    try:
        res = op(val1, val2)
    except TypeError:
        try:
            # 如果 ufunc 没有引发 TypeError，则生成预期结果并引发 AssertionError
            expected = op(np.asarray(val1), val2)
            raise AssertionError("ufunc didn't raise.")
        except TypeError:
            # 如果预期引发了 TypeError，则直接返回
            return
    else:
        # 计算预期结果
        expected = op(np.asarray(val1), val2)

    # 注意，这里只检查 dtype 是否相等，因为如果等效，ufunc 可能会选择较低的 dtype
    assert res == expected

    # 特殊情况：如果 val1 是 float 且 other_type 是 complex 且需要翻转操作数
    if isinstance(val1, float) and other_type is complex and rop:
        # Python 中的 complex 类型接受 float 的子类，所以结果可能是 Python 的 complex 类型
        # 因此，使用 `np.array()` 来确保结果的 dtype 是预期的
        assert np.array(res).dtype == expected.dtype
    else:
        # 否则，检查结果的 dtype 是否与预期的 dtype 相同
        assert res.dtype == expected.dtype
```
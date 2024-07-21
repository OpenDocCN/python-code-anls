# `.\pytorch\test\torch_np\numpy_tests\core\test_scalarmath.py`

```py
# Owner(s): ["module: dynamo"]
# 导入标准库模块
import contextlib
import functools
import itertools
import operator
import sys
import warnings

# 导入第三方库模块
from unittest import expectedFailure as xfail, skipIf as skipif, SkipTest
import numpy
import pytest
from pytest import raises as assert_raises

# 导入自定义模块
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    slowTest as slow,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo,
)

# 如果配置了 TEST_WITH_TORCHDYNAMO，则导入 numpy 的相关函数
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import (
        _gen_alignment_data,
        assert_,
        assert_almost_equal,
        assert_equal,
    )
# 否则导入 torch._numpy 的相关函数
else:
    import torch._numpy as np
    from torch._numpy.testing import (
        _gen_alignment_data,
        assert_,
        assert_almost_equal,
        assert_equal,
        #    assert_array_equal, suppress_warnings, _gen_alignment_data,
        #    assert_warns,
    )

# 定义 skip 函数，使用 functools.partial 包装 skipif 函数
skip = functools.partial(skipif, True)

# 设定 PYPY 环境变量为 False
IS_PYPY = False

# 定义常见数据类型列表
types = [
    np.bool_,
    np.byte,
    np.ubyte,
    np.short,
    np.intc,
    np.int_,
    np.longlong,
    np.single,
    np.double,
    np.csingle,
    np.cdouble,
]

# 获取浮点数类型列表
floating_types = np.floating.__subclasses__()

# 获取复数浮点数类型列表
complex_floating_types = np.complexfloating.__subclasses__()

# 定义包含对象及 None 的列表
objecty_things = [object(), None]

# 合理的标量运算符列表
reasonable_operators_for_scalars = [
    operator.lt,
    operator.le,
    operator.eq,
    operator.ne,
    operator.ge,
    operator.gt,
    operator.add,
    operator.floordiv,
    operator.mod,
    operator.mul,
    operator.pow,
    operator.sub,
    operator.truediv,
]


# 这个类测试不同数据类型的操作
class TestTypes(TestCase):
    # 测试各种数据类型的比较
    def test_types(self):
        for atype in types:
            a = atype(1)
            assert_(a == 1, f"error with {atype!r}: got {a!r}")
    def test_type_add(self):
        # 遍历类型列表
        for k, atype in enumerate(types):
            # 创建标量和数组，使用给定的类型
            a_scalar = atype(3)
            a_array = np.array([3], dtype=atype)
            # 再次遍历类型列表
            for l, btype in enumerate(types):
                # 创建另一个标量和数组，使用另一个类型
                b_scalar = btype(1)
                b_array = np.array([1], dtype=btype)
                # 计算标量和数组的和
                c_scalar = a_scalar + b_scalar
                c_array = a_array + b_array
                # 断言：检查标量和数组的数据类型是否相同
                # 如果不相同，输出错误信息，包含类型索引和名称
                assert_equal(
                    c_scalar.dtype,
                    c_array.dtype,
                    "error with types (%d/'%s' + %d/'%s')"
                    % (k, np.dtype(atype).name, l, np.dtype(btype).name),
                )

    def test_type_create(self):
        # 遍历类型列表
        for k, atype in enumerate(types):
            # 创建使用给定类型的数组
            a = np.array([1, 2, 3], atype)
            # 创建使用给定类型的数组
            b = atype([1, 2, 3])
            # 断言：检查两个数组是否相等
            assert_equal(a, b)

    @skipIfTorchDynamo()  # 在 torch.Dynamo 下会冻结（循环展开，嗯）
    def test_leak(self):
        # 测试标量对象的内存泄漏
        # 如果有泄漏，valgrind 将显示为仍可访问的约 2.6MB
        for i in range(200000):
            # 执行加法操作，测试是否有内存泄漏
            np.add(1, 1)
class TestBaseMath(TestCase):
    def test_blocked(self):
        # test alignments offsets for simd instructions
        # alignments for vz + 2 * (vs - 1) + 1
        for dt, sz in [(np.float32, 11), (np.float64, 7), (np.int32, 11)]:
            for out, inp1, inp2, msg in _gen_alignment_data(
                dtype=dt, type="binary", max_size=sz
            ):
                exp1 = np.ones_like(inp1)
                inp1[...] = np.ones_like(inp1)
                inp2[...] = np.zeros_like(inp2)
                assert_almost_equal(np.add(inp1, inp2), exp1, err_msg=msg)
                assert_almost_equal(np.add(inp1, 2), exp1 + 2, err_msg=msg)
                assert_almost_equal(np.add(1, inp2), exp1, err_msg=msg)

                np.add(inp1, inp2, out=out)  # Perform element-wise addition of inp1 and inp2, storing result in out
                assert_almost_equal(out, exp1, err_msg=msg)

                inp2[...] += np.arange(inp2.size, dtype=dt) + 1  # Increment each element of inp2 by its index + 1
                assert_almost_equal(
                    np.square(inp2), np.multiply(inp2, inp2), err_msg=msg
                )  # Check if squaring inp2 is equivalent to element-wise multiplication of inp2 with itself
                # skip true divide for ints
                if dt != np.int32:
                    assert_almost_equal(
                        np.reciprocal(inp2), np.divide(1, inp2), err_msg=msg
                    )  # Check if reciprocal of inp2 is equivalent to 1 divided by inp2

                inp1[...] = np.ones_like(inp1)
                np.add(inp1, 2, out=out)  # Add 2 to each element of inp1, storing result in out
                assert_almost_equal(out, exp1 + 2, err_msg=msg)
                inp2[...] = np.ones_like(inp2)
                np.add(2, inp2, out=out)  # Add 2 to each element of inp2, storing result in out
                assert_almost_equal(out, exp1 + 2, err_msg=msg)

    @xpassIfTorchDynamo  # (reason="pytorch does not have .view")
    def test_lower_align(self):
        # check data that is not aligned to element size
        # i.e doubles are aligned to 4 bytes on i386
        d = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)  # Create a view of a slice of np.zeros array as np.float64
        o = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)  # Create another view of the same type and slice
        assert_almost_equal(d + d, d * 2)  # Check if adding d to itself equals d multiplied by 2
        np.add(d, d, out=o)  # Add d to itself, store result in o
        np.add(np.ones_like(d), d, out=o)  # Add 1 to each element of d, store result in o
        np.add(d, np.ones_like(d), out=o)  # Add 1 to each element of d, store result in o
        np.add(np.ones_like(d), d)  # Add 1 to each element of d (no out parameter specified)
        np.add(d, np.ones_like(d))  # Add 1 to each element of d (no out parameter specified)


class TestPower(TestCase):
    def test_small_types(self):
        for t in [np.int8, np.int16, np.float16]:
            a = t(3)  # Create instance of type t with value 3
            b = a**4  # Compute a raised to the power of 4
            assert_(b == 81, f"error with {t!r}: got {b!r}")  # Assert if b equals 81

    def test_large_types(self):
        for t in [np.int32, np.int64, np.float32, np.float64]:
            a = t(51)  # Create instance of type t with value 51
            b = a**4  # Compute a raised to the power of 4
            msg = f"error with {t!r}: got {b!r}"  # Prepare error message string
            if np.issubdtype(t, np.integer):
                assert_(b == 6765201, msg)  # Assert if b equals 6765201 for integer types
            else:
                assert_almost_equal(b, 6765201, err_msg=msg)  # Assert if b is almost equal to 6765201 for floating-point types

    @skip(reason="NP_VER: fails on CI on older NumPy")
    @xpassIfTorchDynamo  # (reason="Value-based casting: (2)**(-2) -> 0 in pytorch.")
    # 定义测试函数，用于测试整数的负指数幂
    def test_integers_to_negative_integer_power(self):
        # 定义指数数组，包含了各种数据类型的负整数
        exp = [np.array(-1, dt)[()] for dt in "bhil"]

        # 对基数为 1 的特殊情况进行测试
        base = [np.array(1, dt)[()] for dt in "bhilB"]
        # 遍历基数和指数的所有组合
        for i1, i2 in itertools.product(base, exp):
            # 如果基数的数据类型不是 np.uint64，则预期会引发 ValueError
            if i1.dtype != np.uint64:
                assert_raises(ValueError, operator.pow, i1, i2)
            else:
                # 否则计算基数的 i2 次方
                res = operator.pow(i1, i2)
                # 断言结果的数据类型为 np.float64
                assert_(res.dtype.type is np.float64)
                # 断言结果接近于 1.0
                assert_almost_equal(res, 1.0)

        # 对基数为 -1 的特殊情况进行测试
        base = [np.array(-1, dt)[()] for dt in "bhil"]
        # 遍历基数和指数的所有组合
        for i1, i2 in itertools.product(base, exp):
            # 如果基数的数据类型不是 np.uint64，则预期会引发 ValueError
            if i1.dtype != np.uint64:
                assert_raises(ValueError, operator.pow, i1, i2)
            else:
                # 否则计算基数的 i2 次方
                res = operator.pow(i1, i2)
                # 断言结果的数据类型为 np.float64
                assert_(res.dtype.type is np.float64)
                # 断言结果接近于 -1.0
                assert_almost_equal(res, -1.0)

        # 对基数为 2 的普通情况进行测试
        base = [np.array(2, dt)[()] for dt in "bhilB"]
        # 遍历基数和指数的所有组合
        for i1, i2 in itertools.product(base, exp):
            # 如果基数的数据类型不是 np.uint64，则预期会引发 ValueError
            if i1.dtype != np.uint64:
                assert_raises(ValueError, operator.pow, i1, i2)
            else:
                # 否则计算基数的 i2 次方
                res = operator.pow(i1, i2)
                # 断言结果的数据类型为 np.float64
                assert_(res.dtype.type is np.float64)
                # 断言结果接近于 0.5
                assert_almost_equal(res, 0.5)

    # 定义测试函数，用于测试混合数据类型的指数幂运算
    def test_mixed_types(self):
        # 定义数据类型列表
        typelist = [
            np.int8,
            np.int16,
            np.float16,
            np.float32,
            np.float64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
        ]
        # 遍历两个数据类型的所有组合
        for t1 in typelist:
            for t2 in typelist:
                # 创建两个指定类型的数值对象
                a = t1(3)
                b = t2(2)
                # 计算 a 的 b 次方
                result = a**b
                # 构建错误信息消息
                msg = f"error with {t1!r} and {t2!r}: got {result!r}, expected {9!r}"
                # 如果结果的数据类型是整数类型，断言结果等于 9
                if np.issubdtype(np.dtype(result), np.integer):
                    assert_(result == 9, msg)
                else:
                    # 否则，断言结果接近于 9
                    assert_almost_equal(result, 9, err_msg=msg)

    # 定义测试函数，用于测试模幂运算
    def test_modular_power(self):
        # 定义变量 a, b, c
        a = 5
        b = 4
        c = 10
        # 计算 a 的 b 次方对 c 取模的预期结果，但不使用该结果
        expected = pow(a, b, c)  # noqa: F841
        # 遍历不同数据类型的数组
        for t in (np.int32, np.float32, np.complex64):
            # 注意，3 操作数幂运算只对第一个参数进行调度
            # 断言对 t(a) 进行指数幂运算并同时使用 c 会引发 TypeError
            assert_raises(TypeError, operator.pow, t(a), b, c)
            # 断言对 np.array(t(a)) 进行指数幂运算并同时使用 c 会引发 TypeError
            assert_raises(TypeError, operator.pow, np.array(t(a)), b, c)
def floordiv_and_mod(x, y):
    # 返回 x 除以 y 的整数部分和余数
    return (x // y, x % y)


def _signs(dt):
    # 如果 dt 在无符号整数的类型码中，则返回一个元组 (+1,)
    # 否则返回一个元组 (+1, -1)
    if dt in np.typecodes["UnsignedInteger"]:
        return (+1,)
    else:
        return (+1, -1)


@instantiate_parametrized_tests
class TestModulus(TestCase):
    def test_modulus_basic(self):
        # dt = np.typecodes["AllInteger"] + np.typecodes["Float"]
        dt = "Bbhil" + "efd"
        for op in [floordiv_and_mod, divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                for sg1, sg2 in itertools.product(_signs(dt1), _signs(dt2)):
                    fmt = "op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s"
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    # 创建 np.array 对象，使用参数 sg1 和 sg2 分别乘以常数 71 和 19
                    a = np.array(sg1 * 71, dtype=dt1)[()]
                    b = np.array(sg2 * 19, dtype=dt2)[()]
                    # 应用操作 op(a, b)，获取商和余数
                    div, rem = op(a, b)
                    # 断言条件：div * b + rem 应等于 a，如果不等则输出错误信息 msg
                    assert_equal(div * b + rem, a, err_msg=msg)
                    # 如果 sg2 == -1，则断言 b < rem <= 0，否则断言 b > rem >= 0
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    @slow
    def test_float_modulus_exact(self):
        # 测试小整数的浮点模运算是否精确。同样适用于这些整数乘以二的幂次。
        nlst = list(range(-127, 0))
        plst = list(range(1, 128))
        dividend = nlst + [0] + plst
        divisor = nlst + plst
        arg = list(itertools.product(dividend, divisor))
        tgt = [divmod(*t) for t in arg]

        a, b = np.array(arg, dtype=int).T
        # 将精确的整数结果转换为浮点数，以便使用有符号零，这是被检查的。
        tgtdiv, tgtrem = np.array(tgt, dtype=float).T
        tgtdiv = np.where((tgtdiv == 0.0) & ((b < 0) ^ (a < 0)), -0.0, tgtdiv)
        tgtrem = np.where((tgtrem == 0.0) & (b < 0), -0.0, tgtrem)

        for op in [floordiv_and_mod, divmod]:
            for dt in np.typecodes["Float"]:
                msg = f"op: {op.__name__}, dtype: {dt}"
                fa = a.astype(dt)
                fb = b.astype(dt)
                # 使用列表推导使 a_ 和 b_ 成为标量
                div, rem = zip(*[op(a_, b_) for a_, b_ in zip(fa, fb)])
                # 断言条件：div 应等于 tgtdiv，rem 应等于 tgtrem，否则输出错误信息 msg
                assert_equal(div, tgtdiv, err_msg=msg)
                assert_equal(rem, tgtrem, err_msg=msg)
    # 定义测试方法，用于测试浮点数取模的精度问题
    def test_float_modulus_roundoff(self):
        # 注释: 测试用例名称标识，这里是解决 GitHub 问题编号 6127
        # dt = np.typecodes["Float"]
        dt = "efd"
        # 迭代不同的除法和取模操作
        for op in [floordiv_and_mod, divmod]:
            # 对浮点数类型组合进行迭代测试
            for dt1, dt2 in itertools.product(dt, dt):
                # 对不同符号组合进行迭代测试
                for sg1, sg2 in itertools.product((+1, -1), (+1, -1)):
                    # 格式化消息字符串，描述当前测试情况
                    fmt = "op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s"
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    # 创建具体的浮点数数组 a 和 b，用于操作
                    a = np.array(sg1 * 78 * 6e-8, dtype=dt1)[()]
                    b = np.array(sg2 * 6e-8, dtype=dt2)[()]
                    # 执行除法和取模操作
                    div, rem = op(a, b)
                    # 断言等式应该成立，当使用 fmod 函数时
                    assert_equal(div * b + rem, a, err_msg=msg)
                    # 如果 sg2 为负数，断言 rem 应该在 (b, 0] 区间内
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    # 参数化测试方法，用于测试浮点数取模的边界情况
    @parametrize("dt", "efd")
    def test_float_modulus_corner_cases(self, dt):
        # 如果 dt 类型为 'e'，暂时标记为待修复问题，跳过测试
        if dt == "e":
            raise SkipTest("RuntimeError: 'nextafter_cpu' not implemented for 'Half'")

        # 创建浮点数数组 a 和 b，用于操作
        b = np.array(1.0, dtype=dt)
        a = np.nextafter(np.array(0.0, dtype=dt), -b)
        # 使用 operator.mod 计算取模
        rem = operator.mod(a, b)
        # 断言 rem 应小于等于 b
        assert_(rem <= b, f"dt: {dt}")
        rem = operator.mod(-a, -b)
        # 断言 rem 应大于等于 -b
        assert_(rem >= -b, f"dt: {dt}")

        # 检查 NaN 和 Inf 的情况
        #     with suppress_warnings() as sup:
        #         sup.filter(RuntimeWarning, "invalid value encountered in remainder")
        #         sup.filter(RuntimeWarning, "divide by zero encountered in remainder")
        #         sup.filter(RuntimeWarning, "divide by zero encountered in floor_divide")
        #         sup.filter(RuntimeWarning, "divide by zero encountered in divmod")
        #         sup.filter(RuntimeWarning, "invalid value encountered in divmod")
        for dt in "efd":
            fone = np.array(1.0, dtype=dt)
            fzer = np.array(0.0, dtype=dt)
            finf = np.array(np.inf, dtype=dt)
            fnan = np.array(np.nan, dtype=dt)
            rem = operator.mod(fone, fzer)
            # 断言 rem 是 NaN
            assert_(np.isnan(rem), f"dt: {dt}")
            # MSVC 2008 在这里返回 NaN，因此禁用此检查。
            # rem = operator.mod(fone, finf)
            # assert_(rem == fone, 'dt: %s' % dt)
            rem = operator.mod(fone, fnan)
            # 断言 rem 是 NaN
            assert_(np.isnan(rem), f"dt: {dt}")
            rem = operator.mod(finf, fone)
            # 断言 rem 是 NaN
            assert_(np.isnan(rem), f"dt: {dt}")
            # 对每个操作，执行除法和取模操作
            for op in [floordiv_and_mod, divmod]:
                div, mod = op(fone, fzer)
                assert_(np.isinf(div)) and assert_(np.isnan(mod))
class TestComplexDivision(TestCase):
    @skip(reason="With pytorch, 1/(0+0j) is nan + nan*j, not inf + nan*j")
    def test_zero_division(self):
        # 针对复数类型为np.complex64和np.complex128的测试
        for t in [np.complex64, np.complex128]:
            # 设置复数类型t的零值和非零值
            a = t(0.0)
            b = t(1.0)
            # 断言除以零会得到无穷大
            assert_(np.isinf(b / a))
            # 设置为包含无穷大和NaN的复数值
            b = t(complex(np.inf, np.inf))
            assert_(np.isinf(b / a))
            b = t(complex(np.inf, np.nan))
            assert_(np.isinf(b / a))
            b = t(complex(np.nan, np.inf))
            assert_(np.isinf(b / a))
            b = t(complex(np.nan, np.nan))
            # 断言除以零会得到NaN
            assert_(np.isnan(b / a))
            # 设置复数类型t的零值
            b = t(0.0)
            # 断言除以零会得到NaN
            assert_(np.isnan(b / a))

    def test_signed_zeros(self):
        # 针对复数类型为np.complex64和np.complex128的测试
        for t in [np.complex64, np.complex128]:
            # 定义测试数据元组列表，每个元组包含分子、分母和预期结果
            data = (
                ((0.0, -1.0), (0.0, 1.0), (-1.0, -0.0)),
                ((0.0, -1.0), (0.0, -1.0), (1.0, -0.0)),
                ((0.0, -1.0), (-0.0, -1.0), (1.0, 0.0)),
                ((0.0, -1.0), (-0.0, 1.0), (-1.0, 0.0)),
                ((0.0, 1.0), (0.0, -1.0), (-1.0, 0.0)),
                ((0.0, -1.0), (0.0, -1.0), (1.0, -0.0)),
                ((-0.0, -1.0), (0.0, -1.0), (1.0, -0.0)),
                ((-0.0, 1.0), (0.0, -1.0), (-1.0, -0.0)),
            )
            # 遍历测试数据
            for cases in data:
                # 分别获取分子、分母和预期结果
                n = cases[0]
                d = cases[1]
                ex = cases[2]
                # 计算结果并断言实部和虚部与预期结果一致
                result = t(complex(n[0], n[1])) / t(complex(d[0], d[1]))
                # 分别检查实部和虚部，避免在数组上下文中比较
                assert_equal(result.real, ex[0])
                assert_equal(result.imag, ex[1])

    def test_branches(self):
        # 针对复数类型为np.complex64和np.complex128的测试
        for t in [np.complex64, np.complex128]:
            # 定义测试数据列表
            data = list()

            # 触发分支: real(fabs(denom)) > imag(fabs(denom))
            # 随后执行else条件，因为两者都不等于0
            data.append(((2.0, 1.0), (2.0, 1.0), (1.0, 0.0)))

            # 触发分支: real(fabs(denom)) < imag(fabs(denom))
            data.append(((1.0, 2.0), (1.0, 2.0), (1.0, 0.0)))

            # 遍历测试数据
            for cases in data:
                # 分别获取分子、分母和预期结果
                n = cases[0]
                d = cases[1]
                ex = cases[2]
                # 计算结果并断言实部和虚部与预期结果一致
                result = t(complex(n[0], n[1])) / t(complex(d[0], d[1]))
                # 分别检查实部和虚部，避免在数组上下文中比较
                assert_equal(result.real, ex[0])
                assert_equal(result.imag, ex[1])
class TestConversion(TestCase):
    # 定义测试类 TestConversion，继承自 TestCase

    def test_int_from_long(self):
        # 测试从长整型转换为整型的函数
        # 注意：此测试假设默认的浮点数类型是 float64
        l = [1e6, 1e12, 1e18, -1e6, -1e12, -1e18]
        li = [10**6, 10**12, 10**18, -(10**6), -(10**12), -(10**18)]
        for T in [None, np.float64, np.int64]:
            # 创建 numpy 数组 a，使用类型 T
            a = np.array(l, dtype=T)
            # 断言：将数组中每个元素转换为整型后与预期列表 li 相等
            assert_equal([int(_m) for _m in a], li)

    @skipif(numpy.__version__ < "1.24", reason="NP_VER: fails on NumPy 1.23.x")
    @xpassIfTorchDynamo  # (reason="pytorch does not emit this warning.")
    def test_iinfo_long_values_1(self):
        # 测试 np.iinfo 对象中的长整型值
        for code in "bBh":
            # 在 DeprecationWarning 警告下，使用 np.iinfo(code).max + 1 创建数组 res
            with pytest.warns(DeprecationWarning):
                res = np.array(np.iinfo(code).max + 1, dtype=code)
            # 获取 np.iinfo(code).min 作为目标值 tgt
            tgt = np.iinfo(code).min
            # 断言：res 等于 tgt
            assert_(res == tgt)

    def test_iinfo_long_values_2(self):
        # 继续测试 np.iinfo 对象中的长整型值
        for code in np.typecodes["AllInteger"]:
            # 创建数组 res，使用 np.iinfo(code).max 和 code 类型
            res = np.array(np.iinfo(code).max, dtype=code)
            # 获取 np.iinfo(code).max 作为目标值 tgt
            tgt = np.iinfo(code).max
            # 断言：res 等于 tgt
            assert_(res == tgt)

        for code in np.typecodes["AllInteger"]:
            # 创建数据类型对象，并使用 np.iinfo(code).max 初始化为数组 res
            res = np.dtype(code).type(np.iinfo(code).max)
            # 获取 np.iinfo(code).max 作为目标值 tgt
            tgt = np.iinfo(code).max
            # 断言：res 等于 tgt
            assert_(res == tgt)

    def test_int_raise_behaviour(self):
        # 测试整型溢出的行为
        def overflow_error_func(dtype):
            # 尝试将 np.iinfo(dtype).max + 1 转换为 dtype 类型，期望引发 OverflowError 或 RuntimeError 异常
            dtype(np.iinfo(dtype).max + 1)

        for code in [np.int_, np.longlong]:
            # 断言：调用 overflow_error_func 函数，期望引发 OverflowError 或 RuntimeError 异常
            assert_raises((OverflowError, RuntimeError), overflow_error_func, code)
    # 测试 NumPy 标量的关系运算符

    def test_numpy_scalar_relational_operators(self):
        # 对所有整数类型进行测试
        for dt1 in np.typecodes["AllInteger"]:
            # 断言 1 大于数组中的零
            assert_(1 > np.array(0, dtype=dt1)[()], f"type {dt1} failed")
            # 断言 1 不小于数组中的零
            assert_(not 1 < np.array(0, dtype=dt1)[()], f"type {dt1} failed")

            # 嵌套循环，对第二个整数类型进行测试
            for dt2 in np.typecodes["AllInteger"]:
                # 断言数组中的 1 大于另一个数组中的零
                assert_(
                    np.array(1, dtype=dt1)[()] > np.array(0, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )
                # 断言数组中的 1 不小于另一个数组中的零
                assert_(
                    not np.array(1, dtype=dt1)[()] < np.array(0, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )

        # 对有符号整数和浮点数进行测试
        for dt1 in "bhl" + np.typecodes["Float"]:
            # 断言 1 大于数组中的 -1
            assert_(1 > np.array(-1, dtype=dt1)[()], f"type {dt1} failed")
            # 断言 1 不小于数组中的 -1
            assert_(not 1 < np.array(-1, dtype=dt1)[()], f"type {dt1} failed")
            # 断言数组中的 -1 等于另一个数组中的 -1
            assert_(-1 == np.array(-1, dtype=dt1)[()], f"type {dt1} failed")

            # 嵌套循环，对第二个有符号整数和浮点数进行测试
            for dt2 in "bhl" + np.typecodes["Float"]:
                # 断言数组中的 1 大于另一个数组中的 -1
                assert_(
                    np.array(1, dtype=dt1)[()] > np.array(-1, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )
                # 断言数组中的 1 不小于另一个数组中的 -1
                assert_(
                    not np.array(1, dtype=dt1)[()] < np.array(-1, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )
                # 断言数组中的 -1 等于另一个数组中的 -1
                assert_(
                    np.array(-1, dtype=dt1)[()] == np.array(-1, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )

    # 测试 NumPy 标量的关系运算符（第二部分）

    def test_numpy_scalar_relational_operators_2(self):
        # 对无符号整数进行测试
        for dt1 in "B":
            # 断言 -1 小于数组中的 1
            assert_(-1 < np.array(1, dtype=dt1)[()], f"type {dt1} failed")
            # 断言 -1 不大于数组中的 1
            assert_(not -1 > np.array(1, dtype=dt1)[()], f"type {dt1} failed")
            # 断言 -1 不等于数组中的 1
            assert_(-1 != np.array(1, dtype=dt1)[()], f"type {dt1} failed")

            # 嵌套循环，对无符号整数和有符号整数进行测试
            for dt2 in "bhil":
                # 断言数组中的 1 大于另一个数组中的 -1
                assert_(
                    np.array(1, dtype=dt1)[()] > np.array(-1, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )
                # 断言数组中的 1 不小于另一个数组中的 -1
                assert_(
                    not np.array(1, dtype=dt1)[()] < np.array(-1, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )
                # 断言数组中的 1 不等于另一个数组中的 -1
                assert_(
                    np.array(1, dtype=dt1)[()] != np.array(-1, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )

    # 测试标量与 None 的比较

    def test_scalar_comparison_to_none(self):
        # 标量应该返回 False，不应产生警告
        # 这些比较由 PEP8 标志，忽略它们
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always", "", FutureWarning)
            assert_(np.float32(1) is not None)
            assert_(np.float32(1) is not None)
        # 断言警告列表长度为 0
        assert_(len(w) == 0)
@xpassIfTorchDynamo  # 装饰器，用于条件通过时跳过测试（理由是“可以将 repr 委托给 PyTorch”）
class TestRepr(TestCase):
    def _test_type_repr(self, t):
        # 获取类型 t 的信息
        finfo = np.finfo(t)
        # 计算最后一个分数位索引
        last_fraction_bit_idx = finfo.nexp + finfo.nmant
        # 计算最后一个指数位索引
        last_exponent_bit_idx = finfo.nexp
        # 计算存储字节数
        storage_bytes = np.dtype(t).itemsize * 8
        # 对以下类型进行迭代测试
        for which in ["small denorm", "small norm"]:
            # 构造一个存储字节数量的零数组
            constr = np.array([0x00] * storage_bytes, dtype=np.uint8)
            if which == "small denorm":
                # 如果类型为小的非规范化数
                byte = last_fraction_bit_idx // 8
                bytebit = 7 - (last_fraction_bit_idx % 8)
                # 设置对应字节的比特位为 1
                constr[byte] = 1 << bytebit
            elif which == "small norm":
                # 如果类型为小的规范化数
                byte = last_exponent_bit_idx // 8
                bytebit = 7 - (last_exponent_bit_idx % 8)
                # 设置对应字节的比特位为 1
                constr[byte] = 1 << bytebit
            else:
                # 抛出值错误异常
                raise ValueError("hmm")
            # 从构造数组中视图出类型 t 的值
            val = constr.view(t)[0]
            # 获取 val 的字符串表示
            val_repr = repr(val)
            # 用 eval 重新创建 val 的副本
            val2 = t(eval(val_repr))
            # 断言 val 和 val2 相等
            if not (val2 == 0 and val < 1e-100):
                assert_equal(val, val2)

    def test_float_repr(self):
        # 测试 np.float32 和 np.float64 类型的 repr 方法
        for t in [np.float32, np.float64]:
            self._test_type_repr(t)


@skip(reason="Array scalars do not decay to python scalars.")
class TestMultiply(TestCase):
    # 跳过测试，原因是“数组标量不会衰减为 Python 标量”
    def test_seq_repeat(self):
        # 测试基本序列乘以 numpy 整数时会重复，与其他类型相乘时会引发错误。
        # 这些行为可能存在争议，并可能会进行更改。
        
        # 接受的类型为所有整数类型的集合
        accepted_types = set(np.typecodes["AllInteger"])
        # 不推荐使用的类型集合
        deprecated_types = {"?"}
        # 禁止使用的类型为所有类型的集合减去接受的类型和不推荐的类型
        forbidden_types = set(np.typecodes["All"]) - accepted_types - deprecated_types
        # 还需移除空标量，因为不能默认构造空标量
        
        forbidden_types -= {"V"}  # 不能默认构造空标量

        # 对于列表和元组这两种序列类型
        for seq_type in (list, tuple):
            seq = seq_type([1, 2, 3])
            
            # 对于每种接受的 numpy 类型
            for numpy_type in accepted_types:
                i = np.dtype(numpy_type).type(2)
                # 断言序列乘以整数类型后结果与整数乘以序列类型后结果相等
                assert_equal(seq * i, seq * int(i))
                assert_equal(i * seq, int(i) * seq)

            # 对于每种不推荐使用的 numpy 类型
            for numpy_type in deprecated_types:
                i = np.dtype(numpy_type).type()
                # 断言使用 assert_warns 检查 DeprecationWarning，操作符乘法应该引发警告
                assert_equal(
                    assert_warns(DeprecationWarning, operator.mul, seq, i), seq * int(i)
                )
                assert_equal(
                    assert_warns(DeprecationWarning, operator.mul, i, seq), int(i) * seq
                )

            # 对于每种禁止使用的 numpy 类型
            for numpy_type in forbidden_types:
                i = np.dtype(numpy_type).type()
                # 断言乘法操作会引发 TypeError 异常
                assert_raises(TypeError, operator.mul, seq, i)
                assert_raises(TypeError, operator.mul, i, seq)

    def test_no_seq_repeat_basic_array_like(self):
        # 测试一个类似数组但不知如何进行乘法运算的对象不会尝试序列重复（会引发 TypeError）。
        # 参见也许存在的问题 gh-7428。
        
        # 定义一个类 ArrayLike，模拟一个数组对象但不知如何进行乘法操作
        class ArrayLike:
            def __init__(self, arr):
                self.arr = arr

            def __array__(self):
                return self.arr

        # 对于简单的 ArrayLike 对象和内存视图对象（原始报告中的情况）
        for arr_like in (ArrayLike(np.ones(3)), memoryview(np.ones(3))):
            # 断言乘法操作后的结果应与预期的全数组相等
            assert_array_equal(arr_like * np.float32(3.0), np.full(3, 3.0))
            assert_array_equal(np.float32(3.0) * arr_like, np.full(3, 3.0))
            assert_array_equal(arr_like * np.int_(3), np.full(3, 3))
            assert_array_equal(np.int_(3) * arr_like, np.full(3, 3))
class TestNegative(TestCase):
    def test_exceptions(self):
        a = np.ones((), dtype=np.bool_)[()]
        # XXX: TypeError from numpy, RuntimeError from torch
        assert_raises((TypeError, RuntimeError), operator.neg, a)

    def test_result(self):
        types = np.typecodes["AllInteger"] + np.typecodes["AllFloat"]
        # 遍历所有整数和浮点数类型
        for dt in types:
            a = np.ones((), dtype=dt)[()]
            if dt in np.typecodes["UnsignedInteger"]:
                st = np.dtype(dt).type
                max = st(np.iinfo(dt).max)
                # 断言取负值后结果与最大无符号整数相等
                assert_equal(operator.neg(a), max)
            else:
                # 断言取负值后结果加上原始值等于零
                assert_equal(operator.neg(a) + a, 0)


class TestSubtract(TestCase):
    def test_exceptions(self):
        a = np.ones((), dtype=np.bool_)[()]
        # XXX: TypeError from numpy
        with assert_raises((TypeError, RuntimeError)):
            operator.sub(a, a)  # RuntimeError from torch

    def test_result(self):
        types = np.typecodes["AllInteger"] + np.typecodes["AllFloat"]
        # 遍历所有整数和浮点数类型
        for dt in types:
            a = np.ones((), dtype=dt)[()]
            # 断言对相同值进行减法运算结果为零
            assert_equal(operator.sub(a, a), 0)


@instantiate_parametrized_tests
class TestAbs(TestCase):
    def _test_abs_func(self, absfunc, test_dtype):
        x = test_dtype(-1.5)
        # 断言绝对值函数的正确性
        assert_equal(absfunc(x), 1.5)
        x = test_dtype(0.0)
        res = absfunc(x)
        # assert_equal() 检查零的符号
        assert_equal(res, 0.0)
        x = test_dtype(-0.0)
        res = absfunc(x)
        assert_equal(res, 0.0)

        x = test_dtype(np.finfo(test_dtype).max)
        # 断言对最大值取绝对值结果应该等于该最大值本身
        assert_equal(absfunc(x), x.real)

        # with suppress_warnings() as sup:
        #     sup.filter(UserWarning)
        x = test_dtype(np.finfo(test_dtype).tiny)
        # 断言对极小值取绝对值结果应该等于该极小值本身
        assert_equal(absfunc(x), x.real)

        x = test_dtype(np.finfo(test_dtype).min)
        # 断言对最小值取绝对值结果应该等于其相反数
        assert_equal(absfunc(x), -x.real)

    @parametrize("dtype", floating_types + complex_floating_types)
    def test_builtin_abs(self, dtype):
        self._test_abs_func(abs, dtype)

    @parametrize("dtype", floating_types + complex_floating_types)
    def test_numpy_abs(self, dtype):
        self._test_abs_func(np.abs, dtype)


@instantiate_parametrized_tests
class TestBitShifts(TestCase):
    @parametrize("type_code", np.typecodes["AllInteger"])
    @parametrize("op", [operator.rshift, operator.lshift])
    # 定义一个测试函数，用于测试按位移动操作，其中type_code是数据类型代码，op是位移操作函数
    def test_shift_all_bits(self, type_code, op):
        """Shifts where the shift amount is the width of the type or wider"""
        # 用type_code创建一个NumPy数据类型对象
        dt = np.dtype(type_code)
        # 计算数据类型的位数（每个数据类型的字节数乘以8）
        nbits = dt.itemsize * 8
        # 如果数据类型是无符号64位整数、32位整数或16位整数，则抛出跳过测试的异常
        if dt in (np.dtype(np.uint64), np.dtype(np.uint32), np.dtype(np.uint16)):
            raise SkipTest("NYI: bitshift uint64")

        # 对于每个测试值和每个位移量执行测试
        for val in [5, -5]:
            for shift in [nbits, nbits + 4]:
                # 将测试值转换为指定的数据类型，并取出单个值
                val_scl = np.array(val).astype(dt)[()]
                # 将位移量转换为数据类型对应的值
                shift_scl = dt.type(shift)

                # 执行位移操作
                res_scl = op(val_scl, shift_scl)
                # 如果测试值为负且操作是右移，则断言结果保留了符号位为负1
                if val_scl < 0 and op is operator.rshift:
                    # sign bit is preserved
                    assert_equal(res_scl, -1)
                else:
                    # 否则，断言结果为0
                    if type_code in ("i", "l") and shift == np.iinfo(type_code).bits:
                        # FIXME: make xfail
                        # 如果测试类型是整数类型且位移等于类型的位数，跳过测试，并附带问题链接
                        raise SkipTest(
                            "https://github.com/pytorch/pytorch/issues/70904"
                        )
                    assert_equal(res_scl, 0)

                # 对于数组中的每个元素执行相同的位移操作，断言结果与单个值的结果相同
                val_arr = np.array([val_scl] * 32, dtype=dt)
                shift_arr = np.array([shift] * 32, dtype=dt)
                res_arr = op(val_arr, shift_arr)
                assert_equal(res_arr, res_scl)
# 装饰器：标记该测试类跳过，原因是依赖于 pytest 来进行哈希测试
@skip(reason="Will rely on pytest for hashing")
# 装饰器：实例化参数化测试
@instantiate_parametrized_tests
# 定义测试类 TestHash，继承自 TestCase
class TestHash(TestCase):
    # 参数化测试方法，测试整数的哈希值
    @parametrize("type_code", np.typecodes["AllInteger"])
    def test_integer_hashes(self, type_code):
        # 获取整数类型的标量类型
        scalar = np.dtype(type_code).type
        # 遍历范围为 0 到 127 的整数
        for i in range(128):
            # 断言每个整数的哈希值等于其标量形式的哈希值
            assert hash(i) == hash(scalar(i))

    # 参数化测试方法，测试浮点数和复数的哈希值
    @parametrize("type_code", np.typecodes["AllFloat"])
    def test_float_and_complex_hashes(self, type_code):
        # 获取浮点数或复数类型的标量类型
        scalar = np.dtype(type_code).type
        # 遍历一些特定的值进行测试
        for val in [np.pi, np.inf, 3, 6.0]:
            numpy_val = scalar(val)
            # 如果是复数类型，则将其转换为 Python 的复数类型
            if numpy_val.dtype.kind == "c":
                val = complex(numpy_val)
            else:
                val = float(numpy_val)
            # 断言转换后的值与原始的 NumPy 标量值相等
            assert val == numpy_val
            # 断言转换后的值的哈希值与原始的 NumPy 标量值的哈希值相等
            assert hash(val) == hash(numpy_val)

        # 如果不同平台上的 NaN 哈希值不同，则需要额外的断言
        if hash(float(np.nan)) != hash(float(np.nan)):
            assert hash(scalar(np.nan)) != hash(scalar(np.nan))

    # 参数化测试方法，测试复数的哈希值
    @parametrize("type_code", np.typecodes["Complex"])
    def test_complex_hashes(self, type_code):
        # 获取复数类型的标量类型
        scalar = np.dtype(type_code).type
        # 遍历一些特定的复数值进行测试
        for val in [np.pi + 1j, np.inf - 3j, 3j, 6.0 + 1j]:
            numpy_val = scalar(val)
            # 断言转换后的复数的哈希值与原始的 NumPy 标量值的哈希值相等
            assert hash(complex(numpy_val)) == hash(numpy_val)


# 上下文管理器函数：用于设置递归限制
@contextlib.contextmanager
def recursionlimit(n):
    # 保存当前递归限制的原始值
    o = sys.getrecursionlimit()
    try:
        # 设置新的递归限制值
        sys.setrecursionlimit(n)
        # 使用 yield 返回上下文管理器的控制权
        yield
    finally:
        # 恢复原始的递归限制值
        sys.setrecursionlimit(o)


# 装饰器：实例化参数化测试
@instantiate_parametrized_tests
# 定义测试类 TestScalarOpsMisc，继承自 TestCase
class TestScalarOpsMisc(TestCase):
    # 装饰器：标记该测试方法为预期失败，原因是 pytorch 在整数溢出时没有警告
    @xfail  # (reason="pytorch does not warn on overflow")
    # 参数化测试方法，测试整数操作的溢出情况
    @parametrize("dtype", "Bbhil")
    @parametrize(
        "operation",
        [
            lambda min, max: max + max,
            lambda min, max: min - max,
            lambda min, max: max * max,
        ],
    )
    def test_scalar_integer_operation_overflow(self, dtype, operation):
        # 获取指定类型的标量类型
        st = np.dtype(dtype).type
        # 获取指定类型的最小值和最大值的标量形式
        min = st(np.iinfo(dtype).min)
        max = st(np.iinfo(dtype).max)

        # 使用 pytest 的 warn 断言，检查是否会出现溢出警告
        with pytest.warns(RuntimeWarning, match="overflow encountered"):
            operation(min, max)

    # 装饰器：跳过该测试方法，原因是整数溢出在某些平台上会导致 pytorch 崩溃
    @skip(reason="integer overflow UB: crashes pytorch under ASAN")
    # 参数化测试方法，测试整数操作中的溢出情况
    @parametrize("dtype", "bhil")
    @parametrize(
        "operation",
        [
            lambda min, neg_1: -min,
            lambda min, neg_1: abs(min),
            lambda min, neg_1: min * neg_1,
            subtest(
                lambda min, neg_1: min // neg_1,
                decorators=[skip(reason="broken on some platforms")],
            ),
        ],
    )
    # 定义一个测试方法，用于测试标量有符号整数溢出情况
    def test_scalar_signed_integer_overflow(self, dtype, operation):
        # 获取指定数据类型的标量类型
        st = np.dtype(dtype).type
        # 获取该数据类型的最小有符号整数值
        min = st(np.iinfo(dtype).min)
        # 定义一个等于 -1 的同类型变量
        neg_1 = st(-1)

        # 使用 pytest 的 warns 函数捕获 RuntimeWarning，匹配字符串 "overflow encountered"
        with pytest.warns(RuntimeWarning, match="overflow encountered"):
            # 执行指定操作，期望触发溢出警告
            operation(min, neg_1)

    # 装饰器，条件为 NumPy 版本小于 "1.24" 时跳过该测试
    @skipif(numpy.__version__ < "1.24", reason="NP_VER: fails on NumPy 1.23.x")
    # 装饰器，指定条件下跳过该测试
    @xpassIfTorchDynamo  # (reason="pytorch does not warn on overflow")
    # 参数化装饰器，指定测试数据类型为无符号字节型整数 "B"
    @parametrize("dtype", "B")
    # 定义一个测试方法，用于测试标量无符号整数溢出情况
    def test_scalar_unsigned_integer_overflow(self, dtype):
        # 获取指定数据类型的标量类型，初始化为 8
        val = np.dtype(dtype).type(8)
        
        # 使用 pytest 的 warns 函数捕获 RuntimeWarning，匹配字符串 "overflow encountered"
        with pytest.warns(RuntimeWarning, match="overflow encountered"):
            # 执行负值操作，期望触发溢出警告
            -val

        # 获取指定数据类型的标量类型，初始化为 0
        zero = np.dtype(dtype).type(0)
        -zero  # 无警告

    # 装饰器，标记为预期失败，理由是 PyTorch 在除零时会引发 RuntimeError
    @xfail  # (reason="pytorch raises RuntimeError on division by zero")
    # 参数化装饰器，指定测试数据类型为所有整数类型
    @parametrize("dtype", np.typecodes["AllInteger"])
    # 参数化装饰器，指定测试操作为整数除法和取模运算
    @parametrize(
        "operation",
        [
            lambda val, zero: val // zero,
            lambda val, zero: val % zero,
        ],
    )
    # 定义一个测试方法，用于测试标量整数除零操作
    def test_scalar_integer_operation_divbyzero(self, dtype, operation):
        # 获取指定数据类型的标量类型，初始化为 100
        st = np.dtype(dtype).type
        val = st(100)
        zero = st(0)

        # 使用 pytest 的 warns 函数捕获 RuntimeWarning，匹配字符串 "divide by zero"
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            # 执行指定操作，期望触发除零警告
            operation(val, zero)
ops_with_names = [
    ("__lt__", "__gt__", operator.lt, True),
    ("__le__", "__ge__", operator.le, True),
    ("__eq__", "__eq__", operator.eq, True),
    # Note __op__ and __rop__ may be identical here:
    ("__ne__", "__ne__", operator.ne, True),
    ("__gt__", "__lt__", operator.gt, True),
    ("__ge__", "__le__", operator.ge, True),
    ("__floordiv__", "__rfloordiv__", operator.floordiv, False),
    ("__truediv__", "__rtruediv__", operator.truediv, False),
    ("__add__", "__radd__", operator.add, False),
    ("__mod__", "__rmod__", operator.mod, False),
    ("__mul__", "__rmul__", operator.mul, False),
    ("__pow__", "__rpow__", operator.pow, False),
    ("__sub__", "__rsub__", operator.sub, False),
]

# 装饰器，用于实例化参数化测试类
@instantiate_parametrized_tests
class TestScalarSubclassingMisc(TestCase):
    # 装饰器，标记跳过此测试的原因
    @skip(reason="We do not support subclassing scalars.")
    # 参数化测试，将ops_with_names中的元组拆解为__op__, __rop__, op, cmp四个参数
    @parametrize("__op__, __rop__, op, cmp", ops_with_names)
    # 参数化测试，验证sctype参数为np.float32或np.float64
    @parametrize("sctype", [np.float32, np.float64])
    def test_subclass_deferral(self, sctype, __op__, __rop__, op, cmp):
        """
        This test covers scalar subclass deferral.  Note that this is exceedingly
        complicated, especially since it tends to fall back to the array paths and
        these additionally add the "array priority" mechanism.

        The behaviour was modified subtly in 1.22 (to make it closer to how Python
        scalars work).  Due to its complexity and the fact that subclassing NumPy
        scalars is probably a bad idea to begin with.  There is probably room
        for adjustments here.
        """

        # 定义简单的自定义类，继承自sctype
        class myf_simple1(sctype):
            pass

        # 定义简单的自定义类，继承自sctype
        class myf_simple2(sctype):
            pass

        # 定义操作函数，返回__op__对应的操作
        def op_func(self, other):
            return __op__

        # 定义右操作函数，返回__rop__对应的操作
        def rop_func(self, other):
            return __rop__

        # 动态创建类myf_op，继承自sctype，包含__op__和__rop__方法
        myf_op = type("myf_op", (sctype,), {__op__: op_func, __rop__: rop_func})

        # 继承必须覆盖，否则将正确丢失：
        # 执行操作op，验证返回结果类型为sctype或np.bool_
        res = op(myf_simple1(1), myf_simple2(2))
        assert type(res) == sctype or type(res) == np.bool_
        # 验证op(myf_simple1(1), myf_simple2(2))的结果等于op(1, 2)，即继承操作
        assert op(myf_simple1(1), myf_simple2(2)) == op(1, 2)

        # 两个独立的子类实际上并没有定义顺序。虽然可以尝试，但我们不这样做，因为Python的`int`也没有定义：
        # 验证op(myf_op(1), myf_simple1(2))的结果等于__op__
        assert op(myf_op(1), myf_simple1(2)) == __op__
        # 验证op(myf_simple1(1), myf_op(2))的结果等于op(1, 2)，即继承操作
        assert op(myf_simple1(1), myf_op(2)) == op(1, 2)

    # 装饰器，标记跳过此测试的原因
    @skip(reason="We do not support subclassing scalars.")
    # 参数化测试，将ops_with_names中的元组拆解为__op__, __rop__, op, cmp四个参数
    @parametrize("__op__, __rop__, op, cmp", ops_with_names)
    # 参数化测试，验证subtype参数为float, int, complex, np.float16之一
    # @np._no_nep50_warning()
    # 定义测试方法，用于测试自定义的数值类型子类在不同操作下的行为
    def test_pyscalar_subclasses(self, subtype, __op__, __rop__, op, cmp):
        # 定义操作函数，根据参数返回对应的操作结果
        def op_func(self, other):
            return __op__

        # 定义右操作函数，根据参数返回对应的右操作结果
        def rop_func(self, other):
            return __rop__

        # 创建类型为 "myt" 的新类，继承自 subtype，具有指定的操作方法和__array_ufunc__设置为None
        myt = type(
            "myt",
            (subtype,),
            {__op__: op_func, __rop__: rop_func, "__array_ufunc__": None},
        )

        # 断言对于 myt 类型的实例和 np.float64(2)，使用 op 操作后结果应该为 __op__
        assert op(myt(1), np.float64(2)) == __op__
        # 断言对于 np.float64(1) 和 myt(2)，使用 op 操作后结果应该为 __rop__
        assert op(np.float64(1), myt(2)) == __rop__

        # 如果操作为模数或整除，并且 subtype 是复数类型，则不进行测试，因为这些操作不支持复数
        if op in {operator.mod, operator.floordiv} and subtype == complex:
            return  # module is not support for complex.  Do not test.

        # 如果 __rop__ 等于 __op__，则不进行后续操作
        if __rop__ == __op__:
            return

        # 当没有指示延迟操作时，子类应按正常方式处理
        # 创建类型为 "myt" 的新类，继承自 subtype，具有指定的右操作方法
        myt = type("myt", (subtype,), {__rop__: rop_func})

        # 对于 float32 进行检查，因为 float64 的 float 子类可能会有不同的行为
        res = op(myt(1), np.float16(2))
        expected = op(subtype(1), np.float16(2))
        assert res == expected
        assert type(res) == type(expected)
        res = op(np.float32(2), myt(1))
        expected = op(np.float32(2), subtype(1))
        assert res == expected
        assert type(res) == type(expected)
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```
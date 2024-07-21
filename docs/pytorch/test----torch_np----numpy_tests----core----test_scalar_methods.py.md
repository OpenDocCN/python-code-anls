# `.\pytorch\test\torch_np\numpy_tests\core\test_scalar_methods.py`

```
"""
Test the scalar constructors, which also do type-coercion
"""
# 导入需要的模块和类
import fractions  # 导入 fractions 模块，用于处理分数
import functools  # 导入 functools 模块，用于创建部分函数
import sys  # 导入 sys 模块，用于与 Python 解释器交互
import types  # 导入 types 模块，用于操作类型信息
from typing import Any, Type  # 导入 Any 和 Type 类型提示

# 导入 pytest 测试框架相关函数和类
import pytest
from pytest import raises as assert_raises  # 导入 raises 别名为 assert_raises

# 导入 torch 相关测试工具和类
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)

# 根据是否使用 torchdynamo 导入 numpy 或 torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np  # 导入 numpy 库，用于数值计算
    from numpy.testing import assert_equal  # 导入 assert_equal 函数，用于断言 numpy 数组相等
else:
    import torch._numpy as np  # 导入 torch._numpy 库，用于数值计算
    from torch._numpy.testing import assert_equal  # 导入 assert_equal 函数，用于断言 torch._numpy 数组相等

# 创建跳过装饰器的别名
skip = functools.partial(skipif, True)

# 装饰器：跳过测试类的原因是 "XXX: scalar.as_integer_ratio not implemented"
@skip(reason="XXX: scalar.as_integer_ratio not implemented")
@instantiate_parametrized_tests  # 实例化参数化测试
class TestAsIntegerRatio(TestCase):  # 定义测试类 TestAsIntegerRatio，继承自 TestCase 类

    # 使用 cpython 测试 "test_floatasratio" 的部分推导
    @parametrize("ftype", [np.half, np.single, np.double])
    @parametrize(
        "f, ratio",
        [
            (0.875, (7, 8)),
            (-0.875, (-7, 8)),
            (0.0, (0, 1)),
            (11.5, (23, 2)),
        ],
    )
    def test_small(self, ftype, f, ratio):  # 定义测试方法 test_small
        assert_equal(ftype(f).as_integer_ratio(), ratio)  # 断言调用 as_integer_ratio 后结果与预期 ratio 相等

    # 测试简单分数的方法
    @parametrize("ftype", [np.half, np.single, np.double])
    def test_simple_fractions(self, ftype):  # 定义测试方法 test_simple_fractions
        R = fractions.Fraction  # 创建 fractions.Fraction 的别名 R
        assert_equal(R(0, 1), R(*ftype(0.0).as_integer_ratio()))  # 断言生成的分数与预期分数相等
        assert_equal(R(5, 2), R(*ftype(2.5).as_integer_ratio()))  # 断言生成的分数与预期分数相等
        assert_equal(R(1, 2), R(*ftype(0.5).as_integer_ratio()))  # 断言生成的分数与预期分数相等
        assert_equal(R(-2100, 1), R(*ftype(-2100.0).as_integer_ratio()))  # 断言生成的分数与预期分数相等

    # 测试错误情况的方法
    @parametrize("ftype", [np.half, np.single, np.double])
    def test_errors(self, ftype):  # 定义测试方法 test_errors
        assert_raises(OverflowError, ftype("inf").as_integer_ratio)  # 断言调用 as_integer_ratio 抛出 OverflowError 异常
        assert_raises(OverflowError, ftype("-inf").as_integer_ratio)  # 断言调用 as_integer_ratio 抛出 OverflowError 异常
        assert_raises(ValueError, ftype("nan").as_integer_ratio)  # 断言调用 as_integer_ratio 抛出 ValueError 异常

    # 测试与已知值比较的方法
    def test_against_known_values(self):  # 定义测试方法 test_against_known_values
        R = fractions.Fraction  # 创建 fractions.Fraction 的别名 R
        assert_equal(R(1075, 512), R(*np.half(2.1).as_integer_ratio()))  # 断言生成的分数与预期分数相等
        assert_equal(R(-1075, 512), R(*np.half(-2.1).as_integer_ratio()))  # 断言生成的分数与预期分数相等
        assert_equal(R(4404019, 2097152), R(*np.single(2.1).as_integer_ratio()))  # 断言生成的分数与预期分数相等
        assert_equal(R(-4404019, 2097152), R(*np.single(-2.1).as_integer_ratio()))  # 断言生成的分数与预期分数相等
        assert_equal(
            R(4728779608739021, 2251799813685248), R(*np.double(2.1).as_integer_ratio())
        )  # 断言生成的分数与预期分数相等
        assert_equal(
            R(-4728779608739021, 2251799813685248),
            R(*np.double(-2.1).as_integer_ratio()),
        )  # 断言生成的分数与预期分数相等
        # longdouble 是平台相关的
    @parametrize(
        "ftype, frac_vals, exp_vals",
        [
            # 使用 hypothesis 生成的 dtype 测试用例
            # 每种 dtype 使用前五个生成的案例
            (
                np.half,  # 使用 np.half 类型
                [
                    0.0,
                    0.01154830649280303,
                    0.31082276347447274,
                    0.527350517124794,
                    0.8308562335072596,
                ],  # 分数部分的测试值列表
                [0, 1, 0, -8, 12],  # 指数部分的测试值列表
            ),
            (
                np.single,  # 使用 np.single 类型
                [
                    0.0,
                    0.09248576989263226,
                    0.8160498218131407,
                    0.17389442853722373,
                    0.7956044195067877,
                ],  # 分数部分的测试值列表
                [0, 12, 10, 17, -26],  # 指数部分的测试值列表
            ),
            (
                np.double,  # 使用 np.double 类型
                [
                    0.0,
                    0.031066908499895136,
                    0.5214135908877832,
                    0.45780736035689296,
                    0.5906586745934036,
                ],  # 分数部分的测试值列表
                [0, -801, 51, 194, -653],  # 指数部分的测试值列表
            ),
        ],
    )
    def test_roundtrip(self, ftype, frac_vals, exp_vals):
        # 对于给定的分数和指数值，进行循环测试
        for frac, exp in zip(frac_vals, exp_vals):
            # 根据分数和指数创建浮点数 f
            f = np.ldexp(ftype(frac), exp)
            # 断言 f 的类型为 ftype
            assert f.dtype == ftype
            # 获取 f 的分子和分母
            n, d = f.as_integer_ratio()

            try:
                # 尝试将分子和分母转换为 longdouble 类型
                nf = np.longdouble(n)
                df = np.longdouble(d)
            except (OverflowError, RuntimeWarning):
                # 如果转换出错，则抛出跳过测试的异常
                raise SkipTest("longdouble too small on this platform")  # noqa: B904

            # 断言 nf/df 等于 f，用于验证转换的正确性
            assert_equal(nf / df, f, f"{n}/{d}")
# 使用装饰器@skip，标记该测试类为跳过测试，原因是旧版本的 numpy 存在 .is_integer 方法的问题
# 使用装饰器@instantiate_parametrized_tests，标记该测试类需要实例化参数化测试

@skip(reason="NP_VER: older numpies has problems with .is_integer")
@instantiate_parametrized_tests
class TestIsInteger(TestCase):
    
    # 使用装饰器@parametrize，为测试方法test_special提供参数化测试数据
    @parametrize("str_value", ["inf", "nan"])
    @parametrize("code", np.typecodes["Float"])
    def test_special(self, code, str_value):
        # 获取对应类型代码的 numpy 数据类型对象，并使用 str_value 创建该类型的实例
        cls = np.dtype(code).type
        value = cls(str_value)
        # 断言该实例的 .is_integer() 方法返回 False
        assert not value.is_integer()

    # 使用装饰器@parametrize，为测试方法test_true提供参数化测试数据
    @parametrize(
        "code", "efd" + "Bbhil"
    )  # np.typecodes["Float"] + np.typecodes["AllInteger"])
    def test_true(self, code: str) -> None:
        # 创建一个浮点数数组，类型为指定的 code，并在范围内的每个值上断言 .is_integer() 方法返回 True
        float_array = np.arange(-5, 5).astype(code)
        for value in float_array:
            assert value.is_integer()

    # 使用装饰器@parametrize，为测试方法test_false提供参数化测试数据
    @parametrize("code", "bhil")  # np.typecodes["Float"])
    def test_false(self, code: str) -> None:
        # 创建一个整数数组，类型为指定的 code，并将每个值乘以 1.1，然后断言 .is_integer() 方法返回 False
        float_array = np.arange(-5, 5).astype(code)
        float_array *= 1.1
        for value in float_array:
            if value == 0:
                continue
            assert not value.is_integer()


# 使用装饰器@skip，标记该测试类为跳过测试，原因是类型系统的实现细节不同
# 使用装饰器@skipif，当 Python 版本低于 3.9 时跳过该测试类，原因是需要 Python 3.9 及以上版本
# 使用装饰器@instantiate_parametrized_tests，标记该测试类需要实例化参数化测试

@skip(reason="XXX: implementation details of the type system differ")
@skipif(sys.version_info < (3, 9), reason="Requires python 3.9")
@instantiate_parametrized_tests
class TestClassGetItem(TestCase):
    
    # 使用装饰器@parametrize，为测试方法test_abc提供参数化测试数据
    @parametrize(
        "cls",
        [
            np.number,
            np.integer,
            np.inexact,
            np.unsignedinteger,
            np.signedinteger,
            np.floating,
        ],
    )
    def test_abc(self, cls: Type[np.number]) -> None:
        # 使用任意类型 Any 创建 cls 的别名，断言该别名是 types.GenericAlias 类型，并且其原始类型是 cls
        alias = cls[Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is cls

    # 测试方法，验证 np.complexfloating[Any, Any] 是 types.GenericAlias 类型，并且其原始类型是 np.complexfloating
    def test_abc_complexfloating(self) -> None:
        alias = np.complexfloating[Any, Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is np.complexfloating

    # 使用装饰器@parametrize，为测试方法test_abc_complexfloating_subscript_tuple提供参数化测试数据
    @parametrize("arg_len", range(4))
    def test_abc_complexfloating_subscript_tuple(self, arg_len: int) -> None:
        arg_tup = (Any,) * arg_len
        if arg_len in (1, 2):
            assert np.complexfloating[arg_tup]
        else:
            match = f"Too {'few' if arg_len == 0 else 'many'} arguments"
            with pytest.raises(TypeError, match=match):
                np.complexfloating[arg_tup]

    # 使用装饰器@parametrize，为测试方法test_abc_non_numeric提供参数化测试数据
    @parametrize("cls", [np.generic])
    def test_abc_non_numeric(self, cls: Type[np.generic]) -> None:
        # 断言对于任意类型 cls，使用 Any 参数时会引发 TypeError 异常
        with pytest.raises(TypeError):
            cls[Any]

    # 使用装饰器@parametrize，为测试方法test_concrete提供参数化测试数据
    @parametrize("code", np.typecodes["All"])
    def test_concrete(self, code: str) -> None:
        # 获取对应类型代码的 numpy 数据类型对象，并使用 Any 参数时会引发 TypeError 异常
        cls = np.dtype(code).type
        with pytest.raises(TypeError):
            cls[Any]

    # 使用装饰器@parametrize，为测试方法test_subscript_tuple提供参数化测试数据
    @parametrize("arg_len", range(4))
    def test_subscript_tuple(self, arg_len: int) -> None:
        arg_tup = (Any,) * arg_len
        if arg_len == 1:
            assert np.number[arg_tup]
        else:
            with pytest.raises(TypeError):
                np.number[arg_tup]

    # 测试方法，验证 np.number[Any] 是合法的表达式
    def test_subscript_scalar(self) -> None:
        assert np.number[Any]


# 使用装饰器@instantiate_parametrized_tests，标记该测试类需要实例化参数化测试

@instantiate_parametrized_tests
class TestClassGetitemMisc(TestCase):
    pass
    # 使用 pytest 的装饰器 @skipif 根据条件跳过测试，条件是 Python 版本大于等于 3.9
    @skipif(sys.version_info >= (3, 9), reason="Requires python 3.8")
    # 使用 pytest 的装饰器 @parametrize 对测试方法参数化，参数为类的不同类型
    @parametrize("cls", [np.number, np.complexfloating, np.int64])
    # 定义一个测试方法，测试类的 __getitem__ 方法在 Python 3.8 环境下的行为
    def test_class_getitem_38(self, cls: Type[np.number]) -> None:
        # 期望在执行以下代码块时抛出 TypeError 异常
        with pytest.raises(TypeError):
            # 调用类的 __getitem__ 方法，传入任意类型的参数
            cls[Any]


这段代码是一个用于测试类的 `__getitem__` 方法在 Python 3.8 环境下行为的单元测试。其中通过 `@skipif` 装饰器和 `@parametrize` 装饰器来对测试进行条件化和参数化，使用了 `pytest.raises` 来断言是否抛出了期望的异常。
# 装饰器，跳过这个测试类的执行，原因是“scalartype(...).bit_count()”方法未实现
@skip(reason="scalartype(...).bit_count() not implemented")
# 装饰器，实例化参数化测试
@instantiate_parametrized_tests
# 测试类，继承自 TestCase 类
class TestBitCount(TestCase):
    # 从 CPython 的测试 "test_bit_count" 中衍生出的一部分

    # 参数化测试方法，测试输入类型为整数和无符号整数
    @parametrize("itype", np.sctypes["int"] + np.sctypes["uint"])
    def test_small(self, itype):
        # 对于每种整数类型，从最小值（若为正数则从零开始）迭代到 128
        for a in range(max(np.iinfo(itype).min, 0), 128):
            # 构建消息字符串，用于描述当前的 smoke test
            msg = f"Smoke test for {itype}({a}).bit_count()"
            # 断言当前整数类型的 bit_count() 方法计算结果与 Python 内置 bin(a).count("1") 的结果一致
            assert itype(a).bit_count() == bin(a).count("1"), msg

    # 测试 bit_count() 方法的具体行为
    def test_bit_count(self):
        # 对于指定的指数列表进行迭代
        for exp in [10, 17, 63]:
            a = 2**exp
            # 断言对于 2**exp 的无符号 64 位整数，bit_count() 结果为 1
            assert np.uint64(a).bit_count() == 1
            # 断言对于 2**exp - 1 的无符号 64 位整数，bit_count() 结果为 exp
            assert np.uint64(a - 1).bit_count() == exp
            # 断言对于 2**exp 和 63 的异或结果的无符号 64 位整数，bit_count() 结果为 7
            assert np.uint64(a ^ 63).bit_count() == 7
            # 断言对于 (2**exp - 1) 和 510 的异或结果的无符号 64 位整数，bit_count() 结果为 exp - 8
            assert np.uint64((a - 1) ^ 510).bit_count() == exp - 8

# 如果作为主程序执行，则运行测试
if __name__ == "__main__":
    run_tests()
```
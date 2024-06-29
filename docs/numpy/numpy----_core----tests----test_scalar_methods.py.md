# `.\numpy\numpy\_core\tests\test_scalar_methods.py`

```py
"""
Test the scalar constructors, which also do type-coercion
"""
# 导入所需的模块和库
import fractions  # 导入 fractions 模块
import platform  # 导入 platform 模块
import types  # 导入 types 模块
from typing import Any, Type  # 从 typing 模块导入 Any 和 Type 类型

import pytest  # 导入 pytest 测试框架
import numpy as np  # 导入 NumPy 库

from numpy._core import sctypes  # 从 NumPy 内核中导入 sctypes
from numpy.testing import assert_equal, assert_raises, IS_MUSL  # 从 NumPy 测试模块导入断言和异常处理相关函数


class TestAsIntegerRatio:
    # derived in part from the cpython test "test_floatasratio"

    @pytest.mark.parametrize("ftype", [
        np.half, np.single, np.double, np.longdouble])
    @pytest.mark.parametrize("f, ratio", [
        (0.875, (7, 8)),
        (-0.875, (-7, 8)),
        (0.0, (0, 1)),
        (11.5, (23, 2)),
        ])
    def test_small(self, ftype, f, ratio):
        # 断言浮点数转化为分数的正确性
        assert_equal(ftype(f).as_integer_ratio(), ratio)

    @pytest.mark.parametrize("ftype", [
        np.half, np.single, np.double, np.longdouble])
    def test_simple_fractions(self, ftype):
        R = fractions.Fraction
        # 断言简单浮点数转化为分数的正确性
        assert_equal(R(0, 1),
                     R(*ftype(0.0).as_integer_ratio()))
        assert_equal(R(5, 2),
                     R(*ftype(2.5).as_integer_ratio()))
        assert_equal(R(1, 2),
                     R(*ftype(0.5).as_integer_ratio()))
        assert_equal(R(-2100, 1),
                     R(*ftype(-2100.0).as_integer_ratio()))

    @pytest.mark.parametrize("ftype", [
        np.half, np.single, np.double, np.longdouble])
    def test_errors(self, ftype):
        # 测试处理特定浮点数值时的错误情况
        assert_raises(OverflowError, ftype('inf').as_integer_ratio)
        assert_raises(OverflowError, ftype('-inf').as_integer_ratio)
        assert_raises(ValueError, ftype('nan').as_integer_ratio)

    def test_against_known_values(self):
        R = fractions.Fraction
        # 断言浮点数转化为分数的正确性，与已知值进行对比
        assert_equal(R(1075, 512),
                     R(*np.half(2.1).as_integer_ratio()))
        assert_equal(R(-1075, 512),
                     R(*np.half(-2.1).as_integer_ratio()))
        assert_equal(R(4404019, 2097152),
                     R(*np.single(2.1).as_integer_ratio()))
        assert_equal(R(-4404019, 2097152),
                     R(*np.single(-2.1).as_integer_ratio()))
        assert_equal(R(4728779608739021, 2251799813685248),
                     R(*np.double(2.1).as_integer_ratio()))
        assert_equal(R(-4728779608739021, 2251799813685248),
                     R(*np.double(-2.1).as_integer_ratio()))
        # longdouble is platform dependent
    @pytest.mark.parametrize("ftype, frac_vals, exp_vals", [
        # 定义测试参数化，包括数据类型、小数部分值和指数部分值
        # 使用假设生成的数据类型测试用例
        # 每种数据类型的前五个生成的情况
        (np.half, [0.0, 0.01154830649280303, 0.31082276347447274,
                   0.527350517124794, 0.8308562335072596],
                  [0, 1, 0, -8, 12]),
        (np.single, [0.0, 0.09248576989263226, 0.8160498218131407,
                     0.17389442853722373, 0.7956044195067877],
                    [0, 12, 10, 17, -26]),
        (np.double, [0.0, 0.031066908499895136, 0.5214135908877832,
                     0.45780736035689296, 0.5906586745934036],
                    [0, -801, 51, 194, -653]),
        # 使用 pytest.param 定义额外的测试参数
        pytest.param(
            np.longdouble,
            [0.0, 0.20492557202724854, 0.4277180662199366, 0.9888085019891495,
             0.9620175814461964],
            [0, -7400, 14266, -7822, -8721],
            marks=[
                # 添加跳过条件的标记，如果 np.double 和 np.longdouble 的信息相同，则跳过
                pytest.mark.skipif(
                    np.finfo(np.double) == np.finfo(np.longdouble),
                    reason="long double is same as double"),
                # 添加跳过条件的标记，如果平台机器名以 "ppc" 开头，则跳过，原因是 IBM 双倍精度
                pytest.mark.skipif(
                    platform.machine().startswith("ppc"),
                    reason="IBM double double"),
            ]
        )
    ])
    def test_roundtrip(self, ftype, frac_vals, exp_vals):
        # 遍历小数部分值和指数部分值
        for frac, exp in zip(frac_vals, exp_vals):
            # 使用 ldexp 函数创建浮点数
            f = np.ldexp(ftype(frac), exp)
            # 断言生成的浮点数的数据类型与预期的数据类型相同
            assert f.dtype == ftype
            # 将浮点数转换为分数形式
            n, d = f.as_integer_ratio()

            try:
                # 尝试将分子和分母转换为 longdouble 类型
                nf = np.longdouble(n)
                df = np.longdouble(d)
                # 如果分母不是有限的浮点数，则抛出溢出错误
                if not np.isfinite(df):
                    raise OverflowError
            except (OverflowError, RuntimeWarning):
                # 如果在此平台上 longdouble 太小，则跳过测试
                pytest.skip("longdouble too small on this platform")

            # 断言分数形式的值与原始浮点数相等
            assert_equal(nf / df, f, "{}/{}".format(n, d))
class TestIsInteger:
    # 定义测试类 TestIsInteger，用于测试浮点数和整数类型的方法 is_integer()

    @pytest.mark.parametrize("str_value", ["inf", "nan"])
    @pytest.mark.parametrize("code", np.typecodes["Float"])
    def test_special(self, code: str, str_value: str) -> None:
        # 测试特殊值，如无穷大和NaN，对应于浮点数类型
        cls = np.dtype(code).type
        # 根据类型码创建数据类型对象
        value = cls(str_value)
        # 使用类型对象创建具体数值对象
        assert not value.is_integer()
        # 断言该数值对象不是整数

    @pytest.mark.parametrize(
        "code", np.typecodes["Float"] + np.typecodes["AllInteger"]
    )
    def test_true(self, code: str) -> None:
        # 测试各种浮点数和整数类型的数据是否是整数
        float_array = np.arange(-5, 5).astype(code)
        # 创建一个指定类型码的数组，范围从-5到4
        for value in float_array:
            assert value.is_integer()
            # 断言每个数组元素是整数

    @pytest.mark.parametrize("code", np.typecodes["Float"])
    def test_false(self, code: str) -> None:
        # 测试浮点数类型的数据是否不是整数
        float_array = np.arange(-5, 5).astype(code)
        # 创建一个指定类型码的数组，范围从-5到4
        float_array *= 1.1
        # 数组中的每个元素乘以1.1
        for value in float_array:
            if value == 0:
                continue
            assert not value.is_integer()
            # 断言每个乘以1.1后的数组元素不是整数


class TestClassGetItem:
    # 定义测试类 TestClassGetItem，用于测试 numpy 类型别名和数据类型对象的行为

    @pytest.mark.parametrize("cls", [
        np.number,
        np.integer,
        np.inexact,
        np.unsignedinteger,
        np.signedinteger,
        np.floating,
    ])
    def test_abc(self, cls: Type[np.number]) -> None:
        # 测试 numpy 的类型别名的基本行为
        alias = cls[Any]
        # 使用 cls 创建泛型别名
        assert isinstance(alias, types.GenericAlias)
        # 断言别名是泛型别名类型
        assert alias.__origin__ is cls
        # 断言别名的原始类型是 cls

    def test_abc_complexfloating(self) -> None:
        # 测试 np.complexfloating 的类型别名行为
        alias = np.complexfloating[Any, Any]
        # 创建复数浮点数类型别名
        assert isinstance(alias, types.GenericAlias)
        # 断言别名是泛型别名类型
        assert alias.__origin__ is np.complexfloating
        # 断言别名的原始类型是 np.complexfloating

    @pytest.mark.parametrize("arg_len", range(4))
    def test_abc_complexfloating_subscript_tuple(self, arg_len: int) -> None:
        # 测试复数浮点数类型别名的下标元组行为
        arg_tup = (Any,) * arg_len
        # 创建由参数长度决定的 Any 元组
        if arg_len in (1, 2):
            assert np.complexfloating[arg_tup]
            # 如果参数长度为 1 或 2，断言能够使用参数元组索引复数浮点数类型
        else:
            match = f"Too {'few' if arg_len == 0 else 'many'} arguments"
            # 根据参数长度生成匹配错误消息
            with pytest.raises(TypeError, match=match):
                np.complexfloating[arg_tup]
                # 断言在其他情况下索引复数浮点数类型会引发类型错误异常

    @pytest.mark.parametrize("cls", [np.generic, np.flexible, np.character])
    def test_abc_non_numeric(self, cls: Type[np.generic]) -> None:
        # 测试非数值型别名的行为
        with pytest.raises(TypeError):
            cls[Any]
            # 断言创建非数值型别名会引发类型错误异常

    @pytest.mark.parametrize("code", np.typecodes["All"])
    def test_concrete(self, code: str) -> None:
        # 测试具体数据类型对象的行为
        cls = np.dtype(code).type
        # 创建指定类型码的数据类型对象
        with pytest.raises(TypeError):
            cls[Any]
            # 断言创建具体数据类型对象会引发类型错误异常

    @pytest.mark.parametrize("arg_len", range(4))
    def test_subscript_tuple(self, arg_len: int) -> None:
        # 测试数据类型对象的下标元组行为
        arg_tup = (Any,) * arg_len
        # 创建由参数长度决定的 Any 元组
        if arg_len == 1:
            assert np.number[arg_tup]
            # 断言能够使用参数元组索引数据类型对象
        else:
            with pytest.raises(TypeError):
                np.number[arg_tup]
                # 断言在其他情况下索引数据类型对象会引发类型错误异常

    def test_subscript_scalar(self) -> None:
        # 测试数据类型对象的标量下标行为
        assert np.number[Any]
        # 断言能够使用标量形式的 Any 索引数据类型对象


class TestBitCount:
    # 定义测试类 TestBitCount，部分派生自 cpython 的 "test_bit_count" 测试

    @pytest.mark.parametrize("itype", sctypes['int']+sctypes['uint'])
    # 参数化整数和无符号整数类型
    # 定义一个测试方法，用于对指定整数类型进行小规模测试
    def test_small(self, itype):
        # 使用整数类型的最小非负值和 0 之间的范围进行迭代
        for a in range(max(np.iinfo(itype).min, 0), 128):
            # 构建测试消息字符串，用于显示当前测试的信息
            msg = f"Smoke test for {itype}({a}).bit_count()"
            # 断言当前整数类型的位数计算结果等于其二进制表示中 '1' 的个数，若不等则输出消息
            assert itype(a).bit_count() == bin(a).count("1"), msg

    # 定义位数计算方法的测试方法
    def test_bit_count(self):
        # 对指定的指数列表进行迭代
        for exp in [10, 17, 63]:
            # 计算 2 的 exp 次方，并将结果转换为 np.uint64 类型的整数 a
            a = 2**exp
            # 断言 np.uint64 类型的 a 的位数计算结果等于 1，若不等则抛出异常
            assert np.uint64(a).bit_count() == 1
            # 断言 np.uint64 类型的 (a - 1) 的位数计算结果等于 exp，若不等则抛出异常
            assert np.uint64(a - 1).bit_count() == exp
            # 断言 np.uint64 类型的 (a ^ 63) 的位数计算结果等于 7，若不等则抛出异常
            assert np.uint64(a ^ 63).bit_count() == 7
            # 断言 np.uint64 类型的 ((a - 1) ^ 510) 的位数计算结果等于 exp - 8，若不等则抛出异常
            assert np.uint64((a - 1) ^ 510).bit_count() == exp - 8
```
# `.\pytorch\test\test_sympy_utils.py`

```py
# Owner(s): ["oncall: pt2"]

import itertools  # 导入 itertools 库，用于生成迭代器的工具函数
import math  # 导入 math 库，提供数学运算函数
import sys  # 导入 sys 库，提供对解释器相关的系统调用接口

import sympy  # 导入 sympy 库，用于符号计算
from typing import Callable, List, Tuple, Type  # 从 typing 模块导入类型提示相关的工具
from torch.testing._internal.common_device_type import skipIf  # 导入 torch 的测试相关模块
from torch.testing._internal.common_utils import (
    TEST_Z3,  # 从 torch 的测试工具模块导入常用测试相关常量和函数
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import FloorDiv  # 导入 torch 的符号计算工具模块中的 FloorDiv 函数
from torch.utils._sympy.solve import (  # 导入 torch 的符号计算工具模块中的解决器相关函数
    INEQUALITY_TYPES,
    mirror_rel_op,
    try_solve,
)
from torch.utils._sympy.value_ranges import (  # 导入 torch 的符号计算工具模块中的值范围分析相关类
    ValueRangeAnalysis,
    ValueRanges,
)
from torch.utils._sympy.reference import (  # 导入 torch 的符号计算工具模块中的引用分析相关类
    ReferenceAnalysis,
    PythonReferenceAnalysis,
)
from torch.utils._sympy.interp import sympy_interp  # 导入 torch 的符号计算工具模块中的插值函数
from torch.utils._sympy.singleton_int import SingletonInt  # 导入 torch 的符号计算工具模块中的单例整数类
from torch.utils._sympy.numbers import (  # 导入 torch 的符号计算工具模块中的数值类型
    int_oo,
    IntInfinity,
    NegativeIntInfinity,
)
from sympy.core.relational import (  # 导入 sympy 库中的关系运算相关类和函数
    is_ge,
    is_le,
    is_gt,
    is_lt,
)
import functools  # 导入 functools 库，提供高阶函数和操作工具
import torch.fx as fx  # 导入 torch 的特效模块，用于构建和操作有向计算图


UNARY_OPS = [  # 定义包含一元运算名称的列表常量
    "reciprocal",  # 倒数运算
    "square",  # 平方运算
    "abs",  # 绝对值运算
    "neg",  # 取负运算
    "exp",  # 指数运算
    "log",  # 对数运算
    "sqrt",  # 平方根运算
    "floor",  # 向下取整运算
    "ceil",  # 向上取整运算
]
BINARY_OPS = [  # 定义包含二元运算名称的列表常量
    "truediv",  # 真除法运算
    "floordiv",  # 向下取整除法运算
    # "truncdiv",  # TODO: 未完成的截断除法运算
    # NB: pow is float_pow
    "add",  # 加法运算
    "mul",  # 乘法运算
    "sub",  # 减法运算
    "pow",  # 幂运算
    "pow_by_natural",  # 自然数幂运算
    "minimum",  # 最小值运算
    "maximum",  # 最大值运算
    "mod",  # 取模运算
]

UNARY_BOOL_OPS = ["not_"]  # 定义包含一元布尔运算名称的列表常量
BINARY_BOOL_OPS = ["or_", "and_"]  # 定义包含二元布尔运算名称的列表常量
COMPARE_OPS = ["eq", "ne", "lt", "gt", "le", "ge"]  # 定义包含比较运算符名称的列表常量

# 一组常量，包括常数、2 的幂次方和质数等
CONSTANTS = [
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    8,
    16,
    32,
    64,
    100,
    101,
    2**24,
    2**32,
    2**37 - 1,
    sys.maxsize - 1,
    sys.maxsize,
]
# 用于 N^2 情况的较少常量集合
LESS_CONSTANTS = [-1, 0, 1, 2, 100]
# SymPy 的关系类型列表
RELATIONAL_TYPES = [sympy.Eq, sympy.Ne, sympy.Gt, sympy.Ge, sympy.Lt, sympy.Le]


def valid_unary(fn, v):
    # 检查一元运算是否有效，如对数运算要求 v 大于 0
    if fn == "log" and v <= 0:
        return False
    # 倒数运算要求 v 不等于 0
    elif fn == "reciprocal" and v == 0:
        return False
    # 平方根运算要求 v 大于等于 0
    elif fn == "sqrt" and v < 0:
        return False
    return True


def valid_binary(fn, a, b):
    # 检查二元运算是否有效，如幂运算要求 b 不过大，或者底数 a 为正
    if fn == "pow" and (
        b > 4  # 对于整数 b，避免扩展成 x*x*... 的形式
        or a <= 0  # 不允许底数为负数
        or (a == b == 0)  # 0 的 0 次幂未定义
    ):
        return False
    # 自然数幂运算要求 b 不过大，不小于 0，且底数 a 不能为 0
    elif fn == "pow_by_natural" and (
        b > 4  # 对于整数 b，避免扩展成 x*x*... 的形式
        or b < 0  # 指数 b 必须为自然数
        or (a == b == 0)  # 0 的 0 次幂未定义
    ):
        return False
    # 取模运算要求 a 大于等于 0，且除数 b 大于 0
    elif fn == "mod" and (a < 0 or b <= 0):
        return False
    # 浮点数和整数除法要求除数 b 不为 0
    elif (fn in ["div", "truediv", "floordiv"]) and b == 0:
        return False
    return True


def generate_range(vals):
    # 使用 itertools.product 生成 vals 中元素的笛卡尔积，并迭代每对元素 (a1, a2)
    for a1, a2 in itertools.product(vals, repeat=2):
        # 检查 a1 是否为 sympy.true 或 sympy.false
        if a1 in [sympy.true, sympy.false]:
            # 如果 a1 为 sympy.true 并且 a2 为 sympy.false，则跳过本次迭代
            if a1 == sympy.true and a2 == sympy.false:
                continue
        else:
            # 如果 a1 不是 sympy.true 或 sympy.false，则比较 a1 和 a2 的大小
            if a1 > a2:
                # 如果 a1 大于 a2，则跳过本次迭代
                continue
        # 检查是否存在只允许无限值的范围，如果是，则跳过本次迭代
        if a1 == sympy.oo or a2 == -sympy.oo:
            continue
        # 生成一个 ValueRanges 对象，并将其作为生成器的输出
        yield ValueRanges(a1, a2)
# 定义一个测试类 TestNumbers，继承自 TestCase
class TestNumbers(TestCase):
    
    # 定义测试方法 test_int_infinity，测试无限整数类型 IntInfinity
    def test_int_infinity(self):
        # 断言 int_oo 是 IntInfinity 类型的实例
        self.assertIsInstance(int_oo, IntInfinity)
        # 断言 -int_oo 是 NegativeIntInfinity 类型的实例
        self.assertIsInstance(-int_oo, NegativeIntInfinity)
        # 断言 int_oo 是整数类型
        self.assertTrue(int_oo.is_integer)
        
        # 下面的操作是为了测试对象的单例性质，不应用于数字的比较
        
        # 断言 int_oo 加上 int_oo 还是 int_oo 自身
        self.assertIs(int_oo + int_oo, int_oo)
        # 断言 int_oo 加上 1 还是 int_oo 自身
        self.assertIs(int_oo + 1, int_oo)
        # 断言 int_oo 减去 1 还是 int_oo 自身
        self.assertIs(int_oo - 1, int_oo)
        # 断言 -int_oo 减去 1 还是 -int_oo
        self.assertIs(-int_oo - 1, -int_oo)
        # 断言 -int_oo 加上 1 还是 -int_oo
        self.assertIs(-int_oo + 1, -int_oo)
        # 断言 -int_oo 加上 -int_oo 还是 -int_oo
        self.assertIs(-int_oo + (-int_oo), -int_oo)
        # 断言 -int_oo 减去 int_oo 还是 -int_oo
        self.assertIs(-int_oo - int_oo, -int_oo)
        # 断言 1 加上 int_oo 还是 int_oo
        self.assertIs(1 + int_oo, int_oo)
        # 断言 1 减去 int_oo 还是 -int_oo
        self.assertIs(1 - int_oo, -int_oo)
        # 断言 int_oo 乘以 int_oo 还是 int_oo
        self.assertIs(int_oo * int_oo, int_oo)
        # 断言 2 乘以 int_oo 还是 int_oo
        self.assertIs(2 * int_oo, int_oo)
        # 断言 int_oo 乘以 2 还是 int_oo
        self.assertIs(int_oo * 2, int_oo)
        # 断言 -1 乘以 int_oo 还是 -int_oo
        self.assertIs(-1 * int_oo, -int_oo)
        # 断言 -int_oo 乘以 int_oo 还是 -int_oo
        self.assertIs(-int_oo * int_oo, -int_oo)
        # 断言 2 乘以 -int_oo 还是 -int_oo
        self.assertIs(2 * -int_oo, -int_oo)
        # 断言 -int_oo 乘以 2 还是 -int_oo
        self.assertIs(-int_oo * 2, -int_oo)
        # 断言 -1 乘以 -int_oo 还是 int_oo
        self.assertIs(-1 * -int_oo, int_oo)
        # 断言 int_oo 除以 2 还是 sympy.oo (正无穷)
        self.assertIs(int_oo / 2, sympy.oo)
        # 断言 -(-int_oo) 还是 int_oo
        self.assertIs(-(-int_oo), int_oo)  # noqa: B002
        # 断言 int_oo 的绝对值还是 int_oo
        self.assertIs(abs(int_oo), int_oo)
        # 断言 -int_oo 的绝对值还是 int_oo
        self.assertIs(abs(-int_oo), int_oo)
        # 断言 int_oo 的平方还是 int_oo
        self.assertIs(int_oo ** 2, int_oo)
        # 断言 -int_oo 的平方还是 int_oo
        self.assertIs((-int_oo) ** 2, int_oo)
        # 断言 -int_oo 的立方还是 -int_oo
        self.assertIs((-int_oo) ** 3, -int_oo)
        # 断言 int_oo 的倒数是 0
        self.assertEqual(int_oo ** -1, 0)
        # 断言 -int_oo 的倒数是 0
        self.assertEqual((-int_oo) ** -1, 0)
        # 断言 int_oo 的 int_oo 次方还是 int_oo
        self.assertIs(int_oo ** int_oo, int_oo)
        # 断言 int_oo 等于 int_oo
        self.assertTrue(int_oo == int_oo)
        # 断言 int_oo 不等于 int_oo
        self.assertFalse(int_oo != int_oo)
        # 断言 -int_oo 等于 -int_oo
        self.assertTrue(-int_oo == -int_oo)
        # 断言 int_oo 不等于 2
        self.assertFalse(int_oo == 2)
        # 断言 int_oo 不等于 2
        self.assertTrue(int_oo != 2)
        # 断言 int_oo 不等于 sys.maxsize
        self.assertFalse(int_oo == sys.maxsize)
        # 断言 int_oo 大于等于 sys.maxsize
        self.assertTrue(int_oo >= sys.maxsize)
        # 断言 int_oo 大于等于 2
        self.assertTrue(int_oo >= 2)
        # 断言 int_oo 大于等于 -int_oo
        self.assertTrue(int_oo >= -int_oo)

    # 定义测试方法 test_relation，测试关系操作
    def test_relation(self):
        # 断言 sympy.Add(2, int_oo) 等于 int_oo
        self.assertIs(sympy.Add(2, int_oo), int_oo)
        # 断言 -int_oo 不大于 2
        self.assertFalse(-int_oo > 2)

    # 定义测试方法 test_lt_self，测试小于操作
    def test_lt_self(self):
        # 断言 int_oo 不小于 int_oo
        self.assertFalse(int_oo < int_oo)
        # 断言 -4 和 -int_oo 中较小的是 -int_oo
        self.assertIs(min(-int_oo, -4), -int_oo)
        # 断言 -int_oo 和 -int_oo 中较小的是 -int_oo
        self.assertIs(min(-int_oo, -int_oo), -int_oo)

    # 定义测试方法 test_float_cast，测试转换为浮点数操作
    def test_float_cast(self):
        # 断言将 int_oo 转换为浮点数是正无穷 math.inf
        self.assertEqual(float(int_oo), math.inf)
        # 断言将 -int_oo 转换为浮点数是负无穷 -math.inf
        self.assertEqual(float(-int_oo), -math.inf)

    # 定义测试方法 test_mixed_oo_int_oo，测试混合操作
    def test_mixed_oo_int_oo(self):
        # 任意选择的断言，断言 int_oo 小于 sympy.oo (正无穷)
        self.assertTrue(int_oo < sympy.oo)
        # 断言 int_oo 不大于 sympy.oo (正无穷)
        self.assertFalse(int_oo > sympy.oo)
        # 断言 sympy.oo 大于 int_oo
        self.assertTrue(sympy.oo > int_oo)
        # 断言 sympy.oo 不小于 int_oo
        self.assertFalse(sympy.oo < int_oo)
        # 断言 int_oo 和 sympy.oo 中较大的是 sympy.oo
        self.assertIs(max(int_oo, sympy.oo), sympy.oo)
        # 断言 -int_oo 大于 -sympy.oo
        self.assertTrue(-int_oo > -sympy.oo)
        # 断言 -int_oo 和 -sympy.oo 中较小的是 -sympy.oo
        self.assertIs(min(-int_oo, -sympy.oo), -sympy.oo)
    # 测试一元引用函数，验证对给定数据类型的一元操作
    def test_unary_ref(self, fn, dtype):
        # 根据数据类型选择对应的 sympy 类型
        dtype = {"int": sympy.Integer, "float": sympy.Float}[dtype]
        # 遍历预定义的常量集合
        for v in CONSTANTS:
            # 如果不满足一元操作的有效性条件，则跳过
            if not valid_unary(fn, v):
                continue
            # 使用子测试来处理每个常量值 v
            with self.subTest(v=v):
                # 将常量 v 转换为指定的数据类型
                v = dtype(v)
                # 调用 ReferenceAnalysis 类的相应函数，计算参考结果
                ref_r = getattr(ReferenceAnalysis, fn)(v)
                # 调用 ValueRangeAnalysis 类的相应函数，计算测试结果
                r = getattr(ValueRangeAnalysis, fn)(v)
                # 断言测试结果的下界和上界是否为整数
                self.assertEqual(r.lower.is_integer, r.upper.is_integer)
                # 断言测试结果的下界和上界是否相等
                self.assertEqual(r.lower, r.upper)
                # 断言参考结果是否为整数
                self.assertEqual(ref_r.is_integer, r.upper.is_integer)
                # 断言参考结果是否与测试结果的下界相等
                self.assertEqual(ref_r, r.lower)

    # 测试指数运算的一半
    def test_pow_half(self):
        # 调用 ValueRangeAnalysis 类的 pow 方法，计算未知值的指数运算结果
        ValueRangeAnalysis.pow(ValueRanges.unknown(), ValueRanges.wrap(0.5))

    # 参数化测试二元引用函数，验证对给定数据类型的二元操作
    @parametrize("fn", BINARY_OPS)
    @parametrize("dtype", ("int", "float"))
    def test_binary_ref(self, fn, dtype):
        to_dtype = {"int": sympy.Integer, "float": sympy.Float}
        # 不对浮点数进行整数方法的测试
        if dtype == "float" and fn in ["pow_by_natural", "mod"]:
            return
        # 根据数据类型选择对应的 sympy 类型
        dtype = to_dtype[dtype]
        # 使用 itertools.product 生成常量的所有二元组合
        for a, b in itertools.product(CONSTANTS, repeat=2):
            # 如果不满足二元操作的有效性条件，则跳过
            if not valid_binary(fn, a, b):
                continue
            # 将常量 a 和 b 转换为指定的数据类型
            a = dtype(a)
            b = dtype(b)
            # 使用子测试来处理每对常量 a, b
            with self.subTest(a=a, b=b):
                # 调用 ValueRangeAnalysis 类的相应函数，计算测试结果
                r = getattr(ValueRangeAnalysis, fn)(a, b)
                # 如果测试结果为未知，则跳过后续断言
                if r == ValueRanges.unknown():
                    continue
                # 调用 ReferenceAnalysis 类的相应函数，计算参考结果
                ref_r = getattr(ReferenceAnalysis, fn)(a, b)

                # 断言测试结果的下界和上界是否为整数
                self.assertEqual(r.lower.is_integer, r.upper.is_integer)
                # 断言参考结果的下界和上界是否为整数
                self.assertEqual(ref_r.is_integer, r.upper.is_integer)
                # 断言测试结果的下界和上界是否相等
                self.assertEqual(r.lower, r.upper)
                # 断言参考结果是否与测试结果的下界相等
                self.assertEqual(ref_r, r.lower)

    # 测试乘法运算中的零乘以未知值的情况
    def test_mul_zero_unknown(self):
        # 断言零乘以未知值的结果为零
        self.assertEqual(
            ValueRangeAnalysis.mul(ValueRanges.wrap(0), ValueRanges.unknown()),
            ValueRanges.wrap(0),
        )

    # 参数化测试一元布尔操作函数，验证对给定布尔值范围的一元布尔操作
    @parametrize("fn", UNARY_BOOL_OPS)
    def test_unary_bool_ref_range(self, fn):
        # 定义布尔值的集合
        vals = [sympy.false, sympy.true]
        # 生成布尔值范围的测试数据
        for a in generate_range(vals):
            # 使用子测试来处理每个布尔值 a
            with self.subTest(a=a):
                # 调用 ValueRangeAnalysis 类的相应函数，计算测试结果
                ref_r = getattr(ValueRangeAnalysis, fn)(a)
                unique = set()
                # 遍历布尔值集合中的每个布尔值
                for a0 in vals:
                    # 如果当前布尔值不在测试数据中，则跳过
                    if a0 not in a:
                        continue
                    # 使用子测试来处理每个布尔值 a0
                    with self.subTest(a0=a0):
                        # 调用 ReferenceAnalysis 类的相应函数，计算参考结果
                        r = getattr(ReferenceAnalysis, fn)(a0)
                        # 断言参考结果是否在测试结果中
                        self.assertIn(r, ref_r)
                        # 将参考结果添加到唯一结果集合中
                        unique.add(r)
                # 如果测试结果的下界和上界相等，则唯一结果集合中应有一个元素
                if ref_r.lower == ref_r.upper:
                    self.assertEqual(len(unique), 1)
                # 否则唯一结果集合中应有两个元素
                else:
                    self.assertEqual(len(unique), 2)

    # 参数化测试二元布尔操作函数
    # 定义测试方法，用于测试二进制布尔参考范围的函数
    def test_binary_bool_ref_range(self, fn):
        # 初始值设定为布尔常量 False 和 True
        vals = [sympy.false, sympy.true]
        # 生成 vals 的笛卡尔积，对每个组合(a, b)调用生成范围函数
        for a, b in itertools.product(generate_range(vals), repeat=2):
            # 使用子测试检查每个(a, b)组合
            with self.subTest(a=a, b=b):
                # 调用 ValueRangeAnalysis 类中的指定函数 fn，返回参考分析的结果
                ref_r = getattr(ValueRangeAnalysis, fn)(a, b)
                # 用于存储唯一结果的集合
                unique = set()
                # 生成 vals 的笛卡尔积，对每个组合(a0, b0)调用生成范围函数
                for a0, b0 in itertools.product(vals, repeat=2):
                    # 如果 a0 不在 a 中或者 b0 不在 b 中，跳过本次循环
                    if a0 not in a or b0 not in b:
                        continue
                    # 使用子测试检查每个(a0, b0)组合
                    with self.subTest(a0=a0, b0=b0):
                        # 调用 ReferenceAnalysis 类中的指定函数 fn，返回参考分析的结果
                        r = getattr(ReferenceAnalysis, fn)(a0, b0)
                        # 断言 r 在 ref_r 中
                        self.assertIn(r, ref_r)
                        # 将 r 添加到唯一结果集合中
                        unique.add(r)
                # 如果参考范围的下界等于上界，断言唯一结果集合的长度为 1
                if ref_r.lower == ref_r.upper:
                    self.assertEqual(len(unique), 1)
                # 否则，断言唯一结果集合的长度为 2
                else:
                    self.assertEqual(len(unique), 2)

    @parametrize("fn", UNARY_OPS)
    # 定义测试方法，用于测试一元参考范围的函数
    def test_unary_ref_range(self, fn):
        # TODO: bring back sympy.oo testing for float unary fns
        # 初始值设定为常量集合 CONSTANTS
        vals = CONSTANTS
        # 生成 vals 的笛卡尔积，对每个元素 a 调用生成范围函数
        for a in generate_range(vals):
            # 使用子测试检查每个元素 a
            with self.subTest(a=a):
                # 调用 ValueRangeAnalysis 类中的指定函数 fn，返回参考分析的结果
                ref_r = getattr(ValueRangeAnalysis, fn)(a)
                # 生成 CONSTANTS 的笛卡尔积，对每个元素 a0 调用生成范围函数
                for a0 in CONSTANTS:
                    # 如果 a0 不在 a 中，跳过本次循环
                    if a0 not in a:
                        continue
                    # 如果对于该一元函数和 a0 不合法，跳过本次循环
                    if not valid_unary(fn, a0):
                        continue
                    # 使用子测试检查每个元素 a0
                    with self.subTest(a0=a0):
                        # 调用 ReferenceAnalysis 类中的指定函数 fn，返回参考分析的结果
                        r = getattr(ReferenceAnalysis, fn)(sympy.Integer(a0))
                        # 断言 r 在 ref_r 中
                        self.assertIn(r, ref_r)

    # This takes about 4s for all the variants
    @parametrize("fn", BINARY_OPS + COMPARE_OPS)
    # 定义测试方法，用于测试二元参考范围的函数
    def test_binary_ref_range(self, fn):
        # TODO: bring back sympy.oo testing for float unary fns
        # 初始值设定为常量集合 LESS_CONSTANTS
        vals = LESS_CONSTANTS
        # 生成 vals 的笛卡尔积，对每个组合(a, b)调用生成范围函数
        for a, b in itertools.product(generate_range(vals), repeat=2):
            # 对于 pow 函数，如果指数 b.upper 太大，则跳过本次循环（但 oo 是可以接受的）
            if fn == "pow" and b.upper > 4 and b.upper != sympy.oo:
                continue
            # 使用子测试检查每个组合(a, b)
            with self.subTest(a=a, b=b):
                # 生成 LESS_CONSTANTS 的笛卡尔积，对每个组合(a0, b0)调用生成范围函数
                for a0, b0 in itertools.product(LESS_CONSTANTS, repeat=2):
                    # 如果 a0 不在 a 中或者 b0 不在 b 中，跳过本次循环
                    if a0 not in a or b0 not in b:
                        continue
                    # 如果对于该二元函数和 a0, b0 不合法，跳过本次循环
                    if not valid_binary(fn, a0, b0):
                        continue
                    # 使用子测试检查每个组合(a0, b0)
                    with self.subTest(a0=a0, b0=b0):
                        # 调用 ValueRangeAnalysis 类中的指定函数 fn，返回参考分析的结果
                        ref_r = getattr(ValueRangeAnalysis, fn)(a, b)
                        # 调用 ReferenceAnalysis 类中的指定函数 fn，返回参考分析的结果
                        r = getattr(ReferenceAnalysis, fn)(
                            sympy.Integer(a0), sympy.Integer(b0)
                        )
                        # 如果 r 是有限的，断言 r 在 ref_r 中
                        if r.is_finite:
                            self.assertIn(r, ref_r)
# 定义一个测试类 TestSympyInterp，继承自 TestCase
class TestSympyInterp(TestCase):

    # 使用参数化装饰器 parametrize，参数 fn 取自 UNARY_OPS、BINARY_OPS、UNARY_BOOL_OPS、BINARY_BOOL_OPS、COMPARE_OPS 中的值
    @parametrize("fn", UNARY_OPS + BINARY_OPS + UNARY_BOOL_OPS + BINARY_BOOL_OPS + COMPARE_OPS)
    # 定义测试方法 test_interp，参数 fn 是从 parametrize 中传入的
    def test_interp(self, fn):
        # 如果 fn 是以下操作之一，则跳过此次测试：'div', 'truncdiv', 'minimum', 'maximum', 'mod'
        # 因为 SymPy 没有实现这些表达式的截断
        if fn in ("div", "truncdiv", "minimum", "maximum", "mod"):
            return

        # 初始化 is_integer 变量为 None
        is_integer = None
        # 如果 fn 是 "pow_by_natural"，则设置 is_integer 为 True
        if fn == "pow_by_natural":
            is_integer = True

        # 创建 SymPy 的虚拟变量 x 和 y，这些变量可以是整数（如果 is_integer 为 True）
        x = sympy.Dummy('x', integer=is_integer)
        y = sympy.Dummy('y', integer=is_integer)

        # 初始化 vals 为 CONSTANTS，这是一个包含常量值的列表
        # 如果 fn 是一元或二元布尔操作之一，则将 vals 设置为 [True, False]
        if fn in {*UNARY_BOOL_OPS, *BINARY_BOOL_OPS}:
            vals = [True, False]

        # 初始化 arity 为 1
        # 如果 fn 是二元操作、二元布尔操作或比较操作之一，则将 arity 设置为 2
        arity = 1
        if fn in {*BINARY_OPS, *BINARY_BOOL_OPS, *COMPARE_OPS}:
            arity = 2

        # 初始化 symbols 为包含 x 的列表
        # 如果 arity 为 2，则 symbols 包含 x 和 y
        symbols = [x]
        if arity == 2:
            symbols = [x, y]

        # 使用 itertools 的 product 函数，生成 vals 中元素的所有组合，长度为 arity
        for args in itertools.product(vals, repeat=arity):
            # 如果 arity 为 1 并且 fn 的一元操作不可用，则跳过当前循环
            if arity == 1 and not valid_unary(fn, *args):
                continue
            # 如果 arity 为 2 并且 fn 的二元操作不可用，则跳过当前循环
            elif arity == 2 and not valid_binary(fn, *args):
                continue

            # 使用 self.subTest 方法，创建一个子测试，参数为 args
            with self.subTest(args=args):
                # 将 args 中的每个元素 sympify 化，得到 sargs 列表
                sargs = [sympy.sympify(a) for a in args]
                # 调用 ReferenceAnalysis 类中的 fn 方法，传入 symbols 作为参数，得到 sympy_expr
                sympy_expr = getattr(ReferenceAnalysis, fn)(*symbols)
                # 再次调用 ReferenceAnalysis 类中的 fn 方法，传入 sargs 作为参数，得到 ref_r
                ref_r = getattr(ReferenceAnalysis, fn)(*sargs)
                # 调用 sympy_interp 函数，传入 ReferenceAnalysis 类、symbols 和 sympy_expr 作为参数，得到 r
                r = sympy_interp(ReferenceAnalysis, dict(zip(symbols, sargs)), sympy_expr)
                # 使用 self.assertEqual 断言 ref_r 等于 r
                self.assertEqual(ref_r, r)

    # 使用参数化装饰器 parametrize，参数 fn 同样取自 UNARY_OPS、BINARY_OPS、UNARY_BOOL_OPS、BINARY_BOOL_OPS、COMPARE_OPS 中的值
    @parametrize("fn", UNARY_OPS + BINARY_OPS + UNARY_BOOL_OPS + BINARY_BOOL_OPS + COMPARE_OPS)
    # 定义测试方法，用于测试特定的 Python 函数（fn）
    def test_python_interp_fx(self, fn):
        # 如果函数是 "log" 或 "exp"，则不进行测试
        if fn in ("log", "exp"):
            return

        # 如果函数是 "truncdiv" 或 "mod"，由于 sympy 不支持符号形状的截断，因此不进行测试
        if fn in ("truncdiv", "mod"):
            return

        # 初始化参数值为 CONSTANTS，如果函数是一元或二元布尔运算，则将参数值设置为 [True, False]
        vals = CONSTANTS
        if fn in {*UNARY_BOOL_OPS, *BINARY_BOOL_OPS}:
            vals = [True, False]

        # 初始化函数的参数个数为 1
        arity = 1
        # 如果函数是二元操作、二元布尔操作或比较操作之一，则将参数个数设置为 2
        if fn in {*BINARY_OPS, *BINARY_BOOL_OPS, *COMPARE_OPS}:
            arity = 2

        # 如果函数是 "pow_by_natural"，则设置参数为整数
        is_integer = None
        if fn == "pow_by_natural":
            is_integer = True

        # 创建符号变量 x 和 y，如果 is_integer 为 True，则将其设置为整数类型
        x = sympy.Dummy('x', integer=is_integer)
        y = sympy.Dummy('y', integer=is_integer)

        # 初始化符号列表为 [x]，如果参数个数为 2，则符号列表为 [x, y]
        symbols = [x]
        if arity == 2:
            symbols = [x, y]

        # 遍历参数值的笛卡尔积，以进行函数行为的测试
        for args in itertools.product(vals, repeat=arity):
            # 如果参数个数为 1 并且不满足一元函数的有效性要求，则跳过
            if arity == 1 and not valid_unary(fn, *args):
                continue
            # 如果参数个数为 2 并且不满足二元函数的有效性要求，则跳过
            elif arity == 2 and not valid_binary(fn, *args):
                continue
            # 如果函数是 "truncdiv" 且第二个参数为 0，则跳过
            if fn == "truncdiv" and args[1] == 0:
                continue
            # 如果函数是 "pow" 或 "pow_by_natural" 且第一个参数为 0 且第二个参数小于等于 0，则跳过
            elif fn in ("pow", "pow_by_natural") and (args[0] == 0 and args[1] <= 0):
                continue
            # 如果函数是 "floordiv" 且第二个参数为 0，则跳过
            elif fn == "floordiv" and args[1] == 0:
                continue
            # 使用子测试来执行具体的函数行为测试，参数为 args
            with self.subTest(args=args):
                # 对于特定的函数，创建 sympy 表达式
                if fn == "minimum":
                    sympy_expr = sympy.Min(x, y)
                elif fn == "maximum":
                    sympy_expr = sympy.Max(x, y)
                else:
                    sympy_expr = getattr(ReferenceAnalysis, fn)(*symbols)

                # 根据参数个数选择适当的 trace_f 函数
                if arity == 1:
                    def trace_f(px):
                        return sympy_interp(PythonReferenceAnalysis, {x: px}, sympy_expr)
                else:
                    def trace_f(px, py):
                        return sympy_interp(PythonReferenceAnalysis, {x: px, y: py}, sympy_expr)

                # 调用 fx 的 symbolic_trace 方法来获得符号化跟踪结果 gm
                gm = fx.symbolic_trace(trace_f)

                # 断言符号化解析结果与 gm(*args) 的结果相等
                self.assertEqual(
                    sympy_interp(PythonReferenceAnalysis, dict(zip(symbols, args)), sympy_expr),
                    gm(*args)
                )
# 返回给定类型的名称字符串
def type_name_fn(type: Type) -> str:
    return type.__name__

# 参数化关系类型的装饰器函数，接受一组类型作为参数
def parametrize_relational_types(*types):
    # 内部包装函数，接受一个可调用对象 f
    def wrapper(f: Callable):
        # 调用 parametrize 函数，使用给定的关系类型或默认的 RELATIONAL_TYPES 参数来参数化函数 f，
        # 使用 type_name_fn 函数来获取类型的名称作为参数名
        return parametrize("op", types or RELATIONAL_TYPES, name_fn=type_name_fn)(f)
    return wrapper

# 测试用例类 TestSympySolve，继承自 TestCase
class TestSympySolve(TestCase):

    # 创建并返回包含整数符号的列表
    def _create_integer_symbols(self) -> List[sympy.Symbol]:
        return sympy.symbols("a b c", integer=True)

    # 测试方法 test_give_up
    def test_give_up(self):
        from sympy import Eq, Ne

        # 使用 _create_integer_symbols 方法创建整数符号 a, b, c
        a, b, c = self._create_integer_symbols()

        # 测试用例列表 cases
        cases = [
            # 不是关系运算。
            a + b,
            # 'a' 出现在两侧。
            Eq(a, a + 1),
            # 'a' 在两侧均不出现。
            Eq(b, c + 1),
            # 结果是 'sympy.And'。
            Eq(FloorDiv(a, b), c),
            # 结果是 'sympy.Or'。
            Ne(FloorDiv(a, b), c),
        ]

        # 遍历测试用例 cases
        for case in cases:
            # 调用 try_solve 函数，尝试解决给定的 case，使用符号 a 作为变量
            e = try_solve(case, a)
            # 断言 e 的值为 None
            self.assertEqual(e, None)

    # 参数化关系类型的测试方法 test_noop
    @parametrize_relational_types()
    def test_noop(self, op):
        # 使用 _create_integer_symbols 方法创建整数符号 a, b, _
        a, b, _ = self._create_integer_symbols()

        # 左右操作数
        lhs, rhs = a, 42 * b
        # 构造关系表达式
        expr = op(lhs, rhs)

        # 调用 try_solve 函数，尝试解决表达式 expr，使用符号 a 作为变量
        r = try_solve(expr, a)
        # 断言 r 的值不为 None
        self.assertNotEqual(r, None)

        # 解析 r 得到的表达式和右侧值
        r_expr, r_rhs = r
        # 断言解析得到的表达式与原始表达式 expr 相等
        self.assertEqual(r_expr, expr)
        # 断言解析得到的右侧值与原始右侧值 rhs 相等
        self.assertEqual(r_rhs, rhs)

    # 参数化关系类型的测试方法 test_noop_rhs
    @parametrize_relational_types()
    def test_noop_rhs(self, op):
        # 使用 _create_integer_symbols 方法创建整数符号 a, b, _
        a, b, _ = self._create_integer_symbols()

        # 左右操作数
        lhs, rhs = 42 * b, a

        # 获取操作符 op 的镜像函数
        mirror = mirror_rel_op(op)
        # 断言镜像函数不为 None
        self.assertNotEqual(mirror, None)

        # 构造关系表达式
        expr = op(lhs, rhs)

        # 调用 try_solve 函数，尝试解决表达式 expr，使用符号 a 作为变量
        r = try_solve(expr, a)
        # 断言 r 的值不为 None
        self.assertNotEqual(r, None)

        # 解析 r 得到的表达式和右侧值
        r_expr, r_rhs = r
        # 断言解析得到的表达式与镜像表达式 mirror(rhs, lhs) 相等
        self.assertEqual(r_expr, mirror(rhs, lhs))
        # 断言解析得到的右侧值与左侧值 lhs 相等
        self.assertEqual(r_rhs, lhs)

    # 私有方法，用于测试给定的案例列表 cases
    def _test_cases(self, cases: List[Tuple[sympy.Basic, sympy.Basic]], thing: sympy.Basic, op: Type[sympy.Rel], **kwargs):
        # 遍历测试用例 cases
        for source, expected in cases:
            # 调用 try_solve 函数，尝试解决给定的 source 表达式，使用 thing 作为变量，使用 op 作为操作符
            r = try_solve(source, thing, **kwargs)

            # 断言 r 与 expected 的值符合预期
            self.assertTrue(
                (r is None and expected is None)
                or (r is not None and expected is not None)
            )

            # 如果 r 不为 None，则解析 r 得到的表达式和右侧值
            if r is not None:
                r_expr, r_rhs = r
                # 断言解析得到的右侧值与 expected 相等
                self.assertEqual(r_rhs, expected)
                # 断言解析得到的表达式与 op(thing, expected) 相等
                self.assertEqual(r_expr, op(thing, expected))

    # 测试加法的测试方法 test_addition
    def test_addition(self):
        from sympy import Eq

        # 使用 _create_integer_symbols 方法创建整数符号 a, b, c
        a, b, c = self._create_integer_symbols()

        # 测试用例列表 cases
        cases = [
            (Eq(a + b, 0), -b),
            (Eq(a + 5, b - 5), b - 10),
            (Eq(a + c * b, 1), 1 - c * b),
        ]

        # 调用 _test_cases 方法，测试给定的测试用例 cases，使用 a 和 Eq 作为参数
        self._test_cases(cases, a, Eq)

    # 参数化关系类型的测试方法 test_multiplication_division
    @parametrize_relational_types(sympy.Eq, sympy.Ne)
    def test_multiplication_division(self, op):
        # 使用 _create_integer_symbols 方法创建整数符号 a, b, c
        a, b, c = self._create_integer_symbols()

        # 测试用例列表 cases
        cases = [
            (op(a * b, 1), 1 / b),
            (op(a * 5, b - 5), (b - 5) / 5),
            (op(a * b, c), c / b),
        ]

        # 调用 _test_cases 方法，测试给定的测试用例 cases，使用 a 和 op 作为参数
        self._test_cases(cases, a, op)

    # 参数化关系类型的测试方法，使用 INEQUALITY_TYPES 参数
    @parametrize_relational_types(*INEQUALITY_TYPES)
    # 定义测试方法，用于测试乘法、除法和不等式
    def test_multiplication_division_inequality(self, op):
        # 创建整数符号a、b，并忽略第三个返回值
        a, b, _ = self._create_integer_symbols()
        # 创建一个负整数符号intneg和一个正整数符号intpos
        intneg = sympy.Symbol("neg", integer=True, negative=True)
        intpos = sympy.Symbol("pos", integer=True, positive=True)

        # 不同的测试用例
        cases = [
            # 乘以正数，然后除以正数
            (op(a * intpos, 1), 1 / intpos),
            # 除以正数，然后乘以正数
            (op(a / (5 * intpos), 1), 5 * intpos),
            # 乘以5，然后与b-5比较，得到b-5除以5
            (op(a * 5, b - 5), (b - 5) / 5),
            # 'b' 不是严格正数也不是严格负数，所以不能对两边都除以'b'
            (op(a * b, 1), None),
            (op(a / b, 1), None),
            (op(a * b * intpos, 1), None),
        ]

        mirror_cases = [
            # 乘以负数，然后除以1
            (op(a * intneg, 1), 1 / intneg),
            # 除以负数，然后乘以1
            (op(a / (5 * intneg), 1), 5 * intneg),
            # 乘以-5，然后与b-5比较，得到-(b-5)除以5
            (op(a * -5, b - 5), -(b - 5) / 5),
        ]
        
        # 获取镜像操作
        mirror_op = mirror_rel_op(op)
        assert mirror_op is not None
        
        # 测试用例
        self._test_cases(cases, a, op)
        self._test_cases(mirror_cases, a, mirror_op)

    # 使用参数化的关系类型进行测试
    @parametrize_relational_types()
    def test_floordiv(self, op):
        from sympy import Eq, Ne, Gt, Ge, Lt, Le
        
        # 创建符号'a', 'b', 'c'，其中'a'是任意符号，'pos'是正数符号
        a, b, c = sympy.symbols("a b c")
        pos = sympy.Symbol("pos", positive=True)
        integer = sympy.Symbol("integer", integer=True)

        # 特殊情况测试字典
        special_case = {
            # 'FloorDiv' 转换为 'And'，无法进一步简化
            Eq: (Eq(FloorDiv(a, pos), integer), None),
            # 'FloorDiv' 转换为 'Or'，无法进一步简化
            Ne: (Ne(FloorDiv(a, pos), integer), None),
            Gt: (Gt(FloorDiv(a, pos), integer), (integer + 1) * pos),
            Ge: (Ge(FloorDiv(a, pos), integer), integer * pos),
            Lt: (Lt(FloorDiv(a, pos), integer), integer * pos),
            Le: (Le(FloorDiv(a, pos), integer), (integer + 1) * pos),
        }[op]

        # 一般情况测试用例列表
        cases: List[Tuple[sympy.Basic, sympy.Basic]] = [
            # 'b' 不是严格正数
            (op(FloorDiv(a, b), integer), None),
            # 'c' 不是严格正数
            (op(FloorDiv(a, pos), c), None),
        ]

        # 在 'FloorDiv' 转换后可能会改变结果
        # 具体来说：
        #   - [Ge, Gt] => Ge
        #   - [Le, Lt] => Lt
        if op in (sympy.Gt, sympy.Ge):
            r_op = sympy.Ge
        elif op in (sympy.Lt, sympy.Le):
            r_op = sympy.Lt
        else:
            r_op = op
        
        # 测试用例
        self._test_cases([special_case, *cases], a, r_op)
        self._test_cases([(special_case[0], None), *cases], a, r_op, floordiv_inequality=False)
    def test_floordiv_eq_simplify(self):
        # 导入 sympy 库中的 Eq 和 Lt 类
        from sympy import Eq, Lt, Le

        # 定义一个正整数符号变量 a
        a = sympy.Symbol("a", positive=True, integer=True)

        # 定义一个内部函数 check，用于验证解是否符合预期
        def check(expr, expected):
            # 尝试解析表达式 expr 关于变量 a 的解
            r = try_solve(expr, a)
            # 确保解不为空
            self.assertNotEqual(r, None)
            r_expr, _ = r
            # 断言解得到的表达式与预期相等
            self.assertEqual(r_expr, expected)

        # (a + 10) // 3 == 3
        # =====================================
        # 3 * 3 <= a + 10         (总是成立)
        #          a + 10 < 4 * 3 (不确定)
        check(Eq(FloorDiv(a + 10, 3), 3), Lt(a, (3 + 1) * 3 - 10))

        # (a + 10) // 2 == 4
        # =====================================
        # 4 * 2 <= 10 - a         (不确定)
        #          10 - a < 5 * 2 (总是成立)
        check(Eq(FloorDiv(10 - a, 2), 4), Le(a, -(4 * 2 - 10)))

    @skipIf(not TEST_Z3, "Z3 not installed")
    def test_z3_proof_floordiv_eq_simplify(self):
        # 导入 z3 库
        import z3
        # 导入 sympy 库中的 Eq 和 Lt 类
        from sympy import Eq, Lt

        # 定义一个正整数符号变量 a
        a = sympy.Symbol("a", positive=True, integer=True)
        # 创建一个整数变量 a_，用于 z3 解析
        a_ = z3.Int("a")

        # (a + 10) // 3 == 3
        # =====================================
        # 3 * 3 <= a + 10         (总是成立)
        #          a + 10 < 4 * 3 (不确定)
        solver = z3.SolverFor("QF_NRA")

        # 添加关于 'a_' 的断言
        solver.add(a_ > 0)

        # 构造表达式 Eq(FloorDiv(a + 10, 3), 3) 并解析得到的表达式
        expr = Eq(FloorDiv(a + 10, 3), 3)
        r_expr, _ = try_solve(expr, a)

        # 检查 'try_solve' 是否确实返回了下面的 'expected'
        expected = Lt(a, (3 + 1) * 3 - 10)
        self.assertEqual(r_expr, expected)

        # 检查是否存在整数 'a_' 满足下面的方程
        solver.add(
            # expr
            (z3.ToInt((a_ + 10) / 3.0) == 3)
            !=
            # expected
            (a_ < (3 + 1) * 3 - 10)
        )

        # 断言不存在这样的整数
        # 即变换是合理的
        r = solver.check()
        self.assertEqual(r, z3.unsat)
# 定义一个名为 TestSingletonInt 的测试类，用于测试单例整数相关的功能
class TestSingletonInt(TestCase):

# 调用 instantiate_parametrized_tests 函数，传入 TestValueRanges 类作为参数，生成参数化测试
instantiate_parametrized_tests(TestValueRanges)

# 调用 instantiate_parametrized_tests 函数，传入 TestSympyInterp 类作为参数，生成参数化测试
instantiate_parametrized_tests(TestSympyInterp)

# 调用 instantiate_parametrized_tests 函数，传入 TestSympySolve 类作为参数，生成参数化测试
instantiate_parametrized_tests(TestSympySolve)

# 如果当前脚本作为主程序运行，则执行 run_tests 函数来运行所有的测试
if __name__ == "__main__":
    run_tests()
```
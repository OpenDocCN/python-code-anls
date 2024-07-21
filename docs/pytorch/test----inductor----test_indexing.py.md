# `.\pytorch\test\inductor\test_indexing.py`

```py
# 导入必要的模块和库
import os  # 导入操作系统接口模块
import unittest  # 导入单元测试框架

import sympy  # 导入符号计算库

import torch  # 导入PyTorch深度学习框架

# 导入Triton相关模块
from torch._inductor.codegen.cpp import cexpr
from torch._inductor.codegen.triton import texpr
from torch._inductor.codegen.wrapper import pexpr

# 导入运行时工具函数
from torch._inductor.runtime.runtime_utils import do_bench_gpu

# 导入大小变量分配器
from torch._inductor.sizevars import SizeVarAllocator

# 导入Inductor测试框架的测试用例基类
from torch._inductor.test_case import TestCase as InductorTestCase

# 导入辅助函数和工具函数
from torch._inductor.utils import run_and_get_triton_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

# 导入Inductor测试工具函数和常量
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU

# 导入符号计算相关函数
from torch.utils._sympy.functions import (
    FloorDiv,          # 向下整除
    ModularIndexing,   # 模索引
    RoundDecimal,      # 四舍五入到小数
    RoundToInt,        # 四舍五入到整数
)

# 检查是否设置了性能测试环境变量
DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"


class TestIndexingSimplification(InductorTestCase):
    # 测试类，继承自InductorTestCase，用于索引简化功能的测试
    def test_indexing_join(self):
        # 创建一个 SizeVarAllocator 的实例
        sizevars = SizeVarAllocator()
        # 定义三个整数符号变量 i0, i1, i2
        i0 = sympy.Symbol("i0", integer=True)
        i1 = sympy.Symbol("i1", integer=True)
        i2 = sympy.Symbol("i2", integer=True)

        # 将两个 ModularIndexing 调用合并成一个更大的调用，如果可能的话
        expr1 = ModularIndexing(i0, 1, 32) + 32 * ModularIndexing(i0, 32, 4)
        # 使用 simplify_with_ranges 方法简化表达式，并断言结果是否正确
        self.assertEqual(
            sizevars.simplify_with_ranges(expr1, {}), ModularIndexing(i0, 1, 128)
        )

        # 当乘数存在时也应该工作
        self.assertEqual(
            sizevars.simplify_with_ranges(2 * expr1, {}),
            2 * ModularIndexing(i0, 1, 128),
        )

        # 当除数不为1时也应该工作
        expr2 = ModularIndexing(i0, 3, 32) + 32 * ModularIndexing(i0, 32 * 3, 4)
        # 使用 simplify_with_ranges 方法简化表达式，并断言结果是否正确
        simplified = sizevars.simplify_with_ranges(expr2, {})
        self.assertEqual(simplified, ModularIndexing(i0, 3, 128))
        self.assertEqual(expr2.subs({i0: 39485}), simplified.subs({i0: 39485}))

        # 在模数错误时不应该发生合并
        expr3 = ModularIndexing(i0, 1, 30) + 32 * ModularIndexing(i0, 32, 4)
        # 使用 simplify_with_ranges 方法简化表达式，并断言结果是否正确
        self.assertEqual(sizevars.simplify_with_ranges(expr3, {}), expr3)

        # 检查在模数大于1时也能正确工作
        expr4 = ModularIndexing(i0, 10, i1) + i1 * ModularIndexing(i0, i1 * 10, i2)
        # 替换符号并比较结果
        res0 = expr4.subs({i0: 24056, i1: 13, i2: 19})
        simplified = sizevars.simplify_with_ranges(expr4, {})
        res1 = simplified.subs({i0: 24056, i1: 13, i2: 19})
        self.assertEqual(res0, res1)
        self.assertEqual(simplified, ModularIndexing(i0, 10, i1 * i2))

        # 同样适用于带有偏移量的情况
        self.assertEqual(
            sizevars.simplify_with_ranges(expr4 + 10, {}),
            ModularIndexing(i0, 10, i1 * i2) + 10,
        )

        # 对 ModularIndexing 和 FloorDiv 的组合进行测试
        expr5 = 197 * FloorDiv(i0, 197) + ModularIndexing(i0, 1, 197)
        # 使用 simplify_with_ranges 方法简化表达式，并断言结果是否正确
        simplified = sizevars.simplify_with_ranges(expr5, {})
        self.assertEqual(simplified, i0)
        self.assertEqual(expr5.subs({i0: 39485}), simplified.subs({i0: 39485}))

        # 当乘数存在时也应该工作
        self.assertEqual(
            sizevars.simplify_with_ranges(2 * expr5, {}),
            2 * i0,
        )

        # 当除数不为1时也应该工作
        expr6 = 197 * FloorDiv(i0, 197 * 3) + ModularIndexing(i0, 3, 197)
        # 使用 simplify_with_ranges 方法简化表达式，并断言结果是否正确
        simplified = sizevars.simplify_with_ranges(expr6, {})
        self.assertEqual(simplified, FloorDiv(i0, 3))
        self.assertEqual(expr6.subs({i0: 39485}), simplified.subs({i0: 39485}))
    # 测试函数：测试合并模块化索引对是否成功合并
    def test_modular_indexing_pairs_merged(self):
        # 创建一个大小变量分配器对象
        sizevars = SizeVarAllocator()
        # 创建一个整数符号变量 x
        x = sympy.Symbol("x", integer=True, positive=True)
        # 设定常量 a 和 b 的值
        a = 1024
        b = 32
        # 创建第一个模块化索引表达式对象 expr1
        expr1 = ModularIndexing(x, 1, a)
        # 创建第二个模块化索引表达式对象 expr2，基于 expr1 和常量 b
        expr2 = ModularIndexing(expr1, 1, b)
        # 创建预期结果对象 expected，预期结果为将 x, 1, b 作为参数的模块化索引对象
        expected = ModularIndexing(x, 1, b)

        # 调用 sizevars 对象的合并模块化索引对方法，获得实际输出结果 actual
        actual = sizevars.combine_modular_indexing_pairs(expr2)
        # 断言预期结果与实际结果相等
        self.assertEqual(expected, actual)
        # 断言 expr2 与 actual 不相等
        self.assertNotEqual(expr2, actual)

    # 测试函数：测试未成功合并的模块化索引对
    def test_modular_indexing_pairs_not_merged(self):
        # 创建一个大小变量分配器对象
        sizevars = SizeVarAllocator()
        # 创建一个整数符号变量 x
        x = sympy.Symbol("x", integer=True, positive=True)
        # 设定常量 a 和 b 的值
        a = 1024
        b = 3  # 选择一个无法合并的常量 b
        # 创建第一个模块化索引表达式对象 expr1
        expr1 = ModularIndexing(x, 1, a)
        # 创建第二个模块化索引表达式对象 expr2，基于 expr1 和常量 b
        expr2 = ModularIndexing(expr1, 1, b)

        # 调用 sizevars 对象的合并模块化索引对方法，获得实际输出结果 actual
        actual = sizevars.combine_modular_indexing_pairs(expr2)
        # 断言预期结果与实际结果相等
        self.assertEqual(expr2, actual)
        # 断言 ModularIndexing(x, 1, b) 与 actual 不相等
        self.assertNotEqual(ModularIndexing(x, 1, b), actual)

    # 测试函数：测试展开 FloorDiv 是否被跳过
    def test_expand_floor_div_skipped(self):
        # 创建一个大小变量分配器对象
        sizevars = SizeVarAllocator()
        # 创建整数符号变量 x 和 y
        x = sympy.Symbol("x", integer=True, positive=True)
        y = sympy.Symbol("y", integer=True, positive=True)

        # 创建一个复杂的 FloorDiv 表达式 expr
        expr = FloorDiv(x, 2) + FloorDiv(y, 3)
        # 由于表达式中包含多个 FloorDiv，无法简化，返回 False
        # 在这种情况下，我们返回 False
        self.assertFalse(sizevars.expand_floor_div(expr))

    # 测试函数：测试展开 FloorDiv 是否被应用
    def test_expand_floor_div_applied(self):
        # 创建一个大小变量分配器对象
        sizevars = SizeVarAllocator()
        # 创建整数符号变量 x 和 y
        x = sympy.Symbol("x", integer=True, positive=True)
        y = sympy.Symbol("y", integer=True, positive=True)

        # 创建一个复杂的 FloorDiv 表达式 expr
        expr = x * 5 + FloorDiv(y, 3)
        # 调用 sizevars 对象的展开 FloorDiv 方法，获得实际输出结果 actual 和分母 denominator
        actual, denominator = sizevars.expand_floor_div(expr)
        # 断言 expr 与 actual 不相等
        self.assertNotEqual(expr, actual)
        # 创建预期结果对象 expected，预期结果为将 x * 15 + y 的结果整体 FloorDiv
        expected = FloorDiv(x * 15 + y, 3)
        # 断言预期结果与 FloorDiv(actual, denominator) 相等
        self.assertEqual(expected, FloorDiv(actual, denominator))

    # 测试函数：测试 Int8 解包
    @unittest.skipUnless(HAS_GPU, "Need GPU for this test")
    def test_int8_unpack(self):
        # 定义一个使用 Torch 编译的函数 f
        @torch.compile
        def f(x):
            # 分别计算 x 的高位和低位元素
            first_elements = x >> 4
            second_elements = x & 15
            # 将计算得到的元素堆叠在一起，按最后一个维度重新排列
            unpacked = torch.stack([first_elements, second_elements], dim=-1).view(
                *x.size()[:-1], -1
            )
            return unpacked * 2

        # 创建一个在指定 GPU 类型上生成的随机整数 Tensor 对象 x
        x = torch.randint(0, 255, (2, 4096, 5504), dtype=torch.uint8, device=GPU_TYPE)

        # 运行并获取 Triton 代码的结果 triton_code
        triton_code = run_and_get_triton_code(f, x)
        # 确保代码中的 2 次加载使用简化的索引，而不是类似于 tl.load(in_ptr0 + ((5504*x1) + (x0 // 2)) 的形式
        self.assertEqual(2, triton_code.count("tl.load(in_ptr0 + ((x2 // 2)),"))
        if DO_PERF_TEST:
            # 如果进行性能测试，则打印执行时间
            ms = do_bench_gpu(lambda: f(x))
            print(f"{ms=:.03f}")
# 定义一个测试类 ExprPrinterTests，继承自 InductorTestCase，用于测试表达式打印功能
class ExprPrinterTests(InductorTestCase):

    # 定义测试方法 test_print_pow，测试幂运算打印功能
    def test_print_pow(self):
        # 创建三个整数类型的符号变量
        s1 = sympy.Symbol("foo", integer=True)
        s2 = sympy.Symbol("bar", integer=True)
        s3 = sympy.Symbol("baz", integer=True)

        # 定义常见的测试用例 common_cases 列表，包含元组 (表达式, 期望结果)
        common_cases = [
            # expr, result
            # 直接测试 Pow 函数
            (
                sympy.Pow(s1 + s2, 0),
                lambda _, L: f"1{L}",  # 注意：在 _print_Pow 处理之前已经简化
            ),  # 注意：在 _print_Pow 处理之前已经简化
        ]

        # GPU 特定的测试用例 gpu_cases 是 common_cases 加上特定的表达式
        gpu_cases = common_cases + [
            (sympy.Pow(s1 + s2, 2), lambda c, L: "(bar + foo)*(bar + foo)")
        ]

        # CPU 特定的测试用例 cpu_cases 是 common_cases 加上特定的表达式
        cpu_cases = common_cases + [
            (
                sympy.Pow(s1 + s2, 2),
                lambda c, L: "static_cast<long>((bar + foo)*(bar + foo))",
            )
        ]

        # 对 GPU 特定的测试用例进行迭代
        for expr, result in gpu_cases:
            # 断言 texpr 函数对表达式的输出与期望结果一致
            self.assertEqual(texpr(expr), result(1, ""))
            # 断言 pexpr 函数对表达式的输出与期望结果一致
            self.assertEqual(pexpr(expr), result(1, ""))

        # 对 CPU 特定的测试用例进行迭代
        for expr, result in cpu_cases:
            # 断言 cexpr 函数对表达式的输出与期望结果一致，并说明 1.0 用于浮点数除法
            self.assertEqual(cexpr(expr), result(1.0, "L"))  # 1.0 for FP div

    # 定义测试方法 test_print_floor，测试向下取整打印功能
    def test_print_floor(self):
        # 针对整数和非整数类型进行迭代测试
        for integer in [True, False]:
            # 创建一个符号变量 s1，指定是否为整数类型
            s1 = sympy.Symbol("s1", integer=integer)
            # 创建向下取整表达式
            expr = sympy.floor(s1 / 2)
            # 如果是整数类型
            if integer:
                # 断言 pexpr 函数对表达式的输出与期望结果一致
                self.assertEqual(pexpr(expr), "math.floor((1/2)*s1)")
                # 断言 cexpr 函数对表达式的输出与期望结果一致，使用 static_cast<long> 进行类型转换
                self.assertEqual(
                    cexpr(expr), "static_cast<long>(std::floor((1.0/2.0)*s1))"
                )
            else:
                # 断言 pexpr 函数对表达式的输出与期望结果一致
                self.assertExpectedInline(pexpr(expr), """math.floor((1/2)*s1)""")
                # 断言 texpr 函数对表达式的输出与期望结果一致，使用 libdevice 库进行处理，并转换为 tl.int64 类型
                self.assertExpectedInline(
                    texpr(expr),
                    """libdevice.floor((1/2)*s1).to(tl.int64)""",
                )
                # 断言 cexpr 函数对表达式的输出与期望结果一致
                self.assertExpectedInline(cexpr(expr), """std::floor((1.0/2.0)*s1)""")

    # 定义测试方法 test_print_ceil，测试向上取整打印功能
    def test_print_ceil(self):
        # 针对整数和非整数类型进行迭代测试
        for integer in [True, False]:
            # 创建一个符号变量 s1，指定是否为整数类型
            s1 = sympy.Symbol("s1", integer=integer)
            # 创建向上取整表达式
            expr = sympy.ceiling(s1 / 2)
            # 如果是整数类型
            if integer:
                # 断言 pexpr 函数对表达式的输出与期望结果一致
                self.assertExpectedInline(pexpr(expr), """math.ceil((1/2)*s1)""")
                # 断言 cexpr 函数对表达式的输出与期望结果一致，使用 static_cast<long> 进行类型转换
                self.assertExpectedInline(
                    cexpr(expr), """static_cast<long>(std::ceil((1.0/2.0)*s1))"""
                )
            else:
                # 断言 pexpr 函数对表达式的输出与期望结果一致
                self.assertExpectedInline(pexpr(expr), """math.ceil((1/2)*s1)""")
                # 断言 cexpr 函数对表达式的输出与期望结果一致
                self.assertExpectedInline(cexpr(expr), """std::ceil((1.0/2.0)*s1)""")

    # 定义测试方法 test_print_round，测试四舍五入到整数打印功能
    def test_print_round(self):
        # 创建 RoundToInt 类型的表达式，将符号变量 x 除以 2，然后进行四舍五入
        expr = RoundToInt(sympy.Symbol("x", integer=True) / 2)
        # 断言 pexpr 函数对表达式的输出与期望结果一致
        self.assertExpectedInline(pexpr(expr), """round((1/2)*x)""")
        # 断言 cexpr 函数对表达式的输出与期望结果一致，使用 std::lrint 进行四舍五入
        self.assertExpectedInline(cexpr(expr), """std::lrint((1.0/2.0)*x)""")
        # 断言 texpr 函数对表达式的输出与期望结果一致，使用 libdevice 库进行处理
        self.assertExpectedInline(texpr(expr), """libdevice.llrint((1/2)*x)""")

    # 使用 parametrize 装饰器来定义测试参数化方法，参数为 ndigits 的取值范围 [-1, 0, 1]
    @parametrize("ndigits", [-1, 0, 1])
    # 定义一个测试方法，用于测试 RoundDecimal 类的打印输出
    def test_print_round_decimal(self, ndigits):
        # 创建 RoundDecimal 表达式，使用符号变量 x 的一半，并指定小数点位数 ndigits
        expr = RoundDecimal(sympy.Symbol("x", integer=True) / 2, ndigits)
        # 断言 pexpr 函数生成的字符串与预期相等，表达式为 round((1/2)*x, {ndigits})
        self.assertEqual(pexpr(expr), f"round((1/2)*x, {ndigits})")
        # 断言 cexpr 函数生成的字符串与预期相等，表达式为 static_cast<double>(std::nearbyint(1e{ndigits} * ((1.0/2.0)*x)) * 1e{-ndigits})
        self.assertEqual(
            cexpr(expr),
            f"static_cast<double>(std::nearbyint(1e{ndigits} * ((1.0/2.0)*x)) * 1e{-ndigits})",
        )
        # 断言 texpr 函数生成的字符串与预期相等，表达式为 libdevice.nearbyint(1e{ndigits} * ((1/2)*x)) * 1e{-ndigits}
        self.assertEqual(
            texpr(expr),
            f"libdevice.nearbyint(1e{ndigits} * ((1/2)*x)) * 1e{-ndigits}",
        )

    # 定义一个测试方法，用于测试 FloorDiv 类的打印输出
    def test_print_floor_div(self):
        # 创建两个整数符号变量 s1 和 s2
        s1 = sympy.Symbol("s1", integer=True)
        s2 = sympy.Symbol("s2", integer=True)
        # 创建 FloorDiv 表达式，表示 s1 除以 s2 的整数除法
        expr = FloorDiv(s1, s2)
        # 断言 pexpr 函数生成的字符串与预期相等，表达式为 (s1 // s2)
        self.assertEqual(pexpr(expr), "(s1 // s2)")
        # 断言 cexpr 函数生成的字符串与预期相等，表达式为 c10::div_floor_integer(s1, s2)
        self.assertEqual(cexpr(expr), "c10::div_floor_integer(s1, s2)")

        # 重新定义 s1 和 s2，s2 为负整数符号
        s1 = sympy.Symbol("s1", integer=True)
        s2 = sympy.S(-1)
        # 创建 FloorDiv 表达式，表示 s1 除以 -1 的整数除法
        expr = FloorDiv(s1, s2)
        # 断言 pexpr 函数生成的字符串与预期相等，表达式为 (-1)*s1
        self.assertEqual(pexpr(expr), "(-1)*s1")
        # 断言 cexpr 函数生成的字符串与预期相等，表达式为 (-1L)*s1
        self.assertEqual(cexpr(expr), "(-1L)*s1")

    # 定义一个测试方法，用于测试 sympy.Min 和 sympy.Max 函数的打印输出
    def test_print_Min_Max(self):
        # 定义测试用例，包括 sympy.Min 和 sympy.Max 函数以及相应的比较运算符
        cases = (
            (sympy.Min, "min", "<"),
            (sympy.Max, "max", ">"),
        )
        # 遍历测试用例
        for f, s, cmp in cases:
            # 创建整数符号变量 x
            x = sympy.Symbol("x", integer=True)
            # 创建 Min 或 Max 表达式，参数为 -2 和 x
            expr = f(-2, x)
            # 断言 texpr 函数生成的字符串与预期相等，表达式为 ((-2) * ((-2) {cmp}= (x)) + (x) * ((x) {cmp} (-2)))
            self.assertEqual(
                texpr(expr), f"((-2) * ((-2) {cmp}= (x)) + (x) * ((x) {cmp} (-2)))"
            )
            # 断言 cexpr 函数生成的字符串与预期相等，表达式为 std::{s}(-2L, x)
            self.assertEqual(cexpr(expr), f"std::{s}(-2L, x)")

            # 创建 Min 或 Max 表达式，参数为 x, 2*x, 3*x
            expr = f(x, 2 * x, 3 * x)
            # 断言 texpr 函数生成的字符串与预期相等，表达式为 (详细内容见实际代码，此处省略较长的注释)
            self.assertEqual(
                texpr(expr),
                f"((x) * ((x) {cmp}= (((2*x) * ((2*x) {cmp}= (3*x)) + (3*x) * ((3*x) {cmp} (2*x))))) + (((2*x) * ((2*x) {cmp}= (3*x)) + (3*x) * ((3*x) {cmp} (2*x)))) * ((((2*x) * ((2*x) {cmp}= (3*x)) + (3*x) * ((3*x) {cmp} (2*x)))) {cmp} (x)))",  # noqa: B950 line too long
            )
            # 断言 cexpr 函数生成的字符串与预期相等，表达式为 std::{s}({{x, 2L*x, 3L*x}})
            self.assertEqual(cexpr(expr), f"std::{s}({{x, 2L*x, 3L*x}})")
# 实例化一个带参数的测试，参数是 ExprPrinterTests，用于表达式打印测试
instantiate_parametrized_tests(ExprPrinterTests)

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 从 torch._inductor.test_case 模块导入 run_tests 函数
    from torch._inductor.test_case import run_tests
    
    # 如果 HAS_CPU 或 HAS_GPU 为真，则运行名为 "sympy" 的测试
    if HAS_CPU or HAS_GPU:
        run_tests("sympy")
```
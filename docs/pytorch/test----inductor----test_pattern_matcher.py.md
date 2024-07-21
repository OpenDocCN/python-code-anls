# `.\pytorch\test\inductor\test_pattern_matcher.py`

```py
# Owner(s): ["module: inductor"]

# 导入所需的模块和库
import copy  # 导入copy模块，用于复制对象
import os  # 导入os模块，提供与操作系统交互的功能
import unittest  # 导入unittest模块，用于编写和运行单元测试

import torch  # 导入PyTorch库
import torch._dynamo.config as dynamo_config  # 导入torch._dynamo.config模块
import torch._inductor.config as inductor_config  # 导入torch._inductor.config模块
import torch._inductor.fx_passes.post_grad  # 导入torch._inductor.fx_passes.post_grad模块
import torch.nn.functional as F  # 导入torch.nn.functional模块，提供神经网络函数

from torch._dynamo.utils import count_calls, counters  # 从torch._dynamo.utils导入count_calls和counters函数
from torch._higher_order_ops.out_dtype import out_dtype  # 导入torch._higher_order_ops.out_dtype模块中的out_dtype
from torch._inductor.fx_passes import joint_graph  # 导入torch._inductor.fx_passes模块中的joint_graph函数

# 导入PatternMatcher所需的各类对象和函数
from torch._inductor.pattern_matcher import (
    Arg,  # 导入Arg类
    CallFunction,  # 导入CallFunction类
    gen_pattern,  # 导入gen_pattern函数
    KeywordArg,  # 导入KeywordArg类
    Match,  # 导入Match类
    PatternMatcherPass,  # 导入PatternMatcherPass类
    PatternPrettyPrinter,  # 导入PatternPrettyPrinter类
    register_graph_pattern,  # 导入register_graph_pattern函数
    stable_topological_sort,  # 导入stable_topological_sort函数
)

from torch._inductor.test_case import run_tests, TestCase  # 导入run_tests和TestCase类
from torch._inductor.utils import run_and_get_code  # 导入run_and_get_code函数
from torch._inductor.virtualized import V  # 导入V类
from torch.testing import FileCheck  # 导入FileCheck类
from torch.testing._internal.common_cuda import SM80OrLater  # 导入SM80OrLater类
from torch.testing._internal.common_utils import IS_LINUX, LazyVal, skipIfRocm  # 导入IS_LINUX、LazyVal和skipIfRocm变量
from torch.testing._internal.inductor_utils import HAS_CUDA  # 导入HAS_CUDA变量
from torch.utils import _pytree as pytree  # 导入_pytree模块

# 定义LazyVal对象is_a100_linux，用于判断是否为Linux系统且有CUDA设备，并且设备为NVIDIA A100
is_a100_linux = LazyVal(
    lambda: IS_LINUX
    and torch.cuda.is_available()
    and "A100" in torch.cuda.get_device_name(0)
)


class TestPatternMatcher(TestCase):
    # 定义TestPatternMatcher类，继承自unittest.TestCase

    # 定义common方法，用于通用的模式匹配测试
    def common(
        self,
        fn,  # 待测试的函数
        args,  # 函数的参数
        expected_matches,  # 预期匹配数量
        expected_nodes,  # 预期节点数量
        additional_check=lambda code: None,  # 额外检查函数，默认为空函数
        reference_in_float=False,  # 是否将输入参数转换为float32参考值，默认为False
    ):
        # 清空计数器
        counters.clear()
        # 设置随机数种子为42
        torch.manual_seed(42)

        # 根据reference_in_float参数，创建参考输入ref_inputs
        if reference_in_float:
            ref_inputs = pytree.tree_map_only(
                torch.Tensor, lambda x: x.to(torch.float32), args
            )
        else:
            ref_inputs = args

        # 计算预期输出
        expected = fn(*ref_inputs)

        # 再次设置随机数种子为42
        torch.manual_seed(42)

        # 运行函数fn，并获取运行时生成的代码及其输出
        actual, codes = run_and_get_code(torch.compile(fn), *args)

        # 如果codes是长度为1的列表，则将其转换为单个元素
        if len(codes) == 1:
            codes = codes[0]

        # 使用torch.testing.assert_close比较实际输出和预期输出，检查数据类型（如果不是reference_in_float）
        torch.testing.assert_close(actual, expected, check_dtype=not reference_in_float)

        # 断言模式匹配计数器的值是否等于预期匹配数量
        self.assertEqual(
            counters["inductor"]["pattern_matcher_count"], expected_matches
        )

        # 断言模式匹配节点计数器的值是否等于预期节点数量
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], expected_nodes)

        # 执行额外的检查函数，传入运行时生成的代码codes
        additional_check(codes)

        # 清空计数器
        counters.clear()
    def test_mm_plus_mm(self):
        def fn(a, b, c, d):
            return torch.add(torch.mm(a, b), torch.mm(c, d))
        # 定义一个函数 fn，接受四个参数 a, b, c, d，分别进行矩阵乘法和加法操作后返回结果

        # 当 m1 == n1 and m2 == n2 时，mm_plus_mm 可以匹配到融合操作
        fusible_args_list = [
            (
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
            ),
            (
                torch.randn(1, 4, device="cuda"),
                torch.randn(4, 2, device="cuda"),
                torch.randn(1, 5, device="cuda"),
                torch.randn(5, 2, device="cuda"),
            ),
        ]
        # 遍历可融合参数列表，并调用 self.common 方法执行 fn 函数，设置相关参数和操作
        for args in fusible_args_list:
            self.common(fn, args, 1, 3)

        # 如果不能融合，只能匹配到 add(mm()) 操作
        unfusible_args_list = [
            # https://github.com/pytorch/pytorch/issues/100670.
            (
                torch.randn(1, 4, device="cuda"),
                torch.randn(4, 2, device="cuda"),
                torch.randn(1, 2, device="cuda"),
                torch.randn(2, 1, device="cuda"),
            ),
            (
                torch.randn(1, 2, device="cuda"),
                torch.randn(2, 1, device="cuda"),
                torch.randn(1, 4, device="cuda"),
                torch.randn(4, 2, device="cuda"),
            ),
        ]
        # 遍历不可融合参数列表，并调用 self.common 方法执行 fn 函数，设置相关参数和操作
        for args in unfusible_args_list:
            self.common(fn, args, 1, 2)

    def _test_fused_int_mm_mul_impl(self, fn, args, fused_int_mm_mul_expected=True):
        # 重置 torch._dynamo 状态和计数器
        torch._dynamo.reset()
        counters.clear()
        # 计算函数 fn 的参考结果
        ref = fn(*args)
        # 运行并获取编译后的代码
        test, (code,) = run_and_get_code(torch.compile(fn, mode="max-autotune"), *args)
        # 断言是否预期到融合整数乘法和矩阵乘法操作
        self.assertEqual("fused_int_mm_mul" in code, fused_int_mm_mul_expected)
        if fused_int_mm_mul_expected:
            indices = ~ref.isinf()
            # 使用 torch.testing.assert_close 检查非无穷值的索引处的结果是否接近，同时检查 dtype 是否正确

    @skipIfRocm
    @unittest.skipIf(not SM80OrLater, "need sm_80")
    @inductor_config.patch(force_fuse_int_mm_with_mul=True)
    # 定义测试函数 test_fused_int_mm_mul
    def test_fused_int_mm_mul(self):
        # 定义内部函数 fn1，接受三个参数 a, b, c，计算 torch.ops.aten.mm.default 的输出和 c 的乘积
        def fn1(a, b, c):
            return out_dtype(torch.ops.aten.mm.default, torch.int32, a, b) * c
        
        # 定义内部函数 fn2，接受三个参数 a, b, c，计算 torch.ops.aten.mm.default 的输出和 c 的乘积，然后转换为 torch.bfloat16 类型
        def fn2(a, b, c):
            return (out_dtype(torch.ops.aten.mm.default, torch.int32, a, b) * c).to(
                torch.bfloat16
            )
        
        # 参数列表，包含三组参数，每组参数是三个张量，用于测试 fn1 和 fn2 函数
        args_list = [
            (
                torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda"),
                torch.randint(-128, 127, (32, 8), dtype=torch.int8, device="cuda"),
                torch.randn((32, 1), dtype=torch.float16, device="cuda") * 0 + 0.5,
            ),
            (
                torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda"),
                torch.randint(-128, 127, (32, 8), dtype=torch.int8, device="cuda"),
                torch.randn((1, 8), dtype=torch.bfloat16, device="cuda"),
            ),
            (
                torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda"),
                torch.randint(-128, 127, (32, 8), dtype=torch.int8, device="cuda"),
                torch.randn((1, 8), dtype=torch.float32, device="cuda"),
            ),
        ]

        # 遍历参数列表，对每组参数调用 _test_fused_int_mm_mul_impl 函数测试 fn1 和 fn2
        for args in args_list:
            self._test_fused_int_mm_mul_impl(fn1, args, True)
            self._test_fused_int_mm_mul_impl(fn2, args, True)

    # 跳过条件为 ROCm 环境的测试装饰器
    @skipIfRocm
    # 跳过条件：如果不满足 SM80OrLater 条件，需要 sm_80 架构支持
    @unittest.skipIf(not SM80OrLater, "need sm_80")
    # 使用 inductor_config.patch 修饰器，强制开启 fuse_int_mm_with_mul 功能
    @inductor_config.patch(force_fuse_int_mm_with_mul=True)
    # 定义测试函数 test_fused_int_mm_mul_gating
    def test_fused_int_mm_mul_gating(self):
        # 定义内部函数 fn1，接受三个参数 a, b, c，计算 torch.ops.aten.mm.default 的输出和 c 的乘积
        def fn1(a, b, c):
            return out_dtype(torch.ops.aten.mm.default, torch.int32, a, b) * c

        # 第一组参数 args1
        args1 = (
            torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda"),
            torch.randint(-128, 127, (32, 8), dtype=torch.int8, device="cuda"),
            torch.randn((8), dtype=torch.float32, device="cuda"),
        )

        # 第二组参数 args2
        args2 = (
            torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda"),
            torch.randint(-128, 127, (32, 8), dtype=torch.int8, device="cuda"),
            torch.randn((32, 1), dtype=torch.float16, device="cuda"),
        )

        # 使用 _test_fused_int_mm_mul_impl 函数测试 fn1，第一组参数，预期不使用 fusion 功能
        self._test_fused_int_mm_mul_impl(fn1, args1, False)
        # 使用 _test_fused_int_mm_mul_impl 函数测试 fn1，第二组参数转为 CPU 张量，预期不使用 fusion 功能
        self._test_fused_int_mm_mul_impl(fn1, [arg.cpu() for arg in args2], False)
        
        # 设置 inductor_config.force_fuse_int_mm_with_mul 为 False
        inductor_config.force_fuse_int_mm_with_mul = False
        # 使用 _test_fused_int_mm_mul_impl 函数测试 fn1，第二组参数，预期不使用 fusion 功能
        self._test_fused_int_mm_mul_impl(fn1, args2, False)

    # 定义私有方法 _test_mixed_impl，用于测试混合运算功能
    def _test_mixed_impl(
        self,
        fn,
        args,
        mixed_mm_expected,
        fallback_mixed_mm_expected,
        rtol=None,
        atol=None,
    ):
        # 重置 torch._dynamo 和 counters
        torch._dynamo.reset()
        counters.clear()
        # 计算函数 fn 在参数 args 上的输出作为参考结果 ref
        ref = fn(*args)
        # 编译函数 fn 并执行，获取代码 code
        test, (code,) = run_and_get_code(torch.compile(fn), *args)
        # 使用 torch.testing.assert_close 检查参考结果 ref 和测试结果 test 的接近程度
        torch.testing.assert_close(ref, test, rtol=rtol, atol=atol)
        # 检查代码中是否包含 "mixed_mm" 标志，预期为 mixed_mm_expected
        self.assertEqual("mixed_mm" in code, mixed_mm_expected)
        # 检查代码中是否包含 "fallback_mixed_mm" 标志，预期为 fallback_mixed_mm_expected
        self.assertEqual("fallback_mixed_mm" in code, fallback_mixed_mm_expected)

    # 跳过条件：如果不满足 SM80OrLater 条件，需要 sm_80 架构支持
    @unittest.skipIf(not SM80OrLater, "need sm_80")
    @inductor_config.patch(mixed_mm_choice="triton")
    # 使用装饰器设置配置，选择使用 Triton 混合矩阵乘法

    def test_mixed_mm(self):
        # 定义测试函数 test_mixed_mm

        def fn(a, b):
            # 定义函数 fn，执行矩阵乘法操作
            return torch.mm(a, b.to(a.dtype))

        args_list = [
            # 定义多组测试参数列表
            (
                torch.randn(8, 8, device="cuda"),
                torch.randint(-128, 127, (8, 8), dtype=torch.int8, device="cuda"),
            ),
            (
                torch.randn(8, 2, device="cuda", dtype=torch.bfloat16),
                torch.randint(-128, 127, (2, 8), dtype=torch.int8, device="cuda"),
            ),
            (
                torch.randn(8, 5, device="cuda", dtype=torch.float16),
                torch.randint(0, 255, (5, 2), dtype=torch.uint8, device="cuda"),
            ),
            (
                torch.randn(8, 8, device="cuda", dtype=torch.float32),
                torch.randn(8, 8, device="cuda", dtype=torch.bfloat16),
            ),
        ]

        for args in args_list:
            # 遍历参数列表，对每组参数执行 _test_mixed_impl 测试
            self._test_mixed_impl(fn, args, True, False)

    @unittest.skipIf(not SM80OrLater, "need sm_80")
    # 如果 SM80OrLater 为假，则跳过测试，要求 SM80 或更高版本的 GPU
    @inductor_config.patch(mixed_mm_choice="triton")
    # 使用装饰器设置配置，选择使用 Triton 混合矩阵乘法

    def test_mixed_mm_bad_cases(self):
        # 定义测试函数 test_mixed_mm_bad_cases

        def fn(a, b):
            # 定义函数 fn，执行矩阵乘法操作
            return torch.mm(a, b.to(a.dtype))

        # 当 b 被转置且不是连续存储时，跳过 Triton 使用备用方案
        args_list = [
            (
                torch.randn(8, 8, device="cuda", dtype=torch.float16),
                torch.randint(-128, 127, (4, 8), dtype=torch.int8, device="cuda").t()[
                    :, ::2
                ],
            ),
            (
                torch.randn(8, 8, device="cuda", dtype=torch.bfloat16),
                torch.randint(0, 255, (4, 8), dtype=torch.uint8, device="cuda").t()[
                    :, ::2
                ],
            ),
        ]

        for args in args_list:
            # 遍历参数列表，对每组参数执行 _test_mixed_impl 测试
            self._test_mixed_impl(fn, args, True, True)

    @unittest.skipIf(not SM80OrLater, "need sm_80")
    # 如果 SM80OrLater 为假，则跳过测试，要求 SM80 或更高版本的 GPU
    @inductor_config.patch(mixed_mm_choice="triton", max_autotune_gemm=True)
    # 使用装饰器设置配置，选择使用 Triton 混合矩阵乘法，并开启最大自动调优 GEMM
    # 定义一个测试函数 test_mixed_mm_epi_works，用于测试混合数据类型矩阵乘法的功能
    def test_mixed_mm_epi_works(self):
        
        # 定义一个内部函数 fn，接受四个参数 a, b, c, d，并返回 torch.mm(a, b.to(a.dtype)) * c + d 的结果
        def fn(a, b, c, d):
            return torch.mm(a, b.to(a.dtype)) * c + d
        
        # 定义多组参数 args_list，每组参数包含四个张量，用于测试不同的数据类型和设备
        args_list = [
            (
                torch.randn(8, 8, device="cuda"),  # 随机生成一个 8x8 的张量，使用 CUDA 设备
                torch.randint(-128, 127, (8, 8), dtype=torch.int8, device="cuda"),  # 随机生成一个 8x8 的 int8 类型张量，使用 CUDA 设备
                torch.randn(8, device="cuda"),  # 随机生成一个长度为 8 的张量，使用 CUDA 设备
                torch.randn(8, device="cuda"),  # 随机生成一个长度为 8 的张量，使用 CUDA 设备
            ),
            (
                torch.randn(8, 2, device="cuda", dtype=torch.bfloat16),  # 随机生成一个 8x2 的 bfloat16 类型张量，使用 CUDA 设备
                torch.randint(-128, 127, (2, 8), dtype=torch.int8, device="cuda"),  # 随机生成一个 2x8 的 int8 类型张量，使用 CUDA 设备
                torch.randn(8, device="cuda", dtype=torch.bfloat16),  # 随机生成一个长度为 8 的 bfloat16 类型张量，使用 CUDA 设备
                torch.randn(8, device="cuda", dtype=torch.bfloat16),  # 随机生成一个长度为 8 的 bfloat16 类型张量，使用 CUDA 设备
            ),
            (
                torch.randn(8, 5, device="cuda", dtype=torch.float16),  # 随机生成一个 8x5 的 float16 类型张量，使用 CUDA 设备
                torch.randint(0, 255, (5, 2), dtype=torch.uint8, device="cuda"),  # 随机生成一个 5x2 的 uint8 类型张量，使用 CUDA 设备
                torch.randn(2, device="cuda", dtype=torch.float16),  # 随机生成一个长度为 2 的 float16 类型张量，使用 CUDA 设备
                torch.randn(2, device="cuda", dtype=torch.float16),  # 随机生成一个长度为 2 的 float16 类型张量，使用 CUDA 设备
            ),
        ]
        
        # 遍历参数列表 args_list，对每组参数调用 self._test_mixed_impl 方法进行测试
        for args in args_list:
            self._test_mixed_impl(fn, args, True, False)
    
    # 根据条件跳过测试，若不满足 SM80OrLater 条件则跳过，需支持 sm_80 GPU
    @unittest.skipIf(not SM80OrLater, "need sm_80")
    # 根据条件跳过测试，若不满足 is_a100_linux 条件则跳过，仅在 Linux A100 硬件上运行
    @unittest.skipIf(not is_a100_linux, "heuristic only run on Linux A100")
    # 使用 inductor_config.patch 方法设置 mixed_mm_choice 参数为 "heuristic"
    @inductor_config.patch(mixed_mm_choice="heuristic")
    # 定义一个名为 test_mixed_mm_heuristic_no 的测试方法
    def test_mixed_mm_heuristic_no():
        # 定义一个名为 fn 的内部函数，接受两个参数 a 和 b，返回 torch.mm(a, b.to(a.dtype)) 的结果
        def fn(a, b):
            return torch.mm(a, b.to(a.dtype))

        # 下面是一组例子，这些例子不应该被启发式算法选择
        # 定义 mat1_dtype 变量，指定为 torch.float16 类型
        mat1_dtype = torch.float16
        # 定义 args_list 列表，包含多组元组，每组元组包含两个张量，指定了各种不同的形状和数据类型
        args_list = [
            (
                torch.randn(1, 4097, dtype=mat1_dtype, device="cuda"),  # 随机生成指定形状和数据类型的张量，使用 cuda 设备
                torch.randint(-128, 127, (4097, 4096), dtype=torch.int8, device="cuda"),  # 随机生成指定形状和数据类型的张量，使用 cuda 设备
            ),
            (
                torch.randn(1, 4096, dtype=mat1_dtype, device="cuda"),  # 随机生成指定形状和数据类型的张量，使用 cuda 设备
                torch.randint(-128, 127, (4096, 4097), dtype=torch.int8, device="cuda"),  # 随机生成指定形状和数据类型的张量，使用 cuda 设备
            ),
            (
                torch.randn(8, 8, dtype=mat1_dtype, device="cuda"),  # 随机生成指定形状和数据类型的张量，使用 cuda 设备
                torch.randint(-128, 127, (8, 8), dtype=torch.int8, device="cuda"),  # 随机生成指定形状和数据类型的张量，使用 cuda 设备
            ),
            (
                torch.randn(8, 2048, dtype=mat1_dtype, device="cuda"),  # 随机生成指定形状和数据类型的张量，使用 cuda 设备
                torch.randint(-128, 127, (2048, 2048), dtype=torch.int8, device="cuda"),  # 随机生成指定形状和数据类型的张量，使用 cuda 设备
            ),
            (
                torch.randn(8, 2048, dtype=mat1_dtype, device="cuda"),  # 随机生成指定形状和数据类型的张量，使用 cuda 设备
                torch.randint(  # 随机生成指定形状和数据类型的张量，使用 cuda 设备
                    -128, 127, (2048, 2048), dtype=torch.int8, device="cuda"
                ).t(),  # 对生成的张量进行转置操作
            ),
            (
                torch.randn(8, 4096, dtype=mat1_dtype, device="cuda"),  # 随机生成指定形状和数据类型的张量，使用 cuda 设备
                torch.randint(  # 随机生成指定形状和数据类型的张量，使用 cuda 设备
                    -128, 127, (4096, 4096), dtype=torch.int8, device="cuda"
                )[:, ::2],  # 对生成的张量进行切片操作
            ),
            (
                torch.randn(1, 4096, dtype=torch.float32, device="cuda"),  # 随机生成指定形状和数据类型的张量，使用 cuda 设备
                torch.randint(-128, 127, (4096, 4096), dtype=torch.int8, device="cuda"),  # 随机生成指定形状和数据类型的张量，使用 cuda 设备
            ),
        ]

        # 遍历 args_list 列表中的每组参数 args，并调用 self._test_mixed_impl 方法进行测试
        for args in args_list:
            self._test_mixed_impl(fn, args, True, True)

    # 使用 unittest.skipIf 装饰器，检查条件是否为假，如果为假，则跳过测试
    @unittest.skipIf(not SM80OrLater, "need sm_80")
    @unittest.skipIf(not is_a100_linux, "heuristic only run on Linux A100")
    # 使用 inductor_config.patch 装饰器，将 mixed_mm_choice 参数设置为 "heuristic"
    @inductor_config.patch(mixed_mm_choice="heuristic")
    # 定义测试函数 test_mixed_mm_heuristic_yes，用于测试混合矩阵乘法的启发式策略
    def test_mixed_mm_heuristic_yes(self):
        # 定义内部函数 fn，执行矩阵乘法并确保结果类型与第一个矩阵相同
        def fn(a, b):
            return torch.mm(a, b.to(a.dtype))

        # 设置第一个矩阵的数据类型为 float16
        mat1_dtype = torch.float16
        # 以下是一组示例参数，这些参数将会被启发式策略选中
        args_list = [
            (
                # 创建一个形状为 (1, 4096) 的随机张量，数据类型为 mat1_dtype，在 CUDA 设备上
                torch.randn(1, 4096, dtype=mat1_dtype, device="cuda"),
                # 创建一个形状为 (4096, 4096) 的随机整数张量，数据类型为 int8，在 CUDA 设备上
                torch.randint(-128, 127, (4096, 4096), dtype=torch.int8, device="cuda"),
            ),
            (
                torch.randn(4, 4096, dtype=mat1_dtype, device="cuda"),
                torch.randint(-128, 127, (4096, 4096), dtype=torch.int8, device="cuda"),
            ),
            (
                torch.randn(8, 4096, dtype=mat1_dtype, device="cuda"),
                torch.randint(-128, 127, (4096, 4096), dtype=torch.int8, device="cuda"),
            ),
            (
                torch.randn(8, 4096, dtype=mat1_dtype, device="cuda"),
                torch.randint(
                    -128, 127, (4096, 4096), dtype=torch.int8, device="cuda"
                ).t(),  # 转置后的整数张量
            ),
            (
                torch.randn(16, 4096, dtype=mat1_dtype, device="cuda"),
                torch.randint(
                    -128, 127, (8192, 4096), dtype=torch.int8, device="cuda"
                ).t(),  # 转置后的整数张量
            ),
            (
                torch.randn(32, 4096, dtype=mat1_dtype, device="cuda"),
                torch.randint(-128, 127, (4096, 8192), dtype=torch.int8, device="cuda"),
            ),
            (
                torch.randn(64, 4096, dtype=mat1_dtype, device="cuda"),
                torch.randint(-128, 127, (4096, 4096), dtype=torch.int8, device="cuda"),
            ),
        ]

        # 对每组参数进行测试
        for args in args_list:
            # 调用 _test_mixed_impl 方法，进行混合实现的测试，期望结果为 True
            self._test_mixed_impl(fn, args, True, False, rtol=0.01, atol=0.04)

    # 如果不满足条件 SM80OrLater，跳过当前测试
    @unittest.skipIf(not SM80OrLater, "need sm_80")
    # 定义一个测试函数，用于测试混合矩阵乘法和门控功能
    def test_mixed_mm_gating(self):
        # 定义一个内部函数fn，实现torch中的矩阵乘法，将b转换为与a相同的数据类型后进行乘法运算
        def fn(a, b):
            return torch.mm(a, b.to(a.dtype))

        # 准备测试参数args，包括两个张量，一个是随机生成的8x8张量，另一个是随机整数张量，数据类型为torch.int8，均放置在CUDA设备上
        args = (
            torch.randn(8, 8, device="cuda"),
            torch.randint(-128, 127, (8, 8), dtype=torch.int8, device="cuda"),
        )

        # 使用inductor_config.patch上下文管理器，修改配置为不使用混合矩阵乘法，执行测试函数_test_mixed_impl，并验证期望的结果
        with inductor_config.patch(
            {"mixed_mm_choice": "default", "use_mixed_mm": False}
        ):
            self._test_mixed_impl(fn, args, False, False)

        # 使用inductor_config.patch上下文管理器，修改配置为使用混合矩阵乘法，执行测试函数_test_mixed_impl，并验证期望的结果
        with inductor_config.patch(
            {"mixed_mm_choice": "default", "use_mixed_mm": True}
        ):
            self._test_mixed_impl(fn, args, True, True)

        # 使用inductor_config.patch上下文管理器，修改配置为使用triton选择的混合矩阵乘法，执行测试函数_test_mixed_impl，并验证期望的结果
        with inductor_config.patch(
            {"mixed_mm_choice": "triton", "use_mixed_mm": False}
        ):
            self._test_mixed_impl(fn, args, True, False)

        # 使用inductor_config.patch上下文管理器，修改配置为强制使用混合矩阵乘法，忽略use_mixed_mm设置，执行测试函数_test_mixed_impl，并验证期望的结果
        with inductor_config.patch({"mixed_mm_choice": "triton", "use_mixed_mm": True}):
            self._test_mixed_impl(fn, args, True, False)

        # 使用inductor_config.patch上下文管理器，修改配置为不使用混合矩阵乘法，执行测试函数_test_mixed_impl，并验证期望的结果
        with inductor_config.patch({"mixed_mm_choice": "aten", "use_mixed_mm": False}):
            self._test_mixed_impl(fn, args, True, True)

        # 使用inductor_config.patch上下文管理器，修改配置为使用混合矩阵乘法，执行测试函数_test_mixed_impl，并验证期望的结果
        with inductor_config.patch({"mixed_mm_choice": "aten", "use_mixed_mm": True}):
            self._test_mixed_impl(fn, args, True, True)

        # 使用inductor_config.patch上下文管理器，修改配置为使用混合矩阵乘法，但由于只有fallback选项，仍然使用fallback_mixed_mm核心，执行测试函数_test_mixed_impl，并验证期望的结果
        with inductor_config.patch(
            {"mixed_mm_choice": "aten", "use_mixed_mm": True, "max_autotune_gemm": True}
        ):
            self._test_mixed_impl(fn, args, True, True)

    # 使用inductor_config.patch修饰器，设置使用混合矩阵乘法为True的测试函数
    @inductor_config.patch(use_mixed_mm=True)
    def test_mixed_mm_cpu(self):
        # 定义一个内部函数fn，实现torch中的矩阵乘法，将b转换为与a相同的数据类型后进行乘法运算
        def fn(a, b):
            return torch.mm(a, b.to(a.dtype))

        # 准备测试参数args，包括两个张量，一个是随机生成的8x8张量，另一个是随机整数张量，数据类型为torch.int8
        args = (
            torch.randn(8, 8),
            torch.randint(-128, 127, (8, 8), dtype=torch.int8),
        )

        # 执行测试函数_test_mixed_impl，并验证期望的结果为不使用混合矩阵乘法
        self._test_mixed_impl(fn, args, False, False)

    # 使用unittest.skipIf修饰器，当条件为True时跳过测试，同时使用inductor_config.patch修饰器，设置使用混合矩阵乘法为True
    @unittest.skipIf(not SM80OrLater, "need sm_80")
    @inductor_config.patch(use_mixed_mm=True)
    # 定义测试函数 test_uint4x2_mixed_mm
    def test_uint4x2_mixed_mm(self):
        
        # 定义内部函数 fn，执行 torch.mm 操作
        def fn(a, b):
            return torch.mm(
                a,
                torch.cat((b & 0xF, b >> 4), 1)  # 将 b 按位与和右移拼接成新的张量
                .reshape(-1, b.shape[1])  # 将拼接后的张量重塑形状
                .to(a.dtype)  # 转换为与 a 相同的数据类型
                .sub(8),  # 减去标量值 8
            )

        # 定义检查函数 check_uint4x2_mixed_mm，用于验证混合 mm 操作是否符合预期
        def check_uint4x2_mixed_mm(args, expect_mixed_mm):
            torch._dynamo.reset()  # 重置 Torch 动态编译器状态
            counters.clear()  # 清除计数器
            ref = fn(*args)  # 计算参考结果
            test, (code,) = run_and_get_code(torch.compile(fn), *args)  # 运行并获取编译后的代码和测试结果
            torch.testing.assert_close(ref, test)  # 断言参考结果与测试结果近似
            self.assertEqual("uint4x2_mixed_mm" in code, expect_mixed_mm)  # 验证代码中是否包含 "uint4x2_mixed_mm"

        # 定义包含预期混合 mm 的参数列表
        args_expect_mixed_mm = [
            (
                torch.randn(8, 8, device="cuda"),  # 随机张量 a
                torch.randint(0, 255, (4, 8), dtype=torch.uint8, device="cuda"),  # 随机整数张量 b
            ),
            (
                torch.randn(8, 8, device="cuda", dtype=torch.float16),  # 随机张量 a，指定数据类型为 torch.float16
                torch.randint(0, 255, (4, 8), dtype=torch.uint8, device="cuda")  # 随机整数张量 b
                .t()  # 转置
                .contiguous()  # 保证张量是连续的
                .t(),  # 再次转置
            ),
        ]

        # 对每组参数执行检查混合 mm 操作的验证
        for args in args_expect_mixed_mm:
            check_uint4x2_mixed_mm(args, True)

        # 定义不包含预期混合 mm 的参数列表
        args_expect_no_mixed_mm = [
            (
                torch.randn(8, 8, device="cuda"),  # 随机张量 a
                torch.randint(0, 255, (4, 8), dtype=torch.int32, device="cuda"),  # 随机整数张量 b，数据类型为 torch.int32
            ),
            (
                torch.randn(8, 8, device="cuda"),  # 随机张量 a
                torch.randint(0, 255, (4, 8), dtype=torch.int64, device="cuda"),  # 随机整数张量 b，数据类型为 torch.int64
            ),
        ]

        # 对每组参数执行检查不包含混合 mm 操作的验证
        for args in args_expect_no_mixed_mm:
            check_uint4x2_mixed_mm(args, False)

    # 如果不支持 SM80 或更高的 GPU 架构，则跳过测试
    @unittest.skipIf(not SM80OrLater, "need sm_80")
    # 使用混合 mm 的修饰器配置
    @inductor_config.patch(use_mixed_mm=True)
    # 定义测试函数 test_uint4x2_mixed_mm_epi
    def test_uint4x2_mixed_mm_epi(self):
        
        # 定义内部函数 fn，执行带有额外计算的 torch.mm 操作
        def fn(a, b, c, d):
            return (
                torch.mm(
                    a,
                    torch.cat((b & 0xF, b >> 4), 1)  # 将 b 按位与和右移拼接成新的张量
                    .reshape(-1, b.shape[1])  # 将拼接后的张量重塑形状
                    .to(a.dtype)  # 转换为与 a 相同的数据类型
                    .sub(8),  # 减去标量值 8
                )
                * c  # 乘以张量 c
                + d  # 加上张量 d
            )

        # 定义包含参数列表的 args_list
        args_list = [
            (
                torch.randn(8, 8, device="cuda"),  # 随机张量 a
                torch.randint(0, 255, (4, 8), dtype=torch.uint8, device="cuda"),  # 随机整数张量 b
                torch.randn(8, device="cuda"),  # 随机张量 c
                torch.randn(8, device="cuda"),  # 随机张量 d
            ),
        ]

        # 对每组参数执行测试
        for args in args_list:
            torch._dynamo.reset()  # 重置 Torch 动态编译器状态
            counters.clear()  # 清除计数器
            ref = fn(*args)  # 计算参考结果
            test, (code,) = run_and_get_code(torch.compile(fn), *args)  # 运行并获取编译后的代码和测试结果
            torch.testing.assert_close(ref, test)  # 断言参考结果与测试结果近似
            self.assertTrue("uint4x2_mixed_mm" in code)  # 验证代码中是否包含 "uint4x2_mixed_mm"
            self.assertTrue("fused_add_mm_mul" in code)  # 验证代码中是否包含 "fused_add_mm_mul"

    # 使用混合 mm 的修饰器配置
    @inductor_config.patch(use_mixed_mm=True)
    # 定义一个测试方法，用于测试 uint4x2_mixed_mm 失败的情况是否匹配
    def test_uint4x2_mixed_mm_fail_to_match(self):
        
        # 定义一个函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 使用 torch.mm 计算矩阵乘法，其中 b 被分为低四位和高四位后拼接起来
            return torch.mm(
                a,
                torch.cat((b & 0xF, b >> 4), 1)  # 拼接操作，低四位和高四位
                .reshape(-1, b.shape[1])  # 重新形状为指定的形状
                .to(a.dtype)  # 转换为 a 的数据类型
                .sub(8),  # 减去 8
            )

        # 定义参数列表 args_list
        args_list = [
            (  # cpu
                torch.randn(8, 8),  # 随机生成 8x8 的张量
                torch.randint(0, 255, (4, 8), dtype=torch.uint8),  # 生成 4x8 的 uint8 类型的随机整数张量
            ),
            (  # int8
                torch.randn(8, 8, device="cuda"),  # 在 CUDA 设备上生成 8x8 的随机张量
                torch.randint(-128, 127, (4, 8), dtype=torch.int8, device="cuda"),  # 在 CUDA 设备上生成 4x8 的 int8 类型的随机整数张量
            ),  # 对于 int8，由于数字在 Triton 和 PyTorch 之间的位移不匹配，因此不匹配
        ]  # 对 int8 的位移在 Triton 和 PyTorch 之间不匹配

        # 遍历参数列表 args_list
        for args in args_list:
            torch._dynamo.reset()  # 重置 torch._dynamo
            counters.clear()  # 清空计数器
            ref = fn(*args)  # 调用函数 fn 计算参考结果 ref
            test, (code,) = run_and_get_code(torch.compile(fn), *args)  # 运行并获取编译后的代码和测试结果
            torch.testing.assert_close(ref, test)  # 使用 PyTorch 的测试工具检查 ref 和 test 是否接近
            self.assertFalse("uint4x2_mixed_mm" in code)  # 断言编译后的代码中不包含 "uint4x2_mixed_mm"

    # 定义一个测试方法，用于测试 uint4x2_mixed_mm 门控的工作情况
    @inductor_config.patch(mixed_mm_choice="default")
    @inductor_config.patch(use_mixed_mm=False)
    def test_uint4x2_mixed_mm_gating_works(self):
        
        # 定义一个函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 使用 torch.mm 计算矩阵乘法，其中 b 被分为低四位和高四位后拼接起来
            return torch.mm(
                a,
                torch.cat((b & 0xF, b >> 4), 1)  # 拼接操作，低四位和高四位
                .reshape(-1, b.shape[1])  # 重新形状为指定的形状
                .to(a.dtype)  # 转换为 a 的数据类型
                .sub(8),  # 减去 8
            )

        # 定义参数列表 args_list
        args_list = [
            (
                torch.randn(8, 8, device="cuda"),  # 在 CUDA 设备上生成 8x8 的随机张量
                torch.randint(0, 255, (4, 8), dtype=torch.uint8, device="cuda"),  # 在 CUDA 设备上生成 4x8 的 uint8 类型的随机整数张量
            ),
        ]

        # 遍历参数列表 args_list
        for args in args_list:
            torch._dynamo.reset()  # 重置 torch._dynamo
            counters.clear()  # 清空计数器
            ref = fn(*args)  # 调用函数 fn 计算参考结果 ref
            test, (code,) = run_and_get_code(torch.compile(fn), *args)  # 运行并获取编译后的代码和测试结果
            torch.testing.assert_close(ref, test)  # 使用 PyTorch 的测试工具检查 ref 和 test 是否接近
            self.assertFalse("uint4x2_mixed_mm" in code)  # 断言编译后的代码中不包含 "uint4x2_mixed_mm"
    def test_addmm(self):
        # 定义一个嵌套函数 fn，接受三个参数，返回两个张量的和以及它们的矩阵乘积加和
        def fn(a, b, c):
            return torch.add(a, torch.mm(b, c)), torch.mm(b, c) + a

        # 准备多组参数列表进行测试
        args_list = [
            (
                torch.randn(16, 16, device="cuda"),  # 第一个参数 a，16x16 大小的随机张量在 CUDA 上
                torch.randn(16, 16, device="cuda"),  # 第二个参数 b，16x16 大小的随机张量在 CUDA 上
                torch.randn(16, 16, device="cuda"),  # 第三个参数 c，16x16 大小的随机张量在 CUDA 上
                True,  # 应当进行融合的标志
            ),
            (
                torch.randn(8, device="cuda"),  # 第一个参数 a，8 大小的随机张量在 CUDA 上
                torch.randn(16, 16, device="cuda"),  # 第二个参数 b，16x16 大小的随机张量在 CUDA 上
                torch.randn(16, 8, device="cuda"),  # 第三个参数 c，16x8 大小的随机张量在 CUDA 上
                True,  # 应当进行融合的标志
            ),
            (
                torch.randn(16, 16, device="cuda"),  # 第一个参数 a，16x16 大小的随机张量在 CUDA 上
                torch.randn(1, 16, device="cuda"),  # 第二个参数 b，1x16 大小的随机张量在 CUDA 上
                torch.randn(16, 16, device="cuda"),  # 第三个参数 c，16x16 大小的随机张量在 CUDA 上
                False,  # 不应进行融合的标志
            ),
            (
                torch.randn(1, 16, 16, device="cuda"),  # 第一个参数 a，1x16x16 大小的随机张量在 CUDA 上
                torch.randn(16, 16, device="cuda"),  # 第二个参数 b，16x16 大小的随机张量在 CUDA 上
                torch.randn(16, 16, device="cuda"),  # 第三个参数 c，16x16 大小的随机张量在 CUDA 上
                False,  # 不应进行融合的标志
            ),
            (
                4,  # 第一个参数 a，标量值 4
                torch.randn(16, 16, device="cuda"),  # 第二个参数 b，16x16 大小的随机张量在 CUDA 上
                torch.randn(16, 16, device="cuda"),  # 第三个参数 c，16x16 大小的随机张量在 CUDA 上
                False,  # 不应进行融合的标志
            ),
        ]
        # 遍历参数列表进行测试
        for a, b, c, should_fuse in args_list:
            # 重置 Torch 动态图计算状态
            torch._dynamo.reset()
            # 清空计数器
            counters.clear()
            # 构造参数元组
            args = (a, b, c)
            # 调用 fn 函数得到期望的结果 e1, e2
            e1, e2 = fn(*args)
            # 对 fn 函数进行编译并调用，得到实际的结果 a1, a2
            a1, a2 = torch.compile(fn)(*args)
            # 使用 Torch 测试工具断言 a1 等于 e1，a2 等于 e2
            torch.testing.assert_close(a1, e1)
            torch.testing.assert_close(a2, e2)
            # 根据应当融合的标志确定期望的计数和节点数
            count, nodes = (2, 4) if should_fuse else (0, 0)
            # 断言计数器中的模式匹配计数和节点数与期望值相等
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], count)
            self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], nodes)

    def test_addmm_symbolic_scalar(self):
        # 定义一个函数 fn，接受两个参数，计算第一个参数的大小作为偏置，返回加和和矩阵乘积加和
        def fn(m1, m2):
            bias = m1.size(0)
            return torch.add(bias, torch.mm(m1, m2)), torch.mm(m1, m2) + bias

        # 创建两个16x16大小的随机张量 m1 和 m2 在 CUDA 上
        m1 = torch.randn(16, 16, device="cuda")
        m2 = torch.randn(16, 16, device="cuda")

        # 清空计数器
        counters.clear()
        # 计算预期的结果 expect
        expect = fn(m1, m2)
        # 使用动态编译执行 fn 函数得到实际结果 actual
        actual = torch.compile(fn, dynamic=True)(m1, m2)
        # 断言预期结果与实际结果相等
        self.assertEqual(expect, actual)
        # 断言计数器中的模式匹配计数为 0
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 0)

    def test_addmm_broadcasting_bias(self):
        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.functional.linear
                self.linear_weight = torch.randn(4, 4).cuda()
                self.bias = torch.randn(1, 4).cuda()

            # 定义 forward 方法，对输入张量进行线性变换
            def forward(self, x):
                x = self.linear(x, self.linear_weight, self.bias)
                return x

        # 创建一个大小为 1x3x4 的随机输入张量在 CUDA 上
        input_tensor = torch.randn(1, 3, 4).cuda()

        # 创建 Model 类的实例 func 在 CUDA 上
        func = Model().cuda()

        # 调用 func 对输入张量进行前向传播得到结果 res1
        res1 = func(input_tensor)
        # 对 func 进行编译得到 jit_func
        jit_func = torch.compile(func)
        # 使用 jit_func 对输入张量进行前向传播得到结果 res2
        res2 = jit_func(input_tensor)

        # 断言 res1 与 res2 相等
        self.assertEqual(res1, res2)
    def test_cat_mm(self):
        # 定义一个函数 fn，接受三个参数 a, b, c，将三个矩阵的矩阵乘积拼接起来
        def fn(a, b, c):
            return torch.cat(
                [
                    torch.mm(a, b),  # 计算 a 与 b 的矩阵乘积
                    torch.mm(b, c),  # 计算 b 与 c 的矩阵乘积
                    torch.mm(a, c),  # 计算 a 与 c 的矩阵乘积
                ],
                1,  # 指定沿着列拼接
            )

        # 生成三个随机矩阵作为参数，使用 CUDA 加速
        args = [
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
        ]
        # 调用共同的测试函数 common，对 fn 进行测试，期望的输出维度是 1x4
        self.common(fn, args, 1, 4)

    def test_cat_addmm(self):
        # 定义一个函数 fn，接受三个参数 a, b, c，将三个矩阵的 torch.addmm 结果拼接起来
        def fn(a, b, c):
            return torch.cat(
                [
                    torch.addmm(a, b, c),  # 计算 torch.addmm(a, b, c) 的结果
                    torch.addmm(b, c, a),  # 计算 torch.addmm(b, c, a) 的结果
                    torch.addmm(c, a, b),  # 计算 torch.addmm(c, a, b) 的结果
                ],
                1,  # 指定沿着列拼接
            )

        # 生成三个随机矩阵作为参数，使用 CUDA 加速
        args = [
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
        ]
        # 调用共同的测试函数 common，对 fn 进行测试，期望的输出维度是 1x4
        self.common(fn, args, 1, 4)

    def test_cat_slice_cat_cuda(self):
        # 定义一个函数 fn，接受两个参数 a, b，执行一系列操作后返回拼接结果
        def fn(a, b):
            cat_1 = torch.ops.aten.cat.default([a, b], 1)  # 在维度 1 上拼接 a 和 b
            slice_1 = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)  # 对 cat_1 进行切片操作，从索引 0 开始到最大整数
            slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 19)  # 对 slice_1 进行第二次切片，从索引 0 到 19
            return torch.ops.aten.cat.default([cat_1, slice_2], 1)  # 再次在维度 1 上拼接 cat_1 和 slice_2

        # 第一组参数是两个随机矩阵，使用 CUDA 加速
        args = [
            torch.randn(2, 32, device="cuda"),
            torch.randn(2, 16, device="cuda"),
        ]
        # 调用共同的测试函数 common，对 fn 进行测试，期望的输出维度是 1x3
        self.common(fn, args, 1, 3)

        # 第二组参数是两个随机矩阵，使用 CUDA 加速
        args = [
            torch.randn(2, 8, device="cuda"),
            torch.randn(2, 16, device="cuda"),
        ]
        counters.clear()
        # 期望结果等于 fn(*args) 的值
        expected = fn(*args)
        # 使用 torch.compile(fn) 编译 fn(*args) 的实际结果
        actual = torch.compile(fn)(*args)
        # 断言实际结果与期望结果的近似程度
        torch.testing.assert_close(actual, expected)
        # 当 dynamo_config.assume_static_by_default 为真时，验证是否回退到非优化路径
        if dynamo_config.assume_static_by_default:
            # 检查 pattern_matcher_count 是否为 1
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
            # 检查 pattern_matcher_nodes 是否为 3
            self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 3)

        # 定义一个新的函数 fn，接受两个参数 a, b，执行一系列操作后返回拼接结果
        def fn(a, b):
            cat_1 = torch.ops.aten.cat.default([a, b], 1)  # 在维度 1 上拼接 a 和 b
            slice_1 = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)  # 对 cat_1 进行切片操作，从索引 0 开始到最大整数
            slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 0, -1)  # 对 slice_1 进行第二次切片，从索引 0 到 -1
            return torch.ops.aten.cat.default([cat_1, slice_2], 1)  # 再次在维度 1 上拼接 cat_1 和 slice_2

        # 第一组参数是两个随机矩阵，使用 CUDA 加速
        args = [
            torch.randn(2, 8, device="cuda"),
            torch.randn(2, 16, device="cuda"),
        ]
        # 调用共同的测试函数 common，对 fn 进行测试，期望的输出维度是 1x3
        self.common(fn, args, 1, 3)
    # 定义一个测试函数，用于测试无意义的类型转换
    def test_pointless_convert(self):
        # 定义内部函数 fn1，接受参数 x，对其进行两次类型转换
        def fn1(x):
            # 调用 Torch 操作符转换元素类型，将 x 转换为 torch.float16 类型
            x = torch.ops.prims.convert_element_type.default(x, torch.float16)
            # 再次调用 Torch 操作符转换元素类型，将 x 转换为 torch.float32 类型
            x = torch.ops.prims.convert_element_type.default(x, torch.float32)
            return x

        # 使用 Torch 的符号化跟踪，将 fn1 转换为计算图
        gm = torch.fx.symbolic_trace(fn1)
        # 断言计算图中调用转换类型操作的次数为 2
        self.assertEqual(count_calls(gm.graph), 2)
        # 对计算图应用联合图传递的处理
        joint_graph.joint_graph_passes(gm)
        # 再次断言计算图中调用转换类型操作的次数为 1
        self.assertEqual(count_calls(gm.graph), 1)

        # 定义内部函数 fn2，接受参数 x，对其进行不同的类型转换操作
        def fn2(x):
            # 调用 Torch 操作符转换元素类型，将 x 转换为 torch.int32 类型
            x = torch.ops.prims.convert_element_type.default(x, torch.int32)
            # 再次调用 Torch 操作符转换元素类型，将 x 转换为 torch.float32 类型
            x = torch.ops.prims.convert_element_type.default(x, torch.float32)
            return x

        # 使用 Torch 的符号化跟踪，将 fn2 转换为计算图
        gm = torch.fx.symbolic_trace(fn2)
        # 断言计算图中调用转换类型操作的次数为 2
        self.assertEqual(count_calls(gm.graph), 2)
        # 对计算图应用联合图传递的处理
        joint_graph.joint_graph_passes(gm)
        # 再次断言计算图中调用转换类型操作的次数仍为 2
        self.assertEqual(count_calls(gm.graph), 2)

    # 由于问题＃108388，常量折叠显式关闭
    # 在测试中重新打开常量折叠
    @inductor_config.patch(joint_graph_constant_folding=True)
    # 定义一个测试函数，用于测试无意义的累加求和操作
    def test_pointless_cumsum(self):
        # 定义内部函数 fn1，无参数，创建一个全为1的张量，进行累加求和操作，并与自身相乘
        def fn1():
            ones = torch.full(
                [1, 128], 1, layout=torch.strided, dtype=torch.float32
            ).to(torch.int64)
            return torch.cumsum(ones, 1) * ones

        # 定义内部函数 fn2，无参数，创建一个全为1的张量，进行累加求和操作
        def fn2():
            ones = torch.full(
                [55, 10], 1, layout=torch.strided, dtype=torch.float32
            ).to(torch.int64)
            return torch.cumsum(ones, 1)

        # 定义内部函数 fn3，无参数，创建一个全为2的张量，进行累加求和操作
        def fn3():
            twos = torch.full([5, 4, 3], 2, dtype=torch.int64)
            return torch.cumsum(twos, 0)

        # 定义内部函数 fn4，无参数，创建一个全为0.1的张量，进行累加求和操作
        def fn4():
            x = torch.full([100], 0.1, dtype=torch.float32)
            return torch.cumsum(x, 0)

        # 定义内部函数 fn5，无参数，创建一个全为1的张量，转换为布尔型并进行累加求和操作
        def fn5():
            t1 = torch.full([2, 4], 1)
            t2 = t1.to(dtype=torch.bool)
            return torch.cumsum(t2, 1)

        # 定义内部函数 fn6，无参数，创建一个全为 True 的张量，进行累加求和操作
        def fn6():
            x = torch.full([10, 10], True, dtype=torch.int32)
            return torch.cumsum(x, 1)

        # 遍历所有定义的函数，并分别执行它们，获取结果和生成的代码
        for fn in (fn1, fn2, fn3, fn4, fn5, fn6):
            result, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True))
            # 断言生成的代码中不包含 "aten.cumsum"
            self.assertNotIn("aten.cumsum", code)
            # 断言运行函数的结果与直接调用函数得到的结果相等
            self.assertEqual(result, fn())
            # 断言计数器中"inductor"部分的"pattern_matcher_count"计数为 1
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
            # 清空计数器以便下一次循环使用
            counters.clear()
    # 定义一个测试函数，用于测试 split_with_sizes 和 cat 的组合效果
    def test_splitwithsizes_cat(self):
        # Good case（良好情况）

        # 定义一个函数 fn，接受参数 a
        def fn(a):
            # 使用 torch.ops.aten.split_with_sizes.default 对 a 进行切分操作，指定切分大小 [8, 24]，在维度 1 上进行切分
            split_with_sizes = torch.ops.aten.split_with_sizes.default(a, [8, 24], 1)
            # 获取切分后的第一个部分
            getitem = split_with_sizes[0]
            # 获取切分后的第二个部分
            getitem_1 = split_with_sizes[1]
            # 使用 torch.ops.aten.cat.default 对切分后的部分进行连接，沿着维度 1 连接
            cat = torch.ops.aten.cat.default([getitem, getitem_1], 1)
            # 返回连接后的结果的平方
            return cat**2

        # 准备参数列表，包含一个使用 CUDA 设备上生成的随机张量
        args = [
            torch.randn(2, 32, device="cuda"),
        ]
        # 调用 self.common 函数进行测试，验证 fn 的结果符合预期，期望输出结果在 1 到 4 之间
        self.common(fn, args, 1, 4)

        # Not all getitems are passed to cat（不是所有的 getitem 都被传递给 cat）

        # 定义一个函数 fn，接受参数 a
        def fn(a):
            # 使用 torch.ops.aten.split_with_sizes.default 对 a 进行切分操作，指定切分大小 [8, 8, 16]，在维度 1 上进行切分
            split_with_sizes = torch.ops.aten.split_with_sizes.default(a, [8, 8, 16], 1)
            # 获取切分后的第一个部分
            getitem = split_with_sizes[0]
            # 获取切分后的第二个部分
            getitem_1 = split_with_sizes[1]
            # 获取切分后的第三个部分
            getitem_2 = split_with_sizes[2]
            # 使用 torch.ops.aten.cat.default 对部分 getitem 和 getitem_1 进行连接，沿着维度 1 连接
            cat = torch.ops.aten.cat.default([getitem, getitem_1], 1)
            # 返回连接后的结果的平方，加上第三个部分 getitem_2 的结果
            return cat**2 + getitem_2

        # 准备参数列表，包含一个使用 CUDA 设备上生成的随机张量
        args = [
            torch.randn(2, 32, device="cuda"),
        ]
        # 调用 self.common 函数进行测试，验证 fn 的结果符合预期，期望输出结果为 0
        self.common(fn, args, 0, 0)

        # Different dimensions  (TODO this case should be handled by replacing with a reshape)
        # 不同的维度（TODO：这种情况应该通过重新形状处理来处理）

        # 定义一个函数 fn，接受参数 a
        def fn(a):
            # 使用 torch.ops.aten.split_with_sizes.default 对 a 进行切分操作，指定切分大小 [8, 8, 8, 8]，在维度 1 上进行切分
            split_with_sizes = torch.ops.aten.split_with_sizes.default(
                a, [8, 8, 8, 8], 1
            )
            # 使用 torch.ops.aten.cat.default 对切分后的结果在维度 0 上进行连接
            cat = torch.ops.aten.cat.default(split_with_sizes, 0)
            # 返回连接后的结果的平方
            return cat**2

        # 准备参数列表，包含一个使用 CUDA 设备上生成的随机张量
        args = [
            torch.randn(2, 32, device="cuda"),
        ]
        # 调用 self.common 函数进行测试，验证 fn 的结果符合预期，期望输出结果为 0
        self.common(fn, args, 0, 0)

        # https://github.com/pytorch/pytorch/issues/99686.
        # 处理 GitHub 上的一个问题

        # 定义一个函数 fn，接受参数 a
        def fn(a):
            # 使用 torch.ops.aten.split_with_sizes.default 对 a 进行切分操作，指定切分大小 [3, 2, 3]，在维度 1 上进行切分
            x = torch.ops.aten.split_with_sizes.default(a, [3, 2, 3], dim=1)
            # 使用 torch.ops.aten.cat.default 对切分后的结果进行连接，按顺序连接 x 的第二部分、第一部分和第三部分，沿着维度 1 连接
            cat = torch.ops.aten.cat.default([x[1], x[0], x[2]], dim=1)
            # 返回连接后的结果
            return cat

        # 准备参数列表，包含一个使用 CUDA 设备上生成的随机张量
        args = [
            torch.randn(1, 8, device="cuda"),
        ]
        # 调用 self.common 函数进行测试，验证 fn 的结果符合预期，期望输出结果为 0
        self.common(fn, args, 0, 0)
    # 定义测试方法 test_cat_splitwithsizes，用于测试 torch.ops.aten.cat.default 和 torch.ops.aten.split_with_sizes.default 的功能
    def test_cat_splitwithsizes(self):
        # good case
        # 定义函数 fn，接受三个参数 a, b, c
        def fn(a, b, c):
            # 将 a, b, c 按照维度 1 进行拼接，生成新的张量 cat
            cat = torch.ops.aten.cat.default([a, b, c], 1)
            # 在维度 1 上按照指定的大小 [2, 3, 5] 将 cat 分割成多个子张量，存放在 split_with_sizes 中
            split_with_sizes = torch.ops.aten.split_with_sizes.default(
                cat, [2, 3, 5], 1
            )
            # 返回每个分割子张量的平方组成的列表
            return [s**2 for s in split_with_sizes]

        # 定义测试参数 args
        args = [
            torch.randn(2, 2, device="cuda"),
            torch.randn(2, 3, device="cuda"),
            torch.randn(2, 5, device="cuda"),
        ]
        # 调用 self.common 方法进行测试，预期的输出结果是 1 和 2
        self.common(fn, args, 1, 2)

        # cat node has other users
        # 定义函数 fn，接受三个参数 a, b, c
        def fn(a, b, c):
            # 将 a, b, c 按照维度 1 进行拼接，生成新的张量 cat
            cat = torch.ops.aten.cat.default([a, b, c], 1)
            # 在维度 1 上按照指定的大小 [2, 3, 5] 将 cat 分割成多个子张量，存放在 split_with_sizes 中
            split_with_sizes = torch.ops.aten.split_with_sizes.default(
                cat, [2, 3, 5], 1
            )
            # 返回每个分割子张量的平方组成的列表，同时将 cat 的立方添加到结果列表中
            return [s**2 for s in split_with_sizes] + [cat**3]

        # 定义测试参数 args
        args = [
            torch.randn(2, 2, device="cuda"),
            torch.randn(2, 3, device="cuda"),
            torch.randn(2, 5, device="cuda"),
        ]
        # 调用 self.common 方法进行测试，预期的输出结果是 0 和 0
        self.common(fn, args, 0, 0)

        # cat and split dims are different
        # 定义函数 fn，接受三个参数 a, b, c
        def fn(a, b, c):
            # 将 a, b, c 按照维度 1 进行拼接，生成新的张量 cat
            cat = torch.ops.aten.cat.default([a, b, c], 1)
            # 在维度 0 上按照指定的大小 [2, 3, 5] 将 cat 分割成多个子张量，存放在 split_with_sizes 中
            split_with_sizes = torch.ops.aten.split_with_sizes.default(
                cat, [2, 3, 5], 0
            )
            # 返回每个分割子张量的平方组成的列表
            return [s**2 for s in split_with_sizes]

        # 定义测试参数 args
        args = [
            torch.randn(10, 2, device="cuda"),
            torch.randn(10, 3, device="cuda"),
            torch.randn(10, 5, device="cuda"),
        ]
        # 调用 self.common 方法进行测试，预期的输出结果是 0 和 0
        self.common(fn, args, 0, 0)

        # cat and split lengths are different
        # 定义函数 fn，接受三个参数 a, b, c
        def fn(a, b, c):
            # 将 a, b, c 按照维度 1 进行拼接，生成新的张量 cat
            cat = torch.ops.aten.cat.default([a, b, c], 1)
            # 在维度 1 上按照指定的大小 [5, 5] 将 cat 分割成多个子张量，存放在 split_with_sizes 中
            split_with_sizes = torch.ops.aten.split_with_sizes.default(cat, [5, 5], 1)
            # 返回每个分割子张量的平方组成的列表
            return [s**2 for s in split_with_sizes]

        # 定义测试参数 args
        args = [
            torch.randn(2, 2, device="cuda"),
            torch.randn(2, 3, device="cuda"),
            torch.randn(2, 5, device="cuda"),
        ]
        # 调用 self.common 方法进行测试，预期的输出结果是 0 和 0
        self.common(fn, args, 0, 0)

        # cat input sizes and split sizes are different
        # 定义函数 fn，接受三个参数 a, b, c
        def fn(a, b, c):
            # 将 a, b, c 按照维度 1 进行拼接，生成新的张量 cat
            cat = torch.ops.aten.cat.default([a, b, c], 1)
            # 在维度 1 上按照指定的大小 [2, 5, 3] 将 cat 分割成多个子张量，存放在 split_with_sizes 中
            split_with_sizes = torch.ops.aten.split_with_sizes.default(
                cat, [2, 5, 3], 1
            )
            # 返回每个分割子张量的平方组成的列表
            return [s**2 for s in split_with_sizes]

        # 定义测试参数 args
        args = [
            torch.randn(2, 2, device="cuda"),
            torch.randn(2, 3, device="cuda"),
            torch.randn(2, 5, device="cuda"),
        ]
        # 调用 self.common 方法进行测试，预期的输出结果是 0 和 0
        self.common(fn, args, 0, 0)
    def test_symint_pattern_matching(self):
        # 导入必要的模块和类
        import torch._inductor.config as config
        from torch._inductor.pattern_matcher import (
            fwd_only,
            PatternMatcherPass,
            register_replacement,
        )

        # 初始化保存图的变量
        saved_graph = None

        # 自定义的模式匹配器子类，继承自PatternMatcherPass
        class _CustomPass(PatternMatcherPass):
            def __init__(self):
                super().__init__()

            # 重写__call__方法，将模式匹配器应用于图g
            def __call__(self, g: torch.fx.graph.Graph):
                self.apply(g)
                nonlocal saved_graph
                saved_graph = g

        # 使用config.patch上下文管理器来设置配置选项
        with config.patch(
            pattern_matcher=False,  # 禁用内置模式匹配器
            post_grad_custom_pre_pass=None,  # 定义预处理的自定义模式匹配器为空
            post_grad_custom_post_pass=_CustomPass(),  # 设置后处理的自定义模式匹配器为_CustomPass的实例
        ):

            # 定义一个简单的加法函数
            def add(x, y):
                return x + y

            # 定义一个符号减法函数，用于模式匹配测试
            def sym_minus(x, y):
                return (x - (-y.size(0))) - (y * -1) - y.size(0)

            # 设备类型设置为CPU
            device = "cpu"

            # 定义测试参数
            my_args = [
                torch.empty([8, 1], device=device),  # 创建一个8x1的空张量在CPU上
                torch.empty([10], device=device),   # 创建一个大小为10的空张量在CPU上
            ]

            # 初始化一个标志来检测是否调用了额外检查函数
            invoked = False

            # 定义额外的检查函数，用于模式匹配
            def extra_check(match):
                nonlocal invoked
                invoked = True
                return True

            # 注册替换函数，将add函数替换为sym_minus函数
            register_replacement(
                add,
                sym_minus,
                my_args,
                fwd_only,
                [config.post_grad_custom_post_pass],  # 使用自定义后处理模式匹配器
                extra_check=extra_check,  # 指定额外的检查函数
            )

            # 使用@torch.compile标记动态编译函数
            @torch.compile(dynamic=True)
            def foo(x, y):
                return x + y

            # 创建两个随机张量x和y
            x = torch.rand([8, 1])
            y = torch.rand([10])

            # 断言动态编译的函数foo与普通的加法操作等效
            self.assertEqual(foo(x, y), x + y)

            # 断言已经调用了额外的检查函数
            self.assertTrue(invoked)

            # 进行图形分析，检查保存的图中是否包含特定模式
            FileCheck().check("sym_size_int").check_same("num_users=2").check_same(
                "target=torch.ops.aten.sym_size"
            ).run(str(saved_graph))
    # 定义一个测试函数，用于验证模式匹配是否考虑到突变的影响
    def test_match_with_mutation(self):
        # 初始化计数器
        counter = 0
        # 创建模式匹配器实例，阻止跨突变的匹配，命名为 "test"
        test_pass = PatternMatcherPass(
            prevent_match_across_mutations=True, pass_name="test"
        )

        # 定义一个装饰器函数，注册一个图模式，匹配 torch.add(x, torch.sin(x)) 的形式
        @register_graph_pattern(
            CallFunction(
                torch.add, KeywordArg("x"), CallFunction(torch.sin, KeywordArg("x"))
            ),
            pass_dict=test_pass,
        )
        def _test(match, x):
            nonlocal counter
            counter += 1

        # 定义多个测试函数，用于生成不同的计算图
        def fn0(x, y):
            a = torch.sin(x)
            b = torch.add(x, a)
            return b

        def fn1(x, y):
            a = torch.sin(x)
            x.copy_(y)
            b = torch.add(x, a)
            return b

        def fn2(x, y):
            a = torch.sin(x)
            with torch.no_grad():
                b = torch.add(x, a)
            return b

        def fn3(x, y):
            a = torch.sin(x)
            with torch.autocast("cuda"):
                b = torch.add(x, a)
            return b

        def fn4(x, y):
            a = torch.sin(x)
            torch.manual_seed(1234)
            b = torch.add(x, a)
            return b

        def fn5(x, y):
            a = torch.sin(x)
            torch.add(y, 1, out=x)
            b = torch.add(x, a)
            return b

        # 准备测试参数
        args = [
            torch.randn(5, 5, device="cuda"),
            torch.randn(5, 5, device="cuda"),
        ]

        # 使用模拟对象打补丁，修改预设的配置和模式
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pre_grad.config.pre_grad_fusion_options",
            {"test": {}},
        ), unittest.mock.patch(
            "torch._inductor.fx_passes.pre_grad.PRE_GRAD_FUSIONS",
            [],
        ), unittest.mock.patch(
            "torch._inductor.fx_passes.pre_grad.PRE_GRAD_PATTERNS",
            {"test": test_pass},
        ):
            # 遍历每个测试函数并执行测试
            for fn in (fn0, fn1, fn2, fn3, fn4, fn5):
                counter = 0
                # 获取预期结果和实际结果
                expected = fn(*copy.deepcopy(args))
                actual = torch.compile(fn)(*copy.deepcopy(args))
                # 断言匹配次数是否符合预期
                # 应当只有 fn 是 fn0 时才匹配成功
                self.assertEqual(counter, int(fn is fn0))
                # 断言实际计算结果与预期结果的接近程度
                torch.testing.assert_close(actual, expected)

    # 测试移除无意义克隆操作的函数
    def test_remove_pointless_clones(self):
        # 定义一个编译的函数 fn，其完整图形被编译
        @torch.compile(fullgraph=True)
        def fn(a, b):
            # 执行矩阵乘法并克隆结果
            return torch.mm(a, b).clone()

        # 运行函数并获取其生成的代码
        result, (code) = run_and_get_code(fn, torch.randn(8, 8), torch.randn(8, 8))
        # 断言代码中是否存在指定的返回形式
        self.assertIn("return (buf0, )", code[0])
        # 断言代码中是否不包含特定的字符串，这里是 async_compile.cpp
        self.assertNotIn("async_compile.cpp", code[0])
    # 定义测试函数，用于测试 addmm 操作的不同情况
    def test_unfuse_bias_addmm(self):
        # 准备输入参数列表，包括三个随机生成的 CUDA 张量
        args = [
            torch.randn(20, device="cuda"),
            torch.randn(10, 15, device="cuda"),
            torch.randn(15, 20, device="cuda"),
        ]

        # 定义并编译一个函数 fn，该函数执行 torch.ops.aten.addmm 操作并返回结果
        @torch.compile()
        def fn(inp, a, b):
            return torch.ops.aten.addmm(inp, a, b)

        # 运行函数 fn，并获取其生成的代码
        _, (code) = run_and_get_code(fn, args[0], args[1], args[2])
        # 使用 FileCheck 检查生成的代码中是否包含 "extern_kernels.addmm(" 字符串
        FileCheck().check("extern_kernels.addmm(").run(code[0])

        # 定义并编译另一个函数 fn2，该函数在执行 addmm 操作后使用 torch.nn.functional.gelu 进行处理
        @torch.compile()
        def fn2(inp, a, b):
            return torch.nn.functional.gelu(torch.ops.aten.addmm(inp, a, b))

        # 运行函数 fn2，并获取其生成的代码
        _, (code) = run_and_get_code(fn2, args[0], args[1], args[2])
        # 使用 FileCheck 检查生成的代码中是否不包含 "extern_kernels.addmm(" 字符串
        FileCheck().check_not("extern_kernels.addmm(").run(code[0])

        # 定义并编译另一个函数 fn2，该函数在执行 addmm 操作后先进行 unsqueeze(0)，然后使用 gelu 处理
        @torch.compile()
        def fn2(inp, a, b):
            return torch.nn.functional.gelu(
                torch.ops.aten.addmm(inp, a, b).unsqueeze(0)
            )

        # 运行函数 fn2，并获取其生成的代码
        _, (code) = run_and_get_code(fn2, args[0], args[1], args[2])
        # 使用 FileCheck 检查生成的代码中是否不包含 "extern_kernels.addmm(" 字符串
        FileCheck().check_not("extern_kernels.addmm(").run(code[0])

    # 测试序列化模式是否最新的函数
    def test_serialized_patterns_up_to_date(self):
        # 导入必要的模块
        import torch.utils._pytree as pytree
        from torch._inductor.fx_passes import joint_graph
        from torch._inductor.pattern_matcher import _known_precompiled_patterns

        # 确保模式已加载，移除可能的环境变量设置
        os.environ.pop("PYTORCH_GEN_PATTERNS", None)
        # 执行懒初始化以加载模式
        joint_graph.lazy_init()

        # 进入 FakeTensorMode 上下文，以处理模拟张量
        with torch._subclasses.FakeTensorMode() as mode:
            # 遍历已知的预编译模式列表
            for (
                search_fn,
                example_inputs,
                trace_fn,
                scalar_workaround,
                search_fn_pattern,
            ) in _known_precompiled_patterns:
                # 将示例输入映射到当前 FakeTensorMode，以更新为当前模式的模拟张量
                def remap_fake_tensor(x):
                    if isinstance(x, torch.Tensor):
                        return torch._subclasses.FakeTensor.from_tensor(x, mode)
                    return x

                example_inputs = pytree.tree_map(remap_fake_tensor, example_inputs)

                # 生成模式
                pattern = gen_pattern(
                    search_fn, example_inputs, trace_fn, scalar_workaround
                )
                # 运行模式的美化打印器
                pattern_pp = PatternPrettyPrinter.run(pattern)

                # 使用断言检查生成的模式是否与预期的搜索函数模式匹配
                self.assertEqual(
                    pattern_pp,
                    PatternPrettyPrinter.run(search_fn_pattern),
                    msg=f"Found mismatched pattern {search_fn.__name__}. Run torchgen/fuse/gen_patterns.py",
                )

                # 验证序列化器，确保生成的模式也与搜索函数模式匹配
                self.assertTrue(pattern.pattern_eq(search_fn_pattern))
    def test_match_equivalent_function_invocations1(self):
        # 初始化计数器
        counter = 0
        # 创建一个 PatternMatcherPass 实例，阻止跨变异进行匹配
        test_pass = PatternMatcherPass(prevent_match_across_mutations=True)

        # 定义输入参数列表
        args = [
            torch.randn(20, device="cuda"),
            torch.randn(10, 15, device="cuda"),
            torch.randn(15, 20, device="cuda"),
        ]

        # 定义三个函数 f0, f1, f2，分别调用 torch.ops.aten.addmm 函数
        def f0(inp, a, b):
            return torch.ops.aten.addmm(inp, a, b)

        def f1(inp, a, b):
            return torch.ops.aten.addmm(inp, a, b, beta=1.0)

        def f2(inp, a, b):
            return torch.ops.aten.addmm(inp, a, b, beta=1.0, alpha=1.0)

        # 注册图模式，匹配 torch.ops.aten.addmm 函数调用，并传递一个参数字典和测试实例
        @register_graph_pattern(
            CallFunction(
                torch.ops.aten.addmm,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            pass_dict=test_pass,
        )
        # 定义 addmm_replacement 函数，用于替换匹配到的 addmm 函数调用
        def addmm_replacement(match: Match, inp, mat1, mat2, beta, alpha):
            nonlocal counter
            # 增加计数器值
            counter += 1

            # 定义替换函数 repl，实现新的计算逻辑
            def repl(inp, x1, x2):
                return (x1 @ x2) * alpha + inp * beta

            # 在虚拟模式下，使用匹配对象替换示例函数 repl，并传递输入参数
            with V.fake_mode:
                match.replace_by_example(repl, [inp, mat1, mat2])

        # 使用 unittest.mock.patch 注入模拟模块，扩展 pass_patterns 并添加 test_pass
        with unittest.mock.patch(
            "torch._inductor.fx_passes.post_grad.pass_patterns",
            torch._inductor.fx_passes.post_grad.pass_patterns + [test_pass],
        ):
            # 对每个函数 fn 进行测试
            for fn in (f0, f1, f2):
                # 重置计数器
                counter = 0
                # 计算预期结果
                expected = fn(*copy.deepcopy(args))
                # 编译优化函数 fn
                opt_fn = torch.compile(fn)
                # 运行优化后的函数，并获取结果及代码
                actual, (code) = run_and_get_code(opt_fn, args[0], args[1], args[2])
                # 断言模式应该匹配一次
                self.assertEqual(counter, 1)
                # 检查实际输出是否与预期接近
                torch.testing.assert_close(actual, expected)
                # 检查 addmm 是否被替换
                FileCheck().check_not("extern_kernels.addmm(").run(code[0])
    def test_match_equivalent_function_invocations2(self):
        counter = 0  # 初始化计数器为0，用于统计匹配次数

        # 创建一个 PatternMatcherPass 对象，禁止跨突变进行匹配
        test_pass = PatternMatcherPass(prevent_match_across_mutations=True)

        # 准备输入参数列表
        args = [
            torch.randn(20, device="cuda"),
            torch.randn(10, 15, device="cuda"),
            torch.randn(15, 20, device="cuda"),
        ]

        # 定义三个函数，每个函数调用了 torch 的 aten.addmm 操作
        def f0(inp, a, b):
            return torch.ops.aten.addmm(inp, a, b)

        def f1(inp, a, b):
            return torch.ops.aten.addmm(inp, a, b, beta=1.0)

        def f2(inp, a, b):
            return torch.ops.aten.addmm(inp, a, b, beta=1.0, alpha=1.0)

        # 注册一个图模式匹配器，用于匹配调用 torch.ops.aten.addmm 的函数
        @register_graph_pattern(
            CallFunction(torch.ops.aten.addmm, Arg(), Arg(), Arg()),
            pass_dict=test_pass,
        )
        def addmm_replacement(match: Match, inp, mat1, mat2):
            nonlocal counter  # 使用外部的计数器变量
            counter += 1  # 每匹配一次，计数器加1

            # 定义一个替换函数 repl，用于替换匹配到的模式
            def repl(inp, x1, x2):
                return x1 @ x2 + inp

            # 使用虚拟环境 V.fake_mode 执行替换
            with V.fake_mode:
                match.replace_by_example(repl, [inp, mat1, mat2])

        # 使用 unittest.mock.patch 修改 "torch._inductor.fx_passes.post_grad.pass_patterns"
        # 将 test_pass 添加到 pass_patterns 列表中
        with unittest.mock.patch(
            "torch._inductor.fx_passes.post_grad.pass_patterns",
            torch._inductor.fx_passes.post_grad.pass_patterns + [test_pass],
        ):
            # 对 f0, f1, f2 进行测试
            for fn in (f0, f1, f2):
                counter = 0  # 每个函数测试前重置计数器
                expected = fn(*copy.deepcopy(args))  # 深拷贝参数，获取预期输出
                actual = torch.compile(fn)(*copy.deepcopy(args))  # 编译函数并执行，获取实际输出
                self.assertEqual(counter, 1)  # 断言计数器只增加了一次，表示只匹配了一次模式
                torch.testing.assert_close(actual, expected)  # 断言实际输出与预期输出的近似性
    # 定义一个测试函数，用于测试模式匹配功能调用的等价性
    def test_match_equivalent_function_invocations3(self):
        # 初始化计数器
        counter = 0
        # 创建一个模式匹配器对象，设置禁止跨变异匹配
        test_pass = PatternMatcherPass(prevent_match_across_mutations=True)

        # 准备输入参数列表
        args = [
            torch.randn(20, device="cuda"),
            torch.randn(10, 15, device="cuda"),
            torch.randn(15, 20, device="cuda"),
        ]

        # 定义三个函数，每个函数调用 torch 的 aten.addmm 操作
        def f0(inp, a, b):
            return torch.ops.aten.addmm(inp, a, b)

        def f1(inp, a, b):
            return torch.ops.aten.addmm(inp, a, b, beta=1.0)

        def f2(inp, a, b):
            return torch.ops.aten.addmm(inp, a, b, beta=1.0, alpha=1.0)

        # 注册一个图形模式匹配器，用于匹配 torch.ops.aten.addmm 函数调用
        @register_graph_pattern(
            CallFunction(
                torch.ops.aten.addmm, Arg(), Arg(), Arg(), beta=KeywordArg("beta")
            ),
            pass_dict=test_pass,
        )
        # 定义一个替换函数，用于匹配到的模式
        def addmm_replacement(match: Match, inp, mat1, mat2, beta):
            nonlocal counter
            counter += 1

            # 定义一个替换函数的具体实现
            def repl(inp, x1, x2):
                return x1 @ x2 + inp

            # 在虚拟模式下替换匹配到的模式
            with V.fake_mode:
                match.replace_by_example(repl, [inp, mat1, mat2])

        # 使用 unittest.mock.patch 修饰器，扩展 torch._inductor.fx_passes.post_grad.pass_patterns 列表
        with unittest.mock.patch(
            "torch._inductor.fx_passes.post_grad.pass_patterns",
            torch._inductor.fx_passes.post_grad.pass_patterns + [test_pass],
        ):
            # 对每个函数进行测试
            for fn in (f0, f1, f2):
                # 重置计数器
                counter = 0
                # 复制参数列表，执行预期函数
                expected = fn(*copy.deepcopy(args))
                # 编译函数并执行，获取实际结果
                actual = torch.compile(fn)(*copy.deepcopy(args))
                # 断言替换计数为1
                self.assertEqual(counter, 1)
                # 断言实际结果与预期结果近似相等
                torch.testing.assert_close(actual, expected)

    # 定义一个测试函数，用于测试稳定的拓扑排序算法
    def test_stable_topological_sort(self):
        # 定义一个简单的加法函数
        def fn1(a, b):
            return a + b

        # 创建一个空的 Torch FX 图形对象
        graph = torch.fx.Graph()
        # 创建两个图形占位符节点
        a = graph.placeholder("x")
        b = graph.placeholder("y")
        # 在图中调用函数 fn1
        c = graph.call_function(fn1, (a, b))
        # 对图执行稳定的拓扑排序
        stable_topological_sort(graph)
        # 断言图的节点顺序符合预期
        self.assertEqual(list(graph.nodes), [a, b, c])

        # 创建另一个空的 Torch FX 图形对象
        graph = torch.fx.Graph()
        # 创建两个图形占位符节点（调换顺序）
        b = graph.placeholder("y")
        a = graph.placeholder("x")
        # 在图中调用函数 fn1
        c = graph.call_function(fn1, (a, b))
        # 对图执行稳定的拓扑排序
        stable_topological_sort(graph)
        # 断言图的节点顺序符合预期
        self.assertEqual(list(graph.nodes), [b, a, c])

        # 创建另一个空的 Torch FX 图形对象
        graph = torch.fx.Graph()
        # 创建两个图形占位符节点，并尝试在调用函数 fn1 时将 a 添加到 c 中
        a = graph.placeholder("x")
        b = graph.placeholder("y")
        c = graph.call_function(fn1, (b, a))
        c.append(a)
        # 对图执行稳定的拓扑排序
        stable_topological_sort(graph)
        # 断言图的节点顺序符合预期
        self.assertEqual(list(graph.nodes), [b, a, c])
    # 定义内部函数 `mul_softmax`，用于计算输入张量 `a` 和 `b` 的按维度0进行softmax操作后的乘积
    def test_scaled_softmax(self):
        def mul_softmax(a, b):
            return F.softmax(a * b, dim=0)

        # 定义内部函数 `div_softmax`，用于计算输入张量 `x` 除以 `inv_scale` 后按维度0进行softmax操作
        def div_softmax(x, inv_scale):
            return F.softmax(x / inv_scale, dim=0)

        # 生成一个大小为10x10的随机张量 `x`
        x = torch.randn(10, 10)
        # 设定比例尺度为1e6，并计算其倒数
        scale = 1e6
        inv_scale = 1 / scale
        # 调用外部方法 `self.common` 进行计算，使用 `mul_softmax` 函数，传入参数 `(x, scale)`，期望输出的数量为1，比较深度为3
        self.common(mul_softmax, (x, scale), 1, 3)
        # 调用外部方法 `self.common` 进行计算，使用 `mul_softmax` 函数，传入参数 `(scale, x)`，期望输出的数量为1，比较深度为3
        self.common(mul_softmax, (scale, x), 1, 3)
        # 调用外部方法 `self.common` 进行计算，使用 `div_softmax` 函数，传入参数 `(x, inv_scale)`，期望输出的数量为1，比较深度为3
        self.common(div_softmax, (x, inv_scale), 1, 3)

        # 重新生成比例尺度为大小为10的随机张量，并计算其倒数
        scale = torch.randn(10) * 1e6
        inv_scale = 1 / scale
        # 调用外部方法 `self.common` 进行计算，使用 `mul_softmax` 函数，传入参数 `(x, scale)`，期望输出的数量为1，比较深度为3
        self.common(mul_softmax, (x, scale), 1, 3)
        # 调用外部方法 `self.common` 进行计算，使用 `mul_softmax` 函数，传入参数 `(scale, x)`，期望输出的数量为1，比较深度为3
        self.common(mul_softmax, (scale, x), 1, 3)
        # 调用外部方法 `self.common` 进行计算，使用 `div_softmax` 函数，传入参数 `(x, inv_scale)`，期望输出的数量为1，比较深度为3
        self.common(div_softmax, (x, inv_scale), 1, 3)

        # 重新生成比例尺度为大小为1x10的随机张量，并计算其倒数
        scale = torch.randn(1, 10) * 1e6
        inv_scale = 1 / scale
        # 调用外部方法 `self.common` 进行计算，使用 `mul_softmax` 函数，传入参数 `(x, scale)`，期望输出的数量为1，比较深度为3
        self.common(mul_softmax, (x, scale), 1, 3)
        # 调用外部方法 `self.common` 进行计算，使用 `mul_softmax` 函数，传入参数 `(scale, x)`，期望输出的数量为1，比较深度为3
        self.common(mul_softmax, (scale, x), 1, 3)
        # 调用外部方法 `self.common` 进行计算，使用 `div_softmax` 函数，传入参数 `(x, inv_scale)`，期望输出的数量为1，比较深度为3
        self.common(div_softmax, (x, inv_scale), 1, 3)

        # 测试类型提升匹配
        # 生成一个大小为10x10的随机张量 `x`，并将其数据类型设置为 `torch.bfloat16`
        x = torch.randn(10, 10, dtype=torch.bfloat16)
        # 生成一个大小为10的随机张量 `scale`，乘以1e6，并计算其倒数
        scale = torch.randn(10, dtype=torch.bfloat16) * 1e6
        inv_scale = 1 / scale
        # 调用外部方法 `self.common` 进行计算，使用 `mul_softmax` 函数，传入参数 `(x, scale)`，期望输出的数量为1，比较深度为4，参考类型设置为浮点数
        self.common(mul_softmax, (x, scale), 1, 4, reference_in_float=True)
        # 调用外部方法 `self.common` 进行计算，使用 `mul_softmax` 函数，传入参数 `(scale, x)`，期望输出的数量为1，比较深度为4，参考类型设置为浮点数
        self.common(mul_softmax, (scale, x), 1, 4, reference_in_float=True)
        # 调用外部方法 `self.common` 进行计算，使用 `div_softmax` 函数，传入参数 `(x, inv_scale)`，期望输出的数量为1，比较深度为4，参考类型设置为浮点数
        self.common(div_softmax, (x, inv_scale), 1, 4, reference_in_float=True)

        # 如果 softmax 维度中的比例尺度发生变化，则不匹配
        scale = torch.randn(10, 10)
        # 调用外部方法 `self.common` 进行计算，使用 `mul_softmax` 函数，传入参数 `(x, scale)`，期望输出的数量为0，比较深度为0
        self.common(mul_softmax, (x, scale), 0, 0)
        # 调用外部方法 `self.common` 进行计算，使用 `mul_softmax` 函数，传入参数 `(scale, x)`，期望输出的数量为0，比较深度为0
        self.common(mul_softmax, (scale, x), 0, 0)
        # 调用外部方法 `self.common` 进行计算，使用 `div_softmax` 函数，传入参数 `(x, scale)`，期望输出的数量为0，比较深度为0
        self.common(div_softmax, (x, scale), 0, 0)
# 如果当前脚本作为主程序运行，则执行以下代码块
if __name__ == "__main__":
    # 如果操作系统是 Linux 并且具有 CUDA 支持
    if IS_LINUX and HAS_CUDA:
        # 运行测试函数
        run_tests()
```
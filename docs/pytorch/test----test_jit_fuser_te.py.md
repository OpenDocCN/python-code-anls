# `.\pytorch\test\test_jit_fuser_te.py`

```py
# Owner(s): ["NNC"]

# 导入所需的库和模块
import contextlib  # 上下文管理器相关功能
import math  # 数学函数
import operator  # 运算符模块
import os  # 系统操作模块
import unittest  # 单元测试框架
import warnings  # 警告处理模块
from typing import List  # 类型提示支持

import torch  # PyTorch 深度学习框架
import torch.nn.functional as F  # PyTorch 中的函数API
from torch.testing import FileCheck  # PyTorch 测试工具

# 设置必要的环境变量，确保 GRAPH_EXECUTOR 被正确推断
# 这些设置必须在 `common_utils` 之前被设置，
# 否则 `GRAPH_EXECUTOR` 可能会被错误地运行或跳过某些测试
torch._C._jit_set_profiling_executor(True)
torch._C._get_graph_executor_optimize(True)

from itertools import combinations, permutations, product  # 迭代器操作函数

from textwrap import dedent  # 文本处理模块，用于去除缩进

from jit.test_fuser_common import TestFuserCommon  # 导入测试用例类

from test_jit import (  # 导入多个函数和类
    backward_graph,
    get_lstm_inputs,
    get_milstm_inputs,
    LSTMCellC,
    LSTMCellF,
    LSTMCellS,
    MiLSTMCell,
)

from torch.testing._internal.common_device_type import (  # 导入设备类型相关的函数和类
    instantiate_device_type_tests,
    onlyCPU,
    OpDTypes,
    ops,
)

from torch.testing._internal.common_jit import JitCommonTestCase  # 导入 JIT 测试用例基类

from torch.testing._internal.common_methods_invocations import op_db  # 导入操作数据库相关函数

from torch.testing._internal.common_utils import (  # 导入通用测试工具函数和变量
    enable_profiling_mode_for_profiling_tests,
    GRAPH_EXECUTOR,
    IS_FBCODE,
    ProfilingMode,
    run_tests,
    skipIfTorchDynamo,
    slowTest,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
)

from torch.testing._internal.jit_metaprogramming_utils import create_traced_fn  # JIT 元编程相关工具函数

from torch.testing._internal.jit_utils import (  # JIT 工具函数和测试用例类
    clone_inputs,
    get_traced_sample_variant_pairs,
    JitTestCase,
    NoTracerWarnContextManager,
    RUN_CUDA,
    RUN_CUDA_HALF,
    RUN_CUDA_MULTI_GPU,
    set_fusion_group_inlining,
    TensorExprTestOptions,
    warmup_backward,
)

FUSION_GROUP = "prim::TensorExprGroup"  # 定义融合组的名称
LLVM_ENABLED = torch._C._llvm_enabled()  # 检查 LLVM 是否启用

autograd_check_set = {  # 自动求导检查集合
    "aten::__is__",
    "prim::AutogradAllNonZero",
    "prim::AutogradAllZero",
    "prim::ListConstruct",
}


def strip_profiling_nodes(nodes):
    # 过滤掉指定的分析节点
    profiling_opcodes = {"prim::BailoutTemplate", "prim::BailOut"}
    return [n for n in nodes if n.kind() not in profiling_opcodes]


def warmup_forward(f, *args, profiling_count=2):
    # 执行前向传播的预热，以便进行性能分析
    for i in range(profiling_count):
        results = f(*args)

    return results


@contextlib.contextmanager
def texpr_reductions_enabled():
    # 开启张量表达式的降维优化
    old = torch._C._jit_set_texpr_reductions_enabled(True)
    try:
        yield
    finally:
        torch._C._jit_set_texpr_reductions_enabled(old)


@contextlib.contextmanager
def texpr_enable_strategy(strategy):
    # 设置张量表达式的融合策略
    old = torch._C._jit_set_fusion_strategy(strategy)
    try:
        yield
    finally:
        torch._C._jit_set_fusion_strategy(old)


@contextlib.contextmanager
def inline_fusion_groups():
    # 开启融合组的内联
    old_inlining = torch._C._debug_get_fusion_group_inlining()
    torch._C._debug_set_fusion_group_inlining(True)
    try:
        yield
    finally:
        torch._C._debug_set_fusion_group_inlining(old_inlining)


class TestTEFuser(JitTestCase):
    # TEFuser 测试用例类，继承自 JIT 测试用例基类
    # 设置测试环境，在父类的设置方法基础上进行初始化
    def setUp(self):
        super().setUp()
        # 初始化 TensorExprTestOptions 实例
        self.tensorexpr_options = TensorExprTestOptions()

        # 根据是否有动态形状设置融合策略
        fusion_strategy = [("DYNAMIC", 20)] if self.dynamic_shapes else [("STATIC", 20)]
        # 设置旧的融合策略并保存当前融合策略
        self.old_fusion_strategy = torch._C._jit_set_fusion_strategy(fusion_strategy)

        # 根据 CUDA 是否可用设置设备列表
        self.devices = ["cpu"] if not torch.cuda.is_available() else ["cpu", "cuda"]
        # 设置整数数据类型列表
        self.int_dtypes = [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
        ]
        # 设置浮点数据类型列表
        self.fp_dtypes = [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ]
        # 组合整数和浮点数据类型列表
        self.dtypes = self.int_dtypes + self.fp_dtypes

    # 清理测试环境，在父类的清理方法基础上进行清理
    def tearDown(self):
        # 恢复设置的 TensorExprTestOptions
        self.tensorexpr_options.restore()
        # 恢复旧的融合策略
        torch._C._jit_set_fusion_strategy(self.old_fusion_strategy)
        super().tearDown()

    # 断言图中所有节点都已被融合，除非指定排除的节点
    def assertAllFused(self, graph, except_for=None):
        except_for = except_for if except_for is not None else set()
        # 待处理的守卫节点类型列表
        guards = (
            "prim::TypeCheck",
            "prim::RequiresGradCheck",
            "prim::TensorExprDynamicGuard",
        )
        guard_found = False

        # 判断是否为自动求导守卫节点
        def autodiff_guard(node):
            if node.kind() != "aten::all":
                return False
            inps = list(node.inputs())
            if len(inps) != 1 or inps[0].node().kind() != "prim::ListConstruct":
                return False
            li_inps = list(inps[0].node().inputs())
            for li_inp in li_inps:
                if li_inp.node().kind() in (
                    "prim::AutogradAllNonZero",
                    "prim::AutogradAllZero",
                ):
                    return True
            return False

        # 判断节点是否为守卫节点
        def is_guard(node):
            return node.kind() in guards or autodiff_guard(node)

        # 遍历图中所有节点，进行融合节点的检查
        for node in graph.block().nodes():
            if node.kind() == "prim::Constant":
                continue
            # 如果找到守卫节点，确保只找到一个
            if is_guard(node):
                self.assertFalse(guard_found)
                guard_found = True
                continue
            # 如果节点在排除列表中，则跳过
            if node.kind() in except_for:
                continue
            # 如果节点为条件语句，则前面应该有守卫节点
            if node.kind() == "prim::If":
                self.assertTrue(is_guard(node.prev()))
                continue
            # 如果找到未预期的节点类型，断言失败并输出节点类型信息
            self.assertTrue(False, "Found unexpected node:" + node.kind())

        # 确保至少找到一个守卫节点
        self.assertTrue(guard_found)

    # 断言最后一个执行的优化图中所有节点都已被融合
    def assertLastGraphAllFused(self):
        self.assertAllFused(torch.jit.last_executed_optimized_graph())

    # 查找图中的所有融合组
    def findFusionGroups(self, graph):
        result = []
        # 遍历图中所有节点，查找融合组
        for n in graph.nodes():
            if n.kind() == FUSION_GROUP:
                result.append(n.g("Subgraph"))
                continue
            for block in n.blocks():
                result += self.findFusionGroups(block)
        return result
    def test_typecheck(self):
        # 创建一个形状为 [1] 的张量，所有元素为1
        a = torch.ones(1)

        # 定义一个融合内核函数，计算 (a + b) * 2.0
        def fused_kernel(a, b):
            return (a + b) * 2.0

        # 对融合内核函数进行脚本化
        scripted = self.checkScript(fused_kernel, (a, a))

        # 获取张量 a, a 对应的计算图
        graph = scripted.graph_for(a, a)

        # 检查是否成功进行了融合
        fusion_groups = self.findFusionGroups(graph)
        self.assertEqual(len(fusion_groups), 1)

        # 现在使用一个更大的张量 (大小为2)
        a = torch.ones(2)

        # 如果不触发重新编译，我们仍会创建一个大小为1的张量
        # 如果类型检查失败，我们会悄无声息地计算错误的结果
        self.assertEqual(scripted(a, a), fused_kernel(a, a))

    def test_sum_simple(self):
        # 定义一个函数 func，计算输入张量 x 的平方和
        def func(x):
            x2 = x * x
            return x2.sum()

        # 启用表达式缩减环境
        with texpr_reductions_enabled():
            # 创建一个浮点类型的张量 a，包含从0到14的数列，分配在CPU上
            a = torch.tensor(list(range(0, 15)), dtype=torch.float, device="cpu")

            # 将张量 a 重新形状为 5x3 的矩阵
            a = a.reshape(5, 3)

            # 对函数 func 进行脚本化
            scripted = self.checkScript(func, (a,))

            # 断言最后的计算图中所有操作都被融合了
            self.assertLastGraphAllFused()

    def test_nop(self):
        # 空函数，无操作

    def test_sum_dim(self):
        # 定义一个函数 func，计算输入张量 x 按维度0的和并乘以2
        def func(x):
            return x.sum((0,)) * 2

        # 定义一个函数 func_neg，计算输入张量 x 按维度-2的和并乘以2
        def func_neg(x):
            return x.sum((-2,)) * 2

        # 启用表达式缩减环境
        with texpr_reductions_enabled():
            # 创建一个浮点类型的张量 a，包含从0到14的数列，分配在CPU上
            a = torch.tensor(list(range(0, 15)), dtype=torch.float, device="cpu")

            # 将张量 a 重新形状为 5x3 的矩阵
            a = a.reshape(5, 3)

            # 对函数 func 进行脚本化，并断言最后的计算图中所有操作都被融合了
            scripted = self.checkScript(func, (a,))
            self.assertLastGraphAllFused()

            # 对函数 func_neg 进行脚本化，并断言最后的计算图中所有操作都被融合了
            scripted = self.checkScript(func_neg, (a,))
            self.assertLastGraphAllFused()

    def test_sum_keepdim_cast(self):
        # 定义一个函数 func，计算输入张量 x 按维度0的和并保持维度，数据类型转换为双精度，并乘以2
        def func(x):
            return x.sum((0,), keepdim=True, dtype=torch.double) * 2

        # 启用表达式缩减环境
        with texpr_reductions_enabled():
            # 创建一个浮点类型的张量 a，包含从0到14的数列，分配在CPU上
            a = torch.tensor(list(range(0, 15)), dtype=torch.float, device="cpu")

            # 将张量 a 重新形状为 5x3 的矩阵

            # 对函数 func 进行脚本化，并断言最后的计算图中所有操作都被融合了
            self.checkScript(func, (a,))
            self.assertLastGraphAllFused()

    def test_abs(self):
        # 遍历所有设备
        for device in self.devices:

            # 定义一个函数 func，计算输入张量 x 的绝对值并乘以2
            def func(x):
                return x.abs() * 2

            # 创建一个随机张量 a，形状为 [5]，分配在指定设备上
            a = torch.randn(5, device=device)

            # 对函数 func 进行脚本化，并断言最后的计算图中所有操作都被融合了
            scripted = self.checkScript(func, (a,))
            self.assertLastGraphAllFused()

    def test_unsqueeze_size_calculation(self):
        # 遍历所有设备
        for device in self.devices:

            # 定义一个函数 foo，接受两个输入 b 和 d，对 d 进行在维度1上的unsqueeze操作
            def foo(b, d):
                x = d.unsqueeze(1)
                y = x * 42.0
                z = b + y
                r = z / 42.0
                return r

            # 创建两个张量作为输入
            inputs = (
                torch.rand(20, 28, device=device, requires_grad=True),
                torch.rand(20, device=device),
            )

            # 对函数 foo 进行脚本化，并断言最后的计算图中所有操作都被融合了
            scripted = self.checkScript(foo, inputs)
            self.assertAllFused(scripted.graph_for(*inputs))
    def test_zero_element_tensors(self):
        # 针对不同设备进行测试
        for device in self.devices:

            def decode(sin_t, cos_t):
                # 计算反正切值，返回角度 theta
                theta = torch.atan2(sin_t.float(), cos_t.float())
                return theta

            # 创建零元素张量 sin 和 cos
            sin = torch.zeros(0, device=device)
            cos = torch.zeros(0, device=device)
            inputs = [sin, cos]
            # 调用自定义函数检查编译后的脚本
            ge = self.checkScript(decode, inputs)

    def test_arg_configurations_smoke(self):
        if self.dynamic_shapes:
            # 如果存在动态形状，则跳过测试
            self.skipTest("TODO: chunk dynamic shapes")

        # 用于验证连续和非连续参数是否使用相同的内核的简单测试
        # TODO: 添加可选的调试计数器到融合器，以验证我们确实可以区分不同的配置
        for device in self.devices:

            def f(x, y):
                # 按照维度 1 将 x + y 的结果分块成 z1 和 z2
                z1, z2 = (x + y).chunk(2, dim=1)
                return z1 * z2

            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)
            traced_f = torch.jit.trace(
                f,
                (
                    x,
                    y,
                ),
            )
            # 断言两种不同的调用方式结果相等
            self.assertEqual(traced_f(x.t().contiguous(), y), traced_f(x.t(), y))

    def test_broadcast(self):
        # 针对不同设备进行测试
        for device in self.devices:

            def scaleshift(x, scale, shift):
                # 对输入 x 进行尺度缩放和位移操作
                return x * scale + shift

            # 创建输入列表，包括张量 x、尺度 scale 和位移 shift
            inputs = [
                torch.randn(4, 4, dtype=torch.float, device=device),
                torch.randn(4, dtype=torch.float, device=device),
                torch.randn(4, dtype=torch.float, device=device),
            ]
            # 调用自定义函数检查编译后的脚本
            self.checkScript(scaleshift, inputs)

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_HALF, "no half support")
    @unittest.skipIf(
        GRAPH_EXECUTOR != ProfilingMode.LEGACY, "no half support with profiling on"
    )
    def test_cuda_half(self):
        # 在 GPU 上创建随机张量 x 和 y，数据类型为半精度（half）
        x = torch.randn(4, 4, dtype=torch.half, device="cuda")
        y = torch.randn(4, 4, dtype=torch.half, device="cuda")

        # 准备待测试的函数列表
        funcs = [self.fn_test_comparison_gt_lt, self.fn_test_relu, self.fn_test_exp]

        # 注意：非融合的输入必须为浮点数，以防止精度损失
        inputs = (x.float(), y.float())
        fusion_inputs = (x, y)
        for fn in funcs:
            # 复制输入张量并启用梯度跟踪
            local_inputs = [t.clone().requires_grad_() for t in inputs]
            local_fusion_inputs = [t.clone().requires_grad_() for t in fusion_inputs]

            # 验证函数的输出
            fusion = torch.jit.trace(fn, local_fusion_inputs, check_trace=False)
            outputs = fn(*local_inputs)
            fusion_outputs = fusion(*local_fusion_inputs)
            # 将输出转换为半精度张量并进行断言验证
            outputs_half = [t.half() for t in outputs]
            self.assertEqual(outputs_half, fusion_outputs)

            # 验证梯度
            for output, fusion_output in zip(outputs_half, fusion_outputs):
                # 计算输出的浮点数形式的梯度
                grads = torch.autograd.grad(
                    output.float().sum(),
                    local_inputs,
                    allow_unused=True,
                    retain_graph=True,
                )
                # 计算融合输出的梯度
                fusion_grads = torch.autograd.grad(
                    fusion_output.sum(),
                    local_fusion_inputs,
                    allow_unused=True,
                    retain_graph=True,
                )
                # 将梯度转换为半精度张量并进行断言验证
                grads_half = [t.half() for t in grads]
                self.assertEqual(grads_half, fusion_grads)

    def test_checks_cat_inputs(self):
        # 启用融合组内联以测试
        with set_fusion_group_inlining(True):
            for device in self.devices:
                # 定义一个函数，将输入张量 x 和 y 进行拼接操作
                def f(x, y):
                    return torch.cat([x + 2 * x + x**2, y + 4 * y + y**3], dim=0)

                # 注意：y 可以广播到 x，但是 f(x, y) 的输出应该是 3x4 的形状，而不是 4x4
                x = torch.randn(2, 4, dtype=torch.float, device=device)
                y = torch.randn(1, 4, dtype=torch.float, device=device)

                # 对脚本化版本进行检查，并断言其输出形状为 (3, 4)
                scripted = self.checkScript(f, (x, y))
                self.assertEqual(scripted(x, y).shape, (3, 4))
                # 断言脚本化图中所有节点都已融合
                self.assertAllFused(scripted.graph_for(x, y))

    def test_chunk(self):
        # 如果支持动态形状，则跳过测试
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        for device in self.devices:
            # 定义一个函数，对输入张量 x 进行分块操作，并返回计算结果
            def fn(x):
                a, b, c = x.chunk(3, 1)
                return a * b + c

            inputs = [torch.randn(10, 6, dtype=torch.float, device=device)]

            # 对函数进行脚本化，并断言最后生成的图中所有节点都已融合
            self.checkScript(fn, inputs)
            self.assertLastGraphAllFused()
    def test_chunk_correctness(self):
        # 如果存在动态形状，跳过测试
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        # 对每个设备执行测试
        for device in self.devices:

            # 定义在第0维度上按4分块的函数
            def chunk_4_0(x):
                x0, x1, x2, x3 = x.chunk(4, 0)
                return x0 + x1 + x2 + x3

            # 定义在第1维度上按4分块的函数
            def chunk_4_1(x):
                x0, x1, x2, x3 = x.chunk(4, 1)
                return x0 + x1 + x2 + x3

            # 定义在最后一维度上按4分块的函数
            def chunk_4_last(x):
                x0, x1, x2, x3 = x.chunk(4, 2)
                return x0 + x1 + x2 + x3

            # 待测试的函数列表
            fns = [chunk_4_0, chunk_4_1, chunk_4_last]

            # 待测试的张量列表
            tensors = [
                # splitSize = 1
                torch.randn(4, 4, 4, dtype=torch.float, device=device),
                # contiguous case
                torch.randn(12, 8, 16, dtype=torch.float, device=device),
                # non-contiguous case
                torch.randn(12, 8, 16, dtype=torch.float, device=device).transpose(
                    1, 2
                ),
            ]

            # 遍历每个张量和每个函数进行测试
            for tensor in tensors:
                for fn in fns:
                    self.checkScript(fn, [tensor])
                    self.assertLastGraphAllFused()

    def test_chunk_distributes(self):
        # 如果存在动态形状，跳过测试
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        # 对每个设备执行测试
        for device in self.devices:

            # 定义函数 f，对输入张量 x 和 y 执行相加，并在第1维度上按2分块
            def f(x, y):
                z1, z2 = (x + y).chunk(2, dim=1)
                return z1 * z2

            # 创建两个随机张量 x 和 y
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)

            # 检查函数 f 的跟踪版本并获取计算图
            ge = self.checkTrace(f, (x, y))
            graph = ge.graph_for(x, y)

            # XXX: 旧的 fuser 执行广播操作，但新的 fuser 不执行
            FileCheck().check("with " + FUSION_GROUP + "_").check_count(
                "ConstantChunk", 1, exactly=True
            ).run(str(graph))

    def test_chunk_motion_deduplicates_inputs(self):
        # 如果存在动态形状，跳过测试
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        # 对每个设备执行测试
        for device in self.devices:

            # 定义函数 func1，对输入张量 x 执行平方，然后在第1维度上按2分块
            def func1(x):
                z = x * x
                z0, z1 = z.chunk(2)
                return z0 * z1

            # 定义函数 func2，对输入张量 x 执行立方，然后在第1维度上按2分块
            def func2(x):
                z = x * x * x
                z0, z1 = z.chunk(2)
                return z0 * z1

            # 输入张量列表
            inputs = [
                torch.tensor([1.1, 1.2], device=device, dtype=torch.float),
            ]

            # 分别对 func1 和 func2 进行脚本模式的测试
            for func in [func1, func2]:
                self.checkScript(func, inputs)
                self.assertLastGraphAllFused()
    def test_chunk_multiple(self):
        # 如果支持动态形状，则跳过该测试
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        # 遍历每个设备执行测试
        for device in self.devices:
            # 定义一个函数 fn，接受四个参数 s, x, y, z
            # 这里故意打乱参数顺序，用于测试融合编译器是否正确添加额外参数
            def fn(s, x, y, z):
                # 对张量 z 进行分块操作，分成两部分 z1, z2
                z1, z2 = z.chunk(2, 2)
                # 对张量 x 进行分块操作，分成三部分 x1, x2, x3
                x1, x2, x3 = x.chunk(3, 1)
                # 对张量 y 进行分块操作，分成两部分 y1, y2
                y1, y2 = y.chunk(2, 0)
                # 返回所有分块的张量之和
                return s + x1 + x2 + x3 + y1 + y2 + z1 + z2

            # 输入张量列表
            inputs = [
                torch.randn(5, 2, 3, dtype=torch.float, device=device),
                torch.randn(5, 6, 3, dtype=torch.float, device=device),
                torch.randn(10, 2, 3, dtype=torch.float, device=device),
                torch.randn(5, 2, 6, dtype=torch.float, device=device),
            ]

            # 对函数 fn 进行脚本化并进行检查
            ge = self.checkScript(fn, inputs)
            # 断言所有操作是否被融合
            self.assertAllFused(ge.graph_for(*inputs))

    def test_minmax(self):
        # 遍历每个设备执行测试
        for device in self.devices:

            # 定义一个求最大值的函数 tmax，接受两个参数 a, b
            def tmax(a, b):
                return torch.max(2 * a, b)

            # 定义一个求最小值的函数 tmin，接受两个参数 a, b
            def tmin(a, b):
                return torch.min(2 * a, b)

            # 创建随机张量 a 和 b，以及一个包含 NaN 的张量 nan
            a = torch.randn(4, 4, dtype=torch.float)
            b = torch.randn(4, 4, dtype=torch.float)
            nan = torch.tensor(float("nan"), dtype=torch.float)

            # 遍历函数 tmax 和 tmin 以及它们的输入组合进行测试
            for f, inputs, device in product(
                (tmax, tmin), ([a, b], [a, nan], [b, nan]), self.devices
            ):
                # 将输入张量列表中的张量转移到指定设备
                inputs = [t.to(device) for t in inputs]
                # 对函数 f 进行脚本化并进行检查
                s = self.checkScript(f, inputs)
                # 断言所有操作是否被融合
                self.assertAllFused(s.graph_for(*inputs))
    # 定义测试函数 test_clamp，用于测试 torch.clamp 函数在不同条件下的行为
    def test_clamp(self):
        # 遍历设备列表
        for device in self.devices:

            # 定义一个将两个参数相加后进行 clamp 操作的函数 func2
            def func2(a, b):
                return torch.clamp(a + b, min=0, max=2)

            # 定义一个将两个参数相加后进行 clamp 操作的函数 funcInf，最大值为正无穷
            def funcInf(a, b):
                return torch.clamp(a + b, min=0, max=float("inf"))

            # 定义一个将两个参数相加后进行 clamp 操作的函数 funcNegInf，最小值为负无穷
            def funcNegInf(a, b):
                return torch.clamp(a + b, min=float("-inf"), max=0)

            # 定义一个将两个参数相加后进行 clamp 操作的函数 funcOptMin，只设定最大值
            def funcOptMin(a, b):
                return torch.clamp(a + b, max=2)

            # 定义一个将两个参数相加后进行 clamp 操作的函数 funcOptMax，只设定最小值
            def funcOptMax(a, b):
                return torch.clamp(a + b, min=0)

            # 创建一个随机张量 a，用于测试
            a = torch.randn(4, 4, dtype=torch.float, device=device, requires_grad=True)
            # 创建一个随机张量 b，用于测试
            b = torch.randn(4, 4, dtype=torch.float, device=device)
            # 创建一个包含 NaN 值的张量 nan，用于测试
            nan = torch.tensor(float("nan"), dtype=torch.float, device=device)

            # 将所有的 clamp 函数放入 funcs 元组中
            funcs = (func2, funcInf, funcNegInf, funcOptMin, funcOptMax)
            # 遍历 funcs 中的每个函数和它们的输入组合
            for f, inputs in product(funcs, [[a, b], [a, nan]]):
                inp1, inp2 = inputs
                # 对每个函数进行脚本化检查，启用性能分析模式
                s = self.checkScript(f, (inp1, inp2), profiling=ProfilingMode.PROFILING)
                # 断言脚本中的所有操作均被融合
                self.assertAllFused(
                    s.graph_for(inp1, inp2),
                    except_for={"aten::size", "aten::_size_if_not_equal"},
                )
                # 执行脚本化的函数调用
                c = s(inp1, inp2)
                # 在性能分析测试中，启用前向传播
                with enable_profiling_mode_for_profiling_tests():
                    warmup_backward(c.sum())
                # 获取反向传播的计算图
                graph = backward_graph(s)
                # 断言反向传播的计算图中的所有操作均被融合
                self.assertAllFused(
                    graph,
                    except_for={"aten::Float", "aten::_grad_sum_to_size"}.union(
                        autograd_check_set
                    ),
                )

    # 定义测试函数 test_clamp_double，测试 torch.clamp 在双精度张量上的行为
    def test_clamp_double(self):
        # 遍历设备列表
        for device in self.devices:

            # 定义一个在双精度张量上进行 clamp 操作的函数 clamp_double
            def clamp_double(x, eta: float):
                return 1 - x.clamp(eta, 1 - eta)

            # 创建一个双精度张量 x，用于测试
            x = torch.tensor([1.0, 1.0], dtype=torch.double, device=device)
            # 设定 eta 参数
            eta = 1e-9
            # 对 clamp_double 函数进行脚本化检查，启用性能分析模式，并设定数值容差
            s = self.checkScript(
                clamp_double,
                (x, eta),
                profiling=ProfilingMode.PROFILING,
                atol=1e-10,
                rtol=1e-5,
            )
            # 断言脚本中的所有操作均被融合，除了减法操作
            self.assertAllFused(s.graph_for(x, eta), except_for={"aten::sub"})

    # 定义测试函数 test_clamp_int，测试 torch.clamp 在整数张量上的行为
    def test_clamp_int(self):
        # 遍历设备列表
        for device in self.devices:

            # 定义一个在整数张量上进行 clamp 操作的函数 clamp_int
            def clamp_int(x, eta: int):
                return x.clamp(0, eta)

            # 创建一个整数张量 x，用于测试
            x = torch.tensor([1, 1], device=device)
            # 设定 eta 参数
            eta = 1 << 32
            # 对 clamp_int 函数进行脚本化检查，启用性能分析模式
            s = self.checkScript(clamp_int, (x, eta), profiling=ProfilingMode.PROFILING)
            # 断言脚本中的所有操作均被融合
            self.assertAllFused(s.graph_for(x, eta))
    def test_add_bool(self):
        # 定义一个简单的加法函数 f(x, y, z)
        def f(x, y, z):
            return x + y + z

        # 遍历设备和尺寸组合
        sizes = [(1,), (2,), (4, 4)]
        for device, size in product(self.devices, sizes):
            # 创建随机的布尔张量 x, y, z，指定设备和数据类型为布尔型
            x = torch.randint(0, 2, size, dtype=torch.bool, device=device)
            y = torch.randint(0, 2, size, dtype=torch.bool, device=device)
            z = torch.randint(0, 2, size, dtype=torch.bool, device=device)
            # 对函数 f 进行跟踪检查，不要求输入张量梯度
            ge = self.checkTrace(f, (x, y, z), inputs_require_grads=False)
            # 断言所有操作已融合在一个图中
            self.assertAllFused(ge.graph_for(x, y, z))

    def test_mul_bool(self):
        # 定义一个简单的乘法函数 f(x, y, z)
        def f(x, y, z):
            return x * y * z

        # 遍历设备
        for device in self.devices:
            # 创建随机的布尔张量 x, y, z，指定设备和数据类型为布尔型
            x = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)
            y = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)
            z = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)
            # 对函数 f 进行跟踪检查，不要求输入张量梯度
            ge = self.checkTrace(f, (x, y, z), inputs_require_grads=False)
            # 断言所有操作已融合在一个图中
            self.assertAllFused(ge.graph_for(x, y, z))

    def test_div_bool(self):
        # 定义一个简单的除法函数 f(x, y, z)
        def f(x, y, z):
            return (x + y) / z

        # 遍历设备
        for device in self.devices:
            # 创建随机的布尔张量 x, y，并使用全 1 张量作为 z，指定设备和数据类型为布尔型
            x = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)
            y = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)
            z = torch.ones_like(x, dtype=torch.bool, device=device)
            # 对函数 f 进行跟踪检查，不要求输入张量梯度
            ge = self.checkTrace(f, (x, y, z), inputs_require_grads=False)
            # 断言所有操作已融合在一个图中
            self.assertAllFused(ge.graph_for(x, y, z))

    def test_bitwise_ops(self):
        # 定义一个应用函数，接受一个二元操作并返回一个函数 fn(x, y, z)
        def apply(fn):
            return lambda x, y, z: fn(fn(x, y), z)

        # 定义要应用的二进制操作函数列表
        binary_ops = [
            operator.__and__,
            operator.__or__,
            operator.__xor__,
            operator.__lshift__,
            operator.__rshift__,
        ]
        # 获取设备列表
        devices = self.devices
        # 遍历整数数据类型、二进制操作和设备的组合
        for dtype, op, device in product(self.int_dtypes, binary_ops, devices):
            try:
                # 创建数据张量 x, y, z
                x = self.data_for(dtype, device)
                y = self.data_for(dtype, device)
                z = self.data_for(dtype, device)
                # 应用二元操作函数 fn = apply(op)
                fn = apply(op)
                # 计算参考结果 ref = fn(x, y, z)
                ref = fn(x, y, z)
            except Exception:
                # 如果 eager 模式不支持某种数据类型/操作/设备组合，跳过
                continue
            try:
                # 对函数 fn 进行跟踪检查，获取跟踪的张量
                t = torch.jit.trace(fn, (x, y, z))
                # 断言跟踪后的结果与参考结果一致
                self.assertEqual(ref, t(x, y, z))
                # 断言所有操作已融合在一个图中
                self.assertAllFused(t.graph_for(x, y, z))
            except Exception as e:
                # 如果出现异常，抛出运行时错误
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device])
                ) from e
    # 定义一个内部函数 apply，接受一个函数 fn，并返回一个新函数，这个新函数接受三个参数 x, y, z，并将 fn 作用于它们的结果
    def apply(fn):
        return lambda x, y, z: fn(fn(x, y), z)

    # 定义了两个 Torch 操作函数的列表：最小值和最大值
    binary_ops = [torch.min, torch.max]
    # 获取测试中指定的设备列表
    devices = self.devices
    # 对于整数类型的数据、上述两种操作函数和设备的所有组合进行迭代测试
    for dtype, op, device in product(self.int_dtypes, binary_ops, devices):
        try:
            # 使用 self.data_for 函数生成对应 dtype 和 device 的数据 x, y, z
            x = self.data_for(dtype, device)
            y = self.data_for(dtype, device)
            z = self.data_for(dtype, device)
            # 调用 apply 函数，传入 op 函数，生成一个新的函数 fn
            fn = apply(op)
            # 计算参考结果 ref，即 fn(x, y, z) 的值
            ref = fn(x, y, z)
        except Exception:
            # 如果在生成数据或应用函数时发生异常，捕获异常并继续下一轮迭代
            # 如果 eager 模式不支持特定的 dtype/op/device 组合，则 fuser 也不会支持，因此忽略此类异常
            continue
        try:
            # 使用 torch.jit.trace 对 fn 进行追踪，以便后续比较
            t = torch.jit.trace(fn, (x, y, z))
            # 断言追踪的结果与参考结果一致
            self.assertEqual(ref, t(x, y, z))
            # 断言所有操作都被融合（fused）到一个图中
            self.assertAllFused(t.graph_for(x, y, z))
        except Exception as e:
            # 如果追踪过程中出现异常，则抛出自定义的 RuntimeError 异常，包含详细错误信息
            raise RuntimeError(
                " ".join(["Failed:", str(dtype), op.__name__, device])
            ) from e

    # 定义一个内部函数 f，接受两个参数 x, y，并进行比较操作
    def f(x, y):
        # 生成一个掩码 mask，标记 x 中等于 0 的位置
        mask = (x == 0).type_as(x)
        # 计算 z，使用掩码对 x 和 y 进行加权求和
        z = x * mask + y
        # 更新掩码，标记 x 中不等于 0 的位置
        mask = (x != 0).type_as(x)
        # 再次计算 z，使用新掩码对 z 进行加权求和
        z = z * mask + y
        # 返回结果 z
        return z

    # 生成随机数据 x, y，并对函数 f 进行追踪
    x = torch.randn(4, 4, dtype=torch.float, device=device)
    y = torch.randn(4, 4, dtype=torch.float, device=device)
    ge = self.checkTrace(f, (x, y))
    # 断言所有操作都被融合到一个图中
    self.assertAllFused(ge.graph_for(x, y))

    # 静态方法，定义一个函数 fn_test_comparison_gt_lt，接受两个参数 x, y，并进行比较操作
    @staticmethod
    def fn_test_comparison_gt_lt(x, y):
        # 生成一个掩码 mask，标记 x 中大于 0 的位置
        mask = (x > 0).type_as(x)
        # 计算 z，使用掩码对 x 和 y 进行加权求和
        z = x * mask + y
        # 更新掩码，标记 x 中小于 0 的位置
        mask = (x < 0).type_as(x)
        # 再次计算 z，使用新掩码对 z 进行加权求和
        z = z * mask + y
        # 返回结果 z
        return z

    # 生成随机数据 x, y，并对静态方法 fn_test_comparison_gt_lt 进行追踪
    x = torch.randn(4, 4, dtype=torch.float, device=device)
    y = torch.randn(4, 4, dtype=torch.float, device=device)
    ge = self.checkTrace(self.fn_test_comparison_gt_lt, (x, y))
    # 断言所有操作都被融合到一个图中
    self.assertAllFused(ge.graph_for(x, y))

    # 定义一个内部函数 f，接受两个参数 x, y，并进行比较操作
    def f(x, y):
        # 生成一个掩码 mask，标记 x 中大于等于 0 的位置
        mask = (x >= 0).type_as(x)
        # 计算 z，使用掩码对 x 和 y 进行加权求和
        z = x * mask + y
        # 更新掩码，标记 x 中小于等于 0 的位置
        mask = (x <= 0).type_as(x)
        # 再次计算 z，使用新掩码对 z 进行加权求和
        z = z * mask + y
        # 返回结果 z
        return z

    # 生成随机数据 x, y，并对函数 f 进行追踪
    x = torch.randn(4, 4, dtype=torch.float, device=device)
    y = torch.randn(4, 4, dtype=torch.float, device=device)
    ge = self.checkTrace(f, (x, y))
    # 断言所有操作都被融合到一个图中
    self.assertAllFused(ge.graph_for(x, y))
    # 将 x, y 设置为需要梯度计算的张量
    x.requires_grad_(True)
    y.requires_grad_(True)
    # 断言所有操作都被融合到一个图中，但排除特定的操作名称
    self.assertAllFused(
        ge.graph_for(x, y),
        except_for=(
            "aten::size",
            "prim::BroadcastSizes",
            "aten::_size_if_not_equal",
        ),
    )
    # 定义一个测试方法，测试 torch.addcmul() 的行为
    def test_addcmul(self):
        # 遍历设备列表中的每一个设备
        for device in self.devices:
            # 生成一个随机张量 t，形状为 (1, 4)，数据类型为 float，放置在当前设备上
            t = torch.randn(1, 4, dtype=torch.float, device=device)
            # 生成一个随机张量 t1，形状为 (4, 1)，数据类型为 float，放置在当前设备上
            t1 = torch.randn(4, 1, dtype=torch.float, device=device)
            # 生成一个随机张量 t2，形状为 (1, 4)，数据类型为 float，放置在当前设备上
            t2 = torch.randn(1, 4, dtype=torch.float, device=device)

            # 定义一个内部函数 foo，接受参数 t, t1, t2，执行 torch.addcmul 操作
            def foo(t, t1, t2):
                return t.addcmul(t + 1, t2, value=0.1)

            # 对 foo 函数进行追踪，并检查结果
            ge = self.checkTrace(foo, (t, t1, t2), allow_unused=True)
            # 获取张量 t, t1, t2 所对应的计算图
            graph = ge.graph_for(t, t1, t2)
            # 查找图中的融合组
            fusion_groups = self.findFusionGroups(graph)
            # 断言只有一个融合组
            self.assertEqual(len(fusion_groups), 1)
            # 运行 FileCheck 检查融合组中是否包含 "aten::add" 和 "aten::addcmul"
            FileCheck().check("aten::add(").check("aten::addcmul(").run(
                str(fusion_groups[0])
            )

    # TODO: 在这里存在 CUDA 内存泄漏，因为追踪的计算图持有一个常量化的张量。
    # 由于 Python 全局 CompilationUnit 在进程结束前一直存活，这部分内存实际上泄漏了。
    # 从测试中移除了 `_cuda` 后缀以禁用泄漏检查。
    # 如果这是一个真实的问题，我们需要重新审视 Python 中 Torchscript 函数的生命周期。
    # 定义一个测试方法，测试 torch.lerp() 的行为
    def test_lerp(self):
        # 遍历设备列表中的每一个设备
        for device in self.devices:
            # 生成一个随机张量 start，形状为 (4, 1)，数据类型为 float，放置在当前设备上
            start = torch.randn(4, 1, dtype=torch.float, device=device)
            # 生成一个随机张量 end，形状为 (1, 4)，数据类型为 float，放置在当前设备上
            end = torch.randn(1, 4, dtype=torch.float, device=device)
            # 生成一个标量张量 weight，数值为 0.5，数据类型为 float，放置在当前设备上
            weight = torch.tensor(0.5, dtype=torch.float, device=device)

            # 定义一个内部函数，使用标量权重进行插值
            def foo_weight_scalar(start, end):
                return torch.lerp(start + 1, end, 0.5)

            # 定义一个内部函数，使用张量权重进行插值
            def foo_weight_tensor(start, end):
                return torch.lerp(start + 1, end, weight)

            # 对 foo_weight_scalar 函数进行追踪，并检查结果
            ge_weight_scalar = self.checkTrace(foo_weight_scalar, (start, end))
            # 获取张量 start, end 所对应的计算图
            graph = ge_weight_scalar.graph_for(start, end)
            # 断言所有操作都已融合
            self.assertAllFused(graph)

            # TODO: 当 TE 启用对标量张量的支持时取消注释
            # 对 foo_weight_tensor 函数进行追踪，并检查结果
            # ge_weight_tensor = self.checkTrace(foo_weight_tensor, (start, end))
            # 获取张量 start, end 所对应的计算图
            # graph = ge_weight_tensor.graph_for(start, end)
            # 断言所有操作都已融合

    # 定义一个测试方法，测试 torch.cat() 的行为
    def test_concat(self):
        # 设置融合组内联以避免单个连接节点错误
        with set_fusion_group_inlining(True):
            # 遍历设备列表中的每一个设备
            for device in self.devices:
                # 生成一个随机张量 hx，形状为 (3, 20)，数据类型为 float，放置在当前设备上
                hx = torch.randn(3, 20, dtype=torch.float, device=device)
                # 生成一个随机张量 cx，形状为 (3, 20)，数据类型为 float，放置在当前设备上
                cx = torch.randn(3, 20, dtype=torch.float, device=device)

                # 定义一个内部函数 foo，对 hx 和 cx 执行 torch.cat() 操作
                def foo(hx, cx):
                    return torch.cat((hx + cx, hx * cx))

                # 对 foo 函数进行追踪，并检查结果
                ge = self.checkTrace(foo, (hx, cx))
                # 获取张量 hx, cx 所对应的计算图
                graph = ge.graph_for(hx, cx)
                # 断言所有操作都已融合
                self.assertAllFused(graph)
                # XXX: TE fuser 可以处理融合组中的连接操作。
                # FileCheck().check("FusedConcat").check_next("return").run(str(graph))
    # 测试函数：验证只有在尺寸计算中使用的输出
    def test_remove_output_used_only_in_size(self):
        for device in self.devices:

            # 定义内部测试函数：将两个张量相加，并在此基础上进行操作
            def test_fuse(a, b):
                c = a + b  # 计算两个张量的和
                d = c + b  # 计算和再加上第二个张量
                return d  # 返回结果张量

            # 对测试函数进行脚本化
            scripted_f = torch.jit.script(test_fuse)
            # 创建张量 x 和 y，都为1，并在给定设备上启用梯度跟踪
            x = torch.ones(1, requires_grad=True, device=device)
            y = torch.ones(1, requires_grad=True, device=device)
            # 运行预热前向传播
            warmup_forward(scripted_f, x, y, profiling_count=3)
            # 获取生成的图形对象
            g = scripted_f.graph_for(x, y)
            # 查找所有具有 "prim::DifferentiableGraph" 类型的节点
            diff_nodes = g.findAllNodes("prim::DifferentiableGraph")
            # 断言不同节点的数量为1
            self.assertEqual(len(diff_nodes), 1)
            # 从不同图中提取子图 g
            g = diff_nodes[0].g("Subgraph")
            # 获取所有节点为 "prim::If" 类型的列表
            if_nodes = [n for n in g.nodes() if n.kind() == "prim::If"]
            # 断言 if 节点的数量为1
            self.assertEqual(len(if_nodes), 1)

            # 断言 if 节点及其内部的融合组仅有一个输出
            self.assertEqual(len(list(if_nodes[0].outputs())), 1)

    # 测试函数：验证 prim::FusedConcat 的输出不会成为 FusionGroup 内部任何节点的输入
    def test_concat_invariant(self):
        for device in self.devices:
            # 不变性：prim::FusedConcat 的输出不会成为 FusionGroup 内部任何节点的输入
            def fn(x, y, z):
                x1 = x + y  # 计算张量 x 和 y 的和
                y1 = x - y  # 计算张量 x 和 y 的差
                w = torch.cat([x1, y1])  # 进行张量拼接
                return w + z  # 返回拼接后的张量加上 z

            # 创建具有随机值的张量 x、y、z，并在指定设备上检查跟踪
            x = torch.randn(2, 2, dtype=torch.float, device=device)
            y = torch.randn(2, 2, dtype=torch.float, device=device)
            z = torch.randn(4, 2, dtype=torch.float, device=device)
            ge = self.checkTrace(fn, (x, y, z))
            # 获取张量的图形表示
            graph = ge.graph_for(x, y, z)
            # 断言所有融合的操作，除了 "aten::add" 之外的所有操作都融合
            self.assertAllFused(graph, except_for={"aten::add"})
            # XXX: TE 融合器可以处理融合组内部的拼接操作。
            # FileCheck().check("FusedConcat").check_next("return").run(str(graph))

    @staticmethod
    # 静态方法：测试函数，计算输入张量 x 和 y 的指数
    def fn_test_exp(x, y):
        return (x + 0.5 * y).exp()  # 返回输入张量 x 和 y 的指数值

    # 测试函数：验证指数函数的融合
    def test_exp(self):
        for device in self.devices:
            # 创建具有随机值的张量 x、y，并在指定设备上检查跟踪
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)
            # 获取张量的图形表示，并断言所有操作都已融合
            ge = self.checkTrace(self.fn_test_exp, (x, y))
            self.assertAllFused(ge.graph_for(x, y))

    # 测试函数：验证阈值函数的融合
    def test_threshold(self):
        for device in self.devices:

            # 定义内部函数 f，对输入张量 x 执行阈值操作，并返回结果
            def f(x):
                return torch.threshold(x, 0, -10) + x + x + x

            # 创建具有指定设备上的张量 x，并使用 checkScript 进行脚本化
            x = torch.tensor([-1, -0.5, 0, 1, 2, 3], device=device)
            scripted = self.checkScript(f, (x,))
            # 断言所有操作都已融合
            self.assertAllFused(scripted.graph_for(x))
    def test_scalar_arg(self):
        # 遍历每个设备
        for device in self.devices:

            # 定义一个测试标量参数的函数
            def fn_test_scalar_arg(x: torch.Tensor, p: float) -> torch.Tensor:
                return p * (x * x + x)

            # 生成一个随机张量作为输入 x
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            # 设置标量参数 p
            p = 3
            # 对 fn_test_scalar_arg 函数进行脚本化检查
            scripted = self.checkScript(fn_test_scalar_arg, (x, p))
            # 断言所有操作已融合在一起，并获取其图形表示
            self.assertAllFused(scripted.graph_for(x, p))

            # 将张量 x 设置为需要梯度计算
            x.requires_grad_(True)

            # 定义另一个需要梯度计算的测试函数
            def fn_test_scalar_arg_requires_grad(
                x: torch.Tensor, p: float
            ) -> torch.Tensor:
                return p * (x * x + x)

            # 对 fn_test_scalar_arg_requires_grad 函数进行脚本化处理
            scripted = torch.jit.script(fn_test_scalar_arg_requires_grad)
            # 在脚本化函数上调用输入 x 和标量参数 p
            out = scripted(x, p)
            out = scripted(x, p)
            out = scripted(x, p)
            # 断言所有操作已融合在一起，但除了指定的操作
            self.assertAllFused(
                scripted.graph_for(x, p),
                except_for=(
                    "aten::size",
                    "prim::BroadcastSizes",
                    "aten::_size_if_not_equal",
                ),
            )

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    def test_fusion_reuse_multi_gpu(self):
        # 定义一个简单的函数 fn，用于测试多 GPU 下的融合复用
        def fn(x, y):
            return x * y * x * y

        # 创建输入张量列表，用于 CPU 和 CUDA 设备
        inputs_cpu = [
            torch.randn(4, 4, dtype=torch.float),
            torch.randn(4, 4, dtype=torch.float),
        ]
        inputs_cuda0 = [x.cuda(0) for x in inputs_cpu]
        inputs_cuda1 = [y.cuda(1) for y in inputs_cpu]

        # 对函数 fn 进行脚本化检查，应该不会崩溃；这些应该编译不同的内核
        ge = self.checkScript(fn, inputs_cpu)
        # 断言所有操作已融合在一起，并获取其图形表示
        self.assertAllFused(ge.graph_for(*inputs_cpu))
        # 在不同的 CUDA 设备上调用脚本化函数
        ge(*inputs_cuda0)
        ge(*inputs_cuda1)

    # TODO: we're currently not checking 'device' in the type info when pulling
    # nodes into a fusion group. We should fix that and re-enable this test.
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    # 测试多GPU下的内核缓存
    def test_kernel_cache_multi_gpu(self):
        # 定义一个不可融合的函数
        def not_fusible(x):
            return x

        # 定义一个函数 fn，接受三个输入参数 x, y, z，每个参数都执行五次乘方运算
        def fn(x, y, z):
            x_out = x * x * x * x * x  # fusion: lambda x. x * x * x * x * x
            y_out = y * y * y * y * y
            z_out = z * z * z * z * z
            return not_fusible(x_out), not_fusible(y_out), not_fusible(z_out)

        # 生成三个不同的输入张量，分别在 CPU 和两个 CUDA 设备上
        inputs = [
            torch.randn(4, 4, dtype=torch.float),
            torch.randn(4, 4, dtype=torch.float, device="cuda:0"),
            torch.randn(4, 4, dtype=torch.float, device="cuda:1"),
        ]

        # 获取当前 KernelSpec 缓存的大小
        prev_cache_size = torch._C._jit_debug_fuser_num_cached_kernel_specs()

        # 使用 fn 函数的 TorchScript 表示，检查融合情况
        ge = self.checkScript(fn, inputs)
        self.assertGraphContainsExactly(ge.graph_for(*inputs), FUSION_GROUP, 3, True)

        # 获取更新后的 KernelSpec 缓存大小
        new_cache_size = torch._C._jit_debug_fuser_num_cached_kernel_specs()

        # XXX: 假设当前的测试用例不会与其它测试用例共用相同的内核
        # FIXME: 应使用 TE 融合器查询缓存的方式来进行断言
        # self.assertEqual(new_cache_size - prev_cache_size, 1)

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    # 测试非零设备上的 CUDA
    def test_nonzero_device_cuda(self):
        device = "cuda:" + str(1)
        x = torch.tensor([0.4], dtype=torch.float, device=device)
        y = torch.tensor([0.7], dtype=torch.float, device=device)

        # 定义一个函数 doit，执行复杂的数学运算
        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y) + x))

        # 检查 doit 函数的 TorchScript 表示，确保所有操作都被融合
        ge = self.checkTrace(doit, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    # 测试 LSTM 模型
    def test_lstm(self):
        # 遍历设备列表中的每一个设备
        for device in self.devices:
            # 获取特定设备上的 LSTM 输入
            inputs = get_lstm_inputs(device, training=True)
            # 对 LSTMCellS 模型进行 TorchScript 化，并检查所有节点是否被融合
            module = self.checkScript(LSTMCellS, inputs)
            self.assertAllFused(
                module.graph_for(inputs), except_for={"prim::TupleConstruct"}
            )

    # 测试 LSTM 模型的拼接操作
    def test_lstm_concat(self):
        # 设置启用内联融合组
        with set_fusion_group_inlining(True):
            # 遍历设备列表中的每一个设备
            for device in self.devices:
                # 获取特定设备上的 LSTM 输入
                inputs = get_lstm_inputs(device)
                # 检查 LSTMCellC 模型的 TorchScript 表示，确保除了指定节点外所有节点都被融合
                ge = self.checkTrace(LSTMCellC, inputs)
                graph = ge.graph_for(*inputs)
                except_nodes = {"prim::TupleConstruct", "aten::linear"}
                # TODO... Chunk
                if self.dynamic_shapes:
                    except_nodes = except_nodes.union(
                        {"aten::add", "prim::ConstantChunk"}
                    )
                self.assertAllFused(ge.graph_for(*inputs), except_for=except_nodes)
                # XXX: TE 融合器可以处理融合组内的拼接操作
                # FileCheck().check("FusedConcat").check_next("return").run(str(graph))
    def test_lstm_gates_permutations(self):
        for device in self.devices:
            # lstm has gates = x.mm(w_ih.t()) + hx.mm(w_hh.t()) + b_ih + b_hh.
            # Test that any permutation of this will still result in one FusionGroup.
            choices = ["x.mm(w_ih.t())", "hx.mm(w_hh.t())", "b_ih", "b_hh"]
            template = dedent(
                """
            def cell(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
                gates = {} + {} + {} + {}
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
                return ingate * forgetgate * cellgate * outgate
            """
            )
            for permutation in permutations(choices, len(choices)):
                code = template.format(*permutation)
                scope = {}
                exec(code, globals(), scope)
                cu = torch.jit.CompilationUnit(code)
                fusion_group_len = 2 if self.dynamic_shapes else 1
                inputs = get_lstm_inputs(device, training=False)
                self.assertEqual(cu.cell(*inputs), scope["cell"](*inputs))
                forward_graph = cu.cell.graph_for(*inputs)
                self.assertGraphContainsExactly(
                    forward_graph, FUSION_GROUP, fusion_group_len
                )

    # TODO: Fuser doesn't work at all when inputs require grad. Fix that
    def test_lstm_traced(self):
        for device in self.devices:
            inputs = get_lstm_inputs(device)
            ge = self.checkTrace(LSTMCellF, inputs)
            graph = ge.graph_for(*inputs)
            fusion_groups = self.findFusionGroups(graph)
            # TODO: chunk
            fusion_group_len = 2 if self.dynamic_shapes else 1
            self.assertEqual(len(fusion_groups), fusion_group_len)
            f = FileCheck()
            if not self.dynamic_shapes:
                f.check("Chunk")
            f.check("aten::sigmoid").check("aten::tanh").run(
                str(fusion_groups[0 if not self.dynamic_shapes else 1])
            )

    def test_milstm(self):
        if self.dynamic_shapes:
            self.skipTest("don't run conv with dynamic shapes")

        for device in self.devices:
            inputs = get_milstm_inputs(device, training=True)
            module = self.checkScript(MiLSTMCell, inputs)
            forward_graph = module.graph_for(*inputs)
            # TODO: chunk
            fusion_group_len = 2 if self.dynamic_shapes else 1
            self.assertGraphContainsExactly(
                forward_graph, FUSION_GROUP, fusion_group_len, consider_subgraphs=True
            )
            FileCheck().check("DifferentiableGraph").check("TupleConstruct").check_next(
                "return"
            ).check(FUSION_GROUP).run(str(forward_graph))
            hy, cy = module(*inputs)
            warmup_backward((hy + cy).sum())

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skip("rand_like is not supported yet")


注释：

    def test_lstm_gates_permutations(self):
        for device in self.devices:
            # lstm has gates = x.mm(w_ih.t()) + hx.mm(w_hh.t()) + b_ih + b_hh.
            # 测试任意排列顺序是否仍将结果合并为一个 FusionGroup。
            choices = ["x.mm(w_ih.t())", "hx.mm(w_hh.t())", "b_ih", "b_hh"]
            template = dedent(
                """
            def cell(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
                gates = {} + {} + {} + {}
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
                return ingate * forgetgate * cellgate * outgate
            """
            )
            for permutation in permutations(choices, len(choices)):
                code = template.format(*permutation)
                scope = {}
                exec(code, globals(), scope)
                cu = torch.jit.CompilationUnit(code)
                fusion_group_len = 2 if self.dynamic_shapes else 1
                inputs = get_lstm_inputs(device, training=False)
                self.assertEqual(cu.cell(*inputs), scope["cell"](*inputs))
                forward_graph = cu.cell.graph_for(*inputs)
                self.assertGraphContainsExactly(
                    forward_graph, FUSION_GROUP, fusion_group_len
                )

    # TODO: Fuser doesn't work at all when inputs require grad. Fix that
    def test_lstm_traced(self):
        for device in self.devices:
            inputs = get_lstm_inputs(device)
            ge = self.checkTrace(LSTMCellF, inputs)
            graph = ge.graph_for(*inputs)
            fusion_groups = self.findFusionGroups(graph)
            # TODO: chunk
            fusion_group_len = 2 if self.dynamic_shapes else 1
            self.assertEqual(len(fusion_groups), fusion_group_len)
            f = FileCheck()
            if not self.dynamic_shapes:
                f.check("Chunk")
            f.check("aten::sigmoid").check("aten::tanh").run(
                str(fusion_groups[0 if not self.dynamic_shapes else 1])
            )

    def test_milstm(self):
        if self.dynamic_shapes:
            self.skipTest("don't run conv with dynamic shapes")

        for device in self.devices:
            inputs = get_milstm_inputs(device, training=True)
            module = self.checkScript(MiLSTMCell, inputs)
            forward_graph = module.graph_for(*inputs)
            # TODO: chunk
            fusion_group_len = 2 if self.dynamic_shapes else 1
            self.assertGraphContainsExactly(
                forward_graph, FUSION_GROUP, fusion_group_len, consider_subgraphs=True
            )
            FileCheck().check("DifferentiableGraph").check("TupleConstruct").check_next(
                "return"
            ).check(FUSION_GROUP).run(str(forward_graph))
            hy, cy = module(*inputs)
            warmup_backward((hy + cy).sum())

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skip("rand_like is not supported yet")
    def test_rand_cuda(self):
        # 定义一个继承自torch.jit.ScriptModule的类M，用于在CUDA设备上执行脚本化模块
        class M(torch.jit.ScriptModule):
            # 定义常量列表，包含属性"d"
            __constants__ = ["d"]

            # 初始化方法，设置self.d为CUDA设备
            def __init__(self):
                super().__init__()
                self.d = torch.device("cuda")

            # 使用torch.jit.script_method装饰器定义的方法，创建操作，返回x * x + x + torch.rand_like(x)
            @torch.jit.script_method
            def create(self, x):
                return x * x + x + torch.rand_like(x)

        # 在CUDA设备上创建一个形状为[3, 4, 5]的全零张量x
        x = torch.zeros([3, 4, 5], dtype=torch.float, device="cuda")
        # 创建类M的实例m
        m = M()
        # 使用m的create方法对x进行操作，生成out1和out2两个结果
        out1 = m.create(x)
        out2 = m.create(x)
        # 断言out1和out2不相等
        self.assertNotEqual(out1, out2)
        # 断言out1中所有元素大于等于0
        self.assertTrue(torch.all(out1 >= 0))
        # 断言out1中所有元素小于1
        self.assertTrue(torch.all(out1 < 1))
        # 断言out2中所有元素大于等于0
        self.assertTrue(torch.all(out2 >= 0))
        # 断言out2中所有元素小于1
        self.assertTrue(torch.all(out2 < 1))
        # 断言所有操作都被融合优化
        self.assertAllFused(m.create.graph_for(x))

    @staticmethod
    def fn_test_relu(x, y):
        # 返回F.relu(x + 0.5 * y)的结果
        return F.relu(x + 0.5 * y)

    def test_relu(self):
        # 遍历self.devices中的设备
        for device in self.devices:
            # 使用指定设备创建形状为[4, 4]的随机张量x和y
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)
            # 检查fn_test_relu函数的追踪版本ge
            ge = self.checkTrace(self.fn_test_relu, (x, y))
            # 断言所有操作都被融合优化
            self.assertAllFused(ge.graph_for(x, y))

    def test_erf(self):
        # 遍历self.devices中的设备
        for device in self.devices:
            # 仅在GPU上执行该测试
            if device == "cpu":
                continue

            # 定义fn_test_erf函数，返回F.relu(torch.erf(x) - torch.erfc(x))的结果
            def fn_test_erf(x):
                return F.relu(torch.erf(x) - torch.erfc(x))

            # 使用指定设备创建形状为[4, 4]的随机张量x
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            # 检查fn_test_erf函数的脚本化版本ge，使用性能分析模式
            ge = self.checkScript(fn_test_erf, (x,), profiling=ProfilingMode.PROFILING)
            # 断言所有操作都被融合优化
            self.assertAllFused(ge.graph_for(x))
            # 将x设置为需要梯度计算
            x.requires_grad_(True)
            # 再次检查fn_test_erf函数的脚本化版本ge，使用性能分析模式
            ge = self.checkScript(fn_test_erf, (x,), profiling=ProfilingMode.PROFILING)
            # 断言所有操作都被融合优化，但排除特定的操作
            self.assertAllFused(
                ge.graph_for(x),
                except_for=(
                    "aten::size",
                    "prim::BroadcastSizes",
                    "aten::_size_if_not_equal",
                ),
            )

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skip("rand_like is not supported yet")
    # 定义一个测试函数，用于测试在 CUDA 上执行随机数生成和广播操作
    def test_rand_broadcast_cuda(self):
        # 定义一个函数，生成一个与给定张量 y 相同形状的随机张量，然后进行广播乘法和加法操作
        def fn_test_rand(x, y):
            r = torch.rand_like(y)
            return r * x + x
        
        # 如果使用性能分析，需要使用不同的函数来测试不同形状，否则会使用缓存的脚本
        def fn_test_rand2(x, y):
            r = torch.rand_like(y)
            return r * x * x

        # 创建 CUDA 设备上的随机张量 x 和 y
        x = torch.randn(4, 4, dtype=torch.float, device="cuda")
        y = torch.randn(4, 4, dtype=torch.float, device="cuda")
        
        # 将函数 fn_test_rand 编译为 TorchScript
        script_f = torch.jit.script(fn_test_rand)
        # 预热 TorchScript 函数
        warmup_forward(script_f, x, y)
        # 执行 TorchScript 函数
        out = script_f(x, y)
        # 断言所有操作都被融合到单个内核
        self.assertAllFused(script_f.graph_for(x, y))
        
        # 开启 x 的梯度计算
        x.requires_grad_(True)
        out = script_f(x, y)
        # 断言所有操作被融合到单个内核，但排除指定的操作
        self.assertAllFused(
            script_f.graph_for(x, y),
            except_for=(
                "aten::size",
                "prim::BroadcastSizes",
                "aten::_size_if_not_equal",
            ),
        )

        # 测试随机数广播生成正确的结果
        x = torch.ones(4, 4, dtype=torch.float, device="cuda")
        y = torch.ones(4, dtype=torch.float, device="cuda")
        
        # 将函数 fn_test_rand2 编译为 TorchScript
        script_f = torch.jit.script(fn_test_rand2)
        # 预热 TorchScript 函数
        warmup_forward(script_f, x, y)
        # 执行 TorchScript 函数
        out = script_f(x, y)
        # 断言结果符合预期
        self.assertEqual(out[0, :] + torch.zeros(4, 4, device="cuda"), out)

    # 如果未开启 CUDA，跳过此测试；标记为不支持 rand_like 函数
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skip("rand_like is not supported yet")
    def test_rand_diamond(self):
        # 定义一个函数，生成两个随机张量的差，并返回其和
        def fn_test_diamond(x, y):
            r = torch.rand_like(y)
            a = x + r
            b = y - r
            return a + b

        # 创建 CUDA 设备上的随机张量 x 和 y
        x = torch.randn(4, 4, dtype=torch.float, device="cuda")
        y = torch.randn(4, 4, dtype=torch.float, device="cuda")
        
        # 将函数 fn_test_diamond 编译为 TorchScript
        script_f = torch.jit.script(fn_test_diamond)
        # 预热 TorchScript 函数
        warmup_forward(script_f, x, y)
        # 执行 TorchScript 函数
        out = script_f(x, y)
        # 断言函数输出与 x + y 相等
        self.assertEqual(out, x + y)

    # 测试标量计算函数
    def test_scalar(self):
        # 定义一个简单的函数，对输入的 x 和 y 进行标量计算
        def fn(x, y):
            return 2 * x + y
        
        # 创建 CPU 设备上的标量张量 x 和 y
        x = torch.tensor(0.1, dtype=torch.float, device="cpu")
        y = torch.tensor(1, dtype=torch.float, device="cpu")
        
        # 检查 TorchScript 执行结果
        ge = self.checkScript(fn, (x, y))
        # 断言所有操作被融合到单个内核
        self.assertAllFused(ge.graph_for(x, y))

    # 测试内联优化后的图形
    def test_inlined_optimized_graph(self):
        # 定义一个 TorchScript 函数，应用 ReLU 激活函数并返回结果
        @torch.jit.script
        def foo(x):
            return torch.relu(x + x)

        # 多次调用不同形状的输入张量，以执行函数 foo
        for _ in range(3):
            foo(torch.rand([4, 4]))

        for _ in range(3):
            foo(torch.rand([10]))

        for _ in range(3):
            foo(torch.rand([2, 2, 2]))

        # 获取最后执行的优化图形
        g = torch.jit.last_executed_optimized_graph()

        # 运行 FileCheck 检查，验证图形中的特定操作和结构
        FileCheck().check_count("prim::If", 1, exactly=True).check(
            "prim::TensorExpr"
        ).run(g)
        # 内联优化后再次运行 FileCheck 检查
        torch._C._jit_pass_inline(g)
        f = FileCheck()
        for _ in range(3):
            f.check("prim::If").check("prim::TensorExpr")
        f.run(g)
    def test_small_constant(self):
        # 对于每个设备进行测试
        for device in self.devices:

            # 定义一个函数 fn_test_small_constant，用于计算常量和张量的乘法和加法
            def fn_test_small_constant(x, y):
                return (1e-8 * x + 5e-9 * y) * 1e8

            # 创建两个随机张量 x 和 y，指定类型为 float，设备为当前设备
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)

            # 检查函数 fn_test_small_constant 是否被正确转换为图并进行融合
            ge = self.checkTrace(fn_test_small_constant, (x, y))
            # 断言所有操作都已经被融合到同一个图中
            self.assertAllFused(ge.graph_for(x, y))

    # 当前我们不将常量拉入融合组，因为在某些情况下，这可能会从原始图中移除常量，
    # 现在我们的融合组需要为其它用户返回该常量。
    # 我们不应该从不将常量拉入融合组，而是应该在重写其用户时更加小心。
    # TODO: 修复此问题并重新启用测试。
    def test_tensor_scalar_ops(self):
        # 对于每个设备进行测试
        for device in self.devices:

            # 定义一个应该进行融合的函数 should_fuse
            def should_fuse(x):
                z = 3.0
                y = x + z
                return x * y

            # 定义一个应该进行融合的函数 should_fuse_scalar，其中 z 是一个标量
            def should_fuse_scalar(x, z):
                y = x + int(z)
                return x * y

            # 创建输入张量列表，只包含一个随机张量 x
            inputs = [torch.randn(2, 2, dtype=torch.float, device=device)]
            # 检查函数 should_fuse 是否被正确转换为图并进行融合
            ge = self.checkScript(should_fuse, inputs)
            # 获取转换后的图并查找融合组
            graph = ge.graph_for(*inputs)
            fusion_groups = self.findFusionGroups(graph)
            # 断言只存在一个融合组
            self.assertEqual(len(fusion_groups), 1)
            # 使用 FileCheck 检查融合组中是否包含 "aten::add" 和 "aten::mul" 操作
            FileCheck().check("aten::add").check("aten::mul").run(str(fusion_groups[0]))

            # 创建包含两个输入张量的输入列表，第二个输入是一个常量标量 3.0
            inputs = [
                torch.randn(2, 2, dtype=torch.float, device=device),
                torch.tensor(3.0, dtype=torch.float, device=device),
            ]
            # 检查函数 should_fuse_scalar 是否被正确转换为图并进行融合
            ge = self.checkScript(should_fuse_scalar, inputs)
            # 检查融合后的图在标量输入变化时是否计算出正确结果
            inputs = [
                torch.randn(2, 2, dtype=torch.float, device=device),
                torch.tensor(7.0, dtype=torch.float, device=device),
            ]
            self.assertEqual(ge(*inputs), should_fuse_scalar(*inputs))
            # 断言 TE fuser 支持非常量标量的融合
            self.assertGraphContainsExactly(
                ge.graph_for(*inputs), FUSION_GROUP, 1, consider_subgraphs=True
            )

    def test_where_and_typing(self):
        # 对于每个设备进行测试
        for device in self.devices:

            # 定义一个函数 f，计算张量 x 和 y 的比较结果及根据结果的条件选择
            def f(x, y):
                mask = x > y
                res = torch.where(mask, x, y)
                return mask, res

            # 创建两个随机张量 x 和 y，指定类型为 double，设备为当前设备
            x = torch.randn(4, 4, dtype=torch.double, device=device)
            y = torch.randn(4, 4, dtype=torch.double, device=device)

            # 检查函数 f 是否被正确转换为脚本并进行融合，除了 prim::TupleConstruct 外的所有操作
            script_f = self.checkScript(f, (x, y))
            self.assertAllFused(
                script_f.graph_for(x, y), except_for={"prim::TupleConstruct"}
            )
    # 定义一个测试方法，用于测试禁用状态下的功能
    def test_disabled(self):
        # 存储旧的 CPU 融合状态
        old_cpu_fuser_state = torch._C._jit_can_fuse_on_cpu()
        # 覆盖 CPU 融合状态为禁用
        torch._C._jit_override_can_fuse_on_cpu(False)

        # 定义一个简单的函数 fn，计算输入的平方加上输入本身
        def fn(a):
            return a**2 + a

        # 创建一个在 CPU 上的随机张量 x
        x = torch.randn(4, dtype=torch.float, device="cpu")
        # 使用自定义方法 checkScript 检查 fn 函数的脚本化版本，并返回结果
        s = self.checkScript(fn, (x,))
        # 获取张量 x 对应的计算图
        g = s.graph_for(x)
        # 检查计算图中的融合组数量是否为零
        self.assertEqual(len(self.findFusionGroups(g)), 0)

        # 恢复原来的 CPU 融合状态
        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuser_state)

    # 为给定数据类型和设备生成数据
    def data_for(self, dtype, device="cuda", size=None):
        # 如果未指定 size，则创建一个范围在 [1, 3) 的浮点数张量 v，根据设备类型
        if size is None:
            v = torch.arange(1, 3, dtype=torch.float, device=device)
        else:
            # 否则，根据指定的 size 创建随机张量 v，根据设备类型
            v = torch.rand(*size, device=device)
        
        # 根据数据类型选择返回不同的数据
        if dtype == torch.bool:
            # 如果数据类型是布尔型，则返回大于 2 的布尔掩码
            return v > 2
        elif dtype in [torch.qint8, torch.quint8, torch.qint32]:
            # 如果数据类型是量化类型之一，则对张量 v 进行量化并返回
            return torch.quantize_per_tensor(v, 0.1, 1, dtype=dtype)
        else:
            # 否则，将张量 v 转换为指定的数据类型并返回
            return v.to(dtype)
    def test_torch_to(self):
        # 定义一个测试函数，测试不进行任何操作
        @torch.jit.script
        def foo(x):
            return x.to(torch.float)

        # 调用 foo 函数两次，分别传入包含单个浮点数 3.0 的张量
        foo(torch.tensor([3.0], dtype=torch.float))
        foo(torch.tensor([3.0], dtype=torch.float))
        
        # 使用 FileCheck 检查最后优化的图中是否没有 "TensorExpr"，表示未进行张量表达式的优化
        FileCheck().check_not("TensorExpr").run(
            torch.jit.last_executed_optimized_graph()
        )

        # 测试不融合非常量输入的情况
        @torch.jit.script
        def foo(x, dtype: int):
            return x.to(dtype)

        # 调用 foo 函数两次，分别传入包含单个浮点数 3.0 的张量和 torch.int 类型
        foo(torch.tensor([3.0], dtype=torch.float), torch.int)
        foo(torch.tensor([3.0], dtype=torch.float), torch.int)
        
        # 使用 FileCheck 检查最后优化的图中是否没有 "TensorExpr"，表示未进行张量表达式的优化
        FileCheck().check_not("TensorExpr").run(
            torch.jit.last_executed_optimized_graph()
        )

        # 测试不融合到锁页内存的输入情况
        @torch.jit.script
        def foo(x, dtype: int):
            return x.to(pin_memory=True)

        # 调用 foo 函数两次，分别传入包含单个浮点数 3.0 的张量和 torch.int 类型
        foo(torch.tensor([3.0], dtype=torch.float), torch.int)
        foo(torch.tensor([3.0], dtype=torch.float), torch.int)
        
        # 使用 FileCheck 检查最后优化的图中是否没有 "TensorExpr"，表示未进行张量表达式的优化
        FileCheck().check_not("TensorExpr").run(
            torch.jit.last_executed_optimized_graph()
        )

        # 测试不支持跨设备的情况（如果 CUDA 可用）
        if torch.cuda.is_available():

            @torch.jit.script
            def foo(x):
                return x.to(device="cuda")

            # 调用 foo 函数两次，分别传入包含单个浮点数 3.0 的张量
            foo(torch.tensor([3.0], dtype=torch.float))
            foo(torch.tensor([3.0], dtype=torch.float))
            
            # 使用 FileCheck 检查最后优化的图中是否没有 "TensorExpr"，表示未进行张量表达式的优化
            FileCheck().check_not("TensorExpr").run(
                torch.jit.last_executed_optimized_graph()
            )

        # 定义大小和数据类型的列表
        sizes = [(1, 4), (4, 4)]
        # 为了加快测试速度，使用较小的数据类型集合
        dtypes = [
            torch.bool,
            torch.int,
            torch.float16,
            torch.float32,
            torch.float64,
        ]

        # 定义一个自定义的 PyTorch 模块
        class MyMod(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                return x.to(self.dtype)

        # 定义一个空列表用于存储错误的数据类型
        bad_dtypes = []

        # 使用 product 函数迭代数据类型、输出数据类型、设备和大小的组合
        for dtype, output_dtype, device, size in product(
            dtypes, dtypes, self.devices, sizes
        ):
            # 当 dtype 是 torch.float16 或 torch.bfloat16 且设备为 "cpu" 时，跳过当前循环
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            # 当 dtype 等于 output_dtype 时，跳过当前循环
            if dtype == output_dtype:
                continue

            # 为给定的 dtype、设备和大小创建数据
            x = self.data_for(dtype, device, size=size)
            # 创建 MyMod 实例，并使用 output_dtype 初始化
            mod = MyMod(output_dtype)
            # 使用 mod 实例的 forward 方法生成参考结果 ref
            ref = mod.forward(x)
            # 使用冻结的模型来将非张量参数传递给 `to` 函数，使其保持常量
            mod = torch.jit.freeze(torch.jit.script(mod.eval()))
            # 预热模型的前向传播函数
            warmup_forward(mod.forward, x)
            # 断言参考结果与模型的 forward 方法返回值相等
            self.assertEqual(ref, mod.forward(x))
            # 断言最后一个图是否全部融合
            self.assertLastGraphAllFused()

    @unittest.skip("Temporarily disabled")
    # 定义一个测试函数，用于测试 torch.masked_fill 方法
    def test_masked_fill(self):
        # 定义不同的数据类型列表，包括整数和浮点数类型
        dtypes = [
            torch.int8,      # 8位整数
            torch.int16,     # 16位整数
            torch.int32,     # 32位整数
            torch.int64,     # 64位整数
            # TODO: 当 https://github.com/pytorch/pytorch/issues/55905 问题关闭时，添加回来
            # torch.float16,  # 16位浮点数
            torch.float32,   # 32位浮点数
            torch.float64,   # 64位浮点数
            torch.bool,      # 布尔类型
        ]
        # 定义不同的张量尺寸
        sizes = [(2,), (4, 4)]
        # 对每一种数据类型、设备、标量值和尺寸进行组合遍历
        for self_dtype, device, scalar_val, size in product(
            dtypes, self.devices, [0.4, 3], sizes
        ):
            # 使用 self.data_for 方法生成指定数据类型、设备和尺寸的输入数据张量
            input_v = self.data_for(self_dtype, device, size=size)
            # 使用 self.data_for 方法生成布尔类型的掩码张量
            mask = self.data_for(torch.bool, device, size=size)

            # 定义一个匿名函数 fn，用于调用 torch.masked_fill 方法
            def fn(input_v, mask):
                return torch.masked_fill(input_v, mask, scalar_val)

            # 计算参考结果 ref
            ref = fn(input_v, mask)
            try:
                # 对函数 fn 进行 Torch JIT 追踪
                t = torch.jit.trace(fn, (input_v, mask))
                # 断言追踪后的函数 t 与 ref 的结果接近
                torch.testing.assert_close(ref, t(input_v, mask))
                # 断言所有操作均已融合
                self.assertLastGraphAllFused()
            except Exception as e:
                # 如果测试失败，抛出运行时异常并提供失败详细信息
                raise RuntimeError(
                    " ".join(
                        [
                            "Failed:",
                            str(self_dtype),
                            op.__name__,  # noqa: F821
                            device,
                            str(size),
                        ]
                    )
                ) from e

    # 定义一个测试函数，用于测试 torch.isnan 方法
    def test_isnan(self):
        # 生成一个包含随机数的张量 x
        x = torch.rand([4])
        # 将第一个元素设置为 NaN
        x[0] = float("nan")
        # 定义输入列表，包含两个张量：一个包含 NaN 的张量 x，一个包含 NaN 和 0.5 的张量
        inputs = [x, torch.tensor([float("nan"), 0.5])]
        # 定义不同的数据类型列表，包括整数和浮点数类型
        dtypes = [
            torch.int8,      # 8位整数
            torch.int16,     # 16位整数
            torch.int32,     # 32位整数
            torch.int64,     # 64位整数
            torch.float16,   # 16位浮点数
            torch.float32,   # 32位浮点数
            torch.float64,   # 64位浮点数
            torch.bool,      # 布尔类型
        ]

        # 对每一种输入、设备和数据类型进行组合遍历
        for inp, device, dtype in product(inputs, self.devices, dtypes):
            # TODO: 当 https://github.com/pytorch/pytorch/issues/55905 问题关闭时，添加回来
            # 如果数据类型是 torch.float16 或 torch.bfloat16 并且设备是 CPU，则跳过该测试
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            # 将输入张量转移到指定设备和数据类型
            inp = inp.to(device=device, dtype=dtype)
            try:
                # 使用 Torch JIT 追踪 lambda 函数，检查张量是否为 NaN
                f = torch.jit.trace(lambda x: x.isnan(), (inp,))
                # 对追踪的函数进行预热
                warmup_forward(f, inp)
                # 断言追踪后的函数结果与 inp 是否为 NaN 的结果一致
                self.assertEqual(f(inp), inp.isnan())
                # 断言所有操作均已融合
                self.assertLastGraphAllFused()
            except Exception as e:
                # 如果测试失败，抛出运行时异常并提供失败详细信息
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), "isnan", device])
                ) from e
    # 定义一个内部函数apply，用于将函数fn应用于输入x和布尔值参数approximate
    def test_gelu(self):
        def apply(fn):
            return lambda x, approximate: fn(x, approximate)

        # 定义一个包含GELU函数的列表
        unary_ops = [
            F.gelu,
        ]
        # 定义不同的数据尺寸
        sizes = [(1,), (2,), (4, 4)]
        
        # 使用product生成器，迭代所有可能的dtype、操作、设备和尺寸组合
        for dtype, op, device, size in product(
            self.dtypes, unary_ops, self.devices, sizes
        ):
            # TODO: Add back when https://github.com/pytorch/pytorch/issues/55905 is closed
            # 如果dtype是torch.float16或torch.bfloat16，并且设备为"cpu"，则跳过当前循环
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            try:
                # 准备数据x，根据dtype、device和size参数生成
                x = self.data_for(dtype, device, size=size)
                # 准备条件数据cond，类型为torch.bool
                cond = self.data_for(torch.bool, device)
                # 应用函数op到x和cond，生成一个新的函数fn
                fn = apply(op)
                # 计算参考结果ref，使用fn应用到x和cond上
                ref = fn(x, cond)
            except Exception:
                # 如果出现异常，捕获并继续下一个循环
                # 如果eager模式不支持特定的dtype/op/device组合，fuser也不支持
                continue
            try:
                # 使用torch.jit.trace对fn进行跟踪，输入为(x, cond)
                t = torch.jit.trace(fn, (x, cond))
                # 使用torch.testing.assert_close比较ref和t(x, cond)，确保它们在误差允许范围内相等
                torch.testing.assert_close(ref, t(x, cond))
                # 断言所有操作都被融合（fused）到一个图中，用于检查性能优化
                self.assertAllFused(t.graph_for(x, cond))
            except Exception as e:
                # 如果出现异常，将其重新引发为RuntimeError，并包含错误信息
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device, str(size)])
                ) from e
    # 定义一个测试方法，用于测试各种二进制操作
    def test_binary_ops(self):
        # 定义一个函数，将操作符函数作用在两个参数上
        def apply(fn):
            return lambda x, y: fn(x, y)

        # 定义一组二进制操作，包括位运算和torch库中的张量操作
        binary_ops = [
            operator.__and__,        # 位与操作
            operator.__or__,         # 位或操作
            operator.__xor__,        # 位异或操作
            torch.add,               # 张量加法
            torch.sub,               # 张量减法
            torch.mul,               # 张量乘法
            torch.min,               # 张量最小值
            torch.max,               # 张量最大值
            lambda x, y: torch.lerp(x, y, 0.5),  # 线性插值
            torch.atan2,             # 反正切函数
            torch.div,               # 张量除法
            torch.eq,                # 张量相等比较
            torch.ne,                # 张量不等比较
            torch.ge,                # 张量大于等于比较
            torch.gt,                # 张量大于比较
            torch.lt,                # 张量小于比较
            torch.fmod,              # 张量取模
            torch.remainder,         # 张量求余
            lambda x, y: y.type_as(x),  # 张量类型转换
        ]
        
        # 仅适用于浮点数的操作列表
        fp_only = [
            torch.fmod,              # 张量取模
            torch.remainder,         # 张量求余
        ]
        
        # 获取测试用的设备列表
        devices = self.devices
        
        # 针对每种数据类型、操作和设备的组合进行测试
        for dtype, op, device in product(self.dtypes, binary_ops, devices):
            # 跳过不支持的数据类型和设备组合
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            
            try:
                # 准备测试数据
                x = self.data_for(dtype, device)
                y = self.data_for(dtype, device)
                fn = apply(op)
                
                # 执行操作得到参考结果
                ref = fn(x, y)
            except Exception:
                # 如果遇到异常，继续下一个组合的测试
                continue
            
            try:
                # 使用 JIT 跟踪函数
                t = torch.jit.trace(fn, (x, y))
                
                # 断言跟踪后的结果与参考结果一致
                self.assertEqual(ref, t(x, y))
                
                # 对于支持的操作，验证所有操作是否被融合
                if op not in fp_only or dtype.is_floating_point:
                    self.assertAllFused(t.graph_for(x, y))
            except Exception as e:
                # 如果测试失败，抛出异常信息
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device])
                ) from e
    def test_binary_scalar_ops(self):
        # 定义一个内部函数，用于生成二元操作的函数
        def apply(fn):
            return lambda x, y: fn(x, y)

        # 定义了一个 IR 模板字符串，用于生成计算图
        ir_template = """
        graph(%x : {dtype_x}, %y : {dtype_y}):
          %z = {op}(%x, %y)
          return (%z)"""

        # 定义了一组二元操作符
        binary_ops = [
            "aten::mul",
            "aten::add",
            "aten::sub",
            "aten::div",
            "aten::lt",
            "aten::le",
            "aten::eq",
            "aten::ne",
            "aten::gt",
            "aten::ge",
            "aten::__or__",
            "aten::__xor__",
            "aten::__and__",
            "aten::__lshift__",
            "aten::__rshift__",
        ]
        # 支持的数据类型
        dtypes = ["int", "float", "bool"]
        # 不同数据类型对应的数值示例
        values = {"int": [10, 3], "float": [12.34, 2.78], "bool": [True, False]}
        # 测试设备列表
        devices = self.devices

        # 使用 product 函数迭代所有可能的数据类型组合、操作符和设备
        for dtype_x, dtype_y, op, device in product(
            dtypes, dtypes, binary_ops, devices
        ):
            # 根据模板和当前局部变量生成具体的 IR 代码
            code = ir_template.format(**locals())

            # 解析并解释计算图
            try:
                # 解析 IR 代码为计算图对象
                graph = torch._C.parse_ir(code)
                # 对数据类型组合中的每对值进行解释执行
                for x, y in product(values[dtype_x], values[dtype_y]):
                    ref = torch._C._jit_interpret_graph(graph, (x, y))
            except Exception:
                # 如果无法解释此 IR，跳过当前组合
                continue

            # 编译计算图
            try:
                # 使用 TensorExprKernel 对象编译计算图
                k = torch._C._te.TensorExprKernel(graph)
            except Exception as e:
                # 如果编译失败，抛出运行时错误
                raise RuntimeError(
                    " ".join(["Compilation failed:", device, str(code)])
                ) from e

            # 运行编译后的计算图
            for x, y in product(values[dtype_x], values[dtype_y]):
                # 通过解释执行获取参考结果
                ref = torch._C._jit_interpret_graph(graph, (x, y))
                try:
                    # 运行 TensorExprKernel 对象并比较结果
                    res = k.run((x, y))
                    self.assertEqual(ref, res)
                except Exception as e:
                    # 如果运行时出现错误，抛出带有详细信息的运行时错误
                    raise RuntimeError(
                        " ".join(
                            ["Failed at runtime:", device, str(x), str(y), str(code)]
                        )
                    ) from e
    # 定义一个测试矩阵乘法的方法，用于单元测试
    def test_matmul(self):
        # 如果支持动态形状，则跳过测试，因为动态形状下不运行卷积操作
        if self.dynamic_shapes:
            self.skipTest("don't run conv with dynamic shapes")

        # 定义一个简单的函数，用于执行矩阵乘法
        def fn(x, y):
            return torch.matmul(x, y)

        # 只支持在 CPU 上执行，因此设备列表只包含 "cpu"
        devices = ["cpu"]  # No cuda support for ext calls yet
        
        # 不同的矩阵尺寸组合，用于测试矩阵乘法的不同情况
        sizes = [
            [[128, 128], [128, 128]],
            [[10, 10], [10, 10]],
            [[1, 16], [16, 128]],
            [[128], [128]],
            [[128], [128, 128]],
            [[3], [3]],
            [[3, 4], [4]],
            [[10, 3, 4], [4]],
            [[10, 3, 4], [10, 4, 5]],
            [[10, 3, 4], [4, 5]],
        ]

        # 需要跳过 'is-fused' 检查的矩阵尺寸组合，因为这些尺寸目前不支持
        skip_is_fused_check_sizes = [
            "[[128], [128]]",
            "[[128], [128, 128]]",
            "[[3], [3]]",
            "[[3, 4], [4]]",
            "[[10, 3, 4], [4]]",
            "[[10, 3, 4], [10, 4, 5]]",
            "[[10, 3, 4], [4, 5]]",
        ]

        # 遍历数据类型、尺寸和设备的组合
        for dtype, size, device in product(self.dtypes, sizes, devices):
            # 跳过不支持的组合，如在 CPU 上不支持 torch.float16 和 torch.bfloat16
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            try:
                size_x, size_y = size
                # 为给定的数据类型、设备和尺寸创建数据张量 x 和 y
                x = self.data_for(dtype, device, size=size_x)
                y = self.data_for(dtype, device, size=size_y)
                # 计算参考结果
                ref = fn(x, y)
            except Exception as e:
                # 如果 eager 模式不支持特定的数据类型、操作或设备组合，忽略错误
                continue
            try:
                # 对 fn 函数进行追踪，以便后续的图优化和验证
                t = torch.jit.trace(fn, (x, y))
                # 执行追踪模型，检查是否与参考结果一致
                t(x, y)
                self.assertEqual(ref, t(x, y))
                # 如果尺寸不在需要跳过 'is-fused' 检查的列表中，验证所有操作是否被融合
                if str(size) not in skip_is_fused_check_sizes:
                    self.assertAllFused(t.graph_for(x, y))
            except Exception as e:
                # 捕获异常并抛出运行时错误，以便进行错误处理和调试
                raise RuntimeError(" ".join(["Failed:", str(dtype), device])) from e
    # 定义测试函数，用于测试二进制张量和标量操作的功能
    def test_binary_tensor_scalar_ops(self):
        # 禁用 Torch 内部的代码生成钩子，确保测试的纯粹性和一致性
        with torch._jit_internal._disable_emit_hooks():

            # 定义一个函数，用于将标量应用到给定的二元操作函数上
            def apply_with_scalar(fn, scalar):
                return lambda x: fn(x, scalar)

            # FIXME: 在 IR Eval 中失败：torch.int64 和 cpu 之间的位与运算
            # 定义支持的二元操作列表
            binary_ops = [
                operator.__and__,
                operator.__or__,
                operator.__xor__,
                torch.add,
                torch.sub,
                torch.mul,
                torch.eq,
                torch.ne,
                torch.ge,
                torch.lt,
                torch.gt,
            ]
            # 获取测试设备列表
            devices = self.devices
            # 可能应该将此部分分成不同的测试，以便通过仅使用特定操作相关的标量值来加快速度
            # 定义用于测试的标量值列表
            scalars = [1.5, 3, 0, -2.0, -1]
            # 遍历数据类型、二元操作、设备和标量值的组合
            for dtype, op, device, scalar in product(
                self.dtypes, binary_ops, devices, scalars
            ):
                # 跳过不支持在 CPU 上使用 torch.float16 和 torch.bfloat16 的组合
                if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                    continue
                try:
                    # 获取指定数据类型和设备上的输入数据张量 x
                    x = self.data_for(dtype, device)
                    # 创建一个新的操作函数，将标量应用到该操作上
                    fn = apply_with_scalar(op, scalar)
                    # 计算参考结果，即直接使用操作函数 fn 对输入张量 x 进行操作
                    ref = fn(x)
                except Exception:
                    # 如果即时模式不支持给定的数据类型/操作/设备组合，则跳过该组合
                    continue
                try:
                    # 对操作函数 fn 进行 JIT 追踪，以获得优化后的计算图 t
                    t = torch.jit.trace(fn, (x))
                    # 断言追踪后的结果与参考结果一致
                    self.assertEqual(ref, t(x))
                    # 断言所有操作都被融合到一个图中，确保所有操作被优化和组合
                    self.assertAllFused(t.graph_for(x))
                except Exception as e:
                    # 如果发生异常，则抛出运行时错误，提示失败的原因和相关信息
                    raise RuntimeError(
                        " ".join(["Failed:", str(dtype), op.__name__, device])
                    ) from e
    def test_binary_div_ops(self):
        def apply_with_scalar(fn, scalar):
            return lambda x: fn(x, scalar)

        # 定义三种二元操作函数：torch.div, torch.remainder, torch.fmod
        binary_ops = [
            torch.div,
            torch.remainder,
            torch.fmod,
        ]
        # 获取测试中的设备列表
        devices = self.devices
        # 可能需要将此测试拆分成多个单独的测试，以通过仅使用特定操作相关的标量值来加快速度
        scalars = [1.5, 3, -2.0, -1]  # 跳过0
        # 遍历数据类型、操作、设备和标量值的所有组合
        for dtype, op, device, scalar in product(
            self.dtypes, binary_ops, devices, scalars
        ):
            # 如果数据类型为 torch.float16 或 torch.bfloat16，并且设备是 "cpu"，则跳过
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            try:
                # 获取相应数据类型和设备的数据 x
                x = self.data_for(dtype, device)
                # 使用当前操作和标量创建一个新的函数 fn
                fn = apply_with_scalar(op, scalar)
                # 计算参考结果 ref = fn(x)
                ref = fn(x)
            except Exception:
                # 如果在 eager 模式下不支持特定的数据类型、操作或设备组合，则捕获所有异常以避免需要猜测 eager 可能抛出的错误
                continue
            try:
                # 对函数 fn 进行 Torch JIT 追踪，传入参数 (x)
                t = torch.jit.trace(fn, (x))
                # 断言追踪结果 t(x) 与参考结果 ref 相等
                self.assertEqual(ref, t(x))
            except Exception as e:
                # 如果出现异常，抛出自定义的运行时错误，指明失败的数据类型、操作和设备组合
                raise RuntimeError(
                    f"Failed: {dtype} {op.__name__} {device} {scalar}"
                ) from e

    def test_binary_pow(self):
        def apply_with_scalar(fn, scalar):
            return lambda x: fn(x, scalar)

        # 定义数据类型列表，其中包括某些数据类型的问题说明
        dtypes = [
            # FIXME: 'pow' fails with dtype=torch.float16/device=cuda/scalar=0
            # torch.float16,
            torch.float32,
            torch.float64,
            # torch.bool intentionally not included
        ]
        # 定义包含 torch.pow 操作的二元操作列表
        binary_ops = [
            torch.pow,
        ]
        # 可能需要将此测试拆分成多个单独的测试，以通过仅使用特定操作相关的标量值来加快速度
        scalars = [1.5, 3, 0, -2.0, -1]
        # 遍历数据类型、操作、设备和标量值的所有组合
        for dtype, op, device, scalar in product(
            dtypes, binary_ops, self.devices, scalars
        ):
            # 如果数据类型为 torch.float16 或 torch.bfloat16，并且设备是 "cpu"，则跳过
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            try:
                # 获取相应数据类型和设备的数据 x
                x = self.data_for(dtype, device)
                # 使用当前操作和标量创建一个新的函数 fn
                fn = apply_with_scalar(op, scalar)
                # 计算参考结果 ref = fn(x)
                ref = fn(x)
            except Exception:
                # 如果在 eager 模式下不支持特定的数据类型、操作或设备组合，则捕获所有异常以避免需要猜测 eager 可能抛出的错误
                continue
            try:
                # 对函数 fn 进行 Torch JIT 追踪，传入参数 (x)
                t = torch.jit.trace(fn, (x))
                # 断言追踪结果 t(x) 与参考结果 ref 相等
                self.assertEqual(ref, t(x))
                # 断言所有操作是否被融合到一个图中
                self.assertAllFused(t.graph_for(x))
            except Exception as e:
                # 如果出现异常，抛出自定义的运行时错误，指明失败的数据类型、操作和设备
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device])
                ) from e
    # 定义一个测试方法，用于测试三元运算符的功能
    def test_ternary_ops(self):
        # 定义一个函数 apply，接受一个函数 fn 并返回一个 lambda 函数，用于将 fn 应用于三个参数
        def apply(fn):
            return lambda x, y, z: fn(x, y, z)

        # 定义一个包含 torch 库中特定函数的列表 ternary_ops
        ternary_ops = [
            torch.lerp,
            torch.addcmul,
        ]
        # 获取测试环境中的设备列表
        devices = self.devices
        # 遍历所有可能的数据类型、操作函数和设备的组合
        for dtype, op, device in product(self.dtypes, ternary_ops, devices):
            # 如果数据类型是 torch.float16 或 torch.bfloat16 并且设备是 "cpu"，则跳过当前组合
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            try:
                # 根据当前数据类型和设备生成数据 x, y, z
                x = self.data_for(dtype, device)
                y = self.data_for(dtype, device)
                z = self.data_for(dtype, device)
                # 应用操作函数 op 到 x, y, z，生成参考结果 ref
                fn = apply(op)
                ref = fn(x, y, z)
            except Exception:
                # 如果在尝试生成数据或应用操作时出现异常，继续下一个组合的测试
                # 在 eager 模式下不支持特定的数据类型、操作或设备组合时，捕获所有异常以避免猜测可能抛出的错误
                continue
            try:
                # 将函数 fn 进行 JIT 追踪，以加速后续的执行
                t = torch.jit.trace(fn, (x, y, z))
                # 断言追踪后的函数 t 对 x, y, z 的执行结果与 ref 相等
                self.assertEqual(ref, t(x, y, z))
                # 断言追踪后的函数 t 的计算图中所有操作都已融合（即由 JIT 进行了优化）
                self.assertAllFused(t.graph_for(x, y, z))
            except Exception as e:
                # 如果在 JIT 追踪或断言过程中出现异常，则抛出 RuntimeError
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device])
                ) from e

    # 定义另一个测试方法，用于测试三元归一化操作符的功能
    def test_ternary_norm_ops(self):
        # 定义一个函数 apply，接受一个函数 fn 并返回一个 lambda 函数，用于将 fn 应用于三个参数
        def apply(fn):
            return lambda x, y, z: fn(x, y, z)

        # 定义一个包含 torch.nn.functional 库中特定函数的列表 ternary_ops
        ternary_ops = [
            F.batch_norm,
        ]
        # 获取测试环境中的设备列表
        devices = self.devices
        # 遍历所有可能的数据类型、操作函数和设备的组合
        for dtype, op, device in product(self.dtypes, ternary_ops, devices):
            # 如果数据类型是 torch.float16 或 torch.bfloat16 并且设备是 "cpu"，则跳过当前组合
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            try:
                # 根据当前数据类型和设备生成数据 x, y, z
                x = self.data_for(dtype, device, size=[5, 3, 128, 128])
                y = self.data_for(dtype, device, size=[3])
                z = self.data_for(dtype, device, size=[3])
                # 应用操作函数 op 到 x, y, z，生成参考结果 ref
                fn = apply(op)
                ref = fn(x, y, z)
            except Exception:
                # 如果在尝试生成数据或应用操作时出现异常，继续下一个组合的测试
                # 在 eager 模式下不支持特定的数据类型、操作或设备组合时，捕获所有异常以避免猜测可能抛出的错误
                continue
            try:
                # 将函数 fn 进行 JIT 追踪，以加速后续的执行
                t = torch.jit.trace(fn, (x, y, z))
                # 断言追踪后的函数 t 对 x, y, z 的执行结果与 ref 相等
                self.assertEqual(ref, t(x, y, z))
                # 断言追踪后的函数 t 的计算图中所有操作都已融合（即由 JIT 进行了优化）
                self.assertAllFused(t.graph_for(x, y, z))
            except Exception as e:
                # 如果在 JIT 追踪或断言过程中出现异常，则抛出 RuntimeError
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device])
                ) from e

    # 标记以下测试方法为跳过测试，因为融合器不包括 ListConstruct 节点到组，导致测试失败
    @unittest.skip(
        "FIXME: fuser doesn't include ListConstruct nodes to the group causing a failure"
    )
    # 定义测试方法，用于测试列表操作
    def test_list_ops(self):
        # 定义一个函数 apply，接受一个函数 fn，并返回一个新函数，该新函数接受三个参数并将 fn 应用到这些参数的列表上
        def apply(fn):
            return lambda x, y, z: fn([x * x, y * y, z * z])

        # 将 self.devices 赋值给 devices
        devices = self.devices
        # 定义一个包含 torch.cat 函数的列表 list_ops
        list_ops = [
            torch.cat,
        ]
        # 遍历 self.dtypes、list_ops 和 devices 的笛卡尔积
        for dtype, op, device in product(self.dtypes, list_ops, devices):
            # 如果 dtype 是 torch.float16 或 torch.bfloat16 并且 device 是 "cpu"，则跳过本次循环
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            try:
                # 分别用指定的 dtype、device 和大小创建数据 x、y、z
                x = self.data_for(dtype, device, size=[5, 4, 1, 7])
                y = self.data_for(dtype, device, size=[5, 4, 1, 7])
                z = self.data_for(dtype, device, size=[5, 4, 1, 7])
                # 将 op 应用到 x、y、z 上，并赋值给 fn
                fn = apply(op)
                # 计算参考值 ref
                ref = fn(x, y, z)
            except Exception:
                # 如果出现异常，捕获并继续下一次循环
                # 如果 eager 模式不支持给定的 dtype/op/device 组合，那么 fuser 也不会支持。捕获所有可能抛出的异常，避免猜测可能的错误。
                continue
            try:
                # 对 fn 使用 torch.jit.trace 进行追踪，传入参数 (x, y, z)
                t = torch.jit.trace(fn, (x, y, z))
                # 断言追踪的结果与 ref 相等
                self.assertEqual(ref, t(x, y, z))
                # 断言所有的操作都被融合到一个图中
                self.assertAllFused(t.graph_for(x, y, z))
            except Exception as e:
                # 如果出现异常，抛出 RuntimeError 异常，附带错误信息
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device])
                ) from e

    # 定义测试方法，用于测试 torch.where 相关操作
    def test_where_ops(self):
        # 定义一个函数 apply，接受一个函数 fn，并返回一个新函数，该新函数接受三个参数并将 fn 应用到这些参数上
        def apply(fn):
            return lambda cond, x, y: fn(cond, x, y)

        # 定义包含不同 torch.where 操作和匿名函数的列表 ops
        ops = [
            torch.where,
            lambda cond, x, y: torch.where(cond, x, 3.1415),
            lambda cond, x, y: torch.where(cond, 42, y),
        ]
        # 将 self.devices 赋值给 devices
        devices = self.devices
        # 遍历 self.dtypes、ops 和 devices 的笛卡尔积
        for dtype, op, device in product(self.dtypes, ops, devices):
            # 如果 dtype 是 torch.float16 或 torch.bfloat16 并且 device 是 "cpu"，则跳过本次循环
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            try:
                # 使用指定的 dtype 和 device 创建数据 cond、x、y
                cond = self.data_for(torch.bool, device)
                x = self.data_for(dtype, device)
                y = self.data_for(dtype, device)
                # 将 op 应用到 cond、x、y 上，并赋值给 fn
                fn = apply(op)
                # 计算参考值 ref
                ref = fn(cond, x, y)
            except Exception:
                # 如果出现异常，捕获并继续下一次循环
                # 如果 eager 模式不支持给定的 dtype/op/device 组合，那么 fuser 也不会支持。捕获所有可能抛出的异常，避免猜测可能的错误。
                continue
            try:
                # 对 fn 使用 torch.jit.trace 进行追踪，传入参数 (cond, x, y)
                t = torch.jit.trace(fn, (cond, x, y))
                # 断言追踪的结果与 ref 相等
                self.assertEqual(ref, t(cond, x, y))
                # 断言所有的操作都被融合到一个图中
                self.assertAllFused(t.graph_for(cond, x, y))
            except Exception as e:
                # 如果出现异常，抛出 RuntimeError 异常，附带错误信息
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device])
                ) from e
    def test_unsupported_dtypes(self):
        # 遍历设备列表进行测试
        for device in self.devices:

            # 定义一个简单的函数 fn，计算输入 x 的平方加 x 的结果
            def fn(x):
                return x * x + x

            # 定义不支持的数据类型列表
            unsupported_dtypes = [
                torch.uint8,        # 无符号 8 位整数
                torch.complex32,    # 32 位复数
                torch.complex64,    # 64 位复数
                torch.complex128,   # 128 位复数
                torch.qint8,        # 量化整数 8 位
                torch.quint8,       # 无符号量化整数 8 位
                torch.qint32,       # 量化整数 32 位
            ]
            # 遍历不支持的数据类型进行测试
            for dtype in unsupported_dtypes:
                try:
                    # 获取特定数据类型和设备的测试数据 x
                    x = self.data_for(dtype, device)
                    # 计算参考结果 ref
                    ref = fn(x)
                except Exception:
                    # 如果 eager 模式不支持特定的数据类型/操作/设备组合，
                    # 那么融合器也不支持。捕获所有异常以避免需要猜测 eager 可能抛出的错误。
                    continue

                # 使用 torch.jit.trace 对函数 fn 进行跟踪
                t = torch.jit.trace(fn, (x,))
                # 断言跟踪后的结果与参考结果相等
                self.assertEqual(ref, t(x))
                # 断言融合组数量为 0
                self.assertEqual(len(self.findFusionGroups(t.graph_for(x))), 0)

    def test_superslomo(self):
        # 复制设备列表并排除不支持 LLVM 的 CPU 设备
        devices = self.devices.copy()
        if not LLVM_ENABLED:
            devices.remove("cpu")
        # 遍历设备列表进行测试
        for device in devices:
            # 定义 Super-SloMo 中提取的测试函数 eager
            def eager(t0, t1, t2, t3, t4):
                t5 = torch.mul(t0, t4)       # 元素级乘法
                t6 = torch.mul(t2, t3)       # 元素级乘法
                t7 = torch.mul(t6, t1)       # 元素级乘法
                t9 = torch.add(t5, t7)       # 元素级加法
                t11 = torch.add(t0, t6)      # 元素级加法
                ft_p = torch.div(t9, t11)    # 元素级除法
                return (ft_p, t11, t9, t6)

            # 生成随机张量作为输入
            t0 = torch.rand(1, 6, 352, 352, device=device).transpose(0, 1)
            t1 = torch.rand(6, 3, 352, 352, device=device)
            t2 = torch.rand(6, device=device)[None, None, None, :].permute(3, 0, 1, 2)
            t3 = torch.rand(6, 1, 352, 352, device=device)
            t4 = torch.rand(6, 3, 352, 352, device=device)
            inputs = [t0, t1, t2, t3, t4]

            # 使用 torch.jit.script 将函数 eager 脚本化
            script = torch.jit.script(eager)
            # 进行多次测试以验证脚本化函数的输出与原始函数输出一致
            for _ in range(4):
                for pair in zip(script(*inputs), eager(*inputs)):
                    test, ref = pair
                    # 断言脚本化函数的输出与原始函数的输出非常接近
                    torch.testing.assert_close(test, ref)
                    # 断言所有融合组均为 "prim::TupleConstruct"
                    self.assertAllFused(
                        script.graph_for(*inputs), except_for={"prim::TupleConstruct"}
                    )
    # 定义一个名为 test_sub_gt_and 的测试方法
    def test_sub_gt_and(self):
        # 遍历 self.devices 中的设备列表
        for device in self.devices:

            # 定义一个内部函数 eager，接收五个参数 t1, t2, t3, t4, t，返回 w 或 k
            def eager(t1, t2, t3, t4, t: float):
                # 计算差值 w 和 h
                w = t1 - t2
                h = t3 - t4
                # 判断 w 和 h 是否大于 t，返回一个逻辑与运算的结果 k
                k = (w > t) & (h > t)
                # 断言 k 的数据类型为 torch.bool
                assert k.dtype == torch.bool
                if t > 0.5:
                    # 如果 t 大于 0.5，返回 k 加 1 的结果
                    # 通过在永不执行的条件语句中使用 k，防止对其类型进行分析，保持其为 "Tensor"。
                    # 如果将 "Tensor" 传播回 k 的定义处，必须小心不要创建包含它的融合组。
                    return k + 1
                # 否则返回 w 的值
                return w

            # 生成一个随机的大小为 8 的 torch.Tensor t，数据类型为 torch.float，放在指定的设备上
            t = torch.rand(8, dtype=torch.float, device=device)
            # 对 eager 函数进行脚本化，并使用 checkScript 方法检查脚本化结果
            scripted = self.checkScript(eager, (t, t, t, t, 0.1))

    # 如果 TorchDynamo 太慢，则跳过执行以下测试方法
    @skipIfTorchDynamo("too slow")
    def test_chunk_mul_one(self):
        # 如果 self.dynamic_shapes 为 True，则跳过该测试方法
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        # 遍历 self.devices 中的设备列表
        for device in self.devices:

            # 定义一个内部函数 eager，接收一个参数 x，将 x 按照指定维度和数量分块
            def eager(x):
                z, y, w = torch.chunk(x, 3, -1)
                # 返回分块后的结果 z * 3, y, w
                return z * 3, y, w

            # 生成一个随机的大小为 (64, 1, 3072) 的 torch.Tensor x，数据类型为 torch.float，放在指定的设备上
            x = torch.rand(64, 1, 3072, dtype=torch.float, device=device)
            # 对 eager 函数进行脚本化，并使用 checkScript 方法检查脚本化结果
            z, y, w = eager(x)
            script = self.checkScript(eager, (x,))

    # 定义一个测试方法 test_eq_unsqueeze_type_as
    def test_eq_unsqueeze_type_as(self):
        # 遍历 self.devices 中的设备列表
        for device in self.devices:

            # 定义一个内部函数 eager，接收两个参数 a 和 b
            def eager(a, b):
                # 判断 b 中是否等于 1，生成一个掩码 mask
                mask = b == 1
                # 在指定维度 -1 上对 mask 进行 unsqueeze 操作
                mask = torch.unsqueeze(mask, -1)
                # 将 mask 的数据类型转换为和 a 一致的类型，并赋值给 x
                x = mask.type_as(a)
                # 返回 x 和 mask
                return x, mask

            # 生成一个随机的大小为 (1, 64, 1024) 的 torch.Tensor a，数据类型为 torch.float，放在指定的设备上
            a = torch.rand(1, 64, 1024, device=device, dtype=torch.float)
            # 生成一个随机整数大小为 (1, 64) 的 torch.Tensor b，数据类型为 torch.long，放在指定的设备上
            b = torch.randint(-2, 2, (1, 64), device=device, dtype=torch.long)
            # 对 eager 函数进行脚本化，并使用 checkScript 方法检查脚本化结果
            script = self.checkScript(eager, (a, b))

    # 定义一个测试方法 test_neg_pow
    def test_neg_pow(self):
        # 定义一个内部函数 eager_tt，接收两个 torch.Tensor 类型的参数 a 和 b
        def eager_tt(a: torch.Tensor, b: torch.Tensor):
            # 计算 a 的 b 次幂并取负
            return torch.neg(torch.pow(a, b))

        # 定义一个内部函数 eager_ts，接收一个 torch.Tensor 类型的参数 a 和一个 float 类型的参数 b
        def eager_ts(a: torch.Tensor, b: float):
            # 计算 a 的 b 次幂并取负
            return torch.neg(torch.pow(a, b))

        # 定义一个内部函数 eager_st，接收一个 float 类型的参数 a 和一个 torch.Tensor 类型的参数 b
        def eager_st(a: float, b: torch.Tensor):
            # 计算 a 的 b 次幂并取负
            return torch.neg(torch.pow(a, b))

        # 生成一个随机的大小为 (1,) 的 torch.Tensor a，数据类型为 torch.float
        a = torch.rand(1, dtype=torch.float)
        # 生成一个随机的大小为 (1,) 的 torch.Tensor b，数据类型为 torch.float
        b = torch.rand(1, dtype=torch.float)
        # 获取 b 的标量值 s
        s = b.item()
        # 对 eager_tt 函数进行脚本化，并使用 checkScript 方法检查脚本化结果
        script = self.checkScript(eager_tt, (a, b))
        # TODO: 重新启用融合，目前不起作用。现在只测试正确性
        # self.assertAllFused(script.graph_for(a, b))
        # 对 eager_ts 函数进行脚本化，并使用 checkScript 方法检查脚本化结果
        script = self.checkScript(eager_ts, (a, s))
        # self.assertAllFused(script.graph_for(a, s))
        # 对 eager_st 函数进行脚本化，并使用 checkScript 方法检查脚本化结果
        script = self.checkScript(eager_st, (s, b))
        # self.assertAllFused(script.graph_for(s, b))

    # 如果 LLVM_ENABLED 不为真，则跳过执行以下测试方法
    @unittest.skipIf(not LLVM_ENABLED, "Too slow to run with the TE interpreter")
    def test_conv2d_depthwise(self):
        # 如果支持动态形状，则跳过测试
        if self.dynamic_shapes:
            self.skipTest("don't run conv with dynamic shapes")

        # 定义一个 eager 函数，执行深度卷积操作
        def eager(input, weight, bias):
            return torch.conv2d(input, weight, bias, stride=1, padding=1, groups=72)

        # 创建一个随机输入张量，形状为 (1, 72, 56, 56)，数据类型为 float
        input = torch.rand((1, 72, 56, 56), dtype=torch.float)
        # 创建一个随机卷积核张量，形状为 (72, 1, 3, 3)，数据类型为 float
        weight = torch.rand((72, 1, 3, 3), dtype=torch.float)
        # 创建一个随机偏置张量，形状为 (72)，数据类型为 float
        bias = torch.rand((72), dtype=torch.float)

        # 检查 eager 函数的脚本表示，并返回其结果
        script = self.checkScript(eager, (input, weight, bias))
        # 断言所有操作都被融合到一个图中
        self.assertAllFused(script.graph_for(input, weight, bias))

    def test_conv2d(self):
        # 如果支持动态形状，则跳过测试
        if self.dynamic_shapes:
            self.skipTest("don't run conv with dynamic shapes")

        # 定义一个 eager 函数，执行普通卷积操作
        def eager(input, weight, bias):
            return torch.conv2d(input, weight, bias, stride=1, padding=1, groups=1)

        # 创建一个随机输入张量，形状为 (1, 64, 56, 56)，数据类型为 float
        input = torch.rand((1, 64, 56, 56), dtype=torch.float)
        # 创建一个随机卷积核张量，形状为 (64, 64, 3, 3)，数据类型为 float
        weight = torch.rand((64, 64, 3, 3), dtype=torch.float)
        # 创建一个随机偏置张量，形状为 (64)，数据类型为 float
        bias = torch.rand((64), dtype=torch.float)

        # 检查 eager 函数的脚本表示，并使用 FileCheck 检查优化图中是否没有 "TensorExpr"
        FileCheck().check_not("TensorExpr").run(
            torch.jit.last_executed_optimized_graph()
        )

    def test_type_as_cat(self):
        # 使用内联融合组

        # 定义一个 eager 函数，执行 torch.cat 操作
        def eager(x, y):
            return torch.cat((x, y.type_as(x)), dim=1)

        # 复制数据类型列表
        dtypes = self.dtypes.copy()
        # 移除不支持的数据类型 torch.float16 和 torch.bfloat16
        dtypes.remove(torch.float16)
        dtypes.remove(torch.bfloat16)
        # 对于每一对数据类型组合 (dtype1, dtype2)
        for dtype1, dtype2 in product(dtypes, dtypes):
            # 创建一个随机整数张量 x，形状为 (1, 13)，数据类型为 dtype1
            x = torch.randint(
                2,
                (
                    1,
                    13,
                ),
            ).to(dtype1)
            # 创建一个形状为 (1, 1) 的零张量 zero，数据类型为 dtype2
            zero = torch.tensor([[0]]).to(dtype2)
            # 创建一个形状为 (1, 1) 的一张量 one，数据类型为 dtype2
            one = torch.tensor([[1]]).to(dtype2)
            # 跟踪 eager 函数的脚本表示
            script = torch.jit.trace(eager, (x, zero))
            # 执行多次断言，验证脚本输出与 eager 函数输出的接近程度
            for _ in range(3):
                torch.testing.assert_close(script(x, zero), eager(x, zero))
                torch.testing.assert_close(script(x, one), eager(x, one))
            # 断言所有操作都被融合到一个图中
            self.assertAllFused(script.graph_for(x, one))

    def test_to_device(self):
        # 定义一个 eager 函数，将张量移动到 CPU 设备并应用 ReLU 激活函数
        def eager(x):
            return x.to(device="cpu").relu()

        # 创建一个随机张量 x，形状为 (8)
        x = torch.rand(8)
        # 检查 eager 函数的脚本表示
        script = self.checkScript(eager, (x,))
        # 断言所有操作都被融合到一个图中
        self.assertAllFused(script.graph_for(x))

    def test_dims(self):
        # 定义一个 eager 函数，对 x 和 y 执行除法操作
        def eager(x, y):
            return x / (y + 0.0001)

        # 创建一个线性空间张量 x，从 -1 到 1，共 768 个元素，数据类型为 torch.float32
        x = torch.linspace(-1, 1, 768, dtype=torch.float32).as_strided(
            (1, 1, 768), (768, 1, 1)
        )
        # 创建一个形状为 (1, 1, 1) 的张量 y，数值为 2.0，数据类型为 torch.float32
        y = torch.tensor([[[2.0]]], dtype=torch.float32)
        # 检查 eager 函数的脚本表示
        script = self.checkScript(eager, (x, y))
        # 断言所有操作都被融合到一个图中
        self.assertAllFused(script.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # 定义一个测试方法，用于测试动态通道最后维度的情况
    def test_channels_last_dims_dynamic(self):
        # 定义一个函数 eager，对两个输入进行加法操作，并添加一个微小的常数
        def eager(x, y):
            return x + (y + 0.0001)

        # 定义一个索引列表
        indices = [0, 1, 2, 3]
        # 初始化空集合用于存放索引的子集
        sets = []
        # 遍历索引列表长度加一的范围
        for i in range(0, len(indices) + 1):
            # 遍历所有的组合子集
            for subset in combinations(indices, i):
                # 将每个子集添加到 sets 列表中，不考虑性能问题
                sets.append(subset)  # noqa: PERF402

        # 遍历所有子集
        for set in sets:
            # 初始化大小为 [2, 3, 4, 5] 的列表
            size = [2, 3, 4, 5]
            # 根据当前子集设置对应索引的大小为1
            for index in set:
                size[index] = 1
            # 创建一个随机填充的张量，格式为通道最后，并移到 GPU 上
            inp = torch.rand(size).to(memory_format=torch.channels_last).cuda()
            # 启用动态策略
            with texpr_enable_strategy([("DYNAMIC", 20)]):
                # 对 eager 函数进行 Torch 脚本跟踪
                foo_s = torch.jit.trace(eager, (inp, inp))
                # 多次调用 foo_s 函数
                for _ in range(3):
                    out = foo_s(inp, inp)
                # 直接调用 eager 函数以备比较
                out_eager = eager(inp, inp)
                # 断言两种方法的输出相等
                self.assertEqual(out_eager, out)
                # 断言输出张量在通道最后格式下是连续的
                self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
                # 获取最近一次执行的优化图
                g = torch.jit.last_executed_optimized_graph()
                # 使用 FileCheck 验证图中是否包含 "TensorExpr"
                FileCheck().check("TensorExpr").run(g)

    # 定义一个测试方法，用于测试特定规格的耗尽情况
    def test_exhaust_specializations(self):
        # 启用静态策略
        with texpr_enable_strategy([("STATIC", 1)]):

            # 定义一个 Torch 脚本函数 foo，实现对输入的三倍加法
            @torch.jit.script
            def foo(x):
                return x + x + x

            # 多次调用 foo 函数，对大小为 [2, 2] 的张量进行计算
            for _ in range(3):
                foo(torch.rand([2, 2]))

            # 多次调用 foo 函数，对大小为 [4, 4, 4] 的张量进行计算
            for _ in range(3):
                foo(torch.rand([4, 4, 4]))

            # 获取最近一次执行的优化图
            g = torch.jit.last_executed_optimized_graph()
            # 执行 JIT 内联优化
            torch._C._jit_pass_inline(g)
            # 使用 FileCheck 验证图中是否恰好包含两次 "TensorExpr" 出现
            FileCheck().check_count("TensorExpr", 2, exactly=True).run(g)

    # 定义一个测试方法，用于测试 unsqueeze 操作中可变维度的情况
    def test_unsqueeze_var_dim(self):
        # 定义一个函数 eager，实现 x 乘以在维度 z 上 unsqueeze 的 y
        def eager(x, y, z: int):
            return x * torch.unsqueeze(y, dim=z)

        # 创建大小为 [4, 4, 64] 的随机张量，并按照指定顺序排列
        x = torch.rand(4, 4, 64).permute(1, 0, 2)
        # 创建大小为 [4, 4] 的随机张量 y
        y = torch.rand(4, 4)
        # 设定 z 的值为 2
        z = 2
        # 对 eager 函数进行 Torch 脚本检查
        script = self.checkScript(eager, (x, y, z))

    # 定义一个测试方法，用于测试前向和后向传播的基本情况
    def _test_fwd_bwd(self, fn):
        # 创建一个从 -10 到 10 的浮点数张量 x，需要梯度计算
        x = torch.arange(-10, 10, dtype=torch.float32, requires_grad=True)
        # 创建一个从 -10 到 10 的浮点数张量 xs，需要梯度计算
        xs = torch.arange(-10, 10, dtype=torch.float32, requires_grad=True)
        # 对传入的函数 fn 进行 Torch 脚本化
        script = torch.jit.script(fn)
        # 进行 11 次循环
        for i in range(11):
            # 调用函数 fn 计算输出 y
            y = fn(x)
            # 创建一个与 y 形状相同的随机张量 g0
            g0 = torch.rand_like(y)
            # 对 y 进行反向传播
            y.backward(g0)

            # 使用 Torch 脚本化的 fn 计算输出 ys
            ys = script(xs)
            # 对 ys 进行反向传播
            ys.backward(g0)

            # 在无梯度更新的上下文中
            with torch.no_grad():
                # 更新 x 和 xs 的值
                x -= 0.1 * x.grad
                xs -= 0.1 * xs.grad
                # 清空 x 和 xs 的梯度信息
                x.grad = None
                xs.grad = None
        # 使用 Torch 提供的测试函数检验 y 和 ys 是否近似相等
        torch.testing.assert_close(y, ys)

    # 定义一个测试方法，用于测试 ReLU 函数的前向和后向传播
    def test_relu_fwd_bwd(self):
        # 定义一个函数 eager，实现对输入的 ReLU 函数调用
        def eager(x):
            return torch.relu(x * 1.01)

        # 调用 _test_fwd_bwd 方法进行测试
        self._test_fwd_bwd(eager)

    # 定义一个测试方法，用于测试硬切线函数的前向和后向传播
    def test_hardswish_fwd_bwd(self):
        # 定义一个函数 eager，实现对输入的硬切线函数调用，并乘以一个常数
        def eager(x):
            return F.hardswish(x) * 1.01

        # 调用 _test_fwd_bwd 方法进行测试
        self._test_fwd_bwd(eager)

    # 定义一个测试方法，用于测试硬 Sigmoid 函数的前向和后向传播
    def test_hardsigmoid_fwd_bwd(self):
        # 定义一个函数 eager，实现对输入的硬 Sigmoid 函数调用，并乘以一个常数
        def eager(x):
            return F.hardsigmoid(x) * 1.01

        # 调用 _test_fwd_bwd 方法进行测试
        self._test_fwd_bwd(eager)
    def test_cat_graph_opt(self):
        # 定义一个内部函数 foo，接收三个参数 x, y, z，返回它们在 torch.cat 上的对数
        def foo(x, y, z):
            return torch.log(torch.cat([x, y, z]))

        # 使用 self.checkScript 检查脚本化后的 foo 函数
        self.checkScript(
            foo, (torch.rand([5, 5]), torch.rand([2, 5]), torch.rand([1, 5]))
        )
        # TODO: 不确定为什么未更新的图没有反映在 last_optimized_graph 中
        self.assertLastGraphAllFused()

    def test_dynamic_cat(self):
        # 使用 inline_fusion_groups 上下文管理器进行内联融合组操作
        with inline_fusion_groups():

            @torch.jit.script
            # 定义函数 repro，接收三个类型为 List[torch.Tensor] 的参数列表 xs, ys, zs
            def repro(
                xs: List[torch.Tensor], ys: List[torch.Tensor], zs: List[torch.Tensor]
            ):
                # 返回一个列表，其中每个元素是 torch.cat 操作的结果
                return [
                    torch.cat([x, torch.cat([y, z], dim=-1)], dim=-1)
                    for x, y, z in zip(xs, ys, zs)
                ]

            # 循环执行三次
            for _ in range(3):
                N = 3
                # 创建长度为 N 的 xs 列表，每个元素是 torch.ones(21)
                xs = [torch.ones(21) for _ in range(N)]
                # 创建长度为 N 的 ys 列表，每个元素是 torch.ones(N-i)
                ys = [torch.ones(N - i) for i in range(N)]
                # 创建长度为 N 的 zs 列表，每个元素是 torch.ones(i)
                zs = [torch.ones(i) for i in range(N)]
                # 调用 repro 函数
                repro(xs, ys, zs)

    def test_scalar_only_inputs(self):
        # 定义函数 eager，接收一个标量参数 b，并返回 torch.ones(1) 与 b 的乘积
        def eager(b: float):
            a = torch.ones(1)
            return a * b

        # 使用 self.checkScript 检查脚本化后的 eager 函数
        script = self.checkScript(eager, (1.0,))

    def test_cat_2k_args(self):
        # 使用 inline_fusion_groups 上下文管理器进行内联融合组操作
        with inline_fusion_groups():

            # 定义函数 eager，接收一个参数 x，返回 torch.relu(torch.cat([x for _ in range(2000)]))
            def eager(x):
                return torch.relu(torch.cat([x for _ in range(2000)]))

            # 创建一个随机的 torch.Tensor x
            x = torch.randn(1)
            # 使用 self.checkTrace 检查 eager 函数的追踪版本
            trace = self.checkTrace(eager, (x,))
            # 查找追踪图中的融合组
            fusion_groups = self.findFusionGroups(trace.graph_for(x))
            # 断言融合组的数量为 0
            self.assertEqual(len(fusion_groups), 0)

    def test_adaptive_avg_pool2d(self):
        # TODO: 一旦 OpInfo DB 中支持 adaptive_avg_pool2d，应将此测试移至那里
        # 使用 inline_fusion_groups 上下文管理器进行内联融合组操作
        with inline_fusion_groups():

            # 定义函数 foo1，接收一个参数 x，返回 torch.nn.functional.adaptive_avg_pool2d(x, (2, 2))
            def foo1(x):
                return torch.nn.functional.adaptive_avg_pool2d(x, (2, 2))

            # 定义函数 foo2，接收一个参数 x，返回 torch.nn.functional.adaptive_avg_pool2d(x, (2))
            def foo2(x):
                return torch.nn.functional.adaptive_avg_pool2d(x, (2))

            # 创建一个大小为 (4, 4, 4) 的随机 torch.Tensor x
            x = torch.randn(4, 4, 4)
            # 对 foo1 和 foo2 中的每个函数进行追踪
            for foo in [foo1, foo2]:
                f = torch.jit.trace(foo, (x,))
                # 创建一个 TensorExprKernel 对象 kernel，并运行其结果与正确值进行比较
                kernel = torch._C._te.TensorExprKernel(f.graph)
                correct_val = f(x)
                self.assertEqual(kernel.run((x,)), correct_val)
    # 定义测试函数 test_unrolled_cat
    def test_unrolled_cat(self):
        # 启用内联融合组上下文

        # 定义内部函数 eager，对输入张量 x 中的每个元素进行 relu 操作并拼接到一个张量中返回
        def eager(x):
            ret = torch.empty(0)
            for i in range(x.shape[0]):
                ret = torch.cat([ret, x[i].relu()])
            return ret

        # 对 eager 函数进行 Torch 脚本编译
        script = torch.jit.script(eager)

        # 预热，使用大小为 1 的张量进行多次调用，使得循环路径在 profile 数据中“烧入”，然后展开
        x = torch.ones(1, 1)
        for _ in range(3):
            script(x)

        # 断言 eager 函数对于输入 x 和脚本化的 eager 函数在数值上相等
        torch.testing.assert_close(eager(x), script(x))

        # 使用大小为 (8, 1) 的张量进行调用，检验展开路径的运行结果，此时因为大小为 1 的情况已经烧入，可能导致张量尺寸不正确
        x = torch.ones((8, 1))
        torch.testing.assert_close(eager(x), script(x))

    # 跳过 Torch Dynamo 测试，因为速度太慢
    @skipIfTorchDynamo("too slow")
    # 如果在 ASAN（地址无关的存储器错误检测工具）下测试，因为耗时超过 10 分钟
    @unittest.skipIf(TEST_WITH_ASAN, "takes 10+ minutes on asan")
    # 如果在 ROCm 平台下测试，因为张量的相似性对于 NaN 值不适用
    @unittest.skipIf(TEST_WITH_ROCM, "Tensor-likes are not close for nans")
    # 定义批归一化测试函数 test_batch_norm
    def test_batch_norm(self):
        # 定义测试函数 test，对给定函数 fn 和参数 args 进行 Torch 脚本追踪，并断言其融合了所有操作
        def test(fn, args):
            trace = torch.jit.trace(fn, args)
            self.assertAllFused(trace.graph_for(*args))
            # TODO: 这里的 NaN 是否正常？或者之前默默地通过了，因为 `equal_nan=True` 是默认值？
            # 断言使用相等性检查测试函数和其脚本化版本的输出
            torch.testing.assert_close(fn(*args), trace(*args), equal_nan=True)

        # 定义批归一化函数 bn，对输入 i 和 x 执行批归一化操作，然后 relu
        def bn(i, x):
            return torch.batch_norm(i, x, x, x, x, False, 0.1, 1e-4, False).relu()

        # 定义无权重的批归一化函数 bn_no_weight，只对输入 x 执行批归一化操作，然后 relu
        def bn_no_weight(i, x):
            return torch.batch_norm(i, None, x, x, x, False, 0.1, 1e-4, False).relu()

        # 定义无偏置的批归一化函数 bn_no_bias，只对输入 x 执行批归一化操作，然后 relu
        def bn_no_bias(i, x):
            return torch.batch_norm(i, x, None, x, x, False, 0.1, 1e-4, False).relu()

        # 定义无权重和偏置的批归一化函数 bn_neither，只对输入 x 执行批归一化操作，然后 relu
        def bn_neither(i, x):
            return torch.batch_norm(i, None, None, x, x, False, 0.1, 1e-4, False).relu()

        # 针对测试设备列表中的每个设备进行测试
        for device in self.devices:
            # 创建随机张量 i 和 x
            i = torch.randn(4, 16, 32, 40, device=device)
            x = torch.randn(16, device=device)
            # 对所有批归一化函数进行测试
            for fn in [bn, bn_no_weight, bn_no_bias, bn_neither]:
                test(fn, (i, x))

    # 定义性能分析器测试函数 test_profiler
    def test_profiler(self):
        # 定义 Torch 脚本函数 test，执行输入 x、y 和 z 的乘加操作并返回结果
        @torch.jit.script
        def test(x, y, z):
            return x * y + z

        # 创建随机张量列表 args，每个张量包含 4 个随机元素
        args = [torch.randn(4) for _ in range(3)]
        # 使用 Torch 自动求导性能分析器进行测试
        with torch.autograd.profiler.profile() as prof:
            for _ in range(3):
                test(*args)
        # 断言性能分析结果中包含 "fused_mul_add"
        self.assertIn("fused_mul_add", prof.table())

    # 定义跳过梯度检查测试函数 test_skip_grad_in_check
    def test_skip_grad_in_check(self):
        # 定义 Torch 脚本函数 foo，对输入 x 执行加 2 后除以 2 的操作并返回结果
        @torch.jit.script
        def foo(x):
            return (x + 2) / 2

        # 创建随机输入张量 inp，形状为 [4, 4]
        inp = torch.rand([4, 4])
        # 多次调用 Torch 脚本化函数 foo
        for _ in range(3):
            foo(inp)

        # 将输入张量 inp 设置为需要梯度计算
        inp.requires_grad_(True)
        # 使用 Torch 推理模式进行测试
        with torch.inference_mode():
            for _ in range(3):
                foo(inp)
        # 获取最后一次优化过的图形 g
        g = torch.jit.last_executed_optimized_graph()
        # 对图形 g 进行两次内联优化
        torch._C._jit_pass_inline(g)
        torch._C._jit_pass_inline(g)
        # 运行 FileCheck 检查 g 中的 "prim::If" 操作数目是否为 1
        FileCheck().check_count("prim::If", 1, exactly=True).run(g)
    # 如果未启用 CUDA，则跳过测试；这个测试要求半精度的 NNC 融合需要 CUDA 支持
    @unittest.skipIf(not RUN_CUDA, "half-precision NNC fusion requires CUDA")
    def test_autocast_up(self):
        def f(x):
            # 将输入张量转换为全精度，并启用自动类型转换
            y = x._autocast_to_full_precision(True, True)
            # 对全精度张量应用指数函数
            z = torch.exp(y)
            return z

        # 在 CUDA 设备上生成一个随机的半精度张量
        x = torch.rand((2, 2), dtype=torch.half, device="cuda")
        # 对函数 f 进行脚本化编译
        scr = torch.jit.script(f)
        # 调用脚本化的函数两次
        scr(x)
        scr(x)
        # 断言最后生成的计算图中所有操作均被融合
        self.assertLastGraphAllFused()

    # 如果未启用 CUDA，则跳过测试；这个测试要求半精度的 NNC 融合需要 CUDA 支持
    @unittest.skipIf(not RUN_CUDA, "half-precision NNC fusion requires CUDA")
    def test_autocast_down(self):
        def f(x):
            # 对输入张量应用 sigmoid 函数
            y = torch.sigmoid(x)
            # 将结果张量转换为降低精度，并启用自动类型转换
            z = y._autocast_to_reduced_precision(True, True, torch.half, torch.half)
            return z

        # 在 CUDA 设备上生成一个随机的单精度张量
        x = torch.rand((2, 2), dtype=torch.float, device="cuda")
        # 对函数 f 进行脚本化编译
        scr = torch.jit.script(f)
        # 调用脚本化的函数两次
        scr(x)
        scr(x)
        # 断言最后生成的计算图中所有操作均被融合
        self.assertLastGraphAllFused()

    # 如果未启用 LLVM，则跳过测试；这个测试需要 TensorExprKernel 进行编译
    @unittest.skipIf(not LLVM_ENABLED, "Compiles with TensorExprKernel")
    def test_to_dtype(self):
        def f(x):
            # 对输入张量应用 sigmoid 函数
            y = torch.sigmoid(x)
            # 将结果张量转换为降低精度，并启用自动类型转换到半精度
            z = y._autocast_to_reduced_precision(True, True, torch.half, torch.bfloat16)
            # 将降低精度的结果张量转换回全精度
            h = z._autocast_to_full_precision(True, True)
            # 将全精度张量转换为 bfloat16 类型
            i = h.to(dtype=torch.bfloat16)
            # 将 bfloat16 类型的张量转换为单精度
            j = i.to(dtype=torch.float32)
            return j

        # 在单精度浮点数类型下生成一个随机张量
        x = torch.rand((2, 2), dtype=torch.float32)
        # 对函数 f 进行追踪（trace）编译
        scr = torch.jit.trace(f, x)
        # 调用追踪化的函数两次
        scr(x)
        scr(x)
        # 断言最后生成的计算图中所有操作均被融合
        self.assertLastGraphAllFused()
        # 使用指定的误差容忍度比较函数 f 的输出和追踪化函数的输出
        self.assertEqual(f(x), scr(x), atol=4e-3, rtol=4e-3)

        # 在 bfloat16 类型下生成一个随机张量
        bf_x = torch.rand((2, 2), dtype=torch.bfloat16)
        # 对函数 f 进行追踪（trace）编译
        bf_scr = torch.jit.trace(f, bf_x)
        # 调用追踪化的函数两次
        bf_scr(bf_x)
        bf_scr(bf_x)
        # 获取 bfloat16 输入张量对应的计算图
        graph = bf_scr.graph_for(bf_x)
        # 查找计算图中的融合组（fusion groups）
        fusion_groups = self.findFusionGroups(graph)
        # 断言找到的融合组数量为 2
        self.assertEqual(len(fusion_groups), 2)
        # 使用指定的误差容忍度比较函数 f 的输出和追踪化函数的输出
        self.assertEqual(f(bf_x), bf_scr(bf_x), atol=4e-3, rtol=4e-3)
    def test_with_strict_fusion(self):
        def success(x):
            # 使用 torch.jit.strict_fusion() 来确保操作被融合优化
            with torch.jit.strict_fusion():
                # 返回 x + x + x 的结果
                return x + x + x

        # 对 success 函数进行脚本化，检查其脚本化后的版本
        scripted = self.checkScript(success, (torch.rand([4]),))
        # 获取最近执行的优化图
        g = torch.jit.last_executed_optimized_graph()
        # 使用 FileCheck 检查优化图中是否没有 "aten::add"，并且有 "prim::TensorExprGroup"
        FileCheck().check_not("aten::add").check("prim::TensorExprGroup").run(g)

        def foo(x):
            with torch.jit.strict_fusion():
                # 返回 x + x + torch.rand([4]) + 3 的结果
                return x + x + torch.rand([4]) + 3

        # 检查 foo 函数在脚本化时是否引发异常
        with self.assertRaises(Exception) as error_out:
            foo_s = torch.jit.script(foo)
            foo_s(torch.rand([4]))
            foo_s(torch.rand([4]))
            # 打印最近执行的优化图
            print(torch.jit.last_executed_optimized_graph())
        # 使用 FileCheck 检查异常消息中是否包含 "Found unfused operators"
        fc = FileCheck().check("Found unfused operators")
        # 检查异常消息中是否包含 "aten::rand(SymInt[] size" 和 "torch.rand([4]"
        fc.check("aten::rand(SymInt[] size").check("torch.rand([4]").run(str(error_out.exception)))

        # 捕获警告信息
        with warnings.catch_warnings(record=True) as warns:
            foo(torch.rand([4]))

        # 使用 FileCheck 检查警告消息中是否包含 "Only works in script mode"
        FileCheck().check("Only works in script mode").run(str(warns[0]))

        def test_autodiff(x):
            with torch.jit.strict_fusion():
                # 返回 torch.rand([4]) + x + x + x 的结果
                return torch.rand([4]) + x + x + x

        # 对 test_autodiff 函数进行脚本化
        foo_s = torch.jit.script(test_autodiff)
        inp = torch.rand([4], requires_grad=True)
        # 检查对输入进行自动微分时是否引发异常
        with self.assertRaises(Exception) as error_out:
            for _ in range(3):
                foo_s(inp)
        # 使用 FileCheck 检查异常消息中是否包含 "unfused operators" 和 "aten::rand"
        f = FileCheck().check("unfused operators").check("aten::rand")
        f.run(str(error_out.exception))

        def test_separate_fusions(x, y):
            with torch.jit.strict_fusion():
                # 返回 x + x + x 和 y + y + y 的结果
                return x + x + x, y + y + y

        inp = torch.rand([4], requires_grad=True)
        # 检查对输入进行分离融合时是否引发异常
        with self.assertRaises(Exception) as error_out:
            for _ in range(3):
                foo_s = torch.jit.script(test_separate_fusions)
                foo_s(inp, inp)

        # 使用 FileCheck 检查异常消息中是否包含 "Found multiple fusions"
        f = FileCheck().check("Found multiple fusions")
        f.run(str(error_out.exception))
    def test_constant_chunk_shapes(self):
        # 检查常量分块形状的测试函数

        # 如果支持动态形状，则跳过此测试
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        # 对每个设备执行测试
        for device in self.devices:

            # 定义一个函数 f，接受两个参数 x 和 y
            def f(x, y):
                # 创建一个张量 r，其值为常数 4
                r = torch.tensor(4)
                # 对 (x + y + r) 在维度 1 上进行分块，得到 z1 和 z2
                z1, z2 = (x + y + r).chunk(2, dim=1)
                # 返回 z1 和 z2 的乘积
                return z1 * z2

            # 创建两个随机张量 x 和 y，数据类型为 torch.float，设备为当前设备
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)

            # 检查跟踪的图形
            ge = self.checkTrace(f, (x, y))
            graph = ge.graph_for(x, y)

            # 确保我们正在测试正确的场景
            FileCheck().check("with " + FUSION_GROUP + "_").check_count(
                "ConstantChunk", 1, exactly=True
            ).run(str(graph))

            # 对函数 f 进行跟踪编译
            f_traced = torch.jit.trace(f, (x, y))

            # 多次执行跟踪后的函数，确保不会出错
            for i in range(4):
                res = f_traced(x, y)

            # 断言跟踪后的结果与未跟踪的结果相等
            self.assertEqual(res, f(x, y))

    @unittest.skipIf(not RUN_CUDA_HALF, "half-precision NNC fusion requires CUDA")
    def test_pow_multiple_dtype(self):
        # 测试不同数据类型下的幂操作

        # 定义一个函数 fn，接受一个张量 p 和一个浮点数 gamma 参数
        def fn(p: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
            # 对输入张量 p 进行 sigmoid 操作
            p = torch.sigmoid(p)
            # 计算 p 的 gamma 次幂
            result = p**gamma
            # 返回计算结果
            return result

        # 创建一个随机张量 x，数据类型为半精度，设备为 CUDA
        x = torch.rand((2, 2), dtype=torch.half, device="cuda")

        # 计算参考结果 ref
        ref = fn(x)

        # 对函数 fn 进行脚本化编译
        script_fn = torch.jit.script(fn)
        for i in range(4):
            # 多次执行脚本化函数，得到结果 res
            res = script_fn(x)

        # 断言脚本化函数的结果与参考结果相等
        self.assertEqual(ref, res)
# 定义一个继承自 TestTEFuser 的类，设置 dynamic_shapes 属性为 False
class TestTEFuserStatic(TestTEFuser):
    dynamic_shapes = False

# 定义一个继承自 TestTEFuser 的类，设置 dynamic_shapes 属性为 True
class TestTEFuserDynamic(TestTEFuser):
    dynamic_shapes = True

# 删除之前定义的 TestTEFuser 类

# 列出一组字符串，表示需要测试的操作名称列表
works_list = [
    "__radd__",
    "__rdiv__",
    "__rmul__",
    "__rmod__",
    "abs",
    "acos",
    "add",
    "addcmul",
    "addmm.decomposed",
    "asin",
    "atan",
    "atan2",
    "ceil",
    "clamp",
    "clamp.scalar",
    "contiguous",
    "cos",
    "cosh",
    "div.no_rounding_mode",
    "div.true_rounding",
    "div.floor_rounding",
    "div.trunc_rounding",
    "eq",
    "erf",
    "erfc",
    "exp",
    "expand",
    "expand_as",
    "expm1",
    "floor",
    "fmod",
    "fmod.autodiffed",
    "ge",
    "gt",
    "isnan",
    "le",
    "lerp",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "lt",
    "masked_fill",
    "max.binary",
    "mean",
    "min.binary",
    "mm",
    "mul",
    "ne",
    "neg",
    "nn.functional.hardshrink",
    "nn.functional.hardsigmoid",
    "nn.functional.hardswish",
    "nn.functional.softplus",
    "nn.functional.hardtanh",
    "nn.functional.leaky_relu",
    "nn.functional.relu",
    "nn.functional.relu6",
    "nn.functional.softsign",
    "nn.functional.tanhshrink",
    "nn.functional.threshold",
    "permute",
    "pow",
    "reciprocal",
    "remainder",
    "remainder.autodiffed",
    "reshape",
    "reshape_as",
    "round",
    "rsub",
    "rsub.rsub_tensor",
    "rsqrt",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "sub",
    "sum",
    "t",
    "tan",
    "tanh",
    "transpose",
    "true_divide",
    "trunc",
    "unsqueeze",
    "view",
    "view_as",
    "where",
    "bool",
    "byte",
    "char",
    "double",
    "float",
    "half",
    "int",
    "long",
    "short",
    "bool.channels_last",
    "byte.channels_last",
    "char.channels_last",
    "double.channels_last",
    "float.channels_last",
    "half.channels_last",
    "int.channels_last",
    "long.channels_last",
    "short.channels_last",
]

# 已知会失败的操作名称列表
known_failures = [
    "__rmatmul__",
    "frac",
    "matmul",
]

# 如果 OpInfo 测试导致此测试失败，则将其添加到此列表中
skip_ops = ["conj"]

# 定义一个函数，根据操作对象返回操作的名称
def get_name(op):
    l = [op.name]
    if op.variant_test_name != "":
        l.append(op.variant_test_name)
    return ".".join(l)

# 定义一个类的目的是允许使用 super() 调用
# super() [无参数] 失败，可能是由于 instantiate_device_type_tests 的工作方式
# super(TestNNCOpInfo, self) 失败，因为 TestNNCOpInfo 从全局作用域中删除
# super(JitCommonTestCase, self).fn() 将跳过 JitCommonTestCase.fn() 的实现
class TestNNCOpInfoParent(JitCommonTestCase):
    pass

# 继承自 TestNNCOpInfoParent 类的测试类
class TestNNCOpInfo(TestNNCOpInfoParent):
    # 设置测试环境
    def setUp(self):
        super(TestNNCOpInfoParent, self).setUp()
        self.tensorexpr_options = TensorExprTestOptions()

    # 清理测试环境
    def tearDown(self):
        self.tensorexpr_options.restore()
        super(TestNNCOpInfoParent, self).tearDown()
    # 定义一个方法，用于在特定设备上编译给定操作的代码
    def te_compile(self, device, dtype, op):
        # 如果操作在跳过列表中，则不进行编译
        if op.name in skip_ops:
            return
        # 调用操作对象的方法获取样本输入数据迭代器，不需要梯度
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
        # 遍历样本输入数据迭代器
        for sample_input in sample_inputs_itr:
            # 获取参数的值列表，包括输入和其他参数
            arg_values = [sample_input.input] + list(sample_input.args)
            # 获取关键字参数的值
            kwarg_values = sample_input.kwargs
            # 初始化参数名列表、参数值列表和 FX 函数参数列表
            param_names = []
            param_values = []
            fx_args = []
            # 遍历参数值列表
            for idx, v in enumerate(arg_values):
                # 如果参数是 torch.Tensor 对象，则生成一个参数名并添加到列表中，同时将参数值添加到值列表中
                if isinstance(v, torch.Tensor):
                    param_names.append(f"arg_{idx}")
                    param_values.append(v)
                    fx_args.append(param_names[-1])
                else:
                    # 如果参数不是 torch.Tensor 对象，则将其表示形式添加到 FX 函数参数列表中
                    fx_args.append(f"{repr(v)}")

            # 遍历关键字参数的值
            for k, v in kwarg_values.items():
                # 如果关键字参数是 torch.Tensor 对象，则生成一个参数名并添加到列表中，同时将参数值添加到值列表中
                if isinstance(v, torch.Tensor):
                    param_names.append(k)
                    param_values.append(v)
                    fx_args.append(f"{k} = {k}")
                else:
                    # 如果关键字参数不是 torch.Tensor 对象，则将其表示形式添加到 FX 函数参数列表中
                    fx_args.append(f"{k} = {repr(v)}")

            # 组装生成 FX 函数的代码块
            code = f"""
# 定义一个函数 f，接受参数列表作为输入
def f({', '.join(param_names)}):
    # 调用 op 模块中的 op 函数，并传入参数列表的参数
    return op.op({', '.join(fx_args)})

# 使用 exec 函数执行字符串形式的代码块，将 torch、math.inf 和 op 分别映射到 g 字典的键
g = {"torch": torch, "inf": math.inf, "op": op}
exec(code, g)
# 从 g 字典中获取键为 "f" 的函数对象
f = g["f"]
# 将函数 f 的模块名设置为 "test"
f.__module__ = "test"
# 调用函数 f，传入参数列表 param_values，并将结果存储在 out 中
out = f(*param_values)

# 使用 torch.jit.trace 对函数 f 进行跟踪，生成 ts_g
ts_g = torch.jit.trace(f, param_values)
# 使用 ts_g 的图形生成 TensorExprKernel 对象 kernel
kernel = torch._C._te.TensorExprKernel(ts_g.graph)
# 计算函数 f 在参数列表 param_values 上的正确值
correct_val = f(*param_values)
# 断言 TensorExprKernel 对象 kernel 执行与 fallback 方法执行的结果与 correct_val 相等
self.assertEqual(kernel.run(tuple(param_values)), correct_val)
self.assertEqual(kernel.fallback(tuple(param_values)), correct_val)

# 使用 onlyCPU 装饰器限定该测试方法仅在 CPU 上执行
# 使用 unittest.skipIf 装饰器，若 LLVM_ENABLED 为 False 则跳过该测试方法
# 使用 ops 装饰器，传入 op_db 中的操作符列表，并限定操作数的数据类型为 torch.float
def test_working(self, device, dtype, op):
    self.te_compile(device, dtype, op)

# 使用 onlyCPU 装饰器限定该测试方法仅在 CPU 上执行
# 使用 unittest.skipIf 装饰器，若 LLVM_ENABLED 为 False 则跳过该测试方法
# 使用 ops 装饰器，传入 op_db 中的已知失败操作符列表，并限定操作数的数据类型为 torch.float
def test_failures(self, device, dtype, op):
    try:
        self.te_compile(device, dtype, op)
    except Exception as e:
        pass
    else:
        raise RuntimeError(
            "Expected test to fail. If it now works, move op into works_list"
        )

# 使用 onlyCPU 装饰器限定该测试方法仅在 CPU 上执行
# 使用 unittest.skipIf 装饰器，若 LLVM_ENABLED 为 False 则跳过该测试方法
# 使用 ops 装饰器，传入 op_db 中的不支持的操作符列表，并限定操作数的数据类型为 torch.float
def test_unsupported(self, device, dtype, op):
    # 如果 op 的名称在 skip_ops 列表中，则直接返回
    if get_name(op) in skip_ops:
        return
    try:
        # 忽略警告 "TracerWarning" 类型的警告信息
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", TracerWarning)  # noqa: F821
            self.te_compile(device, dtype, op)
    except Exception as e:
        pass
    else:
        raise RuntimeError(
            "Expected test to fail. If it now works, move op into works_list"
        )

# 使用 slowTest 装饰器，标记该测试方法为较慢的测试
# 使用 onlyCPU 装饰器限定该测试方法仅在 CPU 上执行
# 使用 ops 装饰器，传入整个 op_db 列表，支持所有的操作数据类型 OpDTypes.supported
    # 定义测试方法，用于验证神经网络计算正确性
    def test_nnc_correctness(self, device, dtype, op):
        # 如果操作不支持追踪，则跳过测试
        if not op.supports_tracing:
            self.skipTest("Requires tracing support")

        # 使用上下文管理器禁止追踪警告
        with NoTracerWarnContextManager() as no_warn:
            # 获取追踪样本和变体对
            variant_sample_pairs = get_traced_sample_variant_pairs(device, dtype, op)

            # 遍历每个变体和样本对
            for variant, sample in variant_sample_pairs:
                # 创建追踪的函数
                trace = create_traced_fn(self, variant, cache_traced_fn=True)
                # 获取参考结果
                ref = variant(
                    *clone_inputs((sample.input, *sample.args)), **sample.kwargs
                )

                # 运行追踪的函数，并获取结果
                trace(*clone_inputs((sample.input, *sample.args)), **sample.kwargs)
                val = trace(
                    *clone_inputs((sample.input, *sample.args)), **sample.kwargs
                )

                # 设置绝对误差和相对误差容差
                atol = 2e-1 if dtype == torch.bfloat16 else 1e-5
                rtol = 2e-1 if dtype == torch.bfloat16 else 1e-5
                # 断言追踪结果与参考结果的一致性
                self.assertEqual(ref, val, atol=atol, rtol=rtol)

            # 清除所有函数以释放内存，避免由于大量追踪函数导致内存不足
            # 参考 GitHub 问题链接：https://github.com/pytorch/pytorch/issues/35600
            # 每次 torch.jit.trace 都会向 _python_cu 编译单元添加状态
            torch.jit._state._python_cu.drop_all_functions()
# 根据 IS_FBCODE 的值决定测试用例运行的设备类型
only_for = ("cuda") if IS_FBCODE else ("cpu", "cuda")
# 根据设备类型实例化设备测试，其中 TestNNCOpInfo 是被测试的类
instantiate_device_type_tests(TestNNCOpInfo, globals(), only_for=only_for)


# 该类的目的是允许 super() 调用，继承自 JitTestCase 类
class TestLoopnestRandomizationParent(JitTestCase):
    pass


# 继承 TestLoopnestRandomizationParent 类，测试循环嵌套的随机化
class TestLoopnestRandomization(TestLoopnestRandomizationParent):

    # 设置测试环境
    def setUp(self):
        super(TestLoopnestRandomizationParent, self).setUp()

        # 记录并修改 CPU 合并状态
        self.old_cpu_fuser_state = torch._C._jit_can_fuse_on_cpu()
        # 记录并修改是否强制使用 LLVM CPU 的状态
        self.old_must_use_cpu_state = torch._C._jit_get_te_must_use_llvm_cpu()
        # 记录并修改 GPU 合并状态
        self.old_gpu_fuser_state = torch._C._jit_can_fuse_on_gpu()

        # 强制使用 LLVM CPU
        torch._C._jit_override_can_fuse_on_cpu(True)
        # TODO: 强制使用 LLVM，需要添加到 asan、mac、windows 构建以及 sandcastle
        # torch._C._jit_set_te_must_use_llvm_cpu(True)
        # 强制使用 GPU 合并
        torch._C._jit_override_can_fuse_on_gpu(True)

        # 启用性能分析执行器
        self.old_profiling_executor = torch._C._jit_set_profiling_executor(True)
        # 获取图执行器优化模式
        self.old_profiling_mode = torch._C._get_graph_executor_optimize(True)

        # 获取融合组内联状态并关闭
        self.old_fusion_inlining = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)

        # 获取 TExpr 合并状态并启用
        self.texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
        torch._C._jit_set_texpr_fuser_enabled(True)

        # 记录并关闭必须使用 LLVM CPU 的状态
        self.old_te_must_use_llvm_cpu = torch._C._jit_get_te_must_use_llvm_cpu()
        torch._C._jit_set_te_must_use_llvm_cpu(False)

        # 设置种子为 1，用于测试随机转换的代码路径
        os.environ["PYTORCH_TENSOREXPR_RANDOM_TRANSFORM_SEED"] = "1"

    # 清理测试环境
    def tearDown(self):
        # 恢复性能分析执行器状态
        torch._C._jit_set_profiling_executor(self.old_profiling_executor)
        # 恢复图执行器优化模式状态
        torch._C._get_graph_executor_optimize(self.old_profiling_mode)

        # 恢复 GPU 合并状态
        torch._C._jit_override_can_fuse_on_gpu(self.old_gpu_fuser_state)
        # 恢复 CPU 合并状态
        torch._C._jit_override_can_fuse_on_cpu(self.old_cpu_fuser_state)
        # 恢复是否强制使用 LLVM CPU 的状态
        torch._C._jit_set_te_must_use_llvm_cpu(self.old_must_use_cpu_state)
        # 恢复融合组内联状态
        torch._C._debug_set_fusion_group_inlining(self.old_fusion_inlining)

        # 恢复 TExpr 合并状态
        torch._C._jit_set_texpr_fuser_enabled(self.texpr_fuser_state)
        # 恢复必须使用 LLVM CPU 的状态
        torch._C._jit_set_te_must_use_llvm_cpu(self.old_te_must_use_llvm_cpu)

        # 将种子重新设置为 0
        os.environ["PYTORCH_TENSOREXPR_RANDOM_TRANSFORM_SEED"] = "0"
        super(TestLoopnestRandomizationParent, self).tearDown()

    # 测试函数，仅限 CPU 执行
    @onlyCPU
    @unittest.skipIf(not LLVM_ENABLED, "Compiles with TensorExprKernel")
    def test_relu(self, device):
        # 定义测试函数，计算 F.relu(x + 0.5 * y)
        def fn_test_relu(x, y):
            return F.relu(x + 0.5 * y)

        # 生成随机输入 x 和 y
        x = torch.randn(4, 4, dtype=torch.float, device=device)
        y = torch.randn(4, 4, dtype=torch.float, device=device)

        # 跟踪测试函数
        fn = fn_test_relu
        traced_fn = torch.jit.trace(fn, (x, y))

        # 计算参考结果和追踪函数结果
        ref = fn(x, y)
        res = traced_fn(x, y)
        # 断言结果是否接近
        assert torch.allclose(ref, res)

# 根据设备类型实例化设备测试，仅限 CPU
instantiate_device_type_tests(TestLoopnestRandomization, globals(), only_for=("cpu"))
# 如果这个模块是作为主程序执行（而不是被导入到其他模块中），则运行测试函数
if __name__ == "__main__":
    run_tests()
```
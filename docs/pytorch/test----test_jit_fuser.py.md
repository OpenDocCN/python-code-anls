# `.\pytorch\test\test_jit_fuser.py`

```py
# Owner(s): ["oncall: jit"]

# 导入所需的模块和库
import unittest  # 导入unittest模块，用于编写和运行测试
import os  # 导入os模块，提供与操作系统交互的功能
import sys  # 导入sys模块，提供对Python解释器的访问
import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入神经网络函数模块
from torch.testing import FileCheck  # 导入FileCheck类，用于测试检查点
from unittest import skipIf  # 导入skipIf装饰器，根据条件跳过测试用例

# 导入内部测试和工具函数
from torch.testing._internal.common_utils import run_tests, IS_SANDCASTLE, ProfilingMode, GRAPH_EXECUTOR, \
    enable_profiling_mode_for_profiling_tests, IS_WINDOWS, TemporaryDirectoryName, shell
from torch.testing._internal.jit_utils import JitTestCase, enable_cpu_fuser, _inline_everything, \
    RUN_CUDA, RUN_CUDA_HALF, RUN_CUDA_MULTI_GPU, warmup_backward
from textwrap import dedent  # 导入dedent函数，用于消除代码的缩进
from itertools import product, permutations  # 导入product和permutations函数，用于迭代工具
from torch.testing._internal.common_cuda import with_tf32_off  # 导入禁用TF32加速的装饰器

# 从test_jit模块导入测试函数和类
from test_jit import backward_graph, all_backward_graphs, get_lstm_inputs, get_milstm_inputs, \
    LSTMCellC, LSTMCellF, LSTMCellS, MiLSTMCell

# 如果运行模式是ProfilingMode.PROFILING，则设置PyTorch的jit执行和模式为profiling
if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)


def strip_profiling_nodes(nodes):
    # 定义要剔除的profiling相关操作码
    profiling_opcodes = {'prim::BailoutTemplate', 'prim::BailOut'}
    # 返回不包含profiling节点的节点列表
    return [n for n in nodes if n.kind() not in profiling_opcodes]


def warmup_forward(f, *args):
    # 预热函数，多次调用函数以进行性能预热
    profiling_count = 2
    for i in range(profiling_count):
        results = f(*args)

    return results


@skipIf(GRAPH_EXECUTOR == ProfilingMode.LEGACY, "skip due to SIGIOT failures, #67646")
class TestFuser(JitTestCase):
    def assertAllFused(self, graph, except_for=()):
        # 断言所有节点都被融合，排除except_for中列出的节点种类

        # 找到所有的DifferentiableGraph节点
        diff_graphs = [n for n in graph.nodes() if n.kind() == 'prim::DifferentiableGraph']
        if len(diff_graphs) > 0:
            self.assertEqual(len(diff_graphs), 1)
            graph = diff_graphs[0].g('Subgraph')

        # 允许的节点种类集合
        allowed_nodes = {'prim::Constant', 'prim::FusionGroup', 'prim::BailoutTemplate',
                         'prim::BailOut', 'prim::TupleConstruct'} | set(except_for)
        # 断言所有节点都属于允许的节点种类
        self.assertTrue(all(node.kind() in allowed_nodes for node in graph.nodes()),
                        f'got {graph}')
        # 断言只有一个融合组节点
        self.assertTrue([node.kind() for node in graph.nodes()].count('prim::FusionGroup') == 1)

    def _test_fused_abs(self, device='cpu'):
        # 测试绝对值函数的融合

        def func(x):
            return x.abs() * 2

        a = torch.randn(5, device=device)
        scripted = self.checkScript(func, (a,))
        self.assertAllFused(scripted.graph_for(a))

    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @enable_cpu_fuser
    def test_abs_cpu(self):
        # 测试CPU上的绝对值函数融合

        self._test_fused_abs()

    @unittest.skipIf(not IS_WINDOWS, "This is meant to be Windows-specific")
    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @enable_cpu_fuser
    def test_abs_cpu_unicode_temp_dir(self):
        # 使用临时目录名作为后缀创建临时目录，目录名为中文字符
        with TemporaryDirectoryName(suffix='\u4e2d\u6587') as dname:
            # 复制当前进程的环境变量，并设置TMP环境变量为新创建的临时目录名
            shell_env = os.environ.copy()
            shell_env['TMP'] = dname
            # 构建命令列表，用于执行当前文件中指定的测试用例
            cmd = [sys.executable, os.path.basename(__file__), type(self).__name__ + '.test_abs_cpu']
            legacy_jit_flag = '--jit-executor=legacy'
            # 如果命令行参数中包含旧的JIT执行器标志，则添加到命令列表中
            for v in sys.argv:
                if v == legacy_jit_flag:
                    cmd.append(legacy_jit_flag)
            # 在指定的环境下执行命令，当前工作目录为当前文件所在目录
            return_code = shell(cmd, cwd=os.path.dirname(__file__), env=shell_env)
            # 断言返回码为0，表示命令执行成功
            self.assertEqual(return_code, 0)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_abs_cuda(self):
        # 执行在CUDA设备上运行的_test_fused_abs测试函数
        self._test_fused_abs(device="cuda")

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_zero_element_tensors(self):
        def decode(sin_t, cos_t):
            # 使用torch.atan2计算sin_t和cos_t的反正切，返回角度
            theta = torch.atan2(sin_t.float(), cos_t.float())
            return theta

        # 在CUDA设备上创建大小为0的零张量sin和cos
        sin = torch.zeros(0, device="cuda")
        cos = torch.zeros(0, device="cuda")
        # 将输入组成列表，并通过self.checkScript检查脚本
        inputs = [sin, cos]
        ge = self.checkScript(decode, inputs)

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_arg_configurations_smoke_cuda(self):
        # 用于检测在CUDA设备上运行时，是否能正确区分连续和非连续的内核
        def f(x, y):
            z1, z2 = (x + y).chunk(2, dim=1)
            return z1 * z2

        # 在CUDA设备上创建随机张量x和y
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')
        # 对函数f进行跟踪，并使用self.assertEqual断言张量的处理结果是否一致
        traced_f = torch.jit.trace(f, (x, y,))
        self.assertEqual(traced_f(x.t().contiguous(), y), traced_f(x.t(), y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_broadcast_cuda(self):
        # 在CUDA设备上测试张量的广播计算
        def scaleshift(x, scale, shift):
            return x * scale + shift

        # 创建输入张量列表，都在CUDA设备上
        inputs = [
            torch.randn(4, 4, dtype=torch.float, device='cuda'),
            torch.randn(4, dtype=torch.float, device='cuda'),
            torch.randn(4, dtype=torch.float, device='cuda'),
        ]
        # 使用self.checkTrace检查张量的跟踪，并使用self.assertAllFused验证所有操作是否被融合
        ge = self.checkTrace(scaleshift, inputs)
        self.assertAllFused(ge.graph_for(*inputs))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "no bfloat support with profiling on")
    def test_cuda_bfloat16(self):
        # 在CUDA设备上测试bfloat16类型的张量操作
        def foo(x, y):
            return (x + y).relu()
        # 使用torch.jit.script对foo函数进行脚本化
        m = torch.jit.script(foo)
        # 创建CUDA设备上的bfloat16类型的随机张量x和y，并使用self.assertAllFused验证操作是否被融合
        x = torch.randn(65536).cuda().bfloat16()
        y = torch.randn_like(x)
        self.assertAllFused(m.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_HALF, "no half support")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "no half support with profiling on")
    def test_cuda_half(self):
        # 创建两个在 CUDA 设备上的半精度随机张量
        x = torch.randn(4, 4, dtype=torch.half, device='cuda')
        y = torch.randn(4, 4, dtype=torch.half, device='cuda')

        # 定义需要测试的函数列表
        funcs = [
            self.fn_test_comparison_gt_lt,
            self.fn_test_relu,
            self.fn_test_exp
        ]

        # 注意: 非融合输入必须为浮点型，以防止精度损失
        inputs = (x.float(), y.float())
        fusion_inputs = (x, y)
        for fn in funcs:
            # 创建需要梯度的本地输入副本
            local_inputs = [t.clone().requires_grad_() for t in inputs]
            local_fusion_inputs = [t.clone().requires_grad_() for t in fusion_inputs]

            # 验证输出
            fusion = torch.jit.trace(fn, local_fusion_inputs, check_trace=False)
            outputs = fn(*local_inputs)
            fusion_outputs = fusion(*local_fusion_inputs)
            outputs_half = [t.half() for t in outputs]
            self.assertEqual(outputs_half, fusion_outputs)

            # 验证梯度
            for output, fusion_output in zip(outputs_half, fusion_outputs):
                grads = torch.autograd.grad(
                    output.float().sum(), local_inputs, allow_unused=True, retain_graph=True)
                fusion_grads = torch.autograd.grad(
                    fusion_output.sum(), local_fusion_inputs, allow_unused=True, retain_graph=True)
                grads_half = [t.half() for t in grads]
                self.assertEqual(grads_half, fusion_grads)

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_checks_cat_inputs(self):
        # 我们不应将 cat 节点视为广播。在运行内核之前，需要检查它们的所有输入是否具有相同的映射大小。
        def f(x, y):
            return torch.cat([x + 2 * x + x ** 2, y + 4 * y + y ** 3], dim=0)

        # 注意: y 可以广播到 x，但是 f(x, y) 的输出应该是 3x4 的形状，而不是 4x4。
        x = torch.randn(2, 4, dtype=torch.float, device='cuda')
        y = torch.randn(1, 4, dtype=torch.float, device='cuda')

        scripted = self.checkScript(f, (x, y))
        self.assertAllFused(scripted.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "No CUDA")
    def test_remainder_cuda(self):
        def cuda_rem(x, y):
            return 1 + torch.remainder(x, y) - 1

        a = torch.rand([512], dtype=torch.float).cuda()
        b = torch.rand([512], dtype=torch.float).cuda()
        inputs = [a, b]
        ge = self.checkScript(cuda_rem, inputs)
        graph = ge.graph_for(*inputs)
        self.assertAllFused(graph)

    @unittest.skipIf(not RUN_CUDA, "No CUDA")
    # 定义一个测试函数，用于测试在 CUDA 设备上执行 tensor.chunk 操作的脚本
    def test_chunk_cuda(self):
        # 定义一个匿名函数 fn，参数 x 为一个 tensor，对其进行 3 分块操作
        def fn(x):
            a, b, c = x.chunk(3, 1)
            return a * b + c

        # 创建一个输入列表，包含一个在 CUDA 设备上随机生成的 tensor
        inputs = [torch.randn(10, 6, dtype=torch.float, device='cuda')]

        # 使用 self.checkScript 方法检查并返回脚本化的函数 fn 的图形表达式
        ge = self.checkScript(fn, inputs)
        # 获取检查后的图形对象
        graph = ge.graph_for(*inputs)
        # 断言图中所有操作均已融合
        self.assertAllFused(graph)
        # 使用 FileCheck 检查图中是否存在符合指定条件的常量分块操作
        FileCheck().check("prim::ConstantChunk[chunks=3, dim=1]").run(str(graph))

    # 静态方法：测试不同 chunk 方法在给定设备上的正确性
    @staticmethod
    def _test_chunk_correctness(self, device='cpu'):
        # 定义四分块函数 chunk_4_0，返回在指定维度上四分块后的张量之和
        def chunk_4_0(x):
            x0, x1, x2, x3 = x.chunk(4, 0)
            return x0 + x1 + x2 + x3

        # 定义四分块函数 chunk_4_1，返回在指定维度上四分块后的张量之和
        def chunk_4_1(x):
            x0, x1, x2, x3 = x.chunk(4, 1)
            return x0 + x1 + x2 + x3

        # 定义四分块函数 chunk_4_last，返回在指定维度上四分块后的张量之和
        def chunk_4_last(x):
            x0, x1, x2, x3 = x.chunk(4, 2)
            return x0 + x1 + x2 + x3

        # 将所有四分块函数放入列表中
        fns = [chunk_4_0, chunk_4_1, chunk_4_last]
        # 创建不同测试用例的张量列表
        tensors = [
            # 拆分大小为 1
            torch.randn(4, 4, 4, dtype=torch.float, device=device),

            # 连续情况
            torch.randn(12, 8, 16, dtype=torch.float, device=device),

            # 非连续情况
            torch.randn(12, 8, 16, dtype=torch.float, device=device).transpose(1, 2),
        ]

        # 遍历所有张量和函数组合，使用 self.checkScript 检查脚本化的函数在给定输入上的表现
        for tensor in tensors:
            for fn in fns:
                self.checkScript(fn, [tensor])

    # 跳过测试，如果在 Sandcastle 上运行，因为当前不支持在 CPU 上使用 fuser
    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @enable_cpu_fuser
    # 测试在 CPU 上 chunk 方法的正确性
    def test_chunk_correctness(self):
        return self._test_chunk_correctness(self, 'cpu')

    # 跳过测试，如果未启用 CUDA，因为需要 CUDA 环境
    @unittest.skipIf(not RUN_CUDA, "No CUDA")
    # 测试在 CUDA 上 chunk 方法的正确性
    def test_chunk_correctness_cuda(self):
        return self._test_chunk_correctness(self, 'cuda')

    # 跳过测试，如果未启用 CUDA，因为 fuser 需要 CUDA
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # 测试 chunk 方法在 CUDA 上的分布情况
    def test_chunk_distributes_cuda(self):
        # 定义一个函数 f，输入两个张量 x 和 y，返回这两个张量按指定维度分块后的乘积
        def f(x, y):
            z1, z2 = (x + y).chunk(2, dim=1)
            return z1 * z2

        # 创建两个随机张量 x 和 y，均在 CUDA 设备上
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        # 使用 self.checkTrace 方法检查函数 f 的跟踪，并返回跟踪引擎
        ge = self.checkTrace(f, (x, y))
        # 获取跟踪引擎的图形表达式
        graph = ge.graph_for(x, y)
        # 使用 FileCheck 检查图中是否存在广播张量和融合组，并且恰好包含两次常量分块操作
        FileCheck().check("broadcast_tensors").check('with prim::FusionGroup_') \
            .check_count('ConstantChunk', 2, exactly=True).run(str(graph))

    # 跳过测试，如果未启用 CUDA，因为 fuser 需要 CUDA
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # 测试 chunk 方法在 CUDA 上的输入去重情况
    def test_chunk_motion_deduplicates_inputs(self):
        # 定义函数 func1，输入张量 x，返回 x 的平方后分块后的乘积
        def func1(x):
            z = x * x
            z0, z1 = z.chunk(2)
            return z0 * z1

        # 定义函数 func2，输入张量 x，返回 x 的立方后分块后的乘积
        def func2(x):
            z = x * x * x
            z0, z1 = z.chunk(2)
            return z0 * z1

        # 创建一个张量列表作为输入，张量在 CUDA 设备上，包含两个值
        inputs = [
            torch.tensor([1.1, 1.2], device='cuda', dtype=torch.float),
        ]
        
        # 对每个函数应用测试脚本，使用 self.assertGraphContainsExactly 确认图中确切包含一个融合组
        for func in [func1, func2]:
            module = self.checkScript(func, inputs)
            forward_graph = module.graph_for(*inputs)
            self.assertGraphContainsExactly(forward_graph, 'prim::FusionGroup', 1)
            fusion_group = list(forward_graph.nodes())[-1]
            # 确保融合组的输入数为 1
            self.assertEqual(len(list(fusion_group.inputs())), 1)
    # 如果未设置运行 CUDA，则跳过此测试
    @unittest.skipIf(not RUN_CUDA, "No CUDA")
    # 定义测试函数，用于测试编译器是否正确处理额外参数的顺序
    def test_chunk_multiple_cuda(self):
        def fn(s, x, y, z):
            # 对张量 z 进行分块，每个分块有 2 个元素，分成 2 组
            z1, z2 = z.chunk(2, 2)
            # 对张量 x 进行分块，每个分块有 3 个元素，分成 3 组
            x1, x2, x3 = x.chunk(3, 1)
            # 对张量 y 进行分块，每个分块有 2 个元素，分成 2 组
            y1, y2 = y.chunk(2, 0)
            # 返回计算结果，包括所有分块的和
            return s + x1 + x2 + x3 + y1 + y2 + z1 + z2

        # 输入数据，包括四个不同形状的张量
        inputs = [
            torch.randn(5, 2, 3, dtype=torch.float, device='cuda'),
            torch.randn(5, 6, 3, dtype=torch.float, device='cuda'),
            torch.randn(10, 2, 3, dtype=torch.float, device='cuda'),
            torch.randn(5, 2, 6, dtype=torch.float, device='cuda'),
        ]

        # 检查脚本化函数 fn 的图是否通过测试
        ge = self.checkScript(fn, inputs)
        # 断言所有操作都被融合到一个图中
        self.assertAllFused(ge.graph_for(*inputs))

    # 如果未设置运行 CUDA，则跳过此测试
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # 测试 torch.max 和 torch.min 函数
    def test_minmax(self):
        # 定义 torch.max 函数的简单版本
        def tmax(a, b):
            return torch.max(2 * a, b)

        # 定义 torch.min 函数的简单版本
        def tmin(a, b):
            return torch.min(2 * a, b)

        # 创建两个形状为 (4, 4) 的 CUDA 张量 a 和 b
        a = torch.randn(4, 4, dtype=torch.float, device="cuda")
        b = torch.randn(4, 4, dtype=torch.float, device="cuda")
        # 创建一个值为 NaN 的 CUDA 张量
        nan = torch.tensor(float('nan'), dtype=torch.float, device="cuda")

        # 对每个函数和输入进行组合，使用 product 生成所有组合
        for f, inputs in product(
                (tmax, tmin),
                ([a, b], [a, nan], [b, nan])):
            # 检查脚本化函数 f 的图是否通过测试
            s = self.checkScript(f, inputs)
            # 断言所有操作都被融合到一个图中
            self.assertAllFused(s.graph_for(*inputs))

    # 如果未设置运行 CUDA，则跳过此测试
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # 测试 torch.clamp 函数的多个变体
    def test_clamp(self):
        # 定义 torch.clamp 函数的简单版本，限制在 [0, 2] 范围内
        def func2(a, b):
            return torch.clamp(a + b, min=0, max=2)

        # 定义 torch.clamp 函数的简单版本，将上限设为无穷大
        def funcInf(a, b):
            return torch.clamp(a + b, min=0, max=float('inf'))

        # 定义 torch.clamp 函数的简单版本，仅设定上限
        def funcOptMin(a, b):
            return torch.clamp(a + b, max=2)

        # 定义 torch.clamp 函数的简单版本，仅设定下限
        def funcOptMax(a, b):
            return torch.clamp(a + b, min=0)

        # 创建一个形状为 (4, 4) 的 CUDA 张量 a，并启用梯度跟踪
        a = torch.randn(4, 4, dtype=torch.float, device='cuda', requires_grad=True)
        # 创建一个形状为 (4, 4) 的 CUDA 张量 b
        b = torch.randn(4, 4, dtype=torch.float, device='cuda')
        # 创建一个值为 NaN 的 CUDA 张量
        nan = torch.tensor(float('nan'), dtype=torch.float, device='cuda')

        # 准备要测试的函数列表
        funcs = (func2, funcInf, funcOptMin, funcOptMax)
        # 对每个函数和输入进行组合，使用 product 生成所有组合
        for f, inputs in product(funcs, [[a, b], [a, nan]]):
            # 禁用 JIT 函数缓存以获取准确的分析结果
            f.__disable_jit_function_caching__ = True
            # 提取输入张量
            inp1, inp2 = inputs
            # 检查脚本化函数 f 的图是否通过测试，并启用性能分析
            s = self.checkScript(f, (inp1, inp2), profiling=ProfilingMode.PROFILING)
            # 断言所有操作都被融合到一个图中，除了指定的例外操作
            self.assertAllFused(s.graph_for(inp1, inp2), except_for={'aten::size', 'aten::_size_if_not_equal'})
            # 计算函数结果
            c = s(inp1, inp2)
            # 在性能分析测试中启用梯度跟踪
            with enable_profiling_mode_for_profiling_tests():
                warmup_backward(c.sum())
            # 提取反向传播图
            graph = backward_graph(s)
            # 断言所有操作都被融合到一个图中，除了指定的例外操作
            self.assertAllFused(graph, except_for={'aten::Float', 'aten::_grad_sum_to_size'})
    # 定义一个测试函数，用于测试在 GPU 上的神经网络模型推断过程中的 dropout 和 ReLU 操作
    def test_dropout(self):
        # 定义一个简单的函数 func，应用 dropout 后再使用 ReLU 激活函数
        def func(x):
            # 对输入 x 应用 dropout 操作
            x = torch.nn.functional.dropout(x)
            # 对应用 dropout 后的结果应用 ReLU 激活函数
            return torch.nn.functional.relu(x)

        # 创建一个随机张量 a，数据类型为 float，存储在 GPU 上，并且需要计算梯度
        a = torch.randn(4, 4, dtype=torch.float, device='cuda', requires_grad=True)
        # 对 func 函数进行 TorchScript 编译，得到一个脚本模块 s
        s = torch.jit.script(func)
        # 用输入张量 a 执行脚本模块 s，得到输出张量 c
        c = s(a)
        # 再次用输入张量 a 执行脚本模块 s，得到输出张量 c（用于测试脚本模块 s 是否缓存）
        c = s(a)
        # 运行一个预热的反向传播过程，以确保所有操作都已注册
        warmup_backward(c.sum())
        # 使用 backward_graph 函数获取反向传播图，并跳过额外的故障节点检查
        graph = backward_graph(s, skip_check=True)
        # 断言反向传播图中的所有操作都已融合，除了 'aten::div' 和 'prim::Constant'
        self.assertAllFused(graph, except_for={'aten::div', 'prim::Constant'})

    # 如果 CUDA 不可用，则跳过该测试
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_comparison_eq_ne(self):
        # 定义一个函数 f，执行元素级别的相等和不等比较，并根据结果执行运算
        def f(x, y):
            # 创建一个 x 是否等于 0 的掩码，并将其转换为与 x 相同的数据类型
            mask = (x == 0).type_as(x)
            # 计算 z = x * mask + y，只有当 x 等于 0 时 mask 才为 1，否则为 0
            z = x * mask + y
            # 创建一个 x 是否不等于 0 的掩码，并将其转换为与 x 相同的数据类型
            mask = (x != 0).type_as(x)
            # 计算 z = z * mask + y，只有当 x 不等于 0 时 mask 才为 1，否则为 0
            z = z * mask + y
            return z

        # 创建两个随机张量 x 和 y，数据类型为 float，存储在 GPU 上
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        # 使用 checkTrace 函数对 f 函数进行 TorchScript 脚本化，并获取生成的图形
        ge = self.checkTrace(f, (x, y))
        # 断言生成的 TorchScript 图形中的所有操作都已融合
        self.assertAllFused(ge.graph_for(x, y))

    # 定义一个静态方法，用于执行元素级别的大于和小于比较，并根据结果执行运算
    @staticmethod
    def fn_test_comparison_gt_lt(x, y):
        # 创建一个 x 大于 0 的掩码，并将其转换为与 x 相同的数据类型
        mask = (x > 0).type_as(x)
        # 计算 z = x * mask + y，只有当 x 大于 0 时 mask 才为 1，否则为 0
        z = x * mask + y
        # 创建一个 x 小于 0 的掩码，并将其转换为与 x 相同的数据类型
        mask = (x < 0).type_as(x)
        # 计算 z = z * mask + y，只有当 x 小于 0 时 mask 才为 1，否则为 0
        z = z * mask + y
        return z

    # 如果 CUDA 不可用，则跳过该测试
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_comparison_gt_lt_cuda(self):
        # 创建两个随机张量 x 和 y，数据类型为 float，存储在 GPU 上
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        # 使用 checkTrace 函数对 fn_test_comparison_gt_lt 方法进行 TorchScript 脚本化，并获取生成的图形
        ge = self.checkTrace(self.fn_test_comparison_gt_lt, (x, y))
        # 断言生成的 TorchScript 图形中的所有操作都已融合
        self.assertAllFused(ge.graph_for(x, y))

    # 如果 CUDA 不可用，则跳过该测试
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_comparison_ge_le_cuda(self):
        # 定义一个函数 f，执行元素级别的大于等于和小于等于比较，并根据结果执行运算
        def f(x, y):
            # 创建一个 x 大于等于 0 的掩码，并将其转换为与 x 相同的数据类型
            mask = (x >= 0).type_as(x)
            # 计算 z = x * mask + y，只有当 x 大于等于 0 时 mask 才为 1，否则为 0
            z = x * mask + y
            # 创建一个 x 小于等于 0 的掩码，并将其转换为与 x 相同的数据类型
            mask = (x <= 0).type_as(x)
            # 计算 z = z * mask + y，只有当 x 小于等于 0 时 mask 才为 1，否则为 0
            z = z * mask + y
            return z

        # 创建两个随机张量 x 和 y，数据类型为 float，存储在 GPU 上
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        # 使用 checkTrace 函数对 f 函数进行 TorchScript 脚本化，并获取生成的图形
        ge = self.checkTrace(f, (x, y))
        # 断言生成的 TorchScript 图形中的所有操作都已融合
        self.assertAllFused(ge.graph_for(x, y))
        # 设置 x 和 y 的 requires_grad 属性为 True，以便后续进行梯度计算的融合图形断言
        x.requires_grad_(True)
        y.requires_grad_(True)
        # 断言生成的 TorchScript 图形中的所有操作都已融合，但排除 "aten::size"、"prim::BroadcastSizes" 和 "aten::_size_if_not_equal"
        self.assertAllFused(ge.graph_for(x, y), except_for=("aten::size", "prim::BroadcastSizes",
                                                            "aten::_size_if_not_equal"))

    # 如果 CUDA 不可用，则跳过该测试
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_addcmul_cuda(self):
        # 创建三个随机张量 t、t1、t2，数据类型为 float，存储在 GPU 上
        t = torch.randn(1, 4, dtype=torch.float, device='cuda')
        t1 = torch.randn(4, 1, dtype=torch.float, device='cuda')
        t2 = torch.randn(1, 4, dtype=torch.float, device='cuda')

        # 定义一个函数 foo，执行 addcmul 操作
        def foo(t, t1, t2):
            return t.addcmul(t + 1, t2, value=0.1)

        # 使用 checkTrace 函数对 foo 函数进行 TorchScript 脚本化，并获取生成的图形，允许未使用的张量
        ge = self.checkTrace(foo, (t, t1, t2), allow_unused=True)
        # 获取 foo 函数的 TorchScript 图形
        graph = ge.graph_for(t, t1, t2)
        # 断言生成的 TorchScript 图形
    # 如果没有启用 CUDA，跳过测试（因为 fuser 需要 CUDA）
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # 测试 torch.lerp 函数的 CUDA 版本
    def test_lerp(self):
        # 生成随机张量，指定在 CUDA 设备上运行
        start = torch.randn(4, 1, dtype=torch.float, device='cuda')
        end = torch.randn(1, 4, dtype=torch.float, device='cuda')
        weight = torch.tensor(0.5, dtype=torch.float, device='cuda')

        # 标量权重重载
        def foo_weight_scalar(start, end):
            # 使用 torch.lerp 计算线性插值
            return torch.lerp(start + 1, end, 0.5)

        # 张量权重重载
        def foo_weight_tensor(start, end):
            # 使用预先定义的权重张量进行 torch.lerp 计算
            return torch.lerp(start + 1, end, weight)

        # 对标量权重版本进行 TorchScript 跟踪
        ge_weight_scalar = self.checkTrace(foo_weight_scalar, (start, end))
        # 获取 TorchScript 图
        graph = ge_weight_scalar.graph_for(start, end)
        # 断言所有操作已融合
        self.assertAllFused(graph)

        # 对张量权重版本进行 TorchScript 跟踪
        ge_weight_tensor = self.checkTrace(foo_weight_tensor, (start, end))
        # 获取 TorchScript 图
        graph = ge_weight_tensor.graph_for(start, end)
        # 断言所有操作已融合
        self.assertAllFused(graph)

    # 如果没有启用 CUDA，跳过测试（因为 fuser 需要 CUDA）
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # 测试 torch.cat 函数的 CUDA 版本
    def test_concat_cuda(self):
        # 生成随机张量，指定在 CUDA 设备上运行
        hx = torch.randn(3, 20, dtype=torch.float, device='cuda')
        cx = torch.randn(3, 20, dtype=torch.float, device='cuda')

        def foo(hx, cx):
            # 使用 torch.cat 连接张量的和与积
            return torch.cat((hx + cx, hx * cx))

        # 对函数进行 TorchScript 跟踪
        ge = self.checkTrace(foo, (hx, cx))
        # 获取 TorchScript 图
        graph = ge.graph_for(hx, cx)
        # 断言所有操作已融合
        self.assertAllFused(graph)
        # 使用 FileCheck 工具检查融合情况
        FileCheck().check("FusedConcat").check_next("return").run(str(graph))

    # 如果没有启用 CUDA，跳过测试（因为 fuser 需要 CUDA）
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # 测试带不变条件的 torch.cat 函数的 CUDA 版本
    def test_concat_invariant_cuda(self):
        # 定义函数 fn，执行一系列张量操作
        def fn(x, y, z):
            x1 = x + y
            y1 = x - y
            w = torch.cat([x1, y1])
            return w + z

        # 生成随机张量，指定在 CUDA 设备上运行
        x = torch.randn(2, 2, dtype=torch.float, device='cuda')
        y = torch.randn(2, 2, dtype=torch.float, device='cuda')
        z = torch.randn(4, 2, dtype=torch.float, device='cuda')
        # 对函数进行 TorchScript 跟踪
        ge = self.checkTrace(fn, (x, y, z))
        # 获取 TorchScript 图
        graph = ge.graph_for(x, y, z)
        # 断言所有操作已融合，但排除 aten::add 操作
        self.assertAllFused(graph, except_for={'aten::add'})
        # 使用 FileCheck 工具检查融合情况
        FileCheck().check("FusedConcat").check_next("return").run(str(graph))

    # 静态方法：测试自定义函数的指数计算（CUDA 版本）
    @staticmethod
    def fn_test_exp(x, y):
        # 计算 (x + .5 * y) 的指数
        return (x + .5 * y).exp()

    # 如果没有启用 CUDA，跳过测试（因为 fuser 需要 CUDA）
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # 测试 torch.exp 函数的 CUDA 版本
    def test_exp_cuda(self):
        # 生成随机张量，指定在 CUDA 设备上运行
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        # 对自定义函数进行 TorchScript 跟踪
        ge = self.checkTrace(self.fn_test_exp, (x, y))
        # 断言所有操作已融合
        self.assertAllFused(ge.graph_for(x, y))

    # 如果没有启用 CUDA，跳过测试（因为 fuser 需要 CUDA）
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # 如果图执行器不是 ProfilingMode.LEGACY，跳过测试（因为与分析搭配存在问题）
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "broken with profiling on")
    @torch._jit_internal._disable_emit_hooks_decorator
    # 用装饰器声明将该方法内的所有函数都进行内联优化
    @_inline_everything
    def test_fuse_decompose_normalization(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类，用于测试规范化模块
        class ResLike(torch.jit.ScriptModule):
            # 构造函数，初始化时传入规范化模块
            def __init__(self, norm_module):
                super().__init__()
                self.nm = norm_module

            # 前向传播函数的 JIT 脚本方法
            @torch.jit.script_method
            def forward(self, x, y):
                return y + torch.relu(self.nm(x))

        # 测试规范化分解函数
        def test_norm_decompose(nm, in_opt_graph, not_in_opt_graph, in_fusegraph):
            # 创建一个 ResLike 类的实例，并移动到 CUDA 设备
            model = ResLike(nm).cuda()
            # 创建另一个未优化的 ResLike 类的实例，并加载状态字典，也移到 CUDA 设备
            model_noopt = ResLike(nm).cuda()
            model_noopt.load_state_dict(model.state_dict())
            # 创建 CUDA 设备上的随机输入张量 x 和 y
            x = torch.randn(2, 16, 8, 8, device='cuda')
            y = torch.randn(2, 16, 8, 8, device='cuda')

            # 用 torch.no_grad() 上下文管理器，关闭梯度计算
            with torch.no_grad():
                # 计算模型输出，获取模型的计算图，并将其转化为字符串
                out = model(x, y)
                graph = model.graph_for(x, y)
                rep = str(graph)

                # 在未优化执行环境下，再次计算输出，获取计算图字符串表示
                with torch.jit.optimized_execution(False):
                    out_noopt = model_noopt(x, y)
                    rep_noopt = str(model_noopt.graph_for(x, y))
                # 断言两个计算结果相等，允许的绝对误差为 3e-5
                self.assertEqual(out, out_noopt, atol=3e-5)

            # 检查是否成功将归一化操作分解
            for node_in_graph in in_opt_graph:
                self.assertIn(node_in_graph, rep)

            for node_not_in_graph in not_in_opt_graph:
                # 确保优化后计算图中不包含未优化节点，但未优化的中包含
                self.assertNotIn(node_not_in_graph, rep)
                self.assertIn(node_not_in_graph, rep_noopt)

            # 获取融合组并检查
            fusion_groups = [node for node in graph.nodes() if node.kind() == 'prim::FusionGroup']
            self.assertEqual(len(fusion_groups), 1)
            fused_graph = str(fusion_groups[0].g('Subgraph'))
            for node_in_fusegraph in in_fusegraph:
                self.assertIn(node_in_fusegraph, fused_graph)

        # 测试批归一化分解
        bm = nn.BatchNorm2d(16)
        test_norm_decompose(bm, ['aten::batch_norm_update_stats'],
                            ['aten::batch_norm('], ['aten::sqrt'])

        # 测试层归一化分解
        lm = nn.LayerNorm(8)
        test_norm_decompose(lm, ['aten::batch_norm_stats'],
                            ['aten::layer_norm('], ['aten::sub', 'aten::mul', 'aten::add'])

    # 如果没有 CUDA 运行环境，则跳过该测试
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_threshold(self):
        # 定义一个函数 f，应用阈值操作并返回结果
        def f(x):
            return torch.threshold(x, 0, -10) + x + x + x

        # 创建一个 CUDA 设备上的张量 x
        x = torch.tensor([-1, -0.5, 0, 1, 2, 3], device='cuda')
        # 使用 self.checkScript 方法检查 f 函数的脚本化版本
        scripted = self.checkScript(f, (x,))
        # 断言所有操作都已融合
        self.assertAllFused(scripted.graph_for(x))

    # 如果没有 CUDA 运行环境，则跳过该测试
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    # 如果在 Sandcastle 环境下，跳过这个测试；否则继续执行
    @unittest.skip("deduplicating introduces aliasing in backward graph's outputs")
    # 跳过这个测试，因为去重会导致反向图输出的别名问题

    @enable_cpu_fuser
    # 启用 CPU 的融合器

    def test_fuser_deduplication(self):
        # 测试融合器的去重功能

        # 查看在去除 fuser 编译中的 _grad_sum_to_size 后，融合内核的输出是否被去重
        # 参见 PR #14957 中的讨论。
        def f(x, y):
            return torch.sigmoid(x + y)

        # 创建随机张量并设置需要梯度信息
        b = torch.randn(5, 5, requires_grad=True)
        a = torch.randn(5, 5, requires_grad=True)

        # 对函数 f 进行脚本化并进行融合检查
        s = self.checkScript(f, (a, b))
        self.assertAllFused(s.graph_for(a, b), except_for={
                            'aten::size', 'aten::_size_if_not_equal', 'prim::BroadcastSizes'})

        # 使用函数 f 进行计算
        c = s(a, b)

        # 进行反向传播的预热并获取梯度结果
        results = warmup_backward(c.sum(), [a, b])
        ga2, gb2 = results.pop()

        # 获取反向图
        graph = backward_graph(s)

        # 检查反向图中的所有操作是否都被融合
        self.assertAllFused(graph)

        # 检查张量 a 和 b 是否共享存储，即在融合器中作为单个输出生成
        self.assertEqual(ga2.data_ptr(), gb2.data_ptr())
    def test_fuser_iou(self):
        # This checks if most of Intersection over Union is fused.
        # In particular, the backward contains many _grad_sum_to_size.
        # 这个函数用来测试 Intersection over Union 是否能够完全融合。
        # 特别是，反向传播中包含了许多 _grad_sum_to_size 操作。

        def iou(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2):
            # Calculate Intersection over Union (IoU) between two sets of bounding boxes.
            # 计算两组边界框之间的 Intersection over Union (IoU)。
            ltx = torch.max(b1x1, b2x1)  # [N,M]
            lty = torch.max(b1y1, b2y1)
            rbx = torch.min(b1x2, b2x2)
            rby = torch.min(b1y2, b2y2)

            w = (rbx - ltx).clamp(min=0, max=float('inf'))  # [N,M]
            h = (rby - lty).clamp(min=0, max=float('inf'))  # [N,M]
            inter = w * h  # [N,M]

            area1 = (b1x2 - b1x1) * (b1y2 - b1y1)  # [N,1]
            area2 = (b2x2 - b2x1) * (b2y2 - b2y1)  # [1,M]
            iou = inter / (area1 + area2 - inter)
            return iou

        box1 = torch.randn(5, 4, requires_grad=True)
        box2 = torch.randn(5, 4, requires_grad=True)
        # unsqueezing can currently not be fused
        # 不支持融合的操作
        b1x1 = box1[:, 0].unsqueeze(1)  # [N,1]
        b1y1 = box1[:, 1].unsqueeze(1)
        b1x2 = box1[:, 2].unsqueeze(1)
        b1y2 = box1[:, 3].unsqueeze(1)
        b2x1 = box2[:, 0].unsqueeze(0)  # [1,N]
        b2y1 = box2[:, 1].unsqueeze(0)
        b2x2 = box2[:, 2].unsqueeze(0)
        b2y2 = box2[:, 3].unsqueeze(0)

        s = self.checkScript(iou, (b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2))
        self.assertAllFused(s.graph_for(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2),
                            except_for={'aten::size', 'prim::BroadcastSizes', 'aten::_size_if_not_equal'})

        with enable_profiling_mode_for_profiling_tests(True):
            c = s(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2)
            warmup_backward(c.sum(), [b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2])
            graph = backward_graph(s)
            self.assertAllFused(graph, except_for={'aten::size', 'prim::BroadcastSizes', 'aten::_size_if_not_equal'})

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    @enable_cpu_fuser
    def test_fusion_reuse_multi_gpu(self):
        # Test function for demonstrating fusion reuse on multiple GPUs.
        # 用于展示多个 GPU 上融合重用的测试函数。

        def fn(x, y):
            return x * y * x * y

        inputs_cpu = [
            torch.randn(4, 4, dtype=torch.float),
            torch.randn(4, 4, dtype=torch.float),
        ]
        inputs_cuda0 = [x.cuda(0) for x in inputs_cpu]
        inputs_cuda1 = [y.cuda(1) for y in inputs_cpu]

        # Should not crash; these should compile different kernels.
        # 不应该崩溃；这些应该编译成不同的内核。
        ge = self.checkScript(fn, inputs_cpu)
        self.assertAllFused(ge.graph_for(*inputs_cpu))
        ge(*inputs_cuda0)
        ge(*inputs_cuda1)

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    @enable_cpu_fuser
    # 定义一个测试函数，用于测试多GPU情况下的核心缓存
    def test_kernel_cache_multi_gpu(self):
        # 定义一个不可融合的函数，用于测试
        def not_fusible(x):
            return x

        # 定义一个函数 fn，对输入的三个张量进行乘方运算，返回结果
        def fn(x, y, z):
            # 对 x 进行五次乘方运算，用于融合测试
            x_out = x * x * x * x * x  # fusion: lambda x. x * x * x * x * x
            # 对 y 进行五次乘方运算
            y_out = y * y * y * y * y
            # 对 z 进行五次乘方运算
            z_out = z * z * z * z * z
            return not_fusible(x_out), not_fusible(y_out), not_fusible(z_out)

        # 准备输入数据，包括一个 CPU 张量和两个 CUDA 张量
        inputs = [
            torch.randn(4, 4, dtype=torch.float),
            torch.randn(4, 4, dtype=torch.float, device='cuda:0'),
            torch.randn(4, 4, dtype=torch.float, device='cuda:1'),
        ]

        # 获取当前 JIT 编译器的融合内核规格缓存大小
        prev_cache_size = torch._C._jit_debug_fuser_num_cached_kernel_specs()

        # 调用 checkScript 方法检查融合函数，并获取其图形表示
        ge = self.checkScript(fn, inputs)
        # 断言图中确实包含 3 个 FusionGroup
        self.assertGraphContainsExactly(
            ge.graph_for(*inputs), 'prim::FusionGroup', 3, True)
        
        # 获取更新后的融合内核规格缓存大小
        new_cache_size = torch._C._jit_debug_fuser_num_cached_kernel_specs()
        # 断言新缓存大小比旧缓存大小增加了一个内核规格
        # XXX: 这假设同一个内核没有被其它测试使用
        self.assertEqual(new_cache_size - prev_cache_size, 1)

    # 在不支持多GPU运行时跳过此测试
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    def test_nonzero_device_cuda(self):
        # 设置设备为 cuda:1
        device = 'cuda:' + str(1)
        # 创建两个浮点数张量 x 和 y，并指定设备为 CUDA
        x = torch.tensor([0.4], dtype=torch.float, device=device)
        y = torch.tensor([0.7], dtype=torch.float, device=device)

        # 定义一个函数 doit，对输入的 x 和 y 执行一系列计算
        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y) + x))

        # 使用 checkTrace 方法检查 doit 函数的追踪版本，并获取其图形表示
        ge = self.checkTrace(doit, (x, y))
        # 断言该图中所有操作都已融合
        self.assertAllFused(ge.graph_for(x, y))

    # 在不支持 CUDA 运行时跳过此测试
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_lstm_cuda(self):
        # 获取 LSTM 的 CUDA 输入数据
        inputs = get_lstm_inputs('cuda', training=True)
        # 使用 checkScript 方法检查 LSTMCellS 模块，并获取其表示
        module = self.checkScript(LSTMCellS, inputs)
        return
        # 获取前向图表示
        forward_graph = module.graph_for(*inputs)
        # 断言前向图中确实包含一个 FusionGroup
        self.assertGraphContainsExactly(
            forward_graph, 'prim::FusionGroup', 1, consider_subgraphs=True)
        # 断言剥离后的前向图节点数量为 2
        self.assertTrue(len(strip_profiling_nodes(forward_graph.nodes())) == 2)
        # 执行文件检查以确保所有操作都可微分，但 TupleConstruct 返回
        FileCheck().check("DifferentiableGraph").check_next("TupleConstruct") \
            .check_next("return").run(str(forward_graph))

        # 启用测试专用的性能分析模式
        with enable_profiling_mode_for_profiling_tests(True):
            # 调用模块前向传播
            hy, cy = module(*inputs)
            # 进行反向传播的热身
            warmup_backward((hy + cy).sum())
            # 获取反向传播图
            backward = backward_graph(module)
        # 断言所有操作都已融合，除了 "aten::t", "aten::mm", "aten::_grad_sum_to_size" 之外的操作
        self.assertAllFused(backward, except_for=("aten::t", "aten::mm",
                                                  "aten::_grad_sum_to_size"))

    # 在不支持 CUDA 运行时跳过此测试
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # 默认情况下，在 Ampere 或更高版本的 GPU 上，LSTM 使用 TF32 精度计算浮点张量
    # 我们希望使用完整精度来计算浮点张量，以使用默认精度
    @with_tf32_off
    # 定义一个测试方法，用于在 CUDA 上测试 LSTM 模型的融合情况
    def test_lstm_concat_cuda(self):
        # 获取在 CUDA 上的 LSTM 输入数据
        inputs = get_lstm_inputs('cuda')
        # 检查 LSTMCellC 类的跟踪情况，并生成计算图
        ge = self.checkTrace(LSTMCellC, inputs)
        # 获取计算图
        graph = ge.graph_for(*inputs)
        # 使用 FileCheck 检查计算图，确保包含 "FusedConcat" 和 "return"
        FileCheck().check("FusedConcat").check_next("return").run(str(graph))

    # 在不支持 CUDA 的情况下跳过测试
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # 测试 LSTM 门的排列组合在 CUDA 上的融合情况
    def test_lstm_gates_permutations_cuda(self):
        # LSTM 的门包括 x.mm(w_ih.t()) + hx.mm(w_hh.t()) + b_ih + b_hh
        # 测试任何排列组合是否仍然会得到一个 FusionGroup
        choices = ['x.mm(w_ih.t())', 'hx.mm(w_hh.t())', 'b_ih', 'b_hh']
        # 模板代码，定义了一个 cell 函数，用于测试排列组合的效果
        template = dedent('''
        def cell(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
            gates = {} + {} + {} + {}
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            return ingate * forgetgate * cellgate * outgate
        ''')
        # 遍历所有排列组合
        for permutation in permutations(choices, len(choices)):
            # 根据排列组合填充模板代码
            code = template.format(*permutation)
            scope = {}
            # 在全局作用域执行模板代码
            exec(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)

            # 获取在 CUDA 上的 LSTM 输入数据
            inputs = get_lstm_inputs('cuda', training=False)
            # 断言通过编译单元 cu 计算的结果与 scope['cell'] 函数的结果相同
            self.assertEqual(cu.cell(*inputs), scope['cell'](*inputs))
            # 获取 cell 函数的前向计算图
            forward_graph = cu.cell.graph_for(*inputs)
            # 断言前向计算图中确切包含一个 "prim::FusionGroup"
            self.assertGraphContainsExactly(forward_graph, 'prim::FusionGroup', 1)

    # 在不支持 CUDA 的情况下跳过测试
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # 测试在 CUDA 上追踪 LSTM 的情况
    @with_tf32_off
    def test_lstm_traced_cuda(self):
        # 获取在 CUDA 上的 LSTM 输入数据
        inputs = get_lstm_inputs('cuda')
        # 检查 LSTMCellF 类的跟踪情况，并生成计算图
        ge = self.checkTrace(LSTMCellF, inputs)
        # 获取计算图
        graph = ge.graph_for(*inputs)
        # 使用 FileCheck 检查计算图，确保不包含 "aten::add", "Chunk", "aten::sigmoid", "aten::tanh"，但包含 "FusionGroup" 和 "TupleConstruct"
        FileCheck().check_not("aten::add").check_not("Chunk").check_not("aten::sigmoid") \
            .check_not("aten::tanh").check("FusionGroup").check_next("TupleConstruct") \
            .check_next("return").check_not("FusionGroup_2").run(str(graph))

    # 在 Sandcastle 上不支持 CPU 融合支持时跳过测试
    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    # 测试存在问题，跳过测试
    @unittest.skip("Test is flaky, see https://github.com/pytorch/pytorch/issues/8746")
    # 启用 CPU 上的融合支持
    @enable_cpu_fuser
    # 测试 LSTMCellF 在 CPU 上的追踪功能
    def test_lstm_traced_cpu(self):
        # 获取 CPU 上的 LSTM 输入
        inputs = get_lstm_inputs('cpu')
        try:
            # 检查 LSTMCellF 的追踪结果
            ge = self.checkTrace(LSTMCellF, inputs)
            # 获取追踪后的计算图
            graph = ge.graph_for(*inputs)
            # 检查计算图中是否包含融合组件 FusionGroup
            FileCheck.check("FusionGroup").run(str(graph))
        except RuntimeError as e:
            if 'Failed to compile' in e.args[0]:
                # 如果编译失败，发出警告，并跳过测试
                warnings.warn('CPU fuser test has failed! This is not a hard failure, '
                              'because the kernels sometimes trigger bugs in compilers '
                              '(most notably GCC 7.2).')
                raise unittest.SkipTest('Failed to compile') from e
            else:
                raise

    # 测试 MiLSTMCell 在 CUDA 上的功能
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_milstm_cuda(self):
        # 获取 CUDA 上的 MiLSTMCell 输入
        inputs = get_milstm_inputs('cuda', training=True)
        # 检查 MiLSTMCell 的脚本模块
        module = self.checkScript(MiLSTMCell, inputs)
        # 获取前向传播的计算图
        forward_graph = module.graph_for(*inputs)
        # 断言前向传播计算图中确切包含一个融合组件 FusionGroup
        self.assertGraphContainsExactly(
            forward_graph, 'prim::FusionGroup', 1, consider_subgraphs=True)
        # 使用 FileCheck 检查前向传播计算图中的内容
        FileCheck().check("DifferentiableGraph").check_next("TupleConstruct") \
            .check_next("return").check("FusionGroup").run(str(forward_graph))
        # 执行模块的前向传播
        hy, cy = module(*inputs)
        # 对前向传播结果进行热身后向传播
        warmup_backward((hy + cy).sum())

    # 测试随机数生成在 CUDA 上的功能
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.LEGACY, "borked on the legacy executor")
    def test_rand_cuda(self):
        # 定义一个带有随机数生成方法的脚本模块 M
        class M(torch.jit.ScriptModule):
            __constants__ = ['d']

            def __init__(self):
                super().__init__()
                self.d = torch.device('cuda')

            @torch.jit.script_method
            def create(self, x):
                return x * x + x + torch.rand_like(x)

        # 在 CUDA 上创建一个张量 x
        x = torch.zeros([3, 4, 5], dtype=torch.float, device='cuda')
        # 实例化 M 类
        m = M()
        # 使用 M 类的 create 方法生成两个输出
        out1 = m.create(x)
        out2 = m.create(x)
        # 确保两次生成的输出不相等
        self.assertNotEqual(out1, out2)
        # 断言 out1 的所有元素都大于等于 0
        self.assertTrue(torch.all(out1 >= 0))
        # 断言 out1 的所有元素都小于 1
        self.assertTrue(torch.all(out1 < 1))
        # 断言 out2 的所有元素都大于等于 0
        self.assertTrue(torch.all(out2 >= 0))
        # 断言 out2 的所有元素都小于 1
        self.assertTrue(torch.all(out2 < 1))
        # 断言在 create 方法的计算图中所有操作都被融合
        self.assertAllFused(m.create.graph_for(x))

    # 静态方法：测试在 CUDA 上的 relu 函数
    @staticmethod
    def fn_test_relu(x, y):
        return F.relu(x + .5 * y)

    # 测试在 CUDA 上的 relu 函数
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_relu_cuda(self):
        # 在 CUDA 上创建两个随机张量 x 和 y
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        # 检查 relu 函数的追踪结果
        ge = self.checkTrace(self.fn_test_relu, (x, y))
        # 断言在 relu 函数的计算图中所有操作都被融合
        self.assertAllFused(ge.graph_for(x, y))

    # 测试在 CUDA 上的 relu 函数，如果没有 CUDA，则跳过测试
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # 定义名为 test_erf_cuda 的测试方法
    def test_erf_cuda(self):
        # 定义内部函数 fn_test_erf，计算 torch.erf(x) - torch.erfc(x) 的结果并通过 F.relu 进行修正
        def fn_test_erf(x):
            return F.relu(torch.erf(x) - torch.erfc(x))

        # 在 CUDA 设备上生成一个随机张量 x，形状为 4x4，数据类型为 float
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        # 对 fn_test_erf 函数进行 TorchScript 脚本化，并检查脚本化后的图形
        ge = self.checkTrace(fn_test_erf, (x,))
        # 断言所有操作都已融合在一起，即在计算图中没有额外的操作
        self.assertAllFused(ge.graph_for(x))
        # 将张量 x 设置为需要梯度计算
        x.requires_grad_(True)
        # 再次对 fn_test_erf 函数进行 TorchScript 脚本化，并检查脚本化后的图形
        ge = self.checkTrace(fn_test_erf, (x,))
        # 断言所有操作都已融合在一起，但排除 "aten::size", "prim::BroadcastSizes", "aten::_size_if_not_equal" 这些操作
        self.assertAllFused(ge.graph_for(x), except_for=("aten::size", "prim::BroadcastSizes",
                                                         "aten::_size_if_not_equal"))

    # 如果不支持 CUDA 或者设置中 GRAPH_EXECUTOR 为 ProfilingMode.LEGACY，则跳过此测试
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.LEGACY, "borked on the legacy executor")
    def test_rand_broadcast_cuda(self):
        # 定义内部函数 fn_test_rand，生成与 y 张量相同形状的随机张量 r，然后返回 r * x + x 的结果
        def fn_test_rand(x, y):
            r = torch.rand_like(y)
            return r * x + x

        # 在 CUDA 设备上生成两个随机张量 x 和 y，形状为 4x4，数据类型为 float
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')
        # 对 fn_test_rand 函数进行 TorchScript 脚本化
        script_f = torch.jit.script(fn_test_rand)
        # 对脚本化后的函数应用 x 和 y 作为输入，并检查脚本化后的图形
        out = script_f(x, y)
        self.assertAllFused(script_f.graph_for(x, y))
        # 将张量 x 设置为需要梯度计算
        x.requires_grad_(True)
        # 再次对脚本化后的函数应用 x 和 y 作为输入，并检查脚本化后的图形，排除指定的操作
        out = script_f(x, y)
        self.assertAllFused(script_f.graph_for(x, y), except_for=("aten::size", "prim::BroadcastSizes",
                                                                  "aten::_size_if_not_equal"))
        # 测试广播随机生成的结果是否正确
        x = torch.ones(4, 4, dtype=torch.float, device='cuda')
        y = torch.ones(4, dtype=torch.float, device='cuda')
        out = script_f(x, y)
        # 断言张量 out 的第一个元素等于第二个元素
        self.assertEqual(out[0], out[1])

    # 如果在 Sandcastle 上运行或设置中启用了 CPU fuser，则跳过此测试
    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @enable_cpu_fuser
    def test_scalar(self):
        # 定义内部函数 fn，计算 2 * x + y 的结果
        def fn(x, y):
            return 2 * x + y

        # 在 CPU 设备上生成两个张量 x 和 y，分别是 0.1 和 1，数据类型为 float
        x = torch.tensor(0.1, dtype=torch.float, device='cpu')
        y = torch.tensor(1, dtype=torch.float, device='cpu')
        # 对 fn 函数进行 TorchScript 脚本化，并检查脚本化后的图形
        ge = self.checkScript(fn, (x, y))
        # 断言所有操作都已融合在一起
        self.assertAllFused(ge.graph_for(x, y))

    # 如果不支持 CUDA，则跳过此测试
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_small_constant_cuda(self):
        # 定义内部函数 fn_test_small_constant，计算 (1e-8 * x + 5e-9 * y) * 1e8 的结果
        def fn_test_small_constant(x, y):
            return (1e-8 * x + 5e-9 * y) * 1e8
        
        # 在 CUDA 设备上生成两个随机张量 x 和 y，形状为 4x4，数据类型为 float
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')
        # 对 fn_test_small_constant 函数进行 Trace 跟踪
        ge = self.checkTrace(fn_test_small_constant, (x, y))
        # 断言所有操作都已融合在一起
        self.assertAllFused(ge.graph_for(x, y))

    # 如果不支持 CUDA，则跳过此测试
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # 定义一个测试函数，用于测试在 CUDA 上的张量标量操作
    def test_tensor_scalar_ops_cuda(self):
        # 内部定义一个函数 should_fuse，用于测试是否能够融合操作
        def should_fuse(x):
            # 设置一个标量 z 为 3.0
            z = 3.
            # 计算 y = x + z
            y = x + z
            # 返回 x * y 的结果
            return x * y

        # XXX: 目前只有在标量是常数时才支持融合 (#9940)
        # 定义一个函数 should_not_fuse，测试当标量不是常数时的情况
        def should_not_fuse(x, z):
            # 计算 y = x + int(z)
            y = x + int(z)
            # 返回 x * y 的结果
            return x * y

        # 创建一个输入列表，包含一个在 CUDA 上的随机张量
        inputs = [torch.randn(2, 2, dtype=torch.float, device='cuda')]
        # 对 should_fuse 函数进行脚本化，并检查融合情况
        ge = self.checkScript(should_fuse, inputs)
        # 断言所有操作都已融合在一起
        self.assertAllFused(ge.graph_for(*inputs))

        # 创建另一个输入列表，包含一个在 CUDA 上的随机张量和一个标量张量
        inputs = [
            torch.randn(2, 2, dtype=torch.float, device='cuda'),
            torch.tensor(3., dtype=torch.float, device='cuda'),
        ]
        # 对 should_not_fuse 函数进行脚本化，并检查不融合情况
        ge = self.checkScript(should_not_fuse, inputs)
        # 断言图中确实包含一个 FusionGroup，但不考虑子图
        self.assertGraphContainsExactly(
            ge.graph_for(*inputs), 'prim::FusionGroup', 0, consider_subgraphs=True)

    # 如果在 Sandcastle 环境中，则跳过测试
    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @enable_cpu_fuser
    # 定义测试函数 test_where_and_typing
    def test_where_and_typing(self):
        # 内部定义一个函数 f，测试 torch.where 和类型注解
        def f(x, y):
            # 创建一个掩码，用于比较 x 和 y 的大小关系
            mask = x > y
            # 根据掩码选择 x 或 y 的元素形成结果张量
            res = torch.where(mask, x, y)
            # 返回掩码和结果张量
            return mask, res

        # 创建两个随机张量 x 和 y，数据类型为 double
        x = torch.randn(4, 4, dtype=torch.double)
        y = torch.randn(4, 4, dtype=torch.double)

        # 对函数 f 进行脚本化，并检查所有操作是否融合，除了 TupleConstruct
        script_f = self.checkScript(f, (x, y))
        self.assertAllFused(script_f.graph_for(x, y), except_for={'prim::TupleConstruct'})

    # 如果不运行在 CUDA 上，则跳过测试
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # 如果图执行器不是遗留模式，则跳过测试
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "no half support with profiling on")
    # 定义测试函数 test_grad_sum_to_size_elimination，使用了 self 参数，表明这是一个测试类的方法
    def test_grad_sum_to_size_elimination(self):

        # 定义内部函数 my_broadcasted_cell，实现对三个参数进行广播相加的操作
        def my_broadcasted_cell(a, b, c):
            return (a + b) + c

        # 创建两个张量 s1 和 s2，使用随机数据初始化，设备为 CUDA，且需要梯度跟踪
        s1 = torch.randn(5, 1, requires_grad=True, device='cuda')
        s2 = torch.randn(5, 5, requires_grad=True, device='cuda')

        # 调用 self.checkScript 方法，对 my_broadcasted_cell 函数进行脚本化检查，使用 s1 作为参数
        module = self.checkScript(my_broadcasted_cell, (s1, s1, s1), profiling=ProfilingMode.PROFILING)
        # 获取前向计算图
        forward_graph = module.graph_for(s1, s1, s1)
        # 断言所有操作都已融合，除了 "aten::size", "prim::BroadcastSizes", "aten::_size_if_not_equal" 之外的操作
        self.assertAllFused(forward_graph, except_for=("aten::size", "prim::BroadcastSizes",
                                                       "aten::_size_if_not_equal"))

        # 初始化一个空集合 old_plans，用于存储旧的计划图
        old_plans = set()
        # 循环三次
        for i in range(3):
            # 如果 i < 1，则使用 s2 替换部分参数，触发 _grad_sum_to_size 操作
            args = s2 if i < 1 else s1, s2 if i < 2 else s1, s2
            # 将参数 args 中的每个张量进行去除梯度并重新跟踪梯度的操作
            args = [a.detach_().requires_grad_() for a in args]
            # 重新编译模块，以避免触发中止操作
            module = self.checkScript(my_broadcasted_cell, args, profiling=ProfilingMode.PROFILING)
            # 调用模块进行前向传播计算
            res = module(s2 if i < 1 else s1, s2 if i < 2 else s1, s2)
            # 对结果进行反向传播，计算梯度
            warmup_backward(res.sum(), args)
            grads = torch.autograd.grad(res.sum(), args)
            # 断言每个输入参数的形状与其对应梯度的形状相同
            for inp, gr in zip(args, grads):
                self.assertEqual(inp.shape, gr.shape)
            backward = None
            # 处理 Python 2 中反向图未按顺序的问题的临时解决方案
            for g in all_backward_graphs(module):
                if str(g) not in old_plans:
                    assert backward is None
                    backward = g
                    old_plans.add(str(backward))
            # 如果 i > 0，则预期存在一个 _grad_sum_to_size 操作节点
            num_grads = 1 if i > 0 else 0
            # 断言计算得到的反向图中 _grad_sum_to_size 操作节点的数量符合预期
            self.assertEqual(len([n for n in backward.nodes() if n.kind() == 'aten::_grad_sum_to_size']), num_grads)
# 如果当前脚本被直接执行（而不是作为模块被导入），则执行以下代码
if __name__ == '__main__':
    # 调用函数 run_tests()，用于执行测试或主程序的入口点
    run_tests()
```
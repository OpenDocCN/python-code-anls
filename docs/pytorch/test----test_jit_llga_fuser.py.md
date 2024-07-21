# `.\pytorch\test\test_jit_llga_fuser.py`

```py
# Owner(s): ["module: mkldnn"]

# 导入系统相关模块
import sys
# 导入 PyTorch
import torch
# 导入单元测试模块
import unittest
# 导入迭代工具模块
import itertools
# 导入 PyTorch 神经网络模块
import torch.nn as nn
# 导入装饰器相关模块
from functools import wraps
# 导入并发执行相关模块
from concurrent import futures
# 导入 PyTorch 的函数式接口模块
import torch.nn.functional as F
# 导入 PyTorch 的 FX 实验性优化模块
import torch.fx.experimental.optimization as optimization
# 导入 TorchVision 的测试工具模块
from torch.testing._internal.jit_utils import JitTestCase
# 导入 PyTorch 的内部通用测试工具模块
from torch.testing._internal.common_utils import run_tests, TEST_SCIPY, IS_WINDOWS, IS_MACOS
# 导入 PyTorch 的内部通用设备类型测试模块
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    dtypes
)

# 由于 JIT 追踪存在内存泄漏问题，导致追踪的模型对象在内存中持久存在，因此我们使用此包装器来运行 TorchVision 模型的单元测试
# 这些单元测试现在在单独的进程中运行，以避免内存泄漏问题导致的内存累积
def separate_process(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with futures.ProcessPoolExecutor() as executor:
            future = executor.submit(func, *args, **kwargs)
            futures.wait([future])
    return wrapper

# 检查当前平台是否支持 AVX-512 指令集
def is_avx512_supported():
    if sys.platform != 'linux':
        return False
    with open("/proc/cpuinfo", encoding="ascii") as f:
        lines = f.read()
    return "avx512" in lines

# 判断是否不支持 AVX-512 指令集
IS_AVX512_UNSUPPORTED = not is_avx512_supported()

# 定义 LLGA Fusion Group 的名称
LLGA_FUSION_GROUP = 'prim::oneDNNFusionGroup'
# 判断是否不支持 LLGA 优化（MKLDNN 不可用或者在 Windows 或 macOS 上）
LLGA_NOT_ENABLED = not torch.backends.mkldnn.is_available() or IS_WINDOWS or IS_MACOS

# 预热模型前向传播函数，以便进行性能分析
def warmup_forward(f, *args, profiling_count=3):
    for i in range(profiling_count):
        results = f(*args)

    return results

# 继承 JitTestCase 类，用于定义 LLGA 相关的测试用例
class JitLlgaTestCase(JitTestCase):

    # 设置测试环境，在测试之前执行
    def setUp(self):
        # 由于 JIT 模式下和 eager 模式下 AMP 的支持存在差异，我们在 JIT 模式下禁用 AMP，并利用 eager 模式下的 AMP
        self.original_autocast_mode = torch._C._jit_set_autocast_mode(False)
        torch.jit.enable_onednn_fusion(True)

    # 清理测试环境，在测试之后执行
    def tearDown(self):
        torch.jit.enable_onednn_fusion(False)
        torch._C._jit_set_autocast_mode(self.original_autocast_mode)
    # 检查是否需要追踪给定模型 `m` 在输入数据 `x` 上的执行路径，确保输出与参考输出一致
    def checkTrace(self, m, x, dtype=torch.float32, *args, **kwargs):
        # 如果 `m` 是 torch.nn.Module 类型，则设置为评估模式
        if isinstance(m, torch.nn.Module):
            m.eval()
        
        # 使用 `torch.no_grad()` 和 `_disable_emit_hooks()` 禁用梯度计算和 JIT 编译的 hook
        with torch.no_grad(), torch._jit_internal._disable_emit_hooks():
            # 如果数据类型为 torch.bfloat16
            if dtype == torch.bfloat16:
                # 依赖于 eager 模式下的自动混合精度（AMP）支持 BF16
                with torch.cpu.amp.autocast(cache_enabled=False, dtype=torch.bfloat16):
                    # 对模型 `m` 和输入数据 `x` 进行跟踪
                    traced = torch.jit.trace(m, x)
                    # 如果 `m` 是 torch.nn.Module 类型，则冻结跟踪后的模型
                    if isinstance(m, torch.nn.Module):
                        traced = torch.jit.freeze(traced)
                    # 预热跟踪后的模型的前向推理
                    warmup_forward(traced, *x)
                    # 获取参考输出 `ref_o`，即使用原始模型 `m` 进行前向推理得到的输出
                    ref_o = m(*x)
                    # 获取跟踪后模型的计算图
                    fwd_graph = traced.graph_for(*x)
            else:
                # 对模型 `m` 和输入数据 `x` 进行跟踪
                traced = torch.jit.trace(m, x)
                # 如果 `m` 是 torch.nn.Module 类型，则冻结跟踪后的模型
                if isinstance(m, torch.nn.Module):
                    traced = torch.jit.freeze(traced)
                # 预热跟踪后的模型的前向推理
                warmup_forward(traced, *x)
                # 获取参考输出 `ref_o`，即使用原始模型 `m` 进行前向推理得到的输出
                ref_o = m(*x)
                # 获取跟踪后模型的计算图
                fwd_graph = traced.graph_for(*x)

            # 使用跟踪后的模型 `traced` 对输入数据 `x` 进行前向推理
            jit_o = traced(*x)
            # 使用断言检查 JIT 后的输出 `jit_o` 是否与参考输出 `ref_o` 一致
            self.assertEqual(jit_o, ref_o)
            # 返回跟踪后的模型 `traced` 和其生成的前向计算图 `fwd_graph`
            return traced, fwd_graph


    # 断言给定的计算图 `graph` 中确实包含指定的融合模式 `fused_patterns`
    def assertFused(self, graph, fused_patterns):
        # 遍历每个融合模式 `pat`，并使用 `assertGraphContainsExactly` 断言其在 `graph` 中确切地出现 0 次
        for pat in fused_patterns:
            self.assertGraphContainsExactly(graph, pat, 0)

    # 在给定的计算图 `graph` 中查找所有融合组，返回找到的结果列表 `result`
    def findFusionGroups(self, graph):
        result = []
        # 遍历计算图 `graph` 的所有节点 `n`
        for n in graph.nodes():
            # 如果节点 `n` 的类型为 LLGA_FUSION_GROUP，则将其子图添加到结果列表 `result` 中
            if n.kind() == LLGA_FUSION_GROUP:
                result.append(n.g('Subgraph'))
                continue
            # 否则，递归地在节点 `n` 的每个子块中查找融合组，并将结果追加到 `result` 中
            for block in n.blocks():
                result += self.findFusionGroups(block)
        # 返回找到的所有融合组的列表 `result`
        return result

    # 检查给定计算图 `graph` 是否确实包含指定的模式列表 `patterns`
    def checkPatterns(self, graph, patterns):
        # 查找给定计算图 `graph` 中的所有融合组
        fusion_groups = self.findFusionGroups(graph)
        # 使用断言确保找到的融合组数量与给定模式列表 `patterns` 的长度相等
        assert len(fusion_groups) == len(patterns), "length of subgraphs not equal to length of given patterns"

        # 遍历每个融合组 `fusion_groups[i]` 和对应的模式列表 `patterns[i]`
        for i in range(len(fusion_groups)):
            # 遍历模式列表 `patterns[i]` 中的每个模式 `pattern`
            for pattern in patterns[i]:
                # 使用 `assertGraphContains` 断言融合组 `fusion_groups[i]` 中包含模式 `pattern`
                self.assertGraphContains(fusion_groups[i], pattern)
#`
# 尝试导入torchvision模块，如果成功导入则设置HAS_TORCHVISION为True
try:
    import torchvision
    HAS_TORCHVISION = True
# 如果导入错误，则将HAS_TORCHVISION设置为False
except ImportError:
    HAS_TORCHVISION = False
# 如果运行时出现错误（RuntimeError），同样将HAS_TORCHVISION设置为False
except RuntimeError:
    HAS_TORCHVISION = False

# 使用unittest.skipIf装饰器，如果没有安装torchvision，则跳过测试
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, 'no torchvision')

# 根据给定的函数名获取对应的元素操作函数
def get_eltwise_fn(name):
    # 如果torch模块中有对应名称的属性，则返回该属性
    if hasattr(torch, name):
        return getattr(torch, name)
    # 如果torch.nn.functional模块中有对应名称的属性，则返回该属性
    elif hasattr(F, name):
        return getattr(F, name)
    # 如果名称为'hardswish_'，则返回一个带有inplace=True参数的Hardswish实例
    elif name == 'hardswish_':
        return torch.nn.Hardswish(inplace=True)
    # 如果以上条件都不满足，则引发NameError异常，指出未找到对应名称的元素操作函数
    else:
        raise NameError(f'Eltwise function {name} not found')

# 使用unittest.skipIf装饰器，如果IS_AVX512_UNSUPPORTED为True，则跳过测试
@unittest.skipIf(IS_AVX512_UNSUPPORTED, "This test fails for BF16 on machines without AVX512.")
# 使用unittest.skipIf装饰器，如果LLGA_NOT_ENABLED为True，则跳过测试
@unittest.skipIf(LLGA_NOT_ENABLED, "MKL-DNN build is disabled")
# 定义一个测试类TestOp，继承自JitLlgaTestCase
class TestOp(JitLlgaTestCase):
    # 使用onlyCPU装饰器，测试方法仅在CPU上运行
    @onlyCPU
    # 使用dtypes装饰器，测试方法接受torch.float32和torch.bfloat16两种数据类型作为参数
    @dtypes(torch.float32, torch.bfloat16)
    # 定义测试方法test_conv2d，接受dtype参数
    def test_conv2d(self, dtype):
        # 使用itertools.product生成多个参数组合，依次遍历每种组合
        for [spatial, in_channels, out_channels, kernel, padding, stride, dilation, g, bias] in itertools.product(
                [7, 8],
                [8, 15],
                [7, 16],
                [3, 4],
                [0, 2],
                [1, 2],
                [1, 2],
                [1, 2],
                [True, False]):

            # 创建一个卷积层m，参数根据当前的参数组合确定
            m = nn.Conv2d(in_channels=in_channels * g,
                          out_channels=out_channels * g,
                          kernel_size=kernel,
                          padding=padding,
                          stride=stride,
                          dilation=dilation,
                          groups=g,
                          bias=bias)

            # 生成随机输入张量x
            x = torch.rand(1, in_channels * g, spatial, spatial)
            # 调用self.checkTrace方法对卷积层m进行跟踪，返回计算结果和计算图
            _, graph = self.checkTrace(m, [x], dtype)
            # 断言计算图中精确包含LLGA_FUSION_GROUP类型的操作一次
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    # 使用onlyCPU装饰器，测试方法仅在CPU上运行
    @onlyCPU
    # 使用dtypes装饰器，测试方法接受torch.float32和torch.bfloat16两种数据类型作为参数
    @dtypes(torch.float32, torch.bfloat16)
    # 定义测试方法test_bn2d，接受dtype参数
    def test_bn2d(self, dtype):
        # 创建一个BatchNorm2d层m，并将其设置为评估模式
        m = nn.BatchNorm2d(32).eval()
        # 生成随机输入张量x
        x = torch.rand(1, 32, 28, 28)
        # 调用self.checkTrace方法对BatchNorm2d层m进行跟踪，返回计算结果和计算图
        _, graph = self.checkTrace(m, [x], dtype)
        # 断言计算图中精确不包含LLGA_FUSION_GROUP类型的操作
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)

    # 使用onlyCPU装饰器，测试方法仅在CPU上运行
    @onlyCPU
    # 使用dtypes装饰器，测试方法接受torch.float32和torch.bfloat16两种数据类型作为参数
    @dtypes(torch.float32, torch.bfloat16)
    # 定义测试方法test_eltwise，接受dtype参数
    def test_eltwise(self, dtype):
        # 定义一个M类，继承自nn.Module，用于测试元素操作函数
        class M(nn.Module):
            def __init__(self, eltwise_fn):
                super().__init__()
                self.eltwise = eltwise_fn

            def forward(self, xfloat32, torch.bfloat16)
    # 定义测试函数，用于测试 MaxPool2d 层的功能
    def test_max_pool2d(self, dtype):
        # 使用 itertools.product 生成所有参数组合进行测试
        for [spatial, kernel, padding, stride, dilation, ceil_mode] in itertools.product(
                [15, 16, 17, 18, 19],    # 空间尺寸
                [4, 5],                  # 卷积核大小
                [0, 1, 2],               # 填充大小
                [1, 2],                  # 步幅大小
                [1],                     # 膨胀大小
                [True, False]):          # 是否使用 ceil 模式

            # 创建 MaxPool2d 层对象
            m = nn.MaxPool2d(kernel_size=kernel,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             ceil_mode=ceil_mode)

            # 创建随机输入张量
            x = torch.rand(1, 4, spatial, spatial)
            # 进行跟踪并返回计算图
            _, graph = self.checkTrace(m, [x], dtype)
            # 断言计算图中包含正好一个 LLGA_FUSION_GROUP
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    # 限定测试仅在 CPU 上执行，测试 AvgPool2d 层的功能
    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_avg_pool2d(self, dtype):
        # 使用 itertools.product 生成所有参数组合进行测试
        for [spatial, kernel, padding, stride, ceil_mode, count_include_pad] in itertools.product(
                [15, 16, 17, 18, 19],    # 空间尺寸
                [4, 5],                  # 卷积核大小
                [0, 1, 2],               # 填充大小
                [1, 2, 4],               # 步幅大小
                [False],                 # 是否使用 ceil 模式（目前不完全支持）
                [True, False]):          # 是否包括填充值在内

            # 创建 AvgPool2d 层对象
            m = nn.AvgPool2d(kernel_size=kernel,
                             stride=stride,
                             padding=padding,
                             ceil_mode=ceil_mode,
                             count_include_pad=count_include_pad)

            # 创建随机输入张量
            x = torch.rand(1, 4, spatial, spatial)
            # 进行跟踪并返回计算图
            _, graph = self.checkTrace(m, [x], dtype)
            # 断言计算图中包含正好一个 LLGA_FUSION_GROUP
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    # 限定测试仅在 CPU 上执行，测试可变核大小的 AvgPool2d 层的功能
    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_variable_kernel_avg_pool2d(self, dtype):
        # 定义一个简单的 Module 类型，重写其 forward 方法
        class M(nn.Module):
            def forward(self, x):
                # 使用 F.avg_pool2d 进行平均池化，核大小为输入张量的空间维度，不包含填充值
                x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0, count_include_pad=False)
                return x

        # 创建一个随机输入张量
        x = torch.randn(1, 1000, 1, 1)
        # 创建 M 类的实例
        m = M()
        # 进行跟踪并返回计算图
        _, graph = self.checkTrace(m, [x], dtype)
        # 断言计算图中不包含任何 LLGA_FUSION_GROUP
        # TODO: 使用形状特化后，应包含一个 LLGA_FUSION_GROUP
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)

    # 限定测试仅在 CPU 上执行，测试 Softmax 层的功能
    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_softmax(self, dtype):
        # 使用 itertools.product 生成所有维度参数组合进行测试
        for dim in [-4, -3, -2, -1, 0, 1, 2, 3]:
            # 创建 Softmax 层对象
            m = nn.Softmax(dim=dim)
            # 创建随机输入张量
            x = torch.rand(8, 12, 12, 12)
            # 进行跟踪并返回计算图
            _, graph = self.checkTrace(m, [x], dtype)
            # 断言计算图中不包含任何 LLGA_FUSION_GROUP
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)
    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    # 定义一个测试函数，用于测试线性操作
    def test_linear(self, dtype):
        # 遍历是否包含偏置项的情况
        for bias in [True, False]:
            # 创建随机张量 x
            x = torch.rand(32, 28)
            # 创建线性层模型 m，指定输入维度为 28，输出维度为 64，是否包含偏置由 bias 决定
            m = torch.nn.Linear(in_features=28, out_features=64, bias=bias)
            # 对模型进行追踪并获取计算图
            _, graph = self.checkTrace(m, [x], dtype)
            # 断言计算图中包含一个 LLGA_FUSION_GROUP 节点
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
            # 断言计算图中融合了 'aten::linear' 操作
            self.assertFused(graph, ['aten::linear'])


    # 生成二进制输入的生成器函数
    def _gen_binary_inputs(self, gen_permute=True):
        # 遍历多组输入形状的情况
        for xshape, yshape in [
            [[1, 32, 28, 28], [1, 32, 28, 28]],
            [[1, 32, 28, 28], [1, 1, 28, 28]],
            [[1, 32, 28, 28], [28]],
            [[1, 32, 28, 28], [1]],
        ]:
            # 生成随机张量 x 和 y
            yield torch.rand(xshape), torch.rand(yshape)
            # 如果需要生成置换的情况，并且 xshape 不等于 yshape
            if gen_permute and xshape != yshape:
                # 生成随机张量 y 和 x 的置换
                yield torch.rand(yshape), torch.rand(xshape)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    # 定义一个测试函数，用于测试加法操作
    def test_add(self, dtype):
        # 定义一个前向传播函数，实现 x 和 y 的加法，并乘以 2
        def forward_add(x, y):
            return torch.add(x, y, alpha=2)

        # 遍历生成的二进制输入
        for x, y in self._gen_binary_inputs():
            # 对前向传播函数进行追踪并获取计算图
            _, graph = self.checkTrace(forward_add, [x, y], dtype)
            # 断言计算图中包含一个 LLGA_FUSION_GROUP 节点
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    # 定义一个测试函数，用于测试加法操作（对标量）
    def test_add_scalar(self, dtype):
        # 定义一个加法操作，将 42 和 x 相加，再加上 3.14
        def add_scalar(x):
            return 42 + x + 3.14

        # 创建随机张量 x
        x = torch.rand(32, 32)
        # 对加法操作进行追踪并获取计算图
        _, graph = self.checkTrace(add_scalar, [x], dtype)
        # 断言计算图中包含一个 LLGA_FUSION_GROUP 节点
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    # 定义一个测试函数，用于测试 addmm 操作
    def test_addmm(self, dtype):
        # 定义一个 addmm 操作，其中 alpha 和 beta 默认为 1
        def addmm(x, y, z):
            return torch.addmm(z, x, y)

        # 创建随机张量 x, y, z
        x = torch.rand(64, 32)
        y = torch.rand(32, 32)
        z = torch.rand(64, 32)
        # 对 addmm 操作进行追踪并获取计算图
        _, graph = self.checkTrace(addmm, [x, y, z], dtype)
        # 断言计算图中包含一个 LLGA_FUSION_GROUP 节点
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    # 定义一个测试函数，用于测试乘法操作
    def test_mul(self, dtype):
        # 定义一个前向传播函数，实现 x 和 y 的乘法，并乘以 3
        def forward_mul(x, y):
            return torch.mul(x, y) * 3

        # 遍历生成的二进制输入
        for x, y in self._gen_binary_inputs():
            # 对前向传播函数进行追踪并获取计算图
            _, graph = self.checkTrace(forward_mul, [x, y], dtype)
            # 断言计算图中包含一个 LLGA_FUSION_GROUP 节点
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    # 定义一个测试函数，用于测试身份二进制操作
    def test_identity_binary(self, dtype):
        # 定义一个前向传播函数，实现 x * 1 + 0.0 的操作
        def forward(x):
            return x * 1 + 0.0

        # 创建随机张量 x
        x = torch.rand(32)
        # 对前向传播函数进行追踪并获取计算图
        _, graph = self.checkTrace(forward, [x], dtype)
        # 断言计算图中融合了 'aten::add' 和 'aten::mul' 操作
        self.assertFused(graph, ['aten::add', 'aten::mul'])
    # 定义一个测试函数，用于测试 LayerNorm 模块
    def test_layer_norm(self, dtype):
        # TODO: support more normalized_shape
        # 创建一个具有输入大小为 10 的 LayerNorm 模块实例
        m = torch.nn.LayerNorm(10)
        # 生成一个形状为 (2, 5, 10, 10) 的随机张量作为输入
        x = torch.randn(2, 5, 10, 10)
        # 检查并追踪模块 m 在输入 x 上的计算图，并返回计算图对象
        _, graph = self.checkTrace(m, [x], dtype)
        # 断言计算图中确切包含一个 LLGA_FUSION_GROUP（特定融合组）操作
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    # 装饰器，限制该测试仅在 CPU 上运行，并支持指定的数据类型
    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_cat(self, dtype):
        # 定义一个按维度连接张量的内部函数
        def cat_along_dim(d):
            # 定义实际进行按维度连接的前向函数
            def forward_cat(*inputs):
                return torch.cat(inputs, d)
            return forward_cat

        # 遍历不同形状的输入张量列表
        for xshape in [
            [8, 8, 8, 8],
            [64, 8, 32],
            [2048, 64],
        ]:
            # 遍历当前输入张量的各个维度
            for d in range(len(xshape)):
                # 生成指定形状的随机张量 x
                x = torch.rand(xshape)
                # 检查并追踪按维度连接函数 cat_along_dim(d) 在输入 x 上的计算图，并返回计算图对象
                _, graph = self.checkTrace(cat_along_dim(d), [x, x, x], dtype)
                # 断言计算图中确切包含一个 LLGA_FUSION_GROUP 操作
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    # 装饰器，限制该测试仅在 CPU 上运行，并支持指定的数据类型
    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_typecheck(self, dtype):
        # 生成指定形状的随机张量 x，指定数据类型为 dtype
        x = torch.rand(32, 28, dtype=dtype)
        # 创建一个线性模块，指定输入特征为 28，输出特征为 64，包括偏置，并指定数据类型为 dtype
        m = torch.nn.Linear(in_features=28, out_features=64, bias=True, dtype=dtype)
        # 检查并追踪模块 m 在输入 x 上的计算图，并返回追踪后的模块和计算图对象
        traced, graph = self.checkTrace(m, [x], dtype)
        # 断言计算图中确切包含一个 LLGA_FUSION_GROUP 操作
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        # 断言计算图中融合了 'aten::linear' 操作
        self.assertFused(graph, ['aten::linear'])
        # 改变输入张量的形状，预期会进入回退计算图
        x = torch.rand(5, 28, dtype=dtype)
        # 断言模块 m 在改变形状后的输入 x 上的输出与追踪后的模块在相同输入 x 上的输出一致
        self.assertEqual(m(x), traced(x))
# 如果 AVX512 不支持，则跳过此测试；主要针对不支持 AVX512 的机器上的 BF16 运行失败情况
@unittest.skipIf(IS_AVX512_UNSUPPORTED, "This test fails for BF16 on machines without AVX512.")
# 如果 LLGA 没有启用，则跳过此测试；主要针对未启用 MKL-DNN 构建的情况
@unittest.skipIf(LLGA_NOT_ENABLED, "MKL-DNN build is disabled")
# 定义测试类 TestFusionPattern，继承自 JitLlgaTestCase
class TestFusionPattern(JitLlgaTestCase):

    # 标记仅在 CPU 下运行的测试
    @onlyCPU
    # 参数化测试数据类型为 torch.float32 和 torch.bfloat16
    @dtypes(torch.float32, torch.bfloat16)
    def test_conv2d_eltwise(self, dtype):
        # 定义模型类 M，继承自 nn.Module
        class M(nn.Module):
            def __init__(self, eltwise_fn):
                super().__init__()
                # 第一个卷积层，输入通道数 32，输出通道数 32，卷积核大小 3x3，填充 1，带偏置
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                # 第二个卷积层，输入通道数 32，输出通道数 32，卷积核大小 3x3，填充 1，不带偏置
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
                # eltwise 操作，传入的函数为 eltwise_fn
                self.eltwise = eltwise_fn

            # 前向传播函数
            def forward(self, x):
                # 第一次卷积操作
                x = self.conv1(x)
                # eltwise 操作
                x = self.eltwise(x)
                # 第二次卷积操作
                x = self.conv2(x)
                # eltwise 操作
                x = self.eltwise(x)
                return x

        # 遍历不同的 eltwise 函数和 inplace 参数组合
        for eltwise in ['relu', 'leaky_relu', 'sigmoid', 'square',
                        'abs', 'exp', 'hardswish', 'tanh', 'hardtanh']:
            for inplace in [True, False]:
                # 构造 eltwise 函数名，如果 inplace 为 True，则在函数名后加上 '_'
                eltwise_fn_name = eltwise + '_' if inplace else eltwise
                # 获取 eltwise 函数对象
                eltwise_fn = get_eltwise_fn(eltwise_fn_name)

                # 创建模型实例 m，传入 eltwise 函数对象
                m = M(eltwise_fn)
                # 创建输入数据 x，大小为 1x32x28x28 的随机张量
                x = torch.rand(1, 32, 28, 28)
                # 进行模型的 JIT 编译和图的检查，返回编译后的模型和图
                _, graph = self.checkTrace(m, [x], dtype=dtype)
                # 断言图中确切包含 LLGA_FUSION_GROUP 的融合组操作，次数为 2 次
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 2)
                # 测试是否通过变异删除 pass 将 relu_ 替换为 relu
                self.assertFused(graph, ['aten::' + eltwise_fn_name])
                # 测试是否将 relu 融合到融合组中
                self.assertFused(graph, ['aten::' + eltwise])

    # 标记仅在 CPU 下运行的测试
    @onlyCPU
    # 参数化测试数据类型为 torch.float32 和 torch.bfloat16
    @dtypes(torch.float32, torch.bfloat16)
    # 定义一个测试函数，用于测试 Conv2d 和 SiLU 激活函数的组合
    def test_conv2d_silu(self, dtype):
        # 定义一个简单的神经网络模型类
        class M(nn.Module):
            # 初始化函数，设置模型的结构
            def __init__(self, inplace):
                super().__init__()
                # 第一个卷积层，输入和输出通道数都是 32，卷积核大小为 3x3，填充为 1，包含偏置项
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                # 第二个卷积层，输入和输出通道数都是 32，卷积核大小为 3x3，填充为 1，包含偏置项
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                # SiLU 激活函数层，inplace 参数决定是否原地操作
                self.eltwise = nn.SiLU(inplace=inplace)

            # 前向传播函数
            def forward(self, x):
                # 第一次卷积操作
                x = self.conv1(x)
                # 应用 SiLU 激活函数
                x = self.eltwise(x)
                # 第二次卷积操作
                x = self.conv2(x)
                return x

        # 针对 inplace 参数为 False 和 True 分别进行测试
        for inplace in [False, True]:
            # 针对内存格式为 torch.contiguous_format 和 torch.channels_last 分别进行测试
            for memory_format in [torch.contiguous_format, torch.channels_last]:
                # 创建 M 类的实例
                m = M(inplace)
                # 创建一个随机张量作为输入 x，大小为 [1, 32, 28, 28]，并指定内存格式
                x = torch.rand(1, 32, 28, 28).to(memory_format=memory_format)

                # 调用自定义函数 checkTrace，返回计算结果和图形表示
                _, graph = self.checkTrace(m, [x], dtype)
                # 断言图中确实包含两个 LLGA_FUSION_GROUP 模式的融合操作
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 2)
                # 提示信息：oneDNN 图中不包含 silu 操作，桥接将 silu 转换为 sigmoid - mul 操作
                # 原地操作将在 JIT 图中变成非原地操作
                patterns = [
                    ["aten::_convolution", 'aten::sigmoid', 'aten::mul'],
                    ["aten::_convolution"]
                ]
                # 断言图中存在融合的 Conv2d 和 Silu 操作
                silu_op = 'aten::silu_' if inplace else 'aten::silu'
                self.assertFused(graph, ['aten::_convolution', silu_op])
                # 检查图中的模式匹配情况
                self.checkPatterns(graph, patterns)

    # 仅在 CPU 上运行的装饰器，指定数据类型为 torch.float32 和 torch.bfloat16
    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_ensure_tensor_is_rewrapped(self, dtype):
        # 定义一个测试方法，确保张量被正确地重包装
        class M(nn.Module):
            def __init__(self, eltwise_fn):
                super().__init__()
                # 定义四个卷积层，每个卷积层都是32通道的3x3卷积，带有填充和偏置
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv4 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                # 设置元素级操作函数
                self.eltwise = eltwise_fn
                # 定义一个自适应平均池化层，输出尺寸为5x7
                self.adaptive_avg_pool_2d = nn.AdaptiveAvgPool2d((5, 7))

            def forward(self, x, y):
                # 第一个卷积操作
                x = self.conv1(x)
                # 使用元素级操作函数
                x = self.eltwise(x)
                # 第二个卷积操作
                x = self.conv2(x)
                # 使用元素级操作函数
                x = self.eltwise(x)
                # 第三个卷积操作
                y = self.conv3(y)
                # 使用元素级操作函数
                y = self.eltwise(y)
                # 第四个卷积操作
                y = self.conv4(y)
                # 使用元素级操作函数
                y = self.eltwise(y)

                # 张量相加
                x = torch.add(x, y)
                # 执行自适应平均池化操作
                x = self.adaptive_avg_pool_2d(x)
                return x

        # 设置元素级操作函数名称为'relu'
        eltwise_fn_name = 'relu'
        # 获取对应的元素级操作函数
        eltwise_fn = get_eltwise_fn(eltwise_fn_name)
        # 创建M类的实例对象m，传入元素级操作函数
        m = M(eltwise_fn)
        # 将m对象移动到指定的内存格式torch.channels_last
        m = m.to(memory_format=torch.channels_last)
        # 创建输入张量x和y，形状为[1, 32, 28, 28]，并设置为channels_last内存格式
        x = torch.rand(1, 32, 28, 28).to(memory_format=torch.channels_last)
        y = torch.rand(1, 32, 28, 28).to(memory_format=torch.channels_last)
        # 简单测试输出是否准确
        # 第二个分区的输出作为adaptive_avg_pool2d的输入，这在LLGA中不受支持。
        # 在resnext101 32x16d模型中，我们遇到了准确性问题。
        _, graph = self.checkTrace(m, [x, y], dtype)
        # 断言图中确实包含LLGA_FUSION_GROUP，数量为4
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 4)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_conv2d_clamp(self, dtype):
        # 定义一个名为 test_conv2d_clamp 的测试方法，接受一个 dtype 参数

        class M(nn.Module):
            # 定义一个名为 M 的内部类，继承自 nn.Module

            def __init__(self):
                # 初始化方法
                super().__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                # 创建一个卷积层，输入通道 32，输出通道 32，卷积核大小 3x3，填充 1，包含偏置
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                # 创建第二个卷积层，与前面类似
                self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                # 创建第三个卷积层，与前面类似
                self.conv4 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                # 创建第四个卷积层，与前面类似
                self.conv5 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                # 创建第五个卷积层，与前面类似

            def forward(self, x):
                # 前向传播方法，接受输入张量 x

                x = self.conv1(x)
                # 对输入 x 进行第一次卷积操作
                x = torch.clamp(x, min=float('-inf'))
                # 对张量 x 进行元素级别的下限裁剪，将小于负无穷的值裁剪为负无穷
                x = self.conv2(x)
                # 对裁剪后的张量 x 进行第二次卷积操作
                x = torch.clamp(x, min=-5)
                # 再次对张量 x 进行元素级别的下限裁剪，将小于 -5 的值裁剪为 -5
                x = self.conv3(x)
                # 对裁剪后的张量 x 进行第三次卷积操作
                x = torch.clamp(x, min=0, max=float('inf'))
                # 对张量 x 进行元素级别的上下限裁剪，将小于 0 的值裁剪为 0，大于正无穷的值裁剪为正无穷
                x = self.conv4(x)
                # 对裁剪后的张量 x 进行第四次卷积操作
                x = torch.clamp(x, min=1, max=5)
                # 对张量 x 进行元素级别的上下限裁剪，将小于 1 的值裁剪为 1，大于 5 的值裁剪为 5
                x = self.conv5(x)
                # 对裁剪后的张量 x 进行第五次卷积操作
                x = torch.clamp(x, max=2)
                # 对张量 x 进行元素级别的上限裁剪，将大于 2 的值裁剪为 2
                return x
                # 返回处理后的张量 x

        for inplace in [False, True]:
            # 遍历 inplace 参数为 False 和 True 的情况
            for memory_format in [torch.contiguous_format, torch.channels_last]:
                # 遍历 memory_format 参数为 torch.contiguous_format 和 torch.channels_last 的情况
                x = torch.rand(1, 32, 28, 28).to(memory_format=memory_format)
                # 生成一个随机张量 x，形状为 (1, 32, 28, 28)，并根据 memory_format 进行格式化
                m = M()
                # 创建一个 M 类的实例 m
                _, graph = self.checkTrace(m, [x], dtype)
                # 对模型 m 进行追踪，获取计算图
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 5)
                # 断言计算图中融合组件 LLGA_FUSION_GROUP 的数量为 5
                self.assertFused(graph, ['aten::_convolution', "aten::clamp"])
                # 断言计算图中包含 'aten::_convolution' 和 'aten::clamp' 这两种操作

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_conv2d_bn(self, dtype):
        # 定义一个名为 test_conv2d_bn 的测试方法，仅在 CPU 上执行，接受一个 dtype 参数

        class M(nn.Module):
            # 定义一个名为 M 的内部类，继承自 nn.Module

            def __init__(self):
                # 初始化方法
                super().__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                # 创建一个卷积层，输入通道 32，输出通道 32，卷积核大小 3x3，填充 1，包含偏置
                self.bn1 = nn.BatchNorm2d(32)
                # 创建一个批标准化层，输入通道 32

            def forward(self, x):
                # 前向传播方法，接受输入张量 x

                x = self.conv1(x)
                # 对输入 x 进行卷积操作
                x = self.bn1(x)
                # 对卷积后的结果 x 进行批标准化操作
                return x
                # 返回处理后的张量 x

        m = M().eval()
        # 创建一个 M 类的实例 m，并将其设置为评估模式
        if dtype == torch.bfloat16:
            m = optimization.fuse(m)
            # 如果 dtype 是 torch.bfloat16，则尝试对模型 m 进行融合优化
        x = torch.rand(1, 32, 28, 28)
        # 生成一个随机张量 x，形状为 (1, 32, 28, 28)
        _, graph = self.checkTrace(m, [x], dtype)
        # 对模型 m 进行追踪，获取计算图
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        # 断言计算图中融合组件 LLGA_FUSION_GROUP 的数量为 1
        self.assertFused(graph, ['aten::_convolution', 'aten::batch_norm'])
        # 断言计算图中包含 'aten::_convolution' 和 'aten::batch_norm' 这两种操作
    # 定义一个测试函数，测试卷积层、批归一化和ReLU激活函数的组合
    def test_conv2d_bn_relu(self, dtype):
        # 定义一个简单的神经网络模型
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                # 添加卷积层，输入和输出通道数都是32，卷积核大小为3x3，填充为1，包含偏置
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                # 添加批归一化层，通道数为32
                self.bn1 = nn.BatchNorm2d(32)

            def forward(self, x):
                # 模型前向传播过程：卷积 -> 批归一化 -> ReLU激活函数
                x = self.conv1(x)
                x = self.bn1(x)
                x = F.relu(x)
                return x

        # 创建并评估模型
        m = M().eval()
        # 如果指定了数据类型为torch.bfloat16，则优化模型
        if dtype == torch.bfloat16:
            m = optimization.fuse(m)
        # 创建随机输入张量，形状为[1, 32, 28, 28]
        x = torch.rand(1, 32, 28, 28)
        # 使用自定义方法检查模型的计算图
        _, graph = self.checkTrace(m, [x], dtype)
        # 断言计算图中确实包含指定类型的融合操作组
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        # 断言计算图中融合了指定的操作，包括卷积、批归一化和ReLU激活函数
        self.assertFused(graph, ['aten::_convolution', 'aten::batch_norm',
                                 'aten::relu'])

    # 标记为仅在CPU上运行的测试函数，测试批归一化层与逐元素操作的结合
    @onlyCPU
    # 使用指定数据类型进行测试，包括torch.float32和torch.bfloat16
    @dtypes(torch.float32, torch.bfloat16)
    def test_bn2d_eltwise(self, dtype):
        # 定义一个包含批归一化层和逐元素操作的神经网络模型
        class M(nn.Module):
            def __init__(self, eltwise_fn):
                super().__init__()
                # 添加批归一化层，通道数为32
                self.bn = nn.BatchNorm2d(32)
                # 设置逐元素操作函数
                self.eltwise = eltwise_fn

            def forward(self, x):
                # 模型前向传播过程：批归一化 -> 逐元素操作
                x = self.bn(x)
                x = self.eltwise(x)
                return x

        # 遍历不同类型的逐元素操作函数（例如ReLU）
        for eltwise in ['relu']:
            # 获取指定逐元素操作函数的函数对象
            eltwise_fn = get_eltwise_fn(eltwise)
            # 创建并评估模型
            m = M(eltwise_fn).eval()
            # 创建随机输入张量，形状为[1, 32, 28, 28]
            x = torch.rand(1, 32, 28, 28)
            # 使用自定义方法检查模型的计算图
            _, graph = self.checkTrace(m, [x], dtype)
            # 断言计算图中确实包含指定类型的融合操作组
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
            # 断言计算图中融合了指定的逐元素操作
            self.assertFused(graph, ['aten::' + eltwise])

    # 标记为仅在CPU上运行的测试函数，测试线性层与逐元素操作的结合
    @onlyCPU
    # 使用指定数据类型进行测试，包括torch.float32和torch.bfloat16
    @dtypes(torch.float32, torch.bfloat16)
    def test_linear_eltwise(self, dtype):
        # 定义一个包含线性层和逐元素操作的神经网络模型
        class M(nn.Module):
            def __init__(self, eltwise_fn, bias):
                super().__init__()
                # 添加线性层，输入特征数为28，输出特征数为64，包含偏置参数
                self.linear = nn.Linear(28, 64, bias)
                # 设置逐元素操作函数
                self.eltwise = eltwise_fn

            def forward(self, x):
                # 模型前向传播过程：线性变换 -> 逐元素操作
                x = self.linear(x)
                x = self.eltwise(x)
                return x

        # 遍历不同的偏置存在情况和逐元素操作函数
        for [has_bias, eltwise] in itertools.product(
                [True, False],
                ['relu', 'gelu', 'sigmoid', 'hardtanh', 'relu6', 'elu']):
            # 获取指定逐元素操作函数的函数对象
            eltwise_fn = get_eltwise_fn(eltwise)
            # 创建模型实例
            m = M(eltwise_fn, has_bias)
            # 创建随机输入张量，形状为[32, 28]
            x = torch.rand(32, 28, requires_grad=False)
            # 使用自定义方法检查模型的计算图
            _, graph = self.checkTrace(m, [x], dtype)
            # 断言计算图中确实包含指定类型的融合操作组
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
            # 断言计算图中融合了指定的逐元素操作
            self.assertFused(graph, ['aten::' + eltwise])
    # 定义一个测试函数 test_conv2d_sum，接受一个 dtype 参数
    def test_conv2d_sum(self, dtype):
        # 定义一个名为 M 的内部类，继承自 nn.Module
        class M(nn.Module):
            # 构造方法，初始化神经网络模块
            def __init__(self, bias=False):
                super().__init__()
                # 定义第一个卷积层，输入和输出通道数都为 32，卷积核大小为 3，padding 为 1，是否使用偏置根据 bias 参数确定
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=bias)
                # 定义第一个批归一化层，输入通道数为 32
                self.bn1 = nn.BatchNorm2d(32)
                # 定义第二个卷积层，输入和输出通道数都为 32，卷积核大小为 3，padding 为 1，是否使用偏置根据 bias 参数确定
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=bias)
                # 定义第二个批归一化层，输入通道数为 32
                self.bn2 = nn.BatchNorm2d(32)
                # 定义 ReLU 激活函数
                self.relu = nn.ReLU()
                # 定义第三个卷积层，输入和输出通道数都为 32，卷积核大小为 3，padding 为 1，是否使用偏置根据 bias 参数确定
                self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=bias)
                # 定义第三个批归一化层，输入通道数为 32
                self.bn3 = nn.BatchNorm2d(32)

            # 前向传播方法，接受输入张量 x 和 y
            def forward(self, x, y):
                # 对输入 x 进行第一次卷积操作
                x = self.conv1(x)
                # 对卷积结果进行批归一化
                x = self.bn1(x)
                # 对输入 y 进行第二次卷积操作
                y = self.conv2(y)
                # 对卷积结果进行批归一化
                y = self.bn2(y)
                # 将经过 ReLU 激活的 x 和 y 相加，并经过 ReLU 激活函数
                z = self.relu(x + y)
                # 对相加后的结果进行第三次卷积操作
                z = self.conv3(z)
                # 对卷积结果进行批归一化
                z = self.bn3(z)
                # 返回最终的张量 z
                return z

        # 遍历 bias 取值为 True 和 False 的情况
        for bias in [True, False]:
            # 创建 M 类的实例 m，并设置为评估模式
            m = M(bias).eval()
            # 如果 dtype 是 torch.bfloat16 类型，则调用 optimization.fuse 方法
            if dtype == torch.bfloat16:
                m = optimization.fuse(m)
            # 创建输入张量 x 和 y，大小为 (1, 32, 16, 16)，不需要梯度计算
            x = torch.rand(1, 32, 16, 16, requires_grad=False)
            y = torch.rand(1, 32, 16, 16, requires_grad=False)
            # 调用 self.checkTrace 方法，对模型 m 进行追踪，并返回追踪后的结果和计算图 graph
            _, graph = self.checkTrace(m, [x, y], dtype)
            # 断言计算图 graph 中包含的 LLGA_FUSION_GROUP 数量为 3
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)

    # 仅在 CPU 环境下执行下面的测试函数
    @onlyCPU
    # 标注使用的数据类型为 torch.float32 和 torch.bfloat16
    @dtypes(torch.float32, torch.bfloat16)
    # 定义测试函数 test_wildcard，接受一个 dtype 参数
    def test_wildcard(self, dtype):
        # 定义一个名为 M 的内部类，继承自 nn.Module
        class M(nn.Module):
            # 构造方法，初始化神经网络模块
            def __init__(self):
                super().__init__()
                # 定义一个卷积层，输入和输出通道数都为 32，卷积核大小为 3，padding 为 1，使用偏置
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                # 定义一个 ReLU 激活函数层
                self.eltwise = nn.ReLU()

            # 前向传播方法，接受输入张量 x
            def forward(self, x):
                # 对输入 x 进行卷积操作
                x = self.conv1(x)
                # 对卷积结果进行 ReLU 激活
                y = self.eltwise(x)
                # 返回两个张量的列表，分别是卷积后的结果和经过 ReLU 激活的结果
                return [x, y]

        # 创建 M 类的实例 m
        m = M()
        # 创建输入张量 x，大小为 (1, 32, 28, 28)
        x = torch.rand(1, 32, 28, 28)
        # 调用 self.checkTrace 方法，对模型 m 进行追踪，并返回追踪后的结果和计算图 graph
        _, graph = self.checkTrace(m, [x], dtype)
        # 断言计算图 graph 中包含的 LLGA_FUSION_GROUP 数量为 1
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        # 断言计算图 graph 中包含的融合操作是 'aten::_convolution'
        self.assertFused(graph, ['aten::_convolution'])

    # 仅在 CPU 环境下执行以下测试函数，标注使用的数据类型为 torch.int32
    # 定义一个测试方法，测试不支持的数据类型（dtype）
    def test_wildcard_unsupported_dtype(self, dtype):
        # 定义一个简单的神经网络模块
        class M(nn.Module):
            # 前向传播方法
            def forward(self, x):
                # 执行整数除法操作
                y = x // 2
                return y

        # 在 shufflenet_v2_x1_0 中，channels_per_group 的计算方式为：
        # channels_per_group = num_channels // groups
        # JIT IR 将 groups 转换为不支持的 Long 数据类型，这在 oneDNN 图中不被支持，
        # 例如 Long(requires_grad=0, device=cpu) = prim::Constant[value={2}]()
        # 这个测试仅确保桥接代码能够处理不支持的输入数据类型，针对 oneDNN 图不支持的运算符。
        # 在这个特定的单元测试中，aten::floor_divide 会作为通配符添加到图构建阶段。
        m = M()
        # 创建一个具有指定数据类型的张量 x
        x = torch.tensor([32], dtype=dtype)
        # 检查模型的追踪结果和图
        _, graph = self.checkTrace(m, [x], dtype)
        # 断言图中精确包含 LLGA_FUSION_GROUP 0 次
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)

    # 只在 CPU 上执行的测试，并且仅支持 torch.float32 和 torch.bfloat16 两种数据类型
    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_rewrap_tensor_input_to_pytorch(self, dtype):
        # 定义一个神经网络模块
        class M(nn.Module):
            # 初始化方法
            def __init__(self, eltwise_fn):
                super().__init__()
                # 定义两个卷积层
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                # 设置元素操作函数和自适应平均池化层
                self.eltwise = eltwise_fn
                self.adaptive_avg_pool_2d = nn.AdaptiveAvgPool2d((5, 7))

            # 前向传播方法
            def forward(self, x, y):
                # 第一次卷积操作
                x = self.conv1(x)
                # 应用元素操作函数
                x = self.eltwise(x)
                # 第二次卷积操作
                x = self.conv2(x)
                # 再次应用元素操作函数
                x = self.eltwise(x)
                # 将 x 和 y 相加
                x = torch.add(x, y)
                # 应用自适应平均池化层
                x = self.adaptive_avg_pool_2d(x)
                return x

        # 获取元素操作函数的名称为 'relu'
        eltwise_fn_name = 'relu'
        # 获取对应的元素操作函数
        eltwise_fn = get_eltwise_fn(eltwise_fn_name)
        # 创建 M 类的实例对象
        m = M(eltwise_fn)
        # 将模型转移到 channels_last 的内存格式
        m = m.to(memory_format=torch.channels_last)
        # 创建 channels_last 内存格式的随机张量 x 和 y
        x = torch.rand(1, 32, 28, 28).to(memory_format=torch.channels_last)
        y = torch.rand(1, 32, 28, 28).to(memory_format=torch.channels_last)
        # 简单测试输出是否准确
        # 第二分区的输出是 adaptive_avg_pool2d 的输入，而 LLGA 不支持这一操作，
        # 因此需要 PyTorch 处理，确保 channels_last 张量的正确步幅信息。
        # 检查模型的追踪结果和图
        graph, _ = self.checkTrace(m, [x, y], dtype)
@unittest.skipIf(LLGA_NOT_ENABLED, "MKL-DNN build is disabled")
# 如果LLGA未启用，则跳过测试，显示相应的消息
class TestEnableDisableLlgaFuser(JitTestCase):
    def setUp(self):
        super().setUp()
        # 设置LLGA融合器为禁用状态，并保存当前状态
        self.is_enabled = torch._C._jit_set_llga_enabled(False)

    def tearDown(self):
        # 恢复LLGA融合器的状态为之前保存的状态
        torch._C._jit_set_llga_enabled(self.is_enabled)
        super().tearDown()

    def test_context_manager(self):
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        with torch.jit.fuser('fuser3'):
            with torch.jit.fuser('fuser3'):

                def t1(x, y):
                    # 执行张量运算：x + y
                    o = x + y
                    # 执行张量运算：o + 2.0
                    o = o + 2.0
                    return o
                # 将函数t1编译为Torch脚本
                t_jit = torch.jit.script(t1)
                # 使用编译后的脚本进行张量计算
                t_jit(x, y)
                t_jit(x, y)
                # 断言编译后的计算图包含LLGA融合组
                self.assertGraphContains(t_jit.graph_for(x, y), LLGA_FUSION_GROUP)

            def t2(x, y):
                # 执行张量运算：x + y
                o = x + y
                # 执行张量运算：o + 3.0
                o = o + 3.0
                return o
            # 将函数t2编译为Torch脚本
            t_jit_2 = torch.jit.script(t2)
            # 使用编译后的脚本进行张量计算
            t_jit_2(x, y)
            t_jit_2(x, y)
            # 断言编译后的计算图包含LLGA融合组
            self.assertGraphContains(t_jit_2.graph_for(x, y), LLGA_FUSION_GROUP)

        def t3(x, y):
            # 执行张量运算：x + y
            o = x + y
            # 执行张量运算：o + 4.0
            o = o + 4.0
            return o
        # 将函数t3编译为Torch脚本
        t_jit_3 = torch.jit.script(t3)
        # 使用编译后的脚本进行张量计算
        t_jit_3(x, y)
        t_jit_3(x, y)
        # 断言编译后的计算图确切包含0个LLGA融合组
        self.assertGraphContainsExactly(t_jit_3.graph_for(x, y), LLGA_FUSION_GROUP, 0)


@unittest.skipIf(LLGA_NOT_ENABLED, "MKL-DNN build is disabled")
@unittest.skip("Enable when integration with dynamo aot_autograd is more stable")
# 如果LLGA未启用，跳过测试，并显示相应的消息；当前测试被暂时禁用
class TestDynamoAOT(JitTestCase):
    def test_dynamo_aot_ts_onednn(self):
        class Seq(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建包含多个层的序列神经网络模型
                self.layers = nn.Sequential(
                    nn.Linear(10, 10),
                    nn.ReLU(),
                    nn.Linear(10, 10),
                    nn.ReLU(),
                )

            def forward(self, x):
                # 应用序列中的层来进行前向传播
                return self.layers(x)

        mod = Seq()

        import torch._dynamo
        # 使用torch._dynamo.optimize优化模型为AOT编译形式，并启用无Python运行
        aot_mod = torch._dynamo.optimize("aot_ts", nopython=True)(mod)

        for _ in range(10):
            with torch.jit.fuser("fuser3"):
                # 计算损失，对模型执行反向传播
                loss = aot_mod(torch.rand([10, 10])).sum()
                loss.backward()

        # 重置Dynamo状态
        torch._dynamo.reset()


@unittest.skipIf(IS_AVX512_UNSUPPORTED, "This test fails for BF16 on machines without AVX512.")
@unittest.skipIf(LLGA_NOT_ENABLED, "MKL-DNN build is disabled")
# 如果IS_AVX512_UNSUPPORTED为真，则跳过测试，显示相应的消息；如果LLGA未启用，跳过测试，并显示相应的消息
class TestModel(JitLlgaTestCase):
    @skipIfNoTorchVision
    def _test_vision(self, model_name, dtype):
        m = getattr(torchvision.models, model_name)().eval()
        if dtype == torch.bfloat16:
            # 对模型应用优化融合
            m = optimization.fuse(m)
        x = torch.rand(1, 3, 224, 224) / 10
        _, graph = self.checkTrace(m, [x], dtype)
        # 断言计算图中包含一系列特定的操作
        self.assertFused(graph, ['aten::_convolution', 'aten::batch_norm',
                                 'aten::relu', 'aten::linear',
                                 'aten::avg_pool2d', 'aten::max_pool2d'])
# 遍历模型名和是否启用的列表
for model_name, enabled in [
    ['resnet50', True],  # 模型名称为'resnet50'，启用标志为True
    ['resnext50_32x4d', True],  # 模型名称为'resnext50_32x4d'，启用标志为True
    ['resnext101_32x8d', True],  # 模型名称为'resnext101_32x8d'，启用标志为True
    ['densenet121', True],  # 模型名称为'densenet121'，启用标志为True
    ['densenet161', True],  # 模型名称为'densenet161'，启用标志为True
    ['densenet169', True],  # 模型名称为'densenet169'，启用标志为True
    ['densenet201', True],  # 模型名称为'densenet201'，启用标志为True
    ['efficientnet_b0', True],  # 模型名称为'efficientnet_b0'，启用标志为True
    ['efficientnet_b1', True],  # 模型名称为'efficientnet_b1'，启用标志为True
    ['efficientnet_b2', True],  # 模型名称为'efficientnet_b2'，启用标志为True
    ['efficientnet_b3', True],  # 模型名称为'efficientnet_b3'，启用标志为True
    ['efficientnet_b4', True],  # 模型名称为'efficientnet_b4'，启用标志为True
    ['efficientnet_b5', True],  # 模型名称为'efficientnet_b5'，启用标志为True
    ['efficientnet_b6', True],  # 模型名称为'efficientnet_b6'，启用标志为True
    ['efficientnet_b7', True],  # 模型名称为'efficientnet_b7'，启用标志为True
    ['regnet_y_400mf', True],  # 模型名称为'regnet_y_400mf'，启用标志为True
    ['googlenet', TEST_SCIPY],  # 模型名称为'googlenet'，启用标志为TEST_SCIPY的值（未提供具体定义）
    ['mobilenet_v2', True],  # 模型名称为'mobilenet_v2'，启用标志为True
    ['mobilenet_v3_large', True],  # 模型名称为'mobilenet_v3_large'，启用标志为True
    ['mnasnet1_0', True],  # 模型名称为'mnasnet1_0'，启用标志为True
    ['squeezenet1_0', True],  # 模型名称为'squeezenet1_0'，启用标志为True
    ['vgg16', True],  # 模型名称为'vgg16'，启用标志为True
    ['alexnet', True],  # 模型名称为'alexnet'，启用标志为True
    ['shufflenet_v2_x1_0', True],  # 模型名称为'shufflenet_v2_x1_0'，启用标志为True
    ['wide_resnet50_2', True],  # 模型名称为'wide_resnet50_2'，启用标志为True
]:
    # 定义一个包装函数，接受模型名和数据类型作为参数
    def _wrapper(mname, dtype):
        # 定义一个测试函数，使用unittest.skipIf装饰器来根据enabled的值判断是否跳过测试
        @unittest.skipIf(not enabled, 'Disabled')
        @separate_process
        def test(self, dtype=dtype):
            # 调用测试方法_test_vision，传入模型名和数据类型，返回测试结果
            return self._test_vision(mname, dtype)
        return test

    # 遍历数据类型列表 [torch.bfloat16, torch.float32]
    for dtype in [torch.bfloat16, torch.float32]:
        # 设置TestModel类的测试方法名，格式为'test_vision_<模型名>_<数据类型>'
        setattr(TestModel, 'test_vision_{}_{}'.format(model_name, str(dtype).split("torch.")[1]), _wrapper(model_name, dtype))

# 调用函数instantiate_device_type_tests，传入TestFusionPattern类和全局变量，实例化设备类型测试
instantiate_device_type_tests(TestFusionPattern, globals())
# 调用函数instantiate_device_type_tests，传入TestOp类和全局变量，实例化设备类型测试
instantiate_device_type_tests(TestOp, globals())

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == '__main__':
    run_tests()
```
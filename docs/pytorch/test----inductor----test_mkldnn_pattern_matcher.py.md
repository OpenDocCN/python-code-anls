# `.\pytorch\test\inductor\test_mkldnn_pattern_matcher.py`

```py
# Owner(s): ["oncall: cpu inductor"]
# 引入所需的模块和库
import contextlib  # 上下文管理工具模块
import copy  # 复制对象模块
import itertools  # 迭代工具模块
import unittest  # 单元测试模块

import torch  # PyTorch核心库
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq  # X86 Inductor量化器
from torch._dynamo import config as dynamo_config  # 动态配置模块
from torch._dynamo.utils import counters  # 计数器工具模块
from torch._export import capture_pre_autograd_graph  # 捕获前自动求导图模块
from torch._inductor import config, metrics  # Inductor配置和度量模块
from torch._inductor.test_case import run_tests, TestCase  # 测试用例模块
from torch._inductor.utils import run_and_get_code  # 运行并获取代码模块
from torch.ao.quantization.quantize_pt2e import (  # PT2E量化模块
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer  # X86 Inductor量化器
from torch.nn import functional as F  # 神经网络的函数库别名F
from torch.testing._internal.common_quantization import (  # 量化测试通用模块
    skipIfNoDynamoSupport,
    skipIfNoONEDNN,
    skipIfNoONEDNNBF16,
)
from torch.testing._internal.common_utils import IS_LINUX, skipIfRocm, TEST_MKL  # 测试通用工具
from torch.testing._internal.inductor_utils import _check_has_dynamic_shape, HAS_CPU  # Inductor工具

# The dict value is match_nodes(computation_op+unary_op)
# unary_list字典存储了一组键为torch.nn模块的实例，值为整数的键值对
unary_list = {
    torch.nn.ReLU(): 2,
    torch.nn.Sigmoid(): 2,
    torch.nn.Tanh(): 2,
    torch.nn.Hardswish(): 6,
    torch.nn.LeakyReLU(0.1, inplace=False): 4,
    # Use floats for min/max, otherwise they can get converted to symints
    torch.nn.Hardtanh(min_val=-0.5, max_val=4.0, inplace=False): 3,
    torch.nn.Hardtanh(min_val=-0.5, max_val=float("inf"), inplace=False): 3,
    torch.nn.GELU(approximate="none"): 6,
    torch.nn.GELU(approximate="tanh"): 10,
    torch.nn.ReLU6(): 3,
    torch.nn.SiLU(): 3,
    torch.nn.Hardsigmoid(): 5,
}

# non_decomposed_unary_list列表存储了一组torch.nn模块的类，用于表示不可分解的一元操作
non_decomposed_unary_list = [
    torch.nn.ReLU,
    torch.nn.Sigmoid,
    torch.nn.Tanh,
]

# The dict value is (match_count, match_nodes, inplace)
# binary_list字典存储了一组键为lambda函数（表示二元操作）的键值对
binary_list = {
    lambda x, y: torch.add(x, y): (1, 2, False),  # call_function
    lambda x, y: torch.add(y, x): (1, 2, False),  # call_function
    lambda x, y: x.add(y): (1, 2, False),  # call_method
    lambda x, y: x.add_(y): (1, 2, True),  # call_method
    lambda x, y: torch.sub(x, y): (1, 2, False),  # call_function
    lambda x, y: x.sub(y): (1, 2, False),  # call_method
    lambda x, y: x.sub_(y): (1, 2, True),  # call_method
}

# quantization_add_fn_list列表存储了一组lambda函数，用于量化中的加法操作
quantization_add_fn_list = [
    lambda x, y: torch.add(x, y),
    lambda x, y: x.add(y),
]

# quantization_inplace_add_fn_list列表存储了一组lambda函数，用于量化中的原位加法操作
quantization_inplace_add_fn_list = [
    lambda x, y: x.add_(y),
]


def get_default_quantizer(is_qat, is_dynamic):
    # 创建默认的X86 Inductor量化器对象
    quantizer = X86InductorQuantizer()
    # 设置全局的X86 Inductor量化配置
    quantizer.set_global(
        xiq.get_default_x86_inductor_quantization_config(
            is_qat=is_qat, is_dynamic=is_dynamic
        )
    )
    return quantizer


def cal_conv_generated_kernel_number(mod, input, dtype):
    # this function is to decide how many kernels are generated
    # while testing conv2d/3d/deconv2d
    # the assumption is:
    #   (1) There will be a to_dtype kernel for input for lp
    #   (2) inductor always use channe_last format, there will
    #       be a to_channel_last format for input
    # 计算在测试conv2d/3d/deconv2d时生成的内核数量的函数
    # 假设：
    #   (1) 对于lp，输入将有一个to_dtype内核
    #   (2) Inductor始终使用channel_last格式，输入将有一个to_channel_last格式
    #   (3) to_dtype and to_channel_last for input can be fused
    #   (4) inductor always get channel last format from mkldnn_conv_pointwise(binary),
    #       and force the output to have the same stride with eager.
    #       So there will be a to_contiguous for output if eager output is contiguous
    mod = copy.deepcopy(mod)
    # 深拷贝模型，确保对原模型没有影响
    input = input.clone()
    # 克隆输入张量，保持原输入不变
    if dtype == torch.float32:
        maybe_autocast = contextlib.nullcontext()
    else:
        maybe_autocast = torch.cpu.amp.autocast(dtype=dtype)
        # 创建一个上下文管理器，根据dtype启用自动混合精度转换
    with torch.no_grad(), maybe_autocast:
        # 关闭梯度计算并应用自动混合精度转换
        output = mod(input)
        # 使用模型进行推理，得到输出张量
    input_kernel, output_kernel = 0, 0
    # 初始化输入和输出内核标记为0
    if (
        input.is_contiguous(memory_format=torch.contiguous_format)
        or dtype != torch.float32
    ):
        input_kernel = 1
        # 如果输入张量是连续的或者dtype不是torch.float32，则将输入内核标记设置为1
    if output.is_contiguous(memory_format=torch.contiguous_format):
        output_kernel = 1
        # 如果输出张量是连续的，则将输出内核标记设置为1
    return input_kernel + output_kernel
    # 返回输入和输出内核标记之和
# 使用装饰器 @config.patch({"freezing": True}) 来修改配置，设置 freezing 参数为 True
@config.patch({"freezing": True})
# 定义一个测试类 TestPatternMatcherBase，继承自 TestCase 类
class TestPatternMatcherBase(TestCase):

    # 检查是否存在未分解的一元函数 unary_fn
    def _check_unary_is_decomposed(self, unary_fn):
        # 返回一个布尔值，指示是否有任何一元函数 unary_fn 是 torch.nn.ReLU、torch.nn.Sigmoid 或 torch.nn.Tanh
        return not any(
            isinstance(unary_fn, fn)
            for fn in [torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh]
        )

    # 克隆输入 inputs 中的元素
    def _clone_inputs(self, inputs):
        # 定义一个内部函数 clone，用于克隆输入 x
        def clone(x):
            # 如果 x 不是 torch.Tensor，则直接返回 x
            if not isinstance(x, torch.Tensor):
                return x
            # 如果 x 是 torch.Tensor，则克隆它并返回
            return x.clone()

        # 对输入 inputs 中的每个元素应用 clone 函数，并返回结果的元组
        return tuple(clone(x) for x in inputs)

    # 生成量化 QDQ 模型
    def _generate_qdq_quantized_model(
        self, mod, inputs, is_qat=False, is_dynamic=False, quantizer=None
    ):
        # 如果是量化训练，则使用 nullcontext 上下文管理器，否则使用 torch.no_grad()
        maybe_no_grad = contextlib.nullcontext() if is_qat else torch.no_grad()
        with maybe_no_grad:
            # 捕获预自动图的图形 export_model
            export_model = capture_pre_autograd_graph(
                mod,
                inputs,
            )
            # 如果未指定 quantizer，则使用默认的量化器
            quantizer = (
                quantizer if quantizer else get_default_quantizer(is_qat, is_dynamic)
            )
            # 根据是否量化训练选择合适的准备函数 prepare_model
            prepare_model = (
                prepare_qat_pt2e(export_model, quantizer)
                if is_qat
                else prepare_pt2e(export_model, quantizer)
            )
            # 在 prepare_model 上应用输入 inputs
            prepare_model(*inputs)
            # 将 prepare_model 转换为 pt2e 格式的模型 convert_model
            convert_model = convert_pt2e(prepare_model)
            # 将导出的模型移动到评估模式
            torch.ao.quantization.move_exported_model_to_eval(convert_model)
            # 返回转换后的模型 convert_model
            return convert_model

    # 通用测试函数
    def _test_common(
        self,
        mod,
        inputs,
        matcher_count=None,
        matcher_nodes=None,
        atol=1e-5,
        rtol=1.3e-6,
        check_autocast=torch.float32,
        check_quantization=False,
        is_qat=False,
        matcher_check_fn=None,
        dtype=None,
        is_dynamic=False,
        quantizer=None,
    ):
        # 清空计数器
        counters.clear()
        # 重置 Torch 动态计算图
        torch._dynamo.reset()
        # 断言匹配器检查函数不为空，或者 matcher_count 和 matcher_nodes 都不为空
        assert matcher_check_fn is not None or (
            matcher_count is not None and matcher_nodes is not None
        )
        # 根据 check_autocast 的类型设置 maybe_autocast 和公差值 atol, rtol
        if (
            check_autocast == torch.bfloat16
            and torch.ops.mkldnn._is_mkldnn_bf16_supported()
        ):
            maybe_autocast = torch.cpu.amp.autocast(dtype=torch.bfloat16)
            atol, rtol = 1e-2, 1e-2
        elif (
            check_autocast == torch.float16
            and torch.ops.mkldnn._is_mkldnn_fp16_supported()
        ):
            maybe_autocast = torch.cpu.amp.autocast(dtype=torch.float16)
            atol, rtol = 1e-2, 1e-2
        else:
            # 当 check_autocast 为 torch.float32 时
            assert check_autocast == torch.float32
            # 使用 nullcontext 创建 maybe_autocast 上下文管理器
            maybe_autocast = contextlib.nullcontext()

        # 如果需要检查量化
        if check_quantization:
            # 生成量化模型并转换
            convert_model = self._generate_qdq_quantized_model(
                mod, inputs, is_qat, is_dynamic, quantizer
            )
            # 使用 torch.no_grad() 上下文，并可能应用自动混合精度
            with torch.no_grad(), maybe_autocast:
                # 编译并执行转换后的模型
                _ = torch.compile(convert_model)(*inputs)
                # 如果 matcher_count 不为空，断言计数器的匹配器数量符合预期
                if matcher_count is not None:
                    self.assertEqual(
                        counters["inductor"]["pattern_matcher_count"], matcher_count
                    )
                # 如果 matcher_nodes 不为空，断言计数器的匹配器节点数符合预期
                if matcher_nodes is not None:
                    self.assertEqual(
                        counters["inductor"]["pattern_matcher_nodes"],
                        matcher_nodes,
                    )
                # 如果 matcher_check_fn 不为空，执行匹配器检查函数
                if matcher_check_fn is not None:
                    matcher_check_fn()
        else:
            # 如果不需要检查量化
            with torch.no_grad(), maybe_autocast:
                # 克隆输入
                clone_inputs = self._clone_inputs(inputs)
                # 计算期望输出
                expected = mod(*inputs)
                # 编译并执行原始模型
                actual = torch.compile(mod)(*clone_inputs)
                # 断言实际输出与期望输出在给定公差下相似
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                # 如果 matcher_count 不为空，断言计数器的匹配器数量符合预期
                if matcher_count is not None:
                    self.assertEqual(
                        counters["inductor"]["pattern_matcher_count"], matcher_count
                    )
                # 如果 matcher_nodes 不为空，断言计数器的匹配器节点数符合预期
                if matcher_nodes is not None:
                    self.assertEqual(
                        counters["inductor"]["pattern_matcher_nodes"],
                        matcher_nodes,
                    )
                # 如果 matcher_check_fn 不为空，执行匹配器检查函数
                if matcher_check_fn is not None:
                    matcher_check_fn()
    ):
        # 在上下文管理器中，禁用 Torch 的梯度计算
        with torch.no_grad():
            # 克隆输入以确保不修改原始输入
            clone_inputs = self._clone_inputs(inputs)
            # 如果需要检查量化情况，生成量化后的模型
            if check_quantization:
                mod = self._generate_qdq_quantized_model(mod, inputs)
            # 计算预期输出
            expected = mod(*inputs)
            # 运行并获取编译后的代码以及其他返回值
            actual, (source_code,) = run_and_get_code(
                torch.compile(mod, fullgraph=True, dynamic=check_dynamic),
                *clone_inputs,
            )
            # 检查包含在源代码中的操作
            for op in include_ops:
                self.assertIn(op, source_code)
            # 如果提供了操作的数量，确保每种操作出现的次数符合预期
            if num_include_ops is not None:
                assert len(include_ops) == len(num_include_ops)
                for i in range(len(include_ops)):
                    self.assertEqual(
                        source_code.count(include_ops[i]), num_include_ops[i]
                    )
            # 检查排除在源代码中的操作
            for op in exclude_ops:
                self.assertNotIn(op, source_code)
            # 如果检查动态形状设置，则调用检查函数
            if check_dynamic is not None:
                _check_has_dynamic_shape(self, source_code)
            # 如果不需要检查量化，跳过由于预设的量化设置导致的测试跳过
            if not check_quantization:
                # 使用指定的绝对误差和相对误差检查实际输出与预期输出的接近程度
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
class TestPatternMatcher(TestPatternMatcherBase):
    # 继承自 TestPatternMatcherBase 的测试类 TestPatternMatcher

    def _test_conv_unary_cpu_base(self, dim=4):
        # 定义测试函数 _test_conv_unary_cpu_base，接受一个维度参数 dim，默认为 4
        assert dim == 4 or dim == 5

        class M(torch.nn.Module):
            # 定义内部类 M，继承自 torch.nn.Module

            def __init__(
                self,
                unary_fn,
                **kwargs,
            ):
                # M 类的构造函数，接受一个一元函数 unary_fn 和其他关键字参数 kwargs
                super().__init__()
                if dim == 4:
                    # 如果 dim 等于 4，则使用二维卷积层
                    self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                else:
                    # 否则，使用三维卷积层
                    self.conv = torch.nn.Conv3d(3, 16, kernel_size=3, stride=1)
                self.unary_fn = unary_fn
                # 将传入的 unary_fn 设置为对象的属性

            def forward(self, x):
                # M 类的前向传播函数，接受输入 x
                x = self.conv(x)
                # 对输入 x 进行卷积操作
                return self.unary_fn(x)
                # 返回经过 unary_fn 处理后的结果

        dtypes = [
            torch.float,
        ]
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            # 如果支持 MKL-DNN 的 BF16 数据类型，则添加到 dtypes 中
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            # 如果支持 MKL-DNN 的 FP16 数据类型，则添加到 dtypes 中
            dtypes.append(torch.float16)
        cl_format = torch.channels_last if dim == 4 else torch.channels_last_3d
        # 根据 dim 的值选择通道布局格式

        options = itertools.product(
            unary_list.keys(),
            [torch.contiguous_format, cl_format],
            dtypes,
        )
        # 生成参数组合的迭代器 options，其中包括一元函数列表的键、内存布局格式和数据类型

        for (
            unary_fn,
            memory_format,
            dtype,
        ) in options:
            # 遍历 options 中的每一个元组
            metrics.reset()
            # 重置性能指标

            if dim == 4:
                x_shape = (1, 3, 56, 56)
            else:
                x_shape = (1, 3, 20, 56, 56)
            # 根据 dim 的值设置输入张量 x_shape 的形状

            mod = M(unary_fn).to(memory_format=memory_format).eval()
            # 创建 M 类的实例 mod，设置内存格式为 memory_format，并进入评估模式

            v = (
                torch.randn(x_shape, dtype=torch.float32)
                .add(1)
                .to(memory_format=memory_format)
            )
            # 生成一个随机张量 v，根据 memory_format 设置内存布局格式

            # Add 1 for weight packing pass.
            match_nodes = unary_list[unary_fn] + 1
            # 获取 unary_fn 对应的匹配节点数，并增加 1

            if dtype in (
                torch.float16,
                torch.bfloat16,
            ) and self._check_unary_is_decomposed(unary_fn):
                # 如果数据类型是 torch.float16 或 torch.bfloat16，并且检查 unary_fn 是否可以分解
                # 则为自动类型转换添加额外的节点
                match_nodes += 2

            self._test_common(mod, (v,), 2, match_nodes, check_autocast=dtype)
            # 调用 _test_common 方法，传递模型 mod、输入 v，以及相关参数，执行通用测试

            generated_kernel_count = cal_conv_generated_kernel_number(mod, v, dtype)
            # 计算卷积操作生成的内核数量

            self.assertEqual(metrics.generated_kernel_count, generated_kernel_count)
            # 断言生成的内核数量与性能指标中记录的数量相等

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_conv2d_unary_cpu(self):
        # 标记为跳过条件不满足的情况下的测试函数 test_conv2d_unary_cpu

        self._test_conv_unary_cpu_base(dim=4)
        # 调用 _test_conv_unary_cpu_base 方法进行二维卷积的测试

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_conv3d_unary_cpu(self):
        # 标记为跳过条件不满足的情况下的测试函数 test_conv3d_unary_cpu

        self._test_conv_unary_cpu_base(dim=5)
        # 调用 _test_conv_unary_cpu_base 方法进行三维卷积的测试
    # 定义一个测试方法，用于测试线性单元运算
    def test_linear_unary(self):
        # 定义一个内部的 PyTorch 模块类 M
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(
                self,
                unary_fn,        # 一元函数
                in_features,     # 输入特征数
                out_features,    # 输出特征数
                bias,            # 是否使用偏置
                **kwargs,        # 其他关键字参数
            ):
                super().__init__()
                # 创建一个线性层
                self.linear = torch.nn.Linear(
                    in_features,
                    out_features,
                    bias,
                    **kwargs,
                )
                self.unary_fn = unary_fn  # 设置一元函数

            # 前向传播方法
            def forward(self, x):
                x = self.linear(x)  # 线性层的前向传播
                return self.unary_fn(x)  # 对线性层的输出应用一元函数

        dtypes = []
        # 检查是否支持 MKL-DNN 的 bfloat16 类型，若支持则添加到 dtypes 列表中
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        # 检查是否支持 MKL-DNN 的 float16 类型，若支持则添加到 dtypes 列表中
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        # 生成 unary_list 和 dtypes 的所有可能组合
        options = itertools.product(unary_list, [True, False], dtypes)
        for unary_fn, bias, dtype in options:
            metrics.reset()  # 重置度量
            # 创建一个 M 类的实例 mod，并设为评估模式
            mod = M(unary_fn, 10, 30, bias=bias).eval()
            v = torch.randn(2, 10)  # 生成一个随机张量 v
            # 包装通过 + 一元融合
            matcher_count = 2
            # 添加 1 用于权重包装通过
            matcher_nodes = unary_list[unary_fn] + 1
            # 检查一元函数是否被分解
            if self._check_unary_is_decomposed(unary_fn):
                # 对于自动类型转换，额外添加 dtype 转换节点
                matcher_nodes += 2
            # 进行公共测试
            self._test_common(
                mod, (v,), matcher_count, matcher_nodes, check_autocast=dtype
            )
            # 仅生成了 1 个 "to" 内核
            self.assertEqual(metrics.generated_kernel_count, 1)

    # 若未开启 MKL，则跳过测试
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    # 定义一个测试方法，用于测试 float32 类型的线性模块
    def test_linear_fp32(self):
        # 定义一个内部的 PyTorch 模块类 M
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self, bias):
                super().__init__()
                # 创建一个线性层
                self.linear = torch.nn.Linear(10, 30, bias)

            # 前向传播方法
            def forward(self, x):
                return self.linear(x)  # 返回线性层的输出

        for bias in [True, False]:
            # 创建一个 M 类的实例 mod，并设为评估模式
            mod = M(bias=bias).eval()
            v = torch.randn(2, 10)  # 生成一个随机张量 v
            # 包装通过
            matcher_count = 1
            matcher_nodes = 1
            # 进行公共测试
            self._test_common(mod, (v,), matcher_count, matcher_nodes)
    def test_linear_add_bias(self):
        # 定义一个测试方法，用于测试带有偏置的线性模块
        class M(torch.nn.Module):
            def __init__(self, dtype, unary_fn, cast_bias):
                super().__init__()
                # 创建第一个线性层，输入维度为10，输出维度为64，无偏置
                self.linear1 = torch.nn.Linear(10, 64, bias=False)
                # 随机初始化64维偏置向量
                self.bias1 = torch.randn(64)
                # 创建第二个线性层，输入维度为10，输出维度为64，无偏置
                self.linear2 = torch.nn.Linear(10, 64, bias=False)
                # 随机初始化64维偏置向量
                self.bias2 = torch.randn(64)
                # 如果需要进行偏置类型转换
                if cast_bias:
                    # 将偏置1转换为指定的数据类型
                    self.bias1 = self.bias1.to(dtype=dtype)
                    # 将偏置2转换为指定的数据类型
                    self.bias2 = self.bias2.to(dtype=dtype)
                # 保存一元函数
                self.unary_fn = unary_fn

            def forward(self, x):
                # 计算第一个线性层的输出并加上偏置1
                a = self.linear1(x) + self.bias1
                # 计算第二个线性层的输出并加上偏置2
                b = self.linear2(x) + self.bias2
                # 返回经过一元函数处理后的结果
                return self.unary_fn(a), self.unary_fn(b)

        dtypes = []
        # 如果支持MKLDNN的BF16格式，则添加到数据类型列表中
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        # 如果支持MKLDNN的FP16格式，则添加到数据类型列表中
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        # 生成一元函数和数据类型的所有组合
        options = itertools.product(unary_list, dtypes)
        for unary_fn, dtype in options:
            # 重置性能指标
            metrics.reset()
            # 创建一个带有指定数据类型和一元函数的M对象，并设置为评估模式
            fold_mod = M(dtype, unary_fn, cast_bias=True).eval()
            v = torch.randn(2, 10)
            matcher_count = 3
            # 添加1个权重打包通过，每个线性层添加2个偏置折叠通过
            matcher_nodes = unary_list[unary_fn] + 3
            if self._check_unary_is_decomposed(unary_fn):
                # 如果一元函数被分解，则增加2个数据类型转换节点用于自动转换
                matcher_nodes += 2
            # 由于有2个线性层，所以将matcher_count和matcher_nodes加倍
            self._test_common(
                fold_mod,
                (v,),
                matcher_count * 2,
                matcher_nodes * 2,
                check_autocast=dtype,
            )
            # 断言生成的内核数量为1
            self.assertEqual(metrics.generated_kernel_count, 1)
            # 如果偏置与权重不是相同的数据类型，则不会对偏置进行折叠
            # 参考：https://github.com/pytorch/pytorch/pull/129138
            metrics.reset()
            # 创建一个不带偏置折叠的M对象，并设置为评估模式
            mod = M(dtype, unary_fn, cast_bias=False).eval()
            self._test_common(mod, (v,), 2, 2, check_autocast=dtype)
            # 1个内核用于“to_lowp”，2个内核用于一元操作
            self.assertEqual(metrics.generated_kernel_count, 3)
    # 定义一个测试用例函数，用于测试转置卷积操作的基本情况
    def _test_conv_transpose_unary_base(self, dim=4):
        # 断言维度参数必须是4或者5
        assert dim == 4 or dim == 5

        # 定义一个内部类M，继承自torch.nn.Module
        class M(torch.nn.Module):
            # 初始化函数，接受一个unary_fn参数和其他关键字参数
            def __init__(
                self,
                unary_fn,
                **kwargs,
            ):
                super().__init__()
                # 根据维度选择不同的转置卷积层
                if dim == 4:
                    self.conv_transpose = torch.nn.ConvTranspose2d(
                        3, 16, 3, stride=2, padding=1
                    )
                else:
                    self.conv_transpose = torch.nn.ConvTranspose3d(
                        3, 16, 3, stride=2, padding=1
                    )
                # 将传入的unary_fn保存在实例中
                self.unary_fn = unary_fn

            # 前向传播函数，接受输入x，执行转置卷积和unary_fn操作后返回结果
            def forward(self, x):
                x = self.conv_transpose(x)
                return self.unary_fn(x)

        # 定义一个数据类型列表
        dtypes = [
            torch.float,
        ]
        # 如果支持MKL-DNN的BF16类型，则添加到数据类型列表中
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        # 如果支持MKL-DNN的FP16类型，则添加到数据类型列表中
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)

        # 根据维度和数据类型列表生成不同的存储格式
        cl_format = torch.channels_last if dim == 4 else torch.channels_last_3d
        options = itertools.product(
            unary_list,  # 使用unary_list中的函数
            [torch.contiguous_format, cl_format],  # 选择不同的存储格式
            dtypes,  # 使用上面定义的数据类型
        )

        # 遍历参数组合
        for unary_fn, memory_format, dtype in options:
            # 重置度量指标
            metrics.reset()
            # 根据维度选择输入张量的形状
            if dim == 4:
                x_shape = (1, 3, 28, 28)
            else:
                x_shape = (1, 3, 17, 28, 28)
            # 创建并评估模型M的实例
            mod = M(unary_fn).eval()

            # 根据指定的存储格式创建随机张量v
            v = torch.randn(x_shape, dtype=torch.float32).to(
                memory_format=memory_format
            )
            # 为权重打包通过，在匹配节点数上加1
            match_nodes = unary_list[unary_fn] + 1
            # 如果数据类型是float16或者bfloat16，并且_unary_is_decomposed(unary_fn)返回True，则为自动转换添加额外的数据类型转换节点
            if dtype in (
                torch.float16,
                torch.bfloat16,
            ) and self._check_unary_is_decomposed(unary_fn):
                match_nodes += 2
            # 执行通用测试函数，检查自动转换
            self._test_common(mod, (v,), 2, match_nodes, check_autocast=dtype)
            # 计算卷积生成的内核数量
            generated_kernel_count = cal_conv_generated_kernel_number(mod, v, dtype)
            # 断言生成的内核数量与度量中的生成内核数量相等
            self.assertEqual(metrics.generated_kernel_count, generated_kernel_count)

    # 使用装饰器标记的测试函数，测试2D转置卷积操作的CPU版本
    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_conv_transpose2d_unary_cpu(self):
        # 调用_test_conv_transpose_unary_base函数，维度设为4
        self._test_conv_transpose_unary_base(dim=4)

    # 使用装饰器标记的测试函数，测试3D转置卷积操作的CPU版本
    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_conv_transpose3d_unary_cpu(self):
        # 调用_test_conv_transpose_unary_base函数，维度设为5
        self._test_conv_transpose_unary_base(dim=5)
    # 定义一个测试函数，用于测试二进制操作的卷积模块，支持二维和三维卷积
    def _test_conv_binary_base(self, dim=4):
        # 断言维度必须是4或5
        assert dim == 4 or dim == 5

        # 定义一个内部类M，继承自torch.nn.Module，用于创建包含卷积层的模型
        class M(torch.nn.Module):
            # 初始化函数，接受二进制操作函数、是否使用ReLU等参数
            def __init__(
                self,
                binary_fn,
                has_relu,
                **kwargs,
            ):
                super().__init__()
                # 根据维度选择创建二维或三维卷积层
                if dim == 4:
                    self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                    self.conv2 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                else:
                    self.conv1 = torch.nn.Conv3d(3, 16, kernel_size=3, stride=1)
                    self.conv2 = torch.nn.Conv3d(3, 16, kernel_size=3, stride=1)
                self.binary_fn = binary_fn  # 存储传入的二进制操作函数
                self.has_relu = has_relu    # 存储是否使用ReLU标志

            # 前向传播函数，接受输入x，执行卷积操作，并根据has_relu决定是否使用ReLU激活
            def forward(self, x):
                x1 = self.conv1(x)  # 执行第一个卷积层
                x2 = self.conv2(x)  # 执行第二个卷积层
                if has_relu:        # 如果has_relu为True
                    return self.binary_fn(x1, x2).relu()  # 使用二进制函数后接ReLU激活函数
                else:
                    return self.binary_fn(x1, x2)  # 否则只返回二进制函数的结果

        # 支持的数据类型列表
        dtypes = [
            torch.float,
        ]
        # 如果支持MKL-DNN的BF16格式，添加BF16数据类型
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        # 如果支持MKL-DNN的FP16格式，添加FP16数据类型
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        # 内存格式，根据维度选择不同的格式
        cl_format = torch.channels_last if dim == 4 else torch.channels_last_3d
        # 测试的内存格式包括连续内存和通道优先内存
        test_memory_format = [torch.contiguous_format, cl_format]
        # 使用itertools生成参数组合
        options = itertools.product(
            binary_list,  # 二进制操作函数列表
            [True, False],  # 是否使用ReLU
            test_memory_format,  # 测试的内存格式
            dtypes,  # 数据类型
        )

        # 遍历所有参数组合
        for (
            binary_fn,
            has_relu,
            memory_format,
            dtype,
        ) in options:
            metrics.reset()  # 重置度量指标
            # 根据维度选择输入张量的形状
            if dim == 4:
                x_shape = (1, 3, 56, 56)
            else:
                x_shape = (1, 3, 20, 56, 56)
            # 创建模型并设为评估模式
            mod = M(binary_fn, has_relu).eval()
            # 生成随机张量，并进行数据类型转换和内存格式设置
            v = (
                torch.randn(x_shape, dtype=torch.float32, requires_grad=True)
                .add(1)
                .to(memory_format=memory_format)
            )
            # 匹配计数初始化为二进制列表中指定二进制函数的数量加2
            match_count = binary_list[binary_fn][0] + 2
            # 匹配节点数初始化为二进制列表中指定二进制函数的节点数
            match_nodes = binary_list[binary_fn][1]
            # 如果has_relu为True，增加一个ReLU激活节点
            if has_relu:
                match_nodes += 1
            # 调用通用测试函数，验证模型输出与预期匹配计数和节点数
            self._test_common(
                mod, (v,), match_count, match_nodes + 2, check_autocast=dtype
            )
            # 计算卷积生成的核数量
            generated_kernel_count = cal_conv_generated_kernel_number(mod, v, dtype)
            # 断言生成的核数量与度量指标中的生成核数量相等
            self.assertEqual(metrics.generated_kernel_count, generated_kernel_count)

    # 跳过无Dynamo支持的测试
    @skipIfNoDynamoSupport
    # 跳过无ONEDNN支持的测试
    @skipIfNoONEDNN
    # 跳过在Rocm环境下的测试
    @skipIfRocm
    # 测试二维二进制卷积
    def test_conv2d_binary(self):
        self._test_conv_binary_base(dim=4)

    # 跳过无Dynamo支持的测试
    @skipIfNoDynamoSupport
    # 跳过无ONEDNN支持的测试
    @skipIfNoONEDNN
    # 跳过在Rocm环境下的测试
    @skipIfRocm
    # 测试三维二进制卷积
    def test_conv3d_binary(self):
        self._test_conv_binary_base(dim=5)
    def test_linear_binary(self):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化方法，接受二进制函数、输入通道数、输出通道数、是否带偏置等参数
            def __init__(self, binary_fn, in_channels, out_channels, bias, **kwargs):
                super().__init__()
                # 创建一个线性层，输入通道数、输出通道数，并设置是否带偏置
                self.linear = torch.nn.Linear(
                    in_channels, out_channels, bias=bias, **kwargs
                )
                # 保存传入的二进制函数
                self.binary_fn = binary_fn

            # 前向传播方法，接受输入张量 x 和 y，对 x 进行线性变换和二进制函数处理
            def forward(self, x, y):
                x = self.linear(x)  # 线性变换
                x = self.binary_fn(x, y.clone())  # 应用二进制函数
                return x

        dtypes = []
        # 如果支持 MKLDNN 的 bfloat16 类型，添加到 dtypes 列表中
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        # 如果支持 MKLDNN 的 float16 类型，添加到 dtypes 列表中
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        # 生成 options 列表，其中包含所有组合的二进制函数、输入形状、是否带偏置、数据类型
        options = itertools.product(
            binary_list, [[2, 3, 10], [2, 10]], [True, False], dtypes
        )
        out_feature = 30
        # 遍历 options 列表中的每个组合
        for binary_fn, input_shape, bias, dtype in options:
            metrics.reset()
            # 初始化匹配计数和匹配节点数
            match_count = 2
            match_nodes = 3
            # 如果输入形状长度为 3
            if len(input_shape) == 3:
                is_inplace = binary_list[binary_fn][2]
                # 如果二进制函数支持原地操作，更新匹配计数和匹配节点数
                match_count = match_count + 5 if is_inplace else match_count + 3
                match_nodes = match_nodes + 7 if is_inplace else match_nodes + 5
            # 创建 M 类的实例 mod，传入二进制函数、输入通道数、输出特征数、是否带偏置
            mod = M(binary_fn, input_shape[-1], out_feature, bias).eval()
            # 创建随机输入张量 v
            v = torch.randn(input_shape)
            # 创建随机张量 other，与 v 形状相同，最后一个维度是输出特征数，并转换为指定的数据类型 dtype
            other = torch.randn(input_shape[:-1] + [out_feature]).to(dtype)
            # 调用 _test_common 方法进行模型测试，传入模型 mod、输入数据、匹配计数、匹配节点数和数据类型检查选项
            self._test_common(
                mod,
                (
                    v,
                    other,
                ),
                match_count,
                match_nodes,
                check_autocast=dtype,
            )
            # 断言生成的内核数为 1
            self.assertEqual(metrics.generated_kernel_count, 1)
    def test_multi_linear_share_same_input(self):
        # 定义一个名为M的内部类，继承自torch.nn.Module
        class M(torch.nn.Module):
            # 类构造函数，初始化模型结构
            def __init__(
                self,
            ):
                super().__init__()
                # 第一个线性层，输入大小16，输出大小16，无偏置
                self.w1 = torch.nn.Linear(16, 16, bias=False)
                # 第二个线性层，输入大小16，输出大小16，无偏置
                self.w2 = torch.nn.Linear(16, 16, bias=False)

            # 前向传播函数
            def forward(self, x):
                # 对输入x先应用w1，然后silu激活函数，再对结果应用w2和relu激活函数
                return F.silu(self.w1(x)) * F.relu(self.w2(x))

        dtypes = []
        # 如果支持MKLDNN的BF16格式，则添加torch.bfloat16到dtypes列表
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        # 如果支持MKLDNN的FP16格式，则添加torch.float16到dtypes列表
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        # 遍历dtypes列表中的每种数据类型
        for dtype in dtypes:
            # 创建模型M的实例，并将其转移到指定的dtype上，并设置为评估模式
            mod = M().to(dtype).eval()
            # 生成一个形状为(2, 4, 16)的随机张量v，并将其转移到指定的dtype上
            v = torch.randn(2, 4, 16).to(dtype)
            # 测试函数_test_common，验证模型的匹配次数和节点数，并设置容差为1e-2
            match_count = 10
            match_nodes = 19
            self._test_common(mod, (v,), match_count, match_nodes, rtol=1e-2, atol=1e-2)

    def _qconv2d_cpu_test_helper(self, int8_mixed_bf16=False):
        # 定义一个名为M的内部类，继承自torch.nn.Module
        class M(torch.nn.Module):
            # 类构造函数，初始化模型结构
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                # 第一个卷积层，输入通道数3，输出通道数128，核大小3x3，步长1
                self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)
                # 第二个卷积层，输入通道数128，输出通道数128，核大小3x3，步长1
                self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1)

            # 前向传播函数
            def forward(self, x):
                # 先应用第一个卷积层conv，再应用第二个卷积层conv2
                return self.conv2(self.conv(x))

        mod = M().eval()
        # 生成一个形状为(1, 3, 8, 8)的随机浮点数张量v，并加1
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        def matcher_check_fn():
            # 匹配检查函数，用于验证量化Conv2d模块的匹配次数和节点数
            # 当int8_mixed_bf16为True时，节点数为12；否则为8
            # 期望匹配两次qconv2d_weight_prepack_matcher_count
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 2
            )
            # 检查节点数是否符合预期
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_nodes"],
                12 if int8_mixed_bf16 else 8,
            )

        # 调用测试函数_test_common，验证模型的量化，自动类型转换等功能
        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
            matcher_check_fn=matcher_check_fn,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_cpu(self):
        r"""
        This testcase will quantize a single Conv2d module.
        """
        # 调用_qconv2d_cpu_test_helper函数，测试单个Conv2d模块的量化
        self._qconv2d_cpu_test_helper()

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    # 装饰器，如果没有ONEDNN支持则跳过测试
    @skipIfNoONEDNN
    # 定义一个测试方法，用于测试使用int8_mixed_bf16量化的Conv2d模块
    def test_qconv2d_int8_mixed_bf16(self):
        r"""
        This testcase will quantize a single Conv2d module with int8_mixed_bf16 quantization.
        """
        # 调用_qconv2d_cpu_test_helper方法进行测试
        self._qconv2d_cpu_test_helper(int8_mixed_bf16=True)

    # 定义一个辅助方法，用于测试使用unary操作的CPU量化Conv2d
    def _qconv2d_unary_cpu_test_helper(
        self,
        int8_mixed_bf16=False,
        unary_op=torch.nn.ReLU(),
        qconv2d_unary_matcher_nodes=None,
    ):
        # 内部定义一个简单的Module类
        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                # 第一个卷积层，输入通道3，输出通道128，卷积核大小3x3，步长1
                self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)
                # 复制并保存传入的unary操作函数
                self.unary_fn = copy.deepcopy(unary_op)
                # 第二个卷积层，输入通道128，输出通道128，卷积核大小3x3，步长1
                self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1)
                # 复制并保存传入的unary操作函数
                self.unary_fn2 = copy.deepcopy(unary_op)

            # 前向传播方法
            def forward(self, x):
                # 对输入x进行第一次卷积后应用unary_fn
                tmp = self.unary_fn(self.conv(x))
                # 对第一次卷积结果tmp进行第二次卷积后应用unary_fn2
                return self.unary_fn2(self.conv2(tmp))

        # 创建M类的实例mod，并设置为评估模式
        mod = M().eval()
        # 创建一个形状为(1, 3, 8, 8)的随机张量v，数据类型为float32，不需要梯度
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        # 定义一个匹配检查函数matcher_check_fn
        def matcher_check_fn():
            # 检查量化权重预打包中Dequant-Conv2D模式匹配的次数是否为2
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 2
            )
            # 检查在后梯度融合过程中QConv2D Unary融合的次数是否为2
            self.assertEqual(counters["inductor"]["qconv2d_unary_matcher_count"], 2)
            # 如果提供了qconv2d_unary_matcher_nodes参数，则检查节点数量是否匹配
            if qconv2d_unary_matcher_nodes:
                self.assertEqual(
                    counters["inductor"]["qconv2d_unary_matcher_nodes"],
                    qconv2d_unary_matcher_nodes,
                )

        # 调用共同测试方法_test_common，传入模型mod、输入张量v，进行量化检查和自动转换检查
        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
            matcher_check_fn=matcher_check_fn,
        )

    # 装饰器，如果没有Dynamo支持则跳过测试
    @skipIfNoDynamoSupport
    # 装饰器，如果没有ONEDNN支持则跳过测试
    @skipIfNoONEDNN
    # 定义一个测试方法，用于测试Conv2d->ReLU模式的量化
    def test_qconv2d_relu_cpu(self):
        r"""
        This testcase will quantize Conv2d->ReLU pattern.
        """
        # 调用_qconv2d_unary_cpu_test_helper方法进行测试

        self._qconv2d_unary_cpu_test_helper()

    # 装饰器，如果没有Dynamo支持则跳过测试
    @skipIfNoDynamoSupport
    # 装饰器，如果没有ONEDNNBF16支持则跳过测试
    @skipIfNoONEDNNBF16
    # 装饰器，如果没有ONEDNN支持则跳过测试
    @skipIfNoONEDNN
    # 定义一个测试方法，用于测试Conv2d->ReLU模式的int8_mixed_bf16量化
    def test_qconv2d_relu_int8_mixed_bf16(self):
        r"""
        This testcase will quantize Conv2d->ReLU pattern with int8_mixed_bf16 quantization.
        """
        # 调用_qconv2d_unary_cpu_test_helper方法进行测试，传入int8_mixed_bf16=True
        self._qconv2d_unary_cpu_test_helper(int8_mixed_bf16=True)

    # 装饰器，如果没有Dynamo支持则跳过测试
    @skipIfNoDynamoSupport
    # 装饰器，如果没有ONEDNN支持则跳过测试
    @skipIfNoONEDNN
    # 定义一个测试方法，用于测试Conv2d->ReLU6模式的量化
    def test_qconv2d_relu6_cpu(self):
        r"""
        This testcase will quantize Conv2d->ReLU6 pattern.
        """
        # 调用_qconv2d_unary_cpu_test_helper方法进行测试，传入unary_op=torch.nn.ReLU6()
        self._qconv2d_unary_cpu_test_helper(unary_op=torch.nn.ReLU6())

    # 装饰器，如果没有Dynamo支持则跳过测试
    @skipIfNoDynamoSupport
    # 装饰器，如果没有ONEDNN支持则跳过测试
    # 定义一个测试方法，用于测试Conv2d->Hardtanh模式的量化
    def test_qconv2d_hardtanh_cpu(self):
        r"""
        This testcase will quantize Conv2d->Hardtanh pattern.
        """
        # 调用_qconv2d_unary_cpu_test_helper方法进行测试，传入unary_op=torch.nn.Hardtanh()
        self._qconv2d_unary_cpu_test_helper(unary_op=torch.nn.Hardtanh())
    # 根据装饰器 `skipIfNoONEDNN` 条件跳过测试，仅当 ONEDNN 可用时执行
    def test_qconv2d_hardtanh_int8_mixed_bf16_cpu(self):
        r"""
        This testcase will quantize Conv2d->Hardtanh pattern.
        Match.nodes:
            [qconv2d_pointwise_default, convert_element_type, clamp_min, clamp_max, convert_element_type, quantize_per_tensor]
            [qconv2d_pointwise_default, convert_element_type, clamp_min, clamp_max, convert_element_type]
        """
        # 使用辅助函数测试 CPU 端的量化 Conv2d->Hardtanh 模式
        self._qconv2d_unary_cpu_test_helper(
            unary_op=torch.nn.Hardtanh(),
            int8_mixed_bf16=True,  # 启用混合整数 8 位和 BF16 数据类型
            qconv2d_unary_matcher_nodes=11,  # 期望匹配的节点数为 11
        )
    
    # 根据装饰器 `skipIfNoDynamoSupport` 和 `skipIfNoONEDNN` 条件跳过测试，仅当 Dynamo 支持和 ONEDNN 可用时执行
    def test_qconv2d_hardswish_cpu(self):
        r"""
        This testcase will quantize Conv2d->Hardswish pattern.
        """
        # 使用辅助函数测试 CPU 端的量化 Conv2d->Hardswish 模式
        self._qconv2d_unary_cpu_test_helper(unary_op=torch.nn.Hardswish())
    
    # 根据装饰器 `skipIfNoDynamoSupport`、`skipIfNoONEDNNBF16` 和 `skipIfNoONEDNN` 条件跳过测试，仅当 Dynamo 支持、ONEDNN 和 ONEDNN BF16 可用时执行
    def test_qconv2d_hardswish_int8_mixed_bf16_cpu(self):
        r"""
        This testcase will quantize Conv2d->Hardswish pattern.
        Match.nodes:
            [qconv2d_pointwise_default, convert_element_type, add, clamp_min,
             clamp_max, mul, div, convert_element_type, quantize_per_tensor]
            [qconv2d_pointwise_default, convert_element_type, add, clamp_min, clamp_max, mul, div, convert_element_type]
        """
        # 使用辅助函数测试 CPU 端的量化 Conv2d->Hardswish 模式，启用混合整数 8 位和 BF16 数据类型
        self._qconv2d_unary_cpu_test_helper(
            unary_op=torch.nn.Hardswish(),
            int8_mixed_bf16=True,  # 启用混合整数 8 位和 BF16 数据类型
            qconv2d_unary_matcher_nodes=17,  # 期望匹配的节点数为 17
        )
    
    # 根据装饰器 `skipIfNoDynamoSupport` 和 `skipIfNoONEDNN` 条件跳过测试，仅当 Dynamo 支持和 ONEDNN 可用时执行
    def test_qconv2d_silu_cpu(self):
        r"""
        This testcase will quantize Conv2d->SiLU pattern.
        """
        # 使用辅助函数测试 CPU 端的量化 Conv2d->SiLU 模式
        self._qconv2d_unary_cpu_test_helper(unary_op=torch.nn.SiLU())
    
    # 根据装饰器 `skipIfNoDynamoSupport`、`skipIfNoONEDNNBF16` 和 `skipIfNoONEDNN` 条件跳过测试，仅当 Dynamo 支持、ONEDNN 和 ONEDNN BF16 可用时执行
    def test_qconv2d_silu_int8_mixed_bf16_cpu(self):
        r"""
        This testcase will quantize Conv2d->SiLU pattern.
        Match.nodes:
            [qconv2d_pointwise_default, convert_element_type, sigmoid, mul,
             convert_element_type, quantize_per_tensor]
            [qconv2d_pointwise_default, convert_element_type, sigmoid, mul, convert_element_type]
        """
        # 使用辅助函数测试 CPU 端的量化 Conv2d->SiLU 模式，启用混合整数 8 位和 BF16 数据类型
        self._qconv2d_unary_cpu_test_helper(
            unary_op=torch.nn.SiLU(),
            int8_mixed_bf16=True,  # 启用混合整数 8 位和 BF16 数据类型
            qconv2d_unary_matcher_nodes=11,  # 期望匹配的节点数为 11
        )
    def _qconv2d_add_cpu_test_helper(self, use_relu=False, int8_mixed_bf16=False):
        r"""
        This testcase will quantize a Conv2d->Add pattern as:
                 X
               /   \
        Conv1(X)   Conv2(X)
               \   /
                Add
                 |
           Optional(relu)
                 |
                 Y
        """

        # 定义内部模块M，用于测试量化Conv2d->Add模式
        class M(torch.nn.Module):
            def __init__(
                self,
                add_fn,
                use_relu,
                **kwargs,
            ):
                super().__init__()
                # 定义两个Conv2d层
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.add_fn = add_fn
                self.relu = torch.nn.ReLU()
                # 定义两个Conv2d层和一个Add函数
                self.conv3 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv4 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.add_fn2 = add_fn
                self.relu2 = torch.nn.ReLU()
                self.use_relu = use_relu

            # 定义前向传播函数
            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                tmp = self.add_fn(x1, x2)  # 执行Add操作
                if self.use_relu:
                    tmp = self.relu(tmp)  # 如果use_relu为True，则执行ReLU激活函数
                tmp1 = self.conv3(tmp)
                tmp2 = self.conv4(tmp)
                res = self.add_fn2(tmp1, tmp2)  # 执行第二次Add操作
                if self.use_relu:
                    res = self.relu2(res)  # 如果use_relu为True，则执行第二次ReLU激活函数
                return res

        # 对于quantization_add_fn_list和quantization_inplace_add_fn_list中的每一个函数add_fn，执行以下测试
        for add_fn in quantization_add_fn_list + quantization_inplace_add_fn_list:
            mod = M(add_fn, use_relu).eval()  # 创建模块M的实例mod，并设置为评估模式
            v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
                1
            )

            # 定义matcher_check_fn函数，用于检查量化匹配器的结果
            def matcher_check_fn():
                # 1. Dequant-Conv2D模式在量化权重预打包中匹配了4次
                self.assertEqual(
                    counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 4
                )
                # 2. 在后处理融合过程中，Qconv2d二元一元融合匹配了2次
                self.assertEqual(
                    counters["inductor"]["qconv2d_binary_matcher_count"], 2
                )

            # 调用测试公共方法_test_common，传入模块mod、输入数据v等参数进行测试
            self._test_common(
                mod,
                (v,),
                check_quantization=True,
                check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
                matcher_check_fn=matcher_check_fn,
            )

    # 使用skipIfNoDynamoSupport修饰，如果不支持Dynamo，则跳过该测试
    @skipIfNoDynamoSupport
    # 使用skipIfNoONEDNN修饰，如果不支持ONEDNN，则跳过该测试
    @skipIfNoONEDNN
    # 定义测试方法test_qconv2d_add_cpu，测试量化Conv2d->Add模式在CPU上的运行
    def test_qconv2d_add_cpu(self):
        self._qconv2d_add_cpu_test_helper()

    # 使用skipIfNoDynamoSupport修饰，如果不支持Dynamo，则跳过该测试
    @skipIfNoDynamoSupport
    # 使用skipIfNoONEDNNBF16修饰，如果不支持ONEDNN BF16，则跳过该测试
    @skipIfNoONEDNNBF16
    # 使用skipIfNoONEDNN修饰，如果不支持ONEDNN，则跳过该测试
    @skipIfNoONEDNN
    # 定义测试方法test_qconv2d_add_int8_mixed_bf16，测试量化Conv2d->Add模式在int8和混合BF16下的运行
    def test_qconv2d_add_int8_mixed_bf16(self):
        self._qconv2d_add_cpu_test_helper(int8_mixed_bf16=True)

    # 使用skipIfNoDynamoSupport修饰，如果不支持Dynamo，则跳过该测试
    @skipIfNoDynamoSupport
    # 使用skipIfNoONEDNN修饰，如果不支持ONEDNN，则跳过该测试
    @skipIfNoONEDNN
    # 定义测试方法test_qconv2d_add_relu_cpu，测试量化Conv2d->Add模式在CPU上使用ReLU的运行
    def test_qconv2d_add_relu_cpu(self):
        self._qconv2d_add_cpu_test_helper(use_relu=True)

    # 使用skipIfNoDynamoSupport修饰
    # 根据条件装饰器 `@skipIfNoONEDNNBF16` 和 `@skipIfNoONEDNN`，跳过没有ONEDNNBF16支持和ONEDNN支持的情况
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qconv2d_add_relu_int8_mixed_bf16(self):
        # 调用测试辅助函数 `_qconv2d_add_cpu_test_helper`，测试QConv2D加ReLU整数混合BF16操作
        self._qconv2d_add_cpu_test_helper(use_relu=True, int8_mixed_bf16=True)

    # 根据条件装饰器 `@skipIfNoDynamoSupport` 和 `@skipIfNoONEDNN`，跳过没有Dynamo支持和ONEDNN支持的情况
    def test_qconv2d_add_broadcast_shapes_cpu(self):
        r"""
        This testcase will quantize Conv2d->add pattern using broadcast shape inputs.
        Conv2d->Add fusion will fail for the broadcast shape inputs case.
        """

        # 定义一个模块 M，用于测试量化 Conv2d->add 模式，使用广播形状的输入
        class M(torch.nn.Module):
            def __init__(self, use_bias):
                super().__init__()
                self.conv = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1)

            def forward(self, x1, x2):
                # 在前向传播中，对 conv(x1) 的结果和 x2 执行 torch.add 操作
                return torch.add(self.conv(x1), x2)

        # bias_list 包含两种情况：True 和 False
        bias_list = [True, False]
        for bias in bias_list:
            # 创建模块实例 mod，并将其设置为评估模式
            mod = M(bias).eval()
            # 随机生成输入张量 x1 和 x2，分别为 (2, 32, 9, 9) 和 (2, 32, 1, 1)
            x1 = torch.randn((2, 32, 9, 9))
            x2 = torch.randn((2, 32, 1, 1))

            # 定义匹配器检查函数 matcher_check_fn
            def matcher_check_fn():
                # 1. 在量化权重预打包中匹配到 Dequant-Conv2D 模式，预期次数为 1
                self.assertEqual(
                    counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 1
                )
                # 2. 在后向融合过程中未匹配到 Qconv2d 二元一元融合，预期次数为 0
                self.assertEqual(
                    counters["inductor"]["qconv2d_binary_matcher_count"], 0
                )

            # 调用通用测试函数 `_test_common`，测试模块 mod 的输出，检查量化，同时执行匹配器检查函数 matcher_check_fn
            self._test_common(
                mod,
                (x1, x2),
                check_quantization=True,
                matcher_check_fn=matcher_check_fn,
            )

    # 根据条件装饰器 `@skipIfNoDynamoSupport` 和 `@skipIfNoONEDNN`，跳过没有Dynamo支持和ONEDNN支持的情况
    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_add_2(self):
        r"""
        This testcase prevents this pattern be matched as a conv_binary fusion by mistake.
                Conv(X)  3
                    \   /
                     Add
        We see this pattern in Mobilenet v3 large which add is decomposed from torch.nn.Hardswish or torch.nn.Hardsigmoid.
        """
        # 定义一个测试用例，用于防止将此模式误识别为卷积二进制融合。
        # 在Mobilenet v3 large中我们看到这种模式，其中Add是从torch.nn.Hardswish或torch.nn.Hardsigmoid中解构出来的。

        class M(torch.nn.Module):
            def __init__(
                self,
                post_op,
            ):
                super().__init__()
                # 定义一个2D卷积层，输入通道为3，输出通道为6，卷积核大小为3x3，步长为1
                self.conv = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.post_op = post_op  # 保存后续操作对象

            def forward(self, x):
                # 执行卷积操作后，再施加后续操作
                return self.post_op(self.conv(x))

        for post_op in [
            torch.nn.Hardswish(inplace=True),  # 使用inplace方式应用Hardswish激活函数
            torch.nn.Hardsigmoid(inplace=True),  # 使用inplace方式应用Hardsigmoid激活函数
        ]:
            mod = M(post_op).eval()  # 创建模型实例并设置为评估模式
            v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
                1
            )
            # 创建一个形状为(1, 3, 8, 8)的张量，填充随机数，并加1

            def matcher_check_fn():
                # 检查量化时不应命中卷积二进制融合
                self.assertEqual(
                    counters["inductor"]["qconv2d_binary_matcher_count"], 0
                )

            self._test_common(
                mod,
                (v,),
                check_quantization=True,
                matcher_check_fn=matcher_check_fn,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_add_3(self):
        r"""
        This testcase will test below model:
             x
           /   \
        conv1  maxpool
          \    /   \
           add    conv2
            \     /
              cat
        Based on default recipe of x86InductorQuantizer, we will see this pattern after convert:
        qconv1    maxpool
         \           |
          \         q1
           \       /   \
            \     dq1  qconv2
             \   /
              add
               |
               q2
        Since q1 has 2 users and qconv2 is not ancestor node of qconv1, we shouldn't fuse:
                int8
                 /
        qconv1 dq1
           \   /
            add
             |
             q2
             |
            int8
        Instead we can match and fuse this pattern into qconv_binary:
        qconv1  fp32
            \   /
             add
              |
             fp32
        """

        # 定义一个包含多层卷积和池化的神经网络模型
        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                # 第一层卷积层，输入通道数为3，输出通道数为3，卷积核大小为3x3，步长为1
                self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1)
                # 第二层卷积层，输入通道数为3，输出通道数为3，卷积核大小为1x1，步长为1
                self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
                # 最大池化层，池化核大小为3x3，步长为1，填充为0，膨胀率为1
                self.maxpool = torch.nn.MaxPool2d(
                    kernel_size=3, stride=1, padding=0, dilation=1
                )

            # 前向传播函数，接受输入x，返回模型输出
            def forward(self, x):
                # 经过第一层卷积
                tmp1 = self.conv1(x)
                # 经过最大池化
                tmp2 = self.maxpool(x)
                # 将第一层卷积和最大池化的输出相加
                add = torch.add(tmp1, tmp2)
                # 经过第二层卷积
                tmp3 = self.conv2(tmp2)
                # 沿着通道维度拼接add和第二层卷积的输出
                return torch.cat((add, tmp3), dim=1)

        # 创建模型实例并设为评估模式
        mod = M().eval()
        # 创建随机张量作为输入数据，形状为(1, 3, 8, 8)，数据类型为float32，不需要梯度，且每个元素加1
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        # 定义一个匹配器检查函数，用于检查量化后的模型是否符合预期
        def matcher_check_fn():
            # 断言量化后的模型中的qconv2d_binary匹配计数为1
            self.assertEqual(counters["inductor"]["qconv2d_binary_matcher_count"], 1)
            # 断言匹配的qconv_binary模式节点数为2，即包括qconv和add
            self.assertEqual(counters["inductor"]["qconv2d_binary_matcher_nodes"], 2)

        # 调用测试通用函数，传入模型、输入数据和匹配器检查函数
        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            matcher_check_fn=matcher_check_fn,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_qconv2d(self):
        r"""
        This testcase will quantize a single Conv2d module with qat flow.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                # 定义一个包含3个输入通道、128个输出通道的3x3卷积层
                self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)
                # 定义一个包含128个通道的批归一化层
                self.bn = torch.nn.BatchNorm2d(128)

            def forward(self, x):
                # 在前向传播中，先进行卷积，然后对输出进行批归一化
                return self.bn(self.conv(x))

        # 创建一个M类的实例，并设置为训练模式
        mod = M().train()
        # 创建一个形状为(1, 3, 8, 8)的随机张量，数据类型为float32，需要计算梯度，并加1
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            # 1. Dequant-conv pattern matched in quantization weight prepack * 1
            #    匹配量化权重预打包中的解量化-卷积模式 * 1
            #    [dequantize_per_tensor, dequantize_per_channel, clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 1
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_nodes"], 4
            )
            # 2. QConv2D Unary fusion in post-grad fusion pass * 1
            #    后向量化融合过程中的QConv2D一元融合 * 1
            #    [qconv2d_pointwise_default, quantize_per_tensor]
            self.assertEqual(counters["inductor"]["qconv2d_unary_matcher_count"], 1)
            self.assertEqual(counters["inductor"]["qconv2d_unary_matcher_nodes"], 2)

        # 执行通用测试函数，传入模型实例、输入数据、检查量化标志为真、使用QAT流程、匹配检查函数
        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            is_qat=True,
            matcher_check_fn=matcher_check_fn,
        )

    def _qat_qconv2d_unary_cpu_test_helper(
        self,
        unary_op=torch.nn.ReLU(),
    ):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # M 类的初始化方法
            def __init__(
                self,
                **kwargs,
            ):
                # 调用父类的初始化方法
                super().__init__()
                # 定义一个 3x3 的二维卷积层，输入和输出通道数均为 3
                self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1)
                # 深拷贝传入的 unary_op 对象，并赋值给实例变量 unary_fn
                self.unary_fn = copy.deepcopy(unary_op)
                # 定义一个 3 维度的批标准化层
                self.bn = torch.nn.BatchNorm2d(3)
                # 定义第二个 3x3 的二维卷积层，输入和输出通道数均为 3
                self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1)
                # 再次深拷贝传入的 unary_op 对象，并赋值给实例变量 unary_fn2
                self.unary_fn2 = copy.deepcopy(unary_op)
                # 定义第二个 3 维度的批标准化层
                self.bn2 = torch.nn.BatchNorm2d(3)

            # M 类的前向传播方法
            def forward(self, x):
                # 对输入 x 进行卷积、批标准化、和 unary_fn 函数操作，得到 tmp
                tmp = self.unary_fn(self.bn(self.conv(x)))
                # 对 tmp 再进行卷积、批标准化、和 unary_fn2 函数操作，作为输出
                return self.unary_fn2(self.bn2(self.conv2(tmp)))

        # 创建 M 类的实例 mod
        mod = M()
        # 生成一个形状为 (1, 3, 8, 8) 的随机张量 v，数据类型为 float32，且要求计算梯度
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        # 定义 matcher_check_fn 函数，用于检查量化和模式匹配相关计数器的值
        def matcher_check_fn():
            # 检查量化权重预打包匹配器计数器中的值是否为 2
            # 匹配到的模式：[convert_element_type_1, sub, mul_1, dequantize_per_channel, clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 2
            )
            # 检查量化卷积+一元操作融合匹配器计数器中的值是否为 2
            # 匹配到的模式：[qconv2d_pointwise_default, relu, div_1, round_2, add_1, clamp_min_1, clamp_max_1, convert_element_type_2]
            self.assertEqual(counters["inductor"]["qconv2d_unary_matcher_count"], 2)

        # 调用外部定义的 _test_common 方法进行通用测试
        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            is_qat=True,
            matcher_check_fn=matcher_check_fn,
        )

    # 使用装饰器跳过没有 Dynamo 支持的测试
    @skipIfNoDynamoSupport
    # 使用装饰器跳过没有 ONEDNN 支持的测试
    @skipIfNoONEDNN
    # 定义一个测试方法，测试量化训练时 Conv2d->ReLU 模式
    def test_qat_qconv2d_relu(self):
        r"""
        This testcase will quantize Conv2d->ReLU pattern with qat flow.
        """
        # 调用 _qat_qconv2d_unary_cpu_test_helper 方法辅助测试
        self._qat_qconv2d_unary_cpu_test_helper()

    # 使用装饰器跳过没有 Dynamo 支持的测试
    @skipIfNoDynamoSupport
    # 使用装饰器跳过没有 ONEDNN 支持的测试
    @skipIfNoONEDNN
    # 定义一个测试方法，测试量化训练时 Conv2d->ReLU6 模式
    def test_qat_qconv2d_relu6(self):
        r"""
        This testcase will quantize Conv2d->ReLU6 pattern with qat flow.
        """
        # 调用 _qat_qconv2d_unary_cpu_test_helper 方法辅助测试，并传入 ReLU6 作为 unary_op
        self._qat_qconv2d_unary_cpu_test_helper(unary_op=torch.nn.ReLU6())

    # 使用装饰器跳过没有 Dynamo 支持的测试
    @skipIfNoDynamoSupport
    # 使用装饰器跳过没有 ONEDNN 支持的测试
    # 定义一个测试方法，测试量化训练时 Conv2d->Hardtanh 模式
    def test_qat_qconv2d_hardtanh(self):
        r"""
        This testcase will quantize Conv2d->Hardtanh pattern with qat flow.
        """
        # 调用 _qat_qconv2d_unary_cpu_test_helper 方法辅助测试，并传入 Hardtanh 作为 unary_op
        self._qat_qconv2d_unary_cpu_test_helper(unary_op=torch.nn.Hardtanh())

    # 使用装饰器跳过没有 Dynamo 支持的测试
    @skipIfNoDynamoSupport
    # 使用装饰器跳过没有 ONEDNN 支持的测试
    # 定义一个测试方法，测试量化训练时 Conv2d->SiLU 模式
    def test_qat_qconv2d_silu(self):
        r"""
        This testcase will quantize Conv2d->SiLU pattern with qat flow.
        """
        # 调用 _qat_qconv2d_unary_cpu_test_helper 方法辅助测试，并传入 SiLU 作为 unary_op
        self._qat_qconv2d_unary_cpu_test_helper(unary_op=torch.nn.SiLU())

    # 使用装饰器跳过没有 Dynamo 支持的测试
    @skipIfNoDynamoSupport
    # 使用装饰器跳过没有 ONEDNN 支持的测试
    # 定义一个测试方法，测试量化训练时 Conv2d->Hardswish 模式
    def test_qat_qconv2d_hardswish(self):
        r"""
        This testcase will quantize Conv2d->Hardswish pattern with qat flow.
        """
        # 调用 _qat_qconv2d_unary_cpu_test_helper 方法辅助测试，并传入 Hardswish 作为 unary_op
        self._qat_qconv2d_unary_cpu_test_helper(unary_op=torch.nn.Hardswish())

    # 使用装饰器跳过没有 Dynamo 支持的测试
    @skipIfNoDynamoSupport
    # 使用装饰器跳过没有 ONEDNN 支持的测试
    # 使用装饰器跳过在 ROCm 平台上的测试
    @skipIfRocm
    def test_qat_qconv2d_add(self):
        r"""
        This testcase will quantize a Conv2d->Add pattern as:
                 X
               /   \
        Conv1(X)   Conv2(X)
               \   /
                Add
                 |
                 Y
        """

        # 定义一个名为 M 的内部类，继承自 torch.nn.Module
        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                # 第一个卷积层：输入通道数为 3，输出通道数为 6，卷积核大小为 3x3，步长为 1
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                # 第一个批归一化层，输入通道数为 6
                self.bn1 = torch.nn.BatchNorm2d(6)
                # 第二个卷积层：输入通道数为 3，输出通道数为 6，卷积核大小为 3x3，步长为 1
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                # 第二个批归一化层，输入通道数为 6
                self.bn2 = torch.nn.BatchNorm2d(6)

            # 前向传播函数，接收输入 x，进行卷积、批归一化和相加操作
            def forward(self, x):
                # 对输入 x 分别进行 conv1 -> bn1 和 conv2 -> bn2 的操作，并相加
                x1 = self.bn1(self.conv1(x))
                x2 = self.bn2(self.conv2(x))
                return x1 + x2

        # 创建 M 类的实例 mod，并设置为训练模式
        mod = M().train()
        # 生成一个形状为 (1, 3, 8, 8) 的随机张量 v，数据类型为 float32，且需要梯度计算，然后加 1
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        # 定义 matcher_check_fn 函数，用于检查量化相关模式匹配的情况
        def matcher_check_fn():
            # 检查量化卷积权重预打包匹配器的计数和节点数
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 2
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_nodes"], 8
            )
            # 检查量化卷积二进制融合匹配器的计数和节点数
            self.assertEqual(counters["inductor"]["qconv2d_binary_matcher_count"], 1)
            self.assertEqual(counters["inductor"]["qconv2d_binary_matcher_nodes"], 4)

        # 执行通用的测试方法 _test_common，传入模型 mod、随机张量 v，以及其他相关参数
        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            is_qat=True,
            matcher_check_fn=matcher_check_fn,
        )

    # 跳过没有 Dynamo 支持、没有 ONEDNN 支持以及在 Rocm 环境下的测试
    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_qconv2d_add_relu(self):
        r"""
        This testcase will quantize a Conv2d->Add->ReLU pattern as:
                 X
               /   \
        Conv1(X)   Conv2(X)
               \   /
                Add
                 |
                ReLU
                 |
                 Y
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                # 定义第一个卷积层，输入通道数为3，输出通道数为6，卷积核大小为3x3，步长为1
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.bn1 = torch.nn.BatchNorm2d(6)  # 第一个卷积层的批归一化层
                # 定义第二个卷积层，输入通道数为3，输出通道数为6，卷积核大小为3x3，步长为1
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.bn2 = torch.nn.BatchNorm2d(6)  # 第二个卷积层的批归一化层
                self.relu = torch.nn.ReLU()  # ReLU 激活函数

            def forward(self, x):
                # 对输入 x 分别进行第一层卷积、批归一化和 ReLU 激活
                x1 = self.bn1(self.conv1(x))
                # 对输入 x 分别进行第二层卷积、批归一化和 ReLU 激活
                x2 = self.bn2(self.conv2(x))
                # 将两个路径的输出相加，并经过 ReLU 激活
                return self.relu(x1 + x2)

        mod = M().train()  # 创建并训练模型 M
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)  # 创建输入张量 v，加上偏置1

        def matcher_check_fn():
            # 检查量化后权重预打包匹配器的计数，预期为两次匹配
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 2
            )
            # 检查量化后权重预打包匹配器节点数，预期为8个节点
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_nodes"], 8
            )
            # 检查量化卷积二进制融合在后向量化融合过程中的计数，预期为一次匹配
            self.assertEqual(counters["inductor"]["qconv2d_binary_matcher_count"], 1)
            # 检查量化卷积二进制融合在后向量化融合过程中的节点数，预期为5个节点
            self.assertEqual(counters["inductor"]["qconv2d_binary_matcher_nodes"], 5)

        # 运行通用测试函数，传入模型 mod、输入张量 v，开启量化检查，启用 QAT 模式，并传入匹配器检查函数
        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            is_qat=True,
            matcher_check_fn=matcher_check_fn,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_dequant_promotion_cpu(self):
        r"""
        This testcase tests if dequant node before conv2d is promoted correctly:
                 X
                 |
              Conv1(X)
               /   \
        Conv2(X)   Conv3(X)
               \   /
                Add
                 |
                 Y
        """

        # 定义一个名为 M 的 PyTorch 模型类
        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                # 定义三个卷积层：输入通道数为3，输出通道数为6，卷积核大小为3x3，步长为1
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv3 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)

            def forward(self, x):
                # 对输入 x 应用第一个卷积层
                temp = self.conv1(x)
                # 将第一个卷积层的输出应用于第二个和第三个卷积层，并相加
                temp = self.conv2(temp) + self.conv3(temp)
                return temp

        # 创建 M 类的一个实例 mod，并设为评估模式
        mod = M().eval()
        # 创建一个形状为 (1, 3, 8, 8) 的随机张量 v，数据类型为 torch.float32，不需要梯度，每个元素加 1
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        def matcher_check_fn():
            # 检查量化推断中的模式匹配结果
            # 1. 检查 dequantize_per_tensor 的推广匹配次数为 1
            self.assertEqual(counters["inductor"]["dequant_promotion_matcher_count"], 1)
            # 检查 dequantize_per_tensor 的推广匹配节点数为 1
            self.assertEqual(counters["inductor"]["dequant_promotion_matcher_nodes"], 1)
            # 2. 检查在量化权重预打包中的 Dequant-conv 模式匹配次数为 3
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 3
            )
            # 检查在量化权重预打包中的 Dequant-conv 模式匹配节点数为 12
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_nodes"], 12
            )
            # 3. 检查在后向融合中的 Qconv2d 二进制融合模式匹配次数为 1
            self.assertEqual(counters["inductor"]["qconv2d_binary_matcher_count"], 1)
            # 检查在后向融合中的 Qconv2d 二进制融合模式匹配节点数为 2
            self.assertEqual(counters["inductor"]["qconv2d_binary_matcher_nodes"], 2)

        # 调用 _test_common 方法进行测试
        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            matcher_check_fn=matcher_check_fn,
        )

    def _qlinear_cpu_test_helper(
        self,
        inputs,
        int8_mixed_bf16=False,
        do_permute=False,
        matcher_check_fn=None,
        bias=True,
        is_dynamic=False,
        is_qat=False,
    ):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self, use_bias, do_permute=False):
                super().__init__()
                # 定义一个线性层，输入维度为4，输出维度为3，是否使用偏置由 use_bias 决定
                self.linear = torch.nn.Linear(4, 3, use_bias)
                # 定义第二个线性层，输入维度为3，输出维度为4，是否使用偏置同样由 use_bias 决定
                self.linear2 = torch.nn.Linear(3, 4, use_bias)
                # 是否进行数据维度置换的标志
                self.do_permute = do_permute

            # 前向传播方法
            def forward(self, x):
                # 如果需要进行数据维度置换
                if self.do_permute:
                    # 对输入 x 进行维度置换，维度顺序变为 (0, 2, 3, 1)，然后重新调整形状为 (2, 12, 4)
                    x = torch.reshape(torch.permute(x, (0, 2, 3, 1)), (2, 12, 4))
                # 对置换后的数据进行线性变换
                return self.linear2(self.linear(x))

        # 创建 M 类的实例 mod，并设置为评估模式
        mod = M(bias, do_permute=do_permute).eval()

        # 定义一个内部函数 _default_matcher_check_fn
        def _default_matcher_check_fn():
            # 使用断言检查 counters["inductor"]["qlinear_weight_prepack_matcher_count"] 的值为2
            self.assertEqual(
                counters["inductor"]["qlinear_weight_prepack_matcher_count"], 2
            )

        # 调用 self._test_common 方法，传入模型 mod 和相关参数
        self._test_common(
            mod,
            inputs,
            # 如果 int8_mixed_bf16 为真，则检查自动类型转换为 torch.bfloat16，否则为 torch.float
            check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
            # 检查量化是否开启
            check_quantization=True,
            # 如果有 matcher_check_fn 参数，则使用传入的函数，否则使用默认的 _default_matcher_check_fn
            matcher_check_fn=matcher_check_fn
            if matcher_check_fn is not None
            else _default_matcher_check_fn,
            # 是否进行量化训练
            is_qat=is_qat,
            # 是否动态量化
            is_dynamic=is_dynamic,
        )
    def test_qlinear_input_dim_exceeds_2(self):
        r"""
        This testcase will quantize a single Linear Module.
        """
        # 对是否有偏置进行循环测试
        for bias in [True, False]:
            # 调用_qlinear_cpu_test_helper方法，测试线性模块的量化情况，传入形状为(2, 3, 4)的随机张量作为输入
            self._qlinear_cpu_test_helper((torch.randn((2, 3, 4)),), bias=bias)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_int8_mixed_bf16_input_dim_exceeds_2(self):
        r"""
        This testcase will quantize a single Linear Module with int8_mixed_bf16 quantization.
        """
        # 对是否有偏置进行循环测试
        for bias in [True, False]:
            # 调用_qlinear_cpu_test_helper方法，测试带有int8_mixed_bf16量化的线性模块，传入形状为(2, 3, 4)的随机张量作为输入
            self._qlinear_cpu_test_helper(
                (torch.randn((2, 3, 4)),), int8_mixed_bf16=True, bias=bias
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_input_dim_exceeds_2_and_not_contiguous(self):
        r"""
        This testcase will quantize a single Linear Module.
        * Input dim exceeds 2
        * Input not contiguous
        """
        # 对是否有偏置进行循环测试
        for bias in [True, False]:

            def matcher_check_fn():
                # 检查权重预打包匹配计数是否为2
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 2
                )
                # 检查权重预打包匹配节点数是否为13或12（取决于是否有偏置）
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_nodes"],
                    13 if bias else 12,
                )

            # 调用_qlinear_cpu_test_helper方法，测试线性模块的量化情况，传入形状为(2, 4, 3, 4)的随机张量作为输入，
            # 并进行维度置换(do_permute=True)，同时传入matcher_check_fn作为匹配检查函数
            self._qlinear_cpu_test_helper(
                (torch.randn((2, 4, 3, 4)),),
                do_permute=True,
                matcher_check_fn=matcher_check_fn,
                bias=bias,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_int8_mixed_bf16_input_dim_exceeds_2_and_not_contiguous(self):
        r"""
        This testcase will quantize a single Linear Module for int8_bf16.
        * Input dim exceeds 2
        * Input not contiguous
        """
        # 对是否有偏置进行循环测试
        for bias in [True, False]:

            def matcher_check_fn():
                # 检查权重预打包匹配计数是否为2
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 2
                )
                # 检查权重预打包匹配节点数是否为17或16（取决于是否有偏置）
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_nodes"],
                    17 if bias else 16,
                )

            # 调用_qlinear_cpu_test_helper方法，测试带有int8_mixed_bf16量化的线性模块，传入形状为(2, 4, 3, 4)的随机张量作为输入，
            # 并进行维度置换(do_permute=True)，同时传入matcher_check_fn作为匹配检查函数
            self._qlinear_cpu_test_helper(
                (torch.randn((2, 4, 3, 4)),),
                int8_mixed_bf16=True,
                do_permute=True,
                matcher_check_fn=matcher_check_fn,
                bias=bias,
            )

    def _qlinear_unary_cpu_test_helper(
        self, inputs, unary_op=torch.nn.ReLU(), int8_mixed_bf16=False
    ):
        # 定义一个内部的模型类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化函数，接受一个 use_bias 参数
            def __init__(self, use_bias):
                super().__init__()
                # 定义一个线性层，输入和输出大小均为 4，使用给定的 use_bias
                self.linear = torch.nn.Linear(4, 4, use_bias)
                # 深拷贝给定的 unary_op，并赋值给 self.unary_fn
                self.unary_fn = copy.deepcopy(unary_op)
                # 定义第二个线性层，输入和输出大小均为 4，使用给定的 use_bias
                self.linear2 = torch.nn.Linear(4, 4, use_bias)
                # 再次深拷贝 unary_op，并赋值给 self.unary_fn2
                self.unary_fn2 = copy.deepcopy(unary_op)

            # 前向传播函数，接受输入 x
            def forward(self, x):
                # 对输入 x 应用 self.linear，然后通过 unary_fn 处理得到 tmp
                tmp = self.unary_fn(self.linear(x))
                # 对 tmp 应用 self.linear2，然后再次通过 unary_fn2 处理，最终输出结果
                return self.unary_fn2(self.linear2(tmp))

        # bias_list 包含 True 和 False 两个元素
        bias_list = [True, False]
        # 遍历 bias_list 中的每一个 bias
        for bias in bias_list:
            # 创建一个 M 类的实例 mod，并设置为评估模式
            mod = M(bias).eval()

            # 定义一个匹配器检查函数 matcher_check_fn
            def matcher_check_fn():
                # 断言：在量化权重预打包中匹配到 dequant-linear 模式的次数为 2
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 2
                )
                # 断言：在后向融合过程中，QLinear Unary 被融合的次数为 2
                self.assertEqual(counters["inductor"]["qlinear_unary_matcher_count"], 2)

            # 调用 _test_common 方法进行测试
            self._test_common(
                mod,
                inputs,
                # 如果 int8_mixed_bf16 为真，检查自动类型转换为 torch.bfloat16，否则为 torch.float
                check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
                check_quantization=True,
                matcher_check_fn=matcher_check_fn,
            )

    # 用于支持 Dynamo 的装饰器
    @skipIfNoDynamoSupport
    # 用于支持 ONEDNN 的装饰器
    @skipIfNoONEDNN
    def test_qlinear_relu_cpu(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern.
        """
        # 调用 _qlinear_unary_cpu_test_helper 方法，传入一个大小为 (2, 4) 的随机张量作为输入
        self._qlinear_unary_cpu_test_helper((torch.randn((2, 4)),))

    # 用于支持 Dynamo 的装饰器
    @skipIfNoDynamoSupport
    # 用于支持 ONEDNNBF16 的装饰器
    @skipIfNoONEDNNBF16
    # 用于支持 ONEDNN 的装饰器
    @skipIfNoONEDNN
    def test_qlinear_relu_int8_mixed_bf16(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern with int8_mixed_bf16 quantization.
        """
        # 调用 _qlinear_unary_cpu_test_helper 方法，传入一个大小为 (2, 4) 的随机张量作为输入，并启用 int8_mixed_bf16 模式
        self._qlinear_unary_cpu_test_helper(
            (torch.randn((2, 4)),), int8_mixed_bf16=True
        )

    # 用于支持 Dynamo 的装饰器
    @skipIfNoDynamoSupport
    # 用于支持 ONEDNN 的装饰器
    @skipIfNoONEDNN
    def test_qlinear_relu_input_dim_exceeds_2(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern.
        """
        # 调用 _qlinear_unary_cpu_test_helper 方法，传入一个大小为 (2, 3, 4) 的随机张量作为输入
        self._qlinear_unary_cpu_test_helper((torch.randn((2, 3, 4)),))

    # 用于支持 Dynamo 的装饰器
    @skipIfNoDynamoSupport
    # 用于支持 ONEDNNBF16 的装饰器
    @skipIfNoONEDNNBF16
    # 用于支持 ONEDNN 的装饰器
    @skipIfNoONEDNN
    def test_qlinear_relu_int8_mixed_bf16_input_dim_exceeds_2(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern with int8_mixed_bf16 quantization.
        """
        # 调用 _qlinear_unary_cpu_test_helper 方法，传入一个大小为 (2, 3, 4) 的随机张量作为输入，并启用 int8_mixed_bf16 模式
        self._qlinear_unary_cpu_test_helper(
            (torch.randn((2, 3, 4)),), int8_mixed_bf16=True
        )

    # 用于支持 Dynamo 的装饰器
    @skipIfNoDynamoSupport
    # 用于支持 ONEDNN 的装饰器
    @skipIfNoONEDNN
    def test_qlinear_gelu_cpu(self):
        r"""
        This testcase will quantize a Linear->GELU pattern.
        """
        # 遍历 gelu 列表中的每一个激活函数
        for gelu in [torch.nn.GELU("none"), torch.nn.GELU("tanh")]:
            # 调用 _qlinear_unary_cpu_test_helper 方法，传入一个大小为 (2, 4) 的随机张量作为输入，并指定 gelu 函数
            self._qlinear_unary_cpu_test_helper((torch.randn((2, 4)),), gelu)
    def test_qlinear_gelu_int8_mixed_bf16(self):
        r"""
        This testcase will quantize a Linear->GELU pattern with int8_mixed_bf16 quantization.
        """
        # 循环遍历两种 GELU 激活函数配置，进行测试
        for gelu in [torch.nn.GELU("none"), torch.nn.GELU("tanh")]:
            # 调用辅助函数，测试 QLinear Unary 模式在 CPU 上的量化
            self._qlinear_unary_cpu_test_helper(
                (torch.randn((2, 4)),), gelu, int8_mixed_bf16=True
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_add_cpu(self):
        # 调用辅助函数，测试 QLinear Add 模式在 CPU 上的量化
        self._qlinear_add_cpu_test_helper()

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_add_int8_mixed_bf16(self):
        # 调用辅助函数，测试 QLinear Add 模式在 CPU 上的 int8_mixed_bf16 量化
        self._qlinear_add_cpu_test_helper(int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_add_relu_cpu(self):
        # 调用辅助函数，测试 QLinear Add 模式在 CPU 上的 ReLU 量化
        self._qlinear_add_cpu_test_helper(use_relu=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_add_relu_int8_mixed_bf16(self):
        # 调用辅助函数，测试 QLinear Add 模式在 CPU 上的 ReLU 和 int8_mixed_bf16 量化
        self._qlinear_add_cpu_test_helper(use_relu=True, int8_mixed_bf16=True)

    def _qlinear_dequant_promotion_cpu_test_helper(
        self,
        inputs,
        int8_mixed_bf16=False,
        is_dynamic=False,
        matcher_check_fn=None,
    ):
        # 定义一个内部测试类 M
        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                # 创建三个线性层对象
                self.linear1 = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 4)
                self.linear3 = torch.nn.Linear(4, 4)

            def forward(self, x):
                # 前向传播函数，应用线性层和加法操作
                temp = self.linear1(x)
                temp = self.linear2(temp) + self.linear3(temp)
                return temp

        # 实例化 M 类为模型对象，并设置为评估模式
        mod = M().eval()

        def default_matcher_check_fn():
            # 默认的匹配器检查函数，验证量化过程中的匹配情况
            # 1. 验证 dequant promotion 匹配计数为 1
            self.assertEqual(counters["inductor"]["dequant_promotion_matcher_count"], 1)
            # 2. 验证 qlinear_weight_prepack 匹配计数为 3
            self.assertEqual(
                counters["inductor"]["qlinear_weight_prepack_matcher_count"], 3
            )
            # 3. 验证 qlinear_unary 匹配计数为 1
            self.assertEqual(counters["inductor"]["qlinear_unary_matcher_count"], 1)

        # 调用通用测试函数，测试模型 mod 在输入 inputs 上的表现
        self._test_common(
            mod,
            inputs,
            check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
            check_quantization=True,
            matcher_check_fn=matcher_check_fn
            if matcher_check_fn is not None
            else default_matcher_check_fn,
            is_dynamic=is_dynamic,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_dequant_promotion_cpu(self):
        r"""
        This testcase test if dequant node before linear is promoted correctly:
                  X
                  |
               Linear1(X)
                /   \
        Linear2(X)   Linear3(X)
                \   /
                 Add
                  |
                  Y
        """
        # 调用帮助函数 _qlinear_dequant_promotion_cpu_test_helper，测试 CPU 下的量化线性层前的去量化节点是否正确推广
        self._qlinear_dequant_promotion_cpu_test_helper((torch.randn((2, 4)),))

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_dequant_promotion_int8_mixed_bf16(self):
        r"""
        Test with int8_mixed_bf16 quantization.
        This testcase test if dequant node before linear is promoted correctly:
                  X
                  |
               Linear1(X)
                /   \
        Linear2(X)   Linear3(X)
                \   /
                 Add
                  |
                  Y
        """
        # 调用帮助函数 _qlinear_dequant_promotion_cpu_test_helper，测试 int8_mixed_bf16 量化模式下量化线性层前的去量化节点是否正确推广
        self._qlinear_dequant_promotion_cpu_test_helper(
            (torch.randn((2, 4)),), int8_mixed_bf16=True
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_dequant_promotion_cpu_input_dim_exceeds_2(self):
        r"""
        This testcase test if dequant node before linear is promoted correctly:
                  X
                  |
               Linear1(X)
                /   \
        Linear2(X)   Linear3(X)
                \   /
                 Add
                  |
                  Y
        """
        # 调用帮助函数 _qlinear_dequant_promotion_cpu_test_helper，测试 CPU 下输入维度超过 2 时量化线性层前的去量化节点是否正确推广
        self._qlinear_dequant_promotion_cpu_test_helper((torch.randn((2, 3, 4)),))

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_dequant_promotion_int8_mixed_bf16_input_dim_exceeds_2(self):
        r"""
        Test with int8_mixed_bf16 quantization.
        This testcase test if dequant node before linear is promoted correctly:
                  X
                  |
               Linear1(X)
                /   \
        Linear2(X)   Linear3(X)
                \   /
                 Add
                  |
                  Y
        """
        # 调用帮助函数 _qlinear_dequant_promotion_cpu_test_helper，测试 int8_mixed_bf16 量化模式下输入维度超过 2 时量化线性层前的去量化节点是否正确推广
        self._qlinear_dequant_promotion_cpu_test_helper(
            (torch.randn((2, 3, 4)),), int8_mixed_bf16=True
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_dequant_promotion_dynamic_cpu(self):
        r"""
        This testcase test if dequant node before linear is promoted correctly:
                  X
                  |
               Linear1(X)
                /   \
        Linear2(X)   Linear3(X)
                \   /
                 Add
                  |
                  Y
        """

        def matcher_check_fn():
            # 1. Dequant pattern matcher for dequant promotion * 1
            # 检查去量化模式匹配器是否匹配了一次去量化促进
            self.assertEqual(counters["inductor"]["dequant_promotion_matcher_count"], 1)
            # 2. dequant-linear pattern matched in quantization weight prepack * 3
            # 检查量化权重预打包中是否匹配了三次去量化-线性模式
            self.assertEqual(
                counters["inductor"]["qlinear_weight_prepack_matcher_count"], 3
            )

        self._qlinear_dequant_promotion_cpu_test_helper(
            (torch.randn((2, 4)),),
            matcher_check_fn=matcher_check_fn,
            is_dynamic=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_mul_cpu(self):
        r"""
        This testcase will quantize a Linear->Mul pattern.
        """

        class M(torch.nn.Module):
            def __init__(self, use_bias):
                super().__init__()
                self.linear = torch.nn.Linear(4, 5, use_bias)

            def forward(self, x1, x2):
                return torch.mul(self.linear(x1), x2)

        bias_list = [True, False]
        for bias in bias_list:
            mod = M(bias).eval()
            x1 = torch.randn((2, 4))
            x2 = torch.randn((2, 5))

            def matcher_check_fn():
                # 检查量化权重预打包中是否匹配了一次去量化-线性模式
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 1
                )

            self._test_common(
                mod,
                (x1, x2),
                check_quantization=True,
                matcher_check_fn=matcher_check_fn,
            )

    @skipIfNoDynamoSupport
    def test_qmaxpool2d(self):
        r"""
        This testcase will quantize Conv2d->ReLU->MaxPool2d pattern.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                kwargs,
            ):
                super().__init__()
                # 定义卷积层，输入通道为3，输出通道为64，卷积核大小为7x7，步长为2，填充为3，无空洞
                self.conv = torch.nn.Conv2d(
                    3, 64, 7, bias=True, stride=2, padding=3, dilation=1
                )
                # ReLU 激活函数
                self.relu = torch.nn.ReLU()
                # 最大池化层，池化窗口大小为3x3，根据传入的参数 kwargs 进行设置
                self.maxpool = torch.nn.MaxPool2d(3, **kwargs)

            def forward(self, x):
                # 前向传播函数，先卷积，然后 ReLU，最后最大池化
                return self.maxpool(self.relu(self.conv(x)))

        kwargs_list = [
            {"stride": 2},
            {"stride": 2, "padding": 1},
            {"stride": 2, "padding": 1, "dilation": 1},
            {"stride": 2, "padding": 1, "dilation": 1, "ceil_mode": False},
        ]
        # 遍历参数列表，创建模型并进行测试
        for kwargs in kwargs_list:
            mod = M(kwargs).eval()
            # 创建输入数据张量，形状为(1, 3, 8, 8)，数据类型为 float32
            v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
                1
            )

            def matcher_check_fn():
                # 检查量化匹配器的计数
                self.assertEqual(counters["inductor"]["qmaxpool2d_matcher_count"], 1)
                self.assertEqual(
                    counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 1
                )
                self.assertEqual(counters["inductor"]["qconv2d_unary_matcher_count"], 1)

            # 调用通用测试方法
            self._test_common(
                mod,
                (v,),
                check_quantization=True,
                matcher_check_fn=matcher_check_fn,
            )

    @skipIfNoDynamoSupport
    def test_qflatten(self):
        r"""
        This testcase will quantize Conv2d->AdaptiveAvgPool2d->flatten pattern.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                # 定义卷积层，输入通道为3，输出通道为64，卷积核大小为7x7，步长为2，填充为3，无空洞
                self.conv = torch.nn.Conv2d(
                    3, 64, 7, bias=True, stride=2, padding=3, dilation=1
                )
                # ReLU 激活函数
                self.relu = torch.nn.ReLU()
                # 自适应平均池化层，输出大小为(1, 1)
                self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                # 前向传播函数，先卷积，然后 ReLU，接着自适应平均池化，最后展平
                return torch.flatten(
                    self.adaptive_avg_pool2d(self.relu(self.conv(x))), 1
                )

        mod = M().eval()
        # 创建输入数据张量，形状为(1, 3, 8, 8)，数据类型为 float32
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        def matcher_check_fn():
            # 检查量化匹配器的计数
            self.assertEqual(counters["inductor"]["qreshape_matcher_count"], 1)

        # 调用通用测试方法
        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            matcher_check_fn=matcher_check_fn,
        )

    @skipIfNoDynamoSupport
    # 定义一个名为 test_qcat 的测试方法
    def test_qcat(self):
        r"""
        This testcase will quantize cat based pattern:
                X
             /     \
        Conv1(X)  Pow(x)
            \        \
             \     Conv2(X)
              \    /
               Cat
                |
                Y
        """

        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(
                self,
            ):
                super().__init__()
                # 创建一个 2D 卷积层，输入通道为 3，输出通道为 64，卷积核大小为 7x7
                # 包括偏置项，步幅为 2，填充为 3，扩张系数为 1
                self.conv = torch.nn.Conv2d(
                    3, 64, 7, bias=True, stride=2, padding=3, dilation=1
                )
                # 创建第二个 2D 卷积层，与第一个参数相同
                self.conv2 = torch.nn.Conv2d(
                    3, 64, 7, bias=True, stride=2, padding=3, dilation=1
                )

            # 前向传播方法
            def forward(self, x):
                # 对输入 x 进行第一个卷积操作
                temp1 = self.conv(x)
                # 对 x 的平方进行第二个卷积操作
                temp2 = self.conv2(torch.pow(x, 2))
                # 将两个卷积层的结果在第二维度上拼接起来
                return torch.cat((temp1, temp2), 1)

        # 创建 M 类的实例 mod，并设为评估模式
        mod = M().eval()
        # 创建一个形状为 (1, 3, 8, 8) 的张量 v，其值为标准正态分布加 1
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        # 定义一个名为 matcher_check_fn 的函数
        def matcher_check_fn():
            # 断言 counters["inductor"]["qcat_matcher_count"] 的值为 1
            self.assertEqual(counters["inductor"]["qcat_matcher_count"], 1)
            # 断言 counters["inductor"]["qconv2d_weight_prepack_matcher_count"] 的值为 2
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 2
            )
            # 断言 counters["inductor"]["qconv2d_unary_matcher_count"] 的值为 2
            self.assertEqual(counters["inductor"]["qconv2d_unary_matcher_count"], 2)

        # 调用 _test_common 方法，测试 mod 模型，传入参数 v，并进行量化检查，使用 matcher_check_fn 进行检查

    # 定义一个名为 test_hardtanh_pattern_fallback 的测试方法
    # 此方法用于测试在特定情况下的 HardTanh 模式回退
    def test_hardtanh_pattern_fallback(self):
        # 定义一个内部类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个转置卷积层，输入通道为 3，输出通道为 32，卷积核大小为 3x3
                # 步幅为 1，填充为 1
                self.conv_transpose = torch.nn.ConvTranspose2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            # 前向传播方法，接受输入 x，最小值 min_value，最大值 max_value
            def forward(self, x, min_value, max_value):
                # 对输入 x 进行转置卷积操作
                conv_transpose_output = self.conv_transpose(x)
                # 对转置卷积输出进行最小值截断操作，下界为 min_value
                clamp_min_output = torch.clamp_min(conv_transpose_output, min_value)
                # 对截断后的张量进行最大值截断操作，上界为 max_value
                clamp_max_output = torch.clamp_max(clamp_min_output, max_value)
                return clamp_max_output

        # 创建一个形状为 (1, 3, 28, 28) 的张量 v
        v = torch.randn(1, 3, 28, 28)
        # 创建两个 min_value 和 max_value 的列表，分别为标量和形状为 (1, 32, 28, 28) 的张量
        min_values = [3, torch.randn(1, 32, 28, 28)]
        max_values = [0, torch.randn(1, 32, 28, 28)]
        
        # 遍历 min_values 和 max_values 列表
        for min_value, max_value in zip(min_values, max_values):
            # 创建 Model 类的实例 mod，并设为评估模式
            mod = Model().eval()
            # 调用 _test_common 方法，测试 mod 模型，传入参数 v、min_value 和 max_value
            # 参数 2 和 4 分别用于检查 quantized_op_count 和 matcher_count
    # 定义一个测试用例方法，用于测试 Leaky ReLU 模式的回退情况
    def test_leaky_relu_pattern_fallback(self):
        # 定义一个名为 Model 的内部类，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 构造方法，初始化模型结构
            def __init__(self):
                super().__init__()
                # 定义一个卷积层，输入通道为 3，输出通道为 32，卷积核大小为 3x3，步长为 1，填充为 1
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            # 前向传播方法，接收输入 x 和负斜率 negative_slope
            def forward(self, x, negative_slope):
                # 对输入 x 进行卷积操作
                conv_out = self.conv(x)
                # 使用 torch.where 函数根据条件选择返回值，大于 0 的保持不变，否则乘以 negative_slope
                return torch.where(conv_out > 0, conv_out, conv_out * negative_slope)

        # 定义负斜率的列表，包含常数 0.1 和形状为 (1, 32, 28, 28) 的随机张量
        negative_slopes = [0.1, torch.randn(1, 32, 28, 28)]
        
        # 使用 torch.no_grad() 上下文管理器，关闭梯度计算
        with torch.no_grad():
            # 生成形状为 (1, 3, 28, 28) 的随机张量 v
            v = torch.randn(1, 3, 28, 28)
            # 遍历负斜率列表
            for negative_slope in negative_slopes:
                # 创建并评估模型实例 mod
                mod = Model().eval()
                # 调用 self._test_common 方法进行通用测试，传入参数 (v, negative_slope)，期望结果数量为 2，重复次数为 5

    # https://github.com/pytorch/pytorch/issues/99838.
    # 定义一个测试方法，用于测试 Conv2d 添加标量的情况
    def test_conv2d_add_scalar(self):
        # 定义一个名为 Model 的内部类，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 构造方法，初始化模型结构
            def __init__(self):
                super().__init__()
                # 定义一个卷积层，输入通道为 3，输出通道为 32，卷积核大小为 3x3，步长为 1，填充为 1
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            # 前向传播方法，接收输入 x
            def forward(self, x):
                # 对输入 x 进行卷积操作
                out_conv = self.conv(x)
                # 使用 torch.add 函数对卷积输出加上标量 1.0
                out = torch.add(out_conv, 1.0)
                # 返回加法结果
                return out

        # 使用 torch.no_grad() 上下文管理器，关闭梯度计算
        with torch.no_grad():
            # 创建并评估模型实例 mod
            mod = Model().eval()
            # 生成形状为 (1, 3, 28, 28) 的随机张量 v
            v = torch.randn(1, 3, 28, 28)
            # 调用 self._test_common 方法进行通用测试，传入参数 (v,)，期望结果数量为 1，重复次数为 1

    # 定义一个测试方法，用于测试 Conv2d 二进制就地融合传递在 CPU 上的情况
    def test_conv2d_binary_inplace_fusion_pass_cpu(
        self, include_ops=None, exclude_ops=None
    ):
    ):
        # 定义一个名为 Model_v1 的类，继承自 torch.nn.Module
        class Model_v1(torch.nn.Module):
            # 初始化方法，定义了一个卷积层对象
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            # 前向传播方法，接收两个参数 x 和 other
            def forward(self, x, other):
                # 对输入 x 进行卷积操作
                conv_out = self.conv(x)
                # 返回卷积输出和 other 的 ReLU 激活的和
                return torch.add(conv_out, other.relu())

        # 定义一个名为 Model_v2 的类，继承自 torch.nn.Module
        class Model_v2(torch.nn.Module):
            # 初始化方法，定义了三个卷积层对象
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )
                self.conv2 = torch.nn.Conv2d(
                    in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
                )
                self.conv3 = torch.nn.Conv2d(
                    in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            # 前向传播方法，接收两个参数 x 和 _
            def forward(self, x, _):
                # 第一个卷积层
                conv_out1 = self.conv(x)
                # 对第一个卷积层输出的结果进行平方操作
                pow_out = torch.pow(conv_out1, 2)
                # 第二个卷积层
                conv_out2 = self.conv2(pow_out)
                # 第三个卷积层
                conv_out3 = self.conv3(conv_out2)
                # 将第三个卷积层的输出与第一个卷积层的平方结果相加
                res = torch.add(conv_out3, pow_out)
                # 返回最终结果
                return res

        # 创建一个 1x3x28x28 的随机张量输入，存储格式设置为通道为最后一维
        input = torch.randn(1, 3, 28, 28).to(memory_format=torch.channels_last)
        # 创建两个与 input 大小相同的随机张量作为 others 列表的元素
        others = [
            torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last),
            torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last),
        ]
        # 创建并初始化 Model_v1 和 Model_v2，并设置存储格式为通道为最后一维，然后设为评估模式
        mod_v1 = Model_v1().to(memory_format=torch.channels_last).eval()
        mod_v2 = Model_v2().to(memory_format=torch.channels_last).eval()

        # 如果 include_ops 参数为 None，则设为指定默认值的列表
        if include_ops is None:
            include_ops = ["mkldnn._convolution_pointwise_.binary"]
        # 如果 exclude_ops 参数为 None，则设为指定默认值的列表
        if exclude_ops is None:
            exclude_ops = ["mkldnn._convolution_pointwise.binary"]

        # 遍历 others 列表和 [mod_v1, mod_v2] 列表，分别调用 _test_code_common 方法
        for other, mod in zip(others, [mod_v1, mod_v2]):
            self._test_code_common(mod, (input, other), include_ops, exclude_ops)

    # 定义一个名为 test_conv2d_binary_inplace_fusion_failed_cpu 的测试方法
    def test_conv2d_binary_inplace_fusion_failed_cpu(
        self, include_ops=None, exclude_ops=None
    # 定义一个测试用例，测试二维卷积与其他操作融合失败的情况
    def test_conv2d_binary_fusion_failed(self):
        # 在 alpha 不等于 1 或其他情况下，我们不支持融合，或者其他操作的输出与卷积输出大小不同的情况。
        
        # 定义模型类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个二维卷积层，输入通道数为 3，输出通道数为 32，卷积核大小为 3，步长为 1，填充为 1
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            # 定义前向传播函数，接受输入 x、其他操作的输出 other 和参数 alpha
            def forward(self, x, other, alpha):
                # 对输入 x 进行卷积操作
                conv_out = self.conv(x)
                # 返回卷积输出与 other 的加权和，权重为 alpha
                return torch.add(conv_out, other, alpha=alpha)

        # GitHub 上的问题链接：https://github.com/pytorch/pytorch/issues/100802.
        # 当 add 操作的输入是同一个张量时，我们无法进行融合。
        class Model2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个二维卷积层，输入通道数为 3，输出通道数为 16，卷积核大小为 3，步长为 1，填充为 1
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
                )

            # 定义前向传播函数，接受输入 x
            def forward(self, x):
                # 对输入 x 进行卷积操作
                out = self.conv(x)
                # 对卷积结果 out 自身进行加法操作
                out = torch.add(out, out)
                # 返回结果 out
                return out

        # GitHub 上的问题链接：https://github.com/pytorch/pytorch/issues/101374.
        # 当 add 操作的输入具有不同的数据类型时，我们无法进行融合。
        class Model3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个二维卷积层，输入通道数为 3，输出通道数为 16，卷积核大小为 3，步长为 1，填充为 1
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
                )

            # 定义前向传播函数，接受输入 x
            def forward(self, x):
                # 对输入 x 进行卷积操作
                temp = self.conv(x)
                # 创建一个与 temp 形状相同且数据类型为 double 的张量 other，值为全 1
                other = torch.ones(temp.shape, dtype=torch.double)
                # 对 temp 和 other 进行加法操作
                out = torch.add(temp, other)
                # 返回结果 out
                return out

        # 创建一个形状为 (1, 3, 28, 28) 的随机张量 input，并将其存储格式设为通道最后格式
        input = torch.randn(1, 3, 28, 28).to(memory_format=torch.channels_last)
        # 创建两个张量作为 others 列表的元素，分别为随机张量，形状为 (1, 32, 28, 28)，存储格式为通道最后格式，
        # 和形状为 (32, 28, 28) 的随机张量
        others = [
            torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last),
            torch.randn(32, 28, 28),
        ]
        # 包含的操作列表，用于测试用例
        include_ops = ["mkldnn._convolution_pointwise"]
        # 排除的操作列表，用于测试用例
        exclude_ops = [
            "mkldnn._convolution_pointwise.binary",
            "mkldnn._convolution_pointwise_.binary",
        ]

        # case1
        # 遍历 others 列表中的元素以及 alpha 列表中的值，执行以下操作
        for other, alpha in zip(others, [0.1, 1.0]):
            # 创建一个 Model 类的实例 mod，并设置存储格式为通道最后格式，然后转换为评估模式
            mod = Model().to(memory_format=torch.channels_last).eval()
            # 调用自定义的 _test_code_common 方法，传入 mod、input、other 和 alpha 等参数，以及 include_ops 和 exclude_ops
            self._test_code_common(mod, (input, other, alpha), include_ops, exclude_ops)
        
        # case2:
        # 创建一个 Model2 类的实例 mod，并设置存储格式为通道最后格式，然后转换为评估模式
        mod = Model2().to(memory_format=torch.channels_last).eval()
        # 调用自定义的 _test_code_common 方法，传入 mod 和 input 参数，以及 include_ops 和 exclude_ops
        self._test_code_common(mod, (input,), include_ops, exclude_ops)
        
        # case3:
        # 创建一个 Model3 类的实例 mod，并设置存储格式为通道最后格式，然后转换为评估模式
        mod = Model3().to(memory_format=torch.channels_last).eval()
        # 调用自定义的 _test_code_common 方法，传入 mod 和 input 参数，以及 include_ops 和 exclude_ops
        self._test_code_common(mod, (input,), include_ops, exclude_ops)
    def test_reproduce_99842_issue(self):
        # 定义一个继承自torch.nn.Module的模型类Model
        class Model(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 添加一个二维卷积层，输入通道数3，输出通道数64，卷积核大小3x3，步长1，填充1
                self.conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

            # 前向传播方法
            def forward(self, input_tensor):
                # 对输入数据进行卷积操作
                x = self.conv(input_tensor)
                # 将卷积结果与全1张量相加，并使用ReLU激活函数
                x = F.relu(x + torch.ones(x.size()))
                return x

        # 生成一个随机输入张量，大小为1x3x14x14
        input = torch.randn(1, 3, 14, 14)
        # 创建Model类的实例，并设为评估模式
        mod = Model().eval()
        # 定义需要包含的操作列表
        include_ops = ["mkldnn._convolution_pointwise_.binary"]
        # 调用自定义的测试函数_test_code_common，传入模型实例、输入数据、需要包含的操作列表和空列表
        self._test_code_common(mod, (input,), include_ops, [])

    def test_reproduce_113440_issue_1(self):
        # 定义一个继承自torch.nn.Module的模型类Mod
        class Mod(torch.nn.Module):
            # 初始化方法
            def __init__(
                self,
                add_fn,
                **kwargs,
            ):
                super().__init__()
                # 添加两个二维卷积层，每层输入通道数3，输出通道数6，卷积核大小3x3，步长1
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.add_fn = add_fn  # 保存传入的add_fn函数
                self.relu = torch.nn.ReLU(inplace=True)  # 添加ReLU激活函数层
                self.conv3 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv4 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.add_fn2 = add_fn  # 再次保存传入的add_fn函数
                self.relu2 = torch.nn.ReLU(inplace=True)  # 添加第二个ReLU激活函数层
                self.use_relu = True  # 标志位，指示是否使用ReLU激活函数

            # 前向传播方法
            def forward(self, x):
                # 第一个输入数据经过第一个卷积层
                x1 = self.conv1(x)
                # 第二个输入数据经过第二个卷积层
                x2 = self.conv2(x)
                # 将两个卷积结果按照传入的add_fn函数进行相加操作
                tmp = self.add_fn(x1, x2)
                # 如果标志位use_relu为True，则对结果应用ReLU激活函数
                if self.use_relu:
                    tmp = self.relu(tmp)
                # 对临时结果应用第三个卷积层
                tmp1 = self.conv3(tmp)
                # 对临时结果应用第四个卷积层
                tmp2 = self.conv4(tmp)
                # 将两个卷积结果再次按照传入的add_fn函数进行相加操作
                res = self.add_fn2(tmp1, tmp2)
                # 如果标志位use_relu为True，则对结果应用第二个ReLU激活函数
                if self.use_relu:
                    res = self.relu2(res)
                return res

        # 禁用梯度计算环境
        with torch.no_grad():
            # 创建一个包含随机数据的示例输入元组
            example_inputs = (
                torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
                    1
                ),
            )
            # 获取示例输入的设备信息
            example_inputs[0].get_device()
            # 创建Mod类的实例，并设为评估模式，传入lambda函数作为add_fn参数
            m = Mod(
                lambda x, y: x.add_(y),
            ).eval()
            # 编译模型m
            om = torch.compile(m)
            # 调用编译后的模型om，传入示例输入数据，两次
            om(*example_inputs)
            om(*example_inputs)
    # 定义一个测试函数，用于复现特定问题113440的第二个问题
    def test_reproduce_113440_issue_2(self):
        # 定义一个继承自torch.nn.Module的子类Mod
        class Mod(torch.nn.Module):
            # 初始化函数，接受一个add_fn参数和其他关键字参数
            def __init__(
                self,
                add_fn,
                **kwargs,
            ):
                # 调用父类的初始化方法
                super().__init__()
                # 定义多个卷积层，每个层的参数不同
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                # 将传入的add_fn函数赋值给对象的属性
                self.add_fn = add_fn
                # 创建一个ReLU激活函数对象
                self.relu = torch.nn.ReLU(inplace=True)
                self.conv3 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv4 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                # 再次将传入的add_fn函数赋值给对象的属性
                self.add_fn2 = add_fn
                # 创建第二个ReLU激活函数对象
                self.relu2 = torch.nn.ReLU(inplace=True)

                self.conv5 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv6 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv7 = torch.nn.Conv2d(6, 6, kernel_size=1, stride=1)
                # 第三次将传入的add_fn函数赋值给对象的属性
                self.add_fn3 = add_fn
                # 创建第三个ReLU激活函数对象
                self.relu3 = torch.nn.ReLU(inplace=True)

                # 设置一个标志位
                self.use_relu = True

            # 前向传播函数，接受输入x，返回处理后的结果
            def forward(self, x):
                # 对输入x分别进行两次卷积操作，得到x1和x2
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                # 调用传入的add_fn函数对x1进行原地加法操作，结果保存在tmp中
                tmp = self.add_fn(x1, x2)
                # 如果标志位use_relu为True，则对tmp应用ReLU激活函数
                if self.use_relu:
                    tmp = self.relu(tmp)

                # 对tmp进行一次卷积操作，得到tmp1
                tmp1 = self.conv3(tmp)
                # 对tmp1应用第二个ReLU激活函数，得到最终结果res
                res = self.relu2(tmp1)

                # 返回最终结果res
                return res

        # 使用torch.no_grad()上下文管理器
        with torch.no_grad():
            # 创建一个例子输入，是一个符合要求的随机张量
            example_inputs = (
                torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
                    1
                ),
            )
            # 创建Mod类的实例m，传入一个lambda函数作为add_fn参数，并设置为eval模式
            m = Mod(
                lambda x, y: x.add_(y),
            ).eval()
            # 编译模型m，得到om
            om = torch.compile(m)
            # 分别用example_inputs作为输入调用om两次
            om(*example_inputs)
            om(*example_inputs)

    # torch._dynamo.config.patch装饰器，配置inline_inbuilt_nn_modules为True
    @torch._dynamo.config.patch("inline_inbuilt_nn_modules", True)
    # 定义一个测试方法，用于验证特定问题的复现
    def test_reproduce_121253_issue(self):
        # 定义一个简单的神经网络模块类
        class Mod(torch.nn.Module):
            def __init__(self, weight, bias, beta, alpha):
                super().__init__()
                # 初始化模块的权重、偏置、beta和alpha参数
                self.weight = weight
                self.bias = bias
                self.beta = beta
                self.alpha = alpha

            # 定义模块的前向传播方法
            def forward(self, x):
                # 使用torch.addmm函数进行线性变换
                return torch.addmm(
                    self.bias, x, self.weight, beta=self.beta, alpha=self.alpha
                )

        # 初始化数据类型列表，开始时只包含torch.float32类型
        dtypes = [torch.float32]
        # 检查是否支持MKL-DNN的bfloat16数据类型，若支持则添加到数据类型列表中
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        # 遍历数据类型列表
        for dtype in dtypes:
            # 根据数据类型选择不同的线性操作名称
            linear_op = (
                "mkl._mkl_linear"
                if dtype == torch.float32
                else "mkldnn._linear_pointwise"
            )
            # 遍历beta和alpha参数的组合
            for beta, alpha in zip([1.0, 0.1, 0.0], [1.0, 0.1, 1.0]):
                # 初始化随机权重和偏置参数
                weight = torch.nn.Parameter(torch.randn(64, 64, dtype=dtype))
                bias = torch.nn.Parameter(torch.randn(64, dtype=dtype))
                # 创建Mod类的实例，并将其转换到指定的数据类型并设为评估模式
                mod = Mod(weight, bias, beta, alpha).to(dtype).eval()
                # 使用torch.no_grad()上下文管理器，确保在评估模式下执行前向传播
                with torch.no_grad():
                    # 生成随机输入数据x
                    x = torch.randn(1, 64, dtype=dtype)
                    # 初始化包含和排除操作的空列表
                    include_ops = []
                    exclude_ops = []
                    # 根据beta和alpha参数值判断是否需要排除特定的线性操作
                    if (beta != 1.0 and beta != 0.0) or alpha != 1.0:
                        exclude_ops = [linear_op]
                    else:
                        include_ops = [linear_op]
                    # 调用测试共同方法来测试模块的行为
                    self._test_code_common(mod, (x,), include_ops, exclude_ops)

    # 使用装饰器进行条件跳过测试，若无Dynamo支持则跳过测试
    @skipIfNoDynamoSupport
    def test_woq_int8(self):
        # 定义一个简单的神经网络模块类
        class M(torch.nn.Module):
            # 定义模块的前向传播方法，接受输入x、权重weight和缩放因子scales
            def forward(self, x, weight, scales):
                # 使用torch.nn.functional.linear函数执行线性操作，并乘以缩放因子
                return torch.nn.functional.linear(x, weight.to(dtype=x.dtype)) * scales

        # 创建M类的实例，并设为评估模式
        mod = M().eval()
        # 初始化输入x、权重w和缩放因子s的形状
        x_shape = (1, 1, 256)
        w_shape = (12, 256)
        s_shape = 12
        # 定义多个输入x的步幅（stride）
        x_strides = [
            (256, 256, 1),  # 线性分发到mm
            (256, 32, 1),   # 线性分发到bmm
        ]
        # 遍历输入x的步幅列表
        for x_stride in x_strides:
            # 生成随机输入x，并使用as_strided方法设置步幅
            x = torch.randn(x_shape, dtype=torch.bfloat16).as_strided(x_shape, x_stride)
            # 生成随机权重w，数据类型为torch.int8
            w = torch.randint(-128, 127, w_shape, dtype=torch.int8)
            # 生成随机缩放因子s，数据类型为torch.bfloat16
            s = torch.randn(s_shape, dtype=torch.bfloat16)

            # 定义匹配器检查函数，验证特定计数器的匹配次数
            def matcher_check_fn():
                self.assertEqual(counters["inductor"]["woq_matcher_count"], 1)

            # 调用测试共同方法来测试模块的行为，禁用量化检查，设置绝对误差和相对误差的阈值
            self._test_common(
                mod,
                (x, w, s),
                matcher_check_fn=matcher_check_fn,
                check_quantization=False,
                atol=0.001,
                rtol=0.07,
            )
# 使用修饰器 patch 进行动态配置，设置 dynamic_shapes 为 True，assume_static_by_default 为 False
@dynamo_config.patch({"dynamic_shapes": True, "assume_static_by_default": False})
# 定义 TestDynamicPatternMatcher 类，继承自 TestPatternMatcherBase 类
class TestDynamicPatternMatcher(TestPatternMatcherBase):
    # 将 TestPatternMatcher 类的 _test_conv_unary_cpu_base 属性赋值给 _test_conv_unary_cpu_base
    _test_conv_unary_cpu_base = TestPatternMatcher._test_conv_unary_cpu_base
    # 定义 test_conv2d_unary_dynamic_shapes 方法，其实现与 TestPatternMatcher 类的 test_conv2d_unary_cpu 方法相同
    test_conv2d_unary_dynamic_shapes = TestPatternMatcher.test_conv2d_unary_cpu
    # 定义 test_conv3d_unary_dynamic_shapes 方法，其实现与 TestPatternMatcher 类的 test_conv3d_unary_cpu 方法相同
    test_conv3d_unary_dynamic_shapes = TestPatternMatcher.test_conv3d_unary_cpu
    # 将 TestPatternMatcher 类的 _test_conv_binary_base 属性赋值给 _test_conv_binary_base
    _test_conv_binary_base = TestPatternMatcher._test_conv_binary_base
    # 定义 test_conv2d_binary_dynamic_shapes 方法，其实现与 TestPatternMatcher 类的 test_conv2d_binary 方法相同
    test_conv2d_binary_dynamic_shapes = TestPatternMatcher.test_conv2d_binary
    # 定义 test_conv3d_binary_dynamic_shapes 方法，其实现与 TestPatternMatcher 类的 test_conv3d_binary 方法相同
    test_conv3d_binary_dynamic_shapes = TestPatternMatcher.test_conv3d_binary
    # 定义 test_linear_unary_dynamic_shapes 方法，其实现与 TestPatternMatcher 类的 test_linear_unary 方法相同
    test_linear_unary_dynamic_shapes = TestPatternMatcher.test_linear_unary

    # 定义 test_conv_transpose2d_dynamic_shapes 方法
    def test_conv_transpose2d_dynamic_shapes(self):
        # 我们目前不支持 conv_transpose2d。
        # 定义 M 类，继承自 torch.nn.Module
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个 ConvTranspose2d 层
                self.conv_transpose2d = torch.nn.ConvTranspose2d(
                    3, 16, 3, stride=2, padding=1
                )

            # 前向传播函数，接收输入 x，返回 conv_transpose2d 层的输出
            def forward(self, x):
                return self.conv_transpose2d(x)

        # 定义输入张量的形状
        x_shape = (1, 3, 28, 28)
        # 创建 M 类的实例，并设置为评估模式
        mod = M().eval()
        # 生成一个符合正态分布的随机张量 v，数据类型为 torch.float32
        v = torch.randn(x_shape, dtype=torch.float32)
        # 调用 _test_common 方法，传入模型 mod、输入张量 v，以及额外的参数 0 和 0
        self._test_common(mod, (v,), 0, 0)

    # 定义 test_multi_linear_share_same_input_dynamic 方法
    def test_multi_linear_share_same_input_dynamic(self):
        # llama pattern.
        # 定义 M 类，继承自 torch.nn.Module
        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                # 初始化两个 Linear 层，没有偏置项
                self.w1 = torch.nn.Linear(16, 16, bias=False)
                self.w2 = torch.nn.Linear(16, 16, bias=False)

            # 前向传播函数，接收输入 x，返回 w1(x) 经过 silu 和 w2(x) 经过 relu 的乘积
            def forward(self, x):
                return F.silu(self.w1(x)) * F.relu(self.w2(x))

        # 初始化一个空列表 dtypes
        dtypes = []
        # 如果支持 MKLDNN 的 bfloat16 数据类型，将其添加到 dtypes 列表中
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        # 如果支持 MKLDNN 的 float16 数据类型，将其添加到 dtypes 列表中
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        # 遍历 dtypes 列表中的数据类型
        for dtype in dtypes:
            # 创建 M 类的实例，并将其移动到指定的 dtype，同时设置为评估模式
            mod = M().to(dtype).eval()
            # 生成一个符合正态分布的随机张量 v，形状为 (2, 4, 16)，并将其转换为指定的 dtype
            v = torch.randn(2, 4, 16).to(dtype)
            # 对模型进行常见测试，传入模型 mod、输入张量 v，以及额外的 match_count 和 match_nodes 参数
            # 还指定了相对和绝对误差的容忍度 rtol 和 atol
            match_count = 10
            match_nodes = 19
            self._test_common(mod, (v,), match_count, match_nodes, rtol=1e-2, atol=1e-2)
    def test_qconv2d_maxpool2d_linear_dynamic_cpu(self, include_ops=None):
        r"""
        This testcase will quantize a single Conv2d->Maxpool2d->Linear module
        with dynamic batch size input.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                # 定义一个2x2的卷积层，输入通道数为3，输出通道数为16，步长为1，填充为1
                self.conv = torch.nn.Conv2d(
                    3, 16, (2, 2), stride=(1, 1), padding=(1, 1)
                )
                # 定义ReLU激活函数
                self.relu = torch.nn.ReLU()
                # 定义一个3x3的最大池化层，步长为2，填充为1
                self.maxpool2d = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                # 定义一个自适应平均池化层，输出大小为1x1
                self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                # 定义一个线性层，输入特征数为16，输出特征数为16
                self.linear = torch.nn.Linear(16, 16)

            def forward(self, x):
                # 卷积层->ReLU激活
                temp = self.relu(self.conv(x))
                # 最大池化
                temp = self.maxpool2d(temp)
                # 自适应平均池化
                temp = self.avgpool(temp)
                # 展平操作
                temp = torch.flatten(temp, 1)
                # 线性层
                return self.linear(temp)

        mod = M().eval()  # 创建一个评估模式下的M实例
        v = torch.randn((2, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)  # 创建一个形状为(2, 3, 8, 8)，类型为float32的张量，均值为1
        if include_ops is None:
            include_ops = [
                "torch.ops.onednn.qconv2d_pointwise",  # 包括一种操作：onednn中的qconv2d_pointwise
                "torch.ops.quantized.max_pool2d",     # 包括一种操作：量化的max_pool2d
                "torch.ops.onednn.qlinear_pointwise", # 包括一种操作：onednn中的qlinear_pointwise
            ]
        exclude_ops = []  # 排除操作为空列表
        self._test_code_common(
            mod,
            (v,),            # 输入为v的元组
            include_ops,
            exclude_ops,
            check_quantization=True,  # 检查量化为True
            check_dynamic=True,      # 检查动态操作为True
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qat_bn_conv2d(self):
        r"""
        This testcase will quantize a single BN Conv2d module with qat flow.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                # 定义一个3通道到3通道的卷积层，卷积核大小为3x3
                self.conv = torch.nn.Conv2d(3, 3, 3)
                # 定义一个BatchNorm2d层，输入通道数为3
                self.bn1 = torch.nn.BatchNorm2d(3)
                # 定义一个BatchNorm2d层，输入通道数为3
                self.bn2 = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                # BatchNorm -> 卷积层 -> BatchNorm
                x = self.conv(self.bn1(x))
                return self.bn2(x)

        mod = M().train()  # 创建一个训练模式下的M实例
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)  # 创建一个形状为(1, 3, 8, 8)，类型为float32的张量，均值为1

        def matcher_check_fn():
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 1
            )

        self._test_common(
            mod,
            (v,),                 # 输入为v的元组
            check_quantization=True,  # 检查量化为True
            is_qat=True,          # 是量化训练为True
            matcher_check_fn=matcher_check_fn,  # 匹配器检查函数为matcher_check_fn
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
# 如果脚本正在运行在 Linux 系统，并且具备 CPU 支持，并且 PyTorch 的 MKLDNN 加速可用
if __name__ == "__main__":
    # 检查是否是 Linux 系统
    if IS_LINUX:
        # 检查是否有 CPU 支持
        if HAS_CPU:
            # 检查 PyTorch 是否支持 MKLDNN 加速
            if torch.backends.mkldnn.is_available():
                # 运行测试函数
                run_tests()
```
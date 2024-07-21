# `.\pytorch\test\quantization\bc\test_backward_compatibility.py`

```
# Owner(s): ["oncall: quantization"]

# 导入必要的库和模块
import os
import sys
import unittest
from typing import Set

# 导入torch相关模块
import torch
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.quantization.quantize_fx as quantize_fx
import torch.nn as nn

# 导入量化相关的观察器和方法
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver
from torch.fx import GraphModule

# 导入内部测试模块和量化相关的测试工具
from torch.testing._internal.common_quantization import skipIfNoFBGEMM
from torch.testing._internal.common_quantized import (
    override_qengines,
    qengine_is_fbgemm,
)

# 导入测试工具
from torch.testing._internal.common_utils import IS_AVX512_VNNI_SUPPORTED, TestCase
from torch.testing._internal.quantization_torch_package_models import (
    LinearReluFunctional,
)

# 定义一个函数，用于去除字符串前缀
def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text

# 定义一个函数，用于获取测试文件的各种文件名
def get_filenames(self, subname):
    # 注意：我们从定义测试的模块获取 __file__，因此我们将期望目录放在测试脚本所在的位置，
    # 而不是 test/common_utils.py 所在的位置。
    
    module_id = self.__class__.__module__
    munged_id = remove_prefix(self.id(), module_id + ".")
    test_file = os.path.realpath(sys.modules[module_id].__file__)
    base_name = os.path.join(os.path.dirname(test_file), "../serialized", munged_id)

    subname_output = ""
    if subname:
        base_name += "_" + subname
        subname_output = f" ({subname})"

    # 定义各种预期的文件名
    input_file = base_name + ".input.pt"
    state_dict_file = base_name + ".state_dict.pt"
    scripted_module_file = base_name + ".scripted.pt"
    traced_module_file = base_name + ".traced.pt"
    expected_file = base_name + ".expected.pt"
    package_file = base_name + ".package.pt"
    get_attr_targets_file = base_name + ".get_attr_targets.pt"

    # 返回所有文件名的元组
    return (
        input_file,
        state_dict_file,
        scripted_module_file,
        traced_module_file,
        expected_file,
        package_file,
        get_attr_targets_file,
    )


class TestSerialization(TestCase):
    """Test backward compatiblity for serialization and numerics"""

    # 从 TestCase.assertExpected 复制并修改的方法
    def _test_op(
        self,
        qmodule,
        subname=None,
        input_size=None,
        input_quantized=True,
        generate=False,
        prec=None,
        new_zipfile_serialization=False,
    ):
        r"""Test quantized modules serialized previously can be loaded
        with current code, make sure we don't break backward compatibility for the
        serialization of quantized modules
        """
        (
            input_file,
            state_dict_file,
            scripted_module_file,
            traced_module_file,
            expected_file,
            _package_file,
            _get_attr_targets_file,
        ) = get_filenames(self, subname)
        # 从测试辅助函数获取文件名列表，用于保存和加载测试数据和模型

        # only generate once.
        if generate and qengine_is_fbgemm():
            # 生成数据和模型文件，仅当 generate 标志为真且量化引擎为 fbgemm 时
            input_tensor = torch.rand(*input_size).float()
            if input_quantized:
                # 如果输入量化标志为真，则将输入张量进行量化
                input_tensor = torch.quantize_per_tensor(
                    input_tensor, 0.5, 2, torch.quint8
                )
            torch.save(input_tensor, input_file)
            # 保存量化模块的状态字典
            torch.save(
                qmodule.state_dict(),
                state_dict_file,
                _use_new_zipfile_serialization=new_zipfile_serialization,
            )
            # 保存量化模块的脚本化版本
            torch.jit.save(torch.jit.script(qmodule), scripted_module_file)
            # 保存量化模块的跟踪版本
            torch.jit.save(torch.jit.trace(qmodule, input_tensor), traced_module_file)
            # 保存期望输出结果
            torch.save(qmodule(input_tensor), expected_file)

        # 加载各种文件
        input_tensor = torch.load(input_file)
        qmodule.load_state_dict(torch.load(state_dict_file))
        qmodule_scripted = torch.jit.load(scripted_module_file)
        qmodule_traced = torch.jit.load(traced_module_file)
        expected = torch.load(expected_file)
        # 断言各加载结果是否与期望一致
        self.assertEqual(qmodule(input_tensor), expected, atol=prec)
        self.assertEqual(qmodule_scripted(input_tensor), expected, atol=prec)
        self.assertEqual(qmodule_traced(input_tensor), expected, atol=prec)

    def _test_op_graph(
        self,
        qmodule,
        subname=None,
        input_size=None,
        input_quantized=True,
        generate=False,
        prec=None,
        new_zipfile_serialization=False,
    ):
        r"""
        Input: a floating point module

        If generate == True, traces and scripts the module and quantizes the results with
        PTQ, and saves the results.

        If generate == False, traces and scripts the module and quantizes the results with
        PTQ, and compares to saved results.
        """
        (
            input_file,
            state_dict_file,
            scripted_module_file,
            traced_module_file,
            expected_file,
            _package_file,
            _get_attr_targets_file,
        ) = get_filenames(self, subname)

        # only generate once.
        if generate and qengine_is_fbgemm():
            # Generate input tensor and save it
            input_tensor = torch.rand(*input_size).float()
            torch.save(input_tensor, input_file)

            # Convert module to TorchScript and trace it
            scripted = torch.jit.script(qmodule)
            traced = torch.jit.trace(qmodule, input_tensor)

            # Quantize using FBGEMM engine
            def _eval_fn(model, data):
                model(data)

            qconfig_dict = {"": torch.ao.quantization.default_qconfig}
            scripted_q = torch.ao.quantization.quantize_jit(
                scripted, qconfig_dict, _eval_fn, [input_tensor]
            )
            traced_q = torch.ao.quantization.quantize_jit(
                traced, qconfig_dict, _eval_fn, [input_tensor]
            )

            # Save quantized modules and results
            torch.jit.save(scripted_q, scripted_module_file)
            torch.jit.save(traced_q, traced_module_file)
            torch.save(scripted_q(input_tensor), expected_file)

        # Load necessary files
        input_tensor = torch.load(input_file)
        qmodule_scripted = torch.jit.load(scripted_module_file)
        qmodule_traced = torch.jit.load(traced_module_file)
        expected = torch.load(expected_file)

        # Assert expected results with specified precision
        self.assertEqual(qmodule_scripted(input_tensor), expected, atol=prec)
        self.assertEqual(qmodule_traced(input_tensor), expected, atol=prec)

    def _test_obs(
        self, obs, input_size, subname=None, generate=False, check_numerics=True
    ):
        """
        Test observer code can be loaded from state_dict.
        """
        (
            input_file,
            state_dict_file,
            _,
            traced_module_file,
            expected_file,
            _package_file,
            _get_attr_targets_file,
        ) = get_filenames(self, None)

        # Generate data if needed
        if generate:
            input_tensor = torch.rand(*input_size).float()
            torch.save(input_tensor, input_file)
            torch.save(obs(input_tensor), expected_file)
            torch.save(obs.state_dict(), state_dict_file)

        # Load data and state_dict
        input_tensor = torch.load(input_file)
        obs.load_state_dict(torch.load(state_dict_file))
        expected = torch.load(expected_file)

        # Perform numeric checks if specified
        if check_numerics:
            self.assertEqual(obs(input_tensor), expected)
    # 定义一个测试函数，用于验证使用 torch.package 创建的文件在当前的 FX 图模式量化转换中能否正常工作
    def _test_package(self, fp32_module, input_size, generate=False):
        """
        Verifies that files created in the past with torch.package
        work on today's FX graph mode quantization transforms.
        """
        # 获取文件名列表
        (
            input_file,  # 输入张量数据文件
            state_dict_file,  # 状态字典文件（未使用）
            _scripted_module_file,  # 脚本模块文件（未使用）
            _traced_module_file,  # 追踪模块文件（未使用）
            expected_file,  # 预期输出张量数据文件
            package_file,  # torch.package 文件
            get_attr_targets_file,  # get_attr 目标文件
        ) = get_filenames(self, None)

        # 定义包的名称和资源名称
        package_name = "test"
        resource_name_model = "test.pkl"

        # 定义函数：执行量化转换的逻辑
        def _do_quant_transforms(
            m: torch.nn.Module,
            input_tensor: torch.Tensor,
        ) -> torch.nn.Module:
            example_inputs = (input_tensor,)
            # 执行量化转换并保存结果
            qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
            mp = quantize_fx.prepare_fx(m, {"": qconfig}, example_inputs=example_inputs)
            mp(input_tensor)
            mq = quantize_fx.convert_fx(mp)
            return mq

        # 定义函数：获取 get_attr 目标字符串集合
        def _get_get_attr_target_strings(m: GraphModule) -> Set[str]:
            results = set()
            for node in m.graph.nodes:
                if node.op == "get_attr":
                    results.add(node.target)
            return results

        # 如果指定生成文件且量化引擎是 fbgemm，则执行以下逻辑
        if generate and qengine_is_fbgemm():
            # 生成随机输入张量并保存
            input_tensor = torch.randn(*input_size)
            torch.save(input_tensor, input_file)

            # 使用 torch.package 保存模型
            with torch.package.PackageExporter(package_file) as exp:
                exp.intern("torch.testing._internal.quantization_torch_package_models")
                exp.save_pickle(package_name, resource_name_model, fp32_module)

            # 执行量化转换并保存结果
            mq = _do_quant_transforms(fp32_module, input_tensor)
            get_attrs = _get_get_attr_target_strings(mq)
            torch.save(get_attrs, get_attr_targets_file)
            q_result = mq(input_tensor)
            torch.save(q_result, expected_file)

        # 加载输入张量和预期输出张量
        input_tensor = torch.load(input_file)
        expected_output_tensor = torch.load(expected_file)
        expected_get_attrs = torch.load(get_attr_targets_file)

        # 从 package 加载模型，并验证输出和 get_attr 目标是否匹配
        imp = torch.package.PackageImporter(package_file)
        m = imp.load_pickle(package_name, resource_name_model)
        mq = _do_quant_transforms(m, input_tensor)

        get_attrs = _get_get_attr_target_strings(mq)
        # 使用断言验证 get_attr 目标是否与预期相符
        self.assertTrue(
            get_attrs == expected_get_attrs,
            f"get_attrs: expected {expected_get_attrs}, got {get_attrs}",
        )
        # 执行量化后模型的输出并使用断言验证与预期输出张量的接近程度
        output_tensor = mq(input_tensor)
        self.assertTrue(torch.allclose(output_tensor, expected_output_tensor))

    # 覆盖量化引擎的装饰器
    @override_qengines
    # 定义一个测试线性层的方法，创建带有偏置的量化线性层对象，输入维度为3，输出维度为1，数据类型为torch.qint8
    def test_linear(self):
        module = nnq.Linear(3, 1, bias_=True, dtype=torch.qint8)
        # 调用测试操作的辅助方法，验证线性层的功能，输入大小为[1, 3]，不生成数据
        self._test_op(module, input_size=[1, 3], generate=False)

    # 重写量化引擎后的测试方法，测试带ReLU的线性层
    def test_linear_relu(self):
        module = nniq.LinearReLU(3, 1, bias=True, dtype=torch.qint8)
        # 调用测试操作的辅助方法，验证带ReLU的线性层功能，输入大小为[1, 3]，不生成数据
        self._test_op(module, input_size=[1, 3], generate=False)

    # 重写量化引擎后的测试方法，测试动态量化的线性层
    def test_linear_dynamic(self):
        # 创建动态量化为qint8的线性层对象，输入维度为3，输出维度为1，带有偏置
        module_qint8 = nnqd.Linear(3, 1, bias_=True, dtype=torch.qint8)
        # 调用测试操作的辅助方法，验证动态量化的qint8线性层功能，输入大小为[1, 3]，不生成数据
        self._test_op(
            module_qint8,
            "qint8",
            input_size=[1, 3],
            input_quantized=False,
            generate=False,
        )
        # 如果量化引擎是FBGEMM，则创建动态量化为float16的线性层对象
        if qengine_is_fbgemm():
            module_float16 = nnqd.Linear(3, 1, bias_=True, dtype=torch.float16)
            # 调用测试操作的辅助方法，验证动态量化的float16线性层功能，输入大小为[1, 3]，不生成数据
            self._test_op(
                module_float16,
                "float16",
                input_size=[1, 3],
                input_quantized=False,
                generate=False,
            )

    # 重写量化引擎后的测试方法，测试二维卷积层
    def test_conv2d(self):
        # 创建带有偏置的二维量化卷积层对象，输入通道数和输出通道数均为3，卷积核大小为3，步长为1，填充为0，扩张为1，组数为1，填充模式为"zeros"
        module = nnq.Conv2d(
            3,
            3,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        # 调用测试操作的辅助方法，验证二维量化卷积层的功能，输入大小为[1, 3, 6, 6]，不生成数据
        self._test_op(module, input_size=[1, 3, 6, 6], generate=False)

    # 重写量化引擎后的测试方法，测试不带偏置的二维卷积层
    def test_conv2d_nobias(self):
        # 创建不带偏置的二维量化卷积层对象，输入通道数和输出通道数均为3，卷积核大小为3，步长为1，填充为0，扩张为1，组数为1，填充模式为"zeros"
        module = nnq.Conv2d(
            3,
            3,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
        )
        # 调用测试操作的辅助方法，验证不带偏置的二维量化卷积层的功能，输入大小为[1, 3, 6, 6]，不生成数据
        self._test_op(module, input_size=[1, 3, 6, 6], generate=False)

    # 重写量化引擎后的测试方法，测试包含图模式的二维卷积层
    def test_conv2d_graph(self):
        # 创建包含量化前处理的量化存根和带有偏置的二维卷积层的序列模块
        module = nn.Sequential(
            torch.ao.quantization.QuantStub(),
            nn.Conv2d(
                3,
                3,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            ),
        )
        # 调用图模式下测试操作的辅助方法，验证包含图模式的二维卷积层的功能，输入大小为[1, 3, 6, 6]，不生成数据
        self._test_op_graph(module, input_size=[1, 3, 6, 6], generate=False)

    # 重写量化引擎后的测试方法，测试包含图模式的无偏置二维卷积层
    def test_conv2d_nobias_graph(self):
        # 创建包含量化前处理的量化存根和不带偏置的二维卷积层的序列模块
        module = nn.Sequential(
            torch.ao.quantization.QuantStub(),
            nn.Conv2d(
                3,
                3,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                padding_mode="zeros",
            ),
        )
        # 调用图模式下测试操作的辅助方法，验证包含图模式的无偏置二维卷积层的功能，输入大小为[1, 3, 6, 6]，不生成数据
        self._test_op_graph(module, input_size=[1, 3, 6, 6], generate=False)

    # 重写量化引擎后的测试方法
    @override_qengines
    def test_conv2d_graph_v2(self):
        # 测试和 test_conv2d_graph 相同的功能，但针对 ConvPackedParams{n}d 的第二个版本进行测试
        module = nn.Sequential(
            torch.ao.quantization.QuantStub(),  # 添加量化辅助模块
            nn.Conv2d(
                3,  # 输入通道数
                3,  # 输出通道数
                kernel_size=3,  # 卷积核大小为 3x3
                stride=1,  # 步长为 1
                padding=0,  # 填充为 0
                dilation=1,  # 空洞率为 1
                groups=1,  # 分组卷积数量为 1
                bias=True,  # 使用偏置项
                padding_mode="zeros",  # 填充模式为 "zeros"
            ),
        )
        self._test_op_graph(module, input_size=[1, 3, 6, 6], generate=False)  # 调用测试操作图函数，传入模块和输入大小

    @override_qengines
    def test_conv2d_nobias_graph_v2(self):
        # 测试和 test_conv2d_nobias_graph 相同的功能，但针对 ConvPackedParams{n}d 的第二个版本进行测试
        module = nn.Sequential(
            torch.ao.quantization.QuantStub(),  # 添加量化辅助模块
            nn.Conv2d(
                3,  # 输入通道数
                3,  # 输出通道数
                kernel_size=3,  # 卷积核大小为 3x3
                stride=1,  # 步长为 1
                padding=0,  # 填充为 0
                dilation=1,  # 空洞率为 1
                groups=1,  # 分组卷积数量为 1
                bias=False,  # 不使用偏置项
                padding_mode="zeros",  # 填充模式为 "zeros"
            ),
        )
        self._test_op_graph(module, input_size=[1, 3, 6, 6], generate=False)  # 调用测试操作图函数，传入模块和输入大小

    @override_qengines
    def test_conv2d_graph_v3(self):
        # 测试和 test_conv2d_graph 相同的功能，但针对 ConvPackedParams{n}d 的第三个版本进行测试
        module = nn.Sequential(
            torch.ao.quantization.QuantStub(),  # 添加量化辅助模块
            nn.Conv2d(
                3,  # 输入通道数
                3,  # 输出通道数
                kernel_size=3,  # 卷积核大小为 3x3
                stride=1,  # 步长为 1
                padding=0,  # 填充为 0
                dilation=1,  # 空洞率为 1
                groups=1,  # 分组卷积数量为 1
                bias=True,  # 使用偏置项
                padding_mode="zeros",  # 填充模式为 "zeros"
            ),
        )
        self._test_op_graph(module, input_size=[1, 3, 6, 6], generate=False)  # 调用测试操作图函数，传入模块和输入大小

    @override_qengines
    def test_conv2d_nobias_graph_v3(self):
        # 测试和 test_conv2d_nobias_graph 相同的功能，但针对 ConvPackedParams{n}d 的第三个版本进行测试
        module = nn.Sequential(
            torch.ao.quantization.QuantStub(),  # 添加量化辅助模块
            nn.Conv2d(
                3,  # 输入通道数
                3,  # 输出通道数
                kernel_size=3,  # 卷积核大小为 3x3
                stride=1,  # 步长为 1
                padding=0,  # 填充为 0
                dilation=1,  # 空洞率为 1
                groups=1,  # 分组卷积数量为 1
                bias=False,  # 不使用偏置项
                padding_mode="zeros",  # 填充模式为 "zeros"
            ),
        )
        self._test_op_graph(module, input_size=[1, 3, 6, 6], generate=False)  # 调用测试操作图函数，传入模块和输入大小

    @override_qengines
    def test_conv2d_relu(self):
        module = nniq.ConvReLU2d(
            3,  # 输入通道数
            3,  # 输出通道数
            kernel_size=3,  # 卷积核大小为 3x3
            stride=1,  # 步长为 1
            padding=0,  # 填充为 0
            dilation=1,  # 空洞率为 1
            groups=1,  # 分组卷积数量为 1
            bias=True,  # 使用偏置项
            padding_mode="zeros",  # 填充模式为 "zeros"
        )
        self._test_op(module, input_size=[1, 3, 6, 6], generate=False)
        # TODO: graph mode quantized conv2d module
    # 定义一个测试方法，用于测试 Conv3d 操作
    def test_conv3d(self):
        # 如果当前的量化引擎是 fbgemm
        if qengine_is_fbgemm():
            # 创建一个 Conv3d 模块，设置参数如下
            module = nnq.Conv3d(
                3,                      # 输入通道数
                3,                      # 输出通道数
                kernel_size=3,          # 卷积核大小为 3x3x3
                stride=1,               # 步长为 1
                padding=0,              # 填充为 0
                dilation=1,             # 膨胀系数为 1
                groups=1,               # 分组数为 1
                bias=True,              # 启用偏置
                padding_mode="zeros",   # 填充模式为 "zeros"
            )
            # 对创建的 Conv3d 模块进行测试
            self._test_op(module, input_size=[1, 3, 6, 6, 6], generate=False)
            # TODO: graph mode quantized conv3d module

    # 重写量化引擎设置，用于测试带 ReLU 的 Conv3d 操作
    @override_qengines
    def test_conv3d_relu(self):
        # 如果当前的量化引擎是 fbgemm
        if qengine_is_fbgemm():
            # 创建一个带 ReLU 的 Conv3d 模块，设置参数如下
            module = nniq.ConvReLU3d(
                3,                      # 输入通道数
                3,                      # 输出通道数
                kernel_size=3,          # 卷积核大小为 3x3x3
                stride=1,               # 步长为 1
                padding=0,              # 填充为 0
                dilation=1,             # 膨胀系数为 1
                groups=1,               # 分组数为 1
                bias=True,              # 启用偏置
                padding_mode="zeros",   # 填充模式为 "zeros"
            )
            # 对创建的带 ReLU 的 Conv3d 模块进行测试
            self._test_op(module, input_size=[1, 3, 6, 6, 6], generate=False)
            # TODO: graph mode quantized conv3d module

    # 测试 LSTM 模块
    @override_qengines
    @unittest.skipIf(
        IS_AVX512_VNNI_SUPPORTED,
        "This test fails on machines with AVX512_VNNI support. Ref: GH Issue 59098",
    )
    def test_lstm(self):
        # 定义一个 LSTM 模块
        class LSTMModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个 LSTM 层，设置输入大小为 3，隐藏状态大小为 7，层数为 1
                self.lstm = nnqd.LSTM(input_size=3, hidden_size=7, num_layers=1).to(
                    dtype=torch.float
                )

            def forward(self, x):
                # 将输入 x 输入到 LSTM 层中
                x = self.lstm(x)
                return x

        # 如果当前的量化引擎是 fbgemm
        if qengine_is_fbgemm():
            # 创建一个 LSTM 模块
            mod = LSTMModule()
            # 对创建的 LSTM 模块进行测试
            self._test_op(
                mod,
                input_size=[4, 4, 3],
                input_quantized=False,
                generate=False,
                new_zipfile_serialization=True,
            )

    # 测试按通道观察器
    def test_per_channel_observer(self):
        # 创建一个按通道的最小最大值观察器
        obs = PerChannelMinMaxObserver()
        # 对观察器进行测试
        self._test_obs(obs, input_size=[5, 5], generate=False)

    # 测试按张量观察器
    def test_per_tensor_observer(self):
        # 创建一个按张量的最小最大值观察器
        obs = MinMaxObserver()
        # 对观察器进行测试
        self._test_obs(obs, input_size=[5, 5], generate=False)

    # 测试默认的 QAT（量化感知训练）配置
    def test_default_qat_qconfig(self):
        # 定义一个简单的神经网络模型
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.relu = nn.ReLU()

            def forward(self, x):
                # 模型的前向传播：线性层 -> ReLU 激活函数
                x = self.linear(x)
                x = self.relu(x)
                return x

        # 创建一个模型实例
        model = Model()
        # 设置线性层的权重为随机生成的 5x5 权重
        model.linear.weight = torch.nn.Parameter(torch.randn(5, 5))
        # 获取默认的 QAT 配置
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
        # 包装模型以支持量化训练
        ref_model = torch.ao.quantization.QuantWrapper(model)
        ref_model = torch.ao.quantization.prepare_qat(ref_model)
        # 对量化训练后的模型进行测试
        self._test_obs(
            ref_model, input_size=[5, 5], generate=False, check_numerics=False
        )

    # 如果没有 fbgemm，则跳过测试
    @skipIfNoFBGEMM
    # 定义测试方法 test_linear_relu_package_quantization_transforms
    def test_linear_relu_package_quantization_transforms(self):
        # 创建一个 LinearReluFunctional 的实例 m，并将其设置为评估模式
        m = LinearReluFunctional(4).eval()
        # 调用 _test_package 方法进行测试，传入参数 m 作为被测试的模型，
        # input_size=(1, 1, 4, 4) 指定输入大小，generate=False 表示不生成数据而是直接测试
        self._test_package(m, input_size=(1, 1, 4, 4), generate=False)
```
# `.\pytorch\test\test_mobile_optimizer.py`

```py
# Owner(s): ["oncall: mobile"]

# 导入必要的库和模块
import unittest
import torch
import torch.nn as nn
import torch.utils.bundled_inputs
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfNoXNNPACK
from torch.testing._internal.jit_utils import get_forward, get_forward_graph
from torch.utils.mobile_optimizer import (LintCode,
                                          generate_mobile_module_lints,
                                          optimize_for_mobile,
                                          MobileOptimizerType)
from torch.nn import functional as F
from torch.testing._internal.common_quantized import override_quantized_engine

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# 导入文件检查工具
FileCheck = torch._C.FileCheck

# 定义测试类 TestOptimizer，继承自 TestCase
class TestOptimizer(TestCase):

    # 装饰器：如果没有 XNNPACK 支持，则跳过测试
    @skipIfNoXNNPACK
    @skipIfNoXNNPACK
    def test_quantized_conv_no_asan_failures(self):
        # 测试说明：在已经量化的卷积模块上运行 fold_conv_bn 时出现 ASAN 失败。
        # 验证此问题是否仍然存在。

        # 如果当前环境不支持 qnnpack 引擎，则跳过测试
        if 'qnnpack' not in torch.backends.quantized.supported_engines:
            return

        # 定义子模块 Child，包含一个卷积层
        class Child(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv2(x)
                return x

        # 定义父模块 Parent，包含量化和反量化桩以及一个卷积层和子模块 Child
        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.child = Child()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv1(x)
                x = self.child(x)
                x = self.dequant(x)
                return x

        # 使用 qnnpack 引擎运行模型优化的上下文
        with override_quantized_engine('qnnpack'):
            # 创建 Parent 模型实例
            model = Parent()
            # 设置模型的量化配置为 qnnpack
            model.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
            # 在原地准备模型进行量化
            torch.ao.quantization.prepare(model, inplace=True)
            # 将模型应用于随机生成的输入
            model(torch.randn(4, 1, 4, 4))
            # 将模型转换为 Torch 脚本
            torch.ao.quantization.convert(model, inplace=True)
            # 对模型进行移动端优化
            model_optim = optimize_for_mobile(model)
            # 此行代码不应出现 ASAN 失败
            # 返回优化后的模型
            return model_optim
    def test_generate_mobile_module_lints(self):
        # 定义一个测试类 MyTestModule，继承自 torch.nn.Module
        class MyTestModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 添加一个线性层，输入输出都是 4
                self.fc = torch.nn.Linear(4, 4)
                # 添加一个 Dropout 层，丢弃概率为 0.5
                self.dropout = torch.nn.Dropout(p=0.5)

            # 前向传播方法
            def forward(self, inputs):
                # 对输入数据应用线性层
                out = self.fc(inputs)
                # 对输出数据应用 Dropout 层
                out = self.dropout(out)
                return out

        # 定义一个测试类 MyBNModule，继承自 torch.nn.Module
        class MyBNModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 添加一个二维批归一化层，输入通道数为 4，affine 参数为 True
                self.bn = torch.nn.BatchNorm2d(4, affine=True)

            # 前向传播方法
            def forward(self, inputs):
                # 对输入数据应用批归一化层
                bn = self.bn(inputs)
                return bn

        # 定义一个测试类 MyBundledInputModule，继承自 torch.nn.Module
        class MyBundledInputModule(torch.nn.Module):
            # 前向传播方法
            def forward(self, inputs):
                return inputs

        # 定义一个函数，用于根据 lint 类型统计 lint 数量
        def get_lint_count_by_type(lint_type, module_lint_List):
            return len([lint_dict for lint_dict in module_lint_List if lint_dict['name'] == lint_type.name])

        # 使用 torch.jit.script 对 MyTestModule 进行脚本化，并生成移动端 lint 列表
        test_module = torch.jit.script(MyTestModule())
        test_module_lint_list = generate_mobile_module_lints(test_module)
        # 断言移动端 lint 列表的长度为 4
        self.assertEqual(len(test_module_lint_list), 4)
        # 断言 BUNDLED_INPUT 类型的 lint 数量为 1
        self.assertEqual(get_lint_count_by_type(LintCode.BUNDLED_INPUT, test_module_lint_list), 1)
        # 断言 DROPOUT 类型的 lint 数量为 1
        self.assertEqual(get_lint_count_by_type(LintCode.DROPOUT, test_module_lint_list), 1)
        # 断言 REQUIRES_GRAD 类型的 lint 数量为 2
        self.assertEqual(get_lint_count_by_type(LintCode.REQUIRES_GRAD, test_module_lint_list), 2)

        # 使用 torch.jit.script 对 MyBNModule 进行脚本化，并生成移动端 lint 列表
        bn_module = torch.jit.script(MyBNModule())
        bn_module_lint_list = generate_mobile_module_lints(bn_module)
        # 断言移动端 lint 列表的长度为 4
        self.assertEqual(len(bn_module_lint_list), 4)
        # 断言 BUNDLED_INPUT 类型的 lint 数量为 1
        self.assertEqual(get_lint_count_by_type(LintCode.BUNDLED_INPUT, bn_module_lint_list), 1)
        # 断言 BATCHNORM 类型的 lint 数量为 1
        self.assertEqual(get_lint_count_by_type(LintCode.BATCHNORM, bn_module_lint_list), 1)
        # 断言 REQUIRES_GRAD 类型的 lint 数量为 2
        self.assertEqual(get_lint_count_by_type(LintCode.REQUIRES_GRAD, bn_module_lint_list), 2)

        # 使用 torch.jit.script 对 MyBundledInputModule 进行脚本化
        bi_module = torch.jit.script(MyBundledInputModule())
        # 使用 torch.utils.bundled_inputs.augment_model_with_bundled_inputs 方法扩展模型 bi_module
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
            bi_module, [(torch.tensor([1]),)], [])
        # 生成移动端 lint 列表
        bi_module_lint_list = generate_mobile_module_lints(bi_module)
        # 断言移动端 lint 列表的长度为 0
        self.assertEqual(len(bi_module_lint_list), 0)

    @skipIfNoXNNPACK
    # 定义测试函数，验证在优化为移动端时是否保留捆绑输入方法
    def test_preserve_bundled_inputs_methods(self):
        # 定义继承自 torch.nn.Module 的类 MyBundledInputModule
        class MyBundledInputModule(torch.nn.Module):
            # 定义前向传播方法，直接返回输入
            def forward(self, inputs):
                return inputs

        # 定义继承自 torch.nn.Module 的类 MyIncompleteBundledInputModule
        class MyIncompleteBundledInputModule(torch.nn.Module):
            # 定义前向传播方法，直接返回输入
            def forward(self, inputs):
                return inputs

            # 使用 torch.jit.export 标记的方法，但未实现其具体功能
            @torch.jit.export
            def get_all_bundled_inputs(self):
                pass

        # 使用 torch.jit.script 将 MyBundledInputModule 实例化为 bi_module
        bi_module = torch.jit.script(MyBundledInputModule())
        # 对 bi_module 进行优化，返回优化后的模块
        module_optim_bi_not_preserved = optimize_for_mobile(bi_module)

        # 断言优化后的模块没有添加捆绑输入方法
        self.assertFalse(
            hasattr(module_optim_bi_not_preserved, 'get_all_bundled_inputs') or
            hasattr(module_optim_bi_not_preserved, 'get_num_bundled_inputs')
        )

        # 将捆绑输入方法添加到 bi_module 中
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
            bi_module, [(torch.tensor([1]),)], [])
        # 重新优化 bi_module，确保捆绑输入方法被保留
        module_optim_bi_preserved = optimize_for_mobile(bi_module)

        # 断言优化后的模块保留了所有捆绑输入方法
        self.assertTrue(
            hasattr(module_optim_bi_preserved, 'get_all_bundled_inputs') and
            hasattr(module_optim_bi_preserved, 'get_num_bundled_inputs')
        )

        # 获取第一个捆绑输入，并调用优化后的模块
        bundled_input = module_optim_bi_preserved.get_all_bundled_inputs()[0]
        module_optim_bi_preserved(*bundled_input)

        # 如果模块中没有全部三个捆绑输入方法，
        # 在用户未指定的情况下，我们将不会尝试保留它们。
        incomplete_bi_module = torch.jit.script(MyIncompleteBundledInputModule())
        incomplete_bi_module_optim = optimize_for_mobile(incomplete_bi_module)
        self.assertFalse(hasattr(incomplete_bi_module_optim, 'get_all_bundled_inputs'))

        # 明确指定保留 get_all_bundled_inputs 方法，即使它是唯一的捆绑输入方法。
        incomplete_bi_module_optim = optimize_for_mobile(incomplete_bi_module, preserved_methods=['get_all_bundled_inputs'])
        self.assertTrue(hasattr(incomplete_bi_module_optim, 'get_all_bundled_inputs'))

    # 根据 XNNPACK 的可用性决定是否跳过测试
    @skipIfNoXNNPACK
    @skipIfNoXNNPACK
    # 只有在 HAS_TORCHVISION 为真时才运行测试，因为需要 torchvision。
    @unittest.skipUnless(HAS_TORCHVISION, "Needs torchvision")
    # 测试 MobileNet 在优化为移动端后的表现
    def test_mobilenet_optimize_for_mobile(self):
        # 创建 MobileNet v3 small 模型实例 m
        m = torchvision.models.mobilenet_v3_small()
        # 使用 torch.jit.script 将模型 m 转换为 TorchScript
        m = torch.jit.script(m)
        # 优化模型 m 为移动端模型
        m = optimize_for_mobile(m)

        # 创建输入张量 x，进行三次前向传播
        x = torch.zeros(1, 3, 56, 56)
        self.assertEqual(m(x).numel(), 1000)
        self.assertEqual(m(x).numel(), 1000)
        self.assertEqual(m(x).numel(), 1000)
# 如果当前脚本作为主程序执行（而不是被导入为模块），则执行 run_tests() 函数
if __name__ == '__main__':
    run_tests()
```
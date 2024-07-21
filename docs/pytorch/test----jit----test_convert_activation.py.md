# `.\pytorch\test\jit\test_convert_activation.py`

```py
# Owner(s): ["oncall: jit"]

import os  # 导入操作系统相关的模块
import sys  # 导入系统相关的模块
import unittest  # 导入单元测试框架

from itertools import product  # 导入用于迭代器操作的product函数

import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的函数接口模块
from torch.testing import FileCheck  # 导入用于测试的文件检查工具

try:
    import torchvision  # 尝试导入torchvision

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")  # 如果没有torchvision，则跳过测试

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)  # 将测试文件目录添加到系统路径中
from torch.testing._internal.jit_utils import JitTestCase  # 导入用于JIT测试的工具类

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

activations = [  # 定义多个激活函数
    F.celu,
    F.elu,
    F.hardsigmoid,
    F.hardswish,
    F.hardtanh,
    F.leaky_relu,
    F.relu,
    F.relu6,
    F.rrelu,
    F.selu,
    F.silu,
]


class TestFunctionalToInplaceActivation(JitTestCase):
    def test_check_no_type_promotion(self):
        dtypes = [  # 定义多个数据类型
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float32,
            torch.float64,
        ]
        # restore_mutation.h contains a mapping from activation operators
        # to whether they allow type conversion. Use this checking to
        # guard the mapping, and if any later change breaks the assumption
        # we need to update the mapping correspondingly.
        for activation, dtype in product(activations, dtypes):  # 使用product生成激活函数和数据类型的组合
            inp = torch.normal(0, 5, size=(4, 4)).to(dtype)  # 生成指定数据类型的随机输入
            try:
                out = activation(inp)  # 对输入应用激活函数
                self.assertEqual(dtype, out.dtype)  # 断言输出的数据类型与输入一致
            except RuntimeError:
                # Skip the not implemented error
                pass

    def test_functional_to_inplace_activation(self):
        for activation in activations:  # 遍历每个激活函数

            def test_basic(x):  # 定义测试函数
                y = x + 1  # 加法操作
                z = activation(y)  # 应用激活函数
                return z

            fn = torch.jit.script(test_basic)  # 使用JIT对测试函数进行脚本化编译
            self.run_pass("inline", fn.graph)  # 在编译图中运行"inline"优化
            self.run_pass("constant_propagation", fn.graph)  # 在编译图中运行"constant_propagation"优化
            FileCheck().check(f"aten::{activation.__name__}(").run(fn.graph)  # 检查图中是否包含特定激活函数的操作
            self.run_pass("functional_to_inplace_activation", fn.graph)  # 在编译图中运行"functional_to_inplace_activation"优化
            FileCheck().check_not(f"aten::{activation.__name__}(").run(fn.graph)  # 检查图中是否不再包含原始的激活函数操作
            FileCheck().check(f"aten::{activation.__name__}_").run(fn.graph)  # 检查图中是否添加了对应的原地激活函数操作
            inp = torch.rand([2, 2])  # 生成随机输入
            self.assertEqual(fn(inp), test_basic(inp))  # 断言编译后的函数与原始测试函数在相同输入下的输出一致
    def test_no_functional_to_inplace(self):
        # inplace conversion should not happen because sigmoid may
        # perform type conversion
        def test1():
            # 创建一个2x2的全1张量
            y = torch.ones([2, 2])
            # 对张量y应用sigmoid函数
            z = torch.sigmoid(y)
            return z

        # 将test1函数转换为Torch脚本
        fn = torch.jit.script(test1)
        # 在fn的计算图上运行"functional_to_inplace_activation"传递
        self.run_pass("functional_to_inplace_activation", fn.graph)
        # 检查计算图中是否不包含"inplace"版本的sigmoid操作符
        FileCheck().check_not("aten::sigmoid_").run(fn.graph)

        # inplace conversion should not happen because y is alias
        # the input x
        def test2(x):
            # 获取输入张量x的第一个元素，并赋值给y（此处y是x的引用）
            y = x[0]
            # 对张量y应用ReLU函数
            z = torch.relu(y)
            return z

        # 将test2函数转换为Torch脚本
        fn = torch.jit.script(test2)
        # 在fn的计算图上运行"functional_to_inplace_activation"传递
        self.run_pass("functional_to_inplace_activation", fn.graph)
        # 检查计算图中是否不包含"inplace"版本的ReLU操作符
        FileCheck().check_not("aten::relu_").run(fn.graph)

        # inplace conversion should not happen because self.x is
        # at the global scope
        # 定义一个继承自nn.Module的Test3类
        class Test3(nn.Module):
            def __init__(self, x):
                super().__init__()
                self.x = x

            def forward(self):
                # 对self.x应用ReLU函数
                y = torch.relu(self.x)
                return y

        # 创建Test3类的实例，并转换为Torch脚本
        fn = torch.jit.script(Test3(torch.rand([2, 2])).eval())
        # 在fn的计算图上运行"functional_to_inplace_activation"传递
        self.run_pass("functional_to_inplace_activation", fn.graph)
        # 检查计算图中是否不包含"inplace"版本的ReLU操作符
        FileCheck().check_not("aten::relu_").run(fn.graph)

    @skipIfNoTorchVision
    def test_resnet18_correctness(self):
        # 创建一个ResNet-18模型
        model = torchvision.models.resnet18()
        # 将ResNet-18模型转换为静态图，并冻结其权重
        frozen_model = torch.jit.freeze(torch.jit.script(model.eval()))
        (
            N,
            C,
            H,
            W,
        ) = (
            10,
            3,
            224,
            224,
        )
        # 创建一个形状为[N, C, H, W]的随机输入张量
        inp = torch.randn(N, C, H, W)
        # 在frozen_model的计算图上运行"functional_to_inplace_activation"传递
        self.run_pass("functional_to_inplace_activation", frozen_model.graph)
        # 断言原始模型和冻结模型在相同输入上产生相同的输出
        self.assertEqual(model(inp), frozen_model(inp))
# 定义一个名为 TestInplaceToFunctionalActivation 的测试类，继承自 JitTestCase
class TestInplaceToFunctionalActivation(JitTestCase):

    # 定义一个测试方法 test_inplace_to_functional_activation
    def test_inplace_to_functional_activation(self):
        
        # 遍历 activations 列表中的每个激活函数 activation
        for activation in activations:
            
            # 定义一个内部测试函数 test_basic，接受参数 x
            def test_basic(x):
                # 计算 y = x + 1
                y = x + 1
                # 调用 activation 函数，将其作用于 y，并使用 inplace 模式
                activation(y, inplace=True)
                # 返回 y
                return y
            
            # 对 test_basic 函数进行 Torch JIT 脚本化
            fn = torch.jit.script(test_basic)
            
            # 在 JIT 图上运行内联传递优化
            self.run_pass("inline", fn.graph)
            
            # 在 JIT 图上运行常量传播优化
            self.run_pass("constant_propagation", fn.graph)
            
            # 使用 FileCheck 验证在 JIT 图中是否存在特定的 aten::<activation_name>_ 操作
            FileCheck().check(f"aten::{activation.__name__}_").run(fn.graph)
            
            # 在 JIT 图上运行原地操作转换为函数式激活函数的优化
            self.run_pass("inplace_to_functional_activation", fn.graph)
            
            # 使用 FileCheck 验证在 JIT 图中是否不再存在原地操作后的激活函数调用
            FileCheck().check_not(f"aten::{activation.__name__}_").run(fn.graph)
            
            # 使用 FileCheck 验证在 JIT 图中是否存在 aten::<activation_name>( 形式的函数式激活函数调用
            FileCheck().check(f"aten::{activation.__name__}(").run(fn.graph)
        
        # 遍历包含 torch.relu_, torch.sigmoid_, torch.tanh_ 的列表
        for activation in [
            torch.relu_,
            torch.sigmoid_,
            torch.tanh_,
        ]:
            
            # 定义一个内部测试函数 test_basic，接受参数 x
            def test_basic(x):
                # 计算 y = x + 1
                y = x + 1
                # 调用 activation 函数，作用于 y
                activation(y)
                # 返回 y
                return y
            
            # 对 test_basic 函数进行 Torch JIT 脚本化
            fn = torch.jit.script(test_basic)
            
            # 在 JIT 图上运行内联传递优化
            self.run_pass("inline", fn.graph)
            
            # 在 JIT 图上运行常量传播优化
            self.run_pass("constant_propagation", fn.graph)
            
            # 使用 FileCheck 验证在 JIT 图中是否存在特定的 aten::<activation_name> 操作
            FileCheck().check(f"aten::{activation.__name__}").run(fn.graph)
            
            # 在 JIT 图上运行原地操作转换为函数式激活函数的优化
            self.run_pass("inplace_to_functional_activation", fn.graph)
            
            # 使用 FileCheck 验证在 JIT 图中是否不再存在原地操作后的激活函数调用
            FileCheck().check_not(f"aten::{activation.__name__}").run(fn.graph)
            
            # 使用 FileCheck 验证在 JIT 图中是否存在 aten::<activation_name>( 形式的函数式激活函数调用
            FileCheck().check(f"aten::{activation.__name__[:-1]}(").run(fn.graph)

            # 创建一个形状为 [2, 2] 的随机输入张量 inp
            inp = torch.rand([2, 2])
            
            # 使用断言验证 JIT 脚本化函数 fn 对输入 inp 的输出与直接调用 test_basic 对 inp 的输出相等
            self.assertEqual(fn(inp), test_basic(inp))

    # 使用 skipIfNoTorchVision 装饰器标记的测试方法 test_resnet18_correctness
    @skipIfNoTorchVision
    def test_resnet18_correctness(self):
        # 创建一个 torchvision 中的 ResNet-18 模型
        model = torchvision.models.resnet18()
        
        # 对模型进行 Torch JIT 脚本化，并使用 freeze 方法冻结模型
        frozen_model = torch.jit.freeze(torch.jit.script(model.eval()))
        
        # 定义输入张量的形状 [N, C, H, W] 并生成随机输入 inp
        (N, C, H, W) = (10, 3, 224, 224)
        inp = torch.randn(N, C, H, W)
        
        # 在冻结的模型图上运行原地操作转换为函数式激活函数的优化
        self.run_pass("inplace_to_functional_activation", frozen_model.graph)
        
        # 使用断言验证原始模型和冻结模型对相同输入 inp 的输出是否相等
        self.assertEqual(model(inp), frozen_model(inp))
```
# `.\pytorch\test\onnx\test_fx_to_onnx_with_onnxruntime.py`

```py
# Owner(s): ["module: onnx"]
from __future__ import annotations

import itertools  # 导入 itertools 模块，用于迭代操作
import math  # 导入 math 模块，提供数学运算函数
import operator  # 导入 operator 模块，提供基本操作符的函数
import os  # 导入 os 模块，提供与操作系统交互的功能
import tempfile  # 导入 tempfile 模块，用于创建临时文件和目录
import unittest  # 导入 unittest 模块，支持单元测试框架

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type  # 导入类型提示相关的模块

import onnx_test_common  # 导入自定义的 ONNX 测试通用模块
import onnxruntime  # type: ignore[import] 导入 ONNX 运行时模块，忽略类型检查
import parameterized  # type: ignore[import] 导入 parameterized 模块，用于参数化测试
import pytorch_test_common  # 导入自定义的 PyTorch 测试通用模块

import transformers  # type: ignore[import] 导入 transformers 模块，忽略类型检查

import torch  # 导入 PyTorch 模块
import torch.onnx  # 导入 torch.onnx 模块，用于 ONNX 导出
from torch import nn  # 导入 nn 模块，神经网络相关的类和函数

from torch._subclasses import fake_tensor  # 导入 fake_tensor 类型
from torch.onnx._internal import _beartype, exporter  # 导入 ONNX 内部模块
from torch.onnx._internal.fx import (  # 导入 ONNX FX 相关模块
    diagnostics,
    fx_symbolic_graph_extractor,
    patcher,
    serialization as fx_serialization,
)
from torch.testing._internal import common_utils  # 导入内部的测试工具函数

try:
    import torchvision  # type: ignore[import] 导入 torchvision 模块，忽略类型检查

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
except RuntimeError:
    HAS_TORCHVISION = False
skip_if_no_torchvision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


def _parameterized_class_attrs_and_values():
    input_values = []
    input_values.extend(
        itertools.product(
            (True, False),  # 生成布尔值 True 和 False 的组合
            (True, False),  # 生成布尔值 True 和 False 的组合
            (
                pytorch_test_common.TorchModelType.TORCH_NN_MODULE,  # 使用 PyTorch 测试通用模块中的枚举类型
                pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,  # 使用 PyTorch 测试通用模块中的枚举类型
            ),
        )
    )
    return {
        "attrs": ["op_level_debug", "dynamic_shapes", "model_type"],  # 返回参数化类的属性列表
        "input_values": input_values,  # 返回参数化类的输入值列表
    }


def _parameterize_class_name(cls: Type, idx: int, input_dicts: Mapping[Any, Any]):
    """Combine class name with the parameterized arguments.

    This function is passed to `parameterized.parameterized_class` as the
    `class_name_func` argument.
    """
    suffixes = []
    for k, v in input_dicts.items():
        suffixes.append(f"{k}_{v}")  # 构建基于输入字典的后缀列表
    return f"{cls.__name__}_{'_'.join(suffixes)}"  # 返回格式化后的参数化类名称


@parameterized.parameterized_class(
    **_parameterized_class_attrs_and_values(),  # 使用预定义的参数化类属性和值
    class_name_func=_parameterize_class_name,  # 指定类名格式化函数
)
class TestFxToOnnxWithOnnxRuntime(onnx_test_common._TestONNXRuntime):
    op_level_debug: bool  # 测试用例的操作级别调试标志
    dynamic_shapes: bool  # 测试用例的动态形状标志
    model_type: pytorch_test_common.TorchModelType  # 测试用例的模型类型

    def setUp(self):
        super().setUp()  # 调用父类的 setUp 方法
        self.ort_version = onnxruntime.__version__  # 设置 ONNX 运行时版本信息

    def test_simple_function(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                # TODO(justinchuby): Replicate torch's type casting policy
                # in the exporter for type promotion support
                y = x + 1.0  # 执行张量 x 的加法操作
                z = y.relu()  # 执行张量 y 的 ReLU 激活操作
                return (y, z)  # 返回操作后的张量结果

        func = Foo()  # 创建 Foo 类的实例对象

        tensor_x = torch.randn(1, 1, 2, dtype=torch.float32)  # 创建指定形状和数据类型的随机张量

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (tensor_x,))  # 运行基于 FX 到 ONNX 导出器和 ONNX 运行时的测试

    @pytorch_test_common.xfail(
        error_message="Unexpectedly found a <class 'torch.Tensor'> in the inputs.",  # 失败时的错误消息
        reason="https://github.com/pytorch/pytorch/issues/96379",  # 失败的原因说明
    )
    def test_func_with_args_and_tensor_kwargs(self):
        # 这是一个测试函数，用于测试带有位置参数和张量关键字参数的情况

        # 在 Dynamo 追踪的图中，非张量的可选关键字参数会被折叠成常量，并且如果未提供值给追踪器，将从输入列表中移除。
        # 因此，对于像这样的函数：
        #   def func(x, b=1.0)
        # 例如，如果首先使用参数 (x,) 进行 Dynamo 追踪模型，然后使用参数 (x, b=2.0) 调用追踪的图形，
        # 它将会在某处投诉模型被调用带有额外的参数，因为修改后的函数被追踪为：
        #   def forward(self, x : torch.Tensor):
        #     add = x + 1.0;  x = None
        #     relu = add.relu()
        #     return (add, relu)
        # 总结一下，为了被追踪为图形输入，可选关键字参数的值必须提供。否则，它们在 Dynamo 中被视为图内常量。
        # 张量可选关键字参数是一个例外。它总是作为输入被追踪。
        # 不清楚这种行为是否有意为之。但总的来说，设置可变默认值是一个不好的做法。

        # `DynamoOptimizeExporter` 通过将参数和关键字参数绑定到模型签名，并填充未提供的可选参数的默认值，来应用一种解决方法。
        class Foo(torch.nn.Module):
            def forward(self, x, b=torch.tensor(1.0)):
                y = x + b
                z = y.relu()
                return (y, z)

        func = Foo()

        tensor_x = torch.randn(1, 2, 3, dtype=torch.float32)

        # 测试没有提供可选关键字参数的情况。
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (tensor_x,))
        # 测试只提供位置参数的情况。
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func, (tensor_x, torch.tensor(8.0))
        )
        # 测试同时指定可选关键字参数的情况。
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func, (tensor_x,), input_kwargs={"b": torch.tensor(5.0)}
        )

    @pytorch_test_common.skip_dynamic_fx_test(
        "sympy operation tests don't need dynamic shape"
    )
    # 定义一个名为 test_sympy_operatons_return_numeric 的测试方法
    def test_sympy_operatons_return_numeric(self):
        # 定义一个名为 Foo 的内部类，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x, y):
                # TODO: 当 SymBool 支持时，添加布尔测试来推断类型
                # 以返回四则运算的结果张量
                return (
                    torch.tensor([operator.add(x.item(), y.item())]),  # 加法操作
                    torch.tensor([operator.sub(x.item(), y.item())]),  # 减法操作
                    torch.tensor([operator.mul(x.item(), y.item())]),  # 乘法操作
                    torch.tensor([operator.truediv(x.item(), y.item())]),  # 真除法操作
                    # 下面的操作需要 torch.sym_float，可能容易降低到 ONNX 格式，
                    # 但我不知道应该放在哪里
                    # torch.tensor([operator.floordiv(x.item(), y.item())]),
                    # 注意：取绝对值以确保底数和指数均为非负数，避免生成运行时断言
                    torch.tensor([operator.pow(abs(x.item()), abs(y.item()))]),  # 幂运算操作
                    torch.tensor([operator.abs(x.item())]),  # 绝对值操作
                    torch.tensor([operator.neg(x.item())]),  # 取负操作
                    torch.tensor([math.ceil(x.item())]),  # 向上取整操作
                    torch.tensor([math.floor(x.item())]),  # 向下取整操作
                )

        # 创建 Foo 类的实例对象
        func = Foo()

        # 生成随机张量 x 和 y，数据类型为 torch.float32
        x = torch.randn(1, dtype=torch.float32)
        y = torch.randn(1, dtype=torch.float32)

        # 调用 self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime 方法，
        # 将 func 和 (x, y) 作为参数传入
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func,
            (
                x,
                y,
            ),
        )

    # 标记测试为预期失败，指定错误信息和原因链接
    @pytorch_test_common.xfail(
        error_message="Model inputs incompatible with the format that was exported",
        reason="https://github.com/pytorch/pytorch/issues/99534",
    )
    # 定义一个测试函数，用于测试在非张量参数下的预期失败情况
    def test_xfail_func_with_non_tensor_args(self):
        # 定义一个继承自torch.nn.Module的类Foo
        class Foo(torch.nn.Module):
            # 定义模型的前向传播方法，接受输入x和可选参数b，默认为1.0
            def forward(self, x, b=1.0):
                # 计算y，即输入x与参数b的和
                y = x + b
                # 对y应用ReLU激活函数，将非正值置零
                z = y.relu()
                # 返回y和z作为输出
                return (y, z)

        # 创建Foo类的实例func
        func = Foo()

        # 生成一个形状为(1, 1, 2)的随机张量tensor_x，数据类型为float32
        tensor_x = torch.randn(1, 1, 2, dtype=torch.float32)

        # 使用torch.onnx.dynamo_export将func模型导出为ONNX程序，传入tensor_x作为输入，
        # 8.0作为参数b，并设置导出选项
        onnx_program = torch.onnx.dynamo_export(
            func,
            tensor_x,
            8.0,
            export_options=torch.onnx.ExportOptions(
                op_level_debug=self.op_level_debug,
                dynamic_shapes=self.dynamic_shapes,
            ),
        )

        # 调用onnx_test_common.assert_dynamic_shapes验证ONNX程序的动态形状
        onnx_test_common.assert_dynamic_shapes(onnx_program, self.dynamic_shapes)

        # 使用onnx_program.adapt_torch_inputs_to_onnx将tensor_x适配为ONNX格式的输入，
        # b设置为8.0，并获得参考输出ref_outputs
        onnx_format_args = onnx_program.adapt_torch_inputs_to_onnx(tensor_x, b=8.0)
        ref_outputs = onnx_program.adapt_torch_outputs_to_onnx(func(tensor_x, 8.0))

        # 使用onnx_test_common.run_ort在ONNX程序上执行ORT（ONNX Runtime），获取ORT的输出ort_outputs
        ort_outputs = onnx_test_common.run_ort(onnx_program, onnx_format_args)

        # 对每对参考输出ref_output和ORT输出ort_output进行逐一断言，要求它们非常接近
        for ref_output, ort_output in zip(ref_outputs, ort_outputs):
            torch.testing.assert_close(ref_output, torch.tensor(ort_output))

        # 测试不同的非张量输入情况，这里预期测试失败（xfail）
        onnx_format_args = onnx_program.adapt_torch_inputs_to_onnx(tensor_x, b=9.0)
        ref_outputs = onnx_program.adapt_torch_outputs_to_onnx(func(tensor_x, 9.0))
        _ = onnx_test_common.run_ort(onnx_program, onnx_format_args)

        # 对每对参考输出ref_output和ORT输出ort_output进行逐一断言，要求它们非常接近
        for ref_output, ort_output in zip(ref_outputs, ort_outputs):
            torch.testing.assert_close(ref_output, torch.tensor(ort_output))

    # 定义一个测试函数，用于测试具有嵌套输入结构的功能
    def test_func_with_nested_input_structure(self):
        # 定义一个继承自torch.nn.Module的类Foo
        class Foo(torch.nn.Module):
            # 定义模型的前向传播方法，接受字典x_dict、元组y_tuple和列表z_list作为输入
            def forward(
                self,
                x_dict: Dict[str, torch.Tensor],
                y_tuple: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                z_list: List[List[torch.Tensor]],
            ):
                # 根据x_dict中的键值，选择或者生成一个随机张量x
                if "a" in x_dict:
                    x = x_dict["a"]
                elif "b" in x_dict:
                    x = x_dict["b"]
                else:
                    x = torch.randn(3)

                # 解包y_tuple中的元组，获取y1、y2和y3
                y1, (y2, y3) = y_tuple

                # 计算z，将x、y1、y2和y3相加
                z = x + y1 + y2 + y3

                # 对z_list中的每个子列表z_sub_list，将其张量堆叠并求和，累加到z中
                for z_sub_list in z_list:
                    z = z + torch.stack(z_sub_list).sum()

                # 返回最终的z作为输出
                return z

        # 创建Foo类的实例func
        func = Foo()

        # 构建示例输入数据：x_dict是包含键"a"和"c"的字典，分别对应随机张量；y_tuple包含
        # 一个随机张量y1和一个元组，其中包含两个随机张量y2和y3；z_list是一个包含两个子列表的列表，
        # 每个子列表包含随机张量
        x_dict = {"a": torch.randn(3), "c": torch.randn(3)}
        y_tuple = (torch.randn(3), (torch.randn(3), torch.randn(3)))
        z_list = [
            [torch.randn(3), torch.randn(3)],
            [torch.randn(3), torch.randn(3), torch.randn(3)],
        ]

        # 调用self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime方法，
        # 将func和构建的输入(x_dict, y_tuple, z_list)作为参数传入，执行测试
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func, (x_dict, y_tuple, z_list)
        )
    def test_func_with_nested_output_structure(self):
        # 定义一个内部类 Foo，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 重写 forward 方法，定义模型的前向传播逻辑
            def forward(self, x, y, z):
                # 计算 x 和 y 的和
                x = x + y
                # 计算 y 和 z 的和
                y = y + z
                # 计算新的 z，为 x 和 y 的和
                z = x + y
                # 构建输出 out1，包含元组 (x, (y, z))
                out1 = (x, (y, z))
                # 构建输出 out2，包含列表 [[x, y], [y, z]]
                out2 = [[x, y], [y, z]]
                # 构建输出 out3，包含字典 {"z": z, "x": x}
                out3 = {"z": z, "x": x}
                # 返回三个输出作为结果
                return out1, out2, out3

        # 创建 Foo 类的实例 func
        func = Foo()

        # 生成随机张量 x、y、z
        x = torch.randn(3)
        y = torch.randn(3)
        z = torch.randn(3)

        # 调用自定义方法运行测试，将 func 和 (x, y, z) 作为参数传递
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (x, y, z))

    def test_mnist(self):
        # 定义一个神经网络模型 MNISTModel，继承自 nn.Module
        class MNISTModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义卷积层 conv1，输入通道 1，输出通道 32，卷积核大小 3x3
                self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=True)
                # 定义卷积层 conv2，输入通道 32，输出通道 64，卷积核大小 3x3
                self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=True)
                # 定义全连接层 fc1，输入特征数 9216，输出特征数 128
                self.fc1 = nn.Linear(9216, 128, bias=True)
                # 定义全连接层 fc2，输入特征数 128，输出特征数 10
                self.fc2 = nn.Linear(128, 10, bias=True)

            # 定义模型的前向传播逻辑，接受 tensor_x 作为输入
            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.conv1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.conv2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = torch.max_pool2d(tensor_x, 2)
                tensor_x = torch.flatten(tensor_x, 1)
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                output = torch.log_softmax(tensor_x, dim=1)
                # 返回输出结果
                return output

        # 创建随机张量 tensor_x，模拟输入数据
        tensor_x = torch.rand((64, 1, 28, 28), dtype=torch.float32)
        # 调用自定义方法运行测试，将 MNISTModel 实例和 tensor_x 作为参数传递
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            MNISTModel(), (tensor_x,)
        )

    def test_log_sigmoid(self):
        # 定义一个神经网络模型 Model，继承自 torch.nn.Module
        # 在构造函数中初始化一个 torch.nn.LogSigmoid 模块
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = torch.nn.LogSigmoid()

            # 定义模型的前向传播逻辑，接受 x 作为输入
            def forward(self, x):
                # 使用 LogSigmoid 模块处理输入 x，并返回处理后的结果
                return self.m(x)

        # 创建随机输入张量 input
        input = torch.randn(2)
        # 调用自定义方法运行测试，将 Model 类的实例和 input 作为参数传递
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(Model(), (input,))

    @skip_if_no_torchvision
    def test_resnet18(self):
        # TODO(bowbao): Note [training vs eval in dynamo_export]
        # We are preparing to export models in evaluation mode for this export.
        # This test specifically addresses issues with 'functionalization' during training.
        # `model.eval()` is explicitly called for models containing batch normalization layers.
        # Ref: https://github.com/pytorch/pytorch/issues/99662#issuecomment-1528178221
        
        # Instantiate ResNet-18 model without pretrained weights and switch to evaluation mode
        model = torchvision.models.resnet18(weights=None).eval()
        # Create a dummy input tensor of shape (1, 3, 224, 224)
        dummy_input = torch.randn(1, 3, 224, 224)

        # Call a test function to export the model to ONNX and run tests
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            model,
            (dummy_input,),
        )

    @pytorch_test_common.xfail_dynamic_fx_test(
        error_message="[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions for input"
    )
    @skip_if_no_torchvision
    def test_shufflenet_v2(self):
        # TODO(bowbao): see Note [training vs eval in dynamo_export]
        # Instantiate ShuffleNet V2 model (x0.5 variant) without pretrained weights and in eval mode
        model = torchvision.models.shufflenet_v2_x0_5(weights=None).eval()
        # Create dummy input tensors of shapes (1, 3, 224, 224) and (3, 3, 224, 224)
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
        test_inputs = torch.randn(3, 3, 224, 224, requires_grad=False)

        # Call a test function to export the model to ONNX and run tests with additional inputs
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            model,
            (dummy_input,),
            additional_test_inputs=[((test_inputs,),)],
            rtol=1e-3,
            atol=1e-5,
        )

    def test_add(self):
        # Define a dynamic module that adds two tensors using torch.ops.aten.add
        class DynamicAdd(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.add(x, y)

        # Create random input tensors of shapes (2, 3), (3, 4), and (2, 3)
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        another_x = torch.randn(3, 4)
        another_y = torch.randn(3, 4)

        # Call a test function to export the DynamicAdd module to ONNX and run tests
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            DynamicAdd(),
            (x, y),
            additional_test_inputs=[((another_x, another_y),)],
        )

    def test_sigmoid_add(self):
        # Define a dynamic module that adds two tensors and applies sigmoid activation
        class DynamicAdd(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x, y):
                z = torch.ops.aten.add(x, y)
                return self.sigmoid(z)

        # Create random input tensors of shapes (2, 3), (1, 4), and (1, 4)
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        x = x[1:, :]
        y = y[1:, :]
        input_x = torch.randn(1, 4)
        input_y = torch.randn(1, 4)

        # Call a test function to export the DynamicAdd module to ONNX and run tests
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            DynamicAdd(), (x, y), additional_test_inputs=[((input_x, input_y),)]
        )
    # 定义一个测试方法，用于测试矩阵乘法操作
    def test_matmul(self):
        # 定义一个动态矩阵乘法的 Torch 模块
        class DynamicMatMul(torch.nn.Module):
            # 实现前向传播方法，执行矩阵乘法操作
            def forward(self, x, y):
                return torch.ops.aten.matmul(x, y)

        # 创建两个随机张量 x 和 y，形状分别为 (2, 3, 6) 和 (2, 6, 4)
        x = torch.randn(2, 3, 6)
        y = torch.randn(2, 6, 4)
        # 创建额外的输入张量 input_x 和 input_y，形状分别为 (2, 3, 4) 和 (2, 4, 4)
        input_x = torch.randn(2, 3, 4)
        input_y = torch.randn(2, 4, 4)

        # 运行测试，将 DynamicMatMul 实例作为模块，输入为 (x, y)，额外测试输入为 ((input_x, input_y),)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            DynamicMatMul(), (x, y), additional_test_inputs=[((input_x, input_y),)]
        )

    # 标记为预期失败的动态 FX 测试，报错信息为指定的形状不匹配错误
    @pytorch_test_common.xfail_dynamic_fx_test(
        error_message="The values for attribute 'shape' do not match: torch.Size([]) != torch.Size([1])"
    )
    # 测试标量张量的方法
    def test_scalar_tensor(self):
        # 定义一个测试模块，实现前向传播方法返回两个标量张量
        class test(torch.nn.Module):
            def forward(self, x):
                return torch.scalar_tensor(x.size(0)), torch.scalar_tensor(
                    x.size(1), dtype=torch.int64
                )

        # 创建随机张量 x 和 y，形状分别为 (2, 3, 4) 和 (7, 8, 9)
        x = torch.randn(2, 3, 4)
        y = torch.randn(7, 8, 9)
        # 运行测试，将 test 实例作为模块，输入为 (x,)，额外测试输入为 ((y,),)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            test(),
            (x,),
            additional_test_inputs=[((y,),)],
        )

    # 定义一个测试方法，用于测试转置操作推断形状
    def test_transpose_infer_shape(self):
        # 定义一个包含转置和卷积操作的 Torch 模块
        class TransposeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 3, stride=2)

            # 实现前向传播方法，执行卷积操作后进行转置
            def forward(self, x):
                x = self.conv(x)
                return x.transpose(0, 1)

        # 创建随机张量 x 和 y，形状分别为 (32, 3, 64, 64) 和 (16, 3, 8, 64)
        x = torch.randn(32, 3, 64, 64)
        y = torch.randn(16, 3, 8, 64)
        # 运行测试，将 TransposeModule 实例作为模块，输入为 (x,)，额外测试输入为 ((y,),)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            TransposeModule(),
            (x,),
            additional_test_inputs=[((y,),)],
        )

    # 标记为预期失败的测试，报错信息为包含不支持的 FX 节点调用
    @pytorch_test_common.xfail(
        error_message=("Unsupported FX nodes: {'call_function': [")
    )
    # 测试 squeeze 操作的运行时维度
    def test_squeeze_runtime_dim(self):
        # 定义一个进行 squeeze 操作的 Torch 模块
        class Squeeze(torch.nn.Module):
            def forward(self, d1, d2):
                # 创建一个全零张量 t，形状由输入 d1 和 d2 决定
                t = torch.zeros(d1[0], d2[0])  # problematic user code for dynamo
                # 执行 squeeze 操作，去除维度 0
                return t.squeeze(0)

        # 创建多个 tensor 输入 d1, d3, d4，并运行测试
        d1 = torch.tensor([1])
        d3 = torch.tensor([3])
        d4 = torch.tensor([4])
        # 运行两次测试，每次将 Squeeze 实例作为模块，输入为 (d1, d4) 和 (d3, d4)，额外测试输入分别为 ((d3, d4),) 和 ((d1, d3),)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Squeeze(), (d1, d4), additional_test_inputs=[((d3, d4),)]
        )
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Squeeze(), (d3, d4), additional_test_inputs=[((d1, d3),)]
        )

    # 定义一个测试方法，用于测试切片操作
    def test_slice(self):
        # 定义一个动态切片导出模块的 Torch 模块
        class DynamicSliceExportMod(torch.nn.Module):
            def forward(self, x):
                results = []
                # 循环执行多次切片操作，将结果添加到列表中
                for i in range(4):
                    results.append(x[: x.size(0) - i, i : x.size(2), i:3])
                # 返回结果元组
                return tuple(results)

        # 创建随机张量 x 和 y，形状分别为 (5, 5, 5) 和 (6, 7, 8)
        x = torch.rand(5, 5, 5)
        y = torch.randn(6, 7, 8)
        # 运行测试，将 DynamicSliceExportMod 实例作为模块，输入为 (x,)，额外测试输入为 ((y,),)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            DynamicSliceExportMod(),
            (x,),
            additional_test_inputs=[((y,),)],
        )
    # 在测试中使用装饰器，如果模型类型为 ExportedProgram，则跳过测试，报错信息为 "Expected 1 outputs, got 2"
    @pytorch_test_common.xfail_if_model_type_is_exportedprogram(
        error_message="Expected 1 outputs, got 2",
    )
    def test_mutation(self):
        # 定义一个继承自 torch.nn.Module 的模型类 MutationModel
        class MutationModel(torch.nn.Module):
            # 模型的前向传播方法
            def forward(self, x):
                # 对输入张量 x 进行形状变换，并在原地加上 2.0
                x.view(3, 2, -1).add_(2.0)
                return x
    
        # 运行测试，使用 FX 转 ONNX 导出器和 ONNX 运行时
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            MutationModel(), (torch.randn(12),), has_mutation=True
        )
    
    # 测试 arange 方法
    def test_arange(self):
        # 定义一个继承自 torch.nn.Module 的模型类 ArangeModel
        class ArangeModel(torch.nn.Module):
            # 模型的前向传播方法
            def forward(self, input):
                # 返回三个张量：0 到 input.shape[0]-1 的整数序列张量、0 到 11 的整数序列张量、从 input.shape[0] 开始到 input.shape[0]+4 的整数序列张量
                return (
                    torch.arange(input.shape[0]),
                    torch.arange(12),
                    torch.arange(start=input.shape[0], end=input.shape[0] + 5),
                )
    
        # 创建一个形状为 (5, 3, 2) 的随机张量 x 和一个形状为 (8, 3, 2) 的随机张量 y
        x = torch.randn(5, 3, 2)
        y = torch.randn(8, 3, 2)
        # 运行测试，使用 FX 转 ONNX 导出器和 ONNX 运行时
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            ArangeModel(),
            (x,),
            additional_test_inputs=[((y,),)],
        )
    
    # 在测试中使用两个装饰器，如果动态 FX 测试失败，则跳过测试，报错信息为 "[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Slice node."
    @pytorch_test_common.xfail_dynamic_fx_test(
        error_message="[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Slice node. "
    )
    @pytorch_test_common.xfail_if_model_type_is_exportedprogram(
        error_message="Expected 1 outputs, got 2"
    )
    def test_expand_as_fill_zero(self):
        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            # 模型的前向传播方法
            def forward(self, x):
                # 将输入张量 x 的右半部分（x.size(0) 到末尾）填充为 0
                x[:, x.size(0) :] = 0
                return x
    
        # 创建形状为 (2, 5) 的全为 1 的张量 x 和形状为 (3, 4) 的随机张量 x2
        x = torch.ones(2, 5)
        x2 = torch.randn(3, 4)
        # 运行测试，使用 FX 转 ONNX 导出器和 ONNX 运行时
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Model(),
            (x,),
            additional_test_inputs=[((x2,),)],
        )
    
    # 在测试中使用两个装饰器，如果动态 FX 测试失败，则跳过测试，报错信息为 "[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Slice node."
    @pytorch_test_common.xfail_dynamic_fx_test(
        error_message="[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Slice node. "
    )
    @pytorch_test_common.xfail_if_model_type_is_exportedprogram(
        error_message="Expected 1 outputs, got 2"
    )
    def test_expand_as_fill_tensor(self):
        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            # 模型的前向传播方法
            def forward(self, x):
                # 将输入张量 x 的右半部分（x.size(0) 到末尾）填充为 [1, 2, 3]
                x[:, x.size(0) :] = torch.tensor([1, 2, 3])
                return x
    
        # 创建形状为 (2, 5, 3) 的全为 1 的张量 x 和形状为 (3, 4, 3) 的随机张量 x2
        x = torch.ones(2, 5, 3)
        x2 = torch.randn(3, 4, 3)
        # 运行测试，使用 FX 转 ONNX 导出器和 ONNX 运行时
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Model(),
            (x,),
            additional_test_inputs=[((x2,),)],
        )
    
    # 如果模型类型不是 ExportedProgram，则跳过测试，报错信息为 "at::functionalization::impl::isFunctionalTensor(self_) INTERNAL ASSERT FAILED"
    @pytorch_test_common.xfail_if_model_type_is_not_exportedprogram(
        error_message="at::functionalization::impl::isFunctionalTensor(self_) INTERNAL ASSERT FAILED"
    )
    # 定义一个测试函数，用于测试模型在输入为 x 时是否能正确扩展为与输入 x 相同形状的张量
    def test_expand_as_fill_separate_tensor(self):
        # 定义一个简单的模型类，重写 forward 方法，返回一个固定的张量 aa
        class Model(torch.nn.Module):
            def forward(self, x):
                # 创建一个固定的二维张量 aa
                aa = torch.tensor([[0], [1], [2]])
                # 将 aa 扩展为与输入 x 相同形状的张量，并返回结果
                return aa.expand_as(x)

        # 创建两个测试输入张量 x 和 x2
        x = torch.ones(3, 2)
        x2 = torch.randn(3, 5)
        # 运行测试函数，并导出为 ONNX 格式，然后使用 ONNX 运行时执行
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Model(),
            (x,),
            additional_test_inputs=[((x2,),)],
        )

    # 标记为需要 CUDA 的测试，测试 _scaled_dot_product_flash_attention 函数
    @pytorch_test_common.skipIfNoCuda
    def test__scaled_dot_product_flash_attention(self):
        # 定义一个简单的模型类 Foo，重写 forward 方法，调用底层的 C++ 扩展函数
        class Foo(torch.nn.Module):
            def forward(self, x):
                # 调用底层的 C++ 扩展函数 _scaled_dot_product_flash_attention
                (
                    output,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = torch.ops.aten._scaled_dot_product_flash_attention(x, x, x)
                # 返回扩展函数的输出
                return output

        # 创建一个 CUDA 设备上的输入张量 x
        x = torch.randn(1, 1, 1, 32, device=torch.device("cuda"))
        # 运行测试函数，并导出为 ONNX 格式，然后使用 ONNX 运行时执行
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (x,))

    # 测试视图操作中的动态形状处理，包含输入和输出的动态维度
    def test_view_dynamic_zero_dim(self):
        # 定义一个 ViewModel 类，重写 forward 方法，进行输入张量的视图操作
        class ViewModel(torch.nn.Module):
            def forward(self, input):
                # 将输入张量重新视图为固定形状
                input = input.view(-1, 2)
                # 将视图后的张量再次重新视图为不同的形状
                return input.view(1, -1)

        # 创建两个测试输入张量 x 和 y
        x = torch.ones(2)
        y = torch.empty(0)
        # 运行测试函数，并导出为 ONNX 格式，然后使用 ONNX 运行时执行
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            ViewModel(),
            (x,),
            additional_test_inputs=[((y,),)],
        )

    # 测试在指定维度范围内展平张量的操作
    def test_flatten_dynamic_axes(self):
        # 定义一个 MyModule 类，重写 forward 方法，使用 flatten 函数在指定维度范围内展平张量
        class MyModule(torch.nn.Module):
            def forward(self, x):
                # 使用 flatten 函数在指定维度范围内展平输入张量 x
                return torch.flatten(x, start_dim=2, end_dim=3)

        # 创建两个测试输入张量 x 和 y
        batch_size = 3
        x = torch.randn(batch_size, 5, 4, 5)
        y = torch.randn(5, 5, 4, 5)
        # 创建模型实例
        model = MyModule()
        # 运行测试函数，并导出为 ONNX 格式，然后使用 ONNX 运行时执行
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            model, (x,), additional_test_inputs=[((y,),)]
        )

    # 测试处理可能为 None 的输入参数情况
    def test_none_input(self):
        # 定义一个 NoneInputModel 类，重写 forward 方法，处理可能为 None 的输入参数
        class NoneInputModel(torch.nn.Module):
            def forward(
                self, x: torch.Tensor, y: Optional[torch.Tensor], z: torch.Tensor
            ):
                # 如果 y 是 None，则返回 x 和 z 的加法结果
                if y is None:
                    return x + z
                # 否则返回 x、y 和 z 的加法结果
                return x + y + z

        # 运行测试函数，并导出为 ONNX 格式，然后使用 ONNX 运行时执行
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            NoneInputModel(), (torch.randn(1, 2), None, torch.randn(1, 2))
        )

    # 测试带有数据依赖输出的操作
    def test_operator_with_data_dependent_output(self):
        # 定义一个 Foo 类，重写 forward 方法，使用全局最小值填充输入张量 x
        class Foo(torch.nn.Module):
            def forward(self, x):
                # 使用全局最小值填充输入张量 x，并返回结果
                return x + torch.full(x.shape, torch.tensor(torch.finfo(x.dtype).min))

        # 创建 Foo 类的实例 func
        func = Foo()

        # 运行测试函数，并导出为 ONNX 格式，然后使用 ONNX 运行时执行
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func, (torch.randn(3, 4),)
        )
    # 定义一个测试方法，用于测试返回标量输出的操作符
    def test_operator_with_scalar_output(self):
        # 定义一个继承自torch.nn.Module的类Foo
        class Foo(torch.nn.Module):
            # 定义该类的前向传播方法，接受两个输入x和y，并返回x的标量值加上y的结果
            def forward(self, x, y):
                return x.item() + y

        # 创建Foo类的实例func
        func = Foo()

        # 运行测试，并导出到ONNX格式，使用ONNX运行时执行
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func, (torch.tensor([1]), torch.randn(3, 4))
        )

    # 用于测试具有动态输出形状的操作符
    @pytorch_test_common.xfail_if_model_type_is_exportedprogram(
        # 如果模型包含不支持的FX节点，会引发失败，给出相应错误消息和原因链接
        error_message="Unsupported FX nodes: {'call_function': ['aten._assert_async.msg']}",
        reason="https://github.com/pytorch/pytorch/issues/112622",
    )
    def test_operator_with_dynamic_output_shape(self):
        # 定义一个继承自torch.nn.Module的类Foo
        class Foo(torch.nn.Module):
            # 定义该类的前向传播方法，接受输入x，并返回x的非零元素的索引
            def forward(self, x):
                return x.nonzero()

        # 创建Foo类的实例func
        func = Foo()

        # 运行测试，并导出到ONNX格式，使用ONNX运行时执行
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func, (torch.randn(3, 4),)
        )

    # 测试从配置创建GPT-2小模型
    @pytorch_test_common.xfail_if_model_type_is_exportedprogram(
        # 如果模型包含特定的输入树规范，会引发失败，给出相应错误消息
        error_message="Trying to flatten user inputs with exported input tree spec"
    )
    @pytorch_test_common.xfail_dynamic_fx_test(
        # 如果模型包含特定的名称不为空的错误，会引发失败，给出相应原因和模型类型
        error_message="!(it.GetName().empty())",
        reason="With after onnx==1.16, constant folding in optimizer causes this error.",
        model_type=pytorch_test_common.TorchModelType.TORCH_NN_MODULE,
    )
    def test_gpt2_tiny_from_config(self):
        # 配置GPT-2小模型的参数
        config = transformers.GPT2Config(
            num_hidden_layers=4,
            vocab_size=8096,
            hidden_size=16,
            intermediate_size=16,
            max_position_embeddings=512,
            num_attention_heads=2,
            hidden_dropout_prob=0.0,
            attention_dropout_prob=0.0,
        )
        # 根据配置创建GPT-2模型并设置为评估模式
        model = transformers.GPT2Model(config).eval()

        # 定义一个生成输入的函数，返回输入ID、注意力掩码和位置ID
        def input_generator(batch: int, seq: int):
            input_ids = torch.randint(0, 8096, (batch, seq))
            attention_mask = torch.ones(batch, seq, dtype=torch.bool)
            position_ids = torch.arange(0, seq, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).view(-1, seq)
            return input_ids, attention_mask, position_ids

        # 创建编码的输入
        input_ids, attention_mask, position_ids = input_generator(2, 128)

        # 创建另一个编码的输入，用于测试动态形状
        (
            another_input_ids,
            another_attention_mask,
            another_position_ids,
        ) = input_generator(3, 256)

        # 运行测试，并导出到ONNX格式，使用ONNX运行时执行
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            model,
            (input_ids,),
            input_kwargs={
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            additional_test_inputs=[
                (
                    (another_input_ids,),
                    {
                        "attention_mask": another_attention_mask,
                        "position_ids": another_position_ids,
                    },
                )
            ],
        )
    def test_prims_device_put(self):
        # 定义一个自定义的 nn.Module 子类 CustomModule
        class CustomModule(nn.Module):
            # 重写 forward 方法，用于模型前向传播
            def forward(self, x):
                # 假设 x 是在 CPU 上的张量，使用 device_put() 将其移到指定设备（这里是 CPU）
                x = torch.ops.prims.device_put(x, "cpu")
                return x

        # 运行测试，并导出到 ONNX 格式进行验证
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            CustomModule(), (torch.randn(1, 2, 3),)
        )

    @_beartype.beartype
    def _test_fx_symbolic_tracer_large_scale_exporter(
        self,
        model_name: str,
        create_model: Callable,
        create_args: Callable,
        create_pytorch_only_kwargs: Callable,
    ):
        # 测试用例装饰器，用于处理动态功能测试失败的情况
        @pytorch_test_common.xfail_dynamic_fx_test(
            error_message="shape_env should be set if tracing with 'symbolic'"
        )
        # 定义一个测试函数，用于测试符号跟踪器的大规模导出功能
        def test_fx_symbolic_tracer_large_scale_exporter_with_toy_mlp(self):
            # 定义一个简单的多层感知机模型 MLPModel
            class MLPModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc0 = nn.Linear(8, 8, bias=True)
                    self.fc1 = nn.Linear(8, 4, bias=True)
                    self.fc2 = nn.Linear(4, 2, bias=True)
                    self.fc3 = nn.Linear(2, 2, bias=True)

                # 定义模型的前向传播方法
                def forward(self, tensor_x: torch.Tensor):
                    tensor_x = self.fc0(tensor_x)
                    tensor_x = torch.sigmoid(tensor_x)
                    tensor_x = self.fc1(tensor_x)
                    tensor_x = torch.sigmoid(tensor_x)
                    tensor_x = self.fc2(tensor_x)
                    tensor_x = torch.sigmoid(tensor_x)
                    output = self.fc3(tensor_x)
                    return output

            # 创建模型的函数
            def create_model() -> nn.Module:
                return MLPModel()

            # 创建模型输入的函数
            def create_args():
                return (torch.rand((97, 8), dtype=torch.float32),)

            # 创建仅用于 PyTorch 的额外关键字参数的函数
            def create_pytorch_only_extra_kwargs():
                return {}

            # 调用内部的符号跟踪器大规模导出功能测试函数
            self._test_fx_symbolic_tracer_large_scale_exporter(
                "toy_mlp1",
                create_model,
                create_args,
                create_pytorch_only_extra_kwargs,
            )
    # 定义一个测试方法，用于测试符号跟踪大规模导出器与小型GPT-2模型
    def test_fx_symbolic_tracer_large_scale_exporter_with_tiny_gpt2(self):
        # 指定模型名称为 "sshleifer/tiny-gpt2"
        model_name = "sshleifer/tiny-gpt2"
        # 指定设备为 CPU

        # 定义一个内部方法，用于创建模型并返回其作为 nn.Module 对象
        def create_model() -> nn.Module:
            # 使用预训练模型名称加载自动模型，并将其移到指定设备上，然后设置为评估模式
            return transformers.AutoModel.from_pretrained(model_name).to(device).eval()

        # 定义一个内部方法，用于创建参数，并返回输入 ID 和注意力掩码
        def create_args():
            # 使用预训练模型名称加载自动分词器
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            # 对文本 "Hello world!" 进行分词并返回 PyTorch 张量
            kwargs = tokenizer("Hello world!", return_tensors="pt")
            input_ids = kwargs["input_ids"]
            attention_mask = kwargs["attention_mask"]
            return input_ids, None, attention_mask

        # 定义一个内部方法，返回一个额外的 PyTorch 参数字典
        def create_pytorch_only_extra_kwargs():
            return {"return_dict": False}

        # 调用外部方法来测试符号跟踪大规模导出器，传递模型名称、创建模型方法、创建参数方法和额外的 PyTorch 参数方法作为参数
        self._test_fx_symbolic_tracer_large_scale_exporter(
            "tiny_gpt2",
            create_model,
            create_args,
            create_pytorch_only_extra_kwargs,
        )
# 定义一个返回参数化类属性和值的函数，包括假选项
def _parameterized_class_attrs_and_values_with_fake_options():
    # 初始化输入值为空列表
    input_values = []
    # 扩展输入值列表，生成所有可能的组合
    input_values.extend(
        itertools.product(
            (True, False),  # 第一个元组：布尔值True和False
            (True, False),  # 第二个元组：布尔值True和False
            (True, False),  # 第三个元组：布尔值True和False
            (True, False),  # 第四个元组：布尔值True和False
            (
                pytorch_test_common.TorchModelType.TORCH_NN_MODULE,  # 第五个元组：Torch模型类型
                pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,  # 第五个元组：Torch模型类型
            ),
        )
    )
    # 返回一个包含属性和输入值的字典
    return {
        "attrs": [  # 类的属性列表
            "op_level_debug",
            "dynamic_shapes",
            "load_checkpoint_during_init",
            "export_within_fake_mode",
            "model_type",
        ],
        "input_values": input_values,  # 输入值列表
    }


# 使用parameterized_class装饰器参数化测试类
@parameterized.parameterized_class(
    **_parameterized_class_attrs_and_values_with_fake_options(),  # 使用参数化类的属性和值
    class_name_func=_parameterize_class_name,  # 类名生成函数
)
# 测试特定Fake Tensor场景的ONNX导出
class TestFxToOnnxFakeTensorWithOnnxRuntime(onnx_test_common._TestONNXRuntime):
    """ONNX export test for specific Fake Tensor scenarios

    TODO: Should we merge this with  `TestFxToOnnxWithOnnxRuntime`? Considerably increases export time
    """

    # 类的属性定义
    op_level_debug: bool
    dynamic_shapes: bool
    load_checkpoint_during_init: bool
    export_within_fake_mode: bool
    model_type: pytorch_test_common.TorchModelType

    # 设置测试环境
    def setUp(self):
        super().setUp()
        # 设置ONNX Runtime的版本号
        self.ort_version = onnxruntime.__version__

    # 使用beartype装饰器类型检查
    @_beartype.beartype
    # 测试Fake Tensor模式的导出器
    def _test_fake_tensor_mode_exporter(
        self,
        model_name: str,
        create_model: Callable,
        create_args: Callable,
        create_kwargs: Callable,
        load_checkpoint_during_init: bool,
        export_within_fake_mode: bool,
        model_type: pytorch_test_common.TorchModelType,
    ):
        def create_model() -> nn.Module:
            # 定义一个简单的神经网络模型
            class Model(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linear = torch.nn.Linear(2, 2)

                def forward(self, x):
                    out = self.linear(x)
                    return out

            return Model()

        def create_args():
            return (torch.rand(5, 2, 2),)

        def create_kwargs():
            return {}

        # 调用内部测试方法
        self._test_fake_tensor_mode_exporter(
            "simple",  # 模型名称
            create_model,  # 创建模型的函数
            create_args,  # 创建参数的函数
            create_kwargs,  # 创建关键字参数的函数
            load_checkpoint_during_init=self.load_checkpoint_during_init,  # 加载初始化时是否加载检查点
            export_within_fake_mode=self.export_within_fake_mode,  # 在Fake模式下导出
            model_type=self.model_type,  # 模型类型
        )

    # 使用skip_dynamic_fx_test装饰器跳过动态FX测试
    @pytorch_test_common.skip_dynamic_fx_test(
        reason="Dynamic shape check is not expected for exported program in this test suite.",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,  # 模型类型
    )
    # 测试简单的Fake Tensor模式
    def test_fake_tensor_mode_simple(self):
        pass  # 此处的具体测试逻辑未提供，暂时为空
    # 在这个测试函数上添加装饰器，跳过动态形状检查的测试
    @pytorch_test_common.skip_dynamic_fx_test(
        reason="Dynamic shape check is not expected for exported program in this test suite.",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
    )
    # 继续添加装饰器，标记为在动态特性测试中有望失败
    @pytorch_test_common.xfail_dynamic_fx_test(
        error_message="!(it.GetName().empty())",
        reason="With after onnx==1.16, constant folding in optimizer causes this error.",
        model_type=pytorch_test_common.TorchModelType.TORCH_NN_MODULE,
    )
    # 继续添加装饰器，标记为在模型类型不是导出程序时有望失败
    @pytorch_test_common.xfail_if_model_type_is_not_exportedprogram(
        error_message="Expected 4 inputs, got 2",
        reason="https://github.com/pytorch/pytorch/issues/115745",
    )
    # 定义一个测试函数，测试在假张量模式下的 Huggingface Tiny GPT-2 模型
    def test_fake_tensor_mode_huggingface_tiny_gpt2(self):
        # 模型名称
        model_name = "sshleifer/tiny-gpt2"
        # 设备类型
        device = "cpu"
    
        # 定义创建模型的函数，返回一个被预训练模型加载后的模块并转移到设备上并设置为评估模式
        def create_model() -> nn.Module:
            return transformers.AutoModel.from_pretrained(model_name).to(device).eval()
    
        # 定义创建参数的函数，使用模型的 tokenizer 对输入文本进行编码，返回输入张量和注意力掩码
        def create_args():
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            kwargs = tokenizer("Hello world!", return_tensors="pt")
            input_ids = kwargs["input_ids"]
            attention_mask = kwargs["attention_mask"]
            return input_ids, None, attention_mask
    
        # 定义创建关键字参数的函数，这里返回一个字典，指定是否返回字典
        def create_kwargs():
            return {"return_dict": False}
    
        # 调用测试函数 _test_fake_tensor_mode_exporter，测试假张量模式下的模型导出
        self._test_fake_tensor_mode_exporter(
            "tiny_gpt2",
            create_model,
            create_args,
            create_kwargs,
            load_checkpoint_during_init=self.load_checkpoint_during_init,
            export_within_fake_mode=self.export_within_fake_mode,
            model_type=self.model_type,
        )
    
    # 继续添加装饰器，跳过动态形状检查的测试
    @pytorch_test_common.skip_dynamic_fx_test(
        reason="Dynamic shape check is not expected for exported program in this test suite.",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
    )
    @pytorch_test_common.skip_dynamic_fx_test(
        reason="Dynamic shape check is not expected for exported program in this test suite.",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
    )



# 使用装饰器标记该测试函数为跳过动态形状检查的测试
@pytorch_test_common.skip_dynamic_fx_test(
    reason="Dynamic shape check is not expected for exported program in this test suite.",
    model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
)



        config = transformers.T5Config(
            vocab_size=8096, d_model=64, num_layers=2, num_heads=2
        )



# 创建一个 T5 模型的配置对象，指定词汇表大小、模型维度、层数和注意力头数
config = transformers.T5Config(
    vocab_size=8096, d_model=64, num_layers=2, num_heads=2
)



        batch, seq = 4, 256



# 定义批量大小和序列长度
batch, seq = 4, 256



        def create_args():
            return tuple()



# 创建一个函数，用于生成不带参数的元组作为输入参数
def create_args():
    return tuple()



        def create_kwargs():
            input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            attention_mask = torch.ones((batch, seq), dtype=torch.bool)
            decoder_input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": decoder_input_ids,
            }



# 创建一个函数，用于生成关键字参数字典，包含 T5 模型的输入所需的各种张量
def create_kwargs():
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))
    attention_mask = torch.ones((batch, seq), dtype=torch.bool)
    decoder_input_ids = torch.randint(0, config.vocab_size, (batch, seq))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
    }



        def create_model():
            return transformers.T5Model(config).eval()



# 创建一个函数，用于生成已配置为评估模式的 T5 模型实例
def create_model():
    return transformers.T5Model(config).eval()



        self._test_fake_tensor_mode_exporter(
            "huggingface_google_t5",
            create_model,
            create_args,
            create_kwargs,
            load_checkpoint_during_init=self.load_checkpoint_during_init,
            export_within_fake_mode=self.export_within_fake_mode,
            model_type=self.model_type,
        )



# 调用测试函数 _test_fake_tensor_mode_exporter，导出 Huggingface Google T5 模型的伪张量模式
self._test_fake_tensor_mode_exporter(
    "huggingface_google_t5",
    create_model,
    create_args,
    create_kwargs,
    load_checkpoint_during_init=self.load_checkpoint_during_init,
    export_within_fake_mode=self.export_within_fake_mode,
    model_type=self.model_type,
)
    # 使用 pytorch_test_common.xfail_dynamic_fx_test 装饰器标记测试函数，表示预期该测试会失败
    # 如果失败，将显示指定的错误消息和原因
    @pytorch_test_common.xfail_dynamic_fx_test(
        error_message="scaled_dot_product_attention(): argument 'is_causal' must be bool, not SymBool",
        reason="Dynamo error: scaled_dot_product_attention(): argument 'is_causal' must be bool, not SymBool",
        model_type=pytorch_test_common.TorchModelType.TORCH_NN_MODULE,
    )
    # 使用 pytorch_test_common.xfail_op_level_debug_test 装饰器标记测试函数，表示预期该测试会失败
    # 如果失败，将显示指定的错误消息和原因
    @pytorch_test_common.xfail_op_level_debug_test(
        error_message="Could not find an implementation for Trilu(14) node",
        reason="ORT error during op level dubug",
        model_type=pytorch_test_common.TorchModelType.TORCH_NN_MODULE,
    )
    # 使用 pytorch_test_common.xfail_if_model_type_is_exportedprogram 装饰器标记测试函数，表示预期该测试会失败
    # 如果失败，将显示指定的错误消息和原因
    @pytorch_test_common.xfail_if_model_type_is_exportedprogram(
        error_message="n=copy_, n.args[0]=zeros_like, placeholders={",
        reason="aot_autograd doesn't support it.",
    )
    # 定义测试函数 test_fake_tensor_mode_huggingface_openai_whisper
    def test_fake_tensor_mode_huggingface_openai_whisper(self):
        # 创建 transformers.WhisperConfig 对象作为配置
        config = transformers.WhisperConfig(
            vocab_size=8096,
            num_mel_bins=40,
            encoder_layers=2,
            encoder_attention_heads=2,
            decoder_layers=2,
            decoder_attention_heads=2,
            decoder_ffn_dim=384,
            encoder_ffn_dim=384,
            d_model=64,
            decoder_start_token_id=8001,
            pad_token_id=8000,
            bos_token_id=8000,
            eos_token_id=8000,
            begin_suppress_tokens=[220, 8000],
        )
        # 创建 transformers.WhisperFeatureExtractor 对象作为特征提取器
        feature_extractor = transformers.WhisperFeatureExtractor(feature_size=40)
        # 设置设备为 CPU
        device = "cpu"
        # 设置批处理大小为 4
        batch = 4

        # 定义创建模型的函数，返回一个经配置初始化后的模型对象，并设为评估模式
        def create_model() -> nn.Module:
            return transformers.AutoModel.from_config(config).to(device).eval()

        # 定义创建参数的函数，返回一个空元组
        def create_args():
            return ()

        # 定义创建关键字参数的函数，返回一个包含输入特征和解码器输入 ID 的字典
        def create_kwargs():
            input_features = torch.randn(
                (
                    batch,
                    feature_extractor.feature_size,
                    feature_extractor.nb_max_frames,
                ),
                dtype=torch.float32,
            )
            decoder_input_ids = torch.tensor([[1, 1]]) * config.decoder_start_token_id
            return {
                "input_features": input_features,
                "decoder_input_ids": decoder_input_ids,
                "return_dict": False,
            }

        # 调用 self._test_fake_tensor_mode_exporter 方法来执行测试
        self._test_fake_tensor_mode_exporter(
            "openai_whisper",
            create_model,
            create_args,
            create_kwargs,
            load_checkpoint_during_init=self.load_checkpoint_during_init,
            export_within_fake_mode=self.export_within_fake_mode,
            model_type=self.model_type,
        )

    # 使用 pytorch_test_common.skip_dynamic_fx_test 装饰器标记测试函数，跳过动态形状检查的测试
    # 因为在该测试套件中，不期望在导出程序中进行动态形状检查
    @pytorch_test_common.skip_dynamic_fx_test(
        reason="Dynamic shape check is not expected for exported program in this test suite.",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
    )
    @pytorch_test_common.skip_dynamic_fx_test(
        reason="Dynamic shape check is not expected for exported program in this test suite.",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
    )
    @pytorch_test_common.xfail_dynamic_fx_test(
        error_message="SymIntArrayRef expected to contain only concrete integers",
        model_type=pytorch_test_common.TorchModelType.TORCH_NN_MODULE,
    )

在这个测试方法上方，使用了装饰器来标记测试函数跳过动态张量检查和标记为预期失败的测试。第一个装饰器`@pytorch_test_common.skip_dynamic_fx_test`指定了跳过的原因和模型类型。第二个装饰器`@pytorch_test_common.xfail_dynamic_fx_test`标记了在某些条件下会失败的测试，指定了失败时的错误消息和模型类型。


    def test_fake_tensor_mode_huggingface_bigscience_bloom_560m(self):
        config = transformers.BloomConfig()
        batch, seq = 4, 256

        def create_args():
            return tuple()

        def create_kwargs():
            input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            attention_mask = torch.ones(batch, seq, dtype=torch.bool)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        def create_model():
            return transformers.BloomModel(config).eval()

        self._test_fake_tensor_mode_exporter(
            "huggingface_bigscience_bloom_560m",
            create_model,
            create_args,
            create_kwargs,
            load_checkpoint_during_init=self.load_checkpoint_during_init,
            export_within_fake_mode=self.export_within_fake_mode,
            model_type=self.model_type,
        )

这是一个测试方法，用于测试"huggingface_bigscience_bloom_560m"模型在假张量模式下的导出功能。在方法内部定义了用于创建参数和模型的几个辅助函数`create_args`、`create_kwargs`和`create_model`。最后调用了`self._test_fake_tensor_mode_exporter`方法来执行实际的测试，传入了模型名称、创建模型函数、创建参数函数、创建关键字参数函数以及其他必要的参数。


    @pytorch_test_common.skip_dynamic_fx_test(
        reason="Dynamic shape check is not expected for exported program in this test suite.",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
    )
    @pytorch_test_common.xfail_if_model_type_is_not_exportedprogram(
        error_message="Expected 5 inputs, got 3",
        reason="https://github.com/pytorch/pytorch/issues/115745",
    )

在这个测试方法上方，使用了两个装饰器来标记测试函数。第一个装饰器`@pytorch_test_common.skip_dynamic_fx_test`指定了跳过的原因和模型类型。第二个装饰器`@pytorch_test_common.xfail_if_model_type_is_not_exportedprogram`标记了如果模型类型不是`TORCH_EXPORT_EXPORTEDPROGRAM`则会失败的测试，指定了失败时的错误消息和失败原因的链接。
    # 定义一个测试方法，用于测试在伪张量模式下导出 Huggingface GPT-2 模型
    def test_fake_tensor_mode_huggingface_gpt2(self):
        # 创建一个 GPT-2 的配置对象，指定词汇大小、位置数、嵌入维度、层数和头数
        config = transformers.GPT2Config(
            vocab_size=8096, n_positions=256, n_embd=256, n_layer=2, n_head=2
        )

        # 定义一个创建模型的内部函数，返回一个处于评估模式的 GPT-2 模型对象
        def create_model():
            return transformers.GPT2Model(config).eval()

        # 定义一个创建参数的内部函数，返回一个空元组
        def create_args():
            return tuple()

        # 定义一个创建关键字参数的内部函数
        def create_kwargs():
            # 设置批量大小和序列长度
            batch, seq = 4, 256

            # 生成指定范围内随机整数的张量作为输入 ids
            input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            # 创建全为真的注意力遮罩张量
            attention_mask = torch.ones(batch, seq, dtype=torch.bool)
            # 创建序列位置 id 的张量
            position_ids = torch.arange(0, seq, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).view(-1, seq)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }

        # 调用私有方法 _test_fake_tensor_mode_exporter 进行测试，传入相关参数
        self._test_fake_tensor_mode_exporter(
            "huggingface_gpt2",
            create_model,
            create_args,
            create_kwargs,
            load_checkpoint_during_init=self.load_checkpoint_during_init,
            export_within_fake_mode=self.export_within_fake_mode,
            model_type=self.model_type,
        )

    # 使用装饰器跳过动态形状 FX 测试，给出跳过的原因和模型类型
    @pytorch_test_common.skip_dynamic_fx_test(
        reason="Dynamic shape check is not expected for exported program in this test suite.",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
    )
    # 使用装饰器标记预期动态形状 FX 测试失败，提供失败时的错误信息和模型类型
    @pytorch_test_common.xfail_dynamic_fx_test(
        error_message="SymIntArrayRef expected to contain only concrete integers",
        model_type=pytorch_test_common.TorchModelType.TORCH_NN_MODULE,
    )
    # 使用装饰器标记预期在非导出程序模型类型下失败的测试，提供失败时的错误信息和跳过的原因
    @pytorch_test_common.xfail_if_model_type_is_not_exportedprogram(
        error_message="Expected 9 inputs, got 3",
        reason="https://github.com/pytorch/pytorch/issues/115745",
    )
    def test_fake_tensor_mode_huggingface_databricks_dolly_v2_3b(self):
        # 创建一个 GPTNeoXConfig 对象，配置模型参数：词汇大小为 8096，隐藏层大小为 256，2 层隐藏层，2 个注意力头
        config = transformers.GPTNeoXConfig(
            vocab_size=8096, hidden_size=256, num_hidden_layers=2, num_attention_heads=2
        )
        # 定义批量大小和序列长度
        batch, seq = 4, 256

        def create_model():
            # 返回一个在评估模式下的 GPTNeoXModel 对象
            return transformers.GPTNeoXModel(config).eval()

        def create_args():
            # 返回一个空元组
            return tuple()

        def create_kwargs():
            # 创建输入张量字典
            input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            attention_mask = torch.ones(batch, seq, dtype=torch.bool)
            position_ids = torch.arange(0, seq, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).view(-1, seq)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }

        # 调用自定义的测试函数 _test_fake_tensor_mode_exporter，传入多个参数和关键字参数
        self._test_fake_tensor_mode_exporter(
            "huggingface_databricks_dolly_v2_3b",
            create_model,
            create_args,
            create_kwargs,
            load_checkpoint_during_init=self.load_checkpoint_during_init,
            export_within_fake_mode=self.export_within_fake_mode,
            model_type=self.model_type,
        )
# 如果这个脚本作为主程序运行（而不是被导入到其他脚本中），则执行以下代码块
if __name__ == "__main__":
    # 导入的 common_utils 模块中的 run_tests 函数，用于执行测试用例
    common_utils.run_tests()
```
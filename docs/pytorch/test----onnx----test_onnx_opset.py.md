# `.\pytorch\test\onnx\test_onnx_opset.py`

```py
# Owner(s): ["module: onnx"]

# 导入必要的库
import io
import itertools

import onnx  # 导入ONNX库
import pytorch_test_common  # 导入PyTorch测试公共模块

import torch  # 导入PyTorch
import torch.onnx  # 导入PyTorch的ONNX模块
from torch.nn import Module  # 从torch.nn中导入Module类
from torch.onnx import producer_name, producer_version  # 导入producer_name和producer_version
from torch.onnx._globals import GLOBALS  # 导入GLOBALS
from torch.testing._internal import common_utils  # 导入PyTorch内部的common_utils模块


def check_onnx_opset_operator(
    model, ops, opset_version=GLOBALS.export_onnx_opset_version
):
    # 检查模型的生产者名称、版本和导入的opset版本是否与参数匹配
    assert (
        model.producer_name == producer_name
        and model.producer_version == producer_version
        and model.opset_import[0].version == opset_version
    )

    # 使用ONNX的检查器检查模型
    onnx.checker.check_model(model)

    # 检查目标类型和属性
    graph = model.graph
    # ops应该包含每个节点的对象
    # 在graph.node中，按正确的顺序。
    # 至少应指定op_name，
    # 但也可以选择指定op的属性
    assert len(ops) == len(graph.node)
    for i in range(0, len(ops)):
        assert graph.node[i].op_type == ops[i]["op_name"]
        if "attributes" in ops[i]:
            attributes = ops[i]["attributes"]
            assert len(attributes) == len(graph.node[i].attribute)
            for j in range(0, len(attributes)):
                for attribute_field in attributes[j].keys():
                    assert attributes[j][attribute_field] == getattr(
                        graph.node[i].attribute[j], attribute_field
                    )


def check_onnx_opsets_operator(
    module,
    x,
    ops,
    opset_versions,
    training=torch.onnx.TrainingMode.EVAL,
    input_names=None,
    dynamic_axes=None,
):
    # 对于每个opset_version，导出模型并检查
    for opset_version in opset_versions:
        f = io.BytesIO()
        # 使用torch.onnx.export导出模型到字节流f
        torch.onnx.export(
            module,
            x,
            f,
            opset_version=opset_version,
            training=training,
            input_names=input_names,
            dynamic_axes=dynamic_axes,
        )
        # 从导出的字节流中加载模型
        model = onnx.load(io.BytesIO(f.getvalue()))
        # 使用check_onnx_opset_operator检查导出的模型
        check_onnx_opset_operator(model, ops[opset_version], opset_version)


class TestONNXOpset(pytorch_test_common.ExportTestCase):
    def test_opset_fallback(self):
        # 定义一个简单的MyModule类，继承自Module
        class MyModule(Module):
            def forward(self, x):
                return torch.isnan(x)

        # 定义操作列表ops，仅包含一个操作{"op_name": "IsNaN"}
        ops = [{"op_name": "IsNaN"}]
        # ops字典，包含opset版本9和10对应的操作列表
        ops = {9: ops, 10: ops}
        # 创建一个包含NaN值的torch张量x
        x = torch.tensor([1.0, float("nan"), 2.0])
        # 调用check_onnx_opsets_operator检查不同opset版本下的操作
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])
    # 定义一个单元测试函数 test_topk，用于测试 topk 操作的 ONNX 导出
    def test_topk(self):
        # 定义一个继承自 Module 的简单模块 MyModule，重写 forward 方法实现 topk 操作
        class MyModule(Module):
            def forward(self, x):
                return torch.topk(x, 3)

        # 定义第一个 ONNX 运算图 ops_9，包含一个 TopK 运算节点
        ops_9 = [
            {
                "op_name": "TopK",
                "attributes": [
                    {"name": "axis", "i": -1, "type": 2},  # 设置 axis 参数为 -1
                    {"name": "k", "i": 3, "type": 2},      # 设置 k 参数为 3
                ],
            }
        ]
        
        # 定义第二个 ONNX 运算图 ops_10，包含一个 Constant 运算节点和一个带有 axis 参数的 TopK 运算节点
        ops_10 = [
            {"op_name": "Constant"},   # Constant 运算节点
            {"op_name": "TopK", "attributes": [{"name": "axis", "i": -1, "type": 2}]},  # 带有 axis 参数的 TopK 运算节点
        ]
        
        # ops 字典包含两个版本的 ONNX 运算图 ops_9 和 ops_10
        ops = {9: ops_9, 10: ops_10}
        
        # 创建一个张量 x，从 1.0 到 6.0，需要梯度计算
        x = torch.arange(1.0, 6.0, requires_grad=True)
        
        # 调用 check_onnx_opsets_operator 函数，测试 MyModule 模块在不同 opset 版本下的 ONNX 导出
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])

        # 在第二部分进行动态 k 值的测试
        # 定义一个继承自 torch.jit.ScriptModule 的脚本模块 MyModuleDynamic，包含一个脚本方法 forward，实现 topk 操作
        class MyModuleDynamic(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input, k):
                return torch.topk(input, k)

        # ops_10 包含一个 Constant 运算节点，一个 Reshape 运算节点和一个带有 axis 参数的 TopK 运算节点
        ops_10 = [
            {"op_name": "Constant", "attributes": [{"name": "value", "type": 4}]},  # Constant 运算节点
            {"op_name": "Reshape"},  # Reshape 运算节点
            {"op_name": "TopK", "attributes": [{"name": "axis", "i": -1, "type": 2}]},  # 带有 axis 参数的 TopK 运算节点
        ]
        
        # ops 字典包含一个版本的 ONNX 运算图 ops_10
        ops = {10: ops_10}
        
        # 创建一个张量 x，从 1.0 到 6.0，需要梯度计算
        x = torch.arange(1.0, 6.0, requires_grad=True)
        
        # 创建一个张量 k，值为 3
        k = torch.tensor(3)
        
        # 创建 MyModuleDynamic 实例
        module = MyModuleDynamic()
        
        # 调用 check_onnx_opsets_operator 函数，测试 MyModuleDynamic 模块在 opset 10 下的 ONNX 导出
        check_onnx_opsets_operator(module, (x, k), ops, opset_versions=[10])
    def test_maxpool(self):
        # 创建一个 MaxPool1d 模块，设置 kernel size 为 2，stride 为 1
        module = torch.nn.MaxPool1d(2, stride=1)

        # 定义运算符描述列表 ops_9，描述了 MaxPool 运算符的属性和参数
        ops_9 = [
            {
                "op_name": "MaxPool",
                "attributes": [
                    {"name": "kernel_shape", "ints": [2], "type": 7},
                    {"name": "pads", "ints": [0, 0], "type": 7},
                    {"name": "strides", "ints": [1], "type": 7},
                ],
            }
        ]
        # 定义运算符描述列表 ops_10，描述了 MaxPool 运算符的属性和参数（包含 ceil_mode）
        ops_10 = [
            {
                "op_name": "MaxPool",
                "attributes": [
                    {"name": "ceil_mode", "i": 0, "type": 2},
                    {"name": "dilations", "ints": [1], "type": 7},
                    {"name": "kernel_shape", "ints": [2], "type": 7},
                    {"name": "pads", "ints": [0, 0], "type": 7},
                    {"name": "strides", "ints": [1], "type": 7},
                ],
            }
        ]
        # 创建 ops 字典，包含 opset 版本 9 和 10 对应的运算符描述列表
        ops = {9: ops_9, 10: ops_10}
        # 生成一个大小为 (20, 16, 50) 的随机张量 x，并检查对应的 ONNX 运算符
        x = torch.randn(20, 16, 50)
        check_onnx_opsets_operator(module, x, ops, opset_versions=[9, 10])

        # 添加一个带有 dilation 参数的 MaxPool1d 模块
        module = torch.nn.MaxPool1d(2, stride=1, dilation=2)

        # 定义运算符描述列表 ops_10，描述了 MaxPool 运算符的属性和参数（包含 ceil_mode 和 dilations）
        ops_10 = [
            {
                "op_name": "MaxPool",
                "attributes": [
                    {"name": "ceil_mode", "i": 0, "type": 2},
                    {"name": "dilations", "ints": [2], "type": 7},
                    {"name": "kernel_shape", "ints": [2], "type": 7},
                    {"name": "pads", "ints": [0, 0], "type": 7},
                    {"name": "strides", "ints": [1], "type": 7},
                ],
            }
        ]
        # 创建 ops 字典，包含 opset 版本 10 对应的运算符描述列表
        ops = {10: ops_10}
        # 生成一个大小为 (20, 16, 50) 的随机张量 x，并检查对应的 ONNX 运算符
        x = torch.randn(20, 16, 50)
        check_onnx_opsets_operator(module, x, ops, opset_versions=[10])

    def test_upsample(self):
        # 定义一个继承自 Module 的自定义模块 MyModule，实现了对输入张量的上采样操作
        class MyModule(Module):
            def forward(self, x):
                # 计算输入张量各维度乘以2后的大小，并转换为整数
                size = [v * 2 for v in x.size()[2:]]
                size = [int(i) for i in size]
                # 使用最近邻插值法对输入张量 x 进行上采样，并返回结果
                return torch.nn.functional.interpolate(x, size=size, mode="nearest")

        # 创建 MyModule 类的实例 module
        module = MyModule()
        # 定义运算符描述列表 ops8，描述了 Upsample 运算符的属性和参数（包含 mode 和 scales）
        ops8 = [
            {
                "op_name": "Upsample",
                "attributes": [
                    {"name": "mode", "s": (b"nearest"), "type": 3},
                    {"name": "scales", "floats": [1.0, 1.0, 2.0, 2.0], "type": 6},
                ],
            }
        ]
        # 定义运算符描述列表 ops9，描述了 Upsample 运算符的属性和参数（仅包含 mode）
        ops9 = [
            {"op_name": "Constant"},
            {
                "op_name": "Upsample",
                "attributes": [{"name": "mode", "s": (b"nearest"), "type": 3}],
            },
        ]
        # 创建 ops 字典，包含 opset 版本 8 和 9 对应的运算符描述列表
        ops = {8: ops8, 9: ops9}
        # 生成一个大小为 (2, 2, 2, 2) 的随机张量 x，并检查对应的 ONNX 运算符
        x = torch.randn(2, 2, 2, 2)
        check_onnx_opsets_operator(module, x, ops, opset_versions=[8, 9])
    # 定义一个测试方法，用于测试将常量转换为其他类型的操作
    def test_cast_constant(self):
        # 定义一个继承自Module的子类MyModule，重写forward方法实现数据处理逻辑
        class MyModule(Module):
            def forward(self, x):
                return x - 1

        # 创建MyModule类的实例
        module = MyModule()
        # 定义一系列操作列表ops_8，其中包含Constant、Cast和Sub操作
        ops_8 = [
            {"op_name": "Constant"},
            {"op_name": "Cast", "attributes": [{"name": "to", "i": 7, "type": 2}]},
            {"op_name": "Sub"},
        ]
        # 定义另一个操作列表ops_9，仅包含Constant和Sub操作
        ops_9 = [{"op_name": "Constant"}, {"op_name": "Sub"}]
        # ops字典将opset版本映射到相应的操作列表
        ops = {8: ops_8, 9: ops_9}
        # 创建一个5x6的长整型张量x，所有元素初始化为1
        x = torch.ones(5, 6, dtype=torch.long)
        # 调用check_onnx_opsets_operator函数，验证MyModule对x在不同opset版本下的操作
        check_onnx_opsets_operator(module, x, ops, opset_versions=[8, 9])

    # 定义一个测试方法，测试张量切片操作
    def test_slice(self):
        # 定义一个继承自Module的子类MyModule，重写forward方法实现切片操作
        class MyModule(Module):
            def forward(self, x):
                return x[0:1]

        # 定义一个切片操作列表ops_9，包含指定轴、起始和结束位置的Slice操作
        ops_9 = [
            {
                "op_name": "Slice",
                "attributes": [
                    {"name": "axes", "ints": [0], "type": 7},
                    {"name": "ends", "ints": [1], "type": 7},
                    {"name": "starts", "ints": [0], "type": 7},
                ],
            }
        ]
        # 定义另一个操作列表ops_10，包含多个Constant和一个空的Slice操作
        ops_10 = [
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Slice", "attributes": []},
        ]
        # ops字典将opset版本映射到相应的操作列表
        ops = {9: ops_9, 10: ops_10}
        # 创建一个长度为3的随机张量x
        x = torch.randn(3)
        # 调用check_onnx_opsets_operator函数，验证MyModule对x在不同opset版本下的操作
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])

        # 定义一个继承自torch.jit.ScriptModule的动态切片模型DynamicSliceModel
        class DynamicSliceModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x[1 : x.size(0)]

        # 创建DynamicSliceModel类的实例
        module = DynamicSliceModel()
        # 创建一个形状为(1, 2)的随机张量x
        x = torch.rand(1, 2)
        # 定义一个包含Shape、Constant、Gather、Unsqueeze和Slice操作的操作列表ops_10
        ops_10 = [
            {"op_name": "Shape"},
            {"op_name": "Constant"},
            {"op_name": "Gather", "attributes": [{"name": "axis", "i": 0, "type": 2}]},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {
                "op_name": "Unsqueeze",
                "attributes": [{"name": "axes", "i": 0, "type": 7}],
            },
            {"op_name": "Constant"},
            {"op_name": "Slice", "attributes": []},
        ]
        # ops字典将opset版本映射到相应的操作列表
        ops = {10: ops_10}
        # 调用check_onnx_opsets_operator函数，验证DynamicSliceModel对x在opset版本10下的操作
        check_onnx_opsets_operator(
            module,
            x,
            ops,
            opset_versions=[10],
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )

        # 重新定义一个仅包含Slice操作的操作列表ops_10
        ops_10 = [
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Slice", "attributes": []},
        ]
        # ops字典将opset版本映射到相应的操作列表
        ops = {10: ops_10}
        # 再次调用check_onnx_opsets_operator函数，验证DynamicSliceModel对x在opset版本10下的操作
        check_onnx_opsets_operator(module, x, ops, opset_versions=[10])
    # 定义一个测试方法，用于测试 torch.flip 操作在不同条件下的行为
    def test_flip(self):
        # 定义一个继承自 Module 的子类 MyModule，重写了 forward 方法以实现反转张量 x
        class MyModule(Module):
            def forward(self, x):
                return torch.flip(x, dims=[0])

        # 定义一组操作列表 ops_10，每个操作都是常量操作
        ops_10 = [
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Slice", "attributes": []},
        ]
        # 将 ops_10 存入 ops 字典，使用版本号 10 作为键
        ops = {10: ops_10}
        
        # 导入 numpy 库，创建一个形状为 (2, 3) 的张量 x
        import numpy
        x = torch.tensor(numpy.arange(6.0).reshape(2, 3))
        
        # 调用函数 check_onnx_opsets_operator 测试 MyModule 在给定输入 x 和 ops 字典下的行为，使用 opset 版本为 10
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[10])

    # 定义一个测试方法，用于测试 torch.nn.Dropout 操作在不同模式下的行为
    def test_dropout(self):
        # 定义一个继承自 Module 的子类 MyModule，初始化时添加了一个 dropout 层
        class MyModule(Module):
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5)

            def forward(self, x):
                return self.dropout(x)

        # 创建一个形状为 (1, 2, 3) 的张量 x，使用标准正态分布随机数填充
        x = torch.randn(1, 2, 3)

        # 测试训练模式下的 Dropout 操作
        ops_training = [
            {
                "op_name": "Dropout",
                "attributes": [{"name": "ratio", "f": 0.5, "type": 1}],
            }
        ]
        # 存入 ops 字典，版本号分别为 9 和 10
        ops_training = {9: ops_training, 10: ops_training}
        check_onnx_opsets_operator(
            MyModule(),
            x,
            ops_training,
            opset_versions=[9, 10],
            training=torch.onnx.TrainingMode.TRAINING,
        )

        # 测试评估模式下的 Identity 操作
        ops_eval = [{"op_name": "Identity"}]
        # 存入 ops 字典，版本号分别为 9 和 10
        ops_eval = {9: ops_eval, 10: ops_eval}
        check_onnx_opsets_operator(
            MyModule(),
            x,
            ops_eval,
            opset_versions=[9, 10],
            training=torch.onnx.TrainingMode.EVAL,
        )

    # 定义一个测试方法，用于测试 torch.full 操作在不同 opset 版本下的行为
    def test_full(self):
        # 定义一个继承自 Module 的子类 MyModule，重写了 forward 方法以创建一个填充张量 x 的操作
        class MyModule(Module):
            def forward(self, x):
                return torch.full((3, 4), x)

        # 定义一组操作列表 ops
        ops = [
            {"op_name": "Constant"},
            {"op_name": "ConstantOfShape"},
            {"op_name": "Add"},
        ]
        # 存入 ops 字典，版本号分别为 9 和 10
        ops = {9: ops, 10: ops}
        
        # 创建一个值为 12.0 的张量 x
        x = torch.tensor(12.0)
        
        # 调用函数 check_onnx_opsets_operator 测试 MyModule 在给定输入 x 和 ops 字典下的行为，使用 opset 版本为 9 和 10
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])
    def test_affine_grid(self):
        # 定义一个测试方法，用于测试仿射网格生成功能
        
        class MyModule(Module):
            # 自定义模块，继承自 Module 类
            def __init__(self, align_corners):
                super().__init__()
                self.align_corners = align_corners

            def forward(self, theta, size):
                # 模块的前向传播方法，调用 torch.nn.functional.affine_grid 生成仿射网格
                return torch.nn.functional.affine_grid(
                    theta, size, align_corners=self.align_corners
                )

        opset_version = 20
        # 2D 操作集定义
        ops_2d = {
            opset_version: [
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Concat"},
                {"op_name": "AffineGrid"},
            ]
        }

        # 3D 操作集定义
        ops_3d = {
            opset_version: [
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Concat"},
                {"op_name": "AffineGrid"},
            ]
        }

        # 2D 仿射变换参数
        theta_2d = torch.empty(1, 2, 3, dtype=torch.double)
        size_2d = torch.Size([1, 1, 2, 2])

        # 3D 仿射变换参数
        theta_3d = torch.empty(1, 3, 4, dtype=torch.double)
        size_3d = torch.Size([1, 1, 2, 2, 2])

        # 使用 itertools.product 遍历输入和对齐角点的组合
        for inputs, align_corners in itertools.product(
            ((theta_2d, size_2d, ops_2d), (theta_3d, size_3d, ops_3d)),
            (True, False),
        ):
            theta, size, ops = inputs
            args = (
                theta,
                size,
            )

            # 调用自定义模块的方法，检查 ONNX 操作集的运算符
            check_onnx_opsets_operator(
                MyModule(align_corners=align_corners),
                args,
                ops,
                opset_versions=[opset_version],
                training=torch.onnx.TrainingMode.TRAINING,
            )

            # 调用自定义模块的方法，检查 ONNX 操作集的运算符（评估模式）
            check_onnx_opsets_operator(
                MyModule(align_corners=align_corners),
                args,
                ops,
                opset_versions=[opset_version],
                training=torch.onnx.TrainingMode.EVAL,
            )
    def test_grid_sample(self):
        # 定义一个测试函数 test_grid_sample，用于测试 grid_sample 函数的不同参数组合
        class MyModule(torch.nn.Module):
            # 定义一个简单的神经网络模块 MyModule，用于测试 grid_sample 函数
            def __init__(self, mode, padding_mode, align_corners):
                super().__init__()
                self.mode = mode
                self.padding_mode = padding_mode
                self.align_corners = align_corners

            def forward(self, x, grid):
                # 神经网络前向传播函数，调用 torch.nn.functional.grid_sample 函数
                return torch.nn.functional.grid_sample(
                    x,
                    grid,
                    mode=self.mode,  # 设置插值模式
                    padding_mode=self.padding_mode,  # 设置填充模式
                    align_corners=self.align_corners,  # 设置角点对齐方式
                )

        # 使用 itertools.product 生成参数组合进行测试
        for mode, padding_mode, align_corners, opset_version in itertools.product(
            ("bilinear", "nearest", "bicubic"),
            ("zeros", "border", "reflection"),
            (True, False),
            (16, 20),
        ):

            def test_eval_and_training(
                ops, opset_version, mode, padding_mode, align_corners, x_shape, grid
            ):
                # 定义测试函数 test_eval_and_training，用于测试模型在不同模式下的行为
                args = (
                    torch.randn(*x_shape),  # 随机生成输入数据 x
                    torch.randn(grid),  # 随机生成 grid 数据
                )
                check_onnx_opsets_operator(
                    MyModule(
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=align_corners,
                    ),
                    args,
                    ops,
                    opset_versions=[opset_version],
                    training=torch.onnx.TrainingMode.TRAINING,  # 设置训练模式
                )
                check_onnx_opsets_operator(
                    MyModule(
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=align_corners,
                    ),
                    args,
                    ops,
                    opset_versions=[opset_version],
                    training=torch.onnx.TrainingMode.EVAL,  # 设置评估模式
                )

            ops = {opset_version: [{"op_name": "GridSample"}]}
            # 如果 opset_version 是 20 并且 mode 不是 "bicubic"，则进行额外的测试
            # mode = convert_grid_sample_mode(mode) if opset_version == 20 else mode
            n, c, d_in, h_in, w_in, d_out, h_out, w_out = 1, 1, 2, 3, 2, 3, 2, 4
            test_eval_and_training(
                ops,
                opset_version,
                mode,
                padding_mode,
                align_corners,
                (n, c, h_in, w_in),  # 设置输入数据 x 的形状
                (n, h_out, w_out, 2),  # 设置 grid 的形状
            )
            # 如果 opset_version 是 20 并且 mode 不是 "bicubic"，则进行额外的测试
            if opset_version == 20 and mode != "bicubic":
                test_eval_and_training(
                    ops,
                    opset_version,
                    mode,
                    padding_mode,
                    align_corners,
                    (n, c, d_in, h_in, w_in),  # 设置输入数据 x 的形状
                    (n, d_out, h_out, w_out, 3),  # 设置 grid 的形状
                )
    # 定义一个测试方法，用于测试 flatten 操作
    def test_flatten(self):
        # 定义一个继承自 Module 的自定义模块 MyModule
        class MyModule(Module):
            # 定义模块的前向传播方法，对输入张量 x 进行 flatten 操作
            def forward(self, x):
                return torch.flatten(x)

        # 创建 MyModule 的实例
        module = MyModule()

        # 定义两种不同的操作序列，ops_0d 适用于零维张量，ops_1d 适用于一维张量
        ops_0d = [{"op_name": "Constant"}, {"op_name": "Reshape"}]
        ops_1d = [{"op_name": "Identity"}]

        # 遍历两种不同的张量形状：空列表（零维张量）和 [3]（一维张量）
        for shape in ([], [3]):
            # 根据当前形状创建一个随机张量 x
            x = torch.randn(shape)

            # 遍历两个 ONNX 操作集的版本：9 和 10
            for opset_version in [9, 10]:
                # 根据张量的维度选择相应的操作序列 ops
                ops = {opset_version: (ops_0d if len(shape) == 0 else ops_1d)}

                # 调用函数 check_onnx_opsets_operator 来检查 ONNX 操作的兼容性
                check_onnx_opsets_operator(
                    module, x, ops, opset_versions=[opset_version]
                )
# 如果当前脚本作为主程序运行（而不是作为模块导入），则执行以下代码块
if __name__ == "__main__":
    # 调用 common_utils 模块中的 run_tests 函数，用于执行测试
    common_utils.run_tests()
```
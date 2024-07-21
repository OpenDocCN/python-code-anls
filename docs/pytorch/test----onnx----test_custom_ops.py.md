# `.\pytorch\test\onnx\test_custom_ops.py`

```
# Owner(s): ["module: onnx"]

# 导入必要的模块和库
import onnx_test_common  # 导入自定义的ONNX测试共用函数
import pytorch_test_common  # 导入自定义的PyTorch测试共用函数

import torch  # 导入PyTorch库
import torch.utils.cpp_extension  # 导入PyTorch的C++扩展工具
from torch.onnx import symbolic_helper  # 导入ONNX符号化帮助函数
from torch.testing._internal import common_utils  # 导入PyTorch内部测试工具

# 定义一个测试类，继承自pytorch_test_common.ExportTestCase
class TestCustomAutogradFunction(pytorch_test_common.ExportTestCase):
    opset_version = 9  # 指定ONNX运算集版本号为9
    keep_initializers_as_inputs = False  # 不将初始化器保留为输入
    onnx_shape_inference = True  # 启用ONNX形状推断功能

    # 定义一个测试方法
    def test_symbolic(self):
        # 定义一个自定义的自动求导函数MyClip
        class MyClip(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, scalar):
                ctx.save_for_backward(input)  # 保存输入张量用于反向传播
                return input.clamp(min=scalar)  # 对输入张量进行截断操作

            @staticmethod
            def symbolic(g, input, scalar):
                return g.op("Clip", input, min_f=scalar)  # 使用ONNX符号化操作"Clip"

        # 定义一个包含自定义操作的模块MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.clip = MyClip.apply  # 使用自定义的MyClip函数作为模块的一部分

            def forward(self, x):
                h = self.clip(x, 2)  # 对输入x进行MyClip操作
                return h

        x = torch.randn(2, 3, 4, requires_grad=True)  # 创建一个随机张量x，需要梯度
        model = MyModule()  # 创建一个MyModule实例
        onnx_test_common.run_model_test(self, model, input_args=(x,))  # 运行模型测试

    # 定义另一个测试方法
    def test_register_op(self):
        # 定义自定义的自动求导函数MyClip
        class MyClip(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, scalar):
                ctx.save_for_backward(input)  # 保存输入张量用于反向传播
                return input.clamp(min=scalar)  # 对输入张量进行截断操作

        # 定义自定义的自动求导函数MyRelu
        class MyRelu(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)  # 保存输入张量用于反向传播
                return input.clamp(min=0)  # 对输入张量进行ReLU操作

        # 定义包含自定义操作的模块MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.clip = MyClip.apply  # 使用自定义的MyClip函数作为模块的一部分
                self.relu = MyRelu.apply  # 使用自定义的MyRelu函数作为模块的一部分

            def forward(self, x):
                h = self.clip(x, 2)  # 对输入x进行MyClip操作
                h = self.relu(h)  # 对h进行ReLU操作
                return h

        # 定义一个符号化Python操作函数symbolic_pythonop
        def symbolic_pythonop(ctx: torch.onnx.SymbolicContext, g, *args, **kwargs):
            n = ctx.cur_node  # 获取当前节点
            name = kwargs["name"]  # 获取操作的名称
            if name == "MyClip":
                return g.op("Clip", args[0], min_f=args[1], outputs=n.outputsSize())  # 使用ONNX操作"Clip"
            elif name == "MyRelu":
                return g.op("Relu", args[0], outputs=n.outputsSize())  # 使用ONNX操作"Relu"
            else:
                return symbolic_helper._unimplemented(
                    "prim::PythonOp", "unknown node kind: " + name
                )  # 抛出未实现异常

        from torch.onnx import register_custom_op_symbolic  # 导入注册自定义符号化操作的函数

        register_custom_op_symbolic("prim::PythonOp", symbolic_pythonop, 1)  # 注册自定义符号化操作"prim::PythonOp"

        x = torch.randn(2, 3, 4, requires_grad=True)  # 创建一个随机张量x，需要梯度
        model = MyModule()  # 创建一个MyModule实例
        onnx_test_common.run_model_test(self, model, input_args=(x,))  # 运行模型测试


# 定义另一个测试类，继承自pytorch_test_common.ExportTestCase
class TestExportAsContribOps(pytorch_test_common.ExportTestCase):
    opset_version = 14  # 指定ONNX运算集版本号为14
    keep_initializers_as_inputs = False  # 不将初始化器保留为输入
    onnx_shape_inference = True  # 启用ONNX形状推断功能
    # 定义一个测试函数，用于测试带有循环的自定义操作
    def test_contrib_op_with_loop(self):
        # 定义一个继承自 torch.nn.Module 的子类 M
        class M(torch.nn.Module):
            # 构造函数，初始化模块
            def __init__(self):
                super().__init__()
                # 实例化一个 GELU 激活函数对象，指定不使用近似方法
                self.gelu = torch.nn.GELU(approximate="none")

            # 前向传播函数
            def forward(self, x):
                # 初始化两个空列表 res 和 res2
                res = []
                res2 = []
                # 遍历输入张量 x 的第一个维度大小次数
                for i in range(x.size(0)):
                    # 如果 res 列表不为空
                    if len(res) > 0:
                        # 将 res[0] 添加到 res2 列表中
                        res2.append(res[0])
                    else:
                        # 否则，将 x[0] 经过 GELU 激活函数后的结果添加到 res2 列表中
                        res2.append(self.gelu(x[0]))
                    # 将 x[0] 添加到 res 列表中
                    res.append(x[0])
                # 返回堆叠后的张量 res 和 res2
                return torch.stack(res), torch.stack(res2)

        # 定义一个符号化的自定义 GELU 函数
        def symbolic_custom_gelu(g, input, approximate):
            # 使用 ONNX 的 g.op 方法创建一个名为 "com.microsoft::Gelu" 的操作，类型与 input 相同
            return g.op("com.microsoft::Gelu", input).setType(input.type())

        # 导入 torch.onnx 中的 register_custom_op_symbolic 函数
        from torch.onnx import register_custom_op_symbolic

        # 注册自定义操作符 "::gelu" 的符号化函数 symbolic_custom_gelu，该操作具有 1 个输入参数
        register_custom_op_symbolic("::gelu", symbolic_custom_gelu, 1)

        # 生成一个形状为 (3, 3, 4) 的随机张量 x，并指定需要梯度计算
        x = torch.randn(3, 3, 4, requires_grad=True)
        # 使用 torch.jit.script 方法将模型 M 实例化为一个脚本化模型
        model = torch.jit.script(M())
        # 运行模型测试，调用 onnx_test_common.run_model_test 方法，传入模型和输入参数 x
        onnx_test_common.run_model_test(self, model, input_args=(x,))
# 如果当前脚本作为主程序运行，则执行以下代码块
if __name__ == "__main__":
    # 调用 common_utils 模块中的 run_tests 函数，用于执行测试用例
    common_utils.run_tests()
```
# `.\pytorch\test\onnx\test_onnxscript_runtime.py`

```py
"""Test the support on onnxscript in PyTorch-ONNX converter with onnxruntime."""

# 从 typing 模块导入 List 类型
from typing import List

# 导入自定义的测试通用模块
import onnx_test_common
# 导入 onnxscript 库
import onnxscript
# 从 onnxscript.onnx_types 模块导入 FLOAT 类型
from onnxscript.onnx_types import FLOAT

# 导入 torch 库
import torch
# 从 torch.onnx._internal 模块导入 jit_utils
from torch.onnx._internal import jit_utils
# 从 torch.testing._internal 模块导入 common_utils
from torch.testing._internal import common_utils

# 定义一个测试类，继承自 onnx_test_common._TestONNXRuntime
class TestONNXScriptRuntime(onnx_test_common._TestONNXRuntime):
    # 设置 opset 版本为 15
    opset_version = 15

    # 定义测试方法 test_selu_from_onnxscript_example
    def test_selu_from_onnxscript_example(self):
        # 创建一个形状为 (1, 2, 3, 4) 的张量 x，要求梯度计算
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        # 创建一个 SELU 激活函数模型
        model = torch.nn.SELU()

        # 从 onnxscript.onnx_opset 模块导入 opset15，并重命名为 op
        from onnxscript.onnx_opset import opset15 as op

        # 创建一个自定义的 opset 对象，domain 为 "onnx-script"，版本号为 1
        custom_opset = onnxscript.values.Opset(domain="onnx-script", version=1)

        # 定义一个使用 onnxscript 脚本的函数 Selu
        @onnxscript.script(custom_opset)
        def Selu(
            X,
        ):
            # 设定 alpha 的默认值为 1.67326，被自动封装为常量
            alpha = 1.67326  # auto wrapped as Constants
            # 设定 gamma 的值为 1.0507
            gamma = 1.0507
            # 将 alpha 转换为与 X 相同类型的张量 alphaX
            alphaX = op.CastLike(alpha, X)
            # 将 gamma 转换为与 X 相同类型的张量 gammaX
            gammaX = op.CastLike(gamma, X)
            # 计算负部分 neg 和正部分 pos
            neg = gammaX * (alphaX * op.Exp(X) - alphaX)
            pos = gammaX * X
            zero = op.CastLike(0, X)
            # 根据 X 的值进行条件选择返回
            return op.Where(X <= zero, neg, pos)

        # 定义一个自定义的 selu 函数 custom_selu，接受图上下文 g 和输入张量 X
        def custom_selu(g: jit_utils.GraphContext, X):
            # 调用 g 的 onnxscript_op 方法，使用 Selu 函数处理 X，并设置返回类型为 X 的类型
            return g.onnxscript_op(Selu, X).setType(X.type())

        # 注册自定义操作的符号化名称为 "aten::selu"，符号化函数为 custom_selu，使用 opset_version 设定的 opset 版本号
        torch.onnx.register_custom_op_symbolic(
            symbolic_name="aten::selu",
            symbolic_fn=custom_selu,
            opset_version=self.opset_version,
        )
        # 运行测试，验证模型在输入 x 上的输出
        self.run_test(model, x)
    # 定义一个测试方法，用于测试层归一化功能
    def test_layer_norm(self):
        # 生成一个形状为 (2, 3) 的随机张量 x
        x = torch.randn(2, 3)
        # 生成一个形状为 (2, 3) 的随机张量 y
        y = torch.randn(2, 3)
        # 生成一个形状为 (2, 3) 的随机张量 z
        z = torch.randn(2, 3)

        # 定义一个名为 N 的内部类，继承自 torch.nn.Module
        class N(torch.nn.Module):
            # 构造方法，接受概率参数 prob
            def __init__(self, prob):
                super().__init__()
                # 使用给定的概率创建一个 Dropout 模块
                self.dropout = torch.nn.Dropout(prob)

            # 前向传播方法，接受输入张量 x，对其进行 dropout 操作并返回结果
            def forward(self, x):
                return self.dropout(x)

        # 定义一个名为 M 的内部类，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 构造方法，接受层数 num_layers 参数
            def __init__(self, num_layers):
                super().__init__()
                # 记录传入的层数
                self.num_layers = num_layers
                # 创建一个包含 num_layers 个 LayerNorm 模块的 ModuleList
                self.lns = torch.nn.ModuleList(
                    [torch.nn.LayerNorm(3, eps=i) for i in range(num_layers)]
                )
                # 创建一个 CELU 激活函数模块，参数为 1.0
                self.celu1 = torch.nn.CELU(1.0)
                # 创建一个 CELU 激活函数模块，参数为 2.0
                self.celu2 = torch.nn.CELU(2.0)
                # 创建一个 N 类的实例，设置 dropout 概率为 0.5
                self.dropout = N(0.5)

            # 前向传播方法，接受输入张量 x, y, z，对其进行一系列操作并返回结果
            def forward(self, x, y, z):
                # 对 x 使用 celu1 激活函数
                res1 = self.celu1(x)
                # 对 y 使用 celu2 激活函数
                res2 = self.celu2(y)
                # 遍历 self.lns 中的每个 LayerNorm 模块，对 z 执行归一化操作
                for ln in self.lns:
                    z = ln(z)
                # 返回 res1 与 res2 的和，以及对 z 执行 dropout 操作的结果
                return res1 + res2, self.dropout(z)

        # 创建一个 M 类的实例，层数为 3
        model = M(3)

        # 导入自定义操作集 opset15
        from onnxscript.onnx_opset import opset15 as op

        # 创建一个自定义的 opset，指定领域为 "onnxscript"，版本为 1
        custom_opset = onnxscript.values.Opset(domain="onnxscript", version=1)

        # 定义一个装饰器函数，用于创建 layer_norm 函数
        @onnxscript.script(custom_opset)
        def layer_norm(
            X, axes: List[int], weight: FLOAT[...], bias: FLOAT[...], eps: float
        ):
            # 计算 X 沿指定轴的均值
            mean = op.ReduceMean(X, axes=axes)
            # 计算 X 与均值的差
            D = X - mean  # op.Sub(X, mean)
            # 计算 D 的平方
            DD = D * D  # op.Mul(D, D)
            # 计算 DD 沿指定轴的平均值，得到方差
            var = op.ReduceMean(DD, axes=axes)
            # 将方差加上 eps，得到带 epsilon 的方差
            vareps = var + eps  # op.Add(var, eps)
            # 计算标准差
            stddev = op.Sqrt(vareps)
            # 计算标准差的倒数
            invstddev = op.Reciprocal(stddev)
            # 对 D 应用标准化，并与权重 weight 类型相匹配
            normalized = D * invstddev  # op.Mul(D, invstddev)
            normalizedw = op.CastLike(
                normalized, weight
            )  # 若忽略此 Op，可能会导致类型问题
            # 对标准化后的结果应用权重和偏置
            normalizedscaled = normalizedw * weight  # op.Mul(normalized, weight)
            return normalizedscaled + bias

        # 定义一个符号帮助器函数，用于解析和处理输入参数
        @torch.onnx.symbolic_helper.parse_args("v", "is", "v", "v", "f", "none")
        def custom_layer_norm(
            g, input, normalized_shape, weight, bias, eps, cudnn_enable
        ):
            # 创建轴列表，用于标识要归一化的维度
            axes = [-i for i in range(len(normalized_shape), 0, -1)]
            # 调用自定义 op，在 ONNX 图中执行 layer_norm 操作
            return g.onnxscript_op(
                layer_norm, input, weight, bias, axes_i=axes, eps_f=eps
            ).setType(input.type())

        # 注册自定义操作符号化函数
        torch.onnx.register_custom_op_symbolic(
            symbolic_name="aten::layer_norm",
            symbolic_fn=custom_layer_norm,
            opset_version=self.opset_version,
        )

        # 运行测试方法，测试模型在输入 (x, y, z) 上的输出
        self.run_test(model, (x, y, z))
# 如果当前脚本被直接执行（而不是被导入作为模块），则执行以下代码块
if __name__ == "__main__":
    # 调用 common_utils 模块中的 run_tests 函数，用于执行测试用例
    common_utils.run_tests()
```
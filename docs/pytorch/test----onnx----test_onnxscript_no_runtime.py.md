# `.\pytorch\test\onnx\test_onnxscript_no_runtime.py`

```py
"""Test the support on onnxscript in PyTorch-ONNX converter."""
# 导入所需的模块和库
import io
from typing import List

# 导入 ONNX 相关的模块和类
import onnx
import onnxscript
from onnxscript.onnx_types import FLOAT

# 导入 PyTorch 相关的模块和类
import torch
from torch.onnx._internal import jit_utils
from torch.testing._internal import common_utils


class TestONNXScriptExport(common_utils.TestCase):
    # 设置 opset 版本号为 15
    opset_version = 15

    def test_loop_registration(self):
        # 测试 torch/onnx/utils.py 中 _find_onnxscript_op 函数的控制流功能，
        # 该函数具有递归逻辑，用于处理模型 proto 中带有子图的每个节点
        class NestedLoopsModel(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.selu = torch.nn.SELU()

            @torch.jit.script_method
            def forward(self, x):
                y = x
                # 使用循环处理输入 x 的第 3 维
                for i in range(x.size(3)):
                    if i == 0:
                        y = self.selu(x)
                    else:
                        y += i
                return y

        # 创建 NestedLoopsModel 的实例
        model = NestedLoopsModel()
        # 创建输入张量
        inputs = torch.zeros(1, 2, 3, 4)

        # 导入 onnxscript 中的 opset15
        from onnxscript.onnx_opset import opset15 as op

        # 定义自定义 opset 为 domain="onnx-script", version=2
        custom_opset = onnxscript.values.Opset(domain="onnx-script", version=2)

        # 定义名为 Selu 的自定义操作
        @onnxscript.script(custom_opset)
        def Selu(X):
            alpha = 1.6732632423543772848170429916717
            gamma = 1.0507009873554804934193349852946
            alphaX = op.CastLike(alpha, X)
            gammaX = op.CastLike(gamma, X)
            neg = gammaX * (alphaX * op.Exp(X) - alphaX)
            pos = gammaX * X
            zero = op.CastLike(0, X)
            return op.Where(X <= zero, neg, pos)

        # 定义自定义 selu 函数
        def custom_selu(g, X):
            # 打印信息，指出 custom_selu 正在使用
            print("custom_selu is used!")
            # 调用 g.onnxscript_op 执行 Selu 操作，并设置类型为 X.type()
            return g.onnxscript_op(Selu, X).setType(X.type())

        # 注册自定义操作符号化函数 "aten::selu"
        torch.onnx.register_custom_op_symbolic(
            symbolic_name="aten::selu",
            symbolic_fn=custom_selu,
            opset_version=15,
        )

        # 创建保存模型的 BytesIO 对象
        saved_model = io.BytesIO()
        # 导出模型到 ONNX 格式，使用 opset_version=15
        torch.onnx.export(
            torch.jit.script(model), inputs, f=saved_model, opset_version=15
        )
        # 加载导出的 ONNX 模型
        loop_selu_proto = onnx.load(io.BytesIO(saved_model.getvalue()))
        # 断言模型中函数的数量为 1
        self.assertEqual(len(loop_selu_proto.functions), 1)
```
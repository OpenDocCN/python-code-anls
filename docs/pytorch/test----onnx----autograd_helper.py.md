# `.\pytorch\test\onnx\autograd_helper.py`

```
# Owner(s): ["module: onnx"]
# 导入 PyTorch 库
import torch


# Autograd funtion that is a replica of the autograd funtion in
# test_utility_funs.py (test_autograd_module_name)
# 自定义的 Autograd 函数，模拟 test_utility_funs.py 中的 autograd 函数（test_autograd_module_name）
class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 在 forward 方法中保存输入张量，以便在反向传播时使用
        ctx.save_for_backward(input)
        # 返回输入张量的非负部分
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # 从上下文中获取保存的输入张量
        (input,) = ctx.saved_tensors
        # 克隆梯度输出，用于计算输入的梯度
        grad_input = grad_output.clone()
        # 将小于零的输入位置的梯度置为零
        grad_input[input < 0] = 0
        # 返回输入的梯度
        return grad_input
```
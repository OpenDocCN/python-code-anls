# `.\pytorch\benchmarks\fastrnns\scratch.py`

```py
import torch

@torch.jit.script
def fn(x, scale, shift):
    # 定义一个脚本化函数，实现输入 x 的按比例缩放和平移
    return scale * x / shift

@torch.jit.script
def recurrent(x, scale, shift):
    # 定义一个脚本化函数，进行重复调用 fn 函数的循环操作
    y = x
    for i in range(100):
        y = fn(y, scale, shift)
    return y

x = torch.randn(2, 2, device="cuda")
scale = torch.randn(2, 2, device="cuda", requires_grad=True)
shift = torch.randn(2, 2, device="cuda", requires_grad=True)
inputs = [x, scale, shift]

out = recurrent(x, scale, shift)
recurrent.graph_for(x, scale, shift)

import torch

@torch.jit.script
def recurrent_scaleshift(x, scale, shift):
    # 定义一个脚本化函数，通过乘以 scale 和加上 shift 实现简单的递归操作
    y = x
    for i in range(64):
        y = scale * y + shift
    return y

x = torch.randn(2, 2, device="cuda")
scale = torch.randn(2, 2, device="cuda", requires_grad=True)
shift = torch.randn(2, 2, device="cuda", requires_grad=True)
inputs = [x, scale, shift]
out = recurrent_scaleshift(x, scale, shift)
recurrent_scaleshift.graph_for(x, scale, shift)

import torch

x = torch.tensor([])
x.requires_grad = True
x.mean().backward()  # 执行张量 x 的均值操作并反向传播梯度
x = x.cuda()  # 将张量 x 移动到 CUDA 设备上
x.mean().backward()  # 执行 CUDA 设备上的张量 x 的均值操作并反向传播梯度
```
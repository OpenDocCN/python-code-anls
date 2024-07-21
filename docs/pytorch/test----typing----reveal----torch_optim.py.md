# `.\pytorch\test\typing\reveal\torch_optim.py`

```py
# 导入 torch 库，用于神经网络和优化器等机器学习任务
import torch

# 定义函数 foo，接受一个 torch.optim.Optimizer 类型的参数 opt，并且没有返回值
def foo(opt: torch.optim.Optimizer) -> None:
    # 调用优化器对象的 zero_grad 方法，将模型参数的梯度清零
    opt.zero_grad()

# 创建一个 Adagrad 优化器对象 opt_adagrad，优化参数为一个值为 0.0 的张量
opt_adagrad = torch.optim.Adagrad([torch.tensor(0.0)])
# 打印 opt_adagrad 对象的类型，预期类型为 Adagrad 类型
reveal_type(opt_adagrad)  # E: {Adagrad}
# 调用 foo 函数，传入 Adagrad 优化器对象 opt_adagrad
foo(opt_adagrad)

# 创建一个 Adam 优化器对象 opt_adam，优化参数为一个值为 0.0 的张量，学习率为 0.01，eps 为 1e-6
opt_adam = torch.optim.Adam([torch.tensor(0.0)], lr=1e-2, eps=1e-6)
# 打印 opt_adam 对象的类型，预期类型为 Adam 类型
reveal_type(opt_adam)  # E: {Adam}
# 调用 foo 函数，传入 Adam 优化器对象 opt_adam
foo(opt_adam)
```
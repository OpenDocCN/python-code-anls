# `.\pytorch\functorch\examples\compilation\fuse_module.py`

```
# 导入计时模块
import timeit

# 导入 PyTorch 库
import torch
import torch.nn as nn

# 导入自定义的编译函数和模块
from functorch.compile import compiled_module, tvm_compile


# 定义一个无操作函数，接受两个参数但返回第一个参数
def nop(f, _):
    return f


# 初始化前向和反向编译器，用于编译模块
fw_compiler = tvm_compile(target="llvm", tuning_logfile="fw_keops")
bw_compiler = tvm_compile(target="llvm", tuning_logfile="bw_keops")

# 将编译器替换为 nop 函数，实际上取消了编译的操作
fw_compiler = nop
bw_compiler = nop


# 定义运行函数，执行模块的前向和反向传播，并返回输出和梯度
def run(mod, input):
    out = mod(input)
    out.sum().backward()
    grads = [p.grad for p in mod.parameters()]
    return (out, *grads)


# 定义一个简单的神经网络模块
class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(1))
        self.register_buffer("buf", torch.randn(1))

    def forward(self, x):
        return (self.param * x + self.buf).sum(dim=0)


# 生成一个随机输入
input = torch.randn(1)
# 创建 Foo 类的实例 mod
mod = Foo()
# 使用编译函数编译模块 mod，得到 compiled_mod
compiled_mod = compiled_module(mod, fw_compiler, bw_compiler)

# 对比未编译和已编译模块的输出结果是否一致
for a, b in zip(run(mod, input), run(compiled_mod, input)):
    torch.testing.assert_close(a, b)

# 对未编译模块进行前向和反向传播，并更新参数
out = mod(input)
out.sum().backward()
mod.param.data -= mod.param.grad

# 对已编译模块进行前向和反向传播，并更新参数
compiled_mod.orig_module.param.data -= compiled_mod.orig_module.param.grad
compiled_mod.orig_module.param.grad = None

# 再次对比未编译和已编译模块的输出结果是否一致
for a, b in zip(run(mod, input), run(compiled_mod, input)):
    torch.testing.assert_close(a, b)

# 多次运行未编译和已编译模块的计算时间，并打印结果
for _ in range(5):
    i = 10000
    t = timeit.Timer("mod(input)", globals=globals()).timeit(10000)
    print(f"eager {t/i*1e6}")
    t = timeit.Timer("compiled_mod(input)", globals=globals()).timeit(10000)
    print(f"compiled {t/i*1e6}")
```
# `.\pytorch\torch\testing\_internal\static_module.py`

```
# mypy: allow-untyped-defs
# Owner(s): ["module: unknown"]

# 导入 PyTorch 库
import torch

# 定义一个静态模块类
class StaticModule:
    def __init__(self, scripted):
        # 检查传入的对象是否是 nn.Module 的实例
        if hasattr(scripted, "_c"):
            # 如果对象有 _c 属性，则使用 torch._C._jit_to_static_module 方法创建静态模块
            self.static_module = torch._C._jit_to_static_module(scripted._c)
        else:
            # 否则，使用 torch._C._jit_to_static_module 方法创建静态模块，传入图形对象
            self.static_module = torch._C._jit_to_static_module(scripted.graph)

    def __call__(self, *args, **kwargs):
        # 实现 __call__ 方法，使对象可调用
        return self.static_module(*args, **kwargs)

    def benchmark(self, args, kwargs, warmup_runs, main_runs):
        # 调用静态模块的 benchmark 方法进行性能测试
        self.static_module.benchmark(args, kwargs, warmup_runs, main_runs)

    def runAsync(self, args, kwargs):
        # 调用静态模块的 runAsync 方法以异步方式运行
        return self.static_module.runAsync(args, kwargs)

    def benchmark_individual_ops(self, args, kwargs, warmup_runs, main_runs):
        # 对静态模块中的各个操作进行单独的性能测试
        return self.static_module.benchmark_individual_ops(
            args, kwargs, warmup_runs, main_runs
        )
```
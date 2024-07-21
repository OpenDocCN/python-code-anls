# `.\pytorch\torch\_strobelight\examples\compile_time_profile_example.py`

```py
# 引入 torch 库
import torch

# 从 torch._strobelight.compile_time_profiler 中导入 StrobelightCompileTimeProfiler 类
from torch._strobelight.compile_time_profiler import StrobelightCompileTimeProfiler

# 主程序入口，判断是否在主程序中执行
if __name__ == "__main__":
    # 启用 Strobelight 编译时性能分析器
    StrobelightCompileTimeProfiler.enable()

    # 定义一个简单的函数 fn，计算 x * y + z
    def fn(x, y, z):
        return x * y + z

    # 使用 @torch.compile() 装饰器定义编译函数 work
    @torch.compile()
    def work(n):
        # 嵌套循环，执行 fn 函数调用
        for i in range(3):
            for j in range(5):
                fn(torch.rand(n, n), torch.rand(n, n), torch.rand(n, n))

    # 迭代三次，每次重置 _dynamo 动态编译器状态
    for i in range(3):
        torch._dynamo.reset()
        # 调用 work 函数
        work(i)
```
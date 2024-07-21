# `.\pytorch\torch\utils\_strobelight\examples\cli_function_profiler_example.py`

```
# 引入torch模块，用于深度学习任务
import torch

# 从torch.utils._strobelight.cli_function_profiler模块中导入strobelight和StrobelightCLIFunctionProfiler类
from torch.utils._strobelight.cli_function_profiler import (
    strobelight,
    StrobelightCLIFunctionProfiler,
)

# 程序的入口点，当作为主程序执行时执行以下内容
if __name__ == "__main__":

    # 定义一个函数fn，接受三个参数x, y, z，并返回它们的乘积加上z的结果
    def fn(x, y, z):
        return x * y + z

    # 使用strobelight装饰器，配置采样频率为每10000次函数调用采样一次，不在错误处停止采样
    # 同时使用torch.compile()编译函数
    @strobelight(sample_each=10000, stop_at_error=False)
    @torch.compile()
    def work():
        # 循环10次
        for i in range(10):
            # 重置torch._dynamo状态
            torch._dynamo.reset()
            # 再次循环5次
            for j in range(5):
                # 重置torch._dynamo状态
                torch._dynamo.reset()
                # 调用函数fn，传入随机生成的大小为j*j的张量作为参数
                fn(torch.rand(j, j), torch.rand(j, j), torch.rand(j, j))

    # 执行work函数
    work()

    # 创建一个StrobelightCLIFunctionProfiler实例，不在错误处停止采样
    profiler = StrobelightCLIFunctionProfiler(stop_at_error=False)

    # 使用strobelight装饰器，传入profiler实例和样本标签["something", "another"]
    @strobelight(profiler, sample_tags=["something", "another"])
    def work2():
        sum = 0
        # 循环1亿次
        for i in range(100000000):
            # 累加操作
            sum += 1

    # 执行work2函数
    work2()
```
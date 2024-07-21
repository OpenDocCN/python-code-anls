# `.\pytorch\torch\_strobelight\examples\cli_function_profiler_example.py`

```
# 引入 torch 库
import torch

# 从 torch._strobelight.cli_function_profiler 中导入所需的符号
from torch._strobelight.cli_function_profiler import (
    strobelight,
    StrobelightCLIFunctionProfiler,
)

# 当脚本作为主程序运行时执行以下代码
if __name__ == "__main__":

    # 定义一个函数 fn，接受三个参数并返回它们的乘积加上第三个参数
    def fn(x, y, z):
        return x * y + z

    # 使用 strobelight 装饰器，配置默认的分析器或可选的分析参数
    @strobelight(sample_each=10000, stop_at_error=False)
    # 使用 torch.compile() 装饰器
    def work():
        # 循环 10 次
        for i in range(10):
            # 重置 torch._dynamo 状态
            torch._dynamo.reset()
            # 内部循环 5 次
            for j in range(5):
                # 再次重置 torch._dynamo 状态
                torch._dynamo.reset()
                # 调用函数 fn，并传入随机生成的大小为 (j, j) 的张量作为参数
                fn(torch.rand(j, j), torch.rand(j, j), torch.rand(j, j))

    # 执行 work 函数
    work()

    # 创建 StrobelightCLIFunctionProfiler 的实例，停止在错误发生时
    profiler = StrobelightCLIFunctionProfiler(stop_at_error=False)

    # 使用指定的分析器实例和样本标签作为参数来调用 strobelight 装饰器
    @strobelight(profiler, sample_tags=["something", "another"])
    def work2():
        sum = 0
        # 循环一亿次
        for i in range(100000000):
            # 累加操作
            sum += 1

    # 执行 work2 函数
    work2()
```
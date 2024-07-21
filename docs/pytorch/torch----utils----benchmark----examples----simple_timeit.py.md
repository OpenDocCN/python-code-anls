# `.\pytorch\torch\utils\benchmark\examples\simple_timeit.py`

```py
# mypy: allow-untyped-defs
"""Trivial use of Timer API:

$ python -m examples.simple_timeit
"""

# 导入 torch 库
import torch

# 导入 benchmark_utils 模块，用于性能基准测试
import torch.utils.benchmark as benchmark_utils


# 主函数
def main():
    # 创建一个 Timer 对象，用于测量执行语句的时间
    timer = benchmark_utils.Timer(
        stmt="x + y",  # 要测量执行时间的语句
        globals={"x": torch.ones((4, 8)), "y": torch.ones((1, 8))},  # 提供给语句的全局变量
        label="Broadcasting add (4x8)",  # 标签，用于标识此次性能测试
    )

    # 循环运行性能测试，打印结果
    for i in range(3):
        print(f"Run: {i}\n{'-' * 40}")
        # 执行 timeit 方法，测量语句的执行时间，并打印结果
        print(f"timeit:\n{timer.timeit(10000)}\n")
        # 执行 blocked_autorange 方法，测量语句的执行时间，并打印结果
        print(f"autorange:\n{timer.blocked_autorange()}\n\n")


# 如果当前脚本作为主程序运行，则执行主函数
if __name__ == "__main__":
    main()
```
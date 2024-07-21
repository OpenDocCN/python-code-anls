# `.\pytorch\benchmarks\dynamo\microbenchmarks\dynamo_microbenchmarks.py`

```
import cProfile
import pstats
import timeit

import torch


@torch.compile(backend="eager", fullgraph=True)
# 定义一个使用 Torch 编译器修饰的函数，启用 eager 模式，并且生成完整的计算图
def symbolic_convert_overhead_stress_test(x, y, n):
    # 在循环中执行变量交换，循环次数为 n
    while n > 0:
        n -= 1
        x, y = y, x
    # 返回 x 和 y 的和
    return x + y


def main():
    # 定义一个函数 fn，用于执行测试
    def fn():
        # 重置 Torch 的动态图计算状态
        torch._dynamo.reset()
        # 调用 symbolic_convert_overhead_stress_test 函数，传入参数 x, y 和 100000
        symbolic_convert_overhead_stress_test(x, y, 100000)

    # 生成两个长度为 16 的随机张量 x 和 y
    x = torch.randn(16)
    y = torch.randn(16)
    # 测量执行 fn 函数一次的时间，重复 3 次，取最小值
    t = min(timeit.repeat(fn, number=1, repeat=3))
    # 打印测试结果，显示函数执行时间
    print(f"symbolic_convert_overhead_stress_test: {t:.1f}s")


def profile():
    # 生成两个长度为 16 的随机张量 x 和 y
    x = torch.randn(16)
    y = torch.randn(16)
    # 重置 Torch 的动态图计算状态
    torch._dynamo.reset()
    # 创建 cProfile 的 Profile 对象
    pr = cProfile.Profile()
    # 启用性能分析
    pr.enable()
    # 调用 symbolic_convert_overhead_stress_test 函数，传入参数 x, y 和 33000
    # 这个数值大约可以抵消 cProfile 的开销
    symbolic_convert_overhead_stress_test(x, y, 33000)
    # 禁用性能分析
    pr.disable()
    # 创建 pstats 的 Stats 对象，载入性能分析数据
    ps = pstats.Stats(pr)
    # 将性能分析数据保存到文件 dynamo_microbenchmarks.prof
    ps.dump_stats("dynamo_microbenchmarks.prof")
    # 打印提示信息，使用 snakeviz 工具查看性能分析结果文件
    print("snakeviz dynamo_microbenchmarks.prof")


if __name__ == "__main__":
    # 执行主函数 main
    main()
    # 执行性能分析函数 profile
    profile()
```
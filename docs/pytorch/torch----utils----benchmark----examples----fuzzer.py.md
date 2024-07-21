# `.\pytorch\torch\utils\benchmark\examples\fuzzer.py`

```
# 导入 sys 模块，用于处理系统相关的操作
import sys

# 导入 torch.utils.benchmark 中的 benchmark_utils，用于性能基准测试
import torch.utils.benchmark as benchmark_utils

# 主函数定义
def main():
    # 创建一个 Fuzzer 对象 add_fuzzer，用于生成模糊测试参数
    add_fuzzer = benchmark_utils.Fuzzer(
        # 定义模糊测试的参数集合
        parameters=[
            [
                # 定义模糊参数 k0, k1, k2，指定最小值、最大值和分布类型
                benchmark_utils.FuzzedParameter(
                    name=f"k{i}",
                    minval=16,
                    maxval=16 * 1024,
                    distribution="loguniform",
                ) for i in range(3)
            ],
            # 定义模糊参数 d，指定分布概率
            benchmark_utils.FuzzedParameter(
                name="d",
                distribution={2: 0.6, 3: 0.4},
            ),
        ],
        # 定义模糊测试的张量集合
        tensors=[
            [
                # 定义模糊张量 x 和 y，指定名称、大小、维度参数、连续概率和元素数量范围
                benchmark_utils.FuzzedTensor(
                    name=name,
                    size=("k0", "k1", "k2"),
                    dim_parameter="d",
                    probability_contiguous=0.75,
                    min_elements=64 * 1024,
                    max_elements=128 * 1024,
                ) for name in ("x", "y")
            ],
        ],
        # 指定随机种子
        seed=0,
    )

    # 设置测试次数
    n = 250
    # 存储测量结果的列表
    measurements = []
    # 迭代执行模糊测试，获取张量、张量属性和元数据
    for i, (tensors, tensor_properties, _) in enumerate(add_fuzzer.take(n=n)):
        # 获取张量 x 和 y
        x, x_order = tensors["x"], str(tensor_properties["x"]["order"])
        y, y_order = tensors["y"], str(tensor_properties["y"]["order"])
        # 获取张量 x 的形状描述
        shape = ", ".join(tuple(f'{i:>4}' for i in x.shape))

        # 构建测量描述字符串，包括元素数量、形状、顺序等信息
        description = "".join([
            f"{x.numel():>7} | {shape:<16} | ",
            f"{'contiguous' if x.is_contiguous() else x_order:<12} | ",
            f"{'contiguous' if y.is_contiguous() else y_order:<12} | ",
        ])

        # 创建一个 Timer 对象 timer，用于执行计时任务
        timer = benchmark_utils.Timer(
            stmt="x + y",  # 指定要计时的语句
            globals=tensors,  # 指定全局变量
            description=description,  # 指定描述信息
        )

        # 执行并记录计时结果，使用 blocked_autorange 方法进行自动调整计时
        measurements.append(timer.blocked_autorange(min_run_time=0.1))
        # 添加元数据信息到测量结果中，包括张量元素数量
        measurements[-1].metadata = {"numel": x.numel()}
        # 输出当前进度信息
        print(f"\r{i + 1} / {n}", end="")
        sys.stdout.flush()
    print()

    # 输出额外的信息，用于美化输出
    print(f"Average attempts per valid config: {1. / (1. - add_fuzzer.rejection_rate):.1f}")

    # 定义一个函数 time_fn，用于按照计时结果排序
    def time_fn(m):
        return m.median / m.metadata["numel"]

    # 根据时间函数 time_fn 对测量结果进行排序
    measurements.sort(key=time_fn)

    # 定义输出模板，用于格式化输出最佳和最差结果
    template = f"{{:>6}}{' ' * 19}Size    Shape{' ' * 13}X order        Y order\n{'-' * 80}"

    # 输出最佳结果的标题和内容
    print(template.format("Best:"))
    for m in measurements[:15]:
        print(f"{time_fn(m) * 1e9:>4.1f} ns / element     {m.description}")

    # 输出最差结果的标题和内容
    print("\n" + template.format("Worst:"))
    for m in measurements[-15:]:
        print(f"{time_fn(m) * 1e9:>4.1f} ns / element     {m.description}")


# 如果运行为主程序，则执行主函数 main()
if __name__ == "__main__":
    main()
```
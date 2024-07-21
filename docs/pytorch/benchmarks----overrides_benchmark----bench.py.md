# `.\pytorch\benchmarks\overrides_benchmark\bench.py`

```py
# 导入 argparse 模块，用于解析命令行参数
import argparse
# 导入 time 模块，用于测量代码执行时间

# 从 common 模块中导入 SubTensor, SubWithTorchFunction, WithTorchFunction 类
from common import SubTensor, SubWithTorchFunction, WithTorchFunction

# 导入 torch 模块
import torch

# 设置全局常量 NUM_REPEATS 和 NUM_REPEAT_OF_REPEATS
NUM_REPEATS = 1000
NUM_REPEAT_OF_REPEATS = 1000

# 定义用于比较两个张量加法性能的函数 bench
def bench(t1, t2):
    # 存储每次测量的时间
    bench_times = []
    # 执行 NUM_REPEAT_OF_REPEATS 次重复测量
    for _ in range(NUM_REPEAT_OF_REPEATS):
        # 记录每次测量开始时间
        time_start = time.time()
        # 执行 NUM_REPEATS 次张量加法操作
        for _ in range(NUM_REPEATS):
            torch.add(t1, t2)
        # 计算本次测量的总耗时，并存入 bench_times 列表
        bench_times.append(time.time() - time_start)

    # 取最小的测量耗时，并将其转换为毫秒
    bench_time = float(torch.min(torch.tensor(bench_times))) / 1000
    # 计算测量耗时的标准差，并将其转换为毫秒
    bench_std = float(torch.std(torch.tensor(bench_times))) / 1000

    return bench_time, bench_std

# 主函数入口
def main():
    global NUM_REPEATS
    global NUM_REPEAT_OF_REPEATS

    # 创建 argparse.ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser(
        description="Run the __torch_function__ benchmarks."
    )
    # 添加命令行参数 --nreps 和 -n，用于指定每次测量重复执行的次数
    parser.add_argument(
        "--nreps",
        "-n",
        type=int,
        default=NUM_REPEATS,
        help="The number of repeats for one measurement.",
    )
    # 添加命令行参数 --nrepreps 和 -m，用于指定测量的总次数
    parser.add_argument(
        "--nrepreps",
        "-m",
        type=int,
        default=NUM_REPEAT_OF_REPEATS,
        help="The number of measurements.",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 更新全局常量 NUM_REPEATS 和 NUM_REPEAT_OF_REPEATS 的值为命令行参数指定的值
    NUM_REPEATS = args.nreps
    NUM_REPEAT_OF_REPEATS = args.nrepreps

    # 定义要测试的数据类型列表
    types = torch.tensor, SubTensor, WithTorchFunction, SubWithTorchFunction

    # 遍历数据类型列表
    for t in types:
        # 创建两个张量 tensor_1 和 tensor_2，数据类型为当前遍历到的 t 类型
        tensor_1 = t([1.0])
        tensor_2 = t([2.0])

        # 执行 bench 函数，获取最小耗时 bench_min 和标准差 bench_std
        bench_min, bench_std = bench(tensor_1, tensor_2)
        # 打印当前数据类型 t 的最小耗时和标准差，单位为微秒（us）
        print(
            f"Type {t.__name__} had a minimum time of {10**6 * bench_min} us"
            f" and a standard deviation of {(10**6) * bench_std} us."
        )

# 判断是否在主程序入口运行当前脚本
if __name__ == "__main__":
    # 调用主函数 main
    main()
```
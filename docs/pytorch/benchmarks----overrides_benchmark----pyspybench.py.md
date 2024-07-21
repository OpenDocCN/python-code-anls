# `.\pytorch\benchmarks\overrides_benchmark\pyspybench.py`

```
# 导入 argparse 模块，用于解析命令行参数
import argparse

# 从 common 模块导入 SubTensor, SubWithTorchFunction, WithTorchFunction 类
from common import SubTensor, SubWithTorchFunction, WithTorchFunction  # noqa: F401

# 导入 torch 模块
import torch

# 将 torch.tensor 赋值给 Tensor 变量，以便简化后续代码中的使用
Tensor = torch.tensor

# 设置循环次数的常量
NUM_REPEATS = 1000000

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建 ArgumentParser 对象，用于解析命令行参数的描述
    parser = argparse.ArgumentParser(
        description="Run the torch.add for a given class a given number of times."
    )
    
    # 添加命令行参数，指定要 benchmark 的 Tensor 类名
    parser.add_argument(
        "tensor_class", metavar="TensorClass", type=str, help="The class to benchmark."
    )
    
    # 添加可选参数，指定重复次数，默认值为 NUM_REPEATS
    parser.add_argument(
        "--nreps", "-n", type=int, default=NUM_REPEATS, help="The number of repeats."
    )
    
    # 解析命令行参数
    args = parser.parse_args()

    # 根据命令行参数获取 Tensor 类名，并通过 globals() 函数获取对应的类对象
    TensorClass = globals()[args.tensor_class]
    
    # 更新 NUM_REPEATS 为命令行指定的重复次数
    NUM_REPEATS = args.nreps

    # 创建两个 TensorClass 类的实例 t1 和 t2，分别初始化为 [1.0] 和 [2.0]
    t1 = TensorClass([1.0])
    t2 = TensorClass([2.0])

    # 循环执行 NUM_REPEATS 次 torch.add(t1, t2)
    for _ in range(NUM_REPEATS):
        torch.add(t1, t2)
```
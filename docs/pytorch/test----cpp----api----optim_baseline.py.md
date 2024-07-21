# `.\pytorch\test\cpp\api\optim_baseline.py`

```
"""Script to generate baseline values from PyTorch optimization algorithms"""

# 引入必要的库
import argparse  # 用于解析命令行参数
import math  # 数学函数库
import sys  # 系统相关的功能

import torch  # PyTorch库
import torch.optim  # PyTorch优化器模块


HEADER = """
#include <torch/types.h>

#include <vector>

namespace expected_parameters {
"""

FOOTER = "} // namespace expected_parameters"

PARAMETERS = "inline std::vector<std::vector<torch::Tensor>> {}() {"

# 定义各种优化器的工厂函数
OPTIMIZERS = {
    "LBFGS": lambda p: torch.optim.LBFGS(p, 1.0),
    "LBFGS_with_line_search": lambda p: torch.optim.LBFGS(
        p, 1.0, line_search_fn="strong_wolfe"
    ),
    "Adam": lambda p: torch.optim.Adam(p, 1.0),
    "Adam_with_weight_decay": lambda p: torch.optim.Adam(p, 1.0, weight_decay=1e-2),
    "Adam_with_weight_decay_and_amsgrad": lambda p: torch.optim.Adam(
        p, 1.0, weight_decay=1e-6, amsgrad=True
    ),
    "AdamW": lambda p: torch.optim.AdamW(p, 1.0),
    "AdamW_without_weight_decay": lambda p: torch.optim.AdamW(p, 1.0, weight_decay=0),
    "AdamW_with_amsgrad": lambda p: torch.optim.AdamW(p, 1.0, amsgrad=True),
    "Adagrad": lambda p: torch.optim.Adagrad(p, 1.0),
    "Adagrad_with_weight_decay": lambda p: torch.optim.Adagrad(
        p, 1.0, weight_decay=1e-2
    ),
    "Adagrad_with_weight_decay_and_lr_decay": lambda p: torch.optim.Adagrad(
        p, 1.0, weight_decay=1e-6, lr_decay=1e-3
    ),
    "RMSprop": lambda p: torch.optim.RMSprop(p, 0.1),
    "RMSprop_with_weight_decay": lambda p: torch.optim.RMSprop(
        p, 0.1, weight_decay=1e-2
    ),
    "RMSprop_with_weight_decay_and_centered": lambda p: torch.optim.RMSprop(
        p, 0.1, weight_decay=1e-6, centered=True
    ),
    "RMSprop_with_weight_decay_and_centered_and_momentum": lambda p: torch.optim.RMSprop(
        p, 0.1, weight_decay=1e-6, centered=True, momentum=0.9
    ),
    "SGD": lambda p: torch.optim.SGD(p, 0.1),
    "SGD_with_weight_decay": lambda p: torch.optim.SGD(p, 0.1, weight_decay=1e-2),
    "SGD_with_weight_decay_and_momentum": lambda p: torch.optim.SGD(
        p, 0.1, momentum=0.9, weight_decay=1e-2
    ),
    "SGD_with_weight_decay_and_nesterov_momentum": lambda p: torch.optim.SGD(
        p, 0.1, momentum=0.9, weight_decay=1e-6, nesterov=True
    ),
}

# 初始化模型权重的函数
def weight_init(module):
    if isinstance(module, torch.nn.Linear):
        stdev = 1.0 / math.sqrt(module.weight.size(1))  # 计算标准差
        for p in module.parameters():
            p.data.uniform_(-stdev, stdev)  # 用均匀分布初始化权重

# 运行优化算法的函数
def run(optimizer_name, iterations, sample_every):
    torch.manual_seed(0)  # 设置随机种子
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3),  # 创建一个线性层
        torch.nn.Sigmoid(),  # 创建一个Sigmoid激活函数层
        torch.nn.Linear(3, 1),  # 创建另一个线性层
        torch.nn.Sigmoid(),  # 创建另一个Sigmoid激活函数层
    )
    model = model.to(torch.float64).apply(weight_init)  # 将模型转换为双精度浮点型，并初始化权重

    optimizer = OPTIMIZERS[optimizer_name](model.parameters())  # 使用给定优化器初始化优化器对象

    input = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float64)  # 创建输入张量

    values = []  # 初始化一个空列表用于存储结果
    # 对于给定的迭代次数执行循环
    for i in range(iterations):
        # 清零优化器的梯度
        optimizer.zero_grad()

        # 前向传播模型，计算输出
        output = model.forward(input)
        
        # 计算输出的总和作为损失
        loss = output.sum()
        
        # 反向传播，计算损失关于模型参数的梯度
        loss.backward()

        # 定义一个闭包函数，返回固定的张量 [10.0]
        def closure():
            return torch.tensor([10.0])

        # 使用优化器来更新模型参数，传入闭包函数作为参数
        optimizer.step(closure)

        # 如果当前迭代次数可以被 sample_every 整除
        if i % sample_every == 0:
            # 将模型参数的每个副本展平并转换为 NumPy 数组，添加到 values 列表中
            values.append(
                [p.clone().flatten().data.numpy() for p in model.parameters()]
            )

    # 返回存储了每隔 sample_every 迭代时模型参数副本的列表
    return values
# 主函数，程序的入口点
def main():
    # 创建命令行参数解析器，用于生成基准优化输出的 PyTorch
    parser = argparse.ArgumentParser(
        "Produce optimization output baseline from PyTorch"
    )
    # 添加命令行参数选项：迭代次数，默认为 1001
    parser.add_argument("-i", "--iterations", default=1001, type=int)
    # 添加命令行参数选项：每隔多少次采样，默认为 100
    parser.add_argument("-s", "--sample-every", default=100, type=int)
    # 解析命令行参数
    options = parser.parse_args()

    # 创建一个空字典，用于存储优化器参数映射
    optimizer_parameter_map = {}
    # 遍历所有优化器名称和其对应的参数
    for optimizer in OPTIMIZERS.keys():
        # 打印评估信息到标准错误输出
        sys.stderr.write(f"Evaluating {optimizer} ...\n")
        # 运行优化器，并将结果存储到参数映射中
        optimizer_parameter_map[optimizer] = run(
            optimizer, options.iterations, options.sample_every
        )

    # 调用 emit 函数，生成优化器参数的 C++ 代码
    emit(optimizer_parameter_map)


# 如果当前脚本作为主程序运行，则执行 main 函数
if __name__ == "__main__":
    main()
```
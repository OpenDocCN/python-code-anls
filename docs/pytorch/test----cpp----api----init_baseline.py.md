# `.\pytorch\test\cpp\api\init_baseline.py`

```
"""Script to generate baseline values from PyTorch initialization algorithms"""

# 引入系统相关模块
import sys

# 引入PyTorch库
import torch

# 定义C++头部内容
HEADER = """
#include <torch/types.h>

#include <vector>

namespace expected_parameters {
"""

# 定义C++尾部内容
FOOTER = "} // namespace expected_parameters"

# 定义参数生成函数的起始部分
PARAMETERS = "inline std::vector<std::vector<torch::Tensor>> {}() {{"


# 初始化算法字典，每个算法对应一个初始化函数
INITIALIZERS = {
    "Xavier_Uniform": lambda w: torch.nn.init.xavier_uniform(w),
    "Xavier_Normal": lambda w: torch.nn.init.xavier_normal(w),
    "Kaiming_Normal": lambda w: torch.nn.init.kaiming_normal(w),
    "Kaiming_Uniform": lambda w: torch.nn.init.kaiming_uniform(w),
}


# 生成C++代码的函数
def emit(initializer_parameter_map):
    # 打印生成标记，用于标识文件生成自哪里
    print("// @{} from {}".format("generated", __file__))
    # 打印C++头部
    print(HEADER)
    # 遍历每个初始化器及其对应的权重列表
    for initializer_name, weights in initializer_parameter_map.items():
        # 打印参数生成函数的声明，使用初始化器名称
        print(PARAMETERS.format(initializer_name))
        # 打印返回值部分
        print("  return {")
        # 遍历权重列表中的每个样本
        for sample in weights:
            print("    {")
            # 遍历每个样本中的参数
            for parameter in sample:
                # 格式化打印参数值
                parameter_values = "{{{}}}".format(", ".join(map(str, parameter)))
                print(f"      torch::tensor({parameter_values}),")
            print("    },")
        print("  };")
        print("}\n")
    # 打印C++尾部
    print(FOOTER)


# 运行初始化器并获取权重函数
def run(initializer):
    # 设置随机种子
    torch.manual_seed(0)

    # 创建神经网络层，设置输入输出维度
    layer1 = torch.nn.Linear(7, 15)
    # 使用指定初始化器初始化权重
    INITIALIZERS[initializer](layer1.weight)

    layer2 = torch.nn.Linear(15, 15)
    INITIALIZERS[initializer](layer2.weight)

    layer3 = torch.nn.Linear(15, 2)
    INITIALIZERS[initializer](layer3.weight)

    # 获取权重数据并转为numpy数组
    weight1 = layer1.weight.data.numpy()
    weight2 = layer2.weight.data.numpy()
    weight3 = layer3.weight.data.numpy()

    # 返回权重数组列表
    return [weight1, weight2, weight3]


# 主函数，生成各种初始化器的参数映射并输出C++代码
def main():
    # 初始化器参数映射字典
    initializer_parameter_map = {}
    # 遍历每个初始化器
    for initializer in INITIALIZERS.keys():
        # 打印正在评估的初始化器名称到标准错误流
        sys.stderr.write(f"Evaluating {initializer} ...\n")
        # 运行初始化器并存储权重结果
        initializer_parameter_map[initializer] = run(initializer)

    # 生成C++代码
    emit(initializer_parameter_map)


# 如果运行的是主程序，则调用主函数
if __name__ == "__main__":
    main()
```
# `.\pytorch\test\mkldnn_verbose.py`

```
# 导入 argparse 库，用于解析命令行参数
import argparse

# 导入 torch 库，用于深度学习模型构建和计算
import torch


# 定义一个继承自 torch.nn.Module 的模型类 Module
class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 添加一个二维卷积层，输入通道数为 1，输出通道数为 10，卷积核大小为 5x5，步长为 1
        self.conv = torch.nn.Conv2d(1, 10, 5, 1)

    # 定义前向传播方法
    def forward(self, x):
        # 对输入 x 进行卷积操作
        y = self.conv(x)
        return y


# 定义运行模型的函数，接收一个 verbose 级别参数
def run_model(level):
    # 创建一个 Module 类的实例，并设为评估模式
    m = Module().eval()
    # 创建一个随机张量作为输入数据，形状为 (1, 1, 112, 112)
    d = torch.rand(1, 1, 112, 112)
    # 设置 torch.backends.mkldnn.verbose 的级别为传入的 level 参数
    with torch.backends.mkldnn.verbose(level):
        # 将输入数据 d 输入模型 m 进行计算
        m(d)


# 当脚本作为主程序运行时
if __name__ == "__main__":
    # 创建 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser()
    # 添加命令行参数 --verbose-level，默认值为 0，类型为整数
    parser.add_argument("--verbose-level", default=0, type=int)
    # 解析命令行参数，将结果存储在 args 中
    args = parser.parse_args()
    try:
        # 调用 run_model 函数，传入命令行参数中的 verbose-level 值
        run_model(args.verbose_level)
    except Exception as e:
        # 捕获可能的异常并打印出错信息
        print(e)
```
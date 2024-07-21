# `.\pytorch\test\mkl_verbose.py`

```py
# 导入 argparse 模块，用于处理命令行参数
import argparse

# 导入 PyTorch 库
import torch


# 定义运行模型的函数，输入参数为输出详细级别
def run_model(level):
    # 创建一个输入大小为20，输出大小为30的线性模型
    m = torch.nn.Linear(20, 30)
    # 生成一个大小为128x20的随机输入张量
    input = torch.randn(128, 20)
    # 使用指定的详细级别启用 MKL 后端的详细日志
    with torch.backends.mkl.verbose(level):
        # 将输入张量传递给模型 m 进行前向计算
        m(input)


# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建 ArgumentParser 对象，用于处理命令行参数
    parser = argparse.ArgumentParser()
    # 添加一个名为 "--verbose-level" 的命令行选项，默认值为 0，类型为整数
    parser.add_argument("--verbose-level", default=0, type=int)
    # 解析命令行参数，并将其存储在 args 对象中
    args = parser.parse_args()
    try:
        # 调用 run_model 函数，并传入命令行参数中指定的详细级别
        run_model(args.verbose_level)
    except Exception as e:
        # 如果发生异常，打印异常信息
        print(e)
```
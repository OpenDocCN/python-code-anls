# `.\pytorch\test\bottleneck_test\test_args.py`

```
# 导入 argparse 库，用于命令行参数解析
import argparse

# 导入 PyTorch 库
import torch

# 如果这个脚本是主程序，则执行以下代码块
if __name__ == "__main__":
    # 创建 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser()

    # 添加一个必选参数 "--foo"，并设置帮助信息
    parser.add_argument("--foo", help="foo", required=True)
    
    # 添加一个必选参数 "--bar"，并设置帮助信息
    parser.add_argument("--bar", help="bar", required=True)
    
    # 解析命令行参数，并将结果存储在 "_" 变量中（通常用于忽略）
    _ = parser.parse_args()

    # 创建一个 3x3 的张量 x，所有元素为 1，并且需要计算梯度
    x = torch.ones((3, 3), requires_grad=True)
    
    # 计算张量 x 中所有元素乘以 3 的和，然后进行反向传播
    (3 * x).sum().backward()
```
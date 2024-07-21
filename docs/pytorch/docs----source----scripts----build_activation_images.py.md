# `.\pytorch\docs\source\scripts\build_activation_images.py`

```py
"""
This script will generate input-out plots for all of the activation
functions. These are for use in the documentation, and potentially in
online tutorials.
"""

from pathlib import Path  # 导入 Path 类用于处理文件路径

import matplotlib  # 导入 matplotlib 库
from matplotlib import pyplot as plt  # 从 matplotlib 中导入 pyplot 模块，并重命名为 plt

import torch  # 导入 PyTorch 库

matplotlib.use("Agg")  # 设置 matplotlib 使用非交互式后端，适合生成图像文件

# Create a directory for the images, if it doesn't exist
ACTIVATION_IMAGE_PATH = Path(__file__).parent / "activation_images"  # 获取当前脚本所在目录的父目录，加上子目录名称 "activation_images" 构成图片保存路径

if not ACTIVATION_IMAGE_PATH.exists():  # 如果图片保存路径不存在
    ACTIVATION_IMAGE_PATH.mkdir()  # 创建该路径

# In a refactor, these ought to go into their own module or entry
# points so we can generate this list programmatically
functions = [  # 定义包含多种激活函数的列表
    torch.nn.ELU(),  # 指数线性单元（ELU）激活函数
    torch.nn.Hardshrink(),  # Hardshrink 激活函数
    torch.nn.Hardtanh(),  # Hardtanh 激活函数
    torch.nn.Hardsigmoid(),  # Hardsigmoid 激活函数
    torch.nn.Hardswish(),  # Hardswish 激活函数
    torch.nn.LeakyReLU(negative_slope=0.1),  # 泄漏整流线性单元（LeakyReLU）激活函数
    torch.nn.LogSigmoid(),  # 对数 Sigmoid 激活函数
    torch.nn.PReLU(),  # 参数化的修正线性单元（PReLU）激活函数
    torch.nn.ReLU(),  # 线性修正单元（ReLU）激活函数
    torch.nn.ReLU6(),  # 修正线性单元 6（ReLU6）激活函数
    torch.nn.RReLU(),  # 随机修正线性单元（RReLU）激活函数
    torch.nn.SELU(),  # 缩放整流单元（SELU）激活函数
    torch.nn.SiLU(),  # sigmoid 激活函数的近似版本（SiLU）激活函数
    torch.nn.Mish(),  # Mish 激活函数
    torch.nn.CELU(),  # 修正的指数线性单元（CELU）激活函数
    torch.nn.GELU(),  # 高斯误差线性单元（GELU）激活函数
    torch.nn.Sigmoid(),  # Sigmoid 激活函数
    torch.nn.Softplus(),  # Softplus 激活函数
    torch.nn.Softshrink(),  # Softshrink 激活函数
    torch.nn.Softsign(),  # Softsign 激活函数
    torch.nn.Tanh(),  # Tanh 激活函数
    torch.nn.Tanhshrink(),  # Tanhshrink 激活函数
]


def plot_function(function, **args):
    """
    Plot a function on the current plot. The additional arguments may
    be used to specify color, alpha, etc.
    """
    xrange = torch.arange(-7.0, 7.0, 0.01)  # 创建一个范围从 -7 到 7（不包括）的 Tensor
    plt.plot(xrange.numpy(), function(xrange).detach().numpy(), **args)  # 绘制给定激活函数在指定范围内的曲线


# Step through all the functions
for function in functions:  # 遍历所有的激活函数
    function_name = function._get_name()  # 获取当前激活函数的名称
    plot_path = ACTIVATION_IMAGE_PATH / f"{function_name}.png"  # 构造当前激活函数图像保存的路径

    if not plot_path.exists():  # 如果当前激活函数的图像文件不存在
        plt.clf()  # 清空当前图形
        plt.grid(color="k", alpha=0.2, linestyle="--")  # 添加网格到图中

        plot_function(function)  # 绘制当前激活函数的曲线

        plt.title(function)  # 设置图的标题为当前激活函数的名称
        plt.xlabel("Input")  # 设置 x 轴标签
        plt.ylabel("Output")  # 设置 y 轴标签
        plt.xlim([-7, 7])  # 设置 x 轴范围
        plt.ylim([-7, 7])  # 设置 y 轴范围

        plt.savefig(plot_path)  # 保存当前图像到指定路径
        print(f"Saved activation image for {function_name} at {plot_path}")  # 打印保存成功的消息
```
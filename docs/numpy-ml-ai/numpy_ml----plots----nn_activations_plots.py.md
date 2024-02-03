# `numpy-ml\numpy_ml\plots\nn_activations_plots.py`

```py
# 禁用 flake8 的警告
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 seaborn 库的样式为白色背景
sns.set_style("white")
# 设置 seaborn 库的上下文为 notebook，字体缩放比例为 0.7
sns.set_context("notebook", font_scale=0.7)

# 从自定义模块中导入激活函数类
from numpy_ml.neural_nets.activations import (
    Affine,
    ReLU,
    LeakyReLU,
    Tanh,
    Sigmoid,
    ELU,
    Exponential,
    SELU,
    HardSigmoid,
    SoftPlus,
)


# 定义绘制激活函数图像的函数
def plot_activations():
    # 创建包含 2 行 5 列子图的图像
    fig, axes = plt.subplots(2, 5, sharex=True, sharey=True)
    # 定义激活函数列表
    fns = [
        Affine(),
        Tanh(),
        Sigmoid(),
        ReLU(),
        LeakyReLU(),
        ELU(),
        Exponential(),
        SELU(),
        HardSigmoid(),
        SoftPlus(),
    ]

    # 遍历子图和激活函数列表
    for ax, fn in zip(axes.flatten(), fns):
        # 生成输入数据
        X = np.linspace(-3, 3, 100).astype(float).reshape(100, 1)
        # 绘制激活函数图像及其一阶导数和二阶导数
        ax.plot(X, fn(X), label=r"$y$", alpha=1.0)
        ax.plot(X, fn.grad(X), label=r"$\frac{dy}{dx}$", alpha=1.0)
        ax.plot(X, fn.grad2(X), label=r"$\frac{d^2 y}{dx^2}$", alpha=1.0)
        # 绘制虚线表示 x 轴和 y 轴
        ax.hlines(0, -3, 3, lw=1, linestyles="dashed", color="k")
        ax.vlines(0, -1.2, 1.2, lw=1, linestyles="dashed", color="k")
        # 设置 y 轴范围和 x 轴范围
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(-3, 3)
        # 设置 x 轴和 y 轴的刻度
        ax.set_xticks([])
        ax.set_yticks([-1, 0, 1])
        ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # 设置子图标题为激活函数名称
        ax.set_title("{}".format(fn))
        # 显示图例
        ax.legend(frameon=False)
        # 移除图像的左边和底部边框
        sns.despine(left=True, bottom=True)

    # 设置图像大小
    fig.set_size_inches(10, 5)
    # 调整子图布局
    plt.tight_layout()
    # 保存图像为文件
    plt.savefig("img/plot.png", dpi=300)
    # 关闭所有图像
    plt.close("all")


# 当作为脚本直接运行时，调用绘制激活函数图像的函数
if __name__ == "__main__":
    plot_activations()
```
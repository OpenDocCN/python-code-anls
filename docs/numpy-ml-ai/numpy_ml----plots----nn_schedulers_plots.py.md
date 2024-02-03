# `numpy-ml\numpy_ml\plots\nn_schedulers_plots.py`

```py
# 禁用 flake8 检查
# 导入所需的库
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 seaborn 库的样式和上下文
# 参考链接：https://seaborn.pydata.org/generated/seaborn.set_context.html
# 参考链接：https://seaborn.pydata.org/generated/seaborn.set_style.html
sns.set_style("white")
sns.set_context("notebook", font_scale=0.7)

# 从自定义模块中导入不同的学习率调度器
from numpy_ml.neural_nets.schedulers import (
    ConstantScheduler,
    ExponentialScheduler,
    NoamScheduler,
    KingScheduler,
)

# 定义一个自定义的损失函数 king_loss_fn
def king_loss_fn(x):
    if x <= 250:
        return -0.25 * x + 82.50372665317208
    elif 250 < x <= 600:
        return 20.00372665317208
    elif 600 < x <= 700:
        return -0.2 * x + 140.00372665317207
    else:
        return 0.003726653172066108

# 定义一个绘制不同学习率调度器效果的函数 plot_schedulers
def plot_schedulers():
    # 创建一个包含4个子图的画布
    fig, axes = plt.subplots(2, 2)

    # 遍历不同的学习率调度器，并绘制对应的学习率曲线
    for ax, schs, title in zip(
        axes.flatten(), schedulers, ["Constant", "Exponential", "Noam", "King"]
    ):
        t0 = time.time()
        print("Running {} scheduler".format(title))
        X = np.arange(1, 1000)
        loss = np.array([king_loss_fn(x) for x in X])

        # 将损失值缩放以适应与学习率相同的轴
        scale = 0.01 / loss[0]
        loss *= scale

        if title == "King":
            ax.plot(X, loss, ls=":", label="Loss")

        for sc, lg in schs:
            Y = np.array([sc(x, ll) for x, ll in zip(X, loss)])
            ax.plot(X, Y, label=lg, alpha=0.6)

        ax.legend(fontsize=5)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Learning rate")
        ax.set_title("{} scheduler".format(title))
        print(
            "Finished plotting {} runs of {} in {:.2f}s".format(
                len(schs), title, time.time() - t0
            )
        )

    # 调整子图布局并保存绘图结果
    plt.tight_layout()
    plt.savefig("plot.png", dpi=300)
    plt.close("all")

# 如果作为脚本运行，则调用 plot_schedulers 函数
if __name__ == "__main__":
    plot_schedulers()
```
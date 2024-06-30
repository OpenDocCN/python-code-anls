# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_sgd_loss_functions.py`

```
"""
==========================
SGD: convex loss functions
==========================

A plot that compares the various convex loss functions supported by
:class:`~sklearn.linear_model.SGDClassifier` .

"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入matplotlib库并重命名为plt
import numpy as np  # 导入numpy库并重命名为np


# 定义修改后的Huber损失函数
def modified_huber_loss(y_true, y_pred):
    z = y_pred * y_true  # 计算预测值和真实值的乘积
    loss = -4 * z  # 计算初始损失
    loss[z >= -1] = (1 - z[z >= -1]) ** 2  # 应用条件：当 z >= -1 时，计算损失
    loss[z >= 1.0] = 0  # 应用条件：当 z >= 1.0 时，将损失置为0
    return loss  # 返回最终的损失值


# 设置 x 轴的取值范围
xmin, xmax = -4, 4
# 在 xmin 和 xmax 之间生成100个等间隔的点
xx = np.linspace(xmin, xmax, 100)
lw = 2  # 设置线宽为2

# 绘制不同损失函数的曲线
plt.plot([xmin, 0, 0, xmax], [1, 1, 0, 0], color="gold", lw=lw, label="Zero-one loss")
plt.plot(xx, np.where(xx < 1, 1 - xx, 0), color="teal", lw=lw, label="Hinge loss")
plt.plot(xx, -np.minimum(xx, 0), color="yellowgreen", lw=lw, label="Perceptron loss")
plt.plot(xx, np.log2(1 + np.exp(-xx)), color="cornflowerblue", lw=lw, label="Log loss")
plt.plot(
    xx,
    np.where(xx < 1, 1 - xx, 0) ** 2,
    color="orange",
    lw=lw,
    label="Squared hinge loss",
)
plt.plot(
    xx,
    modified_huber_loss(xx, 1),
    color="darkorchid",
    lw=lw,
    linestyle="--",
    label="Modified Huber loss",
)

plt.ylim((0, 8))  # 设置 y 轴的取值范围
plt.legend(loc="upper right")  # 显示图例，位置在右上角
plt.xlabel(r"Decision function $f(x)$")  # 设置 x 轴标签
plt.ylabel("$L(y=1, f(x))$")  # 设置 y 轴标签
plt.show()  # 显示绘制的图形
```
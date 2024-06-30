# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_bayesian_ridge_curvefit.py`

```
"""
============================================
Bayesian Ridge Regression 曲线拟合
============================================

使用贝叶斯岭回归拟合正弦曲线。

详见 :ref:`bayesian_ridge_regression` 获取更多有关回归器的信息。

一般情况下，当使用贝叶斯岭回归拟合多项式曲线时，初始化正则化参数（alpha, lambda）的选择可能很重要。
这是因为正则化参数是通过依赖初始值的迭代过程确定的。

在本例中，通过不同的初始值对正弦曲线进行多项式拟合。

当使用默认值（alpha_init = 1.90, lambda_init = 1.）时，所得曲线的偏差较大，方差较小。
因此，lambda_init 应该相对较小（1.e-3），以减少偏差。

此外，通过评估这些模型的对数边际似然（L），我们可以确定哪个模型更好。
可以得出具有较大 L 值的模型更有可能是更好的模型。
"""

# 作者: Yoshihiro Uchida <nimbus1after2a1sun7shower@gmail.com>

# %%
# 生成带噪声的正弦数据
# -------------------
import numpy as np

# 定义正弦函数
def func(x):
    return np.sin(2 * np.pi * x)

# 设置数据大小和随机种子
size = 25
rng = np.random.RandomState(1234)
x_train = rng.uniform(0.0, 1.0, size)  # 在[0, 1]上均匀分布的随机数作为训练数据
y_train = func(x_train) + rng.normal(scale=0.1, size=size)  # 添加噪声后的训练数据
x_test = np.linspace(0.0, 1.0, 100)  # 用于测试的均匀间隔的数据点

# %%
# 拟合三次多项式
# ---------------
from sklearn.linear_model import BayesianRidge

n_order = 3
X_train = np.vander(x_train, n_order + 1, increasing=True)  # 构建训练数据的Vandermonde矩阵
X_test = np.vander(x_test, n_order + 1, increasing=True)    # 构建测试数据的Vandermonde矩阵
reg = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)

# %%
# 绘制真实曲线和预测曲线，包括对数边际似然 (L)
# -------------------------------------------
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for i, ax in enumerate(axes):
    # 使用不同的初始值对贝叶斯岭回归进行拟合
    if i == 0:
        init = [1 / np.var(y_train), 1.0]  # 默认值
    elif i == 1:
        init = [1.0, 1e-3]
        reg.set_params(alpha_init=init[0], lambda_init=init[1])  # 设置正则化参数的初始值
    reg.fit(X_train, y_train)  # 拟合模型
    ymean, ystd = reg.predict(X_test, return_std=True)  # 预测测试数据的均值和标准差

    ax.plot(x_test, func(x_test), color="blue", label="sin($2\\pi x$)")  # 绘制真实的sin曲线
    ax.scatter(x_train, y_train, s=50, alpha=0.5, label="observation")  # 绘制训练数据点
    ax.plot(x_test, ymean, color="red", label="predict mean")  # 绘制预测均值曲线
    ax.fill_between(
        x_test, ymean - ystd, ymean + ystd, color="pink", alpha=0.5, label="predict std"
    )  # 绘制预测标准差区间
    ax.set_ylim(-1.3, 1.3)  # 设置y轴范围
    ax.legend()  # 显示图例
    title = "$\\alpha$_init$={:.2f},\\ \\lambda$_init$={}$".format(init[0], init[1])
    if i == 0:
        title += " (Default)"  # 添加默认值的标记
    ax.set_title(title, fontsize=12)  # 设置子图标题
    # 格式化字符串，将模型的参数和分数插入文本中
    text = "$\\alpha={:.1f}$\n$\\lambda={:.3f}$\n$L={:.1f}$".format(
        reg.alpha_, reg.lambda_, reg.scores_[-1]
    )
    # 在图形 ax 上添加文本，位于坐标 (0.05, -1.0)，字体大小为 12
    ax.text(0.05, -1.0, text, fontsize=12)
# 调整图表布局使其紧凑，以便更好地适应显示区域
plt.tight_layout()
# 显示当前的所有图表
plt.show()
```
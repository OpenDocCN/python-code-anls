# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_monotonic_constraints.py`

```
# 导入 matplotlib 的 pyplot 模块，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于生成随机数和数学运算
import numpy as np

# 导入集成学习中的 HistGradientBoostingRegressor 梯度提升回归器
from sklearn.ensemble import HistGradientBoostingRegressor
# 导入 PartialDependenceDisplay 用于展示偏依赖图
from sklearn.inspection import PartialDependenceDisplay

# 创建一个随机数生成器实例 rng
rng = np.random.RandomState(0)

# 设定样本数量
n_samples = 1000
# 生成第一个特征 f_0，服从均匀分布
f_0 = rng.rand(n_samples)
# 生成第二个特征 f_1，服从均匀分布
f_1 = rng.rand(n_samples)
# 将特征 f_0 和 f_1 合并成特征矩阵 X
X = np.c_[f_0, f_1]
# 生成噪声数据，服从正态分布
noise = rng.normal(loc=0.0, scale=0.01, size=n_samples)

# 定义目标变量 y，与 f_0 正相关，与 f_1 负相关，带有正弦和余弦的周期性变化及噪声
y = 5 * f_0 + np.sin(10 * np.pi * f_0) - 5 * f_1 - np.cos(10 * np.pi * f_1) + noise

# %%
# 在不施加任何约束条件下，对数据集进行训练
gbdt_no_cst = HistGradientBoostingRegressor()
gbdt_no_cst.fit(X, y)

# %%
# 在施加单调增加（1）和单调减少（-1）约束条件下，对数据集进行训练
gbdt_with_monotonic_cst = HistGradientBoostingRegressor(monotonic_cst=[1, -1])
gbdt_with_monotonic_cst.fit(X, y)

# %%
# 展示预测对两个特征的偏依赖关系
fig, ax = plt.subplots()
# 使用 PartialDependenceDisplay 从估计器 gbdt_no_cst 创建偏依赖展示
disp = PartialDependenceDisplay.from_estimator(
    gbdt_no_cst,
    X,
    features=[0, 1],
    feature_names=(
        "First feature",
        "Second feature",
    ),
    line_kw={"linewidth": 4, "label": "unconstrained", "color": "tab:blue"},
    ax=ax,
)
# 使用 PartialDependenceDisplay 从估计器 gbdt_with_monotonic_cst 创建偏依赖展示
PartialDependenceDisplay.from_estimator(
    gbdt_with_monotonic_cst,
    X,
    features=[0, 1],
    line_kw={"linewidth": 4, "label": "constrained", "color": "tab:orange"},
    ax=disp.axes_,
)

# 在每个特征的偏依赖展示中，绘制样本点
for f_idx in (0, 1):
    disp.axes_[0, f_idx].plot(
        X[:, f_idx], y, "o", alpha=0.3, zorder=-1, color="tab:green"
    )
    disp.axes_[0, f_idx].set_ylim(-6, 6)

# 添加图例
plt.legend()
# 设置整体标题
fig.suptitle("Monotonic constraints effect on partial dependences")
# 展示图形
plt.show()

# %%
# 可以看到，未约束模型的预测捕捉了数据的波动，而约束模型则跟随了数据的总体趋势，忽略了局部变化。

# %%
# .. _monotonic_cst_features_names:
#
# 使用特征名称指定单调性约束
# ----------------------------------------------------
#
# 注意，如果训练数据有特征名称，可以通过传递字典来指定单调性约束：
import pandas as pd
# 使用输入数据 X 创建一个 Pandas DataFrame，指定列名为 "f_0" 和 "f_1"
X_df = pd.DataFrame(X, columns=["f_0", "f_1"])

# 使用 HistGradientBoostingRegressor 初始化一个梯度提升决策树回归器，指定特征 "f_0" 为单调递增，"f_1" 为单调递减
gbdt_with_monotonic_cst_df = HistGradientBoostingRegressor(
    monotonic_cst={"f_0": 1, "f_1": -1}
).fit(X_df, y)

# 检查 GBDT 回归器在 X_df 上的预测结果与 gbdt_with_monotonic_cst 模型在 X 上的预测结果是否非常接近
np.allclose(
    gbdt_with_monotonic_cst_df.predict(X_df), gbdt_with_monotonic_cst.predict(X)
)
```
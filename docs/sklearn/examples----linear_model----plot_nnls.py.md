# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_nnls.py`

```
# 导入 matplotlib.pyplot 模块，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 模块，并使用 np 作为别名
import numpy as np

# 导入 r2_score 函数，用于评估回归模型的拟合效果
from sklearn.metrics import r2_score

# %%
# 生成一些随机数据
np.random.seed(42)

# 定义数据集大小
n_samples, n_features = 200, 50

# 生成随机的特征矩阵 X
X = np.random.randn(n_samples, n_features)

# 生成真实的回归系数 true_coef，并将其阈值化为非负数
true_coef = 3 * np.random.randn(n_features)
true_coef[true_coef < 0] = 0

# 生成因变量 y，使用 X 和 true_coef 进行线性组合
y = np.dot(X, true_coef)

# 为因变量 y 添加一些噪声
y += 5 * np.random.normal(size=(n_samples,))

# %%
# 将数据集分割为训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# %%
# 拟合非负最小二乘法模型
from sklearn.linear_model import LinearRegression

# 创建 LinearRegression 对象，设定参数 positive=True，以实现非负约束
reg_nnls = LinearRegression(positive=True)

# 在训练集上拟合模型，并对测试集进行预测
y_pred_nnls = reg_nnls.fit(X_train, y_train).predict(X_test)

# 计算 NNLS 模型的 R^2 分数
r2_score_nnls = r2_score(y_test, y_pred_nnls)
print("NNLS R2 score", r2_score_nnls)

# %%
# 拟合普通最小二乘法模型（OLS）
reg_ols = LinearRegression()

# 在训练集上拟合 OLS 模型，并对测试集进行预测
y_pred_ols = reg_ols.fit(X_train, y_train).predict(X_test)

# 计算 OLS 模型的 R^2 分数
r2_score_ols = r2_score(y_test, y_pred_ols)
print("OLS R2 score", r2_score_ols)

# %%
# 比较 OLS 和 NNLS 模型的回归系数，可以观察到它们高度相关（虚线表示单位关系），
# 但非负约束将一些系数收缩至0。
# 非负最小二乘法本质上产生稀疏结果。
fig, ax = plt.subplots()

# 绘制散点图，比较 OLS 和 NNLS 的回归系数
ax.plot(reg_ols.coef_, reg_nnls.coef_, linewidth=0, marker=".")

# 设置图形的横纵坐标轴范围
low_x, high_x = ax.get_xlim()
low_y, high_y = ax.get_ylim()
low = max(low_x, low_y)
high = min(high_x, high_y)

# 绘制单位关系的虚线
ax.plot([low, high], [low, high], ls="--", c=".3", alpha=0.5)

# 设置图形的 x 轴和 y 轴标签
ax.set_xlabel("OLS regression coefficients", fontweight="bold")
ax.set_ylabel("NNLS regression coefficients", fontweight="bold")
```
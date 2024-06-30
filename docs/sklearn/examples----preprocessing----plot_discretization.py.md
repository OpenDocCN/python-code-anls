# `D:\src\scipysrc\scikit-learn\examples\preprocessing\plot_discretization.py`

```
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于数值计算
import numpy as np

# 导入线性回归模型和决策树回归模型
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeRegressor

# 设置随机种子，以确保结果可重复
rnd = np.random.RandomState(42)
# 生成服从均匀分布的随机数作为特征 X
X = rnd.uniform(-3, 3, size=100)
# 根据正弦函数生成目标值 y，并添加噪声
y = np.sin(X) + rnd.normal(size=len(X)) / 3
# 将 X 转换成二维数组形式
X = X.reshape(-1, 1)

# 使用 KBinsDiscretizer 对数据集进行转换
enc = KBinsDiscretizer(n_bins=10, encode="onehot")
X_binned = enc.fit_transform(X)

# 使用原始数据集进行预测
# 创建包含两个子图的图像窗口，共享 y 轴
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 4))
# 生成预测线的数据点
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
# 使用线性回归模型拟合原始数据集并绘制预测结果
reg = LinearRegression().fit(X, y)
ax1.plot(line, reg.predict(line), linewidth=2, color="green", label="linear regression")
# 使用决策树回归模型拟合原始数据集并绘制预测结果
reg = DecisionTreeRegressor(min_samples_split=3, random_state=0).fit(X, y)
ax1.plot(line, reg.predict(line), linewidth=2, color="red", label="decision tree")
# 绘制原始数据点
ax1.plot(X[:, 0], y, "o", c="k")
# 添加图例
ax1.legend(loc="best")
# 设置 y 轴标签
ax1.set_ylabel("Regression output")
# 设置 x 轴标签
ax1.set_xlabel("Input feature")
# 设置子图标题
ax1.set_title("Result before discretization")

# 使用转换后的数据集进行预测
# 将预测线的数据点进行转换
line_binned = enc.transform(line)
# 使用线性回归模型拟合转换后的数据集并绘制预测结果
reg = LinearRegression().fit(X_binned, y)
ax2.plot(
    line,
    reg.predict(line_binned),
    linewidth=2,
    color="green",
    linestyle="-",
    label="linear regression",
)
# 使用决策树回归模型拟合转换后的数据集并绘制预测结果
reg = DecisionTreeRegressor(min_samples_split=3, random_state=0).fit(X_binned, y)
ax2.plot(
    line,
    reg.predict(line_binned),
    linewidth=2,
    color="red",
    linestyle="-",
    label="decision tree",
)
    line,                         # 变量 `line`，通常用于表示数据中的一行或一组值
    reg.predict(line_binned),     # 使用回归模型 `reg` 对输入数据 `line_binned` 进行预测
    linewidth=2,                  # 线条宽度设为 2
    color="red",                  # 绘图颜色设置为红色
    linestyle=":",                # 线条样式设置为虚线
    label="decision tree",        # 设置图例标签为 "decision tree"
)
# 在图形ax2上绘制散点图，X[:, 0]表示X的第一列作为X轴，y作为y轴，点形状为圆圈'o'，颜色为黑色'k'
ax2.plot(X[:, 0], y, "o", c="k")

# 使用enc对象的bin_edges_属性绘制垂直线段，表示数据的分箱边界
# plt.gca().get_ylim()获取当前轴的y轴限制范围作为垂直线段的高度范围
# linewidth设置线宽为1，alpha设置透明度为0.2
ax2.vlines(enc.bin_edges_[0], *plt.gca().get_ylim(), linewidth=1, alpha=0.2)

# 在图形ax2上添加图例，位置为最佳位置'best'
ax2.legend(loc="best")

# 设置x轴的标签文本为"Input feature"
ax2.set_xlabel("Input feature")

# 设置图形ax2的标题为"Result after discretization"
ax2.set_title("Result after discretization")

# 调整子图之间的布局，使其紧凑显示
plt.tight_layout()

# 显示图形
plt.show()
```
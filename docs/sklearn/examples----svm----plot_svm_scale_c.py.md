# `D:\src\scipysrc\scikit-learn\examples\svm\plot_svm_scale_c.py`

```
# %%
# Data generation
# ---------------
#
# 在这个示例中，我们研究了在使用 L1 或 L2 惩罚项时，重新参数化正则化参数 `C` 来考虑样本数量的影响。
# 为此，我们使用一个合成数据集，其中特征数量很多，但只有少数特征包含信息。因此，我们预期正则化会将系数收缩至接近零（L2 惩罚项）或完全为零（L1 惩罚项）。

from sklearn.datasets import make_classification

n_samples, n_features = 100, 300
X, y = make_classification(
    n_samples=n_samples, n_features=n_features, n_informative=5, random_state=1
)

# %%
# L1-penalty case
# ---------------
#
# 在 L1 惩罚项情况下，理论认为在强正则化的条件下，估计器无法像知道真实分布的模型那样进行预测
# （即使在样本量趋近于无穷大的极限情况下）。这是因为它可能会将某些本来具有预测能力的特征的权重设为零，
# 这会引入偏差。然而，理论也指出，通过调整 `C`，可以找到一组正确的非零参数及其符号。
#
# 我们定义一个带有 L1 惩罚项的线性支持向量机（Linear SVC）。

from sklearn.svm import LinearSVC

model_l1 = LinearSVC(penalty="l1", loss="squared_hinge", dual=False, tol=1e-3)

# %%
# We compute the mean test score for different values of `C` via
# cross-validation.

import numpy as np
import pandas as pd

from sklearn.model_selection import ShuffleSplit, validation_curve
# 创建一个由10个对数分布的值组成的数组，用作模型的参数C值（正则化强度）
Cs = np.logspace(-2.3, -1.3, 10)

# 创建一个包含3个均匀分布的训练集大小的数组
train_sizes = np.linspace(0.3, 0.7, 3)

# 为每个训练集大小创建一个标签，用于结果的标识
labels = [f"fraction: {train_size}" for train_size in train_sizes]

# 定义用于数据混洗和交叉验证的参数字典
shuffle_params = {
    "test_size": 0.3,
    "n_splits": 150,
    "random_state": 1,
}

# 初始化结果字典，包含C参数的列表
results = {"C": Cs}

# 针对每个标签和训练集大小进行循环
for label, train_size in zip(labels, train_sizes):
    # 使用ShuffleSplit创建交叉验证对象cv，指定训练集大小和其他混洗参数
    cv = ShuffleSplit(train_size=train_size, **shuffle_params)
    
    # 计算验证曲线，返回每个参数C对应的训练和测试得分
    train_scores, test_scores = validation_curve(
        model_l1,
        X,
        y,
        param_name="C",
        param_range=Cs,
        cv=cv,
        n_jobs=2,
    )
    
    # 将测试得分的平均值存储到结果字典中对应的标签下
    results[label] = test_scores.mean(axis=1)

# 将结果字典转换为DataFrame格式
results = pd.DataFrame(results)

# %%
# 导入matplotlib.pyplot库用于绘图
import matplotlib.pyplot as plt

# 创建一个包含两个子图的图形对象
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 6))

# 在第一个子图上绘制不缩放C参数的结果曲线（对数缩放）
results.plot(x="C", ax=axes[0], logx=True)
axes[0].set_ylabel("CV score")  # 设置y轴标签
axes[0].set_title("No scaling")  # 设置子图标题

# 针对每个标签，在第一个子图上绘制最佳C参数的竖线
for label in labels:
    best_C = results.loc[results[label].idxmax(), "C"]
    axes[0].axvline(x=best_C, linestyle="--", color="grey", alpha=0.7)

# 在第二个子图上绘制缩放了C参数的结果曲线（按照sqrt(1 / n_samples)进行缩放）
for train_size_idx, label in enumerate(labels):
    train_size = train_sizes[train_size_idx]
    # 计算缩放后的C参数并添加到结果DataFrame中
    results_scaled = results[[label]].assign(
        C_scaled=Cs * float(n_samples * np.sqrt(train_size))
    )
    results_scaled.plot(x="C_scaled", ax=axes[1], logx=True, label=label)
    best_C_scaled = results_scaled["C_scaled"].loc[results[label].idxmax()]
    axes[1].axvline(x=best_C_scaled, linestyle="--", color="grey", alpha=0.7)

axes[1].set_title("Scaling C by sqrt(1 / n_samples)")  # 设置第二个子图标题

# 设置整体图形的标题
_ = fig.suptitle("Effect of scaling C with L1 penalty")

# %%
# 在小的C值区域（强正则化），模型学到的所有系数都为零，导致严重的欠拟合。
#
# 使用默认的缩放方法会得到一个相对稳定的最优C值，而从欠拟合区域过渡出来取决于训练样本的数量。
# 重新参数化会导致更加稳定的结果。
#
# 参见例如 :arxiv:`On the prediction performance of the Lasso <1402.1700>` 或
# :arxiv:`Simultaneous analysis of Lasso and Dantzig selector <0801.1095>`，
# 其中正则化参数通常假定与1 / sqrt(n_samples)成比例。
#
# L2惩罚情况
# ---------------
# 我们可以使用L2惩罚做类似的实验。在这种情况下，理论表明为了实现预测一致性，惩罚参数应该随着样本数的增长保持不变。
model_l2 = LinearSVC(penalty="l2", loss="squared_hinge", dual=True)
Cs = np.logspace(-8, 4, 11)

# 重新生成标签以匹配新的训练集大小
labels = [f"fraction: {train_size}" for train_size in train_sizes]

# 初始化结果字典，包含C参数的列表
results = {"C": Cs}

# 针对每个标签和训练集大小进行循环
for label, train_size in zip(labels, train_sizes):
    # 使用ShuffleSplit创建交叉验证对象cv，指定训练集大小和其他混洗参数
    cv = ShuffleSplit(train_size=train_size, **shuffle_params)
    # 使用 validation_curve 函数进行验证曲线分析，分别计算训练集和测试集的得分
    train_scores, test_scores = validation_curve(
        model_l2,       # 使用的模型，这里是一个带 L2 正则化的模型
        X,              # 特征数据集
        y,              # 目标数据集
        param_name="C", # 调整的参数名称，这里是正则化参数 C
        param_range=Cs, # 参数 C 的取值范围
        cv=cv,          # 使用的交叉验证策略
        n_jobs=2,       # 并行运行的作业数量
    )
    # 计算测试集得分的平均值，并将结果存入结果字典中
    results[label] = test_scores.mean(axis=1)
results = pd.DataFrame(results)
# 将结果转换为 Pandas 的 DataFrame 格式，方便后续的数据处理和可视化

# %%
import matplotlib.pyplot as plt
# 导入 matplotlib 库用于绘图

fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 6))
# 创建一个包含两个子图的图形窗口，共享 y 轴，设置图形大小为 12x6

# plot results without scaling C
results.plot(x="C", ax=axes[0], logx=True)
# 在第一个子图中绘制结果，x 轴为 "C" 列，使用对数尺度展示 x 轴
axes[0].set_ylabel("CV score")
# 设置第一个子图的 y 轴标签为 "CV score"
axes[0].set_title("No scaling")
# 设置第一个子图的标题为 "No scaling"

for label in labels:
    best_C = results.loc[results[label].idxmax(), "C"]
    # 找到每个标签对应的最佳 C 值
    axes[0].axvline(x=best_C, linestyle="--", color="grey", alpha=0.8)
    # 在第一个子图中添加垂直虚线，表示最佳 C 值的位置

# plot results by scaling C
for train_size_idx, label in enumerate(labels):
    results_scaled = results[[label]].assign(
        C_scaled=Cs * float(n_samples * np.sqrt(train_sizes[train_size_idx]))
    )
    # 对 C 进行缩放后，在结果 DataFrame 中添加 C_scaled 列
    results_scaled.plot(x="C_scaled", ax=axes[1], logx=True, label=label)
    # 在第二个子图中绘制缩放后的结果，使用对数尺度展示 x 轴，显示标签
    best_C_scaled = results_scaled["C_scaled"].loc[results[label].idxmax()]
    # 找到缩放后每个标签对应的最佳 C_scaled 值
    axes[1].axvline(x=best_C_scaled, linestyle="--", color="grey", alpha=0.8)
    # 在第二个子图中添加垂直虚线，表示最佳缩放后的 C 值的位置
axes[1].set_title("Scaling C by sqrt(1 / n_samples)")
# 设置第二个子图的标题为 "Scaling C by sqrt(1 / n_samples)"

fig.suptitle("Effect of scaling C with L2 penalty")
# 设置整个图形的总标题为 "Effect of scaling C with L2 penalty"
plt.show()
# 显示图形

# %%
# For the L2 penalty case, the reparametrization seems to have a smaller impact
# on the stability of the optimal value for the regularization. The transition
# out of the overfitting region occurs in a more spread range and the accuracy
# does not seem to be degraded up to chance level.
#
# Try increasing the value to `n_splits=1_000` for better results in the L2
# case, which is not shown here due to the limitations on the documentation
# builder.
#
# 对于 L2 惩罚项，重新参数化似乎对正则化的最优值稳定性影响较小。从过拟合区域过渡发生在更广泛的范围内，精度似乎没有降到随机水平。
#
# 尝试增加 `n_splits=1_000` 的值以获得更好的 L2 惩罚项结果，但由于文档生成器的限制，此处未显示。
```
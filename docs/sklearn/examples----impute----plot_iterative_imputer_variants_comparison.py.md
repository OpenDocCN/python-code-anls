# `D:\src\scipysrc\scikit-learn\examples\impute\plot_iterative_imputer_variants_comparison.py`

```
# 导入需要的库和模块
import matplotlib.pyplot as plt  # 导入用于绘图的matplotlib库
import numpy as np  # 导入数值计算库numpy
import pandas as pd  # 导入数据处理库pandas

from sklearn.datasets import fetch_california_housing  # 导入加利福尼亚房价数据集获取函数
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林回归器

# 若要使用实验性特性，需要显式启用此选项
from sklearn.experimental import enable_iterative_imputer  # 启用迭代式填充器的实验性特性（需要忽略警告）
from sklearn.impute import IterativeImputer, SimpleImputer  # 导入迭代式填充器和简单填充器
from sklearn.kernel_approximation import Nystroem  # 导入Nystroem核近似函数
from sklearn.linear_model import BayesianRidge, Ridge  # 导入贝叶斯岭回归和岭回归模型
from sklearn.model_selection import cross_val_score  # 导入交叉验证评分函数
from sklearn.neighbors import KNeighborsRegressor  # 导入K近邻回归器
from sklearn.pipeline import make_pipeline  # 导入构建管道的函数

N_SPLITS = 5  # 设置交叉验证的分割数为5

rng = np.random.RandomState(0)  # 创建一个随机数生成器

X_full, y_full = fetch_california_housing(return_X_y=True)  # 获取加利福尼亚房价数据集的特征和目标值
# 为了示例目的，使用约2000个样本足矣
# 如果需要更慢的运行和不同的误差条，可以删除以下两行代码
X_full = X_full[::10]  # 每10个样本中取一个样本，减少数据量
y_full = y_full[::10]  # 每10个样本中取一个目标值，减少数据量
n_samples, n_features = X_full.shape  # 获取样本数和特征数
# 使用贝叶斯岭回归器初始化对象，用于整个数据集的评分估计，不包含缺失值
br_estimator = BayesianRidge()

# 计算完整数据集的评分，评分指标为负均方误差，使用交叉验证进行评估
score_full_data = pd.DataFrame(
    cross_val_score(
        br_estimator, X_full, y_full, scoring="neg_mean_squared_error", cv=N_SPLITS
    ),
    columns=["Full Data"],
)

# 对每一行添加一个缺失值
X_missing = X_full.copy()
y_missing = y_full
missing_samples = np.arange(n_samples)
missing_features = rng.choice(n_features, n_samples, replace=True)
X_missing[missing_samples, missing_features] = np.nan

# 使用均值和中位数策略对缺失值进行估计后的评分
score_simple_imputer = pd.DataFrame()
for strategy in ("mean", "median"):
    estimator = make_pipeline(
        SimpleImputer(missing_values=np.nan, strategy=strategy), br_estimator
    )
    score_simple_imputer[strategy] = cross_val_score(
        estimator, X_missing, y_missing, scoring="neg_mean_squared_error", cv=N_SPLITS
    )

# 使用不同的估算器进行迭代填补缺失值后的评分估计
estimators = [
    BayesianRidge(),
    RandomForestRegressor(
        n_estimators=4,
        max_depth=10,
        bootstrap=True,
        max_samples=0.5,
        n_jobs=2,
        random_state=0,
    ),
    make_pipeline(
        Nystroem(kernel="polynomial", degree=2, random_state=0), Ridge(alpha=1e3)
    ),
    KNeighborsRegressor(n_neighbors=15),
]
score_iterative_imputer = pd.DataFrame()

# 迭代填充器对于容差值敏感，取决于内部使用的估算器。
# 我们调整了容差值，以保持此示例在有限计算资源下运行，并尽量不改变与保持较严格默认值相比的结果。
tolerances = (1e-3, 1e-1, 1e-1, 1e-2)
for impute_estimator, tol in zip(estimators, tolerances):
    estimator = make_pipeline(
        IterativeImputer(
            random_state=0, estimator=impute_estimator, max_iter=25, tol=tol
        ),
        br_estimator,
    )
    score_iterative_imputer[impute_estimator.__class__.__name__] = cross_val_score(
        estimator, X_missing, y_missing, scoring="neg_mean_squared_error", cv=N_SPLITS
    )

# 合并所有评分结果到一个数据框中，包括原始数据、简单填充和迭代填充
scores = pd.concat(
    [score_full_data, score_simple_imputer, score_iterative_imputer],
    keys=["Original", "SimpleImputer", "IterativeImputer"],
    axis=1,
)

# 绘制加利福尼亚住房回归结果的条形图
fig, ax = plt.subplots(figsize=(13, 6))
means = -scores.mean()  # 取平均值并取负以得到均方误差
errors = scores.std()   # 计算标准差作为误差条
means.plot.barh(xerr=errors, ax=ax)  # 水平条形图表示均方误差，误差条用于显示误差范围
ax.set_title("California Housing Regression with Different Imputation Methods")  # 设置图表标题
ax.set_xlabel("MSE (smaller is better)")  # 设置 x 轴标签
ax.set_yticks(np.arange(means.shape[0]))  # 设置 y 轴刻度位置
ax.set_yticklabels([" w/ ".join(label) for label in means.index.tolist()])  # 设置 y 轴刻度标签
plt.tight_layout(pad=1)  # 调整布局以确保图表完整显示
plt.show()  # 显示图表
```
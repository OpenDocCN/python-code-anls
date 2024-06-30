# `D:\src\scipysrc\scikit-learn\examples\svm\plot_svm_anova.py`

```
# %%
# Load some data to play with
# ---------------------------
# 导入必要的库和模块：numpy用于数值计算，sklearn.datasets中的load_iris用于加载iris数据集
import numpy as np
from sklearn.datasets import load_iris

# 加载iris数据集，返回特征X和目标y
X, y = load_iris(return_X_y=True)

# Add non-informative features
# ----------------------------
# 创建随机数生成器rng，用于生成随机数种子
rng = np.random.RandomState(0)
# 为原始特征X添加36个非信息特征，使用2倍的随机数填充
X = np.hstack((X, 2 * rng.random((X.shape[0], 36))))

# %%
# Create the pipeline
# -------------------
# 导入所需的库和模块：特征选择使用SelectPercentile和f_classif，数据标准化使用StandardScaler，
# 支持向量分类器使用SVC
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 创建机器学习流水线Pipeline，包括特征选择（anova）、数据标准化（scaler）、支持向量分类器（svc）
clf = Pipeline(
    [
        ("anova", SelectPercentile(f_classif)),
        ("scaler", StandardScaler()),
        ("svc", SVC(gamma="auto")),
    ]
)

# %%
# Plot the cross-validation score as a function of percentile of features
# -----------------------------------------------------------------------
# 导入matplotlib.pyplot用于绘图，sklearn.model_selection中的cross_val_score用于交叉验证评分
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# 初始化列表，用于存储不同百分位特征选择下的平均得分和标准差
score_means = list()
score_stds = list()
# 定义要尝试的特征选择百分位列表
percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

# 循环遍历不同的特征选择百分位
for percentile in percentiles:
    # 设置流水线中特征选择的百分位参数
    clf.set_params(anova__percentile=percentile)
    # 使用交叉验证计算当前配置下的模型得分
    this_scores = cross_val_score(clf, X, y)
    # 计算并存储交叉验证得分的平均值和标准差
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())

# 绘制误差条形图，展示不同特征选择百分位下的模型性能
plt.errorbar(percentiles, score_means, np.array(score_stds))
plt.title("Performance of the SVM-Anova varying the percentile of features selected")
plt.xticks(np.linspace(0, 100, 11, endpoint=True))
plt.xlabel("Percentile")
plt.ylabel("Accuracy Score")
plt.axis("tight")
plt.show()
```
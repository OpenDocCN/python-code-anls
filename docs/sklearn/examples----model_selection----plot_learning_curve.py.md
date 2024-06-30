# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_learning_curve.py`

```
# %%
# Learning Curve
# ==============
#
# Learning curves show the effect of adding more samples during the training
# process. The effect is depicted by checking the statistical performance of
# the model in terms of training score and testing score.
#
# Here, we compute the learning curve of a naive Bayes classifier and a SVM
# classifier with a RBF kernel using the digits dataset.
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 加载手写数字数据集
X, y = load_digits(return_X_y=True)

# 创建一个高斯朴素贝叶斯分类器对象
naive_bayes = GaussianNB()

# 创建一个使用RBF核的支持向量机分类器对象
svc = SVC(kernel="rbf", gamma=0.001)

# %%
# The :meth:`~sklearn.model_selection.LearningCurveDisplay.from_estimator`
# displays the learning curve given the dataset and the predictive model to
# analyze. To get an estimate of the scores uncertainty, this method uses
# a cross-validation procedure.
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit

# 创建一个包含两个子图的图像框架
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True)

# 设置通用参数字典，用于绘制学习曲线
common_params = {
    "X": X,
    "y": y,
    "train_sizes": np.linspace(0.1, 1.0, 5),
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    "score_type": "both",
    "n_jobs": 4,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "Accuracy",
}

# 遍历分类器对象，绘制学习曲线
for ax_idx, estimator in enumerate([naive_bayes, svc]):
    LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
    handles, label = ax[ax_idx].get_legend_handles_labels()
    ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
    ax[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__}")

# %%
# We first analyze the learning curve of the naive Bayes classifier. Its shape
# can be found in more complex datasets very often: the training score is very
# high when using few samples for training and decreases when increasing the
# number of samples, whereas the test score is very low at the beginning and
# then increases when adding samples. The training and test scores become more
# realistic when all the samples are used for training.
#
# We see another typical learning curve for the SVM classifier with RBF kernel.
# The training score remains high regardless of the size of the training set.
# %%
# 导入所需的学习曲线函数
from sklearn.model_selection import learning_curve

# 定义通用参数字典，用于传递给学习曲线函数
common_params = {
    "X": X,  # 训练数据特征
    "y": y,  # 训练数据标签
    "train_sizes": np.linspace(0.1, 1.0, 5),  # 训练集大小的比例，从0.1到1.0，共5个点
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),  # 交叉验证方法
    "n_jobs": 4,  # 并行运行的作业数
    "return_times": True,  # 返回拟合时间和评分时间
}

# 使用朴素贝叶斯模型进行学习曲线分析，获取训练大小、测试分数、拟合时间和评分时间
train_sizes, _, test_scores_nb, fit_times_nb, score_times_nb = learning_curve(
    naive_bayes, **common_params
)

# 使用支持向量机模型进行学习曲线分析，获取训练大小、测试分数、拟合时间和评分时间
train_sizes, _, test_scores_svm, fit_times_svm, score_times_svm = learning_curve(
    svc, **common_params
)

# %%
# 创建一个图形和子图对象，用于显示拟合时间和评分时间的可扩展性
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), sharex=True)

# 对每个模型进行迭代，分别展示拟合时间和评分时间的可扩展性
for ax_idx, (fit_times, score_times, estimator) in enumerate(
    zip(
        [fit_times_nb, fit_times_svm],  # 拟合时间
        [score_times_nb, score_times_svm],  # 评分时间
        [naive_bayes, svc],  # 分类器模型
    )
):
    # 绘制拟合时间的可扩展性曲线
    ax[0, ax_idx].plot(train_sizes, fit_times.mean(axis=1), "o-")
    ax[0, ax_idx].fill_between(
        train_sizes,
        fit_times.mean(axis=1) - fit_times.std(axis=1),
        fit_times.mean(axis=1) + fit_times.std(axis=1),
        alpha=0.3,
    )
    ax[0, ax_idx].set_ylabel("Fit time (s)")  # 设置Y轴标签为拟合时间
    ax[0, ax_idx].set_title(
        f"Scalability of the {estimator.__class__.__name__} classifier"
    )  # 设置子图标题，展示分类器的可扩展性

    # 绘制评分时间的可扩展性曲线
    ax[1, ax_idx].plot(train_sizes, score_times.mean(axis=1), "o-")
    ax[1, ax_idx].fill_between(
        train_sizes,
        score_times.mean(axis=1) - score_times.std(axis=1),
        score_times.mean(axis=1) + score_times.std(axis=1),
        alpha=0.3,
    )
    ax[1, ax_idx].set_ylabel("Score time (s)")  # 设置Y轴标签为评分时间
    ax[1, ax_idx].set_xlabel("Number of training samples")  # 设置X轴标签为训练样本数量

# %%
# 我们可以看到支持向量机和朴素贝叶斯分类器在可扩展性方面有很大的差异。
# SVM分类器的拟合和评分时间复杂度随样本数量的增加而迅速增加。
# 其拟合时间复杂度是样本数量的平方以上，这使得其难以扩展到超过几万个样本的数据集。
# 相比之下，朴素贝叶斯分类器在拟合和评分时间上的复杂度较低，可扩展性更好。
#
# 因此，我们可以检查增加训练时间和
# %%
# 创建一个包含两个子图的图表，设置尺寸为 (16, 6) 英寸
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# 遍历不同的估算器和对应的训练/测试时间及分数数据
for ax_idx, (fit_times, test_scores, estimator) in enumerate(
    zip(
        [fit_times_nb, fit_times_svm],  # 训练时间数据列表（朴素贝叶斯，支持向量机）
        [test_scores_nb, test_scores_svm],  # 测试分数数据列表（朴素贝叶斯，支持向量机）
        [naive_bayes, svc],  # 估算器对象列表（朴素贝叶斯，支持向量机）
    )
):
    # 在当前子图上绘制训练时间与测试分数的关系图
    ax[ax_idx].plot(fit_times.mean(axis=1), test_scores.mean(axis=1), "o-")
    # 填充训练时间与测试分数的标准差区间
    ax[ax_idx].fill_between(
        fit_times.mean(axis=1),
        test_scores.mean(axis=1) - test_scores.std(axis=1),
        test_scores.mean(axis=1) + test_scores.std(axis=1),
        alpha=0.3,
    )
    # 设置当前子图的 y 轴标签为 "Accuracy"
    ax[ax_idx].set_ylabel("Accuracy")
    # 设置当前子图的 x 轴标签为 "Fit time (s)"
    ax[ax_idx].set_xlabel("Fit time (s)")
    # 设置当前子图的标题，显示当前估算器的性能
    ax[ax_idx].set_title(
        f"Performance of the {estimator.__class__.__name__} classifier"
    )

# 显示图表
plt.show()

# %%
# 在这些图中，我们可以寻找交叉验证分数不再增加的拐点，而只有训练时间在增加。
```
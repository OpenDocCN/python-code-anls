# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_grid_search_stats.py`

```
"""
==================================================
Statistical comparison of models using grid search
==================================================

This example illustrates how to statistically compare the performance of models
trained and evaluated using :class:`~sklearn.model_selection.GridSearchCV`.

"""

# %%
# We will start by simulating moon shaped data (where the ideal separation
# between classes is non-linear), adding to it a moderate degree of noise.
# Datapoints will belong to one of two possible classes to be predicted by two
# features. We will simulate 50 samples for each class:

import matplotlib.pyplot as plt  # 导入绘图库matplotlib
import seaborn as sns  # 导入绘图库seaborn

from sklearn.datasets import make_moons  # 导入make_moons生成月牙形数据

X, y = make_moons(noise=0.352, random_state=1, n_samples=100)  # 生成月牙形数据

sns.scatterplot(
    x=X[:, 0], y=X[:, 1], hue=y, marker="o", s=25, edgecolor="k", legend=False
).set_title("Data")  # 绘制数据散点图
plt.show()

# %%
# We will compare the performance of :class:`~sklearn.svm.SVC` estimators that
# vary on their `kernel` parameter, to decide which choice of this
# hyper-parameter predicts our simulated data best.
# We will evaluate the performance of the models using
# :class:`~sklearn.model_selection.RepeatedStratifiedKFold`, repeating 10 times
# a 10-fold stratified cross validation using a different randomization of the
# data in each repetition. The performance will be evaluated using
# :class:`~sklearn.metrics.roc_auc_score`.

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold  # 导入网格搜索和重复分层K折交叉验证
from sklearn.svm import SVC  # 导入支持向量机分类器

param_grid = [
    {"kernel": ["linear"]},  # 线性核参数
    {"kernel": ["poly"], "degree": [2, 3]},  # 多项式核参数，包括2次和3次
    {"kernel": ["rbf"]},  # 高斯径向基核参数
]

svc = SVC(random_state=0)  # 创建支持向量机分类器对象

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)  # 创建重复分层K折交叉验证对象

search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring="roc_auc", cv=cv)  # 创建网格搜索交叉验证对象
search.fit(X, y)  # 在数据集上拟合模型

# %%
# We can now inspect the results of our search, sorted by their
# `mean_test_score`:

import pandas as pd  # 导入数据处理库pandas

results_df = pd.DataFrame(search.cv_results_)  # 将交叉验证结果转换为DataFrame格式
results_df = results_df.sort_values(by=["rank_test_score"])  # 按照测试集平均得分排序
results_df = results_df.set_index(
    results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
).rename_axis("kernel")  # 以核函数类型作为索引
results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]  # 显示关键结果指标

# %%
# We can see that the estimator using the `'rbf'` kernel performed best,
# closely followed by `'linear'`. Both estimators with a `'poly'` kernel
# performed worse, with the one using a two-degree polynomial achieving a much
# lower performance than all other models.
#
# Usually, the analysis just ends here, but half the story is missing. The
# output of :class:`~sklearn.model_selection.GridSearchCV` does not provide
# information on the certainty of the differences between the models.
# We don't know if these are **statistically** significant.
# To evaluate this, we need to conduct a statistical test.
# Specifically, to contrast the performance of two models we should
# statistically compare their AUC scores. There are 100 samples (AUC
# scores) for each model as we repeated 10 times a 10-fold cross-validation.
#
# However, the scores of the models are not independent: all models are
# evaluated on the **same** 100 partitions, increasing the correlation
# between the performance of the models.
# Since some partitions of the data can make the distinction of the classes
# particularly easy or hard to find for all models, the models scores will
# co-vary.
#
# Let's inspect this partition effect by plotting the performance of all models
# in each fold, and calculating the correlation between models across folds:

# create df of model scores ordered by performance
model_scores = results_df.filter(regex=r"split\d*_test_score")

# plot 30 examples of dependency between cv fold and AUC scores
fig, ax = plt.subplots()
sns.lineplot(
    data=model_scores.transpose().iloc[:30],
    dashes=False,
    palette="Set1",
    marker="o",
    alpha=0.5,
    ax=ax,
)
ax.set_xlabel("CV test fold", size=12, labelpad=10)
ax.set_ylabel("Model AUC", size=12)
ax.tick_params(bottom=True, labelbottom=False)
plt.show()

# print correlation of AUC scores across folds
print(f"Correlation of models:\n {model_scores.transpose().corr()}")

# %%
# We can observe that the performance of the models highly depends on the fold.
#
# As a consequence, if we assume independence between samples we will be
# underestimating the variance computed in our statistical tests, increasing
# the number of false positive errors (i.e. detecting a significant difference
# between models when such does not exist) [1]_.
#
# Several variance-corrected statistical tests have been developed for these
# cases. In this example we will show how to implement one of them (the so
# called Nadeau and Bengio's corrected t-test) under two different statistical
# frameworks: frequentist and Bayesian.

# %%
# Comparing two models: frequentist approach
# ------------------------------------------
#
# We can start by asking: "Is the first model significantly better than the
# second model (when ranked by `mean_test_score`)?"
#
# To answer this question using a frequentist approach we could
# run a paired t-test and compute the p-value. This is also known as
# Diebold-Mariano test in the forecast literature [5]_.
# Many variants of such a t-test have been developed to account for the
# 'non-independence of samples problem'
# described in the previous section. We will use the one proven to obtain the
# highest replicability scores (which rate how similar the performance of a
# model is when evaluating it on different random partitions of the same
# dataset) while maintaining a low rate of false positives and false negatives:
# the Nadeau and Bengio's corrected t-test [2]_ that uses a 10 times repeated
# 10-fold cross validation [3]_.
#
# This corrected paired t-test is computed as:
#
# .. math::
#    t=\frac{\frac{1}{k \cdot r}\sum_{i=1}^{k}\sum_{j=1}^{r}x_{ij}}
# %%
import numpy as np
from scipy.stats import t


def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences, df, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val

# %%
model_1_scores = model_scores.iloc[0].values  # scores of the best model
model_2_scores = model_scores.iloc[1].values  # scores of the second-best model

differences = model_1_scores - model_2_scores

n = differences.shape[0]  # number of test sets
df = n - 1
n_train = len(list(cv.split(X, y))[0][0])
n_test = len(list(cv.split(X, y))[0][1])

t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
print(f"Corrected t-value: {t_stat:.3f}\nCorrected p-value: {p_val:.3f}")

# %%
# We can compare the corrected t- and p-values with the uncorrected ones:
# %%
# initialize uncorrected t-statistic for two independent samples
t_stat_uncorrected = np.mean(differences) / np.sqrt(np.var(differences, ddof=1) / n)
# calculate uncorrected p-value using the survival function (1 - CDF) of the t-distribution
p_val_uncorrected = t.sf(np.abs(t_stat_uncorrected), df)

# print out uncorrected t-value and p-value formatted to three decimal places
print(
    f"Uncorrected t-value: {t_stat_uncorrected:.3f}\n"
    f"Uncorrected p-value: {p_val_uncorrected:.3f}"
)

# %%
# Using the conventional significance alpha level at `p=0.05`, we observe that
# the uncorrected t-test concludes that the first model is significantly better
# than the second.
#
# With the corrected approach, in contrast, we fail to detect this difference.
#
# In the latter case, however, the frequentist approach does not let us
# conclude that the first and second model have an equivalent performance. If
# we wanted to make this assertion we need to use a Bayesian approach.

# %%
# Comparing two models: Bayesian approach
# ---------------------------------------
# We can use Bayesian estimation to calculate the probability that the first
# model is better than the second. Bayesian estimation will output a
# distribution followed by the mean :math:`\mu` of the differences in the
# performance of two models.
#
# To obtain the posterior distribution we need to define a prior that models
# our beliefs of how the mean is distributed before looking at the data,
# and multiply it by a likelihood function that computes how likely our
# observed differences are, given the values that the mean of differences
# could take.
#
# Bayesian estimation can be carried out in many forms to answer our question,
# but in this example we will implement the approach suggested by Benavoli and
# colleagues [4]_.
#
# One way of defining our posterior using a closed-form expression is to select
# a prior conjugate to the likelihood function. Benavoli and colleagues [4]_
# show that when comparing the performance of two classifiers we can model the
# prior as a Normal-Gamma distribution (with both mean and variance unknown)
# conjugate to a normal likelihood, to thus express the posterior as a normal
# distribution.
# Marginalizing out the variance from this normal posterior, we can define the
# posterior of the mean parameter as a Student's t-distribution. Specifically:
#
# .. math::
#    St(\mu;n-1,\overline{x},(\frac{1}{n}+\frac{n_{test}}{n_{train}})
#    \hat{\sigma}^2)
#
# where :math:`n` is the total number of samples,
# :math:`\overline{x}` represents the mean difference in the scores,
# :math:`n_{test}` is the number of samples used for testing,
# :math:`n_{train}` is the number of samples used for training,
# and :math:`\hat{\sigma}^2` represents the variance of the observed
# differences.
#
# Notice that we are using Nadeau and Bengio's corrected variance in our
# Bayesian approach as well.
#
# Let's compute and plot the posterior:

# initialize random variable for Student's t-distribution as the posterior
# distribution, specifying the degrees of freedom, mean, and scale
t_post = t(
    df, loc=np.mean(differences), scale=corrected_std(differences, n_train, n_test)
)

# %%
# Let's plot the posterior distribution:
# create a range of values for x-axis based on percentiles of the posterior distribution
x = np.linspace(t_post.ppf(0.001), t_post.ppf(0.999), 100)

# plot the probability density function of the posterior distribution
plt.plot(x, t_post.pdf(x))
# 设置 x 轴的刻度范围和间隔
plt.xticks(np.arange(-0.04, 0.06, 0.01))
# 使用 t_post 的概率密度函数填充 x 轴与 t_post 函数之间的区域，颜色为蓝色，透明度为 0.2
plt.fill_between(x, t_post.pdf(x), 0, facecolor="blue", alpha=0.2)
# 设置 y 轴标签为 "Probability density"
plt.ylabel("Probability density")
# 设置 x 轴标签为 "Mean difference ($\mu$)"
plt.xlabel(r"Mean difference ($\mu$)")
# 设置图表标题为 "Posterior distribution"
plt.title("Posterior distribution")
# 显示图表
plt.show()

# %%
# 我们可以通过计算后验分布曲线下从零到无穷的面积来计算第一个模型优于第二个模型的概率。
# 同样地，我们也可以计算第二个模型优于第一个模型的概率，方法是计算曲线从负无穷到零的面积。

better_prob = 1 - t_post.cdf(0)

print(
    f"Probability of {model_scores.index[0]} being more accurate than "
    f"{model_scores.index[1]}: {better_prob:.3f}"
)
print(
    f"Probability of {model_scores.index[1]} being more accurate than "
    f"{model_scores.index[0]}: {1 - better_prob:.3f}"
)

# %%
# 与频率主义方法相比，我们可以计算一个模型优于另一个模型的概率。
#
# 注意，我们得到了与频率主义方法中类似的结果。鉴于我们选择的先验分布，我们基本上
# 执行了相同的计算，但是我们可以得出不同的断言。

# %%
# 实际等价区域
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 有时候我们对确定模型是否具有等价性能感兴趣，其中 "等价" 是以实际方式定义的。
# 一种朴素的方法 [4]_ 是定义估计量在精度上相差不到 1% 时为实际等价。但是我们也可以
# 根据我们尝试解决的问题来定义这种实际等价性。例如，精度差异为 5% 将意味着销售增加
# 1000 美元，我们认为高于此金额对我们的业务是相关的。
#
# 在本例中，我们将定义实际等价区域 (ROPE) 为 :math:`[-0.01, 0.01]`。也就是说，
# 如果两个模型在性能上相差不到 1%，我们将认为它们是实际等价的。
#
# 要计算分类器在实际等价性上的概率，我们计算后验在 ROPE 区间下的曲线面积：

rope_interval = [-0.01, 0.01]
rope_prob = t_post.cdf(rope_interval[1]) - t_post.cdf(rope_interval[0])

print(
    f"Probability of {model_scores.index[0]} and {model_scores.index[1]} "
    f"being practically equivalent: {rope_prob:.3f}"
)

# %%
# 我们可以绘制后验分布在 ROPE 区间上的分布情况：

x_rope = np.linspace(rope_interval[0], rope_interval[1], 100)

plt.plot(x, t_post.pdf(x))
plt.xticks(np.arange(-0.04, 0.06, 0.01))
plt.vlines([-0.01, 0.01], ymin=0, ymax=(np.max(t_post.pdf(x)) + 1))
plt.fill_between(x_rope, t_post.pdf(x_rope), 0, facecolor="blue", alpha=0.2)
plt.ylabel("Probability density")
plt.xlabel(r"Mean difference ($\mu$)")
plt.title("Posterior distribution under the ROPE")
plt.show()
# %%
# 根据建议 [4]_，我们可以进一步解释这些概率，使用与频率派方法相同的标准：
# 是否落入 ROPE 区间的概率大于 95%（alpha 值为 5%）？在这种情况下，我们可以
# 得出两个模型在实质上是等效的结论。

# %%
# 贝叶斯估计方法还允许我们计算对差异估计的不确定性。这可以通过计算可信区间来实现。
# 对于给定的概率，它们展示了估计量的取值范围，对于我们的情况是性能平均差异。
# 例如，一个 50% 可信区间 [x, y] 告诉我们，真实（平均）性能差异在 x 和 y 之间的概率为 50%。
#
# 让我们使用 50%，75% 和 95% 来确定我们数据的可信区间：

cred_intervals = []
intervals = [0.5, 0.75, 0.95]

for interval in intervals:
    cred_interval = list(t_post.interval(interval))
    cred_intervals.append([interval, cred_interval[0], cred_interval[1]])

# 创建 DataFrame 来展示可信区间
cred_int_df = pd.DataFrame(
    cred_intervals, columns=["interval", "lower value", "upper value"]
).set_index("interval")
cred_int_df

# %%
# 如表所示，有 50% 的概率真实的模型间平均差异在 0.000977 到 0.019023 之间，
# 70% 的概率在 -0.005422 到 0.025422 之间，95% 的概率在 -0.016445 到 0.036445 之间。

# %%
# 所有模型的成对比较：频率派方法
# -------------------------------------------------------
#
# 我们还可能有兴趣比较所有通过 :class:`~sklearn.model_selection.GridSearchCV`
# 评估过的模型的性能。在这种情况下，我们将多次运行统计测试，这会引发“多重比较问题”
# <https://en.wikipedia.org/wiki/Multiple_comparisons_problem>。
#
# 解决这个问题有许多可能的方法，但标准做法是应用“Bonferroni 修正”
# <https://en.wikipedia.org/wiki/Bonferroni_correction>。Bonferroni 修正可以通过
# 将 p 值乘以我们正在测试的比较次数来计算。
#
# 让我们使用修正的 t 检验比较模型的性能：

from itertools import combinations
from math import factorial

# 计算比较次数
n_comparisons = factorial(len(model_scores)) / (
    factorial(2) * factorial(len(model_scores) - 2)
)
pairwise_t_test = []

# 对所有模型成对进行 t 检验
for model_i, model_k in combinations(range(len(model_scores)), 2):
    model_i_scores = model_scores.iloc[model_i].values
    model_k_scores = model_scores.iloc[model_k].values
    differences = model_i_scores - model_k_scores
    t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
    p_val *= n_comparisons  # 实施 Bonferroni 修正
    # Bonferroni 修正可能会输出大于 1 的 p 值
    p_val = 1 if p_val > 1 else p_val
    # 将以下信息添加到 pairwise_t_test 列表中：
    #   - model_scores.index[model_i]: 第一个模型的索引
    #   - model_scores.index[model_k]: 第二个模型的索引
    #   - t_stat: T 统计量（用于表示两个样本均值之间的差异）
    #   - p_val: 相关的 p 值（用于衡量统计显著性）
    pairwise_t_test.append(
        [model_scores.index[model_i], model_scores.index[model_k], t_stat, p_val]
    )
pairwise_comp_df = pd.DataFrame(
    pairwise_t_test, columns=["model_1", "model_2", "t_stat", "p_val"]
).round(3)
pairwise_comp_df
# 创建一个包含两个模型间配对 t 检验结果的 DataFrame，四舍五入保留三位小数

# %%
# 我们观察到，在进行多重比较校正后，唯一与其他模型显著不同的是 `'2_poly'` 模型。
# `'rbf'` 模型，被:class:`~sklearn.model_selection.GridSearchCV`评为排名第一的模型，
# 与 `'linear'` 或 `'3_poly'` 模型没有显著差异。

# %%
# 所有模型的配对比较：贝叶斯方法
# ----------------------------------------------------
#
# 当使用贝叶斯估计来比较多个模型时，我们不需要进行多重比较校正（原因见 [4]_）。
#
# 我们可以像第一部分一样进行配对比较：

pairwise_bayesian = []

for model_i, model_k in combinations(range(len(model_scores)), 2):
    model_i_scores = model_scores.iloc[model_i].values
    model_k_scores = model_scores.iloc[model_k].values
    differences = model_i_scores - model_k_scores
    t_post = t(
        df, loc=np.mean(differences), scale=corrected_std(differences, n_train, n_test)
    )
    worse_prob = t_post.cdf(rope_interval[0])
    better_prob = 1 - t_post.cdf(rope_interval[1])
    rope_prob = t_post.cdf(rope_interval[1]) - t_post.cdf(rope_interval[0])

    pairwise_bayesian.append([worse_prob, better_prob, rope_prob])

pairwise_bayesian_df = pd.DataFrame(
    pairwise_bayesian, columns=["worse_prob", "better_prob", "rope_prob"]
).round(3)

pairwise_comp_df = pairwise_comp_df.join(pairwise_bayesian_df)
pairwise_comp_df
# 使用贝叶斯方法计算模型之间表现更好、更差或几乎相等的概率。

# %%
# 使用贝叶斯方法，我们可以计算一个模型相对于另一个的表现更好、更差或几乎等效的概率。
#
# 结果显示，被:class:`~sklearn.model_selection.GridSearchCV`评为排名第一的 `'rbf'` 模型，
# 有约 6.8% 的概率比 `'linear'` 差，有约 1.8% 的概率比 `'3_poly'` 差。
# `'rbf'` 和 `'linear'` 有约 43% 的概率表现几乎相等，而 `'rbf'` 和 `'3_poly'` 有约 10% 的概率如此。
#
# 与使用频率主义方法得出的结论类似，所有模型都有100%的概率优于 `'2_poly'`，且没有一个与后者表现几乎相等。

# %%
# 总结
# ------------------
# - 性能指标的轻微差异可能很容易只是偶然发生的，而不是一个模型比另一个系统地预测更好。正如本例所示，
#   统计学可以告诉您这种可能性有多大。
# - 在比较GridSearchCV评估的两个模型的性能时，需要校正计算的方差，因为模型的评分彼此并不独立。
# - 使用（校正方差的）配对 t 检验的频率主义方法可以告诉我们一个模型的性能是否优于另一个模型。
#   degree of certainty above chance.
# - A Bayesian approach can provide the probabilities of one model being
#   better, worse or practically equivalent than another. It can also tell us
#   how confident we are of knowing that the true differences of our models
#   fall under a certain range of values.
# - If multiple models are statistically compared, a multiple comparisons
#   correction is needed when using the frequentist approach.
```
# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_grid_search_text_feature_extraction.py`

```
"""
==========================================================
Sample pipeline for text feature extraction and evaluation
==========================================================

The dataset used in this example is :ref:`20newsgroups_dataset` which will be
automatically downloaded, cached and reused for the document classification
example.

In this example, we tune the hyperparameters of a particular classifier using a
:class:`~sklearn.model_selection.RandomizedSearchCV`. For a demo on the
performance of some other classifiers, see the
:ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`
notebook.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Data loading
# ------------
# We load two categories from the training set. You can adjust the number of
# categories by adding their names to the list or setting `categories=None` when
# calling the dataset loader :func:`~sklearn.datasets.fetch_20newsgroups` to get
# the 20 of them.

# 导入fetch_20newsgroups函数从sklearn.datasets
from sklearn.datasets import fetch_20newsgroups

# 定义需要加载的新闻组类别
categories = [
    "alt.atheism",
    "talk.religion.misc",
]

# 加载训练集数据，subset="train"表示加载训练集数据，指定categories和其他参数
data_train = fetch_20newsgroups(
    subset="train",
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=("headers", "footers", "quotes"),
)

# 加载测试集数据，subset="test"表示加载测试集数据，指定categories和其他参数
data_test = fetch_20newsgroups(
    subset="test",
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=("headers", "footers", "quotes"),
)

# 打印加载的数据集信息
print(f"Loading 20 newsgroups dataset for {len(data_train.target_names)} categories:")
print(data_train.target_names)
print(f"{len(data_train.data)} documents")

# %%
# Pipeline with hyperparameter tuning
# -----------------------------------
#
# We define a pipeline combining a text feature vectorizer with a simple
# classifier yet effective for text classification.

# 导入文本特征提取器和分类器相关模块
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

# 定义一个包含文本特征向量化器和分类器的Pipeline对象
pipeline = Pipeline(
    [
        ("vect", TfidfVectorizer()),  # 使用TfidfVectorizer作为文本特征提取器
        ("clf", ComplementNB()),      # 使用ComplementNB作为分类器
    ]
)
pipeline

# %%
# We define a grid of hyperparameters to be explored by the
# :class:`~sklearn.model_selection.RandomizedSearchCV`. Using a
# :class:`~sklearn.model_selection.GridSearchCV` instead would explore all the
# possible combinations on the grid, which can be costly to compute, whereas the
# parameter `n_iter` of the :class:`~sklearn.model_selection.RandomizedSearchCV`
# controls the number of different random combination that are evaluated. Notice
# that setting `n_iter` larger than the number of possible combinations in a
# grid would lead to repeating already-explored combinations. We search for the
# best parameter combination for both the feature extraction (`vect__`) and the
# classifier (`clf__`).

# 导入numpy模块
import numpy as np

# 定义一个包含各种超参数的字典，用于随机搜索最佳参数组合
parameter_grid = {
    "vect__max_df": (0.2, 0.4, 0.6, 0.8, 1.0),  # 控制TfidfVectorizer的max_df参数
    "vect__min_df": (1, 3, 5, 10),             # 控制TfidfVectorizer的min_df参数
    "vect__ngram_range": ((1, 1), (1, 2)),     # 控制TfidfVectorizer的ngram_range参数，选择unigrams或bigrams
}
    # 定义一个名为 "vect__norm" 的参数，其值是一个包含 "l1" 和 "l2" 字符串的元组
    "vect__norm": ("l1", "l2"),
    # 定义一个名为 "clf__alpha" 的参数，其值是一个包含从 10^-6 到 10^6 的13个数值的 NumPy 数组
    "clf__alpha": np.logspace(-6, 6, 13),
# %%
# 本例中，`n_iter=40` 并不是超参数网格的详尽搜索。实际上，增加参数 `n_iter`
# 可以获得更详细的分析结果。因此，计算时间也会增加。我们可以通过增加参数
# `n_jobs` 来利用并行化评估参数组合，从而减少计算时间。
from pprint import pprint

from sklearn.model_selection import RandomizedSearchCV

# 创建 RandomizedSearchCV 对象，用于随机搜索最佳超参数组合
random_search = RandomizedSearchCV(
    estimator=pipeline,  # 使用的管道模型
    param_distributions=parameter_grid,  # 超参数网格
    n_iter=40,  # 迭代次数，随机搜索的次数
    random_state=0,  # 随机种子，用于可重复性
    n_jobs=2,  # 并行作业数，加快搜索速度
    verbose=1,  # 控制详细程度的日志输出
)

# 输出搜索开始的提示信息
print("Performing grid search...")
print("Hyperparameters to be evaluated:")
pprint(parameter_grid)

# %%
from time import time

# 记录开始时间
t0 = time()
# 执行随机搜索，训练模型，寻找最佳超参数组合
random_search.fit(data_train.data, data_train.target)
# 计算并输出完成所需的时间
print(f"Done in {time() - t0:.3f}s")

# %%
# 输出找到的最佳超参数组合
print("Best parameters combination found:")
best_parameters = random_search.best_estimator_.get_params()
for param_name in sorted(parameter_grid.keys()):
    print(f"{param_name}: {best_parameters[param_name]}")

# %%
# 计算并输出在测试集上的准确率
test_accuracy = random_search.score(data_test.data, data_test.target)
print(
    "Accuracy of the best parameters using the inner CV of "
    f"the random search: {random_search.best_score_:.3f}"
)
print(f"Accuracy on test set: {test_accuracy:.3f}")

# %%
# 管道中需要使用 `vect` 和 `clf` 前缀以避免可能的歧义，但在可视化结果时不是必要的。
# 因此，我们定义一个函数来重命名调整后的超参数，以提高可读性。
import pandas as pd

def shorten_param(param_name):
    """Remove components' prefixes in param_name."""
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name

# 将随机搜索的结果转换为 DataFrame，并重新命名参数名称
cv_results = pd.DataFrame(random_search.cv_results_)
cv_results = cv_results.rename(shorten_param, axis=1)

# %%
# 我们可以使用 `plotly.express.scatter` 来可视化评分时间与平均测试得分之间的权衡。
# 通过将鼠标悬停在给定点上，可以显示相应的参数。误差线对应于交叉验证的不同折中计算的标准偏差。
import plotly.express as px

param_names = [shorten_param(name) for name in parameter_grid.keys()]
labels = {
    "mean_score_time": "CV Score time (s)",
    "mean_test_score": "CV score (accuracy)",
}
fig = px.scatter(
    cv_results,
    x="mean_score_time",
    y="mean_test_score",
    error_x="std_score_time",
    error_y="std_test_score",
    hover_data=param_names,
    labels=labels,
)
fig.update_layout(
    title={
        "text": "trade-off between scoring time and mean test score",
        "y": 0.95,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    }
)
fig

# %%
# Notice that the cluster of models in the upper-left corner of the plot have
# the best trade-off between accuracy and scoring time. In this case, using
# bigrams increases the required scoring time without improving considerably the
# accuracy of the pipeline.
#
# .. note:: For more information on how to customize an automated tuning to
#    maximize score and minimize scoring time, see the example notebook
#    :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_digits.py`.
#
# We can also use a `plotly.express.parallel_coordinates
# <https://plotly.com/python-api-reference/generated/plotly.express.parallel_coordinates.html>`_
# to further visualize the mean test score as a function of the tuned
# hyperparameters. This helps finding interactions between more than two
# hyperparameters and provide intuition on their relevance for improving the
# performance of a pipeline.
#
# We apply a `math.log10` transformation on the `alpha` axis to spread the
# active range and improve the readability of the plot. A value :math:`x` on
# said axis is to be understood as :math:`10^x`.

import math  # 导入 math 模块

column_results = param_names + ["mean_test_score", "mean_score_time"]

transform_funcs = dict.fromkeys(column_results, lambda x: x)
# Using a logarithmic scale for alpha
transform_funcs["alpha"] = math.log10  # 将 alpha 转换为以 10 为底的对数

# L1 norms are mapped to index 1, and L2 norms to index 2
transform_funcs["norm"] = lambda x: 2 if x == "l2" else 1  # 将 "l1" 归为 1，将 "l2" 归为 2

# Unigrams are mapped to index 1 and bigrams to index 2
transform_funcs["ngram_range"] = lambda x: x[1]  # 将 ngram_range 中的 unigrams 映射为 1，bigrams 映射为 2

fig = px.parallel_coordinates(
    cv_results[column_results].apply(transform_funcs),
    color="mean_test_score",
    color_continuous_scale=px.colors.sequential.Viridis_r,
    labels=labels,
)
fig.update_layout(
    title={
        "text": "Parallel coordinates plot of text classifier pipeline",  # 设置图表标题
        "y": 0.99,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    }
)
fig

# %%
# The parallel coordinates plot displays the values of the hyperparameters on
# different columns while the performance metric is color coded. It is possible
# to select a range of results by clicking and holding on any axis of the
# parallel coordinate plot. You can then slide (move) the range selection and
# cross two selections to see the intersections. You can undo a selection by
# clicking once again on the same axis.
#
# In particular for this hyperparameter search, it is interesting to notice that
# the top performing models do not seem to depend on the regularization `norm`,
# but they do depend on a trade-off between `max_df`, `min_df` and the
# regularization strength `alpha`. The reason is that including noisy features
# (i.e. `max_df` close to :math:`1.0` or `min_df` close to :math:`0`) tend to
# overfit and therefore require a stronger regularization to compensate. Having
# less features require less regularization and less scoring time.
#
# The best accuracy scores are obtained when `alpha` is between :math:`10^{-6}`
# 此行是一个注释，以井号(#)开头，用于对代码的某一部分进行说明或者标记。
```
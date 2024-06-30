# `D:\src\scipysrc\scikit-learn\examples\compose\plot_compare_reduction.py`

```
"""
=================================================================
Selecting dimensionality reduction with Pipeline and GridSearchCV
=================================================================

This example constructs a pipeline that does dimensionality
reduction followed by prediction with a support vector
classifier. It demonstrates the use of ``GridSearchCV`` and
``Pipeline`` to optimize over different classes of estimators in a
single CV run -- unsupervised ``PCA`` and ``NMF`` dimensionality
reductions are compared to univariate feature selection during
the grid search.

Additionally, ``Pipeline`` can be instantiated with the ``memory``
argument to memoize the transformers within the pipeline, avoiding to fit
again the same transformers over and over.

Note that the use of ``memory`` to enable caching becomes interesting when the
fitting of a transformer is costly.
"""

# Authors: Robert McGibbon
#          Joel Nothman
#          Guillaume Lemaitre

# %%
# Illustration of ``Pipeline`` and ``GridSearchCV``
###############################################################################

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.decomposition import NMF, PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

# Load the digits dataset
X, y = load_digits(return_X_y=True)

# Define a Pipeline with three stages: scaling, dimensionality reduction, and classification
pipe = Pipeline(
    [
        ("scaling", MinMaxScaler()),  # Stage 1: MinMaxScaler for feature scaling
        # Stage 2: Dimensionality reduction stage, placeholder for parameter grid search
        ("reduce_dim", "passthrough"),
        ("classify", LinearSVC(dual=False, max_iter=10000)),  # Stage 3: Linear SVC classifier
    ]
)

# Define options for number of features to be considered
N_FEATURES_OPTIONS = [2, 4, 8]
# Define options for regularization parameter C in Linear SVC
C_OPTIONS = [1, 10, 100, 1000]

# Define parameter grid for GridSearchCV
param_grid = [
    {
        "reduce_dim": [PCA(iterated_power=7), NMF(max_iter=1000)],  # Use PCA and NMF for reduction
        "reduce_dim__n_components": N_FEATURES_OPTIONS,  # Vary number of components for reduction
        "classify__C": C_OPTIONS,  # Vary C parameter for Linear SVC
    },
    {
        "reduce_dim": [SelectKBest(mutual_info_classif)],  # Use SelectKBest with mutual_info_classif
        "reduce_dim__k": N_FEATURES_OPTIONS,  # Vary number of top features to select
        "classify__C": C_OPTIONS,  # Vary C parameter for Linear SVC
    },
]

# Labels for different reduction techniques used in the grid
reducer_labels = ["PCA", "NMF", "KBest(mutual_info_classif)"]

# Instantiate GridSearchCV with defined pipeline and parameter grid
grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid)
grid.fit(X, y)

# %%
import pandas as pd

# Extract mean test scores from grid search results
mean_scores = np.array(grid.cv_results_["mean_test_score"])
# Reshape scores to match the grid structure
mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# Select the maximum score across C options for each configuration
mean_scores = mean_scores.max(axis=0)
# Create a DataFrame for easier plotting
mean_scores = pd.DataFrame(
    mean_scores.T, index=N_FEATURES_OPTIONS, columns=reducer_labels
)

# Plotting the results
ax = mean_scores.plot.bar()
ax.set_title("Comparing feature reduction techniques")
ax.set_xlabel("Reduced number of features")
ax.set_ylabel("Digit classification accuracy")
ax.set_ylim((0, 1))  # Set y-axis limit from 0 to 1
ax.legend(loc="upper left")  # Place legend in upper left corner

plt.show()

# %%
# Caching transformers within a ``Pipeline``
###############################################################################
# It is sometimes worthwhile storing the state of a specific transformer
# since it could be used again. Using a pipeline in ``GridSearchCV`` triggers
# such situations. Therefore, we use the argument ``memory`` to enable caching.
#
# .. warning::
#     Note that this example is, however, only an illustration since for this
#     specific case fitting PCA is not necessarily slower than loading the
#     cache. Hence, use the ``memory`` constructor parameter when the fitting
#     of a transformer is costly.

# 导入必要的库
from shutil import rmtree

from joblib import Memory

# 创建一个临时文件夹来存储管道中的变换器
location = "cachedir"
# 初始化一个 Memory 对象来进行缓存
memory = Memory(location=location, verbose=10)
# 创建一个 Pipeline 对象，包含 PCA 和 LinearSVC 两个步骤，并启用缓存
cached_pipe = Pipeline(
    [("reduce_dim", PCA()), ("classify", LinearSVC(dual=False, max_iter=10000))],
    memory=memory,
)

# This time, a cached pipeline will be used within the grid search


# 在退出之前清除临时缓存
memory.clear(warn=False)
# 删除临时文件夹及其内容
rmtree(location)

# %%
# The ``PCA`` fitting is only computed at the evaluation of the first
# configuration of the ``C`` parameter of the ``LinearSVC`` classifier. The
# other configurations of ``C`` will trigger the loading of the cached ``PCA``
# estimator data, leading to save processing time. Therefore, the use of
# caching the pipeline using ``memory`` is highly beneficial when fitting
# a transformer is costly.
```
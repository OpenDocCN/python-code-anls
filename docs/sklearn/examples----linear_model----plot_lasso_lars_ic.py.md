# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_lasso_lars_ic.py`

```
"""
==============================================
Lasso model selection via information criteria
==============================================

This example reproduces the example of Fig. 2 of [ZHT2007]_. A
:class:`~sklearn.linear_model.LassoLarsIC` estimator is fit on a
diabetes dataset and the AIC and the BIC criteria are used to select
the best model.

.. note::
    It is important to note that the optimization to find `alpha` with
    :class:`~sklearn.linear_model.LassoLarsIC` relies on the AIC or BIC
    criteria that are computed in-sample, thus on the training set directly.
    This approach differs from the cross-validation procedure. For a comparison
    of the two approaches, you can refer to the following example:
    :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py`.

.. rubric:: References

.. [ZHT2007] :arxiv:`Zou, Hui, Trevor Hastie, and Robert Tibshirani.
    "On the degrees of freedom of the lasso."
    The Annals of Statistics 35.5 (2007): 2173-2192.
    <0712.0881>`
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# We will use the diabetes dataset.
from sklearn.datasets import load_diabetes

# Load the diabetes dataset into variables X (features) and y (target).
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Get the number of samples in the dataset.
n_samples = X.shape[0]

# Display the first few rows of the features (X).
X.head()

# %%
# Scikit-learn provides an estimator called
# :class:`~sklearn.linear_model.LassoLarsIC` that uses either Akaike's
# information criterion (AIC) or the Bayesian information criterion (BIC) to
# select the best model. Before fitting
# this model, we will scale the dataset.
#
# In the following, we are going to fit two models to compare the values
# reported by AIC and BIC.
from sklearn.linear_model import LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Create a pipeline that scales the data and applies LassoLarsIC with AIC criterion.
lasso_lars_ic = make_pipeline(StandardScaler(), LassoLarsIC(criterion="aic")).fit(X, y)

# %%
# To be in line with the definition in [ZHT2007]_, we need to rescale the
# AIC and the BIC. Indeed, Zou et al. are ignoring some constant terms
# compared to the original definition of AIC derived from the maximum
# log-likelihood of a linear model. You can refer to
# :ref:`mathematical detail section for the User Guide <lasso_lars_ic>`.
def zou_et_al_criterion_rescaling(criterion, n_samples, noise_variance):
    """Rescale the information criterion to follow the definition of Zou et al."""
    return criterion - n_samples * np.log(2 * np.pi * noise_variance) - n_samples

# %%
import numpy as np

# Rescale the AIC criterion using the function defined above.
aic_criterion = zou_et_al_criterion_rescaling(
    lasso_lars_ic[-1].criterion_,  # Accessing the criterion value from the fitted LassoLarsIC model
    n_samples,  # Number of samples in the dataset
    lasso_lars_ic[-1].noise_variance_,  # Estimated noise variance from the model
)

# Find the index of the alpha value in the alphas_ array that matches the selected alpha.
index_alpha_path_aic = np.flatnonzero(
    lasso_lars_ic[-1].alphas_ == lasso_lars_ic[-1].alpha_
)[0]

# %%
# Switch the LassoLarsIC criterion to BIC and refit the model.
lasso_lars_ic.set_params(lassolarsic__criterion="bic").fit(X, y)

# Rescale the BIC criterion using the function defined earlier.
bic_criterion = zou_et_al_criterion_rescaling(
    lasso_lars_ic[-1].criterion_,  # Accessing the criterion value from the fitted LassoLarsIC model
    n_samples,  # Number of samples in the dataset
    lasso_lars_ic[-1].noise_variance_,  # Estimated noise variance from the model
)

# Find the index of the alpha value in the alphas_ array that matches the selected alpha.
index_alpha_path_bic = np.flatnonzero(
    # 检查 LassoLarsIC 对象中最后一个模型的最优 alpha 是否与当前 alpha 相等
    lasso_lars_ic[-1].alphas_ == lasso_lars_ic[-1].alpha_
# %%
# 现在我们已经收集了AIC和BIC，我们可以检查两个标准的最小值是否发生在相同的alpha值处。
# 然后，我们可以简化接下来的绘图过程。
index_alpha_path_aic == index_alpha_path_bic

# %%
# 最后，我们可以绘制AIC和BIC标准，以及随后选择的正则化参数。
import matplotlib.pyplot as plt

# 绘制AIC标准的折线图，用蓝色标记
plt.plot(aic_criterion, color="tab:blue", marker="o", label="AIC criterion")
# 绘制BIC标准的折线图，用橙色标记
plt.plot(bic_criterion, color="tab:orange", marker="o", label="BIC criterion")
# 在图上添加竖线，表示选择的alpha值，线条为虚线，颜色为黑色
plt.vlines(
    index_alpha_path_bic,
    aic_criterion.min(),
    aic_criterion.max(),
    color="black",
    linestyle="--",
    label="Selected alpha",
)
# 添加图例
plt.legend()
# 设置y轴标签
plt.ylabel("Information criterion")
# 设置x轴标签
plt.xlabel("Lasso model sequence")
# 设置图标题
_ = plt.title("Lasso model selection via AIC and BIC")
```
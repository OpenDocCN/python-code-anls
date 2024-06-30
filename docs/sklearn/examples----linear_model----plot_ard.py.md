# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_ard.py`

```
"""
====================================
Comparing Linear Bayesian Regressors
====================================

This example compares two different bayesian regressors:

 - a :ref:`automatic_relevance_determination`
 - a :ref:`bayesian_ridge_regression`

In the first part, we use an :ref:`ordinary_least_squares` (OLS) model as a
baseline for comparing the models' coefficients with respect to the true
coefficients. Thereafter, we show that the estimation of such models is done by
iteratively maximizing the marginal log-likelihood of the observations.

In the last section we plot predictions and uncertainties for the ARD and the
Bayesian Ridge regressions using a polynomial feature expansion to fit a
non-linear relationship between `X` and `y`.

"""

# Author: Arturo Amor <david-arturo.amor-quiroz@inria.fr>

# %%
# Models robustness to recover the ground truth weights
# =====================================================
#
# Generate synthetic dataset
# --------------------------
#
# We generate a dataset where `X` and `y` are linearly linked: 10 of the
# features of `X` will be used to generate `y`. The other features are not
# useful at predicting `y`. In addition, we generate a dataset where `n_samples
# == n_features`. Such a setting is challenging for an OLS model and leads
# potentially to arbitrary large weights. Having a prior on the weights and a
# penalty alleviates the problem. Finally, gaussian noise is added.

from sklearn.datasets import make_regression

# Generate synthetic data for regression, with 100 samples, 100 features,
# 10 informative features, noise level 8, and a random state for reproducibility.
X, y, true_weights = make_regression(
    n_samples=100,
    n_features=100,
    n_informative=10,
    noise=8,
    coef=True,
    random_state=42,
)

# %%
# Fit the regressors
# ------------------
#
# We now fit both Bayesian models and the OLS to later compare the models'
# coefficients.

import pandas as pd
from sklearn.linear_model import ARDRegression, BayesianRidge, LinearRegression

# Fit Ordinary Least Squares (OLS), Bayesian Ridge Regression (BRR),
# and Automatic Relevance Determination (ARD) to the generated data.
olr = LinearRegression().fit(X, y)
brr = BayesianRidge(compute_score=True, max_iter=30).fit(X, y)
ard = ARDRegression(compute_score=True, max_iter=30).fit(X, y)

# Create a DataFrame to store the coefficients of the true generative process,
# ARD, Bayesian Ridge, and OLS models for comparison.
df = pd.DataFrame(
    {
        "Weights of true generative process": true_weights,
        "ARDRegression": ard.coef_,
        "BayesianRidge": brr.coef_,
        "LinearRegression": olr.coef_,
    }
)

# %%
# Plot the true and estimated coefficients
# ----------------------------------------
#
# Visualize the coefficients of the true generative process and the models'
# coefficients using a heatmap.

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import SymLogNorm

# Create a heatmap to compare the coefficients across different models.
plt.figure(figsize=(10, 6))
ax = sns.heatmap(
    df.T,
    norm=SymLogNorm(linthresh=10e-4, vmin=-80, vmax=80),
    cbar_kws={"label": "coefficients' values"},
    cmap="seismic_r",
)
plt.ylabel("linear model")
plt.xlabel("coefficients")
plt.tight_layout(rect=(0, 0, 1, 0.95))
_ = plt.title("Models' coefficients")

# %%
# Due to the added noise, none of the models recover the true weights. Indeed,
# %%
# Plotting polynomial regressions with std errors of the scores
# -------------------------------------------------------------
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import ARDRegression, BayesianRidge
import matplotlib.pyplot as plt

# Fit polynomial regressions with Bayesian models
# ------------------------------------------------
# Create a pipeline for ARDRegression with degree 10 polynomial features
# and standard scaler for preprocessing.
ard_poly = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),
    StandardScaler(),
    ARDRegression(),
).fit(X, y)

# Create a pipeline for BayesianRidge with degree 10 polynomial features
# and standard scaler for preprocessing.
brr_poly = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),
    StandardScaler(),
    BayesianRidge(),
).fit(X, y)

# Predict using the fitted models to get mean predictions and standard deviations
# of the posterior distribution for the model parameters.
y_ard, y_ard_std = ard_poly.predict(X_plot, return_std=True)
y_brr, y_brr_std = brr_poly.predict(X_plot, return_std=True)

# Plotting
# ---------
# Plot the results of ARDRegression and BayesianRidge with degree 10 polynomial
# regressions, showing the mean predictions and standard deviations as error bars.
plt.errorbar(X_plot[:, 0], y_ard, yerr=y_ard_std, label="ARD", color="navy", fmt='-')
plt.errorbar(X_plot[:, 0], y_brr, yerr=y_brr_std, label="BayesianRidge", color="red", fmt='-')
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.xlabel("Input feature")
plt.ylabel("Target")
plt.title("Polynomial regression with Bayesian models")
plt.legend()
plt.show()
ax = sns.scatterplot(
    data=full_data, x="input_feature", y="target", color="black", alpha=0.75
)
# 创建一个散点图，显示完整数据集中输入特征和目标的关系，设置点颜色为黑色，透明度为0.75

ax.plot(X_plot, y_plot, color="black", label="Ground Truth")
# 在图上绘制一条线，表示真实数据的拟合曲线，颜色为黑色，标签为"Ground Truth"

ax.plot(X_plot, y_brr, color="red", label="BayesianRidge with polynomial features")
# 绘制贝叶斯岭回归模型在多项式特征上的预测曲线，颜色为红色，标签为"BayesianRidge with polynomial features"

ax.plot(X_plot, y_ard, color="navy", label="ARD with polynomial features")
# 绘制自适应回归模型在多项式特征上的预测曲线，颜色为海军蓝色，标签为"ARD with polynomial features"

ax.fill_between(
    X_plot.ravel(),
    y_ard - y_ard_std,
    y_ard + y_ard_std,
    color="navy",
    alpha=0.3,
)
# 在图中填充自适应回归模型预测值的标准差区间，颜色为海军蓝色，透明度为0.3

ax.fill_between(
    X_plot.ravel(),
    y_brr - y_brr_std,
    y_brr + y_brr_std,
    color="red",
    alpha=0.3,
)
# 在图中填充贝叶斯岭回归模型预测值的标准差区间，颜色为红色，透明度为0.3

ax.legend()
# 在图中添加图例，展示每条曲线的标签

_ = ax.set_title("Polynomial fit of a non-linear feature")
# 设置图的标题为"Polynomial fit of a non-linear feature"，并且忽略返回的matplotlib对象

# %%
# The error bars represent one standard deviation of the predicted gaussian
# distribution of the query points. Notice that the ARD regression captures the
# ground truth the best when using the default parameters in both models, but
# further reducing the `lambda_init` hyperparameter of the Bayesian Ridge can
# reduce its bias (see example
# :ref:`sphx_glr_auto_examples_linear_model_plot_bayesian_ridge_curvefit.py`).
# Finally, due to the intrinsic limitations of a polynomial regression, both
# models fail when extrapolating.
# 错误条代表查询点预测高斯分布的一个标准差。注意，在两个模型中使用默认参数时，ARD回归最能捕捉到真实情况，但进一步减少贝叶斯岭回归的lambda_init超参数可以减少其偏差（参见示例:ref:`sphx_glr_auto_examples_linear_model_plot_bayesian_ridge_curvefit.py`）。最后，由于多项式回归的固有局限性，两个模型在外推时均会失败。
```
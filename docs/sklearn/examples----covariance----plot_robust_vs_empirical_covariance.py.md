# `D:\src\scipysrc\scikit-learn\examples\covariance\plot_robust_vs_empirical_covariance.py`

```
"""
=======================================
Robust vs Empirical covariance estimate
=======================================

The usual covariance maximum likelihood estimate is very sensitive to the
presence of outliers in the data set. In such a case, it would be better to
use a robust estimator of covariance to guarantee that the estimation is
resistant to "erroneous" observations in the data set. [1]_, [2]_

Minimum Covariance Determinant Estimator
----------------------------------------
The Minimum Covariance Determinant estimator is a robust, high-breakdown point
(i.e. it can be used to estimate the covariance matrix of highly contaminated
datasets, up to
:math:`\frac{n_\text{samples} - n_\text{features}-1}{2}` outliers) estimator of
covariance. The idea is to find
:math:`\frac{n_\text{samples} + n_\text{features}+1}{2}`
observations whose empirical covariance has the smallest determinant, yielding
a "pure" subset of observations from which to compute standards estimates of
location and covariance. After a correction step aiming at compensating the
fact that the estimates were learned from only a portion of the initial data,
we end up with robust estimates of the data set location and covariance.

The Minimum Covariance Determinant estimator (MCD) has been introduced by
P.J.Rousseuw in [3]_.

Evaluation
----------
In this example, we compare the estimation errors that are made when using
various types of location and covariance estimates on contaminated Gaussian
distributed data sets:

- The mean and the empirical covariance of the full dataset, which break
  down as soon as there are outliers in the data set
- The robust MCD, that has a low error provided
  :math:`n_\text{samples} > 5n_\text{features}`
- The mean and the empirical covariance of the observations that are known
  to be good ones. This can be considered as a "perfect" MCD estimation,
  so one can trust our implementation by comparing to this case.


References
----------
.. [1] Johanna Hardin, David M Rocke. The distribution of robust distances.
    Journal of Computational and Graphical Statistics. December 1, 2005,
    14(4): 928-946.
.. [2] Zoubir A., Koivunen V., Chakhchoukh Y. and Muma M. (2012). Robust
    estimation in signal processing: A tutorial-style treatment of
    fundamental concepts. IEEE Signal Processing Magazine 29(4), 61-80.
.. [3] P. J. Rousseeuw. Least median of squares regression. Journal of American
    Statistical Ass., 79:871, 1984.

"""

import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np

from sklearn.covariance import EmpiricalCovariance, MinCovDet

# example settings
n_samples = 80  # 总样本数
n_features = 5  # 特征数
repeat = 10  # 重复次数

# range of outliers to generate
range_n_outliers = np.concatenate(
    (
        np.linspace(0, n_samples / 8, 5),  # 生成从0到n_samples/8的5个值
        np.linspace(n_samples / 8, n_samples / 2, 5)[1:-1],  # 生成从n_samples/8到n_samples/2的5个值，去掉第一个和最后一个
    )
).astype(int)

# definition of arrays to store results
err_loc_mcd = np.zeros((range_n_outliers.size, repeat))  # 初始化用于存储结果的数组，大小为(range_n_outliers.size, repeat)
err_cov_mcd = np.zeros((range_n_outliers.size, repeat))
err_loc_emp_full = np.zeros((range_n_outliers.size, repeat))
err_cov_emp_full = np.zeros((range_n_outliers.size, repeat))
err_loc_emp_pure = np.zeros((range_n_outliers.size, repeat))
err_cov_emp_pure = np.zeros((range_n_outliers.size, repeat))

# 初始化存储误差的数组，用于不同情况下的结果存储

# computation
for i, n_outliers in enumerate(range_n_outliers):
    for j in range(repeat):
        rng = np.random.RandomState(i * j)

        # generate data
        # 生成正常数据
        X = rng.randn(n_samples, n_features)
        
        # add some outliers
        # 添加异常值
        outliers_index = rng.permutation(n_samples)[:n_outliers]
        outliers_offset = 10.0 * (
            np.random.randint(2, size=(n_outliers, n_features)) - 0.5
        )
        X[outliers_index] += outliers_offset
        
        # identify inliers
        # 标识内点数据
        inliers_mask = np.ones(n_samples).astype(bool)
        inliers_mask[outliers_index] = False

        # fit a Minimum Covariance Determinant (MCD) robust estimator to data
        # 使用最小协方差行列式（MCD）拟合数据的鲁棒估计器
        mcd = MinCovDet().fit(X)
        # compare raw robust estimates with the true location and covariance
        # 将原始的鲁棒估计与真实的位置和协方差进行比较
        err_loc_mcd[i, j] = np.sum(mcd.location_ ** 2)
        err_cov_mcd[i, j] = mcd.error_norm(np.eye(n_features))

        # compare estimators learned from the full data set with true parameters
        # 将从完整数据集学习的估计器与真实参数进行比较
        err_loc_emp_full[i, j] = np.sum(X.mean(0) ** 2)
        err_cov_emp_full[i, j] = (
            EmpiricalCovariance().fit(X).error_norm(np.eye(n_features))
        )

        # compare with an empirical covariance learned from a pure data set
        # 从纯净数据集学习的经验协方差进行比较（即“完美”的MCD）
        pure_X = X[inliers_mask]
        pure_location = pure_X.mean(0)
        pure_emp_cov = EmpiricalCovariance().fit(pure_X)
        err_loc_emp_pure[i, j] = np.sum(pure_location ** 2)
        err_cov_emp_pure[i, j] = pure_emp_cov.error_norm(np.eye(n_features))

# Display results
# 显示结果
font_prop = matplotlib.font_manager.FontProperties(size=11)
plt.subplot(2, 1, 1)
lw = 2
plt.errorbar(
    range_n_outliers,
    err_loc_mcd.mean(1),
    yerr=err_loc_mcd.std(1) / np.sqrt(repeat),
    label="Robust location",
    lw=lw,
    color="m",
)
plt.errorbar(
    range_n_outliers,
    err_loc_emp_full.mean(1),
    yerr=err_loc_emp_full.std(1) / np.sqrt(repeat),
    label="Full data set mean",
    lw=lw,
    color="green",
)
plt.errorbar(
    range_n_outliers,
    err_loc_emp_pure.mean(1),
    yerr=err_loc_emp_pure.std(1) / np.sqrt(repeat),
    label="Pure data set mean",
    lw=lw,
    color="black",
)
plt.title("Influence of outliers on the location estimation")
plt.ylabel(r"Error ($||\mu - \hat{\mu}||_2^2$)")
plt.legend(loc="upper left", prop=font_prop)

plt.subplot(2, 1, 2)
x_size = range_n_outliers.size
plt.errorbar(
    range_n_outliers,
    err_cov_mcd.mean(1),
    yerr=err_cov_mcd.std(1),
    label="Robust covariance (mcd)",
    color="m",
)
plt.errorbar(
    range_n_outliers[: (x_size // 5 + 1)],
    err_cov_emp_full.mean(1)[: (x_size // 5 + 1)],
    yerr=err_cov_emp_full.std(1)[: (x_size // 5 + 1)],
    label="Full data set covariance",
    color="green",
)
    # 设置标签文本为"Full data set empirical covariance"
    label="Full data set empirical covariance",
    # 设置线条颜色为绿色
    color="green",
)
# 使用 matplotlib 的 plot 函数绘制折线图，横轴为 range_n_outliers 的部分切片，纵轴为 err_cov_emp_full.mean(1) 的部分切片，线条颜色为绿色，线条样式为虚线
plt.plot(
    range_n_outliers[(x_size // 5) : (x_size // 2 - 1)],
    err_cov_emp_full.mean(1)[(x_size // 5) : (x_size // 2 - 1)],
    color="green",
    ls="--",
)
# 使用 matplotlib 的 errorbar 函数绘制带有误差条的折线图，横轴为 range_n_outliers，纵轴为 err_cov_emp_pure.mean(1)，误差条的长度为 err_cov_emp_pure.std(1)，图例标签为"Pure data set empirical covariance"，线条颜色为黑色
plt.errorbar(
    range_n_outliers,
    err_cov_emp_pure.mean(1),
    yerr=err_cov_emp_pure.std(1),
    label="Pure data set empirical covariance",
    color="black",
)
# 设置图表标题为"Influence of outliers on the covariance estimation"
plt.title("Influence of outliers on the covariance estimation")
# 设置横轴标签为"Amount of contamination (%)"
plt.xlabel("Amount of contamination (%)")
# 设置纵轴标签为"RMSE"
plt.ylabel("RMSE")
# 设置图例位置为"upper center"，使用定义的字体属性 font_prop
plt.legend(loc="upper center", prop=font_prop)

# 显示绘制的图表
plt.show()
```
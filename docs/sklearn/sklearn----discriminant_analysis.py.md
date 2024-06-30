# `D:\src\scipysrc\scikit-learn\sklearn\discriminant_analysis.py`

```
# Linear and quadratic discriminant analysis functions.

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 引入警告模块，用于处理潜在的警告信息
import warnings
# 从 numbers 模块导入 Integral 和 Real 类型，用于类型检查
from numbers import Integral, Real

# 导入科学计算库 NumPy
import numpy as np
# 导入 SciPy 的线性代数模块
import scipy.linalg
from scipy import linalg

# 导入基础类及相关混合类
from .base import (
    BaseEstimator,
    ClassifierMixin,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
# 导入协方差估计相关函数
from .covariance import empirical_covariance, ledoit_wolf, shrunk_covariance
# 导入线性分类器的基类混合类
from .linear_model._base import LinearClassifierMixin
# 导入数据预处理模块中的标准化器
from .preprocessing import StandardScaler
# 导入数组操作 API 相关函数
from .utils._array_api import _expit, device, get_namespace, size
# 导入参数验证工具相关函数
from .utils._param_validation import HasMethods, Interval, StrOptions
# 导入数学扩展工具中的 softmax 函数
from .utils.extmath import softmax
# 导入多类别分类相关函数
from .utils.multiclass import check_classification_targets, unique_labels
# 导入验证函数，用于检查对象是否已拟合
from .utils.validation import check_is_fitted

# 声明公开接口，仅包括 LinearDiscriminantAnalysis 和 QuadraticDiscriminantAnalysis
__all__ = ["LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis"]


def _cov(X, shrinkage=None, covariance_estimator=None):
    """Estimate covariance matrix (using optional covariance_estimator).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    shrinkage : {'empirical', 'auto'} or float, default=None
        Shrinkage parameter, possible values:
          - None or 'empirical': no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

        Shrinkage parameter is ignored if  `covariance_estimator`
        is not None.

    covariance_estimator : estimator, default=None
        If not None, `covariance_estimator` is used to estimate
        the covariance matrices instead of relying on the empirical
        covariance estimator (with potential shrinkage).
        The object should have a fit method and a ``covariance_`` attribute
        like the estimators in :mod:`sklearn.covariance``.
        if None the shrinkage parameter drives the estimate.

        .. versionadded:: 0.24

    Returns
    -------
    s : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.
    """
    # 如果未提供自定义的协方差估计器，则使用标准的经验协方差估计方法
    if covariance_estimator is None:
        # 确定收缩参数的值，默认为经验协方差估计（无收缩）
        shrinkage = "empirical" if shrinkage is None else shrinkage
        # 如果收缩参数是字符串类型
        if isinstance(shrinkage, str):
            # 自动收缩方法
            if shrinkage == "auto":
                # 标准化特征
                sc = StandardScaler()
                X = sc.fit_transform(X)
                # Ledoit-Wolf方法估计协方差矩阵
                s = ledoit_wolf(X)[0]
                # 重新缩放协方差矩阵
                s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
            # 经验协方差估计方法
            elif shrinkage == "empirical":
                s = empirical_covariance(X)
        # 如果收缩参数是实数类型
        elif isinstance(shrinkage, Real):
            # 使用固定的收缩参数来进行协方差估计
            s = shrunk_covariance(empirical_covariance(X), shrinkage)
    else:
        # 如果条件不满足前面的情况，执行以下代码块
        if shrinkage is not None and shrinkage != 0:
            # 如果 shrinkage 参数不为 None 并且不为 0，则抛出数值错误异常
            raise ValueError(
                "covariance_estimator and shrinkage parameters "
                "are not None. Only one of the two can be set."
            )
        # 使用输入数据 X 对协方差估计器进行拟合
        covariance_estimator.fit(X)
        # 检查协方差估计器是否具有 covariance_ 属性
        if not hasattr(covariance_estimator, "covariance_"):
            # 如果没有 covariance_ 属性，则抛出数值错误异常，显示缺少的属性名
            raise ValueError(
                "%s does not have a covariance_ attribute"
                % covariance_estimator.__class__.__name__
            )
        # 获取协方差估计器的协方差矩阵
        s = covariance_estimator.covariance_
    # 返回协方差矩阵 s
    return s
def _class_means(X, y):
    """Compute class means.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.

    Returns
    -------
    means : array-like of shape (n_classes, n_features)
        Class means.
    """
    xp, is_array_api_compliant = get_namespace(X)
    # 获取数据类型的命名空间和数组API兼容性标志

    classes, y = xp.unique_inverse(y)
    # 获取唯一的类别值和反向映射的目标值

    means = xp.zeros((classes.shape[0], X.shape[1]), device=device(X), dtype=X.dtype)
    # 创建全零数组以存储类均值，形状为 (类别数, 特征数)，设备与输入数据相同，数据类型与输入数据相同

    if is_array_api_compliant:
        # 如果数组API兼容
        for i in range(classes.shape[0]):
            means[i, :] = xp.mean(X[y == i], axis=0)
            # 计算每个类别的均值并存储到means数组中
    else:
        # 如果不兼容数组API
        # TODO: 探索使用bincount + add.at的选择，因为看起来性能上不太优化
        cnt = np.bincount(y)
        # 计算每个类别的样本数
        np.add.at(means, y, X)
        # 使用add.at方法累加每个类别的特征值
        means /= cnt[:, None]
        # 将累加值除以每个类别的样本数，计算均值

    return means
    # 返回计算得到的类均值数组


def _class_cov(X, y, priors, shrinkage=None, covariance_estimator=None):
    """Compute weighted within-class covariance matrix.

    The per-class covariance are weighted by the class priors.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.

    priors : array-like of shape (n_classes,)
        Class priors.

    shrinkage : 'auto' or float, default=None
        Shrinkage parameter, possible values:
          - None: no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

        Shrinkage parameter is ignored if `covariance_estimator` is not None.

    covariance_estimator : estimator, default=None
        If not None, `covariance_estimator` is used to estimate
        the covariance matrices instead of relying the empirical
        covariance estimator (with potential shrinkage).
        The object should have a fit method and a ``covariance_`` attribute
        like the estimators in sklearn.covariance.
        If None, the shrinkage parameter drives the estimate.

        .. versionadded:: 0.24

    Returns
    -------
    cov : array-like of shape (n_features, n_features)
        Weighted within-class covariance matrix
    """
    classes = np.unique(y)
    # 获取唯一的类别值

    cov = np.zeros(shape=(X.shape[1], X.shape[1]))
    # 创建全零的协方差矩阵，形状为 (特征数, 特征数)

    for idx, group in enumerate(classes):
        Xg = X[y == group, :]
        # 获取每个类别的数据子集

        cov += priors[idx] * np.atleast_2d(_cov(Xg, shrinkage, covariance_estimator))
        # 计算加权的类内协方差矩阵，根据类先验权重加权

    return cov
    # 返回加权的类内协方差矩阵


class LinearDiscriminantAnalysis(
    ClassNamePrefixFeaturesOutMixin,
    LinearClassifierMixin,
    TransformerMixin,
    BaseEstimator,
):
    """Linear Discriminant Analysis.

    A classifier with a linear decision boundary, generated by fitting class
    conditional densities to the data and using Bayes' rule.

    The model fits a Gaussian density to each class, assuming that all classes
    share the same covariance matrix.
    """
    # 线性判别分析类的定义，实现线性决策边界分类器，并使用贝叶斯规则生成
    # 该模型可以通过使用 `transform` 方法将输入数据投影到最具判别性的方向，从而减少其维度。

    # 版本新增：0.17

    # 要比较 `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` 和 `sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`，请参阅 :ref:`sphx_glr_auto_examples_classification_plot_lda_qda.py`。

    # 在 :ref:`User Guide <lda_qda>` 中查看更多信息。

    Parameters
    ----------
    solver : {'svd', 'lsqr', 'eigen'}, default='svd'
        要使用的求解器，可能的取值：
          - 'svd': 奇异值分解（默认）。
            不计算协方差矩阵，因此推荐用于具有大量特征的数据。
          - 'lsqr': 最小二乘解法。
            可与收缩或自定义协方差估计器结合使用。
          - 'eigen': 特征值分解。
            可与收缩或自定义协方差估计器结合使用。

        .. versionchanged:: 1.2
            `solver="svd"` 现在具有实验性的 Array API 支持。详细信息请参阅 :ref:`Array API User Guide <array_api>`。

    shrinkage : 'auto' 或 float, default=None
        收缩参数，可能的取值：
          - None: 不进行收缩（默认）。
          - 'auto': 使用 Ledoit-Wolf 引理进行自动收缩。
          - 0 到 1 之间的浮点数: 固定的收缩参数。

        如果使用 `covariance_estimator`，应将此参数保留为 None。
        注意，收缩仅在 'lsqr' 和 'eigen' 求解器中有效。

        查看示例用法，请参阅 :ref:`sphx_glr_auto_examples_classification_plot_lda.py`。

    priors : 形状为 (n_classes,) 的数组，默认为 None
        类的先验概率。默认情况下，从训练数据推断类比例。

    n_components : int，默认为 None
        维度减少的组件数（<= min(n_classes - 1, n_features)）。如果为 None，则将设置为 min(n_classes - 1, n_features)。此参数仅影响 `transform` 方法。

        查看示例用法，请参阅 :ref:`sphx_glr_auto_examples_decomposition_plot_pca_vs_lda.py`。

    store_covariance : bool，默认为 False
        如果为 True，在求解器为 'svd' 时显式计算加权类内协方差矩阵。对于其他求解器，该矩阵始终会被计算和存储。

        .. versionadded:: 0.17

    tol : float，默认为 1.0e-4
        用于估计 X 的秩的奇异值的绝对阈值，以确定哪些维度的奇异值不显著。仅在求解器为 'svd' 时使用。

        .. versionadded:: 0.17
    # covariance_estimator : covariance estimator, default=None
    # 如果不为 None，将使用 covariance_estimator 来估计协方差矩阵，而不是依赖于经验协方差估计器（可能会收缩）。
    # 这个对象应该有一个 fit 方法和一个 `covariance_` 属性，类似于 sklearn.covariance 模块中的估计器。
    # 如果 shrinkage 参数被使用，应该将其保持为 None。
    # 注意，covariance_estimator 只能与 'lsqr' 和 'eigen' 求解器一起使用。
    # .. versionadded:: 0.24

Attributes
----------
    # coef_ : ndarray of shape (n_features,) or (n_classes, n_features)
    # 权重向量（们）。

    # intercept_ : ndarray of shape (n_classes,)
    # 截距项。

    # covariance_ : array-like of shape (n_features, n_features)
    # 加权类内协方差矩阵。它对应于 `sum_k prior_k * C_k`，其中 `C_k` 是类 `k` 中样本的协方差矩阵。
    # `C_k` 是使用（可能收缩的）有偏估计协方差计算的。如果 solver 是 'svd'，仅在 `store_covariance` 为 True 时存在。

    # explained_variance_ratio_ : ndarray of shape (n_components,)
    # 每个选定成分解释的方差百分比。如果未设置 `n_components`，则存储所有成分，解释方差之和为 1.0。
    # 仅在使用 eigen 或 svd 求解器时可用。

    # means_ : array-like of shape (n_classes, n_features)
    # 类别均值。

    # priors_ : array-like of shape (n_classes,)
    # 类先验（总和为 1）。

    # scalings_ : array-like of shape (rank, n_classes - 1)
    # 在由类别质心张成的空间中特征的缩放。仅在 'svd' 和 'eigen' 求解器下可用。

    # xbar_ : array-like of shape (n_features,)
    # 总体均值。仅在 solver 是 'svd' 时存在。

    # classes_ : array-like of shape (n_classes,)
    # 唯一的类标签。

    # n_features_in_ : int
    # 在拟合期间看到的特征数。

    # .. versionadded:: 0.24

    # feature_names_in_ : ndarray of shape (`n_features_in_`,)
    # 在 `fit` 期间看到的特征名称。仅当 `X` 具有全部为字符串的特征名时定义。

See Also
--------
    # QuadraticDiscriminantAnalysis : 二次判别分析。

Examples
--------
    # >>> import numpy as np
    # >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    # >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    # >>> y = np.array([1, 1, 1, 2, 2, 2])
    # >>> clf = LinearDiscriminantAnalysis()
    # >>> clf.fit(X, y)
    # LinearDiscriminantAnalysis()
    # >>> print(clf.predict([[-0.8, -1]]))
    # [1]
    # 参数约束字典，定义了每个参数可接受的值或类型范围
    _parameter_constraints: dict = {
        "solver": [StrOptions({"svd", "lsqr", "eigen"})],  # solver参数应为{"svd", "lsqr", "eigen"}中的一个字符串
        "shrinkage": [StrOptions({"auto"}), Interval(Real, 0, 1, closed="both"), None],  # shrinkage参数可为"auto"字符串、0到1之间的实数闭区间，或为None
        "n_components": [Interval(Integral, 1, None, closed="left"), None],  # n_components参数为大于等于1的整数或为None
        "priors": ["array-like", None],  # priors参数为类似数组的对象或为None
        "store_covariance": ["boolean"],  # store_covariance参数应为布尔值
        "tol": [Interval(Real, 0, None, closed="left")],  # tol参数为大于等于0的实数或为None
        "covariance_estimator": [HasMethods("fit"), None],  # covariance_estimator参数应具有"fit"方法的对象或为None
    }

    # 初始化方法，设定各参数的初始值
    def __init__(
        self,
        solver="svd",  # 默认使用svd求解器
        shrinkage=None,  # 默认不使用shrinkage
        priors=None,  # 默认无先验信息
        n_components=None,  # 默认不指定主成分数目
        store_covariance=False,  # 默认不存储协方差矩阵（仅在svd求解器中使用）
        tol=1e-4,  # 默认容差为1e-4（仅在svd求解器中使用）
        covariance_estimator=None,  # 默认不指定协方差估计器
    ):
        self.solver = solver  # 将传入的solver参数赋给实例的solver属性
        self.shrinkage = shrinkage  # 将传入的shrinkage参数赋给实例的shrinkage属性
        self.priors = priors  # 将传入的priors参数赋给实例的priors属性
        self.n_components = n_components  # 将传入的n_components参数赋给实例的n_components属性
        self.store_covariance = store_covariance  # 将传入的store_covariance参数赋给实例的store_covariance属性
        self.tol = tol  # 将传入的tol参数赋给实例的tol属性
        self.covariance_estimator = covariance_estimator  # 将传入的covariance_estimator参数赋给实例的covariance_estimator属性
    def _solve_lstsq(self, X, y, shrinkage, covariance_estimator):
        """
        Least squares solver.

        The least squares solver computes a straightforward solution of the
        optimal decision rule based directly on the discriminant functions. It
        can only be used for classification (with any covariance estimator),
        because estimation of eigenvectors is not performed. Therefore, dimensionality
        reduction with the transform is not supported.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_classes)
            Target values.

        shrinkage : 'auto', float or None
            Shrinkage parameter, possible values:
              - None: no shrinkage.
              - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
              - float between 0 and 1: fixed shrinkage parameter.

            Shrinkage parameter is ignored if `covariance_estimator` is not None.

        covariance_estimator : estimator, default=None
            If not None, `covariance_estimator` is used to estimate
            the covariance matrices instead of relying on the empirical
            covariance estimator (with potential shrinkage).
            The object should have a fit method and a ``covariance_`` attribute
            like the estimators in sklearn.covariance.
            If None, the shrinkage parameter drives the estimate.

            .. versionadded:: 0.24

        Notes
        -----
        This solver is based on [1]_, section 2.6.2, pp. 39-41.

        References
        ----------
        .. [1] R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification
           (Second Edition). John Wiley & Sons, Inc., New York, 2001. ISBN
           0-471-05669-3.
        """
        # 计算类均值
        self.means_ = _class_means(X, y)
        # 计算类协方差矩阵
        self.covariance_ = _class_cov(
            X, y, self.priors_, shrinkage, covariance_estimator
        )
        # 使用最小二乘法计算系数
        self.coef_ = linalg.lstsq(self.covariance_, self.means_.T)[0].T
        # 计算截距
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )
        self.means_ = _class_means(X, y)
        # 计算类别均值向量

        self.covariance_ = _class_cov(
            X, y, self.priors_, shrinkage, covariance_estimator
        )
        # 计算类别协方差矩阵

        Sw = self.covariance_  # within scatter
        # 计算类内散布矩阵

        St = _cov(X, shrinkage, covariance_estimator)  # total scatter
        # 计算总散布矩阵

        Sb = St - Sw  # between scatter
        # 计算类间散布矩阵

        evals, evecs = linalg.eigh(Sb, Sw)
        # 使用广义特征值问题求解 Rayleigh 系数的最优解

        self.explained_variance_ratio_ = np.sort(evals / np.sum(evals))[::-1][
            : self._max_components
        ]
        # 计算解释方差比率，按降序排列并选取前 self._max_components 个

        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
        # 对特征向量按特征值降序排序

        self.scalings_ = evecs
        # 保存特征向量作为变换矩阵的列

        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        # 计算线性变换的系数

        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )
        # 计算线性变换的截距
    # 使用奇异值分解（SVD）求解器
    def _solve_svd(self, X, y):
        """SVD solver.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            训练数据。

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            目标值。
        """
        # 获取数组操作的命名空间和是否符合数组API的标志
        xp, is_array_api_compliant = get_namespace(X)

        # 根据数组API的标志选择使用哪个SVD函数
        if is_array_api_compliant:
            svd = xp.linalg.svd
        else:
            svd = scipy.linalg.svd

        # 获取样本数和特征数
        n_samples, n_features = X.shape
        # 获取类别数
        n_classes = self.classes_.shape[0]

        # 计算各类别的均值
        self.means_ = _class_means(X, y)
        # 如果需要存储协方差矩阵，则计算并存储
        if self.store_covariance:
            self.covariance_ = _class_cov(X, y, self.priors_)

        # 将每个类别中心化后的数据存储在Xc列表中
        Xc = []
        for idx, group in enumerate(self.classes_):
            Xg = X[y == group]
            Xc.append(Xg - self.means_[idx, :])

        # 计算加权均值
        self.xbar_ = self.priors_ @ self.means_

        # 将Xc列表中的数组按行连接成一个大数组
        Xc = xp.concatenate(Xc, axis=0)

        # 1) 对每个特征按类别内标准差进行缩放
        std = xp.std(Xc, axis=0)
        # 避免在标准化时出现除以零的情况
        std[std == 0] = 1.0
        fac = xp.asarray(1.0 / (n_samples - n_classes))

        # 2) 类内方差缩放
        X = xp.sqrt(fac) * (Xc / std)
        # 对中心化并缩放后的数据进行SVD分解
        U, S, Vt = svd(X, full_matrices=False)

        # 计算秩
        rank = xp.sum(xp.astype(S > self.tol, xp.int32))
        # 类内协方差的缩放为：V' / std / S
        scalings = (Vt[:rank, :] / std).T / S[:rank]
        fac = 1.0 if n_classes == 1 else 1.0 / (n_classes - 1)

        # 3) 类间方差缩放
        # 缩放加权的中心
        X = (
            (xp.sqrt((n_samples * self.priors_) * fac)) * (self.means_ - self.xbar_).T
        ).T @ scalings
        # 中心点位于一个最多有 n_classes-1 维的空间中
        # 使用SVD找到在由 n_classes 个中心点张成的空间中的投影
        _, S, Vt = svd(X, full_matrices=False)

        # 如果最大组件数为0，则解释方差比例为空数组
        if self._max_components == 0:
            self.explained_variance_ratio_ = xp.empty((0,), dtype=S.dtype)
        else:
            self.explained_variance_ratio_ = (S**2 / xp.sum(S**2))[
                : self._max_components
            ]

        # 计算秩
        rank = xp.sum(xp.astype(S > self.tol * S[0], xp.int32))
        # 计算缩放后的系数
        self.scalings_ = scalings @ Vt.T[:, :rank]
        # 计算系数
        coef = (self.means_ - self.xbar_) @ self.scalings_
        # 计算截距
        self.intercept_ = -0.5 * xp.sum(coef**2, axis=1) + xp.log(self.priors_)
        # 计算系数
        self.coef_ = coef @ self.scalings_.T
        # 计算截距
        self.intercept_ -= self.xbar_ @ self.coef_.T

    @_fit_context(
        # LinearDiscriminantAnalysis.covariance_estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def transform(self, X):
        """Project data to maximize class separation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components) or \
            (n_samples, min(rank, n_components))
            Transformed data. In the case of the 'svd' solver, the shape
            is (n_samples, min(rank, n_components)).
        """
        # 如果使用 'lsqr' 解算器，则抛出未实现错误
        if self.solver == "lsqr":
            raise NotImplementedError(
                "transform not implemented for 'lsqr' solver (use 'svd' or 'eigen')."
            )
        # 检查模型是否已拟合
        check_is_fitted(self)
        # 获取命名空间和数组兼容性信息
        xp, _ = get_namespace(X)
        # 验证输入数据，不重置
        X = self._validate_data(X, reset=False)

        # 根据解算器类型进行数据变换
        if self.solver == "svd":
            # 使用 SVD 解算器进行数据变换
            X_new = (X - self.xbar_) @ self.scalings_
        elif self.solver == "eigen":
            # 使用特征值分解解算器进行数据变换
            X_new = X @ self.scalings_

        # 返回变换后的数据，截取到最大组件数
        return X_new[:, : self._max_components]

    def predict_proba(self, X):
        """Estimate probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Estimated probabilities.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)
        # 获取命名空间和数组兼容性信息
        xp, is_array_api_compliant = get_namespace(X)
        # 计算决策函数
        decision = self.decision_function(X)
        # 如果类别数为2，使用 logistic 函数计算概率
        if size(self.classes_) == 2:
            proba = _expit(decision, xp)
            return xp.stack([1 - proba, proba], axis=1)
        else:
            # 对于多类别情况，使用 softmax 函数计算概率
            return softmax(decision)

    def predict_log_proba(self, X):
        """Estimate log probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Estimated log probabilities.
        """
        # 获取命名空间和数组兼容性信息
        xp, _ = get_namespace(X)
        # 预测类别概率
        prediction = self.predict_proba(X)

        # 获取预测结果数据类型的信息
        info = xp.finfo(prediction.dtype)
        # 获取最小正常值或者 tiny 值（适用于 NumPy 1.22+）
        if hasattr(info, "smallest_normal"):
            smallest_normal = info.smallest_normal
        else:
            smallest_normal = info.tiny

        # 处理概率为零的情况，避免取对数出现问题
        prediction[prediction == 0.0] += smallest_normal
        # 返回对数概率
        return xp.log(prediction)
    # 覆盖父类方法，将决策函数应用于样本数组
    def decision_function(self, X):
        """Apply decision function to an array of samples.

        The decision function is equal (up to a constant factor) to the
        log-posterior of the model, i.e. `log p(y = k | x)`. In a binary
        classification setting this instead corresponds to the difference
        `log p(y = 1 | x) - log p(y = 0 | x)`. See :ref:`lda_qda_math`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Array of samples (test vectors).

        Returns
        -------
        C : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function values related to each class, per sample.
            In the two-class case, the shape is (n_samples,), giving the
            log likelihood ratio of the positive class.
        """
        # 仅为了文档说明而覆盖的方法，实际上调用父类的决策函数
        return super().decision_function(X)

    # 返回一个字典，指定此类支持数组 API
    def _more_tags(self):
        return {"array_api_support": True}
class QuadraticDiscriminantAnalysis(ClassifierMixin, BaseEstimator):
    """Quadratic Discriminant Analysis.

    A classifier with a quadratic decision boundary, generated
    by fitting class conditional densities to the data
    and using Bayes' rule.

    The model fits a Gaussian density to each class.

    .. versionadded:: 0.17

    For a comparison between
    :class:`~sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`
    and :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis`, see
    :ref:`sphx_glr_auto_examples_classification_plot_lda_qda.py`.

    Read more in the :ref:`User Guide <lda_qda>`.

    Parameters
    ----------
    priors : array-like of shape (n_classes,), default=None
        Class priors. By default, the class proportions are inferred from the
        training data.

    reg_param : float, default=0.0
        Regularizes the per-class covariance estimates by transforming S2 as
        ``S2 = (1 - reg_param) * S2 + reg_param * np.eye(n_features)``,
        where S2 corresponds to the `scaling_` attribute of a given class.

    store_covariance : bool, default=False
        If True, the class covariance matrices are explicitly computed and
        stored in the `self.covariance_` attribute.

        .. versionadded:: 0.17

    tol : float, default=1.0e-4
        Absolute threshold for the covariance matrix to be considered rank
        deficient after applying some regularization (see `reg_param`) to each
        `Sk` where `Sk` represents covariance matrix for k-th class. This
        parameter does not affect the predictions. It controls when a warning
        is raised if the covariance matrix is not full rank.

        .. versionadded:: 0.17

    Attributes
    ----------
    covariance_ : list of len n_classes of ndarray \
            of shape (n_features, n_features)
        For each class, gives the covariance matrix estimated using the
        samples of that class. The estimations are unbiased. Only present if
        `store_covariance` is True.

    means_ : array-like of shape (n_classes, n_features)
        Class-wise means.

    priors_ : array-like of shape (n_classes,)
        Class priors (sum to 1).

    rotations_ : list of len n_classes of ndarray of shape (n_features, n_k)
        For each class k an array of shape (n_features, n_k), where
        ``n_k = min(n_features, number of elements in class k)``
        It is the rotation of the Gaussian distribution, i.e. its
        principal axis. It corresponds to `V`, the matrix of eigenvectors
        coming from the SVD of `Xk = U S Vt` where `Xk` is the centered
        matrix of samples from class k.
    """

    def __init__(self, priors=None, reg_param=0.0, store_covariance=False, tol=1.0e-4):
        # 初始化方法，设置分类器的参数
        self.priors = priors
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        self.tol = tol

    def fit(self, X, y):
        # 根据输入的训练数据 X 和标签 y 训练 QDA 模型
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # 初始化需要计算的属性
        self.means_ = np.zeros((n_classes, n_features))
        self.covariance_ = [None] * n_classes
        self.priors_ = np.zeros(n_classes)

        # 计算每个类别的样本均值和先验概率
        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.means_[i, :] = Xc.mean(axis=0)
            self.priors_[i] = float(len(Xc)) / n_samples

            if self.store_covariance:
                self.covariance_[i] = np.cov(Xc.T, bias=True)

                # 正则化协方差矩阵
                shrinkage = self.reg_param * np.eye(n_features)
                self.covariance_[i] = (1.0 - self.reg_param) * self.covariance_[i] + shrinkage

        # 计算旋转矩阵
        self.rotations_ = []
        for i, cov in enumerate(self.covariance_):
            if cov is None:
                n_k = 1
            else:
                n_k = min(n_features, np.linalg.matrix_rank(cov))

            _, V = np.linalg.eigh(cov)
            self.rotations_.append(V[:, :n_k])

        return self

    def predict_proba(self, X):
        # 根据训练好的 QDA 模型，预测输入数据 X 的类别概率
        if not hasattr(self, 'means_'):
            raise NotFittedError("Estimator not fitted, call `fit` first")

        log_proba = np.zeros((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            if self.store_covariance:
                # 计算类别 c 的对数概率
                n_k = self.covariance_[i].shape[0]
                mahalanobis = np.sum((X - self.means_[i]) @ np.linalg.pinv(self.covariance_[i]) * (X - self.means_[i]), axis=1)
                log_proba[:, i] = -0.5 * (mahalanobis + np.log(np.linalg.det(self.covariance_[i]))) - np.log(self.priors_[i])
            else:
                # 计算类别 c 的对数概率（假设所有类别的协方差矩阵相同）
                mahalanobis = np.sum((X - self.means_[i]) ** 2 / self.covariance_)  # assuming diagonal covariance matrix
                log_proba[:, i] = -0.5 * mahalanobis - np.log(self.priors_[i])

        # 计算类别概率
        proba = np.exp(log_proba - log_proba.max(axis=1)[:, np.newaxis])
        proba /= proba.sum(axis=1)[:, np.newaxis]

        return proba

    def predict(self, X):
        # 根据训练好的 QDA 模型，预测输入数据 X 的类别
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    # `scalings_` 是一个列表，其长度为类别数目 `n_classes`，每个元素都是一个形状为 (n_k,) 的 ndarray。
    # 每个元素表示对应类别的高斯分布在其主轴上的缩放，即在旋转坐标系中的方差。
    # 其值计算为 `S^2 / (n_samples - 1)`，其中 `S` 是从 `Xk` 的奇异值分解（SVD）中得到的奇异值对角矩阵，
    # 其中 `Xk` 是从类别 k 的样本中心化得到的矩阵。
    scalings_ : list of len n_classes of ndarray of shape (n_k,)

    # `classes_` 是一个形状为 (n_classes,) 的 ndarray，存储了唯一的类别标签。
    classes_ : ndarray of shape (n_classes,)

    # `n_features_in_` 是一个整数，表示在拟合过程中看到的特征数目。
    # 这是在版本 0.24 中添加的功能。
    n_features_in_ : int

    # `feature_names_in_` 是一个形状为 (n_features_in_,) 的 ndarray。
    # 它存储了在拟合过程中看到的特征名称，仅在 `X` 具有全部为字符串的特征名时定义。
    # 这是在版本 1.0 中添加的功能。
    feature_names_in_ : ndarray of shape (n_features_in_,)

    # 以下是一些相关的链接和示例，显示了如何使用这些特性：
    # LinearDiscriminantAnalysis : 线性判别分析。
    #
    # 示例：
    # >>> from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    # >>> import numpy as np
    # >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    # >>> y = np.array([1, 1, 1, 2, 2, 2])
    # >>> clf = QuadraticDiscriminantAnalysis()
    # >>> clf.fit(X, y)
    # QuadraticDiscriminantAnalysis()
    # >>> print(clf.predict([[-0.8, -1]]))
    # [1]

    _parameter_constraints: dict = {
        # 参数 `priors` 应为数组或 None 类型。
        "priors": ["array-like", None],
        # 参数 `reg_param` 应为介于 [0, 1] 区间闭区间的实数。
        "reg_param": [Interval(Real, 0, 1, closed="both")],
        # 参数 `store_covariance` 应为布尔值。
        "store_covariance": ["boolean"],
        # 参数 `tol` 应为大于等于 0 的实数。
        "tol": [Interval(Real, 0, None, closed="left")],
    }

    # 初始化方法，定义了一些参数的默认值。
    def __init__(
        self, *, priors=None, reg_param=0.0, store_covariance=False, tol=1.0e-4
    ):
        self.priors = priors
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        self.tol = tol

    # 使用装饰器 `_fit_context` 进行装饰，传入参数 `prefer_skip_nested_validation=True`。
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the model according to the given training data and parameters.

            .. versionchanged:: 0.19
               ``store_covariances`` has been moved to main constructor as
               ``store_covariance``

            .. versionchanged:: 0.19
               ``tol`` has been moved to main constructor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values (integers).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate and preprocess the input data
        X, y = self._validate_data(X, y)
        # Check if y contains valid classification targets
        check_classification_targets(y)
        # Identify unique classes in y and convert y to indices
        self.classes_, y = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        # Ensure there are at least two classes for classification
        if n_classes < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % (n_classes)
            )
        
        # Compute class priors if not provided
        if self.priors is None:
            self.priors_ = np.bincount(y) / float(n_samples)
        else:
            self.priors_ = np.array(self.priors)

        cov = None
        store_covariance = self.store_covariance
        
        # Prepare to store covariance matrices if specified
        if store_covariance:
            cov = []
        
        means = []
        scalings = []
        rotations = []
        
        # Iterate over each class to compute mean, covariance, and other statistics
        for ind in range(n_classes):
            Xg = X[y == ind, :]
            meang = Xg.mean(0)
            means.append(meang)
            
            # Ensure each class has more than one sample for valid covariance
            if len(Xg) == 1:
                raise ValueError(
                    "y has only 1 sample in class %s, covariance is ill defined."
                    % str(self.classes_[ind])
                )
            
            Xgc = Xg - meang
            
            # Compute Singular Value Decomposition (SVD) of centered data
            _, S, Vt = np.linalg.svd(Xgc, full_matrices=False)
            S2 = (S**2) / (len(Xg) - 1)
            S2 = ((1 - self.reg_param) * S2) + self.reg_param
            
            # Calculate rank of covariance matrix to check for collinearity issues
            rank = np.sum(S2 > self.tol)
            if rank < n_features:
                warnings.warn(
                    f"The covariance matrix of class {ind} is not full rank. "
                    "Increasing the value of parameter `reg_param` might help"
                    " reducing the collinearity.",
                    linalg.LinAlgWarning,
                )
            
            # Store covariance matrices if specified
            if self.store_covariance or store_covariance:
                # cov = V * (S^2 / (n-1)) * V.T
                cov.append(np.dot(S2 * Vt.T, Vt))
            
            scalings.append(S2)
            rotations.append(Vt.T)
        
        # Assign computed values to attributes of the estimator object
        if self.store_covariance or store_covariance:
            self.covariance_ = cov
        
        self.means_ = np.asarray(means)
        self.scalings_ = scalings
        self.rotations_ = rotations
        
        return self
    def _decision_function(self, X):
        # 返回后验对数，参见《ESL》第110页的公式(4.12)。
        # 检查模型是否已拟合
        check_is_fitted(self)

        # 验证输入数据 X 的格式，并不重置
        X = self._validate_data(X, reset=False)
        
        # 初始化一个空列表，用于存储每个类别的范数平方
        norm2 = []
        
        # 对每个类别进行循环
        for i in range(len(self.classes_)):
            # 获取当前类别的旋转矩阵 R、缩放矩阵 S 和均值向量
            R = self.rotations_[i]
            S = self.scalings_[i]
            Xm = X - self.means_[i]
            
            # 计算转换后的数据 X2
            X2 = np.dot(Xm, R * (S ** (-0.5)))
            
            # 计算 X2 的范数平方，并添加到 norm2 列表中
            norm2.append(np.sum(X2**2, axis=1))
        
        # 将 norm2 转换为 NumPy 数组，并转置以匹配形状 [len(X), n_classes]
        norm2 = np.array(norm2).T
        
        # 计算每个类别的 u 值，即缩放矩阵 S 的对数之和
        u = np.asarray([np.sum(np.log(s)) for s in self.scalings_])
        
        # 返回最终的后验对数值，按照公式组合 norm2、u 和先验概率 priors_
        return -0.5 * (norm2 + u) + np.log(self.priors_)

    def decision_function(self, X):
        """对样本数组应用决策函数。

        决策函数等价于模型的后验对数（经过常数倍数调整），即 `log p(y = k | x)`。
        在二分类设置中，这相当于差值 `log p(y = 1 | x) - log p(y = 0 | x)`。
        参见 :ref:`lda_qda_math`。

        Parameters
        ----------
        X : shape 为 (n_samples, n_features) 的数组
            样本数组（测试向量）。

        Returns
        -------
        C : shape 为 (n_samples,) 或 (n_samples, n_classes) 的 ndarray
            每个样本相关的决策函数值。
            在二分类情况下，形状为 (n_samples,)，表示正类的对数似然比。
        """
        # 调用内部的 _decision_function 方法获得决策函数的输出
        dec_func = self._decision_function(X)
        
        # 处理二分类的特殊情况
        if len(self.classes_) == 2:
            return dec_func[:, 1] - dec_func[:, 0]
        
        # 多分类情况下直接返回决策函数的输出
        return dec_func

    def predict(self, X):
        """对测试向量 X 执行分类。

        返回每个样本在 X 中的预测类别 C。

        Parameters
        ----------
        X : shape 为 (n_samples, n_features) 的数组
            待评分的向量，其中 `n_samples` 是样本数，`n_features` 是特征数。

        Returns
        -------
        C : shape 为 (n_samples,) 的 ndarray
            估计的概率。
        """
        # 调用内部的 _decision_function 方法获取决策函数值
        d = self._decision_function(X)
        
        # 根据决策函数的输出确定每个样本的预测类别
        y_pred = self.classes_.take(d.argmax(1))
        
        # 返回预测结果
        return y_pred
    # 返回分类的后验概率。
    # 参数 X: 形状为 (n_samples, n_features) 的样本/测试向量数组。
    def predict_proba(self, X):
        # 调用 _decision_function 方法计算分类的值。
        values = self._decision_function(X)
        # 计算基于高斯模型的似然性，但只计算到一个乘法常数。
        likelihood = np.exp(values - values.max(axis=1)[:, np.newaxis])
        # 计算后验概率。
        return likelihood / likelihood.sum(axis=1)[:, np.newaxis]

    # 返回分类的后验概率的对数。
    # 参数 X: 形状为 (n_samples, n_features) 的样本/测试向量数组。
    def predict_log_proba(self, X):
        # XXX : 可以优化以避免精度溢出。
        # 调用 predict_proba 方法获取后验概率。
        probas_ = self.predict_proba(X)
        # 返回后验概率的对数。
        return np.log(probas_)
```
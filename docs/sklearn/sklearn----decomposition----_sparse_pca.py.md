# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\_sparse_pca.py`

```
"""Matrix factorization with Sparse PCA."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Integral, Real  # 导入 Integral 和 Real 类型，用于参数验证

import numpy as np  # 导入 NumPy 库

from ..base import (
    BaseEstimator,  # 导入基础估计器类
    ClassNamePrefixFeaturesOutMixin,  # 导入带类名前缀特征输出的混合类
    TransformerMixin,  # 导入转换器混合类
    _fit_context,  # 导入内部使用的拟合上下文函数
)
from ..linear_model import ridge_regression  # 导入岭回归函数
from ..utils import check_random_state  # 导入随机状态检查函数
from ..utils._param_validation import Hidden, Interval, StrOptions  # 导入参数验证相关模块
from ..utils.extmath import svd_flip  # 导入奇异值分解矩阵翻转函数
from ..utils.validation import check_array, check_is_fitted  # 导入数组验证和拟合检查函数
from ._dict_learning import MiniBatchDictionaryLearning, dict_learning  # 导入字典学习相关类

class _BaseSparsePCA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Base class for SparsePCA and MiniBatchSparsePCA"""

    _parameter_constraints: dict = {
        "n_components": [None, Interval(Integral, 1, None, closed="left")],  # 参数约束字典，指定 n_components 的范围
        "alpha": [Interval(Real, 0.0, None, closed="left")],  # 指定 alpha 的范围
        "ridge_alpha": [Interval(Real, 0.0, None, closed="left")],  # 指定 ridge_alpha 的范围
        "max_iter": [Interval(Integral, 0, None, closed="left")],  # 指定 max_iter 的范围
        "tol": [Interval(Real, 0.0, None, closed="left")],  # 指定 tol 的范围
        "method": [StrOptions({"lars", "cd"})],  # 指定 method 的选项为 {"lars", "cd"}
        "n_jobs": [Integral, None],  # 指定 n_jobs 的类型为整数或 None
        "verbose": ["verbose"],  # 指定 verbose 参数为 "verbose"
        "random_state": ["random_state"],  # 指定 random_state 参数为 "random_state"
    }

    def __init__(
        self,
        n_components=None,
        *,
        alpha=1,
        ridge_alpha=0.01,
        max_iter=1000,
        tol=1e-8,
        method="lars",
        n_jobs=None,
        verbose=False,
        random_state=None,
    ):
        """
        Initialize the SparsePCA base estimator.

        Parameters
        ----------
        n_components : int or None, default=None
            Number of sparse atoms to extract. If None, n_components is set
            to the number of features.

        alpha : float, default=1
            Sparsity controlling parameter. Higher values lead to sparser
            components.

        ridge_alpha : float, default=0.01
            Amount of ridge shrinkage to apply in order to improve
            conditioning when calling ridge_regression.

        max_iter : int, default=1000
            Maximum number of iterations to perform.

        tol : float, default=1e-8
            Tolerance for the optimization.

        method : {'lars', 'cd'}, default='lars'
            Optimization method to use. 'lars' uses the least angle regression
            method while 'cd' uses coordinate descent.

        n_jobs : int or None, default=None
            Number of parallel jobs to run. None means 1 unless in a
            joblib.parallel_backend context. -1 means using all processors.

        verbose : bool, default=False
            Enable verbose output.

        random_state : int, RandomState instance or None, default=None
            Controls the random seed given at initialization.

        """
        self.n_components = n_components
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        random_state = check_random_state(self.random_state)  # 检查和获取随机状态
        X = self._validate_data(X)  # 验证输入数据 X 的格式

        self.mean_ = X.mean(axis=0)  # 计算特征均值
        X = X - self.mean_  # 中心化数据

        if self.n_components is None:
            n_components = X.shape[1]  # 如果未指定 n_components，则设置为特征数
        else:
            n_components = self.n_components  # 否则使用指定的 n_components

        return self._fit(X, n_components, random_state)  # 调用内部拟合方法进行模型拟合
    def transform(self, X):
        """
        Least Squares projection of the data onto the sparse components.

        To avoid instability issues in case the system is under-determined,
        regularization can be applied (Ridge regression) via the
        `ridge_alpha` parameter.

        Note that Sparse PCA components orthogonality is not enforced as in PCA
        hence one cannot use a simple linear projection.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data to be transformed, must have the same number of
            features as the data used to train the model.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)

        # 验证并处理输入数据 X，保持数据不变
        X = self._validate_data(X, reset=False)
        # 中心化数据，减去均值
        X = X - self.mean_

        # 使用岭回归进行最小二乘投影
        U = ridge_regression(
            self.components_.T, X.T, self.ridge_alpha, solver="cholesky"
        )

        return U

    def inverse_transform(self, X):
        """
        Transform data from the latent space to the original space.

        This inversion is an approximation due to the loss of information
        induced by the forward decomposition.

        .. versionadded:: 1.2

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_components)
            Data in the latent space.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Reconstructed data in the original space.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)
        # 检查输入数据 X 的格式并进行必要的转换
        X = check_array(X)

        # 返回从潜在空间到原始空间的逆变换
        return (X @ self.components_) + self.mean_

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        # 返回输出特征的数量，即组件的数量
        return self.components_.shape[0]

    def _more_tags(self):
        # 返回有关模型的额外标签信息，这里指定了数据类型的保持方式
        return {
            "preserves_dtype": [np.float64, np.float32],
        }
class SparsePCA(_BaseSparsePCA):
    """Sparse Principal Components Analysis (SparsePCA).

    Finds the set of sparse components that can optimally reconstruct
    the data.  The amount of sparseness is controllable by the coefficient
    of the L1 penalty, given by the parameter alpha.

    Read more in the :ref:`User Guide <SparsePCA>`.

    Parameters
    ----------
    n_components : int, default=None
        Number of sparse atoms to extract. If None, then ``n_components``
        is set to ``n_features``.

    alpha : float, default=1
        Sparsity controlling parameter. Higher values lead to sparser
        components.

    ridge_alpha : float, default=0.01
        Amount of ridge shrinkage to apply in order to improve
        conditioning when calling the transform method.

    max_iter : int, default=1000
        Maximum number of iterations to perform.

    tol : float, default=1e-8
        Tolerance for the stopping condition.

    method : {'lars', 'cd'}, default='lars'
        Method to be used for optimization.
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    U_init : ndarray of shape (n_samples, n_components), default=None
        Initial values for the loadings for warm restart scenarios. Only used
        if `U_init` and `V_init` are not None.

    V_init : ndarray of shape (n_components, n_features), default=None
        Initial values for the components for warm restart scenarios. Only used
        if `U_init` and `V_init` are not None.

    verbose : int or bool, default=False
        Controls the verbosity; the higher, the more messages. Defaults to 0.

    random_state : int, RandomState instance or None, default=None
        Used during dictionary learning. Pass an int for reproducible results
        across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Sparse components extracted from the data.

    error_ : ndarray
        Vector of errors at each iteration.

    n_components_ : int
        Estimated number of components.

        .. versionadded:: 0.23

    n_iter_ : int
        Number of iterations run.

    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
        Equal to ``X.mean(axis=0)``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24
    """

    def __init__(self, n_components=None, alpha=1.0, ridge_alpha=0.01,
                 max_iter=1000, tol=1e-8, method='lars', n_jobs=None,
                 U_init=None, V_init=None, verbose=False,
                 random_state=None):
        """
        Initialize the SparsePCA object with specified parameters.

        Parameters
        ----------
        n_components : int, default=None
            Number of sparse atoms to extract.

        alpha : float, default=1.0
            Sparsity controlling parameter.

        ridge_alpha : float, default=0.01
            Amount of ridge shrinkage applied during transformation.

        max_iter : int, default=1000
            Maximum number of iterations.

        tol : float, default=1e-8
            Tolerance for stopping condition.

        method : {'lars', 'cd'}, default='lars'
            Optimization method: 'lars' or 'cd'.

        n_jobs : int, default=None
            Number of parallel jobs to run.

        U_init : ndarray of shape (n_samples, n_components), default=None
            Initial loadings for warm restart scenarios.

        V_init : ndarray of shape (n_components, n_features), default=None
            Initial components for warm restart scenarios.

        verbose : bool or int, default=False
            Verbosity level.

        random_state : int, RandomState instance or None, default=None
            Seed for random number generator.

        Returns
        -------
        None
        """
        super().__init__(n_components=n_components, alpha=alpha,
                         ridge_alpha=ridge_alpha, max_iter=max_iter,
                         tol=tol, method=method, n_jobs=n_jobs,
                         U_init=U_init, V_init=V_init, verbose=verbose,
                         random_state=random_state)
    # 参数约束字典，继承自_BaseSparsePCA类的参数约束，并添加了额外的"U_init"和"V_init"参数约束
    _parameter_constraints: dict = {
        **_BaseSparsePCA._parameter_constraints,
        "U_init": [None, np.ndarray],
        "V_init": [None, np.ndarray],
    }

    # 初始化方法，设置了SparsePCA类的各种参数
    def __init__(
        self,
        n_components=None,  # 主成分数量，默认为None
        *,
        alpha=1,  # 主成分分解中稀疏度的惩罚系数，默认为1
        ridge_alpha=0.01,  # 岭回归中的惩罚系数，默认为0.01
        max_iter=1000,  # 最大迭代次数，默认为1000
        tol=1e-8,  # 算法收敛的容忍阈值，默认为1e-8
        method="lars",  # 使用的求解方法，默认为"lars"
        n_jobs=None,  # 并行运行的作业数，默认为None，即不并行
        U_init=None,  # U矩阵的初始值，默认为None
        V_init=None,  # V矩阵的初始值，默认为None
        verbose=False,  # 是否输出详细信息，默认为False
        random_state=None,  # 随机数种子，默认为None
    ):
        # 调用父类_BaseSparsePCA的初始化方法，传递相应参数
        super().__init__(
            n_components=n_components,
            alpha=alpha,
            ridge_alpha=ridge_alpha,
            max_iter=max_iter,
            tol=tol,
            method=method,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )
        # 初始化SparsePCA对象的U_init属性
        self.U_init = U_init
        # 初始化SparsePCA对象的V_init属性
        self.V_init = V_init
    # 定义 SparsePCA 类的特殊 `fit` 方法，用于拟合数据

    # 如果设置了初始成分 V_init，则使用其转置作为 code_init，否则为 None
    code_init = self.V_init.T if self.V_init is not None else None

    # 如果设置了初始字典 U_init，则使用其转置作为 dict_init，否则为 None
    dict_init = self.U_init.T if self.U_init is not None else None

    # 调用 dict_learning 函数进行字典学习
    # X.T 转置后的数据用于输入，n_components 表示要学习的字典成分数目
    # 其他参数包括 alpha（正则化参数）、tol（收敛容差）、max_iter（最大迭代次数）、method（优化方法）、n_jobs（并行数）、verbose（是否输出详细信息）、random_state（随机种子）、code_init（初始编码）、dict_init（初始字典）、return_n_iter（是否返回迭代次数）
    code, dictionary, E, self.n_iter_ = dict_learning(
        X.T,
        n_components,
        alpha=self.alpha,
        tol=self.tol,
        max_iter=self.max_iter,
        method=self.method,
        n_jobs=self.n_jobs,
        verbose=self.verbose,
        random_state=random_state,
        code_init=code_init,
        dict_init=dict_init,
        return_n_iter=True,
    )

    # 将特征向量的符号翻转，以确保输出结果的确定性
    code, dictionary = svd_flip(code, dictionary, u_based_decision=True)

    # 将编码转置后作为组件（components_）
    self.components_ = code.T

    # 计算组件的范数，并进行归一化处理
    components_norm = np.linalg.norm(self.components_, axis=1)[:, np.newaxis]
    components_norm[components_norm == 0] = 1  # 避免除以零的情况
    self.components_ /= components_norm

    # 设置模型的成分数目为组件的长度
    self.n_components_ = len(self.components_)

    # 将误差 E 赋值给模型的 error_ 属性
    self.error_ = E

    # 返回当前对象自身，以支持方法链式调用
    return self
class MiniBatchSparsePCA(_BaseSparsePCA):
    """Mini-batch Sparse Principal Components Analysis.

    Finds the set of sparse components that can optimally reconstruct
    the data.  The amount of sparseness is controllable by the coefficient
    of the L1 penalty, given by the parameter alpha.

    For an example comparing sparse PCA to PCA, see
    :ref:`sphx_glr_auto_examples_decomposition_plot_faces_decomposition.py`

    Read more in the :ref:`User Guide <SparsePCA>`.

    Parameters
    ----------
    n_components : int, default=None
        Number of sparse atoms to extract. If None, then ``n_components``
        is set to ``n_features``.

    alpha : int, default=1
        Sparsity controlling parameter. Higher values lead to sparser
        components.

    ridge_alpha : float, default=0.01
        Amount of ridge shrinkage to apply in order to improve
        conditioning when calling the transform method.

    max_iter : int, default=1_000
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.

        .. versionadded:: 1.2

        .. deprecated:: 1.4
           `max_iter=None` is deprecated in 1.4 and will be removed in 1.6.
           Use the default value (i.e. `100`) instead.

    callback : callable, default=None
        Callable that gets invoked every five iterations.

    batch_size : int, default=3
        The number of features to take in each mini batch.

    verbose : int or bool, default=False
        Controls the verbosity; the higher, the more messages. Defaults to 0.

    shuffle : bool, default=True
        Whether to shuffle the data before splitting it in batches.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    method : {'lars', 'cd'}, default='lars'
        Method to be used for optimization.
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.

    random_state : int, RandomState instance or None, default=None
        Used for random shuffling when ``shuffle`` is set to ``True``,
        during online dictionary learning. Pass an int for reproducible results
        across multiple function calls.
        See :term:`Glossary <random_state>`.

    tol : float, default=1e-3
        Control early stopping based on the norm of the differences in the
        dictionary between 2 steps.

        To disable early stopping based on changes in the dictionary, set
        `tol` to 0.0.

        .. versionadded:: 1.1
    """
    # Mini-batch Sparse PCA 类，继承自 _BaseSparsePCA 类

    def __init__(self, n_components=None, alpha=1, ridge_alpha=0.01,
                 max_iter=1000, callback=None, batch_size=3, verbose=False,
                 shuffle=True, n_jobs=None, method='lars', random_state=None,
                 tol=1e-3):
        # 初始化方法，设置 Mini-batch Sparse PCA 的参数

        # 调用父类 _BaseSparsePCA 的初始化方法
        super().__init__(n_components=n_components, alpha=alpha,
                         ridge_alpha=ridge_alpha, max_iter=max_iter,
                         method=method, n_jobs=n_jobs, random_state=random_state,
                         tol=tol)

        # 设置 Mini-batch Sparse PCA 特有的参数
        self.callback = callback
        self.batch_size = batch_size
        self.verbose = verbose
        self.shuffle = shuffle

        # 版本 1.4 弃用了 max_iter=None，将其设置为默认值 100
        if max_iter is None:
            warnings.warn("`max_iter=None` is deprecated and will be removed in 1.6. "
                          "Use `max_iter=100` instead.", DeprecationWarning)
            self.max_iter = 100
        else:
            self.max_iter = max_iter
    """
    _parameter_constraints: dict = {
        **_BaseSparsePCA._parameter_constraints,
        "max_iter": [Interval(Integral, 0, None, closed="left"), Hidden(None)],
        "callback": [None, callable],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "shuffle": ["boolean"],
        "max_no_improvement": [Interval(Integral, 0, None, closed="left"), None],
    }
    """

    """
    def __init__(
        self,
        n_components=None,
        *,
        alpha=1,
        ridge_alpha=0.01,
        max_iter=1_000,
        callback=None,
        batch_size=3,
        verbose=False,
        shuffle=True,
        n_jobs=None,
        method="lars",
        random_state=None,
        tol=1e-3,
        max_no_improvement=10,
    """
    ):
        super().__init__(
            n_components=n_components,
            alpha=alpha,
            ridge_alpha=ridge_alpha,
            max_iter=max_iter,
            tol=tol,
            method=method,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )
        self.callback = callback
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_no_improvement = max_no_improvement


    def _fit(self, X, n_components, random_state):
        """Specialized `fit` for MiniBatchSparsePCA."""

        transform_algorithm = "lasso_" + self.method
        # 创建 MiniBatchDictionaryLearning 的实例对象 est，用于拟合数据
        est = MiniBatchDictionaryLearning(
            n_components=n_components,
            alpha=self.alpha,
            max_iter=self.max_iter,
            dict_init=None,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            n_jobs=self.n_jobs,
            fit_algorithm=self.method,
            random_state=random_state,
            transform_algorithm=transform_algorithm,
            transform_alpha=self.alpha,
            verbose=self.verbose,
            callback=self.callback,
            tol=self.tol,
            max_no_improvement=self.max_no_improvement,
        )
        est.set_output(transform="default")
        # 使用转置后的数据 X.T 对 est 进行拟合
        est.fit(X.T)

        # 将拟合后的组件和迭代次数保存到当前对象的属性中
        self.components_, self.n_iter_ = est.transform(X.T).T, est.n_iter_

        # 计算组件的范数，并进行归一化处理
        components_norm = np.linalg.norm(self.components_, axis=1)[:, np.newaxis]
        components_norm[components_norm == 0] = 1
        self.components_ /= components_norm
        self.n_components_ = len(self.components_)

        return self
```
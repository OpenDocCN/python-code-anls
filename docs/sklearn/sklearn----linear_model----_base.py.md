# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_base.py`

```
```python`
"""
Generalized Linear Models.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numbers  # 导入 numbers 模块，提供数值类型相关的工具
import warnings  # 导入 warnings 模块，用于发出警告
from abc import ABCMeta, abstractmethod  # 导入 ABCMeta 和 abstractmethod，支持抽象基类的定义
from numbers import Integral  # 从 numbers 模块导入 Integral 类型

import numpy as np  # 导入 numpy 库，提供数值计算支持
import scipy.sparse as sp  # 导入 scipy.sparse 模块，提供稀疏矩阵支持
from scipy import linalg, optimize, sparse  # 从 scipy 导入线性代数、优化和稀疏矩阵功能
from scipy.sparse.linalg import lsqr  # 从 scipy.sparse.linalg 导入 lsqr 函数，进行稀疏线性最小二乘解
from scipy.special import expit  # 从 scipy.special 导入 expit 函数，计算逻辑回归的 sigmoid 函数

from ..base import (
    BaseEstimator,  # 导入 BaseEstimator，所有模型的基类
    ClassifierMixin,  # 导入 ClassifierMixin，分类模型的混入类
    MultiOutputMixin,  # 导入 MultiOutputMixin，多输出模型的混入类
    RegressorMixin,  # 导入 RegressorMixin，回归模型的混入类
    _fit_context,  # 导入 _fit_context，内部拟合上下文管理器
)
from ..utils import check_array, check_random_state  # 导入数据检查和随机状态检查工具函数
from ..utils._array_api import (
    _asarray_with_order,  # 导入将数组转换为指定顺序的函数
    _average,  # 导入计算平均值的函数
    get_namespace,  # 导入获取命名空间的函数
    get_namespace_and_device,  # 导入获取命名空间和设备的函数
    indexing_dtype,  # 导入索引数据类型的函数
    supported_float_dtypes,  # 导入支持的浮点数据类型的函数
)
from ..utils._seq_dataset import (
    ArrayDataset32,  # 导入 ArrayDataset32 类，32位浮点数组数据集
    ArrayDataset64,  # 导入 ArrayDataset64 类，64位浮点数组数据集
    CSRDataset32,  # 导入 CSRDataset32 类，32位浮点稀疏数据集
    CSRDataset64,  # 导入 CSRDataset64 类，64位浮动稀疏数据集
)
from ..utils.extmath import safe_sparse_dot  # 导入 safe_sparse_dot 函数，安全的稀疏矩阵点乘
from ..utils.parallel import Parallel, delayed  # 导入并行处理的 Parallel 和 delayed 函数
from ..utils.sparsefuncs import mean_variance_axis  # 导入计算稀疏矩阵的均值和方差的函数
from ..utils.validation import _check_sample_weight, check_is_fitted  # 导入样本权重检查和模型拟合验证函数

# TODO: bayesian_ridge_regression 和 bayesian_regression_ard 应合并到各自对象中。

SPARSE_INTERCEPT_DECAY = 0.01  # 设置稀疏数据的截距更新衰减因子，防止截距振荡。

def make_dataset(X, y, sample_weight, random_state=None):
    """Create ``Dataset`` abstraction for sparse and dense inputs.

    This also returns the ``intercept_decay`` which is different
    for sparse datasets.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data

    y : array-like, shape (n_samples, )
        Target values.

    sample_weight : numpy array of shape (n_samples,)
        The weight of each sample

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset random sampling. It is not
        used for dataset shuffling.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    dataset
        The ``Dataset`` abstraction
    intercept_decay
        The intercept decay
    """

    rng = check_random_state(random_state)  # 创建随机数生成器
    # seed 不应为 0 在 SequentialDataset64 中
    seed = rng.randint(1, np.iinfo(np.int32).max)  # 生成一个在指定范围内的随机整数作为种子

    if X.dtype == np.float32:
        CSRData = CSRDataset32  # 如果数据类型为 float32，使用 CSRDataset32
        ArrayData = ArrayDataset32  # 如果数据类型为 float32，使用 ArrayDataset32
    else:
        CSRData = CSRDataset64  # 否则，使用 CSRDataset64
        ArrayData = ArrayDataset64  # 否则，使用 ArrayDataset64

    if sp.issparse(X):  # 判断 X 是否为稀疏矩阵
        dataset = CSRData(X.data, X.indptr, X.indices, y, sample_weight, seed=seed)  # 创建稀疏数据集对象
        intercept_decay = SPARSE_INTERCEPT_DECAY  # 设置稀疏数据集的截距衰减因子
    else:
        X = np.ascontiguousarray(X)  # 确保 X 为连续数组
        dataset = ArrayData(X, y, sample_weight, seed=seed)  # 创建密集数据集对象
        intercept_decay = 1.0  # 设置密集数据集的截距衰减因子

    return dataset, intercept_decay  # 返回数据集对象和截距衰减因子


def _preprocess_data(
    X,
    y,
    *,
    fit_intercept,
    copy=True,
    copy_y=True,
    sample_weight=None,
    check_input=True,
):
    """Common data preprocessing for fitting linear models.

    This helper is in charge of the following steps:

    - Ensure that `sample_weight` is an array or `None`.
    - If `check_input=True`, perform standard input validation of `X`, `y`.
    - Perform copies if requested to avoid side-effects in case of inplace
      modifications of the input.

    Then, if `fit_intercept=True` this preprocessing centers both `X` and `y` as
    follows:
        - if `X` is dense, center the data and
        store the mean vector in `X_offset`.
        - if `X` is sparse, store the mean in `X_offset`
        without centering `X`. The centering is expected to be handled by the
        linear solver where appropriate.
        - in either case, always center `y` and store the mean in `y_offset`.
        - both `X_offset` and `y_offset` are always weighted by `sample_weight`
          if not set to `None`.

    If `fit_intercept=False`, no centering is performed and `X_offset`, `y_offset`
    are set to zero.

    Returns
    -------
    X_out : {ndarray, sparse matrix} of shape (n_samples, n_features)
        If copy=True a copy of the input X is triggered, otherwise operations are
        inplace.
        If input X is dense, then X_out is centered.
    y_out : {ndarray, sparse matrix} of shape (n_samples,) or (n_samples, n_targets)
        Centered version of y. Possibly performed inplace on input y depending
        on the copy_y parameter.
    X_offset : ndarray of shape (n_features,)
        The mean per column of input X.
    y_offset : float or ndarray of shape (n_features,)
    X_scale : ndarray of shape (n_features,)
        Always an array of ones. TODO: refactor the code base to make it
        possible to remove this unused variable.
    """
    # 获取 X, y, sample_weight 的命名空间和设备信息
    xp, _, device_ = get_namespace_and_device(X, y, sample_weight)
    # 获取样本数和特征数
    n_samples, n_features = X.shape
    # 检查 X 是否为稀疏矩阵
    X_is_sparse = sp.issparse(X)

    # 如果 sample_weight 是数字，则置为 None
    if isinstance(sample_weight, numbers.Number):
        sample_weight = None
    # 如果 sample_weight 不为 None，则将其转换为 xp 数组
    if sample_weight is not None:
        sample_weight = xp.asarray(sample_weight)

    # 如果 check_input 为 True，则进行输入验证
    if check_input:
        # 验证并复制 X，接受稀疏矩阵类型为 csr 或 csc，数据类型为支持的浮点类型
        X = check_array(
            X, copy=copy, accept_sparse=["csr", "csc"], dtype=supported_float_dtypes(xp)
        )
        # 验证并复制 y，数据类型与 X 保持一致，确保为二维数组
        y = check_array(y, dtype=X.dtype, copy=copy_y, ensure_2d=False)
    else:
        # 将 y 转换为与 X 相同的数据类型
        y = xp.astype(y, X.dtype, copy=copy_y)
        # 如果需要复制，则复制 X
        if copy:
            if X_is_sparse:
                X = X.copy()
            else:
                X = _asarray_with_order(X, order="K", copy=True, xp=xp)

    # 获取 X 的数据类型
    dtype_ = X.dtype

    # 如果 fit_intercept 为 True，则对数据进行中心化处理
    if fit_intercept:
        # 如果 X 是稀疏矩阵，计算均值和方差
        if X_is_sparse:
            X_offset, X_var = mean_variance_axis(X, axis=0, weights=sample_weight)
        else:
            # 计算 X 的均值并中心化 X
            X_offset = _average(X, axis=0, weights=sample_weight, xp=xp)
            X_offset = xp.astype(X_offset, X.dtype, copy=False)
            X -= X_offset

        # 计算 y 的均值并中心化 y
        y_offset = _average(y, axis=0, weights=sample_weight, xp=xp)
        y -= y_offset
    else:
        # 如果不需要对数据进行标准化，则初始化 X_offset 为全零数组
        X_offset = xp.zeros(n_features, dtype=X.dtype, device=device_)
        # 如果 y 是一维数组，则初始化 y_offset 为标量 0.0
        if y.ndim == 1:
            y_offset = xp.asarray(0.0, dtype=dtype_, device=device_)
        else:
            # 如果 y 是二维数组，则初始化 y_offset 为全零数组
            y_offset = xp.zeros(y.shape[1], dtype=dtype_, device=device_)

    # XXX: X_scale is no longer needed. It is an historic artifact from the
    # time where linear model exposed the normalize parameter.
    # 初始化 X_scale 为全一数组，但现在不再使用，这是历史遗留问题，用于线性模型中的 normalize 参数
    X_scale = xp.ones(n_features, dtype=X.dtype, device=device_)
    # 返回处理后的数据 X, y, X_offset, y_offset, X_scale
    return X, y, X_offset, y_offset, X_scale
# TODO: _rescale_data should be factored into _preprocess_data.
# Currently, the fact that sag implements its own way to deal with
# sample_weight makes the refactoring tricky.

# 定义一个函数 _rescale_data，用于按照样本权重的平方根对数据进行重新缩放
def _rescale_data(X, y, sample_weight, inplace=False):
    """Rescale data sample-wise by square root of sample_weight.

    For many linear models, this enables easy support for sample_weight because

        (y - X w)' S (y - X w)

    with S = diag(sample_weight) becomes

        ||y_rescaled - X_rescaled w||_2^2

    when setting

        y_rescaled = sqrt(S) y
        X_rescaled = sqrt(S) X

    Returns
    -------
    X_rescaled : {array-like, sparse matrix}
        Rescaled feature matrix.

    y_rescaled : {array-like, sparse matrix}
        Rescaled target values.
    """
    # Assume that _validate_data and _check_sample_weight have been called by
    # the caller.

    # 获取命名空间，根据输入的 X, y, sample_weight 确定使用的库或模块
    xp, _ = get_namespace(X, y, sample_weight)
    # 获取样本数
    n_samples = X.shape[0]
    # 计算样本权重的平方根
    sample_weight_sqrt = xp.sqrt(sample_weight)

    # 如果 X 或者 y 是稀疏矩阵
    if sp.issparse(X) or sp.issparse(y):
        # 创建对角线稀疏矩阵，以 sample_weight_sqrt 为对角线元素
        sw_matrix = sparse.dia_matrix(
            (sample_weight_sqrt, 0), shape=(n_samples, n_samples)
        )

    # 如果 X 是稀疏矩阵
    if sp.issparse(X):
        # 使用 safe_sparse_dot 对 X 进行稀疏矩阵乘法
        X = safe_sparse_dot(sw_matrix, X)
    else:
        # 如果 inplace=True，则直接对 X 进行乘法操作；否则创建新的 X_rescaled
        if inplace:
            X *= sample_weight_sqrt[:, None]
        else:
            X = X * sample_weight_sqrt[:, None]

    # 如果 y 是稀疏矩阵
    if sp.issparse(y):
        # 使用 safe_sparse_dot 对 y 进行稀疏矩阵乘法
        y = safe_sparse_dot(sw_matrix, y)
    else:
        # 如果 inplace=True，则直接对 y 进行乘法操作；否则创建新的 y_rescaled
        if inplace:
            if y.ndim == 1:
                y *= sample_weight_sqrt
            else:
                y *= sample_weight_sqrt[:, None]
        else:
            if y.ndim == 1:
                y = y * sample_weight_sqrt
            else:
                y = y * sample_weight_sqrt[:, None]

    # 返回处理后的 X, y 和 sample_weight_sqrt
    return X, y, sample_weight_sqrt


class LinearModel(BaseEstimator, metaclass=ABCMeta):
    """Base class for Linear Models"""

    @abstractmethod
    def fit(self, X, y):
        """Fit model."""
    
    # 内部方法，计算决策函数的值
    def _decision_function(self, X):
        # 检查模型是否已经拟合
        check_is_fitted(self)
        
        # 验证输入数据 X，接受稀疏矩阵格式，并保持不重置
        X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
        # 获取模型的系数 coef_
        coef_ = self.coef_
        # 如果 coef_ 是一维数组，则计算预测值并返回
        if coef_.ndim == 1:
            return X @ coef_ + self.intercept_
        else:
            # 否则，进行转置并计算预测值并返回
            return X @ coef_.T + self.intercept_

    # 预测方法，使用线性模型进行预测
    def predict(self, X):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        return self._decision_function(X)
    def _set_intercept(self, X_offset, y_offset, X_scale):
        """Set the intercept_"""

        xp, _ = get_namespace(X_offset, y_offset, X_scale)

        # 如果需要拟合截距
        if self.fit_intercept:
            # 我们始终希望 coef_.dtype=X.dtype。例如，如果 warm_start=True，X.dtype 可能与 coef_.dtype 不同。
            # 将 xp 转换为与 self.coef_ 相同的 dtype，并进行 inplace 操作
            coef_ = xp.astype(self.coef_, X_scale.dtype, copy=False)
            # 将 coef_ 除以 X_scale，以确保 coef_ 的尺度与 X 相匹配
            coef_ = self.coef_ = xp.divide(coef_, X_scale)

            # 如果 coef_ 是一维数组
            if coef_.ndim == 1:
                # 计算截距
                intercept_ = y_offset - X_offset @ coef_
            else:
                # 如果 coef_ 是二维数组，则进行转置后计算截距
                intercept_ = y_offset - X_offset @ coef_.T

            # 设置拟合的截距值
            self.intercept_ = intercept_

        else:
            # 如果不需要拟合截距，则将截距设置为 0.0
            self.intercept_ = 0.0

    def _more_tags(self):
        return {"requires_y": True}
# XXX Should this derive from LinearModel? It should be a mixin, not an ABC.
# Maybe the n_features checking can be moved to LinearModel.
class LinearClassifierMixin(ClassifierMixin):
    """Mixin for linear classifiers.

    Handles prediction for sparse and dense X.
    """

    def decision_function(self, X):
        """
        Predict confidence scores for samples.

        The confidence score for a sample is proportional to the signed
        distance of that sample to the hyperplane.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the confidence scores.

        Returns
        -------
        scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Confidence scores per `(n_samples, n_classes)` combination. In the
            binary case, confidence score for `self.classes_[1]` where >0 means
            this class would be predicted.
        """
        # 确保模型已经拟合
        check_is_fitted(self)
        # 获取 X 的命名空间信息
        xp, _ = get_namespace(X)

        # 验证数据 X，接受稀疏矩阵 CSR 格式，不重置数据
        X = self._validate_data(X, accept_sparse="csr", reset=False)
        # 计算预测得分，使用稀疏矩阵的点乘和加上截距
        scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
        return xp.reshape(scores, (-1,)) if scores.shape[1] == 1 else scores

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the predictions.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        # 获取 X 的命名空间信息
        xp, _ = get_namespace(X)
        # 调用 decision_function 方法获取预测得分
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            # 对于二元分类情况，通过判断得分是否大于零来确定类别
            indices = xp.astype(scores > 0, indexing_dtype(xp))
        else:
            # 对于多类别分类，选择得分最高的类别作为预测结果
            indices = xp.argmax(scores, axis=1)

        return xp.take(self.classes_, indices, axis=0)

    def _predict_proba_lr(self, X):
        """Probability estimation for OvR logistic regression.

        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        # 计算决策函数的输出
        prob = self.decision_function(X)
        # 对输出进行 logistic 函数转换，计算正类的概率
        expit(prob, out=prob)
        if prob.ndim == 1:
            # 对于二元分类，返回每个样本属于各类别的概率
            return np.vstack([1 - prob, prob]).T
        else:
            # 对于多类别分类，进行 OvR 归一化处理，使得各类别概率和为1
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob


class SparseCoefMixin:
    """Mixin for converting coef_ to and from CSR format.

    L1-regularizing estimators should inherit this.
    """
    def densify(self):
        """
        Convert coefficient matrix to dense array format.

        Converts the ``coef_`` member (back) to a numpy.ndarray. This is the
        default format of ``coef_`` and is required for fitting, so calling
        this method is only required on models that have previously been
        sparsified; otherwise, it is a no-op.

        Returns
        -------
        self
            Fitted estimator.
        """
        # 检查估计器是否已经拟合，如果未拟合则引发异常
        msg = "Estimator, %(name)s, must be fitted before densifying."
        check_is_fitted(self, msg=msg)
        
        # 如果 coef_ 是稀疏矩阵，则将其转换为密集数组
        if sp.issparse(self.coef_):
            self.coef_ = self.coef_.toarray()
        
        # 返回拟合后的估计器自身
        return self

    def sparsify(self):
        """
        Convert coefficient matrix to sparse format.

        Converts the ``coef_`` member to a scipy.sparse matrix, which for
        L1-regularized models can be much more memory- and storage-efficient
        than the usual numpy.ndarray representation.

        The ``intercept_`` member is not converted.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        For non-sparse models, i.e. when there are not many zeros in ``coef_``,
        this may actually *increase* memory usage, so use this method with
        care. A rule of thumb is that the number of zero elements, which can
        be computed with ``(coef_ == 0).sum()``, must be more than 50% for this
        to provide significant benefits.

        After calling this method, further fitting with the partial_fit
        method (if any) will not work until you call densify.
        """
        # 检查估计器是否已经拟合，如果未拟合则引发异常
        msg = "Estimator, %(name)s, must be fitted before sparsifying."
        check_is_fitted(self, msg=msg)
        
        # 将 coef_ 转换为稀疏矩阵格式
        self.coef_ = sp.csr_matrix(self.coef_)
        
        # 返回拟合后的估计器自身
        return self
# 定义线性回归类，继承自MultiOutputMixin、RegressorMixin和LinearModel类
class LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):
    """
    Ordinary least squares Linear Regression.

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This will only provide
        speedup in case of sufficiently large problems, that is if firstly
        `n_targets > 1` and secondly `X` is sparse or if `positive` is set
        to `True`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive. This
        option is only supported for dense arrays.

        .. versionadded:: 0.24

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    rank_ : int
        Rank of matrix `X`. Only available when `X` is dense.

    singular_ : array of shape (min(X, y),)
        Singular values of `X`. Only available when `X` is dense.

    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model. Set to 0.0 if
        `fit_intercept = False`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    Ridge : Ridge regression addresses some of the
        problems of Ordinary Least Squares by imposing a penalty on the
        size of the coefficients with l2 regularization.
    Lasso : The Lasso is a linear model that estimates
        sparse coefficients with l1 regularization.
    ElasticNet : Elastic-Net is a linear regression
        model trained with both l1 and l2 -norm regularization of the
        coefficients.

    Notes
    -----
    From the implementation point of view, this is just plain Ordinary
    Least Squares (scipy.linalg.lstsq) or Non Negative Least Squares.
    """
    # 定义了一个参数约束字典，用于指定模型参数的类型约束
    _parameter_constraints: dict = {
        "fit_intercept": ["boolean"],  # fit_intercept 参数应为布尔类型
        "copy_X": ["boolean"],          # copy_X 参数应为布尔类型
        "n_jobs": [None, Integral],     # n_jobs 参数可以为 None 或整数类型
        "positive": ["boolean"],        # positive 参数应为布尔类型
    }
    
    # 定义了一个线性回归类
    class LinearRegression:
    
        # 初始化方法，设定了几个模型参数的默认值
        def __init__(
            self,
            *,
            fit_intercept=True,  # 是否拟合截距，默认为 True
            copy_X=True,         # 是否复制输入数据，默认为 True
            n_jobs=None,         # 并行作业数，默认为 None（即不并行）
            positive=False,      # 是否限制预测结果为非负数，默认为 False
        ):
            self.fit_intercept = fit_intercept  # 将参数值赋给对象的属性
            self.copy_X = copy_X
            self.n_jobs = n_jobs
            self.positive = positive
    
        # 使用装饰器进行函数修饰
        @_fit_context(prefer_skip_nested_validation=True)
# 在线性模型拟合开始前执行的辅助函数，用于带有 L1 或 L0 惩罚的线性模型
def _pre_fit(
    X,
    y,
    Xy,
    precompute,
    fit_intercept,
    copy,
    check_input=True,
    sample_weight=None,
):
    """Function used at beginning of fit in linear models with L1 or L0 penalty.

    This function applies _preprocess_data and additionally computes the gram matrix
    `precompute` as needed as well as `Xy`.
    """
    # 获取样本数和特征数
    n_samples, n_features = X.shape

    if sparse.issparse(X):
        # 如果 X 是稀疏矩阵，则不需要复制，因为在稀疏矩阵情况下 X 不会就地修改
        precompute = False
        # 对稀疏矩阵 X 进行预处理，并获取处理后的 X, y, X_offset, y_offset, X_scale
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=fit_intercept,
            copy=False,
            check_input=check_input,
            sample_weight=sample_weight,
        )
    else:
        # 如果需要，在 fit 中进行了拷贝
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=fit_intercept,
            copy=copy,
            check_input=check_input,
            sample_weight=sample_weight,
        )
        # 只在稠密情况下重新缩放。稀疏 cd solver 直接处理 sample_weight。
        if sample_weight is not None:
            # 这将触发必要的拷贝。
            X, y, _ = _rescale_data(X, y, sample_weight=sample_weight)

    if hasattr(precompute, "__array__"):
        if fit_intercept and not np.allclose(X_offset, np.zeros(n_features)):
            warnings.warn(
                (
                    "Gram 矩阵已提供，但 X 被居中以适应截距：重新计算 Gram 矩阵。"
                ),
                UserWarning,
            )
            # TODO: 而不是警告和重新计算，我们可以在后期（在 `copy=True` 时）仅对用户提供的 Gram 矩阵进行居中。
            # 重新计算 Gram 矩阵
            precompute = "auto"
            Xy = None
        elif check_input:
            # 如果我们将使用用户预先计算的 Gram 矩阵，我们会进行快速检查，确保其不是完全错误的。
            _check_precomputed_gram_matrix(X, precompute, X_offset, X_scale)

    # 如果 n_samples > n_features，则预先计算
    if isinstance(precompute, str) and precompute == "auto":
        precompute = n_samples > n_features

    if precompute is True:
        # 确保 'precompute' 数组是连续的。
        precompute = np.empty(shape=(n_features, n_features), dtype=X.dtype, order="C")
        np.dot(X.T, X, out=precompute)

    if not hasattr(precompute, "__array__"):
        Xy = None  # 如果 precompute 不是 Gram，不能使用 Xy

    if hasattr(precompute, "__array__") and Xy is None:
        common_dtype = np.result_type(X.dtype, y.dtype)
        if y.ndim == 1:
            # Xy 是 1 维的，确保它是连续的。
            Xy = np.empty(shape=n_features, dtype=common_dtype, order="C")
            np.dot(X.T, y, out=Xy)
        else:
            # 确保无论 X 或 y 是否连续，Xy 总是 F 连续的：目标是快速提取特定目标的数据。
            n_targets = y.shape[1]
            Xy = np.empty(shape=(n_features, n_targets), dtype=common_dtype, order="F")
            np.dot(y.T, X, out=Xy.T)

    return X, y, X_offset, y_offset, X_scale, precompute, Xy
```
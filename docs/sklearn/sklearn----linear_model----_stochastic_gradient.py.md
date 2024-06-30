# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_stochastic_gradient.py`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
"""Classification, regression and One-Class SVM using Stochastic Gradient
Descent (SGD).
"""

# 导入警告模块
import warnings
# 导入抽象基类模块和数值类型模块
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real

# 导入NumPy库
import numpy as np

# 导入scikit-learn的基础模块和相关函数
from ..base import (
    BaseEstimator,
    OutlierMixin,
    RegressorMixin,
    _fit_context,
    clone,
    is_classifier,
)
# 导入自定义的异常模块
from ..exceptions import ConvergenceWarning
# 导入模型选择模块中的交叉验证划分类
from ..model_selection import ShuffleSplit, StratifiedShuffleSplit
# 导入工具模块中的随机数状态检查和类别权重计算函数
from ..utils import check_random_state, compute_class_weight
# 导入参数验证模块中的特殊参数类型
from ..utils._param_validation import Hidden, Interval, StrOptions
# 导入数学扩展模块中的稀疏矩阵乘法函数
from ..utils.extmath import safe_sparse_dot
# 导入元估计器模块中的条件导入函数
from ..utils.metaestimators import available_if
# 导入多类别分类模块中的部分拟合验证函数
from ..utils.multiclass import _check_partial_fit_first_call
# 导入并行计算模块中的并行执行函数和延迟函数
from ..utils.parallel import Parallel, delayed
# 导入验证模块中的样本权重检查和已拟合检查函数
from ..utils.validation import _check_sample_weight, check_is_fitted
# 导入SGD基础分类器混合模块中的分类器和稀疏系数混合函数
from ._base import LinearClassifierMixin, SparseCoefMixin, make_dataset
# 导入SGD快速模块中的不同损失函数和优化器
from ._sgd_fast import (
    EpsilonInsensitive,
    Hinge,
    Huber,
    Log,
    ModifiedHuber,
    SquaredEpsilonInsensitive,
    SquaredHinge,
    SquaredLoss,
    _plain_sgd32,
    _plain_sgd64,
)

# 学习率类型字典
LEARNING_RATE_TYPES = {
    "constant": 1,
    "optimal": 2,
    "invscaling": 3,
    "adaptive": 4,
    "pa1": 5,
    "pa2": 6,
}

# 惩罚类型字典
PENALTY_TYPES = {"none": 0, "l2": 2, "l1": 1, "elasticnet": 3}

# 默认的 epsilon 参数值
DEFAULT_EPSILON = 0.1
# Default value of ``epsilon`` parameter.

# 最大整数值常量
MAX_INT = np.iinfo(np.int32).max


class _ValidationScoreCallback:
    """Callback for early stopping based on validation score"""

    def __init__(self, estimator, X_val, y_val, sample_weight_val, classes=None):
        self.estimator = clone(estimator)
        self.estimator.t_ = 1  # to pass check_is_fitted
        if classes is not None:
            self.estimator.classes_ = classes
        self.X_val = X_val
        self.y_val = y_val
        self.sample_weight_val = sample_weight_val

    def __call__(self, coef, intercept):
        est = self.estimator
        est.coef_ = coef.reshape(1, -1)
        est.intercept_ = np.atleast_1d(intercept)
        return est.score(self.X_val, self.y_val, self.sample_weight_val)


class BaseSGD(SparseCoefMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for SGD classification and regression."""

    # SGD分类和回归的基类
    _parameter_constraints: dict = {
        "fit_intercept": ["boolean"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left"), None],
        "shuffle": ["boolean"],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
        "warm_start": ["boolean"],
        "average": [Interval(Integral, 0, None, closed="left"), "boolean"],
    }
    def __init__(
        self,
        loss,
        *,
        penalty="l2",
        alpha=0.0001,
        C=1.0,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        epsilon=0.1,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        warm_start=False,
        average=False,
    ):
        # 初始化函数，设置模型的各种参数
        self.loss = loss
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.alpha = alpha
        self.C = C
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.warm_start = warm_start
        self.average = average
        self.max_iter = max_iter
        self.tol = tol

    @abstractmethod
    def fit(self, X, y):
        """Fit model."""
        # 抽象方法，用于拟合模型，具体实现由子类定义

    def _more_validate_params(self, for_partial_fit=False):
        """Validate input params."""
        # 验证输入参数的有效性
        if self.early_stopping and for_partial_fit:
            raise ValueError("early_stopping should be False with partial_fit")
        if (
            self.learning_rate in ("constant", "invscaling", "adaptive")
            and self.eta0 <= 0.0
        ):
            raise ValueError("eta0 must be > 0")
        if self.learning_rate == "optimal" and self.alpha == 0:
            raise ValueError(
                "alpha must be > 0 since "
                "learning_rate is 'optimal'. alpha is used "
                "to compute the optimal learning rate."
            )

        # raises ValueError if not registered
        # 如果 penalty 类型未注册，则引发 ValueError
        self._get_penalty_type(self.penalty)
        # 获取 learning_rate 的类型，如果未注册则引发 ValueError
        self._get_learning_rate_type(self.learning_rate)

    def _get_loss_function(self, loss):
        """Get concrete ``LossFunction`` object for str ``loss``."""
        # 获取具体的损失函数对象，根据给定的损失函数字符串
        loss_ = self.loss_functions[loss]
        loss_class, args = loss_[0], loss_[1:]
        if loss in ("huber", "epsilon_insensitive", "squared_epsilon_insensitive"):
            args = (self.epsilon,)
        return loss_class(*args)

    def _get_learning_rate_type(self, learning_rate):
        # 获取学习率类型对应的具体值
        return LEARNING_RATE_TYPES[learning_rate]

    def _get_penalty_type(self, penalty):
        # 将 penalty 转换为小写字符串，获取对应的惩罚类型
        penalty = str(penalty).lower()
        return PENALTY_TYPES[penalty]

    def _allocate_parameter_mem(
        self,
        n_classes,
        n_features,
        input_dtype,
        coef_init=None,
        intercept_init=None,
        one_class=0,
        """Allocate mem for parameters; initialize if provided."""
        # 如果类别数大于2，为多类别情况分配 coef_
        if n_classes > 2:
            # 如果提供了 coef_init，则使用提供的值初始化 coef_
            if coef_init is not None:
                coef_init = np.asarray(coef_init, dtype=input_dtype, order="C")
                # 检查提供的 coef_ 是否符合数据集的要求
                if coef_init.shape != (n_classes, n_features):
                    raise ValueError("Provided ``coef_`` does not match dataset. ")
                self.coef_ = coef_init
            else:
                # 否则初始化为全零数组
                self.coef_ = np.zeros(
                    (n_classes, n_features), dtype=input_dtype, order="C"
                )

            # 如果提供了 intercept_init，则使用提供的值初始化 intercept_
            if intercept_init is not None:
                intercept_init = np.asarray(
                    intercept_init, order="C", dtype=input_dtype
                )
                # 检查提供的 intercept_init 是否符合数据集的要求
                if intercept_init.shape != (n_classes,):
                    raise ValueError("Provided intercept_init does not match dataset.")
                self.intercept_ = intercept_init
            else:
                # 否则初始化为全零数组
                self.intercept_ = np.zeros(n_classes, dtype=input_dtype, order="C")
        else:
            # 如果类别数为2或以下，为 coef_ 分配空间
            if coef_init is not None:
                coef_init = np.asarray(coef_init, dtype=input_dtype, order="C")
                coef_init = coef_init.ravel()
                # 检查提供的 coef_init 是否符合数据集的要求
                if coef_init.shape != (n_features,):
                    raise ValueError("Provided coef_init does not match dataset.")
                self.coef_ = coef_init
            else:
                # 否则初始化为全零数组
                self.coef_ = np.zeros(n_features, dtype=input_dtype, order="C")

            # 如果提供了 intercept_init，则使用提供的值初始化 intercept_
            if intercept_init is not None:
                intercept_init = np.asarray(intercept_init, dtype=input_dtype)
                # 检查提供的 intercept_init 是否符合数据集的要求
                if intercept_init.shape != (1,) and intercept_init.shape != ():
                    raise ValueError("Provided intercept_init does not match dataset.")
                # 如果是单类别分类，则设置 offset_
                if one_class:
                    self.offset_ = intercept_init.reshape(
                        1,
                    )
                else:
                    self.intercept_ = intercept_init.reshape(
                        1,
                    )
            else:
                # 否则初始化为全零数组
                if one_class:
                    self.offset_ = np.zeros(1, dtype=input_dtype, order="C")
                else:
                    self.intercept_ = np.zeros(1, dtype=input_dtype, order="C")

        # 如果需要初始化平均参数
        if self.average > 0:
            # 将标准 coef_ 设置为当前 coef_
            self._standard_coef = self.coef_
            # 初始化平均 coef_ 为全零数组
            self._average_coef = np.zeros(
                self.coef_.shape, dtype=input_dtype, order="C"
            )
            # 如果是单类别分类，则设置标准 intercept_
            if one_class:
                self._standard_intercept = 1 - self.offset_
            else:
                self._standard_intercept = self.intercept_

            # 初始化平均 intercept_ 为全零数组
            self._average_intercept = np.zeros(
                self._standard_intercept.shape, dtype=input_dtype, order="C"
            )
    def _make_validation_split(self, y, sample_mask):
        """Split the dataset between training set and validation set.

        Parameters
        ----------
        y : ndarray of shape (n_samples, )
            Target values.

        sample_mask : ndarray of shape (n_samples, )
            A boolean array indicating whether each sample should be included
            for validation set.

        Returns
        -------
        validation_mask : ndarray of shape (n_samples, )
            Equal to True on the validation set, False on the training set.
        """
        # 获取样本数
        n_samples = y.shape[0]
        # 创建一个全为 False 的验证集掩码
        validation_mask = np.zeros(n_samples, dtype=np.bool_)
        # 如果不使用早停策略，直接返回全 False 的验证集掩码
        if not self.early_stopping:
            return validation_mask

        # 根据分类器类型选择分割器类型
        if is_classifier(self):
            splitter_type = StratifiedShuffleSplit
        else:
            splitter_type = ShuffleSplit
        # 创建分割器对象
        cv = splitter_type(
            test_size=self.validation_fraction, random_state=self.random_state
        )
        # 利用分割器对象生成训练集和验证集的索引
        idx_train, idx_val = next(cv.split(np.zeros(shape=(y.shape[0], 1)), y))

        # 检查验证集的样本权重是否全部为零，若是则抛出异常
        if not np.any(sample_mask[idx_val]):
            raise ValueError(
                "The sample weights for validation set are all zero, consider using a"
                " different random state."
            )

        # 检查训练集或验证集是否为空集，若是则抛出异常
        if idx_train.shape[0] == 0 or idx_val.shape[0] == 0:
            raise ValueError(
                "Splitting %d samples into a train set and a validation set "
                "with validation_fraction=%r led to an empty set (%d and %d "
                "samples). Please either change validation_fraction, increase "
                "number of samples, or disable early_stopping."
                % (
                    n_samples,
                    self.validation_fraction,
                    idx_train.shape[0],
                    idx_val.shape[0],
                )
            )

        # 将验证集的掩码中对应索引位置设为 True
        validation_mask[idx_val] = True
        return validation_mask

    def _make_validation_score_cb(
        self, validation_mask, X, y, sample_weight, classes=None
    ):
        # 若不使用早停策略，则返回空值
        if not self.early_stopping:
            return None

        # 返回验证分数回调对象，用于评估验证集的性能
        return _ValidationScoreCallback(
            self,
            X[validation_mask],
            y[validation_mask],
            sample_weight[validation_mask],
            classes=classes,
        )
# 初始化 fit_binary 函数的准备工作，获取与正类相关的标签数组 y_i，系数 coef，截距 intercept，以及平均系数 average_coef 和平均截距 average_intercept
def _prepare_fit_binary(est, y, i, input_dtype):
    y_i = np.ones(y.shape, dtype=input_dtype, order="C")  # 创建一个与 y 形状相同的全为1的数组 y_i，数据类型为 input_dtype
    y_i[y != est.classes_[i]] = -1.0  # 将 y_i 中与正类索引 i 对应的位置设为 -1
    average_intercept = 0  # 初始化平均截距为 0
    average_coef = None  # 初始化平均系数为 None

    if len(est.classes_) == 2:  # 如果类别数量为 2
        if not est.average:  # 如果不需要平均
            coef = est.coef_.ravel()  # 将 coef 设置为扁平化后的 est.coef_
            intercept = est.intercept_[0]  # 将截距 intercept 设置为 est.intercept_ 的第一个元素
        else:  # 如果需要平均
            coef = est._standard_coef.ravel()  # 将 coef 设置为扁平化后的 est._standard_coef
            intercept = est._standard_intercept[0]  # 将截距 intercept 设置为 est._standard_intercept 的第一个元素
            average_coef = est._average_coef.ravel()  # 将平均系数 average_coef 设置为扁平化后的 est._average_coef
            average_intercept = est._average_intercept[0]  # 将平均截距 average_intercept 设置为 est._average_intercept 的第一个元素
    else:  # 如果类别数量大于 2
        if not est.average:  # 如果不需要平均
            coef = est.coef_[i]  # 将 coef 设置为 est.coef_ 的第 i 行
            intercept = est.intercept_[i]  # 将截距 intercept 设置为 est.intercept_ 的第 i 个元素
        else:  # 如果需要平均
            coef = est._standard_coef[i]  # 将 coef 设置为 est._standard_coef 的第 i 行
            intercept = est._standard_intercept[i]  # 将截距 intercept 设置为 est._standard_intercept 的第 i 个元素
            average_coef = est._average_coef[i]  # 将平均系数 average_coef 设置为 est._average_coef 的第 i 行
            average_intercept = est._average_intercept[i]  # 将平均截距 average_intercept 设置为 est._average_intercept 的第 i 个元素

    return y_i, coef, intercept, average_coef, average_intercept  # 返回所需的数组和值


# 拟合单个二分类器
def fit_binary(
    est,
    i,
    X,
    y,
    alpha,
    C,
    learning_rate,
    max_iter,
    pos_weight,
    neg_weight,
    sample_weight,
    validation_mask=None,
    random_state=None,
):
    """Fit a single binary classifier.

    The i'th class is considered the "positive" class.

    Parameters
    ----------
    est : Estimator object
        The estimator to fit

    i : int
        Index of the positive class

    X : numpy array or sparse matrix of shape [n_samples,n_features]
        Training data

    y : numpy array of shape [n_samples, ]
        Target values

    alpha : float
        The regularization parameter

    C : float
        Maximum step size for passive aggressive

    learning_rate : str
        The learning rate. Accepted values are 'constant', 'optimal',
        'invscaling', 'pa1' and 'pa2'.

    max_iter : int
        The maximum number of iterations (epochs)

    pos_weight : float
        The weight of the positive class

    neg_weight : float
        The weight of the negative class

    sample_weight : numpy array of shape [n_samples, ]
        The weight of each sample

    validation_mask : numpy array of shape [n_samples, ], default=None
        Precomputed validation mask in case _fit_binary is called in the
        context of a one-vs-rest reduction.

    random_state : int, RandomState instance, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    # 调用 _prepare_fit_binary 函数，获取相关的 y_i, coef, intercept, average_coef, average_intercept
    y_i, coef, intercept, average_coef, average_intercept = _prepare_fit_binary(
        est, y, i, input_dtype=X.dtype
    )
    assert y_i.shape[0] == y.shape[0] == sample_weight.shape[0]  # 断言确保 y_i, y, sample_weight 的长度相同
    # 使用 check_random_state 函数确保随机状态是有效的随机数生成器对象
    random_state = check_random_state(random_state)
    
    # 使用 make_dataset 函数生成数据集和截距衰减参数
    dataset, intercept_decay = make_dataset(
        X, y_i, sample_weight, random_state=random_state
    )
    
    # 获取估计器对象的惩罚类型
    penalty_type = est._get_penalty_type(est.penalty)
    
    # 获取估计器对象的学习率类型
    learning_rate_type = est._get_learning_rate_type(learning_rate)
    
    # 如果验证掩码为空，则使用 est._make_validation_split 函数创建验证掩码
    if validation_mask is None:
        validation_mask = est._make_validation_split(y_i, sample_mask=sample_weight > 0)
    
    # 创建类数组，包含类标签 [-1, 1]
    classes = np.array([-1, 1], dtype=y_i.dtype)
    
    # 创建验证分数回调函数 validation_score_cb
    validation_score_cb = est._make_validation_score_cb(
        validation_mask, X, y_i, sample_weight, classes=classes
    )
    
    # 根据 random_state 生成一个种子，用于设置随机数生成器的种子
    seed = random_state.randint(MAX_INT)
    
    # 设置容差 tol 为 est.tol 或者负无穷
    tol = est.tol if est.tol is not None else -np.inf
    
    # 获取用于普通随机梯度下降的函数 _plain_sgd
    _plain_sgd = _get_plain_sgd_function(input_dtype=coef.dtype)
    
    # 调用 _plain_sgd 函数执行普通随机梯度下降，并更新 coef, intercept, average_coef,
    # average_intercept, n_iter_
    coef, intercept, average_coef, average_intercept, n_iter_ = _plain_sgd(
        coef,
        intercept,
        average_coef,
        average_intercept,
        est._loss_function_,
        penalty_type,
        alpha,
        C,
        est.l1_ratio,
        dataset,
        validation_mask,
        est.early_stopping,
        validation_score_cb,
        int(est.n_iter_no_change),
        max_iter,
        tol,
        int(est.fit_intercept),
        int(est.verbose),
        int(est.shuffle),
        seed,
        pos_weight,
        neg_weight,
        learning_rate_type,
        est.eta0,
        est.power_t,
        0,
        est.t_,
        intercept_decay,
        est.average,
    )
    
    # 如果估计器使用了平均化，则更新平均截距
    if est.average:
        if len(est.classes_) == 2:
            est._average_intercept[0] = average_intercept
        else:
            est._average_intercept[i] = average_intercept
    
    # 返回更新后的 coef, intercept 和 n_iter_
    return coef, intercept, n_iter_
# 返回一个与输入数据类型相关的普通随机梯度下降函数，可能是32位或64位
def _get_plain_sgd_function(input_dtype):
    return _plain_sgd32 if input_dtype == np.float32 else _plain_sgd64

# 定义一个基本的随机梯度下降分类器类，继承自LinearClassifierMixin和BaseSGD，使用ABCMeta作为元类
class BaseSGDClassifier(LinearClassifierMixin, BaseSGD, metaclass=ABCMeta):
    # 损失函数及其对应的参数值
    loss_functions = {
        "hinge": (Hinge, 1.0),
        "squared_hinge": (SquaredHinge, 1.0),
        "perceptron": (Hinge, 0.0),
        "log_loss": (Log,),
        "modified_huber": (ModifiedHuber,),
        "squared_error": (SquaredLoss,),
        "huber": (Huber, DEFAULT_EPSILON),
        "epsilon_insensitive": (EpsilonInsensitive, DEFAULT_EPSILON),
        "squared_epsilon_insensitive": (SquaredEpsilonInsensitive, DEFAULT_EPSILON),
    }

    # 参数约束字典，包含了基类BaseSGD定义的参数约束以及新的参数约束
    _parameter_constraints: dict = {
        **BaseSGD._parameter_constraints,
        "loss": [StrOptions(set(loss_functions))],
        "early_stopping": ["boolean"],
        "validation_fraction": [Interval(Real, 0, 1, closed="neither")],
        "n_iter_no_change": [Interval(Integral, 1, None, closed="left")],
        "n_jobs": [Integral, None],
        "class_weight": [StrOptions({"balanced"}), dict, None],
    }

    # 抽象方法，用于初始化分类器对象的参数
    @abstractmethod
    def __init__(
        self,
        loss="hinge",
        *,
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        epsilon=DEFAULT_EPSILON,
        n_jobs=None,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight=None,
        warm_start=False,
        average=False,
    ):
        # 调用父类BaseSGDClassifier的初始化方法，设置分类器的参数
        super().__init__(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            warm_start=warm_start,
            average=average,
        )
        # 设置分类器的类别权重参数和工作线程数参数
        self.class_weight = class_weight
        self.n_jobs = n_jobs

    # 部分拟合方法，用于分类器的部分拟合过程
    def _partial_fit(
        self,
        X,
        y,
        alpha,
        C,
        loss,
        learning_rate,
        max_iter,
        classes,
        sample_weight,
        coef_init,
        intercept_init,
    ):
        # 检查是否是第一次调用_fit方法
        first_call = not hasattr(self, "classes_")
        # 验证并处理输入数据，确保格式正确
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",  # 接受稀疏矩阵格式为csr
            dtype=[np.float64, np.float32],  # 数据类型为np.float64或np.float32
            order="C",  # C风格的顺序
            accept_large_sparse=False,  # 不接受大型稀疏矩阵
            reset=first_call,  # 如果是第一次调用，则重置数据
        )

        if first_call:
            # 如果是第一次调用，发出警告，建议使用False代替0来禁用平均化
            if not isinstance(self.average, (bool, np.bool_)) and self.average == 0:
                warnings.warn(
                    (
                        "Passing average=0 to disable averaging is deprecated and will"
                        " be removed in 1.7. Please use average=False instead."
                    ),
                    FutureWarning,
                )

        n_samples, n_features = X.shape

        # 检查是否是_partial_fit的第一次调用
        _check_partial_fit_first_call(self, classes)

        n_classes = self.classes_.shape[0]

        # 从输入参数中分配数据结构
        self._expanded_class_weight = compute_class_weight(
            self.class_weight, classes=self.classes_, y=y
        )
        # 检查样本权重是否符合规范
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        # 如果不存在coef_属性或者有coef_init参数，则分配参数内存
        if getattr(self, "coef_", None) is None or coef_init is not None:
            self._allocate_parameter_mem(
                n_classes=n_classes,
                n_features=n_features,
                input_dtype=X.dtype,
                coef_init=coef_init,
                intercept_init=intercept_init,
            )
        # 否则，检查特征数是否与先前的数据匹配
        elif n_features != self.coef_.shape[-1]:
            raise ValueError(
                "Number of features %d does not match previous data %d."
                % (n_features, self.coef_.shape[-1])
            )

        # 设置损失函数
        self._loss_function_ = self._get_loss_function(loss)
        # 如果不存在t_属性，则设置为1.0
        if not hasattr(self, "t_"):
            self.t_ = 1.0

        # 委托给具体的训练过程，多类别分类情况下调用_fit_multiclass方法
        if n_classes > 2:
            self._fit_multiclass(
                X,
                y,
                alpha=alpha,
                C=C,
                learning_rate=learning_rate,
                sample_weight=sample_weight,
                max_iter=max_iter,
            )
        # 二分类情况下调用_fit_binary方法
        elif n_classes == 2:
            self._fit_binary(
                X,
                y,
                alpha=alpha,
                C=C,
                learning_rate=learning_rate,
                sample_weight=sample_weight,
                max_iter=max_iter,
            )
        else:
            # 抛出异常，要求类别数必须大于1
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % n_classes
            )

        # 返回self对象
        return self

    # 定义_fit方法的另一个版本，用于处理初始化过程和参数设置
    def _fit(
        self,
        X,
        y,
        alpha,
        C,
        loss,
        learning_rate,
        coef_init=None,
        intercept_init=None,
        sample_weight=None,
        ):
            # 如果模型对象具有属性 "classes_"，则删除该属性，以便在 _partial_fit 方法中认为这不是第一次调用
            if hasattr(self, "classes_"):
                delattr(self, "classes_")

        # TODO(1.7) remove 0 from average parameter constraint
        # 如果 average 参数不是布尔型或者 np.bool_ 类型，并且其值为 0
        if not isinstance(self.average, (bool, np.bool_)) and self.average == 0:
            # 发出警告，提示将在版本 1.7 中移除对 average=0 的支持，建议改用 average=False
            warnings.warn(
                (
                    "Passing average=0 to disable averaging is deprecated and will be "
                    "removed in 1.7. Please use average=False instead."
                ),
                FutureWarning,
            )

        # labels 可以编码为 float、int 或字符串字面值
        # np.unique 按升序排序；最大的类标识是正类
        y = self._validate_data(y=y)
        # 获取类别列表
        classes = np.unique(y)

        # 如果启用了热启动并且模型已经具有 coef_ 属性
        if self.warm_start and hasattr(self, "coef_"):
            # 如果未提供 coef_init，则使用现有的 coef_
            if coef_init is None:
                coef_init = self.coef_
            # 如果未提供 intercept_init，则使用现有的 intercept_
            if intercept_init is None:
                intercept_init = self.intercept_
        else:
            # 否则，将 coef_ 和 intercept_ 设置为 None
            self.coef_ = None
            self.intercept_ = None

        # 如果 average 大于 0
        if self.average > 0:
            # 将标准化后的 coef_ 和 intercept_ 设置为当前的 coef_ 和 intercept_
            self._standard_coef = self.coef_
            self._standard_intercept = self.intercept_
            # 将平均后的 coef_ 和 intercept_ 设置为 None
            self._average_coef = None
            self._average_intercept = None

        # 清除多次调用 fit 方法时的迭代计数
        self.t_ = 1.0

        # 调用 _partial_fit 方法进行模型的部分拟合
        self._partial_fit(
            X,
            y,
            alpha,
            C,
            loss,
            learning_rate,
            self.max_iter,
            classes,
            sample_weight,
            coef_init,
            intercept_init,
        )

        # 如果设置了 tol 参数，并且 tol 大于负无穷，并且迭代次数达到了 max_iter
        if (
            self.tol is not None
            and self.tol > -np.inf
            and self.n_iter_ == self.max_iter
        ):
            # 发出警告，提示在收敛之前达到了最大迭代次数，建议增加 max_iter 以提高拟合效果
            warnings.warn(
                (
                    "Maximum number of iteration reached before "
                    "convergence. Consider increasing max_iter to "
                    "improve the fit."
                ),
                ConvergenceWarning,
            )
        # 返回自身对象
        return self
    # 在给定数据集 X 和标签 y 上拟合一个二元分类器
    def _fit_binary(self, X, y, alpha, C, sample_weight, learning_rate, max_iter):
        """Fit a binary classifier on X and y."""
        # 调用 fit_binary 函数，返回分类器的系数、截距和迭代次数
        coef, intercept, n_iter_ = fit_binary(
            self,
            1,
            X,
            y,
            alpha,
            C,
            learning_rate,
            max_iter,
            self._expanded_class_weight[1],  # 正类的权重
            self._expanded_class_weight[0],  # 负类的权重
            sample_weight,
            random_state=self.random_state,
        )

        # 更新累计的训练步数
        self.t_ += n_iter_ * X.shape[0]
        # 记录当前训练迭代次数
        self.n_iter_ = n_iter_

        # 如果需要计算平均系数
        if self.average > 0:
            # 若累计步数大于等于指定的平均步数
            if self.average <= self.t_ - 1:
                # 使用平均系数
                self.coef_ = self._average_coef.reshape(1, -1)
                self.intercept_ = self._average_intercept
            else:
                # 否则使用标准系数
                self.coef_ = self._standard_coef.reshape(1, -1)
                self._standard_intercept = np.atleast_1d(intercept)
                self.intercept_ = self._standard_intercept
        else:
            # 直接使用计算得到的系数
            self.coef_ = coef.reshape(1, -1)
            # 将截距转换成长度为1的数组
            self.intercept_ = np.atleast_1d(intercept)
    def _fit_multiclass(self, X, y, alpha, C, learning_rate, sample_weight, max_iter):
        """Fit a multi-class classifier by combining binary classifiers

        Each binary classifier predicts one class versus all others. This
        strategy is called OvA (One versus All) or OvR (One versus Rest).
        """
        # 使用多类标签预先计算验证集分割，确保类的平衡性。
        validation_mask = self._make_validation_split(y, sample_mask=sample_weight > 0)

        # 使用 joblib 并行拟合 OvA 分类器。
        # 在 fit_binary 外部为每个任务选择随机种子，避免在线程之间共享估算器的随机状态，从而可能导致非确定性行为。
        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(MAX_INT, size=len(self.classes_))
        result = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose, require="sharedmem"
        )(
            delayed(fit_binary)(
                self,
                i,
                X,
                y,
                alpha,
                C,
                learning_rate,
                max_iter,
                self._expanded_class_weight[i],
                1.0,
                sample_weight,
                validation_mask=validation_mask,
                random_state=seed,
            )
            for i, seed in enumerate(seeds)
        )

        # 取每个二元拟合的 n_iter_ 的最大值
        n_iter_ = 0.0
        for i, (_, intercept, n_iter_i) in enumerate(result):
            self.intercept_[i] = intercept
            n_iter_ = max(n_iter_, n_iter_i)

        self.t_ += n_iter_ * X.shape[0]
        self.n_iter_ = n_iter_

        if self.average > 0:
            if self.average <= self.t_ - 1.0:
                self.coef_ = self._average_coef
                self.intercept_ = self._average_intercept
            else:
                self.coef_ = self._standard_coef
                self._standard_intercept = np.atleast_1d(self.intercept_)
                self.intercept_ = self._standard_intercept

    @_fit_context(prefer_skip_nested_validation=True)
    # 定义一个方法，用于在给定样本上执行一次随机梯度下降的迭代过程
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Perform one epoch of stochastic gradient descent on given samples.

        Internally, this method uses ``max_iter = 1``. Therefore, it is not
        guaranteed that a minimum of the cost function is reached after calling
        it once. Matters such as objective convergence, early stopping, and
        learning rate adjustments should be handled by the user.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of the training data.

        y : ndarray of shape (n_samples,)
            Subset of the target values.

        classes : ndarray of shape (n_classes,), default=None
            Classes across all calls to partial_fit.
            Can be obtained by via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.

        sample_weight : array-like, shape (n_samples,), default=None
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        # 如果当前对象没有属性"classes_"，则调用内部方法进行参数验证
        if not hasattr(self, "classes_"):
            self._more_validate_params(for_partial_fit=True)

            # 如果 class_weight 设置为 "balanced"，则抛出 ValueError
            if self.class_weight == "balanced":
                raise ValueError(
                    "class_weight '{0}' is not supported for "
                    "partial_fit. In order to use 'balanced' weights,"
                    " use compute_class_weight('{0}', "
                    "classes=classes, y=y). "
                    "In place of y you can use a large enough sample "
                    "of the full training set target to properly "
                    "estimate the class frequency distributions. "
                    "Pass the resulting weights as the class_weight "
                    "parameter.".format(self.class_weight)
                )

        # 调用内部方法 _partial_fit 执行部分拟合操作，返回 self 对象的实例
        return self._partial_fit(
            X,
            y,
            alpha=self.alpha,
            C=1.0,
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_iter=1,
            classes=classes,
            sample_weight=sample_weight,
            coef_init=None,
            intercept_init=None,
        )

    # 应用装饰器 @_fit_context，传递参数 prefer_skip_nested_validation=True
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
        """Fit linear model with Stochastic Gradient Descent.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target values.

        coef_init : ndarray of shape (n_classes, n_features), default=None
            The initial coefficients to warm-start the optimization.

        intercept_init : ndarray of shape (n_classes,), default=None
            The initial intercept to warm-start the optimization.

        sample_weight : array-like, shape (n_samples,), default=None
            Weights applied to individual samples.
            If not provided, uniform weights are assumed. These weights will
            be multiplied with class_weight (passed through the
            constructor) if class_weight is specified.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        # 调用内部方法，进行更多的参数验证和设置
        self._more_validate_params()

        # 调用内部的 _fit 方法，进行模型拟合
        return self._fit(
            X,
            y,
            alpha=self.alpha,  # 设置学习率调度器的初始步长
            C=1.0,  # 设置正则化参数
            loss=self.loss,  # 设置损失函数
            learning_rate=self.learning_rate,  # 设置学习率的调度器
            coef_init=coef_init,  # 初始化系数（可选）
            intercept_init=intercept_init,  # 初始化截距（可选）
            sample_weight=sample_weight,  # 样本权重（可选）
        )
class SGDClassifier(BaseSGDClassifier):
    """Linear classifiers (SVM, logistic regression, etc.) with SGD training.

    This estimator implements regularized linear models with stochastic
    gradient descent (SGD) learning: the gradient of the loss is estimated
    each sample at a time and the model is updated along the way with a
    decreasing strength schedule (aka learning rate). SGD allows minibatch
    (online/out-of-core) learning via the `partial_fit` method.
    For best results using the default learning rate schedule, the data should
    have zero mean and unit variance.

    This implementation works with data represented as dense or sparse arrays
    of floating point values for the features. The model it fits can be
    controlled with the loss parameter; by default, it fits a linear support
    vector machine (SVM).

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using either the squared euclidean norm
    L2 or the absolute norm L1 or a combination of both (Elastic Net). If the
    parameter update crosses the 0.0 value because of the regularizer, the
    update is truncated to 0.0 to allow for learning sparse models and achieve
    online feature selection.

    Read more in the :ref:`User Guide <sgd>`.

    Parameters
    ----------
    loss : {'hinge', 'log_loss', 'modified_huber', 'squared_hinge',\
        'perceptron', 'squared_error', 'huber', 'epsilon_insensitive',\
        'squared_epsilon_insensitive'}, default='hinge'
        The loss function to be used.

        - 'hinge' gives a linear SVM.
        - 'log_loss' gives logistic regression, a probabilistic classifier.
        - 'modified_huber' is another smooth loss that brings tolerance to
          outliers as well as probability estimates.
        - 'squared_hinge' is like hinge but is quadratically penalized.
        - 'perceptron' is the linear loss used by the perceptron algorithm.
        - The other losses, 'squared_error', 'huber', 'epsilon_insensitive' and
          'squared_epsilon_insensitive' are designed for regression but can be useful
          in classification as well; see
          :class:`~sklearn.linear_model.SGDRegressor` for a description.

        More details about the losses formulas can be found in the
        :ref:`User Guide <sgd_mathematical_formulation>`.

    penalty : {'l2', 'l1', 'elasticnet', None}, default='l2'
        The penalty (aka regularization term) to be used. Defaults to 'l2'
        which is the standard regularizer for linear SVM models. 'l1' and
        'elasticnet' might bring sparsity to the model (feature selection)
        not achievable with 'l2'. No penalty is added when set to `None`.
    """

    # 定义 SGDClassifier 类，继承自 BaseSGDClassifier 类

    def __init__(self, loss='hinge', penalty='l2'):
        # 初始化方法，设置损失函数和正则化方法的参数，默认为 'hinge' 损失和 'l2' 正则化

        super().__init__(loss=loss, penalty=penalty)
        # 调用父类 BaseSGDClassifier 的初始化方法，传递损失函数和正则化参数
    alpha : float, default=0.0001
        # 正则化项的常数乘子。值越高，正则化效果越强。在 `learning_rate` 设置为 'optimal' 时，也用于计算学习率。
        # 取值范围为 `[0.0, inf)`。

    l1_ratio : float, default=0.15
        # Elastic Net 混合参数，满足 0 <= l1_ratio <= 1。
        # 当 `penalty` 设置为 'elasticnet' 时使用。
        # 取值范围为 `[0.0, 1.0]`。

    fit_intercept : bool, default=True
        # 是否估算截距项。如果为 False，则假设数据已经中心化。

    max_iter : int, default=1000
        # 训练数据的最大迭代次数（即 epochs）。
        # 仅影响 ``fit`` 方法的行为，不影响 :meth:`partial_fit` 方法。
        # 取值范围为 `[1, inf)`。

        .. versionadded:: 0.19
        # 添加于版本 0.19。

    tol : float or None, default=1e-3
        # 停止训练的准则。如果不是 None，则在 ``n_iter_no_change`` 个连续的 epochs 中，
        # 当 (loss > best_loss - tol) 时停止训练。
        # 收敛性检查根据 `early_stopping` 参数决定，是针对训练损失还是验证损失。
        # 取值范围为 `[0.0, inf)`。

        .. versionadded:: 0.19
        # 添加于版本 0.19。

    shuffle : bool, default=True
        # 每个 epoch 后是否对训练数据进行洗牌。

    verbose : int, default=0
        # 冗余度级别。
        # 取值范围为 `[0, inf)`。

    epsilon : float, default=0.1
        # epsilon-insensitive 损失函数中的 epsilon 参数；
        # 仅当 `loss` 是 'huber'、'epsilon_insensitive' 或 'squared_epsilon_insensitive' 时有效。
        # 对于 'huber'，确定在哪个阈值以下获取预测变得不那么重要。
        # 对于 epsilon-insensitive，如果当前预测与正确标签之间的差异小于此阈值，则忽略它们。
        # 取值范围为 `[0.0, inf)`。

    n_jobs : int, default=None
        # 用于执行 OVA（一对所有，用于多类问题）计算的 CPU 数量。
        # ``None`` 表示使用 1 个处理器，除非在 :obj:`joblib.parallel_backend` 上下文中。
        # ``-1`` 表示使用所有处理器。详见 :term:`Glossary <n_jobs>`。

    random_state : int, RandomState instance, default=None
        # 在 ``shuffle`` 设置为 ``True`` 时用于对数据进行洗牌。
        # 为确保多次函数调用输出的一致性，传入一个整数。
        # 详见 :term:`Glossary <random_state>`。
        # 整数取值范围为 `[0, 2**32 - 1]`。
    learning_rate : str, default='optimal'
        学习率调度方式:

        - 'constant': `eta = eta0`  # 常数学习率，`eta` 等于初始学习率 `eta0`
        - 'optimal': `eta = 1.0 / (alpha * (t + t0))`
          其中 `t0` 由Leon Bottou提出的启发式方法选择。 # 最优学习率，根据迭代次数 `t` 和 `t0` 调整学习率 `eta`
        - 'invscaling': `eta = eta0 / pow(t, power_t)`  # 逆比例缩放学习率，`eta` 随迭代次数 `t` 指数下降
        - 'adaptive': `eta = eta0`, 只要训练误差持续减少。 # 自适应学习率，如果连续 `n_iter_no_change` 次迭代训练误差不减少，学习率缩小为当前的五分之一。

            .. versionadded:: 0.20
                添加了 'adaptive' 选项

    eta0 : float, default=0.0
        'constant', 'invscaling' 或 'adaptive' 调度的初始学习率。默认值为0.0因为 'optimal' 调度不使用 `eta0`。
        取值范围必须在 `[0.0, inf)`。

    power_t : float, default=0.5
        逆比例缩放学习率的指数。
        取值范围在 `(-inf, inf)`。

    early_stopping : bool, default=False
        是否使用早停法在验证分数未改善时终止训练。若设为 `True`，将自动将训练数据的一部分作为验证集，并在验证分数连续 `n_iter_no_change` 次没有至少 `tol` 的改善时终止训练。

        .. versionadded:: 0.20
            添加了 'early_stopping' 选项

    validation_fraction : float, default=0.1
        作为早停法验证集的训练数据比例。必须在 `(0.0, 1.0)` 范围内。
        仅在 `early_stopping` 为 `True` 时使用。

        .. versionadded:: 0.20
            添加了 'validation_fraction' 选项

    n_iter_no_change : int, default=5
        在停止拟合之前等待没有改善的迭代次数。
        收敛性根据 `early_stopping` 参数检查训练损失或验证损失。
        整数值必须在 `[1, max_iter)` 范围内。

        .. versionadded:: 0.20
            添加了 'n_iter_no_change' 选项

    class_weight : dict, {class_label: weight} or "balanced", default=None
        用于类别权重的预设参数。

        关联类别的权重。如果未给出，则假定所有类别权重为1。

        "balanced" 模式使用 `y` 的值自动调整权重，反比于输入数据中类别频率的 `n_samples / (n_classes * np.bincount(y))`。
    warm_start : bool, default=False
        # 是否启用热启动。若为 True，则使用前一次 fit 调用的解作为初始化；否则，擦除之前的解。
        # 参见术语表中的“热启动”定义。

        # 当 warm_start 设置为 True 时，多次调用 fit 或 partial_fit 可能会得到不同的解，因为数据被重新洗牌的方式不同。
        # 如果使用动态学习率，学习率会根据已经观察到的样本数进行调整。调用 fit 会重置这个计数器，而 partial_fit 会增加现有的计数器。

    average : bool or int, default=False
        # 是否进行平均处理。若为 True，则计算所有更新的平均 SGD 权重，并存储在 coef_ 属性中。若设置为大于1的整数，则在总样本数达到 average 后开始平均处理。
        # 例如，average=10 表示在观察到 10 个样本后开始进行平均处理。
        # 整数值必须在范围 [1, n_samples] 内。

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features) if n_classes == 2 else \
            (n_classes, n_features)
        # 分配给特征的权重。

    intercept_ : ndarray of shape (1,) if n_classes == 2 else (n_classes,)
        # 决策函数中的常数项。

    n_iter_ : int
        # 达到停止准则之前的实际迭代次数。对于多类别拟合，是每个二元拟合中的最大值。

    classes_ : array of shape (n_classes,)
        # 类的数组。

    t_ : int
        # 训练期间执行的权重更新次数。等同于 ``(n_iter_ * n_samples + 1)``。

    n_features_in_ : int
        # 在 fit 过程中观察到的特征数量。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 fit 过程中观察到的特征名称。仅当 `X` 具有全部为字符串的特征名称时定义。

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.svm.LinearSVC : 线性支持向量分类器。
    LogisticRegression : 逻辑回归。
    Perceptron : 继承自 SGDClassifier。``Perceptron()`` 相当于 ``SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None)``。

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import SGDClassifier
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import make_pipeline
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> Y = np.array([1, 1, 2, 2])
    >>> # 始终对输入进行缩放。最便捷的方法是使用管道。
    >>> clf = make_pipeline(StandardScaler(),
    ...                     SGDClassifier(max_iter=1000, tol=1e-3))
    >>> clf.fit(X, Y)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('sgdclassifier', SGDClassifier())])
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """
    # 创建一个数据处理管道，包括标准化和随机梯度下降分类器
    # 使用训练好的分类器进行预测，输出预测结果
    # 预测结果为类别 1

    _parameter_constraints: dict = {
        **BaseSGDClassifier._parameter_constraints,
        "penalty": [StrOptions({"l2", "l1", "elasticnet"}), None],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "l1_ratio": [Interval(Real, 0, 1, closed="both")],
        "power_t": [Interval(Real, None, None, closed="neither")],
        "epsilon": [Interval(Real, 0, None, closed="left")],
        "learning_rate": [
            StrOptions({"constant", "optimal", "invscaling", "adaptive"}),
            Hidden(StrOptions({"pa1", "pa2"})),
        ],
        "eta0": [Interval(Real, 0, None, closed="left")],
    }
    # 定义了 SGDClassifier 的参数约束，继承自 BaseSGDClassifier 的约束

    def __init__(
        self,
        loss="hinge",
        *,
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        epsilon=DEFAULT_EPSILON,
        n_jobs=None,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight=None,
        warm_start=False,
        average=False,
    ):
        # 初始化函数，设置 SGDClassifier 的各种参数
        super().__init__(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            n_jobs=n_jobs,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            class_weight=class_weight,
            warm_start=warm_start,
            average=average,
        )

    def _check_proba(self):
        # 检查是否支持概率预测，若不支持则抛出异常
        if self.loss not in ("log_loss", "modified_huber"):
            raise AttributeError(
                "probability estimates are not available for loss=%r" % self.loss
            )
        return True
    # 声明一个装饰器函数，用于条件性地提供函数

    @available_if(_check_proba)
    # 根据 _check_proba 函数的返回值来决定装饰器是否可用
    # 定义预测概率的方法，用于预测模型输出的类别概率
    def predict_proba(self, X):
        """Probability estimates.

        This method is only available for log loss and modified Huber loss.

        Multiclass probability estimates are derived from binary (one-vs.-rest)
        estimates by simple normalization, as recommended by Zadrozny and
        Elkan.

        Binary probability estimates for loss="modified_huber" are given by
        (clip(decision_function(X), -1, 1) + 1) / 2. For other loss functions
        it is necessary to perform proper probability calibration by wrapping
        the classifier with
        :class:`~sklearn.calibration.CalibratedClassifierCV` instead.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.

        References
        ----------
        Zadrozny and Elkan, "Transforming classifier scores into multiclass
        probability estimates", SIGKDD'02,
        https://dl.acm.org/doi/pdf/10.1145/775047.775151

        The justification for the formula in the loss="modified_huber"
        case is in the appendix B in:
        http://jmlr.csail.mit.edu/papers/volume2/zhang02c/zhang02c.pdf
        """
        
        # 确保模型已拟合
        check_is_fitted(self)

        # 根据损失函数类型选择相应的概率预测方法
        if self.loss == "log_loss":
            return self._predict_proba_lr(X)

        elif self.loss == "modified_huber":
            # 检查是否是二分类问题
            binary = len(self.classes_) == 2
            # 计算决策函数得分
            scores = self.decision_function(X)

            if binary:
                # 对于二分类，初始化概率数组并进行修剪和缩放
                prob2 = np.ones((scores.shape[0], 2))
                prob = prob2[:, 1]
            else:
                # 对于多分类，直接使用得分作为概率
                prob = scores

            # 对得分进行修剪和缩放
            np.clip(scores, -1, 1, prob)
            prob += 1.0
            prob /= 2.0

            if binary:
                # 对于二分类问题，调整概率值
                prob2[:, 0] -= prob
                prob = prob2
            else:
                # 处理可能将所有类别概率分配为零的情况，以产生均匀的概率
                prob_sum = prob.sum(axis=1)
                all_zero = prob_sum == 0
                if np.any(all_zero):
                    prob[all_zero, :] = 1
                    prob_sum[all_zero] = len(self.classes_)

                # 进行概率归一化
                prob /= prob_sum.reshape((prob.shape[0], -1))

            return prob

        else:
            # 抛出异常，说明不支持当前损失函数的概率预测
            raise NotImplementedError(
                "predict_(log_)proba only supported when"
                " loss='log_loss' or loss='modified_huber' "
                "(%r given)" % self.loss
            )

    # 如果满足 _check_proba 函数条件，则方法可用
    @available_if(_check_proba)
    # 返回预测结果的对数概率值。

    """Log of probability estimates.

    This method is only available for log loss and modified Huber loss.

    When loss="modified_huber", probability estimates may be hard zeros
    and ones, so taking the logarithm is not possible.

    See ``predict_proba`` for details.
    """

    # 参数 X 是形状为 (n_samples, n_features) 的输入数据，用于预测。

    """
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input data for prediction.
    """

    # 返回形状为 (n_samples, n_classes) 的数组，其中每个元素是样本在模型中每个类别的对数概率。
    
    """
    Returns
    -------
    T : array-like, shape (n_samples, n_classes)
        Returns the log-probability of the sample for each class in the
        model, where classes are ordered as they are in
        `self.classes_`.
    """
    return np.log(self.predict_proba(X))


    # 返回一个字典，其中包含一些关于模型行为的额外标签信息。

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            },
            "preserves_dtype": [np.float64, np.float32],
        }
# 定义一个基础的随机梯度下降回归器，继承自RegressorMixin和BaseSGD
class BaseSGDRegressor(RegressorMixin, BaseSGD):
    
    # 定义损失函数的映射字典，将损失函数名称映射到对应的损失函数类或类与默认值的元组
    loss_functions = {
        "squared_error": (SquaredLoss,),
        "huber": (Huber, DEFAULT_EPSILON),
        "epsilon_insensitive": (EpsilonInsensitive, DEFAULT_EPSILON),
        "squared_epsilon_insensitive": (SquaredEpsilonInsensitive, DEFAULT_EPSILON),
    }

    # 定义参数约束字典，继承自BaseSGD的参数约束并添加额外的约束
    _parameter_constraints: dict = {
        **BaseSGD._parameter_constraints,
        "loss": [StrOptions(set(loss_functions))],
        "early_stopping": ["boolean"],
        "validation_fraction": [Interval(Real, 0, 1, closed="neither")],
        "n_iter_no_change": [Interval(Integral, 1, None, closed="left")],
    }

    # 抽象方法：初始化方法，定义了回归器的各种参数
    @abstractmethod
    def __init__(
        self,
        loss="squared_error",
        *,
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        epsilon=DEFAULT_EPSILON,
        random_state=None,
        learning_rate="invscaling",
        eta0=0.01,
        power_t=0.25,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        warm_start=False,
        average=False,
    ):
        # 调用父类的初始化方法，传入所有参数
        super().__init__(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            warm_start=warm_start,
            average=average,
        )

    # 定义一个内部方法_partial_fit，用于执行部分拟合过程
    def _partial_fit(
        self,
        X,
        y,
        alpha,
        C,
        loss,
        learning_rate,
        max_iter,
        sample_weight,
        coef_init,
        intercept_init,
        ):
        ):
            # 检查是否是首次调用该方法
            first_call = getattr(self, "coef_", None) is None
            # 验证输入数据 X 和 y，确保格式正确且满足要求
            X, y = self._validate_data(
                X,
                y,
                accept_sparse="csr",  # 接受稀疏矩阵格式
                copy=False,  # 不复制输入数据
                order="C",  # 使用 C 风格（行优先）存储
                dtype=[np.float64, np.float32],  # 指定数据类型为浮点数
                accept_large_sparse=False,  # 不接受大型稀疏矩阵
                reset=first_call,  # 如果是首次调用则重置验证器状态
            )
            # 将 y 的数据类型转换为 X 的数据类型
            y = y.astype(X.dtype, copy=False)

            if first_call:
                # 如果是首次调用，检查平均参数约束中是否包含 0
                if not isinstance(self.average, (bool, np.bool_)) and self.average == 0:
                    # 发出警告，建议使用 False 替代 0 来禁用平均化
                    warnings.warn(
                        (
                            "Passing average=0 to disable averaging is deprecated and will"
                            " be removed in 1.7. Please use average=False instead."
                        ),
                        FutureWarning,
                    )

            n_samples, n_features = X.shape

            # 检查并调整样本权重
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

            # 根据输入参数分配数据结构
            if first_call:
                self._allocate_parameter_mem(
                    n_classes=1,
                    n_features=n_features,
                    input_dtype=X.dtype,
                    coef_init=coef_init,
                    intercept_init=intercept_init,
                )
            # 如果启用平均化且尚未分配平均系数，则初始化
            if self.average > 0 and getattr(self, "_average_coef", None) is None:
                self._average_coef = np.zeros(n_features, dtype=X.dtype, order="C")
                self._average_intercept = np.zeros(1, dtype=X.dtype, order="C")

            # 调用内部方法执行回归拟合
            self._fit_regressor(
                X, y, alpha, C, loss, learning_rate, sample_weight, max_iter
            )

            # 返回当前对象实例，支持链式调用
            return self

        @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, sample_weight=None):
        """Perform one epoch of stochastic gradient descent on given samples.

        Internally, this method uses ``max_iter = 1``. Therefore, it is not
        guaranteed that a minimum of the cost function is reached after calling
        it once. Matters such as objective convergence and early stopping
        should be handled by the user.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training data.

        y : numpy array of shape (n_samples,)
            Subset of target values.

        sample_weight : array-like, shape (n_samples,), default=None
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        # 检查是否已经定义了 coef_ 属性，如果没有则调用 _more_validate_params 方法进行参数验证
        if not hasattr(self, "coef_"):
            self._more_validate_params(for_partial_fit=True)

        # 调用内部的 _partial_fit 方法执行部分拟合操作
        return self._partial_fit(
            X,
            y,
            self.alpha,                  # 正则化参数 alpha
            C=1.0,                       # 惩罚项参数 C，默认为1.0
            loss=self.loss,              # 损失函数类型
            learning_rate=self.learning_rate,  # 学习率类型
            max_iter=1,                  # 最大迭代次数设为1，表明只进行一次迭代
            sample_weight=sample_weight,  # 样本权重
            coef_init=None,              # 系数初始化，默认为 None
            intercept_init=None,         # 截距初始化，默认为 None
        )

    def _fit(
        self,
        X,
        y,
        alpha,
        C,
        loss,
        learning_rate,
        coef_init=None,
        intercept_init=None,
        sample_weight=None,
    ):
        # TODO(1.7) remove 0 from average parameter constraint
        # 检查参数 `average` 是否为布尔类型或者 numpy 中的布尔类型，并且不是 0
        if not isinstance(self.average, (bool, np.bool_)) and self.average == 0:
            # 发出警告，建议在版本 1.7 中移除将 average=0 用于禁用平均化的用法
            warnings.warn(
                (
                    "Passing average=0 to disable averaging is deprecated and will be "
                    "removed in 1.7. Please use average=False instead."
                ),
                FutureWarning,
            )

        # 如果开启了 `warm_start` 并且当前对象具有 `coef_` 属性
        if self.warm_start and getattr(self, "coef_", None) is not None:
            # 如果未提供 `coef_init` 参数，则使用当前对象的 `coef_` 属性作为初始系数
            if coef_init is None:
                coef_init = self.coef_
            # 如果未提供 `intercept_init` 参数，则使用当前对象的 `intercept_` 属性作为初始截距
            if intercept_init is None:
                intercept_init = self.intercept_
        else:
            # 否则，将对象的 `coef_` 和 `intercept_` 属性设置为 None
            self.coef_ = None
            self.intercept_ = None

        # 清除多次调用 fit 方法时的迭代计数
        self.t_ = 1.0

        # 调用 _partial_fit 方法，进行模型的部分拟合
        self._partial_fit(
            X,
            y,
            alpha,
            C,
            loss,
            learning_rate,
            self.max_iter,
            sample_weight,
            coef_init,
            intercept_init,
        )

        # 如果设置了 `tol` 参数，并且 `n_iter_` 等于 `max_iter`，发出警告
        if (
            self.tol is not None
            and self.tol > -np.inf
            and self.n_iter_ == self.max_iter
        ):
            warnings.warn(
                (
                    "Maximum number of iteration reached before "
                    "convergence. Consider increasing max_iter to "
                    "improve the fit."
                ),
                ConvergenceWarning,
            )

        # 返回对象本身，用于方法链式调用
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    # 使用装饰器定义 fit 方法，进行线性模型的 SGD 拟合
    def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
        """Fit linear model with Stochastic Gradient Descent.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target values.

        coef_init : ndarray of shape (n_features,), default=None
            The initial coefficients to warm-start the optimization.

        intercept_init : ndarray of shape (1,), default=None
            The initial intercept to warm-start the optimization.

        sample_weight : array-like, shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Fitted `SGDRegressor` estimator.
        """
        # 进一步验证参数的合法性和有效性
        self._more_validate_params()

        # 调用 _fit 方法进行具体的拟合过程
        return self._fit(
            X,
            y,
            alpha=self.alpha,
            C=1.0,
            loss=self.loss,
            learning_rate=self.learning_rate,
            coef_init=coef_init,
            intercept_init=intercept_init,
            sample_weight=sample_weight,
        )
    # 使用线性模型进行预测

    def _decision_function(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            输入数据，可以是数组或稀疏矩阵

        Returns
        -------
        ndarray of shape (n_samples,)
            每个样本在X中的预测目标值。
        """
        # 检查模型是否已拟合
        check_is_fitted(self)

        # 验证输入数据X，并根据需要接受稀疏矩阵格式，不重置数据
        X = self._validate_data(X, accept_sparse="csr", reset=False)

        # 计算预测得分，使用安全的稀疏矩阵乘法，加上截距项
        scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_

        # 将预测分数展平为一维数组并返回
        return scores.ravel()

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            输入数据。

        Returns
        -------
        ndarray of shape (n_samples,)
           每个样本在X中的预测目标值。
        """
        # 调用_decision_function方法进行预测并返回结果
        return self._decision_function(X)

    def _fit_regressor(
        self, X, y, alpha, C, loss, learning_rate, sample_weight, max_iter
        ):
        #
        # 获取损失函数
        loss_function = self._get_loss_function(loss)
        # 获取惩罚类型
        penalty_type = self._get_penalty_type(self.penalty)
        # 获取学习率类型
        learning_rate_type = self._get_learning_rate_type(learning_rate)

        # 如果对象没有属性"t_"，则设置为1.0
        if not hasattr(self, "t_"):
            self.t_ = 1.0

        # 创建验证集掩码
        validation_mask = self._make_validation_split(y, sample_mask=sample_weight > 0)
        # 创建验证集评分回调
        validation_score_cb = self._make_validation_score_cb(
            validation_mask, X, y, sample_weight
        )

        # 检查随机状态并生成种子
        random_state = check_random_state(self.random_state)
        seed = random_state.randint(0, MAX_INT)

        # 创建数据集和截距衰减
        dataset, intercept_decay = make_dataset(
            X, y, sample_weight, random_state=random_state
        )

        # 设置容差值
        tol = self.tol if self.tol is not None else -np.inf

        # 如果启用了平均化，则使用平均化系数和截距
        if self.average:
            coef = self._standard_coef
            intercept = self._standard_intercept
            average_coef = self._average_coef
            average_intercept = self._average_intercept
        else:
            coef = self.coef_
            intercept = self.intercept_
            average_coef = None  # 未使用
            average_intercept = [0]  # 未使用

        # 获取普通随机梯度下降函数
        _plain_sgd = _get_plain_sgd_function(input_dtype=coef.dtype)
        # 执行随机梯度下降
        coef, intercept, average_coef, average_intercept, self.n_iter_ = _plain_sgd(
            coef,
            intercept[0],
            average_coef,
            average_intercept[0],
            loss_function,
            penalty_type,
            alpha,
            C,
            self.l1_ratio,
            dataset,
            validation_mask,
            self.early_stopping,
            validation_score_cb,
            int(self.n_iter_no_change),
            max_iter,
            tol,
            int(self.fit_intercept),
            int(self.verbose),
            int(self.shuffle),
            seed,
            1.0,
            1.0,
            learning_rate_type,
            self.eta0,
            self.power_t,
            0,
            self.t_,
            intercept_decay,
            self.average,
        )

        # 更新迭代次数
        self.t_ += self.n_iter_ * X.shape[0]

        # 如果启用了平均化
        if self.average > 0:
            self._average_intercept = np.atleast_1d(average_intercept)
            self._standard_intercept = np.atleast_1d(intercept)

            if self.average <= self.t_ - 1.0:
                # 更新系数和截距
                self.coef_ = average_coef
                self.intercept_ = np.atleast_1d(average_intercept)
            else:
                self.coef_ = coef
                self.intercept_ = np.atleast_1d(intercept)

        else:
            self.intercept_ = np.atleast_1d(intercept)
# 定义一个SGDRegressor类，继承自BaseSGDRegressor，用于通过随机梯度下降（SGD）来拟合线性模型。

class SGDRegressor(BaseSGDRegressor):
    """Linear model fitted by minimizing a regularized empirical loss with SGD.

    # 通过随机梯度下降（SGD）最小化带有正则化的经验损失来拟合的线性模型。

    SGD stands for Stochastic Gradient Descent: the gradient of the loss is
    estimated each sample at a time and the model is updated along the way with
    a decreasing strength schedule (aka learning rate).

    # SGD代表随机梯度下降：每次估计损失的梯度时，逐个样本更新模型，并且使用递减的学习率（即学习率）来更新模型。

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using either the squared euclidean norm
    L2 or the absolute norm L1 or a combination of both (Elastic Net). If the
    parameter update crosses the 0.0 value because of the regularizer, the
    update is truncated to 0.0 to allow for learning sparse models and achieve
    online feature selection.

    # 正则化器是添加到损失函数中的惩罚项，通过使用平方欧几里得范数L2、绝对范数L1或两者的组合（弹性网络）将模型参数收缩向零向量。如果由于正则化器参数更新越过了0.0值，则更新截断为0.0，以便学习稀疏模型并实现在线特征选择。

    This implementation works with data represented as dense numpy arrays of
    floating point values for the features.

    # 该实现适用于以浮点数特征表示的密集numpy数组数据。

    Read more in the :ref:`User Guide <sgd>`.

    # 详细信息请参阅用户指南<sgd>。

    Parameters
    ----------
    loss : str, default='squared_error'
        The loss function to be used. The possible values are 'squared_error',
        'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'

        # 要使用的损失函数。可能的值有'squared_error'（平方误差）、'huber'（Huber损失）、'epsilon_insensitive'（ε不敏感损失）或'squared_epsilon_insensitive'（平方ε不敏感损失）。

        The 'squared_error' refers to the ordinary least squares fit.
        'huber' modifies 'squared_error' to focus less on getting outliers
        correct by switching from squared to linear loss past a distance of
        epsilon. 'epsilon_insensitive' ignores errors less than epsilon and is
        linear past that; this is the loss function used in SVR.
        'squared_epsilon_insensitive' is the same but becomes squared loss past
        a tolerance of epsilon.

        # 'squared_error'是指普通的最小二乘拟合。'huber'修改'squared_error'，通过在距离ε以后从平方损失切换到线性损失来减少对离群值的关注。'epsilon_insensitive'忽略小于ε的误差，并且在此之后是线性的；这是SVR中使用的损失函数。'squared_epsilon_insensitive'与之类似，但是在ε的公差之后变成平方损失。

        More details about the losses formulas can be found in the
        :ref:`User Guide <sgd_mathematical_formulation>`.

        # 损失函数的详细公式可以在用户指南<sgd_mathematical_formulation>中找到。

    penalty : {'l2', 'l1', 'elasticnet', None}, default='l2'
        The penalty (aka regularization term) to be used. Defaults to 'l2'
        which is the standard regularizer for linear SVM models. 'l1' and
        'elasticnet' might bring sparsity to the model (feature selection)
        not achievable with 'l2'. No penalty is added when set to `None`.

        # 要使用的惩罚（即正则化项）。默认为'l2'，这是线性SVM模型的标准正则化器。'l1'和'elasticnet'可能会使模型稀疏化（特征选择），而这是'l2'不能实现的。当设置为`None`时不添加惩罚项。

    alpha : float, default=0.0001
        Constant that multiplies the regularization term. The higher the
        value, the stronger the regularization. Also used to compute the
        learning rate when `learning_rate` is set to 'optimal'.
        Values must be in the range `[0.0, inf)`.

        # 乘以正则化项的常数。值越高，正则化越强。还用于计算当`learning_rate`设置为'optimal'时的学习率。值必须在范围`[0.0, inf)`内。

    l1_ratio : float, default=0.15
        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
        Only used if `penalty` is 'elasticnet'.
        Values must be in the range `[0.0, 1.0]`.

        # 弹性网络的混合参数，其中0 <= l1_ratio <= 1。l1_ratio=0对应L2惩罚，l1_ratio=1对应L1惩罚。仅当`penalty`为'elasticnet'时使用。值必须在范围`[0.0, 1.0]`内。

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered.

        # 是否应该估计截距。如果为False，则假定数据已经居中。

    """
    max_iter : int, default=1000
        # 最大迭代次数，即训练数据的最大遍历次数（也称为 epochs）。
        # 仅影响 ``fit`` 方法的行为，不影响 :meth:`partial_fit` 方法。
        # 取值范围为 `[1, inf)`。

        .. versionadded:: 0.19

    tol : float or None, default=1e-3
        # 停止训练的标准。如果不是 None，则当 (loss > best_loss - tol) 连续
        # ``n_iter_no_change`` 个 epochs 时，训练将停止。
        # 收敛性根据 `early_stopping` 参数检查训练损失或验证损失。
        # 取值范围为 `[0.0, inf)`。

        .. versionadded:: 0.19

    shuffle : bool, default=True
        # 每个 epoch 后是否对训练数据进行洗牌。

    verbose : int, default=0
        # 冗长级别。
        # 取值范围为 `[0, inf)`。

    epsilon : float, default=0.1
        # epsilon-insensitive 损失函数中的 epsilon；仅当 `loss` 为
        # 'huber', 'epsilon_insensitive', 或 'squared_epsilon_insensitive' 时。
        # 对于 'huber'，确定变得不那么重要的阈值。
        # 对于 epsilon-insensitive，如果当前预测与正确标签之间的任何差异小于此阈值，则忽略它们。
        # 取值范围为 `[0.0, inf)`。

    random_state : int, RandomState instance, default=None
        # 用于在 ``shuffle`` 设置为 ``True`` 时对数据进行洗牌。
        # 传递一个 int 以在多次函数调用之间获得可重复的输出。
        # 参见 :term:`Glossary <random_state>`。

    learning_rate : str, default='invscaling'
        # 学习率调度：

        # - 'constant': `eta = eta0`
        # - 'optimal': `eta = 1.0 / (alpha * (t + t0))`
        #   其中 t0 是由 Leon Bottou 提出的启发式选择的。
        # - 'invscaling': `eta = eta0 / pow(t, power_t)`
        # - 'adaptive': eta = eta0，只要训练保持下降。
        #   每次连续 n_iter_no_change 个 epochs 未能通过 tol 减少训练损失
        #   或如果 early_stopping 为 True，则未能通过 tol 增加验证分数，
        #   当前学习率将除以 5。

            .. versionadded:: 0.20
                # 添加了 'adaptive' 选项

    eta0 : float, default=0.01
        # 'constant', 'invscaling' 或 'adaptive' 调度的初始学习率。默认值为 0.01。
        # 取值范围为 `[0.0, inf)`。

    power_t : float, default=0.25
        # 逆比例缩放学习率的指数。
        # 取值范围为 `(-inf, inf)`.
    # 是否启用早期停止策略，用于在验证分数不再改善时终止训练
    early_stopping : bool, default=False
        # 如果设置为 True，将自动保留训练数据的一部分作为验证集，
        # 当验证分数在 `score` 方法返回的分数不再改善至少 `tol` 的情况下，
        # 连续 `n_iter_no_change` 次时，终止训练。
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to True, it will automatically set aside
        a fraction of training data as validation and terminate
        training when validation score returned by the `score` method is not
        improving by at least `tol` for `n_iter_no_change` consecutive
        epochs.

        .. versionadded:: 0.20
            # 添加了 'early_stopping' 选项
            Added 'early_stopping' option

    # 用作早期停止策略的验证集的比例
    validation_fraction : float, default=0.1
        # 将训练数据的比例设置为验证集以进行早期停止策略。
        # 仅在 `early_stopping` 为 True 时使用。
        # 值必须在 `(0.0, 1.0)` 的范围内。
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if `early_stopping` is True.
        Values must be in the range `(0.0, 1.0)`.

        .. versionadded:: 0.20
            # 添加了 'validation_fraction' 选项
            Added 'validation_fraction' option

    # 在停止拟合之前等待改善的迭代次数
    n_iter_no_change : int, default=5
        # 在停止拟合之前等待没有改进的迭代次数。
        # 收敛性根据 `early_stopping` 参数检查训练损失或验证损失。
        # 整数值必须在 `[1, max_iter)` 范围内。
        Number of iterations with no improvement to wait before stopping
        fitting.
        Convergence is checked against the training loss or the
        validation loss depending on the `early_stopping` parameter.
        Integer values must be in the range `[1, max_iter)`.

        .. versionadded:: 0.20
            # 添加了 'n_iter_no_change' 选项
            Added 'n_iter_no_change' option

    # 是否启用热启动
    warm_start : bool, default=False
        # 当设置为 True 时，重用上一次调用 `fit` 的解作为初始化，
        # 否则，只是擦除先前的解。
        # 参见 :term:`术语表 <warm_start>`。
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

        # 当 warm_start 设置为 True 时，重复调用 `fit` 或 `partial_fit`
        # 可能会导致与仅调用 `fit` 时不同的解，这是因为数据洗牌的方式不同。
        # 如果使用动态学习率，学习率将根据已经观察到的样本数进行调整。
        # 调用 `fit` 会重置计数器，而 `partial_fit` 则会增加现有的计数器。
        Repeatedly calling fit or partial_fit when warm_start is True can
        result in a different solution than when calling fit a single time
        because of the way the data is shuffled.
        If a dynamic learning rate is used, the learning rate is adapted
        depending on the number of samples already seen. Calling ``fit`` resets
        this counter, while ``partial_fit``  will result in increasing the
        existing counter.

    # 是否计算平均的 SGD 权重
    average : bool or int, default=False
        # 当设置为 True 时，计算所有更新的平均 SGD 权重，并将结果存储在 `coef_` 属性中。
        # 如果设置为大于 1 的整数，将在总样本数达到 `average` 后开始进行平均。
        # 因此，`average=10` 将在看到 10 个样本后开始进行平均。
        When set to True, computes the averaged SGD weights across all
        updates and stores the result in the ``coef_`` attribute. If set to
        an int greater than 1, averaging will begin once the total number of
        samples seen reaches `average`. So ``average=10`` will begin
        averaging after seeing 10 samples.

    # 权重分配给特征的数组
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Weights assigned to the features.

    # 截距项的数组
    intercept_ : ndarray of shape (1,)
        The intercept term.

    # 达到停止条件之前实际的迭代次数
    n_iter_ : int
        The actual number of iterations before reaching the stopping criterion.

    # 训练期间执行的权重更新次数
    t_ : int
        Number of weight updates performed during training.
        Same as ``(n_iter_ * n_samples + 1)``.

    # 在 `fit` 期间看到的特征数
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24
            # 添加了 'n_features_in_' 选项
            Added 'n_features_in_' option
    # 定义一个属性 `_parameter_constraints`，是一个字典，包含了基于 `BaseSGDRegressor._parameter_constraints` 的扩展
    _parameter_constraints: dict = {
        **BaseSGDRegressor._parameter_constraints,
        "penalty": [StrOptions({"l2", "l1", "elasticnet"}), None],  # 约束参数 `penalty` 的取值范围为 {"l2", "l1", "elasticnet"} 或者 None
        "alpha": [Interval(Real, 0, None, closed="left")],  # 约束参数 `alpha` 的取值范围为大于0的实数
        "l1_ratio": [Interval(Real, 0, 1, closed="both")],  # 约束参数 `l1_ratio` 的取值范围为 [0, 1]
        "power_t": [Interval(Real, None, None, closed="neither")],  # 约束参数 `power_t` 的取值范围为开区间
        "learning_rate": [
            StrOptions({"constant", "optimal", "invscaling", "adaptive"}),  # 约束参数 `learning_rate` 的取值范围为 {"constant", "optimal", "invscaling", "adaptive"}
            Hidden(StrOptions({"pa1", "pa2"})),  # 隐藏参数，取值为 {"pa1", "pa2"}
        ],
        "epsilon": [Interval(Real, 0, None, closed="left")],  # 约束参数 `epsilon` 的取值范围为大于0的实数
        "eta0": [Interval(Real, 0, None, closed="left")],  # 约束参数 `eta0` 的取值范围为大于0的实数
    }

    # 初始化方法，定义了多个参数以配置 SGDRegressor 模型的属性
    def __init__(
        self,
        loss="squared_error",  # 损失函数，默认为平方误差
        *,
        penalty="l2",  # 正则化项，默认为 "l2"
        alpha=0.0001,  # 正则化项的系数，默认为 0.0001
        l1_ratio=0.15,  # L1 正则化的比例，默认为 0.15
        fit_intercept=True,  # 是否计算截距，默认为 True
        max_iter=1000,  # 最大迭代次数，默认为 1000
        tol=1e-3,  # 迭代收敛的阈值，默认为 1e-3
        shuffle=True,  # 是否在每次迭代前洗牌样本，默认为 True
        verbose=0,  # 是否输出详细日志信息，默认为 0，即不输出
        epsilon=DEFAULT_EPSILON,  # epsilon 参数的默认值，用于某些更新策略，默认值根据上下文确定
        random_state=None,  # 随机数种子，默认为 None
        learning_rate="invscaling",  # 学习率更新策略，默认为 "invscaling"
        eta0=0.01,  # 初始学习率，默认为 0.01
        power_t=0.25,  # 学习率的幂指数，默认为 0.25
        early_stopping=False,  # 是否启用早停止策略，默认为 False
        validation_fraction=0.1,  # 验证集所占比例，默认为 0.1
        n_iter_no_change=5,  # 连续几次迭代效果没有改善时停止迭代，默认为 5
        warm_start=False,  # 是否热启动，默认为 False
        average=False,  # 是否计算平均模型，默认为 False
    ):
        super().__init__(
            # 调用父类的初始化方法，传入以下参数来初始化LinearModel对象
            loss=loss,                         # 损失函数，用于计算损失
            penalty=penalty,                   # 惩罚项，用于正则化
            alpha=alpha,                       # 正则化强度
            l1_ratio=l1_ratio,                 # L1 正则化与总正则化的比例
            fit_intercept=fit_intercept,       # 是否拟合截距
            max_iter=max_iter,                 # 最大迭代次数
            tol=tol,                           # 收敛判断的容差
            shuffle=shuffle,                   # 是否在每次迭代前打乱数据
            verbose=verbose,                   # 是否输出详细信息
            epsilon=epsilon,                   # 用于计算平滑 Hinge 损失的阈值
            random_state=random_state,         # 控制随机性的种子
            learning_rate=learning_rate,       # 学习率的类型
            eta0=eta0,                         # 初始学习率
            power_t=power_t,                   # 学习率的指数衰减率
            early_stopping=early_stopping,     # 是否启用早停
            validation_fraction=validation_fraction,  # 训练数据的验证集比例
            n_iter_no_change=n_iter_no_change,        # 连续迭代不改变的次数
            warm_start=warm_start,             # 是否重用前一次训练结果
            average=average,                   # 是否计算平均 SGD 权重
        )

    def _more_tags(self):
        # 返回一个字典，包含一些额外的标签信息，用于扩展测试和功能性标记
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            },
            "preserves_dtype": [np.float64, np.float32],  # 表明模型能够保留数据类型为 np.float64 和 np.float32 的数据
        }
class SGDOneClassSVM(BaseSGD, OutlierMixin):
    """Solves linear One-Class SVM using Stochastic Gradient Descent.

    This implementation is meant to be used with a kernel approximation
    technique (e.g. `sklearn.kernel_approximation.Nystroem`) to obtain results
    similar to `sklearn.svm.OneClassSVM` which uses a Gaussian kernel by
    default.

    Read more in the :ref:`User Guide <sgd_online_one_class_svm>`.

    .. versionadded:: 1.0

    Parameters
    ----------
    nu : float, default=0.5
        The nu parameter of the One Class SVM: an upper bound on the
        fraction of training errors and a lower bound of the fraction of
        support vectors. Should be in the interval (0, 1]. By default 0.5
        will be taken.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. Defaults to True.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        `partial_fit`. Defaults to 1000.
        Values must be in the range `[1, inf)`.

    tol : float or None, default=1e-3
        The stopping criterion. If it is not None, the iterations will stop
        when (loss > previous_loss - tol). Defaults to 1e-3.
        Values must be in the range `[0.0, inf)`.

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.
        Defaults to True.

    verbose : int, default=0
        The verbosity level.

    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    learning_rate : {'constant', 'optimal', 'invscaling', 'adaptive'}, default='optimal'
        The learning rate schedule to use with `fit`. (If using `partial_fit`,
        learning rate must be controlled directly).

        - 'constant': `eta = eta0`
          Fixed learning rate.
        - 'optimal': `eta = 1.0 / (alpha * (t + t0))`
          Adaptive learning rate based on heuristic by Leon Bottou.
        - 'invscaling': `eta = eta0 / pow(t, power_t)`
          Inverse scaling learning rate.
        - 'adaptive': `eta = eta0`
          Adaptive learning rate that decreases when training stagnates.

    eta0 : float, default=0.0
        The initial learning rate for the 'constant', 'invscaling' or
        'adaptive' schedules. The default value is 0.0 as eta0 is not used by
        the default schedule 'optimal'.
        Values must be in the range `[0.0, inf)`.
    """
    power_t : float, default=0.5
        The exponent for inverse scaling learning rate.
        Values must be in the range `(-inf, inf)`.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

        Repeatedly calling fit or partial_fit when warm_start is True can
        result in a different solution than when calling fit a single time
        because of the way the data is shuffled.
        If a dynamic learning rate is used, the learning rate is adapted
        depending on the number of samples already seen. Calling ``fit`` resets
        this counter, while ``partial_fit``  will result in increasing the
        existing counter.

    average : bool or int, default=False
        When set to True, computes the averaged SGD weights and stores the
        result in the ``coef_`` attribute. If set to an int greater than 1,
        averaging will begin once the total number of samples seen reaches
        average. So ``average=10`` will begin averaging after seeing 10
        samples.
    # 定义一个参数约束字典，继承自 BaseSGD 类的参数约束，并添加了一些特定于 OneClassSVM 的参数约束
    _parameter_constraints: dict = {
        **BaseSGD._parameter_constraints,  # 继承基类 BaseSGD 的参数约束
        "nu": [Interval(Real, 0.0, 1.0, closed="right")],  # 约束参数 nu 的取值范围为 [0.0, 1.0]
        "learning_rate": [  # 约束参数 learning_rate 的取值范围为一组字符串选项
            StrOptions({"constant", "optimal", "invscaling", "adaptive"}),
            Hidden(StrOptions({"pa1", "pa2"})),  # 隐藏的参数选项，不在文档中显示
        ],
        "eta0": [Interval(Real, 0, None, closed="left")],  # 约束参数 eta0 的取值范围为 [0, ∞)
        "power_t": [Interval(Real, None, None, closed="neither")],  # 约束参数 power_t 的取值范围为开区间
    }
    
    # 初始化 SGDOneClassSVM 类的构造函数
    def __init__(
        self,
        nu=0.5,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        warm_start=False,
        average=False,
    ):
        self.nu = nu  # 设置参数 nu 的初始值
        # 调用父类 BaseSGD 的构造函数，设置一些默认参数和传入的参数值
        super(SGDOneClassSVM, self).__init__(
            loss="hinge",  # 损失函数选择为 hinge
            penalty="l2",  # 惩罚项选择为 l2
            C=1.0,  # 惩罚系数 C 设置为 1.0
            l1_ratio=0,  # l1_ratio 设置为 0，表明使用 l2 正则化
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=DEFAULT_EPSILON,  # 默认的 epsilon 值
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=False,  # 不启用 early stopping
            validation_fraction=0.1,  # 验证集的比例为 0.1
            n_iter_no_change=5,  # 连续迭代次数不变的最大次数为 5
            warm_start=warm_start,
            average=average,
        )
    
    # 定义 SGDOneClassSVM 类的局部拟合方法 _partial_fit
    def _partial_fit(
        self,
        X,
        alpha,
        C,
        loss,
        learning_rate,
        max_iter,
        sample_weight,
        coef_init,
        offset_init,
        ):
            # 检查是否是第一次调用，即模型是否已经有coef_属性
            first_call = getattr(self, "coef_", None) is None
            # 验证数据 X，确保其格式和类型满足要求
            X = self._validate_data(
                X,
                None,
                accept_sparse="csr",  # 接受稀疏矩阵格式
                dtype=[np.float64, np.float32],  # 数据类型要求为浮点数
                order="C",  # C风格的数组顺序
                accept_large_sparse=False,  # 不接受大型稀疏矩阵
                reset=first_call,  # 如果是第一次调用，则重置数据
            )

            if first_call:
                # 如果是第一次调用，并且average参数不是布尔类型或者是0，发出警告
                if not isinstance(self.average, (bool, np.bool_)) and self.average == 0:
                    warnings.warn(
                        (
                            "Passing average=0 to disable averaging is deprecated and will"
                            " be removed in 1.7. Please use average=False instead."
                        ),
                        FutureWarning,
                    )

            n_features = X.shape[1]

            # 根据输入参数检查样本权重，确保其格式和数据类型与X匹配
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

            # 根据条件分配参数内存空间
            if getattr(self, "coef_", None) is None or coef_init is not None:
                self._allocate_parameter_mem(
                    n_classes=1,
                    n_features=n_features,
                    input_dtype=X.dtype,
                    coef_init=coef_init,
                    intercept_init=offset_init,
                    one_class=1,
                )
            elif n_features != self.coef_.shape[-1]:
                # 如果特征数不匹配，则引发值错误
                raise ValueError(
                    "Number of features %d does not match previous data %d."
                    % (n_features, self.coef_.shape[-1])
                )

            if self.average and getattr(self, "_average_coef", None) is None:
                # 如果需要平均化且平均系数尚未初始化，则分配内存空间
                self._average_coef = np.zeros(n_features, dtype=X.dtype, order="C")
                self._average_intercept = np.zeros(1, dtype=X.dtype, order="C")

            # 设置损失函数
            self._loss_function_ = self._get_loss_function(loss)
            if not hasattr(self, "t_"):
                self.t_ = 1.0

            # 委托具体的训练过程
            self._fit_one_class(
                X,
                alpha=alpha,
                C=C,
                learning_rate=learning_rate,
                sample_weight=sample_weight,
                max_iter=max_iter,
            )

            return self

        @_fit_context(prefer_skip_nested_validation=True)
    # 针对线性单类SVM使用随机梯度下降进行部分拟合。

    def partial_fit(self, X, y=None, sample_weight=None):
        """Fit linear One-Class SVM with Stochastic Gradient Descent.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of the training data.
        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        # 如果对象中没有 coef_ 属性，则调用 _more_validate_params 进行更多的参数验证
        if not hasattr(self, "coef_"):
            self._more_validate_params(for_partial_fit=True)

        # 计算 alpha 值
        alpha = self.nu / 2
        # 调用 _partial_fit 方法进行部分拟合
        return self._partial_fit(
            X,
            alpha,
            C=1.0,
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_iter=1,
            sample_weight=sample_weight,
            coef_init=None,
            offset_init=None,
        )

    def _fit(
        self,
        X,
        alpha,
        C,
        loss,
        learning_rate,
        coef_init=None,
        offset_init=None,
        sample_weight=None,
    ):
        # TODO(1.7) remove 0 from average parameter constraint
        # 如果 average 参数不是布尔型且等于 0，则发出警告
        if not isinstance(self.average, (bool, np.bool_)) and self.average == 0:
            warnings.warn(
                (
                    "Passing average=0 to disable averaging is deprecated and will be "
                    "removed in 1.7. Please use average=False instead."
                ),
                FutureWarning,
            )

        # 如果设置了 warm_start 并且对象已有 coef_ 属性，则使用已有的 coef_ 和 offset_ 进行初始化
        if self.warm_start and hasattr(self, "coef_"):
            if coef_init is None:
                coef_init = self.coef_
            if offset_init is None:
                offset_init = self.offset_
        else:
            # 否则，将 coef_ 和 offset_ 设置为 None
            self.coef_ = None
            self.offset_ = None

        # 清除迭代计数以供多次调用 fit 方法使用
        self.t_ = 1.0

        # 调用 _partial_fit 方法进行拟合
        self._partial_fit(
            X,
            alpha,
            C,
            loss,
            learning_rate,
            self.max_iter,
            sample_weight,
            coef_init,
            offset_init,
        )

        # 如果设置了 tol 参数且 tol 大于负无穷，并且达到了最大迭代次数但未收敛，则发出警告
        if (
            self.tol is not None
            and self.tol > -np.inf
            and self.n_iter_ == self.max_iter
        ):
            warnings.warn(
                (
                    "Maximum number of iteration reached before "
                    "convergence. Consider increasing max_iter to "
                    "improve the fit."
                ),
                ConvergenceWarning,
            )

        # 返回 self 对象
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, coef_init=None, offset_init=None, sample_weight=None):
        """
        使用随机梯度下降拟合线性单类SVM模型。

        这解决了等价于单类SVM原始优化问题的优化问题，并返回权重向量 w 和偏移量 rho，
        使得决策函数为 <w, x> - rho。

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            训练数据。
        y : Ignored
            未使用，为了API的一致性而存在。
        coef_init : array, shape (n_classes, n_features)
            初始化系数，用于优化的热启动。
        offset_init : array, shape (n_classes,)
            初始化偏移量，用于优化的热启动。
        sample_weight : array-like, shape (n_samples,), optional
            应用于各个样本的权重。
            如果未提供，则假定均匀权重。这些权重将与 class_weight（通过构造函数传递）相乘，
            如果指定了 class_weight。

        Returns
        -------
        self : object
            返回拟合后的实例。
        """
        self._more_validate_params()  # 调用内部方法，进一步验证参数

        alpha = self.nu / 2  # 计算 alpha 参数，nu 为模型的超参数
        self._fit(
            X,
            alpha=alpha,
            C=1.0,
            loss=self.loss,
            learning_rate=self.learning_rate,
            coef_init=coef_init,
            offset_init=offset_init,
            sample_weight=sample_weight,
        )  # 调用内部方法 _fit 进行模型拟合

        return self  # 返回拟合后的实例本身

    def decision_function(self, X):
        """
        返回到分隔超平面的符号距离。

        对于内点，符号距离为正；对于异常点，符号距离为负。

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            测试数据。

        Returns
        -------
        dec : array-like, shape (n_samples,)
            样本的决策函数值。
        """

        check_is_fitted(self, "coef_")  # 检查模型是否已拟合

        X = self._validate_data(X, accept_sparse="csr", reset=False)  # 验证并转换测试数据格式
        decisions = safe_sparse_dot(X, self.coef_.T, dense_output=True) - self.offset_  # 计算决策函数值

        return decisions.ravel()  # 返回决策函数值的扁平化数组

    def score_samples(self, X):
        """
        返回样本的原始评分函数值。

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            测试数据。

        Returns
        -------
        score_samples : array-like, shape (n_samples,)
            样本的未经偏移的评分函数值。
        """
        score_samples = self.decision_function(X) + self.offset_  # 计算样本的评分函数值
        return score_samples  # 返回评分函数值数组
    # 定义一个预测方法，用于返回样本的标签（1 表示内点，-1 表示异常点）

    def predict(self, X):
        """Return labels (1 inlier, -1 outlier) of the samples.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Testing data.

        Returns
        -------
        y : array, shape (n_samples,)
            Labels of the samples.
        """
        # 根据决策函数的结果，将大于等于0的结果设为1，小于0的结果设为-1
        y = (self.decision_function(X) >= 0).astype(np.int32)
        y[y == 0] = -1  # 为了与异常检测器保持一致，将等于0的标签设为-1
        return y

    # 定义一个方法返回更多的标签信息
    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                )
            },
            "preserves_dtype": [np.float64, np.float32],
        }
```
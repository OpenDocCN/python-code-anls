# `D:\src\scipysrc\scikit-learn\sklearn\svm\_base.py`

```
# 导入警告模块，用于处理警告信息
import warnings
# 导入抽象基类元类和抽象方法装饰器
from abc import ABCMeta, abstractmethod
# 导入整数和实数类型
from numbers import Integral, Real

# 导入NumPy库并使用别名np
import numpy as np
# 导入SciPy稀疏矩阵模块并使用别名sp
import scipy.sparse as sp

# 导入基本估计器类、分类器混合类和_fit_context方法
from ..base import BaseEstimator, ClassifierMixin, _fit_context
# 导入收敛警告和模型未拟合异常
from ..exceptions import ConvergenceWarning, NotFittedError
# 导入标签编码器
from ..preprocessing import LabelEncoder
# 导入数据校验、随机状态检查、列或1d数据、计算类权重等实用工具函数
from ..utils import check_array, check_random_state, column_or_1d, compute_class_weight
# 导入区间和字符串选项参数验证
from ..utils._param_validation import Interval, StrOptions
# 导入安全稀疏点积方法
from ..utils.extmath import safe_sparse_dot
# 导入metaestimators中的available_if函数
from ..utils.metaestimators import available_if
# 导入多类分类决策函数和检查分类目标工具函数
from ..utils.multiclass import _ovr_decision_function, check_classification_targets
# 导入数据校验模块的各种功能
from ..utils.validation import (
    _check_large_sparse,
    _check_sample_weight,
    _num_samples,
    check_consistent_length,
    check_is_fitted,
)
# 导入_liblinear作为liblinear模块（类型忽略）
from . import _liblinear as liblinear  # type: ignore

# mypy错误：错误：模块'sklearn.svm'没有'_libsvm'属性
# （其他导入也是如此）
# 导入_libsvm作为libsvm模块（类型忽略）
from . import _libsvm as libsvm  # type: ignore
# 导入_libsvm_sparse作为libsvm_sparse模块（类型忽略）
from . import _libsvm_sparse as libsvm_sparse  # type: ignore

# 定义LIBSVM_IMPL常量，包含支持的LibSVM实现类型列表
LIBSVM_IMPL = ["c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr"]


def _one_vs_one_coef(dual_coef, n_support, support_vectors):
    """Generate primal coefficients from dual coefficients
    for the one-vs-one multi class LibSVM in the case
    of a linear kernel."""

    # 计算类别数目
    n_class = dual_coef.shape[0] + 1

    # 初始化系数列表
    coef = []
    # 计算每个类别的支持向量的位置索引
    sv_locs = np.cumsum(np.hstack([[0], n_support]))
    for class1 in range(n_class):
        # 获取类别1的支持向量
        sv1 = support_vectors[sv_locs[class1] : sv_locs[class1 + 1], :]
        for class2 in range(class1 + 1, n_class):
            # 获取类别2的支持向量
            sv2 = support_vectors[sv_locs[class2] : sv_locs[class2 + 1], :]

            # 类别1的双重系数
            alpha1 = dual_coef[class2 - 1, sv_locs[class1] : sv_locs[class1 + 1]]
            # 类别2的双重系数
            alpha2 = dual_coef[class1, sv_locs[class2] : sv_locs[class2 + 1]]
            # 构建类别1对类别2的权重
            coef.append(safe_sparse_dot(alpha1, sv1) + safe_sparse_dot(alpha2, sv2))
    return coef


class BaseLibSVM(BaseEstimator, metaclass=ABCMeta):
    """Base class for estimators that use libsvm as backing library.

    This implements support vector machine classification and regression.

    Parameter documentation is in the derived `SVC` class.
    """
    # 定义参数约束的字典，每个参数对应一组类型或取值范围
    _parameter_constraints: dict = {
        "kernel": [
            StrOptions({"linear", "poly", "rbf", "sigmoid", "precomputed"}),  # kernel参数可选的字符串取值集合
            callable,  # kernel参数可以是可调用对象
        ],
        "degree": [Interval(Integral, 0, None, closed="left")],  # degree参数必须是大于等于0的整数
        "gamma": [
            StrOptions({"scale", "auto"}),  # gamma参数可选的字符串取值集合
            Interval(Real, 0.0, None, closed="left"),  # gamma参数必须是大于等于0.0的实数
        ],
        "coef0": [Interval(Real, None, None, closed="neither")],  # coef0参数没有具体的范围限制
        "tol": [Interval(Real, 0.0, None, closed="neither")],  # tol参数必须是大于等于0.0的实数
        "C": [Interval(Real, 0.0, None, closed="neither")],  # C参数必须是大于等于0.0的实数
        "nu": [Interval(Real, 0.0, 1.0, closed="right")],  # nu参数必须是0.0到1.0之间的实数，包括0.0不包括1.0
        "epsilon": [Interval(Real, 0.0, None, closed="left")],  # epsilon参数必须是大于等于0.0的实数
        "shrinking": ["boolean"],  # shrinking参数必须是布尔类型
        "probability": ["boolean"],  # probability参数必须是布尔类型
        "cache_size": [Interval(Real, 0, None, closed="neither")],  # cache_size参数必须是大于等于0的实数
        "class_weight": [StrOptions({"balanced"}), dict, None],  # class_weight参数可选的字符串取值集合、字典类型或None
        "verbose": ["verbose"],  # verbose参数必须是verbose类型
        "max_iter": [Interval(Integral, -1, None, closed="left")],  # max_iter参数必须是大于等于-1的整数
        "random_state": ["random_state"],  # random_state参数必须是random_state类型
    }

    # 这些字符串数组的顺序必须与LibSVM中的整数值顺序相匹配。
    # XXX 在稠密情况下这些实际上是相同的。需要将其因素化出来。
    _sparse_kernels = ["linear", "poly", "rbf", "sigmoid", "precomputed"]

    @abstractmethod
    def __init__(
        self,
        kernel,
        degree,
        gamma,
        coef0,
        tol,
        C,
        nu,
        epsilon,
        shrinking,
        probability,
        cache_size,
        class_weight,
        verbose,
        max_iter,
        random_state,
    ):
        # 如果self._impl不在LIBSVM_IMPL中，则引发值错误异常
        if self._impl not in LIBSVM_IMPL:
            raise ValueError(
                "impl should be one of %s, %s was given" % (LIBSVM_IMPL, self._impl)
            )

        # 初始化对象的属性
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.nu = nu
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.probability = probability
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.random_state = random_state

    def _more_tags(self):
        # 用于cross_val_score的标记函数
        return {"pairwise": self.kernel == "precomputed"}

    @_fit_context(prefer_skip_nested_validation=True)
    def _validate_targets(self, y):
        """验证y和class_weight。

        SVR和one-class的默认实现；在BaseSVC中被覆盖。
        """
        # 将y转换为一列或1维数组，并转换为np.float64类型，不复制数据
        return column_or_1d(y, warn=True).astype(np.float64, copy=False)

    def _warn_from_fit_status(self):
        assert self.fit_status_ in (0, 1)
        # 如果fit_status_为1，则发出警告，提示solver提前终止
        if self.fit_status_ == 1:
            warnings.warn(
                "Solver terminated early (max_iter=%i)."
                "  Consider pre-processing your data with"
                " StandardScaler or MinMaxScaler." % self.max_iter,
                ConvergenceWarning,
            )
    # 定义一个私有方法 _dense_fit，用于训练支持向量机模型
    def _dense_fit(self, X, y, sample_weight, solver_type, kernel, random_seed):
        # 如果 kernel 是可调用对象，则需存储 X 的引用以在预测时计算核函数
        # TODO: 添加关键字参数 copy，实现按需复制 X
        self.__Xfit = X
        # 计算核矩阵，并将其赋给 X
        X = self._compute_kernel(X)

        # 检查核矩阵 X 的形状是否为方阵
        if X.shape[0] != X.shape[1]:
            raise ValueError("X.shape[0] should be equal to X.shape[1]")

        # 设置 libsvm 库的详细程度
        libsvm.set_verbosity_wrap(self.verbose)

        # 调用 libsvm 的 fit 函数进行模型训练，并返回多个训练结果参数
        (
            self.support_,
            self.support_vectors_,
            self._n_support,
            self.dual_coef_,
            self.intercept_,
            self._probA,
            self._probB,
            self.fit_status_,
            self._num_iter,
        ) = libsvm.fit(
            X,
            y,
            svm_type=solver_type,
            sample_weight=sample_weight,
            class_weight=getattr(self, "class_weight_", np.empty(0)),
            kernel=kernel,
            C=self.C,
            nu=self.nu,
            probability=self.probability,
            degree=self.degree,
            shrinking=self.shrinking,
            tol=self.tol,
            cache_size=self.cache_size,
            coef0=self.coef0,
            gamma=self._gamma,
            epsilon=self.epsilon,
            max_iter=self.max_iter,
            random_seed=random_seed,
        )

        # 检查并警告从 fit 过程中得到的警告信息
        self._warn_from_fit_status()
    def _sparse_fit(self, X, y, sample_weight, solver_type, kernel, random_seed):
        # 将稀疏矩阵的数据部分转换为 np.float64 类型的数组，按 C 的顺序存储
        X.data = np.asarray(X.data, dtype=np.float64, order="C")
        # 对稀疏矩阵的列索引进行排序

        kernel_type = self._sparse_kernels.index(kernel)

        # 设置 LibSVM 的详细输出级别
        libsvm_sparse.set_verbosity_wrap(self.verbose)

        (
            self.support_,
            self.support_vectors_,
            dual_coef_data,
            self.intercept_,
            self._n_support,
            self._probA,
            self._probB,
            self.fit_status_,
            self._num_iter,
        ) = libsvm_sparse.libsvm_sparse_train(
            X.shape[1],
            X.data,
            X.indices,
            X.indptr,
            y,
            solver_type,
            kernel_type,
            self.degree,
            self._gamma,
            self.coef0,
            self.tol,
            self.C,
            getattr(self, "class_weight_", np.empty(0)),
            sample_weight,
            self.nu,
            self.cache_size,
            self.epsilon,
            int(self.shrinking),
            int(self.probability),
            self.max_iter,
            random_seed,
        )

        # 根据拟合状态进行警告处理
        self._warn_from_fit_status()

        if hasattr(self, "classes_"):
            # 如果模型具有类别属性，确定类别的数量（用于分类）
            n_class = len(self.classes_) - 1
        else:  # 如果是回归模型
            n_class = 1
        n_SV = self.support_vectors_.shape[0]

        # 创建双重系数的索引数组
        dual_coef_indices = np.tile(np.arange(n_SV), n_class)
        if not n_SV:
            # 如果支持向量数为零，创建空的 CSR 矩阵
            self.dual_coef_ = sp.csr_matrix([])
        else:
            # 计算双重系数索引的指针数组
            dual_coef_indptr = np.arange(
                0, dual_coef_indices.size + 1, dual_coef_indices.size / n_class
            )
            # 根据数据、索引和指针创建 CSR 矩阵
            self.dual_coef_ = sp.csr_matrix(
                (dual_coef_data, dual_coef_indices, dual_coef_indptr), (n_class, n_SV)
            )

    def predict(self, X):
        """Perform regression on samples in X.

        For an one-class model, +1 (inlier) or -1 (outlier) is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted values.
        """
        # 验证 X 是否适用于预测，并根据稀疏标志选择适当的预测方法
        X = self._validate_for_predict(X)
        predict = self._sparse_predict if self._sparse else self._dense_predict
        return predict(X)
    def _dense_predict(self, X):
        # 计算核函数变换后的数据
        X = self._compute_kernel(X)
        
        # 如果数据是一维的，将其转换为二维数组
        if X.ndim == 1:
            X = check_array(X, order="C", accept_large_sparse=False)

        # 获取当前的核函数
        kernel = self.kernel
        
        # 如果核函数是一个可调用的函数，将核函数类型设置为"precomputed"
        if callable(self.kernel):
            kernel = "precomputed"
            # 检查输入数据的列数是否与训练时的样本数相匹配
            if X.shape[1] != self.shape_fit_[0]:
                raise ValueError(
                    "X.shape[1] = %d should be equal to %d, "
                    "the number of samples at training time"
                    % (X.shape[1], self.shape_fit_[0])
                )

        # 获取当前的 SVM 类型
        svm_type = LIBSVM_IMPL.index(self._impl)

        # 使用 libsvm 库进行预测
        return libsvm.predict(
            X,
            self.support_,
            self.support_vectors_,
            self._n_support,
            self._dual_coef_,
            self._intercept_,
            self._probA,
            self._probB,
            svm_type=svm_type,
            kernel=kernel,
            degree=self.degree,
            coef0=self.coef0,
            gamma=self._gamma,
            cache_size=self.cache_size,
        )

    def _sparse_predict(self, X):
        # 前提条件：X 是 dtype 为 np.float64 的 csr_matrix
        # 获取当前的核函数
        kernel = self.kernel
        
        # 如果核函数是一个可调用的函数，将核函数类型设置为"precomputed"
        if callable(kernel):
            kernel = "precomputed"

        # 确定稀疏矩阵的核函数类型
        kernel_type = self._sparse_kernels.index(kernel)

        # 设置 C 的值，这里 C 并不影响计算结果
        C = 0.0

        # 使用 libsvm_sparse 库进行稀疏数据的预测
        return libsvm_sparse.libsvm_sparse_predict(
            X.data,
            X.indices,
            X.indptr,
            self.support_vectors_.data,
            self.support_vectors_.indices,
            self.support_vectors_.indptr,
            self._dual_coef_.data,
            self._intercept_,
            LIBSVM_IMPL.index(self._impl),
            kernel_type,
            self.degree,
            self._gamma,
            self.coef0,
            self.tol,
            C,
            getattr(self, "class_weight_", np.empty(0)),
            self.nu,
            self.epsilon,
            self.shrinking,
            self.probability,
            self._n_support,
            self._probA,
            self._probB,
        )

    def _compute_kernel(self, X):
        """Return the data transformed by a callable kernel"""
        # 如果核函数是可调用的，则计算核函数变换后的数据
        if callable(self.kernel):
            # 对于给定的预计算核函数作为函数的情况，我们必须显式计算核矩阵
            kernel = self.kernel(X, self.__Xfit)
            # 如果核函数是稀疏矩阵，则转换为稠密矩阵
            if sp.issparse(kernel):
                kernel = kernel.toarray()
            # 将结果转换为 np.float64 类型的二维数组
            X = np.asarray(kernel, dtype=np.float64, order="C")
        return X
    def _decision_function(self, X):
        """Evaluates the decision function for the samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X : array-like of shape (n_samples, n_class * (n_class-1) / 2)
            Returns the decision function of the sample for each class
            in the model.
        """
        # NOTE: _validate_for_predict contains check for is_fitted
        # hence must be placed before any other attributes are used.
        # 调用_validate_for_predict方法，确保模型已经拟合并验证输入数据X的合法性
        X = self._validate_for_predict(X)
        # 根据输入数据X计算核函数（如果有的话）
        X = self._compute_kernel(X)

        # 根据模型是否稀疏选择不同的决策函数计算方式
        if self._sparse:
            # 如果模型是稀疏的，则调用稀疏决策函数计算dec_func
            dec_func = self._sparse_decision_function(X)
        else:
            # 否则调用密集决策函数计算dec_func
            dec_func = self._dense_decision_function(X)

        # 对于二元分类情况，需要对coef（系数）、intercept（截距）和
        # decision function的符号进行反转
        if self._impl in ["c_svc", "nu_svc"] and len(self.classes_) == 2:
            return -dec_func.ravel()

        return dec_func

    def _dense_decision_function(self, X):
        # 将输入数据X转换为指定的数据类型，并进行合法性检查
        X = check_array(X, dtype=np.float64, order="C", accept_large_sparse=False)

        kernel = self.kernel
        if callable(kernel):
            kernel = "precomputed"

        # 调用libsvm库中的decision_function计算密集模式下的决策函数
        return libsvm.decision_function(
            X,
            self.support_,
            self.support_vectors_,
            self._n_support,
            self._dual_coef_,
            self._intercept_,
            self._probA,
            self._probB,
            svm_type=LIBSVM_IMPL.index(self._impl),
            kernel=kernel,
            degree=self.degree,
            cache_size=self.cache_size,
            coef0=self.coef0,
            gamma=self._gamma,
        )

    def _sparse_decision_function(self, X):
        # 将稀疏矩阵X的数据部分转换为指定的数据类型，并进行合法性检查
        X.data = np.asarray(X.data, dtype=np.float64, order="C")

        kernel = self.kernel
        if hasattr(kernel, "__call__"):
            kernel = "precomputed"

        # 获取稀疏核的索引
        kernel_type = self._sparse_kernels.index(kernel)

        # 调用libsvm_sparse库中的libsvm_sparse_decision_function计算稀疏模式下的决策函数
        return libsvm_sparse.libsvm_sparse_decision_function(
            X.data,
            X.indices,
            X.indptr,
            self.support_vectors_.data,
            self.support_vectors_.indices,
            self.support_vectors_.indptr,
            self._dual_coef_.data,
            self._intercept_,
            LIBSVM_IMPL.index(self._impl),
            kernel_type,
            self.degree,
            self._gamma,
            self.coef0,
            self.tol,
            self.C,
            getattr(self, "class_weight_", np.empty(0)),
            self.nu,
            self.epsilon,
            self.shrinking,
            self.probability,
            self._n_support,
            self._probA,
            self._probB,
        )
    # 验证用于预测的输入数据 X 是否符合要求
    def _validate_for_predict(self, X):
        # 检查模型是否已经拟合，即是否已经训练过
        check_is_fitted(self)

        # 如果 kernel 不是可调用对象，则验证数据 X 是否合规并返回 X
        if not callable(self.kernel):
            X = self._validate_data(
                X,
                accept_sparse="csr",  # 接受稀疏矩阵格式
                dtype=np.float64,     # 数据类型为 float64
                order="C",            # C 风格的数组存储顺序
                accept_large_sparse=False,  # 不接受大稀疏矩阵
                reset=False,          # 不重置数据
            )

        # 如果模型使用稀疏数据格式并且 X 不是稀疏矩阵，则将 X 转换为 csr_matrix 格式
        if self._sparse and not sp.issparse(X):
            X = sp.csr_matrix(X)
        
        # 如果模型使用稀疏数据格式，则对 X 进行排序
        if self._sparse:
            X.sort_indices()

        # 如果 X 是稀疏矩阵并且模型不是使用稀疏数据格式，并且 kernel 不是可调用对象，则抛出错误
        if sp.issparse(X) and not self._sparse and not callable(self.kernel):
            raise ValueError(
                "cannot use sparse input in %r trained on dense data"
                % type(self).__name__
            )

        # 如果 kernel 是 "precomputed"，则检查 X 的列数是否等于模型拟合时的样本数
        if self.kernel == "precomputed":
            if X.shape[1] != self.shape_fit_[0]:
                raise ValueError(
                    "X.shape[1] = %d should be equal to %d, "
                    "the number of samples at training time"
                    % (X.shape[1], self.shape_fit_[0])
                )

        # 修复 https://nvd.nist.gov/vuln/detail/CVE-2020-28975
        # 检查 _n_support 是否与支持向量 support_vectors_ 一致
        sv = self.support_vectors_
        if not self._sparse and sv.size > 0 and self.n_support_.sum() != sv.shape[0]:
            raise ValueError(
                f"The internal representation of {self.__class__.__name__} was altered"
            )

        # 返回经验证的输入数据 X
        return X

    @property
    def coef_(self):
        """当 kernel="linear" 时，返回特征的权重。

        Returns
        -------
        ndarray of shape (n_features, n_classes)
        """
        # 如果 kernel 不是 "linear"，则抛出 AttributeError
        if self.kernel != "linear":
            raise AttributeError("coef_ is only available when using a linear kernel")

        # 获取特征权重
        coef = self._get_coef()

        # 由于 coef_ 是只读属性，最好将其标记为不可变，以避免潜在的 bug
        if sp.issparse(coef):
            # 对于稀疏矩阵，设置其数据为不可写入
            coef.data.flags.writeable = False
        else:
            # 对于常规的稠密数组，设置其 flags 为不可写入
            coef.flags.writeable = False

        # 返回特征权重 coef
        return coef

    # 获取特征权重的内部方法
    def _get_coef(self):
        return safe_sparse_dot(self._dual_coef_, self.support_vectors_)

    @property
    def n_support_(self):
        """每个类别的支持向量数量。"""
        try:
            # 检查模型是否已经拟合，如果没有拟合则抛出 NotFittedError
            check_is_fitted(self)
        except NotFittedError:
            raise AttributeError

        # 获取 LIBSVM_IMPL 中实现的 SVM 类型
        svm_type = LIBSVM_IMPL.index(self._impl)
        if svm_type in (0, 1):
            # 对于 SVC 和 NuSVC 类型的 SVM，返回 _n_support
            return self._n_support
        else:
            # 对于 SVR 和 OneClass 类型的 SVM，_n_support 的大小为 2，这里将其调整为大小为 1 的数组返回
            return np.array([self._n_support[0]])
class BaseSVC(ClassifierMixin, BaseLibSVM, metaclass=ABCMeta):
    """ABC for LibSVM-based classifiers."""

    # 定义参数约束字典，继承自BaseLibSVM的约束，添加了decision_function_shape和break_ties参数的约束
    _parameter_constraints: dict = {
        **BaseLibSVM._parameter_constraints,
        "decision_function_shape": [StrOptions({"ovr", "ovo"})],
        "break_ties": ["boolean"],
    }

    # 删除不使用的参数epsilon和nu的约束
    for unused_param in ["epsilon", "nu"]:
        _parameter_constraints.pop(unused_param)

    @abstractmethod
    def __init__(
        self,
        kernel,
        degree,
        gamma,
        coef0,
        tol,
        C,
        nu,
        shrinking,
        probability,
        cache_size,
        class_weight,
        verbose,
        max_iter,
        decision_function_shape,
        random_state,
        break_ties,
    ):
        # 设置决策函数形状和break_ties参数的值
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        # 调用父类BaseLibSVM的初始化方法，传入各种参数
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            nu=nu,
            epsilon=0.0,  # 设定epsilon为0.0
            shrinking=shrinking,
            probability=probability,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            random_state=random_state,
        )

    def _validate_targets(self, y):
        # 将y变成列向量或者1维向量，并发出警告
        y_ = column_or_1d(y, warn=True)
        # 检查分类目标y
        check_classification_targets(y)
        # 将y_转换为唯一的类别和反转索引
        cls, y = np.unique(y_, return_inverse=True)
        # 计算类别权重
        self.class_weight_ = compute_class_weight(self.class_weight, classes=cls, y=y_)
        # 如果类别数少于2，抛出错误
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % len(cls)
            )

        # 设置self.classes_为类别
        self.classes_ = cls

        # 返回y的浮点数形式的数组，按照C顺序
        return np.asarray(y, dtype=np.float64, order="C")
    def decision_function(self, X):
        """Evaluate the decision function for the samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X : ndarray of shape (n_samples, n_classes * (n_classes-1) / 2)
            Returns the decision function of the sample for each class
            in the model.
            If decision_function_shape='ovr', the shape is (n_samples,
            n_classes).

        Notes
        -----
        If decision_function_shape='ovo', the function values are proportional
        to the distance of the samples X to the separating hyperplane. If the
        exact distances are required, divide the function values by the norm of
        the weight vector (``coef_``). See also `this question
        <https://stats.stackexchange.com/questions/14876/
        interpreting-distance-from-hyperplane-in-svm>`_ for further details.
        If decision_function_shape='ovr', the decision function is a monotonic
        transformation of ovo decision function.
        """
        # Compute the decision function values for samples X
        dec = self._decision_function(X)
        # Implement decision function shape 'ovr' for multiclass classification
        if self.decision_function_shape == "ovr" and len(self.classes_) > 2:
            return _ovr_decision_function(dec < 0, -dec, len(self.classes_))
        return dec

    def predict(self, X):
        """Perform classification on samples in X.

        For an one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples_test, n_samples_train)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for samples in X.
        """
        # Ensure the estimator is fitted
        check_is_fitted(self)
        # Check if break_ties is improperly set for ovo shape
        if self.break_ties and self.decision_function_shape == "ovo":
            raise ValueError(
                "break_ties must be False when decision_function_shape is 'ovo'"
            )

        # Handle break_ties and decision_function_shape 'ovr' for multiclass
        if (
            self.break_ties
            and self.decision_function_shape == "ovr"
            and len(self.classes_) > 2
        ):
            y = np.argmax(self.decision_function(X), axis=1)
        else:
            # Use superclass's predict method for prediction
            y = super().predict(X)
        # Return predicted class labels converted to appropriate data type
        return self.classes_.take(np.asarray(y, dtype=np.intp))

    # Hacky way of getting predict_proba to raise an AttributeError when
    # probability=False using properties. Do not use this in new code; when
    # probabilities are not available depending on a setting, introduce two
    # estimators.
    def _check_proba(self):
        # 检查是否允许使用概率预测
        if not self.probability:
            # 如果 probability=False，抛出属性错误异常
            raise AttributeError(
                "predict_proba is not available when probability=False"
            )
        # 检查概率预测是否仅适用于 SVC 和 NuSVC
        if self._impl not in ("c_svc", "nu_svc"):
            raise AttributeError("predict_proba only implemented for SVC and NuSVC")
        # 返回 True 表示检查通过
        return True

    @available_if(_check_proba)
    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        The model needs to have probability information computed at training
        time: fit with attribute `probability` set to True.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        T : ndarray of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.

        Notes
        -----
        The probability model is created using cross validation, so
        the results can be slightly different than those obtained by
        predict. Also, it will produce meaningless results on very small
        datasets.
        """
        # 验证输入数据 X 是否符合预测要求
        X = self._validate_for_predict(X)
        # 如果没有计算好的概率信息，则抛出未拟合错误异常
        if self.probA_.size == 0 or self.probB_.size == 0:
            raise NotFittedError(
                "predict_proba is not available when fitted with probability=False"
            )
        # 根据数据稀疏性选择合适的预测概率函数
        pred_proba = (
            self._sparse_predict_proba if self._sparse else self._dense_predict_proba
        )
        # 返回预测的概率结果
        return pred_proba(X)

    @available_if(_check_proba)
    def predict_log_proba(self, X):
        """Compute log probabilities of possible outcomes for samples in X.

        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples_test, n_samples_train)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        T : ndarray of shape (n_samples, n_classes)
            Returns the log-probabilities of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.

        Notes
        -----
        The probability model is created using cross validation, so
        the results can be slightly different than those obtained by
        predict. Also, it will produce meaningless results on very small
        datasets.
        """
        # 返回预测的对数概率结果
        return np.log(self.predict_proba(X))
    # 使用输入数据 X 计算核矩阵
    X = self._compute_kernel(X)

    # 获取当前的核函数类型，如果核函数是可调用对象，则设为 "precomputed"
    kernel = self.kernel
    if callable(kernel):
        kernel = "precomputed"

    # 确定 SVM 类型对应的索引，根据当前实现方式选择对应的预测概率函数
    svm_type = LIBSVM_IMPL.index(self._impl)
    
    # 调用 libsvm 库的 predict_proba 函数，进行概率预测
    pprob = libsvm.predict_proba(
        X,
        self.support_,
        self.support_vectors_,
        self._n_support,
        self._dual_coef_,
        self._intercept_,
        self._probA,
        self._probB,
        svm_type=svm_type,
        kernel=kernel,
        degree=self.degree,
        cache_size=self.cache_size,
        coef0=self.coef0,
        gamma=self._gamma,
    )

    # 返回预测的概率
    return pprob


```    
    # 将稀疏矩阵 X 的数据部分转换为 np.float64 类型的数组，按 C 风格排序
    X.data = np.asarray(X.data, dtype=np.float64, order="C")

    # 获取当前的核函数类型，如果核函数是可调用对象，则设为 "precomputed"
    kernel = self.kernel
    if callable(kernel):
        kernel = "precomputed"

    # 确定稀疏数据情况下使用的核函数类型索引
    kernel_type = self._sparse_kernels.index(kernel)

    # 调用 libsvm_sparse 库的 libsvm_sparse_predict_proba 函数，进行稀疏数据的概率预测
    return libsvm_sparse.libsvm_sparse_predict_proba(
        X.data,
        X.indices,
        X.indptr,
        self.support_vectors_.data,
        self.support_vectors_.indices,
        self.support_vectors_.indptr,
        self._dual_coef_.data,
        self._intercept_,
        LIBSVM_IMPL.index(self._impl),
        kernel_type,
        self.degree,
        self._gamma,
        self.coef0,
        self.tol,
        self.C,
        getattr(self, "class_weight_", np.empty(0)),
        self.nu,
        self.epsilon,
        self.shrinking,
        self.probability,
        self._n_support,
        self._probA,
        self._probB,
    )



    # 根据分类器类型获取系数（coef）
    if self.dual_coef_.shape[0] == 1:
        # 二元分类器情况下
        coef = safe_sparse_dot(self.dual_coef_, self.support_vectors_)
    else:
        # 一对一分类器情况下，计算系数
        coef = _one_vs_one_coef(
            self.dual_coef_, self._n_support, self.support_vectors_
        )
        # 如果系数是稀疏矩阵，则转换为 CSR 格式
        if sp.issparse(coef[0]):
            coef = sp.vstack(coef).tocsr()
        else:
            coef = np.vstack(coef)

    # 返回计算得到的系数（coef）
    return coef



    @property
    def probA_(self):
        """在 `probability=True` 时学习到的 Platt 缩放参数 probA 的属性方法。

        返回
        -------
        形状为 (n_classes * (n_classes - 1) / 2) 的 ndarray
        """
        return self._probA



    @property
    def probB_(self):
        """在 `probability=True` 时学习到的 Platt 缩放参数 probB 的属性方法。

        返回
        -------
        形状为 (n_classes * (n_classes - 1) / 2) 的 ndarray
        """
        return self._probB
# 寻找适合 liblinear 解算器的 magic number。

def _get_liblinear_solver_type(multi_class, penalty, loss, dual):
    """Find the liblinear magic number for the solver.

    This number depends on the values of the following attributes:
      - multi_class
      - penalty
      - loss
      - dual

    The same number is also internally used by LibLinear to determine
    which solver to use.
    """

    # 嵌套字典，包含一级：可用的损失函数，
    # 二级：给定损失函数的可用惩罚方式，
    # 三级：指定损失函数和惩罚方式时是否可用双重求解器
    _solver_type_dict = {
        "logistic_regression": {"l1": {False: 6}, "l2": {False: 0, True: 7}},
        "hinge": {"l2": {True: 3}},
        "squared_hinge": {"l1": {False: 5}, "l2": {False: 2, True: 1}},
        "epsilon_insensitive": {"l2": {True: 13}},
        "squared_epsilon_insensitive": {"l2": {False: 11, True: 12}},
        "crammer_singer": 4,
    }

    if multi_class == "crammer_singer":
        # 如果 multi_class 是 crammer_singer，则直接返回相应的解算器类型
        return _solver_type_dict[multi_class]
    elif multi_class != "ovr":
        # 如果 multi_class 不是 ovr，抛出数值错误
        raise ValueError(
            "`multi_class` must be one of `ovr`, `crammer_singer`, got %r" % multi_class
        )

    _solver_pen = _solver_type_dict.get(loss, None)
    if _solver_pen is None:
        # 如果 loss 不在支持列表中，生成错误信息
        error_string = "loss='%s' is not supported" % loss
    else:
        _solver_dual = _solver_pen.get(penalty, None)
        if _solver_dual is None:
            # 如果 penalty 和 loss 组合不支持，生成错误信息
            error_string = (
                "The combination of penalty='%s' and loss='%s' is not supported"
                % (penalty, loss)
            )
        else:
            solver_num = _solver_dual.get(dual, None)
            if solver_num is None:
                # 如果 dual 不支持，生成错误信息
                error_string = (
                    "The combination of penalty='%s' and "
                    "loss='%s' are not supported when dual=%s" % (penalty, loss, dual)
                )
            else:
                # 返回找到的解算器类型的 magic number
                return solver_num

    # 如果以上条件都不满足，抛出数值错误，显示参数不支持的详细信息
    raise ValueError(
        "Unsupported set of arguments: %s, Parameters: penalty=%r, loss=%r, dual=%r"
        % (error_string, penalty, loss, dual)
    )


def _fit_liblinear(
    X,
    y,
    C,
    fit_intercept,
    intercept_scaling,
    class_weight,
    penalty,
    dual,
    verbose,
    max_iter,
    tol,
    random_state=None,
    multi_class="ovr",
    loss="logistic_regression",
    epsilon=0.1,
    sample_weight=None,
):
    """Used by Logistic Regression (and CV) and LinearSVC/LinearSVR.

    Preprocessing is done in this function before supplying it to liblinear.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples,)
        Target vector relative to X

    C : float
        Inverse of cross-validation parameter. The lower the C, the higher
        the penalization.
    fit_intercept : bool
        是否拟合截距。如果设置为True，则特征向量会扩展以包括一个截距项：``[x_1, ..., x_n, 1]``，其中1对应于截距。如果设置为False，则计算中不使用截距（即数据预期已经中心化）。

    intercept_scaling : float
        Liblinear内部对截距进行惩罚，将其视为特征向量中的任何其他项。为了减少正则化对截距的影响，可以将`intercept_scaling`参数设置为大于1的值；`intercept_scaling`值越高，正则化对其的影响越小。
        然后，权重变成 `[w_x_1, ..., w_x_n, w_intercept*intercept_scaling]`，其中 `w_x_1, ..., w_x_n` 表示特征权重，而截距权重被 `intercept_scaling` 缩放。这种缩放允许截距项具有与其他特征不同的正则化行为。

    class_weight : dict or 'balanced', default=None
        类别权重，形式为 ``{class_label: weight}``。如果未提供，则假定所有类别权重均为1。对于多输出问题，可以按照y的列顺序提供一系列字典。
        "balanced" 模式使用y的值自动调整权重，与输入数据中类别频率成反比，计算方式为 ``n_samples / (n_classes * np.bincount(y))``。

    penalty : {'l1', 'l2'}
        正则化中使用的惩罚范数。

    dual : bool
        对偶或原始形式，

    verbose : int
        设置verbose为任何正数以增加详细程度。

    max_iter : int
        迭代次数。

    tol : float
        停止条件。

    random_state : int, RandomState instance or None, default=None
        控制用于对数据进行洗牌的伪随机数生成。传递一个整数以在多次函数调用中获得可重复的输出。
        参见 :term:`术语表 <random_state>`。

    multi_class : {'ovr', 'crammer_singer'}, default='ovr'
        `ovr` 训练n_classes个一对多分类器，而 `crammer_singer` 优化所有类别的联合目标。
        尽管 `crammer_singer` 从理论上讲很有趣，因为它是一致的，但在实践中很少使用，并且很少导致更好的准确性，计算成本更高。
        如果选择 `crammer_singer`，将忽略选项loss、penalty和dual。

    loss : {'logistic_regression', 'hinge', 'squared_hinge', \
            'epsilon_insensitive', 'squared_epsilon_insensitive}, \
            default='logistic_regression'
        用于拟合模型的损失函数。
    epsilon : float, default=0.1
        Epsilon参数在epsilon-insensitive损失函数中的应用。注意，该参数的值取决于目标变量y的尺度。如果不确定，请将epsilon设置为0。

    sample_weight : array-like of shape (n_samples,), default=None
        每个样本分配的权重。

    Returns
    -------
    coef_ : ndarray of shape (n_features, n_features + 1)
        通过最小化目标函数得到的系数向量。

    intercept_ : float
        添加到向量中的截距项。

    n_iter_ : array of int
        每个类别运行的迭代次数。

    """
    # 如果损失函数不是 "epsilon_insensitive" 或 "squared_epsilon_insensitive"，则执行以下逻辑
    if loss not in ["epsilon_insensitive", "squared_epsilon_insensitive"]:
        # 使用LabelEncoder对目标变量y进行编码
        enc = LabelEncoder()
        y_ind = enc.fit_transform(y)
        # 获取编码后的类别
        classes_ = enc.classes_
        # 如果类别数量小于2，则抛出异常
        if len(classes_) < 2:
            raise ValueError(
                "This solver needs samples of at least 2 classes"
                " in the data, but the data contains only one"
                " class: %r" % classes_[0]
            )
        # 根据类别和class_weight计算class_weight_
        class_weight_ = compute_class_weight(class_weight, classes=classes_, y=y)
    else:
        # 如果损失函数是 "epsilon_insensitive" 或 "squared_epsilon_insensitive"，则class_weight_为空数组
        class_weight_ = np.empty(0, dtype=np.float64)
        y_ind = y

    # 设置LibLinear的详细程度
    liblinear.set_verbosity_wrap(verbose)
    # 检查随机状态，确保一致性
    rnd = check_random_state(random_state)
    # 如果verbose为True，则打印LibLinear的信息
    if verbose:
        print("[LibLinear]", end="")

    # 初始化bias为-1.0
    bias = -1.0
    # 如果fit_intercept为True
    if fit_intercept:
        # 检查intercept_scaling是否小于等于0，若是，则抛出异常
        if intercept_scaling <= 0:
            raise ValueError(
                "Intercept scaling is %r but needs to be greater "
                "than 0. To disable fitting an intercept,"
                " set fit_intercept=False." % intercept_scaling
            )
        else:
            bias = intercept_scaling

    # 设置LibSVM的详细程度
    libsvm.set_verbosity_wrap(verbose)
    libsvm_sparse.set_verbosity_wrap(verbose)
    liblinear.set_verbosity_wrap(verbose)

    # 如果X是稀疏矩阵，检查其是否过大
    if sp.issparse(X):
        _check_large_sparse(X)

    # 将目标变量y_ind转换为浮点数数组，并将其拉平
    y_ind = np.asarray(y_ind, dtype=np.float64).ravel()
    # 要求y_ind满足特定的要求
    y_ind = np.require(y_ind, requirements="W")

    # 检查并设置样本权重sample_weight，要求其为浮点数类型
    sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)

    # 根据multi_class、penalty、loss、dual获取LibLinear的solver类型
    solver_type = _get_liblinear_solver_type(multi_class, penalty, loss, dual)
    # 调用LibLinear的训练函数，返回raw_coef_和n_iter_
    raw_coef_, n_iter_ = liblinear.train_wrap(
        X,
        y_ind,
        sp.issparse(X),
        solver_type,
        tol,
        bias,
        C,
        class_weight_,
        max_iter,
        rnd.randint(np.iinfo("i").max),
        epsilon,
        sample_weight,
    )
    # 上述签名中的rnd.randint(..)：
    # 用于设置在[0..INT_MAX]范围内的srand种子；由于Numpy在32位平台上的限制，
    # 无法达到srand支持的UINT_MAX上限。

    # 获取n_iter_的最大值
    n_iter_max = max(n_iter_)
    # 如果达到最大迭代次数 n_iter_max 大于等于设定的最大迭代次数 max_iter，则发出警告
    if n_iter_max >= max_iter:
        # 发出警告信息，提示用户增加迭代次数以提高收敛性
        warnings.warn(
            "Liblinear failed to converge, increase the number of iterations.",
            ConvergenceWarning,
        )
    
    # 如果设置了拟合截距 fit_intercept 为 True，则调整系数矩阵 coef_ 和截距 intercept_
    if fit_intercept:
        # 提取出系数矩阵，不包括最后一列（截距项）
        coef_ = raw_coef_[:, :-1]
        # 计算截距，使用截距缩放因子 intercept_scaling 乘以最后一列系数
        intercept_ = intercept_scaling * raw_coef_[:, -1]
    else:
        # 否则直接使用原始的系数矩阵
        coef_ = raw_coef_
        # 截距设置为 0.0
        intercept_ = 0.0
    
    # 返回计算得到的系数矩阵 coef_，截距 intercept_，以及迭代次数 n_iter_
    return coef_, intercept_, n_iter_
```
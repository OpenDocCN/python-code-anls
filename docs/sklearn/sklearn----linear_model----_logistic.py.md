# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_logistic.py`

```
"""
Logistic Regression
"""

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Fabian Pedregosa <f@bianp.net>
#         Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Manoj Kumar <manojkumarsivaraj334@gmail.com>
#         Lars Buitinck
#         Simon Wu <s8wu@uwaterloo.ca>
#         Arthur Mensch <arthur.mensch@m4x.org

import numbers  # 导入 numbers 模块，用于数值类型检查
import warnings  # 导入 warnings 模块，用于警告处理
from numbers import Integral, Real  # 从 numbers 模块导入 Integral 和 Real 类型

import numpy as np  # 导入 NumPy 库，用于数值计算
from joblib import effective_n_jobs  # 导入 joblib 库中的 effective_n_jobs 函数，用于获取有效的并行作业数
from scipy import optimize  # 导入 scipy 库中的 optimize 模块，用于优化算法

from sklearn.metrics import get_scorer_names  # 从 sklearn.metrics 模块导入 get_scorer_names 函数，用于获取评分器的名称

from .._loss.loss import HalfBinomialLoss, HalfMultinomialLoss  # 导入自定义 loss 模块中的 HalfBinomialLoss 和 HalfMultinomialLoss 类
from ..base import _fit_context  # 从父级目录中导入 _fit_context
from ..metrics import get_scorer  # 从 metrics 模块导入 get_scorer 函数，用于获取评分器
from ..model_selection import check_cv  # 从 model_selection 模块导入 check_cv 函数，用于交叉验证检查
from ..preprocessing import LabelBinarizer, LabelEncoder  # 从 preprocessing 模块导入 LabelBinarizer 和 LabelEncoder 类
from ..svm._base import _fit_liblinear  # 从 svm._base 模块导入 _fit_liblinear 函数，用于基于 liblinear 的 SVM 拟合
from ..utils import (  # 从 utils 模块导入多个函数和类
    Bunch,
    check_array,
    check_consistent_length,
    check_random_state,
    compute_class_weight,
)
from ..utils._param_validation import Hidden, Interval, StrOptions  # 从 utils._param_validation 模块导入 Hidden, Interval, StrOptions 类
from ..utils.extmath import row_norms, softmax  # 从 utils.extmath 模块导入 row_norms 和 softmax 函数
from ..utils.metadata_routing import (  # 从 utils.metadata_routing 模块导入多个函数和类
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from ..utils.multiclass import check_classification_targets  # 从 utils.multiclass 模块导入 check_classification_targets 函数
from ..utils.optimize import _check_optimize_result, _newton_cg  # 从 utils.optimize 模块导入 _check_optimize_result 和 _newton_cg 函数
from ..utils.parallel import Parallel, delayed  # 从 utils.parallel 模块导入 Parallel 和 delayed 函数
from ..utils.validation import (  # 从 utils.validation 模块导入多个函数
    _check_method_params,
    _check_sample_weight,
    check_is_fitted,
)
from ._base import BaseEstimator, LinearClassifierMixin, SparseCoefMixin  # 从当前目录的 _base 模块导入 BaseEstimator, LinearClassifierMixin, SparseCoefMixin 类
from ._glm.glm import NewtonCholeskySolver  # 从当前目录的 _glm.glm 模块导入 NewtonCholeskySolver 类
from ._linear_loss import LinearModelLoss  # 从当前目录的 _linear_loss 模块导入 LinearModelLoss 类
from ._sag import sag_solver  # 从当前目录的 _sag 模块导入 sag_solver 函数

_LOGISTIC_SOLVER_CONVERGENCE_MSG = (
    "Please also refer to the documentation for alternative solver options:\n"
    "    https://scikit-learn.org/stable/modules/linear_model.html"
    "#logistic-regression"
)


def _check_solver(solver, penalty, dual):
    if solver not in ["liblinear", "saga"] and penalty not in ("l2", None):
        raise ValueError(
            f"Solver {solver} supports only 'l2' or None penalties, got {penalty} "
            "penalty."
        )
    if solver != "liblinear" and dual:
        raise ValueError(f"Solver {solver} supports only dual=False, got dual={dual}")

    if penalty == "elasticnet" and solver != "saga":
        raise ValueError(
            f"Only 'saga' solver supports elasticnet penalty, got solver={solver}."
        )

    if solver == "liblinear" and penalty is None:
        raise ValueError("penalty=None is not supported for the liblinear solver")

    return solver


def _check_multi_class(multi_class, solver, n_classes):
    """Computes the multi class type, either "multinomial" or "ovr".

    For `n_classes` > 2 and a solver that supports it, returns "multinomial".
    For all other cases, in particular binary classification, return "ovr".
    """
    # 如果 multi_class 被设为 "auto"，根据 solver 和 n_classes 的值进行判断
    if multi_class == "auto":
        # 如果 solver 是 "liblinear" 或 "newton-cholesky"
        if solver in ("liblinear", "newton-cholesky"):
            # 设置 multi_class 为 "ovr"
            multi_class = "ovr"
        # 否则，如果类别数大于2
        elif n_classes > 2:
            # 设置 multi_class 为 "multinomial"
            multi_class = "multinomial"
        # 否则（类别数为1或2）
        else:
            # 设置 multi_class 为 "ovr"
            multi_class = "ovr"
    
    # 如果 multi_class 被设为 "multinomial" 并且 solver 是 "liblinear" 或 "newton-cholesky"
    if multi_class == "multinomial" and solver in ("liblinear", "newton-cholesky"):
        # 抛出 ValueError，提示 solver 不支持 multinomial 后端
        raise ValueError("Solver %s does not support a multinomial backend." % solver)
    
    # 返回最终确定的 multi_class 值
    return multi_class
# 定义一个函数，用于计算逻辑回归模型的路径，即不同正则化参数下的模型结果
def _logistic_regression_path(
    X,
    y,
    pos_class=None,
    Cs=10,
    fit_intercept=True,
    max_iter=100,
    tol=1e-4,
    verbose=0,
    solver="lbfgs",
    coef=None,
    class_weight=None,
    dual=False,
    penalty="l2",
    intercept_scaling=1.0,
    multi_class="auto",
    random_state=None,
    check_input=True,
    max_squared_sum=None,
    sample_weight=None,
    l1_ratio=None,
    n_threads=1,
):
    """Compute a Logistic Regression model for a list of regularization
    parameters.

    This is an implementation that uses the result of the previous model
    to speed up computations along the set of solutions, making it faster
    than sequentially calling LogisticRegression for the different parameters.
    Note that there will be no speedup with liblinear solver, since it does
    not handle warm-starting.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        输入数据。

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        输入数据，目标值。

    pos_class : int, default=None
        在进行一对多拟合时，指定的类别。如果为None，则假设问题是二元的。

    Cs : int or array-like of shape (n_cs,), default=10
        正则化参数的值列表，或者指定要使用的正则化参数数量。在这种情况下，参数将在对数尺度上
        在1e-4到1e4之间选择。

    fit_intercept : bool, default=True
        是否为模型拟合截距。在这种情况下，返回数组的形状是 (n_cs, n_features + 1)。

    max_iter : int, default=100
        求解器的最大迭代次数。

    tol : float, default=1e-4
        停止迭代的标准。对于 newton-cg 和 lbfgs 求解器，当 ``max{|g_i | i = 1, ..., n} <= tol``
        满足时迭代停止，其中 ``g_i`` 是梯度的第i个分量。

    verbose : int, default=0
        对于 liblinear 和 lbfgs 求解器，设置 verbose 为正整数以增加详细程度。

    solver : {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}, \
            default='lbfgs'
        数值求解器的选择。

    coef : array-like of shape (n_features,), default=None
        逻辑回归系数的初始化值。对于 liblinear 求解器无用。

    class_weight : dict or 'balanced', default=None
        类别权重的设置。如果不指定，则所有类别的权重都为1。

    dual : bool, default=False
        在求解 liblinear 求解器时是否使用对偶问题。

    penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
        正则化类型。

    intercept_scaling : float, default=1.0
        拦截缩放参数。

    multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
        多类别分类的策略。

    random_state : int, RandomState instance, default=None
        伪随机数生成器的种子或伪随机数生成器实例。

    check_input : bool, default=True
        是否检查输入的有效性。

    max_squared_sum : float, default=None
        最大的平方和。

    sample_weight : array-like of shape (n_samples,), default=None
        样本权重。

    l1_ratio : float, default=None
        L1正则化项在总惩罚中的比例。

    n_threads : int, default=1
        并行线程数。
    # 类别权重，可以是字典形式 `{类别标签: 权重}` 或者 'balanced'（平衡模式）
    # 如果未提供，则所有类别权重默认为1。
    class_weight : dict or 'balanced', default=None
        # "balanced" 模式根据输入数据中各类别的频率自动调整权重，权重与类别频率成反比，
        # 计算公式为 `n_samples / (n_classes * np.bincount(y))`。
        # 如果指定了 sample_weight（通过 fit 方法传递），则这些权重将与 sample_weight 相乘。
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    # 对偶或原始形式。对于使用 liblinear solver 的 l2 惩罚，只实现了对偶形式。
    # 当 n_samples > n_features 时，推荐使用 dual=False。
    dual : bool, default=False
        # Dual or primal formulation. Dual formulation is only implemented for
        # l2 penalty with liblinear solver. Prefer dual=False when
        # n_samples > n_features.
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    # 惩罚项类型，可选 'l1', 'l2', 'elasticnet'，默认为 'l2'。
    # 用于指定惩罚的范数。'newton-cg', 'sag' 和 'lbfgs' solver 仅支持 l2 惩罚。
    # 'elasticnet' 仅支持 'saga' solver。
    penalty : {'l1', 'l2', 'elasticnet'}, default='l2'
        # Used to specify the norm used in the penalization. The 'newton-cg',
        # 'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is
        # only supported by the 'saga' solver.
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is
        only supported by the 'saga' solver.

    # 截距缩放因子，默认为 1.0。
    # 仅在使用 'liblinear' solver 并且 self.fit_intercept 设置为 True 时有效。
    # 在这种情况下，x 变成 [x, self.intercept_scaling]，
    # 即将一个常量值等于 intercept_scaling 的“合成”特征追加到实例向量中。
    # 截距变成 `intercept_scaling * synthetic_feature_weight`。
    intercept_scaling : float, default=1.
        # Useful only when the solver 'liblinear' is used
        # and self.fit_intercept is set to True. In this case, x becomes
        # [x, self.intercept_scaling],
        # i.e. a "synthetic" feature with constant value equal to
        # intercept_scaling is appended to the instance vector.
        # The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        # Note! the synthetic feature weight is subject to l1/l2 regularization
        # as all other features.
        # To lessen the effect of regularization on synthetic feature weight
        # (and therefore on the intercept) intercept_scaling has to be increased.
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    # 多类别分类策略，可选 'ovr', 'multinomial', 'auto'，默认为 'auto'。
    # 如果选择 'ovr'，则为每个标签拟合一个二元问题。
    # 对于 'multinomial'，即使数据是二元的，也会最小化整个概率分布的多项式损失。
    # 当 solver='liblinear' 时不支持 'multinomial'。
    # 'auto' 会根据数据是否为二元或 solver='liblinear' 自动选择 'ovr' 或 'multinomial'。
    multi_class : {'ovr', 'multinomial', 'auto'}, default='auto'
        # If the option chosen is 'ovr', then a binary problem is fit for each
        # label. For 'multinomial' the loss minimised is the multinomial loss fit
        # across the entire probability distribution, *even when the data is
        # binary*. 'multinomial' is unavailable when solver='liblinear'.
        # 'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        # and otherwise selects 'multinomial'.

        # .. versionadded:: 0.18
        #    Stochastic Average Gradient descent solver for 'multinomial' case.
        # .. versionchanged:: 0.22
        #     Default changed from 'ovr' to 'auto' in 0.22.

    # 随机数种子，整数或 RandomState 实例，默认为 None。
    # 当 ``solver`` == 'sag', 'saga' 或 'liblinear' 时用于对数据进行洗牌。
    # 详见 :term:`Glossary <random_state>`。
    random_state : int, RandomState instance, default=None
        # Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        # data. See :term:`Glossary <random_state>` for details.
        Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        data. See :term:`Glossary <random_state>` for details.

    # 是否检查输入，布尔值，默认为 True。
    # 如果为 False，则输入数组 X 和 y 不会被检查。
    check_input : bool, default=True
        # If False, the input arrays X and y will not be checked.

    # 最大样本上的 X 的平方和，仅在 SAG solver 中使用。默认为 None。
    # 如果为 None，则将通过所有样本进行计算。
    # 该值应预先计算以加快交叉验证的速度。
    max_squared_sum : float, default=None
        # Maximum squared sum of X over samples. Used only in SAG solver.
        # If None, it will be computed, going through all the samples.
        # The value should be precomputed to speed up cross validation.
    sample_weight : array-like of shape(n_samples,), default=None
        # 定义样本权重，形状为 (n_samples,)，默认为 None
        Array of weights that are assigned to individual samples.
        # 数组，用于给每个样本分配权重

    l1_ratio : float, default=None
        # 定义 Elastic-Net 混合参数 l1_ratio，范围为 0 到 1，仅在 penalty='elasticnet' 时使用
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

    n_threads : int, default=1
        # 定义 OpenMP 线程数，默认为 1
        Number of OpenMP threads to use.

    Returns
    -------
    coefs : ndarray of shape (n_cs, n_features) or (n_cs, n_features + 1)
        # 返回 Logistic 回归模型的系数数组，形状为 (n_cs, n_features) 或 (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept. For
        ``multiclass='multinomial'``, the shape is (n_classes, n_cs,
        n_features) or (n_classes, n_cs, n_features + 1).

    Cs : ndarray
        # 返回用于交叉验证的 Cs 网格
        Grid of Cs used for cross-validation.

    n_iter : array of shape (n_cs,)
        # 返回每个 Cs 的实际迭代次数数组，形状为 (n_cs,)
        Actual number of iteration for each Cs.

    Notes
    -----
    You might get slightly different results with the solver liblinear than
    with the others since this uses LIBLINEAR which penalizes the intercept.
    # 注意事项：使用 LIBLINEAR 求解器可能会与其他求解器产生略微不同的结果，因为它对截距进行了惩罚。

    .. versionchanged:: 0.19
        The "copy" parameter was removed.
    """
    # 如果 Cs 是整数，则生成对数间隔的值作为 Cs
    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, Cs)

    # 检查并获取求解器
    solver = _check_solver(solver, penalty, dual)

    # 数据预处理。
    if check_input:
        # 检查输入数据 X 和 y
        X = check_array(
            X,
            accept_sparse="csr",
            dtype=np.float64,
            accept_large_sparse=solver not in ["liblinear", "sag", "saga"],
        )
        y = check_array(y, ensure_2d=False, dtype=None)
        # 检查 X 和 y 的长度是否一致
        check_consistent_length(X, y)
    n_samples, n_features = X.shape

    # 获取类别列表
    classes = np.unique(y)
    # 检查随机状态
    random_state = check_random_state(random_state)

    # 检查多类别设置
    multi_class = _check_multi_class(multi_class, solver, len(classes))
    # 如果未指定正类别且不是多项式分类，则指定第二类别为正类别
    if pos_class is None and multi_class != "multinomial":
        if classes.size > 2:
            raise ValueError("To fit OvR, use the pos_class argument")
        # np.unique(y) gives labels in sorted order.
        pos_class = classes[1]

    # 检查样本权重或类别权重
    if sample_weight is not None or class_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype, copy=True)

    # 如果类别权重是字典（由用户提供），则将权重分配给原始标签；如果是 "balanced"，则在使用 OvR 后分配类别权重
    le = LabelEncoder()
    if isinstance(class_weight, dict) or (
        multi_class == "multinomial" and class_weight is not None
    ):
        # 根据类别权重计算加权值
        class_weight_ = compute_class_weight(class_weight, classes=classes, y=y)
        # 将样本权重乘以类别权重
        sample_weight *= class_weight_[le.fit_transform(y)]

    # 对于进行一对多分类，需要先对标签进行掩码处理。对于多项式情况，这是不必要的。
    if multi_class == "ovr":
        # 初始化权重向量w0，考虑截距时需增加特征数
        w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)
        # 创建标签掩码，选择正类别的标签
        mask = y == pos_class
        # 创建二元标签数组
        y_bin = np.ones(y.shape, dtype=X.dtype)
        if solver in ["lbfgs", "newton-cg", "newton-cholesky"]:
            # HalfBinomialLoss，用于这些求解器，表示y在[0, 1]范围内而非[-1, 1]。
            mask_classes = np.array([0, 1])
            y_bin[~mask] = 0.0
        else:
            mask_classes = np.array([-1, 1])
            y_bin[~mask] = -1.0

        # 对于计算类别权重
        if class_weight == "balanced":
            class_weight_ = compute_class_weight(
                class_weight, classes=mask_classes, y=y_bin
            )
            # 将样本权重乘以类别权重
            sample_weight *= class_weight_[le.fit_transform(y_bin)]

    else:
        if solver in ["sag", "saga", "lbfgs", "newton-cg"]:
            # 对于SAG、lbfgs和newton-cg的多项式求解器，需要使用LabelEncoder，而非LabelBinarizer，
            # 即y作为整数的1维数组。LabelEncoder相比LabelBinarizer在大量类别时节省内存。
            le = LabelEncoder()
            Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)
        else:
            # 对于liblinear求解器，应用LabelBinarizer，即y进行独热编码。
            lbin = LabelBinarizer()
            Y_multi = lbin.fit_transform(y)
            if Y_multi.shape[1] == 1:
                Y_multi = np.hstack([1 - Y_multi, Y_multi])

        # 初始化权重矩阵w0，考虑类别数和截距时需增加特征数
        w0 = np.zeros(
            (classes.size, n_features + int(fit_intercept)), order="F", dtype=X.dtype
        )

    # 重要说明：
    # 所有依赖LinearModelLoss的求解器需要按照样本数或样本权重之和来缩放惩罚项，
    # 因为这里实现的逻辑回归目标是（不幸地）C * sum(pointwise_loss) + penalty，
    # 而不是（如LinearModelLoss所做的）mean(pointwise_loss) + 1/C * penalty。
    if solver in ["lbfgs", "newton-cg", "newton-cholesky"]:
        # 这需要在样本权重乘以类别权重后计算。根据类别权重计算等价于使用样本权重已经被测试过了。
        sw_sum = n_samples if sample_weight is None else np.sum(sample_weight)
    if coef is not None:
        # 如果 coef 参数不为空，进行下面的初始化工作

        # 在多类别分类中，根据不同策略设置权重初始化
        if multi_class == "ovr":
            # 对于 "ovr" 策略，检查 coef 的大小是否符合预期
            if coef.size not in (n_features, w0.size):
                raise ValueError(
                    "Initialization coef is of shape %d, expected shape %d or %d"
                    % (coef.size, n_features, w0.size)
                )
            # 将 coef 的值复制到 w0 的前部分
            w0[: coef.size] = coef
        else:
            # 对于二元问题，coef 的第一维应为 1；多类问题下，应为类别数
            n_classes = classes.size
            if n_classes == 2:
                n_classes = 1

            # 检查 coef 的形状是否符合预期
            if coef.shape[0] != n_classes or coef.shape[1] not in (
                n_features,
                n_features + 1,
            ):
                raise ValueError(
                    "Initialization coef is of shape (%d, %d), expected "
                    "shape (%d, %d) or (%d, %d)"
                    % (
                        coef.shape[0],
                        coef.shape[1],
                        classes.size,
                        n_features,
                        classes.size,
                        n_features + 1,
                    )
                )

            # 根据类别数的不同情况，初始化权重 w0
            if n_classes == 1:
                w0[0, : coef.shape[1]] = -coef
                w0[1, : coef.shape[1]] = coef
            else:
                w0[:, : coef.shape[1]] = coef

    # 处理多项式分类问题中的特殊情况
    if multi_class == "multinomial":
        if solver in ["lbfgs", "newton-cg"]:
            # 对于 lbfgs 和 newton-cg 求解器，将 w0 摊平成一维数组
            w0 = w0.ravel(order="F")
            # 初始化线性模型损失函数
            loss = LinearModelLoss(
                base_loss=HalfMultinomialLoss(n_classes=classes.size),
                fit_intercept=fit_intercept,
            )
        
        # 设置目标变量为 Y_multi
        target = Y_multi
        
        # 根据求解器选择不同的损失函数和梯度函数
        if solver == "lbfgs":
            func = loss.loss_gradient
        elif solver == "newton-cg":
            func = loss.loss
            grad = loss.gradient
            hess = loss.gradient_hessian_product  # hess = [gradient, hessp]
        
        # 在 SAG 求解器下的热启动参数设置
        warm_start_sag = {"coef": w0.T}
    else:
        # 如果未指定 solver，则使用默认的 target 变量 y_bin
        target = y_bin
        # 根据 solver 类型选择不同的损失函数和相关函数
        if solver == "lbfgs":
            # 使用半二项损失作为基础损失函数，创建线性模型损失对象
            loss = LinearModelLoss(
                base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept
            )
            # 设置梯度函数为损失函数的梯度函数
            func = loss.loss_gradient
        elif solver == "newton-cg":
            # 使用半二项损失作为基础损失函数，创建线性模型损失对象
            loss = LinearModelLoss(
                base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept
            )
            # 设置 func 为损失函数的损失函数，grad 为梯度函数，hess 为梯度-海森矩阵乘积函数
            func = loss.loss
            grad = loss.gradient
            hess = loss.gradient_hessian_product  # hess = [gradient, hessp]
        elif solver == "newton-cholesky":
            # 使用半二项损失作为基础损失函数，创建线性模型损失对象
            loss = LinearModelLoss(
                base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept
            )
        # 使用初始参数 w0 构建 warm_start_sag 字典，包含系数 coef
        warm_start_sag = {"coef": np.expand_dims(w0, axis=1)}

    # 初始化系数列表 coefs
    coefs = list()
    # 初始化迭代次数数组 n_iter，长度为 Cs 数组的长度，类型为整数
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    # 返回系数数组 coefs、参数数组 Cs、迭代次数数组 n_iter 的 numpy 数组形式
    return np.array(coefs), np.array(Cs), n_iter
# helper function for LogisticCV
def _log_reg_scoring_path(
    X,
    y,
    train,
    test,
    *,
    pos_class,
    Cs,
    scoring,
    fit_intercept,
    max_iter,
    tol,
    class_weight,
    verbose,
    solver,
    penalty,
    dual,
    intercept_scaling,
    multi_class,
    random_state,
    max_squared_sum,
    sample_weight,
    l1_ratio,
    score_params,
):
    """Computes scores across logistic_regression_path

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target labels.

    train : list of indices
        The indices of the train set.

    test : list of indices
        The indices of the test set.

    pos_class : int
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.

    Cs : int or list of floats
        Each of the values in Cs describes the inverse of
        regularization strength. If Cs is as an int, then a grid of Cs
        values are chosen in a logarithmic scale between 1e-4 and 1e4.

    scoring : callable
        A string (see :ref:`scoring_parameter`) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``. For a list of scoring functions
        that can be used, look at :mod:`sklearn.metrics`.

    fit_intercept : bool
        If False, then the bias term is set to zero. Else the last
        term of each coef_ gives us the intercept.

    max_iter : int
        Maximum number of iterations for the solver.

    tol : float
        Tolerance for stopping criteria.

    class_weight : dict or 'balanced'
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    verbose : int
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    solver : {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}
        Decides which solver to use.

    penalty : {'l1', 'l2', 'elasticnet'}
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is
        only supported by the 'saga' solver.

    dual : bool
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    intercept_scaling : float, optional, default=1
        Useful only when the solver 'liblinear' is used and self.fit_intercept
        is set to True. In this case, x becomes [x, self.intercept_scaling]
        where x is a row of sample_weight.

    multi_class : str, {'ovr', 'multinomial', 'auto'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. If the option chosen is 'multinomial', then the loss minimized
        is the multinomial loss fit across the entire probability distribution,
        even when the data is binary. 'auto' selects 'ovr' for binary problems
        and 'multinomial' otherwise.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator to use when shuffling
        the data.

    max_squared_sum : float, optional, default=Inf
        Maximum squared sum of all features (over samples) when dual=False.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weight. If None, then samples are equally weighted.

    l1_ratio : float, optional, default=None
        The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used
        if penalty='elasticnet'. Setting ``l1_ratio=0`` is equivalent to
        using penalty='l2', while setting ``l1_ratio=1`` is equivalent to
        using penalty='l1'. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

    score_params : dict, optional, default=None
        Parameters passed to the score function.

    Returns
    -------
    scores : ndarray of shape (n_Cs,)
        Scores (per-fold) across the different Cs values.

    """
    # Function body intentionally left empty
    pass
    # 从输入数据 X 中根据索引 train 取出训练集数据 X_train
    X_train = X[train]
    # 从输入数据 X 中根据索引 test 取出测试集数据 X_test
    X_test = X[test]
    # 从标签数据 y 中根据索引 train 取出训练集标签 y_train
    y_train = y[train]
    # 从标签数据 y 中根据索引 test 取出测试集标签 y_test
    y_test = y[test]

    # 如果提供了样本权重 sample_weight
    if sample_weight is not None:
        # 根据输入数据 X 和样本权重 sample_weight 检查和调整样本权重
        sample_weight = _check_sample_weight(sample_weight, X)
        # 根据索引 train 取出训练集样本权重
        sample_weight = sample_weight[train]
    coefs, Cs, n_iter = _logistic_regression_path(
        X_train,
        y_train,
        Cs=Cs,
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        pos_class=pos_class,
        multi_class=multi_class,
        tol=tol,
        verbose=verbose,
        dual=dual,
        penalty=penalty,
        intercept_scaling=intercept_scaling,
        random_state=random_state,
        check_input=False,
        max_squared_sum=max_squared_sum,
        sample_weight=sample_weight,
    )


    # 运行逻辑回归路径优化函数，获取系数、正则化参数列表和迭代次数
    coefs, Cs, n_iter = _logistic_regression_path(
        X_train,                    # 训练数据特征
        y_train,                    # 训练数据标签
        Cs=Cs,                      # 正则化参数的候选列表
        l1_ratio=l1_ratio,          # L1 正则化比例
        fit_intercept=fit_intercept,    # 是否拟合截距
        solver=solver,              # 优化算法
        max_iter=max_iter,          # 最大迭代次数
        class_weight=class_weight,  # 类别权重
        pos_class=pos_class,        # 正类标签
        multi_class=multi_class,    # 多分类策略
        tol=tol,                    # 收敛阈值
        verbose=verbose,            # 是否输出详细信息
        dual=dual,                  # 对偶形式
        penalty=penalty,            # 正则化类型
        intercept_scaling=intercept_scaling,  # 截距缩放因子
        random_state=random_state,  # 随机种子
        check_input=False,          # 是否检查输入数据
        max_squared_sum=max_squared_sum,    # 特征平方和的最大值
        sample_weight=sample_weight,    # 样本权重
    )

    log_reg = LogisticRegression(solver=solver, multi_class=multi_class)


    # 创建逻辑回归模型对象
    log_reg = LogisticRegression(solver=solver, multi_class=multi_class)

    # Logistic Regression 的 score 方法具有 classes_ 属性。
    if multi_class == "ovr":
        # 如果是一对多分类，则设置类别为 [-1, 1]
        log_reg.classes_ = np.array([-1, 1])
    elif multi_class == "multinomial":
        # 如果是多项式分类，则设置类别为训练集中的唯一类别
        log_reg.classes_ = np.unique(y_train)
    else:
        # 如果多分类策略既不是一对多也不是多项式，则抛出异常
        raise ValueError(
            "multi_class should be either multinomial or ovr, got %d" % multi_class
        )

    if pos_class is not None:
        # 如果指定了正类标签，则将测试集中等于正类标签的样本标签设为 1，其余为 -1
        mask = y_test == pos_class
        y_test = np.ones(y_test.shape, dtype=np.float64)
        y_test[~mask] = -1.0

    scores = list()

    # 获取评分器对象
    scoring = get_scorer(scoring)
    for w in coefs:
        if multi_class == "ovr":
            # 如果是一对多分类，需要将 w 转换为二维数组
            w = w[np.newaxis, :]
        if fit_intercept:
            # 如果拟合截距，将 w 的最后一列作为截距，其余作为系数
            log_reg.coef_ = w[:, :-1]
            log_reg.intercept_ = w[:, -1]
        else:
            # 如果不拟合截距，直接将 w 设置为系数，截距为 0
            log_reg.coef_ = w
            log_reg.intercept_ = 0.0

        if scoring is None:
            # 如果评分器为空，则使用模型的 score 方法评估在测试集上的得分
            scores.append(log_reg.score(X_test, y_test))
        else:
            # 如果评分器不为空，则使用指定的评分方法计算得分
            score_params = score_params or {}
            score_params = _check_method_params(X=X, params=score_params, indices=test)
            scores.append(scoring(log_reg, X_test, y_test, **score_params))

    # 返回系数列表、正则化参数列表、得分数组和迭代次数
    return coefs, Cs, np.array(scores), n_iter
class LogisticRegression(LinearClassifierMixin, SparseCoefMixin, BaseEstimator):
    """
    Logistic Regression (aka logit, MaxEnt) classifier.

    In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
    scheme if the 'multi_class' option is set to 'ovr', and uses the
    cross-entropy loss if the 'multi_class' option is set to 'multinomial'.
    (Currently the 'multinomial' option is supported only by the 'lbfgs',
    'sag', 'saga' and 'newton-cg' solvers.)

    This class implements regularized logistic regression using the
    'liblinear' library, 'newton-cg', 'sag', 'saga' and 'lbfgs' solvers. **Note
    that regularization is applied by default**. It can handle both dense
    and sparse input. Use C-ordered arrays or CSR matrices containing 64-bit
    floats for optimal performance; any other input format will be converted
    (and copied).

    The 'newton-cg', 'sag', and 'lbfgs' solvers support only L2 regularization
    with primal formulation, or no regularization. The 'liblinear' solver
    supports both L1 and L2 regularization, with a dual formulation only for
    the L2 penalty. The Elastic-Net regularization is only supported by the
    'saga' solver.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    penalty : {'l1', 'l2', 'elasticnet', None}, default='l2'
        Specify the norm of the penalty:

        - `None`: no penalty is added;
        - `'l2'`: add a L2 penalty term and it is the default choice;
        - `'l1'`: add a L1 penalty term;
        - `'elasticnet'`: both L1 and L2 penalty terms are added.

        .. warning::
           Some penalties may not work with some solvers. See the parameter
           `solver` below, to know the compatibility between the penalty and
           solver.

        .. versionadded:: 0.19
           l1 penalty with SAGA solver (allowing 'multinomial' + L1)

    dual : bool, default=False
        Dual (constrained) or primal (regularized, see also
        :ref:`this equation <regularized-logistic-loss>`) formulation. Dual formulation
        is only implemented for l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
    """
    intercept_scaling : float, default=1
        # 截距缩放因子，默认为1
        # 仅在使用 solver 'liblinear' 且 self.fit_intercept 设置为 True 时有效。
        # 在这种情况下，x 变成 [x, self.intercept_scaling]，
        # 即在实例向量末尾添加一个常数值为 intercept_scaling 的“合成”特征。
        # 截距变为 ``intercept_scaling * synthetic_feature_weight``。

        # 注意！合成特征的权重受到 l1/l2 正则化的影响，与其他特征一样。
        # 若要减少正则化对合成特征权重（因此也是截距）的影响，需增加 intercept_scaling。

    class_weight : dict or 'balanced', default=None
        # 类别权重，格式为 ``{class_label: weight}``。
        # 如果未指定，假定所有类别权重均为1。

        # "balanced" 模式根据输入数据中类别的频率自动调整权重，与类别频率成反比。
        # 调整方式为 ``n_samples / (n_classes * np.bincount(y))``。

        # 注意，如果指定了 sample_weight（通过 fit 方法传入），这些权重将与 sample_weight 相乘。

        # .. versionadded:: 0.17
        #    *class_weight='balanced'*

    random_state : int, RandomState instance, default=None
        # 在 ``solver`` 为 'sag', 'saga' 或 'liblinear' 时用于数据的洗牌操作。
        # 详见 :term:`术语表 <random_state>`。
    solver : {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}, \
            default='lbfgs'
    # 优化问题中使用的算法。默认为 'lbfgs'。
    # 选择算法时，可以考虑以下几个方面：
    #
    # - 对于小数据集，'liblinear' 是一个不错的选择，而对于大数据集，'sag' 和 'saga' 更快；
    # - 对于多类问题，只有 'newton-cg', 'sag', 'saga' 和 'lbfgs' 支持多项损失；
    # - 'liblinear' 和 'newton-cholesky' 默认只能处理二分类问题。对于多类别设置，可以使用 `OneVsRestClassifier` 将其包装为一对多的方案；
    # - 'newton-cholesky' 对于 `n_samples` >> `n_features` 的情况特别适用，特别是对于具有稀有类别的独热编码分类特征。请注意，此求解器的内存使用量与 `n_features` 的平方依赖关系相关，因为它显式计算 Hessian 矩阵。
    #
    # .. warning::
    #    算法的选择取决于所选择的惩罚以及（多项式）多类支持：
    #
    #    ================= ============================== ======================
    #    solver            penalty                        multinomial multiclass
    #    ================= ============================== ======================
    #    'lbfgs'           'l2', None                     是
    #    'liblinear'       'l1', 'l2'                     否
    #    'newton-cg'       'l2', None                     是
    #    'newton-cholesky' 'l2', None                     否
    #    'sag'             'l2', None                     是
    #    'saga'            'elasticnet', 'l1', 'l2', None 是
    #    ================= ============================== ======================
    #
    # .. note::
    #    'sag' 和 'saga' 只有在特征大致相同的情况下才能保证快速收敛。您可以使用 :mod:`sklearn.preprocessing` 中的缩放器预处理数据。
    #
    # .. seealso::
    #    更多关于 :class:`LogisticRegression` 的信息，请参考用户指南，特别是总结了求解器/惩罚支持的 :ref:`Table <Logistic_regression>`。
    #
    # .. versionadded:: 0.17
    #    引入了随机平均梯度下降求解器（Stochastic Average Gradient descent solver）。
    # .. versionadded:: 0.19
    #    引入了SAGA求解器。
    # .. versionchanged:: 0.22
    #    默认求解器从0.22版本更改为'lbfgs'。
    # .. versionadded:: 1.2
    #    引入了newton-cholesky求解器。

    max_iter : int, default=100
    # 求解器收敛所允许的最大迭代次数。
    multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
        # 多类别分类的策略选择，默认为'auto'
        如果选择为'ovr'，则对每个标签进行二元分类。
        对于'multinomial'，则是针对整个概率分布进行多项式损失的最小化，即使数据是二元的。
        当solver='liblinear'时，不可用'multinomial'。
        'auto'会根据数据是否为二元或solver='liblinear'来自动选择策略。

        .. versionadded:: 0.18
           在0.18版本中添加了随机平均梯度下降（SAG）求解器用于'multinomial'情况。
        .. versionchanged:: 0.22
            默认值从0.22版本开始从'ovr'改为'auto'。
        .. deprecated:: 1.5
           `multi_class`从1.5版本开始弃用，并将在1.7版本中移除。
           从那时起，推荐始终使用'multinomial'用于`n_classes >= 3`的情况。
           不支持'multinomial'的求解器会引发错误。
           如果仍想使用OvR，请使用`sklearn.multiclass.OneVsRestClassifier(LogisticRegression())`。

    verbose : int, default=0
        # 对于liblinear和lbfgs求解器，设置为任何正数以增加详细信息输出的级别。

    warm_start : bool, default=False
        # 当设置为True时，重用上一次fit调用的解作为初始化，否则仅清除上一个解决方案。
        对于liblinear求解器无效。参见：术语表中的“Warm Start”。

        .. versionadded:: 0.17
           0.17版本中添加了*warm_start*以支持*lbfgs*、*newton-cg*、*sag*、*saga*求解器。

    n_jobs : int, default=None
        # 当multi_class='ovr'时，并行化处理类别时使用的CPU核心数。
        当solver设置为'liblinear'时，无论是否指定'multi_class'，此参数都会被忽略。
        None意味着默认为1，除非在joblib.parallel_backend上下文中。
        -1意味着使用所有处理器。
        更多细节请参见：术语表中的“n_jobs”。

    l1_ratio : float, default=None
        # 弹性网络混合参数，满足`0 <= l1_ratio <= 1`。
        仅在`penalty='elasticnet'`时使用。
        设置`l1_ratio=0`等效于使用`penalty='l2'`，而设置`l1_ratio=1`等效于使用`penalty='l1'`。
        对于`0 < l1_ratio < 1`，惩罚项是L1和L2的组合。

    Attributes
    ----------

    classes_ : ndarray of shape (n_classes, )
        # 分类器已知的类标签列表。

    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        # 决策函数中特征的系数。
        当给定问题是二元分类时，`coef_`的形状为(1, n_features)。
        特别地，在`multi_class='multinomial'`时，`coef_`对应于结果1（True），`-coef_`对应于结果0（False）。
    intercept_ : ndarray of shape (1,) or (n_classes,)
        # 拦截器（偏置）添加到决策函数中。
        
        If `fit_intercept` is set to False, the intercept is set to zero.
        # 如果 `fit_intercept` 设置为 False，则拦截器设置为零。
        `intercept_` is of shape (1,) when the given problem is binary.
        # 当问题是二元的时候，`intercept_` 的形状为 (1,)。
        In particular, when `multi_class='multinomial'`, `intercept_`
        corresponds to outcome 1 (True) and `-intercept_` corresponds to
        outcome 0 (False).
        # 特别是当 `multi_class='multinomial'` 时，`intercept_` 对应于结果 1（True），`-intercept_` 对应于结果 0（False）。

    n_features_in_ : int
        # 在拟合期间观察到的特征数量。
        
        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在拟合期间观察到的特征的名称。仅在 `X` 具有所有字符串特征名称时定义。
        
        .. versionadded:: 1.0

    n_iter_ : ndarray of shape (n_classes,) or (1, )
        # 所有类别的实际迭代次数。如果是二元或多项式的，则仅返回一个元素。对于 liblinear 求解器，仅给出所有类的最大迭代次数。
        
        .. versionchanged:: 0.20
            在 SciPy <= 1.0.0 中，lbfgs 的迭代次数可能超过 ``max_iter``。现在 ``n_iter_`` 将最多报告 ``max_iter``。

    See Also
    --------
    SGDClassifier : Incrementally trained logistic regression (when given
        the parameter ``loss="log_loss"``).
    LogisticRegressionCV : Logistic regression with built-in cross validation.

    Notes
    -----
    The underlying C implementation uses a random number generator to
    select features when fitting the model. It is thus not uncommon,
    to have slightly different results for the same input data. If
    that happens, try with a smaller tol parameter.
    # 底层的 C 实现在拟合模型时使用随机数生成器选择特征。因此，对于相同的输入数据，结果可能略有不同。如果出现这种情况，请尝试较小的 tol 参数。

    Predict output may not match that of standalone liblinear in certain
    cases. See :ref:`differences from liblinear <liblinear_differences>`
    in the narrative documentation.
    # 在某些情况下，预测输出可能与独立的 liblinear 不匹配。参见叙述文档中的 :ref:`differences from liblinear <liblinear_differences>`。

    References
    ----------

    L-BFGS-B -- Software for Large-scale Bound-constrained Optimization
        Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis Morales.
        http://users.iems.northwestern.edu/~nocedal/lbfgsb.html

    LIBLINEAR -- A Library for Large Linear Classification
        https://www.csie.ntu.edu.tw/~cjlin/liblinear/

    SAG -- Mark Schmidt, Nicolas Le Roux, and Francis Bach
        Minimizing Finite Sums with the Stochastic Average Gradient
        https://hal.inria.fr/hal-00860051/document

    SAGA -- Defazio, A., Bach F. & Lacoste-Julien S. (2014).
            :arxiv:`"SAGA: A Fast Incremental Gradient Method With Support
            for Non-Strongly Convex Composite Objectives" <1407.0202>`

    Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent
        methods for logistic regression and maximum entropy models.
        Machine Learning 85(1-2):41-75.
        https://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    _parameter_constraints: dict = {
        "penalty": [StrOptions({"l1", "l2", "elasticnet"}), None],
        "dual": ["boolean"],
        "tol": [Interval(Real, 0, None, closed="left")],
        "C": [Interval(Real, 0, None, closed="right")],
        "fit_intercept": ["boolean"],
        "intercept_scaling": [Interval(Real, 0, None, closed="neither")],
        "class_weight": [dict, StrOptions({"balanced"}), None],
        "random_state": ["random_state"],
        "solver": [
            StrOptions(
                {"lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"}
            )
        ],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "verbose": ["verbose"],
        "warm_start": ["boolean"],
        "n_jobs": [None, Integral],
        "l1_ratio": [Interval(Real, 0, 1, closed="both"), None],
        "multi_class": [
            StrOptions({"auto", "ovr", "multinomial"}),
            Hidden(StrOptions({"deprecated"})),
        ],
    }
    # 参数约束字典，定义了每个参数可接受的取值范围或类型限制

    def __init__(
        self,
        penalty="l2",
        *,
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="deprecated",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):
        # 初始化方法，设置逻辑回归模型的各个参数
        self.penalty = penalty  # 惩罚项，默认为'l2'
        self.dual = dual  # 是否使用对偶形式，默认为False
        self.tol = tol  # 收敛阈值，默认为1e-4
        self.C = C  # 正则化强度，默认为1.0
        self.fit_intercept = fit_intercept  # 是否拟合截距，默认为True
        self.intercept_scaling = intercept_scaling  # 截距缩放，默认为1
        self.class_weight = class_weight  # 类别权重，默认为None
        self.random_state = random_state  # 随机种子，默认为None
        self.solver = solver  # 求解器类型，默认为'lbfgs'
        self.max_iter = max_iter  # 最大迭代次数，默认为100
        self.multi_class = multi_class  # 多分类策略，默认为'deprecated'
        self.verbose = verbose  # 是否输出详细信息，默认为0
        self.warm_start = warm_start  # 是否热启动，默认为False
        self.n_jobs = n_jobs  # 并行运算数，默认为None
        self.l1_ratio = l1_ratio  # L1 正则化比例，默认为None

    @_fit_context(prefer_skip_nested_validation=True)
    # 装饰器，用于执行拟合过程的上下文设置，优先跳过嵌套验证
    def predict_proba(self, X):
        """
        Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        Else use a one-vs-rest approach, i.e. calculate the probability
        of each class assuming it to be positive using the logistic function
        and normalize these values across all the classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)

        # 判断是否使用一对多策略
        ovr = self.multi_class in ["ovr", "warn"] or (
            self.multi_class in ["auto", "deprecated"]
            and (
                self.classes_.size <= 2
                or self.solver in ("liblinear", "newton-cholesky")
            )
        )
        if ovr:
            # 如果使用一对多策略，则调用父类的逻辑回归预测概率方法
            return super()._predict_proba_lr(X)
        else:
            # 否则，计算决策函数的输出
            decision = self.decision_function(X)
            if decision.ndim == 1:
                # 对于多类问题且二元结果，需要进行 softmax 预测
                decision_2d = np.c_[-decision, decision]
            else:
                decision_2d = decision
            # 使用 softmax 函数计算概率值
            return softmax(decision_2d, copy=False)

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        # 返回预测概率的对数值
        return np.log(self.predict_proba(X))
class LogisticRegressionCV(LogisticRegression, LinearClassifierMixin, BaseEstimator):
    """Logistic Regression CV (aka logit, MaxEnt) classifier.

    See glossary entry for :term:`cross-validation estimator`.

    This class implements logistic regression using liblinear, newton-cg, sag
    or lbfgs optimizer. The newton-cg, sag and lbfgs solvers support only L2
    regularization with primal formulation. The liblinear solver supports both
    L1 and L2 regularization, with a dual formulation only for the L2 penalty.
    Elastic-Net penalty is only supported by the saga solver.

    For the grid of `Cs` values and `l1_ratios` values, the best hyperparameter
    is selected by the cross-validator
    :class:`~sklearn.model_selection.StratifiedKFold`, but it can be changed
    using the :term:`cv` parameter. The 'newton-cg', 'sag', 'saga' and 'lbfgs'
    solvers can warm-start the coefficients (see :term:`Glossary<warm_start>`).

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    Cs : int or list of floats, default=10
        Each of the values in Cs describes the inverse of regularization
        strength. If Cs is as an int, then a grid of Cs values are chosen
        in a logarithmic scale between 1e-4 and 1e4.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    cv : int or cross-validation generator, default=None
        The default cross-validation generator used is Stratified K-Folds.
        If an integer is provided, then it is the number of folds used.
        See the module :mod:`sklearn.model_selection` module for the
        list of possible cross-validation objects.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    dual : bool, default=False
        Dual (constrained) or primal (regularized, see also
        :ref:`this equation <regularized-logistic-loss>`) formulation. Dual formulation
        is only implemented for l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    penalty : {'l1', 'l2', 'elasticnet'}, default='l2'
        Specify the norm of the penalty:

        - `'l2'`: add a L2 penalty term (used by default);
        - `'l1'`: add a L1 penalty term;
        - `'elasticnet'`: both L1 and L2 penalty terms are added.

        .. warning::
           Some penalties may not work with some solvers. See the parameter
           `solver` below, to know the compatibility between the penalty and
           solver.
"""
    # scoring 参数可以是字符串或可调用对象（函数），默认为 None。
    # 如果是字符串，参考 scoring_parameter；如果是可调用对象，其签名应为 scorer(estimator, X, y)。
    # 默认的评分选项是 'accuracy'。

    scoring : str or callable, default=None

    # solver 参数用于指定优化问题中使用的算法，默认为 'lbfgs'。
    # 选择算法时，可以考虑以下几个方面：
    #
    # - 对于小数据集，'liblinear' 是一个不错的选择，而对于大数据集，'sag' 和 'saga' 更快；
    # - 对于多类分类问题，只有 'newton-cg', 'sag', 'saga' 和 'lbfgs' 支持 multinomial 损失；
    # - 在 LogisticRegressionCV 中，'liblinear' 可能速度较慢，因为它不支持 warm-starting；
    # - 'liblinear' 和 'newton-cholesky' 默认只能处理二元分类。对于多类别设置，
    #   可以使用 `OneVsRestClassifier` 进行包装。
    # - 'newton-cholesky' 在 `n_samples` >> `n_features` 的情况下是一个不错的选择，
    #   特别是对于具有稀有类别的独热编码分类特征。请注意，该求解器的内存使用量与 `n_features`
    #   呈二次依赖，因为它显式计算 Hessian 矩阵。
    #
    # .. warning::
    #    算法的选择取决于选择的惩罚和（多项式）多类支持情况：
    #
    #    ================= ============================== ======================
    #    solver            penalty                        multinomial multiclass
    #    ================= ============================== ======================
    #    'lbfgs'           'l2'                           yes
    #    'liblinear'       'l1', 'l2'                     no
    #    'newton-cg'       'l2'                           yes
    #    'newton-cholesky' 'l2',                          no
    #    'sag'             'l2',                          yes
    #    'saga'            'elasticnet', 'l1', 'l2'       yes
    #    ================= ============================== ======================
    #
    # .. note::
    #    'sag' 和 'saga' 的快速收敛仅在具有近似相同尺度的特征上得到保证。
    #    您可以使用 :mod:`sklearn.preprocessing` 中的缩放器预处理数据。
    #
    # .. versionadded:: 0.17
    #    随机平均梯度下降算法（Stochastic Average Gradient descent solver）。
    # .. versionadded:: 0.19
    #    SAGA 求解器。
    # .. versionadded:: 1.2
    #    newton-cholesky 求解器。

    solver : {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}, \
            default='lbfgs'

    # tol 参数指定优化算法停止标准的容差值，默认为 1e-4。

    tol : float, default=1e-4

    # max_iter 参数指定优化算法的最大迭代次数，默认为 100。

    max_iter : int, default=100
    class_weight : dict or 'balanced', default=None
        # class_weight可以是一个字典，格式为{类标签: 权重}，或者字符串'balanced'
        # 如果未提供，则假定所有类的权重都为1
        # "balanced"模式根据y的值自动调整权重，与输入数据中类别频率成反比，计算公式为 n_samples / (n_classes * np.bincount(y))
        # 注意，如果指定了sample_weight（通过fit方法传递），这些权重将与sample_weight相乘
        .. versionadded:: 0.17
           class_weight == 'balanced'

    n_jobs : int, default=None
        # 在交叉验证循环中使用的CPU核数
        # None表示默认为1，除非在joblib.parallel_backend上下文中
        # -1表示使用所有处理器。有关详细信息，请参见术语表中的“n_jobs”
        for more details.
        # 详细信息请参阅Glossary <n_jobs>

    verbose : int, default=0
        # 对于'sag'和'lbfgs'求解器，设置verbose为任何正数以获取详细信息
        # 对于'liblinear'求解器，verbose用于控制冗长度

    refit : bool, default=True
        # 如果设置为True，则分数将在所有折叠上平均，并且将采用对应于最佳分数的coefs和C值，并使用这些参数进行最终的refit
        # 否则，将对所有折叠上最佳分数对应的coefs、intercepts和C值进行平均

    intercept_scaling : float, default=1
        # 仅在使用'sag'和self.fit_intercept为True时有用的求解器'liblinear'情况下
        # 在这种情况下，x变为[x, self.intercept_scaling]，即添加了一个“合成”特征，其常量值等于intercept_scaling
        # 截距变为intercept_scaling * synthetic_feature_weight
        # 注意！合成特征权重受到l1/l2正则化的影响，就像所有其他特征一样
        # 为了减少对合成特征权重（因此对截距）的正则化影响，需要增加intercept_scaling。
    multi_class : {'auto, 'ovr', 'multinomial'}, default='auto'
        # 多分类策略选择，默认为'auto'
        # - 'ovr': 为每个标签拟合一个二元分类问题
        # - 'multinomial': 使用整个概率分布拟合多项式损失，即使数据是二进制的
        #   当 solver='liblinear' 时，不可用
        # - 'auto': 如果数据是二进制的或 solver='liblinear'，则选择'ovr'，否则选择'multinomial'
        # 版本新增: 0.18，添加了针对'multinomial'情况的随机平均梯度下降求解器
        # 版本更改: 0.22，默认从'ovr'更改为'auto'
        # 弃用: 1.5，版本1.5中弃用'multi_class'，将在版本1.7中移除
        # 推荐对于 n_classes >= 3 总是使用'multinomial'，不支持'multinomial'的求解器将引发错误
        # 如果仍希望使用 OvR，请使用 `sklearn.multiclass.OneVsRestClassifier(LogisticRegressionCV())`

    random_state : int, RandomState instance, default=None
        # 当 `solver='sag'`, 'saga' 或 'liblinear' 时用于对数据进行洗牌
        # 注意，这仅适用于求解器，不适用于交叉验证生成器。详见词汇表中的“随机状态”。

    l1_ratios : list of float, default=None
        # Elastic-Net 混合参数的列表，满足 ``0 <= l1_ratio <= 1``
        # 仅在 ``penalty='elasticnet'`` 时使用
        # 当 `l1_ratio=0` 时，相当于使用 ``penalty='l2'``
        # 当 `l1_ratio=1` 时，相当于使用 ``penalty='l1'``
        # 对于 ``0 < l1_ratio < 1``，惩罚项是 L1 和 L2 的组合

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes, )
        # 分类器已知的类标签列表

    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        # 决策函数中特征的系数
        # 当问题是二分类时，`coef_` 的形状为 (1, n_features)

    intercept_ : ndarray of shape (1,) or (n_classes,)
        # 决策函数中添加的截距（偏置）
        # 如果 `fit_intercept=False`，则截距设为零
        # 当问题是二分类时，`intercept_` 的形状为 (1,)

    Cs_ : ndarray of shape (n_cs)
        # 用于交叉验证的 C 值数组，即正则化参数的倒数

    l1_ratios_ : ndarray of shape (n_l1_ratios)
        # 用于交叉验证的 l1_ratio 数组
        # 如果没有使用 l1_ratio（即 penalty 不是 'elasticnet'），则设为 ``[None]``
    # `coefs_paths_`是一个字典，键为类别，值为在交叉验证过程中每个折叠和每个超参数 C 对应的系数路径。
    # 如果选择了 'multi_class' 选项为 'multinomial'，则 `coefs_paths_` 对应于每个类别的系数。
    # 每个值的形状为 ``(n_folds, n_cs, n_features)`` 或 ``(n_folds, n_cs, n_features + 1)``，取决于是否拟合了截距。
    # 如果 `penalty='elasticnet'`，形状为 ``(n_folds, n_cs, n_l1_ratios_, n_features)`` 或 ``(n_folds, n_cs, n_l1_ratios_, n_features + 1)``。
    coefs_paths_ : ndarray of shape (n_folds, n_cs, n_features) or \
                   (n_folds, n_cs, n_features + 1)
        dict with classes as the keys, and the path of coefficients obtained
        during cross-validating across each fold and then across each Cs
        after doing an OvR for the corresponding class as values.
        If the 'multi_class' option is set to 'multinomial', then
        the coefs_paths are the coefficients corresponding to each class.
        Each dict value has shape ``(n_folds, n_cs, n_features)`` or
        ``(n_folds, n_cs, n_features + 1)`` depending on whether the
        intercept is fit or not. If ``penalty='elasticnet'``, the shape is
        ``(n_folds, n_cs, n_l1_ratios_, n_features)`` or
        ``(n_folds, n_cs, n_l1_ratios_, n_features + 1)``.

    # `scores_`是一个字典，键为类别，值为在交叉验证过程中每个折叠后进行 OvR 后获得的得分网格。
    # 如果 'multi_class' 选项为 'multinomial'，则所有类别使用相同的得分，因为这是多项式类别。
    # 每个值的形状为 ``(n_folds, n_cs)`` 或 ``(n_folds, n_cs, n_l1_ratios)``（如果 `penalty='elasticnet'`）。
    scores_ : dict
        dict with classes as the keys, and the values as the
        grid of scores obtained during cross-validating each fold, after doing
        an OvR for the corresponding class. If the 'multi_class' option
        given is 'multinomial' then the same scores are repeated across
        all classes, since this is the multinomial class. Each dict value
        has shape ``(n_folds, n_cs)`` or ``(n_folds, n_cs, n_l1_ratios)`` if
        ``penalty='elasticnet'``.

    # `C_` 是一个数组，形状为 (n_classes,) 或 (n_classes - 1,)，其中存储了每个类别对应的最佳分数所对应的 C 值。
    # 如果 refit 设置为 False，则对于每个类别，最佳 C 值是对应于每个折叠的最佳分数的 C 值的平均值。
    # 当问题为二分类时，`C_` 的形状为 (n_classes,)。
    C_ : ndarray of shape (n_classes,) or (n_classes - 1,)
        Array of C that maps to the best scores across every class. If refit is
        set to False, then for each class, the best C is the average of the
        C's that correspond to the best scores for each fold.
        `C_` is of shape(n_classes,) when the problem is binary.

    # `l1_ratio_` 是一个数组，形状为 (n_classes,) 或 (n_classes - 1,)，其中存储了每个类别对应的最佳分数所对应的 l1_ratio 值。
    # 如果 refit 设置为 False，则对于每个类别，最佳 l1_ratio 值是对应于每个折叠的最佳分数的 l1_ratio 值的平均值。
    # 当问题为二分类时，`l1_ratio_` 的形状为 (n_classes,)。
    l1_ratio_ : ndarray of shape (n_classes,) or (n_classes - 1,)
        Array of l1_ratio that maps to the best scores across every class. If
        refit is set to False, then for each class, the best l1_ratio is the
        average of the l1_ratio's that correspond to the best scores for each
        fold.  `l1_ratio_` is of shape(n_classes,) when the problem is binary.

    # `n_iter_` 是一个数组，形状为 (n_classes, n_folds, n_cs) 或 (1, n_folds, n_cs)，
    # 记录了每个类别、每个折叠和每个超参数 C 的实际迭代次数。
    # 在二分类或多分类问题中，第一维度为 1。
    # 如果 `penalty='elasticnet'`，形状为 ``(n_classes, n_folds, n_cs, n_l1_ratios)`` 或 ``(1, n_folds, n_cs, n_l1_ratios)``。
    n_iter_ : ndarray of shape (n_classes, n_folds, n_cs) or (1, n_folds, n_cs)
        Actual number of iterations for all classes, folds and Cs.
        In the binary or multinomial cases, the first dimension is equal to 1.
        If ``penalty='elasticnet'``, the shape is ``(n_classes, n_folds,
        n_cs, n_l1_ratios)`` or ``(1, n_folds, n_cs, n_l1_ratios)``.

    # `n_features_in_` 是一个整数，表示在 `fit` 过程中看到的特征数量。
    # .. versionadded:: 0.24
    n_features_in_ : int
        Number of features seen during :term:`fit`.

    # `feature_names_in_` 是一个形状为 (`n_features_in_`,) 的数组，
    # 包含在 `fit` 过程中看到的特征的名称。仅当 `X` 的特征名全部为字符串时定义。
    # .. versionadded:: 1.0
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    # 参见
    # --------
    # LogisticRegression : 不调整超参数 `C` 的逻辑回归模型。
    See Also
    --------
    LogisticRegression : Logistic regression without tuning the
        hyperparameter `C`.

    # 示例
    # --------
    # >>> from sklearn.datasets import load_iris
    # >>> from sklearn.linear_model import LogisticRegressionCV
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegressionCV
    """
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    >>> clf.predict_proba(X[:2, :]).shape
    (2, 3)
    >>> clf.score(X, y)
    0.98...
    """

    # 初始化_parameter_constraints字典，使用LogisticRegression._parameter_constraints的内容
    _parameter_constraints: dict = {**LogisticRegression._parameter_constraints}

    # 从_parameter_constraints中移除指定的参数："C", "warm_start", "l1_ratio"
    for param in ["C", "warm_start", "l1_ratio"]:
        _parameter_constraints.pop(param)

    # 更新_parameter_constraints字典，添加新的参数约束信息
    _parameter_constraints.update(
        {
            "Cs": [Interval(Integral, 1, None, closed="left"), "array-like"],
            "cv": ["cv_object"],
            "scoring": [StrOptions(set(get_scorer_names())), callable, None],
            "l1_ratios": ["array-like", None],
            "refit": ["boolean"],
            "penalty": [StrOptions({"l1", "l2", "elasticnet"})],
        }
    )

    # 初始化函数，定义LogisticRegressionCV类的参数
    def __init__(
        self,
        *,
        Cs=10,
        fit_intercept=True,
        cv=None,
        dual=False,
        penalty="l2",
        scoring=None,
        solver="lbfgs",
        tol=1e-4,
        max_iter=100,
        class_weight=None,
        n_jobs=None,
        verbose=0,
        refit=True,
        intercept_scaling=1.0,
        multi_class="deprecated",
        random_state=None,
        l1_ratios=None,
    ):
        # 设置类的各个参数值
        self.Cs = Cs
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.dual = dual
        self.penalty = penalty
        self.scoring = scoring
        self.tol = tol
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.solver = solver
        self.refit = refit
        self.intercept_scaling = intercept_scaling
        self.multi_class = multi_class
        self.random_state = random_state
        self.l1_ratios = l1_ratios

    # 应用_fit_context装饰器，使用prefer_skip_nested_validation=True参数
    @_fit_context(prefer_skip_nested_validation=True)
    def score(self, X, y, sample_weight=None, **score_params):
        """Score using the `scoring` option on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        **score_params : dict
            Parameters to pass to the `score` method of the underlying scorer.

            .. versionadded:: 1.4

        Returns
        -------
        score : float
            Score of self.predict(X) w.r.t. y.
        """
        # 检查参数的有效性
        _raise_for_params(score_params, self, "score")

        # 获取评分函数
        scoring = self._get_scorer()

        # 如果启用了路由，处理路由参数
        if _routing_enabled():
            routed_params = process_routing(
                self,
                "score",
                sample_weight=sample_weight,
                **score_params,
            )
        else:
            # 如果未启用路由，则创建一个空的路由参数对象，并设置样本权重（如果提供）
            routed_params = Bunch()
            routed_params.scorer = Bunch(score={})
            if sample_weight is not None:
                routed_params.scorer.score["sample_weight"] = sample_weight

        # 返回评分结果
        return scoring(
            self,
            X,
            y,
            **routed_params.scorer.score,
        )

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        # 创建并配置元数据路由对象，返回其实例
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)
            .add(
                splitter=self.cv,
                method_mapping=MethodMapping().add(caller="fit", callee="split"),
            )
            .add(
                scorer=self._get_scorer(),
                method_mapping=MethodMapping()
                .add(caller="score", callee="score")
                .add(caller="fit", callee="score"),
            )
        )
        return router

    def _more_tags(self):
        # 返回附加的标签信息，用于更详细的测试和验证
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }

    def _get_scorer(self):
        """Get the scorer based on the scoring method specified.
        The default scoring method is `accuracy`.
        """
        # 获取评分函数，如果未指定，则使用默认的准确率评分方法
        scoring = self.scoring or "accuracy"
        return get_scorer(scoring)
```
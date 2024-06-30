# `D:\src\scipysrc\scikit-learn\sklearn\neural_network\_multilayer_perceptron.py`

```
# 多层感知器（MLP）神经网络

# 作者：scikit-learn 开发者团队
# SPDX 许可证标识：BSD-3-Clause

# 导入警告模块，用于处理警告信息
import warnings
# 导入抽象基类元类及其方法
from abc import ABCMeta, abstractmethod
# 导入链式迭代工具
from itertools import chain
# 导入数值类型判断工具
from numbers import Integral, Real

# 导入科学计算库 NumPy
import numpy as np
# 导入 SciPy 中的优化模块
import scipy.optimize

# 从父模块中导入基本估计器和分类回归器的混合类
from ..base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    _fit_context,
    is_classifier,
)
# 导入收敛警告异常
from ..exceptions import ConvergenceWarning
# 导入度量指标：准确率和 R² 分数
from ..metrics import accuracy_score, r2_score
# 导入模型选择中的训练集拆分函数
from ..model_selection import train_test_split
# 导入标签二值化处理工具
from ..preprocessing import LabelBinarizer
# 导入通用工具函数：安全索引、随机状态检查、处理单列数组、生成批次、洗牌
from ..utils import (
    _safe_indexing,
    check_random_state,
    column_or_1d,
    gen_batches,
    shuffle,
)
# 导入参数验证工具：区间、选项、字符串选项
from ..utils._param_validation import Interval, Options, StrOptions
# 导入数学扩展工具：稀疏矩阵乘法
from ..utils.extmath import safe_sparse_dot
# 导入元估计器工具：条件可用性检查
from ..utils.metaestimators import available_if
# 导入多类别分类工具：部分拟合首次调用检查、目标类型、唯一标签
from ..utils.multiclass import (
    _check_partial_fit_first_call,
    type_of_target,
    unique_labels,
)
# 导入优化工具：检查优化结果
from ..utils.optimize import _check_optimize_result
# 导入验证工具：检查是否已拟合
from ..utils.validation import check_is_fitted

# 导入基础多层感知器（MLP）相关功能：激活函数、导数、损失函数
from ._base import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS
# 导入随机梯度下降优化器和 Adam 优化器
from ._stochastic_optimizers import AdamOptimizer, SGDOptimizer

# 定义可用的随机梯度下降优化器列表
_STOCHASTIC_SOLVERS = ["sgd", "adam"]


def _pack(coefs_, intercepts_):
    """将参数打包成一个单一向量."""
    # 将权重系数和截距项连接成一个扁平的向量
    return np.hstack([l.ravel() for l in coefs_ + intercepts_])


class BaseMultilayerPerceptron(BaseEstimator, metaclass=ABCMeta):
    """MLP 分类和回归的基类。

    警告：不建议直接使用本类，应使用派生类。

    .. versionadded:: 0.18
    """
    # 定义一个参数约束字典，用于描述机器学习模型中各参数的类型和取值范围
    _parameter_constraints: dict = {
        "hidden_layer_sizes": [
            "array-like",  # 隐藏层大小参数类型为类数组
            Interval(Integral, 1, None, closed="left"),  # 取值范围为整数大于等于1
        ],
        "activation": [StrOptions({"identity", "logistic", "tanh", "relu"})],  # 激活函数参数只能是给定集合中的字符串
        "solver": [StrOptions({"lbfgs", "sgd", "adam"})],  # 求解器参数只能是给定集合中的字符串
        "alpha": [Interval(Real, 0, None, closed="left")],  # 正则化参数alpha为实数大于等于0
        "batch_size": [
            StrOptions({"auto"}),  # 批大小参数可以是字符串"auto"
            Interval(Integral, 1, None, closed="left"),  # 或者整数大于等于1
        ],
        "learning_rate": [StrOptions({"constant", "invscaling", "adaptive"})],  # 学习率调整策略为给定集合中的字符串
        "learning_rate_init": [Interval(Real, 0, None, closed="neither")],  # 初始学习率为实数大于0
        "power_t": [Interval(Real, 0, None, closed="left")],  # 学习率衰减指数为实数大于等于0
        "max_iter": [Interval(Integral, 1, None, closed="left")],  # 最大迭代次数为正整数大于等于1
        "shuffle": ["boolean"],  # 是否洗牌参数为布尔值
        "random_state": ["random_state"],  # 随机状态参数类型为"random_state"
        "tol": [Interval(Real, 0, None, closed="left")],  # 优化算法收敛容限为实数大于等于0
        "verbose": ["verbose"],  # 是否显示详细信息参数类型为"verbose"
        "warm_start": ["boolean"],  # 是否热启动参数为布尔值
        "momentum": [Interval(Real, 0, 1, closed="both")],  # 动量参数为实数范围在0到1之间
        "nesterovs_momentum": ["boolean"],  # 是否使用Nesterov动量参数为布尔值
        "early_stopping": ["boolean"],  # 是否提前停止参数为布尔值
        "validation_fraction": [Interval(Real, 0, 1, closed="left")],  # 验证集占比参数为实数范围在0到1之间
        "beta_1": [Interval(Real, 0, 1, closed="left")],  # Adam优化器参数beta_1为实数范围在0到1之间
        "beta_2": [Interval(Real, 0, 1, closed="left")],  # Adam优化器参数beta_2为实数范围在0到1之间
        "epsilon": [Interval(Real, 0, None, closed="neither")],  # 优化器数值稳定性参数为实数大于0
        "n_iter_no_change": [
            Interval(Integral, 1, None, closed="left"),  # 连续多少次迭代未改善后停止参数为正整数大于等于1
            Options(Real, {np.inf}),  # 或者可以是正无穷
        ],
        "max_fun": [Interval(Integral, 1, None, closed="left")],  # 最大函数调用次数参数为正整数大于等于1
    }
    
    # 抽象方法，用于定义机器学习模型的初始化函数及其参数
    def __init__(
        self,
        hidden_layer_sizes,
        activation,
        solver,
        alpha,
        batch_size,
        learning_rate,
        learning_rate_init,
        power_t,
        max_iter,
        loss,
        shuffle,
        random_state,
        tol,
        verbose,
        warm_start,
        momentum,
        nesterovs_momentum,
        early_stopping,
        validation_fraction,
        beta_1,
        beta_2,
        epsilon,
        n_iter_no_change,
        max_fun,
    ):
        self.activation = activation  # 初始化激活函数参数
        self.solver = solver  # 初始化求解器参数
        self.alpha = alpha  # 初始化正则化参数alpha
        self.batch_size = batch_size  # 初始化批大小参数
        self.learning_rate = learning_rate  # 初始化学习率调整策略参数
        self.learning_rate_init = learning_rate_init  # 初始化初始学习率参数
        self.power_t = power_t  # 初始化学习率衰减指数参数
        self.max_iter = max_iter  # 初始化最大迭代次数参数
        self.loss = loss  # 初始化损失函数参数
        self.hidden_layer_sizes = hidden_layer_sizes  # 初始化隐藏层大小参数
        self.shuffle = shuffle  # 初始化是否洗牌参数
        self.random_state = random_state  # 初始化随机状态参数
        self.tol = tol  # 初始化优化算法收敛容限参数
        self.verbose = verbose  # 初始化是否显示详细信息参数
        self.warm_start = warm_start  # 初始化是否热启动参数
        self.momentum = momentum  # 初始化动量参数
        self.nesterovs_momentum = nesterovs_momentum  # 初始化是否使用Nesterov动量参数
        self.early_stopping = early_stopping  # 初始化是否提前停止参数
        self.validation_fraction = validation_fraction  # 初始化验证集占比参数
        self.beta_1 = beta_1  # 初始化Adam优化器参数beta_1
        self.beta_2 = beta_2  # 初始化Adam优化器参数beta_2
        self.epsilon = epsilon  # 初始化优化器数值稳定性参数
        self.n_iter_no_change = n_iter_no_change  # 初始化连续多少次迭代未改善后停止参数
        self.max_fun = max_fun  # 初始化最大函数调用次数参数
    def _unpack(self, packed_parameters):
        """Extract the coefficients and intercepts from packed_parameters."""
        # 遍历网络的每一层（除了输出层），解压并存储系数和截距
        for i in range(self.n_layers_ - 1):
            # 获取当前层的系数和形状，并将其重新整形为对应的形状
            start, end, shape = self._coef_indptr[i]
            self.coefs_[i] = np.reshape(packed_parameters[start:end], shape)

            # 获取当前层的截距
            start, end = self._intercept_indptr[i]
            self.intercepts_[i] = packed_parameters[start:end]

    def _forward_pass(self, activations):
        """Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.

        Parameters
        ----------
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        """
        # 获取隐藏层的激活函数
        hidden_activation = ACTIVATIONS[self.activation]

        # 遍历隐藏层
        for i in range(self.n_layers_ - 1):
            # 计算当前层的激活值，使用稀疏矩阵乘法计算
            activations[i + 1] = safe_sparse_dot(activations[i], self.coefs_[i])
            activations[i + 1] += self.intercepts_[i]

            # 对于隐藏层，应用激活函数
            if (i + 1) != (self.n_layers_ - 1):
                hidden_activation(activations[i + 1])

        # 对于输出层，应用相应的激活函数
        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activations[i + 1])

        return activations

    def _forward_pass_fast(self, X, check_input=True):
        """Predict using the trained model

        This is the same as _forward_pass but does not record the activations
        of all layers and only returns the last layer's activation.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        check_input : bool, default=True
            Perform input data validation or not.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The decision function of the samples for each class in the model.
        """
        # 如果需要，验证输入数据的格式
        if check_input:
            X = self._validate_data(X, accept_sparse=["csr", "csc"], reset=False)

        # 初始化激活值为输入数据
        activation = X

        # 执行前向传播
        hidden_activation = ACTIVATIONS[self.activation]
        for i in range(self.n_layers_ - 1):
            # 计算当前层的激活值，使用稀疏矩阵乘法计算
            activation = safe_sparse_dot(activation, self.coefs_[i])
            activation += self.intercepts_[i]

            # 对于隐藏层，应用激活函数
            if i != self.n_layers_ - 2:
                hidden_activation(activation)

        # 对于输出层，应用相应的激活函数
        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activation)

        return activation

    def _compute_loss_grad(
        self, layer, n_samples, activations, deltas, coef_grads, intercept_grads
    ):
        # 这个方法用于计算损失函数对于权重和截距的梯度
    ):
        """
        计算指定层损失相对于系数和截距的梯度。

        该函数对指定的一层进行反向传播。
        """
        coef_grads[layer] = safe_sparse_dot(activations[layer].T, deltas[layer])
        coef_grads[layer] += self.alpha * self.coefs_[layer]
        coef_grads[layer] /= n_samples

        intercept_grads[layer] = np.mean(deltas[layer], 0)

    def _loss_grad_lbfgs(
        self, packed_coef_inter, X, y, activations, deltas, coef_grads, intercept_grads
    ):
        """
        计算 MLP 损失函数及其对初始化时给定的不同参数的导数。

        返回的梯度被打包成单个向量，以便在 lbfgs 算法中使用。

        Parameters
        ----------
        packed_coef_inter : ndarray
            包含扁平化系数和截距的向量。

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            输入数据。

        y : ndarray of shape (n_samples,)
            目标值。

        activations : list, length = n_layers - 1
            列表的第 i 个元素保存第 i 层的值。

        deltas : list, length = n_layers - 1
            列表的第 i 个元素保存第 i + 1 层的激活值与反向传播误差的差异。
            具体而言，deltas 是损失相对于第 i 层中 z = wx + b 的梯度，
            其中 z 是经过激活函数之前的某一层的值。

        coef_grads : list, length = n_layers - 1
            第 i 个元素包含用于在迭代中更新第 i 层系数参数的变化量。

        intercept_grads : list, length = n_layers - 1
            第 i 个元素包含用于在迭代中更新第 i 层截距参数的变化量。

        Returns
        -------
        loss : float
        grad : array-like, shape (number of nodes of all layers,)
        """
        self._unpack(packed_coef_inter)
        loss, coef_grads, intercept_grads = self._backprop(
            X, y, activations, deltas, coef_grads, intercept_grads
        )
        grad = _pack(coef_grads, intercept_grads)
        return loss, grad
    def _initialize(self, y, layer_units, dtype):
        # set all attributes, allocate weights etc. for first call
        # 初始化函数，用于设置所有属性、分配权重等，第一次调用时执行

        # Initialize parameters
        # 初始化参数
        self.n_iter_ = 0  # 迭代次数初始化为0
        self.t_ = 0  # 计数器初始化为0
        self.n_outputs_ = y.shape[1]  # 输出的维度数量

        # Compute the number of layers
        # 计算网络层数
        self.n_layers_ = len(layer_units)

        # Output for regression
        # 回归任务的输出设置
        if not is_classifier(self):  # 如果不是分类器
            self.out_activation_ = "identity"  # 输出激活函数为恒等函数
        # Output for multi class
        # 多类分类任务的输出设置
        elif self._label_binarizer.y_type_ == "multiclass":  # 如果目标数据的类型是多类分类
            self.out_activation_ = "softmax"  # 输出激活函数为softmax
        # Output for binary class and multi-label
        # 二元分类和多标签任务的输出设置
        else:
            self.out_activation_ = "logistic"  # 输出激活函数为logistic

        # Initialize coefficient and intercept layers
        # 初始化系数和截距层
        self.coefs_ = []  # 系数列表
        self.intercepts_ = []  # 截距列表

        for i in range(self.n_layers_ - 1):
            # Initialize coefficients and intercepts for each layer
            # 初始化每一层的系数和截距
            coef_init, intercept_init = self._init_coef(
                layer_units[i], layer_units[i + 1], dtype
            )
            self.coefs_.append(coef_init)  # 将系数添加到列表中
            self.intercepts_.append(intercept_init)  # 将截距添加到列表中

        if self.solver in _STOCHASTIC_SOLVERS:
            # Initialize additional attributes for stochastic solvers
            # 对于随机求解器，初始化额外的属性
            self.loss_curve_ = []  # 损失曲线列表
            self._no_improvement_count = 0  # 无改善次数计数器初始化为0
            if self.early_stopping:
                # Initialize early stopping related attributes
                # 初始化与早停相关的属性
                self.validation_scores_ = []  # 验证分数列表
                self.best_validation_score_ = -np.inf  # 最佳验证分数初始化为负无穷
                self.best_loss_ = None  # 最佳损失初始化为None
            else:
                # Initialize best_loss for non-early-stopping cases
                # 对于不使用早停的情况，初始化最佳损失
                self.best_loss_ = np.inf  # 最佳损失初始化为正无穷
                self.validation_scores_ = None  # 验证分数置为None
                self.best_validation_score_ = None  # 最佳验证分数置为None

    def _init_coef(self, fan_in, fan_out, dtype):
        # Use the initialization method recommended by
        # Glorot et al.
        # 使用Glorot等人推荐的初始化方法

        factor = 6.0  # 默认初始化因子为6.0
        if self.activation == "logistic":
            factor = 2.0  # 如果激活函数是logistic，则初始化因子为2.0

        init_bound = np.sqrt(factor / (fan_in + fan_out))
        # 计算初始化边界

        # Generate weights and bias:
        # 生成权重和偏置：
        coef_init = self._random_state.uniform(
            -init_bound, init_bound, (fan_in, fan_out)
        )  # 生成系数，均匀分布在[-init_bound, init_bound]之间
        intercept_init = self._random_state.uniform(
            -init_bound, init_bound, fan_out
        )  # 生成截距，均匀分布在[-init_bound, init_bound]之间

        coef_init = coef_init.astype(dtype, copy=False)  # 将系数转换为指定的数据类型
        intercept_init = intercept_init.astype(dtype, copy=False)  # 将截距转换为指定的数据类型

        return coef_init, intercept_init  # 返回系数和截距的初始化结果
    def _fit(self, X, y, incremental=False):
        # 确保 self.hidden_layer_sizes 是一个列表
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError(
                "hidden_layer_sizes must be > 0, got %s." % hidden_layer_sizes
            )
        
        # 判断是否是首次训练或者不支持增量训练
        first_pass = not hasattr(self, "coefs_") or (
            not self.warm_start and not incremental
        )

        # 验证输入数据，并根据需要重置
        X, y = self._validate_input(X, y, incremental, reset=first_pass)

        n_samples, n_features = X.shape

        # 确保 y 是二维数组
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs_ = y.shape[1]

        # 确定每一层的单元数，包括输入层、隐藏层和输出层
        layer_units = [n_features] + hidden_layer_sizes + [self.n_outputs_]

        # 检查随机状态
        self._random_state = check_random_state(self.random_state)

        if first_pass:
            # 第一次训练模型
            self._initialize(y, layer_units, X.dtype)

        # 初始化列表
        activations = [X] + [None] * (len(layer_units) - 1)
        deltas = [None] * (len(activations) - 1)

        # 初始化系数梯度
        coef_grads = [
            np.empty((n_fan_in_, n_fan_out_), dtype=X.dtype)
            for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])
        ]

        # 初始化截距梯度
        intercept_grads = [
            np.empty(n_fan_out_, dtype=X.dtype) for n_fan_out_ in layer_units[1:]
        ]

        # 运行随机优化求解器
        if self.solver in _STOCHASTIC_SOLVERS:
            self._fit_stochastic(
                X,
                y,
                activations,
                deltas,
                coef_grads,
                intercept_grads,
                layer_units,
                incremental,
            )

        # 运行LBFGS求解器
        elif self.solver == "lbfgs":
            self._fit_lbfgs(
                X, y, activations, deltas, coef_grads, intercept_grads, layer_units
            )

        # 验证参数权重
        weights = chain(self.coefs_, self.intercepts_)
        if not all(np.isfinite(w).all() for w in weights):
            raise ValueError(
                "Solver produced non-finite parameter weights. The input data may"
                " contain large values and need to be preprocessed."
            )

        return self

    def _fit_lbfgs(
        self, X, y, activations, deltas, coef_grads, intercept_grads, layer_units
    ):
        # LBFGS求解器的具体实现，用于训练神经网络模型
        # 存储参数的元信息
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0

        # 保存系数的大小和索引，以便更快地解压缩
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        # 保存截距的大小和索引，以便更快地解压缩
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        # 运行 LBFGS 优化器
        packed_coef_inter = _pack(self.coefs_, self.intercepts_)

        if self.verbose is True or self.verbose >= 1:
            iprint = 1
        else:
            iprint = -1

        opt_res = scipy.optimize.minimize(
            self._loss_grad_lbfgs,
            packed_coef_inter,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxfun": self.max_fun,
                "maxiter": self.max_iter,
                "iprint": iprint,
                "gtol": self.tol,
            },
            args=(X, y, activations, deltas, coef_grads, intercept_grads),
        )
        self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)
        self.loss_ = opt_res.fun
        self._unpack(opt_res.x)
    # 应用装饰器 `_fit_context`，设置参数 `prefer_skip_nested_validation=True`，并调用内部方法 `fit`
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns a trained MLP model.
        """
        # 调用内部方法 `_fit`，传递参数 `incremental=False`，执行模型训练
        return self._fit(X, y, incremental=False)

    # 检查所选的求解器是否为随机优化器，若不是则抛出属性错误异常
    def _check_solver(self):
        if self.solver not in _STOCHASTIC_SOLVERS:
            raise AttributeError(
                "partial_fit is only available for stochastic"
                " optimizers. %s is not stochastic." % self.solver
            )
        # 如果求解器在随机优化器列表中，则返回 True
        return True
class MLPClassifier(ClassifierMixin, BaseMultilayerPerceptron):
    """Multi-layer Perceptron classifier.

    This model optimizes the log-loss function using LBFGS or stochastic
    gradient descent.

    .. versionadded:: 0.18

    Parameters
    ----------
    hidden_layer_sizes : array-like of shape(n_layers - 2,), default=(100,)
        The ith element represents the number of neurons in the ith
        hidden layer.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layer.

        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x

        - 'logistic', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).

        - 'tanh', the hyperbolic tan function,
          returns f(x) = tanh(x).

        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)

    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization.

        - 'lbfgs' is an optimizer in the family of quasi-Newton methods.

        - 'sgd' refers to stochastic gradient descent.

        - 'adam' refers to a stochastic gradient-based optimizer proposed
          by Kingma, Diederik, and Jimmy Ba

        For a comparison between Adam optimizer and SGD, see
        :ref:`sphx_glr_auto_examples_neural_networks_plot_mlp_training_curves.py`.

        Note: The default solver 'adam' works pretty well on relatively
        large datasets (with thousands of training samples or more) in terms of
        both training time and validation score.
        For small datasets, however, 'lbfgs' can converge faster and perform
        better.

    alpha : float, default=0.0001
        Strength of the L2 regularization term. The L2 regularization term
        is divided by the sample size when added to the loss.

        For an example usage and visualization of varying regularization, see
        :ref:`sphx_glr_auto_examples_neural_networks_plot_mlp_alpha.py`.

    batch_size : int, default='auto'
        Size of minibatches for stochastic optimizers.
        If the solver is 'lbfgs', the classifier will not use minibatch.
        When set to "auto", `batch_size=min(200, n_samples)`.

    """

    # MLPClassifier 类定义了一个多层感知器分类器，继承自 ClassifierMixin 和 BaseMultilayerPerceptron
    # 这个模型使用 LBFGS 或者随机梯度下降来优化对数损失函数
    # 版本 0.18 中首次引入

    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto'):
        # 初始化函数，定义了多层感知器分类器的参数

        # 隐藏层大小，表示每个隐藏层的神经元数目，默认为 (100,)
        self.hidden_layer_sizes = hidden_layer_sizes

        # 激活函数，控制每个隐藏层所使用的激活函数，默认为 'relu'
        self.activation = activation

        # 权重优化器，决定了在训练过程中如何更新权重，默认为 'adam'
        self.solver = solver

        # L2 正则化强度，默认为 0.0001
        self.alpha = alpha

        # 批量大小，用于随机优化器中的小批量训练，默认为 'auto'
        self.batch_size = batch_size
    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        # 学习率更新的策略，可以是'constant'（常数学习率）、'invscaling'（逆缩放学习率）、'adaptive'（自适应学习率）
        - 'constant' 是由'learning_rate_init'指定的恒定学习率。
        - 'invscaling' 通过逐步减小学习率来更新，每次步长 't' 使用逆缩放指数 'power_t'。
          有效学习率 = learning_rate_init / pow(t, power_t)
        - 'adaptive' 在训练损失持续下降时保持学习率恒定为 'learning_rate_init'。
          每当连续两个 epoch 未能至少按 tol 减少训练损失，或者如果 'early_stopping' 打开，则未能至少按 tol 增加验证分数时，
          当前学习率被除以 5。
        只在 ``solver='sgd'`` 时使用。

    learning_rate_init : float, default=0.001
        # 初始学习率，控制权重更新的步长。仅在 solver='sgd' 或 'adam' 时使用。

    power_t : float, default=0.5
        # 逆缩放学习率的指数。当 learning_rate 设置为 'invscaling' 时使用。仅在 solver='sgd' 时使用。

    max_iter : int, default=200
        # 最大迭代次数。解算器迭代直到收敛（由 'tol' 决定）或达到此迭代次数。对于随机解算器（'sgd'、'adam'），这确定了迭代的次数
        （每个数据点将被使用多少次），而不是梯度步数。

    shuffle : bool, default=True
        # 每次迭代是否对样本进行洗牌。仅在 solver='sgd' 或 'adam' 时使用。

    random_state : int, RandomState instance, default=None
        # 确定权重和偏差初始化的随机数生成，如果使用了早停法则确定训练-测试分割，以及当 solver='sgd' 或 'adam' 时的批量采样。
        传递一个整数以确保多个函数调用之间的可重复结果。
        参见 :term:`术语表 <random_state>`。

    tol : float, default=1e-4
        # 优化的容差。当损失或分数连续 n_iter_no_change 次迭代中没有至少按 tol 改善时，
        除非 learning_rate 设置为 'adaptive'，否则认为收敛已达到，训练停止。

    verbose : bool, default=False
        # 是否将进度消息打印到标准输出。

    warm_start : bool, default=False
        # 如果设置为 True，则重复利用前一次拟合调用的解决方案作为初始化；否则，只是擦除前一解决方案。
        参见 :term:`术语表 <warm_start>`。

    momentum : float, default=0.9
        # 梯度下降更新的动量。应该在 0 到 1 之间。仅在 solver='sgd' 时使用。
    nesterovs_momentum : bool, default=True
        Whether to use Nesterov's momentum. Only used when solver='sgd' and
        momentum > 0.
        是否使用 Nesterov 动量。仅在 solver='sgd' 且 momentum > 0 时使用。

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to true, it will automatically set
        aside 10% of training data as validation and terminate training when
        validation score is not improving by at least ``tol`` for
        ``n_iter_no_change`` consecutive epochs. The split is stratified,
        except in a multilabel setting.
        如果设置为 True，则使用提前停止来在验证分数不再提高时终止训练。它会自动将训练数据的10%设置为验证集，并在验证分数在连续 ``n_iter_no_change`` 个 epoch 中至少没有因 ``tol`` 而改善时终止训练。除了在多标签设置中，拆分是分层的。

        If early stopping is False, then the training stops when the training
        loss does not improve by more than tol for n_iter_no_change consecutive
        passes over the training set.
        如果 early stopping 为 False，则当训练损失在连续 n_iter_no_change 次通过训练集时不再改善超过 tol 时，训练停止。

        Only effective when solver='sgd' or 'adam'.
        仅在 solver='sgd' 或 'adam' 时有效。

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        用于提前停止的训练数据比例，作为验证集设置。必须在 0 和 1 之间。
        Only used if early_stopping is True.
        仅在 early_stopping 为 True 时使用。

    beta_1 : float, default=0.9
        Exponential decay rate for estimates of first moment vector in adam,
        should be in [0, 1). Only used when solver='adam'.
        在 adam 中估计第一时刻向量的指数衰减率，应在 [0, 1) 范围内。
        仅在 solver='adam' 时使用。

    beta_2 : float, default=0.999
        Exponential decay rate for estimates of second moment vector in adam,
        should be in [0, 1). Only used when solver='adam'.
        在 adam 中估计第二时刻向量的指数衰减率，应在 [0, 1) 范围内。
        仅在 solver='adam' 时使用。

    epsilon : float, default=1e-8
        Value for numerical stability in adam. Only used when solver='adam'.
        adam 中用于数值稳定性的值。仅在 solver='adam' 时使用。

    n_iter_no_change : int, default=10
        Maximum number of epochs to not meet ``tol`` improvement.
        Only effective when solver='sgd' or 'adam'.
        不满足 ``tol`` 改进的最大 epoch 数。
        仅在 solver='sgd' 或 'adam' 时有效。

        .. versionadded:: 0.20

    max_fun : int, default=15000
        Only used when solver='lbfgs'. Maximum number of loss function calls.
        The solver iterates until convergence (determined by 'tol'), number
        of iterations reaches max_iter, or this number of loss function calls.
        Note that number of loss function calls will be greater than or equal
        to the number of iterations for the `MLPClassifier`.
        仅在 solver='lbfgs' 时使用。损失函数调用的最大数量。
        解算器迭代直到收敛（由 'tol' 确定），迭代次数达到 max_iter，或者损失函数调用达到这个数量。
        请注意，损失函数调用的数量将大于或等于 `MLPClassifier` 的迭代次数。

        .. versionadded:: 0.22

    Attributes
    ----------
    classes_ : ndarray or list of ndarray of shape (n_classes,)
        Class labels for each output.
        每个输出的类标签。

    loss_ : float
        The current loss computed with the loss function.
        使用损失函数计算的当前损失。

    best_loss_ : float or None
        The minimum loss reached by the solver throughout fitting.
        If `early_stopping=True`, this attribute is set to `None`. Refer to
        the `best_validation_score_` fitted attribute instead.
        求解器在整个拟合过程中达到的最小损失。
        如果 `early_stopping=True`，则此属性设置为 `None`。请参阅 `best_validation_score_` 拟合属性。

    loss_curve_ : list of shape (`n_iter_`,)
        The ith element in the list represents the loss at the ith iteration.
        列表中的第 i 个元素表示第 i 次迭代时的损失。

    validation_scores_ : list of shape (`n_iter_`,) or None
        The score at each iteration on a held-out validation set. The score
        reported is the accuracy score. Only available if `early_stopping=True`,
        otherwise the attribute is set to `None`.
        在保留验证集上的每次迭代中的得分。报告的分数是准确性分数。
        仅在 `early_stopping=True` 时可用，否则该属性设置为 `None`。
    best_validation_score_ : float or None
        # 最佳验证分数（如准确率）触发了早期停止。仅在 `early_stopping=True` 时可用，否则设置为 `None`。

    t_ : int
        # 拟合过程中求解器看到的训练样本数。

    coefs_ : list of shape (n_layers - 1,)
        # 列表中第 i 个元素表示第 i 层对应的权重矩阵。

    intercepts_ : list of shape (n_layers - 1,)
        # 列表中第 i 个元素表示第 i + 1 层对应的偏置向量。

    n_features_in_ : int
        # 在 `fit` 过程中看到的特征数。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 `fit` 过程中看到的特征名称。仅当 `X` 的特征名全为字符串时定义。

        .. versionadded:: 1.0

    n_iter_ : int
        # 求解器运行的迭代次数。

    n_layers_ : int
        # 层数。

    n_outputs_ : int
        # 输出数量。

    out_activation_ : str
        # 输出激活函数的名称。

    See Also
    --------
    MLPRegressor : 多层感知机回归器。
    BernoulliRBM : 伯努利受限玻尔兹曼机（RBM）。

    Notes
    -----
    MLPClassifier 在每个时间步骤都进行迭代训练，因为需要计算损失函数相对于模型参数的偏导数来更新参数。

    它也可以在损失函数中加入正则化项，以缩小模型参数，防止过拟合。

    该实现可以处理表示为浮点数值的密集 numpy 数组或稀疏 scipy 数组的数据。

    References
    ----------
    Hinton, Geoffrey E. "Connectionist learning procedures."
    Artificial intelligence 40.1 (1989): 185-234.

    Glorot, Xavier, and Yoshua Bengio.
    "Understanding the difficulty of training deep feedforward neural networks."
    International Conference on Artificial Intelligence and Statistics. 2010.

    :arxiv:`He, Kaiming, et al (2015). "Delving deep into rectifiers:
    Surpassing human-level performance on imagenet classification." <1502.01852>`

    :arxiv:`Kingma, Diederik, and Jimmy Ba (2014)
    "Adam: A method for stochastic optimization." <1412.6980>`

    Examples
    --------
    >>> from sklearn.neural_network import MLPClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, random_state=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
    ...                                                     random_state=1)
    >>> clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    >>> clf.predict_proba(X_test[:1])
    array([[0.038..., 0.961...]])
    # 使用已训练好的分类器预测输入数据集的前5个样本的类别标签
    clf.predict(X_test[:5, :])

    # 计算分类器在测试数据集上的准确率得分
    clf.score(X_test, y_test)
    0.8...

    """
    # 神经网络分类器的初始化方法，继承父类的初始化方法
    def __init__(
        self,
        # 隐藏层神经元的数量和布局，默认为一个包含100个神经元的元组
        hidden_layer_sizes=(100,),
        # 激活函数的类型，默认为ReLU
        activation="relu",
        # 优化器的选择，默认为Adam优化器
        *,
        solver="adam",
        # 正则化参数alpha，默认为0.0001
        alpha=0.0001,
        # 批量大小，默认为"auto"，根据数据自动选择大小
        batch_size="auto",
        # 学习率类型，默认为常数学习率
        learning_rate="constant",
        # 初始学习率，默认为0.001
        learning_rate_init=0.001,
        # 学习率衰减的指数，默认为0.5
        power_t=0.5,
        # 最大迭代次数，默认为200
        max_iter=200,
        # 是否在每次迭代前打乱数据，默认为True
        shuffle=True,
        # 随机数种子，默认为None
        random_state=None,
        # 迭代停止的容差，默认为1e-4
        tol=1e-4,
        # 是否输出详细的训练信息，默认为False
        verbose=False,
        # 是否热启动，默认为False，即每次重新开始训练
        warm_start=False,
        # 动量参数，默认为0.9
        momentum=0.9,
        # 是否使用Nesterov动量，默认为True
        nesterovs_momentum=True,
        # 是否启用早停策略，默认为False
        early_stopping=False,
        # 用于早停的验证集比例，默认为0.1
        validation_fraction=0.1,
        # Adam优化器的beta_1参数，默认为0.9
        beta_1=0.9,
        # Adam优化器的beta_2参数，默认为0.999
        beta_2=0.999,
        # Adam优化器的数值稳定性参数，默认为1e-8
        epsilon=1e-8,
        # 连续多少次迭代不改善就停止训练，默认为10
        n_iter_no_change=10,
        # 最大函数评估次数，默认为15000
        max_fun=15000,
    ):
        # 调用父类的初始化方法，设置神经网络分类器的各种参数
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            loss="log_loss",  # 损失函数选择对数损失
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )
    # 验证输入数据的有效性，并作出相应处理
    def _validate_input(self, X, y, incremental, reset):
        # 调用_validate_data方法验证输入的数据X和y，并进行预处理
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc"],  # 接受稀疏矩阵的类型
            multi_output=True,  # 支持多输出
            dtype=(np.float64, np.float32),  # 数据类型限定为np.float64或np.float32
            reset=reset,  # 标志是否重置数据处理
        )
        # 如果y的维度为2并且第二维的长度为1，则调用column_or_1d函数处理
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)

        # 动作矩阵用于根据可能的组合执行操作:
        # 当 incremental == True 并且 classes_ 未定义时，已由 _check_partial_fit_first_call
        # 在下面的 _partial_fit 中调用进行了检查。
        # 各种情况已经分组在下面的相应 if 块中。
        #
        # incremental warm_start classes_ def  action
        #    0            0         0        定义 classes_
        #    0            1         0        定义 classes_
        #    0            0         1        重新定义 classes_
        #
        #    0            1         1        检查兼容性 warm_start
        #    1            1         1        检查兼容性 warm_start
        #
        #    1            0         1        检查兼容性上一次拟合结果
        #
        # 注意这里依赖于短路运算，因此第二部分或部分意味着 classes_ 已定义。
        if (not hasattr(self, "classes_")) or (not self.warm_start and not incremental):
            # 如果self中没有属性classes_，或者不是warm_start且incremental也不是True
            # 则初始化_label_binarizer为LabelBinarizer对象，并使用y进行fit
            self._label_binarizer = LabelBinarizer()
            self._label_binarizer.fit(y)
            self.classes_ = self._label_binarizer.classes_
        else:
            # 否则，计算y中的唯一标签集合
            classes = unique_labels(y)
            if self.warm_start:
                # 如果是warm_start模式，则验证classes与self.classes_是否相同
                if set(classes) != set(self.classes_):
                    raise ValueError(
                        "warm_start 只能在 `y` 与之前调用 fit 时相同类别时使用。之前的类别为 "
                        f"{self.classes_}，当前 `y` 的类别为 {classes}"
                    )
            elif len(np.setdiff1d(classes, self.classes_, assume_unique=True)):
                # 否则，如果y中存在不在self.classes_中的类别，则引发异常
                raise ValueError(
                    "`y` 中包含不在 `self.classes_` 中的类别。"
                    f"`self.classes_` 包含 {self.classes_}。'y' 包含 {classes}。"
                )

        # 将_label_binarizer对y进行变换，并将结果转换为bool类型以防止在处理float32数据时向上转型
        y = self._label_binarizer.transform(y).astype(bool)
        # 返回处理后的X和y
        return X, y

    # 使用多层感知机分类器进行预测
    def predict(self, X):
        """Predict using the multi-layer perceptron classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            输入数据.

        Returns
        -------
        y : ndarray, shape (n_samples,) or (n_samples, n_classes)
            预测的类别.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)
        # 调用_predict方法进行预测
        return self._predict(X)
    # 私有预测方法，可选的输入验证
    def _predict(self, X, check_input=True):
        # 调用快速前向传播方法获取预测结果
        y_pred = self._forward_pass_fast(X, check_input=check_input)

        # 如果只有一个输出，将预测结果展平
        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()

        # 使用逆转换器将预测结果转换为原始标签
        return self._label_binarizer.inverse_transform(y_pred)

    # 私有评分方法，没有输入验证
    def _score(self, X, y):
        # 输入验证会移除特征名称，因此这里禁用它
        return accuracy_score(y, self._predict(X, check_input=False))

    # 部分拟合方法，用于在给定数据上进行单次迭代更新模型
    @available_if(lambda est: est._check_solver())
    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, classes=None):
        """Update the model with a single iteration over the given data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target values.

        classes : array of shape (n_classes,), default=None
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.

        Returns
        -------
        self : object
            Trained MLP model.
        """
        # 如果是部分拟合的第一次调用，则初始化标签二值化器
        if _check_partial_fit_first_call(self, classes):
            self._label_binarizer = LabelBinarizer()
            # 如果目标值类型为多标签，对标签二值化器进行拟合
            if type_of_target(y).startswith("multilabel"):
                self._label_binarizer.fit(y)
            else:
                # 否则，对类别进行拟合
                self._label_binarizer.fit(classes)

        # 调用拟合方法进行模型更新，增量设置为True
        return self._fit(X, y, incremental=True)

    # 返回对数概率估计值
    def predict_log_proba(self, X):
        """Return the log of probability estimates.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        log_y_prob : ndarray of shape (n_samples, n_classes)
            The predicted log-probability of the sample for each class
            in the model, where classes are ordered as they are in
            `self.classes_`. Equivalent to `log(predict_proba(X))`.
        """
        # 获取概率估计值并返回其对数
        y_prob = self.predict_proba(X)
        return np.log(y_prob, out=y_prob)
    # 预测每个类别的概率值。

    # 检查模型是否已经拟合
    check_is_fitted(self)

    # 使用快速的前向传播计算预测值
    y_pred = self._forward_pass_fast(X)

    # 如果模型输出只有一个，将预测结果展平
    if self.n_outputs_ == 1:
        y_pred = y_pred.ravel()

    # 如果预测结果是一维的，将其转换成二维数组表示概率，其中一列为类别为0的概率，另一列为类别为1的概率
    if y_pred.ndim == 1:
        return np.vstack([1 - y_pred, y_pred]).T
    else:
        # 如果预测结果已经是二维的（每行对应一个样本，每列对应一个类别），直接返回预测结果
        return y_pred

    # 返回预测的概率值数组，形状为 (样本数, 类别数)，类别顺序与 self.classes_ 中一致



    # 返回额外的标签信息，表明模型支持多标签输出

    return {"multilabel": True}
class MLPRegressor(RegressorMixin, BaseMultilayerPerceptron):
    """Multi-layer Perceptron regressor.

    This model optimizes the squared error using LBFGS or stochastic gradient
    descent.

    .. versionadded:: 0.18

    Parameters
    ----------
    hidden_layer_sizes : array-like of shape(n_layers - 2,), default=(100,)
        The ith element represents the number of neurons in the ith
        hidden layer.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layer.

        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x

        - 'logistic', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).

        - 'tanh', the hyperbolic tan function,
          returns f(x) = tanh(x).

        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)

    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization.

        - 'lbfgs' is an optimizer in the family of quasi-Newton methods.

        - 'sgd' refers to stochastic gradient descent.

        - 'adam' refers to a stochastic gradient-based optimizer proposed by
          Kingma, Diederik, and Jimmy Ba

        For a comparison between Adam optimizer and SGD, see
        :ref:`sphx_glr_auto_examples_neural_networks_plot_mlp_training_curves.py`.

        Note: The default solver 'adam' works pretty well on relatively
        large datasets (with thousands of training samples or more) in terms of
        both training time and validation score.
        For small datasets, however, 'lbfgs' can converge faster and perform
        better.

    alpha : float, default=0.0001
        Strength of the L2 regularization term. The L2 regularization term
        is divided by the sample size when added to the loss.

    batch_size : int, default='auto'
        Size of minibatches for stochastic optimizers.
        If the solver is 'lbfgs', the regressor will not use minibatch.
        When set to "auto", `batch_size=min(200, n_samples)`.

    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates.

        - 'constant' is a constant learning rate given by
          'learning_rate_init'.

        - 'invscaling' gradually decreases the learning rate ``learning_rate_``
          at each time step 't' using an inverse scaling exponent of 'power_t'.
          effective_learning_rate = learning_rate_init / pow(t, power_t)

        - 'adaptive' keeps the learning rate constant to
          'learning_rate_init' as long as training loss keeps decreasing.
          Each time two consecutive epochs fail to decrease training loss by at
          least tol, or fail to increase validation score by at least tol if
          'early_stopping' is on, the current learning rate is divided by 5.

        Only used when solver='sgd'.
    """

    # 定义多层感知机回归器类，继承自RegressorMixin和BaseMultilayerPerceptron
    def __init__(self, hidden_layer_sizes=(100,), activation="relu",
                 solver="adam", alpha=0.0001, batch_size="auto",
                 learning_rate="constant"):
        # 调用父类的初始化方法
        super().__init__()
        # 设置隐藏层大小参数
        self.hidden_layer_sizes = hidden_layer_sizes
        # 设置激活函数参数
        self.activation = activation
        # 设置优化器参数
        self.solver = solver
        # 设置L2正则化强度参数
        self.alpha = alpha
        # 设置批量大小参数
        self.batch_size = batch_size
        # 设置学习率调度策略参数
        self.learning_rate = learning_rate
    learning_rate_init : float, default=0.001
        初始学习率。控制权重更新的步长。仅在 solver='sgd' 或 'adam' 时使用。

    power_t : float, default=0.5
        逆缩放学习率的指数。用于更新有效学习率，当 learning_rate 设置为 'invscaling' 时使用。仅在 solver='sgd' 时使用。

    max_iter : int, default=200
        最大迭代次数。求解器迭代直到收敛（由 'tol' 决定）或达到此迭代次数。对于随机求解器（'sgd'、'adam'），这确定了迭代次数而不是梯度步数。

    shuffle : bool, default=True
        是否在每次迭代中对样本进行洗牌。仅在 solver='sgd' 或 'adam' 时使用。

    random_state : int, RandomState instance, default=None
        确定权重和偏置初始化的随机数生成，如果使用了早期停止，则确定训练-测试分割和求解器为 'sgd' 或 'adam' 时的批量采样。
        传入一个整数以确保在多次函数调用中获得可重复的结果。
        参见：词汇表中的“随机状态”。

    tol : float, default=1e-4
        优化的容差。当损失或得分连续 ``n_iter_no_change`` 次迭代未改善至少 ``tol`` 时，除非 ``learning_rate`` 设置为 'adaptive'，否则认为达到收敛并停止训练。

    verbose : bool, default=False
        是否向标准输出打印进度消息。

    warm_start : bool, default=False
        当设置为 True 时，重用上一次调用 fit 的解作为初始化，否则只擦除之前的解。参见：术语表中的“热启动”。

    momentum : float, default=0.9
        梯度下降更新的动量。应在 0 到 1 之间。仅在 solver='sgd' 时使用。

    nesterovs_momentum : bool, default=True
        是否使用 Nesterov 动量。仅在 solver='sgd' 并且 momentum > 0 时使用。

    early_stopping : bool, default=False
        是否使用早期停止来在验证分数未改善时终止训练。如果设置为 True，则会自动将 ``validation_fraction`` 的训练数据作为验证集，并在连续的迭代中验证分数至少未改善 ``tol``。
        仅在 solver='sgd' 或 'adam' 时有效。

    validation_fraction : float, default=0.1
        用作早期停止的验证集的训练数据比例。必须介于 0 和 1 之间。
        仅在 early_stopping 设置为 True 时使用。
    beta_1 : float, default=0.9
        # Adam优化器中一阶矩估计的指数衰减率，应在[0, 1)范围内。仅在solver='adam'时使用。

    beta_2 : float, default=0.999
        # Adam优化器中二阶矩估计的指数衰减率，应在[0, 1)范围内。仅在solver='adam'时使用。

    epsilon : float, default=1e-8
        # 数值稳定性参数，仅在solver='adam'时使用。

    n_iter_no_change : int, default=10
        # 最大允许的未达到tol改善的迭代次数。仅在solver='sgd'或'adam'时有效。
        # 版本新增：0.20

    max_fun : int, default=15000
        # 仅在solver='lbfgs'时使用。最大的函数调用次数。
        # 求解器迭代直到收敛（由tol确定）、达到max_iter次数，或达到函数调用次数。
        # 注意：函数调用次数将大于等于MLPRegressor的迭代次数。
        # 版本新增：0.22

    Attributes
    ----------
    loss_ : float
        # 使用损失函数计算的当前损失值。

    best_loss_ : float
        # 求解器在拟合过程中达到的最小损失值。
        # 如果early_stopping=True，则此属性设置为None。参考best_validation_score_属性。
        # 仅在solver='sgd'或'adam'时访问。

    loss_curve_ : list of shape (`n_iter_`,)
        # 每个训练步骤结束时评估的损失值列表。
        # 列表中的第i个元素表示第i次迭代时的损失值。
        # 仅在solver='sgd'或'adam'时访问。

    validation_scores_ : list of shape (`n_iter_`,) or None
        # 在保留的验证集上每次迭代的评分。报告的评分是R2分数。
        # 仅在early_stopping=True时可用，否则设置为None。
        # 仅在solver='sgd'或'adam'时访问。

    best_validation_score_ : float or None
        # 触发early stopping的最佳验证分数（即R2分数）。
        # 仅在early_stopping=True时可用，否则设置为None。
        # 仅在solver='sgd'或'adam'时访问。

    t_ : int
        # 拟合过程中求解器看到的训练样本数量。
        # 在数学上等于`n_iters * X.shape[0]`，表示时间步长，由优化器的学习率调度器使用。

    coefs_ : list of shape (n_layers - 1,)
        # 列表中的第i个元素表示第i层对应的权重矩阵。

    intercepts_ : list of shape (n_layers - 1,)
        # 列表中的第i个元素表示第i+1层对应的偏置向量。

    n_features_in_ : int
        # 在拟合期间观察到的特征数量。
        # 版本新增：0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0
        特征名称列表，形状为 (`n_features_in_`,)，记录了在 `fit` 过程中观察到的特征名称。仅当 `X` 的特征名称全为字符串时定义。

    n_iter_ : int
        The number of iterations the solver has run.
        求解器运行的迭代次数。

    n_layers_ : int
        Number of layers.
        神经网络的层数。

    n_outputs_ : int
        Number of outputs.
        输出的数量。

    out_activation_ : str
        Name of the output activation function.
        输出层激活函数的名称。

    See Also
    --------
    BernoulliRBM : Bernoulli Restricted Boltzmann Machine (RBM).
        伯努利受限玻尔兹曼机，一种随机生成的神经网络。

    MLPClassifier : Multi-layer Perceptron classifier.
        多层感知器分类器，用于分类任务的神经网络模型。

    sklearn.linear_model.SGDRegressor : Linear model fitted by minimizing
        a regularized empirical loss with SGD.
        使用随机梯度下降最小化正则化经验损失的线性回归模型。

    Notes
    -----
    MLPRegressor trains iteratively since at each time step
    the partial derivatives of the loss function with respect to the model
    parameters are computed to update the parameters.

    It can also have a regularization term added to the loss function
    that shrinks model parameters to prevent overfitting.

    This implementation works with data represented as dense and sparse numpy
    arrays of floating point values.
    MLPRegressor 是一个迭代训练的模型，因为在每个时间步骤中，
    针对模型参数计算损失函数的偏导数来更新参数。

    还可以在损失函数中添加正则化项，以缩小模型参数以防止过拟合。

    此实现支持以密集和稀疏的浮点数值 numpy 数组表示的数据。

    References
    ----------
    Hinton, Geoffrey E. "Connectionist learning procedures."
    Artificial intelligence 40.1 (1989): 185-234.
    Hinton, Geoffrey E. "连接主义学习过程。"
    人工智能 40.1 (1989): 185-234。

    Glorot, Xavier, and Yoshua Bengio.
    "Understanding the difficulty of training deep feedforward neural networks."
    International Conference on Artificial Intelligence and Statistics. 2010.
    Glorot, Xavier 和 Yoshua Bengio。
    "理解训练深度前馈神经网络的难度。"
    人工智能与统计学国际会议。2010 年。

    :arxiv:`He, Kaiming, et al (2015). "Delving deep into rectifiers:
    Surpassing human-level performance on imagenet classification." <1502.01852>`
    :arxiv:`He, Kaiming 等人 (2015)。"深入探讨整流器：超越图像分类中人类水平的性能。" <1502.01852>`

    :arxiv:`Kingma, Diederik, and Jimmy Ba (2014)
    "Adam: A method for stochastic optimization." <1412.6980>`
    :arxiv:`Kingma, Diederik 和 Jimmy Ba (2014)
    "Adam：一种随机优化方法。" <1412.6980>`

    Examples
    --------
    >>> from sklearn.neural_network import MLPRegressor
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_regression(n_samples=200, random_state=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=1)
    >>> regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
    >>> regr.predict(X_test[:2])
    array([-0.9..., -7.1...])
    >>> regr.score(X_test, y_test)
    0.4...
    """
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        *,
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
    ):
        # 调用父类初始化函数，设置神经网络的各项参数
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            loss="squared_error",
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )

    def predict(self, X):
        """Predict using the multi-layer perceptron model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        # 确保模型已经拟合过数据
        check_is_fitted(self)
        return self._predict(X)

    def _predict(self, X, check_input=True):
        """Private predict method with optional input validation"""
        # 快速前向传播预测数据
        y_pred = self._forward_pass_fast(X, check_input=check_input)
        if y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred

    def _score(self, X, y):
        """Private score method without input validation"""
        # 不进行输入验证，因为输入验证会移除特征名
        y_pred = self._predict(X, check_input=False)
        return r2_score(y, y_pred)

    def _validate_input(self, X, y, incremental, reset):
        # 验证输入数据，并进行必要的类型转换和验证
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc"],
            multi_output=True,
            y_numeric=True,
            dtype=(np.float64, np.float32),
            reset=reset,
        )
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)
        return X, y

    @available_if(lambda est: est._check_solver)
    @_fit_context(prefer_skip_nested_validation=True)
    # 定义一个方法 `partial_fit`，用于在给定数据上进行单次迭代更新模型。

    """
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        输入数据。

    y : ndarray of shape (n_samples,)
        目标数值。

    Returns
    -------
    self : object
        训练好的 MLP 模型。
    """

    # 调用内部方法 `_fit`，传入参数 X, y 和 incremental=True，执行增量训练。
    return self._fit(X, y, incremental=True)
```
# `D:\src\scipysrc\scikit-learn\sklearn\neural_network\_rbm.py`

```
# 导入必要的库和模块
import time  # 时间模块，用于性能计时
from numbers import Integral, Real  # 导入用于数值类型检查的模块

import numpy as np  # 数值计算库
import scipy.sparse as sp  # 稀疏矩阵处理库
from scipy.special import expit  # logistic 函数，用于计算 S 型函数

from ..base import (
    BaseEstimator,  # scikit-learn 基础估计器
    ClassNamePrefixFeaturesOutMixin,  # 类名前缀特征输出混合类
    TransformerMixin,  # 转换器混合类
    _fit_context,  # 拟合上下文，辅助类
)
from ..utils import check_random_state, gen_even_slices  # 随机状态检查、生成均匀切片函数
from ..utils._param_validation import Interval  # 参数验证模块中的区间类
from ..utils.extmath import safe_sparse_dot  # 安全稀疏矩阵乘法
from ..utils.validation import check_is_fitted  # 检查是否拟合函数已应用

# 定义 BernoulliRBM 类，继承自 ClassNamePrefixFeaturesOutMixin、TransformerMixin、BaseEstimator
class BernoulliRBM(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Bernoulli Restricted Boltzmann Machine (RBM).

    二值可见单元和二值隐藏单元的受限玻尔兹曼机。参数使用随机最大似然估计（SML）
    或称为持续对比散度（PCD）[2] 来估计。

    该实现的时间复杂度为 ``O(d ** 2)``，其中 d ~ n_features ~ n_components。

    详细信息请参阅 :ref:`用户指南 <rbm>`。

    Parameters
    ----------
    n_components : int, default=256
        隐藏单元的数量。

    learning_rate : float, default=0.1
        权重更新的学习率。强烈建议调整此超参数。合理的值在 10**[0., -3.] 范围内。

    batch_size : int, default=10
        每个小批量的示例数量。

    n_iter : int, default=10
        在训练过程中执行的迭代/扫描次数。

    verbose : int, default=0
        冗余级别。默认值为零，表示静默模式。值的范围是 [0, inf]。

    random_state : int, RandomState instance or None, default=None
        确定以下随机数生成的内容：

        - 从可见和隐藏层的 Gibbs 采样。
        
        - 在拟合过程中，初始化组件、从层中采样。
        
        - 在评分样本时，破坏数据。

        传递整数以实现多次函数调用间的可重现结果。
        参见 :term:`术语表 <random_state>`。

    Attributes
    ----------
    intercept_hidden_ : array-like of shape (n_components,)
        隐藏单元的偏置。

    intercept_visible_ : array-like of shape (n_features,)
        可见单元的偏置。

    components_ : array-like of shape (n_components, n_features)
        权重矩阵，其中 `n_features` 是可见单元数量，`n_components` 是隐藏单元数量。

    h_samples_ : array-like of shape (batch_size, n_components)
        从模型分布中采样的隐藏激活，其中 `batch_size` 是每个小批量的示例数，
        `n_components` 是隐藏单元的数量。

    n_features_in_ : int
        在拟合过程中看到的特征数量。

        .. versionadded:: 0.24
    """
    # 特征名称列表，形状为 (`n_features_in_`,)
    # 仅在输入 `X` 中所有特征都是字符串名称时定义。

    # 自版本 1.0 起添加

    # 相关内容
    # --------
    # sklearn.neural_network.MLPRegressor : 多层感知机回归器。
    # sklearn.neural_network.MLPClassifier : 多层感知机分类器。
    # sklearn.decomposition.PCA : 无监督线性降维模型。

    # 参考文献
    # ----------
    # [1] Hinton, G. E., Osindero, S. and Teh, Y. A fast learning algorithm for
    #     deep belief nets. Neural Computation 18, pp 1527-1554.
    #     https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    # [2] Tieleman, T. Training Restricted Boltzmann Machines using
    #     Approximations to the Likelihood Gradient. International Conference
    #     on Machine Learning (ICML) 2008

    # 示例
    # --------

    # >>> import numpy as np
    # >>> from sklearn.neural_network import BernoulliRBM
    # >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    # >>> model = BernoulliRBM(n_components=2)
    # >>> model.fit(X)
    # BernoulliRBM(n_components=2)

    # 更详细的使用示例，请参见
    # :ref:`sphx_glr_auto_examples_neural_networks_plot_rbm_logistic_classification.py`.



    # 参数约束字典，包含各参数的取值范围约束
    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="neither")],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "n_iter": [Interval(Integral, 0, None, closed="left")],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
    }



    # 初始化方法，设置各参数的初始值
    def __init__(
        self,
        n_components=256,
        *,
        learning_rate=0.1,
        batch_size=10,
        n_iter=10,
        verbose=0,
        random_state=None,
    ):
        # 设置 RBM 模型的各个参数
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state



    # 转换方法，计算隐藏层激活的概率 P(h=1|v=X)
    def transform(self, X):
        # 检查模型是否已经拟合
        check_is_fitted(self)

        # 验证输入数据 X，并转换为适当格式
        X = self._validate_data(
            X, accept_sparse="csr", reset=False, dtype=(np.float64, np.float32)
        )
        # 返回数据的隐藏层表示
        return self._mean_hiddens(X)
    def _mean_hiddens(self, v):
        """Computes the probabilities P(h=1|v).

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        h : ndarray of shape (n_samples, n_components)
            Corresponding mean field values for the hidden layer.
        """
        # 计算隐藏层单元激活的概率，即 P(h=1|v)
        p = safe_sparse_dot(v, self.components_.T)
        p += self.intercept_hidden_
        return expit(p, out=p)

    def _sample_hiddens(self, v, rng):
        """Sample from the distribution P(h|v).

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer to sample from.

        rng : RandomState instance
            Random number generator to use.

        Returns
        -------
        h : ndarray of shape (n_samples, n_components)
            Values of the hidden layer.
        """
        # 从给定的条件概率分布 P(h|v) 中采样隐藏层的值
        p = self._mean_hiddens(v)
        return rng.uniform(size=p.shape) < p

    def _sample_visibles(self, h, rng):
        """Sample from the distribution P(v|h).

        Parameters
        ----------
        h : ndarray of shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        rng : RandomState instance
            Random number generator to use.

        Returns
        -------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer.
        """
        # 从给定的条件概率分布 P(v|h) 中采样可见层的值
        p = np.dot(h, self.components_)
        p += self.intercept_visible_
        expit(p, out=p)  # 将 p 应用 logistic 函数，更新 p 的值
        return rng.uniform(size=p.shape) < p

    def _free_energy(self, v):
        """Computes the free energy F(v) = - log sum_h exp(-E(v,h)).

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        free_energy : ndarray of shape (n_samples,)
            The value of the free energy.
        """
        # 计算自由能 F(v) = - log sum_h exp(-E(v,h))
        return -safe_sparse_dot(v, self.intercept_visible_) - np.logaddexp(
            0, safe_sparse_dot(v, self.components_.T) + self.intercept_hidden_
        ).sum(axis=1)

    def gibbs(self, v):
        """Perform one Gibbs sampling step.

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer to start from.

        Returns
        -------
        v_new : ndarray of shape (n_samples, n_features)
            Values of the visible layer after one Gibbs step.
        """
        # 执行一步 Gibbs 采样
        check_is_fitted(self)  # 检查模型是否已拟合
        if not hasattr(self, "random_state_"):
            self.random_state_ = check_random_state(self.random_state)
        h_ = self._sample_hiddens(v, self.random_state_)  # 采样隐藏层值
        v_ = self._sample_visibles(h_, self.random_state_)  # 采样可见层值

        return v_

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None):
        """Fit the model to the partial segment of the data X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        # 判断是否为首次拟合，如果是则进行初始化
        first_pass = not hasattr(self, "components_")
        # 对输入数据进行验证和处理，接受稀疏矩阵，转换为浮点数类型
        X = self._validate_data(
            X, accept_sparse="csr", dtype=np.float64, reset=first_pass
        )
        # 如果模型还没有随机状态属性，则初始化随机状态
        if not hasattr(self, "random_state_"):
            self.random_state_ = check_random_state(self.random_state)
        # 如果模型还没有组件属性，则初始化组件矩阵
        if not hasattr(self, "components_"):
            self.components_ = np.asarray(
                self.random_state_.normal(0, 0.01, (self.n_components, X.shape[1])),
                order="F",
            )
            self._n_features_out = self.components_.shape[0]
        # 如果模型还没有隐藏层截距属性，则初始化隐藏层截距向量
        if not hasattr(self, "intercept_hidden_"):
            self.intercept_hidden_ = np.zeros(
                self.n_components,
            )
        # 如果模型还没有可见层截距属性，则初始化可见层截距向量
        if not hasattr(self, "intercept_visible_"):
            self.intercept_visible_ = np.zeros(
                X.shape[1],
            )
        # 如果模型还没有存储隐藏层样本属性，则初始化隐藏层样本矩阵
        if not hasattr(self, "h_samples_"):
            self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        # 调用内部方法进行拟合
        self._fit(X, self.random_state_)

    def _fit(self, v_pos, rng):
        """Inner fit for one mini-batch.

        Adjust the parameters to maximize the likelihood of v using
        Stochastic Maximum Likelihood (SML).

        Parameters
        ----------
        v_pos : ndarray of shape (n_samples, n_features)
            The data to use for training.

        rng : RandomState instance
            Random number generator to use for sampling.
        """
        # 计算正样本的隐藏层均值
        h_pos = self._mean_hiddens(v_pos)
        # 从当前隐藏层样本中采样可见层样本
        v_neg = self._sample_visibles(self.h_samples_, rng)
        # 计算采样后的隐藏层均值
        h_neg = self._mean_hiddens(v_neg)

        # 计算学习率
        lr = float(self.learning_rate) / v_pos.shape[0]
        # 更新组件矩阵
        update = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T
        update -= np.dot(h_neg.T, v_neg)
        self.components_ += lr * update
        # 更新隐藏层截距向量
        self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        # 更新可见层截距向量
        self.intercept_visible_ += lr * (
            np.asarray(v_pos.sum(axis=0)).squeeze() - v_neg.sum(axis=0)
        )

        # 从二项分布中采样
        h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0
        # 将结果取整赋给隐藏层样本矩阵
        self.h_samples_ = np.floor(h_neg, h_neg)
    def score_samples(self, X):
        """Compute the pseudo-likelihood of X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Values of the visible layer. Must be all-boolean (not checked).

        Returns
        -------
        pseudo_likelihood : ndarray of shape (n_samples,)
            Value of the pseudo-likelihood (proxy for likelihood).

        Notes
        -----
        This method is not deterministic: it computes a quantity called the
        free energy on X, then on a randomly corrupted version of X, and
        returns the log of the logistic function of the difference.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)

        # 验证输入数据 X，接受稀疏矩阵格式 csr
        v = self._validate_data(X, accept_sparse="csr", reset=False)
        
        # 检查并设置随机数生成器
        rng = check_random_state(self.random_state)

        # 随机损坏每个样本中的一个特征
        ind = (np.arange(v.shape[0]), rng.randint(0, v.shape[1], v.shape[0]))
        
        if sp.issparse(v):
            # 如果 v 是稀疏矩阵，计算损坏后的数据
            data = -2 * v[ind] + 1
            if isinstance(data, np.matrix):
                # 如果 v 是稀疏矩阵，构建新的稀疏矩阵 v_
                v_ = v + sp.csr_matrix((data.A.ravel(), ind), shape=v.shape)
            else:
                # 如果 v 是稀疏数组，构建新的稀疏数组 v_
                v_ = v + sp.csr_array((data.ravel(), ind), shape=v.shape)
        else:
            # 如果 v 是密集数组，复制 v 并损坏对应位置的特征
            v_ = v.copy()
            v_[ind] = 1 - v_[ind]

        # 计算原始数据和损坏数据的自由能
        fe = self._free_energy(v)
        fe_ = self._free_energy(v_)

        # 计算伪似然（伪似然是似然的代理）
        # 返回 logistic 函数差异的对数
        return -v.shape[1] * np.logaddexp(0, -(fe_ - fe))

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the model to the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        # 对输入数据 X 进行验证和转换，接受稀疏矩阵 csr 格式，数据类型为浮点数
        X = self._validate_data(X, accept_sparse="csr", dtype=(np.float64, np.float32))
        
        # 获取样本数量
        n_samples = X.shape[0]
        
        # 使用指定的随机状态创建随机数生成器 rng
        rng = check_random_state(self.random_state)

        # 初始化模型的 components_，服从正态分布，均值为 0，标准差为 0.01
        self.components_ = np.asarray(
            rng.normal(0, 0.01, (self.n_components, X.shape[1])),
            order="F",  # 使用列优先存储
            dtype=X.dtype,
        )
        
        # 设置输出的特征数为 components_ 的行数
        self._n_features_out = self.components_.shape[0]
        
        # 初始化隐藏单元的截距为零
        self.intercept_hidden_ = np.zeros(self.n_components, dtype=X.dtype)
        
        # 初始化可见单元的截距为零
        self.intercept_visible_ = np.zeros(X.shape[1], dtype=X.dtype)
        
        # 初始化隐藏单元的样本矩阵为零矩阵
        self.h_samples_ = np.zeros((self.batch_size, self.n_components), dtype=X.dtype)

        # 计算批处理的数量
        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        
        # 生成批处理的切片列表
        batch_slices = list(
            gen_even_slices(n_batches * self.batch_size, n_batches, n_samples=n_samples)
        )
        
        # 获取 verbose 标志
        verbose = self.verbose
        
        # 记录开始时间
        begin = time.time()
        
        # 迭代训练模型
        for iteration in range(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                # 调用 _fit 方法进行模型拟合
                self._fit(X[batch_slice], rng)

            # 如果 verbose 为真，打印迭代信息
            if verbose:
                end = time.time()
                print(
                    "[%s] Iteration %d, pseudo-likelihood = %.2f, time = %.2fs"
                    % (
                        type(self).__name__,
                        iteration,
                        self.score_samples(X).mean(),
                        end - begin,
                    )
                )
                begin = end

        # 返回训练好的模型自身
        return self

    def _more_tags(self):
        # 返回额外的标签信息，用于测试和验证
        return {
            "_xfail_checks": {
                "check_methods_subset_invariance": (
                    "fails for the decision_function method"
                ),
                "check_methods_sample_order_invariance": (
                    "fails for the score_samples method"
                ),
            },
            "preserves_dtype": [np.float64, np.float32],  # 保留的数据类型
        }
```
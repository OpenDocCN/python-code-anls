# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\_lda.py`

```
"""
=============================================================
Online Latent Dirichlet Allocation with variational inference
=============================================================

This implementation is modified from Matthew D. Hoffman's onlineldavb code
Link: https://github.com/blei-lab/onlineldavb
"""

# 作者：Chyi-Kwei Yau
# 原始 onlineldavb 实现的作者：Matthew D. Hoffman
from numbers import Integral, Real

import numpy as np
import scipy.sparse as sp
from joblib import effective_n_jobs
from scipy.special import gammaln, logsumexp

from ..base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from ..utils import check_random_state, gen_batches, gen_even_slices
from ..utils._param_validation import Interval, StrOptions
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted, check_non_negative
from ._online_lda_fast import (
    _dirichlet_expectation_1d as cy_dirichlet_expectation_1d,
)
from ._online_lda_fast import (
    _dirichlet_expectation_2d,
)
from ._online_lda_fast import (
    mean_change as cy_mean_change,
)

EPS = np.finfo(float).eps


def _update_doc_distribution(
    X,
    exp_topic_word_distr,
    doc_topic_prior,
    max_doc_update_iter,
    mean_change_tol,
    cal_sstats,
    random_state,
):
    """E-step: update document-topic distribution.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Document word matrix.

    exp_topic_word_distr : ndarray of shape (n_topics, n_features)
        Exponential value of expectation of log topic word distribution.
        In the literature, this is `exp(E[log(beta)])`.

    doc_topic_prior : float
        Prior of document topic distribution `theta`.

    max_doc_update_iter : int
        Max number of iterations for updating document topic distribution in
        the E-step.

    mean_change_tol : float
        Stopping tolerance for updating document topic distribution in E-step.

    cal_sstats : bool
        Parameter that indicate to calculate sufficient statistics or not.
        Set `cal_sstats` to `True` when we need to run M-step.

    random_state : RandomState instance or None
        Parameter that indicate how to initialize document topic distribution.
        Set `random_state` to None will initialize document topic distribution
        to a constant number.

    Returns
    -------
    (doc_topic_distr, suff_stats) :
        `doc_topic_distr` is unnormalized topic distribution for each document.
        In the literature, this is `gamma`. we can calculate `E[log(theta)]`
        from it.
        `suff_stats` is expected sufficient statistics for the M-step.
            When `cal_sstats == False`, this will be None.
    """
    # 检查输入的 X 是否为稀疏矩阵
    is_sparse_x = sp.issparse(X)
    # 获取文档数和特征数
    n_samples, n_features = X.shape
    # 获取主题数
    n_topics = exp_topic_word_distr.shape[0]
    # 如果随机状态对象存在，则生成服从gamma分布的随机数据作为文档-主题分布，否则初始化为全部为1的数组
    if random_state:
        doc_topic_distr = random_state.gamma(100.0, 0.01, (n_samples, n_topics)).astype(
            X.dtype, copy=False
        )
    else:
        doc_topic_distr = np.ones((n_samples, n_topics), dtype=X.dtype)

    # 计算文档-主题分布的指数，这是文献中提到的 `exp(E[log(theta)])`
    exp_doc_topic = np.exp(_dirichlet_expectation_2d(doc_topic_distr))

    # 如果需要计算 suff_stats，则初始化为全零数组，否则设为 None
    suff_stats = (
        np.zeros(exp_topic_word_distr.shape, dtype=X.dtype) if cal_sstats else None
    )

    # 如果输入数据 X 是稀疏矩阵
    if is_sparse_x:
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

    # 这些 Cython 函数通常在很小的数组上（长度为 n_topics）进行嵌套循环调用。
    # 在这种情况下，找到正确的函数签名可能比执行本身更昂贵，因此分发操作放在循环外部。
    ctype = "float" if X.dtype == np.float32 else "double"
    mean_change = cy_mean_change[ctype]
    dirichlet_expectation_1d = cy_dirichlet_expectation_1d[ctype]
    eps = np.finfo(X.dtype).eps

    # 遍历每个样本
    for idx_d in range(n_samples):
        if is_sparse_x:
            # 如果输入数据 X 是稀疏矩阵，获取当前样本的非零元素的索引和对应的计数值
            ids = X_indices[X_indptr[idx_d] : X_indptr[idx_d + 1]]
            cnts = X_data[X_indptr[idx_d] : X_indptr[idx_d + 1]]
        else:
            # 如果输入数据 X 是密集矩阵，获取当前样本的非零元素的索引和对应的计数值
            ids = np.nonzero(X[idx_d, :])[0]
            cnts = X[idx_d, ids]

        # 获取当前样本的文档-主题分布
        doc_topic_d = doc_topic_distr[idx_d, :]
        # 对文档-主题分布的指数进行复制，因为内部循环会覆盖它
        exp_doc_topic_d = exp_doc_topic[idx_d, :].copy()
        # 获取对应于当前样本的主题-词分布的指数
        exp_topic_word_d = exp_topic_word_distr[:, ids]

        # 在文档-主题分布和主题-词分布之间迭代，直到收敛
        for _ in range(0, max_doc_update_iter):
            last_d = doc_topic_d

            # 计算优化的 phi_{dwk}，与 exp(E[log(theta_{dk})]) * exp(E[log(beta_{dw})]) 成比例
            norm_phi = np.dot(exp_doc_topic_d, exp_topic_word_d) + eps

            doc_topic_d = exp_doc_topic_d * np.dot(cnts / norm_phi, exp_topic_word_d.T)
            # 注意：在原地添加 doc_topic_prior 到 doc_topic_d
            dirichlet_expectation_1d(doc_topic_d, doc_topic_prior, exp_doc_topic_d)

            if mean_change(last_d, doc_topic_d) < mean_change_tol:
                break
        doc_topic_distr[idx_d, :] = doc_topic_d

        # 如果需要计算 suff_stats，则贡献文档 d 到期望足够统计量的贡献
        if cal_sstats:
            norm_phi = np.dot(exp_doc_topic_d, exp_topic_word_d) + eps
            suff_stats[:, ids] += np.outer(exp_doc_topic_d, cnts / norm_phi)

    # 返回文档-主题分布和 suff_stats
    return (doc_topic_distr, suff_stats)
# 定义 LatentDirichletAllocation 类，实现隐含狄利克雷分布的在线变分贝叶斯算法
class LatentDirichletAllocation(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    """Latent Dirichlet Allocation with online variational Bayes algorithm.

    使用在线变分贝叶斯算法实现隐含狄利克雷分布。

    The implementation is based on [1]_ and [2]_.
    
    这个实现基于文献 [1] 和 [2]。

    .. versionadded:: 0.17
    添加版本：0.17

    Read more in the :ref:`User Guide <LatentDirichletAllocation>`.
    在用户指南中详细阅读 :ref:`User Guide <LatentDirichletAllocation>`。

    Parameters
    ----------
    n_components : int, default=10
        Number of topics.
        主题数目，默认为10个。

        .. versionchanged:: 0.19
            ``n_topics`` was renamed to ``n_components``
            ``n_topics`` 重命名为 ``n_components``

    doc_topic_prior : float, default=None
        Prior of document topic distribution `theta`. If the value is None,
        defaults to `1 / n_components`.
        文档主题分布 `theta` 的先验值。如果为 None，则默认为 `1 / n_components`。
        在文献 [1] 中称为 `alpha`。

    topic_word_prior : float, default=None
        Prior of topic word distribution `beta`. If the value is None, defaults
        to `1 / n_components`.
        主题词分布 `beta` 的先验值。如果为 None，则默认为 `1 / n_components`。
        在文献 [1] 中称为 `eta`。

    learning_method : {'batch', 'online'}, default='batch'
        Method used to update `_component`. Only used in :meth:`fit` method.
        更新 `_component` 的方法。仅在 :meth:`fit` 方法中使用。

        In general, if the data size is large, the online update will be much
        faster than the batch update.
        通常情况下，如果数据量很大，使用在线更新比批量更新更快。

        Valid options::
        有效选项::

            'batch': Batch variational Bayes method. Use all training data in
                each EM update.
                批量变分贝叶斯方法。在每次 EM 更新中使用所有训练数据。
                旧的 `components_` 在每次迭代中都会被覆盖。

            'online': Online variational Bayes method. In each EM update, use
                mini-batch of training data to update the ``components_``
                variable incrementally. The learning rate is controlled by the
                ``learning_decay`` and the ``learning_offset`` parameters.
                在线变分贝叶斯方法。在每次 EM 更新中，使用训练数据的小批量增量更新 ``components_`` 变量。
                学习率由 ``learning_decay`` 和 ``learning_offset`` 参数控制。

        .. versionchanged:: 0.20
            The default learning method is now ``"batch"``.
            默认学习方法现在为 ``"batch"``。

    learning_decay : float, default=0.7
        It is a parameter that control learning rate in the online learning
        method. The value should be set between (0.5, 1.0] to guarantee
        asymptotic convergence.
        控制在线学习方法中学习率的参数。该值应设置在 (0.5, 1.0] 之间以保证渐近收敛。

        When the value is 0.0 and batch_size is ``n_samples``, the update method is same as batch learning.
        当值为 0.0 且 batch_size 为 ``n_samples`` 时，更新方法与批量学习相同。

        In the literature, this is called kappa.
        在文献中，这被称为 kappa。

    learning_offset : float, default=10.0
        A (positive) parameter that downweights early iterations in online
        learning. It should be greater than 1.0.
        在线学习中减轻早期迭代的参数。它应大于 1.0。
        在文献中，这被称为 tau_0。

    max_iter : int, default=10
        The maximum number of passes over the training data (aka epochs).
        训练数据的最大迭代次数（即 epochs）。
        仅影响 :meth:`fit` 方法的行为，而不影响 :meth:`partial_fit` 方法。

    batch_size : int, default=128
        Number of documents to use in each EM iteration. Only used in online
        learning.
        每个 EM 迭代中使用的文档数量。仅在在线学习中使用。
    evaluate_every : int, default=-1
        # 每隔多少步骤评估困惑度。仅在 `fit` 方法中使用。
        # 将其设置为 0 或负数将不会在训练过程中评估困惑度。评估困惑度有助于检查训练过程中的收敛性，但也会增加总体训练时间。
        # 每次迭代中评估困惑度可能会使训练时间增加一倍。
        How often to evaluate perplexity. Only used in `fit` method.
        Set it to 0 or negative number to not evaluate perplexity in
        training at all. Evaluating perplexity can help you check convergence
        in training process, but it will also increase total training time.
        Evaluating perplexity in every iteration might increase training time
        up to two-fold.

    total_samples : int, default=1e6
        # 总文档数目。仅在 :meth:`partial_fit` 方法中使用。
        Total number of documents. Only used in the :meth:`partial_fit` method.

    perp_tol : float, default=1e-1
        # 困惑度容差。仅在 ``evaluate_every`` 大于 0 时使用。
        Perplexity tolerance. Only used when ``evaluate_every`` is greater than 0.

    mean_change_tol : float, default=1e-3
        # 在 E 步中更新文档主题分布的停止容差。
        Stopping tolerance for updating document topic distribution in E-step.

    max_doc_update_iter : int, default=100
        # 在 E 步中更新文档主题分布的最大迭代次数。
        Max number of iterations for updating document topic distribution in
        the E-step.

    n_jobs : int, default=None
        # 在 E 步中使用的作业数。
        # ``None`` 表示使用 1 个作业，除非在 :obj:`joblib.parallel_backend` 上下文中。
        # ``-1`` 表示使用所有处理器。有关详细信息，请参阅 :term:`Glossary <n_jobs>`。
        The number of jobs to use in the E-step.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        # 冗余级别。
        Verbosity level.

    random_state : int, RandomState instance or None, default=None
        # 为多次函数调用产生可重现结果，传入一个整数。
        # 参见 :term:`Glossary <random_state>`。
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        # 主题词分布的变分参数。由于主题词分布的完全条件是一个狄利克雷分布，
        # ``components_[i, j]`` 可以看作是伪计数，表示单词 `j` 被分配到主题 `i` 的次数。
        # 它也可以看作是每个主题的单词分布的归一化后的分布：
        # ``model.components_ / model.components_.sum(axis=1)[:, np.newaxis]``。
        Variational parameters for topic word distribution. Since the complete
        conditional for topic word distribution is a Dirichlet,
        ``components_[i, j]`` can be viewed as pseudocount that represents the
        number of times word `j` was assigned to topic `i`.
        It can also be viewed as distribution over the words for each topic
        after normalization:
        ``model.components_ / model.components_.sum(axis=1)[:, np.newaxis]``.

    exp_dirichlet_component_ : ndarray of shape (n_components, n_features)
        # 对数主题词分布期望的指数值。
        # 在文献中，这是 `exp(E[log(beta)])`。
        Exponential value of expectation of log topic word distribution.
        In the literature, this is `exp(E[log(beta)])`.

    n_batch_iter_ : int
        # EM 步骤的迭代次数。
        Number of iterations of the EM step.

    n_features_in_ : int
        # 在 :term:`fit` 过程中看到的特征数。
        # .. versionadded:: 0.24
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 `X` 具有所有字符串特征名时定义。
        # .. versionadded:: 1.0
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_iter_ : int
        # 数据集上的传递次数。
        Number of passes over the dataset.

    bound_ : float
        # 训练集上的最终困惑度评分。
        Final perplexity score on training set.

    doc_topic_prior_ : float
        # 文档主题分布 `theta` 的先验。如果该值为 None，则为 `1 / n_components`。
        Prior of document topic distribution `theta`. If the value is None,
        it is `1 / n_components`.
    # random_state_ : RandomState instance
    #     RandomState instance that is generated either from a seed, the random
    #     number generator or by `np.random`.
    # topic_word_prior_ : float
    #     Prior of topic word distribution `beta`. If the value is None, it is
    #     `1 / n_components`.
    #
    # See Also
    # --------
    # sklearn.discriminant_analysis.LinearDiscriminantAnalysis:
    #     A classifier with a linear decision boundary, generated by fitting
    #     class conditional densities to the data and using Bayes' rule.
    #
    # References
    # ----------
    # .. [1] "Online Learning for Latent Dirichlet Allocation", Matthew D.
    #        Hoffman, David M. Blei, Francis Bach, 2010
    #        https://github.com/blei-lab/onlineldavb
    #
    # .. [2] "Stochastic Variational Inference", Matthew D. Hoffman,
    #        David M. Blei, Chong Wang, John Paisley, 2013
    #
    # Examples
    # --------
    # >>> from sklearn.decomposition import LatentDirichletAllocation
    # >>> from sklearn.datasets import make_multilabel_classification
    # >>> # This produces a feature matrix of token counts, similar to what
    # >>> # CountVectorizer would produce on text.
    # >>> X, _ = make_multilabel_classification(random_state=0)
    # >>> lda = LatentDirichletAllocation(n_components=5,
    # ...     random_state=0)
    # >>> lda.fit(X)
    # LatentDirichletAllocation(...)
    # >>> # get topics for some given samples:
    # >>> lda.transform(X[-2:])
    # array([[0.00360392, 0.25499205, 0.0036211 , 0.64236448, 0.09541846],
    #        [0.15297572, 0.00362644, 0.44412786, 0.39568399, 0.003586  ]])
    #
    # _parameter_constraints: dict
    #     Dictionary defining constraints for various parameters used in the
    #     LatentDirichletAllocation model.
    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 0, None, closed="neither")],
        "doc_topic_prior": [None, Interval(Real, 0, 1, closed="both")],
        "topic_word_prior": [None, Interval(Real, 0, 1, closed="both")],
        "learning_method": [StrOptions({"batch", "online"})],
        "learning_decay": [Interval(Real, 0, 1, closed="both")],
        "learning_offset": [Interval(Real, 1.0, None, closed="left")],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "batch_size": [Interval(Integral, 0, None, closed="neither")],
        "evaluate_every": [Interval(Integral, None, None, closed="neither")],
        "total_samples": [Interval(Real, 0, None, closed="neither")],
        "perp_tol": [Interval(Real, 0, None, closed="left")],
        "mean_change_tol": [Interval(Real, 0, None, closed="left")],
        "max_doc_update_iter": [Interval(Integral, 0, None, closed="left")],
        "n_jobs": [None, Integral],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
    }
    def __init__(
        self,
        n_components=10,
        *,
        doc_topic_prior=None,
        topic_word_prior=None,
        learning_method="batch",
        learning_decay=0.7,
        learning_offset=10.0,
        max_iter=10,
        batch_size=128,
        evaluate_every=-1,
        total_samples=1e6,
        perp_tol=1e-1,
        mean_change_tol=1e-3,
        max_doc_update_iter=100,
        n_jobs=None,
        verbose=0,
        random_state=None,
    ):
        # 初始化函数，设置LDA模型的参数
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.learning_method = learning_method
        self.learning_decay = learning_decay
        self.learning_offset = learning_offset
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.total_samples = total_samples
        self.perp_tol = perp_tol
        self.mean_change_tol = mean_change_tol
        self.max_doc_update_iter = max_doc_update_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _init_latent_vars(self, n_features, dtype=np.float64):
        """Initialize latent variables."""
        
        # 初始化潜在变量
        self.random_state_ = check_random_state(self.random_state)
        self.n_batch_iter_ = 1
        self.n_iter_ = 0

        if self.doc_topic_prior is None:
            self.doc_topic_prior_ = 1.0 / self.n_components
        else:
            self.doc_topic_prior_ = self.doc_topic_prior

        if self.topic_word_prior is None:
            self.topic_word_prior_ = 1.0 / self.n_components
        else:
            self.topic_word_prior_ = self.topic_word_prior

        init_gamma = 100.0
        init_var = 1.0 / init_gamma
        # 在文献中，这被称为`lambda`
        self.components_ = self.random_state_.gamma(
            init_gamma, init_var, (self.n_components, n_features)
        ).astype(dtype, copy=False)

        # 在文献中，这是`exp(E[log(beta)])`
        self.exp_dirichlet_component_ = np.exp(
            _dirichlet_expectation_2d(self.components_)
        )
    def _e_step(self, X, cal_sstats, random_init, parallel=None):
        """E-step in EM update.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        cal_sstats : bool
            Parameter that indicate whether to calculate sufficient statistics
            or not. Set ``cal_sstats`` to True when we need to run M-step.

        random_init : bool
            Parameter that indicate whether to initialize document topic
            distribution randomly in the E-step. Set it to True in training
            steps.

        parallel : joblib.Parallel, default=None
            Pre-initialized instance of joblib.Parallel.

        Returns
        -------
        (doc_topic_distr, suff_stats) :
            `doc_topic_distr` is unnormalized topic distribution for each
            document. In the literature, this is called `gamma`.
            `suff_stats` is expected sufficient statistics for the M-step.
            When `cal_sstats == False`, it will be None.
        """

        # Run e-step in parallel
        random_state = self.random_state_ if random_init else None  # 根据 random_init 参数确定是否使用随机状态

        # TODO: make Parallel._effective_n_jobs public instead?
        n_jobs = effective_n_jobs(self.n_jobs)  # 获取有效的并行作业数
        if parallel is None:
            parallel = Parallel(n_jobs=n_jobs, verbose=max(0, self.verbose - 1))  # 初始化并行处理对象

        # 并行执行更新文档分布的操作
        results = parallel(
            delayed(_update_doc_distribution)(
                X[idx_slice, :],  # 切片后的文档-词矩阵
                self.exp_dirichlet_component_,  # 经验狄利克雷分布的组成部分
                self.doc_topic_prior_,  # 文档主题先验
                self.max_doc_update_iter,  # 最大文档更新迭代次数
                self.mean_change_tol,  # 平均变化容忍度
                cal_sstats,  # 是否计算足够的统计量
                random_state,  # 随机状态
            )
            for idx_slice in gen_even_slices(X.shape[0], n_jobs)  # 生成均匀切片以进行并行处理
        )

        # 合并结果
        doc_topics, sstats_list = zip(*results)  # 解压并分离文档主题和统计量列表
        doc_topic_distr = np.vstack(doc_topics)  # 垂直堆叠文档主题分布

        if cal_sstats:
            # 当需要计算足够的统计量时，完成统计量计算步骤
            suff_stats = np.zeros(self.components_.shape, dtype=self.components_.dtype)
            for sstats in sstats_list:
                suff_stats += sstats  # 累加统计量
            suff_stats *= self.exp_dirichlet_component_  # 乘以经验狄利克雷分布的组成部分
        else:
            suff_stats = None  # 如果不需要计算足够的统计量，则设为 None

        return (doc_topic_distr, suff_stats)  # 返回文档主题分布和足够的统计量
    def _em_step(self, X, total_samples, batch_update, parallel=None):
        """
        EM update for 1 iteration.

        update `_component` by batch VB or online VB.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        total_samples : int
            Total number of documents. It is only used when
            batch_update is `False`.

        batch_update : bool
            Parameter that controls updating method.
            `True` for batch learning, `False` for online learning.

        parallel : joblib.Parallel, default=None
            Pre-initialized instance of joblib.Parallel

        Returns
        -------
        doc_topic_distr : ndarray of shape (n_samples, n_components)
            Unnormalized document topic distribution.
        """

        # E-step: Expectation step
        # Perform E-step of the EM algorithm, computing sufficient statistics.
        _, suff_stats = self._e_step(
            X, cal_sstats=True, random_init=True, parallel=parallel
        )

        # M-step: Maximization step
        if batch_update:
            # Batch update of components
            # Update using batch VB method
            self.components_ = self.topic_word_prior_ + suff_stats
        else:
            # Online update
            # Update using online VB method
            # In the literature, the weight is `rho`
            weight = np.power(
                self.learning_offset + self.n_batch_iter_, -self.learning_decay
            )
            doc_ratio = float(total_samples) / X.shape[0]
            self.components_ *= 1 - weight
            self.components_ += weight * (
                self.topic_word_prior_ + doc_ratio * suff_stats
            )

        # update `component_` related variables
        # Exponentiate the components to compute the Dirichlet expectation
        self.exp_dirichlet_component_ = np.exp(
            _dirichlet_expectation_2d(self.components_)
        )
        # Increment the batch iteration counter
        self.n_batch_iter_ += 1
        return

    def _more_tags(self):
        """
        Provide additional tags for estimator introspection.

        Returns
        -------
        dict
            Dictionary with additional tags for the estimator.
        """
        return {
            "preserves_dtype": [np.float64, np.float32],
            "requires_positive_X": True,
        }

    def _check_non_neg_array(self, X, reset_n_features, whom):
        """
        Check and ensure that array `X` is non-negative.

        Parameters
        ----------
        X :  array-like or sparse matrix
            Input data to be validated.

        reset_n_features : bool
            Whether to reset the number of features.

        whom : str
            Identifier for the array.

        Returns
        -------
        X : {array-like, sparse matrix}
            Validated and potentially transformed input `X`.
        """
        dtype = [np.float64, np.float32] if reset_n_features else self.components_.dtype

        # Validate the data X
        X = self._validate_data(
            X,
            reset=reset_n_features,
            accept_sparse="csr",
            dtype=dtype,
        )
        # Check for non-negative values in X
        check_non_negative(X, whom)

        return X

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None):
        """
        Online VB with Mini-Batch update.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Partially fitted estimator.
        """
        first_time = not hasattr(self, "components_")

        # 检查输入数据是否非负，如果是首次运行则重置特征数，用于 LatentDirichletAllocation.partial_fit
        X = self._check_non_neg_array(
            X, reset_n_features=first_time, whom="LatentDirichletAllocation.partial_fit"
        )
        n_samples, n_features = X.shape
        batch_size = self.batch_size

        # 初始化参数或进行检查
        if first_time:
            self._init_latent_vars(n_features, dtype=X.dtype)

        # 检查输入数据特征维度是否与模型的组件维度匹配
        if n_features != self.components_.shape[1]:
            raise ValueError(
                "The provided data has %d dimensions while "
                "the model was trained with feature size %d."
                % (n_features, self.components_.shape[1])
            )

        # 获取有效的并行工作数
        n_jobs = effective_n_jobs(self.n_jobs)
        with Parallel(n_jobs=n_jobs, verbose=max(0, self.verbose - 1)) as parallel:
            # 生成批次并对每个批次执行期望最大化（E-step）
            for idx_slice in gen_batches(n_samples, batch_size):
                self._em_step(
                    X[idx_slice, :],
                    total_samples=self.total_samples,
                    batch_update=False,
                    parallel=parallel,
                )

        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def _unnormalized_transform(self, X):
        """
        Transform data X according to fitted model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        Returns
        -------
        doc_topic_distr : ndarray of shape (n_samples, n_components)
            Document topic distribution for X.
        """
        # 执行 E-step，根据模型对数据 X 进行转换，返回文档主题分布
        doc_topic_distr, _ = self._e_step(X, cal_sstats=False, random_init=False)

        return doc_topic_distr

    def transform(self, X):
        """
        Transform data X according to the fitted model.

        .. versionchanged:: 0.18
           *doc_topic_distr* is now normalized

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        Returns
        -------
        doc_topic_distr : ndarray of shape (n_samples, n_components)
            Document topic distribution for X.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)
        # 检查输入数据是否非负，不重置特征数，用于 LatentDirichletAllocation.transform
        X = self._check_non_neg_array(
            X, reset_n_features=False, whom="LatentDirichletAllocation.transform"
        )
        # 获取未标准化的文档主题分布
        doc_topic_distr = self._unnormalized_transform(X)
        # 对文档主题分布进行归一化处理
        doc_topic_distr /= doc_topic_distr.sum(axis=1)[:, np.newaxis]
        return doc_topic_distr
    def _approx_bound(self, X, doc_topic_distr, sub_sampling):
        """Estimate the variational bound.

        Estimate the variational bound over "all documents" using only the
        documents passed in as X. Since log-likelihood of each word cannot
        be computed directly, we use this bound to estimate it.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        doc_topic_distr : ndarray of shape (n_samples, n_components)
            Document topic distribution. In the literature, this is called
            gamma.

        sub_sampling : bool, default=False
            Compensate for subsampling of documents.
            It is used in calculate bound in online learning.

        Returns
        -------
        score : float
            The estimated variational bound.

        """

        def _loglikelihood(prior, distr, dirichlet_distr, size):
            # calculate log-likelihood
            score = np.sum((prior - distr) * dirichlet_distr)
            score += np.sum(gammaln(distr) - gammaln(prior))
            score += np.sum(gammaln(prior * size) - gammaln(np.sum(distr, 1)))
            return score

        # Check if X is sparse
        is_sparse_x = sp.issparse(X)

        # Get dimensions
        n_samples, n_components = doc_topic_distr.shape
        n_features = self.components_.shape[1]

        # Initialize score
        score = 0

        # Compute E[log p(docs | theta, beta)]
        dirichlet_doc_topic = _dirichlet_expectation_2d(doc_topic_distr)
        dirichlet_component_ = _dirichlet_expectation_2d(self.components_)
        doc_topic_prior = self.doc_topic_prior_
        topic_word_prior = self.topic_word_prior_

        if is_sparse_x:
            # If X is sparse, extract data, indices, and indptr
            X_data = X.data
            X_indices = X.indices
            X_indptr = X.indptr

        # Iterate over each document
        for idx_d in range(0, n_samples):
            if is_sparse_x:
                # If X is sparse, get indices and counts
                ids = X_indices[X_indptr[idx_d] : X_indptr[idx_d + 1]]
                cnts = X_data[X_indptr[idx_d] : X_indptr[idx_d + 1]]
            else:
                # If X is dense, get non-zero indices and counts
                ids = np.nonzero(X[idx_d, :])[0]
                cnts = X[idx_d, ids]

            # Compute E[log p(docs | theta, beta)]
            temp = (
                dirichlet_doc_topic[idx_d, :, np.newaxis] + dirichlet_component_[:, ids]
            )
            norm_phi = logsumexp(temp, axis=0)
            score += np.dot(cnts, norm_phi)

        # Compute E[log p(theta | alpha) - log q(theta | gamma)]
        score += _loglikelihood(
            doc_topic_prior, doc_topic_distr, dirichlet_doc_topic, self.n_components
        )

        # Compensate for the subsampling of the population of documents
        if sub_sampling:
            doc_ratio = float(self.total_samples) / n_samples
            score *= doc_ratio

        # Compute E[log p(beta | eta) - log q (beta | lambda)]
        score += _loglikelihood(
            topic_word_prior, self.components_, dirichlet_component_, n_features
        )

        return score
    def score(self, X, y=None):
        """Calculate approximate log-likelihood as score.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        score : float
            Use approximate bound as score.
        """
        # 确保模型已拟合
        check_is_fitted(self)
        # 检查并处理输入的非负数组或稀疏矩阵X，保持特征数不变
        X = self._check_non_neg_array(
            X, reset_n_features=False, whom="LatentDirichletAllocation.score"
        )

        # 获取文档-主题分布
        doc_topic_distr = self._unnormalized_transform(X)
        # 计算近似下界作为评分
        score = self._approx_bound(X, doc_topic_distr, sub_sampling=False)
        return score

    def _perplexity_precomp_distr(self, X, doc_topic_distr=None, sub_sampling=False):
        """Calculate approximate perplexity for data X with ability to accept
        precomputed doc_topic_distr

        Perplexity is defined as exp(-1. * log-likelihood per word)

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        doc_topic_distr : ndarray of shape (n_samples, n_components), \
                default=None
            Document topic distribution.
            If it is None, it will be generated by applying transform on X.

        Returns
        -------
        score : float
            Perplexity score.
        """
        # 如果未提供预先计算的doc_topic_distr，则通过X生成
        if doc_topic_distr is None:
            doc_topic_distr = self._unnormalized_transform(X)
        else:
            # 检查doc_topic_distr的维度是否匹配
            n_samples, n_components = doc_topic_distr.shape
            if n_samples != X.shape[0]:
                raise ValueError(
                    "Number of samples in X and doc_topic_distr do not match."
                )
            # 检查主题数是否匹配
            if n_components != self.n_components:
                raise ValueError("Number of topics does not match.")

        current_samples = X.shape[0]
        # 计算近似下界
        bound = self._approx_bound(X, doc_topic_distr, sub_sampling)

        # 根据是否使用子采样计算每词下界
        if sub_sampling:
            word_cnt = X.sum() * (float(self.total_samples) / current_samples)
        else:
            word_cnt = X.sum()
        perword_bound = bound / word_cnt

        # 计算困惑度分数
        return np.exp(-1.0 * perword_bound)
    # 计算给定数据 X 的近似困惑度（perplexity）

    # 检查当前对象是否已拟合（即模型已训练好）
    check_is_fitted(self)

    # 检查并确保输入的数据 X 是非负的数组或稀疏矩阵，
    # 同时重置特征数量，并标明调用来源为 LatentDirichletAllocation.perplexity
    X = self._check_non_neg_array(
        X, reset_n_features=True, whom="LatentDirichletAllocation.perplexity"
    )

    # 调用内部方法 _perplexity_precomp_distr 计算困惑度分数，
    # 并返回计算结果
    return self._perplexity_precomp_distr(X, sub_sampling=sub_sampling)

@property
def _n_features_out(self):
    """输出特征的数量。"""
    # 返回模型组件的形状的第一个维度，即输出特征的数量
    return self.components_.shape[0]
```
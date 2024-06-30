# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_iforest.py`

```
# 导入必要的模块和库
import numbers  # 导入 numbers 模块
from numbers import Integral, Real  # 从 numbers 模块导入 Integral 和 Real 类
from warnings import warn  # 导入 warn 函数

import numpy as np  # 导入 numpy 库并重命名为 np
from scipy.sparse import issparse  # 从 scipy.sparse 导入 issparse 函数

# 导入所需的 scikit-learn 内部模块和函数
from ..base import OutlierMixin, _fit_context  # 从 ..base 模块导入 OutlierMixin 和 _fit_context
from ..tree import ExtraTreeRegressor  # 从 ..tree 模块导入 ExtraTreeRegressor 类
from ..tree._tree import DTYPE as tree_dtype  # 从 ..tree._tree 模块导入 DTYPE 并重命名为 tree_dtype
from ..utils import (
    check_array,  # 从 ..utils 导入 check_array 函数
    check_random_state,  # 从 ..utils 导入 check_random_state 函数
    gen_batches,  # 从 ..utils 导入 gen_batches 函数
)
from ..utils._chunking import get_chunk_n_rows  # 从 ..utils._chunking 导入 get_chunk_n_rows 函数
from ..utils._param_validation import Interval, RealNotInt, StrOptions  # 从 ..utils._param_validation 导入 Interval, RealNotInt, StrOptions 类
from ..utils.validation import _num_samples, check_is_fitted  # 从 ..utils.validation 导入 _num_samples, check_is_fitted 函数
from ._bagging import BaseBagging  # 从 ._bagging 导入 BaseBagging 类

# 设置模块的公开接口，仅包含 IsolationForest 类
__all__ = ["IsolationForest"]


class IsolationForest(OutlierMixin, BaseBagging):
    """
    Isolation Forest Algorithm.

    Return the anomaly score of each sample using the IsolationForest algorithm

    The IsolationForest 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.

    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function.

    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.

    Read more in the :ref:`User Guide <isolation_forest>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators in the ensemble.

    max_samples : "auto", int or float, default="auto"
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.

            - If 'auto', the threshold is determined as in the
              original paper.
            - If float, the contamination should be in the range (0, 0.5].

        .. versionchanged:: 0.22
           The default value of ``contamination`` changed from 0.1
           to ``'auto'``.

    """
    # IsolationForest 类继承自 OutlierMixin 和 BaseBagging 类

    def __init__(self, n_estimators=100, max_samples="auto", contamination='auto'):
        # 初始化方法，设置 IsolationForest 类的参数

        super().__init__(n_estimators=n_estimators, base_estimator=ExtraTreeRegressor(),
                         random_state=None, max_samples=max_samples,
                         contamination=contamination)
        # 调用父类 BaseBagging 的初始化方法，设置 n_estimators, base_estimator, random_state, max_samples, contamination 参数
    # max_features控制每个基本估计器从训练集X中抽取的特征数量
    max_features : int or float, default=1.0
        # 如果是int，则抽取max_features个特征
        - If int, then draw `max_features` features.
        # 如果是float，则抽取max_features * n_features_in_的整数部分作为特征数
        - If float, then draw `max(1, int(max_features * n_features_in_))` features.

        Note: 使用小于1.0的float或小于特征数量的整数启用特征子抽样，会增加运行时间。

    # bootstrap决定是否对训练数据进行自助采样
    bootstrap : bool, default=False
        # 如果True，每棵树会在有放回地对训练数据进行随机子集采样
        If True, individual trees are fit on random subsets of the training
        # 如果False，则进行无放回的采样
        data sampled without replacement is performed.

    # n_jobs表示并行运行任务的数量，适用于fit和predict方法
    n_jobs : int, default=None
        # 并行任务的数量，None表示默认为1，除非在joblib.parallel_backend上下文中
        The number of jobs to run in parallel for both :meth:`fit` and
        # -1表示使用所有处理器
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.

    # random_state控制每次运行时的随机性，确保结果可以重现
    random_state : int, RandomState instance or None, default=None
        # 控制选择特征和分割值的伪随机性，适用于每个分支步骤和每棵树
        Controls the pseudo-randomness of the selection of the feature
        # 传递一个int以便于多次调用时结果可重现
        and split values for each branching step and each tree in the forest.
        See :term:`Glossary <random_state>`.

    # verbose控制树构建过程中的详细程度
    verbose : int, default=0
        # 控制树构建过程的详细程度
        Controls the verbosity of the tree building process.

    # warm_start设置为True时，可以重用前一次调用fit的解决方案并添加更多估计器到集成中
    warm_start : bool, default=False
        # 当设置为True时，可以重用前一次调用fit的解决方案并添加更多估计器到集成中
        When set to ``True``, reuse the solution of the previous call to fit
        # 否则，只是适合一个全新的森林
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.21

    # estimator_是用于创建已安装子估计器集合的子估计器模板
    Attributes
    ----------
    estimator_ : :class:`~sklearn.tree.ExtraTreeRegressor` instance
        # 用于创建已安装子估计器集合的子估计器模板
        The child estimator template used to create the collection of
        fitted sub-estimators.

        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.

    # estimators_是已安装的子估计器的集合列表
    estimators_ : list of ExtraTreeRegressor instances
        # 已安装的子估计器的集合列表

    # estimators_features_是每个基本估计器的抽取特征的子集列表
    estimators_features_ : list of ndarray
        # 每个基本估计器的抽取特征的子集列表

    # estimators_samples_是每个基本估计器的抽取样本（即内袋样本）的子集列表
    estimators_samples_ : list of ndarray
        # 每个基本估计器的抽取样本（即内袋样本）的子集列表

    # max_samples_是实际样本数量
    max_samples_ : int
        # 实际样本数量

    # offset_是用于从原始分数定义决策函数的偏移量
    offset_ : float
        # 用于从原始分数定义决策函数的偏移量
        Offset used to define the decision function from the raw scores.
        # 我们有关系：decision_function = score_samples - offset_
        We have the relation: ``decision_function = score_samples - offset_``.
        # 当污染参数设置为“auto”时，偏移量等于-0.5，因为内点的分数接近0，异常点的分数接近-1
        ``offset_`` is defined as follows. When the contamination parameter is
        set to "auto", the offset is equal to -0.5 as the scores of inliers are
        close to 0 and the scores of outliers are close to -1.
        # 当提供的污染参数与“auto”不同时，偏移量被定义为使得在训练中获得预期数量的异常值（决策函数 < 0）
        When a contamination parameter different than "auto" is provided, the offset
        is defined in such a way we obtain the expected number of outliers
        (samples with decision function < 0) in training.

        .. versionadded:: 0.20
    # 记录在拟合过程中观察到的特征数量
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    # 记录在拟合过程中观察到的特征名称列表，仅当 `X` 的特征名称全为字符串时有效
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    # 相关链接和类
    See Also
    --------
    sklearn.covariance.EllipticEnvelope : An object for detecting outliers in a
        Gaussian distributed dataset.
    sklearn.svm.OneClassSVM : Unsupervised Outlier Detection.
        Estimate the support of a high-dimensional distribution.
        The implementation is based on libsvm.
    sklearn.neighbors.LocalOutlierFactor : Unsupervised Outlier Detection
        using Local Outlier Factor (LOF).

    # 实现说明
    Notes
    -----
    The implementation is based on an ensemble of ExtraTreeRegressor. The
    maximum depth of each tree is set to ``ceil(log_2(n))`` where
    :math:`n` is the number of samples used to build the tree
    (see (Liu et al., 2008) for more details).

    # 参考文献
    References
    ----------
    .. [1] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
           Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on.
    .. [2] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation-based
           anomaly detection." ACM Transactions on Knowledge Discovery from
           Data (TKDD) 6.1 (2012): 3.

    # 示例
    Examples
    --------
    >>> from sklearn.ensemble import IsolationForest
    >>> X = [[-1.1], [0.3], [0.5], [100]]
    >>> clf = IsolationForest(random_state=0).fit(X)
    >>> clf.predict([[0.1], [0], [90]])
    array([ 1,  1, -1])

    For an example of using isolation forest for anomaly detection see
    :ref:`sphx_glr_auto_examples_ensemble_plot_isolation_forest.py`.
    """

    # 参数约束字典，指定了初始化 IsolationForest 类时各参数的取值范围和类型
    _parameter_constraints: dict = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "max_samples": [
            StrOptions({"auto"}),
            Interval(Integral, 1, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="right"),
        ],
        "contamination": [
            StrOptions({"auto"}),
            Interval(Real, 0, 0.5, closed="right"),
        ],
        "max_features": [
            Integral,
            Interval(Real, 0, 1, closed="right"),
        ],
        "bootstrap": ["boolean"],
        "n_jobs": [Integral, None],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
        "warm_start": ["boolean"],
    }

    # IsolationForest 类的初始化方法，定义了各个参数的默认值
    def __init__(
        self,
        *,
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ):
        super().__init__(
            estimator=None,
            # 这里的max_features与self.max_features没有关联
            bootstrap=bootstrap,
            bootstrap_features=False,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

        self.contamination = contamination

    def _get_estimator(self):
        return ExtraTreeRegressor(
            # 这里的max_features与self.max_features没有关联
            max_features=1,
            splitter="random",
            random_state=self.random_state,
        )

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by iforest")

    def _parallel_args(self):
        # ExtraTreeRegressor释放了GIL，因此使用基于线程而不是基于进程的后端更有效，
        # 以避免受到通信开销和额外内存复制的影响。
        return {"prefer": "threads"}

    @_fit_context(prefer_skip_nested_validation=True)
    def predict(self, X):
        """
        预测特定样本是否为异常值。

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            输入样本。在内部，将被转换为dtype=np.float32，如果提供了稀疏矩阵，
            则转换为稀疏的csr_matrix。

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            对于每个观察结果，指示是否（+1或-1）应根据拟合模型视为内部点。
        """
        check_is_fitted(self)
        decision_func = self.decision_function(X)
        is_inlier = np.ones_like(decision_func, dtype=int)
        is_inlier[decision_func < 0] = -1
        return is_inlier
    def decision_function(self, X):
        """
        Average anomaly score of X of the base classifiers.

        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.

        The measure of normality of an observation given a tree is the depth
        of the leaf containing this observation, which is equivalent to
        the number of splittings required to isolate this point. In case of
        several observations n_left in the leaf, the average path length of
        a n_left samples isolation tree is added.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal. Negative scores represent outliers,
            positive scores represent inliers.
        """
        # We subtract self.offset_ to make 0 be the threshold value for being
        # an outlier:
        减去 self.offset_ 的目的是将 0 设置为异常值的阈值：

        return self.score_samples(X) - self.offset_

    def score_samples(self, X):
        """
        Opposite of the anomaly score defined in the original paper.

        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.

        The measure of normality of an observation given a tree is the depth
        of the leaf containing this observation, which is equivalent to
        the number of splittings required to isolate this point. In case of
        several observations n_left in the leaf, the average path length of
        a n_left samples isolation tree is added.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal.
        """
        # Check data
        X = self._validate_data(X, accept_sparse="csr", dtype=tree_dtype, reset=False)
        验证数据

        return self._score_samples(X)

    def _score_samples(self, X):
        """Private version of score_samples without input validation.

        Input validation would remove feature names, so we disable it.
        """
        # Code structure from ForestClassifier/predict_proba

        check_is_fitted(self)
        检查模型是否已拟合

        # Take the opposite of the scores as bigger is better (here less abnormal)
        返回负数来反映得分越大越好（即越不异常）

        return -self._compute_chunked_score_samples(X)
    def _compute_chunked_score_samples(self, X):
        n_samples = _num_samples(X)  # 获取样本数目

        if self._max_features == X.shape[1]:
            subsample_features = False  # 如果最大特征数等于样本特征数，不进行特征子采样
        else:
            subsample_features = True  # 否则进行特征子采样

        # 根据工作内存限制计算能够处理的最大行数，以便在计算过程中存储self._max_features个特征的数据
        # （由sklearn.get_config()['working_memory']定义）
        #
        # 注意：
        #  - 即使一行得分超出工作内存，此处将至少获取1行。
        #  - 这仅考虑计算所需数据加载期间的临时内存使用，返回的得分本身是1维的。
        chunk_n_rows = get_chunk_n_rows(
            row_bytes=16 * self._max_features, max_n_rows=n_samples
        )
        slices = gen_batches(n_samples, chunk_n_rows)

        scores = np.zeros(n_samples, order="f")  # 初始化得分数组

        for sl in slices:
            # 在测试样本的切片上计算得分：
            scores[sl] = self._compute_score_samples(X[sl], subsample_features)

        return scores

    def _compute_score_samples(self, X, subsample_features):
        """
        计算每个样本在X中通过额外树的得分。

        Parameters
        ----------
        X : array-like or sparse matrix
            数据矩阵。

        subsample_features : bool
            是否对特征进行子采样。
        """
        n_samples = X.shape[0]  # 获取样本数目

        depths = np.zeros(n_samples, order="f")  # 初始化深度数组

        average_path_length_max_samples = _average_path_length([self._max_samples])  # 计算最大样本数的平均路径长度

        for tree_idx, (tree, features) in enumerate(
            zip(self.estimators_, self.estimators_features_)
        ):
            X_subset = X[:, features] if subsample_features else X

            leaves_index = tree.apply(X_subset, check_input=False)  # 获取叶节点索引

            # 计算深度，考虑每棵树的决策路径长度和每棵树的平均路径长度
            depths += (
                self._decision_path_lengths[tree_idx][leaves_index]
                + self._average_path_length_per_tree[tree_idx][leaves_index]
                - 1.0
            )
        denominator = len(self.estimators_) * average_path_length_max_samples
        scores = 2 ** (
            # 对于单个训练样本，分母和深度均为0。
            # 因此，手动将得分设置为1。
            -np.divide(
                depths, denominator, out=np.ones_like(depths), where=denominator != 0
            )
        )
        return scores

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }
# 计算在一个 iTree 中的平均路径长度，它等同于一个未成功的 BST（二叉搜索树）搜索的平均路径长度，
# 因为 iTree 与 BST 具有相同的结构。

def _average_path_length(n_samples_leaf):
    """
    The average path length in a n_samples iTree, which is equal to
    the average path length of an unsuccessful BST search since the
    latter has the same structure as an isolation tree.
    Parameters
    ----------
    n_samples_leaf : array-like of shape (n_samples,)
        The number of training samples in each test sample leaf, for
        each estimator.

    Returns
    -------
    average_path_length : ndarray of shape (n_samples,)
    """

    # 将 n_samples_leaf 转换为数组，确保为二维
    n_samples_leaf = check_array(n_samples_leaf, ensure_2d=False)

    # 记录原始形状
    n_samples_leaf_shape = n_samples_leaf.shape
    # 将 n_samples_leaf 转换为形状为 (1, -1) 的数组
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    # 创建一个全零数组来存储平均路径长度
    average_path_length = np.zeros(n_samples_leaf.shape)

    # 创建条件掩码
    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    # 对应掩码设置平均路径长度的值
    average_path_length[mask_1] = 0.0
    average_path_length[mask_2] = 1.0
    # 对于其他情况，使用公式计算平均路径长度
    average_path_length[not_mask] = (
        2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
        - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    # 返回重新调整形状后的平均路径长度数组
    return average_path_length.reshape(n_samples_leaf_shape)
```
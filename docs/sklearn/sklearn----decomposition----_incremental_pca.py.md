# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\_incremental_pca.py`

```
# 从 numbers 模块导入 Integral 类型，用于参数验证
from numbers import Integral

# 导入必要的库和模块
import numpy as np
from scipy import linalg, sparse

# 从 scikit-learn 中导入基础函数和工具
from ..base import _fit_context
from ..utils import gen_batches
from ..utils._param_validation import Interval
from ..utils.extmath import _incremental_mean_and_var, svd_flip
from ._base import _BasePCA

# 定义增量主成分分析(IPCA)类，继承自 _BasePCA
class IncrementalPCA(_BasePCA):
    """Incremental principal components analysis (IPCA).

    Linear dimensionality reduction using Singular Value Decomposition of
    the data, keeping only the most significant singular vectors to
    project the data to a lower dimensional space. The input data is centered
    but not scaled for each feature before applying the SVD.

    Depending on the size of the input data, this algorithm can be much more
    memory efficient than a PCA, and allows sparse input.

    This algorithm has constant memory complexity, on the order
    of ``batch_size * n_features``, enabling use of np.memmap files without
    loading the entire file into memory. For sparse matrices, the input
    is converted to dense in batches (in order to be able to subtract the
    mean) which avoids storing the entire dense matrix at any one time.

    The computational overhead of each SVD is
    ``O(batch_size * n_features ** 2)``, but only 2 * batch_size samples
    remain in memory at a time. There will be ``n_samples / batch_size`` SVD
    computations to get the principal components, versus 1 large SVD of
    complexity ``O(n_samples * n_features ** 2)`` for PCA.

    For a usage example, see
    :ref:`sphx_glr_auto_examples_decomposition_plot_incremental_pca.py`.

    Read more in the :ref:`User Guide <IncrementalPCA>`.

    .. versionadded:: 0.16

    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep. If ``n_components`` is ``None``,
        then ``n_components`` is set to ``min(n_samples, n_features)``.

    whiten : bool, default=False
        When True (False by default) the ``components_`` vectors are divided
        by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
        with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometimes
        improve the predictive accuracy of the downstream estimators by
        making data respect some hard-wired assumptions.

    copy : bool, default=True
        If False, X will be overwritten. ``copy=False`` can be used to
        save memory but is unsafe for general use.
    """
    batch_size : int, default=None
        每个批次使用的样本数。仅在调用 ``fit`` 时使用。如果 ``batch_size`` 是 ``None``，则从数据推断并设置为 ``5 * n_features``，以在逼近精度和内存消耗之间取得平衡。

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        特征空间中的主轴，代表数据中方差最大的方向。相当于中心化输入数据的右奇异向量，与其特征向量平行。这些组件按 ``explained_variance_`` 降序排列。

    explained_variance_ : ndarray of shape (n_components,)
        每个选择的主成分解释的方差。

    explained_variance_ratio_ : ndarray of shape (n_components,)
        每个选择的主成分解释的方差百分比。如果存储了所有成分，解释方差的总和等于 1.0。

    singular_values_ : ndarray of shape (n_components,)
        对应于每个选择的主成分的奇异值。奇异值等于低维空间中 ``n_components`` 变量的二范数。

    mean_ : ndarray of shape (n_features,)
        每个特征的经验均值，通过 ``partial_fit`` 的多次调用聚合而成。

    var_ : ndarray of shape (n_features,)
        每个特征的经验方差，通过 ``partial_fit`` 的多次调用聚合而成。

    noise_variance_ : float
        根据 Tipping 和 Bishop 1999 年的概率 PCA 模型估计的噪声协方差。参见 C. Bishop 的《Pattern Recognition and Machine Learning》12.2.1 p. 574 或 http://www.miketipping.com/papers/met-mppca.pdf。

    n_components_ : int
        估计的成分数。在 ``n_components=None`` 时相关。

    n_samples_seen_ : int
        估计器处理的样本数。在新的 ``fit`` 调用时将重置，但在 ``partial_fit`` 调用中递增。

    batch_size_ : int
        从 ``batch_size`` 推断出的批次大小。

    n_features_in_ : int
        在 ``fit`` 过程中看到的特征数。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在 ``fit`` 过程中看到的特征名称。仅在 `X` 具有全部字符串特征名称时定义。

        .. versionadded:: 1.0

    See Also
    --------
    PCA : 主成分分析（PCA）。
    KernelPCA : 核主成分分析（KPCA）。
    SparsePCA : 稀疏主成分分析（SparsePCA）。
    TruncatedSVD : 使用截断奇异值分解进行降维。

    Notes
    -----
    实现来自以下文献的增量 PCA 模型：
    *D. Ross, J. Lim, R. Lin, M. Yang, Incremental Learning for Robust Visual
    """
    _tracking: 论文引用和参考文献，描述了本模型的来源和相关研究背景
    
    _parameter_constraints: 参数约束字典，指定了模型参数的限制条件
    
    __init__: 模型类的初始化方法，用于设置模型的各种参数
    
    _fit_context: 装饰器函数，用于设置模型拟合时的上下文环境
    """
    _parameter_constraints: dict = {
        # "n_components"参数的约束：必须为大于等于1的整数或None
        "n_components": [Interval(Integral, 1, None, closed="left"), None],
        # "whiten"参数的约束：必须为布尔值
        "whiten": ["boolean"],
        # "copy"参数的约束：必须为布尔值
        "copy": ["boolean"],
        # "batch_size"参数的约束：必须为大于等于1的整数或None
        "batch_size": [Interval(Integral, 1, None, closed="left"), None],
    }
    
    def __init__(self, n_components=None, *, whiten=False, copy=True, batch_size=None):
        # 初始化模型参数：降维后的维度
        self.n_components = n_components
        # 初始化模型参数：是否进行白化
        self.whiten = whiten
        # 初始化模型参数：是否复制数据
        self.copy = copy
        # 初始化模型参数：每个批次的大小
        self.batch_size = batch_size
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the model with X, using minibatches of size batch_size.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 初始化模型的各种属性为 None 或者初始值
        self.components_ = None
        self.n_samples_seen_ = 0
        self.mean_ = 0.0
        self.var_ = 0.0
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.noise_variance_ = None

        # 对输入数据 X 进行验证和转换，确保其符合模型的要求
        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc", "lil"],  # 接受稀疏矩阵的格式
            copy=self.copy,
            dtype=[np.float64, np.float32],  # 数据类型限制为浮点数
            force_writeable=True,
        )
        n_samples, n_features = X.shape  # 获取样本数和特征数

        # 如果未指定 batch_size，则设置为默认值，与特征数相关
        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        # 使用生成器 gen_batches 分批处理数据进行训练
        for batch in gen_batches(
            n_samples, self.batch_size_, min_batch_size=self.n_components or 0
        ):
            X_batch = X[batch]  # 获取当前批次的数据
            if sparse.issparse(X_batch):
                X_batch = X_batch.toarray()  # 如果是稀疏矩阵，则转换为密集矩阵
            self.partial_fit(X_batch, check_input=False)  # 调用 partial_fit 方法进行部分拟合

        return self  # 返回拟合后的模型实例

    @_fit_context(prefer_skip_nested_validation=True)
    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set, using minibatches of size batch_size if X is
        sparse.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Projection of X in the first principal components.

        Examples
        --------

        >>> import numpy as np
        >>> from sklearn.decomposition import IncrementalPCA
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2],
        ...               [1, 1], [2, 1], [3, 2]])
        >>> ipca = IncrementalPCA(n_components=2, batch_size=3)
        >>> ipca.fit(X)
        IncrementalPCA(batch_size=3, n_components=2)
        >>> ipca.transform(X) # doctest: +SKIP
        """

        # 如果 X 是稀疏矩阵，执行以下操作
        if sparse.issparse(X):
            # 获取样本数量
            n_samples = X.shape[0]
            # 初始化输出列表
            output = []
            # 遍历生成的批次
            for batch in gen_batches(
                n_samples, self.batch_size_, min_batch_size=self.n_components or 0
            ):
                # 将每个批次的稀疏矩阵转换为稠密数组，然后调用父类的 transform 方法
                output.append(super().transform(X[batch].toarray()))
            # 将输出堆叠为一个 ndarray 并返回
            return np.vstack(output)
        else:
            # 如果 X 不是稀疏矩阵，直接调用父类的 transform 方法
            return super().transform(X)
```
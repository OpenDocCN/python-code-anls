# `D:\src\scipysrc\scikit-learn\sklearn\datasets\_samples_generator.py`

```
@validate_params(
    {
        "n_samples": [Interval(Integral, 1, None, closed="left")],
        "n_features": [Interval(Integral, 1, None, closed="left")],
        "n_informative": [Interval(Integral, 1, None, closed="left")],
        "n_redundant": [Interval(Integral, 0, None, closed="left")],
        "n_repeated": [Interval(Integral, 0, None, closed="left")],
        "n_classes": [Interval(Integral, 1, None, closed="left")],
        "n_clusters_per_class": [Interval(Integral, 1, None, closed="left")],
        "weights": ["array-like", None],
        "flip_y": [Interval(Real, 0, 1, closed="both")],
        "class_sep": [Interval(Real, 0, None, closed="neither")],
        "hypercube": ["boolean"],
        "shift": [Interval(Real, None, None, closed="neither"), "array-like", None],
        "scale": [Interval(Real, 0, None, closed="neither"), "array-like", None],
        "shuffle": ["boolean"],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
def make_classification(
    n_samples=100,
    n_features=20,
    *,
    n_informative=2,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    weights=None,
    flip_y=0.01,
    class_sep=1.0,
    hypercube=True,
    shift=0.0,
    scale=1.0,
    shuffle=True,
    random_state=None,
):
    """Generate a random n-class classification problem.

    This initially creates clusters of points normally distributed (std=1)
    about vertices of an ``n_informative``-dimensional hypercube with sides of
    length ``2*class_sep`` and assigns an equal number of clusters to each
    class. It introduces interdependence between these features and adds
    various types of further noise to the data.

    Without shuffling, ``X`` horizontally stacks features in the following
    # order: the primary ``n_informative`` features, followed by ``n_redundant``
    # linear combinations of the informative features, followed by ``n_repeated``
    # duplicates, drawn randomly with replacement from the informative and
    # redundant features. The remaining features are filled with random noise.
    # Thus, without shuffling, all useful features are contained in the columns
    # ``X[:, :n_informative + n_redundant + n_repeated]``.

    # For an example of usage, see
    # :ref:`sphx_glr_auto_examples_datasets_plot_random_dataset.py`.

    # Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=100
        # The number of samples.

    n_features : int, default=20
        # The total number of features. These comprise ``n_informative``
        # informative features, ``n_redundant`` redundant features,
        # ``n_repeated`` duplicated features and
        # ``n_features-n_informative-n_redundant-n_repeated`` useless features
        # drawn at random.

    n_informative : int, default=2
        # The number of informative features. Each class is composed of a number
        # of gaussian clusters each located around the vertices of a hypercube
        # in a subspace of dimension ``n_informative``. For each cluster,
        # informative features are drawn independently from  N(0, 1) and then
        # randomly linearly combined within each cluster in order to add
        # covariance. The clusters are then placed on the vertices of the
        # hypercube.

    n_redundant : int, default=2
        # The number of redundant features. These features are generated as
        # random linear combinations of the informative features.

    n_repeated : int, default=0
        # The number of duplicated features, drawn randomly from the informative
        # and the redundant features.

    n_classes : int, default=2
        # The number of classes (or labels) of the classification problem.

    n_clusters_per_class : int, default=2
        # The number of clusters per class.

    weights : array-like of shape (n_classes,) or (n_classes - 1,),\
              default=None
        # The proportions of samples assigned to each class. If None, then
        # classes are balanced. Note that if ``len(weights) == n_classes - 1``,
        # then the last class weight is automatically inferred.
        # More than ``n_samples`` samples may be returned if the sum of
        # ``weights`` exceeds 1. Note that the actual class proportions will
        # not exactly match ``weights`` when ``flip_y`` isn't 0.

    flip_y : float, default=0.01
        # The fraction of samples whose class is assigned randomly. Larger
        # values introduce noise in the labels and make the classification
        # task harder. Note that the default setting flip_y > 0 might lead
        # to less than ``n_classes`` in y in some cases.
    # 类别之间的分离因子，用于调整超立方体的大小，默认为1.0
    class_sep : float, default=1.0
        The factor multiplying the hypercube size.  Larger values spread
        out the clusters/classes and make the classification task easier.

    # 是否将类别放置在超立方体的顶点上，默认为True；若为False，则放置在随机多面体的顶点上
    hypercube : bool, default=True
        If True, the clusters are put on the vertices of a hypercube. If
        False, the clusters are put on the vertices of a random polytope.

    # 将特征向量平移指定值，默认为0.0；如果为None，则特征向量会被[-class_sep, class_sep]范围内的随机值平移
    shift : float, ndarray of shape (n_features,) or None, default=0.0
        Shift features by the specified value. If None, then features
        are shifted by a random value drawn in [-class_sep, class_sep].

    # 将特征向量乘以指定值，默认为1.0；如果为None，则特征向量会被[1, 100]范围内的随机值缩放
    scale : float, ndarray of shape (n_features,) or None, default=1.0
        Multiply features by the specified value. If None, then features
        are scaled by a random value drawn in [1, 100]. Note that scaling
        happens after shifting.

    # 是否对样本和特征进行洗牌，默认为True
    shuffle : bool, default=True
        Shuffle the samples and the features.

    # 随机数生成器的种子值，用于确定数据集创建中的随机数生成；设置为int值可实现多次函数调用时的可重现输出
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    # 返回值
    Returns
    -------
    # 生成的样本特征矩阵，形状为(n_samples, n_features)
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    # 每个样本的整数标签，形状为(n_samples,)
    y : ndarray of shape (n_samples,)
        The integer labels for class membership of each sample.

    # 相关函数
    See Also
    --------
    # 简化变体生成器
    make_blobs : Simplified variant.
    # 多标签任务的无关生成器
    make_multilabel_classification : Unrelated generator for multilabel tasks.

    # 注意事项
    Notes
    -----
    # 算法改编自Guyon [1]，用于生成"Madelon"数据集

    # 参考文献
    References
    ----------
    # [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
    #        selection benchmark", 2003.

    # 示例
    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(random_state=42)
    >>> X.shape
    (100, 20)
    >>> y.shape
    (100,)
    >>> list(y[:5])
    [0, 0, 1, 1, 0]
    """
    # 使用check_random_state函数根据给定的random_state生成一个随机数生成器实例
    generator = check_random_state(random_state)

    # 计算特征数、聚类数和样本数
    if n_informative + n_redundant + n_repeated > n_features:
        # 如果信息特征数、冗余特征数和重复特征数之和大于特征总数，抛出数值错误异常
        raise ValueError(
            "Number of informative, redundant and repeated "
            "features must sum to less than the number of total"
            " features"
        )
    # 使用log2函数以避免溢出错误
    if n_informative < np.log2(n_classes * n_clusters_per_class):
        # 如果信息特征数小于n_classes * n_clusters_per_class的二进制对数，抛出数值错误异常
        msg = "n_classes({}) * n_clusters_per_class({}) must be"
        msg += " smaller or equal 2**n_informative({})={}"
        raise ValueError(
            msg.format(n_classes, n_clusters_per_class, n_informative, 2**n_informative)
        )
    ```python`
    # 如果提供了权重，则验证其与类别数量的兼容性
    if weights is not None:
        # 权重列表长度必须与类别数相等或者比类别数少一
        if len(weights) not in [n_classes, n_classes - 1]:
            raise ValueError(
                "Weights specified but incompatible with number of classes."
            )
        # 如果权重列表长度比类别数少一，则进行调整
        if len(weights) == n_classes - 1:
            # 如果权重是列表，则在末尾加上一个权重值，使其总和为1
            if isinstance(weights, list):
                weights = weights + [1.0 - sum(weights)]
            else:
                # 如果权重不是列表，则调整其大小，并确保最后一个权重使总和为1
                weights = np.resize(weights, n_classes)
                weights[-1] = 1.0 - sum(weights[:-1])
    else:
        # 如果未提供权重，则默认为均匀分布的权重
        weights = [1.0 / n_classes] * n_classes
    
    # 计算无用特征的数量
    n_useless = n_features - n_informative - n_redundant - n_repeated
    # 计算簇的数量
    n_clusters = n_classes * n_clusters_per_class
    
    # 根据权重分配样本到各个簇中
    n_samples_per_cluster = [
        int(n_samples * weights[k % n_classes] / n_clusters_per_class)
        for k in range(n_clusters)
    ]
    
    # 确保样本数与分配的样本数一致
    for i in range(n_samples - sum(n_samples_per_cluster)):
        n_samples_per_cluster[i % n_clusters] += 1
    
    # 初始化特征矩阵 X 和标签向量 y
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    
    # 生成聚类的中心点，构建多面体的顶点
    centroids = _generate_hypercube(n_clusters, n_informative, generator).astype(
        float, copy=False
    )
    centroids *= 2 * class_sep
    centroids -= class_sep
    if not hypercube:
        centroids *= generator.uniform(size=(n_clusters, 1))
        centroids *= generator.uniform(size=(1, n_informative))
    
    # 初始情况下，从标准正态分布中抽取信息特征
    X[:, :n_informative] = generator.standard_normal(size=(n_samples, n_informative))
    
    # 创建每个簇；这是 make_blobs 的一个变体
    stop = 0
    for k, centroid in enumerate(centroids):
        start, stop = stop, stop + n_samples_per_cluster[k]
        y[start:stop] = k % n_classes  # 分配标签
        X_k = X[start:stop, :n_informative]  # 切片得到簇的视图
    
        A = 2 * generator.uniform(size=(n_informative, n_informative)) - 1
        X_k[...] = np.dot(X_k, A)  # 引入随机协方差
    
        X_k += centroid  # 将簇移动到一个顶点
    
    # 创建冗余特征
    if n_redundant > 0:
        B = 2 * generator.uniform(size=(n_informative, n_redundant)) - 1
        X[:, n_informative : n_informative + n_redundant] = np.dot(
            X[:, :n_informative], B
        )
    
    # 重复某些特征
    if n_repeated > 0:
        n = n_informative + n_redundant
        indices = ((n - 1) * generator.uniform(size=n_repeated) + 0.5).astype(np.intp)
        X[:, n : n + n_repeated] = X[:, indices]
    
    # 填充无用特征
    if n_useless > 0:
        X[:, -n_useless:] = generator.standard_normal(size=(n_samples, n_useless))
    
    # 随机替换标签
    if flip_y >= 0.0:
        flip_mask = generator.uniform(size=n_samples) < flip_y
        y[flip_mask] = generator.randint(n_classes, size=flip_mask.sum())
    
    # 随机平移和缩放
    # 如果没有指定偏移量，则生成一个随机的偏移量数组，用于特征向量 X 的每个特征
    if shift is None:
        shift = (2 * generator.uniform(size=n_features) - 1) * class_sep
    # 将特征向量 X 按照指定的偏移量进行平移
    X += shift

    # 如果没有指定缩放比例，则生成一个随机的缩放比例数组，用于特征向量 X 的每个特征
    if scale is None:
        scale = 1 + 100 * generator.uniform(size=n_features)
    # 将特征向量 X 按照指定的缩放比例进行缩放
    X *= scale

    # 如果 shuffle 参数为 True，则随机排列样本和特征
    if shuffle:
        # 随机排列样本
        X, y = util_shuffle(X, y, random_state=generator)

        # 随机排列特征索引
        indices = np.arange(n_features)
        generator.shuffle(indices)
        # 将特征向量 X 的列按照随机排列的特征索引重新排序
        X[:, :] = X[:, indices]

    # 返回经过偏移、缩放和（如果指定）随机排列后的特征向量 X 和对应的目标向量 y
    return X, y
@validate_params(
    {
        "n_samples": [Interval(Integral, 1, None, closed="left")],
        "n_features": [Interval(Integral, 1, None, closed="left")],
        "n_classes": [Interval(Integral, 1, None, closed="left")],
        "n_labels": [Interval(Integral, 0, None, closed="left")],
        "length": [Interval(Integral, 1, None, closed="left")],
        "allow_unlabeled": ["boolean"],
        "sparse": ["boolean"],
        "return_indicator": [StrOptions({"dense", "sparse"}), "boolean"],
        "return_distributions": ["boolean"],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
# 定义函数装饰器，用于验证函数参数的合法性
def make_multilabel_classification(
    n_samples=100,
    n_features=20,
    *,
    n_classes=5,
    n_labels=2,
    length=50,
    allow_unlabeled=True,
    sparse=False,
    return_indicator="dense",
    return_distributions=False,
    random_state=None,
):
    """Generate a random multilabel classification problem.

    For each sample, the generative process is:
        - pick the number of labels: n ~ Poisson(n_labels)
        - n times, choose a class c: c ~ Multinomial(theta)
        - pick the document length: k ~ Poisson(length)
        - k times, choose a word: w ~ Multinomial(theta_c)

    In the above process, rejection sampling is used to make sure that
    n is never zero or more than `n_classes`, and that the document length
    is never zero. Likewise, we reject classes which have already been chosen.

    For an example of usage, see
    :ref:`sphx_glr_auto_examples_datasets_plot_random_multilabel_dataset.py`.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.

    n_features : int, default=20
        The total number of features.

    n_classes : int, default=5
        The number of classes of the classification problem.

    n_labels : int, default=2
        The average number of labels per instance. More precisely, the number
        of labels per sample is drawn from a Poisson distribution with
        ``n_labels`` as its expected value, but samples are bounded (using
        rejection sampling) by ``n_classes``, and must be nonzero if
        ``allow_unlabeled`` is False.

    length : int, default=50
        The sum of the features (number of words if documents) is drawn from
        a Poisson distribution with this expected value.

    allow_unlabeled : bool, default=True
        If ``True``, some instances might not belong to any class.

    sparse : bool, default=False
        If ``True``, return a sparse feature matrix.

        .. versionadded:: 0.17
           parameter to allow *sparse* output.

    return_indicator : {'dense', 'sparse'} or False, default='dense'
        If ``'dense'`` return ``Y`` in the dense binary indicator format. If
        ``'sparse'`` return ``Y`` in the sparse binary indicator format.
        ``False`` returns a list of lists of labels.
    return_distributions : bool, default=False
        如果为 True，则返回从数据集中绘制的先验类概率和给定类别的特征条件概率。

    random_state : int, RandomState 实例或 None，默认为 None
        确定数据集创建时的随机数生成。传递一个整数以在多次函数调用中获得可重复的输出。
        参见术语表中的“随机状态”。

    Returns
    -------
    X : 形状为 (n_samples, n_features) 的 ndarray
        生成的样本。

    Y : 形状为 (n_samples, n_classes) 的 {ndarray, 稀疏矩阵}
        标签集。稀疏矩阵应采用 CSR 格式。

    p_c : 形状为 (n_classes,) 的 ndarray
        每个类别被抽取的概率。仅在 return_distributions=True 时返回。

    p_w_c : 形状为 (n_features, n_classes) 的 ndarray
        在给定每个类别的情况下，每个特征被抽取的概率。仅在 return_distributions=True 时返回。

    Examples
    --------
    >>> from sklearn.datasets import make_multilabel_classification
    >>> X, y = make_multilabel_classification(n_labels=3, random_state=42)
    >>> X.shape
    (100, 20)
    >>> y.shape
    (100, 5)
    >>> list(y[:3])
    [array([1, 1, 0, 1, 0]), array([0, 1, 1, 1, 0]), array([0, 1, 0, 0, 0])]
    """

    # 使用给定的 random_state 创建随机数生成器
    generator = check_random_state(random_state)
    
    # 生成每个类别被抽取的概率
    p_c = generator.uniform(size=n_classes)
    p_c /= p_c.sum()  # 将概率归一化为总和为 1
    cumulative_p_c = np.cumsum(p_c)  # 计算累积概率分布
    
    # 生成每个特征在给定每个类别的情况下被抽取的概率
    p_w_c = generator.uniform(size=(n_features, n_classes))
    p_w_c /= np.sum(p_w_c, axis=0)  # 归一化每列的概率分布

    def sample_example():
        _, n_classes = p_w_c.shape

        # 使用拒绝抽样选择每个文档的非零标签数
        y_size = n_classes + 1
        while (not allow_unlabeled and y_size == 0) or y_size > n_classes:
            y_size = generator.poisson(n_labels)

        # 选择 n_classes 个类别
        y = set()
        while len(y) != y_size:
            # 按概率 P(c) 选择一个类别
            c = np.searchsorted(cumulative_p_c, generator.uniform(size=y_size - len(y)))
            y.update(c)
        y = list(y)

        # 使用拒绝抽样选择非零文档长度
        n_words = 0
        while n_words == 0:
            n_words = generator.poisson(length)

        # 生成长度为 n_words 的文档
        if len(y) == 0:
            # 如果样本不属于任何类别，则生成噪声词
            words = generator.randint(n_features, size=n_words)
            return words, y

        # 从选定的类别中替换抽样词汇
        cumulative_p_w_sample = p_w_c.take(y, axis=1).sum(axis=1).cumsum()
        cumulative_p_w_sample /= cumulative_p_w_sample[-1]
        words = np.searchsorted(cumulative_p_w_sample, generator.uniform(size=n_words))
        return words, y

    # 创建数组来存储稀疏矩阵的数据索引
    X_indices = array.array("i")
    # 创建数组来存储稀疏矩阵的指针
    X_indptr = array.array("i", [0])
    # 初始化 Y 为空列表
    Y = []
    # 对于给定的样本数，循环进行样本采样和处理
    for i in range(n_samples):
        # 调用函数sample_example()，返回样本的单词列表和对应的标签y
        words, y = sample_example()
        # 将样本的单词列表扩展到X_indices中
        X_indices.extend(words)
        # 将当前X_indices的长度添加到X_indptr中，表示新的一个样本的起始位置
        X_indptr.append(len(X_indices))
        # 将标签y添加到Y列表中
        Y.append(y)
    
    # 创建一个元素全为1的数组，用于构造稀疏矩阵X
    X_data = np.ones(len(X_indices), dtype=np.float64)
    # 使用scipy库的csr_matrix函数构建稀疏矩阵X
    X = sp.csr_matrix((X_data, X_indices, X_indptr), shape=(n_samples, n_features))
    # 移除X中重复的元素
    X.sum_duplicates()
    
    # 如果不需要稀疏矩阵，则将X转换为密集数组
    if not sparse:
        X = X.toarray()

    # 根据return_indicator参数的不同取值，进行不同的处理
    # 当return_indicator为True、"sparse"或"dense"时，创建MultiLabelBinarizer对象lb
    if return_indicator in (True, "sparse", "dense"):
        lb = MultiLabelBinarizer(sparse_output=(return_indicator == "sparse"))
        # 对Y进行多标签二进制化处理，适应于n_classes个标签的情况
        Y = lb.fit([range(n_classes)]).transform(Y)
    
    # 如果需要返回类别分布信息，则返回X, Y, p_c, p_w_c
    if return_distributions:
        return X, Y, p_c, p_w_c
    # 否则只返回X和Y
    return X, Y
# 对参数进行验证，确保参数的类型和取值范围符合要求
@validate_params(
    {
        "n_samples": [Interval(Integral, 1, None, closed="left")],
        "n_features": [Interval(Integral, 1, None, closed="left")],
        "n_informative": [Interval(Integral, 0, None, closed="left")],
        "n_targets": [Interval(Integral, 1, None, closed="left")],
        "bias": [Interval(Real, None, None, closed="neither")],
        "effective_rank": [Interval(Integral, 1, None, closed="left"), None],
        "tail_strength": [Interval(Real, 0, 1, closed="both")],
        "noise": [Interval(Real, 0, None, closed="left")],
        "shuffle": ["boolean"],
        "coef": ["boolean"],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
# 生成一个随机回归问题的数据集
def make_regression(
    n_samples=100,
    n_features=100,
    *,
    n_informative=10,
    n_targets=1,
    bias=0.0,
    effective_rank=None,
    tail_strength=0.5,
    noise=0.0,
    shuffle=True,
    coef=False,
    random_state=None,
):
    """Generate a random regression problem.

    The input set can either be well conditioned (by default) or have a low
    rank-fat tail singular profile. See :func:`make_low_rank_matrix` for
    more details.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.

    n_features : int, default=100
        The number of features.

    n_informative : int, default=10
        The number of informative features.

    n_targets : int, default=1
        The number of regression targets.

    bias : float, default=0.0
        The bias term in the underlying linear model.

    effective_rank : int or None, default=None
        The approximate number of singular vectors required to explain
        the data. If None, the data is well-conditioned.

    tail_strength : float, default=0.5
        The relative importance of the fat noisy tail of the singular values
        profile. When set to 0.0, the data is well-conditioned.

    noise : float, default=0.0
        The standard deviation of the gaussian noise applied to the output.

    shuffle : bool, default=True
        Whether to shuffle the samples and the features.

    coef : bool, default=False
        Whether to return the coefficients of the underlying linear model.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        The regression targets.

    See Also
    --------
    make_low_rank_matrix : Generate a low rank matrix.

    Notes
    -----
    The regression coefficient for each informative feature is sampled
    from a uniform distribution centered around 0.0.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
    >>> X.shape
    (1000, 20)
    >>> y.shape
    (1000,)
    """
    # 根据随机数种子生成确定性随机数生成器
    rs = check_random_state(random_state)

    shape = (n_samples, n_features)
    # 从标准正态分布中生成随机样本，并按指定形状重新排列
    X = rs.normal(size=shape).reshape(shape)
    # 根据特定条件生成回归目标值
    y = ((X**2.0).sum(axis=1) > 9.34).astype(np.float64, copy=False)
    y[y == 0.0] = -1.0

    return X, y
    The output is generated by applying a (potentially biased) random linear
    regression model with `n_informative` nonzero regressors to the previously
    generated input and some gaussian centered noise with some adjustable
    scale.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.

    n_features : int, default=100
        The number of features.

    n_informative : int, default=10
        The number of informative features, i.e., the number of features used
        to build the linear model used to generate the output.

    n_targets : int, default=1
        The number of regression targets, i.e., the dimension of the y output
        vector associated with a sample. By default, the output is a scalar.

    bias : float, default=0.0
        The bias term in the underlying linear model.

    effective_rank : int, default=None
        If not None:
            The approximate number of singular vectors required to explain most
            of the input data by linear combinations. Using this kind of
            singular spectrum in the input allows the generator to reproduce
            the correlations often observed in practice.
        If None:
            The input set is well conditioned, centered and gaussian with
            unit variance.

    tail_strength : float, default=0.5
        The relative importance of the fat noisy tail of the singular values
        profile if `effective_rank` is not None. When a float, it should be
        between 0 and 1.

    noise : float, default=0.0
        The standard deviation of the gaussian noise applied to the output.

    shuffle : bool, default=True
        Shuffle the samples and the features.

    coef : bool, default=False
        If True, the coefficients of the underlying linear model are returned.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The input samples.

    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        The output values.

    coef : ndarray of shape (n_features,) or (n_features, n_targets)
        The coefficient of the underlying linear model. It is returned only if
        coef is True.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=5, n_features=2, noise=1, random_state=42)
    >>> X
    array([[ 0.4967..., -0.1382... ],
        [ 0.6476...,  1.523...],
        [-0.2341..., -0.2341...],
        [-0.4694...,  0.5425...],
        [ 1.579...,  0.7674...]])
    >>> y
    array([  6.737...,  37.79..., -10.27...,   0.4017...,   42.22...])
    ```
    # 确定实际需要的信息数量，不超过特征数和信息数
    n_informative = min(n_features, n_informative)
    # 检查随机状态并生成随机数生成器
    generator = check_random_state(random_state)

    if effective_rank is None:
        # 随机生成一个条件良好的输入集
        X = generator.standard_normal(size=(n_samples, n_features))

    else:
        # 随机生成一个低秩、长尾的输入集
        X = make_low_rank_matrix(
            n_samples=n_samples,
            n_features=n_features,
            effective_rank=effective_rank,
            tail_strength=tail_strength,
            random_state=generator,
        )

    # 创建一个真实模型，其中只有 n_informative 个特征非零
    # 其他特征与 y 不相关，应被稀疏正则化器（如 L1 或弹性网）忽略
    ground_truth = np.zeros((n_features, n_targets))
    ground_truth[:n_informative, :] = 100 * generator.uniform(
        size=(n_informative, n_targets)
    )

    # 计算输出 y，基于 X 和 ground_truth，并添加偏置
    y = np.dot(X, ground_truth) + bias

    # 添加噪声
    if noise > 0.0:
        y += generator.normal(scale=noise, size=y.shape)

    # 随机排列样本和特征
    if shuffle:
        # 混洗 X 和 y
        X, y = util_shuffle(X, y, random_state=generator)

        # 生成特征索引并混洗
        indices = np.arange(n_features)
        generator.shuffle(indices)
        # 按照索引重新排列 X 的列
        X[:, :] = X[:, indices]
        # 根据索引重新排列 ground_truth
        ground_truth = ground_truth[indices]

    # 压缩 y，去除多余的维度
    y = np.squeeze(y)

    # 如果需要系数，则返回 X、y 和 ground_truth；否则，只返回 X 和 y
    if coef:
        return X, y, np.squeeze(ground_truth)
    else:
        return X, y
@validate_params(
    {
        "n_samples": [Interval(Integral, 0, None, closed="left"), tuple],  # 参数验证装饰器，验证 n_samples 的类型和取值范围
        "shuffle": ["boolean"],  # 参数验证装饰器，验证 shuffle 是否为布尔类型
        "noise": [Interval(Real, 0, None, closed="left"), None],  # 参数验证装饰器，验证 noise 的类型和取值范围
        "random_state": ["random_state"],  # 参数验证装饰器，验证 random_state 的类型
        "factor": [Interval(Real, 0, 1, closed="left")],  # 参数验证装饰器，验证 factor 的取值范围
    },
    prefer_skip_nested_validation=True,  # 参数验证装饰器的选项，跳过嵌套验证
)
def make_circles(
    n_samples=100, *, shuffle=True, noise=None, random_state=None, factor=0.8
):
    """Make a large circle containing a smaller circle in 2d.

    A simple toy dataset to visualize clustering and classification
    algorithms.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int or tuple of shape (2,), dtype=int, default=100
        If int, it is the total number of points generated.
        For odd numbers, the inner circle will have one point more than the
        outer circle.
        If two-element tuple, number of points in outer circle and inner
        circle.

        .. versionchanged:: 0.23
           Added two-element tuple.

    shuffle : bool, default=True
        Whether to shuffle the samples.

    noise : float, default=None
        Standard deviation of Gaussian noise added to the data.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset shuffling and noise.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    factor : float, default=.8
        Scale factor between inner and outer circle in the range `[0, 1)`.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The integer labels (0 or 1) for class membership of each sample.

    Examples
    --------
    >>> from sklearn.datasets import make_circles
    >>> X, y = make_circles(random_state=42)
    >>> X.shape
    (100, 2)
    >>> y.shape
    (100,)
    >>> list(y[:5])
    [1, 1, 1, 0, 0]
    """
    if isinstance(n_samples, numbers.Integral):
        n_samples_out = n_samples // 2  # 计算外圈样本数
        n_samples_in = n_samples - n_samples_out  # 计算内圈样本数
    else:  # n_samples 是一个元组
        if len(n_samples) != 2:
            raise ValueError("When a tuple, n_samples must have exactly two elements.")
        n_samples_out, n_samples_in = n_samples  # 解包得到外圈和内圈样本数

    generator = check_random_state(random_state)  # 使用 random_state 创建随机数生成器对象
    # 生成外圈和内圈均匀分布的角度数组，并转换成 x 和 y 坐标
    linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
    linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
    outer_circ_x = np.cos(linspace_out)
    outer_circ_y = np.sin(linspace_out)
    inner_circ_x = np.cos(linspace_in) * factor
    inner_circ_y = np.sin(linspace_in) * factor

    # 按行堆叠外圈和内圈的 x 和 y 坐标，形成生成样本的 ndarray
    X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
    ).T
    # 创建一个长度为 n_samples_out + n_samples_in 的一维数组 y，
    # 前 n_samples_out 个元素为 0，后 n_samples_in 个元素为 1
    y = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)]
    )
    
    # 如果 shuffle 参数为 True，则对数组 X 和 y 进行随机重排
    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)
    
    # 如果 noise 参数不为 None，则对数组 X 中的每个元素添加高斯噪声，
    # 噪声的标准差由 noise 控制，使用 generator 生成随机数
    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)
    
    # 返回经过处理的特征矩阵 X 和对应的标签数组 y
    return X, y
# 使用装饰器 @validate_params 对 make_moons 函数进行参数验证
@validate_params(
    {
        "n_samples": [Interval(Integral, 1, None, closed="left"), tuple],  # n_samples 参数的验证要求
        "shuffle": ["boolean"],  # shuffle 参数的验证要求
        "noise": [Interval(Real, 0, None, closed="left"), None],  # noise 参数的验证要求
        "random_state": ["random_state"],  # random_state 参数的验证要求
    },
    prefer_skip_nested_validation=True,  # 设置装饰器的参数，优先跳过嵌套验证
)
    {
        # 参数定义：
        "n_samples": [Interval(Integral, 1, None, closed="left"), "array-like"],  # n_samples 是一个整数区间（至少为1），或者类似数组
        "n_features": [Interval(Integral, 1, None, closed="left")],  # n_features 是一个整数（至少为1）
        "centers": [Interval(Integral, 1, None, closed="left"), "array-like", None],  # centers 是一个整数区间（至少为1），或者类似数组，或者为None
        "cluster_std": [Interval(Real, 0, None, closed="left"), "array-like"],  # cluster_std 是一个实数区间（至少为0），或者类似数组
        "center_box": [tuple],  # center_box 是一个元组
        "shuffle": ["boolean"],  # shuffle 是一个布尔值
        "random_state": ["random_state"],  # random_state 是一个随机状态对象
        "return_centers": ["boolean"],  # return_centers 是一个布尔值
    },
    # prefer_skip_nested_validation 设置为 True，偏好跳过嵌套验证
# 定义生成聚类数据的函数，生成多维正态分布的“斑点”数据集

"""Generate isotropic Gaussian blobs for clustering.

For an example of usage, see
:ref:`sphx_glr_auto_examples_datasets_plot_random_dataset.py`.

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
n_samples : int or array-like, default=100
    如果是整数，则表示总共生成的数据点数，平均分布在各个聚类中。
    如果是数组，数组的每个元素表示每个聚类中的样本数。

    .. versionchanged:: v0.20
        可以将数组传递给 ``n_samples`` 参数

n_features : int, default=2
    每个样本的特征数。

centers : int or array-like of shape (n_centers, n_features), default=None
    要生成的中心数量或固定的中心位置。
    如果 n_samples 是整数且 centers 是 None，则生成 3 个中心。
    如果 n_samples 是数组，centers 必须是 None 或与 n_samples 长度相同的数组。

cluster_std : float or array-like of float, default=1.0
    聚类的标准差。

center_box : tuple of float (min, max), default=(-10.0, 10.0)
    在随机生成中心时，每个聚类中心的边界框。

shuffle : bool, default=True
    是否对样本进行洗牌。

random_state : int, RandomState instance or None, default=None
    确定数据集创建的随机数生成。传递整数可实现跨多次函数调用的可重复输出。
    参见 :term:`Glossary <random_state>`。

return_centers : bool, default=False
    如果为 True，则返回每个聚类的中心点。

    .. versionadded:: 0.23

Returns
-------
X : ndarray of shape (n_samples, n_features)
    生成的样本。

y : ndarray of shape (n_samples,)
    每个样本的聚类成员的整数标签。

centers : ndarray of shape (n_centers, n_features)
    每个聚类的中心点。只有在 ``return_centers=True`` 时返回。

See Also
--------
make_classification : 更复杂的变体。

Examples
--------
>>> from sklearn.datasets import make_blobs
>>> X, y = make_blobs(n_samples=10, centers=3, n_features=2,
...                   random_state=0)
>>> print(X.shape)
(10, 2)
>>> y
array([0, 0, 1, 0, 2, 2, 2, 1, 1, 0])
>>> X, y = make_blobs(n_samples=[3, 3, 4], centers=None, n_features=2,
...                   random_state=0)
>>> print(X.shape)
(10, 2)
>>> y
array([0, 1, 2, 0, 2, 2, 2, 1, 1, 0])
"""

# 使用传入的 random_state 参数创建一个随机数生成器
generator = check_random_state(random_state)
    # 检查 n_samples 是否是整数类型
    if isinstance(n_samples, numbers.Integral):
        # 如果 centers 参数为 None，则设置默认为 3
        if centers is None:
            centers = 3

        # 检查 centers 是否是整数类型
        if isinstance(centers, numbers.Integral):
            # 设置 n_centers 为 centers 的值
            n_centers = centers
            # 使用 generator 生成均匀分布的中心点
            centers = generator.uniform(
                center_box[0], center_box[1], size=(n_centers, n_features)
            )

        else:
            # 检查 centers 是否为数组，并获取其特征数和中心点数
            centers = check_array(centers)
            n_features = centers.shape[1]
            n_centers = centers.shape[0]

    else:
        # 设置 n_centers 为 n_samples 的长度
        n_centers = len(n_samples)
        # 如果 centers 参数为 None，则使用 generator 生成均匀分布的中心点
        if centers is None:
            centers = generator.uniform(
                center_box[0], center_box[1], size=(n_centers, n_features)
            )
        # 检查 centers 是否可迭代
        if not isinstance(centers, Iterable):
            raise ValueError(
                "Parameter `centers` must be array-like. Got {!r} instead".format(
                    centers
                )
            )
        # 检查 centers 和 n_samples 的长度是否一致
        if len(centers) != n_centers:
            raise ValueError(
                "Length of `n_samples` not consistent with number of "
                f"centers. Got n_samples = {n_samples} and centers = {centers}"
            )
        # 检查 centers 是否为数组，并获取其特征数
        centers = check_array(centers)
        n_features = centers.shape[1]

    # 如果 cluster_std 是列表且长度不一致，则引发异常
    if hasattr(cluster_std, "__len__") and len(cluster_std) != n_centers:
        raise ValueError(
            "Length of `clusters_std` not consistent with "
            "number of centers. Got centers = {} "
            "and cluster_std = {}".format(centers, cluster_std)
        )

    # 如果 cluster_std 是实数，则创建一个与 centers 长度相同的标准差数组
    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.full(len(centers), cluster_std)

    # 如果 n_samples 是可迭代对象，则将其赋值给 n_samples_per_center
    if isinstance(n_samples, Iterable):
        n_samples_per_center = n_samples
    else:
        # 否则，根据 n_samples 和 n_centers 计算每个中心点的样本数
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers

        # 将剩余的样本数分配给前面的几个中心点
        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1

    # 计算累积样本数
    cum_sum_n_samples = np.cumsum(n_samples_per_center)
    # 创建 X 和 y 数组用于存储数据和标签
    X = np.empty(shape=(sum(n_samples_per_center), n_features), dtype=np.float64)
    y = np.empty(shape=(sum(n_samples_per_center),), dtype=int)

    # 根据每个中心点的样本数和标准差生成数据和标签
    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        start_idx = cum_sum_n_samples[i - 1] if i > 0 else 0
        end_idx = cum_sum_n_samples[i]
        X[start_idx:end_idx] = generator.normal(
            loc=centers[i], scale=std, size=(n, n_features)
        )
        y[start_idx:end_idx] = i

    # 如果 shuffle 参数为 True，则对 X 和 y 进行打乱操作
    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    # 如果 return_centers 参数为 True，则返回 X, y 和 centers
    if return_centers:
        return X, y, centers
    else:
        # 否则，只返回 X 和 y
        return X, y
@validate_params(
    {
        "n_samples": [Interval(Integral, 1, None, closed="left")],
        "noise": [Interval(Real, 0, None, closed="left")],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
定义函数装饰器，用于验证函数参数的合法性，确保参数满足指定的区间和类型要求

def make_friedman2(n_samples=100, *, noise=0.0, random_state=None):
    """Generate the "Friedman #2" regression problem.

    This dataset is described in Friedman [1] and Breiman [2].
    生成“Friedman #2”回归问题的数据集，详见Friedman [1]和Breiman [2]。
    # 生成随机数发生器，用于生成数据的随机性
    generator = check_random_state(random_state)
    
    # 生成包含 n_samples 行，每行4列的随机数矩阵，数值均匀分布在 [0, 1) 区间内
    X = generator.uniform(size=(n_samples, 4))
    
    # 将第一列的值缩放到 [0, 100) 区间内
    X[:, 0] *= 100
    
    # 将第二列的值缩放到 [40π, 560π) 区间内
    X[:, 1] *= 520 * np.pi
    X[:, 1] += 40 * np.pi
    
    # 将第四列的值缩放到 [1, 11) 区间内
    X[:, 3] *= 10
    X[:, 3] += 1
    
    # 根据给定的公式计算 y 值，公式中包含了随机噪声
    y = (
        X[:, 0] ** 2 + (X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) ** 2
    ) ** 0.5 + noise * generator.standard_normal(size=(n_samples,))
    
    # 返回生成的 X 和 y 数组
    return X, y
@validate_params(
    {
        "n_samples": [Interval(Integral, 1, None, closed="left")],
        "n_features": [Interval(Integral, 1, None, closed="left")],
        "effective_rank": [Interval(Integral, 1, None, closed="left")],
        "tail_strength": [Interval(Real, 0, 1, closed="both")],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
# 参数验证装饰器，确保参数满足指定的类型和范围要求
def make_low_rank_matrix(
    n_samples=100,
    n_features=100,
    *,
    effective_rank=10,
    tail_strength=0.5,
    random_state=None,
):
    """Generate a mostly low rank matrix with bell-shaped singular values.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples.

    n_features : int, default=100
        Number of features.

    effective_rank : int, default=10
        Approximate number of singular vectors to construct the matrix.

    tail_strength : float, default=0.5
        Relative importance of the fat noisy tail of the singular values profile.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Generated matrix.

    Examples
    --------
    >>> from sklearn.datasets import make_low_rank_matrix
    >>> X = make_low_rank_matrix(n_samples=100, n_features=20,
    ...                          effective_rank=5, tail_strength=0.01,
    ...                          random_state=42)
    >>> X.shape
    (100, 20)
    """
    # 使用指定的 random_state 创建一个随机数生成器
    generator = check_random_state(random_state)

    # 生成 n_samples x n_features 的均匀分布的随机矩阵
    X = generator.uniform(size=(n_samples, n_features))

    # 根据指定的参数调整矩阵的奇异值分布
    singular_values = np.linspace(0, 1, min(n_samples, n_features))
    X *= singular_values

    # 返回生成的矩阵
    return X
    generator = check_random_state(random_state)
    # 使用给定的随机状态生成随机数发生器

    n = min(n_samples, n_features)
    # 计算样本数和特征数中的较小值作为奇异值分解的维度

    # Random (ortho normal) vectors
    u, _ = linalg.qr(
        generator.standard_normal(size=(n_samples, n)),
        mode="economic",
        check_finite=False,
    )
    # 生成正交归一化的随机向量 u，用 QR 分解生成，经济模式下不计算完全的 QR 分解

    v, _ = linalg.qr(
        generator.standard_normal(size=(n_features, n)),
        mode="economic",
        check_finite=False,
    )
    # 生成正交归一化的随机向量 v，用 QR 分解生成，经济模式下不计算完全的 QR 分解

    # Index of the singular values
    singular_ind = np.arange(n, dtype=np.float64)
    # 创建奇异值的索引数组，数据类型为 float64

    # Build the singular profile by assembling signal and noise components
    # 创建奇异值分布的 profile，结合信号和噪声成分
    low_rank = (1 - tail_strength) * np.exp(-1.0 * (singular_ind / effective_rank) ** 2)
    # 低秩部分，根据给定公式计算奇异值的信号部分

    tail = tail_strength * np.exp(-0.1 * singular_ind / effective_rank)
    # 尾部，根据给定公式计算奇异值的噪声部分

    s = np.identity(n) * (low_rank + tail)
    # 构建奇异值矩阵 s，由信号和噪声部分组成的对角矩阵

    return np.dot(np.dot(u, s), v.T)
    # 返回经过奇异值分解计算得到的矩阵 X，通过 u, s, v.T 的乘积得到
# 声明一个装饰器函数，用于验证函数参数的合法性
@validate_params(
    # 定义参数验证规则字典，包括以下几个参数：
    {
        "n_samples": [Interval(Integral, 1, None, closed="left")],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "n_features": [Interval(Integral, 1, None, closed="left")],
        "n_nonzero_coefs": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    },
    # 设置参数验证时是否跳过嵌套验证，默认为 True
    prefer_skip_nested_validation=True,
)
# 定义一个生成稀疏编码信号的函数，其参数包括：
def make_sparse_coded_signal(
    n_samples,             # 要生成的样本数量
    *,                     # 之后的参数必须使用关键字传递
    n_components,          # 字典中的组件数量
    n_features,            # 要生成数据集的特征数
    n_nonzero_coefs,       # 每个样本中非零系数的数量
    random_state=None,     # 随机数生成的种子或状态，默认为 None
):
    """Generate a signal as a sparse combination of dictionary elements.

    Returns matrices `Y`, `D` and `X` such that `Y = XD` where `X` is of shape
    `(n_samples, n_components)`, `D` is of shape `(n_components, n_features)`, and
    each row of `X` has exactly `n_nonzero_coefs` non-zero elements.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.

    n_components : int
        Number of components in the dictionary.

    n_features : int
        Number of features of the dataset to generate.

    n_nonzero_coefs : int
        Number of active (non-zero) coefficients in each sample.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features)
        The encoded signal (Y).

    dictionary : ndarray of shape (n_components, n_features)
        The dictionary with normalized components (D).

    code : ndarray of shape (n_samples, n_components)
        The sparse code such that each column of this matrix has exactly
        n_nonzero_coefs non-zero items (X).

    Examples
    --------
    >>> from sklearn.datasets import make_sparse_coded_signal
    >>> data, dictionary, code = make_sparse_coded_signal(
    ...     n_samples=50,
    ...     n_components=100,
    ...     n_features=10,
    ...     n_nonzero_coefs=4,
    ...     random_state=0
    ... )
    >>> data.shape
    (50, 10)
    >>> dictionary.shape
    (100, 10)
    >>> code.shape
    (50, 100)
    """
    # 使用给定的 random_state 参数生成随机数生成器对象
    generator = check_random_state(random_state)

    # 生成字典 D
    D = generator.standard_normal(size=(n_features, n_components))
    D /= np.sqrt(np.sum((D**2), axis=0))

    # 生成稀疏编码 X
    X = np.zeros((n_components, n_samples))
    for i in range(n_samples):
        idx = np.arange(n_components)
        generator.shuffle(idx)
        idx = idx[:n_nonzero_coefs]
        X[idx, i] = generator.standard_normal(size=n_nonzero_coefs)

    # 编码信号 Y
    Y = np.dot(D, X)

    # 转置以保持与 API 其他部分的形状一致
    Y, D, X = Y.T, D.T, X.T

    # 返回结果，并将结果中多余的维度去除
    return map(np.squeeze, (Y, D, X))
    {
        # 定义参数字典，描述每个参数的期望值和类型范围
        "n_samples": [Interval(Integral, 1, None, closed="left")],
        "n_features": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    },
    # 设置参数验证时优先跳过嵌套验证
    prefer_skip_nested_validation=True,
# 定义生成稀疏不相关设计的随机回归问题的函数
def make_sparse_uncorrelated(n_samples=100, n_features=10, *, random_state=None):
    """Generate a random regression problem with sparse uncorrelated design.

    This dataset is described in Celeux et al [1]. as::

        X ~ N(0, 1)
        y(X) = X[:, 0] + 2 * X[:, 1] - 2 * X[:, 2] - 1.5 * X[:, 3]

    Only the first 4 features are informative. The remaining features are
    useless.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.

    n_features : int, default=10
        The number of features.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The input samples.

    y : ndarray of shape (n_samples,)
        The output values.

    References
    ----------
    .. [1] G. Celeux, M. El Anbari, J.-M. Marin, C. P. Robert,
           "Regularization in regression: comparing Bayesian and frequentist
           methods in a poorly informative situation", 2009.

    Examples
    --------
    >>> from sklearn.datasets import make_sparse_uncorrelated
    >>> X, y = make_sparse_uncorrelated(random_state=0)
    >>> X.shape
    (100, 10)
    >>> y.shape
    (100,)
    """
    # 使用给定的随机状态创建随机数生成器
    generator = check_random_state(random_state)

    # 生成具有正态分布 N(0, 1) 的随机特征矩阵 X
    X = generator.normal(loc=0, scale=1, size=(n_samples, n_features))
    
    # 生成目标值 y，根据线性组合计算
    y = generator.normal(
        loc=(X[:, 0] + 2 * X[:, 1] - 2 * X[:, 2] - 1.5 * X[:, 3]),
        scale=np.ones(n_samples),
    )

    # 返回特征矩阵 X 和目标值 y
    return X, y


# 使用指定的参数验证装饰器来定义生成随机对称正定矩阵的函数
@validate_params(
    {
        "n_dim": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
def make_spd_matrix(n_dim, *, random_state=None):
    """Generate a random symmetric, positive-definite matrix.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_dim : int
        The matrix dimension.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_dim, n_dim)
        The random symmetric, positive-definite matrix.

    See Also
    --------
    make_sparse_spd_matrix: Generate a sparse symmetric definite positive matrix.

    Examples
    --------
    >>> from sklearn.datasets import make_spd_matrix
    >>> make_spd_matrix(n_dim=2, random_state=42)
    array([[2.09..., 0.34...],
           [0.34..., 0.21...]])
    """
    # 使用给定的随机状态创建随机数生成器
    generator = check_random_state(random_state)

    # 生成具有均匀分布的随机矩阵 A，用于构造正定矩阵
    A = generator.uniform(size=(n_dim, n_dim))
    # 使用线性代数库中的奇异值分解（SVD）函数，计算矩阵 A 的转置与自身的乘积，并返回其奇异值分解结果的三个组成部分 U, _, Vt
    U, _, Vt = linalg.svd(np.dot(A.T, A), check_finite=False)
    
    # 根据 SVD 分解得到的 U, Vt 和一个对角矩阵的乘积，生成一个新的矩阵 X
    X = np.dot(np.dot(U, 1.0 + np.diag(generator.uniform(size=n_dim))), Vt)
    
    # 返回生成的矩阵 X
    return X
@validate_params(
    {
        "n_dim": [Hidden(None), Interval(Integral, 1, None, closed="left")],
        "alpha": [Interval(Real, 0, 1, closed="both")],
        "norm_diag": ["boolean"],
        "smallest_coef": [Interval(Real, 0, 1, closed="both")],
        "largest_coef": [Interval(Real, 0, 1, closed="both")],
        "sparse_format": [
            StrOptions({"bsr", "coo", "csc", "csr", "dia", "dok", "lil"}),
            None,
        ],
        "random_state": ["random_state"],
        "dim": [
            Interval(Integral, 1, None, closed="left"),
            Hidden(StrOptions({"deprecated"})),
        ],
    },
    prefer_skip_nested_validation=True,
)
def make_sparse_spd_matrix(
    n_dim=None,
    *,
    alpha=0.95,
    norm_diag=False,
    smallest_coef=0.1,
    largest_coef=0.9,
    sparse_format=None,
    random_state=None,
    dim="deprecated",
):
    """Generate a sparse symmetric definite positive matrix.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_dim : int, default=1
        The size of the random matrix to generate.

        .. versionchanged:: 1.4
            Renamed from ``dim`` to ``n_dim``.

    alpha : float, default=0.95
        The probability that a coefficient is zero (see notes). Larger values
        enforce more sparsity. The value should be in the range 0 and 1.

    norm_diag : bool, default=False
        Whether to normalize the output matrix to make the leading diagonal
        elements all 1.

    smallest_coef : float, default=0.1
        The value of the smallest coefficient between 0 and 1.

    largest_coef : float, default=0.9
        The value of the largest coefficient between 0 and 1.

    sparse_format : str, default=None
        String representing the output sparse format, such as 'csc', 'csr', etc.
        If ``None``, return a dense numpy ndarray.

        .. versionadded:: 1.4

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    dim : int, default=1
        The size of the random matrix to generate.

        .. deprecated:: 1.4
            `dim` is deprecated and will be removed in 1.6.

    Returns
    -------
    prec : ndarray or sparse matrix of shape (dim, dim)
        The generated matrix. If ``sparse_format=None``, this would be an ndarray.
        Otherwise, this will be a sparse matrix of the specified format.

    See Also
    --------
    make_spd_matrix : Generate a random symmetric, positive-definite matrix.

    Notes
    -----
    The sparsity is actually imposed on the cholesky factor of the matrix.
    Thus alpha does not translate directly into the filling fraction of
    the matrix itself.

    Examples
    --------
    >>> from sklearn.datasets import make_sparse_spd_matrix

    """
    >>> make_sparse_spd_matrix(n_dim=4, norm_diag=False, random_state=42)
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])
    """
    random_state = check_random_state(random_state)

    # TODO(1.6): remove in 1.6
    # Also make sure to change `n_dim` default back to 1 and deprecate None
    # 如果 `n_dim` 不是 None 并且 `dim` 不是 "deprecated"，则抛出错误
    if n_dim is not None and dim != "deprecated":
        raise ValueError(
            "`dim` and `n_dim` cannot be both specified. Please use `n_dim` only "
            "as `dim` is deprecated in v1.4 and will be removed in v1.6."
        )

    # 如果 `dim` 不是 "deprecated"，发出警告，并使用 `_n_dim` 替代 `dim`
    if dim != "deprecated":
        warnings.warn(
            (
                "dim was deprecated in version 1.4 and will be removed in 1.6."
                "Please use ``n_dim`` instead."
            ),
            FutureWarning,
        )
        _n_dim = dim
    # 如果 `n_dim` 是 None，则将 `_n_dim` 设置为 1
    elif n_dim is None:
        _n_dim = 1
    else:
        _n_dim = n_dim

    # 创建一个空的 `_n_dim x _n_dim` 的矩阵 `chol`，并对角线设置为 `-1`
    chol = -sp.eye(_n_dim)

    # 生成一个稀疏矩阵 `aux`，大小为 `_n_dim x _n_dim`，密度为 `1 - alpha`
    aux = sp.random(
        m=_n_dim,
        n=_n_dim,
        density=1 - alpha,
        data_rvs=lambda x: random_state.uniform(
            low=smallest_coef, high=largest_coef, size=x
        ),
        random_state=random_state,
    )

    # 将 `aux` 转换为下三角形式，格式为 "csc"，以避免 "coo" 格式不支持切片操作
    aux = sp.tril(aux, k=-1, format="csc")

    # 随机排列 `_n_dim` 行，以避免最终的 SPD 矩阵中出现不对称
    permutation = random_state.permutation(_n_dim)
    aux = aux[permutation].T[permutation]  # 对 `aux` 进行行列置换
    chol += aux  # 更新 `chol` 矩阵

    # 计算精确逆矩阵 `prec`，即 `chol` 的转置乘以 `chol`
    prec = chol.T @ chol

    # 如果 `norm_diag` 为真，则对 `prec` 进行对角归一化处理
    if norm_diag:
        # 将对角线元素形成对角矩阵 `d`，然后对 `prec` 进行归一化处理
        d = sp.diags(1.0 / np.sqrt(prec.diagonal()))
        prec = d @ prec @ d

    # 如果 `sparse_format` 为 None，则返回稠密矩阵
    if sparse_format is None:
        return prec.toarray()
    else:
        # 否则，根据 `sparse_format` 返回相应格式的稀疏矩阵
        return prec.asformat(sparse_format)
@validate_params(
    {
        "n_samples": [Interval(Integral, 1, None, closed="left")],
        "noise": [Interval(Real, 0, None, closed="left")],
        "random_state": ["random_state"],
        # 参数验证装饰器，确保 n_samples 是大于等于 1 的整数，noise 是大于等于 0 的实数，random_state 是随机种子或 None
    },
    prefer_skip_nested_validation=True,
)
def make_s_curve(n_samples=100, *, noise=0.0, random_state=None):
    """Generate an S curve dataset.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=100
        The number of sample points on the S curve.

    noise : float, default=0.0
        The standard deviation of the gaussian noise.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        The points on the S curve.

    t : ndarray of shape (n_samples,)
        The univariate position of the sample according to the main dimension
        of the points in the manifold.
    """
    # 根据 random_state 创建一个随机数生成器
    generator = check_random_state(random_state)

    # 生成 S 曲线上的样本点
    t = 10 * generator.uniform(size=n_samples)
    x = np.sin(t)
    y = 2.0 * np.sign(t - 5.0) * np.abs(t - 5.0)**(2.0 / 3)
    z = np.cos(t)

    # 向样本点添加高斯噪声
    X = np.column_stack((x, y, z))
    X += noise * generator.standard_normal(size=X.shape)

    return X, t
    # 噪声水平，高斯噪声的标准偏差，默认为0.0
    noise : float, default=0.0

    # 随机数生成器的种子，用于创建数据集。可以是整数、RandomState实例或None，默认为None。
    # 通过设定一个整数可以保证在多次函数调用中得到可重复的输出。
    # 参见“术语表”中的“随机数生成器”。
    random_state : int, RandomState instance or None, default=None

    # 返回值
    # X：形状为(n_samples, 3)的ndarray
    #    数据点的坐标。
    # t：形状为(n_samples,)的ndarray
    #    样本在流形主维度上的一维位置。
    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        The points.

    t : ndarray of shape (n_samples,)
        The univariate position of the sample according
        to the main dimension of the points in the manifold.

    # 示例
    Examples
    --------
    >>> from sklearn.datasets import make_s_curve
    >>> X, t = make_s_curve(noise=0.05, random_state=0)
    >>> X.shape
    (100, 3)
    >>> t.shape
    (100,)
    """
    # 根据random_state确定随机数生成器
    generator = check_random_state(random_state)

    # 根据生成的t值计算数据点的坐标X
    t = 3 * np.pi * (generator.uniform(size=(1, n_samples)) - 0.5)
    X = np.empty(shape=(n_samples, 3), dtype=np.float64)
    X[:, 0] = np.sin(t)
    X[:, 1] = 2.0 * generator.uniform(size=n_samples)
    X[:, 2] = np.sign(t) * (np.cos(t) - 1)

    # 添加高斯噪声到数据点X中
    X += noise * generator.standard_normal(size=(3, n_samples)).T

    # 去除t的冗余维度
    t = np.squeeze(t)

    # 返回生成的数据点和位置信息
    return X, t
@validate_params(
    {
        "mean": ["array-like", None],  # 参数验证装饰器，验证 mean 参数是一个类数组或者 None
        "cov": [Interval(Real, 0, None, closed="left")],  # 验证 cov 参数是大于等于 0 的实数
        "n_samples": [Interval(Integral, 1, None, closed="left")],  # 验证 n_samples 参数是大于等于 1 的整数
        "n_features": [Interval(Integral, 1, None, closed="left")],  # 验证 n_features 参数是大于等于 1 的整数
        "n_classes": [Interval(Integral, 1, None, closed="left")],  # 验证 n_classes 参数是大于等于 1 的整数
        "shuffle": ["boolean"],  # 验证 shuffle 参数是布尔类型
        "random_state": ["random_state"],  # 验证 random_state 参数是随机状态对象
    },
    prefer_skip_nested_validation=True,
)
def make_gaussian_quantiles(
    *,
    mean=None,  # 均值，默认为 None
    cov=1.0,  # 协方差，默认为 1.0
    n_samples=100,  # 样本总数，默认为 100
    n_features=2,  # 每个样本的特征数，默认为 2
    n_classes=3,  # 类别数，默认为 3
    shuffle=True,  # 是否对样本进行洗牌，默认为 True
    random_state=None,  # 随机数生成器的种子或状态，默认为 None
):
    r"""Generate isotropic Gaussian and label samples by quantile.

    This classification dataset is constructed by taking a multi-dimensional
    standard normal distribution and defining classes separated by nested
    concentric multi-dimensional spheres such that roughly equal numbers of
    samples are in each class (quantiles of the :math:`\chi^2` distribution).

    For an example of usage, see
    :ref:`sphx_glr_auto_examples_datasets_plot_random_dataset.py`.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    mean : array-like of shape (n_features,), default=None
        The mean of the multi-dimensional normal distribution.
        If None then use the origin (0, 0, ...).

    cov : float, default=1.0
        The covariance matrix will be this value times the unit matrix. This
        dataset only produces symmetric normal distributions.

    n_samples : int, default=100
        The total number of points equally divided among classes.

    n_features : int, default=2
        The number of features for each sample.

    n_classes : int, default=3
        The number of classes.

    shuffle : bool, default=True
        Shuffle the samples.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The integer labels for quantile membership of each sample.

    Notes
    -----
    The dataset is from Zhu et al [1].

    References
    ----------
    .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    Examples
    --------
    >>> from sklearn.datasets import make_gaussian_quantiles
    >>> X, y = make_gaussian_quantiles(random_state=42)
    >>> X.shape
    (100, 2)
    >>> y.shape
    (100,)
    >>> list(y[:5])
    [2, 0, 1, 0, 2]
    """
    if n_samples < n_classes:
        raise ValueError("n_samples must be at least n_classes")

    generator = check_random_state(random_state)

    if mean is None:
        mean = np.zeros(n_features)
    else:
        mean = np.array(mean)

    # Build multivariate normal distribution
    # 用多变量正态分布生成数据集 X，以给定的均值 mean 和协方差 cov 为参数
    X = generator.multivariate_normal(mean, cov * np.identity(n_features), (n_samples,))
    
    # 根据数据点到原点的距离对数据集 X 进行排序
    idx = np.argsort(np.sum((X - mean[np.newaxis, :]) ** 2, axis=1))
    X = X[idx, :]
    
    # 将数据集 X 按照分位数标记分类，每个类别分配相同数量的样本
    step = n_samples // n_classes
    
    y = np.hstack(
        [
            np.repeat(np.arange(n_classes), step),  # 每个类别中前 step 个样本标记为相应的类别
            np.repeat(n_classes - 1, n_samples - step * n_classes),  # 剩余的样本标记为最后一个类别
        ]
    )
    
    # 如果 shuffle 参数为 True，则使用指定的随机状态 generator 对 X 和 y 进行洗牌
    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)
    
    # 返回生成的数据集 X 和对应的标签 y
    return X, y
# 使用给定的随机状态生成器，或者创建一个新的随机状态生成器
generator = check_random_state(random_state)

# 获取指定形状数据的行数和列数
n_rows, n_cols = shape

# 根据行数生成一个行索引的随机排列
row_idx = generator.permutation(n_rows)

# 根据列数生成一个列索引的随机排列
col_idx = generator.permutation(n_cols)

# 根据随机排列的行索引和列索引重新排列数据，生成一个新的结果数组
result = data[row_idx][:, col_idx]

# 返回重新排列后的数据数组、行索引和列索引
return result, row_idx, col_idx
    # 使用多项式分布生成每个簇中的行数和列数
    row_sizes = generator.multinomial(n_rows, np.repeat(1.0 / n_clusters, n_clusters))
    col_sizes = generator.multinomial(n_cols, np.repeat(1.0 / n_clusters, n_clusters))
    
    # 根据每个簇中的行数生成对应的行标签
    row_labels = np.hstack(
        [np.repeat(val, rep) for val, rep in zip(range(n_clusters), row_sizes)]
    )
    
    # 根据每个簇中的列数生成对应的列标签
    col_labels = np.hstack(
        [np.repeat(val, rep) for val, rep in zip(range(n_clusters), col_sizes)]
    )
    
    # 初始化一个指定形状和数据类型的全零矩阵
    result = np.zeros(shape, dtype=np.float64)
    
    # 根据行标签和列标签的组合，将对应位置的元素加上常数值
    for i in range(n_clusters):
        selector = np.outer(row_labels == i, col_labels == i)
        result[selector] += consts[i]
    
    # 如果指定了噪声值，则将结果矩阵添加符合正态分布的噪声
    if noise > 0:
        result += generator.normal(scale=noise, size=result.shape)
    
    # 如果需要进行洗牌操作，则对结果矩阵进行洗牌，并更新行标签和列标签
    if shuffle:
        result, row_idx, col_idx = _shuffle(result, random_state)
        row_labels = row_labels[row_idx]
        col_labels = col_labels[col_idx]
    
    # 生成行的布尔掩码矩阵，每行代表一个簇的成员情况
    rows = np.vstack([row_labels == c for c in range(n_clusters)])
    
    # 生成列的布尔掩码矩阵，每列代表一个簇的成员情况
    cols = np.vstack([col_labels == c for c in range(n_clusters)])
    
    # 返回最终的结果矩阵及其行列掩码矩阵
    return result, rows, cols
# 使用装饰器 validate_params 对函数参数进行验证，确保参数类型和取值范围符合要求
@validate_params(
    {
        "shape": [tuple],  # shape 参数必须是一个元组
        "n_clusters": [Interval(Integral, 1, None, closed="left"), "array-like"],  # n_clusters 参数可以是整数、大于等于1的区间、或者类数组对象
        "noise": [Interval(Real, 0, None, closed="left")],  # noise 参数是一个大于等于0的实数
        "minval": [Interval(Real, None, None, closed="neither")],  # minval 参数是一个开区间的实数
        "maxval": [Interval(Real, None, None, closed="neither")],  # maxval 参数是一个开区间的实数
        "shuffle": ["boolean"],  # shuffle 参数是一个布尔值
        "random_state": ["random_state"],  # random_state 参数可以是整数、RandomState 实例或者 None
    },
    prefer_skip_nested_validation=True,  # 设置了 prefer_skip_nested_validation 标志，优先跳过嵌套验证
)
def make_checkerboard(
    shape,
    n_clusters,
    *,
    noise=0.0,  # 默认值为 0.0
    minval=10,  # 默认值为 10
    maxval=100,  # 默认值为 100
    shuffle=True,  # 默认值为 True
    random_state=None,  # 默认值为 None
):
    """Generate an array with block checkerboard structure for biclustering.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    shape : tuple of shape (n_rows, n_cols)
        The shape of the result.

    n_clusters : int or array-like or shape (n_row_clusters, n_column_clusters)
        The number of row and column clusters.

    noise : float, default=0.0
        The standard deviation of the gaussian noise.

    minval : float, default=10
        Minimum value of a bicluster.

    maxval : float, default=100
        Maximum value of a bicluster.

    shuffle : bool, default=True
        Shuffle the samples.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape `shape`
        The generated array.

    rows : ndarray of shape (n_clusters, X.shape[0])
        The indicators for cluster membership of each row.

    cols : ndarray of shape (n_clusters, X.shape[1])
        The indicators for cluster membership of each column.

    See Also
    --------
    make_biclusters : Generate an array with constant block diagonal structure
        for biclustering.

    References
    ----------
    .. [1] Kluger, Y., Basri, R., Chang, J. T., & Gerstein, M. (2003).
        Spectral biclustering of microarray data: coclustering genes
        and conditions. Genome research, 13(4), 703-716.

    Examples
    --------
    >>> from sklearn.datasets import make_checkerboard
    >>> data, rows, columns = make_checkerboard(shape=(300, 300), n_clusters=10,
    ...                                         random_state=42)
    >>> data.shape
    (300, 300)
    >>> rows.shape
    (100, 300)
    >>> columns.shape
    (100, 300)
    >>> print(rows[0][:5], columns[0][:5])
    [False False False  True False] [False False False False False]
    """
    generator = check_random_state(random_state)  # 使用 check_random_state 函数根据 random_state 创建一个随机数生成器对象

    if hasattr(n_clusters, "__len__"):  # 如果 n_clusters 可以被长度检测（即是一个序列）
        n_row_clusters, n_col_clusters = n_clusters  # 则将 n_clusters 拆分为行和列的聚类数目
    else:
        n_row_clusters = n_col_clusters = n_clusters  # 否则假设行和列的聚类数目相同，都为 n_clusters

    # row and column clusters of approximately equal sizes
    n_rows, n_cols = shape  # 将 shape 解包为 n_rows 和 n_cols
    # 使用生成器生成服从多项分布的行大小，其中每个簇的概率相同
    row_sizes = generator.multinomial(
        n_rows, np.repeat(1.0 / n_row_clusters, n_row_clusters)
    )
    # 使用生成器生成服从多项分布的列大小，其中每个簇的概率相同
    col_sizes = generator.multinomial(
        n_cols, np.repeat(1.0 / n_col_clusters, n_col_clusters)
    )

    # 构建行标签数组，将每个簇的标签重复对应大小次数后合并
    row_labels = np.hstack(
        [np.repeat(val, rep) for val, rep in zip(range(n_row_clusters), row_sizes)]
    )
    # 构建列标签数组，将每个簇的标签重复对应大小次数后合并
    col_labels = np.hstack(
        [np.repeat(val, rep) for val, rep in zip(range(n_col_clusters), col_sizes)]
    )

    # 初始化一个指定形状和数据类型的全零数组
    result = np.zeros(shape, dtype=np.float64)
    # 遍历每个行簇和列簇，生成选择器并根据均匀分布填充结果数组
    for i in range(n_row_clusters):
        for j in range(n_col_clusters):
            selector = np.outer(row_labels == i, col_labels == j)
            result[selector] += generator.uniform(minval, maxval)

    # 如果存在噪声，则在结果数组上添加正态分布的噪声
    if noise > 0:
        result += generator.normal(scale=noise, size=result.shape)

    # 如果指定了 shuffle 参数，则对结果数组进行重新排序，并调整行列标签
    if shuffle:
        result, row_idx, col_idx = _shuffle(result, random_state)
        row_labels = row_labels[row_idx]
        col_labels = col_labels[col_idx]

    # 构建行选择矩阵，每行代表一个行簇的成员，对应标签为 True
    rows = np.vstack(
        [
            row_labels == label
            for label in range(n_row_clusters)
            for _ in range(n_col_clusters)
        ]
    )
    # 构建列选择矩阵，每列代表一个列簇的成员，对应标签为 True
    cols = np.vstack(
        [
            col_labels == label
            for _ in range(n_row_clusters)
            for label in range(n_col_clusters)
        ]
    )

    # 返回结果数组、行选择矩阵和列选择矩阵
    return result, rows, cols
```
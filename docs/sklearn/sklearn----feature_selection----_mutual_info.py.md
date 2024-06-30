# `D:\src\scipysrc\scikit-learn\sklearn\feature_selection\_mutual_info.py`

```
    # 计算连续变量和离散变量之间的互信息

    Parameters
    ----------
    c : ndarray, shape (n_samples,)
        连续随机变量的样本。

    d : ndarray, shape (n_samples,)
        离散随机变量的样本。

    n_neighbors : int
        每个点要搜索的最近邻数，参见 [1]_。

    Returns
    -------
    mi : float
        估计的互信息，单位为 nat。如果为负，则替换为 0。

    Notes
    -----
    真正的互信息不能为负。如果通过数值方法估计的互信息为负数，
    这意味着（如果方法合适）互信息接近于 0，并将其替换为 0 是一个合理的策略。

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    """
    # 样本数量
    n_samples = c.size

    # 将 c 和 d 重塑为列向量
    c = c.reshape((-1, 1))
    d = d.reshape((-1, 1))

    # 合并 c 和 d 成为一个特征矩阵 xy
    xy = np.hstack((c, d))

    # 使用 NearestNeighbors 选择最快的算法，这里使用切比雪夫距离作为度量方式
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=n_neighbors)

    # 适配数据
    nn.fit(xy)

    # 计算最近邻距离
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    # 明确使用 KDTree，允许查询在指定半径内的邻居数量
    kd = KDTree(c, metric="chebyshev")
    nx = kd.query_radius(c, radius, count_only=True, return_distance=False)
    nx = np.array(nx) - 1.0

    kd = KDTree(d, metric="chebyshev")
    ny = kd.query_radius(d, radius, count_only=True, return_distance=False)
    ny = np.array(ny) - 1.0

    # 计算互信息
    mi = (
        digamma(n_samples)
        + digamma(n_neighbors)
        - np.mean(digamma(nx + 1))
        - np.mean(digamma(ny + 1))
    )

    # 返回互信息，确保不会返回负数
    return max(0, mi)
    mi : float
        Estimated mutual information in nat units. If it turned out to be
        negative it is replaced by 0.

    Notes
    -----
    True mutual information can't be negative. If its estimate by a numerical
    method is negative, it means (providing the method is adequate) that the
    mutual information is close to 0 and replacing it by 0 is a reasonable
    strategy.

    References
    ----------
    .. [1] B. C. Ross "Mutual Information between Discrete and Continuous
       Data Sets". PLoS ONE 9(2), 2014.
    """
    # 获取样本数量
    n_samples = c.shape[0]
    # 将 c 变形为列向量
    c = c.reshape((-1, 1))

    # 初始化 radius、label_counts、k_all 为长度为样本数的空数组
    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)

    # 创建最近邻对象
    nn = NearestNeighbors()

    # 遍历 d 的唯一值
    for label in np.unique(d):
        # 生成当前标签的布尔掩码
        mask = d == label
        # 计算当前标签的样本数
        count = np.sum(mask)

        # 如果当前标签的样本数大于1
        if count > 1:
            # 取 n_neighbors 和 (count - 1) 中的较小值作为 k
            k = min(n_neighbors, count - 1)
            # 设置最近邻的参数
            nn.set_params(n_neighbors=k)
            # 使用当前标签的样本数据拟合最近邻
            nn.fit(c[mask])
            # 计算最近邻的距离
            r = nn.kneighbors()[0]
            # 将最近邻距离中的最后一个元素的下一个浮点数作为半径
            radius[mask] = np.nextafter(r[:, -1], 0)
            # 记录当前标签的 k 值
            k_all[mask] = k

        # 记录当前标签的样本数
        label_counts[mask] = count

    # 忽略具有唯一标签的点
    mask = label_counts > 1
    # 更新样本数量为满足条件的样本数
    n_samples = np.sum(mask)
    # 更新 label_counts、k_all、c、radius 为满足条件的数组
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    c = c[mask]
    radius = radius[mask]

    # 创建 KD 树对象
    kd = KDTree(c)
    # 查询在给定半径内的邻居数量，返回结果是仅计数的数组
    m_all = kd.query_radius(c, radius, count_only=True, return_distance=False)
    m_all = np.array(m_all)

    # 计算估计的互信息 mi
    mi = (
        digamma(n_samples)
        + np.mean(digamma(k_all))
        - np.mean(digamma(label_counts))
        - np.mean(digamma(m_all))
    )

    # 返回 mi 和 0 中的较大值
    return max(0, mi)
# 计算两个变量之间的互信息

def _compute_mi(x, y, x_discrete, y_discrete, n_neighbors=3):
    """Compute mutual information between two variables.

    This is a simple wrapper which selects a proper function to call based on
    whether `x` and `y` are discrete or not.
    """
    # 如果 x 和 y 都是离散变量，则直接调用 mutual_info_score 函数计算互信息
    if x_discrete and y_discrete:
        return mutual_info_score(x, y)
    # 如果 x 是离散变量而 y 是连续变量，则调用 _compute_mi_cd 函数计算互信息
    elif x_discrete and not y_discrete:
        return _compute_mi_cd(y, x, n_neighbors)
    # 如果 x 是连续变量而 y 是离散变量，则调用 _compute_mi_cd 函数计算互信息
    elif not x_discrete and y_discrete:
        return _compute_mi_cd(x, y, n_neighbors)
    # 如果 x 和 y 都是连续变量，则调用 _compute_mi_cc 函数计算互信息
    else:
        return _compute_mi_cc(x, y, n_neighbors)


# 迭代矩阵的列

def _iterate_columns(X, columns=None):
    """Iterate over columns of a matrix.

    Parameters
    ----------
    X : ndarray or csc_matrix, shape (n_samples, n_features)
        Matrix over which to iterate.

    columns : iterable or None, default=None
        Indices of columns to iterate over. If None, iterate over all columns.

    Yields
    ------
    x : ndarray, shape (n_samples,)
        Columns of `X` in dense format.
    """
    # 如果 columns 参数为 None，则迭代所有列的索引
    if columns is None:
        columns = range(X.shape[1])

    # 如果 X 是稀疏矩阵，则使用稀疏矩阵的方式迭代
    if issparse(X):
        for i in columns:
            # 初始化一个全零数组 x，长度为 X 的行数
            x = np.zeros(X.shape[0])
            # 获取第 i 列的起始和结束指针
            start_ptr, end_ptr = X.indptr[i], X.indptr[i + 1]
            # 将数据填充到 x 中对应的位置
            x[X.indices[start_ptr:end_ptr]] = X.data[start_ptr:end_ptr]
            yield x
    # 如果 X 是密集矩阵，则直接迭代列
    else:
        for i in columns:
            yield X[:, i]


# 估计特征与目标之间的互信息

def _estimate_mi(
    X,
    y,
    *,
    discrete_features="auto",
    discrete_target=False,
    n_neighbors=3,
    copy=True,
    random_state=None,
    n_jobs=None,
):
    """Estimate mutual information between the features and the target.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples,)
        Target vector.

    discrete_features : {'auto', bool, array-like}, default='auto'
        If bool, then determines whether to consider all features discrete
        or continuous. If array, then it should be either a boolean mask
        with shape (n_features,) or array with indices of discrete features.
        If 'auto', it is assigned to False for dense `X` and to True for
        sparse `X`.

    discrete_target : bool, default=False
        Whether to consider `y` as a discrete variable.

    n_neighbors : int, default=3
        Number of neighbors to use for MI estimation for continuous variables,
        see [1]_ and [2]_. Higher values reduce variance of the estimation, but
        could introduce a bias.

    copy : bool, default=True
        Whether to make a copy of the given data. If set to False, the initial
        data will be overwritten.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for adding small noise to
        continuous variables in order to remove repeated values.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    """
    n_jobs : int, default=None
        The number of jobs to use for computing the mutual information.
        The parallelization is done on the columns of `X`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        .. versionadded:: 1.5
        参数 n_jobs：int，默认为 None
        用于计算特征与目标之间互信息的作业数量。
        并行化处理在 `X` 的列上进行。
        ``None`` 意味着默认为 1，除非在 :obj:`joblib.parallel_backend` 上下文中。
        ``-1`` 表示使用所有处理器。详见 :term:`术语表 <n_jobs>`。
        .. versionadded:: 1.5

    Returns
    -------
    mi : ndarray, shape (n_features,)
        Estimated mutual information between each feature and the target in
        nat units. A negative value will be replaced by 0.
        返回值
        mi：ndarray，形状为 (n_features,)
        每个特征与目标之间估计的互信息，以 nats 为单位。负值将被替换为 0。

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    .. [2] B. C. Ross "Mutual Information between Discrete and Continuous
           Data Sets". PLoS ONE 9(2), 2014.
    参考文献
    ----------
    .. [1] A. Kraskov, H. Stogbauer 和 P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    .. [2] B. C. Ross "Mutual Information between Discrete and Continuous
           Data Sets". PLoS ONE 9(2), 2014.
    """
    X, y = check_X_y(X, y, accept_sparse="csc", y_numeric=not discrete_target)
    # 检查输入的 X 和 y，确保可以处理稀疏矩阵，且 y 是数值型而非离散的目标值

    n_samples, n_features = X.shape
    # 获取样本数和特征数

    if isinstance(discrete_features, (str, bool)):
        if isinstance(discrete_features, str):
            if discrete_features == "auto":
                discrete_features = issparse(X)
            else:
                raise ValueError("Invalid string value for discrete_features.")
        discrete_mask = np.empty(n_features, dtype=bool)
        discrete_mask.fill(discrete_features)
    else:
        discrete_features = check_array(discrete_features, ensure_2d=False)
        if discrete_features.dtype != "bool":
            discrete_mask = np.zeros(n_features, dtype=bool)
            discrete_mask[discrete_features] = True
        else:
            discrete_mask = discrete_features
    # 处理离散特征的掩码，将离散特征表示为布尔掩码数组

    continuous_mask = ~discrete_mask
    # 计算连续特征的掩码，即非离散特征的布尔掩码数组

    if np.any(continuous_mask) and issparse(X):
        raise ValueError("Sparse matrix `X` can't have continuous features.")
    # 如果 X 是稀疏矩阵且包含连续特征，则抛出值错误异常

    rng = check_random_state(random_state)
    # 检查并返回一个随机数生成器实例，用于后续随机化操作

    if np.any(continuous_mask):
        X = X.astype(np.float64, copy=copy)
        # 将 X 转换为双精度浮点数类型

        X[:, continuous_mask] = scale(
            X[:, continuous_mask], with_mean=False, copy=False
        )
        # 对连续特征进行标准化处理，去除均值，保持原地修改

        # Add small noise to continuous features as advised in Kraskov et. al.
        means = np.maximum(1, np.mean(np.abs(X[:, continuous_mask]), axis=0))
        X[:, continuous_mask] += (
            1e-10
            * means
            * rng.standard_normal(size=(n_samples, np.sum(continuous_mask)))
        )
        # 根据 Kraskov 等人的建议，向连续特征添加小的噪声

    if not discrete_target:
        y = scale(y, with_mean=False)
        # 对目标变量 y 进行标准化处理，去除均值

        y += (
            1e-10
            * np.maximum(1, np.mean(np.abs(y)))
            * rng.standard_normal(size=n_samples)
        )
        # 向目标变量 y 添加小的噪声

    mi = Parallel(n_jobs=n_jobs)(
        delayed(_compute_mi)(x, y, discrete_feature, discrete_target, n_neighbors)
        for x, discrete_feature in zip(_iterate_columns(X), discrete_mask)
    )
    # 使用并行计算方式计算每个特征与目标之间的互信息，并返回结果列表 mi

    return np.array(mi)
    # 将计算得到的互信息结果转换为 ndarray 类型并返回
# 使用装饰器验证函数参数，确保输入的参数类型和取值符合规范
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # X参数应为数组或稀疏矩阵
        "y": ["array-like"],  # y参数应为数组
        "discrete_features": [StrOptions({"auto"}), "boolean", "array-like"],  # discrete_features参数可以为'auto'、布尔值或数组
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],  # n_neighbors参数为大于等于1的整数
        "copy": ["boolean"],  # copy参数应为布尔值
        "random_state": ["random_state"],  # random_state参数可以为随机状态对象
        "n_jobs": [Integral, None],  # n_jobs参数可以为整数或None
    },
    prefer_skip_nested_validation=True,  # 优先跳过嵌套验证
)
def mutual_info_regression(
    X,
    y,
    *,
    discrete_features="auto",  # discrete_features默认为'auto'
    n_neighbors=3,  # n_neighbors默认为3
    copy=True,  # copy默认为True
    random_state=None,  # random_state默认为None
    n_jobs=None,  # n_jobs默认为None
):
    """Estimate mutual information for a continuous target variable.

    Mutual information (MI) [1]_ between two random variables is a non-negative
    value, which measures the dependency between the variables. It is equal
    to zero if and only if two random variables are independent, and higher
    values mean higher dependency.

    The function relies on nonparametric methods based on entropy estimation
    from k-nearest neighbors distances as described in [2]_ and [3]_. Both
    methods are based on the idea originally proposed in [4]_.

    It can be used for univariate features selection, read more in the
    :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples,)
        Target vector.

    discrete_features : {'auto', bool, array-like}, default='auto'
        If bool, then determines whether to consider all features discrete
        or continuous. If array, then it should be either a boolean mask
        with shape (n_features,) or array with indices of discrete features.
        If 'auto', it is assigned to False for dense `X` and to True for
        sparse `X`.

    n_neighbors : int, default=3
        Number of neighbors to use for MI estimation for continuous variables,
        see [2]_ and [3]_. Higher values reduce variance of the estimation, but
        could introduce a bias.

    copy : bool, default=True
        Whether to make a copy of the given data. If set to False, the initial
        data will be overwritten.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for adding small noise to
        continuous variables in order to remove repeated values.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_jobs : int, default=None
        The number of jobs to use for computing the mutual information.
        The parallelization is done on the columns of `X`.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 1.5

    Returns
    -------
    # mi : ndarray, shape (n_features,)
    # 每个特征与目标变量之间的估计互信息，单位为 nats（自然单位）的数组。

    # Notes
    # -----
    # 1. “离散特征”这一术语用于替代“分类特征”，因为它更准确地描述了其本质。
    #    例如，图像的像素强度是离散特征（但几乎不是分类特征），如果将其标记为离散特征，将会获得更好的结果。
    #    此外，请注意，将连续变量视为离散变量或反之通常会导致不正确的结果，因此请注意区分它们。
    # 2. 真正的互信息不可能为负值。如果其估计结果为负数，则将其替换为零。

    # References
    # ----------
    # .. [1] `Mutual Information
    #        <https://en.wikipedia.org/wiki/Mutual_information>`_
    #        在维基百科上的互信息条目。
    # .. [2] A. Kraskov, H. Stogbauer 和 P. Grassberger, "Estimating mutual
    #        information". Phys. Rev. E 69, 2004.
    # .. [3] B. C. Ross "Mutual Information between Discrete and Continuous
    #        Data Sets". PLoS ONE 9(2), 2014.
    # .. [4] L. F. Kozachenko, N. N. Leonenko, "Sample Estimate of the Entropy
    #        of a Random Vector", Probl. Peredachi Inf., 23:2 (1987), 9-16

    # Examples
    # --------
    # >>> from sklearn.datasets import make_regression
    # >>> from sklearn.feature_selection import mutual_info_regression
    # >>> X, y = make_regression(
    # ...     n_samples=50, n_features=3, n_informative=1, noise=1e-4, random_state=42
    # ... )
    # >>> mutual_info_regression(X, y)
    # array([0.1..., 2.6...  , 0.0...])

    # 调用 _estimate_mi 函数来计算互信息
    return _estimate_mi(
        X,
        y,
        discrete_features=discrete_features,
        discrete_target=False,
        n_neighbors=n_neighbors,
        copy=copy,
        random_state=random_state,
        n_jobs=n_jobs,
    )
# 使用装饰器验证函数参数的有效性，确保参数满足指定的数据类型和取值范围
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # X参数应为类似数组或稀疏矩阵
        "y": ["array-like"],  # y参数应为类似数组
        "discrete_features": [StrOptions({"auto"}), "boolean", "array-like"],  # discrete_features参数可选值为'auto'、布尔值或类似数组
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],  # n_neighbors参数为大于等于1的整数
        "copy": ["boolean"],  # copy参数应为布尔值
        "random_state": ["random_state"],  # random_state参数为随机数生成器或None
        "n_jobs": [Integral, None],  # n_jobs参数为整数或None
    },
    prefer_skip_nested_validation=True,  # 首选跳过嵌套验证
)
def mutual_info_classif(
    X,
    y,
    *,
    discrete_features="auto",  # 默认值为'auto'，用于自动确定离散特征
    n_neighbors=3,  # 默认使用3个最近邻居进行互信息估计
    copy=True,  # 默认复制数据
    random_state=None,  # 默认不使用随机状态
    n_jobs=None,  # 默认不使用多个作业
):
    """Estimate mutual information for a discrete target variable.

    Mutual information (MI) [1]_ between two random variables is a non-negative
    value, which measures the dependency between the variables. It is equal
    to zero if and only if two random variables are independent, and higher
    values mean higher dependency.

    The function relies on nonparametric methods based on entropy estimation
    from k-nearest neighbors distances as described in [2]_ and [3]_. Both
    methods are based on the idea originally proposed in [4]_.

    It can be used for univariate features selection, read more in the
    :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples,)
        Target vector.

    discrete_features : 'auto', bool or array-like, default='auto'
        If bool, then determines whether to consider all features discrete
        or continuous. If array, then it should be either a boolean mask
        with shape (n_features,) or array with indices of discrete features.
        If 'auto', it is assigned to False for dense `X` and to True for
        sparse `X`.

    n_neighbors : int, default=3
        Number of neighbors to use for MI estimation for continuous variables,
        see [2]_ and [3]_. Higher values reduce variance of the estimation, but
        could introduce a bias.

    copy : bool, default=True
        Whether to make a copy of the given data. If set to False, the initial
        data will be overwritten.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for adding small noise to
        continuous variables in order to remove repeated values.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_jobs : int, default=None
        The number of jobs to use for computing the mutual information.
        The parallelization is done on the columns of `X`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 1.5

    Returns
    -------
    # mi : ndarray, shape (n_features,)
    # 每个特征与目标之间的估计互信息，单位为 nats（自然单位）。

    Notes
    -----
    # “离散特征”这个术语比“分类特征”更准确地描述了它们的本质。
    # 例如，图像的像素强度是离散特征（但几乎不是分类特征），如果将它们标记为离散特征，会获得更好的结果。
    # 此外，请注意，将连续变量视为离散变量或反之通常会导致不正确的结果，因此在这方面要注意。

    # 互信息的真实值不会是负数。如果估计值为负数，则将其替换为零。

    References
    ----------
    .. [1] `Mutual Information
           <https://en.wikipedia.org/wiki/Mutual_information>`_
           维基百科上关于互信息的解释。
    .. [2] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
           关于估计互信息的物理评论文章。
    .. [3] B. C. Ross "Mutual Information between Discrete and Continuous
           Data Sets". PLoS ONE 9(2), 2014.
           关于离散和连续数据集之间互信息的研究。
    .. [4] L. F. Kozachenko, N. N. Leonenko, "Sample Estimate of the Entropy
           of a Random Vector:, Probl. Peredachi Inf., 23:2 (1987), 9-16
           随机向量熵的样本估计。

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.feature_selection import mutual_info_classif
    >>> X, y = make_classification(
    ...     n_samples=100, n_features=10, n_informative=2, n_clusters_per_class=1,
    ...     shuffle=False, random_state=42
    ... )
    >>> mutual_info_classif(X, y)
    array([0.58..., 0.10..., 0.19..., 0.09... , 0.        ,
           0.     , 0.     , 0.     , 0.      , 0.        ])
    """
    # 检查分类目标的有效性
    check_classification_targets(y)
    # 调用 _estimate_mi 函数来估计互信息，传入相应的参数
    return _estimate_mi(
        X,
        y,
        discrete_features=discrete_features,
        discrete_target=True,
        n_neighbors=n_neighbors,
        copy=copy,
        random_state=random_state,
        n_jobs=n_jobs,
    )
```
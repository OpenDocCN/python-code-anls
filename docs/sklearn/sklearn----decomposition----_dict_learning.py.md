# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\_dict_learning.py`

```
    """Dictionary learning."""

    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause

    import itertools  # 导入 itertools 模块，用于生成迭代器
    import sys  # 导入 sys 模块，用于与 Python 解释器交互
    import time  # 导入 time 模块，用于时间相关功能
    from numbers import Integral, Real  # 从 numbers 模块导入 Integral 和 Real 类，用于数值类型检查
    from warnings import warn  # 从 warnings 模块导入 warn 函数，用于发出警告

    import numpy as np  # 导入 NumPy 库，并使用别名 np
    from joblib import effective_n_jobs  # 从 joblib 库导入 effective_n_jobs 函数，用于获取有效的并行作业数
    from scipy import linalg  # 从 SciPy 库导入 linalg 模块，用于线性代数操作

    from ..base import (  # 从当前包的 base 模块导入多个类和函数
        BaseEstimator,  # 导入 BaseEstimator 类，用于基础估计器的基类
        ClassNamePrefixFeaturesOutMixin,  # 导入 ClassNamePrefixFeaturesOutMixin 类，用于特征输出前缀的混合类
        TransformerMixin,  # 导入 TransformerMixin 类，用于变换器的混合类
        _fit_context,  # 导入 _fit_context 函数，用于内部使用的上下文管理器
    )
    from ..linear_model import Lars, Lasso, LassoLars, orthogonal_mp_gram  # 从当前包的 linear_model 模块导入多个类和函数
    from ..utils import (  # 从当前包的 utils 模块导入多个函数和类
        check_array,  # 导入 check_array 函数，用于验证数据数组
        check_random_state,  # 导入 check_random_state 函数，用于验证随机状态
        gen_batches,  # 导入 gen_batches 函数，用于生成批次数据的生成器
        gen_even_slices,  # 导入 gen_even_slices 函数，用于生成均匀切片的生成器
    )
    from ..utils._param_validation import (  # 从当前包的 utils._param_validation 模块导入多个类和函数
        Hidden,  # 导入 Hidden 类，用于隐藏的参数验证类
        Interval,  # 导入 Interval 类，用于区间验证类
        StrOptions,  # 导入 StrOptions 类，用于字符串选项验证类
        validate_params,  # 导入 validate_params 函数，用于验证参数
    )
    from ..utils.extmath import (  # 从当前包的 utils.extmath 模块导入多个函数
        randomized_svd,  # 导入 randomized_svd 函数，用于随机奇异值分解
        row_norms,  # 导入 row_norms 函数，用于计算行的范数
        svd_flip,  # 导入 svd_flip 函数，用于奇异值分解中的翻转
    )
    from ..utils.parallel import Parallel, delayed  # 从当前包的 utils.parallel 模块导入 Parallel 和 delayed 类

    from ..utils.validation import check_is_fitted  # 从当前包的 utils.validation 模块导入 check_is_fitted 函数，用于检查是否拟合

    def _check_positive_coding(method, positive):
        """检查正编码约束是否支持给定的编码方法。"""
        if positive and method in ["omp", "lars"]:
            raise ValueError(
                "Positive constraint not supported for '{}' coding method.".format(method)
            )

    def _sparse_encode_precomputed(
        X,
        dictionary,
        *,
        gram=None,
        cov=None,
        algorithm="lasso_lars",
        regularization=None,
        copy_cov=True,
        init=None,
        max_iter=1000,
        verbose=0,
        positive=False,
    ):
        """使用预先计算的 Gram 和/或协方差矩阵进行通用稀疏编码。

        结果的每一行是一个 Lasso 问题的解决方案。

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            数据矩阵。

        dictionary : ndarray of shape (n_components, n_features)
            用于解决数据稀疏编码的字典矩阵。某些算法假定行已标准化。

        gram : ndarray of shape (n_components, n_components), default=None
            预先计算的 Gram 矩阵，`dictionary * dictionary'`。
            如果方法为 'threshold'，gram 可以为 `None`。

        cov : ndarray of shape (n_components, n_samples), default=None
            预先计算的协方差，`dictionary * X'`。

        algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}, \
                default='lasso_lars'
            使用的算法：

            * `'lars'`：使用最小角回归方法 (`linear_model.lars_path`)；
            * `'lasso_lars'`：使用 Lars 计算 Lasso 解决方案；
            * `'lasso_cd'`：使用坐标下降法计算 Lasso 解决方案 (`linear_model.Lasso`)。如果
              估计的成分是稀疏的，lasso_lars 将更快；
            * `'omp'`：使用正交匹配追踪来估计稀疏解；
            * `'threshold'`：将投影 `dictionary * data'` 中小于 regularization 的所有系数压缩为零。

        regularization : int or float, default=None
            正则化参数。当算法为 `'lasso_lars'`、`'lasso_cd'` 或 `'threshold'` 时，
            对应于 alpha。否则对应于 `n_nonzero_coefs`。

        """
    # 获取输入数据矩阵 X 的样本数和特征数
    n_samples, n_features = X.shape
    # 获取字典矩阵的行数，即稀疏编码的组件数
    n_components = dictionary.shape[0]

    # 如果算法选择了 "lasso_lars"
    if algorithm == "lasso_lars":
        # 计算正则化参数 alpha，考虑特征数的缩放因子
        alpha = float(regularization) / n_features  # account for scaling
        try:
            # 暂时忽略 NumPy 的错误处理设置
            err_mgt = np.seterr(all="ignore")

            # 创建 LassoLars 对象进行稀疏编码
            # fit_intercept=False 表示不拟合截距
            # verbose 控制详细程度，由传入参数决定
            # precompute 设置预计算的 Gram 矩阵
            # fit_path=False 表示不输出路径
            # positive 根据参数设置是否强制稀疏编码为非负数
            # max_iter 设置最大迭代次数
            lasso_lars = LassoLars(
                alpha=alpha,
                fit_intercept=False,
                verbose=verbose,
                precompute=gram,
                fit_path=False,
                positive=positive,
                max_iter=max_iter,
            )
            # 使用字典的转置和 X 的转置进行拟合，Xy=cov 用于预先计算
            lasso_lars.fit(dictionary.T, X.T, Xy=cov)
            # 获取新的稀疏编码结果
            new_code = lasso_lars.coef_
        finally:
            # 恢复之前的 NumPy 错误处理设置
            np.seterr(**err_mgt)

    # 如果算法选择了 "lasso_cd"
    elif algorithm == "lasso_cd":
        # 计算正则化参数 alpha，考虑特征数的缩放因子
        alpha = float(regularization) / n_features  # account for scaling

        # 创建 Lasso 对象进行稀疏编码
        # fit_intercept=False 表示不拟合截距
        # precompute 设置预计算的 Gram 矩阵
        # max_iter 设置最大迭代次数
        # warm_start=True 表示可以重复使用上一次的解来加速收敛
        # positive 根据参数设置是否强制稀疏编码为非负数
        clf = Lasso(
            alpha=alpha,
            fit_intercept=False,
            precompute=gram,
            max_iter=max_iter,
            warm_start=True,
            positive=positive,
        )

        # 如果提供了初始化值 init
        if init is not None:
            # 在某些情况下，使用坐标下降算法：
            # - 用户可能提供具有只读缓冲区的 NumPy 数组
            # - `joblib` 可能 memmap 数组，使得它们的缓冲区是只读的
            # 如果 init 不可写，则复制为一个可写的 NumPy 数组
            if not init.flags["WRITEABLE"]:
                init = np.array(init)
            # 将初始化的稀疏编码设置到 clf 对象中
            clf.coef_ = init

        # 使用字典的转置和 X 的转置进行拟合，check_input=False 禁用输入检查
        clf.fit(dictionary.T, X.T, check_input=False)
        # 获取新的稀疏编码结果
        new_code = clf.coef_
    elif algorithm == "lars":
        try:
            # 临时忽略所有的 NumPy 错误
            err_mgt = np.seterr(all="ignore")

            # 创建一个 Lars 对象，用于稀疏回归
            # 不传入 verbose=max(0, verbose-1)，因为 Lars.fit 已经自行调整了详细级别
            lars = Lars(
                fit_intercept=False,
                verbose=verbose,
                precompute=gram,
                n_nonzero_coefs=int(regularization),
                fit_path=False,
            )
            # 使用 Lars.fit 方法进行拟合
            lars.fit(dictionary.T, X.T, Xy=cov)
            # 获取拟合后的系数
            new_code = lars.coef_
        finally:
            # 恢复之前的 NumPy 错误设置
            np.seterr(**err_mgt)

    elif algorithm == "threshold":
        # 应用阈值处理到协方差矩阵的每个元素
        new_code = (np.sign(cov) * np.maximum(np.abs(cov) - regularization, 0)).T
        if positive:
            # 如果设置为只保留非负值，则对 new_code 进行修剪操作
            np.clip(new_code, 0, None, out=new_code)

    elif algorithm == "omp":
        # 使用正交匹配追踪方法计算稀疏表示
        new_code = orthogonal_mp_gram(
            Gram=gram,
            Xy=cov,
            n_nonzero_coefs=int(regularization),
            tol=None,
            norms_squared=row_norms(X, squared=True),
            copy_Xy=copy_cov,
        ).T

    # 将 new_code 重新调整形状为 (n_samples, n_components) 的数组并返回
    return new_code.reshape(n_samples, n_components)
@validate_params(
    {
        "X": ["array-like"],  # 参数 X 应为类数组对象
        "dictionary": ["array-like"],  # 参数 dictionary 应为类数组对象
        "gram": ["array-like", None],  # 参数 gram 应为类数组对象或者 None
        "cov": ["array-like", None],  # 参数 cov 应为类数组对象或者 None
        "algorithm": [  # 参数 algorithm 应为以下选项之一
            StrOptions({"lasso_lars", "lasso_cd", "lars", "omp", "threshold"})
        ],
        "n_nonzero_coefs": [Interval(Integral, 1, None, closed="left"), None],  # 参数 n_nonzero_coefs 应为大于等于1的整数或者 None
        "alpha": [Interval(Real, 0, None, closed="left"), None],  # 参数 alpha 应为大于0的实数或者 None
        "copy_cov": ["boolean"],  # 参数 copy_cov 应为布尔值
        "init": ["array-like", None],  # 参数 init 应为类数组对象或者 None
        "max_iter": [Interval(Integral, 0, None, closed="left")],  # 参数 max_iter 应为大于等于0的整数
        "n_jobs": [Integral, None],  # 参数 n_jobs 应为整数或者 None
        "check_input": ["boolean"],  # 参数 check_input 应为布尔值
        "verbose": ["verbose"],  # 参数 verbose 应为详细程度描述
        "positive": ["boolean"],  # 参数 positive 应为布尔值
    },
    prefer_skip_nested_validation=True,  # 更倾向于跳过嵌套验证
)
# XXX : 可能可以移到 linear_model 模块中
def sparse_encode(
    X,
    dictionary,
    *,
    gram=None,
    cov=None,
    algorithm="lasso_lars",
    n_nonzero_coefs=None,
    alpha=None,
    copy_cov=True,
    init=None,
    max_iter=1000,
    n_jobs=None,
    check_input=True,
    verbose=0,
    positive=False,
):
    """稀疏编码。

    结果的每一行都是稀疏编码问题的解决方案。
    目标是找到一个稀疏数组 `code`，使得::

        X ~= code * dictionary

    更多信息请参阅 :ref:`用户指南 <SparseCoder>`。

    参数
    ----------
    X : 形状为 (n_samples, n_features) 的类数组对象
        数据矩阵。

    dictionary : 形状为 (n_components, n_features) 的类数组对象
        用于解决数据稀疏编码的字典矩阵。一些算法假定行已经归一化，以获得有意义的输出。

    gram : 形状为 (n_components, n_components) 的类数组对象，默认为 None
        预先计算的 Gram 矩阵，即 `dictionary * dictionary'`。

    cov : 形状为 (n_components, n_samples) 的类数组对象，默认为 None
        预先计算的协方差，即 `dictionary' * X`。

    algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}, 默认为 'lasso_lars'
        使用的算法：

        * `'lars'`: 使用最小角度回归方法 (`linear_model.lars_path`);
        * `'lasso_lars'`: 使用 Lars 计算 Lasso 解决方案;
        * `'lasso_cd'`: 使用坐标下降法计算 Lasso 解决方案 (`linear_model.Lasso`)。如果估计的组件稀疏，则 lasso_lars 将更快;
        * `'omp'`: 使用正交匹配追踪来估计稀疏解决方案;
        * `'threshold'`: 将小于投影 `dictionary * data'` 的正则化的所有系数压缩为零。

    n_nonzero_coefs : int, 默认为 None
        每列解决方案中的非零系数数目。仅由 `algorithm='lars'` 和 `algorithm='omp'` 使用，并在 `omp` 情况下被 `alpha` 覆盖。如果为 `None`，则 `n_nonzero_coefs=int(n_features / 10)`。

    alpha : float, 默认为 None
        正则化参数。仅由 `algorithm='lasso_lars'` 和 `algorithm='lasso_cd'` 使用。

    copy_cov : bool, 默认为 True
        是否复制协方差。

    init : 形状为 (n_samples, n_features) 的类数组对象，默认为 None
        用于初始化的稀疏代码。仅在 `algorithm='threshold'` 时使用。

    max_iter : int, 默认为 1000
        最大迭代次数。

    n_jobs : int, 默认为 None
        同时运行的作业数量。如果为 None，则作业数量取决于系统。

    check_input : bool, 默认为 True
        是否检查输入数据的有效性。

    verbose : int, 默认为 0
        控制详细程度的级别。

    positive : bool, 默认为 False
        是否强制系数为正。

    """
    pass
    """
    alpha : float, default=None
        如果 `algorithm='lasso_lars'` 或 `algorithm='lasso_cd'`，则 `alpha` 是应用于 L1 范数的惩罚项。
        如果 `algorithm='threshold'`，则 `alpha` 是阈值的绝对值，低于该阈值的系数将被压缩到零。
        如果 `algorithm='omp'`，则 `alpha` 是容差参数：目标重构误差的值。在这种情况下，它会覆盖 `n_nonzero_coefs`。
        如果为 `None`，则默认为 1.

    copy_cov : bool, default=True
        是否复制预先计算的协方差矩阵；如果为 `False`，可能会被覆盖。

    init : ndarray of shape (n_samples, n_components), default=None
        稀疏编码的初始化值。仅在 `algorithm='lasso_cd'` 时使用。

    max_iter : int, default=1000
        如果 `algorithm='lasso_cd'` 或 `'lasso_lars'`，则执行的最大迭代次数。

    n_jobs : int, default=None
        要运行的并行作业数。
        ``None`` 表示除非在 :obj:`joblib.parallel_backend` 上下文中，否则为 1。
        ``-1`` 表示使用所有处理器。详细信息请参见 :term:`术语表 <n_jobs>`。

    check_input : bool, default=True
        如果为 `False`，则不检查输入数组 X 和字典。

    verbose : int, default=0
        控制详细程度；值越高，输出的消息越多。

    positive : bool, default=False
        在找到编码时是否强制非负。

        .. versionadded:: 0.20

    Returns
    -------
    code : ndarray of shape (n_samples, n_components)
        稀疏编码。

    See Also
    --------
    sklearn.linear_model.lars_path : 使用 LARS 算法计算最小角回归或 Lasso 路径。
    sklearn.linear_model.orthogonal_mp : 解决正交匹配追踪问题。
    sklearn.linear_model.Lasso : 使用 L1 先验训练线性模型的正则化器。
    SparseCoder : 从预先计算的固定字典中找到数据的稀疏表示。

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.decomposition import sparse_encode
    >>> X = np.array([[-1, -1, -1], [0, 0, 3]])
    >>> dictionary = np.array(
    ...     [[0, 1, 0],
    ...      [-1, -1, 2],
    ...      [1, 1, 1],
    ...      [0, 1, 1],
    ...      [0, 2, 1]],
    ...    dtype=np.float64
    ... )
    >>> sparse_encode(X, dictionary, alpha=1e-10)
    array([[ 0.,  0., -1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.]])
    """
    if check_input:
        if algorithm == "lasso_cd":
            dictionary = check_array(
                dictionary, order="C", dtype=[np.float64, np.float32]
            )
            X = check_array(X, order="C", dtype=[np.float64, np.float32])
        else:
            dictionary = check_array(dictionary)
            X = check_array(X)
    # 检查字典和数据矩阵 X 的列数是否相同，若不同则抛出值错误异常
    if dictionary.shape[1] != X.shape[1]:
        raise ValueError(
            "Dictionary and X have different numbers of features:"
            "dictionary.shape: {} X.shape{}".format(dictionary.shape, X.shape)
        )

    # 检查算法参数和正性参数是否合法，若不合法则抛出相应异常
    _check_positive_coding(algorithm, positive)

    # 调用稀疏编码函数 _sparse_encode 进行稀疏编码操作
    return _sparse_encode(
        X,
        dictionary,
        gram=gram,
        cov=cov,
        algorithm=algorithm,
        n_nonzero_coefs=n_nonzero_coefs,
        alpha=alpha,
        copy_cov=copy_cov,
        init=init,
        max_iter=max_iter,
        n_jobs=n_jobs,
        verbose=verbose,
        positive=positive,
    )
def _sparse_encode(
    X,
    dictionary,
    *,
    gram=None,
    cov=None,
    algorithm="lasso_lars",
    n_nonzero_coefs=None,
    alpha=None,
    copy_cov=True,
    init=None,
    max_iter=1000,
    n_jobs=None,
    verbose=0,
    positive=False,
):
    """Sparse coding without input/parameter validation."""

    # 获取样本数和特征数
    n_samples, n_features = X.shape
    # 获取字典中的组件数
    n_components = dictionary.shape[0]

    # 根据选择的算法确定正则化参数
    if algorithm in ("lars", "omp"):
        regularization = n_nonzero_coefs
        if regularization is None:
            regularization = min(max(n_features / 10, 1), n_components)
    else:
        regularization = alpha
        if regularization is None:
            regularization = 1.0

    # 如果未提供 Gram 矩阵且算法不是 "threshold"，则计算字典的 Gram 矩阵
    if gram is None and algorithm != "threshold":
        gram = np.dot(dictionary, dictionary.T)

    # 如果未提供协方差矩阵且算法不是 "lasso_cd"，则计算字典与输入数据 X 的协方差
    if cov is None and algorithm != "lasso_cd":
        copy_cov = False
        cov = np.dot(dictionary, X.T)

    # 如果只能使用单线程或算法是 "threshold"，则调用 _sparse_encode_precomputed 进行编码
    if effective_n_jobs(n_jobs) == 1 or algorithm == "threshold":
        code = _sparse_encode_precomputed(
            X,
            dictionary,
            gram=gram,
            cov=cov,
            algorithm=algorithm,
            regularization=regularization,
            copy_cov=copy_cov,
            init=init,
            max_iter=max_iter,
            verbose=verbose,
            positive=positive,
        )
        return code

    # 进入并行代码块
    n_samples = X.shape[0]
    n_components = dictionary.shape[0]
    # 创建一个空的编码矩阵
    code = np.empty((n_samples, n_components))
    # 生成均匀分割的切片
    slices = list(gen_even_slices(n_samples, effective_n_jobs(n_jobs)))

    # 并行计算各切片的编码视图
    code_views = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_sparse_encode_precomputed)(
            X[this_slice],
            dictionary,
            gram=gram,
            cov=cov[:, this_slice] if cov is not None else None,
            algorithm=algorithm,
            regularization=regularization,
            copy_cov=copy_cov,
            init=init[this_slice] if init is not None else None,
            max_iter=max_iter,
            verbose=verbose,
            positive=positive,
        )
        for this_slice in slices
    )
    # 将各切片的编码视图合并到整体编码矩阵中
    for this_slice, this_view in zip(slices, code_views):
        code[this_slice] = this_view
    return code


def _update_dict(
    dictionary,
    Y,
    code,
    A=None,
    B=None,
    verbose=False,
    random_state=None,
    positive=False,
):
    """Update the dense dictionary factor in place.

    Parameters
    ----------
    dictionary : ndarray of shape (n_components, n_features)
        Value of the dictionary at the previous iteration.

    Y : ndarray of shape (n_samples, n_features)
        Data matrix.

    code : ndarray of shape (n_samples, n_components)
        Sparse coding of the data against which to optimize the dictionary.

    A : ndarray of shape (n_components, n_components), default=None
        Together with `B`, sufficient stats of the online model to update the
        dictionary.
    """
    # 稀疏编码的更新字典因子的实现
    # 更新字典的过程不包含输入或参数验证
    B : ndarray of shape (n_features, n_components), default=None
        # B 是一个 ndarray 数组，形状为 (n_features, n_components)，用于更新字典的在线模型的足够统计量。

    verbose: bool, default=False
        # verbose 控制是否打印过程中的详细输出。

    random_state : int, RandomState instance or None, default=None
        # 用于随机初始化字典的随机数种子。传递一个整数以确保多次函数调用时结果可复现。

    positive : bool, default=False
        # 是否在找到字典时强制要求其为正数。

        .. versionadded:: 0.20
        # 该选项在版本 0.20 中添加。

    """
    # 从输入的 code 的形状中提取样本数和组件数
    n_samples, n_components = code.shape

    # 检查并确保 random_state 是一个 RandomState 实例
    random_state = check_random_state(random_state)

    # 如果 A 为 None，则计算 A
    if A is None:
        A = code.T @ code  # 计算 code 的转置乘以 code

    # 如果 B 为 None，则计算 B
    if B is None:
        B = Y.T @ code  # 计算 Y 的转置乘以 code

    # 未使用的组件计数
    n_unused = 0

    # 对于每个组件 k
    for k in range(n_components):
        # 如果 A[k, k] 大于 1e-6
        if A[k, k] > 1e-6:
            # 使用在线模型的统计量更新字典的第 k 个原子
            dictionary[k] += (B[:, k] - A[k] @ dictionary) / A[k, k]
        else:
            # 第 k 个原子几乎不被使用 -> 从数据中随机选择一个新的原子
            newd = Y[random_state.choice(n_samples)]

            # 添加小的噪声以避免稀疏编码不良条件
            noise_level = 0.01 * (newd.std() or 1)  # 避免标准差为 0
            noise = random_state.normal(0, noise_level, size=len(newd))

            # 设置第 k 个原子为新的原子加上噪声
            dictionary[k] = newd + noise

            # 将 code 的第 k 列置为零
            code[:, k] = 0

            # 未使用的计数加一
            n_unused += 1

        # 如果 positive 为 True，则将第 k 个原子限制为非负数
        if positive:
            np.clip(dictionary[k], 0, None, out=dictionary[k])

        # 投影到约束集合 ||V_k|| <= 1
        dictionary[k] /= max(linalg.norm(dictionary[k]), 1)

    # 如果 verbose 为 True 并且有未使用的原子，则打印相应信息
    if verbose and n_unused > 0:
        print(f"{n_unused} unused atoms resampled.")
# 主要的字典学习算法
def _dict_learning(
    X,
    n_components,
    *,
    alpha,
    max_iter,
    tol,
    method,
    n_jobs,
    dict_init,
    code_init,
    callback,
    verbose,
    random_state,
    return_n_iter,
    positive_dict,
    positive_code,
    method_max_iter,
):
    """Main dictionary learning algorithm"""
    t0 = time.time()  # 记录开始时间

    # 初始化编码和字典为 Y 的奇异值分解(SVD)结果
    if code_init is not None and dict_init is not None:
        code = np.array(code_init, order="F")  # 使用给定的编码初始化 code 数组
        # 不复制 V，后续步骤将处理复制
        dictionary = dict_init  # 使用给定的字典初始化 dictionary 变量
    else:
        code, S, dictionary = linalg.svd(X, full_matrices=False)  # 对 X 进行奇异值分解
        # 翻转初始编码的符号以确保确定性输出
        code, dictionary = svd_flip(code, dictionary)
        dictionary = S[:, np.newaxis] * dictionary  # 乘以奇异值构建字典

    r = len(dictionary)  # 字典的长度
    if n_components <= r:  # 如果要求的成分数小于等于当前字典的长度
        code = code[:, :n_components]  # 截取编码数组到指定的成分数
        dictionary = dictionary[:n_components, :]  # 截取字典数组到指定的成分数
    else:
        code = np.c_[code, np.zeros((len(code), n_components - r))]  # 扩展编码数组
        dictionary = np.r_[
            dictionary, np.zeros((n_components - r, dictionary.shape[1]))
        ]  # 扩展字典数组

    # 将字典按 Fortran 排序，以便更适合于该算法中的稀疏编码
    dictionary = np.asfortranarray(dictionary)

    errors = []  # 存储误差列表
    current_cost = np.nan  # 当前成本设置为 NaN

    if verbose == 1:
        print("[dict_learning]", end=" ")  # 如果 verbose 为 1，打印信息 "[dict_learning]"

    # 如果 max_iter 为 0，则迭代次数应为零
    ii = -1  # 迭代次数初始化为 -1
    # 迭代指定次数或直到收敛
    for ii in range(max_iter):
        # 计算从开始到当前时刻的经过时间
        dt = time.time() - t0
        # 如果 verbose 等于 1，打印一个点并刷新输出
        if verbose == 1:
            sys.stdout.write(".")
            sys.stdout.flush()
        # 如果 verbose 为真，打印迭代信息，包括迭代次数、已经经过的时间（秒）、分钟数以及当前的成本
        elif verbose:
            print(
                "Iteration % 3i (elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)"
                % (ii, dt, dt / 60, current_cost)
            )

        # 更新编码
        code = sparse_encode(
            X,
            dictionary,
            algorithm=method,
            alpha=alpha,
            init=code,
            n_jobs=n_jobs,
            positive=positive_code,
            max_iter=method_max_iter,
            verbose=verbose,
        )

        # 原地更新字典
        _update_dict(
            dictionary,
            X,
            code,
            verbose=verbose,
            random_state=random_state,
            positive=positive_dict,
        )

        # 成本函数计算
        current_cost = 0.5 * np.sum((X - code @ dictionary) ** 2) + alpha * np.sum(
            np.abs(code)
        )
        # 将当前成本添加到错误列表中
        errors.append(current_cost)

        # 如果迭代次数大于0，计算上一次和当前的误差变化
        if ii > 0:
            dE = errors[-2] - errors[-1]
            # 如果误差变化小于指定的容差与当前误差乘积，认为已收敛
            if dE < tol * errors[-1]:
                if verbose == 1:
                    # 在 verbose 为1时打印一个换行符
                    print("")
                elif verbose:
                    print("--- Convergence reached after %d iterations" % ii)
                break
        # 每5次迭代执行一次回调函数（如果存在）
        if ii % 5 == 0 and callback is not None:
            callback(locals())

    # 如果需要返回迭代次数，返回编码、字典、错误列表及迭代次数加1
    if return_n_iter:
        return code, dictionary, errors, ii + 1
    # 否则，返回编码、字典、错误列表
    else:
        return code, dictionary, errors
# 使用装饰器 @validate_params 对 dict_learning_online 函数进行参数验证和类型检查
@validate_params(
    {
        "X": ["array-like"],  # 参数 X 应为类数组类型
        "return_code": ["boolean"],  # 参数 return_code 应为布尔类型
        "method": [StrOptions({"cd", "lars"})],  # 参数 method 应为字符串，且取值为 "cd" 或 "lars"
        "method_max_iter": [Interval(Integral, 0, None, closed="left")],  # 参数 method_max_iter 应为整数，且大于等于 0
    },
    prefer_skip_nested_validation=False,  # 禁止跳过嵌套验证
)
def dict_learning_online(
    X,
    n_components=2,  # 默认字典中原子的数量为 2
    *,
    alpha=1,  # 稀疏性控制参数，默认为 1
    max_iter=100,  # 最大迭代次数，默认为 100
    return_code=True,  # 是否返回稀疏编码 U，默认为 True
    dict_init=None,  # 字典的初始值，默认为 None
    callback=None,  # 迭代结束后调用的回调函数，默认为 None
    batch_size=256,  # 每个小批量的样本数，默认为 256
    verbose=False,  # 控制过程的冗长度，默认为 False
    shuffle=True,  # 是否在每次迭代前对数据进行随机排序，默认为 True
    n_jobs=None,  # 并行作业的数量，默认为 None
    method="lars",  # 使用的求解方法，默认为 "lars"
    random_state=None,  # 随机数生成器的种子，默认为 None
    positive_dict=False,  # 是否强制字典的所有条目为非负，默认为 False
    positive_code=False,  # 是否强制稀疏编码的所有条目为非负，默认为 False
    method_max_iter=1000,  # 方法的最大迭代次数，默认为 1000
    tol=1e-3,  # 迭代停止的容差，默认为 0.001
    max_no_improvement=10,  # 在停止前允许的最大迭代次数，默认为 10
):
    """Solve a dictionary learning matrix factorization problem online.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_Fro^2 + alpha * || U ||_1,1
                     (U,V)
                     with || V_k ||_2 = 1 for all  0 <= k < n_components

    where V is the dictionary and U is the sparse code. ||.||_Fro stands for
    the Frobenius norm and ||.||_1,1 stands for the entry-wise matrix norm
    which is the sum of the absolute values of all the entries in the matrix.
    This is accomplished by repeatedly iterating over mini-batches by slicing
    the input data.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.

    n_components : int or None, default=2
        Number of dictionary atoms to extract. If None, then ``n_components``
        is set to ``n_features``.

    alpha : float, default=1
        Sparsity controlling parameter.

    max_iter : int, default=100
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.

        .. versionadded:: 1.1

        .. deprecated:: 1.4
           `max_iter=None` is deprecated in 1.4 and will be removed in 1.6.
           Use the default value (i.e. `100`) instead.

    return_code : bool, default=True
        Whether to also return the code U or just the dictionary `V`.

    dict_init : ndarray of shape (n_components, n_features), default=None
        Initial values for the dictionary for warm restart scenarios.
        If `None`, the initial values for the dictionary are created
        with an SVD decomposition of the data via
        :func:`~sklearn.utils.extmath.randomized_svd`.

    callback : callable, default=None
        A callable that gets invoked at the end of each iteration.

    batch_size : int, default=256
        The number of samples to take in each batch.

        .. versionchanged:: 1.3
           The default value of `batch_size` changed from 3 to 256 in version 1.3.

    verbose : bool, default=False
        To control the verbosity of the procedure.

    shuffle : bool, default=True
        Whether to shuffle the data before each iteration.

    n_jobs : int or None, default=None
        The number of parallel jobs to run for the computation.
        `None` means 1 unless in a `joblib.parallel_backend` context.
        `-1` means using all processors.

    method : {'cd', 'lars'}, default='lars'
        The method used for solving the optimization problem.
        'cd' stands for Coordinate Descent.
        'lars' stands for Least Angle Regression.

    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    positive_dict : bool, default=False
        Whether to enforce that the dictionary has all non-negative entries.

    positive_code : bool, default=False
        Whether to enforce that the code has all non-negative entries.

    method_max_iter : int, default=1000
        Maximum number of iterations for the specific solver method.

    tol : float, default=1e-3
        Tolerance for stopping criteria.

    max_no_improvement : int, default=10
        Maximum number of iterations with no improvement to wait before early stopping.

    """
    # 函数的具体实现未展示在此处
    shuffle : bool, default=True
        是否在分批处理前对数据进行随机排列。

    n_jobs : int, default=None
        并行作业的数量。
        ``None`` 表示除非在 :obj:`joblib.parallel_backend` 上下文中，否则为1。
        ``-1`` 表示使用所有处理器。有关详细信息，请参见 :term:`术语表 <n_jobs>`。

    method : {'lars', 'cd'}, default='lars'
        * `'lars'`: 使用最小角度回归法解决 Lasso 问题 (`linear_model.lars_path`);
        * `'cd'`: 使用坐标下降法计算 Lasso 解 (`linear_model.Lasso`)。如果估计的成分稀疏，则 LARS 方法速度更快。

    random_state : int, RandomState instance or None, default=None
        用于在未指定 ``dict_init`` 时初始化字典、在 ``shuffle`` 设置为 ``True`` 时随机洗牌数据以及更新字典。
        为了在多次函数调用间获得可重复的结果，请传递一个整数。
        请参见 :term:`术语表 <random_state>`。

    positive_dict : bool, default=False
        在找到字典时是否强制为正。

        .. versionadded:: 0.20

    positive_code : bool, default=False
        在找到编码时是否强制为正。

        .. versionadded:: 0.20

    method_max_iter : int, default=1000
        解决 Lasso 问题时执行的最大迭代次数。

        .. versionadded:: 0.22

    tol : float, default=1e-3
        基于字典在两步之间的差异范数来控制早期停止。

        要禁用基于字典变化的早期停止，请将 `tol` 设置为 0.0。

        .. versionadded:: 1.1

    max_no_improvement : int, default=10
        基于连续多个小批次未改善平滑成本函数的数量来控制早期停止。

        要禁用基于成本函数的收敛检测，请将 `max_no_improvement` 设置为 None。

        .. versionadded:: 1.1

    Returns
    -------
    code : ndarray of shape (n_samples, n_components),
        稀疏编码（仅在 `return_code=True` 时返回）。

    dictionary : ndarray of shape (n_components, n_features),
        字典学习问题的解决方案。

    n_iter : int
        运行的迭代次数。仅在 `return_n_iter` 设置为 `True` 时返回。

    See Also
    --------
    dict_learning : 解决字典学习矩阵分解问题。
    DictionaryLearning : 找到能稀疏编码数据的字典。
    MiniBatchDictionaryLearning : 字典学习算法的快速、不太精确版本。
    SparsePCA : 稀疏主成分分析。
    MiniBatchSparsePCA : 小批量稀疏主成分分析。

    Examples
    --------
    >>> import numpy as np
    # TODO(1.6): remove in 1.6
    # 在版本 1.4 中，`max_iter=None` 已被弃用，版本 1.6 将移除此功能。
    # 建议使用默认值 `100` 代替。
    if max_iter is None:
        warn(
            (
                "`max_iter=None` is deprecated in version 1.4 and will be removed in "
                "version 1.6. Use the default value (i.e. `100`) instead."
            ),
            FutureWarning,
        )
        max_iter = 100
    
    # 使用指定的方法来进行转换算法，此处为 Lasso 方法
    transform_algorithm = "lasso_" + method
    
    # 创建 MiniBatchDictionaryLearning 对象，使用给定的参数进行字典学习
    est = MiniBatchDictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        max_iter=max_iter,
        n_jobs=n_jobs,
        fit_algorithm=method,
        batch_size=batch_size,
        shuffle=shuffle,
        dict_init=dict_init,
        random_state=random_state,
        transform_algorithm=transform_algorithm,
        transform_alpha=alpha,
        positive_code=positive_code,
        positive_dict=positive_dict,
        transform_max_iter=method_max_iter,
        verbose=verbose,
        callback=callback,
        tol=tol,
        max_no_improvement=max_no_improvement,
    ).fit(X)
    
    # 如果不需要返回编码结果，则返回学习得到的字典的成分（components）
    if not return_code:
        return est.components_
    # 否则，返回编码（code）和学习得到的字典的成分（components）
    else:
        code = est.transform(X)
        return code, est.components_
# 使用装饰器 validate_params 对函数参数进行验证和类型检查
@validate_params(
    {
        "X": ["array-like"],  # 参数 X 应为类数组类型
        "method": [StrOptions({"lars", "cd"})],  # 参数 method 应为字符串，取值为 {'lars', 'cd'} 中的一种
        "return_n_iter": ["boolean"],  # 参数 return_n_iter 应为布尔值
        "method_max_iter": [Interval(Integral, 0, None, closed="left")],  # 参数 method_max_iter 应为大于等于0的整数
    },
    prefer_skip_nested_validation=False,  # 设置是否跳过嵌套验证为 False
)
# 定义函数 dict_learning，用于解决字典学习的矩阵分解问题
def dict_learning(
    X,  # 数据矩阵，形状为 (n_samples, n_features)
    n_components,  # 需要提取的字典原子数目
    *,
    alpha,  # 稀疏性控制参数
    max_iter=100,  # 最大迭代次数，默认为 100
    tol=1e-8,  # 停止条件的容差，默认为 1e-8
    method="lars",  # 使用的方法，默认为 'lars'
    n_jobs=None,  # 并行工作的作业数，默认为 None
    dict_init=None,  # 字典的初始值，用于热启动场景，默认为 None
    code_init=None,  # 稀疏编码的初始值，用于热启动场景，默认为 None
    callback=None,  # 每五次迭代调用的回调函数，默认为 None
    verbose=False,  # 控制过程的详细程度，默认为 False
    random_state=None,  # 随机数种子，默认为 None
    return_n_iter=False,  # 是否返回迭代次数，默认为 False
    positive_dict=False,  # 字典是否非负，默认为 False
    positive_code=False,  # 稀疏编码是否非负，默认为 False
    method_max_iter=1000,  # 方法的最大迭代次数，默认为 1000
):
    """Solve a dictionary learning matrix factorization problem.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_Fro^2 + alpha * || U ||_1,1
                     (U,V)
                    with || V_k ||_2 = 1 for all  0 <= k < n_components

    where V is the dictionary and U is the sparse code. ||.||_Fro stands for
    the Frobenius norm and ||.||_1,1 stands for the entry-wise matrix norm
    which is the sum of the absolute values of all the entries in the matrix.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.

    n_components : int
        Number of dictionary atoms to extract.

    alpha : int or float
        Sparsity controlling parameter.

    max_iter : int, default=100
        Maximum number of iterations to perform.

    tol : float, default=1e-8
        Tolerance for the stopping condition.

    method : {'lars', 'cd'}, default='lars'
        The method used:

        * `'lars'`: uses the least angle regression method to solve the lasso
           problem (`linear_model.lars_path`);
        * `'cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). Lars will be faster if
          the estimated components are sparse.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    dict_init : ndarray of shape (n_components, n_features), default=None
        Initial value for the dictionary for warm restart scenarios. Only used
        if `code_init` and `dict_init` are not None.

    code_init : ndarray of shape (n_samples, n_components), default=None
        Initial value for the sparse code for warm restart scenarios. Only used
        if `code_init` and `dict_init` are not None.

    callback : callable, default=None
        Callable that gets invoked every five iterations.

    verbose : bool, default=False
        To control the verbosity of the procedure.

    random_state : int, RandomState instance, default=None
        Determines the random number generation for dictionary and sparse code
        initialization. Pass an int for reproducible output across multiple
        function calls.

    return_n_iter : bool, default=False
        Whether to return the number of iterations.

    positive_dict : bool, default=False
        Whether to enforce non-negativity on the dictionary. If True, the
        dictionary components are enforced to be non-negative.

    positive_code : bool, default=False
        Whether to enforce non-negativity on the sparse code. If True, the
        sparse codes are enforced to be non-negative.

    method_max_iter : int, default=1000
        Maximum number of iterations to perform for the method. This parameter
        controls the maximum number of iterations for the optimization method
        (e.g., LARS or CD).

    """
    estimator = DictionaryLearning(
        n_components=n_components,  # 设定字典的大小（成分数）
        alpha=alpha,  # 设定字典学习过程中的稀疏性惩罚项的强度
        max_iter=max_iter,  # 设定最大迭代次数
        tol=tol,  # 设定收敛判据
        fit_algorithm=method,  # 设定拟合算法
        n_jobs=n_jobs,  # 设定并行运行的作业数
        dict_init=dict_init,  # 设定字典的初始化策略
        callback=callback,  # 设定回调函数
        code_init=code_init,  # 设定稀疏编码的初始化策略
        verbose=verbose,  # 设定详细程度
        random_state=random_state,  # 设定随机数生成器的种子或状态
        positive_code=positive_code,  # 是否强制编码为正（非负）
        positive_dict=positive_dict,  # 是否强制字典为正（非负）
        transform_max_iter=method_max_iter,  # 设定变换的最大迭代次数
    ).set_output(transform="default")  # 设置输出的转换类型为默认值
    code = estimator.fit_transform(X)  # 使用给定的数据 X 拟合模型并进行转换
    # 如果设置了 return_n_iter 标志为 True，则返回以下四个值作为元组
    if return_n_iter:
        return (
            code,                   # 返回优化后的代码
            estimator.components_,  # 返回估计器的主成分
            estimator.error_,       # 返回估计器的误差
            estimator.n_iter_,      # 返回估计器的迭代次数
        )
    # 否则，返回以下三个值作为元组
    return code, estimator.components_, estimator.error_
    def __init__(
        self,
        transform_algorithm,
        transform_n_nonzero_coefs,
        transform_alpha,
        split_sign,
        n_jobs,
        positive_code,
        transform_max_iter,
    ):
        # 初始化稀疏编码器的参数
        self.transform_algorithm = transform_algorithm  # 设置变换算法
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs  # 设置非零系数的数量
        self.transform_alpha = transform_alpha  # 设置正则化参数
        self.transform_max_iter = transform_max_iter  # 设置最大迭代次数
        self.split_sign = split_sign  # 设置是否分离正负部分
        self.n_jobs = n_jobs  # 设置并行工作的数量
        self.positive_code = positive_code  # 设置是否限制编码为正数

    def _transform(self, X, dictionary):
        """Private method allowing to accommodate both DictionaryLearning and
        SparseCoder."""
        X = self._validate_data(X, reset=False)  # 验证输入数据格式

        if hasattr(self, "alpha") and self.transform_alpha is None:
            transform_alpha = self.alpha  # 如果对象有 alpha 属性且 transform_alpha 为 None，则使用对象的 alpha
        else:
            transform_alpha = self.transform_alpha  # 否则使用实例的 transform_alpha

        # 执行稀疏编码过程
        code = sparse_encode(
            X,
            dictionary,
            algorithm=self.transform_algorithm,
            n_nonzero_coefs=self.transform_n_nonzero_coefs,
            alpha=transform_alpha,
            max_iter=self.transform_max_iter,
            n_jobs=self.n_jobs,
            positive=self.positive_code,
        )

        if self.split_sign:
            # 如果需要分离正负部分，则进行处理
            n_samples, n_features = code.shape
            split_code = np.empty((n_samples, 2 * n_features))
            split_code[:, :n_features] = np.maximum(code, 0)  # 正部分
            split_code[:, n_features:] = -np.minimum(code, 0)  # 负部分
            code = split_code  # 更新 code

        return code  # 返回稀疏编码结果

    def transform(self, X):
        """Encode the data as a sparse combination of the dictionary atoms.

        Coding method is determined by the object parameter
        `transform_algorithm`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data to be transformed, must have the same number of
            features as the data used to train the model.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)  # 检查模型是否已拟合
        return self._transform(X, self.components_)  # 调用 _transform 方法进行数据变换
    # 字典的形状是 (n_components, n_features)，用于稀疏编码的字典原子，假设行已经归一化为单位范数。

    transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', \
            'threshold'}, default='omp'
        # 数据变换所使用的算法：

        - `'lars'`: 使用最小角回归方法 (`linear_model.lars_path`);
        - `'lasso_lars'`: 使用 Lars 算法计算 Lasso 解决方案;
        - `'lasso_cd'`: 使用坐标下降法计算 Lasso 解决方案 (`linear_model.Lasso`)。如果估计的成分是稀疏的，`'lasso_lars'` 比较快;
        - `'omp'`: 使用正交匹配追踪法估计稀疏解;
        - `'threshold'`: 将投影 `dictionary * X'` 中小于 alpha 的所有系数压缩为零。

    transform_n_nonzero_coefs : int, default=None
        # 目标是每列解中非零系数的数量。仅在 `algorithm='lars'` 和 `algorithm='omp'` 中使用，并在 `omp` 情况下被 `alpha` 覆盖。如果为 `None`，则为 `int(n_features / 10)`。

    transform_alpha : float, default=None
        # 如果 `algorithm='lasso_lars'` 或 `algorithm='lasso_cd'`，`alpha` 是应用于 L1 范数的惩罚。
        # 如果 `algorithm='threshold'`，`alpha` 是小于该阈值的系数的绝对值将被压缩为零。
        # 如果 `algorithm='omp'`，`alpha` 是容差参数：目标重构误差的值。在这种情况下，它覆盖 `n_nonzero_coefs`。
        # 如果为 `None`，默认为 1。

    split_sign : bool, default=False
        # 是否将稀疏特征向量分割成其负部分和正部分的连接。这可以提高下游分类器的性能。

    n_jobs : int, default=None
        # 要运行的并行作业数量。
        # `None` 表示 1，除非在 `joblib.parallel_backend` 上下文中。
        # `-1` 表示使用所有处理器。有关详细信息，请参阅“术语表 <n_jobs>`。

    positive_code : bool, default=False
        # 在找到代码时是否强制为正。

        .. versionadded:: 0.20
            # 添加于版本 0.20。

    transform_max_iter : int, default=1000
        # 如果 `algorithm='lasso_cd'` 或 `lasso_lars`，执行的最大迭代次数。

        .. versionadded:: 0.22
            # 添加于版本 0.22。

    Attributes
    ----------
    n_components_ : int
        # 原子的数量。

    n_features_in_ : int
        # 在 `fit` 过程中看到的特征数量。

        .. versionadded:: 0.24
            # 添加于版本 0.24。

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 `fit` 过程中看到的特征名称。仅当 `X` 的特征名称都是字符串时定义。

        .. versionadded:: 1.0
            # 添加于版本 1.0。

    See Also
    --------
    """
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    MiniBatchDictionaryLearning : A faster, less accurate, version of the
        dictionary learning algorithm.
    MiniBatchSparsePCA : Mini-batch Sparse Principal Components Analysis.
    SparsePCA : Sparse Principal Components Analysis.
    sparse_encode : Sparse coding where each row of the result is the solution
        to a sparse coding problem.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.decomposition import SparseCoder
    >>> X = np.array([[-1, -1, -1], [0, 0, 3]])
    >>> dictionary = np.array(
    ...     [[0, 1, 0],
    ...      [-1, -1, 2],
    ...      [1, 1, 1],
    ...      [0, 1, 1],
    ...      [0, 2, 1]],
    ...    dtype=np.float64
    ... )
    >>> coder = SparseCoder(
    ...     dictionary=dictionary, transform_algorithm='lasso_lars',
    ...     transform_alpha=1e-10,
    ... )
    >>> coder.transform(X)
    array([[ 0.,  0., -1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.]])
    """

    # 定义一个类，实现稀疏编码的功能
    _required_parameters = ["dictionary"]

    def __init__(
        self,
        dictionary,
        *,
        transform_algorithm="omp",
        transform_n_nonzero_coefs=None,
        transform_alpha=None,
        split_sign=False,
        n_jobs=None,
        positive_code=False,
        transform_max_iter=1000,
    ):
        # 调用父类的初始化方法，设置转换算法和其他参数
        super().__init__(
            transform_algorithm,
            transform_n_nonzero_coefs,
            transform_alpha,
            split_sign,
            n_jobs,
            positive_code,
            transform_max_iter,
        )
        # 设置类的字典属性
        self.dictionary = dictionary

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : Ignored
            Not used, present for API consistency by convention.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self

    def transform(self, X, y=None):
        """Encode the data as a sparse combination of the dictionary atoms.

        Coding method is determined by the object parameter
        `transform_algorithm`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        # 调用父类的_transform方法进行数据编码，使用给定的字典
        return super()._transform(X, self.dictionary)
    # 返回一个字典，指定模型不需要拟合（fit），并且保存的数据类型是 np.float64 和 np.float32
    def _more_tags(self):
        return {
            "requires_fit": False,
            "preserves_dtype": [np.float64, np.float32],
        }

    # 返回字典的第一个维度大小，即字典的行数，通常代表原子（atoms）的数量
    @property
    def n_components_(self):
        """Number of atoms."""
        return self.dictionary.shape[0]

    # 返回字典的第二个维度大小，通常表示在拟合（fit）过程中看到的特征数量
    @property
    def n_features_in_(self):
        """Number of features seen during `fit`."""
        return self.dictionary.shape[1]

    # 返回转换后输出特征的数量，与原子数量相同
    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.n_components_
class DictionaryLearning(_BaseSparseCoding, BaseEstimator):
    """Dictionary learning.

    Finds a dictionary (a set of atoms) that performs well at sparsely
    encoding the fitted data.

    Solves the optimization problem::

        (U^*,V^*) = argmin 0.5 || X - U V ||_Fro^2 + alpha * || U ||_1,1
                    (U,V)
                    with || V_k ||_2 <= 1 for all  0 <= k < n_components

    ||.||_Fro stands for the Frobenius norm and ||.||_1,1 stands for
    the entry-wise matrix norm which is the sum of the absolute values
    of all the entries in the matrix.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    n_components : int, default=None
        Number of dictionary elements to extract. If None, then ``n_components``
        is set to ``n_features``.

    alpha : float, default=1.0
        Sparsity controlling parameter.

    max_iter : int, default=1000
        Maximum number of iterations to perform.

    tol : float, default=1e-8
        Tolerance for numerical error.

    fit_algorithm : {'lars', 'cd'}, default='lars'
        * `'lars'`: uses the least angle regression method to solve the lasso
          problem (:func:`~sklearn.linear_model.lars_path`);
        * `'cd'`: uses the coordinate descent method to compute the
          Lasso solution (:class:`~sklearn.linear_model.Lasso`). Lars will be
          faster if the estimated components are sparse.

        .. versionadded:: 0.17
           *cd* coordinate descent method to improve speed.

    transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', \
            'threshold'}, default='omp'
        Algorithm used to transform the data:

        - `'lars'`: uses the least angle regression method
          (:func:`~sklearn.linear_model.lars_path`);
        - `'lasso_lars'`: uses Lars to compute the Lasso solution.
        - `'lasso_cd'`: uses the coordinate descent method to compute the
          Lasso solution (:class:`~sklearn.linear_model.Lasso`). `'lasso_lars'`
          will be faster if the estimated components are sparse.
        - `'omp'`: uses orthogonal matching pursuit to estimate the sparse
          solution.
        - `'threshold'`: squashes to zero all coefficients less than alpha from
          the projection ``dictionary * X'``.

        .. versionadded:: 0.17
           *lasso_cd* coordinate descent method to improve speed.

    transform_n_nonzero_coefs : int, default=None
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and
        `algorithm='omp'`. If `None`, then
        `transform_n_nonzero_coefs=int(n_features / 10)`.
    """
    pass
    # transform_alpha 是一个浮点数，用于指定正则化的强度，具体取决于所选的算法
    # 当 algorithm='lasso_lars' 或 algorithm='lasso_cd' 时，alpha 是应用于 L1 范数的惩罚值
    # 当 algorithm='threshold' 时，alpha 是将系数压缩到零以下的阈值的绝对值
    # 如果为 None，则默认为 alpha 的值
    transform_alpha : float, default=None

    # n_jobs 是一个整数或 None，默认为 None
    # 指定要并行运行的作业数量
    # None 表示除非在 joblib.parallel_backend 上下文中，否则默认为 1
    # -1 表示使用所有处理器
    # 详细信息请参见“术语表”中的“n_jobs”
    n_jobs : int or None, default=None

    # code_init 是一个形状为 (n_samples, n_components) 的 ndarray，默认为 None
    # 用于热启动的代码的初始值
    # 仅在 code_init 和 dict_init 都不为 None 时使用
    code_init : ndarray of shape (n_samples, n_components), default=None

    # dict_init 是一个形状为 (n_components, n_features) 的 ndarray，默认为 None
    # 字典的初始值，用于热启动
    # 仅在 code_init 和 dict_init 都不为 None 时使用
    dict_init : ndarray of shape (n_components, n_features), default=None

    # callback 是一个可调用对象，默认为 None
    # 每五次迭代调用一次的回调函数
    # 在版本 1.3 中添加

    callback : callable, default=None

    # verbose 是一个布尔值，默认为 False
    # 控制过程的详细程度

    verbose : bool, default=False

    # split_sign 是一个布尔值，默认为 False
    # 是否将稀疏特征向量分割为其负部分和正部分的连接
    # 这可以提高下游分类器的性能

    split_sign : bool, default=False

    # random_state 是一个整数、RandomState 实例或 None，默认为 None
    # 用于在未指定 dict_init 时初始化字典、在 shuffle 设置为 True 时随机洗牌数据以及更新字典
    # 传递整数以便在多个函数调用间获得可重复的结果
    # 详细信息请参见“术语表”中的“random_state”

    random_state : int, RandomState instance or None, default=None

    # positive_code 是一个布尔值，默认为 False
    # 在寻找代码时是否强制为正数

    positive_code : bool, default=False

    # positive_dict 是一个布尔值，默认为 False
    # 在寻找字典时是否强制为正数

    positive_dict : bool, default=False

    # transform_max_iter 是一个整数，默认为 1000
    # 如果 algorithm='lasso_cd' 或 'lasso_lars'，则执行的最大迭代次数
    # 在版本 0.22 中添加

    transform_max_iter : int, default=1000

# Attributes 下面是属性部分的说明，包括 components_, error_, n_features_in_, feature_names_in_, n_iter_
# 详细信息可以在相应的类文档或者参考文献中找到
# 以下省略 See Also 部分
    # 定义一个字典，描述了 MiniBatchDictionaryLearning 类的参数约束
    _parameter_constraints: dict = {
        # 希望 n_components 是一个大于等于 1 的整数
        "n_components": [Interval(Integral, 1, None, closed="left"), None],
        # 希望 alpha 是一个大于等于 0 的实数
        "alpha": [Interval(Real, 0, None, closed="left")],
        # 希望 max_iter 是一个大于等于 0 的整数
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        # 希望 tol 是一个大于等于 0 的实数
        "tol": [Interval(Real, 0, None, closed="left")],
        # 希望 fit_algorithm 是 {"lars", "cd"} 中的一个字符串
        "fit_algorithm": [StrOptions({"lars", "cd"})],
        # 希望 transform_algorithm 是 {"lasso_lars", "lasso_cd", "lars", "omp", "threshold"} 中的一个字符串
        "transform_algorithm": [
            StrOptions({"lasso_lars", "lasso_cd", "lars", "omp", "threshold"})
        ],
        # 希望 transform_n_nonzero_coefs 是一个大于等于 1 的整数
        "transform_n_nonzero_coefs": [Interval(Integral, 1, None, closed="left"), None],
        # 希望 transform_alpha 是一个大于等于 0 的实数
        "transform_alpha": [Interval(Real, 0, None, closed="left"), None],
        # 希望 n_jobs 是一个整数或者为 None
        "n_jobs": [Integral, None],
        # 希望 code_init 是一个 numpy 数组或者为 None
        "code_init": [np.ndarray, None],
        # 希望 dict_init 是一个 numpy 数组或者为 None
        "dict_init": [np.ndarray, None],
        # 希望 callback 是一个可调用对象或者为 None
        "callback": [callable, None],
        # 希望 verbose 是一个字符串 "verbose"
        "verbose": ["verbose"],
        # 希望 split_sign 是一个布尔值
        "split_sign": ["boolean"],
        # 希望 random_state 是一个字符串 "random_state"
        "random_state": ["random_state"],
        # 希望 positive_code 是一个布尔值
        "positive_code": ["boolean"],
        # 希望 positive_dict 是一个布尔值
        "positive_dict": ["boolean"],
        # 希望 transform_max_iter 是一个大于等于 0 的整数
        "transform_max_iter": [Interval(Integral, 0, None, closed="left")],
    }
    # 初始化函数，设置稀疏编码器的参数和选项
    def __init__(
        self,
        n_components=None,
        *,
        alpha=1,
        max_iter=1000,
        tol=1e-8,
        fit_algorithm="lars",
        transform_algorithm="omp",
        transform_n_nonzero_coefs=None,
        transform_alpha=None,
        n_jobs=None,
        code_init=None,
        dict_init=None,
        callback=None,
        verbose=False,
        split_sign=False,
        random_state=None,
        positive_code=False,
        positive_dict=False,
        transform_max_iter=1000,
    ):
        # 调用父类的初始化方法，设置转换算法、非零系数、alpha值、是否分割符号等参数
        super().__init__(
            transform_algorithm,
            transform_n_nonzero_coefs,
            transform_alpha,
            split_sign,
            n_jobs,
            positive_code,
            transform_max_iter,
        )
        # 设置稀疏编码器的特定参数
        self.n_components = n_components  # 设置稀疏编码器的成分数
        self.alpha = alpha  # 设置正则化参数alpha
        self.max_iter = max_iter  # 设置最大迭代次数
        self.tol = tol  # 设置容差阈值
        self.fit_algorithm = fit_algorithm  # 设置拟合算法
        self.code_init = code_init  # 设置编码初始化策略
        self.dict_init = dict_init  # 设置字典初始化策略
        self.callback = callback  # 设置回调函数
        self.verbose = verbose  # 设置是否输出详细信息
        self.random_state = random_state  # 设置随机数种子
        self.positive_dict = positive_dict  # 设置是否强制字典为非负数

    # 拟合模型，从数据X中学习
    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 调用父类的fit_transform方法进行拟合和转换
        self.fit_transform(X)
        return self  # 返回自身对象
    def fit_transform(self, X, y=None):
        """
        Fit the model from data in X and return the transformed data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        V : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        # 检查并确保使用的编码方法是正数编码
        _check_positive_coding(method=self.fit_algorithm, positive=self.positive_code)

        # 构建方法字符串，以便在算法中使用
        method = "lasso_" + self.fit_algorithm

        # 检查并设置随机状态
        random_state = check_random_state(self.random_state)
        # 验证并返回验证后的数据 X
        X = self._validate_data(X)

        # 确定输出的成分数目
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        # 调用字典学习函数来执行数据转换
        V, U, E, self.n_iter_ = _dict_learning(
            X,
            n_components,
            alpha=self.alpha,
            tol=self.tol,
            max_iter=self.max_iter,
            method=method,
            method_max_iter=self.transform_max_iter,
            n_jobs=self.n_jobs,
            code_init=self.code_init,
            dict_init=self.dict_init,
            callback=self.callback,
            verbose=self.verbose,
            random_state=random_state,
            return_n_iter=True,
            positive_dict=self.positive_dict,
            positive_code=self.positive_code,
        )
        # 将学到的组件保存在对象中
        self.components_ = U
        # 保存误差信息
        self.error_ = E

        # 返回转换后的数据
        return V

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        # 返回输出特征的数量，即组件的行数
        return self.components_.shape[0]

    def _more_tags(self):
        # 返回更多的标签，这里指定了数据类型的保持方式
        return {
            "preserves_dtype": [np.float64, np.float32],
        }
class MiniBatchDictionaryLearning(_BaseSparseCoding, BaseEstimator):
    """Mini-batch dictionary learning.

    Finds a dictionary (a set of atoms) that performs well at sparsely
    encoding the fitted data.

    Solves the optimization problem::

       (U^*,V^*) = argmin 0.5 || X - U V ||_Fro^2 + alpha * || U ||_1,1
                    (U,V)
                    with || V_k ||_2 <= 1 for all  0 <= k < n_components

    ||.||_Fro stands for the Frobenius norm and ||.||_1,1 stands for
    the entry-wise matrix norm which is the sum of the absolute values
    of all the entries in the matrix.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    n_components : int, default=None
        Number of dictionary elements to extract.

    alpha : float, default=1
        Sparsity controlling parameter.

    max_iter : int, default=1_000
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.

        .. versionadded:: 1.1

        .. deprecated:: 1.4
           `max_iter=None` is deprecated in 1.4 and will be removed in 1.6.
           Use the default value (i.e. `1_000`) instead.

    fit_algorithm : {'lars', 'cd'}, default='lars'
        The algorithm used:

        - `'lars'`: uses the least angle regression method to solve the lasso
          problem (`linear_model.lars_path`)
        - `'cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). Lars will be faster if
          the estimated components are sparse.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    batch_size : int, default=256
        Number of samples in each mini-batch.

        .. versionchanged:: 1.3
           The default value of `batch_size` changed from 3 to 256 in version 1.3.

    shuffle : bool, default=True
        Whether to shuffle the samples before forming batches.

    dict_init : ndarray of shape (n_components, n_features), default=None
        Initial value of the dictionary for warm restart scenarios.
    """

    def __init__(self, n_components=None, alpha=1, max_iter=1000, fit_algorithm='lars',
                 n_jobs=None, batch_size=256, shuffle=True, dict_init=None):
        # 调用父类 _BaseSparseCoding 和 BaseEstimator 的初始化方法
        super().__init__()
        
        # 设置类的属性值
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.fit_algorithm = fit_algorithm
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dict_init = dict_init
    ```python`
        # 定义数据转换算法的参数，支持的算法包括 'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'，默认为 'omp'
        transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', \
                'threshold'}, default='omp'
            # 描述数据转换算法的选择：
            # - `'lars'`: 使用最小角回归方法 (`linear_model.lars_path`)
            # - `'lasso_lars'`: 使用 Lars 算法计算 Lasso 解
            # - `'lasso_cd'`: 使用坐标下降法计算 Lasso 解 (`linear_model.Lasso`)。当估计的组件稀疏时，`'lasso_lars'` 会更快。
            # - `'omp'`: 使用正交匹配追踪来估计稀疏解
            # - `'threshold'`: 将所有小于 alpha 的系数压缩为零，来自投影 ``dictionary * X'``。
    
        # 定义非零系数的目标数量，默认为 None。仅在使用 `algorithm='lars'` 和 `algorithm='omp'` 时使用。
        transform_n_nonzero_coefs : int, default=None
            Number of nonzero coefficients to target in each column of the solution. This is only used by `algorithm='lars'` and `algorithm='omp'`. If `None`, then `transform_n_nonzero_coefs=int(n_features / 10)`.
    
        # 定义变换的 alpha 值，默认为 None。用于不同算法的惩罚系数。
        transform_alpha : float, default=None
            If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the penalty applied to the L1 norm.
            If `algorithm='threshold'`, `alpha` is the absolute value of the threshold below which coefficients will be squashed to zero.
            If `None`, defaults to `alpha`.
    
            # 更新版本说明：从 1.0 改为 `alpha` 默认值。
            .. versionchanged:: 1.2
                When None, default value changed from 1.0 to `alpha`.
    
        # 定义是否启用详细模式，默认为 False，控制过程的详细程度
        verbose : bool or int, default=False
            To control the verbosity of the procedure.
    
        # 定义是否将稀疏特征向量拆分为负部分和正部分的连接，这可能提高下游分类器的性能
        split_sign : bool, default=False
            Whether to split the sparse feature vector into the concatenation of its negative part and its positive part. This can improve the performance of downstream classifiers.
    
        # 随机种子初始化参数，用于初始化字典、数据随机打乱和字典更新，默认为 None
        random_state : int, RandomState instance or None, default=None
            Used for initializing the dictionary when ``dict_init`` is not specified, randomly shuffling the data when ``shuffle`` is set to ``True``, and updating the dictionary. Pass an int for reproducible results across multiple function calls.
            See :term:`Glossary <random_state>`.
    
        # 是否强制正性编码，默认为 False
        positive_code : bool, default=False
            Whether to enforce positivity when finding the code.
    
            # 版本增加说明，从 0.20 开始支持
            .. versionadded:: 0.20
    
        # 是否强制正性字典，默认为 False
        positive_dict : bool, default=False
            Whether to enforce positivity when finding the dictionary.
    
            # 版本增加说明，从 0.20 开始支持
            .. versionadded:: 0.20
    
        # 定义最大迭代次数，默认为 1000，仅在算法为 'lasso_cd' 或 'lasso_lars' 时有效
        transform_max_iter : int, default=1000
            Maximum number of iterations to perform if `algorithm='lasso_cd'` or `'lasso_lars'`.
    
            # 版本增加说明，从 0.22 开始支持
            .. versionadded:: 0.22
    
        # 定义回调函数，默认为 None，每次迭代结束时调用的可调用对象
        callback : callable, default=None
            A callable that gets invoked at the end of each iteration.
    
            # 版本增加说明，从 1.1 开始支持
            .. versionadded:: 1.1
    tol : float, default=1e-3
        Control early stopping based on the norm of the differences in the
        dictionary between 2 steps.
        在两次迭代之间字典差异的范数基础上控制早停止。

        To disable early stopping based on changes in the dictionary, set
        `tol` to 0.0.
        要禁用基于字典变化的早停止，请将 `tol` 设置为 0.0。

        .. versionadded:: 1.1
        .. 版本新增：1.1

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini batches
        that does not yield an improvement on the smoothed cost function.
        控制基于连续的小批次数量的早停止，这些小批次未改善平滑的成本函数。

        To disable convergence detection based on cost function, set
        `max_no_improvement` to None.
        要禁用基于成本函数的收敛检测，请将 `max_no_improvement` 设置为 None。

        .. versionadded:: 1.1
        .. 版本新增：1.1

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Components extracted from the data.
        从数据中提取的组件。

    n_features_in_ : int
        Number of features seen during :term:`fit`.
        在 `fit` 过程中观察到的特征数量。

        .. versionadded:: 0.24
        .. 版本新增：0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        在 `fit` 过程中观察到的特征名称。仅当 `X` 的特征名称全部为字符串时定义。

        .. versionadded:: 1.0
        .. 版本新增：1.0

    n_iter_ : int
        Number of iterations over the full dataset.
        在整个数据集上的迭代次数。

    n_steps_ : int
        Number of mini-batches processed.
        处理的小批次数量。

        .. versionadded:: 1.1
        .. 版本新增：1.1

    See Also
    --------
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    MiniBatchSparsePCA : Mini-batch Sparse Principal Components Analysis.
    SparseCoder : Find a sparse representation of data from a fixed,
        precomputed dictionary.
    SparsePCA : Sparse Principal Components Analysis.
    相关链接

    References
    ----------

    J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009: Online dictionary learning
    for sparse coding (https://www.di.ens.fr/sierra/pdfs/icml09.pdf)
    引用文献：J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009 年：在线字典学习，用于稀疏编码（链接至文献）

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_sparse_coded_signal
    >>> from sklearn.decomposition import MiniBatchDictionaryLearning
    >>> X, dictionary, code = make_sparse_coded_signal(
    ...     n_samples=30, n_components=15, n_features=20, n_nonzero_coefs=10,
    ...     random_state=42)
    >>> dict_learner = MiniBatchDictionaryLearning(
    ...     n_components=15, batch_size=3, transform_algorithm='lasso_lars',
    ...     transform_alpha=0.1, max_iter=20, random_state=42)
    >>> X_transformed = dict_learner.fit_transform(X)
    示例：用法示例

    We can check the level of sparsity of `X_transformed`:
    我们可以检查 `X_transformed` 的稀疏水平：

    >>> np.mean(X_transformed == 0) > 0.5
    True

    We can compare the average squared euclidean norm of the reconstruction
    error of the sparse coded signal relative to the squared euclidean norm of
    the original signal:
    我们可以比较稀疏编码信号重构误差的平均平方欧几里得范数与原始信号的平方欧几里得范数之比：

    >>> X_hat = X_transformed @ dict_learner.components_
    >>> np.mean(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X ** 2, axis=1))
    0.052...
    ```
    # 定义参数的约束字典，包括每个参数的类型和取值范围
    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left"), None],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 0, None, closed="left"), Hidden(None)],
        "fit_algorithm": [StrOptions({"cd", "lars"})],
        "n_jobs": [None, Integral],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "shuffle": ["boolean"],
        "dict_init": [None, np.ndarray],
        "transform_algorithm": [
            StrOptions({"lasso_lars", "lasso_cd", "lars", "omp", "threshold"})
        ],
        "transform_n_nonzero_coefs": [Interval(Integral, 1, None, closed="left"), None],
        "transform_alpha": [Interval(Real, 0, None, closed="left"), None],
        "verbose": ["verbose"],
        "split_sign": ["boolean"],
        "random_state": ["random_state"],
        "positive_code": ["boolean"],
        "positive_dict": ["boolean"],
        "transform_max_iter": [Interval(Integral, 0, None, closed="left")],
        "callback": [None, callable],
        "tol": [Interval(Real, 0, None, closed="left")],
        "max_no_improvement": [Interval(Integral, 0, None, closed="left"), None],
    }

    # 初始化函数，设定类的各个参数
    def __init__(
        self,
        n_components=None,
        *,
        alpha=1,
        max_iter=1_000,
        fit_algorithm="lars",
        n_jobs=None,
        batch_size=256,
        shuffle=True,
        dict_init=None,
        transform_algorithm="omp",
        transform_n_nonzero_coefs=None,
        transform_alpha=None,
        verbose=False,
        split_sign=False,
        random_state=None,
        positive_code=False,
        positive_dict=False,
        transform_max_iter=1000,
        callback=None,
        tol=1e-3,
        max_no_improvement=10,
    ):
        # 调用父类的初始化方法，设置一些转换相关的参数
        super().__init__(
            transform_algorithm,
            transform_n_nonzero_coefs,
            transform_alpha,
            split_sign,
            n_jobs,
            positive_code,
            transform_max_iter,
        )
        # 设置对象的各个参数值
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.fit_algorithm = fit_algorithm
        self.dict_init = dict_init
        self.verbose = verbose
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.split_sign = split_sign
        self.random_state = random_state
        self.positive_dict = positive_dict
        self.callback = callback
        self.max_no_improvement = max_no_improvement
        self.tol = tol

    # 检查参数函数，根据输入的数据 X 对类的参数进行校验和设定
    def _check_params(self, X):
        # n_components 参数的设定
        self._n_components = self.n_components
        if self._n_components is None:
            self._n_components = X.shape[1]

        # 根据 fit_algorithm 和 positive_code 设定 _fit_algorithm 参数
        _check_positive_coding(self.fit_algorithm, self.positive_code)
        self._fit_algorithm = "lasso_" + self.fit_algorithm

        # 根据 batch_size 和输入数据 X 的形状设定 _batch_size 参数
        self._batch_size = min(self.batch_size, X.shape[0])
    def _initialize_dict(self, X, random_state):
        """Initialization of the dictionary."""
        if self.dict_init is not None:
            # 如果已提供初始字典，则使用提供的字典
            dictionary = self.dict_init
        else:
            # 否则，使用输入数据 X 的随机化奇异值分解（SVD）结果来初始化字典
            _, S, dictionary = randomized_svd(
                X, self._n_components, random_state=random_state
            )
            dictionary = S[:, np.newaxis] * dictionary

        if self._n_components <= len(dictionary):
            # 如果字典中的原子数量不超过指定的成分数，则截取相应数量的原子
            dictionary = dictionary[: self._n_components, :]
        else:
            # 否则，在字典末尾填充零向量，使其达到指定的成分数
            dictionary = np.concatenate(
                (
                    dictionary,
                    np.zeros(
                        (self._n_components - len(dictionary), dictionary.shape[1]),
                        dtype=dictionary.dtype,
                    ),
                )
            )

        # 将字典转换为指定的存储顺序（Fortran order），并确保不复制数据
        dictionary = check_array(dictionary, order="F", dtype=X.dtype, copy=False)
        # 将字典转换为写入需求的存储类型
        dictionary = np.require(dictionary, requirements="W")

        return dictionary

    def _update_inner_stats(self, X, code, batch_size, step):
        """Update the inner stats inplace."""
        if step < batch_size - 1:
            theta = (step + 1) * batch_size
        else:
            theta = batch_size**2 + step + 1 - batch_size
        beta = (theta + 1 - batch_size) / (theta + 1)

        # 更新内部统计信息 _A 和 _B
        self._A *= beta
        self._A += code.T @ code / batch_size
        self._B *= beta
        self._B += X.T @ code / batch_size

    def _minibatch_step(self, X, dictionary, random_state, step):
        """Perform the update on the dictionary for one minibatch."""
        batch_size = X.shape[0]

        # 计算该批次的稀疏编码 code
        code = _sparse_encode(
            X,
            dictionary,
            algorithm=self._fit_algorithm,
            alpha=self.alpha,
            n_jobs=self.n_jobs,
            positive=self.positive_code,
            max_iter=self.transform_max_iter,
            verbose=self.verbose,
        )

        # 计算批次的成本函数 batch_cost
        batch_cost = (
            0.5 * ((X - code @ dictionary) ** 2).sum()
            + self.alpha * np.sum(np.abs(code))
        ) / batch_size

        # 更新内部统计信息
        self._update_inner_stats(X, code, batch_size, step)

        # 更新字典 dictionary
        _update_dict(
            dictionary,
            X,
            code,
            self._A,
            self._B,
            verbose=self.verbose,
            random_state=random_state,
            positive=self.positive_dict,
        )

        return batch_cost

    def _check_convergence(
        self, X, batch_cost, new_dict, old_dict, n_samples, step, n_steps
    ):
        """Helper function to encapsulate the early stopping logic.

        Early stopping is based on two factors:
        - A small change of the dictionary between two minibatch updates. This is
          controlled by the tol parameter.
        - No more improvement on a smoothed estimate of the objective function for a
          a certain number of consecutive minibatch updates. This is controlled by
          the max_no_improvement parameter.
        """
        batch_size = X.shape[0]  # 获取当前批次的样本数

        # counts steps starting from 1 for user friendly verbose mode.
        step = step + 1  # 增加步数计数，用于显示当前迭代步数

        # Ignore 100 first steps or 1 epoch to avoid initializing the ewa_cost with a
        # too bad value
        if step <= min(100, n_samples / batch_size):  # 如果步数小于等于100或者样本数除以批次大小，跳过早期步骤
            if self.verbose:
                print(f"Minibatch step {step}/{n_steps}: mean batch cost: {batch_cost}")
            return False  # 返回False，表示不停止训练

        # Compute an Exponentially Weighted Average of the cost function to
        # monitor the convergence while discarding minibatch-local stochastic
        # variability: https://en.wikipedia.org/wiki/Moving_average
        if self._ewa_cost is None:
            self._ewa_cost = batch_cost  # 初始化指数加权平均成本
        else:
            alpha = batch_size / (n_samples + 1)
            alpha = min(alpha, 1)
            self._ewa_cost = self._ewa_cost * (1 - alpha) + batch_cost * alpha  # 更新指数加权平均成本

        if self.verbose:
            print(
                f"Minibatch step {step}/{n_steps}: mean batch cost: "
                f"{batch_cost}, ewa cost: {self._ewa_cost}"
            )

        # Early stopping based on change of dictionary
        dict_diff = linalg.norm(new_dict - old_dict) / self._n_components  # 计算字典变化的范数差
        if self.tol > 0 and dict_diff <= self.tol:  # 如果容忍度大于0且字典变化小于等于容忍度，触发早停
            if self.verbose:
                print(f"Converged (small dictionary change) at step {step}/{n_steps}")
            return True  # 返回True，表示停止训练

        # Early stopping heuristic due to lack of improvement on smoothed
        # cost function
        if self._ewa_cost_min is None or self._ewa_cost < self._ewa_cost_min:
            self._no_improvement = 0
            self._ewa_cost_min = self._ewa_cost
        else:
            self._no_improvement += 1

        if (
            self.max_no_improvement is not None
            and self._no_improvement >= self.max_no_improvement
        ):
            if self.verbose:
                print(
                    "Converged (lack of improvement in objective function) "
                    f"at step {step}/{n_steps}"
                )
            return True  # 返回True，表示停止训练

        return False  # 默认返回False，表示不停止训练

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Validate the input data X to ensure it's of the correct type and format
        X = self._validate_data(
            X, dtype=[np.float64, np.float32], order="C", copy=False
        )

        # Check and set internal parameters based on the input data X
        self._check_params(X)
        # Initialize random state using either provided or default random state
        self._random_state = check_random_state(self.random_state)

        # Initialize the dictionary for dictionary learning using input data and random state
        dictionary = self._initialize_dict(X, self._random_state)
        # Create a copy of the initial dictionary for comparison during iterations
        old_dict = dictionary.copy()

        # Shuffle training data if specified
        if self.shuffle:
            X_train = X.copy()
            self._random_state.shuffle(X_train)
        else:
            X_train = X

        # Extract number of samples and features from the training data
        n_samples, n_features = X_train.shape

        # Print verbose message if verbose mode is enabled
        if self.verbose:
            print("[dict_learning]")

        # Initialize matrices _A and _B with zeros based on component sizes and feature count
        self._A = np.zeros(
            (self._n_components, self._n_components), dtype=X_train.dtype
        )
        self._B = np.zeros((n_features, self._n_components), dtype=X_train.dtype)

        # Deprecated warning about `max_iter=None` and set default max_iter if not specified
        if self.max_iter is None:
            warn(
                (
                    "`max_iter=None` is deprecated in version 1.4 and will be removed"
                    " in version 1.6. Use the default value (i.e. `1_000`) instead."
                ),
                FutureWarning,
            )
            max_iter = 1_000
        else:
            max_iter = self.max_iter

        # Initialize attributes to monitor convergence during iterations
        self._ewa_cost = None
        self._ewa_cost_min = None
        self._no_improvement = 0

        # Generate batches of indices for minibatch processing
        batches = gen_batches(n_samples, self._batch_size)
        # Cycle through batches indefinitely to cover max_iter iterations
        batches = itertools.cycle(batches)
        # Calculate number of steps per iteration and total number of steps
        n_steps_per_iter = int(np.ceil(n_samples / self._batch_size))
        n_steps = max_iter * n_steps_per_iter

        i = -1  # Initialize i to allow max_iter = 0

        # Iterate through batches and perform minibatch steps
        for i, batch in zip(range(n_steps), batches):
            X_batch = X_train[batch]

            # Perform a minibatch step to update dictionary and calculate batch cost
            batch_cost = self._minibatch_step(
                X_batch, dictionary, self._random_state, i
            )

            # Check for convergence using current batch information and dictionary
            if self._check_convergence(
                X_batch, batch_cost, dictionary, old_dict, n_samples, i, n_steps
            ):
                break

            # Invoke callback function if provided
            if self.callback is not None:
                self.callback(locals())

            # Update old_dict to track changes in dictionary across iterations
            old_dict[:] = dictionary

        # Calculate number of steps performed and iterations completed
        self.n_steps_ = i + 1
        self.n_iter_ = np.ceil(self.n_steps_ / n_steps_per_iter)
        # Set components_ attribute to the final dictionary learned
        self.components_ = dictionary

        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None):
        """
        Update the model using the data in X as a mini-batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        # 检查模型是否已经有 components_
        has_components = hasattr(self, "components_")

        # 验证输入数据 X 的格式和类型，并根据是否有 components_ 来重置验证
        X = self._validate_data(
            X, dtype=[np.float64, np.float32], order="C", reset=not has_components
        )

        if not has_components:
            # 如果模型尚未拟合过（即未调用过 fit 或 partial_fit）
            self._check_params(X)  # 检查参数设置
            self._random_state = check_random_state(self.random_state)  # 设置随机状态

            # 初始化字典对象用于存储模型组件
            dictionary = self._initialize_dict(X, self._random_state)

            # 初始化步数计数器
            self.n_steps_ = 0

            # 初始化 A 和 B 矩阵
            self._A = np.zeros((self._n_components, self._n_components), dtype=X.dtype)
            self._B = np.zeros((X.shape[1], self._n_components), dtype=X.dtype)
        else:
            # 如果模型已经拟合过，直接使用已有的 components_
            dictionary = self.components_

        # 执行迷你批次更新步骤
        self._minibatch_step(X, dictionary, self._random_state, self.n_steps_)

        # 更新模型的 components_
        self.components_ = dictionary
        self.n_steps_ += 1

        return self

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        # 返回模型 components_ 的行数，即转换后的输出特征数量
        return self.components_.shape[0]

    def _more_tags(self):
        # 返回一个字典，指定了额外的标签，这里是指保留数据类型的标签
        return {
            "preserves_dtype": [np.float64, np.float32],
        }
```
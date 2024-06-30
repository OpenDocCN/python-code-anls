# `D:\src\scipysrc\scipy\scipy\cluster\_vq.pyx`

```
"""
Cython rewrite of the vector quantization module, originally written
in C at src/vq.c and the wrapper at src/vq_module.c. This should be
easier to maintain than old SWIG output.

Original C version by Damian Eads.
Translated to Cython by David Warde-Farley, October 2009.
"""

# 导入必要的 Cython 模块
cimport cython
# 导入 NumPy 库，并声明相应的 C 接口
import numpy as np
cimport numpy as np
# 导入 Cython 版本的 BLAS 函数
from scipy.linalg.cython_blas cimport dgemm, sgemm

# 导入 C 标准库中的数学函数
from libc.math cimport sqrt

# 定义 NumPy 中所需的数据类型别名
ctypedef np.float64_t float64_t
ctypedef np.float32_t float32_t
ctypedef np.int32_t int32_t

# 使用 Cython 的 fused types 定义模板化的 vq_type
ctypedef fused vq_type:
    float32_t
    float64_t

# 初始化 NumPy C API，确保 NumPy 数组的正确使用
np.import_array()

# 定义内联函数 vec_sqr，计算向量的平方和
cdef inline vq_type vec_sqr(int n, vq_type *p) noexcept:
    cdef vq_type result = 0.0
    cdef int i
    for i in range(n):
        result += p[i] * p[i]
    return result

# 定义内联函数 cal_M，计算 M = obs * code_book.T
cdef inline void cal_M(int nobs, int ncodes, int nfeat, vq_type *obs,
                       vq_type *code_book, vq_type *M) noexcept:
    """
    Calculate M = obs * code_book.T
    """
    cdef vq_type alpha = -2.0, beta = 0.0

    # 调用 BLAS 函数，使用 Fortran ABI，按列主序处理数据
    if vq_type is float32_t:
        sgemm("T", "N", &ncodes, &nobs, &nfeat,
               &alpha, code_book, &nfeat, obs, &nfeat, &beta, M, &ncodes)
    else:
        dgemm("T", "N", &ncodes, &nobs, &nfeat,
              &alpha, code_book, &nfeat, obs, &nfeat, &beta, M, &ncodes)

# 定义 _vq 函数，实现向量量化操作
cdef int _vq(vq_type *obs, vq_type *code_book,
             int ncodes, int nfeat, int nobs,
             int32_t *codes, vq_type *low_dist) except -1:
    """
    The underlying function (template) of _vq.vq.

    Parameters
    ----------
    obs : vq_type*
        The pointer to the observation matrix.
    code_book : vq_type*
        The pointer to the code book matrix.
    ncodes : int
        The number of centroids (codes).
    nfeat : int
        The number of features of each observation.
    nobs : int
        The number of observations.
    codes : int32_t*
        The pointer to the new codes array.
    low_dist : vq_type*
        low_dist[i] is the Euclidean distance from obs[i] to the corresponding
        centroid.
    """
    # 当特征数量少于 5 时，切换回简单算法以避免高开销
    if nfeat < 5:
        _vq_small_nf(obs, code_book, ncodes, nfeat, nobs, codes, low_dist)
        return 0

    cdef np.npy_intp i, j
    cdef vq_type *p_obs
    cdef vq_type *p_codes
    cdef vq_type dist_sqr
    cdef np.ndarray[vq_type, ndim=1] obs_sqr, codes_sqr
    cdef np.ndarray[vq_type, ndim=2] M

    if vq_type is float32_t:
        # 分配内存给观测值的平方和和码本的平方和
        obs_sqr = np.ndarray(nobs, np.float32)
        codes_sqr = np.ndarray(ncodes, np.float32)
        # 分配内存给 M 矩阵
        M = np.ndarray((nobs, ncodes), np.float32)
    else:
        # 创建具有 nobs 个元素的浮点型 ndarray，用于存储每个观测的平方和
        obs_sqr = np.ndarray(nobs, np.float64)
        # 创建具有 ncodes 个元素的浮点型 ndarray，用于存储每个码本代码的平方和
        codes_sqr = np.ndarray(ncodes, np.float64)
        # 创建一个 nobs x ncodes 大小的二维浮点型 ndarray，用于存储观测与码本代码之间的内积
        M = np.ndarray((nobs, ncodes), np.float64)
    
    p_obs = obs
    for i in range(nobs):
        # obs_sqr[i] 是第 i 个观测向量与自身的内积
        obs_sqr[i] = vec_sqr(nfeat, p_obs)
        p_obs += nfeat
    
    p_codes = code_book
    for i in range(ncodes):
        # codes_sqr[i] 是第 i 个码本代码向量与自身的内积
        codes_sqr[i] = vec_sqr(nfeat, p_codes)
        p_codes += nfeat
    
    # 计算 M 矩阵，其中 M[i][j] 是第 i 个观测向量与第 j 个码本代码向量的内积
    # 这里调用了 cal_M 函数，传入观测数目 nobs、码本代码数目 ncodes、特征数目 nfeat、观测数组 obs、码本代码数组 code_book、以及 M 矩阵数据的指针
    cal_M(nobs, ncodes, nfeat, obs, code_book, <vq_type *>M.data)
    
    for i in range(nobs):
        for j in range(ncodes):
            # 计算观测向量 obs[i] 与码本代码 code_book[j] 的距离的平方
            dist_sqr = (M[i, j] +
                        obs_sqr[i] + codes_sqr[j])
            # 如果计算出的距离平方小于 low_dist[i]，更新对应的最低距离和对应的码本代码索引
            if dist_sqr < low_dist[i]:
                codes[i] = j
                low_dist[i] = dist_sqr
    
        # 由于浮点数计算可能存在精度问题，这里检查 low_dist[i] 是否大于 0，若大于则将其开方，否则设置为 0
        if low_dist[i] > 0:
            low_dist[i] = sqrt(low_dist[i])
        else:
            low_dist[i] = 0
    
    # 函数返回 0，表示执行成功
    return 0
cdef void _vq_small_nf(vq_type *obs, vq_type *code_book,
                       int ncodes, int nfeat, int nobs,
                       int32_t *codes, vq_type *low_dist) noexcept:
    """
    Vector quantization using naive algorithm.
    This is preferred when nfeat is small.
    The parameters are the same as those of _vq.
    """
    # Temporary variables
    cdef vq_type dist_sqr, diff  # 定义距离平方和差异变量
    cdef np.npy_intp i, j, k, obs_offset = 0, code_offset  # 定义循环变量和偏移量

    # Index and pointer to keep track of the current position in
    # both arrays so that we don't have to always do index * nfeat.
    cdef vq_type *current_obs  # 当前观测值指针
    cdef vq_type *current_code  # 当前码书值指针

    for i in range(nobs):  # 遍历所有观测值
        code_offset = 0  # 初始化码书偏移量
        current_obs = &(obs[obs_offset])  # 当前观测值指针初始化

        for j in range(ncodes):  # 遍历所有码书
            dist_sqr = 0  # 初始化距离平方和
            current_code = &(code_book[code_offset])  # 当前码书值指针初始化

            # Distance between code_book[j] and obs[i]
            for k in range(nfeat):  # 遍历所有特征
                diff = current_code[k] - current_obs[k]  # 计算当前特征的差异
                dist_sqr += diff * diff  # 计算距离平方和
            code_offset += nfeat  # 更新码书偏移量

            # Replace the code assignment and record distance if necessary
            if dist_sqr < low_dist[i]:  # 如果距离平方和小于当前最小距离
                codes[i] = j  # 更新码书分配
                low_dist[i] = dist_sqr  # 记录最小距离平方和

        low_dist[i] = sqrt(low_dist[i])  # 更新最小距离为其平方根值

        # Update the offset of the current observation
        obs_offset += nfeat  # 更新当前观测值偏移量


def vq(np.ndarray obs, np.ndarray codes):
    """
    Vector quantization ndarray wrapper. Only support float32 and float64.

    Parameters
    ----------
    obs : ndarray
        The observation matrix. Each row is an observation.
    codes : ndarray
        The code book matrix.

    Notes
    -----
    The observation matrix and code book matrix should have same ndim and
    same number of columns (features). Only 1-dimensional and 2-dimensional
    arrays are supported.
    """
    cdef int nobs, ncodes, nfeat  # 定义观测值、码书数目和特征数
    cdef np.ndarray outcodes, outdists  # 定义输出码书和距离数组

    # Ensure the arrays are contiguous
    obs = np.ascontiguousarray(obs)  # 确保观测值数组是连续的
    codes = np.ascontiguousarray(codes)  # 确保码书数组是连续的

    if obs.dtype != codes.dtype:
        raise TypeError('observation and code should have same dtype')  # 若观测值和码书数据类型不同，抛出类型错误异常
    if obs.dtype not in (np.float32, np.float64):
        raise TypeError('type other than float or double not supported')  # 若观测值和码书数据类型不是浮点型，抛出类型错误异常
    if obs.ndim != codes.ndim:
        raise ValueError(
            'observation and code should have same number of dimensions')  # 若观测值和码书数组维度不同，抛出值错误异常

    if obs.ndim == 1:  # 如果观测值数组是一维的
        nfeat = 1  # 特征数为1
        nobs = obs.shape[0]  # 观测值数目为观测值数组的长度
        ncodes = codes.shape[0]  # 码书数目为码书数组的长度
    elif obs.ndim == 2:  # 如果观测值数组是二维的
        nfeat = obs.shape[1]  # 特征数为观测值数组的列数
        nobs = obs.shape[0]  # 观测值数目为观测值数组的行数
        ncodes = codes.shape[0]  # 码书数目为码书数组的行数
        if nfeat != codes.shape[1]:
            raise ValueError('obs and code should have same number of '
                             'features (columns)')  # 若观测值和码书数组的特征数不同，抛出值错误异常
    else:
        raise ValueError('ndim different than 1 or 2 are not supported')  # 若观测值和码书数组的维度不是1或2，抛出值错误异常

    # Initialize outdists and outcodes array.
    # Outdists should be initialized as INF.
    outdists = np.full((nobs,), np.inf, dtype=np.float64)  # 初始化距离数组为无穷大
    # 创建一个空的 NumPy 数组，用于存储每个观测值的距离
    outdists = np.empty((nobs,), dtype=obs.dtype)
    # 创建一个空的 NumPy 数组，用于存储每个观测值的分类码（整数）
    outcodes = np.empty((nobs,), dtype=np.int32)
    # 将距离数组填充为无穷大，以确保每个距离初始值足够大
    outdists.fill(np.inf)
    
    # 检查观测值数组的数据类型是否为 np.float32
    if obs.dtype.type is np.float32:
        # 调用 C 函数 _vq 进行向量量化操作，处理 np.float32 数据类型
        _vq(<float32_t *>obs.data, <float32_t *>codes.data,
            ncodes, nfeat, nobs, <int32_t *>outcodes.data,
            <float32_t *>outdists.data)
    # 检查观测值数组的数据类型是否为 np.float64
    elif obs.dtype.type is np.float64:
        # 调用 C 函数 _vq 进行向量量化操作，处理 np.float64 数据类型
        _vq(<float64_t *>obs.data, <float64_t *>codes.data,
            ncodes, nfeat, nobs, <int32_t *>outcodes.data,
            <float64_t *>outdists.data)
    
    # 返回处理后的分类码数组和距离数组
    return outcodes, outdists
@cython.cdivision(True)
# 启用 Cython 中的 C 除法支持

cdef np.ndarray _update_cluster_means(vq_type *obs, int32_t *labels,
                                      vq_type *cb, int nobs, int nc, int nfeat):
    """
    The underlying function (template) of _vq.update_cluster_means.

    Parameters
    ----------
    obs : vq_type*
        The pointer to the observation matrix.
    labels : int32_t*
        The pointer to the array of the labels (codes) of the observations.
    cb : vq_type*
        The pointer to the new code book matrix.
    nobs : int
        The number of observations.
    nc : int
        The number of centroids (codes).
    nfeat : int
        The number of features of each observation.

    Returns
    -------
    has_members : ndarray
        A boolean array indicating which clusters have members.
    """
    cdef np.npy_intp i, j, cluster_size, label
    # 定义整数变量和指针变量
    cdef vq_type *obs_p
    # 指向观测矩阵的指针变量
    cdef vq_type *cb_p
    # 指向代码簿矩阵的指针变量
    cdef np.ndarray[int, ndim=1] obs_count
    # 用于存储每个簇中观测数量的数组

    # Calculate the sums the numbers of obs in each cluster
    # 计算每个簇中观测数的总和
    obs_count = np.zeros(nc, np.intc)
    # 初始化一个全零数组，用于统计每个簇的观测数量
    obs_p = obs
    # 将观测矩阵的指针赋给obs_p
    for i in range(nobs):
        # 遍历每个观测
        label = labels[i]
        # 获取当前观测的标签（簇编号）
        cb_p = cb + nfeat * label
        # 将代码簿的指针移动到当前标签（簇）的起始位置

        for j in range(nfeat):
            # 遍历每个特征
            cb_p[j] += obs_p[j]
            # 将当前观测的特征值加到对应簇的代码簿中

        # Count the obs in each cluster
        # 统计每个簇中的观测数
        obs_count[label] += 1
        # 对应簇的观测数加一
        obs_p += nfeat
        # 将观测指针移动到下一个观测的起始位置

    cb_p = cb
    # 将代码簿指针重置为起始位置
    for i in range(nc):
        # 遍历每个簇
        cluster_size = obs_count[i]
        # 获取当前簇的观测数

        if cluster_size > 0:
            # 如果当前簇有观测点
            for j in range(nfeat):
                cb_p[j] /= cluster_size
                # 计算每个特征的簇中心

        cb_p += nfeat
        # 将代码簿指针移动到下一个簇的起始位置

    # Return a boolean array indicating which clusters have members
    # 返回一个布尔数组，指示哪些簇包含有观测点
    return obs_count > 0


def update_cluster_means(np.ndarray obs, np.ndarray labels, int nc):
    """
    The update-step of K-means. Calculate the mean of observations in each
    cluster.

    Parameters
    ----------
    obs : ndarray
        The observation matrix. Each row is an observation. Its dtype must be
        float32 or float64.
    labels : ndarray
        The label of each observation. Must be an 1d array.
    nc : int
        The number of centroids.

    Returns
    -------
    cb : ndarray
        The new code book.
    has_members : ndarray
        A boolean array indicating which clusters have members.

    Notes
    -----
    The empty clusters will be set to all zeros and the corresponding elements
    in `has_members` will be `False`. The upper level function should decide
    how to deal with them.
    """
    cdef np.ndarray has_members, cb
    # 定义返回结果的数组变量
    cdef int nfeat

    # Ensure the arrays are contiguous
    # 确保数组是连续的
    obs = np.ascontiguousarray(obs)
    # 将观测矩阵转换为连续的数组
    labels = np.ascontiguousarray(labels)
    # 将标签数组转换为连续的数组

    if obs.dtype not in (np.float32, np.float64):
        # 如果观测矩阵的数据类型不是浮点型
        raise TypeError('type other than float or double not supported')
        # 抛出类型错误异常
    if labels.dtype.type is not np.int32:
        # 如果标签数组的数据类型不是32位整型
        labels = labels.astype(np.int32)
        # 将标签数组转换为32位整型
    if labels.ndim != 1:
        # 如果标签数组不是一维数组
        raise ValueError('labels must be an 1d array')
        # 抛出值错误异常
    # 如果观测数据的维度为1维
    if obs.ndim == 1:
        # 特征数为1
        nfeat = 1
        # 创建一个全零数组作为聚类中心的初始值，数据类型与观测数据相同
        cb = np.zeros(nc, dtype=obs.dtype)
    # 如果观测数据的维度为2维
    elif obs.ndim == 2:
        # 特征数为第二维的长度
        nfeat = obs.shape[1]
        # 创建一个二维全零数组作为聚类中心的初始值，形状为(nc, nfeat)，数据类型与观测数据相同
        cb = np.zeros((nc, nfeat), dtype=obs.dtype)
    else:
        # 如果观测数据的维度不是1或2，则抛出值错误异常
        raise ValueError('ndim different than 1 or 2 are not supported')

    # 如果观测数据的数据类型为 np.float32
    if obs.dtype.type is np.float32:
        # 调用 C 函数 _update_cluster_means 更新聚类中心的均值
        has_members = _update_cluster_means(<float32_t *>obs.data,
                                            <int32_t *>labels.data,
                                            <float32_t *>cb.data,
                                            obs.shape[0], nc, nfeat)
    # 如果观测数据的数据类型为 np.float64
    elif obs.dtype.type is np.float64:
        # 调用 C 函数 _update_cluster_means 更新聚类中心的均值
        has_members = _update_cluster_means(<float64_t *>obs.data,
                                            <int32_t *>labels.data,
                                            <float64_t *>cb.data,
                                            obs.shape[0], nc, nfeat)

    # 返回更新后的聚类中心 cb 和是否有成员属于各聚类的标记 has_members
    return cb, has_members
```
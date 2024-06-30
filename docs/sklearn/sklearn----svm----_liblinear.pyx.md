# `D:\src\scipysrc\scikit-learn\sklearn\svm\_liblinear.pyx`

```
"""
Wrapper for liblinear

Author: fabian.pedregosa@inria.fr
"""

import numpy as np  # 导入 NumPy 库

from ..utils._cython_blas cimport _dot, _axpy, _scal, _nrm2  # 导入 Cython BLAS 函数
from ..utils._typedefs cimport float32_t, float64_t, int32_t  # 导入 Cython 定义的数据类型

include "_liblinear.pxi"  # 包含 Cython 的 _liblinear.pxi 文件


def train_wrap(
    object X,  # 输入数据 X
    const float64_t[::1] Y,  # 类标签 Y，使用 float64_t 类型
    bint is_sparse,  # 是否稀疏数据的标志
    int solver_type,  # 解算器类型
    double eps,  # 精度控制参数 epsilon
    double bias,  # 偏置项
    double C,  # 正则化参数 C
    const float64_t[:] class_weight,  # 类别权重
    int max_iter,  # 最大迭代次数
    unsigned random_seed,  # 随机种子
    double epsilon,  # 误差容忍度
    const float64_t[::1] sample_weight  # 样本权重
):
    cdef parameter *param  # 定义参数对象指针
    cdef problem *problem  # 定义问题对象指针
    cdef model *model  # 定义模型对象指针
    cdef char_const_ptr error_msg  # 错误消息字符串指针
    cdef int len_w  # 权重长度
    cdef bint X_has_type_float64 = X.dtype == np.float64  # 判断 X 是否为 float64 类型
    cdef char * X_data_bytes_ptr  # 数据字节指针
    cdef const float64_t[::1] X_data_64  # X 数据的 float64 数组
    cdef const float32_t[::1] X_data_32  # X 数据的 float32 数组
    cdef const int32_t[::1] X_indices  # 稀疏数据的索引数组
    cdef const int32_t[::1] X_indptr  # 稀疏数据的指针数组

    if is_sparse:  # 如果数据是稀疏的
        X_indices = X.indices  # 获取稀疏数据的索引
        X_indptr = X.indptr  # 获取稀疏数据的指针
        if X_has_type_float64:  # 如果稀疏数据是 float64 类型
            X_data_64 = X.data  # 获取稀疏数据的 float64 数组
            X_data_bytes_ptr = <char *> &X_data_64[0]  # 获取稀疏数据的字节指针
        else:
            X_data_32 = X.data  # 获取稀疏数据的 float32 数组
            X_data_bytes_ptr = <char *> &X_data_32[0]  # 获取稀疏数据的字节指针

        # 设置 CSR 格式问题
        problem = csr_set_problem(
            X_data_bytes_ptr,
            X_has_type_float64,
            <char *> &X_indices[0],
            <char *> &X_indptr[0],
            (<int32_t>X.shape[0]),
            (<int32_t>X.shape[1]),
            (<int32_t>X.nnz),
            bias,
            <char *> &sample_weight[0],
            <char *> &Y[0]
        )
    else:  # 如果数据不是稀疏的
        X_as_1d_array = X.reshape(-1)  # 将数据 X 转换为一维数组
        if X_has_type_float64:  # 如果数据是 float64 类型
            X_data_64 = X_as_1d_array  # 获取数据的 float64 数组
            X_data_bytes_ptr = <char *> &X_data_64[0]  # 获取数据的字节指针
        else:
            X_data_32 = X_as_1d_array  # 获取数据的 float32 数组
            X_data_bytes_ptr = <char *> &X_data_32[0]  # 获取数据的字节指针

        # 设置一般问题
        problem = set_problem(
            X_data_bytes_ptr,
            X_has_type_float64,
            (<int32_t>X.shape[0]),
            (<int32_t>X.shape[1]),
            (<int32_t>np.count_nonzero(X)),
            bias,
            <char *> &sample_weight[0],
            <char *> &Y[0]
        )

    # 创建类别权重标签数组
    cdef int32_t[::1] class_weight_label = np.arange(class_weight.shape[0], dtype=np.intc)

    # 设置参数对象
    param = set_parameter(
        solver_type,
        eps,
        C,
        class_weight.shape[0],
        <char *> &class_weight_label[0] if class_weight_label.size > 0 else NULL,
        <char *> &class_weight[0] if class_weight.size > 0 else NULL,
        max_iter,
        random_seed,
        epsilon
    )

    # 检查参数设置是否正确
    error_msg = check_parameter(problem, param)
    if error_msg:  # 如果有错误消息，则抛出 ValueError 异常
        free_problem(problem)
        free_parameter(param)
        raise ValueError(error_msg)

    cdef BlasFunctions blas_functions  # 定义 BLAS 函数结构体
    blas_functions.dot = _dot[double]
    blas_functions.axpy = _axpy[double]
    blas_functions.scal = _scal[double]
    blas_functions.nrm2 = _nrm2[double]

    # 使用 nogil 上下文进行无全局解释器锁的训练操作
    with nogil:
        model = train(problem, param, &blas_functions)  # 训练模型
    # 释放问题及其参数
    free_problem(problem)
    free_parameter(param)
    # 不要调用 destroy_param(param)，否则会破坏 class_weight_label 和 class_weight

    # 创建存储 coef 矩阵的变量，使用 Fortran 风格，因为这是 liblinear 中使用的风格
    cdef float64_t[::1, :] w
    # 获取模型的类别数目
    cdef int nr_class = get_nr_class(model)

    # 设置标签数量为 nr_class
    cdef int labels_ = nr_class
    # 如果类别数为 2，则将 labels_ 设置为 1
    if nr_class == 2:
        labels_ = 1
    # 创建用于存储迭代次数的数组，长度为 labels_，数据类型为 int32_t
    cdef int32_t[::1] n_iter = np.zeros(labels_, dtype=np.intc)
    # 获取模型的迭代次数，并存储到 n_iter 中
    get_n_iter(model, <int *> &n_iter[0])

    # 获取模型的特征数量
    cdef int nr_feature = get_nr_feature(model)
    # 如果有偏置，则特征数量加一
    if bias > 0:
        nr_feature = nr_feature + 1
    # 如果类别数为 2 并且 solver_type 不等于 4（即 solver 不是 Crammer-Singer）
    if nr_class == 2 and solver_type != 4:
        # 分配一个 1xnr_feature 的空数组 w，按 Fortran 顺序存储
        w = np.empty((1, nr_feature), order='F')
        # 将模型的权重复制到 w 中
        copy_w(&w[0, 0], model, nr_feature)
    else:
        # 计算权重数组 w 的长度，为 nr_class * nr_feature
        len_w = (nr_class) * nr_feature
        # 分配一个 nr_class x nr_feature 的空数组 w，按 Fortran 顺序存储
        w = np.empty((nr_class, nr_feature), order='F')
        # 将模型的权重复制到 w 中
        copy_w(&w[0, 0], model, len_w)

    # 释放并销毁模型
    free_and_destroy_model(&model)

    # 返回权重数组 w 的基础对象和迭代次数数组 n_iter 的基础对象
    return w.base, n_iter.base
# 定义一个函数 `set_verbosity_wrap`，用于设置 libsvm 库的详细程度
def set_verbosity_wrap(int verbosity):
    # 调用 libsvm 库的函数 `set_verbosity`，将传入的详细程度参数作为参数传递给它
    set_verbosity(verbosity)
```
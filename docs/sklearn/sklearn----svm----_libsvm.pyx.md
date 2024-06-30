# `D:\src\scipysrc\scikit-learn\sklearn\svm\_libsvm.pyx`

```
"""
Binding for libsvm_skl
----------------------

These are the bindings for libsvm_skl, which is a fork of libsvm[1]
that adds to libsvm some capabilities, like index of support vectors
and efficient representation of dense matrices.

These are low-level routines, but can be used for flexibility or
performance reasons. See sklearn.svm for a higher-level API.

Low-level memory management is done in libsvm_helper.c. If we happen
to run out of memory a MemoryError will be raised. In practice this is
not very helpful since high chances are malloc fails inside svm.cpp,
where no sort of memory checks are done.

[1] https://www.csie.ntu.edu.tw/~cjlin/libsvm/

Notes
-----
The signature mode='c' is somewhat superficial, since we already
check that arrays are C-contiguous in svm.py

Authors
-------
2010: Fabian Pedregosa <fabian.pedregosa@inria.fr>
      Gael Varoquaux <gael.varoquaux@normalesup.org>
"""

import numpy as np  # 导入NumPy库
from libc.stdlib cimport free  # 导入C标准库中的free函数
from ..utils._cython_blas cimport _dot  # 导入Cython模块中的_dot函数
from ..utils._typedefs cimport float64_t, int32_t, intp_t  # 导入Cython模块中的类型定义

include "_libsvm.pxi"  # 导入libsvm的Cython接口文件

cdef extern from *:
    ctypedef struct svm_parameter:  # 定义Cython中用到的svm_parameter结构体
        pass


################################################################################
# Internal variables
LIBSVM_KERNEL_TYPES = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']  # 定义支持的核函数类型列表


################################################################################
# Wrapper functions

def fit(
    const float64_t[:, ::1] X,  # 输入数据X，二维数组，类型为float64_t
    const float64_t[::1] Y,  # 目标数据Y，一维数组，类型为float64_t
    int svm_type=0,  # SVM类型，默认为0，对应C_SVC
    kernel='rbf',  # 核函数类型，默认为'rbf'（径向基函数）
    int degree=3,  # 多项式核函数的次数，默认为3
    double gamma=0.1,  # RBF、多项式和sigmoid核函数的参数gamma，默认为0.1
    double coef0=0.0,  # 多项式和sigmoid核函数的参数coef0，默认为0.0
    double tol=1e-3,  # 数值停止标准，默认为1e-3
    double C=1.0,  # C-SVM中的参数C，默认为1.0
    double nu=0.5,  # Nu-SVM中的参数nu，默认为0.5
    double epsilon=0.1,  # epsilon-SVR和nu-SVR中的参数epsilon，默认为0.1
    const float64_t[::1] class_weight=np.empty(0),  # 类别权重，默认为空数组
    const float64_t[::1] sample_weight=np.empty(0),  # 样本权重，默认为空数组
    int shrinking=1,  # 是否使用收缩启发式，默认为1（使用）
    int probability=0,  # 是否启用概率估计，默认为0（不启用）
    double cache_size=100.,  # 缓存大小，默认为100.0
    int max_iter=-1,  # 最大迭代次数，默认为-1（无限制）
    int random_seed=0,  # 随机种子，默认为0
):
    """
    Train the model using libsvm (low-level method)

    Parameters
    ----------
    X : array-like, dtype=float64 of shape (n_samples, n_features)
        Input data.

    Y : array, dtype=float64 of shape (n_samples,)
        Target vector.

    svm_type : {0, 1, 2, 3, 4}, default=0
        Type of SVM: C_SVC, NuSVC, OneClassSVM, EpsilonSVR or NuSVR
        respectively.

    kernel : {'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'}, default="rbf"
        Kernel to use in the model: linear, polynomial, RBF, sigmoid
        or precomputed.

    degree : int32, default=3
        Degree of the polynomial kernel (only relevant if kernel is
        set to polynomial).

    gamma : float64, default=0.1
        Gamma parameter in rbf, poly and sigmoid kernels. Ignored by other
        kernels.

    coef0 : float64, default=0
        Independent parameter in poly/sigmoid kernel.

    tol : float64, default=1e-3
        Numeric stopping criterion.

    C : float64, default=1
        C parameter in C-Support Vector Classification.
    """
    # 定义 SVM 模型的参数对象
    cdef svm_parameter param
    # 定义 SVM 的问题对象
    cdef svm_problem problem
    # 定义指向 SVM 模型的指针
    cdef svm_model *model
    # 错误消息的常量字符指针
    cdef const char *error_msg
    # 支持向量的长度
    cdef intp_t SV_len

    # 如果样本权重数组为空，则将其初始化为全为1的数组，长度与样本特征 X 的行数相同
    if len(sample_weight) == 0:
        sample_weight = np.ones(X.shape[0], dtype=np.float64)
    else:
        # 否则，确保样本权重数组与特征 X 的行数相同，否则抛出异常
        assert sample_weight.shape[0] == X.shape[0], (
            f"sample_weight and X have incompatible shapes: sample_weight has "
            f"{sample_weight.shape[0]} samples while X has {X.shape[0]}"
        )

    # 根据指定的 kernel 字符串确定其索引
    kernel_index = LIBSVM_KERNEL_TYPES.index(kernel)
    # 设置 SVM 的问题对象，传入特征矩阵 X、标签数组 Y、样本权重数组 sample_weight、以及 X 的形状信息和 kernel 索引
    set_problem(
        &problem,
        <char*> &X[0, 0],
        <char*> &Y[0],
        <char*> &sample_weight[0],
        <intp_t*> X.shape,
        kernel_index,
    )
    # 如果问题对象的 x 属性为空指针，则抛出内存错误异常
    if problem.x == NULL:
        raise MemoryError("Seems we've run out of memory")

    # 创建一个 int32 的数组，其长度为 class_weight 数组的长度，用于类别权重的标签
    cdef int32_t[::1] class_weight_label = np.arange(
        class_weight.shape[0], dtype=np.int32
    )
    # 调用函数设置参数
    set_parameter(
        &param,  # 参数结构体的指针
        svm_type,  # SVM 类型
        kernel_index,  # 核函数类型的索引
        degree,  # 多项式核函数的阶数
        gamma,  # 核函数的 gamma 参数
        coef0,  # 核函数的常数项参数
        nu,  # SVM 中的 nu 参数
        cache_size,  # 缓存大小
        C,  # SVM 中的惩罚参数 C
        tol,  # 容忍度
        epsilon,  # SVR 中的 epsilon 参数
        shrinking,  # 是否使用启发式方法
        probability,  # 是否进行概率估计
        <int> class_weight.shape[0],  # 类别权重的数量
        <char*> &class_weight_label[0] if class_weight_label.size > 0 else NULL,  # 类别权重标签的指针
        <char*> &class_weight[0] if class_weight.size > 0 else NULL,  # 类别权重的指针
        max_iter,  # 最大迭代次数
        random_seed,  # 随机种子
    )

    # 检查 SVM 参数是否有效，如果不是，进行错误处理
    error_msg = svm_check_parameter(&problem, &param)
    if error_msg:
        # 将错误消息解码为 UTF-8 格式并替换其中的部分内容
        error_repl = error_msg.decode('utf-8').replace("p < 0", "epsilon < 0")
        raise ValueError(error_repl)

    # 定义 BLAS 函数结构体，并设置 dot 函数为双精度版本
    cdef BlasFunctions blas_functions
    blas_functions.dot = _dot[double]

    # 使用 nogil 上下文执行实际的训练过程
    cdef int fit_status = 0
    with nogil:
        # 调用 SVM 训练函数，返回模型
        model = svm_train(&problem, &param, &fit_status, &blas_functions)

    # 以下部分开始从 svm_train 返回的数据中复制数据

    # 获取支持向量的数量
    SV_len = get_l(model)
    # 获取类别的数量
    n_class = get_nr(model)

    # 初始化存储迭代次数的数组
    cdef int[::1] n_iter = np.empty(max(1, n_class * (n_class - 1) // 2), dtype=np.intc)
    copy_n_iter(<char*> &n_iter[0], model)

    # 初始化存储支持向量系数的数组
    cdef float64_t[:, ::1] sv_coef = np.empty((n_class-1, SV_len), dtype=np.float64)
    copy_sv_coef(<char*> &sv_coef[0, 0] if sv_coef.size > 0 else NULL, model)

    # 初始化存储截距的数组，使用 rho 并改变符号
    cdef float64_t[::1] intercept = np.empty(
        int((n_class*(n_class-1))/2), dtype=np.float64
    )
    copy_intercept(<char*> &intercept[0], model, <intp_t*> intercept.shape)

    # 初始化存储支持向量索引的数组
    cdef int32_t[::1] support = np.empty(SV_len, dtype=np.int32)
    copy_support(<char*> &support[0] if support.size > 0 else NULL, model)

    # 初始化存储支持向量的数组
    cdef float64_t[:, ::1] support_vectors
    if kernel_index == 4:
        # 对于预先计算的核，支持向量为空数组
        support_vectors = np.empty((0, 0), dtype=np.float64)
    else:
        # 否则初始化为适当大小的数组
        support_vectors = np.empty((SV_len, X.shape[1]), dtype=np.float64)
        copy_SV(
            <char*> &support_vectors[0, 0] if support_vectors.size > 0 else NULL,
            model,
            <intp_t*> support_vectors.shape,
        )

    # 初始化存储每个类别支持向量数量的数组
    cdef int32_t[::1] n_class_SV
    if svm_type == 0 or svm_type == 1:
        n_class_SV = np.empty(n_class, dtype=np.int32)
        copy_nSV(<char*> &n_class_SV[0] if n_class_SV.size > 0 else NULL, model)
    else:
        # 对于 OneClass 和 SVR，被视为有两个类别
        n_class_SV = np.array([SV_len, SV_len], dtype=np.int32)

    # 初始化概率估计的 A 和 B 参数的数组
    cdef float64_t[::1] probA
    cdef float64_t[::1] probB
    # 如果概率不为零，则执行以下操作
    if probability != 0:
        # 如果 SVM 类型小于 2，表示是 SVC 或者 NuSVC
        if svm_type < 2:  # SVC and NuSVC
            # 根据类别数计算 probA 和 probB 的长度，分配内存空间
            probA = np.empty(int(n_class*(n_class-1)/2), dtype=np.float64)
            probB = np.empty(int(n_class*(n_class-1)/2), dtype=np.float64)
            # 调用外部函数 copy_probB 复制 probB 的数据
            copy_probB(<char*> &probB[0], model, <intp_t*> probB.shape)
        else:
            # 对于其他 SVM 类型，只分配一个元素长度的 probA
            probA = np.empty(1, dtype=np.float64)
            # 对于其他 SVM 类型，不分配 probB，长度为 0
            probB = np.empty(0, dtype=np.float64)
        # 调用外部函数 copy_probA 复制 probA 的数据
        copy_probA(<char*> &probA[0], model, <intp_t*> probA.shape)
    else:
        # 如果概率为零，设置 probA 和 probB 的长度为 0
        probA = np.empty(0, dtype=np.float64)
        probB = np.empty(0, dtype=np.float64)

    # 释放 SVM 模型的内存
    svm_free_and_destroy_model(&model)
    # 释放问题的输入特征数据内存
    free(problem.x)

    # 返回元组，包含 SVM 模型的各种基础信息
    return (
        support.base,        # 支持向量的基础数据
        support_vectors.base,    # 支持向量的基础数据
        n_class_SV.base,     # 支持向量的类别信息
        sv_coef.base,        # 支持向量的系数
        intercept.base,      # SVM 模型的截距
        probA.base,          # 预测概率的 probA
        probB.base,          # 预测概率的 probB
        fit_status,          # 拟合状态
        n_iter.base,         # 迭代次数
    )
cdef void set_predict_params(
    svm_parameter *param,
    int svm_type,
    kernel,
    int degree,
    double gamma,
    double coef0,
    double cache_size,
    int probability,
    int nr_weight,
    char *weight_label,
    char *weight,
) except *:
    """Fill param with prediction time-only parameters."""

    # training-time only parameters
    cdef double C = 0.0  # 初始化正则化参数 C
    cdef double epsilon = 0.1  # 初始化 epsilon 参数
    cdef int max_iter = 0  # 初始化最大迭代次数参数
    cdef double nu = 0.5  # 初始化 nu 参数
    cdef int shrinking = 0  # 初始化 shrinking 参数
    cdef double tol = 0.1  # 初始化 tol 参数
    cdef int random_seed = -1  # 初始化随机种子参数

    # 获取 kernel 在 LIBSVM_KERNEL_TYPES 列表中的索引
    kernel_index = LIBSVM_KERNEL_TYPES.index(kernel)

    # 调用 set_parameter 函数设置 svm_parameter 结构体的各项参数
    set_parameter(
        param,
        svm_type,
        kernel_index,
        degree,
        gamma,
        coef0,
        nu,
        cache_size,
        C,
        tol,
        epsilon,
        shrinking,
        probability,
        nr_weight,
        weight_label,
        weight,
        max_iter,
        random_seed,
    )


def predict(
    const float64_t[:, ::1] X,
    const int32_t[::1] support,
    const float64_t[:, ::1] SV,
    const int32_t[::1] nSV,
    const float64_t[:, ::1] sv_coef,
    const float64_t[::1] intercept,
    const float64_t[::1] probA=np.empty(0),
    const float64_t[::1] probB=np.empty(0),
    int svm_type=0,
    kernel='rbf',
    int degree=3,
    double gamma=0.1,
    double coef0=0.0,
    const float64_t[::1] class_weight=np.empty(0),
    const float64_t[::1] sample_weight=np.empty(0),
    double cache_size=100.0,
):
    """
    Predict target values of X given a model (low-level method)

    Parameters
    ----------
    X : array-like, dtype=float of shape (n_samples, n_features)
        输入样本特征矩阵

    support : array of shape (n_support,)
        训练集中支持向量的索引

    SV : array of shape (n_support, n_features)
        支持向量矩阵

    nSV : array of shape (n_class,)
        每个类别的支持向量数目

    sv_coef : array of shape (n_class-1, n_support)
        决策函数中支持向量的系数

    intercept : array of shape (n_class*(n_class-1)/2)
        决策函数中的截距

    probA, probB : array of shape (n_class*(n_class-1)/2,)
        概率估计

    svm_type : {0, 1, 2, 3, 4}, default=0
        SVM 类型: C_SVC, NuSVC, OneClassSVM, EpsilonSVR 或 NuSVR

    kernel : {'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'}, default="rbf"
        SVM 使用的核函数类型: 线性核、多项式核、径向基核、sigmoid核或预先计算的核

    degree : int32, default=3
        多项式核的阶数（仅当核函数设置为多项式时有效）

    gamma : float64, default=0.1
        RBF、多项式和sigmoid核的 gamma 参数，其他核函数忽略此参数

    coef0 : float64, default=0.0
        多项式/ sigmoid 核的独立参数

    Returns
    -------
    dec_values : array
        预测的目标值
    """
    cdef float64_t[::1] dec_values  # 定义决策值数组
    cdef svm_parameter param  # 定义 SVM 参数结构体
    # 声明一个 SVM 模型指针
    cdef svm_model *model
    # 声明一个整型变量 rv

    # 创建一个 int32_t 类型的一维数组 class_weight_label，长度为 class_weight 的第一个维度大小，用于存储类别权重的标签
    cdef int32_t[::1] class_weight_label = np.arange(
        class_weight.shape[0], dtype=np.int32
    )

    # 设置预测参数，将参数传递给 param 指针
    set_predict_params(
        &param,
        svm_type,
        kernel,
        degree,
        gamma,
        coef0,
        cache_size,
        0,
        <int>class_weight.shape[0],  # 类别权重数组的大小转换为整型
        <char*> &class_weight_label[0] if class_weight_label.size > 0 else NULL,  # 类别权重标签数组的首地址，如果有数据的话
        <char*> &class_weight[0] if class_weight.size > 0 else NULL,  # 类别权重数组的首地址，如果有数据的话
    )
    
    # 根据设置的参数创建 SVM 模型，并将其赋值给 model 指针
    model = set_model(
        &param,
        <int> nSV.shape[0],  # 支持向量的个数转换为整型
        <char*> &SV[0, 0] if SV.size > 0 else NULL,  # 支持向量矩阵的首地址，如果有数据的话
        <intp_t*> SV.shape,  # 支持向量矩阵的形状
        <char*> &support[0] if support.size > 0 else NULL,  # 支持向量标记数组的首地址，如果有数据的话
        <intp_t*> support.shape,  # 支持向量标记数组的形状
        <intp_t*> sv_coef.strides,  # 支持向量系数的步长数组
        <char*> &sv_coef[0, 0] if sv_coef.size > 0 else NULL,  # 支持向量系数矩阵的首地址，如果有数据的话
        <char*> &intercept[0],  # 截距数组的首地址
        <char*> &nSV[0],  # 每个类别的支持向量数量的数组的首地址
        <char*> &probA[0] if probA.size > 0 else NULL,  # SVM 模型的概率 A 的数组的首地址，如果有数据的话
        <char*> &probB[0] if probB.size > 0 else NULL,  # SVM 模型的概率 B 的数组的首地址，如果有数据的话
    )
    
    # 声明一个 BlasFunctions 结构体 blas_functions
    cdef BlasFunctions blas_functions
    # 将 _dot[double] 函数赋值给 blas_functions 的 dot 成员
    blas_functions.dot = _dot[double]
    
    # TODO: 使用 check_model（未实现的功能，待完成）
    
    # 尝试执行以下操作
    try:
        # 创建一个长度为 X 的第一个维度大小的空数组 dec_values
        dec_values = np.empty(X.shape[0])
        # 在无全局解锁区域内执行以下操作
        with nogil:
            # 使用 copy_predict 函数进行预测，将结果存储在 dec_values 数组中
            rv = copy_predict(
                <char*> &X[0, 0],  # 输入数据 X 的首地址
                model,  # SVM 模型指针
                <intp_t*> X.shape,  # 输入数据 X 的形状
                <char*> &dec_values[0],  # 预测结果数组的首地址
                &blas_functions,  # Blas 函数指针结构体
            )
        # 如果 rv 小于 0，则抛出内存错误异常
        if rv < 0:
            raise MemoryError("We've run out of memory")
    finally:
        # 释放 SVM 模型的内存资源
        free_model(model)
    
    # 返回 dec_values 的基础数据对象
    return dec_values.base
def predict_proba(
    const float64_t[:, ::1] X,
    const int32_t[::1] support,
    const float64_t[:, ::1] SV,
    const int32_t[::1] nSV,
    float64_t[:, ::1] sv_coef,
    float64_t[::1] intercept,
    float64_t[::1] probA=np.empty(0),
    float64_t[::1] probB=np.empty(0),
    int svm_type=0,
    kernel='rbf',
    int degree=3,
    double gamma=0.1,
    double coef0=0.0,
    float64_t[::1] class_weight=np.empty(0),
    float64_t[::1] sample_weight=np.empty(0),
    double cache_size=100.0,
):
    """
    Predict probabilities

    svm_model stores all parameters needed to predict a given value.

    For speed, all real work is done at the C level in function
    copy_predict (libsvm_helper.c).

    We have to reconstruct model and parameters to make sure we stay
    in sync with the python object.

    See sklearn.svm.predict for a complete list of parameters.

    Parameters
    ----------
    X : array-like, dtype=float of shape (n_samples, n_features)
        输入数据，形状为 (样本数, 特征数)

    support : array of shape (n_support,)
        训练集中支持向量的索引。

    SV : array of shape (n_support, n_features)
        支持向量。

    nSV : array of shape (n_class,)
        每个类别中支持向量的数量。

    sv_coef : array of shape (n_class-1, n_support)
        决策函数中支持向量的系数。

    intercept : array of shape (n_class*(n_class-1)/2,)
        决策函数中的截距。

    probA, probB : array of shape (n_class*(n_class-1)/2,)
        概率估计值。

    svm_type : {0, 1, 2, 3, 4}, default=0
        SVM 的类型：C_SVC, NuSVC, OneClassSVM, EpsilonSVR 或 NuSVR。

    kernel : {'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'}, default="rbf"
        模型中使用的核函数：线性核，多项式核，RBF 核，sigmoid 核或预计算核。

    degree : int32, default=3
        多项式核函数的阶数（仅在核函数设置为多项式时有效）。

    gamma : float64, default=0.1
        RBF、poly 和 sigmoid 核函数的 Gamma 参数。其他核函数忽略此参数。

    coef0 : float64, default=0.0
        poly/sigmoid 核函数的独立参数。

    Returns
    -------
    dec_values : array
        预测的值。
    """
    cdef float64_t[:, ::1] dec_values
    cdef svm_parameter param
    cdef svm_model *model
    cdef int32_t[::1] class_weight_label = np.arange(
        class_weight.shape[0], dtype=np.int32
    )
    cdef int rv

    set_predict_params(
        &param,
        svm_type,
        kernel,
        degree,
        gamma,
        coef0,
        cache_size,
        1,
        <int> class_weight.shape[0],
        <char*> &class_weight_label[0] if class_weight_label.size > 0 else NULL,
        <char*> &class_weight[0] if class_weight.size > 0 else NULL,
    )
    # 调用函数 `set_model`，设置模型参数并返回模型对象
    model = set_model(
        &param,  # 参数结构体的指针
        <int> nSV.shape[0],  # 支持向量数目的整数值
        <char*> &SV[0, 0] if SV.size > 0 else NULL,  # 支持向量数组的首元素地址，如果不存在则为 NULL
        <intp_t*> SV.shape,  # 支持向量数组的形状信息
        <char*> &support[0],  # 支持向量的首元素地址
        <intp_t*> support.shape,  # 支持向量的形状信息
        <intp_t*> sv_coef.strides,  # 支持向量系数的步进信息
        <char*> &sv_coef[0, 0],  # 支持向量系数的首元素地址
        <char*> &intercept[0],  # 截距的首元素地址
        <char*> &nSV[0],  # 支持向量数目的首元素地址
        <char*> &probA[0] if probA.size > 0 else NULL,  # probA 的首元素地址，如果不存在则为 NULL
        <char*> &probB[0] if probB.size > 0 else NULL,  # probB 的首元素地址，如果不存在则为 NULL
    )
    
    # 获取模型的类别数目
    cdef intp_t n_class = get_nr(model)
    
    # 定义 BLAS 函数结构体并初始化 dot 函数
    cdef BlasFunctions blas_functions
    blas_functions.dot = _dot[double]
    
    # 尝试预测概率值并赋给 dec_values 数组
    try:
        dec_values = np.empty((X.shape[0], n_class), dtype=np.float64)
        # 使用 nogil 上下文加速，调用 C 函数进行概率预测
        with nogil:
            rv = copy_predict_proba(
                <char*> &X[0, 0],  # 输入数据 X 的首元素地址
                model,  # 模型对象
                <intp_t*> X.shape,  # 输入数据 X 的形状信息
                <char*> &dec_values[0, 0],  # 预测结果 dec_values 的首元素地址
                &blas_functions,  # BLAS 函数结构体的指针
            )
        # 如果预测结果 rv 小于 0，抛出内存错误异常
        if rv < 0:
            raise MemoryError("We've run out of memory")
    finally:
        # 释放模型对象的内存资源
        free_model(model)
    
    # 返回 dec_values 数组的基础数据对象
    return dec_values.base
def decision_function(
    const float64_t[:, ::1] X,
    const int32_t[::1] support,
    const float64_t[:, ::1] SV,
    const int32_t[::1] nSV,
    const float64_t[:, ::1] sv_coef,
    const float64_t[::1] intercept,
    const float64_t[::1] probA=np.empty(0),
    const float64_t[::1] probB=np.empty(0),
    int svm_type=0,
    kernel='rbf',
    int degree=3,
    double gamma=0.1,
    double coef0=0.0,
    const float64_t[::1] class_weight=np.empty(0),
    const float64_t[::1] sample_weight=np.empty(0),
    double cache_size=100.0,
):
    """
    Predict margin (libsvm name for this is predict_values)

    We have to reconstruct model and parameters to make sure we stay
    in sync with the python object.

    Parameters
    ----------
    X : array-like, dtype=float, size=[n_samples, n_features]
        输入数据，包含 n_samples 行和 n_features 列的浮点型数组。

    support : array, shape=[n_support]
        训练集中支持向量的索引数组。

    SV : array, shape=[n_support, n_features]
        支持向量的二维浮点型数组，包含 n_support 行和 n_features 列。

    nSV : array, shape=[n_class]
        每个类别中支持向量的数量的整型数组。

    sv_coef : array, shape=[n_class-1, n_support]
        决策函数中支持向量的系数的二维浮点型数组。

    intercept : array, shape=[n_class*(n_class-1)/2]
        决策函数中的截距的浮点型数组。

    probA, probB : array, shape=[n_class*(n_class-1)/2]
        概率估计的浮点型数组。

    svm_type : {0, 1, 2, 3, 4}, optional
        SVM 的类型：C_SVC, NuSVC, OneClassSVM, EpsilonSVR 或 NuSVR。
        默认为 0 表示 C_SVC。

    kernel : {'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'}, optional
        模型中使用的核函数类型：linear（线性）、polynomial（多项式）、
        RBF（径向基函数）、sigmoid（sigmoid函数）或 precomputed（预计算）。
        默认为 'rbf' 表示径向基函数。

    degree : int32, optional
        多项式核函数的阶数（仅在 kernel 设置为 polynomial 时有效）。
        默认为 3。

    gamma : float64, optional
        RBF、poly 和 sigmoid 核函数中的 Gamma 参数，其他核函数忽略此参数。
        默认为 0.1。

    coef0 : float64, optional
        poly/sigmoid 核函数中的独立参数。
        默认为 0.0。

    Returns
    -------
    dec_values : array
        预测的决策值数组。
    """
    cdef float64_t[:, ::1] dec_values  # 定义决策值的二维浮点型数组
    cdef svm_parameter param  # 定义 SVM 参数对象
    cdef svm_model *model  # 定义 SVM 模型指针
    cdef intp_t n_class  # 定义类别数量的整型

    cdef int32_t[::1] class_weight_label = np.arange(
        class_weight.shape[0], dtype=np.int32
    )  # 根据 class_weight 的形状创建整型数组 class_weight_label

    cdef int rv  # 定义返回值变量 rv

    set_predict_params(
        &param,
        svm_type,
        kernel,
        degree,
        gamma,
        coef0,
        cache_size,
        0,
        <int> class_weight.shape[0],  # 设置 class_weight 的长度为整数
        <char*> &class_weight_label[0] if class_weight_label.size > 0 else NULL,  # 设置类别权重标签的指针
        <char*> &class_weight[0] if class_weight.size > 0 else NULL,  # 设置类别权重的指针
    )
    # 调用 set_model 函数，设置模型参数并返回模型对象
    model = set_model(
        &param,  # 参数对象的指针
        <int> nSV.shape[0],  # nSV 数组的第一个维度大小作为整数
        <char*> &SV[0, 0] if SV.size > 0 else NULL,  # SV 数组的第一个元素的地址，如果数组非空，否则为 NULL
        <intp_t*> SV.shape,  # SV 数组的形状
        <char*> &support[0],  # support 数组的第一个元素的地址
        <intp_t*> support.shape,  # support 数组的形状
        <intp_t*> sv_coef.strides,  # sv_coef 数组的步幅
        <char*> &sv_coef[0, 0],  # sv_coef 数组的第一个元素的地址
        <char*> &intercept[0],  # intercept 数组的第一个元素的地址
        <char*> &nSV[0],  # nSV 数组的第一个元素的地址
        <char*> &probA[0] if probA.size > 0 else NULL,  # probA 数组的第一个元素的地址，如果数组非空，否则为 NULL
        <char*> &probB[0] if probB.size > 0 else NULL,  # probB 数组的第一个元素的地址，如果数组非空，否则为 NULL
    )
    
    # 根据 svm_type 的值设置 n_class 变量
    if svm_type > 1:
        n_class = 1
    else:
        n_class = get_nr(model)  # 调用 get_nr 函数获取模型的类别数
        n_class = n_class * (n_class - 1) // 2  # 根据类别数计算二分类器的数量
    
    cdef BlasFunctions blas_functions  # 定义 BlasFunctions 结构体
    blas_functions.dot = _dot[double]  # 设置 BlasFunctions 结构体中 dot 成员的函数指针为 _dot 函数的 double 版本
    
    try:
        # 创建一个空的 dec_values 数组，形状为 (X.shape[0], n_class)，数据类型为 np.float64
        dec_values = np.empty((X.shape[0], n_class), dtype=np.float64)
        with nogil:  # 使用 nogil 语句块执行以下代码（无 GIL 环境）
            # 调用 copy_predict_values 函数，将预测结果复制到 dec_values 数组中
            rv = copy_predict_values(
                <char*> &X[0, 0],  # X 数组的第一个元素的地址
                model,  # 模型对象
                <intp_t*> X.shape,  # X 数组的形状
                <char*> &dec_values[0, 0],  # dec_values 数组的第一个元素的地址
                n_class,  # 二分类器的数量
                &blas_functions,  # BlasFunctions 结构体的指针
            )
        if rv < 0:
            raise MemoryError("We've run out of memory")  # 如果 rv 小于 0，抛出内存错误异常
    finally:
        free_model(model)  # 释放模型对象的内存
    
    # 返回 dec_values 数组的基础数据对象
    return dec_values.base
# 定义一个函数，用于执行交叉验证的操作，是一个底层的例程

def cross_validation(
    const float64_t[:, ::1] X,  # 输入数据 X，二维数组，数据类型为 float64
    const float64_t[::1] Y,     # 目标向量 Y，一维数组，数据类型为 float64
    int n_fold,                 # 交叉验证的折数，整数类型
    int svm_type=0,             # SVM 类型，默认为 0，对应于 C_SVC
    kernel='rbf',               # 核函数类型，默认为 RBF
    int degree=3,               # 多项式核函数的阶数，默认为 3
    double gamma=0.1,           # RBF、poly 和 sigmoid 核函数的参数 gamma，默认为 0.1
    double coef0=0.0,           # poly 和 sigmoid 核函数的独立参数 coef0，默认为 0.0
    double tol=1e-3,            # 数值停止条件，默认为 1e-3
    double C=1.0,               # C-SVC 中的参数 C，默认为 1.0
    double nu=0.5,              # NuSVC 和 NuSVR 中的参数 nu，默认为 0.5
    double epsilon=0.1,         # epsilon-insensitive 损失函数的参数 epsilon，默认为 0.1
    float64_t[::1] class_weight=np.empty(0),  # 类别权重数组，默认为空数组
    float64_t[::1] sample_weight=np.empty(0),  # 样本权重数组，默认为空数组
    int shrinking=0,            # 是否使用收缩启发式算法，默认为 0
    int probability=0,          # 是否启用概率估计，默认为 0
    double cache_size=100.0,    # 用于 gram 矩阵列的缓存大小，默认为 100.0 MB
    int max_iter=-1,            # 求解器的最大迭代次数，默认为 -1，表示无限制
    int random_seed=0,          # 概率估计所使用的随机数生成器种子，默认为 0
):
    """
    Binding of the cross-validation routine (low-level routine)

    Parameters
    ----------

    X : array-like, dtype=float of shape (n_samples, n_features)
        输入数据 X，浮点类型，形状为 (样本数, 特征数)

    Y : array, dtype=float of shape (n_samples,)
        目标向量 Y，浮点类型，形状为 (样本数,)

    n_fold : int32
        交叉验证的折数。

    svm_type : {0, 1, 2, 3, 4}, default=0
        SVM 类型：C_SVC、NuSVC、OneClassSVM、EpsilonSVR 或 NuSVR 中的一种。

    kernel : {'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'}, default='rbf'
        模型中使用的核函数：linear、polynomial、RBF、sigmoid 或 precomputed。

    degree : int32, default=3
        多项式核函数的阶数（仅在 kernel 设置为 polynomial 时有效）。

    gamma : float64, default=0.1
        RBF、poly 和 sigmoid 核函数的参数 gamma。其他核函数忽略该参数。

    coef0 : float64, default=0.0
        poly/sigmoid 核函数的独立参数。

    tol : float64, default=1e-3
        数值停止条件。

    C : float64, default=1
        C-Support Vector Classification 中的参数 C。

    nu : float64, default=0.5
        训练错误的上限和支持向量的下限的分数。应在 (0, 1] 区间内。

    epsilon : double, default=0.1
        epsilon-insensitive 损失函数的参数。

    class_weight : array, dtype=float64, shape (n_classes,), \
            default=np.empty(0)
        SVC 中类别 i 的参数 C 设置为 class_weight[i]*C。
        如果未提供，默认所有类别的权重均为一。

    sample_weight : array, dtype=float64, shape (n_samples,), \
            default=np.empty(0)
        每个样本分配的权重。

    shrinking : int, default=1
        是否使用收缩启发式算法。

    probability : int, default=0
        是否启用概率估计。

    cache_size : float64, default=100
        gram 矩阵列的缓存大小（以兆字节为单位）。

    max_iter : int (-1 for no limit), default=-1
        无论准确性如何，求解器在此迭代次数后停止。

    random_seed : int, default=0
        用于概率估计的随机数生成器种子。

    Returns
    -------
    target : array, float

    """

    cdef svm_parameter param  # 定义 SVM 参数对象
    cdef svm_problem problem  # 定义 SVM 问题对象
    cdef const char *error_msg  # 错误消息字符串
    # 如果样本权重为空，则将其设置为一个全为1的数组，长度为样本数量
    if len(sample_weight) == 0:
        sample_weight = np.ones(X.shape[0], dtype=np.float64)
    # 如果样本权重不为空，则确保其长度与样本数量相同，否则抛出异常
    else:
        assert sample_weight.shape[0] == X.shape[0], (
            f"sample_weight and X have incompatible shapes: sample_weight has "
            f"{sample_weight.shape[0]} samples while X has {X.shape[0]}"
        )

    # 如果样本数量小于折数，则抛出数值错误异常
    if X.shape[0] < n_fold:
        raise ValueError("Number of samples is less than number of folds")

    # 设置问题
    kernel_index = LIBSVM_KERNEL_TYPES.index(kernel)
    set_problem(
        &problem,
        <char*> &X[0, 0],
        <char*> &Y[0],
        <char*> &sample_weight[0] if sample_weight.size > 0 else NULL,
        <intp_t*> X.shape,
        kernel_index,
    )
    # 如果问题的 x 为空，则抛出内存错误异常
    if problem.x == NULL:
        raise MemoryError("Seems we've run out of memory")
    # 创建一个整型数组作为类别权重标签
    cdef int32_t[::1] class_weight_label = np.arange(
        class_weight.shape[0], dtype=np.int32
    )

    # 设置参数
    set_parameter(
        &param,
        svm_type,
        kernel_index,
        degree,
        gamma,
        coef0,
        nu,
        cache_size,
        C,
        tol,
        tol,
        shrinking,
        probability,
        <int> class_weight.shape[0],
        <char*> &class_weight_label[0] if class_weight_label.size > 0 else NULL,
        <char*> &class_weight[0] if class_weight.size > 0 else NULL,
        max_iter,
        random_seed,
    )

    # 检查参数，如果有错误信息则抛出数值错误异常
    error_msg = svm_check_parameter(&problem, &param)
    if error_msg:
        raise ValueError(error_msg)

    # 创建一个浮点数数组作为目标值
    cdef float64_t[::1] target
    cdef BlasFunctions blas_functions
    blas_functions.dot = _dot[double]
    try:
        # 创建一个空数组作为目标值，使用 GIL 释放
        target = np.empty((X.shape[0]), dtype=np.float64)
        with nogil:
            # 进行 SVM 交叉验证
            svm_cross_validation(
                &problem,
                &param,
                n_fold,
                <double *> &target[0],
                &blas_functions,
            )
    finally:
        # 释放问题的 x
        free(problem.x)

    # 返回目标值的基础数组
    return target.base
# 定义一个函数，用于设置 libsvm 库的输出详细程度
def set_verbosity_wrap(int verbosity):
    # 调用底层函数 set_verbosity，将参数 verbosity 传递给它
    set_verbosity(verbosity)
```
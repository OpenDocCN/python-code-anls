# `D:\src\scipysrc\scikit-learn\sklearn\svm\_libsvm_sparse.pyx`

```
import  numpy as np
from scipy import sparse
from ..utils._cython_blas cimport _dot
from ..utils._typedefs cimport float64_t, int32_t, intp_t

cdef extern from *:
    ctypedef char* const_char_p "const char*"

################################################################################
# Includes

# 导入外部头文件 "_svm_cython_blas_helpers.h" 中定义的结构和函数指针类型
cdef extern from "_svm_cython_blas_helpers.h":
    ctypedef double (*dot_func)(int, const double*, int, const double*, int)
    cdef struct BlasFunctions:
        dot_func dot

# 导入 "svm.h" 中定义的 SVM 相关结构和函数声明
cdef extern from "svm.h":
    cdef struct svm_csr_node
    cdef struct svm_csr_model
    cdef struct svm_parameter
    cdef struct svm_csr_problem
    char *svm_csr_check_parameter(svm_csr_problem *, svm_parameter *)
    svm_csr_model *svm_csr_train(svm_csr_problem *, svm_parameter *, int *, BlasFunctions *) nogil
    void svm_csr_free_and_destroy_model(svm_csr_model** model_ptr_ptr)

# 导入 "libsvm_sparse_helper.c" 中的方法声明，用于处理 libsvm 的隐藏字段
cdef extern from "libsvm_sparse_helper.c":
    # this file contains methods for accessing libsvm 'hidden' fields
    svm_csr_problem * csr_set_problem (
        char *, intp_t *, char *, intp_t *, char *, char *, char *, int)
    svm_csr_model *csr_set_model(svm_parameter *param, int nr_class,
                                 char *SV_data, intp_t *SV_indices_dims,
                                 char *SV_indices, intp_t *SV_intptr_dims,
                                 char *SV_intptr,
                                 char *sv_coef, char *rho, char *nSV,
                                 char *probA, char *probB)
    svm_parameter *set_parameter (int , int , int , double, double ,
                                  double , double , double , double,
                                  double, int, int, int, char *, char *, int,
                                  int)
    void copy_sv_coef   (char *, svm_csr_model *)
    void copy_n_iter  (char *, svm_csr_model *)
    void copy_support   (char *, svm_csr_model *)
    void copy_intercept (char *, svm_csr_model *, intp_t *)
    int copy_predict (char *, svm_csr_model *, intp_t *, char *, BlasFunctions *)
    int csr_copy_predict_values (intp_t *data_size, char *data, intp_t *index_size,
                                 char *index, intp_t *intptr_size, char *size,
                                 svm_csr_model *model, char *dec_values, int nr_class, BlasFunctions *)
    int csr_copy_predict (intp_t *data_size, char *data, intp_t *index_size,
                          char *index, intp_t *intptr_size, char *size,
                          svm_csr_model *model, char *dec_values, BlasFunctions *) nogil
    int csr_copy_predict_proba (intp_t *data_size, char *data, intp_t *index_size,
                                char *index, intp_t *intptr_size, char *size,
                                svm_csr_model *model, char *dec_values, BlasFunctions *) nogil

    int  copy_predict_values(char *, svm_csr_model *, intp_t *, char *, int, BlasFunctions *)
    // 复制稀疏向量（Sparse Vector）的支持向量（Support Vectors）到新的内存空间中
    int csr_copy_SV(char *values, intp_t *n_indices,
                    char *indices, intp_t *n_indptr, char *indptr,
                    svm_csr_model *model, int n_features)
    
    // 获取支持向量（Support Vectors）中非零元素的个数
    intp_t get_nonzero_SV(svm_csr_model *)
    
    // 复制支持向量（Support Vectors）的数量到新的内存空间中
    void copy_nSV(char *, svm_csr_model *)
    
    // 复制模型中的 probA 参数到新的内存空间中
    void copy_probA(char *, svm_csr_model *, intp_t *)
    
    // 复制模型中的 probB 参数到新的内存空间中
    void copy_probB(char *, svm_csr_model *, intp_t *)
    
    // 获取模型中的样本数目
    intp_t get_l(svm_csr_model *)
    
    // 获取模型中的特征数目
    intp_t get_nr(svm_csr_model *)
    
    // 释放存储支持向量问题（Support Vector Problem）的内存空间
    int free_problem(svm_csr_problem *)
    
    // 释放支持向量机模型（SVM Model）的内存空间
    int free_model(svm_csr_model *)
    
    // 释放支持向量机参数（SVM Parameter）的内存空间
    int free_param(svm_parameter *)
    
    // 释放支持向量机模型（SVM Model）的支持向量的内存空间
    int free_model_SV(svm_csr_model *model)
    
    // 设置日志输出的详细程度
    void set_verbosity(int)
def libsvm_sparse_train (int n_features,
                         const float64_t[::1] values,
                         const int32_t[::1] indices,
                         const int32_t[::1] indptr,
                         const float64_t[::1] Y,
                         int svm_type, int kernel_type, int degree, double gamma,
                         double coef0, double eps, double C,
                         const float64_t[::1] class_weight,
                         const float64_t[::1] sample_weight,
                         double nu, double cache_size, double p, int
                         shrinking, int probability, int max_iter,
                         int random_seed):
    """
    Wrap svm_train from libsvm using a scipy.sparse.csr matrix

    Work in progress.

    Parameters
    ----------
    n_features : number of features.
        XXX: can we retrieve this from any other parameter ?

    values : array-like, dtype=float64, size=[M]
        Non-zero values of the sparse matrix in CSR format.

    indices : array-like, dtype=int32, size=[M]
        Column indices of the sparse matrix in CSR format.

    indptr : array-like, dtype=int32, size=[N + 1]
        Indices to locate rows in `values` and `indices` for the CSR format.

    Y : array-like, dtype=float64, size=[N]
        Target vector for training.

    svm_type : int
        Type of SVM (C_SVC, NU_SVC, etc.).

    kernel_type : int
        Type of kernel function (linear, polynomial, etc.).

    degree : int
        Degree of the polynomial kernel function (if applicable).

    gamma : float
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

    coef0 : float
        Independent term in kernel function. Only significant in 'poly' and 'sigmoid'.

    eps : float
        Epsilon in the stopping criterion.

    C : float
        Penalty parameter C of the error term.

    class_weight : array-like, dtype=float64, size=[K]
        Weights associated with classes in a C_SVC problem.

    sample_weight : array-like, dtype=float64, size=[N]
        Weights associated with samples (data points).

    nu : float
        Parameter of the SVM algorithm for NU_SVC, ONE_CLASS, and NU_SVR.

    cache_size : float
        Size of the kernel cache (in MB).

    p : float
        Epsilon in loss function of epsilon-SVR.

    shrinking : int
        Whether to use the shrinking heuristic.

    probability : int
        Whether to train a SVC or SVR model for probability estimates.

    max_iter : int
        Hard limit on iterations within solver.

    random_seed : int
        Random seed for reproducibility.

    Notes
    -------------------
    See sklearn.svm.predict for a complete list of parameters.

    """

    cdef svm_parameter *param
    cdef svm_csr_problem *problem
    cdef svm_csr_model *model
    cdef const_char_p error_msg

    if len(sample_weight) == 0:
        sample_weight = np.ones(Y.shape[0], dtype=np.float64)
    else:
        assert sample_weight.shape[0] == indptr.shape[0] - 1, \
               "sample_weight and X have incompatible shapes: " + \
               "sample_weight has %s samples while X has %s" % \
               (sample_weight.shape[0], indptr.shape[0] - 1)

    # we should never end up here with a precomputed kernel matrix,
    # as this is always dense.
    assert(kernel_type != 4)

    # set libsvm problem using sparse CSR format
    problem = csr_set_problem(
        <char *> &values[0],  # Non-zero values of the sparse matrix
        <intp_t *> indices.shape,  # Shape of `indices` array
        <char *> &indices[0],  # Column indices of the sparse matrix
        <intp_t *> indptr.shape,  # Shape of `indptr` array
        <char *> &indptr[0],  # Indices to locate rows in `values` and `indices`
        <char *> &Y[0],  # Target vector
        <char *> &sample_weight[0],  # Sample weights
        kernel_type,  # Type of kernel function
    )

    cdef int32_t[::1] \
        class_weight_label = np.arange(class_weight.shape[0], dtype=np.int32)

    # set parameters for the SVM model
    param = set_parameter(
        svm_type,  # Type of SVM
        kernel_type,  # Type of kernel function
        degree,  # Degree of polynomial kernel (if applicable)
        gamma,  # Kernel coefficient
        coef0,  # Independent term in kernel function
        nu,  # Parameter for SVM algorithm
        cache_size,  # Size of kernel cache
        C,  # Penalty parameter
        eps,  # Epsilon in stopping criterion
        p,  # Epsilon in loss function of epsilon-SVR
        shrinking,  # Whether to use shrinking heuristic
        probability,  # Whether to train model for probability estimates
        <int> class_weight.shape[0],  # Number of class weights
        <char *> &class_weight_label[0] if class_weight_label.size > 0 else NULL,  # Class weight labels
        <char *> &class_weight[0] if class_weight.size > 0 else NULL,  # Class weights
        max_iter,  # Maximum number of iterations
        random_seed,  # Random seed
    )

    # check if parameters or problem initialization failed
    if (param == NULL or problem == NULL):
        raise MemoryError("Seems we've run out of memory")
    
    # check SVM parameter constraints
    error_msg = svm_csr_check_parameter(problem, param)
    if error_msg:
        free_problem(problem)
        free_param(param)
        raise ValueError(error_msg)

    cdef BlasFunctions blas_functions
    blas_functions.dot = _dot[double]
    # 调用 svm_train，这里进行真正的训练工作
    cdef int fit_status = 0
    with nogil:
        model = svm_csr_train(problem, param, &fit_status, &blas_functions)

    # 获取支持向量的长度和类别数目
    cdef intp_t SV_len = get_l(model)
    cdef intp_t n_class = get_nr(model)

    # 创建并复制 n_iter 数组，用于存储迭代次数
    cdef int[::1] n_iter
    n_iter = np.empty(max(1, n_class * (n_class - 1) // 2), dtype=np.intc)
    copy_n_iter(<char *> &n_iter[0], model)

    # 复制 model.sv_coef 数据到新数组 sv_coef_data
    # 我们创建一个新数组而不是调整大小，以确保不会保留先前的信息
    cdef float64_t[::1] sv_coef_data
    sv_coef_data = np.empty((n_class-1) * SV_len, dtype=np.float64)
    copy_sv_coef(<char *> &sv_coef_data[0] if sv_coef_data.size > 0 else NULL, model)

    # 复制支持向量的索引到数组 support
    cdef int32_t[::1] support
    support = np.empty(SV_len, dtype=np.int32)
    copy_support(<char *> &support[0] if support.size > 0 else NULL, model)

    # 复制 model.rho 到截距数组 intercept
    # 截距即 model.rho，但符号相反
    cdef float64_t[::1] intercept
    intercept = np.empty(n_class * (n_class - 1) // 2, dtype=np.float64)
    copy_intercept(<char *> &intercept[0], model, <intp_t *> intercept.shape)

    # 复制 model.SV 到稀疏矩阵 support_vectors_
    # 我们擦除 SV 中的任何先前信息
    # TODO: 自定义核函数
    cdef intp_t nonzero_SV
    nonzero_SV = get_nonzero_SV(model)

    cdef float64_t[::1] SV_data
    cdef int32_t[::1] SV_indices, SV_indptr
    SV_data = np.empty(nonzero_SV, dtype=np.float64)
    SV_indices = np.empty(nonzero_SV, dtype=np.int32)
    SV_indptr = np.empty(<intp_t> SV_len + 1, dtype=np.int32)
    csr_copy_SV(
        <char *> &SV_data[0] if SV_data.size > 0 else NULL,
        <intp_t *> SV_indices.shape,
        <char *> &SV_indices[0] if SV_indices.size > 0 else NULL,
        <intp_t *> SV_indptr.shape,
        <char *> &SV_indptr[0] if SV_indptr.size > 0 else NULL,
        model,
        n_features,
    )
    support_vectors_ = sparse.csr_matrix(
        (SV_data, SV_indices, SV_indptr), (SV_len, n_features)
    )

    # 复制 model.nSV 到数组 n_class_SV
    # TODO: 仅在分类中执行此操作
    cdef int32_t[::1] n_class_SV
    n_class_SV = np.empty(n_class, dtype=np.int32)
    copy_nSV(<char *> &n_class_SV[0], model)

    # 复制概率相关信息到 probA 和 probB 数组
    cdef float64_t[::1] probA, probB
    if probability != 0:
        if svm_type < 2:  # SVC 和 NuSVC
            probA = np.empty(n_class * (n_class - 1) // 2, dtype=np.float64)
            probB = np.empty(n_class * (n_class - 1) // 2, dtype=np.float64)
            copy_probB(<char *> &probB[0], model, <intp_t *> probB.shape)
        else:
            probA = np.empty(1, dtype=np.float64)
            probB = np.empty(0, dtype=np.float64)
        copy_probA(<char *> &probA[0], model, <intp_t *> probA.shape)
    else:
        probA = np.empty(0, dtype=np.float64)
        probB = np.empty(0, dtype=np.float64)

    # 释放并销毁 SVM 模型
    svm_csr_free_and_destroy_model(&model)
    free_problem(problem)
    free_param(param)
    # 返回一个包含多个值的元组，依次为：
    # - support.base: 支持向量机模型中支持向量的数据
    # - support_vectors_.base: 支持向量机模型中支持向量的数据
    # - sv_coef_data.base: 支持向量机模型中支持向量系数的数据
    # - intercept.base: 支持向量机模型中截距的数据
    # - n_class_SV.base: 支持向量机模型中支持向量的类别信息
    # - probA.base: 支持向量机模型中用于计算概率的参数A的数据
    # - probB.base: 支持向量机模型中用于计算概率的参数B的数据
    # - fit_status: 支持向量机模型的拟合状态
    # - n_iter.base: 支持向量机模型的迭代次数信息
    return (
        support.base,
        support_vectors_,
        sv_coef_data.base,
        intercept.base,
        n_class_SV.base,
        probA.base,
        probB.base,
        fit_status,
        n_iter.base,
    )
# 定义一个函数用于预测稀疏数据的值，基于 libsvm 模型进行预测

def libsvm_sparse_predict (const float64_t[::1] T_data,
                           const int32_t[::1] T_indices,
                           const int32_t[::1] T_indptr,
                           const float64_t[::1] SV_data,
                           const int32_t[::1] SV_indices,
                           const int32_t[::1] SV_indptr,
                           const float64_t[::1] sv_coef,
                           const float64_t[::1] intercept,
                           int svm_type, int kernel_type, int
                           degree, double gamma, double coef0, double
                           eps, double C,
                           const float64_t[:] class_weight,
                           double nu, double p, int
                           shrinking, int probability,
                           const int32_t[::1] nSV,
                           const float64_t[::1] probA,
                           const float64_t[::1] probB):
    """
    Predict values T given a model.

    For speed, all real work is done at the C level in function
    copy_predict (libsvm_helper.c).

    We have to reconstruct model and parameters to make sure we stay
    in sync with the python object.

    See sklearn.svm.predict for a complete list of parameters.

    Parameters
    ----------
    T_data : array-like, dtype=float64
        数据向量的数据部分
    T_indices : array-like, dtype=int32
        数据向量的索引部分
    T_indptr : array-like, dtype=int32
        数据向量的指针部分
    SV_data : array-like, dtype=float64
        支持向量的数据部分
    SV_indices : array-like, dtype=int32
        支持向量的索引部分
    SV_indptr : array-like, dtype=int32
        支持向量的指针部分
    sv_coef : array-like, dtype=float64
        支持向量的系数
    intercept : array-like, dtype=float64
        截距
    svm_type : int
        SVM 的类型
    kernel_type : int
        核函数的类型
    degree : int
        多项式核函数的次数
    gamma : double
        核函数的参数
    coef0 : double
        核函数的截距参数
    eps : double
        退出标准
    C : double
        惩罚系数
    class_weight : array-like, dtype=float64
        类别权重
    nu : double
        Nu 参数
    p : double
        epsilon-SVR 中损失函数的参数
    shrinking : int
        是否使用启发式方法
    probability : int
        是否启用概率估计
    nSV : array-like, dtype=int32
        每个类别的支持向量数目
    probA : array-like, dtype=float64
        概率估计的 A 参数
    probB : array-like, dtype=float64
        概率估计的 B 参数

    Returns
    -------
    dec_values : array
        预测的值
    """
    # 声明一个数组以存储预测的结果
    cdef float64_t[::1] dec_values
    # 声明用于 SVM 参数和模型的指针
    cdef svm_parameter *param
    cdef svm_csr_model *model
    # 创建一个与类别权重形状相匹配的标签数组
    cdef int32_t[::1] \
        class_weight_label = np.arange(class_weight.shape[0], dtype=np.int32)
    # 初始化返回值
    cdef int rv
    # 设置 SVM 参数
    param = set_parameter(
        svm_type,
        kernel_type,
        degree,
        gamma,
        coef0,
        nu,
        100.0,  # cache size has no effect on predict
        C,
        eps,
        p,
        shrinking,
        probability,
        <int> class_weight.shape[0],
        <char *> &class_weight_label[0] if class_weight_label.size > 0 else NULL,
        <char *> &class_weight[0] if class_weight.size > 0 else NULL,
        -1,
        -1,  # random seed has no effect on predict either
    )
    # 设置模型参数
    model = csr_set_model(
        param, <int> nSV.shape[0],
        <char *> &SV_data[0] if SV_data.size > 0 else NULL,
        <intp_t *>SV_indices.shape,
        <char *> &SV_indices[0] if SV_indices.size > 0 else NULL,
        <intp_t *> SV_indptr.shape,
        <char *> &SV_indptr[0] if SV_indptr.size > 0 else NULL,
        <char *> &sv_coef[0] if sv_coef.size > 0 else NULL,
        <char *> &intercept[0],
        <char *> &nSV[0],
        <char *> &probA[0] if probA.size > 0 else NULL,
        <char *> &probB[0] if probB.size > 0 else NULL,
    )
    # TODO: use check_model
    # 创建一个空数组来存储决策值
    dec_values = np.empty(T_indptr.shape[0]-1)
    # 声明 BLAS 函数
    cdef BlasFunctions blas_functions
    # 设置 BLAS 函数中的点积函数为双精度的 dot 函数
    blas_functions.dot = _dot[double]
    # 使用 `nogil` 上下文执行以下代码块，禁用 GIL（全局解释器锁）
    with nogil:
        # 调用 csr_copy_predict 函数进行预测
        rv = csr_copy_predict(
            <intp_t *> T_data.shape,   # T_data 的形状的整型指针
            <char *> &T_data[0],       # T_data 的首地址的字符指针
            <intp_t *> T_indices.shape,    # T_indices 的形状的整型指针
            <char *> &T_indices[0],        # T_indices 的首地址的字符指针
            <intp_t *> T_indptr.shape,     # T_indptr 的形状的整型指针
            <char *> &T_indptr[0],         # T_indptr 的首地址的字符指针
            model,                           # 模型对象的地址
            <char *> &dec_values[0],         # dec_values 的首地址的字符指针
            &blas_functions,                 # blas_functions 的地址
        )
    
    # 检查 rv 是否小于 0，如果是，则抛出内存错误异常
    if rv < 0:
        raise MemoryError("We've run out of memory")
    
    # 释放模型的支持向量和参数
    free_model_SV(model)
    free_model(model)
    free_param(param)
    
    # 返回 dec_values 的基础数据对象
    return dec_values.base
# 定义函数 libsvm_sparse_predict_proba，用于根据稀疏数据 T 预测概率值
def libsvm_sparse_predict_proba(
    const float64_t[::1] T_data,  # 测试数据的非零元素值
    const int32_t[::1] T_indices,  # 测试数据的非零元素索引
    const int32_t[::1] T_indptr,  # 测试数据的行指针
    const float64_t[::1] SV_data,  # 支持向量的非零元素值
    const int32_t[::1] SV_indices,  # 支持向量的非零元素索引
    const int32_t[::1] SV_indptr,  # 支持向量的行指针
    const float64_t[::1] sv_coef,  # 支持向量的系数
    const float64_t[::1] intercept,  # 截距
    int svm_type,  # SVM 类型
    int kernel_type,  # 核函数类型
    int degree,  # 多项式核函数的阶数
    double gamma,  # 核函数的 gamma 参数
    double coef0,  # 核函数的常数项参数
    double eps,  # 求解精度
    double C,  # 惩罚系数
    const float64_t[:] class_weight,  # 类别权重
    double nu,  # SVM 的 nu 参数
    double p,  # 损失函数的 p 参数
    int shrinking,  # 是否使用收缩启发式
    int probability,  # 是否启用概率估计
    const int32_t[::1] nSV,  # 每个类别的支持向量数量
    const float64_t[::1] probA,  # SVM 概率估计的 A 参数
    const float64_t[::1] probB,  # SVM 概率估计的 B 参数
):
    """
    Predict values T given a model.
    """
    cdef float64_t[:, ::1] dec_values  # 存储决策函数值的二维数组
    cdef svm_parameter *param  # SVM 模型的参数结构体指针
    cdef svm_csr_model *model  # 基于 CSR 格式的 SVM 模型指针
    cdef int32_t[::1] \
        class_weight_label = np.arange(class_weight.shape[0], dtype=np.int32)  # 类别权重的标签数组

    # 设置 SVM 模型参数
    param = set_parameter(
        svm_type,
        kernel_type,
        degree,
        gamma,
        coef0,
        nu,
        100.0,  # 缓存大小在预测中无效
        C,
        eps,
        p,
        shrinking,
        probability,
        <int> class_weight.shape[0],
        <char *> &class_weight_label[0] if class_weight_label.size > 0 else NULL,
        <char *> &class_weight[0] if class_weight.size > 0 else NULL,
        -1,
        -1,  # 随机种子在预测中也无效
    )

    # 设置 CSR 格式的 SVM 模型
    model = csr_set_model(
        param,
        <int> nSV.shape[0],
        <char *> &SV_data[0] if SV_data.size > 0 else NULL,
        <intp_t *> SV_indices.shape,
        <char *> &SV_indices[0] if SV_indices.size > 0 else NULL,
        <intp_t *> SV_indptr.shape,
        <char *> &SV_indptr[0] if SV_indptr.size > 0 else NULL,
        <char *> &sv_coef[0] if sv_coef.size > 0 else NULL,
        <char *> &intercept[0],
        <char *> &nSV[0],
        <char *> &probA[0] if probA.size > 0 else NULL,
        <char *> &probB[0] if probB.size > 0 else NULL,
    )

    # 获取模型的类别数目
    cdef intp_t n_class = get_nr(model)

    cdef int rv
    # 分配空间来存储预测的概率估计值
    dec_values = np.empty((T_indptr.shape[0]-1, n_class), dtype=np.float64)

    cdef BlasFunctions blas_functions
    blas_functions.dot = _dot[double]
    # 调用 CSR 格式的预测函数进行概率估计
    with nogil:
        rv = csr_copy_predict_proba(
            <intp_t *> T_data.shape,
            <char *> &T_data[0],
            <intp_t *> T_indices.shape,
            <char *> &T_indices[0],
            <intp_t *> T_indptr.shape,
            <char *> &T_indptr[0],
            model,
            <char *> &dec_values[0, 0],
            &blas_functions,
        )

    # 检查预测是否成功
    if rv < 0:
        raise MemoryError("We've run out of memory")

    # 释放模型和参数
    free_model_SV(model)
    free_model(model)
    free_param(param)

    # 返回决策函数值的基础数组
    return dec_values.base


# 定义函数 libsvm_sparse_decision_function，用于根据稀疏数据 T 决策函数值
def libsvm_sparse_decision_function(
    const float64_t[::1] T_data,  # 测试数据的非零元素值
    const int32_t[::1] T_indices,  # 测试数据的非零元素索引
    const int32_t[::1] T_indptr,  # 测试数据的行指针
    const float64_t[::1] SV_data,  # 支持向量的非零元素值
    const int32_t[::1] SV_indices,  # 支持向量的非零元素索引
    const int32_t[::1] SV_indptr,  # 支持向量的行指针
    const float64_t[::1] sv_coef,
    # sv_coef：用于存储支持向量机模型中支持向量的系数数组

    const float64_t[::1] intercept,
    # intercept：存储支持向量机模型的截距数组

    int svm_type,
    # svm_type：支持向量机类型，指定了模型是分类还是回归

    int kernel_type,
    # kernel_type：核函数类型，指定了支持向量机模型中使用的核函数类型

    int degree,
    # degree：核函数的多项式核的阶数，仅在核函数类型为多项式时有效

    double gamma,
    # gamma：核函数的系数，影响支持向量机模型的复杂度和拟合效果

    double coef0,
    # coef0：核函数中的独立项，仅在核函数类型为多项式或sigmoid时有效

    double eps,
    # eps：数值优化过程中的收敛阈值，影响支持向量机模型的拟合精度

    double C,
    # C：正则化参数，控制支持向量机模型的平衡性，影响模型对训练数据的拟合程度

    const float64_t[:] class_weight,
    # class_weight：类别权重数组，用于处理数据集中类别不平衡问题

    double nu,
    # nu：SVC中的参数，控制支持向量的数量，仅在svm_type为NU_SVC或ONE_CLASS时有效

    double p,
    # p：SVR中的参数，指定损失函数中的epsilon值

    int shrinking,
    # shrinking：是否使用启发式收缩法来加速训练过程的标志

    int probability,
    # probability：是否启用概率估计的标志，用于SVC或SVR中的概率预测

    const int32_t[::1] nSV,
    # nSV：每个类别中的支持向量数量的数组

    const float64_t[::1] probA,
    # probA：用于概率估计的SVC模型参数

    const float64_t[::1] probB,
    # probB：用于概率估计的SVC模型参数
):
    """
    Predict margin (libsvm name for this is predict_values)

    We have to reconstruct model and parameters to make sure we stay
    in sync with the python object.
    """
    # 声明一个二维数组 dec_values，用于存储预测的决策值
    cdef float64_t[:, ::1] dec_values
    # 声明一个指针 param，用于存储 SVM 参数
    cdef svm_parameter *param
    # 声明一个整型变量 n_class，用于存储分类的数量

    # 声明一个指针 model，用于存储 CSR 格式的 SVM 模型
    cdef svm_csr_model *model
    # 声明一个整型一维数组 class_weight_label，用于存储类别权重的标签
    cdef int32_t[::1] \
        class_weight_label = np.arange(class_weight.shape[0], dtype=np.int32)
    
    # 调用 set_parameter 函数设置 SVM 参数，并赋值给 param
    param = set_parameter(
        svm_type,
        kernel_type,
        degree,
        gamma,
        coef0,
        nu,
        100.0,  # cache size has no effect on predict
        C,
        eps,
        p,
        shrinking,
        probability,
        <int> class_weight.shape[0],
        <char *> &class_weight_label[0] if class_weight_label.size > 0 else NULL,
        <char *> &class_weight[0] if class_weight.size > 0 else NULL,
        -1,
        -1,
    )

    # 调用 csr_set_model 函数设置 SVM 模型，并赋值给 model
    model = csr_set_model(
        param,
        <int> nSV.shape[0],
        <char *> &SV_data[0] if SV_data.size > 0 else NULL,
        <intp_t *> SV_indices.shape,
        <char *> &SV_indices[0] if SV_indices.size > 0 else NULL,
        <intp_t *> SV_indptr.shape,
        <char *> &SV_indptr[0] if SV_indptr.size > 0 else NULL,
        <char *> &sv_coef[0] if sv_coef.size > 0 else NULL,
        <char *> &intercept[0],
        <char *> &nSV[0],
        <char *> &probA[0] if probA.size > 0 else NULL,
        <char *> &probB[0] if probB.size > 0 else NULL,
    )

    # 根据 svm_type 确定分类的数量
    if svm_type > 1:
        n_class = 1
    else:
        n_class = get_nr(model)
        n_class = n_class * (n_class - 1) // 2

    # 初始化 dec_values 数组，用于存储预测的决策值
    dec_values = np.empty((T_indptr.shape[0] - 1, n_class), dtype=np.float64)
    # 声明 BlasFunctions 结构体 blas_functions，用于执行线性代数计算
    cdef BlasFunctions blas_functions
    blas_functions.dot = _dot[double]
    # 调用 csr_copy_predict_values 函数进行预测，并将结果存储在 dec_values 中
    if csr_copy_predict_values(
            <intp_t *> T_data.shape,
            <char *> &T_data[0],
            <intp_t *> T_indices.shape,
            <char *> &T_indices[0],
            <intp_t *> T_indptr.shape,
            <char *> &T_indptr[0],
            model,
            <char *> &dec_values[0, 0],
            n_class,
            &blas_functions,
    ) < 0:
        raise MemoryError("We've run out of memory")
    
    # 释放 SVM 模型和参数的内存空间
    free_model_SV(model)
    free_model(model)
    free_param(param)

    # 返回 dec_values 数组的基础对象
    return dec_values.base


def set_verbosity_wrap(int verbosity):
    """
    Control verbosity of libsvm library
    """
    # 调用 set_verbosity 函数设置 libsvm 库的输出详细程度
    set_verbosity(verbosity)
```
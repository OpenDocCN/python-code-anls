# `D:\src\scipysrc\scikit-learn\sklearn\manifold\_barnes_hut_tsne.pyx`

```
# 导入必要的库和模块
import numpy as np  # 导入 NumPy 库
cimport numpy as cnp  # 导入 Cython 版本的 NumPy 库
from libc.stdio cimport printf  # 导入 printf 函数的 C 标准库接口
from libc.math cimport log  # 导入 log 函数的 C 标准库接口
from libc.stdlib cimport malloc, free  # 导入 malloc 和 free 函数的 C 标准库接口
from libc.time cimport clock, clock_t  # 导入 clock 和 clock_t 函数的 C 标准库接口
from cython.parallel cimport prange, parallel  # 导入 Cython 并行化模块

from ..neighbors._quad_tree cimport _QuadTree  # 导入自定义的 QuadTree 类


cnp.import_array()  # 导入 NumPy 的 C 接口函数


cdef char* EMPTY_STRING = ""  # 定义一个空字符串变量


# 定义浮点数的最小正值，用于避免在计算 KL 散度时取对数零的情况
cdef float FLOAT32_TINY = np.finfo(np.float32).tiny


# 定义一个小的浮点数值，用于避免除零或趋向正无穷的情况
cdef float FLOAT64_EPS = np.finfo(np.float64).eps


# 定义一个枚举类型 DEBUGFLAG，用于在 Cython 中条件编译打印调试信息
cdef enum:
    DEBUGFLAG = 0


# 定义一个函数 compute_gradient，计算梯度
# 参数说明：
# - val_P: 浮点数数组，存储 P 值
# - pos_reference: 浮点数二维数组，参考位置数组
# - neighbors: 64 位整数数组，存储邻居索引
# - indptr: 64 位整数数组，存储指针索引
# - tot_force: 浮点数二维数组，总力数组
# - qt: _QuadTree 对象，四叉树对象
# - theta: 浮点数，阈值参数
# - dof: 整数，自由度
# - start: 长整数，起始位置
# - compute_error: 布尔值，是否计算误差
# - num_threads: 整数，线程数目
# 返回值：浮点数，梯度值
cdef float compute_gradient(float[:] val_P,
                            float[:, :] pos_reference,
                            cnp.int64_t[:] neighbors,
                            cnp.int64_t[:] indptr,
                            float[:, :] tot_force,
                            _QuadTree qt,
                            float theta,
                            int dof,
                            long start,
                            bint compute_error,
                            int num_threads) noexcept nogil:
    # 创建四叉树后，计算梯度，包括正向和负向力量
    cdef:
        long i, coord
        int ax
        long n_samples = pos_reference.shape[0]
        int n_dimensions = qt.n_dimensions
        clock_t t1 = 0, t2 = 0
        double sQ
        float error
        int take_timing = 1 if qt.verbose > 15 else 0

    # 如果详细输出大于 11，则打印分配力量数组的信息
    if qt.verbose > 11:
        printf("[t-SNE] Allocating %li elements in force arrays\n",
               n_samples * n_dimensions * 2)

    # 分配负向力量数组和正向力量数组的内存空间
    cdef float* neg_f = <float*> malloc(sizeof(float) * n_samples * n_dimensions)
    cdef float* pos_f = <float*> malloc(sizeof(float) * n_samples * n_dimensions)

    # 如果需要计时，则记录开始时间
    if take_timing:
        t1 = clock()

    # 计算负向梯度，并返回 sQ 值
    sQ = compute_gradient_negative(pos_reference, neg_f, qt, dof, theta, start,
                                   num_threads)

    # 如果需要计时，则记录结束时间并打印计算负向梯度所用时间
    if take_timing:
        t2 = clock()
        printf("[t-SNE] Computing negative gradient: %e ticks\n", ((float) (t2 - t1)))

    # 如果需要计时，则记录开始时间
    if take_timing:
        t1 = clock()

    # 计算正向梯度，并返回误差值
    error = compute_gradient_positive(val_P, pos_reference, neighbors, indptr,
                                      pos_f, n_dimensions, dof, sQ, start,
                                      qt.verbose, compute_error, num_threads)
    # 如果需要记录时间信息
    if take_timing:
        # 记录当前时间 t2
        t2 = clock()
        # 输出计算正梯度所用的时间
        printf("[t-SNE] Computing positive gradient: %e ticks\n",
               ((float) (t2 - t1)))
    
    # 使用并行循环进行迭代，从 start 到 n_samples，不使用全局解锁（nogil=True），使用指定数量的线程
    for i in prange(start, n_samples, nogil=True, num_threads=num_threads,
                    schedule='static'):
        # 遍历每个坐标维度
        for ax in range(n_dimensions):
            # 计算当前坐标在 tot_force 中的索引
            coord = i * n_dimensions + ax
            # 计算正梯度，并将结果存储在 tot_force 中
            tot_force[i, ax] = pos_f[coord] - (neg_f[coord] / sQ)

    # 释放 neg_f 数组占用的内存
    free(neg_f)
    # 释放 pos_f 数组占用的内存
    free(pos_f)
    # 返回计算的误差值
    return error
# 定义一个 Cython 函数，计算 t-SNE 算法中的正梯度
cdef float compute_gradient_positive(float[:] val_P,
                                     float[:, :] pos_reference,
                                     cnp.int64_t[:] neighbors,
                                     cnp.int64_t[:] indptr,
                                     float* pos_f,
                                     int n_dimensions,
                                     int dof,
                                     double sum_Q,
                                     cnp.int64_t start,
                                     int verbose,
                                     bint compute_error,
                                     int num_threads) noexcept nogil:
    # 定义变量
    cdef:
        int ax  # 迭代变量，用于循环遍历每一个维度
        long i, j, k  # 迭代变量，用于循环遍历样本点和它的近邻
        long n_samples = indptr.shape[0] - 1  # 样本点数量，即指标指针数组的长度减一
        float C = 0.0  # 初始化误差值为零
        float dij, qij, pij  # 定义距离、概率和归一化概率
        float exponent = (dof + 1.0) / 2.0  # 计算幂指数
        float float_dof = (float) (dof)  # 将自由度转换为浮点数
        float* buff  # 定义缓冲区数组，存储计算中间结果
        clock_t t1 = 0, t2 = 0  # 定义时钟变量，用于计算执行时间
        float dt  # 执行时间差异

    if verbose > 10:
        t1 = clock()  # 获取开始时钟时间

    with nogil, parallel(num_threads=num_threads):
        # 分配私有缓冲区变量
        buff = <float *> malloc(sizeof(float) * n_dimensions)

        for i in prange(start, n_samples, schedule='static'):
            # 初始化梯度向量为零
            for ax in range(n_dimensions):
                pos_f[i * n_dimensions + ax] = 0.0
            # 计算近邻的正相互作用
            for k in range(indptr[i], indptr[i+1]):
                j = neighbors[k]
                dij = 0.0
                pij = val_P[k]
                for ax in range(n_dimensions):
                    buff[ax] = pos_reference[i, ax] - pos_reference[j, ax]
                    dij += buff[ax] * buff[ax]
                qij = float_dof / (float_dof + dij)
                if dof != 1:  # 如果自由度不等于1，则计算指数
                    qij = qij ** exponent
                dij = pij * qij

                # 仅在需要计算误差时执行
                if compute_error:
                    qij = qij / sum_Q
                    C += pij * log(max(pij, FLOAT32_TINY) / max(qij, FLOAT32_TINY))
                for ax in range(n_dimensions):
                    pos_f[i * n_dimensions + ax] += dij * buff[ax]

        free(buff)  # 释放缓冲区内存

    if verbose > 10:
        t2 = clock()  # 获取结束时钟时间
        dt = ((float) (t2 - t1))  # 计算执行时间
        printf("[t-SNE] Computed error=%1.4f in %1.1e ticks\n", C, dt)  # 打印执行时间和误差值

    return C  # 返回计算出的误差值
cdef double compute_gradient_negative(float[:, :] pos_reference,
                                      float* neg_f,
                                      _QuadTree qt,
                                      int dof,
                                      float theta,
                                      long start,
                                      int num_threads) noexcept nogil:
    # 定义函数 compute_gradient_negative，计算 t-SNE 的负梯度
    cdef:
        int ax  # 声明整型变量 ax
        int n_dimensions = qt.n_dimensions  # 获取 QuadTree 对象的维度数
        int offset = n_dimensions + 2  # 计算偏移量，QuadTree 中每个节点的数据项数量
        long i, j, idx  # 声明长整型变量 i, j, idx，用于迭代和索引
        long n_samples = pos_reference.shape[0]  # 获取参考位置数组的样本数
        long n = n_samples - start  # 计算需要处理的样本数
        long dta = 0  # 初始化时间测量变量 dta
        long dtb = 0  # 初始化时间测量变量 dtb
        float size, dist2s, mult  # 声明浮点数变量 size, dist2s, mult
        float exponent = (dof + 1.0) / 2.0  # 计算指数项
        float float_dof = (float) (dof)  # 将整数 dof 转换为浮点数
        double qijZ, sum_Q = 0.0  # 初始化双精度浮点数变量 qijZ 和 sum_Q
        float* force  # 声明浮点数指针 force
        float* neg_force  # 声明浮点数指针 neg_force
        float* pos  # 声明浮点数指针 pos
        clock_t t1 = 0, t2 = 0, t3 = 0  # 初始化时钟变量 t1, t2, t3
        int take_timing = 1 if qt.verbose > 20 else 0  # 根据 QuadTree 对象的详细程度设置是否进行时间测量

    with nogil, parallel(num_threads=num_threads):
        # 使用 nogil 和 parallel 指令进入无 GIL 线程并行环境
        # 分配线程本地缓冲区
        summary = <float*> malloc(sizeof(float) * n * offset)  # 分配 summary 数组的内存空间
        pos = <float *> malloc(sizeof(float) * n_dimensions)  # 分配 pos 数组的内存空间
        force = <float *> malloc(sizeof(float) * n_dimensions)  # 分配 force 数组的内存空间
        neg_force = <float *> malloc(sizeof(float) * n_dimensions)  # 分配 neg_force 数组的内存空间

        for i in prange(start, n_samples, schedule='static'):
            # 在并行区域内迭代处理样本数据
            # 清空数组
            for ax in range(n_dimensions):
                force[ax] = 0.0  # 初始化 force 数组元素为 0
                neg_force[ax] = 0.0  # 初始化 neg_force 数组元素为 0
                pos[ax] = pos_reference[i, ax]  # 从 pos_reference 中复制数据到 pos 数组

            # 确定哪些节点正在汇总并收集它们的质心、增量和大小到向量化数组中
            if take_timing:
                t1 = clock()  # 记录开始时间
            idx = qt.summarize(pos, summary, theta*theta)  # 调用 QuadTree 对象的 summarize 方法
            if take_timing:
                t2 = clock()  # 记录第二个时间点
            # 计算 t-SNE 的负力
            # 对于 digits 数据集，遍历树的成本大约是以下循环的 10-15 倍
            for j in range(idx // offset):
                dist2s = summary[j * offset + n_dimensions]  # 从 summary 中读取距离平方
                size = summary[j * offset + n_dimensions + 1]  # 从 summary 中读取节点大小
                qijZ = float_dof / (float_dof + dist2s)  # 计算 qijZ
                if dof != 1:  # 如果 dof 不等于 1，则使用指数项
                    qijZ = qijZ ** exponent  # 计算 qijZ 的指数项

                sum_Q += size * qijZ  # 计算 sum_Q 的贡献
                mult = size * qijZ * qijZ  # 计算 mult

                for ax in range(n_dimensions):
                    neg_force[ax] += mult * summary[j * offset + ax]  # 计算负力的贡献
            if take_timing:
                t3 = clock()  # 记录第三个时间点
            for ax in range(n_dimensions):
                neg_f[i * n_dimensions + ax] = neg_force[ax]  # 将负力写入 neg_f 数组
            if take_timing:
                dta += t2 - t1  # 累计时间 dta
                dtb += t3 - t2  # 累计时间 dtb

        free(pos)  # 释放 pos 数组的内存空间
        free(force)  # 释放 force 数组的内存空间
        free(neg_force)  # 释放 neg_force 数组的内存空间
        free(summary)  # 释放 summary 数组的内存空间
    # 如果需要记录时间信息
    if take_timing:
        # 打印 t-SNE 树构建过程中的时钟周期数
        printf("[t-SNE] Tree: %li clock ticks | ", dta)
        # 打印 t-SNE 强制计算过程中的时钟周期数
        printf("Force computation: %li clock ticks\n", dtb)

    # 将 sum_Q 设置为最大值和浮点数机器精度 EPSILON 中的较大者，以避免除以 0 的情况
    sum_Q = max(sum_Q, FLOAT64_EPS)
    # 返回更新后的 sum_Q
    return sum_Q
# 定义一个 C 函数，用于计算梯度，通过引用传递 forces 数组并在原地填充该数组
def gradient(float[:] val_P,                    # 输入参数：浮点数组 val_P
             float[:, :] pos_output,            # 输入参数：浮点二维数组 pos_output
             cnp.int64_t[:] neighbors,          # 输入参数：64 位整数数组 neighbors
             cnp.int64_t[:] indptr,             # 输入参数：64 位整数数组 indptr
             float[:, :] forces,                # 输入/输出参数：浮点二维数组 forces
             float theta,                       # 输入参数：浮点数 theta
             int n_dimensions,                  # 输入参数：整数 n_dimensions
             int verbose,                       # 输入参数：整数 verbose
             int dof=1,                         # 输入参数（默认值为 1）：整数 dof
             long skip_num_points=0,            # 输入参数（默认值为 0）：长整数 skip_num_points
             bint compute_error=1,              # 输入参数（默认值为 1）：布尔值 compute_error
             int num_threads=1):                # 输入参数（默认值为 1）：整数 num_threads
    # 此函数设计用于从外部 Python 调用
    cdef float C                               # 声明 C 类型的局部变量 C
    cdef int n                                 # 声明 C 类型的局部变量 n
    n = pos_output.shape[0]                    # 获取 pos_output 的行数赋值给 n
    assert val_P.itemsize == 4                  # 检查 val_P 的元素大小是否为 4
    assert pos_output.itemsize == 4             # 检查 pos_output 的元素大小是否为 4
    assert forces.itemsize == 4                 # 检查 forces 的元素大小是否为 4
    m = "Forces array and pos_output shapes are incompatible"  # 错误消息
    assert n == forces.shape[0], m              # 检查 forces 的行数是否与 n 相等，否则抛出错误消息 m
    m = "Pij and pos_output shapes are incompatible"  # 错误消息
    assert n == indptr.shape[0] - 1, m          # 检查 indptr 的行数减一是否与 n 相等，否则抛出错误消息 m
    if verbose > 10:
        printf("[t-SNE] Initializing tree of n_dimensions %i\n", n_dimensions)  # 打印消息：初始化 n_dimensions 维度的树
    cdef _QuadTree qt = _QuadTree(pos_output.shape[1], verbose)  # 创建 _QuadTree 对象 qt
    if verbose > 10:
        printf("[t-SNE] Inserting %li points\n", pos_output.shape[0])  # 打印消息：插入 pos_output.shape[0] 个点
    qt.build_tree(pos_output)                   # 构建树结构并将 pos_output 插入 qt
    if verbose > 10:
        # XXX: format hack to workaround lack of `const char *` type
        # in the generated C code that triggers error with gcc 4.9
        # and -Werror=format-security
        printf("[t-SNE] Computing gradient\n%s", EMPTY_STRING)  # 打印消息：计算梯度
    # 调用 compute_gradient 函数计算梯度，并将结果赋给 C
    C = compute_gradient(val_P, pos_output, neighbors, indptr, forces,
                         qt, theta, dof, skip_num_points, compute_error,
                         num_threads)
    if verbose > 10:
        # XXX: format hack to workaround lack of `const char *` type
        # in the generated C code
        # and -Werror=format-security
        printf("[t-SNE] Checking tree consistency\n%s", EMPTY_STRING)  # 打印消息：检查树的一致性
    m = "Tree consistency failed: unexpected number of points on the tree"  # 错误消息
    assert qt.cells[0].cumulative_size == qt.n_points, m  # 检查树的一致性，否则抛出错误消息 m
    if not compute_error:
        C = np.nan                              # 如果 compute_error 为假，则将 C 设为 NaN
    return C                                    # 返回 C
```
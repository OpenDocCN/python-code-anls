# `D:\src\scipysrc\scikit-learn\sklearn\preprocessing\_target_encoder_fast.pyx`

```
# 导入 `isnan` 函数，用于检查是否为 NaN（Not a Number）
from libc.math cimport isnan
# 导入 C++ 的 vector 容器，用于存储双精度浮点数指针
from libcpp.vector cimport vector

# 从 _typedefs 模块中导入特定类型的数据类型定义
from ..utils._typedefs cimport float32_t, float64_t, int32_t, int64_t

# 导入 numpy 库并用 np 别名表示
import numpy as np

# 定义 INT_DTYPE 和 Y_DTYPE 融合类型
ctypedef fused INT_DTYPE:
    int64_t
    int32_t

ctypedef fused Y_DTYPE:
    int64_t
    int32_t
    float64_t
    float32_t

# 定义函数 _fit_encoding_fast，用于快速拟合目标编码
def _fit_encoding_fast(
    INT_DTYPE[:, ::1] X_int,    # 整数类型的二维数组 X_int，表示输入特征
    const Y_DTYPE[:] y,         # 常量数组 y，表示目标变量
    int64_t[::1] n_categories,  # 整数类型的一维数组 n_categories，表示每个特征的类别数量
    double smooth,              # 双精度浮点数 smooth，表示平滑参数
    double y_mean,              # 双精度浮点数 y_mean，表示目标变量 y 的均值
):
    """Fit a target encoding on X_int and y.

    This implementation uses Eq 7 from [1] to compute the encoding.
    As stated in the paper, Eq 7 is the same as Eq 3.

    [1]: Micci-Barreca, Daniele. "A preprocessing scheme for high-cardinality
         categorical attributes in classification and prediction problems"
    """
    cdef:
        int64_t sample_idx, feat_idx, cat_idx, n_cats    # 定义 C 语言级别的变量
        INT_DTYPE X_int_tmp                             # 定义 C 语言级别的变量
        int n_samples = X_int.shape[0]                  # 获取 X_int 的样本数
        int n_features = X_int.shape[1]                 # 获取 X_int 的特征数
        double smooth_sum = smooth * y_mean             # 计算平滑参数乘以目标变量均值
        int64_t max_n_cats = np.max(n_categories)       # 获取 n_categories 的最大值
        double[::1] sums = np.empty(max_n_cats, dtype=np.float64)   # 定义和初始化 sums 数组
        double[::1] counts = np.empty(max_n_cats, dtype=np.float64) # 定义和初始化 counts 数组
        list encodings = []                            # 定义空列表 encodings
        double[::1] current_encoding                   # 定义当前编码数组
        # Gives access to encodings without gil          # 注释，无需 GIL（全局解释器锁）访问编码

        vector[double*] encoding_vec                   # 定义双精度浮点数指针的 vector 容器

    encoding_vec.resize(n_features)                    # 调整 encoding_vec 的大小为特征数

    # 循环遍历每个特征索引
    for feat_idx in range(n_features):
        current_encoding = np.empty(shape=n_categories[feat_idx], dtype=np.float64)  # 创建当前特征的编码数组
        encoding_vec[feat_idx] = &current_encoding[0]  # 将当前编码数组的指针存入 encoding_vec
        encodings.append(np.asarray(current_encoding))  # 将当前编码数组转换为 NumPy 数组并添加到 encodings 中

    # 使用 nogil 块进行无 GIL 计算
    with nogil:
        # 循环遍历每个特征索引
        for feat_idx in range(n_features):
            n_cats = n_categories[feat_idx]            # 获取当前特征的类别数量

            # 初始化 sums 和 counts 数组
            for cat_idx in range(n_cats):
                sums[cat_idx] = smooth_sum              # 初始化 sums
                counts[cat_idx] = smooth                # 初始化 counts

            # 循环遍历每个样本索引
            for sample_idx in range(n_samples):
                X_int_tmp = X_int[sample_idx, feat_idx] # 获取当前样本在当前特征下的整数值
                # 如果整数值为 -1，表示未知类别，跳过
                if X_int_tmp == -1:
                    continue
                sums[X_int_tmp] += y[sample_idx]        # 累加目标变量值到 sums
                counts[X_int_tmp] += 1.0                # 累加计数器

            # 计算每个类别的目标编码值
            for cat_idx in range(n_cats):
                if counts[cat_idx] == 0:
                    encoding_vec[feat_idx][cat_idx] = y_mean  # 若计数为 0，则使用目标变量均值
                else:
                    encoding_vec[feat_idx][cat_idx] = sums[cat_idx] / counts[cat_idx]  # 计算目标编码值

    return encodings  # 返回编码结果的列表


# 定义函数 _fit_encoding_fast_auto_smooth，用于自动平滑拟合目标编码
def _fit_encoding_fast_auto_smooth(
    INT_DTYPE[:, ::1] X_int,        # 整数类型的二维数组 X_int，表示输入特征
    const Y_DTYPE[:] y,             # 常量数组 y，表示目标变量
    int64_t[::1] n_categories,      # 整数类型的一维数组 n_categories，表示每个特征的类别数量
    double y_mean,                  # 双精度浮点数 y_mean，表示目标变量 y 的均值
    double y_variance,              # 双精度浮点数 y_variance，表示目标变量 y 的方差
):
    """Fit a target encoding on X_int and y with auto smoothing.

    This implementation uses Eq 5 and 6 from [1].

    [1]: Micci-Barreca, Daniele. "A preprocessing scheme for high-cardinality
         categorical attributes in classification and prediction problems"
    """
    cdef:
        # 定义变量：样本索引，特征索引，类别索引，类别数
        int64_t sample_idx, feat_idx, cat_idx, n_cats
        # 定义整数数据类型 X_int_tmp
        INT_DTYPE X_int_tmp
        # 定义变量：差异
        double diff
        # 初始化变量：样本数为 X_int 的行数
        int n_samples = X_int.shape[0]
        # 初始化变量：特征数为 X_int 的列数
        int n_features = X_int.shape[1]
        # 计算最大类别数，并将其设为 max_n_cats
        int64_t max_n_cats = np.max(n_categories)
        # 创建空数组：存放每个类别的均值
        double[::1] means = np.empty(max_n_cats, dtype=np.float64)
        # 创建空数组：存放每个类别的计数
        int64_t[::1] counts = np.empty(max_n_cats, dtype=np.int64)
        # 创建空数组：存放每个类别的平方差之和
        double[::1] sum_of_squared_diffs = np.empty(max_n_cats, dtype=np.float64)
        # 定义变量：lambda_
        double lambda_
        # 初始化列表：存放编码
        list encodings = []
        # 创建空数组：当前编码
        double[::1] current_encoding
        # 定义注释：不使用 GIL 获取 encodings 的访问权
        # 定义向量：编码向量
        vector[double*] encoding_vec

    # 调整编码向量的大小为特征数
    encoding_vec.resize(n_features)
    # 对于每个特征索引
    for feat_idx in range(n_features):
        # 创建空的当前编码数组，形状为该特征的类别数
        current_encoding = np.empty(shape=n_categories[feat_idx], dtype=np.float64)
        # 将当前编码数组的地址赋给编码向量对应特征索引的位置
        encoding_vec[feat_idx] = &current_encoding[0]
        # 将当前编码数组添加到编码列表中
        encodings.append(np.asarray(current_encoding))

    # TODO: 使用 OpenMP prange 并行化此部分。当特征数大于等于线程数时，
    # 可以考虑并行化外部循环。当特征数较小时，可能更好地并行化内部循环，
    # 在样本数和类别数上进行。但是处理线程本地临时变量的代码可能会更加复杂。
    # 使用 nogil 上下文，表示以下代码段将不会受到 GIL（全局解释器锁）的限制
    with nogil:
        # 遍历特征的索引范围
        for feat_idx in range(n_features):
            # 获取当前特征索引下的类别数
            n_cats = n_categories[feat_idx]

            # 初始化均值、计数和平方差总和数组
            for cat_idx in range(n_cats):
                means[cat_idx] = 0.0
                counts[cat_idx] = 0
                sum_of_squared_diffs[cat_idx] = 0.0

            # 第一次遍历计算均值
            for sample_idx in range(n_samples):
                # 获取当前样本在当前特征索引下的整数表示
                X_int_tmp = X_int[sample_idx, feat_idx]

                # 如果整数表示为 -1，则表示未知类别，跳过此样本
                if X_int_tmp == -1:
                    continue

                # 更新计数和均值数组
                counts[X_int_tmp] += 1
                means[X_int_tmp] += y[sample_idx]

            # 计算每个类别的均值
            for cat_idx in range(n_cats):
                means[cat_idx] /= counts[cat_idx]

            # 第二次遍历计算平方差总和
            for sample_idx in range(n_samples):
                X_int_tmp = X_int[sample_idx, feat_idx]

                # 如果整数表示为 -1，则表示未知类别，跳过此样本
                if X_int_tmp == -1:
                    continue

                # 计算当前样本的预测值与均值的差值，并更新平方差总和数组
                diff = y[sample_idx] - means[X_int_tmp]
                sum_of_squared_diffs[X_int_tmp] += diff * diff

            # 计算每个类别的权重 lambda
            for cat_idx in range(n_cats):
                lambda_ = (
                    y_variance * counts[cat_idx] /
                    (y_variance * counts[cat_idx] + sum_of_squared_diffs[cat_idx] /
                     counts[cat_idx])
                )

                # 处理 lambda 为 NaN 的情况
                if isnan(lambda_):
                    # 如果 lambda 为 NaN，则使用 y_mean 作为编码向量的值
                    encoding_vec[feat_idx][cat_idx] = y_mean
                else:
                    # 计算编码向量的值，根据加权均值公式计算
                    encoding_vec[feat_idx][cat_idx] = (
                        lambda_ * means[cat_idx] + (1 - lambda_) * y_mean
                    )

    # 返回编码向量结果
    return encodings
```
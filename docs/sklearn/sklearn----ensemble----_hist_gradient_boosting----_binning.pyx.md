# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\_binning.pyx`

```
# Author: Nicolas Hug

# 导入必要的模块和函数
from cython.parallel import prange
from libc.math cimport isnan

# 从自定义的模块中导入特定数据类型
from .common cimport X_DTYPE_C, X_BINNED_DTYPE_C


def _map_to_bins(const X_DTYPE_C [:, :] data,
                 list binning_thresholds,
                 const unsigned char[::1] is_categorical,
                 const unsigned char missing_values_bin_idx,
                 int n_threads,
                 X_BINNED_DTYPE_C [::1, :] binned):
    """Bin continuous and categorical values to discrete integer-coded levels.

    A given value x is mapped into bin value i iff
    thresholds[i - 1] < x <= thresholds[i]

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        The data to bin.
    binning_thresholds : list of arrays
        For each feature, stores the increasing numeric values that are
        used to separate the bins.
    is_categorical : ndarray of unsigned char of shape (n_features,)
        Indicates categorical features.
    n_threads : int
        Number of OpenMP threads to use.
    binned : ndarray, shape (n_samples, n_features)
        Output array, must be fortran aligned.
    """
    # 声明局部变量 feature_idx 作为特征索引
    cdef:
        int feature_idx

    # 循环遍历数据的每一个特征
    for feature_idx in range(data.shape[1]):
        # 调用内部函数，将每一列数据映射到对应的分箱中
        _map_col_to_bins(
            data[:, feature_idx],               # 当前特征的数据列
            binning_thresholds[feature_idx],    # 当前特征的分箱阈值
            is_categorical[feature_idx],        # 当前特征是否为分类特征的标志
            missing_values_bin_idx,             # 缺失值对应的分箱索引
            n_threads,                         # 使用的线程数
            binned[:, feature_idx]             # 输出的分箱结果数组对应的列
        )


# 使用Cython定义的函数，用于对单列数据进行分箱操作
cdef void _map_col_to_bins(
    const X_DTYPE_C [:] data,
    const X_DTYPE_C [:] binning_thresholds,
    const unsigned char is_categorical,
    const unsigned char missing_values_bin_idx,
    int n_threads,
    X_BINNED_DTYPE_C [:] binned
):
    """Binary search to find the bin index for each value in the data."""
    # 声明局部变量
    cdef:
        int i         # 数据索引
        int left      # 二分查找的左边界
        int right     # 二分查找的右边界
        int middle    # 二分查找的中间位置

    # 使用并行循环对数据进行遍历，使用静态调度，禁用GIL
    for i in prange(data.shape[0], schedule='static', nogil=True,
                    num_threads=n_threads):
        # 判断数据是否为NaN或者负数的分类特征值（按照LightGBM的惯例）
        if (
            isnan(data[i]) or
            # To follow LightGBM's conventions, negative values for
            # categorical features are considered as missing values.
            (is_categorical and data[i] < 0)
        ):
            # 如果是NaN或者负数分类特征值，将其映射到缺失值对应的分箱索引
            binned[i] = missing_values_bin_idx
        else:
            # 对于已知的数值，使用二分查找找到对应的分箱索引
            left, right = 0, binning_thresholds.shape[0]
            while left < right:
                # 计算中间位置，避免溢出
                middle = left + (right - left - 1) // 2
                if data[i] <= binning_thresholds[middle]:
                    right = middle
                else:
                    left = middle + 1

            # 将找到的分箱索引赋值给输出数组
            binned[i] = left
```
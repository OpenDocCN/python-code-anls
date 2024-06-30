# `D:\src\scipysrc\scikit-learn\sklearn\utils\stats.py`

```
import numpy as np  # 导入NumPy库，用于数值计算

from .extmath import stable_cumsum  # 导入外部函数stable_cumsum，用于稳定累加求和


def _weighted_percentile(array, sample_weight, percentile=50):
    """Compute weighted percentile

    Computes lower weighted percentile. If `array` is a 2D array, the
    `percentile` is computed along the axis 0.

        .. versionchanged:: 0.24
            Accepts 2D `array`.

    Parameters
    ----------
    array : 1D or 2D array
        Values to take the weighted percentile of.

    sample_weight: 1D or 2D array
        Weights for each value in `array`. Must be same shape as `array` or
        of shape `(array.shape[0],)`.

    percentile: int or float, default=50
        Percentile to compute. Must be value between 0 and 100.

    Returns
    -------
    percentile : int if `array` 1D, ndarray if `array` 2D
        Weighted percentile.
    """
    n_dim = array.ndim  # 获取数组的维度

    if n_dim == 0:
        return array[()]  # 如果数组是0维，则直接返回数组的元素作为加权百分位数

    if array.ndim == 1:
        array = array.reshape((-1, 1))  # 如果数组是1维，则将其转换为列向量形式

    # 当sample_weight是1维时，对每个array.shape[1]重复使用它
    if array.shape != sample_weight.shape and array.shape[0] == sample_weight.shape[0]:
        sample_weight = np.tile(sample_weight, (array.shape[1], 1)).T  # 将sample_weight广播到与array相同的形状

    sorted_idx = np.argsort(array, axis=0)  # 沿着axis=0对数组进行排序，并返回排序后的索引
    sorted_weights = np.take_along_axis(sample_weight, sorted_idx, axis=0)  # 按照排序后的索引获取对应的权重值

    # 计算加权累积分布函数（CDF）
    weight_cdf = stable_cumsum(sorted_weights, axis=0)

    adjusted_percentile = percentile / 100 * weight_cdf[-1]  # 根据加权累积分布函数计算调整后的百分位数

    # 处理百分位数为0的情况，忽略权重为0的前导观测值
    mask = adjusted_percentile == 0
    adjusted_percentile[mask] = np.nextafter(
        adjusted_percentile[mask], adjusted_percentile[mask] + 1
    )

    # 查找每个样本的百分位数索引
    percentile_idx = np.array(
        [
            np.searchsorted(weight_cdf[:, i], adjusted_percentile[i])
            for i in range(weight_cdf.shape[1])
        ]
    )
    percentile_idx = np.array(percentile_idx)

    # 处理罕见情况下，percentile_idx等于sorted_idx.shape[0]
    max_idx = sorted_idx.shape[0] - 1
    percentile_idx = np.apply_along_axis(
        lambda x: np.clip(x, 0, max_idx), axis=0, arr=percentile_idx
    )

    col_index = np.arange(array.shape[1])  # 创建列索引数组
    percentile_in_sorted = sorted_idx[percentile_idx, col_index]  # 获取排序后的百分位数在原始数组中的索引
    percentile = array[percentile_in_sorted, col_index]  # 根据索引获取百分位数的值
    return percentile[0] if n_dim == 1 else percentile  # 返回计算的加权百分位数，如果数组是1维，则返回一个值，否则返回数组
```
# `D:\src\scipysrc\scikit-learn\sklearn\metrics\cluster\_bicluster.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
from scipy.optimize import linear_sum_assignment  # 导入线性求和匹配算法

from ...utils._param_validation import StrOptions, validate_params  # 导入参数验证相关模块
from ...utils.validation import check_array, check_consistent_length  # 导入数组验证和长度一致性检查函数

__all__ = ["consensus_score"]  # 指定在使用 from ... import * 时导入的模块列表


def _check_rows_and_columns(a, b):
    """Unpacks the row and column arrays and checks their shape."""
    check_consistent_length(*a)  # 检查参数a中的元素长度是否一致
    check_consistent_length(*b)  # 检查参数b中的元素长度是否一致
    checks = lambda x: check_array(x, ensure_2d=False)  # 定义一个lambda函数，用于验证数组x是否符合要求
    a_rows, a_cols = map(checks, a)  # 对a中的元素分别调用checks函数，得到行和列数组
    b_rows, b_cols = map(checks, b)  # 对b中的元素分别调用checks函数，得到行和列数组
    return a_rows, a_cols, b_rows, b_cols  # 返回验证后的行和列数组


def _jaccard(a_rows, a_cols, b_rows, b_cols):
    """Jaccard coefficient on the elements of the two biclusters."""
    intersection = (a_rows * b_rows).sum() * (a_cols * b_cols).sum()  # 计算两个二双聚类元素的交集

    a_size = a_rows.sum() * a_cols.sum()  # 计算a的大小
    b_size = b_rows.sum() * b_cols.sum()  # 计算b的大小

    return intersection / (a_size + b_size - intersection)  # 返回Jaccard系数


def _pairwise_similarity(a, b, similarity):
    """Computes pairwise similarity matrix.

    result[i, j] is the Jaccard coefficient of a's bicluster i and b's
    bicluster j.

    """
    a_rows, a_cols, b_rows, b_cols = _check_rows_and_columns(a, b)  # 检查并获取行和列数组
    n_a = a_rows.shape[0]  # 获取a的行数
    n_b = b_rows.shape[0]  # 获取b的行数
    result = np.array(
        [
            [similarity(a_rows[i], a_cols[i], b_rows[j], b_cols[j]) for j in range(n_b)]  # 计算a的第i个双聚类与b的第j个双聚类的相似度
            for i in range(n_a)  # 遍历a的双聚类
        ]
    )
    return result  # 返回相似度矩阵


@validate_params(
    {
        "a": [tuple],  # 参数a是一个元组
        "b": [tuple],  # 参数b是一个元组
        "similarity": [callable, StrOptions({"jaccard"})],  # 参数similarity是可调用对象或者字符串"jaccard"
    },
    prefer_skip_nested_validation=True,
)
def consensus_score(a, b, *, similarity="jaccard"):
    """The similarity of two sets of biclusters.

    Similarity between individual biclusters is computed. Then the
    best matching between sets is found using the Hungarian algorithm.
    The final score is the sum of similarities divided by the size of
    the larger set.

    Read more in the :ref:`User Guide <biclustering>`.

    Parameters
    ----------
    a : tuple (rows, columns)
        Tuple of row and column indicators for a set of biclusters.

    b : tuple (rows, columns)
        Another set of biclusters like ``a``.

    similarity : 'jaccard' or callable, default='jaccard'
        May be the string "jaccard" to use the Jaccard coefficient, or
        any function that takes four arguments, each of which is a 1d
        indicator vector: (a_rows, a_columns, b_rows, b_columns).

    Returns
    -------
    consensus_score : float
       Consensus score, a non-negative value, sum of similarities
       divided by size of larger set.

    References
    ----------

    * Hochreiter, Bodenhofer, et. al., 2010. `FABIA: factor analysis
      for bicluster acquisition
      <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2881408/>`__.

    Examples
    --------
    >>> from sklearn.metrics import consensus_score
    >>> a = ([[True, False], [False, True]], [[False, True], [True, False]])
    """
    # 如果相似度度量选择的是 Jaccard 相似度，则将相似度函数指定为 _jaccard
    if similarity == "jaccard":
        similarity = _jaccard
    # 计算输入矩阵 a 和 b 的相似度矩阵，根据指定的相似度函数
    matrix = _pairwise_similarity(a, b, similarity)
    # 使用匈牙利算法找到最大匹配，即最大化相似度
    row_indices, col_indices = linear_sum_assignment(1.0 - matrix)
    # 计算矩阵 a 和 b 中每个成员的数量
    n_a = len(a[0])
    n_b = len(b[0])
    # 返回最大匹配中的相似度得分，该得分是匹配对中所有元素相似度的加权平均
    return matrix[row_indices, col_indices].sum() / max(n_a, n_b)
```
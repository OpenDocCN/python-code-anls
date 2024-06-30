# `D:\src\scipysrc\scikit-learn\sklearn\utils\_mask.py`

```
# 导入必要的模块和函数，使用了 contextlib.suppress 来忽略 ImportError 和 AttributeError 异常
from contextlib import suppress

# 导入 numpy 库并简称为 np
import numpy as np
# 导入 scipy.sparse 库并简称为 sp
from scipy import sparse as sp

# 从本地模块中导入特定函数和类
from ._missing import is_scalar_nan
from ._param_validation import validate_params
from .fixes import _object_dtype_isnan


def _get_dense_mask(X, value_to_mask):
    # 在上下文中尝试导入 pandas 并忽略 ImportError 和 AttributeError 异常
    with suppress(ImportError, AttributeError):
        import pandas

        # 如果 value_to_mask 是 pandas.NA，则使用 pandas.isna() 创建布尔掩码
        if value_to_mask is pandas.NA:
            return pandas.isna(X)

    # 如果 value_to_mask 是标量 NaN
    if is_scalar_nan(value_to_mask):
        # 如果 X 的 dtype 是浮点型
        if X.dtype.kind == "f":
            Xt = np.isnan(X)
        # 如果 X 的 dtype 是整数型或无符号整数型
        elif X.dtype.kind in ("i", "u"):
            # 整数型数组不可能有 NaN，返回全 False 的布尔数组
            Xt = np.zeros(X.shape, dtype=bool)
        else:
            # 对象类型的数组无法使用 np.isnan()，使用自定义函数检测 NaN
            Xt = _object_dtype_isnan(X)
    else:
        # 使用 X == value_to_mask 创建布尔掩码
        Xt = X == value_to_mask

    return Xt


def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == value_to_mask.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Input data, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    value_to_mask : {int, float}
        The value which is to be masked in X.

    Returns
    -------
    X_mask : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Missing mask.
    """
    # 如果 X 不是稀疏矩阵
    if not sp.issparse(X):
        # 直接调用 _get_dense_mask() 处理稠密数据
        return _get_dense_mask(X, value_to_mask)

    # 将 X.data 和 value_to_mask 传递给 _get_dense_mask() 处理
    Xt = _get_dense_mask(X.data, value_to_mask)

    # 根据原始稀疏矩阵 X 的格式创建新的稀疏矩阵
    sparse_constructor = sp.csr_matrix if X.format == "csr" else sp.csc_matrix
    Xt_sparse = sparse_constructor(
        (Xt, X.indices.copy(), X.indptr.copy()), shape=X.shape, dtype=bool
    )

    return Xt_sparse


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "mask": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def safe_mask(X, mask):
    """Return a mask which is safe to use on X.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        Data on which to apply mask.

    mask : array-like
        Mask to be used on X.

    Returns
    -------
    mask : ndarray
        Array that is safe to use on X.

    Examples
    --------
    >>> from sklearn.utils import safe_mask
    >>> from scipy.sparse import csr_matrix
    >>> data = csr_matrix([[1], [2], [3], [4], [5]])
    >>> condition = [False, True, True, False, True]
    >>> mask = safe_mask(data, condition)
    >>> data[mask].toarray()
    array([[2],
           [3],
           [5]])
    """
    # 将 mask 转换为 ndarray 格式
    mask = np.asarray(mask)
    
    # 如果 mask 的 dtype 是有符号整数类型，直接返回 mask
    if np.issubdtype(mask.dtype, np.signedinteger):
        return mask

    # 如果 X 有方法 toarray()，创建索引并使用索引 mask
    if hasattr(X, "toarray"):
        ind = np.arange(mask.shape[0])
        mask = ind[mask]
    return mask


def axis0_safe_slice(X, mask, len_mask):
    """Return a mask which is safer to use on X than safe_mask.
    """
    This mask is safer than safe_mask since it returns an
    empty array, when a sparse matrix is sliced with a boolean mask
    with all False, instead of raising an unhelpful error in older
    versions of SciPy.
    
    See: https://github.com/scipy/scipy/issues/5361
    
    Also note that we can avoid doing the dot product by checking if
    the len_mask is not zero in _huber_loss_and_gradient but this
    is not going to be the bottleneck, since the number of outliers
    and non_outliers are typically non-zero and it makes the code
    tougher to follow.
    
    Parameters
    ----------
    X : {array-like, sparse matrix}
        Data on which to apply mask.
    
    mask : ndarray
        Mask to be used on X.
    
    len_mask : int
        The length of the mask.
    
    Returns
    -------
    mask : ndarray
        Array that is safe to use on X.
    """
    # 如果 len_mask 不为零，则返回对 X 应用 safe_mask 的结果
    if len_mask != 0:
        return X[safe_mask(X, mask), :]
    # 如果 len_mask 为零，则返回一个形状为 (0, X.shape[1]) 的零数组
    return np.zeros(shape=(0, X.shape[1]))
# 将索引列表转换为布尔掩码数组的函数

def indices_to_mask(indices, mask_length):
    """Convert list of indices to boolean mask.

    Parameters
    ----------
    indices : list-like
        List of integers treated as indices.
    mask_length : int
        Length of boolean mask to be generated.
        This parameter must be greater than max(indices).

    Returns
    -------
    mask : 1d boolean nd-array
        Boolean array that is True where indices are present, else False.

    Examples
    --------
    >>> from sklearn.utils._mask import indices_to_mask
    >>> indices = [1, 2 , 3, 4]
    >>> indices_to_mask(indices, 5)
    array([False,  True,  True,  True,  True])
    """
    # 检查 mask_length 是否大于 indices 中的最大值，否则引发 ValueError 异常
    if mask_length <= np.max(indices):
        raise ValueError("mask_length must be greater than max(indices)")

    # 创建一个长度为 mask_length 的布尔数组，初始值为 False
    mask = np.zeros(mask_length, dtype=bool)
    # 将 indices 中的位置设为 True
    mask[indices] = True

    # 返回生成的布尔数组作为结果
    return mask
```
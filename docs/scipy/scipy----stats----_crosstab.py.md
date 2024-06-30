# `D:\src\scipysrc\scipy\scipy\stats\_crosstab.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
from scipy.sparse import coo_matrix  # 导入 scipy.sparse 库中的 coo_matrix 类
from scipy._lib._bunch import _make_tuple_bunch  # 导入 scipy 库中的 _make_tuple_bunch 函数

# 定义一个命名元组 CrosstabResult，包含 elements 和 count 两个属性
CrosstabResult = _make_tuple_bunch(
    "CrosstabResult", ["elements", "count"]
)

def crosstab(*args, levels=None, sparse=False):
    """
    Return table of counts for each possible unique combination in ``*args``.

    When ``len(args) > 1``, the array computed by this function is
    often referred to as a *contingency table* [1]_.

    The arguments must be sequences with the same length.  The second return
    value, `count`, is an integer array with ``len(args)`` dimensions.  If
    `levels` is None, the shape of `count` is ``(n0, n1, ...)``, where ``nk``
    is the number of unique elements in ``args[k]``.

    Parameters
    ----------
    *args : sequences
        A sequence of sequences whose unique aligned elements are to be
        counted.  The sequences in args must all be the same length.
    levels : sequence, optional
        If `levels` is given, it must be a sequence that is the same length as
        `args`.  Each element in `levels` is either a sequence or None.  If it
        is a sequence, it gives the values in the corresponding sequence in
        `args` that are to be counted.  If any value in the sequences in `args`
        does not occur in the corresponding sequence in `levels`, that value
        is ignored and not counted in the returned array `count`.  The default
        value of `levels` for ``args[i]`` is ``np.unique(args[i])``
    sparse : bool, optional
        If True, return a sparse matrix.  The matrix will be an instance of
        the `scipy.sparse.coo_matrix` class.  Because SciPy's sparse matrices
        must be 2-d, only two input sequences are allowed when `sparse` is
        True.  Default is False.

    Returns
    -------
    res : CrosstabResult
        An object containing the following attributes:

        elements : tuple of numpy.ndarrays.
            Tuple of length ``len(args)`` containing the arrays of elements
            that are counted in `count`.  These can be interpreted as the
            labels of the corresponding dimensions of `count`. If `levels` was
            given, then if ``levels[i]`` is not None, ``elements[i]`` will
            hold the values given in ``levels[i]``.
        count : numpy.ndarray or scipy.sparse.coo_matrix
            Counts of the unique elements in ``zip(*args)``, stored in an
            array. Also known as a *contingency table* when ``len(args) > 1``.

    See Also
    --------
    numpy.unique

    Notes
    -----
    .. versionadded:: 1.7.0

    References
    ----------
    .. [1] "Contingency table", http://en.wikipedia.org/wiki/Contingency_table

    Examples
    --------
    >>> from scipy.stats.contingency import crosstab

    Given the lists `a` and `x`, create a contingency table that counts the
    frequencies of the corresponding pairs.

    >>> a = ['A', 'B', 'A', 'A', 'B', 'B', 'A', 'A', 'B', 'B']
    """
    # 如果 levels 未指定，对每个参数序列调用 np.unique() 函数获取唯一值
    if levels is None:
        levels = [np.unique(arg) for arg in args]

    # 如果 sparse 为 True，则返回稀疏矩阵 scipy.sparse.coo_matrix
    if sparse:
        # 检查是否有超过两个输入序列，因为稀疏矩阵要求只能有两个维度
        if len(args) > 2:
            raise ValueError("Sparse output requires at most two input sequences")
        # 创建稀疏矩阵并返回
        return CrosstabResult(elements=tuple(levels), count=coo_matrix(np.histogram2d(*args, bins=[level.size for level in levels])[0]))
    
    # 返回普通的 numpy.ndarray
    return CrosstabResult(elements=tuple(levels), count=np.histogramdd(args, bins=[level.size for level in levels])[0])
    # 获取参数列表的长度
    nargs = len(args)
    # 如果参数数量为0，则抛出类型错误异常
    if nargs == 0:
        raise TypeError("At least one input sequence is required.")

    # 获取第一个参数的长度
    len0 = len(args[0])
    # 检查所有输入序列是否具有相同的长度
    if not all(len(a) == len0 for a in args[1:]):
        raise ValueError("All input sequences must have the same length.")

    # 当 sparse 参数为 True 时，只允许两个输入序列
    if sparse and nargs != 2:
        raise ValueError("When `sparse` is True, only two input sequences "
                         "are allowed.")


这段代码片段的作用是对输入的序列进行验证和处理，确保输入的序列满足要求，例如长度相同以及对 sparse 参数的限制。
    if levels is None:
        # 如果 levels 参数为 None，则对每个参数调用 np.unique，并返回其唯一值及对应的反向索引
        actual_levels, indices = zip(*[np.unique(a, return_inverse=True)
                                       for a in args])
    else:
        # 如果 levels 参数不为 None...
        if len(levels) != nargs:
            # 如果 levels 列表的长度不等于参数 nargs 的个数，抛出数值错误异常
            raise ValueError('len(levels) must equal the number of input '
                             'sequences')

        # 将 args 中的每个元素转换为 ndarray
        args = [np.asarray(arg) for arg in args]
        # 创建一个形状为 (nargs, len0) 的布尔型零数组 mask 和整数型零数组 inv
        mask = np.zeros((nargs, len0), dtype=np.bool_)
        inv = np.zeros((nargs, len0), dtype=np.intp)
        # 初始化 actual_levels 列表
        actual_levels = []
        # 遍历 levels 和 args 的每对元素，k 为索引，levels_list 是 levels 中的元素，arg 是 args 中的元素
        for k, (levels_list, arg) in enumerate(zip(levels, args)):
            if levels_list is None:
                # 如果 levels_list 为 None，则对 arg 调用 np.unique，并返回其唯一值和反向索引
                levels_list, inv[k, :] = np.unique(arg, return_inverse=True)
                # 设置 mask 的第 k 行为 True
                mask[k, :] = True
            else:
                # 否则，将 arg 与 levels_list 进行比较，将结果保存在 q 中
                q = arg == np.asarray(levels_list).reshape(-1, 1)
                # 设置 mask 的第 k 行为任一列为 True 的结果
                mask[k, :] = np.any(q, axis=0)
                # 获取非零元素的索引，并将结果保存在 inv 的第 k 行上
                qnz = q.T.nonzero()
                inv[k, qnz[0]] = qnz[1]
            # 将 levels_list 添加到 actual_levels 中
            actual_levels.append(levels_list)

        # 判断 mask 是否全为 True，并将结果保存在 mask_all 中
        mask_all = mask.all(axis=0)
        # 将 inv 中的元素组成元组 indices
        indices = tuple(inv[:, mask_all])

    if sparse:
        # 如果 sparse 为 True，创建一个 COO 格式的稀疏矩阵 count
        count = coo_matrix((np.ones(len(indices[0]), dtype=int),
                            (indices[0], indices[1])))
        # 合并重复的条目
        count.sum_duplicates()
    else:
        # 如果 sparse 不为 True，创建一个形状为 actual_levels 的零数组 count
        shape = [len(u) for u in actual_levels]
        count = np.zeros(shape, dtype=int)
        # 在 count 中累加 indices 处的值加 1
        np.add.at(count, indices, 1)

    # 返回交叉表结果，包括 actual_levels 和 count
    return CrosstabResult(actual_levels, count)
```
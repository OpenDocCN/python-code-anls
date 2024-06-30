# `D:\src\scipysrc\scipy\scipy\sparse\_matrix_io.py`

```
# 导入 NumPy 库并重命名为 np
import numpy as np
# 导入 SciPy 库并重命名为 sp
import scipy as sp

# 声明模块的公共接口，仅包括 save_npz 和 load_npz 两个函数
__all__ = ['save_npz', 'load_npz']

# 设置 pickle 操作的参数，禁止允许恶意输入
PICKLE_KWARGS = dict(allow_pickle=False)


def save_npz(file, matrix, compressed=True):
    """ Save a sparse matrix or array to a file using ``.npz`` format.

    Parameters
    ----------
    file : str or file-like object
        Either the file name (string) or an open file (file-like object)
        where the data will be saved. If file is a string, the ``.npz``
        extension will be appended to the file name if it is not already
        there.
    matrix: spmatrix or sparray
        The sparse matrix or array to save.
        Supported formats: ``csc``, ``csr``, ``bsr``, ``dia`` or ``coo``.
    compressed : bool, optional
        Allow compressing the file. Default: True

    See Also
    --------
    scipy.sparse.load_npz: Load a sparse matrix from a file using ``.npz`` format.
    numpy.savez: Save several arrays into a ``.npz`` archive.
    numpy.savez_compressed : Save several arrays into a compressed ``.npz`` archive.

    Examples
    --------
    Store sparse matrix to disk, and load it again:

    >>> import numpy as np
    >>> import scipy as sp
    >>> sparse_matrix = sp.sparse.csc_matrix([[0, 0, 3], [4, 0, 0]])
    >>> sparse_matrix
    <Compressed Sparse Column sparse matrix of dtype 'int64'
        with 2 stored elements and shape (2, 3)>
    >>> sparse_matrix.toarray()
    array([[0, 0, 3],
           [4, 0, 0]], dtype=int64)

    >>> sp.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)
    >>> sparse_matrix = sp.sparse.load_npz('/tmp/sparse_matrix.npz')

    >>> sparse_matrix
    <Compressed Sparse Column sparse matrix of dtype 'int64'
        with 2 stored elements and shape (2, 3)>
    >>> sparse_matrix.toarray()
    array([[0, 0, 3],
           [4, 0, 0]], dtype=int64)
    """
    # 创建一个空字典，用于存储需要保存的数据
    arrays_dict = {}
    # 根据矩阵的不同格式，更新字典中的内容
    if matrix.format in ('csc', 'csr', 'bsr'):
        arrays_dict.update(indices=matrix.indices, indptr=matrix.indptr)
    elif matrix.format == 'dia':
        arrays_dict.update(offsets=matrix.offsets)
    elif matrix.format == 'coo':
        arrays_dict.update(row=matrix.row, col=matrix.col)
    else:
        # 如果矩阵格式不支持保存，则抛出 NotImplementedError 异常
        msg = f'Save is not implemented for sparse matrix of format {matrix.format}.'
        raise NotImplementedError(msg)
    # 添加矩阵的格式、形状、数据到字典中
    arrays_dict.update(
        format=matrix.format.encode('ascii'),
        shape=matrix.shape,
        data=matrix.data
    )
    # 如果矩阵是 SciPy 的稀疏数组，则设置 _is_array=True
    if isinstance(matrix, sp.sparse.sparray):
        arrays_dict.update(_is_array=True)
    # 根据 compressed 参数决定是使用 np.savez_compressed 还是 np.savez 保存数据
    if compressed:
        np.savez_compressed(file, **arrays_dict)
    else:
        np.savez(file, **arrays_dict)


def load_npz(file):
    """ Load a sparse array/matrix from a file using ``.npz`` format.

    Parameters
    ----------
    file : str or file-like object
        Either the file name (string) or an open file (file-like object)
        where the data will be loaded.

    Returns
    -------
    """
    result : csc_array, csr_array, bsr_array, dia_array or coo_array
        加载后包含数据的稀疏数组/矩阵。

    Raises
    ------
    OSError
        如果输入文件不存在或无法读取。

    See Also
    --------
    scipy.sparse.save_npz: 使用 `.npz` 格式将稀疏数组/矩阵保存到文件中。
    numpy.load: 从 `.npz` 存档中加载多个数组。

    Examples
    --------
    将稀疏数组/矩阵存储到磁盘，然后再次加载：

    >>> import numpy as np
    >>> import scipy as sp
    >>> sparse_array = sp.sparse.csc_array([[0, 0, 3], [4, 0, 0]])
    >>> sparse_array
    <Compressed Sparse Column sparse array of dtype 'int64'
        with 2 stored elements and shape (2, 3)>
    >>> sparse_array.toarray()
    array([[0, 0, 3],
           [4, 0, 0]], dtype=int64)

    >>> sp.sparse.save_npz('/tmp/sparse_array.npz', sparse_array)
    >>> sparse_array = sp.sparse.load_npz('/tmp/sparse_array.npz')

    >>> sparse_array
    <Compressed Sparse Column sparse array of dtype 'int64'
        with 2 stored elements and shape (2, 3)>
    >>> sparse_array.toarray()
    array([[0, 0, 3],
           [4, 0, 0]], dtype=int64)

    在此示例中，我们强制结果从 csr_matrix 转换为 csr_array
    >>> sparse_matrix = sp.sparse.csc_matrix([[0, 0, 3], [4, 0, 0]])
    >>> sp.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)
    >>> tmp = sp.sparse.load_npz('/tmp/sparse_matrix.npz')
    >>> sparse_array = sp.sparse.csr_array(tmp)
    ```
    # 使用 np.load() 函数加载给定的文件，同时传递 PICKLE_KWARGS 作为关键字参数
    with np.load(file, **PICKLE_KWARGS) as loaded:
        # 从加载的数据中获取稀疏矩阵的存储格式
        sparse_format = loaded.get('format')
        # 如果存储格式为空，则抛出值错误，指明文件不包含稀疏数组或矩阵
        if sparse_format is None:
            raise ValueError(f'The file {file} does not contain '
                             f'a sparse array or matrix.')
        # 将稀疏格式转换为字符串形式
        sparse_format = sparse_format.item()

        # 如果稀疏格式不是字符串，处理 Python 2 与 Python 3 的向后兼容性问题；
        # SciPy 版本低于 1.0.0 的文件可能包含 Unicode 或字节字符串
        if not isinstance(sparse_format, str):
            sparse_format = sparse_format.decode('ascii')

        # 根据加载数据的 '_is_array' 键确定稀疏类型
        if loaded.get('_is_array'):
            sparse_type = sparse_format + '_array'
        else:
            sparse_type = sparse_format + '_matrix'

        # 尝试通过字符串拼接得到类名，表示稀疏矩阵或数组的类
        try:
            cls = getattr(sp.sparse, f'{sparse_type}')
        except AttributeError as e:
            # 如果找不到对应的类，抛出值错误，指明未知的稀疏格式
            raise ValueError(f'Unknown format "{sparse_type}"') from e

        # 根据稀疏格式选择不同的加载方式，并返回相应的稀疏矩阵对象
        if sparse_format in ('csc', 'csr', 'bsr'):
            return cls((loaded['data'], loaded['indices'], loaded['indptr']),
                       shape=loaded['shape'])
        elif sparse_format == 'dia':
            return cls((loaded['data'], loaded['offsets']),
                       shape=loaded['shape'])
        elif sparse_format == 'coo':
            return cls((loaded['data'], (loaded['row'], loaded['col'])),
                       shape=loaded['shape'])
        else:
            # 对于不支持的稀疏格式，抛出未实现错误
            raise NotImplementedError(f'Load is not implemented for '
                                      f'sparse matrix of format {sparse_format}.')
```
# `D:\src\scipysrc\scipy\scipy\sparse\_extract.py`

```
"""Functions to extract parts of sparse matrices
"""

__docformat__ = "restructuredtext en"

__all__ = ['find', 'tril', 'triu']


from ._coo import coo_matrix, coo_array
from ._base import sparray


def find(A):
    """Return the indices and values of the nonzero elements of a matrix

    Parameters
    ----------
    A : dense or sparse array or matrix
        Matrix whose nonzero elements are desired.

    Returns
    -------
    (I,J,V) : tuple of arrays
        I,J, and V contain the row indices, column indices, and values
        of the nonzero entries.


    Examples
    --------
    >>> from scipy.sparse import csr_array, find
    >>> A = csr_array([[7.0, 8.0, 0],[0, 0, 9.0]])
    >>> find(A)
    (array([0, 0, 1], dtype=int32),
     array([0, 1, 2], dtype=int32),
     array([ 7.,  8.,  9.]))

    """

    # Convert the input matrix A to COO format
    A = coo_array(A, copy=True)
    # Sum duplicate entries in the COO matrix representation
    A.sum_duplicates()
    # Mask to remove entries with explicit zeros
    nz_mask = A.data != 0
    # Return row indices, column indices, and data of nonzero entries
    return A.row[nz_mask], A.col[nz_mask], A.data[nz_mask]


def tril(A, k=0, format=None):
    """Return the lower triangular portion of a sparse array or matrix

    Returns the elements on or below the k-th diagonal of A.
        - k = 0 corresponds to the main diagonal
        - k > 0 is above the main diagonal
        - k < 0 is below the main diagonal

    Parameters
    ----------
    A : dense or sparse array or matrix
        Matrix whose lower triangular portion is desired.
    k : integer : optional
        The top-most diagonal of the lower triangle.
    format : string
        Sparse format of the result, e.g. format="csr", etc.

    Returns
    -------
    L : sparse matrix
        Lower triangular portion of A in sparse format.

    See Also
    --------
    triu : upper triangle in sparse format

    Examples
    --------
    >>> from scipy.sparse import csr_array, tril
    >>> A = csr_array([[1, 2, 0, 0, 3], [4, 5, 0, 6, 7], [0, 0, 8, 9, 0]],
    ...               dtype='int32')
    >>> A.toarray()
    array([[1, 2, 0, 0, 3],
           [4, 5, 0, 6, 7],
           [0, 0, 8, 9, 0]])
    >>> tril(A).toarray()
    array([[1, 0, 0, 0, 0],
           [4, 5, 0, 0, 0],
           [0, 0, 8, 0, 0]])
    >>> tril(A).nnz
    4
    >>> tril(A, k=1).toarray()
    array([[1, 2, 0, 0, 0],
           [4, 5, 0, 0, 0],
           [0, 0, 8, 9, 0]])
    >>> tril(A, k=-1).toarray()
    array([[0, 0, 0, 0, 0],
           [4, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> tril(A, format='csc')
    <Compressed Sparse Column sparse array of dtype 'int32'
        with 4 stored elements and shape (3, 5)>

    """
    # Determine the correct COO format based on input type
    coo_sparse = coo_array if isinstance(A, sparray) else coo_matrix
    # Convert input matrix A to COO format if it is not already in that format
    A = coo_sparse(A, copy=False)
    # Mask to select elements on or below the k-th diagonal
    mask = A.row + k >= A.col
    # Extract row indices, column indices, and data based on the mask
    row = A.row[mask]
    col = A.col[mask]
    data = A.data[mask]
    # Create a new COO matrix with the selected elements
    new_coo = coo_sparse((data, (row, col)), shape=A.shape, dtype=A.dtype)
    # Return the new COO matrix formatted as specified
    return new_coo.asformat(format)


def triu(A, k=0, format=None):
    """Return the upper triangular portion of a sparse array or matrix

    Returns the elements on or above the k-th diagonal of A.
        - k = 0 corresponds to the main diagonal
        - k > 0 is above the main diagonal
        - k < 0 is below the main diagonal

    Parameters
    ----------
    A : dense or sparse array or matrix
        Matrix whose upper triangular portion is desired.
    k : integer : optional
        The bottom-most diagonal of the upper triangle.
    format : string
        Sparse format of the result, e.g. format="csr", etc.

    Returns
    -------
    U : sparse matrix
        Upper triangular portion of A in sparse format.

    See Also
    --------
    tril : lower triangle in sparse format

    Examples
    --------
    >>> from scipy.sparse import csr_array, triu
    >>> A = csr_array([[1, 2, 0], [3, 4, 5], [0, 6, 7]], dtype='int32')
    >>> A.toarray()
    array([[1, 2, 0],
           [3, 4, 5],
           [0, 6, 7]])
    >>> triu(A).toarray()
    array([[1, 2, 0],
           [0, 4, 5],
           [0, 0, 7]])
    >>> triu(A, k=1).toarray()
    array([[0, 2, 0],
           [0, 0, 5],
           [0, 0, 0]])
    >>> triu(A, format='csc')
    <Compressed Sparse Column sparse array of dtype 'int32'
        with 5 stored elements in Compressed Sparse Column format>

    """
    """Return the upper triangular portion of a sparse array or matrix
    
    Returns the elements on or above the k-th diagonal of A.
        - k = 0 corresponds to the main diagonal
        - k > 0 is above the main diagonal
        - k < 0 is below the main diagonal
    
    Parameters
    ----------
    A : dense or sparse array or matrix
        Matrix whose upper triangular portion is desired.
    k : integer : optional
        The bottom-most diagonal of the upper triangle.
    format : string
        Sparse format of the result, e.g. format="csr", etc.
    
    Returns
    -------
    L : sparse array or matrix 
        Upper triangular portion of A in sparse format.
        Sparse array if A is a sparse array, otherwise matrix.
    
    See Also
    --------
    tril : lower triangle in sparse format
    
    Examples
    --------
    >>> from scipy.sparse import csr_array, triu
    >>> A = csr_array([[1, 2, 0, 0, 3], [4, 5, 0, 6, 7], [0, 0, 8, 9, 0]],
    ...                dtype='int32')
    >>> A.toarray()
    array([[1, 2, 0, 0, 3],
           [4, 5, 0, 6, 7],
           [0, 0, 8, 9, 0]])
    >>> triu(A).toarray()
    array([[1, 2, 0, 0, 3],
           [0, 5, 0, 6, 7],
           [0, 0, 8, 9, 0]])
    >>> triu(A).nnz
    8
    >>> triu(A, k=1).toarray()
    array([[0, 2, 0, 0, 3],
           [0, 0, 0, 6, 7],
           [0, 0, 0, 9, 0]])
    >>> triu(A, k=-1).toarray()
    array([[1, 2, 0, 0, 3],
           [4, 5, 0, 6, 7],
           [0, 0, 8, 9, 0]])
    >>> triu(A, format='csc')
    <Compressed Sparse Column sparse array of dtype 'int32'
        with 8 stored elements and shape (3, 5)>
    
    """
    coo_sparse = coo_array if isinstance(A, sparray) else coo_matrix
    
    # 将输入矩阵 A 转换为 COO 格式，以便处理
    A = coo_sparse(A, copy=False)
    
    # 创建一个布尔掩码，用于选择 A 中符合要求的元素位置
    mask = A.row + k <= A.col
    
    # 根据掩码选择符合要求的行、列和数据
    row = A.row[mask]
    col = A.col[mask]
    data = A.data[mask]
    
    # 使用选定的行、列和数据重新构造一个 COO 格式的稀疏矩阵
    new_coo = coo_sparse((data, (row, col)), shape=A.shape, dtype=A.dtype)
    
    # 将新创建的 COO 格式稀疏矩阵按指定格式转换为所需的稀疏格式并返回
    return new_coo.asformat(format)
```
# `D:\src\scipysrc\scipy\scipy\sparse\_matrix.py`

```
    """This class provides a base class for all sparse matrix classes.

    It cannot be instantiated.  Most of the work is provided by subclasses.
    """

    # Property returning the container class for Block Sparse Row (BSR) matrices
    @property
    def _bsr_container(self):
        from ._bsr import bsr_matrix
        return bsr_matrix

    # Property returning the container class for Coordinate (COO) format matrices
    @property
    def _coo_container(self):
        from ._coo import coo_matrix
        return coo_matrix

    # Property returning the container class for Compressed Sparse Column (CSC) matrices
    @property
    def _csc_container(self):
        from ._csc import csc_matrix
        return csc_matrix

    # Property returning the container class for Compressed Sparse Row (CSR) matrices
    @property
    def _csr_container(self):
        from ._csr import csr_matrix
        return csr_matrix

    # Property returning the container class for Diagonal (DIA) matrices
    @property
    def _dia_container(self):
        from ._dia import dia_matrix
        return dia_matrix

    # Property returning the container class for Dictionary of Keys (DOK) matrices
    @property
    def _dok_container(self):
        from ._dok import dok_matrix
        return dok_matrix

    # Property returning the container class for List of Lists (LIL) matrices
    @property
    def _lil_container(self):
        from ._lil import lil_matrix
        return lil_matrix

    # Restore matrix multiplication
    def __mul__(self, other):
        """Overrides the multiplication operator '*' for matrices."""
        return self._matmul_dispatch(other)

    # Restore right-side matrix multiplication
    def __rmul__(self, other):
        """Overrides the right-side multiplication operator '*' for matrices."""
        return self._rmatmul_dispatch(other)

    # Restore matrix power operation
    def __pow__(self, power):
        """Overrides the power operator '**' for matrices."""
        from .linalg import matrix_power
        return matrix_power(self, power)

    ## Backward compatibility

    def set_shape(self, shape):
        """Set the shape of the matrix in-place.

        Parameters
        ----------
        shape : tuple
            The new shape (dimensions) of the matrix.
        """
        # Make sure copy is False since this is in place
        # Make sure format is unchanged because we are doing a __dict__ swap
        new_self = self.reshape(shape, copy=False).asformat(self.format)
        self.__dict__ = new_self.__dict__

    def get_shape(self):
        """Get the current shape of the matrix.

        Returns
        -------
        tuple
            Shape (dimensions) of the matrix.
        """
        return self._shape

    shape = property(fget=get_shape, fset=set_shape,
                     doc="Shape of the matrix")

    def asfptype(self):
        """Upcast matrix to a floating point format if necessary.

        Returns
        -------
        self
            Matrix in a floating point format.
        """
        return self._asfptype()

    def getmaxprint(self):
        """Get the maximum number of elements to display when printed.

        Returns
        -------
        int
            Maximum number of elements to display.
        """
        return self._getmaxprint()

    def getformat(self):
        """Get the storage format of the matrix.

        Returns
        -------
        str
            Matrix storage format.
        """
        return self.format

    def getnnz(self, axis=None):
        """Get the number of stored values, including explicit zeros.

        Parameters
        ----------
        axis : {None, 0, 1}, optional
            Axis along which the number of stored values is counted.
            None counts values across the whole array, 0 counts along columns,
            and 1 counts along rows.

        Returns
        -------
        int
            Number of stored values.
        """
        return self._getnnz(axis=axis)

    def getH(self):
        """Return the Hermitian transpose (conjugate transpose) of this matrix.

        Returns
        -------
        self
            Hermitian transpose of the matrix.
        """
        return self.conjugate().transpose()

    def getcol(self, j):
        """Return a copy of column j of the matrix.

        Parameters
        ----------
        j : int
            Column index.

        Returns
        -------
        self
            Copy of column j as a sparse matrix (column vector).
        """
        return self._getcol(j)
    # 返回稀疏矩阵中第 i 行的副本，作为 (1 x n) 稀疏矩阵（行向量）。
    def getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n) sparse
        matrix (row vector).
        """
        # 调用私有方法 _getrow 获取稀疏矩阵中第 i 行的副本
        return self._getrow(i)

    # 返回此稀疏矩阵的密集表示。
    def todense(self, order=None, out=None):
        """
        Return a dense representation of this sparse matrix.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Whether to store multi-dimensional data in C (row-major)
            or Fortran (column-major) order in memory. The default
            is 'None', which provides no ordering guarantees.
            Cannot be specified in conjunction with the `out`
            argument.

        out : ndarray, 2-D, optional
            If specified, uses this array (or `numpy.matrix`) as the
            output buffer instead of allocating a new array to
            return. The provided array must have the same shape and
            dtype as the sparse matrix on which you are calling the
            method.

        Returns
        -------
        arr : numpy.matrix, 2-D
            A NumPy matrix object with the same shape and containing
            the same data represented by the sparse matrix, with the
            requested memory order. If `out` was passed and was an
            array (rather than a `numpy.matrix`), it will be filled
            with the appropriate values and returned wrapped in a
            `numpy.matrix` object that shares the same memory.
        """
        # 调用父类的 todense 方法将稀疏矩阵转换为密集矩阵，并返回结果
        return super().todense(order, out)
```
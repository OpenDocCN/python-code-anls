# `D:\src\scipysrc\sympy\sympy\polys\matrices\_dfm.py`

```
#
# sympy.polys.matrices.dfm
#
# This modules defines the DFM class which is a wrapper for dense flint
# matrices as found in python-flint.
#
# As of python-flint 0.4.1 matrices over the following domains can be supported
# by python-flint:
#
#   ZZ: flint.fmpz_mat
#   QQ: flint.fmpq_mat
#   GF(p): flint.nmod_mat (p prime and p < ~2**62)
#
# The underlying flint library has many more domains, but these are not yet
# supported by python-flint.
#
# The DFM class is a wrapper for the flint matrices and provides a common
# interface for all supported domains that is interchangeable with the DDM
# and SDM classes so that DomainMatrix can be used with any as its internal
# matrix representation.
#

# TODO:
#
# Implement the following methods that are provided by python-flint:
#
# - hnf (Hermite normal form)
# - snf (Smith normal form)
# - minpoly
# - is_hnf
# - is_snf
# - rank
#
# The other types DDM and SDM do not have these methods and the algorithms
# for hnf, snf and rank are already implemented. Algorithms for minpoly,
# is_hnf and is_snf would need to be added.
#
# Add more methods to python-flint to expose more of Flint's functionality
# and also to make some of the above methods simpler or more efficient e.g.
# slicing, fancy indexing etc.

from sympy.external.gmpy import GROUND_TYPES  # Importing GROUND_TYPES from gmpy module
from sympy.external.importtools import import_module  # Importing import_module function from importtools module
from sympy.utilities.decorator import doctest_depends_on  # Importing doctest_depends_on decorator

from sympy.polys.domains import ZZ, QQ  # Importing ZZ and QQ from polys.domains module

from .exceptions import (  # Importing specific exceptions from local exceptions module
    DMBadInputError,
    DMDomainError,
    DMNonSquareMatrixError,
    DMNonInvertibleMatrixError,
    DMRankError,
    DMShapeError,
    DMValueError,
)

if GROUND_TYPES != 'flint':  # Conditionally skipping doctests if GROUND_TYPES is not 'flint'
    __doctest_skip__ = ['*']

flint = import_module('flint')  # Importing flint module using import_module function

__all__ = ['DFM']  # Exposing only DFM class from this module

@doctest_depends_on(ground_types=['flint'])  # Decorator indicating doctest dependency on 'flint' ground type
class DFM:
    """
    Dense FLINT matrix. This class is a wrapper for matrices from python-flint.

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.matrices.dfm import DFM
    >>> dfm = DFM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    >>> dfm
    [[1, 2], [3, 4]]
    >>> dfm.rep
    [1, 2]
    [3, 4]
    >>> type(dfm.rep)  # doctest: +SKIP
    <class 'flint._flint.fmpz_mat'>

    Usually, the DFM class is not instantiated directly, but is created as the
    internal representation of :class:`~.DomainMatrix`. When
    `SYMPY_GROUND_TYPES` is set to `flint` and `python-flint` is installed, the
    :class:`DFM` class is used automatically as the internal representation of
    :class:`~.DomainMatrix` in dense format if the domain is supported by
    python-flint.

    >>> from sympy.polys.matrices.domainmatrix import DM
    >>> dM = DM([[1, 2], [3, 4]], ZZ)
    >>> dM.rep
    [[1, 2], [3, 4]]

    A :class:`~.DomainMatrix` can be converted to :class:`DFM` by calling the
    :meth:`to_dfm` method:

    >>> dM.to_dfm()
    [[1, 2], [3, 4]]

    """

    fmt = 'dense'  # Specifies the format of the matrix as dense
    is_DFM = True   # Indicates that this is a DFM matrix
    is_DDM = False  # Indicates that this is not a DDM matrix
    def __new__(cls, rowslist, shape, domain):
        """创建一个新的实例对象，接受行列表、形状和域作为参数。"""
        # 获取适合指定域的 flint 矩阵函数
        flint_mat = cls._get_flint_func(domain)

        # 如果形状中不包含 0
        if 0 not in shape:
            try:
                # 使用 flint 矩阵函数创建表示矩阵的对象 rep
                rep = flint_mat(rowslist)
            except (ValueError, TypeError):
                # 如果输入不符合预期，抛出异常
                raise DMBadInputError(f"Input should be a list of list of {domain}")
        else:
            # 否则，使用 flint 矩阵函数创建具有指定形状的对象 rep
            rep = flint_mat(*shape)

        # 调用类方法 _new 创建新的实例对象并返回
        return cls._new(rep, shape, domain)

    @classmethod
    def _new(cls, rep, shape, domain):
        """从 flint 矩阵创建内部实例。"""
        # 检查表示矩阵 rep 的形状和域是否符合预期
        cls._check(rep, shape, domain)
        # 使用 object.__new__ 方法创建新的实例对象 obj
        obj = object.__new__(cls)
        # 将表示矩阵 rep、形状 shape 和域 domain 设置为实例对象的属性
        obj.rep = rep
        obj.shape = obj.rows, obj.cols = shape
        obj.domain = domain
        # 返回新创建的实例对象
        return obj

    def _new_rep(self, rep):
        """创建一个具有相同形状和域但具有新表示矩阵 rep 的新 DFM 对象。"""
        # 调用 _new 方法创建一个新的 DFM 对象，并返回
        return self._new(rep, self.shape, self.domain)

    @classmethod
    def _check(cls, rep, shape, domain):
        """检查表示矩阵 rep 的形状和域是否与 DFM 的期望匹配。"""
        # 获取表示矩阵 rep 的形状
        repshape = (rep.nrows(), rep.ncols())
        # 如果表示矩阵形状与 DFM 的形状不匹配，抛出异常
        if repshape != shape:
            raise DMBadInputError("Shape of rep does not match shape of DFM")
        # 如果域是 ZZ，但表示矩阵 rep 不是 flint.fmpz_mat 类型，抛出异常
        if domain == ZZ and not isinstance(rep, flint.fmpz_mat):
            raise RuntimeError("Rep is not a flint.fmpz_mat")
        # 如果域是 QQ，但表示矩阵 rep 不是 flint.fmpq_mat 类型，抛出异常
        elif domain == QQ and not isinstance(rep, flint.fmpq_mat):
            raise RuntimeError("Rep is not a flint.fmpq_mat")
        # 如果域既不是 ZZ 也不是 QQ，抛出异常，因为 DFM 仅支持这两种域
        elif domain not in (ZZ, QQ):
            raise NotImplementedError("Only ZZ and QQ are supported by DFM")

    @classmethod
    def _supports_domain(cls, domain):
        """如果给定的域被 DFM 支持，则返回 True。"""
        return domain in (ZZ, QQ)

    @classmethod
    def _get_flint_func(cls, domain):
        """返回给定域的 flint 矩阵类。"""
        # 根据域返回相应的 flint 矩阵类
        if domain == ZZ:
            return flint.fmpz_mat
        elif domain == QQ:
            return flint.fmpq_mat
        else:
            raise NotImplementedError("Only ZZ and QQ are supported by DFM")

    @property
    def _func(self):
        """返回一个可调用对象，用于创建与当前域相同的 flint 矩阵。"""
        # 返回与当前域相同的 flint 矩阵类的函数
        return self._get_flint_func(self.domain)

    def __str__(self):
        """返回对象的字符串表示形式。"""
        # 返回对象转换为 DDM 后的字符串表示形式
        return str(self.to_ddm())

    def __repr__(self):
        """返回对象的表达式表示形式。"""
        # 返回对象的表达式表示形式，以 'DFM' 开头
        return f'DFM{repr(self.to_ddm())[3:]}'

    def __eq__(self, other):
        """比较对象是否相等。"""
        # 如果 other 不是 DFM 类型，则返回 NotImplemented
        if not isinstance(other, DFM):
            return NotImplemented
        # 首先比较域，因为我们不希望具有不同域的矩阵被认为是相等的，
        # 例如，具有相同条目的 flint fmpz_mat 和 fmpq_mat 将被认为是相等的。
        return self.domain == other.domain and self.rep == other.rep

    @classmethod
    def from_list(cls, rowslist, shape, domain):
        """从一个嵌套列表构建对象。"""
        # 使用给定的行列表、形状和域构建对象并返回
        return cls(rowslist, shape, domain)
    def to_list(self):
        """
        Convert to a nested list.
        """
        return self.rep.tolist()

    def copy(self):
        """
        Return a copy of self.
        """
        return self._new_rep(self._func(self.rep))

    def to_ddm(self):
        """
        Convert to a DDM (Domain Matrix).
        """
        return DDM.from_list(self.to_list(), self.shape, self.domain)

    def to_sdm(self):
        """
        Convert to a SDM (Sparse Domain Matrix).
        """
        return SDM.from_list(self.to_list(), self.shape, self.domain)

    def to_dfm(self):
        """
        Return self (Domain Matrix).
        """
        return self

    def to_dfm_or_ddm(self):
        """
        Convert to a :class:`DFM`.

        This method returns self since DFM is already represented by self.

        See Also
        ========

        to_ddm
        to_sdm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm_or_ddm
        """
        return self

    @classmethod
    def from_ddm(cls, ddm):
        """
        Convert from a DDM (Domain Matrix).

        Uses the class method from_list to create a new instance of DFM.
        """
        return cls.from_list(ddm.to_list(), ddm.shape, ddm.domain)

    @classmethod
    def from_list_flat(cls, elements, shape, domain):
        """
        Inverse of :meth:`to_list_flat`.

        Constructs a new instance of DFM from a flat list of elements.
        """
        func = cls._get_flint_func(domain)
        try:
            rep = func(*shape, elements)
        except ValueError:
            raise DMBadInputError(f"Incorrect number of elements for shape {shape}")
        except TypeError:
            raise DMBadInputError(f"Input should be a list of {domain}")
        return cls(rep, shape, domain)

    def to_list_flat(self):
        """
        Convert to a flat list.

        Returns the flat list representation of the matrix.
        """
        return self.rep.entries()

    def to_flat_nz(self):
        """
        Convert to a flat list of non-zeros.

        Converts the matrix to a flat list representation containing only non-zero elements.
        """
        return self.to_ddm().to_flat_nz()

    @classmethod
    def from_flat_nz(cls, elements, data, domain):
        """
        Inverse of :meth:`to_flat_nz`.

        Constructs a new instance of DFM from a flat list of non-zero elements and associated data.
        """
        return DDM.from_flat_nz(elements, data, domain).to_dfm()

    def to_dod(self):
        """
        Convert to a DOD (Dictionary of Dictionaries).

        Converts the matrix to a DOD representation.
        """
        return self.to_ddm().to_dod()

    @classmethod
    def from_dod(cls, dod, shape, domain):
        """
        Inverse of :meth:`to_dod`.

        Constructs a new instance of DFM from a DOD representation.
        """
        return DDM.from_dod(dod, shape, domain).to_dfm()

    def to_dok(self):
        """
        Convert to a DOK (Dictionary of Keys).

        Converts the matrix to a DOK representation.
        """
        return self.to_ddm().to_dok()

    @classmethod
    def from_dok(cls, dok, shape, domain):
        """
        Inverse of :meth:`to_dod`.

        Constructs a new instance of DFM from a DOK representation.
        """
        return DDM.from_dok(dok, shape, domain).to_dfm()

    def iter_values(self):
        """
        Iterate over the non-zero values of the matrix.

        Generates non-zero values from the matrix in row-major order.
        """
        m, n = self.shape
        rep = self.rep
        for i in range(m):
            for j in range(n):
                repij = rep[i, j]
                if repij:
                    yield rep[i, j]
    # 迭代器方法，遍历稀疏矩阵中非零元素的索引和值
    def iter_items(self):
        """Iterate over indices and values of nonzero elements of the matrix."""
        m, n = self.shape
        rep = self.rep
        for i in range(m):
            for j in range(n):
                repij = rep[i, j]
                # 如果 repij 不为零，则生成器返回索引 (i, j) 和对应的值 repij
                if repij:
                    yield ((i, j), repij)

    # 转换矩阵到指定的新域
    def convert_to(self, domain):
        """Convert to a new domain."""
        if domain == self.domain:
            return self.copy()  # 如果目标域与当前域相同，则返回当前对象的副本
        elif domain == QQ and self.domain == ZZ:
            # 如果当前域是整数域 ZZ，目标域是有理数域 QQ，则使用 flint 库转换表示
            return self._new(flint.fmpq_mat(self.rep), self.shape, domain)
        elif domain == ZZ and self.domain == QQ:
            # 如果当前域是有理数域 QQ，目标域是整数域 ZZ，则进行特定转换操作
            # XXX: python-flint 没有实现 fmpz_mat.from_fmpq_mat 方法
            return self.to_ddm().convert_to(domain).to_dfm()
        else:
            # 如果不支持当前域到目标域的转换，则抛出 NotImplementedError 异常
            # 调用者在调用此方法时需要确保先将矩阵转换为 DDM
            raise NotImplementedError("Only ZZ and QQ are supported by DFM")

    # 获取矩阵的指定位置 (i, j) 处的元素值
    def getitem(self, i, j):
        """Get the ``(i, j)``-th entry."""
        # XXX: flint matrices do not support negative indices
        # XXX: They also raise ValueError instead of IndexError
        m, n = self.shape
        if i < 0:
            i += m  # 处理负索引，将其转换为对应的正索引
        if j < 0:
            j += n  # 处理负索引，将其转换为对应的正索引
        try:
            return self.rep[i, j]  # 返回指定位置的元素值
        except ValueError:
            # 如果索引超出矩阵范围，则抛出 IndexError 异常
            raise IndexError(f"Invalid indices ({i}, {j}) for Matrix of shape {self.shape}")

    # 设置矩阵的指定位置 (i, j) 处的元素值
    def setitem(self, i, j, value):
        """Set the ``(i, j)``-th entry."""
        # XXX: flint matrices do not support negative indices
        # XXX: They also raise ValueError instead of IndexError
        m, n = self.shape
        if i < 0:
            i += m  # 处理负索引，将其转换为对应的正索引
        if j < 0:
            j += n  # 处理负索引，将其转换为对应的正索引
        try:
            self.rep[i, j] = value  # 设置指定位置的元素值为给定的 value
        except ValueError:
            # 如果索引超出矩阵范围，则抛出 IndexError 异常
            raise IndexError(f"Invalid indices ({i}, {j}) for Matrix of shape {self.shape}")

    # 提取子矩阵，不进行任何检查
    def _extract(self, i_indices, j_indices):
        """Extract a submatrix with no checking."""
        # 从矩阵中提取子矩阵，i_indices 和 j_indices 是行和列的索引列表
        M = self.rep
        lol = [[M[i, j] for j in j_indices] for i in i_indices]
        shape = (len(i_indices), len(j_indices))  # 计算提取的子矩阵的形状
        return self.from_list(lol, shape, self.domain)  # 使用 lol 和 domain 创建新的子矩阵对象
    def extract(self, rowslist, colslist):
        """Extract a submatrix."""
        # XXX: flint matrices do not support fancy indexing or negative indices
        #
        # Check and convert negative indices before calling _extract.
        
        # 获取矩阵的行数 m 和列数 n
        m, n = self.shape

        new_rows = []
        new_cols = []

        # 处理行索引列表 rowslist
        for i in rowslist:
            if i < 0:
                # 处理负索引，转换为正索引
                i_pos = i + m
            else:
                i_pos = i
            # 检查索引是否有效
            if not 0 <= i_pos < m:
                raise IndexError(f"Invalid row index {i} for Matrix of shape {self.shape}")
            new_rows.append(i_pos)

        # 处理列索引列表 colslist
        for j in colslist:
            if j < 0:
                # 处理负索引，转换为正索引
                j_pos = j + n
            else:
                j_pos = j
            # 检查索引是否有效
            if not 0 <= j_pos < n:
                raise IndexError(f"Invalid column index {j} for Matrix of shape {self.shape}")
            new_cols.append(j_pos)

        # 调用 _extract 方法提取子矩阵
        return self._extract(new_rows, new_cols)

    def extract_slice(self, rowslice, colslice):
        """Slice a DFM."""
        # XXX: flint matrices do not support slicing
        
        # 获取矩阵的行数 m 和列数 n
        m, n = self.shape
        
        # 根据行切片和列切片生成索引列表
        i_indices = range(m)[rowslice]
        j_indices = range(n)[colslice]
        
        # 调用 _extract 方法对切片后的矩阵进行提取
        return self._extract(i_indices, j_indices)

    def neg(self):
        """Negate a DFM matrix."""
        # 调用 _new_rep 方法对矩阵进行取反操作
        return self._new_rep(-self.rep)

    def add(self, other):
        """Add two DFM matrices."""
        # 调用 _new_rep 方法对两个矩阵进行加法操作
        return self._new_rep(self.rep + other.rep)

    def sub(self, other):
        """Subtract two DFM matrices."""
        # 调用 _new_rep 方法对两个矩阵进行减法操作
        return self._new_rep(self.rep - other.rep)

    def mul(self, other):
        """Multiply a DFM matrix from the right by a scalar."""
        # 调用 _new_rep 方法将矩阵右乘以标量 other
        return self._new_rep(self.rep * other)

    def rmul(self, other):
        """Multiply a DFM matrix from the left by a scalar."""
        # 调用 _new_rep 方法将矩阵左乘以标量 other
        return self._new_rep(other * self.rep)

    def mul_elementwise(self, other):
        """Elementwise multiplication of two DFM matrices."""
        # XXX: flint matrices do not support elementwise multiplication
        # 转换为 DDM 进行元素级乘法操作，然后转回 DFM
        return self.to_ddm().mul_elementwise(other.to_ddm()).to_dfm()

    def matmul(self, other):
        """Multiply two DFM matrices."""
        # 根据矩阵乘法规则计算两个矩阵的乘积
        shape = (self.rows, other.cols)
        return self._new(self.rep * other.rep, shape, self.domain)

    # XXX: For the most part DomainMatrix does not expect DDM, SDM, or DFM to
    # have arithmetic operators defined. The only exception is negation.
    # Perhaps that should be removed.

    def __neg__(self):
        """Negate a DFM matrix."""
        # 调用 neg 方法对矩阵进行取反操作
        return self.neg()

    @classmethod
    def zeros(cls, shape, domain):
        """Return a zero DFM matrix."""
        # 获取与指定域相关的 flint 函数
        func = cls._get_flint_func(domain)
        # 使用 flint 函数创建一个全零矩阵，并返回其实例
        return cls._new(func(*shape), shape, domain)

    # XXX: flint matrices do not have anything like ones or eye
    # In the methods below we convert to DDM and then back to DFM which is
    # probably about as efficient as implementing these methods directly.
    @classmethod
    def ones(cls, shape, domain):
        """Return a one DFM matrix."""
        # XXX: flint matrices do not have anything like ones
        # 使用 DDM 类的 ones 方法创建一个指定形状和域的矩阵，然后转换为 DFM 类型返回
        return DDM.ones(shape, domain).to_dfm()

    @classmethod
    def eye(cls, n, domain):
        """Return the identity matrix of size n."""
        # XXX: flint matrices do not have anything like eye
        # 使用 DDM 类的 eye 方法创建一个大小为 n 的单位矩阵，然后转换为 DFM 类型返回
        return DDM.eye(n, domain).to_dfm()

    @classmethod
    def diag(cls, elements, domain):
        """Return a diagonal matrix."""
        # 使用 DDM 类的 diag 方法创建一个由给定元素组成的对角矩阵，然后转换为 DFM 类型返回
        return DDM.diag(elements, domain).to_dfm()

    def applyfunc(self, func, domain):
        """Apply a function to each entry of a DFM matrix."""
        # 将 DFM 矩阵转换为 DDM 类型，然后对每个条目应用给定函数，再转换回 DFM 类型返回结果
        return self.to_ddm().applyfunc(func, domain).to_dfm()

    def transpose(self):
        """Transpose a DFM matrix."""
        # 返回一个新的 DFM 矩阵，其表示是当前矩阵表示的转置，形状是原矩阵列数和行数的交换，域不变
        return self._new(self.rep.transpose(), (self.cols, self.rows), self.domain)

    def hstack(self, *others):
        """Horizontally stack matrices."""
        # 将当前 DFM 矩阵及其他输入的 DFM 矩阵按水平方向堆叠，返回新的 DFM 矩阵
        return self.to_ddm().hstack(*[o.to_ddm() for o in others]).to_dfm()

    def vstack(self, *others):
        """Vertically stack matrices."""
        # 将当前 DFM 矩阵及其他输入的 DFM 矩阵按垂直方向堆叠，返回新的 DFM 矩阵
        return self.to_ddm().vstack(*[o.to_ddm() for o in others]).to_dfm()

    def diagonal(self):
        """Return the diagonal of a DFM matrix."""
        # 获取当前 DFM 矩阵的表示，然后返回其对角线上的元素构成的列表
        M = self.rep
        m, n = self.shape
        return [M[i, i] for i in range(min(m, n))]

    def is_upper(self):
        """Return ``True`` if the matrix is upper triangular."""
        # 检查当前 DFM 矩阵的表示是否为上三角矩阵，如果是则返回 True，否则返回 False
        M = self.rep
        for i in range(self.rows):
            for j in range(i):
                if M[i, j]:
                    return False
        return True

    def is_lower(self):
        """Return ``True`` if the matrix is lower triangular."""
        # 检查当前 DFM 矩阵的表示是否为下三角矩阵，如果是则返回 True，否则返回 False
        M = self.rep
        for i in range(self.rows):
            for j in range(i + 1, self.cols):
                if M[i, j]:
                    return False
        return True

    def is_diagonal(self):
        """Return ``True`` if the matrix is diagonal."""
        # 检查当前 DFM 矩阵是否为对角矩阵，如果是则返回 True，否则返回 False
        return self.is_upper() and self.is_lower()

    def is_zero_matrix(self):
        """Return ``True`` if the matrix is the zero matrix."""
        # 检查当前 DFM 矩阵是否为零矩阵，如果是则返回 True，否则返回 False
        M = self.rep
        for i in range(self.rows):
            for j in range(self.cols):
                if M[i, j]:
                    return False
        return True

    def nnz(self):
        """Return the number of non-zero elements in the matrix."""
        # 返回当前 DFM 矩阵中非零元素的数量，通过转换为 DDM 类型计算得到
        return self.to_ddm().nnz()

    def scc(self):
        """Return the strongly connected components of the matrix."""
        # 返回当前 DFM 矩阵的强连通分量，通过转换为 DDM 类型计算得到
        return self.to_ddm().scc()

    @doctest_depends_on(ground_types='flint')
    # 定义一个方法 `det`，用于计算矩阵的行列式，使用 FLINT 库进行计算

    """
    Compute the determinant of the matrix using FLINT.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> dfm = M.to_DM().to_dfm()
    >>> dfm
    [[1, 2], [3, 4]]
    >>> dfm.det()
    -2

    Notes
    =====

    Calls the ``.det()`` method of the underlying FLINT matrix.

    For :ref:`ZZ` or :ref:`QQ` this calls ``fmpz_mat_det`` or
    ``fmpq_mat_det`` respectively.

    At the time of writing the implementation of ``fmpz_mat_det`` uses one
    of several algorithms depending on the size of the matrix and bit size
    of the entries. The algorithms used are:

    - Cofactor for very small (up to 4x4) matrices.
    - Bareiss for small (up to 25x25) matrices.
    - Modular algorithms for larger matrices (up to 60x60) or for larger
      matrices with large bit sizes.
    - Modular "accelerated" for larger matrices (60x60 upwards) if the bit
      size is smaller than the dimensions of the matrix.

    The implementation of ``fmpq_mat_det`` clears denominators from each
    row (not the whole matrix) and then calls ``fmpz_mat_det`` and divides
    by the product of the denominators.

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.det
        Higher level interface to compute the determinant of a matrix.
    """

    # XXX: At least the first three algorithms described above should also
    # be implemented in the pure Python DDM and SDM classes which at the
    # time of writng just use Bareiss for all matrices and domains.
    # Probably in Python the thresholds would be different though.
    # 警告：至少应在纯 Python 的 DDM 和 SDM 类中实现上述描述的前三种算法，
    # 尽管在编写时它们只对所有矩阵和域使用 Bareiss 算法。
    # 但在 Python 中，这些阈值可能会有所不同。

    return self.rep.det()
    def charpoly(self):
        """
        使用 FLINT 计算矩阵的特征多项式。

        示例
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2], [3, 4]])
        >>> dfm = M.to_DM().to_dfm()  # 需要 ground types = 'flint'
        >>> dfm
        [[1, 2], [3, 4]]
        >>> dfm.charpoly()
        [1, -5, -2]

        注意
        =====

        调用底层 FLINT 矩阵的 ``.charpoly()`` 方法。

        对于 :ref:`ZZ` 或 :ref:`QQ`，这将分别调用 ``fmpz_mat_charpoly`` 或
        ``fmpq_mat_charpoly``。

        在编写时，``fmpq_mat_charpoly`` 方法会清除整个矩阵的分母，然后调用
        ``fmpz_mat_charpoly``。特征多项式的系数然后乘以分母的幂次。

        ``fmpz_mat_charpoly`` 方法使用模算法和 CRT 重构。模算法使用
        ``nmod_mat_charpoly``，对于小矩阵和非素数模数或其他情况下使用 Danilevsky 方法。

        参见
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.charpoly
            计算矩阵的特征多项式的高级接口。
        """
        # FLINT 的多项式系数与 SymPy 相比是反向的顺序。
        return self.rep.charpoly().coeffs()[::-1]

    @doctest_depends_on(ground_types='flint')
    def inv(self):
        """
        Compute the inverse of a matrix using FLINT.

        Examples
        ========

        >>> from sympy import Matrix, QQ
        >>> M = Matrix([[1, 2], [3, 4]])
        >>> dfm = M.to_DM().to_dfm().convert_to(QQ)
        >>> dfm
        [[1, 2], [3, 4]]
        >>> dfm.inv()
        [[-2, 1], [3/2, -1/2]]
        >>> dfm.matmul(dfm.inv())
        [[1, 0], [0, 1]]

        Notes
        =====

        Calls the ``.inv()`` method of the underlying FLINT matrix.

        For now this will raise an error if the domain is :ref:`ZZ` but will
        use the FLINT method for :ref:`QQ`.

        The FLINT methods for :ref:`ZZ` and :ref:`QQ` are ``fmpz_mat_inv`` and
        ``fmpq_mat_inv`` respectively. The ``fmpz_mat_inv`` method computes an
        inverse with denominator. This is implemented by calling
        ``fmpz_mat_solve`` (see notes in :meth:`lu_solve` about the algorithm).

        The ``fmpq_mat_inv`` method clears denominators from each row and then
        multiplies those into the rhs identity matrix before calling
        ``fmpz_mat_solve``.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.inv
            Higher level method for computing the inverse of a matrix.
        """
        # TODO: Implement similar algorithms for DDM and SDM.
        #
        # XXX: The flint fmpz_mat and fmpq_mat inv methods both return fmpq_mat
        # by default. The fmpz_mat method has an optional argument to return
        # fmpz_mat instead for unimodular matrices.
        #
        # The convention in DomainMatrix is to raise an error if the matrix is
        # not over a field regardless of whether the matrix is invertible over
        # its domain or over any associated field. Maybe DomainMatrix.inv
        # should be changed to always return a matrix over an associated field
        # except with a unimodular argument for returning an inverse over a
        # ring if possible.
        #
        # For now we follow the existing DomainMatrix convention...
        K = self.domain  # 获取矩阵的定义域
        m, n = self.shape  # 获取矩阵的行数和列数

        if m != n:
            raise DMNonSquareMatrixError("cannot invert a non-square matrix")  # 若矩阵不是方阵则抛出异常

        if K == ZZ:
            raise DMDomainError("field expected, got %s" % K)  # 若定义域是整数环，则抛出异常
        elif K == QQ:
            try:
                return self._new_rep(self.rep.inv())  # 若定义域是有理数域，则计算矩阵的逆
            except ZeroDivisionError:
                raise DMNonInvertibleMatrixError("matrix is not invertible")  # 若矩阵不可逆则抛出异常
        else:
            # 若定义域是其他类型（如其他域），则抛出未实现异常
            raise NotImplementedError("DFM.inv() is not implemented for %s" % K)

    def lu(self):
        """Return the LU decomposition of the matrix."""
        L, U, swaps = self.to_ddm().lu()  # 计算矩阵的LU分解
        return L.to_dfm(), U.to_dfm(), swaps

    # XXX: The lu_solve function should be renamed to solve. Whether or not it
    # 使用LU分解是一个实现细节。一个名为lu_solve的方法在多次使用LU分解来解决不同右手边的情况下是有意义的，但这将意味着不同的调用签名。
    #
    # 底层的python-flint方法有一个algorithm=参数，所以我们可以使用它，并且可以有例如solve_lu和solve_modular，或者也许一个method=参数来在两者之间进行选择。
    # Flint本身有比python-flint暴露的更多算法可供选择。

    @doctest_depends_on(ground_types='flint')
    # 定义一个方法 lu_solve，用于使用 FLINT 解决矩阵方程
    """
    Solve a matrix equation using FLINT.
    
    Examples
    ========
    
    >>> from sympy import Matrix, QQ
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> dfm = M.to_DM().to_dfm().convert_to(QQ)
    >>> dfm
    [[1, 2], [3, 4]]
    >>> rhs = Matrix([1, 2]).to_DM().to_dfm().convert_to(QQ)
    >>> dfm.lu_solve(rhs)
    [[0], [1/2]]
    
    Notes
    =====
    
    Calls the ``.solve()`` method of the underlying FLINT matrix.
    
    For now this will raise an error if the domain is :ref:`ZZ` but will
    use the FLINT method for :ref:`QQ`.
    
    The FLINT methods for :ref:`ZZ` and :ref:`QQ` are ``fmpz_mat_solve``
    and ``fmpq_mat_solve`` respectively. The ``fmpq_mat_solve`` method
    uses one of two algorithms:
    
    - For small matrices (<25 rows) it clears denominators between the
      matrix and rhs and uses ``fmpz_mat_solve``.
    - For larger matrices it uses ``fmpq_mat_solve_dixon`` which is a
      modular approach with CRT reconstruction over :ref:`QQ`.
    
    The ``fmpz_mat_solve`` method uses one of four algorithms:
    
    - For very small (<= 3x3) matrices it uses a Cramer's rule.
    - For small (<= 15x15) matrices it uses a fraction-free LU solve.
    - Otherwise it uses either Dixon or another multimodular approach.
    
    See Also
    ========
    
    sympy.polys.matrices.domainmatrix.DomainMatrix.lu_solve
        Higher level interface to solve a matrix equation.
    """
    # 检查矩阵的域是否匹配，如果不匹配则抛出异常
    if not self.domain == rhs.domain:
        raise DMDomainError("Domains must match: %s != %s" % (self.domain, rhs.domain))
    
    # 检查矩阵的域是否为一个字段，如果不是则抛出异常
    if not self.domain.is_Field:
        raise DMDomainError("Field expected, got %s" % self.domain)
    
    # 获取矩阵的维度信息
    m, n = self.shape
    j, k = rhs.shape
    
    # 检查矩阵的维度是否匹配，如果不匹配则抛出异常
    if m != j:
        raise DMShapeError("Matrix size mismatch: %s * %s vs %s * %s" % (m, n, j, k))
    
    # 计算解的形状
    sol_shape = (n, k)
    
    # 对于非方阵，将当前矩阵和右侧矩阵转换为双精度矩阵并求解
    if m != n:
        return self.to_ddm().lu_solve(rhs.to_ddm()).to_dfm()
    
    try:
        # 使用 FLINT 库的 solve 方法求解矩阵方程
        sol = self.rep.solve(rhs.rep)
    except ZeroDivisionError:
        raise DMNonInvertibleMatrixError("Matrix det == 0; not invertible.")
    
    # 返回新的解矩阵对象
    return self._new(sol, sol_shape, self.domain)
    def nullspace(self):
        """Return a basis for the nullspace of the matrix."""
        # Code to compute nullspace using flint:
        #
        # V, nullity = self.rep.nullspace()
        # V_dfm = self._new_rep(V)._extract(range(self.rows), range(nullity))
        #
        # XXX: That gives the nullspace but does not give us nonpivots. So we
        # use the slower DDM method anyway. It would be better to change the
        # signature of the nullspace method to not return nonpivots.
        #
        # XXX: Also python-flint exposes a nullspace method for fmpz_mat but
        # not for fmpq_mat. This is the reverse of the situation for DDM etc
        # which only allow nullspace over a field. The nullspace method for
        # DDM, SDM etc should be changed to allow nullspace over ZZ as well.
        # The DomainMatrix nullspace method does allow the domain to be a ring
        # but does not directly call the lower-level nullspace methods and uses
        # rref_den instead. Nullspace methods should also be added to all
        # matrix types in python-flint.
        
        # Convert matrix to DDM (Domain Dense Matrix) and compute nullspace
        ddm, nonpivots = self.to_ddm().nullspace()
        return ddm.to_dfm(), nonpivots

    def nullspace_from_rref(self, pivots=None):
        """Return a basis for the nullspace of the matrix."""
        # XXX: Use the flint nullspace method!!!
        
        # Convert matrix to SDM (Sparse Dense Matrix) and compute nullspace from reduced row echelon form
        sdm, nonpivots = self.to_sdm().nullspace_from_rref(pivots=pivots)
        return sdm.to_dfm(), nonpivots

    def particular(self):
        """Return a particular solution to the system."""
        
        # Convert matrix to DDM and compute particular solution
        return self.to_ddm().particular().to_dfm()

    def _lll(self, transform=False, delta=0.99, eta=0.51, rep='zbasis', gram='approx'):
        """Call the fmpz_mat.lll() method but check rank to avoid segfaults."""

        # XXX: There are tests that pass e.g. QQ(5,6) for delta. That fails
        # with a TypeError in flint because if QQ is fmpq then conversion with
        # float fails. We handle that here but there are two better fixes:
        #
        # - Make python-flint's fmpq convert with float(x)
        # - Change the tests because delta should just be a float.

        # Function to convert delta and eta to floats if they are of type QQ (rational number)
        def to_float(x):
            if QQ.of_type(x):
                return float(x.numerator) / float(x.denominator)
            else:
                return float(x)

        delta = to_float(delta)
        eta = to_float(eta)

        if not 0.25 < delta < 1:
            raise DMValueError("delta must be between 0.25 and 1")

        # Check if the matrix has full row rank
        m, n = self.shape
        if self.rep.rank() != m:
            raise DMRankError("Matrix must have full row rank for Flint LLL.")

        # Call the LLL (Lenstra-Lenstra-Lovász) algorithm from flint
        # to reduce the matrix using LLL reduction
        return self.rep.lll(transform=transform, delta=delta, eta=eta, rep=rep, gram=gram)

    @doctest_depends_on(ground_types='flint')
    # 定义一个名为 lll 的方法，用于计算使用 FLINT 库进行 LLL 约简的基础。

    # 如果矩阵的域不是整数环 ZZ，则抛出异常 DMDomainError
    if self.domain != ZZ:
        raise DMDomainError("ZZ expected, got %s" % self.domain)
    
    # 如果矩阵的行数多于列数，则抛出异常 DMShapeError
    elif self.rows > self.cols:
        raise DMShapeError("Matrix must not have more rows than columns.")
    
    # 调用内部方法 _lll 进行 LLL 约简，获取结果 rep
    rep = self._lll(delta=delta)
    
    # 返回经过 LLL 约简后的新表示结果
    return self._new_rep(rep)

@doctest_depends_on(ground_types='flint')
def lll_transform(self, delta=0.75):
    """Compute LLL-reduced basis and transform using FLINT.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2, 3], [4, 5, 6]]).to_DM().to_dfm()
    >>> M_lll, T = M.lll_transform()
    >>> M_lll
    [[2, 1, 0], [-1, 1, 3]]
    >>> T
    [[-2, 1], [3, -1]]
    >>> T.matmul(M) == M_lll
    True

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.lll
        Higher level interface to compute LLL-reduced basis.
    lll
        Compute LLL-reduced basis without transform matrix.
    """
    
    # 如果矩阵的域不是整数环 ZZ，则抛出异常 DMDomainError
    if self.domain != ZZ:
        raise DMDomainError("ZZ expected, got %s" % self.domain)
    
    # 如果矩阵的行数多于列数，则抛出异常 DMShapeError
    elif self.rows > self.cols:
        raise DMShapeError("Matrix must not have more rows than columns.")
    
    # 调用内部方法 _lll 进行 LLL 约简，并获取约简后的基础和变换矩阵 T
    rep, T = self._lll(transform=True, delta=delta)
    
    # 使用约简后的基础创建新的表示结果 basis
    basis = self._new_rep(rep)
    
    # 使用变换矩阵 T 创建一个新的 DomainMatrix 对象 T_dfm
    T_dfm = self._new(T, (self.rows, self.rows), self.domain)
    
    # 返回约简后的基础和变换矩阵 T_dfm
    return basis, T_dfm
# 避免循环导入

# 从 sympy.polys.matrices.ddm 模块导入 DDM 类
from sympy.polys.matrices.ddm import DDM
# 从 sympy.polys.matrices.ddm 模块导入 SDM 类
from sympy.polys.matrices.ddm import SDM
```
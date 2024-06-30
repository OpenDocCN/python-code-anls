# `D:\src\scipysrc\sympy\sympy\polys\matrices\domainmatrix.py`

```
"""

Module for the DomainMatrix class.

A DomainMatrix represents a matrix with elements that are in a particular
Domain. Each DomainMatrix internally wraps a DDM which is used for the
lower-level operations. The idea is that the DomainMatrix class provides the
convenience routines for converting between Expr and the poly domains as well
as unifying matrices with different domains.

"""
# 导入必要的模块和类
from collections import Counter
from functools import reduce
from typing import Union as tUnion, Tuple as tTuple

# 导入 sympy 外部依赖
from sympy.external.gmpy import GROUND_TYPES
from sympy.utilities.decorator import doctest_depends_on

# 导入 sympy 核心功能
from sympy.core.sympify import _sympify

# 导入自定义的模块和异常类
from ..domains import Domain
from ..constructor import construct_domain
from .exceptions import (
    DMFormatError,
    DMBadInputError,
    DMShapeError,
    DMDomainError,
    DMNotAField,
    DMNonSquareMatrixError,
    DMNonInvertibleMatrixError
)

# 导入域标量和多项式相关模块
from .domainscalar import DomainScalar
from sympy.polys.domains import ZZ, EXRAW, QQ
from sympy.polys.densearith import dup_mul
from sympy.polys.densebasic import dup_convert
from sympy.polys.densetools import (
    dup_mul_ground,
    dup_quo_ground,
    dup_content,
    dup_clear_denoms,
    dup_primitive,
    dup_transform,
)
from sympy.polys.factortools import dup_factor_list
from sympy.polys.polyutils import _sort_factors

# 导入不同的矩阵类型
from .ddm import DDM
from .sdm import SDM
from .dfm import DFM
from .rref import _dm_rref, _dm_rref_den

# 根据 GROUND_TYPES 的不同设定，决定是否跳过特定的 doctest 测试
if GROUND_TYPES != 'flint':
    __doctest_skip__ = ['DomainMatrix.to_dfm', 'DomainMatrix.to_dfm_or_ddm']
else:
    __doctest_skip__ = ['DomainMatrix.from_list']

def DM(rows, domain):
    """Convenient alias for DomainMatrix.from_list

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DM
    >>> DM([[1, 2], [3, 4]], ZZ)
    DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)

    See Also
    ========

    DomainMatrix.from_list
    """
    # 使用 DomainMatrix 类的 from_list 方法创建 DomainMatrix 对象
    return DomainMatrix.from_list(rows, domain)


class DomainMatrix:
    r"""
    Associate Matrix with :py:class:`~.Domain`

    Explanation
    ===========

    DomainMatrix uses :py:class:`~.Domain` for its internal representation
    which makes it faster than the SymPy Matrix class (currently) for many
    common operations, but this advantage makes it not entirely compatible
    with Matrix. DomainMatrix are analogous to numpy arrays with "dtype".
    In the DomainMatrix, each element has a domain such as :ref:`ZZ`
    or  :ref:`QQ(a)`.


    Examples
    ========

    Creating a DomainMatrix from the existing Matrix class:

    >>> from sympy import Matrix
    >>> from sympy.polys.matrices import DomainMatrix
    >>> Matrix1 = Matrix([
    ...    [1, 2],
    ...    [3, 4]])
    >>> A = DomainMatrix.from_Matrix(Matrix1)
    >>> A
    DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}, (2, 2), ZZ)

    Directly forming a DomainMatrix:

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> A = DomainMatrix([
    """
    DomainMatrix class representing a matrix in a specific domain.

    Parameters
    ==========

    rows : list or dict
        Represents elements of DomainMatrix as list of lists or dict of dicts.
    shape : tuple
        Represents dimensions (rows, columns) of DomainMatrix.
    domain : Domain
        Represents the domain of the elements in DomainMatrix.

    Raises
    ======

    TypeError
        If rows is an instance of SDM, DDM, or DFM, or if rows is not a list or dict.

    See Also
    ========

    DDM
    SDM
    Domain
    Poly

    """
    rep: tUnion[SDM, DDM, DFM]
    shape: tTuple[int, int]
    domain: Domain

    def __new__(cls, rows, shape, domain, *, fmt=None):
        """
        Creates a :py:class:`~.DomainMatrix`.

        Parameters
        ==========

        rows : Represents elements of DomainMatrix as list of lists
        shape : Represents dimension of DomainMatrix
        domain : Represents :py:class:`~.Domain` of DomainMatrix

        Raises
        ======

        TypeError
            If any of rows, shape and domain are not provided

        """
        if isinstance(rows, (DDM, SDM, DFM)):
            raise TypeError("Use from_rep to initialise from SDM/DDM")
        elif isinstance(rows, list):
            rep = DDM(rows, shape, domain)
        elif isinstance(rows, dict):
            rep = SDM(rows, shape, domain)
        else:
            msg = "Input should be list-of-lists or dict-of-dicts"
            raise TypeError(msg)

        if fmt is not None:
            if fmt == 'sparse':
                rep = rep.to_sdm()
            elif fmt == 'dense':
                rep = rep.to_ddm()
            else:
                raise ValueError("fmt should be 'sparse' or 'dense'")

        # Use python-flint for dense matrices if possible
        if rep.fmt == 'dense' and DFM._supports_domain(domain):
            rep = rep.to_dfm()

        return cls.from_rep(rep)

    def __reduce__(self):
        """
        Reduces the object to a serializable form for pickling.

        Returns
        =======

        tuple
            Tuple of arguments representing the serialized object.

        Raises
        ======

        RuntimeError
            If the representation format of the matrix is invalid (should not occur in practice).

        """
        rep = self.rep
        if rep.fmt == 'dense':
            arg = self.to_list()
        elif rep.fmt == 'sparse':
            arg = dict(rep)
        else:
            raise RuntimeError # pragma: no cover
        args = (arg, rep.shape, rep.domain)
        return (self.__class__, args)

    def __getitem__(self, key):
        """
        Retrieves an element or slice from the matrix.

        Parameters
        ==========

        key : tuple
            Tuple representing the index or slice to retrieve.

        Returns
        =======

        DomainScalar or DomainMatrix
            Scalar or matrix slice corresponding to the requested index or slice.

        Raises
        ======

        IndexError
            If the row or column index is out of range.

        """
        i, j = key
        m, n = self.shape
        if not (isinstance(i, slice) or isinstance(j, slice)):
            return DomainScalar(self.rep.getitem(i, j), self.domain)

        if not isinstance(i, slice):
            if not -m <= i < m:
                raise IndexError("Row index out of range")
            i = i % m
            i = slice(i, i+1)
        if not isinstance(j, slice):
            if not -n <= j < n:
                raise IndexError("Column index out of range")
            j = j % n
            j = slice(j, j+1)

        return self.from_rep(self.rep.extract_slice(i, j))

    def getitem_sympy(self, i, j):
        """
        Retrieves an element from the matrix and converts it to a sympy expression.

        Parameters
        ==========

        i : int
            Row index of the element.
        j : int
            Column index of the element.

        Returns
        =======

        sympy.Basic
            SymPy expression representing the matrix element.

        """
        return self.domain.to_sympy(self.rep.getitem(i, j))

    def extract(self, rowslist, colslist):
        """
        Extracts a submatrix based on the specified rows and columns.

        Parameters
        ==========

        rowslist : list
            List of row indices or slice indicating rows to extract.
        colslist : list
            List of column indices or slice indicating columns to extract.

        Returns
        =======

        DomainMatrix
            Submatrix extracted based on the specified rows and columns.

        """
        return self.from_rep(self.rep.extract(rowslist, colslist))
    def __setitem__(self, key, value):
        # 解包键元组
        i, j = key
        # 检查值是否符合定义域类型，否则抛出类型错误
        if not self.domain.of_type(value):
            raise TypeError
        # 如果 i 和 j 均为整数，调用 rep 对象的 setitem 方法设置值
        if isinstance(i, int) and isinstance(j, int):
            self.rep.setitem(i, j, value)
        else:
            # 如果 i 和 j 不是整数，抛出未实现错误
            raise NotImplementedError

    @classmethod
    def from_rep(cls, rep):
        """Create a new DomainMatrix efficiently from DDM/SDM.

        Examples
        ========

        Create a :py:class:`~.DomainMatrix` with an dense internal
        representation as :py:class:`~.DDM`:

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.polys.matrices.ddm import DDM
        >>> drep = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> dM = DomainMatrix.from_rep(drep)
        >>> dM
        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)

        Create a :py:class:`~.DomainMatrix` with a sparse internal
        representation as :py:class:`~.SDM`:

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import ZZ
        >>> drep = SDM({0:{1:ZZ(1)},1:{0:ZZ(2)}}, (2, 2), ZZ)
        >>> dM = DomainMatrix.from_rep(drep)
        >>> dM
        DomainMatrix({0: {1: 1}, 1: {0: 2}}, (2, 2), ZZ)

        Parameters
        ==========

        rep: SDM or DDM
            The internal sparse or dense representation of the matrix.

        Returns
        =======

        DomainMatrix
            A :py:class:`~.DomainMatrix` wrapping *rep*.

        Notes
        =====

        This takes ownership of rep as its internal representation. If rep is
        being mutated elsewhere then a copy should be provided to
        ``from_rep``. Only minimal verification or checking is done on *rep*
        as this is supposed to be an efficient internal routine.

        """
        # 检查 rep 是否为 DDM 或 SDM 类型，否则抛出类型错误
        if not (isinstance(rep, (DDM, SDM)) or (DFM is not None and isinstance(rep, DFM))):
            raise TypeError("rep should be of type DDM or SDM")
        # 创建一个新的 DomainMatrix 实例
        self = super().__new__(cls)
        # 将 rep 赋值给实例的 rep 属性
        self.rep = rep
        # 将 rep 的 shape 属性赋值给实例的 shape 属性
        self.shape = rep.shape
        # 将 rep 的 domain 属性赋值给实例的 domain 属性
        self.domain = rep.domain
        # 返回创建的 DomainMatrix 实例
        return self

    @classmethod
    @doctest_depends_on(ground_types=['python', 'gmpy'])
    `
        def from_list(cls, rows, domain):
            r"""
            Convert a list of lists into a DomainMatrix
    
            Parameters
            ==========
    
            rows: list of lists
                Each element of the inner lists should be either the single arg,
                or tuple of args, that would be passed to the domain constructor
                in order to form an element of the domain. See examples.
    
            Returns
            =======
    
            DomainMatrix containing elements defined in rows
    
            Examples
            ========
    
            >>> from sympy.polys.matrices import DomainMatrix
            >>> from sympy import FF, QQ, ZZ
            >>> A = DomainMatrix.from_list([[1, 0, 1], [0, 0, 1]], ZZ)
            >>> A
            DomainMatrix([[1, 0, 1], [0, 0, 1]], (2, 3), ZZ)
            >>> B = DomainMatrix.from_list([[1, 0, 1], [0, 0, 1]], FF(7))
            >>> B
            DomainMatrix([[1 mod 7, 0 mod 7, 1 mod 7], [0 mod 7, 0 mod 7, 1 mod 7]], (2, 3), GF(7))
            >>> C = DomainMatrix.from_list([[(1, 2), (3, 1)], [(1, 4), (5, 1)]], QQ)
            >>> C
            DomainMatrix([[1/2, 3], [1/4, 5]], (2, 2), QQ)
    
            See Also
            ========
    
            from_list_sympy
    
            """
            # 获取 rows 的行数
            nrows = len(rows)
            # 获取列数，如果没有行，则列数为 0
            ncols = 0 if not nrows else len(rows[0])
            # 定义一个 lambda 函数，根据元素 e 的类型来调用 domain 的构造函数
            conv = lambda e: domain(*e) if isinstance(e, tuple) else domain(e)
            # 使用列表推导式将每一行的元素转换为域的元素
            domain_rows = [[conv(e) for e in row] for row in rows]
            # 返回一个 DomainMatrix 对象，包含转换后的数据，行数，列数和域
            return DomainMatrix(domain_rows, (nrows, ncols), domain)
    
        @classmethod
        def from_list_sympy(cls, nrows, ncols, rows, **kwargs):
            r"""
            Convert a list of lists of Expr into a DomainMatrix using construct_domain
    
            Parameters
            ==========
    
            nrows: number of rows
            ncols: number of columns
            rows: list of lists
    
            Returns
            =======
    
            DomainMatrix containing elements of rows
    
            Examples
            ========
    
            >>> from sympy.polys.matrices import DomainMatrix
            >>> from sympy.abc import x, y, z
            >>> A = DomainMatrix.from_list_sympy(1, 3, [[x, y, z]])
            >>> A
            DomainMatrix([[x, y, z]], (1, 3), ZZ[x,y,z])
    
            See Also
            ========
    
            sympy.polys.constructor.construct_domain, from_dict_sympy
    
            """
            # 确保行数与传入的 nrows 相等
            assert len(rows) == nrows
            # 确保每一行的列数与传入的 ncols 相等
            assert all(len(row) == ncols for row in rows)
    
            # 将每个元素转换为 SymPy 表达式对象
            items_sympy = [_sympify(item) for row in rows for item in row]
    
            # 获取域和元素的映射关系
            domain, items_domain = cls.get_domain(items_sympy, **kwargs)
    
            # 根据行列数，将元素映射到域行上
            domain_rows = [[items_domain[ncols*r + c] for c in range(ncols)] for r in range(nrows)]
    
            # 返回一个 DomainMatrix 对象，包含构造后的数据，行数，列数和域
            return DomainMatrix(domain_rows, (nrows, ncols), domain)
    
        @classmethod
    @classmethod
    def from_dict_sympy(cls, nrows, ncols, elemsdict, **kwargs):
        """
        从字典形式的输入创建 DomainMatrix 的类方法

        Parameters
        ==========

        nrows: 矩阵的行数
        ncols: 矩阵的列数
        elemsdict: 包含非零元素的字典字典形式的 DomainMatrix

        Returns
        =======

        包含 elemsdict 元素的 DomainMatrix 对象

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.abc import x,y,z
        >>> elemsdict = {0: {0:x}, 1:{1: y}, 2: {2: z}}
        >>> A = DomainMatrix.from_dict_sympy(3, 3, elemsdict)
        >>> A
        DomainMatrix({0: {0: x}, 1: {1: y}, 2: {2: z}}, (3, 3), ZZ[x,y,z])

        See Also
        ========

        from_list_sympy

        """
        # 检查所有行索引是否在有效范围内
        if not all(0 <= r < nrows for r in elemsdict):
            raise DMBadInputError("Row out of range")
        # 检查所有列索引是否在有效范围内
        if not all(0 <= c < ncols for row in elemsdict.values() for c in row):
            raise DMBadInputError("Column out of range")

        # 将 elemsdict 中的元素转化为 sympy 对象列表
        items_sympy = [_sympify(item) for row in elemsdict.values() for item in row.values()]
        # 获取元素的定义域和转化后的元素列表
        domain, items_domain = cls.get_domain(items_sympy, **kwargs)

        idx = 0
        items_dict = {}
        # 遍历 elemsdict 创建元素字典
        for i, row in elemsdict.items():
            items_dict[i] = {}
            for j in row:
                items_dict[i][j] = items_domain[idx]
                idx += 1

        # 返回 DomainMatrix 对象
        return DomainMatrix(items_dict, (nrows, ncols), domain)

    @classmethod
    def from_Matrix(cls, M, fmt='sparse',**kwargs):
        r"""
        将 Matrix 转换为 DomainMatrix 的类方法

        Parameters
        ==========

        M: Matrix 对象

        Returns
        =======

        返回与 M 元素相同的 DomainMatrix 对象

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.polys.matrices import DomainMatrix
        >>> M = Matrix([
        ...    [1.0, 3.4],
        ...    [2.4, 1]])
        >>> A = DomainMatrix.from_Matrix(M)
        >>> A
        DomainMatrix({0: {0: 1.0, 1: 3.4}, 1: {0: 2.4, 1: 1.0}}, (2, 2), RR)

        可以使用 fmt='dense' 保持内部表示为 ddm
        >>> from sympy import Matrix, QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix.from_Matrix(Matrix([[QQ(1, 2), QQ(3, 4)], [QQ(0, 1), QQ(0, 1)]]), fmt='dense')
        >>> A.rep
        [[1/2, 3/4], [0, 0]]

        See Also
        ========

        Matrix

        """
        # 如果 fmt 为 'dense'，则调用 from_list_sympy 方法
        if fmt == 'dense':
            return cls.from_list_sympy(*M.shape, M.tolist(), **kwargs)

        # 否则调用 from_dict_sympy 方法，转换为稀疏表示
        return cls.from_dict_sympy(*M.shape, M.todod(), **kwargs)

    @classmethod
    def get_domain(cls, items_sympy, **kwargs):
        """
        获取元素的定义域的类方法

        Parameters
        ==========

        items_sympy: sympy 对象的列表

        Returns
        =======

        返回元素的定义域和转换后的元素列表

        """
        # 使用 construct_domain 函数获取元素的定义域和转换后的元素列表
        K, items_K = construct_domain(items_sympy, **kwargs)
        return K, items_K
    def choose_domain(self, **opts):
        """
        Convert to a domain found by :func:`~.construct_domain`.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> M = DM([[1, 2], [3, 4]], ZZ)
        >>> M
        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)
        >>> M.choose_domain(field=True)
        DomainMatrix([[1, 2], [3, 4]], (2, 2), QQ)

        >>> from sympy.abc import x
        >>> M = DM([[1, x], [x**2, x**3]], ZZ[x])
        >>> M.choose_domain(field=True).domain
        ZZ(x)

        Keyword arguments are passed to :func:`~.construct_domain`.

        See Also
        ========

        construct_domain
        convert_to
        """
        # 将当前矩阵转换为由 construct_domain 函数确定的域
        elements, data = self.to_sympy().to_flat_nz()
        # 调用 construct_domain 函数获取域 dom 和转换后的元素 elements_dom
        dom, elements_dom = construct_domain(elements, **opts)
        # 使用转换后的元素 elements_dom、数据 data 和域 dom，构建并返回一个新的 DomainMatrix 对象
        return self.from_flat_nz(elements_dom, data, dom)

    def copy(self):
        # 复制当前 DomainMatrix 对象的表示并返回新的 DomainMatrix 对象
        return self.from_rep(self.rep.copy())

    def convert_to(self, K):
        r"""
        Change the domain of DomainMatrix to desired domain or field

        Parameters
        ==========

        K : Represents the desired domain or field.
            Alternatively, ``None`` may be passed, in which case this method
            just returns a copy of this DomainMatrix.

        Returns
        =======

        DomainMatrix
            DomainMatrix with the desired domain or field

        Examples
        ========

        >>> from sympy import ZZ, ZZ_I
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.convert_to(ZZ_I)
        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ_I)

        """
        if K == self.domain:
            return self.copy()

        rep = self.rep

        # DFM、DDM 和 SDM 类型不进行任何隐式转换，因此在 DDM 和 DFM 之间进行转换管理
        if rep.is_DFM and not DFM._supports_domain(K):
            rep_K = rep.to_ddm().convert_to(K)
        elif rep.is_DDM and DFM._supports_domain(K):
            rep_K = rep.convert_to(K).to_dfm()
        else:
            rep_K = rep.convert_to(K)

        # 使用转换后的表示 rep_K，构建并返回一个新的 DomainMatrix 对象
        return self.from_rep(rep_K)

    def to_sympy(self):
        # 将当前 DomainMatrix 对象转换为表示 EXRAW 的 SymPy 对象并返回
        return self.convert_to(EXRAW)

    def to_field(self):
        r"""
        Returns a DomainMatrix with the appropriate field

        Returns
        =======

        DomainMatrix
            DomainMatrix with the appropriate field

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.to_field()
        DomainMatrix([[1, 2], [3, 4]], (2, 2), QQ)

        """
        # 获取当前域的对应字段 K
        K = self.domain.get_field()
        # 将当前 DomainMatrix 对象转换为字段 K，并返回新的 DomainMatrix 对象
        return self.convert_to(K)
    def to_sparse(self):
        """
        Return a sparse DomainMatrix representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)
        >>> A.rep
        [[1, 0], [0, 2]]
        >>> B = A.to_sparse()
        >>> B.rep
        {0: {0: 1}, 1: {1: 2}}
        """
        # 如果当前矩阵已经是稀疏格式，则直接返回自身
        if self.rep.fmt == 'sparse':
            return self

        # 否则，将当前矩阵转换为稀疏格式，并返回转换后的矩阵对象
        return self.from_rep(self.rep.to_sdm())

    def to_dense(self):
        """
        Return a dense DomainMatrix representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix({0: {0: 1}, 1: {1: 2}}, (2, 2), QQ)
        >>> A.rep
        {0: {0: 1}, 1: {1: 2}}
        >>> B = A.to_dense()
        >>> B.rep
        [[1, 0], [0, 2]]

        """
        # 将当前矩阵的表示方式赋给局部变量 rep
        rep = self.rep

        # 如果当前矩阵已经是密集格式，则直接返回自身
        if rep.fmt == 'dense':
            return self

        # 否则，将当前矩阵转换为密集格式，并返回转换后的矩阵对象
        return self.from_rep(rep.to_dfm_or_ddm())

    def to_ddm(self):
        """
        Return a :class:`~.DDM` representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix({0: {0: 1}, 1: {1: 2}}, (2, 2), QQ)
        >>> ddm = A.to_ddm()
        >>> ddm
        [[1, 0], [0, 2]]
        >>> type(ddm)
        <class 'sympy.polys.matrices.ddm.DDM'>

        See Also
        ========

        to_sdm
        to_dense
        sympy.polys.matrices.ddm.DDM.to_sdm
        """
        # 将当前矩阵的表示转换为 DDM 格式，并返回转换后的结果
        return self.rep.to_ddm()

    def to_sdm(self):
        """
        Return a :class:`~.SDM` representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)
        >>> sdm = A.to_sdm()
        >>> sdm
        {0: {0: 1}, 1: {1: 2}}
        >>> type(sdm)
        <class 'sympy.polys.matrices.sdm.SDM'>

        See Also
        ========

        to_ddm
        to_sparse
        sympy.polys.matrices.sdm.SDM.to_ddm
        """
        # 将当前矩阵的表示转换为 SDM 格式，并返回转换后的结果
        return self.rep.to_sdm()

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm(self):
        """
        Return a :class:`~.DFM` representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)
        >>> dfm = A.to_dfm()
        >>> dfm
        [[1, 0], [0, 2]]
        >>> type(dfm)
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        to_ddm
        to_dense
        DFM
        """
        # 将当前矩阵的表示转换为 DFM 格式，并返回转换后的结果
        return self.rep.to_dfm()

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm_or_ddm(self):
        """
        Return a :class:`~.DFM` or :class:`~.DDM` representation of *self*.

        Explanation
        ===========

        The :class:`~.DFM` representation can only be used if the ground types
        are ``flint`` and the ground domain is supported by ``python-flint``.
        This method will return a :class:`~.DFM` representation if possible,
        but will return a :class:`~.DDM` representation otherwise.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)
        >>> dfm = A.to_dfm_or_ddm()
        >>> dfm
        [[1, 0], [0, 2]]
        >>> type(dfm)  # Depends on the ground domain and ground types
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        to_ddm: Always return a :class:`~.DDM` representation.
        to_dfm: Returns a :class:`~.DFM` representation or raise an error.
        to_dense: Convert internally to a :class:`~.DFM` or :class:`~.DDM`
        DFM: The :class:`~.DFM` dense FLINT matrix representation.
        DDM: The Python :class:`~.DDM` dense domain matrix representation.
        """
        return self.rep.to_dfm_or_ddm()

    @classmethod
    def _unify_domain(cls, *matrices):
        """Convert matrices to a common domain
        
        Explanation
        ===========
        
        Create a set of domains for all input matrices and check if there's only one unique domain.
        If so, return the input matrices unchanged. Otherwise, unify all domains into a single domain
        and convert each matrix to that unified domain.

        Parameters
        ==========
        matrices : tuple of matrices
            Input matrices to be unified.

        Returns
        =======
        tuple
            Tuple of matrices all converted to a common domain.

        See Also
        ========
        
        DomainMatrix: Class representing matrices over various domains.
        """
        domains = {matrix.domain for matrix in matrices}
        if len(domains) == 1:
            return matrices
        domain = reduce(lambda x, y: x.unify(y), domains)
        return tuple(matrix.convert_to(domain) for matrix in matrices)

    @classmethod
    def _unify_fmt(cls, *matrices, fmt=None):
        """Convert matrices to the same format.

        Explanation
        ===========
        
        Determine the format of each input matrix and check if they all share the same format.
        If they do, return the matrices unchanged. If not, convert all matrices to a specified format
        given by *fmt* ('dense' or 'sparse').

        Parameters
        ==========
        matrices : tuple of matrices
            Input matrices to be converted.
        fmt : str, optional
            Desired format to which matrices should be converted ('dense' or 'sparse').

        Returns
        =======
        tuple
            Tuple of matrices converted to the specified format.

        Raises
        ======
        ValueError
            If *fmt* is neither 'sparse' nor 'dense'.

        See Also
        ========
        
        to_sparse: Convert matrix to sparse format.
        to_dense: Convert matrix to dense format.
        """
        formats = {matrix.rep.fmt for matrix in matrices}
        if len(formats) == 1:
            return matrices
        if fmt == 'sparse':
            return tuple(matrix.to_sparse() for matrix in matrices)
        elif fmt == 'dense':
            return tuple(matrix.to_dense() for matrix in matrices)
        else:
            raise ValueError("fmt should be 'sparse' or 'dense'")
    def unify(self, *others, fmt=None):
        """
        Unifies the domains and the format of self and other
        matrices.

        Parameters
        ==========

        others : DomainMatrix
            Other DomainMatrix objects to unify with self.

        fmt: string 'dense', 'sparse' or `None` (default)
            The preferred format to convert to if self and other are not
            already in the same format. If `None` or not specified then no
            conversion if performed.

        Returns
        =======

        Tuple[DomainMatrix]
            Matrices with unified domain and format

        Examples
        ========

        Unify the domain of DomainMatrix that have different domains:

        >>> from sympy import ZZ, QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
        >>> B = DomainMatrix([[QQ(1, 2), QQ(2)]], (1, 2), QQ)
        >>> Aq, Bq = A.unify(B)
        >>> Aq
        DomainMatrix([[1, 2]], (1, 2), QQ)
        >>> Bq
        DomainMatrix([[1/2, 2]], (1, 2), QQ)

        Unify the format (dense or sparse):

        >>> A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
        >>> B = DomainMatrix({0:{0: ZZ(1)}}, (2, 2), ZZ)
        >>> B.rep
        {0: {0: 1}}

        >>> A2, B2 = A.unify(B, fmt='dense')
        >>> B2.rep
        [[1, 0], [0, 0]]

        See Also
        ========

        convert_to, to_dense, to_sparse

        """
        matrices = (self,) + others
        # 统一多个 DomainMatrix 对象的定义域
        matrices = DomainMatrix._unify_domain(*matrices)
        if fmt is not None:
            # 根据指定格式（dense 或 sparse），统一多个 DomainMatrix 对象的表示格式
            matrices = DomainMatrix._unify_fmt(*matrices, fmt=fmt)
        return matrices

    def to_Matrix(self):
        r"""
        Convert DomainMatrix to Matrix

        Returns
        =======

        Matrix
            MutableDenseMatrix for the DomainMatrix

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.to_Matrix()
        Matrix([
            [1, 2],
            [3, 4]])

        See Also
        ========

        from_Matrix

        """
        from sympy.matrices.dense import MutableDenseMatrix

        # 如果 DomainMatrix 的域是 ZZ、QQ 或 EXRAW，则根据表示的格式决定转换方法
        if self.domain in (ZZ, QQ, EXRAW):
            if self.rep.fmt == "sparse":
                rep = self.copy()
            else:
                rep = self.to_sparse()
        else:
            # 否则，先转换为 EXRAW 类型再转换为稀疏表示
            rep = self.convert_to(EXRAW).to_sparse()

        return MutableDenseMatrix._fromrep(rep)

    def to_list(self):
        """
        Convert :class:`DomainMatrix` to list of lists.

        See Also
        ========

        from_list
        to_list_flat
        to_flat_nz
        to_dok
        """
        return self.rep.to_list()
    def to_list_flat(self):
        """
        Convert :class:`DomainMatrix` to flat list.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> A.to_list_flat()
        [1, 2, 3, 4]

        See Also
        ========

        from_list_flat
        to_list
        to_flat_nz
        to_dok
        """
        # 调用代表对象的 `to_list_flat` 方法来获取扁平化后的列表
        return self.rep.to_list_flat()

    @classmethod
    def from_list_flat(cls, elements, shape, domain):
        """
        Create :class:`DomainMatrix` from flat list.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> element_list = [ZZ(1), ZZ(2), ZZ(3), ZZ(4)]
        >>> A = DomainMatrix.from_list_flat(element_list, (2, 2), ZZ)
        >>> A
        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)
        >>> A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        to_list_flat
        """
        # 调用 `DDM.from_list_flat` 方法创建一个 `DDM` 对象
        ddm = DDM.from_list_flat(elements, shape, domain)
        # 调用当前类的 `from_rep` 方法，并将 `ddm.to_dfm_or_ddm()` 的结果作为参数返回
        return cls.from_rep(ddm.to_dfm_or_ddm())

    def to_flat_nz(self):
        """
        Convert :class:`DomainMatrix` to list of nonzero elements and data.

        Explanation
        ===========

        Returns a tuple ``(elements, data)`` where ``elements`` is a list of
        elements of the matrix with zeros possibly excluded. The matrix can be
        reconstructed by passing these to :meth:`from_flat_nz`. The idea is to
        be able to modify a flat list of the elements and then create a new
        matrix of the same shape with the modified elements in the same
        positions.

        The format of ``data`` differs depending on whether the underlying
        representation is dense or sparse but either way it represents the
        positions of the elements in the list in a way that
        :meth:`from_flat_nz` can use to reconstruct the matrix. The
        :meth:`from_flat_nz` method should be called on the same
        :class:`DomainMatrix` that was used to call :meth:`to_flat_nz`.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [1, 2, 3, 4]
        >>> A == A.from_flat_nz(elements, data, A.domain)
        True

        Create a matrix with the elements doubled:

        >>> elements_doubled = [2*x for x in elements]
        >>> A2 = A.from_flat_nz(elements_doubled, data, A.domain)
        >>> A2 == 2*A
        True

        See Also
        ========

        from_flat_nz
        """
        # 调用代表对象的 `to_flat_nz` 方法获取非零元素和数据的元组
        return self.rep.to_flat_nz()
    def from_flat_nz(self, elements, data, domain):
        """
        从平坦的非零元素表示重建 :class:`DomainMatrix`。

        调用 :meth:`to_flat_nz` 后使用此方法。

        See Also
        ========

        to_flat_nz
        """
        # 调用 self.rep 的 from_flat_nz 方法，生成表示 DomainMatrix 的对象
        rep = self.rep.from_flat_nz(elements, data, domain)
        # 使用 rep 构建并返回新的 DomainMatrix 对象
        return self.from_rep(rep)

    def to_dod(self):
        """
        将 :class:`DomainMatrix` 转换为字典的字典（dod）格式。

        Explanation
        ===========

        返回一个表示矩阵的字典的字典。

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2), ZZ(0)], [ZZ(3), ZZ(0), ZZ(4)]], ZZ)
        >>> A.to_dod()
        {0: {0: 1, 1: 2}, 1: {0: 3, 2: 4}}
        >>> A.to_sparse() == A.from_dod(A.to_dod(), A.shape, A.domain)
        True
        >>> A == A.from_dod_like(A.to_dod())
        True

        See Also
        ========

        from_dod
        from_dod_like
        to_dok
        to_list
        to_list_flat
        to_flat_nz
        sympy.matrices.matrixbase.MatrixBase.todod
        """
        return self.rep.to_dod()

    @classmethod
    def from_dod(cls, dod, shape, domain):
        """
        从字典的字典（dod）格式创建稀疏的 :class:`DomainMatrix`。

        See :meth:`to_dod` for explanation.

        See Also
        ========

        to_dod
        from_dod_like
        """
        return cls.from_rep(SDM.from_dod(dod, shape, domain))

    def from_dod_like(self, dod, domain=None):
        """
        从字典的字典（dod）格式创建类似于 ``self`` 的 :class:`DomainMatrix`。

        See :meth:`to_dod` for explanation.

        See Also
        ========

        to_dod
        from_dod
        """
        if domain is None:
            domain = self.domain
        return self.from_rep(self.rep.from_dod(dod, self.shape, domain))

    def to_dok(self):
        """
        将 :class:`DomainMatrix` 转换为字典的键（dok）格式。

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(0)],
        ...    [ZZ(0), ZZ(4)]], (2, 2), ZZ)
        >>> A.to_dok()
        {(0, 0): 1, (1, 1): 4}

        可以通过调用 :meth:`from_dok` 重建矩阵，尽管重建的矩阵总是稀疏格式：

        >>> A.to_sparse() == A.from_dok(A.to_dok(), A.shape, A.domain)
        True

        See Also
        ========

        from_dok
        to_list
        to_list_flat
        to_flat_nz
        """
        return self.rep.to_dok()
    def from_dok(cls, dok, shape, domain):
        """
        Create :class:`DomainMatrix` from dictionary of keys (dok) format.

        See :meth:`to_dok` for explanation.

        See Also
        ========

        to_dok
        """
        # 使用静态方法 from_rep 创建 DomainMatrix 对象，从给定的 dok 格式字典、形状和域创建
        return cls.from_rep(SDM.from_dok(dok, shape, domain))

    def iter_values(self):
        """
        Iterate over nonzero elements of the matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[ZZ(1), ZZ(0)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> list(A.iter_values())
        [1, 3, 4]

        See Also
        ========

        iter_items
        to_list_flat
        sympy.matrices.matrixbase.MatrixBase.iter_values
        """
        # 调用 self.rep 的 iter_values 方法，迭代矩阵中非零元素的值
        return self.rep.iter_values()

    def iter_items(self):
        """
        Iterate over indices and values of nonzero elements of the matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[ZZ(1), ZZ(0)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> list(A.iter_items())
        [((0, 0), 1), ((1, 0), 3), ((1, 1), 4)]

        See Also
        ========

        iter_values
        to_dok
        sympy.matrices.matrixbase.MatrixBase.iter_items
        """
        # 调用 self.rep 的 iter_items 方法，迭代矩阵中非零元素的索引和值
        return self.rep.iter_items()

    def nnz(self):
        """
        Number of nonzero elements in the matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[1, 0], [0, 4]], ZZ)
        >>> A.nnz()
        2
        """
        # 返回矩阵中非零元素的数量，调用 self.rep 的 nnz 方法
        return self.rep.nnz()

    def __repr__(self):
        # 返回 DomainMatrix 对象的字符串表示，包括其表示形式、形状和域
        return 'DomainMatrix(%s, %r, %r)' % (str(self.rep), self.shape, self.domain)

    def transpose(self):
        """Matrix transpose of ``self``"""
        # 返回矩阵的转置，调用 self.from_rep 方法创建相应的转置对象
        return self.from_rep(self.rep.transpose())

    def flat(self):
        rows, cols = self.shape
        # 返回矩阵的扁平化表示，即将矩阵中的元素按行优先展开成列表
        return [self[i,j].element for i in range(rows) for j in range(cols)]

    @property
    def is_zero_matrix(self):
        # 返回表示对象是否为零矩阵的布尔值，调用 self.rep 的 is_zero_matrix 方法
        return self.rep.is_zero_matrix()

    @property
    def is_upper(self):
        """
        Says whether this matrix is upper-triangular. True can be returned
        even if the matrix is not square.
        """
        # 返回表示对象是否为上三角矩阵的布尔值，即使矩阵不是方阵也可以返回 True，调用 self.rep 的 is_upper 方法
        return self.rep.is_upper()

    @property
    def is_lower(self):
        """
        Says whether this matrix is lower-triangular. True can be returned
        even if the matrix is not square.
        """
        # 返回表示对象是否为下三角矩阵的布尔值，即使矩阵不是方阵也可以返回 True，调用 self.rep 的 is_lower 方法
        return self.rep.is_lower()

    @property
    def
    def is_diagonal(self):
        """
        True if the matrix is diagonal.

        Can return true for non-square matrices. A matrix is diagonal if
        ``M[i,j] == 0`` whenever ``i != j``.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> M = DM([[ZZ(1), ZZ(0)], [ZZ(0), ZZ(1)]], ZZ)
        >>> M.is_diagonal
        True

        See Also
        ========

        is_upper
        is_lower
        is_square
        diagonal
        """
        # 调用内部表示的方法检查矩阵是否对角化
        return self.rep.is_diagonal()

    def diagonal(self):
        """
        Get the diagonal entries of the matrix as a list.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> M = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> M.diagonal()
        [1, 4]

        See Also
        ========

        is_diagonal
        diag
        """
        # 返回矩阵对角线元素构成的列表
        return self.rep.diagonal()

    @property
    def is_square(self):
        """
        True if the matrix is square.
        """
        # 判断矩阵是否为方阵，即行数等于列数
        return self.shape[0] == self.shape[1]

    def rank(self):
        # 对矩阵进行行简化，并返回主元列的数量，即矩阵的秩
        rref, pivots = self.rref()
        return len(pivots)

    def hstack(A, *B):
        r"""Horizontally stack the given matrices.

        Parameters
        ==========

        B: DomainMatrix
            Matrices to stack horizontally.

        Returns
        =======

        DomainMatrix
            DomainMatrix by stacking horizontally.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix

        >>> A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
        >>> A.hstack(B)
        DomainMatrix([[1, 2, 5, 6], [3, 4, 7, 8]], (2, 4), ZZ)

        >>> C = DomainMatrix([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)
        >>> A.hstack(B, C)
        DomainMatrix([[1, 2, 5, 6, 9, 10], [3, 4, 7, 8, 11, 12]], (2, 6), ZZ)

        See Also
        ========

        unify
        """
        # 将给定的矩阵水平堆叠起来，返回一个新的 DomainMatrix 对象
        A, *B = A.unify(*B, fmt=A.rep.fmt)
        return DomainMatrix.from_rep(A.rep.hstack(*(Bk.rep for Bk in B)))
    def applyfunc(self, func, domain=None):
        # 如果未指定 domain，则使用对象自身的 domain
        if domain is None:
            domain = self.domain
        # 调用对象的 from_rep 方法，将 func 应用到对象的表示上，返回处理后的新对象
        return self.from_rep(self.rep.applyfunc(func, domain))

    def __add__(A, B):
        # 如果 B 不是 DomainMatrix 类型，则返回 NotImplemented
        if not isinstance(B, DomainMatrix):
            return NotImplemented
        # 将 A 和 B 统一到相同格式，并返回它们的加法结果
        A, B = A.unify(B, fmt='dense')
        return A.add(B)

    def __sub__(A, B):
        # 如果 B 不是 DomainMatrix 类型，则返回 NotImplemented
        if not isinstance(B, DomainMatrix):
            return NotImplemented
        # 将 A 和 B 统一到相同格式，并返回它们的减法结果
        A, B = A.unify(B, fmt='dense')
        return A.sub(B)

    def __neg__(A):
        # 返回对象 A 的相反数
        return A.neg()

    def __mul__(A, B):
        """A * B"""
        # 如果 B 是 DomainMatrix 类型，则将 A 和 B 统一到相同格式，并返回它们的矩阵乘法结果
        if isinstance(B, DomainMatrix):
            A, B = A.unify(B, fmt='dense')
            return A.matmul(B)
        # 如果 B 在 A 的 domain 中，则返回 A 与标量 B 的乘积
        elif B in A.domain:
            return A.scalarmul(B)
        # 如果 B 是 DomainScalar 类型，则将 A 和 B 统一后返回 A 与 B 的乘积
        elif isinstance(B, DomainScalar):
            A, B = A.unify(B)
            return A.scalarmul(B.element)
        else:
            # 否则返回 NotImplemented
            return NotImplemented

    def __rmul__(A, B):
        # 如果 B 在 A 的 domain 中，则返回 A 与标量 B 的右乘
        if B in A.domain:
            return A.rscalarmul(B)
        # 如果 B 是 DomainScalar 类型，则将 A 和 B 统一后返回 A 与 B 的右乘
        elif isinstance(B, DomainScalar):
            A, B = A.unify(B)
            return A.rscalarmul(B.element)
        else:
            # 否则返回 NotImplemented
            return NotImplemented

    def __pow__(A, n):
        """A ** n"""
        # 如果 n 不是整数类型，则返回 NotImplemented
        if not isinstance(n, int):
            return NotImplemented
        # 返回 A 的 n 次幂
        return A.pow(n)

    def _check(a, op, b, ashape, bshape):
        # 检查两个对象的 domain 是否相同，如果不同则抛出异常
        if a.domain != b.domain:
            msg = "Domain mismatch: %s %s %s" % (a.domain, op, b.domain)
            raise DMDomainError(msg)
        # 检查两个对象的形状是否相同，如果不同则抛出异常
        if ashape != bshape:
            msg = "Shape mismatch: %s %s %s" % (a.shape, op, b.shape)
            raise DMShapeError(msg)
        # 检查两个对象的表示格式是否相同，如果不同则抛出异常
        if a.rep.fmt != b.rep.fmt:
            msg = "Format mismatch: %s %s %s" % (a.rep.fmt, op, b.rep.fmt)
            raise DMFormatError(msg)
        # 检查两个对象的类型是否相同，如果不同则抛出异常
        if type(a.rep) != type(b.rep):
            msg = "Type mismatch: %s %s %s" % (type(a.rep), op, type(b.rep))
            raise DMFormatError(msg)
    def add(A, B):
        r"""
        Adds two DomainMatrix matrices of the same Domain

        Parameters
        ==========

        A, B: DomainMatrix
            matrices to add

        Returns
        =======

        DomainMatrix
            DomainMatrix after Addition

        Raises
        ======

        DMShapeError
            If the dimensions of the two DomainMatrix are not equal

        ValueError
            If the domain of the two DomainMatrix are not same

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([
        ...    [ZZ(4), ZZ(3)],
        ...    [ZZ(2), ZZ(1)]], (2, 2), ZZ)

        >>> A.add(B)
        DomainMatrix([[5, 5], [5, 5]], (2, 2), ZZ)

        See Also
        ========

        sub, matmul

        """
        # 检查两个 DomainMatrix 的维度是否相同，抛出异常如果不相同
        A._check('+', B, A.shape, B.shape)
        # 返回由底层表示相加得到的新 DomainMatrix 对象
        return A.from_rep(A.rep.add(B.rep))


    def sub(A, B):
        r"""
        Subtracts two DomainMatrix matrices of the same Domain

        Parameters
        ==========

        A, B: DomainMatrix
            matrices to subtract

        Returns
        =======

        DomainMatrix
            DomainMatrix after Subtraction

        Raises
        ======

        DMShapeError
            If the dimensions of the two DomainMatrix are not equal

        ValueError
            If the domain of the two DomainMatrix are not same

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([
        ...    [ZZ(4), ZZ(3)],
        ...    [ZZ(2), ZZ(1)]], (2, 2), ZZ)

        >>> A.sub(B)
        DomainMatrix([[-3, -1], [1, 3]], (2, 2), ZZ)

        See Also
        ========

        add, matmul

        """
        # 检查两个 DomainMatrix 的维度是否相同，抛出异常如果不相同
        A._check('-', B, A.shape, B.shape)
        # 返回由底层表示相减得到的新 DomainMatrix 对象
        return A.from_rep(A.rep.sub(B.rep))

    def neg(A):
        r"""
        Returns the negative of DomainMatrix

        Parameters
        ==========

        A : Represents a DomainMatrix

        Returns
        =======

        DomainMatrix
            DomainMatrix after Negation

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.neg()
        DomainMatrix([[-1, -2], [-3, -4]], (2, 2), ZZ)

        """
        # 返回由底层表示取反得到的新 DomainMatrix 对象
        return A.from_rep(A.rep.neg())
    def mul(A, b):
        r"""
        Performs term by term multiplication for the second DomainMatrix
        w.r.t first DomainMatrix. Returns a DomainMatrix whose rows are
        list of DomainMatrix matrices created after term by term multiplication.

        Parameters
        ==========

        A, B: DomainMatrix
            matrices to multiply term-wise

        Returns
        =======

        DomainMatrix
            DomainMatrix after term by term multiplication

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> b = ZZ(2)

        >>> A.mul(b)
        DomainMatrix([[2, 4], [6, 8]], (2, 2), ZZ)

        See Also
        ========

        matmul

        """
        # Return the result of term by term multiplication of A with b
        return A.from_rep(A.rep.mul(b))

    def rmul(A, b):
        # Right scalar multiplication of A by b, using from_rep method
        return A.from_rep(A.rep.rmul(b))

    def matmul(A, B):
        r"""
        Performs matrix multiplication of two DomainMatrix matrices

        Parameters
        ==========

        A, B: DomainMatrix
            to multiply

        Returns
        =======

        DomainMatrix
            DomainMatrix after multiplication

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([
        ...    [ZZ(1), ZZ(1)],
        ...    [ZZ(0), ZZ(1)]], (2, 2), ZZ)

        >>> A.matmul(B)
        DomainMatrix([[1, 3], [3, 7]], (2, 2), ZZ)

        See Also
        ========

        mul, pow, add, sub

        """
        # Check dimensions and return the result of matrix multiplication
        A._check('*', B, A.shape[1], B.shape[0])
        return A.from_rep(A.rep.matmul(B.rep))

    def _scalarmul(A, lamda, reverse):
        # Helper function for scalar multiplication based on lambda and reverse flag
        if lamda == A.domain.zero:
            return DomainMatrix.zeros(A.shape, A.domain)
        elif lamda == A.domain.one:
            return A.copy()
        elif reverse:
            return A.rmul(lamda)
        else:
            return A.mul(lamda)

    def scalarmul(A, lamda):
        # Scalar multiplication of A by lamda using _scalarmul with reverse=False
        return A._scalarmul(lamda, reverse=False)

    def rscalarmul(A, lamda):
        # Right scalar multiplication of A by lamda using _scalarmul with reverse=True
        return A._scalarmul(lamda, reverse=True)

    def mul_elementwise(A, B):
        # Element-wise multiplication of matrices A and B
        assert A.domain == B.domain
        return A.from_rep(A.rep.mul_elementwise(B.rep))
    def __truediv__(A, lamda):
        """ Method for Scalar Division"""
        # 检查 lamda 是否为整数或者 ZZ 类型，如果是则将其转换为 DomainScalar 类型
        if isinstance(lamda, int) or ZZ.of_type(lamda):
            lamda = DomainScalar(ZZ(lamda), ZZ)
        # 如果 A 的域是一个 Field 并且 lamda 在这个域内，则将 lamda 转换为对应的 DomainScalar
        elif A.domain.is_Field and lamda in A.domain:
            K = A.domain
            lamda = DomainScalar(K.convert(lamda), K)

        # 如果 lamda 不是 DomainScalar 类型，则返回 NotImplemented
        if not isinstance(lamda, DomainScalar):
            return NotImplemented

        # 将 A 和 lamda 统一为同一域
        A, lamda = A.to_field().unify(lamda)
        # 如果 lamda 的值为零，则引发 ZeroDivisionError
        if lamda.element == lamda.domain.zero:
            raise ZeroDivisionError
        # 如果 lamda 的值为一，则直接返回 A
        if lamda.element == lamda.domain.one:
            return A

        # 返回 A 乘以 lamda 的倒数
        return A.mul(1 / lamda.element)

    def pow(A, n):
        r"""
        Computes A**n

        Parameters
        ==========

        A : DomainMatrix
            待求幂的矩阵 A

        n : int
            A 的指数

        Returns
        =======

        DomainMatrix
            计算结果 A**n 的 DomainMatrix 对象

        Raises
        ======

        NotImplementedError
            如果 n 是负数时抛出异常

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(1)],
        ...    [ZZ(0), ZZ(1)]], (2, 2), ZZ)

        >>> A.pow(2)
        DomainMatrix([[1, 2], [0, 1]], (2, 2), ZZ)

        See Also
        ========

        matmul
            矩阵乘法

        """
        # 获取矩阵 A 的行数和列数
        nrows, ncols = A.shape
        # 如果 A 不是方阵，则抛出异常 DMNonSquareMatrixError
        if nrows != ncols:
            raise DMNonSquareMatrixError('Power of a nonsquare matrix')
        # 如果 n 是负数，则抛出 NotImplementedError
        if n < 0:
            raise NotImplementedError('Negative powers')
        # 如果 n 等于 0，则返回单位矩阵
        elif n == 0:
            return A.eye(nrows, A.domain)
        # 如果 n 等于 1，则返回 A 自身
        elif n == 1:
            return A
        # 如果 n 是奇数，则返回 A 的平方乘以 A**(n-1)
        elif n % 2 == 1:
            return A * A**(n - 1)
        # 如果 n 是偶数，则计算 A 的平方根乘以 A**(n//2)
        else:
            sqrtAn = A ** (n // 2)
            return sqrtAn * sqrtAn
    def scc(self):
        """
        Compute the strongly connected components of a DomainMatrix

        Explanation
        ===========
        
        A square matrix can be considered as the adjacency matrix for a
        directed graph where the row and column indices are the vertices. In
        this graph if there is an edge from vertex ``i`` to vertex ``j`` if
        ``M[i, j]`` is nonzero. This routine computes the strongly connected
        components of that graph which are subsets of the rows and columns that
        are connected by some nonzero element of the matrix. The strongly
        connected components are useful because many operations such as the
        determinant can be computed by working with the submatrices
        corresponding to each component.

        Examples
        ========

        Find the strongly connected components of a matrix:

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> M = DomainMatrix([[ZZ(1), ZZ(0), ZZ(2)],
        ...                   [ZZ(0), ZZ(3), ZZ(0)],
        ...                   [ZZ(4), ZZ(6), ZZ(5)]], (3, 3), ZZ)
        >>> M.scc()
        [[1], [0, 2]]

        Compute the determinant from the components:

        >>> MM = M.to_Matrix()
        >>> MM
        Matrix([
        [1, 0, 2],
        [0, 3, 0],
        [4, 6, 5]])
        >>> MM[[1], [1]]
        Matrix([[3]])
        >>> MM[[0, 2], [0, 2]]
        Matrix([
        [1, 2],
        [4, 5]])
        >>> MM.det()
        -9
        >>> MM[[1], [1]].det() * MM[[0, 2], [0, 2]].det()
        -9

        The components are given in reverse topological order and represent a
        permutation of the rows and columns that will bring the matrix into
        block lower-triangular form:

        >>> MM[[1, 0, 2], [1, 0, 2]]
        Matrix([
        [3, 0, 0],
        [0, 1, 2],
        [6, 4, 5]])

        Returns
        =======

        List of lists of integers
            Each list represents a strongly connected component.

        See also
        ========

        sympy.matrices.matrixbase.MatrixBase.strongly_connected_components
        sympy.utilities.iterables.strongly_connected_components
        """

        # 如果矩阵不是方阵，则抛出异常
        if not self.is_square:
            raise DMNonSquareMatrixError('Matrix must be square for scc')

        # 调用底层表示的 scc 方法计算强连通分量并返回结果
        return self.rep.scc()
    # 定义一个方法用于清除分母，保持矩阵的定义域不变
    def clear_denoms(self, convert=False):
        """
        Clear denominators, but keep the domain unchanged.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[(1,2), (1,3)], [(1,4), (1,5)]], QQ)
        >>> den, Anum = A.clear_denoms()
        >>> den.to_sympy()
        60
        >>> Anum.to_Matrix()
        Matrix([
        [30, 20],
        [15, 12]])
        >>> den * A == Anum
        True

        The numerator matrix will be in the same domain as the original matrix
        unless ``convert`` is set to ``True``:

        >>> A.clear_denoms()[1].domain
        QQ
        >>> A.clear_denoms(convert=True)[1].domain
        ZZ

        The denominator is always in the associated ring:

        >>> A.clear_denoms()[0].domain
        ZZ
        >>> A.domain.get_ring()
        ZZ

        See Also
        ========

        sympy.polys.polytools.Poly.clear_denoms
        clear_denoms_rowwise
        """
        # 将矩阵转换为稀疏非零元素形式
        elems0, data = self.to_flat_nz()

        # 获取当前矩阵的域
        K0 = self.domain
        # 如果域具有关联环，获取其环；否则保持不变
        K1 = K0.get_ring() if K0.has_assoc_Ring else K0

        # 调用函数清除分母，得到分母和转换后的稀疏非零元素
        den, elems1 = dup_clear_denoms(elems0, K0, K1, convert=convert)

        # 如果设置了 convert=True，则分子和分母的域均为 K1
        if convert:
            Kden, Knum = K1, K1
        else:
            # 否则，分母的域为 K1，分子的域为 K0
            Kden, Knum = K1, K0

        # 将分母包装为域标量
        den = DomainScalar(den, Kden)
        # 从稀疏非零元素和数据中恢复出矩阵 num
        num = self.from_flat_nz(elems1, data, Knum)

        # 返回分母和分子
        return den, num
    def clear_denoms_rowwise(self, convert=False):
        """
        Clear denominators from each row of the matrix.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[(1,2), (1,3), (1,4)], [(1,5), (1,6), (1,7)]], QQ)
        >>> den, Anum = A.clear_denoms_rowwise()
        >>> den.to_Matrix()
        Matrix([
        [12,   0],
        [ 0, 210]])
        >>> Anum.to_Matrix()
        Matrix([
        [ 6,  4,  3],
        [42, 35, 30]])

        The denominator matrix is a diagonal matrix with the denominators of
        each row on the diagonal. The invariants are:

        >>> den * A == Anum
        True
        >>> A == den.to_field().inv() * Anum
        True

        The numerator matrix will be in the same domain as the original matrix
        unless ``convert`` is set to ``True``:

        >>> A.clear_denoms_rowwise()[1].domain
        QQ
        >>> A.clear_denoms_rowwise(convert=True)[1].domain
        ZZ

        The domain of the denominator matrix is the associated ring:

        >>> A.clear_denoms_rowwise()[0].domain
        ZZ

        See Also
        ========

        sympy.polys.polytools.Poly.clear_denoms
        clear_denoms
        """
        # 将矩阵转换为字典格式表示
        dod = self.to_dod()

        # 确定当前矩阵的域
        K0 = self.domain
        # 获取矩阵域的环，如果当前域有关联的环的话
        K1 = K0.get_ring() if K0.has_assoc_Ring else K0

        # 初始化对角线元素列表
        diagonals = [K0.one] * self.shape[0]
        # 初始化字典表示的矩阵的数值部分
        dod_num = {}
        
        # 遍历矩阵的每一行
        for i, rowi in dod.items():
            # 拆分每行的索引和元素
            indices, elems = zip(*rowi.items())
            # 清除当前行的分母，并返回分母和分子
            den, elems_num = dup_clear_denoms(elems, K0, K1, convert=convert)
            # 将清除分母后的元素重新组成字典表示的数值部分
            rowi_num = dict(zip(indices, elems_num))
            # 更新对角线元素列表
            diagonals[i] = den
            # 更新字典表示的矩阵的数值部分
            dod_num[i] = rowi_num

        # 根据是否需要转换，确定分母矩阵和分子矩阵的域
        if convert:
            Kden, Knum = K1, K1
        else:
            Kden, Knum = K1, K0

        # 根据对角线元素列表创建对角矩阵作为分母矩阵
        den = self.diag(diagonals, Kden)
        # 根据字典表示的数值部分创建新的矩阵作为分子矩阵
        num = self.from_dod_like(dod_num, Knum)

        # 返回分母矩阵和分子矩阵
        return den, num
    def cancel_denom_elementwise(self, denom):
        """
        Cancel factors between the elements of a matrix and a denominator.

        Returns a matrix of numerators and a matrix of denominators.

        Requires ``gcd`` in the ground domain.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[2, 3], [4, 12]], ZZ)
        >>> denom = ZZ(6)
        >>> numers, denoms = M.cancel_denom_elementwise(denom)
        >>> numers.to_Matrix()
        Matrix([
        [1, 1],
        [2, 2]])
        >>> denoms.to_Matrix()
        Matrix([
        [3, 2],
        [3, 1]])
        >>> M_frac = (M.to_field() / denom).to_Matrix()
        >>> M_frac
        Matrix([
        [1/3, 1/2],
        [2/3,   2]])
        >>> denoms_inverted = denoms.to_Matrix().applyfunc(lambda e: 1/e)
        >>> numers.to_Matrix().multiply_elementwise(denoms_inverted) == M_frac
        True

        Use :meth:`cancel_denom` to cancel factors between the matrix and the
        denominator while preserving the form of a matrix with a scalar
        denominator.

        See Also
        ========

        cancel_denom
        """
        K = self.domain  # 获取矩阵所在的域
        M = self  # 获取当前矩阵对象

        if K.is_zero(denom):  # 检查给定的 denominator 是否为零
            raise ZeroDivisionError('denominator is zero')  # 抛出零除错误
        elif K.is_one(denom):  # 检查给定的 denominator 是否为一
            M_numers = M.copy()  # 复制当前矩阵作为分子矩阵
            M_denoms = M.ones(M.shape, M.domain)  # 创建一个全一矩阵作为分母矩阵
            return (M_numers, M_denoms)

        elements, data = M.to_flat_nz()  # 获取非零元素列表和对应的数据结构

        cofactors = [K.cofactors(numer, denom) for numer in elements]  # 计算每个元素与 denominator 的最大公因数和对应的分子、分母
        gcds, numers, denoms = zip(*cofactors)  # 解压缩得到最大公因数、分子列表和分母列表

        M_numers = M.from_flat_nz(list(numers), data, K)  # 根据分子列表和数据结构创建新的分子矩阵
        M_denoms = M.from_flat_nz(list(denoms), data, K)  # 根据分母列表和数据结构创建新的分母矩阵

        return (M_numers, M_denoms)  # 返回分子矩阵和分母矩阵的元组

    def content(self):
        """
        Return the gcd of the elements of the matrix.

        Requires ``gcd`` in the ground domain.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[2, 4], [4, 12]], ZZ)
        >>> M.content()
        2

        See Also
        ========

        primitive
        cancel_denom
        """
        K = self.domain  # 获取矩阵所在的域
        elements, _ = self.to_flat_nz()  # 获取非零元素列表

        return dup_content(elements, K)  # 调用 dup_content 函数计算矩阵元素的最大公因数并返回
    # 定义一个方法 `primitive`，用于提取矩阵元素的最大公约数
    def primitive(self):
        """
        Factor out gcd of the elements of a matrix.

        Requires ``gcd`` in the ground domain.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[2, 4], [4, 12]], ZZ)
        >>> content, M_primitive = M.primitive()
        >>> content
        2
        >>> M_primitive
        DomainMatrix([[1, 2], [2, 6]], (2, 2), ZZ)
        >>> content * M_primitive == M
        True
        >>> M_primitive.content() == ZZ(1)
        True

        See Also
        ========

        content
        cancel_denom
        """
        # 获取矩阵的定义域
        K = self.domain
        # 将矩阵转换为非零元素的列表以及数据
        elements, data = self.to_flat_nz()
        # 调用 `dup_primitive` 函数计算矩阵元素的最大公约数和提取后的矩阵
        content, prims = dup_primitive(elements, K)
        # 根据提取后的矩阵数据创建新的 `DomainMatrix` 对象
        M_primitive = self.from_flat_nz(prims, data, K)
        # 返回最大公约数和提取后的矩阵
        return content, M_primitive

    # 定义一个方法 `columnspace`，返回矩阵的列空间
    def columnspace(self):
        r"""
        Returns the columnspace for the DomainMatrix

        Returns
        =======

        DomainMatrix
            The columns of this matrix form a basis for the columnspace.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [QQ(1), QQ(-1)],
        ...    [QQ(2), QQ(-2)]], (2, 2), QQ)
        >>> A.columnspace()
        DomainMatrix([[1], [2]], (2, 1), QQ)

        """
        # 检查定义域是否为一个域（即是否为字段）
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        # 对矩阵进行行简化阶梯形，并获取主元列索引
        rref, pivots = self.rref()
        # 获取矩阵的行数和列数
        rows, cols = self.shape
        # 提取矩阵的列空间，返回一个新的 `DomainMatrix` 对象
        return self.extract(range(rows), pivots)

    # 定义一个方法 `rowspace`，返回矩阵的行空间
    def rowspace(self):
        r"""
        Returns the rowspace for the DomainMatrix

        Returns
        =======

        DomainMatrix
            The rows of this matrix form a basis for the rowspace.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [QQ(1), QQ(-1)],
        ...    [QQ(2), QQ(-2)]], (2, 2), QQ)
        >>> A.rowspace()
        DomainMatrix([[1, -1]], (1, 2), QQ)

        """
        # 检查定义域是否为一个域（即是否为字段）
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        # 对矩阵进行行简化阶梯形，并获取主元列索引
        rref, pivots = self.rref()
        # 获取矩阵的行数和列数
        rows, cols = self.shape
        # 提取矩阵的行空间，返回一个新的 `DomainMatrix` 对象
        return self.extract(range(len(pivots)), range(cols))
    def nullspace_from_rref(self, pivots=None):
        """
        Compute nullspace from rref and pivots.

        The domain of the matrix can be any domain.

        The matrix must be in reduced row echelon form already. Otherwise the
        result will be incorrect. Use :meth:`rref` or :meth:`rref_den` first
        to get the reduced row echelon form or use :meth:`nullspace` instead.

        See Also
        ========

        nullspace
        rref
        rref_den
        sympy.polys.matrices.sdm.SDM.nullspace_from_rref
        sympy.polys.matrices.ddm.DDM.nullspace_from_rref
        """
        # 调用特定表示的 nullspace_from_rref 方法，返回零空间和非主列集合
        null_rep, nonpivots = self.rep.nullspace_from_rref(pivots)
        # 根据返回的零空间表示构造一个新的 DomainMatrix 对象
        return self.from_rep(null_rep)

    def inv(self):
        r"""
        Finds the inverse of the DomainMatrix if exists

        Returns
        =======

        DomainMatrix
            DomainMatrix after inverse

        Raises
        ======

        ValueError
            If the domain of DomainMatrix not a Field

        DMNonSquareMatrixError
            If the DomainMatrix is not a not Square DomainMatrix

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...     [QQ(2), QQ(-1), QQ(0)],
        ...     [QQ(-1), QQ(2), QQ(-1)],
        ...     [QQ(0), QQ(0), QQ(2)]], (3, 3), QQ)
        >>> A.inv()
        DomainMatrix([[2/3, 1/3, 1/6], [1/3, 2/3, 1/3], [0, 0, 1/2]], (3, 3), QQ)

        See Also
        ========

        neg

        """
        # 检查 DomainMatrix 的域是否为一个字段，否则抛出异常
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        # 获取矩阵的行数和列数
        m, n = self.shape
        # 如果矩阵不是方阵，抛出非方阵异常
        if m != n:
            raise DMNonSquareMatrixError
        # 调用特定表示的 inv 方法，计算矩阵的逆
        inv = self.rep.inv()
        # 根据逆矩阵的表示构造一个新的 DomainMatrix 对象
        return self.from_rep(inv)

    def det(self):
        r"""
        Returns the determinant of a square :class:`DomainMatrix`.

        Returns
        =======

        determinant: DomainElement
            Determinant of the matrix.

        Raises
        ======

        ValueError
            If the domain of DomainMatrix is not a Field

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.det()
        -2

        """
        # 获取矩阵的行数和列数
        m, n = self.shape
        # 如果矩阵不是方阵，抛出非方阵异常
        if m != n:
            raise DMNonSquareMatrixError
        # 调用特定表示的 det 方法，计算矩阵的行列式
        return self.rep.det()
    def adj_det(self):
        """
        计算方阵 :class:`DomainMatrix` 的伴随矩阵和行列式。

        Returns
        =======

        (adjugate, determinant) : (DomainMatrix, DomainScalar)
            返回该矩阵的伴随矩阵和行列式。

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([
        ...     [ZZ(1), ZZ(2)],
        ...     [ZZ(3), ZZ(4)]], ZZ)
        >>> adjA, detA = A.adj_det()
        >>> adjA
        DomainMatrix([[4, -2], [-3, 1]], (2, 2), ZZ)
        >>> detA
        -2

        See Also
        ========

        adjugate
            返回伴随矩阵。
        det
            返回行列式。
        inv_den
            返回一个矩阵/分母对，表示逆矩阵，可能与伴随矩阵和行列式相比有一个公因子。
        """
        # 获取矩阵的行数和列数
        m, n = self.shape
        # 创建单位矩阵 I_m
        I_m = self.eye((m, m), self.domain)
        # 调用 solve_den_charpoly 方法计算伴随矩阵 adjA 和行列式 detA，check=False 表示不检查参数
        adjA, detA = self.solve_den_charpoly(I_m, check=False)
        # 如果矩阵表示为稠密格式，则将 adjA 转换为稠密矩阵
        if self.rep.fmt == "dense":
            adjA = adjA.to_dense()
        # 返回伴随矩阵 adjA 和行列式 detA
        return adjA, detA

    def adjugate(self):
        """
        计算方阵 :class:`DomainMatrix` 的伴随矩阵。

        伴随矩阵是余子式矩阵的转置，与逆矩阵的关系为::

            adj(A) = det(A) * A.inv()

        与逆矩阵不同，伴随矩阵可以在基本域中计算和表达，无需使用除法或分数。

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> A.adjugate()
        DomainMatrix([[4, -2], [-3, 1]], (2, 2), ZZ)

        Returns
        =======

        DomainMatrix
            返回具有相同域的该矩阵的伴随矩阵。

        See Also
        ========

        adj_det
        """
        # 调用 adj_det 方法获取伴随矩阵 adjA 和行列式 detA
        adjA, detA = self.adj_det()
        # 返回伴随矩阵 adjA
        return adjA
    def inv_den(self, method=None):
        """
        Return the inverse as a :class:`DomainMatrix` with denominator.

        Returns
        =======
        
        (inv, den) : (:class:`DomainMatrix`, :class:`~.DomainElement`)
            The inverse matrix and its denominator.

        This is more or less equivalent to :meth:`adj_det` except that ``inv``
        and ``den`` are not guaranteed to be the adjugate and inverse. The
        ratio ``inv/den`` is equivalent to ``adj/det`` but some factors
        might be cancelled between ``inv`` and ``den``. In simple cases this
        might just be a minus sign so that ``(inv, den) == (-adj, -det)`` but
        factors more complicated than ``-1`` can also be cancelled.
        Cancellation is not guaranteed to be complete so ``inv`` and ``den``
        may not be on lowest terms. The denominator ``den`` will be zero if and
        only if the determinant is zero.

        If the actual adjugate and determinant are needed, use :meth:`adj_det`
        instead. If the intention is to compute the inverse matrix or solve a
        system of equations then :meth:`inv_den` is more efficient.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...     [ZZ(2), ZZ(-1), ZZ(0)],
        ...     [ZZ(-1), ZZ(2), ZZ(-1)],
        ...     [ZZ(0), ZZ(0), ZZ(2)]], (3, 3), ZZ)
        >>> Ainv, den = A.inv_den()
        >>> den
        6
        >>> Ainv
        DomainMatrix([[4, 2, 1], [2, 4, 2], [0, 0, 3]], (3, 3), ZZ)
        >>> A * Ainv == den * A.eye(A.shape, A.domain).to_dense()
        True

        Parameters
        ==========

        method : str, optional
            The method to use to compute the inverse. Can be one of ``None``,
            ``'rref'`` or ``'charpoly'``. If ``None`` then the method is
            chosen automatically (see :meth:`solve_den` for details).

        See Also
        ========

        inv
        det
        adj_det
        solve_den
        """
        # 构造单位矩阵 I，形状为当前矩阵的形状，域为当前矩阵的域
        I = self.eye(self.shape, self.domain)
        # 调用 solve_den 方法计算当前矩阵的逆矩阵和其分母，返回结果
        return self.solve_den(I, method=method)
    def solve_den_rref(self, b):
        """
        Solve matrix equation $Ax = b$ using fraction-free RREF

        Solves the matrix equation $Ax = b$ for $x$ and returns the solution
        as a numerator/denominator pair.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> b = DM([[ZZ(5)], [ZZ(6)]], ZZ)
        >>> xnum, xden = A.solve_den_rref(b)
        >>> xden
        -2
        >>> xnum
        DomainMatrix([[8], [-9]], (2, 1), ZZ)
        >>> A * xnum == xden * b
        True

        See Also
        ========

        solve_den
        solve_den_charpoly
        """
        # 将当前对象赋值给变量 A，表示要解的方程组的系数矩阵
        A = self
        # 获取矩阵 A 和向量 b 的行列数
        m, n = A.shape
        bm, bn = b.shape

        # 检查矩阵 A 和向量 b 的行数是否相同，如果不同则抛出异常
        if m != bm:
            raise DMShapeError("Matrix equation shape mismatch.")

        # 检查矩阵 A 是否为欠定的，即行数小于列数，如果是则抛出异常
        if m < n:
            raise DMShapeError("Underdetermined matrix equation.")

        # 构造增广矩阵 Aaug，将向量 b 作为新的一列加在矩阵 A 的右侧
        Aaug = A.hstack(b)
        # 对增广矩阵 Aaug 进行分数消元得到简化行阶梯形式 Aaug_rref、分母 denom 和主元列 pivots
        Aaug_rref, denom, pivots = Aaug.rref_den()

        # XXX: 在这里检查是否存在超出最后一列的主元列。如果有，则 rref_den 可能执行了一些不必要的消元。
        # 更好的做法是在 rref 方法中增加一个参数，指示应该用多少列进行消元。
        if len(pivots) != n or pivots and pivots[-1] >= n:
            raise DMNonInvertibleMatrixError("Non-unique solution.")

        # 提取出解 x 的分子部分 xnum 和分母部分 xden
        xnum = Aaug_rref[:n, n:]
        xden = denom

        # 返回解 x 的分子和分母
        return xnum, xden
    def solve_den_charpoly(self, b, cp=None, check=True):
        """
        Solve matrix equation $Ax = b$ using the characteristic polynomial.

        This method solves the square matrix equation $Ax = b$ for $x$ using
        the characteristic polynomial without any division or fractions in the
        ground domain.

        Examples
        ========

        Solve a matrix equation over the integers:

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> b = DM([[ZZ(5)], [ZZ(6)]], ZZ)
        >>> xnum, detA = A.solve_den_charpoly(b)
        >>> detA
        -2
        >>> xnum
        DomainMatrix([[8], [-9]], (2, 1), ZZ)
        >>> A * xnum == detA * b
        True

        Parameters
        ==========

        self : DomainMatrix
            The ``n x n`` matrix `A` in the equation `Ax = b`. Must be square
            and invertible.
        b : DomainMatrix
            The ``n x m`` matrix `b` for the rhs.
        cp : list, optional
            The characteristic polynomial of the matrix `A` if known. If not
            given, it will be computed using :meth:`charpoly`.
        check : bool, optional
            If ``True`` (the default) check that the determinant is not zero
            and raise an error if it is. If ``False`` then if the determinant
            is zero the return value will be equal to ``(A.adjugate()*b, 0)`.

        Returns
        =======

        (xnum, detA) : (DomainMatrix, DomainElement)
            The solution of the equation `Ax = b` as a matrix numerator and
            scalar denominator pair. The denominator is equal to the
            determinant of `A` and the numerator is ``adj(A)*b``.

        The solution $x$ is given by ``x = xnum / detA``. The division free
        invariant is ``A * xnum == detA * b``.

        If ``b`` is the identity matrix, then ``xnum`` is the adjugate matrix
        and we have ``A * adj(A) == detA * I``.

        See Also
        ========

        solve_den
            Main frontend for solving matrix equations with denominator.
        solve_den_rref
            Solve matrix equations using fraction-free RREF.
        inv_den
            Invert a matrix using the characteristic polynomial.
        """
        # Unify matrix `b` to match the domain of `self`
        A, b = self.unify(b)
        # Get dimensions of matrices `self` and `b`
        m, n = self.shape
        mb, nb = b.shape

        # Check if `self` is square
        if m != n:
            raise DMNonSquareMatrixError("Matrix must be square")

        # Check if number of rows in `b` matches number of rows in `self`
        if mb != m:
            raise DMShapeError("Matrix and vector must have the same number of rows")

        # Compute the characteristic polynomial and determinant of `self`
        f, detA = self.adj_poly_det(cp=cp)

        # Check if `self` is invertible based on `check` flag
        if check and not detA:
            raise DMNonInvertibleMatrixError("Matrix is not invertible")

        # Compute adj(A)*b = det(A)*inv(A)*b using Horner's method without
        # constructing inv(A) explicitly.
        adjA_b = self.eval_poly_mul(f, b)

        return (adjA_b, detA)
    def adj_poly_det(self, cp=None):
        """
        Return the polynomial $p$ such that $p(A) = adj(A)$ and also the
        determinant of $A.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], QQ)
        >>> p, detA = A.adj_poly_det()
        >>> p
        [-1, 5]
        >>> p_A = A.eval_poly(p)
        >>> p_A
        DomainMatrix([[4, -2], [-3, 1]], (2, 2), QQ)
        >>> p[0]*A**1 + p[1]*A**0 == p_A
        True
        >>> p_A == A.adjugate()
        True
        >>> A * A.adjugate() == detA * A.eye(A.shape, A.domain).to_dense()
        True

        See Also
        ========

        adjugate
        eval_poly
        adj_det
        """

        # Cayley-Hamilton theorem states that a matrix satisfies its own minimal polynomial
        #
        #   p[0]*A^n + p[1]*A^(n-1) + ... + p[n]*I = 0
        #
        # with p[0]=1 and p[n]=(-1)^n*det(A) or
        #
        #   det(A)*I = -(-1)^n*(p[0]*A^(n-1) + p[1]*A^(n-2) + ... + p[n-1]*A).
        #
        # Define a new polynomial f with f[i] = -(-1)^n*p[i] for i=0..n-1. Then
        #
        #   det(A)*I = f[0]*A^n + f[1]*A^(n-1) + ... + f[n-1]*A.
        #
        # Multiplying on the right by inv(A) gives
        #
        #   det(A)*inv(A) = f[0]*A^(n-1) + f[1]*A^(n-2) + ... + f[n-1].
        #
        # So adj(A) = det(A)*inv(A) = f(A)

        A = self  # Assign the current object to variable A for convenience
        m, n = self.shape  # Retrieve the dimensions of the matrix

        if m != n:
            raise DMNonSquareMatrixError("Matrix must be square")  # Check if the matrix is square

        if cp is None:
            cp = A.charpoly()  # Compute the characteristic polynomial if not provided

        if len(cp) % 2:
            # If n (degree of characteristic polynomial) is odd
            detA = cp[-1]  # Determine the determinant of A
            f = [-cpi for cpi in cp[:-1]]  # Define the polynomial f as -(-1)^n*p[i] for i=0..n-1
        else:
            # If n is even
            detA = -cp[-1]  # Determine the determinant of A with opposite sign
            f = cp[:-1]  # Use the remaining coefficients of the characteristic polynomial

        return f, detA  # Return the polynomial f and the determinant detA

    def eval_poly(self, p):
        """
        Evaluate polynomial function of a matrix $p(A)$.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], QQ)
        >>> p = [QQ(1), QQ(2), QQ(3)]
        >>> p_A = A.eval_poly(p)
        >>> p_A
        DomainMatrix([[12, 14], [21, 33]], (2, 2), QQ)
        >>> p_A == p[0]*A**2 + p[1]*A + p[2]*A**0
        True

        See Also
        ========

        eval_poly_mul
        """
        A = self  # Assign the current object to variable A for convenience
        m, n = A.shape  # Retrieve the dimensions of the matrix

        if m != n:
            raise DMNonSquareMatrixError("Matrix must be square")  # Check if the matrix is square

        if not p:
            return self.zeros(self.shape, self.domain)  # Return a zero matrix if polynomial p is empty
        elif len(p) == 1:
            return p[0] * self.eye(self.shape, self.domain)  # Return p[0]*I if polynomial p has only one term

        # Evaluate p(A) using Horner's method:
        # XXX: Use Paterson-Stockmeyer method?
        I = A.eye(A.shape, A.domain)  # Create an identity matrix of the same shape and domain as A
        p_A = p[0] * I  # Initialize p_A with p[0]*I
        for pi in p[1:]:
            p_A = A*p_A + pi*I  # Horner's method to evaluate the polynomial p(A)

        return p_A  # Return the evaluated polynomial p(A)
    # 定义一个方法 eval_poly_mul，用于计算多项式矩阵乘积 p(A) × B
    def eval_poly_mul(self, p, B):
        r"""
        Evaluate polynomial matrix product $p(A) \times B$.

        Evaluate the polynomial matrix product $p(A) \times B$ using Horner's
        method without creating the matrix $p(A)$ explicitly. If $B$ is a
        column matrix then this method will only use matrix-vector multiplies
        and no matrix-matrix multiplies are needed.

        If $B$ is square or wide or if $A$ can be represented in a simpler
        domain than $B$ then it might be faster to evaluate $p(A)$ explicitly
        (see :func:`eval_poly`) and then multiply with $B.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], QQ)
        >>> b = DM([[QQ(5)], [QQ(6)]], QQ)
        >>> p = [QQ(1), QQ(2), QQ(3)]
        >>> p_A_b = A.eval_poly_mul(p, b)
        >>> p_A_b
        DomainMatrix([[144], [303]], (2, 1), QQ)
        >>> p_A_b == p[0]*A**2*b + p[1]*A*b + p[2]*b
        True
        >>> A.eval_poly_mul(p, b) == A.eval_poly(p)*b
        True

        See Also
        ========

        eval_poly
        solve_den_charpoly
        """
        # 将当前对象 A 赋值给变量 A
        A = self
        # 获取矩阵 A 和矩阵 B 的形状
        m, n = A.shape
        mb, nb = B.shape

        # 如果矩阵 A 不是方阵，则抛出异常 DMNonSquareMatrixError
        if m != n:
            raise DMNonSquareMatrixError("Matrix must be square")

        # 如果矩阵 A 和矩阵 B 的行数不匹配，则抛出异常 DMShapeError
        if mb != n:
            raise DMShapeError("Matrices are not aligned")

        # 如果矩阵 A 和矩阵 B 的域不相同，则抛出异常 DMDomainError
        if A.domain != B.domain:
            raise DMDomainError("Matrices must have the same domain")

        # 初始化 p(A) × B 的结果为 p[0]*B
        p_A_B = p[0]*B

        # 使用 Horner 法则计算 p(A) × B
        for p_i in p[1:]:
            p_A_B = A*p_A_B + p_i*B

        # 返回计算结果 p(A) × B
        return p_A_B
    def lu(self):
        r"""
        Returns Lower and Upper decomposition of the DomainMatrix
        
        Returns
        =======
        
        (L, U, exchange)
            L, U are Lower and Upper decomposition of the DomainMatrix,
            exchange is the list of indices of rows exchanged in the
            decomposition.
        
        Raises
        ======
        
        ValueError
            If the domain of DomainMatrix not a Field
        
        Examples
        ========
        
        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [QQ(1), QQ(-1)],
        ...    [QQ(2), QQ(-2)]], (2, 2), QQ)
        >>> L, U, exchange = A.lu()
        >>> L
        DomainMatrix([[1, 0], [2, 1]], (2, 2), QQ)
        >>> U
        DomainMatrix([[1, -1], [0, 0]], (2, 2), QQ)
        >>> exchange
        []
        
        See Also
        ========
        
        lu_solve
        
        """
        # 检查 DomainMatrix 的域是否是一个字段
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        # 调用实际矩阵的 LU 分解方法，获取 L, U 矩阵及行交换信息
        L, U, swaps = self.rep.lu()
        # 将实际矩阵的结果转换为 DomainMatrix 对象并返回
        return self.from_rep(L), self.from_rep(U), swaps

    def lu_solve(self, rhs):
        r"""
        Solver for DomainMatrix x in the A*x = B
        
        Parameters
        ==========
        
        rhs : DomainMatrix B
        
        Returns
        =======
        
        DomainMatrix
            x in A*x = B
        
        Raises
        ======
        
        DMShapeError
            If the DomainMatrix A and rhs have different number of rows
        
        ValueError
            If the domain of DomainMatrix A not a Field
        
        Examples
        ========
        
        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [QQ(1), QQ(2)],
        ...    [QQ(3), QQ(4)]], (2, 2), QQ)
        >>> B = DomainMatrix([
        ...    [QQ(1), QQ(1)],
        ...    [QQ(0), QQ(1)]], (2, 2), QQ)
        
        >>> A.lu_solve(B)
        DomainMatrix([[-2, -1], [3/2, 1]], (2, 2), QQ)
        
        See Also
        ========
        
        lu
        
        """
        # 检查 A 和 rhs 的行数是否相同
        if self.shape[0] != rhs.shape[0]:
            raise DMShapeError("Shape")
        # 检查 DomainMatrix A 的域是否是一个字段
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        # 调用实际矩阵的 LU 解法，求解方程并返回结果
        sol = self.rep.lu_solve(rhs.rep)
        return self.from_rep(sol)

    def _solve(A, b):
        # XXX: Not sure about this method or its signature. It is just created
        # because it is needed by the holonomic module.
        # 检查 A 和 b 的行数是否相同
        if A.shape[0] != b.shape[0]:
            raise DMShapeError("Shape")
        # 检查 A 和 b 的域是否相同且是一个字段
        if A.domain != b.domain or not A.domain.is_Field:
            raise DMNotAField('Not a field')
        # 构造增广矩阵 Aaug，并进行行简化阶梯形式计算
        Aaug = A.hstack(b)
        Arref, pivots = Aaug.rref()
        # 获取特解和零空间的表示
        particular = Arref.from_rep(Arref.rep.particular())
        nullspace_rep, nonpivots = Arref[:,:-1].rep.nullspace()
        nullspace = Arref.from_rep(nullspace_rep)
        # 返回特解和零空间
        return particular, nullspace
    # 定义一个方法，计算给定方阵的特征多项式
    def charpoly(self):
        """
        计算一个方阵的特征多项式。

        使用无除法算术在完全展开的形式下计算特征多项式。如果需要特征多项式的因式分解，
        则调用 :meth:`charpoly_factor_list` 比调用 :meth:`charpoly` 然后对结果进行因式分解更有效率。

        Returns
        =======
        
        list: list of DomainElement
            特征多项式的系数列表

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.charpoly()
        [1, -5, -2]

        See Also
        ========

        charpoly_factor_list
            计算特征多项式的因式分解。
        charpoly_factor_blocks
            特征多项式的部分因式分解，可以比完全因式分解或完全展开多项式更高效地计算。
        """
        # 将当前对象表示为 M
        M = self
        # 获取 M 的定义域 K
        K = M.domain

        # 调用 charpoly_factor_blocks 方法计算特征多项式的因子块
        factors = M.charpoly_factor_blocks()

        # 初始化特征多项式的系数列表，起始为 K 的单位元素
        cp = [K.one]

        # 遍历每个因子及其重数
        for f, mult in factors:
            # 根据重数，使用 dup_mul 方法进行多项式乘法操作
            for _ in range(mult):
                cp = dup_mul(cp, f, K)

        # 返回计算得到的特征多项式系数列表
        return cp
    def charpoly_factor_list(self):
        """
        Full factorization of the characteristic polynomial.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[6, -1, 0, 0],
        ...         [9, 12, 0, 0],
        ...         [0,  0, 1, 2],
        ...         [0,  0, 5, 6]], ZZ)

        Compute the factorization of the characteristic polynomial:

        >>> M.charpoly_factor_list()
        [([1, -9], 2), ([1, -7, -4], 1)]

        Use :meth:`charpoly` to get the unfactorized characteristic polynomial:

        >>> M.charpoly()
        [1, -25, 203, -495, -324]

        The same calculations with ``Matrix``:

        >>> M.to_Matrix().charpoly().as_expr()
        lambda**4 - 25*lambda**3 + 203*lambda**2 - 495*lambda - 324
        >>> M.to_Matrix().charpoly().as_expr().factor()
        (lambda - 9)**2*(lambda**2 - 7*lambda - 4)

        Returns
        =======

        list: list of pairs (factor, multiplicity)
            A full factorization of the characteristic polynomial.

        See Also
        ========

        charpoly
            Expanded form of the characteristic polynomial.
        charpoly_factor_blocks
            A partial factorisation of the characteristic polynomial that can
            be computed more efficiently.
        """
        # 将当前对象赋值给 M
        M = self
        # 获取矩阵的定义域
        K = M.domain

        # 通过 M.charpoly_factor_blocks 提供的部分因式分解更高效
        factors = M.charpoly_factor_blocks()

        # 用于存储不可约因子的列表
        factors_irreducible = []

        # 遍历每个因子及其重数
        for factor_i, mult_i in factors:
            # 使用 dup_factor_list 函数对因子进行完全分解
            _, factors_list = dup_factor_list(factor_i, K)

            # 将每个不可约因子及其合成重数添加到结果列表
            for factor_j, mult_j in factors_list:
                factors_irreducible.append((factor_j, mult_i * mult_j))

        # 整理不可约因子列表并返回
        return _collect_factors(factors_irreducible)
    def charpoly_factor_blocks(self):
        """
        Partial factorisation of the characteristic polynomial.

        This factorisation arises from a block structure of the matrix (if any)
        and so the factors are not guaranteed to be irreducible. The
        :meth:`charpoly_factor_blocks` method is the most efficient way to get
        a representation of the characteristic polynomial but the result is
        neither fully expanded nor fully factored.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[6, -1, 0, 0],
        ...         [9, 12, 0, 0],
        ...         [0,  0, 1, 2],
        ...         [0,  0, 5, 6]], ZZ)

        This computes a partial factorization using only the block structure of
        the matrix to reveal factors:

        >>> M.charpoly_factor_blocks()
        [([1, -18, 81], 1), ([1, -7, -4], 1)]

        These factors correspond to the two diagonal blocks in the matrix:

        >>> DM([[6, -1], [9, 12]], ZZ).charpoly()
        [1, -18, 81]
        >>> DM([[1, 2], [5, 6]], ZZ).charpoly()
        [1, -7, -4]

        Use :meth:`charpoly_factor_list` to get a complete factorization into
        irreducibles:

        >>> M.charpoly_factor_list()
        [([1, -9], 2), ([1, -7, -4], 1)]

        Use :meth:`charpoly` to get the expanded characteristic polynomial:

        >>> M.charpoly()
        [1, -25, 203, -495, -324]

        Returns
        =======

        list: list of pairs (factor, multiplicity)
            A partial factorization of the characteristic polynomial.

        See Also
        ========

        charpoly
            Compute the fully expanded characteristic polynomial.
        charpoly_factor_list
            Compute a full factorization of the characteristic polynomial.
        """
        M = self  # Assign the current matrix object to M for convenience

        if not M.is_square:
            raise DMNonSquareMatrixError("not square")  # Raise an error if M is not square

        # scc returns indices that permute the matrix into block triangular
        # form and can extract the diagonal blocks. M.charpoly() is equal to
        # the product of the diagonal block charpolys.
        components = M.scc()  # Find Strongly Connected Components (SCC) of the matrix

        block_factors = []  # Initialize an empty list to store block factors

        # Iterate over each set of indices representing a component
        for indices in components:
            block = M.extract(indices, indices)  # Extract a block matrix based on indices
            block_factors.append((block.charpoly_base(), 1))  # Append tuple of (block characteristic polynomial, multiplicity 1)

        # Return the collected block factors using a helper function
        return _collect_factors(block_factors)
    def charpoly_base(self):
        """
        Base case for :meth:`charpoly_factor_blocks` after block decomposition.
        
        This method is used internally by :meth:`charpoly_factor_blocks` as the
        base case for computing the characteristic polynomial of a block. It is
        more efficient to call :meth:`charpoly_factor_blocks`, :meth:`charpoly`
        or :meth:`charpoly_factor_list` rather than call this method directly.
        
        This will use either the dense or the sparse implementation depending
        on the sparsity of the matrix and will clear denominators if possible
        before calling :meth:`charpoly_berk` to compute the characteristic
        polynomial using the Berkowitz algorithm.
        
        See Also
        ========
        
        charpoly
        charpoly_factor_list
        charpoly_factor_blocks
        charpoly_berk
        """
        # Assign self to M and its domain to K
        M = self
        K = M.domain
        
        # Calculate density of non-zero entries in the matrix
        density = self.nnz() / self.shape[0]**2
        
        # Decide whether to convert M to sparse or dense matrix based on density
        if density < 0.5:
            M = M.to_sparse()
        else:
            M = M.to_dense()
        
        # Determine if clearing denominators is possible and beneficial
        clear_denoms = K.is_Field and K.has_assoc_Ring
        
        if clear_denoms:
            # Perform denominator clearing and save the divisor
            clear_denoms = True
            d, M = M.clear_denoms(convert=True)
            d = d.element
            K_f = K
            K_r = M.domain
        
        # Compute the characteristic polynomial using the Berkowitz algorithm
        cp = M.charpoly_berk()
        
        if clear_denoms:
            # Restore the denominator in the characteristic polynomial over K_f
            
            # Convert cp to K_f from K_r
            cp = dup_convert(cp, K_r, K_f)
            p = [K_f.one, K_f.zero]
            q = [K_f.one/d]
            
            # Transform cp using p and q
            cp = dup_transform(cp, p, q, K_f)
        
        # Return the computed characteristic polynomial
        return cp
    def charpoly_berk(self):
        """
        Compute the characteristic polynomial using the Berkowitz algorithm.

        This method directly calls the underlying implementation of the
        Berkowitz algorithm (:meth:`sympy.polys.matrices.dense.ddm_berk` or
        :meth:`sympy.polys.matrices.sdm.sdm_berk`).

        This is used by :meth:`charpoly` and other methods as the base case for
        computing the characteristic polynomial. However, those methods will
        apply other optimizations such as block decomposition, clearing
        denominators, and converting between dense and sparse representations
        before calling this method. It is more efficient to call those methods
        instead of this one, but this method is provided for direct access to
        the Berkowitz algorithm.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import QQ
        >>> M = DM([[6, -1, 0, 0],
        ...         [9, 12, 0, 0],
        ...         [0,  0, 1, 2],
        ...         [0,  0, 5, 6]], QQ)
        >>> M.charpoly_berk()
        [1, -25, 203, -495, -324]

        See Also
        ========

        charpoly
        charpoly_base
        charpoly_factor_list
        charpoly_factor_blocks
        sympy.polys.matrices.dense.ddm_berk
        sympy.polys.matrices.sdm.sdm_berk
        """
        return self.rep.charpoly()

    @classmethod
    def eye(cls, shape, domain):
        """
        Return identity matrix of size n or shape (m, n).

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> DomainMatrix.eye(3, QQ)
        DomainMatrix({0: {0: 1}, 1: {1: 1}, 2: {2: 1}}, (3, 3), QQ)

        """
        if isinstance(shape, int):
            shape = (shape, shape)
        return cls.from_rep(SDM.eye(shape, domain))

    @classmethod
    def diag(cls, diagonal, domain, shape=None):
        """
        Return diagonal matrix with entries from ``diagonal``.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import ZZ
        >>> DomainMatrix.diag([ZZ(5), ZZ(6)], ZZ)
        DomainMatrix({0: {0: 5}, 1: {1: 6}}, (2, 2), ZZ)

        """
        if shape is None:
            N = len(diagonal)
            shape = (N, N)
        return cls.from_rep(SDM.diag(diagonal, domain, shape))

    @classmethod
    def zeros(cls, shape, domain, *, fmt='sparse'):
        """
        Returns a zero DomainMatrix of size shape, belonging to the specified domain

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> DomainMatrix.zeros((2, 3), QQ)
        DomainMatrix({}, (2, 3), QQ)

        """
        return cls.from_rep(SDM.zeros(shape, domain))
    # 返回一个指定大小、指定域的全1 DomainMatrix 对象

    def ones(cls, shape, domain):
        """Returns a DomainMatrix of 1s, of size shape, belonging to the specified domain

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> DomainMatrix.ones((2,3), QQ)
        DomainMatrix([[1, 1, 1], [1, 1, 1]], (2, 3), QQ)

        """
        # 使用类方法 cls.from_rep 创建一个包含指定域和形状的 DomainMatrix 对象，其中数据都是 1
        return cls.from_rep(DDM.ones(shape, domain).to_dfm_or_ddm())

    # 检查两个 DomainMatrix 对象是否相等
    def __eq__(A, B):
        r"""
        Checks for two DomainMatrix matrices to be equal or not

        Parameters
        ==========

        A, B: DomainMatrix
            to check equality

        Returns
        =======

        Boolean
            True for equal, else False

        Raises
        ======

        NotImplementedError
            If B is not a DomainMatrix

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([
        ...    [ZZ(1), ZZ(1)],
        ...    [ZZ(0), ZZ(1)]], (2, 2), ZZ)
        >>> A.__eq__(A)
        True
        >>> A.__eq__(B)
        False

        """
        # 如果 B 不是 DomainMatrix 类型，则返回 NotImplemented
        if not isinstance(A, type(B)):
            return NotImplemented
        # 返回比较两个对象的域和表示是否相等的布尔值
        return A.domain == B.domain and A.rep == B.rep

    # 统一比较两个 DomainMatrix 对象的相等性
    def unify_eq(A, B):
        # 如果形状不同，直接返回 False
        if A.shape != B.shape:
            return False
        # 如果域不同，尝试统一它们的域
        if A.domain != B.domain:
            A, B = A.unify(B)
        # 最终比较 A 和 B 是否相等
        return A == B
    # 定义一个函数 lll，实现Lenstra–Lenstra–Lovász（LLL）基底约简算法
    def lll(A, delta=QQ(3, 4)):
        """
        Performs the Lenstra–Lenstra–Lovász (LLL) basis reduction algorithm.
        See [1]_ and [2]_.

        Parameters
        ==========

        delta : QQ, optional
            The Lovász parameter. Must be in the interval (0.25, 1), with larger
            values producing a more reduced basis. The default is 0.75 for
            historical reasons.

        Returns
        =======

        The reduced basis as a DomainMatrix over ZZ.

        Throws
        ======

        DMValueError: if delta is not in the range (0.25, 1)
        DMShapeError: if the matrix is not of shape (m, n) with m <= n
        DMDomainError: if the matrix domain is not ZZ
        DMRankError: if the matrix contains linearly dependent rows

        Examples
        ========

        >>> from sympy.polys.domains import ZZ, QQ
        >>> from sympy.polys.matrices import DM
        >>> x = DM([[1, 0, 0, 0, -20160],
        ...         [0, 1, 0, 0, 33768],
        ...         [0, 0, 1, 0, 39578],
        ...         [0, 0, 0, 1, 47757]], ZZ)
        >>> y = DM([[10, -3, -2, 8, -4],
        ...         [3, -9, 8, 1, -11],
        ...         [-3, 13, -9, -3, -9],
        ...         [-12, -7, -11, 9, -1]], ZZ)
        >>> assert x.lll(delta=QQ(5, 6)) == y

        Notes
        =====

        The implementation is derived from the Maple code given in Figures 4.3
        and 4.4 of [3]_ (pp.68-69). It uses the efficient method of only calculating
        state updates as they are required.

        See also
        ========

        lll_transform

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Lenstra%E2%80%93Lenstra%E2%80%93Lov%C3%A1sz_lattice_basis_reduction_algorithm
        .. [2] https://web.archive.org/web/20221029115428/https://web.cs.elte.hu/~lovasz/scans/lll.pdf
        .. [3] Murray R. Bremner, "Lattice Basis Reduction: An Introduction to the LLL Algorithm and Its Applications"

        """
        # 调用 A 对象的表示方法，应用 LLL 算法进行基底约简，并返回结果作为 DomainMatrix 对象
        return DomainMatrix.from_rep(A.rep.lll(delta=delta))
    # 定义一个函数 lll_transform，执行Lenstra–Lenstra–Lovász (LLL)基底约简算法，
    # 返回约简后的基底和变换矩阵。

    """
    Performs the Lenstra–Lenstra–Lovász (LLL) basis reduction algorithm
    and returns the reduced basis and transformation matrix.

    Explanation
    ===========

    Parameters, algorithm and basis are the same as for :meth:`lll` except that
    the return value is a tuple `(B, T)` with `B` the reduced basis and
    `T` a transformation matrix. The original basis `A` is transformed to
    `B` with `T*A == B`. If only `B` is needed then :meth:`lll` should be
    used as it is a little faster.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ, QQ
    >>> from sympy.polys.matrices import DM
    >>> X = DM([[1, 0, 0, 0, -20160],
    ...         [0, 1, 0, 0, 33768],
    ...         [0, 0, 1, 0, 39578],
    ...         [0, 0, 0, 1, 47757]], ZZ)
    >>> B, T = X.lll_transform(delta=QQ(5, 6))
    >>> T * X == B
    True

    See also
    ========

    lll
    """

    reduced, transform = A.rep.lll_transform(delta=delta)
    # 使用 A 的表示进行 LLL 算法的基底约简，返回约简后的基底和变换矩阵

    return DomainMatrix.from_rep(reduced), DomainMatrix.from_rep(transform)
    # 将约简后的基底和变换矩阵分别封装成 DomainMatrix 对象并返回
# 导入所需模块中的函数以及类
from sympy.polys.matrices.domainmatrix import _collect_factors

# 定义一个函数用于收集重复的因子并排序结果
def _collect_factors(factors_list):
    # 创建一个计数器对象，用于统计因子出现的次数
    factors = Counter()

    # 遍历输入的因子列表
    for factor, exponent in factors_list:
        # 将因子转换为元组，作为字典的键，并累加指数
        factors[tuple(factor)] += exponent

    # 将因子字典中的键值对转换为列表形式，其中因子以列表形式出现
    factors_list = [(list(f), e) for f, e in factors.items()]

    # 调用排序因子函数对因子列表进行排序
    return _sort_factors(factors_list)
```
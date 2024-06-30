# `D:\src\scipysrc\sympy\sympy\polys\matrices\sdm.py`

```
"""

Module for the SDM class.

"""

# 从操作符模块中导入加法、取反、正、减法、乘法等运算符函数
from operator import add, neg, pos, sub, mul
# 导入默认字典数据结构
from collections import defaultdict

# 从sympy.external.gmpy中导入GROUND_TYPES常量
from sympy.external.gmpy import GROUND_TYPES
# 从sympy.utilities.decorator中导入doctest_depends_on装饰器
from sympy.utilities.decorator import doctest_depends_on
# 从sympy.utilities.iterables中导入_strongly_connected_components函数
from sympy.utilities.iterables import _strongly_connected_components

# 从当前包中导入异常类：DMBadInputError、DMDomainError、DMShapeError
from .exceptions import DMBadInputError, DMDomainError, DMShapeError

# 从sympy.polys.domains中导入QQ域
from sympy.polys.domains import QQ

# 从当前包中导入DDM类
from .ddm import DDM

# 如果GROUND_TYPES不等于'flint'，则设置__doctest_skip__列表，跳过指定的测试用例
if GROUND_TYPES != 'flint':
    __doctest_skip__ = ['SDM.to_dfm', 'SDM.to_dfm_or_ddm']


class SDM(dict):
    r"""Sparse matrix based on polys domain elements

    This is a dict subclass and is a wrapper for a dict of dicts that supports
    basic matrix arithmetic +, -, *, **.


    In order to create a new :py:class:`~.SDM`, a dict
    of dicts mapping non-zero elements to their
    corresponding row and column in the matrix is needed.

    We also need to specify the shape and :py:class:`~.Domain`
    of our :py:class:`~.SDM` object.

    We declare a 2x2 :py:class:`~.SDM` matrix belonging
    to QQ domain as shown below.
    The 2x2 Matrix in the example is

    .. math::
           A = \left[\begin{array}{ccc}
                0 & \frac{1}{2} \\
                0 & 0 \end{array} \right]


    >>> from sympy.polys.matrices.sdm import SDM
    >>> from sympy import QQ
    >>> elemsdict = {0:{1:QQ(1, 2)}}
    >>> A = SDM(elemsdict, (2, 2), QQ)
    >>> A
    {0: {1: 1/2}}

    We can manipulate :py:class:`~.SDM` the same way
    as a Matrix class

    >>> from sympy import ZZ
    >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
    >>> B  = SDM({0:{0: ZZ(3)}, 1:{1:ZZ(4)}}, (2, 2), ZZ)
    >>> A + B
    {0: {0: 3, 1: 2}, 1: {0: 1, 1: 4}}

    Multiplication

    >>> A*B
    {0: {1: 8}, 1: {0: 3}}
    >>> A*ZZ(2)
    {0: {1: 4}, 1: {0: 2}}

    """

    # 类属性：格式为稀疏矩阵
    fmt = 'sparse'
    # 是否为DFM（分布式稀疏矩阵）：False
    is_DFM = False
    # 是否为DDM（分布式稠密矩阵）：False
    is_DDM = False

    def __init__(self, elemsdict, shape, domain):
        # 调用父类字典的初始化方法，并传入elemsdict作为初始内容
        super().__init__(elemsdict)
        # 设置矩阵的形状（行数、列数）
        self.shape = self.rows, self.cols = m, n = shape
        # 设置矩阵的域（QQ）
        self.domain = domain

        # 检查所有行索引是否在合法范围内
        if not all(0 <= r < m for r in self):
            raise DMBadInputError("Row out of range")
        # 检查所有列索引是否在合法范围内
        if not all(0 <= c < n for row in self.values() for c in row):
            raise DMBadInputError("Column out of range")

    def getitem(self, i, j):
        # 尝试获取矩阵中(i, j)位置的元素
        try:
            return self[i][j]
        except KeyError:
            m, n = self.shape
            # 如果行索引i或列索引j超出范围，进行模运算后再次尝试获取
            if -m <= i < m and -n <= j < n:
                try:
                    return self[i % m][j % n]
                except KeyError:
                    # 如果仍然找不到，则返回域的零元素
                    return self.domain.zero
            else:
                # 抛出索引错误异常
                raise IndexError("index out of range")
    def setitem(self, i, j, value):
        # 获取矩阵的行数 m 和列数 n
        m, n = self.shape
        # 检查索引 i 和 j 是否在有效范围内
        if not (-m <= i < m and -n <= j < n):
            raise IndexError("index out of range")
        # 将 i 和 j 规范化到非负数范围内
        i, j = i % m, j % n
        # 如果 value 存在（非空），尝试在 self[i] 中设置键 j 的值
        if value:
            try:
                self[i][j] = value
            # 如果 self[i] 不存在（KeyError），创建一个新的字典 {j: value} 并将其赋给 self[i]
            except KeyError:
                self[i] = {j: value}
        else:
            # 获取索引 i 处的行数据 rowi
            rowi = self.get(i, None)
            # 如果 rowi 不为 None，则尝试删除键 j
            if rowi is not None:
                try:
                    del rowi[j]
                # 如果键 j 不存在（KeyError），则忽略
                except KeyError:
                    pass
                else:
                    # 如果删除后 rowi 变为空（不包含任何键值对），则删除索引 i 处的行数据
                    if not rowi:
                        del self[i]

    def extract_slice(self, slice1, slice2):
        # 获取矩阵的行数 m 和列数 n
        m, n = self.shape
        # 根据 slice1 和 slice2 提取行索引 ri 和列索引 ci
        ri = range(m)[slice1]
        ci = range(n)[slice2]

        # 初始化空字典 sdm
        sdm = {}
        # 遍历矩阵中的每一行 i 和其对应的行数据 row
        for i, row in self.items():
            # 如果行索引 i 存在于 ri 中
            if i in ri:
                # 创建新的行数据，仅包含列索引 ci 中存在的列和对应的元素
                row = {ci.index(j): e for j, e in row.items() if j in ci}
                # 如果新的行数据不为空，则将其加入 sdm 中，键为 ri 中对应的索引位置
                if row:
                    sdm[ri.index(i)] = row

        # 返回使用 sdm 创建的新对象，其形状为 (len(ri), len(ci))，域为 self.domain
        return self.new(sdm, (len(ri), len(ci)), self.domain)

    def extract(self, rows, cols):
        # 如果 self 或 rows 或 cols 为空，则返回一个与所需形状和域相符的零矩阵
        if not (self and rows and cols):
            return self.zeros((len(rows), len(cols)), self.domain)

        # 获取矩阵的行数 m 和列数 n
        m, n = self.shape
        # 检查 rows 中的行索引和 cols 中的列索引是否在有效范围内
        if not (-m <= min(rows) <= max(rows) < m):
            raise IndexError('Row index out of range')
        if not (-n <= min(cols) <= max(cols) < n):
            raise IndexError('Column index out of range')

        # 创建默认字典用于映射 self 中的行和列到输出矩阵中的行和列列表
        rowmap = defaultdict(list)
        colmap = defaultdict(list)
        # 构建映射：从 self 中的行和列到输出矩阵中的行和列列表
        for i2, i1 in enumerate(rows):
            rowmap[i1 % m].append(i2)
        for j2, j1 in enumerate(cols):
            colmap[j1 % n].append(j2)

        # 创建集合用于跳过零行和列的高效处理
        rowset = set(rowmap)
        colset = set(colmap)

        # 初始化 sdm1 为 self
        sdm1 = self
        # 初始化空字典 sdm2
        sdm2 = {}
        # 遍历 rowset 与 sdm1 中的键的交集
        for i1 in rowset & sdm1.keys():
            # 获取 sdm1 中索引 i1 处的行数据 row1
            row1 = sdm1[i1]
            # 初始化新的行数据 row2
            row2 = {}
            # 遍历 colset 与 row1 中的键的交集
            for j1 in colset & row1.keys():
                # 将 row1[j1] 的值赋给 row1_j1
                row1_j1 = row1[j1]
                # 将 row1_j1 复制到 colmap[j1] 中的每个位置 j2
                for j2 in colmap[j1]:
                    row2[j2] = row1_j1
            # 如果 row2 不为空，则将其添加到 sdm2 中，键为 rowmap[i1] 中对应的索引位置
            if row2:
                for i2 in rowmap[i1]:
                    sdm2[i2] = row2.copy()

        # 返回使用 sdm2 创建的新对象，其形状为 (len(rows), len(cols))，域为 self.domain
        return self.new(sdm2, (len(rows), len(cols)), self.domain)

    def __str__(self):
        # 初始化空列表 rowsstr
        rowsstr = []
        # 遍历矩阵中的每一行 i 和其对应的行数据 row
        for i, row in self.items():
            # 将行数据中的每个元素格式化为字符串并加入 elemsstr
            elemsstr = ', '.join('%s: %s' % (j, elem) for j, elem in row.items())
            # 将格式化后的行数据字符串加入 rowsstr
            rowsstr.append('%s: {%s}' % (i, elemsstr))
        # 返回格式化后的字符串，表示整个矩阵
        return '{%s}' % ', '.join(rowsstr)

    def __repr__(self):
        # 获取类名
        cls = type(self).__name__
        # 获取行数据的字符串表示
        rows = dict.__repr__(self)
        # 返回对象的详细字符串表示，包括行数据、形状和域
        return '%s(%s, %s, %s)' % (cls, rows, self.shape, self.domain)

    @classmethod
    def new(cls, sdm, shape, domain):
        """
        Create a new instance of :py:class:`~.SDM`.

        Parameters
        ==========

        sdm: A dict of dicts for non-zero elements in SDM
            Dictionary representing the sparse matrix data.
        shape: tuple representing dimension of SDM
            Dimensions of the sparse matrix.
        domain: Represents :py:class:`~.Domain` of SDM
            Domain of the elements in the sparse matrix.

        Returns
        =======

        An :py:class:`~.SDM` object
            Instance of the Sparse Diagonal Matrix.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> elemsdict = {0:{1: QQ(2)}}
        >>> A = SDM.new(elemsdict, (2, 2), QQ)
        >>> A
        {0: {1: 2}}

        """
        return cls(sdm, shape, domain)

    def copy(A):
        """
        Returns a copy of a :py:class:`~.SDM` object.

        Parameters
        ==========

        A: :py:class:`~.SDM`
            Instance of the Sparse Diagonal Matrix to be copied.

        Returns
        =======

        :py:class:`~.SDM`
            Copy of the input Sparse Diagonal Matrix object.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> elemsdict = {0:{1:QQ(2)}, 1:{}}
        >>> A = SDM(elemsdict, (2, 2), QQ)
        >>> B = A.copy()
        >>> B
        {0: {1: 2}, 1: {}}

        """
        Ac = {i: Ai.copy() for i, Ai in A.items()}  # Create a deep copy of each dictionary in A
        return A.new(Ac, A.shape, A.domain)

    @classmethod
    def from_list(cls, ddm, shape, domain):
        """
        Create :py:class:`~.SDM` object from a list of lists.

        Parameters
        ==========

        ddm: list of lists containing domain elements
            List of lists representing the sparse matrix data.
        shape: tuple
            Dimensions of :py:class:`~.SDM` matrix
        domain: object
            Represents :py:class:`~.Domain` of :py:class:`~.SDM` object

        Returns
        =======

        :py:class:`~.SDM`
            Sparse Diagonal Matrix containing elements of ddm

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> ddm = [[QQ(1, 2), QQ(0)], [QQ(0), QQ(3, 4)]]
        >>> A = SDM.from_list(ddm, (2, 2), QQ)
        >>> A
        {0: {0: 1/2}, 1: {1: 3/4}}

        See Also
        ========

        to_list
        from_list_flat
        from_dok
        from_ddm
        """

        m, n = shape
        if not (len(ddm) == m and all(len(row) == n for row in ddm)):
            raise DMBadInputError("Inconsistent row-list/shape")
        getrow = lambda i: {j:ddm[i][j] for j in range(n) if ddm[i][j]}  # Lambda function to extract non-zero elements from a row
        irows = ((i, getrow(i)) for i in range(m))  # Generator for rows with non-zero elements
        sdm = {i: row for i, row in irows if row}  # Dictionary comprehension to create the sparse matrix representation
        return cls(sdm, shape, domain)

    @classmethod
    @classmethod
    def from_ddm(cls, ddm):
        """
        从一个 :py:class:`~.DDM` 创建一个 :py:class:`~.SDM` 对象。

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> ddm = DDM( [[QQ(1, 2), 0], [0, QQ(3, 4)]], (2, 2), QQ)
        >>> A = SDM.from_ddm(ddm)
        >>> A
        {0: {0: 1/2}, 1: {1: 3/4}}
        >>> SDM.from_ddm(ddm).to_ddm() == ddm
        True

        See Also
        ========

        to_ddm
        from_list
        from_list_flat
        from_dok
        """
        # 调用类方法 from_list，从给定的 ddm 对象、形状和定义域创建 SDM 对象
        return cls.from_list(ddm, ddm.shape, ddm.domain)

    def to_list(M):
        """
        将 :py:class:`~.SDM` 对象转换为列表的列表。

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> elemsdict = {0:{1:QQ(2)}, 1:{}}
        >>> A = SDM(elemsdict, (2, 2), QQ)
        >>> A.to_list()
        [[0, 2], [0, 0]]

        """
        # 获取 SDM 对象的行数和列数
        m, n = M.shape
        # 获取零元素
        zero = M.domain.zero
        # 创建一个二维列表 ddm，用于存储 SDM 对象的元素
        ddm = [[zero] * n for _ in range(m)]
        # 将 SDM 对象的元素填充到 ddm 中对应的位置
        for i, row in M.items():
            for j, e in row.items():
                ddm[i][j] = e
        return ddm

    def to_list_flat(M):
        """
        将 :py:class:`~.SDM` 对象转换为扁平列表。

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{0: QQ(3)}}, (2, 2), QQ)
        >>> A.to_list_flat()
        [0, 2, 3, 0]
        >>> A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        from_list_flat
        to_list
        to_dok
        to_ddm
        """
        # 获取 SDM 对象的行数和列数
        m, n = M.shape
        # 获取零元素
        zero = M.domain.zero
        # 创建一个扁平列表 flat，用于存储 SDM 对象的元素
        flat = [zero] * (m * n)
        # 将 SDM 对象的元素填充到 flat 中对应的位置
        for i, row in M.items():
            for j, e in row.items():
                flat[i*n + j] = e
        return flat

    @classmethod
    def from_list_flat(cls, elements, shape, domain):
        """
        从扁平元素列表创建 :py:class:`~.SDM` 对象。

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM.from_list_flat([QQ(0), QQ(2), QQ(0), QQ(0)], (2, 2), QQ)
        >>> A
        {0: {1: 2}}
        >>> A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        to_list_flat
        from_list
        from_dok
        from_ddm
        """
        # 获取形状的行数和列数
        m, n = shape
        # 检查扁平元素列表的长度是否与形状相符
        if len(elements) != m * n:
            raise DMBadInputError("Inconsistent flat-list shape")
        # 使用 defaultdict 创建一个空的 SDM 对象 sdm
        sdm = defaultdict(dict)
        # 将扁平元素列表中的元素填充到 SDM 对象 sdm 中
        for inj, element in enumerate(elements):
            if element:
                i, j = divmod(inj, n)
                sdm[i][j] = element
        return cls(sdm, shape, domain)
    def to_flat_nz(M):
        """
        Convert :class:`SDM` to a flat list of nonzero elements and data.

        Explanation
        ===========

        This is used to operate on a list of the elements of a matrix and then
        reconstruct a modified matrix with elements in the same positions using
        :meth:`from_flat_nz`. Zero elements are omitted from the list.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{0: QQ(3)}}, (2, 2), QQ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [2, 3]
        >>> A == A.from_flat_nz(elements, data, A.domain)
        True

        See Also
        ========

        from_flat_nz
        to_list_flat
        sympy.polys.matrices.ddm.DDM.to_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_flat_nz
        """
        # Convert the matrix `M` into a dictionary of keys format (`dok`)
        dok = M.to_dok()
        # Extract indices from the `dok` format
        indices = tuple(dok)
        # Extract non-zero elements from the `dok` format
        elements = list(dok.values())
        # Store the indices and the shape of matrix `M` as `data`
        data = (indices, M.shape)
        # Return the non-zero elements and the `data` tuple
        return elements, data

    @classmethod
    def from_flat_nz(cls, elements, data, domain):
        """
        Reconstruct a :class:`~.SDM` after calling :meth:`to_flat_nz`.

        See :meth:`to_flat_nz` for explanation.

        See Also
        ========

        to_flat_nz
        from_list_flat
        sympy.polys.matrices.ddm.DDM.from_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_flat_nz
        """
        # Unpack `data` tuple into `indices` and `shape`
        indices, shape = data
        # Create a dictionary of keys (`dok`) from `indices` and `elements`
        dok = dict(zip(indices, elements))
        # Use class method `from_dok` to reconstruct the `SDM` object
        return cls.from_dok(dok, shape, domain)

    def to_dod(M):
        """
        Convert to dictionary of dictionaries (dod) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> A.to_dod()
        {0: {1: 2}, 1: {0: 3}}

        See Also
        ========

        from_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dod
        """
        # Convert each row in `M` to a dictionary and return the result
        return {i: row.copy() for i, row in M.items()}

    @classmethod
    def from_dod(cls, dod, shape, domain):
        """
        Create :py:class:`~.SDM` from dictionary of dictionaries (dod) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> dod = {0: {1: QQ(2)}, 1: {0: QQ(3)}}
        >>> A = SDM.from_dod(dod, (2, 2), QQ)
        >>> A
        {0: {1: 2}, 1: {0: 3}}
        >>> A == SDM.from_dod(A.to_dod(), A.shape, A.domain)
        True

        See Also
        ========

        to_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dod
        """
        # Initialize a defaultdict to store the sparse matrix representation
        sdm = defaultdict(dict)
        # Iterate over rows (`i`) and elements (`row`) in `dod`
        for i, row in dod.items():
            # Iterate over columns (`j`) and elements (`e`) in each `row`
            for j, e in row.items():
                # If `e` (element) is non-zero, store it in `sdm`
                if e:
                    sdm[i][j] = e
        # Return an instance of `SDM` constructed from `sdm`, `shape`, and `domain`
        return cls(sdm, shape, domain)
    def to_dok(M):
        """
        Convert to dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> A.to_dok()
        {(0, 1): 2, (1, 0): 3}

        See Also
        ========

        from_dok
        to_list
        to_list_flat
        to_ddm
        """
        # 返回一个字典，将稀疏矩阵 M 转换为字典格式 (dok)
        return {(i, j): e for i, row in M.items() for j, e in row.items()}

    @classmethod
    def from_dok(cls, dok, shape, domain):
        """
        Create :py:class:`~.SDM` from dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> dok = {(0, 1): QQ(2), (1, 0): QQ(3)}
        >>> A = SDM.from_dok(dok, (2, 2), QQ)
        >>> A
        {0: {1: 2}, 1: {0: 3}}
        >>> A == SDM.from_dok(A.to_dok(), A.shape, A.domain)
        True

        See Also
        ========

        to_dok
        from_list
        from_list_flat
        from_ddm
        """
        # 从字典格式 (dok) 创建一个 :py:class:`~.SDM` 对象
        sdm = defaultdict(dict)
        for (i, j), e in dok.items():
            if e:
                sdm[i][j] = e
        return cls(sdm, shape, domain)

    def iter_values(M):
        """
        Iterate over the nonzero values of a :py:class:`~.SDM` matrix.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> list(A.iter_values())
        [2, 3]

        """
        # 迭代一个 :py:class:`~.SDM` 矩阵中所有非零元素的值
        for row in M.values():
            yield from row.values()

    def iter_items(M):
        """
        Iterate over indices and values of the nonzero elements.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> list(A.iter_items())
        [((0, 1), 2), ((1, 0), 3)]

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.iter_items
        """
        # 迭代一个 :py:class:`~.SDM` 矩阵中所有非零元素的索引和值
        for i, row in M.items():
            for j, e in row.items():
                yield (i, j), e

    def to_ddm(M):
        """
        Convert a :py:class:`~.SDM` object to a :py:class:`~.DDM` object

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)
        >>> A.to_ddm()
        [[0, 2], [0, 0]]

        """
        # 将 :py:class:`~.SDM` 对象转换为 :py:class:`~.DDM` 对象
        return DDM(M.to_list(), M.shape, M.domain)

    def to_sdm(M):
        """
        Convert to :py:class:`~.SDM` format (returns self).
        """
        # 返回 :py:class:`~.SDM` 格式的对象本身
        return M

    @doctest_depends_on(ground_types=['flint'])
    @classmethod
    def zeros(cls, shape, domain):
        r"""
        返回一个大小为 shape 的 :py:class:`~.SDM`，属于指定的 domain

        在下面的示例中，我们声明一个矩阵 A，其中

        .. math::
            A := \left[\begin{array}{ccc}
            0 & 0 & 0 \\
            0 & 0 & 0 \end{array} \right]

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM.zeros((2, 3), QQ)
        >>> A
        {}

        """
        return cls({}, shape, domain)

    @classmethod
    def ones(cls, shape, domain):
        one = domain.one
        m, n = shape
        row = dict(zip(range(n), [one]*n))
        sdm = {i: row.copy() for i in range(m)}
        return cls(sdm, shape, domain)

    @classmethod
    def eye(cls, shape, domain):
        """

        返回一个大小为 size x size 的单位 :py:class:`~.SDM` 矩阵，属于指定的 domain

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> I = SDM.eye((2, 2), QQ)
        >>> I
        {0: {0: 1}, 1: {1: 1}}

        """
        if isinstance(shape, int):
            rows, cols = shape, shape
        else:
            rows, cols = shape
        one = domain.one
        sdm = {i: {i: one} for i in range(min(rows, cols))}
        return cls(sdm, (rows, cols), domain)

    @classmethod
    def diag(cls, diagonal, domain, shape=None):
        if shape is None:
            shape = (len(diagonal), len(diagonal))
        sdm = {i: {i: v} for i, v in enumerate(diagonal) if v}
        return cls(sdm, shape, domain)
    def transpose(M):
        """
        返回一个 :py:class:`~.SDM` 矩阵的转置

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)
        >>> A.transpose()
        {1: {0: 2}}

        """
        # 调用内部函数 sdm_transpose 对矩阵 M 进行转置操作
        MT = sdm_transpose(M)
        # 返回一个新的 SDM 对象，参数为转置后的数据 MT、反转的形状 M.shape 和 M 的域 M.domain
        return M.new(MT, M.shape[::-1], M.domain)

    def __add__(A, B):
        # 如果 B 不是 SDM 类型，则返回 NotImplemented
        if not isinstance(B, SDM):
            return NotImplemented
        # 如果 A 和 B 的形状不匹配，抛出 DMShapeError 异常
        elif A.shape != B.shape:
            raise DMShapeError("Matrix size mismatch: %s + %s" % (A.shape, B.shape))
        # 调用 A 的 add 方法，返回结果
        return A.add(B)

    def __sub__(A, B):
        # 如果 B 不是 SDM 类型，则返回 NotImplemented
        if not isinstance(B, SDM):
            return NotImplemented
        # 如果 A 和 B 的形状不匹配，抛出 DMShapeError 异常
        elif A.shape != B.shape:
            raise DMShapeError("Matrix size mismatch: %s - %s" % (A.shape, B.shape))
        # 调用 A 的 sub 方法，返回结果
        return A.sub(B)

    def __neg__(A):
        # 调用 A 的 neg 方法，返回结果
        return A.neg()

    def __mul__(A, B):
        """A * B"""
        # 如果 B 是 SDM 类型，则调用 A 的 matmul 方法，返回结果
        if isinstance(B, SDM):
            return A.matmul(B)
        # 如果 B 在 A 的域中，则调用 A 的 mul 方法，返回结果
        elif B in A.domain:
            return A.mul(B)
        # 否则返回 NotImplemented
        else:
            return NotImplemented

    def __rmul__(a, b):
        # 如果 b 在 a 的域中，则调用 a 的 rmul 方法，返回结果
        if b in a.domain:
            return a.rmul(b)
        # 否则返回 NotImplemented
        else:
            return NotImplemented

    def matmul(A, B):
        """
        对两个 SDM 矩阵进行矩阵乘法

        Parameters
        ==========

        A, B: SDM 矩阵

        Returns
        =======

        SDM
            矩阵乘法后的 SDM

        Raises
        ======

        DMDomainError
            如果 A 的域与 B 的域不匹配
        DMShapeError
            如果 A 和 B 的形状不匹配

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> B = SDM({0:{0:ZZ(2), 1:ZZ(3)}, 1:{0:ZZ(4)}}, (2, 2), ZZ)
        >>> A.matmul(B)
        {0: {0: 8}, 1: {0: 2, 1: 3}}

        """
        # 如果 A 的域与 B 的域不匹配，则抛出 DMDomainError 异常
        if A.domain != B.domain:
            raise DMDomainError
        # 获取矩阵 A 和 B 的形状
        m, n = A.shape
        n2, o = B.shape
        # 如果 A 的列数不等于 B 的行数，则抛出 DMShapeError 异常
        if n != n2:
            raise DMShapeError
        # 调用 sdm_matmul 函数进行矩阵乘法，返回结果 C
        C = sdm_matmul(A, B, A.domain, m, o)
        # 返回一个新的 SDM 对象，参数为乘法结果 C、形状 (m, o) 和 A 的域 A.domain
        return A.new(C, (m, o), A.domain)

    def mul(A, b):
        """
        将矩阵 A 中的每个元素与标量 b 相乘

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.mul(ZZ(3))
        {0: {1: 6}, 1: {0: 3}}

        """
        # 调用 unop_dict 函数，对 A 中的每个元素应用 lambda 函数 aij*b，返回结果 Csdm
        Csdm = unop_dict(A, lambda aij: aij*b)
        # 返回一个新的 SDM 对象，参数为乘法结果 Csdm、原始形状 A.shape 和 A 的域 A.domain
        return A.new(Csdm, A.shape, A.domain)

    def rmul(A, b):
        # 调用 unop_dict 函数，对 A 中的每个元素应用 lambda 函数 b*aij，返回结果 Csdm
        Csdm = unop_dict(A, lambda aij: b*aij)
        # 返回一个新的 SDM 对象，参数为乘法结果 Csdm、原始形状 A.shape 和 A 的域 A.domain
        return A.new(Csdm, A.shape, A.domain)
    def mul_elementwise(A, B):
        # 检查两个矩阵的定义域是否相同，若不同则抛出异常
        if A.domain != B.domain:
            raise DMDomainError
        # 检查两个矩阵的形状是否相同，若不同则抛出异常
        if A.shape != B.shape:
            raise DMShapeError
        # 获取零元素，该元素将用于构造新矩阵中的零元素
        zero = A.domain.zero
        # 定义一个函数，用于生成与定义域相同的零元素
        fzero = lambda e: zero
        # 对矩阵 A 和 B 执行元素级乘法操作，并得到结果矩阵的稀疏字典表示
        Csdm = binop_dict(A, B, mul, fzero, fzero)
        # 返回一个新的 SDM 对象，包含乘法结果的稀疏字典表示、原形状及定义域
        return A.new(Csdm, A.shape, A.domain)

    def add(A, B):
        """
        Adds two :py:class:`~.SDM` matrices

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> B = SDM({0:{0: ZZ(3)}, 1:{1:ZZ(4)}}, (2, 2), ZZ)
        >>> A.add(B)
        {0: {0: 3, 1: 2}, 1: {0: 1, 1: 4}}

        """
        # 对矩阵 A 和 B 执行加法操作，并得到结果矩阵的稀疏字典表示
        Csdm = binop_dict(A, B, add, pos, pos)
        # 返回一个新的 SDM 对象，包含加法结果的稀疏字典表示、原形状及定义域
        return A.new(Csdm, A.shape, A.domain)

    def sub(A, B):
        """
        Subtracts two :py:class:`~.SDM` matrices

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> B  = SDM({0:{0: ZZ(3)}, 1:{1:ZZ(4)}}, (2, 2), ZZ)
        >>> A.sub(B)
        {0: {0: -3, 1: 2}, 1: {0: 1, 1: -4}}

        """
        # 对矩阵 A 和 B 执行减法操作，并得到结果矩阵的稀疏字典表示
        Csdm = binop_dict(A, B, sub, pos, neg)
        # 返回一个新的 SDM 对象，包含减法结果的稀疏字典表示、原形状及定义域
        return A.new(Csdm, A.shape, A.domain)

    def neg(A):
        """
        Returns the negative of a :py:class:`~.SDM` matrix

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.neg()
        {0: {1: -2}, 1: {0: -1}}

        """
        # 对矩阵 A 执行取负操作，并得到结果矩阵的稀疏字典表示
        Csdm = unop_dict(A, neg)
        # 返回一个新的 SDM 对象，包含取负结果的稀疏字典表示、原形状及定义域
        return A.new(Csdm, A.shape, A.domain)

    def convert_to(A, K):
        """
        Converts the :py:class:`~.Domain` of a :py:class:`~.SDM` matrix to K

        Examples
        ========

        >>> from sympy import ZZ, QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.convert_to(QQ)
        {0: {1: 2}, 1: {0: 1}}

        """
        # 获取矩阵 A 的当前定义域
        Kold = A.domain
        # 如果目标定义域 K 与当前定义域相同，则返回当前矩阵 A 的副本
        if K == Kold:
            return A.copy()
        # 对矩阵 A 中的每个元素执行从当前定义域到目标定义域 K 的转换，并得到结果矩阵的稀疏字典表示
        Ak = unop_dict(A, lambda e: K.convert_from(e, Kold))
        # 返回一个新的 SDM 对象，包含转换后的稀疏字典表示、原形状及目标定义域 K
        return A.new(Ak, A.shape, K)

    def nnz(A):
        """Number of non-zero elements in the :py:class:`~.SDM` matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.nnz()
        2

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nnz
        """
        # 计算矩阵 A 中非零元素的个数并返回
        return sum(map(len, A.values()))
    def scc(A):
        """计算方阵 A 的强连通分量。

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0: ZZ(2)}, 1:{1:ZZ(1)}}, (2, 2), ZZ)
        >>> A.scc()
        [[0], [1]]

        See also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.scc
        """
        # 获取矩阵 A 的行数和列数
        rows, cols = A.shape
        # 断言行数和列数相等，即 A 是方阵
        assert rows == cols
        # 构建节点集合 V
        V = range(rows)
        # 构建邻接表 Emap，表示每个节点的出边列表
        Emap = {v: list(A.get(v, [])) for v in V}
        # 调用 _strongly_connected_components 函数计算强连通分量并返回结果
        return _strongly_connected_components(V, Emap)

    def rref(A):
        """
        返回矩阵 A 的行简化阶梯形式及主元列的列表。

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(2), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.rref()
        ({0: {0: 1, 1: 2}}, [0])

        """
        # 调用 sdm_irref 函数获取矩阵 A 的初等行变换后的矩阵 B 和主元列的列表 pivots
        B, pivots, _ = sdm_irref(A)
        # 返回新的 SDM 对象，其包含 B 的内容，形状与 A 相同，定义域与 A 相同，以及主元列的列表 pivots
        return A.new(B, A.shape, A.domain), pivots

    def rref_den(A):
        """
        返回矩阵 A 的带有分母的行简化阶梯形式及主元列的列表。

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(2), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.rref_den()
        ({0: {0: 1, 1: 2}}, 1, [0])

        """
        # 获取矩阵 A 的定义域 K
        K = A.domain
        # 调用 sdm_rref_den 函数获取带分母的行简化阶梯形式 A_rref_sdm、分母 denom 和主元列的列表 pivots
        A_rref_sdm, denom, pivots = sdm_rref_den(A, K)
        # 构建新的 SDM 对象 A_rref，其包含 A_rref_sdm 的内容，形状与 A 相同，定义域与 A 相同
        A_rref = A.new(A_rref_sdm, A.shape, A.domain)
        # 返回新的 SDM 对象 A_rref、分母 denom 和主元列的列表 pivots
        return A_rref, denom, pivots

    def inv(A):
        """
        返回矩阵 A 的逆矩阵。

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.inv()
        {0: {0: -2, 1: 1}, 1: {0: 3/2, 1: -1/2}}

        """
        # 将矩阵 A 转换为 DFM 或 DDM 格式，再计算其逆矩阵，最后转换为 SDM 格式返回
        return A.to_dfm_or_ddm().inv().to_sdm()

    def det(A):
        """
        返回矩阵 A 的行列式值。

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.det()
        -2

        """
        # 对于非常稀疏的矩阵，可能行列式为零，但是当前实现将其转换为密集矩阵并使用 ddm_idet 函数计算行列式
        # 如果 GROUND_TYPES=flint，则优先使用 Flint 的实现（dfm）
        return A.to_dfm_or_ddm().det()
    def lu(A):
        """
        对矩阵 A 进行 LU 分解

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.lu()
        ({0: {0: 1}, 1: {0: 3, 1: 1}}, {0: {0: 1, 1: 2}, 1: {1: -2}}, [])

        """
        # 将 SDM 矩阵 A 转换为 DDM 格式并进行 LU 分解
        L, U, swaps = A.to_ddm().lu()
        # 将结果转换回 SDM 格式并返回
        return A.from_ddm(L), A.from_ddm(U), swaps

    def lu_solve(A, b):
        """
        使用 LU 分解解决方程组 Ax = b

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> b = SDM({0:{0:QQ(1)}, 1:{0:QQ(2)}}, (2, 1), QQ)
        >>> A.lu_solve(b)
        {1: {0: 1/2}}

        """
        # 将矩阵 A 和向量 b 转换为 DDM 格式并使用 LU 分解求解线性方程组
        return A.from_ddm(A.to_ddm().lu_solve(b.to_ddm()))

    def nullspace(A):
        """
        返回 :py:class:`~.SDM` 矩阵 A 的零空间

        矩阵的定义域必须是一个域。

        推荐使用 :meth:`~.DomainMatrix.nullspace` 方法获取矩阵的零空间。

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0: QQ(2), 1: QQ(4)}}, (2, 2), QQ)
        >>> A.nullspace()
        ({0: {0: -2, 1: 1}}, [1])


        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
            获取矩阵零空间的首选方法。

        """
        # 获取矩阵 A 的列数
        ncols = A.shape[1]
        # 获取矩阵 A 的域的单位元素
        one = A.domain.one
        # 对 A 进行行简化形式处理，返回 B 矩阵、主元、非零列
        B, pivots, nzcols = sdm_irref(A)
        # 从简化行阶梯形式 B 中计算零空间 K 和非主元列索引 nonpivots
        K, nonpivots = sdm_nullspace_from_rref(B, one, ncols, pivots, nzcols)
        # 将 K 转换为字典形式并确定其形状
        K = dict(enumerate(K))
        shape = (len(K), ncols)
        # 返回新的 SDM 矩阵和非主元列索引
        return A.new(K, shape, A.domain), nonpivots
    def nullspace_from_rref(A, pivots=None):
        """
        Returns nullspace for a :py:class:`~.SDM` matrix ``A`` in RREF.

        The domain of the matrix can be any domain.

        The matrix must already be in reduced row echelon form (RREF).

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0: QQ(2), 1: QQ(4)}}, (2, 2), QQ)
        >>> A_rref, pivots = A.rref()
        >>> A_null, nonpivots = A_rref.nullspace_from_rref(pivots)
        >>> A_null
        {0: {0: -2, 1: 1}}
        >>> pivots
        [0]
        >>> nonpivots
        [1]

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
            The higher-level function that would usually be called instead of
            calling this one directly.

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace_from_rref
            The higher-level direct equivalent of this function.

        sympy.polys.matrices.ddm.DDM.nullspace_from_rref
            The equivalent function for dense :py:class:`~.DDM` matrices.

        """
        # 获取矩阵的行数和列数
        m, n = A.shape
        # 获取矩阵的域（例如 QQ 代表有理数域）
        K = A.domain

        # 如果未指定主元，则找到矩阵中每一行的最小元素作为主元列表
        if pivots is None:
            pivots = sorted(map(min, A.values()))

        # 如果主元列表为空，则返回一个 n x n 的单位矩阵和所有列的列表作为非主元
        if not pivots:
            return A.eye((n, n), K), list(range(n))
        # 如果主元的数量等于列数，则返回一个 m x n 的零矩阵和空列表，表示零空间为空
        elif len(pivots) == n:
            return A.zeros((0, n), K), []

        # 在无分式的 RREF 中，用于主元的非零条目不一定是 1。
        # 断言第一个主元的值不为零
        pivot_val = A[0][pivots[0]]
        assert not K.is_zero(pivot_val)

        # 创建主元索引的集合
        pivots_set = set(pivots)

        # 创建一个字典，将每列中的非零条目与其行索引映射起来，形成矩阵的转置
        nonzero_cols = defaultdict(list)
        for i, Ai in A.items():
            for j, Aij in Ai.items():
                nonzero_cols[j].append((i, Aij))

        # 初始化基础列表和非主元列表
        basis = []
        nonpivots = []
        # 遍历每一列
        for j in range(n):
            # 如果列索引在主元集合中，则跳过
            if j in pivots_set:
                continue
            # 将当前列索引加入非主元列表
            nonpivots.append(j)

            # 创建一个向量字典，将当前列的主元值作为值
            vec = {j: pivot_val}
            # 对于当前列中的每个非零条目，将其加入向量字典，并使用相应的系数调整主元值
            for ip, Aij in nonzero_cols[j]:
                vec[pivots[ip]] = -Aij

            # 将形成的向量字典加入基础列表
            basis.append(vec)

        # 根据基础列表创建 SDM 类型的新矩阵 A_null
        sdm = dict(enumerate(basis))
        A_null = A.new(sdm, (len(basis), n), K)

        return (A_null, nonpivots)
    # 求解给定矩阵 A 的特解
    def particular(A):
        # 获取矩阵 A 的列数
        ncols = A.shape[1]
        # 对矩阵 A 进行初等行变换，返回最简行阶梯形式 B、主元列列表 pivots 和非零列列表 nzcols
        B, pivots, nzcols = sdm_irref(A)
        # 根据最简行阶梯形式 B、列数 ncols 和主元列列表 pivots 求解特解 P
        P = sdm_particular_from_rref(B, ncols, pivots)
        # 如果存在特解 P，则构造表示 P 的字典 rep；否则返回空字典
        rep = {0:P} if P else {}
        # 构建并返回新的 :py:class:`~.SDM` 对象，其中包含结果字典 rep，行数为 1，列数为 ncols-1，域与 A 相同
        return A.new(rep, (1, ncols-1), A.domain)
    
    # 将多个 :py:class:`~.SDM` 矩阵水平堆叠
    def hstack(A, *B):
        """Horizontally stacks :py:class:`~.SDM` matrices.
    
        Examples
        ========
    
        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
    
        >>> A = SDM({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}, (2, 2), ZZ)
        >>> B = SDM({0: {0: ZZ(5), 1: ZZ(6)}, 1: {0: ZZ(7), 1: ZZ(8)}}, (2, 2), ZZ)
        >>> A.hstack(B)
        {0: {0: 1, 1: 2, 2: 5, 3: 6}, 1: {0: 3, 1: 4, 2: 7, 3: 8}}
    
        >>> C = SDM({0: {0: ZZ(9), 1: ZZ(10)}, 1: {0: ZZ(11), 1: ZZ(12)}}, (2, 2), ZZ)
        >>> A.hstack(B, C)
        {0: {0: 1, 1: 2, 2: 5, 3: 6, 4: 9, 5: 10}, 1: {0: 3, 1: 4, 2: 7, 3: 8, 4: 11, 5: 12}}
        """
        # 复制矩阵 A 的内容到 Anew 字典中
        Anew = dict(A.copy())
        # 获取矩阵 A 的行数和列数
        rows, cols = A.shape
        # 获取矩阵 A 的域（数据类型）
        domain = A.domain
    
        # 遍历参数中的每个 :py:class:`~.SDM` 矩阵 Bk
        for Bk in B:
            # 获取矩阵 Bk 的行数和列数
            Bkrows, Bkcols = Bk.shape
            # 断言矩阵 Bk 的行数与矩阵 A 相同
            assert Bkrows == rows
            # 断言矩阵 Bk 的域与矩阵 A 相同
            assert Bk.domain == domain
    
            # 遍历矩阵 Bk 的元素
            for i, Bki in Bk.items():
                # 获取 Anew 中索引 i 处的字典，若不存在则新建
                Ai = Anew.get(i, None)
                if Ai is None:
                    Anew[i] = Ai = {}
                # 将矩阵 Bk 中的元素 Bkij 水平添加到对应行索引 i 的字典中，列索引为当前列数 cols
                for j, Bkij in Bki.items():
                    Ai[j + cols] = Bkij
            # 更新堆叠后的矩阵的列数
            cols += Bkcols
    
        # 构建并返回新的 :py:class:`~.SDM` 对象，包含堆叠后的内容 Anew，行数为 rows，列数为 cols，域与 A 相同
        return A.new(Anew, (rows, cols), A.domain)
    
    # 将多个 :py:class:`~.SDM` 矩阵垂直堆叠
    def vstack(A, *B):
        """Vertically stacks :py:class:`~.SDM` matrices.
    
        Examples
        ========
    
        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
    
        >>> A = SDM({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}, (2, 2), ZZ)
        >>> B = SDM({0: {0: ZZ(5), 1: ZZ(6)}, 1: {0: ZZ(7), 1: ZZ(8)}}, (2, 2), ZZ)
        >>> A.vstack(B)
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}, 2: {0: 5, 1: 6}, 3: {0: 7, 1: 8}}
    
        >>> C = SDM({0: {0: ZZ(9), 1: ZZ(10)}, 1: {0: ZZ(11), 1: ZZ(12)}}, (2, 2), ZZ)
        >>> A.vstack(B, C)
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}, 2: {0: 5, 1: 6}, 3: {0: 7, 1: 8}, 4: {0: 9, 1: 10}, 5: {0: 11, 1: 12}}
        """
        # 复制矩阵 A 的内容到 Anew 字典中
        Anew = dict(A.copy())
        # 获取矩阵 A 的行数和列数
        rows, cols = A.shape
        # 获取矩阵 A 的域（数据类型）
        domain = A.domain
    
        # 遍历参数中的每个 :py:class:`~.SDM` 矩阵 Bk
        for Bk in B:
            # 获取矩阵 Bk 的行数和列数
            Bkrows, Bkcols = Bk.shape
            # 断言矩阵 Bk 的列数与矩阵 A 相同
            assert Bkcols == cols
            # 断言矩阵 Bk 的域与矩阵 A 相同
            assert Bk.domain == domain
    
            # 遍历矩阵 Bk 的元素
            for i, Bki in Bk.items():
                # 将矩阵 Bk 中的元素 Bki 垂直添加到 Anew 中的下一行，行索引为当前行数 rows
                Anew[rows + i] = Bki
            # 更新堆叠后的矩阵的行数
            rows += Bkrows
    
        # 构建并返回新的 :py:class:`~.SDM` 对象，包含堆叠后的内容 Anew，行数为 rows，列数为 cols，域与 A 相同
        return A.new(Anew, (rows, cols), A.domain)
    
    # 将矩阵中的每个元素应用给定的函数 func，并返回结果构成的新 :py:class:`~.SDM` 对象
    def applyfunc(self, func, domain):
        # 对矩阵 self 中的每个元素应用函数 func，并构建新的字典表示 sdm
        sdm = {i: {j: func(e) for j, e in row.items()} for i, row in self.items()}
        # 返回一个新的 :py:class:`~.SDM` 对象，包含字典表示 sdm、与原矩阵相同的形状和给定的域
        return self.new(sdm, self.shape, domain)
    def charpoly(A):
        """
        Returns the coefficients of the characteristic polynomial
        of the :py:class:`~.SDM` matrix. These elements will be domain elements.
        The domain of the elements will be same as domain of the :py:class:`~.SDM`.

        Examples
        ========

        >>> from sympy import QQ, Symbol
        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy.polys import Poly
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.charpoly()
        [1, -5, -2]

        We can create a polynomial using the
        coefficients using :py:class:`~.Poly`

        >>> x = Symbol('x')
        >>> p = Poly(A.charpoly(), x, domain=A.domain)
        >>> p
        Poly(x**2 - 5*x - 2, x, domain='QQ')

        """
        K = A.domain  # 获取矩阵 A 的定义域
        n, _ = A.shape  # 获取矩阵 A 的形状信息
        pdict = sdm_berk(A, n, K)  # 调用 sdm_berk 函数计算 A 的 Berkowitz 格式多项式
        plist = [K.zero] * (n + 1)  # 创建一个长度为 n+1 的零系数列表
        for i, pi in pdict.items():
            plist[i] = pi  # 将计算得到的多项式系数填入 plist 中对应的位置
        return plist

    def is_zero_matrix(self):
        """
        Says whether this matrix has all zero entries.
        """
        return not self  # 返回矩阵是否全零的布尔值

    def is_upper(self):
        """
        Says whether this matrix is upper-triangular. True can be returned
        even if the matrix is not square.
        """
        return all(i <= j for i, row in self.items() for j in row)  # 检查矩阵是否为上三角矩阵

    def is_lower(self):
        """
        Says whether this matrix is lower-triangular. True can be returned
        even if the matrix is not square.
        """
        return all(i >= j for i, row in self.items() for j in row)  # 检查矩阵是否为下三角矩阵

    def is_diagonal(self):
        """
        Says whether this matrix is diagonal. True can be returned
        even if the matrix is not square.
        """
        return all(i == j for i, row in self.items() for j in row)  # 检查矩阵是否为对角矩阵

    def diagonal(self):
        """
        Returns the diagonal of the matrix as a list.
        """
        m, n = self.shape  # 获取矩阵的形状信息
        zero = self.domain.zero  # 获取零元素
        return [row.get(i, zero) for i, row in self.items() if i < n]  # 返回矩阵的对角线元素列表

    def lll(A, delta=QQ(3, 4)):
        """
        Returns the LLL-reduced basis for the :py:class:`~.SDM` matrix.
        """
        return A.to_dfm_or_ddm().lll(delta=delta).to_sdm()  # 调用 LLL 算法对 SDM 矩阵进行约化处理并返回结果

    def lll_transform(A, delta=QQ(3, 4)):
        """
        Returns the LLL-reduced basis and transformation matrix.
        """
        reduced, transform = A.to_dfm_or_ddm().lll_transform(delta=delta)  # 对 SDM 矩阵进行 LLL 变换并返回约化后的矩阵及其变换矩阵
        return reduced.to_sdm(), transform.to_sdm()  # 返回约化后的 SDM 矩阵及其变换矩阵
def sdm_matmul(A, B, K, m, o):
    #
    # Should be fast if A and B are very sparse.
    # Consider e.g. A = B = eye(1000).
    #
    # The idea here is that we compute C = A*B in terms of the rows of C and
    # B since the dict of dicts representation naturally stores the matrix as
    # rows. The ith row of C (Ci) is equal to the sum of Aik * Bk where Bk is
    # the kth row of B. The algorithm below loops over each nonzero element
    # Aik of A and if the corresponding row Bj is nonzero then we do
    #    Ci += Aik * Bk.
    # To make this more efficient we don't need to loop over all elements Aik.
    # Instead for each row Ai we compute the intersection of the nonzero
    # columns in Ai with the nonzero rows in B. That gives the k such that
    # Aik and Bk are both nonzero. In Python the intersection of two sets
    # of int can be computed very efficiently.
    #

    # 如果 K 是 EXRAW，则调用特定的扩展矩阵乘法函数
    if K.is_EXRAW:
        return sdm_matmul_exraw(A, B, K, m, o)

    # 初始化结果矩阵 C
    C = {}

    # 获取 B 的非零行的集合
    B_knz = set(B)
    # 遍历字典 A 中的每个键值对 (i, Ai)，其中 Ai 是一个字典
    for i, Ai in A.items():
        # 初始化空字典 Ci，用于存储结果
        Ci = {}
        # 获取 Ai 中所有键的集合
        Ai_knz = set(Ai)
        # 遍历 Ai 中键的集合与字典 B 的键的集合的交集
        for k in Ai_knz & B_knz:
            # 获取 Ai[k] 的值 Aik
            Aik = Ai[k]
            # 遍历字典 B 中键 k 对应的每个键值对 (j, Bkj)
            for j, Bkj in B[k].items():
                # 获取 Ci 中键 j 对应的值 Cij
                Cij = Ci.get(j, None)
                # 如果 Cij 不为 None，则更新 Cij 的值为原值加上 Aik 乘以 Bkj 的结果
                if Cij is not None:
                    Cij = Cij + Aik * Bkj
                    # 如果更新后的 Cij 不为 0，则更新 Ci[j] 的值为 Cij
                    if Cij:
                        Ci[j] = Cij
                    # 如果更新后的 Cij 为 0，则从 Ci 中移除键 j
                    else:
                        Ci.pop(j)
                # 如果 Ci 中键 j 对应的值 Cij 为 None，则直接将 Aik 乘以 Bkj 的结果赋给 Cij
                else:
                    Cij = Aik * Bkj
                    # 如果 Cij 不为 0，则更新 Ci[j] 的值为 Cij
                    if Cij:
                        Ci[j] = Cij
            # 如果 Ci 非空，则将 Ci 赋给 C 中的键 i
            if Ci:
                C[i] = Ci
    # 返回结果字典 C
    return C
# 定义一个函数 sdm_matmul_exraw，用于稀疏矩阵的扩展乘法运算
def sdm_matmul_exraw(A, B, K, m, o):
    # 与上面的 sdm_matmul 类似，但有以下不同：
    # - 处理例如 0*oo -> nan 的情况（sdm_matmul 跳过乘以零的操作）
    # - 使用 K.sum (Add(*items)) 实现表达式的高效加法运算

    # 初始化 zero 变量，用于表示零元素
    zero = K.zero
    # 初始化结果字典 C
    C = {}

    # 将 B 中的非零元素放入集合 B_knz 中
    B_knz = set(B)
    
    # 遍历 A 的每一行
    for i, Ai in A.items():
        # 创建 Ci_list，用于存储每一列的乘积结果列表
        Ci_list = defaultdict(list)
        # 将 Ai 中的非零元素放入集合 Ai_knz 中
        Ai_knz = set(Ai)

        # 处理非零行/列对
        for k in Ai_knz & B_knz:
            Aik = Ai[k]
            # 若 Aik 乘以 zero 等于 zero，则说明 Aik 是零元素
            if zero * Aik == zero:
                # 主要的内部循环：
                for j, Bkj in B[k].items():
                    Ci_list[j].append(Aik * Bkj)
            else:
                for j in range(o):
                    Ci_list[j].append(Aik * B[k].get(j, zero))

        # 处理 B 中的零行，检查 A 中的无穷元素
        for k in Ai_knz - B_knz:
            zAik = zero * Ai[k]
            # 若 zAik 不等于 zero，则说明 Ai 中有无穷元素
            if zAik != zero:
                for j in range(o):
                    Ci_list[j].append(zAik)

        # 使用 K.sum (Add(*terms)) 进行项的加法以提高效率
        Ci = {}
        for j, Cij_list in Ci_list.items():
            Cij = K.sum(Cij_list)
            # 若 Cij 不为零，则将其加入 Ci 中
            if Cij:
                Ci[j] = Cij
        # 若 Ci 非空，则将其加入结果字典 C 中
        if Ci:
            C[i] = Ci

    # 查找 B 中所有的无穷元素
    for k, Bk in B.items():
        for j, Bkj in Bk.items():
            if zero * Bkj != zero:
                for i in range(m):
                    Aik = A.get(i, {}).get(k, zero)
                    # 若 Aik 是零元素，则在上述循环中已经处理
                    if Aik == zero:
                        Ci = C.get(i, {})
                        Cij = Ci.get(j, zero) + Aik * Bkj
                        # 若 Cij 不为零，则将其加入 Ci 中
                        if Cij != zero:
                            Ci[j] = Cij
                        else:  # pragma: no cover
                            # 不确定如何到达这里，但为了安全起见，引发异常
                            raise RuntimeError
                        C[i] = Ci

    # 返回最终的结果字典 C
    return C
    """
    The cost of this algorithm is determined purely by the nonzero elements of
    the matrix. No part of the cost of any step in this algorithm depends on
    the number of rows or columns in the matrix. No step depends even on the
    number of nonzero rows apart from the primary loop over those rows. The
    implementation is much faster than ddm_rref for sparse matrices. In fact
    at the time of writing it is also (slightly) faster than the dense
    implementation even if the input is a fully dense matrix so it seems to be
    faster in all cases.

    The elements of the matrix should support exact division with ``/``. For
    example elements of any domain that is a field (e.g. ``QQ``) should be
    fine. No attempt is made to handle inexact arithmetic.

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref
        The higher-level function that would normally be used to call this
        routine.
    sympy.polys.matrices.dense.ddm_irref
        The dense equivalent of this routine.
    sdm_rref_den
        Fraction-free version of this routine.
    """
    #
    # Any zeros in the matrix are not stored at all so an element is zero if
    # its row dict has no index at that key. A row is entirely zero if its
    # row index is not in the outer dict. Since rref reorders the rows and
    # removes zero rows we can completely discard the row indices. The first
    # step then copies the row dicts into a list sorted by the index of the
    # first nonzero column in each row.
    #
    # The algorithm then processes each row Ai one at a time. Previously seen
    # rows are used to cancel their pivot columns from Ai. Then a pivot from
    # Ai is chosen and is cancelled from all previously seen rows. At this
    # point Ai joins the previously seen rows. Once all rows are seen all
    # elimination has occurred and the rows are sorted by pivot column index.
    #
    # The previously seen rows are stored in two separate groups. The reduced
    # group consists of all rows that have been reduced to a single nonzero
    # element (the pivot). There is no need to attempt any further reduction
    # with these. Rows that still have other nonzeros need to be considered
    # when Ai is cancelled from the previously seen rows.
    #
    # A dict nonzerocolumns is used to map from a column index to a set of
    # previously seen rows that still have a nonzero element in that column.
    # This means that we can cancel the pivot from Ai into the previously seen
    # rows without needing to loop over each row that might have a zero in
    # that column.
    #

    # Row dicts sorted by index of first nonzero column
    # (Maybe sorting is not needed/useful.)
    Arows = sorted((Ai.copy() for Ai in A.values()), key=min)

    # Each processed row has an associated pivot column.
    # pivot_row_map maps from the pivot column index to the row dict.
    # This means that we can represent a set of rows purely as a set of their
    # 存储每行的主元素的列索引的映射
    pivot_row_map = {}

    # 存储完全化简为非零的行的主元素的列索引的集合
    reduced_pivots = set()

    # 存储未完全化简的行的主元素的列索引的集合
    nonreduced_pivots = set()

    # 字典，键为列索引，值为具有在该列上存在非零元素的行的主元素的集合
    nonzero_columns = defaultdict(set)

    while Arows:
        # 从 Arows 中取出一个行
        Ai = Arows.pop()

        # 从完全化简的主元素的列索引集合中移除该行中已经完全化简的列
        Ai = {j: Aij for j, Aij in Ai.items() if j not in reduced_pivots}

        # 对于未完全化简的主元素的列索引集合与 Ai 中存在的列进行全行消除
        for j in nonreduced_pivots & set(Ai):
            Aj = pivot_row_map[j]
            Aij = Ai[j]
            Ainz = set(Ai)
            Ajnz = set(Aj)
            for k in Ajnz - Ainz:
                Ai[k] = - Aij * Aj[k]
            Ai.pop(j)
            Ainz.remove(j)
            for k in Ajnz & Ainz:
                Aik = Ai[k] - Aij * Aj[k]
                if Aik:
                    Ai[k] = Aik
                else:
                    Ai.pop(k)

        # 如果行已经被完全消除则跳过
        if not Ai:
            continue

        # 选择 Ai 中的一个主元素
        j = min(Ai)
        Aij = Ai[j]
        pivot_row_map[j] = Ai
        Ainz = set(Ai)

        # 将主元素归一化为1
        Aijinv = Aij**-1
        for l in Ai:
            Ai[l] *= Aijinv

        # 使用 Aij 来消除之前出现的所有行中的列 j
        for k in nonzero_columns.pop(j, ()):
            Ak = pivot_row_map[k]
            Akj = Ak[j]
            Aknz = set(Ak)
            for l in Ainz - Aknz:
                Ak[l] = - Akj * Ai[l]
                nonzero_columns[l].add(k)
            Ak.pop(j)
            Aknz.remove(j)
            for l in Ainz & Aknz:
                Akl = Ak[l] - Akj * Ai[l]
                if Akl:
                    Ak[l] = Akl
                else:
                    # 删除非零元素
                    Ak.pop(l)
                    if l != j:
                        nonzero_columns[l].remove(k)
            if len(Ak) == 1:
                reduced_pivots.add(k)
                nonreduced_pivots.remove(k)

        if len(Ai) == 1:
            reduced_pivots.add(j)
        else:
            nonreduced_pivots.add(j)
            for l in Ai:
                if l != j:
                    nonzero_columns[l].add(j)

    # 所有操作完成！
    pivots = sorted(reduced_pivots | nonreduced_pivots)
    pivot2row = {p: n for n, p in enumerate(pivots)}
    nonzero_columns = {c: {pivot2row[p] for p in s} for c, s in nonzero_columns.items()}
    rows = [pivot_row_map[i] for i in pivots]
    rref = dict(enumerate(rows))
    # 返回函数结果，包括三个变量：rref（行简化阶梯形式矩阵）、pivots（主元所在列的索引列表）、nonzero_columns（非零列的索引列表）
    return rref, pivots, nonzero_columns
#
# We represent each row of the matrix as a dict mapping column indices to
# nonzero elements. We will build the RREF matrix starting from an empty
# matrix and appending one row at a time. At each step we will have the
# RREF of the rows we have processed so far.
#

#
# Our representation of the RREF divides it into three parts:
#
# 1. Fully reduced rows having only a single nonzero element (the pivot).
# 2. Partially reduced rows having nonzeros after the pivot.
# 3. The current denominator and divisor.
#

#
# For example if the incremental RREF might be:
#
#   [2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#   [0, 0, 2, 0, 0, 0, 7, 0, 0, 0]
#   [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
#   [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
#   [0, 0, 0, 0, 0, 0, 0, 0, 2, 0]
#
# Here the second row is partially reduced and the other rows are fully
# reduced. The denominator would be 2 in this case. We distinguish the
# fully reduced rows because we can handle them more efficiently when
# adding a new row.
#
    # 处理添加新行时的步骤：首先将新行乘以当前的分母。
    # 然后通过与之前行的交叉消除来减少新行。
    # 如果新行没有被减少为零，则将其主导元素作为新的主元素，从之前的行中交叉消除新行，并更新分母。
    # 在无分数版本中，最后一步需要通过新的主元素和当前除数来乘除整个矩阵。
    # 逐行构建RREF的优点在于，在稀疏情况下，我们只需要处理矩阵的相对稀疏的上部行。
    # FFGJ的简化版本[1]中，简单版本会在每一步中乘除所有密集的下部行。

    # 处理特殊情况：如果 A 是空的，则返回空字典、单位元素和空列表。
    if not A:
        return ({}, K.one, [])
    # 如果 A 只有一行，则从 A 中获取唯一的值 Ai，并找到其最小索引 j。
    # 将该行 Ai 复制到字典中的索引 0 处，返回字典、Aij 和包含 j 的列表。
    elif len(A) == 1:
        Ai, = A.values()
        j = min(Ai)
        Aij = Ai[j]
        return ({0: Ai.copy()}, Aij, [j])

    # 对于不精确的域，如 RR[x]，我们使用 quo 并丢弃余数。
    # 或许让 K.exquo 自动处理这一点会更好。
    if K.is_Exact:
        exquo = K.exquo
    else:
        exquo = K.quo

    # 确保行按顺序排列，以使结果在一开始就是确定性的。
    _, rows_in_order = zip(*sorted(A.items()))

    # 初始化列到已减少行和未减少行的映射字典。
    col_to_row_reduced = {}
    col_to_row_unreduced = {}

    # 获取已减少和未减少行的键集合。
    reduced = col_to_row_reduced.keys()
    unreduced = col_to_row_unreduced.keys()

    # 到目前为止RREF的表示。
    A_rref_rows = []
    # 分母和除数的初始化。
    denom = None
    divisor = None

    # 将要添加到RREF中的行，按其主导元素的列索引排序。
    A_rows = sorted(rows_in_order, key=min)

    # 如果分母尚未设置，则将其设置为单位元素。
    if denom is None:
        denom = K.one

    # 合并已减少和未减少行的映射字典，形成列到行的映射。
    col_to_row = {**col_to_row_reduced, **col_to_row_unreduced}

    # 创建行到列索引的映射。
    row_to_col = {i: j for j, i in col_to_row.items()}

    # 创建带有列索引和行数据的元组列表。
    A_rref_rows_col = [(row_to_col[i], Ai) for i, Ai in enumerate(A_rref_rows)]

    # 对行进行排序，按其主导元素的列索引排序。
    pivots, A_rref = zip(*sorted(A_rref_rows_col))
    pivots = list(pivots)

    # 将主导元素的值插入到每一行中。
    for i, Ai in enumerate(A_rref):
        Ai[pivots[i]] = denom

    # 将结果转换为字典形式。
    A_rref_sdm = dict(enumerate(A_rref))

    # 返回RREF的字典表示，分母和主导元素的列表。
    return A_rref_sdm, denom, pivots
    # 初始化空字典 P，用于存储特解
    P = {}
    # 遍历主元列的索引和值 (i, j)
    for i, j in enumerate(pivots):
        # 获取矩阵 A 的第 i 行第 ncols-1 列的值 Ain
        Ain = A[i].get(ncols-1, None)
        # 如果 Ain 不为 None，计算特解 P[j] = Ain / A[i][j]
        if Ain is not None:
            P[j] = Ain / A[i][j]
    # 返回特解字典 P
    return P


def sdm_berk(M, n, K):
    """
    Berkowitz algorithm for computing the characteristic polynomial.

    Explanation
    ===========

    The Berkowitz algorithm is a division-free algorithm for computing the
    characteristic polynomial of a matrix over any commutative ring using only
    arithmetic in the coefficient ring. This implementation is for sparse
    matrices represented in a dict-of-dicts format (like :class:`SDM`).

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.polys.matrices.sdm import sdm_berk
    >>> from sympy.polys.domains import ZZ
    >>> M = {0: {0: ZZ(1), 1:ZZ(2)}, 1: {0:ZZ(3), 1:ZZ(4)}}
    >>> sdm_berk(M, 2, ZZ)
    {0: 1, 1: -5, 2: -2}
    >>> Matrix([[1, 2], [3, 4]]).charpoly()
    PurePoly(lambda**2 - 5*lambda - 2, lambda, domain='ZZ')

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.charpoly
        The high-level interface to this function.
    sympy.polys.matrices.dense.ddm_berk
        The dense version of this function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Samuelson%E2%80%93Berkowitz_algorithm
    """
    # 获取零元素和单位元素
    zero = K.zero
    one = K.one

    # 如果 n 等于 0，返回 {0: one}
    if n == 0:
        return {0: one}
    # 如果 n 等于 1，计算特殊情况下的字典 pdict
    elif n == 1:
        pdict = {0: one}
        # 如果 M 中存在 M[0][0]，则设置 pdict[1] = -M[0][0]
        if M00 := M.get(0, {}).get(0, zero):
            pdict[1] = -M00

    # 按照稀疏矩阵的结构，将 M 拆分为四个部分：a, R, C, A
    a, R, C, A = K.zero, {}, {}, defaultdict(dict)
    for i, Mi in M.items():
        for j, Mij in Mi.items():
            # 根据 i 和 j 的值分别存储到 a, R, C, A 中对应的位置
            if i and j:
                A[i-1][j-1] = Mij
            elif i:
                C[i-1] = Mij
            elif j:
                R[j-1] = Mij
            else:
                a = Mij

    # 初始化 AnC 为 C
    AnC = C
    # 计算 R*C 的乘积，结果存储在 RC 中
    RC = sdm_dotvec(R, C, K)

    # 初始化 Tvals 列表，包含 T 矩阵的首列非零元素的初始值
    Tvals = [one, -a, -RC]
    # 对于 i 从 3 到 n+1 的范围进行循环，计算 AnC = sdm_matvecmul(A, AnC, K)
    for i in range(3, n+1):
        AnC = sdm_matvecmul(A, AnC, K)
        # 如果 AnC 为空（假值），则跳出循环
        if not AnC:
            break
        # 计算 RAnC = sdm_dotvec(R, AnC, K)，并将其加入到 Tvals 列表中
        RAnC = sdm_dotvec(R, AnC, K)
        Tvals.append(-RAnC)

    # 剥离 Tvals 列表末尾的零值元素
    while Tvals and not Tvals[-1]:
        Tvals.pop()

    # 调用 sdm_berk 函数计算 q = sdm_berk(A, n-1, K)
    q = sdm_berk(A, n-1, K)

    # 重新排序 Tvals 列表，以便后续操作
    Tvals = Tvals[::-1]

    # 初始化空字典 Tq 用于存储结果
    Tq = {}

    # 遍历计算 Tq 的值，根据算法特性和稀疏性进行优化计算
    for i in range(min(q), min(max(q)+len(Tvals), n+1)):
        # 构建 Ti 字典，其键从 i-len(Tvals)+1 到 i，值来自 Tvals 列表
        Ti = dict(enumerate(Tvals, i-len(Tvals)+1))
        # 如果 Ti 和 q 的点积非零，则将结果存入 Tq 中
        if Tqi := sdm_dotvec(Ti, q, K):
            Tq[i] = Tqi

    # 返回计算结果 Tq 字典
    return Tq
```
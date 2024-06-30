# `D:\src\scipysrc\sympy\sympy\polys\matrices\ddm.py`

```
# 导入 itertools 中的 chain 函数，用于扁平化多个可迭代对象的结果
from itertools import chain

# 导入 GROUND_TYPES，此处应该是外部模块中的一个设置，用于确定底层类型
from sympy.external.gmpy import GROUND_TYPES

# 导入装饰器函数 doctest_depends_on，用于管理 doctest 的依赖项
from sympy.utilities.decorator import doctest_depends_on

# 导入自定义异常类，包括 DMBadInputError、DMDomainError、DMNonSquareMatrixError 和 DMShapeError
from .exceptions import (
    DMBadInputError,
    DMDomainError,
    DMNonSquareMatrixError,
    DMShapeError,
)

# 导入符号计算库 SymPy 中的 QQ 域
from sympy.polys.domains import QQ

# 导入自定义模块 dense 中的多个函数，这些函数用于操作密集矩阵（DDM 类的基础数据结构）
from .dense import (
    ddm_transpose,
    ddm_iadd,
    ddm_isub,
    ddm_ineg,
    ddm_imul,
    ddm_irmul,
    ddm_imatmul,
    ddm_irref,
    ddm_irref_den,
    ddm_idet,
    ddm_iinv,
    ddm_ilu_split,
    ddm_ilu_solve,
    ddm_berk,
)

# 导入自定义模块 lll 中的函数，用于执行 LLL 算法及其变体
from .lll import ddm_lll, ddm_lll_transform

# 如果 GROUND_TYPES 不是 'flint'，则在 doctest 中跳过特定的测试示例，防止错误的测试结果
if GROUND_TYPES != 'flint':
    __doctest_skip__ = ['DDM.to_dfm', 'DDM.to_dfm_or_ddm']

# 定义 DDM 类，继承自内置列表类 list
class DDM(list):
    """Dense matrix based on polys domain elements
    
    This is a list subclass and is a wrapper for a list of lists that supports
    basic matrix arithmetic +, -, *, **.
    """
    
    # 设定格式为'dense'
    fmt = 'dense'
    # 标记不是DFM类型
    is_DFM = False
    # 标记是DDM类型
    is_DDM = True
    
    # 初始化方法，接受rowslist（行列表）、shape（形状）和domain（域）
    def __init__(self, rowslist, shape, domain):
        # 如果rowslist不是列表，或者其中有任何一个元素不是列表，抛出异常
        if not (isinstance(rowslist, list) and all(type(row) is list for row in rowslist)):
            raise DMBadInputError("rowslist must be a list of lists")
        # 解构shape元组为m和n
        m, n = shape
        # 如果rowslist的长度不等于m，或者任何一行的长度不等于n，抛出异常
        if len(rowslist) != m or any(len(row) != n for row in rowslist):
            raise DMBadInputError("Inconsistent row-list/shape")
    
        # 调用父类初始化方法，将rowslist传递给父类构造方法
        super().__init__(rowslist)
        # 设置矩阵的形状、行数和列数
        self.shape = (m, n)
        self.rows = m
        self.cols = n
        # 设置矩阵的域
        self.domain = domain
    
    # 获取元素的方法，给定索引i和j
    def getitem(self, i, j):
        return self[i][j]
    
    # 设置元素的方法，给定索引i、j和值value
    def setitem(self, i, j, value):
        self[i][j] = value
    
    # 提取切片的方法，给定slice1和slice2作为参数
    def extract_slice(self, slice1, slice2):
        # 从self中提取切片slice1和slice2的数据，并创建一个新的DDM对象
        ddm = [row[slice2] for row in self[slice1]]
        rows = len(ddm)
        cols = len(ddm[0]) if ddm else len(range(self.shape[1])[slice2])
        return DDM(ddm, (rows, cols), self.domain)
    
    # 提取部分数据的方法，给定行列表rows和列列表cols作为参数
    def extract(self, rows, cols):
        # 遍历行列表rows，从self中提取指定行和列的数据，并创建一个新的DDM对象
        ddm = []
        for i in rows:
            rowi = self[i]
            ddm.append([rowi[j] for j in cols])
        return DDM(ddm, (len(rows), len(cols)), self.domain)
    
    # 从列表创建DDM对象的类方法，接受rowslist（行列表）、shape（形状）和domain（域）作为参数
    @classmethod
    def from_list(cls, rowslist, shape, domain):
        """
        Create a :class:`DDM` from a list of lists.
    
        Examples
        ========
    
        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM.from_list([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)
        >>> A
        [[0, 1], [-1, 0]]
        >>> A == DDM([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)
        True
    
        See Also
        ========
    
        from_list_flat
        """
        return cls(rowslist, shape, domain)
    
    # 从另一个DDM对象创建DDM对象的类方法，接受other作为参数
    @classmethod
    def from_ddm(cls, other):
        return other.copy()
    
    # 将DDM对象转换为列表的方法
    def to_list(self):
        """
        Convert to a list of lists.
    
        Examples
        ========
    
        >>> from sympy import QQ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_list()
        [[1, 2], [3, 4]]
    
        See Also
        ========
    
        to_list_flat
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_list
        """
        return list(self)
    def to_list_flat(self):
        """
        Convert to a flat list of elements.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_list_flat()
        [1, 2, 3, 4]
        >>> A == DDM.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.to_list_flat
        """
        # Initialize an empty list to store flattened elements
        flat = []
        # Iterate over each row in the matrix
        for row in self:
            # Extend the 'flat' list with elements from the current row
            flat.extend(row)
        # Return the flattened list of elements
        return flat

    @classmethod
    def from_list_flat(cls, flat, shape, domain):
        """
        Create a :class:`DDM` from a flat list of elements.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM.from_list_flat([1, 2, 3, 4], (2, 2), QQ)
        >>> A
        [[1, 2], [3, 4]]
        >>> A == DDM.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        to_list_flat
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_list_flat
        """
        # Ensure 'flat' is a list
        assert type(flat) is list
        # Extract rows and columns from 'shape'
        rows, cols = shape
        # Check if the length of 'flat' matches expected matrix dimensions
        if not (len(flat) == rows*cols):
            # Raise an error for inconsistent flat-list shape
            raise DMBadInputError("Inconsistent flat-list shape")
        # Convert 'flat' into a list of lists (matrix format)
        lol = [flat[i*cols:(i+1)*cols] for i in range(rows)]
        # Return a new instance of DDM initialized with the matrix 'lol'
        return cls(lol, shape, domain)

    def flatiter(self):
        """
        Return an iterator over the flattened elements of the matrix.

        This method uses itertools.chain.from_iterable to flatten the matrix.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> list(A.flatiter())
        [1, 2, 3, 4]
        """
        return chain.from_iterable(self)

    def flat(self):
        """
        Return a flattened list of elements from the matrix.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.flat()
        [1, 2, 3, 4]
        """
        items = []
        for row in self:
            items.extend(row)
        return items

    def to_flat_nz(self):
        """
        Convert to a flat list of nonzero elements and data.

        Explanation
        ===========

        This is used to operate on a list of the elements of a matrix and then
        reconstruct a matrix using :meth:`from_flat_nz`. Zero elements are
        included in the list but that may change in the future.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [1, 2, 3, 4]
        >>> A == DDM.from_flat_nz(elements, data, A.domain)
        True

        See Also
        ========

        from_flat_nz
        sympy.polys.matrices.sdm.SDM.to_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_flat_nz
        """
        # Convert the matrix to a SparseDomainMatrix (SDM) and then to flat non-zero elements
        return self.to_sdm().to_flat_nz()

    @classmethod
    def from_flat_nz(cls, elements, data, domain):
        """
        根据给定的元素、数据和域，从压缩的非零元素表示重构一个DDM对象。

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [1, 2, 3, 4]
        >>> A == DDM.from_flat_nz(elements, data, A.domain)
        True

        See Also
        ========

        to_flat_nz
        sympy.polys.matrices.sdm.SDM.from_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_flat_nz
        """
        return SDM.from_flat_nz(elements, data, domain).to_ddm()

    def to_dod(self):
        """
        将DDM对象转换为字典的字典（dod）格式。

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dod()
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}

        See Also
        ========

        from_dod
        sympy.polys.matrices.sdm.SDM.to_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dod
        """
        dod = {}
        for i, row in enumerate(self):
            row = {j:e for j, e in enumerate(row) if e}  # 创建每一行的非零元素字典
            if row:
                dod[i] = row  # 将非空行添加到dod中
        return dod

    @classmethod
    def from_dod(cls, dod, shape, domain):
        """
        根据字典的字典（dod）格式，创建一个DDM对象。

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> dod = {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
        >>> A = DDM.from_dod(dod, (2, 2), QQ)
        >>> A
        [[1, 2], [3, 4]]

        See Also
        ========

        to_dod
        sympy.polys.matrices.sdm.SDM.from_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_dod
        """
        rows, cols = shape
        lol = [[domain.zero] * cols for _ in range(rows)]  # 创建一个零填充的二维列表lol
        for i, row in dod.items():
            for j, element in row.items():
                lol[i][j] = element  # 根据dod填充lol中的元素
        return DDM(lol, shape, domain)  # 返回新创建的DDM对象

    def to_dok(self):
        """
        将DDM对象转换为字典的键（dok）格式。

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dok()
        {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}

        See Also
        ========

        from_dok
        sympy.polys.matrices.sdm.SDM.to_dok
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dok
        """
        dok = {}
        for i, row in enumerate(self):
            for j, element in enumerate(row):
                if element:
                    dok[i, j] = element  # 将非零元素作为键值对添加到dok中
        return dok
    def from_dok(cls, dok, shape, domain):
        """
        Create a :class:`DDM` from a dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> dok = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
        >>> A = DDM.from_dok(dok, (2, 2), QQ)
        >>> A
        [[1, 2], [3, 4]]

        See Also
        ========

        to_dok
        sympy.polys.matrices.sdm.SDM.from_dok
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_dok
        """
        # 解析出行数和列数
        rows, cols = shape
        # 初始化一个二维列表，用于存储矩阵数据，每个元素初始化为域的零元素
        lol = [[domain.zero] * cols for _ in range(rows)]
        # 遍历字典 dok 中的每个元素，将元素放入 lol 对应位置
        for (i, j), element in dok.items():
            lol[i][j] = element
        # 返回一个新的 DDM 对象，用 lol 列表初始化，传入形状和域
        return DDM(lol, shape, domain)

    def iter_values(self):
        """
        Iterater over the non-zero values of the matrix.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[QQ(1), QQ(0)], [QQ(3), QQ(4)]], (2, 2), QQ)
        >>> list(A.iter_values())
        [1, 3, 4]

        See Also
        ========

        iter_items
        to_list_flat
        sympy.polys.matrices.domainmatrix.DomainMatrix.iter_values
        """
        # 遍历矩阵的每一行，使用生成器过滤出非零元素并返回
        for row in self:
            yield from filter(None, row)

    def iter_items(self):
        """
        Iterate over indices and values of nonzero elements of the matrix.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[QQ(1), QQ(0)], [QQ(3), QQ(4)]], (2, 2), QQ)
        >>> list(A.iter_items())
        [((0, 0), 1), ((1, 0), 3), ((1, 1), 4)]

        See Also
        ========

        iter_values
        to_dok
        sympy.polys.matrices.domainmatrix.DomainMatrix.iter_items
        """
        # 遍历矩阵的每个非零元素，返回元素的索引和值的元组
        for i, row in enumerate(self):
            for j, element in enumerate(row):
                if element:
                    yield (i, j), element

    def to_ddm(self):
        """
        Convert to a :class:`DDM`.

        This just returns ``self`` but exists to parallel the corresponding
        method in other matrix types like :class:`~.SDM`.

        See Also
        ========

        to_sdm
        to_dfm
        to_dfm_or_ddm
        sympy.polys.matrices.sdm.SDM.to_ddm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_ddm
        """
        # 直接返回当前对象，用于与其他矩阵类型的方法保持一致性
        return self

    def to_sdm(self):
        """
        Convert to a :class:`~.SDM`.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_sdm()
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
        >>> type(A.to_sdm())
        <class 'sympy.polys.matrices.sdm.SDM'>

        See Also
        ========

        SDM
        sympy.polys.matrices.sdm.SDM.to_ddm
        """
        # 将当前 DDM 对象转换为 SDM 对象并返回
        return SDM.from_list(self, self.shape, self.domain)
    @doctest_depends_on(ground_types=['flint'])
    def to_dfm(self):
        """
        Convert to :class:`~.DDM` to :class:`~.DFM`.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dfm()
        [[1, 2], [3, 4]]
        >>> type(A.to_dfm())
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        DFM
        sympy.polys.matrices._dfm.DFM.to_ddm
        """
        # 将 DDM 对象转换为 DFM 对象，并返回结果
        return DFM(list(self), self.shape, self.domain)

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm_or_ddm(self):
        """
        Convert to :class:`~.DFM` if possible or otherwise return self.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dfm_or_ddm()
        [[1, 2], [3, 4]]
        >>> type(A.to_dfm_or_ddm())
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        to_dfm
        to_ddm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm_or_ddm
        """
        # 如果当前域支持 DFM，则转换为 DFM 对象并返回；否则返回当前对象自身
        if DFM._supports_domain(self.domain):
            return self.to_dfm()
        return self

    def convert_to(self, K):
        # 保存当前对象的域
        Kold = self.domain
        # 如果目标域与当前域相同，则复制当前对象并返回
        if K == Kold:
            return self.copy()
        # 将当前对象的元素按照目标域 K 进行转换，并生成新的 DDM 对象返回
        rows = [[K.convert_from(e, Kold) for e in row] for row in self]
        return DDM(rows, self.shape, K)

    def __str__(self):
        # 将对象转换为字符串形式，用于打印输出
        rowsstr = ['[%s]' % ', '.join(map(str, row)) for row in self]
        return '[%s]' % ', '.join(rowsstr)

    def __repr__(self):
        # 返回对象的字符串表示形式，用于调试和显示
        cls = type(self).__name__
        rows = list.__repr__(self)
        return '%s(%s, %s, %s)' % (cls, rows, self.shape, self.domain)

    def __eq__(self, other):
        # 检查两个 DDM 对象是否相等
        if not isinstance(other, DDM):
            return False
        return (super().__eq__(other) and self.domain == other.domain)

    def __ne__(self, other):
        # 检查两个 DDM 对象是否不相等
        return not self.__eq__(other)

    @classmethod
    def zeros(cls, shape, domain):
        # 创建一个指定形状和域的全零 DDM 对象并返回
        z = domain.zero
        m, n = shape
        rowslist = [[z] * n for _ in range(m)]
        return DDM(rowslist, shape, domain)

    @classmethod
    def ones(cls, shape, domain):
        # 创建一个指定形状和域的全一 DDM 对象并返回
        one = domain.one
        m, n = shape
        rowlist = [[one] * n for _ in range(m)]
        return DDM(rowlist, shape, domain)

    @classmethod
    def eye(cls, size, domain):
        # 创建一个单位矩阵 DDM 对象，大小为 size，域为 domain，并返回
        if isinstance(size, tuple):
            m, n = size
        elif isinstance(size, int):
            m = n = size
        one = domain.one
        ddm = cls.zeros((m, n), domain)
        for i in range(min(m, n)):
            ddm[i][i] = one
        return ddm

    def copy(self):
        # 复制当前 DDM 对象并返回副本
        copyrows = [row[:] for row in self]
        return DDM(copyrows, self.shape, self.domain)
    # 定义一个实例方法，用于返回矩阵的转置
    def transpose(self):
        # 获取矩阵的行数和列数
        rows, cols = self.shape
        # 如果矩阵的行数不为零，调用ddm_transpose函数获取矩阵的转置
        if rows:
            ddmT = ddm_transpose(self)
        else:
            # 如果矩阵的行数为零，创建一个空的转置矩阵
            ddmT = [[]] * cols
        # 返回转置后的矩阵对象
        return DDM(ddmT, (cols, rows), self.domain)

    # 定义两个矩阵相加的特殊方法
    def __add__(a, b):
        # 如果b不是DDM类型的对象，则返回NotImplemented
        if not isinstance(b, DDM):
            return NotImplemented
        # 调用a对象的add方法进行矩阵加法操作
        return a.add(b)

    # 定义两个矩阵相减的特殊方法
    def __sub__(a, b):
        # 如果b不是DDM类型的对象，则返回NotImplemented
        if not isinstance(b, DDM):
            return NotImplemented
        # 调用a对象的sub方法进行矩阵减法操作
        return a.sub(b)

    # 定义矩阵取负的特殊方法
    def __neg__(a):
        # 调用a对象的neg方法进行矩阵取负操作
        return a.neg()

    # 定义矩阵与标量相乘的特殊方法（左乘）
    def __mul__(a, b):
        # 如果b是a对象的域（domain）中的元素，则调用a对象的mul方法进行左乘操作
        if b in a.domain:
            return a.mul(b)
        else:
            # 否则返回NotImplemented
            return NotImplemented

    # 定义矩阵与标量相乘的特殊方法（右乘）
    def __rmul__(a, b):
        # 如果b是a对象的域（domain）中的元素，则调用a对象的mul方法进行右乘操作
        if b in a.domain:
            return a.mul(b)
        else:
            # 否则返回NotImplemented
            return NotImplemented

    # 定义矩阵乘法的特殊方法
    def __matmul__(a, b):
        # 如果b是DDM类型的对象，则调用a对象的matmul方法进行矩阵乘法操作
        if isinstance(b, DDM):
            return a.matmul(b)
        else:
            # 否则返回NotImplemented
            return NotImplemented

    @classmethod
    # 定义一个类方法，用于检查两个矩阵在进行二元操作时的域和形状是否匹配
    def _check(cls, a, op, b, ashape, bshape):
        # 如果a对象的域（domain）与b对象的域（domain）不相等，则抛出域错误异常
        if a.domain != b.domain:
            msg = "Domain mismatch: %s %s %s" % (a.domain, op, b.domain)
            raise DMDomainError(msg)
        # 如果a对象的形状（shape）与b对象的形状（shape）不相等，则抛出形状错误异常
        if ashape != bshape:
            msg = "Shape mismatch: %s %s %s" % (a.shape, op, b.shape)
            raise DMShapeError(msg)

    # 定义矩阵加法操作方法
    def add(a, b):
        """a + b"""
        # 检查a对象与b对象的域和形状是否匹配
        a._check(a, '+', b, a.shape, b.shape)
        # 复制a对象，然后对其进行与b对象相加的原位操作
        c = a.copy()
        ddm_iadd(c, b)
        # 返回相加后的结果矩阵对象
        return c

    # 定义矩阵减法操作方法
    def sub(a, b):
        """a - b"""
        # 检查a对象与b对象的域和形状是否匹配
        a._check(a, '-', b, a.shape, b.shape)
        # 复制a对象，然后对其进行与b对象相减的原位操作
        c = a.copy()
        ddm_isub(c, b)
        # 返回相减后的结果矩阵对象
        return c

    # 定义矩阵取负操作方法
    def neg(a):
        """-a"""
        # 复制a对象，然后对其进行取负的原位操作
        b = a.copy()
        ddm_ineg(b)
        # 返回取负后的结果矩阵对象
        return b

    # 定义矩阵与标量乘法操作方法（左乘）
    def mul(a, b):
        # 复制a对象，然后对其进行与标量b相乘的原位操作
        c = a.copy()
        ddm_imul(c, b)
        # 返回乘法后的结果矩阵对象
        return c

    # 定义矩阵与标量乘法操作方法（右乘）
    def rmul(a, b):
        # 复制a对象，然后对其进行与标量b相乘的原位操作
        c = a.copy()
        ddm_irmul(c, b)
        # 返回乘法后的结果矩阵对象
        return c

    # 定义矩阵乘法操作方法
    def matmul(a, b):
        """a @ b (matrix product)"""
        # 获取a矩阵的行数和列数，以及b矩阵的列数和行数
        m, o = a.shape
        o2, n = b.shape
        # 检查a对象与b对象的形状是否匹配
        a._check(a, '*', b, o, o2)
        # 创建一个与结果矩阵相同形状的零矩阵
        c = a.zeros((m, n), a.domain)
        # 对结果矩阵进行矩阵乘法的原位操作
        ddm_imatmul(c, a, b)
        # 返回矩阵乘法后的结果矩阵对象
        return c

    # 定义矩阵逐元素乘法操作方法
    def mul_elementwise(a, b):
        # 断言a对象与b对象的形状相同
        assert a.shape == b.shape
        # 断言a对象与b对象的域相同
        assert a.domain == b.domain
        # 对两个矩阵进行逐元素乘法操作，生成结果矩阵
        c = [[aij * bij for aij, bij in zip(ai, bi)] for ai, bi in zip(a, b)]
        # 返回逐元素乘法后的结果矩阵对象
        return DDM(c, a.shape, a.domain)
    def hstack(A, *B):
        """Horizontally stacks :py:class:`~.DDM` matrices.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM

        >>> A = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DDM([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
        >>> A.hstack(B)
        [[1, 2, 5, 6], [3, 4, 7, 8]]

        >>> C = DDM([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)
        >>> A.hstack(B, C)
        [[1, 2, 5, 6, 9, 10], [3, 4, 7, 8, 11, 12]]
        """
        Anew = list(A.copy())  # 复制 A 的内容到新列表 Anew
        rows, cols = A.shape  # 获取 A 的行数和列数
        domain = A.domain  # 获取 A 的域

        for Bk in B:  # 遍历参数 B 中的每个 DDM 对象 Bk
            Bkrows, Bkcols = Bk.shape  # 获取 Bk 的行数和列数
            assert Bkrows == rows  # 断言 Bk 的行数与 A 的行数相同
            assert Bk.domain == domain  # 断言 Bk 的域与 A 的域相同

            cols += Bkcols  # 更新列数，增加 Bk 的列数

            for i, Bki in enumerate(Bk):  # 遍历 Bk 的每一行 Bki
                Anew[i].extend(Bki)  # 将 Bki 的内容扩展到 Anew 的对应行中

        return DDM(Anew, (rows, cols), A.domain)  # 返回一个新的 DDM 对象，表示水平堆叠后的结果

    def vstack(A, *B):
        """Vertically stacks :py:class:`~.DDM` matrices.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM

        >>> A = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DDM([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
        >>> A.vstack(B)
        [[1, 2], [3, 4], [5, 6], [7, 8]]

        >>> C = DDM([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)
        >>> A.vstack(B, C)
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
        """
        Anew = list(A.copy())  # 复制 A 的内容到新列表 Anew
        rows, cols = A.shape  # 获取 A 的行数和列数
        domain = A.domain  # 获取 A 的域

        for Bk in B:  # 遍历参数 B 中的每个 DDM 对象 Bk
            Bkrows, Bkcols = Bk.shape  # 获取 Bk 的行数和列数
            assert Bkcols == cols  # 断言 Bk 的列数与 A 的列数相同
            assert Bk.domain == domain  # 断言 Bk 的域与 A 的域相同

            rows += Bkrows  # 更新行数，增加 Bk 的行数

            Anew.extend(Bk.copy())  # 将 Bk 的内容复制并扩展到 Anew 中

        return DDM(Anew, (rows, cols), A.domain)  # 返回一个新的 DDM 对象，表示垂直堆叠后的结果

    def applyfunc(self, func, domain):
        elements = [list(map(func, row)) for row in self]  # 对 self 中每个元素应用 func 函数
        return DDM(elements, self.shape, domain)  # 返回一个新的 DDM 对象，表示应用函数后的结果

    def nnz(a):
        """Number of non-zero entries in :py:class:`~.DDM` matrix.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nnz
        """
        return sum(sum(map(bool, row)) for row in a)  # 计算矩阵中非零元素的个数

    def scc(a):
        """Strongly connected components of a square matrix *a*.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM
        >>> A = DDM([[ZZ(1), ZZ(0)], [ZZ(0), ZZ(1)]], (2, 2), ZZ)
        >>> A.scc()
        [[0], [1]]

        See also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.scc

        """
        return a.to_sdm().scc()  # 返回矩阵 a 转换为 sdm 后的强连通分量列表

    @classmethod
    def diag(cls, values, domain):
        """Returns a square diagonal matrix with *values* on the diagonal.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM
        >>> DDM.diag([ZZ(1), ZZ(2), ZZ(3)], ZZ)
        [[1, 0, 0], [0, 2, 0], [0, 0, 3]]

        See also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.diag
        """
        # 使用给定的值和域生成一个对角矩阵，并将其转换为 DDM（密集对角矩阵）格式
        return SDM.diag(values, domain).to_ddm()

    def rref(a):
        """Reduced-row echelon form of a and list of pivots.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.rref
            Higher level interface to this function.
        sympy.polys.matrices.dense.ddm_irref
            The underlying algorithm.
        """
        # 复制矩阵 a
        b = a.copy()
        # 获取矩阵 a 的定义域
        K = a.domain
        # 根据定义域类型判断是否进行部分主元素选择
        partial_pivot = K.is_RealField or K.is_ComplexField
        # 调用 ddm_irref 函数计算矩阵的行阶梯形式及主元列表
        pivots = ddm_irref(b, _partial_pivot=partial_pivot)
        # 返回变换后的矩阵 b 和主元列表
        return b, pivots

    def rref_den(a):
        """Reduced-row echelon form of a with denominator and list of pivots

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den
            Higher level interface to this function.
        sympy.polys.matrices.dense.ddm_irref_den
            The underlying algorithm.
        """
        # 复制矩阵 a
        b = a.copy()
        # 获取矩阵 a 的定义域
        K = a.domain
        # 调用 ddm_irref_den 函数计算矩阵的分母形式行阶梯形式及主元列表
        denom, pivots = ddm_irref_den(b, K)
        # 返回变换后的矩阵 b、分母形式及主元列表
        return b, denom, pivots

    def nullspace(a):
        """Returns a basis for the nullspace of a.

        The domain of the matrix must be a field.

        See Also
        ========

        rref
        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
        """
        # 对矩阵 a 进行行阶梯形式变换，并获取其零空间的基
        rref, pivots = a.rref()
        # 根据行阶梯形式及主元列表计算零空间的基
        return rref.nullspace_from_rref(pivots)
    def nullspace_from_rref(a, pivots=None):
        """Compute the nullspace of a matrix from its rref.

        The domain of the matrix can be any domain.

        Returns a tuple (basis, nonpivots).

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
            The higher level interface to this function.
        """
        # 获取矩阵的行数和列数
        m, n = a.shape
        # 获取矩阵的域
        K = a.domain

        # 如果未提供主元列表，则从矩阵的行简化阶梯形式中找到主元
        if pivots is None:
            pivots = []
            last_pivot = -1
            for i in range(m):
                ai = a[i]
                for j in range(last_pivot+1, n):
                    if ai[j]:
                        last_pivot = j
                        pivots.append(j)
                        break

        # 如果没有找到主元，返回基础矩阵和所有非主元索引的列表
        if not pivots:
            return (a.eye(n, K), list(range(n)))

        # 获取第一个主元的值
        pivot_val = a[0][pivots[0]]

        basis = []
        nonpivots = []
        # 构建基础矩阵和非主元索引的列表
        for i in range(n):
            if i in pivots:
                continue
            nonpivots.append(i)
            vec = [pivot_val if i == j else K.zero for j in range(n)]
            for ii, jj in enumerate(pivots):
                vec[jj] -= a[ii][i]
            basis.append(vec)

        # 使用基础矩阵和其形状创建 DomainMatrix 对象
        basis_ddm = DDM(basis, (len(basis), n), K)

        return (basis_ddm, nonpivots)

    def particular(a):
        # 将矩阵转换为分解形式，然后返回其特解的 DomainMatrix 对象
        return a.to_sdm().particular().to_ddm()

    def det(a):
        """Determinant of a"""
        # 获取矩阵的行数和列数
        m, n = a.shape
        # 如果不是方阵，抛出异常
        if m != n:
            raise DMNonSquareMatrixError("Determinant of non-square matrix")
        # 复制矩阵，获取其域
        b = a.copy()
        K = b.domain
        # 计算矩阵的行列式
        deta = ddm_idet(b, K)
        return deta

    def inv(a):
        """Inverse of a"""
        # 获取矩阵的行数和列数
        m, n = a.shape
        # 如果不是方阵，抛出异常
        if m != n:
            raise DMNonSquareMatrixError("Determinant of non-square matrix")
        # 复制矩阵，获取其域
        ainv = a.copy()
        K = a.domain
        # 计算矩阵的逆
        ddm_iinv(ainv, a, K)
        return ainv

    def lu(a):
        """L, U decomposition of a"""
        # 获取矩阵的行数和列数
        m, n = a.shape
        # 获取矩阵的域
        K = a.domain

        # 复制矩阵作为 U
        U = a.copy()
        # 创建单位矩阵作为 L
        L = a.eye(m, K)
        # 执行 LU 分解，并返回置换列表
        swaps = ddm_ilu_split(L, U, K)

        return L, U, swaps

    def lu_solve(a, b):
        """x where a*x = b"""
        # 获取矩阵 a 和向量 b 的形状信息
        m, n = a.shape
        m2, o = b.shape
        # 检查矩阵和向量的兼容性
        a._check(a, 'lu_solve', b, m, m2)
        # 如果域不是域，则抛出异常
        if not a.domain.is_Field:
            raise DMDomainError("lu_solve requires a field")

        # 进行 LU 分解
        L, U, swaps = a.lu()
        # 创建结果向量 x，并解方程组 LUx = b
        x = a.zeros((n, o), a.domain)
        ddm_ilu_solve(x, L, U, swaps, b)
        return x

    def charpoly(a):
        """Coefficients of characteristic polynomial of a"""
        # 获取矩阵的域和形状信息
        K = a.domain
        m, n = a.shape
        # 如果不是方阵，抛出异常
        if m != n:
            raise DMNonSquareMatrixError("Charpoly of non-square matrix")
        # 计算特征多项式的向量
        vec = ddm_berk(a, K)
        # 提取特征多项式的系数
        coeffs = [vec[i][0] for i in range(n+1)]
        return coeffs
    # 判断该矩阵是否为零矩阵，即所有元素是否均为零
    def is_zero_matrix(self):
        # 获取零元素
        zero = self.domain.zero
        # 检查矩阵中所有元素是否都等于零
        return all(Mij == zero for Mij in self.flatiter())

    # 判断该矩阵是否为上三角矩阵，即除了对角线及其以下的元素外，其余元素是否均为零
    def is_upper(self):
        # 获取零元素
        zero = self.domain.zero
        # 检查对角线以上的元素是否都等于零
        return all(Mij == zero for i, Mi in enumerate(self) for Mij in Mi[:i])

    # 判断该矩阵是否为下三角矩阵，即除了对角线及其以上的元素外，其余元素是否均为零
    def is_lower(self):
        # 获取零元素
        zero = self.domain.zero
        # 检查对角线以下的元素是否都等于零
        return all(Mij == zero for i, Mi in enumerate(self) for Mij in Mi[i+1:])

    # 判断该矩阵是否为对角矩阵，即同时满足上三角和下三角条件
    def is_diagonal(self):
        return self.is_upper() and self.is_lower()

    # 返回矩阵对角线上的元素构成的列表
    def diagonal(self):
        m, n = self.shape
        # 返回从矩阵对角线上提取的元素列表
        return [self[i][i] for i in range(min(m, n))]

    # 使用 LLL 算法对矩阵进行 LLL 变换
    def lll(A, delta=QQ(3, 4)):
        return ddm_lll(A, delta=delta)

    # 使用 LLL 算法对矩阵进行 LLL 变换，并返回变换后的矩阵
    def lll_transform(A, delta=QQ(3, 4)):
        return ddm_lll_transform(A, delta=delta)
# 从当前目录下的模块中导入SDM类
from .sdm import SDM
# 从当前目录下的模块中导入DFM类
from .dfm import DFM
```
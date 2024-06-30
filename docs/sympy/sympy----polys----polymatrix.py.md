# `D:\src\scipysrc\sympy\sympy\polys\polymatrix.py`

```
# 从 sympy.core.expr 模块导入 Expr 类，表示表达式
# 从 sympy.core.symbol 模块导入 Dummy 类，表示符号
# 从 sympy.core.sympify 模块导入 _sympify 函数，用于将对象转换为 sympy 对象

from sympy.core.expr import Expr
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify

# 从 sympy.polys.polyerrors 模块导入 CoercionFailed 异常类，表示多项式强制类型转换失败
# 从 sympy.polys.polytools 模块导入 Poly 和 parallel_poly_from_expr 函数
# Poly 表示多项式类，parallel_poly_from_expr 用于从表达式生成多项式
# 从 sympy.polys.domains 模块导入 QQ，表示有理数域

from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import Poly, parallel_poly_from_expr
from sympy.polys.domains import QQ

# 从 sympy.polys.matrices 模块导入 DomainMatrix 类，表示多项式矩阵
# 从 sympy.polys.matrices.domainscalar 模块导入 DomainScalar 类，表示多项式矩阵的标量

from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.domainscalar import DomainScalar

class MutablePolyDenseMatrix:
    """
    A mutable matrix of objects from poly module or to operate with them.

    Examples
    ========

    >>> from sympy.polys.polymatrix import PolyMatrix
    >>> from sympy import Symbol, Poly
    >>> x = Symbol('x')
    >>> pm1 = PolyMatrix([[Poly(x**2, x), Poly(-x, x)], [Poly(x**3, x), Poly(-1 + x, x)]])
    >>> v1 = PolyMatrix([[1, 0], [-1, 0]], x)
    >>> pm1*v1
    PolyMatrix([
    [    x**2 + x, 0],
    [x**3 - x + 1, 0]], ring=QQ[x])

    >>> pm1.ring
    ZZ[x]

    >>> v1*pm1
    PolyMatrix([
    [ x**2, -x],
    [-x**2,  x]], ring=QQ[x])

    >>> pm2 = PolyMatrix([[Poly(x**2, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(1, x, domain='QQ'), \
            Poly(x**3, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(-x**3, x, domain='QQ')]])
    >>> v2 = PolyMatrix([1, 0, 0, 0, 0, 0], x)
    >>> v2.ring
    QQ[x]
    >>> pm2*v2
    PolyMatrix([[x**2]], ring=QQ[x])

    """

    def __new__(cls, *args, ring=None):

        # 如果没有参数传入
        if not args:
            # 抛出类型错误，需要为空 PolyMatrix 指定环
            if ring is None:
                raise TypeError("The ring needs to be specified for an empty PolyMatrix")
            rows, cols, items, gens = 0, 0, [], ()
        # 如果第一个参数是列表
        elif isinstance(args[0], list):
            elements, gens = args[0], args[1:]
            # 如果 elements 为空列表
            if not elements:
                rows, cols, items = 0, 0, []
            # 如果 elements 的第一个元素是列表或元组
            elif isinstance(elements[0], (list, tuple)):
                rows, cols = len(elements), len(elements[0])
                # 展开 elements 中的元素，存入 items
                items = [e for row in elements for e in row]
            else:
                rows, cols = len(elements), 1
                items = elements
        # 如果前三个参数分别为整数、整数和列表
        elif [type(a) for a in args[:3]] == [int, int, list]:
            rows, cols, items, gens = args[0], args[1], args[2], args[3:]
        # 如果前三个参数分别为整数、整数和函数对象
        elif [type(a) for a in args[:3]] == [int, int, type(lambda: 0)]:
            rows, cols, func, gens = args[0], args[1], args[2], args[3:]
            # 使用 func 函数生成 items
            items = [func(i, j) for i in range(rows) for j in range(cols)]
        else:
            # 如果参数类型不符合上述条件，抛出类型错误
            raise TypeError("Invalid arguments")

        # 如果 gens 的长度为 1，并且 gens[0] 是元组
        if len(gens) == 1 and isinstance(gens[0], tuple):
            gens = gens[0]
            # 将 gens 转换为元组形式 (x, y)

        # 调用 from_list 方法创建类的实例
        return cls.from_list(rows, cols, items, gens, ring)

    @classmethod
    def from_list(cls, rows, cols, items, gens, ring):
        # 将 items 列表中的每个元素转换为表达式对象（Expr）或多项式对象（Poly）
        items = [_sympify(item) for item in items]
        # 如果 items 不为空且所有元素均为多项式对象（Poly），则 polys 为 True
        if items and all(isinstance(item, Poly) for item in items):
            polys = True
        else:
            polys = False

        # 确定多项式的环境（ring）
        if ring is not None:
            # 如果 ring 是字符串，解析其为多项式的定义域字符串，例如 'QQ[x]'
            if isinstance(ring, str):
                ring = Poly(0, Dummy(), domain=ring).domain
        elif polys:
            # 如果 ring 为 None 且 items 中都是多项式，从第一个多项式中推断环境
            p = items[0]
            for p2 in items[1:]:
                p, _ = p.unify(p2)
            ring = p.domain[p.gens]
        else:
            # 如果 items 包含表达式和生成器，使用并行处理将其转换为多项式，并确定环境
            items, info = parallel_poly_from_expr(items, gens, field=True)
            ring = info['domain'][info['gens']]
            polys = True

        # 如果所有元素都是多项式，进行高效的转换
        if polys:
            # 创建一个零多项式对象 p_ring，并根据环境 ring 的定义从 items 转换成列表形式
            p_ring = Poly(0, ring.symbols, domain=ring.domain)
            to_ring = ring.ring.from_list
            convert_poly = lambda p: to_ring(p.unify(p_ring)[0].rep.to_list())
            elements = [convert_poly(p) for p in items]
        else:
            # 如果不是所有元素都是多项式，将表达式转换为环境元素
            convert_expr = ring.from_sympy
            elements = [convert_expr(e.as_expr()) for e in items]

        # 将元素列表 elements 转换为二维列表 elements_lol，用于构造 DomainMatrix
        elements_lol = [[elements[i*cols + j] for j in range(cols)] for i in range(rows)]
        # 使用 elements_lol 和 ring 构造 DomainMatrix 对象 dm
        dm = DomainMatrix(elements_lol, (rows, cols), ring)
        # 从 DomainMatrix 对象 dm 构造当前类的实例并返回
        return cls.from_dm(dm)

    @classmethod
    def from_dm(cls, dm):
        # 创建当前类的实例 obj
        obj = super().__new__(cls)
        # 将 DomainMatrix dm 转换为稀疏表示
        dm = dm.to_sparse()
        # 获取 dm 的环境 R，并将其赋给 obj 的属性 _dm、ring 和 domain
        R = dm.domain
        obj._dm = dm
        obj.ring = R
        obj.domain = R.domain
        obj.gens = R.symbols
        # 返回创建的实例 obj
        return obj

    def to_Matrix(self):
        # 将对象的 _dm 属性转换为 Matrix 对象并返回
        return self._dm.to_Matrix()

    @classmethod
    def from_Matrix(cls, other, *gens, ring=None):
        # 使用给定的 Matrix 对象 other、行数和列数以及其扁平化的元素构造当前类的实例
        return cls(*other.shape, other.flat(), *gens, ring=ring)

    def set_gens(self, gens):
        # 使用当前对象的 to_Matrix 方法创建一个新对象，并设定新的生成器 gens
        return self.from_Matrix(self.to_Matrix(), gens)

    def __repr__(self):
        # 如果矩阵的行数和列数大于零，返回对象的字符串表示形式 'Poly' + Matrix 的表示 + 环境信息
        if self.rows * self.cols:
            return 'Poly' + repr(self.to_Matrix())[:-1] + f', ring={self.ring})'
        else:
            # 如果矩阵为空，返回空 PolyMatrix 的字符串表示形式，包含环境信息
            return f'PolyMatrix({self.rows}, {self.cols}, [], ring={self.ring})'

    @property
    def shape(self):
        # 返回对象的矩阵形状（行数，列数）
        return self._dm.shape

    @property
    def rows(self):
        # 返回对象的行数
        return self.shape[0]

    @property
    def cols(self):
        # 返回对象的列数
        return self.shape[1]

    def __len__(self):
        # 返回对象中元素的总数，即行数乘以列数
        return self.rows * self.cols
    def __getitem__(self, key):
        # 定义内部函数，将给定对象转换为多项式对象
        def to_poly(v):
            ground = self._dm.domain.domain  # 获取多项式的底层域
            gens = self._dm.domain.symbols   # 获取多项式的生成元
            return Poly(v.to_dict(), gens, domain=ground)  # 根据给定的值创建多项式对象

        dm = self._dm  # 获取当前对象的域矩阵

        if isinstance(key, slice):
            items = dm.flat()[key]  # 获取切片范围内的项
            return [to_poly(item) for item in items]  # 返回切片范围内所有项的多项式对象列表
        elif isinstance(key, int):
            i, j = divmod(key, self.cols)  # 将索引键转换为行列索引
            e = dm[i,j]  # 获取域矩阵中指定位置的元素
            return to_poly(e.element)  # 返回该元素的多项式表示

        i, j = key
        if isinstance(i, int) and isinstance(j, int):
            return to_poly(dm[i, j].element)  # 返回指定位置元素的多项式表示
        else:
            return self.from_dm(dm[i, j])  # 返回从域矩阵中提取的对象

    def __eq__(self, other):
        if not isinstance(self, type(other)):
            return NotImplemented  # 如果类型不匹配，返回未实现错误
        return self._dm == other._dm  # 比较两个对象的域矩阵是否相等

    def __add__(self, other):
        if isinstance(other, type(self)):
            return self.from_dm(self._dm + other._dm)  # 返回两个对象的域矩阵相加的结果
        return NotImplemented  # 如果类型不匹配，返回未实现错误

    def __sub__(self, other):
        if isinstance(other, type(self)):
            return self.from_dm(self._dm - other._dm)  # 返回两个对象的域矩阵相减的结果
        return NotImplemented  # 如果类型不匹配，返回未实现错误

    def __mul__(self, other):
        if isinstance(other, type(self)):
            return self.from_dm(self._dm * other._dm)  # 返回两个对象的域矩阵相乘的结果
        elif isinstance(other, int):
            other = _sympify(other)  # 将整数转换为符号表达式
        if isinstance(other, Expr):
            Kx = self.ring  # 获取环对象
            try:
                other_ds = DomainScalar(Kx.from_sympy(other), Kx)  # 尝试将符号表达式转换为域标量对象
            except (CoercionFailed, ValueError):
                other_ds = DomainScalar.from_sympy(other)  # 若转换失败则直接从符号表达式创建域标量对象
            return self.from_dm(self._dm * other_ds)  # 返回对象与域标量对象相乘的结果
        return NotImplemented  # 如果类型不匹配，返回未实现错误

    def __rmul__(self, other):
        if isinstance(other, int):
            other = _sympify(other)  # 将整数转换为符号表达式
        if isinstance(other, Expr):
            other_ds = DomainScalar.from_sympy(other)  # 将符号表达式转换为域标量对象
            return self.from_dm(other_ds * self._dm)  # 返回域标量对象与对象相乘的结果
        return NotImplemented  # 如果类型不匹配，返回未实现错误

    def __truediv__(self, other):
        if isinstance(other, Poly):
            other = other.as_expr()  # 如果参数是多项式，则将其转换为表达式
        elif isinstance(other, int):
            other = _sympify(other)  # 将整数转换为符号表达式
        if not isinstance(other, Expr):
            return NotImplemented  # 如果参数不是表达式，则返回未实现错误

        other = self.domain.from_sympy(other)  # 将符号表达式转换为当前对象的域
        inverse = self.ring.convert_from(1/other, self.domain)  # 计算给定参数的倒数，并转换为当前对象的域
        inverse = DomainScalar(inverse, self.ring)  # 创建域标量对象
        dm = self._dm * inverse  # 将当前对象的域矩阵乘以倒数
        return self.from_dm(dm)  # 返回乘积的结果

    def __neg__(self):
        return self.from_dm(-self._dm)  # 返回当前对象的域矩阵的相反数

    def transpose(self):
        return self.from_dm(self._dm.transpose())  # 返回当前对象的域矩阵的转置结果

    def row_join(self, other):
        dm = DomainMatrix.hstack(self._dm, other._dm)  # 水平连接当前对象和另一个对象的域矩阵
        return self.from_dm(dm)  # 返回连接结果的对象

    def col_join(self, other):
        dm = DomainMatrix.vstack(self._dm, other._dm)  # 垂直连接当前对象和另一个对象的域矩阵
        return self.from_dm(dm)  # 返回连接结果的对象

    def applyfunc(self, func):
        M = self.to_Matrix().applyfunc(func)  # 对当前对象转换为矩阵，并应用给定函数
        return self.from_Matrix(M, self.gens)  # 返回应用函数后的对象

    @classmethod
    def eye(cls, n, gens):
        # 创建一个单位矩阵 PolyMatrix 类的类方法
        return cls.from_dm(DomainMatrix.eye(n, QQ[gens]))

    @classmethod
    def zeros(cls, m, n, gens):
        # 创建一个全零矩阵 PolyMatrix 类的类方法
        return cls.from_dm(DomainMatrix.zeros((m, n), QQ[gens]))

    def rref(self, simplify='ignore', normalize_last='ignore'):
        # 如果域是 K[x]，则在域 K 中计算矩阵的行简化阶梯形式（RREF）
        if not (self.domain.is_Field and all(p.is_ground for p in self)):
            raise ValueError("PolyMatrix rref is only for ground field elements")
        # 获取内部的 DomainMatrix 对象
        dm = self._dm
        # 转换为域 K 中的 DomainMatrix 对象
        dm_ground = dm.convert_to(dm.domain.domain)
        # 计算域 K 中的 RREF 和主元素位置
        dm_rref, pivots = dm_ground.rref()
        # 将结果转换回原始域并返回 PolyMatrix 对象
        dm_rref = dm_rref.convert_to(dm.domain)
        return self.from_dm(dm_rref), pivots

    def nullspace(self):
        # 如果域是 K[x]，则在域 K 中计算矩阵的零空间
        if not (self.domain.is_Field and all(p.is_ground for p in self)):
            raise ValueError("PolyMatrix nullspace is only for ground field elements")
        # 获取内部的 DomainMatrix 对象
        dm = self._dm
        K, Kx = self.domain, self.ring
        # 在域 K 中计算零空间，设置 divide_last=True 以优化计算
        dm_null_rows = dm.convert_to(K).nullspace(divide_last=True).convert_to(Kx)
        # 转置并获取零空间的基向量
        dm_null = dm_null_rows.transpose()
        dm_basis = [dm_null[:,i] for i in range(dm_null.shape[1])]
        # 将结果转换为 PolyMatrix 对象并返回
        return [self.from_dm(dmvec) for dmvec in dm_basis]

    def rank(self):
        # 返回矩阵的秩，即列数减去零空间的维数
        return self.cols - len(self.nullspace())
# 定义了三个变量 MutablePolyMatrix、PolyMatrix 和 MutablePolyDenseMatrix，它们都指向同一个对象。
MutablePolyMatrix = PolyMatrix = MutablePolyDenseMatrix
```
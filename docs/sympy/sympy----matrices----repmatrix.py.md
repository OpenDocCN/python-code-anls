# `D:\src\scipysrc\sympy\sympy\matrices\repmatrix.py`

```
from collections import defaultdict

# 从 operator 模块中导入 index 函数，并将其重命名为 index_
from operator import index as index_

# 从 sympy.core.expr 模块中导入 Expr 类
from sympy.core.expr import Expr

# 从 sympy.core.kind 模块中导入 Kind, NumberKind, UndefinedKind 类
from sympy.core.kind import Kind, NumberKind, UndefinedKind

# 从 sympy.core.numbers 模块中导入 Integer, Rational 类
from sympy.core.numbers import Integer, Rational

# 从 sympy.core.sympify 模块中导入 _sympify 函数和 SympifyError 异常
from sympy.core.sympify import _sympify, SympifyError

# 从 sympy.core.singleton 模块中导入 S 单例对象
from sympy.core.singleton import S

# 从 sympy.polys.domains 模块中导入 ZZ, QQ, GF, EXRAW 类
from sympy.polys.domains import ZZ, QQ, GF, EXRAW

# 从 sympy.polys.matrices 模块中导入 DomainMatrix 类
from sympy.polys.matrices import DomainMatrix

# 从 sympy.polys.matrices.exceptions 模块中导入 DMNonInvertibleMatrixError 异常
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError

# 从 sympy.polys.polyerrors 模块中导入 CoercionFailed 异常
from sympy.polys.polyerrors import CoercionFailed

# 从 sympy.utilities.exceptions 模块中导入 sympy_deprecation_warning 函数
from sympy.utilities.exceptions import sympy_deprecation_warning

# 从 sympy.utilities.iterables 模块中导入 is_sequence 函数
from sympy.utilities.iterables import is_sequence

# 从 sympy.utilities.misc 模块中导入 filldedent, as_int 函数
from sympy.utilities.misc import filldedent, as_int

# 从当前包的 exceptions 模块中导入 ShapeError, NonSquareMatrixError, NonInvertibleMatrixError 异常
from .exceptions import ShapeError, NonSquareMatrixError, NonInvertibleMatrixError

# 从当前包的 matrixbase 模块中导入 classof, MatrixBase 类
from .matrixbase import classof, MatrixBase

# 从当前包的 kind 模块中导入 MatrixKind 类
from .kind import MatrixKind


class RepMatrix(MatrixBase):
    """Matrix implementation based on DomainMatrix as an internal representation.

    The RepMatrix class is a superclass for Matrix, ImmutableMatrix,
    SparseMatrix and ImmutableSparseMatrix which are the main usable matrix
    classes in SymPy. Most methods on this class are simply forwarded to
    DomainMatrix.
    """

    #
    # MatrixBase is the common superclass for all of the usable explicit matrix
    # classes in SymPy. The idea is that MatrixBase is an abstract class though
    # and that subclasses will implement the lower-level methods.
    #
    # RepMatrix is a subclass of MatrixBase that uses DomainMatrix as an
    # internal representation and delegates lower-level methods to
    # DomainMatrix. All of SymPy's standard explicit matrix classes subclass
    # RepMatrix and so use DomainMatrix internally.
    #
    # A RepMatrix uses an internal DomainMatrix with the domain set to ZZ, QQ
    # or EXRAW. The EXRAW domain is equivalent to the previous implementation
    # of Matrix that used Expr for the elements. The ZZ and QQ domains are used
    # when applicable just because they are compatible with the previous
    # implementation but are much more efficient. Other domains such as QQ[x]
    # are not used because they differ from Expr in some way (e.g. automatic
    # expansion of powers and products).
    #

    # _rep 属性指向一个 DomainMatrix 对象
    _rep: DomainMatrix

    def __eq__(self, other):
        # 如果 other 不是 RepMatrix 的实例，则尝试将其转换为表达式
        if not isinstance(other, RepMatrix):
            try:
                other = _sympify(other)
            except SympifyError:
                return NotImplemented
            # 如果转换后仍不是 RepMatrix 的实例，则返回 NotImplemente
            if not isinstance(other, RepMatrix):
                return NotImplemented

        # 调用 DomainMatrix 的 unify_eq 方法比较两个 RepMatrix 对象的 _rep 属性
        return self._rep.unify_eq(other._rep)
    # 定义一个方法，将当前对象转换为 DomainMatrix 类的实例
    def to_DM(self, domain=None, **kwargs):
        """Convert to a :class:`~.DomainMatrix`.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2], [3, 4]])
        >>> M.to_DM()
        DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}, (2, 2), ZZ)

        The :meth:`DomainMatrix.to_Matrix` method can be used to convert back:

        >>> M.to_DM().to_Matrix() == M
        True

        The domain can be given explicitly or otherwise it will be chosen by
        :func:`construct_domain`. Any keyword arguments (besides ``domain``)
        are passed to :func:`construct_domain`:

        >>> from sympy import QQ, symbols
        >>> x = symbols('x')
        >>> M = Matrix([[x, 1], [1, x]])
        >>> M
        Matrix([
        [x, 1],
        [1, x]])
        >>> M.to_DM().domain
        ZZ[x]
        >>> M.to_DM(field=True).domain
        ZZ(x)
        >>> M.to_DM(domain=QQ[x]).domain
        QQ[x]

        See Also
        ========

        DomainMatrix
        DomainMatrix.to_Matrix
        DomainMatrix.convert_to
        DomainMatrix.choose_domain
        construct_domain
        """
        # 如果给定了 domain 参数，则调用 _rep.convert_to 方法进行转换
        if domain is not None:
            if kwargs:
                raise TypeError("Options cannot be used with domain parameter")
            return self._rep.convert_to(domain)

        # 否则，获取当前对象的 _rep 属性作为 rep
        rep = self._rep
        dom = rep.domain

        # 如果没有额外的 kwargs 参数，并且当前的 domain 是 ZZ 或 QQ，则可能可以避免调用 construct_domain 或执行任何转换
        if not kwargs:
            if dom.is_ZZ:
                return rep.copy()
            elif dom.is_QQ:
                # 所有元素可能是整数，尝试转换为 ZZ
                try:
                    return rep.convert_to(ZZ)
                except CoercionFailed:
                    pass
                return rep.copy()

        # 否则，让 construct_domain 根据 kwargs 选择一个合适的 domain
        rep_dom = rep.choose_domain(**kwargs)

        # XXX: 应该有一个选项在 construct_domain 中选择使用 EXRAW 而不是 EX。至少转换为 EX 并不会触发 EX.simplify，这正是我们想要的，
        # 但这可能应该被视为 EX 中的一个 bug。也许这应该在 DomainMatrix.choose_domain 中处理，而不是在这里处理...
        if rep_dom.domain.is_EX:
            rep_dom = rep_dom.convert_to(EXRAW)

        # 返回转换后的 DomainMatrix 对象
        return rep_dom
    def _unify_element_sympy(cls, rep, element):
        # 获取当前表示的域
        domain = rep.domain
        # 将元素转换为 SymPy 对象
        element = _sympify(element)

        # 如果域不是 EXRAW
        if domain != EXRAW:
            # 域只能是 ZZ, QQ 或 EXRAW
            if element.is_Integer:
                new_domain = domain
            elif element.is_Rational:
                new_domain = QQ
            else:
                new_domain = EXRAW

            # XXX: 这会将矩阵中所有元素的域都转换，可能会很慢。比如，如果 __setitem__ 将一个元素改为不适合当前域的值时
            if new_domain != domain:
                # 转换表示为新的域
                rep = rep.convert_to(new_domain)
                domain = new_domain

            # 如果域不是 EXRAW，则将元素转换为新的域类型
            if domain != EXRAW:
                element = new_domain.from_sympy(element)

        # 如果域是 EXRAW 并且元素不是 Expr 类型，则发出警告
        if domain == EXRAW and not isinstance(element, Expr):
            sympy_deprecation_warning(
                """
                非 Expr 对象在 Matrix 中已不推荐使用。Matrix 表示数学矩阵。若要表示非数值实体的容器，请使用列表、TableForm、NumPy 数组或其他数据结构。
                """,
                deprecated_since_version="1.9",
                active_deprecations_target="deprecated-non-expr-in-matrix",
                stacklevel=4,
            )

        return rep, element

    @classmethod
    def _dod_to_DomainMatrix(cls, rows, cols, dod, types):
        # 如果 types 中存在非 Expr 类型，则发出警告
        if not all(issubclass(typ, Expr) for typ in types):
            sympy_deprecation_warning(
                """
                非 Expr 对象在 Matrix 中已不推荐使用。Matrix 表示数学矩阵。若要表示非数值实体的容器，请使用列表、TableForm、NumPy 数组或其他数据结构。
                """,
                deprecated_since_version="1.9",
                active_deprecations_target="deprecated-non-expr-in-matrix",
                stacklevel=6,
            )

        # 创建一个 DomainMatrix 对象，使用 EXRAW 域
        rep = DomainMatrix(dod, (rows, cols), EXRAW)

        # 如果 types 中都是 Rational 类型，则根据情况转换为 ZZ 或 QQ 域
        if all(issubclass(typ, Rational) for typ in types):
            if all(issubclass(typ, Integer) for typ in types):
                rep = rep.convert_to(ZZ)
            else:
                rep = rep.convert_to(QQ)

        return rep

    @classmethod
    def _flat_list_to_DomainMatrix(cls, rows, cols, flat_list):
        # 将扁平列表转换为元素的字典形式
        elements_dod = defaultdict(dict)
        for n, element in enumerate(flat_list):
            if element != 0:
                i, j = divmod(n, cols)
                elements_dod[i][j] = element

        # 获取 flat_list 中元素的类型集合
        types = set(map(type, flat_list))

        # 使用 _dod_to_DomainMatrix 方法将字典形式的元素转换为 DomainMatrix 对象
        rep = cls._dod_to_DomainMatrix(rows, cols, elements_dod, types)
        return rep

    @classmethod
    # 将稀疏矩阵表示转换为稠密域矩阵表示，并返回结果
    def _smat_to_DomainMatrix(cls, rows, cols, smat):
        # 使用 defaultdict 创建一个空的嵌套字典 elements_dod，用于存储非零元素
        elements_dod = defaultdict(dict)
        # 遍历稀疏矩阵 smat 的每个元素 (i, j)，将非零元素加入到 elements_dod 中对应的位置
        for (i, j), element in smat.items():
            if element != 0:
                elements_dod[i][j] = element

        # 获取 smat 的值集合的类型
        types = set(map(type, smat.values()))

        # 调用 _dod_to_DomainMatrix 方法，将 elements_dod 转换为 DomainMatrix 表示，并返回结果
        rep = cls._dod_to_DomainMatrix(rows, cols, elements_dod, types)
        return rep

    # 返回 self 对应的 SymPy 表示的扁平列表
    def flat(self):
        return self._rep.to_sympy().to_list_flat()

    # 返回 self 对应的 SymPy 表示的列表
    def _eval_tolist(self):
        return self._rep.to_sympy().to_list()

    # 返回 self 对应的 SymPy 表示的 dok (Dictionary of Keys) 格式
    def _eval_todok(self):
        return self._rep.to_sympy().to_dok()

    # 使用 dok (Dictionary of Keys) 表示初始化一个 DomainMatrix 实例，并返回结果
    @classmethod
    def _eval_from_dok(cls, rows, cols, dok):
        return cls._fromrep(cls._smat_to_DomainMatrix(rows, cols, dok))

    # 返回 self 对应的值的列表
    def _eval_values(self):
        return list(self._eval_iter_values())

    # 返回一个迭代器，迭代器包含 self 对应的值
    def _eval_iter_values(self):
        rep = self._rep
        K = rep.domain
        values = rep.iter_values()
        # 如果 K 不是 EXRAW 类型，则将每个值映射为 SymPy 表示
        if not K.is_EXRAW:
            values = map(K.to_sympy, values)
        return values

    # 返回一个迭代器，迭代器包含 self 对应的键值对
    def _eval_iter_items(self):
        rep = self._rep
        K = rep.domain
        to_sympy = K.to_sympy
        items = rep.iter_items()
        # 如果 K 不是 EXRAW 类型，则将每个值映射为 SymPy 表示
        if not K.is_EXRAW:
            items = ((i, to_sympy(v)) for i, v in items)
        return items

    # 返回 self 的副本
    def copy(self):
        return self._fromrep(self._rep.copy())

    # 返回 self 的元素类型的 MatrixKind
    @property
    def kind(self) -> MatrixKind:
        domain = self._rep.domain
        element_kind: Kind
        # 根据 domain 的类型确定 element_kind
        if domain in (ZZ, QQ):
            element_kind = NumberKind
        elif domain == EXRAW:
            kinds = {e.kind for e in self.values()}
            if len(kinds) == 1:
                [element_kind] = kinds
            else:
                element_kind = UndefinedKind
        else: # pragma: no cover
            raise RuntimeError("Domain should only be ZZ, QQ or EXRAW")
        return MatrixKind(element_kind)

    # 检查矩阵是否包含指定模式的元素，并返回布尔值
    def _eval_has(self, *patterns):
        # 获取 dok 格式的表示
        dok = self.todok()
        # 如果 dok 的长度不等于矩阵行数乘以列数，则矩阵包含零元素
        zhas = len(dok) != self.rows*self.cols
        # 如果 zhas 为 True，则判断 S.Zero 是否包含指定模式
        return zhas or any(value.has(*patterns) for value in dok.values())

    # 检查矩阵是否为单位矩阵，并返回布尔值
    def _eval_is_Identity(self):
        # 检查对角线上的元素是否为 1
        if not all(self[i, i] == 1 for i in range(self.rows)):
            return False
        # 检查矩阵是否只有非零元素的个数等于行数
        return len(self.todok()) == self.rows

    # 检查矩阵是否为对称矩阵，并返回布尔值
    def _eval_is_symmetric(self, simpfunc):
        # 计算 self 与其转置矩阵的差，并对每个元素应用函数 simpfunc
        diff = (self - self.T).applyfunc(simpfunc)
        # 如果差的值为 0，则矩阵为对称矩阵
        return len(diff.values()) == 0

    # 返回 self 的转置矩阵的 SparseMatrix 表示
    def _eval_transpose(self):
        """返回这个 SparseMatrix 的转置 SparseMatrix。

        示例
        ========

        >>> from sympy import SparseMatrix
        >>> a = SparseMatrix(((1, 2), (3, 4)))
        >>> a
        Matrix([
        [1, 2],
        [3, 4]])
        >>> a.T
        Matrix([
        [1, 3],
        [2, 4]])
        """
        return self._fromrep(self._rep.transpose())
    # 返回列连接后的结果矩阵
    def _eval_col_join(self, other):
        return self._fromrep(self._rep.vstack(other._rep))

    # 返回行连接后的结果矩阵
    def _eval_row_join(self, other):
        return self._fromrep(self._rep.hstack(other._rep))

    # 根据给定的行和列列表提取子矩阵
    def _eval_extract(self, rowsList, colsList):
        return self._fromrep(self._rep.extract(rowsList, colsList))

    # 获取矩阵的元素或子矩阵
    def __getitem__(self, key):
        return _getitem_RepMatrix(self, key)

    # 创建一个指定大小的零矩阵对象
    @classmethod
    def _eval_zeros(cls, rows, cols):
        rep = DomainMatrix.zeros((rows, cols), ZZ)
        return cls._fromrep(rep)

    # 创建一个指定大小的单位矩阵对象
    @classmethod
    def _eval_eye(cls, rows, cols):
        rep = DomainMatrix.eye((rows, cols), ZZ)
        return cls._fromrep(rep)

    # 返回矩阵与另一个矩阵的按元素加法结果
    def _eval_add(self, other):
        return classof(self, other)._fromrep(self._rep + other._rep)

    # 返回矩阵与另一个矩阵的矩阵乘法结果
    def _eval_matrix_mul(self, other):
        return classof(self, other)._fromrep(self._rep * other._rep)

    # 返回矩阵与另一个矩阵的按元素乘法结果
    def _eval_matrix_mul_elementwise(self, other):
        selfrep, otherrep = self._rep.unify(other._rep)
        newrep = selfrep.mul_elementwise(otherrep)
        return classof(self, other)._fromrep(newrep)

    # 返回矩阵与标量的乘法结果
    def _eval_scalar_mul(self, other):
        rep, other = self._unify_element_sympy(self._rep, other)
        return self._fromrep(rep.scalarmul(other))

    # 返回标量与矩阵的右乘法结果
    def _eval_scalar_rmul(self, other):
        rep, other = self._unify_element_sympy(self._rep, other)
        return self._fromrep(rep.rscalarmul(other))

    # 返回矩阵的按元素取绝对值的结果
    def _eval_Abs(self):
        return self._fromrep(self._rep.applyfunc(abs))

    # 返回矩阵的共轭矩阵（对于整数或有理数矩阵直接返回复制）
    def _eval_conjugate(self):
        rep = self._rep
        domain = rep.domain
        if domain in (ZZ, QQ):
            return self.copy()
        else:
            return self._fromrep(rep.applyfunc(lambda e: e.conjugate()))
    def equals(self, other, failing_expression=False):
        """
        Applies ``equals`` to corresponding elements of the matrices,
        trying to prove that the elements are equivalent, returning True
        if they are, False if any pair is not, and None (or the first
        failing expression if failing_expression is True) if it cannot
        be decided if the expressions are equivalent or not. This is, in
        general, an expensive operation.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x
        >>> A = Matrix([x*(x - 1), 0])
        >>> B = Matrix([x**2 - x, 0])
        >>> A == B
        False
        >>> A.simplify() == B.simplify()
        True
        >>> A.equals(B)
        True
        >>> A.equals(2)
        False

        See Also
        ========
        sympy.core.expr.Expr.equals
        """
        # 检查矩阵形状是否相同
        if self.shape != getattr(other, 'shape', None):
            return False

        rv = True
        # 遍历矩阵的每个元素
        for i in range(self.rows):
            for j in range(self.cols):
                # 调用元素的 equals 方法比较对应元素
                ans = self[i, j].equals(other[i, j], failing_expression)
                if ans is False:
                    return False
                elif ans is not True and rv is True:
                    rv = ans
        return rv

    def inv_mod(M, m):
        """
        Returns the inverse of the integer matrix ``M`` modulo ``m``.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix(2, 2, [1, 2, 3, 4])
        >>> A.inv_mod(5)
        Matrix([
        [3, 1],
        [4, 2]])
        >>> A.inv_mod(3)
        Matrix([
        [1, 1],
        [0, 1]])

        """

        # 检查矩阵是否为方阵
        if not M.is_square:
            raise NonSquareMatrixError()

        try:
            m = as_int(m)
        except ValueError:
            raise TypeError("inv_mod: modulus m must be an integer")

        # 创建 Galois 域对象
        K = GF(m, symmetric=False)

        try:
            # 将矩阵转换为有限域中的表示
            dM = M.to_DM(K)
        except CoercionFailed:
            raise ValueError("inv_mod: matrix entries must be integers")

        try:
            # 计算矩阵在有限域中的逆矩阵
            dMi = dM.inv()
        except DMNonInvertibleMatrixError as exc:
            msg = f'Matrix is not invertible (mod {m})'
            raise NonInvertibleMatrixError(msg) from exc

        # 将结果转换回普通矩阵表示
        return dMi.to_Matrix()
    def lll(self, delta=0.75):
        """LLL-reduced basis for the rowspace of a matrix of integers.

        Performs the Lenstra–Lenstra–Lovász (LLL) basis reduction algorithm.

        The implementation is provided by :class:`~DomainMatrix`. See
        :meth:`~DomainMatrix.lll` for more details.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 0, 0, 0, -20160],
        ...             [0, 1, 0, 0, 33768],
        ...             [0, 0, 1, 0, 39578],
        ...             [0, 0, 0, 1, 47757]])
        >>> M.lll()
        Matrix([
        [ 10, -3,  -2,  8,  -4],
        [  3, -9,   8,  1, -11],
        [ -3, 13,  -9, -3,  -9],
        [-12, -7, -11,  9,  -1]])

        See Also
        ========

        lll_transform
        sympy.polys.matrices.domainmatrix.DomainMatrix.lll
        """
        # 将 delta 转换为 SymPy 的有理数对象 QQ
        delta = QQ.from_sympy(_sympify(delta))
        # 将 self._rep 转换为整数矩阵对象
        dM = self._rep.convert_to(ZZ)
        # 调用整数矩阵对象的 lll 方法进行 LLL 基底约简，并返回结果
        basis = dM.lll(delta=delta)
        # 将约简后的基底转换为当前对象的表示形式并返回
        return self._fromrep(basis)

    def lll_transform(self, delta=0.75):
        """LLL-reduced basis and transformation matrix.

        Performs the Lenstra–Lenstra–Lovász (LLL) basis reduction algorithm.

        The implementation is provided by :class:`~DomainMatrix`. See
        :meth:`~DomainMatrix.lll_transform` for more details.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 0, 0, 0, -20160],
        ...             [0, 1, 0, 0, 33768],
        ...             [0, 0, 1, 0, 39578],
        ...             [0, 0, 0, 1, 47757]])
        >>> B, T = M.lll_transform()
        >>> B
        Matrix([
        [ 10, -3,  -2,  8,  -4],
        [  3, -9,   8,  1, -11],
        [ -3, 13,  -9, -3,  -9],
        [-12, -7, -11,  9,  -1]])
        >>> T
        Matrix([
        [ 10, -3,  -2,  8],
        [  3, -9,   8,  1],
        [ -3, 13,  -9, -3],
        [-12, -7, -11,  9]])

        The transformation matrix maps the original basis to the LLL-reduced
        basis:

        >>> T * M == B
        True

        See Also
        ========

        lll
        sympy.polys.matrices.domainmatrix.DomainMatrix.lll_transform
        """
        # 将 delta 转换为 SymPy 的有理数对象 QQ
        delta = QQ.from_sympy(_sympify(delta))
        # 将 self._rep 转换为整数矩阵对象
        dM = self._rep.convert_to(ZZ)
        # 调用整数矩阵对象的 lll_transform 方法进行 LLL 基底约简和变换矩阵计算，并返回结果
        basis, transform = dM.lll_transform(delta=delta)
        # 将约简后的基底和变换矩阵转换为当前对象的表示形式并返回
        B = self._fromrep(basis)
        T = self._fromrep(transform)
        return B, T
class MutableRepMatrix(RepMatrix):
    """Mutable matrix based on DomainMatrix as the internal representation"""

    #
    # MutableRepMatrix is a subclass of RepMatrix that adds/overrides methods
    # to make the instances mutable. MutableRepMatrix is a superclass for both
    # MutableDenseMatrix and MutableSparseMatrix.
    #

    is_zero = False  # 初始化类属性 is_zero 为 False

    def __new__(cls, *args, **kwargs):
        return cls._new(*args, **kwargs)

    @classmethod
    def _new(cls, *args, copy=True, **kwargs):
        if copy is False:
            # 如果 copy 参数为 False，则直接使用传入的 rows, cols, flat_list，不创建副本
            if len(args) != 3:
                raise TypeError("'copy=False' requires a matrix be initialized as rows,cols,[list]")
            rows, cols, flat_list = args
        else:
            # 否则，使用 _handle_creation_inputs 方法处理输入，创建 flat_list 的浅拷贝
            rows, cols, flat_list = cls._handle_creation_inputs(*args, **kwargs)
            flat_list = list(flat_list)  # 创建 flat_list 的浅拷贝

        # 使用 _flat_list_to_DomainMatrix 方法将 flat_list 转换为 DomainMatrix 的表示
        rep = cls._flat_list_to_DomainMatrix(rows, cols, flat_list)

        return cls._fromrep(rep)

    @classmethod
    def _fromrep(cls, rep):
        # 使用父类的 __new__ 方法创建对象，并设置 rows 和 cols 属性
        obj = super().__new__(cls)
        obj.rows, obj.cols = rep.shape
        obj._rep = rep  # 设置对象的内部表示为 rep
        return obj

    def copy(self):
        # 返回一个新的 MutableRepMatrix 对象，其内部表示是当前对象 _rep 的副本
        return self._fromrep(self._rep.copy())

    def as_mutable(self):
        # 返回当前对象的副本，作为可变的 MutableRepMatrix 对象
        return self.copy()

    def __setitem__(self, key, value):
        """
        根据 key 设置矩阵中的元素为 value。

        Examples
        ========

        >>> from sympy import Matrix, I, zeros, ones
        >>> m = Matrix(((1, 2+I), (3, 4)))
        >>> m
        Matrix([
        [1, 2 + I],
        [3,     4]])
        >>> m[1, 0] = 9
        >>> m
        Matrix([
        [1, 2 + I],
        [9,     4]])
        >>> m[1, 0] = [[0, 1]]

        要替换第 r 行，可以将值分配给 r*m 的位置，其中 m 是列数：

        >>> M = zeros(4)
        >>> m = M.cols
        >>> M[3*m] = ones(1, m)*2; M
        Matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 2, 2, 2]])

        要替换第 c 列，可以将值分配给 c 的位置：

        >>> M[2] = ones(m, 1)*4; M
        Matrix([
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [2, 2, 4, 2]])
        """
        rv = self._setitem(key, value)
        if rv is not None:
            i, j, value = rv
            self._rep, value = self._unify_element_sympy(self._rep, value)
            self._rep.rep.setitem(i, j, value)

    def _eval_col_del(self, col):
        # 删除列 col，更新 _rep 为删除列后的 DomainMatrix，并更新 cols 属性
        self._rep = DomainMatrix.hstack(self._rep[:,:col], self._rep[:,col+1:])
        self.cols -= 1

    def _eval_row_del(self, row):
        # 删除行 row，更新 _rep 为删除行后的 DomainMatrix，并更新 rows 属性
        self._rep = DomainMatrix.vstack(self._rep[:row,:], self._rep[row+1:, :])
        self.rows -= 1

    def _eval_col_insert(self, col, other):
        # 在列 col 处插入新的列 other，将 other 转换为 DomainMatrix，并与当前对象合并
        other = self._new(other)
        return self.hstack(self[:,:col], other, self[:,col:])
    def _eval_row_insert(self, row, other):
        # 创建一个新的矩阵对象 `other`，与当前对象格式相同
        other = self._new(other)
        # 将当前矩阵在指定行 `row` 处与 `other` 矩阵垂直合并，返回合并后的新矩阵
        return self.vstack(self[:row,:], other, self[row:,:])

    def col_op(self, j, f):
        """对第 `j` 列进行原地操作，使用两个参数的函数作为处理函数，参数为 (self[i, j], i)。

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.col_op(1, lambda v, i: v + 2*M[i, 0]); M
        Matrix([
        [1, 2, 0],
        [0, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        col
        row_op
        """
        # 遍历矩阵的行数，对第 `j` 列的每个元素应用函数 `f`
        for i in range(self.rows):
            self[i, j] = f(self[i, j], i)

    def col_swap(self, i, j):
        """原地交换矩阵的两列。

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 0], [1, 0]])
        >>> M
        Matrix([
        [1, 0],
        [1, 0]])
        >>> M.col_swap(0, 1)
        >>> M
        Matrix([
        [0, 1],
        [0, 1]])

        See Also
        ========

        col
        row_swap
        """
        # 遍历矩阵的行数，交换第 `i` 列和第 `j` 列的对应元素
        for k in range(0, self.rows):
            self[k, i], self[k, j] = self[k, j], self[k, i]

    def row_op(self, i, f):
        """对第 `i` 行进行原地操作，使用两个参数的函数作为处理函数，参数为 (self[i, j], j)。

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.row_op(1, lambda v, j: v + 2*M[0, j]); M
        Matrix([
        [1, 0, 0],
        [2, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        row
        zip_row_op
        col_op

        """
        # 遍历矩阵的列数，对第 `i` 行的每个元素应用函数 `f`
        for j in range(self.cols):
            self[i, j] = f(self[i, j], j)

    def row_mult(self,i,factor):
        """将第 `i` 行的所有元素乘以给定的因子 `factor`，原地修改。

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.row_mult(1,7); M
        Matrix([
        [1, 0, 0],
        [0, 7, 0],
        [0, 0, 1]])

        """
        # 遍历第 `i` 行的所有列，将每个元素乘以 `factor`
        for j in range(self.cols):
            self[i,j] *= factor

    def row_add(self,s,t,k):
        """将第 `s` 行乘以 `k` 后加到第 `t` 行，原地修改。

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.row_add(0, 2,3); M
        Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [3, 0, 1]])
        """

        # 遍历行中所有的列，将第 `s` 行乘以 `k` 后加到第 `t` 行对应列上
        for j in range(self.cols):
            self[t,j] += k*self[s,j]
    # 交换矩阵中的两行，实现就地操作。
    def row_swap(self, i, j):
        """Swap the two given rows of the matrix in-place.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[0, 1], [1, 0]])
        >>> M
        Matrix([
        [0, 1],
        [1, 0]])
        >>> M.row_swap(0, 1)
        >>> M
        Matrix([
        [1, 0],
        [0, 1]])

        See Also
        ========

        row
        col_swap
        """
        for k in range(0, self.cols):
            # 交换第 i 行和第 j 行在第 k 列上的元素
            self[i, k], self[j, k] = self[j, k], self[i, k]

    # 对矩阵的第 i 行进行就地操作，使用二参数的函数对象，其参数为 ``(self[i, j], self[k, j])``。
    def zip_row_op(self, i, k, f):
        """In-place operation on row ``i`` using two-arg functor whose args are
        interpreted as ``(self[i, j], self[k, j])``.

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.zip_row_op(1, 0, lambda v, u: v + 2*u); M
        Matrix([
        [1, 0, 0],
        [2, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        row
        row_op
        col_op

        """
        for j in range(self.cols):
            # 使用函数对象 f 对第 i 行和第 k 行的第 j 列元素进行操作
            self[i, j] = f(self[i, j], self[k, j])

    # 从列表中复制元素到矩阵中指定的部分。
    def copyin_list(self, key, value):
        """Copy in elements from a list.

        Parameters
        ==========

        key : slice
            The section of this matrix to replace.
        value : iterable
            The iterable to copy values from.

        Examples
        ========

        >>> from sympy import eye
        >>> I = eye(3)
        >>> I[:2, 0] = [1, 2] # col
        >>> I
        Matrix([
        [1, 0, 0],
        [2, 1, 0],
        [0, 0, 1]])
        >>> I[1, :2] = [[3, 4]]
        >>> I
        Matrix([
        [1, 0, 0],
        [3, 4, 0],
        [0, 0, 1]])

        See Also
        ========

        copyin_matrix
        """
        # 检查 value 是否为有序可迭代对象，否则抛出类型错误异常
        if not is_sequence(value):
            raise TypeError("`value` must be an ordered iterable, not %s." % type(value))
        # 使用 value 中的值替换矩阵中 key 指定的部分，并返回修改后的矩阵
        return self.copyin_matrix(key, type(self)(value))
    def copyin_matrix(self, key, value):
        """Copy in values from a matrix into the given bounds.

        Parameters
        ==========

        key : slice
            The section of this matrix to replace.
        value : Matrix
            The matrix to copy values from.

        Examples
        ========

        >>> from sympy import Matrix, eye
        >>> M = Matrix([[0, 1], [2, 3], [4, 5]])
        >>> I = eye(3)
        >>> I[:3, :2] = M
        >>> I
        Matrix([
        [0, 1, 0],
        [2, 3, 0],
        [4, 5, 1]])
        >>> I[0, 1] = M
        >>> I
        Matrix([
        [0, 0, 1],
        [2, 2, 3],
        [4, 4, 5]])

        See Also
        ========

        copyin_list
        """

        # 根据给定的键计算边界索引
        rlo, rhi, clo, chi = self.key2bounds(key)
        # 获取矩阵 `value` 的形状
        shape = value.shape
        # 计算行和列的偏移量
        dr, dc = rhi - rlo, chi - clo
        # 检查 `value` 是否与指定的子矩阵具有相同的维度
        if shape != (dr, dc):
            raise ShapeError(filldedent("The Matrix `value` doesn't have the "
                                        "same dimensions "
                                        "as the in sub-Matrix given by `key`."))

        # 将矩阵 `value` 的值复制到当前对象的指定边界内
        for i in range(value.rows):
            for j in range(value.cols):
                self[i + rlo, j + clo] = value[i, j]

    def fill(self, value):
        """Fill self with the given value.

        Notes
        =====

        Unless many values are going to be deleted (i.e. set to zero)
        this will create a matrix that is slower than a dense matrix in
        operations.

        Examples
        ========

        >>> from sympy import SparseMatrix
        >>> M = SparseMatrix.zeros(3); M
        Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
        >>> M.fill(1); M
        Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])

        See Also
        ========

        zeros
        ones
        """

        # 将输入值转换为符号对象
        value = _sympify(value)
        # 如果值为空，则用零填充当前对象
        if not value:
            self._rep = DomainMatrix.zeros(self.shape, EXRAW)
        else:
            # 创建一个字典，表示所有元素的值都为给定值
            elements_dod = {i: dict.fromkeys(range(self.cols), value) for i in range(self.rows)}
            self._rep = DomainMatrix(elements_dod, self.shape, EXRAW)
# 返回由给定键定义的 self 的部分。如果键涉及切片，则将返回列表（如果键是单个切片）或矩阵（如果键是涉及切片的元组）。
def _getitem_RepMatrix(self, key):
    if isinstance(key, tuple):  # 如果键是元组
        i, j = key
        try:
            # 尝试从 self._rep 中获取符号表达式 index_(i) 和 index_(j) 对应的元素
            return self._rep.getitem_sympy(index_(i), index_(j))
        except (TypeError, IndexError):
            # 处理索引或切片超出边界的情况
            if (isinstance(i, Expr) and not i.is_number) or (isinstance(j, Expr) and not j.is_number):
                if ((j < 0) is True) or ((j >= self.shape[1]) is True) or \
                   ((i < 0) is True) or ((i >= self.shape[0]) is True):
                    raise ValueError("index out of boundary")
                from sympy.matrices.expressions.matexpr import MatrixElement
                return MatrixElement(self, i, j)

            # 处理行索引 i 是切片的情况
            if isinstance(i, slice):
                i = range(self.rows)[i]
            elif is_sequence(i):
                pass
            else:
                i = [i]

            # 处理列索引 j 是切片的情况
            if isinstance(j, slice):
                j = range(self.cols)[j]
            elif is_sequence(j):
                pass
            else:
                j = [j]

            # 调用 self 的 extract 方法，提取给定的行和列，并返回结果
            return self.extract(i, j)

    else:
        rows, cols = self.shape

        # 如果 self 是空矩阵，则抛出适当的异常
        if not rows * cols:
            return [][key]

        # 获取 self._rep 的表示和域
        rep = self._rep.rep
        domain = rep.domain
        is_slice = isinstance(key, slice)

        # 如果 key 是切片，则根据切片提取所有元素的值
        if is_slice:
            values = [rep.getitem(*divmod(n, cols)) for n in range(rows * cols)[key]]
        else:
            # 否则，获取单个元素的值
            values = [rep.getitem(*divmod(index_(key), cols))]

        # 如果域不是 EXRAW，则将值转换为 sympy 对象
        if domain != EXRAW:
            to_sympy = domain.to_sympy
            values = [to_sympy(val) for val in values]

        # 如果 key 是切片，则返回值列表；否则返回单个值
        if is_slice:
            return values
        else:
            return values[0]
```
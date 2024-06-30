# `D:\src\scipysrc\sympy\sympy\polys\solvers.py`

```
    """
    Low-level linear systems solver.
    """

# 从 sympy 库中导入相关模块和异常类
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import connected_components

# 从 sympy 核心模块中导入 sympify 函数和数值类型 Integer, Rational
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer, Rational

# 从 sympy 矩阵模块中导入 MutableDenseMatrix 类
from sympy.matrices.dense import MutableDenseMatrix

# 从 sympy 多项式模块中导入整数和有理数的多项式环 ZZ, QQ
from sympy.polys.domains import ZZ, QQ

# 从 sympy 多项式模块中导入额外的环 EX, 生成字符串环 sring
from sympy.polys.domains import EX
from sympy.polys.rings import sring

# 从 sympy 多项式模块中导入特定异常类 NotInvertible 和 DomainMatrix 类
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.domainmatrix import DomainMatrix


class PolyNonlinearError(Exception):
    """
    Raised by solve_lin_sys for nonlinear equations
    """
    pass


class RawMatrix(MutableDenseMatrix):
    """
    .. deprecated:: 1.9

       This class fundamentally is broken by design. Use ``DomainMatrix`` if
       you want a matrix over the polys domains or ``Matrix`` for a matrix
       with ``Expr`` elements. The ``RawMatrix`` class will be removed/broken
       in future in order to reestablish the invariant that the elements of a
       Matrix should be of type ``Expr``.

    """
    # 定义静态方法 _sympify 用于 sympify 转换
    _sympify = staticmethod(lambda x, *args, **kwargs: x)

    def __init__(self, *args, **kwargs):
        # 发出 sympy 弃用警告，提示不再使用 RawMatrix 类
        sympy_deprecation_warning(
            """
            The RawMatrix class is deprecated. Use either DomainMatrix or
            Matrix instead.
            """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-rawmatrix",
        )

        # 初始化域为整数环 ZZ
        domain = ZZ
        # 遍历矩阵的行和列
        for i in range(self.rows):
            for j in range(self.cols):
                # 获取当前位置的元素值
                val = self[i,j]
                # 如果元素值是多项式对象
                if getattr(val, 'is_Poly', False):
                    # 获取多项式的环并转换为表达式
                    K = val.domain[val.gens]
                    val_sympy = val.as_expr()
                # 如果元素值有 parent 属性
                elif hasattr(val, 'parent'):
                    # 获取其父环并将其转换为 sympy 表达式
                    K = val.parent()
                    val_sympy = K.to_sympy(val)
                # 如果元素值是整数或者 Integer 类型
                elif isinstance(val, (int, Integer)):
                    K = ZZ
                    val_sympy = sympify(val)
                # 如果元素值是有理数类型
                elif isinstance(val, Rational):
                    K = QQ
                    val_sympy = val
                else:
                    # 否则，检查是否属于整数环或有理数环，将其转换为 sympy 表达式
                    for K in ZZ, QQ:
                        if K.of_type(val):
                            val_sympy = K.to_sympy(val)
                            break
                    else:
                        # 如果找不到合适的环，则抛出类型错误异常
                        raise TypeError
                # 统一当前域和元素值所属的环
                domain = domain.unify(K)
                # 更新矩阵中的元素值为 sympy 表达式
                self[i,j] = val_sympy
        # 将当前域设置为矩阵的环
        self.ring = domain


def eqs_to_matrix(eqs_coeffs, eqs_rhs, gens, domain):
    """
    Get matrix from linear equations in dict format.

    Explanation
    ===========

    Get the matrix representation of a system of linear equations represented
    as dicts with low-level DomainElement coefficients. This is an
    *internal* function that is used by solve_lin_sys.

    Parameters
    ==========

    eqs_coeffs : dict
        Dictionary representing coefficients of equations.
    eqs_rhs : list
        List representing right-hand side of equations.
    gens : list
        List of generators (symbols) of the equations.
    domain : Domain
        Domain over which the equations are defined.

    """
    # 将符号映射到索引的字典，用于构建增广矩阵
    sym2index = {x: n for n, x in enumerate(gens)}
    
    # 矩阵的行数，即方程的数量
    nrows = len(eqs_coeffs)
    
    # 矩阵的列数，包括未知数和右侧常数项
    ncols = len(gens) + 1
    
    # 初始化矩阵的行，每行包含 ncols 个元素，初始值为 domain 的零元素
    rows = [[domain.zero] * ncols for _ in range(nrows)]
    
    # 遍历每个方程的系数和右侧常数，填充增广矩阵的每一行
    for row, eq_coeff, eq_rhs in zip(rows, eqs_coeffs, eqs_rhs):
        # 将每个符号对应的系数转换为 domain 中的元素，填入相应的列
        for sym, coeff in eq_coeff.items():
            row[sym2index[sym]] = domain.convert(coeff)
        # 将右侧常数转换为 domain 中的元素，填入最后一列
        row[-1] = -domain.convert(eq_rhs)
    
    # 返回用 DomainMatrix 表示的增广矩阵，包括其维度和所属域
    return DomainMatrix(rows, (nrows, ncols), domain)
# 解决一组线性方程组，这些方程是多项式环中的元素
def solve_lin_sys(eqs, ring, _raw=True):
    # 解决给定的多项式环中的线性方程组，返回的字典中的键和值将是多项式元素
    Explanation
    ===========
    
    # 解决给定的多项式环中的线性方程组，返回的字典中的键和值将是多项式元素
    Example sympy
    # 如果 _raw=False，则返回的结果字典的键为 Symbol, 值为 Expr
    # 如果 _raw=True，则返回的结果字典的键为 Symbol, 值为 DomainElement

    # 确保 ring 的域是一个字段
    assert ring.domain.is_Field

    # 将每个方程式转换为字典形式，其中每个项是一个 (Symbol, coeff) 的键值对
    eqs_dict = [dict(eq) for eq in eqs]

    # 获取环中的单位单项式
    one_monom = ring.one.monoms()[0]
    # 获取环的零元素
    zero = ring.domain.zero

    # 初始化存储方程式右侧和系数的列表
    eqs_rhs = []
    eqs_coeffs = []

    # 遍历每个方程式的字典表示
    for eq_dict in eqs_dict:
        # 弹出单位单项式对应的右侧值，默认为零
        eq_rhs = eq_dict.pop(one_monom, zero)
        # 初始化存储方程式系数的字典
        eq_coeffs = {}
        
        # 遍历方程式中的每个单项式和系数
        for monom, coeff in eq_dict.items():
            # 检查单项式是否为一次项，否则抛出异常
            if sum(monom) != 1:
                msg = "Nonlinear term encountered in solve_lin_sys"
                raise PolyNonlinearError(msg)
            # 将单项式的系数映射到对应的符号变量上
            eq_coeffs[ring.gens[monom.index(1)]] = coeff
        
        # 如果方程式没有系数，且右侧也为零，则继续处理下一个方程式
        if not eq_coeffs:
            if not eq_rhs:
                continue
            else:
                return None
        
        # 将当前方程式的右侧值和系数存入列表
        eqs_rhs.append(eq_rhs)
        eqs_coeffs.append(eq_coeffs)

    # 调用内部函数 _solve_lin_sys 解决线性方程组，并返回结果
    result = _solve_lin_sys(eqs_coeffs, eqs_rhs, ring)

    # 如果结果不为空且需要返回表达式形式的结果
    if result is not None and as_expr:
        # 定义一个函数，将结果转换为 SymPy 表达式
        def to_sympy(x):
            # 尝试获取 x 的 as_expr 方法，如果存在则调用返回其表达式，否则使用域的 to_sympy 方法
            as_expr = getattr(x, 'as_expr', None)
            if as_expr:
                return as_expr()
            else:
                return ring.domain.to_sympy(x)

        # 将结果中的每个符号和值转换为 SymPy 表达式，并构造新的结果字典
        tresult = {to_sympy(sym): to_sympy(val) for sym, val in result.items()}

        # 去除结果中形如 1.0x 的项，保留为 1x 的形式
        result = {}
        for k, v in tresult.items():
            if k.is_Mul:
                c, s = k.as_coeff_Mul()
                result[s] = v/c
            else:
                result[k] = v

    # 返回处理后的结果字典
    return result
def _solve_lin_sys(eqs_coeffs, eqs_rhs, ring):
    """Solve a linear system from dict of PolynomialRing coefficients

    Explanation
    ===========

    This is an **internal** function used by :func:`solve_lin_sys` after the
    equations have been preprocessed. The role of this function is to split
    the system into connected components and pass those to
    :func:`_solve_lin_sys_component`.

    Examples
    ========

    Setup a system for $x-y=0$ and $x+y=2$ and solve:

    >>> from sympy import symbols, sring
    >>> from sympy.polys.solvers import _solve_lin_sys
    >>> x, y = symbols('x, y')
    >>> R, (xr, yr) = sring([x, y], [x, y])
    >>> eqs = [{xr:R.one, yr:-R.one}, {xr:R.one, yr:R.one}]
    >>> eqs_rhs = [R.zero, -2*R.one]
    >>> _solve_lin_sys(eqs, eqs_rhs, R)
    {y: 1, x: 1}

    See also
    ========

    solve_lin_sys: This function is used internally by :func:`solve_lin_sys`.
    """
    
    # Extract the generators (variables) from the ring
    V = ring.gens
    
    # Initialize an empty list for edges (pairs of variables)
    E = []
    
    # Iterate over each equation's coefficients dictionary
    for eq_coeffs in eqs_coeffs:
        # Obtain a list of symbols (variables) from the coefficients
        syms = list(eq_coeffs)
        # Create edges by pairing consecutive symbols
        E.extend(zip(syms[:-1], syms[1:]))
    
    # Form the graph G with vertices (generators) V and edges E
    G = V, E
    
    # Find connected components in the graph G
    components = connected_components(G)
    
    # Map symbols to their respective connected component index
    sym2comp = {}
    for n, component in enumerate(components):
        for sym in component:
            sym2comp[sym] = n
    
    # Initialize subsystems as empty lists of coefficients and right-hand sides
    subsystems = [([], []) for _ in range(len(components))]
    
    # Assign equations to their corresponding subsystem based on sym2comp
    for eq_coeff, eq_rhs in zip(eqs_coeffs, eqs_rhs):
        # Get the first symbol (variable) in the current equation
        sym = next(iter(eq_coeff), None)
        # Retrieve the subsystem corresponding to the symbol's component
        sub_coeff, sub_rhs = subsystems[sym2comp[sym]]
        # Append the current equation's coefficients and rhs to the subsystem
        sub_coeff.append(eq_coeff)
        sub_rhs.append(eq_rhs)
    
    # Initialize an empty dictionary for the solution
    sol = {}
    
    # Solve each subsystem using _solve_lin_sys_component and update the solution
    for subsystem in subsystems:
        subsol = _solve_lin_sys_component(subsystem[0], subsystem[1], ring)
        if subsol is None:
            return None
        sol.update(subsol)
    
    # Return the final solution dictionary
    return sol


def _solve_lin_sys_component(eqs_coeffs, eqs_rhs, ring):
    """Solve a linear system from dict of PolynomialRing coefficients

    Explanation
    ===========

    This is an **internal** function used by :func:`solve_lin_sys` after the
    equations have been preprocessed. After :func:`_solve_lin_sys` splits the
    system into connected components this function is called for each
    component. The system of equations is solved using Gauss-Jordan
    elimination with division followed by back-substitution.

    Examples
    ========

    Setup a system for $x-y=0$ and $x+y=2$ and solve:

    >>> from sympy import symbols, sring
    >>> from sympy.polys.solvers import _solve_lin_sys_component
    >>> x, y = symbols('x, y')
    >>> R, (xr, yr) = sring([x, y], [x, y])
    >>> eqs = [{xr:R.one, yr:-R.one}, {xr:R.one, yr:R.one}]
    >>> eqs_rhs = [R.zero, -2*R.one]
    >>> _solve_lin_sys_component(eqs, eqs_rhs, R)
    {y: 1, x: 1}

    See also
    ========

    solve_lin_sys: This function is used internally by :func:`solve_lin_sys`.
    """
    
    # Transform the equations and right-hand sides into matrix form
    matrix = eqs_to_matrix(eqs_coeffs, eqs_rhs, ring.gens, ring.domain)
    
    # Convert the matrix to a field for reduced row echelon form (rref)
    # (Further code for rref is not provided in the snippet)
    # 如果矩阵的定义域不是一个域，则将其转换为域
    if not matrix.domain.is_Field:
        matrix = matrix.to_field()

    # 对矩阵进行行简化（行阶梯形）操作
    echelon, pivots = matrix.rref()

    # 构造可返回的解的形式
    keys = ring.gens

    # 如果存在主元，并且最后一个主元索引等于变量的数量，则返回空解集
    if pivots and pivots[-1] == len(keys):
        return None

    # 如果主元的数量等于变量的数量，则有唯一解
    if len(pivots) == len(keys):
        # 提取解的值并组成解的字典
        sol = []
        for s in [row[-1] for row in echelon.rep.to_ddm()]:
            a = s
            sol.append(a)
        sols = dict(zip(keys, sol))
    else:
        # 否则，解集为空或者有无穷多解，需要进一步处理
        sols = {}
        g = ring.gens

        # 提取基础域的系数并转换为当前环的元素
        if hasattr(ring, 'ring'):
            convert = ring.ring.ground_new
        else:
            convert = ring.ground_new

        # 将矩阵转换为基础域的表达式
        echelon = echelon.rep.to_ddm()

        # 收集矩阵中出现的值，并将其映射到当前环的元素
        vals_set = {v for row in echelon for v in row}
        vals_map = {v: convert(v) for v in vals_set}

        # 将矩阵的每个元素转换为当前环的元素
        echelon = [[vals_map[eij] for eij in ei] for ei in echelon]

        # 根据主元计算非主元变量的值，并构建解的字典
        for i, p in enumerate(pivots):
            v = echelon[i][-1] - sum(echelon[i][j]*g[j] for j in range(p+1, len(g)) if echelon[i][j])
            sols[keys[p]] = v

    # 返回解的字典
    return sols
```
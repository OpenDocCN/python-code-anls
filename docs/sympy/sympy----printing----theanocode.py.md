# `D:\src\scipysrc\sympy\sympy\printing\theanocode.py`

```
"""
.. deprecated:: 1.8

  ``sympy.printing.theanocode`` is deprecated. Theano has been renamed to
  Aesara. Use ``sympy.printing.aesaracode`` instead. See
  :ref:`theanocode-deprecated` for more information.

"""
# 引入从未来导入的语法，确保代码能在旧版本的 Python 中运行
from __future__ import annotations
# 导入类型提示相关的模块
from typing import Any

# 导入外部模块以及函数
from sympy.external import import_module
# 导入打印相关的类
from sympy.printing.printer import Printer
# 导入集合工具模块
from sympy.utilities.iterables import is_sequence
# 导入 sympy 主模块
import sympy
# 导入 functools 中的 partial 函数
from functools import partial

# 导入装饰器相关的函数
from sympy.utilities.decorator import doctest_depends_on
# 导入 sympy 异常模块
from sympy.utilities.exceptions import sympy_deprecation_warning

# 指定 doctest 的依赖项
__doctest_requires__ = {('theano_function',): ['theano']}

# 尝试导入 theano 模块
theano = import_module('theano')

# 如果成功导入 theano 模块，则执行以下代码块
if theano:
    # 导入 theano 中的标量和张量
    ts = theano.scalar
    tt = theano.tensor
    # 从 theano.sandbox 中导入线性代数相关的模块
    from theano.sandbox import linalg as tlinalg

    # 定义一个映射字典，将 sympy 函数映射到对应的 theano.tensor 函数
    mapping = {
            sympy.Add: tt.add,
            sympy.Mul: tt.mul,
            sympy.Abs: tt.abs_,
            sympy.sign: tt.sgn,
            sympy.ceiling: tt.ceil,
            sympy.floor: tt.floor,
            sympy.log: tt.log,
            sympy.exp: tt.exp,
            sympy.sqrt: tt.sqrt,
            sympy.cos: tt.cos,
            sympy.acos: tt.arccos,
            sympy.sin: tt.sin,
            sympy.asin: tt.arcsin,
            sympy.tan: tt.tan,
            sympy.atan: tt.arctan,
            sympy.atan2: tt.arctan2,
            sympy.cosh: tt.cosh,
            sympy.acosh: tt.arccosh,
            sympy.sinh: tt.sinh,
            sympy.asinh: tt.arcsinh,
            sympy.tanh: tt.tanh,
            sympy.atanh: tt.arctanh,
            sympy.re: tt.real,
            sympy.im: tt.imag,
            sympy.arg: tt.angle,
            sympy.erf: tt.erf,
            sympy.gamma: tt.gamma,
            sympy.loggamma: tt.gammaln,
            sympy.Pow: tt.pow,
            sympy.Eq: tt.eq,
            sympy.StrictGreaterThan: tt.gt,
            sympy.StrictLessThan: tt.lt,
            sympy.LessThan: tt.le,
            sympy.GreaterThan: tt.ge,
            sympy.And: tt.and_,
            sympy.Or: tt.or_,
            sympy.Max: tt.maximum,  # SymPy accept >2 inputs, Theano only 2
            sympy.Min: tt.minimum,  # SymPy accept >2 inputs, Theano only 2
            sympy.conjugate: tt.conj,
            sympy.core.numbers.ImaginaryUnit: lambda:tt.complex(0,1),
            # 矩阵相关
            sympy.MatAdd: tt.Elemwise(ts.add),
            sympy.HadamardProduct: tt.Elemwise(ts.mul),
            sympy.Trace: tlinalg.trace,
            sympy.Determinant : tlinalg.det,
            sympy.Inverse: tlinalg.matrix_inverse,
            sympy.Transpose: tt.DimShuffle((False, False), [1, 0]),
    }


# 定义一个用于创建 Theano 符号表达式图的打印类
class TheanoPrinter(Printer):
    """ Code printer which creates Theano symbolic expression graphs.

    Parameters
    ==========

    """
    cache : dict
        # 缓存字典，用于存储数据以供使用。如果为 None（默认值），将使用全局缓存。
        # 若要创建不依赖或改变全局状态的打印机，请传入一个空字典。
        # 注意：字典在打印机初始化时不会被复制，会直接进行原地更新。
        # 因此，如果在创建多个打印机或多次调用 .theano_code 或 .theano_function 时使用相同的字典对象，
        # 则缓存将在所有这些应用之间共享。

    Attributes
    ==========

    cache : dict
        # 缓存 Theano 变量的字典，用于存储为 SymPy 符号类似对象（例如 sympy.core.symbol.Symbol
        # 或 sympy.matrices.expressions.MatrixSymbol）创建的 Theano 变量。
        # 此缓存确保在表达式中（或多个表达式中）对给定符号的所有引用都打印为相同的 Theano 变量，
        # 只创建一次。符号仅通过名称和类型进行区分。
        # 缓存内容的格式应被视为不透明，用户不应直接操作其内容。
    """
    printmethod = "_theano"

    def __init__(self, *args, **kwargs):
        """
        初始化函数，接受任意位置参数和关键字参数。
        """
        self.cache = kwargs.pop('cache', {})  # 从 kwargs 中取出 'cache' 参数，若不存在则使用空字典
        super().__init__(*args, **kwargs)

    def _get_key(self, s, name=None, dtype=None, broadcastable=None):
        """ Get the cache key for a SymPy object.

        Parameters
        ==========

        s : sympy.core.basic.Basic
            # 要获取键的 SymPy 对象。

        name : str
            # 对象的名称，如果它没有 'name' 属性。
        """
        
        if name is None:
            name = s.name

        return (name, type(s), s.args, dtype, broadcastable)

    def _get_or_create(self, s, name=None, dtype=None, broadcastable=None):
        """
        从缓存中获取 SymPy 符号对应的 Theano 变量，如果不存在则创建。
        """

        # 默认值
        if name is None:
            name = s.name
        if dtype is None:
            dtype = 'floatX'
        if broadcastable is None:
            broadcastable = ()

        key = self._get_key(s, name, dtype=dtype, broadcastable=broadcastable)

        if key in self.cache:
            return self.cache[key]

        # 创建新的 Theano 变量，并将其存入缓存
        value = tt.tensor(name=name, dtype=dtype, broadcastable=broadcastable)
        self.cache[key] = value
        return value

    def _print_Symbol(self, s, **kwargs):
        """
        打印 SymPy 符号对象 s 对应的 Theano 变量。
        """
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, dtype=dtype, broadcastable=bc)

    def _print_AppliedUndef(self, s, **kwargs):
        """
        打印应用未定义函数对象 s 对应的 Theano 变量。
        """
        name = str(type(s)) + '_' + str(s.args[0])
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, name=name, dtype=dtype, broadcastable=bc)
    def _print_Basic(self, expr, **kwargs):
        # 获取表达式类型对应的操作符
        op = mapping[type(expr)]
        # 对表达式的每个参数进行打印，并存储在列表中
        children = [self._print(arg, **kwargs) for arg in expr.args]
        # 调用操作符函数，传入参数列表，并返回结果
        return op(*children)

    def _print_Number(self, n, **kwargs):
        # 如果是整数，在下面已处理，这里将其解释为浮点数
        return float(n.evalf())

    def _print_MatrixSymbol(self, X, **kwargs):
        # 获取矩阵符号对应的数据类型
        dtype = kwargs.get('dtypes', {}).get(X)
        # 获取或创建矩阵符号对应的张量
        return self._get_or_create(X, dtype=dtype, broadcastable=(None, None))

    def _print_DenseMatrix(self, X, **kwargs):
        # 如果 tt 模块没有 stacklists 属性，则抛出未实现错误
        if not hasattr(tt, 'stacklists'):
            raise NotImplementedError(
               "Matrix translation not yet supported in this version of Theano")
        # 将 DenseMatrix 转换为 Theano 张量
        return tt.stacklists([
            [self._print(arg, **kwargs) for arg in L]
            for L in X.tolist()
        ])

    _print_ImmutableMatrix = _print_ImmutableDenseMatrix = _print_DenseMatrix

    def _print_MatMul(self, expr, **kwargs):
        # 获取 MatMul 表达式的参数列表
        children = [self._print(arg, **kwargs) for arg in expr.args]
        # 初始化结果为第一个参数
        result = children[0]
        # 对剩余参数进行矩阵乘法运算
        for child in children[1:]:
            result = tt.dot(result, child)
        return result

    def _print_MatPow(self, expr, **kwargs):
        # 获取 MatPow 表达式的参数列表
        children = [self._print(arg, **kwargs) for arg in expr.args]
        # 初始化结果为单位矩阵
        result = 1
        # 如果指数为正整数，进行矩阵乘法的迭代计算
        if isinstance(children[1], int) and children[1] > 0:
            for i in range(children[1]):
                result = tt.dot(result, children[0])
        else:
            # 抛出未实现错误，目前 Theano 只能处理非负整数次幂的矩阵
            raise NotImplementedError('''Only non-negative integer
           powers of matrices can be handled by Theano at the moment''')
        return result

    def _print_MatrixSlice(self, expr, **kwargs):
        # 打印父矩阵表达式，并获取行列切片
        parent = self._print(expr.parent, **kwargs)
        rowslice = self._print(slice(*expr.rowslice), **kwargs)
        colslice = self._print(slice(*expr.colslice), **kwargs)
        # 返回行列切片后的子矩阵
        return parent[rowslice, colslice]

    def _print_BlockMatrix(self, expr, **kwargs):
        # 获取块矩阵的行数和列数
        nrows, ncols = expr.blocks.shape
        # 对每个块矩阵元素进行打印，并将结果组成列表
        blocks = [[self._print(expr.blocks[r, c], **kwargs)
                        for c in range(ncols)]
                        for r in range(nrows)]
        # 使用 Theano 的 join 函数将所有块拼接成一个矩阵
        return tt.join(0, *[tt.join(1, *row) for row in blocks])

    def _print_slice(self, expr, **kwargs):
        # 打印切片对象的起始、停止和步长，如果是 sympy.Basic 对象，则进行打印
        return slice(*[self._print(i, **kwargs)
                        if isinstance(i, sympy.Basic) else i
                        for i in (expr.start, expr.stop, expr.step)])

    def _print_Pi(self, expr, **kwargs):
        # 返回数学常数 π 的值
        return 3.141592653589793

    def _print_Exp1(self, expr, **kwargs):
        # 返回数学常数 e 的指数
        return ts.exp(1)
    # 打印 Piecewise 函数的表达式
    def _print_Piecewise(self, expr, **kwargs):
        import numpy as np
        # 提取第一个条件和相应的值
        e, cond = expr.args[0].args
        
        # 打印第一个条件及其对应的值
        p_cond = self._print(cond, **kwargs)
        p_e = self._print(e, **kwargs)

        # 如果只有一个条件
        if len(expr.args) == 1:
            # 如果条件成立返回值，否则返回 NaN
            return tt.switch(p_cond, p_e, np.nan)

        # 返回第一个条件成立时的值 p_e，否则递归地计算剩余条件
        p_remaining = self._print(sympy.Piecewise(*expr.args[1:]), **kwargs)
        return tt.switch(p_cond, p_e, p_remaining)

    # 打印 Rational 类型的表达式
    def _print_Rational(self, expr, **kwargs):
        # 返回表达式的分子和分母的真实除法
        return tt.true_div(self._print(expr.p, **kwargs),
                           self._print(expr.q, **kwargs))

    # 打印 Integer 类型的表达式
    def _print_Integer(self, expr, **kwargs):
        # 直接返回整数值
        return expr.p

    # 打印 factorial 函数的表达式
    def _print_factorial(self, expr, **kwargs):
        # 将 factorial 转换为 gamma 函数的表达式再打印
        return self._print(sympy.gamma(expr.args[0] + 1), **kwargs)

    # 打印 Derivative 类型的表达式
    def _print_Derivative(self, deriv, **kwargs):
        # 打印导数的表达式
        rv = self._print(deriv.expr, **kwargs)
        # 对于每一个变量，打印其表达式
        for var in deriv.variables:
            var = self._print(var, **kwargs)
            # 使用 tt.Rop 计算导数
            rv = tt.Rop(rv, var, tt.ones_like(var))
        return rv

    # 空的打印函数，直接返回表达式
    def emptyPrinter(self, expr):
        return expr
    def doprint(self, expr, dtypes=None, broadcastables=None):
        """ Convert a SymPy expression to a Theano graph variable.

        The ``dtypes`` and ``broadcastables`` arguments are used to specify the
        data type, dimension, and broadcasting behavior of the Theano variables
        corresponding to the free symbols in ``expr``. Each is a mapping from
        SymPy symbols to the value of the corresponding argument to
        ``theano.tensor.Tensor``.

        See the corresponding `documentation page`__ for more information on
        broadcasting in Theano.

        .. __: http://deeplearning.net/software/theano/tutorial/broadcasting.html

        Parameters
        ==========

        expr : sympy.core.expr.Expr
            SymPy expression to print.

        dtypes : dict, optional
            Mapping from SymPy symbols to Theano datatypes to use when creating
            new Theano variables for those symbols. Corresponds to the ``dtype``
            argument to ``theano.tensor.Tensor``. Defaults to ``'floatX'``
            for symbols not included in the mapping.

        broadcastables : dict, optional
            Mapping from SymPy symbols to the value of the ``broadcastable``
            argument to ``theano.tensor.Tensor`` to use when creating Theano
            variables for those symbols. Defaults to the empty tuple for symbols
            not included in the mapping (resulting in a scalar).

        Returns
        =======

        theano.gof.graph.Variable
            A variable corresponding to the expression's value in a Theano
            symbolic expression graph.

        """
        # 如果未提供 dtypes，则默认为空字典
        if dtypes is None:
            dtypes = {}
        # 如果未提供 broadcastables，则默认为空字典
        if broadcastables is None:
            broadcastables = {}

        # 调用内部方法 _print，将 SymPy 表达式转换为 Theano 图形变量
        return self._print(expr, dtypes=dtypes, broadcastables=broadcastables)
# 全局缓存，用于存储任意类型键值对的字典
global_cache: dict[Any, Any] = {}


# 将 SymPy 表达式转换为 Theano 图变量的函数
def theano_code(expr, cache=None, **kwargs):
    """
    Convert a SymPy expression into a Theano graph variable.

    .. deprecated:: 1.8

      ``sympy.printing.theanocode`` is deprecated. Theano has been renamed to
      Aesara. Use ``sympy.printing.aesaracode`` instead. See
      :ref:`theanocode-deprecated` for more information.

    Parameters
    ==========

    expr : sympy.core.expr.Expr
        SymPy expression object to convert.

    cache : dict
        Cached Theano variables (see :class:`TheanoPrinter.cache
        <TheanoPrinter>`). Defaults to the module-level global cache.

    dtypes : dict
        Passed to :meth:`.TheanoPrinter.doprint`.

    broadcastables : dict
        Passed to :meth:`.TheanoPrinter.doprint`.

    Returns
    =======

    theano.gof.graph.Variable
        A variable corresponding to the expression's value in a Theano symbolic
        expression graph.

    """
    # 发出 SymPy 代码已弃用警告
    sympy_deprecation_warning(
        """
        sympy.printing.theanocode is deprecated. Theano has been renamed to
        Aesara. Use sympy.printing.aesaracode instead.""",
        deprecated_since_version="1.8",
        active_deprecations_target='theanocode-deprecated')

    # 如果没有安装 Theano，则抛出 ImportError
    if not theano:
        raise ImportError("theano is required for theano_code")

    # 如果未提供缓存参数，则使用全局缓存
    if cache is None:
        cache = global_cache

    # 使用 TheanoPrinter 实例进行表达式的打印和转换
    return TheanoPrinter(cache=cache, settings={}).doprint(expr, **kwargs)


# 用于处理维度相关参数的函数
def dim_handling(inputs, dim=None, dims=None, broadcastables=None):
    r"""
    Get value of ``broadcastables`` argument to :func:`.theano_code` from
    keyword arguments to :func:`.theano_function`.

    Included for backwards compatibility.

    Parameters
    ==========

    inputs
        Sequence of input symbols.

    dim : int
        Common number of dimensions for all inputs. Overrides other arguments
        if given.

    dims : dict
        Mapping from input symbols to number of dimensions. Overrides
        ``broadcastables`` argument if given.

    broadcastables : dict
        Explicit value of ``broadcastables`` argument to
        :meth:`.TheanoPrinter.doprint`. If not None function will return this value unchanged.

    Returns
    =======
    dict
        Dictionary mapping elements of ``inputs`` to their "broadcastable"
        values (tuple of ``bool``\ s).
    """
    # 如果给定了 dim 参数，则将所有输入符号映射为指定维度的广播值
    if dim is not None:
        return dict.fromkeys(inputs, (False,) * dim)

    # 如果给定了 dims 参数，则将每个输入符号映射为对应的广播值，以最大维度为准
    if dims is not None:
        maxdim = max(dims.values())
        return {
            s: (False,) * d + (True,) * (maxdim - d)
            for s, d in dims.items()
        }

    # 如果提供了 broadcastables 参数，则直接返回其值
    if broadcastables is not None:
        return broadcastables

    # 若无任何参数，则返回空字典
    return {}


# 从 SymPy 表达式创建 Theano 函数的函数
@doctest_depends_on(modules=('theano',))
def theano_function(inputs, outputs, scalar=False, *,
        dim=None, dims=None, broadcastables=None, **kwargs):
    """
    Create a Theano function from SymPy expressions.
    """
    .. deprecated:: 1.8
    
      ``sympy.printing.theanocode`` is deprecated. Theano has been renamed to
      Aesara. Use ``sympy.printing.aesaracode`` instead. See
      :ref:`theanocode-deprecated` for more information.
    
    # 标记：版本过时提醒
    # 此部分提示代码段已弃用，建议使用 `sympy.printing.aesaracode` 替代，详细信息参见 `theanocode-deprecated` 链接。
    
    
    The inputs and outputs are converted to Theano variables using
    :func:`.theano_code` and then passed to ``theano.function``.
    
    # 标记：输入输出转换为 Theano 变量
    # 将输入和输出转换为 Theano 变量，使用 `theano_code` 函数，然后传递给 `theano.function`。
    
    
    Parameters
    ==========
    
    # 标记：参数定义开始
    
    
    inputs
        Sequence of symbols which constitute the inputs of the function.
    
    # 标记：输入参数
    # `inputs`：函数的输入参数，由符号序列构成。
    
    
    outputs
        Sequence of expressions which constitute the outputs(s) of the
        function. The free symbols of each expression must be a subset of
        ``inputs``.
    
    # 标记：输出参数
    # `outputs`：函数的输出参数，由表达式序列构成。每个表达式的自由符号必须是 `inputs` 的子集。
    
    
    scalar : bool
        Convert 0-dimensional arrays in output to scalars. This will return a
        Python wrapper function around the Theano function object.
    
    # 标记：标量选项
    # `scalar`：布尔值，将输出中的零维数组转换为标量。这将返回一个围绕 Theano 函数对象的 Python 包装器函数。
    
    
    cache : dict
        Cached Theano variables (see :class:`TheanoPrinter.cache
        <TheanoPrinter>`). Defaults to the module-level global cache.
    
    # 标记：缓存选项
    # `cache`：字典，缓存的 Theano 变量（参见 `TheanoPrinter.cache <TheanoPrinter>`）。默认为模块级全局缓存。
    
    
    dtypes : dict
        Passed to :meth:`.TheanoPrinter.doprint`.
    
    # 标记：数据类型选项
    # `dtypes`：字典，传递给 `TheanoPrinter.doprint` 方法。
    
    
    broadcastables : dict
        Passed to :meth:`.TheanoPrinter.doprint`.
    
    # 标记：广播选项
    # `broadcastables`：字典，传递给 `TheanoPrinter.doprint` 方法。
    
    
    dims : dict
        Alternative to ``broadcastables`` argument. Mapping from elements of
        ``inputs`` to integers indicating the dimension of their associated
        arrays/tensors. Overrides ``broadcastables`` argument if given.
    
    # 标记：维度选项
    # `dims`：字典，替代 `broadcastables` 参数。将 `inputs` 的元素映射到表示其关联数组/张量维度的整数。如果给定，则覆盖 `broadcastables` 参数。
    
    
    dim : int
        Another alternative to the ``broadcastables`` argument. Common number of
        dimensions to use for all arrays/tensors.
        ``theano_function([x, y], [...], dim=2)`` is equivalent to using
        ``broadcastables={x: (False, False), y: (False, False)}``.
    
    # 标记：维度参数
    # `dim`：整数，另一种 `broadcastables` 参数的替代方案。用于所有数组/张量的公共维度数。例如，`theano_function([x, y], [...], dim=2)` 等同于使用 `broadcastables={x: (False, False), y: (False, False)}`。
    
    
    Returns
    =======
    
    # 标记：返回值说明开始
    
    
    callable
        A callable object which takes values of ``inputs`` as positional
        arguments and returns an output array for each of the expressions
        in ``outputs``. If ``outputs`` is a single expression the function will
        return a Numpy array, if it is a list of multiple expressions the
        function will return a list of arrays. See description of the ``squeeze``
        argument above for the behavior when a single output is passed in a list.
        The returned object will either be an instance of
        ``theano.compile.function_module.Function`` or a Python wrapper
        function around one. In both cases, the returned value will have a
        ``theano_function`` attribute which points to the return value of
        ``theano.function``.
    
    # 标记：可调用对象的返回值
    # `callable`：一个可调用对象，接受 `inputs` 的值作为位置参数，并为 `outputs` 中的每个表达式返回一个输出数组。如果 `outputs` 是单个表达式，则函数将返回一个 NumPy 数组；如果是多个表达式的列表，则函数将返回一个数组列表。有关 `squeeze` 参数的行为，请参阅上述描述，当以列表形式传递单个输出时的行为。返回的对象将是 `theano.compile.function_module.Function` 的实例或围绕其的 Python 包装器函数。在两种情况下，返回值将具有指向 `theano.function` 返回值的 `theano_function` 属性。
    
    
    Examples
    ========
    
    # 标记：示例开始
    
    
    >>> from sympy.abc import x, y, z
    >>> from sympy.printing.theanocode import theano_function
    
    # 标记：示例导入模块
    # 示例代码导入必要的模块 `sympy.abc` 中的 `x, y, z` 以及 `sympy.printing.theanocode` 中的 `theano_function`。
    
    
    A simple function with one input and one output:
    
    >>> f1 = theano_function([x], [x**2 - 1], scalar=True)
    >>> f1(3)
    8.0
    
    # 标记：简单函数示例
    # 简单的函数示例，具有一个输入和一个输出。
    
    
    A function with multiple inputs and one output:
    
    >>> f2 = theano_function([x, y, z], [(x**z + y**z)**(1/z)], scalar=True)
    >>> f2(3, 4, 2)
    5.0
    
    # 标记：多输入单输出函数示例
    # 具有多个输入和一个输出的函数示例。
    
    
    A function with multiple inputs and multiple outputs:
    
    # 标记：多输入多输出函数示例
    # 具有多个输入和多个输出的函数示例。
    # 创建 theano 函数，接受 x 和 y 作为输入，返回两个结果：x^2 + y^2 和 x^2 - y^2，标量形式
    >>> f3 = theano_function([x, y], [x**2 + y**2, x**2 - y**2], scalar=True)
    
    # 调用 f3 函数，传入参数 2 和 3，返回结果 [13.0, -5.0]
    >>> f3(2, 3)
    [13.0, -5.0]

    # 输出提示信息，表明 sympy.printing.theanocode 已被弃用，建议使用 sympy.printing.aesaracode
    See also
    ========

    dim_handling

    """
    sympy_deprecation_warning(
        """
        sympy.printing.theanocode is deprecated. Theano has been renamed to Aesara. Use sympy.printing.aesaracode instead""",
        deprecated_since_version="1.8",
        active_deprecations_target='theanocode-deprecated')

    # 如果没有安装 theano，抛出 ImportError 异常
    if not theano:
        raise ImportError("theano is required for theano_function")

    # 从 kwargs 中弹出非 theano 的关键字参数
    cache = kwargs.pop('cache', {})
    dtypes = kwargs.pop('dtypes', {})

    # 根据输入参数调用 dim_handling 函数，获取广播参数
    broadcastables = dim_handling(
        inputs, dim=dim, dims=dims, broadcastables=broadcastables,
    )

    # 使用 theano_code 函数生成输入和输出的 Theano 代码
    code = partial(theano_code, cache=cache, dtypes=dtypes,
                   broadcastables=broadcastables)
    tinputs = list(map(code, inputs))   # 转换输入
    toutputs = list(map(code, outputs)) # 转换输出

    # 将常数表达式转换为变量
    toutputs = [output if isinstance(output, theano.Variable) else tt.as_tensor_variable(output) for output in toutputs]

    # 如果只有一个输出，则将其作为单个元素，而不是列表
    if len(toutputs) == 1:
        toutputs = toutputs[0]

    # 编译 Theano 函数
    func = theano.function(tinputs, toutputs, **kwargs)

    # 检查输出是否为 0 维
    is_0d = [len(o.variable.broadcastable) == 0 for o in func.outputs]

    # 如果不是标量或输出不包含任何 0 维的数组，直接返回编译好的函数
    if not scalar or not any(is_0d):
        func.theano_function = func
        return func

    # 创建一个包装器函数，将 0 维输出转换为标量
    def wrapper(*args):
        out = func(*args)
        # out 可能是 array(1.0) 或 [array(1.0), array(2.0)]

        # 如果输出是序列，则将每个 0 维输出转换为标量
        if is_sequence(out):
            return [o[()] if is_0d[i] else o for i, o in enumerate(out)]
        else:
            return out[()]

    wrapper.__wrapped__ = func  # 将原始函数保存在 wrapper 函数的 __wrapped__ 属性中
    wrapper.__doc__ = func.__doc__  # 复制原始函数的文档字符串
    wrapper.theano_function = func  # 将原始函数保存在 wrapper 函数的 theano_function 属性中
    return wrapper
```
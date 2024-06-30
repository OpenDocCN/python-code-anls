# `D:\src\scipysrc\sympy\sympy\printing\aesaracode.py`

```
# 导入未来版本支持的注解和任意类型
from __future__ import annotations
from typing import Any

# 从 sympy.external 中导入 import_module 函数
from sympy.external import import_module
# 从 sympy.printing.printer 中导入 Printer 类
from sympy.printing.printer import Printer
# 从 sympy.utilities.iterables 中导入 is_sequence 函数
from sympy.utilities.iterables import is_sequence
# 导入 sympy 库
import sympy
# 从 functools 中导入 partial 函数
from functools import partial

# 尝试导入 aesara 模块
aesara = import_module('aesara')

# 如果 aesara 模块成功导入
if aesara:
    # 导入 aesara.scalar 模块并赋值给 aes
    aes = aesara.scalar
    # 导入 aesara.tensor 模块并赋值给 aet
    aet = aesara.tensor
    # 从 aesara.tensor 中导入 nlinalg 子模块
    from aesara.tensor import nlinalg
    # 从 aesara.tensor.elemwise 中导入 Elemwise 类
    from aesara.tensor.elemwise import Elemwise
    # 从 aesara.tensor.elemwise 中导入 DimShuffle 类
    from aesara.tensor.elemwise import DimShuffle

    # 为了与 NumPy 匹配，在 Aesara 2.8.11 (2023 年发布) 中，用 `true_divide` 替换了 `true_div`
    # XXX: 当不再需要支持旧版本时，请移除此部分
    true_divide = getattr(aet, 'true_divide', None)
    if true_divide is None:
        true_divide = aet.true_div

    # 将 sympy 中的运算函数映射到对应的 aesara 中的函数
    mapping = {
            sympy.Add: aet.add,
            sympy.Mul: aet.mul,
            sympy.Abs: aet.abs,
            sympy.sign: aet.sgn,
            sympy.ceiling: aet.ceil,
            sympy.floor: aet.floor,
            sympy.log: aet.log,
            sympy.exp: aet.exp,
            sympy.sqrt: aet.sqrt,
            sympy.cos: aet.cos,
            sympy.acos: aet.arccos,
            sympy.sin: aet.sin,
            sympy.asin: aet.arcsin,
            sympy.tan: aet.tan,
            sympy.atan: aet.arctan,
            sympy.atan2: aet.arctan2,
            sympy.cosh: aet.cosh,
            sympy.acosh: aet.arccosh,
            sympy.sinh: aet.sinh,
            sympy.asinh: aet.arcsinh,
            sympy.tanh: aet.tanh,
            sympy.atanh: aet.arctanh,
            sympy.re: aet.real,
            sympy.im: aet.imag,
            sympy.arg: aet.angle,
            sympy.erf: aet.erf,
            sympy.gamma: aet.gamma,
            sympy.loggamma: aet.gammaln,
            sympy.Pow: aet.pow,
            sympy.Eq: aet.eq,
            sympy.StrictGreaterThan: aet.gt,
            sympy.StrictLessThan: aet.lt,
            sympy.LessThan: aet.le,
            sympy.GreaterThan: aet.ge,
            sympy.And: aet.bitwise_and,  # 位运算与
            sympy.Or: aet.bitwise_or,  # 位运算或
            sympy.Not: aet.invert,  # 位运算非
            sympy.Xor: aet.bitwise_xor,  # 位运算异或
            sympy.Max: aet.maximum,  # Sympy 支持 >2 输入，Aesara 只支持 2 个输入
            sympy.Min: aet.minimum,  # Sympy 支持 >2 输入，Aesara 只支持 2 个输入
            sympy.conjugate: aet.conj,
            sympy.core.numbers.ImaginaryUnit: lambda:aet.complex(0,1),
            # 矩阵运算
            sympy.MatAdd: Elemwise(aes.add),
            sympy.HadamardProduct: Elemwise(aes.mul),
            sympy.Trace: nlinalg.trace,
            sympy.Determinant : nlinalg.det,
            sympy.Inverse: nlinalg.matrix_inverse,
            sympy.Transpose: DimShuffle((False, False), [1, 0]),
    }


class AesaraPrinter(Printer):
    """ 创建 Aesara 符号表达式图的代码打印器。

    Parameters
    ==========

    """
    cache : dict
        # 用于缓存的字典。如果为 None（默认），将使用全局的缓存。为了创建不依赖或修改全局状态的打印机，传入一个空字典。
        # 注意：该字典在打印机初始化时不会被复制，而是在原地更新。因此，如果在创建多个打印机或多次调用 .aesara_code 或 .aesara_function 时使用相同的字典对象，
        # 则这些应用程序之间会共享缓存。

    Attributes
    ==========

    cache : dict
        # 缓存 Aesara 变量的字典，用于已创建的 SymPy 符号类似对象（例如 sympy.core.symbol.Symbol 或 sympy.matrices.expressions.MatrixSymbol）。
        # 这用于确保表达式中对于给定符号的所有引用都被打印为同一个 Aesara 变量，该变量仅创建一次。符号仅通过名称和类型进行区分。
        # 缓存内容的格式对用户来说应该是不透明的。
    """
    printmethod = "_aesara"

    def __init__(self, *args, **kwargs):
        # 初始化方法，接受任意位置参数和关键字参数
        # 使用 kwargs.pop('cache', {}) 获取 cache 参数，如果不存在则使用空字典 {}
        self.cache = kwargs.pop('cache', {})
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

    def _get_key(self, s, name=None, dtype=None, broadcastable=None):
        """ Get the cache key for a SymPy object.

        Parameters
        ==========

        s : sympy.core.basic.Basic
            SymPy object to get key for.

        name : str
            Name of object, if it does not have a ``name`` attribute.
        """
        # 获取 SymPy 对象 s 的缓存键值

        if name is None:
            name = s.name

        return (name, type(s), s.args, dtype, broadcastable)

    def _get_or_create(self, s, name=None, dtype=None, broadcastable=None):
        """
        Get the Aesara variable for a SymPy symbol from the cache, or create it
        if it does not exist.
        """
        # 从缓存中获取或创建 SymPy 符号 s 对应的 Aesara 变量

        # 默认值设置
        if name is None:
            name = s.name
        if dtype is None:
            dtype = 'floatX'
        if broadcastable is None:
            broadcastable = ()

        key = self._get_key(s, name, dtype=dtype, broadcastable=broadcastable)

        if key in self.cache:
            return self.cache[key]

        # 创建新的 Aesara 变量，并将其存入缓存
        value = aet.tensor(name=name, dtype=dtype, shape=broadcastable)
        self.cache[key] = value
        return value

    def _print_Symbol(self, s, **kwargs):
        # 处理对 SymPy 符号类的打印方法

        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, dtype=dtype, broadcastable=bc)

    def _print_AppliedUndef(self, s, **kwargs):
        # 处理对 AppliedUndef 类的打印方法

        name = str(type(s)) + '_' + str(s.args[0])
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, name=name, dtype=dtype, broadcastable=bc)
    # 定义方法_print_Basic，根据表达式类型映射找到相应的操作符
    def _print_Basic(self, expr, **kwargs):
        # 从映射中获取操作符
        op = mapping[type(expr)]
        # 对表达式中的每个参数调用_print方法，返回结果列表
        children = [self._print(arg, **kwargs) for arg in expr.args]
        # 使用操作符将参数列表组装成结果
        return op(*children)

    # 定义方法_print_Number，将表达式解释为浮点数
    def _print_Number(self, n, **kwargs):
        # 如果表达式是整数，在下面已经处理过了，这里直接解释为浮点数
        return float(n.evalf())

    # 定义方法_print_MatrixSymbol，根据参数获取或创建矩阵符号
    def _print_MatrixSymbol(self, X, **kwargs):
        # 获取数据类型，如果未指定则为None
        dtype = kwargs.get('dtypes', {}).get(X)
        # 调用方法_get_or_create，获取或创建指定矩阵符号的对象
        return self._get_or_create(X, dtype=dtype, broadcastable=(None, None))

    # 定义方法_print_DenseMatrix，将密集矩阵转换为Aesara的对象表示
    def _print_DenseMatrix(self, X, **kwargs):
        # 检查aet模块是否有stacklists属性，如果没有抛出NotImplementedError异常
        if not hasattr(aet, 'stacklists'):
            raise NotImplementedError(
               "Matrix translation not yet supported in this version of Aesara")
        # 将密集矩阵转换为列表形式，然后使用aet.stacklists创建Aesara对象
        return aet.stacklists([
            [self._print(arg, **kwargs) for arg in L]
            for L in X.tolist()
        ])

    # 将不可变矩阵的打印方法_print_ImmutableMatrix和_print_ImmutableDenseMatrix重定向到_print_DenseMatrix
    _print_ImmutableMatrix = _print_ImmutableDenseMatrix = _print_DenseMatrix

    # 定义方法_print_MatMul，实现矩阵乘法的打印表示
    def _print_MatMul(self, expr, **kwargs):
        # 对表达式中的每个参数调用_print方法，获取参数列表
        children = [self._print(arg, **kwargs) for arg in expr.args]
        # 将第一个参数作为初始结果
        result = children[0]
        # 对剩余的参数进行矩阵乘法操作，更新结果
        for child in children[1:]:
            result = aet.dot(result, child)
        # 返回最终结果
        return result

    # 定义方法_print_MatPow，实现矩阵的幂运算的打印表示
    def _print_MatPow(self, expr, **kwargs):
        # 对表达式中的每个参数调用_print方法，获取参数列表
        children = [self._print(arg, **kwargs) for arg in expr.args]
        # 初始结果设为1
        result = 1
        # 如果第二个参数是非负整数，则进行相应次数的矩阵乘法
        if isinstance(children[1], int) and children[1] > 0:
            for i in range(children[1]):
                result = aet.dot(result, children[0])
        else:
            # 抛出NotImplementedError异常，提示Aesara目前只支持非负整数幂次
            raise NotImplementedError('''Only non-negative integer
           powers of matrices can be handled by Aesara at the moment''')
        # 返回最终结果
        return result

    # 定义方法_print_MatrixSlice，实现矩阵切片的打印表示
    def _print_MatrixSlice(self, expr, **kwargs):
        # 获取父矩阵的打印表示
        parent = self._print(expr.parent, **kwargs)
        # 获取行切片和列切片的打印表示
        rowslice = self._print(slice(*expr.rowslice), **kwargs)
        colslice = self._print(slice(*expr.colslice), **kwargs)
        # 返回父矩阵的指定切片
        return parent[rowslice, colslice]

    # 定义方法_print_BlockMatrix，实现块矩阵的打印表示
    def _print_BlockMatrix(self, expr, **kwargs):
        # 获取块矩阵的行数和列数
        nrows, ncols = expr.blocks.shape
        # 获取块矩阵中每个块的打印表示，组成二维列表
        blocks = [[self._print(expr.blocks[r, c], **kwargs)
                        for c in range(ncols)]
                        for r in range(nrows)]
        # 使用aet.join函数按指定维度将块矩阵的块连接起来
        return aet.join(0, *[aet.join(1, *row) for row in blocks])

    # 定义方法_print_slice，实现切片对象的打印表示
    def _print_slice(self, expr, **kwargs):
        # 将切片对象的起始、终止和步长参数分别调用_print方法获取其打印表示
        return slice(*[self._print(i, **kwargs)
                        if isinstance(i, sympy.Basic) else i
                        for i in (expr.start, expr.stop, expr.step)])

    # 定义方法_print_Pi，返回圆周率π的值
    def _print_Pi(self, expr, **kwargs):
        return 3.141592653589793
    # 定义一个方法来打印 Piecewise 表达式
    def _print_Piecewise(self, expr, **kwargs):
        import numpy as np
        # 获取第一个条件和对应的值
        e, cond = expr.args[0].args

        # 打印第一个条件和其对应的值
        p_cond = self._print(cond, **kwargs)
        p_e = self._print(e, **kwargs)

        # 如果只有一个条件
        if len(expr.args) == 1:
            # 如果条件成立，返回值；否则返回 NaN
            return aet.switch(p_cond, p_e, np.nan)

        # 返回第一个条件成立时的值 p_e，否则继续评估剩余的条件
        p_remaining = self._print(sympy.Piecewise(*expr.args[1:]), **kwargs)
        return aet.switch(p_cond, p_e, p_remaining)

    # 定义一个方法来打印有理数 Rational
    def _print_Rational(self, expr, **kwargs):
        return true_divide(self._print(expr.p, **kwargs),
                           self._print(expr.q, **kwargs))

    # 定义一个方法来打印整数 Integer
    def _print_Integer(self, expr, **kwargs):
        return expr.p

    # 定义一个方法来打印阶乘 factorial
    def _print_factorial(self, expr, **kwargs):
        # 转换为 gamma 函数的打印形式
        return self._print(sympy.gamma(expr.args[0] + 1), **kwargs)

    # 定义一个方法来打印 Derivative（导数）对象
    def _print_Derivative(self, deriv, **kwargs):
        from aesara.gradient import Rop

        # 打印导数表达式
        rv = self._print(deriv.expr, **kwargs)
        # 遍历导数变量
        for var in deriv.variables:
            var = self._print(var, **kwargs)
            # 构建 Rop（逆模式自动微分）
            rv = Rop(rv, var, aet.ones_like(var))
        return rv

    # 定义一个空的打印方法，直接返回表达式本身
    def emptyPrinter(self, expr):
        return expr
    def doprint(self, expr, dtypes=None, broadcastables=None):
        """ Convert a SymPy expression to a Aesara graph variable.

        The ``dtypes`` and ``broadcastables`` arguments are used to specify the
        data type, dimension, and broadcasting behavior of the Aesara variables
        corresponding to the free symbols in ``expr``. Each is a mapping from
        SymPy symbols to the value of the corresponding argument to
        ``aesara.tensor.var.TensorVariable``.

        See the corresponding `documentation page`__ for more information on
        broadcasting in Aesara.

        .. __: https://aesara.readthedocs.io/en/latest/reference/tensor/shapes.html#broadcasting

        Parameters
        ==========

        expr : sympy.core.expr.Expr
            SymPy expression to print.

        dtypes : dict, optional
            Mapping from SymPy symbols to Aesara datatypes to use when creating
            new Aesara variables for those symbols. Corresponds to the ``dtype``
            argument to ``aesara.tensor.var.TensorVariable``. Defaults to ``'floatX'``
            for symbols not included in the mapping.

        broadcastables : dict, optional
            Mapping from SymPy symbols to the value of the ``broadcastable``
            argument to ``aesara.tensor.var.TensorVariable`` to use when creating Aesara
            variables for those symbols. Defaults to the empty tuple for symbols
            not included in the mapping (resulting in a scalar).

        Returns
        =======

        aesara.graph.basic.Variable
            A variable corresponding to the expression's value in a Aesara
            symbolic expression graph.

        """
        # 如果 dtypes 参数未指定，则将其设置为空字典
        if dtypes is None:
            dtypes = {}
        # 如果 broadcastables 参数未指定，则将其设置为空字典
        if broadcastables is None:
            broadcastables = {}

        # 调用内部方法 _print，将 SymPy 表达式转换为 Aesara 图形变量
        return self._print(expr, dtypes=dtypes, broadcastables=broadcastables)
# 定义一个全局的缓存字典，用于存储 Aesara 变量以提高效率
global_cache: dict[Any, Any] = {}

# 将 SymPy 表达式转换为 Aesara 图变量
def aesara_code(expr, cache=None, **kwargs):
    """
    Convert a SymPy expression into a Aesara graph variable.

    Parameters
    ==========

    expr : sympy.core.expr.Expr
        SymPy expression object to convert.

    cache : dict, optional
        Cached Aesara variables (see :class:`AesaraPrinter.cache
        <AesaraPrinter>`). Defaults to the module-level global cache.

    dtypes : dict, optional
        Passed to :meth:`.AesaraPrinter.doprint`.

    broadcastables : dict, optional
        Passed to :meth:`.AesaraPrinter.doprint`.

    Returns
    =======

    aesara.graph.basic.Variable
        A variable corresponding to the expression's value in a Aesara symbolic
        expression graph.

    """
    # 检查是否导入了 aesara 模块，如果没有则抛出 ImportError 异常
    if not aesara:
        raise ImportError("aesara is required for aesara_code")

    # 如果未提供缓存参数，则使用全局缓存
    if cache is None:
        cache = global_cache

    # 创建 AesaraPrinter 对象并调用 doprint 方法进行表达式转换
    return AesaraPrinter(cache=cache, settings={}).doprint(expr, **kwargs)


# 获取 broadcastables 参数的值，用于向 aesara_code 函数传递
def dim_handling(inputs, dim=None, dims=None, broadcastables=None):
    r"""
    Get value of ``broadcastables`` argument to :func:`.aesara_code` from
    keyword arguments to :func:`.aesara_function`.

    Included for backwards compatibility.

    Parameters
    ==========

    inputs
        Sequence of input symbols.

    dim : int, optional
        Common number of dimensions for all inputs. Overrides other arguments
        if given.

    dims : dict, optional
        Mapping from input symbols to number of dimensions. Overrides
        ``broadcastables`` argument if given.

    broadcastables : dict, optional
        Explicit value of ``broadcastables`` argument to
        :meth:`.AesaraPrinter.doprint`. If not None function will return this value unchanged.

    Returns
    =======
    dict
        Dictionary mapping elements of ``inputs`` to their "broadcastable"
        values (tuple of ``bool``\ s).
    """
    # 如果提供了 dim 参数，则将所有输入的 broadcastable 设为 (False,) * dim
    if dim is not None:
        return dict.fromkeys(inputs, (False,) * dim)

    # 如果提供了 dims 参数，则根据 dims 中每个符号的维度设置 broadcastable
    if dims is not None:
        maxdim = max(dims.values())
        return {
            s: (False,) * d + (True,) * (maxdim - d)
            for s, d in dims.items()
        }

    # 如果提供了 broadcastables 参数，则直接返回其值
    if broadcastables is not None:
        return broadcastables

    # 默认情况下返回空字典
    return {}


# 创建一个 Aesara 函数，从 SymPy 表达式生成
def aesara_function(inputs, outputs, scalar=False, *,
        dim=None, dims=None, broadcastables=None, **kwargs):
    """
    Create a Aesara function from SymPy expressions.

    The inputs and outputs are converted to Aesara variables using
    :func:`.aesara_code` and then passed to ``aesara.function``.

    Parameters
    ==========

    inputs
        Sequence of symbols which constitute the inputs of the function.

    outputs
        Sequence of expressions which constitute the outputs(s) of the
        function. The free symbols of each expression must be a subset of
        ``inputs``.

    scalar : bool, optional
        Convert 0-dimensional arrays in output to scalars. This will return a
        Python wrapper function around the Aesara function object.
    """
    # 函数主体未提供代码，只有文档字符串描述其功能
    # 如果没有导入 aesara 模块，则抛出 ImportError 异常
    if not aesara:
        raise ImportError("Aesara is required for aesara_function")

    # 从关键字参数中弹出非 aesara 相关的参数
    cache = kwargs.pop('cache', {})
    dtypes = kwargs.pop('dtypes', {})
    
    # 使用 dim_handling 函数处理输入的维度信息，得到广播标志信息
    broadcastables = dim_handling(
        inputs, dim=dim, dims=dims, broadcastables=broadcastables,
    )

    # 使用 aesara_code 部分函数生成输入和输出的代码
    code = partial(aesara_code, cache=cache, dtypes=dtypes,
                   broadcastables=broadcastables)
    
    # 将 inputs 中的每个元素转换为代码形式
    tinputs = list(map(code, inputs))
    
    # 将 outputs 中的每个元素转换为代码形式
    toutputs = list(map(code, outputs))

    # 如果输出不是 aesara 的变量，则将其转换为张量变量
    toutputs = [output if isinstance(output, aesara.graph.basic.Variable) else aet.as_tensor_variable(output) for output in toutputs]
    # 如果 toutputs 只有一个元素，将其解包赋值给 toutputs
    if len(toutputs) == 1:
        toutputs = toutputs[0]

    # 使用 aesara 库的 function 函数编译函数 func，接受 tinputs 作为输入，toutputs 作为输出，kwargs 是其他关键字参数
    func = aesara.function(tinputs, toutputs, **kwargs)

    # 检查 func 的输出是否为零维张量（scalar）
    is_0d = [len(o.variable.broadcastable) == 0 for o in func.outputs]

    # 如果不需要包装器（wrapper），或者 func 的输出不包含任何零维张量（scalar）
    if not scalar or not any(is_0d):
        # 将 func 本身设置为其 aesara_function 属性，然后直接返回 func
        func.aesara_function = func
        return func

    # 创建一个包装器函数，将零维输出转换为标量
    def wrapper(*args):
        # 调用 func 函数计算输出 out
        out = func(*args)
        # out 可能是数组(1.0)或者 [数组(1.0), 数组(2.0)] 的形式

        # 如果 out 是一个序列（列表或元组）
        if is_sequence(out):
            # 遍历 out 中的每个元素，如果是零维张量，将其转换为标量（scalar），否则保持不变
            return [o[()] if is_0d[i] else o for i, o in enumerate(out)]
        else:
            # 如果 out 是单个元素的情况，直接将其转换为标量返回
            return out[()]

    # 将 wrapper 函数的 __wrapped__ 属性设置为 func，保留原始函数的引用
    wrapper.__wrapped__ = func
    # 将 wrapper 函数的文档字符串（docstring）设置为 func 的文档字符串
    wrapper.__doc__ = func.__doc__
    # 将 wrapper 函数的 aesara_function 属性设置为 func，保留 func 的引用
    wrapper.aesara_function = func
    # 返回创建的 wrapper 函数作为结果
    return wrapper
```
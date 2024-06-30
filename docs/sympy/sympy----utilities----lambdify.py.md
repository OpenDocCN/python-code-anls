# `D:\src\scipysrc\sympy\sympy\utilities\lambdify.py`

```
"""
This module provides convenient functions to transform SymPy expressions to
lambda functions which can be used to calculate numerical values very fast.
"""

# 从__future__导入annotations，用于支持类型注解
from __future__ import annotations
# 导入Any用于指定类型模糊的参数类型
from typing import Any

# 导入builtins模块，用于获取Python内置对象
import builtins
# 导入inspect模块，用于检查源码
import inspect
# 导入keyword模块，用于检查Python关键字
import keyword
# 导入textwrap模块，用于格式化文本块
import textwrap
# 导入linecache模块，用于获取行缓存数据
import linecache

# 从sympy.external导入import_module，用于导入外部模块
# 忽略F401类型的导入警告
from sympy.external import import_module  # noqa:F401
# 从sympy.utilities.exceptions导入sympy_deprecation_warning，用于发出sympy的弃用警告
from sympy.utilities.exceptions import sympy_deprecation_warning
# 从sympy.utilities.decorator导入doctest_depends_on，用于依赖doctest的装饰器
from sympy.utilities.decorator import doctest_depends_on
# 从sympy.utilities.iterables导入is_sequence, iterable, NotIterable, flatten，用于处理可迭代对象
from sympy.utilities.iterables import (is_sequence, iterable,
    NotIterable, flatten)
# 从sympy.utilities.misc导入filldedent，用于填充文本缩进
from sympy.utilities.misc import filldedent

# 指定lambdify函数需要的外部依赖
__doctest_requires__ = {('lambdify',): ['numpy', 'tensorflow']}


# 默认命名空间，用于定义不能通过简单变量映射定义的转换，例如I => 1j
MATH_DEFAULT: dict[str, Any] = {}
MPMATH_DEFAULT: dict[str, Any] = {}
NUMPY_DEFAULT: dict[str, Any] = {"I": 1j}
SCIPY_DEFAULT: dict[str, Any] = {"I": 1j}
CUPY_DEFAULT: dict[str, Any] = {"I": 1j}
JAX_DEFAULT: dict[str, Any] = {"I": 1j}
TENSORFLOW_DEFAULT: dict[str, Any] = {}
SYMPY_DEFAULT: dict[str, Any] = {}
NUMEXPR_DEFAULT: dict[str, Any] = {}

# 这些是lambda函数将使用的命名空间。
# 这些与上述名称分开，因为它们在本文件中被修改，而默认值应保持不变。

MATH = MATH_DEFAULT.copy()
MPMATH = MPMATH_DEFAULT.copy()
NUMPY = NUMPY_DEFAULT.copy()
SCIPY = SCIPY_DEFAULT.copy()
CUPY = CUPY_DEFAULT.copy()
JAX = JAX_DEFAULT.copy()
TENSORFLOW = TENSORFLOW_DEFAULT.copy()
SYMPY = SYMPY_DEFAULT.copy()
NUMEXPR = NUMEXPR_DEFAULT.copy()


# SymPy和其他模块函数名称之间的映射。
MATH_TRANSLATIONS = {
    "ceiling": "ceil",
    "E": "e",
    "ln": "log",
}

# 注意：这个字典在Function._eval_evalf中被重用，以允许Function的子类自动evalf。
MPMATH_TRANSLATIONS = {
    "Abs": "fabs",
    "elliptic_k": "ellipk",
    "elliptic_f": "ellipf",
    "elliptic_e": "ellipe",
    "elliptic_pi": "ellippi",
    "ceiling": "ceil",
    "chebyshevt": "chebyt",
    "chebyshevu": "chebyu",
    "assoc_legendre": "legenp",
    "E": "e",
    "I": "j",
    "ln": "log",
    #"lowergamma":"lower_gamma",
    "oo": "inf",
    #"uppergamma":"upper_gamma",
    "LambertW": "lambertw",
    "MutableDenseMatrix": "matrix",
    "ImmutableDenseMatrix": "matrix",
    "conjugate": "conj",
    "dirichlet_eta": "altzeta",
    "Ei": "ei",
    "Shi": "shi",
    "Chi": "chi",
    "Si": "si",
    "Ci": "ci",
    "RisingFactorial": "rf",
    "FallingFactorial": "ff",
    "betainc_regularized": "betainc",
}

NUMPY_TRANSLATIONS: dict[str, str] = {
    "Heaviside": "heaviside",
}
SCIPY_TRANSLATIONS: dict[str, str] = {
    "jn" : "spherical_jn",
    "yn" : "spherical_yn"
}
CUPY_TRANSLATIONS: dict[str, str] = {}
JAX_TRANSLATIONS: dict[str, str] = {}

TENSORFLOW_TRANSLATIONS: dict[str, str] = {}

NUMEXPR_TRANSLATIONS: dict[str, str] = {}
# 可用模块的配置字典，每个键值对对应一个模块及其相关信息
MODULES = {
    "math": (MATH, MATH_DEFAULT, MATH_TRANSLATIONS, ("from math import *",)),
    "mpmath": (MPMATH, MPMATH_DEFAULT, MPMATH_TRANSLATIONS, ("from mpmath import *",)),
    "numpy": (NUMPY, NUMPY_DEFAULT, NUMPY_TRANSLATIONS, ("import numpy; from numpy import *; from numpy.linalg import *",)),
    "scipy": (SCIPY, SCIPY_DEFAULT, SCIPY_TRANSLATIONS, ("import scipy; import numpy; from scipy.special import *",)),
    "cupy": (CUPY, CUPY_DEFAULT, CUPY_TRANSLATIONS, ("import cupy",)),
    "jax": (JAX, JAX_DEFAULT, JAX_TRANSLATIONS, ("import jax",)),
    "tensorflow": (TENSORFLOW, TENSORFLOW_DEFAULT, TENSORFLOW_TRANSLATIONS, ("import tensorflow",)),
    "sympy": (SYMPY, SYMPY_DEFAULT, {}, (
        "from sympy.functions import *",
        "from sympy.matrices import *",
        "from sympy import Integral, pi, oo, nan, zoo, E, I",)),
    "numexpr" : (NUMEXPR, NUMEXPR_DEFAULT, NUMEXPR_TRANSLATIONS,
                 ("import_module('numexpr')", )),
}

def _import(module, reload=False):
    """
    创建一个全局翻译字典以供模块使用。

    参数 module 必须是以下字符串之一："math", "mpmath", "numpy", "sympy", "tensorflow", "jax"。
    这些字典将 Python 函数名映射到其他模块中的等效函数名。
    """
    try:
        # 尝试获取指定模块的相关信息
        namespace, namespace_default, translations, import_commands = MODULES[
            module]
    except KeyError:
        # 如果指定的模块不存在于 MODULES 字典中，则抛出 NameError 异常
        raise NameError(
            "'%s' module cannot be used for lambdification" % module)

    # 清空命名空间或退出
    if namespace != namespace_default:
        # 如果命名空间已经生成过，且不是强制重新加载，则直接返回
        if reload:
            # 如果强制重新加载，则清空命名空间并更新为默认状态
            namespace.clear()
            namespace.update(namespace_default)
        else:
            return

    # 遍历导入命令列表，并执行导入操作
    for import_command in import_commands:
        if import_command.startswith('import_module'):
            # 如果导入命令以 'import_module' 开头，则使用 eval 执行
            module = eval(import_command)

            if module is not None:
                # 如果模块不为空，则更新命名空间
                namespace.update(module.__dict__)
                continue
        else:
            try:
                # 尝试执行一般的导入命令
                exec(import_command, {}, namespace)
                continue
            except ImportError:
                pass

        # 如果导入失败，则抛出 ImportError 异常
        raise ImportError(
            "Cannot import '%s' with '%s' command" % (module, import_command))

    # 将翻译后的名称添加到命名空间中
    for sympyname, translation in translations.items():
        namespace[sympyname] = namespace[translation]

    # 对于计算 SymPy 表达式的模数，使用内置的 abs 函数，而不是之前所有翻译模块中使用的 fabs 函数。
    # 这是因为 math 模块中的 fabs 函数不接受复数参数（参见问题 9474）。
    # 唯一的例外是 mpmath 翻译模块，因为 mpmath.fabs 返回 mpf 对象，与 abs() 不同。
    # 如果在命名空间中不存在 'Abs' 键
    if 'Abs' not in namespace:
        # 将内置函数 abs 赋给 'Abs' 键
        namespace['Abs'] = abs
# 用于动态生成的文件名，插入到 linecache 中。
_lambdify_generated_counter = 1

# 基于特定条件执行 doctest 的装饰器，依赖于 numpy、scipy 和 tensorflow 模块，仅支持 Python 3 及以上版本。
@doctest_depends_on(modules=('numpy', 'scipy', 'tensorflow',), python_version=(3,))
def lambdify(args, expr, modules=None, printer=None, use_imps=True,
             dummify=False, cse=False, docstring_limit=1000):
    """Convert a SymPy expression into a function that allows for fast
    numeric evaluation.

    .. warning::
       This function uses ``exec``, and thus should not be used on
       unsanitized input.

    .. deprecated:: 1.7
       Passing a set for the *args* parameter is deprecated as sets are
       unordered. Use an ordered iterable such as a list or tuple.

    Explanation
    ===========

    For example, to convert the SymPy expression ``sin(x) + cos(x)`` to an
    equivalent NumPy function that numerically evaluates it:

    >>> from sympy import sin, cos, symbols, lambdify
    >>> import numpy as np
    >>> x = symbols('x')
    >>> expr = sin(x) + cos(x)
    >>> expr
    sin(x) + cos(x)
    >>> f = lambdify(x, expr, 'numpy')
    >>> a = np.array([1, 2])
    >>> f(a)
    [1.38177329 0.49315059]

    The primary purpose of this function is to provide a bridge from SymPy
    expressions to numerical libraries such as NumPy, SciPy, NumExpr, mpmath,
    and tensorflow. In general, SymPy functions do not work with objects from
    other libraries, such as NumPy arrays, and functions from numeric
    libraries like NumPy or mpmath do not work on SymPy expressions.
    ``lambdify`` bridges the two by converting a SymPy expression to an
    equivalent numeric function.

    The basic workflow with ``lambdify`` is to first create a SymPy expression
    representing whatever mathematical function you wish to evaluate. This
    should be done using only SymPy functions and expressions. Then, use
    ``lambdify`` to convert this to an equivalent function for numerical
    evaluation. For instance, above we created ``expr`` using the SymPy symbol
    ``x`` and SymPy functions ``sin`` and ``cos``, then converted it to an
    equivalent NumPy function ``f``, and called it on a NumPy array ``a``.

    Parameters
    ==========

    args : Symbol or iterable of Symbols
        The arguments to the function.
    expr : Expr
        The expression to be converted into a numerical function.
    modules : str or list or tuple, optional
        The library or libraries to target for numerical evaluation (e.g., 'numpy').
    printer : Printer, optional
        The printer to use for printing the expression.
    use_imps : bool, optional
        Whether to use 'imps' in the generated code.
    dummify : bool, optional
        Whether to automatically dummify unspecified functions.
    cse : bool, optional
        Whether to use common subexpression elimination (CSE).
    docstring_limit : int, optional
        The character limit for docstrings in the generated function.

    Returns
    =======

    callable
        A callable function that can be used to evaluate the expression numerically.

    Notes
    =====

    This function converts a SymPy expression into a function that can be
    efficiently evaluated numerically using libraries like NumPy. It should
    only be used with trusted input due to its use of ``exec``.

    """
    args : List[Symbol]
        # 定义参数列表args，其中包含的符号(Symbol)表示参数的嵌套结构，将传递给函数的参数也按照这种嵌套结构组织。

        Variables can be symbols, undefined functions, or matrix symbols.
        # 变量可以是符号(Symbol)，未定义的函数或矩阵符号。

        >>> from sympy import Eq
        >>> from sympy.abc import x, y, z

        The list of variables should match the structure of how the
        arguments will be passed to the function. Simply enclose the
        parameters as they will be passed in a list.
        # 变量列表应与将传递给函数的参数的结构匹配。简单地将参数按照它们将被传递的方式封装在一个列表中。

        To call a function like ``f(x)`` then ``[x]``
        should be the first argument to ``lambdify``; for this
        case a single ``x`` can also be used:
        # 要调用像``f(x)``这样的函数，那么``[x]``应该是``lambdify``的第一个参数；在这种情况下，也可以使用单个``x``：

        >>> f = lambdify(x, x + 1)
        >>> f(1)
        2
        >>> f = lambdify([x], x + 1)
        >>> f(1)
        2

        To call a function like ``f(x, y)`` then ``[x, y]`` will
        be the first argument of the ``lambdify``:
        # 要调用像``f(x, y)``这样的函数，那么``[x, y]``将是``lambdify``的第一个参数：

        >>> f = lambdify([x, y], x + y)
        >>> f(1, 1)
        2

        To call a function with a single 3-element tuple like
        ``f((x, y, z))`` then ``[(x, y, z)]`` will be the first
        argument of the ``lambdify``:
        # 要调用一个带有单个3元组的函数，如``f((x, y, z))``，那么``[(x, y, z)]``将是``lambdify``的第一个参数：

        >>> f = lambdify([(x, y, z)], Eq(z**2, x**2 + y**2))
        >>> f((3, 4, 5))
        True

        If two args will be passed and the first is a scalar but
        the second is a tuple with two arguments then the items
        in the list should match that structure:
        # 如果将传递两个参数，并且第一个是标量，但第二个是具有两个参数的元组，则列表中的项目应与该结构匹配：

        >>> f = lambdify([x, (y, z)], x + y + z)
        >>> f(1, (2, 3))
        6

    expr : Expr
        # 定义表达式expr，可以是一个表达式、表达式列表或矩阵，用于求值。

        Lists may be nested.
        # 列表可以是嵌套的。

        If the expression is a list, the output will also be a list.
        # 如果表达式是一个列表，则输出也将是一个列表。

        >>> f = lambdify(x, [x, [x + 1, x + 2]])
        >>> f(1)
        [1, [2, 3]]

        If it is a matrix, an array will be returned (for the NumPy module).
        # 如果是一个矩阵，则将返回一个数组（适用于NumPy模块）。

        >>> from sympy import Matrix
        >>> f = lambdify(x, Matrix([x, x + 1]))
        >>> f(1)
        [[1]
        [2]]

        Note that the argument order here (variables then expression) is used
        to emulate the Python ``lambda`` keyword. ``lambdify(x, expr)`` works
        (roughly) like ``lambda x: expr``
        (see :ref:`lambdify-how-it-works` below).
        # 请注意，这里的参数顺序（变量然后表达式）用于模拟Python中的``lambda``关键字。``lambdify(x, expr)``的工作方式大致类似于``lambda x: expr``
        # （参见下面的：ref:`lambdify-how-it-works`）。
    # modules 参数，指定要使用的数值计算库
    # 可选参数，默认情况下是以下列表：
    # - ``["scipy", "numpy"]`` 如果安装了 SciPy
    # - ``["numpy"]`` 如果只安装了 NumPy
    # - ``["math", "mpmath", "sympy"]`` 如果两者都未安装
    # 这意味着，如果安装了 SciPy 或 NumPy，SymPy 函数将尽可能地被替换为相应库中的函数；
    # 否则将使用 Python 标准库的 ``math`` 或 ``mpmath`` 函数。
    # modules 可以是以下类型之一：
    # - 字符串 ``"math"``, ``"mpmath"``, ``"numpy"``, ``"numexpr"``, ``"scipy"``, ``"sympy"``, ``"tensorflow"`` 或 ``"jax"``
    #   这使用相应模块的打印机和命名空间映射。
    # - 一个模块（如 ``math``），这将使用模块的全局命名空间。
    #   如果模块是上述已知模块之一，则还将使用相应的打印机和命名空间映射
    #   （例如，``modules=numpy`` 等效于 ``modules="numpy"``）。
    # - 将 SymPy 函数的名称映射到任意函数的字典（例如，``{'sin': custom_sin}``）。
    # - 包含上述参数混合的列表，优先考虑先出现的条目
    #   （例如，使用 NumPy 模块但用自定义版本覆盖 ``sin`` 函数，可以使用 ``[{'sin': custom_sin}, 'numpy']``）。
    modules: str, optional
    
    # dummify 参数，指定是否将不是有效 Python 标识符的变量替换为虚拟符号
    # 可选参数，默认为 False。
    # 这允许像 ``Function('f')(t)`` 这样的未定义函数作为参数提供。
    # 默认情况下，只有变量不是有效 Python 标识符时才会进行虚拟化。
    # 设置 ``dummify=True`` 可以替换所有参数为虚拟符号
    # （如果 ``args`` 不是字符串），例如确保参数不会重新定义任何内置名称。
    dummify: bool, optional
    
    # cse 参数，指定是否对常见子表达式进行识别和预计算以提高大表达式的计算效率
    # 可选参数，默认为 False。
    # 当设置为 ``True`` 时，使用 ``sympy.simplify.cse`` 进行处理；
    # 否则（默认情况下），用户可以传递一个与 ``cse`` 签名匹配的函数。
    cse: bool, or callable, optional
    docstring_limit : int or None
        # 参数说明：控制自动生成的函数文档字符串中表达式的渲染限制
        When lambdifying large expressions, a significant proportion of the time
        spent inside ``lambdify`` is spent producing a string representation of
        the expression for use in the automatically generated docstring of the
        returned function. For expressions containing hundreds or more nodes the
        resulting docstring often becomes so long and dense that it is difficult
        to read. To reduce the runtime of lambdify, the rendering of the full
        expression inside the docstring can be disabled.

        # 当为 None 时，在文档字符串中渲染完整表达式；当为 0 或负整数时，在文档字符串中渲染省略号代替表达式
        When ``None``, the full expression is rendered in the docstring. When
        ``0`` or a negative ``int``, an ellipsis is rendering in the docstring
        instead of the expression.

        # 当为正整数时，若表达式中的节点数超过 docstring_limit，则在文档字符串中渲染省略号；否则正常渲染表达式字符串
        When a strictly positive ``int``, if the number of nodes in the expression exceeds ``docstring_limit`` an
        ellipsis is rendered in the docstring, otherwise a string representation
        of the expression is rendered as normal. The default is ``1000``.

    Examples
    ========

    >>> from sympy.utilities.lambdify import implemented_function
    >>> from sympy import sqrt, sin, Matrix
    >>> from sympy import Function
    >>> from sympy.abc import w, x, y, z

    # 示例：将 SymPy 表达式转换为可调用的函数
    >>> f = lambdify(x, x**2)
    >>> f(2)
    4

    # 示例：多个变量的 lambdify 示例
    >>> f = lambdify((x, y, z), [z, y, x])
    >>> f(1,2,3)
    [3, 2, 1]

    # 示例：对单个变量应用 lambdify 示例
    >>> f = lambdify(x, sqrt(x))
    >>> f(4)
    2.0

    # 示例：对多个变量应用 sin 函数的 lambdify 示例
    >>> f = lambdify((x, y), sin(x*y)**2)
    >>> f(0, 5)
    0.0

    # 示例：使用 lambdify 创建返回矩阵的函数
    >>> row = lambdify((x, y), Matrix((x, x + y)).T, modules='sympy')
    >>> row(1, 2)
    Matrix([[1, 3]])

    ``lambdify`` can be used to translate SymPy expressions into mpmath
    functions. This may be preferable to using ``evalf`` (which uses mpmath on
    the backend) in some cases.

    # 示例：将 SymPy 表达式转换为 mpmath 函数的 lambdify 示例
    >>> f = lambdify(x, sin(x), 'mpmath')
    >>> f(1)
    0.8414709848078965

    # 示例：处理元组参数的 lambdify 示例
    Tuple arguments are handled and the lambdified function should
    be called with the same type of arguments as were used to create
    the function:

    >>> f = lambdify((x, (y, z)), x + y)
    >>> f(1, (2, 4))
    3

    # 示例：使用 flatten 函数处理的 lambdify 示例
    The ``flatten`` function can be used to always work with flattened
    arguments:

    >>> from sympy.utilities.iterables import flatten
    >>> args = w, (x, (y, z))
    >>> vals = 1, (2, (3, 4))
    >>> f = lambdify(flatten(args), w + x + y + z)
    >>> f(*flatten(vals))
    10

    # 示例：使用 implemented_function 创建带有数值实现的 lambdify 示例
    Functions present in ``expr`` can also carry their own numerical
    implementations, in a callable attached to the ``_imp_`` attribute. This
    can be used with undefined functions using the ``implemented_function``
    factory:

    >>> f = implemented_function(Function('f'), lambda x: x+1)
    >>> func = lambdify(x, f(x))
    >>> func(4)
    5

    # 示例：在默认情况下，lambdify 总是优先使用 _imp_ 实现
    ``lambdify`` always prefers ``_imp_`` implementations to implementations
    in other namespaces, unless the ``use_imps`` input parameter is False.

    # 示例：与 Tensorflow 一起使用的 lambdify 示例
    Usage with Tensorflow:

    >>> import tensorflow as tf
    >>> from sympy import Max, sin, lambdify
    >>> from sympy.abc import x
    从 sympy 库中导入变量 x
    
    >>> f = Max(x, sin(x))
    创建一个 sympy 表达式 f，表示 x 和 sin(x) 中的最大值
    
    >>> func = lambdify(x, f, 'tensorflow')
    使用 lambdify 函数将 sympy 表达式转换为 TensorFlow 中的可执行函数
    
    After tensorflow v2, eager execution is enabled by default.
    If you want to get the compatible result across tensorflow v1 and v2
    as same as this tutorial, run this line.
    
    >>> tf.compat.v1.enable_eager_execution()
    在 TensorFlow v2 中，默认启用即时执行（eager execution）。
    如果需要在 TensorFlow v1 和 v2 中保持兼容性，可以运行此行代码。
    
    If you have eager execution enabled, you can get the result out
    immediately as you can use numpy.
    
    If you pass tensorflow objects, you may get an ``EagerTensor``
    object instead of value.
    
    >>> result = func(tf.constant(1.0))
    调用 func 函数计算给定输入的结果，这里输入为 TensorFlow 常量 1.0
    >>> print(result)
    打印 func 函数计算的结果，得到一个 TensorFlow 张量对象
    tf.Tensor(1.0, shape=(), dtype=float32)
    >>> print(result.__class__)
    打印结果对象的类型信息
    <class 'tensorflow.python.framework.ops.EagerTensor'>
    
    You can use ``.numpy()`` to get the numpy value of the tensor.
    
    >>> result.numpy()
    调用 .numpy() 方法获取张量的 numpy 值
    1.0
    
    >>> var = tf.Variable(2.0)
    创建一个 TensorFlow 变量 var，赋值为 2.0
    >>> result = func(var) # also works for tf.Variable and tf.Placeholder
    调用 func 函数计算给定变量 var 的结果，同样适用于 TensorFlow 变量和占位符
    >>> result.numpy()
    获取计算结果的 numpy 值
    2.0
    
    And it works with any shape array.
    
    >>> tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    创建一个形状为 (2, 2) 的 TensorFlow 常量张量 tensor
    >>> result = func(tensor)
    调用 func 函数计算给定张量 tensor 的结果
    >>> result.numpy()
    获取计算结果的 numpy 值
    [[1. 2.]
     [3. 4.]]
    
    Notes
    =====
    
    - For functions involving large array calculations, numexpr can provide a
      significant speedup over numpy. Please note that the available functions
      for numexpr are more limited than numpy but can be expanded with
      ``implemented_function`` and user defined subclasses of Function. If
      specified, numexpr may be the only option in modules. The official list
      of numexpr functions can be found at:
      https://numexpr.readthedocs.io/en/latest/user_guide.html#supported-functions
    
      对于涉及大型数组计算的函数，numexpr 可以比 numpy 提供显著的加速。请注意，
      numexpr 的可用函数比 numpy 更有限，但可以通过 ``implemented_function``
      和用户定义的 Function 子类进行扩展。如果指定，numexpr 可能是模块中唯一的选择。
      numexpr 官方函数列表可以在以下链接找到：
      https://numexpr.readthedocs.io/en/latest/user_guide.html#supported-functions
    
    - In the above examples, the generated functions can accept scalar
      values or numpy arrays as arguments.  However, in some cases
      the generated function relies on the input being a numpy array:
    
      在上面的示例中，生成的函数可以接受标量值或 numpy 数组作为参数。但是，在某些情况下，
      生成的函数依赖于输入是 numpy 数组：
    
      >>> import numpy
      >>> from sympy import Piecewise
      >>> from sympy.testing.pytest import ignore_warnings
      >>> f = lambdify(x, Piecewise((x, x <= 1), (1/x, x > 1)), "numpy")
    
      >>> with ignore_warnings(RuntimeWarning):
      ...     f(numpy.array([-1, 0, 1, 2]))
      [-1.   0.   1.   0.5]
    
      >>> f(0)
      Traceback (most recent call last):
          ...
      ZeroDivisionError: division by zero
    
      In such cases, the input should be wrapped in a numpy array:
    
      在这种情况下，应该将输入包装在 numpy 数组中：
    
      >>> with ignore_warnings(RuntimeWarning):
      ...     float(f(numpy.array([0])))
      0.0
    
      Or if numpy functionality is not required another module can be used:
    
      或者如果不需要 numpy 功能，可以使用另一个模块：
    
      >>> f = lambdify(x, Piecewise((x, x <= 1), (1/x, x > 1)), "math")
      >>> f(0)
      0
    
    .. _lambdify-how-it-works:
    
    How it works
    ============
    
    When using this function, it helps a great deal to have an idea of what it
    is doing. At its core, lambdify is nothing more than a namespace
    translation, on top of a special printer that makes some corner cases work
    properly.
    
    使用此函数时，了解其工作原理非常有帮助。在其核心，lambdify 只不过是一个命名空间的转换，
    在一个特殊的打印机之上，使一些边缘情况能够正常工作。
    # 导入 sympy 和 numpy 库
    import sympy
    import numpy
    
    # 创建包含 sympy.sin 和 sympy.cos 函数的全局变量字典
    module_dictionary = {'sin': sympy.sin, 'cos': sympy.cos}
    
    # 使用 exec 函数执行字符串中的代码，并将全局变量字典传递进去模拟模块的全局命名空间
    exec('''
    def sin_cos(x):
        return sin(x) + cos(x)
    ''', module_dictionary)
    
    # 从字典中获取执行后的 sin_cos 函数
    sin_cos = module_dictionary['sin_cos']
    
    # 调用 sin_cos 函数，输出计算结果
    sin_cos(1)
    # 解释 lambdify 函数如何工作的说明文档
    So now we can get an idea of how ``lambdify`` works. The name "lambdify"
    comes from the fact that we can think of something like ``lambdify(x,
    sin(x) + cos(x), 'numpy')`` as ``lambda x: sin(x) + cos(x)``, where
    ``sin`` and ``cos`` come from the ``numpy`` namespace. This is also why
    the symbols argument is first in ``lambdify``, as opposed to most SymPy
    functions where it comes after the expression: to better mimic the
    ``lambda`` keyword.
    
    ``lambdify`` takes the input expression (like ``sin(x) + cos(x)``) and
    
    1. Converts it to a string
    2. Creates a module globals dictionary based on the modules that are
       passed in (by default, it uses the NumPy module)
    3. Creates the string ``"def func({vars}): return {expr}"``, where ``{vars}`` is the
       list of variables separated by commas, and ``{expr}`` is the string
       created in step 1., then ``exec``s that string with the module globals
       namespace and returns ``func``.
    
    In fact, functions returned by ``lambdify`` support inspection. So you can
    see exactly how they are defined by using ``inspect.getsource``, or ``??`` if you
    are using IPython or the Jupyter notebook.
    
    >>> f = lambdify(x, sin(x) + cos(x))
    >>> import inspect
    >>> print(inspect.getsource(f))
    def _lambdifygenerated(x):
        return sin(x) + cos(x)
    
    This shows us the source code of the function, but not the namespace it
    was defined in. We can inspect that by looking at the ``__globals__``
    attribute of ``f``:
    
    >>> f.__globals__['sin']
    <ufunc 'sin'>
    >>> f.__globals__['cos']
    <ufunc 'cos'>
    >>> f.__globals__['sin'] is numpy.sin
    True
    
    This shows us that ``sin`` and ``cos`` in the namespace of ``f`` will be
    ``numpy.sin`` and ``numpy.cos``.
    
    Note that there are some convenience layers in each of these steps, but at
    the core, this is how ``lambdify`` works. Step 1 is done using the
    ``LambdaPrinter`` printers defined in the printing module (see
    :mod:`sympy.printing.lambdarepr`). This allows different SymPy expressions
    to define how they should be converted to a string for different modules.
    You can change which printer ``lambdify`` uses by passing a custom printer
    in to the ``printer`` argument.
    
    Step 2 is augmented by certain translations. There are default
    translations for each module, but you can provide your own by passing a
    list to the ``modules`` argument. For instance,
    
    >>> def mysin(x):
    ...     print('taking the sin of', x)
    ...     return numpy.sin(x)
    ...
    >>> f = lambdify(x, sin(x), [{'sin': mysin}, 'numpy'])
    >>> f(1)
    taking the sin of 1
    0.8414709848078965
    
    The globals dictionary is generated from the list by merging the
    dictionary ``{'sin': mysin}`` and the module dictionary for NumPy. The
    merging is done so that earlier items take precedence, which is why
    ``mysin`` is used above instead of ``numpy.sin``.
    If you want to modify the way ``lambdify`` works for a given function, it
    is usually easiest to do so by modifying the globals dictionary as such.
    In more complicated cases, it may be necessary to create and pass in a
    custom printer.

    Finally, step 3 is augmented with certain convenience operations, such as
    the addition of a docstring.

    Understanding how ``lambdify`` works can make it easier to avoid certain
    gotchas when using it. For instance, a common mistake is to create a
    lambdified function for one module (say, NumPy), and pass it objects from
    another (say, a SymPy expression).

    For instance, say we create

    >>> from sympy.abc import x
    >>> f = lambdify(x, x + 1, 'numpy')

    Now if we pass in a NumPy array, we get that array plus 1

    >>> import numpy
    >>> a = numpy.array([1, 2])
    >>> f(a)
    [2 3]

    But what happens if you make the mistake of passing in a SymPy expression
    instead of a NumPy array:

    >>> f(x + 1)
    x + 2

    This worked, but it was only by accident. Now take a different lambdified
    function:

    >>> from sympy import sin
    >>> g = lambdify(x, x + sin(x), 'numpy')

    This works as expected on NumPy arrays:

    >>> g(a)
    [1.84147098 2.90929743]

    But if we try to pass in a SymPy expression, it fails

    >>> g(x + 1)
    Traceback (most recent call last):
    ...
    TypeError: loop of ufunc does not support argument 0 of type Add which has
               no callable sin method

    Now, let's look at what happened. The reason this fails is that ``g``
    calls ``numpy.sin`` on the input expression, and ``numpy.sin`` does not
    know how to operate on a SymPy object. **As a general rule, NumPy
    functions do not know how to operate on SymPy expressions, and SymPy
    functions do not know how to operate on NumPy arrays. This is why lambdify
    exists: to provide a bridge between SymPy and NumPy.**

    However, why is it that ``f`` did work? That's because ``f`` does not call
    any functions, it only adds 1. So the resulting function that is created,
    ``def _lambdifygenerated(x): return x + 1`` does not depend on the globals
    namespace it is defined in. Thus it works, but only by accident. A future
    version of ``lambdify`` may remove this behavior.

    Be aware that certain implementation details described here may change in
    future versions of SymPy. The API of passing in custom modules and
    printers will not change, but the details of how a lambda function is
    created may change. However, the basic idea will remain the same, and
    understanding it will be helpful to understanding the behavior of
    lambdify.

    **In general: you should create lambdified functions for one module (say,
    NumPy), and only pass it input types that are compatible with that module
    (say, NumPy arrays).** Remember that by default, if the ``module``
    argument is not provided, ``lambdify`` creates functions using the NumPy
    # 导入符号和表达式类别，从 sympy 核心模块
    from sympy.core.symbol import Symbol
    from sympy.core.expr import Expr

    # 如果用户没有指定模块，尝试导入 scipy 和 numpy
    if modules is None:
        try:
            _import("scipy")
        except ImportError:
            try:
                _import("numpy")
            except ImportError:
                # 如果都导入失败，使用 math 和 mpmath（如果可用）以及 sympy 模块
                # 注意：这可能导致在不同系统上行为不同，并可能是导致无法重现错误的原因
                modules = ["math", "mpmath", "sympy"]
            else:
                modules = ["numpy"]
        else:
            modules = ["numpy", "scipy"]

    # 准备存储命名空间的列表
    namespaces = []

    # 如果使用函数实现，将表达式添加到命名空间
    if use_imps:
        namespaces.append(_imp_namespace(expr))

    # 检查模块是否是字典或字符串，或者是否可迭代
    if isinstance(modules, (dict, str)) or not hasattr(modules, '__iter__'):
        # 如果是字典或字符串，或者不可迭代，直接将其作为命名空间之一
        namespaces.append(modules)
    else:
        # 否则将模块列表扩展到命名空间列表中
        namespaces += list(modules)

    # 如果模块中同时存在 numexpr 和其他模块，抛出类型错误
    if _module_present('numexpr', modules) and len(modules) > 1:
        raise TypeError("numexpr must be the only item in 'modules'")

    # 逆序遍历命名空间列表，依次获取命名空间并合并到一个字典中
    namespace = {}
    for m in namespaces[::-1]:
        buf = _get_namespace(m)
        namespace.update(buf)

    # 如果表达式具有 atoms 方法，则尝试从表达式中提取符号
    if hasattr(expr, "atoms"):
        # 尝试提取符号，并将符号添加到命名空间字典中
        syms = expr.atoms(Symbol)
        for term in syms:
            namespace.update({str(term): term})
    # 如果打印器为空，根据可用的第三方数学库选择合适的打印器
    if printer is None:
        # 检查是否存在 mpmath 模块，并导入对应的打印器 MpmathPrinter
        if _module_present('mpmath', namespaces):
            from sympy.printing.pycode import MpmathPrinter as Printer # type: ignore
        # 如果不存在 mpmath 模块，则检查 scipy 模块，并导入 SciPyPrinter 打印器
        elif _module_present('scipy', namespaces):
            from sympy.printing.numpy import SciPyPrinter as Printer # type: ignore
        # 如果不存在 scipy 模块，则检查 numpy 模块，并导入 NumPyPrinter 打印器
        elif _module_present('numpy', namespaces):
            from sympy.printing.numpy import NumPyPrinter as Printer # type: ignore
        # 如果不存在 numpy 模块，则检查 cupy 模块，并导入 CuPyPrinter 打印器
        elif _module_present('cupy', namespaces):
            from sympy.printing.numpy import CuPyPrinter as Printer # type: ignore
        # 如果不存在 cupy 模块，则检查 jax 模块，并导入 JaxPrinter 打印器
        elif _module_present('jax', namespaces):
            from sympy.printing.numpy import JaxPrinter as Printer # type: ignore
        # 如果不存在 jax 模块，则检查 numexpr 模块，并导入 NumExprPrinter 打印器
        elif _module_present('numexpr', namespaces):
            from sympy.printing.lambdarepr import NumExprPrinter as Printer # type: ignore
        # 如果不存在 numexpr 模块，则检查 tensorflow 模块，并导入 TensorflowPrinter 打印器
        elif _module_present('tensorflow', namespaces):
            from sympy.printing.tensorflow import TensorflowPrinter as Printer # type: ignore
        # 如果以上模块都不存在，则默认使用 SymPyPrinter 打印器
        elif _module_present('sympy', namespaces):
            from sympy.printing.pycode import SymPyPrinter as Printer # type: ignore
        else:
            from sympy.printing.pycode import PythonCodePrinter as Printer # type: ignore
        
        # 准备一个空字典，用于存储用户定义的函数
        user_functions = {}
        # 遍历命名空间列表的逆序，找出所有字典类型的模块，将其键值对存入 user_functions
        for m in namespaces[::-1]:
            if isinstance(m, dict):
                for k in m:
                    user_functions[k] = k
        
        # 使用选定的打印器初始化 printer 对象，配置打印选项
        printer = Printer({
            'fully_qualified_modules': False,  # 禁用完全限定模块名
            'inline': True,  # 打印器是否内联
            'allow_unknown_functions': True,  # 允许打印未知函数
            'user_functions': user_functions  # 用户自定义函数字典
        })

    # 如果 args 是集合类型，则发出 sympy 废弃警告
    if isinstance(args, set):
        sympy_deprecation_warning(
            """
    # 将函数参数作为集合传递给 lambdify() 已被弃用，因为集合是无序的，这会导致不可预测的结果。应改为使用列表或元组作为函数参数。
    """
    使用集合将函数参数传递给 lambdify() 已被弃用。因为集合是无序的，这可能导致不可预测的结果。
    取而代之，应该使用列表或元组作为函数参数。
    """
    deprecated_since_version = "1.6.3"
    active_deprecations_target = "deprecated-lambdify-arguments-set"

    # 获取参数的名称，用于创建文档字符串
    iterable_args = (args,) if isinstance(args, Expr) else args
    names = []

    # 获取调用帧，用于通过检查获取名称（如果需要的话）
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()  # type: ignore
    for n, var in enumerate(iterable_args):
        if hasattr(var, 'name'):
            names.append(var.name)
        else:
            # 变量是可迭代的。尝试通过检查调用帧获取名称。
            name_list = [var_name for var_name, var_val in callers_local_vars
                         if var_val is var]
            if len(name_list) == 1:
                names.append(name_list[0])
            else:
                # 无法确定地推断名称。将使用 arg_#。
                names.append('arg_' + str(n))

    # 创建函数定义代码并执行它
    funcname = '_lambdifygenerated'

    # 根据模块的存在性选择打印器
    if _module_present('tensorflow', namespaces):
        funcprinter = _TensorflowEvaluatorPrinter(printer, dummify)
    else:
        funcprinter = _EvaluatorPrinter(printer, dummify)

    # 如果启用 CSE（公共子表达式消除），进行相关处理
    if cse == True:
        from sympy.simplify.cse_main import cse as _cse
        cses, _expr = _cse(expr, list=False)
    elif callable(cse):
        cses, _expr = cse(expr)
    else:
        cses, _expr = (), expr

    # 生成函数字符串表示
    funcstr = funcprinter.doprint(funcname, iterable_args, _expr, cses=cses)

    # 从代码打印器中收集模块导入信息
    imp_mod_lines = []
    for mod, keys in (getattr(printer, 'module_imports', None) or {}).items():
        for k in keys:
            if k not in namespace:
                ln = "from %s import %s" % (mod, k)
                try:
                    exec(ln, {}, namespace)
                except ImportError:
                    ln = "%s = %s.%s" % (k, mod, k)
                    exec(ln, {}, namespace)
                imp_mod_lines.append(ln)

    # 提供 lambda 表达式所需的内置函数，并提供 range 的兼容实现
    namespace.update({'builtins': builtins, 'range': range})

    # 编译并执行生成的函数代码
    funclocals = {}
    global _lambdify_generated_counter
    filename = '<lambdifygenerated-%s>' % _lambdify_generated_counter
    _lambdify_generated_counter += 1
    c = compile(funcstr, filename, 'exec')
    exec(c, namespace, funclocals)
    # mtime 必须为 None，否则 linecache.checkcache 将移除它
    # 将函数内容缓存到 linecache.cache 中，以便后续调用时可以快速获取
    linecache.cache[filename] = (len(funcstr), None, funcstr.splitlines(True), filename) # type: ignore

    # 获取函数对象
    func = funclocals[funcname]

    # 应用文档字符串到函数对象
    # 构建函数签名
    sig = "func({})".format(", ".join(str(i) for i in names))
    # 格式化函数签名，确保每行以合适的缩进开始
    sig = textwrap.fill(sig, subsequent_indent=' '*8)

    # 根据表达式长度决定是否截断文档字符串中的表达式和源代码
    if _too_large_for_docstring(expr, docstring_limit):
        # 表达式过长，进行截断处理，并提示用户参考 lambdify 的 `docstring_limit` 设置
        expr_str = "EXPRESSION REDACTED DUE TO LENGTH, (see lambdify's `docstring_limit`)"
        src_str = "SOURCE CODE REDACTED DUE TO LENGTH, (see lambdify's `docstring_limit`)"
    else:
        # 表达式不过长，将其转换为字符串
        expr_str = str(expr)
        # 如果表达式长度超过 78，进行适当的截断处理
        if len(expr_str) > 78:
            expr_str = textwrap.wrap(expr_str, 75)[0] + '...'
        # 源代码为函数字符串
        src_str = funcstr

    # 将文档字符串应用到函数对象的 __doc__ 属性中
    func.__doc__ = (
        "Created with lambdify. Signature:\n\n"
        "{sig}\n\n"
        "Expression:\n\n"
        "{expr}\n\n"
        "Source code:\n\n"
        "{src}\n\n"
        "Imported modules:\n\n"
        "{imp_mods}"
        ).format(sig=sig, expr=expr_str, src=src_str, imp_mods='\n'.join(imp_mod_lines))

    # 返回更新后的函数对象
    return func
# 检查模块名是否在给定的模块列表中存在
def _module_present(modname, modlist):
    if modname in modlist:
        return True
    # 遍历模块列表，检查每个模块是否具有 '__name__' 属性且属性值与 modname 相等
    for m in modlist:
        if hasattr(m, '__name__') and m.__name__ == modname:
            return True
    # 如果模块名不在列表中存在，则返回 False
    return False


def _get_namespace(m):
    """
    This is used by _lambdify to parse its arguments.
    """
    # 如果 m 是字符串类型，则导入该模块并返回其命名空间中的第一个元素
    if isinstance(m, str):
        _import(m)
        return MODULES[m][0]
    # 如果 m 是字典类型，则直接返回该字典
    elif isinstance(m, dict):
        return m
    # 如果 m 具有 "__dict__" 属性，则返回其 __dict__ 属性
    elif hasattr(m, "__dict__"):
        return m.__dict__
    else:
        # 如果 m 类型不符合预期，则引发类型错误异常
        raise TypeError("Argument must be either a string, dict or module but it is: %s" % m)


def _recursive_to_string(doprint, arg):
    """Functions in lambdify accept both SymPy types and non-SymPy types such as python
    lists and tuples. This method ensures that we only call the doprint method of the
    printer with SymPy types (so that the printer safely can use SymPy-methods)."""
    # 导入必要的 SymPy 类型
    from sympy.matrices.matrixbase import MatrixBase
    from sympy.core.basic import Basic

    # 如果 arg 是 SymPy 的 Basic 或 MatrixBase 类型，则调用 doprint 方法
    if isinstance(arg, (Basic, MatrixBase)):
        return doprint(arg)
    # 如果 arg 是可迭代的类型
    elif iterable(arg):
        # 根据 arg 的类型确定左右括号的类型
        if isinstance(arg, list):
            left, right = "[", "]"
        elif isinstance(arg, tuple):
            left, right = "(", ")"
            if not arg:
                return "()"  # 如果是空元组，则直接返回空括号字符串
        else:
            # 对于未处理的类型抛出未实现的错误
            raise NotImplementedError("unhandled type: %s, %s" % (type(arg), arg))
        # 递归处理每个元素并使用逗号连接起来，形成字符串表示
        return left +', '.join(_recursive_to_string(doprint, e) for e in arg) + right
    # 如果 arg 是字符串类型，则直接返回
    elif isinstance(arg, str):
        return arg
    else:
        # 对于其他类型，调用 doprint 方法
        return doprint(arg)


def lambdastr(args, expr, printer=None, dummify=None):
    """
    Returns a string that can be evaluated to a lambda function.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.utilities.lambdify import lambdastr
    >>> lambdastr(x, x**2)
    'lambda x: (x**2)'
    >>> lambdastr((x,y,z), [z,y,x])
    'lambda x,y,z: ([z, y, x])'

    Although tuples may not appear as arguments to lambda in Python 3,
    lambdastr will create a lambda function that will unpack the original
    arguments so that nested arguments can be handled:

    >>> lambdastr((x, (y, z)), x + y)
    'lambda _0,_1: (lambda x,y,z: (x + y))(_0,_1[0],_1[1])'
    """
    # 导入 SymPy 中的相关模块和函数
    from sympy.matrices import DeferredVector
    from sympy.core.basic import Basic
    from sympy.core.function import (Derivative, Function)
    from sympy.core.symbol import (Dummy, Symbol)
    from sympy.core.sympify import sympify

    # 如果传入了 printer 参数
    if printer is not None:
        # 如果 printer 是函数，则使用其作为 lambdarepr 函数
        if inspect.isfunction(printer):
            lambdarepr = printer
        else:
            # 如果 printer 是类，则实例化后使用其 doprint 方法
            if inspect.isclass(printer):
                lambdarepr = lambda expr: printer().doprint(expr)
            else:
                # 否则，使用 printer 的 doprint 方法
                lambdarepr = lambda expr: printer.doprint(expr)
    else:
        # 如果没有传入 printer 参数，则使用 lambdarepr 函数的默认实现
        # 注意：这里存在循环导入的问题，因此在这里导入而不是在顶部
        from sympy.printing.lambdarepr import lambdarepr
    def sub_args(args, dummies_dict):
        if isinstance(args, str):
            # 如果参数是字符串，直接返回
            return args
        elif isinstance(args, DeferredVector):
            # 如果参数是 DeferredVector 类型，返回其字符串表示
            return str(args)
        elif iterable(args):
            # 如果参数是可迭代对象
            # 递归处理每个元素并展开为列表
            dummies = flatten([sub_args(a, dummies_dict) for a in args])
            # 用逗号连接处理后的元素字符串
            return ",".join(str(a) for a in dummies)
        else:
            # 替换参数中的函数、符号、导数为虚拟符号（Dummy symbols）
            if isinstance(args, (Function, Symbol, Derivative)):
                # 创建新的虚拟符号
                dummies = Dummy()
                # 更新字典，将原始参数映射到虚拟符号
                dummies_dict.update({args : dummies})
                return str(dummies)
            else:
                # 其他类型的参数直接转换为字符串
                return str(args)

    def sub_expr(expr, dummies_dict):
        # 将表达式转换为 SymPy 的表达式对象
        expr = sympify(expr)
        # 如果表达式是 Basic 类型
        if isinstance(expr, Basic):
            # 使用字典中的映射替换表达式中的虚拟符号
            expr = expr.xreplace(dummies_dict)
        # 如果表达式是列表类型，递归处理每个元素
        elif isinstance(expr, list):
            expr = [sub_expr(a, dummies_dict) for a in expr]
        return expr

    # 判断是否可迭代（但排除字符串、DeferredVector 和 NotIterable 类型）
    def isiter(l):
        return iterable(l, exclude=(str, DeferredVector, NotIterable))

    # 将嵌套结构的索引展开为一维列表
    def flat_indexes(iterable):
        n = 0
        for el in iterable:
            if isiter(el):
                for ndeep in flat_indexes(el):
                    yield (n,) + ndeep
            else:
                yield (n,)
            n += 1

    # 如果 dummify 参数未指定，则根据条件判断是否需要设置为 True
    if dummify is None:
        dummify = any(isinstance(a, Basic) and
            a.atoms(Function, Derivative) for a in (
            args if isiter(args) else [args]))

    # 如果参数 args 是可迭代对象且包含嵌套结构，则生成对应的虚拟符号列表
    if isiter(args) and any(isiter(i) for i in args):
        # 为每个参数生成唯一的虚拟符号名字
        dum_args = [str(Dummy(str(i))) for i in range(len(args))]
        # 生成带索引的参数表示形式，包括嵌套结构的索引
        indexed_args = ','.join([
            dum_args[ind[0]] + ''.join(["[%s]" % k for k in ind[1:]])
                    for ind in flat_indexes(args)])

        # 根据参数、表达式、打印机函数和 dummify 参数生成 lambda 函数的字符串表示
        lstr = lambdastr(flatten(args), expr, printer=printer, dummify=dummify)
        return 'lambda %s: (%s)(%s)' % (','.join(dum_args), lstr, indexed_args)

    # 初始化虚拟符号字典
    dummies_dict = {}
    # 如果需要 dummify，则对参数进行转换
    if dummify:
        args = sub_args(args, dummies_dict)
    else:
        # 如果参数是字符串，则保持不变
        if isinstance(args, str):
            pass
        # 如果参数是可迭代对象（但排除 DeferredVector），则转换为逗号连接的字符串
        elif iterable(args, exclude=DeferredVector):
            args = ",".join(str(a) for a in args)

    # 如果需要 dummify，则对表达式进行转换
    if dummify:
        # 如果表达式是字符串，则保持不变
        if isinstance(expr, str):
            pass
        else:
            # 否则，对表达式进行递归替换虚拟符号
            expr = sub_expr(expr, dummies_dict)
    # 将表达式转换为字符串表示
    expr = _recursive_to_string(lambdarepr, expr)
    # 返回 lambda 函数的字符串表示
    return "lambda %s: (%s)" % (args, expr)
class _EvaluatorPrinter:
    # 定义一个名为 _EvaluatorPrinter 的类，用于生成和打印函数定义的字符串表示
    def __init__(self, printer=None, dummify=False):
        # 初始化函数，接受 printer 和 dummify 两个参数
        self._dummify = dummify

        # 由于存在循环导入问题，需要在此处导入 LambdaPrinter
        from sympy.printing.lambdarepr import LambdaPrinter

        # 如果未提供 printer 参数，则使用 LambdaPrinter 创建默认的 printer
        if printer is None:
            printer = LambdaPrinter()

        # 如果 printer 是函数，则直接将其设为 _exprrepr
        if inspect.isfunction(printer):
            self._exprrepr = printer
        else:
            # 如果 printer 是类，则实例化一个对象
            if inspect.isclass(printer):
                printer = printer()

            # 使用 printer 的 doprint 方法作为 _exprrepr
            self._exprrepr = printer.doprint

            # 下面的代码被注释掉了，原本用于处理 Symbol 和 Dummy 的打印方式
            # if hasattr(printer, '_print_Symbol'):
            #     symbolrepr = printer._print_Symbol
            # if hasattr(printer, '_print_Dummy'):
            #     dummyrepr = printer._print_Dummy

        # 用于以标准方式打印生成函数的参数
        self._argrepr = LambdaPrinter().doprint

    def doprint(self, funcname, args, expr, *, cses=()):
        """
        Returns the function definition code as a string.
        """
        # 导入 Dummy 类
        from sympy.core.symbol import Dummy

        # 函数体列表
        funcbody = []

        # 如果 args 不可迭代，则转换为列表
        if not iterable(args):
            args = [args]

        # 如果存在 cses，则解压缩变量和表达式
        if cses:
            subvars, subexprs = zip(*cses)
            exprs = [expr] + list(subexprs)
            argstrs, exprs = self._preprocess(args, exprs)
            expr, subexprs = exprs[0], exprs[1:]
            cses = zip(subvars, subexprs)
        else:
            argstrs, expr = self._preprocess(args, expr)

        # 生成参数解包和最终参数列表
        funcargs = []
        unpackings = []

        for argstr in argstrs:
            if iterable(argstr):
                funcargs.append(self._argrepr(Dummy()))
                unpackings.extend(self._print_unpacking(argstr, funcargs[-1]))
            else:
                funcargs.append(argstr)

        # 生成函数签名
        funcsig = 'def {}({}):'.format(funcname, ', '.join(funcargs))

        # 在解包之前包装输入参数
        funcbody.extend(self._print_funcargwrapping(funcargs))

        # 添加解包部分
        funcbody.extend(unpackings)

        # 处理 cses 中的变量替换或删除
        for s, e in cses:
            if e is None:
                funcbody.append('del {}'.format(self._exprrepr(s)))
            else:
                funcbody.append('{} = {}'.format(self._exprrepr(s), self._exprrepr(e)))

        # 将表达式转换为字符串表示
        str_expr = _recursive_to_string(self._exprrepr, expr)

        # 如果表达式包含换行符，则用括号括起来
        if '\n' in str_expr:
            str_expr = '({})'.format(str_expr)

        # 添加返回语句
        funcbody.append('return {}'.format(str_expr))

        # 组装最终函数字符串列表
        funclines = [funcsig]
        funclines.extend(['    ' + line for line in funcbody])

        # 返回最终的函数定义字符串
        return '\n'.join(funclines) + '\n'

    @classmethod
    def _is_safe_ident(cls, ident):
        # 类方法，用于检查标识符是否安全，即是否为字符串且符合标识符规则
        return isinstance(ident, str) and ident.isidentifier() \
                and not keyword.iskeyword(ident)
    # 定义一个方法 `_preprocess`，用于预处理参数 `args` 和表达式 `expr`，
    # 替换那些不是有效 Python 标识符的参数。
    def _preprocess(self, args, expr):
        """Preprocess args, expr to replace arguments that do not map
        to valid Python identifiers.

        Returns string form of args, and updated expr.
        """
        from sympy.core.basic import Basic  # 导入 SymPy 的基础模块 Basic
        from sympy.core.sorting import ordered  # 导入 SymPy 的排序模块 ordered
        from sympy.core.function import (Derivative, Function)  # 导入 SymPy 的函数模块中的 Derivative 和 Function 类
        from sympy.core.symbol import Dummy, uniquely_named_symbol  # 导入 SymPy 的符号模块中的 Dummy 类和 uniquely_named_symbol 函数
        from sympy.matrices import DeferredVector  # 导入 SymPy 的矩阵模块中的 DeferredVector 类
        from sympy.core.expr import Expr  # 导入 SymPy 的表达式模块中的 Expr 类

        # 如果存在 Dummy 类型的参数，可能会与 Symbol 类型参数发生名称冲突，
        # 强制将所有参数进行虚拟化处理。
        dummify = self._dummify or any(
            isinstance(arg, Dummy) for arg in flatten(args))

        # 创建一个列表，用于存储参数的字符串形式
        argstrs = [None]*len(args)
        # 使用 ordered 函数对参数进行排序，并且逆序处理
        for arg, i in reversed(list(ordered(zip(args, range(len(args)))))):
            # 如果参数是可迭代的，则递归调用 _preprocess 方法处理参数和表达式
            if iterable(arg):
                s, expr = self._preprocess(arg, expr)
            # 如果参数是 DeferredVector 类型，则直接将其转换为字符串
            elif isinstance(arg, DeferredVector):
                s = str(arg)
            # 如果参数是 Basic 类型且是符号，则将其转换为字符串
            elif isinstance(arg, Basic) and arg.is_symbol:
                s = str(arg)
                # 如果需要虚拟化处理或者该字符串不是安全的标识符，则创建一个 Dummy 符号
                # 并将该参数在表达式中进行替换
                if dummify or not self._is_safe_ident(s):
                    dummy = Dummy()
                    if isinstance(expr, Expr):
                        dummy = uniquely_named_symbol(
                            dummy.name, expr, modify=lambda s: '_' + s)
                    s = self._argrepr(dummy)
                    expr = self._subexpr(expr, {arg: dummy})
            # 如果需要虚拟化处理或者参数是函数或者导数的实例，则创建一个 Dummy 符号
            elif dummify or isinstance(arg, (Function, Derivative)):
                dummy = Dummy()
                s = self._argrepr(dummy)
                expr = self._subexpr(expr, {arg: dummy})
            else:
                # 否则，将参数转换为字符串
                s = str(arg)
            # 将处理后的参数字符串存储在对应的位置
            argstrs[i] = s
        # 返回参数字符串列表和更新后的表达式
        return argstrs, expr

    # 定义一个方法 `_subexpr`，用于在表达式中替换虚拟化的符号
    def _subexpr(self, expr, dummies_dict):
        from sympy.matrices import DeferredVector  # 导入 SymPy 的矩阵模块中的 DeferredVector 类
        from sympy.core.sympify import sympify  # 导入 SymPy 的 sympify 函数

        # 将表达式转换为 SymPy 表达式
        expr = sympify(expr)
        # 获取表达式的 xreplace 方法
        xreplace = getattr(expr, 'xreplace', None)
        if xreplace is not None:
            # 如果存在 xreplace 方法，则使用 dummies_dict 进行表达式的替换
            expr = xreplace(dummies_dict)
        else:
            # 否则，根据表达式的类型进行相应的处理
            if isinstance(expr, DeferredVector):
                pass  # 如果是 DeferredVector 类型，则不进行任何处理
            elif isinstance(expr, dict):
                # 如果是字典类型，则递归调用 _subexpr 方法处理键和值
                k = [self._subexpr(sympify(a), dummies_dict) for a in expr.keys()]
                v = [self._subexpr(sympify(a), dummies_dict) for a in expr.values()]
                expr = dict(zip(k, v))
            elif isinstance(expr, tuple):
                # 如果是元组类型，则递归调用 _subexpr 方法处理每个元素
                expr = tuple(self._subexpr(sympify(a), dummies_dict) for a in expr)
            elif isinstance(expr, list):
                # 如果是列表类型，则递归调用 _subexpr 方法处理每个元素
                expr = [self._subexpr(sympify(a), dummies_dict) for a in expr]
        # 返回处理后的表达式
        return expr
    # 生成函数参数包装的代码。
    # args 是生成函数的参数列表（字符串）。
    # 返回值是将插入到函数定义开头的代码行列表。
    def _print_funcargwrapping(self, args):
        """Generate argument wrapping code.

        args is the argument list of the generated function (strings).

        Return value is a list of lines of code that will be inserted  at
        the beginning of the function definition.
        """
        return []

    # 生成参数解包的代码。
    # arg 是待解包的函数参数（字符串）。
    # unpackto 是要解包到的变量名列表或嵌套列表（字符串）。
    def _print_unpacking(self, unpackto, arg):
        """Generate argument unpacking code.

        arg is the function argument to be unpacked (a string), and
        unpackto is a list or nested lists of the variable names (strings) to
        unpack to.
        """
        # 定义一个内部函数，用于处理左侧（解包到的变量名列表）。
        def unpack_lhs(lvalues):
            return '[{}]'.format(', '.join(
                # 递归处理嵌套的变量名列表，生成字符串形式的解包表达式。
                unpack_lhs(val) if iterable(val) else val for val in lvalues))

        # 返回解包操作的代码行列表。
        return ['{} = {}'.format(unpack_lhs(unpackto), arg)]
class _TensorflowEvaluatorPrinter(_EvaluatorPrinter):
    # 继承自 _EvaluatorPrinter 的私有类 _TensorflowEvaluatorPrinter

    def _print_unpacking(self, lvalues, rvalue):
        """Generate argument unpacking code.

        This method is used when the input value is not interable,
        but can be indexed (see issue #14655).
        """
        # 生成参数解包代码的方法

        def flat_indexes(elems):
            # 辅助函数：扁平化索引

            n = 0

            for el in elems:
                if iterable(el):
                    for ndeep in flat_indexes(el):
                        yield (n,) + ndeep
                else:
                    yield (n,)

                n += 1

        indexed = ', '.join('{}[{}]'.format(rvalue, ']['.join(map(str, ind)))
                                for ind in flat_indexes(lvalues))
        # 构建索引表达式，用于解包操作

        return ['[{}] = [{}]'.format(', '.join(flatten(lvalues)), indexed)]
        # 返回解包结果的赋值语句列表

def _imp_namespace(expr, namespace=None):
    """ Return namespace dict with function implementations

    We need to search for functions in anything that can be thrown at
    us - that is - anything that could be passed as ``expr``.  Examples
    include SymPy expressions, as well as tuples, lists and dicts that may
    contain SymPy expressions.

    Parameters
    ----------
    expr : object
       Something passed to lambdify, that will generate valid code from
       ``str(expr)``.
    namespace : None or mapping
       Namespace to fill.  None results in new empty dict

    Returns
    -------
    namespace : dict
       dict with keys of implemented function names within ``expr`` and
       corresponding values being the numerical implementation of
       function

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.utilities.lambdify import implemented_function, _imp_namespace
    >>> from sympy import Function
    >>> f = implemented_function(Function('f'), lambda x: x+1)
    >>> g = implemented_function(Function('g'), lambda x: x*10)
    >>> namespace = _imp_namespace(f(g(x)))
    >>> sorted(namespace.keys())
    ['f', 'g']
    """
    # 延迟导入以避免循环导入
    from sympy.core.function import FunctionClass
    if namespace is None:
        namespace = {}
    # 元组、列表、字典都是有效的表达式
    if is_sequence(expr):
        for arg in expr:
            _imp_namespace(arg, namespace)
        return namespace
    elif isinstance(expr, dict):
        for key, val in expr.items():
            # 字典的键可以是函数
            _imp_namespace(key, namespace)
            _imp_namespace(val, namespace)
        return namespace
    # SymPy 表达式可能是函数本身
    func = getattr(expr, 'func', None)
    # 检查 func 是否为 FunctionClass 的实例
    if isinstance(func, FunctionClass):
        # 获取 func 对象的 _imp_ 属性
        imp = getattr(func, '_imp_', None)
        # 如果 _imp_ 属性存在
        if imp is not None:
            # 获取表达式的函数名
            name = expr.func.__name__
            # 如果函数名已经在 namespace 中，并且对应的实现不是 imp，则抛出 ValueError
            if name in namespace and namespace[name] != imp:
                raise ValueError('We found more than one '
                                 'implementation with name '
                                 '"%s"' % name)
            # 将函数名和对应的实现 imp 加入到 namespace 中
            namespace[name] = imp
    # 如果表达式具有 args 属性，说明它可能接受函数作为参数
    if hasattr(expr, 'args'):
        # 遍历表达式的每一个参数
        for arg in expr.args:
            # 递归调用 _imp_namespace 函数，处理每一个参数并更新 namespace
            _imp_namespace(arg, namespace)
    # 返回更新后的 namespace
    return namespace
# 定义一个函数，用于将数值实现添加到符号函数中
def implemented_function(symfunc, implementation):
    """ Add numerical ``implementation`` to function ``symfunc``.

    ``symfunc`` can be an ``UndefinedFunction`` instance, or a name string.
    In the latter case we create an ``UndefinedFunction`` instance with that
    name.

    Be aware that this is a quick workaround, not a general method to create
    special symbolic functions. If you want to create a symbolic function to be
    used by all the machinery of SymPy you should subclass the ``Function``
    class.

    Parameters
    ----------
    symfunc : ``str`` or ``UndefinedFunction`` instance
       If ``str``, then create new ``UndefinedFunction`` with this as
       name.  If ``symfunc`` is an Undefined function, create a new function
       with the same name and the implemented function attached.
    implementation : callable
       numerical implementation to be called by ``evalf()`` or ``lambdify``

    Returns
    -------
    afunc : sympy.FunctionClass instance
       function with attached implementation

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.utilities.lambdify import implemented_function
    >>> from sympy import lambdify
    >>> f = implemented_function('f', lambda x: x+1)
    >>> lam_f = lambdify(x, f(x))
    >>> lam_f(4)
    5
    """
    # 延迟导入以避免循环导入
    from sympy.core.function import UndefinedFunction
    # 如果symfunc是UndefinedFunction实例，则获取其关键字参数
    kwargs = {}
    if isinstance(symfunc, UndefinedFunction):
        kwargs = symfunc._kwargs
        symfunc = symfunc.__name__
    # 如果symfunc是字符串，则创建一个新的UndefinedFunction实例
    if isinstance(symfunc, str):
        # 关键字参数传递给UndefinedFunction，并且将数值实现作为静态方法附加到新创建的类
        symfunc = UndefinedFunction(
            symfunc, _imp_=staticmethod(implementation), **kwargs)
    # 如果symfunc不是字符串也不是UndefinedFunction实例，则抛出异常
    elif not isinstance(symfunc, UndefinedFunction):
        raise ValueError(filldedent('''
            symfunc should be either a string or
            an UndefinedFunction instance.'''))
    # 返回最终创建或修改后的符号函数实例
    return symfunc
    # 从 sympy.abc 模块中导入符号变量 x, y, z
    # 导入 _too_large_for_docstring 函数从 sympy.utilities.lambdify 模块
    from sympy.abc import x, y, z
    from sympy.utilities.lambdify import _too_large_for_docstring
    
    # 定义一个简单的表达式 expr = x
    expr = x
    
    # 调用 _too_large_for_docstring 函数进行测试，预期结果应为 False
    _too_large_for_docstring(expr, None)  # False
    
    # 再次调用 _too_large_for_docstring 函数，传入 limit 为 100，预期结果应为 False
    _too_large_for_docstring(expr, 100)  # False
    
    # 继续测试，传入 limit 为 1，预期结果应为 False
    _too_large_for_docstring(expr, 1)    # False
    
    # 接着测试，传入 limit 为 0，预期结果应为 True
    _too_large_for_docstring(expr, 0)    # True
    
    # 最后一次测试，传入 limit 为 -1，预期结果应为 True
    _too_large_for_docstring(expr, -1)   # True
    
    # 以下是对列表表达式的测试 expr = [x, y, z]
    # 调用 _too_large_for_docstring 函数，传入 limit 为 None，预期结果应为 False
    _too_large_for_docstring(expr, None)  # False
    
    # 继续测试，传入 limit 为 100，预期结果应为 False
    _too_large_for_docstring(expr, 100)   # False
    
    # 接着测试，传入 limit 为 1，预期结果应为 True
    _too_large_for_docstring(expr, 1)     # True
    
    # 继续测试，传入 limit 为 0，预期结果应为 True
    _too_large_for_docstring(expr, 0)     # True
    
    # 最后一次测试，传入 limit 为 -1，预期结果应为 True
    _too_large_for_docstring(expr, -1)    # True
    
    # 以下是对复杂嵌套列表表达式的测试
    expr = [x, [y], z, [[x+y], [x*y*z, [x+y+z]]]]
    
    # 调用 _too_large_for_docstring 函数，传入 limit 为 None，预期结果应为 False
    _too_large_for_docstring(expr, None)  # False
    
    # 继续测试，传入 limit 为 100，预期结果应为 False
    _too_large_for_docstring(expr, 100)   # False
    
    # 接着测试，传入 limit 为 1，预期结果应为 True
    _too_large_for_docstring(expr, 1)     # True
    
    # 继续测试，传入 limit 为 0，预期结果应为 True
    _too_large_for_docstring(expr, 0)     # True
    
    # 最后一次测试，传入 limit 为 -1，预期结果应为 True
    _too_large_for_docstring(expr, -1)    # True
    
    # 对于一个非常复杂的表达式 expr = ((x + y + z)**5).expand()
    expr = ((x + y + z)**5).expand()
    
    # 调用 _too_large_for_docstring 函数，传入 limit 为 None，预期结果应为 False
    _too_large_for_docstring(expr, None)  # False
    
    # 继续测试，传入 limit 为 100，预期结果应为 True
    _too_large_for_docstring(expr, 100)   # True
    
    # 接着测试，传入 limit 为 1，预期结果应为 True
    _too_large_for_docstring(expr, 1)     # True
    
    # 继续测试，传入 limit 为 0，预期结果应为 True
    _too_large_for_docstring(expr, 0)     # True
    
    # 最后一次测试，传入 limit 为 -1，预期结果应为 True
    _too_large_for_docstring(expr, -1)    # True
    
    # 对于一个矩阵表达式 expr = Matrix([[...]])
    from sympy import Matrix
    expr = Matrix([[(x + y + z), ((x + y + z)**2).expand(),
                    ((x + y + z)**3).expand(), ((x + y + z)**4).expand()]])
    
    # 调用 _too_large_for_docstring 函数，传入 limit 为 None，预期结果应为 False
    _too_large_for_docstring(expr, None)  # False
    
    # 继续测试，传入 limit 为 1000，预期结果应为 False
    _too_large_for_docstring(expr, 1000)  # False
    
    # 接着测试，传入 limit 为 100，预期结果应为 True
    _too_large_for_docstring(expr, 100)   # True
    
    # 继续测试，传入 limit 为 1，预期结果应为 True
    _too_large_for_docstring(expr, 1)     # True
    
    # 继续测试，传入 limit 为 0，预期结果应为 True
    _too_large_for_docstring(expr, 0)     # True
    
    # 最后一次测试，传入 limit 为 -1，预期结果应为 True
    _too_large_for_docstring(expr, -1)    # True
    
    """
    # 以下是实现 _too_large_for_docstring 函数的部分，需要导入 postorder_traversal 函数
    # 以遍历表达式的后序遍历，并检查节点数是否超过限制
    # 如果 limit 为 None，直接返回 False
    # 初始化计数器 i = 0
    # 对于 expr 的后序遍历结果，每次遍历计数器 i 自增 1
    # 如果 i 超过了 limit，返回 True
    # 否则，最终返回 False
    """
    def _too_large_for_docstring(expr, limit):
        # Must be imported here to avoid a circular import error
        from sympy.core.traversal import postorder_traversal
    
        if limit is None:
            return False
    
        i = 0
        for _ in postorder_traversal(expr):
            i += 1
            if i > limit:
                return True
        return False
```
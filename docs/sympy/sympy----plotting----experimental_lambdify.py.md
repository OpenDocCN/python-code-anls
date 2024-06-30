# `D:\src\scipysrc\sympy\sympy\plotting\experimental_lambdify.py`

```
# 导入 re 模块，用于正则表达式操作
import re
# 导入 SymPy 中的数值相关模块和符号模块
from sympy.core.numbers import (I, NumberSymbol, oo, zoo)
from sympy.core.symbol import Symbol
# 导入 SymPy 中的工具函数，用于生成标记化的符号
from sympy.utilities.iterables import numbered_symbols

#  我们将表达式字符串解析成一个识别函数的树形结构。
#  然后我们翻译函数的名称，并且翻译一些不是函数名称的字符串（根据翻译字典进行操作）。
#  如果翻译涉及到其他模块（如 numpy），该模块将被导入，并且 'func' 被翻译为 'module.func'。
#  如果某个函数无法翻译，该部分树的内部节点将不被翻译。
#  如果我们有 Integral(sqrt(x))，sqrt 不会被翻译为 np.sqrt，并且 Integral 不会崩溃。
#  通过遍历表达式的 (func, args) 树形结构生成命名空间。创建这个命名空间涉及许多丑陋的解决方案。

#  该命名空间包含了 SymPy 表达式所需的所有名称，以及用于翻译的模块名称。
#  这些模块仅作为名称导入（例如 import numpy as np），以便保持命名空间的小型和可管理性。

#  请注意，如果出现 bug，请不要试图在此处修复！请使用下面 Q&A 中提出的方法重新编写。
#  这样新函数将能够同样良好地工作，同样简单，但不需要任何新的解决方案。
#  如果你坚持在这里修复，查看 sympy_expression_namespace 和 lambdify 函数中的解决方案。

#  Q: 为什么不使用 Python 的抽象语法树（AST）？
#  A: 因为在这种情况下它更加复杂，而且并没有更强大。

#  Q: 如果我有 Symbol('sin') 或者 g=Function('f') 呢？
#  A: 你会破坏算法。我们应该使用 srepr 来防御这种情况吗？
#  有关 Symbol('sin') 的问题在于它会被打印为 'sin'。解析器会将其区分于函数 'sin'，因为函数通过开括号进行检测，
#  但是如果 lambda 表达式中也有 sin 函数，它将无法区分这种差异。
#  解决方案（复杂）是使用 srepr 可能还有抽象语法树（AST）。
#  关于 g=Function('f') 的问题在于它会被打印为 'f'，但是在全局命名空间中我们只有 'g'。
#  但是由于构造函数中也使用了相同的打印器，所以不会有问题。

#  Q: 如果某些打印器未按预期打印会怎么样？
#  A: 算法将无法工作。你必须使用 srepr 处理这些情况。但是即使 srepr 有时也无法很好地打印。
#  所有与打印器相关的问题都应该被视为 bug。
# 从 sympy.external 导入 import_module 函数
# 导入警告模块
from sympy.external import import_module
import warnings

# 定义 vectorized_lambdify 类，用于返回一个智能化、矢量化和 lambdified 的函数
class vectorized_lambdify:
    """ Return a sufficiently smart, vectorized and lambdified function.

    Returns only reals.

    Explanation
    ===========

    This function uses experimental_lambdify to created a lambdified
    expression ready to be used with numpy. Many of the functions in SymPy
    are not implemented in numpy so in some cases we resort to Python cmath or
    even to evalf.

    The following translations are tried:
      only numpy complex
      - on errors raised by SymPy trying to work with ndarray:
          only Python cmath and then vectorize complex128

    When using Python cmath there is no need for evalf or float/complex
    because Python cmath calls those.

    This function never tries to mix numpy directly with evalf because numpy
    does not understand SymPy Float. If this is needed one can use the
    float_wrap_evalf/complex_wrap_evalf options of experimental_lambdify or
    better one can be explicit about the dtypes that numpy works with.
    Check numpy bug http://projects.scipy.org/numpy/ticket/1013 to know what
    types of errors to expect.
    """
    
    # 初始化方法，接受参数 args 和表达式 expr
    def __init__(self, args, expr):
        self.args = args  # 设置参数列表
        self.expr = expr  # 设置表达式

        # 导入 numpy 模块
        self.np = import_module('numpy')

        # 使用 experimental_lambdify 创建 lambdified 函数，用于 numpy
        self.lambda_func_1 = experimental_lambdify(
            args, expr, use_np=True)
        self.vector_func_1 = self.lambda_func_1  # 设置第一个 lambdified 函数

        # 使用 experimental_lambdify 创建 lambdified 函数，使用 Python cmath
        self.lambda_func_2 = experimental_lambdify(
            args, expr, use_python_cmath=True)
        # 对第二个 lambdified 函数进行向量化，输出类型为 complex
        self.vector_func_2 = self.np.vectorize(
            self.lambda_func_2, otypes=[complex])

        # 默认选择第一个 lambdified 函数作为主要使用的函数
        self.vector_func = self.vector_func_1
        self.failure = False  # 初始化失败标志为 False
    # 定义一个特殊方法 __call__()，使得对象可以像函数一样被调用
    def __call__(self, *args):
        # 将 self.np 赋值给 np，通常是一个用于数学运算的库（如numpy）
        np = self.np

        try:
            # 将参数 args 中的每个元素转换为复数类型的 numpy 数组，并返回生成器对象
            temp_args = (np.array(a, dtype=complex) for a in args)
            # 调用实例属性 vector_func 执行向量化函数，传入上述生成器对象作为参数
            results = self.vector_func(*temp_args)
            # 使用 numpy.ma.masked_where() 方法根据条件遮盖数组中满足条件的部分
            results = np.ma.masked_where(
                np.abs(results.imag) > 1e-7 * np.abs(results),
                results.real, copy=False)
            # 返回处理后的结果
            return results
        except ValueError:
            # 捕获 ValueError 异常
            if self.failure:
                # 如果 self.failure 已经为 True，则抛出异常
                raise

            # 设置 self.failure 为 True，标记执行失败
            self.failure = True
            # 将 self.vector_func 替换为备用的向量化函数 self.vector_func_2
            self.vector_func = self.vector_func_2
            # 发出警告，提示表达式的评估存在问题，并尝试备用方法
            warnings.warn(
                'The evaluation of the expression is problematic. '
                'We are trying a failback method that may still work. '
                'Please report this as a bug.')
            # 递归调用 __call__() 方法，重新尝试计算
            return self.__call__(*args)
class lambdify:
    """Returns the lambdified function.

    Explanation
    ===========

    This class creates a lambdified function from a given expression
    using experimental_lambdify. It attempts different configurations
    based on the availability of Python cmath and math libraries. If 
    certain functions are not directly available in Python cmath, it
    falls back to using evalf method.

    Attributes
    ----------
    args : tuple
        Arguments of the function.
    expr : expression
        Mathematical expression to be lambdified.
    lambda_func_1 : function
        Lambdified function using experimental_lambdify with Python cmath
        and evalf.
    lambda_func_2 : function
        Lambdified function using experimental_lambdify with Python math
        and evalf.
    lambda_func_3 : function
        Lambdified function using experimental_lambdify with evalf and
        complex_wrap_evalf.
    lambda_func : function
        Currently active lambdified function.
    failure : bool
        Flag indicating whether a failure has occurred during evaluation.
    """

    def __init__(self, args, expr):
        self.args = args
        self.expr = expr
        self.lambda_func_1 = experimental_lambdify(
            args, expr, use_python_cmath=True, use_evalf=True)
        self.lambda_func_2 = experimental_lambdify(
            args, expr, use_python_math=True, use_evalf=True)
        self.lambda_func_3 = experimental_lambdify(
            args, expr, use_evalf=True, complex_wrap_evalf=True)
        self.lambda_func = self.lambda_func_1
        self.failure = False

    def __call__(self, args):
        try:
            # The result can be sympy.Float. Hence wrap it with complex type.
            result = complex(self.lambda_func(args))
            if abs(result.imag) > 1e-7 * abs(result):
                return None
            return result.real
        except (ZeroDivisionError, OverflowError):
            return None
        except TypeError as e:
            if self.failure:
                raise e

            if self.lambda_func == self.lambda_func_1:
                self.lambda_func = self.lambda_func_2
                return self.__call__(args)

            self.failure = True
            self.lambda_func = self.lambda_func_3
            warnings.warn(
                'The evaluation of the expression is problematic. '
                'We are trying a failback method that may still work. '
                'Please report this as a bug.', stacklevel=2)
            return self.__call__(args)


def experimental_lambdify(*args, **kwargs):
    """Creates a Lambdifier object and returns it."""
    l = Lambdifier(*args, **kwargs)
    return l


class Lambdifier:
    """Class for lambdifying expressions."""

    def __init__(self, *args, **kwargs):
        # Implementation details for Lambdifier are in another part of the code.
        pass
    # 初始化函数，接受多个参数，设置各种选项和标志位
    def __init__(self, args, expr, print_lambda=False, use_evalf=False,
                 float_wrap_evalf=False, complex_wrap_evalf=False,
                 use_np=False, use_python_math=False, use_python_cmath=False,
                 use_interval=False):

        # 设置是否打印 lambda 表达式的标志位
        self.print_lambda = print_lambda
        # 设置是否使用 evalf 的标志位
        self.use_evalf = use_evalf
        # 设置是否对 evalf 返回的浮点数进行包装的标志位
        self.float_wrap_evalf = float_wrap_evalf
        # 设置是否对 evalf 返回的复数进行包装的标志位
        self.complex_wrap_evalf = complex_wrap_evalf
        # 设置是否使用 numpy 的标志位
        self.use_np = use_np
        # 设置是否使用 Python 的 math 库的标志位
        self.use_python_math = use_python_math
        # 设置是否使用 Python 的 cmath 库的标志位
        self.use_python_cmath = use_python_cmath
        # 设置是否使用区间数学的标志位
        self.use_interval = use_interval

        # 构造参数字符串
        # - 检查参数是否全为 Symbol 类型
        if not all(isinstance(a, Symbol) for a in args):
            raise ValueError('The arguments must be Symbols.')
        # - 使用 numbered_symbols 来为表达式中的自由符号创建编号的符号
        syms = numbered_symbols(exclude=expr.free_symbols)
        newargs = [next(syms) for _ in args]
        # - 使用新的符号替换表达式中的参数
        expr = expr.xreplace(dict(zip(args, newargs)))
        # - 构造参数字符串
        argstr = ', '.join([str(a) for a in newargs])
        del syms, newargs, args

        # 构造翻译字典并进行翻译
        # - 获取字符串形式的表达式
        exprstr = str(expr)
        # - 使用 tree2str_translate 将字符串表达式转换为新表达式
        newexpr = self.tree2str_translate(self.str2tree(exprstr))

        # 构造命名空间
        namespace = {}
        # - 将符号表达式的原子加入命名空间
        namespace.update(self.sympy_atoms_namespace(expr))
        # - 将符号表达式的表达式加入命名空间
        namespace.update(self.sympy_expression_namespace(expr))
        # XXX Workaround
        # 解决方案
        # 由于 Pow(a,Half) 打印为 sqrt(a)，而 sympy_expression_namespace 无法捕捉这种情况，进行丑陋的解决方案。
        from sympy.functions.elementary.miscellaneous import sqrt
        namespace.update({'sqrt': sqrt})
        # 将 Eq 函数定义为 lambda 函数，用于比较相等
        namespace.update({'Eq': lambda x, y: x == y})
        # 将 Ne 函数定义为 lambda 函数，用于比较不等
        namespace.update({'Ne': lambda x, y: x != y})
        # End workaround.
        # 如果使用 Python 的 math 库，则加入 math 到命名空间
        if use_python_math:
            namespace.update({'math': __import__('math')})
        # 如果使用 Python 的 cmath 库，则加入 cmath 到命名空间
        if use_python_cmath:
            namespace.update({'cmath': __import__('cmath')})
        # 如果使用 numpy 库，则尝试导入 numpy 并加入其到命名空间
        if use_np:
            try:
                namespace.update({'np': __import__('numpy')})
            except ImportError:
                raise ImportError(
                    'experimental_lambdify failed to import numpy.')
        # 如果使用区间数学，则加入 intervalmath 到命名空间，并将 math 加入命名空间
        if use_interval:
            namespace.update({'imath': __import__(
                'sympy.plotting.intervalmath', fromlist=['intervalmath'])})
            namespace.update({'math': __import__('math')})

        # 构造 lambda 表达式
        # - 如果设置了打印 lambda 表达式，则打印新表达式
        if self.print_lambda:
            print(newexpr)
        # - 构造 lambda 表达式的字符串形式
        eval_str = 'lambda %s : ( %s )' % (argstr, newexpr)
        # - 使用 exec 在命名空间中执行 lambda 表达式的字符串形式
        self.eval_str = eval_str
        exec("MYNEWLAMBDA = %s" % eval_str, namespace)
        # - 将 lambda 函数赋值给 self.lambda_func
        self.lambda_func = namespace['MYNEWLAMBDA']

    # 实现调用运算符，调用 lambda 函数并返回结果
    def __call__(self, *args, **kwargs):
        return self.lambda_func(*args, **kwargs)
    # SymPy 到其他模块的翻译字典
    ##############################################################################
    ###
    # builtins
    ###
    # 在 builtins 模块中具有不同名称的函数
    builtin_functions_different = {
        'Min': 'min',  # SymPy 中的 Min 对应于内置模块中的 min 函数
        'Max': 'max',  # SymPy 中的 Max 对应于内置模块中的 max 函数
        'Abs': 'abs',  # SymPy 中的 Abs 对应于内置模块中的 abs 函数
    }

    # 应该进行翻译的字符串
    builtin_not_functions = {
        'I': '1j',      # SymPy 中的 'I' 对应于 Python 中的复数单位 '1j'
        # 继续添加其他需要翻译的字符串
#        'oo': '1e400',
    }

    ###
    # numpy
    ###

    # Functions that are the same in numpy
    numpy_functions_same = [
        'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'exp', 'log',
        'sqrt', 'floor', 'conjugate', 'sign',
    ]

    # Functions with different names in numpy
    numpy_functions_different = {
        "acos": "arccos",
        "acosh": "arccosh",
        "arg": "angle",
        "asin": "arcsin",
        "asinh": "arcsinh",
        "atan": "arctan",
        "atan2": "arctan2",
        "atanh": "arctanh",
        "ceiling": "ceil",
        "im": "imag",
        "ln": "log",
        "Max": "amax",
        "Min": "amin",
        "re": "real",
        "Abs": "abs",
    }

    # Strings that should be translated
    numpy_not_functions = {
        'pi': 'np.pi',
        'oo': 'np.inf',
        'E': 'np.e',
    }

    ###
    # Python math
    ###

    # Functions that are the same in math
    math_functions_same = [
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
        'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
        'exp', 'log', 'erf', 'sqrt', 'floor', 'factorial', 'gamma',
    ]

    # Functions with different names in math
    math_functions_different = {
        'ceiling': 'ceil',
        'ln': 'log',
        'loggamma': 'lgamma'
    }

    # Strings that should be translated
    math_not_functions = {
        'pi': 'math.pi',
        'E': 'math.e',
    }

    ###
    # Python cmath
    ###

    # Functions that are the same in cmath
    cmath_functions_same = [
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
        'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
        'exp', 'log', 'sqrt',
    ]

    # Functions with different names in cmath
    cmath_functions_different = {
        'ln': 'log',
        'arg': 'phase',
    }

    # Strings that should be translated
    cmath_not_functions = {
        'pi': 'cmath.pi',
        'E': 'cmath.e',
    }

    ###
    # intervalmath
    ###

    interval_not_functions = {
        'pi': 'math.pi',
        'E': 'math.e'
    }

    interval_functions_same = [
        'sin', 'cos', 'exp', 'tan', 'atan', 'log',
        'sqrt', 'cosh', 'sinh', 'tanh', 'floor',
        'acos', 'asin', 'acosh', 'asinh', 'atanh',
        'Abs', 'And', 'Or'
    ]

    interval_functions_different = {
        'Min': 'imin',
        'Max': 'imax',
        'ceiling': 'ceil',

    }

    ###
    # mpmath, etc
    ###
    #TODO

    ###
    # Create the final ordered tuples of dictionaries
    ###

    # For strings
    def get_dict_str(self):
        dict_str = dict(self.builtin_not_functions)
        if self.use_np:
            dict_str.update(self.numpy_not_functions)
        if self.use_python_math:
            dict_str.update(self.math_not_functions)
        if self.use_python_cmath:
            dict_str.update(self.cmath_not_functions)
        if self.use_interval:
            dict_str.update(self.interval_not_functions)
        return dict_str
    # 返回一个包含所有内置函数的字典
    def get_dict_fun(self):
        dict_fun = dict(self.builtin_functions_different)
        # 如果使用 NumPy，添加 NumPy 函数到字典中
        if self.use_np:
            for s in self.numpy_functions_same:
                dict_fun[s] = 'np.' + s
            for k, v in self.numpy_functions_different.items():
                dict_fun[k] = 'np.' + v
        # 如果使用 Python 标准库中的 math，添加 math 函数到字典中
        if self.use_python_math:
            for s in self.math_functions_same:
                dict_fun[s] = 'math.' + s
            for k, v in self.math_functions_different.items():
                dict_fun[k] = 'math.' + v
        # 如果使用 Python 标准库中的 cmath，添加 cmath 函数到字典中
        if self.use_python_cmath:
            for s in self.cmath_functions_same:
                dict_fun[s] = 'cmath.' + s
            for k, v in self.cmath_functions_different.items():
                dict_fun[k] = 'cmath.' + v
        # 如果使用 interval，添加 imath 中的函数到字典中
        if self.use_interval:
            for s in self.interval_functions_same:
                dict_fun[s] = 'imath.' + s
            for k, v in self.interval_functions_different.items():
                dict_fun[k] = 'imath.' + v
        # 返回包含所有函数的字典
        return dict_fun

    ##############################################################################
    # The translator functions, tree parsers, etc.
    ##############################################################################

    def str2tree(self, exprstr):
        """Converts an expression string to a tree.

        Explanation
        ===========

        Functions are represented by ('func_name(', tree_of_arguments).
        Other expressions are (head_string, mid_tree, tail_str).
        Expressions that do not contain functions are directly returned.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> from sympy import Integral, sin
        >>> from sympy.plotting.experimental_lambdify import Lambdifier
        >>> str2tree = Lambdifier([x], x).str2tree

        >>> str2tree(str(Integral(x, (x, 1, y))))
        ('', ('Integral(', 'x, (x, 1, y)'), ')')
        >>> str2tree(str(x+y))
        'x + y'
        >>> str2tree(str(x+y*sin(z)+1))
        ('x + y*', ('sin(', 'z'), ') + 1')
        >>> str2tree('sin(y*(y + 1.1) + (sin(y)))')
        ('', ('sin(', ('y*(y + 1.1) + (', ('sin(', 'y'), '))')), ')')
        """
        # 在表达式字符串中查找第一个 '函数名(' 的匹配
        first_par = re.search(r'(\w+\()', exprstr)
        # 如果没有找到匹配的 '函数名('，直接返回表达式字符串
        if first_par is None:
            return exprstr
        else:
            start = first_par.start()
            end = first_par.end()
            head = exprstr[:start]
            func = exprstr[start:end]
            tail = exprstr[end:]
            count = 0
            # 计算函数的参数个数
            for i, c in enumerate(tail):
                if c == '(':
                    count += 1
                elif c == ')':
                    count -= 1
                # 找到函数参数结束的位置
                if count == -1:
                    break
            # 递归地将函数的参数部分和尾部转换为树形结构
            func_tail = self.str2tree(tail[:i])
            tail = self.str2tree(tail[i:])
            # 返回树的表示形式
            return (head, (func, func_tail), tail)

    @classmethod
    def tree2str(cls, tree):
        """Converts a tree to string without translations.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> from sympy import sin
        >>> from sympy.plotting.experimental_lambdify import Lambdifier
        >>> str2tree = Lambdifier([x], x).str2tree
        >>> tree2str = Lambdifier([x], x).tree2str

        >>> tree2str(str2tree(str(x+y*sin(z)+1)))
        'x + y*sin(z) + 1'
        """
        # 如果树已经是字符串，则直接返回该字符串
        if isinstance(tree, str):
            return tree
        else:
            # 否则，将树中所有子树转换为字符串并连接起来
            return ''.join(map(cls.tree2str, tree))

    def tree2str_translate(self, tree):
        """Converts a tree to string with translations.

        Explanation
        ===========

        Function names are translated by translate_func.
        Other strings are translated by translate_str.
        """
        # 如果树是字符串，则使用 translate_str 方法翻译该字符串
        if isinstance(tree, str):
            return self.translate_str(tree)
        # 如果树是一个元组且长度为2，则使用 translate_func 方法翻译函数名和参数树
        elif isinstance(tree, tuple) and len(tree) == 2:
            return self.translate_func(tree[0][:-1], tree[1])
        else:
            # 否则，递归地将树中所有子树转换为字符串并连接起来
            return ''.join([self.tree2str_translate(t) for t in tree])

    def translate_str(self, estr):
        """Translate substrings of estr using in order the dictionaries in
        dict_tuple_str."""
        # 使用 dict_str 中的模式替换 estr 中的子字符串
        for pattern, repl in self.dict_str.items():
                estr = re.sub(pattern, repl, estr)
        return estr

    def translate_func(self, func_name, argtree):
        """Translate function names and the tree of arguments.

        Explanation
        ===========

        If the function name is not in the dictionaries of dict_tuple_fun then the
        function is surrounded by a float((...).evalf()).

        The use of float is necessary as np.<function>(sympy.Float(..)) raises an
        error."""
        # 如果 func_name 在 dict_fun 中，则使用相应的转换名和参数树进行翻译
        if func_name in self.dict_fun:
            new_name = self.dict_fun[func_name]
            argstr = self.tree2str_translate(argtree)
            return new_name + '(' + argstr
        # 如果 func_name 是 'Eq' 或 'Ne'，则转换为 lambda 表达式
        elif func_name in ['Eq', 'Ne']:
            op = {'Eq': '==', 'Ne': '!='}
            return "(lambda x, y: x {} y)({}".format(op[func_name], self.tree2str_translate(argtree))
        else:
            # 否则，根据配置选择是否包装函数调用表达式
            template = '(%s(%s)).evalf(' if self.use_evalf else '%s(%s'
            if self.float_wrap_evalf:
                template = 'float(%s)' % template
            elif self.complex_wrap_evalf:
                template = 'complex(%s)' % template

            # 仅在最外层表达式进行包装
            float_wrap_evalf = self.float_wrap_evalf
            complex_wrap_evalf = self.complex_wrap_evalf
            self.float_wrap_evalf = False
            self.complex_wrap_evalf = False
            ret =  template % (func_name, self.tree2str_translate(argtree))
            self.float_wrap_evalf = float_wrap_evalf
            self.complex_wrap_evalf = complex_wrap_evalf
            return ret
    ##############################################################################
    # The namespace constructors
    ##############################################################################

    @classmethod
    def sympy_expression_namespace(cls, expr):
        """Traverses the (func, args) tree of an expression and creates a SymPy
        namespace. All other modules are imported only as a module name. That way
        the namespace is not polluted and rests quite small. It probably causes much
        more variable lookups and so it takes more time, but there are no tests on
        that for the moment."""

        # 如果表达式为None，则返回空字典
        if expr is None:
            return {}
        else:
            # 获取表达式的函数名的字符串表示
            funcname = str(expr.func)

            # XXX Workaround
            # 这里使用一个丑陋的解决方法，因为str(func(x))并不总是与str(func)相同。
            # 例如：
            # >>> str(Integral(x))
            # "Integral(x)"
            # >>> str(Integral)
            # "<class 'sympy.integrals.integrals.Integral'>"
            # >>> str(sqrt(x))
            # "sqrt(x)"
            # >>> str(sqrt)
            # "<function sqrt at 0x3d92de8>"
            # >>> str(sin(x))
            # "sin(x)"
            # >>> str(sin)
            # "sin"
            # 其中之一可以使用，但不能同时使用。代码认为sin的示例是正确的。
            
            # 匹配可能的正则表达式列表来识别函数名
            regexlist = [
                r'<class \'sympy[\w.]*?.([\w]*)\'>$',
                # 例如 Integral 的示例
                r'<function ([\w]*) at 0x[\w]*>$',    # 例如 sqrt 的示例
            ]
            for r in regexlist:
                m = re.match(r, funcname)
                if m is not None:
                    funcname = m.groups()[0]
            # End of the workaround

            # XXX debug: print funcname

            # 初始化参数字典
            args_dict = {}

            # 遍历表达式的参数
            for a in expr.args:
                # 如果参数是 Symbol, NumberSymbol, I, zoo, oo 中的一种，则跳过
                if (isinstance(a, (Symbol, NumberSymbol)) or a in [I, zoo, oo]):
                    continue
                else:
                    # 递归获取参数的命名空间，并更新到args_dict中
                    args_dict.update(cls.sympy_expression_namespace(a))

            # 将函数名和其对应的函数对象更新到args_dict中
            args_dict.update({funcname: expr.func})

            return args_dict

    @staticmethod
    def sympy_atoms_namespace(expr):
        """For no real reason this function is separated from
        sympy_expression_namespace. It can be moved to it."""

        # 获取表达式中的所有原子符号(Symbol)和常数(NumberSymbol, I, zoo, oo)，
        # 并将它们构建成一个字典返回
        atoms = expr.atoms(Symbol, NumberSymbol, I, zoo, oo)
        d = {}
        for a in atoms:
            # XXX debug: print 'atom:' + str(a)
            # 将每个原子符号转换为字符串，并作为键，原子符号对象作为值存入字典d中
            d[str(a)] = a
        return d
```
# `D:\src\scipysrc\sympy\sympy\printing\lambdarepr.py`

```
# 从.pycode模块中导入PythonCodePrinter和MpmathPrinter类
# 从.numpy模块中导入NumPyPrinter类，用于向后兼容
from .pycode import (
    PythonCodePrinter,
    MpmathPrinter,
)
from .numpy import NumPyPrinter  # NumPyPrinter用于向后兼容
# 从sympy.core.sorting模块中导入default_sort_key函数
from sympy.core.sorting import default_sort_key

# 定义__all__变量，指定导出的符号名称列表
__all__ = [
    'PythonCodePrinter',
    'MpmathPrinter',  # 为向后兼容而发布MpmathPrinter
    'NumPyPrinter',
    'LambdaPrinter',
    'NumPyPrinter',  # 重复的导出名称，可能是个错误
    'IntervalPrinter',
    'lambdarepr',
]

# LambdaPrinter类，继承自PythonCodePrinter类，用于生成可用于lambdify的表达式字符串
class LambdaPrinter(PythonCodePrinter):
    """
    This printer converts expressions into strings that can be used by
    lambdify.
    """
    # 定义打印方法的名称为"_lambdacode"
    printmethod = "_lambdacode"

    # 下面的方法重写了PythonCodePrinter中的打印方法，用于特定类型的表达式
    def _print_And(self, expr):
        result = ['(']
        for arg in sorted(expr.args, key=default_sort_key):
            result.extend(['(', self._print(arg), ')'])
            result.append(' and ')
        result = result[:-1]
        result.append(')')
        return ''.join(result)

    def _print_Or(self, expr):
        result = ['(']
        for arg in sorted(expr.args, key=default_sort_key):
            result.extend(['(', self._print(arg), ')'])
            result.append(' or ')
        result = result[:-1]
        result.append(')')
        return ''.join(result)

    def _print_Not(self, expr):
        result = ['(', 'not (', self._print(expr.args[0]), '))']
        return ''.join(result)

    def _print_BooleanTrue(self, expr):
        return "True"

    def _print_BooleanFalse(self, expr):
        return "False"

    def _print_ITE(self, expr):
        result = [
            '((', self._print(expr.args[1]),
            ') if (', self._print(expr.args[0]),
            ') else (', self._print(expr.args[2]), '))'
        ]
        return ''.join(result)

    def _print_NumberSymbol(self, expr):
        return str(expr)

    def _print_Pow(self, expr, **kwargs):
        # XXX 临时解决方案。应该将Python数学打印器与PythonCodePrinter隔离开来吗？
        return super(PythonCodePrinter, self)._print_Pow(expr, **kwargs)


# numexpr通过修改传递给numexpr.evaluate的字符串来工作，
# 而不是通过填充命名空间。因此需要一个特殊的打印器...
class NumExprPrinter(LambdaPrinter):
    # printmethod属性指定打印方法的名称为"_numexprcode"
    printmethod = "_numexprcode"

    # _numexpr_functions字典定义了SymPy名称到numexpr名称的映射关系
    # 如果不在此字典中的函数将引发TypeError异常
    _numexpr_functions = {
        'sin' : 'sin',
        'cos' : 'cos',
        'tan' : 'tan',
        'asin': 'arcsin',
        'acos': 'arccos',
        'atan': 'arctan',
        'atan2' : 'arctan2',
        'sinh' : 'sinh',
        'cosh' : 'cosh',
        'tanh' : 'tanh',
        'asinh': 'arcsinh',
        'acosh': 'arccosh',
        'atanh': 'arctanh',
        'ln' : 'log',
        'log': 'log',
        'exp': 'exp',
        'sqrt' : 'sqrt',
        'Abs' : 'abs',
        'conjugate' : 'conj',
        'im' : 'imag',
        're' : 'real',
        'where' : 'where',
        'complex' : 'complex',
        'contains' : 'contains',
    }

    # module属性指定打印器所属的模块为'numexpr'
    module = 'numexpr'
    # 打印虚数单位 '1j'，这是一个简单的特殊情况处理
    def _print_ImaginaryUnit(self, expr):
        return '1j'

    # 打印序列的字符串表示，使用指定的分隔符，默认为 ', '
    # 这是从 pretty.py 中简化的 _print_seq 函数
    def _print_seq(self, seq, delimiter=', '):
        s = [self._print(item) for item in seq]  # 对序列中的每个元素进行打印
        if s:
            return delimiter.join(s)  # 使用指定的分隔符连接所有打印结果
        else:
            return ""  # 如果序列为空则返回空字符串

    # 打印函数表达式
    def _print_Function(self, e):
        func_name = e.func.__name__  # 获取函数名

        nstr = self._numexpr_functions.get(func_name, None)  # 根据函数名获取相应的字符串表示
        if nstr is None:
            # 检查是否为 implemented_function
            if hasattr(e, '_imp_'):
                return "(%s)" % self._print(e._imp_(*e.args))  # 打印 implemented_function 的结果
            else:
                # 抛出类型错误，numexpr 不支持该函数
                raise TypeError("numexpr does not support function '%s'" %
                                func_name)
        # 返回函数调用的字符串表示，包括函数名和参数
        return "%s(%s)" % (nstr, self._print_seq(e.args))

    # 打印 Piecewise 函数表达式
    def _print_Piecewise(self, expr):
        "Piecewise function printer"
        exprs = [self._print(arg.expr) for arg in expr.args]  # 打印每个分段函数的表达式部分
        conds = [self._print(arg.cond) for arg in expr.args]  # 打印每个分段函数的条件部分

        ans = []  # 存储打印结果的列表
        parenthesis_count = 0  # 记录需要添加的右括号数量
        is_last_cond_True = False  # 标记最后一个条件是否为 True
        for cond, expr in zip(conds, exprs):
            if cond == 'True':
                ans.append(expr)  # 如果条件为 'True'，直接添加表达式
                is_last_cond_True = True
                break
            else:
                ans.append('where(%s, %s, ' % (cond, expr))  # 添加条件和表达式的 where 函数调用形式
                parenthesis_count += 1

        if not is_last_cond_True:
            # 处理最后一个条件不为 True 的情况
            ans.append('log(-1)')  # 添加 log(-1) 的字符串表示，用于处理特定情况

        # 返回组合好的字符串表示，包括 where 函数调用和额外的右括号
        return ''.join(ans) + ')' * parenthesis_count

    # 打印 ITE (If-Then-Else) 表达式
    def _print_ITE(self, expr):
        from sympy.functions.elementary.piecewise import Piecewise
        return self._print(expr.rewrite(Piecewise))  # 调用 rewrite 方法将 ITE 转换为 Piecewise 再打印

    # 标记函数，用于标记所有矩阵类型的打印输出为黑名单
    def blacklisted(self, expr):
        raise TypeError("numexpr cannot be used with %s" %
                        expr.__class__.__name__)

    # 将所有矩阵类型的打印输出标记为黑名单
    _print_SparseRepMatrix = \
    _print_MutableSparseMatrix = \
    _print_ImmutableSparseMatrix = \
    _print_Matrix = \
    _print_DenseMatrix = \
    _print_MutableDenseMatrix = \
    _print_ImmutableMatrix = \
    _print_ImmutableDenseMatrix = \
    blacklisted
    # 将以下 Python 表达式加入黑名单
    _print_list = \
    _print_tuple = \
    _print_Tuple = \
    _print_dict = \
    _print_Dict = \
    blacklisted

    # 定义用于打印 NumExprEvaluate 的方法
    def _print_NumExprEvaluate(self, expr):
        # 构建评估表达式的字符串表示，使用实例的模块格式和表达式
        evaluate = self._module_format(self.module +".evaluate")
        return "%s('%s', truediv=True)" % (evaluate, self._print(expr.expr))

    # 定义处理表达式的方法
    def doprint(self, expr):
        # 导入必要的模块
        from sympy.codegen.ast import CodegenAST
        from sympy.codegen.pynodes import NumExprEvaluate
        # 如果表达式不是 CodegenAST 类型，则转换为 NumExprEvaluate 类型
        if not isinstance(expr, CodegenAST):
            expr = NumExprEvaluate(expr)
        return super().doprint(expr)

    # 定义处理 Return 表达式的方法
    def _print_Return(self, expr):
        # 导入必要的模块
        from sympy.codegen.pynodes import NumExprEvaluate
        # 获取 Return 表达式中的返回值
        r, = expr.args
        # 如果返回值不是 NumExprEvaluate 类型，则转换为 NumExprEvaluate 类型
        if not isinstance(r, NumExprEvaluate):
            expr = expr.func(NumExprEvaluate(r))
        return super()._print_Return(expr)

    # 定义处理赋值表达式的方法
    def _print_Assignment(self, expr):
        # 导入必要的模块
        from sympy.codegen.pynodes import NumExprEvaluate
        # 获取赋值表达式左右两侧以及其他参数
        lhs, rhs, *args = expr.args
        # 如果右侧表达式不是 NumExprEvaluate 类型，则转换为 NumExprEvaluate 类型
        if not isinstance(rhs, NumExprEvaluate):
            expr = expr.func(lhs, NumExprEvaluate(rhs), *args)
        return super()._print_Assignment(expr)

    # 定义处理代码块表达式的方法
    def _print_CodeBlock(self, expr):
        # 导入必要的模块
        from sympy.codegen.ast import CodegenAST
        from sympy.codegen.pynodes import NumExprEvaluate
        # 对表达式中的参数进行处理，若不是 CodegenAST 类型则转换为 NumExprEvaluate 类型
        args = [ arg if isinstance(arg, CodegenAST) else NumExprEvaluate(arg) for arg in expr.args ]
        return super()._print_CodeBlock(self, expr.func(*args))
class IntervalPrinter(MpmathPrinter, LambdaPrinter):
    """Use ``lambda`` printer but print numbers as ``mpi`` intervals. """
    # 定义一个类 IntervalPrinter，继承自 MpmathPrinter 和 LambdaPrinter

    def _print_Integer(self, expr):
        # 重载 _print_Integer 方法，将整数 expr 格式化为 mpi 类型的字符串
        return "mpi('%s')" % super(PythonCodePrinter, self)._print_Integer(expr)

    def _print_Rational(self, expr):
        # 重载 _print_Rational 方法，将有理数 expr 格式化为 mpi 类型的字符串
        return "mpi('%s')" % super(PythonCodePrinter, self)._print_Rational(expr)

    def _print_Half(self, expr):
        # 重载 _print_Half 方法，将 Half 类型（半数）的 expr 格式化为 mpi 类型的字符串
        return "mpi('%s')" % super(PythonCodePrinter, self)._print_Rational(expr)

    def _print_Pow(self, expr):
        # 重载 _print_Pow 方法，使用 MpmathPrinter 的 _print_Pow 方法处理指数表达式 expr（有理数为真）
        return super(MpmathPrinter, self)._print_Pow(expr, rational=True)

# 遍历 NumExprPrinter 类中的 _numexpr_functions 列表，并为每个函数动态设置打印方法
for k in NumExprPrinter._numexpr_functions:
    setattr(NumExprPrinter, '_print_%s' % k, NumExprPrinter._print_Function)

def lambdarepr(expr, **settings):
    """
    Returns a string usable for lambdifying.
    """
    # 返回一个字符串，该字符串可用于创建 lambda 函数
    return LambdaPrinter(settings).doprint(expr)
```
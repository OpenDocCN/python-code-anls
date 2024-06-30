# `D:\src\scipysrc\sympy\sympy\printing\codeprinter.py`

```
from __future__ import annotations
from typing import Any

from functools import wraps  # 导入 wraps 函数，用于包装函数保留原函数信息

from sympy.core import Add, Mul, Pow, S, sympify, Float  # 导入 SymPy 核心模块中的多个类和函数
from sympy.core.basic import Basic  # 导入 SymPy 核心基础类 Basic
from sympy.core.expr import UnevaluatedExpr  # 导入 SymPy 核心表达式类 UnevaluatedExpr
from sympy.core.function import Lambda  # 导入 SymPy 核心函数类 Lambda
from sympy.core.mul import _keep_coeff  # 导入 SymPy 核心乘法类 _keep_coeff
from sympy.core.sorting import default_sort_key  # 导入 SymPy 核心排序函数 default_sort_key
from sympy.core.symbol import Symbol  # 导入 SymPy 核心符号类 Symbol
from sympy.functions.elementary.complexes import re  # 导入 SymPy 复数相关函数 re
from sympy.printing.str import StrPrinter  # 导入 SymPy 打印字符串类 StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE  # 导入 SymPy 打印优先级相关函数

class requires:
    """ Decorator for registering requirements on print methods. """
    def __init__(self, **kwargs):
        self._req = kwargs  # 初始化需求字典 _req

    def __call__(self, method):
        def _method_wrapper(self_, *args, **kwargs):
            for k, v in self._req.items():
                getattr(self_, k).update(v)  # 更新打印方法的要求
            return method(self_, *args, **kwargs)
        return wraps(method)(_method_wrapper)  # 返回包装后的方法，保留原方法信息

class AssignmentError(Exception):
    """
    Raised if an assignment variable for a loop is missing.
    """
    pass

class PrintMethodNotImplementedError(NotImplementedError):
    """
    Raised if a _print_* method is missing in the Printer.
    """
    pass

def _convert_python_lists(arg):
    if isinstance(arg, list):
        from sympy.codegen.abstract_nodes import List
        return List(*(_convert_python_lists(e) for e in arg))  # 转换 Python 列表为 SymPy 抽象节点列表
    elif isinstance(arg, tuple):
        return tuple(_convert_python_lists(e) for e in arg)  # 转换 Python 元组为 SymPy 抽象节点元组
    else:
        return arg  # 返回原始参数

class CodePrinter(StrPrinter):
    """
    The base class for code-printing subclasses.
    """

    _operators = {
        'and': '&&',  # 逻辑与的代码打印输出为 &&
        'or': '||',   # 逻辑或的代码打印输出为 ||
        'not': '!',   # 逻辑非的代码打印输出为 !
    }

    _default_settings: dict[str, Any] = {
        'order': None,  # 打印顺序设置为 None
        'full_prec': 'auto',  # 打印精度设置为 auto
        'error_on_reserved': False,  # 不报告保留字错误
        'reserved_word_suffix': '_',  # 保留字后缀设置为 _
        'human': True,  # 人类友好模式设置为 True
        'inline': False,  # 不内联打印
        'allow_unknown_functions': False,  # 不允许未知函数
        'strict': None  # 严格模式设置为 None，如果 human == True 则为 True
    }

    # Functions which are "simple" to rewrite to other functions that
    # may be supported
    # function_to_rewrite : (function_to_rewrite_to, iterable_with_other_functions_required)
    # 可重写函数的字典，将数学函数映射为其替代函数和依赖列表
    _rewriteable_functions = {
            'cot': ('tan', []),                   # cot 函数替换为 tan 函数，无依赖
            'csc': ('sin', []),                   # csc 函数替换为 sin 函数，无依赖
            'sec': ('cos', []),                   # sec 函数替换为 cos 函数，无依赖
            'acot': ('atan', []),                 # acot 函数替换为 atan 函数，无依赖
            'acsc': ('asin', []),                 # acsc 函数替换为 asin 函数，无依赖
            'asec': ('acos', []),                 # asec 函数替换为 acos 函数，无依赖
            'coth': ('exp', []),                  # coth 函数替换为 exp 函数，无依赖
            'csch': ('exp', []),                  # csch 函数替换为 exp 函数，无依赖
            'sech': ('exp', []),                  # sech 函数替换为 exp 函数，无依赖
            'acoth': ('log', []),                 # acoth 函数替换为 log 函数，无依赖
            'acsch': ('log', []),                 # acsch 函数替换为 log 函数，无依赖
            'asech': ('log', []),                 # asech 函数替换为 log 函数，无依赖
            'catalan': ('gamma', []),             # catalan 函数替换为 gamma 函数，无依赖
            'fibonacci': ('sqrt', []),            # fibonacci 函数替换为 sqrt 函数，无依赖
            'lucas': ('sqrt', []),                # lucas 函数替换为 sqrt 函数，无依赖
            'beta': ('gamma', []),                # beta 函数替换为 gamma 函数，无依赖
            'sinc': ('sin', ['Piecewise']),       # sinc 函数替换为 sin 函数，依赖 Piecewise
            'Mod': ('floor', []),                 # Mod 函数替换为 floor 函数，无依赖
            'factorial': ('gamma', []),           # factorial 函数替换为 gamma 函数，无依赖
            'factorial2': ('gamma', ['Piecewise']),  # factorial2 函数替换为 gamma 函数，依赖 Piecewise
            'subfactorial': ('uppergamma', []),   # subfactorial 函数替换为 uppergamma 函数，无依赖
            'RisingFactorial': ('gamma', ['Piecewise']),  # RisingFactorial 函数替换为 gamma 函数，依赖 Piecewise
            'FallingFactorial': ('gamma', ['Piecewise']),  # FallingFactorial 函数替换为 gamma 函数，依赖 Piecewise
            'binomial': ('gamma', []),            # binomial 函数替换为 gamma 函数，无依赖
            'frac': ('floor', []),                # frac 函数替换为 floor 函数，无依赖
            'Max': ('Piecewise', []),             # Max 函数替换为 Piecewise 函数，无依赖
            'Min': ('Piecewise', []),             # Min 函数替换为 Piecewise 函数，无依赖
            'Heaviside': ('Piecewise', []),       # Heaviside 函数替换为 Piecewise 函数，无依赖
            'erf2': ('erf', []),                  # erf2 函数替换为 erf 函数，无依赖
            'erfc': ('erf', []),                  # erfc 函数替换为 erf 函数，无依赖
            'Li': ('li', []),                     # Li 函数替换为 li 函数，无依赖
            'Ei': ('li', []),                     # Ei 函数替换为 li 函数，无依赖
            'dirichlet_eta': ('zeta', []),        # dirichlet_eta 函数替换为 zeta 函数，无依赖
            'riemann_xi': ('zeta', ['gamma']),    # riemann_xi 函数替换为 zeta 函数，依赖 gamma
            'SingularityFunction': ('Piecewise', [])  # SingularityFunction 函数替换为 Piecewise 函数，无依赖
    }

    # 初始化函数，接受设置参数并调用父类的初始化方法
    def __init__(self, settings=None):
        super().__init__(settings=settings)
        # 如果设置中的 strict 属性为 None，则将其设为与 human 属性相同（默认为 True）
        if self._settings.get('strict', True) == None:
            self._settings['strict'] = self._settings.get('human', True) == True
        # 如果实例中没有 reserved_words 属性，则创建一个空的集合
        if not hasattr(self, 'reserved_words'):
            self.reserved_words = set()

    # 处理 UnevaluatedExpr 类型的函数，将其表达式中符合条件的部分进行替换
    def _handle_UnevaluatedExpr(self, expr):
        return expr.replace(re, lambda arg: arg if isinstance(
            arg, UnevaluatedExpr) and arg.args[0].is_real else re(arg))
    def doprint(self, expr, assign_to=None):
        """
        Print the expression as code.

        Parameters
        ----------
        expr : Expression
            The expression to be printed.

        assign_to : Symbol, string, MatrixSymbol, list of strings or Symbols (optional)
            If provided, the printed code will set the expression to a variable or multiple variables
            with the name or names given in ``assign_to``.
        """
        # 导入必要的模块和类
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.codegen.ast import CodeBlock, Assignment

        def _handle_assign_to(expr, assign_to):
            # 处理没有指定 assign_to 的情况，直接返回表达式的 sympify 结果
            if assign_to is None:
                return sympify(expr)
            # 处理 assign_to 是列表或元组的情况，将表达式分别赋值给多个变量
            if isinstance(assign_to, (list, tuple)):
                if len(expr) != len(assign_to):
                    raise ValueError('Failed to assign an expression of length {} to {} variables'.format(len(expr), len(assign_to)))
                return CodeBlock(*[_handle_assign_to(lhs, rhs) for lhs, rhs in zip(expr, assign_to)])
            # 处理 assign_to 是字符串的情况，根据表达式是否是矩阵选择创建 MatrixSymbol 或 Symbol
            if isinstance(assign_to, str):
                if expr.is_Matrix:
                    assign_to = MatrixSymbol(assign_to, *expr.shape)
                else:
                    assign_to = Symbol(assign_to)
            elif not isinstance(assign_to, Basic):
                raise TypeError("{} cannot assign to object of type {}".format(
                        type(self).__name__, type(assign_to)))
            # 返回赋值操作的 Assignment 对象
            return Assignment(assign_to, expr)

        # 将表达式中的 Python 列表转换为适当的表达式类型
        expr = _convert_python_lists(expr)
        # 处理 assign_to，将表达式赋值给指定的变量
        expr = _handle_assign_to(expr, assign_to)

        # 移除由于 UnevaluatedExpr.is_real 始终为 None 而产生的 re(...) 节点
        expr = self._handle_UnevaluatedExpr(expr)

        # 初始化存储不支持的表达式和需要声明和初始化的数值常量的集合
        self._not_supported = set()
        self._number_symbols = set()

        # 将打印后的表达式拆分为行
        lines = self._print(expr).splitlines()

        # 格式化输出
        if self._settings["human"]:
            frontlines = []
            # 如果有不支持的表达式，则将其添加到输出前面作为注释
            if self._not_supported:
                frontlines.append(self._get_comment(
                        "Not supported in {}:".format(self.language)))
                for expr in sorted(self._not_supported, key=str):
                    frontlines.append(self._get_comment(type(expr).__name__))
            # 将需要声明的数值常量逐个添加到前面作为声明语句
            for name, value in sorted(self._number_symbols, key=str):
                frontlines.append(self._declare_number_const(name, value))
            lines = frontlines + lines
            # 格式化整体代码块的输出
            lines = self._format_code(lines)
            result = "\n".join(lines)
        else:
            # 对于非人类可读模式，整理代码行并收集数值常量的信息
            lines = self._format_code(lines)
            num_syms = {(k, self._print(v)) for k, v in self._number_symbols}
            result = (num_syms, self._not_supported, "\n".join(lines))
        
        # 清空不支持的表达式和数值常量集合，并返回结果
        self._not_supported = set()
        self._number_symbols = set()
        return result
    def _get_expression_indices(self, expr, assign_to):
        # 导入 sympy.tensor 模块中的 get_indices 函数
        from sympy.tensor import get_indices
        # 调用 get_indices 函数获取表达式 expr 和赋值目标 assign_to 的指标
        rinds, junk = get_indices(expr)
        linds, junk = get_indices(assign_to)

        # 支持标量的广播
        if linds and not rinds:
            rinds = linds
        # 如果右手边的指标不等于左手边的指标，抛出 ValueError 异常
        if rinds != linds:
            raise ValueError("lhs indices must match non-dummy"
                    " rhs indices in %s" % expr)

        # 调用 self._sort_optimized 方法，返回优化排序后的指标
        return self._sort_optimized(rinds, assign_to)

    def _sort_optimized(self, indices, expr):
        # 导入 sympy.tensor.indexed 模块中的 Indexed 类
        from sympy.tensor.indexed import Indexed

        # 如果 indices 为空列表，直接返回空列表
        if not indices:
            return []

        # 创建一个空的分数表
        score_table = {}
        # 遍历 indices 列表，初始化每个指标的分数为 0
        for i in indices:
            score_table[i] = 0

        # 在表达式 expr 中找到所有的 Indexed 对象
        arrays = expr.atoms(Indexed)
        # 遍历每个 Indexed 对象
        for arr in arrays:
            # 遍历每个 Indexed 对象的索引
            for p, ind in enumerate(arr.indices):
                try:
                    # 尝试更新分数表中对应索引的分数，调用 self._rate_index_position 方法
                    score_table[ind] += self._rate_index_position(p)
                except KeyError:
                    pass

        # 返回根据 score_table 中分数排序后的 indices 列表
        return sorted(indices, key=lambda x: score_table[x])

    def _rate_index_position(self, p):
        """计算基于索引位置的分数的函数

        这个方法用于按照优化顺序排序循环，参见 CodePrinter._sort_optimized()
        """
        # 抛出 NotImplementedError 异常，要求子类实现该方法
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _get_statement(self, codestring):
        """格式化代码字符串，并加上正确的行结束符。"""
        # 抛出 NotImplementedError 异常，要求子类实现该方法
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _get_comment(self, text):
        """将文本字符串格式化为注释。"""
        # 抛出 NotImplementedError 异常，要求子类实现该方法
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _declare_number_const(self, name, value):
        """在函数顶部声明一个数值常量。"""
        # 抛出 NotImplementedError 异常，要求子类实现该方法
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _format_code(self, lines):
        """接收代码行列表，并进行适当的格式化处理。

        这可能包括缩进、长行包装等操作。
        """
        # 抛出 NotImplementedError 异常，要求子类实现该方法
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _get_loop_opening_ending(self, indices):
        """返回一个包含开头和结束的代码行列表的元组。

        这些代码行用于循环结构。
        """
        # 抛出 NotImplementedError 异常，要求子类实现该方法
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")
    # 定义一个方法，用于打印形如 'Dummy_' 开头的表达式，返回添加下划线的名称
    def _print_Dummy(self, expr):
        if expr.name.startswith('Dummy_'):
            return '_' + expr.name
        else:
            return '%s_%d' % (expr.name, expr.dummy_index)

    # 定义一个方法，用于打印代码块表达式，将其每个子表达式打印为字符串，并用换行连接返回
    def _print_CodeBlock(self, expr):
        return '\n'.join([self._print(i) for i in expr.args])

    # 定义一个方法，用于打印字符串表达式，直接返回字符串的表示形式
    def _print_String(self, string):
        return str(string)

    # 定义一个方法，用于打印带引号的字符串表达式，返回添加双引号的字符串内容
    def _print_QuotedString(self, arg):
        return '"%s"' % arg.text

    # 定义一个方法，用于打印注释表达式，调用 _get_comment 方法获取注释的字符串表示形式并返回
    def _print_Comment(self, string):
        return self._get_comment(str(string))

    # 定义一个方法，用于打印赋值表达式，根据不同的赋值类型返回相应的字符串表示
    def _print_Assignment(self, expr):
        from sympy.codegen.ast import Assignment
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.tensor.indexed import IndexedBase
        lhs = expr.lhs
        rhs = expr.rhs
        # 对多行赋值进行特殊处理
        if isinstance(expr.rhs, Piecewise):
            # 将 Piecewise 中的每个子表达式转换为 Assignment，然后继续打印
            expressions = []
            conditions = []
            for (e, c) in rhs.args:
                expressions.append(Assignment(lhs, e))
                conditions.append(c)
            temp = Piecewise(*zip(expressions, conditions))
            return self._print(temp)
        elif isinstance(lhs, MatrixSymbol):
            # 对矩阵符号的每个元素进行赋值并打印
            lines = []
            for (i, j) in self._traverse_matrix_indices(lhs):
                temp = Assignment(lhs[i, j], rhs[i, j])
                code0 = self._print(temp)
                lines.append(code0)
            return "\n".join(lines)
        elif self._settings.get("contract", False) and (lhs.has(IndexedBase) or
                rhs.has(IndexedBase)):
            # 检查是否需要循环打印，并进行必要的循环处理
            return self._doprint_loops(rhs, lhs)
        else:
            # 打印普通的赋值语句
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))

    # 定义一个方法，用于打印增强赋值表达式，打印增强赋值运算符和表达式的字符串形式
    def _print_AugmentedAssignment(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        return self._get_statement("{} {} {}".format(
            *(self._print(arg) for arg in [lhs_code, expr.op, rhs_code])))

    # 定义一个方法，用于打印函数调用表达式，返回函数名及其参数列表的字符串形式
    def _print_FunctionCall(self, expr):
        return '%s(%s)' % (
            expr.name,
            ', '.join((self._print(arg) for arg in expr.function_args)))

    # 定义一个方法，用于打印变量表达式，返回变量符号的打印结果
    def _print_Variable(self, expr):
        return self._print(expr.symbol)
    # 重写父类方法，处理符号表达式的打印输出，检查是否为保留关键字并处理相应情况
    def _print_Symbol(self, expr):
        name = super()._print_Symbol(expr)  # 调用父类方法获取符号的名称
        # 如果符号名称在保留关键字列表中
        if name in self.reserved_words:
            # 如果设置要求在保留关键字出现时报错
            if self._settings['error_on_reserved']:
                msg = ('This expression includes the symbol "{}" which is a '
                       'reserved keyword in this language.')
                raise ValueError(msg.format(name))  # 抛出值错误异常
            return name + self._settings['reserved_word_suffix']  # 否则返回处理后的符号名称
        else:
            return name  # 如果不是保留关键字，则直接返回符号名称

    # 检查函数名称是否为已知函数或者具有自己的打印方法，用于检查是否可以重写
    def _can_print(self, name):
        """ Check if function ``name`` is either a known function or has its own
            printing method. Used to check if rewriting is possible."""
        return name in self.known_functions or getattr(self, '_print_{}'.format(name), False)

    # 处理函数表达式的打印输出
    def _print_Function(self, expr):
        # 如果函数名称在已知函数列表中
        if expr.func.__name__ in self.known_functions:
            cond_func = self.known_functions[expr.func.__name__]
            # 如果条件函数是字符串
            if isinstance(cond_func, str):
                return "%s(%s)" % (cond_func, self.stringify(expr.args, ", "))  # 返回格式化的字符串表达式
            else:
                # 否则，遍历条件函数列表，根据条件选择相应的函数处理表达式
                for cond, func in cond_func:
                    if cond(*expr.args):
                        break
                if func is not None:
                    try:
                        # 尝试对参数进行括号化处理后应用函数
                        return func(*[self.parenthesize(item, 0) for item in expr.args])
                    except TypeError:
                        return "%s(%s)" % (func, self.stringify(expr.args, ", "))  # 处理类型错误时返回格式化的字符串表达式
        # 如果函数名称不在已知函数列表中，但是是内联函数表达式
        elif hasattr(expr, '_imp_') and isinstance(expr._imp_, Lambda):
            return self._print(expr._imp_(*expr.args))  # 对内联函数进行打印处理
        # 如果函数名称在可重写函数列表中
        elif expr.func.__name__ in self._rewriteable_functions:
            # 简单重写为支持的函数
            target_f, required_fs = self._rewriteable_functions[expr.func.__name__]
            if self._can_print(target_f) and all(self._can_print(f) for f in required_fs):
                return '(' + self._print(expr.rewrite(target_f)) + ')'  # 返回重写后的函数表达式

        # 如果表达式是函数且允许使用未知函数，并且设置中允许
        if expr.is_Function and self._settings.get('allow_unknown_functions', False):
            return '%s(%s)' % (self._print(expr.func), ', '.join(map(self._print, expr.args)))  # 返回格式化的未知函数表达式
        else:
            return self._print_not_supported(expr)  # 否则返回不支持打印的表达式处理结果

    _print_Expr = _print_Function  # 将 _print_Expr 方法重定向到 _print_Function 方法

    # 针对 Heaviside 函数，禁止继承其打印方法到代码打印器
    _print_Heaviside = None

    # 处理数学常数符号的打印输出
    def _print_NumberSymbol(self, expr):
        if self._settings.get("inline", False):
            return self._print(Float(expr.evalf(self._settings["precision"])))  # 如果设置中允许内联，则内联处理
        else:
            # 否则将未实现的数学常数符号注册并评估
            self._number_symbols.add((expr, Float(expr.evalf(self._settings["precision"]))))
            return str(expr)  # 返回数学常数符号的字符串表示

    # 处理 Catalan 常数符号的打印输出
    def _print_Catalan(self, expr):
        return self._print_NumberSymbol(expr)  # 调用 _print_NumberSymbol 处理 Catalan 常数符号的打印输出

    # 处理 EulerGamma 常数符号的打印输出
    def _print_EulerGamma(self, expr):
        return self._print_NumberSymbol(expr)  # 调用 _print_NumberSymbol 处理 EulerGamma 常数符号的打印输出
    # 返回以 NumberSymbol 形式打印表达式的结果
    def _print_GoldenRatio(self, expr):
        return self._print_NumberSymbol(expr)

    # 返回以 NumberSymbol 形式打印表达式的结果
    def _print_TribonacciConstant(self, expr):
        return self._print_NumberSymbol(expr)

    # 返回以 NumberSymbol 形式打印表达式的结果
    def _print_Exp1(self, expr):
        return self._print_NumberSymbol(expr)

    # 返回以 NumberSymbol 形式打印表达式的结果
    def _print_Pi(self, expr):
        return self._print_NumberSymbol(expr)

    # 打印 And 表达式，并根据运算符优先级对参数进行排序和括号化
    def _print_And(self, expr):
        PREC = precedence(expr)
        return (" %s " % self._operators['and']).join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))

    # 打印 Or 表达式，并根据运算符优先级对参数进行排序和括号化
    def _print_Or(self, expr):
        PREC = precedence(expr)
        return (" %s " % self._operators['or']).join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))

    # 打印 Xor 表达式，如果运算符不存在，则打印 expr 的 nnf 形式
    # 根据运算符优先级对参数进行括号化
    def _print_Xor(self, expr):
        if self._operators.get('xor') is None:
            return self._print(expr.to_nnf())
        PREC = precedence(expr)
        return (" %s " % self._operators['xor']).join(self.parenthesize(a, PREC)
                for a in expr.args)

    # 打印 Equivalent 表达式，如果运算符不存在，则打印 expr 的 nnf 形式
    # 根据运算符优先级对参数进行括号化
    def _print_Equivalent(self, expr):
        if self._operators.get('equivalent') is None:
            return self._print(expr.to_nnf())
        PREC = precedence(expr)
        return (" %s " % self._operators['equivalent']).join(self.parenthesize(a, PREC)
                for a in expr.args)

    # 打印 Not 表达式，并括号化其参数
    def _print_Not(self, expr):
        PREC = precedence(expr)
        return self._operators['not'] + self.parenthesize(expr.args[0], PREC)

    # 打印 BooleanFunction 表达式的 nnf 形式
    def _print_BooleanFunction(self, expr):
        return self._print(expr.to_nnf())


这些注释解释了每个方法在代码中的作用和功能，符合所给的注意事项和示例的格式要求。
    # 定义一个方法用于打印 Mul 表达式
    def _print_Mul(self, expr):
        # 获取表达式的优先级
        prec = precedence(expr)

        # 将表达式分解为系数和基本表达式
        c, e = expr.as_coeff_Mul()

        # 如果系数为负数，则转换为正数并记录符号
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        # 分别存储在分子和分母中的表达式项
        a = []  # 分子中的项
        b = []  # 分母中的项（如果有）

        # 将具有特定属性的表达式项分别存入分子和分母列表中
        pow_paren = []  # 将收集所有满足条件的 Pow 表达式（具有多个基元且指数为 -1）

        # 根据打印顺序整理表达式项
        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            args = Mul.make_args(expr)

        # 遍历表达式项，分类存入分子或分母列表
        for item in args:
            # 如果是可交换的、是 Pow 类型且指数为有理数且为负数
            if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    # 如果 Pow 的指数为 -1 并且基数项大于一个，则加入到 pow_paren 列表
                    if len(item.args[0].args) != 1 and isinstance(item.base, Mul):   # 避免类似 #14160 的情况
                        pow_paren.append(item)
                    b.append(Pow(item.base, -item.exp))
            else:
                a.append(item)

        # 如果 a 为空，则设置为包含单位元素 S.One 的列表
        a = a or [S.One]

        # 如果 a 中只有一个项并且符号为负，则对这一项进行括号化
        if len(a) == 1 and sign == "-":
            a_str = [self.parenthesize(a[0], 0.5*(PRECEDENCE["Pow"]+PRECEDENCE["Mul"]))]
        else:
            a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]

        # 对于 pow_paren 中的 Pow 表达式，如果其基数在分母 b 中，则在 b_str 中对应项加括号
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        # 如果分母为空，则直接返回分子项组成的字符串
        if not b:
            return sign + '*'.join(a_str)
        elif len(b) == 1:
            # 如果分母只有一个项，则返回分子项加上分母项的字符串
            return sign + '*'.join(a_str) + "/" + b_str[0]
        else:
            # 如果分母有多个项，则返回分子项加上整体分母项组成的字符串
            return sign + '*'.join(a_str) + "/(%s)" % '*'.join(b_str)

    # 处理不支持打印的表达式类型
    def _print_not_supported(self, expr):
        # 如果设置了严格模式，抛出打印方法未实现的异常
        if self._settings.get('strict', False):
            raise PrintMethodNotImplementedError("Unsupported by %s: %s" % (str(type(self)), str(type(expr))) + \
                             "\nSet the printer option 'strict' to False in order to generate partially printed code.")
        
        # 尝试将表达式添加到不支持打印的集合中，如果表达式不可哈希则忽略
        try:
            self._not_supported.add(expr)
        except TypeError:
            pass  # 不可哈希的类型，忽略
        
        # 返回空打印结果
        return self.emptyPrinter(expr)

    # 以下类型的打印无法简单地转换为 C 或 Fortran
    _print_Basic = _print_not_supported
    _print_ComplexInfinity = _print_not_supported
    _print_Derivative = _print_not_supported
    _print_ExprCondPair = _print_not_supported
    # 将打印函数指定为不支持的打印函数（通常用于未实现的打印方法）
    _print_GeometryEntity = _print_not_supported
    # 将打印函数指定为不支持的打印函数
    _print_Infinity = _print_not_supported
    # 将打印函数指定为不支持的打印函数
    _print_Integral = _print_not_supported
    # 将打印函数指定为不支持的打印函数
    _print_Interval = _print_not_supported
    # 将打印函数指定为不支持的打印函数
    _print_AccumulationBounds = _print_not_supported
    # 将打印函数指定为不支持的打印函数
    _print_Limit = _print_not_supported
    # 将打印函数指定为不支持的打印函数
    _print_MatrixBase = _print_not_supported
    # 将打印函数指定为不支持的打印函数
    _print_DeferredVector = _print_not_supported
    # 将打印函数指定为不支持的打印函数
    _print_NaN = _print_not_supported
    # 将打印函数指定为不支持的打印函数
    _print_NegativeInfinity = _print_not_supported
    # 将打印函数指定为不支持的打印函数
    _print_Order = _print_not_supported
    # 将打印函数指定为不支持的打印函数
    _print_RootOf = _print_not_supported
    # 将打印函数指定为不支持的打印函数
    _print_RootsOf = _print_not_supported
    # 将打印函数指定为不支持的打印函数
    _print_RootSum = _print_not_supported
    # 将打印函数指定为不支持的打印函数
    _print_Uniform = _print_not_supported
    # 将打印函数指定为不支持的打印函数
    _print_Unit = _print_not_supported
    # 将打印函数指定为不支持的打印函数
    _print_Wild = _print_not_supported
    # 将打印函数指定为不支持的打印函数
    _print_WildFunction = _print_not_supported
    # 将打印函数指定为不支持的打印函数
    _print_Relational = _print_not_supported
# 定义 C 代码打印函数。这些函数包含在本文件中，以便可以在顶层 __init__.py 中导入，而无需导入 sympy.codegen 模块。

def ccode(expr, assign_to=None, standard='c99', **settings):
    """将 SymPy 表达式转换为 C 代码字符串

    Parameters
    ==========

    expr : Expr
        要转换的 SymPy 表达式。
    assign_to : optional
        当给出时，将其用作将表达式分配给的变量名。可以是字符串、Symbol、MatrixSymbol 或 Indexed 类型。
        这在换行或生成多行语句的表达式中很有帮助。
    standard : str, optional
        指定 C 代码标准的字符串。如果您的编译器支持更现代的标准，可以将其设置为 'c99' 以允许打印器使用更多数学函数。[默认为 'c89']。
    precision : integer, optional
        数字精度，如 pi 的精度 [默认为 17]。
    user_functions : dict, optional
        一个字典，其中键是 FunctionClass 或 UndefinedFunction 实例的字符串表示，值是它们期望的 C 字符串表示。
        或者字典值可以是元组列表，例如 [(argument_test, cfunction_string)] 或 [(argument_test, cfunction_formater)]。有关示例，请参见下文。
    dereference : iterable, optional
        应在打印的代码表达式中解引用的符号的可迭代对象。这些将作为地址传递给函数的值。
        例如，如果 dereference=[a]，则生成的代码将打印 (*a) 而不是 a。
    human : bool, optional
        如果为 True，则结果是一个单一字符串，其中可能包含一些常数声明，例如数字符号。如果为 False，则相同的信息将作为元组返回：(symbols_to_declare, not_supported_functions, code_text)。[默认为 True]。
    contract: bool, optional
        如果为 True，则假定 Indexed 实例遵循张量缩并规则，并生成相应的索引的嵌套循环。
        设置 contract=False 将不生成循环，而是由用户负责在代码中提供索引的值。[默认为 True]。

    Examples
    ========

    >>> from sympy import ccode, symbols, Rational, sin, ceiling, Abs, Function
    >>> x, tau = symbols("x, tau")
    >>> expr = (2*tau)**Rational(7, 2)
    >>> ccode(expr)
    '8*M_SQRT2*pow(tau, 7.0/2.0)'
    >>> ccode(expr, math_macros={})
    '8*sqrt(2)*pow(tau, 7.0/2.0)'
    >>> ccode(sin(x), assign_to="s")
    's = sin(x);'
    >>> from sympy.codegen.ast import real, float80
    >>> ccode(expr, type_aliases={real: float80})
    '8*M_SQRT2l*powl(tau, 7.0L/2.0L)'

    Simple custom printing can be defined for certain types by passing a
    """
    # 实现将 SymPy 表达式转换为 C 代码字符串的功能，根据传入的参数设置输出格式和选项
    pass
    """
    ``Piecewise`` 表达式会转换为条件语句。如果提供了 ``assign_to`` 变量，则创建一个 if 语句，否则使用三元运算符。注意，如果 ``Piecewise`` 缺少默认项，即 ``(expr, True)``，则会引发错误，以防生成无法评估的表达式。
    
    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(ccode(expr, tau, standard='C89'))
    if (x > 0) {
    tau = x + 1;
    }
    else {
    tau = x;
    }
    """
    
    from sympy.printing.c import c_code_printers
    
    # 返回给定表达式的 C 代码表示
    def ccode(expr, assign_to=None, contract=True, standard='C89', **settings):
        # 使用给定的标准选择合适的 C 代码打印机
        return c_code_printers[standard.lower()](settings).doprint(expr, assign_to)
    """
# 打印给定表达式的 C 语言表示
def print_ccode(expr, **settings):
    # 调用 ccode 函数获取表达式的 C 语言代码，并打印出来
    print(ccode(expr, **settings))

# 将表达式转换为 Fortran 代码的字符串
def fcode(expr, assign_to=None, **settings):
    """Converts an expr to a string of fortran code

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        DEPRECATED. Use type_mappings instead. The precision for numbers such
        as pi [default=17].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations. Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)]. See below
        for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols. If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text). [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].
    source_format : optional
        The source format can be either 'fixed' or 'free'. [default='fixed']
    standard : integer, optional
        The Fortran standard to be followed. This is specified as an integer.
        Acceptable standards are 66, 77, 90, 95, 2003, and 2008. Default is 77.
        Note that currently the only distinction internally is between
        standards before 95, and those 95 and after. This may change later as
        more features are added.
    name_mangling : bool, optional
        If True, then the variables that would become identical in
        case-insensitive Fortran are mangled by appending different number
        of ``_`` at the end. If False, SymPy Will not interfere with naming of
        variables. [default=True]

    Examples
    ========

    >>> from sympy import fcode, symbols, Rational, sin, ceiling, floor
    >>> x, tau = symbols("x, tau")
    >>> fcode((2*tau)**Rational(7, 2))
    '      8*sqrt(2.0d0)*tau**(7.0d0/2.0d0)'
    >>> fcode(sin(x), assign_to="s")
    '      s = sin(x)'

    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg. Alternatively, the
    dictionary value can be a list of tuples i.e. [(argument_test,

    """
    # 调用 SymPy 的 fcode 函数将表达式转换为 Fortran 代码字符串
    return fcode(expr, assign_to=assign_to, **settings)
    from sympy.printing.fortran import FCodePrinter
    return FCodePrinter(settings).doprint(expr, assign_to=assign_to)



    # 导入 Fortran 代码打印器 FCodePrinter
    from sympy.printing.fortran import FCodePrinter
    # 使用给定的设置实例化 FCodePrinter，并调用其 doprint 方法生成 Fortran 代码
    # expr 是需要转换为 Fortran 代码的 SymPy 表达式
    # assign_to 是可选的，用于指定代码赋值目标的 SymPy 表达式
    # 返回生成的 Fortran 代码字符串
    return FCodePrinter(settings).doprint(expr, assign_to=assign_to)
def print_fcode(expr, **settings):
    """
    打印给定表达式的Fortran表示形式。

    :param expr: 要打印的表达式
    :param settings: 可选参数字典，传递给fcode函数的设置项
    """
    print(fcode(expr, **settings))

def cxxcode(expr, assign_to=None, standard='c++11', **settings):
    """
    返回给定表达式的C++等价代码，类似于ccode函数的功能。

    :param expr: 要转换为C++代码的表达式
    :param assign_to: 可选，指定赋值的目标变量
    :param standard: 可选，指定C++标准，默认为'c++11'
    :param settings: 其他可选参数，传递给C++代码打印器
    :return: 转换后的C++代码字符串
    """
    from sympy.printing.cxx import cxx_code_printers
    return cxx_code_printers[standard.lower()](settings).doprint(expr, assign_to)
```
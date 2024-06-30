# `D:\src\scipysrc\sympy\sympy\printing\smtlib.py`

```
import typing  # 导入 typing 模块，用于类型提示

import sympy  # 导入 sympy 库，用于符号计算
from sympy.core import Add, Mul  # 导入 sympy 的加法和乘法核心功能
from sympy.core import Symbol, Expr, Float, Rational, Integer, Basic  # 导入符号、表达式及数值类型
from sympy.core.function import UndefinedFunction, Function  # 导入未定义函数和函数相关功能
from sympy.core.relational import Relational, Unequality, Equality, LessThan, GreaterThan, StrictLessThan, StrictGreaterThan  # 导入关系运算符和比较关系
from sympy.functions.elementary.complexes import Abs  # 导入复数相关函数
from sympy.functions.elementary.exponential import exp, log, Pow  # 导入指数、对数和幂函数
from sympy.functions.elementary.hyperbolic import sinh, cosh, tanh  # 导入双曲函数
from sympy.functions.elementary.miscellaneous import Min, Max  # 导入最小值和最大值函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数
from sympy.functions.elementary.trigonometric import sin, cos, tan, asin, acos, atan, atan2  # 导入三角函数及其反函数
from sympy.logic.boolalg import And, Or, Xor, Implies, Boolean  # 导入布尔逻辑运算符
from sympy.logic.boolalg import BooleanTrue, BooleanFalse, BooleanFunction, Not, ITE  # 导入布尔逻辑相关功能
from sympy.printing.printer import Printer  # 导入打印功能
from sympy.sets import Interval  # 导入区间功能
from mpmath.libmp.libmpf import prec_to_dps, to_str as mlib_to_str  # 导入精度和转换函数
from sympy.assumptions.assume import AppliedPredicate  # 导入应用的谓词功能
from sympy.assumptions.relation.binrel import AppliedBinaryRelation  # 导入应用的二元关系功能
from sympy.assumptions.ask import Q  # 导入询问功能
from sympy.assumptions.relation.equality import StrictGreaterThanPredicate, StrictLessThanPredicate, GreaterThanPredicate, LessThanPredicate, EqualityPredicate  # 导入关系断言相关功能


class SMTLibPrinter(Printer):
    printmethod = "_smtlib"  # 设置打印方法为 "_smtlib"

    # 基于 dReal，一个用于解决可以编码为实数上一阶逻辑公式的自动推理工具。
    # dReal 特别擅长处理涉及多种非线性实函数的问题。
    _default_settings: dict = {
        'precision': None,  # 精度设为 None
        'known_types': {
            bool: 'Bool',  # 布尔类型映射为 'Bool'
            int: 'Int',    # 整数类型映射为 'Int'
            float: 'Real'  # 浮点数类型映射为 'Real'
        },
        'known_constants': {
            # pi: 'MY_VARIABLE_PI_DECLARED_ELSEWHERE',  # 已知常数，如 pi 的声明可以在其它地方
        },
        'known_functions': {
            Add: '+',   # 加法映射为 '+'
            Mul: '*',   # 乘法映射为 '*'

            Equality: '=',                # 等于运算映射为 '='
            LessThan: '<=',               # 小于等于映射为 '<='
            GreaterThan: '>=',            # 大于等于映射为 '>='
            StrictLessThan: '<',          # 严格小于映射为 '<'
            StrictGreaterThan: '>',       # 严格大于映射为 '>'

            EqualityPredicate(): '=',            # 等于断言映射为 '='
            LessThanPredicate(): '<=',           # 小于等于断言映射为 '<='
            GreaterThanPredicate(): '>=',        # 大于等于断言映射为 '>='
            StrictLessThanPredicate(): '<',      # 严格小于断言映射为 '<'
            StrictGreaterThanPredicate(): '>',   # 严格大于断言映射为 '>'

            exp: 'exp',   # 指数函数映射为 'exp'
            log: 'log',   # 对数函数映射为 'log'
            Abs: 'abs',   # 绝对值函数映射为 'abs'
            sin: 'sin',   # 正弦函数映射为 'sin'
            cos: 'cos',   # 余弦函数映射为 'cos'
            tan: 'tan',   # 正切函数映射为 'tan'
            asin: 'arcsin',  # 反正弦函数映射为 'arcsin'
            acos: 'arccos',  # 反余弦函数映射为 'arccos'
            atan: 'arctan',  # 反正切函数映射为 'arctan'
            atan2: 'arctan2',  # 反正切二元函数映射为 'arctan2'
            sinh: 'sinh',  # 双曲正弦函数映射为 'sinh'
            cosh: 'cosh',  # 双曲余弦函数映射为 'cosh'
            tanh: 'tanh',  # 双曲正切函数映射为 'tanh'
            Min: 'min',   # 最小值函数映射为 'min'
            Max: 'max',   # 最大值函数映射为 'max'
            Pow: 'pow',   # 幂函数映射为 'pow'

            And: 'and',   # 逻辑与映射为 'and'
            Or: 'or',     # 逻辑或映射为 'or'
            Xor: 'xor',   # 异或映射为 'xor'
            Not: 'not',   # 非映射为 'not'
            ITE: 'ite',   # 条件表达式映射为 'ite'
            Implies: '=>'  # 蕴含映射为 '=>'
        }
    }

    symbol_table: dict  # 符号表
    def __init__(self, settings: typing.Optional[dict] = None,
                 symbol_table=None):
        # 如果未提供设置，则使用空字典
        settings = settings or {}
        # 如果未提供符号表，则使用空字典
        self.symbol_table = symbol_table or {}
        # 调用父类的初始化方法，传入设置
        Printer.__init__(self, settings)
        # 从设置中获取精度值并保存在实例变量中
        self._precision = self._settings['precision']
        # 从设置中获取已知类型的字典并复制到实例变量中
        self._known_types = dict(self._settings['known_types'])
        # 从设置中获取已知常量的字典并复制到实例变量中
        self._known_constants = dict(self._settings['known_constants'])
        # 从设置中获取已知函数的字典并复制到实例变量中
        self._known_functions = dict(self._settings['known_functions'])

        # 确保所有已知类型的名称是合法的
        for _ in self._known_types.values(): assert self._is_legal_name(_)
        # 确保所有已知常量的名称是合法的
        for _ in self._known_constants.values(): assert self._is_legal_name(_)
        # 确保所有已知函数的名称是合法的（这行代码被注释掉了）
        # for _ in self._known_functions.values(): assert self._is_legal_name(_)  # +, *, <, >, etc.

    def _is_legal_name(self, s: str):
        # 如果字符串为空，则不合法
        if not s: return False
        # 如果字符串以数字开头，则不合法
        if s[0].isnumeric(): return False
        # 字符串中所有字符必须是字母数字或下划线
        return all(_.isalnum() or _ == '_' for _ in s)

    def _s_expr(self, op: str, args: typing.Union[list, tuple]) -> str:
        # 将参数列表中的每个元素转换成字符串，并用空格连接起来
        args_str = ' '.join(
            a if isinstance(a, str)
            else self._print(a)
            for a in args
        )
        # 返回形如 (op args_str) 的表达式字符串
        return f'({op} {args_str})'

    def _print_Function(self, e):
        # 如果 e 是已知函数中的键，则获取对应的操作符
        if e in self._known_functions:
            op = self._known_functions[e]
        # 如果 e 的类型是已知函数中的键，则获取对应的操作符
        elif type(e) in self._known_functions:
            op = self._known_functions[type(e)]
        # 如果 e 的类型是 UndefinedFunction，则直接使用其名称
        elif type(type(e)) == UndefinedFunction:
            op = e.name
        # 如果 e 是 AppliedBinaryRelation 类型，并且其函数在已知函数中，则获取对应的操作符并返回其字符串表示
        elif isinstance(e, AppliedBinaryRelation) and e.function in self._known_functions:
            op = self._known_functions[e.function]
            return self._s_expr(op, e.arguments)
        else:
            op = self._known_functions[e]  # 如果未找到对应的操作符，则抛出 KeyError

        # 根据操作符和参数列表生成表达式字符串并返回
        return self._s_expr(op, e.args)

    def _print_Relational(self, e: Relational):
        # 调用 _print_Function 方法处理 Relational 类型对象并返回其字符串表示
        return self._print_Function(e)

    def _print_BooleanFunction(self, e: BooleanFunction):
        # 调用 _print_Function 方法处理 BooleanFunction 类型对象并返回其字符串表示
        return self._print_Function(e)

    def _print_Expr(self, e: Expr):
        # 调用 _print_Function 方法处理 Expr 类型对象并返回其字符串表示
        return self._print_Function(e)

    def _print_Unequality(self, e: Unequality):
        # 如果 e 的类型在已知函数中，则返回 Relational 的字符串表示（默认情况）
        if type(e) in self._known_functions:
            return self._print_Relational(e)
        else:
            # 获取 Equality 和 Not 操作符对应的字符串
            eq_op = self._known_functions[Equality]
            not_op = self._known_functions[Not]
            # 构造不等式的字符串表示
            return self._s_expr(not_op, [self._s_expr(eq_op, e.args)])

    def _print_Piecewise(self, e: Piecewise):
        def _print_Piecewise_recursive(args: typing.Union[list, tuple]):
            # 从参数列表中获取表达式和条件
            e, c = args[0]
            # 如果只有一个分支，则条件必须是 True 或 BooleanTrue 类型
            if len(args) == 1:
                assert (c is True) or isinstance(c, BooleanTrue)
                # 返回表达式 e 的字符串表示
                return self._print(e)
            else:
                # 获取 ITE 操作符对应的字符串
                ite = self._known_functions[ITE]
                # 递归处理剩余的分支，并构造完整的 Piecewise 表达式字符串
                return self._s_expr(ite, [
                    c, e, _print_Piecewise_recursive(args[1:])
                ])

        # 调用递归函数处理 Piecewise 类型对象并返回其字符串表示
        return _print_Piecewise_recursive(e.args)
    # 打印 Interval 对象的字符串表示
    def _print_Interval(self, e: Interval):
        # 如果起始和结束都是无穷，则返回空字符串
        if e.start.is_infinite and e.end.is_infinite:
            return ''
        # 如果起始和结束的无穷属性不一致，则抛出异常
        elif e.start.is_infinite != e.end.is_infinite:
            raise ValueError(f'One-sided intervals (`{e}`) are not supported in SMT.')
        else:
            # 返回形如 '[start, end]' 的字符串
            return f'[{e.start}, {e.end}]'

    # 打印 AppliedPredicate 对象的字符串表示
    def _print_AppliedPredicate(self, e: AppliedPredicate):
        # 根据不同的函数类型生成相应的关系表达式
        if e.function == Q.positive:
            rel = Q.gt(e.arguments[0], 0)
        elif e.function == Q.negative:
            rel = Q.lt(e.arguments[0], 0)
        elif e.function == Q.zero:
            rel = Q.eq(e.arguments[0], 0)
        elif e.function == Q.nonpositive:
            rel = Q.le(e.arguments[0], 0)
        elif e.function == Q.nonnegative:
            rel = Q.ge(e.arguments[0], 0)
        elif e.function == Q.nonzero:
            rel = Q.ne(e.arguments[0], 0)
        else:
            # 如果函数类型不支持，则抛出异常
            raise ValueError(f"Predicate (`{e}`) is not handled.")

        # 返回对应关系表达式的字符串表示
        return self._print_AppliedBinaryRelation(rel)

    # 打印 AppliedBinaryRelation 对象的字符串表示
    def _print_AppliedBinaryRelation(self, e: AppliedPredicate):
        # 如果是不等式运算，则调用 _print_Unequality 方法
        if e.function == Q.ne:
            return self._print_Unequality(Unequality(*e.arguments))
        else:
            # 否则调用 _print_Function 方法
            return self._print_Function(e)

    """
    todo: Sympy 目前不支持量词（quantifiers），但在 SMT 中量词可能非常有用。
    目前用户可以扩展这个类并构建自己的量词支持。
    例如，可以查看 `test_quantifier_extensions()` 在 test_smtlib.py 中的示例。
    """

    # def _print_ForAll(self, e: ForAll):
    #     return self._s('forall', [
    #         self._s('', [
    #             self._s(sym.name, [self._type_name(sym), Interval(start, end)])
    #             for sym, start, end in e.limits
    #         ]),
    #         e.function
    #     ])

    # 打印 BooleanTrue 对象的字符串表示
    def _print_BooleanTrue(self, x: BooleanTrue):
        return 'true'

    # 打印 BooleanFalse 对象的字符串表示
    def _print_BooleanFalse(self, x: BooleanFalse):
        return 'false'

    # 打印 Float 对象的字符串表示
    def _print_Float(self, x: Float):
        # 获取精度对应的位数
        dps = prec_to_dps(x._prec)
        # 将 Float 对象转换为字符串表示
        str_real = mlib_to_str(x._mpf_, dps, strip_zeros=True, min_fixed=None, max_fixed=None)

        # 如果字符串中包含 'e'，则表示为科学计数法
        if 'e' in str_real:
            (mant, exp) = str_real.split('e')

            # 处理指数部分的正负号
            if exp[0] == '+':
                exp = exp[1:]

            mul = self._known_functions[Mul]
            pow = self._known_functions[Pow]

            # 返回科学计数法的字符串表示
            return r"(%s %s (%s 10 %s))" % (mul, mant, pow, exp)
        elif str_real in ["+inf", "-inf"]:
            # 如果是正无穷或负无穷，则抛出异常
            raise ValueError("Infinite values are not supported in SMT.")
        else:
            # 否则返回普通实数的字符串表示
            return str_real

    # 打印 float 类型数据的字符串表示
    def _print_float(self, x: float):
        # 调用 _print_Float 方法打印 Float 对象的字符串表示
        return self._print(Float(x))

    # 打印 Rational 对象的字符串表示
    def _print_Rational(self, x: Rational):
        # 返回 Rational 对象的分子和分母的分数表示
        return self._s_expr('/', [x.p, x.q])

    # 打印 Integer 对象的字符串表示
    def _print_Integer(self, x: Integer):
        # 断言分母为 1，即为整数
        assert x.q == 1
        # 返回整数的字符串表示
        return str(x.p)

    # 打印 int 类型数据的字符串表示
    def _print_int(self, x: int):
        # 直接返回整数的字符串表示
        return str(x)

    # 打印 Symbol 对象的字符串表示
    def _print_Symbol(self, x: Symbol):
        # 断言符号名称合法
        assert self._is_legal_name(x.name)
        # 返回符号的名称字符串
        return x.name
    # 打印给定常数对应的名称，如果存在的话
    def _print_NumberSymbol(self, x):
        name = self._known_constants.get(x)
        if name:
            return name
        else:
            # 如果找不到对应的常数名称，计算出数值后打印为浮点数格式
            f = x.evalf(self._precision) if self._precision else x.evalf()
            return self._print_Float(f)

    # 打印未定义函数的名称，并确保名称合法
    def _print_UndefinedFunction(self, x):
        assert self._is_legal_name(x.name)
        return x.name

    # 打印常数 e 的表示方式，考虑是否在已知函数中
    def _print_Exp1(self, x):
        return (
            # 如果 e 在已知函数中，则使用函数形式打印
            self._print_Function(exp(1, evaluate=False))
            # 否则打印为常数符号
            if exp in self._known_functions else
            self._print_NumberSymbol(x)
        )

    # 抛出未实现异常，表示不能将给定表达式转换为 SMT（Satisfiability Modulo Theories）格式
    def emptyPrinter(self, expr):
        raise NotImplementedError(f'Cannot convert `{repr(expr)}` of type `{type(expr)}` to SMT.')
# 定义函数 smtlib_code，将 SymPy 表达式或系统转换为 smtlib 代码字符串
def smtlib_code(
    expr,
    auto_assert=True, auto_declare=True,  # 设置默认参数：自动进行断言和声明
    precision=None,  # 精度参数，用于数值如 pi 的 evalf() 操作
    symbol_table=None,  # 符号表，将符号映射到 Python 类型
    known_types=None, known_constants=None, known_functions=None,  # 已知的类型、常量和函数的映射表
    prefix_expressions=None, suffix_expressions=None,  # 前缀和后缀表达式列表
    log_warn=None  # 警告日志记录函数，用于记录潜在风险操作中的所有警告
):
    r"""Converts ``expr`` to a string of smtlib code.

    Parameters
    ==========

    expr : Expr | List[Expr]
        A SymPy expression or system to be converted.
    auto_assert : bool, optional
        If false, do not modify expr and produce only the S-Expression equivalent of expr.
        If true, assume expr is a system and assert each boolean element.
    auto_declare : bool, optional
        If false, do not produce declarations for the symbols used in expr.
        If true, prepend all necessary declarations for variables used in expr based on symbol_table.
    precision : integer, optional
        The ``evalf(..)`` precision for numbers such as pi.
    symbol_table : dict, optional
        A dictionary where keys are ``Symbol`` or ``Function`` instances and values are their Python type i.e. ``bool``, ``int``, ``float``, or ``Callable[...]``.
        If incomplete, an attempt will be made to infer types from ``expr``.
    known_types: dict, optional
        A dictionary where keys are ``bool``, ``int``, ``float`` etc. and values are their corresponding SMT type names.
        If not given, a partial listing compatible with several solvers will be used.
    known_functions : dict, optional
        A dictionary where keys are ``Function``, ``Relational``, ``BooleanFunction``, or ``Expr`` instances and values are their SMT string representations.
        If not given, a partial listing optimized for dReal solver (but compatible with others) will be used.
    known_constants: dict, optional
        A dictionary where keys are ``NumberSymbol`` instances and values are their SMT variable names.
        When using this feature, extra caution must be taken to avoid naming collisions between user symbols and listed constants.
        If not given, constants will be expanded inline i.e. ``3.14159`` instead of ``MY_SMT_VARIABLE_FOR_PI``.
    prefix_expressions: list, optional
        A list of lists of ``str`` and/or expressions to convert into SMTLib and prefix to the output.
    suffix_expressions: list, optional
        A list of lists of ``str`` and/or expressions to convert into SMTLib and postfix to the output.
    log_warn: lambda function, optional
        A function to record all warnings during potentially risky operations.
        Soundness is a core value in SMT solving, so it is good to log all assumptions made.

    Examples
    ========
    >>> from sympy import smtlib_code, symbols, sin, Eq
    >>> x = symbols('x')
    >>> smtlib_code(sin(x).series(x).removeO(), log_warn=print)
    Could not infer type of `x`. Defaulting to float.
    Non-Boolean expression `x**5/120 - x**3/6 + x` will not be asserted. Converting to SMTLib verbatim.

    """
    """
    '(declare-const x Real)\n(+ x (* (/ -1 6) (pow x 3)) (* (/ 1 120) (pow x 5)))'
    """

    # 导入 sympy 的 Rational 类
    >>> from sympy import Rational
    # 定义符号 x, y, tau
    >>> x, y, tau = symbols("x, y, tau")
    # 调用 smtlib_code 函数，计算 (2*tau)**Rational(7, 2)，并输出警告信息到 log_warn
    >>> smtlib_code((2*tau)**Rational(7, 2), log_warn=print)
    # 输出警告信息，指示无法推断 tau 的类型，默认为 float
    Could not infer type of `tau`. Defaulting to float.
    # 由于表达式非布尔值，不会被断言为真。将其直接转换为 SMTLib 格式
    Non-Boolean expression `8*sqrt(2)*tau**(7/2)` will not be asserted. Converting to SMTLib verbatim.
    # 返回表示表达式的 SMTLib 代码字符串
    '(declare-const tau Real)\n(* 8 (pow 2 (/ 1 2)) (pow tau (/ 7 2)))'

    """
    ``Piecewise`` expressions are implemented with ``ite`` expressions by default.
    Note that if the ``Piecewise`` lacks a default term, represented by
    ``(expr, True)`` then an error will be thrown.  This is to prevent
    generating an expression that may not evaluate to anything.
    """

    # 导入 sympy 的 Piecewise 类
    >>> from sympy import Piecewise
    # 创建 Piecewise 对象 pw，定义条件表达式
    >>> pw = Piecewise((x + 1, x > 0), (x, True))
    # 调用 smtlib_code 函数，将等式 Eq(pw, 3) 转换为 SMTLib 格式，并使用 symbol_table 定义的转换规则
    >>> smtlib_code(Eq(pw, 3), symbol_table={x: float}, log_warn=print)
    # 返回表示等式的 SMTLib 代码字符串
    '(declare-const x Real)\n(assert (= (ite (> x 0) (+ 1 x) x) 3))'

    """
    Custom printing can be defined for certain types by passing a dictionary of
    PythonType : "SMT Name" to the ``known_types``, ``known_constants``, and ``known_functions`` kwargs.
    """

    # 导入 typing 模块中的 Callable 类
    >>> from typing import Callable
    # 导入 sympy 的 Function 和 Add 类
    >>> from sympy import Function, Add
    # 定义函数 f 和 g
    >>> f = Function('f')
    >>> g = Function('g')
    # 定义 SMT 求解器可理解的函数字典 smt_builtin_funcs 和 user_def_funcs
    >>> smt_builtin_funcs = {  # functions our SMT solver will understand
    ...   f: "existing_smtlib_fcn",
    ...   Add: "sum",
    ... }
    >>> user_def_funcs = {  # functions defined by the user must have their types specified explicitly
    ...   g: Callable[[int], float],
    ... }
    # 调用 smtlib_code 函数，将 f(x) + g(x) 转换为 SMTLib 格式，并使用 symbol_table 和 known_functions 定义的转换规则
    >>> smtlib_code(f(x) + g(x), symbol_table=user_def_funcs, known_functions=smt_builtin_funcs, log_warn=print)
    # 输出警告信息，指示表达式 f(x) + g(x) 不会被断言为真。将其直接转换为 SMTLib 格式
    Non-Boolean expression `f(x) + g(x)` will not be asserted. Converting to SMTLib verbatim.
    # 返回表示表达式的 SMTLib 代码字符串
    '(declare-const x Int)\n(declare-fun g (Int) Real)\n(sum (existing_smtlib_fcn x) (g x))'

    """
    log_warn = log_warn or (lambda _: None)
    """

    # 如果 log_warn 为 None，则设定为一个空的 lambda 函数

    if not isinstance(expr, list): expr = [expr]
    expr = [
        sympy.sympify(_, strict=True, evaluate=False, convert_xor=False)
        for _ in expr
    ]

    # 如果 expr 不是列表，则将其转换为列表
    # 使用 sympy.sympify 将 expr 中的每个表达式字符串转换为 sympy 表达式对象

    if not symbol_table: symbol_table = {}
    symbol_table = _auto_infer_smtlib_types(
        *expr, symbol_table=symbol_table
    )

    # 如果 symbol_table 不存在，则设定为空字典；调用 _auto_infer_smtlib_types 函数推断表达式的 SMTLib 类型

    # See [FALLBACK RULES]
    # Need SMTLibPrinter to populate known_functions and known_constants first.

    settings = {}
    if precision: settings['precision'] = precision
    del precision

    if known_types: settings['known_types'] = known_types
    del known_types

    if known_functions: settings['known_functions'] = known_functions
    del known_functions

    if known_constants: settings['known_constants'] = known_constants
    del known_constants

    if not prefix_expressions: prefix_expressions = []
    if not suffix_expressions: suffix_expressions = []

    # 设定 settings 字典，并根据条件设置 'precision', 'known_types', 'known_functions', 'known_constants' 键值对

    p = SMTLibPrinter(settings, symbol_table)
    del symbol_table

    # 创建 SMTLibPrinter 对象 p，并使用 settings 和 symbol_table 初始化

    # [FALLBACK RULES]
    ```
    # 遍历表达式列表中的每一个表达式
    for e in expr:
        # 遍历当前表达式中的符号（变量和函数）
        for sym in e.atoms(Symbol, Function):
            # 如果符号是一个变量且不在已知常量和符号表中
            if (
                sym.is_Symbol and
                sym not in p._known_constants and
                sym not in p.symbol_table
            ):
                # 记录警告日志，指示无法推断变量 `{sym}` 的类型，默认为 float
                log_warn(f"Could not infer type of `{sym}`. Defaulting to float.")
                # 将该变量默认映射为 float 类型
                p.symbol_table[sym] = float

            # 如果符号是一个函数且不在已知函数和符号表中，并且不是分段定义函数
            if (
                sym.is_Function and
                type(sym) not in p._known_functions and
                type(sym) not in p.symbol_table and
                not sym.is_Piecewise
            ):
                # 抛出类型错误，指示未知类型的未定义函数 `{sym}`
                raise TypeError(
                    f"Unknown type of undefined function `{sym}`. "
                    f"Must be mapped to ``str`` in known_functions or mapped to ``Callable[..]`` in symbol_table."
                )

    # 声明列表，用于存储自动生成的声明
    declarations = []

    # 如果需要自动声明变量和函数
    if auto_declare:
        # 收集所有未知常量的名称和符号
        constants = {sym.name: sym for e in expr for sym in e.free_symbols
                     if sym not in p._known_constants}
        # 收集所有未知函数的名称和函数对象
        functions = {fnc.name: fnc for e in expr for fnc in e.atoms(Function)
                     if type(fnc) not in p._known_functions and not fnc.is_Piecewise}
        # 生成常量和函数的声明语句，并添加到声明列表中
        declarations = \
            [
                _auto_declare_smtlib(sym, p, log_warn)
                for sym in constants.values()
            ] + [
                _auto_declare_smtlib(fnc, p, log_warn)
                for fnc in functions.values()
            ]
        # 过滤掉空的声明
        declarations = [decl for decl in declarations if decl]

    # 如果需要自动断言表达式
    if auto_assert:
        # 自动生成表达式的断言语句，并替换原始表达式列表
        expr = [_auto_assert_smtlib(e, p, log_warn) for e in expr]

    # 将所有输出组合成一个字符串，并用换行符连接
    return '\n'.join([
        # ';; PREFIX EXPRESSIONS',
        # 将前缀表达式添加到输出列表中，如果是字符串直接添加，否则调用打印方法
        *[
            e if isinstance(e, str) else p.doprint(e)
            for e in prefix_expressions
        ],

        # ';; DECLARATIONS',
        # 将声明语句添加到输出列表中，并按字母顺序排序
        *sorted(e for e in declarations),

        # ';; EXPRESSIONS',
        # 将所有表达式添加到输出列表中，如果是字符串直接添加，否则调用打印方法
        *[
            e if isinstance(e, str) else p.doprint(e)
            for e in expr
        ],

        # ';; SUFFIX EXPRESSIONS',
        # 将后缀表达式添加到输出列表中，如果是字符串直接添加，否则调用打印方法
        *[
            e if isinstance(e, str) else p.doprint(e)
            for e in suffix_expressions
        ],
    ])
def _auto_declare_smtlib(sym: typing.Union[Symbol, Function], p: SMTLibPrinter, log_warn: typing.Callable[[str], None]):
    # 如果符号是 Symbol 类型
    if sym.is_Symbol:
        # 获取符号的类型签名
        type_signature = p.symbol_table[sym]
        assert isinstance(type_signature, type)
        # 根据已知的类型映射获取具体类型
        type_signature = p._known_types[type_signature]
        # 构建并返回声明常量的 SMT-LIB 表达式
        return p._s_expr('declare-const', [sym, type_signature])

    # 如果符号是 Function 类型
    elif sym.is_Function:
        # 获取函数的类型签名
        type_signature = p.symbol_table[type(sym)]
        assert callable(type_signature)
        # 获取函数参数和返回类型的类型映射
        type_signature = [p._known_types[_] for _ in type_signature.__args__]
        assert len(type_signature) > 0
        # 构建参数签名和返回类型
        params_signature = f"({' '.join(type_signature[:-1])})"
        return_signature = type_signature[-1]
        # 构建并返回声明函数的 SMT-LIB 表达式
        return p._s_expr('declare-fun', [type(sym), params_signature, return_signature])

    # 如果符号既不是 Symbol 也不是 Function 类型
    else:
        # 记录警告日志，表示不能声明非 Symbol/Function 类型的符号
        log_warn(f"Non-Symbol/Function `{sym}` will not be declared.")
        return None


def _auto_assert_smtlib(e: Expr, p: SMTLibPrinter, log_warn: typing.Callable[[str], None]):
    # 如果表达式是布尔类型，或者在符号表中，并且类型为 bool
    if isinstance(e, Boolean) or (
        e in p.symbol_table and p.symbol_table[e] == bool
    ) or (
        e.is_Function and
        type(e) in p.symbol_table and
        p.symbol_table[type(e)].__args__[-1] == bool
    ):
        # 构建并返回断言的 SMT-LIB 表达式
        return p._s_expr('assert', [e])
    else:
        # 记录警告日志，表示不能断言非布尔表达式，将其直接转换为 SMT-LIB 表达式
        log_warn(f"Non-Boolean expression `{e}` will not be asserted. Converting to SMTLib verbatim.")
        return e


def _auto_infer_smtlib_types(
    *exprs: Basic,
    symbol_table: typing.Optional[dict] = None
) -> dict:
    # [TYPE INFERENCE RULES]
    # X is alone in an expr => X is bool
    # X in BooleanFunction.args => X is bool
    # X matches to a bool param of a symbol_table function => X is bool
    # X matches to an int param of a symbol_table function => X is int
    # X.is_integer => X is int
    # X == Y, where X is T => Y is T

    # [FALLBACK RULES]
    # see _auto_declare_smtlib(..)
    # X is not bool and X is not int and X is Symbol => X is float
    # else (e.g. X is Function) => error. must be specified explicitly.

    _symbols = dict(symbol_table) if symbol_table else {}

    def safe_update(syms: set, inf):
        # 安全更新符号的类型映射
        for s in syms:
            assert s.is_Symbol
            # 如果已经存在不一致的类型映射，抛出类型错误
            if (old_type := _symbols.setdefault(s, inf)) != inf:
                raise TypeError(f"Could not infer type of `{s}`. Apparently both `{old_type}` and `{inf}`?")

    # 显式类型推断规则

    # 对于每个表达式中的符号，如果是 Symbol 类型，推断为 bool
    safe_update({
        e
        for e in exprs
        if e.is_Symbol
    }, bool)

    # 对于每个表达式中布尔函数的参数符号，推断为 bool
    safe_update({
        symbol
        for e in exprs
        for boolfunc in e.atoms(BooleanFunction)
        for symbol in boolfunc.args
        if symbol.is_Symbol
    }, bool)

    # 对于每个表达式中函数的参数符号，如果匹配符号表中函数的 bool 类型参数，推断为 bool
    safe_update({
        symbol
        for e in exprs
        for boolfunc in e.atoms(Function)
        if type(boolfunc) in _symbols
        for symbol, param in zip(boolfunc.args, _symbols[type(boolfunc)].__args__)
        if symbol.is_Symbol and param == bool
    }, bool)
    # 更新符号类型字典，将符号关联到整数类型
    safe_update({
        symbol  # 遍历表达式中的函数，提取参数符号
        for e in exprs
        for intfunc in e.atoms(Function)
        if type(intfunc) in _symbols
        for symbol, param in zip(intfunc.args, _symbols[type(intfunc)].__args__)
        if symbol.is_Symbol and param == int
    }, int)

    # 更新符号类型字典，将符号关联到整数类型
    safe_update({
        symbol  # 遍历表达式中的符号，识别整数类型的符号
        for e in exprs
        for symbol in e.atoms(Symbol)
        if symbol.is_integer
    }, int)

    # 更新符号类型字典，将符号关联到浮点数类型
    safe_update({
        symbol  # 遍历表达式中的符号，识别实数但不是整数类型的符号
        for e in exprs
        for symbol in e.atoms(Symbol)
        if symbol.is_real and not symbol.is_integer
    }, float)

    # 等式关系规则处理
    rels = [rel for expr in exprs for rel in expr.atoms(Equality)]
    rels = [
               (rel.lhs, rel.rhs) for rel in rels if rel.lhs.is_Symbol
           ] + [
               (rel.rhs, rel.lhs) for rel in rels if rel.rhs.is_Symbol
           ]
    for infer, reltd in rels:
        # 推断符号的类型
        inference = (
            _symbols[infer] if infer in _symbols else
            _symbols[reltd] if reltd in _symbols else
            _symbols[type(reltd)].__args__[-1]
            if reltd.is_Function and type(reltd) in _symbols else
            bool if reltd.is_Boolean else
            int if reltd.is_integer or reltd.is_Integer else
            float if reltd.is_real else
            None
        )
        if inference: safe_update({infer}, inference)

    # 返回更新后的符号类型字典
    return _symbols
```
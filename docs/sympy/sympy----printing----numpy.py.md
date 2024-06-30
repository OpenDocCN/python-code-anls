# `D:\src\scipysrc\sympy\sympy\printing\numpy.py`

```
# 从 sympy.core 模块中导入 S 符号
from sympy.core import S
# 从 sympy.core.function 模块中导入 Lambda 函数
from sympy.core.function import Lambda
# 从 sympy.core.power 模块中导入 Pow 函数
from sympy.core.power import Pow
# 从当前包中导入 PythonCodePrinter、_known_functions_math、_print_known_const、_print_known_func、_unpack_integral_limits、ArrayPrinter
from .pycode import PythonCodePrinter, _known_functions_math, _print_known_const, _print_known_func, _unpack_integral_limits, ArrayPrinter
# 从当前包中导入 CodePrinter 类
from .codeprinter import CodePrinter

# 定义不在 numpy 中的函数列表
_not_in_numpy = 'erf erfc factorial gamma loggamma'.split()
# 筛选出在 numpy 中的函数列表
_in_numpy = [(k, v) for k, v in _known_functions_math.items() if k not in _not_in_numpy]
# 创建 _known_functions_numpy 字典，将函数名映射到 numpy 中的函数名
_known_functions_numpy = dict(_in_numpy, **{
    'acos': 'arccos',
    'acosh': 'arccosh',
    'asin': 'arcsin',
    'asinh': 'arcsinh',
    'atan': 'arctan',
    'atan2': 'arctan2',
    'atanh': 'arctanh',
    'exp2': 'exp2',
    'sign': 'sign',
    'logaddexp': 'logaddexp',
    'logaddexp2': 'logaddexp2',
    'isnan': 'isnan'
})
# 创建 _known_constants_numpy 字典，将常数名映射到 numpy 中的常数名
_known_constants_numpy = {
    'Exp1': 'e',
    'Pi': 'pi',
    'EulerGamma': 'euler_gamma',
    'NaN': 'nan',
    'Infinity': 'inf',
}

# 创建 _numpy_known_functions 字典，将函数名映射到 numpy 模块中的完整路径
_numpy_known_functions = {k: 'numpy.' + v for k, v in _known_functions_numpy.items()}
# 创建 _numpy_known_constants 字典，将常数名映射到 numpy 模块中的完整路径
_numpy_known_constants = {k: 'numpy.' + v for k, v in _known_constants_numpy.items()}

# 定义 NumPyPrinter 类，继承自 ArrayPrinter 和 PythonCodePrinter
class NumPyPrinter(ArrayPrinter, PythonCodePrinter):
    """
    Numpy printer which handles vectorized piecewise functions,
    logical operators, etc.
    """

    # 定义模块名为 'numpy'
    _module = 'numpy'
    # 将 _known_functions_numpy 中的函数映射赋值给 _kf
    _kf = _numpy_known_functions
    # 将 _known_constants_numpy 中的常数映射赋值给 _kc
    _kc = _numpy_known_constants

    # 初始化方法，接受一个 settings 参数，传递给 CodePrinter.__init__()
    def __init__(self, settings=None):
        """
        `settings` is passed to CodePrinter.__init__()
        `module` specifies the array module to use, currently 'NumPy', 'CuPy'
        or 'JAX'.
        """
        # 设置语言属性为 "Python with numpy"
        self.language = "Python with {}".format(self._module)
        # 设置打印方法属性为 "_numpycode"
        self.printmethod = "_{}code".format(self._module)

        # 将 PythonCodePrinter 类中的 _kf 合并到当前类的 _kf 中
        self._kf = {**PythonCodePrinter._kf, **self._kf}

        # 调用父类 PythonCodePrinter 的初始化方法
        super().__init__(settings=settings)

    # 定义 _print_seq 方法，用于打印序列为元组格式
    def _print_seq(self, seq):
        "General sequence printer: converts to tuple"
        # 以逗号分隔序列中的每个元素，转换成元组格式
        delimiter=', '
        return '({},)'.format(delimiter.join(self._print(item) for item in seq))

    # 定义 _print_NegativeInfinity 方法，打印负无穷为 '-' 加上正无穷的打印结果
    def _print_NegativeInfinity(self, expr):
        return '-' + self._print(S.Infinity)

    # 定义 _print_MatMul 方法，打印矩阵乘法
    def _print_MatMul(self, expr):
        "Matrix multiplication printer"
        # 如果矩阵乘法的第一个系数矩阵不是单位矩阵，则将其加入到表达式列表中
        if expr.as_coeff_matrices()[0] is not S.One:
            expr_list = expr.as_coeff_matrices()[1]+[(expr.as_coeff_matrices()[0])]
            return '({})'.format(').dot('.join(self._print(i) for i in expr_list))
        # 否则，直接打印矩阵乘法的每个参数
        return '({})'.format(').dot('.join(self._print(i) for i in expr.args))

    # 定义 _print_MatPow 方法，打印矩阵幂
    def _print_MatPow(self, expr):
        "Matrix power printer"
        return '{}({}, {})'.format(self._module_format(self._module + '.linalg.matrix_power'),
            self._print(expr.args[0]), self._print(expr.args[1]))

    # 定义 _print_Inverse 方法，打印矩阵求逆
    def _print_Inverse(self, expr):
        "Matrix inverse printer"
        return '{}({})'.format(self._module_format(self._module + '.linalg.inv'),
            self._print(expr.args[0]))
    def _print_DotProduct(self, expr):
        # DotProduct allows any shape order, but numpy.dot does matrix
        # multiplication, so we have to make sure it gets 1 x n by n x 1.
        # 解释 DotProduct 允许任意形状的顺序，但 numpy.dot 执行矩阵乘法，
        # 因此我们要确保输入是 1 x n 和 n x 1 的形式。
        arg1, arg2 = expr.args
        if arg1.shape[0] != 1:
            arg1 = arg1.T  # 如果第一个参数不是行向量，转置为行向量
        if arg2.shape[1] != 1:
            arg2 = arg2.T  # 如果第二个参数不是列向量，转置为列向量

        return "%s(%s, %s)" % (self._module_format(self._module + '.dot'),
                               self._print(arg1),
                               self._print(arg2))

    def _print_MatrixSolve(self, expr):
        # 返回矩阵求解的字符串表示
        return "%s(%s, %s)" % (self._module_format(self._module + '.linalg.solve'),
                               self._print(expr.matrix),
                               self._print(expr.vector))

    def _print_ZeroMatrix(self, expr):
        # 返回零矩阵的字符串表示
        return '{}({})'.format(self._module_format(self._module + '.zeros'),
            self._print(expr.shape))

    def _print_OneMatrix(self, expr):
        # 返回全一矩阵的字符串表示
        return '{}({})'.format(self._module_format(self._module + '.ones'),
            self._print(expr.shape))

    def _print_FunctionMatrix(self, expr):
        # 返回由函数生成的矩阵的字符串表示
        from sympy.abc import i, j
        lamda = expr.lamda
        if not isinstance(lamda, Lambda):
            lamda = Lambda((i, j), lamda(i, j))
        return '{}(lambda {}: {}, {})'.format(self._module_format(self._module + '.fromfunction'),
            ', '.join(self._print(arg) for arg in lamda.args[0]),
            self._print(lamda.args[1]), self._print(expr.shape))

    def _print_HadamardProduct(self, expr):
        # 返回哈达玛积（元素相乘）的字符串表示
        func = self._module_format(self._module + '.multiply')
        return ''.join('{}({}, '.format(func, self._print(arg)) \
            for arg in expr.args[:-1]) + "{}{}".format(self._print(expr.args[-1]),
            ')' * (len(expr.args) - 1))

    def _print_KroneckerProduct(self, expr):
        # 返回克罗内克积的字符串表示
        func = self._module_format(self._module + '.kron')
        return ''.join('{}({}, '.format(func, self._print(arg)) \
            for arg in expr.args[:-1]) + "{}{}".format(self._print(expr.args[-1]),
            ')' * (len(expr.args) - 1))

    def _print_Adjoint(self, expr):
        # 返回伴随矩阵的字符串表示
        return '{}({}({}))'.format(
            self._module_format(self._module + '.conjugate'),
            self._module_format(self._module + '.transpose'),
            self._print(expr.args[0]))

    def _print_DiagonalOf(self, expr):
        # 返回矩阵的对角线向量的字符串表示
        vect = '{}({})'.format(
            self._module_format(self._module + '.diag'),
            self._print(expr.arg))
        return '{}({}, (-1, 1))'.format(
            self._module_format(self._module + '.reshape'), vect)

    def _print_DiagMatrix(self, expr):
        # 返回由向量生成的对角矩阵的字符串表示
        return '{}({})'.format(self._module_format(self._module + '.diagflat'),
            self._print(expr.args[0]))
    # 打印对角矩阵表达式的字符串表示
    def _print_DiagonalMatrix(self, expr):
        return '{}({}, {}({}, {}))'.format(self._module_format(self._module + '.multiply'),
            self._print(expr.arg), self._module_format(self._module + '.eye'),
            self._print(expr.shape[0]), self._print(expr.shape[1]))

    # 打印分段函数的字符串表示
    def _print_Piecewise(self, expr):
        "Piecewise function printer"
        from sympy.logic.boolalg import ITE, simplify_logic
        def print_cond(cond):
            """ Problem having an ITE in the cond. """
            if cond.has(ITE):
                return self._print(simplify_logic(cond))
            else:
                return self._print(cond)
        # 构建分段函数的表达式部分字符串
        exprs = '[{}]'.format(','.join(self._print(arg.expr) for arg in expr.args))
        # 构建分段函数的条件部分字符串
        conds = '[{}]'.format(','.join(print_cond(arg.cond) for arg in expr.args))
        # 返回使用条件和表达式构建的字符串表示，并指定默认值为 S.NaN
        return '{}({}, {}, default={})'.format(
            self._module_format(self._module + '.select'), conds, exprs,
            self._print(S.NaN))

    # 打印关系表达式（如相等和不等式）的字符串表示
    def _print_Relational(self, expr):
        "Relational printer for Equality and Unequality"
        # 定义关系运算符到字符串的映射
        op = {
            '==' :'equal',
            '!=' :'not_equal',
            '<'  :'less',
            '<=' :'less_equal',
            '>'  :'greater',
            '>=' :'greater_equal',
        }
        # 如果表达式的关系操作符在映射中，则使用对应的字符串格式化输出
        if expr.rel_op in op:
            lhs = self._print(expr.lhs)
            rhs = self._print(expr.rhs)
            return '{op}({lhs}, {rhs})'.format(op=self._module_format(self._module + '.'+op[expr.rel_op]),
                                               lhs=lhs, rhs=rhs)
        # 否则调用超类的默认打印方法处理表达式
        return super()._print_Relational(expr)

    # 打印逻辑与表达式的字符串表示
    def _print_And(self, expr):
        "Logical And printer"
        # 由于LambdaPrinter使用Python中的'and'关键字，所以需要重写
        # 返回使用逻辑与函数（logical_and）处理的字符串表示
        return '{}.reduce(({}))'.format(self._module_format(self._module + '.logical_and'), ','.join(self._print(i) for i in expr.args))

    # 打印逻辑或表达式的字符串表示
    def _print_Or(self, expr):
        "Logical Or printer"
        # 由于LambdaPrinter使用Python中的'or'关键字，所以需要重写
        # 返回使用逻辑或函数（logical_or）处理的字符串表示
        return '{}.reduce(({}))'.format(self._module_format(self._module + '.logical_or'), ','.join(self._print(i) for i in expr.args))
    def _print_Not(self, expr):
        # 打印逻辑非表达式
        # LambdaPrinter 需要覆盖此方法，因为它使用 Python 的 'not' 关键字。
        # 即使 LambdaPrinter 没有定义此方法，我们仍然需要定义自己的方法，因为 StrPrinter 没有定义它。
        return '{}({})'.format(self._module_format(self._module + '.logical_not'), ','.join(self._print(i) for i in expr.args))

    def _print_Pow(self, expr, rational=False):
        # XXX 负整数指数错误的解决方法
        if expr.exp.is_integer and expr.exp.is_negative:
            expr = Pow(expr.base, expr.exp.evalf(), evaluate=False)
        return self._hprint_Pow(expr, rational=rational, sqrt=self._module + '.sqrt')

    def _print_Min(self, expr):
        # 打印最小值函数表达式
        return '{}({}.asarray([{}]), axis=0)'.format(self._module_format(self._module + '.amin'), self._module_format(self._module), ','.join(self._print(i) for i in expr.args))

    def _print_Max(self, expr):
        # 打印最大值函数表达式
        return '{}({}.asarray([{}]), axis=0)'.format(self._module_format(self._module + '.amax'), self._module_format(self._module), ','.join(self._print(i) for i in expr.args))

    def _print_arg(self, expr):
        # 打印参数的角度函数表达式
        return "%s(%s)" % (self._module_format(self._module + '.angle'), self._print(expr.args[0]))

    def _print_im(self, expr):
        # 打印虚部函数表达式
        return "%s(%s)" % (self._module_format(self._module + '.imag'), self._print(expr.args[0]))

    def _print_Mod(self, expr):
        # 打印求模函数表达式
        return "%s(%s)" % (self._module_format(self._module + '.mod'), ', '.join(
            (self._print(arg) for arg in expr.args)))

    def _print_re(self, expr):
        # 打印实部函数表达式
        return "%s(%s)" % (self._module_format(self._module + '.real'), self._print(expr.args[0]))

    def _print_sinc(self, expr):
        # 打印sinc函数表达式
        return "%s(%s)" % (self._module_format(self._module + '.sinc'), self._print(expr.args[0]/S.Pi))

    def _print_MatrixBase(self, expr):
        # 打印矩阵基类表达式
        func = self.known_functions.get(expr.__class__.__name__, None)
        if func is None:
            func = self._module_format(self._module + '.array')
        return "%s(%s)" % (func, self._print(expr.tolist()))

    def _print_Identity(self, expr):
        # 打印单位矩阵表达式
        shape = expr.shape
        if all(dim.is_Integer for dim in shape):
            return "%s(%s)" % (self._module_format(self._module + '.eye'), self._print(expr.shape[0]))
        else:
            raise NotImplementedError("Symbolic matrix dimensions are not yet supported for identity matrices")

    def _print_BlockMatrix(self, expr):
        # 打印块矩阵表达式
        return '{}({})'.format(self._module_format(self._module + '.block'),
                               self._print(expr.args[0].tolist()))

    def _print_NDimArray(self, expr):
        # 打印多维数组表达式
        if len(expr.shape) == 1:
            return self._module + '.array(' + self._print(expr.args[0]) + ')'
        if len(expr.shape) == 2:
            return self._print(expr.tomatrix())
        # 可以扩展到更多维度
        return super()._print_not_supported(self, expr)
    # 定义变量 _add，表示字符串 "add"
    _add = "add"
    # 定义变量 _einsum，表示字符串 "einsum"
    _einsum = "einsum"
    # 定义变量 _transpose，表示字符串 "transpose"
    _transpose = "transpose"
    # 定义变量 _ones，表示字符串 "ones"
    _ones = "ones"
    # 定义变量 _zeros，表示字符串 "zeros"
    _zeros = "zeros"
    
    # 将 _print_lowergamma 设定为 CodePrinter 类的 _print_not_supported 方法的引用
    _print_lowergamma = CodePrinter._print_not_supported
    # 将 _print_uppergamma 设定为 CodePrinter 类的 _print_not_supported 方法的引用
    _print_uppergamma = CodePrinter._print_not_supported
    # 将 _print_fresnelc 设定为 CodePrinter 类的 _print_not_supported 方法的引用
    _print_fresnelc = CodePrinter._print_not_supported
    # 将 _print_fresnels 设定为 CodePrinter 类的 _print_not_supported 方法的引用
    _print_fresnels = CodePrinter._print_not_supported
# 遍历 _numpy_known_functions 列表中的每个函数名，并为 NumPyPrinter 动态设置对应的打印函数
for func in _numpy_known_functions:
    setattr(NumPyPrinter, f'_print_{func}', _print_known_func)

# 遍历 _numpy_known_constants 列表中的每个常量名，并为 NumPyPrinter 动态设置对应的打印函数
for const in _numpy_known_constants:
    setattr(NumPyPrinter, f'_print_{const}', _print_known_const)

# SciPy 特殊函数名称到 NumPy 兼容函数名称的映射字典
_known_functions_scipy_special = {
    'Ei': 'expi',
    'erf': 'erf',
    'erfc': 'erfc',
    'besselj': 'jv',
    'bessely': 'yv',
    'besseli': 'iv',
    'besselk': 'kv',
    'cosm1': 'cosm1',
    'powm1': 'powm1',
    'factorial': 'factorial',
    'gamma': 'gamma',
    'loggamma': 'gammaln',
    'digamma': 'psi',
    'polygamma': 'polygamma',
    'RisingFactorial': 'poch',
    'jacobi': 'eval_jacobi',
    'gegenbauer': 'eval_gegenbauer',
    'chebyshevt': 'eval_chebyt',
    'chebyshevu': 'eval_chebyu',
    'legendre': 'eval_legendre',
    'hermite': 'eval_hermite',
    'laguerre': 'eval_laguerre',
    'assoc_laguerre': 'eval_genlaguerre',
    'beta': 'beta',
    'LambertW': 'lambertw',
}

# SciPy 常量名称到 SciPy.constants 兼容常量名称的映射字典
_known_constants_scipy_constants = {
    'GoldenRatio': 'golden_ratio',
    'Pi': 'pi',
}

# 将 _known_functions_scipy_special 中的映射关系转换为 'scipy.special.' + value 格式，并存入 _scipy_known_functions 字典
_scipy_known_functions = {k: "scipy.special." + v for k, v in _known_functions_scipy_special.items()}

# 将 _known_constants_scipy_constants 中的映射关系转换为 'scipy.constants.' + value 格式，并存入 _scipy_known_constants 字典
_scipy_known_constants = {k: "scipy.constants." + v for k, v in _known_constants_scipy_constants.items()}

# SciPyPrinter 类继承自 NumPyPrinter 类
class SciPyPrinter(NumPyPrinter):
    
    # _kf 属性包含了 NumPyPrinter._kf 和 _scipy_known_functions 的合并结果
    _kf = {**NumPyPrinter._kf, **_scipy_known_functions}
    
    # _kc 属性包含了 NumPyPrinter._kc 和 _scipy_known_constants 的合并结果
    _kc = {**NumPyPrinter._kc, **_scipy_known_constants}

    # 初始化方法，设置语言为 "Python with SciPy and NumPy"
    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.language = "Python with SciPy and NumPy"

    # 打印 SparseRepMatrix 对象的方法，将其转换为字符串表示
    def _print_SparseRepMatrix(self, expr):
        i, j, data = [], [], []
        for (r, c), v in expr.todok().items():
            i.append(r)
            j.append(c)
            data.append(v)

        return "{name}(({data}, ({i}, {j})), shape={shape})".format(
            name=self._module_format('scipy.sparse.coo_matrix'),
            data=data, i=i, j=j, shape=expr.shape
        )

    # ImmutableSparseMatrix 的打印方法与 SparseRepMatrix 相同
    _print_ImmutableSparseMatrix = _print_SparseRepMatrix

    # assoc_legendre 方法的打印方法，调用 scipy.special.lpmv 打印函数
    # 参数顺序与 assoc_legendre 方法不同
    def _print_assoc_legendre(self, expr):
        return "{0}({2}, {1}, {3})".format(
            self._module_format('scipy.special.lpmv'),
            self._print(expr.args[0]),
            self._print(expr.args[1]),
            self._print(expr.args[2]))

    # lowergamma 方法的打印方法，组合调用 scipy.special.gamma 和 scipy.special.gammainc 打印函数
    def _print_lowergamma(self, expr):
        return "{0}({2})*{1}({2}, {3})".format(
            self._module_format('scipy.special.gamma'),
            self._module_format('scipy.special.gammainc'),
            self._print(expr.args[0]),
            self._print(expr.args[1]))

    # uppergamma 方法的打印方法，组合调用 scipy.special.gamma 和 scipy.special.gammaincc 打印函数
    def _print_uppergamma(self, expr):
        return "{0}({2})*{1}({2}, {3})".format(
            self._module_format('scipy.special.gamma'),
            self._module_format('scipy.special.gammaincc'),
            self._print(expr.args[0]),
            self._print(expr.args[1]))
    # 格式化并返回 betainc 函数的字符串表示
    def _print_betainc(self, expr):
        betainc = self._module_format('scipy.special.betainc')
        # 格式化 beta 函数的字符串表示
        beta = self._module_format('scipy.special.beta')
        # 格式化表达式中的参数列表
        args = [self._print(arg) for arg in expr.args]
        # 返回格式化后的表达式字符串
        return f"({betainc}({args[0]}, {args[1]}, {args[3]}) - {betainc}({args[0]}, {args[1]}, {args[2]})) \
            * {beta}({args[0]}, {args[1]})"

    # 格式化并返回 betainc_regularized 函数的字符串表示
    def _print_betainc_regularized(self, expr):
        return "{0}({1}, {2}, {4}) - {0}({1}, {2}, {3})".format(
            self._module_format('scipy.special.betainc'),
            self._print(expr.args[0]),
            self._print(expr.args[1]),
            self._print(expr.args[2]),
            self._print(expr.args[3]))

    # 格式化并返回 fresnels 函数的字符串表示
    def _print_fresnels(self, expr):
        return "{}({})[0]".format(
                self._module_format("scipy.special.fresnel"),
                self._print(expr.args[0]))

    # 格式化并返回 fresnelc 函数的字符串表示
    def _print_fresnelc(self, expr):
        return "{}({})[1]".format(
                self._module_format("scipy.special.fresnel"),
                self._print(expr.args[0]))

    # 格式化并返回 airyai 函数的字符串表示
    def _print_airyai(self, expr):
        return "{}({})[0]".format(
                self._module_format("scipy.special.airy"),
                self._print(expr.args[0]))

    # 格式化并返回 airyaiprime 函数的字符串表示
    def _print_airyaiprime(self, expr):
        return "{}({})[1]".format(
                self._module_format("scipy.special.airy"),
                self._print(expr.args[0]))

    # 格式化并返回 airybi 函数的字符串表示
    def _print_airybi(self, expr):
        return "{}({})[2]".format(
                self._module_format("scipy.special.airy"),
                self._print(expr.args[0]))

    # 格式化并返回 airybiprime 函数的字符串表示
    def _print_airybiprime(self, expr):
        return "{}({})[3]".format(
                self._module_format("scipy.special.airy"),
                self._print(expr.args[0]))

    # 格式化并返回 bernoulli 函数的字符串表示
    def _print_bernoulli(self, expr):
        # 用 zeta 函数重写 bernoulli 函数，因为 scipy 的实现与 SymPy 的不一致
        return self._print(expr._eval_rewrite_as_zeta(*expr.args))

    # 格式化并返回 harmonic 函数的字符串表示
    def _print_harmonic(self, expr):
        # 用 zeta 函数重写 harmonic 函数
        return self._print(expr._eval_rewrite_as_zeta(*expr.args))

    # 格式化并返回 Integral 类的字符串表示
    def _print_Integral(self, e):
        # 解析积分的变量和限制
        integration_vars, limits = _unpack_integral_limits(e)

        if len(limits) == 1:
            # 对于一维情况，优先选择 quad 而不是 nquad（更美观，但不是必须的）
            module_str = self._module_format("scipy.integrate.quad")
            limit_str = "%s, %s" % tuple(map(self._print, limits[0]))
        else:
            module_str = self._module_format("scipy.integrate.nquad")
            limit_str = "({})".format(", ".join(
                "(%s, %s)" % tuple(map(self._print, l)) for l in limits))

        # 返回格式化后的积分表达式字符串
        return "{}(lambda {}: {}, {})[0]".format(
                module_str,
                ", ".join(map(self._print, integration_vars)),
                self._print(e.args[0]),
                limit_str)

    # 格式化并返回 Si 函数的字符串表示
    def _print_Si(self, expr):
        return "{}({})[0]".format(
                self._module_format("scipy.special.sici"),
                self._print(expr.args[0]))
    # 定义一个方法 _print_Ci，接受一个表达式参数 expr
    def _print_Ci(self, expr):
        # 返回格式化的字符串，调用 self._module_format 方法，传入 "scipy.special.sici"，
        # 并将表达式 expr.args[0] 的打印结果作为第二个参数插入到字符串中
        return "{}({})[1]".format(
                self._module_format("scipy.special.sici"),
                self._print(expr.args[0]))
# 对于_scipy_known_functions中的每个函数，将其对应的打印方法设置为_print_known_func
for func in _scipy_known_functions:
    setattr(SciPyPrinter, f'_print_{func}', _print_known_func)

# 对于_scipy_known_constants中的每个常量，将其对应的打印方法设置为_print_known_const
for const in _scipy_known_constants:
    setattr(SciPyPrinter, f'_print_{const}', _print_known_const)


# 创建一个字典_cupy_known_functions，将_known_functions_numpy中的每个项前缀为"cupy."并存储
_cupy_known_functions = {k: "cupy." + v for k, v in _known_functions_numpy.items()}
# 创建一个字典_cupy_known_constants，将_known_constants_numpy中的每个项前缀为"cupy."并存储
_cupy_known_constants = {k: "cupy." + v for k, v in _known_constants_numpy.items()}

# 定义CuPyPrinter类，继承自NumPyPrinter类，用于打印CuPy相关的向量化分段函数和逻辑操作符等
class CuPyPrinter(NumPyPrinter):
    """
    CuPy printer which handles vectorized piecewise functions,
    logical operators, etc.
    """

    # 指定模块名称为'cupy'
    _module = 'cupy'
    # 将_cupy_known_functions设为类属性，存储CuPy已知函数的映射关系
    _kf = _cupy_known_functions
    # 将_cupy_known_constants设为类属性，存储CuPy已知常量的映射关系
    _kc = _cupy_known_constants

    # 初始化方法，接受settings参数，并调用父类的初始化方法
    def __init__(self, settings=None):
        super().__init__(settings=settings)

# 对于_cupy_known_functions中的每个函数，将其对应的打印方法设置为_print_known_func
for func in _cupy_known_functions:
    setattr(CuPyPrinter, f'_print_{func}', _print_known_func)

# 对于_cupy_known_constants中的每个常量，将其对应的打印方法设置为_print_known_const
for const in _cupy_known_constants:
    setattr(CuPyPrinter, f'_print_{const}', _print_known_const)


# 创建一个字典_jax_known_functions，将_known_functions_numpy中的每个项前缀为'jax.numpy.'并存储
_jax_known_functions = {k: 'jax.numpy.' + v for k, v in _known_functions_numpy.items()}
# 创建一个字典_jax_known_constants，将_known_constants_numpy中的每个项前缀为'jax.numpy.'并存储
_jax_known_constants = {k: 'jax.numpy.' + v for k, v in _known_constants_numpy.items()}

# 定义JaxPrinter类，继承自NumPyPrinter类，用于打印JAX相关的向量化分段函数和逻辑操作符等
class JaxPrinter(NumPyPrinter):
    """
    JAX printer which handles vectorized piecewise functions,
    logical operators, etc.
    """
    # 指定模块名称为'jax.numpy'
    _module = "jax.numpy"

    # 将_jax_known_functions设为类属性，存储JAX已知函数的映射关系
    _kf = _jax_known_functions
    # 将_jax_known_constants设为类属性，存储JAX已知常量的映射关系
    _kc = _jax_known_constants

    # 初始化方法，接受settings参数，并调用父类的初始化方法，同时设置printmethod属性为'_jaxcode'
    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.printmethod = '_jaxcode'

    # 重写_print_And方法，用于打印逻辑And操作的表达式
    def _print_And(self, expr):
        "Logical And printer"
        return "{}({}.asarray([{}]), axis=0)".format(
            self._module_format(self._module + ".all"),
            self._module_format(self._module),
            ",".join(self._print(i) for i in expr.args),
        )

    # 重写_print_Or方法，用于打印逻辑Or操作的表达式
    def _print_Or(self, expr):
        "Logical Or printer"
        return "{}({}.asarray([{}]), axis=0)".format(
            self._module_format(self._module + ".any"),
            self._module_format(self._module),
            ",".join(self._print(i) for i in expr.args),
        )

# 对于_jax_known_functions中的每个函数，将其对应的打印方法设置为_print_known_func
for func in _jax_known_functions:
    setattr(JaxPrinter, f'_print_{func}', _print_known_func)

# 对于_jax_known_constants中的每个常量，将其对应的打印方法设置为_print_known_const
for const in _jax_known_constants:
    setattr(JaxPrinter, f'_print_{const}', _print_known_const)
```
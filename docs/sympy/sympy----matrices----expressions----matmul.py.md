# `D:\src\scipysrc\sympy\sympy\matrices\expressions\matmul.py`

```
# 从 sympy.assumptions.ask 模块中导入 ask 和 Q 函数
# 用于进行假设查询和条件检查
from sympy.assumptions.ask import ask, Q
# 从 sympy.assumptions.refine 模块中导入 handlers_dict
# 用于处理条件精化的字典
from sympy.assumptions.refine import handlers_dict
# 从 sympy.core 模块中导入 Basic, sympify, S
# 分别是基本类、符号化函数和符号 S
from sympy.core import Basic, sympify, S
# 从 sympy.core.mul 模块中导入 mul, Mul
# 用于处理乘法表达式的类和函数
from sympy.core.mul import mul, Mul
# 从 sympy.core.numbers 模块中导入 Number, Integer
# 用于处理数值和整数的类
from sympy.core.numbers import Number, Integer
# 从 sympy.core.symbol 模块中导入 Dummy
# 用于创建符号变量的类
from sympy.core.symbol import Dummy
# 从 sympy.functions 模块中导入 adjoint
# 用于计算伴随矩阵的函数
from sympy.functions import adjoint
# 从 sympy.strategies 模块中导入多个策略函数
# 用于策略模式的实现
from sympy.strategies import (rm_id, unpack, typed, flatten, exhaust,
        do_one, new)
# 从 sympy.matrices.exceptions 模块中导入 NonInvertibleMatrixError
# 用于非可逆矩阵错误的异常类
from sympy.matrices.exceptions import NonInvertibleMatrixError
# 从 sympy.matrices.matrixbase 模块中导入 MatrixBase
# 用于定义矩阵基类
from sympy.matrices.matrixbase import MatrixBase
# 从 sympy.utilities.exceptions 模块中导入 sympy_deprecation_warning
# 用于发出 SymPy 废弃警告的函数
from sympy.utilities.exceptions import sympy_deprecation_warning
# 从 sympy.matrices.expressions._shape 模块中导入 validate_matmul_integer
# 用于验证矩阵乘法的整数参数的函数
from sympy.matrices.expressions._shape import validate_matmul_integer as validate

# 从 .inverse 模块中导入 Inverse
# 从 .matexpr 模块中导入 MatrixExpr
# 从 .matpow 模块中导入 MatPow
# 从 .transpose 模块中导入 transpose
# 从 .permutation 模块中导入 PermutationMatrix
# 从 .special 模块中导入 ZeroMatrix, Identity, GenericIdentity, OneMatrix
from .inverse import Inverse
from .matexpr import MatrixExpr
from .matpow import MatPow
from .transpose import transpose
from .permutation import PermutationMatrix
from .special import ZeroMatrix, Identity, GenericIdentity, OneMatrix

# XXX: MatMul 应该或许不应该直接从 Mul 类继承
# 声明 MatMul 类，继承自 MatrixExpr 和 Mul 类
class MatMul(MatrixExpr, Mul):
    """
    矩阵表达式的乘积

    Examples
    ========

    >>> from sympy import MatMul, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 4)
    >>> B = MatrixSymbol('B', 4, 3)
    >>> C = MatrixSymbol('C', 3, 6)
    >>> MatMul(A, B, C)
    A*B*C
    """
    # 表示该类是 MatMul 类型的标志
    is_MatMul = True

    # 定义一个通用的单位矩阵
    identity = GenericIdentity()

    # 构造函数，接受多个参数，支持是否进行求值评估的选项
    # 这个函数可以创建 MatMul 类的实例
    def __new__(cls, *args, evaluate=False, check=None, _sympify=True):
        # 如果没有参数，则返回通用单位矩阵
        if not args:
            return cls.identity

        # 在构造函数中，应当积极地删除这个因子，以避免从 GenericIdentity().shape 中引发 TypeErrors
        # 使用 filter 函数去除参数中的单位矩阵
        args = list(filter(lambda i: cls.identity != i, args))
        # 如果需要符号化参数，则对参数列表进行符号化处理
        if _sympify:
            args = list(map(sympify, args))
        # 创建 Basic 类的新实例
        obj = Basic.__new__(cls, *args)
        # 将对象分解为系数和矩阵列表
        factor, matrices = obj.as_coeff_matrices()

        # 如果指定了 check 参数，则发出 SymPy 废弃警告
        if check is not None:
            sympy_deprecation_warning(
                "Passing check to MatMul is deprecated and the check argument will be removed in a future version.",
                deprecated_since_version="1.11",
                active_deprecations_target='remove-check-argument-from-matrix-operations')

        # 如果 check 不为 False，则对矩阵进行验证
        if check is not False:
            validate(*matrices)

        # 如果矩阵列表为空，则返回因子
        if not matrices:
            return factor

        # 如果需要进行求值评估，则调用 _evaluate 方法
        if evaluate:
            return cls._evaluate(obj)

        # 返回对象实例
        return obj

    # 类方法，用于对表达式进行评估
    @classmethod
    def _evaluate(cls, expr):
        return canonicalize(expr)

    # 返回矩阵表达式的形状
    @property
    def shape(self):
        # 从参数中提取出所有矩阵，并返回第一个矩阵的行数和最后一个矩阵的列数的元组
        matrices = [arg for arg in self.args if arg.is_Matrix]
        return (matrices[0].rows, matrices[-1].cols)
    def _entry(self, i, j, expand=True, **kwargs):
        # 避免循环导入
        from sympy.concrete.summations import Sum
        from sympy.matrices.immutable import ImmutableMatrix

        # 将自身表示为系数和矩阵的列表
        coeff, matrices = self.as_coeff_matrices()

        if len(matrices) == 1:  # 当只有一个矩阵时，如 2*X，直接返回矩阵的乘积
            return coeff * matrices[0][i, j]

        # 初始化索引列表
        indices = [None]*(len(matrices) + 1)
        ind_ranges = [None]*(len(matrices) - 1)
        indices[0] = i
        indices[-1] = j

        # 定义生成虚拟变量的函数
        def f():
            counter = 1
            while True:
                yield Dummy("i_%i" % counter)
                counter += 1

        # 从关键字参数中获取虚拟变量生成器
        dummy_generator = kwargs.get("dummy_generator", f())

        # 为每个矩阵生成虚拟变量
        for k in range(1, len(matrices)):
            indices[k] = next(dummy_generator)

        # 计算每个矩阵之间的乘积，并根据需要扩展虚拟变量
        matrices = [arg._entry(indices[k], indices[k+1], dummy_generator=dummy_generator) for k, arg in enumerate(matrices)]
        expr_in_sum = Mul.fromiter(matrices)

        # 如果其中有不可变矩阵，需要扩展表达式
        if any(v.has(ImmutableMatrix) for v in matrices):
            expand = True

        # 构造求和表达式，考虑索引范围
        result = coeff * Sum(
                expr_in_sum,
                *zip(indices[1:-1], [0]*len(ind_ranges), ind_ranges)
            )

        # 如果索引范围中有符号表达式，不进行结果计算
        if not any(isinstance(v, (Integer, int)) for v in ind_ranges):
            expand = False

        # 根据需要进行结果计算或直接返回结果
        return result.doit() if expand else result

    def as_coeff_matrices(self):
        # 提取并返回标量和矩阵
        scalars = [x for x in self.args if not x.is_Matrix]
        matrices = [x for x in self.args if x.is_Matrix]
        coeff = Mul(*scalars)
        if coeff.is_commutative is False:
            raise NotImplementedError("noncommutative scalars in MatMul are not supported.")

        return coeff, matrices

    def as_coeff_mmul(self):
        # 返回系数和矩阵乘积的形式
        coeff, matrices = self.as_coeff_matrices()
        return coeff, MatMul(*matrices)

    def expand(self, **kwargs):
        # 调用父类方法扩展，并执行自定义评估
        expanded = super(MatMul, self).expand(**kwargs)
        return self._evaluate(expanded)

    def _eval_transpose(self):
        """矩阵乘积的转置运算。

        Notes
        =====

        应用以下规则进行转置运算：

        两个矩阵相乘的转置：
        `\\left(A B\\right)^{T} = B^{T} A^{T}`

        矩阵乘以标量的转置：
        `\\left(c A\\right)^{T} = c A^{T}`

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Transpose
        """
        # 提取并返回系数和矩阵列表的形式
        coeff, matrices = self.as_coeff_matrices()
        return MatMul(
            coeff, *[transpose(arg) for arg in matrices[::-1]]).doit()

    def _eval_adjoint(self):
        # 对矩阵乘积进行共轭转置运算
        return MatMul(*[adjoint(arg) for arg in self.args[::-1]]).doit()
    # 对象方法，用于计算矩阵表达式的迹
    def _eval_trace(self):
        # 提取表达式的系数和乘积项
        factor, mmul = self.as_coeff_mmul()
        # 如果系数不为1，则调用迹函数对乘积项求迹并返回结果
        if factor != 1:
            from .trace import trace
            return factor * trace(mmul.doit())

    # 对象方法，用于计算矩阵表达式的行列式
    def _eval_determinant(self):
        # 导入行列式计算模块
        from sympy.matrices.expressions.determinant import Determinant
        # 提取表达式的系数和矩阵列表
        factor, matrices = self.as_coeff_matrices()
        # 筛选出仅为方阵的矩阵列表
        square_matrices = only_squares(*matrices)
        # 返回表达式的系数的行数次方乘以所有方阵的行列式的乘积
        return factor**self.rows * Mul(*list(map(Determinant, square_matrices)))

    # 对象方法，用于计算矩阵表达式的逆
    def _eval_inverse(self):
        # 如果所有参数均为方阵，则返回参数的逆的乘积，否则返回自身的逆
        if all(arg.is_square for arg in self.args if isinstance(arg, MatrixExpr)):
            return MatMul(*(
                arg.inverse() if isinstance(arg, MatrixExpr) else arg**-1
                    for arg in self.args[::-1]
                )
            ).doit()
        return Inverse(self)

    # 对象方法，对矩阵表达式执行“doit”操作
    def doit(self, **hints):
        # 深度标记，如果设置为True则递归执行“doit”，否则返回原参数
        deep = hints.get('deep', True)
        if deep:
            # 对所有参数递归执行“doit”
            args = tuple(arg.doit(**hints) for arg in self.args)
        else:
            args = self.args

        # 规范化参数并返回规范化后的表达式
        expr = canonicalize(MatMul(*args))
        return expr

    # 对象方法，用于处理参数的交换特性（commutativity）
    def args_cnc(self, cset=False, warn=True, **kwargs):
        # 分离参数中的可交换项和不可交换项
        coeff_c = [x for x in self.args if x.is_commutative]
        coeff_nc = [x for x in self.args if not x.is_commutative]
        if cset:
            clen = len(coeff_c)
            coeff_c = set(coeff_c)
            # 如果设置了cset，检查是否有重复的可交换参数，并抛出错误
            if clen and warn and len(coeff_c) != clen:
                raise ValueError('repeated commutative arguments: %s' %
                                 [ci for ci in coeff_c if list(self.args).count(ci) > 1])
        # 返回分离后的可交换项和不可交换项列表
        return [coeff_c, coeff_nc]

    # 对象方法，用于计算矩阵表达式对某个变量的导数
    def _eval_derivative_matrix_lines(self, x):
        # 导入转置操作
        from .transpose import Transpose
        # 找出参数中包含变量x的索引
        with_x_ind = [i for i, arg in enumerate(self.args) if arg.has(x)]
        lines = []
        for ind in with_x_ind:
            left_args = self.args[:ind]
            right_args = self.args[ind+1:]

            # 构建右边矩阵乘积或单位矩阵
            if right_args:
                right_mat = MatMul.fromiter(right_args)
            else:
                right_mat = Identity(self.shape[1])
                
            # 构建左边转置矩阵乘积或单位矩阵
            if left_args:
                left_rev = MatMul.fromiter([Transpose(i).doit() if i.is_Matrix else i for i in reversed(left_args)])
            else:
                left_rev = Identity(self.shape[0])

            # 计算参数对变量x的导数行列式列表
            d = self.args[ind]._eval_derivative_matrix_lines(x)
            for i in d:
                i.append_first(left_rev)
                i.append_second(right_mat)
                lines.append(i)

        return lines
# 将 Mul 和 MatMul 注册为 MatMul 的处理程序类
mul.register_handlerclass((Mul, MatMul), MatMul)

# 定义函数 newmul，用于创建新的 MatMul 对象，去除首个参数为 1 的情况
def newmul(*args):
    if args[0] == 1:
        args = args[1:]
    return new(MatMul, *args)

# 定义函数 any_zeros，用于检查 mul 中是否存在零矩阵，如果存在则返回对应形状的零矩阵
def any_zeros(mul):
    if any(arg.is_zero or (arg.is_Matrix and arg.is_ZeroMatrix)
           for arg in mul.args):
        matrices = [arg for arg in mul.args if arg.is_Matrix]
        return ZeroMatrix(matrices[0].rows, matrices[-1].cols)
    return mul

# 定义函数 merge_explicit，用于合并显式的 MatrixBase 参数
def merge_explicit(matmul):
    """ Merge explicit MatrixBase arguments
    
    >>> from sympy import MatrixSymbol, Matrix, MatMul, pprint
    >>> from sympy.matrices.expressions.matmul import merge_explicit
    >>> A = MatrixSymbol('A', 2, 2)
    >>> B = Matrix([[1, 1], [1, 1]])
    >>> C = Matrix([[1, 2], [3, 4]])
    >>> X = MatMul(A, B, C)
    >>> pprint(X)
      [1  1] [1  2]
    A*[    ]*[    ]
      [1  1] [3  4]
    >>> pprint(merge_explicit(X))
      [4  6]
    A*[    ]
      [4  6]

    >>> X = MatMul(B, A, C)
    >>> pprint(X)
    [1  1]   [1  2]
    [    ]*A*[    ]
    [1  1]   [3  4]
    >>> pprint(merge_explicit(X))
    [1  1]   [1  2]
    [    ]*A*[    ]
    [1  1]   [3  4]
    """
    if not any(isinstance(arg, MatrixBase) for arg in matmul.args):
        return matmul
    newargs = []
    last = matmul.args[0]
    for arg in matmul.args[1:]:
        if isinstance(arg, (MatrixBase, Number)) and isinstance(last, (MatrixBase, Number)):
            last = last * arg
        else:
            newargs.append(last)
            last = arg
    newargs.append(last)

    return MatMul(*newargs)

# 定义函数 remove_ids，从 MatMul 中移除单位矩阵
def remove_ids(mul):
    """ Remove Identities from a MatMul

    This is a modified version of sympy.strategies.rm_id.
    This is necesssary because MatMul may contain both MatrixExprs and Exprs
    as args.

    See Also
    ========

    sympy.strategies.rm_id
    """
    # Separate Exprs from MatrixExprs in args
    factor, mmul = mul.as_coeff_mmul()
    # Apply standard rm_id for MatMuls
    result = rm_id(lambda x: x.is_Identity is True)(mmul)
    if result != mmul:
        return newmul(factor, *result.args)  # Recombine and return
    else:
        return mul

# 定义函数 factor_in_front，将乘法中的系数移到矩阵前面
def factor_in_front(mul):
    factor, matrices = mul.as_coeff_matrices()
    if factor != 1:
        return newmul(factor, *matrices)
    return mul

# 定义函数 combine_powers，合并连续的相同基数的幂，同时取消可能的矩阵逆
def combine_powers(mul):
    r"""Combine consecutive powers with the same base into one, e.g.
    $$A \times A^2 \Rightarrow A^3$$

    This also cancels out the possible matrix inverses using the
    knowledgebase of :class:`~.Inverse`, e.g.,
    $$ Y \times X \times X^{-1} \Rightarrow Y $$
    """
    factor, args = mul.as_coeff_matrices()
    new_args = [args[0]]
    # 对参数列表 args 中的每一个元素进行遍历，从第二个元素开始到最后一个元素
    for i in range(1, len(args)):
        # 将上一个处理过的参数记为 A
        A = new_args[-1]
        # 将当前处理的参数记为 B
        B = args[i]
    
        # 如果 B 是 Inverse 类型且其参数是 MatMul 类型
        if isinstance(B, Inverse) and isinstance(B.arg, MatMul):
            # 获取 B 的参数列表 Bargs
            Bargs = B.arg.args
            # 记录 Bargs 的长度
            l = len(Bargs)
            # 如果 B 的参数列表与 new_args 的末尾 l 个元素相同
            if list(Bargs) == new_args[-l:]:
                # 将 new_args 中最后 l 个元素替换为单位矩阵 Identity(B.shape[0])
                new_args = new_args[:-l] + [Identity(B.shape[0])]
                # 继续下一次循环处理下一个参数
                continue
    
        # 如果 A 是 Inverse 类型且其参数是 MatMul 类型
        if isinstance(A, Inverse) and isinstance(A.arg, MatMul):
            # 获取 A 的参数列表 Aargs
            Aargs = A.arg.args
            # 记录 Aargs 的长度
            l = len(Aargs)
            # 如果 A 的参数列表与 args 中当前位置开始的长度为 l 的部分相同
            if list(Aargs) == args[i:i+l]:
                # 创建一个 A.shape[0] 大小的单位矩阵
                identity = Identity(A.shape[0])
                # 将 new_args 中的最后一个元素替换为单位矩阵 identity
                new_args[-1] = identity
                # 将 args 中当前位置开始的长度为 l 的部分都替换为单位矩阵 identity
                for j in range(i, i+l):
                    args[j] = identity
                # 继续下一次循环处理下一个参数
                continue
    
        # 如果 A 或 B 不是方阵
        if A.is_square == False or B.is_square == False:
            # 将 B 添加到 new_args 中
            new_args.append(B)
            # 继续下一次循环处理下一个参数
            continue
    
        # 如果 A 是 MatPow 类型
        if isinstance(A, MatPow):
            # 获取 A 的基础部分和指数部分
            A_base, A_exp = A.args
        else:
            A_base, A_exp = A, S.One
    
        # 如果 B 是 MatPow 类型
        if isinstance(B, MatPow):
            # 获取 B 的基础部分和指数部分
            B_base, B_exp = B.args
        else:
            B_base, B_exp = B, S.One
    
        # 如果 A 的基础部分与 B 的基础部分相同
        if A_base == B_base:
            # 计算新的指数部分为 A_exp + B_exp
            new_exp = A_exp + B_exp
            # 将 new_args 中的最后一个元素替换为 MatPow(A_base, new_exp) 的结果
            new_args[-1] = MatPow(A_base, new_exp).doit(deep=False)
            # 继续下一次循环处理下一个参数
            continue
        # 如果 B 的基础部分不是 MatrixBase 类型
        elif not isinstance(B_base, MatrixBase):
            # 尝试计算 B_base 的逆矩阵 B_base_inv
            try:
                B_base_inv = B_base.inverse()
            except NonInvertibleMatrixError:
                B_base_inv = None
            # 如果 B_base_inv 不为 None 并且 A 的基础部分等于 B_base_inv
            if B_base_inv is not None and A_base == B_base_inv:
                # 计算新的指数部分为 A_exp - B_exp
                new_exp = A_exp - B_exp
                # 将 new_args 中的最后一个元素替换为 MatPow(A_base, new_exp) 的结果
                new_args[-1] = MatPow(A_base, new_exp).doit(deep=False)
                # 继续下一次循环处理下一个参数
                continue
    
        # 将 B 添加到 new_args 中
        new_args.append(B)
    
    # 返回使用 newmul 函数处理后的结果，参数为 factor 和 new_args 中的所有元素
    return newmul(factor, *new_args)
# 将多个置换矩阵的乘积精细化为循环的乘积。
def combine_permutations(mul):
    # 获取乘积中的参数列表
    args = mul.args
    # 获取参数列表的长度
    l = len(args)
    # 如果参数列表长度小于2，直接返回原始乘积
    if l < 2:
        return mul

    # 初始化结果列表，将第一个参数添加进去
    result = [args[0]]
    # 遍历剩余的参数
    for i in range(1, l):
        # 获取当前和前一个参数
        A = result[-1]
        B = args[i]
        # 如果当前和前一个参数都是置换矩阵
        if isinstance(A, PermutationMatrix) and \
            isinstance(B, PermutationMatrix):
            # 提取置换矩阵的循环部分
            cycle_1 = A.args[0]
            cycle_2 = B.args[0]
            # 更新结果列表中的最后一个元素为新的置换矩阵乘积
            result[-1] = PermutationMatrix(cycle_1 * cycle_2)
        else:
            # 如果不是置换矩阵，直接将参数 B 添加到结果列表中
            result.append(B)

    # 返回新的乘积结果
    return MatMul(*result)

# 合并 OneMatrix 的乘积
def combine_one_matrices(mul):
    """
    合并 OneMatrix 的乘积

    例如：OneMatrix(2, 3) * OneMatrix(3, 4) -> 3 * OneMatrix(2, 4)
    """
    # 提取乘积的系数和参数列表
    factor, args = mul.as_coeff_matrices()
    # 初始化新的参数列表，将第一个参数添加进去
    new_args = [args[0]]

    # 遍历剩余的参数
    for B in args[1:]:
        # 获取当前和前一个参数
        A = new_args[-1]
        # 如果当前和前一个参数都是 OneMatrix
        if not isinstance(A, OneMatrix) or not isinstance(B, OneMatrix):
            # 如果不是 OneMatrix，直接将参数 B 添加到新的参数列表中
            new_args.append(B)
            continue
        # 移除新参数列表中的最后一个元素，替换为新的 OneMatrix
        new_args.pop()
        new_args.append(OneMatrix(A.shape[0], B.shape[1]))
        # 更新系数乘以 A 的列数
        factor *= A.shape[1]

    # 返回新的乘积结果
    return newmul(factor, *new_args)

# 分配有理项到 MatMul 表达式中简化
def distribute_monom(mul):
    """
    简化 MatMul 表达式，将有理数分配到 MatMul 中。

    例如：2*(A+B) -> 2*A + 2*B
    """
    # 提取乘积的参数列表
    args = mul.args
    # 如果参数列表长度为2
    if len(args) == 2:
        # 导入 MatAdd 类
        from .matadd import MatAdd
        # 如果第一个参数是 MatAdd 且第二个参数是有理数
        if args[0].is_MatAdd and args[1].is_Rational:
            # 对 MatAdd 中的每个矩阵乘以有理数并进行运算，返回 MatAdd 对象
            return MatAdd(*[MatMul(mat, args[1]).doit() for mat in args[0].args])
        # 如果第二个参数是 MatAdd 且第一个参数是有理数
        if args[1].is_MatAdd and args[0].is_Rational:
            # 对 MatAdd 中的每个矩阵乘以有理数并进行运算，返回 MatAdd 对象
            return MatAdd(*[MatMul(args[0], mat).doit() for mat in args[1].args])
    # 如果不符合以上条件，直接返回原始乘积
    return mul

# 规则列表，用于规范化 MatMul 对象
rules = (
    distribute_monom, any_zeros, remove_ids, combine_one_matrices, combine_powers, unpack, rm_id(lambda x: x == 1),
    merge_explicit, factor_in_front, flatten, combine_permutations)

# 构建规范化函数，对 MatMul 对象应用一系列规则来实现规范化
canonicalize = exhaust(typed({MatMul: do_one(*rules)}))

# 仅对方阵进行因子分解
def only_squares(*matrices):
    """只有当矩阵是方阵时才进行因子分解"""
    # 如果第一个矩阵的行数不等于最后一个矩阵的列数，抛出运行时错误
    if matrices[0].rows != matrices[-1].cols:
        raise RuntimeError("Invalid matrices being multiplied")
    # 初始化输出列表和起始索引
    out = []
    start = 0
    # 遍历矩阵列表
    for i, M in enumerate(matrices):
        # 如果当前矩阵的列数等于起始矩阵的行数
        if M.cols == matrices[start].rows:
            # 将起始到当前索引的矩阵乘积进行运算并添加到输出列表中
            out.append(MatMul(*matrices[start:i+1]).doit())
            # 更新起始索引为当前索引加一
            start = i+1
    # 返回输出列表
    return out

# 优化 MatMul 表达式
def refine_MatMul(expr, assumptions):
    """
    优化 MatMul 表达式

    >>> from sympy import MatrixSymbol, Q, assuming, refine
    >>> X = MatrixSymbol('X', 2, 2)
    >>> expr = X * X.T
    >>> print(expr)
    X*X.T
    >>> with assuming(Q.orthogonal(X)):
    ...     print(refine(expr))
    I
    """
    # 初始化新参数列表和表达式参数列表
    newargs = []
    exprargs = []

    # 遍历表达式的参数
    for args in expr.args:
        # 如果参数是矩阵，添加到表达式参数列表中
        if args.is_Matrix:
            exprargs.append(args)
        else:
            # 否则添加到新参数列表中
            newargs.append(args)

    # 将最后一个矩阵参数添加到表达式参数列表中
    last = exprargs[0]
    for arg in exprargs[1:]:
        # 遍历表达式参数列表中除第一个元素外的所有元素
        if arg == last.T and ask(Q.orthogonal(arg), assumptions):
            # 如果当前参数是上一个参数的转置，并且符合正交矩阵的条件
            last = Identity(arg.shape[0])
            # 将上一个参数替换为一个同维度的单位矩阵
        elif arg == last.conjugate() and ask(Q.unitary(arg), assumptions):
            # 如果当前参数是上一个参数的共轭，并且符合酉矩阵的条件
            last = Identity(arg.shape[0])
            # 将上一个参数替换为一个同维度的单位矩阵
        else:
            # 如果以上条件都不满足
            newargs.append(last)
            # 将上一个参数加入到新参数列表中
            last = arg
            # 将当前参数设为上一个参数
    newargs.append(last)
    # 将最后一个参数加入到新参数列表中

    return MatMul(*newargs)
    # 返回将所有参数进行矩阵乘法的结果
# 将函数 'refine_MatMul' 添加到名为 'handlers_dict' 的字典中，关键字为 'MatMul'
handlers_dict['MatMul'] = refine_MatMul
```
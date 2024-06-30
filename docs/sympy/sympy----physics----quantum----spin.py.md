# `D:\src\scipysrc\sympy\sympy\physics\quantum\spin.py`

```
"""Quantum mechanical angular momemtum."""

# 导入所需的符号计算模块
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.numbers import int_valued
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.simplify.simplify import simplify
from sympy.matrices import zeros
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import pretty_symbol

# 导入量子力学相关模块
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.operator import (HermitianOperator, Operator,
                                            UnitaryOperator)
from sympy.physics.quantum.state import Bra, Ket, State
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.hilbert import ComplexSpace, DirectSumHilbertSpace
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.qapply import qapply

# 公开的类和函数列表
__all__ = [
    'm_values',
    'Jplus',
    'Jminus',
    'Jx',
    'Jy',
    'Jz',
    'J2',
    'Rotation',
    'WignerD',
    'JxKet',
    'JxBra',
    'JyKet',
    'JyBra',
    'JzKet',
    'JzBra',
    'JzOp',
    'J2Op',
    'JxKetCoupled',
    'JxBraCoupled',
    'JyKetCoupled',
    'JyBraCoupled',
    'JzKetCoupled',
    'JzBraCoupled',
    'couple',
    'uncouple'
]

# 计算给定角动量 j 对应的 m 值
def m_values(j):
    j = sympify(j)  # 将 j 转换为 SymPy 的表达式
    size = 2*j + 1  # 计算可能的 m 值数量
    if not size.is_Integer or not size > 0:
        raise ValueError(
            'Only integer or half-integer values allowed for j, got: %r' % j
        )
    return size, [j - i for i in range(int(2*j + 1))]

#-----------------------------------------------------------------------------
# Spin Operators
#-----------------------------------------------------------------------------

# 自定义角动量算符基类
class SpinOpBase:
    """Base class for spin operators."""

    @classmethod
    def _eval_hilbert_space(cls, label):
        # 考虑所有可能的 j 值，因此我们的空间是无限的
        return ComplexSpace(S.Infinity)

    @property
    def name(self):
        return self.args[0]

    def _print_contents(self, printer, *args):
        return '%s%s' % (self.name, self._coord)

    def _print_contents_pretty(self, printer, *args):
        a = stringPict(str(self.name))
        b = stringPict(self._coord)
        return self._print_subscript_pretty(a, b)

    def _print_contents_latex(self, printer, *args):
        return r'%s_%s' % ((self.name, self._coord))
    # 生成基于给定基础的表示，使用选项中的 'j' 参数，默认为 S.Half
    def _represent_base(self, basis, **options):
        # 获取 'j' 参数的值，默认为 S.Half
        j = options.get('j', S.Half)
        # 根据 'j' 参数计算矩阵大小和其对应的 m 值列表
        size, mvals = m_values(j)
        # 创建一个大小为 size x size 的零矩阵
        result = zeros(size, size)
        # 遍历矩阵的每一个元素
        for p in range(size):
            for q in range(size):
                # 计算矩阵元素值，使用给定的 j 和 m 值
                me = self.matrix_element(j, mvals[p], j, mvals[q])
                result[p, q] = me
        # 返回生成的表示矩阵
        return result

    # 应用操作符到给定的 ket，在原始基础 orig_basis 下
    def _apply_op(self, ket, orig_basis, **options):
        # 将 ket 重写为当前对象的基础 self.basis 下的表示 state
        state = ket.rewrite(self.basis)
        # 如果 state 只有一个项
        if isinstance(state, State):
            # 计算并返回结果
            ret = (hbar*state.m)*state
        # 如果 state 是多个状态的线性组合
        elif isinstance(state, Sum):
            # 调用 _apply_operator_Sum 处理 Sum 类型的 state
            ret = self._apply_operator_Sum(state, **options)
        else:
            # 对 self*state 应用 qapply 函数并返回结果
            ret = qapply(self*state)
        # 如果 ret 等于 self*state，则抛出未实现异常
        if ret == self*state:
            raise NotImplementedError
        # 将结果 ret 重写为原始基础 orig_basis 下的表示并返回
        return ret.rewrite(orig_basis)

    # 应用 Jx 操作符到给定的 ket
    def _apply_operator_JxKet(self, ket, **options):
        return self._apply_op(ket, 'Jx', **options)

    # 应用 Jx 操作符到耦合态的 ket
    def _apply_operator_JxKetCoupled(self, ket, **options):
        return self._apply_op(ket, 'Jx', **options)

    # 应用 Jy 操作符到给定的 ket
    def _apply_operator_JyKet(self, ket, **options):
        return self._apply_op(ket, 'Jy', **options)

    # 应用 Jy 操作符到耦合态的 ket
    def _apply_operator_JyKetCoupled(self, ket, **options):
        return self._apply_op(ket, 'Jy', **options)

    # 应用 Jz 操作符到给定的 ket
    def _apply_operator_JzKet(self, ket, **options):
        return self._apply_op(ket, 'Jz', **options)

    # 应用 Jz 操作符到耦合态的 ket
    def _apply_operator_JzKetCoupled(self, ket, **options):
        return self._apply_op(ket, 'Jz', **options)

    # 应用 TensorProduct 操作符到给定的 tp
    def _apply_operator_TensorProduct(self, tp, **options):
        # 仅当 self 是 JxOp、JyOp 或 JzOp 类型时，才有易于找到的解耦操作符
        if not isinstance(self, (JxOp, JyOp, JzOp)):
            raise NotImplementedError
        # 创建一个空列表，用于存放结果
        result = []
        # 遍历 tp.args 中的每一个元素
        for n in range(len(tp.args)):
            # 创建一个新的参数列表 arg，并将前 n-1 个元素添加进去
            arg = []
            arg.extend(tp.args[:n])
            # 将当前 self 对 tp.args[n] 应用操作符并添加到 arg 中
            arg.append(self._apply_operator(tp.args[n]))
            # 将后续元素添加到 arg 中
            arg.extend(tp.args[n + 1:])
            # 根据 arg 创建一个新的 TensorProduct 对象，并添加到结果列表中
            result.append(tp.__class__(*arg))
        # 将结果列表中的所有对象相加并展开并返回
        return Add(*result).expand()

    # 将此方法移到 qapply_Mul 中（待办事项）
    def _apply_operator_Sum(self, s, **options):
        # 对 s.function 应用 qapply 函数得到 new_func
        new_func = qapply(self*s.function)
        # 如果 new_func 等于 self*s.function，则抛出未实现异常
        if new_func == self*s.function:
            raise NotImplementedError
        # 创建一个新的 Sum 对象，其中函数为 new_func，限制条件为 s.limits，并返回
        return Sum(new_func, *s.limits)

    # 计算迹，使用给定的选项（待办事项）
    def _eval_trace(self, **options):
        # TODO: 使用选项来使用不同的 j 值
        # 目前使用默认基础来评估迹
        # 每次都重新表示来计算迹是否高效？
        return self._represent_default_basis().trace()
class JplusOp(SpinOpBase, Operator):
    """The J+ operator."""

    _coord = '+'  # 设置运算符的符号为'+'

    basis = 'Jz'  # 设置基础为'Jz'

    def _eval_commutator_JminusOp(self, other):
        # 计算 J- 操作符的对易子
        return 2*hbar*JzOp(self.name)

    def _apply_operator_JzKet(self, ket, **options):
        j = ket.j  # 获取态矢量 ket 的量子数 j
        m = ket.m  # 获取态矢量 ket 的量子数 m
        if m.is_Number and j.is_Number:
            if m >= j:
                return S.Zero  # 如果 m >= j，则返回零态
        # 应用 JzKet 的升算符操作
        return hbar*sqrt(j*(j + S.One) - m*(m + S.One))*JzKet(j, m + S.One)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        j = ket.j  # 获取态矢量 ket 的量子数 j
        m = ket.m  # 获取态矢量 ket 的量子数 m
        jn = ket.jn  # 获取态矢量 ket 的耦合量子数 jn
        coupling = ket.coupling  # 获取态矢量 ket 的耦合
        if m.is_Number and j.is_Number:
            if m >= j:
                return S.Zero  # 如果 m >= j，则返回零态
        # 应用耦合的 JzKet 的升算符操作
        return hbar*sqrt(j*(j + S.One) - m*(m + S.One))*JzKetCoupled(j, m + S.One, jn, coupling)

    def matrix_element(self, j, m, jp, mp):
        # 计算矩阵元素
        result = hbar*sqrt(j*(j + S.One) - mp*(mp + S.One))
        result *= KroneckerDelta(m, mp + 1)  # 乘以克罗内克 δ 函数
        result *= KroneckerDelta(j, jp)  # 再乘以克罗内克 δ 函数
        return result  # 返回计算结果

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)  # 默认表示基础为 JzOp

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)  # 表示 JzOp 的基础

    def _eval_rewrite_as_xyz(self, *args, **kwargs):
        return JxOp(args[0]) + I*JyOp(args[0])  # 重写为 JxOp 和 JyOp 的和


class JminusOp(SpinOpBase, Operator):
    """The J- operator."""

    _coord = '-'  # 设置运算符的符号为'-'

    basis = 'Jz'  # 设置基础为'Jz'

    def _apply_operator_JzKet(self, ket, **options):
        j = ket.j  # 获取态矢量 ket 的量子数 j
        m = ket.m  # 获取态矢量 ket 的量子数 m
        if m.is_Number and j.is_Number:
            if m <= -j:
                return S.Zero  # 如果 m <= -j，则返回零态
        # 应用 JzKet 的降算符操作
        return hbar*sqrt(j*(j + S.One) - m*(m - S.One))*JzKet(j, m - S.One)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        j = ket.j  # 获取态矢量 ket 的量子数 j
        m = ket.m  # 获取态矢量 ket 的量子数 m
        jn = ket.jn  # 获取态矢量 ket 的耦合量子数 jn
        coupling = ket.coupling  # 获取态矢量 ket 的耦合
        if m.is_Number and j.is_Number:
            if m <= -j:
                return S.Zero  # 如果 m <= -j，则返回零态
        # 应用耦合的 JzKet 的降算符操作
        return hbar*sqrt(j*(j + S.One) - m*(m - S.One))*JzKetCoupled(j, m - S.One, jn, coupling)

    def matrix_element(self, j, m, jp, mp):
        # 计算矩阵元素
        result = hbar*sqrt(j*(j + S.One) - mp*(mp - S.One))
        result *= KroneckerDelta(m, mp - 1)  # 乘以克罗内克 δ 函数
        result *= KroneckerDelta(j, jp)  # 再乘以克罗内克 δ 函数
        return result  # 返回计算结果

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)  # 默认表示基础为 JzOp

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)  # 表示 JzOp 的基础

    def _eval_rewrite_as_xyz(self, *args, **kwargs):
        return JxOp(args[0]) - I*JyOp(args[0])  # 重写为 JxOp 和 JyOp 的差


class JxOp(SpinOpBase, HermitianOperator):
    """The Jx operator."""

    _coord = 'x'  # 设置运算符的符号为'x'

    basis = 'Jx'  # 设置基础为'Jx'

    def _eval_commutator_JyOp(self, other):
        # 计算 Jy 操作符的对易子
        return I*hbar*JzOp(self.name)

    def _eval_commutator_JzOp(self, other):
        # 计算 Jz 操作符的对易子
        return -I*hbar*JyOp(self.name)
    # 对给定的 ket 应用 J+ 操作符和 J- 操作符，然后返回它们的平均值
    def _apply_operator_JzKet(self, ket, **options):
        # 创建 J+ 操作符对象并对 ket 应用该操作符
        jp = JplusOp(self.name)._apply_operator_JzKet(ket, **options)
        # 创建 J- 操作符对象并对 ket 应用该操作符
        jm = JminusOp(self.name)._apply_operator_JzKet(ket, **options)
        # 返回 J+ 和 J- 操作符应用后的结果的平均值
        return (jp + jm)/Integer(2)

    # 对耦合 J+ 和 J- 操作符应用到给定的 ket，并返回它们的平均值
    def _apply_operator_JzKetCoupled(self, ket, **options):
        # 创建耦合 J+ 操作符对象并对 ket 应用该操作符
        jp = JplusOp(self.name)._apply_operator_JzKetCoupled(ket, **options)
        # 创建耦合 J- 操作符对象并对 ket 应用该操作符
        jm = JminusOp(self.name)._apply_operator_JzKetCoupled(ket, **options)
        # 返回耦合 J+ 和 J- 操作符应用后的结果的平均值
        return (jp + jm)/Integer(2)

    # 使用默认基底来表示 Jz 操作符
    def _represent_default_basis(self, **options):
        # 调用 _represent_JzOp 方法，传入 None 作为基底参数
        return self._represent_JzOp(None, **options)

    # 表示 Jz 操作符在给定基底下的表示
    def _represent_JzOp(self, basis, **options):
        # 创建 J+ 操作符对象并在给定基底下表示它
        jp = JplusOp(self.name)._represent_JzOp(basis, **options)
        # 创建 J- 操作符对象并在给定基底下表示它
        jm = JminusOp(self.name)._represent_JzOp(basis, **options)
        # 返回 J+ 和 J- 操作符在给定基底下表示的结果的平均值
        return (jp + jm)/Integer(2)

    # 将当前对象重写为 J+ 和 J- 操作符之和除以 2 的形式
    def _eval_rewrite_as_plusminus(self, *args, **kwargs):
        # 创建 J+ 操作符对象和 J- 操作符对象，然后将它们相加并除以 2
        return (JplusOp(args[0]) + JminusOp(args[0]))/2
class JyOp(SpinOpBase, HermitianOperator):
    """The Jy operator."""

    _coord = 'y'  # 设置操作符的坐标属性为 'y'

    basis = 'Jy'  # 设置操作符的基础属性为 'Jy'

    def _eval_commutator_JzOp(self, other):
        # 计算 JzOp 与当前操作符的对易子，返回结果为 -i*hbar*JxOp 的形式
        return I*hbar*JxOp(self.name)

    def _eval_commutator_JxOp(self, other):
        # 计算 JxOp 与当前操作符的对易子，返回结果为 -i*hbar*J2Op 的形式
        return -I*hbar*J2Op(self.name)

    def _apply_operator_JzKet(self, ket, **options):
        # 对 JzKet 应用当前操作符，返回 (J+ - J-)/(2*i) 的结果
        jp = JplusOp(self.name)._apply_operator_JzKet(ket, **options)
        jm = JminusOp(self.name)._apply_operator_JzKet(ket, **options)
        return (jp - jm)/(Integer(2)*I)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        # 对 JzKetCoupled 应用当前操作符，返回 (J+ - J-)/(2*i) 的结果
        jp = JplusOp(self.name)._apply_operator_JzKetCoupled(ket, **options)
        jm = JminusOp(self.name)._apply_operator_JzKetCoupled(ket, **options)
        return (jp - jm)/(Integer(2)*I)

    def _represent_default_basis(self, **options):
        # 默认基础情况下，表示当前操作符的 JzOp
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        # 表示在特定基础下，表示当前操作符的 JzOp
        jp = JplusOp(self.name)._represent_JzOp(basis, **options)
        jm = JminusOp(self.name)._represent_JzOp(basis, **options)
        return (jp - jm)/(Integer(2)*I)

    def _eval_rewrite_as_plusminus(self, *args, **kwargs):
        # 以加减号形式重写当前操作符，返回 (J+ - J-)/(2*i) 的结果
        return (JplusOp(args[0]) - JminusOp(args[0]))/(2*I)


class JzOp(SpinOpBase, HermitianOperator):
    """The Jz operator."""

    _coord = 'z'  # 设置操作符的坐标属性为 'z'

    basis = 'Jz'  # 设置操作符的基础属性为 'Jz'

    def _eval_commutator_JxOp(self, other):
        # 计算 JxOp 与当前操作符的对易子，返回结果为 i*hbar*JyOp 的形式
        return I*hbar*JyOp(self.name)

    def _eval_commutator_JyOp(self, other):
        # 计算 JyOp 与当前操作符的对易子，返回结果为 -i*hbar*JxOp 的形式
        return -I*hbar*JxOp(self.name)

    def _eval_commutator_JplusOp(self, other):
        # 计算 JplusOp 与当前操作符的对易子，返回结果为 hbar*JplusOp 的形式
        return hbar*JplusOp(self.name)

    def _eval_commutator_JminusOp(self, other):
        # 计算 JminusOp 与当前操作符的对易子，返回结果为 -hbar*JminusOp 的形式
        return -hbar*JminusOp(self.name)

    def matrix_element(self, j, m, jp, mp):
        # 计算矩阵元素，返回 hbar*mp*delta(m, mp)*delta(j, jp) 的结果
        result = hbar*mp
        result *= KroneckerDelta(m, mp)
        result *= KroneckerDelta(j, jp)
        return result

    def _represent_default_basis(self, **options):
        # 默认基础情况下，表示当前操作符的 JzOp
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        # 表示在特定基础下，表示当前操作符的 JzOp
        return self._represent_base(basis, **options)


class J2Op(SpinOpBase, HermitianOperator):
    """The J^2 operator."""

    _coord = '2'  # 设置操作符的坐标属性为 '2'

    def _eval_commutator_JxOp(self, other):
        # 计算 JxOp 与当前操作符的对易子，返回零值
        return S.Zero

    def _eval_commutator_JyOp(self, other):
        # 计算 JyOp 与当前操作符的对易子，返回零值
        return S.Zero

    def _eval_commutator_JzOp(self, other):
        # 计算 JzOp 与当前操作符的对易子，返回零值
        return S.Zero

    def _eval_commutator_JplusOp(self, other):
        # 计算 JplusOp 与当前操作符的对易子，返回零值
        return S.Zero

    def _eval_commutator_JminusOp(self, other):
        # 计算 JminusOp 与当前操作符的对易子，返回零值
        return S.Zero

    def _apply_operator_JxKet(self, ket, **options):
        # 对 JxKet 应用当前操作符，返回 hbar^2*j*(j + 1)*ket 的结果
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_JxKetCoupled(self, ket, **options):
        # 对 JxKetCoupled 应用当前操作符，返回 hbar^2*j*(j + 1)*ket 的结果
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_JyKet(self, ket, **options):
        # 对 JyKet 应用当前操作符，返回 hbar^2*j*(j + 1)*ket 的结果
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_JyKetCoupled(self, ket, **options):
        # 对 JyKetCoupled 应用当前操作符，返回 hbar^2*j*(j + 1)*ket 的结果
        j = ket.j
        return hbar**2*j*(j + 1)*ket
    # 计算给定 JzKet 对象的作用，返回结果
    def _apply_operator_JzKet(self, ket, **options):
        # 获取 JzKet 对象的量子数 j
        j = ket.j
        # 计算并返回结果，包括常数 hbar^2 和 j*(j + 1) 的乘积
        return hbar**2*j*(j + 1)*ket

    # 计算给定 JzKetCoupled 对象的作用，返回结果
    def _apply_operator_JzKetCoupled(self, ket, **options):
        # 获取 JzKetCoupled 对象的量子数 j
        j = ket.j
        # 计算并返回结果，包括常数 hbar^2 和 j*(j + 1) 的乘积
        return hbar**2*j*(j + 1)*ket

    # 计算给定量子数 j, m, jp, mp 的矩阵元素并返回结果
    def matrix_element(self, j, m, jp, mp):
        # 计算矩阵元素，包括常数 hbar^2*j*(j + 1)，以及 KroneckerDelta 的乘积
        result = (hbar**2)*j*(j + 1)
        result *= KroneckerDelta(m, mp)
        result *= KroneckerDelta(j, jp)
        return result

    # 使用默认基底表示 JzOp 操作符
    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    # 根据指定的基底表示 JzOp 操作符
    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    # 以漂亮的方式打印对象内容到 printer
    def _print_contents_pretty(self, printer, *args):
        # 创建一个漂亮格式的对象 a，内容为对象的名称
        a = prettyForm(str(self.name))
        # 创建一个漂亮格式的对象 b，内容为字符串 '2'
        b = prettyForm('2')
        # 返回 a 的 b 次幂的漂亮格式对象
        return a**b

    # 以 LaTeX 格式打印对象内容
    def _print_contents_latex(self, printer, *args):
        # 返回对象名称的平方的 LaTeX 表示
        return r'%s^2' % str(self.name)

    # 将对象的重写表达为 JxOp, JyOp 和 JzOp 操作符的平方和
    def _eval_rewrite_as_xyz(self, *args, **kwargs):
        return JxOp(args[0])**2 + JyOp(args[0])**2 + JzOp(args[0])**2

    # 将对象的重写表达为 JzOp(a) 的平方加上 JplusOp(a)*JminusOp(a) 和 JminusOp(a)*JplusOp(a) 的 1/2
    def _eval_rewrite_as_plusminus(self, *args, **kwargs):
        # 获取参数 a
        a = args[0]
        # 返回 JzOp(a) 的平方加上 JplusOp(a)*JminusOp(a) 和 JminusOp(a)*JplusOp(a) 的 1/2
        return JzOp(a)**2 + \
            S.Half*(JplusOp(a)*JminusOp(a) + JminusOp(a)*JplusOp(a))
# 定义一个旋转操作符类，继承自 UnitaryOperator 类
class Rotation(UnitaryOperator):
    
    """Wigner D operator in terms of Euler angles.

    Defines the rotation operator in terms of the Euler angles defined by
    the z-y-z convention for a passive transformation. That is the coordinate
    axes are rotated first about the z-axis, giving the new x'-y'-z' axes. Then
    this new coordinate system is rotated about the new y'-axis, giving new
    x''-y''-z'' axes. Then this new coordinate system is rotated about the
    z''-axis. Conventions follow those laid out in [1]_.
    
    参数
    ==========

    alpha : Number, Symbol
        第一个欧拉角
    beta : Number, Symbol
        第二个欧拉角
    gamma : Number, Symbol
        第三个欧拉角

    示例
    ========

    一个简单的旋转操作符示例：

        >>> from sympy import pi
        >>> from sympy.physics.quantum.spin import Rotation
        >>> Rotation(pi, 0, pi/2)
        R(pi,0,pi/2)

    使用符号表达式的欧拉角，并计算逆旋转操作符：

        >>> from sympy import symbols
        >>> a, b, c = symbols('a b c')
        >>> Rotation(a, b, c)
        R(a,b,c)
        >>> Rotation(a, b, c).inverse()
        R(-c,-b,-a)

    另请参阅
    ========

    WignerD: 符号 Wigner-D 函数
    D: Wigner-D 函数
    d: Wigner 小 d 函数

    参考文献
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """

    @classmethod
    def _eval_args(cls, args):
        # 调用父类的 _eval_args 方法，验证参数是否为 QExpr 类型
        args = QExpr._eval_args(args)
        # 如果参数数量不等于 3，则抛出 ValueError 异常
        if len(args) != 3:
            raise ValueError('3 Euler angles required, got: %r' % args)
        return args

    @classmethod
    def _eval_hilbert_space(cls, label):
        # 考虑所有可能的 j 值，因此我们的空间是无限的
        return ComplexSpace(S.Infinity)

    @property
    def alpha(self):
        # 返回属性 label 的第一个元素作为 alpha 欧拉角
        return self.label[0]

    @property
    def beta(self):
        # 返回属性 label 的第二个元素作为 beta 欧拉角
        return self.label[1]

    @property
    def gamma(self):
        # 返回属性 label 的第三个元素作为 gamma 欧拉角
        return self.label[2]

    def _print_operator_name(self, printer, *args):
        # 打印操作符名称为 'R'
        return 'R'

    def _print_operator_name_pretty(self, printer, *args):
        # 如果打印器使用 Unicode，则返回美化后的操作符名称
        if printer._use_unicode:
            return prettyForm('\N{SCRIPT CAPITAL R}' + ' ')
        else:
            return prettyForm("R ")

    def _print_operator_name_latex(self, printer, *args):
        # 返回 LaTeX 格式的操作符名称
        return r'\mathcal{R}'

    def _eval_inverse(self):
        # 计算旋转操作符的逆，欧拉角取负值
        return Rotation(-self.gamma, -self.beta, -self.alpha)

    @classmethod
    def D(cls, j, m, mp, alpha, beta, gamma):
        """Wigner D-function.

        Returns an instance of the WignerD class corresponding to the Wigner-D
        function specified by the parameters.

        Parameters
        ===========

        j : Number
            Total angular momentum
        m : Number
            Eigenvalue of angular momentum along axis after rotation
        mp : Number
            Eigenvalue of angular momentum along rotated axis
        alpha : Number, Symbol
            First Euler angle of rotation
        beta : Number, Symbol
            Second Euler angle of rotation
        gamma : Number, Symbol
            Third Euler angle of rotation

        Examples
        ========

        Return the Wigner-D matrix element for a defined rotation, both
        numerical and symbolic:

            >>> from sympy.physics.quantum.spin import Rotation
            >>> from sympy import pi, symbols
            >>> alpha, beta, gamma = symbols('alpha beta gamma')
            >>> Rotation.D(1, 1, 0, pi, pi/2, -pi)
            WignerD(1, 1, 0, pi, pi/2, -pi)

        See Also
        ========

        WignerD: Symbolic Wigner-D function

        """
        # 返回符合指定参数的 Wigner-D 函数实例
        return WignerD(j, m, mp, alpha, beta, gamma)

    @classmethod
    def d(cls, j, m, mp, beta):
        """Wigner small-d function.

        Returns an instance of the WignerD class corresponding to the Wigner-D
        function specified by the parameters with the alpha and gamma angles
        given as 0.

        Parameters
        ===========

        j : Number
            Total angular momentum
        m : Number
            Eigenvalue of angular momentum along axis after rotation
        mp : Number
            Eigenvalue of angular momentum along rotated axis
        beta : Number, Symbol
            Second Euler angle of rotation

        Examples
        ========

        Return the Wigner-D matrix element for a defined rotation, both
        numerical and symbolic:

            >>> from sympy.physics.quantum.spin import Rotation
            >>> from sympy import pi, symbols
            >>> beta = symbols('beta')
            >>> Rotation.d(1, 1, 0, pi/2)
            WignerD(1, 1, 0, 0, pi/2, 0)

        See Also
        ========

        WignerD: Symbolic Wigner-D function

        """
        # 返回符合指定参数的 Wigner-D 函数实例，其中 alpha 和 gamma 角度为 0
        return WignerD(j, m, mp, 0, beta, 0)

    def matrix_element(self, j, m, jp, mp):
        # 计算 Wigner-D 矩阵元素，乘以 KroneckerDelta(j, jp) 来确保 j 和 jp 相等时才有非零值
        result = self.__class__.D(
            jp, m, mp, self.alpha, self.beta, self.gamma
        )
        result *= KroneckerDelta(j, jp)
        return result
    def _represent_base(self, basis, **options):
        j = sympify(options.get('j', S.Half))
        # TODO: move evaluation up to represent function/implement elsewhere
        # 使用 sympify 函数将参数 'j' 转换为符号表达式，默认为 S.Half
        evaluate = sympify(options.get('doit'))
        # 使用 sympify 函数将参数 'doit' 转换为符号表达式，表示是否执行计算
        size, mvals = m_values(j)
        # 调用 m_values 函数计算给定 j 的尺寸和 m 值列表
        result = zeros(size, size)
        # 创建一个 size x size 的零矩阵作为结果
        for p in range(size):
            for q in range(size):
                # 循环计算矩阵元素
                me = self.matrix_element(j, mvals[p], j, mvals[q])
                # 计算矩阵元素值
                if evaluate:
                    result[p, q] = me.doit()
                    # 如果 evaluate 为真，则计算矩阵元素的值
                else:
                    result[p, q] = me
                    # 否则直接使用未计算的矩阵元素
        return result
        # 返回计算得到的矩阵

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)
        # 调用 _represent_JzOp 方法，返回默认基底的表示

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)
        # 调用 _represent_base 方法，表示 Jz 操作符的基底

    def _apply_operator_uncoupled(self, state, ket, *, dummy=True, **options):
        a = self.alpha
        b = self.beta
        g = self.gamma
        j = ket.j
        m = ket.m
        # 从 ket 参数中获取 j 和 m 值
        if j.is_number:
            s = []
            size = m_values(j)
            sz = size[1]
            for mp in sz:
                r = Rotation.D(j, m, mp, a, b, g)
                z = r.doit()
                s.append(z*state(j, mp))
            return Add(*s)
            # 如果 j 是数值，则计算对角化态的线性组合
        else:
            if dummy:
                mp = Dummy('mp')
            else:
                mp = symbols('mp')
            return Sum(Rotation.D(j, m, mp, a, b, g)*state(j, mp), (mp, -j, j))
            # 如果 j 不是数值，则对 m' 进行求和计算

    def _apply_operator_JxKet(self, ket, **options):
        return self._apply_operator_uncoupled(JxKet, ket, **options)
        # 调用 _apply_operator_uncoupled 方法，应用 JxKet 算符到 ket 上

    def _apply_operator_JyKet(self, ket, **options):
        return self._apply_operator_uncoupled(JyKet, ket, **options)
        # 调用 _apply_operator_uncoupled 方法，应用 JyKet 算符到 ket 上

    def _apply_operator_JzKet(self, ket, **options):
        return self._apply_operator_uncoupled(JzKet, ket, **options)
        # 调用 _apply_operator_uncoupled 方法，应用 JzKet 算符到 ket 上

    def _apply_operator_coupled(self, state, ket, *, dummy=True, **options):
        a = self.alpha
        b = self.beta
        g = self.gamma
        j = ket.j
        m = ket.m
        jn = ket.jn
        coupling = ket.coupling
        # 从 ket 参数中获取 j, m, jn 和 coupling 值
        if j.is_number:
            s = []
            size = m_values(j)
            sz = size[1]
            for mp in sz:
                r = Rotation.D(j, m, mp, a, b, g)
                z = r.doit()
                s.append(z*state(j, mp, jn, coupling))
            return Add(*s)
            # 如果 j 是数值，则计算耦合态的线性组合
        else:
            if dummy:
                mp = Dummy('mp')
            else:
                mp = symbols('mp')
            return Sum(Rotation.D(j, m, mp, a, b, g)*state(
                j, mp, jn, coupling), (mp, -j, j))
            # 如果 j 不是数值，则对 m' 进行求和计算

    def _apply_operator_JxKetCoupled(self, ket, **options):
        return self._apply_operator_coupled(JxKetCoupled, ket, **options)
        # 调用 _apply_operator_coupled 方法，应用耦合 JxKetCoupled 算符到 ket 上

    def _apply_operator_JyKetCoupled(self, ket, **options):
        return self._apply_operator_coupled(JyKetCoupled, ket, **options)
        # 调用 _apply_operator_coupled 方法，应用耦合 JyKetCoupled 算符到 ket 上

    def _apply_operator_JzKetCoupled(self, ket, **options):
        return self._apply_operator_coupled(JzKetCoupled, ket, **options)
        # 调用 _apply_operator_coupled 方法，应用耦合 JzKetCoupled 算符到 ket 上
# 定义一个表示 Wigner-D 函数的类，继承自 Expr 类
class WignerD(Expr):
    r"""Wigner-D function

    The Wigner D-function gives the matrix elements of the rotation
    operator in the jm-representation. For the Euler angles `\alpha`,
    `\beta`, `\gamma`, the D-function is defined such that:

    .. math ::
        <j,m| \mathcal{R}(\alpha, \beta, \gamma ) |j',m'> = \delta_{jj'} D(j, m, m', \alpha, \beta, \gamma)

    Where the rotation operator is as defined by the Rotation class [1]_.

    The Wigner D-function defined in this way gives:

    .. math ::
        D(j, m, m', \alpha, \beta, \gamma) = e^{-i m \alpha} d(j, m, m', \beta) e^{-i m' \gamma}

    Where d is the Wigner small-d function, which is given by Rotation.d.

    The Wigner small-d function gives the component of the Wigner
    D-function that is determined by the second Euler angle. That is the
    Wigner D-function is:

    .. math ::
        D(j, m, m', \alpha, \beta, \gamma) = e^{-i m \alpha} d(j, m, m', \beta) e^{-i m' \gamma}

    Where d is the small-d function. The Wigner D-function is given by
    Rotation.D.

    Note that to evaluate the D-function, the j, m and mp parameters must
    be integer or half integer numbers.

    Parameters
    ==========

    j : Number
        Total angular momentum
    m : Number
        Eigenvalue of angular momentum along axis after rotation
    mp : Number
        Eigenvalue of angular momentum along rotated axis
    alpha : Number, Symbol
        First Euler angle of rotation
    beta : Number, Symbol
        Second Euler angle of rotation
    gamma : Number, Symbol
        Third Euler angle of rotation

    Examples
    ========

    Evaluate the Wigner-D matrix elements of a simple rotation:

        >>> from sympy.physics.quantum.spin import Rotation
        >>> from sympy import pi
        >>> rot = Rotation.D(1, 1, 0, pi, pi/2, 0)
        >>> rot
        WignerD(1, 1, 0, pi, pi/2, 0)
        >>> rot.doit()
        sqrt(2)/2

    Evaluate the Wigner-d matrix elements of a simple rotation

        >>> rot = Rotation.d(1, 1, 0, pi/2)
        >>> rot
        WignerD(1, 1, 0, 0, pi/2, 0)
        >>> rot.doit()
        -sqrt(2)/2

    See Also
    ========

    Rotation: Rotation operator

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """

    # 设置该类的运算是否可交换
    is_commutative = True

    # 构造函数，接受参数并返回一个 WignerD 对象
    def __new__(cls, *args, **hints):
        # 如果参数个数不等于6，则引发 ValueError 异常
        if not len(args) == 6:
            raise ValueError('6 parameters expected, got %s' % args)
        # 将参数转换为 Sympy 可计算对象
        args = sympify(args)
        # 获取是否要求立即计算的标志位，默认为 False
        evaluate = hints.get('evaluate', False)
        # 如果设置了 evaluate 为 True，则返回 _eval_wignerd 方法的计算结果
        if evaluate:
            return Expr.__new__(cls, *args)._eval_wignerd()
        # 否则返回 WignerD 对象
        return Expr.__new__(cls, *args)

    # j 属性，返回 WignerD 对象的第一个参数
    @property
    def j(self):
        return self.args[0]

    # m 属性，返回 WignerD 对象的第二个参数
    @property
    def m(self):
        return self.args[1]

    # mp 属性，返回 WignerD 对象的第三个参数
    @property
    def mp(self):
        return self.args[2]

    # alpha 属性，返回 WignerD 对象的第四个参数
    @property
    def alpha(self):
        return self.args[3]

    # beta 属性，返回 WignerD 对象的第五个参数
    @property
    def beta(self):
        return self.args[4]

    # gamma 属性，返回 WignerD 对象的第六个参数
    @property
    def gamma(self):
        return self.args[5]
    # 返回对象的第六个参数 gamma
    def gamma(self):
        return self.args[5]

    # 根据对象的属性 alpha 和 gamma 的值生成 LaTeX 格式的字符串表示
    def _latex(self, printer, *args):
        # 如果 alpha 和 gamma 都为 0，则使用简化的表示方式
        if self.alpha == 0 and self.gamma == 0:
            return r'd^{%s}_{%s,%s}\left(%s\right)' % \
                (
                    printer._print(self.j), printer._print(
                        self.m), printer._print(self.mp),
                    printer._print(self.beta) )
        # 否则使用完整的表示方式，包括 alpha, beta 和 gamma
        return r'D^{%s}_{%s,%s}\left(%s,%s,%s\right)' % \
            (
                printer._print(
                    self.j), printer._print(self.m), printer._print(self.mp),
                printer._print(self.alpha), printer._print(self.beta), printer._print(self.gamma) )

    # 根据对象的属性生成漂亮打印格式的字符串表示
    def _pretty(self, printer, *args):
        # 获取顶部和底部的打印格式
        top = printer._print(self.j)
        bot = printer._print(self.m)
        bot = prettyForm(*bot.right(','))  # 在底部打印格式后添加逗号
        bot = prettyForm(*bot.right(printer._print(self.mp)))  # 在底部打印格式后添加 mp 的打印格式

        # 计算最大宽度并调整顶部和底部的打印格式
        pad = max(top.width(), bot.width())
        top = prettyForm(*top.left(' '))
        bot = prettyForm(*bot.left(' '))
        if pad > top.width():
            top = prettyForm(*top.right(' '*(pad - top.width())))
        if pad > bot.width():
            bot = prettyForm(*bot.right(' '*(pad - bot.width())))

        # 根据 alpha 和 gamma 的值选择不同的参数格式
        if self.alpha == 0 and self.gamma == 0:
            args = printer._print(self.beta)
            s = stringPict('d' + ' '*pad)
        else:
            args = printer._print(self.alpha)
            args = prettyForm(*args.right(','))  # 在参数格式后添加逗号
            args = prettyForm(*args.right(printer._print(self.beta)))  # 在参数格式后添加 beta 的打印格式
            args = prettyForm(*args.right(','))  # 再次在参数格式后添加逗号
            args = prettyForm(*args.right(printer._print(self.gamma)))  # 最后添加 gamma 的打印格式

            s = stringPict('D' + ' '*pad)

        args = prettyForm(*args.parens())  # 在所有参数格式外围添加括号
        s = prettyForm(*s.above(top))  # 在顶部打印格式上方放置 s
        s = prettyForm(*s.below(bot))  # 在底部打印格式下方放置 s
        s = prettyForm(*s.right(args))  # 将参数格式放置在 s 的右侧
        return s

    # 设置 hints 字典中的 evaluate 键为 True，然后调用 WignerD 类的构造函数
    def doit(self, **hints):
        hints['evaluate'] = True
        return WignerD(*self.args, **hints)
    # 定义一个方法用于计算 Wigner-d 矩阵元素
    def _eval_wignerd(self):
        # 获取输入参数
        j = self.j
        m = self.m
        mp = self.mp
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        
        # 如果 alpha、beta 和 gamma 均为零，根据 KroneckerDelta 返回结果
        if alpha == 0 and beta == 0 and gamma == 0:
            return KroneckerDelta(m, mp)
        
        # 如果 j 不是数值类型，抛出 ValueError 异常
        if not j.is_number:
            raise ValueError(
                'j parameter must be numerical to evaluate, got %s' % j)
        
        # 初始化结果变量 r
        r = 0
        
        # 当 beta 等于 pi/2 时，根据 Varshalovich 方程 (5)，Section 4.16, page 113，设定 alpha=gamma=0 的情况
        if beta == pi/2:
            # 根据 Varshalovich 方程 (5)，计算 Wigner-d 矩阵元素
            for k in range(2*j + 1):
                if k > j + mp or k > j - m or k < mp - m:
                    continue
                r += (S.NegativeOne)**k * binomial(j + mp, k) * binomial(j - mp, k + m - mp)
            r *= (S.NegativeOne)**(m - mp) / (2**j * sqrt(factorial(j + m) * factorial(j - m) / (factorial(j + mp) * factorial(j - mp))))
        else:
            # 根据 Varshalovich 方程 (5)，Section 4.7.2, page 87，设置 beta1=beta2=pi/2 的情况
            # 此时 alpha=gamma=pi/2，beta=phi+pi，利用 Eq. (1)，Section 4.4. page 79 进行简化
            # d(j, m, mp, beta+pi) = (-1)**(j-mp) * d(j, m, -mp, beta)
            # 这与 Eq. (10)，Section 4.16 几乎相同，但需要将 mp 替换为 -mp
            size, mvals = m_values(j)
            for mpp in mvals:
                r += Rotation.d(j, m, mpp, pi/2).doit() * (cos(-mpp*beta) + I*sin(-mpp*beta)) * Rotation.d(j, mpp, -mp, pi/2).doit()
            
            # 经验性归一化因子，使结果与 Varshalovich Tables 4.3-4.12 匹配
            # 注意，这个归一化并不完全来自上述方程
            r = r * I**(2*j - m - mp) * (-1)**(2*m)
        
        # 最后，对整个表达式进行简化处理
        r = simplify(r)
        
        # 最终乘以 exp(-I*m*alpha) 和 exp(-I*mp*gamma)
        r *= exp(-I*m*alpha) * exp(-I*mp*gamma)
        
        # 返回计算结果
        return r
# 创建 JxOp 对象，用于表示 Jx 操作符
Jx = JxOp('J')
# 创建 JyOp 对象，用于表示 Jy 操作符
Jy = JyOp('J')
# 创建 JzOp 对象，用于表示 Jz 操作符
Jz = JzOp('J')
# 创建 J2Op 对象，用于表示 J^2 操作符
J2 = J2Op('J')
# 创建 JplusOp 对象，用于表示 J^+ 操作符
Jplus = JplusOp('J')
# 创建 JminusOp 对象，用于表示 J^- 操作符
Jminus = JminusOp('J')


#-----------------------------------------------------------------------------
# Spin States
#-----------------------------------------------------------------------------


class SpinState(State):
    """Base class for angular momentum states."""

    _label_separator = ','

    def __new__(cls, j, m):
        # 使用 sympify 将 j 和 m 转换为符号表达式
        j = sympify(j)
        m = sympify(m)
        # 检查 j 是否为数值，且为整数或半整数
        if j.is_number:
            if 2*j != int(2*j):
                raise ValueError(
                    'j must be integer or half-integer, got: %s' % j)
            if j < 0:
                raise ValueError('j must be >= 0, got: %s' % j)
        # 检查 m 是否为数值，且为整数或半整数
        if m.is_number:
            if 2*m != int(2*m):
                raise ValueError(
                    'm must be integer or half-integer, got: %s' % m)
        # 如果 j 和 m 都是数值，检查 m 是否在合理范围内
        if j.is_number and m.is_number:
            if abs(m) > j:
                raise ValueError('Allowed values for m are -j <= m <= j, got j, m: %s, %s' % (j, m))
            if int(j - m) != j - m:
                raise ValueError('Both j and m must be integer or half-integer, got j, m: %s, %s' % (j, m))
        # 调用父类 State 的构造函数创建新的 SpinState 对象
        return State.__new__(cls, j, m)

    @property
    def j(self):
        # 返回角动量量子数 j
        return self.label[0]

    @property
    def m(self):
        # 返回角动量量子数 m
        return self.label[1]

    @classmethod
    def _eval_hilbert_space(cls, label):
        # 计算 Hilbert 空间的维数，由 2*j + 1 决定
        return ComplexSpace(2*label[0] + 1)

    def _represent_base(self, **options):
        # 根据给定的选项返回基底表示
        j = self.j
        m = self.m
        alpha = sympify(options.get('alpha', 0))
        beta = sympify(options.get('beta', 0))
        gamma = sympify(options.get('gamma', 0))
        size, mvals = m_values(j)
        result = zeros(size, 1)
        # 遍历 mvals，计算旋转矩阵元素
        for p, mval in enumerate(mvals):
            if m.is_number:
                result[p, 0] = Rotation.D(
                    self.j, mval, self.m, alpha, beta, gamma).doit()
            else:
                result[p, 0] = Rotation.D(self.j, mval,
                                          self.m, alpha, beta, gamma)
        return result

    def _eval_rewrite_as_Jx(self, *args, **options):
        # 将 SpinState 重写为 Jx 的基底
        if isinstance(self, Bra):
            return self._rewrite_basis(Jx, JxBra, **options)
        return self._rewrite_basis(Jx, JxKet, **options)

    def _eval_rewrite_as_Jy(self, *args, **options):
        # 将 SpinState 重写为 Jy 的基底
        if isinstance(self, Bra):
            return self._rewrite_basis(Jy, JyBra, **options)
        return self._rewrite_basis(Jy, JyKet, **options)

    def _eval_rewrite_as_Jz(self, *args, **options):
        # 将 SpinState 重写为 Jz 的基底
        if isinstance(self, Bra):
            return self._rewrite_basis(Jz, JzBra, **options)
        return self._rewrite_basis(Jz, JzKet, **options)
    # 重写基础函数，根据给定的基向量和特征向量进行重新构造
    def _rewrite_basis(self, basis, evect, **options):
        # 导入表示模块中的represent函数
        from sympy.physics.quantum.represent import represent
        # 获取自旋量子数j
        j = self.j
        # 获取除j以外的其他参数
        args = self.args[2:]
        
        # 如果j是数值型
        if j.is_number:
            # 如果self是CoupledSpinState的实例
            if isinstance(self, CoupledSpinState):
                # 如果j是整数
                if j == int(j):
                    start = j**2
                else:
                    start = (2*j - 1)*(2*j + 1)/4
            else:
                start = 0
            
            # 将self表示为指定基础的向量
            vect = represent(self, basis=basis, **options)
            # 构造结果，是各个分量的加和
            result = Add(
                *[vect[start + i]*evect(j, j - i, *args) for i in range(2*j + 1)])
            
            # 如果self是CoupledSpinState的实例并且选项中的'coupled'是False
            if isinstance(self, CoupledSpinState) and options.get('coupled') is False:
                # 解耦结果
                return uncouple(result)
            # 返回结果
            return result
        else:
            # 初始化计数器i
            i = 0
            # 创建mi符号变量
            mi = symbols('mi')
            # 确保不引入状态中已有的符号
            while self.subs(mi, 0) != self:
                i += 1
                mi = symbols('mi%d' % i)
                break
            
            # TODO: 更好地获取旋转角度的方法
            # 如果self是CoupledSpinState的实例
            if isinstance(self, CoupledSpinState):
                test_args = (0, mi, (0, 0))
            else:
                test_args = (0, mi)
            
            # 如果self是Ket的实例
            if isinstance(self, Ket):
                angles = represent(
                    self.__class__(*test_args), basis=basis)[0].args[3:6]
            else:
                angles = represent(self.__class__(
                    *test_args), basis=basis)[0].args[0].args[3:6]
            
            # 如果角度为(0, 0, 0)，返回self
            if angles == (0, 0, 0):
                return self
            else:
                # 获取态
                state = evect(j, mi, *args)
                # 构造旋转算子
                lt = Rotation.D(j, mi, self.m, *angles)
                # 返回态的和式
                return Sum(lt*state, (mi, -j, j))

    # 对Jx的Bra求内积
    def _eval_innerproduct_JxBra(self, bra, **hints):
        result = KroneckerDelta(self.j, bra.j)
        # 如果bra的对偶类不是self的类
        if bra.dual_class() is not self.__class__:
            result *= self._represent_JxOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(
                self.j, bra.j)*KroneckerDelta(self.m, bra.m)
        return result

    # 对Jy的Bra求内积
    def _eval_innerproduct_JyBra(self, bra, **hints):
        result = KroneckerDelta(self.j, bra.j)
        # 如果bra的对偶类不是self的类
        if bra.dual_class() is not self.__class__:
            result *= self._represent_JyOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(
                self.j, bra.j)*KroneckerDelta(self.m, bra.m)
        return result

    # 对Jz的Bra求内积
    def _eval_innerproduct_JzBra(self, bra, **hints):
        result = KroneckerDelta(self.j, bra.j)
        # 如果bra的对偶类不是self的类
        if bra.dual_class() is not self.__class__:
            result *= self._represent_JzOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(
                self.j, bra.j)*KroneckerDelta(self.m, bra.m)
        return result
    # 定义一个方法 `_eval_trace`，它接受参数 `bra` 和其他可能的提示信息 (`hints`)
    def _eval_trace(self, bra, **hints):

        # 一种实现方法是假设基底集合 `k` 被传递进来。
        # 然后我们可以在这里应用迹的离散形式公式
        # Tr(|i><j| ) = \Sum_k <k|i><j|k>
        # 然后对每个内积应用 `qapply()`，并对它们求和。

        # 或者

        # |i><j| 的内积 = 迹(外积)。
        # 我们可以直接使用这个，除非有情况下这不成立

        # 返回 `bra` 与 `self` 的乘积并执行操作
        return (bra*self).doit()
class JxKet(SpinState, Ket):
    """Eigenket of Jx.

    See JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        # 返回JxBra作为对偶类（dual class）
        return JxBra

    @classmethod
    def coupled_class(self):
        # 返回JxKetCoupled作为耦合类（coupled class）
        return JxKetCoupled

    def _represent_default_basis(self, **options):
        # 返回默认基础下的Jx算符的表示
        return self._represent_JxOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        # 返回基于Jx算符的表示
        return self._represent_base(**options)

    def _represent_JyOp(self, basis, **options):
        # 返回基于Jy算符的表示，设置了alpha参数为3π/2
        return self._represent_base(alpha=pi*Rational(3, 2), **options)

    def _represent_JzOp(self, basis, **options):
        # 返回基于Jz算符的表示，设置了beta参数为π/2
        return self._represent_base(beta=pi/2, **options)


class JxBra(SpinState, Bra):
    """Eigenbra of Jx.

    See JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        # 返回JxKet作为对偶类（dual class）
        return JxKet

    @classmethod
    def coupled_class(self):
        # 返回JxBraCoupled作为耦合类（coupled class）
        return JxBraCoupled


class JyKet(SpinState, Ket):
    """Eigenket of Jy.

    See JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        # 返回JyBra作为对偶类（dual class）
        return JyBra

    @classmethod
    def coupled_class(self):
        # 返回JyKetCoupled作为耦合类（coupled class）
        return JyKetCoupled

    def _represent_default_basis(self, **options):
        # 返回默认基础下的Jy算符的表示
        return self._represent_JyOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        # 返回基于Jx算符的表示，设置了gamma参数为π/2
        return self._represent_base(gamma=pi/2, **options)

    def _represent_JyOp(self, basis, **options):
        # 返回基于Jy算符的表示
        return self._represent_base(**options)

    def _represent_JzOp(self, basis, **options):
        # 返回基于Jz算符的表示，设置了alpha参数为3π/2，beta参数为-π/2，gamma参数为π/2
        return self._represent_base(alpha=pi*Rational(3, 2), beta=-pi/2, gamma=pi/2, **options)


class JyBra(SpinState, Bra):
    """Eigenbra of Jy.

    See JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        # 返回JyKet作为对偶类（dual class）
        return JyKet

    @classmethod
    def coupled_class(self):
        # 返回JyBraCoupled作为耦合类（coupled class）
        return JyBraCoupled


class JzKet(SpinState, Ket):
    """Eigenket of Jz.

    Spin state which is an eigenstate of the Jz operator. Uncoupled states,
    that is states representing the interaction of multiple separate spin
    states, are defined as a tensor product of states.

    Parameters
    ==========

    j : Number, Symbol
        Total spin angular momentum
    m : Number, Symbol
        Eigenvalue of the Jz spin operator

    Examples
    ========

    *Normal States:*

    Defining simple spin states, both numerical and symbolic:

        >>> from sympy.physics.quantum.spin import JzKet, JxKet
        >>> from sympy import symbols
        >>> JzKet(1, 0)
        |1,0>
        >>> j, m = symbols('j m')
        >>> JzKet(j, m)
        |j,m>

    Rewriting the JzKet in terms of eigenkets of the Jx operator:

    """
    """

    @classmethod
    def dual_class(self):
        # 返回 JzBra 类作为对偶类
        return JzBra

    @classmethod
    def coupled_class(self):
        # 返回 JzKetCoupled 类作为耦合态类
        return JzKetCoupled

    def _represent_default_basis(self, **options):
        # 默认情况下使用 Jz 算符表示态的方法
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        # 使用 Jx 算符表示态的方法，基于 _represent_base 方法
        return self._represent_base(beta=pi*Rational(3, 2), **options)

    def _represent_JyOp(self, basis, **options):
        # 使用 Jy 算符表示态的方法，基于 _represent_base 方法，指定了特定角度
        return self._represent_base(alpha=pi*Rational(3, 2), beta=pi/2, gamma=pi/2, **options)

    def _represent_JzOp(self, basis, **options):
        # 使用 Jz 算符表示态的方法，基于 _represent_base 方法
        return self._represent_base(**options)
class JzBra(SpinState, Bra):
    """Eigenbra of Jz.

    See the JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        # 返回 JzKet 类作为该类的对偶类
        return JzKet

    @classmethod
    def coupled_class(self):
        # 返回 JzBraCoupled 类作为该类的耦合类
        return JzBraCoupled


# Method used primarily to create coupled_n and coupled_jn by __new__ in
# CoupledSpinState
# This same method is also used by the uncouple method, and is separated from
# the CoupledSpinState class to maintain consistency in defining coupling
def _build_coupled(jcoupling, length):
    # 创建一个由长度为 length 的单元素列表组成的 n_list
    n_list = [[n + 1] for n in range(length)]
    coupled_jn = []
    coupled_n = []
    for n1, n2, j_new in jcoupling:
        # 将 jcoupling 中的 j_new 添加到 coupled_jn 列表中
        coupled_jn.append(j_new)
        # 将 (n1, n2) 对应的 n_list 元素添加到 coupled_n 列表中
        coupled_n.append((n_list[n1 - 1], n_list[n2 - 1]))
        # 对 n_list 中的元素进行排序，并将排序后的结果更新到原来的位置
        n_sort = sorted(n_list[n1 - 1] + n_list[n2 - 1])
        n_list[n_sort[0] - 1] = n_sort
    # 返回耦合后的 n 列表和 j 列表
    return coupled_n, coupled_jn


class CoupledSpinState(SpinState):
    """Base class for coupled angular momentum states."""

    def _print_label(self, printer, *args):
        # 创建标签列表，包含主量子数 j 和磁量子数 m
        label = [printer._print(self.j), printer._print(self.m)]
        # 添加每个 jn 到标签列表中
        for i, ji in enumerate(self.jn, start=1):
            label.append('j%d=%s' % (
                i, printer._print(ji)
            ))
        # 添加每个耦合 jn 和对应的 (n1, n2) 到标签列表中，除了最后一个
        for jn, (n1, n2) in zip(self.coupled_jn[:-1], self.coupled_n[:-1]):
            label.append('j(%s)=%s' % (
                ','.join(str(i) for i in sorted(n1 + n2)), printer._print(jn)
            ))
        # 返回以逗号分隔的标签列表
        return ','.join(label)

    def _print_label_pretty(self, printer, *args):
        # 创建标签列表，包含主量子数 j 和磁量子数 m
        label = [self.j, self.m]
        # 添加每个 jn 到标签列表中
        for i, ji in enumerate(self.jn, start=1):
            symb = 'j%d' % i
            symb = pretty_symbol(symb)
            symb = prettyForm(symb + '=')
            item = prettyForm(*symb.right(printer._print(ji)))
            label.append(item)
        # 添加每个耦合 jn 和对应的 (n1, n2) 到标签列表中，除了最后一个
        for jn, (n1, n2) in zip(self.coupled_jn[:-1], self.coupled_n[:-1]):
            n = ','.join(pretty_symbol("j%d" % i)[-1] for i in sorted(n1 + n2))
            symb = prettyForm('j' + n + '=')
            item = prettyForm(*symb.right(printer._print(jn)))
            label.append(item)
        # 返回美化后的标签列表
        return self._print_sequence_pretty(
            label, self._label_separator, printer, *args
        )

    def _print_label_latex(self, printer, *args):
        # 创建标签列表，包含主量子数 j 和磁量子数 m
        label = [
            printer._print(self.j, *args),
            printer._print(self.m, *args)
        ]
        # 添加每个 jn 到标签列表中
        for i, ji in enumerate(self.jn, start=1):
            label.append('j_{%d}=%s' % (i, printer._print(ji, *args)))
        # 添加每个耦合 jn 和对应的 (n1, n2) 到标签列表中，除了最后一个
        for jn, (n1, n2) in zip(self.coupled_jn[:-1], self.coupled_n[:-1]):
            n = ','.join(str(i) for i in sorted(n1 + n2))
            label.append('j_{%s}=%s' % (n, printer._print(jn, *args)))
        # 返回以指定分隔符连接的 LaTeX 格式标签列表
        return self._label_separator.join(label)

    @property
    def jn(self):
        # 返回标签的第三个元素，表示 jn 列表
        return self.label[2]

    @property
    def coupling(self):
        # 返回标签的第四个元素，表示耦合
        return self.label[3]

    @property
    # 继续填充剩余的代码注释
    # 返回与标签索引3对应的值作为参数调用 _build_coupled 函数，并返回其结果的第二个元素
    def coupled_jn(self):
        return _build_coupled(self.label[3], len(self.label[2]))[1]

    # 返回与标签索引3对应的值作为参数调用 _build_coupled 函数，并返回其结果的第一个元素
    @property
    def coupled_n(self):
        return _build_coupled(self.label[3], len(self.label[2]))[0]

    # 类方法，根据给定的标签计算 Hilbert 空间
    @classmethod
    def _eval_hilbert_space(cls, label):
        # 将标签中索引为2的所有元素求和
        j = Add(*label[2])
        # 如果 j 是数值类型
        if j.is_number:
            # 返回一个直和 Hilbert 空间，包含多个复数空间，范围从 2*j+1 到 1，步长为 -2
            return DirectSumHilbertSpace(*[ ComplexSpace(x) for x in range(int(2*j + 1), 0, -2) ])
        else:
            # 如果 j 不是数值类型，返回一个 ComplexSpace，其维度为 2*j + 1
            # TODO: 需要修复 Hilbert 空间，参见问题 5732
            # 期望的行为：
            # ji = symbols('ji')
            # ret = Sum(ComplexSpace(2*ji + 1), (ji, 0, j))
            # 临时修复：
            return ComplexSpace(2*j + 1)

    # 根据选项表示耦合的基底
    def _represent_coupled_base(self, **options):
        # 调用未耦合类的方法，获取其返回值
        evect = self.uncoupled_class()
        # 如果 self.j 不是数值类型，引发 ValueError 异常
        if not self.j.is_number:
            raise ValueError(
                'State must not have symbolic j value to represent')
        # 如果 self.hilbert_space.dimension 不是数值类型，引发 ValueError 异常
        if not self.hilbert_space.dimension.is_number:
            raise ValueError(
                'State must not have symbolic j values to represent')
        # 创建一个指定维度的零矩阵
        result = zeros(self.hilbert_space.dimension, 1)
        # 根据 self.j 的值确定起始位置
        if self.j == int(self.j):
            start = self.j**2
        else:
            start = (2*self.j - 1)*(1 + 2*self.j)/4
        # 将 evect(self.j, self.m)._represent_base(**options) 的结果复制到 result 的指定位置
        result[start:start + 2*self.j + 1, 0] = evect(
            self.j, self.m)._represent_base(**options)
        return result

    # 将对象重写为 Jx 的表示形式
    def _eval_rewrite_as_Jx(self, *args, **options):
        # 如果对象是 Bra 类的实例，调用 _rewrite_basis 方法使用 Jx 和 JxBraCoupled 作为参数
        if isinstance(self, Bra):
            return self._rewrite_basis(Jx, JxBraCoupled, **options)
        # 否则，调用 _rewrite_basis 方法使用 Jx 和 JxKetCoupled 作为参数
        return self._rewrite_basis(Jx, JxKetCoupled, **options)

    # 将对象重写为 Jy 的表示形式
    def _eval_rewrite_as_Jy(self, *args, **options):
        # 如果对象是 Bra 类的实例，调用 _rewrite_basis 方法使用 Jy 和 JyBraCoupled 作为参数
        if isinstance(self, Bra):
            return self._rewrite_basis(Jy, JyBraCoupled, **options)
        # 否则，调用 _rewrite_basis 方法使用 Jy 和 JyKetCoupled 作为参数
        return self._rewrite_basis(Jy, JyKetCoupled, **options)

    # 将对象重写为 Jz 的表示形式
    def _eval_rewrite_as_Jz(self, *args, **options):
        # 如果对象是 Bra 类的实例，调用 _rewrite_basis 方法使用 Jz 和 JzBraCoupled 作为参数
        if isinstance(self, Bra):
            return self._rewrite_basis(Jz, JzBraCoupled, **options)
        # 否则，调用 _rewrite_basis 方法使用 Jz 和 JzKetCoupled 作为参数
        return self._rewrite_basis(Jz, JzKetCoupled, **options)
class JxKetCoupled(CoupledSpinState, Ket):
    """Coupled eigenket of Jx.

    See JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        # 返回 JxBraCoupled 类作为对偶类
        return JxBraCoupled

    @classmethod
    def uncoupled_class(self):
        # 返回 JxKet 类作为未耦合类
        return JxKet

    def _represent_default_basis(self, **options):
        # 用于在默认基础上表示对象，使用 _represent_JzOp 方法
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        # 用于在 Jx 操作符的基础上表示对象，使用 _represent_coupled_base 方法
        return self._represent_coupled_base(**options)

    def _represent_JyOp(self, basis, **options):
        # 用于在 Jy 操作符的基础上表示对象，使用 _represent_coupled_base 方法
        return self._represent_coupled_base(alpha=pi*Rational(3, 2), **options)

    def _represent_JzOp(self, basis, **options):
        # 用于在 Jz 操作符的基础上表示对象，使用 _represent_coupled_base 方法
        return self._represent_coupled_base(beta=pi/2, **options)


class JxBraCoupled(CoupledSpinState, Bra):
    """Coupled eigenbra of Jx.

    See JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        # 返回 JxKetCoupled 类作为对偶类
        return JxKetCoupled

    @classmethod
    def uncoupled_class(self):
        # 返回 JxBra 类作为未耦合类
        return JxBra


class JyKetCoupled(CoupledSpinState, Ket):
    """Coupled eigenket of Jy.

    See JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        # 返回 JyBraCoupled 类作为对偶类
        return JyBraCoupled

    @classmethod
    def uncoupled_class(self):
        # 返回 JyKet 类作为未耦合类
        return JyKet

    def _represent_default_basis(self, **options):
        # 用于在默认基础上表示对象，使用 _represent_JzOp 方法
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        # 用于在 Jx 操作符的基础上表示对象，使用 _represent_coupled_base 方法
        return self._represent_coupled_base(gamma=pi/2, **options)

    def _represent_JyOp(self, basis, **options):
        # 用于在 Jy 操作符的基础上表示对象，使用 _represent_coupled_base 方法
        return self._represent_coupled_base(**options)

    def _represent_JzOp(self, basis, **options):
        # 用于在 Jz 操作符的基础上表示对象，使用 _represent_coupled_base 方法
        return self._represent_coupled_base(alpha=pi*Rational(3, 2), beta=-pi/2, gamma=pi/2, **options)


class JyBraCoupled(CoupledSpinState, Bra):
    """Coupled eigenbra of Jy.

    See JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        # 返回 JyKetCoupled 类作为对偶类
        return JyKetCoupled

    @classmethod
    def uncoupled_class(self):
        # 返回 JyBra 类作为未耦合类
        return JyBra


class JzKetCoupled(CoupledSpinState, Ket):
    r"""Coupled eigenket of Jz

    Spin state that is an eigenket of Jz which represents the coupling of
    separate spin spaces.

    The arguments for creating instances of JzKetCoupled are ``j``, ``m``,
    ``jn`` and an optional ``jcoupling`` argument. The ``j`` and ``m`` options
    are the total angular momentum quantum numbers, as used for normal states
    (e.g. JzKet).

    The other required parameter in ``jn``, which is a tuple defining the `j_n`
    angular momentum quantum numbers of the product spaces. So for example, if

    """
    
    @classmethod
    def dual_class(self):
        # 返回 JzBraCoupled 类作为对偶类
        return JzBraCoupled

    @classmethod
    def uncoupled_class(self):
        # 返回 JzKet 类作为未耦合类
        return JzKet

    def _represent_default_basis(self, **options):
        # 用于在默认基础上表示对象，使用 _represent_JzOp 方法
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        # 用于在 Jx 操作符的基础上表示对象，使用 _represent_coupled_base 方法
        return self._represent_coupled_base(alpha=pi/2, **options)

    def _represent_JyOp(self, basis, **options):
        # 用于在 Jy 操作符的基础上表示对象，使用 _represent_coupled_base 方法
        return self._represent_coupled_base(alpha=pi*Rational(3, 2), **options)

    def _represent_JzOp(self, basis, **options):
        # 用于在 Jz 操作符的基础上表示对象，使用 _represent_coupled_base 方法
        return self._represent_coupled_base(**options)
    # a state represented the coupling of the product basis state
    # `\left|j_1,m_1\right\rangle\times\left|j_2,m_2\right\rangle`, the ``jn``
    # for this state would be ``(j1,j2)``.
    
    # The final option is ``jcoupling``, which is used to define how the spaces
    # specified by ``jn`` are coupled, which includes both the order these spaces
    # are coupled together and the quantum numbers that arise from these
    # couplings. The ``jcoupling`` parameter itself is a list of lists, such that
    # each of the sublists defines a single coupling between the spin spaces. If
    # there are N coupled angular momentum spaces, that is ``jn`` has N elements,
    # then there must be N-1 sublists. Each of these sublists making up the
    # ``jcoupling`` parameter have length 3. The first two elements are the
    # indices of the product spaces that are considered to be coupled together.
    # For example, if we want to couple `j_1` and `j_4`, the indices would be 1
    # and 4. If a state has already been coupled, it is referenced by the
    # smallest index that is coupled, so if `j_2` and `j_4` has already been
    # coupled to some `j_{24}`, then this value can be coupled by referencing it
    # with index 2. The final element of the sublist is the quantum number of the
    # coupled state. So putting everything together, into a valid sublist for
    # ``jcoupling``, if `j_1` and `j_2` are coupled to an angular momentum space
    # with quantum number `j_{12}` with the value ``j12``, the sublist would be
    # ``(1,2,j12)``, N-1 of these sublists are used in the list for
    # ``jcoupling``.
    
    # Note the ``jcoupling`` parameter is optional, if it is not specified, the
    # default coupling is taken. This default value is to coupled the spaces in
    # order and take the quantum number of the coupling to be the maximum value.
    # For example, if the spin spaces are `j_1`, `j_2`, `j_3`, `j_4`, then the
    # default coupling couples `j_1` and `j_2` to `j_{12}=j_1+j_2`, then,
    # `j_{12}` and `j_3` are coupled to `j_{123}=j_{12}+j_3`, and finally
    # `j_{123}` and `j_4` to `j=j_{123}+j_4`. The jcoupling value that would
    # correspond to this is:
    # 
    #     ``((1,2,j1+j2),(1,3,j1+j2+j3))``
    
    # Parameters
    # ==========
    
    # args : tuple
    #     The arguments that must be passed are ``j``, ``m``, ``jn``, and
    #     ``jcoupling``. The ``j`` value is the total angular momentum. The ``m``
    #     value is the eigenvalue of the Jz spin operator. The ``jn`` list are
    #     the j values of angular momentum spaces coupled together. The
    #     ``jcoupling`` parameter is an optional parameter defining how the spaces
    #     are coupled together. See the above description for how these coupling
    #     parameters are defined.
    
    # Examples
    # ========
    @classmethod
    def dual_class(self):
        # 返回 JzBraCoupled 类，用于耦合态的另一半
        return JzBraCoupled

    @classmethod
    def uncoupled_class(self):
        # 返回 JzKet 类，用于未耦合的态
        return JzKet

    def _represent_default_basis(self, **options):
        # 返回使用默认基础的表示结果，这里调用 _represent_JzOp 方法
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        # 返回使用 Jx 算符的表示结果，调用 _represent_coupled_base 方法，设置 beta 参数为 3π/2
        return self._represent_coupled_base(beta=pi*Rational(3, 2), **options)

    def _represent_JyOp(self, basis, **options):
        # 返回使用 Jy 算符的表示结果，调用 _represent_coupled_base 方法，设置 alpha 和 beta 参数
        return self._represent_coupled_base(alpha=pi*Rational(3, 2), beta=pi/2, gamma=pi/2, **options)

    def _represent_JzOp(self, basis, **options):
        # 返回使用 Jz 算符的表示结果，调用 _represent_coupled_base 方法
        return self._represent_coupled_base(**options)
class JzBraCoupled(CoupledSpinState, Bra):
    """Coupled eigenbra of Jz.

    See the JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        # 返回 JzKetCoupled 类作为对偶类
        return JzKetCoupled

    @classmethod
    def uncoupled_class(self):
        # 返回 JzBra 类作为未耦合类
        return JzBra

#-----------------------------------------------------------------------------
# Coupling/uncoupling
#-----------------------------------------------------------------------------


def couple(expr, jcoupling_list=None):
    """ Couple a tensor product of spin states

    This function can be used to couple an uncoupled tensor product of spin
    states. All of the eigenstates to be coupled must be of the same class. It
    will return a linear combination of eigenstates that are subclasses of
    CoupledSpinState determined by Clebsch-Gordan angular momentum coupling
    coefficients.

    Parameters
    ==========

    expr : Expr
        An expression involving TensorProducts of spin states to be coupled.
        Each state must be a subclass of SpinState and they all must be the
        same class.

    jcoupling_list : list or tuple
        Elements of this list are sub-lists of length 2 specifying the order of
        the coupling of the spin spaces. The length of this must be N-1, where N
        is the number of states in the tensor product to be coupled. The
        elements of this sublist are the same as the first two elements of each
        sublist in the ``jcoupling`` parameter defined for JzKetCoupled. If this
        parameter is not specified, the default value is taken, which couples
        the first and second product basis spaces, then couples this new coupled
        space to the third product space, etc

    Examples
    ========

    Couple a tensor product of numerical states for two spaces:

        >>> from sympy.physics.quantum.spin import JzKet, couple
        >>> from sympy.physics.quantum.tensorproduct import TensorProduct
        >>> couple(TensorProduct(JzKet(1,0), JzKet(1,1)))
        -sqrt(2)*|1,1,j1=1,j2=1>/2 + sqrt(2)*|2,1,j1=1,j2=1>/2


    Numerical coupling of three spaces using the default coupling method, i.e.
    first and second spaces couple, then this couples to the third space:

        >>> couple(TensorProduct(JzKet(1,1), JzKet(1,1), JzKet(1,0)))
        sqrt(6)*|2,2,j1=1,j2=1,j3=1,j(1,2)=2>/3 + sqrt(3)*|3,2,j1=1,j2=1,j3=1,j(1,2)=2>/3

    Perform this same coupling, but we define the coupling to first couple
    the first and third spaces:

        >>> couple(TensorProduct(JzKet(1,1), JzKet(1,1), JzKet(1,0)), ((1,3),(1,2)) )
        sqrt(2)*|2,2,j1=1,j2=1,j3=1,j(1,3)=1>/2 - sqrt(6)*|2,2,j1=1,j2=1,j3=1,j(1,3)=2>/6 + sqrt(3)*|3,2,j1=1,j2=1,j3=1,j(1,3)=2>/3

    """
    Couple a tensor product of symbolic states:

    >>> from sympy import symbols
    >>> j1,m1,j2,m2 = symbols('j1 m1 j2 m2')
    >>> couple(TensorProduct(JzKet(j1,m1), JzKet(j2,m2)))
    Sum(CG(j1, m1, j2, m2, j, m1 + m2)*|j,m1 + m2,j1=j1,j2=j2>, (j, m1 + m2, j1 + j2))

"""
a = expr.atoms(TensorProduct)  # 获取表达式中所有的 TensorProduct 对象
for tp in a:  # 遍历每一个 TensorProduct 对象
    # 允许表达式中存在其他的 TensorProduct 对象
    if not all(isinstance(state, SpinState) for state in tp.args):
        continue
    # 如果所有的状态都是 SpinState 类型，则引发错误，因为只允许相同的基态张量积
    if not all(state.__class__ is tp.args[0].__class__ for state in tp.args):
        raise TypeError('All states must be the same basis')
    # 使用 _couple 函数对当前的 TensorProduct 对象进行耦合操作，并替换原表达式中的该对象
    expr = expr.subs(tp, _couple(tp, jcoupling_list))
return expr
def _couple(tp, jcoupling_list):
    # 从参数tp中获取态(states)列表
    states = tp.args
    # 从第一个态创建耦合的态(coupled_evect)
    coupled_evect = states[0].coupled_class()

    # 如果未指定jcoupling_list，则定义默认的耦合关系为空列表
    if jcoupling_list is None:
        jcoupling_list = []
        # 自动生成耦合关系，从第二个态开始依次与前一个态耦合
        for n in range(1, len(states)):
            jcoupling_list.append( (1, n + 1) )

    # 检查jcoupling_list的有效性
    if not len(jcoupling_list) == len(states) - 1:
        # 若jcoupling_list长度不符合要求，则引发TypeError异常
        raise TypeError('jcoupling_list must be length %d, got %d' %
                        (len(states) - 1, len(jcoupling_list)))
    if not all( len(coupling) == 2 for coupling in jcoupling_list):
        # 若任何耦合关系不是由两个元素组成，则引发ValueError异常
        raise ValueError('Each coupling must define 2 spaces')
    if any(n1 == n2 for n1, n2 in jcoupling_list):
        # 若存在某些自身耦合的情况，则引发ValueError异常
        raise ValueError('Spin spaces cannot couple to themselves')
    if all(sympify(n1).is_number and sympify(n2).is_number for n1, n2 in jcoupling_list):
        # 如果所有耦合关系的元素都是数字，则进行额外的检查
        j_test = [0]*len(states)
        for n1, n2 in jcoupling_list:
            # 检查引用最小n值的耦合关系是否正确
            if j_test[n1 - 1] == -1 or j_test[n2 - 1] == -1:
                raise ValueError('Spaces coupling j_n\'s are referenced by smallest n value')
            j_test[max(n1, n2) - 1] = -1

    # 获取每个态的角动量值(j values)和磁量子数(m values)
    jn = [state.j for state in states]
    mn = [state.m for state in states]

    # 创建coupling_list，定义所有态之间的耦合关系
    coupling_list = []
    n_list = [ [i + 1] for i in range(len(states)) ]
    for j_coupling in jcoupling_list:
        # 获取耦合关系中的第一个和第二个态的最小n值
        n1, n2 = j_coupling
        # 获取第一个和第二个态耦合的所有n值列表
        j1_n = list(n_list[n1 - 1])
        j2_n = list(n_list[n2 - 1])
        # 将耦合关系添加到coupling_list中
        coupling_list.append( (j1_n, j2_n) )
        # 更新最小n值对应的新的j_n值，将第一个和第二个态的所有n值排序后合并
        n_list[ min(n1, n2) - 1 ] = sorted(j1_n + j2_n)
    # 检查所有状态对象的属性 j 和 m 是否都是数字类型
    if all(state.j.is_number and state.m.is_number for state in states):
        # 数值耦合
        # 遍历耦合列表中每对耦合，计算最大可能 j 值之间的差值
        diff_max = [ Add( *[ jn[n - 1] - mn[n - 1] for n in coupling[0] +
                         coupling[1] ] ) for coupling in coupling_list ]
        # 初始化结果列表
        result = []
        # 遍历差值范围内的每一个值
        for diff in range(diff_max[-1] + 1):
            # 确定可用的配置数
            n = len(coupling_list)
            tot = binomial(diff + n - 1, diff)

            # 遍历每个配置编号
            for config_num in range(tot):
                # 将配置编号转换为差值列表
                diff_list = _confignum_to_difflist(config_num, diff, n)

                # 跳过非物理配置
                # 这是对 diff_max 松散限制的一种懒惰检查
                if any(d > m for d, m in zip(diff_list, diff_max)):
                    continue

                # 确定术语
                cg_terms = []
                coupled_j = list(jn)
                jcoupling = []
                # 遍历耦合列表和差值列表，计算耦合后的 j 和 m 值
                for (j1_n, j2_n), coupling_diff in zip(coupling_list, diff_list):
                    j1 = coupled_j[ min(j1_n) - 1 ]
                    j2 = coupled_j[ min(j2_n) - 1 ]
                    j3 = j1 + j2 - coupling_diff
                    coupled_j[ min(j1_n + j2_n) - 1 ] = j3
                    m1 = Add( *[ mn[x - 1] for x in j1_n] )
                    m2 = Add( *[ mn[x - 1] for x in j2_n] )
                    m3 = m1 + m2
                    cg_terms.append( (j1, m1, j2, m2, j3, m3) )
                    jcoupling.append( (min(j1_n), min(j2_n), j3) )

                # 更好的检查状态是否物理上的
                if any(abs(term[5]) > term[4] for term in cg_terms):
                    continue
                if any(term[0] + term[2] < term[4] for term in cg_terms):
                    continue
                if any(abs(term[0] - term[2]) > term[4] for term in cg_terms):
                    continue

                # 计算系数
                coeff = Mul( *[ CG(*term).doit() for term in cg_terms] )
                # 计算耦合后的状态
                state = coupled_evect(j3, m3, jn, jcoupling)
                # 将结果添加到最终结果列表中
                result.append(coeff * state)

        # 返回所有结果的求和
        return Add(*result)
    else:
        # 如果不是直接耦合，则为符号耦合
        cg_terms = []  # 初始化耦合项列表
        jcoupling = []  # 初始化耦合关系列表
        sum_terms = []  # 初始化求和项列表
        coupled_j = list(jn)  # 复制原始角动量列表
        # 遍历耦合列表中的每对耦合
        for j1_n, j2_n in coupling_list:
            # 根据耦合列表中的角动量编号获取实际角动量
            j1 = coupled_j[min(j1_n) - 1]
            j2 = coupled_j[min(j2_n) - 1]
            # 根据状态列表的长度判断是否是符号耦合
            if len(j1_n + j2_n) == len(states):
                j3 = symbols('j')  # 创建新的符号角动量
            else:
                j3_name = 'j' + ''.join(["%s" % n for n in j1_n + j2_n])
                j3 = symbols(j3_name)  # 根据角动量编号创建新的符号角动量
            coupled_j[min(j1_n + j2_n) - 1] = j3  # 更新角动量列表
            # 计算总角动量和
            m1 = Add(*[mn[x - 1] for x in j1_n])
            m2 = Add(*[mn[x - 1] for x in j2_n])
            m3 = m1 + m2
            # 将角动量耦合项添加到列表中
            cg_terms.append((j1, m1, j2, m2, j3, m3))
            # 将耦合关系添加到列表中
            jcoupling.append((min(j1_n), min(j2_n), j3))
            # 将求和项添加到列表中
            sum_terms.append((j3, m3, j1 + j2))
        # 计算耦合系数
        coeff = Mul(*[CG(*term) for term in cg_terms])
        # 计算耦合状态
        state = coupled_evect(j3, m3, jn, jcoupling)
        # 返回总和表达式
        return Sum(coeff * state, *sum_terms)
# 定义函数 `uncouple`，用于解耦合的自旋状态

""" Uncouple a coupled spin state

Gives the uncoupled representation of a coupled spin state. Arguments must
be either a spin state that is a subclass of CoupledSpinState or a spin
state that is a subclass of SpinState and an array giving the j values
of the spaces that are to be coupled

Parameters
==========

expr : Expr
    The expression containing states that are to be coupled. If the states
    are a subclass of SpinState, the ``jn`` and ``jcoupling`` parameters
    must be defined. If the states are a subclass of CoupledSpinState,
    ``jn`` and ``jcoupling`` will be taken from the state.

jn : list or tuple
    The list of the j-values that are coupled. If state is a
    CoupledSpinState, this parameter is ignored. This must be defined if
    state is not a subclass of CoupledSpinState. The syntax of this
    parameter is the same as the ``jn`` parameter of JzKetCoupled.

jcoupling_list : list or tuple
    The list defining how the j-values are coupled together. If state is a
    CoupledSpinState, this parameter is ignored. This must be defined if
    state is not a subclass of CoupledSpinState. The syntax of this
    parameter is the same as the ``jcoupling`` parameter of JzKetCoupled.

Examples
========

Uncouple a numerical state using a CoupledSpinState state:

    >>> from sympy.physics.quantum.spin import JzKetCoupled, uncouple
    >>> from sympy import S
    >>> uncouple(JzKetCoupled(1, 0, (S(1)/2, S(1)/2)))
    sqrt(2)*|1/2,-1/2>x|1/2,1/2>/2 + sqrt(2)*|1/2,1/2>x|1/2,-1/2>/2

Perform the same calculation using a SpinState state:

    >>> from sympy.physics.quantum.spin import JzKet
    >>> uncouple(JzKet(1, 0), (S(1)/2, S(1)/2))
    sqrt(2)*|1/2,-1/2>x|1/2,1/2>/2 + sqrt(2)*|1/2,1/2>x|1/2,-1/2>/2

Uncouple a numerical state of three coupled spaces using a CoupledSpinState state:

    >>> uncouple(JzKetCoupled(1, 1, (1, 1, 1), ((1,3,1),(1,2,1)) ))
    |1,-1>x|1,1>x|1,1>/2 - |1,0>x|1,0>x|1,1>/2 + |1,1>x|1,0>x|1,0>/2 - |1,1>x|1,1>x|1,-1>/2

Perform the same calculation using a SpinState state:

    >>> uncouple(JzKet(1, 1), (1, 1, 1), ((1,3,1),(1,2,1)) )
    |1,-1>x|1,1>x|1,1>/2 - |1,0>x|1,0>x|1,1>/2 + |1,1>x|1,0>x|1,0>/2 - |1,1>x|1,1>x|1,-1>/2

Uncouple a symbolic state using a CoupledSpinState state:

    >>> from sympy import symbols
    >>> j,m,j1,j2 = symbols('j m j1 j2')
    >>> uncouple(JzKetCoupled(j, m, (j1, j2)))
    Sum(CG(j1, m1, j2, m2, j, m)*|j1,m1>x|j2,m2>, (m1, -j1, j1), (m2, -j2, j2))

Perform the same calculation using a SpinState state

    >>> uncouple(JzKet(j, m), (j1, j2))
    Sum(CG(j1, m1, j2, m2, j, m)*|j1,m1>x|j2,m2>, (m1, -j1, j1), (m2, -j2, j2))

"""
# 提取所有的 SpinState 子类对象
a = expr.atoms(SpinState)
# 遍历每个 SpinState 对象
for state in a:
    # 使用 _uncouple 函数解耦当前的 SpinState 对象，并更新表达式 expr
    expr = expr.subs(state, _uncouple(state, jn, jcoupling_list))
    # 返回表达式的计算结果
    return expr
# 定义函数 _uncouple，用于解耦给定的状态和耦合列表
def _uncouple(state, jn, jcoupling_list):
    # 如果 state 是 CoupledSpinState 类的实例
    if isinstance(state, CoupledSpinState):
        # 从 state 中获取 jn、coupled_n 和 coupled_jn
        jn = state.jn
        coupled_n = state.coupled_n
        coupled_jn = state.coupled_jn
        # 使用 state 的 uncoupled_class 方法创建 evect
        evect = state.uncoupled_class()
    # 如果 state 是 SpinState 类的实例
    elif isinstance(state, SpinState):
        # 如果 jn 为 None，则抛出数值错误
        if jn is None:
            raise ValueError("Must specify j-values for coupled state")
        # 如果 jn 不是列表或元组，则抛出类型错误
        if not isinstance(jn, (list, tuple)):
            raise TypeError("jn must be list or tuple")
        # 如果 jcoupling_list 为 None，则使用默认值
        if jcoupling_list is None:
            jcoupling_list = []
            # 构建 jcoupling_list
            for i in range(1, len(jn)):
                jcoupling_list.append(
                    (1, 1 + i, Add(*[jn[j] for j in range(i + 1)])) )
        # 如果 jcoupling_list 不是列表或元组，则抛出类型错误
        if not isinstance(jcoupling_list, (list, tuple)):
            raise TypeError("jcoupling must be a list or tuple")
        # 如果 jcoupling_list 的长度不等于 jn 的长度减去 1，则抛出数值错误
        if not len(jcoupling_list) == len(jn) - 1:
            raise ValueError("Must specify 2 fewer coupling terms than the number of j values")
        # 调用 _build_coupled 函数构建 coupled_n 和 coupled_jn
        coupled_n, coupled_jn = _build_coupled(jcoupling_list, len(jn))
        # 使用 state 的 __class__ 方法创建 evect
        evect = state.__class__
    else:
        # 如果 state 类型既不是 CoupledSpinState 也不是 SpinState，则抛出类型错误
        raise TypeError("state must be a spin state")

    # 从 state 中获取 j 和 m
    j = state.j
    m = state.m
    # 初始化 coupling_list 和 j_list
    coupling_list = []
    j_list = list(jn)

    # 创建 coupling_list，定义所有空间之间的耦合
    for j3, (n1, n2) in zip(coupled_jn, coupled_n):
        # 获取耦合为第一和第二空间的 j 值
        j1 = j_list[n1[0] - 1]
        j2 = j_list[n2[0] - 1]
        # 构建 coupling_list
        coupling_list.append( (n1, n2, j1, j2, j3) )
        # 在 j_list 中设置新值
        j_list[min(n1 + n2) - 1] = j3

    # 如果 j 和 m 均为数值类型
    if j.is_number and m.is_number:
        # 计算 diff_max 和 diff
        diff_max = [ 2*x for x in jn ]
        diff = Add(*jn) - m

        n = len(jn)
        # 计算 tot
        tot = binomial(diff + n - 1, diff)

        result = []
        # 遍历 config_num 范围内的所有配置
        for config_num in range(tot):
            # 将 config_num 转换为 diff_list
            diff_list = _confignum_to_difflist(config_num, diff, n)
            # 如果任何 diff_list 中的元素大于对应的 diff_max，则跳过
            if any(d > p for d, p in zip(diff_list, diff_max)):
                continue

            cg_terms = []
            # 遍历 coupling_list 中的每对耦合
            for coupling in coupling_list:
                j1_n, j2_n, j1, j2, j3 = coupling
                # 计算 m1、m2 和 m3
                m1 = Add( *[ jn[x - 1] - diff_list[x - 1] for x in j1_n ] )
                m2 = Add( *[ jn[x - 1] - diff_list[x - 1] for x in j2_n ] )
                m3 = m1 + m2
                cg_terms.append( (j1, m1, j2, m2, j3, m3) )
            # 计算 coeff
            coeff = Mul( *[ CG(*term).doit() for term in cg_terms ] )
            # 计算 state
            state = TensorProduct(
                *[ evect(j, j - d) for j, d in zip(jn, diff_list) ] )
            result.append(coeff*state)
        # 返回结果的总和
        return Add(*result)
    else:
        # 如果没有进入前面的条件，执行符号耦合计算

        # 构造字符串，表示耦合系数的符号表达式
        m_str = "m1:%d" % (len(jn) + 1)
        # 根据符号表达式创建符号对象
        mvals = symbols(m_str)

        # 构造符号耦合项的列表，每个项由三个部分组成
        cg_terms = [
            (j1, Add(*[mvals[n - 1] for n in j1_n]),
             j2, Add(*[mvals[n - 1] for n in j2_n]),
             j3, Add(*[mvals[n - 1] for n in j1_n + j2_n]))
            for j1_n, j2_n, j1, j2, j3 in coupling_list[:-1]
        ]

        # 将最后一个耦合项添加到耦合项列表中
        cg_terms.append(*[
            (j1, Add(*[mvals[n - 1] for n in j1_n]),
             j2, Add(*[mvals[n - 1] for n in j2_n]),
             j, m)
            for j1_n, j2_n, j1, j2, j3 in [coupling_list[-1]]
        ])

        # 计算符号耦合系数，将每个耦合项转换为 CG 对象后进行乘积
        cg_coeff = Mul(*[CG(*cg_term) for cg_term in cg_terms])

        # 构造态的张量积，基于已知的角动量和符号耦合系数
        sum_terms = [(m, -j, j) for j, m in zip(jn, mvals)]
        state = TensorProduct(*[evect(j, m) for j, m in zip(jn, mvals)])

        # 返回最终的求和表达式，包括符号耦合系数乘以态的张量积
        return Sum(cg_coeff * state, *sum_terms)
# 根据给定的配置数字将差异（diff）分配到长度为list_len的槽位中
def _confignum_to_difflist(config_num, diff, list_len):
    # 初始化空列表用于存储计算得到的差异列表
    diff_list = []
    # 遍历槽位的范围
    for n in range(list_len):
        # 保存当前的diff值作为先前的差异
        prev_diff = diff
        # 计算当前槽位后面剩余的槽位数量
        rem_spots = list_len - n - 1
        # 计算在剩余槽位中分配diff的配置数量
        rem_configs = binomial(diff + rem_spots - 1, diff)
        # 当config_num大于等于当前剩余配置数量时，继续迭代以找到正确的差异分配方式
        while config_num >= rem_configs:
            config_num -= rem_configs
            diff -= 1
            rem_configs = binomial(diff + rem_spots - 1, diff)
        # 将当前槽位计算得到的差异值添加到差异列表中
        diff_list.append(prev_diff - diff)
    # 返回最终得到的差异列表
    return diff_list
```
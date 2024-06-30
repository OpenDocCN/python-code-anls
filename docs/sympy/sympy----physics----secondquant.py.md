# `D:\src\scipysrc\sympy\sympy\physics\secondquant.py`

```
"""
Second quantization operators and states for bosons.

This follow the formulation of Fetter and Welecka, "Quantum Theory
of Many-Particle Systems."
"""
from collections import defaultdict  # 导入 defaultdict 类用于创建默认字典

from sympy.core.add import Add  # 导入 Add 类
from sympy.core.basic import Basic  # 导入 Basic 类
from sympy.core.cache import cacheit  # 导入 cacheit 函数
from sympy.core.containers import Tuple  # 导入 Tuple 类
from sympy.core.expr import Expr  # 导入 Expr 类
from sympy.core.function import Function  # 导入 Function 类
from sympy.core.mul import Mul  # 导入 Mul 类
from sympy.core.numbers import I  # 导入 I 对象
from sympy.core.power import Pow  # 导入 Pow 类
from sympy.core.singleton import S  # 导入 S 对象
from sympy.core.sorting import default_sort_key  # 导入 default_sort_key 函数
from sympy.core.symbol import Dummy, Symbol  # 导入 Dummy 和 Symbol 类
from sympy.core.sympify import sympify  # 导入 sympify 函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入 sqrt 函数
from sympy.functions.special.tensor_functions import KroneckerDelta  # 导入 KroneckerDelta 函数
from sympy.matrices.dense import zeros  # 导入 zeros 函数
from sympy.printing.str import StrPrinter  # 导入 StrPrinter 类
from sympy.utilities.iterables import has_dups  # 导入 has_dups 函数

__all__ = [
    'Dagger',
    'KroneckerDelta',
    'BosonicOperator',
    'AnnihilateBoson',
    'CreateBoson',
    'AnnihilateFermion',
    'CreateFermion',
    'FockState',
    'FockStateBra',
    'FockStateKet',
    'FockStateBosonKet',
    'FockStateBosonBra',
    'FockStateFermionKet',
    'FockStateFermionBra',
    'BBra',
    'BKet',
    'FBra',
    'FKet',
    'F',
    'Fd',
    'B',
    'Bd',
    'apply_operators',
    'InnerProduct',
    'BosonicBasis',
    'VarBosonicBasis',
    'FixedBosonicBasis',
    'Commutator',
    'matrix_rep',
    'contraction',
    'wicks',
    'NO',
    'evaluate_deltas',
    'AntiSymmetricTensor',
    'substitute_dummies',
    'PermutationOperator',
    'simplify_index_permutations',
]

class SecondQuantizationError(Exception):
    pass


class AppliesOnlyToSymbolicIndex(SecondQuantizationError):
    pass


class ContractionAppliesOnlyToFermions(SecondQuantizationError):
    pass


class ViolationOfPauliPrinciple(SecondQuantizationError):
    pass


class SubstitutionOfAmbigousOperatorFailed(SecondQuantizationError):
    pass


class WicksTheoremDoesNotApply(SecondQuantizationError):
    pass


class Dagger(Expr):
    """
    Hermitian conjugate of creation/annihilation operators.

    Examples
    ========

    >>> from sympy import I
    >>> from sympy.physics.secondquant import Dagger, B, Bd
    >>> Dagger(2*I)
    -2*I
    >>> Dagger(B(0))
    CreateBoson(0)
    >>> Dagger(Bd(0))
    AnnihilateBoson(0)

    """

    def __new__(cls, arg):
        arg = sympify(arg)  # 将参数转换为 SymPy 表达式
        r = cls.eval(arg)  # 调用 eval 方法处理参数
        if isinstance(r, Basic):
            return r
        obj = Basic.__new__(cls, arg)  # 创建新的 Dagger 对象
        return obj

    @classmethod
    # 定义类方法 eval，用于评估 Dagger 实例
    def eval(cls, arg):
        """
        Evaluates the Dagger instance.

        Examples
        ========

        >>> from sympy import I
        >>> from sympy.physics.secondquant import Dagger, B, Bd
        >>> Dagger(2*I)
        -2*I
        >>> Dagger(B(0))
        CreateBoson(0)
        >>> Dagger(Bd(0))
        AnnihilateBoson(0)

        The eval() method is called automatically.

        """
        # 获取参数 arg 的 _dagger_ 属性
        dagger = getattr(arg, '_dagger_', None)
        # 如果 _dagger_ 属性存在，则调用该方法返回结果
        if dagger is not None:
            return dagger()
        # 如果 arg 是 Basic 类型
        if isinstance(arg, Basic):
            # 如果 arg 是加法表达式
            if arg.is_Add:
                # 对 arg 中每个元素递归调用 Dagger 方法并返回加法结果
                return Add(*tuple(map(Dagger, arg.args)))
            # 如果 arg 是乘法表达式
            if arg.is_Mul:
                # 对 arg 中每个元素逆序递归调用 Dagger 方法并返回乘法结果
                return Mul(*tuple(map(Dagger, reversed(arg.args))))
            # 如果 arg 是数值类型
            if arg.is_Number:
                # 直接返回数值本身
                return arg
            # 如果 arg 是幂次方表达式
            if arg.is_Pow:
                # 递归调用 Dagger 方法并返回幂次方结果
                return Pow(Dagger(arg.args[0]), arg.args[1])
            # 如果 arg 是虚数单位 I
            if arg == I:
                # 返回其相反数
                return -arg
        else:
            # 对于非 Basic 类型的参数，返回 None
            return None

    # 定义私有方法 _dagger_，用于返回 self 对象的第一个参数
    def _dagger_(self):
        return self.args[0]
class TensorSymbol(Expr):
    # TensorSymbol 类，继承自 Expr 类

    is_commutative = True
    # 设置类属性 is_commutative 为 True，表示张量符号是可交换的


class AntiSymmetricTensor(TensorSymbol):
    """Stores upper and lower indices in separate Tuple's.

    Each group of indices is assumed to be antisymmetric.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.secondquant import AntiSymmetricTensor
    >>> i, j = symbols('i j', below_fermi=True)
    >>> a, b = symbols('a b', above_fermi=True)
    >>> AntiSymmetricTensor('v', (a, i), (b, j))
    AntiSymmetricTensor(v, (a, i), (b, j))
    >>> AntiSymmetricTensor('v', (i, a), (b, j))
    -AntiSymmetricTensor(v, (a, i), (b, j))

    As you can see, the indices are automatically sorted to a canonical form.

    """

    def __new__(cls, symbol, upper, lower):
        # 构造函数，创建一个新的 AntiSymmetricTensor 实例

        try:
            # 尝试对上标和下标进行按反交换费米子排序
            upper, signu = _sort_anticommuting_fermions(
                upper, key=cls._sortkey)
            lower, signl = _sort_anticommuting_fermions(
                lower, key=cls._sortkey)

        except ViolationOfPauliPrinciple:
            # 如果出现保利不原则的违规情况，返回零
            return S.Zero

        symbol = sympify(symbol)  # 将符号转换为 SymPy 对象
        upper = Tuple(*upper)  # 将上标转换为元组
        lower = Tuple(*lower)  # 将下标转换为元组

        if (signu + signl) % 2:
            # 如果符号标志为奇数，返回负的 TensorSymbol 实例
            return -TensorSymbol.__new__(cls, symbol, upper, lower)
        else:
            # 否则返回正的 TensorSymbol 实例
            return TensorSymbol.__new__(cls, symbol, upper, lower)

    @classmethod
    def _sortkey(cls, index):
        """Key for sorting of indices.

        particle < hole < general

        FIXME: This is a bottle-neck, can we do it faster?
        """
        # 用于索引排序的关键字函数

        h = hash(index)  # 获取索引的哈希值
        label = str(index)  # 获取索引的字符串表示

        if isinstance(index, Dummy):
            # 如果索引是 Dummy 类型
            if index.assumptions0.get('above_fermi'):
                # 如果索引在费米面上方，返回元组 (20, label, h)
                return (20, label, h)
            elif index.assumptions0.get('below_fermi'):
                # 如果索引在费米面下方，返回元组 (21, label, h)
                return (21, label, h)
            else:
                # 否则返回元组 (22, label, h)
                return (22, label, h)

        # 如果索引不是 Dummy 类型
        if index.assumptions0.get('above_fermi'):
            # 如果索引在费米面上方，返回元组 (10, label, h)
            return (10, label, h)
        elif index.assumptions0.get('below_fermi'):
            # 如果索引在费米面下方，返回元组 (11, label, h)
            return (11, label, h)
        else:
            # 否则返回元组 (12, label, h)
            return (12, label, h)

    def _latex(self, printer):
        # 返回 LaTeX 格式的字符串表示
        return "{%s^{%s}_{%s}}" % (
            self.symbol,
            "".join([i.name for i in self.args[1]]),
            "".join([i.name for i in self.args[2]])
        )

    @property
    def symbol(self):
        """
        Returns the symbol of the tensor.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import AntiSymmetricTensor
        >>> i, j = symbols('i,j', below_fermi=True)
        >>> a, b = symbols('a,b', above_fermi=True)
        >>> AntiSymmetricTensor('v', (a, i), (b, j))
        AntiSymmetricTensor(v, (a, i), (b, j))
        >>> AntiSymmetricTensor('v', (a, i), (b, j)).symbol
        v

        """
        return self.args[0]  # 返回张量的符号属性

    @property
    def
    # 定义一个方法，返回张量的上标（第二个索引对）
    def upper(self):
        """
        返回张量的上标。

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import AntiSymmetricTensor
        >>> i, j = symbols('i,j', below_fermi=True)
        >>> a, b = symbols('a,b', above_fermi=True)
        >>> AntiSymmetricTensor('v', (a, i), (b, j))
        AntiSymmetricTensor(v, (a, i), (b, j))
        >>> AntiSymmetricTensor('v', (a, i), (b, j)).upper
        (a, i)
        """
        return self.args[1]

    @property
    # 定义一个属性，返回张量的下标（第三个索引对）
    def lower(self):
        """
        返回张量的下标。

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import AntiSymmetricTensor
        >>> i, j = symbols('i,j', below_fermi=True)
        >>> a, b = symbols('a,b', above_fermi=True)
        >>> AntiSymmetricTensor('v', (a, i), (b, j))
        AntiSymmetricTensor(v, (a, i), (b, j))
        >>> AntiSymmetricTensor('v', (a, i), (b, j)).lower
        (b, j)
        """
        return self.args[2]

    # 定义字符串表示方法，返回格式化的字符串
    def __str__(self):
        return "%s(%s,%s)" % self.args
class SqOperator(Expr):
    """
    Base class for Second Quantization operators.
    """

    op_symbol = 'sq'  # 操作符号，表示二次量子化操作

    is_commutative = False  # 这个操作符是否可交换，通常是False，即不可交换的性质

    def __new__(cls, k):
        obj = Basic.__new__(cls, sympify(k))  # 创建一个新的实例，将输入参数k转换为符号表达式
        return obj

    @property
    def state(self):
        """
        Returns the state index related to this operator.
        返回与此操作符相关的状态索引。
        """
        return self.args[0]  # 返回操作符的第一个参数作为状态索引

    @property
    def is_symbolic(self):
        """
        Returns True if the state is a symbol (as opposed to a number).
        如果状态是符号（而不是数字），则返回True。
        """
        if self.state.is_Integer:
            return False  # 如果状态是整数，则返回False
        else:
            return True  # 否则返回True，表示状态是符号类型

    def __repr__(self):
        return NotImplemented  # 返回未实现提示，表示在子类中实现

    def __str__(self):
        return "%s(%r)" % (self.op_symbol, self.state)  # 返回操作符的字符串表示形式，如'sq(p)'或'b(x)'

    def apply_operator(self, state):
        """
        Applies an operator to itself.
        将操作符应用到自身。
        """
        raise NotImplementedError('implement apply_operator in a subclass')  # 抛出未实现错误，子类需要实现此方法


class BosonicOperator(SqOperator):
    pass  # 简单继承SqOperator，没有额外的实现


class Annihilator(SqOperator):
    pass  # 简单继承SqOperator，没有额外的实现


class Creator(SqOperator):
    pass  # 简单继承SqOperator，没有额外的实现


class AnnihilateBoson(BosonicOperator, Annihilator):
    """
    Bosonic annihilation operator.
    波色子湮灭算符。

    Examples
    ========

    >>> from sympy.physics.secondquant import B
    >>> from sympy.abc import x
    >>> B(x)
    AnnihilateBoson(x)
    """

    op_symbol = 'b'  # 操作符号，表示波色子湮灭算符

    def _dagger_(self):
        return CreateBoson(self.state)  # 返回与当前状态相关的波色子创建算符实例

    def apply_operator(self, state):
        """
        Apply state to self if self is not symbolic and state is a FockStateKet, else
        multiply self by state.
        如果self不是符号，并且state是FockStateKet，则应用state到self，否则将self乘以state。

        Examples
        ========

        >>> from sympy.physics.secondquant import B, BKet
        >>> from sympy.abc import x, y, n
        >>> B(x).apply_operator(y)
        y*AnnihilateBoson(x)
        >>> B(0).apply_operator(BKet((n,)))
        sqrt(n)*FockStateBosonKet((n - 1,))

        """
        if not self.is_symbolic and isinstance(state, FockStateKet):
            element = self.state
            amp = sqrt(state[element])
            return amp*state.down(element)  # 如果不是符号且state是FockStateKet，则返回对state应用算符后的结果
        else:
            return Mul(self, state)  # 否则返回self乘以state的乘积

    def __repr__(self):
        return "AnnihilateBoson(%s)" % self.state  # 返回波色子湮灭算符的字符串表示形式

    def _latex(self, printer):
        if self.state is S.Zero:
            return "b_{0}"  # 如果状态是零，则返回LaTeX表示形式"b_{0}"
        else:
            return "b_{%s}" % self.state.name  # 否则返回LaTeX表示形式"b_{状态名称}"


class CreateBoson(BosonicOperator, Creator):
    """
    Bosonic creation operator.
    波色子产生算符。
    """

    op_symbol = 'b+'  # 操作符号，表示波色子产生算符
    # 返回一个 AnnihilateBoson 对象，使用 self.state 作为参数
    def _dagger_(self):
        return AnnihilateBoson(self.state)

    # 如果 self 不是符号化的，并且 state 是 FockStateKet 类型，则将 state 应用到 self 上；
    # 否则将 self 乘以 state。
    def apply_operator(self, state):
        """
        Apply state to self if self is not symbolic and state is a FockStateKet, else
        multiply self by state.

        Examples
        ========

        >>> from sympy.physics.secondquant import B, Dagger, BKet
        >>> from sympy.abc import x, y, n
        >>> Dagger(B(x)).apply_operator(y)
        y*CreateBoson(x)
        >>> B(0).apply_operator(BKet((n,)))
        sqrt(n)*FockStateBosonKet((n - 1,))
        """
        # 如果 self 不是符号化并且 state 是 FockStateKet 类型
        if not self.is_symbolic and isinstance(state, FockStateKet):
            # 获取 self.state
            element = self.state
            # 计算系数 amp，使用 sqrt(state[element] + 1)
            amp = sqrt(state[element] + 1)
            # 返回 amp 乘以 state.up(element) 的结果
            return amp * state.up(element)
        else:
            # 否则返回 self 与 state 的乘积
            return Mul(self, state)

    # 返回一个字符串，表示 CreateBoson 对象的状态
    def __repr__(self):
        return "CreateBoson(%s)" % self.state

    # 返回一个 LaTeX 字符串，用于打印器打印对象的 LaTeX 表示
    def _latex(self, printer):
        if self.state is S.Zero:
            return "{b^\\dagger_{0}}"
        else:
            return "{b^\\dagger_{%s}}" % self.state.name
# 定义 AnnihilateBoson 的别名 B
B = AnnihilateBoson
# 定义 CreateBoson 的别名 Bd
Bd = CreateBoson

# 定义 FermionicOperator 类，继承自 SqOperator 类
class FermionicOperator(SqOperator):

    @property
    def is_restricted(self):
        """
        判断这个 FermionicOperator 是否受限于费米面以下或以上的轨道？

        Returns
        =======
        
        1  : 受限于费米面以上的轨道
        0  : 没有限制
        -1 : 受限于费米面以下的轨道

        Examples
        ========
        
        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F, Fd
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> F(a).is_restricted
        1
        >>> Fd(a).is_restricted
        1
        >>> F(i).is_restricted
        -1
        >>> Fd(i).is_restricted
        -1
        >>> F(p).is_restricted
        0
        >>> Fd(p).is_restricted
        0

        """
        # 获取第一个参数的假设条件
        ass = self.args[0].assumptions0
        # 如果假设条件中有 "below_fermi" 则返回 -1
        if ass.get("below_fermi"):
            return -1
        # 如果假设条件中有 "above_fermi" 则返回 1
        if ass.get("above_fermi"):
            return 1
        # 否则返回 0，表示没有限制
        return 0

    @property
    def is_above_fermi(self):
        """
        这个 FermionicOperator 的索引是否允许值在费米面以上？

        Examples
        ========
        
        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> F(a).is_above_fermi
        True
        >>> F(i).is_above_fermi
        False
        >>> F(p).is_above_fermi
        True

        Note
        ====
        
        对于创生算符 Fd，也适用相同的规则

        """
        # 返回索引是否不是 below_fermi 的假设条件
        return not self.args[0].assumptions0.get("below_fermi")

    @property
    def is_below_fermi(self):
        """
        这个 FermionicOperator 的索引是否允许值在费米面以下？

        Examples
        ========
        
        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> F(a).is_below_fermi
        False
        >>> F(i).is_below_fermi
        True
        >>> F(p).is_below_fermi
        True

        对于创生算符 Fd，也适用相同的规则

        """
        # 返回索引是否不是 above_fermi 的假设条件
        return not self.args[0].assumptions0.get("above_fermi")

    @property
    def is_only_below_fermi(self):
        """
        是否索引此FermionicOperator仅限于费米面以下的值？

        示例
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> F(a).is_only_below_fermi
        False
        >>> F(i).is_only_below_fermi
        True
        >>> F(p).is_only_below_fermi
        False

        对产生算符Fd同样适用
        """
        # 返回是否仅限于费米面以下的索引值
        return self.is_below_fermi and not self.is_above_fermi

    @property
    def is_only_above_fermi(self):
        """
        是否索引此FermionicOperator仅限于费米面以上的值？

        示例
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> F(a).is_only_above_fermi
        True
        >>> F(i).is_only_above_fermi
        False
        >>> F(p).is_only_above_fermi
        False

        对产生算符Fd同样适用
        """
        # 返回是否仅限于费米面以上的索引值
        return self.is_above_fermi and not self.is_below_fermi

    def _sortkey(self):
        """
        返回用于排序的键值，基于FermionicOperator对象的哈希值和标签。

        如果是只有产生算符，优先级为1；如果是只有湮灭算符，优先级为4；
        如果是湮灭算符的实例，优先级为3；如果是产生算符的实例，优先级为2。
        """
        h = hash(self)  # 计算对象自身的哈希值
        label = str(self.args[0])  # 获取对象的第一个参数的字符串表示

        if self.is_only_q_creator:
            return 1, label, h  # 如果是只有产生算符，则返回优先级1
        if self.is_only_q_annihilator:
            return 4, label, h  # 如果是只有湮灭算符，则返回优先级4
        if isinstance(self, Annihilator):
            return 3, label, h  # 如果是湮灭算符的实例，则返回优先级3
        if isinstance(self, Creator):
            return 2, label, h  # 如果是产生算符的实例，则返回优先级2
class AnnihilateFermion(FermionicOperator, Annihilator):
    """
    Fermionic annihilation operator.
    """

    op_symbol = 'f'

    def _dagger_(self):
        # 返回当前状态的对应的创建费米子操作符
        return CreateFermion(self.state)

    def apply_operator(self, state):
        """
        Apply state to self if self is not symbolic and state is a FockStateKet, else
        multiply self by state.

        Examples
        ========

        >>> from sympy.physics.secondquant import B, Dagger, BKet
        >>> from sympy.abc import x, y, n
        >>> Dagger(B(x)).apply_operator(y)
        y*CreateBoson(x)
        >>> B(0).apply_operator(BKet((n,)))
        sqrt(n)*FockStateBosonKet((n - 1,))
        """
        if isinstance(state, FockStateFermionKet):
            # 使用当前状态作用于给定状态的下降算符
            element = self.state
            return state.down(element)

        elif isinstance(state, Mul):
            # 将乘积分解为系数部分和非系数部分
            c_part, nc_part = state.args_cnc()
            if isinstance(nc_part[0], FockStateFermionKet):
                # 如果非系数部分的第一个元素是费米子态，使用当前状态作用于下降算符
                element = self.state
                return Mul(*(c_part + [nc_part[0].down(element)] + nc_part[1:]))
            else:
                return Mul(self, state)

        else:
            # 如果状态不是费米子态，则将当前操作符与状态进行乘积
            return Mul(self, state)

    @property
    def is_q_creator(self):
        """
        Can we create a quasi-particle?  (create hole or create particle)
        If so, would that be above or below the fermi surface?

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> F(a).is_q_creator
        0
        >>> F(i).is_q_creator
        -1
        >>> F(p).is_q_creator
        -1

        """
        if self.is_below_fermi:
            # 如果操作符在费米面以下，则不能创建准粒子
            return -1
        return 0

    @property
    def is_q_annihilator(self):
        """
        Can we destroy a quasi-particle?  (annihilate hole or annihilate particle)
        If so, would that be above or below the fermi surface?

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> a = Symbol('a', above_fermi=1)
        >>> i = Symbol('i', below_fermi=1)
        >>> p = Symbol('p')

        >>> F(a).is_q_annihilator
        1
        >>> F(i).is_q_annihilator
        0
        >>> F(p).is_q_annihilator
        1

        """
        if self.is_above_fermi:
            # 如果操作符在费米面以上，则可以销毁准粒子
            return 1
        return 0

    @property
    def is_only_q_creator(self):
        """
        Always create a quasi-particle?  (create hole or create particle)

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> a = Symbol('a', above_fermi=True)  # Symbol 'a' above the Fermi level
        >>> i = Symbol('i', below_fermi=True)  # Symbol 'i' below the Fermi level
        >>> p = Symbol('p')  # General symbol

        >>> F(a).is_only_q_creator
        False  # 'a' is above the Fermi level, so it does not create a quasi-particle
        >>> F(i).is_only_q_creator
        True   # 'i' is below the Fermi level, so it creates a quasi-particle
        >>> F(p).is_only_q_creator
        False  # General symbol 'p' does not create a quasi-particle

        """
        return self.is_only_below_fermi

    @property
    def is_only_q_annihilator(self):
        """
        Always destroy a quasi-particle?  (annihilate hole or annihilate particle)

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> a = Symbol('a', above_fermi=True)  # Symbol 'a' above the Fermi level
        >>> i = Symbol('i', below_fermi=True)  # Symbol 'i' below the Fermi level
        >>> p = Symbol('p')  # General symbol

        >>> F(a).is_only_q_annihilator
        True   # 'a' is above the Fermi level, so it annihilates a quasi-particle
        >>> F(i).is_only_q_annihilator
        False  # 'i' is below the Fermi level, so it does not annihilate a quasi-particle
        >>> F(p).is_only_q_annihilator
        False  # General symbol 'p' does not annihilate a quasi-particle

        """
        return self.is_only_above_fermi

    def __repr__(self):
        """
        Return a string representation of the object.

        This method returns a string representing an object of type AnnihilateFermion
        based on its state.

        """
        return "AnnihilateFermion(%s)" % self.state

    def _latex(self, printer):
        """
        Return a LaTeX representation of the object.

        This method returns a LaTeX representation for rendering the object
        in a formatted way, based on its state.

        Parameters:
        ----------
        printer : printer object
            The printer object used for converting the object into LaTeX.

        Returns:
        -------
        str
            LaTeX representation of the object.

        """
        if self.state is S.Zero:
            return "a_{0}"  # LaTeX representation for the zero state
        else:
            return "a_{%s}" % self.state.name  # LaTeX representation with state name
class CreateFermion(FermionicOperator, Creator):
    """
    Fermionic creation operator.
    """

    op_symbol = 'f+'

    def _dagger_(self):
        # 返回对应的湮灭算符对象，使用当前状态
        return AnnihilateFermion(self.state)

    def apply_operator(self, state):
        """
        Apply state to self if self is not symbolic and state is a FockStateKet, else
        multiply self by state.

        Examples
        ========

        >>> from sympy.physics.secondquant import B, Dagger, BKet
        >>> from sympy.abc import x, y, n
        >>> Dagger(B(x)).apply_operator(y)
        y*CreateBoson(x)
        >>> B(0).apply_operator(BKet((n,)))
        sqrt(n)*FockStateBosonKet((n - 1,))
        """
        if isinstance(state, FockStateFermionKet):
            # 如果状态是费米子态，则作用当前状态对其进行提升操作
            element = self.state
            return state.up(element)

        elif isinstance(state, Mul):
            # 如果状态是乘法表达式，拆分为系数部分和非系数部分
            c_part, nc_part = state.args_cnc()
            if isinstance(nc_part[0], FockStateFermionKet):
                # 如果非系数部分的第一个元素是费米子态，则作用当前状态对其进行提升操作
                element = self.state
                return Mul(*(c_part + [nc_part[0].up(element)] + nc_part[1:]))

        # 默认情况下，返回当前创建算符乘以给定状态
        return Mul(self, state)

    @property
    def is_q_creator(self):
        """
        Can we create a quasi-particle?  (create hole or create particle)
        If so, would that be above or below the fermi surface?

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import Fd
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> Fd(a).is_q_creator
        1
        >>> Fd(i).is_q_creator
        0
        >>> Fd(p).is_q_creator
        1

        """
        if self.is_above_fermi:
            # 如果可以创建准粒子（在费米面之上），返回1
            return 1
        return 0

    @property
    def is_q_annihilator(self):
        """
        Can we destroy a quasi-particle?  (annihilate hole or annihilate particle)
        If so, would that be above or below the fermi surface?

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import Fd
        >>> a = Symbol('a', above_fermi=1)
        >>> i = Symbol('i', below_fermi=1)
        >>> p = Symbol('p')

        >>> Fd(a).is_q_annihilator
        0
        >>> Fd(i).is_q_annihilator
        -1
        >>> Fd(p).is_q_annihilator
        -1

        """
        if self.is_below_fermi:
            # 如果可以湮灭准粒子（在费米面之下），返回-1
            return -1
        return 0

    @property
    def is_only_q_creator(self):
        """
        Always create a quasi-particle?  (create hole or create particle)

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import Fd
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> Fd(a).is_only_q_creator
        True
        >>> Fd(i).is_only_q_creator
        False
        >>> Fd(p).is_only_q_creator
        False

        """
        return self.is_only_above_fermi

    @property
    # 以下是另一个属性的定义，但由于截断而不完整
    # 检查当前对象是否仅为准粒子湮灭操作符（即只湮灭空穴或粒子）
    def is_only_q_annihilator(self):
        """
        Always destroy a quasi-particle?  (annihilate hole or annihilate particle)

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import Fd
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> Fd(a).is_only_q_annihilator
        False
        >>> Fd(i).is_only_q_annihilator
        True
        >>> Fd(p).is_only_q_annihilator
        False

        """
        # 返回 self.is_only_below_fermi 的值，用于判断是否为仅在费米面以下操作的湮灭操作符
        return self.is_only_below_fermi

    # 返回对象的字符串表示，格式为 "CreateFermion(状态)"
    def __repr__(self):
        return "CreateFermion(%s)" % self.state

    # 返回对象的 LaTeX 表示，根据状态是否为零决定输出不同的 LaTeX 代码
    def _latex(self, printer):
        if self.state is S.Zero:
            return "{a^\\dagger_{0}}"
        else:
            return "{a^\\dagger_{%s}}" % self.state.name
# 创建费米子的产生算子Fd和湮灭算子F
Fd = CreateFermion
F = AnnihilateFermion

# 表示一个多粒子费米子Fock态，带有一系列占据数的序列。
# 在任何可以使用FockState的地方，也可以使用S.Zero。
# 所有代码必须检查这一点！
# 基类用于表示FockState。
class FockState(Expr):
    is_commutative = False  # 不可交换的属性

    def __new__(cls, occupations):
        """
        occupations 是一个列表，具有两种可能的含义：

        - 对于玻色子，它是一个占据数列表。
          第 i 个元素是状态 i 中的粒子数。

        - 对于费米子，它是一个占据的轨道列表。
          第一个元素是首先占据的状态，第 i 个元素是第 i 个占据的状态。
        """
        occupations = list(map(sympify, occupations))
        obj = Basic.__new__(cls, Tuple(*occupations))
        return obj

    def __getitem__(self, i):
        i = int(i)
        return self.args[0][i]

    def __repr__(self):
        return ("FockState(%r)") % (self.args)

    def __str__(self):
        return "%s%r%s" % (getattr(self, 'lbracket', ""), self._labels(), getattr(self, 'rbracket', ""))

    def _labels(self):
        return self.args[0]

    def __len__(self):
        return len(self.args[0])

    def _latex(self, printer):
        return "%s%s%s" % (getattr(self, 'lbracket_latex', ""), printer._print(self._labels()), getattr(self, 'rbracket_latex', ""))


# FockState的玻色子子类
class BosonState(FockState):
    """
    FockStateBoson(Ket/Bra)的基类。
    """

    def up(self, i):
        """
        执行产生算子的操作。

        示例
        ========

        >>> from sympy.physics.secondquant import BBra
        >>> b = BBra([1, 2])
        >>> b
        FockStateBosonBra((1, 2))
        >>> b.up(1)
        FockStateBosonBra((1, 3))
        """
        i = int(i)
        new_occs = list(self.args[0])
        new_occs[i] = new_occs[i] + S.One
        return self.__class__(new_occs)

    def down(self, i):
        """
        执行湮灭算子的操作。

        示例
        ========

        >>> from sympy.physics.secondquant import BBra
        >>> b = BBra([1, 2])
        >>> b
        FockStateBosonBra((1, 2))
        >>> b.down(1)
        FockStateBosonBra((1, 1))
        """
        i = int(i)
        new_occs = list(self.args[0])
        if new_occs[i] == S.Zero:
            return S.Zero
        else:
            new_occs[i] = new_occs[i] - S.One
            return self.__class__(new_occs)


# FockState的费米子子类
class FermionState(FockState):
    """
    FockStateFermion(Ket/Bra)的基类。
    """

    fermi_level = 0
    # 定义一个特殊方法 `__new__`，用于创建新的对象实例
    def __new__(cls, occupations, fermi_level=0):
        # 将 occupations 列表中的每个元素转换为 sympy 符号
        occupations = list(map(sympify, occupations))
        
        # 如果 occupations 的长度大于 1，则尝试按照 hash 值排序，处理反对易的费米子
        if len(occupations) > 1:
            try:
                (occupations, sign) = _sort_anticommuting_fermions(
                    occupations, key=hash)
            # 如果违反 Pauli 原理则返回零
            except ViolationOfPauliPrinciple:
                return S.Zero
        else:
            sign = 0

        # 设置类属性 fermi_level
        cls.fermi_level = fermi_level

        # 如果占据态中的空穴数超过 fermi_level，则返回零
        if cls._count_holes(occupations) > fermi_level:
            return S.Zero

        # 根据 sign 的奇偶性，返回正负号乘以 FockState 对象
        if sign % 2:
            return S.NegativeOne * FockState.__new__(cls, occupations)
        else:
            return FockState.__new__(cls, occupations)

    # 定义一个方法 `up`，表示创建算符的作用
    def up(self, i):
        """
        Performs the action of a creation operator.

        Explanation
        ===========

        If below fermi we try to remove a hole,
        if above fermi we try to create a particle.

        If general index p we return ``Kronecker(p,i)*self``
        where ``i`` is a new symbol with restriction above or below.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import FKet
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> FKet([]).up(a)
        FockStateFermionKet((a,))

        A creator acting on vacuum below fermi vanishes

        >>> FKet([]).up(i)
        0


        """
        # 检查索引 i 是否在当前态的参数列表中
        present = i in self.args[0]

        # 如果 i 在 fermi 级别以下，则尝试移除一个空穴
        if self._only_above_fermi(i):
            if present:
                return S.Zero
            else:
                return self._add_orbit(i)
        # 如果 i 在 fermi 级别以上，则尝试创建一个粒子
        elif self._only_below_fermi(i):
            if present:
                return self._remove_orbit(i)
            else:
                return S.Zero
        else:
            # 如果 i 既不在 fermi 级别以上也不在以下，创建一个虚拟粒子或空穴
            if present:
                hole = Dummy("i", below_fermi=True)
                return KroneckerDelta(i, hole) * self._remove_orbit(i)
            else:
                particle = Dummy("a", above_fermi=True)
                return KroneckerDelta(i, particle) * self._add_orbit(i)
    def down(self, i):
        """
        Performs the action of an annihilation operator.

        Explanation
        ===========

        If below fermi we try to create a hole,
        If above fermi we try to remove a particle.

        If general index p we return ``Kronecker(p,i)*self``
        where ``i`` is a new symbol with restriction above or below.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import FKet
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        An annihilator acting on vacuum above fermi vanishes

        >>> FKet([]).down(a)
        0

        Also below fermi, it vanishes, unless we specify a fermi level > 0

        >>> FKet([]).down(i)
        0
        >>> FKet([],4).down(i)
        FockStateFermionKet((i,))

        """
        present = i in self.args[0]  # 检查 i 是否在当前状态的参数列表中

        if self._only_above_fermi(i):  # 如果 i 在 fermi 面上方
            if present:
                return self._remove_orbit(i)  # 移除粒子轨道 i
            else:
                return S.Zero  # 如果 i 不在参数列表中，则返回零

        elif self._only_below_fermi(i):  # 如果 i 在 fermi 面下方
            if present:
                return S.Zero  # 如果 i 在参数列表中，返回零
            else:
                return self._add_orbit(i)  # 在参数列表中添加粒子轨道 i

        else:  # 如果 i 位置不明确（既不在 fermi 面上方也不在下方）
            if present:
                hole = Dummy("i", below_fermi=True)
                return KroneckerDelta(i, hole)*self._add_orbit(i)  # 返回 Kronecker delta 乘以添加粒子轨道 i
            else:
                particle = Dummy("a", above_fermi=True)
                return KroneckerDelta(i, particle)*self._remove_orbit(i)  # 返回 Kronecker delta 乘以移除粒子轨道 i

    @classmethod
    def _only_below_fermi(cls, i):
        """
        Tests if given orbit is only below fermi surface.

        If nothing can be concluded we return a conservative False.
        """
        if i.is_number:
            return i <= cls.fermi_level  # 如果 i 是数字，检查其是否小于等于 fermi 水平
        if i.assumptions0.get('below_fermi'):
            return True  # 如果 i 的假设表明在 fermi 面下方，则返回 True
        return False  # 其它情况返回 False

    @classmethod
    def _only_above_fermi(cls, i):
        """
        Tests if given orbit is only above fermi surface.

        If fermi level has not been set we return True.
        If nothing can be concluded we return a conservative False.
        """
        if i.is_number:
            return i > cls.fermi_level  # 如果 i 是数字，检查其是否大于 fermi 水平
        if i.assumptions0.get('above_fermi'):
            return True  # 如果 i 的假设表明在 fermi 面上方，则返回 True
        return not cls.fermi_level  # 如果 fermi 水平未设置，则返回 True；否则返回 False

    def _remove_orbit(self, i):
        """
        Removes particle/fills hole in orbit i. No input tests performed here.
        """
        new_occs = list(self.args[0])  # 创建参数列表的副本
        pos = new_occs.index(i)  # 获取轨道 i 在参数列表中的位置
        del new_occs[pos]  # 在副本中删除轨道 i
        if (pos) % 2:
            return S.NegativeOne*self.__class__(new_occs, self.fermi_level)  # 返回带有更新参数列表的对象，位置为奇数
        else:
            return self.__class__(new_occs, self.fermi_level)  # 返回带有更新参数列表的对象，位置为偶数

    def _add_orbit(self, i):
        """
        Adds particle/creates hole in orbit i. No input tests performed here.
        """
        return self.__class__((i,) + self.args[0], self.fermi_level)  # 返回带有添加轨道 i 后的参数列表的对象

    @classmethod
    # 计算列表中符合条件的能级下方的状态数量，并返回该数量
    def _count_holes(cls, list):
        return len([i for i in list if cls._only_below_fermi(i)])
    
    # 将列表中小于或等于当前对象的费米能级的值取负，并返回结果元组
    def _negate_holes(self, list):
        return tuple([-i if i <= self.fermi_level else i for i in list])
    
    # 返回对象的字符串表示形式，包括费米能级信息（如果存在）
    def __repr__(self):
        if self.fermi_level:
            return "FockStateKet(%r, fermi_level=%s)" % (self.args[0], self.fermi_level)
        else:
            return "FockStateKet(%r)" % (self.args[0],)
    
    # 返回对象参数的标签，通过调用_negate_holes方法得到
    def _labels(self):
        return self._negate_holes(self.args[0])
class FockStateKet(FockState):
    """
    Representation of a ket.
    """
    # 左尖括号的表示
    lbracket = '|'
    # 右尖括号的表示
    rbracket = '>'
    # 左尖括号的 LaTeX 表示
    lbracket_latex = r'\left|'
    # 右尖括号的 LaTeX 表示
    rbracket_latex = r'\right\rangle'


class FockStateBra(FockState):
    """
    Representation of a bra.
    """
    # 左尖括号的表示
    lbracket = '<'
    # 右尖括号的表示
    rbracket = '|'
    # 左尖括号的 LaTeX 表示
    lbracket_latex = r'\left\langle'
    # 右尖括号的 LaTeX 表示
    rbracket_latex = r'\right|'

    def __mul__(self, other):
        # 如果乘法的另一个操作数是 FockStateKet 类型，则返回它们的内积
        if isinstance(other, FockStateKet):
            return InnerProduct(self, other)
        else:
            # 否则，按默认行为进行乘法操作
            return Expr.__mul__(self, other)


class FockStateBosonKet(BosonState, FockStateKet):
    """
    Many particle Fock state with a sequence of occupation numbers.

    Occupation numbers can be any integer >= 0.

    Examples
    ========

    >>> from sympy.physics.secondquant import BKet
    >>> BKet([1, 2])
    FockStateBosonKet((1, 2))
    """
    def _dagger_(self):
        # 返回该 BosonKet 的 Hermitian 转置，即对应的 BosonBra
        return FockStateBosonBra(*self.args)


class FockStateBosonBra(BosonState, FockStateBra):
    """
    Describes a collection of BosonBra particles.

    Examples
    ========

    >>> from sympy.physics.secondquant import BBra
    >>> BBra([1, 2])
    FockStateBosonBra((1, 2))
    """
    def _dagger_(self):
        # 返回该 BosonBra 的 Hermitian 转置，即对应的 BosonKet
        return FockStateBosonKet(*self.args)


class FockStateFermionKet(FermionState, FockStateKet):
    """
    Many-particle Fock state with a sequence of occupied orbits.

    Explanation
    ===========

    Each state can only have one particle, so we choose to store a list of
    occupied orbits rather than a tuple with occupation numbers (zeros and ones).

    states below fermi level are holes, and are represented by negative labels
    in the occupation list.

    For symbolic state labels, the fermi_level caps the number of allowed hole-
    states.

    Examples
    ========

    >>> from sympy.physics.secondquant import FKet
    >>> FKet([1, 2])
    FockStateFermionKet((1, 2))
    """
    def _dagger_(self):
        # 返回该 FermionKet 的 Hermitian 转置，即对应的 FermionBra
        return FockStateFermionBra(*self.args)


class FockStateFermionBra(FermionState, FockStateBra):
    """
    See Also
    ========

    FockStateFermionKet

    Examples
    ========

    >>> from sympy.physics.secondquant import FBra
    >>> FBra([1, 2])
    FockStateFermionBra((1, 2))
    """
    def _dagger_(self):
        # 返回该 FermionBra 的 Hermitian 转置，即对应的 FermionKet
        return FockStateFermionKet(*self.args)

# 简化的别名定义
BBra = FockStateBosonBra
BKet = FockStateBosonKet
FBra = FockStateFermionBra
FKet = FockStateFermionKet


def _apply_Mul(m):
    """
    Take a Mul instance with operators and apply them to states.

    Explanation
    ===========

    This method applies all operators with integer state labels
    to the actual states. For symbolic state labels, nothing is done.
    When inner products of FockStates are encountered (like <a|b>),
    they are converted to instances of InnerProduct.

    This does not currently work on double inner products like,
    <a|b><c|d>.

    If the argument is not a Mul, it is simply returned as is.
    """
    # 如果 m 不是 Mul 类型的对象，则直接返回 m
    if not isinstance(m, Mul):
        return m
    # 将模式对象 m 分解为有色部分和非有色部分
    c_part, nc_part = m.args_cnc()
    # 计算非有色部分的长度
    n_nc = len(nc_part)
    # 如果非有色部分长度为 0 或 1，则直接返回模式 m
    if n_nc in (0, 1):
        return m
    else:
        # 获取非有色部分的最后一个对象和倒数第二个对象
        last = nc_part[-1]
        next_to_last = nc_part[-2]
        # 如果最后一个对象是 FockStateKet 类型
        if isinstance(last, FockStateKet):
            # 如果倒数第二个对象是 SqOperator 类型
            if isinstance(next_to_last, SqOperator):
                # 如果 SqOperator 是符号型的，返回模式 m
                if next_to_last.is_symbolic:
                    return m
                else:
                    # 应用 SqOperator 到 FockStateKet 上，得到结果
                    result = next_to_last.apply_operator(last)
                    # 如果结果为 0，则返回 SymPy 的零对象
                    if result == 0:
                        return S.Zero
                    else:
                        # 将计算结果与有色部分重新组合成一个新的乘积表达式，并应用 _apply_Mul 函数
                        return _apply_Mul(Mul(*(c_part + nc_part[:-2] + [result])))
            # 如果倒数第二个对象是 Pow 类型
            elif isinstance(next_to_last, Pow):
                # 如果 Pow 对象的基础是 SqOperator 类型且指数是整数
                if isinstance(next_to_last.base, SqOperator) and \
                        next_to_last.exp.is_Integer:
                    # 如果 SqOperator 是符号型的，返回模式 m
                    if next_to_last.base.is_symbolic:
                        return m
                    else:
                        # 反复应用 SqOperator 到 FockStateKet 上，得到结果
                        result = last
                        for i in range(next_to_last.exp):
                            result = next_to_last.base.apply_operator(result)
                            if result == 0:
                                break
                        # 如果结果为 0，则返回 SymPy 的零对象
                        if result == 0:
                            return S.Zero
                        else:
                            # 将计算结果与有色部分重新组合成一个新的乘积表达式，并应用 _apply_Mul 函数
                            return _apply_Mul(Mul(*(c_part + nc_part[:-2] + [result])))
                else:
                    # 否则返回模式 m
                    return m
            # 如果倒数第二个对象是 FockStateBra 类型
            elif isinstance(next_to_last, FockStateBra):
                # 计算 FockStateBra 和 FockStateKet 的内积
                result = InnerProduct(next_to_last, last)
                # 如果结果为 0，则返回 SymPy 的零对象
                if result == 0:
                    return S.Zero
                else:
                    # 将计算结果与有色部分重新组合成一个新的乘积表达式，并应用 _apply_Mul 函数
                    return _apply_Mul(Mul(*(c_part + nc_part[:-2] + [result])))
            else:
                # 其他情况下返回模式 m
                return m
        else:
            # 如果最后一个对象不是 FockStateKet 类型，则直接返回模式 m
            return m
# 定义一个函数，用于对 SymPy 表达式中的算符和状态应用操作
def apply_operators(e):
    # 对表达式进行展开
    e = e.expand()
    # 获取表达式中所有的乘法项
    muls = e.atoms(Mul)
    # 生成乘法项的替换列表，每个乘法项都会被 _apply_Mul 函数处理
    subs_list = [(m, _apply_Mul(m)) for m in iter(muls)]
    # 对表达式应用生成的替换列表
    return e.subs(subs_list)


# 定义一个类，表示未求值的态矢内积
class InnerProduct(Basic):
    """
    An unevaluated inner product between a bra and ket.
    
    Explanation
    ===========
    
    Currently this class just reduces things to a product of
    Kronecker Deltas.  In the future, we could introduce abstract
    states like ``|a>`` and ``|b>``, and leave the inner product unevaluated as
    ``<a|b>``.
    
    """
    is_commutative = True

    def __new__(cls, bra, ket):
        # 检查参数 bra 是否为 FockStateBra 类型
        if not isinstance(bra, FockStateBra):
            raise TypeError("must be a bra")
        # 检查参数 ket 是否为 FockStateKet 类型
        if not isinstance(ket, FockStateKet):
            raise TypeError("must be a ket")
        # 调用 eval 方法进行求值
        return cls.eval(bra, ket)

    @classmethod
    def eval(cls, bra, ket):
        # 初始化结果为 1
        result = S.One
        # 逐对比较 bra 和 ket 的元素，计算 Kronecker Delta 的乘积
        for i, j in zip(bra.args[0], ket.args[0]):
            result *= KroneckerDelta(i, j)
            # 如果结果为 0，则提前结束循环
            if result == 0:
                break
        # 返回最终的内积结果
        return result

    @property
    def bra(self):
        """Returns the bra part of the state"""
        return self.args[0]

    @property
    def ket(self):
        """Returns the ket part of the state"""
        return self.args[1]

    def __repr__(self):
        # 返回内积表示的字符串形式
        sbra = repr(self.bra)
        sket = repr(self.ket)
        return "%s|%s" % (sbra[:-1], sket[1:])

    def __str__(self):
        # 返回内积的字符串表示形式
        return self.__repr__()


# 定义一个函数，用于在给定基底中找到算符的表示矩阵
def matrix_rep(op, basis):
    """
    Find the representation of an operator in a basis.

    Examples
    ========

    >>> from sympy.physics.secondquant import VarBosonicBasis, B, matrix_rep
    >>> b = VarBosonicBasis(5)
    >>> o = B(0)
    >>> matrix_rep(o, b)
    Matrix([
    [0, 1,       0,       0, 0],
    [0, 0, sqrt(2),       0, 0],
    [0, 0,       0, sqrt(3), 0],
    [0, 0,       0,       0, 2],
    [0, 0,       0,       0, 0]])
    """
    # 创建一个全零矩阵，大小为基底的长度
    a = zeros(len(basis))
    # 遍历基底中的每一对基底态，计算对应的算符表示
    for i in range(len(basis)):
        for j in range(len(basis)):
            # 计算算符在基底态之间的表示矩阵元素
            a[i, j] = apply_operators(Dagger(basis[i])*op*basis[j])
    # 返回计算得到的表示矩阵
    return a


class BosonicBasis:
    """
    Base class for a basis set of bosonic Fock states.
    """
    pass


# 定义一个类，表示单态变粒子数的玻色子基底集合
class VarBosonicBasis:
    """
    A single state, variable particle number basis set.

    Examples
    ========

    >>> from sympy.physics.secondquant import VarBosonicBasis
    >>> b = VarBosonicBasis(5)
    >>> b
    [FockState((0,)), FockState((1,)), FockState((2,)),
     FockState((3,)), FockState((4,))]
    """

    def __init__(self, n_max):
        # 初始化最大粒子数
        self.n_max = n_max
        # 构建基底态
        self._build_states()

    def _build_states(self):
        # 构建基底态的方法，具体实现未提供
        pass
    # 构建基础态集合
    def _build_states(self):
        # 初始化空列表用于存储基础态
        self.basis = []
        # 循环创建基础态直到达到最大允许数目 self.n_max
        for i in range(self.n_max):
            # 创建单个基础态并添加到基础态列表中
            self.basis.append(FockStateBosonKet([i]))
        # 记录基础态的数量
        self.n_basis = len(self.basis)
    
    # 返回指定基础态在基础态列表中的索引
    def index(self, state):
        """
        返回状态在基础态列表中的索引。
    
        Examples
        ========
    
        >>> from sympy.physics.secondquant import VarBosonicBasis
        >>> b = VarBosonicBasis(3)
        >>> state = b.state(1)
        >>> b
        [FockState((0,)), FockState((1,)), FockState((2,))]
        >>> state
        FockStateBosonKet((1,))
        >>> b.index(state)
        1
        """
        return self.basis.index(state)
    
    # 返回指定索引处的基础态
    def state(self, i):
        """
        返回单个基础态。
    
        Examples
        ========
    
        >>> from sympy.physics.secondquant import VarBosonicBasis
        >>> b = VarBosonicBasis(5)
        >>> b.state(3)
        FockStateBosonKet((3,))
        """
        return self.basis[i]
    
    # 通过索引操作符 [] 获取基础态
    def __getitem__(self, i):
        return self.state(i)
    
    # 返回基础态列表的长度
    def __len__(self):
        return len(self.basis)
    
    # 返回基础态列表的字符串表示形式
    def __repr__(self):
        return repr(self.basis)
class FixedBosonicBasis(BosonicBasis):
    """
    Fixed particle number basis set.

    Examples
    ========

    >>> from sympy.physics.secondquant import FixedBosonicBasis
    >>> b = FixedBosonicBasis(2, 2)
    >>> state = b.state(1)
    >>> b
    [FockState((2, 0)), FockState((1, 1)), FockState((0, 2))]
    >>> state
    FockStateBosonKet((1, 1))
    >>> b.index(state)
    1
    """
    def __init__(self, n_particles, n_levels):
        self.n_particles = n_particles  # 初始化粒子数
        self.n_levels = n_levels  # 初始化能级数
        self._build_particle_locations()  # 构建粒子位置信息
        self._build_states()  # 构建基态

    def _build_particle_locations(self):
        tup = ["i%i" % i for i in range(self.n_particles)]  # 创建包含粒子编号的元组
        first_loop = "for i0 in range(%i)" % self.n_levels  # 第一个循环的字符串表示
        other_loops = ''
        for cur, prev in zip(tup[1:], tup):
            temp = "for %s in range(%s + 1) " % (cur, prev)  # 后续循环的字符串表示
            other_loops = other_loops + temp
        tup_string = "(%s)" % ", ".join(tup)  # 将粒子编号元组转换成字符串形式
        list_comp = "[%s %s %s]" % (tup_string, first_loop, other_loops)  # 构建列表推导式的字符串形式
        result = eval(list_comp)  # 执行列表推导式，生成粒子位置信息的列表
        if self.n_particles == 1:
            result = [(item,) for item in result]  # 如果只有一个粒子，将每个元素转换为只含一个元素的元组
        self.particle_locations = result  # 将粒子位置信息保存到实例变量中

    def _build_states(self):
        self.basis = []  # 初始化基态列表
        for tuple_of_indices in self.particle_locations:
            occ_numbers = self.n_levels*[0]  # 初始化占据数列表
            for level in tuple_of_indices:
                occ_numbers[level] += 1  # 计算每个能级的占据数
            self.basis.append(FockStateBosonKet(occ_numbers))  # 将生成的 FockStateBosonKet 实例加入基态列表
        self.n_basis = len(self.basis)  # 计算基态数量

    def index(self, state):
        """Returns the index of state in basis.

        Examples
        ========

        >>> from sympy.physics.secondquant import FixedBosonicBasis
        >>> b = FixedBosonicBasis(2, 3)
        >>> b.index(b.state(3))
        3
        """
        return self.basis.index(state)  # 返回指定状态在基态列表中的索引位置

    def state(self, i):
        """Returns the state that lies at index i of the basis

        Examples
        ========

        >>> from sympy.physics.secondquant import FixedBosonicBasis
        >>> b = FixedBosonicBasis(2, 3)
        >>> b.state(3)
        FockStateBosonKet((1, 0, 1))
        """
        return self.basis[i]  # 返回基态列表中指定索引位置的状态

    def __getitem__(self, i):
        return self.state(i)  # 实现索引操作，返回指定索引位置的状态

    def __len__(self):
        return len(self.basis)  # 返回基态列表的长度

    def __repr__(self):
        return repr(self.basis)  # 返回基态列表的字符串表示


class Commutator(Function):
    """
    The Commutator:  [A, B] = A*B - B*A

    The arguments are ordered according to .__cmp__()

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.secondquant import Commutator
    >>> A, B = symbols('A,B', commutative=False)
    >>> Commutator(B, A)
    -Commutator(A, B)

    Evaluate the commutator with .doit()

    >>> comm = Commutator(A,B); comm
    Commutator(A, B)
    >>> comm.doit()
    A*B - B*A


    For two second quantization operators the commutator is evaluated
    immediately:

    >>> from sympy.physics.secondquant import Fd, F
    """
    # 定义符号 `a`，表示费米子上面的态
    >>> a = symbols('a', above_fermi=True)
    # 定义符号 `i`，表示费米子下面的态
    >>> i = symbols('i', below_fermi=True)
    # 定义符号 `p` 和 `q`，表示未指定费米子态
    >>> p,q = symbols('p,q')
    
    # 计算费米子的产生算符 `Fd(a)` 和 `Fd(i)` 的对易子
    >>> Commutator(Fd(a),Fd(i))
    # 返回对易子的计算结果 `2*NO(CreateFermion(a)*CreateFermion(i))`
    2*NO(CreateFermion(a)*CreateFermion(i))
    
    # 对于更复杂的表达式，需要通过 `.doit()` 方法来触发计算
    >>> comm = Commutator(Fd(p)*Fd(q),F(i)); comm
    # 显示对易子的未简化表达式 `Commutator(CreateFermion(p)*CreateFermion(q), AnnihilateFermion(i))`
    Commutator(CreateFermion(p)*CreateFermion(q), AnnihilateFermion(i))
    >>> comm.doit(wicks=True)
    # 计算并简化对易子，返回 `-KroneckerDelta(i, p)*CreateFermion(q) + KroneckerDelta(i, q)*CreateFermion(p)`
    -KroneckerDelta(i, p)*CreateFermion(q) + KroneckerDelta(i, q)*CreateFermion(p)
    
    
    """
    # 设置此类对象为非交换的
    is_commutative = False
    
    @classmethod
    def eval(cls, a, b):
        """
        计算对易子 [A, B] 的规范形式，其中 A < B。
    
        Examples
        ========
    
        >>> from sympy.physics.secondquant import Commutator, F, Fd
        >>> from sympy.abc import x
        >>> c1 = Commutator(F(x), Fd(x))
        >>> c2 = Commutator(Fd(x), F(x))
        >>> Commutator.eval(c1, c2)
        0
        """
        # 如果其中一个操作数为 None，则返回零
        if not (a and b):
            return S.Zero
        # 如果 A 和 B 相等，则返回零
        if a == b:
            return S.Zero
        # 如果 A 或 B 是可交换的，则返回零
        if a.is_commutative or b.is_commutative:
            return S.Zero
    
        #
        # [A+B,C]  ->  [A,C] + [B,C]
        #
        # 展开操作数 A
        a = a.expand()
        if isinstance(a, Add):
            # 如果 A 是加法表达式，则返回 [A,C] 的加法结果
            return Add(*[cls(term, b) for term in a.args])
        # 展开操作数 B
        b = b.expand()
        if isinstance(b, Add):
            # 如果 B 是加法表达式，则返回 [A,B] 的加法结果
            return Add(*[cls(a, term) for term in b.args])
    
        #
        # [xA,yB]  ->  xy*[A,B]
        #
        # 将 A 和 B 分解为系数和非系数部分
        ca, nca = a.args_cnc()
        cb, ncb = b.args_cnc()
        c_part = list(ca) + list(cb)
        if c_part:
            # 如果存在非系数部分，则返回它们的乘积乘以 [A,B]
            return Mul(Mul(*c_part), cls(Mul._from_args(nca), Mul._from_args(ncb)))
    
        #
        # 单个二次量子化算符
        #
        # 如果 A 和 B 都是玻色算符
        if isinstance(a, BosonicOperator) and isinstance(b, BosonicOperator):
            # 如果 B 是产生算符，A 是湮灭算符，则返回它们态的克罗内克 δ 函数
            if isinstance(b, CreateBoson) and isinstance(a, AnnihilateBoson):
                return KroneckerDelta(a.state, b.state)
            # 如果 A 是产生算符，B 是湮灭算符，则返回它们态的负克罗内克 δ 函数
            if isinstance(a, CreateBoson) and isinstance(b, AnnihilateBoson):
                return S.NegativeOne*KroneckerDelta(a.state, b.state)
            # 其他情况返回零
            else:
                return S.Zero
        # 如果 A 和 B 都是费米算符
        if isinstance(a, FermionicOperator) and isinstance(b, FermionicOperator):
            # 使用 wicks 函数计算它们的收缩，并返回其差
            return wicks(a*b) - wicks(b*a)
    
        #
        # 规范排序参数
        #
        # 如果 A 的排序键大于 B 的排序键，则返回它们的负对易子
        if a.sort_key() > b.sort_key():
            return S.NegativeOne*cls(b, a)
    # 定义一个方法，用于计算对易子表达式
    def doit(self, **hints):
        """
        Enables the computation of complex expressions.

        Examples
        ========

        >>> from sympy.physics.secondquant import Commutator, F, Fd
        >>> from sympy import symbols
        >>> i, j = symbols('i,j', below_fermi=True)
        >>> a, b = symbols('a,b', above_fermi=True)
        >>> c = Commutator(Fd(a)*F(i),Fd(b)*F(j))
        >>> c.doit(wicks=True)
        0
        """
        # 从参数中获取第一个和第二个表达式
        a = self.args[0]
        b = self.args[1]

        # 如果 hints 中有"wicks"键
        if hints.get("wicks"):
            # 对第一个和第二个表达式分别进行 doit 操作
            a = a.doit(**hints)
            b = b.doit(**hints)
            try:
                # 尝试应用 Wick 定理进行计算，并返回结果
                return wicks(a*b) - wicks(b*a)
            except ContractionAppliesOnlyToFermions:
                # 如果遇到只适用于费米子的收缩操作异常，不处理
                pass
            except WicksTheoremDoesNotApply:
                # 如果 Wick 定理不适用异常，不处理
                pass

        # 对未进行 Wick 简化的表达式进行一般的对易子运算，并返回结果
        return (a*b - b*a).doit(**hints)

    # 定义对象的字符串表示形式，包含两个参数的表达式
    def __repr__(self):
        return "Commutator(%s,%s)" % (self.args[0], self.args[1])

    # 定义对象的字符串输出形式，格式化为对易子表示
    def __str__(self):
        return "[%s,%s]" % (self.args[0], self.args[1])

    # 定义对象的 LaTeX 输出形式，使用打印器来处理参数的 LaTeX 表示
    def _latex(self, printer):
        return "\\left[%s,%s\\right]" % tuple([
            printer._print(arg) for arg in self.args])
# 定义一个类 NO，继承自 Expr 类
class NO(Expr):
    """
    This Object is used to represent normal ordering brackets.

    i.e.  {abcd}  sometimes written  :abcd:

    Explanation
    ===========

    Applying the function NO(arg) to an argument means that all operators in
    the argument will be assumed to anticommute, and have vanishing
    contractions.  This allows an immediate reordering to canonical form
    upon object creation.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.secondquant import NO, F, Fd
    >>> p,q = symbols('p,q')
    >>> NO(Fd(p)*F(q))
    NO(CreateFermion(p)*AnnihilateFermion(q))
    >>> NO(F(q)*Fd(p))
    -NO(CreateFermion(p)*AnnihilateFermion(q))


    Note
    ====

    If you want to generate a normal ordered equivalent of an expression, you
    should use the function wicks().  This class only indicates that all
    operators inside the brackets anticommute, and have vanishing contractions.
    Nothing more, nothing less.

    """
    # 设置属性 is_commutative 为 False，表示这个类的对象不满足交换律
    is_commutative = False
    def __new__(cls, arg):
        """
        Use anticommutation to get canonical form of operators.

        Explanation
        ===========

        Employ associativity of normal ordered product: {ab{cd}} = {abcd}
        but note that {ab}{cd} /= {abcd}.

        We also employ distributivity: {ab + cd} = {ab} + {cd}.

        Canonical form also implies expand() {ab(c+d)} = {abc} + {abd}.

        """

        # 将输入参数转换为符号表达式
        arg = sympify(arg)
        # 展开符号表达式
        arg = arg.expand()
        # 如果是加法表达式，则对每个项递归调用 __new__ 方法
        if arg.is_Add:
            return Add(*[ cls(term) for term in arg.args])

        # 如果是乘法表达式
        if arg.is_Mul:

            # 将常数系数与正规序列分离出来
            c_part, seq = arg.args_cnc()
            if c_part:
                coeff = Mul(*c_part)
                if not seq:
                    return coeff
            else:
                coeff = S.One

            # 处理反对易子表达式 {ab{cd}} = {abcd}
            newseq = []
            foundit = False
            for fac in seq:
                if isinstance(fac, NO):
                    newseq.extend(fac.args)
                    foundit = True
                else:
                    newseq.append(fac)
            if foundit:
                return coeff * cls(Mul(*newseq))

            # 如果第一个因子是玻色算符，抛出未实现异常
            if isinstance(seq[0], BosonicOperator):
                raise NotImplementedError

            try:
                # 对反对易费米子进行排序
                newseq, sign = _sort_anticommuting_fermions(seq)
            except ViolationOfPauliPrinciple:
                return S.Zero

            # 根据排列符号确定返回值
            if sign % 2:
                return (S.NegativeOne * coeff) * cls(Mul(*newseq))
            elif sign:
                return coeff * cls(Mul(*newseq))
            else:
                pass  # 当 sign==0 时，不需要排列操作

            # 如果无法处理乘法表达式，标记为正规序
            if coeff != S.One:
                return coeff * cls(Mul(*newseq))
            return Expr.__new__(cls, Mul(*newseq))

        # 如果是正规序对象，则直接返回
        if isinstance(arg, NO):
            return arg

        # 如果不是乘法或加法表达式，正规序不适用，直接返回原参数
        return arg

    @property
    def has_q_creators(self):
        """
        Return 0 if the leftmost argument of the first argument is a not a
        q_creator, else 1 if it is above fermi or -1 if it is below fermi.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import NO, F, Fd

        >>> a = symbols('a', above_fermi=True)
        >>> i = symbols('i', below_fermi=True)
        >>> NO(Fd(a)*Fd(i)).has_q_creators
        1
        >>> NO(F(i)*F(a)).has_q_creators
        -1
        >>> NO(Fd(i)*F(a)).has_q_creators           #doctest: +SKIP
        0

        """
        # 返回左起第一个参数是否为 q_creator 的值
        return self.args[0].args[0].is_q_creator
    # 返回 0，如果第一个参数的最右边参数不是 q 湮灭算符；返回 1，如果它在费米面上；返回 -1，如果它在费米面下。
    def has_q_annihilators(self):
        """
        如果第一个参数的最右边参数不是 q 湮灭算符，则返回 0；
        如果在费米面上，则返回 1；
        如果在费米面下，则返回 -1。

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import NO, F, Fd

        >>> a = symbols('a', above_fermi=True)
        >>> i = symbols('i', below_fermi=True)
        >>> NO(Fd(a)*Fd(i)).has_q_annihilators
        -1
        >>> NO(F(i)*F(a)).has_q_annihilators
        1
        >>> NO(Fd(a)*F(i)).has_q_annihilators
        0

        """
        return self.args[0].args[-1].is_q_annihilator

    # 如果 hints 字典中的 "remove_brackets" 键为 True，则移除表达式中的括号；否则启用复杂计算。
    def doit(self, **hints):
        """
        如果 hints 中的 "remove_brackets" 键为 True，则移除表达式中的括号；
        否则，进行复杂计算。

        Examples
        ========

        >>> from sympy.physics.secondquant import NO, Fd, F
        >>> from textwrap import fill
        >>> from sympy import symbols, Dummy
        >>> p,q = symbols('p,q', cls=Dummy)
        >>> print(fill(str(NO(Fd(p)*F(q)).doit())))
        KroneckerDelta(_a, _p)*KroneckerDelta(_a, _q)*CreateFermion(_a)*AnnihilateFermion(_a) +
        KroneckerDelta(_a, _p)*KroneckerDelta(_i, _q)*CreateFermion(_a)*AnnihilateFermion(_i) -
        KroneckerDelta(_a, _q)*KroneckerDelta(_i, _p)*AnnihilateFermion(_a)*CreateFermion(_i) -
        KroneckerDelta(_i, _p)*KroneckerDelta(_i, _q)*AnnihilateFermion(_i)*CreateFermion(_i)
        """
        if hints.get("remove_brackets", True):
            # 调用 _remove_brackets 方法移除表达式中的括号
            return self._remove_brackets()
        else:
            # 调用 self.args[0].doit(**hints) 创建一个新的对象，并传递给 doit 方法
            return self.__new__(type(self), self.args[0].doit(**hints))
    def _remove_brackets(self):
        """
        Returns the sorted string without normal order brackets.

        The returned string have the property that no nonzero
        contractions exist.
        """

        # 检查是否有创造算符同时也是湮灭算符
        subslist = []
        # 遍历所有的创造算符
        for i in self.iter_q_creators():
            # 如果创造算符同时也是湮灭算符
            if self[i].is_q_annihilator:
                # 获取创造算符的状态假设
                assume = self[i].state.assumptions0

                # 只有带有虚拟指标的运算符可以分成两项
                if isinstance(self[i].state, Dummy):

                    # 创建带有费米限制的指标
                    assume.pop("above_fermi", None)
                    assume["below_fermi"] = True
                    below = Dummy('i', **assume)
                    assume.pop("below_fermi", None)
                    assume["above_fermi"] = True
                    above = Dummy('a', **assume)

                    cls = type(self[i])
                    # 将运算符分解成两项并加入替换列表中
                    split = (
                        self[i].__new__(cls, below)
                        * KroneckerDelta(below, self[i].state)
                        + self[i].__new__(cls, above)
                        * KroneckerDelta(above, self[i].state)
                    )
                    subslist.append((self[i], split))
                else:
                    # 如果运算符不是虚拟指标，则抛出异常
                    raise SubstitutionOfAmbigousOperatorFailed(self[i])
        # 如果存在替换列表，则进行替换并返回处理后的结果
        if subslist:
            result = NO(self.subs(subslist))
            if isinstance(result, Add):
                # 如果结果是加法表达式，则逐项执行运算
                return Add(*[term.doit() for term in result.args])
        else:
            # 如果没有需要替换的内容，则直接返回原始参数的第一个元素
            return self.args[0]

    def _expand_operators(self):
        """
        Returns a sum of NO objects that contain no ambiguous q-operators.

        Explanation
        ===========

        If an index q has range both above and below fermi, the operator F(q)
        is ambiguous in the sense that it can be both a q-creator and a q-annihilator.
        If q is dummy, it is assumed to be a summation variable and this method
        rewrites it into a sum of NO terms with unambiguous operators:

        {Fd(p)*F(q)} = {Fd(a)*F(b)} + {Fd(a)*F(i)} + {Fd(j)*F(b)} -{F(i)*Fd(j)}

        where a,b are above and i,j are below fermi level.
        """
        # 调用_remove_brackets方法处理括号后，返回不含模糊算符的NO对象的和
        return NO(self._remove_brackets)

    def __getitem__(self, i):
        # 如果i是切片对象，则返回对应索引范围内的元素列表
        if isinstance(i, slice):
            indices = i.indices(len(self))
            return [self.args[0].args[i] for i in range(*indices)]
        else:
            # 否则返回对应索引的元素
            return self.args[0].args[i]

    def __len__(self):
        # 返回self.args[0].args的长度
        return len(self.args[0].args)
    def iter_q_annihilators(self):
        """
        Iterates over the annihilation operators.

        Examples
        ========

        >>> from sympy import symbols
        >>> i, j = symbols('i j', below_fermi=True)
        >>> a, b = symbols('a b', above_fermi=True)
        >>> from sympy.physics.secondquant import NO, F, Fd
        >>> no = NO(Fd(a)*F(i)*F(b)*Fd(j))

        >>> no.iter_q_creators()
        <generator object... at 0x...>
        >>> list(no.iter_q_creators())
        [0, 1]
        >>> list(no.iter_q_annihilators())
        [3, 2]

        """
        # 获取操作符列表
        ops = self.args[0].args
        # 反向遍历操作符列表的索引
        iter = range(len(ops) - 1, -1, -1)
        # 对每个索引进行迭代
        for i in iter:
            # 如果操作符是湮灭算符
            if ops[i].is_q_annihilator:
                # 生成器，返回索引值
                yield i
            else:
                # 否则终止迭代
                break

    def iter_q_creators(self):
        """
        Iterates over the creation operators.

        Examples
        ========

        >>> from sympy import symbols
        >>> i, j = symbols('i j', below_fermi=True)
        >>> a, b = symbols('a b', above_fermi=True)
        >>> from sympy.physics.secondquant import NO, F, Fd
        >>> no = NO(Fd(a)*F(i)*F(b)*Fd(j))

        >>> no.iter_q_creators()
        <generator object... at 0x...>
        >>> list(no.iter_q_creators())
        [0, 1]
        >>> list(no.iter_q_annihilators())
        [3, 2]

        """

        # 获取操作符列表
        ops = self.args[0].args
        # 正向遍历操作符列表的索引
        iter = range(0, len(ops))
        # 对每个索引进行迭代
        for i in iter:
            # 如果操作符是产生算符
            if ops[i].is_q_creator:
                # 生成器，返回索引值
                yield i
            else:
                # 否则终止迭代
                break

    def get_subNO(self, i):
        """
        Returns a NO() without FermionicOperator at index i.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import F, NO
        >>> p, q, r = symbols('p,q,r')

        >>> NO(F(p)*F(q)*F(r)).get_subNO(1)
        NO(AnnihilateFermion(p)*AnnihilateFermion(r))

        """
        # 获取第一个参数
        arg0 = self.args[0]  # it's a Mul by definition of how it's created
        # 创建一个新的乘法表达式，排除指定索引处的操作符
        mul = arg0._new_rawargs(*(arg0.args[:i] + arg0.args[i + 1:]))
        # 返回新创建的 NO 对象
        return NO(mul)

    def _latex(self, printer):
        # 返回用 LaTeX 表示的字符串
        return "\\left\\{%s\\right\\}" % printer._print(self.args[0])

    def __repr__(self):
        # 返回表示对象的字符串
        return "NO(%s)" % self.args[0]

    def __str__(self):
        # 返回对象的字符串表示形式
        return ":%s:" % self.args[0]
# 计算费米算符 a 和 b 的缩并（contraction）。

def contraction(a, b):
    """
    Calculates contraction of Fermionic operators a and b.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.secondquant import F, Fd, contraction
    >>> p, q = symbols('p,q')
    >>> a, b = symbols('a,b', above_fermi=True)
    >>> i, j = symbols('i,j', below_fermi=True)

    A contraction is non-zero only if a quasi-creator is to the right of a
    quasi-annihilator:

    >>> contraction(F(a),Fd(b))
    KroneckerDelta(a, b)
    >>> contraction(Fd(i),F(j))
    KroneckerDelta(i, j)

    For general indices a non-zero result restricts the indices to below/above
    the fermi surface:

    >>> contraction(Fd(p),F(q))
    KroneckerDelta(_i, q)*KroneckerDelta(p, q)
    >>> contraction(F(p),Fd(q))
    KroneckerDelta(_a, q)*KroneckerDelta(p, q)

    Two creators or two annihilators always vanishes:

    >>> contraction(F(p),F(q))
    0
    >>> contraction(Fd(p),Fd(q))
    0

    """
    # 检查 a 和 b 是否为费米算符类型
    if isinstance(b, FermionicOperator) and isinstance(a, FermionicOperator):
        # 如果 a 是湮灭算符，b 是产生算符
        if isinstance(a, AnnihilateFermion) and isinstance(b, CreateFermion):
            # 如果 b 状态在费米面以下，返回零
            if b.state.assumptions0.get("below_fermi"):
                return S.Zero
            # 如果 a 状态在费米面以下，返回零
            if a.state.assumptions0.get("below_fermi"):
                return S.Zero
            # 如果 b 状态在费米面以上，返回 KroneckerDelta(a.state, b.state)
            if b.state.assumptions0.get("above_fermi"):
                return KroneckerDelta(a.state, b.state)
            # 如果 a 状态在费米面以上，返回 KroneckerDelta(a.state, b.state)
            if a.state.assumptions0.get("above_fermi"):
                return KroneckerDelta(a.state, b.state)

            # 否则返回 KroneckerDelta(a.state, b.state) * KroneckerDelta(b.state, Dummy('a', above_fermi=True))
            return (KroneckerDelta(a.state, b.state) *
                    KroneckerDelta(b.state, Dummy('a', above_fermi=True)))
        
        # 如果 a 是产生算符，b 是湮灭算符
        if isinstance(b, AnnihilateFermion) and isinstance(a, CreateFermion):
            # 如果 b 状态在费米面以上，返回零
            if b.state.assumptions0.get("above_fermi"):
                return S.Zero
            # 如果 a 状态在费米面以上，返回零
            if a.state.assumptions0.get("above_fermi"):
                return S.Zero
            # 如果 b 状态在费米面以下，返回 KroneckerDelta(a.state, b.state)
            if b.state.assumptions0.get("below_fermi"):
                return KroneckerDelta(a.state, b.state)
            # 如果 a 状态在费米面以下，返回 KroneckerDelta(a.state, b.state)
            if a.state.assumptions0.get("below_fermi"):
                return KroneckerDelta(a.state, b.state)

            # 否则返回 KroneckerDelta(a.state, b.state) * KroneckerDelta(b.state, Dummy('i', below_fermi=True))
            return (KroneckerDelta(a.state, b.state) *
                    KroneckerDelta(b.state, Dummy('i', below_fermi=True)))

        # 如果是两个湮灭算符或两个产生算符，返回零
        return S.Zero

    else:
        # 如果不是费米算符类型，则抛出异常
        t = ( isinstance(i, FermionicOperator) for i in (a, b) )
        raise ContractionAppliesOnlyToFermions(*t)


# 生成用于 SQ 算符的规范排序键的函数
def _sqkey(sq_operator):
    """Generates key for canonical sorting of SQ operators."""
    return sq_operator._sortkey()


# 将反交换费米算符按规范顺序排序的函数
def _sort_anticommuting_fermions(string1, key=_sqkey):
    """Sort fermionic operators to canonical order, assuming all pairs anticommute.

    Explanation
    ===========

    Uses a bidirectional bubble sort.  Items in string1 are not referenced
    so in principle they may be any comparable objects.   The sorting depends on the
    operators '>' and '=='.

    """
    # 使用双向冒泡排序，对 string1 中的费米算符按规范顺序排序
    # string1 中的项不被引用，因此原则上可以是任何可比较的对象
    # 排序依赖于操作符 '>' 和 '=='
    If the Pauli principle is violated, an exception is raised.
    如果违反了Pauli原理，则引发异常。

    Returns
    =======
    返回一个元组 (sorted_str, sign)

    sorted_str: 包含排序后操作符的列表
    sign: 整数，指示更改符号的次数
          （如果sign==0，则字符串已经排序）

    """

    verified = False  # 标志变量，指示是否通过了Pauli原理验证
    sign = 0  # 记录符号变化的次数
    rng = list(range(len(string1) - 1))  # 生成索引范围，用于循环遍历操作符列表
    rev = list(range(len(string1) - 3, -1, -1))  # 生成逆序索引范围，用于反向遍历操作符列表

    keys = list(map(key, string1))  # 对操作符列表中的每个元素应用key函数，生成对应的键列表
    key_val = dict(list(zip(keys, string1)))  # 将键列表和操作符列表组合成字典，键为操作符经过key函数处理后的结果，值为操作符本身

    while not verified:
        verified = True  # 假设通过了Pauli原理验证
        for i in rng:
            left = keys[i]  # 当前位置的操作符键
            right = keys[i + 1]  # 下一个位置的操作符键
            if left == right:
                raise ViolationOfPauliPrinciple([left, right])  # 如果出现相同的操作符键，引发异常
            if left > right:
                verified = False  # 未通过Pauli原理验证
                keys[i:i + 2] = [right, left]  # 交换操作符键的位置，以满足排序要求
                sign = sign + 1  # 增加符号变化的次数
        if verified:
            break  # 如果通过了Pauli原理验证，则退出循环
        for i in rev:
            left = keys[i]  # 当前位置的操作符键
            right = keys[i + 1]  # 下一个位置的操作符键
            if left == right:
                raise ViolationOfPauliPrinciple([left, right])  # 如果出现相同的操作符键，引发异常
            if left > right:
                verified = False  # 未通过Pauli原理验证
                keys[i:i + 2] = [right, left]  # 交换操作符键的位置，以满足排序要求
                sign = sign + 1  # 增加符号变化的次数

    string1 = [key_val[k] for k in keys]  # 根据排序后的操作符键列表重新构建排序后的操作符列表
    return (string1, sign)  # 返回排序后的操作符列表和符号变化次数
# 定义一个函数 evaluate_deltas，用于处理包含 KroneckerDelta 符号的表达式，假设使用爱因斯坦求和约定。
def evaluate_deltas(e):
    """
    We evaluate KroneckerDelta symbols in the expression assuming Einstein summation.

    Explanation
    ===========

    If one index is repeated it is summed over and in effect substituted with
    the other one. If both indices are repeated we substitute according to what
    is the preferred index.  this is determined by
    KroneckerDelta.preferred_index and KroneckerDelta.killable_index.

    In case there are no possible substitutions or if a substitution would
    imply a loss of information, nothing is done.

    In case an index appears in more than one KroneckerDelta, the resulting
    substitution depends on the order of the factors.  Since the ordering is platform
    dependent, the literal expression resulting from this function may be hard to
    predict.

    Examples
    ========

    We assume the following:

    >>> from sympy import symbols, Function, Dummy, KroneckerDelta
    >>> from sympy.physics.secondquant import evaluate_deltas
    >>> i,j = symbols('i j', below_fermi=True, cls=Dummy)
    >>> a,b = symbols('a b', above_fermi=True, cls=Dummy)
    >>> p,q = symbols('p q', cls=Dummy)
    >>> f = Function('f')
    >>> t = Function('t')

    The order of preference for these indices according to KroneckerDelta is
    (a, b, i, j, p, q).

    Trivial cases:

    >>> evaluate_deltas(KroneckerDelta(i,j)*f(i))       # d_ij f(i) -> f(j)
    f(_j)
    >>> evaluate_deltas(KroneckerDelta(i,j)*f(j))       # d_ij f(j) -> f(i)
    f(_i)
    >>> evaluate_deltas(KroneckerDelta(i,p)*f(p))       # d_ip f(p) -> f(i)
    f(_i)
    >>> evaluate_deltas(KroneckerDelta(q,p)*f(p))       # d_qp f(p) -> f(q)
    f(_q)
    >>> evaluate_deltas(KroneckerDelta(q,p)*f(q))       # d_qp f(q) -> f(p)
    f(_p)

    More interesting cases:

    >>> evaluate_deltas(KroneckerDelta(i,p)*t(a,i)*f(p,q))
    f(_i, _q)*t(_a, _i)
    >>> evaluate_deltas(KroneckerDelta(a,p)*t(a,i)*f(p,q))
    f(_a, _q)*t(_a, _i)
    >>> evaluate_deltas(KroneckerDelta(p,q)*f(p,q))
    f(_p, _p)

    Finally, here are some cases where nothing is done, because that would
    imply a loss of information:

    >>> evaluate_deltas(KroneckerDelta(i,p)*f(q))
    f(_q)*KroneckerDelta(_i, _p)
    >>> evaluate_deltas(KroneckerDelta(i,p)*f(i))
    f(_i)*KroneckerDelta(_i, _p)
    """

    # 我们只处理包含乘法对象的 Delta 符号
    # 对于一般的函数对象，我们不对参数中的 KroneckerDelta 进行求值，
    # 但在这里我们硬编码了一些例外情况来处理这个规则
    accepted_functions = (
        Add,  # 只接受加法对象
    )
    if isinstance(e, accepted_functions):
        # 对于加法对象，递归地对其参数中的每个元素调用 evaluate_deltas
        return e.func(*[evaluate_deltas(arg) for arg in e.args])
    # 如果表达式 e 是乘法表达式（Mul 类型）
    elif isinstance(e, Mul):
        # 找出所有的 delta 函数并计算每个自由符号的出现次数
        deltas = []  # 存储 delta 函数的列表
        indices = {}  # 记录每个符号的出现次数的字典
        for i in e.args:
            for s in i.free_symbols:
                if s in indices:
                    indices[s] += 1
                else:
                    indices[s] = 0  # 简化后续逻辑的计数方式
            if isinstance(i, KroneckerDelta):
                deltas.append(i)  # 将 KroneckerDelta 对象添加到 deltas 中

        for d in deltas:
            # 如果执行某些操作，并且存在多个 deltas，应递归处理结果表达式以正确处理
            if d.killable_index.is_Symbol and indices[d.killable_index]:
                e = e.subs(d.killable_index, d.preferred_index)
                if len(deltas) > 1:
                    return evaluate_deltas(e)  # 递归处理多个 deltas 的情况
            elif (d.preferred_index.is_Symbol and indices[d.preferred_index]
                  and d.indices_contain_equal_information):
                e = e.subs(d.preferred_index, d.killable_index)
                if len(deltas) > 1:
                    return evaluate_deltas(e)  # 递归处理多个 deltas 的情况
            else:
                pass  # 不需要任何操作的情况

        return e  # 返回处理后的表达式 e
    # 如果 e 不是乘法表达式，可能是符号或数字，无需处理
    else:
        return e  # 直接返回 e
    # 设置用于替换虚拟变量的设置

    # 如果 new_indices 为 True，则生成新的虚拟变量，否则重用现有的虚拟变量
    # pretty_indices 是一个字典，控制新虚拟变量的生成方式
    # 默认情况下，新虚拟变量按照 'above', 'below', 'general' 分组生成对应的标签
    # 如果字母用尽或某个索引组没有关键字，则会使用默认的虚拟变量生成器

    # 用于替换的虚拟变量列表
    replacing_dummies = []

    # 根据表达式中的 dummy 变量生成排序后的虚拟变量顺序
    ordered_dummies = _get_ordered_dummies(expr)

    # 遍历排序后的虚拟变量列表，为每个变量生成一个替换用的虚拟变量
    for dummy in ordered_dummies:
        # 根据索引的属性选择生成的虚拟变量标签
        if dummy.dummy_index_group in pretty_indices:
            dummy_label = pretty_indices[dummy.dummy_index_group]
        else:
            dummy_label = dummy.dummy_index_group[0]

        # 生成虚拟变量的名称，添加到替换虚拟变量列表中
        replacing_dummies.append(Dummy(dummy_label))

    # 创建一个字典，将原始虚拟变量映射到替换后的虚拟变量
    dummy_dict = {dummy: replacing_dummies[i] for i, dummy in enumerate(ordered_dummies)}

    # 使用 dummy_dict 替换表达式中的虚拟变量，得到最终替换后的表达式
    substituted_expr = expr.subs(dummy_dict)

    # 返回替换后的表达式
    return substituted_expr
    # 如果存在新的索引
    if new_indices:
        # 从 pretty_indices 字典中获取 'above' 对应的字符串，若不存在则为空字符串
        letters_above = pretty_indices.get('above', "")
        # 从 pretty_indices 字典中获取 'below' 对应的字符串，若不存在则为空字符串
        letters_below = pretty_indices.get('below', "")
        # 从 pretty_indices 字典中获取 'general' 对应的字符串，若不存在则为空字符串
        letters_general = pretty_indices.get('general', "")
        
        # 计算 letters_above、letters_below 和 letters_general 的长度
        len_above = len(letters_above)
        len_below = len(letters_below)
        len_general = len(letters_general)

        # 定义用于处理索引的内部函数 _i、_a、_p
        def _i(number):
            try:
                # 尝试获取 letters_below 中索引为 number 的元素
                return letters_below[number]
            except IndexError:
                # 如果索引超出范围，则返回 'i_' + (number - len_below) 的字符串
                return 'i_' + str(number - len_below)

        def _a(number):
            try:
                # 尝试获取 letters_above 中索引为 number 的元素
                return letters_above[number]
            except IndexError:
                # 如果索引超出范围，则返回 'a_' + (number - len_above) 的字符串
                return 'a_' + str(number - len_above)

        def _p(number):
            try:
                # 尝试获取 letters_general 中索引为 number 的元素
                return letters_general[number]
            except IndexError:
                # 如果索引超出范围，则返回 'p_' + (number - len_general) 的字符串
                return 'p_' + str(number - len_general)

    # 初始化三个空列表用于存放不同类型的 Dummy 对象
    aboves = []
    belows = []
    generals = []

    # 获取所有的 Dummy 对象并按照默认排序键排序，如果不需要新索引则直接按原顺序排序
    dummies = expr.atoms(Dummy)
    if not new_indices:
        dummies = sorted(dummies, key=default_sort_key)

    # 遍历所有的 Dummy 对象
    a = i = p = 0
    for d in dummies:
        # 获取 Dummy 对象的 assumptions0 属性
        assum = d.assumptions0

        # 根据 assumptions0 的不同属性判断应该使用哪种索引符号，并将 Dummy 对象添加到相应列表中
        if assum.get("above_fermi"):
            if new_indices:
                sym = _a(a)  # 获取 above 类型的索引符号
                a += 1
            l1 = aboves  # 将 Dummy 对象添加到 aboves 列表中
        elif assum.get("below_fermi"):
            if new_indices:
                sym = _i(i)  # 获取 below 类型的索引符号
                i += 1
            l1 = belows  # 将 Dummy 对象添加到 belows 列表中
        else:
            if new_indices:
                sym = _p(p)  # 获取 general 类型的索引符号
                p += 1
            l1 = generals  # 将 Dummy 对象添加到 generals 列表中

        # 如果需要新索引，则创建带有符号 sym 和 assumptions0 属性的 Dummy 对象并添加到对应列表中
        if new_indices:
            l1.append(Dummy(sym, **assum))
        else:
            l1.append(d)  # 否则直接添加原 Dummy 对象到列表中

    # 将表达式 expr 展开
    expr = expr.expand()
    # 获取表达式的所有项
    terms = Add.make_args(expr)
    # 初始化新的项列表
    new_terms = []
    for term in terms:
        # 使用迭代器准备 belows、aboves 和 generals 的迭代器
        i = iter(belows)
        a = iter(aboves)
        p = iter(generals)
        
        # 获取按顺序排列的虚拟量子数，并存储在 ordered 变量中
        ordered = _get_ordered_dummies(term)
        
        # 创建一个空字典 subsdict，用于存储替换后的虚拟量子数
        subsdict = {}
        
        # 遍历按顺序排列的虚拟量子数
        for d in ordered:
            # 如果虚拟量子数 d 在 assumptions0 中有 'below_fermi' 标记，则从 i 中取下一个值进行替换
            if d.assumptions0.get('below_fermi'):
                subsdict[d] = next(i)
            # 如果虚拟量子数 d 在 assumptions0 中有 'above_fermi' 标记，则从 a 中取下一个值进行替换
            elif d.assumptions0.get('above_fermi'):
                subsdict[d] = next(a)
            # 否则，从 p 中取下一个值进行替换
            else:
                subsdict[d] = next(p)
        
        # 创建一个空列表 subslist，用于存储最终的替换列表
        subslist = []
        
        # 创建一个空列表 final_subs，用于存储需要在 subslist 之后进行的替换
        final_subs = []
        
        # 遍历 subsdict 中的键值对
        for k, v in subsdict.items():
            # 如果 k 和 v 相等，则跳过此次循环
            if k == v:
                continue
            
            # 如果 v 在 subsdict 中
            if v in subsdict:
                # 检查替换序列是否可以迅速结束，如果是，则可以避免使用临时符号，确保正确的替换顺序
                if subsdict[v] in subsdict:
                    # 如果出现 (x, y) -> (y, x)，则需要创建一个临时变量 x
                    x = Dummy('x')
                    subslist.append((k, x))
                    final_subs.append((x, v))
                else:
                    # 如果出现 (x, y) -> (y, a)，则 x->y 必须最后完成，但在解决临时变量之前
                    final_subs.insert(0, (k, v))
            else:
                # 将 k 和 v 添加到 subslist 中
                subslist.append((k, v))
        
        # 将 final_subs 中的内容扩展到 subslist 中
        subslist.extend(final_subs)
        
        # 使用 subslist 替换 term 中的内容，并将结果添加到 new_terms 列表中
        new_terms.append(term.subs(subslist))
    
    # 将 new_terms 中的所有项相加并返回结果
    return Add(*new_terms)
class KeyPrinter(StrPrinter):
    """定义一个KeyPrinter类，继承自StrPrinter，用于打印只有相等对象在打印时相等的内容"""

    def _print_Dummy(self, expr):
        """重载StrPrinter类的_print_Dummy方法，返回Dummy对象的字符串表示形式"""
        return "(%s_%i)" % (expr.name, expr.dummy_index)


def __kprint(expr):
    """对给定表达式进行打印操作，返回其字符串表示形式"""
    p = KeyPrinter()
    return p.doprint(expr)


def _get_ordered_dummies(mul, verbose=False):
    """返回乘积mul中所有虚拟指标（dummies）按照规范顺序排序的列表。

    Explanation
    ===========

    规范排序的目的在于能够一致地对术语进行替换，从而简化等效术语。

    单凭虚拟指标的顺序无法确定两个术语是否等效。但是，通过有序虚拟指标进行一致的替换应该能够揭示等效性（或非等效性）。这也意味着，如果两个术语具有相同的虚拟指标序列，则（非）等效性应该已经显而易见。

    Strategy
    --------

    规范顺序由任意排序规则确定。对每个虚拟指标确定一个排序键，该键是依赖于所有存在该指标的因子的元组。因此，虚拟指标按照术语的缩并结构排序，而不仅仅是按虚拟指标符号本身排序。

    在为术语中的所有虚拟指标分配键后，我们检查是否存在相同的键，即无法排序的虚拟指标。如果找到任何这样的指标，我们调用一个特殊方法 _determine_ambiguous()，该方法将基于对 _get_ordered_dummies() 的递归调用确定唯一顺序。

    Key description
    ---------------

    排序键的高级描述：

        1. 虚拟指标的范围
        2. 与外部（非虚拟）指标的关系
        3. 指标在第一个因子中的位置
        4. 指标在第二个因子中的位置

    排序键是一个元组，包含以下组件：

        1. 一个字符，指示虚拟指标的范围（上述、以下或一般）
        2. 一个字符串列表，其中包含所有存在虚拟指标的因子的全掩码字符串表示形式。这里的掩码意味着虚拟指标用符号表示，以指示它是在费米以下、以上或一般。在此时不显示关于虚拟指标的其他信息。列表按字符串顺序排序。
        3. 一个整数，指示指标在第一个因子中的位置，按照第2点的排序。
        4. 一个整数，指示指标在第二个因子中的位置，按照第2点的排序。

    如果因子是AntiSymmetricTensor或SqOperator类型之一，则在项目3和4中指示指标位置为“upper”或“lower”（创建算符被视为上，湮灭算符被视为下）。

    如果掩码因子相同，则这两个因子无法排序。
    """
    pass  # 这里的函数主体由于示例中没有具体实现，因此保持空白
    """
    # 设置字典以避免在 key() 函数中重复计算
    args = Mul.make_args(mul)
    # 对每个因子创建包含虚拟符号的字典
    fac_dum = { fac: fac.atoms(Dummy) for fac in args }
    # 对每个因子创建包含字符串表示的字典
    fac_repr = { fac: __kprint(fac) for fac in args }
    # 获取所有虚拟符号的集合
    all_dums = set().union(*fac_dum.values())
    # 创建一个掩码字典，根据虚拟符号的 assumptions0 属性分配掩码值
    mask = {}
    for d in all_dums:
        if d.assumptions0.get('below_fermi'):
            mask[d] = '0'
        elif d.assumptions0.get('above_fermi'):
            mask[d] = '1'
        else:
            mask[d] = '2'
    # 创建一个虚拟符号的字符串表示字典
    dum_repr = {d: __kprint(d) for d in all_dums}

    # 定义用于排序的 key() 函数
    def _key(d):
        # 根据虚拟符号所在的因子结构，找到相关的其他虚拟符号
        dumstruct = [ fac for fac in fac_dum if d in fac_dum[fac] ]
        other_dums = set().union(*[fac_dum[fac] for fac in dumstruct])
        fac = dumstruct[-1]
        # 处理其他虚拟符号的掩码替换
        if other_dums is fac_dum[fac]:
            other_dums = fac_dum[fac].copy()
        other_dums.remove(d)
        masked_facs = [ fac_repr[fac] for fac in dumstruct ]
        for d2 in other_dums:
            masked_facs = [ fac.replace(dum_repr[d2], mask[d2])
                    for fac in masked_facs ]
        all_masked = [ fac.replace(dum_repr[d], mask[d])
                       for fac in masked_facs ]
        masked_facs = dict(list(zip(dumstruct, masked_facs)))

        # 对于排序不明确的虚拟符号
        if has_dups(all_masked):
            all_masked.sort()
            return mask[d], tuple(all_masked)  # positions are ambiguous

        # 根据完全掩码字符串对因子进行排序
        keydict = dict(list(zip(dumstruct, all_masked)))
        dumstruct.sort(key=lambda x: keydict[x])
        all_masked.sort()

        pos_val = []
        for fac in dumstruct:
            if isinstance(fac, AntiSymmetricTensor):
                if d in fac.upper:
                    pos_val.append('u')
                if d in fac.lower:
                    pos_val.append('l')
            elif isinstance(fac, Creator):
                pos_val.append('u')
            elif isinstance(fac, Annihilator):
                pos_val.append('l')
            elif isinstance(fac, NO):
                ops = [ op for op in fac if op.has(d) ]
                for op in ops:
                    if isinstance(op, Creator):
                        pos_val.append('u')
                    else:
                        pos_val.append('l')
            else:
                # 如果没有匹配的特殊情况，则回退到字符串表示中的位置
                facpos = -1
                while 1:
                    facpos = masked_facs[fac].find(dum_repr[d], facpos + 1)
                    if facpos == -1:
                        break
                    pos_val.append(facpos)
        return (mask[d], tuple(all_masked), pos_val[0], pos_val[-1])
    
    # 创建虚拟符号到排序键值的映射字典
    dumkey = dict(list(zip(all_dums, list(map(_key, all_dums)))))
    # 根据排序键值对所有虚拟符号进行排序
    result = sorted(all_dums, key=lambda x: dumkey[x])
    # 检查给定的 dumkey 值是否存在重复项，如果存在重复则进行处理
    if has_dups(iter(dumkey.values())):
        # 如果存在重复项，则创建一个默认字典用于存储非顺序集合
        unordered = defaultdict(set)
        # 遍历 dumkey 的每一对键值对 (d, k)，将其值 k 加入对应键 k 的集合中
        for d, k in dumkey.items():
            unordered[k].add(d)
        
        # 从 unordered 中删除集合长度小于 2 的键值对应的集合
        for k in [ k for k in unordered if len(unordered[k]) < 2 ]:
            del unordered[k]

        # 将 unordered 字典按键排序后，提取其值（集合），存入列表中
        unordered = [ unordered[k] for k in sorted(unordered) ]
        
        # 调用 _determine_ambiguous 函数处理多余的结果
        result = _determine_ambiguous(mul, result, unordered)
    
    # 返回处理后的结果
    return result
# 确定存在歧义的虚拟变量的方法
def _determine_ambiguous(term, ordered, ambiguous_groups):
    # 如果遇到一个存在歧义的虚拟变量，说明有两个或更多的收缩项之间无法在求和指标独立的情况下唯一排序。
    # 例如：
    #
    # Sum(p, q) v^{p, .}_{q, .}v^{q, .}_{p, .}
    #
    # 假设由 . 表示的指标是虚拟的且范围相同，这些因子无法排序，无法确定 p 和 q 的一致顺序。
    #
    # 此处采用的策略是，重新标记所有非歧义的虚拟变量为非虚拟符号，并再次调用 _get_ordered_dummies。这个过程
    # 应用于整个项，因此有可能从更深层次的递归级别再次调用 _determine_ambiguous()。

    # 如果没有已排序的虚拟变量，则中断递归
    all_ambiguous = set()
    for dummies in ambiguous_groups:
        all_ambiguous |= dummies
    all_ordered = set(ordered) - all_ambiguous
    if not all_ordered:
        # FIXME: 如果到达这里，表示没有已排序的虚拟变量。需要实现处理这种情况的方法。尽管如此，为了仍然返回
        # 一些有用的东西，我们任意选择第一个虚拟变量，并从这个选择中确定其余的变量。这种方法依赖于实际的虚拟标签，
        # 这违反了规范化过程的假设。需要一个更好的实现。
        group = [ d for d in ordered if d in ambiguous_groups[0] ]
        d = group[0]
        all_ordered.add(d)
        ambiguous_groups[0].remove(d)

    stored_counter = _symbol_factory._counter
    subslist = []
    # 为所有已排序的虚拟变量生成新的符号，并更新 term
    for d in [ d for d in ordered if d in all_ordered ]:
        nondum = _symbol_factory._next()
        subslist.append((d, nondum))
    newterm = term.subs(subslist)
    neworder = _get_ordered_dummies(newterm)
    _symbol_factory._set_counter(stored_counter)

    # 使用新信息更新已排序的列表
    for group in ambiguous_groups:
        ordered_group = [ d for d in neworder if d in group ]
        ordered_group.reverse()
        result = []
        for d in ordered:
            if d in group:
                result.append(ordered_group.pop())
            else:
                result.append(d)
        ordered = result
    return ordered


class _SymbolFactory:
    def __init__(self, label):
        self._counterVar = 0
        self._label = label

    def _set_counter(self, value):
        """
        将计数器设置为给定值。
        """
        self._counterVar = value

    @property
    def _counter(self):
        """
        返回当前计数器的值。
        """
        return self._counterVar

    def _next(self):
        """
        生成下一个符号并将计数器增加1。
        """
        s = Symbol("%s%i" % (self._label, self._counterVar))
        self._counterVar += 1
        return s
# 创建一个特定的符号工厂对象，用于生成特定的符号或标签
_symbol_factory = _SymbolFactory('_]"]_')  # most certainly a unique label

# 使用缓存功能装饰器，用于获取给定字符串的收缩（收缩表达式）列表
@cacheit
def _get_contractions(string1, keep_only_fully_contracted=False):
    """
    返回包含收缩项的 Add 对象。

    使用递归查找所有的收缩项。-- 内部辅助函数 --

    将在给定的 leftrange 和 rightrange 索引之间的 string1 中找到非零的收缩项。

    """

    # 是否应存储当前级别的收缩？
    if keep_only_fully_contracted and string1:
        result = []
    else:
        result = [NO(Mul(*string1))]

    for i in range(len(string1) - 1):
        for j in range(i + 1, len(string1)):

            # 执行收缩操作，生成收缩系数 c
            c = contraction(string1[i], string1[j])

            if c:
                sign = (j - i + 1) % 2
                if sign:
                    coeff = S.NegativeOne*c
                else:
                    coeff = c

                #
                # 调用下一级递归
                # ============================
                #
                # 现在我们需要在操作符中找到更多的收缩项
                #
                # oplist = string1[:i]+ string1[i+1:j] + string1[j+1:]
                #
                # 为了防止重复计数，我们不允许已经遇到的收缩项
                # 即 string1[:i] <---> string1[i+1:j]
                # 和   string1[:i] <---> string1[j+1:] 之间的收缩项。
                #
                # 这留下了以下情况：
                oplist = string1[i + 1:j] + string1[j + 1:]

                if oplist:

                    result.append(coeff*NO(
                        Mul(*string1[:i])*_get_contractions( oplist,
                            keep_only_fully_contracted=keep_only_fully_contracted)))

                else:
                    result.append(coeff*NO( Mul(*string1[:i])))

        if keep_only_fully_contracted:
            break   # 下一次迭代结束时，左侧操作符 string1[0] 保持未收缩状态

    return Add(*result)


def wicks(e, **kw_args):
    """
    使用 Wick 定理返回表达式的正规序等价形式。

    示例
    ========

    >>> from sympy import symbols, Dummy
    >>> from sympy.physics.secondquant import wicks, F, Fd
    >>> p, q, r = symbols('p,q,r')
    >>> wicks(Fd(p)*F(q))
    KroneckerDelta(_i, q)*KroneckerDelta(p, q) + NO(CreateFermion(p)*AnnihilateFermion(q))

    默认情况下，表达式会被展开：

    >>> wicks(F(p)*(F(q)+F(r)))
    NO(AnnihilateFermion(p)*AnnihilateFermion(q)) + NO(AnnihilateFermion(p)*AnnihilateFermion(r))

    通过关键字 'keep_only_fully_contracted=True'，只返回完全收缩的项。

    按请求顺序简化结果：
     -- 评估 KroneckerDelta 函数
     -- 一致地替换虚拟变量跨多项式

    >>> p, q, r = symbols('p q r', cls=Dummy)

    """
    # 对给定表达式应用 Wick 定理进行缩并，保留完全缩并的项
    >>> wicks(Fd(p)*(F(q)+F(r)), keep_only_fully_contracted=True)
    KroneckerDelta(_i, _q)*KroneckerDelta(_p, _q) + KroneckerDelta(_i, _r)*KroneckerDelta(_p, _r)

    """

    # 如果表达式为空，则返回零
    if not e:
        return S.Zero

    # 默认选项设置，可以通过额外的关键字参数进行更新
    opts = {
        'simplify_kronecker_deltas': False,   # 是否简化 Kronecker δ 函数
        'expand': True,                       # 是否展开表达式
        'simplify_dummies': False,            # 是否简化哑指标
        'keep_only_fully_contracted': False   # 是否仅保留完全缩并的项
    }
    opts.update(kw_args)  # 使用传入的关键字参数更新选项

    # 检查表达式是否已经是正规序的
    if isinstance(e, NO):
        if opts['keep_only_fully_contracted']:
            return S.Zero
        else:
            return e
    elif isinstance(e, FermionicOperator):
        if opts['keep_only_fully_contracted']:
            return S.Zero
        else:
            return e

    # 将表达式展开并应用 Wick 缩并
    e = e.doit(wicks=True)

    # 确保只有一个项需要考虑
    e = e.expand()
    if isinstance(e, Add):
        if opts['simplify_dummies']:
            # 如果需要简化哑指标，则替换每个加法项的哑指标
            return substitute_dummies(Add(*[ wicks(term, **kw_args) for term in e.args]))
        else:
            # 否则对每个加法项应用 Wick 缩并
            return Add(*[ wicks(term, **kw_args) for term in e.args])

    # 对乘法项进行处理
    if isinstance(e, Mul):

        # 在递归开始之前，将可交换部分因子分离出来
        c_part = []  # 可交换部分列表
        string1 = []  # 非可交换部分列表
        for factor in e.args:
            if factor.is_commutative:
                c_part.append(factor)
            else:
                string1.append(factor)
        n = len(string1)

        # 处理特殊情况
        if n == 0:
            result = e
        elif n == 1:
            if opts['keep_only_fully_contracted']:
                return S.Zero
            else:
                result = e

        else:  # 处理非平凡情况

            if isinstance(string1[0], BosonicOperator):
                raise NotImplementedError

            string1 = tuple(string1)

            # 递归处理更高阶的缩并
            result = _get_contractions(string1,
                keep_only_fully_contracted=opts['keep_only_fully_contracted'] )
            result = Mul(*c_part)*result

        if opts['expand']:
            result = result.expand()
        if opts['simplify_kronecker_deltas']:
            # 如果需要简化 Kronecker δ 函数，则对结果应用简化
            result = evaluate_deltas(result)

        return result

    # 如果没有需要处理的情况，则直接返回表达式本身
    # 这一般发生在表达式不包含 Wick 缩并的时候
    return e
class PermutationOperator(Expr):
    """
    Represents the index permutation operator P(ij).

    P(ij)*f(i)*g(j) = f(i)*g(j) - f(j)*g(i)
    """
    is_commutative = True  # 声明该类对象是可交换的

    def __new__(cls, i, j):
        i, j = sorted(map(sympify, (i, j)), key=default_sort_key)
        obj = Basic.__new__(cls, i, j)  # 创建新的对象实例，确保 i < j
        return obj

    def get_permuted(self, expr):
        """
        Returns -expr with permuted indices.

        Explanation
        ===========

        >>> from sympy import symbols, Function
        >>> from sympy.physics.secondquant import PermutationOperator
        >>> p,q = symbols('p,q')
        >>> f = Function('f')
        >>> PermutationOperator(p,q).get_permuted(f(p,q))
        -f(q, p)

        """
        i = self.args[0]  # 获取索引 i
        j = self.args[1]  # 获取索引 j
        if expr.has(i) and expr.has(j):  # 检查表达式是否包含索引 i 和 j
            tmp = Dummy()  # 创建一个虚拟符号
            expr = expr.subs(i, tmp)  # 将表达式中的 i 替换为 tmp
            expr = expr.subs(j, i)    # 将表达式中的 j 替换为 i
            expr = expr.subs(tmp, j)  # 将表达式中的 tmp 替换为 j
            return S.NegativeOne * expr  # 返回结果乘以 -1
        else:
            return expr  # 如果表达式中没有 i 或者 j，直接返回原表达式

    def _latex(self, printer):
        return "P(%s%s)" % self.args  # 返回 LaTeX 表示的字符串 P(ij)


def simplify_index_permutations(expr, permutation_operators):
    """
    Performs simplification by introducing PermutationOperators where appropriate.

    Explanation
    ===========

    Schematically:
        [abij] - [abji] - [baij] + [baji] ->  P(ab)*P(ij)*[abij]

    permutation_operators is a list of PermutationOperators to consider.

    If permutation_operators=[P(ab),P(ij)] we will try to introduce the
    permutation operators P(ij) and P(ab) in the expression.  If there are other
    possible simplifications, we ignore them.

    >>> from sympy import symbols, Function
    >>> from sympy.physics.secondquant import simplify_index_permutations
    >>> from sympy.physics.secondquant import PermutationOperator
    >>> p,q,r,s = symbols('p,q,r,s')
    >>> f = Function('f')
    >>> g = Function('g')

    >>> expr = f(p)*g(q) - f(q)*g(p); expr
    f(p)*g(q) - f(q)*g(p)
    >>> simplify_index_permutations(expr,[PermutationOperator(p,q)])
    f(p)*g(q)*PermutationOperator(p, q)

    >>> PermutList = [PermutationOperator(p,q),PermutationOperator(r,s)]
    >>> expr = f(p,r)*g(q,s) - f(q,r)*g(p,s) + f(q,s)*g(p,r) - f(p,s)*g(q,r)
    >>> simplify_index_permutations(expr,PermutList)
    f(p, r)*g(q, s)*PermutationOperator(p, q)*PermutationOperator(r, s)

    """

    def _get_indices(expr, ind):
        """
        Collects indices recursively in predictable order.
        """
        result = []
        for arg in expr.args:
            if arg in ind:
                result.append(arg)
            else:
                if arg.args:
                    result.extend(_get_indices(arg, ind))
        return result

    def _choose_one_to_keep(a, b, ind):
        # we keep the one where indices in ind are in order ind[0] < ind[1]
        return min(a, b, key=lambda x: default_sort_key(_get_indices(x, ind)))

    expr = expr.expand()  # 展开表达式
    # 如果表达式是加法表达式的实例
    if isinstance(expr, Add):
        # 提取表达式中的所有项，并转换为集合以确保唯一性
        terms = set(expr.args)

        # 遍历所有的排列操作符
        for P in permutation_operators:
            # 新的项集合
            new_terms = set()
            # 暂时保留的项集合
            on_hold = set()
            
            # 当仍有未处理的项时继续循环
            while terms:
                # 弹出一个项
                term = terms.pop()
                # 获取该项的排列结果
                permuted = P.get_permuted(term)
                
                # 如果排列后的项已经在处理中的项集合或者暂时保留的项集合中
                if permuted in terms | on_hold:
                    try:
                        terms.remove(permuted)
                    except KeyError:
                        on_hold.remove(permuted)
                    # 选择保留哪个项，并加入新的项集合
                    keep = _choose_one_to_keep(term, permuted, P.args)
                    new_terms.add(P * keep)
                else:
                    # 否则，需要考虑一些项可能需要第二次机会，因为排列后的项可能已经有了规范的虚拟顺序
                    permuted1 = permuted
                    # 对排列后的项执行替换虚拟变量的操作
                    permuted = substitute_dummies(permuted)
                    
                    # 如果替换前后的项相同
                    if permuted1 == permuted:
                        # 将原始项放入暂时保留的集合中
                        on_hold.add(term)
                    # 否则，如果排列后的项在处理中的项集合或者暂时保留的项集合中
                    elif permuted in terms | on_hold:
                        try:
                            terms.remove(permuted)
                        except KeyError:
                            on_hold.remove(permuted)
                        # 选择保留哪个项，并加入新的项集合
                        keep = _choose_one_to_keep(term, permuted, P.args)
                        new_terms.add(P * keep)
                    else:
                        # 否则将原始项加入新的项集合中
                        new_terms.add(term)
            
            # 更新待处理的项集合为新的项集合加上暂时保留的项集合
            terms = new_terms | on_hold
        
        # 返回一个新的加法表达式，其中包含更新后的所有项
        return Add(*terms)
    
    # 如果表达式不是加法表达式的实例，直接返回原始表达式
    return expr
```
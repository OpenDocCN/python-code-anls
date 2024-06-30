# `D:\src\scipysrc\sympy\sympy\physics\quantum\boson.py`

```
# 导入必要的模块和类
from sympy.core.mul import Mul  # 导入 Mul 类，用于处理乘法表达式
from sympy.core.numbers import Integer  # 导入 Integer 类，用于处理整数
from sympy.core.singleton import S  # 导入 S 单例，用于表示常用数学符号
from sympy.functions.elementary.complexes import conjugate  # 导入 conjugate 函数，用于复数的共轭
from sympy.functions.elementary.exponential import exp  # 导入 exp 函数，用于指数函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入 sqrt 函数，用于平方根函数
from sympy.physics.quantum import Operator  # 导入 Operator 类，量子算符的基类
from sympy.physics.quantum import HilbertSpace, FockSpace, Ket, Bra, IdentityOperator  # 导入量子力学相关类和函数
from sympy.functions.special.tensor_functions import KroneckerDelta  # 导入 KroneckerDelta 函数，用于克罗内克 δ 符号

__all__ = [
    'BosonOp',
    'BosonFockKet',
    'BosonFockBra',
    'BosonCoherentKet',
    'BosonCoherentBra'
]

class BosonOp(Operator):
    """表示满足 [a, Dagger(a)] == 1 的玻色算符。

    Parameters
    ==========

    name : str
        标记玻色模式的字符串。

    annihilation : bool
        指示玻色算符是否是湮灭算符（True，默认值）或产生算符（False）的布尔值。

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, Commutator
    >>> from sympy.physics.quantum.boson import BosonOp
    >>> a = BosonOp("a")
    >>> Commutator(a, Dagger(a)).doit()
    1
    """

    @property
    def name(self):
        return self.args[0]  # 返回算符的名称

    @property
    def is_annihilation(self):
        return bool(self.args[1])  # 返回算符是否为湮灭算符的布尔值

    @classmethod
    def default_args(self):
        return ("a", True)  # 默认参数，玻色算符名为 "a"，为湮灭算符

    def __new__(cls, *args, **hints):
        if not len(args) in [1, 2]:
            raise ValueError('1 or 2 parameters expected, got %s' % args)

        if len(args) == 1:
            args = (args[0], S.One)  # 如果参数长度为1，第二个参数默认为 1

        if len(args) == 2:
            args = (args[0], Integer(args[1]))  # 如果参数长度为2，第二个参数转换为整数类型

        return Operator.__new__(cls, *args)  # 调用 Operator 的构造方法创建新的玻色算符对象

    def _eval_commutator_BosonOp(self, other, **hints):
        if self.name == other.name:
            # [a^\dagger, a] = -1
            if not self.is_annihilation and other.is_annihilation:
                return S.NegativeOne  # 如果一个是产生算符，另一个是湮灭算符，它们的对易子为 -1

        elif 'independent' in hints and hints['independent']:
            # [a, b] = 0
            return S.Zero  # 如果算符名不同且独立，它们的对易子为 0

        return None  # 其他情况返回空

    def _eval_commutator_FermionOp(self, other, **hints):
        return S.Zero  # 玻色算符与费米算符的对易子为 0

    def _eval_anticommutator_BosonOp(self, other, **hints):
        if 'independent' in hints and hints['independent']:
            # {a, b} = 2 * a * b, because [a, b] = 0
            return 2 * self * other  # 如果算符名独立，它们的反对易子为 2 * a * b

        return None  # 其他情况返回空

    def _eval_adjoint(self):
        return BosonOp(str(self.name), not self.is_annihilation)  # 返回算符的伴随算符

    def __mul__(self, other):
        if other == IdentityOperator(2):
            return self  # 如果乘以单位算符，返回自身

        if isinstance(other, Mul):
            args1 = tuple(arg for arg in other.args if arg.is_commutative)
            args2 = tuple(arg for arg in other.args if not arg.is_commutative)
            x = self
            for y in args2:
                x = x * y  # 对非交换参数依次乘以自身
            return Mul(*args1) * x  # 返回结果乘以交换参数

        return Mul(self, other)  # 其他情况返回乘积
    # 如果是湮灭操作符，返回简单的 LaTeX 格式字符串表示
    def _print_contents_latex(self, printer, *args):
        if self.is_annihilation:
            return r'{%s}' % str(self.name)
        else:
            # 如果不是湮灭操作符，返回带有“†”符号的 LaTeX 格式字符串表示
            return r'{{%s}^\dagger}' % str(self.name)

    # 根据对象是否为湮灭操作符，返回不同格式的字符串表示
    def _print_contents(self, printer, *args):
        if self.is_annihilation:
            return r'%s' % str(self.name)
        else:
            # 如果不是湮灭操作符，返回带有“Dagger()”的字符串表示
            return r'Dagger(%s)' % str(self.name)

    # 根据对象是否为湮灭操作符，返回不同格式的漂亮打印结果
    def _print_contents_pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm
        # 打印对象的第一个参数的漂亮形式
        pform = printer._print(self.args[0], *args)
        if self.is_annihilation:
            # 如果是湮灭操作符，直接返回漂亮形式
            return pform
        else:
            # 如果不是湮灭操作符，返回漂亮形式加上“†”符号的指数形式
            return pform ** prettyForm('\N{DAGGER}')
class BosonFockKet(Ket):
    """Fock state ket for a bosonic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """

    def __new__(cls, n):
        # 使用父类 Ket 的构造函数创建新的实例，传入 Fock 状态数 n
        return Ket.__new__(cls, n)

    @property
    def n(self):
        # 返回实例对象的标签中的第一个元素，即 Fock 状态数 n
        return self.label[0]

    @classmethod
    def dual_class(self):
        # 返回这个类的对偶类 BosonFockBra
        return BosonFockBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        # 返回 FockSpace()，表示这个 Fock 状态所属的 Hilbert 空间
        return FockSpace()

    def _eval_innerproduct_BosonFockBra(self, bra, **hints):
        # 计算这个 Fock 态与给定 Bra 态 bra 的内积
        return KroneckerDelta(self.n, bra.n)

    def _apply_from_right_to_BosonOp(self, op, **options):
        # 如果操作符 op 是湮灭算符，则返回湮灭操作作用后的新的 BosonFockKet 实例
        if op.is_annihilation:
            return sqrt(self.n) * BosonFockKet(self.n - 1)
        else:
            # 如果操作符 op 不是湮灭算符，则返回相应产生操作作用后的新的 BosonFockKet 实例
            return sqrt(self.n + 1) * BosonFockKet(self.n + 1)


class BosonFockBra(Bra):
    """Fock state bra for a bosonic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """

    def __new__(cls, n):
        # 使用父类 Bra 的构造函数创建新的实例，传入 Fock 状态数 n
        return Bra.__new__(cls, n)

    @property
    def n(self):
        # 返回实例对象的标签中的第一个元素，即 Fock 状态数 n
        return self.label[0]

    @classmethod
    def dual_class(self):
        # 返回这个类的对偶类 BosonFockKet
        return BosonFockKet

    @classmethod
    def _eval_hilbert_space(cls, label):
        # 返回 FockSpace()，表示这个 Fock 状态所属的 Hilbert 空间
        return FockSpace()


class BosonCoherentKet(Ket):
    """Coherent state ket for a bosonic mode.

    Parameters
    ==========

    alpha : Number, Symbol
        The complex amplitude of the coherent state.

    """

    def __new__(cls, alpha):
        # 使用父类 Ket 的构造函数创建新的实例，传入 alpha 作为标签
        return Ket.__new__(cls, alpha)

    @property
    def alpha(self):
        # 返回实例对象的标签中的第一个元素，即 alpha，表示相干态的复振幅
        return self.label[0]

    @classmethod
    def dual_class(self):
        # 返回这个类的对偶类 BosonCoherentBra
        return BosonCoherentBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        # 返回 HilbertSpace()，表示这个相干态所属的 Hilbert 空间
        return HilbertSpace()

    def _eval_innerproduct_BosonCoherentBra(self, bra, **hints):
        # 计算这个相干态与给定 Bra 态 bra 的内积
        if self.alpha == bra.alpha:
            return S.One
        else:
            # 计算一般情况下相干态与其对偶态 bra 的内积
            return exp(-(abs(self.alpha)**2 + abs(bra.alpha)**2 - 2 * conjugate(bra.alpha) * self.alpha)/2)

    def _apply_from_right_to_BosonOp(self, op, **options):
        # 如果操作符 op 是湮灭算符，则返回相干态 alpha 乘以自身
        if op.is_annihilation:
            return self.alpha * self
        else:
            # 如果操作符 op 不是湮灭算符，则返回 None，表示不进行操作
            return None


class BosonCoherentBra(Bra):
    """Coherent state bra for a bosonic mode.

    Parameters
    ==========

    alpha : Number, Symbol
        The complex amplitude of the coherent state.

    """

    def __new__(cls, alpha):
        # 使用父类 Bra 的构造函数创建新的实例，传入 alpha 作为标签
        return Bra.__new__(cls, alpha)

    @property
    def alpha(self):
        # 返回实例对象的标签中的第一个元素，即 alpha，表示相干态的复振幅
        return self.label[0]

    @classmethod
    def dual_class(self):
        # 返回这个类的对偶类 BosonCoherentKet
        return BosonCoherentKet

    def _apply_operator_BosonOp(self, op, **options):
        # 如果操作符 op 不是湮灭算符，则返回相干态 alpha 乘以自身
        if not op.is_annihilation:
            return self.alpha * self
        else:
            # 如果操作符 op 是湮灭算符，则返回 None，表示不进行操作
            return None
```
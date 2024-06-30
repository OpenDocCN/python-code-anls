# `D:\src\scipysrc\sympy\sympy\physics\quantum\fermion.py`

```
# 导入必要的模块和类
"""Fermionic quantum operators."""

from sympy.core.numbers import Integer                   # 导入整数类 Integer
from sympy.core.singleton import S                       # 导入单例类 S
from sympy.physics.quantum import Operator               # 导入量子操作符基类 Operator
from sympy.physics.quantum import HilbertSpace, Ket, Bra  # 导入 Hilbert 空间、Ket 和 Bra 类
from sympy.functions.special.tensor_functions import KroneckerDelta  # 导入 Kronecker δ 函数


__all__ = [
    'FermionOp',
    'FermionFockKet',
    'FermionFockBra'
]

# 定义 FermionOp 类，继承自 Operator 类
class FermionOp(Operator):
    """A fermionic operator that satisfies {c, Dagger(c)} == 1.

    Parameters
    ==========

    name : str
        A string that labels the fermionic mode.

    annihilation : bool
        A bool that indicates if the fermionic operator is an annihilation
        (True, default value) or creation operator (False)

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, AntiCommutator
    >>> from sympy.physics.quantum.fermion import FermionOp
    >>> c = FermionOp("c")
    >>> AntiCommutator(c, Dagger(c)).doit()
    1
    """
    
    @property
    def name(self):
        return self.args[0]  # 返回 fermion 操作符的名称
    
    @property
    def is_annihilation(self):
        return bool(self.args[1])  # 返回布尔值，表示这是否是湮灭算符
    
    @classmethod
    def default_args(self):
        return ("c", True)  # 返回默认参数元组 ("c", True)
    
    def __new__(cls, *args, **hints):
        if not len(args) in [1, 2]:
            raise ValueError('1 or 2 parameters expected, got %s' % args)
        
        if len(args) == 1:
            args = (args[0], S.One)  # 如果只有一个参数，默认第二个参数为 1
        
        if len(args) == 2:
            args = (args[0], Integer(args[1]))  # 将第二个参数转换为整数类型
        
        return Operator.__new__(cls, *args)  # 调用父类 Operator 的构造方法
    
    def _eval_commutator_FermionOp(self, other, **hints):
        if 'independent' in hints and hints['independent']:
            # [c, d] = 0，如果是独立的算符，对易子为零
            return S.Zero
        
        return None
    
    def _eval_anticommutator_FermionOp(self, other, **hints):
        if self.name == other.name:
            # {a^\dagger, a} = 1，反对易关系，湮灭算符和创建算符的反对易关系为 1
            if not self.is_annihilation and other.is_annihilation:
                return S.One
        
        elif 'independent' in hints and hints['independent']:
            # {c, d} = 2 * c * d，因为独立算符的对易子为零，所以反对易子可以简化为 2 * c * d
            return 2 * self * other
        
        return None
    
    def _eval_anticommutator_BosonOp(self, other, **hints):
        # because fermions and bosons commute，费米子和玻色子的反对易关系为零
        return 2 * self * other
    
    def _eval_commutator_BosonOp(self, other, **hints):
        return S.Zero  # 玻色子的对易关系为零
    
    def _eval_adjoint(self):
        return FermionOp(str(self.name), not self.is_annihilation)  # 返回算符的伴随算符
    
    def _print_contents_latex(self, printer, *args):
        if self.is_annihilation:
            return r'{%s}' % str(self.name)  # 打印 LaTeX 格式的湮灭算符表示
        else:
            return r'{{%s}^\dagger}' % str(self.name)  # 打印 LaTeX 格式的创建算符表示
    
    def _print_contents(self, printer, *args):
        if self.is_annihilation:
            return r'%s' % str(self.name)  # 打印湮灭算符的表示
        else:
            return r'Dagger(%s)' % str(self.name)  # 打印创建算符的表示
    # 定义一个方法用于将操作符内容以漂亮的形式打印出来
    def _print_contents_pretty(self, printer, *args):
        # 导入需要的打印漂亮格式的类
        from sympy.printing.pretty.stringpict import prettyForm
        # 通过打印机对象打印操作符的第一个参数内容，并返回打印结果
        pform = printer._print(self.args[0], *args)
        # 如果操作符是湮灭算符，则直接返回打印结果
        if self.is_annihilation:
            return pform
        else:
            # 否则返回打印结果的幂运算，底数为打印结果，指数为'\N{DAGGER}'（表示Dagger符号）
            return pform**prettyForm('\N{DAGGER}')

    # 定义一个方法用于求解操作符的幂次运算
    def _eval_power(self, exp):
        # 导入需要的符号常量类
        from sympy.core.singleton import S
        # 如果指数为0，则返回单位元
        if exp == 0:
            return S.One
        # 如果指数为1，则返回自身
        elif exp == 1:
            return self
        # 如果指数大于1且为整数，则返回零（因为这是费米子算符的特性）
        elif (exp > 1) == True and exp.is_integer == True:
            return S.Zero
        # 如果指数小于0或者不为整数，则抛出值错误异常
        elif (exp < 0) == True or exp.is_integer == False:
            raise ValueError("Fermionic operators can only be raised to a"
                " positive integer power")
        # 默认情况下调用父类方法求解幂次运算
        return Operator._eval_power(self, exp)
class FermionFockKet(Ket):
    """Fock state ket for a fermionic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """

    def __new__(cls, n):
        # 检查 Fock 状态数是否为 0 或 1，否则引发值错误异常
        if n not in (0, 1):
            raise ValueError("n must be 0 or 1")
        # 调用父类的构造函数来创建实例
        return Ket.__new__(cls, n)

    @property
    def n(self):
        # 返回该 Fock 态的状态数
        return self.label[0]

    @classmethod
    def dual_class(self):
        # 返回这个类的对偶类是 FermionFockBra
        return FermionFockBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        # 返回与这个态相关的 Hilbert 空间
        return HilbertSpace()

    def _eval_innerproduct_FermionFockBra(self, bra, **hints):
        # 计算该态与给定 FermionFockBra 的内积，返回 KroneckerDelta 函数的结果
        return KroneckerDelta(self.n, bra.n)

    def _apply_from_right_to_FermionOp(self, op, **options):
        # 根据 FermionOp 的作用类型，对该态施加作用
        if op.is_annihilation:
            # 如果是湮灭算符
            if self.n == 1:
                return FermionFockKet(0)
            else:
                return S.Zero
        else:
            # 如果是创生算符
            if self.n == 0:
                return FermionFockKet(1)
            else:
                return S.Zero


class FermionFockBra(Bra):
    """Fock state bra for a fermionic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """

    def __new__(cls, n):
        # 检查 Fock 状态数是否为 0 或 1，否则引发值错误异常
        if n not in (0, 1):
            raise ValueError("n must be 0 or 1")
        # 调用父类的构造函数来创建实例
        return Bra.__new__(cls, n)

    @property
    def n(self):
        # 返回该 Fock 态的状态数
        return self.label[0]

    @classmethod
    def dual_class(self):
        # 返回这个类的对偶类是 FermionFockKet
        return FermionFockKet
```
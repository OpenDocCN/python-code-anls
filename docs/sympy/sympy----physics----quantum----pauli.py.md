# `D:\src\scipysrc\sympy\sympy\physics\quantum\pauli.py`

```
"""Pauli operators and states"""

# 导入必要的类和函数
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.physics.quantum import Operator, Ket, Bra
from sympy.physics.quantum import ComplexSpace
from sympy.matrices import Matrix
from sympy.functions.special.tensor_functions import KroneckerDelta

# 导出的类和函数名列表
__all__ = [
    'SigmaX', 'SigmaY', 'SigmaZ', 'SigmaMinus', 'SigmaPlus', 'SigmaZKet',
    'SigmaZBra', 'qsimplify_pauli'
]

# Pauli sigma 操作符的基类
class SigmaOpBase(Operator):
    """Pauli sigma operator, base class"""

    @property
    def name(self):
        return self.args[0]

    @property
    def use_name(self):
        return bool(self.args[0]) is not False

    @classmethod
    def default_args(self):
        return (False,)

    def __new__(cls, *args, **hints):
        return Operator.__new__(cls, *args, **hints)

    # 与 BosonOp 的对易子
    def _eval_commutator_BosonOp(self, other, **hints):
        return S.Zero


# Pauli sigma x 操作符类
class SigmaX(SigmaOpBase):
    """Pauli sigma x operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympy.physics.quantum import represent
    >>> from sympy.physics.quantum.pauli import SigmaX
    >>> sx = SigmaX()
    >>> sx
    SigmaX()
    >>> represent(sx)
    Matrix([
    [0, 1],
    [1, 0]])
    """

    def __new__(cls, *args, **hints):
        return SigmaOpBase.__new__(cls, *args, **hints)

    # 与 SigmaY 的对易子
    def _eval_commutator_SigmaY(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return 2 * I * SigmaZ(self.name)

    # 与 SigmaZ 的对易子
    def _eval_commutator_SigmaZ(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return - 2 * I * SigmaY(self.name)

    # 与 BosonOp 的对易子
    def _eval_commutator_BosonOp(self, other, **hints):
        return S.Zero

    # 与 SigmaY 的反对易子
    def _eval_anticommutator_SigmaY(self, other, **hints):
        return S.Zero

    # 与 SigmaZ 的反对易子
    def _eval_anticommutator_SigmaZ(self, other, **hints):
        return S.Zero

    # 共轭操作
    def _eval_adjoint(self):
        return self

    # 输出 LaTeX 格式的表示
    def _print_contents_latex(self, printer, *args):
        if self.use_name:
            return r'{\sigma_x^{(%s)}}' % str(self.name)
        else:
            return r'{\sigma_x}'

    # 输出字符串格式的表示
    def _print_contents(self, printer, *args):
        return 'SigmaX()'

    # 计算幂次方操作
    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return SigmaX(self.name).__pow__(int(e) % 2)

    # 默认基础下的表示
    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[0, 1], [1, 0]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')

# SigmaY 类的定义将在这里继续，但超出了示例的范围，不在此处展示
    """Pauli sigma y operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympy.physics.quantum import represent
    >>> from sympy.physics.quantum.pauli import SigmaY
    >>> sy = SigmaY()
    >>> sy
    SigmaY()
    >>> represent(sy)
    Matrix([
    [0, -I],
    [I,  0]])
    """

    # 定义 Pauli sigma y 算符类
    def __new__(cls, *args, **hints):
        # 调用基类 SigmaOpBase 的构造方法创建实例
        return SigmaOpBase.__new__(cls, *args)

    # 计算 Pauli sigma y 算符与 Pauli sigma z 算符的对易子
    def _eval_commutator_SigmaZ(self, other, **hints):
        # 如果两个算符的名称不同，则对易子为零
        if self.name != other.name:
            return S.Zero
        else:
            # 如果名称相同，返回对易子结果：2 * I * SigmaX(self.name)
            return 2 * I * SigmaX(self.name)

    # 计算 Pauli sigma y 算符与 Pauli sigma x 算符的对易子
    def _eval_commutator_SigmaX(self, other, **hints):
        # 如果两个算符的名称不同，则对易子为零
        if self.name != other.name:
            return S.Zero
        else:
            # 如果名称相同，返回对易子结果：- 2 * I * SigmaZ(self.name)
            return - 2 * I * SigmaZ(self.name)

    # 计算 Pauli sigma y 算符与 Pauli sigma x 算符的反对易子
    def _eval_anticommutator_SigmaX(self, other, **hints):
        # 总是返回零，因为 Pauli sigma y 与 Pauli sigma x 的反对易子为零
        return S.Zero

    # 计算 Pauli sigma y 算符与 Pauli sigma z 算符的反对易子
    def _eval_anticommutator_SigmaZ(self, other, **hints):
        # 总是返回零，因为 Pauli sigma y 与 Pauli sigma z 的反对易子为零
        return S.Zero

    # 返回 Pauli sigma y 算符的共轭
    def _eval_adjoint(self):
        return self

    # 返回以 LaTeX 格式打印 Pauli sigma y 算符内容
    def _print_contents_latex(self, printer, *args):
        # 如果使用名称，返回带名称的 LaTeX 格式
        if self.use_name:
            return r'{\sigma_y^{(%s)}}' % str(self.name)
        else:
            # 否则返回默认的 LaTeX 格式
            return r'{\sigma_y}'

    # 返回打印 Pauli sigma y 算符内容的字符串表示
    def _print_contents(self, printer, *args):
        return 'SigmaY()'

    # 计算 Pauli sigma y 算符的幂
    def _eval_power(self, e):
        # 如果指数是正整数，则计算幂操作
        if e.is_Integer and e.is_positive:
            return SigmaY(self.name).__pow__(int(e) % 2)

    # 默认基下的 Pauli sigma y 算符的表示
    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        # 如果请求的格式是 sympy，则返回默认的表示矩阵
        if format == 'sympy':
            return Matrix([[0, -I], [I, 0]])
        else:
            # 抛出未实现的格式表示错误
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')
# 定义 SigmaMinus 类，继承自 SigmaOpBase 类，表示 Pauli sigma minus 操作符
class SigmaMinus(SigmaOpBase):
    
    """Pauli sigma minus operator
    
    Parameters
    ==========
    
    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.
    
    Examples
    ========
    
    >>> from sympy.physics.quantum import represent, Dagger
    >>> from sympy.physics.quantum.pauli import SigmaMinus
    >>> sm = SigmaMinus()
    >>> sm
    SigmaMinus()
    >>> Dagger(sm)
    SigmaPlus()
    >>> represent(sm)
    Matrix([
    [0, 0],
    [1, 0]])
    """

    def __new__(cls, *args, **hints):
        # 调用父类的构造方法创建新的 SigmaMinus 对象
        return SigmaOpBase.__new__(cls, *args)

    def _eval_commutator_SigmaX(self, other, **hints):
        # 如果操作符的名称不同，则返回零
        if self.name != other.name:
            return S.Zero
        else:
            # 如果操作符的名称相同，返回负的 SigmaZ(self.name)
            return -SigmaZ(self.name)

    def _eval_commutator_SigmaY(self, other, **hints):
        # 如果操作符的名称不同，则返回零
        if self.name != other.name:
            return S.Zero
        else:
            # 如果操作符的名称相同，返回复数单位虚数乘以 SigmaZ(self.name)
            return I * SigmaZ(self.name)

    def _eval_commutator_SigmaZ(self, other, **hints):
        # 返回 2 倍的 self，即 2 * self
        return 2 * self

    def _eval_commutator_SigmaMinus(self, other, **hints):
        # 返回 SigmaZ(self.name)
        return SigmaZ(self.name)
    # 返回零矩阵，表示对称算符 SigmaZ 和其他对象的反对易子结果为零
    def _eval_anticommutator_SigmaZ(self, other, **hints):
        return S.Zero

    # 返回单位矩阵，表示对称算符 SigmaX 和其他对象的反对易子结果为单位矩阵
    def _eval_anticommutator_SigmaX(self, other, **hints):
        return S.One

    # 返回复数单位乘以负一，表示对称算符 SigmaY 和其他对象的反对易子结果为虚数单位乘以负一
    def _eval_anticommutator_SigmaY(self, other, **hints):
        return I * S.NegativeOne

    # 返回单位矩阵，表示对称算符 SigmaPlus 和其他对象的反对易子结果为单位矩阵
    def _eval_anticommutator_SigmaPlus(self, other, **hints):
        return S.One

    # 返回该对称算符的共轭算符 SigmaPlus 的对象
    def _eval_adjoint(self):
        return SigmaPlus(self.name)

    # 对称算符的幂运算的求值函数，如果指数 e 是正整数，则返回零
    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return S.Zero

    # 返回 LaTeX 格式的打印内容，根据 use_name 属性决定输出内容
    def _print_contents_latex(self, printer, *args):
        if self.use_name:
            return r'{\sigma_-^{(%s)}}' % str(self.name)
        else:
            return r'{\sigma_-}'

    # 返回默认的字符串格式打印内容，表示 SigmaMinus 对象的字符串形式
    def _print_contents(self, printer, *args):
        return 'SigmaMinus()'

    # 返回默认基底的表示，格式为 SymPy 矩阵形式的表示
    # 如果 format 是 'sympy'，返回固定的二维矩阵
    # 否则，抛出未实现的格式错误
    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[0, 0], [1, 0]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')
class SigmaPlus(SigmaOpBase):
    """Pauli sigma plus operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympy.physics.quantum import represent, Dagger
    >>> from sympy.physics.quantum.pauli import SigmaPlus
    >>> sp = SigmaPlus()
    >>> sp
    SigmaPlus()
    >>> Dagger(sp)
    SigmaMinus()
    >>> represent(sp)
    Matrix([
    [0, 1],
    [0, 0]])
    """

    def __new__(cls, *args, **hints):
        # 调用基类的构造函数来创建新的实例
        return SigmaOpBase.__new__(cls, *args)

    def _eval_commutator_SigmaX(self, other, **hints):
        # 如果操作符名字不同，则对易子为零
        if self.name != other.name:
            return S.Zero
        else:
            # 如果操作符名字相同，则返回与SigmaZ相关联的SigmaPlus
            return SigmaZ(self.name)

    def _eval_commutator_SigmaY(self, other, **hints):
        # 如果操作符名字不同，则对易子为零
        if self.name != other.name:
            return S.Zero
        else:
            # 如果操作符名字相同，则返回与SigmaZ相关联的i乘以SigmaPlus
            return I * SigmaZ(self.name)

    def _eval_commutator_SigmaZ(self, other, **hints):
        # 如果操作符名字不同，则对易子为零
        if self.name != other.name:
            return S.Zero
        else:
            # 如果操作符名字相同，则返回-2乘以自身的SigmaPlus
            return -2 * self

    def _eval_commutator_SigmaMinus(self, other, **hints):
        # 返回与SigmaZ相关联的SigmaPlus
        return SigmaZ(self.name)

    def _eval_anticommutator_SigmaZ(self, other, **hints):
        # 返回反对易子与SigmaZ相关联的零
        return S.Zero

    def _eval_anticommutator_SigmaX(self, other, **hints):
        # 返回反对易子与1相关联的零
        return S.One

    def _eval_anticommutator_SigmaY(self, other, **hints):
        # 返回反对易子与i相关联的零
        return I

    def _eval_anticommutator_SigmaMinus(self, other, **hints):
        # 返回反对易子与1相关联的零
        return S.One

    def _eval_adjoint(self):
        # 返回伴随操作符SigmaMinus
        return SigmaMinus(self.name)

    def _eval_mul(self, other):
        # 返回自身与其他操作符的乘积
        return self * other

    def _eval_power(self, e):
        # 如果指数为正整数，则返回零
        if e.is_Integer and e.is_positive:
            return S.Zero

    def _print_contents_latex(self, printer, *args):
        # 如果使用名称，则打印带标签的符号+
        if self.use_name:
            return r'{\sigma_+^{(%s)}}' % str(self.name)
        else:
            # 否则，打印普通的符号+
            return r'{\sigma_+}'

    def _print_contents(self, printer, *args):
        # 打印普通的SigmaPlus()字符串
        return 'SigmaPlus()'

    def _represent_default_basis(self, **options):
        # 返回默认基础表示，这里是一个2x2矩阵
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[0, 1], [0, 0]])
        else:
            # 如果格式不支持，则引发未实现错误
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaZKet(Ket):
    """Ket for a two-level system quantum system.

    Parameters
    ==========

    n : Number
        The state number (0 or 1).

    """

    def __new__(cls, n):
        # 确保n为0或1，否则引发值错误
        if n not in (0, 1):
            raise ValueError("n must be 0 or 1")
        return Ket.__new__(cls, n)

    @property
    def n(self):
        # 返回标签的第一个字符，即n的值
        return self.label[0]

    @classmethod
    def dual_class(self):
        # 返回对偶的bra类SigmaZBra
        return SigmaZBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        # 返回一个复数空间，大小为2
        return ComplexSpace(2)

    def _eval_innerproduct_SigmaZBra(self, bra, **hints):
        # 返回与SigmaZBra的内积结果，使用KroneckerDelta检查n和bra.n是否相等
        return KroneckerDelta(self.n, bra.n)
    # 将操作 op 从右侧应用到 SigmaZ 算符的方法
    def _apply_from_right_to_SigmaZ(self, op, **options):
        # 如果量子态的指标 n 等于 0，返回自身
        if self.n == 0:
            return self
        else:
            # 否则返回 S.NegativeOne 乘以自身
            return S.NegativeOne * self
    
    # 将操作 op 从右侧应用到 SigmaX 算符的方法
    def _apply_from_right_to_SigmaX(self, op, **options):
        # 如果量子态的指标 n 等于 0，返回 SigmaZKet(1)
        return SigmaZKet(1) if self.n == 0 else SigmaZKet(0)
    
    # 将操作 op 从右侧应用到 SigmaY 算符的方法
    def _apply_from_right_to_SigmaY(self, op, **options):
        # 如果量子态的指标 n 等于 0，返回 I 乘以 SigmaZKet(1)
        # 否则返回 -I 乘以 SigmaZKet(0)
        return I * SigmaZKet(1) if self.n == 0 else (-I) * SigmaZKet(0)
    
    # 将操作 op 从右侧应用到 SigmaMinus 算符的方法
    def _apply_from_right_to_SigmaMinus(self, op, **options):
        # 如果量子态的指标 n 等于 0，返回 SigmaZKet(1)
        # 否则返回 S.Zero
        if self.n == 0:
            return SigmaZKet(1)
        else:
            return S.Zero
    
    # 将操作 op 从右侧应用到 SigmaPlus 算符的方法
    def _apply_from_right_to_SigmaPlus(self, op, **options):
        # 如果量子态的指标 n 等于 0，返回 S.Zero
        # 否则返回 SigmaZKet(0)
        if self.n == 0:
            return S.Zero
        else:
            return SigmaZKet(0)
    
    # 默认基表示方法的实现
    def _represent_default_basis(self, **options):
        # 获取选项中的表示格式，默认为 'sympy'
        format = options.get('format', 'sympy')
        # 如果格式为 'sympy'
        if format == 'sympy':
            # 如果量子态的指标 n 等于 0，返回表示为 [[1], [0]] 的矩阵
            # 否则返回表示为 [[0], [1]] 的矩阵
            return Matrix([[1], [0]]) if self.n == 0 else Matrix([[0], [1]])
        else:
            # 如果格式不是 'sympy'，抛出未实现的错误
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')
class SigmaZBra(Bra):
    """Bra for a two-level quantum system.

    Parameters
    ==========

    n : Number
        The state number (0 or 1).

    """

    def __new__(cls, n):
        # 检查状态数是否为 0 或 1，如果不是则抛出值错误异常
        if n not in (0, 1):
            raise ValueError("n must be 0 or 1")
        # 调用父类 Bra 的 __new__ 方法创建实例
        return Bra.__new__(cls, n)

    @property
    def n(self):
        # 返回 Bra 对象的标签的第一个元素作为状态数
        return self.label[0]

    @classmethod
    def dual_class(self):
        # 返回 SigmaZKet 类作为对偶类
        return SigmaZKet


def _qsimplify_pauli_product(a, b):
    """
    Internal helper function for simplifying products of Pauli operators.
    """
    if not (isinstance(a, SigmaOpBase) and isinstance(b, SigmaOpBase)):
        # 如果 a 或 b 不是 SigmaOpBase 的实例，则返回它们的乘积
        return Mul(a, b)

    if a.name != b.name:
        # 当 Pauli 矩阵的标签不同的时候，它们交换位置（乘法交换律）
        if a.name < b.name:
            return Mul(a, b)
        else:
            return Mul(b, a)

    elif isinstance(a, SigmaX):

        if isinstance(b, SigmaX):
            # SigmaX 与 SigmaX 的乘积为标量 1
            return S.One

        if isinstance(b, SigmaY):
            # SigmaX 与 SigmaY 的乘积为复数单位虚数乘以 SigmaZ(a.name)
            return I * SigmaZ(a.name)

        if isinstance(b, SigmaZ):
            # SigmaX 与 SigmaZ 的乘积为负复数单位虚数乘以 SigmaY(a.name)
            return - I * SigmaY(a.name)

        if isinstance(b, SigmaMinus):
            # SigmaX 与 SigmaMinus 的乘积为复数标量 (1 + SigmaZ(a.name))/2
            return (S.Half + SigmaZ(a.name)/2)

        if isinstance(b, SigmaPlus):
            # SigmaX 与 SigmaPlus 的乘积为复数标量 (1 - SigmaZ(a.name))/2
            return (S.Half - SigmaZ(a.name)/2)

    elif isinstance(a, SigmaY):

        if isinstance(b, SigmaX):
            # SigmaY 与 SigmaX 的乘积为负复数单位虚数乘以 SigmaZ(a.name)
            return - I * SigmaZ(a.name)

        if isinstance(b, SigmaY):
            # SigmaY 与 SigmaY 的乘积为标量 1
            return S.One

        if isinstance(b, SigmaZ):
            # SigmaY 与 SigmaZ 的乘积为复数单位虚数乘以 SigmaX(a.name)
            return I * SigmaX(a.name)

        if isinstance(b, SigmaMinus):
            # SigmaY 与 SigmaMinus 的乘积为负复数标量 -(1 + SigmaZ(a.name))/2
            return -I * (S.One + SigmaZ(a.name))/2

        if isinstance(b, SigmaPlus):
            # SigmaY 与 SigmaPlus 的乘积为复数标量 (1 - SigmaZ(a.name))/2
            return I * (S.One - SigmaZ(a.name))/2

    elif isinstance(a, SigmaZ):

        if isinstance(b, SigmaX):
            # SigmaZ 与 SigmaX 的乘积为复数单位虚数乘以 SigmaY(a.name)
            return I * SigmaY(a.name)

        if isinstance(b, SigmaY):
            # SigmaZ 与 SigmaY 的乘积为负复数单位虚数乘以 SigmaX(a.name)
            return - I * SigmaX(a.name)

        if isinstance(b, SigmaZ):
            # SigmaZ 与 SigmaZ 的乘积为标量 1
            return S.One

        if isinstance(b, SigmaMinus):
            # SigmaZ 与 SigmaMinus 的乘积为负 SigmaMinus(a.name)
            return - SigmaMinus(a.name)

        if isinstance(b, SigmaPlus):
            # SigmaZ 与 SigmaPlus 的乘积为 SigmaPlus(a.name)
            return SigmaPlus(a.name)

    elif isinstance(a, SigmaMinus):

        if isinstance(b, SigmaX):
            # SigmaMinus 与 SigmaX 的乘积为复数标量 (1 - SigmaZ(a.name))/2
            return (S.One - SigmaZ(a.name))/2

        if isinstance(b, SigmaY):
            # SigmaMinus 与 SigmaY 的乘积为负复数单位虚数乘以 (1 - SigmaZ(a.name))/2
            return - I * (S.One - SigmaZ(a.name))/2

        if isinstance(b, SigmaZ):
            # SigmaMinus 与 SigmaZ 的乘积为 SigmaMinus(b.name)
            return SigmaMinus(b.name)

        if isinstance(b, SigmaMinus):
            # SigmaMinus 与 SigmaMinus 的乘积为标量 0
            return S.Zero

        if isinstance(b, SigmaPlus):
            # SigmaMinus 与 SigmaPlus 的乘积为复数标量 (1 - SigmaZ(a.name))/2
            return S.Half - SigmaZ(a.name)/2
    # 如果 a 是 SigmaPlus 类型的对象，则执行以下逻辑
    elif isinstance(a, SigmaPlus):
        # 如果 b 是 SigmaX 类型的对象，则返回 (1 + SigmaZ(a.name)) / 2
        if isinstance(b, SigmaX):
            return (S.One + SigmaZ(a.name)) / 2
        
        # 如果 b 是 SigmaY 类型的对象，则返回 I * (1 + SigmaZ(a.name)) / 2
        if isinstance(b, SigmaY):
            return I * (S.One + SigmaZ(a.name)) / 2
        
        # 如果 b 是 SigmaZ 类型的对象，则返回 -(SigmaX(a.name) + I * SigmaY(a.name)) / 2
        if isinstance(b, SigmaZ):
            #-(SigmaX(a.name) + I * SigmaY(a.name))/2
            return -SigmaPlus(a.name)
        
        # 如果 b 是 SigmaMinus 类型的对象，则返回 (1 + SigmaZ(a.name)) / 2
        if isinstance(b, SigmaMinus):
            return (S.One + SigmaZ(a.name)) / 2
        
        # 如果 b 是 SigmaPlus 类型的对象，则返回零
        if isinstance(b, SigmaPlus):
            return S.Zero
    
    # 如果 a 不是 SigmaPlus 类型的对象，则返回 a * b
    else:
        return a * b
# 定义函数 qsimplify_pauli，用于简化包含 Pauli 算符乘积的表达式
def qsimplify_pauli(e):
    """
    Simplify an expression that includes products of pauli operators.

    Parameters
    ==========

    e : expression
        An expression that contains products of Pauli operators that is
        to be simplified.

    Examples
    ========

    >>> from sympy.physics.quantum.pauli import SigmaX, SigmaY
    >>> from sympy.physics.quantum.pauli import qsimplify_pauli
    >>> sx, sy = SigmaX(), SigmaY()
    >>> sx * sy
    SigmaX()*SigmaY()
    >>> qsimplify_pauli(sx * sy)
    I*SigmaZ()
    """
    # 如果 e 是 Operator 类型，则直接返回 e
    if isinstance(e, Operator):
        return e

    # 如果 e 是 Add、Pow 或 exp 类型，则分别对其 args 应用 qsimplify_pauli 函数后返回同类型对象
    if isinstance(e, (Add, Pow, exp)):
        t = type(e)
        return t(*(qsimplify_pauli(arg) for arg in e.args))

    # 如果 e 是 Mul 类型
    if isinstance(e, Mul):
        # 将 e 的参数分为常数部分 c 和非常数部分 nc
        c, nc = e.args_cnc()

        # 初始化一个空列表用于存储简化后的非常数部分
        nc_s = []
        
        # 遍历 nc 列表
        while nc:
            # 弹出当前列表的第一个元素
            curr = nc.pop(0)

            # 如果 nc 不为空且当前元素与下一个元素都是 SigmaOpBase 类型且名称相同
            while (len(nc) and
                   isinstance(curr, SigmaOpBase) and
                   isinstance(nc[0], SigmaOpBase) and
                   curr.name == nc[0].name):

                # 弹出下一个元素
                x = nc.pop(0)
                # 对 curr 和 x 执行 _qsimplify_pauli_product 函数进行简化
                y = _qsimplify_pauli_product(curr, x)
                # 将简化后的结果再次分为常数部分和非常数部分
                c1, nc1 = y.args_cnc()
                # 将当前 curr 更新为非常数部分的乘积
                curr = Mul(*nc1)
                # 更新常数部分 c
                c = c + c1

            # 将简化后的当前元素添加到 nc_s 列表中
            nc_s.append(curr)

        # 返回乘积形式的结果，常数部分和非常数部分各自乘积
        return Mul(*c) * Mul(*nc_s)

    # 如果 e 不属于以上任何类型，则直接返回 e
    return e
```
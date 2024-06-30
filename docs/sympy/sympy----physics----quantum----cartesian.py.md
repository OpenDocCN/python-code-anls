# `D:\src\scipysrc\sympy\sympy\physics\quantum\cartesian.py`

```
# 导入 SymPy 库中所需的模块和函数，包括虚数单位 I 和圆周率 pi
from sympy.core.numbers import (I, pi)
# 导入 SymPy 中的单例对象 S
from sympy.core.singleton import S
# 导入 SymPy 中的指数函数 exp 和平方根函数 sqrt
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
# 导入 SymPy 中的 Delta 函数 DiracDelta
from sympy.functions.special.delta_functions import DiracDelta
# 导入 SymPy 中的区间对象 Interval
from sympy.sets.sets import Interval

# 导入 SymPy 中的量子力学常数 hbar
from sympy.physics.quantum.constants import hbar
# 导入 SymPy 中的量子力学 Hilbert 空间 L2
from sympy.physics.quantum.hilbert import L2
# 导入 SymPy 中的量子力学算子类 DifferentialOperator 和 HermitianOperator
from sympy.physics.quantum.operator import DifferentialOperator, HermitianOperator
# 导入 SymPy 中的量子力学态类 Ket、Bra 和 State
from sympy.physics.quantum.state import Ket, Bra, State

# 定义可以被外部调用的类名列表
__all__ = [
    'XOp',
    'YOp',
    'ZOp',
    'PxOp',
    'X',
    'Y',
    'Z',
    'Px',
    'XKet',
    'XBra',
    'PxKet',
    'PxBra',
    'PositionState3D',
    'PositionKet3D',
    'PositionBra3D'
]

#-------------------------------------------------------------------------
# 位置算子类定义
#-------------------------------------------------------------------------

# 定义描述 1D 笛卡尔坐标位置算子的类 XOp
class XOp(HermitianOperator):
    """1D cartesian position operator."""

    @classmethod
    def default_args(self):
        return ("X",)

    @classmethod
    def _eval_hilbert_space(self, args):
        # 定义 Hilbert 空间为连续区间 [-∞, +∞] 上的 L2 空间
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    def _eval_commutator_PxOp(self, other):
        # 计算该位置算子与动量算子 PxOp 的对易子，结果为 i*hbar
        return I*hbar

    def _apply_operator_XKet(self, ket, **options):
        # 应用位置算子于位置态 XKet，返回位置乘以位置态本身
        return ket.position*ket

    def _apply_operator_PositionKet3D(self, ket, **options):
        # 应用位置算子于三维位置态 PositionKet3D，返回位置 x 分量乘以位置态本身
        return ket.position_x*ket

    def _represent_PxKet(self, basis, *, index=1, **options):
        # 表示动量态 PxKet 在给定基底上的表示，使用了差分算子 DifferentialOperator 和 Delta 函数
        states = basis._enumerate_state(2, start_index=index)
        coord1 = states[0].momentum
        coord2 = states[1].momentum
        d = DifferentialOperator(coord1)
        delta = DiracDelta(coord1 - coord2)
        
        return I*hbar*(d*delta)

#-------------------------------------------------------------------------
# Y 轴位置算子类定义
#-------------------------------------------------------------------------

# 定义描述 Y 轴笛卡尔坐标位置算子的类 YOp（适用于二维或三维系统）
class YOp(HermitianOperator):
    """ Y cartesian coordinate operator (for 2D or 3D systems) """

    @classmethod
    def default_args(self):
        return ("Y",)

    @classmethod
    def _eval_hilbert_space(self, args):
        # 定义 Hilbert 空间为连续区间 [-∞, +∞] 上的 L2 空间
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    def _apply_operator_PositionKet3D(self, ket, **options):
        # 应用位置算子于三维位置态 PositionKet3D，返回位置 y 分量乘以位置态本身
        return ket.position_y*ket

#-------------------------------------------------------------------------
# Z 轴位置算子类定义
#-------------------------------------------------------------------------

# 定义描述 Z 轴笛卡尔坐标位置算子的类 ZOp（仅适用于三维系统）
class ZOp(HermitianOperator):
    """ Z cartesian coordinate operator (for 3D systems) """

    @classmethod
    def default_args(self):
        return ("Z",)

    @classmethod
    def _eval_hilbert_space(self, args):
        # 定义 Hilbert 空间为连续区间 [-∞, +∞] 上的 L2 空间
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    def _apply_operator_PositionKet3D(self, ket, **options):
        # 应用位置算子于三维位置态 PositionKet3D，返回位置 z 分量乘以位置态本身
        return ket.position_z*ket

#-------------------------------------------------------------------------
# 动量算子类定义
#-------------------------------------------------------------------------

# 定义描述 1D 笛卡尔坐标动量算子的类 PxOp
class PxOp(HermitianOperator):
    """1D cartesian momentum operator."""

    @classmethod
    def default_args(self):
        return ("Px",)

    @classmethod
    # 计算 Hilbert 空间中的 L2 范数，返回一个区间
    def _eval_hilbert_space(self, args):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    # 应用动量算符到态矢量 ket 上，返回动量乘以 ket
    def _apply_operator_PxKet(self, ket, **options):
        return ket.momentum * ket

    # 根据给定的基底 basis 和选项生成态矢量的表达式
    def _represent_XKet(self, basis, *, index=1, **options):
        # 枚举基底 basis 中的两个状态
        states = basis._enumerate_state(2, start_index=index)
        # 获取第一个状态的位置坐标
        coord1 = states[0].position
        # 获取第二个状态的位置坐标
        coord2 = states[1].position
        # 创建位置坐标 coord1 的微分算符对象
        d = DifferentialOperator(coord1)
        # 创建位置坐标为 coord1 和 coord2 差的 Dirac Delta 函数对象
        delta = DiracDelta(coord1 - coord2)

        # 返回表达式：-i*hbar*(d*delta)
        return -I * hbar * (d * delta)
X = XOp('X')
Y = YOp('Y')
Z = ZOp('Z')
Px = PxOp('Px')

#-------------------------------------------------------------------------
# Position eigenstates
#-------------------------------------------------------------------------

class XKet(Ket):
    """1D cartesian position eigenket."""

    @classmethod
    def _operators_to_state(self, op, **options):
        # 将操作符转换为状态对象
        return self.__new__(self, *_lowercase_labels(op), **options)

    def _state_to_operators(self, op_class, **options):
        # 将状态对象转换为操作符对象
        return op_class.__new__(op_class,
                                *_uppercase_labels(self), **options)

    @classmethod
    def default_args(self):
        # 默认参数为单个字符串 'x'
        return ("x",)

    @classmethod
    def dual_class(self):
        # 返回对偶类 XBra
        return XBra

    @property
    def position(self):
        """The position of the state."""
        # 返回状态的位置信息，这里是一个字符串 'x'
        return self.label[0]

    def _enumerate_state(self, num_states, **options):
        # 枚举连续态，返回状态的生成器
        return _enumerate_continuous_1D(self, num_states, **options)

    def _eval_innerproduct_XBra(self, bra, **hints):
        # 计算与 XBra 类型态的内积，使用位置之差的 DiracDelta 函数
        return DiracDelta(self.position - bra.position)

    def _eval_innerproduct_PxBra(self, bra, **hints):
        # 计算与 PxBra 类型态的内积，使用位置和动量的关系
        return exp(-I*self.position*bra.momentum/hbar)/sqrt(2*pi*hbar)


class XBra(Bra):
    """1D cartesian position eigenbra."""

    @classmethod
    def default_args(self):
        # 默认参数为单个字符串 'x'
        return ("x",)

    @classmethod
    def dual_class(self):
        # 返回对偶类 XKet
        return XKet

    @property
    def position(self):
        """The position of the state."""
        # 返回状态的位置信息，这里是一个字符串 'x'
        return self.label[0]


class PositionState3D(State):
    """ Base class for 3D cartesian position eigenstates """

    @classmethod
    def _operators_to_state(self, op, **options):
        # 将操作符转换为状态对象
        return self.__new__(self, *_lowercase_labels(op), **options)

    def _state_to_operators(self, op_class, **options):
        # 将状态对象转换为操作符对象
        return op_class.__new__(op_class,
                                *_uppercase_labels(self), **options)

    @classmethod
    def default_args(self):
        # 默认参数为三个字符串 'x', 'y', 'z'
        return ("x", "y", "z")

    @property
    def position_x(self):
        """ The x coordinate of the state """
        # 返回状态的 x 坐标信息
        return self.label[0]

    @property
    def position_y(self):
        """ The y coordinate of the state """
        # 返回状态的 y 坐标信息
        return self.label[1]

    @property
    def position_z(self):
        """ The z coordinate of the state """
        # 返回状态的 z 坐标信息
        return self.label[2]


class PositionKet3D(Ket, PositionState3D):
    """ 3D cartesian position eigenket """

    def _eval_innerproduct_PositionBra3D(self, bra, **options):
        # 计算与 PositionBra3D 类型态的内积，使用 x, y, z 三个方向的 DiracDelta 函数
        x_diff = self.position_x - bra.position_x
        y_diff = self.position_y - bra.position_y
        z_diff = self.position_z - bra.position_z

        return DiracDelta(x_diff)*DiracDelta(y_diff)*DiracDelta(z_diff)

    @classmethod
    def dual_class(self):
        # 返回对偶类 PositionBra3D
        return PositionBra3D


# XXX: The type:ignore here is because mypy gives Definition of
# "_state_to_operators" in base class "PositionState3D" is incompatible with
# definition in base class "BraBase"
class PositionBra3D(Bra, PositionState3D):  # type: ignore
    """ 3D cartesian position eigenbra """

    @classmethod
    def dual_class(self):
        # 返回这个类的对偶类 PositionKet3D
        return PositionKet3D


#-------------------------------------------------------------------------
# Momentum eigenstates
#-------------------------------------------------------------------------


class PxKet(Ket):
    """1D cartesian momentum eigenket."""

    @classmethod
    def _operators_to_state(self, op, **options):
        # 创建新的 PxKet 实例，使用 op 的小写标签作为参数
        return self.__new__(self, *_lowercase_labels(op), **options)

    def _state_to_operators(self, op_class, **options):
        # 创建新的 op_class 实例，使用当前实例的大写标签作为参数
        return op_class.__new__(op_class,
                                *_uppercase_labels(self), **options)

    @classmethod
    def default_args(self):
        # 返回默认参数 ("px",)
        return ("px",)

    @classmethod
    def dual_class(self):
        # 返回这个类的对偶类 PxBra
        return PxBra

    @property
    def momentum(self):
        """The momentum of the state."""
        # 返回状态的动量，即标签的第一个元素
        return self.label[0]

    def _enumerate_state(self, *args, **options):
        # 枚举连续的 1D 状态
        return _enumerate_continuous_1D(self, *args, **options)

    def _eval_innerproduct_XBra(self, bra, **hints):
        # 计算当前态与 XBra 的内积
        return exp(I*self.momentum*bra.position/hbar)/sqrt(2*pi*hbar)

    def _eval_innerproduct_PxBra(self, bra, **hints):
        # 计算当前态与 PxBra 的内积
        return DiracDelta(self.momentum - bra.momentum)


class PxBra(Bra):
    """1D cartesian momentum eigenbra."""

    @classmethod
    def default_args(self):
        # 返回默认参数 ("px",)
        return ("px",)

    @classmethod
    def dual_class(self):
        # 返回这个类的对偶类 PxKet
        return PxKet

    @property
    def momentum(self):
        """The momentum of the state."""
        # 返回状态的动量，即标签的第一个元素
        return self.label[0]

#-------------------------------------------------------------------------
# Global helper functions
#-------------------------------------------------------------------------


def _enumerate_continuous_1D(*args, **options):
    # 枚举连续的 1D 状态
    state = args[0]
    num_states = args[1]
    state_class = state.__class__
    index_list = options.pop('index_list', [])

    if len(index_list) == 0:
        start_index = options.pop('start_index', 1)
        index_list = list(range(start_index, start_index + num_states))

    enum_states = [0 for i in range(len(index_list))]

    for i, ind in enumerate(index_list):
        label = state.args[0]
        # 创建新的 state_class 实例，附加上索引
        enum_states[i] = state_class(str(label) + "_" + str(ind), **options)

    return enum_states


def _lowercase_labels(ops):
    # 将操作符的标签转换为小写
    if not isinstance(ops, set):
        ops = [ops]

    return [str(arg.label[0]).lower() for arg in ops]


def _uppercase_labels(ops):
    # 将操作符的标签转换为首字母大写
    if not isinstance(ops, set):
        ops = [ops]

    new_args = [str(arg.label[0])[0].upper() +
                str(arg.label[0])[1:] for arg in ops]

    return new_args
```
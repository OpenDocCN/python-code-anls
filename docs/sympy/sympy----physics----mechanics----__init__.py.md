# `D:\src\scipysrc\sympy\sympy\physics\mechanics\__init__.py`

```
# 定义一个列表，包含此模块中的所有公开（export）的对象名
__all__ = [
    'vector',  # 向量

    'CoordinateSym', 'ReferenceFrame', 'Dyadic', 'Vector', 'Point', 'cross',  # 符号坐标、参考系、二阶张量、向量、点、叉乘
    'dot', 'express', 'time_derivative', 'outer', 'kinematic_equations',  # 点乘、表达式、时间导数、外积、运动方程
    'get_motion_params', 'partial_velocity', 'dynamicsymbols', 'vprint',  # 获取运动参数、偏导速度、动态符号、打印函数
    'vsstrrepr', 'vsprint', 'vpprint', 'vlatex', 'init_vprinting',  # 符号表示字符串、符号打印、符号漂亮打印、符号 LaTeX、初始化符号打印
    'curl', 'divergence', 'gradient', 'is_conservative', 'is_solenoidal',  # 旋度、散度、梯度、是否保守、是否无旋
    'scalar_potential', 'scalar_potential_difference',  # 标量势、标量势差

    'KanesMethod',  # Kane方法

    'RigidBody',  # 刚体

    'linear_momentum', 'angular_momentum', 'kinetic_energy', 'potential_energy',  # 线动量、角动量、动能、势能
    'Lagrangian', 'mechanics_printing', 'mprint', 'msprint', 'mpprint',  # 拉格朗日量、力学打印、打印函数
    'mlatex', 'msubs', 'find_dynamicsymbols',  # LaTeX打印、符号替换、查找动态符号

    'inertia', 'inertia_of_point_mass', 'Inertia',  # 惯性、质点惯性、惯性

    'Force', 'Torque',  # 力、力矩

    'Particle',  # 质点

    'LagrangesMethod',  # 拉格朗日方法

    'Linearizer',  # 线性化器

    'Body',  # 身体

    'SymbolicSystem', 'System',  # 符号系统、系统

    'PinJoint', 'PrismaticJoint', 'CylindricalJoint', 'PlanarJoint',  # 销轴、棱柱关节、圆柱关节、平面关节
    'SphericalJoint', 'WeldJoint',  # 球面关节、焊接关节

    'JointsMethod',  # 关节方法

    'WrappingCylinder', 'WrappingGeometryBase', 'WrappingSphere',  # 包装圆柱体、包装几何体基类、包装球体

    'PathwayBase', 'LinearPathway', 'ObstacleSetPathway', 'WrappingPathway',  # 路径基类、线性路径、障碍物路径、包装路径

    'ActuatorBase', 'ForceActuator', 'LinearDamper', 'LinearSpring',  # 作动器基类、力作动器、线性阻尼器、线性弹簧
    'TorqueActuator', 'DuffingSpring'  # 扭矩作动器、杜芬双曲弹簧
]
# 导入 sympy.physics 中的 vector 模块
from sympy.physics import vector

# 导入 sympy.physics.vector 中的各种符号、向量操作等
from sympy.physics.vector import (CoordinateSym, ReferenceFrame, Dyadic, Vector, Point,
        cross, dot, express, time_derivative, outer, kinematic_equations,
        get_motion_params, partial_velocity, dynamicsymbols, vprint,
        vsstrrepr, vsprint, vpprint, vlatex, init_vprinting, curl, divergence,
        gradient, is_conservative, is_solenoidal, scalar_potential,
        scalar_potential_difference)

# 导入当前目录下的 kane.py 中的 KanesMethod 类
from .kane import KanesMethod

# 导入当前目录下的 rigidbody.py 中的 RigidBody 类
from .rigidbody import RigidBody

# 导入当前目录下的 functions.py 中的各种力学函数
from .functions import (linear_momentum, angular_momentum, kinetic_energy,
                        potential_energy, Lagrangian, mechanics_printing,
                        mprint, msprint, mpprint, mlatex, msubs,
                        find_dynamicsymbols)

# 导入当前目录下的 inertia.py 中的惯性相关函数
from .inertia import inertia, inertia_of_point_mass, Inertia

# 导入当前目录下的 loads.py 中的力和力矩类
from .loads import Force, Torque

# 导入当前目录下的 particle.py 中的 Particle 类
from .particle import Particle

# 导入当前目录下的 lagrange.py 中的 LagrangesMethod 类
from .lagrange import LagrangesMethod

# 导入当前目录下的 linearize.py 中的 Linearizer 类
from .linearize import Linearizer

# 导入当前目录下的 body.py 中的 Body 类
from .body import Body

# 导入当前目录下的 system.py 中的 SymbolicSystem 和 System 类
from .system import SymbolicSystem, System

# 导入当前目录下的 jointsmethod.py 中的 JointsMethod 类
from .jointsmethod import JointsMethod

# 导入当前目录下的 joint.py 中的各种关节类
from .joint import (PinJoint, PrismaticJoint, CylindricalJoint, PlanarJoint,
                    SphericalJoint, WeldJoint)

# 导入当前目录下的 wrapping_geometry.py 中的各种包装几何体类
from .wrapping_geometry import (WrappingCylinder, WrappingGeometryBase,
                                WrappingSphere)

# 导入当前目录下的 pathway.py 中的各种路径类
from .pathway import (PathwayBase, LinearPathway, ObstacleSetPathway,
                      WrappingPathway)

# 导入当前目录下的 actuator.py 中的各种作动器类
from .actuator import (ActuatorBase, ForceActuator, LinearDamper, LinearSpring,
                       TorqueActuator, DuffingSpring)
```
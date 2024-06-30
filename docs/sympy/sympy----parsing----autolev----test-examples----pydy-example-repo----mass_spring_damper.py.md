# `D:\src\scipysrc\sympy\sympy\parsing\autolev\test-examples\pydy-example-repo\mass_spring_damper.py`

```
# 导入 sympy.physics.mechanics 库，用作后续动力学计算
import sympy.physics.mechanics as _me
# 导入 sympy 库，用于符号计算
import sympy as _sm
# 导入 math 库，并命名为 m，用于数学计算
import math as m
# 导入 numpy 库，并命名为 _np，用于数值计算
import numpy as _np

# 定义符号变量 m, k, b, g，均为实数
m, k, b, g = _sm.symbols('m k b g', real=True)
# 定义动力学符号 position, speed，表示位置和速度
position, speed = _me.dynamicsymbols('position speed')
# 定义 position_d, speed_d 作为位置和速度的一阶导数
position_d, speed_d = _me.dynamicsymbols('position_ speed_', 1)
# 定义 o 作为动力学符号
o = _me.dynamicsymbols('o')
# 定义作用力 force，为 o*sin(t) 的结果
force = o*_sm.sin(_me.dynamicsymbols._t)

# 创建参考框架 frame_ceiling，表示一个参考框架
frame_ceiling = _me.ReferenceFrame('ceiling')
# 创建点对象 point_origin，表示一个参考点
point_origin = _me.Point('origin')
# 将 point_origin 设置在 frame_ceiling 参考框架下的速度为 0
point_origin.set_vel(frame_ceiling, 0)

# 创建质点对象 particle_block，表示一个质点
particle_block = _me.Particle('block', _me.Point('block_pt'), _sm.Symbol('m'))
# 将质点的位置设置为相对于 point_origin 的 position*frame_ceiling.x
particle_block.point.set_pos(point_origin, position*frame_ceiling.x)
# 将质点的质量设置为 m
particle_block.mass = m
# 将质点的速度设置为 speed*frame_ceiling.x
particle_block.point.set_vel(frame_ceiling, speed*frame_ceiling.x)

# 计算合力的大小 force_magnitude，表示 m*g-k*position-b*speed+force
force_magnitude = m*g - k*position - b*speed + force

# 计算应用在质点上的合力 force_block，并替换 position_d 为 speed
force_block = (force_magnitude*frame_ceiling.x).subs({position_d: speed})

# 定义速度约束方程 kd_eqs，表示 position_d - speed = 0
kd_eqs = [position_d - speed]

# 创建 forceList 列表，包含质点及其受到的合力 force_block
forceList = [(particle_block.point, (force_magnitude*frame_ceiling.x).subs({position_d: speed}))]

# 创建 KanesMethod 对象 kane，用于进行卡恩法（Kane's Method）计算
kane = _me.KanesMethod(frame_ceiling, q_ind=[position], u_ind=[speed], kd_eqs=kd_eqs)

# 计算并得到卡恩方程 fr 和 frstar
fr, frstar = kane.kanes_equations([particle_block], forceList)

# 定义零向量 zero，表示 fr + frstar = 0
zero = fr + frstar

# 导入 pydy.system 库中的 System 类
from pydy.system import System

# 创建 System 对象 sys，包含卡恩方程 kane、常数参数 constants、外部驱动 specifieds、初始条件 initial_conditions 和时间 times
sys = System(kane, constants={m: 1.0, k: 1.0, b: 0.2, g: 9.8},
             specifieds={_me.dynamicsymbols('t'): lambda x, t: t, o: 2},
             initial_conditions={position: 0.1, speed: -1 * 1.0},
             times=_np.linspace(0.0, 10.0, 10.0 / 0.01))

# 对系统进行积分运算，得到 y
y = sys.integrate()
```
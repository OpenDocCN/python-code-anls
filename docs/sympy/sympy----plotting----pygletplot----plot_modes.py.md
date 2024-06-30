# `D:\src\scipysrc\sympy\sympy\plotting\pygletplot\plot_modes.py`

```
from sympy.utilities.lambdify import lambdify
from sympy.core.numbers import pi
from sympy.functions import sin, cos
from sympy.plotting.pygletplot.plot_curve import PlotCurve
from sympy.plotting.pygletplot.plot_surface import PlotSurface

from math import sin as p_sin
from math import cos as p_cos

# 定义一个修饰器，将返回值转换为浮点数元组
def float_vec3(f):
    def inner(*args):
        v = f(*args)
        return float(v[0]), float(v[1]), float(v[2])
    return inner

# 定义一个二维笛卡尔曲线类，继承自PlotCurve
class Cartesian2D(PlotCurve):
    i_vars, d_vars = 'x', 'y'
    intervals = [[-5, 5, 100]]
    aliases = ['cartesian']
    is_default = True

    # 获取基于SymPy的求值器
    def _get_sympy_evaluator(self):
        fy = self.d_vars[0]  # 获取y方向的表达式
        x = self.t_interval.v  # 获取变量t的值

        # 定义并返回一个修饰后的函数
        @float_vec3
        def e(_x):
            return (_x, fy.subs(x, _x), 0.0)
        return e

    # 获取基于Lambda表达式的求值器
    def _get_lambda_evaluator(self):
        fy = self.d_vars[0]  # 获取y方向的表达式
        x = self.t_interval.v  # 获取变量t的值
        return lambdify([x], [x, fy, 0.0])


# 定义一个三维笛卡尔曲面类，继承自PlotSurface
class Cartesian3D(PlotSurface):
    i_vars, d_vars = 'xy', 'z'
    intervals = [[-1, 1, 40], [-1, 1, 40]]
    aliases = ['cartesian', 'monge']
    is_default = True

    # 获取基于SymPy的求值器
    def _get_sympy_evaluator(self):
        fz = self.d_vars[0]  # 获取z方向的表达式
        x = self.u_interval.v  # 获取变量u的值
        y = self.v_interval.v  # 获取变量v的值

        # 定义并返回一个修饰后的函数
        @float_vec3
        def e(_x, _y):
            return (_x, _y, fz.subs(x, _x).subs(y, _y))
        return e

    # 获取基于Lambda表达式的求值器
    def _get_lambda_evaluator(self):
        fz = self.d_vars[0]  # 获取z方向的表达式
        x = self.u_interval.v  # 获取变量u的值
        y = self.v_interval.v  # 获取变量v的值
        return lambdify([x, y], [x, y, fz])


# 定义一个二维参数化曲线类，继承自PlotCurve
class ParametricCurve2D(PlotCurve):
    i_vars, d_vars = 't', 'xy'
    intervals = [[0, 2*pi, 100]]
    aliases = ['parametric']
    is_default = True

    # 获取基于SymPy的求值器
    def _get_sympy_evaluator(self):
        fx, fy = self.d_vars  # 获取x和y方向的表达式
        t = self.t_interval.v  # 获取变量t的值

        # 定义并返回一个修饰后的函数
        @float_vec3
        def e(_t):
            return (fx.subs(t, _t), fy.subs(t, _t), 0.0)
        return e

    # 获取基于Lambda表达式的求值器
    def _get_lambda_evaluator(self):
        fx, fy = self.d_vars  # 获取x和y方向的表达式
        t = self.t_interval.v  # 获取变量t的值
        return lambdify([t], [fx, fy, 0.0])


# 定义一个三维参数化曲线类，继承自PlotCurve
class ParametricCurve3D(PlotCurve):
    i_vars, d_vars = 't', 'xyz'
    intervals = [[0, 2*pi, 100]]
    aliases = ['parametric']
    is_default = True

    # 获取基于SymPy的求值器
    def _get_sympy_evaluator(self):
        fx, fy, fz = self.d_vars  # 获取x、y和z方向的表达式
        t = self.t_interval.v  # 获取变量t的值

        # 定义并返回一个修饰后的函数
        @float_vec3
        def e(_t):
            return (fx.subs(t, _t), fy.subs(t, _t), fz.subs(t, _t))
        return e

    # 获取基于Lambda表达式的求值器
    def _get_lambda_evaluator(self):
        fx, fy, fz = self.d_vars  # 获取x、y和z方向的表达式
        t = self.t_interval.v  # 获取变量t的值
        return lambdify([t], [fx, fy, fz])


# 定义一个参数化曲面类，继承自PlotSurface
class ParametricSurface(PlotSurface):
    i_vars, d_vars = 'uv', 'xyz'
    intervals = [[-1, 1, 40], [-1, 1, 40]]
    aliases = ['parametric']
    is_default = True
    # 定义一个内部方法，用于生成基于 SymPy 表达式的向量值函数
    def _get_sympy_evaluator(self):
        # 从实例属性中获取 SymPy 表达式 fx, fy, fz
        fx, fy, fz = self.d_vars
        # 从实例属性中获取参数 u 和 v 的值
        u = self.u_interval.v
        v = self.v_interval.v

        # 定义一个装饰器函数 float_vec3，将返回值转换为浮点数向量
        @float_vec3
        # 定义内部函数 e，接受参数 _u 和 _v，返回 SymPy 表达式在给定参数下的求值结果
        def e(_u, _v):
            return (fx.subs(u, _u).subs(v, _v),   # 求解 fx 在 (_u, _v) 处的值
                    fy.subs(u, _u).subs(v, _v),   # 求解 fy 在 (_u, _v) 处的值
                    fz.subs(u, _u).subs(v, _v))   # 求解 fz 在 (_u, _v) 处的值
        # 返回内部函数 e，该函数作为一个向量值函数
        return e

    # 定义一个内部方法，用于生成基于 SymPy 表达式的 lambda 函数求值器
    def _get_lambda_evaluator(self):
        # 从实例属性中获取 SymPy 表达式 fx, fy, fz
        fx, fy, fz = self.d_vars
        # 从实例属性中获取参数 u 和 v 的值
        u = self.u_interval.v
        v = self.v_interval.v
        # 使用 lambdify 函数生成 lambda 函数，接受参数 [u, v]，返回 [fx, fy, fz] 的求值结果
        return lambdify([u, v], [fx, fy, fz])
# 极坐标曲线类，继承自 PlotCurve 类
class Polar(PlotCurve):
    # 独立变量和依赖变量分别为 't' 和 'r'
    i_vars, d_vars = 't', 'r'
    # 时间间隔设定为从 0 到 2π，包含 100 个点
    intervals = [[0, 2*pi, 100]]
    # 别名为 'polar'
    aliases = ['polar']
    # 默认设定为非默认曲线
    is_default = False

    # 获取 SymPy 评估器函数
    def _get_sympy_evaluator(self):
        # 提取依赖变量的第一个项
        fr = self.d_vars[0]
        # 提取独立变量的时间间隔值
        t = self.t_interval.v

        # 定义评估函数 e(_t)，返回坐标元组
        def e(_t):
            # 计算当前 _t 对应的 r 值，并转换为浮点数
            _r = float(fr.subs(t, _t))
            # 返回极坐标点的三维表示 (x, y, z)，其中 z 为 0.0
            return (_r*p_cos(_t), _r*p_sin(_t), 0.0)
        
        return e

    # 获取 Lambda 评估器函数
    def _get_lambda_evaluator(self):
        # 提取依赖变量的第一个项
        fr = self.d_vars[0]
        # 提取独立变量的时间间隔值
        t = self.t_interval.v
        # 计算 x 和 y 分量
        fx, fy = fr*cos(t), fr*sin(t)
        # 返回 Lambda 函数，接受 t 为参数，返回 [x, y, 0.0] 的列表
        return lambdify([t], [fx, fy, 0.0])


# 圆柱坐标曲面类，继承自 PlotSurface 类
class Cylindrical(PlotSurface):
    # 独立变量和依赖变量分别为 'th' 和 'r'
    i_vars, d_vars = 'th', 'r'
    # 时间间隔设定为从 0 到 2π，包含 40 个点；高度间隔从 -1 到 1，包含 20 个点
    intervals = [[0, 2*pi, 40], [-1, 1, 20]]
    # 别名为 'cylindrical' 和 'polar'
    aliases = ['cylindrical', 'polar']
    # 默认设定为非默认曲面
    is_default = False

    # 获取 SymPy 评估器函数
    def _get_sympy_evaluator(self):
        # 提取依赖变量的第一个项
        fr = self.d_vars[0]
        # 提取独立变量的角度间隔值
        t = self.u_interval.v
        # 提取独立变量的高度间隔值
        h = self.v_interval.v

        # 定义评估函数 e(_t, _h)，返回坐标元组
        def e(_t, _h):
            # 计算当前 (_t, _h) 对应的 r 值，并转换为浮点数
            _r = float(fr.subs(t, _t).subs(h, _h))
            # 返回圆柱坐标点的三维表示 (x, y, z)
            return (_r*p_cos(_t), _r*p_sin(_t), _h)
        
        return e

    # 获取 Lambda 评估器函数
    def _get_lambda_evaluator(self):
        # 提取依赖变量的第一个项
        fr = self.d_vars[0]
        # 提取独立变量的角度间隔值
        t = self.u_interval.v
        # 提取独立变量的高度间隔值
        h = self.v_interval.v
        # 计算 x 和 y 分量
        fx, fy = fr*cos(t), fr*sin(t)
        # 返回 Lambda 函数，接受 (t, h) 为参数，返回 [x, y, h] 的列表
        return lambdify([t, h], [fx, fy, h])


# 球坐标曲面类，继承自 PlotSurface 类
class Spherical(PlotSurface):
    # 独立变量和依赖变量分别为 'tp' 和 'r'
    i_vars, d_vars = 'tp', 'r'
    # 时间间隔设定为从 0 到 2π，包含 40 个点；极角间隔从 0 到 π，包含 20 个点
    intervals = [[0, 2*pi, 40], [0, pi, 20]]
    # 别名为 'spherical'
    aliases = ['spherical']
    # 默认设定为非默认曲面
    is_default = False

    # 获取 SymPy 评估器函数
    def _get_sympy_evaluator(self):
        # 提取依赖变量的第一个项
        fr = self.d_vars[0]
        # 提取独立变量的极角间隔值
        t = self.u_interval.v
        # 提取独立变量的极角间隔值
        p = self.v_interval.v

        # 定义评估函数 e(_t, _p)，返回坐标元组
        def e(_t, _p):
            # 计算当前 (_t, _p) 对应的 r 值，并转换为浮点数
            _r = float(fr.subs(t, _t).subs(p, _p))
            # 返回球坐标点的三维表示 (x, y, z)
            return (_r*p_cos(_t)*p_sin(_p),
                    _r*p_sin(_t)*p_sin(_p),
                    _r*p_cos(_p))
        
        return e

    # 获取 Lambda 评估器函数
    def _get_lambda_evaluator(self):
        # 提取依赖变量的第一个项
        fr = self.d_vars[0]
        # 提取独立变量的极角间隔值
        t = self.u_interval.v
        # 提取独立变量的极角间隔值
        p = self.v_interval.v
        # 计算 x、y 和 z 分量
        fx = fr * cos(t) * sin(p)
        fy = fr * sin(t) * sin(p)
        fz = fr * cos(p)
        # 返回 Lambda 函数，接受 (t, p) 为参数，返回 [x, y, z] 的列表
        return lambdify([t, p], [fx, fy, fz])


# 将各种曲面注册到相应的类中
Cartesian2D._register()
Cartesian3D._register()
ParametricCurve2D._register()
ParametricCurve3D._register()
ParametricSurface._register()
Polar._register()
Cylindrical._register()
Spherical._register()
```
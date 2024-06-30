# `D:\src\scipysrc\sympy\sympy\plotting\pygletplot\plot_rotation.py`

```
# 尝试导入 ctypes 模块中的 c_float 类型
try:
    from ctypes import c_float
# 如果 ImportError 异常发生，则忽略，继续执行
except ImportError:
    pass

# 导入 pyglet 库中的 gl 模块，命名为 pgl
import pyglet.gl as pgl
# 从 math 模块中导入 sqrt 函数并命名为 _sqrt，导入 acos 函数并命名为 _acos
from math import sqrt as _sqrt, acos as _acos


# 计算向量 a 和 b 的叉积
def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


# 计算向量 a 和 b 的点积
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


# 计算向量 a 的模长
def mag(a):
    return _sqrt(a[0]**2 + a[1]**2 + a[2]**2)


# 计算向量 a 的单位向量
def norm(a):
    # 计算向量 a 的模长
    m = mag(a)
    # 返回单位向量，即每个分量除以模长
    return (a[0] / m, a[1] / m, a[2] / m)


# 根据屏幕坐标 (x, y)，返回球面映射的单位向量
def get_sphere_mapping(x, y, width, height):
    # 将 x 和 y 限制在 [0, width] 和 [0, height] 范围内
    x = min([max([x, 0]), width])
    y = min([max([y, 0]), height])

    # 计算屏幕的半径
    sr = _sqrt((width/2)**2 + (height/2)**2)
    # 计算球面上的坐标 sx 和 sy
    sx = ((x - width / 2) / sr)
    sy = ((y - height / 2) / sr)

    # 计算 sz，球面上点的 z 坐标
    sz = 1.0 - sx**2 - sy**2

    # 如果 sz 大于 0，则计算实际 z 坐标，并返回单位向量
    if sz > 0.0:
        sz = _sqrt(sz)
        return (sx, sy, sz)
    # 如果 sz 小于等于 0，则返回单位向量 (sx, sy, 0)
    else:
        sz = 0
        return norm((sx, sy, sz))


# 根据两个屏幕点的坐标 p1 和 p2，返回用于旋转的模型视图矩阵
def get_spherical_rotatation(p1, p2, width, height, theta_multiplier):
    # 获取 p1 和 p2 的球面映射单位向量
    v1 = get_sphere_mapping(p1[0], p1[1], width, height)
    v2 = get_sphere_mapping(p2[0], p2[1], width, height)

    # 计算 v1 和 v2 的点积，限制在 [-1, 1] 范围内
    d = min(max([dot(v1, v2), -1]), 1)

    # 如果 d 接近于 1，则返回 None，表示不需要旋转
    if abs(d - 1.0) < 0.000001:
        return None

    # 计算旋转轴 raxis 和旋转角度 rtheta
    raxis = norm(cross(v1, v2))
    rtheta = theta_multiplier * rad2deg * _acos(d)

    # 保存当前的模型视图矩阵，加载单位矩阵，然后旋转并获取旋转后的模型视图矩阵
    pgl.glPushMatrix()
    pgl.glLoadIdentity()
    pgl.glRotatef(rtheta, *raxis)
    mat = (c_float*16)()
    pgl.glGetFloatv(pgl.GL_MODELVIEW_MATRIX, mat)
    pgl.glPopMatrix()

    # 返回旋转后的模型视图矩阵
    return mat
```
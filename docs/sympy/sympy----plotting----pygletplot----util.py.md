# `D:\src\scipysrc\sympy\sympy\plotting\pygletplot\util.py`

```
# 尝试导入必要的数据类型定义：单精度浮点数、整数和双精度浮点数
try:
    from ctypes import c_float, c_int, c_double
except ImportError:
    pass

# 导入 Pyglet 的 OpenGL 封装
import pyglet.gl as pgl
# 导入 SymPy 的 S 对象
from sympy.core import S


def get_model_matrix(array_type=c_float, glGetMethod=pgl.glGetFloatv):
    """
    Returns the current modelview matrix.
    """
    # 创建一个用于存储 16 个元素的数组，类型为 array_type
    m = (array_type*16)()
    # 使用 glGetMethod 函数获取当前的模型视图矩阵，存储在 m 中
    glGetMethod(pgl.GL_MODELVIEW_MATRIX, m)
    return m


def get_projection_matrix(array_type=c_float, glGetMethod=pgl.glGetFloatv):
    """
    Returns the current modelview matrix.
    """
    # 创建一个用于存储 16 个元素的数组，类型为 array_type
    m = (array_type*16)()
    # 使用 glGetMethod 函数获取当前的投影矩阵，存储在 m 中
    glGetMethod(pgl.GL_PROJECTION_MATRIX, m)
    return m


def get_viewport():
    """
    Returns the current viewport.
    """
    # 创建一个用于存储 4 个整数的数组
    m = (c_int*4)()
    # 使用 OpenGL 函数 glGetIntegerv 获取当前视口参数，存储在 m 中
    pgl.glGetIntegerv(pgl.GL_VIEWPORT, m)
    return m


def get_direction_vectors():
    # 获取当前模型视图矩阵
    m = get_model_matrix()
    # 返回 X、Y、Z 方向向量
    return ((m[0], m[4], m[8]),
            (m[1], m[5], m[9]),
            (m[2], m[6], m[10]))


def get_view_direction_vectors():
    # 获取当前模型视图矩阵
    m = get_model_matrix()
    # 返回视图方向的 X、Y、Z 方向向量
    return ((m[0], m[1], m[2]),
            (m[4], m[5], m[6]),
            (m[8], m[9], m[10]))


def get_basis_vectors():
    # 返回标准的 X、Y、Z 基向量
    return ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def screen_to_model(x, y, z):
    # 获取双精度浮点数的模型视图矩阵和投影矩阵
    m = get_model_matrix(c_double, pgl.glGetDoublev)
    p = get_projection_matrix(c_double, pgl.glGetDoublev)
    # 获取当前视口参数
    w = get_viewport()
    # 创建用于存储结果的双精度浮点数变量
    mx, my, mz = c_double(), c_double(), c_double()
    # 使用 OpenGL 函数 gluUnProject 进行屏幕坐标到模型坐标的转换
    pgl.gluUnProject(x, y, z, m, p, w, mx, my, mz)
    # 返回转换后的模型坐标
    return float(mx.value), float(my.value), float(mz.value)


def model_to_screen(x, y, z):
    # 获取双精度浮点数的模型视图矩阵和投影矩阵
    m = get_model_matrix(c_double, pgl.glGetDoublev)
    p = get_projection_matrix(c_double, pgl.glGetDoublev)
    # 获取当前视口参数
    w = get_viewport()
    # 创建用于存储结果的双精度浮点数变量
    mx, my, mz = c_double(), c_double(), c_double()
    # 使用 OpenGL 函数 gluProject 进行模型坐标到屏幕坐标的转换
    pgl.gluProject(x, y, z, m, p, w, mx, my, mz)
    # 返回转换后的屏幕坐标
    return float(mx.value), float(my.value), float(mz.value)


def vec_subs(a, b):
    # 返回两个向量对应分量的差值组成的元组
    return tuple(a[i] - b[i] for i in range(len(a)))


def billboard_matrix():
    """
    Removes rotational components of
    current matrix so that primitives
    are always drawn facing the viewer.

    |1|0|0|x|
    |0|1|0|x|
    |0|0|1|x| (x means left unchanged)
    |x|x|x|x|
    """
    # 获取当前模型视图矩阵
    m = get_model_matrix()
    # 将矩阵的旋转部分清除，使图元永远面向观察者
    m[0] = 1
    m[1] = 0
    m[2] = 0
    m[4] = 0
    m[5] = 1
    m[6] = 0
    m[8] = 0
    m[9] = 0
    m[10] = 1
    # 使用 OpenGL 函数 glLoadMatrixf 加载修改后的矩阵
    pgl.glLoadMatrixf(m)


def create_bounds():
    # 创建一个用于存储边界框的列表，初始值设定为无限大和负无限大
    return [[S.Infinity, S.NegativeInfinity, 0],
            [S.Infinity, S.NegativeInfinity, 0],
            [S.Infinity, S.NegativeInfinity, 0]]


def update_bounds(b, v):
    # 如果传入的值 v 为 None，则直接返回
    if v is None:
        return
    # 更新边界框 b，使其包含值 v 的坐标
    for axis in range(3):
        b[axis][0] = min([b[axis][0], v[axis]])
        b[axis][1] = max([b[axis][1], v[axis]])


def interpolate(a_min, a_max, a_ratio):
    # 根据给定的比例在两个值之间插值
    return a_min + a_ratio * (a_max - a_min)


def rinterpolate(a_min, a_max, a_value):
    # 根据给定的值在两个边界内进行反插值
    a_range = a_max - a_min
    if a_max == a_min:
        a_range = 1.0
    return (a_value - a_min) / float(a_range)


def interpolate_color(color1, color2, ratio):
    # 这个函数还未完整，需要根据需要进行实现
    pass
    # 对两个颜色值 color1 和 color2 进行插值处理，生成新的颜色值
    # 按照给定的比例 ratio 在每个颜色通道上进行线性插值
    return tuple(interpolate(color1[i], color2[i], ratio) for i in range(3))
# 计算给定值 v 在范围 [v_min, v_min + v_len] 内的归一化值
def scale_value(v, v_min, v_len):
    return (v - v_min) / v_len


# 对给定的列表 flist 中的值进行归一化处理，返回归一化后的列表
def scale_value_list(flist):
    # 获取 flist 中的最小值和最大值
    v_min, v_max = min(flist), max(flist)
    # 计算值的范围长度
    v_len = v_max - v_min
    # 对 flist 中的每个值进行归一化处理，生成归一化后的列表
    return [scale_value(f, v_min, v_len) for f in flist]


# 生成一个按步长 stride 分隔的范围列表，范围从 r_min 到 r_max
# 最大生成步数为 max_steps，默认为 50
def strided_range(r_min, r_max, stride, max_steps=50):
    # 保存原始的范围起始和结束值
    o_min, o_max = r_min, r_max
    # 如果范围过小，直接返回空列表
    if abs(r_min - r_max) < 0.001:
        return []
    # 尝试生成一个范围对象，如果出错返回空列表
    try:
        range(int(r_min - r_max))
    except (TypeError, OverflowError):
        return []
    # 如果起始值大于结束值，抛出值错误异常
    if r_min > r_max:
        raise ValueError("r_min cannot be greater than r_max")
    # 计算调整后的起始和结束值，使其可以整除步长
    r_min_s = (r_min % stride)
    r_max_s = stride - (r_max % stride)
    if abs(r_max_s - stride) < 0.001:
        r_max_s = 0.0
    r_min -= r_min_s
    r_max += r_max_s
    # 计算范围内步数
    r_steps = int((r_max - r_min)/stride)
    # 如果超过最大步数限制，递归调用以增加步长
    if max_steps and r_steps > max_steps:
        return strided_range(o_min, o_max, stride*2)
    # 返回生成的范围列表
    return [r_min] + [r_min + e*stride for e in range(1, r_steps + 1)] + [r_max]


# 解析给定的选项字符串 s，返回解析后的选项字典
def parse_option_string(s):
    # 如果 s 不是字符串，返回 None
    if not isinstance(s, str):
        return None
    # 初始化选项字典
    options = {}
    # 按分号分隔字符串 s，逐个解析选项和值对
    for token in s.split(';'):
        pieces = token.split('=')
        # 如果没有值，将值设为空字符串
        if len(pieces) == 1:
            option, value = pieces[0], ""
        # 如果有值，解析选项和对应的值
        elif len(pieces) == 2:
            option, value = pieces
        else:
            # 如果格式错误，抛出值错误异常
            raise ValueError("Plot option string '%s' is malformed." % (s))
        # 将解析的选项和值存入选项字典，去除首尾空格
        options[option.strip()] = value.strip()
    # 返回解析后的选项字典
    return options


# 计算两个三维向量 v1 和 v2 的点积
def dot_product(v1, v2):
    return sum(v1[i]*v2[i] for i in range(3))


# 计算两个三维向量 v1 和 v2 的差，返回差向量
def vec_sub(v1, v2):
    return tuple(v1[i] - v2[i] for i in range(3))


# 计算三维向量 v 的模长
def vec_mag(v):
    return sum(v[i]**2 for i in range(3))**(0.5)
```
# `D:\src\scipysrc\sympy\sympy\physics\vector\fieldfunctions.py`

```
# 从 sympy 库中导入不同的模块和函数
from sympy.core.function import diff
from sympy.core.singleton import S
from sympy.integrals.integrals import integrate
from sympy.physics.vector import Vector, express
from sympy.physics.vector.frame import _check_frame
from sympy.physics.vector.vector import _check_vector

# 导出给定的所有符号，使它们可以通过模块名访问
__all__ = ['curl', 'divergence', 'gradient', 'is_conservative',
           'is_solenoidal', 'scalar_potential',
           'scalar_potential_difference']

# 计算给定向量场的旋度，关于给定参考系的坐标符号
def curl(vect, frame):
    """
    Returns the curl of a vector field computed wrt the coordinate
    symbols of the given frame.

    Parameters
    ==========

    vect : Vector
        The vector operand

    frame : ReferenceFrame
        The reference frame to calculate the curl in

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame
    >>> from sympy.physics.vector import curl
    >>> R = ReferenceFrame('R')
    >>> v1 = R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z
    >>> curl(v1, R)
    0
    >>> v2 = R[0]*R[1]*R[2]*R.x
    >>> curl(v2, R)
    R_x*R_y*R.y - R_x*R_z*R.z

    """

    # 检查向量是否有效
    _check_vector(vect)
    # 如果向量为零向量，则返回零向量
    if vect == 0:
        return Vector(0)
    # 将向量表达为给定参考系中的表达式
    vect = express(vect, frame, variables=True)
    # 分解向量分量
    vectx = vect.dot(frame.x)
    vecty = vect.dot(frame.y)
    vectz = vect.dot(frame.z)
    # 初始化输出向量
    outvec = Vector(0)
    # 计算旋度的各分量
    outvec += (diff(vectz, frame[1]) - diff(vecty, frame[2])) * frame.x
    outvec += (diff(vectx, frame[2]) - diff(vectz, frame[0])) * frame.y
    outvec += (diff(vecty, frame[0]) - diff(vectx, frame[1])) * frame.z
    return outvec


# 计算给定向量场的散度，关于给定参考系的坐标符号
def divergence(vect, frame):
    """
    Returns the divergence of a vector field computed wrt the coordinate
    symbols of the given frame.

    Parameters
    ==========

    vect : Vector
        The vector operand

    frame : ReferenceFrame
        The reference frame to calculate the divergence in

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame
    >>> from sympy.physics.vector import divergence
    >>> R = ReferenceFrame('R')
    >>> v1 = R[0]*R[1]*R[2] * (R.x+R.y+R.z)
    >>> divergence(v1, R)
    R_x*R_y + R_x*R_z + R_y*R_z
    >>> v2 = 2*R[1]*R[2]*R.y
    >>> divergence(v2, R)
    2*R_z

    """

    # 检查向量是否有效
    _check_vector(vect)
    # 如果向量为零向量，则返回零标量
    if vect == 0:
        return S.Zero
    # 将向量表达为给定参考系中的表达式
    vect = express(vect, frame, variables=True)
    # 分解向量分量
    vectx = vect.dot(frame.x)
    vecty = vect.dot(frame.y)
    vectz = vect.dot(frame.z)
    # 初始化输出标量
    out = S.Zero
    # 计算散度的各分量
    out += diff(vectx, frame[0])
    out += diff(vecty, frame[1])
    out += diff(vectz, frame[2])
    return out


# 计算给定标量场的梯度，关于给定参考系的坐标符号
def gradient(scalar, frame):
    """
    Returns the vector gradient of a scalar field computed wrt the
    coordinate symbols of the given frame.

    Parameters
    ==========

    scalar : sympifiable
        The scalar field to take the gradient of

    frame : ReferenceFrame
        The frame to calculate the gradient in

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame
    >>> from sympy.physics.vector import gradient
    >>> R = ReferenceFrame('R')
    >>> scalar_field = R[0]**2 + R[1]**2 + R[2]**2
    >>> gradient(scalar_field, R)
    2*R_x*R.x + 2*R_y*R.y + 2*R_z*R.z

    """

    # 将标量场的梯度视为向量场的梯度
    return curl(Vector(scalar), frame)
    # 导入梯度函数来计算向量场的梯度
    from sympy.physics.vector import gradient
    # 创建一个参考坐标系 'R'
    R = ReferenceFrame('R')
    # 定义标量场 s1 = R[0]*R[1]*R[2]
    s1 = R[0]*R[1]*R[2]
    # 计算 s1 关于参考坐标系 R 的梯度
    gradient(s1, R)
    # 输出结果 R_y*R_z*R.x + R_x*R_z*R.y + R_x*R_y*R.z

    # 定义另一个标量场 s2 = 5*R[0]**2*R[2]
    s2 = 5*R[0]**2*R[2]
    # 计算 s2 关于参考坐标系 R 的梯度
    gradient(s2, R)
    # 输出结果 10*R_x*R_z*R.x + 5*R_x**2*R.z

    """

    # 检查给定的参考坐标系是否有效
    _check_frame(frame)
    # 创建一个空的向量对象 outvec
    outvec = Vector(0)
    # 将 scalar 表达为关于参考坐标系的表达式
    scalar = express(scalar, frame, variables=True)
    # 遍历参考坐标系中的每个分量
    for i, x in enumerate(frame):
        # 计算标量关于第 i 个分量的偏导数，并乘以第 i 个分量 x，累加到 outvec 中
        outvec += diff(scalar, frame[i]) * x
    # 返回计算得到的向量 outvec
    return outvec
def scalar_potential(field, frame):
    """
    Returns the scalar potential function of a field in a given frame
    (without the added integration constant).

    Parameters
    ==========

    field : Vector
        The vector field whose scalar potential function is to be
        calculated

    frame : ReferenceFrame
        The frame to do the calculation in

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame
    >>> from sympy.physics.vector import scalar_potential, gradient
    >>> R = ReferenceFrame('R')
    >>> scalar_potential(R.z, R) == R[2]
    True
    >>> scalar_field = 2*R[0]**2*R[1]*R[2]
    >>> grad_field = gradient(scalar_field, R)
    >>> scalar_potential(grad_field, R)
    2*R_x**2*R_y*R_z

    """

    # Check whether field is conservative
    if not is_conservative(field):
        raise ValueError("Field is not conservative")

    # Check if the field is zero vector
    if field == Vector(0):
        return S.Zero

    # Ensure the frame is valid
    _check_frame(frame)

    # Express the field entirely in terms of the frame's variables
    field = express(field, frame, variables=True)

    # Get the list of dimensions of the reference frame
    dimensions = list(frame)

    # Initialize the temporary function for scalar potential
    temp_function = integrate(field.dot(dimensions[0]), frame[0])

    # Iterate over dimensions to calculate the scalar potential function
    for i, dim in enumerate(dimensions[1:]):
        partial_diff = diff(temp_function, frame[i + 1])
        partial_diff = field.dot(dim) - partial_diff
        temp_function += integrate(partial_diff, frame[i + 1])
    # 返回函数对象 temp_function
    return temp_function
# 检查参考框架的有效性，确保其符合要求
_check_frame(frame)

# 如果给定的场是一个向量，则获取其标量势函数
if isinstance(field, Vector):
    # 获取场的标量势函数
    scalar_fn = scalar_potential(field, frame)
else:
    # 如果场是标量，则直接使用该标量
    scalar_fn = field

# 将点的位置表达为所需参考框架中的表达式，使用变量进行替换
position1 = express(point1.pos_from(origin), frame, variables=True)
position2 = express(point2.pos_from(origin), frame, variables=True)

# 为坐标变量创建两个位置的替换字典
subs_dict1 = {}
subs_dict2 = {}
for i, x in enumerate(frame):
    subs_dict1[frame[i]] = x.dot(position1)
    subs_dict2[frame[i]] = x.dot(position2)

# 计算两个位置上的标量势差，并返回结果
return scalar_fn.subs(subs_dict2) - scalar_fn.subs(subs_dict1)
```
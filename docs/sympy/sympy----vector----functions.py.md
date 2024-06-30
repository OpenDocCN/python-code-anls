# `D:\src\scipysrc\sympy\sympy\vector\functions.py`

```
# 从 sympy.vector.coordsysrect 模块中导入 CoordSys3D 类
from sympy.vector.coordsysrect import CoordSys3D
# 从 sympy.vector.deloperator 模块中导入 Del 类
from sympy.vector.deloperator import Del
# 从 sympy.vector.scalar 模块中导入 BaseScalar 类
from sympy.vector.scalar import BaseScalar
# 从 sympy.vector.vector 模块中导入 Vector 和 BaseVector 类
from sympy.vector.vector import Vector, BaseVector
# 从 sympy.vector.operators 模块中导入 gradient, curl, divergence 函数
from sympy.vector.operators import gradient, curl, divergence
# 从 sympy.core.function 模块中导入 diff 函数
from sympy.core.function import diff
# 从 sympy.core.singleton 模块中导入 S 对象
from sympy.core.singleton import S
# 从 sympy.integrals.integrals 模块中导入 integrate 函数
from sympy.integrals.integrals import integrate
# 从 sympy.simplify.simplify 模块中导入 simplify 函数
from sympy.simplify.simplify import simplify
# 从 sympy.core 模块中导入 sympify 函数
from sympy.core import sympify
# 从 sympy.vector.dyadic 模块中导入 Dyadic 类
from sympy.vector.dyadic import Dyadic

# 定义 express 函数，用于将表达式重新表达为给定坐标系中的形式
def express(expr, system, system2=None, variables=False):
    """
    Global function for 'express' functionality.

    Re-expresses a Vector, Dyadic or scalar(sympyfiable) in the given
    coordinate system.

    If 'variables' is True, then the coordinate variables (base scalars)
    of other coordinate systems present in the vector/scalar field or
    dyadic are also substituted in terms of the base scalars of the
    given system.

    Parameters
    ==========

    expr : Vector/Dyadic/scalar(sympyfiable)
        The expression to re-express in CoordSys3D 'system'

    system: CoordSys3D
        The coordinate system the expr is to be expressed in

    system2: CoordSys3D
        The other coordinate system required for re-expression
        (only for a Dyadic Expr)

    variables : boolean
        Specifies whether to substitute the coordinate variables present
        in expr, in terms of those of parameter system

    Examples
    ========

    >>> from sympy.vector import CoordSys3D
    >>> from sympy import Symbol, cos, sin
    >>> N = CoordSys3D('N')
    >>> q = Symbol('q')
    >>> B = N.orient_new_axis('B', q, N.k)
    >>> from sympy.vector import express
    >>> express(B.i, N)
    (cos(q))*N.i + (sin(q))*N.j
    >>> express(N.x, B, variables=True)
    B.x*cos(q) - B.y*sin(q)
    >>> d = N.i.outer(N.i)
    >>> express(d, B, N) == (cos(q))*(B.i|N.i) + (-sin(q))*(B.j|N.i)
    True

    """

    # 如果表达式为零向量或零标量，直接返回表达式
    if expr in (0, Vector.zero):
        return expr

    # 如果 system 不是 CoordSys3D 的实例，则引发 TypeError 异常
    if not isinstance(system, CoordSys3D):
        raise TypeError("system should be a CoordSys3D instance")
    # 如果 expr 是 Vector 类型
    if isinstance(expr, Vector):
        # 如果同时提供了 system2 参数，则引发数值错误
        if system2 is not None:
            raise ValueError("system2 should not be provided for \
                                Vectors")
        # 给定的 expr 是一个 Vector 类型

        # 如果 variables 参数为 True，则替换 Vector 中的坐标变量
        if variables:
            # 查找 Vector 中所有 BaseScalar 和 BaseVector 类型的元素，并确定它们的坐标系统列表
            system_list = {x.system for x in expr.atoms(BaseScalar, BaseVector)} - {system}
            subs_dict = {}
            # 遍历坐标系统列表，并更新替换字典
            for f in system_list:
                subs_dict.update(f.scalar_map(system))
            # 使用替换字典对 Vector 进行替换
            expr = expr.subs(subs_dict)

        # 在当前坐标系中重新表达 Vector
        outvec = Vector.zero
        # 将 Vector 的表达式分解为部分
        parts = expr.separate()
        for x in parts:
            if x != system:
                # 计算当前坐标系到 x 坐标系的旋转矩阵，并将其应用于对应部分，然后转换为向量形式，最后累加到 outvec 中
                temp = system.rotation_matrix(x) * parts[x].to_matrix(x)
                outvec += matrix_to_vector(temp, system)
            else:
                # 如果 x 等于当前坐标系，则直接添加到 outvec 中
                outvec += parts[x]
        
        # 返回处理后的 Vector
        return outvec

    # 如果 expr 是 Dyadic 类型
    elif isinstance(expr, Dyadic):
        # 如果未提供 system2 参数，则将其设置为与 system 相同的值
        if system2 is None:
            system2 = system
        # 如果 system2 不是 CoordSys3D 的实例，则引发类型错误
        if not isinstance(system2, CoordSys3D):
            raise TypeError("system2 should be a CoordSys3D \
                            instance")
        
        # 初始化输出的 Dyadic 对象
        outdyad = Dyadic.zero
        var = variables

        # 遍历 Dyadic 对象的组成部分，并计算每个部分的表达式
        for k, v in expr.components.items():
            # 计算表达式 v 在当前坐标系下的表达式，并与 k 的两个参数在当前和 system2 坐标系下的表达式点乘，然后累加到 outdyad 中
            outdyad += (express(v, system, variables=var) *
                        (express(k.args[0], system, variables=var) |
                         express(k.args[1], system2, variables=var)))

        # 返回处理后的 Dyadic 对象
        return outdyad

    else:
        # 如果同时提供了 system2 参数，则引发数值错误
        if system2 is not None:
            raise ValueError("system2 should not be provided for \
                                Vectors")
        
        # 如果 variables 参数为 True，则给定的 expr 是一个标量场（scalar field）
        if variables:
            # 将 expr 转换为符号表达式
            expr = sympify(expr)
            system_set = set()
            
            # 替换所有坐标变量，找到所有不属于当前系统的坐标系统，并创建替换字典
            for x in expr.atoms(BaseScalar):
                if x.system != system:
                    system_set.add(x.system)
            subs_dict = {}
            for f in system_set:
                subs_dict.update(f.scalar_map(system))
            
            # 使用替换字典对 expr 进行替换
            return expr.subs(subs_dict)
        
        # 如果 variables 参数为 False，直接返回 expr
        return expr
# 返回一个标量场或向量场沿给定向量方向导数的值，该向量方向是用坐标系表达的
def directional_derivative(field, direction_vector):
    # 导入获取坐标系函数
    from sympy.vector.operators import _get_coord_systems
    # 获取包含该场的坐标系
    coord_sys = _get_coord_systems(field)
    # 如果找到了坐标系
    if len(coord_sys) > 0:
        # TODO: 如果有多个坐标系，这里会随机选择一个：
        coord_sys = next(iter(coord_sys))
        # 将场用所选坐标系的变量表达
        field = express(field, coord_sys, variables=True)
        # 获取坐标系的基向量
        i, j, k = coord_sys.base_vectors()
        # 获取坐标系的基标量
        x, y, z = coord_sys.base_scalars()
        # 计算沿给定方向向量的方向导数
        out = Vector.dot(direction_vector, i) * diff(field, x)
        out += Vector.dot(direction_vector, j) * diff(field, y)
        out += Vector.dot(direction_vector, k) * diff(field, z)
        # 如果结果为零且场为向量，则结果设为零向量
        if out == 0 and isinstance(field, Vector):
            out = Vector.zero
        # 返回计算结果
        return out
    # 如果场是向量且没有找到坐标系，则返回零向量
    elif isinstance(field, Vector):
        return Vector.zero
    # 如果场是标量且没有找到坐标系，则返回零标量
    else:
        return S.Zero


# 返回给定场在所给坐标系的基标量下计算的拉普拉斯算子
def laplacian(expr):
    # 创建导数操作对象
    delop = Del()
    # 如果场是向量
    if expr.is_Vector:
        # 返回场的散度的梯度减去旋度的旋度，结果执行
        return (gradient(divergence(expr)) - curl(curl(expr))).doit()
    # 如果场是标量，则返回场的拉普拉斯算子
    return delop.dot(delop(expr)).doit()


# 检查给定场是否是保守场
def is_conservative(field):
    # 无论系统如何，场都是保守的
    # 使用 Vector 的 separate 方法返回的第一个坐标系
    # 这里应该有一个实际的返回值或操作
    # 检查 field 是否不是 Vector 类型，如果不是，则抛出类型错误异常
    if not isinstance(field, Vector):
        raise TypeError("field should be a Vector")
    
    # 检查 field 是否等于 Vector 类的静态属性 zero，如果是，则返回 True
    if field == Vector.zero:
        return True
    
    # 对 field 执行 curl() 操作，然后简化结果，再检查是否等于 Vector 类的静态属性 zero
    return curl(field).simplify() == Vector.zero
# 检查给定的场是否为无旋场
def is_solenoidal(field):
    """
    Checks if a field is solenoidal.

    Parameters
    ==========

    field : Vector
        The field to check for solenoidal property

    Examples
    ========

    >>> from sympy.vector import CoordSys3D
    >>> from sympy.vector import is_solenoidal
    >>> R = CoordSys3D('R')
    >>> is_solenoidal(R.y*R.z*R.i + R.x*R.z*R.j + R.x*R.y*R.k)
    True
    >>> is_solenoidal(R.y * R.j)
    False

    """

    # 如果field不是Vector类型，则抛出类型错误
    if not isinstance(field, Vector):
        raise TypeError("field should be a Vector")
    # 如果field等于Vector.zero，则返回True，即零向量是无旋场
    if field == Vector.zero:
        return True
    # 计算场的散度，并简化结果，若为零则表明是无旋场
    return divergence(field).simplify() is S.Zero


# 返回给定坐标系中场的标量势函数（不包括积分常数）
def scalar_potential(field, coord_sys):
    """
    Returns the scalar potential function of a field in a given
    coordinate system (without the added integration constant).

    Parameters
    ==========

    field : Vector
        The vector field whose scalar potential function is to be
        calculated

    coord_sys : CoordSys3D
        The coordinate system to do the calculation in

    Examples
    ========

    >>> from sympy.vector import CoordSys3D
    >>> from sympy.vector import scalar_potential, gradient
    >>> R = CoordSys3D('R')
    >>> scalar_potential(R.k, R) == R.z
    True
    >>> scalar_field = 2*R.x**2*R.y*R.z
    >>> grad_field = gradient(scalar_field)
    >>> scalar_potential(grad_field, R)
    2*R.x**2*R.y*R.z

    """

    # 检查场是否为保守场
    if not is_conservative(field):
        raise ValueError("Field is not conservative")
    # 如果field为零向量，则返回零
    if field == Vector.zero:
        return S.Zero
    # 将场表达为coord_sys坐标系下的表达式，替换变量
    field = express(field, coord_sys, variables=True)
    dimensions = coord_sys.base_vectors()
    scalars = coord_sys.base_scalars()
    # 计算标量势函数
    temp_function = integrate(field.dot(dimensions[0]), scalars[0])
    for i, dim in enumerate(dimensions[1:]):
        partial_diff = diff(temp_function, scalars[i + 1])
        partial_diff = field.dot(dim) - partial_diff
        temp_function += integrate(partial_diff, scalars[i + 1])
    return temp_function


# 返回给定场在特定坐标系下，两点之间的标量势差
def scalar_potential_difference(field, coord_sys, point1, point2):
    """
    Returns the scalar potential difference between two points in a
    certain coordinate system, wrt a given field.

    If a scalar field is provided, its values at the two points are
    considered. If a conservative vector field is provided, the values
    of its scalar potential function at the two points are used.

    Returns (potential at point2) - (potential at point1)

    The position vectors of the two Points are calculated wrt the
    origin of the coordinate system provided.

    Parameters
    ==========
    # 检查 coord_sys 是否为 CoordSys3D 类型，若不是则引发 TypeError 异常
    if not isinstance(coord_sys, CoordSys3D):
        raise TypeError("coord_sys must be a CoordSys3D")
    
    # 如果 field 是 Vector 类型，则获取其对应的标量势函数
    if isinstance(field, Vector):
        scalar_fn = scalar_potential(field, coord_sys)
    else:
        # 如果 field 是标量，则直接将其作为标量函数
        scalar_fn = field
    
    # 将起始点 point1 和第二点 point2 在给定坐标系 coord_sys 中表示出来
    origin = coord_sys.origin
    position1 = express(point1.position_wrt(origin), coord_sys,
                        variables=True)
    position2 = express(point2.position_wrt(origin), coord_sys,
                        variables=True)
    
    # 准备两个位置的替换字典，用于替换坐标变量
    subs_dict1 = {}
    subs_dict2 = {}
    scalars = coord_sys.base_scalars()
    
    # 遍历基向量，计算两个位置的点乘结果，构建替换字典
    for i, x in enumerate(coord_sys.base_vectors()):
        subs_dict1[scalars[i]] = x.dot(position1)
        subs_dict2[scalars[i]] = x.dot(position2)
    
    # 计算两点处的标量势差，并返回结果
    return scalar_fn.subs(subs_dict2) - scalar_fn.subs(subs_dict1)
# 定义一个函数，将矩阵形式的向量转换为 Vector 实例。
def matrix_to_vector(matrix, system):
    # 初始化输出向量为零向量
    outvec = Vector.zero
    # 获取给定坐标系的基向量列表
    vects = system.base_vectors()
    # 遍历矩阵的每个元素及其索引
    for i, x in enumerate(matrix):
        # 将矩阵元素乘以对应的基向量，并累加到输出向量中
        outvec += x * vects[i]
    # 返回转换后的向量
    return outvec


# 定义一个辅助函数，计算从 from_object 到 to_object 的路径，并返回首个共同祖先的索引及路径列表。
def _path(from_object, to_object):
    # 如果起始对象和目标对象的根不同，则抛出 ValueError 异常
    if from_object._root != to_object._root:
        raise ValueError("No connecting path found between " +
                         str(from_object) + " and " + str(to_object))

    # 初始化 other_path 为空列表
    other_path = []
    # 从目标对象开始，沿着 _parent 属性追溯到根对象，将沿途对象添加到 other_path 中
    obj = to_object
    while obj._parent is not None:
        other_path.append(obj)
        obj = obj._parent
    other_path.append(obj)
    
    # 将 other_path 转换为集合，加快查找速度
    object_set = set(other_path)
    # 初始化 from_path 为空列表
    from_path = []
    # 从起始对象开始，沿着 _parent 属性追溯到共同祖先之前，将沿途对象添加到 from_path 中
    obj = from_object
    while obj not in object_set:
        from_path.append(obj)
        obj = obj._parent
    
    # 计算首个共同祖先的索引
    index = len(from_path)
    # 将 from_path 扩展为完整路径，包括共同祖先到目标对象的路径部分
    from_path.extend(other_path[other_path.index(obj)::-1])
    
    # 返回首个共同祖先的索引及完整路径列表
    return index, from_path


# 定义一个函数，对一系列独立向量进行 Gram-Schmidt 过程正交化处理，并返回正交或正交归一化向量的列表。
def orthogonalize(*vlist, orthonormal=False):
    # 检查所有输入的向量是否为 Vector 类型，如果不是则抛出 TypeError 异常
    if not all(isinstance(vec, Vector) for vec in vlist):
        raise TypeError('Each element must be of Type Vector')
    
    # 初始化正交化后的向量列表为空
    ortho_vlist = []
    # 遍历向量列表 vlist 中的每个向量及其索引 i
    for i, term in enumerate(vlist):
        # 对于当前向量 term，遍历索引 i 之前的所有向量
        for j in range(i):
            # 从当前向量 term 中减去 vlist[j] 在 term 上的投影
            term -= ortho_vlist[j].projection(vlist[i])

        # TODO : 下面的这行代码引入了性能问题
        # 一旦解决了 issue #10279，需要修改此行代码。
        # 检查经简化后的 term 是否等于零向量
        if simplify(term).equals(Vector.zero):
            # 如果经简化后的 term 是零向量，则向量集合不是线性独立的，抛出 ValueError
            raise ValueError("Vector set not linearly independent")

        # 将处理后的 term 加入正交向量列表 ortho_vlist
        ortho_vlist.append(term)

    # 如果需要得到正交归一化的向量集合
    if orthonormal:
        # 对 ortho_vlist 中的每个向量进行归一化处理
        ortho_vlist = [vec.normalize() for vec in ortho_vlist]

    # 返回处理后的正交（或正交归一化）向量列表
    return ortho_vlist
```
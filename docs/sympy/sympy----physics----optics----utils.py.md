# `D:\src\scipysrc\sympy\sympy\physics\optics\utils.py`

```
"""
**Contains**

* refraction_angle
* fresnel_coefficients
* deviation
* brewster_angle
* critical_angle
* lens_makers_formula
* mirror_formula
* lens_formula
* hyperfocal_distance
* transverse_magnification
"""

__all__ = ['refraction_angle',
           'deviation',
           'fresnel_coefficients',
           'brewster_angle',
           'critical_angle',
           'lens_makers_formula',
           'mirror_formula',
           'lens_formula',
           'hyperfocal_distance',
           'transverse_magnification'
           ]

from sympy.core.numbers import (Float, I, oo, pi, zoo)  # 导入数学常数和类型
from sympy.core.singleton import S  # 导入符号 S
from sympy.core.symbol import Symbol  # 导入符号 Symbol
from sympy.core.sympify import sympify  # 导入 sympify 函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.trigonometric import (acos, asin, atan2, cos, sin, tan)  # 导入三角函数
from sympy.matrices.dense import Matrix  # 导入矩阵类型
from sympy.polys.polytools import cancel  # 导入多项式处理工具函数
from sympy.series.limits import Limit  # 导入极限计算类
from sympy.geometry.line import Ray3D  # 导入三维射线类
from sympy.geometry.util import intersection  # 导入几何工具函数 intersection
from sympy.geometry.plane import Plane  # 导入平面类
from sympy.utilities.iterables import is_sequence  # 导入判断是否为序列的函数 is_sequence
from .medium import Medium  # 导入自定义的介质类 Medium


def refractive_index_of_medium(medium):
    """
    Helper function that returns refractive index, given a medium
    """
    if isinstance(medium, Medium):
        n = medium.refractive_index
    else:
        n = sympify(medium)
    return n


def refraction_angle(incident, medium1, medium2, normal=None, plane=None):
    """
    This function calculates transmitted vector after refraction at planar
    surface. ``medium1`` and ``medium2`` can be ``Medium`` or any sympifiable object.
    If ``incident`` is a number then treated as angle of incidence (in radians)
    in which case refraction angle is returned.

    If ``incident`` is an object of `Ray3D`, `normal` also has to be an instance
    of `Ray3D` in order to get the output as a `Ray3D`. Please note that if
    plane of separation is not provided and normal is an instance of `Ray3D`,
    ``normal`` will be assumed to be intersecting incident ray at the plane of
    separation. This will not be the case when `normal` is a `Matrix` or
    any other sequence.
    If ``incident`` is an instance of `Ray3D` and `plane` has not been provided
    and ``normal`` is not `Ray3D`, output will be a `Matrix`.

    Parameters
    ==========

    incident : Matrix, Ray3D, sequence or a number
        Incident vector or angle of incidence
    medium1 : sympy.physics.optics.medium.Medium or sympifiable
        Medium 1 or its refractive index
    medium2 : sympy.physics.optics.medium.Medium or sympifiable
        Medium 2 or its refractive index
    normal : Matrix, Ray3D, or sequence
        Normal vector
    plane : Plane
        Plane of separation of the two media.

    Returns
    =======

    Returns an angle of refraction or a refracted ray depending on inputs.

    Examples
    ========

    >>> from sympy.physics.optics import refraction_angle

    """
    n1 = refractive_index_of_medium(medium1)
    n2 = refractive_index_of_medium(medium2)
    # 获取第一个介质的折射率

    # 检查是否提供了入射角而不是射线
    try:
        angle_of_incidence = float(incident)
    except TypeError:
        angle_of_incidence = None

    # 尝试获取临界角
    try:
        critical_angle_ = critical_angle(medium1, medium2)
    except (ValueError, TypeError):
        critical_angle_ = None

    if angle_of_incidence is not None:
        if normal is not None or plane is not None:
            raise ValueError('Normal/plane not allowed if incident is an angle')
        # 如果提供了入射角而不是射线，并且同时提供了法向量或平面参数，则引发错误

        if not 0.0 <= angle_of_incidence < pi*0.5:
            raise ValueError('Angle of incidence not in range [0:pi/2)')
        # 如果入射角不在 [0, pi/2) 的范围内，则引发错误

        if critical_angle_ and angle_of_incidence > critical_angle_:
            raise ValueError('Ray undergoes total internal reflection')
        # 如果存在临界角并且入射角大于临界角，则引发全反射错误
        return asin(n1*sin(angle_of_incidence)/n2)
        # 返回折射角度

    # 将入射光视为射线以下
    # 标志，用于检查是否返回 Ray3D 对象
    return_ray = False

    if plane is not None and normal is not None:
        raise ValueError("Either plane or normal is acceptable.")
        # 如果同时提供了平面和法向量参数，则引发错误

    if not isinstance(incident, Matrix):
        if is_sequence(incident):
            _incident = Matrix(incident)
        elif isinstance(incident, Ray3D):
            _incident = Matrix(incident.direction_ratio)
        else:
            raise TypeError(
                "incident should be a Matrix, Ray3D, or sequence")
        # 如果 incident 不是 Matrix，则尝试转换为 Matrix

    else:
        _incident = incident
        # 否则直接使用 incident

    # 如果提供了平面，则从平面中获取法向量的方向比率，否则使用 normal 参数
    if plane is not None:
        if not isinstance(plane, Plane):
            # 如果 plane 不是 Plane 类的实例，则抛出类型错误异常
            raise TypeError("plane should be an instance of geometry.plane.Plane")
        # 如果存在 plane，可以计算入射光线和平面的交点，并返回 Ray3D 的实例
        if isinstance(incident, Ray3D):
            # 如果 incident 是 Ray3D 的实例，说明可以返回 Ray3D
            return_ray = True
            # 计算入射光线与平面的交点，并取第一个交点
            intersection_pt = plane.intersection(incident)[0]
        _normal = Matrix(plane.normal_vector)
    else:
        if not isinstance(normal, Matrix):
            if is_sequence(normal):
                # 如果 normal 是一个序列，则将其转换为 Matrix 类型
                _normal = Matrix(normal)
            elif isinstance(normal, Ray3D):
                # 如果 normal 是 Ray3D 的实例，则使用其方向比率创建 Matrix
                _normal = Matrix(normal.direction_ratio)
                if isinstance(incident, Ray3D):
                    # 如果 incident 也是 Ray3D 的实例，则计算它们的交点
                    intersection_pt = intersection(incident, normal)
                    if len(intersection_pt) == 0:
                        # 如果没有交点，则抛出 ValueError
                        raise ValueError(
                            "Normal isn't concurrent with the incident ray.")
                    else:
                        return_ray = True
                        # 取第一个交点作为返回的 intersection_pt
                        intersection_pt = intersection_pt[0]
            else:
                # 如果 normal 不是 Matrix 或 Ray3D 的实例，抛出类型错误异常
                raise TypeError(
                    "Normal should be a Matrix, Ray3D, or sequence")
        else:
            # 如果 normal 已经是 Matrix 的实例，则直接赋值给 _normal
            _normal = normal

    eta = n1/n2  # 计算相对折射率
    # 计算向量的大小
    mag_incident = sqrt(sum(i**2 for i in _incident))
    mag_normal = sqrt(sum(i**2 for i in _normal))
    # 将向量单位化（转换为单位向量）
    _incident /= mag_incident
    _normal /= mag_normal
    c1 = -_incident.dot(_normal)  # cos(入射角)
    cs2 = 1 - eta**2*(1 - c1**2)  # cos(折射角)**2
    if cs2.is_negative:  # 这是全反射的情况
        # 如果 cs2 是负数，表示全反射，则返回零向量 S.Zero
        return S.Zero
    drs = eta*_incident + (eta*c1 - sqrt(cs2))*_normal
    # 将单位向量乘以其大小
    drs = drs*mag_incident
    if not return_ray:
        # 如果不需要返回 Ray3D 对象，则返回向量 drs
        return drs
    else:
        # 如果需要返回 Ray3D 对象，则返回从 intersection_pt 出发，方向为 drs 的 Ray3D 对象
        return Ray3D(intersection_pt, direction_ratio=drs)
def fresnel_coefficients(angle_of_incidence, medium1, medium2):
    """
    This function uses Fresnel equations to calculate reflection and
    transmission coefficients. Those are obtained for both polarisations
    when the electric field vector is in the plane of incidence (labelled 'p')
    and when the electric field vector is perpendicular to the plane of
    incidence (labelled 's'). There are four real coefficients unless the
    incident ray reflects in total internal in which case there are two complex
    ones. Angle of incidence is the angle between the incident ray and the
    surface normal. ``medium1`` and ``medium2`` can be ``Medium`` or any
    sympifiable object.

    Parameters
    ==========

    angle_of_incidence : sympifiable
        The angle between the incident ray and the surface normal.

    medium1 : Medium or sympifiable
        Medium 1 or its refractive index

    medium2 : Medium or sympifiable
        Medium 2 or its refractive index

    Returns
    =======

    Returns a list with four real Fresnel coefficients:
    [reflection p (TM), reflection s (TE),
    transmission p (TM), transmission s (TE)]
    If the ray undergoes total internal reflection then returns a
    list of two complex Fresnel coefficients:
    [reflection p (TM), reflection s (TE)]

    Examples
    ========

    >>> from sympy.physics.optics import fresnel_coefficients
    >>> fresnel_coefficients(0.3, 1, 2)
    [0.317843553417859, -0.348645229818821,
            0.658921776708929, 0.651354770181179]
    >>> fresnel_coefficients(0.6, 2, 1)
    [-0.235625382192159 - 0.971843958291041*I,
             0.816477005968898 - 0.577377951366403*I]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fresnel_equations
    """
    # 检查角度是否在合理范围内 [0, pi/2)
    if not 0 <= 2*angle_of_incidence < pi:
        raise ValueError('Angle of incidence not in range [0:pi/2)')

    # 获取介质1和介质2的折射率
    n1 = refractive_index_of_medium(medium1)
    n2 = refractive_index_of_medium(medium2)

    # 计算折射角
    angle_of_refraction = asin(n1*sin(angle_of_incidence)/n2)

    # 尝试获取临界角，用于判断是否存在全反射
    try:
        angle_of_total_internal_reflection_onset = critical_angle(n1, n2)
    except ValueError:
        angle_of_total_internal_reflection_onset = None

    # 如果不存在全反射或角度小于临界角，则计算四个实数Fresnel系数
    if angle_of_total_internal_reflection_onset is None or \
            angle_of_total_internal_reflection_onset > angle_of_incidence:
        R_s = -sin(angle_of_incidence - angle_of_refraction) \
                / sin(angle_of_incidence + angle_of_refraction)
        R_p = tan(angle_of_incidence - angle_of_refraction) \
                / tan(angle_of_incidence + angle_of_refraction)
        T_s = 2 * sin(angle_of_refraction) * cos(angle_of_incidence) \
                / sin(angle_of_incidence + angle_of_refraction)
        T_p = 2 * sin(angle_of_refraction) * cos(angle_of_incidence) \
                / (sin(angle_of_incidence + angle_of_refraction) \
                * cos(angle_of_incidence - angle_of_refraction))
        return [R_p, R_s, T_p, T_s]
    # 如果条件不满足，则执行以下计算
    else:
        # 计算反射系数 R_s，其中 n 是折射率比值 n2/n1
        n = n2/n1
        # 计算 s 极化的反射系数 R_s，使用复数计算平方根
        R_s = cancel((cos(angle_of_incidence)-\
                I*sqrt(sin(angle_of_incidence)**2 - n**2))\
                /(cos(angle_of_incidence)+\
                I*sqrt(sin(angle_of_incidence)**2 - n**2)))
        # 计算 p 极化的反射系数 R_p，使用复数计算平方根
        R_p = cancel((n**2*cos(angle_of_incidence)-\
                I*sqrt(sin(angle_of_incidence)**2 - n**2))\
                /(n**2*cos(angle_of_incidence)+\
                I*sqrt(sin(angle_of_incidence)**2 - n**2)))
        # 返回计算得到的 R_p 和 R_s 的列表
        return [R_p, R_s]
# 计算射线由于在平面表面折射而产生的偏转角度的函数

def deviation(incident, medium1, medium2, normal=None, plane=None):
    """
    This function calculates the angle of deviation of a ray
    due to refraction at planar surface.

    Parameters
    ==========

    incident : Matrix, Ray3D, sequence or float
        Incident vector or angle of incidence
        射线的入射向量或入射角度
    medium1 : sympy.physics.optics.medium.Medium or sympifiable
        Medium 1 or its refractive index
        第一个介质或其折射率
    medium2 : sympy.physics.optics.medium.Medium or sympifiable
        Medium 2 or its refractive index
        第二个介质或其折射率
    normal : Matrix, Ray3D, or sequence
        Normal vector
        法线向量
    plane : Plane
        Plane of separation of the two media.
        两个介质的分界面

    Returns angular deviation between incident and refracted rays
    返回入射光和折射光之间的角度偏移量

    Examples
    ========

    >>> from sympy.physics.optics import deviation
    >>> from sympy.geometry import Point3D, Ray3D, Plane
    >>> from sympy.matrices import Matrix
    >>> from sympy import symbols
    >>> n1, n2 = symbols('n1, n2')
    >>> n = Matrix([0, 0, 1])
    >>> P = Plane(Point3D(0, 0, 0), normal_vector=[0, 0, 1])
    >>> r1 = Ray3D(Point3D(-1, -1, 1), Point3D(0, 0, 0))
    >>> deviation(r1, 1, 1, n)
    0
    >>> deviation(r1, n1, n2, plane=P)
    -acos(-sqrt(-2*n1**2/(3*n2**2) + 1)) + acos(-sqrt(3)/3)
    >>> round(deviation(0.1, 1.2, 1.5), 5)
    -0.02005
    """
    # 计算折射角度
    refracted = refraction_angle(incident,
                                 medium1,
                                 medium2,
                                 normal=normal,
                                 plane=plane)
    
    try:
        # 尝试将入射角度转换为浮点数
        angle_of_incidence = Float(incident)
    except TypeError:
        # 如果无法转换，设为 None
        angle_of_incidence = None

    # 如果入射角度可用，则返回折射角度减去入射角度的浮点数值
    if angle_of_incidence is not None:
        return float(refracted) - angle_of_incidence
    # 如果折射率不为零
    if refracted != 0:
        # 如果折射率是 Ray3D 类型的实例
        if isinstance(refracted, Ray3D):
            # 将折射率的方向比率转换为 Matrix 对象
            refracted = Matrix(refracted.direction_ratio)

        # 如果入射光不是 Matrix 对象
        if not isinstance(incident, Matrix):
            # 如果入射光是一个可迭代序列
            if is_sequence(incident):
                # 创建一个 Matrix 对象来表示入射光
                _incident = Matrix(incident)
            # 如果入射光是 Ray3D 类型的实例
            elif isinstance(incident, Ray3D):
                # 使用其方向比率创建一个 Matrix 对象来表示入射光
                _incident = Matrix(incident.direction_ratio)
            else:
                # 抛出类型错误，要求 incident 应为 Matrix、Ray3D 或序列
                raise TypeError(
                    "incident should be a Matrix, Ray3D, or sequence")
        else:
            # 如果入射光已经是 Matrix 对象，则直接使用它
            _incident = incident

        # 如果平面未指定
        if plane is None:
            # 如果法向量不是 Matrix 对象
            if not isinstance(normal, Matrix):
                # 如果法向量是一个可迭代序列
                if is_sequence(normal):
                    # 创建一个 Matrix 对象来表示法向量
                    _normal = Matrix(normal)
                # 如果法向量是 Ray3D 类型的实例
                elif isinstance(normal, Ray3D):
                    # 使用其方向比率创建一个 Matrix 对象来表示法向量
                    _normal = Matrix(normal.direction_ratio)
                else:
                    # 抛出类型错误，要求 normal 应为 Matrix、Ray3D 或序列
                    raise TypeError(
                        "normal should be a Matrix, Ray3D, or sequence")
            else:
                # 如果法向量已经是 Matrix 对象，则直接使用它
                _normal = normal
        else:
            # 如果指定了平面，则使用平面的法向量创建一个 Matrix 对象来表示法向量
            _normal = Matrix(plane.normal_vector)

        # 计算入射光、法向量和折射率的模长
        mag_incident = sqrt(sum(i**2 for i in _incident))
        mag_normal = sqrt(sum(i**2 for i in _normal))
        mag_refracted = sqrt(sum(i**2 for i in refracted))

        # 归一化入射光、法向量和折射率
        _incident /= mag_incident
        _normal /= mag_normal
        refracted /= mag_refracted

        # 计算入射角和折射角
        i = acos(_incident.dot(_normal))
        r = acos(refracted.dot(_normal))

        # 返回入射角和折射角的差
        return i - r
# 计算从介质1到介质2的布鲁斯特角（反射角），返回结果以弧度表示
def brewster_angle(medium1, medium2):
    n1 = refractive_index_of_medium(medium1)  # 获取介质1的折射率
    n2 = refractive_index_of_medium(medium2)  # 获取介质2的折射率
    return atan2(n2, n1)  # 使用反正切函数计算布鲁斯特角

# 计算从介质1到介质2的临界入射角（导致全内反射的角度），返回结果以弧度表示
def critical_angle(medium1, medium2):
    n1 = refractive_index_of_medium(medium1)  # 获取介质1的折射率
    n2 = refractive_index_of_medium(medium2)  # 获取介质2的折射率
    if n2 > n1:
        raise ValueError('Total internal reflection impossible for n1 < n2')  # 如果介质2的折射率大于介质1，抛出异常
    else:
        return asin(n2/n1)  # 使用反正弦函数计算临界角

# 计算透镜的焦距，使用笛卡尔符号约定
def lens_makers_formula(n_lens, n_surr, r1, r2, d=0):
    if isinstance(n_lens, Medium):
        n_lens = n_lens.refractive_index  # 如果透镜类型是Medium对象，获取其折射率
    else:
        n_lens = sympify(n_lens)  # 否则，将透镜的折射率转换为符号表达式
    if isinstance(n_surr, Medium):
        n_surr = n_surr.refractive_index  # 如果环境类型是Medium对象，获取其折射率
    else:
        n_surr = sympify(n_surr)  # 否则，将环境的折射率转换为符号表达式
    d = sympify(d)  # 将透镜的厚度转换为符号表达式

    # 计算透镜的焦距
    focal_length = 1 / ((n_lens - n_surr) / n_surr * (1 / r1 - 1 / r2 + (((n_lens - n_surr) * d) / (n_lens * r1 * r2))))

    if focal_length == zoo:  # 如果焦距为无穷大（即平面镜），返回无穷大符号
        return S.Infinity
    return focal_length  # 否则，返回计算出的焦距

# 提供镜子公式的一个参数（焦距、物距或像距），在给定两个参数的情况下有效，仅适用于平行光线
def mirror_formula(focal_length=None, u=None, v=None):
    pass  # 函数尚未实现，暂不提供注释
    u : sympifiable
        Distance of object from the pole on
        the principal axis.
    v : sympifiable
        Distance of the image from the pole
        on the principal axis.

    Examples
    ========

    >>> from sympy.physics.optics import mirror_formula
    >>> from sympy.abc import f, u, v
    >>> mirror_formula(focal_length=f, u=u)
    f*u/(-f + u)
    >>> mirror_formula(focal_length=f, v=v)
    f*v/(-f + v)
    >>> mirror_formula(u=u, v=v)
    u*v/(u + v)

    """
    # 检查给定的参数情况，如果三个参数都提供了，抛出错误
    if focal_length and u and v:
        raise ValueError("Please provide only two parameters")

    # 将输入的符号化为Sympy符号
    focal_length = sympify(focal_length)
    u = sympify(u)
    v = sympify(v)
    
    # 处理无穷大的情况
    if u is oo:
        _u = Symbol('u')
    if v is oo:
        _v = Symbol('v')
    if focal_length is oo:
        _f = Symbol('f')
    
    # 处理只提供两个参数的情况
    if focal_length is None:
        if u is oo and v is oo:
            # 当u和v都为无穷大时，求极限
            return Limit(Limit(_v*_u/(_v + _u), _u, oo), _v, oo).doit()
        if u is oo:
            # 当u为无穷大时，求极限
            return Limit(v*_u/(v + _u), _u, oo).doit()
        if v is oo:
            # 当v为无穷大时，求极限
            return Limit(_v*u/(_v + u), _v, oo).doit()
        # 正常情况下的计算
        return v*u/(v + u)
    
    if u is None:
        if v is oo and focal_length is oo:
            # 当v和focal_length都为无穷大时，求极限
            return Limit(Limit(_v*_f/(_v - _f), _v, oo), _f, oo).doit()
        if v is oo:
            # 当v为无穷大时，求极限
            return Limit(_v*focal_length/(_v - focal_length), _v, oo).doit()
        if focal_length is oo:
            # 当focal_length为无穷大时，求极限
            return Limit(v*_f/(v - _f), _f, oo).doit()
        # 正常情况下的计算
        return v*focal_length/(v - focal_length)
    
    if v is None:
        if u is oo and focal_length is oo:
            # 当u和focal_length都为无穷大时，求极限
            return Limit(Limit(_u*_f/(_u - _f), _u, oo), _f, oo).doit()
        if u is oo:
            # 当u为无穷大时，求极限
            return Limit(_u*focal_length/(_u - focal_length), _u, oo).doit()
        if focal_length is oo:
            # 当focal_length为无穷大时，求极限
            return Limit(u*_f/(u - _f), _f, oo).doit()
        # 正常情况下的计算
        return u*focal_length/(u - focal_length)
# 计算透镜公式，根据给定的焦距、物距和像距计算另外一个参数
def lens_formula(focal_length=None, u=None, v=None):
    """
    This function provides one of the three parameters
    when two of them are supplied.
    This is valid only for paraxial rays.

    Parameters
    ==========

    focal_length : sympifiable
        焦距。
    u : sympifiable
        物距，即物体到透镜的距离。
    v : sympifiable
        像距，即像到透镜的距离。

    Examples
    ========

    >>> from sympy.physics.optics import lens_formula
    >>> from sympy.abc import f, u, v
    >>> lens_formula(focal_length=f, u=u)
    f*u/(f + u)
    >>> lens_formula(focal_length=f, v=v)
    f*v/(f - v)
    >>> lens_formula(u=u, v=v)
    u*v/(u - v)

    """
    # 如果三个参数都提供了，则抛出错误
    if focal_length and u and v:
        raise ValueError("Please provide only two parameters")

    # 将输入的参数转换为Sympy的表达式
    focal_length = sympify(focal_length)
    u = sympify(u)
    v = sympify(v)

    # 处理无穷大的情况
    if u is oo:
        _u = Symbol('u')
    if v is oo:
        _v = Symbol('v')
    if focal_length is oo:
        _f = Symbol('f')

    # 根据不同情况计算并返回焦距、物距或像距中的未知量
    if focal_length is None:
        if u is oo and v is oo:
            return Limit(Limit(_v*_u/(_u - _v), _u, oo), _v, oo).doit()
        if u is oo:
            return Limit(v*_u/(_u - v), _u, oo).doit()
        if v is oo:
            return Limit(_v*u/(u - _v), _v, oo).doit()
        return v*u/(u - v)
    if u is None:
        if v is oo and focal_length is oo:
            return Limit(Limit(_v*_f/(_f - _v), _v, oo), _f, oo).doit()
        if v is oo:
            return Limit(_v*focal_length/(focal_length - _v), _v, oo).doit()
        if focal_length is oo:
            return Limit(v*_f/(_f - v), _f, oo).doit()
        return v*focal_length/(focal_length - v)
    if v is None:
        if u is oo and focal_length is oo:
            return Limit(Limit(_u*_f/(_u + _f), _u, oo), _f, oo).doit()
        if u is oo:
            return Limit(_u*focal_length/(_u + focal_length), _u, oo).doit()
        if focal_length is oo:
            return Limit(u*_f/(u + _f), _f, oo).doit()
        return u*focal_length/(u + focal_length)

# 计算超焦距距离，根据给定的焦距、光圈数和图像格式的圆形模糊度
def hyperfocal_distance(f, N, c):
    """

    Parameters
    ==========

    f: sympifiable
        透镜的焦距。

    N: sympifiable
        透镜的F数。

    c: sympifiable
        图像格式的圆形模糊度（CoC）。

    Example
    =======

    >>> from sympy.physics.optics import hyperfocal_distance
    >>> round(hyperfocal_distance(f = 0.5, N = 8, c = 0.0033), 2)
    9.47
    """

    # 将输入的参数转换为Sympy的表达式，并计算超焦距距离
    f = sympify(f)
    N = sympify(N)
    c = sympify(c)

    return (1/(N * c))*(f**2)

# 计算横向放大率，即反射镜中图像大小与物体大小的比率
def transverse_magnification(si, so):
    """

    Calculates the transverse magnification upon reflection in a mirror,
    which is the ratio of the image size to the object size.

    Parameters
    ==========

    so: sympifiable
        物体到透镜的距离。

    si: sympifiable
        像到透镜的距离。

    Example
    =======
    # 导入 sympy.physics.optics 模块中的 transverse_magnification 函数
    >>> from sympy.physics.optics import transverse_magnification
    # 使用 transverse_magnification 函数计算放大率，给定参数为 30 和 15
    >>> transverse_magnification(30, 15)
    # 返回结果为 -2

    """

    # 将 si 和 so 转换为 sympy 符号对象（如果尚未是符号对象）
    si = sympify(si)
    so = sympify(so)

    # 返回计算的光学放大率，计算公式为 -(si/so)
    return (-(si/so))
```
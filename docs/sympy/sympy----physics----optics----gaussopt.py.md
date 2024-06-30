# `D:\src\scipysrc\sympy\sympy\physics\optics\gaussopt.py`

```
"""
Gaussian optics.

The module implements:

- Ray transfer matrices for geometrical and gaussian optics.

  See RayTransferMatrix, GeometricRay and BeamParameter

- Conjugation relations for geometrical and gaussian optics.

  See geometric_conj*, gauss_conj and conjugate_gauss_beams

The conventions for the distances are as follows:

focal distance
    positive for convergent lenses
object distance
    positive for real objects
image distance
    positive for real images
"""

__all__ = [
    'RayTransferMatrix',
    'FreeSpace',
    'FlatRefraction',
    'CurvedRefraction',
    'FlatMirror',
    'CurvedMirror',
    'ThinLens',
    'GeometricRay',
    'BeamParameter',
    'waist2rayleigh',
    'rayleigh2waist',
    'geometric_conj_ab',
    'geometric_conj_af',
    'geometric_conj_bf',
    'gaussian_conj',
    'conjugate_gauss_beams',
]

# Import necessary modules and functions from SymPy
from sympy.core.expr import Expr
from sympy.core.numbers import (I, pi)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan2
from sympy.matrices.dense import Matrix, MutableDenseMatrix
from sympy.polys.rationaltools import together
from sympy.utilities.misc import filldedent

###
# A, B, C, D matrices
###


class RayTransferMatrix(MutableDenseMatrix):
    """
    Base class for a Ray Transfer Matrix.

    It should be used if there is not already a more specific subclass mentioned
    in See Also.

    Parameters
    ==========

    parameters :
        A, B, C and D or 2x2 matrix (Matrix(2, 2, [A, B, C, D]))

    Examples
    ========

    >>> from sympy.physics.optics import RayTransferMatrix, ThinLens
    >>> from sympy import Symbol, Matrix

    >>> mat = RayTransferMatrix(1, 2, 3, 4)
    >>> mat
    Matrix([
    [1, 2],
    [3, 4]])

    >>> RayTransferMatrix(Matrix([[1, 2], [3, 4]]))
    Matrix([
    [1, 2],
    [3, 4]])

    >>> mat.A
    1

    >>> f = Symbol('f')
    >>> lens = ThinLens(f)
    >>> lens
    Matrix([
    [   1, 0],
    [-1/f, 1]])

    >>> lens.C
    -1/f

    See Also
    ========

    GeometricRay, BeamParameter,
    FreeSpace, FlatRefraction, CurvedRefraction,
    FlatMirror, CurvedMirror, ThinLens

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    """

    def __new__(cls, *args):
        # Determine if arguments provided are suitable for creating a 2x2 matrix
        if len(args) == 4:
            temp = ((args[0], args[1]), (args[2], args[3]))
        elif len(args) == 1 \
            and isinstance(args[0], Matrix) \
                and args[0].shape == (2, 2):
            temp = args[0]
        else:
            # Raise an error if arguments do not match expected formats
            raise ValueError(filldedent('''
                Expecting 2x2 Matrix or the 4 elements of
                the Matrix but got %s''' % str(args)))
        # Create and return a new Matrix object with the determined structure
        return Matrix.__new__(cls, temp)
    # 定义矩阵乘法的特殊方法，用于处理不同类型的乘法操作
    def __mul__(self, other):
        # 如果 other 是 RayTransferMatrix 类型，则返回一个新的 RayTransferMatrix 对象
        if isinstance(other, RayTransferMatrix):
            return RayTransferMatrix(Matrix(self)*Matrix(other))
        # 如果 other 是 GeometricRay 类型，则返回一个新的 GeometricRay 对象
        elif isinstance(other, GeometricRay):
            return GeometricRay(Matrix(self)*Matrix(other))
        # 如果 other 是 BeamParameter 类型，则进行特定计算，返回一个新的 BeamParameter 对象
        elif isinstance(other, BeamParameter):
            # 计算矩阵乘积，并根据结果生成新的 BeamParameter 对象
            temp = Matrix(self)*Matrix(((other.q,), (1,)))
            q = (temp[0]/temp[1]).expand(complex=True)
            return BeamParameter(other.wavelen,
                                 together(re(q)),
                                 z_r=together(im(q)))
        else:
            # 对于其他情况，调用父类 Matrix 的 __mul__ 方法进行处理
            return Matrix.__mul__(self, other)

    @property
    def A(self):
        """
        矩阵的 A 参数。

        示例
        ========

        >>> from sympy.physics.optics import RayTransferMatrix
        >>> mat = RayTransferMatrix(1, 2, 3, 4)
        >>> mat.A
        1
        """
        return self[0, 0]

    @property
    def B(self):
        """
        矩阵的 B 参数。

        示例
        ========

        >>> from sympy.physics.optics import RayTransferMatrix
        >>> mat = RayTransferMatrix(1, 2, 3, 4)
        >>> mat.B
        2
        """
        return self[0, 1]

    @property
    def C(self):
        """
        矩阵的 C 参数。

        示例
        ========

        >>> from sympy.physics.optics import RayTransferMatrix
        >>> mat = RayTransferMatrix(1, 2, 3, 4)
        >>> mat.C
        3
        """
        return self[1, 0]

    @property
    def D(self):
        """
        矩阵的 D 参数。

        示例
        ========

        >>> from sympy.physics.optics import RayTransferMatrix
        >>> mat = RayTransferMatrix(1, 2, 3, 4)
        >>> mat.D
        4
        """
        return self[1, 1]
class FreeSpace(RayTransferMatrix):
    """
    Ray Transfer Matrix for free space.

    Parameters
    ==========

    distance : The distance traveled through free space.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import FreeSpace
    >>> from sympy import symbols
    >>> d = symbols('d')
    >>> FreeSpace(d)
    Matrix([
    [1, d],
    [0, 1]])
    """
    def __new__(cls, d):
        # 创建一个自由空间的光线传输矩阵，参数为距离 d
        return RayTransferMatrix.__new__(cls, 1, d, 0, 1)


class FlatRefraction(RayTransferMatrix):
    """
    Ray Transfer Matrix for refraction.

    Parameters
    ==========

    n1 : Refractive index of the first medium.
    n2 : Refractive index of the second medium.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import FlatRefraction
    >>> from sympy import symbols
    >>> n1, n2 = symbols('n1 n2')
    >>> FlatRefraction(n1, n2)
    Matrix([
    [1,     0],
    [0, n1/n2]])
    """
    def __new__(cls, n1, n2):
        # 创建折射光线传输矩阵，参数为两个介质的折射率 n1 和 n2
        n1, n2 = map(sympify, (n1, n2))
        return RayTransferMatrix.__new__(cls, 1, 0, 0, n1/n2)


class CurvedRefraction(RayTransferMatrix):
    """
    Ray Transfer Matrix for refraction on curved interface.

    Parameters
    ==========

    R : Radius of curvature (positive for concave).
    n1 : Refractive index of the first medium.
    n2 : Refractive index of the second medium.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import CurvedRefraction
    >>> from sympy import symbols
    >>> R, n1, n2 = symbols('R n1 n2')
    >>> CurvedRefraction(R, n1, n2)
    Matrix([
    [               1,     0],
    [(n1 - n2)/(R*n2), n1/n2]])
    """
    def __new__(cls, R, n1, n2):
        # 创建曲面折射光线传输矩阵，参数为曲率半径 R，介质折射率 n1 和 n2
        R, n1, n2 = map(sympify, (R, n1, n2))
        return RayTransferMatrix.__new__(cls, 1, 0, (n1 - n2)/R/n2, n1/n2)


class FlatMirror(RayTransferMatrix):
    """
    Ray Transfer Matrix for reflection.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import FlatMirror
    >>> FlatMirror()
    Matrix([
    [1, 0],
    [0, 1]])
    """
    def __new__(cls):
        # 创建反射光线传输矩阵，表示平面镜
        return RayTransferMatrix.__new__(cls, 1, 0, 0, 1)


class CurvedMirror(RayTransferMatrix):
    """
    Ray Transfer Matrix for reflection from curved surface.

    Parameters
    ==========

    R : Radius of curvature (positive for concave).

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import CurvedMirror
    >>> from sympy import symbols
    >>> R = symbols('R')
    >>> CurvedMirror(R)
    Matrix([
    [   1, 0],
    [-2/R, 1]])
    """
    def __new__(cls, R):
        # 创建曲面镜反射光线传输矩阵，参数为曲率半径 R
        R = sympify(R)
        return RayTransferMatrix.__new__(cls, 1, 0, -2/R, 1)


class ThinLens(RayTransferMatrix):
    """
    Ray Transfer Matrix for a thin lens.

    Parameters
    ==========

    f : The focal distance of the lens.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========
    # 定义一个名为 RayTransferMatrix 的类，用于光学中的射线传输矩阵计算

    Examples
    ========

    # 导入 ThinLens 类和 symbols 函数
    >>> from sympy.physics.optics import ThinLens
    >>> from sympy import symbols
    # 定义符号变量 f
    >>> f = symbols('f')
    # 创建 ThinLens 对象，传入 f 作为参数
    >>> ThinLens(f)
    # 返回一个 2x2 的矩阵对象，表示薄透镜的光学传输特性
    Matrix([
    [   1, 0],
    [-1/f, 1]])
    """
    # 定义 __new__ 方法，用于创建 RayTransferMatrix 类的新实例
    def __new__(cls, f):
        # 将 f 转换为符号表达式
        f = sympify(f)
        # 调用 RayTransferMatrix 类的父类（可能是 Matrix 类）的 __new__ 方法，
        # 创建一个新的 RayTransferMatrix 对象，传入参数 1, 0, -1/f, 1
        return RayTransferMatrix.__new__(cls, 1, 0, -1/f, 1)
###
# Representation for geometric ray
###

class GeometricRay(MutableDenseMatrix):
    """
    Representation for a geometric ray in the Ray Transfer Matrix formalism.

    Parameters
    ==========

    h : height, and
    angle : angle, or
    matrix : a 2x1 matrix (Matrix(2, 1, [height, angle]))

    Examples
    ========

    >>> from sympy.physics.optics import GeometricRay, FreeSpace
    >>> from sympy import symbols, Matrix
    >>> d, h, angle = symbols('d, h, angle')

    >>> GeometricRay(h, angle)
    Matrix([
    [    h],
    [angle]])

    >>> FreeSpace(d)*GeometricRay(h, angle)
    Matrix([
    [angle*d + h],
    [      angle]])

    >>> GeometricRay( Matrix( ((h,), (angle,)) ) )
    Matrix([
    [    h],
    [angle]])

    See Also
    ========

    RayTransferMatrix

    """

    def __new__(cls, *args):
        # Check if the arguments form a valid 2x1 Matrix or handle two separate arguments
        if len(args) == 1 and isinstance(args[0], Matrix) \
                and args[0].shape == (2, 1):
            temp = args[0]
        elif len(args) == 2:
            temp = ((args[0],), (args[1],))
        else:
            # Raise an error if the arguments don't match the expected format
            raise ValueError(filldedent('''
                Expecting 2x1 Matrix or the 2 elements of
                the Matrix but got %s''' % str(args)))
        return Matrix.__new__(cls, temp)

    @property
    def height(self):
        """
        The distance from the optical axis.

        Examples
        ========

        >>> from sympy.physics.optics import GeometricRay
        >>> from sympy import symbols
        >>> h, angle = symbols('h, angle')
        >>> gRay = GeometricRay(h, angle)
        >>> gRay.height
        h
        """
        return self[0]

    @property
    def angle(self):
        """
        The angle with the optical axis.

        Examples
        ========

        >>> from sympy.physics.optics import GeometricRay
        >>> from sympy import symbols
        >>> h, angle = symbols('h, angle')
        >>> gRay = GeometricRay(h, angle)
        >>> gRay.angle
        angle
        """
        return self[1]


###
# Representation for gauss beam
###

class BeamParameter(Expr):
    """
    Representation for a gaussian ray in the Ray Transfer Matrix formalism.

    Parameters
    ==========

    wavelen : the wavelength,
    z : the distance to waist, and
    w : the waist, or
    z_r : the rayleigh range.
    n : the refractive index of medium.

    Examples
    ========

    >>> from sympy.physics.optics import BeamParameter
    >>> p = BeamParameter(530e-9, 1, w=1e-3)
    >>> p.q
    1 + 1.88679245283019*I*pi

    >>> p.q.n()
    1.0 + 5.92753330865999*I
    >>> p.w_0.n()
    0.00100000000000000
    >>> p.z_r.n()
    5.92753330865999

    >>> from sympy.physics.optics import FreeSpace
    >>> fs = FreeSpace(10)
    >>> p1 = fs*p
    >>> p.w.n()
    0.00101413072159615
    >>> p1.w.n()
    0.00210803120913829

    See Also
    ========

    RayTransferMatrix

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Complex_beam_parameter

    """
    """
    #TODO A class Complex may be implemented. The BeamParameter may
    # subclass it. See:
    # https://groups.google.com/d/topic/sympy/7XkU07NRBEs/discussion

    # 定义一个名为BeamParameter的类，可能会实现一个复数类Complex的子类，
    # 参考链接：https://groups.google.com/d/topic/sympy/7XkU07NRBEs/discussion
    """

    # 定义一个新的类方法__new__，用于创建BeamParameter对象
    def __new__(cls, wavelen, z, z_r=None, w=None, n=1):
        # 将输入的波长wavelen、位置z和折射率n转换为符号表达式
        wavelen = sympify(wavelen)
        z = sympify(z)
        n = sympify(n)

        # 根据参数情况计算焦距z_r
        if z_r is not None and w is None:
            z_r = sympify(z_r)
        elif w is not None and z_r is None:
            # 如果给定波束半径w而没有焦距z_r，则计算焦距z_r
            z_r = waist2rayleigh(sympify(w), wavelen, n)
        elif z_r is None and w is None:
            # 如果既未指定波束半径w也未指定焦距z_r，则抛出数值错误
            raise ValueError('Must specify one of w and z_r.')

        # 调用父类Expr的__new__方法创建对象，并返回
        return Expr.__new__(cls, wavelen, z, z_r, n)

    # 定义属性wavelen，返回BeamParameter对象的波长参数
    @property
    def wavelen(self):
        return self.args[0]

    # 定义属性z，返回BeamParameter对象的位置参数
    @property
    def z(self):
        return self.args[1]

    # 定义属性z_r，返回BeamParameter对象的焦距参数
    @property
    def z_r(self):
        return self.args[2]

    # 定义属性n，返回BeamParameter对象的折射率参数
    @property
    def n(self):
        return self.args[3]

    # 定义属性q，返回表示波束的复杂参数
    @property
    def q(self):
        """
        The complex parameter representing the beam.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.q
        1 + 1.88679245283019*I*pi
        """
        return self.z + I*self.z_r

    # 定义属性radius，返回相位前沿的曲率半径
    @property
    def radius(self):
        """
        The radius of curvature of the phase front.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.radius
        1 + 3.55998576005696*pi**2
        """
        return self.z*(1 + (self.z_r/self.z)**2)

    # 定义属性w，返回沿波束任意位置z的波束半径w(z)
    @property
    def w(self):
        """
        The radius of the beam w(z), at any position z along the beam.
        The beam radius at `1/e^2` intensity (axial value).

        See Also
        ========

        w_0 :
            The minimal radius of beam.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.w
        0.001*sqrt(0.2809/pi**2 + 1)
        """
        return self.w_0*sqrt(1 + (self.z/self.z_r)**2)

    # 定义属性w_0，返回波束在峰值强度（1/e^2）下的最小半径
    @property
    def w_0(self):
        """
        The minimal radius of beam at `1/e^2` intensity (peak value).

        See Also
        ========

        w : the beam radius at `1/e^2` intensity (axial value).

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.w_0
        0.00100000000000000
        """
        return sqrt(self.z_r/(pi*self.n)*self.wavelen)

    # 定义属性divergence，返回总角展的一半
    @property
    def divergence(self):
        """
        Half of the total angular spread.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.divergence
        0.00053/pi
        """
        return self.wavelen/pi/self.w_0
    # 返回 Gouy 相位的计算结果，使用 arctan2 函数以避免正负角度歧义
    def gouy(self):
        """
        The Gouy phase.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.gouy
        atan(0.53/pi)
        """
        return atan2(self.z, self.z_r)

    @property
    # 返回高斯光束近似有效的最小腰直径限制
    def waist_approximation_limit(self):
        """
        The minimal waist for which the gauss beam approximation is valid.

        Explanation
        ===========

        The gauss beam is a solution to the paraxial equation. For curvatures
        that are too great it is not a valid approximation.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.waist_approximation_limit
        1.06e-6/pi
        """
        return 2*self.wavelen/pi
# Utilities
###

# 计算从高斯光束腰部到瑞利范围的函数
def waist2rayleigh(w, wavelen, n=1):
    """
    Calculate the rayleigh range from the waist of a gaussian beam.

    See Also
    ========

    rayleigh2waist, BeamParameter

    Examples
    ========

    >>> from sympy.physics.optics import waist2rayleigh
    >>> from sympy import symbols
    >>> w, wavelen = symbols('w wavelen')
    >>> waist2rayleigh(w, wavelen)
    pi*w**2/wavelen
    """
    w, wavelen = map(sympify, (w, wavelen))
    return w**2*n*pi/wavelen


# 计算从高斯光束瑞利范围到腰部的函数
def rayleigh2waist(z_r, wavelen):
    """Calculate the waist from the rayleigh range of a gaussian beam.

    See Also
    ========

    waist2rayleigh, BeamParameter

    Examples
    ========

    >>> from sympy.physics.optics import rayleigh2waist
    >>> from sympy import symbols
    >>> z_r, wavelen = symbols('z_r wavelen')
    >>> rayleigh2waist(z_r, wavelen)
    sqrt(wavelen*z_r)/sqrt(pi)
    """
    z_r, wavelen = map(sympify, (z_r, wavelen))
    return sqrt(z_r/pi*wavelen)


# 几何光学条件下的物象共轭关系函数
def geometric_conj_ab(a, b):
    """
    Conjugation relation for geometrical beams under paraxial conditions.

    Explanation
    ===========

    Takes the distances to the optical element and returns the needed
    focal distance.

    See Also
    ========

    geometric_conj_af, geometric_conj_bf

    Examples
    ========

    >>> from sympy.physics.optics import geometric_conj_ab
    >>> from sympy import symbols
    >>> a, b = symbols('a b')
    >>> geometric_conj_ab(a, b)
    a*b/(a + b)
    """
    a, b = map(sympify, (a, b))
    if a.is_infinite or b.is_infinite:
        return a if b.is_infinite else b
    else:
        return a*b/(a + b)


# 几何光学条件下的物点共轭关系函数
def geometric_conj_af(a, f):
    """
    Conjugation relation for geometrical beams under paraxial conditions.

    Explanation
    ===========

    Takes the object distance (for geometric_conj_af) or the image distance
    (for geometric_conj_bf) to the optical element and the focal distance.
    Then it returns the other distance needed for conjugation.

    See Also
    ========

    geometric_conj_ab

    Examples
    ========

    >>> from sympy.physics.optics.gaussopt import geometric_conj_af, geometric_conj_bf
    >>> from sympy import symbols
    >>> a, b, f = symbols('a b f')
    >>> geometric_conj_af(a, f)
    a*f/(a - f)
    >>> geometric_conj_bf(b, f)
    b*f/(b - f)
    """
    a, f = map(sympify, (a, f))
    return -geometric_conj_ab(a, -f)

# geometric_conj_bf 函数与 geometric_conj_af 函数相同
geometric_conj_bf = geometric_conj_af


# 高斯光束的共轭关系函数
def gaussian_conj(s_in, z_r_in, f):
    """
    Conjugation relation for gaussian beams.

    Parameters
    ==========

    s_in :
        The distance to optical element from the waist.
    z_r_in :
        The rayleigh range of the incident beam.
    f :
        The focal length of the optical element.

    Returns
    =======

    a tuple containing (s_out, z_r_out, m)
    s_out :
        The distance between the new waist and the optical element.
    z_r_out :
        The rayleigh range of the emergent beam.
    """
    # 定义函数 gaussian_conj，计算光学中的高斯物镜的共轭关系
    m :
        # 新腰和旧腰之间的比率，表示物镜传输特性的改变

    Examples
    ========

    >>> from sympy.physics.optics import gaussian_conj
    >>> from sympy import symbols
    >>> s_in, z_r_in, f = symbols('s_in z_r_in f')

    >>> gaussian_conj(s_in, z_r_in, f)[0]
    # 计算物镜传输后的新物距 s_out
    1/(-1/(s_in + z_r_in**2/(-f + s_in)) + 1/f)

    >>> gaussian_conj(s_in, z_r_in, f)[1]
    # 计算物镜传输后的新腰 z_r_out
    z_r_in/(1 - s_in**2/f**2 + z_r_in**2/f**2)

    >>> gaussian_conj(s_in, z_r_in, f)[2]
    # 计算物镜传输后的新腰和物距的比率 m
    1/sqrt(1 - s_in**2/f**2 + z_r_in**2/f**2)
    """
    # 将输入的 s_in, z_r_in, f 符号化为 sympy 对象
    s_in, z_r_in, f = map(sympify, (s_in, z_r_in, f))
    # 计算物镜传输后的新物距 s_out
    s_out = 1 / ( -1/(s_in + z_r_in**2/(s_in - f)) + 1/f )
    # 计算物镜传输后的新腰和物距的比率 m
    m = 1/sqrt((1 - (s_in/f)**2) + (z_r_in/f)**2)
    # 计算物镜传输后的新腰 z_r_out
    z_r_out = z_r_in / ((1 - (s_in/f)**2) + (z_r_in/f)**2)
    # 返回计算结果元组 (s_out, z_r_out, m)
    return (s_out, z_r_out, m)
# 定义了一个函数，用于计算高斯光束的共轭光学设置，以匹配物体/图像的腰径
def conjugate_gauss_beams(wavelen, waist_in, waist_out, **kwargs):
    """
    Find the optical setup conjugating the object/image waists.

    Parameters
    ==========

    wavelen :
        The wavelength of the beam.
    waist_in and waist_out :
        The waists to be conjugated.
    f :
        The focal distance of the element used in the conjugation.

    Returns
    =======

    a tuple containing (s_in, s_out, f)
    s_in :
        The distance before the optical element.
    s_out :
        The distance after the optical element.
    f :
        The focal distance of the optical element.

    Examples
    ========

    >>> from sympy.physics.optics import conjugate_gauss_beams
    >>> from sympy import symbols, factor
    >>> l, w_i, w_o, f = symbols('l w_i w_o f')

    >>> conjugate_gauss_beams(l, w_i, w_o, f=f)[0]
    f*(1 - sqrt(w_i**2/w_o**2 - pi**2*w_i**4/(f**2*l**2)))

    >>> factor(conjugate_gauss_beams(l, w_i, w_o, f=f)[1])
    f*w_o**2*(w_i**2/w_o**2 - sqrt(w_i**2/w_o**2 -
              pi**2*w_i**4/(f**2*l**2)))/w_i**2

    >>> conjugate_gauss_beams(l, w_i, w_o, f=f)[2]
    f
    """
    # 将输入的波长和腰径转换为符号对象
    wavelen, waist_in, waist_out = map(sympify, (wavelen, waist_in, waist_out))
    # 计算腰径比率
    m = waist_out / waist_in
    # 计算光束的瑞利范围
    z = waist2rayleigh(waist_in, wavelen)
    # 检查是否有多余的参数
    if len(kwargs) != 1:
        # 如果有多余的参数，抛出错误
        raise ValueError("The function expects only one named argument")
    elif 'dist' in kwargs:
        # 如果参数中包含'dist'，则抛出未实现的错误，并提供详细信息
        raise NotImplementedError("""Currently only focal length is supported as a parameter""")
    elif 'f' in kwargs:
        # 如果参数中包含'f'，则将其转换为符号对象
        f = sympify(kwargs['f'])
        # 计算输入面前的距离
        s_in = f * (1 - sqrt(1/m**2 - z**2/f**2))
        # 计算输出面后的距离，使用高斯光束的共轭函数
        s_out = gaussian_conj(s_in, z, f)[0]
    elif 's_in' in kwargs:
        # 如果参数中包含's_in'，则抛出未实现的错误，并提供详细信息
        raise NotImplementedError("""Currently only focal length is supported as a parameter""")
    else:
        # 如果没有匹配的参数，则抛出值错误，并提供详细信息
        raise ValueError("""The functions expects the focal length as a named argument""")
    # 返回结果的元组
    return (s_in, s_out, f)

#TODO
#def plot_beam():
#    """Plot the beam radius as it propagates in space."""
#    pass

#TODO
#def plot_beam_conjugation():
#    """
#    Plot the intersection of two beams.
#
#    Represents the conjugation relation.
#
#    See Also
#    ========
#
#    conjugate_gauss_beams
#    """
#    pass
```
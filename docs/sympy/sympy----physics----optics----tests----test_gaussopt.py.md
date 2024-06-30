# `D:\src\scipysrc\sympy\sympy\physics\optics\tests\test_gaussopt.py`

```
from sympy.core.evalf import N  # 导入 N 函数，用于数值求解
from sympy.core.numbers import (Float, I, oo, pi)  # 导入特定数学常数和符号
from sympy.core.symbol import symbols  # 导入 symbols 函数，用于创建符号变量
from sympy.functions.elementary.miscellaneous import sqrt  # 导入开平方函数 sqrt
from sympy.functions.elementary.trigonometric import atan2  # 导入反正切函数 atan2
from sympy.matrices.dense import Matrix  # 导入矩阵类 Matrix
from sympy.polys.polytools import factor  # 导入多项式因式分解函数 factor

from sympy.physics.optics import (BeamParameter, CurvedMirror,  # 导入光学相关类和函数
  CurvedRefraction, FlatMirror, FlatRefraction, FreeSpace, GeometricRay,
  RayTransferMatrix, ThinLens, conjugate_gauss_beams,
  gaussian_conj, geometric_conj_ab, geometric_conj_af, geometric_conj_bf,
  rayleigh2waist, waist2rayleigh)

def streq(a, b):
    return str(a) == str(b)  # 比较两个对象的字符串表示是否相等

def test_gauss_opt():
    mat = RayTransferMatrix(1, 2, 3, 4)  # 创建光线传输矩阵对象
    assert mat == Matrix([[1, 2], [3, 4]])  # 检查矩阵对象与指定矩阵的相等性
    assert mat == RayTransferMatrix( Matrix([[1, 2], [3, 4]]) )  # 再次检查矩阵对象与指定矩阵的相等性
    assert [mat.A, mat.B, mat.C, mat.D] == [1, 2, 3, 4]  # 检查矩阵对象的属性值是否符合预期

    d, f, h, n1, n2, R = symbols('d f h n1 n2 R')  # 创建多个符号变量
    lens = ThinLens(f)  # 创建薄透镜对象
    assert lens == Matrix([[   1, 0], [-1/f, 1]])  # 检查薄透镜对象的矩阵表示是否正确
    assert lens.C == -1/f  # 检查薄透镜对象的属性值是否符合预期
    assert FreeSpace(d) == Matrix([[ 1, d], [0, 1]])  # 检查自由空间对象的矩阵表示是否正确
    assert FlatRefraction(n1, n2) == Matrix([[1, 0], [0, n1/n2]])  # 检查平板折射对象的矩阵表示是否正确
    assert CurvedRefraction(
        R, n1, n2) == Matrix([[1, 0], [(n1 - n2)/(R*n2), n1/n2]])  # 检查曲面折射对象的矩阵表示是否正确
    assert FlatMirror() == Matrix([[1, 0], [0, 1]])  # 检查平板镜对象的矩阵表示是否正确
    assert CurvedMirror(R) == Matrix([[   1, 0], [-2/R, 1]])  # 检查曲面镜对象的矩阵表示是否正确
    assert ThinLens(f) == Matrix([[   1, 0], [-1/f, 1]])  # 再次检查薄透镜对象的矩阵表示是否正确

    mul = CurvedMirror(R)*FreeSpace(d)  # 计算曲面镜与自由空间的复合效果
    mul_mat = Matrix([[   1, 0], [-2/R, 1]])*Matrix([[ 1, d], [0, 1]])  # 计算复合效果的矩阵表示
    assert mul.A == mul_mat[0, 0]  # 检查复合效果的矩阵元素是否符合预期
    assert mul.B == mul_mat[0, 1]  # 检查复合效果的矩阵元素是否符合预期
    assert mul.C == mul_mat[1, 0]  # 检查复合效果的矩阵元素是否符合预期
    assert mul.D == mul_mat[1, 1]  # 检查复合效果的矩阵元素是否符合预期

    angle = symbols('angle')  # 创建角度符号变量
    assert GeometricRay(h, angle) == Matrix([[    h], [angle]])  # 检查几何光线对象的表示是否正确
    assert FreeSpace(
        d)*GeometricRay(h, angle) == Matrix([[angle*d + h], [angle]])  # 检查自由空间与几何光线的复合效果是否正确
    assert GeometricRay( Matrix( ((h,), (angle,)) ) ) == Matrix([[h], [angle]])  # 再次检查几何光线对象的表示是否正确
    assert (FreeSpace(d)*GeometricRay(h, angle)).height == angle*d + h  # 检查复合效果后几何光线对象的高度属性是否正确
    assert (FreeSpace(d)*GeometricRay(h, angle)).angle == angle  # 检查复合效果后几何光线对象的角度属性是否正确

    p = BeamParameter(530e-9, 1, w=1e-3)  # 创建光束参数对象
    assert streq(p.q, 1 + 1.88679245283019*I*pi)  # 检查光束参数对象的属性是否符合预期
    assert streq(N(p.q), 1.0 + 5.92753330865999*I)  # 检查光束参数对象经过数值化后的属性是否符合预期
    assert streq(N(p.w_0), Float(0.00100000000000000))  # 检查光束参数对象的束腰半径数值化后的属性是否符合预期
    assert streq(N(p.z_r), Float(5.92753330865999))  # 检查光束参数对象的瑞利长度数值化后的属性是否符合预期
    fs = FreeSpace(10)  # 创建自由空间对象
    p1 = fs*p  # 在自由空间中传播光束参数
    assert streq(N(p.w), Float(0.00101413072159615))  # 检查传播后光束参数对象的束腰半径数值化后的属性是否符合预期
    assert streq(N(p1.w), Float(0.00210803120913829))  # 检查传播后光束参数对象的束腰半径数值化后的属性是否符合预期

    w, wavelen = symbols('w wavelen')  # 创建符号变量
    assert waist2rayleigh(w, wavelen) == pi*w**2/wavelen  # 检查束腰半径到瑞利长度的转换关系是否正确
    z_r, wavelen = symbols('z_r wavelen')  # 再次创建符号变量
    assert rayleigh2waist(z_r, wavelen) == sqrt(wavelen*z_r)/sqrt(pi)  # 检查瑞利长度到束腰半径的转换关系是否正确

    a, b, f = symbols('a b f')  # 创建符号变量
    assert geometric_conj_ab(a, b) == a*b/(a + b)  # 检查几何共轭关系计算函数的正确性
    assert geometric_conj_af(a, f) == a*f/(a - f)  # 检查几何共轭关系计算函数的正确性
    assert geometric_conj_bf(b, f) == b*f/(b - f)  # 检查几何共轭关系计算函数的正确性
    assert geometric_conj_ab(oo, b) == b  # 检查几何共轭关系计算函数在
    # 断言：验证高斯光束的共轭关系第一个表达式是否正确
    assert gaussian_conj(s_in, z_r_in, f)[0] == 1/(-1/(s_in + z_r_in**2/(-f + s_in)) + 1/f)
    # 断言：验证高斯光束的共轭关系第二个表达式是否正确
    assert gaussian_conj(s_in, z_r_in, f)[1] == z_r_in/(1 - s_in**2/f**2 + z_r_in**2/f**2)
    # 断言：验证高斯光束的共轭关系第三个表达式是否正确
    assert gaussian_conj(s_in, z_r_in, f)[2] == 1/sqrt(1 - s_in**2/f**2 + z_r_in**2/f**2)

    # 定义符号变量：光程l，输入和输出束腰半径w_i、w_o，焦距f
    l, w_i, w_o, f = symbols('l w_i w_o f')
    # 断言：验证共轭高斯光束函数的第一个表达式是否正确
    assert conjugate_gauss_beams(l, w_i, w_o, f=f)[0] == f*(-sqrt(w_i**2/w_o**2 - pi**2*w_i**4/(f**2*l**2)) + 1)
    # 断言：验证共轭高斯光束函数的第二个表达式是否正确
    assert factor(conjugate_gauss_beams(l, w_i, w_o, f=f)[1]) == f*w_o**2*(w_i**2/w_o**2 - sqrt(w_i**2/w_o**2 - pi**2*w_i**4/(f**2*l**2)))/w_i**2
    # 断言：验证共轭高斯光束函数的第三个表达式是否正确
    assert conjugate_gauss_beams(l, w_i, w_o, f=f)[2] == f

    # 定义符号变量：光程长度l，光束半径w_0，以及光束参数对象p
    z, l, w_0 = symbols('z l w_0', positive=True)
    p = BeamParameter(l, z, w=w_0)
    # 断言：验证光束参数对象p的光斑半径计算是否正确
    assert p.radius == z*(pi**2*w_0**4/(l**2*z**2) + 1)
    # 断言：验证光束参数对象p的光束半径w计算是否正确
    assert p.w == w_0*sqrt(l**2*z**2/(pi**2*w_0**4) + 1)
    # 断言：验证光束参数对象p的光束初始半径w_0是否正确
    assert p.w_0 == w_0
    # 断言：验证光束参数对象p的发散角计算是否正确
    assert p.divergence == l/(pi*w_0)
    # 断言：验证光束参数对象p的Gouy相位偏移角计算是否正确
    assert p.gouy == atan2(z, pi*w_0**2/l)
    # 断言：验证光束参数对象p的束腰近似极限是否正确
    assert p.waist_approximation_limit == 2*l/pi

    # 创建光束参数对象p，使用指定的波长530纳米，光程1米，初始光束半径1毫米，折射率2
    p = BeamParameter(530e-9, 1, w=1e-3, n=2)
    # 断言：验证光束参数对象p的复数光束参数q是否与预期相符
    assert streq(p.q, 1 + 3.77358490566038*I*pi)
    # 断言：验证光束参数对象p的瑞利范围计算结果是否与预期接近
    assert streq(N(p.z_r), Float(11.8550666173200))
    # 断言：验证光束参数对象p的初始光束半径w_0计算结果是否与预期接近
    assert streq(N(p.w_0), Float(0.00100000000000000))
```
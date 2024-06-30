# `D:\src\scipysrc\sympy\sympy\physics\optics\polarization.py`

```
    """A Jones vector corresponding to a polarization ellipse with `psi` tilt,
    and `chi` circularity.

    Parameters
    ----------
    psi : Symbol
        Tilt angle of the polarization ellipse.
    chi : Symbol
        Circularity parameter of the polarization ellipse.

    Returns
    -------
    Matrix
        Jones vector representing the polarization state.
    """
    # 定义一个函数，生成一个 Jones 向量，描述了偏振椭圆相对于 x 轴的倾斜角度 psi 和其主轴邻角 chi。
    
    psi : numeric type or SymPy Symbol
        偏振椭圆相对于 x 轴的倾斜角度。
    
    chi : numeric type or SymPy Symbol
        偏振椭圆主轴的邻角。
    
    Returns
    =======
    Matrix :
        返回一个 Jones 向量，描述了偏振状态。
    
    Examples
    ========
    
    The axes on the Poincaré sphere.
    
    >>> from sympy import pprint, symbols, pi
    >>> from sympy.physics.optics.polarization import jones_vector
    >>> psi, chi = symbols("psi, chi", real=True)
    
    A general Jones vector.
    
    >>> pprint(jones_vector(psi, chi), use_unicode=True)
    ⎡-ⅈ⋅sin(χ)⋅sin(ψ) + cos(χ)⋅cos(ψ)⎤
    ⎢                                ⎥
    ⎣ⅈ⋅sin(χ)⋅cos(ψ) + sin(ψ)⋅cos(χ) ⎦
    
    Horizontal polarization.
    
    >>> pprint(jones_vector(0, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎣0⎦
    
    Vertical polarization.
    
    >>> pprint(jones_vector(pi/2, 0), use_unicode=True)
    ⎡0⎤
    ⎢ ⎥
    ⎣1⎦
    
    Diagonal polarization.
    
    >>> pprint(jones_vector(pi/4, 0), use_unicode=True)
    ⎡√2⎤
    ⎢──⎥
    ⎢2 ⎥
    ⎢  ⎥
    ⎢√2⎥
    ⎢──⎥
    ⎣2 ⎦
    
    Anti-diagonal polarization.
    
    >>> pprint(jones_vector(-pi/4, 0), use_unicode=True)
    ⎡ √2 ⎤
    ⎢ ── ⎥
    ⎢ 2  ⎥
    ⎢    ⎥
    ⎢-√2 ⎥
    ⎢────⎥
    ⎣ 2  ⎦
    
    Right-hand circular polarization.
    
    >>> pprint(jones_vector(0, pi/4), use_unicode=True)
    ⎡ √2 ⎤
    ⎢ ── ⎥
    ⎢ 2  ⎥
    ⎢    ⎥
    ⎢√2⋅ⅈ⎥
    ⎢────⎥
    ⎣ 2  ⎦
    
    Left-hand circular polarization.
    
    >>> pprint(jones_vector(0, -pi/4), use_unicode=True)
    ⎡  √2  ⎤
    ⎢  ──  ⎥
    ⎢  2   ⎥
    ⎢      ⎥
    ⎢-√2⋅ⅈ ⎥
    ⎢──────⎥
    ⎣  2   ⎦
# 定义计算 Stokes 向量的函数，用于描述偏振椭圆的性质，包括倾斜角 psi 和圆性角 chi。

def stokes_vector(psi, chi, p=1, I=1):
    """A Stokes vector corresponding to a polarization ellipse with ``psi``
    tilt, and ``chi`` circularity.

    Parameters
    ==========

    psi : numeric type or SymPy Symbol
        The tilt of the polarization relative to the ``x`` axis.
    chi : numeric type or SymPy Symbol
        The angle adjacent to the mayor axis of the polarization ellipse.
    p : numeric type or SymPy Symbol
        The degree of polarization.
    I : numeric type or SymPy Symbol
        The intensity of the field.


    Returns
    =======

    Matrix :
        A Stokes vector.

    Examples
    ========

    The axes on the Poincaré sphere.

    >>> from sympy import pprint, symbols, pi
    >>> from sympy.physics.optics.polarization import stokes_vector
    >>> psi, chi, p, I = symbols("psi, chi, p, I", real=True)
    >>> pprint(stokes_vector(psi, chi, p, I), use_unicode=True)
    ⎡          I          ⎤
    ⎢                     ⎥
    ⎢I⋅p⋅cos(2⋅χ)⋅cos(2⋅ψ)⎥
    ⎢                     ⎥
    ⎢I⋅p⋅sin(2⋅ψ)⋅cos(2⋅χ)⎥
    ⎢                     ⎥
    ⎣    I⋅p⋅sin(2⋅χ)     ⎦


    Horizontal polarization

    >>> pprint(stokes_vector(0, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢1⎥
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎣0⎦

    Vertical polarization

    >>> pprint(stokes_vector(pi/2, 0), use_unicode=True)
    ⎡1 ⎤
    ⎢  ⎥
    ⎢-1⎥
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎣0 ⎦

    Diagonal polarization

    >>> pprint(stokes_vector(pi/4, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎢1⎥
    ⎢ ⎥
    ⎣0⎦

    Anti-diagonal polarization

    >>> pprint(stokes_vector(-pi/4, 0), use_unicode=True)
    ⎡1 ⎤
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎢-1⎥
    ⎢  ⎥
    ⎣0 ⎦

    Right-hand circular polarization

    >>> pprint(stokes_vector(0, pi/4), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎣1⎦

    Left-hand circular polarization

    >>> pprint(stokes_vector(0, -pi/4), use_unicode=True)
    ⎡1 ⎤
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎣-1⎦

    Unpolarized light

    >>> pprint(stokes_vector(0, 0, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎣0⎦

    """
    # 计算 Stokes 向量的各个分量
    S0 = I
    S1 = I*p*cos(2*psi)*cos(2*chi)
    S2 = I*p*sin(2*psi)*cos(2*chi)
    S3 = I*p*sin(2*chi)
    # 返回由这些分量构成的 Stokes 向量
    return Matrix([S0, S1, S2, S3])


def jones_2_stokes(e):
    """Return the Stokes vector for a Jones vector ``e``.

    Parameters
    ==========

    e : SymPy Matrix
        A Jones vector.

    Returns
    =======

    SymPy Matrix
        A Jones vector.

    Examples
    ========

    The axes on the Poincaré sphere.

    >>> from sympy import pprint, pi
    >>> from sympy.physics.optics.polarization import jones_vector
    >>> from sympy.physics.optics.polarization import jones_2_stokes
    >>> H = jones_vector(0, 0)
    >>> V = jones_vector(pi/2, 0)
    >>> D = jones_vector(pi/4, 0)
    >>> A = jones_vector(-pi/4, 0)
    >>> R = jones_vector(0, pi/4)
    >>> L = jones_vector(0, -pi/4)

    """
    # 未实现 jones_2_stokes 函数，暂时没有需要添加的代码
    pass
    # 使用 pprint 函数打印列表推导式的结果，列表中包含调用 jones_2_stokes 函数的结果
    # 对于给定的列表 [H, V, D, A, R, L]
    # 使用 use_unicode=True 参数以支持 Unicode 字符输出

    """
    # 函数 jones_2_stokes 接受一个包含两个复数的元组 e 作为参数
    ex, ey = e
    # 返回一个包含四个元素的 Matrix 对象，表示 Stokes 参数
    return Matrix([Abs(ex)**2 + Abs(ey)**2,
                   Abs(ex)**2 - Abs(ey)**2,
                   2*re(ex*ey.conjugate()),
                   -2*im(ex*ey.conjugate())])
    ```
# 定义一个线偏振器的 Jones 矩阵，其传输轴的角度为 theta
def linear_polarizer(theta=0):
    # 创建一个 2x2 的 SymPy 矩阵，表示线偏振器的 Jones 矩阵
    M = Matrix([[cos(theta)**2, sin(theta)*cos(theta)],
                [sin(theta)*cos(theta), sin(theta)**2]])
    # 返回该 Jones 矩阵
    return M


# 定义一个相位补偿器的 Jones 矩阵，其在角度 theta 处有 delta 的相位差
def phase_retarder(theta=0, delta=0):
    # 创建一个 2x2 的 SymPy 矩阵，表示相位补偿器的 Jones 矩阵
    R = Matrix([[cos(theta)**2 + exp(I*delta)*sin(theta)**2,
                (1-exp(I*delta))*cos(theta)*sin(theta)],
                [(1-exp(I*delta))*cos(theta)*sin(theta),
                sin(theta)**2 + exp(I*delta)*cos(theta)**2]])
    # 返回该 Jones 矩阵乘以 exp(-I*delta/2)
    return R*exp(-I*delta/2)


# 定义一个半波相位补偿器的 Jones 矩阵，其在角度 theta 处
def half_wave_retarder(theta):
    # 创建一个 SymPy 2x2 矩阵，表示半波相位补偿器的 Jones 矩阵
    # 返回该 Jones 矩阵
    return NotImplemented  # 由于该函数未实现，返回未实现提示
    Examples
    ========

    A generic half-wave plate.

    # 导入所需的库和函数
    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import half_wave_retarder
    # 定义实数符号 theta
    >>> theta = symbols("theta", real=True)
    # 创建半波片对象 HWP，传入角度参数 theta
    >>> HWP = half_wave_retarder(theta)
    # 使用 Unicode 打印 HWP 的结果
    >>> pprint(HWP, use_unicode=True)
    ⎡   ⎛     2         2   ⎞                        ⎤
    ⎢-ⅈ⋅⎝- sin (θ) + cos (θ)⎠    -2⋅ⅈ⋅sin(θ)⋅cos(θ)  ⎥
    ⎢                                                ⎥
    ⎢                             ⎛   2         2   ⎞⎥
    ⎣   -2⋅ⅈ⋅sin(θ)⋅cos(θ)     -ⅈ⋅⎝sin (θ) - cos (θ)⎠⎦

    # 返回一个相位延迟器，参数为 theta 和 pi
    """
    return phase_retarder(theta, pi)
    ```
# 定义一个四分之一波片的琼斯矩阵，其快轴角度为 `theta`
def quarter_wave_retarder(theta):
    # 调用 `phase_retarder` 函数，返回一个角度为 π/2 的相位延迟器的琼斯矩阵
    return phase_retarder(theta, pi/2)


# 定义一个透射滤波器的琼斯矩阵，其透射率为 `T`
def transmissive_filter(T):
    # 返回一个表示滤波器的琼斯矩阵，其元素为 [√T, 0; 0, √T]
    return Matrix([[sqrt(T), 0], [0, sqrt(T)]])


# 定义一个反射滤波器的琼斯矩阵，其反射率为 `R`
def reflective_filter(R):
    # 返回一个表示反射滤波器的琼斯矩阵，其元素为 [√R, 0; 0, -√R]
    return Matrix([[sqrt(R), 0], [0, -sqrt(R)]])


# 计算给定琼斯矩阵 `J` 对应的穆勒矩阵
def mueller_matrix(J):
    # 返回与给定琼斯矩阵 `J` 相对应的穆勒矩阵
    """
    定义一个 4x4 的复数矩阵 A
    A = Matrix([[1, 0, 0, 1],
                [1, 0, 0, -1],
                [0, 1, 1, 0],
                [0, -I, I, 0]])
    
    返回简化后的表达式，这是通过 A 和 J 的张量积乘以 A 的逆矩阵得到的结果
    """
    A = Matrix([[1, 0, 0, 1],
                [1, 0, 0, -1],
                [0, 1, 1, 0],
                [0, -I, I, 0]])
    
    return simplify(A*TensorProduct(J, J.conjugate())*A.inv())
# 定义一个函数，返回指定角度下的偏振分束器的琼斯矩阵

def polarizing_beam_splitter(Tp=1, Rs=1, Ts=0, Rp=0, phia=0, phib=0):
    r"""A polarizing beam splitter Jones matrix at angle `theta`.

    Parameters
    ==========

    J : SymPy Matrix
        A Jones matrix.
    Tp : numeric type or SymPy Symbol
        The transmissivity of the P-polarized component.
    Rs : numeric type or SymPy Symbol
        The reflectivity of the S-polarized component.
    Ts : numeric type or SymPy Symbol
        The transmissivity of the S-polarized component.
    Rp : numeric type or SymPy Symbol
        The reflectivity of the P-polarized component.
    phia : numeric type or SymPy Symbol
        The phase difference between transmitted and reflected component for
        output mode a.
    phib : numeric type or SymPy Symbol
        The phase difference between transmitted and reflected component for
        output mode b.


    Returns
    =======

    SymPy Matrix
        A 4x4 matrix representing the PBS. This matrix acts on a 4x1 vector
        whose first two entries are the Jones vector on one of the PBS ports,
        and the last two entries the Jones vector on the other port.

    Examples
    ========

    Generic polarizing beam-splitter.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import polarizing_beam_splitter
    >>> Ts, Rs, Tp, Rp = symbols(r"Ts, Rs, Tp, Rp", positive=True)
    >>> phia, phib = symbols("phi_a, phi_b", real=True)
    >>> PBS = polarizing_beam_splitter(Tp, Rs, Ts, Rp, phia, phib)
    >>> pprint(PBS, use_unicode=False)
    [   ____                           ____                    ]
    [ \/ Tp            0           I*\/ Rp           0         ]
    [                                                          ]
    [                  ____                       ____  I*phi_a]
    [   0            \/ Ts            0      -I*\/ Rs *e       ]
    [                                                          ]
    [    ____                         ____                     ]
    [I*\/ Rp           0            \/ Tp            0         ]
    [                                                          ]
    [               ____  I*phi_b                    ____      ]
    [   0      -I*\/ Rs *e            0            \/ Ts       ]

    """
    # 构建偏振分束器的琼斯矩阵，用于描述光通过分束器后的行为
    PBS = Matrix([[sqrt(Tp), 0, I*sqrt(Rp), 0],
                  [0, sqrt(Ts), 0, -I*sqrt(Rs)*exp(I*phia)],
                  [I*sqrt(Rp), 0, sqrt(Tp), 0],
                  [0, -I*sqrt(Rs)*exp(I*phib), 0, sqrt(Ts)]])
    # 返回构建好的琼斯矩阵
    return PBS
```
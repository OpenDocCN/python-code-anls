# `D:\src\scipysrc\sympy\sympy\physics\optics\tests\test_polarization.py`

```
# 从 sympy.physics.optics.polarization 模块导入所需函数和类
# 从 sympy.core.numbers 模块导入 I (虚数单位), pi (π)
# 从 sympy.core.singleton 模块导入 S (符号 S)
# 从 sympy.core.symbol 模块导入 symbols (符号变量)
# 从 sympy.functions.elementary.exponential 模块导入 exp (指数函数)
# 从 sympy.matrices.dense 模块导入 Matrix (矩阵类)

def test_polarization():
    # 断言：检验当传入参数为 0 和 0 时，jones_vector 函数的返回结果
    assert jones_vector(0, 0) == Matrix([1, 0])
    # 断言：检验当传入参数为 π/2 和 0 时，jones_vector 函数的返回结果
    assert jones_vector(pi/2, 0) == Matrix([0, 1])
    #################################################################
    # 断言：检验当传入参数为 0 和 0 时，stokes_vector 函数的返回结果
    assert stokes_vector(0, 0) == Matrix([1, 1, 0, 0])
    # 断言：检验当传入参数为 π/2 和 0 时，stokes_vector 函数的返回结果
    assert stokes_vector(pi/2, 0) == Matrix([1, -1, 0, 0])
    #################################################################
    # 创建水平分量和垂直分量的 Jones 矢量
    H = jones_vector(0, 0)
    V = jones_vector(pi/2, 0)
    # 创建对角线分量和反对角线分量的 Jones 矢量
    D = jones_vector(pi/4, 0)
    A = jones_vector(-pi/4, 0)
    # 创建右旋和左旋的 Jones 矢量
    R = jones_vector(0, pi/4)
    L = jones_vector(0, -pi/4)

    # 期望的 Stokes 矢量结果列表
    res = [Matrix([1, 1, 0, 0]),
           Matrix([1, -1, 0, 0]),
           Matrix([1, 0, 1, 0]),
           Matrix([1, 0, -1, 0]),
           Matrix([1, 0, 0, 1]),
           Matrix([1, 0, 0, -1])]

    # 断言：检验 jones_2_stokes 函数对于列表中每个 Jones 矢量的处理结果是否符合期望
    assert [jones_2_stokes(e) for e in [H, V, D, A, R, L]] == res
    #################################################################
    # 断言：检验传入参数为 0 时，linear_polarizer 函数的返回结果
    assert linear_polarizer(0) == Matrix([[1, 0], [0, 0]])
    #################################################################
    # 定义一个实数符号变量 delta
    delta = symbols("delta", real=True)
    # 期望的相位延迟器矩阵结果
    res = Matrix([[exp(-I*delta/2), 0], [0, exp(I*delta/2)]])
    # 断言：检验 phase_retarder 函数对于传入参数为 0 和 delta 时的返回结果是否符合期望
    assert phase_retarder(0, delta) == res
    #################################################################
    # 断言：检验传入参数为 0 时，half_wave_retarder 函数的返回结果
    assert half_wave_retarder(0) == Matrix([[-I, 0], [0, I]])
    #################################################################
    # 期望的四分之一波片矩阵结果
    res = Matrix([[exp(-I*pi/4), 0], [0, I*exp(-I*pi/4)]])
    # 断言：检验 quarter_wave_retarder 函数对于传入参数为 0 时的返回结果是否符合期望
    assert quarter_wave_retarder(0) == res
    #################################################################
    # 断言：检验传入参数为 1 时，transmissive_filter 函数的返回结果
    assert transmissive_filter(1) == Matrix([[1, 0], [0, 1]])
    #################################################################
    # 断言：检验传入参数为 1 时，reflective_filter 函数的返回结果
    assert reflective_filter(1) == Matrix([[1, 0], [0, -1]])

    # 期望的 Mueller 矩阵结果
    res = Matrix([[S(1)/2, S(1)/2, 0, 0],
                  [S(1)/2, S(1)/2, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    # 断言：检验 mueller_matrix 函数对于传入参数为 linear_polarizer(0) 时的返回结果是否符合期望
    assert mueller_matrix(linear_polarizer(0)) == res
    #################################################################
    # 期望的偏振分束器矩阵结果
    res = Matrix([[1, 0, 0, 0], [0, 0, 0, -I], [0, 0, 1, 0], [0, -I, 0, 0]])
    # 断言：检验 polarizing_beam_splitter 函数的返回结果是否符合期望
    assert polarizing_beam_splitter() == res
```
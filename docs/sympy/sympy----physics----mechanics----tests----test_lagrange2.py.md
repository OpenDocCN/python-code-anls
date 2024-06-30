# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_lagrange2.py`

```
### 导入所需的符号和力学模块
from sympy import symbols
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.mechanics import ReferenceFrame, Point, Particle
from sympy.physics.mechanics import LagrangesMethod, Lagrangian

### 这个测试用例验证带有两个外力的系统通过Lagrange方法正确形成（参见问题 #8626）
def test_lagrange_2forces():
    ### 两个阻尼弹簧系统的广义坐标
    q1, q2 = dynamicsymbols('q1, q2')
    ### 两个阻尼弹簧系统的广义速度
    q1d, q2d = dynamicsymbols('q1, q2', 1)

    ### 质量、弹簧系数、摩擦系数
    m, k, nu = symbols('m, k, nu')

    ### 定义惯性参考系和参考点
    N = ReferenceFrame('N')
    O = Point('O')

    ### 定义两个质点
    P1 = O.locatenew('P1', q1 * N.x)
    P1.set_vel(N, q1d * N.x)
    P2 = O.locatenew('P1', q2 * N.x)  # 注：这里可能是代码中的错误，应为 'P2'
    P2.set_vel(N, q2d * N.x)

    ### 定义两个粒子
    pP1 = Particle('pP1', P1, m)
    pP1.potential_energy = k * q1**2 / 2

    pP2 = Particle('pP2', P2, m)
    pP2.potential_energy = k * (q1 - q2)**2 / 2

    #### 定义摩擦力
    forcelist = [(P1, - nu * q1d * N.x),
                 (P2, - nu * q2d * N.x)]
    ### 定义Lagrangian，并传入质点和拉格朗日框架
    lag = Lagrangian(N, pP1, pP2)

    ### 创建LagrangesMethod实例，传入拉格朗日量、广义坐标、外力列表和惯性参考系
    l_method = LagrangesMethod(lag, (q1, q2), forcelist=forcelist, frame=N)

    ### 形成Lagrange方程
    l_method.form_lagranges_equations()

    ### 验证第一个方程的正确性
    eq1 = l_method.eom[0]
    assert eq1.diff(q1d) == nu

    ### 验证第二个方程的正确性
    eq2 = l_method.eom[1]
    assert eq2.diff(q2d) == nu
```
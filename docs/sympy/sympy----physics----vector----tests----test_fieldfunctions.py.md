# `D:\src\scipysrc\sympy\sympy\physics\vector\tests\test_fieldfunctions.py`

```
from sympy.core.singleton import S  # 导入符号 S（单例）
from sympy.core.symbol import Symbol  # 导入符号变量 Symbol
from sympy.functions.elementary.trigonometric import (cos, sin)  # 导入三角函数 cos 和 sin
from sympy.physics.vector import ReferenceFrame, Vector, Point, \
     dynamicsymbols  # 导入参考系 ReferenceFrame、向量 Vector、点 Point 和 动态符号 dynamicsymbols
from sympy.physics.vector.fieldfunctions import divergence, \
     gradient, curl, is_conservative, is_solenoidal, \
     scalar_potential, scalar_potential_difference  # 导入场函数 divergence、gradient、curl、is_conservative、is_solenoidal、scalar_potential、scalar_potential_difference
from sympy.testing.pytest import raises  # 导入 pytest 的 raises 函数

R = ReferenceFrame('R')  # 创建一个名为 R 的参考系
q = dynamicsymbols('q')  # 创建一个动态符号 q
P = R.orientnew('P', 'Axis', [q, R.z])  # 根据旋转轴创建一个新的参考系 P

def test_curl():
    assert curl(Vector(0), R) == Vector(0)  # 测试零向量的旋度应为零向量
    assert curl(R.x, R) == Vector(0)  # 测试单位向量 R.x 的旋度应为零向量
    assert curl(2*R[1]**2*R.y, R) == Vector(0)  # 测试向量 2*R[1]**2*R.y 的旋度应为零向量
    assert curl(R[0]*R[1]*R.z, R) == R[0]*R.x - R[1]*R.y  # 测试向量 R[0]*R[1]*R.z 的旋度应为 R[0]*R.x - R[1]*R.y
    assert curl(R[0]*R[1]*R[2] * (R.x+R.y+R.z), R) == \
           (-R[0]*R[1] + R[0]*R[2])*R.x + (R[0]*R[1] - R[1]*R[2])*R.y + \
           (-R[0]*R[2] + R[1]*R[2])*R.z  # 测试复杂向量的旋度计算是否正确
    assert curl(2*R[0]**2*R.y, R) == 4*R[0]*R.z  # 测试向量 2*R[0]**2*R.y 的旋度计算是否正确
    assert curl(P[0]**2*R.x + P.y, R) == \
           - 2*(R[0]*cos(q) + R[1]*sin(q))*sin(q)*R.z  # 测试复杂表达式 P[0]**2*R.x + P.y 的旋度计算是否正确
    assert curl(P[0]*R.y, P) == cos(q)*P.z  # 测试在参考系 P 中向量 P[0]*R.y 的旋度计算是否正确


def test_divergence():
    assert divergence(Vector(0), R) is S.Zero  # 测试零向量的散度应为符号 0
    assert divergence(R.x, R) is S.Zero  # 测试单位向量 R.x 的散度应为符号 0
    assert divergence(R[0]**2*R.x, R) == 2*R[0]  # 测试向量 R[0]**2*R.x 的散度计算是否正确
    assert divergence(R[0]*R[1]*R[2] * (R.x+R.y+R.z), R) == \
           R[0]*R[1] + R[0]*R[2] + R[1]*R[2]  # 测试复杂向量的散度计算是否正确
    assert divergence((1/(R[0]*R[1]*R[2])) * (R.x+R.y+R.z), R) == \
           -1/(R[0]*R[1]*R[2]**2) - 1/(R[0]*R[1]**2*R[2]) - \
           1/(R[0]**2*R[1]*R[2])  # 测试复杂表达式的散度计算是否正确
    v = P[0]*P.x + P[1]*P.y + P[2]*P.z
    assert divergence(v, P) == 3  # 测试在参考系 P 中向量 v 的散度计算是否正确
    assert divergence(v, R).simplify() == 3  # 测试向量 v 在参考系 R 中简化后的散度是否为 3
    assert divergence(P[0]*R.x + R[0]*P.x, R) == 2*cos(q)  # 测试复杂表达式的散度计算是否正确


def test_gradient():
    a = Symbol('a')  # 创建一个符号变量 a
    assert gradient(0, R) == Vector(0)  # 测试标量场 0 的梯度应为零向量
    assert gradient(R[0], R) == R.x  # 测试标量场 R[0] 的梯度计算是否正确
    assert gradient(R[0]*R[1]*R[2], R) == \
           R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z  # 测试复杂标量场的梯度计算是否正确
    assert gradient(2*R[0]**2, R) == 4*R[0]*R.x  # 测试标量场 2*R[0]**2 的梯度计算是否正确
    assert gradient(a*sin(R[1])/R[0], R) == \
           - a*sin(R[1])/R[0]**2*R.x + a*cos(R[1])/R[0]*R.y  # 测试复杂表达式的梯度计算是否正确
    assert gradient(P[0]*P[1], R) == \
           ((-R[0]*sin(q) + R[1]*cos(q))*cos(q) - (R[0]*cos(q) + R[1]*sin(q))*sin(q))*R.x + \
           ((-R[0]*sin(q) + R[1]*cos(q))*sin(q) + (R[0]*cos(q) + R[1]*sin(q))*cos(q))*R.y  # 测试复杂表达式的梯度计算是否正确
    assert gradient(P[0]*R[2], P) == P[2]*P.x + P[0]*P.z  # 测试在参考系 P 中复杂表达式的梯度计算是否正确


scalar_field = 2*R[0]**2*R[1]*R[2]  # 定义一个标量场
grad_field = gradient(scalar_field, R)  # 计算标量场的梯度场
vector_field = R[1]**2*R.x + 3*R[0]*R.y + 5*R[1]*R[2]*R.z  # 定义一个向量场
curl_field = curl(vector_field, R)  # 计算向量场的旋度场


def test_conservative():
    assert is_conservative(0) is True  # 测试标量 0 是否是保守场
    assert is_conservative(R.x) is True  # 测试单位向量 R.x 是否是保守场
    assert is_conservative(2 * R.x + 3 * R.y + 4 * R.z) is True  # 测试复杂向量是否是保守场
    assert is_conservative(R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z) is \
           True  # 测试复杂向量是否是保守场
    assert is_conservative(R[0] * R.y) is False  # 测试复杂向量是否是保守场
    assert is_conservative(grad_field) is True  # 测试梯度场是否是保守场
    assert is_conservative(curl_field) is False  # 测试旋度场是否是保守场
    # 断言：检查表达式 is_conservative(4*R[0]*R[1]*R[2]*R.x + 2*R[0]**2*R[2]*R.y) 的返回值是否为 False
    assert is_conservative(4*R[0]*R[1]*R[2]*R.x + 2*R[0]**2*R[2]*R.y) is False
    
    # 断言：检查表达式 is_conservative(R[2]*P.x + P[0]*R.z) 的返回值是否为 True
    assert is_conservative(R[2]*P.x + P[0]*R.z) is True
# 定义测试函数 test_solenoidal，用于测试 is_solenoidal 函数的各种输入情况
def test_solenoidal():
    # 断言 is_solenoidal(0) 返回 True
    assert is_solenoidal(0) is True
    # 断言 is_solenoidal(R.x) 返回 True
    assert is_solenoidal(R.x) is True
    # 断言 is_solenoidal(2 * R.x + 3 * R.y + 4 * R.z) 返回 True
    assert is_solenoidal(2 * R.x + 3 * R.y + 4 * R.z) is True
    # 断言 is_solenoidal(R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z) 返回 True
    assert is_solenoidal(R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z) is True
    # 断言 is_solenoidal(R[1] * R.y) 返回 False
    assert is_solenoidal(R[1] * R.y) is False
    # 断言 is_solenoidal(grad_field) 返回 False
    assert is_solenoidal(grad_field) is False
    # 断言 is_solenoidal(curl_field) 返回 True
    assert is_solenoidal(curl_field) is True
    # 断言 is_solenoidal((-2*R[1] + 3)*R.z) 返回 True
    assert is_solenoidal((-2*R[1] + 3)*R.z) is True
    # 断言 is_solenoidal(cos(q)*R.x + sin(q)*R.y + cos(q)*P.z) 返回 True
    assert is_solenoidal(cos(q)*R.x + sin(q)*R.y + cos(q)*P.z) is True
    # 断言 is_solenoidal(R[2]*P.x + P[0]*R.z) 返回 True
    assert is_solenoidal(R[2]*P.x + P[0]*R.z) is True

# 定义测试函数 test_scalar_potential，用于测试 scalar_potential 函数的各种输入情况
def test_scalar_potential():
    # 断言 scalar_potential(0, R) 返回 0
    assert scalar_potential(0, R) == 0
    # 断言 scalar_potential(R.x, R) 返回 R[0]
    assert scalar_potential(R.x, R) == R[0]
    # 断言 scalar_potential(R.y, R) 返回 R[1]
    assert scalar_potential(R.y, R) == R[1]
    # 断言 scalar_potential(R.z, R) 返回 R[2]
    assert scalar_potential(R.z, R) == R[2]
    # 断言 scalar_potential(R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z, R) 返回 R[0]*R[1]*R[2]
    assert scalar_potential(R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z, R) == R[0]*R[1]*R[2]
    # 断言 scalar_potential(grad_field, R) 返回 scalar_field
    assert scalar_potential(grad_field, R) == scalar_field
    # 断言 scalar_potential(R[2]*P.x + P[0]*R.z, R) 返回 R[0]*R[2]*cos(q) + R[1]*R[2]*sin(q)
    assert scalar_potential(R[2]*P.x + P[0]*R.z, R) == R[0]*R[2]*cos(q) + R[1]*R[2]*sin(q)
    # 断言 scalar_potential(R[2]*P.x + P[0]*R.z, P) 引发 ValueError
    raises(ValueError, lambda: scalar_potential(R[0] * R.y, R))

# 定义测试函数 test_scalar_potential_difference，用于测试 scalar_potential_difference 函数的各种输入情况
def test_scalar_potential_difference():
    # 创建原点 Point('O') 和其他点 point1, point2, genericpointR, genericpointP
    origin = Point('O')
    point1 = origin.locatenew('P1', 1*R.x + 2*R.y + 3*R.z)
    point2 = origin.locatenew('P2', 4*R.x + 5*R.y + 6*R.z)
    genericpointR = origin.locatenew('RP', R[0]*R.x + R[1]*R.y + R[2]*R.z)
    genericpointP = origin.locatenew('PP', P[0]*P.x + P[1]*P.y + P[2]*P.z)
    # 断言 scalar_potential_difference(S.Zero, R, point1, point2, origin) 返回 0
    assert scalar_potential_difference(S.Zero, R, point1, point2, origin) == 0
    # 断言 scalar_potential_difference(scalar_field, R, origin, genericpointR, origin) 返回 scalar_field
    assert scalar_potential_difference(scalar_field, R, origin, genericpointR, origin) == scalar_field
    # 断言 scalar_potential_difference(grad_field, R, origin, genericpointR, origin) 返回 scalar_field
    assert scalar_potential_difference(grad_field, R, origin, genericpointR, origin) == scalar_field
    # 断言 scalar_potential_difference(grad_field, R, point1, point2, origin) 返回 948
    assert scalar_potential_difference(grad_field, R, point1, point2, origin) == 948
    # 断言 scalar_potential_difference(R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z, R, point1, genericpointR, origin) 返回 R[0]*R[1]*R[2] - 6
    assert scalar_potential_difference(R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z, R, point1, genericpointR, origin) == R[0]*R[1]*R[2] - 6
    # 计算 potential_diff_P，并断言 scalar_potential_difference(grad_field, P, origin, genericpointP, origin).simplify() 等于 potential_diff_P
    potential_diff_P = 2*P[2]*(P[0]*sin(q) + P[1]*cos(q))*(P[0]*cos(q) - P[1]*sin(q))**2
    assert scalar_potential_difference(grad_field, P, origin, genericpointP, origin).simplify() == potential_diff_P
```
# `D:\src\scipysrc\sympy\sympy\vector\tests\test_operators.py`

```
# 导入 sympy.vector 模块中的必要组件：CoordSys3D, Gradient, Divergence, Curl, VectorZero, Laplacian
from sympy.vector import CoordSys3D, Gradient, Divergence, Curl, VectorZero, Laplacian
# 导入 sympy.printing.repr 模块中的 srepr 函数
from sympy.printing.repr import srepr

# 创建三维坐标系 R
R = CoordSys3D('R')
# 定义标量场 s1 = R.x * R.y * R.z
s1 = R.x * R.y * R.z  # type: ignore
# 定义标量场 s2 = R.x + 3 * R.y**2
s2 = R.x + 3 * R.y**2  # type: ignore
# 定义标量场 s3 = R.x**2 + R.y**2 + R.z**2
s3 = R.x**2 + R.y**2 + R.z**2  # type: ignore
# 定义向量场 v1 = R.x * R.i + R.z * R.z * R.j
v1 = R.x * R.i + R.z * R.z * R.j  # type: ignore
# 定义向量场 v2 = R.x * R.i + R.y * R.j + R.z * R.k
v2 = R.x * R.i + R.y * R.j + R.z * R.k  # type: ignore
# 定义向量场 v3 = R.x**2 * R.i + R.y**2 * R.j + R.z**2 * R.k
v3 = R.x**2 * R.i + R.y**2 * R.j + R.z**2 * R.k  # type: ignore

# 定义函数 test_Gradient，测试梯度运算
def test_Gradient():
    # 断言梯度 Gradient(s1) 等于 Gradient(R.x * R.y * R.z)
    assert Gradient(s1) == Gradient(R.x * R.y * R.z)
    # 断言梯度 Gradient(s2) 等于 Gradient(R.x + 3 * R.y**2)
    assert Gradient(s2) == Gradient(R.x + 3 * R.y**2)
    # 断言对梯度 Gradient(s1) 进行实际计算（doit）得到的结果
    # 等于 R.y * R.z * R.i + R.x * R.z * R.j + R.x * R.y * R.k
    assert Gradient(s1).doit() == R.y * R.z * R.i + R.x * R.z * R.j + R.x * R.y * R.k
    # 断言对梯度 Gradient(s2) 进行实际计算（doit）得到的结果
    # 等于 R.i + 6 * R.y * R.j
    assert Gradient(s2).doit() == R.i + 6 * R.y * R.j

# 定义函数 test_Divergence，测试散度运算
def test_Divergence():
    # 断言散度 Divergence(v1) 等于 Divergence(R.x * R.i + R.z * R.z * R.j)
    assert Divergence(v1) == Divergence(R.x * R.i + R.z * R.z * R.j)
    # 断言散度 Divergence(v2) 等于 Divergence(R.x * R.i + R.y * R.j + R.z * R.k)
    assert Divergence(v2) == Divergence(R.x * R.i + R.y * R.j + R.z * R.k)
    # 断言对散度 Divergence(v1) 进行实际计算（doit）得到的结果等于 1
    assert Divergence(v1).doit() == 1
    # 断言对散度 Divergence(v2) 进行实际计算（doit）得到的结果等于 3
    assert Divergence(v2).doit() == 3
    # issue 22384 的问题验证
    # 创建一个使用极坐标变换的三维坐标系 Rc
    Rc = CoordSys3D('R', transformation='cylindrical')
    # 断言对极坐标系 Rc 的单位向量 Rc.i 计算散度的结果等于 1/Rc.r
    assert Divergence(Rc.i).doit() == 1 / Rc.r

# 定义函数 test_Curl，测试旋度运算
def test_Curl():
    # 断言旋度 Curl(v1) 等于 Curl(R.x * R.i + R.z * R.z * R.j)
    assert Curl(v1) == Curl(R.x * R.i + R.z * R.z * R.j)
    # 断言旋度 Curl(v2) 等于 Curl(R.x * R.i + R.y * R.j + R.z * R.k)
    assert Curl(v2) == Curl(R.x * R.i + R.y * R.j + R.z * R.k)
    # 断言对旋度 Curl(v1) 进行实际计算（doit）得到的结果等于 (-2 * R.z) * R.i
    assert Curl(v1).doit() == (-2 * R.z) * R.i
    # 断言对旋度 Curl(v2) 进行实际计算（doit）得到的结果等于 VectorZero()
    assert Curl(v2).doit() == VectorZero()

# 定义函数 test_Laplacian，测试拉普拉斯算子运算
def test_Laplacian():
    # 断言拉普拉斯算子 Laplacian(s3) 等于 Laplacian(R.x**2 + R.y**2 + R.z**2)
    assert Laplacian(s3) == Laplacian(R.x**2 + R.y**2 + R.z**2)
    # 断言拉普拉斯算子 Laplacian(v3) 等于 Laplacian(R.x**2 * R.i + R.y**2 * R.j + R.z**2 * R.k)
    assert Laplacian(v3) == Laplacian(R.x**2 * R.i + R.y**2 * R.j + R.z**2 * R.k)
    # 断言对拉普拉斯算子 Laplacian(s3) 进行实际计算（doit）得到的结果等于 6
    assert Laplacian(s3).doit() == 6
    # 断言对拉普拉斯算子 Laplacian(v3) 进行实际计算（doit）得到的结果等于 2 * R.i + 2 * R.j + 2 * R.k
    assert Laplacian(v3).doit() == 2 * R.i + 2 * R.j + 2 * R.k
    # 断言使用 srepr 函数对拉普拉斯算子 Laplacian(s3) 进行表示得到的结果等于指定的字符串
    assert srepr(Laplacian(s3)) == \
            'Laplacian(Add(Pow(R.x, Integer(2)), Pow(R.y, Integer(2)), Pow(R.z, Integer(2))))'
```
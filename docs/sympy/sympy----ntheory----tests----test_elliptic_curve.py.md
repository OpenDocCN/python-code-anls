# `D:\src\scipysrc\sympy\sympy\ntheory\tests\test_elliptic_curve.py`

```
# 导入椭圆曲线模块中的EllipticCurve类
from sympy.ntheory.elliptic_curve import EllipticCurve

# 定义一个测试函数，用于测试椭圆曲线的加法和乘法运算
def test_elliptic_curve():
    # 创建一个椭圆曲线对象e3，参数为(-1, 9)
    e3 = EllipticCurve(-1, 9)
    # 创建两个点p和q，分别为椭圆曲线上的点(0, 3)和(-1, 3)
    p = e3(0, 3)
    q = e3(-1, 3)
    # 计算点p和q的和r，即p + q
    r = p + q
    # 断言r的x坐标为1，y坐标为-3
    assert r.x == 1 and r.y == -3
    # 计算2p + q的结果
    r = 2*p + q
    # 断言r的x坐标为35，y坐标为207
    assert r.x == 35 and r.y == 207
    # 计算-p + q的结果
    r = -p + q
    # 断言r的x坐标为37，y坐标为225
    assert r.x == 37 and r.y == 225
    
    # 验证以下结果在http://www.lmfdb.org/EllipticCurve/Q上
    # 断言椭圆曲线(-1, 9)的判别式为-34928
    assert EllipticCurve(-1, 9).discriminant == -34928
    # 断言椭圆曲线(-2731, -55146, 1, 0, 1)的判别式为25088
    assert EllipticCurve(-2731, -55146, 1, 0, 1).discriminant == 25088
    
    # 断言椭圆曲线(0, 1)的扭转点数量为6
    assert len(EllipticCurve(0, 1).torsion_points()) == 6
```
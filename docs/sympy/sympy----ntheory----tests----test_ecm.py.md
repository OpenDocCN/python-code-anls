# `D:\src\scipysrc\sympy\sympy\ntheory\tests\test_ecm.py`

```
from sympy.external.gmpy import invert
from sympy.ntheory.ecm import ecm, Point
from sympy.testing.pytest import slow

@slow
# 定义一个慢速测试函数，用于测试 ECM 算法
def test_ecm():
    # 断言 ECM 算法对给定整数的返回值是否符合预期
    assert ecm(3146531246531241245132451321) == {3, 100327907731, 10454157497791297}
    assert ecm(46167045131415113) == {43, 2634823, 407485517}
    assert ecm(631211032315670776841) == {9312934919, 67777885039}
    assert ecm(398883434337287) == {99476569, 4009823}
    assert ecm(64211816600515193) == {281719, 359641, 633767}
    assert ecm(4269021180054189416198169786894227) == {184039, 241603, 333331, 477973, 618619, 974123}
    assert ecm(4516511326451341281684513) == {3, 39869, 131743543, 95542348571}
    assert ecm(4132846513818654136451) == {47, 160343, 2802377, 195692803}
    assert ecm(168541512131094651323) == {79, 113, 11011069, 1714635721}
    # 这个测试大约需要 10 秒，而 factorint 即使在大约 10 分钟内也无法对其进行因式分解
    assert ecm(7060005655815754299976961394452809, B1=100000, B2=1000000) == {6988699669998001, 1010203040506070809}

def test_Point():
    # 定义椭圆曲线的形式为 y**2 = x**3 + a*x**2 + x
    mod = 101
    a = 10
    # 计算 a_24 的值，其中 a_24 = (a + 2) * (4^-1 mod mod)
    a_24 = (a + 2)*invert(4, mod)
    # 创建一个椭圆曲线的点 p1
    p1 = Point(10, 17, a_24, mod)
    # 对 p1 进行倍增操作得到 p2
    p2 = p1.double()
    assert p2 == Point(68, 56, a_24, mod)
    # 对 p2 进行倍增操作得到 p4
    p4 = p2.double()
    assert p4 == Point(22, 64, a_24, mod)
    # 对 p4 进行倍增操作得到 p8
    p8 = p4.double()
    assert p8 == Point(71, 95, a_24, mod)
    # 对 p8 进行倍增操作得到 p16
    p16 = p8.double()
    assert p16 == Point(5, 16, a_24, mod)
    # 对 p16 进行倍增操作得到 p32
    p32 = p16.double()
    assert p32 == Point(33, 96, a_24, mod)

    # 计算 p2 + p1 得到 p3
    p3 = p2.add(p1, p1)
    assert p3 == Point(1, 61, a_24, mod)
    # 计算 p3 + p2 或者 p4 + p1 得到 p5
    p5 = p3.add(p2, p1)
    assert p5 == Point(49, 90, a_24, mod)
    assert p5 == p4.add(p1, p3)
    # 计算 2*p3 得到 p6
    p6 = p3.double()
    assert p6 == Point(87, 43, a_24, mod)
    assert p6 == p4.add(p2, p2)
    # 计算 p5 + p2 得到 p7
    p7 = p5.add(p2, p3)
    assert p7 == Point(69, 23, a_24, mod)
    assert p7 == p4.add(p3, p1)
    assert p7 == p6.add(p1, p5)
    # 计算 p5 + p4 得到 p9
    p9 = p5.add(p4, p1)
    assert p9 == Point(56, 99, a_24, mod)
    assert p9 == p6.add(p3, p3)
    assert p9 == p7.add(p2, p5)
    assert p9 == p8.add(p1, p7)

    # 断言使用 Montgomery 梯子算法计算的结果
    assert p5 == p1.mont_ladder(5)
    assert p9 == p1.mont_ladder(9)
    assert p16 == p1.mont_ladder(16)
    assert p9 == p3.mont_ladder(3)
```
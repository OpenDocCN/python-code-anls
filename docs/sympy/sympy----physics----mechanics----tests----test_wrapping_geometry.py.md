# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_wrapping_geometry.py`

```
"""Tests for the ``sympy.physics.mechanics.wrapping_geometry.py`` module."""

# 导入 pytest 模块，用于单元测试
import pytest

# 从 sympy 库中导入需要使用的符号和函数
from sympy import (
    Integer,
    Rational,
    S,
    Symbol,
    acos,
    cos,
    pi,
    sin,
    sqrt,
)

# 从 sympy.core.relational 模块中导入 Eq 类
from sympy.core.relational import Eq

# 从 sympy.physics.mechanics 模块中导入需要使用的类和函数
from sympy.physics.mechanics import (
    Point,
    ReferenceFrame,
    WrappingCylinder,
    WrappingSphere,
    dynamicsymbols,
)

# 从 sympy.simplify.simplify 模块中导入 simplify 函数
from sympy.simplify.simplify import simplify

# 创建符号 r，要求为正数
r = Symbol('r', positive=True)

# 创建符号 x
x = Symbol('x')

# 创建动力学符号 q
q = dynamicsymbols('q')

# 创建参考坐标系 N
N = ReferenceFrame('N')

# 定义测试 WrappingSphere 类的测试类
class TestWrappingSphere:

    # 测试 WrappingSphere 类的有效构造函数
    @staticmethod
    def test_valid_constructor():
        r = Symbol('r', positive=True)
        pO = Point('pO')
        sphere = WrappingSphere(r, pO)
        # 断言 sphere 是 WrappingSphere 类的实例
        assert isinstance(sphere, WrappingSphere)
        # 断言 sphere 对象具有 radius 属性，并验证其值与 r 相等
        assert hasattr(sphere, 'radius')
        assert sphere.radius == r
        # 断言 sphere 对象具有 point 属性，并验证其值与 pO 相等
        assert hasattr(sphere, 'point')
        assert sphere.point == pO

    # 测试当点不在表面上时，geodesic_length 方法是否会引发 ValueError 异常
    @staticmethod
    @pytest.mark.parametrize('position', [S.Zero, Integer(2)*r*N.x])
    def test_geodesic_length_point_not_on_surface_invalid(position):
        r = Symbol('r', positive=True)
        pO = Point('pO')
        sphere = WrappingSphere(r, pO)

        p1 = Point('p1')
        p1.set_pos(pO, position)
        p2 = Point('p2')
        p2.set_pos(pO, position)

        # 设置错误消息的正则表达式，用于匹配错误信息
        error_msg = r'point .* does not lie on the surface of'
        # 使用 pytest.raises 检查是否引发预期异常，并验证错误消息匹配
        with pytest.raises(ValueError, match=error_msg):
            sphere.geodesic_length(p1, p2)

    # 测试 geodesic_length 方法计算 geodesic length 的准确性
    @staticmethod
    @pytest.mark.parametrize(
        'position_1, position_2, expected',
        [
            (r*N.x, r*N.x, S.Zero),
            (r*N.x, r*N.y, S.Half*pi*r),
            (r*N.x, r*-N.x, pi*r),
            (r*-N.x, r*N.x, pi*r),
            (r*N.x, r*sqrt(2)*S.Half*(N.x + N.y), Rational(1, 4)*pi*r),
            (
                r*sqrt(2)*S.Half*(N.x + N.y),
                r*sqrt(3)*Rational(1, 3)*(N.x + N.y + N.z),
                r*acos(sqrt(6)*Rational(1, 3)),
            ),
        ]
    )
    def test_geodesic_length(position_1, position_2, expected):
        r = Symbol('r', positive=True)
        pO = Point('pO')
        sphere = WrappingSphere(r, pO)

        p1 = Point('p1')
        p1.set_pos(pO, position_1)
        p2 = Point('p2')
        p2.set_pos(pO, position_2)

        # 断言 geodesic_length 方法计算结果与预期结果相等
        assert simplify(Eq(sphere.geodesic_length(p1, p2), expected))
    @pytest.mark.parametrize(
        'position_1, position_2, vector_1, vector_2',
        [  # 参数化测试的参数：位置1、位置2、向量1、向量2
            (r * N.x, r * N.y, N.y, N.x),  # 第一组参数：位置1为r*N.x，位置2为r*N.y，向量1为N.y，向量2为N.x
            (r * N.x, -r * N.y, -N.y, N.x),  # 第二组参数：位置1为r*N.x，位置2为-r*N.y，向量1为-N.y，向量2为N.x
            (
                r * N.y,
                sqrt(2)/2 * r * N.x - sqrt(2)/2 * r * N.y,
                N.x,
                sqrt(2)/2 * N.x + sqrt(2)/2 * N.y,
            ),  # 第三组参数：位置1为r*N.y，位置2为sqrt(2)/2*r*N.x-sqrt(2)/2*r*N.y，向量1为N.x，向量2为sqrt(2)/2*N.x+sqrt(2)/2*N.y
            (
                r * N.x,
                r / 2 * N.x + sqrt(3)/2 * r * N.y,
                N.y,
                sqrt(3)/2 * N.x - 1/2 * N.y,
            ),  # 第四组参数：位置1为r*N.x，位置2为r/2*N.x+sqrt(3)/2*r*N.y，向量1为N.y，向量2为sqrt(3)/2*N.x-1/2*N.y
            (
                r * N.x,
                sqrt(2)/2 * r * N.x + sqrt(2)/2 * r * N.y,
                N.y,
                sqrt(2)/2 * N.x - sqrt(2)/2 * N.y,
            ),  # 第五组参数：位置1为r*N.x，位置2为sqrt(2)/2*r*N.x+sqrt(2)/2*r*N.y，向量1为N.y，向量2为sqrt(2)/2*N.x-sqrt(2)/2*N.y
        ]
    )
    def test_geodesic_end_vectors(position_1, position_2, vector_1, vector_2):
        r = Symbol('r', positive=True)  # 定义符号r为正数
        pO = Point('pO')  # 创建一个名为pO的点
        sphere = WrappingSphere(r, pO)  # 创建一个WrappingSphere对象，半径为r，中心为pO

        p1 = Point('p1')  # 创建一个名为p1的点
        p1.set_pos(pO, position_1)  # 将p1设置在pO点上，位置为position_1
        p2 = Point('p2')  # 创建一个名为p2的点
        p2.set_pos(pO, position_2)  # 将p2设置在pO点上，位置为position_2

        expected = (vector_1, vector_2)  # 定义期望的向量组合

        assert sphere.geodesic_end_vectors(p1, p2) == expected  # 断言球面上p1和p2之间的测地线端点向量等于期望的向量组合

    @staticmethod
    @pytest.mark.parametrize(
        'position',
        [r * N.x, r * cos(q) * N.x + r * sin(q) * N.y]
    )
    def test_geodesic_end_vectors_invalid_coincident(position):
        r = Symbol('r', positive=True)  # 定义符号r为正数
        pO = Point('pO')  # 创建一个名为pO的点
        sphere = WrappingSphere(r, pO)  # 创建一个WrappingSphere对象，半径为r，中心为pO

        p1 = Point('p1')  # 创建一个名为p1的点
        p1.set_pos(pO, position)  # 将p1设置在pO点上，位置为position
        p2 = Point('p2')  # 创建一个名为p2的点
        p2.set_pos(pO, position)  # 将p2设置在pO点上，位置为position（与p1位置相同）

        with pytest.raises(ValueError):  # 断言抛出值错误异常
            _ = sphere.geodesic_end_vectors(p1, p2)  # 调用球面上p1和p2之间的测地线端点向量方法

    @staticmethod
    @pytest.mark.parametrize(
        'position_1, position_2',
        [
            (r * N.x, -r * N.x),  # 第一组参数：位置1为r*N.x，位置2为-r*N.x
            (-r * N.y, r * N.y),  # 第二组参数：位置1为-r*N.y，位置2为r*N.y
            (
                r * cos(q) * N.x + r * sin(q) * N.y,
                -r * cos(q) * N.x - r * sin(q) * N.y,
            ),  # 第三组参数：位置1为r*cos(q)*N.x+r*sin(q)*N.y，位置2为-r*cos(q)*N.x-r*sin(q)*N.y
        ]
    )
    def test_geodesic_end_vectors_invalid_diametrically_opposite(
        position_1,
        position_2,
    ):
        r = Symbol('r', positive=True)  # 定义符号r为正数
        pO = Point('pO')  # 创建一个名为pO的点
        sphere = WrappingSphere(r, pO)  # 创建一个WrappingSphere对象，半径为r，中心为pO

        p1 = Point('p1')  # 创建一个名为p1的点
        p1.set_pos(pO, position_1)  # 将p1设置在pO点上，位置为position_1
        p2 = Point('p2')  # 创建一个名为p2的点
        p2.set_pos(pO, position_2)  # 将p2设置在pO点上，位置为position_2

        with pytest.raises(ValueError):  # 断言抛出值错误异常
            _ = sphere.geodesic_end_vectors(p1, p2)  # 调用球面上p1和p2之间的测地线端点向量方法
# 定义一个测试类 TestWrappingCylinder，用于测试 WrappingCylinder 类的功能
class TestWrappingCylinder:

    # 静态方法：测试有效的构造函数
    @staticmethod
    def test_valid_constructor():
        # 创建一个参考坐标系 N
        N = ReferenceFrame('N')
        # 创建一个符号 r，要求为正数
        r = Symbol('r', positive=True)
        # 创建一个点 pO
        pO = Point('pO')
        # 使用 r、pO 和 N.x 创建 WrappingCylinder 对象
        cylinder = WrappingCylinder(r, pO, N.x)
        # 断言 cylinder 是 WrappingCylinder 类的实例
        assert isinstance(cylinder, WrappingCylinder)
        # 断言 cylinder 具有 radius 属性，并且与 r 相等
        assert hasattr(cylinder, 'radius')
        assert cylinder.radius == r
        # 断言 cylinder 具有 point 属性，并且与 pO 相等
        assert hasattr(cylinder, 'point')
        assert cylinder.point == pO
        # 断言 cylinder 具有 axis 属性，并且与 N.x 相等
        assert hasattr(cylinder, 'axis')
        assert cylinder.axis == N.x

    # 静态方法：参数化测试，检查点是否在表面上
    @staticmethod
    @pytest.mark.parametrize(
        'position, expected',
        [
            (S.Zero, False),
            (r*N.y, True),
            (r*N.z, True),
            (r*(N.y + N.z).normalize(), True),
            (Integer(2)*r*N.y, False),
            (r*(N.x + N.y), True),
            (r*(Integer(2)*N.x + N.y), True),
            (Integer(2)*N.x + r*(Integer(2)*N.y + N.z).normalize(), True),
            (r*(cos(q)*N.y + sin(q)*N.z), True)
        ]
    )
    def test_point_is_on_surface(position, expected):
        # 创建符号 r，要求为正数
        r = Symbol('r', positive=True)
        # 创建点 pO
        pO = Point('pO')
        # 使用 r、pO 和 N.x 创建 WrappingCylinder 对象
        cylinder = WrappingCylinder(r, pO, N.x)

        # 创建点 p1
        p1 = Point('p1')
        # 将 p1 设置在以 pO 为参考点，位置为 position 处
        p1.set_pos(pO, position)

        # 断言 cylinder.point_on_surface(p1) 的返回值与 expected 相等
        assert cylinder.point_on_surface(p1) is expected

    # 静态方法：参数化测试，检查非表面上的点是否会导致无效长度异常
    @staticmethod
    @pytest.mark.parametrize('position', [S.Zero, Integer(2)*r*N.y])
    def test_geodesic_length_point_not_on_surface_invalid(position):
        # 创建符号 r，要求为正数
        r = Symbol('r', positive=True)
        # 创建点 pO
        pO = Point('pO')
        # 使用 r、pO 和 N.x 创建 WrappingCylinder 对象
        cylinder = WrappingCylinder(r, pO, N.x)

        # 创建点 p1 和 p2
        p1 = Point('p1')
        p1.set_pos(pO, position)
        p2 = Point('p2')
        p2.set_pos(pO, position)

        # 设置错误消息的正则表达式模式
        error_msg = r'point .* does not lie on the surface of'
        # 使用 pytest 断言，断言 cylinder.geodesic_length(p1, p2) 会抛出 ValueError 异常，并且异常消息符合 error_msg
        with pytest.raises(ValueError, match=error_msg):
            cylinder.geodesic_length(p1, p2)

    # 静态方法：参数化测试，检查测地线长度计算是否正确
    @staticmethod
    @pytest.mark.parametrize(
        'axis, position_1, position_2, expected',
        [
            (N.x, r*N.y, r*N.y, S.Zero),
            (N.x, r*N.y, N.x + r*N.y, S.One),
            (N.x, r*N.y, -x*N.x + r*N.y, sqrt(x**2)),
            (-N.x, r*N.y, x*N.x + r*N.y, sqrt(x**2)),
            (N.x, r*N.y, r*N.z, S.Half*pi*sqrt(r**2)),
            (-N.x, r*N.y, r*N.z, Integer(3)*S.Half*pi*sqrt(r**2)),
            (N.x, r*N.z, r*N.y, Integer(3)*S.Half*pi*sqrt(r**2)),
            (-N.x, r*N.z, r*N.y, S.Half*pi*sqrt(r**2)),
            (N.x, r*N.y, r*(cos(q)*N.y + sin(q)*N.z), sqrt(r**2*q**2)),
            (
                -N.x, r*N.y,
                r*(cos(q)*N.y + sin(q)*N.z),
                sqrt(r**2*(Integer(2)*pi - q)**2),
            ),
        ]
    )
    def test_geodesic_length(axis, position_1, position_2, expected):
        # 创建符号 r，要求为正数
        r = Symbol('r', positive=True)
        # 创建点 pO
        pO = Point('pO')
        # 使用 r、pO 和 axis 创建 WrappingCylinder 对象
        cylinder = WrappingCylinder(r, pO, axis)

        # 创建点 p1 和 p2
        p1 = Point('p1')
        p1.set_pos(pO, position_1)
        p2 = Point('p2')
        p2.set_pos(pO, position_2)

        # 断言测地线长度计算结果与 expected 相等
        assert simplify(Eq(cylinder.geodesic_length(p1, p2), expected))
    # 使用 pytest 的 parametrize 装饰器为 test_geodesic_end_vectors 方法定义多个参数化的测试用例
    @staticmethod
    @pytest.mark.parametrize(
        'axis, position_1, position_2, vector_1, vector_2',
        [
            # 测试用例 1
            (N.z, r * N.x, r * N.y, N.y, N.x),
            # 测试用例 2
            (N.z, r * N.x, -r * N.x, N.y, N.y),
            # 测试用例 3
            (N.z, -r * N.x, r * N.x, -N.y, -N.y),
            # 测试用例 4
            (-N.z, r * N.x, -r * N.x, -N.y, -N.y),
            # 测试用例 5
            (-N.z, -r * N.x, r * N.x, N.y, N.y),
            # 测试用例 6
            (N.z, r * N.x, -r * N.y, N.y, -N.x),
            # 测试用例 7
            (N.z, r * N.y, sqrt(2)/2 * r * N.x - sqrt(2)/2 * r * N.y, - N.x, - sqrt(2)/2 * N.x - sqrt(2)/2 * N.y),
            # 测试用例 8
            (N.z, r * N.x, r / 2 * N.x + sqrt(3)/2 * r * N.y, N.y, sqrt(3)/2 * N.x - 1/2 * N.y),
            # 测试用例 9
            (N.z, r * N.x, sqrt(2)/2 * r * N.x + sqrt(2)/2 * r * N.y, N.y, sqrt(2)/2 * N.x - sqrt(2)/2 * N.y),
            # 测试用例 10
            (N.z, r * N.x, r * N.x + N.z, N.z, -N.z),
            # 测试用例 11
            (N.z, r * N.x, r * N.y + pi/2 * r * N.z, sqrt(2)/2 * N.y + sqrt(2)/2 * N.z, sqrt(2)/2 * N.x - sqrt(2)/2 * N.z),
            # 测试用例 12
            (N.z, r * N.x, r * cos(q) * N.x + r * sin(q) * N.y, N.y, sin(q) * N.x - cos(q) * N.y),
        ]
    )
    # 测试方法：验证圆柱体的测地线终点向量计算是否正确
    def test_geodesic_end_vectors(
        axis,
        position_1,
        position_2,
        vector_1,
        vector_2,
    ):
        # 声明符号变量 r 为正数
        r = Symbol('r', positive=True)
        # 创建点 pO 作为圆柱体的中心点
        pO = Point('pO')
        # 创建 WrappingCylinder 对象，以 r, pO, axis 作为参数
        cylinder = WrappingCylinder(r, pO, axis)

        # 创建点 p1 和 p2，并分别设置其相对于 pO 的位置
        p1 = Point('p1')
        p1.set_pos(pO, position_1)
        p2 = Point('p2')
        p2.set_pos(pO, position_2)

        # 期望的结果向量组合
        expected = (vector_1, vector_2)
        # 获取测地线终点向量并简化
        end_vectors = tuple(
            end_vector.simplify()
            for end_vector in cylinder.geodesic_end_vectors(p1, p2)
        )

        # 断言计算出的结果与期望的结果相等
        assert end_vectors == expected

    # 使用 pytest 的 parametrize 装饰器为 test_geodesic_end_vectors_invalid_coincident 方法定义多个参数化的测试用例
    @staticmethod
    @pytest.mark.parametrize(
        'axis, position',
        [
            # 测试用例 1
            (N.z, r * N.x),
            # 测试用例 2
            (N.z, r * cos(q) * N.x + r * sin(q) * N.y + N.z),
        ]
    )
    # 测试方法：验证圆柱体在重合位置时测地线终点向量计算是否触发 ValueError 异常
    def test_geodesic_end_vectors_invalid_coincident(axis, position):
        # 声明符号变量 r 为正数
        r = Symbol('r', positive=True)
        # 创建点 pO 作为圆柱体的中心点
        pO = Point('pO')
        # 创建 WrappingCylinder 对象，以 r, pO, axis 作为参数
        cylinder = WrappingCylinder(r, pO, axis)

        # 创建点 p1 和 p2，将它们都设置为相同的位置
        p1 = Point('p1')
        p1.set_pos(pO, position)
        p2 = Point('p2')
        p2.set_pos(pO, position)

        # 断言当 p1 和 p2 位置重合时，调用测地线终点向量方法会触发 ValueError 异常
        with pytest.raises(ValueError):
            _ = cylinder.geodesic_end_vectors(p1, p2)
```
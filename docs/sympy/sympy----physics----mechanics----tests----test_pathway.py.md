# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_pathway.py`

```
"""Tests for the ``sympy.physics.mechanics.pathway.py`` module."""

# 导入pytest库，用于编写和运行测试用例
import pytest

# 导入sympy库中需要使用的符号和函数
from sympy import (
    Rational,
    Symbol,
    cos,
    pi,
    sin,
    sqrt,
)

# 导入sympy.physics.mechanics模块中的各个类和函数
from sympy.physics.mechanics import (
    Force,
    LinearPathway,
    ObstacleSetPathway,
    PathwayBase,
    Point,
    ReferenceFrame,
    WrappingCylinder,
    WrappingGeometryBase,
    WrappingPathway,
    WrappingSphere,
    dynamicsymbols,
)

# 导入sympy.simplify.simplify模块中的simplify函数
from sympy.simplify.simplify import simplify


# 定义一个私有函数_simplify_loads，用于简化载荷列表中的载荷
def _simplify_loads(loads):
    return [
        load.__class__(load.location, load.vector.simplify())
        for load in loads
    ]


# 定义一个测试类TestLinearPathway，用于测试LinearPathway类的功能
class TestLinearPathway:

    # 测试LinearPathway是否是PathwayBase的子类
    def test_is_pathway_base_subclass(self):
        assert issubclass(LinearPathway, PathwayBase)

    # 静态方法，使用pytest的参数化装饰器，测试LinearPathway的有效构造函数
    @staticmethod
    @pytest.mark.parametrize(
        'args, kwargs',
        [
            ((Point('pA'), Point('pB')), {}),
        ]
    )
    def test_valid_constructor(args, kwargs):
        pointA, pointB = args
        instance = LinearPathway(*args, **kwargs)
        assert isinstance(instance, LinearPathway)
        assert hasattr(instance, 'attachments')
        assert len(instance.attachments) == 2
        assert instance.attachments[0] is pointA
        assert instance.attachments[1] is pointB
        assert isinstance(instance.attachments[0], Point)
        assert instance.attachments[0].name == 'pA'
        assert isinstance(instance.attachments[1], Point)
        assert instance.attachments[1].name == 'pB'

    # 静态方法，使用pytest的参数化装饰器，测试LinearPathway的无效构造函数（附件数不正确）
    @staticmethod
    @pytest.mark.parametrize(
        'attachments',
        [
            (Point('pA'), ),
            (Point('pA'), Point('pB'), Point('pZ')),
        ]
    )
    def test_invalid_attachments_incorrect_number(attachments):
        with pytest.raises(ValueError):
            _ = LinearPathway(*attachments)

    # 静态方法，使用pytest的参数化装饰器，测试LinearPathway的无效构造函数（附件不是Point对象）
    @staticmethod
    @pytest.mark.parametrize(
        'attachments',
        [
            (None, Point('pB')),
            (Point('pA'), None),
        ]
    )
    def test_invalid_attachments_not_point(attachments):
        with pytest.raises(TypeError):
            _ = LinearPathway(*attachments)

    # pytest的自动使用fixture，为测试类提供预设环境（设置参考坐标系、点和路径）
    @pytest.fixture(autouse=True)
    def _linear_pathway_fixture(self):
        self.N = ReferenceFrame('N')
        self.pA = Point('pA')
        self.pB = Point('pB')
        self.pathway = LinearPathway(self.pA, self.pB)
        self.q1 = dynamicsymbols('q1')
        self.q2 = dynamicsymbols('q2')
        self.q3 = dynamicsymbols('q3')
        self.q1d = dynamicsymbols('q1', 1)
        self.q2d = dynamicsymbols('q2', 1)
        self.q3d = dynamicsymbols('q3', 1)
        self.F = Symbol('F')

    # 测试LinearPathway实例的属性是否不可变
    def test_properties_are_immutable(self):
        instance = LinearPathway(self.pA, self.pB)
        with pytest.raises(AttributeError):
            instance.attachments = None
        with pytest.raises(TypeError):
            instance.attachments[0] = None
        with pytest.raises(TypeError):
            instance.attachments[1] = None
    # 测试 LinearPathway 对象的 repr 方法是否正确
    def test_repr(self):
        # 创建 LinearPathway 对象，使用 self.pA 和 self.pB 作为参数
        pathway = LinearPathway(self.pA, self.pB)
        # 预期的对象字符串表示
        expected = 'LinearPathway(pA, pB)'
        # 断言对象的字符串表示是否与预期相等
        assert repr(pathway) == expected

    # 测试静态路径长度的计算
    def test_static_pathway_length(self):
        # 设置 self.pB 的位置，使得路径长度为 2*self.N.x
        self.pB.set_pos(self.pA, 2*self.N.x)
        # 断言路径对象的长度是否为 2
        assert self.pathway.length == 2

    # 测试静态路径扩展速度的计算
    def test_static_pathway_extension_velocity(self):
        # 设置 self.pB 的位置，使得路径长度为 2*self.N.x
        self.pB.set_pos(self.pA, 2*self.N.x)
        # 断言路径对象的扩展速度是否为 0
        assert self.pathway.extension_velocity == 0

    # 测试静态路径加载情况的计算
    def test_static_pathway_to_loads(self):
        # 设置 self.pB 的位置，使得路径长度为 2*self.N.x
        self.pB.set_pos(self.pA, 2*self.N.x)
        # 预期的加载情况列表
        expected = [
            (self.pA, - self.F*self.N.x),
            (self.pB, self.F*self.N.x),
        ]
        # 断言路径对象加载情况是否与预期相等
        assert self.pathway.to_loads(self.F) == expected

    # 测试二维路径长度的计算
    def test_2D_pathway_length(self):
        # 设置 self.pB 的位置，使得路径长度为 2*self.q1*self.N.x
        self.pB.set_pos(self.pA, 2*self.q1*self.N.x)
        # 预期的路径长度
        expected = 2*sqrt(self.q1**2)
        # 断言路径对象的长度是否与预期相等
        assert self.pathway.length == expected

    # 测试二维路径扩展速度的计算
    def test_2D_pathway_extension_velocity(self):
        # 设置 self.pB 的位置，使得路径长度为 2*self.q1*self.N.x
        self.pB.set_pos(self.pA, 2*self.q1*self.N.x)
        # 预期的路径扩展速度
        expected = 2*sqrt(self.q1**2)*self.q1d/self.q1
        # 断言路径对象的扩展速度是否与预期相等
        assert self.pathway.extension_velocity == expected

    # 测试二维路径加载情况的计算
    def test_2D_pathway_to_loads(self):
        # 设置 self.pB 的位置，使得路径长度为 2*self.q1*self.N.x
        self.pB.set_pos(self.pA, 2*self.q1*self.N.x)
        # 预期的加载情况列表
        expected = [
            (self.pA, - self.F*(self.q1 / sqrt(self.q1**2))*self.N.x),
            (self.pB, self.F*(self.q1 / sqrt(self.q1**2))*self.N.x),
        ]
        # 断言路径对象加载情况是否与预期相等
        assert self.pathway.to_loads(self.F) == expected

    # 测试三维路径长度的计算
    def test_3D_pathway_length(self):
        # 设置 self.pB 的位置，使得路径长度为 sqrt(self.q1**2 + self.q2**2 + 4*self.q3**2)
        self.pB.set_pos(
            self.pA,
            self.q1*self.N.x - self.q2*self.N.y + 2*self.q3*self.N.z,
        )
        # 预期的路径长度
        expected = sqrt(self.q1**2 + self.q2**2 + 4*self.q3**2)
        # 使用 simplify 函数来断言路径对象长度与预期是否近似相等
        assert simplify(self.pathway.length - expected) == 0

    # 测试三维路径扩展速度的计算
    def test_3D_pathway_extension_velocity(self):
        # 设置 self.pB 的位置，使得路径长度为 sqrt(self.q1**2 + self.q2**2 + 4*self.q3**2)
        self.pB.set_pos(
            self.pA,
            self.q1*self.N.x - self.q2*self.N.y + 2*self.q3*self.N.z,
        )
        # 计算路径长度
        length = sqrt(self.q1**2 + self.q2**2 + 4*self.q3**2)
        # 预期的路径扩展速度
        expected = (
            self.q1*self.q1d/length
            + self.q2*self.q2d/length
            + 4*self.q3*self.q3d/length
        )
        # 使用 simplify 函数来断言路径对象扩展速度与预期是否近似相等
        assert simplify(self.pathway.extension_velocity - expected) == 0

    # 测试三维路径加载情况的计算
    def test_3D_pathway_to_loads(self):
        # 设置 self.pB 的位置，使得路径长度为 sqrt(self.q1**2 + self.q2**2 + 4*self.q3**2)
        self.pB.set_pos(
            self.pA,
            self.q1*self.N.x - self.q2*self.N.y + 2*self.q3*self.N.z,
        )
        # 计算路径长度
        length = sqrt(self.q1**2 + self.q2**2 + 4*self.q3**2)
        # 计算外部点和内部点的力
        pO_force = (
            - self.F*self.q1*self.N.x/length
            + self.F*self.q2*self.N.y/length
            - 2*self.F*self.q3*self.N.z/length
        )
        pI_force = (
            self.F*self.q1*self.N.x/length
            - self.F*self.q2*self.N.y/length
            + 2*self.F*self.q3*self.N.z/length
        )
        # 预期的加载情况列表
        expected = [
            (self.pA, pO_force),
            (self.pB, pI_force),
        ]
        # 断言路径对象加载情况是否与预期相等
        assert self.pathway.to_loads(self.F) == expected
class TestObstacleSetPathway:

    def test_is_pathway_base_subclass(self):
        assert issubclass(ObstacleSetPathway, PathwayBase)

    @staticmethod
    @pytest.mark.parametrize(
        'num_attachments, attachments',
        [
            (3, [Point(name) for name in ('pO', 'pA', 'pI')]),
            (4, [Point(name) for name in ('pO', 'pA', 'pB', 'pI')]),
            (5, [Point(name) for name in ('pO', 'pA', 'pB', 'pC', 'pI')]),
            (6, [Point(name) for name in ('pO', 'pA', 'pB', 'pC', 'pD', 'pI')]),
        ]
    )
    def test_valid_constructor(num_attachments, attachments):
        instance = ObstacleSetPathway(*attachments)
        assert isinstance(instance, ObstacleSetPathway)
        assert hasattr(instance, 'attachments')
        assert len(instance.attachments) == num_attachments
        for attachment in instance.attachments:
            assert isinstance(attachment, Point)

    @staticmethod
    @pytest.mark.parametrize(
        'attachments',
        [[Point('pO')], [Point('pO'), Point('pI')]],
    )
    def test_invalid_constructor_attachments_incorrect_number(attachments):
        # 检查构造函数调用时如果附件数量不正确是否会抛出 ValueError 异常
        with pytest.raises(ValueError):
            _ = ObstacleSetPathway(*attachments)

    @staticmethod
    @pytest.mark.parametrize(
        'attachments',
        [
            (None, Point('pA'), Point('pI')),
            (Point('pO'), None, Point('pI')),
            (Point('pO'), Point('pA'), None),
        ]
    )
    def test_invalid_constructor_attachments_not_point(attachments):
        # 检查构造函数调用时如果附件不是 Point 对象是否会抛出 TypeError 异常
        with pytest.raises(TypeError):
            _ = WrappingPathway(*attachments)  # type: ignore

    def test_properties_are_immutable(self):
        # 测试属性 attachments 是否为不可变类型
        pathway = ObstacleSetPathway(Point('pO'), Point('pA'), Point('pI'))
        with pytest.raises(AttributeError):
            pathway.attachments = None  # type: ignore
        with pytest.raises(TypeError):
            pathway.attachments[0] = None  # type: ignore
        with pytest.raises(TypeError):
            pathway.attachments[1] = None  # type: ignore
        with pytest.raises(TypeError):
            pathway.attachments[-1] = None  # type: ignore

    @staticmethod
    @pytest.mark.parametrize(
        'attachments, expected',
        [
            (
                [Point(name) for name in ('pO', 'pA', 'pI')],
                'ObstacleSetPathway(pO, pA, pI)'
            ),
            (
                [Point(name) for name in ('pO', 'pA', 'pB', 'pI')],
                'ObstacleSetPathway(pO, pA, pB, pI)'
            ),
            (
                [Point(name) for name in ('pO', 'pA', 'pB', 'pC', 'pI')],
                'ObstacleSetPathway(pO, pA, pB, pC, pI)'
            ),
        ]
    )
    def test_repr(attachments, expected):
        # 测试 ObstacleSetPathway 对象的 __repr__ 方法
        pathway = ObstacleSetPathway(*attachments)
        assert repr(pathway) == expected

    @pytest.fixture(autouse=True)
    # 定义一个私有方法用于设置路径中的固定点和参考系
    def _obstacle_set_pathway_fixture(self):
        # 创建一个参考系 'N'
        self.N = ReferenceFrame('N')
        # 创建四个点对象，分别命名为 pO, pI, pA, pB
        self.pO = Point('pO')
        self.pI = Point('pI')
        self.pA = Point('pA')
        self.pB = Point('pB')
        # 创建动力学符号 q 和 q 的一阶导数 qd
        self.q = dynamicsymbols('q')
        self.qd = dynamicsymbols('q', 1)
        # 创建一个符号 F
        self.F = Symbol('F')

    # 测试静态路径长度的方法
    def test_static_pathway_length(self):
        # 设置点 pA, pB, pI 相对于点 pO 在参考系 N 中的位置
        self.pA.set_pos(self.pO, self.N.x)
        self.pB.set_pos(self.pO, self.N.y)
        self.pI.set_pos(self.pO, self.N.z)
        # 创建一个 ObstacleSetPathway 对象 pathway，并计算其长度是否等于 1 + 2 * sqrt(2)
        pathway = ObstacleSetPathway(self.pO, self.pA, self.pB, self.pI)
        assert pathway.length == 1 + 2 * sqrt(2)

    # 测试静态路径延伸速度的方法
    def test_static_pathway_extension_velocity(self):
        # 设置点 pA, pB, pI 相对于点 pO 在参考系 N 中的位置
        self.pA.set_pos(self.pO, self.N.x)
        self.pB.set_pos(self.pO, self.N.y)
        self.pI.set_pos(self.pO, self.N.z)
        # 创建一个 ObstacleSetPathway 对象 pathway，并检查其延伸速度是否为 0
        pathway = ObstacleSetPathway(self.pO, self.pA, self.pB, self.pI)
        assert pathway.extension_velocity == 0

    # 测试静态路径转化为力的方法
    def test_static_pathway_to_loads(self):
        # 设置点 pA, pB, pI 相对于点 pO 在参考系 N 中的位置
        self.pA.set_pos(self.pO, self.N.x)
        self.pB.set_pos(self.pO, self.N.y)
        self.pI.set_pos(self.pO, self.N.z)
        # 创建一个 ObstacleSetPathway 对象 pathway，并计算其转化为力的结果是否符合预期
        pathway = ObstacleSetPathway(self.pO, self.pA, self.pB, self.pI)
        expected = [
            Force(self.pO, -self.F * self.N.x),
            Force(self.pA, self.F * self.N.x),
            Force(self.pA, self.F * sqrt(2) / 2 * (self.N.x - self.N.y)),
            Force(self.pB, self.F * sqrt(2) / 2 * (self.N.y - self.N.x)),
            Force(self.pB, self.F * sqrt(2) / 2 * (self.N.y - self.N.z)),
            Force(self.pI, self.F * sqrt(2) / 2 * (self.N.z - self.N.y)),
        ]
        assert pathway.to_loads(self.F) == expected

    # 测试二维路径长度的方法
    def test_2D_pathway_length(self):
        # 设置点 pA, pB, pI 相对于点 pO 在参考系 N 中的位置
        self.pA.set_pos(self.pO, -(self.N.x + self.N.y))
        self.pB.set_pos(
            self.pO, cos(self.q) * self.N.x - (sin(self.q) + 1) * self.N.y
        )
        self.pI.set_pos(
            self.pO, sin(self.q) * self.N.x + (cos(self.q) - 1) * self.N.y
        )
        # 创建一个 ObstacleSetPathway 对象 pathway，并计算其长度是否满足表达式 2 * sqrt(2) + sqrt(2 + 2*cos(q))
        pathway = ObstacleSetPathway(self.pO, self.pA, self.pB, self.pI)
        expected = 2 * sqrt(2) + sqrt(2 + 2*cos(self.q))
        assert (pathway.length - expected).simplify() == 0

    # 测试二维路径延伸速度的方法
    def test_2D_pathway_extension_velocity(self):
        # 设置点 pA, pB, pI 相对于点 pO 在参考系 N 中的位置
        self.pA.set_pos(self.pO, -(self.N.x + self.N.y))
        self.pB.set_pos(
            self.pO, cos(self.q) * self.N.x - (sin(self.q) + 1) * self.N.y
        )
        self.pI.set_pos(
            self.pO, sin(self.q) * self.N.x + (cos(self.q) - 1) * self.N.y
        )
        # 创建一个 ObstacleSetPathway 对象 pathway，并计算其延伸速度是否符合预期表达式
        pathway = ObstacleSetPathway(self.pO, self.pA, self.pB, self.pI)
        expected = - (sqrt(2) * sin(self.q) * self.qd) / (2 * sqrt(cos(self.q) + 1))
        assert (pathway.extension_velocity - expected).simplify() == 0
    # 定义一个测试方法，用于测试二维路径到载荷的转换功能
    def test_2D_pathway_to_loads(self):
        # 设置点 pA 的位置，相对于点 pO，根据负向的 x 和 y 轴的和
        self.pA.set_pos(self.pO, -(self.N.x + self.N.y))
        # 设置点 pB 的位置，相对于点 pO，根据余弦和正弦函数计算的结果
        self.pB.set_pos(
            self.pO, cos(self.q) * self.N.x - (sin(self.q) + 1) * self.N.y
        )
        # 设置点 pI 的位置，相对于点 pO，根据正弦和余弦函数计算的结果
        self.pI.set_pos(
            self.pO, sin(self.q) * self.N.x + (cos(self.q) - 1) * self.N.y
        )
        # 创建一个阻碍路径对象，传入点 pO, pA, pB, pI
        pathway = ObstacleSetPathway(self.pO, self.pA, self.pB, self.pI)
        # 计算从点 pO 到点 pA 的力向量，根据给定的公式
        pO_pA_force_vec = sqrt(2) / 2 * (self.N.x + self.N.y)
        # 计算从点 pA 到点 pB 的力向量，根据给定的公式
        pA_pB_force_vec = (
            - sqrt(2 * cos(self.q) + 2) / 2 * self.N.x
            + sqrt(2) * sin(self.q) / (2 * sqrt(cos(self.q) + 1)) * self.N.y
        )
        # 计算从点 pB 到点 pI 的力向量，根据给定的公式
        pB_pI_force_vec = cos(self.q + pi/4) * self.N.x - sin(self.q + pi/4) * self.N.y
        # 期望的力列表，包含各个点之间的力
        expected = [
            Force(self.pO, self.F * pO_pA_force_vec),
            Force(self.pA, -self.F * pO_pA_force_vec),
            Force(self.pA, self.F * pA_pB_force_vec),
            Force(self.pB, -self.F * pA_pB_force_vec),
            Force(self.pB, self.F * pB_pI_force_vec),
            Force(self.pI, -self.F * pB_pI_force_vec),
        ]
        # 断言路径对象经过转换后的载荷与期望的力列表相同
        assert _simplify_loads(pathway.to_loads(self.F)) == expected
class TestWrappingPathway:

    # 确保 WrappingPathway 是 PathwayBase 的子类
    def test_is_pathway_base_subclass(self):
        assert issubclass(WrappingPathway, PathwayBase)

    # 创建自动使用的 pytest fixture，设置测试环境
    @pytest.fixture(autouse=True)
    def _wrapping_pathway_fixture(self):
        # 创建测试用的点对象和符号
        self.pA = Point('pA')
        self.pB = Point('pB')
        self.r = Symbol('r', positive=True)
        self.pO = Point('pO')
        self.N = ReferenceFrame('N')
        self.ax = self.N.z
        # 创建 WrappingSphere 和 WrappingCylinder 实例
        self.sphere = WrappingSphere(self.r, self.pO)
        self.cylinder = WrappingCylinder(self.r, self.pO, self.ax)
        # 创建 WrappingPathway 实例并进行初始化
        self.pathway = WrappingPathway(self.pA, self.pB, self.cylinder)
        self.F = Symbol('F')

    # 验证 WrappingPathway 构造函数的有效性
    def test_valid_constructor(self):
        instance = WrappingPathway(self.pA, self.pB, self.cylinder)
        # 断言实例是 WrappingPathway 类的对象
        assert isinstance(instance, WrappingPathway)
        # 断言实例具有 'attachments' 属性，并且长度为 2
        assert hasattr(instance, 'attachments')
        assert len(instance.attachments) == 2
        # 断言第一个 attachment 是 Point 类型并且与 self.pA 相同
        assert isinstance(instance.attachments[0], Point)
        assert instance.attachments[0] == self.pA
        # 断言第二个 attachment 是 Point 类型并且与 self.pB 相同
        assert isinstance(instance.attachments[1], Point)
        assert instance.attachments[1] == self.pB
        # 断言实例具有 'geometry' 属性，并且是 WrappingGeometryBase 类型
        assert hasattr(instance, 'geometry')
        assert isinstance(instance.geometry, WrappingGeometryBase)
        # 断言 geometry 属性等于 self.cylinder
        assert instance.geometry == self.cylinder

    # 参数化测试，验证构造函数中 attachment 数量不正确的情况
    @pytest.mark.parametrize(
        'attachments',
        [
            (Point('pA'), ),
            (Point('pA'), Point('pB'), Point('pZ')),
        ]
    )
    def test_invalid_constructor_attachments_incorrect_number(self, attachments):
        with pytest.raises(TypeError):
            _ = WrappingPathway(*attachments, self.cylinder)

    # 参数化测试，验证构造函数中 attachment 不是 Point 类型的情况
    @staticmethod
    @pytest.mark.parametrize(
        'attachments',
        [
            (None, Point('pB')),
            (Point('pA'), None),
        ]
    )
    def test_invalid_constructor_attachments_not_point(attachments):
        with pytest.raises(TypeError):
            _ = WrappingPathway(*attachments)

    # 测试构造函数中未提供 geometry 参数的情况
    def test_invalid_constructor_geometry_is_not_supplied(self):
        with pytest.raises(TypeError):
            _ = WrappingPathway(self.pA, self.pB)

    # 参数化测试，验证构造函数中 geometry 不是合法的几何对象的情况
    @pytest.mark.parametrize(
        'geometry',
        [
            Symbol('r'),
            dynamicsymbols('q'),
            ReferenceFrame('N'),
            ReferenceFrame('N').x,
        ]
    )
    def test_invalid_geometry_not_geometry(self, geometry):
        with pytest.raises(TypeError):
            _ = WrappingPathway(self.pA, self.pB, geometry)

    # 验证 attachments 属性不可变
    def test_attachments_property_is_immutable(self):
        with pytest.raises(TypeError):
            self.pathway.attachments[0] = self.pB
        with pytest.raises(TypeError):
            self.pathway.attachments[1] = self.pA

    # 验证 geometry 属性不可变
    def test_geometry_property_is_immutable(self):
        with pytest.raises(AttributeError):
            self.pathway.geometry = None

    # 验证 __repr__() 方法的输出是否符合预期
    def test_repr(self):
        expected = (
            f'WrappingPathway(pA, pB, '
            f'geometry={self.cylinder!r})'
        )
        assert repr(self.pathway) == expected
    # 将位置向量从局部坐标系转换为全局坐标系中的向量
    @staticmethod
    def _expand_pos_to_vec(pos, frame):
        return sum(mag*unit for (mag, unit) in zip(pos, frame))

    # 测试球面上静态路径的长度
    @pytest.mark.parametrize(
        'pA_vec, pB_vec, factor',
        [
            # 测试1：路径在球面上的两点为北极和东经90度，路径因子为π/2
            ((1, 0, 0), (0, 1, 0), pi/2),
            # 测试2：路径在球面上的两点为赤道上的两点，路径因子为3π/4
            ((0, 1, 0), (sqrt(2)/2, -sqrt(2)/2, 0), 3*pi/4),
            # 测试3：路径在球面上的两点为赤道上的两点，路径因子为π/3
            ((1, 0, 0), (Rational(1, 2), sqrt(3)/2, 0), pi/3),
        ]
    )
    def test_static_pathway_on_sphere_length(self, pA_vec, pB_vec, factor):
        pA_vec = self._expand_pos_to_vec(pA_vec, self.N)
        pB_vec = self._expand_pos_to_vec(pB_vec, self.N)
        self.pA.set_pos(self.pO, self.r*pA_vec)
        self.pB.set_pos(self.pO, self.r*pB_vec)
        pathway = WrappingPathway(self.pA, self.pB, self.sphere)
        expected = factor*self.r
        # 断言路径长度与预期值的差异为0
        assert simplify(pathway.length - expected) == 0

    # 测试圆柱体表面上静态路径的长度
    @pytest.mark.parametrize(
        'pA_vec, pB_vec, factor',
        [
            # 测试1：路径在圆柱体上的两点为轴线上的两点，路径因子为π/2
            ((1, 0, 0), (0, 1, 0), Rational(1, 2)*pi),
            # 测试2：路径在圆柱体上的两点为直径上的两点，路径因子为π
            ((1, 0, 0), (-1, 0, 0), pi),
            # 测试3：路径在圆柱体上的两点为直径上的两点，路径因子为π
            ((-1, 0, 0), (1, 0, 0), pi),
            # 测试4：路径在圆柱体上的两点为对角线上的两点，路径因子为5π/4
            ((0, 1, 0), (sqrt(2)/2, -sqrt(2)/2, 0), 5*pi/4),
            # 测试5：路径在圆柱体上的两点为赤道上的两点，路径因子为π/3
            ((1, 0, 0), (Rational(1, 2), sqrt(3)/2, 0), pi/3),
            # 测试6：路径在圆柱体上的两点为对角线上的两点，路径因子为sqrt(1 + (5/4*pi)**2)
            (
                (0, 1, 0),
                (sqrt(2)*Rational(1, 2), -sqrt(2)*Rational(1, 2), 1),
                sqrt(1 + (Rational(5, 4)*pi)**2),
            ),
            # 测试7：路径在圆柱体上的两点为赤道上的两点，路径因子为sqrt(1 + (1/3*pi)**2)
            (
                (1, 0, 0),
                (Rational(1, 2), sqrt(3)*Rational(1, 2), 1),
                sqrt(1 + (Rational(1, 3)*pi)**2),
            ),
        ]
    )
    def test_static_pathway_on_cylinder_length(self, pA_vec, pB_vec, factor):
        pA_vec = self._expand_pos_to_vec(pA_vec, self.N)
        pB_vec = self._expand_pos_to_vec(pB_vec, self.N)
        self.pA.set_pos(self.pO, self.r*pA_vec)
        self.pB.set_pos(self.pO, self.r*pB_vec)
        pathway = WrappingPathway(self.pA, self.pB, self.cylinder)
        expected = factor*sqrt(self.r**2)
        # 断言路径长度与预期值的差异为0
        assert simplify(pathway.length - expected) == 0

    # 测试球面上静态路径的延展速度
    @pytest.mark.parametrize(
        'pA_vec, pB_vec',
        [
            # 测试1：路径在球面上的两点为北极和东经90度
            ((1, 0, 0), (0, 1, 0)),
            # 测试2：路径在球面上的两点为赤道上的两点
            ((0, 1, 0), (sqrt(2)*Rational(1, 2), -sqrt(2)*Rational(1, 2), 0)),
            # 测试3：路径在球面上的两点为赤道上的两点
            ((1, 0, 0), (Rational(1, 2), sqrt(3)*Rational(1, 2), 0)),
        ]
    )
    def test_static_pathway_on_sphere_extension_velocity(self, pA_vec, pB_vec):
        pA_vec = self._expand_pos_to_vec(pA_vec, self.N)
        pB_vec = self._expand_pos_to_vec(pB_vec, self.N)
        self.pA.set_pos(self.pO, self.r*pA_vec)
        self.pB.set_pos(self.pO, self.r*pB_vec)
        pathway = WrappingPathway(self.pA, self.pB, self.sphere)
        # 断言球面上的静态路径的延展速度为0
        assert pathway.extension_velocity == 0
    @pytest.mark.parametrize(
        'pA_vec, pB_vec',
        [  # 参数化测试的参数，包含多组pA_vec和pB_vec
            ((1, 0, 0), (0, 1, 0)),  # 第一组参数
            ((1, 0, 0), (-1, 0, 0)),  # 第二组参数
            ((-1, 0, 0), (1, 0, 0)),  # 第三组参数
            ((0, 1, 0), (sqrt(2)/2, -sqrt(2)/2, 0)),  # 第四组参数
            ((1, 0, 0), (Rational(1, 2), sqrt(3)/2, 0)),  # 第五组参数
            ((0, 1, 0), (sqrt(2)*Rational(1, 2), -sqrt(2)/2, 1)),  # 第六组参数
            ((1, 0, 0), (Rational(1, 2), sqrt(3)/2, 1)),  # 第七组参数
        ]
    )
    def test_static_pathway_on_cylinder_extension_velocity(self, pA_vec, pB_vec):
        pA_vec = self._expand_pos_to_vec(pA_vec, self.N)  # 将pA_vec扩展为向量
        pB_vec = self._expand_pos_to_vec(pB_vec, self.N)  # 将pB_vec扩展为向量
        self.pA.set_pos(self.pO, self.r*pA_vec)  # 设置点pA相对于点pO的位置
        self.pB.set_pos(self.pO, self.r*pB_vec)  # 设置点pB相对于点pO的位置
        pathway = WrappingPathway(self.pA, self.pB, self.cylinder)  # 创建一个围绕路径对象
        assert pathway.extension_velocity == 0  # 断言路径的伸展速度为0

    @pytest.mark.parametrize(
        'pA_vec, pB_vec, pA_vec_expected, pB_vec_expected, pO_vec_expected',
        (  # 参数化测试的参数，包含多组pA_vec、pB_vec以及预期的pA_vec_expected、pB_vec_expected、pO_vec_expected
            ((1, 0, 0), (0, 1, 0), (0, 1, 0), (1, 0, 0), (-1, -1, 0)),  # 第一组参数及预期值
            (
                (0, 1, 0),
                (sqrt(2)/2, -sqrt(2)/2, 0),
                (1, 0, 0),
                (sqrt(2)/2, sqrt(2)/2, 0),
                (-1 - sqrt(2)/2, -sqrt(2)/2, 0)
            ),  # 第二组参数及预期值
            (
                (1, 0, 0),
                (Rational(1, 2), sqrt(3)/2, 0),
                (0, 1, 0),
                (sqrt(3)/2, -Rational(1, 2), 0),
                (-sqrt(3)/2, Rational(1, 2) - 1, 0),
            ),  # 第三组参数及预期值
        )
    )
    def test_static_pathway_on_sphere_to_loads(
        self,
        pA_vec,
        pB_vec,
        pA_vec_expected,
        pB_vec_expected,
        pO_vec_expected,
    ):
        pA_vec = self._expand_pos_to_vec(pA_vec, self.N)  # 将pA_vec扩展为向量
        pB_vec = self._expand_pos_to_vec(pB_vec, self.N)  # 将pB_vec扩展为向量
        self.pA.set_pos(self.pO, self.r*pA_vec)  # 设置点pA相对于点pO的位置
        self.pB.set_pos(self.pO, self.r*pB_vec)  # 设置点pB相对于点pO的位置
        pathway = WrappingPathway(self.pA, self.pB, self.sphere)  # 创建一个围绕路径对象

        pA_vec_expected = sum(
            mag*unit for (mag, unit) in zip(pA_vec_expected, self.N)
        )  # 计算pA_vec_expected的加权和
        pB_vec_expected = sum(
            mag*unit for (mag, unit) in zip(pB_vec_expected, self.N)
        )  # 计算pB_vec_expected的加权和
        pO_vec_expected = sum(
            mag*unit for (mag, unit) in zip(pO_vec_expected, self.N)
        )  # 计算pO_vec_expected的加权和
        expected = [
            Force(self.pA, self.F*(self.r**3/sqrt(self.r**6))*pA_vec_expected),  # 创建点pA上的预期力对象
            Force(self.pB, self.F*(self.r**3/sqrt(self.r**6))*pB_vec_expected),  # 创建点pB上的预期力对象
            Force(self.pO, self.F*(self.r**3/sqrt(self.r**6))*pO_vec_expected),  # 创建点pO上的预期力对象
        ]
        assert pathway.to_loads(self.F) == expected  # 断言路径转换为力的预期结果与expected相等
    @pytest.mark.parametrize(
        'pA_vec, pB_vec, pA_vec_expected, pB_vec_expected, pO_vec_expected',
        (  # 参数化测试用例，定义了多组输入和期望输出向量
            ((1, 0, 0), (0, 1, 0), (0, 1, 0), (1, 0, 0), (-1, -1, 0)),
            ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 1, 0), (0, -2, 0)),
            ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, -1, 0), (0, 2, 0)),
            (  # 使用数学函数参数化复杂向量的测试用例
                (0, 1, 0),
                (sqrt(2)/2, -sqrt(2)/2, 0),
                (-1, 0, 0),
                (-sqrt(2)/2, -sqrt(2)/2, 0),
                (1 + sqrt(2)/2, sqrt(2)/2, 0)
            ),
            (  # 使用有理数和数学函数参数化向量的测试用例
                (1, 0, 0),
                (Rational(1, 2), sqrt(3)/2, 0),
                (0, 1, 0),
                (sqrt(3)/2, -Rational(1, 2), 0),
                (-sqrt(3)/2, Rational(1, 2) - 1, 0),
            ),
            (  # 使用数学函数参数化向量的测试用例
                (1, 0, 0),
                (sqrt(2)/2, sqrt(2)/2, 0),
                (0, 1, 0),
                (sqrt(2)/2, -sqrt(2)/2, 0),
                (-sqrt(2)/2, sqrt(2)/2 - 1, 0),
            ),
            ((0, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, -1), (0, 0, 0)),
            (  # 使用数学函数参数化复杂向量的测试用例
                (0, 1, 0),
                (sqrt(2)/2, -sqrt(2)/2, 1),
                (-5*pi/sqrt(16 + 25*pi**2), 0, 4/sqrt(16 + 25*pi**2)),
                (
                    -5*sqrt(2)*pi/(2*sqrt(16 + 25*pi**2)),
                    -5*sqrt(2)*pi/(2*sqrt(16 + 25*pi**2)),
                    -4/sqrt(16 + 25*pi**2),
                ),
                (
                    5*(sqrt(2) + 2)*pi/(2*sqrt(16 + 25*pi**2)),
                    5*sqrt(2)*pi/(2*sqrt(16 + 25*pi**2)),
                    0,
                ),
            ),
        )
    )
    def test_static_pathway_on_cylinder_to_loads(
        self,
        pA_vec,
        pB_vec,
        pA_vec_expected,
        pB_vec_expected,
        pO_vec_expected,
    ):
        pA_vec = self._expand_pos_to_vec(pA_vec, self.N)  # 扩展pA_vec到向量形式
        pB_vec = self._expand_pos_to_vec(pB_vec, self.N)  # 扩展pB_vec到向量形式
        self.pA.set_pos(self.pO, self.r*pA_vec)  # 设置点pA的位置相对于点pO的位移
        self.pB.set_pos(self.pO, self.r*pB_vec)  # 设置点pB的位置相对于点pO的位移
        pathway = WrappingPathway(self.pA, self.pB, self.cylinder)  # 创建一个路径对象

        pA_force_expected = self.F*self._expand_pos_to_vec(pA_vec_expected,
                                                           self.N)  # 计算预期pA点的力
        pB_force_expected = self.F*self._expand_pos_to_vec(pB_vec_expected,
                                                           self.N)  # 计算预期pB点的力
        pO_force_expected = self.F*self._expand_pos_to_vec(pO_vec_expected,
                                                           self.N)  # 计算预期pO点的力
        expected = [
            Force(self.pA, pA_force_expected),  # 创建pA点的力对象
            Force(self.pB, pB_force_expected),  # 创建pB点的力对象
            Force(self.pO, pO_force_expected),  # 创建pO点的力对象
        ]
        assert _simplify_loads(pathway.to_loads(self.F)) == expected  # 断言路径的负载与预期相等
    # 测试在圆柱体上的二维路径的长度
    def test_2D_pathway_on_cylinder_length(self):
        # 定义动力学符号 q
        q = dynamicsymbols('q')
        # 设置点 pA 的位置
        pA_pos = self.r*self.N.x
        # 设置点 pB 的位置
        pB_pos = self.r*(cos(q)*self.N.x + sin(q)*self.N.y)
        # 将点 pA 相对于点 pO 的位置设定为 pA_pos
        self.pA.set_pos(self.pO, pA_pos)
        # 将点 pB 相对于点 pO 的位置设定为 pB_pos
        self.pB.set_pos(self.pO, pB_pos)
        # 预期路径长度为半径 r 乘以 sqrt(q**2)
        expected = self.r*sqrt(q**2)
        # 断言路径长度与预期值的简化结果为零
        assert simplify(self.pathway.length - expected) == 0

    # 测试在圆柱体上的二维路径的延伸速度
    def test_2D_pathway_on_cylinder_extension_velocity(self):
        # 定义动力学符号 q 和 q 的一阶导数 qd
        q = dynamicsymbols('q')
        qd = dynamicsymbols('q', 1)
        # 设置点 pA 的位置
        pA_pos = self.r*self.N.x
        # 设置点 pB 的位置
        pB_pos = self.r*(cos(q)*self.N.x + sin(q)*self.N.y)
        # 将点 pA 相对于点 pO 的位置设定为 pA_pos
        self.pA.set_pos(self.pO, pA_pos)
        # 将点 pB 相对于点 pO 的位置设定为 pB_pos
        self.pB.set_pos(self.pO, pB_pos)
        # 预期延伸速度为半径乘以 (sqrt(q**2)/q) 乘以 q 的一阶导数 qd
        expected = self.r*(sqrt(q**2)/q)*qd
        # 断言路径的延伸速度与预期值的简化结果为零
        assert simplify(self.pathway.extension_velocity - expected) == 0

    # 测试在圆柱体上的二维路径到载荷的转换
    def test_2D_pathway_on_cylinder_to_loads(self):
        # 定义动力学符号 q
        q = dynamicsymbols('q')
        # 设置点 pA 的位置
        pA_pos = self.r*self.N.x
        # 设置点 pB 的位置
        pB_pos = self.r*(cos(q)*self.N.x + sin(q)*self.N.y)
        # 将点 pA 相对于点 pO 的位置设定为 pA_pos
        self.pA.set_pos(self.pO, pA_pos)
        # 将点 pB 相对于点 pO 的位置设定为 pB_pos
        self.pB.set_pos(self.pO, pB_pos)

        # 计算点 pA 和 pB 所受的力
        pA_force = self.F*self.N.y
        pB_force = self.F*(sin(q)*self.N.x - cos(q)*self.N.y)
        # 计算点 pO 所受的力
        pO_force = self.F*(-sin(q)*self.N.x + (cos(q) - 1)*self.N.y)
        # 预期结果是一个由 Force 对象组成的列表
        expected = [
            Force(self.pA, pA_force),
            Force(self.pB, pB_force),
            Force(self.pO, pO_force),
        ]

        # 简化路径上载荷的结果并与预期结果进行断言比较
        loads = _simplify_loads(self.pathway.to_loads(self.F))
        assert loads == expected
```
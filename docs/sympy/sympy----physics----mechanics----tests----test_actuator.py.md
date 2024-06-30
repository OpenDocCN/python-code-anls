# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_actuator.py`

```
"""Tests for the ``sympy.physics.mechanics.actuator.py`` module."""

# 导入 pytest 模块，用于编写和运行测试
import pytest

# 导入 sympy 中需要使用的类和函数
from sympy import (
    S,              # 符号 S，表示精确的数学常量
    Matrix,         # 矩阵类
    Symbol,         # 符号类
    SympifyError,   # sympify 函数可能抛出的异常类
    sqrt,           # 平方根函数
    Abs             # 绝对值函数
)

# 导入 sympy.physics.mechanics 中需要使用的类
from sympy.physics.mechanics import (
    ActuatorBase,   # 执行器基类
    Force,          # 力类
    ForceActuator,  # 力驱动器类
    KanesMethod,    # Kanes 方法
    LinearDamper,   # 线性阻尼器类
    LinearPathway,  # 线性路径类
    LinearSpring,   # 线性弹簧类
    Particle,       # 粒子类
    PinJoint,       # 固定关节类
    Point,          # 点类
    ReferenceFrame, # 参考框架类
    RigidBody,      # 刚体类
    TorqueActuator, # 扭矩驱动器类
    Vector,         # 向量类
    dynamicsymbols, # 动力学符号函数
    DuffingSpring   # 杜芬弹簧类
)

# 导入 sympy.core.expr 中的 Expr 类型
from sympy.core.expr import Expr as ExprType

# 创建 RigidBody 类的实例
target = RigidBody('target')
reaction = RigidBody('reaction')

# 定义 ForceActuator 类的测试类
class TestForceActuator:

    # 使用 pytest.fixture 标记的自动运行的装置，设置测试环境
    @pytest.fixture(autouse=True)
    def _linear_pathway_fixture(self):
        # 定义力符号 'F'
        self.force = Symbol('F')
        # 创建名为 'pA' 和 'pB' 的点实例
        self.pA = Point('pA')
        self.pB = Point('pB')
        # 创建线性路径实例，连接 'pA' 和 'pB'
        self.pathway = LinearPathway(self.pA, self.pB)
        # 定义动力学符号 'q1', 'q2', 'q3'，及其一阶导数 'q1d', 'q2d', 'q3d'
        self.q1 = dynamicsymbols('q1')
        self.q2 = dynamicsymbols('q2')
        self.q3 = dynamicsymbols('q3')
        self.q1d = dynamicsymbols('q1', 1)
        self.q2d = dynamicsymbols('q2', 1)
        self.q3d = dynamicsymbols('q3', 1)
        # 创建参考框架 'N'
        self.N = ReferenceFrame('N')

    # 测试 ForceActuator 是否是 ActuatorBase 的子类
    def test_is_actuator_base_subclass(self):
        assert issubclass(ForceActuator, ActuatorBase)

    # 参数化测试：验证 ForceActuator 构造函数中力的有效性
    @pytest.mark.parametrize(
        'force, expected_force',
        [
            (1, S.One),                         # 整数力值
            (S.One, S.One),                     # 符号 S.One
            (Symbol('F'), Symbol('F')),         # 符号 'F'
            (dynamicsymbols('F'), dynamicsymbols('F')), # 动力学符号 'F'
            (Symbol('F')**2 + Symbol('F'), Symbol('F')**2 + Symbol('F')), # 表达式
        ]
    )
    def test_valid_constructor_force(self, force, expected_force):
        # 创建 ForceActuator 实例
        instance = ForceActuator(force, self.pathway)
        assert isinstance(instance, ForceActuator)     # 断言实例为 ForceActuator 类型
        assert hasattr(instance, 'force')              # 断言实例有属性 'force'
        assert isinstance(instance.force, ExprType)    # 断言 'force' 属性是 ExprType 类型
        assert instance.force == expected_force        # 断言 'force' 属性的值与预期相等

    # 参数化测试：验证 ForceActuator 构造函数中力不可 sympify 的情况
    @pytest.mark.parametrize('force', [None, 'F'])
    def test_invalid_constructor_force_not_sympifyable(self, force):
        with pytest.raises(SympifyError):  # 断言抛出 SympifyError 异常
            _ = ForceActuator(force, self.pathway)

    # 参数化测试：验证 ForceActuator 构造函数中路径的有效性
    @pytest.mark.parametrize(
        'pathway',
        [
            LinearPathway(Point('pA'), Point('pB')), # 合法的线性路径
        ]
    )
    def test_valid_constructor_pathway(self, pathway):
        # 创建 ForceActuator 实例
        instance = ForceActuator(self.force, pathway)
        assert isinstance(instance, ForceActuator)         # 断言实例为 ForceActuator 类型
        assert hasattr(instance, 'pathway')                # 断言实例有属性 'pathway'
        assert isinstance(instance.pathway, LinearPathway) # 断言 'pathway' 属性是 LinearPathway 类型
        assert instance.pathway == pathway                 # 断言 'pathway' 属性的值与预期相等

    # 测试：验证 ForceActuator 构造函数中路径不是 LinearPathway 类型时的异常
    def test_invalid_constructor_pathway_not_pathway_base(self):
        with pytest.raises(TypeError):  # 断言抛出 TypeError 异常
            _ = ForceActuator(self.force, None)

    # 参数化测试：验证属性名和 fixture 属性名是否对应
    @pytest.mark.parametrize(
        'property_name, fixture_attr_name',
        [
            ('force', 'force'),         # 属性名 'force'
            ('pathway', 'pathway'),     # 属性名 'pathway'
        ]
    )
    # 测试属性不可变性的方法，验证在实例化后设置属性会引发 AttributeError 异常
    def test_properties_are_immutable(self, property_name, fixture_attr_name):
        # 创建 ForceActuator 实例
        instance = ForceActuator(self.force, self.pathway)
        # 获取当前测试实例中的指定属性值
        value = getattr(self, fixture_attr_name)
        # 使用 pytest 验证设置属性时会引发 AttributeError 异常
        with pytest.raises(AttributeError):
            setattr(instance, property_name, value)

    # 测试 ForceActuator 实例的字符串表示方法
    def test_repr(self):
        # 创建 ForceActuator 实例
        actuator = ForceActuator(self.force, self.pathway)
        # 预期的字符串表示
        expected = "ForceActuator(F, LinearPathway(pA, pB))"
        # 验证实际的字符串表示与预期是否一致
        assert repr(actuator) == expected

    # 测试在静态路径下生成加载的方法
    def test_to_loads_static_pathway(self):
        # 设置路径 pB 相对于 pA 的位置
        self.pB.set_pos(self.pA, 2*self.N.x)
        # 创建 ForceActuator 实例
        actuator = ForceActuator(self.force, self.pathway)
        # 预期的加载列表
        expected = [
            (self.pA, - self.force*self.N.x),
            (self.pB, self.force*self.N.x),
        ]
        # 验证生成的加载列表与预期是否一致
        assert actuator.to_loads() == expected

    # 测试在二维路径下生成加载的方法
    def test_to_loads_2D_pathway(self):
        # 设置路径 pB 相对于 pA 的位置
        self.pB.set_pos(self.pA, 2*self.q1*self.N.x)
        # 创建 ForceActuator 实例
        actuator = ForceActuator(self.force, self.pathway)
        # 预期的加载列表
        expected = [
            (self.pA, - self.force*(self.q1/sqrt(self.q1**2))*self.N.x),
            (self.pB, self.force*(self.q1/sqrt(self.q1**2))*self.N.x),
        ]
        # 验证生成的加载列表与预期是否一致
        assert actuator.to_loads() == expected

    # 测试在三维路径下生成加载的方法
    def test_to_loads_3D_pathway(self):
        # 设置路径 pB 相对于 pA 的位置
        self.pB.set_pos(
            self.pA,
            self.q1*self.N.x - self.q2*self.N.y + 2*self.q3*self.N.z,
        )
        # 创建 ForceActuator 实例
        actuator = ForceActuator(self.force, self.pathway)
        # 计算路径长度
        length = sqrt(self.q1**2 + self.q2**2 + 4*self.q3**2)
        # 计算施加在 pO 和 pI 上的力
        pO_force = (
            - self.force*self.q1*self.N.x/length
            + self.force*self.q2*self.N.y/length
            - 2*self.force*self.q3*self.N.z/length
        )
        pI_force = (
            self.force*self.q1*self.N.x/length
            - self.force*self.q2*self.N.y/length
            + 2*self.force*self.q3*self.N.z/length
        )
        # 预期的加载列表
        expected = [
            (self.pA, pO_force),
            (self.pB, pI_force),
        ]
        # 验证生成的加载列表与预期是否一致
        assert actuator.to_loads() == expected
# 定义一个测试类 TestLinearSpring
class TestLinearSpring:

    # 在每个测试方法之前执行的 pytest fixture，用于设置测试环境
    @pytest.fixture(autouse=True)
    def _linear_spring_fixture(self):
        # 定义符号变量 stiffness 和 l，表示弹簧的刚度和平衡长度
        self.stiffness = Symbol('k')
        self.l = Symbol('l')
        # 定义点 pA 和 pB，表示弹簧的两端点
        self.pA = Point('pA')
        self.pB = Point('pB')
        # 创建一个线性路径对象 LinearPathway，连接点 pA 和 pB
        self.pathway = LinearPathway(self.pA, self.pB)
        # 定义动力学符号 q，表示弹簧的位移
        self.q = dynamicsymbols('q')
        # 创建一个参考坐标系 N
        self.N = ReferenceFrame('N')

    # 测试 LinearSpring 是否是 ForceActuator 的子类
    def test_is_force_actuator_subclass(self):
        assert issubclass(LinearSpring, ForceActuator)

    # 测试 LinearSpring 是否是 ActuatorBase 的子类
    def test_is_actuator_base_subclass(self):
        assert issubclass(LinearSpring, ActuatorBase)

    # 参数化测试方法，测试 LinearSpring 的构造函数是否正确
    @pytest.mark.parametrize(
        (
            'stiffness, '
            'expected_stiffness, '
            'equilibrium_length, '
            'expected_equilibrium_length, '
            'force'
        ),
        [
            (
                1,
                S.One,
                0,
                S.Zero,
                -sqrt(dynamicsymbols('q')**2),
            ),
            (
                Symbol('k'),
                Symbol('k'),
                0,
                S.Zero,
                -Symbol('k')*sqrt(dynamicsymbols('q')**2),
            ),
            (
                Symbol('k'),
                Symbol('k'),
                S.Zero,
                S.Zero,
                -Symbol('k')*sqrt(dynamicsymbols('q')**2),
            ),
            (
                Symbol('k'),
                Symbol('k'),
                Symbol('l'),
                Symbol('l'),
                -Symbol('k')*(sqrt(dynamicsymbols('q')**2) - Symbol('l')),
            ),
        ]
    )
    def test_valid_constructor(
        self,
        stiffness,
        expected_stiffness,
        equilibrium_length,
        expected_equilibrium_length,
        force,
    ):
        # 设置点 pB 相对于点 pA 的位置，位置由 q*N.x 表示
        self.pB.set_pos(self.pA, self.q*self.N.x)
        # 创建一个 LinearSpring 对象 spring
        spring = LinearSpring(stiffness, self.pathway, equilibrium_length)

        # 断言 spring 是 LinearSpring 类的实例
        assert isinstance(spring, LinearSpring)

        # 断言 spring 对象有属性 stiffness，并且其类型是 ExprType，值等于 expected_stiffness
        assert hasattr(spring, 'stiffness')
        assert isinstance(spring.stiffness, ExprType)
        assert spring.stiffness == expected_stiffness

        # 断言 spring 对象有属性 pathway，并且其类型是 LinearPathway，值等于 self.pathway
        assert hasattr(spring, 'pathway')
        assert isinstance(spring.pathway, LinearPathway)
        assert spring.pathway == self.pathway

        # 断言 spring 对象有属性 equilibrium_length，并且其类型是 ExprType，值等于 expected_equilibrium_length
        assert hasattr(spring, 'equilibrium_length')
        assert isinstance(spring.equilibrium_length, ExprType)
        assert spring.equilibrium_length == expected_equilibrium_length

        # 断言 spring 对象有属性 force，并且其类型是 ExprType，值等于 force
        assert hasattr(spring, 'force')
        assert isinstance(spring.force, ExprType)
        assert spring.force == force

    # 参数化测试方法，测试构造函数中 stiffness 参数不可转换为符号表达式时是否抛出异常
    @pytest.mark.parametrize('stiffness', [None, 'k'])
    def test_invalid_constructor_stiffness_not_sympifyable(self, stiffness):
        with pytest.raises(SympifyError):
            _ = LinearSpring(stiffness, self.pathway, self.l)

    # 测试构造函数中 pathway 参数不是 LinearPathway 类或其子类时是否抛出异常
    def test_invalid_constructor_pathway_not_pathway_base(self):
        with pytest.raises(TypeError):
            _ = LinearSpring(self.stiffness, None, self.l)

    # 参数化测试方法，测试构造函数中 equilibrium_length 参数不可转换为符号表达式时是否抛出异常
    @pytest.mark.parametrize('equilibrium_length', [None, 'l'])
    # 测试构造函数 LinearSpring 对于非法参数 equilibrium_length 是否会引发 SympifyError 异常
    def test_invalid_constructor_equilibrium_length_not_sympifyable(
        self,
        equilibrium_length,
    ):
        # 使用 pytest 检查 LinearSpring 构造函数对于非法 equilibrium_length 是否抛出 SympifyError 异常
        with pytest.raises(SympifyError):
            _ = LinearSpring(self.stiffness, self.pathway, equilibrium_length)

    # 使用 pytest.mark.parametrize 来参数化测试，检查属性 stiffness、pathway 和 equilibrium_length 是否是不可变的
    @pytest.mark.parametrize(
        'property_name, fixture_attr_name',
        [
            ('stiffness', 'stiffness'),
            ('pathway', 'pathway'),
            ('equilibrium_length', 'l'),
        ]
    )
    # 测试 LinearSpring 对象的属性是否为不可变
    def test_properties_are_immutable(self, property_name, fixture_attr_name):
        # 创建 LinearSpring 对象
        spring = LinearSpring(self.stiffness, self.pathway, self.l)
        # 获取当前测试用例中的属性值
        value = getattr(self, fixture_attr_name)
        # 使用 pytest 检查设置属性时是否会引发 AttributeError 异常
        with pytest.raises(AttributeError):
            setattr(spring, property_name, value)

    # 使用 pytest.mark.parametrize 参数化测试，检查 LinearSpring 对象的字符串表示形式是否正确
    @pytest.mark.parametrize(
        'equilibrium_length, expected',
        [
            (S.Zero, 'LinearSpring(k, LinearPathway(pA, pB))'),
            (
                Symbol('l'),
                'LinearSpring(k, LinearPathway(pA, pB), equilibrium_length=l)',
            ),
        ]
    )
    # 测试 LinearSpring 对象的字符串表示形式是否符合预期
    def test_repr(self, equilibrium_length, expected):
        # 设置点 pB 的位置
        self.pB.set_pos(self.pA, self.q*self.N.x)
        # 创建 LinearSpring 对象
        spring = LinearSpring(self.stiffness, self.pathway, equilibrium_length)
        # 检查 LinearSpring 对象的字符串表示是否与预期一致
        assert repr(spring) == expected

    # 测试 LinearSpring 对象的 to_loads 方法
    def test_to_loads(self):
        # 设置点 pB 的位置
        self.pB.set_pos(self.pA, self.q*self.N.x)
        # 创建 LinearSpring 对象
        spring = LinearSpring(self.stiffness, self.pathway, self.l)
        # 计算正常力 normal
        normal = self.q/sqrt(self.q**2)*self.N.x
        # 计算点 pA 和 pB 的力
        pA_force = self.stiffness*(sqrt(self.q**2) - self.l)*normal
        pB_force = -self.stiffness*(sqrt(self.q**2) - self.l)*normal
        # 期望的力列表
        expected = [Force(self.pA, pA_force), Force(self.pB, pB_force)]
        # 获取 LinearSpring 对象的载荷
        loads = spring.to_loads()

        # 遍历每一个载荷和期望的力，检查载荷是否为 Force 对象，并且力的点和向量是否与期望一致
        for load, (point, vector) in zip(loads, expected):
            assert isinstance(load, Force)
            assert load.point == point
            assert (load.vector - vector).simplify() == 0
# 定义测试类 TestLinearDamper，用于测试 LinearDamper 类的功能
class TestLinearDamper:

    # 使用 pytest 的 fixture 自动调用功能，在每个测试方法执行前初始化对象属性
    @pytest.fixture(autouse=True)
    def _linear_damper_fixture(self):
        # 定义阻尼、长度和两个点的符号
        self.damping = Symbol('c')
        self.l = Symbol('l')
        self.pA = Point('pA')
        self.pB = Point('pB')
        # 创建线性路径对象 LinearPathway，连接两个点 pA 和 pB
        self.pathway = LinearPathway(self.pA, self.pB)
        # 定义动力符号 q 和其一阶导数 dq
        self.q = dynamicsymbols('q')
        self.dq = dynamicsymbols('q', 1)
        # 定义控制力符号 u 和参考坐标系 N
        self.u = dynamicsymbols('u')
        self.N = ReferenceFrame('N')

    # 测试 LinearDamper 是否是 ForceActuator 的子类
    def test_is_force_actuator_subclass(self):
        assert issubclass(LinearDamper, ForceActuator)

    # 测试 LinearDamper 是否是 ActuatorBase 的子类
    def test_is_actuator_base_subclass(self):
        assert issubclass(LinearDamper, ActuatorBase)

    # 测试 LinearDamper 的有效构造函数
    def test_valid_constructor(self):
        # 设置点 pB 相对于点 pA 的位置，依赖于动力符号 q
        self.pB.set_pos(self.pA, self.q*self.N.x)
        # 创建 LinearDamper 对象
        damper = LinearDamper(self.damping, self.pathway)

        # 断言 damper 是 LinearDamper 的实例
        assert isinstance(damper, LinearDamper)

        # 断言 damper 具有属性 damping，类型为 ExprType，且与 self.damping 相等
        assert hasattr(damper, 'damping')
        assert isinstance(damper.damping, ExprType)
        assert damper.damping == self.damping

        # 断言 damper 具有属性 pathway，类型为 LinearPathway，且与 self.pathway 相等
        assert hasattr(damper, 'pathway')
        assert isinstance(damper.pathway, LinearPathway)
        assert damper.pathway == self.pathway

    # 测试 LinearDamper 的有效构造函数中的力
    def test_valid_constructor_force(self):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        damper = LinearDamper(self.damping, self.pathway)

        # 计算预期的阻尼力
        expected_force = -self.damping*sqrt(self.q**2)*self.dq/self.q
        # 断言 damper 具有属性 force，类型为 ExprType，且与 expected_force 相等
        assert hasattr(damper, 'force')
        assert isinstance(damper.force, ExprType)
        assert damper.force == expected_force

    # 使用 pytest 参数化测试无效的构造函数，当 damping 参数不符合符号化时应引发 SympifyError
    @pytest.mark.parametrize('damping', [None, 'c'])
    def test_invalid_constructor_damping_not_sympifyable(self, damping):
        with pytest.raises(SympifyError):
            _ = LinearDamper(damping, self.pathway)

    # 测试无效的构造函数，当 pathway 不是 PathwayBase 类型时应引发 TypeError
    def test_invalid_constructor_pathway_not_pathway_base(self):
        with pytest.raises(TypeError):
            _ = LinearDamper(self.damping, None)

    # 使用 pytest 参数化测试属性是否不可变性，属性名称和 fixture 属性名应一致
    @pytest.mark.parametrize(
        'property_name, fixture_attr_name',
        [
            ('damping', 'damping'),
            ('pathway', 'pathway'),
        ]
    )
    def test_properties_are_immutable(self, property_name, fixture_attr_name):
        damper = LinearDamper(self.damping, self.pathway)
        value = getattr(self, fixture_attr_name)
        # 断言试图设置 damper 的属性会引发 AttributeError
        with pytest.raises(AttributeError):
            setattr(damper, property_name, value)

    # 测试 LinearDamper 的字符串表示形式是否符合预期
    def test_repr(self):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        damper = LinearDamper(self.damping, self.pathway)
        expected = 'LinearDamper(c, LinearPathway(pA, pB))'
        # 断言 damper 的字符串表示形式与 expected 相等
        assert repr(damper) == expected

    # 测试 LinearDamper 的载荷生成函数 to_loads 是否生成预期的力列表
    def test_to_loads(self):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        damper = LinearDamper(self.damping, self.pathway)
        # 定义方向向量 direction
        direction = self.q**2/self.q**2*self.N.x
        # 计算点 pA 和 pB 上的预期力
        pA_force = self.damping*self.dq*direction
        pB_force = -self.damping*self.dq*direction
        expected = [Force(self.pA, pA_force), Force(self.pB, pB_force)]
        # 断言 damper 的 to_loads 方法生成的力列表与 expected 相等
        assert damper.to_loads() == expected


class TestForcedMassSpringDamperModel():
    r"""A single degree of freedom translational forced mass-spring-damper.

    Notes
    =====

    This system is well known to have the governing equation:

    .. math::
        m \ddot{x} = F - k x - c \dot{x}

    where $F$ is an externally applied force, $m$ is the mass of the particle
    to which the spring and damper are attached, $k$ is the spring's stiffness,
    $c$ is the damper's damping coefficient, and $x$ is the generalized
    coordinate representing the system's single (translational) degree of
    freedom.

    """

    # 定义 Pytest 的 Fixture，用于创建和初始化质点弹簧阻尼系统模型
    @pytest.fixture(autouse=True)
    def _force_mass_spring_damper_model_fixture(self):
        # 定义系统参数符号
        self.m = Symbol('m')
        self.k = Symbol('k')
        self.c = Symbol('c')
        self.F = Symbol('F')

        # 定义动力学符号变量
        self.q = dynamicsymbols('q')       # 广义坐标
        self.dq = dynamicsymbols('q', 1)   # 广义速度
        self.u = dynamicsymbols('u')       # 外力

        # 创建参考坐标系和基准点
        self.frame = ReferenceFrame('N')
        self.origin = Point('pO')
        self.origin.set_vel(self.frame, 0)

        # 创建连接点
        self.attachment = Point('pA')
        self.attachment.set_pos(self.origin, self.q*self.frame.x)

        # 创建质点和线性路径
        self.mass = Particle('mass', self.attachment, self.m)
        self.pathway = LinearPathway(self.origin, self.attachment)

        # 创建 KanesMethod 对象用于系统动力学建模
        self.kanes_method = KanesMethod(
            self.frame,
            q_ind=[self.q],         # 广义坐标
            u_ind=[self.u],         # 外力
            kd_eqs=[self.dq - self.u],  # 动力学方程
        )
        self.bodies = [self.mass]   # 系统质点列表

        # 定义质量矩阵
        self.mass_matrix = Matrix([[self.m]])

        # 定义系统受力
        self.forcing = Matrix([[self.F - self.c*self.u - self.k*self.q]])

    # 测试外力作用
    def test_force_acuator(self):
        # 计算刚度
        stiffness = -self.k*self.pathway.length
        # 创建弹簧作用
        spring = ForceActuator(stiffness, self.pathway)
        # 计算阻尼
        damping = -self.c*self.pathway.extension_velocity
        # 创建阻尼作用
        damper = ForceActuator(damping, self.pathway)

        # 构建作用力列表
        loads = [
            (self.attachment, self.F*self.frame.x),  # 外力作用点
            *spring.to_loads(),                     # 弹簧作用力
            *damper.to_loads(),                     # 阻尼作用力
        ]
        # 计算 Kanes 方程
        self.kanes_method.kanes_equations(self.bodies, loads)

        # 验证质量矩阵
        assert self.kanes_method.mass_matrix == self.mass_matrix
        # 验证系统受力
        assert self.kanes_method.forcing == self.forcing

    # 测试线性弹簧和线性阻尼
    def test_linear_spring_linear_damper(self):
        # 创建线性弹簧
        spring = LinearSpring(self.k, self.pathway)
        # 创建线性阻尼
        damper = LinearDamper(self.c, self.pathway)

        # 构建作用力列表
        loads = [
            (self.attachment, self.F*self.frame.x),  # 外力作用点
            *spring.to_loads(),                     # 弹簧作用力
            *damper.to_loads(),                     # 阻尼作用力
        ]
        # 计算 Kanes 方程
        self.kanes_method.kanes_equations(self.bodies, loads)

        # 验证质量矩阵
        assert self.kanes_method.mass_matrix == self.mass_matrix
        # 验证系统受力
        assert self.kanes_method.forcing == self.forcing
class TestTorqueActuator:

    @pytest.fixture(autouse=True)
    def _torque_actuator_fixture(self):
        # 定义符号 'T' 作为扭矩
        self.torque = Symbol('T')
        # 定义参考框架 'N' 和 'A'
        self.N = ReferenceFrame('N')
        self.A = ReferenceFrame('A')
        # 设置轴向为参考框架 'N' 的 z 轴
        self.axis = self.N.z
        # 创建一个名为 'target' 的刚体，其参考框架为 'N'
        self.target = RigidBody('target', frame=self.N)
        # 创建一个名为 'reaction' 的刚体，其参考框架为 'A'
        self.reaction = RigidBody('reaction', frame=self.A)

    def test_is_actuator_base_subclass(self):
        # 断言 TorqueActuator 是 ActuatorBase 的子类
        assert issubclass(TorqueActuator, ActuatorBase)

    @pytest.mark.parametrize(
        'torque',
        [
            Symbol('T'),                        # 使用符号 'T'
            dynamicsymbols('T'),                # 使用动态符号 'T'
            Symbol('T')**2 + Symbol('T'),       # 使用符号 'T' 的表达式
        ]
    )
    @pytest.mark.parametrize(
        'target_frame, reaction_frame',
        [
            (target.frame, reaction.frame),     # 使用目标和反作用框架的框架
            (target, reaction.frame),           # 使用目标的实体和反作用框架的框架
            (target.frame, reaction),           # 使用目标框架和反作用的实体框架
            (target, reaction),                 # 使用目标实体和反作用实体框架
        ]
    )
    def test_valid_constructor_with_reaction(
        self,
        torque,
        target_frame,
        reaction_frame,
    ):
        # 创建 TorqueActuator 实例，传入扭矩、轴向、目标框架和反作用框架
        instance = TorqueActuator(
            torque,
            self.axis,
            target_frame,
            reaction_frame,
        )
        # 断言 instance 是 TorqueActuator 的实例
        assert isinstance(instance, TorqueActuator)

        # 断言 instance 有 'torque' 属性，且其类型为 ExprType
        assert hasattr(instance, 'torque')
        assert isinstance(instance.torque, ExprType)
        assert instance.torque == torque  # 断言 instance 的扭矩与输入的扭矩相同

        # 断言 instance 有 'axis' 属性，且其类型为 Vector
        assert hasattr(instance, 'axis')
        assert isinstance(instance.axis, Vector)
        assert instance.axis == self.axis  # 断言 instance 的轴向与预设的轴向相同

        # 断言 instance 有 'target_frame' 属性，且其类型为 ReferenceFrame
        assert hasattr(instance, 'target_frame')
        assert isinstance(instance.target_frame, ReferenceFrame)
        assert instance.target_frame == target.frame  # 断言 instance 的目标框架与输入的目标框架相同

        # 断言 instance 有 'reaction_frame' 属性，且其类型为 ReferenceFrame
        assert hasattr(instance, 'reaction_frame')
        assert isinstance(instance.reaction_frame, ReferenceFrame)
        assert instance.reaction_frame == reaction.frame  # 断言 instance 的反作用框架与输入的反作用框架相同

    @pytest.mark.parametrize(
        'torque',
        [
            Symbol('T'),                        # 使用符号 'T'
            dynamicsymbols('T'),                # 使用动态符号 'T'
            Symbol('T')**2 + Symbol('T'),       # 使用符号 'T' 的表达式
        ]
    )
    @pytest.mark.parametrize('target_frame', [target.frame, target])
    def test_valid_constructor_without_reaction(self, torque, target_frame):
        # 创建 TorqueActuator 实例，传入扭矩、轴向和目标框架（无反作用框架）
        instance = TorqueActuator(torque, self.axis, target_frame)
        # 断言 instance 是 TorqueActuator 的实例
        assert isinstance(instance, TorqueActuator)

        # 断言 instance 有 'torque' 属性，且其类型为 ExprType
        assert hasattr(instance, 'torque')
        assert isinstance(instance.torque, ExprType)
        assert instance.torque == torque  # 断言 instance 的扭矩与输入的扭矩相同

        # 断言 instance 有 'axis' 属性，且其类型为 Vector
        assert hasattr(instance, 'axis')
        assert isinstance(instance.axis, Vector)
        assert instance.axis == self.axis  # 断言 instance 的轴向与预设的轴向相同

        # 断言 instance 有 'target_frame' 属性，且其类型为 ReferenceFrame
        assert hasattr(instance, 'target_frame')
        assert isinstance(instance.target_frame, ReferenceFrame)
        assert instance.target_frame == target.frame  # 断言 instance 的目标框架与输入的目标框架相同

        # 断言 instance 的 'reaction_frame' 属性为 None
        assert hasattr(instance, 'reaction_frame')
        assert instance.reaction_frame is None

    @pytest.mark.parametrize('torque', [None, 'T'])
    # 测试当给定的力矩不可符号化时，构造函数是否引发 SympifyError 异常
    def test_invalid_constructor_torque_not_sympifyable(self, torque):
        with pytest.raises(SympifyError):
            _ = TorqueActuator(torque, self.axis, self.target)

    # 使用参数化测试标记，分别测试传入符号 'a' 和动力学符号 'a' 时，构造函数是否引发 TypeError 异常
    def test_invalid_constructor_axis_not_vector(self, axis):
        with pytest.raises(TypeError):
            _ = TorqueActuator(self.torque, axis, self.target, self.reaction)

    # 使用参数化测试标记，分别测试传入的 frames 是否符合期望的类型，若不是则引发 TypeError 异常
    def test_invalid_constructor_frames_not_frame(self, frames):
        with pytest.raises(TypeError):
            _ = TorqueActuator(self.torque, self.axis, *frames)

    # 使用参数化测试标记，测试对象的属性是否是不可变的，尝试设置属性时是否引发 AttributeError 异常
    def test_properties_are_immutable(self, property_name, fixture_attr_name):
        # 创建 TorqueActuator 实例
        actuator = TorqueActuator(
            self.torque,
            self.axis,
            self.target,
            self.reaction,
        )
        # 获取测试对象的属性值
        value = getattr(self, fixture_attr_name)
        # 尝试设置属性并断言是否引发 AttributeError 异常
        with pytest.raises(AttributeError):
            setattr(actuator, property_name, value)

    # 测试在没有反作用力时，TorqueActuator 实例的字符串表示是否符合预期
    def test_repr_without_reaction(self):
        actuator = TorqueActuator(self.torque, self.axis, self.target)
        expected = 'TorqueActuator(T, axis=N.z, target_frame=N)'
        assert repr(actuator) == expected

    # 测试在有反作用力时，TorqueActuator 实例的字符串表示是否符合预期
    def test_repr_with_reaction(self):
        actuator = TorqueActuator(
            self.torque,
            self.axis,
            self.target,
            self.reaction,
        )
        expected = 'TorqueActuator(T, axis=N.z, target_frame=N, reaction_frame=A)'
        assert repr(actuator) == expected
    # 测试用例：在 PinJoint 构造函数中测试 TorqueActuator 的创建
    def test_at_pin_joint_constructor(self):
        # 创建 PinJoint 实例
        pin_joint = PinJoint(
            'pin',                           # 名称
            self.target,                      # 目标
            self.reaction,                    # 反作用力
            coordinates=dynamicsymbols('q'),  # 动态符号 q
            speeds=dynamicsymbols('u'),       # 动态符号 u
            parent_interframe=self.N,         # 父参考系
            joint_axis=self.axis,             # 关节轴
        )
        # 在 pin_joint 上应用 TorqueActuator，返回实例
        instance = TorqueActuator.at_pin_joint(self.torque, pin_joint)
        # 断言 instance 是 TorqueActuator 类的实例
        assert isinstance(instance, TorqueActuator)

        # 断言 instance 具有 torque 属性
        assert hasattr(instance, 'torque')
        # 断言 instance.torque 是 ExprType 类型的对象
        assert isinstance(instance.torque, ExprType)
        # 断言 instance.torque 等于预期的 self.torque
        assert instance.torque == self.torque

        # 断言 instance 具有 axis 属性
        assert hasattr(instance, 'axis')
        # 断言 instance.axis 是 Vector 类型的对象
        assert isinstance(instance.axis, Vector)
        # 断言 instance.axis 等于预期的 self.axis
        assert instance.axis == self.axis

        # 断言 instance 具有 target_frame 属性
        assert hasattr(instance, 'target_frame')
        # 断言 instance.target_frame 是 ReferenceFrame 类型的对象
        assert isinstance(instance.target_frame, ReferenceFrame)
        # 断言 instance.target_frame 等于预期的 self.A
        assert instance.target_frame == self.A

        # 断言 instance 具有 reaction_frame 属性
        assert hasattr(instance, 'reaction_frame')
        # 断言 instance.reaction_frame 是 ReferenceFrame 类型的对象
        assert isinstance(instance.reaction_frame, ReferenceFrame)
        # 断言 instance.reaction_frame 等于预期的 self.N

    # 测试用例：在不正确的 PinJoint 类型上应用 TorqueActuator
    def test_at_pin_joint_pin_joint_not_pin_joint_invalid(self):
        # 使用 pytest 检查是否引发 TypeError 异常
        with pytest.raises(TypeError):
            _ = TorqueActuator.at_pin_joint(self.torque, Symbol('pin'))

    # 测试用例：测试在没有反作用力时的 TorqueActuator.to_loads() 方法
    def test_to_loads_without_reaction(self):
        # 创建 TorqueActuator 实例
        actuator = TorqueActuator(self.torque, self.axis, self.target)
        # 预期的负载列表
        expected = [
            (self.N, self.torque * self.axis),
        ]
        # 断言调用 actuator 的 to_loads() 方法返回预期的负载列表
        assert actuator.to_loads() == expected

    # 测试用例：测试在有反作用力时的 TorqueActuator.to_loads() 方法
    def test_to_loads_with_reaction(self):
        # 创建 TorqueActuator 实例，包括反作用力
        actuator = TorqueActuator(
            self.torque,
            self.axis,
            self.target,
            self.reaction,
        )
        # 预期的负载列表
        expected = [
            (self.N, self.torque * self.axis),
            (self.A, - self.torque * self.axis),
        ]
        # 断言调用 actuator 的 to_loads() 方法返回预期的负载列表
        assert actuator.to_loads() == expected
class NonSympifyable:
    pass



class TestDuffingSpring:
    @pytest.fixture(autouse=True)
    # 设置用于多个测试中的公共变量
    def _duffing_spring_fixture(self):
        # 定义线性刚度符号对象
        self.linear_stiffness = Symbol('beta')
        # 定义非线性刚度符号对象
        self.nonlinear_stiffness = Symbol('alpha')
        # 定义平衡长度符号对象
        self.equilibrium_length = Symbol('l')
        # 定义点对象 pA
        self.pA = Point('pA')
        # 定义点对象 pB
        self.pB = Point('pB')
        # 创建线性路径对象，连接点 pA 和 pB
        self.pathway = LinearPathway(self.pA, self.pB)
        # 定义动力符号 q
        self.q = dynamicsymbols('q')
        # 创建参考系对象 N
        self.N = ReferenceFrame('N')

    # 检查 DuffingSpring 是否是 ForceActuator 的子类
    def test_is_force_actuator_subclass(self):
        assert issubclass(DuffingSpring, ForceActuator)

    # 检查 DuffingSpring 是否是 ActuatorBase 的子类
    def test_is_actuator_base_subclass(self):
        assert issubclass(DuffingSpring, ActuatorBase)

    @pytest.mark.parametrize(
    # 创建参数化测试，允许使用不同的参数运行相同的测试函数多次
    (
        'linear_stiffness,  '
        'expected_linear_stiffness,  '
        'nonlinear_stiffness,   '
        'expected_nonlinear_stiffness,  '
        'equilibrium_length,    '
        'expected_equilibrium_length,   '
        'force'
    ),
    [
            (
                1,
                S.One,
                1,
                S.One,
                0,
                S.Zero,
                -sqrt(dynamicsymbols('q')**2)-(sqrt(dynamicsymbols('q')**2))**3,
            ),
            (
                Symbol('beta'),
                Symbol('beta'),
                Symbol('alpha'),
                Symbol('alpha'),
                0,
                S.Zero,
                -Symbol('beta')*sqrt(dynamicsymbols('q')**2)-Symbol('alpha')*(sqrt(dynamicsymbols('q')**2))**3,
            ),
            (
                Symbol('beta'),
                Symbol('beta'),
                Symbol('alpha'),
                Symbol('alpha'),
                S.Zero,
                S.Zero,
                -Symbol('beta')*sqrt(dynamicsymbols('q')**2)-Symbol('alpha')*(sqrt(dynamicsymbols('q')**2))**3,
            ),
            (
                Symbol('beta'),
                Symbol('beta'),
                Symbol('alpha'),
                Symbol('alpha'),
                Symbol('l'),
                Symbol('l'),
                -Symbol('beta') * (sqrt(dynamicsymbols('q')**2) - Symbol('l')) - Symbol('alpha') * (sqrt(dynamicsymbols('q')**2) - Symbol('l'))**3,
            ),
        ]
    )

    # 检查 DuffingSpring 的构造函数是否正确初始化其属性
    # 测试不同组合的线性和非线性刚度、平衡长度，以及生成的力表达式
    def test_valid_constructor(
        self,
        linear_stiffness,
        expected_linear_stiffness,
        nonlinear_stiffness,
        expected_nonlinear_stiffness,
        equilibrium_length,
        expected_equilibrium_length,
        force,
    ):
        # 使用 self.pA 和 self.q*self.N.x 设置 self.pB 的位置
        self.pB.set_pos(self.pA, self.q*self.N.x)
        # 使用给定的参数创建 DuffingSpring 对象
        spring = DuffingSpring(linear_stiffness, nonlinear_stiffness, self.pathway, equilibrium_length)

        # 断言 spring 是 DuffingSpring 类的实例
        assert isinstance(spring, DuffingSpring)

        # 断言 spring 具有 'linear_stiffness' 属性
        assert hasattr(spring, 'linear_stiffness')
        # 断言 spring.linear_stiffness 是 ExprType 类型
        assert isinstance(spring.linear_stiffness, ExprType)
        # 断言 spring.linear_stiffness 等于 expected_linear_stiffness
        assert spring.linear_stiffness == expected_linear_stiffness

        # 断言 spring 具有 'nonlinear_stiffness' 属性
        assert hasattr(spring, 'nonlinear_stiffness')
        # 断言 spring.nonlinear_stiffness 是 ExprType 类型
        assert isinstance(spring.nonlinear_stiffness, ExprType)
        # 断言 spring.nonlinear_stiffness 等于 expected_nonlinear_stiffness
        assert spring.nonlinear_stiffness == expected_nonlinear_stiffness

        # 断言 spring 具有 'pathway' 属性
        assert hasattr(spring, 'pathway')
        # 断言 spring.pathway 是 LinearPathway 类型
        assert isinstance(spring.pathway, LinearPathway)
        # 断言 spring.pathway 等于 self.pathway
        assert spring.pathway == self.pathway

        # 断言 spring 具有 'equilibrium_length' 属性
        assert hasattr(spring, 'equilibrium_length')
        # 断言 spring.equilibrium_length 是 ExprType 类型
        assert isinstance(spring.equilibrium_length, ExprType)
        # 断言 spring.equilibrium_length 等于 expected_equilibrium_length
        assert spring.equilibrium_length == expected_equilibrium_length

        # 断言 spring 具有 'force' 属性
        assert hasattr(spring, 'force')
        # 断言 spring.force 是 ExprType 类型
        assert isinstance(spring.force, ExprType)
        # 断言 spring.force 等于 force

    @pytest.mark.parametrize('linear_stiffness', [None, NonSympifyable()])
    def test_invalid_constructor_linear_stiffness_not_sympifyable(self, linear_stiffness):
        # 当 linear_stiffness 不可 sympify 时，应该抛出 SympifyError 异常
        with pytest.raises(SympifyError):
            _ = DuffingSpring(linear_stiffness, self.nonlinear_stiffness, self.pathway, self.equilibrium_length)

    @pytest.mark.parametrize('nonlinear_stiffness', [None, NonSympifyable()])
    def test_invalid_constructor_nonlinear_stiffness_not_sympifyable(self, nonlinear_stiffness):
        # 当 nonlinear_stiffness 不可 sympify 时，应该抛出 SympifyError 异常
        with pytest.raises(SympifyError):
            _ = DuffingSpring(self.linear_stiffness, nonlinear_stiffness, self.pathway, self.equilibrium_length)

    def test_invalid_constructor_pathway_not_pathway_base(self):
        # 当 pathway 不是 PathwayBase 类型时，应该抛出 TypeError 异常
        with pytest.raises(TypeError):
            _ = DuffingSpring(self.linear_stiffness, self.nonlinear_stiffness, NonSympifyable(), self.equilibrium_length)

    @pytest.mark.parametrize('equilibrium_length', [None, NonSympifyable()])
    def test_invalid_constructor_equilibrium_length_not_sympifyable(self, equilibrium_length):
        # 当 equilibrium_length 不可 sympify 时，应该抛出 SympifyError 异常
        with pytest.raises(SympifyError):
            _ = DuffingSpring(self.linear_stiffness, self.nonlinear_stiffness, self.pathway, equilibrium_length)

    @pytest.mark.parametrize(
        'property_name, fixture_attr_name',
        [
            ('linear_stiffness', 'linear_stiffness'),
            ('nonlinear_stiffness', 'nonlinear_stiffness'),
            ('pathway', 'pathway'),
            ('equilibrium_length', 'equilibrium_length')
        ]
    )
    # 检查 DuffingSpring 对象的特定属性在初始化后是否不可变
    # 确保一旦创建了 DuffingSpring 对象，其关键属性不能被修改
    # 定义一个测试方法，用于验证属性是不可变的
    # property_name: 要设置的属性名称
    # fixture_attr_name: 用于设置属性的固定属性名称
    def test_properties_are_immutable(self, property_name, fixture_attr_name):
        # 创建一个 DuffingSpring 实例，使用给定的弹簧参数
        spring = DuffingSpring(self.linear_stiffness, self.nonlinear_stiffness, self.pathway, self.equilibrium_length)
        # 使用 pytest 检查是否会引发 AttributeError 异常
        with pytest.raises(AttributeError):
            # 尝试设置对象的属性，期望引发异常
            setattr(spring, property_name, getattr(self, fixture_attr_name))

    # 使用参数化装饰器标记的测试方法
    # equilibrium_length: 平衡长度
    # expected: 预期的字符串表示形式
    @pytest.mark.parametrize(
        'equilibrium_length, expected',
        [
            (0, 'DuffingSpring(beta, alpha, LinearPathway(pA, pB), equilibrium_length=0)'),
            (Symbol('l'), 'DuffingSpring(beta, alpha, LinearPathway(pA, pB), equilibrium_length=l)'),
        ]
    )
    # 检查 DuffingSpring 类的 __repr__ 方法
    # 验证 DuffingSpring 实例的实际字符串表示形式是否与预期字符串匹配
    def test_repr(self, equilibrium_length, expected):
        # 创建一个 DuffingSpring 实例，使用给定的弹簧参数和平衡长度
        spring = DuffingSpring(self.linear_stiffness, self.nonlinear_stiffness, self.pathway, equilibrium_length)
        # 断言实例的字符串表示形式是否与预期字符串相等
        assert repr(spring) == expected

    # 测试 DuffingSpring 类的 to_loads 方法
    def test_to_loads(self):
        # 设置点 pB 相对于点 pA 的位置
        self.pB.set_pos(self.pA, self.q*self.N.x)
        # 创建一个 DuffingSpring 实例，使用给定的弹簧参数和平衡长度
        spring = DuffingSpring(self.linear_stiffness, self.nonlinear_stiffness, self.pathway, self.equilibrium_length)

        # 计算弹簧相对于平衡长度的位移
        displacement = self.q - self.equilibrium_length

        # 计算弹簧作用的力，包括线性和非线性部分
        force = -self.linear_stiffness * displacement - self.nonlinear_stiffness * displacement**3

        # 预期作用在点 pA 和 pB 上的力
        expected_loads = [Force(self.pA, force * self.N.x), Force(self.pB, -force * self.N.x)]

        # 获取 DuffingSpring.to_loads() 方法返回的力
        calculated_loads = spring.to_loads()

        # 比较预期的力和计算得到的力
        for calculated, expected in zip(calculated_loads, expected_loads):
            # 检查每个计算得到的力和预期的力在点上的匹配性
            assert calculated.point == expected.point
            # 对于参考系 self.N 中的每个维度
            for dim in self.N:
                # 计算力在该维度上的分量
                calculated_component = calculated.vector.dot(dim)
                expected_component = expected.vector.dot(dim)
                # 定义替换字典，将所有符号替换为数值
                substitutions = {self.q: 1, Symbol('l'): 1, Symbol('alpha'): 1, Symbol('beta'): 1}  # 需要添加其他必要的符号
                # 计算分量之间的差异
                diff = (calculated_component - expected_component).subs(substitutions).evalf()
                # 检查差异的绝对值是否小于阈值
                assert Abs(diff) < 1e-9, f"The forces do not match. Difference: {diff}"
```
# `D:\src\scipysrc\sympy\sympy\physics\biomechanics\tests\test_musculotendon.py`

```
# 导入必要的模块和库

"""Tests for the ``sympy.physics.biomechanics.musculotendon.py`` module."""
# 导入测试文件模块的注释

import abc
# 导入抽象基类模块

import pytest
# 导入 pytest 测试框架模块

from sympy.core.expr import UnevaluatedExpr
# 从 sympy 核心表达式模块导入 UnevaluatedExpr 类

from sympy.core.numbers import Float, Integer, Rational
# 从 sympy 核心数字模块导入 Float, Integer, Rational 类

from sympy.core.symbol import Symbol
# 从 sympy 核心符号模块导入 Symbol 类

from sympy.functions.elementary.exponential import exp
# 从 sympy 元素指数函数模块导入 exp 函数

from sympy.functions.elementary.hyperbolic import tanh
# 从 sympy 元素双曲函数模块导入 tanh 函数

from sympy.functions.elementary.miscellaneous import sqrt
# 从 sympy 元素杂项函数模块导入 sqrt 函数

from sympy.functions.elementary.trigonometric import sin
# 从 sympy 元素三角函数模块导入 sin 函数

from sympy.matrices.dense import MutableDenseMatrix as Matrix, eye, zeros
# 从 sympy 密集矩阵模块导入 MutableDenseMatrix 类别名为 Matrix, 以及 eye, zeros 函数

from sympy.physics.biomechanics.activation import (
    FirstOrderActivationDeGroote2016
)
# 从 sympy 生物力学激活模块导入 FirstOrderActivationDeGroote2016 类

from sympy.physics.biomechanics.curve import (
    CharacteristicCurveCollection,
    FiberForceLengthActiveDeGroote2016,
    FiberForceLengthPassiveDeGroote2016,
    FiberForceLengthPassiveInverseDeGroote2016,
    FiberForceVelocityDeGroote2016,
    FiberForceVelocityInverseDeGroote2016,
    TendonForceLengthDeGroote2016,
    TendonForceLengthInverseDeGroote2016,
)
# 从 sympy 生物力学曲线模块导入各种力长度关系类

from sympy.physics.biomechanics.musculotendon import (
    MusculotendonBase,
    MusculotendonDeGroote2016,
    MusculotendonFormulation,
)
# 从 sympy 生物力学肌腱模块导入 MusculotendonBase, MusculotendonDeGroote2016, MusculotendonFormulation 类

from sympy.physics.biomechanics._mixin import _NamedMixin
# 从 sympy 生物力学混合模块导入 _NamedMixin 类

from sympy.physics.mechanics.actuator import ForceActuator
# 从 sympy 力学执行器模块导入 ForceActuator 类

from sympy.physics.mechanics.pathway import LinearPathway
# 从 sympy 力学路径模块导入 LinearPathway 类

from sympy.physics.vector.frame import ReferenceFrame
# 从 sympy 向量框架模块导入 ReferenceFrame 类

from sympy.physics.vector.functions import dynamicsymbols
# 从 sympy 向量函数模块导入 dynamicsymbols 函数

from sympy.physics.vector.point import Point
# 从 sympy 向量点模块导入 Point 类

from sympy.simplify.simplify import simplify
# 从 sympy 简化模块导入 simplify 函数


class TestMusculotendonFormulation:
    @staticmethod
    def test_rigid_tendon_member():
        assert MusculotendonFormulation(0) == 0
        assert MusculotendonFormulation.RIGID_TENDON == 0

    @staticmethod
    def test_fiber_length_explicit_member():
        assert MusculotendonFormulation(1) == 1
        assert MusculotendonFormulation.FIBER_LENGTH_EXPLICIT == 1

    @staticmethod
    def test_tendon_force_explicit_member():
        assert MusculotendonFormulation(2) == 2
        assert MusculotendonFormulation.TENDON_FORCE_EXPLICIT == 2

    @staticmethod
    def test_fiber_length_implicit_member():
        assert MusculotendonFormulation(3) == 3
        assert MusculotendonFormulation.FIBER_LENGTH_IMPLICIT == 3

    @staticmethod
    def test_tendon_force_implicit_member():
        assert MusculotendonFormulation(4) == 4
        assert MusculotendonFormulation.TENDON_FORCE_IMPLICIT == 4


class TestMusculotendonBase:

    @staticmethod
    def test_is_abstract_base_class():
        assert issubclass(MusculotendonBase, abc.ABC)
        # 检查 MusculotendonBase 是否是 abc.ABC 的子类

    @staticmethod
    def test_class():
        assert issubclass(MusculotendonBase, ForceActuator)
        # 检查 MusculotendonBase 是否是 ForceActuator 的子类
        assert issubclass(MusculotendonBase, _NamedMixin)
        # 检查 MusculotendonBase 是否是 _NamedMixin 的子类
        assert MusculotendonBase.__name__ == 'MusculotendonBase'
        # 检查 MusculotendonBase 的类名是否为 'MusculotendonBase'

    @staticmethod
    def test_cannot_instantiate_directly():
        with pytest.raises(TypeError):
            _ = MusculotendonBase()
        # 检查 MusculotendonBase 是否不能直接实例化，应该抛出 TypeError 异常
# 使用 pytest.mark.parametrize 装饰器，为测试类参数化，参数为 [MusculotendonDeGroote2016]
@pytest.mark.parametrize('musculotendon_concrete', [MusculotendonDeGroote2016])
class TestMusculotendonRigidTendon:

    # 使用 pytest.fixture 装饰器，自动使用该 fixture
    @pytest.fixture(autouse=True)
    def _musculotendon_rigid_tendon_fixture(self, musculotendon_concrete):
        # 设置实例变量
        self.name = 'name'  # 设置名称
        self.N = ReferenceFrame('N')  # 创建参考系
        self.q = dynamicsymbols('q')  # 创建动力学符号
        self.origin = Point('pO')  # 创建起点
        self.insertion = Point('pI')  # 创建插入点
        self.insertion.set_pos(self.origin, self.q*self.N.x)  # 设置插入点位置
        self.pathway = LinearPathway(self.origin, self.insertion)  # 创建线性路径
        self.activation = FirstOrderActivationDeGroote2016(self.name)  # 创建激活对象
        self.e = self.activation.excitation  # 获取激活信号
        self.a = self.activation.activation  # 获取激活水平
        self.tau_a = self.activation.activation_time_constant  # 获取激活时间常数
        self.tau_d = self.activation.deactivation_time_constant  # 获取去激活时间常数
        self.b = self.activation.smoothing_rate  # 获取平滑率
        self.formulation = MusculotendonFormulation.RIGID_TENDON  # 设置肌腱公式类型
        self.l_T_slack = Symbol('l_T_slack')  # 设置肌腱松弛长度符号
        self.F_M_max = Symbol('F_M_max')  # 设置肌腱最大等长力符号
        self.l_M_opt = Symbol('l_M_opt')  # 设置肌肉最佳纤维长度符号
        self.v_M_max = Symbol('v_M_max')  # 设置最大肌纤维速度符号
        self.alpha_opt = Symbol('alpha_opt')  # 设置最佳肌肉纤维角度符号
        self.beta = Symbol('beta')  # 设置纤维阻尼系数符号
        # 实例化具体肌腱模型对象，使用给定参数
        self.instance = musculotendon_concrete(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=self.formulation,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
        )
        # 计算激活信号导数表达式
        self.da_expr = (
            (1/(self.tau_a*(Rational(1, 2) + Rational(3, 2)*self.a)))
            *(Rational(1, 2) + Rational(1, 2)*tanh(self.b*(self.e - self.a)))
            + ((Rational(1, 2) + Rational(3, 2)*self.a)/self.tau_d)
            *(Rational(1, 2) - Rational(1, 2)*tanh(self.b*(self.e - self.a)))
        )*(self.e - self.a)

    # 测试状态变量方法
    def test_state_vars(self):
        # 断言实例对象具有属性 'x'
        assert hasattr(self.instance, 'x')
        # 断言实例对象具有属性 'state_vars'
        assert hasattr(self.instance, 'state_vars')
        # 断言实例对象的 'x' 属性与 'state_vars' 属性相同
        assert self.instance.x == self.instance.state_vars
        # 预期的 'x' 值为单行矩阵 [self.a]
        x_expected = Matrix([self.a])
        assert self.instance.x == x_expected  # 断言实例对象的 'x' 与预期相同
        assert self.instance.state_vars == x_expected  # 断言实例对象的 'state_vars' 与预期相同
        assert isinstance(self.instance.x, Matrix)  # 断言实例对象的 'x' 是 Matrix 类型
        assert isinstance(self.instance.state_vars, Matrix)  # 断言实例对象的 'state_vars' 是 Matrix 类型
        assert self.instance.x.shape == (1, 1)  # 断言实例对象的 'x' 形状为 (1, 1)
        assert self.instance.state_vars.shape == (1, 1)  # 断言实例对象的 'state_vars' 形状为 (1, 1)
    # 测试输入变量是否正确设置
    def test_input_vars(self):
        # 断言实例对象是否有属性'r'
        assert hasattr(self.instance, 'r')
        # 断言实例对象是否有属性'input_vars'
        assert hasattr(self.instance, 'input_vars')
        # 断言'r'属性与'input_vars'属性的值相等
        assert self.instance.r == self.instance.input_vars
        # 创建预期的矩阵r_expected，包含self.e作为唯一元素
        r_expected = Matrix([self.e])
        # 断言'r'属性与预期的r_expected矩阵相等
        assert self.instance.r == r_expected
        # 断言'input_vars'属性与预期的r_expected矩阵相等
        assert self.instance.input_vars == r_expected
        # 断言'r'属性确实是Matrix类型
        assert isinstance(self.instance.r, Matrix)
        # 断言'input_vars'属性确实是Matrix类型
        assert isinstance(self.instance.input_vars, Matrix)
        # 断言'r'属性的形状为(1, 1)
        assert self.instance.r.shape == (1, 1)
        # 断言'input_vars'属性的形状为(1, 1)
        assert self.instance.input_vars.shape == (1, 1)

    # 测试常量变量是否正确设置
    def test_constants(self):
        # 断言实例对象是否有属性'p'
        assert hasattr(self.instance, 'p')
        # 断言实例对象是否有属性'constants'
        assert hasattr(self.instance, 'constants')
        # 断言'p'属性与'constants'属性的值相等
        assert self.instance.p == self.instance.constants
        # 创建预期的矩阵p_expected，包含一系列Symbol和self.l_T_slack等作为元素
        p_expected = Matrix(
            [
                self.l_T_slack,
                self.F_M_max,
                self.l_M_opt,
                self.v_M_max,
                self.alpha_opt,
                self.beta,
                self.tau_a,
                self.tau_d,
                self.b,
                Symbol('c_0_fl_T_name'),
                Symbol('c_1_fl_T_name'),
                Symbol('c_2_fl_T_name'),
                Symbol('c_3_fl_T_name'),
                Symbol('c_0_fl_M_pas_name'),
                Symbol('c_1_fl_M_pas_name'),
                Symbol('c_0_fl_M_act_name'),
                Symbol('c_1_fl_M_act_name'),
                Symbol('c_2_fl_M_act_name'),
                Symbol('c_3_fl_M_act_name'),
                Symbol('c_4_fl_M_act_name'),
                Symbol('c_5_fl_M_act_name'),
                Symbol('c_6_fl_M_act_name'),
                Symbol('c_7_fl_M_act_name'),
                Symbol('c_8_fl_M_act_name'),
                Symbol('c_9_fl_M_act_name'),
                Symbol('c_10_fl_M_act_name'),
                Symbol('c_11_fl_M_act_name'),
                Symbol('c_0_fv_M_name'),
                Symbol('c_1_fv_M_name'),
                Symbol('c_2_fv_M_name'),
                Symbol('c_3_fv_M_name'),
            ]
        )
        # 断言'p'属性与预期的p_expected矩阵相等
        assert self.instance.p == p_expected
        # 断言'constants'属性与预期的p_expected矩阵相等
        assert self.instance.constants == p_expected
        # 断言'p'属性确实是Matrix类型
        assert isinstance(self.instance.p, Matrix)
        # 断言'constants'属性确实是Matrix类型
        assert isinstance(self.instance.constants, Matrix)
        # 断言'p'属性的形状为(31, 1)
        assert self.instance.p.shape == (31, 1)
        # 断言'constants'属性的形状为(31, 1)
        assert self.instance.constants.shape == (31, 1)

    # 测试矩阵M是否正确设置
    def test_M(self):
        # 断言实例对象是否有属性'M'
        assert hasattr(self.instance, 'M')
        # 创建预期的矩阵M_expected，包含1作为唯一元素
        M_expected = Matrix([1])
        # 断言'M'属性与预期的M_expected矩阵相等
        assert self.instance.M == M_expected
        # 断言'M'属性确实是Matrix类型
        assert isinstance(self.instance.M, Matrix)
        # 断言'M'属性的形状为(1, 1)
        assert self.instance.M.shape == (1, 1)

    # 测试矩阵F是否正确设置
    def test_F(self):
        # 断言实例对象是否有属性'F'
        assert hasattr(self.instance, 'F')
        # 创建预期的矩阵F_expected，包含self.da_expr作为唯一元素
        F_expected = Matrix([self.da_expr])
        # 断言'F'属性与预期的F_expected矩阵相等
        assert self.instance.F == F_expected
        # 断言'F'属性确实是Matrix类型
        assert isinstance(self.instance.F, Matrix)
        # 断言'F'属性的形状为(1, 1)
        assert self.instance.F.shape == (1, 1)
    # 定义测试方法 test_rhs，用于测试某个对象的 rhs 方法的正确性
    def test_rhs(self):
        # 断言对象实例具有 'rhs' 属性
        assert hasattr(self.instance, 'rhs')
        # 生成一个预期的右手边矩阵，内容为 self.da_expr
        rhs_expected = Matrix([self.da_expr])
        # 调用被测试对象的 rhs 方法，获取实际输出结果
        rhs = self.instance.rhs()
        # 断言输出结果 rhs 是 Matrix 类型
        assert isinstance(rhs, Matrix)
        # 断言输出结果 rhs 的形状为 (1, 1)
        assert rhs.shape == (1, 1)
        # 断言简化后的 rhs 与预期结果 rhs_expected 相等
        assert simplify(rhs - rhs_expected) == zeros(1)
@pytest.mark.parametrize(
    'musculotendon_concrete, curve',
    [  # 使用 pytest 的参数化标记，定义测试参数
        (
            MusculotendonDeGroote2016,  # musculotendon_concrete 参数为 MusculotendonDeGroote2016 类
            CharacteristicCurveCollection(  # curve 参数为 CharacteristicCurveCollection 类的实例，包含多个特征曲线对象
                tendon_force_length=TendonForceLengthDeGroote2016,  # 肌腱力长度特征曲线
                tendon_force_length_inverse=TendonForceLengthInverseDeGroote2016,  # 肌腱力长度逆特征曲线
                fiber_force_length_passive=FiberForceLengthPassiveDeGroote2016,  # 被动肌纤维力长度特征曲线
                fiber_force_length_passive_inverse=FiberForceLengthPassiveInverseDeGroote2016,  # 被动肌纤维力长度逆特征曲线
                fiber_force_length_active=FiberForceLengthActiveDeGroote2016,  # 主动肌纤维力长度特征曲线
                fiber_force_velocity=FiberForceVelocityDeGroote2016,  # 肌纤维力-速度特征曲线
                fiber_force_velocity_inverse=FiberForceVelocityInverseDeGroote2016,  # 肌纤维力-速度逆特征曲线
            ),
        )
    ],
)
class TestFiberLengthExplicit:  # 定义测试类 TestFiberLengthExplicit

    @pytest.fixture(autouse=True)
    def _musculotendon_fiber_length_explicit_fixture(  # 定义用于测试的装置函数
        self,
        musculotendon_concrete,  # musculotendon_concrete 参数
        curve,  # curve 参数
        ):
            # 设定对象名称为 'name'
            self.name = 'name'
            # 创建参照坐标系 N
            self.N = ReferenceFrame('N')
            # 创建广义坐标 q
            self.q = dynamicsymbols('q')
            # 创建起始点 pO 和插入点 pI
            self.origin = Point('pO')
            self.insertion = Point('pI')
            # 设置插入点相对于起始点的位置，根据广义坐标 q
            self.insertion.set_pos(self.origin, self.q*self.N.x)
            # 创建线性路径，连接起始点和插入点
            self.pathway = LinearPathway(self.origin, self.insertion)
            # 使用 DeGroote2016 模型创建第一阶段激活器，名称为 self.name
            self.activation = FirstOrderActivationDeGroote2016(self.name)
            # 获取激活和激励值
            self.e = self.activation.excitation
            self.a = self.activation.activation
            # 获取激活时间常数和去激活时间常数
            self.tau_a = self.activation.activation_time_constant
            self.tau_d = self.activation.deactivation_time_constant
            # 获取平滑率
            self.b = self.activation.smoothing_rate
            # 设定肌腱模型的形式为 FIBER_LENGTH_EXPLICIT
            self.formulation = MusculotendonFormulation.FIBER_LENGTH_EXPLICIT
            # 创建松弛长度、最大等长力、最佳纤维长度、最大纤维速度、最佳肌腱角度和纤维阻尼系数的符号
            self.l_T_slack = Symbol('l_T_slack')
            self.F_M_max = Symbol('F_M_max')
            self.l_M_opt = Symbol('l_M_opt')
            self.v_M_max = Symbol('v_M_max')
            self.alpha_opt = Symbol('alpha_opt')
            self.beta = Symbol('beta')
            # 创建肌腱实例，使用默认设置
            self.instance = musculotendon_concrete(
                self.name,
                self.pathway,
                self.activation,
                musculotendon_dynamics=self.formulation,
                tendon_slack_length=self.l_T_slack,
                peak_isometric_force=self.F_M_max,
                optimal_fiber_length=self.l_M_opt,
                maximal_fiber_velocity=self.v_M_max,
                optimal_pennation_angle=self.alpha_opt,
                fiber_damping_coefficient=self.beta,
                with_defaults=True,
            )
            # 创建 l_M_tilde 动态符号
            self.l_M_tilde = dynamicsymbols('l_M_tilde_name')
            # 获取肌腱路径长度和最佳纤维长度
            l_MT = self.pathway.length
            l_M = self.l_M_tilde*self.l_M_opt
            # 计算肌腱长度和肌纤维长度
            l_T = l_MT - sqrt(l_M**2 - (self.l_M_opt*sin(self.alpha_opt))**2)
            # 获取肌腱力-长度曲线和肌纤维被动力-长度曲线
            fl_T = curve.tendon_force_length.with_defaults(l_T/self.l_T_slack)
            fl_M_pas = curve.fiber_force_length_passive.with_defaults(self.l_M_tilde)
            fl_M_act = curve.fiber_force_length_active.with_defaults(self.l_M_tilde)
            # 获取肌纤维速度反函数
            v_M_tilde = curve.fiber_force_velocity_inverse.with_defaults(
                ((((fl_T*self.F_M_max)/((l_MT - l_T)/l_M))/self.F_M_max) - fl_M_pas)
                /(self.a*fl_M_act)
            )
            # 计算 dl_M_tilde 表达式
            self.dl_M_tilde_expr = (self.v_M_max/self.l_M_opt)*v_M_tilde
            # 计算 da 表达式
            self.da_expr = (
                (1/(self.tau_a*(Rational(1, 2) + Rational(3, 2)*self.a)))
                *(Rational(1, 2) + Rational(1, 2)*tanh(self.b*(self.e - self.a)))
                + ((Rational(1, 2) + Rational(3, 2)*self.a)/self.tau_d)
                *(Rational(1, 2) - Rational(1, 2)*tanh(self.b*(self.e - self.a)))
            )*(self.e - self.a)
    # 测试状态变量是否存在实例中
    def test_state_vars(self):
        assert hasattr(self.instance, 'x')
        # 检查实例中是否有状态变量的属性
        assert hasattr(self.instance, 'state_vars')
        # 断言状态变量 x 和 state_vars 是相同的对象
        assert self.instance.x == self.instance.state_vars
        # 创建预期的状态变量 x，由两个矩阵元素组成
        x_expected = Matrix([self.l_M_tilde, self.a])
        # 断言实例中的 x 和预期的 x 相同
        assert self.instance.x == x_expected
        # 断言实例中的 state_vars 和预期的 x 相同
        assert self.instance.state_vars == x_expected
        # 断言实例中的 x 是 Matrix 类型
        assert isinstance(self.instance.x, Matrix)
        # 断言实例中的 state_vars 是 Matrix 类型
        assert isinstance(self.instance.state_vars, Matrix)
        # 断言实例中的 x 的形状是 (2, 1)
        assert self.instance.x.shape == (2, 1)
        # 断言实例中的 state_vars 的形状是 (2, 1)
        assert self.instance.state_vars.shape == (2, 1)

    # 测试输入变量是否存在实例中
    def test_input_vars(self):
        assert hasattr(self.instance, 'r')
        # 检查实例中是否有输入变量的属性
        assert hasattr(self.instance, 'input_vars')
        # 断言输入变量 r 和 input_vars 是相同的对象
        assert self.instance.r == self.instance.input_vars
        # 创建预期的输入变量 r，由一个矩阵元素组成
        r_expected = Matrix([self.e])
        # 断言实例中的 r 和预期的 r 相同
        assert self.instance.r == r_expected
        # 断言实例中的 input_vars 和预期的 r 相同
        assert self.instance.input_vars == r_expected
        # 断言实例中的 r 是 Matrix 类型
        assert isinstance(self.instance.r, Matrix)
        # 断言实例中的 input_vars 是 Matrix 类型
        assert isinstance(self.instance.input_vars, Matrix)
        # 断言实例中的 r 的形状是 (1, 1)
        assert self.instance.r.shape == (1, 1)
        # 断言实例中的 input_vars 的形状是 (1, 1)
        assert self.instance.input_vars.shape == (1, 1)

    # 测试常量是否存在实例中
    def test_constants(self):
        assert hasattr(self.instance, 'p')
        # 检查实例中是否有常量的属性
        assert hasattr(self.instance, 'constants')
        # 断言常量 p 和 constants 是相同的对象
        assert self.instance.p == self.instance.constants
        # 创建预期的常量 p，由一个包含多个元素的矩阵组成
        p_expected = Matrix(
            [
                self.l_T_slack,
                self.F_M_max,
                self.l_M_opt,
                self.v_M_max,
                self.alpha_opt,
                self.beta,
                self.tau_a,
                self.tau_d,
                self.b,
            ]
        )
        # 断言实例中的 p 和预期的 p 相同
        assert self.instance.p == p_expected
        # 断言实例中的 constants 和预期的 p 相同
        assert self.instance.constants == p_expected
        # 断言实例中的 p 是 Matrix 类型
        assert isinstance(self.instance.p, Matrix)
        # 断言实例中的 constants 是 Matrix 类型
        assert isinstance(self.instance.constants, Matrix)
        # 断言实例中的 p 的形状是 (9, 1)
        assert self.instance.p.shape == (9, 1)
        # 断言实例中的 constants 的形状是 (9, 1)
        assert self.instance.constants.shape == (9, 1)

    # 测试矩阵 M 是否存在实例中
    def test_M(self):
        assert hasattr(self.instance, 'M')
        # 创建预期的矩阵 M，为一个单位矩阵
        M_expected = eye(2)
        # 断言实例中的 M 和预期的 M 相同
        assert self.instance.M == M_expected
        # 断言实例中的 M 是 Matrix 类型
        assert isinstance(self.instance.M, Matrix)
        # 断言实例中的 M 的形状是 (2, 2)
        assert self.instance.M.shape == (2, 2)

    # 测试向量 F 是否存在实例中
    def test_F(self):
        assert hasattr(self.instance, 'F')
        # 创建预期的向量 F，由两个矩阵元素组成
        F_expected = Matrix([self.dl_M_tilde_expr, self.da_expr])
        # 断言实例中的 F 和预期的 F 相同
        assert self.instance.F == F_expected
        # 断言实例中的 F 是 Matrix 类型
        assert isinstance(self.instance.F, Matrix)
        # 断言实例中的 F 的形状是 (2, 1)
        assert self.instance.F.shape == (2, 1)

    # 测试右手边向量 rhs 是否存在实例中，并验证其计算结果
    def test_rhs(self):
        assert hasattr(self.instance, 'rhs')
        # 创建预期的右手边向量 rhs，由两个矩阵元素组成
        rhs_expected = Matrix([self.dl_M_tilde_expr, self.da_expr])
        # 计算实例中的 rhs
        rhs = self.instance.rhs()
        # 断言计算出的 rhs 是 Matrix 类型
        assert isinstance(rhs, Matrix)
        # 断言计算出的 rhs 的形状是 (2, 1)
        assert rhs.shape == (2, 1)
        # 使用简化函数检查计算出的 rhs 与预期的 rhs 是否为零
        assert simplify(rhs - rhs_expected) == zeros(2, 1)
# 使用 pytest.mark.parametrize 装饰器为测试用例提供参数化输入，参数包括 'musculotendon_concrete' 和 'curve'
@pytest.mark.parametrize(
    'musculotendon_concrete, curve',
    [
        # 参数化测试用例的具体参数
        (
            MusculotendonDeGroote2016,  # musculotendon_concrete 参数设置为 MusculotendonDeGroote2016 类
            CharacteristicCurveCollection(  # curve 参数设置为 CharacteristicCurveCollection 实例
                tendon_force_length=TendonForceLengthDeGroote2016,  # 指定特征曲线集中的腱力-长度特性
                tendon_force_length_inverse=TendonForceLengthInverseDeGroote2016,  # 腱力-长度反向特性
                fiber_force_length_passive=FiberForceLengthPassiveDeGroote2016,  # 被动肌纤维力-长度特性
                fiber_force_length_passive_inverse=FiberForceLengthPassiveInverseDeGroote2016,  # 被动肌纤维力-长度反向特性
                fiber_force_length_active=FiberForceLengthActiveDeGroote2016,  # 主动肌纤维力-长度特性
                fiber_force_velocity=FiberForceVelocityDeGroote2016,  # 肌纤维力-速度特性
                fiber_force_velocity_inverse=FiberForceVelocityInverseDeGroote2016,  # 肌纤维力-速度反向特性
            ),
        )
    ],
)
# 定义测试类 TestTendonForceExplicit
class TestTendonForceExplicit:

    # 使用 pytest.fixture 装饰器，autouse=True 表示此 fixture 将自动应用于所有测试方法
    def _musculotendon_tendon_force_explicit_fixture(
        self,
        musculotendon_concrete,  # musculotendon_concrete 参数作为 fixture 输入
        curve,  # curve 参数作为 fixture 输入
    ):
        # 此 fixture 方法没有实际代码，注释中未提供进一步详细的操作说明
        pass  # 占位符，表示此处未实现具体代码逻辑
        ):
        # 设置对象的名称属性为固定字符串 'name'
        self.name = 'name'
        # 创建参考坐标系 N
        self.N = ReferenceFrame('N')
        # 定义一个动力学符号 q
        self.q = dynamicsymbols('q')
        # 创建名为 'pO' 的点对象作为起点
        self.origin = Point('pO')
        # 创建名为 'pI' 的点对象作为插入点
        self.insertion = Point('pI')
        # 将插入点相对于起点设置在 N.x 方向上的位置为 q
        self.insertion.set_pos(self.origin, self.q*self.N.x)
        # 创建起点到插入点的线性路径
        self.pathway = LinearPathway(self.origin, self.insertion)
        # 使用 DeGroote 2016 模型创建第一阶段激活对象
        self.activation = FirstOrderActivationDeGroote2016(self.name)
        # 获取激活值 e 和激活力 a
        self.e = self.activation.excitation
        self.a = self.activation.activation
        # 获取激活和去激活的时间常数
        self.tau_a = self.activation.activation_time_constant
        self.tau_d = self.activation.deactivation_time_constant
        # 获取平滑率参数 b
        self.b = self.activation.smoothing_rate
        # 设置肌腱力的表达方式为显式计算
        self.formulation = MusculotendonFormulation.TENDON_FORCE_EXPLICIT
        # 定义肌腱松弛长度的符号
        self.l_T_slack = Symbol('l_T_slack')
        # 定义肌腱最大等长力的符号
        self.F_M_max = Symbol('F_M_max')
        # 定义肌肉最佳长度的符号
        self.l_M_opt = Symbol('l_M_opt')
        # 定义肌肉最大速度的符号
        self.v_M_max = Symbol('v_M_max')
        # 定义肌肉最佳切线角的符号
        self.alpha_opt = Symbol('alpha_opt')
        # 定义 beta 符号作为肌腱阻尼系数
        self.beta = Symbol('beta')
        # 使用具体的参数创建肌肉-肌腱实例
        self.instance = musculotendon_concrete(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=self.formulation,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
            with_defaults=True,
        )
        # 创建肌腱力的动力学符号
        self.F_T_tilde = dynamicsymbols('F_T_tilde_name')
        # 使用默认参数计算肌腱力的长度
        l_T_tilde = curve.tendon_force_length_inverse.with_defaults(self.F_T_tilde)
        # 获取线性路径的长度和扩展速度
        l_MT = self.pathway.length
        v_MT = self.pathway.extension_velocity
        # 计算实际肌腱长度和肌肉长度
        l_T = l_T_tilde*self.l_T_slack
        l_M = sqrt((l_MT - l_T)**2 + (self.l_M_opt*sin(self.alpha_opt))**2)
        # 计算归一化的肌肉长度
        l_M_tilde = l_M/self.l_M_opt
        # 计算肌肉力的余弦值
        cos_alpha = (l_MT - l_T)/l_M
        # 计算肌腱力和肌肉力
        F_T = self.F_T_tilde*self.F_M_max
        F_M = F_T/cos_alpha
        # 计算归一化的肌肉和肌腱力
        F_M_tilde = F_M/self.F_M_max
        # 计算被动肌肉长度-力关系曲线
        fl_M_pas = curve.fiber_force_length_passive.with_defaults(l_M_tilde)
        # 计算主动肌肉长度-力关系曲线
        fl_M_act = curve.fiber_force_length_active.with_defaults(l_M_tilde)
        # 计算肌肉力-速度关系
        fv_M = (F_M_tilde - fl_M_pas)/(self.a*fl_M_act)
        # 计算归一化的肌肉速度
        v_M_tilde = curve.fiber_force_velocity_inverse.with_defaults(fv_M)
        # 计算肌肉实际速度
        v_M = v_M_tilde*self.v_M_max
        # 计算肌腱实际速度
        v_T = v_MT - v_M/cos_alpha
        # 计算归一化的肌腱速度
        v_T_tilde = v_T/self.l_T_slack
        # 计算肌腱力的变化率表达式
        self.dF_T_tilde_expr = (
            Float('0.2')*Float('33.93669377311689')*exp(
                Float('33.93669377311689')*UnevaluatedExpr(l_T_tilde - Float('0.995'))
            )*v_T_tilde
        )
        # 计算激活力的变化率表达式
        self.da_expr = (
            (1/(self.tau_a*(Rational(1, 2) + Rational(3, 2)*self.a)))
            *(Rational(1, 2) + Rational(1, 2)*tanh(self.b*(self.e - self.a)))
            + ((Rational(1, 2) + Rational(3, 2)*self.a)/self.tau_d)
            *(Rational(1, 2) - Rational(1, 2)*tanh(self.b*(self.e - self.a)))
        )*(self.e - self.a)
    # 检查实例对象是否具有属性 'x'
    assert hasattr(self.instance, 'x')
    # 检查实例对象是否具有属性 'state_vars'
    assert hasattr(self.instance, 'state_vars')
    # 断言实例对象的属性 'x' 等于属性 'state_vars'
    assert self.instance.x == self.instance.state_vars
    # 定义预期的 x 值，是一个包含 self.F_T_tilde 和 self.a 的 Matrix 对象
    x_expected = Matrix([self.F_T_tilde, self.a])
    # 断言实例对象的属性 'x' 等于预期的 x 值
    assert self.instance.x == x_expected
    # 断言实例对象的属性 'state_vars' 也等于预期的 x 值
    assert self.instance.state_vars == x_expected
    # 断言实例对象的属性 'x' 是 Matrix 类型
    assert isinstance(self.instance.x, Matrix)
    # 断言实例对象的属性 'state_vars' 是 Matrix 类型
    assert isinstance(self.instance.state_vars, Matrix)
    # 断言实例对象的属性 'x' 的形状为 (2, 1)
    assert self.instance.x.shape == (2, 1)
    # 断言实例对象的属性 'state_vars' 的形状为 (2, 1)
    assert self.instance.state_vars.shape == (2, 1)

    # 检查实例对象是否具有属性 'r'
    assert hasattr(self.instance, 'r')
    # 检查实例对象是否具有属性 'input_vars'
    assert hasattr(self.instance, 'input_vars')
    # 断言实例对象的属性 'r' 等于属性 'input_vars'
    assert self.instance.r == self.instance.input_vars
    # 定义预期的 r 值，是一个包含 self.e 的 Matrix 对象
    r_expected = Matrix([self.e])
    # 断言实例对象的属性 'r' 等于预期的 r 值
    assert self.instance.r == r_expected
    # 断言实例对象的属性 'input_vars' 也等于预期的 r 值
    assert self.instance.input_vars == r_expected
    # 断言实例对象的属性 'r' 是 Matrix 类型
    assert isinstance(self.instance.r, Matrix)
    # 断言实例对象的属性 'input_vars' 是 Matrix 类型
    assert isinstance(self.instance.input_vars, Matrix)
    # 断言实例对象的属性 'r' 的形状为 (1, 1)
    assert self.instance.r.shape == (1, 1)
    # 断言实例对象的属性 'input_vars' 的形状为 (1, 1)
    assert self.instance.input_vars.shape == (1, 1)

    # 检查实例对象是否具有属性 'p'
    assert hasattr(self.instance, 'p')
    # 检查实例对象是否具有属性 'constants'
    assert hasattr(self.instance, 'constants')
    # 断言实例对象的属性 'p' 等于属性 'constants'
    assert self.instance.p == self.instance.constants
    # 定义预期的 p 值，是一个包含多个常量的 Matrix 对象
    p_expected = Matrix(
        [
            self.l_T_slack,
            self.F_M_max,
            self.l_M_opt,
            self.v_M_max,
            self.alpha_opt,
            self.beta,
            self.tau_a,
            self.tau_d,
            self.b,
        ]
    )
    # 断言实例对象的属性 'p' 等于预期的 p 值
    assert self.instance.p == p_expected
    # 断言实例对象的属性 'constants' 也等于预期的 p 值
    assert self.instance.constants == p_expected
    # 断言实例对象的属性 'p' 是 Matrix 类型
    assert isinstance(self.instance.p, Matrix)
    # 断言实例对象的属性 'constants' 是 Matrix 类型
    assert isinstance(self.instance.constants, Matrix)
    # 断言实例对象的属性 'p' 的形状为 (9, 1)
    assert self.instance.p.shape == (9, 1)
    # 断言实例对象的属性 'constants' 的形状为 (9, 1)
    assert self.instance.constants.shape == (9, 1)

    # 检查实例对象是否具有属性 'M'
    assert hasattr(self.instance, 'M')
    # 定义预期的 M 值，是一个 2x2 的单位矩阵
    M_expected = eye(2)
    # 断言实例对象的属性 'M' 等于预期的 M 值
    assert self.instance.M == M_expected
    # 断言实例对象的属性 'M' 是 Matrix 类型
    assert isinstance(self.instance.M, Matrix)
    # 断言实例对象的属性 'M' 的形状为 (2, 2)
    assert self.instance.M.shape == (2, 2)

    # 检查实例对象是否具有属性 'F'
    assert hasattr(self.instance, 'F')
    # 定义预期的 F 值，是一个包含 self.dF_T_tilde_expr 和 self.da_expr 的 Matrix 对象
    F_expected = Matrix([self.dF_T_tilde_expr, self.da_expr])
    # 断言实例对象的属性 'F' 等于预期的 F 值
    assert self.instance.F == F_expected
    # 断言实例对象的属性 'F' 是 Matrix 类型
    assert isinstance(self.instance.F, Matrix)
    # 断言实例对象的属性 'F' 的形状为 (2, 1)
    assert self.instance.F.shape == (2, 1)

    # 检查实例对象是否具有属性 'rhs'
    assert hasattr(self.instance, 'rhs')
    # 定义预期的 rhs 值，是一个包含 self.dF_T_tilde_expr 和 self.da_expr 的 Matrix 对象
    rhs_expected = Matrix([self.dF_T_tilde_expr, self.da_expr])
    # 调用实例对象的 rhs 方法，获取其返回值
    rhs = self.instance.rhs()
    # 断言 rhs 的类型为 Matrix
    assert isinstance(rhs, Matrix)
    # 断言 rhs 的形状为 (2, 1)
    assert rhs.shape == (2, 1)
    # 断言简化后的 rhs 减去预期的 rhs_expected 的结果为零矩阵
    assert simplify(rhs - rhs_expected) == zeros(2, 1)
# 定义一个测试类 TestMusculotendonDeGroote2016，用于测试 MusculotendonDeGroote2016 类的功能
class TestMusculotendonDeGroote2016:

    # 静态方法，测试 MusculotendonDeGroote2016 类是否为 ForceActuator 的子类
    @staticmethod
    def test_class():
        assert issubclass(MusculotendonDeGroote2016, ForceActuator)
        # 测试 MusculotendonDeGroote2016 类是否为 _NamedMixin 的子类
        assert issubclass(MusculotendonDeGroote2016, _NamedMixin)
        # 测试 MusculotendonDeGroote2016 类的名称是否为 'MusculotendonDeGroote2016'
        assert MusculotendonDeGroote2016.__name__ == 'MusculotendonDeGroote2016'

    # 静态方法，测试 MusculotendonDeGroote2016 类的实例化过程
    @staticmethod
    def test_instance():
        # 创建一个名为 'pO' 的点对象 origin
        origin = Point('pO')
        # 创建一个名为 'pI' 的点对象 insertion
        insertion = Point('pI')
        # 将 insertion 点相对于 origin 点设置位置，位置向量为 q*N.x
        insertion.set_pos(origin, dynamicsymbols('q')*ReferenceFrame('N').x)
        # 创建一个线性路径对象 pathway，连接 origin 和 insertion 两个点
        pathway = LinearPathway(origin, insertion)
        # 创建一个名为 'name' 的 FirstOrderActivationDeGroote2016 激活对象 activation
        activation = FirstOrderActivationDeGroote2016('name')
        # 创建符号变量 l_T_slack，表示肌腱松弛长度
        l_T_slack = Symbol('l_T_slack')
        # 创建符号变量 F_M_max，表示最大等长肌力
        F_M_max = Symbol('F_M_max')
        # 创建符号变量 l_M_opt，表示最佳纤维长度
        l_M_opt = Symbol('l_M_opt')
        # 创建符号变量 v_M_max，表示最大纤维速度
        v_M_max = Symbol('v_M_max')
        # 创建符号变量 alpha_opt，表示最佳羽状肌角度
        alpha_opt = Symbol('alpha_opt')
        # 创建符号变量 beta，表示纤维阻尼系数
        beta = Symbol('beta')
        # 创建 MusculotendonDeGroote2016 类的实例 instance，传入各种参数进行初始化
        instance = MusculotendonDeGroote2016(
            'name',
            pathway,
            activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=l_T_slack,
            peak_isometric_force=F_M_max,
            optimal_fiber_length=l_M_opt,
            maximal_fiber_velocity=v_M_max,
            optimal_pennation_angle=alpha_opt,
            fiber_damping_coefficient=beta,
        )
        # 断言 instance 是否为 MusculotendonDeGroote2016 类的实例
        assert isinstance(instance, MusculotendonDeGroote2016)

    # pytest 的 fixture 方法，用于设置测试环境
    @pytest.fixture(autouse=True)
    def _musculotendon_fixture(self):
        # 设置 self.name 为 'name'
        self.name = 'name'
        # 创建参考坐标系 ReferenceFrame('N')，赋值给 self.N
        self.N = ReferenceFrame('N')
        # 创建动力学符号 q，赋值给 self.q
        self.q = dynamicsymbols('q')
        # 创建名为 'pO' 的点对象 origin，赋值给 self.origin
        self.origin = Point('pO')
        # 创建名为 'pI' 的点对象 insertion，赋值给 self.insertion
        self.insertion = Point('pI')
        # 将 self.insertion 点相对于 self.origin 点设置位置，位置向量为 self.q*self.N.x
        self.insertion.set_pos(self.origin, self.q*self.N.x)
        # 创建线性路径对象 LinearPathway(self.origin, self.insertion)，赋值给 self.pathway
        self.pathway = LinearPathway(self.origin, self.insertion)
        # 创建名为 'name' 的 FirstOrderActivationDeGroote2016 激活对象，赋值给 self.activation
        self.activation = FirstOrderActivationDeGroote2016(self.name)
        # 创建符号变量 l_T_slack，赋值给 self.l_T_slack，表示肌腱松弛长度
        self.l_T_slack = Symbol('l_T_slack')
        # 创建符号变量 F_M_max，赋值给 self.F_M_max，表示最大等长肌力
        self.F_M_max = Symbol('F_M_max')
        # 创建符号变量 l_M_opt，赋值给 self.l_M_opt，表示最佳纤维长度
        self.l_M_opt = Symbol('l_M_opt')
        # 创建符号变量 v_M_max，赋值给 self.v_M_max，表示最大纤维速度
        self.v_M_max = Symbol('v_M_max')
        # 创建符号变量 alpha_opt，赋值给 self.alpha_opt，表示最佳羽状肌角度
        self.alpha_opt = Symbol('alpha_opt')
        # 创建符号变量 beta，赋值给 self.beta，表示纤维阻尼系数
        self.beta = Symbol('beta')
    # 定义测试方法，测试默认参数设置情况下的功能
    def test_with_defaults(self):
        # 创建原点对象 'pO'
        origin = Point('pO')
        # 创建插入点对象 'pI'
        insertion = Point('pI')
        # 设置插入点的位置，相对于原点，动态符号 'q' 乘以参考坐标系 'N' 的 x 分量
        insertion.set_pos(origin, dynamicsymbols('q') * ReferenceFrame('N').x)
        # 创建线性路径对象，连接原点和插入点
        pathway = LinearPathway(origin, insertion)
        # 创建激活对象，采用第一阶段激活模型 DeGroote2016
        activation = FirstOrderActivationDeGroote2016('name')
        # 定义肌腱松弛长度符号
        l_T_slack = Symbol('l_T_slack')
        # 定义肌力最大等长收缩力符号
        F_M_max = Symbol('F_M_max')
        # 定义肌肉最优纤维长度符号
        l_M_opt = Symbol('l_M_opt')
        # 定义最大纤维速度为 10.0
        v_M_max = Float('10.0')
        # 定义最优肌肉膜斜角为 0.0
        alpha_opt = Float('0.0')
        # 定义纤维阻尼系数为 0.1
        beta = Float('0.1')
        # 使用默认参数创建 MusculotendonDeGroote2016 实例
        instance = MusculotendonDeGroote2016.with_defaults(
            'name',
            pathway,
            activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=l_T_slack,
            peak_isometric_force=F_M_max,
            optimal_fiber_length=l_M_opt,
        )
        # 断言实例属性与预期符号值相等
        assert instance.tendon_slack_length == l_T_slack
        assert instance.peak_isometric_force == F_M_max
        assert instance.optimal_fiber_length == l_M_opt
        assert instance.maximal_fiber_velocity == v_M_max
        assert instance.optimal_pennation_angle == alpha_opt
        assert instance.fiber_damping_coefficient == beta

    @pytest.mark.parametrize(
        'l_T_slack, expected',
        [
            # 测试肌腱松弛长度参数化情况下的各种预期值
            (None, Symbol('l_T_slack_name')),
            (Symbol('l_T_slack'), Symbol('l_T_slack')),
            (Rational(1, 2), Rational(1, 2)),
            (Float('0.5'), Float('0.5')),
        ],
    )
    # 测试肌腱松弛长度属性设置和获取方法
    def test_tendon_slack_length(self, l_T_slack, expected):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
        )
        # 断言实例属性与预期符号值相等
        assert instance.l_T_slack == expected
        assert instance.tendon_slack_length == expected

    @pytest.mark.parametrize(
        'F_M_max, expected',
        [
            # 测试肌力最大等长收缩力参数化情况下的各种预期值
            (None, Symbol('F_M_max_name')),
            (Symbol('F_M_max'), Symbol('F_M_max')),
            (Integer(1000), Integer(1000)),
            (Float('1000.0'), Float('1000.0')),
        ],
    )
    # 定义测试函数，用于验证 MusculotendonDeGroote2016 类的 peak_isometric_force 方法
    def test_peak_isometric_force(self, F_M_max, expected):
        # 创建 MusculotendonDeGroote2016 实例，初始化时设置相关参数
        instance = MusculotendonDeGroote2016(
            self.name,  # 名称参数
            self.pathway,  # 路径参数
            self.activation,  # 激活参数
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,  # 肌腱动力学设置为刚性肌腱
            tendon_slack_length=self.l_T_slack,  # 肌腱松弛长度
            peak_isometric_force=F_M_max,  # 峰值等长肌力
            optimal_fiber_length=self.l_M_opt,  # 最佳纤维长度
            maximal_fiber_velocity=self.v_M_max,  # 最大纤维速度
            optimal_pennation_angle=self.alpha_opt,  # 最佳羽毛角
            fiber_damping_coefficient=self.beta,  # 纤维阻尼系数
        )
        # 断言实例的 F_M_max 属性值符合预期
        assert instance.F_M_max == expected
        # 断言实例的 peak_isometric_force 方法返回值符合预期

        assert instance.peak_isometric_force == expected

    # 参数化测试函数，验证 MusculotendonDeGroote2016 类的 optimal_fiber_length 方法
    @pytest.mark.parametrize(
        'l_M_opt, expected',
        [
            (None, Symbol('l_M_opt_name')),  # 参数为 None 时，预期返回 Symbol('l_M_opt_name')
            (Symbol('l_M_opt'), Symbol('l_M_opt')),  # 参数为 Symbol('l_M_opt') 时，预期返回 Symbol('l_M_opt')
            (Rational(1, 2), Rational(1, 2)),  # 参数为 Rational(1, 2) 时，预期返回 Rational(1, 2)
            (Float('0.5'), Float('0.5')),  # 参数为 Float('0.5') 时，预期返回 Float('0.5')
        ],
    )
    def test_optimal_fiber_length(self, l_M_opt, expected):
        # 创建 MusculotendonDeGroote2016 实例，初始化时设置相关参数
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,  # 肌腱动力学设置为刚性肌腱
            tendon_slack_length=self.l_T_slack,  # 肌腱松弛长度
            peak_isometric_force=self.F_M_max,  # 峰值等长肌力
            optimal_fiber_length=l_M_opt,  # 最佳纤维长度
            maximal_fiber_velocity=self.v_M_max,  # 最大纤维速度
            optimal_pennation_angle=self.alpha_opt,  # 最佳羽毛角
            fiber_damping_coefficient=self.beta,  # 纤维阻尼系数
        )
        # 断言实例的 l_M_opt 属性值符合预期
        assert instance.l_M_opt == expected
        # 断言实例的 optimal_fiber_length 方法返回值符合预期
        assert instance.optimal_fiber_length == expected

    # 参数化测试函数，验证 MusculotendonDeGroote2016 类的 maximal_fiber_velocity 方法
    @pytest.mark.parametrize(
        'v_M_max, expected',
        [
            (None, Symbol('v_M_max_name')),  # 参数为 None 时，预期返回 Symbol('v_M_max_name')
            (Symbol('v_M_max'), Symbol('v_M_max')),  # 参数为 Symbol('v_M_max') 时，预期返回 Symbol('v_M_max')
            (Integer(10), Integer(10)),  # 参数为 Integer(10) 时，预期返回 Integer(10)
            (Float('10.0'), Float('10.0')),  # 参数为 Float('10.0') 时，预期返回 Float('10.0')
        ],
    )
    def test_maximal_fiber_velocity(self, v_M_max, expected):
        # 创建 MusculotendonDeGroote2016 实例，初始化时设置相关参数
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,  # 肌腱动力学设置为刚性肌腱
            tendon_slack_length=self.l_T_slack,  # 肌腱松弛长度
            peak_isometric_force=self.F_M_max,  # 峰值等长肌力
            optimal_fiber_length=self.l_M_opt,  # 最佳纤维长度
            maximal_fiber_velocity=v_M_max,  # 最大纤维速度
            optimal_pennation_angle=self.alpha_opt,  # 最佳羽毛角
            fiber_damping_coefficient=self.beta,  # 纤维阻尼系数
        )
        # 断言实例的 v_M_max 属性值符合预期
        assert instance.v_M_max == expected
        # 断言实例的 maximal_fiber_velocity 方法返回值符合预期

    # 参数化测试函数，验证 MusculotendonDeGroote2016 类的 optimal_pennation_angle 方法
    @pytest.mark.parametrize(
        'alpha_opt, expected',
        [
            (None, Symbol('alpha_opt_name')),  # 参数为 None 时，预期返回 Symbol('alpha_opt_name')
            (Symbol('alpha_opt'), Symbol('alpha_opt')),  # 参数为 Symbol('alpha_opt') 时，预期返回 Symbol('alpha_opt')
            (Integer(0), Integer(0)),  # 参数为 Integer(0) 时，预期返回 Integer(0)
            (Float('0.1'), Float('0.1')),  # 参数为 Float('0.1') 时，预期返回 Float('0.1')
        ],
    )
    # 测试最佳肌腱角度函数
    def test_optimal_pennation_angle(self, alpha_opt, expected):
        # 创建 MusculotendonDeGroote2016 类的实例，传入各项参数
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=alpha_opt,
            fiber_damping_coefficient=self.beta,
        )
        # 断言实例的 alpha_opt 属性与期望值相等
        assert instance.alpha_opt == expected
        # 断言实例的 optimal_pennation_angle 属性与期望值相等
        assert instance.optimal_pennation_angle == expected

    # 参数化测试纤维阻尼系数函数
    @pytest.mark.parametrize(
        'beta, expected',
        [
            (None, Symbol('beta_name')),
            (Symbol('beta'), Symbol('beta')),
            (Integer(0), Integer(0)),
            (Rational(1, 10), Rational(1, 10)),
            (Float('0.1'), Float('0.1')),
        ],
    )
    def test_fiber_damping_coefficient(self, beta, expected):
        # 创建 MusculotendonDeGroote2016 类的实例，传入各项参数
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=beta,
        )
        # 断言实例的 beta 属性与期望值相等
        assert instance.beta == expected
        # 断言实例的 fiber_damping_coefficient 属性与期望值相等
        assert instance.fiber_damping_coefficient == expected

    # 测试激活程度函数
    def test_excitation(self):
        # 创建 MusculotendonDeGroote2016 类的实例，传入必要参数
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
        )
        # 断言实例具有 'e' 属性和 'excitation' 属性
        assert hasattr(instance, 'e')
        assert hasattr(instance, 'excitation')
        # 创建动力符号 'e_name'，用于后续断言
        e_expected = dynamicsymbols('e_name')
        # 断言实例的 e 属性与期望值相等
        assert instance.e == e_expected
        # 断言实例的 excitation 属性与期望值相等
        assert instance.excitation == e_expected
        # 断言实例的 e 和 excitation 是同一个对象
        assert instance.e is instance.excitation

    # 测试激活程度属性不可变性函数
    def test_excitation_is_immutable(self):
        # 创建 MusculotendonDeGroote2016 类的实例，传入必要参数
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
        )
        # 使用 pytest 的 assertRaises 断言实例的 'e' 属性不可赋值为 None
        with pytest.raises(AttributeError):
            instance.e = None
        # 使用 pytest 的 assertRaises 断言实例的 'excitation' 属性不可赋值为 None
        with pytest.raises(AttributeError):
            instance.excitation = None

    # 测试激活函数
    def test_activation(self):
        # 创建 MusculotendonDeGroote2016 类的实例，传入必要参数
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
        )
        # 断言实例具有 'a' 属性和 'activation' 属性
        assert hasattr(instance, 'a')
        assert hasattr(instance, 'activation')
        # 创建动力符号 'a_name'，用于后续断言
        a_expected = dynamicsymbols('a_name')
        # 断言实例的 a 属性与期望值相等
        assert instance.a == a_expected
        # 断言实例的 activation 属性与期望值相等
        assert instance.activation == a_expected
    # 测试函数：验证 MusculotendonDeGroote2016 类的 activation 和 a 属性为不可变
    def test_activation_is_immutable(self):
        # 创建 MusculotendonDeGroote2016 类的实例，传入名称、pathway 和 activation 参数
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
        )
        # 使用 pytest 断言检测设置属性 a 为 None 时是否引发 AttributeError 异常
        with pytest.raises(AttributeError):
            instance.a = None
        # 使用 pytest 断言检测设置属性 activation 为 None 时是否引发 AttributeError 异常
        with pytest.raises(AttributeError):
            instance.activation = None

    # 测试函数：验证 MusculotendonDeGroote2016 类的 repr 方法返回预期字符串
    def test_repr(self):
        # 创建 MusculotendonDeGroote2016 类的实例，传入多个参数，包括名称、pathway、activation 等
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
        )
        # 期望的 repr 字符串，展示了 MusculotendonDeGroote2016 实例的参数信息
        expected = (
            'MusculotendonDeGroote2016(\'name\', '
            'pathway=LinearPathway(pO, pI), '
            'activation_dynamics=FirstOrderActivationDeGroote2016(\'name\', '
            'activation_time_constant=tau_a_name, '
            'deactivation_time_constant=tau_d_name, '
            'smoothing_rate=b_name), '
            'musculotendon_dynamics=0, '
            'tendon_slack_length=l_T_slack, '
            'peak_isometric_force=F_M_max, '
            'optimal_fiber_length=l_M_opt, '
            'maximal_fiber_velocity=v_M_max, '
            'optimal_pennation_angle=alpha_opt, '
            'fiber_damping_coefficient=beta)'
        )
        # 使用 assert 断言验证实例的 repr 方法返回的字符串与预期字符串 expected 相符
        assert repr(instance) == expected


这段代码中，两个测试函数分别测试了一个类的不可变性和 `repr` 方法的输出是否符合预期。
```
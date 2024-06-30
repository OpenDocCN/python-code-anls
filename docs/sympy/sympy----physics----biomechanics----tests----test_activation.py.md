# `D:\src\scipysrc\sympy\sympy\physics\biomechanics\tests\test_activation.py`

```
"""Tests for the ``sympy.physics.biomechanics.activation.py`` module."""

# 导入 pytest 模块，用于编写和运行测试
import pytest

# 导入 SymPy 库中的符号变量 Symbol
from sympy import Symbol
# 导入 SymPy 库中的数值类型 Float, Integer, Rational
from sympy.core.numbers import Float, Integer, Rational
# 导入 SymPy 库中的双曲函数 tanh
from sympy.functions.elementary.hyperbolic import tanh
# 导入 SymPy 库中的矩阵类 Matrix
from sympy.matrices import Matrix
# 导入 SymPy 库中的稠密矩阵 zeros 函数
from sympy.matrices.dense import zeros
# 导入 SymPy 物理力学模块中的动力符号 dynamicsymbols
from sympy.physics.mechanics import dynamicsymbols
# 导入 SymPy 生物力学模块中的激活类 ActivationBase, FirstOrderActivationDeGroote2016, ZerothOrderActivation
from sympy.physics.biomechanics import (
    ActivationBase,
    FirstOrderActivationDeGroote2016,
    ZerothOrderActivation,
)
# 导入 SymPy 生物力学模块中的 _NamedMixin 类
from sympy.physics.biomechanics._mixin import _NamedMixin
# 导入 SymPy 简化模块中的 simplify 函数
from sympy.simplify.simplify import simplify


# 定义测试类 TestZerothOrderActivation
class TestZerothOrderActivation:

    # 测试类方法 test_class
    @staticmethod
    def test_class():
        # 断言 ZerothOrderActivation 类是 ActivationBase 的子类
        assert issubclass(ZerothOrderActivation, ActivationBase)
        # 断言 ZerothOrderActivation 类是 _NamedMixin 的子类
        assert issubclass(ZerothOrderActivation, _NamedMixin)
        # 断言 ZerothOrderActivation 类的名称为 'ZerothOrderActivation'
        assert ZerothOrderActivation.__name__ == 'ZerothOrderActivation'

    # pytest 的自动使用的装饰器 fixture，用于设置测试环境
    @pytest.fixture(autouse=True)
    def _zeroth_order_activation_fixture(self):
        # 设置实例变量 self.name 为 'name'
        self.name = 'name'
        # 使用 dynamicsymbols 函数创建动态符号 'e_name'，赋值给实例变量 self.e
        self.e = dynamicsymbols('e_name')
        # 创建 ZerothOrderActivation 类的实例，传入 self.name 作为参数，赋值给 self.instance
        self.instance = ZerothOrderActivation(self.name)

    # 测试方法 test_instance
    def test_instance(self):
        # 创建 ZerothOrderActivation 类的实例，传入 self.name 作为参数，赋值给 instance
        instance = ZerothOrderActivation(self.name)
        # 断言 instance 是 ZerothOrderActivation 类的实例
        assert isinstance(instance, ZerothOrderActivation)

    # 测试方法 test_with_defaults
    def test_with_defaults(self):
        # 调用 ZerothOrderActivation 类的 with_defaults 方法，传入 self.name 作为参数，赋值给 instance
        instance = ZerothOrderActivation.with_defaults(self.name)
        # 断言 instance 是 ZerothOrderActivation 类的实例
        assert isinstance(instance, ZerothOrderActivation)
        # 断言 instance 等于 ZerothOrderActivation 类的一个新实例，传入 self.name 作为参数
        assert instance == ZerothOrderActivation(self.name)

    # 测试方法 test_name
    def test_name(self):
        # 断言 self.instance 具有属性 'name'
        assert hasattr(self.instance, 'name')
        # 断言 self.instance 的 name 属性等于 self.name
        assert self.instance.name == self.name

    # 测试方法 test_order
    def test_order(self):
        # 断言 self.instance 具有属性 'order'
        assert hasattr(self.instance, 'order')
        # 断言 self.instance 的 order 属性等于 0
        assert self.instance.order == 0

    # 测试方法 test_excitation_attribute
    def test_excitation_attribute(self):
        # 断言 self.instance 具有属性 'e'
        assert hasattr(self.instance, 'e')
        # 断言 self.instance 具有属性 'excitation'
        assert hasattr(self.instance, 'excitation')
        # 使用 dynamicsymbols 函数创建动态符号 'e_name'，赋值给 e_expected
        e_expected = dynamicsymbols('e_name')
        # 断言 self.instance 的 e 属性等于 e_expected
        assert self.instance.e == e_expected
        # 断言 self.instance 的 excitation 属性等于 e_expected
        assert self.instance.excitation == e_expected
        # 断言 self.instance 的 e 和 excitation 属性是同一个对象
        assert self.instance.e is self.instance.excitation

    # 测试方法 test_activation_attribute
    def test_activation_attribute(self):
        # 断言 self.instance 具有属性 'a'
        assert hasattr(self.instance, 'a')
        # 断言 self.instance 具有属性 'activation'
        assert hasattr(self.instance, 'activation')
        # 使用 dynamicsymbols 函数创建动态符号 'e_name'，赋值给 a_expected
        a_expected = dynamicsymbols('e_name')
        # 断言 self.instance 的 a 属性等于 a_expected
        assert self.instance.a == a_expected
        # 断言 self.instance 的 activation 属性等于 a_expected
        assert self.instance.activation == a_expected
        # 断言 self.instance 的 a, activation 和 e 属性是同一个对象
        assert self.instance.a is self.instance.activation is self.instance.e

    # 测试方法 test_state_vars_attribute
    def test_state_vars_attribute(self):
        # 断言 self.instance 具有属性 'x'
        assert hasattr(self.instance, 'x')
        # 断言 self.instance 具有属性 'state_vars'
        assert hasattr(self.instance, 'state_vars')
        # 断言 self.instance 的 x 属性等于 self.instance 的 state_vars 属性
        assert self.instance.x == self.instance.state_vars
        # 使用 zeros 函数创建一个 0x1 的矩阵，赋值给 x_expected
        x_expected = zeros(0, 1)
        # 断言 self.instance 的 x 属性等于 x_expected
        assert self.instance.x == x_expected
        # 断言 self.instance 的 state_vars 属性等于 x_expected
        assert self.instance.state_vars == x_expected
        # 断言 self.instance 的 x 和 state_vars 属性是 Matrix 类的实例
        assert isinstance(self.instance.x, Matrix)
        assert isinstance(self.instance.state_vars, Matrix)
        # 断言 self.instance 的 x 和 state_vars 属性的形状为 (0, 1)
        assert self.instance.x.shape == (0, 1)
        assert self.instance.state_vars.shape == (0, 1)
    # 检查实例对象是否具有'r'属性
    assert hasattr(self.instance, 'r')
    # 检查实例对象是否具有'input_vars'属性
    assert hasattr(self.instance, 'input_vars')
    # 断言'r'属性和'input_vars'属性相等
    assert self.instance.r == self.instance.input_vars
    # 创建预期的 Matrix 对象，并断言'r'属性与其相等
    r_expected = Matrix([self.e])
    assert self.instance.r == r_expected
    # 断言'input_vars'属性与预期的 Matrix 对象相等
    assert self.instance.input_vars == r_expected
    # 断言'r'属性和'input_vars'属性确实是 Matrix 类型的对象
    assert isinstance(self.instance.r, Matrix)
    assert isinstance(self.instance.input_vars, Matrix)
    # 断言'r'属性和'input_vars'属性的形状为 (1, 1)
    assert self.instance.r.shape == (1, 1)
    assert self.instance.input_vars.shape == (1, 1)

    # 检查实例对象是否具有'p'属性
    assert hasattr(self.instance, 'p')
    # 检查实例对象是否具有'constants'属性
    assert hasattr(self.instance, 'constants')
    # 断言'p'属性和'constants'属性相等
    assert self.instance.p == self.instance.constants
    # 创建预期的 Matrix 对象，并断言'p'属性与其相等
    p_expected = zeros(0, 1)
    assert self.instance.p == p_expected
    # 断言'constants'属性与预期的 Matrix 对象相等
    assert self.instance.constants == p_expected
    # 断言'p'属性和'constants'属性确实是 Matrix 类型的对象
    assert isinstance(self.instance.p, Matrix)
    assert isinstance(self.instance.constants, Matrix)
    # 断言'p'属性和'constants'属性的形状为 (0, 1)
    assert self.instance.p.shape == (0, 1)
    assert self.instance.constants.shape == (0, 1)

    # 检查实例对象是否具有'M'属性
    assert hasattr(self.instance, 'M')
    # 创建预期的空 Matrix 对象，并断言'M'属性与其相等
    M_expected = Matrix([])
    assert self.instance.M == M_expected
    # 断言'M'属性确实是 Matrix 类型的对象
    assert isinstance(self.instance.M, Matrix)
    # 断言'M'属性的形状为 (0, 0)
    assert self.instance.M.shape == (0, 0)

    # 检查实例对象是否具有'F'属性
    assert hasattr(self.instance, 'F')
    # 创建预期的零 Matrix 对象，并断言'F'属性与其相等
    F_expected = zeros(0, 1)
    assert self.instance.F == F_expected
    # 断言'F'属性确实是 Matrix 类型的对象
    assert isinstance(self.instance.F, Matrix)
    # 断言'F'属性的形状为 (0, 1)
    assert self.instance.F.shape == (0, 1)

    # 检查实例对象是否具有'rhs'属性
    assert hasattr(self.instance, 'rhs')
    # 创建预期的零 Matrix 对象，并调用'rhs'方法获取结果
    rhs_expected = zeros(0, 1)
    rhs = self.instance.rhs()
    # 断言'rhs'方法的返回值与预期的零 Matrix 对象相等
    assert rhs == rhs_expected
    # 断言'rhs'方法返回的对象确实是 Matrix 类型的对象
    assert isinstance(rhs, Matrix)
    # 断言'rhs'方法返回的对象的形状为 (0, 1)
    assert rhs.shape == (0, 1)

    # 创建预期的字符串表示，并断言实例对象的 repr 方法返回结果与其相等
    expected = 'ZerothOrderActivation(\'name\')'
    assert repr(self.instance) == expected
# 定义测试类 TestFirstOrderActivationDeGroote2016，用于测试 FirstOrderActivationDeGroote2016 类
class TestFirstOrderActivationDeGroote2016:

    # 静态方法 test_class，用于测试类的继承关系和类名
    @staticmethod
    def test_class():
        # 断言 FirstOrderActivationDeGroote2016 是 ActivationBase 的子类
        assert issubclass(FirstOrderActivationDeGroote2016, ActivationBase)
        # 断言 FirstOrderActivationDeGroote2016 是 _NamedMixin 的子类
        assert issubclass(FirstOrderActivationDeGroote2016, _NamedMixin)
        # 断言 FirstOrderActivationDeGroote2016 类名为 'FirstOrderActivationDeGroote2016'
        assert FirstOrderActivationDeGroote2016.__name__ == 'FirstOrderActivationDeGroote2016'

    # pytest fixture 方法，用于创建测试环境中的实例及其属性
    @pytest.fixture(autouse=True)
    def _first_order_activation_de_groote_2016_fixture(self):
        # 设置实例的名称属性为 'name'
        self.name = 'name'
        # 创建动态符号 e_name 并赋值给实例的 e 属性
        self.e = dynamicsymbols('e_name')
        # 创建动态符号 a_name 并赋值给实例的 a 属性
        self.a = dynamicsymbols('a_name')
        # 创建符号 tau_a 并赋值给实例的 tau_a 属性
        self.tau_a = Symbol('tau_a')
        # 创建符号 tau_d 并赋值给实例的 tau_d 属性
        self.tau_d = Symbol('tau_d')
        # 创建符号 b 并赋值给实例的 b 属性
        self.b = Symbol('b')
        # 使用名称、tau_a、tau_d、b 创建 FirstOrderActivationDeGroote2016 的实例，并赋给 self.instance 属性
        self.instance = FirstOrderActivationDeGroote2016(
            self.name,
            self.tau_a,
            self.tau_d,
            self.b,
        )

    # 测试实例化方法是否正常工作
    def test_instance(self):
        # 使用名称创建 FirstOrderActivationDeGroote2016 的实例
        instance = FirstOrderActivationDeGroote2016(self.name)
        # 断言 instance 是 FirstOrderActivationDeGroote2016 的实例
        assert isinstance(instance, FirstOrderActivationDeGroote2016)

    # 测试带默认参数的实例化方法是否正常工作
    def test_with_defaults(self):
        # 使用名称创建具有默认参数的 FirstOrderActivationDeGroote2016 实例
        instance = FirstOrderActivationDeGroote2016.with_defaults(self.name)
        # 断言 instance 是 FirstOrderActivationDeGroote2016 的实例
        assert isinstance(instance, FirstOrderActivationDeGroote2016)
        # 断言 instance 的 tau_a 属性值为 Float('0.015')
        assert instance.tau_a == Float('0.015')
        # 断言 instance 的 activation_time_constant 属性值为 Float('0.015')
        assert instance.activation_time_constant == Float('0.015')
        # 断言 instance 的 tau_d 属性值为 Float('0.060')
        assert instance.tau_d == Float('0.060')
        # 断言 instance 的 deactivation_time_constant 属性值为 Float('0.060')
        assert instance.deactivation_time_constant == Float('0.060')
        # 断言 instance 的 b 属性值为 Float('10.0')
        assert instance.b == Float('10.0')
        # 断言 instance 的 smoothing_rate 属性值为 Float('10.0')
        assert instance.smoothing_rate == Float('10.0')

    # 测试实例的名称属性
    def test_name(self):
        # 断言 self.instance 具有 'name' 属性
        assert hasattr(self.instance, 'name')
        # 断言 self.instance 的 name 属性值为 self.name
        assert self.instance.name == self.name

    # 测试实例的 order 属性
    def test_order(self):
        # 断言 self.instance 具有 'order' 属性
        assert hasattr(self.instance, 'order')
        # 断言 self.instance 的 order 属性值为 1
        assert self.instance.order == 1

    # 测试激励属性
    def test_excitation(self):
        # 断言 self.instance 具有 'e' 属性
        assert hasattr(self.instance, 'e')
        # 断言 self.instance 具有 'excitation' 属性
        assert hasattr(self.instance, 'excitation')
        # 创建预期的动态符号 e_name
        e_expected = dynamicsymbols('e_name')
        # 断言 self.instance 的 e 属性值与 e_expected 相等
        assert self.instance.e == e_expected
        # 断言 self.instance 的 excitation 属性值与 e_expected 相等
        assert self.instance.excitation == e_expected
        # 断言 self.instance 的 e 属性与 excitation 属性是同一对象
        assert self.instance.e is self.instance.excitation

    # 测试激励属性不可变性
    def test_excitation_is_immutable(self):
        # 测试尝试设置 self.instance 的 e 属性为 None 是否引发 AttributeError 异常
        with pytest.raises(AttributeError):
            self.instance.e = None
        # 测试尝试设置 self.instance 的 excitation 属性为 None 是否引发 AttributeError 异常
        with pytest.raises(AttributeError):
            self.instance.excitation = None

    # 测试激活属性
    def test_activation(self):
        # 断言 self.instance 具有 'a' 属性
        assert hasattr(self.instance, 'a')
        # 断言 self.instance 具有 'activation' 属性
        assert hasattr(self.instance, 'activation')
        # 创建预期的动态符号 a_name
        a_expected = dynamicsymbols('a_name')
        # 断言 self.instance 的 a 属性值与 a_expected 相等
        assert self.instance.a == a_expected
        # 断言 self.instance 的 activation 属性值与 a_expected 相等
        assert self.instance.activation == a_expected

    # 测试激活属性不可变性
    def test_activation_is_immutable(self):
        # 测试尝试设置 self.instance 的 a 属性为 None 是否引发 AttributeError 异常
        with pytest.raises(AttributeError):
            self.instance.a = None
        # 测试尝试设置 self.instance 的 activation 属性为 None 是否引发 AttributeError 异常
        with pytest.raises(AttributeError):
            self.instance.activation = None

    # 使用参数化测试 tau_a 属性
    @pytest.mark.parametrize(
        'tau_a, expected',
        [
            (None, Symbol('tau_a_name')),
            (Symbol('tau_a'), Symbol('tau_a')),
            (Float('0.015'), Float('0.015')),
        ]
    )
    # 测试激活时间常数是否设置正确
    def test_activation_time_constant(self, tau_a, expected):
        # 创建 FirstOrderActivationDeGroote2016 类的实例，设置激活时间常数
        instance = FirstOrderActivationDeGroote2016(
            'name', activation_time_constant=tau_a,
        )
        # 断言实例的 tau_a 属性与期望值相等
        assert instance.tau_a == expected
        # 断言实例的 activation_time_constant 属性与期望值相等
        assert instance.activation_time_constant == expected
        # 断言 tau_a 和 activation_time_constant 是同一个对象
        assert instance.tau_a is instance.activation_time_constant

    # 测试激活时间常数是否可变，预期会抛出 AttributeError
    def test_activation_time_constant_is_immutable(self):
        with pytest.raises(AttributeError):
            self.instance.tau_a = None
        with pytest.raises(AttributeError):
            self.instance.activation_time_constant = None

    # 使用 pytest.mark.parametrize 参数化测试去激活时间常数的设定
    @pytest.mark.parametrize(
        'tau_d, expected',
        [
            (None, Symbol('tau_d_name')),
            (Symbol('tau_d'), Symbol('tau_d')),
            (Float('0.060'), Float('0.060')),
        ]
    )
    # 测试去激活时间常数是否设置正确
    def test_deactivation_time_constant(self, tau_d, expected):
        # 创建 FirstOrderActivationDeGroote2016 类的实例，设置去激活时间常数
        instance = FirstOrderActivationDeGroote2016(
            'name', deactivation_time_constant=tau_d,
        )
        # 断言实例的 tau_d 属性与期望值相等
        assert instance.tau_d == expected
        # 断言实例的 deactivation_time_constant 属性与期望值相等
        assert instance.deactivation_time_constant == expected
        # 断言 tau_d 和 deactivation_time_constant 是同一个对象
        assert instance.tau_d is instance.deactivation_time_constant

    # 测试去激活时间常数是否可变，预期会抛出 AttributeError
    def test_deactivation_time_constant_is_immutable(self):
        with pytest.raises(AttributeError):
            self.instance.tau_d = None
        with pytest.raises(AttributeError):
            self.instance.deactivation_time_constant = None

    # 使用 pytest.mark.parametrize 参数化测试平滑率的设定
    @pytest.mark.parametrize(
        'b, expected',
        [
            (None, Symbol('b_name')),
            (Symbol('b'), Symbol('b')),
            (Integer('10'), Integer('10')),
        ]
    )
    # 测试平滑率是否设置正确
    def test_smoothing_rate(self, b, expected):
        # 创建 FirstOrderActivationDeGroote2016 类的实例，设置平滑率
        instance = FirstOrderActivationDeGroote2016(
            'name', smoothing_rate=b,
        )
        # 断言实例的 b 属性与期望值相等
        assert instance.b == expected
        # 断言实例的 smoothing_rate 属性与期望值相等
        assert instance.smoothing_rate == expected
        # 断言 b 和 smoothing_rate 是同一个对象
        assert instance.b is instance.smoothing_rate

    # 测试平滑率是否可变，预期会抛出 AttributeError
    def test_smoothing_rate_is_immutable(self):
        with pytest.raises(AttributeError):
            self.instance.b = None
        with pytest.raises(AttributeError):
            self.instance.smoothing_rate = None

    # 测试状态变量的设置与属性
    def test_state_vars(self):
        # 断言实例有 'x' 和 'state_vars' 这两个属性
        assert hasattr(self.instance, 'x')
        assert hasattr(self.instance, 'state_vars')
        # 断言实例的 x 属性与 state_vars 属性相等
        assert self.instance.x == self.instance.state_vars
        # 创建一个预期的 Matrix 对象，并断言实例的 x 属性与其相等
        x_expected = Matrix([self.a])
        assert self.instance.x == x_expected
        # 断言实例的 state_vars 属性与预期的 x_expected 相等
        assert self.instance.state_vars == x_expected
        # 断言实例的 x 和 state_vars 属性都是 Matrix 类型
        assert isinstance(self.instance.x, Matrix)
        assert isinstance(self.instance.state_vars, Matrix)
        # 断言实例的 x 属性和 state_vars 属性的形状都是 (1, 1)
        assert self.instance.x.shape == (1, 1)
        assert self.instance.state_vars.shape == (1, 1)
    # 测试输入变量是否正确设置
    def test_input_vars(self):
        # 断言实例对象是否有属性 'r'
        assert hasattr(self.instance, 'r')
        # 断言实例对象是否有属性 'input_vars'
        assert hasattr(self.instance, 'input_vars')
        # 断言实例对象的 'r' 属性与 'input_vars' 属性相等
        assert self.instance.r == self.instance.input_vars
        # 创建一个预期的 Matrix 对象，包含单一元素 self.e
        r_expected = Matrix([self.e])
        # 断言实例对象的 'r' 属性与预期的 r_expected 对象相等
        assert self.instance.r == r_expected
        # 断言实例对象的 'input_vars' 属性与预期的 r_expected 对象相等
        assert self.instance.input_vars == r_expected
        # 断言实例对象的 'r' 属性是否为 Matrix 类型
        assert isinstance(self.instance.r, Matrix)
        # 断言实例对象的 'input_vars' 属性是否为 Matrix 类型
        assert isinstance(self.instance.input_vars, Matrix)
        # 断言实例对象的 'r' 属性的形状为 (1, 1)
        assert self.instance.r.shape == (1, 1)
        # 断言实例对象的 'input_vars' 属性的形状为 (1, 1)
        assert self.instance.input_vars.shape == (1, 1)

    # 测试常量是否正确设置
    def test_constants(self):
        # 断言实例对象是否有属性 'p'
        assert hasattr(self.instance, 'p')
        # 断言实例对象是否有属性 'constants'
        assert hasattr(self.instance, 'constants')
        # 断言实例对象的 'p' 属性与 'constants' 属性相等
        assert self.instance.p == self.instance.constants
        # 创建一个预期的 Matrix 对象，包含元素 self.tau_a, self.tau_d, self.b
        p_expected = Matrix([self.tau_a, self.tau_d, self.b])
        # 断言实例对象的 'p' 属性与预期的 p_expected 对象相等
        assert self.instance.p == p_expected
        # 断言实例对象的 'constants' 属性与预期的 p_expected 对象相等
        assert self.instance.constants == p_expected
        # 断言实例对象的 'p' 属性是否为 Matrix 类型
        assert isinstance(self.instance.p, Matrix)
        # 断言实例对象的 'constants' 属性是否为 Matrix 类型
        assert isinstance(self.instance.constants, Matrix)
        # 断言实例对象的 'p' 属性的形状为 (3, 1)
        assert self.instance.p.shape == (3, 1)
        # 断言实例对象的 'constants' 属性的形状为 (3, 1)
        assert self.instance.constants.shape == (3, 1)

    # 测试属性 'M' 是否正确设置
    def test_M(self):
        # 断言实例对象是否有属性 'M'
        assert hasattr(self.instance, 'M')
        # 创建一个预期的 Matrix 对象，包含单一元素 1
        M_expected = Matrix([1])
        # 断言实例对象的 'M' 属性与预期的 M_expected 对象相等
        assert self.instance.M == M_expected
        # 断言实例对象的 'M' 属性是否为 Matrix 类型
        assert isinstance(self.instance.M, Matrix)
        # 断言实例对象的 'M' 属性的形状为 (1, 1)
        assert self.instance.M.shape == (1, 1)

    # 测试属性 'F' 是否正确设置
    def test_F(self):
        # 断言实例对象是否有属性 'F'
        assert hasattr(self.instance, 'F')
        # 计算表达式 da_expr
        da_expr = (
            ((1/(self.tau_a*(Rational(1, 2) + Rational(3, 2)*self.a)))
            *(Rational(1, 2) + Rational(1, 2)*tanh(self.b*(self.e - self.a)))
            + ((Rational(1, 2) + Rational(3, 2)*self.a)/self.tau_d)
            *(Rational(1, 2) - Rational(1, 2)*tanh(self.b*(self.e - self.a))))
            *(self.e - self.a)
        )
        # 创建一个预期的 Matrix 对象，包含单一元素 da_expr
        F_expected = Matrix([da_expr])
        # 断言实例对象的 'F' 属性与预期的 F_expected 对象相等
        assert self.instance.F == F_expected
        # 断言实例对象的 'F' 属性是否为 Matrix 类型
        assert isinstance(self.instance.F, Matrix)
        # 断言实例对象的 'F' 属性的形状为 (1, 1)
        assert self.instance.F.shape == (1, 1)

    # 测试方法 'rhs' 的输出是否正确
    def test_rhs(self):
        # 断言实例对象是否有属性 'rhs'
        assert hasattr(self.instance, 'rhs')
        # 计算表达式 da_expr
        da_expr = (
            ((1/(self.tau_a*(Rational(1, 2) + Rational(3, 2)*self.a)))
            *(Rational(1, 2) + Rational(1, 2)*tanh(self.b*(self.e - self.a)))
            + ((Rational(1, 2) + Rational(3, 2)*self.a)/self.tau_d)
            *(Rational(1, 2) - Rational(1, 2)*tanh(self.b*(self.e - self.a))))
            *(self.e - self.a)
        )
        # 创建一个预期的 Matrix 对象，包含单一元素 da_expr
        rhs_expected = Matrix([da_expr])
        # 调用实例对象的方法 'rhs'，并断言其输出与预期的 rhs_expected 对象相等
        rhs = self.instance.rhs()
        assert rhs == rhs_expected
        # 断言 rhs 对象是否为 Matrix 类型
        assert isinstance(rhs, Matrix)
        # 断言 rhs 对象的形状为 (1, 1)
        assert rhs.shape == (1, 1)
        # 断言解出的结果是否为零向量
        assert simplify(self.instance.M.solve(self.instance.F) - rhs) == zeros(1)

    # 测试实例对象的表示是否正确
    def test_repr(self):
        # 创建预期的字符串表示
        expected = (
            'FirstOrderActivationDeGroote2016(\'name\', '
            'activation_time_constant=tau_a, '
            'deactivation_time_constant=tau_d, '
            'smoothing_rate=b)'
        )
        # 断言实例对象的字符串表示与预期的字符串相等
        assert repr(self.instance) == expected
```
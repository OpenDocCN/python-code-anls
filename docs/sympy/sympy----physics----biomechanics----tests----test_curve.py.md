# `D:\src\scipysrc\sympy\sympy\physics\biomechanics\tests\test_curve.py`

```
"""Tests for the ``sympy.physics.biomechanics.characteristic.py`` module."""

# 导入 pytest 模块，用于编写和运行测试用例
import pytest

# 导入 sympy 中的各种符号、函数和类
from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import Function
from sympy.core.numbers import Float, Integer
from sympy.core.symbol import Symbol, symbols
from sympy.external.importtools import import_module
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import cosh, sinh
from sympy.functions.elementary.miscellaneous import sqrt

# 导入 sympy.physics.biomechanics.curve 模块中的特征曲线类
from sympy.physics.biomechanics.curve import (
    CharacteristicCurveCollection,
    CharacteristicCurveFunction,
    FiberForceLengthActiveDeGroote2016,
    FiberForceLengthPassiveDeGroote2016,
    FiberForceLengthPassiveInverseDeGroote2016,
    FiberForceVelocityDeGroote2016,
    FiberForceVelocityInverseDeGroote2016,
    TendonForceLengthDeGroote2016,
    TendonForceLengthInverseDeGroote2016,
)

# 导入 sympy.printing 中的打印器类
from sympy.printing.c import C89CodePrinter, C99CodePrinter, C11CodePrinter
from sympy.printing.cxx import (
    CXX98CodePrinter,
    CXX11CodePrinter,
    CXX17CodePrinter,
)
from sympy.printing.fortran import FCodePrinter
from sympy.printing.lambdarepr import LambdaPrinter
from sympy.printing.latex import LatexPrinter
from sympy.printing.octave import OctaveCodePrinter
from sympy.printing.numpy import (
    CuPyPrinter,
    JaxPrinter,
    NumPyPrinter,
    SciPyPrinter,
)
from sympy.printing.pycode import MpmathPrinter, PythonCodePrinter

# 导入 sympy.utilities.lambdify 模块中的 lambdify 函数
from sympy.utilities.lambdify import lambdify

# 尝试导入 jax 和 numpy 模块，如果导入成功则更新 jax 配置
jax = import_module('jax')
numpy = import_module('numpy')

if jax:
    # 如果 jax 导入成功，更新其配置以支持双精度浮点数
    jax.config.update('jax_enable_x64', True)

# 定义测试类 TestCharacteristicCurveFunction
class TestCharacteristicCurveFunction:

    @staticmethod
    # 使用 pytest.mark.parametrize 装饰器参数化测试用例
    @pytest.mark.parametrize(
        'code_printer, expected',
        [
            # 不同的打印器类及其预期输出字符串
            (C89CodePrinter, '(a + b)*(c + d)*(e + f)'),
            (C99CodePrinter, '(a + b)*(c + d)*(e + f)'),
            (C11CodePrinter, '(a + b)*(c + d)*(e + f)'),
            (CXX98CodePrinter, '(a + b)*(c + d)*(e + f)'),
            (CXX11CodePrinter, '(a + b)*(c + d)*(e + f)'),
            (CXX17CodePrinter, '(a + b)*(c + d)*(e + f)'),
            (FCodePrinter, '      (a + b)*(c + d)*(e + f)'),
            (OctaveCodePrinter, '(a + b).*(c + d).*(e + f)'),
            (PythonCodePrinter, '(a + b)*(c + d)*(e + f)'),
            (NumPyPrinter, '(a + b)*(c + d)*(e + f)'),
            (SciPyPrinter, '(a + b)*(c + d)*(e + f)'),
            (CuPyPrinter, '(a + b)*(c + d)*(e + f)'),
            (JaxPrinter, '(a + b)*(c + d)*(e + f)'),
            (MpmathPrinter, '(a + b)*(c + d)*(e + f)'),
            (LambdaPrinter, '(a + b)*(c + d)*(e + f)'),
        ]
    )
    # 定义一个测试函数 test_print_code_parenthesize，接受两个参数：code_printer 和 expected
    def test_print_code_parenthesize(code_printer, expected):

        # 定义一个继承自 CharacteristicCurveFunction 的示例类 ExampleFunction
        class ExampleFunction(CharacteristicCurveFunction):

            # 类方法 eval，暂时未实现具体功能
            @classmethod
            def eval(cls, a, b):
                pass

            # 实例方法 doit，接受任意关键字参数
            def doit(self, **kwargs):
                # 解包 self.args 为 a 和 b
                a, b = self.args
                # 返回 a 和 b 的和
                return a + b

        # 使用 symbols 函数创建符号对象 a, b, c, d, e, f
        a, b, c, d, e, f = symbols('a, b, c, d, e, f')
        # 创建 ExampleFunction 的实例 f1, f2, f3，分别传入不同的参数对
        f1 = ExampleFunction(a, b)
        f2 = ExampleFunction(c, d)
        f3 = ExampleFunction(e, f)
        # 断言调用 code_printer().doprint(f1*f2*f3) 返回的结果等于预期的 expected 值
        assert code_printer().doprint(f1*f2*f3) == expected
class TestTendonForceLengthDeGroote2016:

    @pytest.fixture(autouse=True)
    def _tendon_force_length_arguments_fixture(self):
        # 创建符号变量 l_T_tilde
        self.l_T_tilde = Symbol('l_T_tilde')
        # 创建符号变量 c_0, c_1, c_2, c_3，并将它们存储在元组中
        self.c0 = Symbol('c_0')
        self.c1 = Symbol('c_1')
        self.c2 = Symbol('c_2')
        self.c3 = Symbol('c_3')
        self.constants = (self.c0, self.c1, self.c2, self.c3)

    @staticmethod
    def test_class():
        # 断言 TendonForceLengthDeGroote2016 是 Function 的子类
        assert issubclass(TendonForceLengthDeGroote2016, Function)
        # 断言 TendonForceLengthDeGroote2016 是 CharacteristicCurveFunction 的子类
        assert issubclass(TendonForceLengthDeGroote2016, CharacteristicCurveFunction)
        # 断言 TendonForceLengthDeGroote2016 类的名称为 'TendonForceLengthDeGroote2016'
        assert TendonForceLengthDeGroote2016.__name__ == 'TendonForceLengthDeGroote2016'

    def test_instance(self):
        # 创建 TendonForceLengthDeGroote2016 类的实例 fl_T
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants)
        # 断言 fl_T 是 TendonForceLengthDeGroote2016 类的实例
        assert isinstance(fl_T, TendonForceLengthDeGroote2016)
        # 断言 fl_T 的字符串表示为 'TendonForceLengthDeGroote2016(l_T_tilde, c_0, c_1, c_2, c_3)'
        assert str(fl_T) == 'TendonForceLengthDeGroote2016(l_T_tilde, c_0, c_1, c_2, c_3)'

    def test_doit(self):
        # 调用 doit 方法计算 fl_T 的结果
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants).doit()
        # 断言计算结果与期望值相等
        assert fl_T == self.c0*exp(self.c3*(self.l_T_tilde - self.c1)) - self.c2

    def test_doit_evaluate_false(self):
        # 使用 evaluate=False 调用 doit 方法计算 fl_T 的结果
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants).doit(evaluate=False)
        # 断言计算结果与期望值相等
        assert fl_T == self.c0*exp(self.c3*UnevaluatedExpr(self.l_T_tilde - self.c1)) - self.c2

    def test_with_defaults(self):
        # 定义常数元组 constants
        constants = (
            Float('0.2'),
            Float('0.995'),
            Float('0.25'),
            Float('33.93669377311689'),
        )
        # 创建两个 TendonForceLengthDeGroote2016 的实例
        fl_T_manual = TendonForceLengthDeGroote2016(self.l_T_tilde, *constants)
        fl_T_constants = TendonForceLengthDeGroote2016.with_defaults(self.l_T_tilde)
        # 断言两个实例相等
        assert fl_T_manual == fl_T_constants

    def test_differentiate_wrt_l_T_tilde(self):
        # 创建 TendonForceLengthDeGroote2016 类的实例 fl_T
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants)
        # 计算 fl_T 对 l_T_tilde 的偏导数，并设置期望值
        expected = self.c0*self.c3*exp(self.c3*UnevaluatedExpr(-self.c1 + self.l_T_tilde))
        # 断言计算结果与期望值相等
        assert fl_T.diff(self.l_T_tilde) == expected

    def test_differentiate_wrt_c0(self):
        # 创建 TendonForceLengthDeGroote2016 类的实例 fl_T
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants)
        # 计算 fl_T 对 c0 的偏导数，并设置期望值
        expected = exp(self.c3*UnevaluatedExpr(-self.c1 + self.l_T_tilde))
        # 断言计算结果与期望值相等
        assert fl_T.diff(self.c0) == expected

    def test_differentiate_wrt_c1(self):
        # 创建 TendonForceLengthDeGroote2016 类的实例 fl_T
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants)
        # 计算 fl_T 对 c1 的偏导数，并设置期望值
        expected = -self.c0*self.c3*exp(self.c3*UnevaluatedExpr(self.l_T_tilde - self.c1))
        # 断言计算结果与期望值相等
        assert fl_T.diff(self.c1) == expected

    def test_differentiate_wrt_c2(self):
        # 创建 TendonForceLengthDeGroote2016 类的实例 fl_T
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants)
        # 计算 fl_T 对 c2 的偏导数，并设置期望值
        expected = Integer(-1)
        # 断言计算结果与期望值相等
        assert fl_T.diff(self.c2) == expected

    def test_differentiate_wrt_c3(self):
        # 创建 TendonForceLengthDeGroote2016 类的实例 fl_T
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants)
        # 计算 fl_T 对 c3 的偏导数，并设置期望值
        expected = self.c0*(self.l_T_tilde - self.c1)*exp(self.c3*UnevaluatedExpr(self.l_T_tilde - self.c1))
        # 断言计算结果与期望值相等
        assert fl_T.diff(self.c3) == expected
    # 定义一个测试方法，用于测试 TendonForceLengthDeGroote2016 类的 inverse 方法
    def test_inverse(self):
        # 创建 TendonForceLengthDeGroote2016 的实例 fl_T，使用给定的 self.l_T_tilde 和 self.constants 参数
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants)
        # 断言调用 fl_T 的 inverse 方法返回的结果是 TendonForceLengthInverseDeGroote2016 类的对象
        assert fl_T.inverse() is TendonForceLengthInverseDeGroote2016

    # 定义一个测试方法，用于测试 TendonForceLengthDeGroote2016 类的打印 Latex 格式的输出
    def test_function_print_latex(self):
        # 创建 TendonForceLengthDeGroote2016 的实例 fl_T，使用给定的 self.l_T_tilde 和 self.constants 参数
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants)
        # 期望的 Latex 输出
        expected = r'\operatorname{fl}^T \left( l_{T tilde} \right)'
        # 断言调用 LatexPrinter().doprint(fl_T) 返回的结果与期望值 expected 相等
        assert LatexPrinter().doprint(fl_T) == expected

    # 定义一个测试方法，用于测试 TendonForceLengthDeGroote2016 类的 doit 方法生成的表达式的 Latex 输出
    def test_expression_print_latex(self):
        # 创建 TendonForceLengthDeGroote2016 的实例 fl_T，使用给定的 self.l_T_tilde 和 self.constants 参数
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants)
        # 期望的 Latex 输出
        expected = r'c_{0} e^{c_{3} \left(- c_{1} + l_{T tilde}\right)} - c_{2}'
        # 断言调用 LatexPrinter().doprint(fl_T.doit()) 返回的结果与期望值 expected 相等
        assert LatexPrinter().doprint(fl_T.doit()) == expected
    @pytest.mark.parametrize(
        'code_printer, expected',
        [  # 参数化测试：定义不同的代码打印器和预期输出
            (
                C89CodePrinter,  # 使用 C89 代码打印器
                '(-0.25 + 0.20000000000000001*exp(33.93669377311689*(l_T_tilde - 0.995)))',  # 预期输出字符串
            ),
            (
                C99CodePrinter,  # 使用 C99 代码打印器
                '(-0.25 + 0.20000000000000001*exp(33.93669377311689*(l_T_tilde - 0.995)))',  # 预期输出字符串
            ),
            (
                C11CodePrinter,  # 使用 C11 代码打印器
                '(-0.25 + 0.20000000000000001*exp(33.93669377311689*(l_T_tilde - 0.995)))',  # 预期输出字符串
            ),
            (
                CXX98CodePrinter,  # 使用 C++98 代码打印器
                '(-0.25 + 0.20000000000000001*exp(33.93669377311689*(l_T_tilde - 0.995)))',  # 预期输出字符串
            ),
            (
                CXX11CodePrinter,  # 使用 C++11 代码打印器
                '(-0.25 + 0.20000000000000001*std::exp(33.93669377311689*(l_T_tilde - 0.995)))',  # 预期输出字符串
            ),
            (
                CXX17CodePrinter,  # 使用 C++17 代码打印器
                '(-0.25 + 0.20000000000000001*std::exp(33.93669377311689*(l_T_tilde - 0.995)))',  # 预期输出字符串
            ),
            (
                FCodePrinter,  # 使用 Fortran 代码打印器
                '      (-0.25d0 + 0.2d0*exp(33.93669377311689d0*(l_T_tilde - 0.995d0)))',  # 预期输出字符串
            ),
            (
                OctaveCodePrinter,  # 使用 Octave 代码打印器
                '(-0.25 + 0.2*exp(33.93669377311689*(l_T_tilde - 0.995)))',  # 预期输出字符串
            ),
            (
                PythonCodePrinter,  # 使用 Python 代码打印器
                '(-0.25 + 0.2*math.exp(33.93669377311689*(l_T_tilde - 0.995)))',  # 预期输出字符串
            ),
            (
                NumPyPrinter,  # 使用 NumPy 代码打印器
                '(-0.25 + 0.2*numpy.exp(33.93669377311689*(l_T_tilde - 0.995)))',  # 预期输出字符串
            ),
            (
                SciPyPrinter,  # 使用 SciPy 代码打印器
                '(-0.25 + 0.2*numpy.exp(33.93669377311689*(l_T_tilde - 0.995)))',  # 预期输出字符串
            ),
            (
                CuPyPrinter,  # 使用 CuPy 代码打印器
                '(-0.25 + 0.2*cupy.exp(33.93669377311689*(l_T_tilde - 0.995)))',  # 预期输出字符串
            ),
            (
                JaxPrinter,  # 使用 Jax 代码打印器
                '(-0.25 + 0.2*jax.numpy.exp(33.93669377311689*(l_T_tilde - 0.995)))',  # 预期输出字符串
            ),
            (
                MpmathPrinter,  # 使用 Mpmath 代码打印器
                '(mpmath.mpf((1, 1, -2, 1)) + mpmath.mpf((0, 3602879701896397, -54, 52))'
                '*mpmath.exp(mpmath.mpf((0, 9552330089424741, -48, 54))*(l_T_tilde + '
                'mpmath.mpf((1, 8962163258467287, -53, 53)))))',  # 预期输出字符串
            ),
            (
                LambdaPrinter,  # 使用 Lambda 代码打印器
                '(-0.25 + 0.2*math.exp(33.93669377311689*(l_T_tilde - 0.995)))',  # 预期输出字符串
            ),
        ]
    )
    def test_print_code(self, code_printer, expected):
        fl_T = TendonForceLengthDeGroote2016.with_defaults(self.l_T_tilde)  # 创建默认参数的 TendonForceLengthDeGroote2016 对象
        assert code_printer().doprint(fl_T) == expected  # 断言使用指定的代码打印器打印的结果等于预期输出字符串

    def test_derivative_print_code(self):
        fl_T = TendonForceLengthDeGroote2016.with_defaults(self.l_T_tilde)  # 创建默认参数的 TendonForceLengthDeGroote2016 对象
        dfl_T_dl_T_tilde = fl_T.diff(self.l_T_tilde)  # 计算相对于 l_T_tilde 的导数
        expected = '6.787338754623378*math.exp(33.93669377311689*(l_T_tilde - 0.995))'  # 预期输出字符串
        assert PythonCodePrinter().doprint(dfl_T_dl_T_tilde) == expected  # 断言使用 Python 代码打印器打印的导数结果等于预期输出字符串
    # 定义测试方法，用于测试 lambdify 函数的功能
    def test_lambdify(self):
        # 创建 TendonForceLengthDeGroote2016 模型对象，并使用默认参数 self.l_T_tilde 进行配置
        fl_T = TendonForceLengthDeGroote2016.with_defaults(self.l_T_tilde)
        # 将 fl_T 转换为可调用函数 fl_T_callable，接受 self.l_T_tilde 作为输入参数
        fl_T_callable = lambdify(self.l_T_tilde, fl_T)
        # 断言调用 fl_T_callable(1.0) 返回的结果接近于 -0.013014055039221595
        assert fl_T_callable(1.0) == pytest.approx(-0.013014055039221595)
    
    # 使用 pytest 装饰器标记跳过测试，条件为 NumPy 未安装
    @pytest.mark.skipif(numpy is None, reason='NumPy not installed')
    def test_lambdify_numpy(self):
        # 创建 TendonForceLengthDeGroote2016 模型对象，并使用默认参数 self.l_T_tilde 进行配置
        fl_T = TendonForceLengthDeGroote2016.with_defaults(self.l_T_tilde)
        # 将 fl_T 转换为 NumPy 可调用函数 fl_T_callable，接受 self.l_T_tilde 作为输入参数
        fl_T_callable = lambdify(self.l_T_tilde, fl_T, 'numpy')
        # 创建 NumPy 数组 l_T_tilde，包含 [0.95, 1.0, 1.01, 1.05] 四个元素
        l_T_tilde = numpy.array([0.95, 1.0, 1.01, 1.05])
        # 创建 NumPy 数组 expected，包含 [-0.2065693181344816, -0.0130140550392216, 0.0827421191989246, 1.04314889144172] 四个元素
        expected = numpy.array([
            -0.2065693181344816,
            -0.0130140550392216,
            0.0827421191989246,
            1.04314889144172,
        ])
        # 使用 NumPy.testing.assert_allclose 断言 fl_T_callable(l_T_tilde) 返回的结果与 expected 数组在精度范围内相等
        numpy.testing.assert_allclose(fl_T_callable(l_T_tilde), expected)
    
    # 使用 pytest 装饰器标记跳过测试，条件为 JAX 未安装
    @pytest.mark.skipif(jax is None, reason='JAX not installed')
    def test_lambdify_jax(self):
        # 创建 TendonForceLengthDeGroote2016 模型对象，并使用默认参数 self.l_T_tilde 进行配置
        fl_T = TendonForceLengthDeGroote2016.with_defaults(self.l_T_tilde)
        # 使用 JAX 提供的 jax.jit 将 fl_T 转换为 JAX 可加速的 fl_T_callable 函数，接受 self.l_T_tilde 作为输入参数
        fl_T_callable = jax.jit(lambdify(self.l_T_tilde, fl_T, 'jax'))
        # 创建 JAX 数组 l_T_tilde，包含 [0.95, 1.0, 1.01, 1.05] 四个元素
        l_T_tilde = jax.numpy.array([0.95, 1.0, 1.01, 1.05])
        # 创建 JAX 数组 expected，包含 [-0.2065693181344816, -0.0130140550392216, 0.0827421191989246, 1.04314889144172] 四个元素
        expected = jax.numpy.array([
            -0.2065693181344816,
            -0.0130140550392216,
            0.0827421191989246,
            1.04314889144172,
        ])
        # 使用 NumPy.testing.assert_allclose 断言 fl_T_callable(l_T_tilde) 返回的结果与 expected 数组在精度范围内相等
        numpy.testing.assert_allclose(fl_T_callable(l_T_tilde), expected)
# 定义一个测试类 TestTendonForceLengthInverseDeGroote2016，用于测试 TendonForceLengthInverseDeGroote2016 类
class TestTendonForceLengthInverseDeGroote2016:

    # pytest fixture，用于设置测试环境
    @pytest.fixture(autouse=True)
    def _tendon_force_length_inverse_arguments_fixture(self):
        # 创建符号对象 fl_T，c0, c1, c2, c3，并将它们赋值给实例的属性
        self.fl_T = Symbol('fl_T')
        self.c0 = Symbol('c_0')
        self.c1 = Symbol('c_1')
        self.c2 = Symbol('c_2')
        self.c3 = Symbol('c_3')
        self.constants = (self.c0, self.c1, self.c2, self.c3)

    # 静态方法测试类的继承关系
    @staticmethod
    def test_class():
        # 断言 TendonForceLengthInverseDeGroote2016 是 Function 的子类
        assert issubclass(TendonForceLengthInverseDeGroote2016, Function)
        # 断言 TendonForceLengthInverseDeGroote2016 是 CharacteristicCurveFunction 的子类
        assert issubclass(TendonForceLengthInverseDeGroote2016, CharacteristicCurveFunction)
        # 断言 TendonForceLengthInverseDeGroote2016 类的名称是 'TendonForceLengthInverseDeGroote2016'
        assert TendonForceLengthInverseDeGroote2016.__name__ == 'TendonForceLengthInverseDeGroote2016'

    # 测试实例化一个 TendonForceLengthInverseDeGroote2016 对象
    def test_instance(self):
        # 创建 TendonForceLengthInverseDeGroote2016 的实例 fl_T_inv，并进行断言
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants)
        assert isinstance(fl_T_inv, TendonForceLengthInverseDeGroote2016)
        assert str(fl_T_inv) == 'TendonForceLengthInverseDeGroote2016(fl_T, c_0, c_1, c_2, c_3)'

    # 测试执行 doit() 方法
    def test_doit(self):
        # 创建 TendonForceLengthInverseDeGroote2016 的实例 fl_T_inv，并调用其 doit() 方法
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants).doit()
        # 断言 doit() 方法返回的结果符合预期
        assert fl_T_inv == log((self.fl_T + self.c2)/self.c0)/self.c3 + self.c1

    # 测试执行 doit(evaluate=False) 方法
    def test_doit_evaluate_false(self):
        # 创建 TendonForceLengthInverseDeGroote2016 的实例 fl_T_inv，并调用其 doit(evaluate=False) 方法
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants).doit(evaluate=False)
        # 断言 doit(evaluate=False) 方法返回的结果符合预期
        assert fl_T_inv == log(UnevaluatedExpr((self.fl_T + self.c2)/self.c0))/self.c3 + self.c1

    # 测试使用默认常数的情况
    def test_with_defaults(self):
        # 创建使用手动常数的 TendonForceLengthInverseDeGroote2016 实例 fl_T_inv_manual
        constants = (
            Float('0.2'),
            Float('0.995'),
            Float('0.25'),
            Float('33.93669377311689'),
        )
        fl_T_inv_manual = TendonForceLengthInverseDeGroote2016(self.fl_T, *constants)
        # 创建使用默认常数的 TendonForceLengthInverseDeGroote2016 实例 fl_T_inv_constants
        fl_T_inv_constants = TendonForceLengthInverseDeGroote2016.with_defaults(self.fl_T)
        # 断言两个实例相等
        assert fl_T_inv_manual == fl_T_inv_constants

    # 测试对 fl_T 进行求导
    def test_differentiate_wrt_fl_T(self):
        # 创建 TendonForceLengthInverseDeGroote2016 的实例 fl_T_inv
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants)
        # 计算对 fl_T 求导的期望值
        expected = 1/(self.c3*(self.fl_T + self.c2))
        # 断言 fl_T_inv 对 fl_T 求导的结果符合预期
        assert fl_T_inv.diff(self.fl_T) == expected

    # 测试对 c0 进行求导
    def test_differentiate_wrt_c0(self):
        # 创建 TendonForceLengthInverseDeGroote2016 的实例 fl_T_inv
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants)
        # 计算对 c0 求导的期望值
        expected = -1/(self.c0*self.c3)
        # 断言 fl_T_inv 对 c0 求导的结果符合预期
        assert fl_T_inv.diff(self.c0) == expected

    # 测试对 c1 进行求导
    def test_differentiate_wrt_c1(self):
        # 创建 TendonForceLengthInverseDeGroote2016 的实例 fl_T_inv
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants)
        # 计算对 c1 求导的期望值
        expected = Integer(1)
        # 断言 fl_T_inv 对 c1 求导的结果符合预期
        assert fl_T_inv.diff(self.c1) == expected

    # 测试对 c2 进行求导
    def test_differentiate_wrt_c2(self):
        # 创建 TendonForceLengthInverseDeGroote2016 的实例 fl_T_inv
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants)
        # 计算对 c2 求导的期望值
        expected = 1/(self.c3*(self.fl_T + self.c2))
        # 断言 fl_T_inv 对 c2 求导的结果符合预期
        assert fl_T_inv.diff(self.c2) == expected

    # 测试对 c3 进行求导
    def test_differentiate_wrt_c3(self):
        # 创建 TendonForceLengthInverseDeGroote2016 的实例 fl_T_inv
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants)
        # 计算对 c3 求导的期望值
        expected = -log(UnevaluatedExpr((self.fl_T + self.c2)/self.c0))/self.c3**2
        # 断言 fl_T_inv 对 c3 求导的结果符合预期
        assert fl_T_inv.diff(self.c3) == expected
    # 定义测试函数，验证 TendonForceLengthInverseDeGroote2016 类的逆函数是否正确
    def test_inverse(self):
        # 创建 TendonForceLengthInverseDeGroote2016 对象 fl_T_inv，使用 self.fl_T 和 self.constants 作为参数
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants)
        # 断言逆函数 inverse() 返回的对象是否是 TendonForceLengthDeGroote2016 类型
        assert fl_T_inv.inverse() is TendonForceLengthDeGroote2016

    # 定义测试函数，验证 LatexPrinter 是否正确打印 fl_T_inv 的 LaTeX 表达式
    def test_function_print_latex(self):
        # 创建 TendonForceLengthInverseDeGroote2016 对象 fl_T_inv，使用 self.fl_T 和 self.constants 作为参数
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants)
        # 期望的 LaTeX 表达式
        expected = r'\left( \operatorname{fl}^T \right)^{-1} \left( fl_{T} \right)'
        # 断言 LatexPrinter().doprint(fl_T_inv) 的输出是否与 expected 相等
        assert LatexPrinter().doprint(fl_T_inv) == expected

    # 定义测试函数，验证 LatexPrinter 是否正确打印 fl_T 的 do-it 表达式的 LaTeX 表达式
    def test_expression_print_latex(self):
        # 创建 TendonForceLengthInverseDeGroote2016 对象 fl_T，使用 self.fl_T 和 self.constants 作为参数
        fl_T = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants)
        # 期望的 LaTeX 表达式
        expected = r'c_{1} + \frac{\log{\left(\frac{c_{2} + fl_{T}}{c_{0}} \right)}}{c_{3}}'
        # 断言 LatexPrinter().doprint(fl_T.doit()) 的输出是否与 expected 相等
        assert LatexPrinter().doprint(fl_T.doit()) == expected

    # 使用 pytest.mark.parametrize 装饰器定义参数化测试
    @pytest.mark.parametrize(
        'code_printer, expected',
        [
            (
                C89CodePrinter,
                '(0.995 + 0.029466630034306838*log(5.0*fl_T + 1.25))',
            ),
            (
                C99CodePrinter,
                '(0.995 + 0.029466630034306838*log(5.0*fl_T + 1.25))',
            ),
            (
                C11CodePrinter,
                '(0.995 + 0.029466630034306838*log(5.0*fl_T + 1.25))',
            ),
            (
                CXX98CodePrinter,
                '(0.995 + 0.029466630034306838*log(5.0*fl_T + 1.25))',
            ),
            (
                CXX11CodePrinter,
                '(0.995 + 0.029466630034306838*std::log(5.0*fl_T + 1.25))',
            ),
            (
                CXX17CodePrinter,
                '(0.995 + 0.029466630034306838*std::log(5.0*fl_T + 1.25))',
            ),
            (
                FCodePrinter,
                '      (0.995d0 + 0.02946663003430684d0*log(5.0d0*fl_T + 1.25d0))',
            ),
            (
                OctaveCodePrinter,
                '(0.995 + 0.02946663003430684*log(5.0*fl_T + 1.25))',
            ),
            (
                PythonCodePrinter,
                '(0.995 + 0.02946663003430684*math.log(5.0*fl_T + 1.25))',
            ),
            (
                NumPyPrinter,
                '(0.995 + 0.02946663003430684*numpy.log(5.0*fl_T + 1.25))',
            ),
            (
                SciPyPrinter,
                '(0.995 + 0.02946663003430684*numpy.log(5.0*fl_T + 1.25))',
            ),
            (
                CuPyPrinter,
                '(0.995 + 0.02946663003430684*cupy.log(5.0*fl_T + 1.25))',
            ),
            (
                JaxPrinter,
                '(0.995 + 0.02946663003430684*jax.numpy.log(5.0*fl_T + 1.25))',
            ),
            (
                MpmathPrinter,
                '(mpmath.mpf((0, 8962163258467287, -53, 53))'
                ' + mpmath.mpf((0, 33972711434846347, -60, 55))'
                '*mpmath.log(mpmath.mpf((0, 5, 0, 3))*fl_T + mpmath.mpf((0, 5, -2, 3))))',
            ),
            (
                LambdaPrinter,
                '(0.995 + 0.02946663003430684*math.log(5.0*fl_T + 1.25))',
            ),
        ]
    )
    # 定义一个测试方法，用于测试代码打印器的输出是否符合预期
    def test_print_code(self, code_printer, expected):
        # 创建一个特定参数的 TendonForceLengthInverseDeGroote2016 对象
        fl_T_inv = TendonForceLengthInverseDeGroote2016.with_defaults(self.fl_T)
        # 断言使用指定的代码打印器打印结果是否与预期相等
        assert code_printer().doprint(fl_T_inv) == expected

    # 定义一个测试方法，用于测试导数代码的打印
    def test_derivative_print_code(self):
        # 创建一个特定参数的 TendonForceLengthInverseDeGroote2016 对象
        fl_T_inv = TendonForceLengthInverseDeGroote2016.with_defaults(self.fl_T)
        # 计算相对于 fl_T 的导数
        dfl_T_inv_dfl_T = fl_T_inv.diff(self.fl_T)
        # 预期的导数代码字符串
        expected = '1/(33.93669377311689*fl_T + 8.484173443279222)'
        # 断言使用 PythonCodePrinter 打印导数代码是否与预期相等
        assert PythonCodePrinter().doprint(dfl_T_inv_dfl_T) == expected

    # 定义一个测试方法，用于测试 lambdify 函数的使用
    def test_lambdify(self):
        # 创建一个特定参数的 TendonForceLengthInverseDeGroote2016 对象
        fl_T_inv = TendonForceLengthInverseDeGroote2016.with_defaults(self.fl_T)
        # 将对象转换为可调用的函数
        fl_T_inv_callable = lambdify(self.fl_T, fl_T_inv)
        # 断言调用转换后的函数并验证结果是否接近预期值
        assert fl_T_inv_callable(0.0) == pytest.approx(1.0015752885)

    # 使用 pytest.mark.skipif 装饰器标记的测试方法，用于测试 lambdify 函数与 NumPy 的结合使用
    @pytest.mark.skipif(numpy is None, reason='NumPy not installed')
    def test_lambdify_numpy(self):
        # 创建一个特定参数的 TendonForceLengthInverseDeGroote2016 对象
        fl_T_inv = TendonForceLengthInverseDeGroote2016.with_defaults(self.fl_T)
        # 将对象转换为可调用的函数，使用 NumPy 模式
        fl_T_inv_callable = lambdify(self.fl_T, fl_T_inv, 'numpy')
        # 创建 NumPy 数组作为输入
        fl_T = numpy.array([-0.2, -0.01, 0.0, 1.01, 1.02, 1.05])
        # 预期的输出结果数组
        expected = numpy.array([
            0.9541505769,
            1.0003724019,
            1.0015752885,
            1.0492347951,
            1.0494677341,
            1.0501557022,
        ])
        # 使用 NumPy 的 assert_allclose 函数断言结果数组是否接近预期值
        numpy.testing.assert_allclose(fl_T_inv_callable(fl_T), expected)

    # 使用 pytest.mark.skipif 装饰器标记的测试方法，用于测试 lambdify 函数与 JAX 的结合使用
    @pytest.mark.skipif(jax is None, reason='JAX not installed')
    def test_lambdify_jax(self):
        # 创建一个特定参数的 TendonForceLengthInverseDeGroote2016 对象
        fl_T_inv = TendonForceLengthInverseDeGroote2016.with_defaults(self.fl_T)
        # 将对象转换为可调用的函数，使用 JAX 模式
        fl_T_inv_callable = jax.jit(lambdify(self.fl_T, fl_T_inv, 'jax'))
        # 创建 JAX 数组作为输入
        fl_T = jax.numpy.array([-0.2, -0.01, 0.0, 1.01, 1.02, 1.05])
        # 预期的输出结果数组
        expected = jax.numpy.array([
            0.9541505769,
            1.0003724019,
            1.0015752885,
            1.0492347951,
            1.0494677341,
            1.0501557022,
        ])
        # 使用 NumPy 的 assert_allclose 函数断言结果数组是否接近预期值
        numpy.testing.assert_allclose(fl_T_inv_callable(fl_T), expected)
# 定义测试类 TestFiberForceLengthPassiveDeGroote2016，用于测试 FiberForceLengthPassiveDeGroote2016 类
class TestFiberForceLengthPassiveDeGroote2016:

    # 设置测试用例的固定参数 fixture，自动使用
    @pytest.fixture(autouse=True)
    def _fiber_force_length_passive_arguments_fixture(self):
        # 创建符号 l_M_tilde
        self.l_M_tilde = Symbol('l_M_tilde')
        # 创建符号 c_0
        self.c0 = Symbol('c_0')
        # 创建符号 c_1
        self.c1 = Symbol('c_1')
        # 将常数符号保存在元组 constants 中
        self.constants = (self.c0, self.c1)

    # 静态方法测试类的基类是否为 Function
    @staticmethod
    def test_class():
        assert issubclass(FiberForceLengthPassiveDeGroote2016, Function)
        # 测试类是否为 CharacteristicCurveFunction 的子类
        assert issubclass(FiberForceLengthPassiveDeGroote2016, CharacteristicCurveFunction)
        # 测试类名是否为 'FiberForceLengthPassiveDeGroote2016'
        assert FiberForceLengthPassiveDeGroote2016.__name__ == 'FiberForceLengthPassiveDeGroote2016'

    # 测试实例化对象的类型和字符串表示是否符合预期
    def test_instance(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *self.constants)
        assert isinstance(fl_M_pas, FiberForceLengthPassiveDeGroote2016)
        assert str(fl_M_pas) == 'FiberForceLengthPassiveDeGroote2016(l_M_tilde, c_0, c_1)'

    # 测试 doit 方法的计算结果是否符合预期
    def test_doit(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *self.constants).doit()
        assert fl_M_pas == (exp((self.c1*(self.l_M_tilde - 1))/self.c0) - 1)/(exp(self.c1) - 1)

    # 测试 doit 方法中 evaluate=False 选项的计算结果是否符合预期
    def test_doit_evaluate_false(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *self.constants).doit(evaluate=False)
        assert fl_M_pas == (exp((self.c1*UnevaluatedExpr(self.l_M_tilde - 1))/self.c0) - 1)/(exp(self.c1) - 1)

    # 测试使用指定默认常数的对象是否与手动创建的对象相等
    def test_with_defaults(self):
        constants = (
            Float('0.6'),
            Float('4.0'),
        )
        fl_M_pas_manual = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *constants)
        fl_M_pas_constants = FiberForceLengthPassiveDeGroote2016.with_defaults(self.l_M_tilde)
        assert fl_M_pas_manual == fl_M_pas_constants

    # 测试对 l_M_tilde 求导数的结果是否符合预期
    def test_differentiate_wrt_l_M_tilde(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = self.c1*exp(self.c1*UnevaluatedExpr(self.l_M_tilde - 1)/self.c0)/(self.c0*(exp(self.c1) - 1))
        assert fl_M_pas.diff(self.l_M_tilde) == expected

    # 测试对 c0 求导数的结果是否符合预期
    def test_differentiate_wrt_c0(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = (
            -self.c1*exp(self.c1*UnevaluatedExpr(self.l_M_tilde - 1)/self.c0)
            *UnevaluatedExpr(self.l_M_tilde - 1)/(self.c0**2*(exp(self.c1) - 1))
        )
        assert fl_M_pas.diff(self.c0) == expected

    # 测试对 c1 求导数的结果是否符合预期
    def test_differentiate_wrt_c1(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = (
            -exp(self.c1)*(-1 + exp(self.c1*UnevaluatedExpr(self.l_M_tilde - 1)/self.c0))/(exp(self.c1) - 1)**2
            + exp(self.c1*UnevaluatedExpr(self.l_M_tilde - 1)/self.c0)*(self.l_M_tilde - 1)/(self.c0*(exp(self.c1) - 1))
        )
        assert fl_M_pas.diff(self.c1) == expected
    # 定义测试函数，用于测试 FiberForceLengthPassiveDeGroote2016 类的逆函数方法
    def test_inverse(self):
        # 创建 FiberForceLengthPassiveDeGroote2016 类的实例 fl_M_pas，传入 l_M_tilde 和常量参数
        fl_M_pas = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *self.constants)
        # 断言调用 fl_M_pas 的 inverse 方法返回值为 FiberForceLengthPassiveInverseDeGroote2016
        assert fl_M_pas.inverse() is FiberForceLengthPassiveInverseDeGroote2016

    # 定义测试函数，用于测试 FiberForceLengthPassiveDeGroote2016 类的 LaTeX 打印功能
    def test_function_print_latex(self):
        # 创建 FiberForceLengthPassiveDeGroote2016 类的实例 fl_M_pas，传入 l_M_tilde 和常量参数
        fl_M_pas = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *self.constants)
        # 预期的 LaTeX 字符串
        expected = r'\operatorname{fl}^M_{pas} \left( l_{M tilde} \right)'
        # 断言使用 LatexPrinter 对象的 doprint 方法处理 fl_M_pas 后的输出与预期字符串相等
        assert LatexPrinter().doprint(fl_M_pas) == expected

    # 定义测试函数，用于测试 FiberForceLengthPassiveDeGroote2016 类的表达式 LaTeX 打印功能
    def test_expression_print_latex(self):
        # 创建 FiberForceLengthPassiveDeGroote2016 类的实例 fl_M_pas，传入 l_M_tilde 和常量参数
        fl_M_pas = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *self.constants)
        # 预期的 LaTeX 字符串
        expected = r'\frac{e^{\frac{c_{1} \left(l_{M tilde} - 1\right)}{c_{0}}} - 1}{e^{c_{1}} - 1}'
        # 断言使用 LatexPrinter 对象的 doprint 方法处理 fl_M_pas 的 doit 方法后的输出与预期字符串相等
        assert LatexPrinter().doprint(fl_M_pas.doit()) == expected
    # 使用 pytest 的 @pytest.mark.parametrize 装饰器标记参数化测试
    @pytest.mark.parametrize(
        'code_printer, expected',  # 参数化测试的参数，包括 code_printer 和 expected
        [
            (
                C89CodePrinter,  # 使用 C89CodePrinter 来打印代码
                '(0.01865736036377405*(-1 + exp(6.666666666666667*(l_M_tilde - 1))))',  # 预期的打印结果
            ),
            (
                C99CodePrinter,  # 使用 C99CodePrinter 来打印代码
                '(0.01865736036377405*(-1 + exp(6.666666666666667*(l_M_tilde - 1))))',  # 预期的打印结果
            ),
            (
                C11CodePrinter,  # 使用 C11CodePrinter 来打印代码
                '(0.01865736036377405*(-1 + exp(6.666666666666667*(l_M_tilde - 1))))',  # 预期的打印结果
            ),
            (
                CXX98CodePrinter,  # 使用 CXX98CodePrinter 来打印代码
                '(0.01865736036377405*(-1 + exp(6.666666666666667*(l_M_tilde - 1))))',  # 预期的打印结果
            ),
            (
                CXX11CodePrinter,  # 使用 CXX11CodePrinter 来打印代码
                '(0.01865736036377405*(-1 + std::exp(6.666666666666667*(l_M_tilde - 1))))',  # 预期的打印结果
            ),
            (
                CXX17CodePrinter,  # 使用 CXX17CodePrinter 来打印代码
                '(0.01865736036377405*(-1 + std::exp(6.666666666666667*(l_M_tilde - 1))))',  # 预期的打印结果
            ),
            (
                FCodePrinter,  # 使用 FCodePrinter 来打印代码
                '      (0.0186573603637741d0*(-1 + exp(6.666666666666667d0*(l_M_tilde - 1\n'  # 预期的打印结果
                '     @ )))',  # 预期的打印结果的延续部分，以 @ 为标记
            ),
            (
                OctaveCodePrinter,  # 使用 OctaveCodePrinter 来打印代码
                '(0.0186573603637741*(-1 + exp(6.66666666666667*(l_M_tilde - 1))))',  # 预期的打印结果
            ),
            (
                PythonCodePrinter,  # 使用 PythonCodePrinter 来打印代码
                '(0.0186573603637741*(-1 + math.exp(6.66666666666667*(l_M_tilde - 1))))',  # 预期的打印结果
            ),
            (
                NumPyPrinter,  # 使用 NumPyPrinter 来打印代码
                '(0.0186573603637741*(-1 + numpy.exp(6.66666666666667*(l_M_tilde - 1))))',  # 预期的打印结果
            ),
            (
                SciPyPrinter,  # 使用 SciPyPrinter 来打印代码
                '(0.0186573603637741*(-1 + numpy.exp(6.66666666666667*(l_M_tilde - 1))))',  # 预期的打印结果
            ),
            (
                CuPyPrinter,  # 使用 CuPyPrinter 来打印代码
                '(0.0186573603637741*(-1 + cupy.exp(6.66666666666667*(l_M_tilde - 1))))',  # 预期的打印结果
            ),
            (
                JaxPrinter,  # 使用 JaxPrinter 来打印代码
                '(0.0186573603637741*(-1 + jax.numpy.exp(6.66666666666667*(l_M_tilde - 1))))',  # 预期的打印结果
            ),
            (
                MpmathPrinter,  # 使用 MpmathPrinter 来打印代码
                '(mpmath.mpf((0, 672202249456079, -55, 50))*(-1 + mpmath.exp('  # 预期的打印结果
                'mpmath.mpf((0, 7505999378950827, -50, 53))*(l_M_tilde - 1))))',  # 预期的打印结果的延续部分
            ),
            (
                LambdaPrinter,  # 使用 LambdaPrinter 来打印代码
                '(0.0186573603637741*(-1 + math.exp(6.66666666666667*(l_M_tilde - 1))))',  # 预期的打印结果
            ),
        ]
    )
    # 测试打印代码的函数，验证各种打印器的输出是否符合预期
    def test_print_code(self, code_printer, expected):
        # 使用 FiberForceLengthPassiveDeGroote2016 类的默认参数 fl_M_pas 初始化对象
        fl_M_pas = FiberForceLengthPassiveDeGroote2016.with_defaults(self.l_M_tilde)
        # 断言使用当前的 code_printer 打印 fl_M_pas 对象的结果与 expected 相等
        assert code_printer().doprint(fl_M_pas) == expected

    # 测试打印代码的导数函数
    def test_derivative_print_code(self):
        # 使用 FiberForceLengthPassiveDeGroote2016 类的默认参数 fl_M_pas 初始化对象
        fl_M_pas = FiberForceLengthPassiveDeGroote2016.with_defaults(self.l_M_tilde)
        # 计算 fl_M_pas 对象关于 l_M_tilde 的导数
        fl_M_pas_dl_M_tilde = fl_M_pas.diff(self.l_M_tilde)
        # 预期的导数打印结果
        expected = '0.12438240242516*math.exp(6.66666666666667*(l_M_tilde - 1))'
        # 断言使用 PythonCodePrinter 打印 fl_M_pas_dl_M_tilde 的结果与 expected 相等
        assert PythonCodePrinter().doprint(fl_M_pas_dl_M_tilde) == expected
    # 定义一个测试方法，测试 lambdify 函数是否能正确处理 FiberForceLengthPassiveDeGroote2016 模型
    def test_lambdify(self):
        # 使用默认参数创建 FiberForceLengthPassiveDeGroote2016 实例
        fl_M_pas = FiberForceLengthPassiveDeGroote2016.with_defaults(self.l_M_tilde)
        # 将 fl_M_pas 转换为可调用的函数 fl_M_pas_callable
        fl_M_pas_callable = lambdify(self.l_M_tilde, fl_M_pas)
        # 断言调用 fl_M_pas_callable(1.0) 的结果是否接近于 0.0
        assert fl_M_pas_callable(1.0) == pytest.approx(0.0)

    # 如果没有安装 NumPy，则跳过这个测试
    @pytest.mark.skipif(numpy is None, reason='NumPy not installed')
    def test_lambdify_numpy(self):
        # 使用默认参数创建 FiberForceLengthPassiveDeGroote2016 实例
        fl_M_pas = FiberForceLengthPassiveDeGroote2016.with_defaults(self.l_M_tilde)
        # 将 fl_M_pas 转换为可调用的函数 fl_M_pas_callable，使用 NumPy 作为目标平台
        fl_M_pas_callable = lambdify(self.l_M_tilde, fl_M_pas, 'numpy')
        # 创建 NumPy 数组 l_M_tilde 包含多个测试值
        l_M_tilde = numpy.array([0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5])
        # 预期的 fl_M_pas_callable(l_M_tilde) 结果
        expected = numpy.array([
            -0.0179917778,
            -0.0137393336,
            -0.0090783522,
            0.0,
            0.0176822155,
            0.0521224686,
            0.5043387669,
        ])
        # 使用 NumPy 的 assert_allclose 函数检查 fl_M_pas_callable(l_M_tilde) 是否接近于 expected
        numpy.testing.assert_allclose(fl_M_pas_callable(l_M_tilde), expected)

    # 如果没有安装 JAX，则跳过这个测试
    @pytest.mark.skipif(jax is None, reason='JAX not installed')
    def test_lambdify_jax(self):
        # 使用默认参数创建 FiberForceLengthPassiveDeGroote2016 实例
        fl_M_pas = FiberForceLengthPassiveDeGroote2016.with_defaults(self.l_M_tilde)
        # 将 fl_M_pas 转换为可调用的函数 fl_M_pas_callable，使用 JAX 作为目标平台，并使用 jax.jit 进行编译
        fl_M_pas_callable = jax.jit(lambdify(self.l_M_tilde, fl_M_pas, 'jax'))
        # 创建 JAX 数组 l_M_tilde 包含多个测试值
        l_M_tilde = jax.numpy.array([0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5])
        # 预期的 fl_M_pas_callable(l_M_tilde) 结果
        expected = jax.numpy.array([
            -0.0179917778,
            -0.0137393336,
            -0.0090783522,
            0.0,
            0.0176822155,
            0.0521224686,
            0.5043387669,
        ])
        # 使用 NumPy 的 assert_allclose 函数检查 fl_M_pas_callable(l_M_tilde) 是否接近于 expected
        numpy.testing.assert_allclose(fl_M_pas_callable(l_M_tilde), expected)
# 定义测试类 TestFiberForceLengthPassiveInverseDeGroote2016，用于测试 FiberForceLengthPassiveInverseDeGroote2016 类的功能
class TestFiberForceLengthPassiveInverseDeGroote2016:

    # 设置测试夹具，用于为测试方法提供必要的参数
    @pytest.fixture(autouse=True)
    def _fiber_force_length_passive_arguments_fixture(self):
        # 创建符号变量 fl_M_pas，并命名为 fl_M_pas
        self.fl_M_pas = Symbol('fl_M_pas')
        # 创建符号变量 c0，并命名为 c_0
        self.c0 = Symbol('c_0')
        # 创建符号变量 c1，并命名为 c_1
        self.c1 = Symbol('c_1')
        # 将 c0 和 c1 封装为元组常量
        self.constants = (self.c0, self.c1)

    # 测试类是否是 Function 的子类
    @staticmethod
    def test_class():
        assert issubclass(FiberForceLengthPassiveInverseDeGroote2016, Function)
        # 测试类是否是 CharacteristicCurveFunction 的子类
        assert issubclass(FiberForceLengthPassiveInverseDeGroote2016, CharacteristicCurveFunction)
        # 断言类的名称是否为 'FiberForceLengthPassiveInverseDeGroote2016'
        assert FiberForceLengthPassiveInverseDeGroote2016.__name__ == 'FiberForceLengthPassiveInverseDeGroote2016'

    # 测试实例化对象是否正确
    def test_instance(self):
        # 创建 FiberForceLengthPassiveInverseDeGroote2016 类的实例 fl_M_pas_inv
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *self.constants)
        # 断言实例是否是 FiberForceLengthPassiveInverseDeGroote2016 类的实例
        assert isinstance(fl_M_pas_inv, FiberForceLengthPassiveInverseDeGroote2016)
        # 断言实例的字符串表示是否符合预期
        assert str(fl_M_pas_inv) == 'FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, c_0, c_1)'

    # 测试 doit 方法的计算结果是否正确
    def test_doit(self):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *self.constants).doit()
        # 断言计算结果是否符合预期
        assert fl_M_pas_inv == self.c0*log(self.fl_M_pas*(exp(self.c1) - 1) + 1)/self.c1 + 1

    # 测试 doit 方法在 evaluate=False 时的计算结果是否正确
    def test_doit_evaluate_false(self):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *self.constants).doit(evaluate=False)
        # 断言计算结果是否符合预期，包括 UnevaluatedExpr 的处理
        assert fl_M_pas_inv == self.c0*log(UnevaluatedExpr(self.fl_M_pas*(exp(self.c1) - 1)) + 1)/self.c1 + 1

    # 测试带有默认常量参数的实例化是否正确
    def test_with_defaults(self):
        # 定义常量参数
        constants = (
            Float('0.6'),
            Float('4.0'),
        )
        # 创建手动设置常量参数的实例 fl_M_pas_inv_manual
        fl_M_pas_inv_manual = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *constants)
        # 调用 with_defaults 方法创建常量参数的实例 fl_M_pas_inv_constants
        fl_M_pas_inv_constants = FiberForceLengthPassiveInverseDeGroote2016.with_defaults(self.fl_M_pas)
        # 断言两个实例是否相等
        assert fl_M_pas_inv_manual == fl_M_pas_inv_constants

    # 测试对 fl_M_pas 的偏导数计算是否正确
    def test_differentiate_wrt_fl_T(self):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *self.constants)
        # 计算 fl_M_pas_inv 对 fl_M_pas 的偏导数，并断言结果是否符合预期
        expected = self.c0*(exp(self.c1) - 1)/(self.c1*(self.fl_M_pas*(exp(self.c1) - 1) + 1))
        assert fl_M_pas_inv.diff(self.fl_M_pas) == expected

    # 测试对 c0 的偏导数计算是否正确
    def test_differentiate_wrt_c0(self):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *self.constants)
        # 计算 fl_M_pas_inv 对 c0 的偏导数，并断言结果是否符合预期
        expected = log(self.fl_M_pas*(exp(self.c1) - 1) + 1)/self.c1
        assert fl_M_pas_inv.diff(self.c0) == expected

    # 测试对 c1 的偏导数计算是否正确
    def test_differentiate_wrt_c1(self):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *self.constants)
        # 计算 fl_M_pas_inv 对 c1 的偏导数，并断言结果是否符合预期
        expected = (
            self.c0*self.fl_M_pas*exp(self.c1)/(self.c1*(self.fl_M_pas*(exp(self.c1) - 1) + 1))
            - self.c0*log(self.fl_M_pas*(exp(self.c1) - 1) + 1)/self.c1**2
        )
        assert fl_M_pas_inv.diff(self.c1) == expected

    # 测试 inverse 方法是否正确返回 FiberForceLengthPassiveDeGroote2016 类
    def test_inverse(self):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *self.constants)
        # 断言 inverse 方法是否返回 FiberForceLengthPassiveDeGroote2016 类
        assert fl_M_pas_inv.inverse() is FiberForceLengthPassiveDeGroote2016
    # 定义一个测试函数，用于打印 LaTeX 格式的输出
    def test_function_print_latex(self):
        # 创建一个 FiberForceLengthPassiveInverseDeGroote2016 类的实例
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *self.constants)
        # 预期的 LaTeX 输出字符串
        expected = r'\left( \operatorname{fl}^M_{pas} \right)^{-1} \left( fl_{M pas} \right)'
        # 断言打印出的 LaTeX 字符串与预期相等
        assert LatexPrinter().doprint(fl_M_pas_inv) == expected

    # 定义另一个测试函数，用于打印表达式的 LaTeX 格式输出
    def test_expression_print_latex(self):
        # 创建一个 FiberForceLengthPassiveInverseDeGroote2016 类的实例
        fl_T = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *self.constants)
        # 预期的 LaTeX 输出字符串
        expected = r'\frac{c_{0} \log{\left(fl_{M pas} \left(e^{c_{1}} - 1\right) + 1 \right)}}{c_{1}} + 1'
        # 断言表达式计算并打印出的 LaTeX 字符串与预期相等
        assert LatexPrinter().doprint(fl_T.doit()) == expected

    # 使用 pytest 的 parametrize 装饰器进行参数化测试
    @pytest.mark.parametrize(
        'code_printer, expected',
        [
            (
                C89CodePrinter,
                '(1 + 0.14999999999999999*log(1 + 53.598150033144236*fl_M_pas))',
            ),
            (
                C99CodePrinter,
                '(1 + 0.14999999999999999*log(1 + 53.598150033144236*fl_M_pas))',
            ),
            (
                C11CodePrinter,
                '(1 + 0.14999999999999999*log(1 + 53.598150033144236*fl_M_pas))',
            ),
            (
                CXX98CodePrinter,
                '(1 + 0.14999999999999999*log(1 + 53.598150033144236*fl_M_pas))',
            ),
            (
                CXX11CodePrinter,
                '(1 + 0.14999999999999999*std::log(1 + 53.598150033144236*fl_M_pas))',
            ),
            (
                CXX17CodePrinter,
                '(1 + 0.14999999999999999*std::log(1 + 53.598150033144236*fl_M_pas))',
            ),
            (
                FCodePrinter,
                '      (1 + 0.15d0*log(1.0d0 + 53.5981500331442d0*fl_M_pas))',
            ),
            (
                OctaveCodePrinter,
                '(1 + 0.15*log(1 + 53.5981500331442*fl_M_pas))',
            ),
            (
                PythonCodePrinter,
                '(1 + 0.15*math.log(1 + 53.5981500331442*fl_M_pas))',
            ),
            (
                NumPyPrinter,
                '(1 + 0.15*numpy.log(1 + 53.5981500331442*fl_M_pas))',
            ),
            (
                SciPyPrinter,
                '(1 + 0.15*numpy.log(1 + 53.5981500331442*fl_M_pas))',
            ),
            (
                CuPyPrinter,
                '(1 + 0.15*cupy.log(1 + 53.5981500331442*fl_M_pas))',
            ),
            (
                JaxPrinter,
                '(1 + 0.15*jax.numpy.log(1 + 53.5981500331442*fl_M_pas))',
            ),
            (
                MpmathPrinter,
                '(1 + mpmath.mpf((0, 5404319552844595, -55, 53))*mpmath.log(1 '
                '+ mpmath.mpf((0, 942908627019595, -44, 50))*fl_M_pas))',
            ),
            (
                LambdaPrinter,
                '(1 + 0.15*math.log(1 + 53.5981500331442*fl_M_pas))',
            ),
        ]
    )
    # 定义一个测试方法，用于打印代码并验证输出结果是否符合预期
    def test_print_code(self, code_printer, expected):
        # 使用默认参数创建 FiberForceLengthPassiveInverseDeGroote2016 实例
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016.with_defaults(self.fl_M_pas)
        # 断言打印后的代码与期望的输出结果相等
        assert code_printer().doprint(fl_M_pas_inv) == expected

    # 定义一个测试方法，用于测试导数打印代码的功能
    def test_derivative_print_code(self):
        # 使用默认参数创建 FiberForceLengthPassiveInverseDeGroote2016 实例
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016.with_defaults(self.fl_M_pas)
        # 计算 fl_M_pas_inv 相对于 self.fl_M_pas 的导数
        dfl_M_pas_inv_dfl_T = fl_M_pas_inv.diff(self.fl_M_pas)
        # 预期的输出结果字符串
        expected = '32.1588900198865/(214.392600132577*fl_M_pas + 4.0)'
        # 断言导数打印后的代码与期望的输出结果相等
        assert PythonCodePrinter().doprint(dfl_M_pas_inv_dfl_T) == expected

    # 定义一个测试方法，用于测试 lambdify 函数的功能
    def test_lambdify(self):
        # 使用默认参数创建 FiberForceLengthPassiveInverseDeGroote2016 实例
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016.with_defaults(self.fl_M_pas)
        # 将 fl_M_pas_inv 转换为可调用的函数 fl_M_pas_inv_callable
        fl_M_pas_inv_callable = lambdify(self.fl_M_pas, fl_M_pas_inv)
        # 断言调用 fl_M_pas_inv_callable(0.0) 的结果近似等于 1.0
        assert fl_M_pas_inv_callable(0.0) == pytest.approx(1.0)

    # 标记为跳过，如果没有安装 NumPy
    @pytest.mark.skipif(numpy is None, reason='NumPy not installed')
    def test_lambdify_numpy(self):
        # 使用默认参数创建 FiberForceLengthPassiveInverseDeGroote2016 实例
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016.with_defaults(self.fl_M_pas)
        # 将 fl_M_pas_inv 转换为可调用的函数 fl_M_pas_inv_callable，使用 NumPy
        fl_M_pas_inv_callable = lambdify(self.fl_M_pas, fl_M_pas_inv, 'numpy')
        # 创建 NumPy 数组 fl_M_pas
        fl_M_pas = numpy.array([-0.01, 0.0, 0.01, 0.02, 0.05, 0.1])
        # 预期的输出结果 NumPy 数组
        expected = numpy.array([
            0.8848253714,
            1.0,
            1.0643754386,
            1.1092744701,
            1.1954331425,
            1.2774998934,
        ])
        # 使用 numpy.testing.assert_allclose 断言 fl_M_pas_inv_callable(fl_M_pas) 的结果与 expected 近似相等
        numpy.testing.assert_allclose(fl_M_pas_inv_callable(fl_M_pas), expected)

    # 标记为跳过，如果没有安装 JAX
    @pytest.mark.skipif(jax is None, reason='JAX not installed')
    def test_lambdify_jax(self):
        # 使用默认参数创建 FiberForceLengthPassiveInverseDeGroote2016 实例
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016.with_defaults(self.fl_M_pas)
        # 将 fl_M_pas_inv 转换为可调用的函数 fl_M_pas_inv_callable，使用 JAX
        fl_M_pas_inv_callable = jax.jit(lambdify(self.fl_M_pas, fl_M_pas_inv, 'jax'))
        # 创建 JAX 数组 fl_M_pas
        fl_M_pas = jax.numpy.array([-0.01, 0.0, 0.01, 0.02, 0.05, 0.1])
        # 预期的输出结果 JAX 数组
        expected = jax.numpy.array([
            0.8848253714,
            1.0,
            1.0643754386,
            1.1092744701,
            1.1954331425,
            1.2774998934,
        ])
        # 使用 numpy.testing.assert_allclose 断言 fl_M_pas_inv_callable(fl_M_pas) 的结果与 expected 近似相等
        numpy.testing.assert_allclose(fl_M_pas_inv_callable(fl_M_pas), expected)
class TestFiberForceLengthActiveDeGroote2016:
    # 定义测试类 TestFiberForceLengthActiveDeGroote2016，用于测试 FiberForceLengthActiveDeGroote2016 类

    @pytest.fixture(autouse=True)
    def _fiber_force_length_active_arguments_fixture(self):
        # 设置测试类的参数夹具，用于自动使用
        self.l_M_tilde = Symbol('l_M_tilde')
        self.c0 = Symbol('c_0')
        self.c1 = Symbol('c_1')
        self.c2 = Symbol('c_2')
        self.c3 = Symbol('c_3')
        self.c4 = Symbol('c_4')
        self.c5 = Symbol('c_5')
        self.c6 = Symbol('c_6')
        self.c7 = Symbol('c_7')
        self.c8 = Symbol('c_8')
        self.c9 = Symbol('c_9')
        self.c10 = Symbol('c_10')
        self.c11 = Symbol('c_11')
        self.constants = (
            self.c0, self.c1, self.c2, self.c3, self.c4, self.c5,
            self.c6, self.c7, self.c8, self.c9, self.c10, self.c11,
        )
        # 将参数符号化并存储为元组 self.constants

    @staticmethod
    def test_class():
        # 测试静态方法：验证 FiberForceLengthActiveDeGroote2016 是否为 Function 的子类
        assert issubclass(FiberForceLengthActiveDeGroote2016, Function)
        # 验证 FiberForceLengthActiveDeGroote2016 是否为 CharacteristicCurveFunction 的子类
        assert issubclass(FiberForceLengthActiveDeGroote2016, CharacteristicCurveFunction)
        # 验证 FiberForceLengthActiveDeGroote2016 类的名称是否为 'FiberForceLengthActiveDeGroote2016'
        assert FiberForceLengthActiveDeGroote2016.__name__ == 'FiberForceLengthActiveDeGroote2016'

    def test_instance(self):
        # 测试实例方法：创建 FiberForceLengthActiveDeGroote2016 实例，并验证其类型和字符串表示
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        assert isinstance(fl_M_act, FiberForceLengthActiveDeGroote2016)
        assert str(fl_M_act) == (
            'FiberForceLengthActiveDeGroote2016(l_M_tilde, c_0, c_1, c_2, c_3, '
            'c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11)'
        )
        # 验证创建的实例类型和字符串表示是否符合预期

    def test_doit(self):
        # 测试方法：调用 doit 方法计算 FiberForceLengthActiveDeGroote2016 实例的值，并验证结果
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants).doit()
        assert fl_M_act == (
            self.c0*exp(-(((self.l_M_tilde - self.c1)/(self.c2 + self.c3*self.l_M_tilde))**2)/2)
            + self.c4*exp(-(((self.l_M_tilde - self.c5)/(self.c6 + self.c7*self.l_M_tilde))**2)/2)
            + self.c8*exp(-(((self.l_M_tilde - self.c9)/(self.c10 + self.c11*self.l_M_tilde))**2)/2)
        )
        # 验证计算结果是否符合预期

    def test_doit_evaluate_false(self):
        # 测试方法：调用 doit 方法并禁用求值，验证计算 FiberForceLengthActiveDeGroote2016 实例的表达式，并验证结果
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants).doit(evaluate=False)
        assert fl_M_act == (
            self.c0*exp(-((UnevaluatedExpr(self.l_M_tilde - self.c1)/(self.c2 + self.c3*self.l_M_tilde))**2)/2)
            + self.c4*exp(-((UnevaluatedExpr(self.l_M_tilde - self.c5)/(self.c6 + self.c7*self.l_M_tilde))**2)/2)
            + self.c8*exp(-((UnevaluatedExpr(self.l_M_tilde - self.c9)/(self.c10 + self.c11*self.l_M_tilde))**2)/2)
        )
        # 验证计算结果是否符合预期，并确保求值被禁用
    # 测试默认参数情况下的功能
    def test_with_defaults(self):
        # 设置常量值
        constants = (
            Float('0.814'),
            Float('1.06'),
            Float('0.162'),
            Float('0.0633'),
            Float('0.433'),
            Float('0.717'),
            Float('-0.0299'),
            Float('0.2'),
            Float('0.1'),
            Float('1.0'),
            Float('0.354'),
            Float('0.0'),
        )
        # 手动计算 FiberForceLengthActiveDeGroote2016 类的实例
        fl_M_act_manual = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *constants)
        # 使用默认参数计算 FiberForceLengthActiveDeGroote2016 类的实例
        fl_M_act_constants = FiberForceLengthActiveDeGroote2016.with_defaults(self.l_M_tilde)
        # 断言手动计算的实例与默认参数计算的实例相等
        assert fl_M_act_manual == fl_M_act_constants

    # 测试关于 l_M_tilde 变量的偏导数计算是否正确
    def test_differentiate_wrt_l_M_tilde(self):
        # 创建 FiberForceLengthActiveDeGroote2016 类的实例
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        # 预期值，包含对 l_M_tilde 的偏导数表达式
        expected = (
            self.c0*(
                self.c3*(self.l_M_tilde - self.c1)**2/(self.c2 + self.c3*self.l_M_tilde)**3
                + (self.c1 - self.l_M_tilde)/((self.c2 + self.c3*self.l_M_tilde)**2)
            )*exp(-(self.l_M_tilde - self.c1)**2/(2*(self.c2 + self.c3*self.l_M_tilde)**2))
            + self.c4*(
                self.c7*(self.l_M_tilde - self.c5)**2/(self.c6 + self.c7*self.l_M_tilde)**3
                + (self.c5 - self.l_M_tilde)/((self.c6 + self.c7*self.l_M_tilde)**2)
            )*exp(-(self.l_M_tilde - self.c5)**2/(2*(self.c6 + self.c7*self.l_M_tilde)**2))
            + self.c8*(
                self.c11*(self.l_M_tilde - self.c9)**2/(self.c10 + self.c11*self.l_M_tilde)**3
                + (self.c9 - self.l_M_tilde)/((self.c10 + self.c11*self.l_M_tilde)**2)
            )*exp(-(self.l_M_tilde - self.c9)**2/(2*(self.c10 + self.c11*self.l_M_tilde)**2))
        )
        # 断言计算得到的偏导数与预期值相等
        assert fl_M_act.diff(self.l_M_tilde) == expected

    # 测试关于 c0 变量的偏导数计算是否正确
    def test_differentiate_wrt_c0(self):
        # 创建 FiberForceLengthActiveDeGroote2016 类的实例
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        # 预期值，包含对 c0 的偏导数表达式
        expected = exp(-(self.l_M_tilde - self.c1)**2/(2*(self.c2 + self.c3*self.l_M_tilde)**2))
        # 断言计算得到的偏导数与预期值相等
        assert fl_M_act.doit().diff(self.c0) == expected

    # 测试关于 c1 变量的偏导数计算是否正确
    def test_differentiate_wrt_c1(self):
        # 创建 FiberForceLengthActiveDeGroote2016 类的实例
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        # 预期值，包含对 c1 的偏导数表达式
        expected = (
            self.c0*(self.l_M_tilde - self.c1)/(self.c2 + self.c3*self.l_M_tilde)**2
            *exp(-(self.l_M_tilde - self.c1)**2/(2*(self.c2 + self.c3*self.l_M_tilde)**2))
        )
        # 断言计算得到的偏导数与预期值相等
        assert fl_M_act.diff(self.c1) == expected

    # 测试关于 c2 变量的偏导数计算是否正确
    def test_differentiate_wrt_c2(self):
        # 创建 FiberForceLengthActiveDeGroote2016 类的实例
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        # 预期值，包含对 c2 的偏导数表达式
        expected = (
            self.c0*(self.l_M_tilde - self.c1)**2/(self.c2 + self.c3*self.l_M_tilde)**3
            *exp(-(self.l_M_tilde - self.c1)**2/(2*(self.c2 + self.c3*self.l_M_tilde)**2))
        )
        # 断言计算得到的偏导数与预期值相等
        assert fl_M_act.diff(self.c2) == expected
    # 对于给定的 l_M_tilde 和 constants，创建 FiberForceLengthActiveDeGroote2016 对象 fl_M_act
    fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
    
    # 计算在对 c3 求偏导时的预期值
    expected = (
        self.c0*self.l_M_tilde*(self.l_M_tilde - self.c1)**2/(self.c2 + self.c3*self.l_M_tilde)**3
        *exp(-(self.l_M_tilde - self.c1)**2/(2*(self.c2 + self.c3*self.l_M_tilde)**2))
    )
    
    # 断言 fl_M_act 对象对 c3 的偏导是否等于预期值 expected
    assert fl_M_act.diff(self.c3) == expected


```  
    # 对于给定的 l_M_tilde 和 constants，创建 FiberForceLengthActiveDeGroote2016 对象 fl_M_act
    fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
    
    # 计算在对 c4 求偏导时的预期值
    expected = exp(-(self.l_M_tilde - self.c5)**2/(2*(self.c6 + self.c7*self.l_M_tilde)**2))
    
    # 断言 fl_M_act 对象对 c4 的偏导是否等于预期值 expected
    assert fl_M_act.diff(self.c4) == expected


```  
    # 对于给定的 l_M_tilde 和 constants，创建 FiberForceLengthActiveDeGroote2016 对象 fl_M_act
    fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
    
    # 计算在对 c5 求偏导时的预期值
    expected = (
        self.c4*(self.l_M_tilde - self.c5)/(self.c6 + self.c7*self.l_M_tilde)**2
        *exp(-(self.l_M_tilde - self.c5)**2/(2*(self.c6 + self.c7*self.l_M_tilde)**2))
    )
    
    # 断言 fl_M_act 对象对 c5 的偏导是否等于预期值 expected
    assert fl_M_act.diff(self.c5) == expected


```  
    # 对于给定的 l_M_tilde 和 constants，创建 FiberForceLengthActiveDeGroote2016 对象 fl_M_act
    fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
    
    # 计算在对 c6 求偏导时的预期值
    expected = (
        self.c4*(self.l_M_tilde - self.c5)**2/(self.c6 + self.c7*self.l_M_tilde)**3
        *exp(-(self.l_M_tilde - self.c5)**2/(2*(self.c6 + self.c7*self.l_M_tilde)**2))
    )
    
    # 断言 fl_M_act 对象对 c6 的偏导是否等于预期值 expected
    assert fl_M_act.diff(self.c6) == expected


```  
    # 对于给定的 l_M_tilde 和 constants，创建 FiberForceLengthActiveDeGroote2016 对象 fl_M_act
    fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
    
    # 计算在对 c7 求偏导时的预期值
    expected = (
        self.c4*self.l_M_tilde*(self.l_M_tilde - self.c5)**2/(self.c6 + self.c7*self.l_M_tilde)**3
        *exp(-(self.l_M_tilde - self.c5)**2/(2*(self.c6 + self.c7*self.l_M_tilde)**2))
    )
    
    # 断言 fl_M_act 对象对 c7 的偏导是否等于预期值 expected
    assert fl_M_act.diff(self.c7) == expected


```  
    # 对于给定的 l_M_tilde 和 constants，创建 FiberForceLengthActiveDeGroote2016 对象 fl_M_act
    fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
    
    # 计算在对 c8 求偏导时的预期值
    expected = exp(-(self.l_M_tilde - self.c9)**2/(2*(self.c10 + self.c11*self.l_M_tilde)**2))
    
    # 断言 fl_M_act 对象对 c8 的偏导是否等于预期值 expected
    assert fl_M_act.diff(self.c8) == expected


```  
    # 对于给定的 l_M_tilde 和 constants，创建 FiberForceLengthActiveDeGroote2016 对象 fl_M_act
    fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
    
    # 计算在对 c9 求偏导时的预期值
    expected = (
        self.c8*(self.l_M_tilde - self.c9)/(self.c10 + self.c11*self.l_M_tilde)**2
        *exp(-(self.l_M_tilde - self.c9)**2/(2*(self.c10 + self.c11*self.l_M_tilde)**2))
    )
    
    # 断言 fl_M_act 对象对 c9 的偏导是否等于预期值 expected
    assert fl_M_act.diff(self.c9) == expected


```  
    # 对于给定的 l_M_tilde 和 constants，创建 FiberForceLengthActiveDeGroote2016 对象 fl_M_act
    fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
    
    # 计算在对 c10 求偏导时的预期值
    expected = (
        self.c8*(self.l_M_tilde - self.c9)**2/(self.c10 + self.c11*self.l_M_tilde)**3
        *exp(-(self.l_M_tilde - self.c9)**2/(2*(self.c10 + self.c11*self.l_M_tilde)**2))
    )
    
    # 断言 fl_M_act 对象对 c10 的偏导是否等于预期值 expected
    assert fl_M_act.diff(self.c10) == expected
    # 定义测试方法，用于测试在给定参数 l_M_tilde 和 constants 的情况下，FiberForceLengthActiveDeGroote2016 类的行为
    def test_differentiate_wrt_c11(self):
        # 创建 FiberForceLengthActiveDeGroote2016 对象 fl_M_act，使用给定的 l_M_tilde 和 constants
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        # 计算期望值，基于给定公式计算 fl_M_act 对于 self.c11 的偏导数
        expected = (
            self.c8*self.l_M_tilde*(self.l_M_tilde - self.c9)**2/(self.c10 + self.c11*self.l_M_tilde)**3
            *exp(-(self.l_M_tilde - self.c9)**2/(2*(self.c10 + self.c11*self.l_M_tilde)**2))
        )
        # 断言 fl_M_act 对于 self.c11 的偏导数是否等于期望值
        assert fl_M_act.diff(self.c11) == expected

    # 定义测试方法，用于测试 LatexPrinter 的输出是否与预期的 LaTeX 表达式 expected 相等
    def test_function_print_latex(self):
        # 创建 FiberForceLengthActiveDeGroote2016 对象 fl_M_act，使用给定的 l_M_tilde 和 constants
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        # 期望的 LaTeX 表达式，表示 fl_M_act 在给定 l_M_tilde 下的函数形式
        expected = r'\operatorname{fl}^M_{act} \left( l_{M tilde} \right)'
        # 断言 fl_M_act 的 LaTeX 表示是否等于期望值 expected
        assert LatexPrinter().doprint(fl_M_act) == expected

    # 定义测试方法，用于测试 LatexPrinter 的输出是否与预期的 LaTeX 表达式 expected 相等
    def test_expression_print_latex(self):
        # 创建 FiberForceLengthActiveDeGroote2016 对象 fl_M_act，使用给定的 l_M_tilde 和 constants
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        # 期望的 LaTeX 表达式，表示 fl_M_act 在给定 l_M_tilde 下的表达式形式
        expected = (
            r'c_{0} e^{- \frac{\left(- c_{1} + l_{M tilde}\right)^{2}}{2 \left(c_{2} + c_{3} l_{M tilde}\right)^{2}}} '
            r'+ c_{4} e^{- \frac{\left(- c_{5} + l_{M tilde}\right)^{2}}{2 \left(c_{6} + c_{7} l_{M tilde}\right)^{2}}} '
            r'+ c_{8} e^{- \frac{\left(- c_{9} + l_{M tilde}\right)^{2}}{2 \left(c_{10} + c_{11} l_{M tilde}\right)^{2}}}'
        )
        # 断言 fl_M_act 的 LaTeX 表示是否等于期望值 expected
        assert LatexPrinter().doprint(fl_M_act.doit()) == expected

    # 定义测试方法，用于测试 PythonCodePrinter 的输出是否与预期的 Python 代码 expected 相等
    def test_print_code(self, code_printer, expected):
        # 创建 FiberForceLengthActiveDeGroote2016 对象 fl_M_act，使用默认参数 self.l_M_tilde
        fl_M_act = FiberForceLengthActiveDeGroote2016.with_defaults(self.l_M_tilde)
        # 断言使用指定的 code_printer 打印 fl_M_act 的代码表示是否等于期望值 expected
        assert code_printer().doprint(fl_M_act) == expected

    # 定义测试方法，用于测试 PythonCodePrinter 的输出是否与预期的 Python 代码 expected 相等
    def test_derivative_print_code(self):
        # 创建 FiberForceLengthActiveDeGroote2016 对象 fl_M_act，使用默认参数 self.l_M_tilde
        fl_M_act = FiberForceLengthActiveDeGroote2016.with_defaults(self.l_M_tilde)
        # 计算 fl_M_act 对于 self.l_M_tilde 的一阶导数
        fl_M_act_dl_M_tilde = fl_M_act.diff(self.l_M_tilde)
        # 期望的 Python 代码，表示 fl_M_act 对于 self.l_M_tilde 的一阶导数的表达式
        expected = (
            '(0.79798269973507 - 0.79798269973507*l_M_tilde)'
            '*math.exp(-3.98991349867535*(l_M_tilde - 1.0)**2) '
            '+ (10.825*(0.717 - l_M_tilde)/(l_M_tilde - 0.1495)**2 '
            '+ 10.825*(l_M_tilde - 0.717)**2/(l_M_tilde - 0.1495)**3)'
            '*math.exp(-12.5*(l_M_tilde - 0.717)**2/(l_M_tilde - 0.1495)**2) '
            '+ (31.0166133211401*(1.06 - l_M_tilde)/(0.390740740740741*l_M_tilde + 1)**2 '
            '+ 13.6174190361677*(0.943396226415094*l_M_tilde - 1)**2'
            '/(0.390740740740741*l_M_tilde + 1)**3)'
            '*math.exp(-21.4067977442463*(0.943396226415094*l_M_tilde - 1)**2'
            '/(0.390740740740741*l_M_tilde + 1)**2)'
        )
        # 断言使用 PythonCodePrinter 打印 fl_M_act_dl_M_tilde 的代码表示是否等于期望值 expected
        assert PythonCodePrinter().doprint(fl_M_act_dl_M_tilde) == expected

    # 定义测试方法，用于测试 lambdify 函数是否能正确生成 fl_M_act 的可调用函数，并计算其在给定点的值
    def test_lambdify(self):
        # 创建 FiberForceLengthActiveDeGroote2016 对象 fl_M_act，使用默认参数 self.l_M_tilde
        fl_M_act = FiberForceLengthActiveDeGroote2016.with_defaults(self.l_M_tilde)
        # 使用 lambdify 将 fl_M_act 转换为一个可调用函数 fl_M_act_callable
        fl_M_act_callable = lambdify(self.l_M_tilde, fl_M_act)
        # 断言 fl_M_act_callable 在输入为 1.0 时的计算结果是否接近于已知值 0.9941398866
        assert fl_M_act_callable(1.0) == pytest.approx(0.9941398866)

    @pytest.mark.skipif(numpy is None, reason='NumPy not installed')
    # 定义一个测试函数，用于测试 lambdify 函数在 numpy 环境下的功能
    def test_lambdify_numpy(self):
        # 使用指定参数创建 FiberForceLengthActiveDeGroote2016 对象实例
        fl_M_act = FiberForceLengthActiveDeGroote2016.with_defaults(self.l_M_tilde)
        # 将 fl_M_act 转换为一个可调用的函数，使用 numpy 作为后端
        fl_M_act_callable = lambdify(self.l_M_tilde, fl_M_act, 'numpy')
        # 创建一个包含多个浮点数的 numpy 数组
        l_M_tilde = numpy.array([0.0, 0.5, 1.0, 1.5, 2.0])
        # 创建一个预期的结果数组，包含与 l_M_tilde 相对应的数值
        expected = numpy.array([
            0.0018501319,
            0.0529122812,
            0.9941398866,
            0.2312431531,
            0.0069595432,
        ])
        # 断言 fl_M_act_callable(l_M_tilde) 的输出与预期结果 expected 在允许的误差范围内相等
        numpy.testing.assert_allclose(fl_M_act_callable(l_M_tilde), expected)

    # 使用 pytest 的标记来标识这个测试用例，在 JAX 没有安装的情况下跳过执行
    @pytest.mark.skipif(jax is None, reason='JAX not installed')
    # 定义一个测试函数，用于测试 lambdify 函数在 jax 环境下的功能
    def test_lambdify_jax(self):
        # 使用指定参数创建 FiberForceLengthActiveDeGroote2016 对象实例
        fl_M_act = FiberForceLengthActiveDeGroote2016.with_defaults(self.l_M_tilde)
        # 将 fl_M_act 转换为一个可调用的函数，并使用 jax 作为后端进行加速编译
        fl_M_act_callable = jax.jit(lambdify(self.l_M_tilde, fl_M_act, 'jax'))
        # 创建一个包含多个浮点数的 jax.numpy 数组
        l_M_tilde = jax.numpy.array([0.0, 0.5, 1.0, 1.5, 2.0])
        # 创建一个预期的结果数组，包含与 l_M_tilde 相对应的数值
        expected = jax.numpy.array([
            0.0018501319,
            0.0529122812,
            0.9941398866,
            0.2312431531,
            0.0069595432,
        ])
        # 断言 fl_M_act_callable(l_M_tilde) 的输出与预期结果 expected 在允许的误差范围内相等
        numpy.testing.assert_allclose(fl_M_act_callable(l_M_tilde), expected)
class TestFiberForceVelocityDeGroote2016:

    @pytest.fixture(autouse=True)
    def _muscle_fiber_force_velocity_arguments_fixture(self):
        # 定义符号变量
        self.v_M_tilde = Symbol('v_M_tilde')
        self.c0 = Symbol('c_0')
        self.c1 = Symbol('c_1')
        self.c2 = Symbol('c_2')
        self.c3 = Symbol('c_3')
        self.constants = (self.c0, self.c1, self.c2, self.c3)

    @staticmethod
    def test_class():
        # 断言类是 Function 的子类
        assert issubclass(FiberForceVelocityDeGroote2016, Function)
        # 断言类是 CharacteristicCurveFunction 的子类
        assert issubclass(FiberForceVelocityDeGroote2016, CharacteristicCurveFunction)
        # 断言类名是 'FiberForceVelocityDeGroote2016'
        assert FiberForceVelocityDeGroote2016.__name__ == 'FiberForceVelocityDeGroote2016'

    def test_instance(self):
        # 创建实例并断言是正确的类
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants)
        assert isinstance(fv_M, FiberForceVelocityDeGroote2016)
        # 断言实例的字符串表示符合预期
        assert str(fv_M) == 'FiberForceVelocityDeGroote2016(v_M_tilde, c_0, c_1, c_2, c_3)'

    def test_doit(self):
        # 调用 doit 方法并断言返回值符合预期
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants).doit()
        expected = (
            self.c0 * log((self.c1 * self.v_M_tilde + self.c2)
            + sqrt((self.c1 * self.v_M_tilde + self.c2)**2 + 1)) + self.c3
        )
        assert fv_M == expected

    def test_doit_evaluate_false(self):
        # 使用 evaluate=False 调用 doit 方法并断言返回值符合预期
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants).doit(evaluate=False)
        expected = (
            self.c0 * log((self.c1 * self.v_M_tilde + self.c2)
            + sqrt(UnevaluatedExpr(self.c1 * self.v_M_tilde + self.c2)**2 + 1)) + self.c3
        )
        assert fv_M == expected

    def test_with_defaults(self):
        # 使用默认常数创建实例并断言与手动设置常数的实例相等
        constants = (
            Float('-0.318'),
            Float('-8.149'),
            Float('-0.374'),
            Float('0.886'),
        )
        fv_M_manual = FiberForceVelocityDeGroote2016(self.v_M_tilde, *constants)
        fv_M_constants = FiberForceVelocityDeGroote2016.with_defaults(self.v_M_tilde)
        assert fv_M_manual == fv_M_constants

    def test_differentiate_wrt_v_M_tilde(self):
        # 对 v_M_tilde 进行微分并断言结果符合预期
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants)
        expected = (
            self.c0*self.c1
            /sqrt(UnevaluatedExpr(self.c1*self.v_M_tilde + self.c2)**2 + 1)
        )
        assert fv_M.diff(self.v_M_tilde) == expected

    def test_differentiate_wrt_c0(self):
        # 对 c0 进行微分并断言结果符合预期
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants)
        expected = log(
            self.c1*self.v_M_tilde + self.c2
            + sqrt(UnevaluatedExpr(self.c1*self.v_M_tilde + self.c2)**2 + 1)
        )
        assert fv_M.diff(self.c0) == expected

    def test_differentiate_wrt_c1(self):
        # 对 c1 进行微分并断言结果符合预期
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants)
        expected = (
            self.c0*self.v_M_tilde
            /sqrt(UnevaluatedExpr(self.c1*self.v_M_tilde + self.c2)**2 + 1)
        )
        assert fv_M.diff(self.c1) == expected
    # 定义测试方法：对 c2 进行求导测试
    def test_differentiate_wrt_c2(self):
        # 创建 FiberForceVelocityDeGroote2016 的实例对象 fv_M，使用给定的常量参数
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants)
        # 计算预期值，这里使用了数学表达式，self.c0 除以一个复杂表达式的平方根
        expected = (
            self.c0
            /sqrt(UnevaluatedExpr(self.c1*self.v_M_tilde + self.c2)**2 + 1)
        )
        # 断言求导结果是否与预期值相等
        assert fv_M.diff(self.c2) == expected

    # 定义测试方法：对 c3 进行求导测试
    def test_differentiate_wrt_c3(self):
        # 创建 FiberForceVelocityDeGroote2016 的实例对象 fv_M，使用给定的常量参数
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants)
        # 期望的导数值为整数 1
        expected = Integer(1)
        # 断言求导结果是否与预期值相等
        assert fv_M.diff(self.c3) == expected

    # 定义测试方法：测试反函数是否正确返回
    def test_inverse(self):
        # 创建 FiberForceVelocityDeGroote2016 的实例对象 fv_M，使用给定的常量参数
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants)
        # 断言调用反函数后返回的对象是否为 FiberForceVelocityInverseDeGroote2016
        assert fv_M.inverse() is FiberForceVelocityInverseDeGroote2016

    # 定义测试方法：测试打印 LaTeX 表达式是否正确
    def test_function_print_latex(self):
        # 创建 FiberForceVelocityDeGroote2016 的实例对象 fv_M，使用给定的常量参数
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants)
        # 期望的 LaTeX 表达式字符串
        expected = r'\operatorname{fv}^M \left( v_{M tilde} \right)'
        # 断言调用 LatexPrinter 打印 fv_M 后得到的字符串是否与期望值相等
        assert LatexPrinter().doprint(fv_M) == expected

    # 定义测试方法：测试表达式求值后打印 LaTeX 表达式是否正确
    def test_expression_print_latex(self):
        # 创建 FiberForceVelocityDeGroote2016 的实例对象 fv_M，使用给定的常量参数
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants)
        # 期望的 LaTeX 表达式字符串
        expected = (
            r'c_{0} \log{\left(c_{1} v_{M tilde} + c_{2} + \sqrt{\left(c_{1} '
            r'v_{M tilde} + c_{2}\right)^{2} + 1} \right)} + c_{3}'
        )
        # 断言调用 LatexPrinter 打印 fv_M.doit() 后得到的字符串是否与期望值相等
        assert LatexPrinter().doprint(fv_M.doit()) == expected

    # 定义测试方法：测试打印代码是否正确
    def test_print_code(self, code_printer, expected):
        # 使用默认参数创建 FiberForceVelocityDeGroote2016 的实例对象 fv_M
        fv_M = FiberForceVelocityDeGroote2016.with_defaults(self.v_M_tilde)
        # 断言调用 code_printer 打印 fv_M 后得到的字符串是否与期望值相等
        assert code_printer().doprint(fv_M) == expected

    # 定义测试方法：测试导数打印代码是否正确
    def test_derivative_print_code(self):
        # 使用默认参数创建 FiberForceVelocityDeGroote2016 的实例对象 fv_M
        fv_M = FiberForceVelocityDeGroote2016.with_defaults(self.v_M_tilde)
        # 计算对 v_M_tilde 的导数 dfv_M_dv_M_tilde
        dfv_M_dv_M_tilde = fv_M.diff(self.v_M_tilde)
        # 期望的 Python 代码字符串
        expected = '2.591382*(1 + (-8.149*v_M_tilde - 0.374)**2)**(-1/2)'
        # 断言调用 PythonCodePrinter 打印 dfv_M_dv_M_tilde 后得到的字符串是否与期望值相等
        assert PythonCodePrinter().doprint(dfv_M_dv_M_tilde) == expected

    # 定义测试方法：测试 lambdify 函数是否能正确生成可调用的函数
    def test_lambdify(self):
        # 使用默认参数创建 FiberForceVelocityDeGroote2016 的实例对象 fv_M
        fv_M = FiberForceVelocityDeGroote2016.with_defaults(self.v_M_tilde)
        # 使用 lambdify 函数将 fv_M 转换为可调用函数 fv_M_callable
        fv_M_callable = lambdify(self.v_M_tilde, fv_M)
        # 断言调用 fv_M_callable(0.0) 的结果是否与预期值非常接近
        assert fv_M_callable(0.0) == pytest.approx(1.002320622548512)

    # 定义测试方法：测试使用 NumPy 的 lambdify 函数是否正确生成可调用的函数
    @pytest.mark.skipif(numpy is None, reason='NumPy not installed')
    def test_lambdify_numpy(self):
        # 使用默认参数创建 FiberForceVelocityDeGroote2016 的实例对象 fv_M
        fv_M = FiberForceVelocityDeGroote2016.with_defaults(self.v_M_tilde)
        # 使用 NumPy 的 lambdify 函数将 fv_M 转换为可调用函数 fv_M_callable
        fv_M_callable = lambdify(self.v_M_tilde, fv_M, 'numpy')
        # 定义输入数组 v_M_tilde
        v_M_tilde = numpy.array([-1.0, -0.5, 0.0, 0.5])
        # 期望的输出数组
        expected = numpy.array([
            0.0120816781,
            0.2438336294,
            1.0023206225,
            1.5850003903,
        ])
        # 使用 numpy.testing.assert_allclose 断言 fv_M_callable(v_M_tilde) 的结果与 expected 数组非常接近
        numpy.testing.assert_allclose(fv_M_callable(v_M_tilde), expected)

    # 定义测试方法：测试使用 JAX 的 lambdify 函数是否能正确生成可调用的函数
    @pytest.mark.skipif(jax is None, reason='JAX not installed')
    def test_lambdify_jax(self):
        # 实现跳过 JAX 的情况下的测试
        pass
    # 定义一个测试函数，用于测试 lambdify_jax 方法
    def test_lambdify_jax(self):
        # 使用默认参数创建 FiberForceVelocityDeGroote2016 类的实例 fv_M
        fv_M = FiberForceVelocityDeGroote2016.with_defaults(self.v_M_tilde)
        # 使用 lambdify 函数创建一个能够使用 JAX 加速的可调用函数 fv_M_callable
        fv_M_callable = jax.jit(lambdify(self.v_M_tilde, fv_M, 'jax'))
        # 创建一个 JAX 的数组 v_M_tilde，包含特定的数值
        v_M_tilde = jax.numpy.array([-1.0, -0.5, 0.0, 0.5])
        # 创建一个期望的结果数组 expected，包含特定的数值
        expected = jax.numpy.array([
            0.0120816781,
            0.2438336294,
            1.0023206225,
            1.5850003903,
        ])
        # 使用 numpy.testing.assert_allclose 函数断言 fv_M_callable(v_M_tilde) 的结果与 expected 数组在允许误差范围内相等
        numpy.testing.assert_allclose(fv_M_callable(v_M_tilde), expected)
# 定义测试类 TestFiberForceVelocityInverseDeGroote2016，用于测试 FiberForceVelocityInverseDeGroote2016 类
class TestFiberForceVelocityInverseDeGroote2016:

    # pytest fixture，用于设置测试环境，自动执行
    @pytest.fixture(autouse=True)
    def _tendon_force_length_inverse_arguments_fixture(self):
        # 创建符号变量 fv_M
        self.fv_M = Symbol('fv_M')
        # 创建符号变量 c_0, c_1, c_2, c_3
        self.c0 = Symbol('c_0')
        self.c1 = Symbol('c_1')
        self.c2 = Symbol('c_2')
        self.c3 = Symbol('c_3')
        # 将符号变量组成常量元组
        self.constants = (self.c0, self.c1, self.c2, self.c3)

    # 静态方法测试类的基本属性
    @staticmethod
    def test_class():
        # 断言 FiberForceVelocityInverseDeGroote2016 是 Function 的子类
        assert issubclass(FiberForceVelocityInverseDeGroote2016, Function)
        # 断言 FiberForceVelocityInverseDeGroote2016 是 CharacteristicCurveFunction 的子类
        assert issubclass(FiberForceVelocityInverseDeGroote2016, CharacteristicCurveFunction)
        # 断言 FiberForceVelocityInverseDeGroote2016 类的名称为 'FiberForceVelocityInverseDeGroote2016'
        assert FiberForceVelocityInverseDeGroote2016.__name__ == 'FiberForceVelocityInverseDeGroote2016'

    # 测试创建实例
    def test_instance(self):
        # 创建 FiberForceVelocityInverseDeGroote2016 实例 fv_M_inv
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants)
        # 断言 fv_M_inv 是 FiberForceVelocityInverseDeGroote2016 类的实例
        assert isinstance(fv_M_inv, FiberForceVelocityInverseDeGroote2016)
        # 断言 str(fv_M_inv) 的输出为 'FiberForceVelocityInverseDeGroote2016(fv_M, c_0, c_1, c_2, c_3)'
        assert str(fv_M_inv) == 'FiberForceVelocityInverseDeGroote2016(fv_M, c_0, c_1, c_2, c_3)'

    # 测试 doit 方法
    def test_doit(self):
        # 创建 FiberForceVelocityInverseDeGroote2016 实例 fv_M_inv，并调用 doit 方法
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants).doit()
        # 计算预期值
        expected = (sinh((self.fv_M - self.c3)/self.c0) - self.c2)/self.c1
        # 断言计算结果与预期值相等
        assert fv_M_inv == expected

    # 测试 doit 方法中 evaluate=False 的情况
    def test_doit_evaluate_false(self):
        # 创建 FiberForceVelocityInverseDeGroote2016 实例 fv_M_inv，并调用 doit 方法
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants).doit(evaluate=False)
        # 计算预期值，使用 UnevaluatedExpr 包装表达式
        expected = (sinh(UnevaluatedExpr(self.fv_M - self.c3)/self.c0) - self.c2)/self.c1
        # 断言计算结果与预期值相等
        assert fv_M_inv == expected

    # 测试使用默认常量值的情况
    def test_with_defaults(self):
        # 定义常量元组 constants
        constants = (
            Float('-0.318'),
            Float('-8.149'),
            Float('-0.374'),
            Float('0.886'),
        )
        # 创建 FiberForceVelocityInverseDeGroote2016 实例 fv_M_inv_manual
        fv_M_inv_manual = FiberForceVelocityInverseDeGroote2016(self.fv_M, *constants)
        # 调用 with_defaults 方法创建 FiberForceVelocityInverseDeGroote2016 实例 fv_M_inv_constants
        fv_M_inv_constants = FiberForceVelocityInverseDeGroote2016.with_defaults(self.fv_M)
        # 断言 fv_M_inv_manual 和 fv_M_inv_constants 相等
        assert fv_M_inv_manual == fv_M_inv_constants

    # 测试对 fv_M 求导数
    def test_differentiate_wrt_fv_M(self):
        # 创建 FiberForceVelocityInverseDeGroote2016 实例 fv_M_inv
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants)
        # 计算对 fv_M 求导数的预期值
        expected = cosh((self.fv_M - self.c3)/self.c0)/(self.c0*self.c1)
        # 断言 fv_M_inv 对 fv_M 求导数的计算结果与预期值相等
        assert fv_M_inv.diff(self.fv_M) == expected

    # 测试对 c0 求导数
    def test_differentiate_wrt_c0(self):
        # 创建 FiberForceVelocityInverseDeGroote2016 实例 fv_M_inv
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants)
        # 计算对 c0 求导数的预期值
        expected = (self.c3 - self.fv_M)*cosh((self.fv_M - self.c3)/self.c0)/(self.c0**2*self.c1)
        # 断言 fv_M_inv 对 c0 求导数的计算结果与预期值相等
        assert fv_M_inv.diff(self.c0) == expected

    # 测试对 c1 求导数
    def test_differentiate_wrt_c1(self):
        # 创建 FiberForceVelocityInverseDeGroote2016 实例 fv_M_inv
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants)
        # 计算对 c1 求导数的预期值
        expected = (self.c2 - sinh((self.fv_M - self.c3)/self.c0))/self.c1**2
        # 断言 fv_M_inv 对 c1 求导数的计算结果与预期值相等
        assert fv_M_inv.diff(self.c1) == expected

    # 测试对 c2 求导数
    def test_differentiate_wrt_c2(self):
        # 创建 FiberForceVelocityInverseDeGroote2016 实例 fv_M_inv
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants)
        # 计算对 c2 求导数的预期值
        expected = -1/self.c1
        # 断言 fv_M_inv 对 c2 求导数的计算结果与预期值相等
        assert fv_M_inv.diff(self.c2) == expected
    # 定义测试函数，用于测试 FiberForceVelocityInverseDeGroote2016 类的 differentiate_wrt_c3 方法
    def test_differentiate_wrt_c3(self):
        # 创建 FiberForceVelocityInverseDeGroote2016 类的实例 fv_M_inv，并传入参数 self.fv_M 和 self.constants
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants)
        # 计算预期值 expected，这里计算的是一个表达式
        expected = -cosh((self.fv_M - self.c3)/self.c0)/(self.c0*self.c1)
        # 断言 differentiate_wrt_c3 方法的返回值与预期值相等
        assert fv_M_inv.diff(self.c3) == expected

    # 定义测试函数，用于测试 FiberForceVelocityInverseDeGroote2016 类的 inverse 方法
    def test_inverse(self):
        # 创建 FiberForceVelocityInverseDeGroote2016 类的实例 fv_M_inv，并传入参数 self.fv_M 和 self.constants
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants)
        # 断言 inverse 方法的返回值是 FiberForceVelocityDeGroote2016 类的实例
        assert fv_M_inv.inverse() is FiberForceVelocityDeGroote2016

    # 定义测试函数，用于测试 FiberForceVelocityInverseDeGroote2016 类的打印 LaTeX 表达式方法
    def test_function_print_latex(self):
        # 创建 FiberForceVelocityInverseDeGroote2016 类的实例 fv_M_inv，并传入参数 self.fv_M 和 self.constants
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants)
        # 计算预期的 LaTeX 表达式 expected
        expected = r'\left( \operatorname{fv}^M \right)^{-1} \left( fv_{M} \right)'
        # 断言使用 LatexPrinter 打印 fv_M_inv 的 LaTeX 表达式等于预期的 LaTeX 表达式
        assert LatexPrinter().doprint(fv_M_inv) == expected

    # 定义测试函数，用于测试 FiberForceVelocityInverseDeGroote2016 类的表达式计算并打印 LaTeX 表达式方法
    def test_expression_print_latex(self):
        # 创建 FiberForceVelocityInverseDeGroote2016 类的实例 fv_M_inv，并传入参数 self.fv_M 和 self.constants
        fv_M = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants)
        # 计算预期的 LaTeX 表达式 expected
        expected = r'\frac{- c_{2} + \sinh{\left(\frac{- c_{3} + fv_{M}}{c_{0}} \right)}}{c_{1}}'
        # 断言使用 LatexPrinter 打印 fv_M.doit() 的 LaTeX 表达式等于预期的 LaTeX 表达式
        assert LatexPrinter().doprint(fv_M.doit()) == expected
    # 使用 pytest 的 parametrize 装饰器为代码生成多组参数化的测试数据
    @pytest.mark.parametrize(
        'code_printer, expected',  # 参数化的两个参数：code_printer 和 expected
        [  # 参数化的测试数据列表开始
            (
                C89CodePrinter,  # 使用 C89CodePrinter 生成的代码输出
                '(-0.12271444348999878*(0.374 - sinh(3.1446540880503142*(fv_M - 0.88600000000000001))))',
            ),
            (
                C99CodePrinter,  # 使用 C99CodePrinter 生成的代码输出
                '(-0.12271444348999878*(0.374 - sinh(3.1446540880503142*(fv_M - 0.88600000000000001))))',
            ),
            (
                C11CodePrinter,  # 使用 C11CodePrinter 生成的代码输出
                '(-0.12271444348999878*(0.374 - sinh(3.1446540880503142*(fv_M - 0.88600000000000001))))',
            ),
            (
                CXX98CodePrinter,  # 使用 CXX98CodePrinter 生成的代码输出
                '(-0.12271444348999878*(0.374 - sinh(3.1446540880503142*(fv_M - 0.88600000000000001))))',
            ),
            (
                CXX11CodePrinter,  # 使用 CXX11CodePrinter 生成的代码输出
                '(-0.12271444348999878*(0.374 - std::sinh(3.1446540880503142*(fv_M - 0.88600000000000001))))',
            ),
            (
                CXX17CodePrinter,  # 使用 CXX17CodePrinter 生成的代码输出
                '(-0.12271444348999878*(0.374 - std::sinh(3.1446540880503142*(fv_M - 0.88600000000000001))))',
            ),
            (
                FCodePrinter,  # 使用 FCodePrinter 生成的代码输出
                '      (-0.122714443489999d0*(0.374d0 - sinh(3.1446540880503142d0*(fv_M -\n'
                '     @ 0.886d0))))',
            ),
            (
                OctaveCodePrinter,  # 使用 OctaveCodePrinter 生成的代码输出
                '(-0.122714443489999*(0.374 - sinh(3.14465408805031*(fv_M - 0.886))))',
            ),
            (
                PythonCodePrinter,  # 使用 PythonCodePrinter 生成的代码输出
                '(-0.122714443489999*(0.374 - math.sinh(3.14465408805031*(fv_M - 0.886))))',
            ),
            (
                NumPyPrinter,  # 使用 NumPyPrinter 生成的代码输出
                '(-0.122714443489999*(0.374 - numpy.sinh(3.14465408805031*(fv_M - 0.886))))',
            ),
            (
                SciPyPrinter,  # 使用 SciPyPrinter 生成的代码输出
                '(-0.122714443489999*(0.374 - numpy.sinh(3.14465408805031*(fv_M - 0.886))))',
            ),
            (
                CuPyPrinter,  # 使用 CuPyPrinter 生成的代码输出
                '(-0.122714443489999*(0.374 - cupy.sinh(3.14465408805031*(fv_M - 0.886))))',
            ),
            (
                JaxPrinter,  # 使用 JaxPrinter 生成的代码输出
                '(-0.122714443489999*(0.374 - jax.numpy.sinh(3.14465408805031*(fv_M - 0.886))))',
            ),
            (
                MpmathPrinter,  # 使用 MpmathPrinter 生成的代码输出
                '(-mpmath.mpf((0, 8842507551592581, -56, 53))*(mpmath.mpf((0, '
                '3368692521273131, -53, 52)) - mpmath.sinh(mpmath.mpf((0, '
                '7081131489576251, -51, 53))*(fv_M + mpmath.mpf((1, '
                '7980378539700519, -53, 53))))))',
            ),
            (
                LambdaPrinter,  # 使用 LambdaPrinter 生成的代码输出
                '(-0.122714443489999*(0.374 - math.sinh(3.14465408805031*(fv_M - 0.886))))',
            ),
        ]  # 参数化的测试数据列表结束
    )
    # 定义一个测试方法，用于验证打印代码的正确性
    def test_print_code(self, code_printer, expected):
        # 使用默认参数创建 FiberForceVelocityInverseDeGroote2016 类的实例
        fv_M_inv = FiberForceVelocityInverseDeGroote2016.with_defaults(self.fv_M)
        # 断言打印出的代码与预期的字符串相等
        assert code_printer().doprint(fv_M_inv) == expected

    # 定义一个测试方法，用于验证导数打印代码的正确性
    def test_derivative_print_code(self):
        # 使用默认参数创建 FiberForceVelocityInverseDeGroote2016 类的实例
        fv_M_inv = FiberForceVelocityInverseDeGroote2016.with_defaults(self.fv_M)
        # 计算 fv_M_inv 对 self.fv_M 的导数
        dfv_M_inv_dfv_M = fv_M_inv.diff(self.fv_M)
        # 预期的导数打印代码字符串
        expected = (
            '0.385894476383644*math.cosh(3.14465408805031*fv_M '
            '- 2.78616352201258)'
        )
        # 断言导数打印出的代码与预期的字符串相等
        assert PythonCodePrinter().doprint(dfv_M_inv_dfv_M) == expected

    # 定义一个测试方法，用于验证 lambdify 函数的正确性
    def test_lambdify(self):
        # 使用默认参数创建 FiberForceVelocityInverseDeGroote2016 类的实例
        fv_M_inv = FiberForceVelocityInverseDeGroote2016.with_defaults(self.fv_M)
        # 将 fv_M_inv 转换为可调用的函数对象
        fv_M_inv_callable = lambdify(self.fv_M, fv_M_inv)
        # 断言调用 fv_M_inv_callable(1.0) 的结果与预期值非常接近
        assert fv_M_inv_callable(1.0) == pytest.approx(-0.0009548832444487479)

    # 带有 NumPy 的 lambdify 测试方法，使用装饰器标记跳过条件
    @pytest.mark.skipif(numpy is None, reason='NumPy not installed')
    def test_lambdify_numpy(self):
        # 使用默认参数创建 FiberForceVelocityInverseDeGroote2016 类的实例
        fv_M_inv = FiberForceVelocityInverseDeGroote2016.with_defaults(self.fv_M)
        # 将 fv_M_inv 转换为可调用的 NumPy 函数对象
        fv_M_inv_callable = lambdify(self.fv_M, fv_M_inv, 'numpy')
        # 创建 NumPy 数组 fv_M
        fv_M = numpy.array([0.8, 0.9, 1.0, 1.1, 1.2])
        # 预期的 NumPy 数组结果
        expected = numpy.array([
            -0.0794881459,
            -0.0404909338,
            -0.0009548832,
            0.043061991,
            0.0959484397,
        ])
        # 断言 fv_M_inv_callable(fv_M) 的结果与预期的数组非常接近
        numpy.testing.assert_allclose(fv_M_inv_callable(fv_M), expected)

    # 带有 JAX 的 lambdify 测试方法，使用装饰器标记跳过条件
    @pytest.mark.skipif(jax is None, reason='JAX not installed')
    def test_lambdify_jax(self):
        # 使用默认参数创建 FiberForceVelocityInverseDeGroote2016 类的实例
        fv_M_inv = FiberForceVelocityInverseDeGroote2016.with_defaults(self.fv_M)
        # 使用 JAX 的 jit 函数优化 lambdify 转换后的函数
        fv_M_inv_callable = jax.jit(lambdify(self.fv_M, fv_M_inv, 'jax'))
        # 创建 JAX 数组 fv_M
        fv_M = jax.numpy.array([0.8, 0.9, 1.0, 1.1, 1.2])
        # 预期的 JAX 数组结果
        expected = jax.numpy.array([
            -0.0794881459,
            -0.0404909338,
            -0.0009548832,
            0.043061991,
            0.0959484397,
        ])
        # 断言 fv_M_inv_callable(fv_M) 的结果与预期的数组非常接近
        numpy.testing.assert_allclose(fv_M_inv_callable(fv_M), expected)
# 定义一个测试类 TestCharacteristicCurveCollection，用于测试 CharacteristicCurveCollection 类的各种行为
class TestCharacteristicCurveCollection:

    # 测试有效的构造函数
    @staticmethod
    def test_valid_constructor():
        # 创建 CharacteristicCurveCollection 对象，传入各种曲线类作为参数
        curves = CharacteristicCurveCollection(
            tendon_force_length=TendonForceLengthDeGroote2016,
            tendon_force_length_inverse=TendonForceLengthInverseDeGroote2016,
            fiber_force_length_passive=FiberForceLengthPassiveDeGroote2016,
            fiber_force_length_passive_inverse=FiberForceLengthPassiveInverseDeGroote2016,
            fiber_force_length_active=FiberForceLengthActiveDeGroote2016,
            fiber_force_velocity=FiberForceVelocityDeGroote2016,
            fiber_force_velocity_inverse=FiberForceVelocityInverseDeGroote2016,
        )
        # 断言各个属性与对应的曲线类相等
        assert curves.tendon_force_length is TendonForceLengthDeGroote2016
        assert curves.tendon_force_length_inverse is TendonForceLengthInverseDeGroote2016
        assert curves.fiber_force_length_passive is FiberForceLengthPassiveDeGroote2016
        assert curves.fiber_force_length_passive_inverse is FiberForceLengthPassiveInverseDeGroote2016
        assert curves.fiber_force_length_active is FiberForceLengthActiveDeGroote2016
        assert curves.fiber_force_velocity is FiberForceVelocityDeGroote2016
        assert curves.fiber_force_velocity_inverse is FiberForceVelocityInverseDeGroote2016

    # 标记为跳过的测试函数，因为 kw_only 数据类只在 Python >3.10 中有效
    @staticmethod
    @pytest.mark.skip(reason='kw_only dataclasses only valid in Python >3.10')
    def test_invalid_constructor_keyword_only():
        # 使用 pytest.raises 检测是否抛出 TypeError 异常
        with pytest.raises(TypeError):
            _ = CharacteristicCurveCollection(
                # 以位置参数的方式传入各个曲线类，预期会抛出 TypeError
                TendonForceLengthDeGroote2016,
                TendonForceLengthInverseDeGroote2016,
                FiberForceLengthPassiveDeGroote2016,
                FiberForceLengthPassiveInverseDeGroote2016,
                FiberForceLengthActiveDeGroote2016,
                FiberForceVelocityDeGroote2016,
                FiberForceVelocityInverseDeGroote2016,
            )

    # 参数化测试函数，用不同的 kwargs 参数进行多次测试
    @staticmethod
    @pytest.mark.parametrize(
        'kwargs',
        [
            {'tendon_force_length': TendonForceLengthDeGroote2016},  # 只传入一个必要参数
            {
                'tendon_force_length': TendonForceLengthDeGroote2016,
                'tendon_force_length_inverse': TendonForceLengthInverseDeGroote2016,
                'fiber_force_length_passive': FiberForceLengthPassiveDeGroote2016,
                'fiber_force_length_passive_inverse': FiberForceLengthPassiveInverseDeGroote2016,
                'fiber_force_length_active': FiberForceLengthActiveDeGroote2016,
                'fiber_force_velocity': FiberForceVelocityDeGroote2016,
                'fiber_force_velocity_inverse': FiberForceVelocityInverseDeGroote2016,
                'extra_kwarg': None,  # 额外的关键字参数，不应该被接受
            },
        ]
    )
    def test_invalid_constructor_wrong_number_args(kwargs):
        # 使用 pytest.raises 检测是否抛出 TypeError 异常
        with pytest.raises(TypeError):
            _ = CharacteristicCurveCollection(**kwargs)

    @staticmethod
    # 定义一个测试函数，用于验证CharacteristicCurveCollection的实例是不可变的
    
    def test_instance_is_immutable():
        # 创建一个CharacteristicCurveCollection实例，包含各种曲线模型
        curves = CharacteristicCurveCollection(
            tendon_force_length=TendonForceLengthDeGroote2016,
            tendon_force_length_inverse=TendonForceLengthInverseDeGroote2016,
            fiber_force_length_passive=FiberForceLengthPassiveDeGroote2016,
            fiber_force_length_passive_inverse=FiberForceLengthPassiveInverseDeGroote2016,
            fiber_force_length_active=FiberForceLengthActiveDeGroote2016,
            fiber_force_velocity=FiberForceVelocityDeGroote2016,
            fiber_force_velocity_inverse=FiberForceVelocityInverseDeGroote2016,
        )
    
        # 使用pytest模块验证各属性不能被重新赋值
        with pytest.raises(AttributeError):
            curves.tendon_force_length = None  # 验证属性 tendon_force_length 不能被重新赋值
        with pytest.raises(AttributeError):
            curves.tendon_force_length_inverse = None  # 验证属性 tendon_force_length_inverse 不能被重新赋值
        with pytest.raises(AttributeError):
            curves.fiber_force_length_passive = None  # 验证属性 fiber_force_length_passive 不能被重新赋值
        with pytest.raises(AttributeError):
            curves.fiber_force_length_passive_inverse = None  # 验证属性 fiber_force_length_passive_inverse 不能被重新赋值
        with pytest.raises(AttributeError):
            curves.fiber_force_length_active = None  # 验证属性 fiber_force_length_active 不能被重新赋值
        with pytest.raises(AttributeError):
            curves.fiber_force_velocity = None  # 验证属性 fiber_force_velocity 不能被重新赋值
        with pytest.raises(AttributeError):
            curves.fiber_force_velocity_inverse = None  # 验证属性 fiber_force_velocity_inverse 不能被重新赋值
```
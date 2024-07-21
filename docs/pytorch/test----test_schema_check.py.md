# `.\pytorch\test\test_schema_check.py`

```py
# Owner(s): ["oncall: jit"]

# 导入必要的库和模块
import os
import sys
import torch
from torch.utils._pytree import tree_map  # 导入特定的模块成员
import unittest

# 导入测试相关的模块和函数
from torch.testing._internal.common_utils import run_tests, TEST_WITH_TORCHDYNAMO
from torch.fx.operator_schemas import normalize_function
from torch._subclasses.schema_check_mode import SchemaCheckMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_device_type import ops, OpDTypes, instantiate_device_type_tests

# 获取当前文件所在目录的父目录并添加到系统路径中
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 定义一个函数，将输入张量展平
def secretly_aliasing(x):
    return x.view(-1)

# 定义一个函数，对输入张量进行秘密修改
def secretly_mutating(x):
    x.mul_(2)  # 原地乘以2
    return x * 3  # 返回乘以3后的结果

# 定义一个函数，直接返回输入张量本身
def output_is_input(x):
    return x

# 创建自定义的 Torch 库对象，用于定义函数签名
custom_lib = torch.library.Library("bad_schemas", "DEF")  # noqa: TOR901
custom_lib.define("secretly_aliasing(Tensor x) -> Tensor")
custom_lib.define("secretly_mutating(Tensor x) -> Tensor")
custom_lib.define("output_is_input(Tensor(a) x) -> Tensor(a)")

# 创建在 CPU 上实现函数的自定义 Torch 库对象
custom_lib_cpu = torch.library.Library("bad_schemas", "IMPL", "CPU")  # noqa: TOR901
custom_lib_cpu.impl("secretly_aliasing", secretly_aliasing)
custom_lib_cpu.impl("secretly_mutating", secretly_mutating)
custom_lib_cpu.impl("output_is_input", output_is_input)

# 创建带有元数据的自定义 Torch 库对象
custom_lib_meta = torch.library.Library("bad_schemas", "IMPL", "Meta")  # noqa: TOR901
custom_lib_meta.impl("secretly_aliasing", secretly_aliasing)
custom_lib_meta.impl("secretly_mutating", secretly_mutating)
custom_lib_meta.impl("output_is_input", output_is_input)

# 以下是一个用于模拟不正确模式的 TorchDispatchTensor 子类
# 用于测试 SchemaCheckMode 的行为是否符合预期
class IncorrectAliasTensor(torch.Tensor):
    ALIAS_ARG_OUT = {"aten::add"}
    ALIAS_OUT_OUT = {"aten::aminmax"}
    MUTATE_ARGS_OUT = {"aten::sub"}

    elem: torch.Tensor

    __slots__ = ['elem']

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # 封装的张量 (IncorrectAliasTensor) 不应持有该类的任何内存
        # 但是应该仍然展示与之前相同的设备
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, elem.size(),
            strides=elem.stride(), storage_offset=elem.storage_offset(),
            # TODO: clone storage aliasing
            dtype=elem.dtype, layout=elem.layout,
            device=elem.device, requires_grad=kwargs.get("requires_grad", False)
        )
        # ...真正的张量作为张量的一个元素被持有
        r.elem = elem.detach() if r.requires_grad else elem
        return r

    def __repr__(self):
        return super().__repr__(tensor_contents=f"{self.elem}")

    @classmethod
    # 定义一个特殊的方法，用于 Torch 模块中的类型分发
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # 定义一个内部函数，用于将参数中的元素展开，如果是指定类型的对象，则返回其内部元素
        def unwrap(e):
            return e.elem if isinstance(e, cls) else e

        # 定义一个内部函数，用于将参数中的元素包装，如果是 Torch 张量，则用特定类型的对象进行包装
        def wrap(e):
            return cls(e) if isinstance(e, torch.Tensor) else e
        
        # 对参数列表中的所有元素应用 unwrap 函数
        unwrapped_args = tree_map(unwrap, args)
        # 调用 func 函数，传入展开后的参数和展开后的关键字参数，得到结果
        out = func(*unwrapped_args, **tree_map(unwrap, kwargs))
        
        # 如果 func 的名称在错误别名张量的 ALIAS_ARG_OUT 中
        if func._schema.name in IncorrectAliasTensor.ALIAS_ARG_OUT:
            # 将 args[0] 的元素设置为 out
            args[0].elem = out
        
        # 如果 func 的名称在错误别名张量的 MUTATE_ARGS_OUT 中
        if func._schema.name in IncorrectAliasTensor.MUTATE_ARGS_OUT:
            # 将 args[0] 的元素设置为一个随机张量，形状与原来相同
            args[0].elem = torch.rand(args[0].elem.shape)
        
        # 如果 func 的名称在错误别名张量的 ALIAS_OUT_OUT 中
        if func._schema.name in IncorrectAliasTensor.ALIAS_OUT_OUT:
            # 对 out 进行操作，将第一个和第二个元素交换后，再进行包装，并返回结果
            incorrect_out = list(out)
            incorrect_out[0] = incorrect_out[1]
            return tree_map(wrap, tuple(incorrect_out))
        
        # 对 out 应用 wrap 函数，并返回结果
        return tree_map(wrap, out)
# 测试各种模式检查功能的类。
class TestSchemaCheck(JitTestCase):
    
    # 设置测试环境，如果使用 TORCHDYNAMO 进行测试则跳过
    def setUp(self):
        if TEST_WITH_TORCHDYNAMO:
            self.skipTest("SchemaCheckMode is ignored by dynamo")
        super().setUp()

    # 测试 SchemaCheckMode 是否记录操作顺序及其梯度
    def test_schema_check_mode_operator_order(self):
        # 进入 SchemaCheckMode 上下文，并记录操作顺序到 schema_check.ops
        with SchemaCheckMode() as schema_check:
            x = torch.rand((3, 3), requires_grad=True)
            x.relu().sin()
        # 断言记录的操作顺序是否正确
        self.assertEqual(["aten::rand", "aten::relu", "aten::detach", "aten::sin"], schema_check.ops)

    # 测试 SchemaCheckMode 是否记录操作顺序但不记录梯度
    def test_schema_check_mode_operator_order_without_grad(self):
        # 进入 SchemaCheckMode 上下文，并记录操作顺序到 schema_check.ops
        with SchemaCheckMode() as schema_check:
            x = torch.rand((3, 3), requires_grad=False)
            x.relu().sin()
        # 断言记录的操作顺序是否正确
        self.assertEqual(["aten::rand", "aten::relu", "aten::sin"], schema_check.ops)

    # 测试 SchemaCheckMode 是否记录变异和别名但不期望任何变异
    def test_schema_check_mode_mutated_aliasing_none(self):
        # 创建随机张量 x，并在 SchemaCheckMode 下记录变异和别名
        # 注意：先前 requires_grad=True，但这会导致保存变量的 detach
        x = torch.rand((3, 3))
        with SchemaCheckMode() as schema_check:
            actual = x.relu().sin()
        # 断言记录的变异和别名是否为空列表
        self.assertEqual([], schema_check.mutated)
        self.assertEqual([], schema_check.aliasing)

    # 测试 SchemaCheckMode 是否记录变异和别名，并期望有变异
    def test_schema_check_mode_mutated_aliasing_mutation(self):
        # 创建具有 requires_grad=False 的随机张量 actual，并在 SchemaCheckMode 下记录变异和别名
        actual = torch.rand((3, 3), requires_grad=False)
        with SchemaCheckMode() as schema_check:
            actual.sinh_()
        # 断言记录的变异和别名是否符合预期
        self.assertEqual([('aten::sinh_', 'input')], schema_check.mutated)
        self.assertEqual([('aten::sinh_', 'input', 'output_0')], schema_check.aliasing)

    # 测试 SchemaCheckMode 是否记录变异和别名，并使用 resize_ 方法
    def test_schema_check_mode_mutated_aliasing_resize_(self):
        # 创建具有 requires_grad=False 的随机张量 actual，并在 SchemaCheckMode 下记录变异和别名
        actual = torch.rand((3, 3), requires_grad=False)
        with SchemaCheckMode() as schema_check:
            actual.resize_(9)
        # 断言记录的变异和别名是否符合预期
        self.assertEqual([('aten::resize_', 'input')], schema_check.mutated)
        self.assertEqual([('aten::resize_', 'input', 'output_0')], schema_check.aliasing)

    # 测试 SchemaCheckMode 是否记录变异和别名，并使用别名输入
    def test_schema_check_mode_mutated_aliasing_aliasing_inputs(self):
        # 创建随机张量 actual，并设置别名 y 为 actual，然后在 SchemaCheckMode 下记录变异和别名
        actual = torch.rand((3, 3))
        y = actual
        with SchemaCheckMode() as schema_check:
            actual.add_(y)
        # 断言记录的变异和别名是否符合预期
        self.assertEqual(
            [
                ('aten::add_', 'input'),
                ('aten::add_', 'other')
            ],
            schema_check.mutated
        )
        self.assertEqual(
            [
                ('aten::add_', 'input', 'output_0'),
                ('aten::add_', 'other', 'output_0')
            ],
            schema_check.aliasing
        )
    # 测试 SchemaCheckMode 是否记录了使用 as_strided 的突变和别名
    def test_schema_check_mode_mutated_aliasing_as_strided(self):
        # 创建一个形状为 (3, 6, 4) 的随机张量 x
        x = torch.rand((3, 6, 4))
        # 使用 SchemaCheckMode 上下文，并捕获其 schema_check 对象
        with SchemaCheckMode() as schema_check:
            # 对 x 执行 as_strided_ 操作，修改其形状为 [3, 6, 4]，步幅为 [9, 1, 1]
            x.as_strided_([3, 6, 4], [9, 1, 1])
        # 断言 schema_check 对象中的 mutated 记录
        self.assertEqual(
            [
                ('aten::as_strided_', 'input')
            ],
            schema_check.mutated
        )
        # 断言 schema_check 对象中的 aliasing 记录
        self.assertEqual(
            [
                ('aten::as_strided_', 'input', 'output_0')
            ],
            schema_check.aliasing
        )
    
    # 测试 SchemaCheckMode 是否记录了多个输出的 mutations 和 aliases
    def test_schema_check_mode_mutated_aliasing_multiple_outputs(self):
        # 创建一个张量 x 包含 [0., 1., 2., ..., 8.]
        x = torch.arange(9.)
        # 创建与 x 形状相同的张量作为输出
        m_actual = torch.arange(9.)
        e_actual = torch.zeros([9], dtype=torch.int32)
        # 使用 SchemaCheckMode 上下文，并捕获其 schema_check 对象
        with SchemaCheckMode() as schema_check:
            # 调用 torch.frexp 函数，将输出分别存储在 m_actual 和 e_actual 中
            torch.frexp(x, out=(m_actual, e_actual))
        # 断言 schema_check 对象中的 mutated 记录
        self.assertEqual(
            [
                ('aten::frexp', 'mantissa'),
                ('aten::frexp', 'exponent')
            ],
            schema_check.mutated
        )
        # 断言 schema_check 对象中的 aliasing 记录
        self.assertEqual(
            [
                ('aten::frexp', 'mantissa', 'output_0'),
                ('aten::frexp', 'exponent', 'output_1')
            ],
            schema_check.aliasing
        )
    
    # 测试 SchemaCheckMode 是否记录了具有别名输出的 mutations 和 aliases
    def test_schema_check_mode_mutated_aliasing_aliasing_outputs(self):
        # 创建一个形状为 (3, 3) 的随机张量 x
        x = torch.rand((3, 3))
        # 创建一个形状为 (3,) 的零张量 actual
        actual = torch.zeros(3)
        # 使用 SchemaCheckMode 上下文，并捕获其 schema_check 对象
        with SchemaCheckMode() as schema_check:
            # 调用 torch.aminmax 函数，将输出存储在 actual 和 actual 中
            torch.aminmax(x, dim=0, out=[actual, actual])
        # 断言 schema_check 对象中的 mutated 记录
        self.assertEqual(
            [
                ('aten::aminmax', 'min'),
                ('aten::aminmax', 'max')
            ],
            schema_check.mutated
        )
        # 断言 schema_check 对象中的 aliasing 记录
        self.assertEqual(
            [
                ('aten::aminmax', 'min', 'output_0'),
                ('aten::aminmax', 'min', 'output_1'),
                ('aten::aminmax', 'max', 'output_0'),
                ('aten::aminmax', 'max', 'output_1')
            ],
            schema_check.aliasing
        )
    
    # 测试 SchemaCheckMode 是否包装了 torch.Tensor
    def test_schema_check_mode_functionality(self):
        # 创建一个形状为 (3, 3) 的随机张量 x，要求梯度跟踪
        x = torch.rand((3, 3), requires_grad=True)
        # 创建预期输出 expected，通过调用 relu 和 sin 函数实现
        expected = x.relu().sin()
        # 使用 SchemaCheckMode 上下文，不捕获 schema_check 对象
        with SchemaCheckMode():
            # 调用 relu 和 sin 函数，获得实际输出 actual
            actual = x.relu().sin()
        # 断言期望值和实际值相等
        self.assertEqual(expected, actual)
    
    # 测试 SchemaCheckMode 是否在替换默认参数时包装了 torch.Tensor
    def test_schema_check_mode_functionality_default_replaced(self):
        # 创建一个形状为 (3, 3) 的随机张量 x，要求梯度跟踪
        x = torch.rand((3, 3), requires_grad=True)
        # 创建预期输出 expected，通过调用 add 函数，并传入 alpha=2
        expected = x.add(x, alpha=2)
        # 使用 SchemaCheckMode 上下文，不捕获 schema_check 对象
        with SchemaCheckMode():
            # 调用 add 函数，并传入 alpha=2，获得实际输出 actual
            actual = x.add(x, alpha=2)
        # 断言期望值和实际值相等
        self.assertEqual(expected, actual)
    # 测试函数：测试在列表输入下的 SchemaCheckMode 功能
    def test_schema_check_mode_functionality_list_input(self):
        # 创建三个随机矩阵 a, b, c
        a = torch.rand((3, 3))
        b = torch.rand((3, 3))
        c = torch.rand((3, 3))
        # 期望结果是三个矩阵的多重点乘
        expected = torch.linalg.multi_dot([a, b, c])
        # 使用 SchemaCheckMode 包裹执行多重点乘操作
        with SchemaCheckMode():
            actual = torch.linalg.multi_dot([a, b, c])
        # 断言期望结果和实际结果相等
        self.assertEqual(expected, actual)

    # 测试函数：测试在参数之后有通配符的 SchemaCheckMode 功能
    def test_schema_check_mode_functionality_wildcard_after(self):
        # 创建一个随机矩阵 x
        x = torch.rand((3, 3))
        # 期望结果是按照指定方式分块的结果
        expected = x.chunk(6)
        # 使用 SchemaCheckMode 包裹执行分块操作
        with SchemaCheckMode():
            actual = x.chunk(6)
        # 断言期望结果和实际结果相等
        self.assertEqual(expected, actual)

    # 测试函数：测试带有张量输入参数的 SchemaCheckMode 功能
    @unittest.skipIf(not torch._C.has_spectral, "ATen not built with FFT.")
    def test_schema_check_mode_functionality_kwarg_tensor(self):
        # 创建一个随机矩阵 x 和一个随机向量 w
        x = torch.rand((3, 5))
        w = torch.rand(4)
        # 期望结果是进行 stft 转换的结果
        expected = torch.stft(x, 4, win_length=4, window=w, return_complex=True)
        # 使用 SchemaCheckMode 包裹执行 stft 转换操作
        with SchemaCheckMode():
            actual = torch.stft(x, 4, win_length=4, window=w, return_complex=True)
        # 断言期望结果和实际结果相等
        self.assertEqual(expected, actual)

    # 测试函数：测试带有可变输入的 SchemaCheckMode 功能
    def test_schema_check_mode_functionality_mutable_inputs(self):
        # 期望结果是一个不需要梯度的随机矩阵
        expected = torch.rand((3, 3), requires_grad=False)
        # 实际结果是 expected 的克隆
        actual = torch.clone(expected)
        # 在期望结果上执行 sinh_ 操作
        expected.sinh_()
        # 使用 SchemaCheckMode 包裹执行 sinh_ 操作
        with SchemaCheckMode():
            actual.sinh_()
        # 断言期望结果和实际结果相等
        self.assertEqual(expected, actual)

    # 测试函数：测试带有输入别名的 SchemaCheckMode 功能
    def test_schema_check_mode_functionality_aliasing_inputs(self):
        # 期望结果是一个随机矩阵
        expected = torch.rand((3, 3))
        # x 是 expected 的别名
        x = expected
        # actual 是 expected 的克隆
        actual = torch.clone(expected)
        # 在期望结果上执行 add_ 操作，使用别名 x
        expected.add_(x)
        # 使用 SchemaCheckMode 包裹执行 add_ 操作，使用别名 actual
        with SchemaCheckMode():
            actual.add_(actual)
        # 断言期望结果和实际结果相等
        self.assertEqual(expected, actual)

    # 测试函数：测试带有多个输出的 SchemaCheckMode 功能
    def test_schema_check_mode_functionality_with_multiple_outputs(self):
        # 创建一个从 0 到 8 的张量 x
        x = torch.arange(9.)
        # 期望结果是 frexp 函数的两个输出 m_expected 和 e_expected
        m_expected, e_expected = torch.frexp(x)
        # 创建两个用于接收 frexp 函数输出的张量
        m_actual = torch.arange(9.)
        e_actual = torch.zeros([9], dtype=torch.int32)
        # 使用 SchemaCheckMode 包裹执行 frexp 函数，指定输出张量
        with SchemaCheckMode():
            torch.frexp(x, out=(m_actual, e_actual))
        # 断言期望结果和实际结果相等
        self.assertEqual(m_expected, m_actual)
        self.assertEqual(e_expected, e_actual)

    # 测试函数：测试带有多个输入输出别名的 SchemaCheckMode 功能
    def test_schema_check_mode_functionality_with_multiple_outputs_aliasing(self):
        # 创建一个随机矩阵 x
        x = torch.rand((3, 3))
        # 创建一个全零向量 actual
        actual = torch.zeros(3)
        # 使用 SchemaCheckMode 包裹执行 aminmax 函数，使用别名 actual
        with SchemaCheckMode():
            torch.aminmax(x, dim=0, out=[actual, actual])
        # 断言期望结果和实际结果相等
        self.assertEqual(torch.amax(x, dim=0), actual)
    # 定义测试函数，验证在设备输入下的模式功能
    def test_schema_check_mode_functionality_device_input(self):
        # 进入模式检查环境
        with SchemaCheckMode():
            # 创建在 CPU 上的双精度随机张量 x
            x = torch.rand((3, 3), device="cpu", dtype=torch.double)
            # 计算 x + x 得到 y
            y = x + x
        # 断言 x + x 与 y 相等
        self.assertEqual(x + x, y)

    # Tests that SchemaCheckMode wraps Torch.tensor in special training op edge case
    # 验证模式检查包装了 Torch.tensor 在特殊的训练操作边缘情况下
    def test_schema_check_mode_functionality_training_op(self):
        # 创建具有梯度的随机张量 x
        x = torch.rand((3, 3), requires_grad=True)
        # 创建批量归一化层 batch
        batch = torch.nn.BatchNorm1d(3, track_running_stats=True)
        # 期望的输出是 batch 处理过的 x
        expected = batch(x)
        # 进入模式检查环境
        with SchemaCheckMode():
            # 实际的输出是 batch 处理过的 x
            actual = batch(x)
        # 断言期望的输出与实际输出相等
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps Torch.tensor with nested training op edge case
    # 验证模式检查包装了 Torch.tensor 在嵌套训练操作边缘情况下
    def test_schema_check_mode_functionality_nested_training_op(self):
        # 创建随机张量 actual
        actual = torch.rand((3, 3))
        # 创建批量归一化层 batch
        batch = torch.nn.BatchNorm1d(3, track_running_stats=True)
        # 期望的输出是经过一系列操作后的 batch 处理过的 expected
        expected = torch.clone(actual)
        expected.sinh_()
        expected.tanh_()
        expected.relu_()
        expected = batch(expected)

        # 进入模式检查环境
        with SchemaCheckMode():
            # 对 actual 进行一系列操作
            actual.sinh_()
            actual.tanh_()
            actual.relu_()
            actual = batch(actual)
        # 断言期望的输出与实际输出相等
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps Torch.tensor with empty list input
    # 验证模式检查包装了 Torch.tensor 在空列表输入情况下
    def test_schema_check_mode_empty_list_input(self):
        # 期望的输出是至少是一维的空列表
        expected = torch.atleast_1d([])
        # 进入模式检查环境
        with SchemaCheckMode():
            # 实际的输出是至少是一维的空列表
            actual = torch.atleast_1d([])
        # 断言期望的输出与实际输出相等
        self.assertEqual(expected, actual)

    # Tests that an exception is raised for a mismatching mutation
    # 验证对于不匹配的变异会引发异常
    def test_mutation_check_fail(self):
        # 期望会抛出 RuntimeError 异常，带有特定信息
        with self.assertRaisesRegex(RuntimeError, "Argument input is not defined as mutable but was mutated"):
            # 创建随机张量 x 和 y
            x = torch.rand((3, 3))
            y = torch.rand((3, 3))
            # 进入模式检查环境
            with SchemaCheckMode():
                # 使用 IncorrectAliasTensor 对象执行一系列操作
                IncorrectAliasTensor(x).sub(IncorrectAliasTensor(y))

    # Tests that an exception is raised for a mismatching mutation over multiple ops
    # 验证对于多个操作的不匹配变异会引发异常
    def test_mutation_check_fail_multiple_operators(self):
        # 期望会抛出 RuntimeError 异常，带有特定信息
        with self.assertRaisesRegex(RuntimeError, "Argument input is not defined as mutable but was mutated"):
            # 创建随机张量 x 和 y
            x = torch.rand((3, 3))
            y = torch.rand((3, 3))
            # 进入模式检查环境
            with SchemaCheckMode():
                # 使用 IncorrectAliasTensor 对象执行一系列操作
                IncorrectAliasTensor(x).sin().cos().sub(IncorrectAliasTensor(y))

    # Tests that an exception is raised for a mismatching alias
    # 验证对于不匹配的别名会引发异常
    def test_alias_check_fail_simple(self):
        # 期望会抛出 RuntimeError 异常，带有特定信息
        with self.assertRaisesRegex(RuntimeError, "Argument input is not defined to alias output but was aliasing"):
            # 创建具有梯度的随机张量 x 和不带梯度的随机张量 y
            x = torch.rand((3, 3), requires_grad=True)
            y = torch.rand((3, 3))
            # 进入模式检查环境
            with SchemaCheckMode():
                # 使用 IncorrectAliasTensor 对象执行一系列操作
                IncorrectAliasTensor(x).add(IncorrectAliasTensor(y), alpha=2)
    # 测试多个操作符情况下的别名检查失败
    def test_alias_check_fail_multiple_operators(self):
        # 断言引发 RuntimeError 异常，且异常信息包含指定字符串
        with self.assertRaisesRegex(RuntimeError, "Argument input is not defined to alias output but was aliasing"):
            # 创建两个张量，指定需要计算梯度
            x = torch.rand((3, 3), requires_grad=True)
            y = torch.zeros((3, 3), requires_grad=True)
            # 进入模式检查模式
            with SchemaCheckMode():
                # 对 x 使用不正确的别名张量，链式调用 sin()、relu() 和 add() 方法
                IncorrectAliasTensor(x).sin().relu().add(IncorrectAliasTensor(y), alpha=2)

    # 测试多个操作符情况下的中心别名检查失败
    def test_alias_check_fail_multiple_operators_centered(self):
        # 断言引发 RuntimeError 异常，且异常信息包含指定字符串
        with self.assertRaisesRegex(RuntimeError, "Argument input is not defined to alias output but was aliasing"):
            # 创建两个张量，指定需要计算梯度
            x = torch.rand((3, 3), requires_grad=True)
            y = torch.zeros((3, 3), requires_grad=True)
            # 进入模式检查模式
            with SchemaCheckMode():
                # 对 x 使用不正确的别名张量，链式调用 sin()、add() 和 relu() 方法
                IncorrectAliasTensor(x).sin().add(IncorrectAliasTensor(y), alpha=2).relu()

    # 测试输出意外地发生别名的情况下引发异常
    def test_alias_check_fail_outputs_unexpectedly_aliasing(self):
        # 断言引发 RuntimeError 异常，且异常信息包含指定字符串
        with self.assertRaisesRegex(RuntimeError, "Outputs 0 and 1 alias unexpectedly"):
            # 创建一个张量
            x = torch.rand((3, 3))
            # 进入模式检查模式
            with SchemaCheckMode() as s:
                # 对 x 使用不正确的别名张量，调用 aminmax(dim=0) 方法
                IncorrectAliasTensor(x).aminmax(dim=0)

    # 当编写此文件时，Python 操作注册并不存在。
    # 可能需要重新编写整个文件以使用它，但我只是添加了额外的测试。
    def test_alias_check_fail_custom_ops_secretly_aliasing(self):
        # 定义一个函数，调用 torch.ops.bad_schemas.secretly_aliasing
        def f(x):
            return torch.ops.bad_schemas.secretly_aliasing(x)

        # 创建一个张量
        x = torch.rand((3, 3))
        # 断言引发 RuntimeError 异常，且异常信息包含指定字符串
        with self.assertRaisesRegex(RuntimeError, "not defined to alias output but was aliasing"):
            # 进入模式检查模式
            with SchemaCheckMode() as s:
                # 调用 f 函数
                out = f(x)

    # 测试自定义操作暗中改变的情况下引发异常
    def test_alias_check_fail_custom_ops_secretly_mutating(self):
        # 定义一个函数，调用 torch.ops.bad_schemas.secretly_mutating
        def f(x):
            return torch.ops.bad_schemas.secretly_mutating(x)

        # 创建一个张量
        x = torch.rand((3, 3))
        # 断言引发 RuntimeError 异常，且异常信息包含指定字符串
        with self.assertRaisesRegex(RuntimeError, "not defined as mutable but was mutated"):
            # 进入模式检查模式
            with SchemaCheckMode() as s:
                # 调用 f 函数
                out = f(x)

    # 测试自定义操作输出直接为输入的情况下引发异常
    def test_alias_check_fail_custom_ops_output_is_input(self):
        # 定义一个函数，调用 torch.ops.bad_schemas.output_is_input
        def f(x):
            return torch.ops.bad_schemas.output_is_input(x)

        # 创建一个张量
        x = torch.rand((3, 3))
        # 断言引发 RuntimeError 异常，且异常信息包含指定字符串
        with self.assertRaisesRegex(RuntimeError, "are not allowed to directly return inputs"):
            # 进入模式检查模式
            with SchemaCheckMode() as s:
                # 调用 f 函数
                out = f(x)

    # 测试 is_alias_of 方法的预期返回结果
    def test_is_alias_of_basic(self):
        # 创建两个张量，指定需要计算梯度
        x = torch.rand((3, 3), requires_grad=True)
        y = torch.rand((3, 3), requires_grad=True)
        # 对 y 执行 inplace 加法操作，可能会导致别名
        y = x.add(x, alpha=2)
        # 断言 x 是否和自身别名
        self.assertTrue(torch._C._is_alias_of(x, x))
        # 断言 x 是否和 y 别名
        self.assertFalse(torch._C._is_alias_of(x, y))

    # 测试 is_alias_of 方法处理空容器的预期返回结果
    def test_is_alias_of_empty_container(self):
        # 创建空列表 x
        x = []
        # 创建一个随机张量 y，要求其梯度
        y = torch.rand((3, 3), requires_grad=True)
        # 断言 x 不是其自身的别名
        self.assertFalse(torch._C._is_alias_of(x, x))
        # 断言 x 不是 y 的别名
        self.assertFalse(torch._C._is_alias_of(x, y))

    # Tests that overlaps returns as expected
    def test_overlaps_basic(self):
        # 创建两个随机张量 x 和 y，要求其梯度
        x = torch.rand((3, 3), requires_grad=True)
        y = torch.rand((3, 3), requires_grad=True)
        # 创建包含 x 和 y 的列表 z
        z = [x, y]
        # 断言 x 是其自身的重叠
        self.assertTrue(torch._C._overlaps(x, x))
        # 断言 x 不与 y 重叠
        self.assertFalse(torch._C._overlaps(x, y))
        # 断言 z 与 x 重叠
        self.assertTrue(torch._C._overlaps(z, x))
        # 断言 z 与 y 重叠
        self.assertTrue(torch._C._overlaps(z, y))

    # Tests that overlaps returns correctly with empty containers
    def test_overlaps_empty_container(self):
        # 创建空列表 x 和包含一个随机张量的列表 y
        x = []
        y = [torch.rand((3, 3), requires_grad=True)]
        # 断言 y 不与 x 重叠
        self.assertFalse(torch._C._overlaps(y, x))
        # 断言 y 与其自身重叠
        self.assertTrue(torch._C._overlaps(y, y))

    # Tests that SchemaInfo Bindings work as expected
    def test_schema_info_bind_basic(self):
        # 定义一个测试模式类 SchemaInfoBindTestMode，继承自 TorchDispatchMode
        class SchemaInfoBindTestMode(TorchDispatchMode):
            def __init__(self, test_self):
                self.test_self = test_self

            # 实现 __torch_dispatch__ 方法
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # 规范化函数调用参数，仅使用关键字参数进行规范化
                named_arg_list = normalize_function(
                    func,
                    args,
                    kwargs,
                    normalize_to_only_use_kwargs=True
                ).kwargs
                # 创建两个 SchemaInfo 对象，基于函数的 _schema 属性
                schema_info_value_test = torch._C._SchemaInfo(func._schema)
                schema_info_values_test = torch._C._SchemaInfo(func._schema)
                # 断言两个 SchemaInfo 对象中指定的参数不会重叠
                self.test_self.assertFalse(schema_info_value_test.may_alias(
                    torch._C._SchemaArgument(torch._C._SchemaArgType.input, 0),
                    torch._C._SchemaArgument(torch._C._SchemaArgType.input, 1)))
                self.test_self.assertFalse(schema_info_values_test.may_alias(
                    torch._C._SchemaArgument(torch._C._SchemaArgType.input, 0),
                    torch._C._SchemaArgument(torch._C._SchemaArgType.input, 1)))
                # 将参数值添加到 schema_info_value_test
                for i in named_arg_list:
                    schema_info_value_test.add_argument_value(i, named_arg_list[i])
                # 将所有参数值添加到 schema_info_values_test
                schema_info_values_test.add_argument_values(named_arg_list)
                # 断言两个 SchemaInfo 对象中指定的参数会重叠
                self.test_self.assertTrue(schema_info_value_test.may_alias(
                    torch._C._SchemaArgument(torch._C._SchemaArgType.input, 0),
                    torch._C._SchemaArgument(torch._C._SchemaArgType.input, 1)))
                self.test_self.assertTrue(schema_info_values_test.may_alias(
                    torch._C._SchemaArgument(torch._C._SchemaArgType.input, 0),
                    torch._C._SchemaArgument(torch._C._SchemaArgType.input, 1)))

                # 调用原始函数并返回结果
                return func(*args, **kwargs)
        
        # 创建一个随机张量 x
        x = torch.rand((3, 3))
        # 使用 SchemaInfoBindTestMode 类进行测试
        with SchemaInfoBindTestMode(self) as schemaInfoCheck:
            # 调用 x 的 add 方法
            x.add(x)
# 定义一个测试类，继承自 JitTestCase 类，用于测试模式和操作信息的模式检查
class TestSchemaCheckModeOpInfo(JitTestCase):

    # 用装饰器 ops 注册测试函数，设置 op_db 和 dtypes 参数为 OpDTypes.supported
    @ops(op_db, dtypes=OpDTypes.supported)
    def test_schema_correctness(self, device, dtype, op):
        # 检查是否为 torch.complex32 类型，如果是则直接返回，因为 torch.equal 不支持该类型
        if (dtype == torch.complex32):
            return
        
        # 对于 op 的每个输入样本，获取输入数据，不要求梯度
        for sample in op.sample_inputs(device, dtype, requires_grad=False):
            # 使用 SchemaCheckMode 上下文管理器，进行模式检查
            with SchemaCheckMode():
                # 调用操作 op，传入输入数据以及其他参数
                op(sample.input, *sample.args, **sample.kwargs)

# 实例化设备类型相关的测试，传入 TestSchemaCheckModeOpInfo 类和全局变量，仅适用于 "cpu" 和 "cuda" 设备
instantiate_device_type_tests(TestSchemaCheckModeOpInfo, globals(), only_for=("cpu", "cuda"))

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == '__main__':
    run_tests()
```
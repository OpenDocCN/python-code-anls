# `.\pytorch\test\export\test_export.py`

```
# Owner(s): ["oncall: export"]
# flake8: noqa

# 导入必要的库和模块
import copy                 # 导入 copy 模块，用于复制对象
import dataclasses          # 导入 dataclasses 模块，用于定义数据类
import io                   # 导入 io 模块，用于处理流式输入输出
import logging              # 导入 logging 模块，用于日志记录
import re                   # 导入 re 模块，用于正则表达式操作
import unittest             # 导入 unittest 模块，用于编写和运行测试
import warnings             # 导入 warnings 模块，用于警告控制
from contextlib import contextmanager  # 从 contextlib 模块中导入 contextmanager 上下文管理器
from dataclasses import dataclass      # 从 dataclasses 模块中导入 dataclass 装饰器
from re import escape                  # 从 re 模块中导入 escape 函数，用于正则表达式的转义处理
from typing import Dict, List          # 导入 Dict 和 List 类型提示

import torch                           # 导入 PyTorch 深度学习库
import torch._dynamo as torchdynamo    # 导入 torchdynamo 模块，这是 PyTorch 的私有模块
import torch.nn.functional as F        # 导入 PyTorch 中的函数式接口模块

# 导入 functorch 中的实验性控制流模块
from functorch.experimental.control_flow import cond, map
from torch import Tensor               # 导入 PyTorch 中的 Tensor 类型
from torch._dynamo.test_case import TestCase  # 从 torch._dynamo.test_case 模块导入 TestCase 类
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse  # 导入 _ExportPassBaseDeprecatedDoNotUse 类
from torch._export.utils import (      # 导入 _export 中的工具函数和类
    get_buffer,                        # get_buffer 函数
    get_param,                         # get_param 函数
    is_buffer,                         # is_buffer 函数
    is_param,                          # is_param 函数
    register_dataclass_as_pytree_node  # register_dataclass_as_pytree_node 函数
)
from torch._subclasses import FakeTensorMode  # 导入 FakeTensorMode 类
from torch.export import Dim, dynamic_dim, export, unflatten  # 导入 export 相关模块和类
from torch.export._trace import (      # 导入 _trace 模块中的 _export 和 _export_to_torch_ir 函数
    _export,
    _export_to_torch_ir,
    DEFAULT_EXPORT_DYNAMO_CONFIG
)
from torch.export.graph_signature import InputKind  # 导入 InputKind 类
from torch.fx.experimental.proxy_tensor import make_fx  # 导入 make_fx 函数
from torch.fx.experimental.symbolic_shapes import ShapeEnv  # 导入 ShapeEnv 类
from torch.testing import FileCheck     # 导入 FileCheck 类
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION  # 导入平台支持的 CUDA 相关常量
from torch.testing._internal.common_device_type import onlyCPU, onlyCUDA  # 导入仅 CPU 和仅 CUDA 的测试装饰器
from torch.testing._internal.common_utils import (  # 导入常用的测试工具函数和类
    find_library_location,             # find_library_location 函数
    IS_FBCODE,                         # IS_FBCODE 常量
    IS_MACOS,                          # IS_MACOS 常量
    IS_SANDCASTLE,                     # IS_SANDCASTLE 常量
    IS_WINDOWS,                        # IS_WINDOWS 常量
    run_tests,                         # run_tests 函数
    TEST_TRANSFORMERS,                 # TEST_TRANSFORMERS 常量
    TestCase as TorchTestCase          # 将 TestCase 类重命名为 TorchTestCase，以避免名称冲突
)
from torch.utils._pytree import (        # 导入 PyTree 相关的工具函数和类
    LeafSpec,                            # LeafSpec 类
    tree_flatten,                        # tree_flatten 函数
    tree_map,                            # tree_map 函数
    tree_unflatten,                      # tree_unflatten 函数
    TreeSpec,                            # TreeSpec 类
    treespec_dumps,                      # treespec_dumps 函数
    treespec_loads                       # treespec_loads 函数
)

try:
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor  # 尝试导入 torchrec.sparse.jagged_tensor 中的 KeyedJaggedTensor 类
    HAS_TORCHREC = True                   # 设置标志指示成功导入了 torchrec.sparse.jagged_tensor
except ImportError:
    HAS_TORCHREC = False                  # 捕获 ImportError，表明未导入 torchrec.sparse.jagged_tensor

try:
    from . import testing                  # 尝试从当前目录导入 testing 模块
except ImportError:
    import testing                         # 如果失败，则从全局环境中导入 testing 模块

# The following import pattern matters as `test_export.export` is patched
# in other files (like test_export_nonstrict.py). `torch.export.export`
# will invalidate the patch.
from torch.export import export           # 从 torch.export 模块导入 export 函数

# 定义 PyTorch 库中的自定义操作
torch.library.define("testlib::returns_tensor_symint", "(Tensor x) -> (Tensor, SymInt)")
torch.library.define(
    "testlib::foo",
    "(Tensor(a!) x, Tensor(b!) z) -> (Tensor, Tensor, Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)
torch.library.define(
    "testlib::foo_mutated",
    "(Tensor(a!) x) -> (Tensor, Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)
torch.library.define(
    "testlib::foo_functional",
    "(Tensor x) -> (Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)
torch.library.define(
    "testlib::foo_unbacked",
    "(Scalar x) -> (Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)

# 定义上述操作的实现

@torch.library.impl("testlib::returns_tensor_symint", "cpu")
@torch.library.impl_abstract("testlib::returns_tensor_symint")
def returns_tensor_symint_impl(x):
    return x, x.shape[0]

@torch.library.impl("testlib::foo", "cpu")
@torch._dynamo.disable
def foo_impl(x, z):
    x.add_(5)
    z.add_(5)
    return x, z, x + z
# 定义一个抽象函数，标记为 Torch 库的抽象实现，用于外部库的调用
@torch.library.impl_abstract("testlib::foo")
def foo_abstract(x, z):
    return x, z, x + z


# 定义一个具体实现函数，标记为 Torch 库的实现，内部调用了 foo 函数并对结果进行处理
@torch.library.impl("testlib::foo_mutated", "CompositeImplicitAutograd")
def foo_mutated(x):
    a, b, c = torch.ops.testlib.foo(x, x.cos())
    return a, a.cos()


# 定义另一个具体实现函数，标记为 Torch 库的实现，内部调用了 foo 函数并对结果进行处理
@torch.library.impl("testlib::foo_functional", "CompositeImplicitAutograd")
def foo_functional(x):
    a, b, c = torch.ops.testlib.foo(x.cos(), x.cos())
    return a.cos()


# 定义一个具体实现函数，标记为 Torch 库的实现，根据输入 x 的大小返回固定的张量
@torch.library.impl("testlib::foo_unbacked", "CompositeImplicitAutograd")
def foo_unbacked(x):
    if x > 2:
        return torch.ones(4, 4)
    if x < 6:
        return torch.ones(4, 4)
    return torch.ones(4, 4)


# 定义一个数据类 Inp，包含 x（张量）、y（张量列表）、z（张量字典）
@dataclass
class Inp:
    x: Tensor
    y: List[Tensor]
    z: Dict[str, Tensor]


# 定义一些常量后缀
NON_STRICT_SUFFIX = "_non_strict"
RETRACEABILITY_SUFFIX = "_retraceability"
SERDES_SUFFIX = "_serdes"
PREDISPATCH_SUFFIX = "_pre_dispatch"
TRAINING_IR_DECOMP_SUFFIX = "_training_ir_to_decomp"


# 函数：检查测试名是否以指定后缀结尾
def is_non_strict_test(test_name):
    return test_name.endswith(NON_STRICT_SUFFIX)


# 函数：检查测试名是否以指定后缀结尾
def is_retracebility_test(test_name):
    return test_name.endswith(RETRACEABILITY_SUFFIX)


# 函数：检查测试名是否以指定后缀结尾
def is_serdes_test(test_name):
    return test_name.endswith(SERDES_SUFFIX)


# 单元测试类：如果 dynamo 不支持，则跳过测试
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestDynamismExpression(TestCase):
    # 测试函数：导出内联约束
    def test_export_inline_constraints(self):
        # 内部类 Module：定义前向传播，创建张量并返回填充的张量
        class Module(torch.nn.Module):
            def forward(self, x):
                b = x.item()
                torch._check_is_size(b)
                return torch.full((b, 1), 1)

        f = Module()
        inp = (torch.tensor([3]),)
        ref = f(*inp)

        gm = export(f, inp)
        res = gm.module()(*inp)

        self.assertTrue(torchdynamo.utils.same(ref, res))

        gm = make_fx(f, tracing_mode="symbolic")(*inp)
        res = gm(*inp)
        self.assertTrue(torchdynamo.utils.same(ref, res))

    # 测试函数：导出约束错误不在范围内
    def test_export_constraints_error_not_in_range(self):
        # 内部类 InvalidInputConflictWithInputConstraints：定义前向传播，返回输入 x 加 1
        class InvalidInputConflictWithInputConstraints(torch.nn.Module):
            def forward(self, x):
                return x + 1

        inp = torch.zeros([3])
        dim_x = torch.export.Dim("dim_x", min=6)
        with self.assertRaisesRegex(torch._dynamo.exc.UserError, "not in range"):
            torch.export.export(
                InvalidInputConflictWithInputConstraints(),
                (inp,),
                dynamic_shapes={"x": {0: dim_x}},
            )

    # 测试函数：导出切片最大大小
    def test_export_slice_maxsize(self):
        # 内部类 Slice：定义前向传播，调用 torch.ops.aten.slice.Tensor 进行切片操作
        class Slice(torch.nn.Module):
            def forward(self, *args):
                return torch.ops.aten.slice.Tensor(*args)

        inp = (torch.rand((10, 3, 224, 224)), 0, 0, 9223372036854775807)
        dynamic_shapes = (({0: Dim("dim")}, None, None, None),)
        torch.export.export(
            Slice(),
            inp,
            dynamic_shapes=dynamic_shapes,
        )
    # 定义一个测试函数，用于测试导出过程中的约束条件错误
    def test_export_constraints_error(self):
        # 定义一个继承自 torch.nn.Module 的子类 ConflictingConstraints
        class ConflictingConstraints(torch.nn.Module):
            # 重写 forward 方法
            def forward(self, x):
                # 将张量 x 转换为标量 b
                b = x.item()
                # 检查 b 是否为合法的尺寸
                torch._check_is_size(b)
                # 检查 b 是否大于等于 4
                torch._check(b >= 4)
                # 检查 b 是否小于等于 5
                torch._check(b <= 5)
                # 返回一个形状为 (b, 1) 的全一张量
                return torch.full((b, 1), 1)

        # 创建输入张量 inp，包含一个元素为 [3] 的张量
        inp = (torch.tensor([3]),)
        # 对 ConflictingConstraints 模型进行导出
        ep = export(ConflictingConstraints(), inp)

        # 使用断言检查是否抛出 RuntimeError，并且错误消息匹配指定的正则表达式
        with self.assertRaisesRegex(
            RuntimeError, r"Invalid value range for 3 between \[4, 5\]"
        ):
            # 调用导出模型的 forward 方法，传入张量 [3]
            ep.module()(torch.tensor([3]))

    # 定义一个测试函数，用于测试默认情况下是否假设静态形状
    def test_export_assume_static_by_default(self):
        # 定义一个继承自 torch.nn.Module 的子类 Module
        class Module(torch.nn.Module):
            # 重写 forward 方法，接收一个形状为 torch.Tensor 的输入 x
            def forward(self, x: torch.Tensor):
                # 如果输入张量 x 的第一个维度为 4
                if x.shape[0] == 4:
                    # 返回 x 加 1 的结果
                    return x + 1
                else:
                    # 否则返回输入张量 x 本身
                    return x

        # 创建 Module 类的实例 branch_on_shape
        branch_on_shape = Module()
        # 创建输入张量 inp，包含一个形状为 (4, 5) 的随机张量
        inp = (torch.rand(4, 5),)

        # 对 branch_on_shape 模型进行导出
        export(branch_on_shape, inp)
# 如果在 Windows 系统上运行测试，则跳过该测试用例
@unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
# 如果当前环境不支持 Dynamo，则跳过该测试用例
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestExport(TestCase):
    def _test_export_same_as_eager(self, f, args, kwargs=None):
        kwargs = kwargs or {}
        # 导出函数 f 的计算图
        exported_program = export(f, args, kwargs)
        # 检查导出后的程序运行结果是否与直接调用 f 函数的结果一致
        self.assertEqual(exported_program.module()(*args, **kwargs), f(*args, **kwargs))
        # 以下代码因为 .module() 方法不支持，所以被注释掉
        # reversed_kwargs = {key: kwargs[key] for key in reversed(kwargs)}
        # self.assertEqual(
        #     exported_program.module()(*args, **reversed_kwargs), f(*args, **reversed_kwargs)
        # )

    def test_basic(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                return x[0] + y

        f = Module()
        inp = ([torch.ones(1, 3)], torch.ones(1, 3))
        self._test_export_same_as_eager(f, inp)

    # 因为非严格模式在训练 IR 中不支持 (T193692164)，所以标记为预期失败
    @testing.expectedFailureTrainingIRToRunDecomp
    def test_external_call_non_strict_real_tensor(self):
        class ExternalMethod:
            def add(self, x):
                return x + x

        class Basic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.external_add = ExternalMethod().add

            def forward(self, x):
                return self.external_add(x)

        f = Basic()
        args = (torch.randn(1, 3),)
        # 导出函数 f 的计算图，使用非严格模式
        ep = export(f, args, strict=False)
        # 检查导出后的程序运行结果是否与直接调用 f 函数的结果一致
        self.assertEqual(ep.module()(*args), f(*args))

    def test_colon_parameter(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册名为 "foo:bar" 的参数
                self.register_parameter("foo:bar", torch.nn.Parameter(torch.ones(3, 3)))

            def forward(self, x):
                # 返回 x 加上模型中的 "foo:bar" 参数
                return x + getattr(self, "foo:bar")

        # 导出模型 M 并传入参数
        ep = export(M(), (torch.randn(3, 3),))
        x = torch.randn(3, 3)
        # 检查导出后的程序运行结果是否与直接调用模型 M 的结果一致
        self.assertEqual(ep.module()(x), M()(x))
    def test_conv_dynamic(self):
        # Simple module for demonstration
        # 定义一个用于演示的简单模块
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 初始化一个2D卷积层，输入通道为3，输出通道为32，卷积核大小为3，填充为1
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, padding=1
                )
                # ReLU激活函数
                self.relu = torch.nn.ReLU()
                # 最大池化层，核大小为3
                self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # 对输入x进行卷积
                a = self.conv(x)
                # 将y加到a上（in-place加法）
                a.add_(y)
                # 对a先使用ReLU，再进行最大池化操作
                return self.maxpool(self.relu(a))

        example_args = (torch.randn(2, 3, 256, 256), torch.ones(2, 32, 256, 256))
        # 定义动态形状
        dynamic_shapes = {"x": {0: Dim("batch")}, "y": {0: Dim("batch")}}
        m = M()
        # 导出模型为ExportedProgram对象
        exported_program: torch.export.ExportedProgram = export(
            m, args=example_args, dynamic_shapes=dynamic_shapes
        )

        args = (torch.randn(17, 3, 256, 256), torch.ones(17, 32, 256, 256))
        # 检查导出的程序与原始模型在给定参数下的输出是否一致
        self.assertEqual(exported_program.module()(*args), m(*args))
        args = (torch.randn(15, 3, 256, 256), torch.ones(15, 32, 256, 256))
        self.assertEqual(exported_program.module()(*args), m(*args))

        from torch._export import capture_pre_autograd_graph

        # 使用capture_pre_autograd_graph捕获预自动求导图（graph）
        gm: torch.fx.GraphModule = capture_pre_autograd_graph(
            m, args=example_args, dynamic_shapes=dynamic_shapes
        )

        args = (torch.randn(17, 3, 256, 256), torch.ones(17, 32, 256, 256))
        # 检查捕获的图与原始模型在给定参数下的输出是否一致
        self.assertEqual(gm(*args), m(*args))
        args = (torch.randn(15, 3, 256, 256), torch.ones(15, 32, 256, 256))
        self.assertEqual(gm(*args), m(*args))

    # Errors because non-strict is not supported in training IR (T193692164)
    @testing.expectedFailureTrainingIRToRunDecomp
    # 预期测试失败，因为在训练IR中不支持非严格模式（T193692164）
    def test_basic_non_strict_real_tensor(self):
        class Basic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个参数，形状为(1, 3)，值为随机数
                self.param = torch.nn.Parameter(torch.randn(1, 3))

            def forward(self, x, y):
                # 返回x的第一个元素加上y再减去self.param
                return x[0] + y - self.param

        f = Basic()
        args = ([torch.randn(1, 3)], torch.randn(1, 3))
        # 导出模型为ExportedProgram对象，非严格模式
        ep = export(f, args, strict=False)
        # 检查导出的程序与原始模型在给定参数下的输出是否一致
        self.assertEqual(ep.module()(*args), f(*args))

    # Errors because non-strict is not supported in training IR (T193692164)
    @testing.expectedFailureTrainingIRToRunDecomp
    # 预期测试失败，因为在训练IR中不支持非严格模式（T193692164）
    def test_basic_non_strict_fake_tensor(self):
        # 定义一个测试函数，测试在非严格模式下的虚假张量处理
        class Basic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个参数，形状为 3x2 的张量
                self.param = torch.nn.Parameter(torch.randn(3, 2))

            def forward(self, x, y):
                # 模型的前向传播，返回 x 的第一个元素加上 y 减去 self.param
                return x[0] + y - self.param

        # 创建一个虚假张量模式的实例
        fake_mode = FakeTensorMode(shape_env=ShapeEnv(tracked_fakes=[]))
        # 实例化 Basic 类，构建模型 f
        f = Basic()
        # 在虚假张量模式下执行以下代码块
        with fake_mode:
            # 准备模型 f 的输入参数
            args = ([torch.empty(3, 2)], torch.empty(3, 2))
        # 导出模型 f，strict 参数设为 False
        ep = export(f, args, strict=False)
        # 准备一个输入示例 inputs
        inputs = ([torch.randn(3, 2)], torch.randn(3, 2))
        # 断言导出模型的结果与模型 f 在给定输入上的结果相等
        self.assertEqual(ep.module()(*inputs), f(*inputs))
    def test_non_strict_dynamic_shapes(self):
        # 定义名为 test_non_strict_dynamic_shapes 的测试方法
        class Foo(torch.nn.Module):
            # 定义名为 Foo 的类，继承自 torch.nn.Module
            def __init__(self):
                # 初始化方法
                super().__init__()
                # 在模块中注册名为 "u" 和 "v" 的缓冲区张量，初始值为 1
                self.register_buffer("u", torch.ones(1))
                self.register_buffer("v", torch.ones(1))

            def forward(self, x, ys, zs, c):
                # 前向传播方法
                # 计算 y，将 ys 和 zs 中的张量相加
                y = ys[0] + ys[1] + zs["a"] + zs["b"]
                # 修改模块中的 self.v，增加 3
                self.v.add_(3)
                # 计算 w，使用 self.u 和修改后的 self.v 计算差
                w = self.u - self.v
                # 检查输入 x 和 c 的形状是否符合条件
                if x.shape[0] < 3 and c.shape[0] != 4:
                    return x + w, x + y
                else:
                    return x - w, x - y

        # 创建 Foo 类的实例 foo
        foo = Foo()

        # 定义输入数据 inp，包含四个张量
        inp = (
            torch.ones(5),                          # x，形状为 (5,)
            [torch.zeros(5), torch.ones(5)],        # ys，包含两个形状为 (5,) 的张量
            {"a": torch.zeros(5), "b": torch.ones(5)},  # zs，包含两个键值对，值为形状为 (5,) 的张量
            torch.ones(4),                          # c，形状为 (4,)
        )
        # 创建一个维度对象 dim，最小值为 3
        dim = torch.export.Dim("dim", min=3)
        # 定义动态形状 dynamic_shapes，包含四个元组
        dynamic_shapes = (
            {0: dim},                               # 对应 inp 中第一个元素 x 的维度约束
            [{0: dim}, {0: dim}],                   # 对应 inp 中 ys 的两个元素的维度约束
            {"a": {0: dim}, "b": {0: dim}},         # 对应 inp 中 zs 的两个元素的维度约束
            None                                    # c 不需要维度约束
        )

        # 导出 foo 模块，使用输入数据 inp 和动态形状 dynamic_shapes，strict 参数设为 False
        ep_ns = torch.export.export(
            foo, inp, dynamic_shapes=dynamic_shapes, strict=False
        )

        # 创建一个形状不符合预期的输入 bad_runtime_inp1
        bad_runtime_inp1 = (
            torch.ones(6),                          # x 形状为 (6,)
            [torch.zeros(5), torch.ones(5)],        # ys 和原始输入相同
            {"a": torch.zeros(5), "b": torch.ones(5)},  # zs 和原始输入相同
            torch.ones(4),                          # c 和原始输入相同
        )
        # 使用 assertRaisesRegex 检查运行时错误，验证 x 的形状不符合预期
        with self.assertRaisesRegex(
            RuntimeError,
            escape(
                "Expected input at *args[1][0].shape[0] to be equal to 6, but got 5"
            ),
        ):
            ep_ns.module()(*bad_runtime_inp1)

        # 创建另一个形状不符合预期的输入 bad_runtime_inp2
        bad_runtime_inp2 = (
            torch.ones(5),                          # x 和原始输入相同
            [torch.zeros(5), torch.ones(5)],        # ys 和原始输入相同
            {"a": torch.zeros(5), "b": torch.ones(5)},  # zs 和原始输入相同
            torch.ones(6),                          # c 形状为 (6,)
        )
        # 使用 assertRaisesRegex 检查运行时错误，验证 c 的形状不符合预期
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[3].shape[0] to be equal to 4, but got 6"),
        ):
            ep_ns.module()(*bad_runtime_inp2)

        # 创建一个形状符合预期的输入 good_runtime_inp
        good_runtime_inp = (
            torch.ones(7),                          # x 形状为 (7,)
            [torch.zeros(7), torch.ones(7)],        # ys 和原始输入相同，形状为 (7,)
            {"a": torch.zeros(7), "b": torch.ones(7)},  # zs 和原始输入相同，形状为 (7,)
            torch.ones(4),                          # c 和原始输入相同
        )
        # 验证形状符合预期的输入是否能够正常运行，无异常抛出
        ep_ns.module()(*good_runtime_inp)

        # 创建一个示例中形状不符合预期的输入 bad_example_inp
        bad_example_inp = (
            torch.ones(2),                          # x 形状为 (2,)
            [torch.zeros(2), torch.ones(2)],        # ys 和原始输入相同，形状为 (2,)
            {"a": torch.zeros(2), "b": torch.ones(2)},  # zs 和原始输入相同，形状为 (2,)
            torch.ones(4),                          # c 和原始输入相同
        )
        # 使用 assertRaisesRegex 检查符号形状错误，验证示例输入的形状不符合约束条件
        with self.assertRaisesRegex(
            torch.fx.experimental.symbolic_shapes.ConstraintViolationError,
            "2 not in range.*3,",
        ):
            # 导出 foo 模块，使用 bad_example_inp 和 dynamic_shapes，strict 参数设为 False
            ep_ns = torch.export.export(
                foo, bad_example_inp, dynamic_shapes=dynamic_shapes, strict=False
            )
    # 定义一个测试方法，用于测试非严格动态形状建议修复
    def test_non_strict_dynamic_shapes_suggested_fixes(self):
        # 定义一个名为Foo的类，继承自torch.nn.Module
        class Foo(torch.nn.Module):
            # 定义前向传播方法，接受参数x和c
            def forward(self, x, c):
                # 如果x的第一个维度小于等于6，返回x加1和c加2的结果
                if x.shape[0] <= 6:
                    return x + 1, c + 2
                # 否则，返回x减1和c减2的结果
                else:
                    return x - 1, c - 2
        
        # 创建Foo类的实例对象foo
        foo = Foo()
        
        # 定义一个不良示例输入bad_example_inp，包含两个张量torch.ones(5)和torch.ones(4)
        bad_example_inp = (
            torch.ones(5),
            torch.ones(4),
        )
        
        # 创建一个维度对象dim，使用torch.export.Dim类，名称为"dim"，最小值为3
        dim = torch.export.Dim("dim", min=3)
        
        # 定义动态形状dynamic_shapes，包含一个字典和一个None
        dynamic_shapes = (
            {0: dim},
            None,
        )
        
        # 使用assertRaisesRegex断言上下文管理器，期望抛出torch._dynamo.exc.UserError异常
        # 并检查异常消息中包含特定的错误信息和建议修复信息
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Constraints violated \\(dim\\)!(.*\n)*.*"
            "Not all values of dim.*satisfy the generated guard(.*\n)*.*"
            "Suggested fixes:(.*\n)*.*"
            "dim = Dim\\('dim', min=3, max=6\\)",
        ):
            # 导出foo模型，使用bad_example_inp作为输入，传递动态形状参数dynamic_shapes，并且不严格执行
            torch.export.export(
                foo, bad_example_inp, dynamic_shapes=dynamic_shapes, strict=False
            )

    # 定义一个测试方法，用于测试状态的基本数据类型
    def test_state_primitives(self):
        # 定义一个名为M的类，继承自torch.nn.Module
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                self.x = 1  # 初始化self.x为1
                self.y = {"k": 2}  # 初始化self.y为字典{"k": 2}
                self.z = (3,)  # 初始化self.z为元组(3,)
            
            # 前向传播方法，接受参数x
            def forward(self, x):
                # 更新self.x，使其加上4
                self.x = self.x + 4
                # 更新self.y中键"k"对应的值，使其加上5
                self.y["k"] = self.y["k"] + 5
                # 更新self.z，使其成为包含更新后元素的新元组
                self.z = (self.z[0] + 6,)
                # 返回x加上self.x、self.y["k"]和self.z[0]的和
                return x + self.x + self.y["k"] + self.z[0]
        
        # 创建M类的实例对象ep，并使用torch.randn(2, 3)作为输入
        ep = export(M(), (torch.randn(2, 3),))
        
        # 使用assertTrue断言，验证ep模型在torch.zeros(2, 3)上的输出与torch.ones(2, 3) * 21是否全部接近
        self.assertTrue(
            torch.allclose(ep.module()(torch.zeros(2, 3)), torch.ones(2, 3) * 21)
        )

    # Predispatch有不同的预期结果
    @testing.expectedFailureTrainingIRToRunDecomp  # T193700910
    def test_export_preserve_linear_at_aot_level(self):
        # 定义一个名为 Foo 的 Torch 模块
        class Foo(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个输入为 3，输出为 3 的线性层
                self.linear = torch.nn.Linear(3, 3)

            # 前向传播方法
            def forward(self, x):
                # 对输入 x 应用线性层
                x = self.linear(x)
                # 使用 torch.ops.aten.chunk.default 将 x 按行分块为 3 份
                return torch.ops.aten.chunk.default(x, 3, 0)

        # 导出 Foo 模块并运行分解
        gm = (
            torch.export.export(
                Foo(),
                (torch.randn(3, 3),),
            )
            .run_decompositions({}, _preserve_ops=(torch.ops.aten.linear.default,))
            .graph_module
        )

        # 确保代码生成中保留了 linear，因为它是 CompositeImplicitAutograd 功能操作
        # chunk 是 CompositeImplicitAutograd 非功能操作，我们对它进行了分解
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, p_linear_weight, p_linear_bias, x):
    # 调用 Torch 的底层操作接口实现线性变换，包括权重和偏置
    linear = torch.ops.aten.linear.default(x, p_linear_weight, p_linear_bias);  x = p_linear_weight = p_linear_bias = None
    # 将线性变换结果按照第一个维度（通常是 batch 维度）进行分割
    split = torch.ops.aten.split.Tensor(linear, 1);  linear = None
    # 获取分割后的第一个部分
    getitem = split[0]
    # 获取分割后的第二个部分
    getitem_1 = split[1]
    # 获取分割后的第三个部分
    getitem_2 = split[2];  split = None
    # 返回分割后的结果作为元组
    return (getitem, getitem_1, getitem_2)



    # TODO(yidi)
    # 针对调用 run_decomposition() 的测试用例，预期出现的失败情况。
    # 顶层条件节点具有预先存在的元数据，
    # 这会由于 interpreter.run() 中 cond 是解释运行中的单一节点而覆盖子图中操作符的元数据，
    # 我们通过复制当前节点的元数据来保留所有在解释期间创建的节点的元数据。
    @testing.expectedFailurePreDispatchRunDecomp
    @testing.expectedFailureRetraceability
    @testing.expectedFailureTrainingIRToRunDecomp  # T193700910
    def test_export_cond_preserve_torch_fn_for_subgraphs(self):
        # 定义一个简单的子模块，包含一个对输入进行余弦操作的方法
        class MySubModule(torch.nn.Module):
            def foo(self, x):
                return x.cos()

            def forward(self, x):
                return self.foo(x)

        # 定义一个包含条件分支的模块类
        class CondBranchClassMethod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.subm = MySubModule()

            def bar(self, x):
                return x.sin()

            def forward(self, x):
                # 根据输入大小判断条件，选择调用 MySubModule 的 forward 方法或者本类的 bar 方法
                return cond(x.shape[0] <= 2, self.subm.forward, self.bar, [x])

        # 准备示例输入
        example_inputs = (torch.randn(1, 3, 3, 3),)
        # 创建模块实例并设置为评估模式
        m = CondBranchClassMethod()
        m.eval()
        # 导出模块，并获取导出后的图模块
        gm = export(m, example_inputs).module()

        # 收集实际的 Torch 函数名称
        actual_torch_fns = []
        for mod in gm.modules():
            for node in mod.graph.nodes:
                if node.name in {"sin", "cos"}:
                    torch_fn = node.meta.get("torch_fn")
                    print(torch_fn)
                    actual_torch_fns.append(torch_fn)

        # 预期的 Torch 函数名称
        exp_torch_fns = [
            ("cos_1", "method_descriptor.cos"),
            ("sin_1", "method_descriptor.sin"),
        ]
        # 断言实际 Torch 函数名称与预期是否一致
        self.assertEqual(actual_torch_fns, exp_torch_fns)
    # 定义一个测试方法，用于测试动态维度约束的基本情况
    def test_derived_dim_basic(self):
        # 定义一个继承自torch.nn.Module的内部类Foo
        class Foo(torch.nn.Module):
            # 实现Module类的forward方法，接受x和y作为输入，返回它们的和，但是y的第一个元素被省略
            def forward(self, x, y):
                return x + y[1:]

        # 创建Foo类的实例
        foo = Foo()

        # 生成大小为5的随机张量x和大小为6的随机张量y
        x, y = torch.randn(5), torch.randn(6)
        
        # 创建一个维度对象dimx，命名为"dimx"，最小值为3，最大值为6
        dimx = torch.export.Dim("dimx", min=3, max=6)

        # 创建一个维度对象dimy，命名为"dimy"，最小值为4，最大值为7，但是这个设置不能工作
        dimy = torch.export.Dim("dimy", min=4, max=7)  # doesn't work
        
        # 使用断言确保在导出过程中引发特定类型的异常，异常消息中包含维度约束的错误信息
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated \\(dimy\\)!(.*\n)*.*"
                "The values of dimy.*must always be related to the values of dimx.*by.*(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "dimy = dimx \\+ 1"
            ),
        ):
            # 导出foo模型，传入输入x和y，设置动态形状约束为{0: dimx}和{0: dimy}
            export(
                foo,
                (x, y),
                dynamic_shapes=({0: dimx}, {0: dimy}),
            )

        # 更新维度对象dimy，设置其为dimx的两倍，但是这个设置也不能工作
        dimy = dimx * 2  # doesn't work
        
        # 使用断言确保在导出过程中引发特定类型的异常，异常消息指出期望的输入尺寸不符合预期
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected input.*size.* to be equal to 2\\*dimx, where dimx = 5, but got 6",
        ):
            # 再次尝试导出foo模型，传入输入x和y，设置动态形状约束为{0: dimx}和{0: dimy}
            export(
                foo,
                (x, y),
                dynamic_shapes=({0: dimx}, {0: dimy}),
            )

        # 更新维度对象dimy，设置其为dimx加1，这个设置能够正常工作
        dimy = dimx + 1  # works
        
        # 执行模型导出操作，将foo模型和输入x、y一起导出，并设置动态形状约束为{0: dimx}和{0: dimy}
        ep = export(
            foo,
            (x, y),
            dynamic_shapes=({0: dimx}, {0: dimy}),
        )
        
        # 使用断言确保在运行时引发特定类型的异常，异常消息指出输入形状不符合预期
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 5, but got 6",
        ):
            # 调用导出模型的module部分，并传入大小为4和6的随机张量，确保引发异常
            ep.module()(torch.randn(4), torch.randn(6))

        # 使用断言确保模型在输入大小为4和5的随机张量上输出的第一维大小为4
        self.assertEqual(ep.module()(torch.randn(4), torch.randn(5)).size()[0], 4)
    def test_derived_dim_nested(self):
        # 定义内嵌类 Foo，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 定义前向传播方法 forward，接受两个参数 x 和 y
            def forward(self, x, y):
                # 返回 x 和 y[1::2] 的和
                return x + y[1::2]

        # 创建 Foo 类的实例 foo
        foo = Foo()

        # 生成随机张量 x 和 y，各自长度为 5 和 11
        x, y = torch.randn(5), torch.randn(11)
        # 创建维度对象 dimx，命名为 "dimx"，最小值为 3，最大值为 6
        dimx = torch.export.Dim("dimx", min=3, max=6)
        # 计算 dimy，为 dimx 的两倍加一
        dimy = dimx * 2 + 1  # works
        # 导出模型 foo，传入参数 (x, y)，并指定动态形状为 {0: dimx} 和 {0: dimy}
        ep = export(
            foo,
            (x, y),
            dynamic_shapes=({0: dimx}, {0: dimy}),
        )
        # 断言导出模型的执行结果张量的第一个维度大小为 4
        self.assertEqual(ep.module()(torch.randn(4), torch.randn(9)).size()[0], 4)

        # 重新定义内嵌类 Foo，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 定义前向传播方法 forward，接受两个参数 z 和 y
            def forward(self, z, y):
                # 返回 z[1:] 和 y[1::2] 的和
                return z[1:] + y[1::2]

        # 创建 Foo 类的实例 foo
        foo = Foo()

        # 生成随机张量 z 和 y，各自长度为 6 和 11
        z, y = torch.randn(6), torch.randn(11)

        # 设置 dimz 等于之前定义的 dimx
        dimz = dimx
        # 计算 dimy，为 dimx 的两倍减一
        dimy = dimx * 2 - 1  # works
        # 导出模型 foo，传入参数 (z, y)，并指定动态形状为 {0: dimz} 和 {0: dimy}
        ep = export(
            foo,
            (z, y),
            dynamic_shapes=({0: dimz}, {0: dimy}),
        )
        
        # 使用断言检查导出模型的执行结果张量的第一个维度大小为 4
        self.assertEqual(ep.module()(torch.randn(5), torch.randn(9)).size()[0], 4)

        # 将 dimz 设置为 dimx 加一
        dimz = dimx + 1
        # 计算 dimy，为 dimx 的两倍减一
        dimy = dimx * 2 - 1  # doesn't work

        # 使用断言检查是否抛出预期的异常信息
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected input.*size.*to be equal to 2\\*dimx - 1, where dimx = 5, but got 11",
        ):
            # 导出模型 foo，传入参数 (z, y)，并指定动态形状为 {0: dimz} 和 {0: dimy}
            export(
                foo,
                (z, y),
                dynamic_shapes=({0: dimz}, {0: dimy}),
            )

        # 计算 dimy，为 dimx 的两倍加一
        dimy = dimx * 2 + 1  # works
        # 导出模型 foo，传入参数 (z, y)，并指定动态形状为 {0: dimz} 和 {0: dimy}
        ep = export(
            foo,
            (z, y),
            dynamic_shapes=({0: dimz}, {0: dimy}),
        )

        # 使用断言检查是否抛出预期的异常信息
        with self.assertRaisesRegex(
            RuntimeError, "Expected input.*shape.*to be <= 7, but got 8"
        ):
            # 执行导出模型的执行结果张量的第一个维度大小为 8 的测试
            ep.module()(torch.randn(8), torch.randn(15))
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 9, but got 8",
        ):
            # 执行导出模型的执行结果张量的第一个维度大小为 5 的测试
            ep.module()(torch.randn(5), torch.randn(8))

        # 使用断言检查导出模型的执行结果张量的第一个维度大小为 4
        self.assertEqual(ep.module()(torch.randn(5), torch.randn(9)).size()[0], 4)
    def test_derived_dim_integer(self):
        # 定义一个内部类 Foo，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, w):
                # 如果输入张量 w 的第一个维度能被 2 整除
                if w.shape[0] % 2 == 0:
                    # 返回 w 的偶数索引位置的子集
                    return w[::2]
                else:
                    # 返回 w 的第二个到倒数第二个元素的奇数索引位置的子集
                    return w[1:-1:2]

        # 创建 Foo 类的实例
        foo = Foo()

        # 生成一个形状为 (10,) 的随机张量 w
        w = torch.randn(10)
        
        # 创建一个导出维度对象 Dim，命名为 "dimx"，范围在 3 到 6 之间
        dimx = torch.export.Dim("dimx", min=3, max=6)
        
        # 计算一个导出维度 dimw，它是 dimx 的两倍加一，这里会导致错误
        dimw = dimx * 2 + 1  # doesn't work
        
        # 使用 assertRaisesRegex 断言捕获异常，确保导出函数在特定条件下抛出用户错误
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected shape.*= 10 of input Tensor to be "
            "of the form 2\\*dimx \\+ 1, where dimx is an integer",
        ):
            # 调用 export 函数，传入 foo 和 w，以及动态维度参数
            export(
                foo,
                (w,),
                dynamic_shapes=({0: dimw},),
            )

        # 更新 dimw 为 dimx 的两倍，这次应该能正常工作
        dimw = dimx * 2  # works
        
        # 调用 export 函数，传入 foo 和 w，以及动态维度参数
        ep = export(
            foo,
            (w,),
            dynamic_shapes=({0: dimw},),
        )

        # 使用 assertRaisesRegex 断言捕获异常，确保导出函数在特定条件下抛出运行时错误
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*= 9 to be "
            "of the form 2\\*s1, where s1 is an integer",
        ):
            # 调用导出模型的模块并传入形状为 (9,) 的随机张量
            ep.module()(torch.randn(9))

        # 使用 self.assertEqual 断言确保导出模型的模块处理形状为 (8,) 的随机张量后输出的大小
        self.assertEqual(ep.module()(torch.randn(8)).size()[0], 4)

        # 使用 assertRaisesRegex 断言捕获异常，确保导出函数在特定条件下抛出运行时错误
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be <= 12, but got 14",
        ):
            # 调用导出模型的模块并传入形状为 (14,) 的随机张量
            ep.module()(torch.randn(14))

    def test_derived_dim_repeat_derived(self):
        # 定义一个内部类 Foo，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, u, v):
                # 返回 u 和 v 的偶数索引位置的元素之和
                return u[::2] + v[::2]

        # 创建 Foo 类的实例
        foo = Foo()

        # 生成形状为 (10,) 的随机张量 u 和 v
        u, v = torch.randn(10), torch.randn(10)
        
        # 创建一个导出维度对象 Dim，命名为 "dimx"，范围在 3 到 6 之间
        dimx = torch.export.Dim("dimx", min=3, max=6)
        
        # 计算一个导出维度 dimw，它是 dimx 的两倍，这里应该能正常工作
        dimw = dimx * 2  # works
        
        # 调用 export 函数，传入 foo 和 (u, v)，以及动态维度参数
        ep = export(
            foo,
            (u, v),
            dynamic_shapes=({0: dimw}, {0: dimw}),
        )

        # 使用 self.assertEqual 断言确保导出模型的模块处理形状为 (8,) 的随机张量后输出的大小
        self.assertEqual(ep.module()(torch.randn(8), torch.randn(8)).size()[0], 4)

    def test_derived_dim_out_of_order(self):
        # 创建一个导出维度对象 Dim，命名为 "dimy"，范围在 5 到 7 之间
        dimy = torch.export.Dim("dimy", min=5, max=7)
        
        # 计算一个导出维度 dimx，它是 dimy 减一，这种计算顺序是错误的，实际上 dimy = dimx + 1
        dimx = dimy - 1  # out of order, effectively dimy = dimx + 1
        
        # 计算一个导出维度 dimz，它是 dimy 加一，这种计算顺序也是错误的，实际上 dimz = dimx + 2
        dimz = dimy + 1  # out of order, effectively dimz = dimx + 2

        # 定义一个内部类 Foo，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x, y, z):
                # 返回 x、y 和 z 的子集之和
                return x + y[1:] + z[2:]

        # 创建 Foo 类的实例
        foo = Foo()

        # 生成形状分别为 (5,)、(6,) 和 (7,) 的随机张量 u、v、w
        u, v, w = torch.randn(5), torch.randn(6), torch.randn(7)
        
        # 调用 export 函数，传入 foo 和 (u, v, w)，以及动态维度参数
        ep = export(
            foo,
            (u, v, w),
            dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}),
        )

        # 使用 assertRaisesRegex 断言捕获异常，确保导出函数在特定条件下抛出运行时错误
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 8, but got 5",
        ):
            # 调用导出模型的模块并传入形状为 (6,)、(7,) 和 (5,) 的随机张量
            ep.module()(torch.randn(6), torch.randn(7), torch.randn(5))

        # 使用 self.assertEqual 断言确保导出模型的模块处理形状为 (6,)、(7,) 和 (8,) 的随机张量后输出的大小
        self.assertEqual(
            ep.module()(torch.randn(6), torch.randn(7), torch.randn(8)).size()[0], 6
        )
    # 定义一个测试方法，测试派生维度的顺序，以及重复使用的派生维度
    def test_derived_dim_out_of_order_repeat_derived(self):
        # 创建一个维度对象dimy，命名为"dimy"，最小值为5，最大值为7
        dimy = torch.export.Dim("dimy", min=5, max=7)
        # 创建一个维度对象dimx，其值比dimy少1，虽然顺序颠倒，但实际上是dimy = dimx + 1
        dimx = dimy - 1
        # 创建一个维度对象dimz，其值比dimy多1，虽然顺序颠倒，但实际上是dimz = dimx + 2
        dimz = dimy + 1
        # 将dimx的值赋给dimx1
        dimx1 = dimx
        # 创建一个维度对象dimx2，其值为dimz减去2，实际上等同于dimx
        dimx2 = dimz - 2

        # 定义一个名为Foo的类，继承自torch.nn.Module
        class Foo(torch.nn.Module):
            # 定义forward方法，接受x, y, z, x1, x2五个参数，返回它们的加和
            def forward(self, x, y, z, x1, x2):
                return x + y[1:] + z[2:] + x1 + x2

        # 创建Foo类的实例对象foo
        foo = Foo()

        # 初始化五个张量u, v, w, u1, u2，分别包含5、6、7、5、5个随机数
        u, v, w, u1, u2 = (
            torch.randn(5),
            torch.randn(6),
            torch.randn(7),
            torch.randn(5),
            torch.randn(5),
        )

        # 调用export函数，将foo和(u, v, w, u1, u2)作为参数，以及动态形状的字典
        ep = export(
            foo,
            (u, v, w, u1, u2),
            dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}, {0: dimx1}, {0: dimx2}),
        )

        # 使用assertRaisesRegex断言捕获RuntimeError异常，验证其错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 6, but got 5",
        ):
            # 调用ep.module()的返回对象，并传入不符合预期形状的五个随机张量
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(8),
                torch.randn(6),
                torch.randn(5),
            )

        # 使用assertEqual断言验证ep.module()返回对象的结果与预期值的大小
        self.assertEqual(
            # 调用ep.module()的返回对象，并传入符合预期形状的五个随机张量，获取其大小的第一个维度值，并验证是否为6
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(8),
                torch.randn(6),
                torch.randn(6),
            ).size()[0],
            6,
        )

        # 再次调用export函数，将foo和(u, v, w, u, u)作为参数，以及相同的动态形状字典
        ep = export(
            foo,
            (u, v, w, u, u),  # 重复使用相同的输入
            dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}, {0: dimx1}, {0: dimx2}),
        )

        # 使用assertRaisesRegex再次断言捕获RuntimeError异常，验证其错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 6, but got 5",
        ):
            # 调用ep.module()的返回对象，并传入不符合预期形状的五个随机张量
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(8),
                torch.randn(6),
                torch.randn(5),
            )

        # 使用assertEqual再次断言验证ep.module()返回对象的结果与预期值的大小
        self.assertEqual(
            # 调用ep.module()的返回对象，并传入符合预期形状的五个随机张量，获取其大小的第一个维度值，并验证是否为6
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(8),
                torch.randn(6),
                torch.randn(6),
            ).size()[0],
            6,
        )
    # 测试在特定情况下派生维度的根源是否能够专门化
    def test_specialize_derived_dim_roots(self):
        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            # 定义模型的前向传播函数
            def forward(self, x, y):
                return x.reshape([-1]) + y

        # 创建一个名为 dy 的维度对象，设置最小值为 6
        dy = Dim("dy", min=6)
        # 创建两个张量 x 和 y，其中 x 的形状为 (6, 2)，y 的形状为 (12)
        x, y = torch.randn(6, 2), torch.randn(12)
        # 定义动态形状字典，指定 x 和 y 的形状分别为 (dy - 6, 2) 和 (dy,)
        dynamic_shapes = {
            "x": (dy - 6, 2),
            "y": (dy,),
        }
        try:
            # 尝试导出模型 Foo，传入输入张量 x 和 y，并指定动态形状
            export(Foo(), (x, y), dynamic_shapes=dynamic_shapes)
            # 如果 export() 调用没有因动态形状错误而失败，则抛出异常
            raise Exception(
                "export() call should have failed with dynamic shapes error."
            )
        except torch._dynamo.exc.UserError as exc:
            # 定义期望的错误消息正则表达式，检查是否符合预期的动态形状错误
            expected_error_msg = (
                "Specializations unexpectedly required \(dy\)!(.*\n)*.*"
                ".*dy - 6.*must be specialized to 6 because the guards generated for it are too complex(.*\n)*.*"
                "Suggested fixes(.*\n)*.*"
                ".*dy = 12(.*\n)*.*"
            )
            # 断言捕获的异常信息中包含期望的错误消息正则表达式
            self.assertTrue(re.search(expected_error_msg, exc.args[0]) is not None)
            # 断言捕获的异常信息中不建议修复非根源维度 dy - 6 = 6 的错误
            self.assertTrue(
                "dy - 6 = 6" not in exc.args[0]
            )  # don't suggest fix for non-root dim

    # 测试保留复合操作的无效性
    def test_keep_composite_ops_invalid(self):
        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            # 定义类的初始化方法，初始化一个线性层 self.linear
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            # 定义模型的前向传播函数
            def forward(self, x):
                # 将输入 x 通过 self.linear 线性层处理
                x = self.linear(x)
                # 返回 torch.ops.aten.chunk.default 操作处理后的结果
                return torch.ops.aten.chunk.default(x, 3, 0)

        # 使用断言检查运行时错误消息是否包含 "aten.chunk.default is a mutating/aliasing op"
        with self.assertRaisesRegex(
            RuntimeError, "aten.chunk.default is a mutating/aliasing op"
        ):
            # 导出模型 Foo，传入一个形状为 (3, 3) 的随机张量作为输入
            _ = torch.export.export(
                Foo(),
                (torch.randn(3, 3),),
            ).run_decompositions({}, _preserve_ops=(torch.ops.aten.chunk.default,))

        # 使用断言检查运行时错误消息是否包含 "aten.add.Tensor is not CompositeImplicitAutograd op, so we will preserve it as"
        with self.assertRaisesRegex(
            RuntimeError,
            "aten.add.Tensor is not CompositeImplicitAutograd op, so we will preserve it as",
        ):
            # 导出模型 Foo，传入一个形状为 (3, 3) 的随机张量作为输入
            _ = torch.export.export(
                Foo(),
                (torch.randn(3, 3),),
            ).run_decompositions({}, _preserve_ops=(torch.ops.aten.add.Tensor,))

        # 使用断言检查运行时错误消息是否包含 "aten.sym_size.default is a metadata query function"
        with self.assertRaisesRegex(
            RuntimeError, "aten.sym_size.default is a metadata query function"
        ):
            # 导出模型 Foo，传入一个形状为 (3, 3) 的随机张量作为输入
            _ = torch.export.export(
                Foo(),
                (torch.randn(3, 3),),
            ).run_decompositions({}, _preserve_ops=(torch.ops.aten.sym_size.default,))

        # 使用断言检查运行时错误消息是否包含 "We can't detect aten.native_batch_norm.default as a functional op statically"
        with self.assertRaisesRegex(
            RuntimeError,
            "We can't detect aten.native_batch_norm.default as a functional op statically",
        ):
            # 导出模型 Foo，传入一个形状为 (3, 3) 的随机张量作为输入
            _ = torch.export.export(
                Foo(),
                (torch.randn(3, 3),),
            ).run_decompositions(
                {}, _preserve_ops=(torch.ops.aten.native_batch_norm.default,)
            )
    def test_keep_composite_ops_linear_convd(self):
        # 定义一个名为 test_keep_composite_ops_linear_convd 的测试方法
        class MyLinear(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的类 MyLinear
            def __init__(self):
                # 构造方法，初始化模块的参数
                super().__init__()
                # 使用正态分布随机初始化权重张量，形状为 (20, 98)
                self.weight = torch.randn(20, 98)
                # 使用正态分布随机初始化偏置张量，形状为 (20,)
                self.bias = torch.randn(20)

            def forward(self, x):
                # 前向传播方法，实现线性变换
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的类 Foo
            def __init__(self):
                # 构造方法，初始化模块的参数
                super().__init__()
                # 创建一个二维卷积层，输入通道 16，输出通道 33，卷积核大小为 3x3
                self.conv = torch.nn.Conv2d(16, 33, 3)
                # 创建一个一维卷积层，输入通道 16，输出通道 33，卷积核大小为 3
                self.conv1d = torch.nn.Conv1d(16, 33, 3)
                # 创建一个自定义的线性层对象
                self.linear = MyLinear()

            def forward(self, x, y):
                # 前向传播方法，定义网络结构
                x_conv = self.conv(x)
                # 对输入 x 进行二维卷积操作
                y_conv_1d = self.conv1d(y)
                # 对输入 y 进行一维卷积操作
                x_linear = self.linear(x_conv)
                # 对经过二维卷积后的结果 x_conv 进行自定义线性层的线性变换
                return x_linear.cos() + y_conv_1d.sum()
                # 返回经过线性变换后的 x_linear 的余弦值加上 y_conv_1d 的求和结果

        ep = torch.export.export(
            Foo(), (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50))
        )
        # 导出模型 Foo 的表示形式，并传入两个随机张量作为输入
        ep_has_linear_convd = ep.run_decompositions(
            # 运行分解操作
            decomp_table={},
            # 使用空字典作为分解表
            _preserve_ops=testing._COMPOSITE_OPS_THAT_CAN_BE_PRESERVED_TESTING_ONLY,
            # 传入用于测试目的的保留操作列表
        )
        self.assertExpectedInline(
            str(ep_has_linear_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, c_linear_weight, c_linear_bias, x, y):
    # 执行二维卷积操作，使用给定的权重和偏置
    conv2d = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias);  x = p_conv_weight = p_conv_bias = None
    # 执行一维卷积操作，使用给定的权重和偏置
    conv1d = torch.ops.aten.conv1d.default(y, p_conv1d_weight, p_conv1d_bias);  y = p_conv1d_weight = p_conv1d_bias = None
    # 对二维卷积结果进行视图变换，调整形状为[31680, 98]
    view = torch.ops.aten.view.default(conv2d, [31680, 98]);  conv2d = None
    # 对线性层的权重进行维度置换，调整形状为[1, 0]
    permute = torch.ops.aten.permute.default(c_linear_weight, [1, 0]);  c_linear_weight = None
    # 执行矩阵乘加操作，对视图和置换后的权重进行加权求和，并加上偏置
    addmm = torch.ops.aten.addmm.default(c_linear_bias, view, permute);  c_linear_bias = view = permute = None
    # 对视图变换后的张量执行余弦函数操作
    view_1 = torch.ops.aten.view.default(addmm, [20, 33, 48, 20]);  addmm = None
    # 对一维卷积结果沿指定维度进行求和
    sum_1 = torch.ops.aten.sum.dim_IntList(conv1d, []);  conv1d = None
    # 将余弦函数结果和求和结果进行张量相加
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    # 返回结果元组
    return (add,)
    # 调用 torch 的 ATen 操作，对 addmm 张量进行默认视图变换，指定新形状 [20, 33, 48, 20]；然后将 addmm 置为 None
    view_1 = torch.ops.aten.view.default(addmm, [20, 33, 48, 20]);  addmm = None
    # 调用 torch 的 ATen 操作，计算 view_1 张量的余弦值；然后将 view_1 置为 None
    cos = torch.ops.aten.cos.default(view_1);  view_1 = None
    # 调用 torch 的 ATen 操作，对 convolution 张量沿指定维度求和，维度为空列表；然后将 convolution 置为 None
    sum_1 = torch.ops.aten.sum.dim_IntList(convolution, []);  convolution = None
    # 调用 torch 的 ATen 操作，将 cos 张量与 sum_1 张量进行元素级相加；然后将 cos 和 sum_1 置为 None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    # 返回包含 add 张量的单元素元组
    return (add,)""",
        )
# 定义了一个名为 forward 的方法，用于执行神经网络的前向传播过程
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, b_linear_weight, b_linear_bias, x, y):
    # 使用 torch 的 ATen 操作执行二维卷积运算，计算卷积结果并存储在 convolution 中
    convolution = torch.ops.aten.convolution.default(x, p_conv_weight, p_conv_bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  x = p_conv_weight = p_conv_bias = None
    # 使用 torch 的 ATen 操作执行一维卷积运算，计算卷积结果并存储在 convolution_1 中
    convolution_1 = torch.ops.aten.convolution.default(y, p_conv1d_weight, p_conv1d_bias, [1], [0], [1], False, [0], 1);  y = p_conv1d_weight = p_conv1d_bias = None
    # 使用 torch 的 ATen 操作将 convolution 的结果变换成指定形状 [31680, 98]，存储在 view 中
    view = torch.ops.aten.view.default(convolution, [31680, 98]);  convolution = None
    # 使用 torch 的 ATen 操作对 b_linear_weight 进行转置操作，结果存储在 t 中
    t = torch.ops.aten.t.default(b_linear_weight);  b_linear_weight = None
    # 使用 torch 的 ATen 操作执行矩阵乘法和加法操作，计算结果并存储在 addmm 中
    addmm = torch.ops.aten.addmm.default(b_linear_bias, view, t);  b_linear_bias = view = t = None
    # 使用 torch 的 ATen 操作将 addmm 的结果变换成指定形状 [20, 33, 48, 20]，存储在 view_1 中
    view_1 = torch.ops.aten.view.default(addmm, [20, 33, 48, 20]);  addmm = None
    # 使用 torch 的 ATen 操作计算 view_1 中张量的余弦值，结果存储在 cos 中
    cos = torch.ops.aten.cos.default(view_1);  view_1 = None
    # 使用 torch 的 ATen 操作计算 convolution_1 中张量的元素之和，结果存储在 sum_1 中
    sum_1 = torch.ops.aten.sum.default(convolution_1);  convolution_1 = None
    # 使用 torch 的 ATen 操作将 cos 和 sum_1 张量对应位置相加，结果存储在 add 中
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    # 返回包含 add 结果的元组作为方法的输出
    return (add,)
    convolution_1 = torch.ops.aten.convolution.default(y, p_conv1d_weight, p_conv1d_bias, [1], [0], [1], False, [0], 1);  y = p_conv1d_weight = p_conv1d_bias = None
    # 执行一个 1 维卷积操作，使用给定的权重和偏置参数，返回结果到 convolution_1，然后清空 y, p_conv1d_weight, p_conv1d_bias 变量
    view = torch.ops.aten.view.default(convolution, [31680, 98]);  convolution = None
    # 将 convolution 张量重塑为指定维度 [31680, 98]，并将结果保存到 view 变量，然后清空 convolution 变量
    permute = torch.ops.aten.permute.default(b_linear_weight, [1, 0]);  b_linear_weight = None
    # 对 b_linear_weight 张量进行维度置换，顺序变为 [1, 0]，结果保存到 permute 变量，然后清空 b_linear_weight 变量
    addmm = torch.ops.aten.addmm.default(b_linear_bias, view, permute);  b_linear_bias = view = permute = None
    # 执行矩阵乘法加法运算，将 view 和 permute 张量相乘，然后加上 b_linear_bias，结果保存到 addmm 变量，然后清空 b_linear_bias, view, permute 变量
    view_1 = torch.ops.aten.view.default(addmm, [20, 33, 48, 20]);  addmm = None
    # 将 addmm 张量重塑为指定维度 [20, 33, 48, 20]，结果保存到 view_1 变量，然后清空 addmm 变量
    cos = torch.ops.aten.cos.default(view_1);  view_1 = None
    # 计算 view_1 张量的余弦值，结果保存到 cos 变量，然后清空 view_1 变量
    sum_1 = torch.ops.aten.sum.dim_IntList(convolution_1, []);  convolution_1 = None
    # 对 convolution_1 张量沿着所有维度求和，结果保存到 sum_1 变量，然后清空 convolution_1 变量
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    # 将 cos 和 sum_1 张量相加，结果保存到 add 变量，然后清空 cos 和 sum_1 变量
    return (add,)""",
        )


这段代码片段包含了一系列使用 PyTorch 的原生操作（torch.ops.aten）来进行张量操作，包括卷积、视图重塑、维度置换、矩阵乘法加法、余弦运算和求和操作。每个操作都会将结果保存到一个新的变量中，并在操作完成后清空之前使用的变量，以节省内存和确保代码的干净性。
    def test_simple_export_for_training(self):
        # 定义一个名为 Foo 的内部类，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个线性层，输入维度为2，输出维度为2
                self.linear = torch.nn.Linear(2, 2)

            # 前向传播方法
            def forward(self, x):
                return self.linear(x)

        # 创建一个 Foo 类的实例对象
        eager_model = Foo()
        # 使用 torch.export._trace._export_for_training 方法导出用于训练的模型
        ep_for_training = torch.export._trace._export_for_training(
            eager_model, (torch.ones(2, 2),)
        )
        # 使用 self.assertExpectedInline 进行断言验证输出结果
        self.assertExpectedInline(
            str(ep_for_training.graph_module.code).strip(),
            """\
def forward(self, p_linear_weight, p_linear_bias, x):
    # 调用 Torch 提供的线性运算，使用给定的权重和偏置对输入 x 进行线性变换
    linear = torch.ops.aten.linear.default(x, p_linear_weight, p_linear_bias);  x = p_linear_weight = p_linear_bias = None
    return (linear,)
"""
        )
gm = ep_for_training.module()
self.assertExpectedInline(
    str(gm.code).strip(),
    """\
def forward(self, x):
    # 将输入 x 按照指定规范展开为平铺形式
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    # 获取模型中注册的线性层的权重和偏置
    linear_weight = self.linear.weight
    linear_bias = self.linear.bias
    # 使用 Torch 提供的线性运算对输入 x 进行线性变换
    linear = torch.ops.aten.linear.default(x, linear_weight, linear_bias);  x = linear_weight = linear_bias = None
    # 将计算结果按指定规范重新组装成数据结构
    return pytree.tree_unflatten((linear,), self._out_spec)""",
)

self.assertTrue(
    torch.allclose(gm(torch.ones(2, 2)), eager_model(torch.ones(2, 2)))
)

def test_export_for_training_with_mutation(self):
    class Foo(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.ones(4, 4))

        def forward(self, x):
            # 对输入 x 进行原位加法操作
            x.add_(5)
            # 对模型中注册的缓冲区进行原位加法操作
            self.buffer.add_(5)
            # 返回加法结果
            return x + self.buffer

    eager_model_for_export = Foo()
    eager_model_for_testing = Foo()
    # 导出用于训练的 EagerPy 模型
    ep_for_training = torch.export._trace._export_for_training(
        eager_model_for_export, (torch.ones(4, 4),)
    )
    self.assertExpectedInline(
        str(ep_for_training.graph_module.code).strip(),
        """\
def forward(self, b_buffer, x):
    # 对输入 x 执行原位加法操作
    add_ = torch.ops.aten.add_.Tensor(x, 5);  x = None
    # 对缓冲区 b_buffer 执行原位加法操作
    add__1 = torch.ops.aten.add_.Tensor(b_buffer, 5);  b_buffer = None
    # 对两个加法操作的结果执行加法运算
    add = torch.ops.aten.add.Tensor(add_, add__1);  add_ = add__1 = None
    return (add,)""",
    )
    gm = ep_for_training.module()
    self.assertExpectedInline(
        str(gm.code).strip(),
        """\
def forward(self, x):
    # 将输入 x 按照指定规范展开为平铺形式
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    # 获取模型中注册的缓冲区
    buffer = self.buffer
    # 对输入 x 执行原位加法操作
    add_ = torch.ops.aten.add_.Tensor(x, 5);  x = None
    # 对缓冲区 buffer 执行原位加法操作
    add__1 = torch.ops.aten.add_.Tensor(buffer, 5);  buffer = None
    # 对两个加法操作的结果执行加法运算
    add = torch.ops.aten.add.Tensor(add_, add__1);  add_ = add__1 = None
    # 将计算结果按指定规范重新组装成数据结构
    return pytree.tree_unflatten((add,), self._out_spec)""",
    )

    self.assertTrue(
        torch.allclose(
            gm(torch.ones(4, 4)), eager_model_for_testing(torch.ones(4, 4))
        )
    )
    def test_export_for_training_with_dynamic_shapes(self):
        # 定义一个简单的 PyTorch 模块 Foo，包含一个缓冲区和一个前向传播方法
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个 4x4 的全一张量作为缓冲区
                self.register_buffer("buffer", torch.ones(4, 4))

            def forward(self, x):
                # 在输入张量 x 上加 5
                x.add_(5)
                # 在模块的缓冲区张量上也加 5
                self.buffer.add_(5)
                # 返回 x 加上缓冲区张量元素总和的结果
                return x + self.buffer.sum()

        # 创建三个 Foo 类型的模块实例，分别用于不同的导出需求
        eager_model_for_export_training = Foo()
        eager_model_for_export_inference = Foo()
        eager_model_for_testing = Foo()

        # 导出用于训练的模块，使用 _export_for_training 函数
        ep_for_training = torch.export._trace._export_for_training(
            eager_model_for_export_training,
            (torch.ones(4, 4),),  # 输入是一个 4x4 的全一张量
            dynamic_shapes=({0: Dim("x")},),  # 指定动态形状
        )

        # 断言导出的模块在给定输入上的输出与测试模块在相同输入上的输出是相近的
        self.assertTrue(
            torch.allclose(
                ep_for_training.module()(torch.ones(2, 4)),
                eager_model_for_testing(torch.ones(2, 4)),
            )
        )

        # 实际导出用于推断的模块，使用 export 函数
        ep_for_real = export(
            eager_model_for_export_inference,
            (torch.ones(4, 4),),  # 输入同样是一个 4x4 的全一张量
            dynamic_shapes=({0: Dim("x")},),  # 同样指定动态形状
        )

        # 断言训练导出模块和推断导出模块的范围约束是相同的
        self.assertEqual(
            str(ep_for_training.range_constraints), str(ep_for_real.range_constraints)
        )

    def test_export_for_training_with_container_type(self):
        # 定义一个简单的 PyTorch 模块 Foo，接受一个容器类型作为输入
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个 4x4 的全一张量作为缓冲区
                self.register_buffer("buffer", torch.ones(4, 4))

            def forward(self, container):
                # 从容器中获取第一个和第二个张量，并分别加 5
                x = container[0][0]
                y = container[0][1]
                x.add_(5)
                y.add_(5)
                # 返回 x + y + 缓冲区张量元素总和的结果
                return x + y + self.buffer.sum()

        # 创建一个 Foo 类型的模块实例
        eager_model = Foo()

        # 导出用于训练的模块，使用 _export_for_training 函数
        ep_for_training = torch.export._trace._export_for_training(
            eager_model,
            ([torch.ones(4, 4), torch.ones(4, 4)],),  # 输入是一个包含两个 4x4 的全一张量的列表
        )

        # 断言导出的模块在给定输入上的输出与原始模块在相同输入上的输出是相近的
        self.assertTrue(
            torch.allclose(
                ep_for_training.module()(
                    ([torch.ones(4, 4), torch.ones(4, 4)]),
                ),
                eager_model(([torch.ones(4, 4), torch.ones(4, 4)])),
            )
        )

    def test_export_for_training_run_decomp(self):
        # 定义一个包含线性层的 PyTorch 模块 Foo，以及一个缓冲区
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个 2x2 的全一张量作为缓冲区
                self.register_buffer("buffer", torch.ones(2, 2))
                # 定义一个线性层，输入输出维度都为 2
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                # 在缓冲区张量上加 5
                self.buffer.add_(5)
                # 返回线性层处理输入 x 后的结果加上缓冲区张量元素总和
                return self.linear(x) + self.buffer.sum()

        # 创建一个 Foo 类型的模块实例
        eager_model = Foo()

        # 导出用于训练的模块，使用 _export_for_training 函数
        ep_for_training = torch.export._trace._export_for_training(
            eager_model,
            (torch.ones(2, 2),),  # 输入是一个 2x2 的全一张量
        )

        # 运行分解得到推断用的导出模块
        ep_for_inference = ep_for_training.run_decompositions()

        # 断言导出的模块的图模块代码与预期结果相匹配
        self.assertExpectedInline(
            str(ep_for_inference.graph_module.code).strip(),
            """\
# 定义一个名为 forward 的方法，用于模型的前向传播
def forward(self, p_linear_weight, p_linear_bias, b_buffer, x):
    # 使用 torch 的原生操作 aten.add.Tensor 对 b_buffer 加上标量 5，并将结果赋给 add
    add = torch.ops.aten.add.Tensor(b_buffer, 5);  b_buffer = None
    # 使用 torch 的原生操作 aten.t.default 对 p_linear_weight 进行转置操作，并将结果赋给 t
    t = torch.ops.aten.t.default(p_linear_weight);  p_linear_weight = None
    # 使用 torch 的原生操作 aten.addmm.default 对 p_linear_bias, x, t 进行矩阵乘法和加法操作，并将结果赋给 addmm
    addmm = torch.ops.aten.addmm.default(p_linear_bias, x, t);  p_linear_bias = x = t = None
    # 使用 torch 的原生操作 aten.sum.default 对 add 中的所有元素进行求和操作，并将结果赋给 sum_1
    sum_1 = torch.ops.aten.sum.default(add)
    # 使用 torch 的原生操作 aten.add.Tensor 将 addmm 和 sum_1 相加，并将结果赋给 add_1
    add_1 = torch.ops.aten.add.Tensor(addmm, sum_1);  addmm = sum_1 = None
    # 返回元组 (add, add_1) 作为前向传播的输出
    return (add, add_1)
    def test_static_dim_constraints(self):
        # 定义一个内部类 Foo，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 构造方法，初始化一个线性层，输入维度为6，输出维度为4
            def __init__(self):
                super().__init__()
                self.l = torch.nn.Linear(6, 4)

            # 前向传播方法，接受三个输入参数 x, y, z
            def forward(self, x, y, z):
                # 对输入 x 进行线性变换，并加上 y 的第2至最后一个元素
                x0 = self.l(x) + y[1:]
                # 返回处理后的 x0 和 z 的两倍
                return x0, z * 2.0

        # 创建 Foo 类的实例对象 foo
        foo = Foo()
        # 定义输入数据 inputs，分别是大小为 (4, 6), (5, 4), (3, 3) 的三个张量
        inputs = (torch.randn(4, 6), torch.randn(5, 4), torch.randn(3, 3))
        # 定义静态维度对象 dx，其最小值为3，最大值为6
        dx = Dim("dx", min=3, max=6)
        # 计算 dy 作为 dx + 1 的结果
        dy = dx + 1
        # 定义静态维度对象 dz，其最小值为3，最大值为6
        dz = Dim("dz", min=3, max=6)

        # 对以下动态形状进行测试，每个元素均为字典，包含两个键值对
        # 每个键表示一个维度的索引，值是一个动态维度对象
        for dynamic_shapes in [
            ({0: dx, 1: 6}, {0: dy, 1: 4}, {0: dz, 1: 3}),
            ((dx, None), (dy, 4), (dz, 3)),
            ((None, 6), (5, None), (None, None)),
            ((4, 6), {0: None, 1: 4}, {0: None, 1: 3}),
        ]:
            # 导出 foo 模型，并传入 inputs 和动态形状 dynamic_shapes
            ep = export(foo, inputs, dynamic_shapes=dynamic_shapes)
            # 断言 foo(*inputs) 等于 ep.module()(*inputs)
            self.assertEqual(foo(*inputs), ep.module()(*inputs))

        # 检查静态维度约束 - 不应该存在静态维度
        ep = export(foo, inputs, dynamic_shapes=((dx, None), (dy, 4), (dz, 3)))
        # 断言 ep.range_constraints 的长度为3
        self.assertEqual(len(ep.range_constraints), 3)
        # 遍历 ep.range_constraints 中的值，确保每个值的下界小于上界
        for vr in ep.range_constraints.values():
            self.assertTrue(vr.lower < vr.upper)

        # 检查引发的错误
        with self.assertRaisesRegex(
            (
                torch.fx.experimental.symbolic_shapes.ConstraintViolationError,
                torch._dynamo.exc.UserError,
            ),
            "Static shape constraint of 5 does not match input size of 4, for .*",
        ):
            # 导出 foo 模型，并传入 inputs 和动态形状 ((5, None), None, None)
            _ = export(foo, inputs, dynamic_shapes=((5, None), None, None))
        with self.assertRaisesRegex(
            (
                torch.fx.experimental.symbolic_shapes.ConstraintViolationError,
                torch._dynamo.exc.UserError,
            ),
            "Static shape constraint of 9 does not match input size of 6, for .*",
        ):
            # 导出 foo 模型，并传入 inputs 和动态形状 ((dx, 9), (dy, 4), (3, 3))
            _ = export(foo, inputs, dynamic_shapes=((dx, 9), (dy, 4), (3, 3)))

    def test_dim_1_2(self):
        # 定义一个简单的神经网络模型 Foo，只有一个前向传播方法，对输入进行乘法运算
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x * 2

        # 定义动态维度对象 dx，其最小值为1，最大值为2
        dx = Dim("dx", min=1, max=2)
        # 导出 Foo 类的实例对象，传入一个大小为 (2, 2) 的张量，并使用动态形状 {0: dx, 1: None}
        ep = export(Foo(), (torch.randn(2, 2),), dynamic_shapes=({0: dx, 1: None},))
        # 分别使用大小为 (1, 2) 和 (2, 2) 的张量调用 ep.module()
        ep.module()(torch.randn(1, 2))
        ep.module()(torch.randn(2, 2))
        # 使用大小为 (3, 2) 的张量调用 ep.module()，预期会引发 RuntimeError
        with self.assertRaisesRegex(
            RuntimeError, "Expected input at .* to be <= 2, but got 3"
        ):
            ep.module()(torch.randn(3, 2))
        # 获取 range_constraints 中的第一个值 vr
        vr = list(ep.range_constraints.values())[0]
        # 断言 vr 的下界为1
        self.assertEqual(vr.lower, 1)
        # 断言 vr 的上界为2
        self.assertEqual(vr.upper, 2)
    # 定义一个测试方法，用于测试动态维度1到2的情况
    def test_derived_dim_1_2(self):
        # 定义一个继承自torch.nn.Module的类Bar
        class Bar(torch.nn.Module):
            # 定义该类的前向传播方法，接收两个参数x和y
            def forward(self, x, y):
                # 返回x和y的第二个元素开始的部分的和
                return x + y[1:]

        # 创建一个名为dx的维度对象，要求其取值范围在1到2之间
        dx = Dim("dx", min=1, max=2)
        # 导出Bar类的模型，传入两个随机张量作为输入，指定动态形状
        ep = export(
            Bar(),
            (torch.randn(2, 2), torch.randn(3, 2)),
            dynamic_shapes=({0: dx, 1: None}, {0: dx + 1, 1: None}),
        )
        # 对导出的模型进行调用，传入两个随机张量作为输入
        ep.module()(torch.randn(1, 2), torch.randn(2, 2))
        # 获取导出模型中所有范围约束的下限，并进行排序
        range_lower_bounds = sorted(vr.lower for vr in ep.range_constraints.values())
        # 获取导出模型中所有范围约束的上限，并进行排序
        range_upper_bounds = sorted(vr.upper for vr in ep.range_constraints.values())
        # 断言下限范围的排序后结果为[1, 2]
        self.assertEqual(range_lower_bounds, [1, 2])
        # 断言上限范围的排序后结果为[2, 3]
        self.assertEqual(range_upper_bounds, [2, 3])

    # 定义一个测试动态形状构建基础功能的方法
    def test_dynamic_shapes_builder_basic(self):
        # 定义一个继承自torch.nn.Module的类M
        class M(torch.nn.Module):
            # 定义该类的前向传播方法，接收三个参数x、y、z
            def forward(self, x, y, z):
                # 返回x、y的第一个元素、z中键为"k"的值的和
                return x + y[0] + z["k"]

        # 创建M类的一个实例m
        m = M()

        # 创建三个随机张量x、y、z作为输入
        x = torch.randn(4)
        y = [torch.randn(4)]
        z = {"k": torch.randn(4)}
        args = (x, y, z)

        # 创建一个torch.export.ShapesCollection对象shapes_collection
        shapes_collection = torch.export.ShapesCollection()
        # 创建一个名为dim的维度对象，设置其最大值为10
        dim = torch.export.Dim("dim", max=10)
        # 将x、y[0]、z["k"]三个张量与对应的维度对象关联起来
        shapes_collection[x] = (dim,)
        shapes_collection[y[0]] = (dim,)
        shapes_collection[z["k"]] = (dim,)

        # 导出模型m，传入args作为位置参数，指定动态形状为shapes_collection
        ep = export(m, args, dynamic_shapes=shapes_collection)
        # 获取导出模型中第一个符号的值
        sym = next(iter(ep.range_constraints.keys()))
        # 遍历导出模型的图中的所有节点
        for node in ep.graph.nodes:
            # 如果节点的操作为"placeholder"
            if node.op == "placeholder":
                # 断言节点的形状元数据的字符串表示为(f"{sym},")
                self.assertEqual(str(tuple(node.meta["val"].shape)), f"({sym},)")

    # 标记为预期失败的测试方法，用于测试动态形状构建关键字参数的情况
    @testing.expectedFailureTrainingIRToRunDecomp  # T193693183
    def test_dynamic_shapes_builder_kwargs(self):
        # 定义一个继承自torch.nn.Module的类M
        class M(torch.nn.Module):
            # 定义该类的前向传播方法，接收三个参数x、y、z
            def forward(self, x, y, z):
                # 返回x、y的第一个元素、z中键为"k"的值的和
                return x + y[0] + z["k"]

        # 创建M类的一个实例m
        m = M()

        # 创建一个随机张量x作为输入
        x = torch.randn(4)
        # 创建一个包含一个随机张量的列表y作为输入
        y = [torch.randn(4)]
        # 创建一个包含键为"k"的随机张量的字典z作为输入
        z = {"k": torch.randn(4)}
        # 将x作为位置参数args的一部分
        args = (x,)
        # 将y和z作为关键字参数kwargs的一部分
        kwargs = {"z": z, "y": y}

        # 创建一个torch.export.ShapesCollection对象shapes_collection
        shapes_collection = torch.export.ShapesCollection()
        # 创建一个名为dim的维度对象，设置其最大值为10
        dim = torch.export.Dim("dim", max=10)
        # 将x、y[0]、z["k"]三个张量与对应的维度对象关联起来
        shapes_collection[x] = (dim,)
        shapes_collection[y[0]] = (dim,)
        shapes_collection[z["k"]] = (dim,)

        # 导出模型m，传入args作为位置参数，kwargs作为关键字参数，指定动态形状为shapes_collection
        ep = export(m, args, kwargs=kwargs, dynamic_shapes=shapes_collection)
        # 获取导出模型中第一个符号的值
        sym = next(iter(ep.range_constraints.keys()))
        # 遍历导出模型的图中的所有节点
        for node in ep.graph.nodes:
            # 如果节点的操作为"placeholder"
            if node.op == "placeholder":
                # 断言节点的形状元数据的字符串表示为(f"{sym},")
                self.assertEqual(str(tuple(node.meta["val"].shape)), f"({sym},)")

    # 标记为预期失败的测试方法，用于测试回溯性
    # 似乎不喜欢数据类的注册，在fx_pytree.tree_flatten_spec中引发了一个动态错误
    @testing.expectedFailureRetraceability
    def test_dynamic_shapes_builder_pytree(self):
        # 注册数据类 Inp 到 Torch 的序列化系统，指定其序列化类型名称
        torch.export.register_dataclass(
            Inp,
            serialized_type_name="test_dynamic_shapes_builder_pytree.Inp",
        )

        # 定义一个简单的 Torch 模型类 M
        class M(torch.nn.Module):
            # 定义模型的前向传播函数，接受类型为 Inp 的输入 inp
            def forward(self, inp: Inp):
                # 返回输入中的三个部分的求和结果
                return inp.x + inp.y[0] + inp.z["k"]

        # 创建模型实例 m
        m = M()
        # 生成随机张量 x
        x = torch.randn(4)
        # 生成随机张量列表 y
        y = [torch.randn(4)]
        # 生成带有键 "k" 的随机张量字典 z
        z = {"k": torch.randn(4)}
        # 构造输入参数为 Inp 类型的元组 args
        args = (Inp(x, y, z),)

        # 创建一个空的 torch.export.ShapesCollection 对象 shapes_collection
        shapes_collection = torch.export.ShapesCollection()
        # 创建一个维度为 "dim" 的 torch.export.Dim 对象，最大值为 10
        dim = torch.export.Dim("dim", max=10)
        # 将 x 的形状维度信息加入 shapes_collection
        shapes_collection[x] = (dim,)
        # 将 y[0] 的形状维度信息加入 shapes_collection
        shapes_collection[y[0]] = (dim,)
        # 将 z["k"] 的形状维度信息加入 shapes_collection
        shapes_collection[z["k"]] = (dim,)

        # 导出模型 m，传入参数 args，并使用动态形状集合 dynamic_shapes
        ep = export(m, args, dynamic_shapes=shapes_collection.dynamic_shapes(m, args))
        # 获取范围约束键的迭代器的第一个元素 sym
        sym = next(iter(ep.range_constraints.keys()))
        # 遍历导出过程中图的所有节点
        for node in ep.graph.nodes:
            # 如果节点操作为 "placeholder"
            if node.op == "placeholder":
                # 断言节点的 meta 属性中的 val 张量的形状为 (sym,)
                self.assertEqual(str(tuple(node.meta["val"].shape)), f"({sym},)")

    def test_torch_check_eq_commutativity(self):
        # 定义一个简单的 Torch 模型类 M1
        class M1(torch.nn.Module):
            # 定义模型的前向传播函数，接受四个输入参数 x1, x2, x3, y
            def forward(self, x1, x2, x3, y):
                # 获取 x1, x2, x3 的标量值 z1, z2, z3
                z1 = x1.item()
                z2 = x2.item()
                z3 = x3.item()
                # 使用 torch._check 检查 z1 是否等于 z2 + z3
                torch._check(z1 == (z2 + z3))
                # 如果 z2 + z3 等于 z1，则返回 y 的两倍
                if z2 + z3 == z1:
                    return y * 2
                # 否则返回 y + 3
                else:
                    return y + 3

        # 导出模型 M1 的实例，传入参数为具体张量值
        export(
            M1(),
            (torch.tensor(6), torch.tensor(3), torch.tensor(3), torch.randn(1)),
        )

        # 定义另一个简单的 Torch 模型类 M2
        class M2(torch.nn.Module):
            # 定义模型的前向传播函数，接受四个输入参数 x1, x2, x3, y
            def forward(self, x1, x2, x3, y):
                # 获取 x1, x2, x3 的标量值 z1, z2, z3
                z1 = x1.item()
                z2 = x2.item()
                z3 = x3.item()
                # 使用 torch._check 检查 z1 是否不等于 z2 + z3
                torch._check(z1 != (z2 + z3))
                # 如果 z2 + z3 等于 z1，则返回 y 的两倍
                if z2 + z3 == z1:
                    return y * 2
                # 否则返回 y + 3
                else:
                    return y + 3

        # 导出模型 M2 的实例，传入参数为具体张量值
        export(
            M2(),
            (torch.tensor(6), torch.tensor(6), torch.tensor(6), torch.randn(1)),
        )

    def test_raise_user_error_when_guard_on_data_dependent_operation(self):
        # 定义一个 Torch 模型类 M
        class M(torch.nn.Module):
            # 定义模型的前向传播函数，接受一个输入参数 x
            def forward(self, x):
                # 对输入 x 进行非零元素索引操作，返回索引张量 y
                y = x.nonzero()
                # 获取张量 y 的形状的第一个维度值 z
                z = y.shape[0]
                # 如果 z 大于 2，则返回 x 的余弦
                if z > 2:
                    return x.cos()
                # 否则返回 x 的正弦
                else:
                    return x.sin()

        # 使用 self.assertRaisesRegex 断言捕获 UserError 异常或 GuardOnDataDependentSymNode 异常
        with self.assertRaisesRegex(
            (
                torchdynamo.exc.UserError,
                torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode,
            ),
            "Could not guard on data-dependent expression",
        ):
            # 导出模型 M 的实例，传入具体的张量作为参数
            _ = export(M(), (torch.tensor([2, 3, 5]),))
    def test_if_functional(self):
        # 定义一个继承自torch.nn.Module的简单模块类
        class Module(torch.nn.Module):
            # 模块的前向传播函数
            def forward(self, x):
                # 对输入张量 x 加上常数 4
                z = x + 4
                # 对张量 z 原地加上常数 4
                z.add_(4)
                # 将张量 z 转换成与 x 相同形状的张量 y
                y = z.view(x.shape)
                # 返回 x 的余弦函数值加上 y 的余弦函数值作为输出
                return x.cos() + y.cos()

        # 创建 Module 类的实例 foo
        foo = Module()
        # 导出模型 foo，输入为一个长度为 3 的张量
        gm = export(foo, (torch.tensor([2, 3, 5]),))

        # 初始化视图节点计数器
        view_count = 0
        # 遍历导出模型的计算图中的节点
        for node in gm.graph.nodes:
            # 检查节点操作是否为调用 torch.ops.aten.add_.Tensor 的函数
            if node.op == "call_function" and node.target == torch.ops.aten.add_.Tensor:
                # 断言不再有原地修改操作
                self.assertNotEqual(
                    node.target,
                    torch.ops.aten.add_.Tensor,
                    "There shouldn't be any inplace mutation node in the graph.",
                )
            # 检查节点操作是否为调用 torch.ops.aten.view.default 的函数
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.view.default
            ):
                # 视图节点计数加一
                view_count += 1

        # 断言计算图中至少有一个视图节点
        self.assertTrue(view_count > 0)

    def test_export_mod_constraints(self):
        # 定义一个简单的动态形状模型类
        class BasicDynamiShapeModel(torch.nn.Module):
            # 模型的前向传播函数，输入为 torch.Tensor 类型，输出也为 torch.Tensor 类型
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 对输入张量 x 进行形状变换，将其第一个维度减一，第二个维度自动计算
                return x.view(x.shape[0] - 1, -1)

        # 创建 BasicDynamiShapeModel 类的实例 m
        m = BasicDynamiShapeModel()
        # 创建一个形状为 (3, 4) 的随机张量 a
        a = torch.randn(3, 4)
        # 定义动态形状的约束条件，x 的第一个维度最小为 3，第二个维度最大为 8000
        dim0_x = torch.export.Dim("dim0_x", min=3)
        dim1_x = torch.export.Dim("dim1_x", max=8000)
        dynamic_shapes = {"x": (dim0_x, dim1_x)}

        # 断言在导出模型时会抛出特定异常
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Specializations unexpectedly required"
                ".*\n.*\\[0\\] must be specialized to 3.*guards.*too complex(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "dim0_x = 3(.*\n)*.*"
                "dim1_x = 2\\*_dim1_x"
            ),
        ):
            # 导出模型 m，输入为张量 a，带有动态形状约束
            torch.export.export(m, (a,), dynamic_shapes=dynamic_shapes)

        # 重新定义动态形状的约束条件，取消第一个维度的特定要求，第二个维度的最大值为 4000 的两倍
        dim0_x = None
        dim1_x = 2 * torch.export.Dim("_dim1_x", max=4000)
        dynamic_shapes = {"x": (dim0_x, dim1_x)}

        # 导出模型 m，输入为张量 a，带有更新后的动态形状约束
        em = torch.export.export(m, (a,), dynamic_shapes=dynamic_shapes)

        # 创建一个形状为 (3, 5) 的随机张量 x
        x = torch.randn(3, 5)
        # 断言在导出模型 em 后进行模型推断时会抛出特定异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected.*shape\\[1\\] = 5 to be of the form 2\\*s1, where s1 is an integer",
        ):
            # 对模型 em 进行推断，输入为张量 x
            em.module()(x)
    def test_not_correct_dim(self):
        # 定义一个函数 f，计算输入张量的余弦
        def f(x):
            return x.cos()

        # 定义一个函数 g，对输入张量加上常数 4
        def g(x):
            return x + 4

        # 创建一个包含单个标量值的张量 inp_for_f
        inp_for_f = torch.tensor(5)
        
        # 使用 assertRaisesRegex 断言捕获 UserError 异常，验证 dynamic_dim 函数对于0维张量的处理
        with self.assertRaisesRegex(
            torchdynamo.exc.UserError, "Cannot mark 0-dimension tensors to be dynamic"
        ):
            constraints = [dynamic_dim(inp_for_f, 0)]

        # 创建一个尺寸为 (5, 5) 的全一张量 inp_for_f_mul_dim
        inp_for_f_mul_dim = torch.ones(5, 5)
        
        # 使用 assertRaisesRegex 断言捕获 UserError 异常，验证 dynamic_dim 函数对于超出范围的维度参数的处理
        with self.assertRaisesRegex(
            torchdynamo.exc.UserError,
            "Expected the dimension passed to dynamic_dim to be in the range \\[0:1\\]",
        ):
            constraints = [dynamic_dim(inp_for_f_mul_dim, 2)]

        # 创建一个标量值 inp_for_g
        inp_for_g = 4
        
        # 使用 assertRaisesRegex 断言捕获 UserError 异常，验证 dynamic_dim 函数对于非张量输入的处理
        with self.assertRaisesRegex(
            torchdynamo.exc.UserError, "Expected tensor as input to dynamic_dim"
        ):
            constraints = [dynamic_dim(inp_for_g, 0)]

    @testing.expectedFailureRetraceability  # T183144629
    def test_map(self):
        # 定义一个 Module 类，其中的 forward 方法接受多个张量作为输入，将一个函数 body 应用于每个张量的对应元素
        class Module(torch.nn.Module):
            def forward(self, xs, y, z):
                # 定义内部函数 body，对三个输入张量的对应元素进行加法操作
                def body(x, y, z):
                    return x + y + z

                # 使用 map 函数将 body 应用于 xs、y、z 的对应元素上
                return map(body, xs, y, z)

        # 创建 Module 的实例 list_tensor_map
        list_tensor_map = Module()
        
        # 准备输入参数 inps，包括一个尺寸为 (6, 4) 的全一张量和两个标量
        inps = (torch.ones(6, 4), torch.tensor(5), torch.tensor(4))
        
        # 调用 _test_export_same_as_eager 方法，测试 list_tensor_map 在给定输入下的导出结果与 eager 模式下的结果是否相同
        self._test_export_same_as_eager(list_tensor_map, inps)

    @unittest.expectedFailure
    def test_crop_like(self):
        # https://fb.workplace.com/groups/1405155842844877/posts/8195050017188725/

        # 从 torchvision 中最小化的裁剪代码，来自 https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional
        # 定义一个 CropLike 类，其 forward 方法接受图像、裁剪高度和裁剪宽度作为输入，返回裁剪后的图像
        class CropLike(torch.nn.Module):
            def forward(self, image, crop_height, crop_width):
                c, image_height, image_width = image.shape
                crop_top = int(round((image_height - crop_height) / 2.0))
                crop_left = int(round((image_width - crop_width) / 2.0))
                return image[
                    ...,
                    crop_top : crop_top + crop_height,
                    crop_left : crop_left + crop_width,
                ]

        # 创建 CropLike 类的实例 crop
        crop = CropLike()
        
        # 创建动态维度的字典 dynamic_dims，用于指定图像、裁剪高度和裁剪宽度的动态维度
        imagew = Dim("width")
        imageh = Dim("height")
        dynamic_dims = {
            "image": {0: None, 1: imageh, 2: imagew},
            "crop_height": None,
            "crop_width": None,
        }
        
        # 准备输入参数 args，包括一个尺寸为 (3, 512, 512) 的随机张量和裁剪高度、裁剪宽度
        args = (torch.rand(3, 512, 512), 150, 150)
        
        # 调用 export 函数，导出 crop 模块，并将结果存储在 ecrop 中，同时指定动态形状 dynamic_dims
        ecrop = export(crop, args=args, dynamic_shapes=dynamic_dims)

        # 创建另一组输入参数 args，包括一个尺寸为 (3, 700, 700) 的随机张量和裁剪高度、裁剪宽度
        args = (torch.rand(3, 700, 700), 150, 150)
        
        # 使用断言验证 crop 模块在两组输入下的导出结果是否相同
        self.assertEqual(ecrop.module()(*args), ecrop(*args))

    @testing.expectedFailureTrainingIRToRunDecomp  # T193693183
    def test_export_func_with_kwargs(self):
        # 定义一个 Module 类，其 forward 方法接受多个位置参数和关键字参数，并返回它们的加法结果
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, kw1, kw2):
                return arg1 + arg2, kw1 + kw2

        # 创建 Module 类的实例 kw_func
        kw_func = Module()
        
        # 准备输入参数 args，包括一个尺寸为 (6, 4) 的全一张量和一个尺寸为 (1, 1) 的全一张量
        args = (torch.ones(6, 4), torch.ones(1, 1))
        
        # 准备关键字参数 kwargs，包括两个尺寸相同的张量
        kwargs = {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}
        
        # 调用 _test_export_same_as_eager 方法，测试 kw_func 在给定输入和关键字参数下的导出结果与 eager 模式下的结果是否相同
        self._test_export_same_as_eager(kw_func, args, kwargs)
    @testing.expectedFailureTrainingIRToRunDecomp  # 标记此测试预期为失败，跟踪编号为T193693183
    def test_export_func_with_pytree_kwargs(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, a, b):
                return arg1 + a["kw1"] + b[0], arg2 + a["kw2"] + b[1]

        kw_func = Module()
        args = (torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {
            "a": {"kw1": torch.ones(2, 3), "kw2": torch.ones(3, 4)},
            "b": [torch.ones(2, 3), torch.ones(3, 4)],
        }
        self._test_export_same_as_eager(kw_func, args, kwargs)

    @testing.expectedFailureTrainingIRToRunDecomp  # 标记此测试预期为失败，跟踪编号为T193693183
    def test_export_func_with_default_kwargs(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, a, b=1):
                return arg1 + arg2, a["kw1"] + a["kw2"] + b

        kw_func = Module()

        class Module2(torch.nn.Module):
            def forward(self, arg1, arg2, a=1, b=2):
                return arg1 + a, arg2 + b

        kw_func2 = Module2()

        args = (torch.ones(6, 4), torch.ones(1, 1))
        kwargs1 = {"a": {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}}
        kwargs2 = {"a": {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}, "b": 2}
        self._test_export_same_as_eager(kw_func, args, kwargs1)
        self._test_export_same_as_eager(kw_func, args, kwargs2)
        kwargs3 = {"b": 1}
        self._test_export_same_as_eager(kw_func2, args, kwargs3)

    def test_export_func_with_var_postional_args(self):
        # 定义一个带有可变位置参数的模块
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, *args):
                return arg1 + args[0], arg2 + args[1]

        kw_func = Module()
        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        self._test_export_same_as_eager(kw_func, args)

    @testing.expectedFailureTrainingIRToRunDecomp  # 标记此测试预期为失败，跟踪编号为T193693183
    def test_export_func_with_keyword_only_args(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, *args, kw1, kw2):
                return arg1 + args[0] + kw1, arg2 + args[1] + kw2

        kw_func = Module()
        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {"kw1": torch.ones(2, 3), "kw2": torch.ones(3, 4)}
        self._test_export_same_as_eager(kw_func, args, kwargs)

    @testing.expectedFailureTrainingIRToRunDecomp  # 标记此测试预期为失败，跟踪编号为T193693183
    # 定义一个测试方法，用于测试带有可变关键字参数的导出函数
    def test_export_func_with_var_keyword_args(self):
        # 定义一个继承自torch.nn.Module的内部类Module
        class Module(torch.nn.Module):
            # 实现Module类的前向传播方法
            def forward(self, arg1, arg2, *args, kw1, kw2, **kwargs):
                # 返回一个元组，包含四个值的计算结果
                return (
                    arg1 + args[0] + kw1 + kwargs["kw3"],  # 计算第一个值
                    arg2 + args[1] + kw2 + kwargs["kw4"],  # 计算第二个值
                )

        # 创建Module类的实例kw_func
        kw_func = Module()
        # 定义一个元组args，包含四个torch.ones张量
        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        # 定义一个字典kwargs，包含四个键值对，键为字符串，值为torch.ones张量
        kwargs = {
            "kw1": torch.ones(2, 3),
            "kw2": torch.ones(3, 4),
            "kw3": torch.ones(2, 3),
            "kw4": torch.ones(3, 4),
        }
        # 调用self._test_export_same_as_eager方法，测试kw_func的导出结果与eager执行的结果是否相同

    # 定义一个测试方法，用于测试未支持的切片操作
    def test_unbacked_slice(self):
        # 定义一个继承自torch.nn.Module的内部类M
        class M(torch.nn.Module):
            # 实现M类的前向传播方法
            def forward(self, scores, score_thr, topk: torch.Tensor, results=None):
                # 创建一个有效掩码，标记scores中大于score_thr的位置
                valid_mask = scores > score_thr
                # 根据有效掩码，筛选出有效的scores值
                scores = scores[valid_mask]
                # 获取有效掩码的非零索引，并将其移动到与scores相同的设备上
                valid_idxs = torch.nonzero(valid_mask).to(scores.device)

                # 计算num_topk，即有效掩码的非零索引数与topk中的最小值
                num_topk = torch.minimum(topk, torch.tensor(valid_idxs.shape[0])).item()
                # 检查num_topk的大小
                torch._check_is_size(num_topk)
                # 检查scores的长度是否大于等于num_topk
                torch._check(scores.shape[0] >= num_topk)
                # 对scores进行降序排序，并返回排序后的scores及其对应的索引
                scores, idxs = scores.sort(descending=True)
                # 从排序后的scores中取前num_topk个值
                scores = scores[:num_topk]
                # 根据idxs获取有效索引的topk索引值
                topk_idxs = valid_idxs[idxs[:num_topk]]
                # 解绑topk_idxs的维度1，得到保留索引和标签
                keep_idxs, labels = topk_idxs.unbind(dim=1)

                # 返回scores、labels和keep_idxs作为前向传播方法的输出结果

        # 定义一个torch.tensor对象score，包含一个4x3的浮点数值张量
        score = torch.tensor(
            [[0.1, 0.3, 0.2], [0.12, 0.7, 0.9], [0.02, 0.8, 0.08], [0.4, 0.1, 0.08]]
        )
        # 定义一个torch.tensor对象bbox_pred，包含一个4x2的浮点数值张量
        bbox_pred = torch.tensor([[0.2, 0.3], [0.4, 0.7], [0.1, 0.1], [0.5, 0.1]])
        # 定义一个浮点数值score_thr，其值为0.15
        score_thr = 0.15
        # 定义一个torch.tensor对象nms_pre，包含一个整数4
        nms_pre = torch.tensor(4)
        # 定义一个元组inputs，包含四个张量和一个字典
        inputs = (score, score_thr, nms_pre, dict(bbox_pred=bbox_pred))

        # 调用torch.export.export方法，导出M类的模型，并将inputs作为输入
        ep = torch.export.export(M(), inputs)
        # 创建M类的实例orig_res，并使用inputs调用其前向传播方法
        orig_res = M()(*inputs)
        # 创建ep的模块实例ep_res，并使用inputs调用其前向传播方法
        ep_res = ep.module()(*inputs)
        # 使用assertTrue方法断言orig_res的第一个元素与ep_res的第一个元素在所有元素上是否接近
        self.assertTrue(torch.allclose(orig_res[0], ep_res[0]))
        # 使用assertTrue方法断言orig_res的第二个元素与ep_res的第二个元素在所有元素上是否接近
        self.assertTrue(torch.allclose(orig_res[1], ep_res[1]))
        # 使用assertTrue方法断言orig_res的第三个元素与ep_res的第三个元素在所有元素上是否接近
        self.assertTrue(torch.allclose(orig_res[2], ep_res[2]))
    def test_unflatten_asserts(self):
        # TODO: strict-export fails
        # 定义一个简单的神经网络模型 M1，其 forward 方法接受两个参数 x 和 y
        class M1(torch.nn.Module):
            def forward(self, x, y):
                # 从张量 x 中提取出一个标量 b
                b = x.item()

                # 检查 b 是否符合张量大小的要求
                torch._check_is_size(b)
                # 检查 b 是否小于 y 的第一个维度大小
                torch._check(b < y.size(0))
                # 返回 y 的前 b 个元素
                return y[:b]

        # 定义另一个神经网络模型 M3，其 forward 方法也接受 x 和 y 作为参数
        class M3(torch.nn.Module):
            def forward(self, x, y):
                # 从张量 x 中提取出一个标量 b
                b = x.item()

                # 检查 b 是否符合张量大小的要求
                torch._check_is_size(b)
                # 检查 b 是否小于 y 的第一个维度大小的两倍
                torch._check(b < y.size(0) * 2)
                # 返回 y 的前 b 个元素
                return y[:b]

        # 定义一个组合模型 M2，继承自 torch.nn.Module
        class M2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建 M1 和 M3 的实例
                self.m1 = M1()
                self.m3 = M3()

            def forward(self, x, y):
                # 调用 M1 和 M3 的 forward 方法，并将结果相加返回
                return self.m1(x, y) + self.m3(x, y)

        # 构造输入数据 inputs
        inputs = (torch.tensor(3), torch.randn(10))

        # 使用 torch.export.export 方法导出模型 M2
        ep = torch.export.export(
            M2(), inputs, dynamic_shapes={"x": None, "y": (Dim("moo"),)}, strict=False
        )
        # 构造原始模型的输出结果 orig_res
        orig_res = M2()(*inputs)
        # 调用导出模型的结果 ep_res
        ep_res = ep.module()(*inputs)
        # 断言原始模型和导出模型的第一个输出结果在数值上相近
        self.assertTrue(torch.allclose(orig_res[0], ep_res[0]))
        # 断言原始模型和导出模型的第二个输出结果在数值上相近
        self.assertTrue(torch.allclose(orig_res[1], ep_res[1]))
        # 断言原始模型和导出模型的第三个输出结果在数值上相近
        self.assertTrue(torch.allclose(orig_res[2], ep_res[2]))

        # 对导出结果进行解扁平化操作
        unflattened = torch.export.unflatten(ep)
        # 调用解扁平化后的结果 ep_res
        ep_res = unflattened(*inputs)
        # 断言原始模型和解扁平化后的输出结果在数值上相近
        self.assertTrue(torch.allclose(orig_res[0], ep_res[0]))
        # 断言原始模型和解扁平化后的输出结果在数值上相近
        self.assertTrue(torch.allclose(orig_res[1], ep_res[1]))
        # 断言原始模型和解扁平化后的输出结果在数值上相近
        self.assertTrue(torch.allclose(orig_res[2], ep_res[2]))

    @testing.expectedFailureTrainingIRToRunDecomp  # T193693183
    # 定义一个测试函数，用于测试带有可变关键字参数的导出功能
    def test_export_func_with_var_keyword_pytree_args(self):
        # 定义一个简单的神经网络模型 Module，其 forward 方法接受多个参数
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, *args, kw1, kw2, **kwargs):
                # 返回一系列参数的组合结果
                return (
                    arg1 + arg2[0][0] + args[0] + kw1[0] + kwargs["kw3"][0],
                    arg2[1] + args[1] + kw2 + kwargs["kw4"],
                )

        # 创建 Module 类的实例 kw_func
        kw_func = Module()
        # 定义输入参数 args 和关键字参数 kwargs
        args = (
            torch.ones(2, 3),
            [(torch.ones(2, 3),), torch.ones(3, 4)],
            torch.ones(2, 3),
            torch.ones(3, 4),
        )
        kwargs = {
            "kw1": (torch.ones(2, 3),),
            "kw2": torch.ones(3, 4),
            "kw3": (torch.ones(2, 3), torch.ones(3, 4)),
            "kw4": torch.ones(3, 4),
        }
        # 调用 _test_export_same_as_eager 方法进行测试
        self._test_export_same_as_eager(kw_func, args, kwargs)

    @testing.expectedFailureSerDer  # we don't save placeholder metadata
    @testing.expectedFailureNonStrict
    @testing.expectedFailureTrainingIRToRunDecomp  # T193692674
    # 定义一个测试方法，用于测试线性卷积
    def test_linear_conv(self):
        # 定义一个简单的线性模型类 MyLinear
        class MyLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化权重矩阵为 20x98 的随机张量
                self.weight = torch.randn(20, 98)
                # 初始化偏置向量为长度为 20 的随机张量
                self.bias = torch.randn(20)

            # 前向传播函数，使用 torch.nn.functional.linear 进行线性变换
            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        # 定义一个主模型类 Foo
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个 2D 卷积层，输入通道数为 16，输出通道数为 33，卷积核大小为 3x3
                self.conv = torch.nn.Conv2d(16, 33, 3)
                # 初始化一个 MyLinear 类的实例
                self.linear = MyLinear()

            # 主模型的前向传播函数
            def forward(self, x):
                # 经过卷积层处理后的特征
                x_conv = self.conv(x)
                # 经过线性层处理后的特征
                x_linear = self.linear(x_conv)
                # 返回线性变换结果的余弦值
                return x_linear.cos()

        # 导出模型 Foo 的图形表示
        ep = export(Foo(), (torch.randn(20, 16, 50, 100),))
        
        # 遍历导出图中的每个节点
        for node in ep.graph.nodes:
            # 如果节点的操作类型为占位符，并且节点名称存在于输入到缓冲区或参数的映射中
            if (
                node.op == "placeholder"
                and node.name in ep.graph_signature.inputs_to_buffers
                or node.name in ep.graph_signature.inputs_to_parameters
            ):
                # 断言节点的元数据中包含 "source_fn_stack" 字段
                self.assertTrue("source_fn_stack" in node.meta)

    # 定义一个测试方法，用于测试建议的修复方案中的新根维度
    def test_suggested_fixes_new_roots(self):
        # 导入 torch.export 模块中的 dims 函数
        from torch.export import dims

        # 定义一个简单的模型类 Foo
        class Foo(torch.nn.Module):
            # 模型的前向传播函数，接收 x, y, z 三个参数
            def forward(self, x, y, z):
                # 根据建议的修复方案，为模数保护引入新的根维度
                # 当 x 的第一个维度在 5 到 36 之间，并且 y 的第一个维度是 3 的倍数时
                if x.shape[0] >= 5 and x.shape[0] <= 36 and y.shape[0] % 3 == 0:
                    # 返回 x, y 和 z 的组合结果
                    return x + y[1:] + z[3:]

        # 创建 Foo 类的实例 foo
        foo = Foo()
        
        # 准备输入数据元组，分别是三个张量，形状分别为 (11,), (12,), (14,)
        inputs = (
            torch.randn(
                11,
            ),
            torch.randn(
                12,
            ),
            torch.randn(
                14,
            ),
        )
        
        # 定义三个维度对象 dx, dy, dz，并调用 dims 函数创建
        dx, dy, dz = dims("dx", "dy", "dz")
        
        # 动态形状字典，指定每个维度的形状
        dynamic_shapes = {
            "x": (dx,),
            "y": (dy,),
            "z": (dz,),
        }
        
        # 使用断言验证导出时抛出的特定异常
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated.*!(.*\n)*.*"
                "Suggested fixes(.*\n)*.*"
                "_dx = Dim\(\\'_dx\\', max=12\)(.*\n)*.*"
                "dx = 3\*_dx - 1(.*\n)*.*"
                "dy = 3\*_dx(.*\n)*.*"
                "dz = 3\*_dx \+ 2"
            ),
        ):
            # 导出模型 Foo，并传入输入数据和动态形状
            export(Foo(), inputs, dynamic_shapes=dynamic_shapes)
        
        # 重新尝试导出模型 Foo，使用新的动态形状
        _dx = Dim("_dx", min=2, max=12)
        dynamic_shapes = {"x": (3 * _dx - 1,), "y": (3 * _dx,), "z": (3 * _dx + 2,)}
        export(Foo(), inputs, dynamic_shapes=dynamic_shapes)
    def test_dynamic_shapes_spec_with_pytree(self):
        # 导入所需的模块和函数
        from torch.export import Dim, export
        from torch.utils._pytree import tree_map

        # 创建包含不同数据结构的输入示例
        inputs = {
            "tensor": torch.randn(3),
            "dict_of_tensors": {k: torch.randn(3) for k in ["A", "B", "C", "D"]},
            "list_of_tensors": [torch.randn(3) for _ in range(4)],
        }

        # 定义一个维度对象表示批处理维度
        batch = Dim("batch")
        # 使用 tree_map 将所有输入统一指定为动态形状
        spec = tree_map(lambda x: {0: batch}, inputs)

        # 定义一个简单的神经网络模块
        class Foo(torch.nn.Module):
            def forward(self, inputs):
                return (
                    inputs["tensor"]
                    + inputs["dict_of_tensors"]["A"]
                    + inputs["list_of_tensors"][0]
                )

        # 导出模块并指定动态形状的输入规范
        ep = export(Foo(), (inputs,), dynamic_shapes={"inputs": spec})
        
        # 提取所有占位符节点的形状信息并转换为字符串列表
        input_shapes = [
            str(node.meta["val"].shape)
            for node in ep.graph_module.graph.nodes
            if node.op == "placeholder"
        ]

        # 断言所有占位符节点的形状应为 torch.Size([s0])
        self.assertEqual(len(input_shapes), 9)
        self.assertTrue(all(shape == "torch.Size([s0])" for shape in input_shapes))

    def test_error_does_not_reference_eager_fallback(self):
        # 定义一个简单的神经网络模块
        class Module(torch.nn.Module):
            def forward(self, x):
                y = x.nonzero()  # 获取输入张量 x 的非零索引
                z = y.shape[0]  # 计算非零索引的数量
                if z > 2:
                    return x.cos()  # 如果非零索引数量大于2，则返回 x 的余弦值
                else:
                    return x.sin()  # 否则返回 x 的正弦值

        fn_ddo = Module()
        # 根据测试方法名称判断是否是非严格测试，选择不同的错误类型和错误信息
        if is_non_strict_test(self._testMethodName):
            error = torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode
            error_msg = r"Could not guard on data-dependent expression"
        else:
            error = torchdynamo.exc.UserError
            error_msg = r"^(?!.*fall back to eager).*"
        
        # 使用断言检查导出过程中是否抛出了特定的错误类型和错误信息
        with self.assertRaisesRegex(error, error_msg):
            _ = export(fn_ddo, (torch.tensor([2, 3, 5]),))
    # 定义一个测试方法，用于测试注册数据类作为 pytree 节点的功能
    def test_pytree_register_data_class(self):
        # 定义一个简单的数据类 MyDataClass，包含 x, y 两个整型字段和可选的 z 字段
        @dataclass
        class MyDataClass:
            x: int
            y: int
            z: int = None

        # 创建 MyDataClass 的实例 dt，设置 x=3, y=4
        dt = MyDataClass(x=3, y=4)
        # 对数据类实例进行扁平化操作，得到扁平化后的数据 flat 和结构描述 spec
        flat, spec = tree_flatten(dt)
        # 断言 spec 的类型为 LeafSpec
        self.assertTrue(spec, LeafSpec())
        # 断言 flat 的长度为 1
        self.assertTrue(len(flat) == 1)

        # 注册 MyDataClass 作为 pytree 节点，设置序列化类型名称为 "test_pytree_register_data_class.MyDataClass"
        register_dataclass_as_pytree_node(
            MyDataClass,
            serialized_type_name="test_pytree_register_data_class.MyDataClass",
        )

        # 再次对数据类实例进行扁平化操作，得到新的 flat 和 spec
        flat, spec = tree_flatten(dt)
        # 断言 spec 符合预期的 TreeSpec 结构
        self.assertEqual(
            spec,
            TreeSpec(MyDataClass, [["x", "y"], ["z"]], [LeafSpec(), LeafSpec()]),
        )
        # 断言 flat 中的数据为 [3, 4]
        self.assertEqual(flat, [3, 4])

        # 使用 tree_unflatten 函数将 flat 和 spec 转换回数据类实例 orig_dt
        orig_dt = tree_unflatten(flat, spec)
        # 断言 orig_dt 是 MyDataClass 类型的实例
        self.assertTrue(isinstance(orig_dt, MyDataClass))
        # 验证 orig_dt 的 x, y, z 属性值分别为 3, 4, None
        self.assertEqual(orig_dt.x, 3)
        self.assertEqual(orig_dt.y, 4)
        self.assertEqual(orig_dt.z, None)

        # 将 spec 序列化为字符串，再反序列化得到 roundtrip_spec
        roundtrip_spec = treespec_loads(treespec_dumps(spec))
        # 断言 roundtrip_spec 与原始 spec 相等
        self.assertEqual(roundtrip_spec, spec)

        # 定义另一个数据类 MyOtherDataClass，具有与 MyDataClass 类似的字段
        @dataclass
        class MyOtherDataClass:  # pytree 注册不允许对同一类注册两次
            x: int
            y: int
            z: int = None

        # 使用注册函数注册 MyOtherDataClass，保留空字段，设置序列化类型名称
        register_dataclass_as_pytree_node(
            MyOtherDataClass,
            return_none_fields=True,
            serialized_type_name="test_pytree_regster_data_class.MyOtherDataClass",
        )

        # 创建 MyOtherDataClass 的实例 dt，设置 x=3, y=4
        dt = MyOtherDataClass(x=3, y=4)
        # 对 MyOtherDataClass 实例进行扁平化操作，得到 flat 和 spec
        flat, spec = tree_flatten(dt)
        # 断言 spec 符合预期的 TreeSpec 结构
        self.assertEqual(
            spec,
            TreeSpec(
                MyOtherDataClass,
                [["x", "y", "z"], []],
                [LeafSpec(), LeafSpec(), LeafSpec()],
            ),
        )
        # 断言 flat 中的数据为 [3, 4, None]
        self.assertEqual(flat, [3, 4, None])

        # 使用 tree_unflatten 函数将 flat 和 spec 转换回数据类实例 orig_dt
        orig_dt = tree_unflatten(flat, spec)
        # 断言 orig_dt 是 MyOtherDataClass 类型的实例
        self.assertTrue(isinstance(orig_dt, MyOtherDataClass))
        # 验证 orig_dt 的 x, y, z 属性值分别为 3, 4, None
        self.assertEqual(orig_dt.x, 3)
        self.assertEqual(orig_dt.y, 4)
        self.assertEqual(orig_dt.z, None)

        # 将 spec 序列化为字符串，再反序列化得到 roundtrip_spec
        roundtrip_spec = treespec_loads(treespec_dumps(spec))
        # 断言 roundtrip_spec 与原始 spec 相等
        self.assertEqual(roundtrip_spec, spec)
    def test_pytree_register_nested_data_class(self):
        @dataclass
        class Inner:
            x: int
            y: int

        @dataclass
        class Outer:
            xy: Inner
            ab: Inner

        xy = Inner(1, 2)  # 创建一个Inner类的实例xy，x=1, y=2
        ab = Inner(3, 4)  # 创建一个Inner类的实例ab，x=3, y=4
        dt = Outer(xy, ab)  # 创建一个Outer类的实例dt，包含xy和ab
        inp = {"dt1": (dt, ({},)), "dt2": ((torch.ones(1),), dt)}  # 构造输入字典inp

        # 将Inner类注册为PyTree节点，用于树形数据结构操作
        register_dataclass_as_pytree_node(
            Inner, serialized_type_name="test_pytree_register_nested_data_class.Inner"
        )
        # 将Outer类注册为PyTree节点，用于树形数据结构操作
        register_dataclass_as_pytree_node(
            Outer, serialized_type_name="test_pytree_register_nested_data_class.Outer"
        )

        flat, spec = tree_flatten(inp)  # 将inp字典扁平化为列表flat，并生成其结构spec
        self.assertEqual(flat, [1, 2, 3, 4, torch.ones(1), 1, 2, 3, 4])  # 断言flat是否符合预期结果

        unflat = tree_unflatten(flat, spec)  # 根据flat和spec恢复出原始字典unflat
        self.assertEqual(unflat, inp)  # 断言unflat是否与原始inp相等

        roundtrip_spec = treespec_loads(treespec_dumps(spec))  # 将spec序列化后再反序列化得到roundtrip_spec
        self.assertEqual(roundtrip_spec, spec)  # 断言roundtrip_spec与原始spec相等

    def test_param_util(self):
        class Basic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(10, 1)  # 创建一个线性层，输入大小10，输出大小1

            def forward(self, x):
                return self.lin(x)  # 前向传播函数，返回线性层的输出结果

        ep = export(Basic(), (torch.randn(5, 10),))  # 导出Basic模型的图结构ep，输入为5x10的随机张量
        num_params = 0  # 记录参数数量的变量
        params = []  # 存储参数的列表

        for node in ep.graph.nodes:  # 遍历导出图的所有节点
            if is_param(ep, node):  # 判断节点是否为模型参数
                num_params += 1  # 参数数量加一
                params.append(get_param(ep, node))  # 获取并存储参数值

        self.assertEqual(num_params, 2)  # 断言参数数量是否为2
        self.assertEqual(params[0].shape, [1, 10])  # 断言第一个参数的形状为[1, 10]（权重）
        self.assertEqual(params[1].shape, [1])  # 断言第二个参数的形状为[1]（偏置）

    @testing.expectedFailureTrainingIRToRunDecomp  # 标注为预期的测试失败，对应特定的任务号 T193700631
    def test_buffer_util(self):
        ep = export(
            torch.nn.BatchNorm2d(100, affine=False), (torch.ones(20, 100, 35, 45),)
        )  # 导出具有指定参数的BatchNorm2d模型的图结构ep

        num_buffer = 0  # 记录缓冲区数量的变量
        buffer = []  # 存储缓冲区的列表

        for node in ep.graph.nodes:  # 遍历导出图的所有节点
            if is_buffer(ep, node):  # 判断节点是否为模型缓冲区
                num_buffer += 1  # 缓冲区数量加一
                buffer.append(get_buffer(ep, node))  # 获取并存储缓冲区值

        self.assertEqual(num_buffer, 3)  # 断言缓冲区数量是否为3

        self.assertEqual(buffer[0].shape, torch.Size([100]))  # 断言第一个缓冲区的形状为[100]（running_mean）
        self.assertEqual(buffer[1].shape, torch.Size([100]))  # 断言第二个缓冲区的形状为[100]（running_var）
        self.assertEqual(buffer[2].shape, torch.Size([]))  # 断言第三个缓冲区的形状为空（num_batches_tracked）

    @testing.expectedFailureTrainingIRToRunDecomp  # 标注为预期的测试失败，对应特定的任务号 T193701564
    @testing.expectedFailureTrainingIRToRunDecomp  # T193700396
    def test_export_dynamo_config(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size=4, hidden_size=5, num_layers=1)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.lstm(inputs)

        config = DEFAULT_EXPORT_DYNAMO_CONFIG
        mod = MyModule()

        @contextmanager
        def _patch_config(kwargs):
            orig_config_dict = dataclasses.asdict(config)

            try:
                for k, v in kwargs.items():
                    setattr(config, k, v)
                yield
            finally:
                for k, v in orig_config_dict.items():
                    setattr(config, k, v)

        inp = (torch.rand(5, 4),)
        exported_program = export(mod, inp, strict=True)

        # 使用 _patch_config 上下文管理器来修改配置，并在测试期间恢复原始配置
        with _patch_config({"allow_rnn": False}):
            # 测试在禁用 RNN 时导出是否引发预期的异常
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "TorchDynamo purposely graph breaks on RNN, GRU, LSTMs",
            ):
                _ = export(mod, inp, strict=True)

    @testing.expectedFailureTrainingIRToRunDecomp  # T193700396
    def test_device_to_static(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return x.to("cpu")

        # 导出模块，并检查在图形中调用的操作
        ep = export(Module(), (torch.tensor(1, device="cpu"),))
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        # 断言至少存在一个操作，并检查其是否为预期操作
        self.assertGreater(len(ops), 0)
        for op in ops:
            self.assertIn(op, (torch.ops.aten._to_copy.default,))

    @testing.expectedFailureTrainingIRToRunDecomp  # T193700396
    def test_device_to_dynamic(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return x.to("cpu")

        # 导出动态形状的模块，并检查在图形中调用的操作
        ep = export(
            Module(),
            (torch.tensor([1, 2], device="cpu"),),
            dynamic_shapes={"x": {0: Dim("i")}},
        )
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        # 断言至少存在一个操作，并检查其是否为预期操作
        self.assertGreater(len(ops), 0)
        for op in ops:
            self.assertIn(op, (torch.ops.aten._to_copy.default,))

    @testing.expectedFailureTrainingIRToRunDecomp  # T193700396
    def test_device_to_mutation(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                y = x.to("cpu")
                y.add_(1)
                return y, x

        # 测试当尝试突变张量时是否引发预期的运行时异常
        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            export(Module(), (torch.tensor(1, device="cpu"),))

    @testing.expectedFailureTrainingIRToRunDecomp  # T193700396
    @testing.expectedFailureTrainingIRToRunDecomp  # 标记这个测试预期会失败，用于训练IR转运行分解，对应任务ID T193701564
    def test_float_conversion(self):
        # 定义一个简单的模块类，用于将输入张量转换为浮点型
        class Module(torch.nn.Module):
            def forward(self, x):
                return x.float()

        # 导出模块，并提供一个例子输入来生成导出结果
        ep = export(Module(), (torch.tensor(1, dtype=torch.float),))
        
        # 初始化一个空列表用于收集所有的操作
        ops = []
        # 遍历导出结果的计算图的节点
        for node in ep.graph.nodes:
            # 如果节点的操作是 "call_function"，则将目标操作添加到 ops 列表中
            if node.op == "call_function":
                ops.append(node.target)
        
        # 断言 ops 列表中至少有一个操作
        self.assertGreater(len(ops), 0)
        
        # 对 ops 列表中的每个操作，断言其是否在预期的操作集合中
        for op in ops:
            self.assertIn(op, (torch.ops.aten._to_copy.default,))

    @testing.expectedFailureTrainingIRToRunDecomp  # 标记这个测试预期会失败，用于训练IR转运行分解，对应任务ID T193700396
    def test_device_to_mutation_float(self):
        # 定义一个包含浮点数转换和原地加法操作的模块类
        class Module(torch.nn.Module):
            def forward(self, x):
                y = x.float()
                y.add_(1)
                return y, x

        # 使用断言检测在导出时是否捕获到特定的运行时错误
        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            export(Module(), (torch.tensor(1, dtype=torch.float),))

    @testing.expectedFailureTrainingIRToRunDecomp  # 标记这个测试预期会失败，用于训练IR转运行分解，对应任务ID T193692674
    def test_module(self):
        # 定义一个自定义的线性层类 MyLinear
        class MyLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化权重和偏置张量
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                # 使用函数形式的线性函数计算
                return torch.nn.functional.linear(x, self.weight, self.bias)

        # 定义一个包含卷积和自定义线性层的复合模块类 Foo
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个卷积层和一个 MyLinear 类型的实例
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x):
                # 分别处理输入元组的两个张量 a 和 b
                a, b = x
                # 对张量 a 进行卷积操作
                a_conv = self.conv(a)
                # 使用自定义线性层处理卷积结果
                a_linear = self.linear(a_conv)
                # 对张量 b 进行卷积操作
                b_conv = self.conv(b)
                # 使用相同的自定义线性层处理卷积结果
                b_linear = self.linear(b_conv)
                # 返回两个元素的元组，分别是 a_linear 的余弦和 b_linear 的正弦，以及 a_linear 的正弦和 b_linear 的余弦
                return (
                    a_linear.cos() + b_linear.sin(),
                    a_linear.sin() + b_linear.cos(),
                )

        # 准备输入数据的容器，包含两个相同形状的张量元组
        inp_container = ((torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),)

        # 导出模块 Foo 并进行输入数据容器 inp_container 的导出
        ep = export(Foo(), inp_container)
        # 再次导出经过 Foo 模块处理的导出结果
        ep_rexported = export(ep.module(), inp_container)

        # 准备用于测试的输入数据的容器，与 inp_container 相同
        inp_test = ((torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),)

        # 使用断言验证两个导出结果的模块在给定相同输入时，产生的输出的近似性
        self.assertTrue(
            torch.allclose(
                ep.module()(*inp_test)[0], ep_rexported.module()(*inp_test)[0]
            )
        )
        self.assertTrue(
            torch.allclose(
                ep.module()(*inp_test)[1], ep_rexported.module()(*inp_test)[1]
            )
        )
    def test_decomp_batch_norm_functional_predispatch(self):
        # 定义一个包含卷积和批归一化层的模型类
        class ConvBatchnorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个卷积层，输入通道为1，输出通道为3，卷积核大小为1x1，步长为1
                self.conv = torch.nn.Conv2d(1, 3, 1, 1)
                # 初始化一个批归一化层，输入通道为3
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                # 对输入进行卷积操作
                x = self.conv(x)
                # 对卷积输出进行批归一化操作
                x = self.bn(x)
                # 返回一个包含批归一化输出的元组
                return (x,)

        # 创建ConvBatchnorm类的实例
        mod = ConvBatchnorm()
        # 将模型设置为评估模式
        mod.eval()
        # 创建一个形状为(1, 1, 3, 3)的随机张量作为输入
        inp = torch.randn(1, 1, 3, 3)

        # 调用torch.export._trace._export方法对模型进行导出，预调度设置为True
        gm = torch.export._trace._export(mod, (inp,), pre_dispatch=True).module()
        # 使用self.assertExpectedInline断言验证导出代码的预期输出
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, x):
    # 使用 fx_pytree 库的 tree_flatten_spec 方法将输入 x 扁平化为列表形式，并符合输入规范
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    # 获取卷积层的权重和偏置
    conv_weight = self.conv.weight
    conv_bias = self.conv.bias
    # 获取批标准化层的权重、偏置、运行时均值和方差
    bn_weight = self.bn.weight
    bn_bias = self.bn.bias
    bn_running_mean = self.bn.running_mean
    bn_running_var = self.bn.running_var
    # 调用 torch 的 conv2d 操作进行卷积运算，清空相关变量以释放内存
    conv2d = torch.ops.aten.conv2d.default(x, conv_weight, conv_bias);  x = conv_weight = conv_bias = None
    # 调用 torch 的 _native_batch_norm_legit_no_training 操作执行批标准化，清空相关变量
    _native_batch_norm_legit_no_training = torch.ops.aten._native_batch_norm_legit_no_training.default(conv2d, bn_weight, bn_bias, bn_running_mean, bn_running_var, 0.1, 1e-05);  conv2d = bn_weight = bn_bias = bn_running_mean = bn_running_var = None
    # 获取 _native_batch_norm_legit_no_training 结果的第一个元素
    getitem = _native_batch_norm_legit_no_training[0];  _native_batch_norm_legit_no_training = None
    # 使用 pytree 库的 tree_unflatten 方法，根据输出规范将结果反扁平化为树形结构并返回
    return pytree.tree_unflatten((getitem,), self._out_spec)
    def test_constrain_size_with_constrain_value(self):
        # 定义一个测试用的神经网络模块
        class Module(torch.nn.Module):
            def forward(self, x, y):
                # 计算输入张量 x 的最大值
                n = x.max().item()
                # 检查 n 是否大于等于 2
                torch._check(n >= 2)
                # 检查 n 是否小于等于 10
                torch._check(n <= 10)
                # 检查 n 是否是有效的大小（正整数）
                torch._check_is_size(n)
                # 返回 y 加上 n 的结果
                return y + n

        # 创建 Module 的实例
        fn = Module()
        
        # 测试当条件不满足时是否抛出 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError, r"Expected cond to be True, but got False"
        ):
            _ = fn(torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3)))

        # 导出 Module 并测试导出的模型
        ep = export(
            fn,
            (torch.randint(3, 4, (2, 2)), torch.randint(3, 5, (2, 3))),
        )
        
        # 测试当条件不满足时是否抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Invalid value range for 1 between"):
            test_inp = (torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3)))
            _ = ep.module()(*test_inp)

    def test_automatic_constrain_size(self):
        # 定义一个测试用的神经网络模块
        class M(torch.nn.Module):
            def forward(self, x, y):
                # 获取输入张量 x 的值
                n = x.item()
                # 返回 y 的和加上一个全为 1 的张量的总和
                return y.sum() + torch.ones(n, 5).sum()

        # 导出 M 并测试导出的模型
        ep = export(M(), (torch.tensor(1), torch.ones(4, 5)))

        # 检查是否抛出预期的 RuntimeError 异常
        error_msg = r"Invalid value range for -1 between"
        with self.assertRaisesRegex(RuntimeError, error_msg):
            _ = ep.module()(torch.tensor(-1), torch.randn(4, 5))

        # 检查模型输出是否与预期结果相似
        self.assertTrue(
            torch.allclose(
                ep.module()(torch.tensor(1), torch.ones(4, 5)),
                M()(torch.tensor(1), torch.ones(4, 5)),
            )
        )

    def test_constrain_decomp(self) -> None:
        # 定义一个测试用的神经网络模块
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.freq = torch.ones(5, 5)

            def forward(self, start_pos: torch.Tensor):
                # 获取输入张量 start_pos 的值
                pos = start_pos.item()
                # 检查 pos 是否是有效的大小（正整数）
                torch._check_is_size(pos)
                # 检查 pos 是否大于等于 0
                torch._check(pos >= 0)
                # 检查 pos 是否小于等于 4
                torch._check(pos <= 4)
                # 返回 self.freq 中位置 pos 处的元素乘以自身
                return self.freq[pos] * self.freq[pos]

        # 导出 M 并测试导出的模型
        ep = torch.export.export(M(), (torch.tensor(1),))
        
        # 检查导出的模型中的特定操作出现次数
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 2, exactly=True
        ).run(ep.graph_module.code)
        
        # 运行 M 模型的分解版本并检查特定操作的出现次数
        decompose_ep = ep.run_decompositions()
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 2, exactly=True
        ).run(decompose_ep.graph_module.code)

    def test_mixed_input(self):
        # 定义一个测试用的神经网络模块
        class Module(torch.nn.Module):
            def forward(self, a, b, alpha: int):
                # 返回张量 a 和 b 的加法结果，使用 alpha 加权
                return torch.add(a, b, alpha=alpha)

        # 创建 Module 的实例
        func = Module()

        # 创建测试用的张量 a、b 和整数 alpha
        a = torch.rand(1, 2)
        b = torch.rand(1, 2)
        alpha = 10

        # 导出 Module 并测试导出的模型
        exported = export(func, (a, b, alpha))
        
        # 遍历导出的模型中的节点并检查是否所有占位符的值是 Tensor 或 int 类型
        for node in exported.graph_module.graph.nodes:
            if node.op == "placeholder":
                self.assertTrue(isinstance(node.meta["val"], (Tensor, int)))
    def test_export_with_inline_constraints(self):
        # 定义一个继承自torch.nn.Module的类Module，用于模型定义
        class Module(torch.nn.Module):
            # 定义模型的前向传播函数，输入参数为x
            def forward(self, x):
                # 将张量x转换为标量a
                a = x.item()
                # 检查a是否大于等于4
                torch._check(a >= 4)
                # 检查a是否小于等于7
                torch._check(a <= 7)
                # 返回一个形状为(a, 4)的空张量
                return torch.empty((a, 4))

        # 创建Module类的实例f
        f = Module()
        # 导出模型f，并传入一个示例输入torch.tensor([5])
        ep = export(f, (torch.tensor([5]),))
        # 断言模型在输入torch.tensor([6])时输出张量的形状为(6, 4)
        self.assertEqual(ep.module()(torch.tensor([6])).shape, (6, 4))

        # 使用FileCheck工具检查导出模型的代码中"torch.ops.aten._assert_scalar.default"出现的次数为2
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 2, exactly=True
        ).run(ep.graph_module.code)

        # 使用断言检查运行时错误信息，确保在输入torch.tensor([30])时引发RuntimeError，并包含特定错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            r"Invalid value range for 30 between \[4, 7\]",
        ) as cm:
            # 调用导出模型处理输入torch.tensor([30])
            ep.module()(torch.tensor([30]))

    def test_export_with_inline_constraints_complex(self):
        # 定义一个继承自torch.nn.Module的类Module，用于模型定义
        class Module(torch.nn.Module):
            # 定义模型的前向传播函数，输入参数为x
            def forward(self, x):
                # 将张量x转换为标量a
                a = x.item()
                # 检查a是否大于等于4
                torch._check(a >= 4)
                # 检查a是否小于等于7
                torch._check(a <= 7)
                # 创建一个形状为(a, 4)的空张量
                empty = torch.empty((a, 4))
                # 返回torch.cat操作的结果，将empty按第一维进行转置，然后与形状为(6, a)的零张量拼接
                return torch.cat((empty.transpose(0, 1), torch.zeros(6, a)), 0)

        # 创建Module类的实例f
        f = Module()
        # 导出模型f，并传入一个示例输入torch.tensor([6])
        ep = export(f, (torch.tensor([6]),))
        # 断言模型在输入torch.tensor([5])时输出张量的形状为(10, 5)
        self.assertEqual(ep.module()(torch.tensor([5])).shape, (10, 5))
        # 使用FileCheck工具检查导出模型的代码中"torch.ops.aten._assert_scalar.default"出现的次数为2
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 2, exactly=True
        ).run(ep.graph_module.code)

    def test_to_module_with_mutated_buffer(self):
        # 定义一个继承自torch.nn.Module的类Foo
        class Foo(torch.nn.Module):
            # 类的初始化方法，在实例化时调用
            def __init__(self):
                # 调用父类的初始化方法
                super().__init__()
                # 注册一个名为"buf"的缓冲区，内容为形状为(1,)的零张量
                self.register_buffer("buf", torch.zeros(1))

            # 定义模型的前向传播函数，输入参数为x
            def forward(self, x):
                # 缓冲区"buf"的值增加1
                self.buf.add_(1)
                # 返回x的元素求和与缓冲区"buf"的元素求和的结果
                return x.sum() + self.buf.sum()

        # 导出Foo类的实例，并传入一个形状为(5, 5)的示例输入
        exported = export(Foo(), (torch.ones(5, 5),))
        # 获取导出模型的实例stateful_gm
        stateful_gm = exported.module()
        # 使用导出模型处理形状为(5, 5)的示例输入，获取返回值
        export_return_val = stateful_gm(torch.ones(5, 5))
        # 创建一个eager实例
        eager = Foo()
        # 使用eager实例处理形状为(5, 5)的示例输入，获取返回值
        eager_return_val = eager(torch.ones(5, 5))
        # 断言导出模型和eager模型的返回值在数值上相等
        self.assertTrue(torch.allclose(eager_return_val, export_return_val))

        # 遍历导出模型的所有缓冲区，并断言它们与形状为(1,)的全1张量在数值上相等
        for name, buffer in stateful_gm.named_buffers():
            self.assertTrue(torch.allclose(torch.ones(1), buffer))

        # 对导出模型的图进行死代码消除操作，并断言操作未引起任何变化
        changed = stateful_gm.graph.eliminate_dead_code()
        self.assertFalse(changed)

        # 再次使用导出模型处理形状为(5, 5)的示例输入，获取返回值
        self.assertTrue(
            torch.allclose(stateful_gm(torch.ones(5, 5)), eager(torch.ones(5, 5)))
        )

        # 遍历导出模型的所有缓冲区，并断言它们与值为2的浮点数张量在数值上相等
        for name, buffer in stateful_gm.named_buffers():
            self.assertTrue(torch.allclose(torch.tensor(2, dtype=torch.float), buffer))
    # 定义测试方法，用于测试带有变异缓冲区的多个情况
    def test_to_module_with_mutated_buffer_multiple(self):
        # 定义 Bar 类，继承自 torch.nn.Module
        class Bar(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 注册名为 "buf" 的缓冲区，初始值为 torch.ones(1)
                self.register_buffer("buf", torch.ones(1))

            # 前向传播方法
            def forward(self, x):
                # 在缓冲区 "buf" 上执行原地加法操作
                self.buf.add_(1)
                # 返回 x 的和加上缓冲区 "buf" 的和
                return x.sum() + self.buf.sum()

        # 定义 Foo 类，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 注册名为 "buf" 的缓冲区，初始值为 torch.zeros(1)
                self.register_buffer("buf", torch.zeros(1))
                # 创建 Bar 类的实例并赋值给属性 "bar"
                self.bar = Bar()

            # 前向传播方法
            def forward(self, x):
                # 在缓冲区 "buf" 上执行原地加法操作
                self.buf.add_(1)
                # 在属性 "bar" 的缓冲区 "buf" 上执行原地加法操作
                self.bar.buf.add_(2)
                # 调用属性 "bar" 的 forward 方法，并传入参数 x
                bar = self.bar(x)
                # 返回 bar 的和加上缓冲区 "buf" 的和
                return bar.sum() + self.buf.sum()

        # 导出模型 Foo 并传入输入参数 torch.ones(5, 5)，得到导出后的模型对象
        exported = export(Foo(), (torch.ones(5, 5),))
        # 获取导出后的模型对象中的 module
        stateful_gm = exported.module()
        # 在导出后的模型对象上执行前向传播，传入参数 torch.ones(5, 5)，并获取返回值
        export_return_val = stateful_gm(torch.ones(5, 5))
        # 创建 Foo 类的实例 eager，并在该实例上执行前向传播，传入参数 torch.ones(5, 5)，并获取返回值
        eager = Foo()
        eager_return_val = eager(torch.ones(5, 5))
        # 断言 eager_return_val 与 export_return_val 在所有元素上近似相等
        self.assertTrue(torch.allclose(eager_return_val, export_return_val))

        # 遍历 stateful_gm 中所有命名的缓冲区，验证其值是否与预期值近似相等
        for name, buffer in stateful_gm.named_buffers():
            if name == "L__self___buf":
                self.assertTrue(torch.allclose(torch.ones(1), buffer))
            if name == "L__self___bar_buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(4, dtype=torch.float), buffer)
                )

        # 在 stateful_gm 的计算图中消除死代码
        changed = stateful_gm.graph.eliminate_dead_code()
        # 断言是否有代码变化
        self.assertFalse(changed)
        # 再次验证 stateful_gm 在传入参数 torch.ones(5, 5) 上执行前向传播的结果是否与 eager 相同
        self.assertTrue(
            torch.allclose(stateful_gm(torch.ones(5, 5)), eager(torch.ones(5, 5)))
        )

        # 遍历 stateful_gm 中所有命名的缓冲区，验证其值是否与预期值近似相等
        for name, buffer in stateful_gm.named_buffers():
            if name == "L__self___buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(2, dtype=torch.float), buffer)
                )
            if name == "L__self___bar_buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(7, dtype=torch.float), buffer)
                )
    # 定义一个测试方法，用于测试运行时断言是否有效
    def test_runtime_assert_for_prim(self):
        # 定义一个继承自torch.nn.Module的类Foo
        class Foo(torch.nn.Module):
            # 定义类Foo的前向传播方法，接受两个参数x和y，返回它们的和
            def forward(self, x, y):
                return x + y
        
        # 创建Foo类的实例foo
        foo = Foo()
        # 创建一个大小为7x5的全1张量tensor_inp
        tensor_inp = torch.ones(7, 5)
        # 创建一个名为dim0_x的torch.export.Dim对象，最小值为6
        dim0_x = torch.export.Dim("dim0_x", min=6)
        # 创建一个动态形状的字典，指定"x"维度为{0: dim0_x}，"y"维度为None
        dynamic_shapes = {"x": {0: dim0_x}, "y": None}
        # 使用torch.export.export方法导出foo模型，输入为(tensor_inp, 5)，并指定动态形状为dynamic_shapes
        exported = torch.export.export(
            foo, (tensor_inp, 5), dynamic_shapes=dynamic_shapes
        )
        # 断言两个张量的所有元素在误差范围内相等，验证导出模型的正确性
        self.assertTrue(
            torch.allclose(
                exported.module()(torch.ones(8, 5), 5), foo(torch.ones(8, 5), 5)
            )
        )
        # 使用self.assertRaisesRegex断言运行时错误，确保捕获到期望的异常信息
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[1] to be equal to 5, but got 6"),
        ):
            # 调用导出模型并传入(torch.ones(8, 5), 6)，期望捕获到特定异常信息
            _ = exported.module()(torch.ones(8, 5), 6)

        # 重新导出foo模型，输入为(tensor_inp, 5.0)，动态形状为dynamic_shapes
        exported = torch.export.export(
            foo, (tensor_inp, 5.0), dynamic_shapes=dynamic_shapes
        )
        # 使用self.assertRaisesRegex再次断言运行时错误，确保捕获到期望的异常信息
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[1] to be equal to 5.0, but got 6.0"),
        ):
            # 调用导出模型并传入(torch.ones(7, 5), 6.0)，期望捕获到特定异常信息
            _ = exported.module()(torch.ones(7, 5), 6.0)

    # 定义一个测试方法，用于测试运行时断言对字符串参数的处理是否有效
    def test_runtime_assert_for_prm_str(self):
        # 定义一个继承自torch.nn.Module的类Foo
        class Foo(torch.nn.Module):
            # 定义类Foo的前向传播方法，接受三个参数a、b和mode，执行torch.div运算
            def forward(self, a, b, mode):
                return torch.div(a, b, rounding_mode=mode)

        # 创建Foo类的实例foo
        foo = Foo()
        # 创建输入参数元组inps，包含一个4x4的随机张量，一个4维的随机张量，和一个字符串"trunc"
        inps = (torch.randn(4, 4), torch.randn(4), "trunc")
        # 使用export方法导出foo模型，输入为inps
        exported = export(foo, inps)
        # 使用self.assertRaisesRegex断言运行时错误，确保捕获到期望的异常信息
        with self.assertRaisesRegex(
            RuntimeError, "to be equal to trunc, but got floor"
        ):
            # 调用导出模型并传入(torch.randn(4, 4), torch.randn(4), "floor")，期望捕获到特定异常信息
            _ = exported.module()(torch.randn(4, 4), torch.randn(4), "floor")
        # 断言两个张量的所有元素在误差范围内相等，验证导出模型的正确性
        self.assertTrue(torch.allclose(exported.module()(*inps), foo(*inps)))

    # 标记为测试预期失败的注释，用于指示测试框架该用例目前应该失败
    @testing.expectedFailureTrainingIRToRunDecomp  # T193701923
    def test_to_module_with_mutated_buffer_multiple_update_sub_later(self):
        # 定义测试方法，用于验证具有多次更新后子级的变异缓冲区转换为模块
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在模块初始化时注册一个缓冲区“buf”，初始值为1
                self.register_buffer("buf", torch.ones(1))

            def forward(self, x):
                # 在前向传播中，对缓冲区“buf”执行加法操作
                self.buf.add_(1)
                return x.sum() + self.buf.sum()

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在模块初始化时注册一个缓冲区“buf”，初始值为0
                self.register_buffer("buf", torch.zeros(1))
                # 创建一个名为bar的Bar类实例
                self.bar = Bar()

            def forward(self, x):
                # 在前向传播中，对主模块的缓冲区“buf”执行加法操作
                self.buf.add_(1)
                # 调用子模块bar的前向传播，获取返回值
                bar = self.bar(x)
                # 对子模块bar的缓冲区“buf”执行加法操作
                self.bar.buf.add_(2)
                # 返回所有操作的和
                return bar.sum() + self.buf.sum()

        # 导出模块Foo并创建具有初始化参数的实例
        exported = export(Foo(), (torch.ones(5, 5),))
        # 获取导出模块的状态图
        stateful_gm = exported.module()
        # 在导出模块上执行前向传播
        export_return_val = stateful_gm(torch.ones(5, 5))
        # 创建一个新的Foo类实例
        eager = Foo()
        # 在新实例上执行前向传播
        eager_return_val = eager(torch.ones(5, 5))
        # 断言导出模块和新实例的返回值相等
        self.assertTrue(torch.allclose(eager_return_val, export_return_val))

        # 遍历导出模块的所有注册缓冲区
        for name, buffer in stateful_gm.named_buffers():
            # 检查名为"L__self___buf"的缓冲区是否与预期值相等
            if name == "L__self___buf":
                self.assertTrue(torch.allclose(torch.ones(1), buffer))
            # 检查名为"L__self___bar_buf"的缓冲区是否与预期值相等
            if name == "L__self___bar_buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(4, dtype=torch.float), buffer)
                )

        # 在状态图上执行消除死代码的操作
        changed = stateful_gm.graph.eliminate_dead_code()
        # 断言消除死代码操作未引起任何更改
        self.assertFalse(changed)
        # 断言再次执行导出模块和新实例的前向传播返回值相等
        self.assertTrue(
            torch.allclose(stateful_gm(torch.ones(5, 5)), eager(torch.ones(5, 5)))
        )

        # 再次遍历导出模块的所有注册缓冲区
        for name, buffer in stateful_gm.named_buffers():
            # 检查名为"L__self___buf"的缓冲区是否与预期值相等
            if name == "L__self___buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(2, dtype=torch.float), buffer)
                )
            # 检查名为"L__self___bar_buf"的缓冲区是否与预期值相等
            if name == "L__self___bar_buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(7, dtype=torch.float), buffer)
                )
    # 定义测试函数 `test_retracable_ep`
    def test_retracable_ep(self):
        # 定义内部类 `Bar`，继承自 `torch.nn.Module`
        class Bar(torch.nn.Module):
            # 类构造函数，初始化模块
            def __init__(self):
                super().__init__()
                # 注册一个名为 `buf` 的缓冲区，初始值为 torch.ones(1)
                self.register_buffer("buf", torch.ones(1))

            # 前向传播函数
            def forward(self, x):
                # 在 `buf` 上加 1
                self.buf.add_(1)
                # 返回 `x` 的和加上 `self.buf` 的和
                return x.sum() + self.buf.sum()

        # 定义内部类 `Foo`，继承自 `torch.nn.Module`
        class Foo(torch.nn.Module):
            # 类构造函数，初始化模块
            def __init__(self):
                super().__init__()
                # 注册一个名为 `buf` 的缓冲区，初始值为 torch.zeros(1)
                self.register_buffer("buf", torch.zeros(1))
                # 创建 `Bar` 类的实例并赋值给属性 `bar`
                self.bar = Bar()

            # 前向传播函数
            def forward(self, x):
                # 在 `buf` 上加 1
                self.buf.add_(1)
                # 调用 `bar` 对象的前向传播函数，将 `x` 传入，并将返回值赋给 `bar`
                bar = self.bar(x)
                # 在 `bar.buf` 上加 2
                self.bar.buf.add_(2)
                # 返回 `bar` 的和加上 `self.buf` 的和
                return bar.sum() + self.buf.sum()

        # 创建一个全为 1 的 5x5 的张量 `inp`
        inp = torch.ones(5, 5)
        # 导出 `Foo` 模块，并传入 `inp` 作为参数，返回导出的模块
        exported = torch.export.export(Foo(), (inp,))
        # 重新导出导出的模块的子模块，并传入 `inp` 作为参数，返回重新导出的模块
        reexported = torch.export.export(exported.module(), (inp,))
        
        # 断言两个张量近似相等，使用 `Foo()` 对 `inp` 的结果与重新导出的模块对 `inp` 的结果
        self.assertTrue(torch.allclose(Foo()(inp), reexported.module()(inp)))

        # 创建一个维度对象 `dim0_x`，表示维度名称为 "dim0_x"
        dim0_x = torch.export.Dim("dim0_x")
        # 导出 `Foo` 模块，并传入 `inp` 和动态形状参数 `({0: dim0_x},)`，返回导出的模块
        exported = torch.export.export(Foo(), (inp,), dynamic_shapes=({0: dim0_x},))
        # 重新导出导出的模块的子模块，并传入 `inp` 作为参数，返回重新导出的模块
        reexported = torch.export.export(exported.module(), (inp,))
        
        # 使用断言检查是否抛出 `RuntimeError` 异常，并检查异常消息中是否包含指定字符串
        with self.assertRaisesRegex(
            RuntimeError, "shape\[0\] to be equal to 5, but got 7"
        ):
            # 调用重新导出的模块对全为 1 的 7x5 张量进行处理
            reexported.module()(torch.ones(7, 5))

        # 重新导出导出的模块的子模块，并传入 `inp` 和动态形状参数 `({0: dim0_x},)`，返回重新导出的模块
        reexported = torch.export.export(
            exported.module(), (inp,), dynamic_shapes=({0: dim0_x},)
        )
        
        # 断言两个张量近似相等，使用 `Foo()` 对全为 1 的 7x5 张量的结果与重新导出的模块对其结果
        self.assertTrue(
            torch.allclose(
                Foo()(torch.ones(7, 5)), reexported.module()(torch.ones(7, 5))
            )
        )

        # 导出 `Foo` 模块，并传入 `inp` 和动态形状参数 `{"x": {0: dim0_x_v2}}`，返回导出的模块
        exported_v2 = torch.export.export(
            Foo(), (inp,), dynamic_shapes={"x": {0: dim0_x_v2}}
        )
        
        # 使用断言检查是否抛出 `RuntimeError` 异常，并检查异常消息中是否包含指定字符串
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[0].shape[0] to be >= 3, but got 2"),
        ):
            # 调用重新导出的模块对全为随机数的 2x2 张量进行处理
            torch.export.export(exported_v2.module(), (torch.randn(2, 2),))

    # 定义测试函数 `test_export_cond`
    def test_export_cond(self):
        # 定义内部类 `A`，继承自 `torch.nn.Module`
        class A(torch.nn.Module):
            # 类构造函数，初始化模块
            def __init__(self):
                super().__init__()
                # 注册一个名为 `buffer` 的缓冲区，初始值为 torch.ones(6, 4)
                self.register_buffer("buffer", torch.ones(6, 4))

            # 前向传播函数
            def forward(self):
                # 返回 `buffer` 的余弦值
                return self.buffer.cos()

        # 定义内部类 `Foo`，继承自 `torch.nn.Module`
        class Foo(torch.nn.Module):
            # 类构造函数，初始化模块
            def __init__(self):
                super().__init__()
                # 创建 `A` 类的实例并赋值给属性 `a`
                self.a = A()

            # 前向传播函数，带有条件分支
            def forward(self, x):
                # 定义真值分支函数 `true_fn`
                def true_fn(x):
                    # 返回 `x` 的余弦值加上 `self.a()` 的和
                    return x.cos() + self.a().sum()

                # 定义假值分支函数 `false_fn`
                def false_fn(x):
                    # 返回 `x` 的正弦值
                    return x.sin()

                # 根据 `x.shape[0] > 4` 的条件选择并调用对应的函数
                return cond(x.shape[0] > 4, true_fn, false_fn, [x])

        # 创建一个全为 1 的 6x4 的张量 `inp`
        inp = torch.ones(6, 4)
        # 导出 `Foo` 模块，并传入 `inp` 作为参数，返回导出的模块
        ep = export(
            Foo(),
            (inp,),
        )
        
        # 断言两个张量近似相等，使用导出的模块对全为 1 的 6x4 张量的结果与 `Foo()` 对其结果
        self.assertTrue(
            torch.allclose(ep.module()(torch.ones(6, 4)), Foo()(torch.ones(6, 4)))
        )
    # 定义测试方法，验证 torch.ops.aten.lift_fresh_copy 函数的导出
    def test_aten_lift_fresh_copy(self):
        # 定义一个简单的 PyTorch 模块 M，其 forward 方法调用 torch.ops.aten.lift_fresh_copy
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.lift_fresh_copy(x)

        # 导出模块 M 的计算图
        ep = export(M(), (torch.ones(6, 4),))
        found = False

        # 期望在计算图的代码中找到 "torch.ops.aten.clone.default" 的确切出现次数为 1
        op = "torch.ops.aten.clone.default"
        FileCheck().check_count(op, 1, exactly=True).run(ep.graph_module.code)

    # 测试条件下的缓冲区处理
    def test_cond_buffers(self):
        # 定义一个复杂一些的 PyTorch 模块 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个不需要梯度的参数 param
                self.register_parameter(
                    "param", torch.nn.Parameter(torch.ones(2, 3), requires_grad=False)
                )
                # 注册一个缓冲区 buffer
                self.register_buffer("buffer", torch.ones(2, 3) + 1)

            # 条件为真时的处理函数
            def true_fn(self, x):
                return x + self.param

            # 条件为假时的处理函数
            def false_fn(self, x):
                return x + self.buffer

            # 前向传播方法，根据条件长度来选择不同的处理函数
            def forward(self, x):
                return cond(x.shape[0] == 4, self.true_fn, self.false_fn, [x])

        # 输入张量
        inp = torch.ones(2, 3)
        # 导出模块 M 的计算图
        ep = torch.export.export(M(), (inp,))
        # 更改输入张量为随机值
        inp = torch.randn(2, 3)
        # 获取导出模块的实例
        epm = ep.module()
        # 使用 assert 判断两个张量是否近似相等
        self.assertTrue(torch.allclose(epm(inp), M()(inp)))

        # 遍历导出模块中的所有子模块
        for gm in epm.named_modules():
            # 如果子模块不是 torch.fx.GraphModule 类型，则继续下一个循环
            if not isinstance(gm, torch.fx.GraphModule):
                continue
            # 断言计算图中的占位符节点数量为 1
            self.assertEqual(
                len([node for node in gm.graph.nodes if node.op == "placeholder"]), 1
            )

    # test_map_buffers 引用模块外部的 map_fn
    @unittest.expectedFailure
    def test_map_buffers(self):
        # 定义一个简单的 PyTorch 模块 M1
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个不需要梯度的参数 param
                self.register_parameter(
                    "param", torch.nn.Parameter(torch.tensor(5), requires_grad=False)
                )
                # 注册一个缓冲区 buffer
                self.register_buffer("buffer", torch.tensor(6) + 1)

        # 创建模块 M1 的实例 m1
        m1 = M1()

        # 定义 map_fn 函数，接受两个输入参数 x 和 y
        def map_fn(x, y):
            # 在函数内部使用了模块 M1 的 param 和 buffer 属性
            z = x + y + m1.param + m1.buffer
            z.add_(4)
            return z

        # 定义一个 PyTorch 模块 M
        class M(torch.nn.Module):
            def forward(self, xs, y):
                # 使用 map 函数将 map_fn 应用于 xs 和 y
                return map(map_fn, xs, y)

        # 示例输入
        example_inputs = (torch.ones(3, 2), torch.tensor(3))
        # 导出模块 M 的计算图
        ep = torch.export.export(M(), example_inputs)
        # 更改示例输入的值
        example_inputs = (torch.randn(3, 2), torch.tensor(3))
        # 获取导出模块的实例
        epm = ep.module()
        # 使用 assert 判断两个张量是否近似相等
        self.assertTrue(torch.allclose(epm(*example_inputs), M()(*example_inputs)))

        # 遍历导出模块中的所有子模块
        for gm in epm.named_modules():
            # 如果子模块不是 torch.fx.GraphModule 类型，则继续下一个循环
            if not isinstance(gm, torch.fx.GraphModule):
                continue
            # 断言计算图中的占位符节点数量为 2
            self.assertEqual(
                len([node for node in gm.graph.nodes if node.op == "placeholder"]), 2
            )

    @testing.expectedFailureSerDer  # We don't preserve metadata on graph module
    @testing.expectedFailureNonStrict
    def test_retrace_graph_level_meta_preservation(self):
        # 定义一个内部的 Torch 模块类 Foo
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # 如果输入张量 x 的第一个维度大于4，则返回其余弦值
                if x.shape[0] > 4:
                    return x.cos()
                # 否则返回其正弦值
                return x.sin()

        # 创建一个形状为 (7, 5) 的输入张量
        inp = torch.ones(7, 5)
        # 创建一个维度描述对象，最小值为 6
        dim0_x = torch.export.Dim("dim0_x", min=6)
        # 导出 Foo 模块的状态，包括动态形状信息
        exported = torch.export.export(Foo(), (inp,), dynamic_shapes={"x": {0: dim0_x}})
        # 获取导出后的模块对象
        stateful_module = exported.module()
        # 断言导出模块的元数据中输入形状约束的长度为 1
        self.assertTrue(len(stateful_module.meta["input_shape_constraints"]), 1)

        # 重新导出模块，包括动态形状信息
        re_exported = export(stateful_module, (inp,), dynamic_shapes=({0: dim0_x},))
        # 断言重新导出模块的图模块的元数据中输入形状约束的长度为 1
        self.assertTrue(
            len(re_exported.graph_module.meta["input_shape_constraints"]) == 1
        )
        # 断言两次调用相同输入时导出模块和重新导出模块的输出值近似相等
        self.assertTrue(
            torch.allclose(
                exported.module()(torch.ones(7, 5)),
                re_exported.module()(torch.ones(7, 5)),
            )
        )

        # 再次重新导出模块，不包括动态形状信息
        re_exported_v2 = export(exported.module(), (inp,))
        # 断言第二次重新导出模块的图模块的元数据中输入形状约束的长度为 0
        self.assertTrue(
            len(re_exported_v2.graph_module.meta["input_shape_constraints"]) == 0
        )
        # 断言两次调用相同输入时导出模块和第二次重新导出模块的输出值近似相等
        self.assertTrue(
            torch.allclose(
                exported.module()(torch.ones(7, 5)),
                re_exported_v2.module()(torch.ones(7, 5)),
            )
        )

    def test_check_is_size_error(self):
        # 定义一个继承自 torch.nn.Module 的模块类 Module
        class Module(torch.nn.Module):
            def forward(self, x):
                # 将输入张量 x 的标量值赋给变量 a
                a = x.item()
                # 返回一个形状为 (a, 4) 的张量，a 的值被视为 view 函数的参数
                # 这里不能自动推断 a 是一个尺寸，因为 view 函数接受 -1
                return torch.randn(24).view(a, 4)

        # 创建 Module 类的实例 f
        f = Module()
        # 如果测试方法名不是严格测试，则使用 Torch FX 中的 GuardOnDataDependentSymNode 错误
        if is_non_strict_test(self._testMethodName):
            error = torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode
            error_msg = r"Could not guard on data-dependent expression"
        else:
            # 否则使用 torch._dynamo.exc.UserError 错误
            error = torch._dynamo.exc.UserError
            error_msg = (
                r"Tried to use data-dependent value in the subsequent computation"
            )
        # 使用断言检查是否抛出了预期的错误和错误消息
        with self.assertRaisesRegex(error, error_msg):
            _ = export(f, (torch.tensor(6),))

    def test_train_eval_on_exported_preautograd_module(self):
        # 定义一个内部的 Torch 模块类 Foo
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # 如果输入张量 x 的第一个维度大于4，则返回其余弦值
                if x.shape[0] > 4:
                    return x.cos()
                # 否则返回其正弦值
                return x.sin()

        # 使用 _export 函数导出 Foo 模块，包括输入张量为 (7, 5) 和预分派参数的信息
        graph_module = _export(Foo(), (torch.ones(7, 5),), pre_dispatch=True).module()
        # 使用断言检查调用 train() 方法时是否抛出了预期的 NotImplementedError
        with self.assertRaisesRegex(
            NotImplementedError, r"Calling train\(\) is not supported yet."
        ):
            graph_module.train()

        # 使用断言检查调用 eval() 方法时是否抛出了预期的 NotImplementedError
        with self.assertRaisesRegex(
            NotImplementedError, r"Calling eval\(\) is not supported yet."
        ):
            graph_module.eval()

    # TODO Retracing a module with constant attrs don't work.(T193692674)
    @testing.expectedFailureTrainingIRToRunDecomp
    @testing.expectedFailureRetraceability  # 标记此测试预期失败，用于问题追溯
    def test_lifted_constants(self) -> None:
        class Module(torch.nn.Module):
            def forward(self, x):
                return x + torch.tensor(3)

        f = Module()
        ep = export(f, (torch.tensor(1),))

        self.assertEqual(len(ep.graph_signature.input_specs), 2)  # 检查导出结果的输入规格数量是否为2
        self.assertEqual(len(ep.constants), 1)  # 检查导出结果的常量数量是否为1

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.tensor(3)

            def forward(self, x):
                list_tensor = [torch.tensor(3), torch.tensor(4)]
                return x + self.a + list_tensor[0] + list_tensor[1]

        ep = export(Foo(), (torch.tensor(1),))

        self.assertEqual(len(ep.graph_signature.input_specs), 4)  # 检查导出结果的输入规格数量是否为4
        self.assertEqual(len(ep.state_dict), 0)  # 检查导出结果的状态字典数量是否为0
        self.assertEqual(len(ep.constants), 3)  # 检查导出结果的常量数量是否为3

        inp = (torch.tensor(5),)
        self.assertTrue(torch.allclose(ep.module()(*inp), Foo()(*inp)))  # 检查导出结果是否与预期结果在输入上一致

        transform = ep.run_decompositions()
        self.assertEqual(len(ep.graph_signature.input_specs), 4)  # 再次检查导出结果的输入规格数量是否为4
        self.assertTrue(torch.allclose(ep.module()(*inp), transform.module()(*inp)))  # 检查转换后的模型运行结果与原模型是否一致

    @testing.expectedFailureRetraceability  # 标记此测试预期失败，用于问题追溯
    @testing.expectedFailureTrainingIRToRunDecomp  # 标记此测试预期失败，用于问题追溯和训练 IR 到运行分解
    def test_tensor_attribute_zero_args(self):
        class Foo(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.x = torch.tensor(value)

            def forward(self):
                return self.x.clone()

        m = Foo([1, 2])
        ep = export(m, ())
        self.assertEqual(ep.graph_signature.lifted_tensor_constants, ["x"])  # 检查导出结果中提升的张量常量是否为 ["x"]
    def test_preserve_shape_dynamism_for_unused_inputs(self):
        @dataclass
        class Input:
            f: torch.Tensor  # 定义数据类，包含一个名为 f 的 torch.Tensor 对象
            p: torch.Tensor  # 定义数据类，包含一个名为 p 的 torch.Tensor 对象

        torch._export.utils.register_dataclass_as_pytree_node(
            Input,
            serialized_type_name="test_preserve_shape_dynamism_for_unused_inputs.Input",  # 将数据类注册为 Pytree 节点
        )

        class Module(torch.nn.Module):
            def forward(self, x: Input):
                return x.f + 1  # 返回输入数据类中 f 属性的值加一

        mod = Module()
        example_inputs = (Input(f=torch.ones(10, 4), p=torch.zeros(10, 4)),)  # 创建示例输入数据类
        ep_static = torch.export.export(mod, example_inputs)  # 导出静态计算图
        for node in ep_static.graph.nodes:
            if node.op == "placeholder":
                for s in node.meta["val"].shape:
                    self.assertIsInstance(s, int)  # 验证静态计算图中占位符的形状为整数类型

        dim0_x_f, dim0_x_p = torch.export.dims("dim0_x_f", "dim0_x_p")  # 定义符号维度
        dynamic_shapes = {"x": [{0: dim0_x_f}, {0: dim0_x_p}]}  # 指定动态形状
        ep_dynamic = torch.export.export(
            mod, example_inputs, dynamic_shapes=dynamic_shapes  # 导出动态计算图
        )
        for node in ep_dynamic.graph.nodes:
            if node.op == "placeholder":
                for i, s in enumerate(node.meta["val"].shape):
                    if i == 0:
                        self.assertIsInstance(s, torch.SymInt)  # 验证动态计算图中第一个占位符的形状为符号整数类型
                    else:
                        self.assertIsInstance(s, int)  # 验证其他占位符的形状为整数类型

    def test_multiple_definitions_same_name_dim(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)  # 返回输入张量 x 和 y 的矩阵乘积

        A = torch.export.Dim("C", min=3)  # 定义符号维度 A
        B = torch.export.Dim("C", max=12)  # 定义符号维度 B
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Found different definitions Dim\\(.*min=3\\) and Dim\\(.*max=12\\) "
            "for the same symbolic dimension",  # 检查同一符号维度的不同定义是否引发错误
        ):
            torch.export.export(
                Foo(),
                (torch.randn(10, 10), torch.randn(10, 10)),
                dynamic_shapes={"x": (A, B), "y": (B, A)},  # 导出带有符号维度的计算图
            )

    def test_export_with_wrong_inputs(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + x  # 返回输入张量 x 的两倍

        exported_program = export(MyModule(), (torch.rand(2, 3),), {})  # 导出模块
        with self.assertRaisesRegex(ValueError, "Trying to flatten user inputs"):
            exported_program.module()(torch.rand(2, 3), torch.rand(2, 3))  # 调用导出模块的实例，并检查错误是否被引发
    # 定义单元测试函数，用于测试简单导出功能
    def test_export_decomps_simple(self):
        # 定义简单的神经网络模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个线性层，输入维度为10，输出维度为1
                self.lin = torch.nn.Linear(10, 1)

            # 前向传播函数
            def forward(self, x):
                return self.lin(x)

        # 创建输入数据（包含一个随机张量）
        inp = (torch.randn(5, 10),)
        # 实例化神经网络模型对象
        m = M()
        # 调用导出函数，生成导出对象 ep
        ep = export(m, inp)
        # 获取导出对象的状态字典
        state_dict = ep.state_dict

        # 断言：验证导出对象执行前向传播后输出与原模型执行前向传播输出在数值上的接近性
        self.assertTrue(torch.allclose(ep.module()(*inp), m(*inp)))

        # 对导出对象执行分解运算
        core_aten_ep = ep.run_decompositions()
        # 使用 FileCheck 检查运算图代码中 torch.ops.aten.permute.default 的出现次数为1
        FileCheck().check_count("torch.ops.aten.permute.default", 1, exactly=True).run(
            core_aten_ep.graph_module.code
        )
        # 使用 FileCheck 检查运算图代码中 torch.ops.aten.t.default 的出现次数为0
        FileCheck().check_count("torch.ops.aten.t.default", 0, exactly=True).run(
            core_aten_ep.graph_module.code
        )
        # 断言：验证导出对象执行分解后输出与原模型执行前向传播输出在数值上的接近性
        self.assertTrue(torch.allclose(core_aten_ep.module()(*inp), m(*inp)))
        # 断言：验证导出对象的状态字典的内存地址与原状态字典的内存地址相同
        self.assertEqual(id(state_dict), id(ep.state_dict))

    # 定义单元测试函数，用于测试动态导出功能
    def test_export_decomps_dynamic(self):
        # 定义简单的神经网络模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个线性层，输入维度为10，输出维度为1
                self.lin = torch.nn.Linear(10, 1)

            # 前向传播函数
            def forward(self, x):
                return self.lin(x)

        # 创建输入数据（包含一个随机张量）
        inp = (torch.randn(5, 10),)
        # 实例化神经网络模型对象
        m = M()
        # 调用导出函数，生成动态导出对象 ep，指定动态形状
        ep = export(m, inp, dynamic_shapes={"x": {0: Dim("batch")}})

        # 对导出对象执行分解运算
        core_aten_ep = ep.run_decompositions()

        # 获取输入节点，即运算图中的占位符节点，检查其形状元数据中的第一个维度是否为 torch.SymInt 类型
        input_node = [
            node for node in core_aten_ep.graph.nodes if node.op == "placeholder"
        ][-1]
        self.assertTrue(isinstance(input_node.meta["val"].shape[0], torch.SymInt))

        # 使用 FileCheck 检查运算图代码中 torch.ops.aten.permute.default 的出现次数为1
        FileCheck().check_count("torch.ops.aten.permute.default", 1, exactly=True).run(
            core_aten_ep.graph_module.code
        )
        # 使用 FileCheck 检查运算图代码中 torch.ops.aten.t.default 的出现次数为0
        FileCheck().check_count("torch.ops.aten.t.default", 0, exactly=True).run(
            core_aten_ep.graph_module.code
        )
        # 断言：验证导出对象执行分解后输出与原模型执行前向传播输出在数值上的接近性
        self.assertTrue(torch.allclose(core_aten_ep.module()(*inp), m(*inp)))

    # 定义单元测试函数，用于测试 torch.nonzero 函数的导出
    def test_nonzero_2(self):
        # 定义包含 torch.nonzero 调用的简单神经网络模型类
        class Module(torch.nn.Module):
            # 前向传播函数，返回输入张量的非零元素的索引
            def forward(self, x):
                return torch.nonzero(x)

        # 实例化模型对象 f
        f = Module()
        # 调用导出函数，生成导出对象 ep
        ep = export(f, (torch.ones(2),))
        # 创建输入张量 inp，包含两个随机元素
        inp = torch.randn(2)
        # 断言：验证导出对象执行前向传播后输出与 torch.nonzero 函数在数值上的接近性
        self.assertTrue(torch.allclose(ep.module()(inp), torch.nonzero(inp)))
    def test_redundant_asserts(self):
        # 定义一个内部类 Foo，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 重写 forward 方法，接受输入 x
            def forward(self, x):
                # 将 x 转换为 Python 标量
                y = x.item()
                # 检查 y 是否符合张量大小的要求
                torch._check_is_size(y)
                # 返回一个大小为 y 的全零张量
                return torch.zeros(y)

        # 创建 Foo 类的实例对象 f
        f = Foo()

        # 导出模型 f，并传入一个张量作为输入
        ep = export(f, (torch.tensor([3]),))

        # 使用 FileCheck 检查导出模型中符号约束函数的调用次数为 1
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range.default", 1, exactly=True
        ).run(ep.graph_module.code)
        # 使用 FileCheck 检查导出模型中标量断言函数的调用次数为 1
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 1, exactly=True
        ).run(ep.graph_module.code)

        # 运行模型的分解操作
        ep = ep.run_decompositions()

        # 使用 FileCheck 检查分解后模型中符号约束函数的调用次数为 1
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range.default", 1, exactly=True
        ).run(ep.graph_module.code)
        # 使用 FileCheck 检查分解后模型中标量断言函数的调用次数为 1
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 1, exactly=True
        ).run(ep.graph_module.code)

    def test_non_arg_name_dynamic_shapes_api(self):
        # 定义一个内部类 Foo，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 重写 forward 方法，接受输入 a 和 b
            def forward(self, a, b):
                # 返回 a 和 b 张量的和
                return a.sum() + b.sum()

        # 创建 Foo 类的实例对象 foo
        foo = Foo()
        # 创建一个动态形状的维度对象 dim
        dim = torch.export.Dim("dim")
        # 导出模型 foo，并传入两个张量作为输入，其中第一个张量具有动态形状
        ep = torch.export.export(
            foo,
            (torch.randn(4, 4), torch.randn(4, 4)),
            dynamic_shapes=(None, {0: dim}),
        )

        # 创建测试输入张量 test_inp
        test_inp = (torch.randn(4, 4), torch.randn(7, 4))
        # 使用断言检查导出模型的输出与原始模型的输出是否相等
        self.assertEqual(ep.module()(*test_inp), foo(*test_inp))

        # 使用另一种动态形状参数配置重新导出模型 foo
        ep_v2 = torch.export.export(
            foo,
            (torch.randn(4, 4), torch.randn(4, 4)),
            dynamic_shapes=(None, None),
        )
        # 使用断言检查是否引发了预期的运行时错误
        with self.assertRaisesRegex(
            RuntimeError, "shape\[0\] to be equal to 4, but got 7"
        ):
            ep_v2.module()(*test_inp)

    def test_constant_output(self):
        # 定义一个输出固定值的模块 ModuleConstant
        class ModuleConstant(torch.nn.Module):
            # 初始化方法，初始化一个随机张量 b
            def __init__(self):
                super().__init__()
                self.b = torch.randn(3, 2)

            # 重写 forward 方法，返回固定的张量 b
            def forward(self):
                return self.b

        # 定义一个嵌套模块 ModuleNestedConstant
        class ModuleNestedConstant(torch.nn.Module):
            # 初始化方法，初始化一个随机张量 bff
            def __init__(self):
                super().__init__()
                self.bff = torch.randn(3, 2)

            # 重写 forward 方法，接受输入 x 和 y，返回一个包含预测值和张量 bff 的字典
            def forward(self, x, y):
                return {"prediction": (x + y, self.bff)}

        # 创建 ModuleConstant 的实例对象 mod
        mod = ModuleConstant()
        # 导出模型 mod，并传入空的输入
        ep = torch.export.export(mod, ())
        # 使用断言检查导出模型的输出与原始模型的输出是否相等
        self.assertEqual(ep.module()(), mod())

        # 创建输入参数 args
        args = (torch.randn(3, 2), torch.randn(3, 2))
        # 创建 ModuleNestedConstant 的实例对象 mod
        mod = ModuleNestedConstant()
        # 导出模型 mod，并传入输入参数 args
        ep = torch.export.export(mod, args)
        # 使用断言检查导出模型的输出与原始模型的输出是否相等
        self.assertEqual(ep.module()(*args), mod(*args))
    def test_nested_module(self):
        # 定义一个名为 M1 的简单神经网络模块，实现前向传播计算输入的加和
        class M1(torch.nn.Module):
            def forward(self, x):
                return x + x

        # 定义一个名为 M2 的神经网络模块，实现前向传播调用 M1 模块并将结果与输入相乘
        class M2(torch.nn.Module):
            def forward(self, x):
                # 创建一个 M1 的实例
                m = M1()
                # 返回 m(x) * x 的结果
                return m(x) * x

        # 定义输入数据
        inps = (torch.randn(3, 3),)
        # 导出 M2 模块并使用给定的输入
        ep = export(M2(), inps)
        # 断言导出的模块与原始模块在给定输入下计算的结果非常接近
        self.assertTrue(torch.allclose(ep.module()(*inps), M2()(*inps)))

        # 查找所有使用 torch.ops.aten.add.Tensor 的调用函数节点并筛选出添加节点
        add_nodes = [
            node
            for node in ep.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor
        ]
        # 断言找到的 add_nodes 数量为 1
        self.assertEqual(len(add_nodes), 1)
        # 获取第一个添加节点
        add_node = add_nodes[0]
        # 断言 add_node 的 nn_module_stack 元数据中模块堆栈的长度为 1
        self.assertEqual(len(add_node.meta["nn_module_stack"]), 1)
        # 断言在 nn_module_stack 的第一个值中包含字符串 "M2"
        self.assertTrue("M2" in list(add_node.meta["nn_module_stack"].values())[0][1])

        # 断言导出的图形表示与预期的内联字符串匹配
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
# 定义一个名为 graph 的函数
graph():
    # 创建名为 %x 的占位符节点，期望用户数量为 2，目标是 x
    %x : [num_users=2] = placeholder[target=x]
    # 调用 torch.ops.aten.add.Tensor 函数，将 %x 和 %x 相加，期望用户数量为 1
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %x), kwargs = {})
    # 调用 torch.ops.aten.mul.Tensor 函数，将 %add 和 %x 相乘，期望用户数量为 1
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %x), kwargs = {})
    # 返回包含 %mul 结果的元组
    return (mul,)""",
        )

        # 将 ep 解压为 unflattened 函数
        unflattened = unflatten(ep)
        # 断言 unflattened 函数应用于 inps 与 M2 类的结果在数值上相等
        self.assertTrue(torch.allclose(unflattened(*inps), M2()(*inps)))

    # 定义测试嵌套模块并带有初始化缓冲区的方法
    def test_nested_module_with_init_buffer(self):
        # 定义名为 M1 的内嵌模块
        class M1(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建大小为 3x3 的全为 1 的张量 b
                self.b = torch.ones(3, 3)

            # 前向传播方法
            def forward(self, x):
                # 返回 x 与张量 b 的和
                return x + self.b

        # 定义名为 M2 的内嵌模块
        class M2(torch.nn.Module):
            # 前向传播方法
            def forward(self, x):
                # 创建 M1 类的实例 m
                m = M1()
                # 返回 m(x) 与 x 的乘积
                return m(x) * x

        # 随机生成大小为 3x3 的张量作为输入 inps
        inps = (torch.randn(3, 3),)
        # 导出 M2 模块，并将输入 inps 应用于导出结果，将结果保存在 ep 中
        ep = export(M2(), inps)
        # 断言导出结果中模块的应用结果与 M2 类的结果在数值上相等
        self.assertTrue(torch.allclose(ep.module()(*inps), M2()(*inps)))

        # 断言导出结果中的状态字典长度为 0
        self.assertEqual(len(ep.state_dict), 0)
        # 断言导出结果中的常量长度为 0
        self.assertEqual(len(ep.constants), 0)

        # 断言内联显示的 ep.graph 与预期字符串匹配
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %x : [num_users=2] = placeholder[target=x]
    %ones : [num_users=1] = call_function[target=torch.ops.aten.ones.default](args = ([3, 3],), kwargs = {device: cpu, pin_memory: False})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %ones), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %x), kwargs = {})
    return (mul,)""",
        )

        # 将 ep 解压为 unflattened 函数
        unflattened = unflatten(ep)
        # 断言 unflattened 函数应用于 inps 与 M2 类的结果在数值上相等
        self.assertTrue(torch.allclose(unflattened(*inps), M2()(*inps)))

    # 标记测试带有常量缓冲区的嵌套模块为预期失败
    @testing.expectedFailureRetraceability  # Retracing tensor constants results in buffers
    @testing.expectedFailureTrainingIRToRunDecomp  # T193692674
    def test_nested_module_with_constant_buffer(self):
        # 定义名为 M1 的内嵌模块
        class M1(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建值为 5 的张量 b
                self.b = torch.tensor(5)

            # 前向传播方法
            def forward(self, x):
                # 返回 x 与张量 b 的和
                return x + self.b

        # 定义名为 M2 的内嵌模块
        class M2(torch.nn.Module):
            # 前向传播方法
            def forward(self, x):
                # 创建 M1 类的实例 m
                m = M1()
                # 返回 m(x) 与 x 的乘积
                return m(x) * x

        # 随机生成大小为 3x3 的张量作为输入 inps
        inps = (torch.randn(3, 3),)
        # 导出 M2 模块，并将输入 inps 应用于导出结果，将结果保存在 ep 中
        ep = export(M2(), inps)
        # 断言导出结果中模块的应用结果与 M2 类的结果在数值上相等
        self.assertTrue(torch.allclose(ep.module()(*inps), M2()(*inps)))

        # 断言导出结果中的状态字典长度为 0
        self.assertEqual(len(ep.state_dict), 0)
        # 断言导出结果中的常量长度为 1
        self.assertEqual(len(ep.constants), 1)

        # 断言内联显示的 ep.graph 与预期字符串匹配
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %c_lifted_tensor_0 : [num_users=1] = placeholder[target=c_lifted_tensor_0]
    %x : [num_users=2] = placeholder[target=x]
    %lift_fresh_copy : [num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%c_lifted_tensor_0,), kwargs = {})
    %detach : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%lift_fresh_copy,), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %detach), kwargs = {})
    # 调用 PyTorch 中的张量加法操作（aten.add.Tensor），将 %x 和 %detach 相加
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %x), kwargs = {})
    # 调用 PyTorch 中的张量乘法操作（aten.mul.Tensor），将 %add 和 %x 相乘
    return (mul,)""",
        )
        # 返回一个包含 %mul 的元组作为结果

        unflattened = unflatten(ep)
        # 将导出的模型张量展开为可用的形式

        self.assertTrue(torch.allclose(unflattened(*inps), M2()(*inps)))
        # 断言：通过比较两个 M2 实例在给定输入 inps 下的输出是否相近来验证导出的模型是否正确

    def test_nested_module_with_parameter(self):
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.ones(3, 3))
                self.b = torch.nn.Parameter(torch.tensor(5.0))

            def forward(self, x):
                return x + self.a * self.b
        # 定义一个包含参数的嵌套模块 M1，其 forward 方法对输入 x 执行一定的操作

        class M2(torch.nn.Module):
            def forward(self, x):
                m = M1()
                return m(x) * x
        # 定义一个使用 M1 的嵌套模块 M2，其 forward 方法调用 M1 并将结果乘以输入 x

        inps = (torch.randn(3, 3),)
        # 生成一个输入张量的元组

        # Strict export segfaults (Issue #128109)
        ep = torch.export.export(M2(), inps, strict=False)
        # 使用 torch.export.export 导出 M2 模型，strict=False 表示在导出过程中允许某些宽松的行为

        self.assertTrue(torch.allclose(ep.module()(*inps), M2()(*inps)))
        # 断言：通过比较导出的模型在给定输入 inps 下的输出与原始模型 M2 的输出是否相近来验证导出的模型是否正确

        self.assertEqual(len(ep.state_dict), 0)
        # 断言：导出的模型的状态字典应为空

        self.assertEqual(len(ep.constants), 1)
        # 断言：导出的模型的常量列表应包含一个元素

        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
# 定义函数 `graph`
graph():
    # 创建名为 `%c_lifted_tensor_0` 的占位符张量
    %c_lifted_tensor_0 : [num_users=1] = placeholder[target=c_lifted_tensor_0]
    # 创建名为 `%x` 的占位符张量
    %x : [num_users=2] = placeholder[target=x]
    # 创建全为 1 的张量 `%ones`，形状为 [3, 3]，在 CPU 上执行，不使用 pin_memory
    %ones : [num_users=1] = call_function[target=torch.ops.aten.ones.default](args = ([3, 3],), kwargs = {device: cpu, pin_memory: False})
    # 对 `%ones` 执行 detach 操作，生成新的张量 `%detach`
    %detach : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%ones,), kwargs = {})
    # 执行 `aten.lift_fresh_copy.default` 函数，创建新的 lifted 张量 `%lift_fresh_copy`
    %lift_fresh_copy : [num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%c_lifted_tensor_0,), kwargs = {})
    # 对 `%lift_fresh_copy` 执行 detach 操作，生成新的张量 `%detach_1`
    %detach_1 : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%lift_fresh_copy,), kwargs = {})
    # 对 `%detach_1` 执行 detach 操作，生成新的张量 `%detach_2`
    %detach_2 : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%detach_1,), kwargs = {})
    # 执行 `%detach` 和 `%detach_2` 的元素级乘法，生成新的张量 `%mul`
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%detach, %detach_2), kwargs = {})
    # 执行 `%x` 和 `%mul` 的元素级加法，生成新的张量 `%add`
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %mul), kwargs = {})
    # 执行 `%add` 和 `%x` 的元素级乘法，生成新的张量 `%mul_1`
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %x), kwargs = {})
    # 返回结果张量 `%mul_1`
    return (mul_1,)""",
    # 定义一个测试方法，用于验证运行时断言和尺寸检查
    def test_runtime_assert_with_size(self):
        # 定义一个简单的神经网络模块类
        class M(torch.nn.Module):
            # 定义模块的前向传播方法
            def forward(self, x, y):
                # 从张量 x 中获取其单个数值
                a = x.item()
                # 检查 a 是否符合尺寸要求
                torch._check_is_size(a)
                # 检查 a 是否小于等于张量 y 的大小
                torch._check(a <= y.size(0))
                # 返回 y 的前 a 个元素
                return y[:a]

        # 导出模块 M，使用静态形状，并指定动态形状的字典
        ep = export(
            M(),
            (torch.tensor(5), torch.ones(10)),
            dynamic_shapes={"x": None, "y": {0: torch.export.Dim("t")}},
        )
        # 定义输入
        inp = (torch.tensor(6), torch.randn(13))
        # 验证导出的模块执行结果与原模块执行结果是否近似相等
        self.assertTrue(torch.allclose(ep.module()(*inp), M()(*inp)))

    # TODO Retracing a module with constant attrs don't work.(T193692674)
    @testing.expectedFailureTrainingIRToRunDecomp
    # 定义一个测试方法，用于验证特定问题 113041 的处理
    def test_issue_113041(self):
        # 定义一个测试用的神经网络模块类
        class TestModule(torch.nn.Module):
            # 定义模块的初始化方法
            def __init__(self):
                super().__init__()
                # 初始化一个常量张量属性 a
                self.a = torch.tensor(1.0)

            # 定义模块的前向传播方法
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 返回输入张量 x 加上属性 a 的结果
                return x + self.a

        # 定义一个前向钩子函数，用于处理模块的输出
        def forward_hook(module: torch.nn.Module, inputs, output) -> torch.Tensor:
            # 将输出乘以2后返回
            return 2 * output

        # 创建一个顺序容器，包含 TestModule 的实例，并设为评估模式
        seq = torch.nn.Sequential(TestModule()).eval()
        # 为 seq 添加一个额外的属性 b，其值为一个张量 2
        seq.b = torch.tensor(2)
        # 注册前向钩子函数到 seq 上
        handle = seq.register_forward_hook(forward_hook)

        # 定义一个简单的神经网络模块类 M
        class M(torch.nn.Module):
            # 定义模块的初始化方法
            def __init__(self):
                super().__init__()
                # 将 seq 设置为模块 M 的一个属性
                self.seq = seq

            # 定义模块的前向传播方法
            def forward(self, x):
                # 返回 seq 对输入 x 的输出加上 seq 的属性 b 的结果
                return self.seq(x) + self.seq.b

        # 定义输入
        inp = (torch.randn(2, 8),)
        # 导出模块 M，并捕获导出错误，因为 dynamo 添加了额外的输入
        ep = export(M(), inp)

    # 定义一个测试方法，用于验证使用虚假张量输入导出模型的功能
    def test_export_with_fake_tensor_inputs(self):
        # 创建一个虚假张量模式对象
        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()

        # 定义一个简单的神经网络模块类 Model
        class Model(torch.nn.Module):
            # 定义模块的初始化方法
            def __init__(self) -> None:
                super().__init__()
                # 定义一个线性层
                self.linear = torch.nn.Linear(2, 2)

            # 定义模块的前向传播方法
            def forward(self, x):
                # 使用线性层处理输入 x
                out = self.linear(x)
                # 返回处理后的结果 out
                return out

        # 将输入张量放置在设备上
        with fake_mode, torch.device("meta"):
            x = torch.rand(5, 2, 2)
            model = Model()

            # 导出模型并使用输入 x
            exported_program = torch.export.export(model, (x,))
            export_res = exported_program.module()(x)
            exp_res = model(x)
            # 获取所有节点的元数据值为 "val" 的列表
            all_meta_val = [
                node.meta["val"]
                for node in exported_program.graph_module.graph.nodes
                if "val" in node.meta
            ]
            # 验证导出结果的尺寸与原模型结果的尺寸是否相等
            self.assertTrue(export_res.size() == exp_res.size())
            # 验证所有元数据值的设备都与输入 x 的设备一致
            self.assertTrue(all(val.device == x.device for val in all_meta_val))
            # 验证所有元数据值的虚假模式都与第一个元数据值的虚假模式一致
            self.assertTrue(
                all(val.fake_mode is all_meta_val[0].fake_mode for val in all_meta_val)
            )
            # 运行分解导出程序，并使用输入 x
            decomposed_ep = exported_program.run_decompositions()
            export_res = decomposed_ep.module()(x)
            # 验证分解导出结果的尺寸与原模型结果的尺寸是否相等
            self.assertTrue(export_res.size() == exp_res.size())
    def test_export_with_fake_tensor_inputs_on_cuda_devices(self):
        # 创建一个假的张量模式对象
        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 定义一个线性层，输入维度为2，输出维度为2
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                # 前向传播函数，对输入数据进行线性变换
                out = self.linear(x)
                return out

        # 将输入数据放置在虚拟设备上
        with fake_mode, torch.device("meta"):
            # 生成一个形状为[5, 2, 2]的随机张量作为输入数据
            x = torch.rand(5, 2, 2)
            # 创建一个模型实例
            model = Model()

        # 手动设置假设备的假设备类型为cuda:0
        x.fake_device = torch.device("cuda:0")
        # 遍历模型的所有参数，并设置它们的假设备类型为cuda:0
        for n, p in model.named_parameters():
            p.fake_device = torch.device("cuda:0")

        # 由于虚假张量在CUDA设备上与aot_autograd结合使用时存在问题，需要将所有张量的requires_grad设置为False
        x.requires_grad = False
        for n, p in model.named_parameters():
            p.requires_grad = False

        def check_device_and_fake_mode():
            # 导出模型及其输入数据
            exported_program = torch.export.export(model, (x,))
            # 在导出的模型上执行前向传播
            export_res = exported_program.module()(x)
            # 在原始模型上执行前向传播
            exp_res = model(x)
            # 提取所有节点的meta["val"]属性
            all_meta_val = [
                node.meta["val"]
                for node in exported_program.graph_module.graph.nodes
                if "val" in node.meta
            ]
            # 断言导出结果的大小与原始模型的输出大小相同
            self.assertTrue(export_res.size() == exp_res.size())
            # 断言所有meta["val"]的设备与输入数据x的设备相同
            self.assertTrue(all(val.device == x.device for val in all_meta_val))
            # 断言所有meta["val"]的fake_mode与第一个元素的fake_mode相同
            self.assertTrue(
                all(val.fake_mode is all_meta_val[0].fake_mode for val in all_meta_val)
            )

        # 调用检查设备和假设备模式的函数
        check_device_and_fake_mode()
    def test_run_decomposition_supports_user_input_mutation(self):
        # 定义一个单操作的神经网络模块
        class SingleOp(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化操作为本地批量归一化
                self.op = torch.ops.aten.native_batch_norm

            def forward(
                self,
                input,
                weight,
                bias,
                running_mean,
                running_var,
                training,
                momentum,
                eps,
                **kwargs,
            ):
                # 执行操作，调用本地批量归一化函数
                return self.op(
                    input,
                    weight,
                    bias,
                    running_mean,
                    running_var,
                    training,
                    momentum,
                    eps,
                    **kwargs,
                )

        # 创建输入数据和参数
        input = torch.randn(5, 5, 5)
        weight = torch.randn(5)
        bias = torch.randn(5)
        running_mean = torch.randn(5)
        running_var = torch.randn(5)
        training = True
        momentum = 0.5
        eps = 0.6

        # 创建单操作模块实例
        model = SingleOp()
        # 执行模型前向传播
        output = model(
            input, weight, bias, running_mean, running_var, training, momentum, eps
        )

        # 导出模型
        ep = torch.export.export(
            model,
            args=(
                input,
                weight,
                bias,
                running_mean,
                running_var,
                training,
                momentum,
                eps,
            ),
        )
        # 运行分解操作，使用给定的分解表
        ep.run_decompositions(decomp_table=torch._decomp.decomposition_table)
        # 断言导出模型的执行结果与直接执行模型的结果相等
        self.assertEqual(
            ep.module()(
                input, weight, bias, running_mean, running_var, training, momentum, eps
            ),
            output,
        )

    def test_export_graph_with_no_inputs(self):
        # 这种模式用于导出一个模型的图，初始化模型状态
        class Module(torch.nn.Module):
            def forward(self):
                # 返回随机张量
                return torch.randn(3, 4), torch.randn(3, 4)

        # 创建模型实例
        f = Module()
        # 导出模型
        ep = torch.export.export(f, ())
        # 调用模型并获取输出
        a, b = ep.module()()
        # 断言输出张量的大小
        self.assertEqual(a.size(), torch.Size([3, 4]))
        self.assertEqual(b.size(), torch.Size([3, 4]))
    def test_pad_sequence(self):
        # 定义一个继承自torch.nn.Module的内部类Module，用于测试pad_sequence方法
        class Module(torch.nn.Module):
            # 定义Module类的前向传播方法，接受输入x，直接调用torch._C._nn.pad_sequence对其进行填充
            def forward(self, x):
                return torch._C._nn.pad_sequence([x])

        # 创建Module类的实例m0
        m0 = Module()
        # 创建一个包含一个元组的输入数据inputs，元组中包含一个大小为(3, 2)的随机张量
        inputs = (torch.randn(3, 2),)
        # 使用torch.export.export方法导出模型m0，传入inputs和动态形状定义
        ep = torch.export.export(
            m0, inputs, dynamic_shapes={"x": {0: Dim("batch_size")}}
        )
        # 断言导出模型的结果与模型直接调用的结果相等
        self.assertEqual(ep.module()(*inputs), m0(*inputs))

        # 定义另一个内部类ModuleBatchFirst，与Module类相似但设置batch_first=True
        class ModuleBatchFirst(torch.nn.Module):
            def forward(self, x):
                return torch._C._nn.pad_sequence([x], batch_first=True)

        # 创建ModuleBatchFirst类的实例m1
        m1 = ModuleBatchFirst()
        # 更新输入数据inputs为一个包含一个大小为(3, 2)的随机张量的元组
        inputs = (torch.randn(3, 2),)
        # 导出模型m1，传入inputs和动态形状定义
        ep = torch.export.export(
            m1, inputs, dynamic_shapes={"x": {0: Dim("batch_size")}}
        )
        # 断言导出模型的结果与模型直接调用的结果相等
        self.assertEqual(ep.module()(*inputs), m1(*inputs))

        # 定义内部类ModuleMulti，接受多个输入x, y, z，并调用torch._C._nn.pad_sequence对它们进行填充
        class ModuleMulti(torch.nn.Module):
            def forward(self, x, y, z):
                return torch._C._nn.pad_sequence([x, y, z])

        # 创建ModuleMulti类的实例m2
        m2 = ModuleMulti()
        # 创建包含三个不同大小的随机张量的输入数据inputs的元组
        inputs = (torch.randn(5, 2), torch.randn(4, 2), torch.randn(3, 2))
        # 导出模型m2，传入inputs和动态形状定义
        ep = torch.export.export(
            m2,
            inputs,
            dynamic_shapes={
                "x": {0: Dim("batch_size")},
                "y": {0: Dim("y")},
                "z": {0: Dim("z")},
            },
        )
        # 断言导出模型的结果与模型直接调用的结果相等
        self.assertEqual(ep.module()(*inputs), m2(*inputs))

        # 定义内部类ModuleMultiBatchFirst，与ModuleMulti类相似但设置batch_first=True
        class ModuleMultiBatchFirst(torch.nn.Module):
            def forward(self, x, y, z):
                return torch._C._nn.pad_sequence([x, y, z], batch_first=True)

        # 创建ModuleMultiBatchFirst类的实例m3
        m3 = ModuleMulti()
        # 更新输入数据inputs为一个包含三个不同大小的随机张量的元组
        inputs = (torch.randn(5, 2), torch.randn(4, 2), torch.randn(3, 2))
        # 导出模型m2，传入inputs和动态形状定义
        ep = torch.export.export(
            m2,
            inputs,
            dynamic_shapes={
                "x": {0: Dim("batch_size")},
                "y": {0: Dim("y")},
                "z": {0: Dim("z")},
            },
        )
        # 断言导出模型的结果与模型直接调用的结果相等
        self.assertEqual(ep.module()(*inputs), m3(*inputs))
    def test_export_input_mutation_static_shape(self):
        # 定义一个模拟的具有变异行为的 PyTorch 模型
        class MutationModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x, y):
                # 修改输入张量 x 的形状，然后原地加上 y
                x.view(3, 2, -1).add_(y)
                return x

        # 创建测试用例的输入
        inputs = (torch.randn(12), torch.tensor(2))
        # 实例化模型
        model = MutationModel()
        # 导出模型
        ep = export(model, inputs)
        # 深拷贝输入以备后续验证使用
        inputs_export = copy.deepcopy(inputs)
        inputs_model = copy.deepcopy(inputs)
        # 断言导出模型的输出与原模型的输出一致
        self.assertEqual(ep.module()(*inputs_export), model(*inputs_model))
        # 断言模型对输入的修改在深拷贝前后一致
        self.assertEqual(inputs[0] + torch.tensor(2), inputs_model[0])
        self.assertEqual(inputs[0] + torch.tensor(2), inputs_export[0])

    def test_export_input_mutation_dynamic_shape(self):
        # 定义一个具有动态形状变异行为的 PyTorch 模型
        class MutationModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x, y):
                # 修改输入张量 x 的部分元素，原地乘以 y
                x[0].mul_(y)
                return x

        # 创建测试用例的输入
        inputs = ((torch.randn(12), torch.randn(3, 2)), 2.0)
        # 实例化模型
        model = MutationModel()
        # 导出模型，同时指定动态形状信息
        ep = torch.export.export(
            model,
            inputs,
            dynamic_shapes={"x": ({0: torch.export.Dim("dim")}, None), "y": None},
        )
        # 获取导出图的节点列表
        nodes = list(ep.graph.nodes)
        # 断言第一个节点是占位符
        self.assertEqual(nodes[0].op, "placeholder")
        # 断言占位符节点的值是张量类型
        self.assertIsInstance(nodes[0].meta["val"], torch.Tensor)
        # 断言占位符节点的形状维度是符号整数类型
        self.assertIsInstance(nodes[0].meta["val"].shape[0], torch.SymInt)

        # 深拷贝输入以备后续验证使用
        inputs_export = copy.deepcopy(inputs)
        inputs_model = copy.deepcopy(inputs)
        # 断言导出模型的输出与原模型的输出一致
        self.assertEqual(ep.module()(*inputs_export), model(*inputs_model))
        # 断言模型对输入的修改在深拷贝前后一致
        self.assertEqual(inputs[0][0] * 2.0, inputs_model[0][0])
        self.assertEqual(inputs[0][0] * 2.0, inputs_export[0][0])

    def test_export_input_mutation_bug(self):
        # 定义一个特定 Bug 的 PyTorch 模型
        class M(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 修改输入张量 x 的部分切片，原地加上 1
                x[:, :2, :] = x[:, :2, :] + 1
                return x

        # 创建测试用例的输入
        inputs = (torch.ones(4, 4, 4),)
        # 导出模型
        ep = torch.export.export(M(), inputs)
        # 实例化导出后的模型
        m = ep.module()

        # 将名称与从 aot_export 获取的占位符名称冲突
        for i, node in enumerate(m.graph.nodes):
            if node.op == "placeholder":
                node.name = f"arg0_{i + 1}"
        # 重新编译模型
        m.recompile()

        # 再次导出模型
        ep = torch.export.export(m, inputs)

        # 更新输入，以便验证
        inputs = (torch.randn(4, 4, 4),)
        # 断言导出模型的输出与原模型的输出一致
        self.assertEqual(
            ep.module()(*copy.deepcopy(inputs)), M()(*copy.deepcopy(inputs))
        )

    def test__scaled_dot_product_flash_attention(self):
        # 定义一个模块，实现缩放点积注意力机制
        class Module(torch.nn.Module):
            # 定义模块的前向传播方法
            def forward(self, q, k, v):
                # 调用 PyTorch 提供的缩放点积注意力函数
                res = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                return res[0]

        # 实例化模块
        m = Module()
        # 创建测试用例的输入
        inputs = (
            torch.randn(5, 4, 3, 2),
            torch.randn(5, 4, 3, 2),
            torch.randn(5, 4, 3, 2),
        )
        # 导出模型
        ep = export(m, inputs)
        # 断言导出模型的输出与原模型的输出一致
        self.assertEqual(ep.module()(*inputs), m(*inputs))

    @testing.expectedFailureSerDer  # symfloat nyi
    # 定义一个测试函数 test_sym_sqrt
    def test_sym_sqrt(self):
        # 导入 math 库
        import math

        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 定义前向传播函数 forward
            def forward(self, x):
                # 返回 x 除以 x.shape[0] 的结果
                return x / torch.sym_sqrt(x.shape[0])

        # 对类 M 进行导出，传入一个包含 torch.ones(16, 4) 的元组作为参数
        ep = export(M(), (torch.ones(16, 4),), dynamic_shapes={"x": {0: Dim("dim")}})
        # 调用 _ExportPassBaseDeprecatedDoNotUse 对象的方法
        _ExportPassBaseDeprecatedDoNotUse()(ep.graph_module)
        # 使用 FileCheck 检查 ep.graph_module.code 中 "torch._sym_sqrt" 出现的次数，确保恰好出现一次
        FileCheck().check_count("torch._sym_sqrt", 1, exactly=True).run(
            ep.graph_module.code
        )

    # 定义一个测试函数 test_check_specialized_int
    def test_check_specialized_int(self):
        # 定义一个继承自 torch.nn.Module 的类 SingleOp
        class SingleOp(torch.nn.Module):
            # 类初始化方法
            def __init__(self):
                super().__init__()
                # 将 torch.ops.aten.scatter_add 赋值给 self.op
                self.op = torch.ops.aten.scatter_add

            # 定义前向传播函数 forward，接收参数 t, dim, index, src 和 kwargs
            def forward(self, t, dim, index, src, **kwargs):
                # 调用 self.op 执行 scatter_add 操作，返回结果
                return self.op(t, dim, index, src, **kwargs)

        # 生成一个形状为 (10, 5) 的随机张量 t
        t = torch.randn(10, 5)
        # 维度设置为 -1
        dim = -1
        # 生成一个包含索引的张量 index，形状为 (5, 5)
        index = torch.tensor(
            [
                [2, 4, 3, 1, 0],
                [0, 2, 1, 4, 3],
                [3, 1, 4, 2, 0],
                [4, 0, 3, 1, 2],
                [3, 0, 4, 1, 2],
            ]
        )
        # 生成一个形状为 (5, 5) 的随机张量 src
        src = torch.randn(5, 5)

        # 创建 SingleOp 类的一个实例 model
        model = SingleOp()
        # 调用 model，传入 t, dim, index, src 作为参数，计算输出结果
        output = model(t, dim, index, src)

        # 对 model 进行导出，传入参数 (t, dim, index, src)
        ep = torch.export.export(model, args=(t, dim, index, src))
        # 运行分解，使用 torch._decomp.decomposition_table 作为分解表
        ep.run_decompositions(decomp_table=torch._decomp.decomposition_table)
        # 断言 ep.module()(t, dim, index, src) 的结果与 output 相等
        self.assertEqual(ep.module()(t, dim, index, src), output)
    def test_fqn(self):
        # 定义一个嵌套的子类 NestedChild，继承自 torch.nn.Module
        class NestedChild(torch.nn.Module):
            # 子类的前向传播方法，对输入 x 执行 x / x 的操作
            def forward(self, x):
                return x / x

        # 定义一个子类 Child1，继承自 torch.nn.Module
        class Child1(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建 NestedChild 类的实例并赋值给 self.nested
                self.nested = NestedChild()
                # 注册参数 "child1param"，形状为 (2, 3)，初始化为全 1 的张量
                self.register_parameter(
                    "child1param", torch.nn.Parameter(torch.ones(2, 3))
                )

            # 子类的前向传播方法
            def forward(self, x):
                # 调用 NestedChild 的 forward 方法
                x = self.nested(x)
                # 返回 x 加上 self.child1param
                return x + self.child1param

        # 定义一个子类 Child2，继承自 torch.nn.Module
        class Child2(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 注册缓冲区 "child2buffer"，形状为 (2, 3)，初始化为全 1 的张量
                self.register_buffer("child2buffer", torch.ones(2, 3))

            # 子类的前向传播方法
            def forward(self, x):
                # 返回 x 减去 self.child2buffer
                return x - self.child2buffer

        # 定义一个主模块 MyModule，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建 Child1 类的实例并赋值给 self.foo
                self.foo = Child1()
                # 创建 Child2 类的实例并赋值给 self.bar
                self.bar = Child2()
                # 注册参数 "rootparam"，形状为 (2, 3)，初始化为全 1 的张量
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3))
                )

            # 主模块的前向传播方法
            def forward(self, x):
                # x 乘以 self.rootparam
                x = x * self.rootparam
                # 调用 self.foo 的 forward 方法
                x = self.foo(x)
                # 调用 self.bar 的 forward 方法
                x = self.bar(x)
                # 返回结果 x
                return x

        # 创建原始的 MyModule 类的实例
        orig_eager = MyModule()
        # 生成测试输入，形状为 (2, 3) 的随机张量
        test_inp = torch.randn(2, 3)

        # 导出成 Torch IR 格式的模型 torch_gm
        torch_gm = _export_to_torch_ir(orig_eager, (torch.rand(2, 3),), {})
        # 遍历原始模型的状态字典，进行断言检查
        for k, v in orig_eager.state_dict().items():
            # 将点号 "." 替换为下划线 "_"，用于标准化键名
            normalized_k = k.replace(".", "_")
            # 断言标准化后的键名存在于 torch_gm 的状态字典中
            self.assertIn(normalized_k, torch_gm.state_dict())
            # 断言对应键的值相等
            self.assertEqual(v, torch_gm.state_dict()[normalized_k])
        # 断言通过 Torch IR 格式的模型 torch_gm 对测试输入 test_inp 的计算结果与原始模型一致
        self.assertTrue(torch.allclose(torch_gm(test_inp), orig_eager(test_inp)))

        # 导出成预自动求导的 Torch 脚本的模型 pre_autograd_gm
        pre_autograd_gm = torch.export._trace._export(
            orig_eager, (torch.rand(2, 3),), {}, pre_dispatch=True
        ).module()
        # 遍历原始模型的状态字典，进行断言检查
        for k, v in orig_eager.state_dict().items():
            # 断言原始键名存在于 pre_autograd_gm 的状态字典中
            self.assertIn(k, pre_autograd_gm.state_dict())
            # 断言对应键的值相等
            self.assertEqual(v, pre_autograd_gm.state_dict()[k])
        # 断言通过预自动求导的 Torch 脚本的模型 pre_autograd_gm 对测试输入 test_inp 的计算结果与原始模型一致
        self.assertTrue(torch.allclose(pre_autograd_gm(test_inp), orig_eager(test_inp)))

        # 导出成 TorchScript 的模型 ep
        ep = export(orig_eager, (torch.rand(2, 3),), {})
        # 遍历原始模型的状态字典，进行断言检查
        for k, v in orig_eager.state_dict().items():
            # 断言原始键名存在于导出模型 ep 的状态字典中
            self.assertIn(k, ep.state_dict)
            # 断言对应键的值相等
            self.assertEqual(v, ep.state_dict[k])
        # 断言通过 TorchScript 的模型 ep 对测试输入 test_inp 的计算结果与原始模型一致
        self.assertTrue(torch.allclose(ep.module()(test_inp), orig_eager(test_inp)))
    def test_nn_module_stack(self):
        # 定义 Leaf 类，继承自 torch.nn.Module，表示神经网络模块
        class Leaf(torch.nn.Module):
            # Leaf 类的初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个线性层，输入维度为4，输出维度为4
                self.linear = torch.nn.Linear(4, 4)

            # Leaf 类的前向传播方法
            def forward(self, x):
                # 将输入 x 经过线性层处理后返回结果
                return self.linear(x)

        # 定义 Bar 类，继承自 torch.nn.Module，表示神经网络模块
        class Bar(torch.nn.Module):
            # Bar 类的初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个 Leaf 类的实例
                self.leaf = Leaf()
                # 注册一个4x4大小的缓冲区，内容为随机生成的张量
                self.register_buffer("buffer", torch.randn(4, 4))

            # Bar 类的前向传播方法
            def forward(self, x):
                # 返回缓冲区张量的元素总和加上 Leaf 类对输入 x 的处理结果的元素总和
                return self.buffer.sum() + self.leaf(x).sum()

        # 定义 Foo 类，继承自 torch.nn.Module，表示神经网络模块
        class Foo(torch.nn.Module):
            # Foo 类的初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个 Bar 类的实例
                self.bar = Bar()

            # Foo 类的前向传播方法
            def forward(self, x):
                # 计算输入 x 与 Bar 类中缓冲区的和
                y = self.bar.buffer + x
                # 返回 Bar 类对输入 x 的处理结果的元素总和，并返回元组形式
                return (self.bar(x) + y.sum(),)

        # 创建一个输入张量元组，大小为4x4
        inp = (torch.randn(4, 4),)
        # 创建一个 Foo 类的实例
        mod = Foo()
        # 导出模型并运行分解，返回严格模式的输出
        ep_strict = torch.export.export(mod, inp).run_decompositions()
        # 导出模型并运行分解，返回非严格模式的输出
        ep_non_strict = torch.export.export(mod, inp, strict=False).run_decompositions()

        # 将非严格模式的输出解平铺
        gm_unflat_non_strict = unflatten(ep_non_strict)
        # 断言解平铺后的模型对象包含属性 "bar"
        self.assertTrue(hasattr(gm_unflat_non_strict, "bar"))
        # 断言解平铺后的模型对象的 "bar" 属性包含属性 "buffer"
        self.assertTrue(hasattr(gm_unflat_non_strict.bar, "buffer"))
        # 断言解平铺后的模型对象的 "bar" 属性包含属性 "leaf"
        self.assertTrue(hasattr(gm_unflat_non_strict.bar, "leaf"))

        # 将严格模式的输出解平铺
        gm_unflat_strict = unflatten(ep_strict)

        # 断言解平铺后的非严格模式和严格模式输出在给定输入下的输出值相等
        self.assertEqual(gm_unflat_non_strict(*inp), gm_unflat_strict(*inp))
        # 断言解平铺后的非严格模式的 "bar" 属性中的 "leaf" 的线性层的计算图与预期相符
        self.assertExpectedInline(
            str(gm_unflat_non_strict.bar.leaf.linear.graph).strip(),
            """\
graph():
    # 定义输入占位符 x，num_users=1 表示只有一个用户使用
    %x : [num_users=1] = placeholder[target=x]
    # 获取权重参数
    %weight : [num_users=1] = get_attr[target=weight]
    # 获取偏置参数
    %bias : [num_users=1] = get_attr[target=bias]
    # 对权重参数进行维度置换操作，将维度顺序从 [0, 1] 转换为 [1, 0]
    %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%weight, [1, 0]), kwargs = {})
    # 执行矩阵相加操作，使用偏置 %bias，输入 %x，和置换后的权重 %permute
    %addmm : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%bias, %x, %permute), kwargs = {})
    # 返回 addmm 结果
    return addmm
    return linear""",
        )
        # 调用 self.assertExpectedInline 方法，断言 gm_unflat_non_strict.bar_different.leaf.linear.graph 的字符串表示符合预期
        self.assertExpectedInline(
            # 获取 gm_unflat_non_strict.bar_different.leaf.linear.graph 的字符串表示，并去除两端的空白字符
            str(gm_unflat_non_strict.bar_different.leaf.linear.graph).strip(),
            """\

- 这段代码片段似乎是单元测试中的一部分，使用了某种自定义的断言方法 `assertExpectedInline` 来验证字符串表示的预期输出。
- 第一个参数传入了一个表达式 `str(gm_unflat_non_strict.bar_different.leaf.linear.graph).strip()`，它应该返回 `gm_unflat_non_strict.bar_different.leaf.linear.graph` 对象的字符串表示形式，并去除两端的空白字符。
- 第二个参数是预期的字符串表示形式，似乎包含了一个多行字符串的开头。
def graph():
    %add_2 : [num_users=1] = placeholder[target=add_2]
    %weight : [num_users=1] = get_attr[target=weight]
    %bias : [num_users=1] = get_attr[target=bias]
    %linear_1 : [num_users=1] = call_function[target=torch.ops.aten.linear.default](args = (%add_2, %weight, %bias), kwargs = {})
    return linear_1



# 定义一个函数 graph，用于构建计算图
def graph():
    # 定义一个占位符 add_2，并设置其属性 num_users=1
    %add_2 : [num_users=1] = placeholder[target=add_2]
    # 获取名为 weight 的属性，并设置其属性 num_users=1
    %weight : [num_users=1] = get_attr[target=weight]
    # 获取名为 bias 的属性，并设置其属性 num_users=1
    %bias : [num_users=1] = get_attr[target=bias]
    # 调用 torch.ops.aten.linear.default 函数，传入参数 %add_2, %weight, %bias，得到线性变换结果 %linear_1
    %linear_1 : [num_users=1] = call_function[target=torch.ops.aten.linear.default](args = (%add_2, %weight, %bias), kwargs = {})
    # 返回计算图中的线性变换结果
    return linear_1



# 在以下部分进行断言测试
gm_flat_non_strict = ep_non_strict.module()
gm_flat_strict = ep_strict.module()

self.assertEqual(gm_flat_non_strict(*inp), gm_flat_strict(*inp))



# 定义一个测试方法 test_stack_trace
def test_stack_trace():
    # 定义一个名为 Foo 的神经网络模型
    class Foo(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 初始化一个线性层，输入和输出维度均为 4
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x):
            # 在前向传播过程中，进行线性变换
            x = self.linear(x)
            # 将 x 乘以 2.0
            x *= 2.0
            return x

    # 导出模型 Foo，并传入一个大小为 (4, 4) 的随机张量作为输入
    ep = export(
        Foo(),
        (torch.randn(4, 4),),
    )
    # 检查堆栈跟踪中是否包含正确的行数
    trace_mul = [node for node in ep.graph.nodes if node.name == "mul"][0].meta.get(
        "stack_trace", ""
    )
    self.assertTrue(
        re.search(r"test_export.py.*in forward\n.*x \*= 2.0", trace_mul)
    )
    # 获取包含 "addmm" 或 "linear" 名称的节点，并检查其堆栈跟踪
    trace_addmm = [
        node for node in ep.graph.nodes if node.name in ["addmm", "linear"]
    ][0].meta.get("stack_trace", "")
    self.assertTrue(
        re.search(
            r"test_export.py.*in forward\n.*x = self.linear\(x\)", trace_addmm
        )
    )



@testing.expectedFailureTrainingIRToRunDecomp  # T193702033
# 定义一个测试方法 test_sym_stack_trace
def test_sym_stack_trace():
    # TODO(avik): update this test with torch._check*
    # 定义一个名为 Foo 的神经网络模型
    class Foo(torch.nn.Module):
        def forward(self, x, y):
            # 对 y 进行符号约束范围操作
            y = torch.sym_constrain_range_for_size(y.item(), min=2)
            # 判断 x 的第一个维度是否为 4，并根据结果返回 x 的形状
            z = x.shape[0] == 4
            z = torch.sym_ite(z, x.shape[0], x.shape[1])
            return z

    # 导出模型 Foo，并传入一个大小为 (4, 4) 的随机张量和一个标量 5 作为输入
    ep = export(
        Foo(),
        (torch.randn(4, 4), torch.tensor(5)),
        dynamic_shapes={"x": (Dim("dx0"), Dim("dx1")), "y": None},
    )
    # 获取包含 "sym_constrain_range_for_size" 或 "sym_constrain_range_for_size_default" 名称的节点，并检查其堆栈跟踪
    trace_constrain_range = [
        node
        for node in ep.graph.nodes
        if node.name
        in ["sym_constrain_range_for_size", "sym_constrain_range_for_size_default"]
    ][0].meta.get("stack_trace", None)
    self.assertTrue(
        re.search(
            r"in forward\n.*torch.sym_constrain_range_for_size",
            trace_constrain_range,
        )
    )
    # 定义一个测试方法，用于测试条件分支在模块堆栈中的导出
    def test_cond_with_module_stack_export_with(self):
        # 定义一个名为Bar的子类，继承自torch.nn.Module
        class Bar(torch.nn.Module):
            # 初始化方法，定义一个线性层
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            # 前向传播方法
            def forward(self, x):
                # 定义一个内部函数true_fn，用于当条件为真时执行
                def true_fn(x):
                    return self.linear(x).cos()  # 返回线性层作用后的余弦值

                # 定义一个内部函数false_fn，用于当条件为假时执行
                def false_fn(x):
                    return self.linear(x).sin()  # 返回线性层作用后的正弦值

                # 根据输入张量x的形状是否大于4来选择执行true_fn或false_fn
                return torch.cond(x.shape[0] > 4, true_fn, false_fn, [x])

        # 定义一个名为CondExport的子类，继承自torch.nn.Module
        class CondExport(torch.nn.Module):
            # 初始化方法，创建一个Bar类的实例
            def __init__(self):
                super().__init__()
                self.bar = Bar()

            # 前向传播方法
            def forward(self, x):
                # 返回输入张量x的余弦值加上通过Bar类实例处理后的结果
                return x.cos() + self.bar(x)

        # 定义一个输入张量inp，包含一个4x4的随机张量
        inp = (torch.randn(4, 4),)
        # 使用torch.export.export方法导出CondExport类的实例，输入为inp，允许非严格模式
        ep = torch.export.export(CondExport(), inp, strict=False)
        # 断言导出的模块的代码是否符合预期，去除首尾空白后进行比较
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
# 定义一个类方法 `forward`，用于模型的前向传播
def forward(self, p_bar_linear_weight, p_bar_linear_bias, x):
    # 计算输入张量 `x` 的余弦值
    cos = torch.ops.aten.cos.default(x)
    # 获取模型中的 `true_graph_0` 属性
    true_graph_0 = self.true_graph_0
    # 获取模型中的 `false_graph_0` 属性
    false_graph_0 = self.false_graph_0
    # 使用条件运算符，根据条件选择执行不同的子图（true_graph_0 或 false_graph_0），传入参数列表
    conditional = torch.ops.higher_order.cond(False, true_graph_0, false_graph_0, [p_bar_linear_bias, p_bar_linear_weight, x])
    # 清空不再需要的变量以释放内存
    true_graph_0 = false_graph_0 = p_bar_linear_bias = p_bar_linear_weight = x = None
    # 从条件运算的结果中取出第一个元素
    getitem = conditional[0]
    # 清空不再需要的变量以释放内存
    conditional = None
    # 将余弦值张量 `cos` 与获取的元素相加
    add = torch.ops.aten.add.Tensor(cos, getitem)
    # 清空不再需要的变量以释放内存
    cos = getitem = None
    # 返回加法结果作为元组的形式
    return (add,)
    def test_predispatch_cond(self):
        # 定义一个内部测试类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 构造函数，初始化模型的两个缓冲区 pred 和 t
            def __init__(self):
                super().__init__()
                self.register_buffer("pred", torch.tensor(False))
                self.register_buffer("t", torch.tensor(10))

            # 前向传播函数，接受输入 x 和 y
            def forward(self, x, y):
                # 定义一个内部函数 true_fn，计算条件为真时的结果
                def true_fn(x, y):
                    # 启用梯度计算环境
                    with torch.enable_grad():
                        return x - 1 + self.t + y

                # 使用 torch.cond 根据 self.pred 条件选择执行 true_fn 或 lambda 函数
                return torch.cond(
                    self.pred,
                    true_fn,
                    lambda x, y: x + 1 - self.t + y,
                    [x, y],  # 传递给选择函数的参数列表
                )

        # 创建 Model 类的实例
        model = Model()
        # 使用 torch.no_grad 上下文管理器，确保在推理模式下执行
        with torch.no_grad():
            # 导出模型程序的追踪版本
            exported_program = torch.export._trace._export(
                model,
                (torch.tensor(10), torch.tensor(12)),  # 输入参数
                {},  # 空的附加参数字典
                dynamic_shapes=None,
                pre_dispatch=True,  # 开启预调度
                strict=False,  # 宽松模式
            )

        # 使用 self.assertExpectedInline 进行内联代码字符串的断言比较
        self.assertExpectedInline(
            str(exported_program.graph_module.code.strip()),
            """\
    def forward(self, b_pred, b_t, x, y):
        # 获取当前对象的 true_graph_0 属性
        true_graph_0 = self.true_graph_0
        # 获取当前对象的 false_graph_0 属性
        false_graph_0 = self.false_graph_0
        # 使用 torch.ops.higher_order.cond 函数执行条件操作，选择 true_graph_0 或 false_graph_0 的计算路径
        conditional = torch.ops.higher_order.cond(b_pred, true_graph_0, false_graph_0, [b_t, x, y]);  b_pred = true_graph_0 = false_graph_0 = b_t = x = y = None
        # 获取条件操作返回的第一个元素
        getitem = conditional[0];  conditional = None
        # 返回结果元组
        return (getitem,)

    def test_predispatch_grad_wrappers(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                # 启用梯度计算
                with torch.enable_grad():
                    x = x - y
                # 禁用梯度计算
                with torch.no_grad():
                    x = x + y
                # 返回最终结果 x
                return x

        # 创建 Model 类的实例 model
        model = Model()
        # 使用 torch.export._trace._export 导出模型
        with torch.no_grad():
            ep_nograd = torch.export._trace._export(
                model,
                (torch.tensor(10), torch.tensor(12)),
                {},
                dynamic_shapes=None,
                pre_dispatch=True,
                strict=False,
            )
        # 检查是否只有 sub 操作被包装在 enable_grad 中
        getattr_nodes = [
            node for node in ep_nograd.graph.nodes if node.op == "get_attr"
        ]
        self.assertEqual(len(getattr_nodes), 1)
        # 获取包装后的子图
        grad_subgraph = getattr(ep_nograd.graph_module, getattr_nodes[0].target)
        # 获取第一个调用函数节点
        op_node = [
            node for node in grad_subgraph.graph.nodes if node.op == "call_function"
        ][0]
        # 检查操作节点是否为 "aten::sub.Tensor"
        self.assertEqual(op_node.target._name, "aten::sub.Tensor")

        # 启用梯度计算
        model = Model()
        ep_grad = torch.export._trace._export(
            model,
            (torch.tensor(10), torch.tensor(12)),
            {},
            dynamic_shapes=None,
            pre_dispatch=True,
            strict=False,
        )
        # 检查是否只有 add 操作被包装在 enable_grad 中
        getattr_nodes = [node for node in ep_grad.graph.nodes if node.op == "get_attr"]
        self.assertEqual(len(getattr_nodes), 1)
        # 获取包装后的子图
        grad_subgraph = getattr(ep_grad.graph_module, getattr_nodes[0].target)
        # 获取第一个调用函数节点
        op_node = [
            node for node in grad_subgraph.graph.nodes if node.op == "call_function"
        ][0]
        # 检查操作节点是否为 "aten::add.Tensor"
        self.assertEqual(op_node.target._name, "aten::add.Tensor")
    def test_layer_sharing(self):
        N, C, H, W = 1, 2, 2, 3

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个 LayerNorm 层，指定输入的维度为 [C, H, W]
                layer = torch.nn.LayerNorm([C, H, W])
                # 使用 ModuleList 封装多个相同的 LayerNorm 层
                self.norms = torch.nn.ModuleList(
                    [
                        layer,
                        layer,
                    ]
                )

            def forward(self, x):
                # 对每个封装的 LayerNorm 层执行前向传播
                for norm in self.norms:
                    x = norm(x)
                return x

        m = Module()
        # 深拷贝 Module 实例
        copied_m = copy.deepcopy(m)
        # 导出拷贝后的 Module 实例
        ep = export(copied_m, (torch.randn(N, C, H, W),))
        # 断言拷贝后的模型参数与原模型一致
        self.assertEqual(copied_m.state_dict(), m.state_dict())
        # 断言导出结果的状态字典与原模型一致
        self.assertEqual(ep.state_dict, m.state_dict())

    @testing.expectedFailureTrainingIRToRunDecomp  # T193692674
    def test_non_persistent_buffer(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个非持久化的 buffer，名为 "foo"，形状为 (2, 3)
                self.register_buffer("foo", torch.rand(2, 3), persistent=False)

            def forward(self, x):
                # 返回 foo 和输入张量 x 的和
                return self.foo + x

        inp = torch.rand(2, 3)
        m = MyModule()
        # 导出模型 MyModule
        ep = export(m, (inp,), {})

        # 断言导出模型对输入 inp 的计算结果与原模型一致
        self.assertEqual(ep.module()(inp), m(inp))
        # 非持久化的 buffer 不应该出现在状态字典中
        self.assertNotIn("foo", ep.state_dict)
        # 但应该出现在 named_buffers 中
        named_buffers = {name: buffer for (name, buffer) in ep.named_buffers()}
        self.assertIn("foo", named_buffers)
        # 同时也应该出现在 constants 中
        self.assertIn("foo", ep.constants)
        # constants 的长度应为 1
        self.assertEqual(len(ep.constants), 1)

        # 检查未提升的模块的相同属性
        mod = ep.module()
        self.assertNotIn("foo", mod.state_dict())
        mod_named_buffers = {name: buffer for (name, buffer) in mod.named_buffers()}
        self.assertIn("foo", mod_named_buffers)
        self.assertIn("foo", ep.constants)
        self.assertEqual(len(ep.constants), 1)
        self.assertEqual(mod(inp), m(inp))

    def test_export_as_backend(self):
        def f(x, y):
            return x + y

        def my_custom_backend(gm, example_inputs):
            # 使用自定义后端导出模型
            gm = (
                torch.export.export(gm, tuple(example_inputs), strict=False)
                .run_decompositions()
                .module()
            )
            return gm

        inp = (torch.randn(3, 3), torch.randn(3, 3))
        # 编译函数 f 使用自定义后端
        new_res = torch.compile(f, backend=my_custom_backend)(*inp)
        # 断言编译后的结果与直接调用函数 f 的结果一致
        self.assertTrue(torch.allclose(f(*inp), new_res))
    def test_nonstrict_retrace_preserves_metadata(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        inp = torch.randn(4, 4)
        m = MyModule()
        # 使用非严格模式导出模型及其计算图
        ep = torch.export.export(m, (inp,), {}, strict=False)
        # 重新追踪模型
        ep2 = torch.export.export(ep.module(), (inp,), {}, strict=False)

        # 检查两个导出的计算图节点的堆栈跟踪元数据是否相同
        for n1, n2 in zip(list(ep.graph.nodes), list(ep2.graph.nodes)):
            self.assertEqual(n1.meta.get("stack_trace"), n2.meta.get("stack_trace"))

    # TODO Retracing a module with constant attrs don't work.(T193692674)
    @testing.expectedFailureTrainingIRToRunDecomp
    def test_fake_weights(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个带有参数的线性层
                self.foo = torch.nn.Parameter(torch.randn(4, 4))
                # 注册一个非持久化的缓冲区
                self.register_buffer("bar", torch.randn(4, 4), persistent=False)
                # 注册一个持久化的缓冲区
                self.register_buffer("baz", torch.randn(4, 4), persistent=True)

            def forward(self, x):
                # 返回加权和的结果
                return self.foo + x + self.bar + self.baz

        # 使用假张量模式创建模块实例
        fake_mode = torch._subclasses.FakeTensorMode(
            shape_env=ShapeEnv(tracked_fakes=[])
        )
        with fake_mode:
            m = MyModule()
        inp = torch.randn(4, 4)
        # 导出模型及其计算图
        ep = export(m, (inp,))
        # 由于模块具有假权重，无法比较输出结果。

    def test_fake_inputs(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个带有参数的线性层
                self.foo = torch.nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                # 返回加权和的结果
                return self.foo + x

        # 使用假张量模式创建模块实例
        fake_mode = torch._subclasses.FakeTensorMode(
            shape_env=ShapeEnv(tracked_fakes=[])
        )
        m = MyModule()
        with fake_mode:
            inp = torch.randn(4, 4)

        # 导出模型及其计算图
        ep = export(m, (inp,))
        # 检查导出的模型在输入为全1张量时的输出是否与原始模型相同
        self.assertEqual(ep.module()(torch.ones(4, 4)), m(torch.ones(4, 4)))

    def test_trace_under_fake(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个带有参数的线性层
                self.foo = torch.nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                # 返回加权和的结果
                return self.foo + x

        # 使用假张量模式创建模块实例
        fake_mode = torch._subclasses.FakeTensorMode(
            shape_env=ShapeEnv(tracked_fakes=[])
        )
        with fake_mode:
            m = MyModule()
            inp = torch.randn(4, 4)
            # 不能使用未经限定的export()，因为它将尝试在新的假张量模式下反序列化。

    # Errors because non-strict is not supported in training IR (T193692164)
    @testing.expectedFailureTrainingIRToRunDecomp
    # 测试编译状态的方法
    def test_compiling_state(self):
        # 定义一个继承自 torch.nn.Module 的测试模块 TestModule1
        class TestModule1(torch.nn.Module):
            # 前向传播函数，根据是否处于编译状态返回不同的结果
            def forward(self, x):
                # 如果 Torch 的动态编译器正在编译中
                if torch._dynamo.is_compiling():
                    return x * 2  # 返回输入 x 的两倍
                else:
                    return x * 3  # 返回输入 x 的三倍

        # 定义第二个测试模块 TestModule2
        class TestModule2(torch.nn.Module):
            # 前向传播函数，根据是否处于编译状态返回不同的结果
            def forward(self, x):
                # 如果 Torch 的工具类编译器正在编译中
                if torch._utils.is_compiling():
                    return x * 2  # 返回输入 x 的两倍
                else:
                    return x * 3  # 返回输入 x 的三倍

        # 定义第三个测试模块 TestModule3
        class TestModule3(torch.nn.Module):
            # 前向传播函数，根据是否处于编译状态返回不同的结果
            def forward(self, x):
                # 如果 Torch 的编译器正在编译中
                if torch.compiler.is_compiling():
                    return x * 2  # 返回输入 x 的两倍
                else:
                    return x * 3  # 返回输入 x 的三倍

        # 遍历三个测试模块，对每一个模块执行导出操作并进行断言
        for m in [TestModule1(), TestModule2(), TestModule3()]:
            input = torch.randn(5)  # 创建一个随机输入张量
            # 严格导出模型并进行断言
            ep_strict = export(m, (input,), strict=True)
            # 非严格导出模型并进行断言
            ep_non_strict = export(m, (input,), strict=False)

            # 断言模型前向传播输出与预期结果相近
            self.assertTrue(torch.allclose(input * 3, m(input)))
            # 断言严格导出模型的前向传播输出与预期结果相近
            self.assertTrue(torch.allclose(input * 2, ep_strict.module()(input)))
            # 断言非严格导出模型的前向传播输出与预期结果相近
            self.assertTrue(torch.allclose(input * 2, ep_non_strict.module()(input)))

    # 测试用户输入和缓冲区变异的方法
    def test_user_input_and_buffer_mutation(self):
        # 定义一个继承自 torch.nn.Module 的测试模块 MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("foo", torch.randn(4, 4))  # 注册一个名为 foo 的缓冲区

            # 前向传播函数，对输入 x 和缓冲区 foo 进行变异并返回结果
            def forward(self, x):
                self.foo.add_(1)  # 对缓冲区 foo 进行原地加法操作
                x.add_(1)  # 对输入 x 进行原地加法操作
                return self.foo + x  # 返回缓冲区 foo 和输入 x 相加的结果

        mod = MyModule()  # 创建 MyModule 的一个实例
        mod_copy = copy.deepcopy(mod)  # 深拷贝模型实例
        ep = export(mod_copy, (torch.rand(4, 4),))  # 导出模型并记录结果

        # 断言模型的缓冲区 foo 与导出模型的缓冲区 foo 相等
        self.assertEqual(mod.foo, ep.module().foo)
        # 断言模型对输入张量 torch.ones(4, 4) 的前向传播输出与导出模型的相等
        self.assertEqual(mod(torch.ones(4, 4)), ep.module()(torch.ones(4, 4)))

    # 标记为预期失败的训练 IR 运行分解方法
    @testing.expectedFailureTrainingIRToRunDecomp  # T193702033
    def test_symint_tensor_return(self):
        # 定义一个继承自 torch.nn.Module 的测试模块 Module
        class Module(torch.nn.Module):
            # 前向传播函数，返回调用 Torch 操作符返回的张量的第一个元素
            def forward(self, x):
                return torch.ops.testlib.returns_tensor_symint(x)[0]

        # 调用私有方法 _test_export_same_as_eager，测试导出模型的行为是否与 eager 模式相同
        self._test_export_same_as_eager(Module(), (torch.randn(4, 4),))
    def test_custom_op_auto_functionalize(self):
        # 定义一个测试函数，用于测试自定义操作的自动功能化
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, z):
                # 模型的前向传播，调用自定义操作 'foo'
                return torch.ops.testlib.foo(x, z)

        # 创建输入数据
        inps = (torch.ones(5), torch.ones(5))
        inps_for_export = (torch.ones(5), torch.ones(5))
        inps_for_export_with_decomp = (torch.ones(5), torch.ones(5))

        # 导出模型并进行测试
        ep = torch.export.export(M(), inps_for_export)
        x_new_eager, z_new_eager, legit_eager = M()(*inps)
        x_new_export, z_new_export, legit_export = ep.module()(*inps_for_export)
        self.assertTrue(torch.allclose(x_new_eager, x_new_export))
        self.assertTrue(torch.allclose(z_new_eager, z_new_export))
        self.assertTrue(torch.allclose(legit_eager, legit_export))

        # 运行分解操作后再次进行测试
        ep = ep.run_decompositions()
        x_new_export, z_new_export, legit_export = ep.module()(
            *inps_for_export_with_decomp
        )
        self.assertTrue(torch.allclose(x_new_eager, x_new_export))
        self.assertTrue(torch.allclose(z_new_eager, z_new_export))
        self.assertTrue(torch.allclose(legit_eager, legit_export))

    def test_custom_op_auto_functionalize_pre_dispatch(self):
        # 定义另一个测试函数，用于测试带预调度的自定义操作的自动功能化
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # 模型的前向传播，调用带预调度的自定义操作 'foo_mutated'
                return torch.ops.testlib.foo_mutated(x)

        # 创建输入数据
        inps = (torch.ones(5),)

        # 导出模型并进行测试
        ep = torch.export.export(M(), inps)
        self.assertExpectedInline(
            str(ep.graph_module.code.strip()),
            """\
def forward(self, x):
    # 调用 Torch 库中的 ATen 操作，计算输入张量 x 的余弦
    cos = torch.ops.aten.cos.default(x)
    # 调用高阶操作自动功能化 auto_functionalized，使用 torch.ops.testlib.foo.default 函数，
    # 将 x 和 cos 作为参数传递，并将结果存储在 auto_functionalized 中
    auto_functionalized = torch._higher_order_ops.auto_functionalize.auto_functionalized(torch.ops.testlib.foo.default, x=x, z=cos);  x = cos = None
    # 从 auto_functionalized 中取出索引为 3 的元素，赋值给 getitem_3，并清空 auto_functionalized 引用
    getitem_3 = auto_functionalized[3];  auto_functionalized = None
    # 调用 Torch 库中的 ATen 操作，计算 getitem_3 张量的余弦
    cos_1 = torch.ops.aten.cos.default(getitem_3)
    # 返回结果元组，包含 getitem_3、getitem_3 和 cos_1
    return (getitem_3, getitem_3, cos_1)""",
        )

ep = torch.export._trace._export(M(), inps, pre_dispatch=True)
self.assertExpectedInline(
    str(ep.graph_module.code.strip()),
    """\
def forward(self, x):
    # 调用 Torch 库中的 ATen 操作，计算输入张量 x 的余弦
    cos = torch.ops.aten.cos.default(x)
    # 调用 Torch 库中的 ATen 操作，计算输入张量 x 的余弦，并赋值给 cos_1；清空 x 引用
    cos_1 = torch.ops.aten.cos.default(x);  x = None
    # 调用高阶操作自动功能化 auto_functionalized，使用 torch.ops.testlib.foo.default 函数，
    # 将 cos 和 cos_1 作为参数传递，并将结果存储在 auto_functionalized 中；清空 cos 和 cos_1 引用
    auto_functionalized = torch._higher_order_ops.auto_functionalize.auto_functionalized(torch.ops.testlib.foo.default, x=cos, z=cos_1);  cos = cos_1 = None
    # 从 auto_functionalized 中取出索引为 3 的元素，赋值给 getitem_3，并清空 auto_functionalized 引用
    getitem_3 = auto_functionalized[3];  auto_functionalized = None
    # 调用 Torch 库中的 ATen 操作，计算 getitem_3 张量的余弦，并赋值给 cos_2；清空 getitem_3 引用
    cos_2 = torch.ops.aten.cos.default(getitem_3);  getitem_3 = None
    # 返回结果元组，包含 cos_2
    return (cos_2,)""",
)

def test_custom_op_auto_warn_pre_dispatch(self):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            # 调用 Torch 库中的 testlib 自定义操作 foo_functional，传入参数 x
            return torch.ops.testlib.foo_functional(x)

    # 构造输入数据元组，包含一个元素，该元素是维度为 5 的张量，每个元素值为 1
    inps = (torch.ones(5),)

    # 导出模型 M 的图结构，并进行分解
    ep = torch.export.export(M(), inps).run_decompositions()
    self.assertExpectedInline(
        str(ep.graph_module.code.strip()),
        """\
def forward(self, x):
    # 调用 Torch 库中的 ATen 操作，计算输入张量 x 的余弦，并赋值给 foo_functional；清空 x 引用
    foo_functional = torch.ops.testlib.foo_functional.default(x);  x = None
    # 返回结果元组，包含 foo_functional
    return (foo_functional,)""",
    )

ep = torch.export._trace._export(M(), inps, pre_dispatch=True)
self.assertExpectedInline(
    str(ep.graph_module.code.strip()),
    """\
def forward(self, x):
    # 调用 Torch 库中的 ATen 操作，计算输入张量 x 的余弦，并赋值给 foo_functional；清空 x 引用
    foo_functional = torch.ops.testlib.foo_functional.default(x);  x = None
    # 返回结果元组，包含 foo_functional
    return (foo_functional,)""",
)

# original input names aren't retraceable:
# compilation will succeed, but names won't match forward() signature.
# TODO Retracing a module with constant attrs don't work.(T193692674)
@testing.expectedFailureRetraceability
@testing.expectedFailureTrainingIRToRunDecomp
    def test_placeholder_naming_collisions(self):
        # 测试嵌套用户输入之间的冲突
        class Foo(torch.nn.Module):
            def forward(self, x, x_foo, x_foo_0):
                return x["foo"][0] + x_foo[0] + x_foo_0

        # 定义输入数据
        inputs = (
            {"foo": [torch.randn(4, 4)]},  # 字典类型输入
            (torch.randn(4, 4),),          # 元组类型输入
            torch.randn(4, 4),             # 张量类型输入
        )
        # 导出模型并获取签名的输入规范
        ep = export(Foo(), inputs)
        # 期望的输入名称列表
        expected_names = ["x_foo_0", "x_foo_0_1", "x_foo_0_2"]
        # 实际的输入名称列表，从导出结果的输入规范中获取
        real_names = [spec.arg.name for spec in ep.graph_signature.input_specs]
        # 断言期望的输入名称与实际的输入名称相同
        self.assertEqual(expected_names, real_names)

        # 测试用户输入与参数、缓冲区、常量之间的冲突
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(4))  # 模型参数
                self.register_buffer("alpha", torch.randn(4), persistent=True)  # 模型缓冲区
                self.register_buffer("beta", torch.randn(4), persistent=False)  # 模型缓冲区
                self.gamma = torch.randn(4)  # 模型常量

            def forward(self, p, b_alpha, b, c_gamma):
                p = p["param"] + self.param  # 参数操作
                b = self.alpha + self.beta + b_alpha + b["beta"]  # 缓冲区操作
                c = self.gamma + c_gamma  # 常量操作
                return p, b, c

        # 定义输入数据
        inputs = (
            {"param": torch.randn(4)},  # 字典类型输入，用于参数
            torch.randn(4),              # 张量类型输入，用于常量
            {"beta": torch.randn(4)},    # 字典类型输入，用于缓冲区
            torch.randn(4),              # 张量类型输入，用于常量
        )
        # 导出模型并获取签名的输入规范
        ep = export(Foo(), inputs)
        # 期望的输入名称和种类
        expected_names = [
            ("p_param_1", InputKind.PARAMETER),         # 参数类型
            ("b_alpha_1", InputKind.BUFFER),            # 缓冲区类型
            ("b_beta_1", InputKind.BUFFER),             # 缓冲区类型
            ("c_gamma_1", InputKind.CONSTANT_TENSOR),   # 常量张量类型
            ("p_param", InputKind.USER_INPUT),          # 用户输入类型
            ("b_alpha", InputKind.USER_INPUT),          # 用户输入类型
            ("b_beta", InputKind.USER_INPUT),           # 用户输入类型
            ("c_gamma", InputKind.USER_INPUT),          # 用户输入类型
        ]
        # 实际的输入名称和种类，从导出结果的输入规范中获取
        real_names = [
            (spec.arg.name, spec.kind) for spec in ep.graph_signature.input_specs
        ]
        # 断言期望的输入名称和种类与实际的输入名称和种类相同
        self.assertEqual(expected_names, real_names)

        # 测试用户输入与调用函数节点之间的冲突
        class Foo(torch.nn.Module):
            def forward(self, mul, add, add_1):
                return mul * mul + add * add_1

        # 导出模型并获取导出结果中的节点名称和操作类型
        ep = export(Foo(), (torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 4)))
        # 期望的节点名称和操作类型列表
        expected_names_and_ops = [
            ("mul", "placeholder"),       # 占位符
            ("add", "placeholder"),       # 占位符
            ("add_1", "placeholder"),     # 占位符
            ("mul_1", "call_function"),   # 调用函数
            ("mul_2", "call_function"),   # 调用函数
            ("add_2", "call_function"),   # 调用函数
            ("output", "output"),         # 输出
        ]
        # 实际的节点名称和操作类型列表，从导出结果的图中获取
        real_names_and_ops = [(node.name, node.op) for node in ep.graph.nodes]
        # 断言期望的节点名称和操作类型与实际的节点名称和操作类型相同
        self.assertEqual(expected_names_and_ops, real_names_and_ops)
    def test_placeholder_naming_collisions_hoo_subgraphs(self):
        # 测试用户输入、顶级节点和HOO子图节点之间的命名冲突
        class Foo(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x, mul, mul_1):
                # 计算 x 的平方
                _mul = x * x
                # 使用条件语句选择不同的函数进行计算
                y = cond(
                    _mul.sum() > 0,
                    lambda x, y, z: x * y * z,
                    lambda x, y, z: x + y + z,
                    [_mul, mul, mul_1],
                )
                # 启用梯度追踪
                with torch.enable_grad():
                    y = y * y
                return y

        # 禁用梯度追踪
        with torch.no_grad():
            # 导出模型
            ep = torch.export._trace._export(
                Foo(),
                (torch.randn(4), torch.randn(4), torch.randn(4)),
                pre_dispatch=True,
            )

        # 测试条件子图
        expected_names_and_ops = [
            ("mul_2", "placeholder"),
            ("mul", "placeholder"),
            ("mul_1", "placeholder"),
            ("mul_3", "call_function"),
            ("mul_4", "call_function"),
            ("output", "output"),
        ]
        real_names_and_ops = [
            (node.name, node.op) for node in ep.graph_module.true_graph_0.graph.nodes
        ]
        self.assertEqual(expected_names_and_ops, real_names_and_ops)

        # 测试 set_grad_enabled 子图
        expected_names_and_ops = [
            ("getitem", "placeholder"),
            ("mul_1", "call_function"),
            ("output", "output"),
        ]
        real_names_and_ops = [
            (node.name, node.op) for node in ep.graph_module.submod_1.graph.nodes
        ]
        self.assertEqual(expected_names_and_ops, real_names_and_ops)

        # 测试用户输入与高阶操作子图之间的命名冲突（请不要这样做）
        class Foo(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input, true_graph, body_graph):
                # 定义映射函数
                def map_body(x, y):
                    return x + y

                # 对输入数据应用映射函数
                x = map(map_body, input, body_graph[0])
                # 对结果进行条件判断和操作
                x = x + true_graph[0] + true_graph[1]
                x = cond(x.sum() > 0, lambda x: x * 2.0, lambda x: x + 2.0, [x])
                x = cond(x.sum() > 0, lambda x: x * 2.0, lambda x: x + 2.0, [x])
                return x

        # 准备输入数据
        inputs = (
            torch.randn(10, 4),
            (torch.randn(4), torch.randn(4)),
            (torch.randn(4),),
        )
        # 导出模型
        ep = export(Foo(), inputs)

        # 检查预期的 getattr 名称
        expected_getattr_names = [
            "body_graph_1",
            "true_graph_2",
            "false_graph_0",
            "true_graph_3",
            "false_graph_1",
        ]
        real_getattr_names = [
            node.name for node in ep.graph.nodes if node.op == "get_attr"
        ]
        self.assertEqual(expected_getattr_names, real_getattr_names)
    def test_constant_input_naming(self):
        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            # 定义 forward 方法，接受 x, y 两个参数和一个可选参数 div，默认为 "floor"
            def forward(self, x, y, div="floor"):
                # 使用 torch.div 方法对 x 和 y 进行除法运算，指定 rounding_mode 为 div
                return torch.div(x, y, rounding_mode=div)

        # 创建 Foo 类的实例 f
        f = Foo()
        # 创建输入元组 inputs，包括两个 torch 张量和一个字符串 "floor"
        inputs = (torch.randn(4), torch.randn(4), "floor")
        # 调用 export 函数，导出模型 f 的输出结果 ep
        ep = export(f, inputs)
        # 获取导出结果 ep 中的 graph_signature 属性的 input_specs 列表中的第三个元素 div_spec
        div_spec = ep.graph_signature.input_specs[2]
        # 使用 self.assertEqual 方法断言 div_spec 中的参数名为 "div"
        self.assertEqual(div_spec.arg.name, "div")
        # 使用 self.assertEqual 方法断言 div_spec 中的参数值为 "floor"
        self.assertEqual(div_spec.arg.value, "floor")

    def test_unbacked_deferred_runtime_retrace(self):
        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            # 定义 forward 方法，接受 x, y 两个参数
            def forward(self, x, y):
                # 对 y 执行 sin 操作，然后计算其和
                y_sum = y.sin().sum()
                # 使用 torch.no_grad() 上下文管理器，执行下列操作时不追踪梯度
                with torch.no_grad():
                    # 获取张量 x 的标量值
                    a = x.item()
                    # 检查 a 的尺寸是否合法
                    torch._check_is_size(a)
                    # 检查 a 是否大于 2
                    torch._check(a > 2)
                    # 检查 a 是否小于 6
                    torch._check(a < 6)
                    # 调用 torch.ops.testlib.foo_unbacked(a)，获取 unbacked_shape
                    unbacked_shape = torch.ops.testlib.foo_unbacked(a)
                # 返回 y + y_sum + unbacked_shape.sum() 的结果
                return y + y_sum + unbacked_shape.sum()

        # 创建输入元组 inps，包括一个整数张量和一个随机生成的 5x5 浮点数张量
        inps = (torch.tensor(4), torch.randn(5, 5))
        # 从 torch.export 模块导入 _trace
        from torch.export import _trace
        # 使用 _trace._export 方法，对 Foo 类进行追踪导出，获取导出结果 ep_pre
        ep_pre = _trace._export(Foo(), inps, pre_dispatch=True, strict=False)
        # 使用 self.assertExpectedInline 方法断言 ep_pre.graph_module.submod_1.code 的字符串表示
        self.assertExpectedInline(
            str(ep_pre.graph_module.submod_1.code).strip(),
            """\
def forward(self, x):
    # 调用 Torch 的 ATen 操作，获取张量 x 的标量值
    item = torch.ops.aten.item.default(x);  x = None
    # 对获取的标量值进行范围约束操作
    sym_constrain_range_for_size_default = torch.ops.aten.sym_constrain_range_for_size.default(item)
    # 对获取的标量值进行指定范围内的约束操作
    sym_constrain_range_default = torch.ops.aten.sym_constrain_range.default(item, min = 3, max = 5)
    # 检查标量值是否大于等于 0
    ge = item >= 0
    # 运行时断言，确保表达式 item >= 0 成立
    _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression 0 <= u1 on node 'ge'");  ge = None
    # 检查标量值是否大于 2
    gt = item > 2
    # 运行时断言，确保表达式 item > 2 成立
    _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(gt, "Runtime assertion failed for expression 2 < u1 on node 'gt'");  gt = None
    # 检查标量值是否小于 6
    lt = item < 6
    # 运行时断言，确保表达式 item < 6 成立
    _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(lt, "Runtime assertion failed for expression u1 < 6 on node 'lt'");  lt = None
    # 调用 Torch 的 testlib 模块中的 foo_unbacked 操作，处理标量值
    foo_unbacked = torch.ops.testlib.foo_unbacked.default(item);  item = None
    # 返回处理后的结果
    return foo_unbacked""",
        )
    # 运行预处理步骤
    ep_aot = ep_pre.run_decompositions()
    # 断言预期的内联代码
    self.assertExpectedInline(
        str(ep_aot.graph_module.code).strip(),
        """\
def forward(self, x, y):
    # 调用 Torch 的 ATen 操作，对张量 y 执行正弦计算
    sin = torch.ops.aten.sin.default(y)
    # 对正弦计算的结果执行维度为 [] 的求和操作
    sum_1 = torch.ops.aten.sum.dim_IntList(sin, []);  sin = None
    # 对输入张量 x 执行本地密集标量操作
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x);  x = None
    # 对本地密集标量进行大小约束操作
    sym_constrain_range_for_size = torch.ops.aten.sym_constrain_range_for_size.default(_local_scalar_dense)
    # 对本地密集标量进行指定范围内的约束操作
    sym_constrain_range = torch.ops.aten.sym_constrain_range.default(_local_scalar_dense, min = 3, max = 5)
    # 检查本地密集标量是否大于等于 0
    ge = _local_scalar_dense >= 0
    # 运行时断言，确保表达式 _local_scalar_dense >= 0 成立
    _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression 0 <= u1 on node 'ge'");  ge = None
    # 检查本地密集标量是否大于 2
    gt = _local_scalar_dense > 2
    # 运行时断言，确保表达式 _local_scalar_dense > 2 成立
    _assert_scalar_1 = torch.ops.aten._assert_scalar.default(gt, "Runtime assertion failed for expression 2 < u1 on node 'gt'");  gt = None
    # 检查本地密集标量是否小于 6;  清除 _local_scalar_dense 变量
    lt = _local_scalar_dense < 6;  _local_scalar_dense = None
    # 运行时断言，确保表达式 u1 < 6 成立
    _assert_scalar_2 = torch.ops.aten._assert_scalar.default(lt, "Runtime assertion failed for expression u1 < 6 on node 'lt'");  lt = None
    # 调用 Torch 的 ATen 操作，创建指定形状和数值的张量
    full = torch.ops.aten.full.default([4, 4], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    # 调用 Torch 的 ATen 操作，计算张量 y 与 sum_1 的元素级加法
    add = torch.ops.aten.add.Tensor(y, sum_1);  y = sum_1 = None
    # 对指定张量执行维度为 [] 的求和操作
    sum_2 = torch.ops.aten.sum.dim_IntList(full, []);  full = None
    # 调用 Torch 的 ATen 操作，计算张量 add 与 sum_2 的元素级加法
    add_1 = torch.ops.aten.add.Tensor(add, sum_2);  add = sum_2 = None
    # 返回结果元组
    return (add_1,)""",
        )
    # 定义一个测试方法，用于测试嵌套动态形状规范
    def test_nested_dynamic_shapes_spec(self):
        # 定义一个名为Foo的内部类，继承自torch.nn.Module
        class Foo(torch.nn.Module):
            # 重写forward方法，接受参数x
            def forward(self, x):
                # 解构x变量，分别赋值给(a0, a1), (b0, b1), (c0, c1, c2)
                (a0, a1), (b0, b1), (c0, c1, c2) = x
                # 返回a0, a1, b0, b1, c0, c1, c2的总和
                return a0 + a1 + b0 + b1 + c0 + c1 + c2

        # 创建Foo类的实例对象f
        f = Foo()
        # 定义inputs变量，包含三个元组作为不同层级的输入数据
        inputs = (
            (1, 2),
            (
                torch.randn(4, 4),
                torch.randn(4, 4),
            ),
            (
                torch.randn(4, 4),
                torch.randn(4, 4),
                torch.randn(4, 4),
            ),
        )
        # 定义动态形状规范dynamic_shapes，用于描述输入数据的形状特性
        dynamic_shapes = {
            "x": (
                (None, None),
                (None, None),
                (None, None, None),
            )
        }
        # 调用export函数，导出模型f的输入inputs，同时传递动态形状规范dynamic_shapes
        export(f, (inputs,), dynamic_shapes=dynamic_shapes)

    # 定义一个测试方法，用于验证禁用强制特化时的错误处理
    def test_disable_forced_specializations_errors(self):
        # 定义一个名为Foo的内部类，继承自torch.nn.Module
        class Foo(torch.nn.Module):
            # 重写forward方法，接受四个参数w, x, y, z
            def forward(self, w, x, y, z):
                # 返回w重塑后的形状与x相加，以及y与z的和
                return w.reshape([-1]) + x, y + z  # simple: s0*s1 = s2, s3 = s4

        # 定义输入inputs，包含四个张量
        inputs = (
            torch.randn(3, 4),
            torch.randn(12),
            torch.randn(4),
            torch.randn(4),
        )
        # 定义动态形状规范dynamic_shapes，描述各个输入的形状需求
        dynamic_shapes = {
            "w": [Dim(f"dw{i}") for i in range(2)],
            "x": [Dim(f"dx{i}") for i in range(1)],
            "y": [Dim("dy")],  # y & z incorrect, export is supposed to fail.
            "z": [Dim("dz")],  # suggested fix should be to match these up.
        }
        # 使用assertRaisesRegex断言，检查是否抛出特定异常及其包含的错误信息
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r".*Specializations unexpectedly required(.*\n)*"
            r"Suggested fixes:(.*\n)*"
            r".*dw0 = 3(.*\n)*"
            r".*dw1 = 4(.*\n)*"
            r".*dx0 = 12(.*\n)*"
            r".*dz = dy(.*\n)*",
        ):
            # 调用_export函数，导出模型Foo的输入inputs，并传递动态形状规范dynamic_shapes
            torch.export._trace._export(
                Foo(),
                inputs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
                _disable_forced_specializations=False,
            )
        # 使用assertRaisesRegex再次断言，检查是否抛出特定异常及其包含的错误信息
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r".*Constraints violated(.*\n)*"
            r"Suggested fixes:(.*\n)*"
            r".*dz = dy(.*\n)*",
        ) as msg:
            # 调用_export函数，导出模型Foo的输入inputs，并传递动态形状规范dynamic_shapes
            torch.export._trace._export(
                Foo(),
                inputs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
                _disable_forced_specializations=True,
            )
    def test_preserve_requires_grad_placeholders(self):
        # 定义一个继承自 torch.nn.Module 的模型类 Module
        class Module(torch.nn.Module):
            # 模型初始化函数
            def __init__(self):
                super().__init__()
                # 创建一个参数 p，其值为 3x3 的随机张量
                self.p = torch.nn.Parameter(torch.randn(3, 3))

            # 前向传播函数，接受两个输入 x 和 y
            def forward(self, x, y):
                # 返回参数 p 加上输入 x 和 y 的结果
                return self.p + x + y

        # 创建 Module 的实例 m
        m = Module()
        # 导出模型 m，同时传入两个张量作为输入，其中第二个张量要求梯度计算
        ep = export(m, (torch.randn(3, 3), torch.randn(3, 3, requires_grad=True)))
        
        # 获取所有占位符节点，这些节点在导出的图中表示为 "placeholder"
        placeholders = [
            node for node in ep.graph_module.graph.nodes if node.op == "placeholder"
        ]
        
        # 断言第一个占位符节点的元数据 "val" 要求梯度计算
        self.assertTrue(placeholders[0].meta["val"].requires_grad)
        # 断言第二个占位符节点的元数据 "val" 不要求梯度计算
        self.assertFalse(placeholders[1].meta["val"].requires_grad)
        # 断言第三个占位符节点的元数据 "val" 要求梯度计算
        self.assertTrue(placeholders[2].meta["val"].requires_grad)

    def test_reshape_view_helper(self):
        # 见：https://github.com/pytorch/pytorch/issues/126607
        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            # 模型初始化函数
            def __init__(self):
                super().__init__()

            # 前向传播函数，接受输入 x
            def forward(self, x):
                # 将输入 x 进行形状变换，将其第二维展开成 -1
                x = x.view(x.size(1), -1)
                # 调用 torch/_refs/__init__/_reshape_view_helper() 在重塑内核上生成守卫条件
                # Ne(s0, 20)，以确保重塑不是空操作
                # Ne(Mod(s0, 20), 0)，以确保首先将 [s0, 20, 16] 展平为 [s0*20, 16]
                # 然后分割维度 -> [20, s0, 16]
                # 检查这些条件是否出现在图中
                return torch.nn.functional.softmax(
                    x, dim=0
                )  # 不认为 softmax 实际上会产生任何问题，只是原始测试的一部分

        # 创建 Model 的实例 model
        model = Model()
        # 创建一个形状为 [1024, 20, 16] 的随机张量 x
        x = torch.rand(1024, 20, 16)
        # 定义动态形状字典 dynamic_shapes
        dynamic_shapes = {"x": {0: Dim("batch")}}
        # 导出模型 model，同时传入张量 x 和动态形状信息 dynamic_shapes
        ep = torch.export._trace._export(
            model,
            (x,),
            dynamic_shapes=dynamic_shapes,
            _allow_complex_guards_as_runtime_asserts=True,
        )
        
        # 使用断言检查运行时错误异常信息，确保第一次调用时出现 Ne(s0, 20) 的错误
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Ne\(s0, 20\)",
        ):
            ep.module()(torch.randn(20, 20, 16))
        
        # 使用断言检查运行时错误异常信息，确保第二次调用时出现 Ne(Mod(s0, 20), 0) 的错误
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Ne\(Mod\(s0, 20\), 0\)",
        ):
            ep.module()(torch.randn(400, 20, 16))
        
        # 调用导出模型的结果，传入形状为 [42, 20, 16] 的随机张量
        ep.module()(torch.randn(42, 20, 16))
    def test_allow_explicit_guards_as_runtime_asserts(self):
        # 检查显式保护条件是否被视为运行时断言

        class Foo(torch.nn.Module):
            def forward(self, x, y):
                # 检查对第一个保护条件的否定也显示为运行时断言
                if x.shape[0] == y.shape[0]:  # False
                    return x + y
                elif x.shape[0] == y.shape[0] ** 3:  # False
                    return x + 2, y + 3
                elif x.shape[0] ** 2 == y.shape[0] * 3:  # True
                    return x * 2.0, y * 3.0

        inputs = (torch.randn(6), torch.randn(12))
        dynamic_shapes = {"x": [Dim("dx", min=4)], "y": [Dim("dy", min=4)]}
        
        # 使用 _export 方法导出模型，设置了动态形状和允许复杂保护条件作为运行时断言
        ep = torch.export._trace._export(
            Foo(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            _allow_complex_guards_as_runtime_asserts=True,
        )
        
        # 检查正向传播
        out0, out1 = ep.module()(torch.randn(9), torch.randn(27))
        self.assertEqual(out0.shape, torch.ones(9).shape)
        self.assertEqual(out1.shape, torch.ones(27).shape)
        
        # 使用 assertRaisesRegex 检查运行时断言失败的情况
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Ne\(s0, s1\)",
        ):  # 只在运行时失败
            ep.module()(torch.randn(4), torch.randn(4))  # 失败
        
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Ne\(s0, s1\**3\)",
        ):
            ep.module()(torch.randn(64), torch.randn(4))  # 失败
        
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Eq\(s0\**2, 3\*s1\)",
        ):
            ep.module()(torch.randn(10), torch.randn(9))  # 失败
        
        # 应该使用命令行标志 TORCH_DYNAMO_DO_NOT_EMIT_RUNTIME_ASSERTS=1 设置此项，
        # 但 dynamo 在 torch 导入时检查这一设置，因此设置 os.environ 不会有任何影响
        # 相反，手动修补 dynamo 配置并测试。
        
        from torch._dynamo import config as _dynamo_config
        
        # 测试设置此标志是否移除运行时断言
        with _dynamo_config.patch(
            do_not_emit_runtime_asserts=True,
        ):
            ep = torch.export._trace._export(
                Foo(),
                inputs,
                dynamic_shapes=dynamic_shapes,
                _allow_complex_guards_as_runtime_asserts=True,
            ).run_decompositions()

        # 检查图中是否不再有运行时断言节点
        self.assertEqual(
            [
                node.target == torch.ops.aten._assert_scalar.default
                for node in ep.graph.nodes
            ].count(True),
            0,
        )
    # 定义一个测试用例，用于测试常量别名化的情况
    def test_constant_aliasing(self):
        # 定义一个继承自 torch.nn.Module 的类 M1
        class M1(torch.nn.Module):
            # 初始化方法，接受 m2 和 foo 两个参数
            def __init__(self, m2, foo):
                super().__init__()
                # 将参数 m2 和 foo 分别赋值给实例属性 self.m2 和 self.foo
                self.m2 = m2
                self.foo = foo

            # 前向传播方法
            def forward(self, x):
                # 返回输入 x、实例属性 foo 和实例属性 m2(x) 的和
                return x + self.foo + self.m2(x)

        # 定义一个继承自 torch.nn.Module 的类 M2
        class M2(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个大小为 (3, 3) 的全为 1 的张量，并将其赋值给实例属性 self.foo
                self.foo = torch.ones(3, 3)

            # 前向传播方法
            def forward(self, x):
                # 返回输入 x 和实例属性 foo 的和
                return x + self.foo

        # 创建 M2 类的实例 m2
        m2 = M2()
        # 创建 M1 类的实例 m1，传入 m2 和 m2.foo 作为参数
        m1 = M1(m2, m2.foo)
        # 创建输入数据，一个大小为 (3, 3) 的全为 1 的张量
        inps = (torch.ones(3, 3),)
        # 使用 torch.export.export 函数导出模型 m1，并传入输入数据 inps，关闭严格模式
        ep = torch.export.export(m1, inps, strict=False)
        
        # 断言：检查导出结果中的常量列表是否包含 "foo" 和 "m2.foo"，并排序后进行比较
        self.assertEqual(sorted(list(ep.constants)), ["foo", "m2.foo"])
        
        # 统计输入签名中类型为 CONSTANT_TENSOR 的数量
        num_constant_inputs = [
            spec.kind == InputKind.CONSTANT_TENSOR
            for spec in ep.graph_signature.input_specs
        ].count(True)
        # 断言：检查 CONSTANT_TENSOR 的输入数量是否为 1
        self.assertEqual(num_constant_inputs, 1)
        
        # 对导出结果进行解扁平化
        unflattened = unflatten(ep)
        # 断言：检查模型 m1 在输入数据 inps 上的输出与解扁平化结果在相同输入数据上的输出是否近似相等
        self.assertTrue(torch.allclose(m1(*inps), unflattened(*inps)))

    # 标记为预期失败的测试用例，用于测试未使用的别名情况
    @testing.expectedFailureRetraceability
    def test_unused_aliases(self):
        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个大小为 4 的随机张量，并将其封装为参数 self.alpha
                self.alpha = torch.nn.Parameter(torch.randn(4))
                # 将 self.alpha 分别赋值给 self.beta 和 self.gamma，作为参数的别名
                self.beta = self.alpha
                self.gamma = self.alpha

            # 前向传播方法
            def forward(self, x):
                # 返回输入 x 和参数 gamma 的和
                return x + self.gamma

        # 创建输入数据，一个大小为 4 的随机张量
        inps = (torch.randn(4),)
        # 使用 export 函数导出 Foo 类的实例，并传入输入数据 inps
        ep = export(Foo(), inps)
        
        # 注释：在严格模式下，占位符节点会被去重，但是仍然检查所有参数是否出现在状态字典中
        # 遍历参数列表 ["alpha", "beta", "gamma"]，检查它们是否都出现在导出结果的状态字典中
        for param in ["alpha", "beta", "gamma"]:
            self.assertTrue(param in ep.state_dict)

        # 将导出结果解扁平化
        unep = unflatten(ep)
        # 再次遍历参数列表 ["alpha", "beta", "gamma"]，检查它们是否都出现在解扁平化后的状态字典中
        for param in ["alpha", "beta", "gamma"]:
            self.assertTrue(param in unep.state_dict())
# 如果 torchdynamo 不支持当前环境，跳过执行该测试用例
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
# 定义 TestOneOffModelExportResult 类，继承自 TestCase 类
class TestOneOffModelExportResult(TestCase):

    # 定义测试函数 test_scaled_dot_product_attention_cpu
    def test_scaled_dot_product_attention_cpu(self):
        """
        This test makes sure we are always getting the same decomposition result for SDPA.
        As of now _scaled_dot_product_flash_attention_for_cpu is expected to show up in
        export() result. Some downstream backend then further decompose it into core ATen
        ops in torch/_decomp/decompositions.py (search for
        _scaled_dot_product_flash_attention_for_cpu).

        Export is decomposing based on the CompositeImplicitAutograd kernel implementation
        of SDPA. If this test fails, it means the kernel is being modified. In this case
        we strongly encourage you to change the decomposition rule under
        torch/_decomp/decompositions.py along with the kernel changes, so all of the
        downstream backends are not being affected.
        """

        # 定义 ScaledDotProductAttention 类，继承自 torch.nn.Module
        class ScaledDotProductAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 定义 forward 方法，实现前向传播
            def forward(self, q, k, v):
                # 调用 F.scaled_dot_product_attention 函数进行注意力计算
                attn_output = F.scaled_dot_product_attention(
                    q, k, v, None, dropout_p=0.0, is_causal=True
                )
                return attn_output

        # 创建 CPU 设备上的随机张量 q, k, v
        q = torch.randn(1, 1, 8, 8, device="cpu")
        k = torch.randn(1, 1, 8, 8, device="cpu")
        v = torch.randn(1, 1, 8, 8, device="cpu")

        # 导入 torch.nn.attention.SDPBackend
        from torch.nn.attention import SDPBackend

        # 使用 sdpa_kernel 上下文管理器，指定 SDPBackend.MATH 作为内核类型
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]):
            # 导出 ScaledDotProductAttention 模型的计算图 ep
            ep = torch.export.export(ScaledDotProductAttention(), (q, k, v))
            # 打印导出的计算图 ep 的结构
            print(ep.graph)
            # 对导出的计算图 ep 进行分解
            ep.run_decompositions()
            # 再次打印分解后的计算图 ep 的结构
            print(ep.graph)

    # 下面是已注释的断言语句，用于验证导出的计算图是否符合预期
    # self.assertExpectedInline(ep.graph_module.code.strip(), """\
    # def forward(self, arg0_1, arg1_1, arg2_1):
    #     _scaled_dot_product_flash_attention_for_cpu = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(arg0_1, arg1_1, arg2_1, 0.0, True);  arg0_1 = arg1_1 = arg2_1 = None
    #     getitem = _scaled_dot_product_flash_attention_for_cpu[0];  _scaled_dot_product_flash_attention_for_cpu = None
    #     return (getitem,)""")

    # 如果当前平台不支持 FLASH 注意力机制，则跳过执行该测试用例
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Can't run fused SDPA on this platform",
    )
        """
        This test makes sure we are always getting the same decomposition result for SDPA.
        As of now _scaled_dot_product_flash_attention is expected to show up in
        export() result (GPU tensors are given). Currently there's no downstream
        backend relies on this export result so if this test fails, feel free to
        change it to the latest export() result.
        """
        # 定义一个名为 test_scaled_dot_product_attention_cuda 的测试方法，用于验证 SDPA 的分解结果是否一致。
        # 目前 _scaled_dot_product_flash_attention 预期出现在 export() 的结果中（给定 GPU 张量）。
        # 目前没有下游后端依赖于这个导出结果，所以如果测试失败，可以自由地将其更改为最新的 export() 结果。

        class ScaledDotProductAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, q, k, v):
                # 调用 F 模块中的 scaled_dot_product_attention 函数进行注意力计算
                attn_output = F.scaled_dot_product_attention(
                    q, k, v, None, dropout_p=0.0, is_causal=True
                )
                return attn_output

        # 生成随机张量 q, k, v，用于输入到 ScaledDotProductAttention 模块中进行前向传播
        q = torch.randn(1, 16, 16, 64, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 16, 16, 64, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, 16, 16, 64, dtype=torch.bfloat16, device="cuda")

        # 调用 torch.export.export 方法导出 ScaledDotProductAttention 模块，并运行分解操作
        ep = torch.export.export(
            ScaledDotProductAttention(), (q, k, v)
        ).run_decompositions()

        # 使用 self.assertExpectedInline 方法验证生成的图模块的代码是否符合预期
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
# 定义一个名为 `forward` 的方法，用于模型前向传播
def forward(self, q, k, v):
    # 使用 PyTorch 的底层运算接口执行缩放点积注意力机制
    _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default(q, k, v, 0.0, True, scale = 0.125);  q = k = v = None
    # 从返回的元组中获取第一个元素
    getitem = _scaled_dot_product_flash_attention[0];  _scaled_dot_product_flash_attention = None
    # 返回元组，其中包含从 `_scaled_dot_product_flash_attention` 获取的元素
    return (getitem,)
    def test_unbacked_sdpa(self):
        import torch  # 导入 torch 库
        from torch.nn.attention import sdpa_kernel, SDPBackend  # 从 torch.nn.attention 模块导入 sdpa_kernel 和 SDPBackend
        from torch.nn.functional import scaled_dot_product_attention  # 从 torch.nn.functional 导入 scaled_dot_product_attention 函数

        class Module(torch.nn.Module):
            def forward(
                self, query: torch.Tensor, cache: torch.Tensor, start_pos: torch.Tensor
            ) -> torch.Tensor:
                # x.sizes(): 1, 128, 16, 128
                sp = start_pos.item()  # 获取 start_pos 的标量值
                torch._check_is_size(sp)  # 检查 sp 是否是合法的尺寸
                torch._check(sp >= 0)  # 检查 sp 是否大于等于 0
                torch._check(sp <= 126)  # 检查 sp 是否小于等于 126
                key = cache[:, : sp + 1, :, :]  # 从 cache 中选择部分数据作为 key，维度为 1, sp+1, 16, 128
                value = cache[:, : sp + 1, :, :]  # 从 cache 中选择部分数据作为 value，维度为 1, sp+1, 16, 128
                query = query.transpose(1, 2)  # 调整 query 的维度顺序为 (bs, n_local_heads, seqlen, head_dim)
                key = key.transpose(1, 2)  # 调整 key 的维度顺序为 (bs, n_local_heads, seqlen, head_dim)
                value = value.transpose(1, 2)  # 调整 value 的维度顺序为 (bs, n_local_heads, seqlen, head_dim)
                # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/attention.cpp#L732
                return scaled_dot_product_attention(query, key, value)  # 使用 scaled_dot_product_attention 计算注意力结果

        cache = torch.randn(1, 128, 16, 128, dtype=torch.float16)  # 生成随机的 cache 数据，维度为 1, 128, 16, 128，数据类型为 torch.float16
        query = torch.randn(1, 1, 16, 128, dtype=torch.float16)  # 生成随机的 query 数据，维度为 1, 1, 16, 128，数据类型为 torch.float16
        start_pos = torch.tensor([0])  # 创建张量 start_pos，包含单个元素 0
        with sdpa_kernel(SDPBackend.MATH), torch.no_grad():  # 使用 sdpa_kernel 和 SDPBackend.MATH 进行上下文管理，禁用梯度计算
            ep = torch.export.export(Module(), (query, cache, start_pos))  # 导出模块 Module 的计算图
            args = (query, cache, start_pos)  # 定义输入参数 args
            self.assertEqual(ep.module()(*args), Module()(*args))  # 断言导出模块的计算结果与原始模块的计算结果相等
            args = (query, cache, torch.tensor([3]))  # 定义新的输入参数 args，修改 start_pos 的值为 3
            self.assertEqual(ep.module()(*args), Module()(*args))  # 断言导出模块的计算结果与原始模块的计算结果相等
            args = (query, cache, torch.tensor([126]))  # 定义新的输入参数 args，修改 start_pos 的值为 126
            self.assertEqual(ep.module()(*args), Module()(*args))  # 断言导出模块的计算结果与原始模块的计算结果相等

    def test_none_input_output(self):
        class Z(torch.nn.Module):
            def forward(self, x, y):
                return x * x  # 返回输入 x 的平方

        ep = torch.export.export(Z(), (torch.tensor(3), None))  # 导出模块 Z 的计算图，其中 y 为空
        res = ep.module()(torch.tensor(4), None)  # 执行导出模块的计算图，输入 x 为 4，y 为空
        self.assertEqual(res, torch.tensor(16))  # 断言计算结果为 16

        class B(torch.nn.Module):
            def forward(self, x, y):
                return x * x, y  # 返回输入 x 的平方和输入 y

        ep = torch.export.export(B(), (torch.tensor(3), None))  # 导出模块 B 的计算图，其中 y 为空
        res = ep.module()(torch.tensor(4), None)  # 执行导出模块的计算图，输入 x 为 4，y 为空
        self.assertEqual(res[0], torch.tensor(16))  # 断言计算结果的第一个元素为 16
        self.assertEqual(res[1], None)  # 断言计算结果的第二个元素为空

        decomp = ep.run_decompositions()  # 运行导出对象的分解
        gm = decomp.module()  # 获取分解后的模块
        res = gm(torch.tensor(4), None)  # 执行分解后的模块，输入 x 为 4，y 为空
        self.assertEqual(res[0], torch.tensor(16))  # 断言计算结果的第一个元素为 16
        self.assertEqual(res[1], None)  # 断言计算结果的第二个元素为空

    def test_print(self):
        class M(torch.nn.Module):
            def forward(self, x):
                print("start")  # 打印输出 "start"
                x1 = x + x  # 计算 x 的加法操作
                print(x1)  # 打印输出 x1 的值
                x2 = x1 * x1  # 计算 x1 的平方
                print(1, 2, 3)  # 打印输出 1, 2, 3
                x3 = x2 + x2  # 计算 x2 的加法操作
                return (x1, x3)  # 返回结果元组 (x1, x3)

        gm = export(M(), (torch.randn(3, 3),)).graph_module  # 导出模块 M 的计算图
        self.assertExpectedInline(
            gm.code.strip(),
            """\
    def forward(self, x):
        # 调用 torch 的底层操作 aten.add.Tensor，对输入 x 执行加法运算并将结果存入 add 变量
        add = torch.ops.aten.add.Tensor(x, x);  x = None
        # 使用 add 变量进行乘法运算，将结果存入 mul 变量
        mul = torch.ops.aten.mul.Tensor(add, add)
        # 再次调用 torch 的底层操作 aten.add.Tensor，对 mul 变量执行加法运算并将结果存入 add_1 变量
        add_1 = torch.ops.aten.add.Tensor(mul, mul);  mul = None
        # 返回两个结果变量 add 和 add_1 的元组作为输出
        return (add, add_1)
    def test_constant_fqn(self):
        # 定义一个内部类 Nested，继承自 torch.nn.Module
        class Nested(torch.nn.Module):
            # 构造方法，初始化模块
            def __init__(self):
                super().__init__()
                # 创建一个常量张量，形状为 (2, 3)
                self.constant = torch.rand(2, 3)
                # 创建一个模型参数，形状为 (2, 3)
                self.parameter = torch.nn.Parameter(torch.rand(2, 3))

            # 前向传播方法，接收输入 x，返回 x 加上常量张量
            def forward(self, x):
                return x + self.constant

        # 定义一个模块类 Mod，继承自 torch.nn.Module
        class Mod(torch.nn.Module):
            # 构造方法，初始化模块
            def __init__(self):
                super().__init__()
                # 创建一个 Nested 类的实例 nested
                self.nested = Nested()

            # 前向传播方法，接收输入 x，返回 nested 模块的前向传播结果
            def forward(self, x):
                return self.nested(x) + self.nested.constant + self.nested.parameter

        # 创建一个 Mod 类的实例 m
        m = Mod()
        # 导出模型 m，传入一个形状为 (2, 3) 的随机张量作为示例输入，strict=True 表示严格模式
        ep = export(m, (torch.rand(2, 3),), strict=True)
        # 断言导出结果中的常量 "nested.constant" 等于模型 m 中 nested 模块的 constant 属性
        self.assertEqual(ep.constants["nested.constant"], m.nested.constant)
        # 断言导出结果中的模块经过调用后的输出与模型 m 经过调用后的输出相等
        self.assertEqual(ep.module()(torch.ones(2, 3)), m(torch.ones(2, 3)))

    def test_constant_name(self):
        # 定义一个内部类 Nested，继承自 torch.nn.Module
        class Nested(torch.nn.Module):
            # 构造方法，初始化模块
            def __init__(self):
                super().__init__()
                # 创建一个常量张量，形状为 (2, 3)
                self.constant = torch.rand(2, 3)
                # 创建一个模型参数，形状为 (2, 3)
                self.parameter = torch.nn.Parameter(torch.rand(2, 3))

            # 前向传播方法，接收输入 x，返回 x 加上常量张量
            def forward(self, x):
                return x + self.constant

        # 定义一个模块类 Mod，继承自 torch.nn.Module
        class Mod(torch.nn.Module):
            # 构造方法，初始化模块
            def __init__(self):
                super().__init__()
                # 创建两个 Nested 类的实例 nested_1 和 nested_2
                self.nested_1 = Nested()
                self.nested_2 = Nested()

            # 前向传播方法，接收输入 x，返回两个 nested 模块的前向传播结果及其属性
            def forward(self, x):
                return (
                    self.nested_1(x)
                    + self.nested_2(x)
                    + self.nested_1.constant
                    + self.nested_2.constant
                    + self.nested_1.parameter
                    + self.nested_2.parameter
                )

        # 创建一个 Mod 类的实例 m
        m = Mod()
        # 导出模型 m，传入一个形状为 (2, 3) 的随机张量作为示例输入，strict=False 表示非严格模式
        ep = export(m, (torch.rand(2, 3),), strict=False)
        # 断言导出结果中的模块经过调用后的输出与模型 m 经过调用后的输出相等
        self.assertEqual(ep.module()(torch.ones(2, 3)), m(torch.ones(2, 3)))

        # 检查当存在多个相同类的实例时，导出结果中的常量名 "nested_1.constant" 和 "nested_2.constant" 分别等于模型中对应实例的 constant 属性
        self.assertEqual(ep.constants["nested_1.constant"], m.nested_1.constant)
        self.assertEqual(ep.constants["nested_2.constant"], m.nested_2.constant)

        # 检查图中的常量名，应该有五个占位符节点，且每个占位符的名称与目标相等
        placeholders = [
            node for node in ep.graph_module.graph.nodes if node.op == "placeholder"
        ]
        self.assertEqual(len(placeholders), 5)
        self.assertTrue(all(ph.name == ph.target for ph in placeholders))
        # 对于重复的常量名，应该添加后缀以区分，例如 "c_nested_1_constant" 和 "c_nested_2_constant"
        self.assertEqual(placeholders[2].name, "c_nested_1_constant")
        self.assertEqual(placeholders[3].name, "c_nested_2_constant")
    # 定义一个测试方法，用于测试嵌套模型的导出和重新追溯
    def test_nested_retrace(self):
        # 定义一个嵌套的 PyTorch 模块 Nested
        class Nested(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个参数 param，其值为从正态分布中随机生成的三个数
                self.param = torch.nn.Parameter(torch.randn(3))

            # 定义前向传播方法，返回输入 x 加上参数 self.param 的结果
            def forward(self, x):
                return x + self.param

        # 定义另一个 PyTorch 模块 Foo
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个嵌套模块 Nested
                self.nested = Nested()

            # 定义前向传播方法，返回输入 x 加上 self.nested(x) 的结果
            def forward(self, x):
                return x + self.nested(x)

        # 创建一个 Foo 类的实例 foo，并将其放置在设备 "meta" 上
        foo = Foo().to("meta")
        # 创建一个输入元组，包含一个全为 1 的张量，并传递给 foo 进行前向传播
        inputs = (torch.ones(3, device="meta"),)
        foo(*inputs)
        # 对 foo 进行第一次导出
        ep = torch.export.export(foo, inputs, strict=False)

        # 从导出结果 ep 中获取模块 foo 的实例 foo_1
        foo_1 = ep.module()
        # 对 foo_1 进行第二次导出
        ep_1 = torch.export.export(foo_1, inputs, strict=False)

        # 遍历两次导出结果的图节点，比较其 meta 数据中的 nn_module_stack
        for node1, node2 in zip(ep.graph.nodes, ep_1.graph.nodes):
            # 获取节点 node1 和 node2 的 nn_module_stack 数据
            nn_module_stack_1 = node1.meta.get("nn_module_stack", None)
            nn_module_stack_2 = node2.meta.get("nn_module_stack", None)

            # 如果 nn_module_stack_1 为 None，则要求 nn_module_stack_2 也为 None
            if nn_module_stack_1 is None:
                self.assertTrue(nn_module_stack_2 is None)
            else:
                # 否则，对比 nn_module_stack_1 和 nn_module_stack_2 中的值是否相等
                for v1, v2 in zip(
                    nn_module_stack_1.values(), nn_module_stack_2.values()
                ):
                    self.assertEqual(v1, v2)
# 如果 torchdynamo 不支持当前环境，跳过此测试类
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
# 定义一个测试类 TestExportCustomClass，继承自 TorchTestCase
class TestExportCustomClass(TorchTestCase):
    # 在每个测试方法运行前执行的设置方法
    def setUp(self):
        # 如果运行环境是 FBCODE
        if IS_FBCODE:
            # 设置自定义类注册库文件路径
            lib_file_path = "//caffe2/test/cpp/jit:test_custom_class_registrations"
        # 如果运行环境是 SANCASTLE 或者 macOS，抛出跳过测试异常
        elif IS_SANDCASTLE or IS_MACOS:
            raise unittest.SkipTest("non-portable load_library call used in test")
        # 如果运行环境是 Windows
        elif IS_WINDOWS:
            # 查找并设置 torchbind_test.dll 库文件路径
            lib_file_path = find_library_location("torchbind_test.dll")
        # 如果运行环境是其他 Unix-like 系统
        else:
            # 查找并设置 libtorchbind_test.so 库文件路径
            lib_file_path = find_library_location("libtorchbind_test.so")
        
        # 加载指定路径的库文件
        torch.ops.load_library(str(lib_file_path))
    def test_lift_custom_obj(self):
        # TODO: 一旦实现自定义类追踪，修复此测试

        # 创建一个自定义对象实例
        custom_obj = torch.classes._TorchScriptTesting._PickleTester([3, 4])

        # 定义一个简单的神经网络模块
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x + x

        f = Foo()

        # 准备输入数据
        inputs = (torch.zeros(4, 4),)
        # 导出模型
        ep = export(f, inputs)

        # 在计算图中找到适当的节点，将其值替换为自定义类的实例
        for node in ep.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                with ep.graph.inserting_before(node):
                    # 将自定义对象设置为图模块的属性
                    setattr(ep.graph_module, "custom_obj", custom_obj)
                    # 获取属性节点
                    getattr_node = ep.graph.get_attr("custom_obj")
                    # 复制所需的 nn_module_stack
                    getattr_node.meta["nn_module_stack"] = node.meta["nn_module_stack"]
                    # 创建自定义节点
                    custom_node = ep.graph.call_function(
                        torch.ops._TorchScriptTesting.take_an_instance.default,
                        (getattr_node,),
                    )
                    custom_node.meta["val"] = torch.ones(4, 4)
                    # 复制所需的 nn_module_stack
                    custom_node.meta["nn_module_stack"] = node.meta["nn_module_stack"]
                    custom_node.meta["torch_fn"] = (
                        "custom_op",
                        "torch.ops._TorchScriptTesting.take_an_instance.default",
                    )
                    arg0, _ = node.args
                    node.args = (arg0, custom_node)

        # 导入必要的模块
        from torch._export.passes.lift_constants_pass import lift_constants_pass
        from torch._export.serde.serialize import deserialize, serialize

        # 提升常量并序列化
        constants = lift_constants_pass(ep.graph_module, ep.graph_signature, {})
        for k, v in constants.items():
            assert k not in ep.constants
            ep._constants[k] = v
        serialized_vals = serialize(ep)
        deserialized_ep = deserialize(serialized_vals)

        # 验证反序列化后的计算图中的自定义节点
        for node in deserialized_ep.graph.nodes:
            if (
                node.op == "call_function"
                and node.target
                == torch.ops._TorchScriptTesting.take_an_instance.default
            ):
                arg = node.args[0]
                self.assertTrue(arg.op == "placeholder")

    def test_tolist_nonstrict_output(self):
        # 定义一个简单的神经网络模块
        class M(torch.nn.Module):
            def forward(self, x):
                x.tolist()

        # 导出模型
        ep = torch.export.export(M(), (torch.ones(3),), strict=False)
# 如果当前脚本被直接执行（而不是被导入到其他模块中），则执行下面的代码块
if __name__ == "__main__":
    # 调用运行测试函数
    run_tests()
```
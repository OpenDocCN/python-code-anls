# `.\pytorch\test\test_fx.py`

```
# Owner(s): ["module: fx"]

# 引入必要的内置模块和第三方库
import builtins
import contextlib
import copy
import functools
import inspect
import math
import numbers
import io
import operator
import os
import pickle
import sys
import torch
import traceback
import typing
import types
import warnings
import unittest

# 从 math 模块中导入 sqrt 函数
from math import sqrt

# 引入 functorch 的控制流实验性模块
from functorch.experimental import control_flow

# 从 torch.multiprocessing 模块导入 Process 类
from torch.multiprocessing import Process

# 从 torch.testing 模块导入 FileCheck 类
from torch.testing import FileCheck

# 导入 torch.testing._internal.common_methods_invocations 模块中的 op_db 变量
from torch.testing._internal.common_methods_invocations import op_db

# 导入 torch.testing._internal.common_device_type 模块中的 ops, onlyCPU, instantiate_device_type_tests 变量
from torch.testing._internal.common_device_type import ops, onlyCPU, instantiate_device_type_tests

# 导入 torch.utils._pytree 模块中的 pytree 变量
import torch.utils._pytree as pytree

# 导入 torch.fx._pytree 模块中的 fx_pytree 变量
import torch.fx._pytree as fx_pytree

# 导入 torch.fx 模块中的各个类和函数
from torch.fx import symbolic_trace, Proxy, Node, GraphModule, Interpreter, Tracer, Transformer, Graph, wrap, PH, CodeGen

# 导入 torch.fx.node 模块中的 Target, Argument, _format_arg 类和函数
from torch.fx.node import Target, Argument, _format_arg

# 导入 torch.fx.passes 模块中的 shape_prop 函数
from torch.fx.passes import shape_prop

# 导入 torch.fx.immutable_collections 模块中的 immutable_dict, immutable_list 变量
from torch.fx.immutable_collections import immutable_dict, immutable_list

# 导入 torch.fx.experimental.rewriter 模块中的 RewritingTracer 类
from torch.fx.experimental.rewriter import RewritingTracer

# 导入 torch.fx.operator_schemas 模块中的 get_signature_for_torch_op 函数
from torch.fx.operator_schemas import get_signature_for_torch_op

# 从 copy 模块中导入 deepcopy 函数
from copy import deepcopy

# 从 collections 模块中导入 namedtuple 类
from collections import namedtuple

# 导入 torch.fx.proxy 模块中的 TraceError 类
from torch.fx.proxy import TraceError

# 导入 torch.fx._compatibility 模块中的 _BACK_COMPAT_OBJECTS, _MARKED_WITH_COMPATIBILITY 变量
from torch.fx._compatibility import _BACK_COMPAT_OBJECTS, _MARKED_WITH_COMPATIBILITY

# 导入 torch.fx._symbolic_trace 模块中的 PHBase, PHWithMeta 类
from torch.fx._symbolic_trace import PHBase, PHWithMeta

# 从 fx.test_subgraph_rewriter 模块导入 TestSubgraphRewriter 类（忽略 F401 错误）
from fx.test_subgraph_rewriter import TestSubgraphRewriter  # noqa: F401

# 从 fx.test_dce_pass 模块导入 TestDCE 类（忽略 F401 错误）
from fx.test_dce_pass import TestDCE  # noqa: F401

# 从 fx.test_fx_const_fold 模块导入 TestConstFold 类（忽略 F401 错误）
from fx.test_fx_const_fold import TestConstFold  # noqa: F401

# 从 fx.test_fx_param_shape_control_flow 模块导入 TestConstParamShapeInControlFlow 类（忽略 F401 错误）
from fx.test_fx_param_shape_control_flow import TestConstParamShapeInControlFlow  # noqa: F401

# 从 fx.test_pass_infra 模块导入 TestPassManager 类（忽略 F401 错误）
from fx.test_pass_infra import TestPassManager  # noqa: F401

# 从 fx.test_common_passes 模块导入 TestCommonPass 类（忽略 F401 错误）
from fx.test_common_passes import TestCommonPass  # noqa: F401

# 从 fx.test_cse_pass 模块导入 TestCSEPass 类（忽略 F401 错误）
from fx.test_cse_pass import TestCSEPass  # noqa: F401

# 从 fx.test_matcher_utils 模块导入 TestMatcher 类（忽略 F401 错误）
from fx.test_matcher_utils import TestMatcher  # noqa: F401

# 从 fx.test_source_matcher_utils 模块导入 TestSourceMatcher 类（忽略 F401 错误）
from fx.test_source_matcher_utils import TestSourceMatcher  # noqa: F401

# 从 fx.test_gradual_type 模块导入 AnnotationsTest, TypeCheckerTest 类（忽略 F401 错误）
from fx.test_gradual_type import AnnotationsTest  # noqa: F401
from fx.test_gradual_type import TypeCheckerTest  # noqa: F401

# 从 torch.testing._internal.common_utils 模块中导入以下函数和变量
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    IS_MACOS,
    IS_WINDOWS,
    find_library_location,
    run_tests,
    skipIfTorchDynamo,
)

# 从 torch.testing._internal.jit_utils 模块导入 JitTestCase 类
from torch.testing._internal.jit_utils import JitTestCase

# 导入 fx.named_tup 模块中的 MyNamedTup 类
from fx.named_tup import MyNamedTup

# 尝试导入 torchvision.models 模块，如果失败设置 HAS_TORCHVISION 为 False
try:
    from torchvision import models as torchvision_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# 根据 HAS_TORCHVISION 值决定是否跳过测试
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

# 从 torch.testing._internal.common_quantization 模块导入 skipIfNoDynamoSupport 函数
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport


# 定义一个简单的 torch.nn.Module 子类 SimpleTest
class SimpleTest(torch.nn.Module):
    def forward(self, x):
        # 返回 x + 3.0 的 relu 激活结果
        return torch.relu(x + 3.0)


# 定义一个非 torch Leaf 函数 a_non_torch_leaf
def a_non_torch_leaf(a, b):
    # 返回 a 和 b 的和
    return a + b


# 用于 test_autowrap_function 的函数 fx_int，将浮点数 x 转换为整数
def fx_int(x: float) -> int:
    return int(x)


# 用于 test_autowrap_function 的函数 fx_int_x2，将浮点数 x 的两倍转换为整数
def fx_int_x2(x: float) -> int:
    # 将输入参数 x 转换为整数类型，然后将其乘以 2，最后返回结果
    return int(x) * 2
# 定义命名元组 Point，用于表示具有 x 和 y 属性的点
Point = namedtuple('Point', ['x', 'y'])

# wrap() 函数用于包装函数，此处调用 wrap() 来包装函数 'a_lifted_leaf'
wrap('a_lifted_leaf')

# 再次调用 wrap() 来确保多次包装不会造成问题
wrap('a_lifted_leaf')

# 定义函数 a_lifted_leaf，接受两个参数并返回它们的和
def a_lifted_leaf(a, b):
    return a[0] + a[1] + b

# wrap() 函数再次被用于包装函数 a_lifted_leaf2
wrap(a_lifted_leaf2)

# wrap() 函数用于包装内置函数 len
wrap('len')

# wrap() 函数用于包装内置函数 getattr
wrap('getattr')

# 定义函数 wrapped_named_tup，接受两个参数，返回它们的属性之和
def wrapped_named_tup(p1, *, p2):
    return p1.x + p2.y

# wrap() 函数用于包装函数 wrapped_named_tup
wrap(wrapped_named_tup)

# 使用装饰器 @wrap 来包装函数 wrapped_via_decorator
@wrap
def wrapped_via_decorator(a):
    return a + 1

# wrap() 函数用于包装函数 wrapped_with_submodule
wrap('wrapped_with_submodule')

# 定义函数 wrapped_with_submodule，接受一个 torch.Tensor 和一个 torch.nn.BatchNorm1d 参数，并返回 batchnorm1d(x)
def wrapped_with_submodule(x: torch.Tensor, batchnorm1d: torch.nn.BatchNorm1d):
    return batchnorm1d(x)

# 定义装饰器 my_decorator，用于包装函数并保留其元数据
def my_decorator(f):
    @functools.wraps(f)
    def wrapper_inside_decorator(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper_inside_decorator

# 使用装饰器 @wrap 和 @my_decorator 来包装函数 wrapped_decorated_fn
@wrap
@my_decorator
def wrapped_decorated_fn(x):
    return x

# 将函数 wrapped_via_decorator 的引用保存到变量 real_wrapped_via_decorator 中
real_wrapped_via_decorator = wrapped_via_decorator

# 将函数 a_lifted_leaf 的引用保存到变量 real_a_lifed_leaf 中
real_a_lifed_leaf = a_lifted_leaf

# 将函数 a_lifted_leaf2 的引用保存到变量 real_a_lifed_leaf2 中
real_a_lifed_leaf2 = a_lifted_leaf2

# 将 sqrt 函数的引用保存到变量 _sqrt 中
_sqrt = sqrt

# wrap() 函数用于包装函数 wrapper_fn
wrap('wrapper_fn')

# 定义函数 wrapper_fn，接受一个参数 x，并调用 torch.foo(x) 返回结果
def wrapper_fn(x):
    return torch.foo(x)

# 定义名为 Pair 的命名元组，包含两个属性 x 和 y，均为 torch.Tensor 类型
class Pair(NamedTuple):
    x : torch.Tensor
    y : torch.Tensor

    # 定义 Pair 类的方法 _custom_fx_repr_fn，返回一个字符串，表示对象的 x 和 y 属性
    def _custom_fx_repr_fn(self) -> str:
        return f"Pair(x={_format_arg(self.x)}, y={_format_arg(self.y)})"

# 用于测试 pytrees 的类 Foo，包含两个属性 a 和 b
class Foo:
    def __init__(self, a, b):
        self.a = a
        self.b = b

# 定义一个继承自 torch.nn.Module 的类 Add，实现了 forward 方法，返回输入的两倍
class Add(torch.nn.Module):
    def forward(self, x):
        return x + x

# 使用装饰器 @torch.fx.has_side_effect 和 @torch.fx.wrap 来包装函数 side_effect_func
@torch.fx.has_side_effect
@torch.fx.wrap
def side_effect_func(x: torch.Tensor):
    print(x)

# 定义类 TestFX，继承自 JitTestCase，用于测试 FX 模块
class TestFX(JitTestCase):

    def setUp(self):
        super().setUp()
        # 设置测试环境，特别是当检测可变操作时，这在跟踪时是一个特性标志
        self.orig_tracer_mutable_flag = torch.fx.proxy.TracerBase.check_mutable_operations
        torch.fx.proxy.TracerBase.check_mutable_operations = True

        # 如果不是在特定环境下（FBCODE、Windows 或 macOS），加载 libtorchbind_test.so 库文件
        if not (IS_FBCODE or IS_WINDOWS or IS_MACOS):
            lib_file_path = find_library_location('libtorchbind_test.so')
            torch.ops.load_library(str(lib_file_path))

    def tearDown(self):
        super().tearDown()
        # 恢复测试环境的状态
        torch.fx.proxy.TracerBase.check_mutable_operations = self.orig_tracer_mutable_flag

    def checkGraphModule(self, m: torch.nn.Module, args, kwargs=None):
        """检查 nn.Module 的结果与 GraphModule 版本在给定的 args/kwargs 下是否匹配。"""
        kwargs = kwargs if kwargs else {}
        # 使用 symbolic_trace 对模块 m 进行符号化跟踪
        gm = symbolic_trace(m)
        # 对生成的图进行检查
        gm.graph.lint()
        # 调用 gm 函数，传入 args 和 kwargs，获取其结果
        test_outs = gm(*args, **kwargs)
        # 调用原始模块 m，传入 args 和 kwargs，获取其结果
        ref_outs = m(*args, **kwargs)
        # 断言两个结果相等
        self.assertEqual(ref_outs, test_outs)
    # 定义一个测试方法，用于测试图模块的功能
    def test_graph_module(self):
        # 定义一个继承自 torch.nn.Module 的子类 MySub
        class MySub(torch.nn.Module):
            # 初始化方法，初始化一个形状为 (4, 3) 的参数 w
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.rand(4, 3))

            # 前向传播方法，返回 w 与输入 x 的和
            def forward(self, x):
                return self.w + x

        # 定义一个继承自 torch.nn.Module 的主模块类 MyModule
        class MyModule(torch.nn.Module):
            # 初始化方法，初始化一个线性层和一个 MySub 实例，以及一个形状为 (3,) 的参数 w
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.sub_mod = MySub()
                self.w = torch.nn.Parameter(torch.rand(3))

            # 前向传播方法，接受三个输入 A、B、c，执行一系列计算并返回结果
            def forward(self, A, B, c):
                # 计算 torch.sigmoid(A) 和 self.lin(c) 的和
                t = torch.sigmoid(A) + self.lin(c)
                # 返回 self.sub_mod 的前向传播结果，参数为 t.data + self.w + t + 1 - A + B // A + -A + A.add(B, alpha=3)
                return self.sub_mod(t.data + self.w + t + 1 - A + B // A + -A + A.add(B, alpha=3))

        # 创建 MyModule 的实例 m
        m = MyModule()
        # 对 m 进行符号跟踪，生成图形模块 gm
        gm = symbolic_trace(m)

        # 将 gm 脚本化，生成 ms
        ms = torch.jit.script(gm)

        # 定义一个继承自 torch.nn.Module 的类 M2
        class M2(torch.nn.Module):
            # 前向传播方法，接受输入 A，返回 A 中的最大值加一和其索引加一
            def forward(self, A):
                m, idx = torch.max(A, 0)
                return m + 1, idx + 1

        # 创建 M2 的实例 m2
        m2 = M2()
        # 对 m2 进行符号跟踪，生成图形模块 gm2
        gm2 = symbolic_trace(m2)

        # 定义一个继承自 torch.nn.Module 的类 T
        class T(torch.nn.Module):
            # 前向传播方法，接受位置参数 args 和关键字参数 kwargs，计算并返回结果 x
            def forward(self, A, b=4, *args, c=5, **kwargs):
                x = A + 1 + args[0] + kwargs['3']
                return x

        # 创建 T 的实例 t
        t = T()
        # 对 t 进行符号跟踪
        symbolic_trace(t)

        # 对描述在 https://github.com/pytorch/pytorch/issues/63883 中的问题进行测试
        # 定义一个继承自 torch.nn.Module 的类 M3
        class M3(torch.nn.Module):
            # 前向传播方法，接受输入 x，返回 torch.relu(x) 的结果
            def forward(self, x):
                return torch.relu(x)

        # 创建 M3 的实例 m3
        m3 = M3()
        # 对 m3 进行符号跟踪，生成图形模块 gm3
        gm3 = symbolic_trace(m3)
        # 创建 gm3 的新实例 new_instance，传入 gm3 和其图形对象
        new_instance = gm3.__new__(type(gm3))
        new_instance.__init__(gm3, gm3.graph)

        # 创建一个形状为 (5, 3) 的随机张量 x
        x = torch.randn(5, 3)
        # 断言 new_instance(x) 的结果与 torch.relu(x) 的结果接近
        torch.testing.assert_close(new_instance(x), torch.relu(x))

    # 定义一个测试方法，用于测试符号跟踪中的文件名信息是否包含预期的基本名称
    def test_informative_co_filename(self):
        # 定义一个继承自 torch.nn.Module 的类 MyModule
        class MyModule(torch.nn.Module):
            # 前向传播方法，接受输入 a，返回 a 的两倍
            def forward(self, a):
                return a * 2

        # 对 MyModule 实例进行符号跟踪，生成图形模块 gm
        gm = symbolic_trace(MyModule())
        # 断言在 gm 的前向传播方法的代码对象中包含当前文件的基本名称
        self.assertIn(os.path.basename(__file__), gm.forward.__code__.co_filename)

    # 定义一个测试方法，用于测试自定义导入功能
    def test_custom_import(self):
        # 创建一个空的 TorchFX 图对象 graph
        graph = torch.fx.Graph()
        # 创建两个占位符节点 a 和 b
        a = graph.placeholder('x')
        b = graph.placeholder('y')
        # 在图中调用函数 a_non_torch_leaf，参数为 a 和 b
        c = graph.call_function(a_non_torch_leaf, (a, b))
        # 在图中调用 torch.sin 函数，参数为 c
        d = graph.call_function(torch.sin, (c,))
        # 将 d 设置为图的输出节点
        graph.output(d)
        # 创建一个 GraphModule 对象 gm，将其绑定到一个空的 torch.nn.Module 实例
        gm = GraphModule(torch.nn.Module(), graph)
        # 创建两个随机张量 x 和 y
        x, y = torch.rand(1), torch.rand(1)
        # 断言 torch.sin(x + y) 的结果等于 gm(x, y) 的结果
        self.assertEqual(torch.sin(x + y), gm(x, y))

    # 定义一个测试方法，用于测试带有 *args 和 **kwargs 的前向传播函数
    def test_args_kwargs(self):
        # 定义一个继承自 torch.nn.Module 的类 T
        class T(torch.nn.Module):
            # 前向传播方法，接受位置参数 args 和关键字参数 kwargs，计算并返回结果 x
            def forward(self, *args, **kwargs):
                x = args[0] + kwargs['foo']
                return x

        # 创建 T 的实例 t
        t = T()
        # 使用 self.checkGraphModule 方法检查 t 的图模块，传入随机张量和一个包含 'foo' 键的随机参数字典
        self.checkGraphModule(t, (torch.rand(1), torch.rand(1)), {'foo': torch.rand(1)})
    # 定义一个测试用例，测试具有可变位置参数和关键字参数的前向方法
    def test_varargs_concrete(self):
        # 定义一个继承自 torch.nn.Module 的内部类 T
        class T(torch.nn.Module):
            # 实现前向方法，接受任意数量的位置参数和关键字参数
            def forward(self, *args, **kwargs):
                # 计算前两个位置参数的和
                x = args[0] + args[1]
                return x

        # 创建两个随机张量作为位置参数
        args = (torch.rand(1), torch.rand(1))

        # 实例化类 T
        t = T()
        # 使用具体参数调用 t 的前向方法，并获得参考输出
        ref_outs = t(*args)
        # 对 t 进行符号化跟踪，使用具体参数为占位符
        gm = symbolic_trace(t, concrete_args=(torch.fx.PH, torch.fx.PH))
        # 对符号化图进行 lint 检查
        gm.graph.lint()
        # 使用具体参数调用符号化图的前向方法，并获得测试输出
        test_outs = gm(*args)
        # 断言参考输出和测试输出相等
        self.assertEqual(ref_outs, test_outs)

    # 定义一个测试用例，测试不带 self 的前向方法定义
    def test_args_kwargs_no_self(self):
        # 定义一个继承自 torch.nn.Module 的内部类 T
        class T(torch.nn.Module):
            # 实现前向方法，接受任意数量的位置参数和关键字参数
            def forward(*args, **kwargs):  # noqa: B902
                # 获取第一个位置参数作为 self
                self = args[0]
                # 对第二个位置参数应用 relu 激活函数
                return torch.relu(args[1])

        # 实例化类 T
        t = T()
        # 使用 self.assertRaisesRegex 断言捕获运行时错误，并验证错误消息
        with self.assertRaisesRegex(RuntimeError, r'cannot be part of \*args expansion'):
            # 使用具体参数调用 self.checkGraphModule 方法，并传递字典作为关键字参数
            self.checkGraphModule(t, (torch.rand(1), torch.rand(1)), {'foo': torch.rand(1)})

    # 定义一个测试用例，测试位移运算的符号化跟踪
    def test_fx_shifts(self):
        # 定义一个继承自 torch.nn.Module 的内部类 MyModule
        class MyModule(torch.nn.Module):
            # 实现前向方法，接受一个张量作为输入
            def forward(self, x):
                # 返回输入张量的左移 3 位和右移 3 位的结果
                return x << 3, x >> 3

        # 创建一个随机长整型张量作为输入
        input = torch.LongTensor(10).random_(0, 1024)

        # 实例化类 MyModule
        m = MyModule()
        # 使用 self.checkGraphModule 方法，对类 MyModule 进行符号化跟踪，并使用输入作为参数
        self.checkGraphModule(m, (input,))

    # 定义一个测试用例，测试按位与和按位或运算的符号化跟踪
    def test_fx_and_or(self):
        # 定义一个继承自 torch.nn.Module 的内部类 MyModule
        class MyModule(torch.nn.Module):
            # 实现前向方法，接受一个张量作为输入
            def forward(self, x):
                # 返回输入张量按位与自身和按位或自身的结果
                return x & x, x | x

        # 创建一个随机长整型张量作为输入
        input = torch.LongTensor(10).random_(0, 1024)

        # 实例化类 MyModule
        m = MyModule()
        # 使用 self.checkGraphModule 方法，对类 MyModule 进行符号化跟踪，并使用输入作为参数
        self.checkGraphModule(m, (input,))

    # 定义一个测试用例，测试字典操作的符号化跟踪
    def test_dict(self):
        # 定义一个继承自 torch.nn.Module 的内部类 MyDictMod
        class MyDictMod(torch.nn.Module):
            # 实现前向方法，接受一个字典作为输入
            def forward(self, d):
                # 返回字典中键为 '3' 的值经过 relu 激活后的结果，
                # 和一个包含键为 '4'、值为 '3' 的相反数的字典
                return d['3'].relu(), {'4': d['3'].neg()}

        # 创建一个包含键为 '3' 的随机张量的字典作为输入
        input_dict = {'3': torch.rand(3, 4)}
        # 实例化类 MyDictMod
        m = MyDictMod()

        # 使用 self.checkGraphModule 方法，对类 MyDictMod 进行符号化跟踪，并使用输入作为参数
        self.checkGraphModule(m, (input_dict,))

    # 定义一个测试用例，测试矩阵乘法的符号化跟踪
    def test_matmul_tracing(self):
        # 创建一个包含三个随机元素的张量作为常量
        const = torch.randn(3)

        # 定义一个矩阵乘法函数 matmul_f，接受一个张量作为输入
        def matmul_f(x):
            # 返回输入张量与常量张量 const 的矩阵乘法结果
            return x @ const

        # 对 matmul_f 进行符号化跟踪
        mod = symbolic_trace(matmul_f)
        # 创建一个随机张量作为输入
        inp = torch.randn(3)
        # 使用 self.assertEqual 断言符号化模块的输出与原始函数的输出相等
        self.assertEqual(mod(inp), matmul_f(inp))

        # 定义一个反向矩阵乘法函数 rmatmul_f，接受一个张量作为输入
        def rmatmul_f(x):
            # 返回常量张量 const 与输入张量的矩阵乘法结果
            return const @ x

        # 对 rmatmul_f 进行符号化跟踪
        mod = symbolic_trace(rmatmul_f)
        # 创建一个随机张量作为输入
        inp = torch.randn(3)
        # 使用 self.assertEqual 断言符号化模块的输出与原始函数的输出相等
        self.assertEqual(mod(inp), rmatmul_f(inp))

    # 定义一个测试用例，测试控制流的符号化跟踪
    @skipIfNoDynamoSupport
    def test_control_flow_tracing(self):
        # 定义一个返回两个参数之和的函数 true
        def true(x, y):
            return x + y

        # 定义一个返回两个参数之差的函数 false
        def false(x, y):
            return x - y

        # 定义一个函数 f，接受两个参数 x 和 y
        def f(x, y):
            # 使用控制流的条件运算符 cond，根据 x[0] == 0 的结果选择 true 或 false 函数
            x = control_flow.cond(x[0] == 0, true, false, [x, y])

        # 使用 self.assertRaisesRegex 断言捕获运行时错误，并验证错误消息
        with self.assertRaisesRegex(RuntimeError, r"Expected pred to be bool or tensor, but got Proxy\(eq\)"):
            # 对函数 f 进行符号化跟踪
            _ = symbolic_trace(f)
    def test_disallow_override(self):
        # 自定义代理以禁止就地张量操作
        class NoMutableCallTracer(Tracer):
            # 创建节点方法，用于追踪操作
            def create_node(self, kind: str, target: Union[str, Callable],
                            args: Tuple[Argument, ...], kwargs: Dict[str, Any], name: Optional[str] = None,
                            type_expr: Optional[Any] = None) -> Node:
                # 如果目标是字符串，则使用目标作为名称，否则使用目标的类型名
                name = target if isinstance(target, str) else torch.typename(target)
                # 检查名称是否以下划线结尾，如果是则抛出运行时错误
                if name[-1] == '_':
                    raise RuntimeError('In-place operations are not supported')
                # 调用父类的创建节点方法
                return super().create_node(kind, target, args, kwargs, name)

        # 测试方法1
        class MyInplaceMod(torch.nn.Module):
            # 前向传播方法
            def forward(self, x):
                # 就地加法操作
                x.add_(3.0)
                return x

        m = MyInplaceMod()

        # 断言捕获运行时错误，验证就地操作不被支持
        with self.assertRaisesRegex(RuntimeError, 'In-place operations'):
            NoMutableCallTracer().trace(m)

        # 测试方法2
        class MyInplaceMod2(torch.nn.Module):
            # 前向传播方法
            def forward(self, x):
                # 就地对数操作
                torch.log_(x)
                return x

        m2 = MyInplaceMod2()
        # 断言捕获运行时错误，验证就地操作不被支持
        with self.assertRaisesRegex(RuntimeError, 'In-place operations'):
            NoMutableCallTracer().trace(m2)

        # 测试方法3
        class MyInplaceMod3(torch.nn.Module):
            # 前向传播方法
            def forward(self, x):
                # 创建张量 y，并执行就地加法操作
                y = torch.ones(3, 4)
                y.add_(x)
                return x

        m3 = MyInplaceMod3()
        # 断言捕获运行时错误，验证就地操作不被支持
        with self.assertRaisesRegex(RuntimeError, 'In-place operations'):
            NoMutableCallTracer().trace(m3)

    def test_leaf_module(self):
        # 自定义代理以确保没有叶子模块，所有内容都应该被追踪
        class NoLeafModulesTracer(Tracer):
            # 判断模块是否为叶子模块的方法
            def is_leaf_module(self, m, qualname):
                return False

        # 测试用的简单 ReLU 模块
        class MyReluMod(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            # 前向传播方法
            def forward(self, x):
                return self.relu(x)

        mrm = MyReluMod()
        # 使用自定义的代理追踪模块
        sym = NoLeafModulesTracer().trace(mrm)
        # 遍历追踪到的符号表示的节点，确保没有调用模块操作
        for node in sym.nodes:
            self.assertNotEqual(node.op, 'call_module')
        # 对符号表示进行静态检查
        sym.lint()

    def test_wrap(self):
        # 断言检查
        self.assertEqual(3 + 4 + 5, a_lifted_leaf((3, 4), 5))

        # 定义一个要追踪的函数
        def to_trace(y):
            return a_lifted_leaf((4, y), 3) + a_lifted_leaf((3, 4), 5) + a_lifted_leaf((y, y), y)

        # 使用符号追踪对函数进行追踪
        m = symbolic_trace(to_trace)
        # 断言检查符号表示中包含特定代码片段
        self.assertIn('a_lifted_leaf', m.code)
        # 断言检查符号追踪结果
        self.assertEqual(27, m(2))
        # 断言检查对象标识相等性
        self.assertIs(a_lifted_leaf, real_a_lifed_leaf)
    # 测试直接包装函数的情况
    def test_wrap_fn_directly(self):
        # 断言调用 a_lifted_leaf2 函数的结果为 3 + 4 + 5
        self.assertEqual(3 + 4 + 5, a_lifted_leaf2((3, 4), 5))

        # 定义内部函数 to_trace，接受参数 y，并返回多个 a_lifted_leaf2 函数的结果求和
        def to_trace(y):
            return a_lifted_leaf2((4, y), 3) + a_lifted_leaf2((3, 4), 5) + a_lifted_leaf2((y, y), y)

        # 对 to_trace 函数进行符号追踪，得到符号化的模型 m
        m = symbolic_trace(to_trace)
        # 断言在符号化的代码中包含 'a_lifted_leaf2'
        self.assertIn('a_lifted_leaf2', m.code)
        # 断言 m(2) 的结果为 27
        self.assertEqual(27, m(2))
        # 断言 a_lifted_leaf2 和 real_a_lifed_leaf2 引用的是同一个对象
        self.assertIs(a_lifted_leaf2, real_a_lifed_leaf2)

    # 测试通过装饰器包装的情况
    def test_wrapped_via_decorator(self):
        # 断言调用 wrapped_via_decorator(0) 的结果为 1
        self.assertEqual(wrapped_via_decorator(0), 1)

        # 定义内部函数 to_trace，接受参数 y，并返回 wrapped_via_decorator 函数的结果
        def to_trace(y):
            return wrapped_via_decorator(y)

        # 对 to_trace 函数进行符号追踪，得到符号化的模型 m
        m = symbolic_trace(to_trace)
        # 断言在符号化的代码中包含 'wrapped_via_decorator'
        self.assertIn('wrapped_via_decorator', m.code)
        # 断言 m(0) 的结果为 1
        self.assertEqual(m(0), 1)
        # 断言 wrapped_via_decorator 和 real_wrapped_via_decorator 引用的是同一个对象
        self.assertIs(wrapped_via_decorator, real_wrapped_via_decorator)
        # 断言 wrapped_via_decorator 没有属性 "__fx_already_patched"
        self.assertFalse(hasattr(wrapped_via_decorator, "__fx_already_patched"))

    # 测试装饰器和转换后的包装情况
    def test_wrapped_via_decorator_and_transformed(self):
        # 断言调用 wrapped_via_decorator(0) 的结果为 1
        self.assertEqual(wrapped_via_decorator(0), 1)

        # 定义内部函数 to_trace，接受参数 y，并返回 wrapped_via_decorator 函数的结果
        def to_trace(y):
            return wrapped_via_decorator(y)

        # 对 to_trace 函数进行符号追踪，得到符号化的模型 m
        m = symbolic_trace(to_trace)
        # 断言在符号化的代码中包含 'wrapped_via_decorator'
        self.assertIn('wrapped_via_decorator', m.code)
        # 断言 m(0) 的结果为 1
        self.assertEqual(m(0), 1)
        # 断言 wrapped_via_decorator 和 real_wrapped_via_decorator 引用的是同一个对象
        self.assertIs(wrapped_via_decorator, real_wrapped_via_decorator)
        # 断言 wrapped_via_decorator 没有属性 "__fx_already_patched"
        self.assertFalse(hasattr(wrapped_via_decorator, "__fx_already_patched"))

        # 对符号化的模型 m 进行转换
        transformed = torch.fx.Transformer(m).transform()
        # 断言在转换后的代码中包含 'wrapped_via_decorator'
        self.assertIn('wrapped_via_decorator', transformed.code)
        # 断言 transformed(0) 的结果为 1
        self.assertEqual(transformed(0), 1)
        # 断言 wrapped_via_decorator 和 real_wrapped_via_decorator 引用的是同一个对象
        self.assertIs(wrapped_via_decorator, real_wrapped_via_decorator)
        # 断言 wrapped_via_decorator 没有属性 "__fx_already_patched"

    # 测试包含子模块的包装情况
    def test_wrap_with_submodule(self):
        # 定义包含子模块的神经网络类 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个不带放缩参数的 BatchNorm1d 模块
                self.batchnorm1d = torch.nn.BatchNorm1d(2, affine=False)

            # 前向传播方法，接受输入 x，返回 wrapped_with_submodule 函数的结果
            def forward(self, x: torch.Tensor):
                return wrapped_with_submodule(x, self.batchnorm1d)

        # 对 M 类进行符号追踪，得到符号化的模型 m
        m = symbolic_trace(M())

        # 断言在符号化的代码中包含 "wrapped_with_submodule"
        self.assertIn("wrapped_with_submodule", m.code)

        # 创建一个形状为 (3, 2) 的随机输入张量
        input = torch.rand(3, 2)
        # 创建一个与模型中使用的 BatchNorm1d 参数相同的参考 BatchNorm1d 模块
        ref_batchnorm1d = torch.nn.BatchNorm1d(2, affine=False)
        # 断言参考 BatchNorm1d 模块对输入张量的输出结果与模型 m 对输入张量的输出结果相等
        self.assertEqual(ref_batchnorm1d(input), m(input))

    # 测试包装后的函数再次进行符号追踪
    def test_wrapped_retrace(self):
        # 定义内部函数 to_trace，接受参数 y，并返回 wrapped_via_decorator 函数的结果
        def to_trace(y):
            return wrapped_via_decorator(y)

        # 对 to_trace 函数进行符号追踪，得到符号化的模型 m
        m = symbolic_trace(to_trace)
        # 断言在符号化的代码中包含 'wrapped_via_decorator'
        self.assertIn('wrapped_via_decorator', m.code)
        # 断言 m(0) 的结果为 1
        self.assertEqual(m(0), 1)

        # 对符号化的模型 m 进行再次符号追踪，得到 retrace 模型
        retraced = symbolic_trace(m)
        # 断言在 retrace 模型的代码中包含 'wrapped_via_decorator'
        self.assertIn('wrapped_via_decorator', retraced.code)
        # 断言 retrace(0) 的结果为 1
        self.assertEqual(retraced(0), 1)

    # 测试装饰函数的包装情况
    def test_wrap_decorated_function(self):
        # 定义内部函数 to_trace，接受参数 y，并返回 wrapped_decorated_fn 函数的结果
        def to_trace(y):
            return wrapped_decorated_fn(y)

        # 对 to_trace 函数进行符号追踪，得到符号化的模型 m
        m = symbolic_trace(to_trace)
        # 断言在符号化的代码中包含 'wrapped_decorated_fn'
        self.assertIn('wrapped_decorated_fn', m.code)
        # 断言 m(1) 的结果为 1
        self.assertEqual(m(1), 1)
    def test_graph_edit_with_proxy(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                return a + b
        m = M()  # 创建一个新的 M 类实例
        g = symbolic_trace(m).graph  # 对模型 m 进行符号化跟踪，获取其图形表示
        new_g = torch.fx.Graph()  # 创建一个新的 Torch FX 图形对象
        val_map : Dict[Node, Node] = {}  # 创建一个空的节点映射字典
        output_val = new_g.graph_copy(g, val_map)  # 复制图 g 到 new_g，并更新节点映射
        t = Proxy(output_val)  # 创建一个代理对象 t，使用复制后的输出值 output_val
        # 测试我们可以使用代理对象来生成更多的图形代码，以供不需要与模块一起工作的内容后续使用。
        new_g.output((t + t).node)  # 在 new_g 图中输出代理对象 t 加自身的节点
        gm = GraphModule(m, new_g)  # 使用模型 m 和新的图 new_g 创建图模块
        gm.graph.lint()  # 对图 gm 执行 lint 检查
        self.assertEqual(gm(3, 4), 14)  # 断言调用 gm 模型的结果为 14

    def test_concrete_arg_none_assert(self):
        class Foo(torch.nn.Module):
            def forward(self, x, val=None):
                return x if val is None else x + val

        f = Foo()  # 创建一个新的 Foo 类实例
        traced = torch.fx.symbolic_trace(f, concrete_args={'val' : None})  # 对模型 f 进行符号化跟踪，特定化参数 'val' 为 None
        with self.assertRaisesRegex(AssertionError, 'val has been specialized to have value None'):
            traced(torch.randn(5), torch.randn(5))  # 断言调用 traced 模型时抛出特定错误

        x = torch.randn(5)  # 创建一个大小为 5 的随机张量 x
        torch.testing.assert_close(traced(x), f(x))  # 断言 traced 模型对输入 x 的结果与原始模型 f 的结果接近

    def test_trace_multiple_funcs(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y

            def minus_forward(self, x, y):
                return x - y

            def multiply_forward(self, x, y):
                return x * y

        f = Foo()  # 创建一个新的 Foo 类实例
        x, y = torch.randn(5), torch.randn(5)  # 创建两个大小为 5 的随机张量 x 和 y

        print(torch.__version__)  # 打印 Torch 的版本号

        tracer = Tracer()  # 创建一个 Tracer 实例
        torch.testing.assert_close(GraphModule(f, tracer.trace(f))(x, y), f(x, y))  # 断言使用 tracer 跟踪并执行 f 模型的结果与直接执行 f 的结果接近

        tracer.traced_func_name = "minus_forward"  # 设置要跟踪的函数名称为 "minus_forward"
        torch.testing.assert_close(
            GraphModule(f, tracer.trace(f))(x, y),
            f.minus_forward(x, y),
        )  # 断言使用 tracer 跟踪并执行 f.minus_forward 函数的结果与直接执行 f.minus_forward 的结果接近

        tracer.traced_func_name = "multiply_forward"  # 设置要跟踪的函数名称为 "multiply_forward"
        torch.testing.assert_close(
            GraphModule(f, tracer.trace(f))(x, y),
            f.multiply_forward(x, y),
        )  # 断言使用 tracer 跟踪并执行 f.multiply_forward 函数的结果与直接执行 f.multiply_forward 的结果接近

        tracer.traced_func_name = "add_forward"  # 设置要跟踪的函数名称为 "add_forward"
        with self.assertRaisesRegex(AssertionError, "doesn't exist in"):
            tracer.trace(f)  # 断言 tracer 跟踪 f 模型时出现特定错误信息

    def test_graph_unique_names(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                return a + b
        m = M()  # 创建一个新的 M 类实例
        g = symbolic_trace(m).graph  # 对模型 m 进行符号化跟踪，获取其图形表示
        new_g = torch.fx.Graph()  # 创建一个新的 Torch FX 图形对象
        val_map : Dict[Node, Node] = {}  # 创建一个空的节点映射字典
        output_val = new_g.graph_copy(g, val_map)  # 复制图 g 到 new_g，并更新节点映射
        t = Proxy(output_val)  # 创建一个代理对象 t，使用复制后的输出值 output_val
        # 测试我们可以使用代理对象来生成更多的图形代码，以供不需要与模块一起工作的内容后续使用。
        new_g.output((t + t).node)  # 在 new_g 图中输出代理对象 t 加自身的节点
        gm = GraphModule(m, new_g)  # 使用模型 m 和新的图 new_g 创建图模块
        seen_names : Set[str] = set()  # 创建一个空的字符串集合 seen_names
        for node in gm.graph.nodes:  # 遍历图 gm 中的所有节点
            assert node.name not in seen_names  # 断言节点的名称不在 seen_names 集合中
            seen_names.add(node.name)  # 将节点名称添加到 seen_names 集合中
    # 定义一个测试类，用于测试堆栈跟踪功能
    def test_stack_traces(self):
        # 定义一个简单的神经网络模块类，重写 forward 方法实现简单的加法操作
        class M(torch.nn.Module):
            def forward(self, a, b):
                return a + b

        # 创建一个 Torch 的 FX 代码跟踪器对象
        tracer = torch.fx.Tracer()
        # 设置跟踪器记录堆栈跟踪信息的开关为 True
        tracer.record_stack_traces = True

        # 使用跟踪器对 M 类进行代码跟踪，得到计算图
        graph = tracer.trace(M())
        # 保存原始节点列表，因为在测试中我们将插入新节点
        orig_graph_nodes = list(graph.nodes)
        # 遍历计算图的每个节点
        for node in orig_graph_nodes:
            # 如果节点操作为 'output'，则跳过
            if node.op == 'output':
                continue
            # 断言节点的堆栈跟踪信息不为空
            self.assertTrue(node.stack_trace is not None)
            # 断言堆栈跟踪信息中包含当前文件名 'test_fx.py'
            assert 'test_fx.py' in node.stack_trace

            # 验证复制节点不会丢失堆栈跟踪信息
            new_node = graph.node_copy(node)
            # 断言复制后的节点的堆栈跟踪信息不为空
            self.assertTrue(new_node.stack_trace is not None)
            # 断言堆栈跟踪信息中包含当前文件名 'test_fx.py'
            assert 'test_fx.py' in new_node.stack_trace

    # 定义另一个测试方法，测试带有转换器的堆栈跟踪功能
    def test_stack_traces_with_transformer(self):
        # 定义一个简单的神经网络模块类，重写 forward 方法实现简单的加法操作
        class M(torch.nn.Module):
            def forward(self, a, b):
                return a + b

        # 创建一个 Torch 的 FX 代码跟踪器对象
        tracer = torch.fx.Tracer()
        # 设置跟踪器记录堆栈跟踪信息的开关为 True
        tracer.record_stack_traces = True

        # 使用跟踪器对 M 类进行代码跟踪，得到计算图
        graph = tracer.trace(M())
        # 将跟踪的计算图转换为 GraphModule 对象
        gm = GraphModule(tracer.root, graph)
        # 对计算图进行转换操作
        new_gm = Transformer(gm).transform()

        # 遍历转换后的计算图的每个节点
        for node in new_gm.graph.nodes:
            # 如果节点操作为 'placeholder' 或 'output'，则跳过
            if node.op in {'placeholder', 'output'}:
                continue
            # 断言节点的堆栈跟踪信息不为空
            self.assertTrue(node.stack_trace is not None)
            # 断言堆栈跟踪信息中包含当前文件名 'test_fx.py'
            assert 'test_fx.py' in node.stack_trace

    # 定义测试方法，测试行号映射功能
    def test_lineno_map(self):
        # 定义一个简单的神经网络模块类，重写 forward 方法包含两个数学运算
        class M(torch.nn.Module):
            def forward(self, a, b):
                a = torch.sin(a)
                b = torch.cos(b)
                return a + b

        # 创建一个 Torch 的 FX 代码跟踪器对象
        tracer = torch.fx.Tracer()
        # 使用跟踪器对 M 类进行代码跟踪，得到计算图
        graph = tracer.trace(M())
        # 将跟踪的计算图转换为 GraphModule 对象
        gm = GraphModule(tracer.root, graph)

        # 预期的行号映射字典，表示每行代码从哪行开始执行
        expected = {1: 2, 2: 3, 3: 4, 4: 5}
        # 断言预期的行号映射字典是 gm._lineno_map 的子集
        self.assertTrue(set(expected.items()).issubset(set(gm._lineno_map.items())))

        # 测试自定义代码生成器
        def transform_code(code):
            return ["print('hello!')\n", *code]

        # 设置图形对象的代码生成回调函数
        gm.graph.on_generate_code(lambda _: transform_code)
        # 重新编译图形对象
        gm.recompile()

        # 更新预期的行号映射字典，因为插入了一行打印语句
        expected = {2: 2, 3: 3, 4: 4, 5: 5}
        # 断言更新后的预期行号映射字典是 gm._lineno_map 的子集
        self.assertTrue(set(expected.items()).issubset(set(gm._lineno_map.items())))

    # 定义测试方法，测试图形对象中节点名称的唯一性
    def test_graph_unique_names_manual(self):
        # 创建一个空的 Torch FX 图形对象
        graph: torch.fx.Graph = torch.fx.Graph()
        # 创建一个占位符节点 'a'
        a: torch.fx.Node = graph.create_node('placeholder', 'x')
        # 创建一个调用模块的节点 'b'
        b: torch.fx.Node = graph.create_node('call_module', 'linear_mod', args=(a,), name='foo_1_1')
        # 创建一个获取属性的节点 'c'
        c: torch.fx.Node = graph.create_node('get_attr', 'y_attr', name='foo_1')
        # 创建一个调用函数的节点 'd'
        d: torch.fx.Node = graph.create_node('call_function', operator.add, args=(b, c))
        # 设置图形的输出节点为 'd'
        graph.output(d)

        # 创建一个新的空 Torch FX 图形对象
        graph2 = torch.fx.Graph()
        # 创建一个空的节点映射字典
        val_map: Dict[Node, Node] = {}
        # 复制图形对象，将 graph 的节点复制到 graph2 中，并更新节点映射关系到 val_map
        graph2.graph_copy(graph, val_map)
        # 创建一个空的节点名称集合
        seen_names: Set[str] = set()
        # 遍历新图形对象的每个节点
        for node in graph2.nodes:
            # 断言节点的名称不在已见名称集合中
            assert node.name not in seen_names
            # 将当前节点名称添加到已见名称集合中
            seen_names.add(node.name)
    def test_unpack(self):
        # 定义一个简单的 Torch 模块类 M，用于测试
        class M(torch.nn.Module):
            # 定义模块的前向传播函数
            def forward(self, a, b):
                # 解包输入元组 a，并赋值给变量 c 和 d
                c, d = a
                # 返回 c + d + b 的结果
                return c + d + b

        # 生成随机张量作为测试输入
        a = (torch.rand(1), torch.rand(1))
        b = torch.rand(1)
        # 创建 M 的实例
        m = M()
        # 调用自定义的检查函数，验证图模块 m 在输入 (a, b) 上的行为
        self.checkGraphModule(m, (a, b))

    def test_reserved_getattr(self):
        """确保不使用保留的内置函数名如 `getattr` 作为节点名称"""
        # 定义一个简单的 Torch 模块类 M，用于测试
        class M(torch.nn.Module):
            # 定义模块的前向传播函数
            def forward(self, a):
                # 访问输入 a 的属性 foo.bar.baz
                return a.foo.bar.baz

        # 创建 M 的实例
        m = M()
        # 对模块 m 进行符号化追踪
        m_g = symbolic_trace(m)
        # 对追踪后的图进行 lint 检查
        m_g.graph.lint()
        # 遍历图中的节点
        for node in m_g.graph.nodes:
            # 断言节点的名称不是 "getattr"
            self.assertTrue(node.name != "getattr")

    @unittest.skip("Hotfix for SEV remediation")
    def test_trace_buffer_slice(self):
        # 定义测试中使用的常量
        bs, d_hid = 10, 23

        # 定义一个包含缓冲区和参数的复杂 Torch 模块类 ExampleCode
        class ExampleCode(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义两个随机初始化的参数
                self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
                self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
                # 定义一个线性层
                self.lin = torch.nn.Linear(d_hid, d_hid)
                # 注册一个缓冲区 buffer，其大小为 bs + 100 x d_hid
                self.register_buffer('buffer', torch.randn(bs + 100, d_hid))

            # 定义模块的前向传播函数
            def forward(self, x):
                # 矩阵乘法操作
                x = torch.mm(x, self.mm_param)
                # 保存跳跃连接的值
                skip_connection = x
                # 使用 ReLU 激活函数
                x = torch.relu(x)
                # 组合多个操作：矩阵乘法、切片操作和加法
                x = torch.mm(x, self.mm_param) + self.buffer[:x.shape[0]]
                # 线性层操作
                x = self.lin(x)
                # 再次使用 ReLU 激活函数
                x = torch.relu(x)
                # 加上之前保存的跳跃连接值
                x = x + skip_connection
                # 继续矩阵乘法操作
                x = torch.mm(x, self.mm_param2)
                # 再次使用线性层操作
                x = self.lin(x)
                # 返回输出结果
                return x

        # 创建 ExampleCode 的实例
        ec = ExampleCode()

        # 对 ExampleCode 进行符号化追踪
        traced = torch.fx.symbolic_trace(ec)

        # 生成随机输入张量 x
        x = torch.randn(bs, d_hid)
        # 使用 Torch 的测试工具验证原始模块与追踪后模块在输入 x 上的结果近似相等
        torch.testing.assert_close(ec(x), traced(x))

    def test_node_tagging(self):
        # 定义一个继承自 Tracer 的自定义追踪器类 TaggingTracer
        class TaggingTracer(Tracer):
            # 重写创建节点的方法，为每个节点添加标签 'foo'
            def create_node(self, kind: str, target: Union[str, Callable],
                            args: Tuple[Argument, ...], kwargs: Dict[str, Any], name: Optional[str] = None,
                            type_expr: Optional[Any] = None) -> Node:
                # 调用父类的创建节点方法
                n = super().create_node(kind, target, args, kwargs, name)
                # 为节点 n 添加标签 'foo'
                n.tag = 'foo'
                # 返回带标签的节点 n
                return n

        # 定义一个简单的 Torch 模块类 M，用于测试
        class M(torch.nn.Module):
            # 定义模块的前向传播函数
            def forward(self, a, b):
                # 返回输入 a 和 b 的和
                return a + b

        # 创建 M 的实例
        m = M()
        # 使用自定义追踪器 TaggingTracer 对模块 m 进行追踪
        g = TaggingTracer().trace(m)
        # 对追踪后的图进行 lint 检查
        g.lint()
        # 遍历图中的节点
        for n in g.nodes:
            # 断言每个节点都有 'tag' 属性，并且其值为 'foo'
            self.assertTrue(hasattr(n, 'tag'))
            self.assertEqual(n.tag, 'foo')
    def test_tensor_attribute(self):
        # 定义一个名为 `TensorAttribute` 的类，继承自 `torch.nn.Module`
        class TensorAttribute(torch.nn.Module):
            # 初始化方法，调用父类的初始化方法，生成一个大小为 (3, 4) 的随机张量 `tensor`
            def __init__(self):
                super().__init__()
                self.tensor = torch.rand(3, 4)

            # 前向传播方法，对输入 `x` 执行线性变换，使用类属性 `self.tensor`
            def forward(self, x):
                return torch.nn.functional.linear(x, self.tensor)

        # 创建 `TensorAttribute` 类的实例 `ta`
        ta = TensorAttribute()
        # 对 `ta` 进行符号跟踪
        traced = symbolic_trace(ta)
        # 调用跟踪后的模型 `traced` 对输入数据 `torch.rand(4, 4)` 进行前向传播
        traced(torch.rand(4, 4))

        # 定义一个名为 `WrapperForQualname` 的类，继承自 `torch.nn.Module`
        class WrapperForQualname(torch.nn.Module):
            # 初始化方法，调用父类的初始化方法，创建一个 `TensorAttribute` 类的实例 `self.ta`
            def __init__(self):
                super().__init__()
                self.ta = TensorAttribute()

            # 前向传播方法，对输入 `x` 执行线性变换，使用 `self.ta` 的属性 `tensor`
            def forward(self, x):
                return torch.nn.functional.linear(x, self.ta.tensor)

        # 创建 `WrapperForQualname` 类的实例 `wfq`
        wfq = WrapperForQualname()
        # 对 `wfq` 进行符号跟踪
        traced2 = symbolic_trace(wfq)
        # 对跟踪后的模型 `traced2` 的图进行检查
        traced2.graph.lint()
        # 调用跟踪后的模型 `traced2` 对输入数据 `torch.rand(4, 4)` 进行前向传播
        traced2(torch.rand(4, 4))

    def test_tensor_attribute_coalseced(self):

        # 定义一个内部函数 `count_attrs(fx_module)`，计算跟踪图中的 `get_attr` 操作的目标数量
        def count_attrs(fx_module):
            targets = set()
            # 遍历 `traced.graph.nodes`，如果节点的操作是 `get_attr`，则将目标添加到集合 `targets` 中
            for node in traced.graph.nodes:
                if node.op == 'get_attr':
                    targets.add(node.target)
            # 返回集合 `targets` 的长度
            return len(targets)

        # 创建一个值为 5 的张量 `val`
        val = torch.tensor(5)

        # 定义函数 `f(x)`，对输入 `x` 执行加法操作，并使用外部变量 `val` 及 `val2`
        def f(x):
            return x + val + val
        # 对函数 `f` 进行符号跟踪
        traced = symbolic_trace(f)
        # 对跟踪后的模型 `traced` 的图进行检查
        traced.graph.lint()
        # 断言跟踪后的模型 `traced` 的属性数量等于 1
        self.assertEqual(count_attrs(traced), 1)

        # 创建一个值为 5 的张量 `val2`
        val2 = torch.tensor(5)

        # 重新定义函数 `f(x)`，在函数内部定义局部变量 `val`，并使用外部变量 `val2`
        def f(x):
            val = torch.tensor(5)
            return x + val + val2

        # 对重新定义后的函数 `f` 进行符号跟踪
        traced = symbolic_trace(f)
        # 对跟踪后的模型 `traced` 的图进行检查
        traced.graph.lint()
        # 断言跟踪后的模型 `traced` 的属性数量等于 2
        self.assertEqual(count_attrs(traced), 2)

    def test_symbolic_trace_sequential(self):
        # 定义一个简单的类 `Simple`，继承自 `torch.nn.Module`
        class Simple(torch.nn.Module):
            # 前向传播方法，对输入 `x` 执行负操作
            def forward(self, x):
                return torch.neg(x)

        # 创建一个顺序容器 `seq`，包含三个 `Simple` 类的实例
        seq = torch.nn.Sequential(
            Simple(),
            Simple(),
            Simple()
        )
        # 对 `seq` 进行符号跟踪
        traced = symbolic_trace(seq)
        # 对跟踪后的模型 `traced` 的图进行检查
        traced.graph.lint()
        # 创建输入数据 `x`，大小为 (3, 4)
        x = torch.rand(3, 4)
        # 断言跟踪后的模型 `traced` 对输入数据 `x` 的输出与原始序列模型 `seq` 对输入数据 `x` 的输出相等
        self.assertEqual(traced(x), seq(x))

    def test_tensor_constant(self):
        # 定义一个名为 `ConstTensor` 的类，继承自 `torch.nn.Module`
        class ConstTensor(torch.nn.Module):
            # 前向传播方法，对输入 `x` 执行线性变换，使用大小为 (3, 4) 的零张量
            def forward(self, x):
                return torch.nn.functional.linear(x, torch.zeros(3, 4))

        # 创建 `ConstTensor` 类的实例 `ct`
        ct = ConstTensor()
        # 对 `ct` 进行符号跟踪
        traced = symbolic_trace(ct)
        # 对跟踪后的模型 `traced` 的图进行检查
        traced.graph.lint()
        # 调用跟踪后的模型 `traced` 对输入数据 `torch.rand(4, 4)` 进行前向传播
        traced(torch.rand(4, 4))

    def test_pickle_graphmodule(self):
        # 定义一个名为 `Nested` 的类，继承自 `torch.nn.Module`
        class Nested(torch.nn.Module):
            # 初始化方法，调用父类的初始化方法，创建一个线性层 `self.st`
            def __init__(self):
                super().__init__()
                self.st = torch.nn.Linear(4, 4)

            # 前向传播方法，对输入 `x` 执行线性变换，使用 `self.st`
            def forward(self, x):
                return self.st(x)

        # 创建 `Nested` 类的实例 `n`
        n = Nested()
        # 对 `n` 进行符号跟踪
        traced = symbolic_trace(n)
        # 对跟踪后的模型 `traced` 的图进行检查
        traced.graph.lint()
        # 将跟踪后的模型 `traced` 序列化为字节流 `pickled`
        pickled = pickle.dumps(traced)
        # 从序列化的字节流 `pickled` 中加载模型，得到 `loaded`
        loaded = pickle.loads(pickled)
        # 对加载后的模型 `loaded` 的图进行检查
        loaded.graph.lint()
        # 创建输入数据 `x`，大小为 (3, 4)
        x = torch.rand(3, 4)
        # 断言加载后的模型 `loaded` 对输入数据 `x` 的输出与跟踪前的模型 `traced` 对输入数据 `x` 的输出相等
        self.assertEqual(loaded(x), traced(x))
    # 测试序列化和反序列化带有自定义导入的 GraphModule 对象
    def test_pickle_custom_import(self):
        # 创建一个空的 Torch FX 图形对象
        graph = torch.fx.Graph()
        # 在图中创建一个占位符节点 'x'
        a = graph.placeholder('x')
        # 在图中创建另一个占位符节点 'y'
        b = graph.placeholder('y')
        # 在图中调用非 Torch 叶子函数 a_non_torch_leaf，并传入节点 a 和 b 作为参数
        c = graph.call_function(a_non_torch_leaf, (a, b))
        # 在图中调用 Torch 中的 sin 函数，并传入节点 c 作为参数
        d = graph.call_function(torch.sin, (c,))
        # 设置图的输出节点为 d
        graph.output(d)
        # 创建一个空的 GraphModule 对象，基于一个空的 nn.Module 和上述图
        gm = GraphModule(torch.nn.Module(), graph)
        # 对 GraphModule 对象进行序列化，得到字节流
        pickled = pickle.dumps(gm)
        # 反序列化得到的字节流，得到一个新的 GraphModule 对象
        loaded = pickle.loads(pickled)
        # 对加载后的图进行 lint 检查
        loaded.graph.lint()
        # 创建两个随机张量 x 和 y
        x, y = torch.rand(1), torch.rand(1)
        # 断言加载后的 GraphModule 和原始的 GraphModule 在输入 x 和 y 下的输出相等
        self.assertEqual(loaded(x, y), gm(x, y))

    # 测试 Torch FX 图形中节点的输入情况
    def test_all_input_nodes(self):
        # 创建一个空的 Torch FX 图形对象
        graph : torch.fx.Graph = torch.fx.Graph()
        # 在图中创建一个占位符节点 'x'
        a : torch.fx.Node = graph.placeholder('x')
        # 在图中调用模块 'linear_mod'，并传入节点 a 作为参数
        b : torch.fx.Node = graph.call_module('linear_mod', args=(a,))
        # 获取属性 'y_attr' 的值作为节点 c
        c : torch.fx.Node = graph.get_attr('y_attr')
        # 在图中调用 operator.add 函数，并传入节点 b 和 c 作为参数
        d : torch.fx.Node = graph.call_function(operator.add, args=(b, c))
        # 在图中调用 torch.unsqueeze 函数，并传入节点 d 和 0 作为参数
        e : torch.fx.Node = graph.call_function(torch.unsqueeze, args=(d, 0))
        # 设置图的输出节点为 e
        graph.output(e)
        # 对图进行 lint 检查
        graph.lint()

        # 断言节点 b 的所有输入节点是 [a]
        self.assertEqual(b.all_input_nodes, [a])
        # 断言节点 c 的所有输入节点是空列表
        self.assertEqual(c.all_input_nodes, [])
        # 断言节点 d 的所有输入节点是 [b, c]
        self.assertEqual(d.all_input_nodes, [b, c])
        # 断言节点 e 的所有输入节点是 [d]
        self.assertEqual(e.all_input_nodes, [d])

    # 测试使用 transform 函数深复制带有转换的 GraphModule 对象
    def test_deepcopy_graphmodule_with_transform(self):
        # 创建一个 SimpleTest 实例
        st = SimpleTest()
        # 对 SimpleTest 实例进行符号化追踪，得到追踪后的对象
        traced = symbolic_trace(st)
        # 对追踪后的图进行 lint 检查
        traced.graph.lint()

        # 定义一个 transform 函数，接受一个追踪对象，并返回一个新的 GraphModule 对象
        def transform(traced):
            # 创建一个空的 Torch FX 图形对象
            new_graph = torch.fx.Graph()
            # 创建一个空字典，用于映射原始图中的节点到复制图中的节点
            val_map : Dict[Node, Node] = {}
            # 复制追踪对象的图到新图中，并更新映射关系到 val_map 中
            output_value = new_graph.graph_copy(traced.graph, val_map)
            # 在新图中创建一个 relu 操作节点，作用于 output_value
            relu_out = new_graph.create_node(
                op='call_method', target='neg', args=(output_value,), kwargs={})
            # 设置新图的输出节点为 relu_out
            new_graph.output(relu_out)
            # 返回一个基于追踪对象和新图的 GraphModule 对象
            return GraphModule(traced, new_graph)

        # 使用 transform 函数对追踪对象进行转换，得到 transformed 对象
        transformed = transform(traced)
        # 对转换后的图进行 lint 检查
        transformed.graph.lint()
        # 对 transformed 对象进行深度复制，得到 copied 对象
        copied = copy.deepcopy(transformed)
        # 断言 transformed 对象和 copied 对象不是同一个类型
        self.assertNotEqual(id(type(transformed)), id(type(copied)))
        # 创建一个形状为 (3, 4) 的随机张量 x
        x = torch.randn(3, 4)
        # 断言 copied 对象和 transformed 对象在输入 x 下的输出是相等的
        self.assertEqual(copied(x), transformed(x))

    # 测试使用 deepcopy 复制具有子模块和参数的对象
    def test_deepcopy_with_submods_params(self):
        # 定义一个名为 Bar 的 nn.Module 子类
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个形状为 (3, 4) 的参数 param
                self.param = torch.nn.Parameter(torch.rand(3, 4))

            def forward(self, x):
                # 返回 relu(x) 加上 param 的结果
                return torch.relu(x) + self.param

        # 定义一个名为 Baz 的 nn.Module 子类
        class Baz(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个形状为 (3, 4) 的参数 param
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                # 创建一个名为 bar 的 Bar 类型子模块
                self.bar = Bar()

            def forward(self, x):
                # 返回 bar(x) 减去 param 的结果
                return self.bar(x) - self.param

        # 创建一个 Baz 类的实例 baz
        baz = Baz()
        # 对 baz 进行符号化追踪，得到追踪后的对象 traced
        traced = symbolic_trace(baz)
        # 对追踪后的图进行 lint 检查
        traced.graph.lint()
        # 使用 deepcopy 对追踪后的对象进行深度复制，得到复制对象 copied
        copied = copy.deepcopy(traced)
        # 对复制对象的图进行 lint 检查
        copied.graph.lint()
    def test_deepcopy_graph_with_tracer_cls(self):
        # 定义一个测试函数，测试深拷贝图形并使用跟踪器类
        class TestTracer(Tracer):
            # 自定义的跟踪器类，继承自 Tracer
            def is_leaf_module(self, module, name):
                # 判断模块是否为叶子模块的方法
                return True

        # 创建一个 Graph 对象，使用自定义的 TestTracer 跟踪器类
        g = Graph(tracer_cls=TestTracer)
        # 创建一个占位符节点 "x"
        x = g.placeholder("x")
        # 将节点 "x" 设置为图形的输出节点
        g.output(x)

        # 使用深拷贝创建图形对象 h
        h = copy.deepcopy(g)
        # 断言 h 的跟踪器类不为空
        self.assertIsNotNone(h._tracer_cls)
        # 断言 g 和 h 的跟踪器类相同
        self.assertTrue(g._tracer_cls == h._tracer_cls)

    def test_unpack_list_better_error(self):
        # 测试在遇到更好的错误时解包列表
        class SomeArgs(torch.nn.Module):
            # 模拟一个接受两个参数的神经网络模块
            def forward(self, a, b):
                return torch.rand(3, 4)

        class UnpacksList(torch.nn.Module):
            # 解包列表的模块
            def __init__(self):
                super().__init__()
                self.sa = SomeArgs()

            def forward(self, x : list):
                # 调用 SomeArgs 模块的 forward 方法，传入列表 x 作为参数
                return self.sa(*x)

        # 创建 UnpacksList 的实例 ul
        ul = UnpacksList()
        # 使用断言检查 symbolic_trace(ul) 是否引发 TraceError 异常，并包含指定错误信息
        with self.assertRaisesRegex(TraceError, 'Proxy object cannot be iterated.'):
            symbolic_trace(ul)

    def test_unpack_dict_better_error(self):
        # 测试在遇到更好的错误时解包字典
        class SomeKwargs(torch.nn.Module):
            # 模拟一个接受两个关键字参数的神经网络模块
            def forward(self, x=3, y=4):
                return torch.rand(3, 4)

        class UnpacksDict(torch.nn.Module):
            # 解包字典的模块
            def __init__(self):
                super().__init__()
                self.sk = SomeKwargs()

            def forward(self, x : dict):
                # 调用 SomeKwargs 模块的 forward 方法，传入字典 x 作为参数
                return self.sk(**x)

        # 创建 UnpacksDict 的实例 ud
        ud = UnpacksDict()
        # 使用断言检查 symbolic_trace(ud) 是否引发 TraceError 异常，并包含指定错误信息
        with self.assertRaisesRegex(TraceError, 'Proxy object cannot be iterated.'):
            symbolic_trace(ud)

    def test_pretty_print_targets(self):
        # 测试图形的漂亮打印，确保友好显示目标名称
        class SomeMod(torch.nn.Module):
            # 模拟一个神经网络模块，包含对 x 的操作
            def forward(self, x):
                return torch.add(x.foo + x.bar, 3.0)

        # 对 SomeMod 模块进行符号化跟踪
        traced = symbolic_trace(SomeMod())
        # 将图形对象转换为字符串
        graph_str = str(traced.graph)
        # 断言字符串中包含特定的操作名称
        self.assertIn('builtins.getattr', graph_str)
        self.assertIn('operator.add', graph_str)
        self.assertIn('torch.add', graph_str)

    def test_pretty_print_node(self):
        # 测试节点的漂亮打印
        class M(torch.nn.Module):
            # 包含参数和线性层的神经网络模块
            def __init__(self):
                super().__init__()
                self.param: torch.nn.Parameter = torch.nn.Parameter(
                    torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x: torch.Tensor, y: int = 2):
                # 网络模块的前向传播，包含多个操作
                return self.linear(x[y] + self.param).clamp(min=0.0, max=1.0)

        # 对 M 模块进行符号化跟踪
        traced = symbolic_trace(M())
        # 获取所有节点的格式化字符串，并按行连接起来
        all_formatted = "\n".join([n.format_node() for n in traced.graph.nodes])

        # 使用 FileCheck 检查打印输出是否符合预期
        FileCheck().check("x").check("placeholder") \
            .check("y").check("placeholder") \
            .check("getitem").check("call_function") \
            .check("param").check("get_attr") \
            .check("add").check("call_function") \
            .check("linear").check("call_module") \
            .check("clamp").check("call_method") \
            .run(all_formatted)
    def test_script_tensor_constant(self):
        # TorchScript seems to ignore attributes that start with `__`.
        # We used to call anonymous Tensor values `__tensor_constant*`, but
        # they were getting ignored by script. Now they're called
        # `_tensor_constant*`
        
        # 定义一个模块，用于展示 TorchScript 忽略以 `__` 开头的属性。
        # 以前我们称匿名张量值为 `__tensor_constant*`，但在脚本中被忽略了。
        # 现在称为 `_tensor_constant*` 以解决此问题。
        class IHaveATensorConstant(torch.nn.Module):
            def forward(self, x):
                return x + torch.rand(3, 4)

        # 对模块进行符号化追踪
        traced = torch.fx.symbolic_trace(IHaveATensorConstant())
        # 将追踪后的模块转换为 TorchScript
        torch.jit.script(traced)

    def test_autowrap_functions(self):
        # 定义一个模块，用于展示自动包装函数功能
        class AutowrapFnTest(torch.nn.Module):
            def forward(self, x):
                return fx_int(x.shape[0] / 2)

        # 定义另一个模块，展示多个自动包装函数功能
        class AutowrapFnTest2(torch.nn.Module):
            def forward(self, x):
                return fx_int(x.shape[0] / 2) + fx_int_x2(x.shape[0] / 2)

        # 检查函数是否被正确包装
        # `int` 通常会因为参数不能是 `Proxy` 而抛出 TypeError
        tracer = Tracer(autowrap_functions=(fx_int,))
        graph = tracer.trace(AutowrapFnTest())
        traced = GraphModule(tracer.root, graph, 'test')
        tracer_2 = Tracer(autowrap_functions=(fx_int, fx_int_x2))
        tracer_2.trace(AutowrapFnTest2())

        # 测试是否可以转换为 TorchScript
        traced_scripted = torch.jit.script(traced)
        self.assertEqual(traced_scripted(torch.rand(4)), 2)

    def test_tuple_no_subscript(self):
        # 定义一个函数，展示不支持下标的元组类型
        def foo(x : Tuple):
            return x[0]

        # 对函数进行符号化追踪
        traced = torch.fx.symbolic_trace(foo)
        x = (torch.randn(5, 3),)
        # 断言追踪后的结果与原始输入的第一个元素相等
        torch.testing.assert_close(traced(x), x[0])

        bio = io.BytesIO()
        
        # 将追踪后的函数对象保存到字节流中
        torch.save(traced, bio)

        bio.seek(0)

        # 从字节流中加载保存的函数对象
        loaded = torch.load(bio)

        # 断言加载后的结果与原始输入的第一个元素相等
        torch.testing.assert_close(loaded(x), x[0])

    def test_torch_fx_len(self):
        # 定义一个模块，展示 Torch FX 中的长度计算
        class FXLenTest(torch.nn.Module):
            def forward(self, x):
                return len(x)

        # 对模块进行符号化追踪
        traced = symbolic_trace(FXLenTest())
        self.assertEqual(traced(torch.rand(3, 4)), 3)

        # 测试是否可以转换为 TorchScript
        scripted = torch.jit.script(FXLenTest())
        self.assertEqual(scripted(torch.rand(3)), 3)

        # 对符号化追踪后的模块再次进行 TorchScript 转换
        traced_scripted = torch.jit.script(traced)
        self.assertEqual(traced_scripted(torch.rand(3)), 3)

        # 测试非代理对象的长度计算
        class FXLenTest2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = [3, 4, 5]

            def forward(self, x):
                return x + len(self.l)

        # 对模块进行符号化追踪
        traced2 = symbolic_trace(FXLenTest2())
        inp = torch.rand(3, 4)
        # 断言结果是否与预期相符
        self.assertEqual(traced2(inp), inp + 3.0)
        self.assertIs(len, builtins.len)

    def test_torch_fx_getattr(self):
        # 定义一个模块，展示 Torch FX 中的 getattr 功能
        class FXGetattrTest(torch.nn.Module):
            def forward(self, x):
                return getattr(x, 'nonexistent_attr', torch.Tensor([2, 3]))

        # 对模块进行符号化追踪
        traced = symbolic_trace(FXGetattrTest())
        # 断言追踪后的结果是否与预期相符
        self.assertEqual(traced(torch.rand(3, 4)), torch.Tensor([2, 3]))
    # 定义一个名为 test_sqrt 的测试方法
    def test_sqrt(self):
        # 定义一个继承自 torch.nn.Module 的类 Sqrt1
        class Sqrt1(torch.nn.Module):
            # 重写 forward 方法，返回输入张量的大小的平方根
            def forward(self, x):
                return sqrt(x.size(0))

        # 定义一个继承自 torch.nn.Module 的类 Sqrt2
        class Sqrt2(torch.nn.Module):
            # 重写 forward 方法，返回输入张量大小的平方根（使用 math 库）
            def forward(self, x):
                return math.sqrt(x.size(0))

        # 定义一个继承自 torch.nn.Module 的类 Sqrt3
        class Sqrt3(torch.nn.Module):
            # 重写 forward 方法，返回 x + math.sqrt(2) + sqrt(2) 的结果
            def forward(self, x):
                return x + math.sqrt(2) + sqrt(2)

        # 使用自定义函数 checkGraphModule 测试 Sqrt1 类的功能
        self.checkGraphModule(Sqrt1(), [torch.zeros(8)])
        # 使用自定义函数 checkGraphModule 测试 Sqrt2 类的功能
        self.checkGraphModule(Sqrt2(), [torch.zeros(8)])
        # 使用自定义函数 checkGraphModule 测试 Sqrt3 类的功能
        self.checkGraphModule(Sqrt3(), [torch.zeros(8)])
        # 断言 sqrt 函数与 _sqrt 相等
        self.assertIs(sqrt, _sqrt)
        # 断言 math.sqrt 函数与 _sqrt 相等
        self.assertIs(math.sqrt, _sqrt)

    # 定义一个名为 test_torch_custom_ops 的测试方法
    def test_torch_custom_ops(self):
        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 重写 forward 方法，调用 torch 自定义运算符 sigmoid 和 cat
            def forward(self, a):
                b = torch.ops.aten.sigmoid(a)
                c = torch.ops.aten.cat([a, b])
                return torch.ops.aten.cat((c, c))

        # 创建 M 类的实例 m
        m = M()
        # 生成一个形状为 (3,) 的随机输入张量
        input = torch.randn(3)
        # 获取 m 对输入 input 的预期输出
        ref_out = m(input)
        # 对模型 m 进行符号跟踪
        gm = symbolic_trace(m)
        # 对符号跟踪后的图进行静态分析
        gm.graph.lint()
        # 对输入 input 运行符号跟踪后的模型，并获取输出
        out = gm(input)
        # 断言运行后的输出与预期输出 ref_out 相等
        self.assertEqual(out, ref_out)

    # 定义一个名为 test_torch_op_overloads 的测试方法
    def test_torch_op_overloads(self):
        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 重写 forward 方法，调用 torch 运算符重载 add.Tensor
            def forward(self, a):
                b = torch.ops.aten.add.Tensor(a, a)
                return b

        # 创建 M 类的实例 m
        m = M()
        # 生成一个形状为 (3,) 的随机输入张量
        input = torch.randn(3)
        # 获取 m 对输入 input 的预期输出
        ref_out = m(input)
        # 对模型 m 进行符号跟踪
        gm = symbolic_trace(m)
        # 对符号跟踪后的图进行静态分析
        gm.graph.lint()
        # 对输入 input 运行符号跟踪后的模型，并获取输出
        out = gm(input)
        # 断言运行后的输出与预期输出 ref_out 相等
        self.assertEqual(out, ref_out)

        # 遍历符号跟踪后的图的节点
        for node in gm.graph.nodes:
            # 如果节点操作是 'call_function'
            if node.op == 'call_function':
                # 断言节点目标是 torch 的运算符重载 OpOverload
                assert isinstance(node.target, torch._ops.OpOverload)
                # 断言节点目标的名称是 'add.Tensor'
                assert node.target.__name__ == 'add.Tensor'

    # 定义一个名为 test_pickle_torch_custom_ops 的测试方法
    def test_pickle_torch_custom_ops(self):
        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 重写 forward 方法，调用 torch 自定义运算符 sigmoid 和 cat
            def forward(self, a):
                b = torch.ops.aten.sigmoid(a)
                c = torch.ops.aten.cat([a, b])
                return torch.ops.aten.cat((c, c))

        # 创建 M 类的实例 m
        m = M()
        # 生成一个形状为 (3,) 的随机输入张量
        input = torch.randn(3)
        # 获取 m 对输入 input 的预期输出
        ref_out = m(input)
        # 对模型 m 进行符号跟踪
        gm = symbolic_trace(m)
        # 对符号跟踪后的图进行静态分析
        gm.graph.lint()
        # 对符号跟踪后的图进行序列化
        pickled = pickle.dumps(gm)
        # 反序列化序列化后的图
        loaded = pickle.loads(pickled)
        # 断言反序列化后的图对输入 input 的输出与原图对输入 input 的输出相等
        self.assertEqual(loaded(input), gm(input))

    # 定义一个名为 test_pretty_print 的测试方法
    def test_pretty_print(self):
        # 创建 SimpleTest 类的实例 st
        st = SimpleTest()
        # 对 st 进行符号跟踪
        traced = symbolic_trace(st)
        # 对符号跟踪后的图进行静态分析
        traced.graph.lint()
        # 将符号跟踪后的模型输出转换为字符串
        printed = str(traced)
        # 断言输出字符串中包含 'SimpleTest()'
        assert 'SimpleTest()' in printed
        # 断言输出字符串中包含 'torch.relu'
        assert 'torch.relu' in printed

    # 定义一个名为 test_pretty_print_graph 的测试方法
    def test_pretty_print_graph(self):
        # 定义一个继承自 torch.nn.Module 的类 KwargPrintTest
        class KwargPrintTest(torch.nn.Module):
            # 重写 forward 方法，对输入张量 x 进行操作
            def forward(self, x):
                return torch.squeeze(x + 3.0, dim=2)

        # 创建 KwargPrintTest 类的实例 st
        st = KwargPrintTest()
        # 对 st 进行符号跟踪
        traced = symbolic_trace(st)
        # 对符号跟踪后的图进行静态分析
        traced.graph.lint()
        # 将符号跟踪后的图输出转换为字符串
        stringed = str(traced.graph)
        # 遍历字符串中的关键词列表
        for s in ['args', 'kwargs', 'num_users']:
            # 断言字符串中包含每个关键词
            assert s in stringed
    def test_custom_proxy_type(self):
        # 定义一个名为 TensorPair 的内部类，表示一个包含左右两个张量的对
        class TensorPair:
            # 初始化函数，接受左右两个张量作为参数
            def __init__(self, left, right):
                self.left, self.right = left, right

            # 定义加法操作，对两个 TensorPair 对象进行按元素加法
            def add(self, other):
                l = self.left + other.left
                r = self.right + other.right
                return TensorPair(l, r)

            # 定义乘法操作，对两个 TensorPair 对象进行按元素乘法
            def mul(self, other):
                l = self.left * other.left
                r = self.right * other.right
                return TensorPair(l, r)

        # 定义一个函数 use_tensor_pair，接受两个 TensorPair 对象作为参数
        def use_tensor_pair(x : TensorPair, y : TensorPair):
            # 对参数 x 和 y 调用 add 方法，得到结果 s
            s = x.add(y)
            # 对结果 s 和参数 x 调用 mul 方法，返回最终结果
            return s.mul(x)

        # 创建两个 TensorPair 对象 x 和 y，初始化为随机的 5x3 张量
        x = TensorPair(torch.randn(5, 3), torch.randn(5, 3))
        y = TensorPair(torch.randn(5, 3), torch.randn(5, 3))

        # 计算参考输出 ref_out，调用 use_tensor_pair 函数
        ref_out = use_tensor_pair(x, y)

        # 使用 symbolic_trace 函数对 use_tensor_pair 进行符号跟踪
        traced = symbolic_trace(use_tensor_pair)

        # 对跟踪后的函数 traced 调用，传入参数 x 和 y
        traced_out = traced(x, y)

        # 断言跟踪输出的左右张量与参考输出的左右张量相等
        self.assertEqual(traced_out.left, ref_out.left)
        self.assertEqual(traced_out.right, ref_out.right)

    def test_custom_proxy_type_literal(self):
        # 定义一个名为 TensorPair 的内部类，使用 torch.fx.ProxyableClassMeta 元类
        class TensorPair(metaclass=torch.fx.ProxyableClassMeta):
            # 初始化函数，接受左右两个张量作为参数
            def __init__(self, left, right):
                self.left, self.right = left, right

            # 定义加法操作，对两个 TensorPair 对象进行按元素加法
            def add(self, other):
                l = self.left + other.left
                r = self.right + other.right
                return TensorPair(l, r)

            # 定义乘法操作，对两个 TensorPair 对象进行按元素乘法
            def mul(self, other):
                l = self.left * other.left
                r = self.right * other.right
                return TensorPair(l, r)

        # 定义一个函数 use_tensor_pair_literal，接受一个 TensorPair 对象作为参数
        def use_tensor_pair_literal(x : TensorPair):
            # 对参数 x 调用 add 方法，使用零张量创建一个新的 TensorPair 对象
            s = x.add(TensorPair(torch.zeros(5, 3), torch.zeros(5, 3)))
            # 对结果 s 和参数 x 调用 mul 方法，返回最终结果
            return s.mul(x)

        # 创建一个 TensorPair 对象 x，初始化为随机的 5x3 张量
        x = TensorPair(torch.randn(5, 3), torch.randn(5, 3))

        # 计算参考输出 ref_out，调用 use_tensor_pair_literal 函数
        ref_out = use_tensor_pair_literal(x)

        # 使用 symbolic_trace 函数对 use_tensor_pair_literal 进行符号跟踪
        traced = symbolic_trace(use_tensor_pair_literal)

        # 对跟踪后的函数 traced 调用，传入参数 x
        traced_out = traced(x)

        # 断言跟踪输出的左右张量与参考输出的左右张量相等
        self.assertEqual(traced_out.left, ref_out.left)
        self.assertEqual(traced_out.right, ref_out.right)

    def test_custom_proxy_dynamic_value(self):
        # 定义一个名为 TensorPair 的内部类，使用 torch.fx.ProxyableClassMeta 元类
        class TensorPair(metaclass=torch.fx.ProxyableClassMeta):
            # 初始化函数，接受左右两个张量作为参数
            def __init__(self, left, right):
                self.left, self.right = left, right

            # 定义加法操作，对两个 TensorPair 对象进行按元素加法
            def add(self, other):
                l = self.left + other.left
                r = self.right + other.right
                return TensorPair(l, r)

            # 定义乘法操作，对两个 TensorPair 对象进行按元素乘法
            def mul(self, other):
                l = self.left * other.left
                r = self.right * other.right
                return TensorPair(l, r)

        # 定义一个函数 use_tensor_pair_ctor，接受一个 TensorPair 对象和一个 torch.Tensor 对象作为参数
        def use_tensor_pair_ctor(x : TensorPair, y : torch.Tensor):
            # 对参数 x 调用 add 方法，使用 y 创建一个新的 TensorPair 对象
            s = x.add(TensorPair(y, y))
            # 对结果 s 和参数 x 调用 mul 方法，返回最终结果
            return s.mul(x)

        # 创建一个 TensorPair 对象 x，初始化为随机的 5x3 张量
        # 创建一个 torch.Tensor 对象 y，初始化为随机的 5x3 张量
        x = TensorPair(torch.randn(5, 3), torch.randn(5, 3))
        y = torch.randn(5, 3)

        # 计算参考输出 ref_out，调用 use_tensor_pair_ctor 函数
        ref_out = use_tensor_pair_ctor(x, y)

        # 使用 symbolic_trace 函数对 use_tensor_pair_ctor 进行符号跟踪
        traced = symbolic_trace(use_tensor_pair_ctor)

        # 对跟踪后的函数 traced 调用，传入参数 x 和 y
        traced_out = traced(x, y)

        # 断言跟踪输出的左右张量与参考输出的左右张量相等
        self.assertEqual(traced_out.left, ref_out.left)
        self.assertEqual(traced_out.right, ref_out.right)
    def test_custom_proxy_input_dependent_control_flow(self):
        # 定义一个代理类，继承自torch.fx.ProxyableClassMeta
        class ZeroTensor(metaclass=torch.fx.ProxyableClassMeta):
            # 初始化方法，根据输入确定是否为零张量
            def __init__(self, inp):
                if inp.sum() == 0:
                    self.is_zero = True
                    self.tensor = torch.tensor([])  # 如果是零则创建空张量
                else:
                    self.is_zero = False
                    self.tensor = inp  # 否则使用输入张量

            # 定义一个加法方法，根据自身状态返回不同的张量
            def add(self, other):
                if self.is_zero:
                    return ZeroTensor(other.tensor)  # 如果自身是零，则返回另一个张量的零张量
                elif other.is_zero:
                    return self  # 如果另一个张量是零，则返回自身

        # 定义一个函数，接受两个张量并返回一个ZeroTensor对象
        def use_zero_tensor(x: torch.Tensor, y: torch.Tensor):
            return ZeroTensor(x + y)

        # 创建两个随机张量
        x, y = torch.randn(5, 3), torch.randn(5, 3)

        # 获取参考输出
        ref_out = use_zero_tensor(x, y)

        # 对use_zero_tensor函数进行符号化追踪
        traced = symbolic_trace(use_zero_tensor)

        # 使用追踪后的函数计算输出
        traced_out = traced(x, y)

        # 断言追踪后的输出与参考输出的零张量状态相同
        self.assertEqual(traced_out.is_zero, ref_out.is_zero)
        # 断言追踪后的输出与参考输出的张量数值相同
        self.assertEqual(traced_out.tensor, ref_out.tensor)

    def test_graph_fns(self):
        # 创建一个空图
        g = Graph()
        # 创建一个图节点a作为占位符
        a = g.placeholder('a')
        # 使用图调用linear模块，输入为a
        b = g.call_module('linear', (a,))
        # 获取图的bias属性
        c = g.get_attr('bias')
        # 使用图调用add方法，输入为b和c
        d = g.call_method('add', (b, c))
        # 使用图调用torch.sin函数，输入为d
        e = g.call_function(torch.sin, (d,))
        # 将e作为图的输出
        g.output(e)

        # 创建一个空的torch.nn.Module对象
        mod = torch.nn.Module()
        # 给模块添加linear属性，值为torch.nn.Linear(3, 4)
        mod.linear = torch.nn.Linear(3, 4)
        # 给模块添加bias属性，值为随机生成的长度为4的张量
        mod.bias = torch.rand(4)
        # 创建一个图模块对象，模块为mod，图为g
        gm = GraphModule(mod, g)
        # 对图模块进行检查
        gm.graph.lint()
        # 创建一个随机输入张量
        input = torch.rand(3)
        # 计算图模块的输出
        r = gm(input)
        # 计算torch.sin(mod.linear(input) + mod.bias)的结果作为参考输出
        ref = torch.sin(mod.linear(input) + mod.bias)
        # 断言计算结果与参考输出相等
        self.assertEqual(r, ref)

    def test_remove_uses(self):
        # 创建一个torch.fx.Graph对象
        g: torch.fx.Graph = Graph()
        # 创建一个图节点x作为占位符
        x: torch.fx.Node = g.placeholder('x')
        # 使用图调用torch.relu函数，输入为x
        relu: torch.fx.Node = g.call_function(torch.relu, (x,))
        # 使用图调用torch.neg函数，输入为relu
        neg: torch.fx.Node = g.call_function(torch.neg, (relu,))
        # 将neg作为图的输出
        g.output(neg)

        # 替换所有使用neg的节点为relu
        neg.replace_all_uses_with(relu)
        # 删除图中的neg节点
        g.erase_node(neg)

        # 断言neg节点不再是relu节点的用户
        self.assertTrue(neg not in relu.users)

    def test_remove_uses_with_custom_filter(self):
        # 创建一个torch.fx.Graph对象
        g: torch.fx.Graph = Graph()
        # 创建一个图节点x作为占位符
        x: torch.fx.Node = g.placeholder('x')
        # 使用图调用torch.relu函数，输入为x
        relu: torch.fx.Node = g.call_function(torch.relu, (x,))
        # 使用图调用torch.neg函数，输入为relu
        neg: torch.fx.Node = g.call_function(torch.neg, (relu,))
        # 将neg作为图的输出
        g.output(neg)

        # 使用自定义过滤器替换所有使用neg的节点为relu
        neg.replace_all_uses_with(relu, lambda x: x != neg)

        # 断言neg节点仍然是relu节点的用户
        self.assertTrue(neg in relu.users)

    def test_nonetype_annotation(self):
        # 创建一个torch.nn.EmbeddingBag对象
        eb = torch.nn.EmbeddingBag(3, 4)
        # 对EmbeddingBag对象进行符号化追踪
        symbolic_trace(eb)

    def test_pickle_nonetype_annotation(self):
        # 创建一个torch.nn.EmbeddingBag对象
        eb = torch.nn.EmbeddingBag(10, 3, mode='sum')
        # 对EmbeddingBag对象进行符号化追踪
        traced = symbolic_trace(eb)
        # 将追踪结果pickle序列化
        pickled = pickle.dumps(traced)
        # 加载pickle序列化的结果
        loaded = pickle.loads(pickled)
        # 对加载后的图进行检查
        loaded.graph.lint()
        # 创建一个随机输入张量
        input = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
        offsets = torch.LongTensor([0, 4])
        # 断言加载后的结果与追踪结果在给定输入和偏移量下的输出相等
        self.assertEqual(loaded(input, offsets), traced(input, offsets))
    def test_return_tuple(self):
        # 定义一个简单的神经网络模块类 M
        class M(torch.nn.Module):
            # 定义前向传播方法，接受一个张量 x，返回一个包含两个张量的元组
            def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                return (x, x + x)

        # 创建一个原始的 M 类实例
        original = M()
        # 对原始模块进行符号化追踪
        traced = symbolic_trace(original)
        # 断言符号化追踪后的模块对输入张量 torch.ones(1) 的输出与原始模块的前向传播结果相同
        self.assertEqual(traced(torch.ones(1)), original.forward(torch.ones(1)))

    def test_construct_root_dict(self):
        # 创建一个空的 TorchFX 图对象 graph
        graph : torch.fx.Graph = torch.fx.Graph()
        # 创建一个表示占位符节点 'x' 的 TorchFX 节点 a
        a : torch.fx.Node = graph.create_node('placeholder', 'x')
        # 创建一个表示调用模块 'foo.bar.baz' 的 TorchFX 节点 b，并传入节点 a 作为参数
        b : torch.fx.Node = graph.create_node('call_module', 'foo.bar.baz', args=(a,))
        # 创建一个表示获取属性 'zip.zap.zam' 的 TorchFX 节点 c
        c : torch.fx.Node = graph.create_node('get_attr', 'zip.zap.zam')
        # 创建一个表示调用函数 operator.add 的 TorchFX 节点 d，并传入节点 b 和 c 作为参数
        d : torch.fx.Node = graph.create_node('call_function', operator.add, args=(b, c))
        # 将节点 d 设置为图的输出节点
        graph.output(d)

        # 创建一个具有输入维度为 (3, 4) 的线性层模块 linear_mod
        linear_mod : torch.nn.Module = torch.nn.Linear(3, 4)
        # 创建一个形状为 (3, 4) 的随机张量 add_param
        add_param : torch.Tensor = torch.rand(3, 4)
        # 使用 TorchFX 创建一个图模块 gm，传入模块字典和图对象 graph
        gm : torch.fx.GraphModule = torch.fx.GraphModule(
            {'foo.bar.baz': linear_mod, 'zip.zap.zam' : add_param}, graph)
        # 对图进行静态分析（lint）
        gm.graph.lint()

        # 断言字符串 'self.foo.bar.baz' 在 gm 的代码中
        assert 'self.foo.bar.baz' in gm.code

        # 创建一个形状为 (3, 3) 的随机输入张量 x
        x : torch.Tensor = torch.rand(3, 3)
        # 将输入张量 x 输入到 gm 中得到输出张量 out
        out : torch.Tensor = gm(x)
        # 计算参考输出 ref_out，为 linear_mod(x) + add_param
        ref_out : torch.Tensor = linear_mod(x) + add_param
        # 断言 gm 对输入 x 的输出与 ref_out 相同
        self.assertEqual(out, ref_out)

    def test_symbolic_trace_assert(self):

        # 定义一个模块 AssertsTensorShape，用于检查输入张量 x 的形状
        class AssertsTensorShape(torch.nn.Module):
            def forward(self, x):
                # 断言输入张量 x 的第二维度大于 4，否则触发 AssertionError
                torch._assert(x.shape[1] > 4, "assert_foobar")
                return x

        # 创建一个 AssertsTensorShape 类的实例 m
        m = AssertsTensorShape()
        # 对 m 进行符号化追踪
        traced = symbolic_trace(m)
        # 验证符号化追踪后的模块在运行时对输入张量 torch.rand(4, 5) 的断言工作正常
        traced(torch.rand(4, 5))
        # 使用断言检查符号化追踪后的模块在输入张量 torch.rand(4, 3) 时是否触发 AssertionError
        with self.assertRaisesRegex(AssertionError, "assert_foobar"):
            traced(torch.rand(4, 3))
        # 验证符号化追踪后的模块可以转换为 Torch 脚本模块
        ms = torch.jit.script(m)
        # 使用断言检查 Torch 脚本模块在输入张量 torch.rand(4, 3) 时是否触发 AssertionError
        with self.assertRaisesRegex(torch.jit.Error, "assert_foobar"):
            ms(torch.rand(4, 3))
    def test_fx_create_arg(self):
        # 定义一个自定义的参数对象类
        class CustomArgObject:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            # 定义特殊方法，用于在 Torch FX 跟踪器中创建节点
            def __fx_create_arg__(self, tracer: torch.fx.Tracer):
                return tracer.create_node(
                    "call_function",
                    CustomArgObject,
                    args=(
                        tracer.create_arg(self.x),
                        tracer.create_arg(self.y),
                    ),
                    kwargs={},
                )

        # 定义一个继承自 torch.nn.Module 的类，其 forward 方法接收 CustomArgObject 类型参数
        class HasCustomArgObjectWhenLeaf(torch.nn.Module):
            def forward(self, o: CustomArgObject):
                # 不会被正常跟踪；将其作为叶子模块的一个好理由。
                for x in o.x:
                    o.y += x
                return o.y

        # 定义一个根模块类，内部包含 HasCustomArgObjectWhenLeaf 的实例
        class Root(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = HasCustomArgObjectWhenLeaf()

            def forward(self, x, y):
                o = CustomArgObject(x, y)
                return self.inner(o)

        # 定义一个 Torch FX 跟踪器，用于跟踪 Root 模块的计算图
        class CreateArgTracer(torch.fx.Tracer):
            def is_leaf_module(self, m, module_qualified_name):
                return type(m) is HasCustomArgObjectWhenLeaf

        # 创建 Root 模块的实例
        m = Root()
        # 对其进行 Torch FX 跟踪
        graph = CreateArgTracer().trace(m)
        # 将跟踪得到的计算图封装为 GraphModule
        gm = torch.fx.GraphModule(m, graph)
        # 断言生成的代码中包含 "CustomArgObject(" 字符串
        assert "CustomArgObject(" in gm.code

    def test_trace_fn_constant(self):
        # 定义一个常量张量
        some_constant = torch.rand(3, 4)

        # 定义一个函数，将输入张量与常量张量相加
        def add_const(x):
            return some_constant + x

        # 对函数进行符号化跟踪
        traced = symbolic_trace(add_const)

        # 创建一个输入张量
        input = torch.rand(3, 4)
        # 断言符号化跟踪的输出与直接调用函数输出一致
        self.assertEqual(traced(input), add_const(input))

    def test_copy_no_remap(self):
        # 对 SimpleTest() 进行符号化跟踪
        traced = symbolic_trace(SimpleTest())
        # 获取跟踪得到的计算图
        g = traced.graph
        # 创建一个新的空计算图
        copied = torch.fx.Graph()
        # 复制原计算图中的每个节点到新计算图
        for node in g.nodes:
            copied.node_copy(node)
        # 断言在新计算图上调用 lint() 方法会抛出 RuntimeError
        with self.assertRaisesRegex(RuntimeError, 'does not belong to this Graph'):
            copied.lint()

    def test_wrong_topo(self):
        # 创建一个空的 Torch FX 计算图
        graph: torch.fx.Graph = torch.fx.Graph()
        # 创建一个占位符节点 'a'
        a: torch.fx.Node = graph.create_node('placeholder', 'x')
        # 创建一个调用模块节点 'b'
        b: torch.fx.Node = graph.create_node('call_module', 'foo.bar.baz', args=(a,))
        # 创建一个获取属性节点 'c'
        c: torch.fx.Node = graph.create_node('get_attr', 'zip.zap.zam')
        # 创建一个调用函数节点 'd'，并将 'b' 和 'c' 作为参数
        d: torch.fx.Node = graph.create_node('call_function', operator.add, args=(b, c))
        # 将 'd' 设置为输出节点
        graph.output(d)
        # 执行拓扑排序并断言会抛出 RuntimeError
        nodes = list(graph.nodes)
        nodes[3].append(nodes[2])
        with self.assertRaisesRegex(RuntimeError, 'was used before it has been defined'):
            graph.lint()

    def test_wrong_target_type(self):
        # 创建一个空的 Torch FX 计算图
        graph: torch.fx.Graph = torch.fx.Graph()
        # 使用错误的值 'foo' 创建一个节点，断言会抛出 ValueError
        with self.assertRaises(ValueError):
            n = torch.fx.Node(graph=graph, name='foo', op='call_function', target='foo',
                              args=(), kwargs={})
    def test_example_shape_prop(self):
        # 定义一个内嵌的测试用例类，继承自 torch.nn.Module
        class TestCase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个形状为 (3, 4) 的张量作为属性 attr
                self.attr = torch.randn(3, 4)
                # 初始化一个线性层，输入输出维度均为 4
                self.submod = torch.nn.Linear(4, 4)

            def forward(self, x):
                # 模型的前向传播：对输入应用 ReLU，然后经过 submod 线性层并取负值
                return torch.neg(self.submod(x.relu() + self.attr))

        # 创建 TestCase 类的实例 tc
        tc = TestCase()
        # 对 tc 进行符号跟踪（symbolic_trace），得到一个跟踪后的模型 tc_traced
        tc_traced = symbolic_trace(tc)
        # 使用随机数据进行 tc_traced 模型的前向传播，得到参考输出 ref_out
        ref_out = tc_traced(torch.rand(3, 4))
        # 创建一个 ShapeProp 对象，传播输入数据的形状信息到 traced 模型
        shape_prop.ShapeProp(tc_traced).propagate(torch.rand(3, 4))

        # 确保测试覆盖了所有的操作码（opcodes）
        opcodes = set()
        # 初始化输出形状和步长为 None
        output_shape: Optional[torch.Shape] = None
        output_stride: Optional[Tuple[int]] = None
        # 遍历 tc_traced 模型图中的每个节点
        for node in tc_traced.graph.nodes:
            opcodes.add(node.op)
            # 如果节点的操作码是 'output'
            if node.op == 'output':
                # 获取输出节点的形状信息
                output_shape = node.args[0].meta['tensor_meta'].shape
                output_stride = node.args[0].meta['tensor_meta'].stride
        # 断言测试所有操作码都被覆盖到
        self.assertEqual(opcodes, {'placeholder', 'get_attr', 'call_function', 'call_method',
                                   'call_module', 'output'})

        # 测试形状传播，并确保结果与实际输出匹配
        self.assertEqual(output_shape, ref_out.shape)
        self.assertEqual(output_stride, ref_out.stride())

    def test_shape_prop_layout(self):
        # 定义一个简单的卷积测试类 ConvTest，继承自 torch.nn.Module
        class ConvTest(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个 Conv2d 模块，输入通道数、输出通道数和卷积核大小分别为 5、5、3
                self.conv_mod = torch.nn.Conv2d(5, 5, 3)

            def forward(self, x):
                # 执行卷积操作
                return self.conv_mod(x)

        # 创建 ConvTest 类的实例 test_mod
        test_mod = ConvTest()
        # 对 test_mod 进行符号跟踪，得到跟踪后的模型 traced
        traced = symbolic_trace(test_mod)
        # 创建一个形状为 (5, 5, 224, 224) 的随机输入张量 x
        x = torch.randn(5, 5, 224, 224)
        # 使用 ShapeProp 对象传播输入张量 x 的形状信息到 traced 模型
        shape_prop.ShapeProp(traced).propagate(x)

        # 断言所有节点的内存格式为 torch.contiguous_format
        assert all(node.meta['tensor_meta'].memory_format is torch.contiguous_format
                   for node in traced.graph.nodes)

        # 创建一个按 channels_last 格式重新排列的输入张量 x_channels_last
        x_channels_last = x.contiguous(memory_format=torch.channels_last)
        # 将 traced 模型转换为 channels_last 内存格式
        traced.to(memory_format=torch.channels_last)
        # 使用 ShapeProp 对象传播输入张量 x_channels_last 的形状信息到 traced 模型
        shape_prop.ShapeProp(traced).propagate(x_channels_last)
        # 遍历 traced 模型图中的每个节点
        for node in traced.graph.nodes:
            # 注意：卷积的实现可能不会保留内存格式
            # 我们只能检查占位符节点是否 channels_last
            if node.op in {'placeholder'}:
                # 断言节点的内存格式为 torch.channels_last
                self.assertEqual(node.meta['tensor_meta'].memory_format, torch.channels_last)
    # 定义测试方法：验证形状属性聚合
    def test_shape_prop_aggregate(self):
        # 定义返回两个值的模块
        class ReturnTwo(torch.nn.Module):
            # 前向传播方法，返回一个元组，第一个元素为常数3，第二个元素为输入张量 x 的和
            def forward(self, x):
                return (3, torch.sum(x))

        # 定义被测试的模块
        class UnderTest(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 设置 ReturnTwo 实例作为属性
                self.rt = ReturnTwo()

            # 前向传播方法，调用 ReturnTwo 实例的前向传播
            def forward(self, x):
                return self.rt(x)

        # 创建 UnderTest 实例
        ut = UnderTest()

        # 定义追踪 ReturnTwo 模块的追踪器类
        class RTTracer(torch.fx.Tracer):
            # 判断是否为叶子模块的方法，根据模块类型是否为 ReturnTwo 判断
            def is_leaf_module(self, m, module_qualified_name):
                return type(m) is ReturnTwo

        # 对 UnderTest 模块进行追踪，获得计算图
        graph = RTTracer().trace(ut)
        # 创建 GraphModule 实例
        mod = torch.fx.GraphModule(ut, graph)

        # 创建形状属性传播器，传播给定形状的随机张量
        shape_prop.ShapeProp(mod).propagate(torch.rand(3, 4))

        # 遍历计算图中的每个节点
        for node in mod.graph.nodes:
            # 如果节点操作为 'call_module'
            if node.op == 'call_module':
                # 断言节点的元数据中包含 'tensor_meta'
                assert 'tensor_meta' in node.meta
                # 获取 tensor_meta
                tensor_meta = node.meta['tensor_meta']
                # 断言第一个元素为 3
                assert tensor_meta[0] == 3
                # 断言第二个元素的形状为空
                assert tensor_meta[1].shape == torch.Size([])
    
    # 定义测试方法：验证三维布局的形状属性
    def test_shape_prop_layout_3d(self):
        # 定义包含三维卷积的模块
        class ConvTest3d(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 设置包含 3D 卷积层的模块
                self.conv_mod = torch.nn.Conv3d(5, 5, 3)

            # 前向传播方法，调用 3D 卷积层
            def forward(self, x):
                return self.conv_mod(x)

        # 创建 ConvTest3d 实例
        test_mod_3d = ConvTest3d()
        # 对模型进行符号追踪
        traced_3d = symbolic_trace(test_mod_3d)
        # 创建形状属性传播器，传播给定形状的随机三维张量
        x_3d = torch.randn(5, 5, 224, 224, 15)
        shape_prop.ShapeProp(traced_3d).propagate(x_3d)

        # 断言所有节点的元数据中的 memory_format 是连续格式
        assert all(node.meta['tensor_meta'].memory_format is torch.contiguous_format
                   for node in traced_3d.graph.nodes)

        # 将输入张量转换为 channels_last_3d 格式
        x_channels_last_3d = x_3d.contiguous(memory_format=torch.channels_last_3d)
        # 将模型转换为 channels_last_3d 格式
        traced_3d.to(memory_format=torch.channels_last_3d)
        # 再次进行形状属性传播
        shape_prop.ShapeProp(traced_3d).propagate(x_channels_last_3d)
        # 遍历计算图中的每个节点
        for node in traced_3d.graph.nodes:
            # 如果节点操作为 'placeholder'
            if node.op in {'placeholder'}:
                # 断言节点的元数据中的 memory_format 是 channels_last_3d
                self.assertEqual(node.meta['tensor_meta'].memory_format, torch.channels_last_3d)
    def test_nn_module_stack(self):
        # 定义一个内部类 SubModule，继承自 torch.nn.Module
        class SubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在 SubModule 中定义一个 2D 卷积层，输入通道数和输出通道数都为 64，卷积核大小为 3x3，填充为 1，无偏置
                self.conv_mod = torch.nn.Conv2d(64, 64, (3, 3), padding=1, bias=False)

            # 实现 SubModule 的前向传播方法
            def forward(self, x):
                return self.conv_mod(x)

        # 定义一个内部类 MyModule，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在 MyModule 中创建一个 SubModule 的实例
                self.sub_mod = SubModule()

            # 实现 MyModule 的前向传播方法
            def forward(self, x):
                return self.sub_mod(x)

        # 创建一个 MyModule 的实例 m
        m = MyModule()
        # 使用 symbolic_trace 函数对模型 m 进行符号化跟踪，得到图形模块 gm
        gm = torch.fx.symbolic_trace(m)

        # 初始化一个空的模块栈字典 mod_stack
        mod_stack = {}
        # 预期的模块栈，包含了预期的子模块路径及其类型信息
        expected_stack = [('sub_mod', ('sub_mod', type(m.sub_mod))),
                          ('sub_mod.conv_mod', ('sub_mod.conv_mod', type(m.sub_mod.conv_mod)))]
        # 遍历 gm 图的每个节点
        for node in gm.graph.nodes:
            # 获取节点元数据中的 nn_module_stack，如果存在则赋值给 mod_stack
            mod_stack = node.meta.get('nn_module_stack', {})
            # 如果 mod_stack 不为空，则跳出循环
            if mod_stack:
                break
        # 将 mod_stack 转换为列表 stack_list
        stack_list = list(mod_stack.items())
        # 使用断言函数验证 stack_list 是否与 expected_stack 相等
        self.assertEqual(stack_list, expected_stack)

    def test_transformer_preserves_nn_module_stack_for_get_attr(self):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在 M 中定义一个参数 weight，类型为 torch.nn.Parameter，值为全1的 1x1 张量
                self.weight = torch.nn.Parameter(torch.ones(1, 1))

            # 实现 M 的前向传播方法
            def forward(self, x):
                return self.weight + x

        # 创建一个 Tracer 实例 tracer
        tracer = torch.fx.Tracer()
        # 使用 tracer 对 M 类进行追踪，得到图形对象 graph
        graph = tracer.trace(M())
        # 创建一个 GraphModule 实例 gm，将 tracer 的根节点和 graph 传入
        gm = GraphModule(tracer.root, graph)
        # 遍历 gm 图的每个节点
        for node in gm.graph.nodes:
            # 如果节点操作为 'get_attr'
            if node.op == 'get_attr':
                # 设置节点的 nn_module_stack 元数据为 "self"
                node.meta["nn_module_stack"] = "self"
                # 设置节点的 stack_trace 元数据为 "stack_trace"
                node.meta["stack_trace"] = "stack_trace"
                # 设置节点的 source_fn_stack 元数据为 "source_fn_stack"
                node.meta["source_fn_stack"] = "source_fn_stack"
        # 创建一个 Transformer 实例，传入 gm，生成新的 GraphModule new_gm
        new_gm = Transformer(gm).transform()
        # 再次遍历 new_gm 图的每个节点
        for node in new_gm.graph.nodes:
            # 如果节点操作为 'get_attr'
            if node.op == 'get_attr':
                # 使用断言函数验证节点的 nn_module_stack 元数据是否为 "self"
                self.assertEqual(node.meta["nn_module_stack"], "self")
                # 使用断言函数验证节点的 stack_trace 元数据是否为 "stack_trace"
                self.assertEqual(node.meta["stack_trace"], "stack_trace")
                # 使用断言函数验证节点的 source_fn_stack 元数据是否为 "source_fn_stack"
                self.assertEqual(node.meta["source_fn_stack"], "source_fn_stack")

    def test_interpreter(self):
        # 定义一个内部类 MyModule，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在 MyModule 中定义一个参数 param，类型为 torch.nn.Parameter，值为形状为 (3, 4) 的随机张量
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                # 在 MyModule 中定义一个全连接层 linear，输入特征数为 4，输出特征数为 5
                self.linear = torch.nn.Linear(4, 5)

            # 实现 MyModule 的前向传播方法
            def forward(self, x):
                # 返回线性层的输出，使用 clamp 函数将结果限制在 0 到 1 之间
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)

        # 创建一个 MyModule 的实例 m
        m = MyModule()
        # 使用 symbolic_trace 函数对模型 m 进行符号化跟踪，得到图形模块 gm
        gm = torch.fx.symbolic_trace(m)

        # 创建一个 Interpreter 实例 interpreter，传入 gm
        interpreter = Interpreter(gm)
        # 创建一个形状为 (3, 4) 的随机输入张量 input
        input = torch.randn(3, 4)
        # 使用断言函数验证 interpreter 运行输入 input 后的输出是否与 gm 直接运行输入 input 的结果相等
        self.assertEqual(interpreter.run(input), gm(input))
        # 使用断言函数验证 interpreter 运行输入 input 后的输出是否与模型 m 直接运行输入 input 的结果相等
        self.assertEqual(interpreter.run(input), m(input))
    def test_interpreter_other_graph(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))  # 初始化模块参数为一个3x4的随机张量
                self.linear = torch.nn.Linear(4, 5)  # 创建一个线性层，输入维度4，输出维度5

            def forward(self, x):
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)  # 模块前向传播函数，对输入x加上参数self.param并经过线性层和clamp操作

        m = MyModule()  # 创建MyModule的实例
        gm = torch.fx.symbolic_trace(m)  # 对模块m进行符号跟踪

        interpreter = Interpreter(gm, graph=gm.graph)  # 创建Interpreter对象，用于执行符号跟踪后的图gm
        input = torch.randn(3, 4)  # 创建一个3x4的随机张量作为输入
        self.assertEqual(interpreter.run(input), gm(input))  # 断言Interpreter对象运行的结果与gm(input)的结果相等
        self.assertEqual(interpreter.run(input), m(input))  # 断言Interpreter对象运行的结果与模块m运行input的结果相等

    def test_interpreter_run_node_override(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))  # 初始化模块参数为一个3x4的随机张量
                self.linear = torch.nn.Linear(4, 5)  # 创建一个线性层，输入维度4，输出维度5

            def forward(self, x):
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)  # 模块前向传播函数，对输入x加上参数self.param并经过线性层和clamp操作

        m = MyModule()  # 创建MyModule的实例
        gm = torch.fx.symbolic_trace(m)  # 对模块m进行符号跟踪

        class RunNodeInterpreter(Interpreter):
            def __init__(self, module):
                super().__init__(module)

            def run_node(self, n : Node) -> Any:
                result = super().run_node(n)  # 调用父类Interpreter的run_node方法执行节点n
                n.cached_value = result  # 将节点n的计算结果缓存到n.cached_value中
                return result

        input = torch.randn(3, 4)  # 创建一个3x4的随机张量作为输入
        RunNodeInterpreter(gm).run(input)  # 使用RunNodeInterpreter执行符号跟踪后的图gm，并运行输入input
        for node in gm.graph.nodes:  # 遍历gm图中的所有节点
            assert hasattr(node, 'cached_value')  # 断言节点node是否具有'cached_value'属性

    def test_interpreter_onthefly_swap(self):

        def fn(x):
            return torch.sigmoid(x).neg()  # 定义一个函数fn，对输入x先进行sigmoid操作再取负

        gm = torch.fx.symbolic_trace(fn)  # 对函数fn进行符号跟踪，得到图gm

        class NegSigmSwapInterpreter(Interpreter):
            def call_function(self, target : Target, args : Tuple, kwargs : Dict) -> Any:
                if target == torch.sigmoid:  # 如果目标函数是torch.sigmoid
                    return torch.neg(*args, **kwargs)  # 则返回torch.neg对应的操作
                return super().call_function(n)  # 如果不是，调用父类Interpreter的call_function方法

            def call_method(self, target : Target, args : Tuple, kwargs : Dict) -> Any:
                if target == 'neg':  # 如果目标方法是'neg'
                    call_self, *args_tail = args
                    return call_self.sigmoid(*args_tail, **kwargs)  # 则返回call_self.sigmoid对应的操作
                return super().call_method(n)  # 如果不是，调用父类Interpreter的call_method方法

        input = torch.randn(3, 4)  # 创建一个3x4的随机张量作为输入
        result = NegSigmSwapInterpreter(gm).run(input)  # 使用NegSigmSwapInterpreter执行符号跟踪后的图gm，并运行输入input
        self.assertEqual(result, torch.neg(input).sigmoid())  # 断言执行结果与torch.neg(input).sigmoid()的结果相等
    def test_interpreter_partial_eval(self):
        # 定义一个简单的神经网络模块，包含一个参数和一个线性层
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))  # 初始化一个参数张量
                self.linear = torch.nn.Linear(4, 5)  # 定义一个线性层

            def forward(self, x):
                # 前向传播函数，对输入进行线性变换并进行ReLU激活函数处理
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)

        gm = torch.fx.symbolic_trace(MyModule())  # 对模块进行符号跟踪
        interp = Interpreter(gm)  # 创建解释器对象
        env = {}
        # 遍历图中的每个节点
        for node in gm.graph.nodes:
            # 如果节点是调用模块且目标为'linear'
            if node.op == 'call_module' and node.target == 'linear':
                env[node] = torch.arange(0, 12, 1).reshape(3, 4) - 6.0  # 设置线性层的环境变量
                break  # 找到一个后即停止
        assert len(env) == 1  # 确保环境变量中有且仅有一个条目
        x = torch.randn(3, 4)  # 创建输入张量
        result = interp.run(x, initial_env=env)  # 运行解释器
        self.assertEqual(result, (torch.arange(0, 12, 1).reshape(3, 4) - 6.0).clamp(0.0, 1.0))  # 断言输出与预期结果一致

    def test_interpreter_star_args(self):
        # 定义一个带有星号参数的简单函数
        def with_star_args(x, *args):
            return x + args[0]  # 返回输入张量与第一个星号参数的和

        gm = torch.fx.symbolic_trace(with_star_args)  # 对函数进行符号跟踪
        interp = Interpreter(gm)  # 创建解释器对象
        result = interp.run(torch.ones(3, 4), torch.ones(3, 4), torch.rand(3, 4))  # 运行解释器，传入多个参数
        self.assertEqual(result, torch.ones(3, 4) * 2.0)  # 断言输出与预期结果一致

    @skipIfNoTorchVision
    def test_interpreter_noop_resnet18(self):
        rn18 = torchvision_models.resnet18()  # 创建一个ResNet-18模型
        transformed = torch.fx.Transformer(symbolic_trace(rn18)).transform()  # 对模型进行符号跟踪和转换
        inp = torch.randn(5, 3, 224, 224)  # 创建输入张量
        self.assertEqual(transformed(inp), rn18(inp))  # 断言变换后的模型与原始模型在输入上的输出一致

    @skipIfNoTorchVision
    def test_interpreter_gc_values(self):
        rn18 = torchvision_models.resnet18()  # 创建一个ResNet-18模型
        interp = Interpreter(symbolic_trace(rn18))  # 对模型进行符号跟踪并创建解释器对象
        inp = torch.rand(5, 3, 224, 224)  # 创建输入张量
        out = interp.run(inp)  # 运行解释器
        env_key_names = {n.name for n in interp.env.keys()}  # 获取解释器环境中的键名集合
        self.assertEqual(env_key_names, {'output'})  # 断言环境中只有一个键名为'output'

    def test_interpreter_default_args(self):
        # 定义一个带有默认参数的简单模块类
        class Model(torch.nn.Module):
            def forward(self, x, y=3.14159):
                return x + y  # 返回输入张量与默认参数的和

        model = Model()  # 创建模型对象
        gm = torch.fx.symbolic_trace(model)  # 对模型进行符号跟踪

        interp = Interpreter(gm)  # 创建解释器对象
        x = torch.randn(5, 3)  # 创建输入张量
        out = interp.run(x)  # 运行解释器
        torch.testing.assert_close(out, x + 3.14159)  # 断言输出与预期结果一致

    def test_interpreter_not_enough_args(self):
        # 定义一个带有两个参数的简单模块类
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y  # 返回输入张量与第二个参数的和

        model = Model()  # 创建模型对象
        gm = torch.fx.symbolic_trace(model)  # 对模型进行符号跟踪

        interp = Interpreter(gm)  # 创建解释器对象
        x = torch.randn(5, 3)  # 创建输入张量
        # 使用断言捕获异常，检查运行时错误消息是否包含特定字符串
        with self.assertRaisesRegex(RuntimeError,
                                    'Expected positional argument for parameter y, but one was not passed in'):
            out = interp.run(x)  # 运行解释器
    def test_transformer_noop(self):
        # 定义一个继承自 torch.nn.Module 的子类 MyModule
        class MyModule(torch.nn.Module):
            # 初始化函数
            def __init__(self):
                super().__init__()
                # 创建一个形状为 (3, 4) 的随机参数，并将其包装为可学习的参数
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                # 创建一个线性层，输入维度为 4，输出维度为 5
                self.linear = torch.nn.Linear(4, 5)

            # 前向传播函数
            def forward(self, x):
                # 返回经过线性层的计算结果，同时对结果进行了夹紧操作，使其在 [0.0, 1.0] 范围内
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)

        # 创建 MyModule 的实例
        m = MyModule()
        # 对 MyModule 进行符号化追踪
        gm = torch.fx.symbolic_trace(m)

        # 使用 Transformer 类对符号化追踪后的模型 gm 进行变换
        new_gm = Transformer(gm).transform()

        # 创建输入张量
        input = torch.randn(3, 4)
        # 断言新模型 new_gm 对相同输入 input 的输出与原始模型 gm 的输出相同
        self.assertEqual(new_gm(input), gm(input))

    def test_transformer_op_swap(self):

        # 定义一个简单的函数 fn，使用 torch.fx.symbolic_trace 进行符号化追踪
        def fn(x):
            return torch.sigmoid(x).neg()

        gm = torch.fx.symbolic_trace(fn)

        # 定义一个自定义的变换器类 NegSigmSwapXformer，继承自 Transformer
        class NegSigmSwapXformer(Transformer):
            # 重写 call_function 方法
            def call_function(self, target : Target, args : Tuple, kwargs : Dict) -> Any:
                # 如果目标函数是 torch.sigmoid，则将其转换为 torch.neg
                if target == torch.sigmoid:
                    return torch.neg(*args, **kwargs)
                # 否则调用父类的 call_function 方法
                return super().call_function(n)  # noqa: F821

            # 重写 call_method 方法
            def call_method(self, target : Target, args : Tuple, kwargs : Dict) -> Any:
                # 如果目标方法是 'neg'，则调用它的反向操作
                if target == 'neg':
                    call_self, *args_tail = args
                    return call_self.sigmoid(*args_tail, **kwargs)
                # 否则调用父类的 call_method 方法
                return super().call_method(n)  # noqa: F821

        # 对 gm 应用 NegSigmSwapXformer 变换
        transformed = NegSigmSwapXformer(gm).transform()
        # 创建输入张量
        input = torch.randn(3, 4)
        # 断言经过变换后的函数对相同输入 input 的输出与 torch.neg(input).sigmoid() 相同
        self.assertEqual(transformed(input), torch.neg(input).sigmoid())

    def test_transformer_multi_outputs(self):
        # 定义一个包含多个输出的神经网络模块 MyModule
        class MyModule(torch.nn.Module):
            # 初始化函数
            def __init__(self):
                super().__init__()
                # 创建一个形状为 (3, 4) 的随机参数，并将其包装为可学习的参数
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                # 创建一个线性层，输入维度为 4，输出维度为 5
                self.linear = torch.nn.Linear(4, 5)

            # 前向传播函数
            def forward(self, x):
                # 对输入张量 x 加上参数 self.param
                x = x + self.param
                # 经过线性层的计算，得到输出 out
                out = self.linear(x)
                # 返回修改后的输入 x 和计算结果 out
                return x, out

        # 创建 MyModule 的实例
        m = MyModule()
        # 对 MyModule 进行符号化追踪
        gm = torch.fx.symbolic_trace(m)

        # 使用 Transformer 类对符号化追踪后的模型 gm 进行变换
        new_gm = Transformer(gm).transform()

        # 创建输入张量
        input = torch.randn(3, 4)
        # 断言新模型 new_gm 对相同输入 input 的输出与原始模型 gm 的输出相同
        self.assertEqual(new_gm(input), gm(input))

    def test_fn_type_annotations(self):
        # 定义一个神经网络模块 Foo
        class Foo(torch.nn.Module):
            # 前向传播函数，接受参数 p（一个 Pair 类型对象）、z（一个张量）、i（一个整数），返回字典
            def forward(self, p : Pair, z : torch.Tensor, i : int) -> Dict[str, torch.Tensor]:
                return {'a': p.x + p.y + z + i}

        # 使用 torch.jit.script 对 Foo 进行脚本化
        foo_scripted = torch.jit.script(Foo())
        # 调用脚本化后的 Foo 对象，传入合适的参数
        foo_scripted(Pair(torch.rand(5), torch.rand(5)), torch.rand(5), 3)

        # 对 Foo 使用 symbolic_trace 进行符号化追踪
        fxed = symbolic_trace(Foo())
        # 对符号化追踪后的 Foo 对象再次进行脚本化
        fxed_scripted = torch.jit.script(fxed)
        # 调用脚本化后的 fxed_scripted 对象，传入合适的参数
        fxed_scripted(Pair(torch.rand(5), torch.rand(5)), torch.rand(5), 3)

    def test_fn_type_annotation_empty(self):
        # 定义一个简单的函数 forward，接受一个列表参数 a，返回列表中的第一个张量
        def forward(a : List[torch.Tensor]):
            return a[0]
        # 对 forward 函数进行 symbolic_trace，然后再脚本化
        torch.jit.script(symbolic_trace(forward))
    def test_wrapped_method(self):
        # 定义一个装饰器函数，用于包装传入的函数 fn，并在其结果上应用 ReLU 激活函数
        def wrap_with_relu(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return torch.relu(fn(*args, **kwargs))
            return wrapper

        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            # 在 forward 方法上应用 wrap_with_relu 装饰器，将 forward 方法包装为带有 ReLU 的版本
            @wrap_with_relu
            def forward(self, x, w):
                return torch.matmul(x, w)

        # 创建 Foo 类的实例 f
        f = Foo()
        # 对 f 进行符号化追踪
        traced = symbolic_trace(f)
        # 创建两个随机张量 x 和 w
        x, w = torch.rand(3, 4), torch.rand(4, 4)
        # 断言：在追踪得到的图中，至少有一个节点的目标是 torch.relu 函数
        self.assertTrue(any(n.target == torch.relu for n in traced.graph.nodes))

    def test_empty_graph_codegen(self):
        # 创建一个空的 torch.fx.Graph 对象
        graph = torch.fx.Graph()
        # 使用这个空图创建一个 GraphModule，其模型为一个空的 torch.nn.Module
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        # 断言：调用 gm() 返回 None
        self.assertEqual(gm(), None)

    def test_sequential(self):
        # 创建一个包含单个 Conv2d 层的序列模型
        m = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1))
        # 对模型 m 进行符号化追踪
        gm = torch.fx.symbolic_trace(m)
        # 深拷贝符号化追踪得到的图
        gm_copy = copy.deepcopy(gm)

    def test_ctx_mgr(self):
        # 定义一个上下文管理器 do_nothing，它不做任何操作
        @contextlib.contextmanager
        def do_nothing():
            yield

        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 在 forward 方法上应用 do_nothing 上下文管理器，实际不进行任何操作
            @do_nothing()
            def forward(self, x):
                return torch.relu(x)

        # 创建 M 类的实例 m
        m = M()
        # 调用 self.checkGraphModule 验证 m 模型的图结构
        self.checkGraphModule(m, (torch.rand(3, 4),))

    def test_typename_print(self):
        # 创建一个空的 torch.fx.Graph 对象，并将其类型指定为 torch.fx.Graph
        graph: torch.fx.Graph = torch.fx.Graph()
        # 创建一个表示占位符节点 'x'
        x: torch.fx.Node = graph.create_node('placeholder', 'x')
        # 创建一个调用函数节点，目标为 torch.relu，参数为 x，类型为 List[float]
        b: torch.fx.Node = graph.create_node('call_function', target=torch.relu, args=(x,),
                                             type_expr=List[float])
        # 将节点 b 设置为图的输出节点
        output: torch.fx.Node = graph.output(b)
        # 断言：在 graph 的字符串表示中，包含 'typing.List[float]' 这个类型信息
        self.assertTrue('typing.List[float]' in str(graph))

    def test_layout(self):
        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 在 forward 方法中创建一个和输入张量 x 同样布局和内存的空张量，并填充为 0
            def forward(self, x):
                return torch.empty_like(x, layout=torch.strided, pin_memory=False).fill_(0)

        # 对 M 类的实例进行符号化追踪
        traced = symbolic_trace(M())
        # 创建一个大小为 (5, 9, 3, 4) 的随机张量 x
        x = torch.rand(5, 9, 3, 4)
        # 断言：调用 traced(x) 返回一个与 x 相同大小且元素全部为 0 的张量
        self.assertEqual(traced(x), torch.zeros_like(x))

    def test_ellipsis(self):
        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 在 forward 方法中，返回 x 加上 y 的第二到第十行数据
            def forward(self, x, y):
                return x + y[:, 1:10, ...]

        # 对 M 类的实例进行符号化追踪
        traced = symbolic_trace(M())
        # 创建两个随机张量 x 和 y
        x, y = torch.rand(5, 9, 3, 4), torch.rand(5, 15, 3, 4)
        # 断言：调用 traced(x, y) 返回 x 加上 y 的第二到第十行数据的结果
        self.assertEqual(traced(x, y), x + y[:, 1:10, ...])

    def test_inf_nan(self):
        # 定义一个继承自 torch.nn.Module 的类 FooMod
        class FooMod(torch.nn.Module):
            # 在 forward 方法中，返回输入 x 加上正无穷、负无穷和 NaN
            def forward(self, x):
                return x + float('inf'), x + float('-inf'), x + float('nan')

        # 创建 FooMod 类的实例 fm
        fm = FooMod()
        # 调用 self.checkGraphModule 验证 fm 模型的图结构
        self.checkGraphModule(fm, (torch.rand(3, 4),))
    # 定义一个测试函数，测试处理无穷大和NaN的情况
    def test_inf_nan_kwds(self):
        # 创建一个空的 Torch FX 图对象
        graph : torch.fx.Graph = torch.fx.Graph()
        # 在图中创建一个占位符节点 'x'
        x : torch.fx.Node = graph.create_node('placeholder', 'x')
        # 在图中创建一个调用函数节点，调用 operator.add 函数，将 'x' 和无穷大(float('inf'))相加，命名为 'inf'
        b : torch.fx.Node = graph.create_node('call_function', operator.add, (x, float('inf')), {}, name='inf')
        # 在图中创建一个调用函数节点，调用 operator.add 函数，将 'x' 和 NaN(float('nan'))相加，命名为 'nan'
        c : torch.fx.Node = graph.create_node('call_function', operator.add, (x, float('nan')), {}, name='nan')
        # 设置图的输出节点为 'inf' 和 'nan'
        graph.output((b, c))

        # 使用 Torch FX 创建一个图模块，使用空的神经网络模块作为输入
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        # 生成一个随机张量 'x'，形状为 (3, 4)
        x = torch.rand(3, 4)
        # 断言图模块对 'x' 的处理结果，应分别为 (x + float('inf'), x + float('nan'))
        self.assertEqual(gm(x), (x + float('inf'), x + float('nan')))

    # 测试深拷贝递归深度
    def test_deepcopy_recursion_depth(self):
        # 获取系统递归限制加上20的深度
        depth = sys.getrecursionlimit() + 20

        # 创建一个 Torch FX 图对象
        g = torch.fx.Graph()
        # 创建一个占位符 'x'
        x = g.placeholder('x')
        # 循环创建深度次数的 torch.relu 节点
        for i in range(depth):
            x = g.call_function(torch.relu, (x,))
        # 设置图的输出节点为 'x'
        g.output(x)

        # 深拷贝图对象 g
        copied_graph = copy.deepcopy(g)

        # 创建原始节点和拷贝节点之间的映射关系字典
        val_map = {}
        for orig_node, new_node in zip(g.nodes, copied_graph.nodes):
            val_map[orig_node] = new_node

        # 验证每个节点的用户使用情况在拷贝后保持一致
        for orig_node, new_node in zip(g.nodes, copied_graph.nodes):
            orig_users = set(orig_node.users.keys())
            orig_users_equiv = {val_map[u] for u in orig_users}
            new_users = set(new_node.users.keys())
            self.assertEqual(orig_users_equiv, new_users)

    # 如果没有安装 TorchVision 则跳过此测试
    @skipIfNoTorchVision
    def test_replace_uses(self):
        # 使用 torchvision 中的 ResNet-18 模型
        rn18 = torchvision_models.resnet18()

        # 定义一个自定义的 Torch FX 跟踪器类 LowerReluTracer
        class LowerReluTracer(torch.fx.Tracer):
            # 判断是否为叶子模块，排除 torch.nn.ReLU 类型模块
            def is_leaf_module(self, m : torch.nn.Module, qualname : str):
                if isinstance(m, torch.nn.ReLU):
                    return False
                return super().is_leaf_module(m, qualname)

        # 使用 LowerReluTracer 类对 rn18 模型进行跟踪，并封装成 GraphModule 对象
        rn18_traced = GraphModule(rn18, LowerReluTracer().trace(rn18))

        # 需要删除的节点列表
        to_erase = []
        # 遍历图中的节点
        for node in rn18_traced.graph.nodes:
            # 如果节点是调用函数类型且目标函数是 torch.relu 或 torch.nn.functional.relu
            if node.op == 'call_function' and node.target in [torch.relu, torch.nn.functional.relu]:
                kwargs = node.kwargs.copy()
                # 删除 kwargs 中的 'inplace' 参数
                kwargs.pop('inplace')
                # 在节点之前插入一个新的调用函数节点，调用 torch.neg 函数
                with rn18_traced.graph.inserting_before(node):
                    new_node = rn18_traced.graph.call_function(
                        the_function=torch.neg, args=node.args, kwargs=node.kwargs)
                # 替换原始节点的所有使用节点为新节点
                node.replace_all_uses_with(replace_with=new_node)
                # 将原始节点添加到待删除列表中
                to_erase.append(node)

        # 删除待删除列表中的节点
        for node in to_erase:
            rn18_traced.graph.erase_node(node)
    def test_replace_input(self):
        # 创建一个空的 Torch FX 图对象
        graph : torch.fx.Graph = torch.fx.Graph()
        # 在图中创建一个名为 'x' 的占位符节点
        x : torch.fx.Node = graph.create_node('placeholder', 'x')
        # 在图中创建一个名为 'y' 的占位符节点
        y : torch.fx.Node = graph.create_node('placeholder', 'y')
        # 在图中创建一个调用 torch.relu 函数的节点 b，输入为节点 x
        b : torch.fx.Node = graph.create_node('call_function', target=torch.relu, args=(x,))
        # 将节点 b 设置为图的输出节点
        output : torch.fx.Node = graph.output(b)

        # 使用节点 b 将节点 x 替换为节点 y
        b.replace_input_with(x, y)

        # 将当前的 Torch FX 图对象包装成一个 Torch Module
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        # 创建输入数据 input_x 和 input_y
        input_x = torch.randn(33, 44)
        input_y = torch.randn(11, 22)
        # 断言使用图模块 gm 对输入 input_x 和 input_y 进行处理后的输出等于 torch.relu(input_y)
        self.assertEqual(gm(input_x, input_y), torch.relu(input_y))

    def test_insertion_point(self):
        # 创建一个空的 Torch FX 图对象
        graph : torch.fx.Graph = torch.fx.Graph()
        # 在图中创建一个名为 'x' 的占位符节点
        x : torch.fx.Node = graph.create_node('placeholder', 'x')
        # 在图中创建一个调用 torch.relu 函数的节点 b，输入为节点 x
        b : torch.fx.Node = graph.create_node('call_function', target=torch.relu, args=(x,))
        # 将节点 b 设置为图的输出节点
        output : torch.fx.Node = graph.output(b)

        # 在节点 b 之前插入一个新的节点 neg，该节点调用 torch.neg 函数对节点 x 进行操作
        with graph.inserting_before(b):
            neg : torch.fx.Node = graph.call_function(the_function=torch.neg, args=(x,))
            _, *relu_args = b.args
            b.args = (neg, *relu_args)

        # 将当前的 Torch FX 图对象包装成一个 Torch Module
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        # 创建输入数据 input
        input = torch.randn(33, 44)
        # 断言使用图模块 gm 对输入 input 进行处理后的输出等于 torch.relu(torch.neg(input))
        self.assertEqual(gm(input), torch.relu(torch.neg(input)))

    def test_update_args_api(self):
        # 创建一个空的 Torch FX 图对象
        graph : torch.fx.Graph = torch.fx.Graph()
        # 在图中创建一个名为 'x' 的占位符节点
        x : torch.fx.Node = graph.create_node('placeholder', 'x')
        # 在图中创建一个名为 'y' 的占位符节点
        y : torch.fx.Node = graph.create_node('placeholder', 'y')
        # 在图中创建一个调用 torch.relu 函数的节点 b，输入为节点 x
        b : torch.fx.Node = graph.create_node('call_function', target=torch.relu, args=(x,))
        # 将节点 b 设置为图的输出节点
        output : torch.fx.Node = graph.output(b)

        # 创建初始的 Torch FX 图模块 orig_gm
        orig_gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        # 创建输入数据 inp_x 和 inp_y
        inp_x, inp_y = torch.randn(5, 3), torch.randn(3, 5)
        # 断言使用图模块 orig_gm 对输入 inp_x 和 inp_y 进行处理后的输出等于 torch.relu(inp_x)
        self.assertEqual(orig_gm(inp_x, inp_y), torch.relu(inp_x))

        # 使用节点 b 更新其第一个参数为节点 y
        b.update_arg(0, y)
        # 创建更新后的 Torch FX 图模块 new_gm
        new_gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        # 断言使用图模块 new_gm 对输入 inp_x 和 inp_y 进行处理后的输出等于 torch.relu(inp_y)
        self.assertEqual(new_gm(inp_x, inp_y), torch.relu(inp_y))

    def test_update_kwargs_api(self):
        # 创建一个空的 Torch FX 图对象
        graph : torch.fx.Graph = torch.fx.Graph()
        # 在图中创建一个名为 'x' 的占位符节点
        x : torch.fx.Node = graph.create_node('placeholder', 'x')
        # 在图中创建一个名为 'y' 的占位符节点
        y : torch.fx.Node = graph.create_node('placeholder', 'y')
        # 在图中创建一个调用 torch.relu 函数的节点 b，关键字参数为 {'input': x}
        b : torch.fx.Node = graph.create_node('call_function', target=torch.relu, kwargs={'input': x})
        # 将节点 b 设置为图的输出节点
        output : torch.fx.Node = graph.output(b)

        # 创建初始的 Torch FX 图模块 orig_gm
        orig_gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        # 创建输入数据 inp_x 和 inp_y
        inp_x, inp_y = torch.randn(5, 3), torch.randn(3, 5)
        # 断言使用图模块 orig_gm 对输入 inp_x 和 inp_y 进行处理后的输出等于 torch.relu(inp_x)
        self.assertEqual(orig_gm(inp_x, inp_y), torch.relu(inp_x))

        # 使用节点 b 更新其关键字参数 'input' 为节点 y
        b.update_kwarg('input', y)
        # 创建更新后的 Torch FX 图模块 new_gm
        new_gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        # 断言使用图模块 new_gm 对输入 inp_x 和 inp_y 进行处理后的输出等于 torch.relu(inp_y)
        self.assertEqual(new_gm(inp_x, inp_y), torch.relu(inp_y))
    # 测试不可变列表的 PyTree 操作
    def test_immutable_list_pytree_ops(self):
        # 创建一个随机张量
        rand_tensor = torch.randn(5, 3)
        # 使用不可变列表创建对象 l
        l = immutable_list([3, [rand_tensor, 42]])

        # 扁平化树结构并返回扁平化后的列表 flattened 和结构 spec
        flattened, spec = pytree.tree_flatten(l)
        # 断言扁平化后的列表应该与期望的列表相等
        assert flattened == [3, rand_tensor, 42]

        # 使用 spec 将扁平化的列表 unflattened 还原为原始结构 l
        unflattened = pytree.tree_unflatten(flattened, spec)
        # 断言还原后的对象应该与原始对象 l 相等
        assert unflattened == l
        # 断言还原后的对象应该是不可变列表类型
        assert isinstance(unflattened, immutable_list)

    # 测试不可变字典的 PyTree 操作
    def test_immutable_dict_pytree_ops(self):
        # 创建一个随机张量
        rand_tensor = torch.randn(5, 3)
        # 使用不可变字典创建对象 d
        d = immutable_dict({'a': 3, 'b': [rand_tensor, 42]})

        # 扁平化树结构并返回扁平化后的列表 flattened 和结构 spec
        flattened, spec = pytree.tree_flatten(d)
        # 断言扁平化后的列表应该与期望的列表相等
        assert flattened == [3, rand_tensor, 42]

        # 使用 spec 将扁平化的列表 unflattened 还原为原始结构 d
        unflattened = pytree.tree_unflatten(flattened, spec)
        # 断言还原后的对象应该与原始对象 d 相等
        assert unflattened == d
        # 断言还原后的对象应该是不可变字典类型
        assert isinstance(unflattened, immutable_dict)

    # 测试在图中节点前面插入操作
    def test_move_before(self):
        # 创建一个空的 torch.fx.Graph 图形对象
        graph: torch.fx.Graph = torch.fx.Graph()
        # 创建一个占位符节点 'x'
        x: torch.fx.Node = graph.create_node('placeholder', 'x')
        # 创建一个调用函数节点，目标函数为 torch.relu，参数为 x
        b: torch.fx.Node = graph.create_node('call_function', target=torch.relu, args=(x,))
        # 设置输出节点为节点 b
        output: torch.fx.Node = graph.output(b)

        # 创建一个调用 torch.neg 函数的节点 neg，并将其插入到节点 b 前面
        neg: torch.fx.Node = graph.call_function(the_function=torch.neg, args=(x,))
        # 获取除了第一个参数以外的其余参数并保存到 relu_args 中
        _, *relu_args = b.args
        # 将节点 b 的参数替换为 (neg, *relu_args)
        b.args = (neg, *relu_args)
        # 将节点 neg 插入到节点 b 前面
        b.prepend(neg)

        # 使用 torch.fx.GraphModule 创建一个图模块 gm，包含图形对象 graph
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        # 创建一个随机输入张量 input
        input = torch.randn(33, 44)
        # 断言图模块 gm 对输入 input 的计算结果应该与 torch.relu(torch.neg(input)) 相等
        self.assertEqual(gm(input), torch.relu(torch.neg(input)))

    # 测试在节点自身前面插入操作
    def test_prepend_self(self):
        # 创建一个空的 torch.fx.Graph 图形对象
        graph: torch.fx.Graph = torch.fx.Graph()
        # 创建一个占位符节点 'x'
        x: torch.fx.Node = graph.create_node('placeholder', 'x')
        # 创建一个调用函数节点，目标函数为 torch.relu，参数为 x
        b: torch.fx.Node = graph.create_node('call_function', target=torch.relu, args=(x,))
        # 设置输出节点为节点 b
        output: torch.fx.Node = graph.output(b)

        # 将节点 b 插入到自身的前面
        b.prepend(b)
        # 将节点 b 追加到节点 x 的后面
        x.append(b)
        # 断言图中节点的数量应该为 3
        self.assertEqual(len(graph.nodes), 3)

    # 测试删除节点时的错误处理
    def test_erase_node_error(self):
        # 创建一个 SimpleTest 的实例 st
        st = SimpleTest()
        # 对 st 进行符号跟踪并获取其图形对象 traced
        traced = symbolic_trace(st)

        # 遍历 traced 图形对象中的节点
        for node in traced.graph.nodes:
            # 如果节点的目标函数是 operator.add 或者 torch.relu
            if node.target in [operator.add, torch.relu]:
                # 断言删除节点 node 时会抛出 RuntimeError 异常，并且异常信息包含特定的错误信息
                with self.assertRaisesRegex(RuntimeError, 'but it still had .* users in the graph'):
                    traced.graph.erase_node(node)

    # 测试深拷贝不可变对象
    def test_copy_it(self):
        # 创建一个不可变字典对象 d
        d = immutable_dict([(3, 4), (5, 6)])
        # 创建一个不可变列表对象 l
        l = immutable_list([(3, 4), (5, 6)])

        # 断言深拷贝后的对象与原对象 d 相等
        self.assertEqual(d, deepcopy(d))
        # 断言深拷贝后的对象与原对象 l 相等
        self.assertEqual(l, deepcopy(l))

    # 测试获取 torch 函数签名
    def test_get_torch_func_signature(self):
        # 遍历 torch 模块中的所有属性名称
        for key in dir(torch):
            # 获取属性名称对应的对象
            obj = getattr(torch, key)
            # 如果对象是可调用的
            if callable(obj):
                # 获取该对象的 torch 操作的签名 schemas
                schemas = get_signature_for_torch_op(obj)
    # 定义一个测试方法，用于测试找到使用情况的功能
    def test_find_uses(self):
        # 创建一个空的 TorchFX 图形对象
        graph = torch.fx.Graph()
        # 创建一个代理对象 x，代表图中的占位符 'x'
        x = torch.fx.Proxy(graph.placeholder('x'))

        # 在图中应用 relu 激活函数到 x
        y = torch.relu(x)
        # 将 x 与自身相加
        z = x + x
        # 对 x 应用取反操作
        u = torch.neg(x)
        # 将 (y + z + u) 的节点设置为图的输出节点
        graph.output((y + z + u).node)
        # 对图进行 lint 检查
        graph.lint()

        # 获取使用 x 的节点列表
        users_of_x = x.node.users
        # 断言使用 x 的节点数量为 3
        self.assertEqual(len(users_of_x), 3)
        # 期望的操作集合包括 'relu', 'add', 'neg'
        expected_ops = {'relu', 'add', 'neg'}
        # 遍历每个使用 x 的节点
        for use in users_of_x:
            # 断言至少有一个使用的节点名称以期望的操作前缀开头
            assert any(use.name.startswith(prefix) for prefix in expected_ops)

    # 定义一个测试方法，用于测试内联图的功能
    def test_inline_graph(self):
        # 定义一个内联到的类 InlineInto
        class InlineInto(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        # 定义一个待内联的类 ToInline
        class ToInline(torch.nn.Module):
            def forward(self, x):
                return torch.neg(x)

        # 对 InlineInto 类进行符号跟踪
        inline_into = symbolic_trace(InlineInto())
        # 对 ToInline 类进行符号跟踪
        to_inline = symbolic_trace(ToInline())

        # 创建一个组合图对象
        combined_graph = torch.fx.Graph()
        # 复制 inline_into 的图形到组合图中，并返回输出节点
        output_node = combined_graph.graph_copy(inline_into.graph, {})

        # 获取 to_inline 图的第一个节点，该节点应为占位符
        input_node = next(iter(to_inline.graph.nodes))
        # 断言 input_node 存在且其操作为 'placeholder'
        assert input_node and input_node.op == 'placeholder'

        # 值映射将 input_node 映射到 output_node
        val_map = {input_node : output_node}
        # 将 to_inline 的图形复制到组合图中，并使用值映射
        output = combined_graph.graph_copy(to_inline.graph, val_map)
        # 设置组合图的输出节点为复制后的节点
        combined_graph.output(output)

        # 创建一个图形模块对象，包含组合图和空的 Torch 模块
        combined_module = torch.fx.GraphModule(torch.nn.Module(), combined_graph)

        # 创建输入张量 input
        input = torch.rand(3, 4)
        # 断言组合模块的输出等于 input 应用 relu 后再取反的结果
        self.assertEqual(combined_module(input), input.relu().neg())

    # 定义一个测试方法，用于测试多个插入点的功能
    def test_multi_insert_point(self):
        # 创建一个空的 TorchFX 图形对象
        graph = torch.fx.Graph()
        # 创建一个代理对象 x，代表图中的占位符 'x'
        x = torch.fx.Proxy(graph.placeholder('x'))
        # 对 x 应用 relu 激活函数
        relu = torch.relu(x)

        # 在 relu 节点之前插入代码块
        with graph.inserting_before(relu.node):
            # 对 x 应用取反操作，并赋值给 y
            y = torch.neg(x)
            # 对 y 应用双曲正切函数，并赋值给 z
            z = torch.tanh(y)

        # 将 relu 和 z 的节点设置为图的输出节点
        graph.output((relu.node, z.node))
        # 对图进行 lint 检查
        graph.lint()

        # 期望的操作顺序为 'x', 'neg', 'tanh', 'relu'
        expected_ops = ['x', 'neg', 'tanh', 'relu']
        # 遍历图中的每个节点和期望的操作
        for node, expected in zip(graph.nodes, expected_ops):
            # 断言每个节点的名称包含期望的操作名
            assert expected in node.name

    # 定义一个测试方法，用于测试重新分配参数和关键字参数的使用情况
    def test_reassign_args_kwargs_uses(self):
        # 创建一个空的 TorchFX 图形对象
        graph = torch.fx.Graph()
        # 创建代理对象 x 和 y，代表图中的占位符 'x' 和 'y'
        x, y = Proxy(graph.placeholder('x')), Proxy(graph.placeholder('y'))
        # 对 x 和 y 应用加法操作，并赋值给 z
        z = x + y
        # 对 z 进行三次加法操作，并赋值给 zed
        zed = z + z + z
        # 将 zed 的节点设置为图的输出节点
        graph.output(zed.node)
        # 对图进行 lint 检查
        graph.lint()

        # 修改 zed 的参数，将第二个参数修改为 x
        zed.node.args = (zed.node.args[0], x.node)
        # 断言 x 节点的用户列表为 [zed.node]
        self.assertEqual(list(x.node.users.keys()), [zed.node])

        # 修改 z 的参数，将两个参数都修改为 y
        z.node.args = (y.node, y.node)
        # 断言 x 节点的用户列表为 [zed.node]
        self.assertEqual(list(x.node.users.keys()), [zed.node])

    # 定义一个测试方法，用于测试跟踪函数的功能
    def test_trace_function(self):
        # 定义一个简单的函数 foo，对输入张量 x 应用 relu 后加上 y
        def foo(x, y):
            return torch.relu(x) + y

        # 创建两个随机张量 x 和 y
        x, y = torch.randn(3, 4), torch.randn(3, 4)
        # 调用 checkGraphModule 方法来检查函数 foo 的图形模块
        self.checkGraphModule(foo, (x, y))
    def test_trace_return_dataclass(self):
        """
        Test case for Module that return dataclass
        """
        from dataclasses import dataclass  # 导入 dataclass 模块

        @dataclass
        class MyOutput:
            foo: torch.Tensor  # 定义 dataclass MyOutput，包含字段 foo
            bar: torch.Tensor  # 定义 dataclass MyOutput，包含字段 bar

        class ModuleReturnDataclass(torch.nn.Module):
            def forward(self, d : torch.Tensor):
                return MyOutput(foo=d + d, bar=d * 3)  # 返回 MyOutput 类型的对象

        module = ModuleReturnDataclass()  # 创建 ModuleReturnDataclass 实例
        traced_graph = symbolic_trace(module).graph  # 对 module 进行符号化跟踪，获取跟踪图
        print(traced_graph)  # 打印跟踪图

        gm = GraphModule(module, traced_graph)  # 创建 GraphModule 对象
        x = torch.rand(1)  # 创建一个大小为 1 的随机张量

        self.assertEqual(module(x), gm(x))  # 断言 module(x) 与 gm(x) 返回值相等

    def test_trace_return_dataclass_nested(self):
        """
        Test case for Module that return dataclass
        """
        from dataclasses import dataclass  # 导入 dataclass 模块

        @dataclass
        class MyOutput:
            foo: torch.Tensor  # 定义 dataclass MyOutput，包含字段 foo
            bar: torch.Tensor  # 定义 dataclass MyOutput，包含字段 bar

        class ModuleReturnDataclass(torch.nn.Module):
            def forward(self, d : torch.Tensor):
                return MyOutput(foo=d + d, bar=d * 3)  # 返回 MyOutput 类型的对象

        class CallsModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = ModuleReturnDataclass()  # 创建 ModuleReturnDataclass 实例

            def forward(self, x):
                tmp = self.m(x)  # 调用 self.m 的 forward 方法
                return MyOutput(foo=tmp.foo, bar=tmp.bar)  # 返回 MyOutput 类型的对象

        module = CallsModule()  # 创建 CallsModule 实例
        traced_graph = symbolic_trace(module).graph  # 对 module 进行符号化跟踪，获取跟踪图
        print(traced_graph)  # 打印跟踪图

        gm = GraphModule(module, traced_graph)  # 创建 GraphModule 对象
        x = torch.rand(1)  # 创建一个大小为 1 的随机张量

        self.assertEqual(module(x), gm(x))  # 断言 module(x) 与 gm(x) 返回值相等

    def test_trace_return_namedtuple(self):
        """
        Test case for Module that return namedtuple
        """
        class MyOutput(NamedTuple):
            foo: torch.Tensor  # 定义 NamedTuple MyOutput，包含字段 foo
            bar: torch.Tensor  # 定义 NamedTuple MyOutput，包含字段 bar

        class ModuleReturnNamedTuple(torch.nn.Module):
            def forward(self, d : torch.Tensor):
                return MyOutput(foo=d, bar=d)  # 返回 MyOutput 类型的对象

        module = ModuleReturnNamedTuple()  # 创建 ModuleReturnNamedTuple 实例

        traced_graph = symbolic_trace(module).graph  # 对 module 进行符号化跟踪，获取跟踪图
        print(traced_graph)  # 打印跟踪图

        gm = GraphModule(module, traced_graph)  # 创建 GraphModule 对象
        x = torch.rand(1)  # 创建一个大小为 1 的随机张量

        self.assertEqual(module(x), gm(x))  # 断言 module(x) 与 gm(x) 返回值相等

    def test_trace_dict_int_keys(self):
        class ModWithDictArg(torch.nn.Module):
            def forward(self, d : Dict[int, torch.Tensor]):
                return d[42]  # 返回字典 d 中键为 42 的值

        class CallsModWithDict(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = ModWithDictArg()  # 创建 ModWithDictArg 实例

            def forward(self, x):
                return self.m({42: x})  # 调用 self.m 的 forward 方法，并传入字典 {42: x}

        class MyTracer(torch.fx.Tracer):
            def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
                return isinstance(m, ModWithDictArg)  # 判断 m 是否为 ModWithDictArg 实例

        traced_graph = MyTracer().trace(CallsModWithDict())  # 使用 MyTracer 对 CallsModWithDict 进行跟踪
    def test_trace_dict_proxy_keys(self):
        class ModWithDictArg(torch.nn.Module):
            def forward(self, d : Dict[torch.Tensor, torch.Tensor]):
                return d[42]
        
        class CallsModWithDict(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = ModWithDictArg()

            def forward(self, x):
                return self.m({x: x})

        class MyTracer(torch.fx.Tracer):
            def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
                return isinstance(m, ModWithDictArg)

        # 测试用例：检查是否会抛出 RuntimeError，包含 'cannot contain a Node' 字符串
        with self.assertRaisesRegex(RuntimeError, 'cannot contain a Node'):
            # 使用 MyTracer 对 CallsModWithDict 进行跟踪
            traced_graph = MyTracer().trace(CallsModWithDict())

    def test_module_deepcopy_edit_nodes(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        # 对 Foo 模块进行符号化跟踪
        traced1 = symbolic_trace(Foo())
        # 深拷贝 traced1
        copied = copy.deepcopy(traced1)

        # 修改深拷贝后图中的节点
        for node in copied.graph.nodes:
            if node.target == torch.relu:
                node.target = torch.neg

        # 重新编译修改后的拷贝和原始 traced1
        copied.recompile()
        traced1.recompile()

        x = torch.randn(15, 15)
        # 检查两个模块在输入 x 下的输出是否一致
        torch.testing.assert_close(traced1(x), torch.relu(x))
        torch.testing.assert_close(copied(x), torch.neg(x))

    def test_direct_param_use(self):
        class TransposeTest(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.b = torch.nn.Parameter(torch.rand(4, 3))

            def forward(self, x):
                return self.b

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = TransposeTest()

            def forward(self, x):
                return self.a.b, self.a.b.t(), self.a.b.view(12)

        # 对 Foo 模块进行符号化跟踪
        traced = torch.fx.symbolic_trace(Foo())
        # 检查 traced 图中是否所有节点的 target 中不含有 'constant' 字符串
        assert all('constant' not in node.target for node in traced.graph.nodes)

    def test_single_default_arg(self):
        class M(torch.nn.Module):
            def forward(self, y=1):
                return y

        m = M()
        # 使用 self.checkGraphModule 方法检查模块 m 在不同参数下的行为
        self.checkGraphModule(m, ())
        self.checkGraphModule(m, (3,))

    def test_multiple_default_args(self):
        class M(torch.nn.Module):
            def forward(self, y=1, z=2):
                return y + z

        m = M()
        # 使用 self.checkGraphModule 方法检查模块 m 在不同参数下的行为
        self.checkGraphModule(m, ())
        self.checkGraphModule(m, (3,))
        self.checkGraphModule(m, (3, 4))

    def test_regular_and_default_args(self):
        class M(torch.nn.Module):
            def forward(self, x, y=1):
                return x + y

        m = M()
        # 使用 self.checkGraphModule 方法检查模块 m 在不同参数下的行为
        self.checkGraphModule(m, (2,))
        self.checkGraphModule(m, (2, 3))

    def test_string_literal_return(self):
        class M(torch.nn.Module):
            def forward(self):
                return "foo"

        m = M()
        # 使用 self.checkGraphModule 方法检查模块 m 在不同参数下的行为
        self.checkGraphModule(m, ())
    # 定义测试方法，验证返回值中的限定名称是否正确
    def test_namedtuple_return_qualname(self):
        # 定义一个继承自 torch.nn.Module 的类 NamedTupReturn
        class NamedTupReturn(torch.nn.Module):
            # 实现 forward 方法，返回一个自定义命名元组 MyNamedTup 对象
            def forward(self, x):
                return MyNamedTup(x, x)

        # 对 NamedTupReturn 类进行符号化追踪
        traced = symbolic_trace(NamedTupReturn())
        # 创建一个形状为 (3, 4) 的随机输入张量
        input = torch.rand(3, 4)
        # 断言符号化追踪结果与预期的 MyNamedTup(input, input) 相等
        self.assertEqual(traced(input), MyNamedTup(input, input))

    # 定义测试方法，验证更新 args 和 kwargs 抛出 AttributeError 异常
    def test_update_args_kwargs_yells_at_you(self):
        # 对 SimpleTest 类进行符号化追踪
        symtraced = symbolic_trace(SimpleTest())
        # 获取追踪图中的第一个节点
        node = next(iter(symtraced.graph.nodes))
        # 使用上下文断言捕获 AttributeError 异常，异常信息包含 '__update_args_kwargs'
        with self.assertRaisesRegex(AttributeError, '__update_args_kwargs'):
            node.__update_args_kwargs((), {})

    # 定义测试方法，验证在 FX 中使用 torchbind 类的属性
    def test_torchbind_class_attribute_in_fx(self):
        # 如果是在 FBCODE、Windows 或 macOS 中运行，则跳过测试
        if IS_FBCODE or IS_WINDOWS or IS_MACOS:
            self.skipTest("torch.classes._TorchScriptTesting._StackString is registered, skipping")

        # 定义一个继承自 torch.nn.Module 的类 FooBar1234
        class FooBar1234(torch.nn.Module):
            # 初始化方法中创建一个 torchbind 类的实例
            def __init__(self):
                super().__init__()
                # 初始化 self.f 为 torch.classes._TorchScriptTesting._StackString 类的实例
                self.f = torch.classes._TorchScriptTesting._StackString(["3", "4"])

            # 实现 forward 方法，返回 self.f 的顶部元素
            def forward(self):
                return self.f.top()

        # 创建 FooBar1234 类的实例 m
        m = FooBar1234()
        # 使用 self.checkGraphModule 方法检查图模块
        self.checkGraphModule(m, ())

    # 定义测试方法，验证在 FX 中使用带有张量参数的 torchbind 类的属性
    def test_torchbind_class_attribute_in_fx_tensor_arg(self):
        # 如果是在 FBCODE、Windows 或 macOS 中运行，则跳过测试
        if IS_FBCODE or IS_WINDOWS or IS_MACOS:
            self.skipTest("torch.classes._TorchScriptTesting._ReLUClass is registered, skipping")

        # 定义一个继承自 torch.nn.Module 的类 FooBar2341
        class FooBar2341(torch.nn.Module):
            # 初始化方法中创建一个 torchbind 类的实例
            def __init__(self):
                super().__init__()
                # 初始化 self.f 为 torch.classes._TorchScriptTesting._ReLUClass 类的实例
                self.f = torch.classes._TorchScriptTesting._ReLUClass()

            # 实现 forward 方法，调用 self.f 的 run 方法并返回结果
            def forward(self, x):
                return self.f.run(x)

        # 创建 FooBar2341 类的实例 m
        m = FooBar2341()
        # 对类实例 m 进行符号化追踪
        traced = symbolic_trace(m)
        # 创建一个形状为 (3, 4) 的随机输入张量
        input = torch.randn(3, 4)
        # 断言符号化追踪结果与原始方法调用结果相等
        self.assertEqual(traced(input), m(input))
        # 断言在追踪图的节点中至少有一个操作为 'call_method'
        self.assertTrue(any(n.op == 'call_method' for n in traced.graph.nodes))

    # 定义测试方法，验证脚本方法的符号化追踪
    def test_script_method_trace(self):
        # 定义一个继承自 torch.nn.Module 的类 Scripted
        class Scripted(torch.nn.Module):
            # 实现 forward 方法，应用 torch.relu 函数并返回结果
            def forward(self, x):
                return torch.relu(x)

        # 定义一个继承自 torch.nn.Module 的类 Holder
        class Holder(torch.nn.Module):
            # 初始化方法中将 Scripted 类脚本化，并赋值给 self.s
            def __init__(self):
                super().__init__()
                self.s = torch.jit.script(Scripted())

            # 实现 forward 方法，调用 self.s 方法并返回结果
            def forward(self, x):
                return self.s(x)

        # 创建 Holder 类的实例 h
        h = Holder()
        # 对实例 h 进行符号化追踪
        traced = symbolic_trace(h)
        # 创建一个形状为 (3, 4) 的随机输入张量
        input = torch.randn(3, 4)
        # 断言符号化追踪结果与原始方法调用结果相等
        self.assertEqual(traced(input), h(input))
        # 断言在追踪图的节点中至少有一个操作为 'call_method'
        self.assertTrue(any(n.op == 'call_method' for n in traced.graph.nodes))

    # 定义测试方法，验证返回命名元组的符号化追踪
    def test_namedtuple_return_trace(self):
        # 定义一个继承自 torch.nn.Module 的类 NamedTupReturn
        class NamedTupReturn(torch.nn.Module):
            # 实现 forward 方法，返回一个自定义命名元组 Pair 对象
            def forward(self, x):
                return Pair(x, x)

        # 对 NamedTupReturn 类进行符号化追踪
        traced = symbolic_trace(NamedTupReturn())
        # 创建一个形状为 (3, 4) 的随机输入张量
        input = torch.rand(3, 4)
        # 断言符号化追踪结果与预期的 Pair(input, input) 相等
        self.assertEqual(traced(input), Pair(input, input))
    def test_named_tuple_inlined(self):
        # 定义一个继承自torch.nn.Module的命名元组模块
        class NamedTupMod(torch.nn.Module):
            # 重写forward方法，接受输入inp，并返回包装后的命名元组
            def forward(self, inp):
                return wrapped_named_tup(Pair(inp, 1.2), p2=Pair(3.4, inp))

        # 创建NamedTupMod实例
        m = NamedTupMod()
        # 创建一个大小为(3, 4)的随机张量作为输入
        input = torch.rand(3, 4)
        # 调用模型m，得到输出ref
        ref = m(input)
        # 对模型进行符号化追踪
        traced = symbolic_trace(m)

        # 使用traced对输入input进行调用，得到输出res
        res = traced(input)
        # 断言ref与res相等
        self.assertEqual(ref, res)

        # 在图的节点中检查Pair命名元组是否在内联的函数调用中起作用
        ph = call_func = None
        for node in traced.graph.nodes:
            # 如果节点的操作是"placeholder"，则将ph设置为该节点
            if node.op == "placeholder":
                ph = node
            # 如果节点的操作是"call_function"且目标函数是wrapped_named_tup，则更新其参数和关键字参数
            elif node.op == "call_function" and node.target == wrapped_named_tup:
                node.update_arg(0, Pair(ph, 1.2))
                node.update_kwarg("p2", Pair(3.4, ph))
                call_func = node
                break
        # 断言call_func不为None
        self.assertTrue(call_func is not None)
        # 断言call_func.args[0]是Pair类型
        self.assertTrue(isinstance(call_func.args[0], Pair))
        # 断言call_func.kwargs["p2"]是Pair类型
        self.assertTrue(isinstance(call_func.kwargs["p2"], Pair))
        # 断言_format_arg函数对call_func.args[0]的输出为"Pair(x=%inp, y=1.2)"
        self.assertEqual(_format_arg(call_func.args[0]), "Pair(x=%inp, y=1.2)")
        # 断言_format_arg函数对call_func.kwargs["p2"]的输出为"Pair(x=3.4, y=%inp)"
        self.assertEqual(_format_arg(call_func.kwargs["p2"]), "Pair(x=3.4, y=%inp)")

        # 消除死代码
        traced.graph.eliminate_dead_code()
        # 重新编译追踪模型
        traced.recompile()
        # 使用重新编译后的追踪模型对输入input进行调用，得到输出res
        res = traced(input)
        # 断言ref与res相等
        self.assertEqual(ref, res)

    def test_return_type_exists(self):
        # 定义一个继承自torch.nn.Module的返回类型模块
        class ReturnTypeModule(torch.nn.Module):
            # 定义一个接受List[str]类型参数x，并返回List[str]类型的方法other
            def other(self, x: List[str]) -> List[str]:
                return x

            # 重写forward方法，接受List[str]类型参数x，并返回List[str]类型的结果
            def forward(self, x: List[str]) -> List[str]:
                return self.other(x)

        # 对ReturnTypeModule进行符号化追踪
        traced = symbolic_trace(ReturnTypeModule())
        # 断言在traced的代码中存在"-> typing_List[str]"字符串
        self.assertIn("-> typing_List[str]", traced._code)
        # 将traced转换为torch脚本
        scripted = torch.jit.script(traced)
        # 断言在scripted的代码中存在"-> List[str]"字符串
        self.assertIn("-> List[str]", scripted.code)

    def getitem_inner(self):
        # 定义一个继承自torch.nn.Module的基础getitem类GetItemBase
        class GetItemBase(torch.nn.Module):
            # 在构造函数中注册一个名为'pe'的缓冲区，内容为8x8的随机张量
            def __init__(self):
                super().__init__()
                self.register_buffer('pe', torch.randn(8, 8))

        # 定义一个继承自GetItemBase的getitem类GetItem1
        class GetItem1(GetItemBase):
            # 重写forward方法，接受输入x，并返回self.pe的部分切片
            def forward(self, x):
                return self.pe[:, :x.size(0)]

        # 定义一个继承自GetItemBase的getitem类GetItem2
        class GetItem2(GetItemBase):
            # 重写forward方法，接受输入x，并返回self.pe的某个索引处的值
            def forward(self, x):
                return self.pe[x.size(0)]

        # 定义一个继承自GetItemBase的getitem类GetItem3
        class GetItem3(GetItemBase):
            # 重写forward方法，固定返回self.pe的索引为4处的值
            def forward(self, x):
                return self.pe[4]  # fx creates `self._tensor_constant0` here

        # 分别对GetItem1、GetItem2和GetItem3实例进行图模块检查
        self.checkGraphModule(GetItem1(), [torch.zeros(4)])
        self.checkGraphModule(GetItem2(), [torch.zeros(4)])
        self.checkGraphModule(GetItem3(), [torch.zeros(4)])

    @unittest.skipUnless(os.environ.get("FX_PATCH_GETITEM") == "1",
                         "Will be checked in test_getitem_subproc")
    def test_getitem(self):
        # 调用getitem_inner函数
        self.getitem_inner()
    # 定义一个测试方法，用于在子进程中运行以解决特定问题
    def test_getitem_subproc(self):
        # 需要在子进程中运行此测试以解决问题：
        # https://github.com/pytorch/pytorch/issues/50710
        proc = Process(target=run_getitem_target)  # 创建一个进程对象，目标函数是 run_getitem_target
        proc.start()  # 启动进程
        proc.join()  # 等待进程结束
        self.assertEqual(proc.exitcode, 0)  # 断言子进程的退出码为0，表示成功

    # 定义一个测试方法，验证在函数调用过程中的编译问题
    def test_user_friendly_call_provenance_with_function(self):
        def fn(x):
            return wrapper_fn(x)

        traced = torch.fx.symbolic_trace(fn)  # 对函数 fn 进行符号跟踪

        with self.assertRaisesRegex(RuntimeError, "'wrapper_fn' is "
                                    "being compiled since it was called"
                                    " from 'fn.forward'"):
            scripted = torch.jit.script(traced)  # 对跟踪结果进行脚本化编译

    # 定义一个测试方法，验证在模块调用过程中的编译问题
    def test_user_friendly_call_provenance_with_module(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return wrapper_fn(x)

        traced = torch.fx.symbolic_trace(M())  # 对模块 M 进行符号跟踪

        with self.assertRaisesRegex(RuntimeError, "'wrapper_fn' is "
                                    "being compiled since it was called"
                                    " from 'M.forward'"):
            scripted = torch.jit.script(traced)  # 对跟踪结果进行脚本化编译

    # 定义一个测试方法，验证在模块中使用不同命名规范的激活函数
    def test_snake_case(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.activations = torch.nn.ModuleDict([
                    ["snake_case", torch.nn.ReLU()],
                    ["PascalCase", torch.nn.LeakyReLU()],
                    ["ALL_CAPS", torch.nn.PReLU()]
                ])

            def forward(self, x):
                a = self.activations["snake_case"](x)
                b = self.activations["PascalCase"](x)
                c = self.activations["ALL_CAPS"](x)
                return a, b, c

        traced = symbolic_trace(M())  # 对模块 M 进行符号跟踪

        check = [
            ("activations_snake_case", "activations.snake_case"),
            ("activations_pascal_case", "activations.PascalCase"),
            ("activations_all_caps", "activations.ALL_CAPS")
        ]

        i = 0
        for node in traced.graph.nodes:
            if node.op == "placeholder" or node.op == "output":
                continue
            name = check[i][0]
            target = check[i][1]
            self.assertEqual(name, node.name)  # 断言节点的名称符合预期
            self.assertEqual(target, node.target)  # 断言节点的目标符合预期
            i += 1
        self.assertEqual(i, 3)  # 最终断言处理过的节点数为3个

    # 定义一个测试方法，验证禁止对不可变列表进行修改的情况
    def test_no_mutation(self):
        from torch.fx.immutable_collections import immutable_list
        x = immutable_list([3, 4])  # 创建一个不可变列表 x
        with self.assertRaisesRegex(NotImplementedError, "new_args"):
            x[0] = 4  # 尝试修改不可变列表的元素，预期抛出 NotImplementedError 异常
    def test_partial_trace(self):
        # 定义一个内部类 Foo，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x, y):
                # 如果 y 为真，则返回 2 * x
                if y:
                    return 2 * x
                else:
                    # 否则返回 x
                    return x
        # 创建 Foo 类的实例 mod
        mod = Foo()
        # 对模型 mod 进行符号化追踪，传入 concrete_args={'y': True}，得到 mod_true
        mod_true = symbolic_trace(mod, concrete_args={'y': True})
        # 对模型 mod 进行符号化追踪，传入 concrete_args={'y': False}，得到 mod_false
        mod_false = symbolic_trace(mod, concrete_args={'y': False})
        # 断言 mod_true 在输入 (3, True) 时的输出为 6
        self.assertEqual(mod_true(3, True), 6)
        # 打印 mod_true 的代码表示
        print(mod_true.code)
        # 断言 mod_true 的图中存在至少一个节点的目标是 torch._assert
        assert any(i.target == torch._assert for i in mod_true.graph.nodes)
        # 使用断言检测 mod_true 在输入 (3, False) 时是否会引发 AssertionError
        with self.assertRaises(AssertionError):
            mod_true(3, False)
        # 断言 mod_false 在输入 (3, False) 时的输出为 3
        self.assertEqual(mod_false(3, False), 3)
        # 使用断言检测 mod_false 在输入 (3, True) 时是否会引发 AssertionError
        with self.assertRaises(AssertionError):
            mod_false(3, True)

        # 定义一个高阶函数 f_higher，接受参数 a 和 f，并返回 f(a)
        def f_higher(a, f):
            return f(a)

        # 对函数 f_higher 进行符号化追踪，传入 concrete_args={'f': lambda x: x * 2}，得到 nf
        nf = symbolic_trace(f_higher, concrete_args={'f': lambda x: x * 2})
        # 断言 nf 在输入 (3, lambda x: x * 2) 时的输出为 6
        self.assertEqual(nf(3, lambda x: x * 2), 6)

    def test_custom_traceback_raised_when_exception_source_is_graphmodule(self):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 定义模型 M 的初始化方法
            def __init__(self):
                super().__init__()
                # 添加一个形状为 (5,) 的参数 W
                self.W = torch.nn.Parameter(torch.randn(5))

            # 定义模型 M 的前向传播方法，计算 W 与 x 的点积
            def forward(self, x):
                return torch.dot(self.W, x)

        # 对模型 M 进行符号化追踪，得到 traced
        traced = torch.fx.symbolic_trace(M())

        # 获取 traced 图中最后一个 op 为 "output" 的节点
        out = [n for n in traced.graph.nodes if n.op == "output"][-1]
        # 在 out 节点之前插入一个 relu 方法调用的节点
        with traced.graph.inserting_before(out):
            relu_out = traced.graph.call_method(method_name='relu',
                                                args=(out.args[0],))
        # 将 out 的参数设为 relu_out
        out.args = (relu_out,)

        # 重新编译 traced 模型
        traced.recompile()

        # 捕获标准错误流，并期望捕获 TypeError 异常
        with self.capture_stderr() as captured:
            with self.assertRaises(TypeError):
                traced(5)

        # 断言捕获到的标准错误流中包含特定文本
        self.assertRegex(captured[0],
                         r"Call using an FX-traced Module, line .* of the "
                         r"traced Module's generated forward function:")

    def test_custom_traceback_not_raised_when_exception_source_is_submodule(self):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 定义模型 M 的初始化方法
            def __init__(self):
                super().__init__()
                # 添加一个线性层，输入维度为 3，输出维度为 4
                self.linear = torch.nn.Linear(3, 4)

            # 定义模型 M 的前向传播方法，返回线性层对输入 x 的输出
            def forward(self, x):
                return self.linear(x)

        # 对模型 M 进行符号化追踪，得到 traced
        traced = torch.fx.symbolic_trace(M())

        # 尝试运行 traced 模型，期望捕获 RuntimeError 异常
        try:
            traced(torch.rand(5, 5))
        except RuntimeError:
            captured = traceback.format_exc()

        # 断言捕获到的异常信息不包含特定文本
        self.assertNotRegex(captured,
                            r"Call using an FX-traced Module, line .* of the "
                            r"traced Module's generated forward function:")
    # 定义一个测试方法，用于测试图模块在数据并行情况下的复制行为
    def test_graph_module_replicate_for_dp(self):
        # 定义一个简单的神经网络模块类
        class Foo(torch.nn.Module):
            # 前向传播方法，应用 ReLU 激活函数
            def forward(self, x):
                return torch.relu(x)

        # 对 Foo 类进行符号化追踪，生成图模块
        gm = torch.fx.symbolic_trace(Foo())

        # 创建一个随机张量作为输入
        x = torch.randn(5, 3)
        # 在图模块上执行前向传播
        out = gm(x)

        # 复制图模块以支持数据并行
        replica = gm._replicate_for_data_parallel()
        # 在复制的图模块上执行前向传播
        out_replica = replica(x)

        # 断言复制前后输出的张量应该相同
        torch.testing.assert_close(out_replica, out)

    # 定义一个测试方法，用于测试 AST 重写器在修改 assert 语句时的行为
    def test_ast_rewriter_rewrites_assert(self):
        # 定义一个简单的神经网络模块类，包含一个 assert 语句
        class M(torch.nn.Module):
            # 前向传播方法，接受一个张量 x 和两个整数 y、z 作为输入
            def forward(self, x: torch.Tensor, y: int, z: int):
                # 断言 y 应该等于 z
                assert y == z
                # 返回 x 加上 x 的结果
                return torch.add(x, x)

        # 创建一个 AST 重写器实例
        ast_rewriter = RewritingTracer()
        # 使用重写器追踪模块 M 的图结构
        graph = ast_rewriter.trace(M())
        # 创建一个图模块对象，使用重写器的根节点和追踪得到的图结构
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对生成的图进行静态分析
        traced.graph.lint()

    # 定义一个测试方法，用于测试 AST 重写器在修改带消息的 assert 语句时的行为
    def test_ast_rewriter_rewrites_assert_with_message(self):
        # 定义一个简单的神经网络模块类，包含一个带消息的 assert 语句
        class M(torch.nn.Module):
            # 前向传播方法，接受一个张量 x 和两个整数 y、z 作为输入
            def forward(self, x: torch.Tensor, y: int, z: int):
                # 断言 y 应该等于 z，并提供错误消息 "msg"
                assert y == z, "msg"
                # 返回 x 加上 x 的结果
                return torch.add(x, x)

        # 创建一个 AST 重写器实例
        ast_rewriter = RewritingTracer()
        # 使用重写器追踪模块 M 的图结构
        graph = ast_rewriter.trace(M())
        # 创建一个图模块对象，使用重写器的根节点和追踪得到的图结构
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对生成的图进行静态分析
        traced.graph.lint()

    # 定义一个测试方法，用于测试异常情况下的函数追踪行为
    def test_throw_out_variant(self):
        # 定义一个简单的函数，对输入张量进行操作并抛出异常
        def foo(x):
            # 创建一个与 x 形状相同的随机张量 y
            y = torch.rand_like(x)
            # 对 x 执行 sigmoid 操作，将结果存入 y 中
            torch.sigmoid(x, out=y)
            return y

        # 定义一个自定义的追踪器类，启用可变操作检查
        class MyTracer(torch.fx.Tracer):
            check_mutable_operations = True

        # 创建 MyTracer 的实例
        tracer = MyTracer()
        # 断言在执行追踪器追踪 foo 函数时会抛出特定异常
        with self.assertRaisesRegex(RuntimeError, 'mutable operation aten::sigmoid.out'):
            traced_graph = tracer.trace(foo)

    # 定义一个测试方法，用于测试 AST 重写器在重新分配子模块时的行为
    def test_ast_rewriter_reassigns_submodules(self):
        # 定义一个包含 BatchNorm2d 子模块的神经网络模块类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个 BatchNorm2d 模块
                self.bn = torch.nn.BatchNorm2d(100)

            # 前向传播方法，接受一个张量 x 作为输入
            def forward(self, x: torch.Tensor):
                # 返回 x 加上 x 的结果
                return torch.add(x, x)

        # 创建一个 AST 重写器实例
        ast_rewriter = RewritingTracer()
        # 使用重写器追踪模块 M 的图结构
        graph = ast_rewriter.trace(M())
        # 创建一个图模块对象，使用重写器的根节点和追踪得到的图结构
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对生成的图进行静态分析
        traced.graph.lint()

    # 定义一个测试方法，用于测试 AST 重写器在包装函数时的行为
    def test_ast_rewriter_wrap(self):
        # 断言对 a_lifted_leaf 函数的调用结果为特定值
        self.assertEqual(3 + 4 + 5, a_lifted_leaf((3, 4), 5))

        # 定义一个将包含 y 的输入传递给 a_lifted_leaf 函数的函数
        def to_trace(y):
            return (
                a_lifted_leaf((4, y), 3)
                + a_lifted_leaf((3, 4), 5)
                + a_lifted_leaf((y, y), y)
            )

        # 创建一个 AST 重写器实例
        ast_rewriter = RewritingTracer()
        # 使用重写器追踪 to_trace 函数的图结构
        graph = ast_rewriter.trace(to_trace)
        # 创建一个图模块对象，使用重写器的根节点和追踪得到的图结构
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 断言生成的图模块代码中包含字符串 "a_lifted_leaf"
        self.assertIn("a_lifted_leaf", traced.code)
        # 断言对 traced(2) 的调用结果为特定值
        self.assertEqual(27, traced(2))
        # 断言 real_a_lifed_leaf 函数与 a_lifted_leaf 函数是同一对象
        self.assertIs(a_lifted_leaf, real_a_lifed_leaf)
    # 测试函数：直接测试AST重写器包装函数
    def test_ast_rewriter_wrap_fn_directly(self):
        # 断言：测试表达式求值结果
        self.assertEqual(3 + 4 + 5, a_lifted_leaf2((3, 4), 5))

        # 定义内部函数：用于跟踪
        def to_trace(y):
            # 返回表达式的求值结果
            return (
                a_lifted_leaf2((4, y), 3)
                + a_lifted_leaf2((3, 4), 5)
                + a_lifted_leaf2((y, y), y)
            )

        # 创建AST重写器对象
        ast_rewriter = RewritingTracer()
        # 对内部函数进行跟踪
        graph = ast_rewriter.trace(to_trace)
        # 创建图模块
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 断言：检查是否包含特定函数名
        self.assertIn("a_lifted_leaf2", traced.code)
        # 断言：检查调用结果
        self.assertEqual(27, traced(2))
        # 断言：检查对象是否相同
        self.assertIs(a_lifted_leaf2, real_a_lifed_leaf2)

    # 测试函数：分析器范围副作用
    def test_profiler_ranges_side_effect(self):
        # 创建空白图对象
        g = torch.fx.Graph()
        # 调用函数：记录进入新函数
        handle = g.call_function(torch.ops.profiler._record_function_enter_new, ('test_range',))
        # 调用函数：记录退出函数
        g.call_function(torch.ops.profiler._record_function_exit, (handle,))
        # 输出空
        g.output(None)

        # 初始化找到的目标字典
        found_targets = {}
        # 遍历图节点
        for node in g.nodes:
            # 如果节点操作为函数调用
            if node.op == 'call_function':
                # 将目标函数添加到字典中
                found_targets.setdefault(node.target)

        # 断言：检查找到的目标函数列表
        self.assertEqual(
            list(found_targets.keys()),
            [torch.ops.profiler._record_function_enter_new, torch.ops.profiler._record_function_exit]
        )

        # 消除死代码
        g.eliminate_dead_code()

        # 重置目标函数字典
        found_targets = {}
        # 再次遍历图节点
        for node in g.nodes:
            # 如果节点操作为函数调用
            if node.op == 'call_function':
                # 将目标函数添加到字典中
                found_targets.setdefault(node.target)

        # 断言：再次检查找到的目标函数列表
        self.assertEqual(
            list(found_targets.keys()),
            [torch.ops.profiler._record_function_enter_new, torch.ops.profiler._record_function_exit]
        )

    # 测试函数：通过装饰器包装AST重写器
    def test_ast_rewriter_wrapped_via_decorator(self):
        # 定义类：继承自PyTorch的Module类
        class F(torch.nn.Module):
            # 前向传播函数定义
            def forward(self, x):
                # 调用通过装饰器包装的函数
                return wrapped_via_decorator(x)

        # 创建AST重写器对象
        ast_rewriter = RewritingTracer()
        # 对类F的实例进行跟踪
        graph = ast_rewriter.trace(F())
        # 创建图模块
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 断言：检查是否包含特定函数名
        self.assertIn("wrapped_via_decorator", traced.code)
        # 断言：检查调用结果
        self.assertEqual(traced(0), 1)
        # 断言：检查对象是否相同
        self.assertIs(wrapped_via_decorator, real_wrapped_via_decorator)
        # 断言：检查函数是否未被fx模块修补过
        self.assertFalse(hasattr(wrapped_via_decorator, "__fx_already_patched"))
    def test_ast_rewriter_wrapped_via_decorator_and_transformed(self):
        # 调用 wrapped_via_decorator 函数，预期返回结果为 1
        self.assertEqual(wrapped_via_decorator(0), 1)

        def to_trace(y):
            # 调用 wrapped_via_decorator 函数，并返回其结果
            return wrapped_via_decorator(y)

        # 创建 RewritingTracer 对象
        ast_rewriter = RewritingTracer()
        # 对 to_trace 函数进行跟踪分析，生成函数调用图
        graph = ast_rewriter.trace(to_trace)
        # 创建 GraphModule 对象，包含跟踪结果的图表示
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 检查 traced 中是否包含 wrapped_via_decorator 函数的代码
        self.assertIn("wrapped_via_decorator", traced.code)
        # 调用 traced 对象，预期返回结果为 1
        self.assertEqual(traced(0), 1)
        # 断言 wrapped_via_decorator 函数与 real_wrapped_via_decorator 函数引用相同
        self.assertIs(wrapped_via_decorator, real_wrapped_via_decorator)
        # 检查 wrapped_via_decorator 函数是否未被修改过
        self.assertFalse(hasattr(wrapped_via_decorator, "__fx_already_patched"))

        # 对 traced 对象进行转换处理，生成 transformed 对象
        transformed = torch.fx.Transformer(traced).transform()
        # 检查 transformed 中是否包含 wrapped_via_decorator 函数的代码
        self.assertIn("wrapped_via_decorator", transformed.code)
        # 调用 transformed 对象，预期返回结果为 1
        self.assertEqual(transformed(0), 1)
        # 再次断言 wrapped_via_decorator 函数与 real_wrapped_via_decorator 函数引用相同
        self.assertIs(wrapped_via_decorator, real_wrapped_via_decorator)
        # 检查 wrapped_via_decorator 函数是否未被修改过
        self.assertFalse(hasattr(wrapped_via_decorator, "__fx_already_patched"))

    def test_ast_rewriter_wrap_with_submodule(self):
        # 定义包含 BatchNorm1d 的自定义模块 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 实例化 BatchNorm1d，设置 affine=False
                self.batchnorm1d = torch.nn.BatchNorm1d(2, affine=False)

            def forward(self, x: torch.Tensor):
                # 调用 wrapped_with_submodule 函数，传入 x 和 self.batchnorm1d 参数
                return wrapped_with_submodule(x, self.batchnorm1d)

        # 创建 RewritingTracer 对象
        ast_rewriter = RewritingTracer()
        # 对 M 类进行跟踪分析，生成函数调用图
        graph = ast_rewriter.trace(M())
        # 创建 GraphModule 对象，包含跟踪结果的图表示
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 检查 traced 中是否包含 wrapped_with_submodule 函数的代码
        self.assertIn("wrapped_with_submodule", traced.code)

        # 准备输入数据
        input = torch.rand(3, 2)
        # 创建标准的 BatchNorm1d 对象，用于验证输出结果
        ref_batchnorm1d = torch.nn.BatchNorm1d(2, affine=False)
        # 断言标准 BatchNorm1d 对象与 traced 对象在输入 input 下的输出结果相等
        self.assertEqual(ref_batchnorm1d(input), traced(input))

    def test_delete_unused_submodules_leaf(self):
        # 定义包含 Linear 和 ReLU 的子模块 SubModule
        class SubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.linear(x)
                x = self.relu(x)
                return x

        # 定义包含 SubModule 的模型 Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submod = SubModule()

            def forward(self, x):
                # 调用 SubModule 的 forward 方法处理输入 x
                x = self.submod(x)
                return x

        # 创建 Model 实例
        model = Model()

        # 自定义的 Tracer 类，用于跟踪模块的使用情况
        class MyCustomTracer(torch.fx.Tracer):
            def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
                # 如果模块的完全限定名为 "submod"，则认为它是叶子模块
                return module_qualified_name == "submod"

        # 准备输入数据
        inputs = torch.randn(1, 10)
        # 使用自定义 Tracer 对模型进行跟踪分析，生成跟踪图
        traced_graph = MyCustomTracer().trace(model)
        # 创建 GraphModule 对象，包含跟踪结果的图表示
        gm2 = torch.fx.GraphModule(model, traced_graph)
        # 删除所有未使用的子模块
        gm2.delete_all_unused_submodules()
        # 断言经过跟踪图的模型 gm2 在输入 inputs 下的输出结果与原始模型 model 在输入 inputs 下的输出结果相等
        torch.testing.assert_close(gm2(inputs), model(inputs))
    def test_fx_stateless(self):
        # 定义一个 MockModule 类，用于模拟一个简单的神经网络模块
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个线性层，输入维度为1，输出维度为1
                self.l1 = torch.nn.Linear(1, 1)
                # 注册一个缓冲区，包含一个值为1的张量
                self.register_buffer('buffer', torch.ones(1))

            # 前向传播函数，对输入 x 执行线性变换并加上缓冲区的值
            def forward(self, x):
                return self.l1(x) + self.buffer

        # 创建一个 MockModule 实例
        module = MockModule()
        # 生成一个形状为 (1, 1) 的随机张量 x
        x = torch.rand((1, 1))
        # 创建一个权重张量，值为 [[1.0]]，并设置 requires_grad 为 True
        weight = torch.tensor([[1.0]], requires_grad=True)
        # 创建一个偏置张量，值为 [0.0]，并设置 requires_grad 为 True
        bias = torch.tensor([0.0], requires_grad=True)
        # 创建一个缓冲区张量，值为 [0.0]
        buffer = torch.tensor([0.0])
        # 将权重、偏置和缓冲区组成一个字典
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer}
        # 对 MockModule 进行符号化追踪，生成 fx_module
        fx_module = torch.fx.symbolic_trace(module)
        # 使用 torch.func.functional_call 调用 fx_module 进行前向传播计算
        res = torch.func.functional_call(fx_module, parameters, x)
        # 对计算结果 res 执行反向传播
        res.backward()
        # 断言权重和偏置的梯度不为 None
        self.assertIsNotNone(weight.grad)
        self.assertIsNotNone(bias.grad)
        # 断言缓冲区的梯度为 None
        self.assertIsNone(buffer.grad)
        # 断言 MockModule 内部的权重和偏置的梯度为 None
        self.assertIsNone(module.l1.weight.grad)
        self.assertIsNone(module.l1.bias.grad)
        self.assertIsNone(module.buffer.grad)

    def _test_graph_module_init_buffer_param_copied(self, use_dict_init: bool):
        # 定义一个 MyModule 类，用于测试模块初始化和参数复制
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个缓冲区 "my_buff"，包含一个形状为 (3, 4) 的随机张量
                self.register_buffer("my_buff", torch.rand(3, 4))
                # 注册一个参数 "my_param"，包含一个形状为 (3, 4) 的随机张量
                self.register_parameter(
                    "my_param", torch.nn.Parameter(torch.rand(3, 4))
                )

            # 前向传播函数，对输入 x 加上缓冲区 "my_buff" 和参数 "my_param"
            def forward(self, x):
                return x + self.my_buff + self.my_param

        # 创建 MyModule 实例 mod
        mod = MyModule()
        # 对 mod 进行符号化追踪，生成 mod_traced
        mod_traced = symbolic_trace(mod)

        # 根据原始模块创建一个新的 GraphModule，可以选择使用字典初始化或根模块初始化
        orig_buff = mod_traced.get_buffer("my_buff")
        orig_param = mod_traced.get_parameter("my_param")
        mod_traced_new = GraphModule(
            {"my_buff": orig_buff, "my_param": orig_param} if use_dict_init else mod,
            mod_traced.graph,
        )

        # 检查新的 GraphModule 中是否找到了缓冲区 "my_buff" 和参数 "my_param"，并且与原始的相同
        try:
            new_buff = mod_traced_new.get_buffer("my_buff")
        except Exception:
            self.fail("Did not find my_buff")
        self.assertEqual(orig_buff, new_buff)

        try:
            new_param = mod_traced_new.get_parameter("my_param")
        except Exception:
            self.fail("Did not find my_param")
        self.assertEqual(orig_param, new_param)

        # 生成一个形状为 (3, 4) 的随机张量 x
        x = torch.rand(3, 4)
        # 使用原始模块 mod_traced 计算输出 orig_out
        orig_out = mod_traced(x)
        # 使用新的 GraphModule mod_traced_new 计算输出 submodules_out
        submodules_out = mod_traced_new(x)

        # 断言 orig_out 和 submodules_out 相等
        self.assertEqual(orig_out, submodules_out)

    def test_graph_module_init_buffer_param_copied_dict_init(self):
        # 测试 GraphModule 的初始化和参数复制，使用字典初始化
        self._test_graph_module_init_buffer_param_copied(use_dict_init=True)

    def test_graph_module_init_buffer_param_copied_mod_init(self):
        # 测试 GraphModule 的初始化和参数复制，使用根模块初始化
        self._test_graph_module_init_buffer_param_copied(use_dict_init=False)
    # 定义一个测试方法，测试没有前向引用的类型注解
    def test_annotations_with_no_forward_references(self):
        # 定义类 A，实现一个接受 torch.Tensor 参数的 __call__ 方法，返回输入的张量加上自身的结果
        class A:
            def __call__(self, x: torch.Tensor):
                return torch.add(x, x)

        # 定义类 M，继承自 torch.nn.Module，实现 forward 方法，接受一个 torch.Tensor 参数和一个 A 类型的参数 a，
        # 返回调用 a 对象处理输入张量 x 后的结果
        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor, a: A) -> torch.Tensor:
                return a(x)

        # 调用 checkGraphModule 方法，传入 M 类的实例，以及一个包含随机生成张量和 A 类实例的元组作为参数
        self.checkGraphModule(M(), (torch.rand(2, 3), A()), kwargs=None)

    # 定义一个测试方法，测试带有前向引用的类型注解
    def test_annotations_with_forward_references(self):
        # 定义类 A，实现一个接受 torch.Tensor 参数的 __call__ 方法，返回输入的张量加上自身的结果
        class A:
            def __call__(self, x: torch.Tensor):
                return torch.add(x, x)

        # 定义类 M，继承自 torch.nn.Module，实现 forward 方法，接受一个 torch.Tensor 参数和一个 A 类型的参数 a，
        # 返回调用 a 对象处理输入张量 x 后的结果
        class M(torch.nn.Module):
            def forward(self, x: 'torch.Tensor', a: 'A') -> 'torch.Tensor':
                return a(x)

        # 调用 checkGraphModule 方法，传入 M 类的实例，以及一个包含随机生成张量和 A 类实例的元组作为参数
        self.checkGraphModule(M(), (torch.rand(2, 3), A()), kwargs=None)

    # 定义一个测试方法，测试带有非 torch 引用和没有内部前向引用的类型注解
    def test_annotations_with_non_torch_reference_and_no_internal_forward_references(self):
        # 定义类 A，实现一个接受 torch.Tensor 参数的 __call__ 方法，返回输入的张量加上自身的结果
        class A:
            def __call__(self, x: torch.Tensor):
                return torch.add(x, x)

        # 定义类 M，继承自 torch.nn.Module，实现 forward 方法，接受一个 torch.Tensor 的列表参数和一个 A 类型的参数 a，
        # 返回调用 a 对象处理列表中第一个张量 x[0] 后的结果
        class M(torch.nn.Module):
            def forward(self, x: List[torch.Tensor], a: A) -> torch.Tensor:
                return a(x[0])

        # 调用 checkGraphModule 方法，传入 M 类的实例，以及一个包含随机生成张量和 A 类实例的元组作为参数
        self.checkGraphModule(M(), (torch.rand(2, 3), A()), kwargs=None)

    # 定义一个测试方法，测试带有非 torch 引用和内部前向引用的类型注解
    def test_annotations_with_non_torch_reference_and_internal_forward_references(self):
        # 定义类 A，实现一个接受 torch.Tensor 参数的 __call__ 方法，返回输入的张量加上自身的结果
        class A:
            def __call__(self, x: torch.Tensor):
                return torch.add(x, x)

        # 定义类 M，继承自 torch.nn.Module，实现 forward 方法，接受一个 torch.Tensor 的列表参数和一个 A 类型的参数 a，
        # 返回调用 a 对象处理列表中第一个张量 x[0] 后的结果的第一个元素
        class M(torch.nn.Module):
            def forward(self, x: List['torch.Tensor'], a: A) -> 'torch.Tensor':
                return a(x)[0]

        # 调用 checkGraphModule 方法，传入 M 类的实例，以及一个包含随机生成张量和 A 类实例的元组作为参数
        self.checkGraphModule(M(), (torch.rand(2, 3), A()), kwargs=None)

    # 如果 Python 版本小于 3.7，则跳过此测试
    @unittest.skipIf(sys.version_info < (3, 7), "`__future__` feature "
                     "`annotations` is not defined in Python <3.7")
    # 定义一个测试方法，测试带有 future 特性的类型注解
    def test_annotation_with_future(self):
        try:
            # 尝试导入 fx.test_future 模块（假设此模块定义了 future 特性的注解）
            import fx.test_future    # noqa: F401
        finally:
            # 删除导入的 __future__ 模块，清除影响
            del sys.modules["__future__"]

    # 如果 Python 版本大于 3.11，则跳过此测试
    @unittest.skipIf(sys.version_info > (3, 11), "Does not work in 3.11")
    # 定义一个测试方法，测试空元组的类型注解
    def test_annotations_empty_tuple(self):
        # 定义类 Foo，继承自 torch.nn.Module，实现 forward 方法，接受一个空元组 x 和一个复杂元组 y，
        # 返回字符串 "foo"
        class Foo(torch.nn.Module):
            def forward(self, x: Tuple[()], y: Tuple[str, Tuple[()]]):
                return "foo"

        # 对类 Foo 进行符号化追踪
        traced = torch.fx.symbolic_trace(Foo())

        # 定义空元组 x 和复杂元组 y 的值
        x = ()
        y = ("bar", ())

        # 运行 FileCheck 对象，检查 traced 的代码
        FileCheck().check("_Tuple[()]")   \
                   .check("typing_Tuple[str,typing_Tuple[()]]") \
                   .run(traced.code)

        # 对 traced 进行脚本化
        scripted = torch.jit.script(traced)

        # 运行 FileCheck 对象，检查 scripted 的代码
        FileCheck().check("Tuple[()]")   \
            .check("Tuple[str, Tuple[()]]")    \
            .run(scripted.code)

    # 如果运行在 Windows 或 Python 版本大于等于 3.10，则跳过此测试
    @unittest.skipIf(IS_WINDOWS, "Python Windows bug? https://bugs.python.org/issue45108")
    @unittest.skipIf(sys.version_info >= (3, 10), "Does not work on Python-3.10")
    # 定义一个测试方法，用于测试断言的行为
    def test_assert(self):
        # 定义一个内部函数 f，接受参数 x
        def f(x):
            # 断言 x 大于 1，如果不成立会引发 AssertionError
            assert x > 1
            # 返回 x + 1
            return x + 1
        
        try:
            # 开启 Torch 的 FX 代理追踪断言
            torch.fx.proxy.TracerBase.trace_asserts = True
            # 对函数 f 进行符号化追踪
            traced = symbolic_trace(f)
        finally:
            # 最终关闭 Torch 的 FX 代理追踪断言
            torch.fx.proxy.TracerBase.trace_asserts = False

        # 断言调用 f(2) 和 traced(2) 的结果相等
        self.assertEqual(f(2), traced(2))
        # 使用 with 语句断言调用 traced(0) 会引发 AssertionError
        with self.assertRaises(AssertionError):
            traced(0)

    # 定义一个测试方法，测试 pytree 的具体功能
    def test_pytree_concrete(self):
        # 定义一个函数 f，接受参数 b 和 a
        def f(b, a):
            # 如果 b 为真，返回 a 中键为 'a' 的值
            if b:
                return a['a']
            else:
                # 否则返回 a 中键为 'z' 的值
                return a['z']

        # 输入参数 inp，包含具体的 pytree 结构
        inp = {'a': {'a': PH, 'z': PH}, 'b': True}
        # 对函数 f 进行符号化追踪，并指定具体参数 inp
        nf = symbolic_trace(f, concrete_args=inp)
        # 使用 pytree.tree_map 将 inp 中的 PH 替换为随机生成的 torch 张量
        val = pytree.tree_map(lambda x: torch.randn(3) if x == PH else x, inp)
        # 断言 nf(**val) 和 f(**val) 的结果相等
        self.assertEqual(nf(**val), f(**val))

        # 对 nf 再次进行符号化追踪
        nf = symbolic_trace(nf)
        # 断言 nf(**val) 和 f(**val) 的结果依然相等
        self.assertEqual(nf(**val), f(**val))
    def test_metadata_on_ph(self):
        def f_sum(a: int, b: int) -> int:
            return a + b

        # 定义一个函数，检查字典中的两个键是否相等
        def f_dict(a: Dict[str, str]) -> bool:
            return a["f1"] == a["f2"]

        # 验证图模块中的节点是否具有预期的元数据
        def verify_metadata(gm: GraphModule, arg_names: List[str], metadata: List[str]):
            for node in gm.graph.nodes:
                if node.op == "placeholder":
                    self.assertTrue(node.name in arg_names)
                    self.assertTrue(node.ph_key in metadata)

        # 对 f_sum 函数进行符号化跟踪，检查生成的图模块的元数据
        verify_metadata(
            gm=symbolic_trace(
                f_sum,
                concrete_args={"a": PHWithMeta(ph_key="a"), "b": PHWithMeta(ph_key="b")}
            ),
            arg_names=["a_1", "b_1"],
            metadata=["a", "b"]
        )
        # 对 f_dict 函数进行符号化跟踪，检查生成的图模块的元数据
        verify_metadata(
            gm=symbolic_trace(
                f_dict,
                concrete_args={"a": {"f1": PHWithMeta(ph_key="f1"), "f2": PHWithMeta(ph_key="f2")}}
            ),
            arg_names=["a_1", "a_2"],
            metadata=["f1", "f2"]
        )

        # 确保节点上的标记不会被具有相同属性名（标记）的 PH 属性覆盖
        class TaggingTracer(Tracer):
            def create_node(self, kind : str, target : Union[str, Callable],
                            args : Tuple[Argument, ...], kwargs : Dict[str, Any], name : Optional[str] = None,
                            type_expr : Optional[Any] = None) -> Node:
                n = super().create_node(kind, target, args, kwargs, name)
                n.tag = "foo"
                return n

        # 定义一个带有标记的占位符类
        class PHWithTag(PHBase):
            def __init__(self, tag: str):
                super().__init__()
                self.tag = tag

        # 使用 TaggingTracer 跟踪 f_sum 函数，并传递带有不同标记的 PHWithTag 作为参数
        g = TaggingTracer().trace(f_sum, concrete_args={"a": PHWithTag(tag="bar"), "b": PHWithTag(tag="bar")})
        for n in g.nodes:
            self.assertTrue(hasattr(n, "tag"))
            # 确保标记仍然是 "foo" 而不是 "bar"（来自 PHWithTag）
            self.assertEqual(n.tag, "foo")

    def test_custom_codegen(self):
        class ListCodeGen(CodeGen):
            def gen_fn_def(self, free_vars, maybe_return_annotation):
                lst_unpack = f"""
def test_interpreter_with_codegen():
    class ListCodeGen(CodeGen):
        # 生成函数定义的代码，接收自由变量和可能的返回注释
        def gen_fn_def(self, free_vars, maybe_return_annotation):
            lst_unpack = f"""
def forward(self, args_list: List[torch.Tensor]){maybe_return_annotation}:
    {', '.join(free_vars)} = args_list"""
            return lst_unpack

        # 返回额外的全局变量声明，这里返回了一个包含 ('List', typing.List) 的列表
        def additional_globals(self):
            return [('List', typing.List)]

        # 处理输入的方法，确保只有一个输入，并返回这个输入
        def process_inputs(self, *inputs):
            assert len(inputs) == 1
            return inputs[0]

        # 生成输出的方法，返回一个以输出参数为元素的列表的字符串表示
        def generate_output(self, output_args):
            return f'return list({repr(output_args)})'

        # 处理输出的方法，将输出转换为列表形式返回
        def process_outputs(self, outputs):
            return list(outputs)

    # 定义一个简单的函数 f，接受两个参数并返回它们的和
    def f(a, b):
        a = a + b
        b = a + b
        return a, b

    # 使用 symbolic_trace 对函数 f 进行符号跟踪，生成 nf 对象
    nf = symbolic_trace(f)
    
    # 生成随机张量列表作为测试输入
    vals = [torch.randn(3), torch.randn(3)]
    
    # 设置 nf 的代码生成器为 ListCodeGen 类的实例
    nf.graph.set_codegen(ListCodeGen())
    
    # 重新编译 nf 对象
    nf.recompile()
    
    # 断言 Interpreter(nf).run(vals) 的运行结果等于 nf(vals)
    self.assertEqual(Interpreter(nf).run(vals), nf(vals))
    # 定义一个测试方法，用于验证使用 torch.fx 创建图形，并进行操作
    def test_imul_code_print(self):
        # 创建一个空的 torch.fx 图形
        graph = torch.fx.Graph()
        # 在图中创建占位符 'a' 和 'b'
        a = graph.placeholder("a")
        b = graph.placeholder("b")
        # 向图中添加 operator.imul 的函数调用，将 a 和 b 作为参数传递
        graph.call_function(operator.imul, (a, b), {})
        # 将 a 设置为输出节点
        graph.output(a)
        # 使用 torch.fx 创建 GraphModule 对象 gm，并重新编译
        gm = torch.fx.GraphModule({}, graph)
        gm.recompile()
        # 断言 gm(2, 3) 的返回值为 6
        self.assertEqual(gm(2, 3), 6)
        # 断言 gm.code 中包含字符串 "a *= b"
        self.assertIn("a *= b", gm.code)

    # 定义一个测试方法，用于验证深拷贝 Tracer 对象的行为
    def test_deepcopy_tracer(self):
        # 定义一个简单的函数 fn，对输入 x 和 y 进行数学运算
        def fn(x, y):
            return (x + y).relu().sin()

        # 创建一个 Tracer 对象
        tracer = Tracer()
        # 深拷贝 tracer 对象，得到 tracer_before
        tracer_before = copy.deepcopy(tracer)
        # 对函数 fn 进行跟踪
        tracer.trace(fn)
        # 再次深拷贝 tracer 对象，得到 tracer_after
        tracer_after = copy.deepcopy(tracer)

        # 断言跟踪后的 tracer.graph 与 tracer_after.graph 相等
        self.assertEqual(str(tracer.graph), str(tracer_after.graph))
        # 断言 tracer_before 中不存在 'graph' 属性或者两个 graph 不相等
        self.assertTrue(not hasattr(tracer_before, 'graph') or str(tracer.graph) != str(tracer_before.graph))

    # 定义一个测试方法，用于验证深拷贝 GraphModule 对象的行为
    def test_deepcopy_graphmodule(self):
        # 使用 symbolic_trace 对 SimpleTest() 进行符号化跟踪，得到模块 m
        m = symbolic_trace(SimpleTest())
        # 向 m 的 meta 属性中添加键值对 'hello': 'world'
        m.meta['hello'] = 'world'
        # 深拷贝模块 m，得到 copy_m
        copy_m = copy.deepcopy(m)
        # 断言 copy_m 的 meta 属性中 'hello' 的值为 'world'
        self.assertEqual(copy_m.meta['hello'], 'world')

    # 定义一个测试方法，用于验证处理循环引用时深拷贝的行为
    def test_deepcopy_no_recursion(self):
        # 使用 symbolic_trace 对 SimpleTest() 进行符号化跟踪，得到模块 m
        m = symbolic_trace(SimpleTest())
        # 将 m 自身赋值给其 meta 属性中的 'hello' 键，形成循环引用
        m.meta['hello'] = m  # circular reference
        # 深拷贝模块 m，得到 copy_m，此时拷贝完成
        copy_m = copy.deepcopy(m)
        # 断言 copy_m 和 copy_m.meta['hello'] 的内存地址相同，即循环引用成功处理
        self.assertEqual(id(copy_m), id(copy_m.meta['hello']))

    # 定义一个测试方法，用于验证枚举类型的使用
    def test_enum(self):
        # 导入枚举类 Enum
        from enum import Enum

        # 定义一个简单的枚举类 Foo
        class Foo(Enum):
            A = 1
            B = 2

        # 定义 leaf_fn 函数，接受一个数组 arr 和一个枚举值 enum_val 作为参数
        def leaf_fn(arr, enum_val):
            # 将 enum_val 添加到 arr 中
            arr.append(enum_val)
            # 返回枚举值的整数值
            return arr[-1].value

        # 定义 foo 函数，接受一个参数 x
        def foo(x):
            # 调用 leaf_fn 函数，将枚举值 Foo.A 作为参数传递
            return leaf_fn(x, Foo.A)

        # 对 foo 函数进行符号化跟踪，得到 traced
        traced = torch.fx.symbolic_trace(foo)
        # 断言 foo([]) 和 traced([]) 的返回值相等
        self.assertEqual(foo([]), traced([]))

    # 定义一个测试方法，用于验证向 Graph 中插入参数的行为
    def test_insert_arg(self):
        # 使用 symbolic_trace 对 SimpleTest() 进行符号化跟踪，得到模块 m
        m = symbolic_trace(SimpleTest())
        # 向模块 m 注册一个缓冲区 "buf"，其值为 torch.tensor(0)
        m.register_buffer("buf", torch.tensor(0))
        # 获取图中最后一个节点作为输出节点
        output_node = next(iter(reversed(m.graph.nodes)))
        # 在 output_node 之前插入代码块
        with m.graph.inserting_before(output_node):
            # 获取名为 "buf" 的属性
            a = m.graph.get_attr("buf")
        # 记录 output_node.args 的长度为 r
        r = len(output_node.args)
        # 在 output_node 的参数列表开头插入参数 a
        output_node.insert_arg(0, a)
        # 断言 output_node.args 的长度比 r 增加了 1
        self.assertEqual(len(output_node.args), r + 1)
        # 断言 a 的使用者数量为 1
        self.assertEqual(len(a.users), 1)
        # 断言 output_node.args[0] 指向 a
        self.assertIs(output_node.args[0], a)
        # 断言 a 的用户中的第一个元素指向 output_node
        self.assertIs(next(iter(a.users.keys())), output_node)
        # 在 output_node 的参数列表的第二个位置插入参数 a
        output_node.insert_arg(2, a)
        # 断言 output_node.args 的长度比 r 增加了 2
        self.assertEqual(len(output_node.args), r + 2)
        # 断言 a 的使用者数量为 1
        self.assertEqual(len(a.users), 1)
        # 断言 output_node.args[2] 指向 a
        self.assertIs(output_node.args[2], a)
        # 断言 a 的用户中的第一个元素指向 output_node
        self.assertIs(next(iter(a.users.keys())), output_node)
        # 对图进行 lint 操作
        m.graph.lint()
# 定义一个函数，用于设置 torch.fx 的目标项处理过程
def run_getitem_target():
    # 导入 _wrapped_methods_to_patch 函数并添加 "__getitem__" 方法的包装
    from torch.fx._symbolic_trace import _wrapped_methods_to_patch
    _wrapped_methods_to_patch.append((torch.Tensor, "__getitem__"))
    
    try:
        # 创建 TestFX 类的实例并调用其 getitem_inner 方法
        TestFX().getitem_inner()
    finally:
        # 弹出最后添加的包装方法，确保环境不受影响
        _wrapped_methods_to_patch.pop()


class TestOperatorSignatures(JitTestCase):
    def setUp(self):
        # 设置检查可变操作在跟踪过程中的状态标志为 True
        # 在测试中启用该功能，但默认情况下禁用
        self.orig_tracer_mutable_flag = torch.fx.proxy.TracerBase.check_mutable_operations
        torch.fx.proxy.TracerBase.check_mutable_operations = True

    def tearDown(self):
        # 恢复原始的检查可变操作状态标志
        torch.fx.proxy.TracerBase.check_mutable_operations = self.orig_tracer_mutable_flag

    @onlyCPU
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_get_torch_func_signature_exhaustive(self, device, dtype, op):
        # 如果操作不是内置函数类型，则跳过测试
        if not isinstance(op.op, types.BuiltinFunctionType):
            raise unittest.SkipTest("This path doesn't work on Python functions")
        
        # 从操作中获取样本输入迭代器
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
        
        # 获取操作的 Torch 函数签名列表
        schemas = get_signature_for_torch_op(op.op)
        
        # 如果没有获取到函数签名，则引发运行时错误
        if not schemas:
            raise RuntimeError('No Schemas Returned')
        
        # 对于每个样本输入，尝试匹配函数签名
        for sample_input in sample_inputs_itr:
            # 遍历函数签名列表，尝试绑定参数并调用操作
            for schema in schemas:
                try:
                    bound_args = schema.bind(sample_input.input, *sample_input.args, **sample_input.kwargs)
                    bound_args.apply_defaults()
                    op(*bound_args.args, **bound_args.kwargs)
                    break
                except TypeError as e:
                    pass
            else:
                # 如果没有找到匹配的函数签名，则引发运行时错误
                raise RuntimeError(f'Did not match any schemas for op {op.name}!')


class TestFXAPIBackwardCompatibility(JitTestCase):
    def setUp(self):
        # 调用父类的 setUp 方法
        super().setUp()
        # 设置 self.maxDiff 为 None，用于测试时不限制比较的最大差异
        self.maxDiff = None
        
        # 设置检查可变操作在跟踪过程中的状态标志为 True
        # 在测试中启用该功能，但默认情况下禁用
        self.orig_tracer_mutable_flag = torch.fx.proxy.TracerBase.check_mutable_operations
        torch.fx.proxy.TracerBase.check_mutable_operations = True

    def tearDown(self):
        # 调用父类的 tearDown 方法
        super().tearDown()
        # 恢复原始的检查可变操作状态标志
        torch.fx.proxy.TracerBase.check_mutable_operations = self.orig_tracer_mutable_flag
    def _fn_to_stable_annotation_str(self, obj):
        """
        Unfortunately we have to serialize function signatures manually since
        serialization for `inspect.Signature` objects is not stable across
        python versions
        """
        # 获取函数名的类型字符串表示
        fn_name = torch.typename(obj)

        # 获取函数的签名对象
        signature = inspect.signature(obj)

        # 构建函数签名的字符串表示，包括函数名和参数列表
        sig_str = f'{fn_name}{signature}'

        # 初始化空列表，用于存储参数的字符串表示
        arg_strs = []

        # 遍历函数签名中的参数
        for k, v in signature.parameters.items():
            # 判断是否存在类型注解，若存在则将其转换为稳定的字符串表示
            maybe_type_annotation = f': {self._annotation_type_to_stable_str(v.annotation, sig_str)}'\
                if v.annotation is not inspect.Signature.empty else ''

            # 定义处理默认值的函数
            def default_val_str(val):
                # 若默认值是元组或列表，则转换为字符串表示
                if isinstance(val, (tuple, list)):
                    str_pieces = ['(' if isinstance(val, tuple) else '[']
                    str_pieces.append(', '.join(default_val_str(v) for v in val))
                    if isinstance(val, tuple) and len(str_pieces) == 2:
                        str_pieces.append(',')
                    str_pieces.append(')' if isinstance(val, tuple) else ']')
                    return ''.join(str_pieces)

                # 处理特定类型的默认值，例如模块和可调用对象
                if isinstance(val, types.ModuleType):
                    return f'<module {val.__name__}>'
                if callable(val):
                    return f'<function {val.__name__}>'
                return str(val)

            # 如果参数有默认值，则获取其默认值的字符串表示
            if v.default is not inspect.Signature.empty:
                default_val_str = default_val_str(v.default) if not isinstance(v.default, str) else f"'{v.default}'"
                maybe_default = f' = {default_val_str}'
            else:
                maybe_default = ''

            # 处理参数类型为可变位置参数或可变关键字参数的情况
            maybe_stars = ''
            if v.kind == inspect.Parameter.VAR_POSITIONAL:
                maybe_stars = '*'
            elif v.kind == inspect.Parameter.VAR_KEYWORD:
                maybe_stars = '**'

            # 构建参数的字符串表示，并添加到参数列表中
            arg_strs.append(f'{maybe_stars}{k}{maybe_type_annotation}{maybe_default}')

        # 处理返回类型的注解，若存在则转换为稳定的字符串表示
        return_annot = f' -> {self._annotation_type_to_stable_str(signature.return_annotation, sig_str)}'\
            if signature.return_annotation is not inspect.Signature.empty else ''

        # 返回最终的函数签名字符串表示，包括函数名、参数列表和返回类型注解
        return f'{fn_name}({", ".join(arg_strs)}){return_annot}'
    def test_function_back_compat(self):
        """
        Test backward compatibility for function signatures with
        @compatibility(is_backward_compatible=True). Currently this checks for
        exact signature matches, which may lead to false positives. If this
        becomes too annoying, we can refine this check to actually parse out
        the saved schema strings and check if the change is truly backward-
        incompatible.
        """
        # 存储函数签名字符串的列表
        signature_strs = []

        # 遍历_BACK_COMPAT_OBJECTS列表中的对象
        for obj in _BACK_COMPAT_OBJECTS:
            # 如果对象不是类，则将其稳定的注释字符串添加到signature_strs中
            if not isinstance(obj, type):
                signature_strs.append(self._fn_to_stable_annotation_str(obj))

        # 对签名字符串列表进行排序
        signature_strs.sort()

        try:
            # 断言函数的签名字符串与预期输出相符
            self.assertExpected('\n'.join(signature_strs) + '\n', 'fx_backcompat_function_signatures')
        except AssertionError as e:
            # 如果断言失败，生成错误消息并抛出AssertionError
            msg = f"{e}\n****** ERROR ******\nAn FX function that has been marked " \
                  f"as backwards-compatible has experienced a signature change. See the " \
                  f"above exception context for more information. If this change was " \
                  f"unintended, please revert it. If it was intended, check with the FX " \
                  f"team to ensure that the proper deprecation protocols have been followed " \
                  f"and subsequently --accept the change."
            raise AssertionError(msg)  # noqa: B904

    def test_class_member_back_compat(self):
        """
        Test backward compatibility for members of classes with
        @compatibility(is_backward_compatible=True). Currently this checks for
        exact matches on the publicly visible members of the class.
        """
        # 存储类方法字符串的列表
        class_method_strs = []

        # 遍历_BACK_COMPAT_OBJECTS列表中的对象
        for obj in _BACK_COMPAT_OBJECTS:
            # 如果对象是类，则获取其公共成员并添加到class_method_strs中
            if isinstance(obj, type):
                public_members = [name for name in obj.__dict__ if not name.startswith('_')]
                class_method_strs.append(f'{torch.typename(obj)} {sorted(public_members)}')

        # 对类方法字符串列表进行排序
        class_method_strs.sort()

        try:
            # 断言类方法字符串与预期输出相符
            self.assertExpected('\n'.join(class_method_strs), 'fx_backcompat_class_members')
        except AssertionError as e:
            # 如果断言失败，生成错误消息并抛出AssertionError
            msg = f"{e}\n****** ERROR ******\nAn FX class that has been marked " \
                  f"as backwards-compatible has experienced change in its public members. See the " \
                  f"above exception context for more information. If this change was " \
                  f"unintended, please revert it. If it was intended, check with the FX " \
                  f"team to ensure that the proper deprecation protocols have been followed " \
                  f"and subsequently --accept the change."
            raise AssertionError(msg) from e
    # 测试公共 API 表面
    def test_public_api_surface(self):
        # 用于存储非向后兼容对象的字典
        non_back_compat_objects = {}

        # 检查符号是否具有向后兼容性标记
        def check_symbols_have_bc_designation(m, seen):
            # 如果模块名不以 'torch.fx' 开头，则返回
            if not m.__name__.startswith('torch.fx'):
                return
            # 如果模块名以 'torch.fx.experimental' 开头，则返回
            if m.__name__.startswith('torch.fx.experimental'):
                return
            # 防止递归到已经检查过的模块
            seen.add(m.__name__)
            # 遍历模块 m 的所有属性
            for k, v in m.__dict__.items():
                # 如果属性 v 已经在 seen 集合中，跳过
                if hasattr(v, '__name__') and v.__name__ in seen:
                    continue
                # 如果 v 就是模块 m 自身，则跳过
                if v is m:
                    continue
                # 如果属性名以 '_' 开头，则跳过
                if k.startswith('_'):
                    continue
                # 如果属性 v 是模块类型，则递归检查其向后兼容性标记
                if isinstance(v, types.ModuleType):
                    check_symbols_have_bc_designation(v, seen)
                # 如果属性 v 是类型或者函数类型，则将其添加到 non_back_compat_objects 字典中
                elif isinstance(v, (type, types.FunctionType)):
                    if v not in _MARKED_WITH_COMPATIBILITY:
                        non_back_compat_objects.setdefault(v)

        # 检查 torch.fx 模块下的符号是否具有向后兼容性标记
        check_symbols_have_bc_designation(torch.fx, set())
        # 检查 torch.fx.passes 模块下的符号是否具有向后兼容性标记
        check_symbols_have_bc_designation(torch.fx.passes, set())

        # 获取非向后兼容对象的字符串表示列表
        non_back_compat_strs = [torch.typename(obj) for obj in non_back_compat_objects.keys()]
        # 仅保留位于 'torch.fx' 中的对象
        non_back_compat_strs = [
            s for s in non_back_compat_strs if s.startswith('torch.fx') and not s.startswith('torch.fx.experimental')]
        # 仅保留位于公共命名空间中的对象
        non_back_compat_strs = [
            s for s in non_back_compat_strs if all(not atom.startswith('_') for atom in s.split('.'))]
        # 对字符串列表进行排序
        non_back_compat_strs.sort()

        # 如果存在非向后兼容对象，则抛出断言错误
        if len(non_back_compat_strs) != 0:
            raise AssertionError(f"Public FX API(s) {non_back_compat_strs} introduced but not given a "
                                 f"backwards-compatibility classification! Please decorate these "
                                 f"API(s) with `@torch.fx._compatibility.compatibility` to specify "
                                 f"BC guarantees.")

    # 测试添加副作用函数
    def test_adding_side_effect_function(self):
        # 定义一个继承自 torch.nn.Module 的测试模块
        class TestModule(torch.nn.Module):
            # 前向传播函数
            def forward(self, x):
                # 调用副作用函数 side_effect_func
                side_effect_func(x)
                return x

        # 对 TestModule 进行符号化跟踪
        gm = torch.fx.symbolic_trace(TestModule())
        # 断言图中节点的数量为 3
        self.assertEqual(len(gm.graph.nodes), 3)
        # 消除死代码
        gm.graph.eliminate_dead_code()
        # 重新编译图
        gm.recompile()
        # 再次断言图中节点的数量为 3
        self.assertEqual(len(gm.graph.nodes), 3)
        
        # 在图的节点中查找是否存在调用 'side_effect_func' 的操作
        found = False
        for node in gm.graph.nodes:
            if node.op == 'call_function' and node.target == side_effect_func:
                found = True
        # 断言找到了对 'side_effect_func' 的调用
        self.assertTrue(found)
    # 定义一个测试方法，验证反序列化后保留未使用属性的情况
    def test_preserve_unused_attr_after_unpickle(self):
        # 对 Add 模块进行符号化跟踪，生成图模块（GraphModule）
        gm = torch.fx.symbolic_trace(Add())
        # 向图模块添加名为 "foo" 的子模块，值也为 Add 类的实例
        gm.add_submodule("foo", Add())
        # 向图模块注册一个未使用的缓冲区，空的 torch Tensor
        gm.register_buffer("dummy_buffer", torch.empty(1))
        # 向图模块注册一个未使用的参数，空的 torch Parameter
        gm.register_parameter("dummy_parameter", torch.nn.Parameter(torch.empty(1)))
        # 创建一个字节流对象
        b = io.BytesIO()
        # 将图模块 gm 序列化后保存到字节流对象 b 中
        torch.save(gm, b)
        # 将字节流对象的指针移动到开头
        b.seek(0)
        # 从字节流对象中加载反序列化后的图模块，并赋值给 reload_gm
        reload_gm = torch.load(b)
        # 断言重新加载的图模块 reload_gm 有属性 "foo"
        self.assertTrue(hasattr(reload_gm, "foo"))
        # 断言重新加载的图模块 reload_gm 有属性 "dummy_buffer"
        self.assertTrue(hasattr(reload_gm, "dummy_buffer"))
        # 断言重新加载的图模块 reload_gm 有属性 "dummy_parameter"
        self.assertTrue(hasattr(reload_gm, "dummy_parameter"))
# 当 Python 版本为 3.12 及以上时，跳过该测试类，因为在这些版本上出现问题
@unittest.skipIf(
    sys.version_info >= (3, 12), "Failing on python 3.12+"
)
class TestFunctionalTracing(JitTestCase):
    # 设置测试环境
    def setUp(self):
        super().setUp()
        # 检查在追踪期间是否存在可变操作的特性标志
        # 在测试中启用，但默认情况下禁用
        self.orig_tracer_mutable_flag = torch.fx.proxy.TracerBase.check_mutable_operations
        torch.fx.proxy.TracerBase.check_mutable_operations = True

    # 清理测试环境
    def tearDown(self):
        super().tearDown()
        torch.fx.proxy.TracerBase.check_mutable_operations = self.orig_tracer_mutable_flag

    # 忽略的函数列表
    IGNORE_FUNCS = ("has_torch_function", "has_torch_function_unary",
                    "has_torch_function_variadic", "handle_torch_function",
                    "boolean_dispatch")
    
    # 待替换的属性字典
    TO_PATCH = {"has_torch_function": None,
                "has_torch_function_unary": None,
                "has_torch_function_variadic": None}

    # 内建函数的异常类型
    BUILT_IN_FUNC = (AssertionError, "")
    
    # 代理不可迭代的异常类型和正则表达式
    PROXY_ITERABLE = (TypeError, r"argument of type 'Proxy' is not iterable")
    
    # 代理对象无法迭代的异常类型和正则表达式
    PROXY_ITERATED = (TraceError, r"Proxy object cannot be iterated")
    
    # 在符号追踪中不支持 `len` 操作的异常类型和正则表达式
    LEN_ERROR = (RuntimeError, r"'len' is not supported in symbolic tracing by default")
    
    # 参数类型不匹配的异常类型和正则表达式
    ARG_TYPE_MISMATCH = (TypeError, r", not Proxy$")
    
    # 符号追踪变量不能作为控制流输入的异常类型和正则表达式
    CONTROL_FLOW = (TraceError, r"symbolically traced variables cannot be used as inputs to control flow")
    
    # `size` 和 `scale_factor` 冲突的异常类型和正则表达式
    INTERPOLATE_ARGS_CONFLICT = (ValueError, r"only one of size or scale_factor should be defined")
    
    # 尝试追踪可变操作的异常类型和正则表达式
    MUTABLE = (RuntimeError, r"Tried to trace mutable operation")

    # nn.functionals 中具有张量输入但没有类型注释的函数列表
    FUNCTIONALS_WITHOUT_ANNOTATION = (
        "adaptive_max_pool1d",
        "adaptive_max_pool2d",
        "adaptive_max_pool3d",
        "fractional_max_pool2d",
        "fractional_max_pool3d",
        "max_pool1d",
        "max_pool2d",
        "max_pool3d",
        "gaussian_nll_loss",
        "upsample",
        "upsample_bilinear",
        "upsample_nearest",
    )

    # Python 3.8 与其他版本之间的不一致行为：
    # - Python 3.8+: 重新引发类似 `PROXY_ITERATED` 的内部异常
    # - 其他 Python 版本：由于相同的内部异常，抛出 `argument of type 'Proxy' is not iterable`
    # 使用以下映射覆盖 Python 3.8 的预期异常
    UNTRACEABLE_FUNCTIONALS_PY38 = {
        "adaptive_max_pool1d": PROXY_ITERATED,
        "adaptive_max_pool2d": PROXY_ITERATED,
        "adaptive_max_pool3d": PROXY_ITERATED,
        "fractional_max_pool2d": PROXY_ITERATED,
        "fractional_max_pool3d": PROXY_ITERATED,
        "max_pool1d": PROXY_ITERATED,
        "max_pool2d": PROXY_ITERATED,
        "max_pool3d": PROXY_ITERATED,
        "group_norm": CONTROL_FLOW
    }

    @classmethod
    # 获取所有小写的函数名列表，包括 torch.nn.functional 模块中定义的函数
    def _get_functional(cls):
        functional_list = []
        for f in dir(torch.nn.functional):
            # 如果函数名不是小写，则跳过
            if not f.islower():
                continue
            # 忽略以下划线开头的函数（内部函数）
            if f.startswith('_'):
                continue
            # 如果函数名在 cls.IGNORE_FUNCS 中，跳过（忽略支持函数）
            if f in cls.IGNORE_FUNCS:
                continue
            # 获取函数对象
            fn = getattr(torch.nn.functional, f)
            # 如果 fn 不是可调用对象（如模块），则跳过
            if not isinstance(fn, Callable):
                continue
            # 如果函数名不在 cls.FUNCTIONALS_WITHOUT_ANNOTATION 中
            if f not in cls.FUNCTIONALS_WITHOUT_ANNOTATION:
                try:
                    # 尝试获取函数的签名信息
                    sig = inspect.signature(fn)
                    has_tensor_arg = False
                    # 检查函数的参数中是否有类型为 torch.Tensor 的参数
                    for param in sig.parameters.values():
                        if isinstance(param.annotation, type) and issubclass(param.annotation, torch.Tensor):
                            has_tensor_arg = True
                    # 如果没有 torch.Tensor 类型的参数，则跳过该函数
                    if not has_tensor_arg:
                        continue
                # 如果获取函数签名时发生 ValueError 异常，则跳过该函数
                except ValueError:
                    pass
            # 将符合条件的函数名和函数对象添加到 functional_list 中
            functional_list.append((f, fn))
        return functional_list

    # 生成用于测试特定函数的测试函数，并返回该测试函数
    @classmethod
    def generate_test_func(cls, func_name, fn):

        def functional_test(self):
            # 如果 func_name 在 UNTRACEABLE_FUNCTIONALS_PY38 中，且 Python 版本在 3.8 到 3.11 之间
            if func_name in self.UNTRACEABLE_FUNCTIONALS_PY38 and \
                    sys.version_info >= (3, 8) and sys.version_info < (3, 12):
                # 获取异常和错误信息
                exc, err = self.UNTRACEABLE_FUNCTIONALS_PY38[func_name]
                # 断言调用 symbolic_trace(fn) 会引发特定的异常和错误信息
                with self.assertRaisesRegex(exc, err):
                    symbolic_trace(fn)
            # 如果 func_name 在 UNTRACEABLE_FUNCTIONALS 中
            elif func_name in self.UNTRACEABLE_FUNCTIONALS:
                # 获取异常和错误信息
                exc, err = self.UNTRACEABLE_FUNCTIONALS[func_name]
                # 断言调用 symbolic_trace(fn) 会引发特定的异常和错误信息
                with self.assertRaisesRegex(exc, err):
                    symbolic_trace(fn)
            # 否则，直接调用 symbolic_trace(fn)
            else:
                symbolic_trace(fn)
        return functional_test

    # 生成所有 torch.nn.functional 中函数的测试用例，并将其作为类的方法动态添加
    @classmethod
    def generate_tests(cls):
        functional_list = cls._get_functional()
        for func_name, fn in functional_list:
            # 构造测试用例的名称
            test_name = "test_nn_functional_" + func_name
            # 生成针对特定函数的测试函数
            functional_test = cls.generate_test_func(func_name, fn)
            # 将生成的测试函数作为类的方法动态添加
            setattr(cls, test_name, functional_test)

    # 在测试类设置之前调用，用于替换 torch.nn.functional 模块中指定函数的实现
    @classmethod
    def setUpClass(cls):

        # 定义一个无操作的函数 no，用于替换 torch.nn.functional 模块中的函数
        def no(*args, **kwargs):
            return False

        # 遍历需要替换的函数名列表
        for name in cls.TO_PATCH.keys():
            # 记录原始的函数对象
            cls.TO_PATCH[name] = getattr(torch.nn.functional, name)
            # 将 torch.nn.functional 模块中的指定函数替换为 no 函数
            setattr(torch.nn.functional, name, no)

    # 在测试类所有测试方法执行完成后调用，用于恢复 torch.nn.functional 模块中函数的原始实现
    @classmethod
    def tearDownClass(cls):
        # 遍历需要恢复的函数名列表
        for name in cls.TO_PATCH.keys():
            # 将 torch.nn.functional 模块中的函数替换为原始记录的函数对象
            setattr(torch.nn.functional, name, cls.TO_PATCH[name])
# 调用TestFunctionalTracing类的generate_tests方法，生成功能追踪测试
TestFunctionalTracing.generate_tests()

# 调用instantiate_device_type_tests函数，为TestOperatorSignatures类中的测试实例化设备类型，使用全局变量
instantiate_device_type_tests(TestOperatorSignatures, globals())

# 如果环境中存在TorchDynamo，且其速度过慢，则跳过该测试类
# 同时检查是否存在TorchVision，若不存在则跳过测试
@skipIfTorchDynamo("too slow")
@skipIfNoTorchVision
class TestVisionTracing(JitTestCase):
    def setUp(self):
        # 在追踪过程中检查可变操作的特性标志
        # 在测试中启用该标志，但默认情况下不启用
        self.orig_tracer_mutable_flag = torch.fx.proxy.TracerBase.check_mutable_operations
        torch.fx.proxy.TracerBase.check_mutable_operations = True

    def tearDown(self):
        # 恢复追踪器基类的可变操作检查标志为原始值
        torch.fx.proxy.TracerBase.check_mutable_operations = self.orig_tracer_mutable_flag

    # 追踪错误的代理对象
    PROXY_ITERATED = (TraceError, r"Proxy object cannot be iterated")
    # 返回值类型与注释类型不一致的错误
    INCONSISTENT_TYPE = (
        RuntimeError,
        r"Return value was annotated as having type __torch__.torchvision.models[.\w]+ but is actually of type Tensor"
    )

    # 不可追踪的模型及其对应的错误类型
    UNTRACEABLE_MODELS = {
        "fasterrcnn_resnet50_fpn": PROXY_ITERATED,
        "fasterrcnn_resnet50_fpn_v2": PROXY_ITERATED,
        "fasterrcnn_mobilenet_v3_large_320_fpn": PROXY_ITERATED,
        "fasterrcnn_mobilenet_v3_large_fpn": PROXY_ITERATED,
        "maskrcnn_resnet50_fpn": PROXY_ITERATED,
        "maskrcnn_resnet50_fpn_v2": PROXY_ITERATED,
        "keypointrcnn_resnet50_fpn": PROXY_ITERATED,
        "retinanet_resnet50_fpn": PROXY_ITERATED,
        "retinanet_resnet50_fpn_v2": PROXY_ITERATED,
        "ssd300_vgg16": PROXY_ITERATED,
        "fcos_resnet50_fpn": PROXY_ITERATED,
        "ssdlite320_mobilenet_v3_large": PROXY_ITERATED,
    }

    # 无法脚本化的模型及其对应的错误类型
    UNSCRIPTABLE_MODELS = {
        "googlenet": INCONSISTENT_TYPE,
        "inception_v3": INCONSISTENT_TYPE,
    }

    # 输出转换函数，将模型输出映射到特定的输出
    output_transform = {
        "fcn_resnet50": lambda x: x["out"],
        "fcn_resnet101": lambda x: x["out"],
        "deeplabv3_resnet50": lambda x: x["out"],
        "deeplabv3_resnet101": lambda x: x["out"],
        "deeplabv3_mobilenet_v3_large": lambda x: x["out"],
        "lraspp_mobilenet_v3_large": lambda x: x["out"],
        "fasterrcnn_resnet50_fpn": lambda x: x[1],
        "fasterrcnn_mobilenet_v3_large_fpn": lambda x: x[1],
        "fasterrcnn_mobilenet_v3_large_320_fpn": lambda x: x[1],
        "maskrcnn_resnet50_fpn": lambda x: x[1],
        "keypointrcnn_resnet50_fpn": lambda x: x[1],
        "retinanet_resnet50_fpn": lambda x: x[1],
    }

    @classmethod
    # 定义一个生成测试函数的方法，返回一个测试函数
    def generate_test_fn(cls, name, x, kwargs):
        # 定义实际的测试函数，内部逻辑为使用给定参数创建模型并测试其符号化跟踪的结果
        def run_test(self):
            # 使用 torchvision_models 模块中的函数根据名称和参数获取模型
            model = torchvision_models.get_model(name, **kwargs)
            # 将模型设置为评估模式
            model = model.eval()
            # 如果模型名称在 UNTRACEABLE_MODELS 列表中
            if name in self.UNTRACEABLE_MODELS:
                # 从 self.UNTRACEABLE_MODELS 中获取错误类型和正则表达式异常信息
                err, exc = self.UNTRACEABLE_MODELS[name]
                # 断言调用 symbolic_trace 函数时会抛出预期的异常
                with self.assertRaisesRegex(err, exc):
                    graph = symbolic_trace(model)
            else:
                # 否则，根据模型符号化跟踪，返回类型为 torch.fx.GraphModule
                graph : torch.fx.GraphModule = symbolic_trace(model)
                # 根据输出转换函数 out_transform 转换模型的输出，并比较预期结果
                a = out_transform(model(x))
                b = out_transform(graph(x))
                self.assertEqual(a, b)

                # 如果模型在 UNSCRIPTABLE_MODELS 列表中
                if name in self.UNSCRIPTABLE_MODELS:
                    # 获取预期的错误类型和异常信息
                    err, exc = self.UNSCRIPTABLE_MODELS[name]
                    # 断言调用 torch.jit.script 函数时会抛出预期的异常
                    with self.assertRaisesRegex(err, exc):
                        script = torch.jit.script(graph)
                else:
                    # 否则，尝试对符号化跟踪后的模型进行脚本化
                    script = torch.jit.script(graph)
                    # 使用输出转换函数转换脚本化模型的输出，并比较预期结果
                    c = out_transform(script(x))
                    self.assertEqual(a, c)

        return run_test

    # 定义生成分类测试的类方法
    @classmethod
    def generate_classification_tests(cls):
        # 遍历 torchvision_models 中的模型列表
        for k in torchvision_models.list_models(module=torchvision_models):
            # 构造测试名称
            test_name = 'test_torchvision_models_' + k
            # 根据模型名称选择合适的输入 x
            x = torch.rand(1, 3, 299, 299) if k in ['inception_v3'] else torch.rand(1, 3, 224, 224)
            # 设置 kwargs 参数，包括 num_classes=50
            kwargs = dict(num_classes=50)
            # 生成对应模型的测试函数
            model_test = cls.generate_test_fn(k, x, kwargs)
            # 将生成的测试函数设置为类的属性，即添加到类中作为一个测试方法
            setattr(cls, test_name, model_test)

    # 定义生成分割测试的类方法
    @classmethod
    def generate_segmentation_tests(cls):
        # 遍历 torchvision_models.segmentation 中的模型列表
        for k in torchvision_models.list_models(module=torchvision_models.segmentation):
            # 构造测试名称
            test_name = 'test_torchvision_models_segmentation_' + k
            # 设置输入 x
            x = torch.rand(1, 3, 32, 32)
            # 设置 kwargs 参数，包括 num_classes=10, pretrained_backbone=False
            kwargs = dict(num_classes=10, pretrained_backbone=False)
            # 生成对应模型的测试函数
            model_test = cls.generate_test_fn(k, x, kwargs)
            # 将生成的测试函数设置为类的属性，即添加到类中作为一个测试方法
            setattr(cls, test_name, model_test)

    # 定义生成检测测试的类方法
    @classmethod
    def generate_detection_tests(cls):
        # 遍历 torchvision_models.detection 中的模型列表
        for k in torchvision_models.list_models(module=torchvision_models.detection):
            # 构造测试名称
            test_name = 'test_torchvision_models_detection_' + k
            # 设置输入 x
            x = [torch.rand(3, 300, 300)]
            # 设置 kwargs 参数，包括 num_classes=10, pretrained_backbone=False
            kwargs = dict(num_classes=10, pretrained_backbone=False)
            # 生成对应模型的测试函数
            model_test = cls.generate_test_fn(k, x, kwargs)
            # 将生成的测试函数设置为类的属性，即添加到类中作为一个测试方法
            setattr(cls, test_name, model_test)

    # 定义生成视频测试的类方法
    @classmethod
    def generate_video_tests(cls):
        # 遍历 torchvision_models.video 中的模型列表
        for k in torchvision_models.list_models(module=torchvision_models.video):
            # 构造测试名称
            test_name = 'test_torchvision_models_video_' + k
            # 根据模型名称选择合适的输入 x
            x = (
                torch.rand(1, 3, 4, 112, 112)
                if k not in {"mvit_v1_b", "mvit_v2_s", "s3d"}
                else torch.rand(1, 3, 16, 224, 224)
            )
            # 设置 kwargs 参数，包括 num_classes=50
            kwargs = dict(num_classes=50)
            # 生成对应模型的测试函数
            model_test = cls.generate_test_fn(k, x, kwargs)
            # 将生成的测试函数设置为类的属性，即添加到类中作为一个测试方法
            setattr(cls, test_name, model_test)
    # 定义一个类方法 generate_tests，用于生成测试数据
    def generate_tests(cls):
        # 调用类方法 generate_classification_tests()，生成分类测试数据
        cls.generate_classification_tests()
        # 调用类方法 generate_detection_tests()，生成检测测试数据
        cls.generate_detection_tests()
        # 调用类方法 generate_segmentation_tests()，生成分割测试数据
        cls.generate_segmentation_tests()
        # 调用类方法 generate_video_tests()，生成视频测试数据
        cls.generate_video_tests()
# 如果存在 torchvision 模块，则执行 TestVisionTracing.generate_tests() 方法
if HAS_TORCHVISION:
    TestVisionTracing.generate_tests()

# 如果当前脚本作为主程序运行，则调用 run_tests() 函数
if __name__ == '__main__':
    run_tests()
```
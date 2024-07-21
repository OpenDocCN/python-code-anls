# `.\pytorch\test\functorch\test_aotdispatch.py`

```py
# 导入必要的库和模块

import copy  # 导入copy模块，用于对象复制
import itertools  # 导入itertools模块，用于高效循环迭代
import unittest  # 导入unittest模块，用于编写和运行测试
import warnings  # 导入warnings模块，用于警告处理
from contextlib import nullcontext  # 导入nullcontext，用于创建一个空的上下文管理器
from functools import partial, wraps  # 导入partial和wraps，用于函数部分应用和装饰器
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型提示相关的类和函数
from unittest.mock import patch  # 导入patch，用于模拟对象

from common_utils import (  # 导入自定义的common_utils模块中的多个函数和装饰器
    decorate,
    decorateForModules,
    skip,
    skipOps,
    xfail,
)

import torch  # 导入PyTorch库
import torch._dynamo as torchdynamo  # 导入torchdynamo，用于动态神经网络
import torch.nn as nn  # 导入torch.nn，用于神经网络构建
import torch.utils._pytree as pytree  # 导入pytree，PyTorch中的树数据结构工具
from functorch import (  # 从functorch库中导入多个函数和类
    grad,
    jacrev,
    make_fx,
    vjp,
    vmap,
)
from functorch.compile import (  # 导入functorch.compile中的多个函数
    aot_function,
    aot_module,
    aot_module_simplified,
    compiled_function,
    compiled_module,
    default_decompositions,
    default_partition,
    get_aot_compilation_context,
    make_boxed_compiler,
    make_boxed_func,
    memory_efficient_fusion,
    min_cut_rematerialization_partition,
    nnc_jit,
    nop,
)
from functorch.experimental import control_flow  # 导入functorch.experimental中的control_flow模块
from torch._decomp import decomposition_table  # 导入decomposition_table，用于分解表达式
from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache  # 导入AOTAutogradCache，AOT自动求导缓存
from torch._functorch.aot_autograd import (  # 导入aot_export_joint_simple和aot_export_module等AOT自动求导函数
    aot_export_joint_simple,
    aot_export_module,
)
from torch._higher_order_ops.out_dtype import out_dtype  # 导入out_dtype，用于输出数据类型
from torch._inductor.codecache import compiled_fx_graph_hash  # 导入compiled_fx_graph_hash，编译的fx图哈希值
from torch._subclasses.fake_tensor import (  # 导入FakeTensorMode和DynamicOutputShapeException等
    DynamicOutputShapeException,
    FakeTensorMode,
)
from torch.fx.experimental.proxy_tensor import is_sym_node  # 导入is_sym_node，用于代理张量操作
from torch.fx.experimental.symbolic_shapes import (  # 导入GuardOnDataDependentSymNode和ShapeEnv等
    GuardOnDataDependentSymNode,
    ShapeEnv,
)
from torch.nn.utils.rnn import PackedSequence  # 导入PackedSequence，用于RNN的打包序列
from torch.testing._internal.common_device_type import (  # 导入测试中的设备类型相关函数和数据
    instantiate_device_type_tests,
    ops,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_methods_invocations import (  # 导入测试中的方法调用相关函数
    op_db,
)
from torch.testing._internal.common_modules import (  # 导入测试中的模块相关数据
    module_db,
    modules,
)
from torch.testing._internal.common_utils import (  # 导入测试中的通用工具函数和类
    compare_equal_outs_and_grads,
    instantiate_parametrized_tests,
    IS_ARM64,
    IS_MACOS,
    IS_WINDOWS,
    IS_X86,
    outs_and_grads,
    parametrize,
    run_tests,
    skipIfRocm,
    skipIfTorchDynamo,
    TestCase,
    xfail_inherited_tests,
    xfailIfTorchDynamo,
)
from torch.testing._internal.custom_tensor import (  # 导入自定义张量相关类
    ConstantExtraMetadataTensor,
)
from torch.testing._internal.hop_db import hop_db  # 导入hop_db，用于操作点测试数据
from torch.testing._internal.optests import (  # 导入optests中的测试函数和工具
    _test_aot_autograd_forwards_backwards_helper,
    aot_autograd_check,
)
from torch.testing._internal.two_tensor import (  # 导入two_tensor中的TwoTensor和TwoTensorMode
    TwoTensor,
    TwoTensorMode,
)

# 尝试导入torchvision，设置USE_TORCHVISION标志
USE_TORCHVISION = False
try:
    import torchvision
    USE_TORCHVISION = True
except ImportError:
    # 如果导入失败，发出警告并提示用户安装torchvision
    warnings.warn(
        "Couldn't import torchvision. Some of our tests use it, try "
        "to install it with commands from pytorch.org, post-fixed with "
        "`--no-deps` to avoid overwriting the pytorch installation",
        UserWarning,
)
    # 该行代码仅包含一个右括号，用于闭合之前的代码块或函数调用
USE_NETWORKX = False
try:
    import networkx  # noqa: F401
    # 尝试导入 networkx 库，如果成功导入，则设置 USE_NETWORKX 为 True
    USE_NETWORKX = True
except ImportError:
    # 如果导入失败，则发出警告，说明一些测试需要使用 networkx 但未安装该库
    warnings.warn("Some tests use networkx but it was not installed", UserWarning)

# NB: numpy is a testing dependency!

# 定义 AOTTestCase 类，继承自 TestCase 类
class AOTTestCase(TestCase):
    pass

# 定义 TestPythonKey 类，继承自 AOTTestCase 类
class TestPythonKey(AOTTestCase):
    # 测试函数 test_make_fx，带参数 device
    def test_make_fx(self, device):
        # 定义函数 f(x)，返回 torch.sin(x)
        def f(x):
            return torch.sin(x)

        # 生成一个长度为 3 的随机张量 inp
        inp = torch.randn(3)
        # 调用 make_fx 函数，并传入函数 f，返回新的函数 fx_f
        fx_f = make_fx(f)(inp)

        # 生成一个新的长度为 3 的随机张量 new_inp
        new_inp = torch.randn(3)
        # 断言 fx_f(new_inp) 等于 f(new_inp)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    # 测试函数 test_make_fx_grad，带参数 device
    def test_make_fx_grad(self, device):
        # 定义函数 f(x)，返回 torch.sin(x).sum()
        def f(x):
            return torch.sin(x).sum()

        # 生成一个长度为 3 的随机张量 inp
        inp = torch.randn(3)
        # 对函数 f 应用 grad 函数
        f = grad(f)
        # 调用 make_fx 函数，并传入函数 f，返回新的函数 fx_f
        fx_f = make_fx(f)(inp)

        # 生成一个新的长度为 3 的随机张量 new_inp
        new_inp = torch.randn(3)
        # 断言 fx_f(new_inp) 等于 f(new_inp)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    # 测试函数 test_scalar_device，带参数 device
    def test_scalar_device(self, device):
        # 定义函数 f(a, b)，返回 a + b
        def f(a, b):
            return a + b

        # 生成一个列表，包含长度为 3 的随机张量和值为 5 的张量，设备为 device
        inps = [torch.randn(3, device=device), torch.tensor(5)]
        # 调用 make_fx 函数，并传入函数 f 和 inps 列表中的参数，返回新的函数 fx_f
        fx_f = make_fx(f)(*inps)
        # 断言 fx_f(*inps) 等于 f(*inps)
        self.assertEqual(fx_f(*inps), f(*inps))

    # 测试函数 test_make_fx_vmap，带参数 device
    def test_make_fx_vmap(self, device):
        # 定义函数 f(x)，返回 torch.sin(x)
        def f(x):
            return torch.sin(x)

        # 生成一个形状为 (5, 3) 的随机张量 inp
        inp = torch.randn(5, 3)
        # 对函数 f 应用 vmap 函数
        f = vmap(f)
        # 调用 make_fx 函数，并传入函数 f，返回新的函数 fx_f
        fx_f = make_fx(f)(inp)

        # 生成一个新的形状为 (5, 3) 的随机张量 new_inp
        new_inp = torch.randn(5, 3)
        # 断言 fx_f(new_inp) 等于 f(new_inp)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    # 测试函数 test_make_fx_jacrev，带参数 device
    def test_make_fx_jacrev(self, device):
        # 定义函数 f(x)，返回 x.sin().sum()
        def f(x):
            return x.sin().sum()

        # 生成一个长度为 3 的随机张量 inp
        inp = torch.randn(3)
        # 对函数 f 应用 jacrev 函数两次
        f = jacrev(jacrev(f))
        # 调用 make_fx 函数，并传入函数 f，返回新的函数 fx_f
        fx_f = make_fx(f)(inp)

        # 生成一个新的长度为 3 的随机张量 new_inp
        new_inp = torch.randn(3)
        # 断言 fx_f(new_inp) 等于 f(new_inp)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    # 测试函数 test_make_fx_vjp，带参数 device
    def test_make_fx_vjp(self, device):
        # 定义函数 f(x)，返回 torch.sin(x).sum()
        def f(x):
            return torch.sin(x).sum()

        # 生成一个长度为 3 的随机张量 primals
        primals = torch.randn(3)
        # 对函数 f 应用 vjp 函数，获取返回值中的 vjp_fn
        _, vjp_fn = vjp(f, primals)
        # 生成一个标量随机张量 cotangent
        cotangent = torch.randn(())
        # 调用 make_fx 函数，并传入 vjp_fn 函数及其它两个参数，返回新的函数 fx_f
        fx_f = make_fx(vjp_fn)(cotangent, True, True)

        # 生成一个新的标量随机张量 new_cotangent
        new_cotangent = torch.randn(())
        # 断言 fx_f(new_cotangent, True, True) 等于 vjp_fn(new_cotangent)
        self.assertEqual(fx_f(new_cotangent, True, True), vjp_fn(new_cotangent))

    # 测试函数 test_make_fx_functionalize，带参数 device
    def test_make_fx_functionalize(self, device):
        # 导入 functorch.experimental 模块中的 functionalize 函数
        from functorch.experimental import functionalize

        # 定义函数 fn(a)，操作 a，返回 a
        def fn(a):
            a = a * 2
            a.relu_()
            return a

        # 生成一个长度为 3 的随机张量 a
        a = torch.randn(3, device=device)
        # 对函数 fn 应用 symbolic_trace 函数
        symbolic_gm = torch.fx.symbolic_trace(fn)
        # 检查 symbolic_gm 中是否包含方法 relu_
        includes_method_relu_ = any(
            str(n.target) == "relu_" for n in symbolic_gm.graph.nodes
        )
        # 断言 includes_method_relu_ 为 True
        self.assertTrue(includes_method_relu_)
        
        # 调用 functionalize(symbolic_gm) 函数，并传入 a，返回新的图形模块 gm
        gm = make_fx(functionalize(symbolic_gm))(a)
        # 检查 gm 中是否包含 aten.relu.default 方法
        includes_aten_relu = any(
            n.target == torch.ops.aten.relu.default for n in gm.graph.nodes
        )
        # 断言 includes_aten_relu 为 True
        self.assertTrue(includes_aten_relu)
    # 测试函数，用于测试未能解析问题，即跳过测试并返回错误消息
    def test_make_fx_no_decompose(self, device):
        # FIXME
        # 跳过测试并返回错误消息，指示最大递归深度已达到
        return self.skipTest("error: maximum recursion reached")

        # 定义一个函数 f(x)，计算 torch.tanh(x) 的总和
        def f(x):
            return torch.tanh(x).sum()

        # 对函数 f 进行自动微分并生成 fx_f 函数
        fx_f = make_fx(grad(f))(torch.randn(5))
        # 获取 fx_f 函数计算图中的操作集合
        ops = {i.target for i in fx_f.graph.nodes}

        # 断言 torch.ops.aten.tanh_backward 是否在操作集合中
        self.assertEqual(torch.ops.aten.tanh_backward in ops, True)

        # 使用自定义的分解表格对函数 f 进行自动微分并生成 fx_f 函数
        fx_f = make_fx(grad(f), decomposition_table)(torch.randn(5))
        # 获取 fx_f 函数计算图中的操作集合
        ops = {i.target for i in fx_f.graph.nodes}
        # 断言 torch.ops.aten.tanh_backward 是否在操作集合中
        self.assertEqual(torch.ops.aten.tanh_backward in ops, False)

    # 测试 NNVM JIT 编译功能
    def test_nnc_jit(self, device):
        # 定义一个简单的函数 f(x)，计算 torch.sin(x)
        def f(x):
            return torch.sin(x)

        # 对函数 f 进行 NNVM JIT 编译
        jit_f = nnc_jit(f)

        # 创建输入张量
        inp = torch.randn(3)
        # 断言 JIT 编译后的函数结果与原始函数 f 的结果相等
        self.assertEqual(jit_f(inp), f(inp))

    # 测试 NNVM JIT 编译对标量输入的处理
    def test_nnc_scalar(self, device):
        # 定义一个简单的函数 f(x)，计算 torch.sin(x)
        def f(x):
            return torch.sin(x)

        # 对函数 f 进行 NNVM JIT 编译
        jit_f = nnc_jit(f)

        # 创建标量输入
        inp = torch.randn(())
        # 断言 JIT 编译后的函数结果与原始函数 f 的结果相等
        self.assertEqual(jit_f(inp), f(inp))

    # 测试 NNVM JIT 编译对 Python 树的处理
    def test_nnc_pytrees(self, device):
        # 定义一个简单的函数 f(x)，计算 [torch.sin(x[0])]
        def f(x):
            return [torch.sin(x[0])]

        # 对函数 f 进行 NNVM JIT 编译
        jit_f = nnc_jit(f)

        # 创建输入列表
        inp = [torch.randn(3)]
        # 断言 JIT 编译后的函数结果与原始函数 f 的结果相等
        self.assertEqual(jit_f(inp), f(inp))

    # 测试 NNVM JIT 编译对外部调用的支持
    def test_external_calls(self, device):
        # 定义一个简单的函数 f(a, b)，计算 torch.mv(a, b)
        def f(a, b):
            return torch.mv(a, b)

        # 对函数 f 进行 NNVM JIT 编译
        jit_f = nnc_jit(f)
        # 创建输入张量列表
        inp = [torch.randn(3, 3), torch.randn(3)]
        # 断言 JIT 编译后的函数结果与原始函数 f 的结果相等
        self.assertEqual(jit_f(*inp), f(*inp))

    # 测试 NNVM JIT 编译对参数传递的支持
    def test_nnc_passthrough(self, device):
        # 定义一个简单的函数 f(x, y)，返回 x + y 和 y
        def f(x, y):
            return x + y, y

        # 创建输入元组
        inp = (torch.randn(3), torch.randn(3))
        # 对函数 f 进行 NNVM JIT 编译
        jit_f = nnc_jit(f)
        # 断言 JIT 编译后的函数结果与原始函数 f 的结果相等
        self.assertEqual(jit_f(*inp), f(*inp))

        # 定义一个简单的函数 f(x)，修改输入字典 x 中键为 "a" 的值，返回修改后的字典
        def f(x):
            x["a"] = x["a"] * 2
            return x

        # 创建输入元组，包含一个字典作为唯一元素
        inp = ({"a": torch.randn(3), "b": torch.randn(3)},)
        # 对函数 f 进行 NNVM JIT 编译
        jit_f = nnc_jit(f)
        # 断言 JIT 编译后的函数结果与原始函数 f 的结果相等
        self.assertEqual(jit_f(*inp), f(*inp))

    # 如果没有安装 torchvision，跳过此测试
    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    # 测试 ResNet18 模型的反向追踪
    def test_resnet18_backward_trace(self, device):
        # 加载 torchvision 中的 ResNet18 模型
        mod = torchvision.models.resnet18()

        # 定义一个函数 f(x)，对输入 x 应用模型 mod，并计算输出的梯度
        def f(x):
            out = mod(x)
            out.sum().backward()
            return [a.grad for a in mod.parameters()]

        # 创建输入张量，需要梯度计算
        inp = torch.randn(3, 3, 250, 250, requires_grad=True)
        # 调用函数 f 获取模型参数的梯度
        grads = f(inp)

        # 清空模型参数的梯度
        mod.zero_grad()
        # 对输入张量应用模型并计算输出的总和的梯度
        mod(inp).sum().backward()
        # 获取重新计算得到的模型参数梯度
        grads2 = [a.grad for a in mod.parameters()]

        # 断言两次梯度计算得到的模型参数梯度列表相等
        self.assertEqual(grads, grads2)
# 返回对象 t 的基础对象，如果 t 是视图，则返回 t 的基础对象；否则返回 t 本身
def get_base(t):
    return t._base if t._is_view() else t


# 判断对象 t 是否存在于 maybe_tensors 中的基础对象中
def is_in_base(t, maybe_tensors):
    # 获取 t 的基础对象
    t_base = get_base(t)
    # 遍历 maybe_tensors 中的每个对象
    for maybe_tensor in maybe_tensors:
        # 如果 maybe_tensor 是 torch.Tensor 类型
        if isinstance(maybe_tensor, torch.Tensor):
            # 如果 t_base 和 maybe_tensor 的基础对象相同，返回 True
            if t_base is get_base(maybe_tensor):
                return True
    # 遍历完所有可能的 tensor 后，返回 False
    return False


# 装饰器函数，用于在特定条件下跳过测试函数
def skipIfDynamoInput(reason):
    """
    Skip TestAOTAutograd if running with dynamo input
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 如果当前测试对象是 TestAOTAutogradWithDynamo 的实例
            if isinstance(self, TestAOTAutogradWithDynamo):
                # 跳过当前测试，并显示跳过的原因
                self.skipTest(
                    f"Skipping {self._testMethodName} in TestAOTAutogradWithDynamo because {reason}"
                )
            else:
                # 否则，执行原始的测试函数
                func(self, *args, **kwargs)

        return wrapper

    return decorator


class TestAOTAutograd(AOTTestCase):
    # 运行 AOT Autograd 的方法，根据参数设置对函数 f 进行编译
    def run_autograd(
        self,
        f: Callable,
        fw_graph_cell: List[Optional[Callable]],
        decompositions: Optional[Dict],
        keep_input_mutations: bool,
        dynamic: bool,
    ):
        """
        Runs aot_autograd with the specified settings on f.
        """
        # 如果 f 是 nn.Module 类型，使用 aot_module 进行编译
        if isinstance(f, nn.Module):
            compiled_f = aot_module(
                f,
                fw_compiler=make_boxed_compiler(
                    partial(extract_graph, graph_cell=fw_graph_cell)
                ),
                bw_compiler=nop,
                decompositions=decompositions,
                keep_inference_input_mutations=keep_input_mutations,
                dynamic=dynamic,
            )
        else:
            # 否则，使用 aot_function 进行编译
            compiled_f = aot_function(
                f,
                fw_compiler=make_boxed_compiler(
                    partial(extract_graph, graph_cell=fw_graph_cell)
                ),
                bw_compiler=nop,
                decompositions=decompositions,
                keep_inference_input_mutations=keep_input_mutations,
                dynamic=dynamic,
            )
        # 返回编译后的函数对象
        return compiled_f

    # 验证 AOT Autograd 的方法，对给定的函数 f 进行测试
    # - 确保输入是非叶子节点，以便图可以对它们进行突变
    # - 尝试突变图的输出（以确保输出上正确设置 autograd 元数据）
    @patch("functorch.compile.config.debug_assert", True)
    def verify_aot_autograd(
        self,
        f,
        inp_: Union[Callable, List[Any]],
        *,
        test_mutation: bool = False,
        keep_inp_mutations: bool = False,
        decompositions: Optional[Dict] = None,
        dynamic: bool = False,
        # 当 inp_ 是 Callable 时才激活
        # TODO: 可能需要整合所有测试，使 inp 成为一个 Callable
        make_inputs_subclasses: bool = False,
    # 测试函数：验证函数接受非张量和空输入的情况
    def test_non_tensor_and_none_inputs(self):
        # 定义函数 f，接受三个参数，返回第一个和第三个参数的乘积
        def f(a, b, c):
            return a * c

        # 输入为整数、None、张量的列表
        inp = [2, None, torch.ones(3, 3, dtype=torch.float32, requires_grad=True)]
        # 使用 verify_aot_autograd 函数验证函数 f 在给定输入下的行为
        self.verify_aot_autograd(f, inp)
        
        # 输入为整数、None、不需要梯度的张量的列表
        inp = [2, None, torch.ones(3, 3, dtype=torch.float32, requires_grad=False)]
        # 使用 verify_aot_autograd 函数验证函数 f 在给定输入下的行为
        self.verify_aot_autograd(f, inp)

    # 测试函数：验证函数接受两个参数并返回它们的和
    def test_single_output(self):
        # 定义函数 f，接受两个参数，返回它们的和
        def f(a, b):
            return a + b

        # 输入为一个需要梯度的张量和一个不需要梯度的张量
        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3)]
        # 使用 verify_aot_autograd 函数验证函数 f 在给定输入下的行为
        self.verify_aot_autograd(f, inp)
        
        # 输入为一个不需要梯度的张量和一个不需要梯度的张量
        inp = [torch.randn(3, 3, requires_grad=False), torch.randn(3, 3)]
        # 使用 verify_aot_autograd 函数验证函数 f 在给定输入下的行为
        self.verify_aot_autograd(f, inp)

    # 测试函数：验证函数接受两个参数并返回它们的和与差
    def test_multi_output(self):
        # 定义函数 f，接受两个参数，返回它们的和与差
        def f(a, b):
            return a + b, a - b

        # 输入为一个需要梯度的张量和一个不需要梯度的张量
        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3)]
        # 使用 verify_aot_autograd 函数验证函数 f 在给定输入下的行为
        self.verify_aot_autograd(f, inp)
        
        # 输入为一个不需要梯度的张量和一个不需要梯度的张量
        inp = [torch.randn(3, 3, requires_grad=False), torch.randn(3, 3)]
        # 使用 verify_aot_autograd 函数验证函数 f 在给定输入下的行为
        self.verify_aot_autograd(f, inp)

    # 测试函数：验证函数接受两个参数并返回它们的和与差作为列表
    def test_multi_output_list(self):
        # 定义函数 f，接受两个参数，返回它们的和与差作为列表
        def f(a, b):
            return [a + b, a - b]

        # 输入为一个需要梯度的张量和一个不需要梯度的张量
        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3)]
        # 使用 verify_aot_autograd 函数验证函数 f 在给定输入下的行为
        self.verify_aot_autograd(f, inp)
        
        # 输入为一个不需要梯度的张量和一个不需要梯度的张量
        inp = [torch.randn(3, 3, requires_grad=False), torch.randn(3, 3)]
        # 使用 verify_aot_autograd 函数验证函数 f 在给定输入下的行为
        self.verify_aot_autograd(f, inp)

    # 测试函数：验证函数接受一个参数，并对其进行操作后返回结果
    # 此处的测试用例针对在伪张量和功能化的交集处出现的 bug
    def test_squeeze_mutation(self):
        # 定义函数 f，接受一个参数，进行克隆并进行压缩操作，然后返回操作后的结果
        def f(a):
            # 克隆参数 a 并对其进行压缩操作
            b = a.clone().squeeze(-1)
            # 对 b 执行加法操作
            b.add_(1.0)
            # 返回参数 a 与 b 相加的结果
            return a + b

        # 输入为一个需要梯度的张量
        inp = [torch.randn(3, 1, requires_grad=True)]
        # 使用 verify_aot_autograd 函数验证函数 f 在给定输入下的行为，并允许动态图
        self.verify_aot_autograd(f, inp, dynamic=True)
        
        # 输入为一个不需要梯度的张量
        inp = [torch.randn(3, 1, requires_grad=False)]
        # 使用 verify_aot_autograd 函数验证函数 f 在给定输入下的行为，并允许动态图
        self.verify_aot_autograd(f, inp, dynamic=True)

    # 测试函数：验证复杂线性模型的正向传播
    # 此处的测试用例是为了验证一个已知的 PyTorch 问题
    def test_complex_linear(self):
        # 定义输入为一个复数类型的张量
        inp = [torch.randn(1, 10, 10, dtype=torch.complex64)]

        # 定义一个继承自 nn.Module 的模型类 F
        class F(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在模型中定义一个复数类型的线性层
                self.linear = nn.Linear(10, 10, dtype=torch.complex64)

            # 实现模型的前向传播
            def forward(self, x):
                return self.linear(x).sum().abs()

        # 使用 verify_aot_autograd 函数验证模型类 F 在给定输入下的行为
        self.verify_aot_autograd(F(), inp)

    # 测试函数：验证嵌入包的正向传播，并检查动态图的情况
    # 此处的测试用例是为了测试在反向传播时，尝试将稀疏张量包装在 FunctionalTensorWrapper 中的行为
    def test_embedding_bag_view_dynamic(self):
        # 定义一个继承自 nn.Module 的模型类 F
        class F(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在模型中定义一个稀疏的 EmbeddingBag 层
                self.emb = torch.nn.EmbeddingBag(100, 8, sparse=True)

            # 实现模型的前向传播
            def forward(self, x, y):
                # 对 EmbeddingBag 层的输出执行视图变换
                return self.emb(x, y).view(-1)

        # 创建输入张量 x 和 y
        x = torch.arange(3)
        y = torch.arange(3)
        # 使用 verify_aot_autograd 函数验证模型类 F 在给定输入下的行为，同时禁用动态图
        self.verify_aot_autograd(F(), [x, y], dynamic=False)
        # 再次使用 verify_aot_autograd 函数验证模型类 F 在给定输入下的行为，允许动态图
        self.verify_aot_autograd(F(), [x, y], dynamic=True)
    # 定义一个测试方法，用于测试输入数据的变异简单情况
    def test_input_mutation_simple(self):
        # 定义内部函数 f，接受参数 a
        def f(a):
            # 对输入张量 a 执行就地乘法运算（乘以2）
            a.mul_(2)
            # 返回经过乘法运算后的结果，乘以3
            return a * 3

        # 创建一个包含单个全1张量的列表，张量需要梯度跟踪
        inp = [torch.ones(3, 3, requires_grad=True)]
        # 调用验证方法 verify_aot_autograd，测试变异（mutation）情况
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        
        # 创建一个包含单个全1张量的列表，张量不需要梯度跟踪
        inp = [torch.ones(3, 3, requires_grad=False)]
        # 再次调用验证方法 verify_aot_autograd，测试变异情况
        self.verify_aot_autograd(f, inp, test_mutation=True)
        
        # 注释：
        # - clone 操作的额外需求是因为我们需要将变异前的输入传递给 grad() 方法，
        #   但 autograd 操作在功能化之上，所以需要手动克隆。
        #   希望后端能够轻松优化这一点。
        # - 返回参数的额外需求是因为编译后的 forward 返回的是（变异后的输入 + 输出）。
        
        # 断言编译后的前向图代码与预期的内联代码相匹配
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    # 调用 Torch 库中的默认克隆操作，克隆 primals_1 并存储在 clone 中
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    # 对 clone 进行乘法运算，结果存储在 mul 中
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    # 对 mul 再次进行乘法运算，结果存储在 mul_1 中
    mul_1 = torch.ops.aten.mul.Tensor(mul, 3)
    # 返回 mul 和 mul_1 组成的列表作为输出
    return [mul, mul_1]
    return [add]""",
        )

    # 这是一个（希望是极其罕见的）极端情况，很难处理，因此我们禁止它。
    # https://github.com/pytorch/pytorch/issues/126236
    # https://github.com/pytorch/pytorch/pull/126113
    @xfailIfTorchDynamo
    def test_set__and_data_mutation_bad(self):
        # 定义函数 f，接受参数 a
        def f(a):
            # 创建 a 的视图 a_view
            a_view = a.view(-1)
            # 创建一个全为 1 的张量 tmp，形状为 (3, 3)，并要求梯度跟踪
            tmp = torch.ones(3, 3, requires_grad=True)
            # 使用 torch.no_grad() 上下文管理器，以下任何对 tmp 的变动都将被跟踪为图输入的变动。
            with torch.no_grad():
                # 使用 tmp 替换 a 的值
                a.set_(tmp)
                # 不好的做法：a_view 现在与每个图输入都分离，因此我们无法识别这导致了输入变动！
                a_view.mul_(2)
            # 返回 a 与 tmp 的和
            return a + tmp

        # 定义输入 inp 为一个包含一个全为 1 的 3x3 张量的列表
        inp = [torch.ones(3, 3, requires_grad=True)]
        # 断言捕获 RuntimeError 异常，指出“不能对冻结存储器进行变异”
        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            # 调用 self.verify_aot_autograd 验证函数 f，输入为 inp，
            # 测试变异为真，保持输入变异为真
            self.verify_aot_autograd(
                f, inp, test_mutation=True, keep_inp_mutations=True
            )

    @skipIfDynamoInput(
        "Test doesn't make sense with dynamo, which changes order of mutations"
    )
    def test_set__not_allowed(self):
        # 定义函数 f，接受参数 a 和 b
        def f(a, b):
            # 使用 torch.no_grad() 上下文管理器
            with torch.no_grad():
                # 使用 b 替换 a 的值
                a.set_(b)
            # 变异 a 将会改变 a 的 grad_fn，这要求我们在图外重放变异。
            # 当输入也接收到 set_() 输入变异时，我们目前禁止这种情况。
            a.mul_(2)
            # 返回 a 与 b 的和
            return a + b

        # 定义输入 inp 为两个全为 1 的 3x3 张量的列表
        inp = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
        ]
        # 断言捕获 AssertionError 异常，指出“但输入存在其他我们无法处理的变异”
        with self.assertRaisesRegex(
            AssertionError, "but the input has other mutations that we cannot"
        ):
            # 调用 self.verify_aot_autograd 验证函数 f，输入为 inp，
            # 测试变异为真，保持输入变异为真
            fw_graph = self.verify_aot_autograd(
                f, inp, test_mutation=True, keep_inp_mutations=True
            )

    def test_input_mutation_set__nop(self):
        # 定义函数 f，接受参数 a
        def f(a):
            # 创建一个 dtype 与 a 相同的张量 b，其值为 0 到 8 的整数
            b = torch.arange(9, dtype=a.dtype)
            # 获取 a 的旧值 a_old
            a_old = torch.ops.aten.alias.default(a)
            # 使用 torch.no_grad() 上下文管理器
            with torch.no_grad():
                # 使用 b 替换 a 的值
                a.set_(b)
                # 使用 a_old 替换 a 的值
                a.set_(a_old)
            # 返回 a 与 b 重塑为 (3, 3) 形状后的和
            return a + b.reshape(3, 3)

        # 定义输入 inp 为一个全为 1 的 3x3 张量的列表
        inp = [torch.ones(3, 3, requires_grad=True)]
        # 调用 self.verify_aot_autograd 验证函数 f，输入为 inp，
        # 测试变异为真，保持输入变异为真
        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )
        # 定义输入 inp 为一个全为 1 的 3x3 张量的列表
        inp = [torch.ones(3, 3, requires_grad=False)]
        # 调用 self.verify_aot_autograd 验证函数 f，输入为 inp，
        # 测试变异为真，保持输入变异为真
        self.verify_aot_autograd(f, inp, test_mutation=True, keep_inp_mutations=True)
        # 注意事项：
        # - 图中没有 set_() 调用（我们将 a.set_(b) 功能化为 "b"）
        # - 只有 **1** 个图输出。我们正确地意识到两次 set_() 调用互相抵消，
        #   因此实际上没有输入被改变。
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
    def forward(self, primals_1):
        # 使用 torch.ops.aten.arange.default 创建一个包含 0 到 8 的张量，数据类型为 torch.float32，在 CPU 上运行
        arange = torch.ops.aten.arange.default(9, dtype=torch.float32, device=device(type='cpu'), pin_memory=False)
        # 使用 torch.ops.aten.alias.default 创建 primals_1 的别名张量，并将 primals_1 置为 None
        alias = torch.ops.aten.alias.default(primals_1); primals_1 = None
        # 使用 torch.ops.aten.view.default 将 arange 重塑为一个 3x3 的张量，并释放 arange 引用
        view = torch.ops.aten.view.default(arange, [3, 3]); arange = None
        # 使用 torch.ops.aten.add.Tensor 将 alias 和 view 相加，并释放 alias 和 view 引用
        add = torch.ops.aten.add.Tensor(alias, view); alias = view = None
        # 返回包含 add 的列表
        return [add]
    def test_nested_subclasses(self):
        # 定义一个使用装饰器的函数，用于编译 Torch 计算图为 AOT eager 模式
        @torch.compile(backend="aot_eager")
        def f(x):
            # 对输入张量 x 执行 sin 和 cos 运算
            return x.sin().cos()

        # 创建一个 requires_grad=True 的全一张量 a
        a = torch.ones(4, requires_grad=True)
        # 克隆张量 a，并分离计算图，同时设置 requires_grad=True
        a2 = a.clone().detach().requires_grad_()
        # 创建 TwoTensor 对象 aa，包含张量 a 和 a2
        aa = TwoTensor(a, a2)
        # 克隆 TwoTensor 对象 aa，并分离计算图，同时设置 requires_grad=True
        aa2 = aa.clone().detach().requires_grad_()
        # 创建 TwoTensor 对象 aaaa，包含 aa 和 aa2
        aaaa = TwoTensor(aa, aa2)
        # 调用函数 f，并传入 aaaa 作为参数，获取输出 out
        out = f(aaaa)
        # 断言 out 是 TwoTensor 类的实例
        self.assertTrue(isinstance(out, TwoTensor))
        # 断言 out.a 是 TwoTensor 类的实例
        self.assertTrue(isinstance(out.a, TwoTensor))
        # 断言 out.b 是 TwoTensor 类的实例
        self.assertTrue(isinstance(out.b, TwoTensor))
        # 断言 out.a.a 是 torch.Tensor 类的实例
        self.assertTrue(isinstance(out.a.a, torch.Tensor))
        # 断言 out.a.b 是 torch.Tensor 类的实例
        self.assertTrue(isinstance(out.a.b, torch.Tensor))
        # 断言 out.b.a 是 torch.Tensor 类的实例
        self.assertTrue(isinstance(out.b.a, torch.Tensor))
        # 断言 out.b.b 是 torch.Tensor 类的实例
        self.assertTrue(isinstance(out.b.b, torch.Tensor))

        # 对 out 的所有元素求和并反向传播梯度
        out.sum().backward()
        # 断言 aaaa.grad 是 TwoTensor 类的实例
        self.assertTrue(isinstance(aaaa.grad, TwoTensor))
        # 断言 aaaa.grad.a 是 TwoTensor 类的实例
        self.assertTrue(isinstance(aaaa.grad.a, TwoTensor))
        # 断言 aaaa.grad.b 是 TwoTensor 类的实例
        self.assertTrue(isinstance(aaaa.grad.b, TwoTensor))

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/127470")
    def test_nested_subclasses_non_nested_grad(self):
        # 定义一个使用装饰器的函数，用于编译 Torch 计算图为 AOT eager 模式
        @torch.compile(backend="aot_eager")
        def f(x):
            # 对输入张量 x 执行 sin 和 cos 运算
            return x.sin().cos()

        # 创建一个 requires_grad=True 的全一张量 a
        a = torch.ones(4, requires_grad=True)
        # 克隆张量 a，并分离计算图，同时设置 requires_grad=True
        a2 = a.clone().detach().requires_grad_()
        # 克隆张量 a，并分离计算图，同时设置 requires_grad=True
        a3 = a.clone().detach().requires_grad_()
        # 克隆张量 a，并分离计算图，同时设置 requires_grad=True
        a4 = a.clone().detach().requires_grad_()
        # 创建 TwoTensor 对象 new_aa，包含张量 a3 和 a4
        new_aa = TwoTensor(a3, a4)
        # 创建 TwoTensor 对象 aa，包含张量 a 和 a2
        aa = TwoTensor(a, a2)

        # 克隆 TwoTensor 对象 aa，并分离计算图，同时设置 requires_grad=True
        aa2 = aa.clone().detach().requires_grad_()
        # 创建 TwoTensor 对象 aaaa，包含 aa 和 aa2
        aaaa = TwoTensor(aa, aa2)
        # 调用函数 f，并传入 new_aa 作为参数，获取输出 out
        out = f(new_aa)
        # 将 out 加上 aaaa 得到 new_out
        new_out = out + aaaa
        # 使用断言检查 RuntimeError 是否包含指定的错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            "The grad inputs should be same tensor subclass type as forward output",
        ):
            # 对 new_out 的所有元素求和并反向传播梯度
            new_out.sum().backward()

    @unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/127470")
    def test_custom_tensor_metadata(self):
        # 定义一个函数 f，接受参数 x，访问其元素和元素的元素，同时访问元素的常量属性，返回计算结果
        def f(x):
            x_elem = x.elem
            x_elem_elem = x_elem.elem
            x_elem_metadata = x_elem.constant_attribute
            return x * x_elem * x_elem_elem * x_elem_metadata

        # 创建一个 requires_grad=True 的全一张量 a
        a = torch.ones(4, requires_grad=True)
        # 使用 ConstantExtraMetadataTensor 类包装张量 a
        custom_a = ConstantExtraMetadataTensor(a)
        # 设置 custom_a 的常量属性为 6
        custom_a.constant_attribute = 6
        # 使用 ConstantExtraMetadataTensor 类包装 custom_a
        custom_aa = ConstantExtraMetadataTensor(custom_a)
        # 设置 custom_aa 的常量属性为 4
        custom_aa.constant_attribute = 4

        # 克隆 ConstantExtraMetadataTensor 对象 custom_aa，并分离计算图，同时设置 requires_grad=True
        custom_aa_compile = custom_aa.clone().detach().requires_grad_()
        # 访问 custom_aa_compile 的 elem 属性，并设置其常量属性为 6
        custom_aa_compile.elem.constant_attribute = 6
        # 调用函数 f，传入 custom_aa 作为参数，获取输出 out_eager
        out_eager = f(custom_aa)

        # 使用 torch.compile 将函数 f 编译为 AOT eager 模式
        compiled_f = torch.compile(f, backend="aot_eager")
        # 调用编译后的函数 compiled_f，传入 custom_aa_compile 作为参数，获取输出 out
        out = compiled_f(custom_aa_compile)

        # 使用断言检查 out_eager 和 out 是否在数值上近似相等
        self.assertTrue(torch.allclose(out_eager, out))

        # 对 out 的所有元素求和并反向传播梯度
        out.sum().backward()

        # 断言 custom_aa_compile.grad 是 ConstantExtraMetadataTensor 类的实例
        self.assertTrue(isinstance(custom_aa_compile.grad, ConstantExtraMetadataTensor))
        # 断言 custom_aa_compile.grad.elem 是 ConstantExtraMetadataTensor 类的实例
        self.assertTrue(
            isinstance(custom_aa_compile.grad.elem, ConstantExtraMetadataTensor)
        )
    # 跳过测试如果 Torch Dynamo 激活（参考链接：https://github.com/pytorch/pytorch/issues/127470）
    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/127470")
    def test_nested_subclasses_complicated_inps(self):
        # 定义函数 f，接受三个参数 x, y, z
        def f(x, y, z):
            # 计算 temp 为 x + y 的和
            temp = x + y
            # 计算 temp_plain 为 x.a + y.b 的和
            temp_plain = x.a + y.b
            # 计算 res 为 temp 和 temp_plain 的和，再求和
            res = temp.sum() + temp_plain.sum()
            # 返回 x 的 sin 函数应用后的余弦加上 res 的结果
            return x.sin().cos() + res
    
        # 创建一个 requires_grad 为 True 的 4x1 的全 1 张量 x
        x = torch.ones(4, requires_grad=True)
        # 克隆 x，并且将 requires_grad_() 设置为 True，得到 x2
        x2 = x.clone().detach().requires_grad_()
        # 使用 x 和 x2 创建 TwoTensor 对象 xx
        xx = TwoTensor(x, x2)
        # 克隆 xx，并且将 requires_grad_() 设置为 True，得到 xx2
        xx2 = xx.clone().detach().requires_grad_()
    
        # 使用 xx 和 xx2 创建嵌套的 TwoTensor 对象 x_nested
        x_nested = TwoTensor(xx, xx2)
        # 克隆 x_nested，并且将 requires_grad_() 设置为 True，得到 x_nested_compile
        x_nested_compile = x_nested.clone().detach().requires_grad_()
    
        # 克隆 x_nested，并且将 requires_grad_() 设置为 True，得到 y_nested
        y_nested = x_nested.clone().detach().requires_grad_()
        # 克隆 y_nested，并且将 requires_grad_() 设置为 True，得到 y_nested_compile
        y_nested_compile = y_nested.clone().detach().requires_grad_()
    
        # 克隆 x，并且将 requires_grad_() 设置为 True，得到 z
        z = x.clone().detach().requires_grad_()
        # 克隆 z，并且将 requires_grad_() 设置为 True，得到 z_compile
        z_compile = z.clone().detach().requires_grad_()
    
        # 计算使用 f 函数计算 x_nested, y_nested, z 的结果，存储在 out_eager 中
        out_eager = f(x_nested, y_nested, z)
        # 使用 torch.compile 函数编译 f 函数，选择后端为 "aot_eager"，得到 compiled_f 函数
        compiled_f = torch.compile(f, backend="aot_eager")
        # 使用 compiled_f 函数计算 x_nested_compile, y_nested_compile, z_compile 的结果，存储在 out 中
        out = compiled_f(x_nested_compile, y_nested_compile, z_compile)
        # 断言 out 与 out_eager 在数值上相似
        self.assertTrue(torch.allclose(out_eager, out))
    
        # 断言 out 的类型为 TwoTensor
        self.assertTrue(isinstance(out, TwoTensor))
        # 断言 out.a 的类型为 TwoTensor
        self.assertTrue(isinstance(out.a, TwoTensor))
        # 断言 out.b 的类型为 TwoTensor
        self.assertTrue(isinstance(out.b, TwoTensor))
        # 断言 out.a.a 的类型为 torch.Tensor
        self.assertTrue(isinstance(out.a.a, torch.Tensor))
        # 断言 out.a.b 的类型为 torch.Tensor
        self.assertTrue(isinstance(out.a.b, torch.Tensor))
        # 断言 out.b.a 的类型为 torch.Tensor
        self.assertTrue(isinstance(out.b.a, torch.Tensor))
        # 断言 out.b.b 的类型为 torch.Tensor
        self.assertTrue(isinstance(out.b.b, torch.Tensor))
    
        # 对 out 进行求和后进行反向传播
        out.sum().backward()
        # 对 out_eager 进行求和后进行反向传播
        out_eager.sum().backward()
    
        # 断言 x_nested_compile.grad 的类型为 TwoTensor
        self.assertTrue(isinstance(x_nested_compile.grad, TwoTensor))
        # 断言 x_nested_compile.grad.a 的类型为 TwoTensor
        self.assertTrue(isinstance(x_nested_compile.grad.a, TwoTensor))
        # 断言 x_nested_compile.grad.b 的类型为 TwoTensor
        self.assertTrue(isinstance(x_nested_compile.grad.b, TwoTensor))
    
        # 断言 y_nested_compile.grad 的类型为 TwoTensor
        self.assertTrue(isinstance(y_nested_compile.grad, TwoTensor))
        # 断言 y_nested_compile.grad.a 的类型为 TwoTensor
        self.assertTrue(isinstance(y_nested_compile.grad.a, TwoTensor))
        # 断言 y_nested_compile.grad.b 的类型为 TwoTensor
        self.assertTrue(isinstance(y_nested_compile.grad.b, TwoTensor))
    
        # 断言 x_nested_compile.grad.a.a 与 x_nested.grad.a.a 在数值上相似
        self.assertTrue(torch.allclose(x_nested_compile.grad.a.a, x_nested.grad.a.a))
        # 断言 x_nested_compile.grad.a.b 与 x_nested.grad.a.b 在数值上相似
        self.assertTrue(torch.allclose(x_nested_compile.grad.a.b, x_nested.grad.a.b))
        # 断言 y_nested_compile.grad.a.a 与 y_nested.grad.a.a 在数值上相似
        self.assertTrue(torch.allclose(y_nested_compile.grad.a.a, y_nested.grad.a.a))
        # 断言 y_nested_compile.grad.a.b 与 y_nested.grad.a.b 在数值上相似
        self.assertTrue(torch.allclose(y_nested_compile.grad.a.b, y_nested.grad.a.b))
    
    
    
    # 如果在 Windows 上运行测试，跳过此测试
    @unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/127470")
    def test_nested_subclasses_complicated_inps_mixed(self):
        # 定义内部函数 f，接受两个参数 x 和 y
        def f(x, y):
            # 从 y 中获取 elem 属性
            y_elem = y.elem
            # 从 y_elem 中获取 elem 属性
            y_elem_elem = y_elem.elem
            # 从 y_elem 中获取 constant_attribute 属性
            y_elem_metadata = y_elem.constant_attribute
            # 返回计算结果，涉及多个属性的乘积加上 x
            return y * y_elem * y_elem_elem * y_elem_metadata + x

        # 创建一个 requires_grad 为 True 的 4x1 的张量 x
        x = torch.ones(4, requires_grad=True)
        # 克隆 x，并且从计算图中分离，设置 requires_grad 为 True
        x2 = x.clone().detach().requires_grad_()
        # 使用 x 和 x2 创建 TwoTensor 对象 xx
        xx = TwoTensor(x, x2)
        # 克隆 xx，并且从计算图中分离，设置 requires_grad 为 True
        xx2 = xx.clone().detach().requires_grad_()

        # 使用 xx 和 xx2 创建嵌套的 TwoTensor 对象 x_nested
        x_nested = TwoTensor(xx, xx2)
        # 克隆 x_nested，并且从计算图中分离，设置 requires_grad 为 True
        x_nested_compile = x_nested.clone().detach().requires_grad_()

        # 创建一个 requires_grad 为 True 的 4x1 的张量 a
        a = torch.ones(4, requires_grad=True)
        # 使用 ConstantExtraMetadataTensor 包装张量 a
        custom_a = ConstantExtraMetadataTensor(a)
        # 设置 custom_a 的 constant_attribute 为 6
        custom_a.constant_attribute = 6
        # 克隆 custom_a，并且从计算图中分离
        custom_aa = ConstantExtraMetadataTensor(custom_a)
        # 设置 custom_aa 的 constant_attribute 为 4
        custom_aa.constant_attribute = 4

        # 克隆 custom_aa，并且从计算图中分离，设置 requires_grad 为 True
        custom_aa_compile = custom_aa.clone().detach().requires_grad_()
        # 设置 custom_aa_compile 的 constant_attribute 为 4
        custom_aa_compile.constant_attribute = 4
        # 设置 custom_aa_compile 内部 elem 的 constant_attribute 为 6
        custom_aa_compile.elem.constant_attribute = 6

        # 编译函数 f，使用 "aot_eager" 后端
        compiled_f = torch.compile(f, backend="aot_eager")
        # 调用 f 函数，传入 x_nested 和 custom_aa，获取输出 out_eager
        out_eager = f(x_nested, custom_aa)
        # 调用编译后的函数 compiled_f，传入 x_nested_compile 和 custom_aa_compile，获取输出 out
        out = compiled_f(x_nested_compile, custom_aa_compile)
        # 使用 assertTrue 验证 out_eager 和 out 在所有元素上是否近似相等
        self.assertTrue(torch.allclose(out_eager, out))

        # 对 out 求和并反向传播
        out.sum().backward()
        # 对 out_eager 求和并反向传播
        out_eager.sum().backward()

        # 使用 assertTrue 验证 x_nested_compile.grad 和 x_nested.grad 在所有元素上是否近似相等
        self.assertTrue(torch.allclose(x_nested_compile.grad, x_nested.grad))
        # 使用 assertTrue 验证 custom_aa_compile.grad 和 custom_aa.grad 在所有元素上是否近似相等
        self.assertTrue(torch.allclose(custom_aa_compile.grad, custom_aa.grad))

    @skipIfTorchDynamo("This test suite already uses dynamo")
    def test_composite_impl_compile(self):
        # 定义类 Foo，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 初始化方法，定义了一个线性层 self.linear
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            # 前向传播方法，接受参数 a
            def forward(self, a):
                return self.linear(a)

        # 创建一个 requires_grad 为 True 的 3x3 的张量列表 inp
        inp = [torch.ones(3, 3, requires_grad=True)]
        # 使用 verify_aot_autograd 验证 AOT 自动求导，返回前向图 fw_graph
        fw_graph = self.verify_aot_autograd(Foo(), inp, test_mutation=True)
        # 创建一个 requires_grad 为 False 的 3x3 的张量列表 inp
        inp = [torch.ones(3, 3, requires_grad=False)]
        # 使用 assertExpectedInline 验证 fw_graph.code 是否符合预期
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
    def forward(self, primals_1, primals_2, primals_3):
        # 调用 Torch 的 ATen 库中的 t.default 操作，对 primals_1 执行转置操作，将结果保存在 t 中；将 primals_1 置为 None
        t = torch.ops.aten.t.default(primals_1);  primals_1 = None
        # 调用 Torch 的 ATen 库中的 addmm.default 操作，对 primals_2 和 primals_3 执行矩阵相乘并加上 t，将结果保存在 addmm 中；将 primals_2 和 primals_3 置为 None
        addmm = torch.ops.aten.addmm.default(primals_2, primals_3, t);  primals_2 = primals_3 = t = None
        # 返回包含 addmm、primals_3 和 t 的列表作为结果
        return [addmm, primals_3, t]
    """
    # 使用 torch.ops.aten.mul.Tensor 对 clone 变量的值乘以 2，并将结果存储在 mul 变量中；然后将 clone 变量设为 None
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    # 使用 torch.ops.aten.mul.Tensor 对 clone_1 变量的值乘以 2，并将结果存储在 mul_1 变量中；然后将 clone_1 变量设为 None
    mul_1 = torch.ops.aten.mul.Tensor(clone_1, 2);  clone_1 = None
    # 使用 torch.ops.aten.add.Tensor 将 mul 变量和 primals_2 变量相加，并将结果存储在 add 变量中；然后将 primals_2 变量设为 None
    add = torch.ops.aten.add.Tensor(mul, primals_2);  primals_2 = None
    # 使用 torch.ops.aten.add.Tensor 将 add 变量和 mul_1 变量相加，并将结果存储在 add_1 变量中；然后将 add 变量设为 None
    add_1 = torch.ops.aten.add.Tensor(add, mul_1);  add = None
    # 返回包含 mul、mul_1 和 add_1 变量的列表作为函数的结果
    return [mul, mul_1, add_1]""",
        )



    # 定义内部函数 f(a, b)，该函数计算 torch.sin(a) 并将结果存储在 b 中
    def f(a, b):
        return torch.sin(a, out=b)

    # 创建包含两个 tensor 的列表 inp
    inp = [torch.randn(3, 3), torch.ones(3, 3)]

    # 调用 self.verify_aot_autograd 方法验证函数 f 的编译及自动求导，启用测试变异，并保留输入参数的变异
    fw_graph = self.verify_aot_autograd(
        f, inp, test_mutation=True, keep_inp_mutations=True
    )
    # 使用 self.assertExpectedInline 方法断言 fw_graph.code 的期望值，去除首尾空白字符
    self.assertExpectedInline(
        fw_graph.code.strip(),
        """\
# 定义一个方法 forward，接受两个参数 arg0_1 和 arg1_1
def forward(self, arg0_1, arg1_1):
    # 调用 torch 操作的 sin 函数，默认模式，计算 arg0_1 的正弦值；清空 arg0_1 引用
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    # 调用 torch 操作的 copy_ 函数，默认模式，将 sin 复制到 arg1_1；清空 arg1_1 和 sin 的引用
    copy_ = torch.ops.aten.copy_.default(arg1_1, sin);  arg1_1 = sin = None
    # 返回一个包含 copy_ 的元组
    return (copy_,)

# 定义一个方法 test_input_mutation_metadata
def test_input_mutation_metadata(self):
    # 定义一个函数 f，接受两个参数 a 和 b
    def f(a, b):
        # 对 a 执行转置操作，将其第 1 和第 0 维度进行交换
        a.transpose_(1, 0)
        # 返回 a 与 b 的和
        return a + b

    # 定义一个函数 create_inp，根据 req_grad 创建包含两个 shape 为 (3, 3) 的 tensor 的列表
    def create_inp(req_grad):
        return [
            torch.ones(3, 3, requires_grad=req_grad),
            torch.ones(3, 3, requires_grad=req_grad),
        ]

    # 调用 self 的 verify_aot_autograd 方法，传入函数 f、create_inp(True) 和 test_mutation=True
    self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
    # 调用 self 的 verify_aot_autograd 方法，传入函数 f、create_inp(False) 和 test_mutation=True
    self.verify_aot_autograd(f, create_inp(False), test_mutation=True)

# 定义一个方法 test_input_mutation_storage_resize_up
def test_input_mutation_storage_resize_up(self):
    # 定义一个函数 f，接受一个参数 a
    def f(a):
        # 调用 torch 操作的 resize_storage_bytes_ 函数，默认模式，将 a 的存储大小调整为 32 字节
        torch.ops.inductor.resize_storage_bytes_(a, 32)
        # 在 torch 操作的上下文中，使用 torch.no_grad() 禁止梯度计算
        with torch.no_grad():
            # 将 a 复制为一个所有元素为 1 的 tensor
            a.copy_(torch.ones(8))
        # 返回 a 加上 1 的结果
        return a + 1

    # 创建一个 shape 为 (8,) 的 tensor，要求计算梯度
    inp = torch.zeros(8, requires_grad=True)
    # 调用 untyped_storage() 方法获取 tensor 的未命名存储，然后将其大小调整为 0
    inp.untyped_storage().resize_(0)

    # 创建一个名为 fw_graph_cell 的列表，包含一个 None 元素
    fw_graph_cell = [None]
    # 将函数 f 编译为 AOT 函数，使用指定的编译器和解析器，保持输入推断变异
    compiled_f = aot_function(
        f,
        fw_compiler=make_boxed_compiler(
            partial(extract_graph, graph_cell=fw_graph_cell)
        ),
        bw_compiler=nop,
        decompositions={},
        keep_inference_input_mutations=True,
        dynamic=False,
    )
    # 对输入 inp 执行编译后的函数 compiled_f，并将结果赋给 out
    out = compiled_f(inp)
    # 断言 fw_graph_cell[0].code 的去除首尾空白后，与指定的字符串匹配
    self.assertExpectedInline(
        fw_graph_cell[0].code.strip(),
        """\
def forward(self, primals_1):
    # 调用 torch 操作的 resize_storage_bytes_ 函数，默认模式，将 primals_1 的存储大小调整为 32 字节
    resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(primals_1, 32)
    # 调用 torch 操作的 ones 函数，默认模式，创建一个所有元素为 1 的 tensor，形状为 [8]
    ones = torch.ops.aten.ones.default([8], device = device(type='cpu'), pin_memory = False)
    # 调用 torch 操作的 copy 函数，默认模式，将 ones 复制到 primals_1；清空 ones 的引用
    copy = torch.ops.aten.copy.default(primals_1, ones);  ones = None
    # 调用 torch 操作的 add.Tensor 函数，将 copy 和 1 相加
    add = torch.ops.aten.add.Tensor(copy, 1)
    # 调用 torch 操作的 copy_ 函数，默认模式，将 primals_1 复制到 copy；清空 primals_1 和 copy 的引用
    copy_ = torch.ops.aten.copy_.default(primals_1, copy);  primals_1 = copy = None
    # 返回包含 add 的列表
    return [add]""",
    )
    # 定义一个测试方法，用于测试输入变异和存储大小调整缩小的情况
    def test_input_mutation_storage_resize_down(self):
        # 定义内部函数f，接收参数a，并计算其正弦值
        def f(a):
            # 计算输入张量a的正弦值，并将结果保存在out中
            out = a.sin()
            # 调用torch.ops.inductor.resize_storage_bytes_函数，将a的存储大小调整为0
            torch.ops.inductor.resize_storage_bytes_(a, 0)
            # 返回计算得到的正弦值
            return out

        # 创建一个全零的张量inp，形状为(8,)，并标记为需要梯度计算
        inp = torch.zeros(8, requires_grad=True)

        # 创建一个列表fw_graph_cell，用于存储前向图的单元格
        fw_graph_cell = [None]
        # 调用aot_function函数，将函数f编译为AOT（Ahead-Of-Time）函数
        compiled_f = aot_function(
            f,  # 编译的函数对象为f
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),  # 使用包装的编译器，并从fw_graph_cell中提取图形
            bw_compiler=nop,  # 反向编译器使用nop（即空操作）
            decompositions={},  # 不使用任何分解方法
            keep_inference_input_mutations=True,  # 保持推断期间输入的变异
            dynamic=False,  # 不允许动态调整
        )

        # 调用编译后的函数compiled_f，传入输入张量inp，并将结果保存在out中
        out = compiled_f(inp)

        # 断言预期的内联文本和fw_graph_cell[0].code.strip()相同
        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
# 定义一个方法用于前向传播，接受一个名为 primals_1 的参数
def forward(self, primals_1):
    # 调用 torch 操作的 sin 方法，对 primals_1 进行正弦运算
    sin = torch.ops.aten.sin.default(primals_1)
    # 调用 torch 操作的 resize_storage_bytes_ 方法，对 primals_1 的存储空间进行重新调整
    resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(primals_1, 0)
    # 返回一个包含 sin 和 primals_1 的列表作为结果
    return [sin, primals_1]
    def test_input_mutation_storage_resize_down_and_set_(self):
        # Meant to mimic ppFSDP

        # 定义一个自定义的 Torch 自动求导函数，用于创建参数
        class TracableCreateParameter(torch.autograd.Function):
            @staticmethod
            def forward(ctx, tensor, placeholder):
                # 断言传入的 tensor 不需要梯度
                assert not tensor.requires_grad
                # 将 tensor 的值复制到 placeholder 上，并返回 placeholder
                return placeholder.set_(tensor)

            @staticmethod
            def backward(ctx, grad):
                # 反向传播时，梯度流向 placeholder
                return None, grad

        def f(dummy_param, param_shard):
            # 模拟数据的全局聚合
            with torch.no_grad():
                allgather_param = torch.cat([param_shard, param_shard])
            # 使用自定义的创建参数函数，将 allgather_param 的值复制到 dummy_param 上
            dummy_param_with_grad_state = TracableCreateParameter.apply(
                allgather_param, dummy_param
            )
            # 对 dummy_param 执行 sin() 操作
            out = dummy_param.sin()
            # 调整 dummy_param 的存储空间大小为 0
            torch.ops.inductor.resize_storage_bytes_(dummy_param, 0)
            return out

        # 模拟本地参数分片
        param_shard = torch.zeros(8, requires_grad=True)
        # 创建一个零大小的 dummy_param，用于计算梯度
        dummy_param = torch.zeros(16, requires_grad=True)
        dummy_param.untyped_storage().resize_(0)

        # 编译函数 f，并保留推断输入的变异
        fw_graph_cell = [None]
        compiled_f = aot_function(
            f,
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),
            bw_compiler=nop,
            decompositions={},
            keep_inference_input_mutations=True,
            dynamic=False,
        )
        # 调用编译后的函数，并传入 dummy_param 和 param_shard
        out = compiled_f(dummy_param, param_shard)
        # 下面是一些重要的说明：
        # (1) 我们为反向传播保存了 cat 操作（sin() 的输入）。
        #     虽然原始代码是 dummy_param.sin()，
        #     但由于 set_() 调用，dummy_param 实际上包含了 `cat` 张量的数据。
        # (2) 在图中我们生成了一个 cat.resize_storage_(0) 操作。
        #     在 set_() 调用之后，cat 就是 dummy_param 的实际数据，这是我们调用 resize_() 的对象。
        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1, primals_2):
    # 调用 torch.ops.aten.cat.default 方法将 primals_1 和 primals_2 进行连接
    cat = torch.ops.aten.cat.default([primals_2, primals_2]);  primals_2 = None
    # 调用 torch.ops.aten.sin.default 方法对 cat 中的张量进行正弦计算
    sin = torch.ops.aten.sin.default(cat)
    # 调用 torch.ops.inductor.resize_storage_bytes_.default 方法对 cat 的存储空间进行重新调整
    resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(cat, 0)
    # 调用 torch.ops.aten.set_.source_Tensor 方法将 cat 的值复制给 primals_1，然后将 primals_1 置为 None
    set_ = torch.ops.aten.set_.source_Tensor(primals_1, cat);  primals_1 = None
    # 返回 sin 和 cat 组成的列表
    return [sin, cat]
    def test_input_mutation_hidden_from_autograd_aliasing(self):
        # 定义内部函数 f，接受一个参数 a
        def f(a):
            # 创建 a 的视图 a_alias，并将其视图展平为一维数组
            a_alias = a.view(-1)
            # 进入 torch.no_grad() 上下文，禁用梯度追踪
            with torch.no_grad():
                # 在禁用梯度的上下文中，原地将 a_alias 中的所有元素乘以 2
                a_alias.mul_(2)
            # 返回 a 加上 1 的结果
            return a + 1

        # 创建一个包含一个元素的列表 inp，元素为 torch.ones(4, requires_grad=True)
        inp = [torch.ones(4, requires_grad=True)]
        # 调用 self.verify_aot_autograd 方法验证 AOT 自动梯度功能
        # 将函数 f、inp、test_mutation=True、keep_inp_mutations=True 作为参数传递
        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )
        # 断言预期的内联代码与 fw_graph.code 的去除首尾空白后的结果相等
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
    # 定义一个方法 forward，接受参数 primals_1
    def forward(self, primals_1):
        # 调用 torch.ops.aten.view.default 方法，对 primals_1 进行视图重塑为一维数组
        view = torch.ops.aten.view.default(primals_1, [-1])
        # 调用 torch.ops.aten.mul.Tensor 方法，对 view 中的每个元素乘以 2，结果存储在 mul 中；然后释放 view
        mul = torch.ops.aten.mul.Tensor(view, 2);  view = None
        # 再次调用 torch.ops.aten.view.default 方法，将 mul 重塑为形状为 [4] 的数组，结果存储在 view_1 中；然后释放 mul
        view_1 = torch.ops.aten.view.default(mul, [4]);  mul = None
        # 调用 torch.ops.aten.add.Tensor 方法，将 view_1 中的每个元素加上 1，结果存储在 add 中
        add = torch.ops.aten.add.Tensor(view_1, 1)
        # 调用 torch.ops.aten.copy_.default 方法，将 view_1 中的数据复制到 primals_1 中，然后释放 view_1 和 primals_1
        copy_ = torch.ops.aten.copy_.default(primals_1, view_1);  primals_1 = view_1 = None
        # 返回包含 add 的列表作为结果
        return [add]
    def test_input_mutation_batchnorm(self):
        def f(inpt, weight, bias, running_mean, running_var):
            # 定义一个函数 f，用于执行批量归一化操作，测试输入是否被正确克隆保存以用于反向传播
            # 在反向传播时，测试确保保存的是克隆后的输入，而不是原始被改变的输入
            return torch._native_batch_norm_legit(
                inpt, weight, bias, running_mean, running_var, True, 0.5, 1e-5
            )

        def create_inp(req_grad):
            # 创建输入列表，包含多个张量，其中一些需要梯度追踪
            return [
                torch.ones(2, 5, 5, 5, requires_grad=req_grad),
                torch.ones(5, requires_grad=req_grad),
                torch.ones(5, requires_grad=req_grad),
                torch.ones(5),
                torch.ones(5),
            ]

        from torch._decomp import get_decompositions

        # 获取分解后的函数列表，模拟执行正向和反向分解
        decompositions = get_decompositions(
            [
                torch.ops.aten._native_batch_norm_legit_functional,
                torch.ops.aten.native_batch_norm_backward,
            ]
        )
        # 验证运行编译后的自动求导模块，测试输入是否正确克隆保存
        self.verify_aot_autograd(
            f, create_inp(True), test_mutation=True, decompositions=decompositions
        )
        self.verify_aot_autograd(
            f, create_inp(False), test_mutation=True, decompositions=decompositions
        )

    def test_batchnorm_inference(self):
        inp = [
            torch.ones(2, 5, 5, 5, requires_grad=True),
            torch.ones(5, requires_grad=True),
            torch.ones(5, requires_grad=True),
            torch.ones(5),
            torch.ones(5),
        ]

        m = torch.nn.BatchNorm2d(4, 4)
        m.eval()
        fw_graph_cell = [None]
        inp = torch.ones(4, 4, 4, 4)
        fw_graph_cell = [None]
        compiled_m = aot_module(
            m,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=nop,
            keep_inference_input_mutations=True,
        )
        inp = torch.ones(4, 4, 4, 4)
        with torch.no_grad():
            out = compiled_m(inp)
        # 预期：在训练模式下（eval 模式），分解后的批量归一化操作不会包含 copy_() 调用
        code = fw_graph_cell[0].code.strip()
        self.assertTrue("copy_" not in str(code))

    def test_input_output_view_simple(self):
        def f(a):
            # 定义一个函数 f，对输入张量 a 执行视图变换（reshape）为一维
            return a.view(-1)

        inp = [torch.ones(2, 2, requires_grad=False).add(1)]
        # 验证运行编译后的自动求导模块，测试输入输出视图是否被正确处理
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(2, 2, requires_grad=True).add(1)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        # 预期：输出与输入有别名的情况下，不会对其进行编译处理
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    # 调用 Torch 操作符，对输入 primals_1 进行视图变换，将其展平为一维数组
    view = torch.ops.aten.view.default(primals_1, [-1]);  primals_1 = None
    # 返回处理后的视图作为列表的单个元素
    return [view]



    def test_input_output_view_mutate_multiple(self):
        def f(a, b, c):
            # 修改输入张量 a 的值，将其每个元素乘以2
            a.mul_(2)
            # 修改输入张量 c 的值，将其每个元素乘以3
            c.mul_(3)
            # 返回变换后的张量 b 和 c
            return b.view(2, 2), c.view(2, 2)

        def create_inp(req_grad):
            # 创建具有指定要求梯度属性的三个2x2全1张量，并将每个张量的每个元素加1
            return [
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
            ]

        # 验证不要求梯度的输入，返回变换后的张量 b 和 c
        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        # 验证要求梯度的输入，返回变换后的张量 b 和 c，并生成前向图
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        # 断言编译的前向图的预期格式
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    # 克隆 primals_1 张量并赋值给 clone
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    # 克隆 primals_3 张量并赋值给 clone_1
    clone_1 = torch.ops.aten.clone.default(primals_3);  primals_3 = None
    # 将 clone 张量的每个元素乘以2，并赋值给 mul
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    # 将 clone_1 张量的每个元素乘以3，并赋值给 mul_1
    mul_1 = torch.ops.aten.mul.Tensor(clone_1, 3);  clone_1 = None
    # 对 primals_2 张量进行形状变换为2x2，并赋值给 view
    view = torch.ops.aten.view.default(primals_2, [2, 2]);  primals_2 = None
    # 对 mul_1 张量进行形状变换为2x2，并赋值给 view_2
    view_2 = torch.ops.aten.view.default(mul_1, [2, 2])
    # 返回处理后的张量 mul, mul_1, view, view_2 作为列表元素
    return [mul, mul_1, view, view_2]""",
        )



    def test_input_output_view_metadata_mutate_multiple(self):
        def f(a, b, c):
            # 修改输入张量 b 的值，将其每个元素乘以3
            b.mul_(3)
            # 修改输入张量 c 的值，将其转置
            c.t_()
            # 返回变换后的张量 a, b, c
            return a.view(2, 2), b.view(2, 2), c.view(2, 2)

        def create_inp(req_grad):
            # 创建具有指定要求梯度属性的三个2x2全1张量，并将每个张量的每个元素加1
            return [
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
            ]

        # 验证不要求梯度的输入，返回变换后的张量 a, b, c
        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        # 验证要求梯度的输入，返回变换后的张量 a, b, c，并生成前向图
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        # 断言编译的前向图的预期格式
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    # 克隆 primals_2 张量并赋值给 clone
    clone = torch.ops.aten.clone.default(primals_2);  primals_2 = None
    # 对 primals_3 张量进行形状变换为2x2，并赋值给 view
    view = torch.ops.aten.view.default(primals_3, [2, 2]);  primals_3 = None
    # 使用 torch 操作符对 clone 张量乘以 3，存储在 mul 中；同时将 clone 设为 None
    mul = torch.ops.aten.mul.Tensor(clone, 3);  clone = None
    # 对 view 张量执行默认的转置操作，存储在 t 中；同时将 view 设为 None
    t = torch.ops.aten.t.default(view);  view = None
    # 对 primals_1 张量执行默认的重塑操作，形状为 [2, 2]，存储在 view_1 中；同时将 primals_1 设为 None
    view_1 = torch.ops.aten.view.default(primals_1, [2, 2]);  primals_1 = None
    # 对 t 张量执行默认的重塑操作，形状为 [2, 2]，存储在 view_3 中
    view_3 = torch.ops.aten.view.default(t, [2, 2])
    # 对 mul 张量执行默认的重塑操作，形状为 [2, 2]，存储在 view_4 中
    view_4 = torch.ops.aten.view.default(mul, [2, 2])
    # 返回包含 mul, t, view_1, view_4, view_3 的列表作为结果
    return [mul, t, view_1, view_4, view_3]
def forward(self, primals_1):
    # 使用 torch.ops.aten.clone.default 方法克隆 primals_1 张量，返回克隆的张量对象
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    # 使用 torch.ops.aten.add.Tensor 方法将克隆的张量对象和标量 1 相加，返回相加后的张量对象
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    # 使用 torch.ops.aten.view.default 方法对相加后的张量对象进行视图变换，返回变换后的张量对象
    view_1 = torch.ops.aten.view.default(add, [-1])
    # 返回视图变换后的张量对象列表
    return [add, view_1]""",
        )

def test_input_mutation_output_view_multiple(self):
    def f(a, b, c, d):
        # 将张量 b 进行转置操作，修改其内部数据，无返回值
        b.transpose_(1, 0)
        # 将张量 c 的所有元素加上标量 1，修改其内部数据，无返回值
        c.add_(1)
        # 返回 d 加 1，张量 b 的对角线元素，以及张量 a 与张量 c 相加的结果元组
        return d + 1, b.diagonal(), a + c

    # 创建四个张量列表，每个张量都包含四个元素，其中前两个张量设置 requires_grad 为 False，后两个设置为 True
    def create_inp(req_grad):
        return [
            torch.arange(4, requires_grad=req_grad, dtype=torch.float32)
            .view(2, 2)
            .add(1),
            torch.arange(4, requires_grad=req_grad, dtype=torch.float32)
            .view(2, 2)
            .add(1),
            torch.ones(2, 2, requires_grad=req_grad).add(1),
            torch.ones(2, 2, requires_grad=req_grad).add(1),
        ]

    # 验证函数 f 在不同输入下的行为，其中第一次调用创建的输入张量列表 requires_grad 为 False
    self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
    # 验证函数 f 在不同输入下的行为，其中第二次调用创建的输入张量列表 requires_grad 为 True
    fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
    # 断言预期的 AOTAutograd 编译后的代码，确保符合预期格式
    self.assertExpectedInline(
        fw_graph.code.strip(),
        """\
def forward(self, primals_1, primals_2, primals_3, primals_4):
    # 使用 torch.ops.aten.view.default 方法对 primals_2 张量进行形状变换为 [2, 2]，返回变换后的张量对象
    view = torch.ops.aten.view.default(primals_2, [2, 2]);  primals_2 = None
    # 使用 torch.ops.aten.clone.default 方法克隆 primals_3 张量，返回克隆的张量对象
    clone = torch.ops.aten.clone.default(primals_3);  primals_3 = None
    # 使用 torch.ops.aten.transpose.int 方法对 view 张量进行转置操作，返回转置后的张量对象
    transpose = torch.ops.aten.transpose.int(view, 1, 0);  view = None
    # 使用 torch.ops.aten.add.Tensor 方法将克隆的张量对象和标量 1 相加，返回相加后的张量对象
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    # 使用 torch.ops.aten.add.Tensor 方法将 primals_4 张量和标量 1 相加，返回相加后的张量对象
    add_1 = torch.ops.aten.add.Tensor(primals_4, 1);  primals_4 = None
    # 使用 torch.ops.aten.diagonal.default 方法获取 transpose 张量的对角线元素
    diagonal = torch.ops.aten.diagonal.default(transpose)
    # 使用 torch.ops.aten.add.Tensor 方法将 primals_1 张量和 add 张量相加，返回相加后的张量对象
    add_2 = torch.ops.aten.add.Tensor(primals_1, add);  primals_1 = None
    # 返回转置后的张量对象、相加后的张量对象、相加后的张量对象、对角线元素张量对象、以及相加后的张量对象列表
    return [transpose, add, add_1, diagonal, add_2]""",
    )

def test_output_aliases_intermediate_single(self):
    def f(a):
        # 使用 torch.ops.aten.mul.Tensor 方法将张量 a 中的每个元素乘以标量 3，返回乘法后的张量对象
        mul = torch.ops.aten.mul.Tensor(a, 3);  primals_1 = None
        # 使用 torch.ops.aten.view.default 方法对 mul 张量进行形状变换为 [-1]，返回变换后的张量对象
        view = torch.ops.aten.view.default(mul, [-1]);  mul = None
        # 返回形状变换后的张量对象列表
        return [view]""",
    )

inp = [torch.ones(3, 3, requires_grad=False)]
# 验证函数 f 在输入 inp 下的行为，其中张量 requires_grad 设置为 False
self.verify_aot_autograd(f, inp, test_mutation=True)
inp = [torch.ones(3, 3, requires_grad=True)]
# 验证函数 f 在输入 inp 下的行为，其中张量 requires_grad 设置为 True
fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
# 断言预期的 AOTAutograd 编译后的代码，确保符合预期格式及附加说明
self.assertExpectedInline(
    fw_graph.code.strip(),
    """\
def forward(self, primals_1):
    # 使用 torch.ops.aten.mul.Tensor 方法将 primals_1 张量中的每个元素乘以标量 3，返回乘法后的张量对象
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    # 使用 torch.ops.aten.view.default 方法对 mul 张量进行形状变换为 [-1]，返回变换后的张量对象
    view = torch.ops.aten.view.default(mul, [-1]);  mul = None
    # 返回形状变换后的张量对象列表
    return [view]""",
)
    # 定义一个测试方法，用于测试多输出视图别名时是否会引发 autograd 错误
    def test_output_aliases_input_multi_output_view_should_raise_autograd_error(self):
        
        # 定义一个函数 f1，接受参数 a，并返回 a 在维度 0 上的解绑定列表
        def f1(a):
            return list(a.unbind(0))
        
        # 将函数 f1 编译为 AOT 函数，使用 nop 作为编译选项
        f1_compiled = aot_function(f1, nop)
        
        # 创建三个张量 inp1, inp2, inp3，形状为 (3, 3)，并且需要梯度计算
        inp1 = torch.ones(3, 3, requires_grad=True).clone()
        inp2 = torch.ones(3, 3, requires_grad=True).clone()
        inp3 = torch.ones(3, 3, requires_grad=True).clone()
        
        # 使用断言来验证运行时错误，错误信息包含 "Such functions do not allow the output views"
        with self.assertRaisesRegex(
            RuntimeError, "Such functions do not allow the output views"
        ):
            # 调用 f1_compiled 函数并传入 inp1，将结果存储在 out_test1 中
            out_test1 = f1_compiled(inp1)
            # 在 eager 模式下，这里会引发 autograd 运行时错误
            out_test1[0].mul_(2)
        
        # 使用断言来验证运行时错误，错误信息包含 "Such functions do not allow the output views"
        with self.assertRaisesRegex(
            RuntimeError, "Such functions do not allow the output views"
        ):
            # 调用 f1_compiled 函数并传入 inp2，将结果存储在 out_test2 中
            out_test2 = f1_compiled(inp2)
            # 如果我们修改了一个张量，在 eager 模式下，任何多输出视图别名的梯度函数都会被替换为错误节点
            inp2.mul_(2)
            # 访问 out_test2[0] 的梯度函数应该会引发错误
            grad_fn = out_test2[0].grad_fn
        
        # 使用断言来验证运行时错误，错误信息包含 "Such functions do not allow the output views"
        with self.assertRaisesRegex(
            RuntimeError, "Such functions do not allow the output views"
        ):
            # 调用 f1_compiled 函数并传入 inp3，将结果存储在 out_test3 中
            out_test3 = f1_compiled(inp3)
            # 对于分离的别名也适用上述情况（它们将多输出视图别名的梯度函数转换为错误节点）
            out_test1[0].detach().mul_(2)
            # 同样的错误情况也适用于分离的别名
            grad_fn = out_test2[0].grad_fn
    # 测试函数，验证多输出视图中的输出别名问题
    def test_output_aliases_input_multi_output_view(self):
        # 所有别名输出都来自多输出视图，因此 AOTAutograd 将隐藏自动梯度中的别名。
        
        # 定义函数 f1，返回输入张量 a 沿着第一个维度解绑后的列表
        def f1(a):
            return list(a.unbind(0))

        # 创建需要梯度的张量 inp 和 inp_ref，形状为 3x3，初始值为全1
        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        
        # 编译函数 f1，使用 aot_function 转换为编译后的版本，nop 为无操作的空操作符
        f1_compiled = aot_function(f1, nop)

        # 计算引用版本和测试版本的输出
        out_ref = f1(inp_ref)
        out_test = f1_compiled(inp)
        
        # 断言：在反向传播图中，所有输出的梯度函数名包含 "CompiledFunctionBackward"，而不是 "AsStridedBackward"。
        # 对于多输出视图情况，无需重新生成视图。
        # 参见注释：[AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
        self.assertTrue(
            all("CompiledFunctionBackward" in str(o.grad_fn) for o in out_test)
        )

        # 对引用版本和测试版本进行反向传播并比较梯度
        sum(out_ref).sum().backward()
        sum(out_test).sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

        # 函数 f3 处理多输出视图，部分输出与输入张量别名
        def f3(a):
            return *list(a.unbind(0)), a.view(a.shape)

        # 创建需要梯度的张量 inp 和 inp_ref，形状为 3x3，初始值为全1
        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        
        # 编译函数 f3，使用 aot_function 转换为编译后的版本，nop 为无操作的空操作符
        f3_compiled = aot_function(f3, nop)

        # 克隆输入张量的引用版本和测试版本
        inp_ref_clone = inp_ref.clone()
        inp_clone = inp.clone()
        
        # 计算引用版本和测试版本的输出
        out_ref = f3(inp_ref_clone)
        out_test = f3_compiled(inp_clone)
        
        # 断言：前三个输出的梯度函数名都包含 "UnbindBackward"
        self.assertTrue(all("UnbindBackward" in str(o.grad_fn) for o in out_test[:3]))

        # 对引用版本和测试版本的最后一个输出进行修改
        out_ref[-1].mul_(2)
        out_test[-1].mul_(2)
        
        # 修改输入张量的视图，应影响别名输出
        inp_ref_clone.view(-1).mul_(3)
        inp_clone.view(-1).mul_(3)
        
        # 执行反向传播
        (inp_ref + out_ref[-1]).sum().backward()
        (inp + out_test[-1]).sum().backward()
        
        # 比较引用版本和测试版本的输入梯度
        self.assertEqual(inp_ref.grad, inp.grad)

    # 测试函数，验证输出别名与中间变异的线性函数
    def test_output_aliases_intermediate_mutation_linear(self):
        # 定义函数 f，对输入张量 x 加 1 后进行形状重塑为一维
        def f(x):
            return (x + 1).view(-1)

        # 创建包含需要梯度的张量列表 inp，每个张量形状为 3x3，初始值为全1
        inp = [torch.ones(3, 3, requires_grad=True)]
        
        # 导入 inductor 的分解操作，例如将 _unsafe_view() 转换为 view()
        from torch._inductor.decomposition import decompositions

        # 编译函数 f，使用 aot_function 转换为编译后的版本，nop 为无操作的空操作符，使用给定的分解操作
        f_compiled = aot_function(f, nop, decompositions=decompositions)

        # 计算引用版本和测试版本的输出
        out_ref = f(*inp)
        out_test = f_compiled(*inp)

        # 修改引用版本和测试版本的输出
        out_ref.mul_(2)
        out_test.mul_(2)
        
        # 断言：引用版本和测试版本的输出应相等
        self.assertEqual(out_ref, out_test)
        def test_output_aliases_intermediate_no_grad(self):
            # 定义一个测试函数，测试输出中间值不需要梯度的情况
            def f(a, b):
                # 计算 a 乘以 3 的结果
                out = torch.mul(a, 3)
                # 第一个输出是一个不需要梯度的中间值的别名
                # 返回重新视图化的 out，并且对 b 加 1
                return out.view(-1), b.add(1)

            # 输入张量列表，其中包含一个不需要梯度的张量
            inp = [torch.ones(3, 3), torch.ones(3, 3, requires_grad=False)]
            # 使用测试函数验证自动求导函数，同时测试是否允许突变
            self.verify_aot_autograd(f, inp, test_mutation=True)
            # 输入张量列表，其中包含一个需要梯度的张量
            inp = [torch.ones(3, 3), torch.ones(3, 3, requires_grad=True)]
            # 使用测试函数验证自动求导函数，同时测试是否允许突变
            fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
            # 重要说明：我们不生成中间基础作为图中的一个输出，
            # 因为这个中间基础本身不需要梯度。
            # （唯一的问题情况是当基础和别名化的输出都需要梯度时）。
            self.assertExpectedInline(
                fw_graph.code.strip(),
                """\
def test_output_aliases_intermediate_returned_multiple_times(self):
    def f(a):
        # 使用 torch.mul 对输入张量 a 执行乘法操作，乘数为 3
        out = torch.mul(a, 3)
        # 对乘法结果 out 执行视图操作，将其展平为一维张量
        out_view = out.view(-1)
        # 返回乘法结果 out，展平结果 out_view，以及乘法结果 out 再次作为返回值
        return out, out_view, out

    # 创建输入张量列表，每个张量都是 3x3 大小的全一张量，不要求梯度
    inp = [torch.ones(3, 3, requires_grad=False)]
    # 使用自定义函数 verify_aot_autograd 对函数 f 进行测试，验证是否支持 AOT 自动求导
    self.verify_aot_autograd(f, inp, test_mutation=True)
    # 创建输入张量列表，每个张量都是 3x3 大小的全一张量，要求梯度
    inp = [torch.ones(3, 3, requires_grad=True)]
    # 对函数 f 进行测试，获取前向图 fw_graph
    fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)

def test_output_aliases_intermediate_multiple(self):
    def f(a):
        # 使用 torch.mul 对输入张量 a 执行乘法操作，乘数为 3
        out = torch.mul(a, 3)
        # AOTAutograd 应该手动生成两个视图输出
        return out.view(-1), out.view(-1)

    # 创建输入张量列表，每个张量都是 3x3 大小的全一张量，不要求梯度
    inp = [torch.ones(3, 3, requires_grad=False)]
    # 使用自定义函数 verify_aot_autograd 对函数 f 进行测试，验证是否支持 AOT 自动求导
    self.verify_aot_autograd(f, inp, test_mutation=True)
    # 创建输入张量列表，每个张量都是 3x3 大小的全一张量，要求梯度
    inp = [torch.ones(3, 3, requires_grad=True)]
    # 对函数 f 进行测试，获取前向图 fw_graph
    fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
    # 断言预期的内联代码与 fw_graph 生成的代码是否一致
    self.assertExpectedInline(
        fw_graph.code.strip(),
        """\
def forward(self, primals_1):
    # 使用 torch.ops.aten.mul.Tensor 对输入张量 primals_1 执行乘法操作，乘数为 3
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    # 对乘法结果 mul 执行视图操作，将其展平为一维张量
    view = torch.ops.aten.view.default(mul, [-1])
    # 再次对乘法结果 mul 执行视图操作，生成第二个展平的一维张量 view_1
    view_1 = torch.ops.aten.view.default(mul, [-1])
    # 返回视图结果 view，视图结果 view_1，以及乘法结果 mul
    return [view, view_1, mul]""",
    )

def test_output_aliases_intermediate_and_returned(self):
    def f(a):
        # 使用 torch.mul 对输入张量 a 执行乘法操作，乘数为 3
        out = torch.mul(a, 3)
        # AOTAutograd 应该手动生成第一个输出（中间结果的视图）
        # 但不生成第二个输出（第一个输出的中间结果）
        return out.view(-1), out

    # 创建输入张量列表，每个张量都是 3x3 大小的全一张量，不要求梯度
    inp = [torch.ones(3, 3, requires_grad=False)]
    # 使用自定义函数 verify_aot_autograd 对函数 f 进行测试，验证是否支持 AOT 自动求导
    self.verify_aot_autograd(f, inp, test_mutation=True)
    # 创建输入张量列表，每个张量都是 3x3 大小的全一张量，要求梯度
    inp = [torch.ones(3, 3, requires_grad=True)]
    # 对函数 f 进行测试，获取前向图 fw_graph
    fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
    # 断言预期的内联代码与 fw_graph 生成的代码是否一致
    self.assertExpectedInline(
        fw_graph.code.strip(),
        """\
def forward(self, primals_1):
    # 使用 torch.ops.aten.mul.Tensor 对输入张量 primals_1 执行乘法操作，乘数为 3
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    # 对乘法结果 mul 执行视图操作，将其展平为一维张量
    view = torch.ops.aten.view.default(mul, [-1])
    # 返回视图结果 view，以及乘法结果 mul
    return [view, mul]""",
    )
        # 定义一个嵌套函数 f，接受参数 a
        def f(a):
            # 使用 Torch 的乘法函数对输入张量 a 进行操作，并将结果赋给 out
            out = torch.mul(a, 3)
            # AOTAutograd 应该手动生成第一个输出（中间视图），但不生成第二个（第二个输出本身是第一个的中间结果）
            # 返回两个张量，第一个是 out，第二个是将 out 重新视图化后的结果
            return out, out.view(-1)

        # 创建一个输入列表 inp，包含一个 requires_grad=False 的 3x3 全 1 张量
        inp = [torch.ones(3, 3, requires_grad=False)]
        # 调用 self.verify_aot_autograd 方法验证函数 f 的 AOTAutograd 特性，传入 inp 作为参数，并测试变异性为真
        self.verify_aot_autograd(f, inp, test_mutation=True)
        # 更新输入列表 inp，将其中的 requires_grad 设置为 True
        inp = [torch.ones(3, 3, requires_grad=True)]
        # 调用 self.verify_aot_autograd 方法验证函数 f 的 AOTAutograd 特性，传入更新后的 inp，并测试变异性为真
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        # 断言前向图的代码与预期的行内字符串相匹配
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def test_output_aliases_intermediate_and_returned_different_grad(self):
    def f(a):
        # 使用 Torch 的 multiply 操作，将输入张量 a 乘以 3
        out = torch.mul(a, 3)
        # AOTAutograd 手动生成第一个输出（中间结果的视图）
        # 但不会生成第二个输出（第一个输出的中间结果本身）
        return out.view(-1), out, out[0].detach()

    inp = [torch.ones(3, 3, requires_grad=False)]
    # 验证 AOTAutograd 是否正确生成代码
    self.verify_aot_autograd(f, inp, test_mutation=True)
    inp = [torch.ones(3, 3, requires_grad=True)]
    # 验证 AOTAutograd 是否正确生成代码并保存在 fw_graph 中
    fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
    self.assertExpectedInline(
        fw_graph.code.strip(),
        """\
def forward(self, primals_1):
    # 使用 Torch 的 multiply 操作，将输入张量 primals_1 乘以 3
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    # 使用 Torch 的 view 操作，将 mul 张量重塑成一维张量
    view = torch.ops.aten.view.default(mul, [-1])
    # 使用 Torch 的 select 操作，选择 mul 张量的第一个维度的第一个元素
    select = torch.ops.aten.select.int(mul, 0, 0)
    # 使用 Torch 的 detach 操作，返回 select 张量的副本并取消关联
    detach = torch.ops.aten.detach.default(select);  select = None
    # 使用 Torch 的 detach 操作，返回 detach 张量的副本并取消关联
    detach_1 = torch.ops.aten.detach.default(detach);  detach = None
    # 使用 Torch 的 detach 操作，返回 detach_1 张量的副本并取消关联
    detach_2 = torch.ops.aten.detach.default(detach_1);  detach_1 = None
    # 返回结果列表
    return [view, mul, detach_2]""",
    )
    # 定义一个测试函数，输入参数为 a
    def test_output_aliases_intermediate_inplace_view_and_view(self):
        
        # 定义内部函数 f，接受参数 a
        def f(a):
            # 计算 a 乘以 3 的结果，并赋值给 out
            out = torch.mul(a, 3)
            # 在 out 的基础上增加一个维度，赋值给 out_view
            out_view = out.unsqueeze(0)
            # 对 out 进行原地转置操作
            out.t_()
            # 再次在 out 的基础上增加一个维度，赋值给 out_view2
            out_view2 = out.unsqueeze(0)
            # 返回两个不同的视图和 out 本身
            return out_view, out, out_view2
        
        # 创建一个包含一个 2x4 的张量列表，每个张量都要求梯度计算
        inp = [torch.ones(2, 4, requires_grad=True)]
        
        # TODO: fix this test.
        # See <github issue link>
        # 调用自定义函数 verify_aot_autograd 进行测试，并记录 GitHub 上的问题链接
        # self.verify_aot_autograd(f, inp, test_mutation=True)

    # 定义一个测试函数，测试输出别名问题和多个混合操作
    def test_output_aliases_intermediate_multiple_mixed(self):
        
        # 定义内部函数 f，接受参数 a
        def f(a):
            # 计算 a 乘以 3 的结果，并赋值给 out1
            out1 = torch.mul(a, 3)
            # 计算 a 乘以 4 的结果，并赋值给 out2
            out2 = torch.mul(a, 4)
            # AOTAutograd 应该在结尾手动生成这两个输出视图
            # 返回 out1 的展平视图、out2 的转置后的结果以及 out1 的转置结果
            return out1.view(-1), out2.transpose(1, 0), out1.transpose(1, 0)
        
        # 创建一个包含一个 3x3 的张量列表，每个张量不要求梯度计算
        inp = [torch.ones(3, 3, requires_grad=False)]
        # 使用 verify_aot_autograd 函数验证 f 在 inp 上的运算，测试是否支持突变
        self.verify_aot_autograd(f, inp, test_mutation=True)
        
        # 创建一个包含一个 3x3 的张量列表，每个张量要求梯度计算
        inp = [torch.ones(3, 3, requires_grad=True)]
        # 使用 verify_aot_autograd 函数验证 f 在 inp 上的运算，测试是否支持突变
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        
        # 断言预期的内联代码和前面生成的前向图的代码一致
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    # 使用 Torch 操作符进行张量乘法，primals_1 乘以 3
    mul = torch.ops.aten.mul.Tensor(primals_1, 3)
    # 使用 Torch 操作符进行张量乘法，primals_1 乘以 4，并将 primals_1 置为 None
    mul_1 = torch.ops.aten.mul.Tensor(primals_1, 4);  primals_1 = None
    # 对 mul 进行视图操作，将其视图重塑为一维数组
    view = torch.ops.aten.view.default(mul, [-1])
    # 对 mul_1 进行转置操作，交换维度 0 和 1，并将 mul_1 置为 None
    transpose = torch.ops.aten.transpose.int(mul_1, 1, 0);  mul_1 = None
    # 对 mul 进行转置操作，交换维度 0 和 1
    transpose_1 = torch.ops.aten.transpose.int(mul, 1, 0)
    # 返回四个张量操作的结果数组
    return [view, transpose, transpose_1, mul]"""

# test_output_all_alias_types 函数的注释已经在示例中提供，不需要重复注释
    def test_input_data_and_metadata_mutation(self):
        # 定义一个内部函数 f，接受参数 a
        def f(a):
            # 在参数 a 上调用方法 t_()，原地修改张量 a 的数据
            a.t_()
            # 将张量 a 的第一个元素乘以 2，再次原地修改张量 a 的数据
            a[0].mul_(2)
            # 返回重新视图化后的张量 a
            return a.view(a.shape)

        # 创建一个包含不可导数的全一张量的列表 inp
        inp = [torch.ones(3, 3, requires_grad=False)]
        # 调用 self.verify_aot_autograd 函数验证自动求导和 AOT 的行为，测试数据变异性
        self.verify_aot_autograd(f, inp, test_mutation=True)
        
        # 创建一个包含可导数的全一张量的列表 inp
        inp = [torch.ones(3, 3, requires_grad=True)]
        # 调用 self.verify_aot_autograd 函数验证自动求导和 AOT 的行为，测试数据变异性
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        
        # 断言前向图的代码与预期的内联代码匹配
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    # 使用 torch 的操作函数对输入进行克隆
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    # 对克隆的张量进行转置操作
    t = torch.ops.aten.t.default(clone)
    # 从转置后的张量中选择第一个元素
    select = torch.ops.aten.select.int(t, 0, 0);  t = None
    # 将选定的元素乘以2
    mul = torch.ops.aten.mul.Tensor(select, 2);  select = None
    # 对克隆的张量再次进行转置操作
    t_1 = torch.ops.aten.t.default(clone);  clone = None
    # 使用 select_scatter 操作进行某种选择散播操作
    select_scatter = torch.ops.aten.select_scatter.default(t_1, mul, 0, 0);  t_1 = mul = None
    # 对转置后的张量再次进行转置操作
    t_2 = torch.ops.aten.t.default(select_scatter);  select_scatter = None
    # 进行连续两次转置操作
    t_4 = torch.ops.aten.t.default(t_2)
    t_6 = torch.ops.aten.t.default(t_2);  t_2 = None
    # 对 t_6 进行形状变换，变换为 3x3 的张量
    view_1 = torch.ops.aten.view.default(t_6, [3, 3]);  t_6 = None
    # 返回结果列表
    return [t_4, view_1]
    @skipIfDynamoInput("Dynamo removes runtime error")
    # 标记为跳过测试，以避免因动态输入而导致运行时错误

    def test_input_data_and_metadata_mutation_aliases_other_input(self):
        # 定义函数 f，参数 a 和 b 是别名
        def f(a, b):
            # 将 a 扩展成原地乘以 2
            a.mul_(2)
            # 转置 b，并且是原地操作
            b.t_()
            # 返回 a 与 b 相乘的结果
            return a.mul(b)

        # 定义一个可调用函数 inp_callable，根据 req_grad 创建一个全为 1 的 tensor
        def inp_callable(req_grad):
            base = torch.ones(2, 2, requires_grad=req_grad)
            # 在测试中，add() 操作很重要，因为需要将图的输入设置为非叶节点，以便进行变异。
            x = base.add(1)
            # 选择 x 的第一个元素作为 inp1 和 inp2
            inp1 = x[0]
            inp2 = x[0]
            # 返回 base 的列表和 inp1、inp2 的列表作为输出
            return [base], [inp1, inp2]

        # 验证静态编译自动求导，分别测试 req_grad 为 False 和 True 的情况，同时测试变异
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )

        # 使用断言检查是否捕获到预期的运行时错误信息，说明图中存在被变异的别名输入
        with self.assertRaisesRegex(
            RuntimeError,
            "Encountered aliased inputs that are mutated in the graph, but",
        ):
            self.verify_aot_autograd(
                f,
                partial(inp_callable, req_grad=False),
                test_mutation=True,
                make_inputs_subclasses=True,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "Encountered aliased inputs that are mutated in the graph, but",
        ):
            self.verify_aot_autograd(
                f,
                partial(inp_callable, req_grad=True),
                test_mutation=True,
                make_inputs_subclasses=True,
            )

    # 定义测试函数，用于检查非连续输入的变异
    # https://github.com/pytorch/pytorch/issues/106456
    def test_input_mutation_noncontiguous(self):
        # 定义函数 f，对输入 tensor a 原地乘以 2，然后加 1
        def f(a):
            a.mul_(2)
            return a + 1

        # 定义一个可调用函数 inp_callable，根据 req_grad 创建一个全为 1 的 tensor
        def inp_callable(req_grad):
            base = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            # 创建一个非连续视图以作为编译器的输入
            inp = x[:, 0]
            # 返回 base 的列表和 inp 的列表作为输出
            return [base], [inp]

        # 验证静态编译自动求导，分别测试 req_grad 为 False 和 True 的情况，同时测试变异
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )

        # 使用断言检查是否捕获到预期的运行时错误信息，说明在 tensor 子类上，非连续输入的变异操作目前不允许
        with self.assertRaisesRegex(
            RuntimeError,
            "Mutations on non-contiguous inputs are currently not allowed on tensor subclasses",
        ):
            self.verify_aot_autograd(
                f,
                partial(inp_callable, req_grad=False),
                test_mutation=True,
                make_inputs_subclasses=True,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "Mutations on non-contiguous inputs are currently not allowed on tensor subclasses",
        ):
            self.verify_aot_autograd(
                f,
                partial(inp_callable, req_grad=True),
                test_mutation=True,
                make_inputs_subclasses=True,
            )
    def test_backward_mutation_data(self):
        # 定义一个用于测试反向传播中数据变异的测试方法
        class BwMutation(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 在上下文中保存张量 x，用于反向传播
                ctx.save_for_backward(x)
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                # 从上下文中获取保存的张量 x
                (x,) = ctx.saved_tensors
                # 执行反向传播中的数据变异操作
                x.mul_(2)
                return grad_output.clone()

        def f(a, b):
            # 应用自定义的反向传播函数 BwMutation
            out = BwMutation.apply(b)
            return a * out

        inp_no_grad = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=False),
        ]

        # 对不需要梯度的缓冲区进行的反向传播变异是允许的
        self.verify_aot_autograd(f, inp_no_grad, test_mutation=True)

        inp_grad = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
        ]
        self.verify_aot_autograd(f, inp_grad, test_mutation=True)

    def test_backward_mutation_metadata(self):
        # 定义一个用于测试反向传播中元数据变异的测试方法
        class BwMutation(torch.autograd.Function):
            @staticmethod
            def forward(ctx, a, b):
                # 在上下文中保存张量 b，用于反向传播
                ctx.save_for_backward(b)
                return a.clone(), b.clone()

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                # 从上下文中获取保存的张量 b
                (b,) = ctx.saved_tensors
                # 执行反向传播中的元数据变异操作
                b.transpose_(1, 0)
                return grad_a.clone(), grad_b.clone()

        def f(a, b):
            # 应用自定义的反向传播函数 BwMutation
            a_, b_ = BwMutation.apply(a, b)
            out = a_ * b_
            return out

        inp_no_grad = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=False),
        ]

        # 预期会抛出异常，因为在反向传播中修改了元数据
        with self.assertRaisesRegex(
            AssertionError, "input that had its metadata mutated in the backward"
        ):
            self.verify_aot_autograd(f, inp_no_grad, test_mutation=True)

    def test_backward_mutation_on_grad_out(self):
        # 定义一个用于测试反向传播中输出梯度变异的测试方法
        class BwMutation(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                # 执行反向传播中的输出梯度变异操作
                grad_output.mul_(2)
                return grad_output.clone()

        def f(a, b):
            tmp = a * b
            out = BwMutation.apply(tmp)
            return out

        inp_grad = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
        ]
        f_compiled = aot_function(f, nop)
        # 预期会抛出异常，因为在反向传播输入中修改了梯度
        with self.assertRaisesRegex(
            AssertionError, "input to the backward that was mutated during the backward"
        ):
            out = f_compiled(*inp_grad)
    # 定义一个测试方法，用于测试反向突变和前向输入
    def test_backward_mutation_forward_inputs(self):
        # 定义一个自定义 Torch 操作 "_test::_clone"，它不突变任何参数
        @torch.library.custom_op("_test::_clone", mutates_args={})
        def f(x: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
            # 返回 x 的克隆
            return x.clone()

        # 定义一个假的函数 f_fake，返回一个与 x 维度相同但未初始化的张量
        def f_fake(x, x1):
            return torch.empty_like(x)

        # 定义反向传播函数，清零 ctx.x1 的梯度
        def backward(ctx, grad):
            with torch.no_grad():
                ctx.x1.zero_()
            # 返回梯度乘以 2 和 None
            return grad * 2, None

        # 设置上下文函数，将输入分配给 ctx.x 和 ctx.x1
        def setup_context(ctx, inputs, output):
            (x, x1) = inputs
            ctx.x = x
            ctx.x1 = x1

        # 注册假函数 f_fake 和自动求导的后向函数
        f.register_fake(f_fake)
        f.register_autograd(backward, setup_context=setup_context)

        # 定义 fn 函数，调用 Torch 操作 "_test::_clone"
        def fn(x: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
            return torch.ops._test._clone(x, x1)

        # 生成随机输入张量 inp_x 和 inp_x1，其中 inp_x 需要梯度，inp_x1 不需要梯度
        inp_x, inp_x1 = torch.randn(3, requires_grad=True), torch.randn(
            3, requires_grad=False
        )

        # 创建参考输入 ref_x 和 ref_x1，它们是输入张量的克隆
        ref_x, ref_x1 = inp_x.clone(), inp_x1.clone()
        # 计算参考输出 ref_y，调用 f 函数
        ref_y = f(ref_x, ref_x1)
        # 对 ref_y 的所有元素求和并进行反向传播
        ref_y.sum().backward()

        # 克隆输入张量 inp_x 和 inp_x1 到 x 和 x1
        x, x1 = inp_x.clone(), inp_x1.clone()
        # 编译 fn 函数为 Ahead-of-Time (AOT) 函数，使用 nop 作为编译选项
        compiled_f = aot_function(fn, nop)
        # 调用编译后的函数计算 y
        y = compiled_f(x, x1)
        # 计算 y 的总和作为损失
        loss = y.sum()
        # 对损失进行反向传播
        loss.backward()

        # 断言 ref_x 等于 x，ref_x1 等于 x1，ref_y 等于 y
        self.assertEqual(ref_x, x)
        self.assertEqual(ref_x1, x1)
        self.assertEqual(ref_y, y)

    # 部分解决 https://github.com/pytorch/pytorch/issues/106457

    # 定义一个测试方法，用于测试反向突变和前向输入，且创建计算图
    def test_backward_mutation_forward_inputs_create_graph(self):
        # 定义一个自定义 Torch 操作 "_test::_clone_create_graph"，它不突变任何参数
        @torch.library.custom_op("_test::_clone_create_graph", mutates_args={})
        def f(x: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
            # 返回 x 的克隆
            return x.clone()

        # 定义一个假的函数 f_fake，返回一个与 x 维度相同但未初始化的张量
        def f_fake(x, x1):
            return torch.empty_like(x)

        # 定义反向传播函数，清零 ctx.x1 的梯度
        def backward(ctx, grad):
            with torch.no_grad():
                ctx.x1.zero_()
            # 返回梯度乘以 2 和 None
            return grad * 2, None

        # 设置上下文函数，将输入分配给 ctx.x 和 ctx.x1
        def setup_context(ctx, inputs, output):
            (x, x1) = inputs
            ctx.x = x
            ctx.x1 = x1

        # 注册假函数 f_fake 和自动求导的后向函数
        f.register_fake(f_fake)
        f.register_autograd(backward, setup_context=setup_context)

        # 定义 fn 函数，调用 Torch 操作 "_test::_clone_create_graph"
        def fn(x: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
            return torch.ops._test._clone_create_graph(x, x1)

        # 生成随机输入张量 inp_x 和 inp_x1，都需要梯度
        inp_x, inp_x1 = torch.randn(3, requires_grad=True), torch.randn(
            3, requires_grad=True
        )

        # 创建参考输入 ref_x 和 ref_x1，它们是输入张量的克隆
        ref_x, ref_x1 = inp_x.clone(), inp_x1.clone()
        # 计算参考输出 ref_y，调用 f 函数
        ref_y = f(ref_x, ref_x1)
        # 对 ref_y 的所有元素求和并进行反向传播
        ref_y.sum().backward()

        # 克隆输入张量 inp_x 和 inp_x1 到 x 和 x1
        x, x1 = inp_x.clone(), inp_x1.clone()
        # 编译 fn 函数为 Ahead-of-Time (AOT) 函数，使用 nop 作为编译选项
        compiled_f = aot_function(fn, nop)
        # 调用编译后的函数计算 y
        y = compiled_f(x, x1)
        # 计算 y 的总和作为损失
        loss = y.sum()

        # 断言在创建计算图时抛出 RuntimeError，指定的错误消息
        with self.assertRaisesRegex(
            RuntimeError,
            "aot_autograd does not support input mutations with requires_grad in backward for create_graph=True",
        ):
            torch.autograd.grad(loss, inp_x, create_graph=True)

        # 断言 ref_x 等于 x，ref_x1 等于 x1，ref_y 等于 y
        self.assertEqual(ref_x, x)
        self.assertEqual(ref_x1, x1)
        self.assertEqual(ref_y, y)
        def test_input_mutation_false_aliasing(self):
            # 定义内部函数 f，接受两个参数 a 和 b，分别对 a 和 b 执行乘法操作，然后将它们的克隆版本展平并相加后返回
            def f(a, b):
                a.mul_(3)  # a 乘以 3
                b.mul_(2)  # b 乘以 2
                return a.clone().view(-1) + b.clone().view(-1)  # 返回 a 和 b 的克隆版本展平后的加和结果

            # No overlap, contiguous
            # 定义输入生成函数 inp_callable1，接受一个参数 req_grad，返回一个 base 张量和两个视图 a 和 b 的列表
            def inp_callable1(req_grad):
                base = torch.ones(4, 4, requires_grad=req_grad)  # 创建一个全为 1 的 4x4 张量 base
                x = base.add(1)  # 创建 x 张量，其值是 base 张量每个元素加 1 后的结果
                # 创建两个共享存储但实际上不重叠的视图 a 和 b
                a = x[0:2]  # 获取 x 的前两行作为 a
                b = x[2:4]  # 获取 x 的后两行作为 b
                return [base], [a, b]  # 返回 base 张量和视图 a、b 的列表

            # 调用验证自动求导的方法，验证函数 f 在 inp_callable1 生成的输入上是否能通过测试变异性
            fw_graph = self.verify_aot_autograd(
                f, partial(inp_callable1, req_grad=False), test_mutation=True
            )
            # 再次调用验证自动求导的方法，这次测试要求梯度追踪
            self.verify_aot_autograd(
                f, partial(inp_callable1, req_grad=True), test_mutation=True
            )
            # 第三次调用验证自动求导的方法，测试变异性，并使输入成为子类
            self.verify_aot_autograd(
                f,
                partial(inp_callable1, req_grad=False),
                test_mutation=True,
                make_inputs_subclasses=True,
            )
            # 当前的测试用例说明：带有训练图的子类输入变异失败了，今天的后向保护不支持。
            # 使用断言检查是否抛出了预期的异常信息，确保在带有子类元数据的情况下编译后向过程失败
            with self.assertRaisesRegex(
                AssertionError,
                "attempted to compile the backward with incorrect subclass metadata",
            ):
                self.verify_aot_autograd(
                    f,
                    partial(inp_callable1, req_grad=True),
                    test_mutation=True,
                    make_inputs_subclasses=True,
                )

            # 重要特征：图中有 2 个输入！
            # 这表明我们没有试图运行复杂的合成基本逻辑，因为我们成功检测到了跨两个输入的错误别名。
            self.assertExpectedInline(
                fw_graph.code.strip(),
                """\
# 定义一个方法 forward，接受两个参数 arg0_1 和 arg1_1
def forward(self, arg0_1, arg1_1):
    # 使用 torch.ops.aten.mul.Tensor 进行张量乘法，arg0_1 乘以 3，并将结果赋值给 mul
    mul = torch.ops.aten.mul.Tensor(arg0_1, 3);  arg0_1 = None
    # 使用 torch.ops.aten.mul.Tensor 进行张量乘法，arg1_1 乘以 2，并将结果赋值给 mul_1
    mul_1 = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None
    # 使用 torch.ops.aten.clone.default 复制 mul 的张量，并将结果赋值给 clone
    clone = torch.ops.aten.clone.default(mul)
    # 使用 torch.ops.aten.view.default 对 clone 进行视图变换，将其形状变为 [-1]，并将结果赋值给 view
    view = torch.ops.aten.view.default(clone, [-1]);  clone = None
    # 使用 torch.ops.aten.clone.default 复制 mul_1 的张量，并将结果赋值给 clone_1
    clone_1 = torch.ops.aten.clone.default(mul_1)
    # 使用 torch.ops.aten.view.default 对 clone_1 进行视图变换，将其形状变为 [-1]，并将结果赋值给 view_1
    view_1 = torch.ops.aten.view.default(clone_1, [-1]);  clone_1 = None
    # 使用 torch.ops.aten.add.Tensor 对 view 和 view_1 进行张量加法，并将结果赋值给 add
    add = torch.ops.aten.add.Tensor(view, view_1);  view = view_1 = None
    # 在 GPU 上运行的测试函数，检查内存泄漏问题
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_mem_leak_from_save_for_bw(self):
        # 查看此问题的全面诊断：https://github.com/pytorch/pytorch/issues/94990
        # 注意 [Detaching saved tensors in AOTAutograd]
        # 此程序创建了一个引用循环。长期来看，我们应该修复这个引用循环
        # （因为它可以自然地尽管偶尔地从 autograd.Function 的使用中出现）。
        # 但 AOTAutograd 使得从跟踪用户程序中更可能出现这种情况，
        # 因此我们通过手动分离保存用于反向的张量来处理它。
        # 如果我们进行双向传播，这完全是错误的并且会给出错误的结果。
        # 幸运的是，今天 AOTAutograd 明确禁止了双向传播。
        def f(a, b):
            # 计算 a + a
            add = a + a
            # 使用 torch.functional.split 对 add 在维度 1 上分割为长度为 [4, 4] 的张量列表
            split = torch.functional.split(add, [4, 4], dim=1)
            # 获取 split 中的第二个张量，赋值给 getitem_2
            getitem_2 = split[1]
            # 使用 unsqueeze 在最后一个维度上对 getitem_2 进行扩展
            unsqueeze = getitem_2.unsqueeze(-1)
            # 对 unsqueeze 和 b 进行张量乘法，将结果赋值给 mul
            mul = unsqueeze * b
            # 返回元组 (getitem_2, mul)
            return (getitem_2, mul)

        # 将函数 f 编译为 AOT 函数，使用 nop 作为参数
        f_compiled = aot_function(f, nop)
        # 准备输入张量列表 inps
        inps = [
            torch.ones(8, 8, device="cuda", requires_grad=True),
            torch.ones(1, 4, 1, device="cuda", requires_grad=True),
        ]
        # 记录 GPU 内存使用情况的开始值
        mem_before = torch.cuda.memory_allocated()
        # 执行编译后的函数 f_compiled，并传入 inps 中的参数
        f_compiled(*inps)
        # 记录 GPU 内存使用情况的结束值
        mem_after = torch.cuda.memory_allocated()
        # 使用 assertTrue 断言检查内存在函数执行前后是否一致
        self.assertTrue(mem_after == mem_before)
    def test_output_aliases_multiple_inputs_get_correct_one(self):
        # 定义一个测试函数，验证多个输入时的别名关系是否正确处理
        # 函数 f 接受两个参数 a 和 b，返回它们的视图
        def f(a, b):
            return a.view(a.shape), b.view(b.shape)

        # 定义一个生成输入的函数，根据需求是否需要梯度
        def inp_callable(req_grad):
            # 创建一个2x2的张量 base，并根据需求设置是否需要梯度
            base = torch.ones(2, 2, requires_grad=req_grad)
            # 对 base 的每个元素乘以2，得到新的张量 x
            x = base.mul(2)
            # 将 x 摊平成一维张量 inp1
            inp1 = x.view(-1)
            # 选择 x 的第一个元素作为 inp2
            inp2 = x[0]
            return [base], [inp1, inp2]

        # 调用自定义的验证函数 verify_aot_autograd 来测试函数 f
        # 使用 inp_callable 函数生成输入
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        # 同上，但此次要求梯度
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        # 同上，但使用 make_inputs_subclasses=True
        self.verify_aot_autograd(
            f,
            partial(inp_callable, req_grad=False),
            test_mutation=True,
            make_inputs_subclasses=True,
        )
        # 同上，但要求梯度并使用 make_inputs_subclasses=True
        self.verify_aot_autograd(
            f,
            partial(inp_callable, req_grad=True),
            test_mutation=True,
            make_inputs_subclasses=True,
        )

    def test_input_mutation_aliases_other_input(self):
        # 定义一个测试函数，验证输入的变异别名是否正常工作
        def f(a, b):
            # 将张量 a 的每个元素加1（原地操作）
            a.add_(1)
            # 返回 a 和 b 相加的结果
            return a + b

        # 定义一个生成输入的函数，根据需求是否需要梯度
        def inp_callable(req_grad):
            # 创建一个4x2的张量 base，并根据需求设置是否需要梯度
            base = torch.ones(4, 2, requires_grad=req_grad)
            # 对 base 的每个元素加1，得到新的张量 x
            x = base.add(1)
            # 选择 x 的第一个元素作为 inp1 和 inp2
            inp1 = x[0]
            inp2 = x[0]
            return [base], [inp1, inp2]

        # 调用自定义的验证函数 verify_aot_autograd 来测试函数 f
        # 使用 inp_callable 函数生成输入
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        # 同上，但此次要求梯度
        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        # 重要的图形部分：
        # - 编译的图形接受一个 base，我们生成 a 和 b（视图）基于 base
        # - clone() 仍然在图形中，因为我们需要在原始（未变异）输入上调用 grad()
        # - 我们在克隆之后重新生成视图，以保持视图之间的关系
        # 断言编译后的代码是否符合预期
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    # 使用 torch.ops.aten.clone.default 方法克隆 primals_1 张量
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    # 使用克隆的张量创建一个步幅视图张量，形状为 [2]，步幅为 [1]，偏移为 0
    as_strided = torch.ops.aten.as_strided.default(clone, [2], [1], 0)
    # 将常数值 1 加到步幅视图张量上，生成新的张量
    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None
    # 使用 torch.ops.aten.as_strided_scatter.default 方法基于 clone 和 add 创建一个步幅散布视图张量
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, add, [2], [1], 0);  clone = add = None
    # 再次使用 as_strided_scatter 创建步幅视图张量，形状为 [2]，步幅为 [1]，偏移为 0
    as_strided_2 = torch.ops.aten.as_strided.default(as_strided_scatter, [2], [1], 0)
    # 使用 as_strided_scatter 创建另一个步幅视图张量，形状为 [2, 2]，步幅为 [2, 1]，偏移为 0
    as_strided_5 = torch.ops.aten.as_strided.default(as_strided_scatter, [2, 2], [2, 1], 0)
    # 将 as_strided_2 和 as_strided_5 相加，生成新的张量
    add_1 = torch.ops.aten.add.Tensor(as_strided_2, as_strided_5);  as_strided_2 = as_strided_5 = None
    # 返回一个包含两个张量的列表
    return [as_strided_scatter, add_1]
    # 使用 torch.ops.aten.clone.default 对 primals_1 进行克隆操作，并将其赋值给 clone；同时将 primals_1 置为 None
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    # 使用 torch.ops.aten.as_strided.default 对 clone 进行视图操作，创建具有指定大小和步幅的张量
    as_strided = torch.ops.aten.as_strided.default(clone, [4], [1], 0)
    # 使用 torch.ops.aten.add.Tensor 对 as_strided 张量加上标量 1，同时将 as_strided 置为 None
    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None
    # 使用 torch.ops.aten.as_strided_scatter.default 对 clone 和 add 进行散布视图操作，创建指定大小和步幅的张量
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, add, [4], [1], 0);  clone = add = None
    # 使用 torch.ops.aten.as_strided.default 对 as_strided_scatter 进行视图操作，创建具有指定大小和步幅的张量
    as_strided_8 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    # 使用 torch.ops.aten.view.default 对 as_strided_8 进行视图操作，创建具有指定形状的张量；同时将 as_strided_8 置为 None
    view_1 = torch.ops.aten.view.default(as_strided_8, [4]);  as_strided_8 = None
    # 返回 as_strided_scatter 和 view_1 的列表作为结果
    return [as_strided_scatter, view_1]""",
        )  # noqa: B950


这些代码行涉及使用 PyTorch 的张量操作函数对张量进行克隆、视图、加法和视图变换操作。
def forward(self, primals_1, primals_2):
    # 调用 torch 库中的 aten.clone.default 方法，克隆 primals_1，并将其赋值给 clone；同时将 primals_1 置为 None
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    # 使用 torch 库中的 aten.as_strided.default 方法，基于 clone 创建视图 as_strided_1，形状为 [4]，步幅为 [1]，偏移为 0
    as_strided_1 = torch.ops.aten.as_strided.default(clone, [4], [1], 0)
    # 调用 torch 库中的 aten.mul.Tensor 方法，将 as_strided_1 中的每个元素乘以 2，并将结果赋值给 mul；同时将 as_strided_1 置为 None
    mul = torch.ops.aten.mul.Tensor(as_strided_1, 2);  as_strided_1 = None
    # 使用 torch 库中的 aten.as_strided_scatter.default 方法，在 clone 的基础上，按照 mul 的内容创建 scatter 视图 as_strided_scatter，形状为 [4]，步幅为 [1]，偏移为 0；同时将 clone 和 mul 置为 None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, mul, [4], [1], 0);  clone = mul = None
    # 调用 torch 库中的 aten.add.Tensor 方法，将 primals_2 中的每个元素加 1，并将结果赋值给 add；同时将 primals_2 置为 None
    add = torch.ops.aten.add.Tensor(primals_2, 1);  primals_2 = None
    # 使用 torch 库中的 aten.as_strided.default 方法，基于 as_strided_scatter 创建视图 as_strided_7，形状为 [4]，步幅为 [1]，偏移为 0
    as_strided_7 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    # 调用 torch 库中的 aten.view.default 方法，将 as_strided_7 视图重塑为一个一维张量 view_1，形状为 [-1]；同时将 as_strided_7 置为 None
    view_1 = torch.ops.aten.view.default(as_strided_7, [-1]);  as_strided_7 = None
    # 返回包含 as_strided_scatter、add 和 view_1 的列表作为结果
    return [as_strided_scatter, add, view_1]
    def test_input_mutation_aliases_and_none_require_gradients(self):
        def f(a, b, c):
            # 对参数 a 进行原位乘法操作（in-place multiplication）
            # 参数 a 和 b 虽然别名，但都不需要梯度（因此它们没有 _base）
            # aot autograd 应该从 `torch.Tensor(a.storage())` 构造出合成的基础
            a.mul_(2)
            # 返回 b 加 1 和 c 加 1 的结果
            return b + 1, c + 1

        def inp_callable(req_grad):
            # 创建一个全为 1 的张量作为基础
            base = torch.ones(2, 2)
            # 根据 req_grad 的值创建一个具有梯度要求的张量 c_arg
            c_arg = torch.ones(2, 2, requires_grad=req_grad)
            # 对基础张量 base 进行加法操作并赋给 x
            x = base.add(1)
            # 返回两个列表，第一个列表包含 base 和 c_arg，第二个列表包含 x 的视图、x 的视图和 c_arg
            return [base, c_arg], [x.view(-1), x.view(-1), c_arg]

        # 调用 self.verify_aot_autograd 进行测试，验证 f 函数的行为
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )

        # 使用断言检查是否会引发 RuntimeError 异常，异常信息包含特定文本
        with self.assertRaisesRegex(
            RuntimeError, "is a tensor subclass. This is not supported today"
        ):
            # 调用 self.verify_aot_autograd，测试 f 函数的行为，并将输入设置为子类
            self.verify_aot_autograd(
                f,
                partial(inp_callable, req_grad=False),
                test_mutation=True,
                make_inputs_subclasses=True,
            )

        # 验证 f 函数在要求梯度的情况下的行为，并获取前向图
        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        # 使用 self.assertExpectedInline 断言前向图的代码内容是否符合预期
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    # 使用 primals_1 创建克隆对象 clone
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    # 使用 clone 创建视图对象 as_strided
    as_strided = torch.ops.aten.as_strided.default(clone, [4], [1], 0)
    # 在 as_strided 上执行加法操作，结果存入 add；释放 as_strided
    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None
    # 使用 clone 和 add 执行 as_strided_scatter 操作，结果存入 as_strided_scatter；释放 clone 和 add
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, add, [4], [1], 0);  clone = add = None
    # 使用 torch.ops.aten.add.Tensor 执行张量相加操作，并将结果赋给 add_1；清空 primals_2 和 primals_3 引用
    add_1 = torch.ops.aten.add.Tensor(primals_2, primals_3);  primals_2 = primals_3 = None
    # 使用 torch.ops.aten.as_strided.default 对 as_strided_scatter 张量执行 stride 操作，得到 as_strided_5，步长为 [4]，填充为 [1]，偏移为 0
    as_strided_5 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    # 使用 torch.ops.aten.unsqueeze.default 在 as_strided_5 张量上执行 unsqueeze 操作，在第 0 维度插入新维度；清空 as_strided_5 引用
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(as_strided_5, 0);  as_strided_5 = None
    # 使用 torch.ops.aten.add.Tensor 执行张量相加操作，将 add_1 和 unsqueeze_1 相加，结果赋给 add_2；清空 add_1 引用
    add_2 = torch.ops.aten.add.Tensor(add_1, unsqueeze_1);  add_1 = None
    # 使用 torch.ops.aten.as_strided.default 对 as_strided_scatter 张量执行 stride 操作，得到 as_strided_14，步长为 [4]，填充为 [1]，偏移为 0
    as_strided_14 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    # 使用 torch.ops.aten.view.default 对 as_strided_14 张量执行视图变换操作，将其形状变为 [-1]；清空 as_strided_14 引用
    view_2 = torch.ops.aten.view.default(as_strided_14, [-1]);  as_strided_14 = None
    # 返回包含 as_strided_scatter, add_2, view_2, unsqueeze_1 的列表
    return [as_strided_scatter, add_2, view_2, unsqueeze_1]
        # 定义一个测试函数，测试输入是否被改变并且有别名现象存在
        # 输入被改变，其中有一个别名指向另一个输入（所以我们创建一个合成的基础）
        # 一个输出是另一个输出的别名
        # 一个输出是一个中间变量的别名
        # a 和 c 是别名
        def f(a, b, c):
            # 修改 c 的值（就地修改）
            c.mul_(2)
            # 修改 b 的元数据
            b.t_()
            # 计算临时变量 tmp
            tmp = a + c
            # 创建 out1，它是 tmp 的视图
            out1 = tmp.view(-1)
            # 创建 out2，它是 b 的转置，但不返回它
            out2 = b.t()
            # 创建 out3，它是 out1 的维度扩展
            out3 = out1.unsqueeze(0)
            # out1 和 out3 是中间变量 tmp 的别名，它们彼此也是别名！
            # out2 是输入 b 的别名，因此我们不返回它
            return out1, out2, out3

        # 定义一个输入生成器函数，根据需要设置梯度
        def inp_callable(req_grad):
            # 创建两个需要梯度的张量作为基础输入
            base1 = torch.ones(2, 2, requires_grad=req_grad)
            base2 = torch.ones(2, 2, requires_grad=req_grad)
            # 在测试中，add() 操作很重要，因为我们需要图的输入不是叶子节点，这样我们才能对它们进行修改
            base1_ = base1.add(1)
            base2_ = base2.add(1)
            # 创建 a，它是 base1_ 的视图
            a = base1_.view(-1)
            # 创建 b，它是 base2_ 的引用
            b = base2_
            # 创建 c，它是 base1_ 的视图，与 a 是别名
            c = base1_.view(-1)
            return [base1, base2], [a, b, c]

        # 验证函数调用的自动微分行为
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        # 验证带梯度的函数调用的自动微分行为
        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        # 预期：
        # - 前向传播中有 2 个输入：synthetic_base_a_c, b
        # - 前向传播中有 1 个输出："tmp"
        #   out2 是一个输入的别名，并且将在编译后的函数外部基于 b 生成
        #   out1 和 out3 是 tmp 的别名，我们将在编译后的函数外部生成它们
        # 断言期望的内联代码
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """
def forward(self, primals_1, primals_2):
    # 使用 torch.ops.aten.clone.default 复制 primals_1，并将 primals_1 置为 None
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    # 使用 torch.ops.aten.view.default 将 primals_2 转换为 2x2 的视图，并将 primals_2 置为 None
    view = torch.ops.aten.view.default(primals_2, [2, 2]);  primals_2 = None
    # 使用 torch.ops.aten.as_strided.default 对 clone 进行大小为 [4]、步长为 [1] 的操作
    as_strided_1 = torch.ops.aten.as_strided.default(clone, [4], [1], 0)
    # 使用 torch.ops.aten.mul.Tensor 对 as_strided_1 中的数据乘以 2，并将 as_strided_1 置为 None
    mul = torch.ops.aten.mul.Tensor(as_strided_1, 2);  as_strided_1 = None
    # 使用 torch.ops.aten.as_strided_scatter.default 对 clone 和 mul 进行大小为 [4]、步长为 [1] 的操作，并将 clone 和 mul 置为 None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, mul, [4], [1], 0);  clone = mul = None
    # 使用 torch.ops.aten.as_strided.default 对 as_strided_scatter 进行大小为 [4]、步长为 [1] 的操作
    as_strided_2 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    # 使用 torch.ops.aten.t.default 对 view 进行转置操作，并将 view 置为 None
    t = torch.ops.aten.t.default(view);  view = None
    # 使用 torch.ops.aten.as_strided.default 对 as_strided_scatter 进行大小为 [4]、步长为 [1] 的操作，并将结果赋给 as_strided_5
    as_strided_5 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    # 使用 torch.ops.aten.add.Tensor 对 as_strided_5 和 as_strided_2 进行元素级加法，并将 as_strided_5 和 as_strided_2 置为 None
    add = torch.ops.aten.add.Tensor(as_strided_5, as_strided_2);  as_strided_5 = as_strided_2 = None
    # 使用 torch.ops.aten.view.default 将 add 变换为一维数组
    view_1 = torch.ops.aten.view.default(add, [-1])
    # 使用 torch.ops.aten.t.default 对 t 进行转置操作
    t_1 = torch.ops.aten.t.default(t)
    # 使用 torch.ops.aten.unsqueeze.default 在 view_1 的第一维度添加一个维度
    unsqueeze = torch.ops.aten.unsqueeze.default(view_1, 0)
    # 返回计算结果的列表
    return [as_strided_scatter, t, view_1, t_1, unsqueeze, add]
    # 测试函数，验证不需要梯度的输出不影响视图
    def test_some_outputs_dont_require_grad_non_view(self):
        # 定义函数 f，接收两个参数 a 和 b，对 a 添加 1 后分离梯度，返回结果和 b
        def f(a, b):
            return a.add(1).detach(), b

        # 输入张量列表，每个张量大小为 3x3，需要计算梯度
        inp = [
            torch.randn(3, 3, requires_grad=True),
            torch.randn(3, 3, requires_grad=True),
        ]
        # 调用 verify_aot_autograd 函数验证自动求导功能
        self.verify_aot_autograd(f, inp)

    # 测试函数，验证内部梯度计算
    def test_inner_grad(self):
        # 定义函数 foo，接收参数 x，计算 torch.exp(x) 的梯度并返回
        def foo(x):
            y = torch.exp(x)
            z = torch.autograd.grad(y, x)
            return z

        # 输入张量列表，包含一个标量张量，需要计算梯度
        inps = [torch.randn((), requires_grad=True)]
        # 调用 verify_aot_autograd 函数验证自动求导功能
        self.verify_aot_autograd(foo, inps)

    # 测试函数，验证梯度计算上下文
    def test_grad_context(self):
        # 定义函数 foo，接收参数 x，返回 x 的两倍
        def foo(x):
            return x * 2

        # 输入张量列表，包含一个标量张量，需要计算梯度
        inps = [torch.randn((), requires_grad=True)]
        graph_size = None

        # 定义函数 get_graph_size，获取计算图的节点数
        def get_graph_size(fx_g, _):
            nonlocal graph_size
            graph_size = len(fx_g.graph.nodes)
            return fx_g

        # 使用 aot_function 编译函数 foo，禁用梯度计算上下文
        f = aot_function(foo, nop, get_graph_size)
        with torch.set_grad_enabled(False):
            f(*inps)
        # 断言 graph_size 应为 None
        self.assertIsNone(graph_size)

        # 重新编译函数 foo，启用梯度计算上下文
        f = aot_function(foo, nop, get_graph_size)
        with torch.set_grad_enabled(True):
            out = f(*inps)
            # 断言 graph_size 应大于 2
            self.assertIsNone(graph_size)
            out.sum().backward()
            self.assertTrue(graph_size > 2)

    # 测试函数，验证输出为字典类型
    def test_output_dict(self):
        # 定义函数 f，接收参数 x，返回包含两个键值对 'a': x 和 'b': x 的字典
        def f(x):
            return {"a": x, "b": x}

        # 输入张量列表，包含一个大小为 3x3 的张量，需要计算梯度
        inp = [torch.randn(3, 3, requires_grad=True)]
        # 调用 verify_aot_autograd 函数验证自动求导功能
        self.verify_aot_autograd(f, inp)

        # 定义函数 f，接收两个参数 x 和 y，返回包含两个键值对 'a': x 和 'b': y + x 的字典
        def f(x, y):
            return {"a": x, "b": y + x}

        # 输入张量列表，包含两个大小为 3 的张量，其中第一个需要计算梯度
        inp = [torch.randn(3, requires_grad=True), torch.randn(3)]
        # 调用 verify_aot_autograd 函数验证自动求导功能
        self.verify_aot_autograd(f, inp)

        # 定义函数 f，接收参数 x，返回每个键值对值乘以 2 后的新字典
        def f(x):
            new_d = {}
            for k in x:
                new_d[k] = x[k] * 2
            return new_d

        a = torch.randn(3, requires_grad=True)
        b = torch.randn(3, requires_grad=True)

        # 定义函数 inp_callable，返回输入列表和输出列表，都包含一个字典，字典包含键 'a' 和 'b' 的张量
        def inp_callable():
            inps = [{"a": a, "b": b}]
            return inps, inps

        # 调用 verify_aot_autograd 函数验证自动求导功能
        self.verify_aot_autograd(f, inp_callable)

    # 测试函数，验证编译模块的梯度计算
    def test_module(self):
        # 创建一个包含线性层和 ReLU 激活函数的序列模块
        mod = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        # 编译模块 mod，使用 nop 函数作为编译和运行时的空操作
        compiled_mod = compiled_module(mod, nop, nop)
        inp = torch.randn(32, 32)
        # 计算模块 mod 的输出和梯度，并反向传播求梯度
        ref_out = mod(inp)
        ref_out.sum().backward()
        # 获取模块 mod 的参数名称和梯度，并按名称排序
        ref_grads = sorted([(name, p.grad) for name, p in mod.named_parameters()])
        # 使用编译后的模块计算输出和梯度，并反向传播求梯度
        out = compiled_mod(inp)
        out.sum().backward()
        # 获取编译后模块的参数名称和梯度，并按名称排序
        grads = sorted([(name, p.grad) for name, p in mod.named_parameters()])
        # 断言编译前后的输出和梯度相等
        self.assertEqual((out, grads), (ref_out, ref_grads))

    # 测试函数，验证批标准化模块的梯度计算
    def test_batchnorm(self):
        # 编译 nn.BatchNorm2d(4) 模块，使用 nop 函数作为编译和运行时的空操作
        mod = compiled_module(nn.BatchNorm2d(4), nop, nop)
        x = torch.ones(1, 4, 2, 2)
        # 计算模块 mod 的输出，并反向传播求梯度
        mod(x).sum().backward()
    # 定义一个测试方法，用于生成列表的代码生成测试
    def test_list_codegen(self):
        # 定义一个函数，将给定的函数包装成另一个函数
        def list_nop(f, _):
            # 定义内部函数 g，它接受一个输入列表并调用原始函数 f
            def g(inps):
                return f(*inps)
            
            # 设置 g._boxed_call 属性为 True
            g._boxed_call = True
            return g
        
        # 定义一个函数 f，接受三个参数并返回它们的三角函数运算结果的乘积
        def f(a, b, c):
            return a.sin() * b.cos() * c.sin()
        
        # 使用 aot_function 函数将函数 f 编译为另一个函数
        f = aot_function(f, list_nop)
        
        # 生成包含三个随机张量的列表
        inp = [torch.randn(5, requires_grad=True) for _ in range(3)]
        
        # 调用 f，并对其结果求和并反向传播
        f(*inp).sum().backward()

    # 使用装饰器 patch 修改 AOT_COUNTER，定义编译上下文的测试方法
    @patch("torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count)
    def test_compilation_context(self, counter):
        # 定义一个函数 f，对输入张量进行双重 sin 函数计算
        def f(x):
            return x.sin().sin()
        
        # 定义一个空列表 count，用于存储编译上下文和计算图节点数
        count = []
        
        # 定义一个编译器函数 compiler，接受 fx_g 函数和其他参数
        def compiler(fx_g, _):
            # 获取 AOT 编译上下文并创建上下文对象
            context = get_aot_compilation_context()
            # 将上下文标识和计算图节点数添加到 count 列表中
            count.append((context[0], len(fx_g.graph.nodes)))
            return fx_g
        
        # 使用 aot_function 将函数 f 编译为另一个函数
        f = aot_function(f, compiler)
        
        # 对随机张量进行 f 的调用，并记录编译上下文和计算图节点数
        out = f(torch.randn(5, requires_grad=True))
        
        # 再次使用 aot_function 编译函数 f
        f = aot_function(f, compiler)
        
        # 对另一个随机张量进行 f 的调用
        f(torch.randn(5))
        
        # 对结果进行求和并反向传播
        out.sum().backward()
        
        # 断言 count 列表的输出结果符合预期
        self.assertExpectedInline(
            str(count),
            """[(['0_forward'], 4), (['1_inference'], 4), (['0_backward'], 8)]""",
        )

    # 定义一个测试方法，验证重复参数的函数功能
    def test_dupe_arg(self):
        # 定义一个函数 f，接受两个参数并返回它们的加法结果
        def f(x, y):
            return x + y
        
        # 生成一个随机张量 x，并对其进行 AOT 自动求导验证
        x = torch.randn(3, 3, requires_grad=True)
        self.verify_aot_autograd(f, [x, x])

    # 定义一个测试方法，验证对重复参数进行操作的函数功能
    def test_dupe_arg_torture(self):
        # 定义一个函数 f，接受两个参数 x 和 y，对 x 进行原地转置和 y 进行维度扩展，然后返回它们的加法结果
        def f(x, y):
            x.t_()
            y.unsqueeze_(0)
            return x + y
        
        # 克隆一个随机张量 x，并对其进行 AOT 自动求导验证
        x = torch.randn(3, 3, requires_grad=True).clone()
        self.verify_aot_autograd(f, [x, x])

    # 定义一个测试方法，验证将重复参数作为输出返回的函数功能
    # 查看 https://github.com/pytorch/pytorch/issues/100224
    def test_dupe_arg_returned_as_output(self):
        # 定义一个函数 f，接受三个参数 a、b 和 a_，对 a 的第一个元素加 1，并返回参数 a_
        def f(a, b, a_):
            a[0].add_(1)
            return a_
        
        # 使用 nop 函数将函数 f 编译为另一个函数 f_compiled
        f_compiled = aot_function(f, nop)
        
        # 创建两个元素为全 1 的张量 a 和 b
        a = torch.ones(2)
        b = torch.ones(2)
        
        # 对函数 f 的调用，记录其输出作为参考值 out_ref
        out_ref = f(a, b, a)
        
        # 创建另外两个元素为全 1 的张量 a2 和 b2，并对函数 f_compiled 进行调用
        out_test = f_compiled(a2, b2, a2)
        
        # 断言函数 f 和 f_compiled 输出的结果相同
        self.assertEqual(out_ref, out_test)
        
        # 断言张量 a 和 a2 是相同的对象引用
        self.assertEqual(a, a2)

    # 使用装饰器 patch 修改 AOT_COUNTER 和 debug_assert，定义编译上下文的测试方法
    @patch("torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count)
    @patch("torch._functorch.config.debug_assert", True)
    # 测试无效的重复左偏向性，验证当只有第一个参数进行了元数据变异时，仍能正确切换到策略2（去重）
    # 参见：https://github.com/pytorch/pytorch/pull/89896#discussion_r1036224447
    def test_invalid_dupe_left_bias(self, counter):
        class F(torch.nn.Module):
            def forward(self, x, y):
                # 将 x 进行转置操作
                x.t_()
                return (x + y,)

        # 创建两个随机张量 x 和 y，其中 x 需要梯度信息
        x = torch.randn(3, 3, requires_grad=True).clone()
        y = torch.randn(3, 3, requires_grad=True)

        # 验证 Ahead-of-time (AOT) Autograd 对模型 F 的编译结果
        self.verify_aot_autograd(F(), [x, x])

        # 对简化后的 AOT 模块 F 进行验证
        fxx = aot_module_simplified(F(), (x, x), nop)

        # 断言运行时调用 fxx(x, y) 时抛出 AssertionError 异常
        self.assertExpectedRaisesInline(
            AssertionError,
            lambda: fxx(x, y),
            """At compilation time, graph 2 was compiled under the assumption that input 1 would be a duplicate of input 0, but at runtime this was not the case.  This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch.""",  # noqa: B950
        )

    # 使用新的计数器对象替换 AOT_COUNTER，并启用调试断言
    @patch("torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count)
    @patch("torch._functorch.config.debug_assert", True)
    def test_invalid_dupe(self, counter):
        # 调用 _test_invalid_dupe 方法进行无效重复测试，传入 fake=False
        self._test_invalid_dupe(counter, fake=False)

    # 为什么存在这个测试的注释，请参见：Dynamo 重新编译保护无效梯度
    @patch("torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count)
    @patch("torch._functorch.config.debug_assert", True)
    def test_invalid_dupe_fake(self, counter):
        # 调用 _test_invalid_dupe 方法进行无效重复测试，传入 fake=True
        self._test_invalid_dupe(counter, fake=True)
    # 定义测试方法 `_test_invalid_dupe`，用于测试无效的重复项情况
    def _test_invalid_dupe(self, counter, fake):
        # 定义内部类 F，继承自 torch.nn.Module
        class F(torch.nn.Module):
            # 定义前向传播方法
            def forward(self, x, y):
                # 在维度0上给张量 x 和 y 添加一个维度
                x.unsqueeze_(0)
                y.unsqueeze_(0)
                # 返回 x + y 的元组
                return (x + y,)

        # 创建形状为 (3, 3) 的随机张量 x 和 y，要求梯度计算，然后克隆它们
        x = torch.randn(3, 3, requires_grad=True).clone()
        y = torch.randn(3, 3, requires_grad=True).clone()

        # 如果 fake 参数为真，则创建 ShapeEnv 对象和 FakeTensorMode 对象
        if fake:
            shape_env = ShapeEnv()
            fake_mode = FakeTensorMode(shape_env=shape_env)

            # 将真实张量 x 和 y 转换为 fake_mode 对象
            fake_x = fake_mode.from_tensor(x)
            fake_y = fake_mode.from_tensor(y)

        # 如果 fake 参数为真，则使用 fake_x 和 fake_y 调用 aot_module_simplified 函数，否则使用 x 和 y
        if fake:
            fxy = aot_module_simplified(F(), (fake_x, fake_y), nop)
        else:
            fxy = aot_module_simplified(F(), (x, y), nop)

        # 调用 fxy 函数并传入 x 和 y
        fxy(x, y)

        # 再次克隆 x 和 y
        x = torch.randn(3, 3, requires_grad=True).clone()
        y = torch.randn(3, 3, requires_grad=True).clone()

        # 使用 x 调用 fxy 函数两次
        fxy(x, x)  # is ok!

        # 如果 fake 参数为真，则使用 fake_x 两次调用 aot_module_simplified 函数，否则使用 x
        if fake:
            fxx = aot_module_simplified(F(), (fake_x, fake_x), nop)
        else:
            fxx = aot_module_simplified(F(), (x, x), nop)

        # 再次克隆 x 和 y
        x = torch.randn(3, 3, requires_grad=True).clone()
        y = torch.randn(3, 3, requires_grad=True).clone()

        # 使用 x 两次调用 fxx 函数
        fxx(x, x)

        # 注意：这里不应该引发异常！一旦我们在这里设置了保护措施，
        # 我们将会正确工作，因为它应该重新编译。
        # 再次克隆 x 和 y
        x = torch.randn(3, 3, requires_grad=True).clone()
        y = torch.randn(3, 3, requires_grad=True).clone()

        # 使用 self.assertExpectedRaisesInline 断言，检查 fxx(x, y) 是否引发 AssertionError 异常
        self.assertExpectedRaisesInline(
            AssertionError,
            lambda: fxx(x, y),
            """At compilation time, graph 1 was compiled under the assumption that input 1 would be a duplicate of input 0, but at runtime this was not the case.  This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch.""",  # noqa: B950
        )

    # 使用 patch 装饰器为 AOT_COUNTER 设置新的调用计数器，并启用 debug_assert
    @patch("torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count)
    @patch("torch._functorch.config.debug_assert", True)
    # 定义测试方法 test_invalid_requires_grad，测试无效的 requires_grad 情况，fake 参数为假
    def test_invalid_requires_grad(self, counter):
        self._test_invalid_requires_grad(counter, fake=False)

    # See Note: Dynamo recompilation guarding invalid grad for why this test exists
    # 使用 patch 装饰器为 AOT_COUNTER 设置新的调用计数器，并启用 debug_assert
    # 定义测试方法 test_invalid_requires_grad_fake，测试无效的 requires_grad 情况，fake 参数为真
    def test_invalid_requires_grad_fake(self, counter):
        self._test_invalid_requires_grad(counter, fake=True)
    # 测试无效 requires_grad 设置的情况
    def _test_invalid_requires_grad(self, counter, fake):
        # 定义一个简单的 PyTorch 模块 F，接受两个输入并返回它们的和
        class F(torch.nn.Module):
            def forward(self, x, y):
                return (x + y,)

        # 创建三个不同的张量 x, y, z，它们都具有梯度追踪功能
        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        # z 是一个张量，但它没有梯度追踪功能
        z = torch.randn(3, 3, requires_grad=False)

        # 如果 fake 为 True，则创建一个 ShapeEnv 对象和 FakeTensorMode 对象
        if fake:
            shape_env = ShapeEnv()
            fake_mode = FakeTensorMode(shape_env=shape_env)

            # 使用 fake_mode 将 x, y, z 转换成伪装的张量
            fake_x = fake_mode.from_tensor(x)
            fake_y = fake_mode.from_tensor(y)
            fake_z = fake_mode.from_tensor(z)

        # 如果 fake 为 True，则使用伪装的张量 fake_x 和 fake_y 调用 aot_module_simplified 函数
        # 否则使用真实的张量 x 和 y 调用该函数
        if fake:
            fxy = aot_module_simplified(F(), (fake_x, fake_y), nop)
        else:
            fxy = aot_module_simplified(F(), (x, y), nop)

        # 检验函数输出和梯度的一致性，针对输入 (x, y)
        compare_equal_outs_and_grads(self, F(), fxy, (x, y))
        # 再次检验函数输出和梯度的一致性，但这次使用输入 (x, z)
        compare_equal_outs_and_grads(self, F(), fxy, (x, z))

        # 如果 fake 为 True，则使用伪装的张量 fake_x 和 fake_z 调用 aot_module_simplified 函数
        # 否则使用真实的张量 x 和 z 调用该函数
        if fake:
            fxz = aot_module_simplified(F(), (fake_x, fake_z), nop)
        else:
            fxz = aot_module_simplified(F(), (x, z), nop)

        # 检验函数输出和梯度的一致性，针对输入 (x, z)
        compare_equal_outs_and_grads(self, F(), fxz, (x, z))

        # 使用 lambda 函数和特定错误信息来断言 fxz(x, y) 的行为
        self.assertExpectedRaisesInline(
            AssertionError,
            lambda: fxz(x, y),
            """At compilation time, graph 1 was compiled under the assumption that input 1 would not require grad, but at runtime this was not the case.  This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch.""",  # noqa: B950
        )

    # 测试自定义的自动求导函数 CustomFn
    def test_custom_autograd(self):
        # 定义一个继承自 torch.autograd.Function 的自定义函数 CustomFn
        class CustomFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 在 forward 方法中简单地返回输入张量的克隆
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                # 在 backward 方法中返回梯度输出加 1
                return grad_output + 1

        # 定义一个简单的函数 f，应用 CustomFn 来处理输入张量 x
        def f(x):
            return CustomFn.apply(x)

        # 验证 f 函数在输入 [torch.randn(3)] 时的自动求导行为
        self.verify_aot_autograd(f, [torch.randn(3)])

    # 在 CUDA 可用时测试自动类型转换的禁用情况
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_autocast_disable_guard(self):
        # 使用 torch._C._DisableAutocast 上下文管理器来禁用自动类型转换
        with torch._C._DisableAutocast():
            # 创建一个在 CUDA 上运行的随机张量 x
            x = torch.rand([4, 4]).cuda()
            # 计算 x 和 x 的矩阵乘积
            y = x @ x
            # 断言 y 的数据类型为 torch.float32
            self.assertEqual(y.dtype, torch.float32)

    # 在 CUDA 可用时测试非幂等的自动混合精度计算
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_nonidempotent_amp(self):
        # 定义一个函数 f，使用 torch.functional.einsum 进行张量运算
        def f(self_s_emb, add_3):
            einsum_2 = torch.functional.einsum("ah,th->t", self_s_emb, add_3)
            log_softmax_2 = einsum_2.log_softmax(-1)
            return (log_softmax_2,)

        # 定义输入参数 args，包含两个在 CUDA 上的张量，分别是 float32 和 float16 类型
        args = [
            torch.rand((1, 256), dtype=torch.float32, device="cuda"),
            torch.rand((30, 256), dtype=torch.float16, device="cuda"),
        ]

        # 在启用自动混合精度计算的情况下，验证函数 f 的自动求导行为
        with torch.cuda.amp.autocast(enabled=True):
            self.verify_aot_autograd(f, args)

        # 将 args 中的所有张量设置为需要梯度追踪
        args = [e.requires_grad_(True) for e in args]

        # 在启用自动混合精度计算的情况下，再次验证函数 f 的自动求导行为
        with torch.cuda.amp.autocast(enabled=True):
            self.verify_aot_autograd(f, args)
    @unittest.skipIf(not torch.backends.cudnn.is_available(), "CUDNN is unavailable")
    @skipIfRocm  # 如果运行环境是ROCm，跳过测试，参考：https://github.com/pytorch/pytorch/issues/96560
    def test_batch_norm_amp(self):
        device = "cuda"
        input_dtype = torch.float16
        param_dtype = torch.float32
        weight, bias = (
            torch.ones(64, device=device, dtype=param_dtype, requires_grad=True)
            for _ in range(2)
        )
        running_mean, running_var = (
            torch.ones(64, device=device, dtype=param_dtype) for _ in range(2)
        )

        def bn(x):
            return torch.ops.aten.cudnn_batch_norm(
                x,
                weight,
                bias,
                running_mean,
                running_var,
                False,
                0.1,
                1e-05,
            )

        inp = torch.ones(
            torch.Size([16, 64, 112, 112]), dtype=input_dtype, device=device
        )

        ref = bn(inp)  # 执行批量归一化操作，参考：torch.ops.aten.cudnn_batch_norm
        cudnn_batch_norm_decomp = torch._decomp.get_decompositions(
            {torch.ops.aten.cudnn_batch_norm}
        )
        aot_fn = make_fx(bn, decomposition_table=cudnn_batch_norm_decomp)(inp)  # 使用静态图优化 (AOT) 对 bn 函数进行优化
        res = aot_fn(inp)  # 在优化后的函数上执行输入
        for a, b in zip(ref, res):
            assert torch.allclose(a, b)  # 检查优化前后结果是否一致

    def test_output_op_depending_on_symint(self):
        """
        It won't be obvious from reading this test what it's testing for.  We should probably make it into a more
        focused unit test.

        An issue with the following program was the expand op would end up depending on a symint whose proxy was
        incorrectly associated with one of the grad tensors rather than input tensors.  It broke partitioner logic
        and the net result was aot_function failed to produce a function and threw an exception instead.
        """
        inp = torch.randn(5, requires_grad=True)

        def f(x):
            return x.expand(x.shape)  # 扩展输入张量的维度

        # TODO(whc) make this work (test setup is wrong somehow)
        # joint_forward_backward = create_joint_forward_backward(f)
        # out = f(inp)
        # joint_inputs =  ([inp], [out.detach().contiguous()])
        # fx_g = make_fx(joint_forward_backward)(*joint_inputs)
        # TODO: assert outputs of fwd graph trace to correct symint

        # e2e test that fails without symint clone fix
        af = aot_function(
            f,
            nop,
            partition_fn=partial(
                min_cut_rematerialization_partition, compiler="inductor"
            ),
            dynamic=True,
        )
        out = af(inp)  # 在优化 (AOT) 函数上执行输入
        self.assertEqual(out, f(inp))  # 检查优化前后结果是否一致

    def test_inference_mode(self):
        m = torch.nn.Linear(4, 4)
        inp = torch.randn(4, 4)

        aot_mod = aot_module(m, fw_compiler=nop)  # 使用静态图优化 (AOT) 对模块进行优化

        with torch.inference_mode():
            out_ref = m(inp)  # 在原始模块上执行推断
            out_test = aot_mod(inp)  # 在优化后的模块上执行推断
        self.assertEqual(out_ref, out_test)  # 检查优化前后推断结果是否一致
        def test_default_partitioner_saves_symints_not_tensors_for_bw(self):
            """
            In this test, the important thing is that primals_1 is **only** needed in the backward
            in order to grab its sizes.
            We need to assert that what we save for the backward are the tensor's sizes, and not the tensor itself.
        
            The way this test is set up, it will actually fail if we try to save the input tensor for backward.
            Why?
            b.masked_fill_(c, 0) has a backward that requires knowing a's sizes
            b.masked_fill_(c, 0) **also** mutates a (because b and a are aliased)
            The autograd engine yells at us if we save "a" for backward, and then try to mutate it.
            """
            inp = torch.randn(2, 2, requires_grad=True)
        
            # 定义函数 f，接受一个张量 a 作为输入
            def f(a):
                # 从张量 a 中获取第一个子张量 b
                b = a[0]
                # 创建一个与 b 相同形状的全 1 布尔张量 c
                c = torch.ones_like(b, dtype=torch.bool)
                # 对 b 进行 masked_fill 操作，将 c 对应位置置为 0，并返回结果 d
                d = b.masked_fill_(c, 0)
                return d
        
            # 编译函数 f 以提前执行
            compiled_f = aot_function(f, nop, dynamic=True)
            # 创建一个形状为 (2, 2) 的全随机张量，并声明需要梯度
            inp_ref = torch.ones(2, 2, requires_grad=True)
            # 克隆 inp_ref 以进行后续测试
            inp_test = torch.ones(2, 2, requires_grad=True)
        
            # 使用普通方式计算输出结果
            out_ref = f(inp_ref.clone())
            # 使用提前编译的函数计算输出结果
            out_test = compiled_f(inp_test.clone())
        
            # 断言两种计算方式的输出结果相等
            self.assertEqual(out_ref, out_test)
        
            # 计算普通方式的梯度并反向传播
            out_ref.sum().backward()
            # 计算提前编译方式的梯度并反向传播
            out_test.sum().backward()
        
            # 断言两种方式的输入梯度相等
            self.assertEqual(inp_ref.grad, inp_test.grad)
        
        def test_buffer_copied_in_graph(self):
            # 定义一个继承自 torch.nn.Module 的模型类 MyModel
            class MyModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 注册一个缓冲区 buf，初始化为零张量
                    self.register_buffer("buf", torch.zeros(1))
                    # 定义两个参数 w1 和 w2，并初始化为零张量
                    self.w1 = torch.nn.Parameter(torch.zeros(1))
                    self.w2 = torch.nn.Parameter(torch.zeros(1))
        
                # 定义前向传播方法，接受输入 x
                def forward(self, x):
                    # 缓冲区 buf 加 1
                    self.buf.add_(1)
                    # 返回计算结果，包括参数 w1、w2 和 buf 的和
                    return (self.w1 * x * self.w2).sum() + self.buf.sum()
        
            # 创建一个 MyModel 类的实例 model_for_eager
            model_for_eager = MyModel()
            # 深度复制 model_for_eager，得到 model_for_compile
            model_for_compile = copy.deepcopy(model_for_eager)
        
            # 定义一个空列表 fw_graph_cell 用于存储前向计算图
            fw_graph_cell = [None]
            # 使用 aot_module 编译模型 model_for_compile
            compiled_f = aot_module(
                model_for_compile,
                fw_compiler=make_boxed_compiler(
                    partial(extract_graph, graph_cell=fw_graph_cell)
                ),
                bw_compiler=nop,
                keep_inference_input_mutations=True,
            )
        
            # 创建一个形状为 (1,) 的张量，声明需要梯度
            inp_ref = torch.ones(1, requires_grad=True)
            # 克隆 inp_ref 以进行后续测试
            inp_test = torch.ones(1, requires_grad=True)
        
            # 使用普通方式计算模型输出结果
            out_ref = model_for_eager(inp_ref.clone())
            # 使用提前编译方式计算模型输出结果
            out_test = compiled_f(inp_test.clone())
        
            # 断言两种计算方式的输出结果相等
            self.assertExpectedInline(
                fw_graph_cell[0].code.strip(),
                """\
def forward(self, primals_1, primals_2, primals_3, primals_4):
    add = torch.ops.aten.add.Tensor(primals_3, 1)
    mul = torch.ops.aten.mul.Tensor(primals_1, primals_4)
    mul_1 = torch.ops.aten.mul.Tensor(mul, primals_2)
    sum_1 = torch.ops.aten.sum.default(mul_1);  mul_1 = None
    sum_2 = torch.ops.aten.sum.default(add)
    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    copy_ = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = None
    return [add_1, primals_1, primals_2, primals_4, mul]


注释：


# 在模型的前向传播方法中进行操作
def forward(self, primals_1, primals_2, primals_3, primals_4):
    # 执行张量加法操作，将 primals_3 和标量 1 相加
    add = torch.ops.aten.add.Tensor(primals_3, 1)
    # 执行张量乘法操作，将 primals_1 和 primals_4 相乘
    mul = torch.ops.aten.mul.Tensor(primals_1, primals_4)
    # 连续执行两次张量乘法操作，将 mul 和 primals_2 相乘
    mul_1 = torch.ops.aten.mul.Tensor(mul, primals_2)
    # 执行第一个求和操作，对 mul_1 中的元素进行求和
    sum_1 = torch.ops.aten.sum.default(mul_1);  mul_1 = None
    # 执行第二个求和操作，对 add 中的元素进行求和
    sum_2 = torch.ops.aten.sum.default(add)
    # 执行张量加法操作，将 sum_1 和 sum_2 相加
    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    # 执行拷贝操作，将 add 的值复制到 primals_3 中
    copy_ = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = None
    # 返回计算结果的列表，包括 add_1、primals_1、primals_2、primals_4 和 mul
    return [add_1, primals_1, primals_2, primals_4, mul]
        def test_buffer_batch_norm(self):
            # 定义一个简单的 PyTorch 模型类 MyModel，包含一个 BatchNorm1d 层
            class MyModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.m = torch.nn.BatchNorm1d(100)

                def forward(self, x):
                    return self.m(x)

            # 创建一个实例化的 MyModel 对象，用于即时编译
            model_for_eager = MyModel()
            # 使用深拷贝复制 model_for_eager 以备用于编译后的模型
            model_for_compile = copy.deepcopy(model_for_eager)

            # 创建存储前向图和后向图的单元列表
            fw_graph_cell = [None]
            bw_graph_cell = [None]

            # 对 model_for_compile 进行 Ahead-of-Time (AOT) 编译
            compiled_f = aot_module(
                model_for_compile,
                # 使用 boxed_compiler 将前向图提取到 fw_graph_cell 中
                fw_compiler=make_boxed_compiler(
                    partial(extract_graph, graph_cell=fw_graph_cell)
                ),
                # 使用 boxed_compiler 将后向图提取到 bw_graph_cell 中
                bw_compiler=make_boxed_compiler(
                    partial(extract_graph, graph_cell=bw_graph_cell)
                ),
                # 保留推理输入的变异状态以供使用
                keep_inference_input_mutations=True,
            )

            # 创建输入张量，20 行 100 列，需要计算梯度
            inp_ref = torch.ones(20, 100, requires_grad=True)
            inp_test = torch.ones(20, 100, requires_grad=True)

            # 使用即时模型计算输出结果
            out_ref = model_for_eager(inp_ref.clone())
            # 使用编译后的模型计算输出结果
            out_test = compiled_f(inp_test.clone())

            # 断言前向图代码的期望输出
            self.assertExpectedInline(
                fw_graph_cell[0].code.strip(),
                """\
def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6):
    # 调用 PyTorch 的张量加法操作，将 primals_5 和标量 1 相加
    add = torch.ops.aten.add.Tensor(primals_5, 1)
    # 调用 PyTorch 的 _native_batch_norm_legit_functional 操作执行批归一化计算
    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(primals_6, primals_1, primals_2, primals_3, primals_4, True, 0.1, 1e-05);  primals_2 = None
    # 获取 _native_batch_norm_legit_functional 的第一个元素
    getitem = _native_batch_norm_legit_functional[0]
    # 获取 _native_batch_norm_legit_functional 的第二个元素
    getitem_1 = _native_batch_norm_legit_functional[1]
    # 获取 _native_batch_norm_legit_functional 的第三个元素
    getitem_2 = _native_batch_norm_legit_functional[2]
    # 获取 _native_batch_norm_legit_functional 的第四个元素
    getitem_3 = _native_batch_norm_legit_functional[3]
    # 获取 _native_batch_norm_legit_functional 的第五个元素
    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
    # 使用 PyTorch 的 copy_ 操作，将 getitem_3 的值复制到 primals_3
    copy_ = torch.ops.aten.copy_.default(primals_3, getitem_3);  primals_3 = None
    # 使用 PyTorch 的 copy_ 操作，将 getitem_4 的值复制到 primals_4
    copy__1 = torch.ops.aten.copy_.default(primals_4, getitem_4);  primals_4 = None
    # 使用 PyTorch 的 copy_ 操作，将 add 的值复制到 primals_5，并清空 add 的引用
    copy__2 = torch.ops.aten.copy_.default(primals_5, add);  primals_5 = add = None
    # 返回一组张量和变量
    return [getitem, primals_1, primals_6, getitem_1, getitem_2, getitem_3, getitem_4]""",  # noqa: B950
        def test_new_inp_requires_grad_now(self):
            # 定义函数 f，接受两个参数 x 和 y，将 y 加到 x 上并返回
            def f(x, y):
                return x.add_(y)

            # 初始化正向图和反向图的储存单元为空列表
            fw_graph_cell = [None]
            bw_graph_cell = [None]

            # 编译函数 f 并生成 Ahead-Of-Time (AOT) 函数对象
            compiled_f = aot_function(
                f,
                # 使用 make_boxed_compiler 创建装箱编译器，提取正向图并存储在 fw_graph_cell 中
                fw_compiler=make_boxed_compiler(
                    partial(extract_graph, graph_cell=fw_graph_cell)
                ),
                # 使用 make_boxed_compiler 创建装箱编译器，提取反向图并存储在 bw_graph_cell 中
                bw_compiler=make_boxed_compiler(
                    partial(extract_graph, graph_cell=bw_graph_cell)
                ),
                keep_inference_input_mutations=True,
            )

            # 定义参考输入 inp_ref 和测试输入 inp_test
            inp_ref = (
                torch.ones(20, 100, requires_grad=False),  # 第一个张量，不需要梯度
                torch.ones(20, 100, requires_grad=True),   # 第二个张量，需要梯度
            )
            inp_test = (
                torch.ones(20, 100, requires_grad=False),  # 第一个张量，不需要梯度
                torch.ones(20, 100, requires_grad=True),   # 第二个张量，需要梯度
            )

            # 对参考输入和测试输入分别调用函数 f 和编译后的函数 compiled_f 得到输出
            out_ref = f(*inp_ref)
            out_test = compiled_f(*inp_test)

            # 断言：fw_graph_cell[0].code 的结果应与预期的内联代码一致
            self.assertExpectedInline(
                fw_graph_cell[0].code.strip(),
                """\
    def forward(self, primals_1, primals_2):
        # 使用 torch.ops.aten.clone.default 方法克隆 primals_1 张量，并将 primals_1 设为 None
        clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        # 使用 torch.ops.aten.add.Tensor 方法将 clone 张量和 primals_2 相加，结果保存在 add 变量中，并将 clone 和 primals_2 设为 None
        add = torch.ops.aten.add.Tensor(clone, primals_2);  clone = primals_2 = None
        # 返回包含两个 add 张量的列表
        return [add, add]""",
        )  # noqa: B950

        self.assertEqual(out_ref, out_test)

        # 对 out_ref 和 out_test 求和，并进行反向传播
        out_ref.sum().backward()
        out_test.sum().backward()

        # 断言预期的内联结果
        self.assertExpectedInline(
            bw_graph_cell[0].code.strip(),
            """\
def forward(self, tangents_1):
    return [None, tangents_1]""",
        )  # noqa: B950

    def test_real_weights_in_symbolic_mode(self):
        from functorch.experimental import functionalize

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                x = self.linear(x)
                return x

        m = M().eval()

        inp = torch.randn(2, 5)

        gm = make_fx(m, tracing_mode="symbolic", _allow_non_fake_inputs=True)(inp)
        self.assertEqual(gm(torch.ones(2, 5)), m(torch.ones(2, 5)))

        gm_functionalized = make_fx(
            functionalize(
                gm,
            ),
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )(inp)
        self.assertEqual(gm_functionalized(torch.ones(2, 5)), m(torch.ones(2, 5)))

        inp_count = 0
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                inp_count += 1

        # No more param lifting
        self.assertEqual(inp_count, 1)

        inp_count = 0
        for node in gm_functionalized.graph.nodes:
            if node.op == "placeholder":
                inp_count += 1

        # No more param lifting
        self.assertEqual(inp_count, 1)

        # 使用断言来测试预期的异常情况，确保所有张量都转换为 FakeTensors
        with self.assertRaisesRegex(
            Exception, "Please convert all Tensors to FakeTensors"
        ):
            make_fx(m, tracing_mode="symbolic", _allow_non_fake_inputs=False)(
                torch.randn(2, 5)
            )

    def test_real_weights_in_symbolic_mode_with_inplace_ops(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(4, 5))

            def forward(self, x):
                # 在 buffer 属性上进行原地操作不被允许，在这里会引发异常
                y = self.buffer.add_(3)
                y.resize_([20])
                assert y.shape == self.buffer.shape
                return x.sum() + self.buffer.sum()

        m = M().eval()
        inp = torch.randn(2, 5)
        # 在属性上进行原地变异不被允许，在这里会引发异常
        with self.assertRaisesRegex(Exception, "Can't call metadata"):
            make_fx(m, tracing_mode="symbolic", _allow_non_fake_inputs=True)(inp)
    def _compile_and_erase_bases(self, *output_view_indices):
        # Overrides _base and _view_func tensor attributes, so as to avoid the view-replay
        # execution path when reconstructing views.

        # 定义一个新的 Tensor 类 NoViewReplayTensor，用于覆盖 _base 和 _view_func 属性，
        # 避免在重建视图时执行视图重播路径。
        class NoViewReplayTensor(torch.Tensor):
            @property
            def _base(self):
                return None

            @property
            def _view_func(self):
                return None

        # 包装 FX 图 'g' 中是视图的输出，使用 NoViewReplayTensor 包装，
        # 因为它们是唯一会被重建的部分。
        def wrapper(g, *args, **kwargs):
            outs = g(*args, **kwargs)
            for i in output_view_indices:
                outs[i] = NoViewReplayTensor(outs[i])
            return outs

        # 返回一个函数，该函数接受一个参数 f，并使用 aot_function 函数将其编译，
        # 并指定 fw_compiler 参数为 partial(wrapper, g)，其中 g 是输入的 FX 图。
        return lambda f: aot_function(f, fw_compiler=lambda g, _: partial(wrapper, g))

    def test_output_aliases_input_view_meta_replay(self):
        @self._compile_and_erase_bases(0)
        def f(a):
            return a.view(-1)

        inp = torch.ones(2, 2, requires_grad=True)
        out = f(inp)

        self.assertIsNotNone(out.grad_fn)
        self.assertExpectedInline(
            str(out.grad_fn.__class__), """<class 'ViewBackward0'>"""
        )

    def test_output_aliases_intermediate_view_meta_replay(self):
        @self._compile_and_erase_bases(0, 1)
        def f(a):
            b = a.clone()
            return b.view(-1), b.view(-1)

        inp = torch.ones(2, 2, requires_grad=True)
        out1, out2 = f(inp)

        self.assertIsNotNone(out1.grad_fn)
        self.assertExpectedInline(
            str(out1.grad_fn.__class__), """<class 'ViewBackward0'>"""
        )

        self.assertIsNotNone(out2.grad_fn)
        self.assertExpectedInline(
            str(out2.grad_fn.__class__), """<class 'ViewBackward0'>"""
        )

    def test_output_aliases_output_view_meta_replay(self):
        @self._compile_and_erase_bases(1)
        def f(a):
            b = a.add(10)
            return b, b.view(-1)

        inp = torch.ones(2, 2, requires_grad=True)
        out1, out2 = f(inp)

        self.assertEqual(out1.untyped_storage(), out2.untyped_storage())
        self.assertIsNotNone(out2.grad_fn)
        self.assertExpectedInline(
            str(out2.grad_fn.__class__), """<class 'ViewBackward0'>"""
        )

    @skipIfTorchDynamo()
    @patch("torch._dynamo.config.assume_static_by_default", False)



        # Overrides _base and _view_func tensor attributes, so as to avoid the view-replay
        # execution path when reconstructing views.
        # 重写 _base 和 _view_func 张量属性，以避免在重建视图时执行视图重播路径。
        class NoViewReplayTensor(torch.Tensor):
            @property
            def _base(self):
                return None

            @property
            def _view_func(self):
                return None

        # Wraps the outputs that are views of the FX graph 'g' with NoViewReplayTensor,
        # since they are the only ones that will get reconstructed.
        # 使用 NoViewReplayTensor 包装 FX 图 'g' 中是视图的输出，
        # 因为它们是唯一会被重建的部分。
        def wrapper(g, *args, **kwargs):
            outs = g(*args, **kwargs)
            for i in output_view_indices:
                outs[i] = NoViewReplayTensor(outs[i])
            return outs

        # Returns a lambda function that accepts a function f and compiles it using aot_function,
        # with fw_compiler parameter set to partial(wrapper, g), where g is the input FX graph.
        # 返回一个 lambda 函数，该函数接受一个函数 f，并使用 aot_function 进行编译，
        # 并指定 fw_compiler 参数为 partial(wrapper, g)，其中 g 是输入的 FX 图。
        return lambda f: aot_function(f, fw_compiler=lambda g, _: partial(wrapper, g))
    # 定义一个测试函数，用于测试动态输出的别名、输入视图元数据和重放。
    # 使用 torch.compile：这样我们可以在 FX 图中使用 SymInt。
    # 使用 inductor 进行编译，以便不追踪 tensor._base。
    #
    # 这应该强制在视图重建路径中使用 as_strided。
    # 前两个视图重放路径不会被采用，因为：
    #   - target_functional_tensor 将是符号化的（_functionalize_is_symbolic 调用）
    #   - tensor._base 将为 None
    @torch.compile(backend="inductor")
    def f(a, sz):
        return a.view(sz), a.view(-1)

    # 创建一个输入张量 inp，形状为 (2, 2)，所有元素均为1，且需要梯度跟踪
    inp = torch.ones(2, 2, requires_grad=True)
    # 调用函数 f，传入 inp 和元组 (4,) 作为参数
    out1, out2 = f(inp, (4,))

    # 断言 out1 的梯度函数不为 None
    self.assertIsNotNone(out1.grad_fn)
    # 断言 out1 的梯度函数类型符合预期
    self.assertExpectedInline(
        str(out1.grad_fn.__class__), """<class 'AsStridedBackward0'>"""
    )

    # 断言 out2 的梯度函数不为 None
    self.assertIsNotNone(out2.grad_fn)
    # 断言 out2 的梯度函数类型符合预期
    self.assertExpectedInline(
        str(out2.grad_fn.__class__), """<class 'ViewBackward0'>"""
    )
# 将传入的 fx_g 参数赋值给 graph_cell 列表的第一个元素，用于提取图形信息
def extract_graph(fx_g, _, graph_cell):
    graph_cell[0] = fx_g
    return fx_g


# 根据输入的 fx_g 参数，获取图中的输入节点和输出节点
def get_ins_outs(fx_g):
    ins = []
    outs = []
    # 遍历图中的节点
    for n in fx_g.graph.nodes:
        # 如果节点操作是 "placeholder"，则将节点添加到输入列表中
        if n.op == "placeholder":
            ins.append(n)
        # 如果节点操作是 "output"，则将其第一个参数作为输出列表的元组
        elif n.op == "output":
            outs = tuple(n.args[0])
    return ins, outs


# 根据输入的 fx_g 参数，返回图中输入和输出节点数量的元组
def get_num_ins_outs(fx_g):
    return tuple(len(i) for i in get_ins_outs(fx_g))


# 根据给定的函数 f、输入 inps、分区器 partitioner，默认使用最小切割再材料化分区，返回前向和反向图形
def get_fw_bw_graph(
    f, inps, partitioner=min_cut_rematerialization_partition, dynamic=False
):
    fw_graph_cell = [None]  # 初始化前向图形的列表
    bw_graph_cell = [None]  # 初始化反向图形的列表
    # 对给定的函数 f 进行 Ahead-of-Time 编译
    aot_function(
        f,
        fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),  # 提取前向图形并存储到 fw_graph_cell 中
        bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),  # 提取反向图形并存储到 bw_graph_cell 中
        partition_fn=partitioner,  # 使用指定的分区器函数进行分区
        decompositions=default_decompositions,  # 使用默认的分解策略
        dynamic=dynamic,  # 是否启用动态模式
    )(*inps).sum().backward()  # 对输入数据执行函数并进行反向传播
    return (fw_graph_cell[0], bw_graph_cell[0])  # 返回前向和反向图形的元组


# 测试模块类，继承自 torch.nn.Module
class TestMod(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(2, requires_grad=True))  # 创建一个参数张量 p
        self.fn = fn  # 存储传入的函数 fn

    def forward(self, *args):
        return self.fn(self.p, *args)  # 调用存储的函数 fn，传入参数 p 和 *args


# 测试 Ahead-of-Time 导出类，继承自 AOTTestCase
class TestAOTExport(AOTTestCase):
    def test_aot_export_ban_dropout_mut_pre_dispatch(self):
        # 定义一个函数 fn，接收参数 p 和 x，执行 dropout 操作并添加常数 1，返回结果元组
        def fn(p, x):
            y = torch.ops.aten.dropout.default(x, 0.1, train=False)
            y.add_(1)
            return (y,)

        mod = TestMod(fn)  # 创建 TestMod 类的实例 mod，传入 fn 作为构造函数参数
        inp = torch.randn(2, 2)  # 创建一个形状为 (2, 2) 的随机张量作为输入

        # 使用断言检查 RuntimeError 异常信息是否包含指定文本，验证预期的异常抛出
        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            # 在预期异常情况下，尝试进行 Ahead-of-Time 导出模块，禁用联合跟踪，启用预调度
            aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)

        # 执行 Ahead-of-Time 导出模块，传入 mod 和输入张量 inp，禁用联合跟踪和预调度
        gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=False)
        
        # 使用断言检查生成的前向图形的代码是否符合预期
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    clone = torch.ops.aten.clone.default(arg1_1);  arg1_1 = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    return (add,)""",
        )

        fw_graph_cell = [None]  # 初始化前向图形的列表
        bw_graph_cell = [None]  # 初始化反向图形的列表

        # 对函数 fn 进行 Ahead-of-Time 编译，提取前向和反向图形，并存储到相应的列表中
        compiled_outs = aot_function(
            fn,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
            partition_fn=default_partition,
            decompositions=default_decompositions,
            dynamic=True,
        )(*inp)
        
        fw_graph = fw_graph_cell[0]  # 获取存储在 fw_graph_cell 中的前向图形
        bw_graph = bw_graph_cell[0]  # 获取存储在 bw_graph_cell 中的反向图形

        # 使用断言检查生成的前向图形的代码是否符合预期
        self.assertExpectedInline(
            str(fw_graph.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    clone = torch.ops.aten.clone.default(arg1_1);  arg1_1 = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    return (add,)""",
        )
    def test_aot_export_predispatch_func_simple(self):
        # 定义一个简单的函数 fn，接受两个参数 p 和 x
        def fn(p, x):
            # 计算 y = x + 2
            y = x + 2
            # 使用 torch.no_grad() 上下文管理器，禁用梯度计算
            with torch.no_grad():
                # 将 y 原地加 2
                y.add_(2)
            # 返回一个元组，包含表达式 x * 2 + y 的结果
            return (x * 2 + y,)

        # 使用 TestMod 类创建一个模型 mod，传入定义的 fn 函数
        mod = TestMod(fn)
        # 创建一个形状为 (2, 2) 的随机输入张量 inp
        inp = torch.randn(2, 2)

        # 再次使用 torch.no_grad() 上下文管理器，确保模型导出时禁用梯度计算
        with torch.no_grad():
            # 调用 aot_export_module 函数导出模型 gm
            # trace_joint=False 表示不追踪整体模型，pre_dispatch=True 表示预调度模式
            gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        
        # 使用 self.assertExpectedInline 断言检查生成的 gm.code 是否符合预期
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    # 调用 Torch C++ 扩展函数 aten.add.Tensor，将 arg1_1 和标量 2 相加
    add = torch.ops.aten.add.Tensor(arg1_1, 2)
    # 调用 Torch C++ 扩展函数 _C._set_grad_enabled，关闭梯度跟踪
    _set_grad_enabled = torch._C._set_grad_enabled(False)
    # 再次调用 _C._set_grad_enabled，确保梯度跟踪仍然关闭
    add_1 = torch.ops.aten.add.Tensor(add, 2);  add = None
    # 再次调用 _C._set_grad_enabled，关闭梯度跟踪
    _set_grad_enabled_1 = torch._C._set_grad_enabled(False)
    # 调用 Torch C++ 扩展函数 aten.mul.Tensor，将 arg1_1 和标量 2 相乘，然后清空 arg1_1
    mul = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None
    # 调用 Torch C++ 扩展函数 aten.add.Tensor，将 mul 和 add_1 相加，然后清空 mul 和 add_1
    add_2 = torch.ops.aten.add.Tensor(mul, add_1);  mul = add_1 = None
    # 返回包含 add_2 的元组
    return (add_2,)
    def test_aot_export_predispatch_outdtype(self):
        # 定义一个内部测试函数 test_aot_export_predispatch_outdtype
        class M(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的模块 M
            def __init__(self, weight):
                # 模块初始化方法，接受权重参数
                super().__init__()
                # 调用父类初始化方法
                self.weight = weight
                # 将传入的权重赋值给模块属性 self.weight

            def forward(self, x):
                # 前向传播方法，接受输入张量 x
                y = x + 2
                # 对输入张量 x 加 2
                y.add_(5)
                # 张量 y 自身加 5
                return (
                    out_dtype(torch.ops.aten.mm.default, torch.int32, y, self.weight),
                    # 调用 out_dtype 函数，传入 torch.ops.aten.mm.default、torch.int32、y 和 self.weight 作为参数，并返回结果元组
                )

        weight = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        # 生成一个随机整数张量作为权重，数值范围在 -128 到 126 之间，数据类型为 torch.int8
        mod = M(weight)
        # 创建一个 M 类的实例 mod，传入之前生成的权重

        inp = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        # 生成一个随机整数张量作为输入，数值范围在 -128 到 126 之间，数据类型为 torch.int8

        gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        # 调用 aot_export_module 函数，将模块 mod 和输入 inp 作为参数进行 Ahead-of-Time 编译，设置 trace_joint=False 和 pre_dispatch=True

        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    # 启用梯度追踪
    _set_grad_enabled = torch._C._set_grad_enabled(True)
    # 执行矩阵乘法操作
    mm = torch.ops.aten.mm.default(arg1_1, arg1_1)
    # 禁用梯度追踪
    _set_grad_enabled_1 = torch._C._set_grad_enabled(False)
    # 对结果矩阵加常数 2
    add = torch.ops.aten.add.Tensor(mm, 2);  mm = None
    # 对输入张量进行求和操作
    sum_1 = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
    # 对两个求和结果进行加法操作
    sum_2 = torch.ops.aten.sum.default(add);  add = None
    # 对两个求和结果进行加法操作
    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    # 返回结果元组
    return (add_1,)
    # 定义一个测试方法，测试自动微分操作的预调度
    def test_aot_export_predispatch_with_autograd_op(self):
        # 定义一个嵌套函数 foo，接受参数 p 和 x
        def foo(p, x):
            # 启用 PyTorch 自动求导机制
            with torch.enable_grad():
                # 计算 y = x + 5
                y = x + 5
                # y 原地加上 5
                y.add_(5)
                # y 再次原地加上 7
                y.add_(7)
                # 返回一个包含 x 的余弦值和 y 的正弦值的元组
                return (x.cos() + y.sin(),)

        # 生成一个形状为 (2, 2) 的随机张量作为输入
        inp = torch.randn(2, 2)
        # 使用 TestMod 类创建一个模型实例 mod，以 foo 函数作为参数
        mod = TestMod(foo)

        # 在禁用梯度的上下文中，执行下列代码块
        with torch.no_grad():
            # 对模型进行预导出，生成 gm 对象，同时禁用追踪联合和预调度
            gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        
        # 使用 self.assertExpectedInline 方法断言 gm.code 的字符串表示是否符合预期
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    # 开启梯度计算
    _set_grad_enabled = torch._C._set_grad_enabled(True)
    # 在张量 arg1_1 上加 5
    add = torch.ops.aten.add.Tensor(arg1_1, 5)
    # 在上一个结果上再加 5，并清空第一个结果
    add_1 = torch.ops.aten.add.Tensor(add, 5);  add = None
    # 在上一个结果上再加 7，并清空第二个结果
    add_2 = torch.ops.aten.add.Tensor(add_1, 7);  add_1 = None
    # 计算张量 arg1_1 的余弦值，并清空 arg1_1
    cos = torch.ops.aten.cos.default(arg1_1);  arg1_1 = None
    # 计算上一个结果的正弦值，并清空该结果
    sin = torch.ops.aten.sin.default(add_2);  add_2 = None
    # 将余弦值和正弦值相加，并清空这两个中间结果
    add_3 = torch.ops.aten.add.Tensor(cos, sin);  cos = sin = None
    # 关闭梯度计算
    _set_grad_enabled_1 = torch._C._set_grad_enabled(False)
    # 返回结果的元组
    return (add_3,)
        return (getitem,)""",  # noqa: B950
        )



        # 返回一个包含单个元素 (getitem,) 的元组
        self.assertExpectedInline(
            # 调用断言方法 assertExpectedInline，比较两个字符串，预期为字符串 'getitem' 的元组
            str(gm.true_graph_0.true_graph_0.code).strip(),
            # 断言期望的内联内容，将字符串 'getitem' 转换为去除首尾空格后的字符串
            """\
# 定义一个名为 forward 的方法，接受一个参数 arg0_1
def forward(self, arg0_1):
    # 使用 torch.ops.aten.sin.default 执行正弦运算，并将结果存储在 sin 变量中；清空 arg0_1 引用
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    # 使用 torch.ops.aten.add.Tensor 将 sin 和标量值 7 相加，并将结果存储在 add 变量中；清空 sin 引用
    add = torch.ops.aten.add.Tensor(sin, 7);  sin = None
    # 使用 torch.ops.aten.sin.default 执行正弦运算，并将结果存储在 sin_1 变量中；清空 add 引用
    sin_1 = torch.ops.aten.sin.default(add);  add = None
    # 返回一个包含 sin_1 结果的元组
    return (sin_1,)

@unittest.skipIf(IS_WINDOWS, "Windows 不支持此测试用例")
@unittest.skipIf(
    not torchdynamo.is_dynamo_supported(), "TorchDynamo 不支持"
)
def test_aot_export_predispatch_map_1(self):
    # 定义一个名为 M 的类，继承自 torch.nn.Module
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()

        # 定义一个名为 forward 的方法，接受两个参数 x 和 y
        def forward(self, x, y):
            # 定义一个名为 true_fn 的内部函数，接受 x 和 r 两个参数
            def true_fn(x, r):
                # 使用 x.sin() 计算正弦值并将结果存储在 y 中
                y = x.sin()
                # 使用 y.add_(5) 对 y 自身进行原地加法操作
                y.add_(5)
                # 返回 y.cos() + r.sum() 的结果
                return y.cos() + r.sum()

            # 定义一个名为 false_fn 的内部函数，接受 x 和 r 两个参数
            def false_fn(x, r):
                # 使用 x.cos() 计算余弦值并将结果存储在 z 中
                z = x.cos()

                # 定义一个名为 f 的内部函数，接受 x 和 y 两个参数
                def f(x, y):
                    # 使用 x.cos() 计算余弦值并将结果存储在 a 中
                    a = x.cos()
                    # 使用 a.add_(5) 对 a 自身进行原地加法操作
                    a.add_(5)
                    # 返回 a + y 的结果
                    return a + y

                # 返回 z + control_flow.map(f, z, r).sum() + control_flow.map(f, z, r).sum() 的结果
                return (
                    z
                    + control_flow.map(f, z, r).sum()
                    + control_flow.map(f, z, r).sum()
                )

            # 使用 torch.cond(x.shape[0] > 4, true_fn, false_fn, [x, y]) 进行条件执行，返回结果赋值给 a
            a = torch.cond(x.shape[0] > 4, true_fn, false_fn, [x, y])
            # 返回一个包含 (a + 3, a + 4) 的元组作为 forward 方法的结果
            return (a + 3, a + 4)

    # 创建输入列表 inps，包含两个张量：torch.randn(2, 2) 和 torch.ones(2)
    inps = [torch.randn(2, 2), torch.ones(2)]
    # 调用 aot_export_module 方法，导出模块 M 的代码，并传入输入列表 inps
    gm, _ = aot_export_module(M(), inps, trace_joint=False, pre_dispatch=True)
    # 断言 gm.code 的字符串表示与预期的方法定义字符串匹配
    self.assertExpectedInline(
        str(gm.code).strip(),
        """\
def forward(self, arg0_1, arg1_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    conditional = torch.ops.higher_order.cond(False, true_graph_0, false_graph_0, [arg0_1, arg1_1]);  true_graph_0 = false_graph_0 = arg0_1 = arg1_1 = None
    getitem = conditional[0];  conditional = None
    add = torch.ops.aten.add.Tensor(getitem, 3)
    add_1 = torch.ops.aten.add.Tensor(getitem, 4);  getitem = None
    return (add, add_1)""",  # noqa: B950
    )
    # 断言 gm.true_graph_0.code 的字符串表示与预期的方法定义字符串匹配
    self.assertExpectedInline(
        str(gm.true_graph_0.code).strip(),
        """\
def forward(self, arg0_1, arg1_1):
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    add = torch.ops.aten.add.Tensor(sin, 5);  sin = None
    cos = torch.ops.aten.cos.default(add);  add = None
    sum_1 = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
    add_1 = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add_1,)""",
    )
    # 断言 gm.false_graph_0.code 的字符串表示与预期的方法定义字符串匹配
    self.assertExpectedInline(
        str(gm.false_graph_0.code).strip(),
        """\
def forward(self, arg0_1, arg1_1):
    cos = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None
    select = torch.ops.aten.select.int(cos, 0, 0)
    body_graph_0 = self.body_graph_0
    map_impl = torch.ops.higher_order.map_impl(body_graph_0, [cos], [arg1_1]);  body_graph_0 = None
    getitem = map_impl[0];  map_impl = None
    sum_1 = torch.ops.aten.sum.default(getitem);  getitem = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  sum_1 = None
""",
    )
    # 使用 torch 的操作选择第一个维度上的第一个元素
    select_1 = torch.ops.aten.select.int(cos, 0, 0)
    # 获取 self 对象的 body_graph_1 属性
    body_graph_1 = self.body_graph_1
    # 使用 torch 的高阶映射操作 map_impl 调用 body_graph_1，传入参数 [cos]，返回结果保存到 map_impl_1
    map_impl_1 = torch.ops.higher_order.map_impl(body_graph_1, [cos], [arg1_1]);  body_graph_1 = cos = arg1_1 = None
    # 从 map_impl_1 中获取第一个元素，赋值给 getitem_1，并清空 map_impl_1
    getitem_1 = map_impl_1[0];  map_impl_1 = None
    # 对 getitem_1 进行默认求和操作，结果保存到 sum_2，并清空 getitem_1
    sum_2 = torch.ops.aten.sum.default(getitem_1);  getitem_1 = None
    # 使用 torch 的张量加法操作，将 add 和 sum_2 相加，结果保存到 add_1，并清空 add 和 sum_2
    add_1 = torch.ops.aten.add.Tensor(add, sum_2);  add = sum_2 = None
    # 返回一个包含 add_1 的元组作为函数结果
    return (add_1,)""",
        )
        # 使用 self 对象的 gm 属性，访问 false_graph_0，然后访问其 body_graph_0 属性，然后访问其 code 属性的字符串表示
        self.assertExpectedInline(
            str(gm.false_graph_0.body_graph_0.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    # 调用 torch.aten.cos.default 操作，计算输入张量 arg0_1 的余弦值，然后释放 arg0_1
    cos = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None
    # 调用 torch.aten.add.Tensor 操作，将之前的余弦值张量 cos 和标量 5 相加，然后释放 cos
    add = torch.ops.aten.add.Tensor(cos, 5);  cos = None
    # 调用 torch.aten.add.Tensor 操作，将之前的相加结果 add 和输入张量 arg1_1 相加，然后释放 add 和 arg1_1
    add_1 = torch.ops.aten.add.Tensor(add, arg1_1);  add = arg1_1 = None
    # 返回包含 add_1 的元组
    return (add_1,)"""

def test_aot_export_predispatch_map_2(self):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            # 计算输入张量 x 的余弦值
            z = x.cos()

            def f(x, y):
                # 计算输入张量 x 的余弦值，并在原地加上 5
                a = x.cos()
                a.add_(5)
                return a + y

            # 使用 control_flow.map 调用函数 f，对 z 和 y 执行映射并求和
            return (z + control_flow.map(f, z, y).sum(),)

    inps = [torch.randn(2, 2), torch.ones(2)]
    # 导出模块 M 的代码和图形表示，用于预分发和追踪
    gm, _ = aot_export_module(M(), inps, trace_joint=False, pre_dispatch=True)
    self.assertExpectedInline(
        str(gm.code).strip(),
        """\
def forward(self, arg0_1, arg1_1):
    cos = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None
    # 获取 body_graph_0，并使用 torch.ops.higher_order.map_impl 执行映射操作，然后释放 body_graph_0 和 arg1_1
    body_graph_0 = self.body_graph_0
    map_impl = torch.ops.higher_order.map_impl(body_graph_0, [cos], [arg1_1]);  body_graph_0 = arg1_1 = None
    # 获取 map_impl 中的第一个元素并求和
    getitem = map_impl[0];  map_impl = None
    sum_1 = torch.ops.aten.sum.default(getitem);  getitem = None
    # 将 cos 和 sum_1 相加，然后释放 cos 和 sum_1
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    # 返回包含 add 的元组
    return (add,)""",
    )  # noqa: B950
    self.assertExpectedInline(
        str(gm.body_graph_0.code).strip(),
        """\
def forward(self, arg0_1, arg1_1):
    cos = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None
    # 使用 torch.ops.aten.add.Tensor 操作，将 cos 和标量 5 相加，然后释放 cos
    add = torch.ops.aten.add.Tensor(cos, 5);  cos = None
    # 使用 torch.ops.aten.add.Tensor 操作，将 add 和 arg1_1 相加，然后释放 add 和 arg1_1
    add_1 = torch.ops.aten.add.Tensor(add, arg1_1);  add = arg1_1 = None
    # 返回包含 add_1 的列表
    return [add_1]""",
    )

@unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
@unittest.skipIf(
    not torchdynamo.is_dynamo_supported(), "TorchDynamo is not supported"
)
def test_aot_export_predispatch_with_cond(self):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            def true_fn(x):
                # 计算输入张量 x 的正弦值
                y = x.sin()
                # 使用 torch.ops.aten.linear.default 执行线性操作，并加上随机生成的张量，然后在原地加上 5，并返回余弦值
                z = torch.ops.aten.linear.default(y, torch.randn(2, 2))
                z.add_(5)
                return z.cos()

            def false_fn(x):
                # 计算输入张量 x 的余弦值，并在原地加上 6，然后返回正弦值
                z = x.cos()
                z.add_(6)
                return z.sin()

            # 根据条件 x.shape[0] > 4，调用 true_fn 或 false_fn 函数
            a = torch.cond(x.shape[0] > 4, true_fn, false_fn, [x])
            # 返回元组，其中第一个元素为 a + 3，第二个元素为 a + 4
            return (a + 3, a + 4)

    inp = torch.randn(2, 2)
    # 导出模块 M 的代码和图形表示，用于预分发和追踪
    gm, _ = aot_export_module(M(), [inp], trace_joint=False, pre_dispatch=True)
    self.assertExpectedInline(
        str(gm.code).strip(),
        """\
def forward(self, arg0_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    # 使用 torch.ops.higher_order.cond 操作，在条件为 False 时，调用 false_graph_0；传递 arg0_1 后释放 true_graph_0、false_graph_0 和 arg0_1
    conditional = torch.ops.higher_order.cond(False, true_graph_0, false_graph_0, [arg0_1]);  true_graph_0 = false_graph_0 = arg0_1 = None""",
    )
    getitem = conditional[0];  conditional = None
    # 从条件列表中获取第一个元素，并清空条件列表引用
    add = torch.ops.aten.add.Tensor(getitem, 3)
    # 使用 Torch 提供的底层操作，在 getitem 和标量 3 之间执行张量加法
    add_1 = torch.ops.aten.add.Tensor(getitem, 4);  getitem = None
    # 使用 Torch 提供的底层操作，在 getitem 和标量 4 之间执行张量加法，并清空 getitem 引用
    return (add, add_1)""",  # noqa: B950
        )
        # 断言预期的内联代码与真实代码相匹配
        self.assertExpectedInline(
            str(gm.true_graph_0.code).strip(),
            """\
def forward(self, arg0_1):
    # 计算输入张量的正弦值
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    # 生成指定形状和设备的随机张量
    randn = torch.ops.aten.randn.default([2, 2], device = device(type='cpu'), pin_memory = False)
    # 对输入张量进行线性变换
    linear = torch.ops.aten.linear.default(sin, randn);  sin = randn = None
    # 将常数 5 加到张量中
    add = torch.ops.aten.add.Tensor(linear, 5);  linear = None
    # 计算张量的余弦值
    cos = torch.ops.aten.cos.default(add);  add = None
    # 返回结果的元组
    return (cos,)



def test_aot_export_predispatch_conv_and_bn(self):
    class ConvBatchnorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 定义卷积层和批归一化层
            self.conv = torch.nn.Conv2d(1, 3, 1, 1)
            self.bn = torch.nn.BatchNorm2d(3)

        def forward(self, x):
            # 执行卷积操作
            x = self.conv(x)
            # 执行批归一化操作
            x = self.bn(x)
            return (x,)

    mod = ConvBatchnorm()
    mod.train()
    inp = torch.randn(1, 1, 3, 3)

    # 导出模块并预分发
    gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
    self.assertExpectedInline(
        str(gm.code).strip(),
        """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1):
            # 执行卷积操作
            conv2d = torch.ops.aten.conv2d.default(arg7_1, arg0_1, arg1_1);  arg7_1 = arg0_1 = arg1_1 = None
            # 将张量和标量值相加
            add = torch.ops.aten.add.Tensor(arg6_1, 1);  arg6_1 = None
            # 执行批归一化操作
            _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(conv2d, arg2_1, arg3_1, arg4_1, arg5_1, True, 0.1, 1e-05);  conv2d = arg2_1 = arg3_1 = arg4_1 = arg5_1 = None
            # 获取批归一化结果的子项
            getitem = _native_batch_norm_legit_functional[0]
            getitem_3 = _native_batch_norm_legit_functional[3]
            getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
            # 返回结果的元组
            return (getitem_3, getitem_4, add, getitem)""",  # noqa: B950
    )

def test_aot_export_predispatch_reshape(self):
    class Reshape(torch.nn.Module):
        def forward(self, x):
            # 对输入张量进行形状重塑
            y = x.reshape(4, 4)
            return (y.sum(),)

    mod = Reshape()
    inp = torch.randn(2, 8)

    # 导出模块并预分发
    gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
    self.assertExpectedInline(
        str(gm.code).strip(),
        """\
def forward(self, arg0_1):
    # 对输入张量进行形状重塑
    view = torch.ops.aten.view.default(arg0_1, [4, 4]);  arg0_1 = None
    # 计算张量的元素之和
    sum_1 = torch.ops.aten.sum.default(view);  view = None
    # 返回结果的元组
    return (sum_1,)""",
    )  # noqa: B950

def test_aot_export_predispatch_contiguous(self):
    class Cont(torch.nn.Module):
        def forward(self, x):
            # 返回连续版本的输入张量
            y = torch.ops.aten.contiguous.default(x)
            # 计算张量的元素之和
            return (y.sum(),)

    mod = Cont()
    inp = torch.randn(2, 8)

    # 导出模块并预分发
    gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
    self.assertExpectedInline(
        str(gm.code).strip(),
        """\
def forward(self, arg0_1):
    # 计算张量的元素之和
    sum_1 = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
        return (sum_1,)""",
        )  # noqa: B950


    def test_aot_export_module_joint(self):
        # 定义一个继承自 torch.nn.Module 的子类 ConvBatchnormRelu，用于定义一个包含卷积、批归一化和ReLU激活的神经网络模型
        class ConvBatchnormRelu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个卷积层，输入通道数为1，输出通道数为3，卷积核大小为1x1，步长为1
                self.conv = torch.nn.Conv2d(1, 3, 1, 1)
                # 定义一个批归一化层，输入通道数为3
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                # 在前向传播中，先对输入进行卷积操作
                x = self.conv(x)
                # 然后对卷积结果进行批归一化操作
                x = self.bn(x)
                # 对批归一化后的结果应用ReLU激活函数
                user_out = torch.nn.functional.relu(x)
                # 计算ReLU激活后的输出的和作为损失值
                loss = user_out.sum()
                # 返回损失值和ReLU激活后的输出（detach用于分离计算图，不进行梯度计算）
                return loss, user_out.detach()

        # 创建 ConvBatchnormRelu 类的实例 mod
        mod = ConvBatchnormRelu()
        # 将模型设置为训练模式
        mod.train()
        # 创建一个形状为(1, 1, 3, 3)的随机输入张量
        inp = torch.randn(1, 1, 3, 3)
        # 使用模型对输入进行前向传播，得到输出 o_ref
        o_ref = mod(inp)
        # 将模型 mod 导出为即时模块，并返回导出的图 fx_g 和签名 signature
        fx_g, signature = aot_export_module(
            mod, [inp], trace_joint=True, output_loss_index=0
        )
        # 对导出的图 fx_g 中的每个节点，移除其元数据中的 "stack_trace" 字段
        for node in fx_g.graph.nodes:
            node.meta.pop("stack_trace", None)
        # 使用 self.assertExpectedInline 方法断言导出的可读输出是否符合预期
        self.assertExpectedInline(
            fx_g.print_readable(print_output=False),
            """\
# 定义了一个匿名类，继承自 torch.nn.Module 类
class <lambda>(torch.nn.Module):
    # 定义了 forward 方法，用于执行前向传播计算
    def forward(self, arg0_1: "f32[3, 1, 1, 1]", arg1_1: "f32[3]", arg2_1: "f32[3]", arg3_1: "f32[3]", arg4_1: "f32[3]", arg5_1: "f32[3]", arg6_1: "i64[]", arg7_1: "f32[1, 1, 3, 3]"):
        # 执行卷积操作，使用 torch.ops.aten.convolution.default 函数
        convolution: "f32[1, 3, 3, 3]" = torch.ops.aten.convolution.default(arg7_1, arg0_1, arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg7_1 = arg0_1 = arg1_1 = None
        # 对参数 arg6_1 执行加法操作，增加其值 1
        add: "i64[]" = torch.ops.aten.add.Tensor(arg6_1, 1);  arg6_1 = None
        # 执行批归一化操作，使用 torch.ops.aten._native_batch_norm_legit_functional.default 函数
        _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(convolution, arg2_1, arg3_1, arg4_1, arg5_1, True, 0.1, 1e-05);  convolution = arg2_1 = arg3_1 = arg4_1 = arg5_1 = None
        # 从批归一化结果中获取特定索引位置的张量
        getitem: "f32[1, 3, 3, 3]" = _native_batch_norm_legit_functional[0]
        # 从批归一化结果中获取特定索引位置的张量
        getitem_3: "f32[3]" = _native_batch_norm_legit_functional[3]
        # 从批归一化结果中获取特定索引位置的张量
        getitem_4: "f32[3]" = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
        # 执行 ReLU 激活函数操作，使用 torch.ops.aten.relu.default 函数
        relu: "f32[1, 3, 3, 3]" = torch.ops.aten.relu.default(getitem);  getitem = None
        # 对 relu 结果执行求和操作，使用 torch.ops.aten.sum.default 函数
        sum_1: "f32[]" = torch.ops.aten.sum.default(relu)
        # 对 relu 结果执行分离操作，使其与计算图分离，使用 torch.ops.aten.detach.default 函数
        detach: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(relu);  relu = None
        # 对 detach 结果执行分离操作，使其与计算图分离，使用 torch.ops.aten.detach.default 函数
        detach_1: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(detach);  detach = None
        # 对 detach_1 结果执行分离操作，使其与计算图分离，使用 torch.ops.aten.detach.default 函数
        detach_2: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(detach_1);  detach_1 = None
        # 返回计算结果的元组，包括 getitem_3, getitem_4, add, sum_1, detach_2
        return (getitem_3, getitem_4, add, sum_1, detach_2)
    def test_aot_export_simplified_basic(self):
        # 定义一个简单的函数 f，返回两个张量的乘积和第二个张量的平方
        def f(x, y):
            return x * y, y * y.detach()

        # 创建两个随机张量 x 和 y，需要计算梯度
        x = torch.randn(2, requires_grad=True)
        y = torch.randn(2, requires_grad=True)

        # 对函数 f 进行 ahead-of-time (AOT) 导出，不进行联合跟踪
        f_graph_fw = aot_export_joint_simple(f, [x, y], trace_joint=False)
        # 计算函数 f 的输出作为参考结果
        out_ref = f(x, y)
        # 通过导出的图形 f_graph_fw 计算输出，无需更改调用约定
        out_test = f_graph_fw(x, y)
        self.assertEqual(out_ref, out_test)

        # 测试反向传播
        x = torch.randn(2, requires_grad=True)
        y = torch.randn(2, requires_grad=True)
        # 对 x 和 y 进行克隆，并设置 requires_grad=True，以备后用
        x2 = x.clone().detach().requires_grad_(True)
        y2 = y.clone().detach().requires_grad_(True)
        x3 = x.clone().detach().requires_grad_(True)
        y3 = y.clone().detach().requires_grad_(True)
        # 对函数 f 进行联合跟踪
        f_graph_joint = aot_export_joint_simple(f, [x, y], trace_joint=True)
        num_fw_outputs = 2
        # 分割联合跟踪的前向图和后向图
        fw_g, bw_g = default_partition(
            f_graph_joint, [x, y], num_fwd_outputs=num_fw_outputs
        )
        # 计算函数 f 的输出作为参考结果
        out_ref2 = f(x2, y2)
        # 使用前向图 fw_g 计算输出
        fw_outs = fw_g(x3, y3)
        # 分离出前向输出和激活值
        out_test2, activations = fw_outs[:num_fw_outputs], fw_outs[num_fw_outputs:]
        self.assertEqual(out_ref2, out_test2)

        # 使用模拟的梯度输出测试运行跟踪后的反向图
        grad_outs = [torch.ones_like(x) for x in out_ref2]
        grads_ref = torch.autograd.grad(out_ref2, [x2, y2], grad_outputs=grad_outs)
        grads_test = bw_g(*activations, *grad_outs)
        for g_ref, g_test in zip(grads_ref, grads_test):
            self.assertEqual(g_ref, g_test)

    def test_aot_export_metadata_mutation_banned(self):
        # 定义一个函数 fn，对输入张量 x 进行转置，并返回 x 的两倍
        def fn(p, x):
            x.t_()
            return (x * 2,)

        # 创建一个 TestMod 实例 mod
        mod = TestMod(fn)
        # 创建一个随机输入张量 inp
        inp = torch.randn(2, 4)
        # 测试是否捕获到元数据变异的运行时错误
        with self.assertRaisesRegex(
            RuntimeError, "Found an input that received a metadata mutation"
        ):
            # 尝试导出函数 fn，不进行联合跟踪
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=False)
            # 尝试导出函数 fn，进行联合跟踪
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=True)
            # 尝试导出模块 mod，不进行联合跟踪
            aot_export_module(mod, [inp], trace_joint=False)

    def test_aot_export_forward_mutation_no_buffer_mut(self):
        # 定义一个简单的 PyTorch 模块 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer1", torch.ones(6, 4))

            def forward(self, x):
                # 在输入张量 x 上执行原位加法
                x.add_(4)
                return (x.cos().sum() + self.buffer1.sum(),)

        # 创建一个 M 类的实例 mod
        mod = M()
        # 创建一个全为1的张量作为输入 inp
        inp = torch.ones(6, 4)
        # 导出模块 mod 的图形表示和签名
        gm, sig = aot_export_module(mod, [inp], trace_joint=False)
        # 断言导出的图形表示字符串与预期的内联字符串一致
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    # 使用 torch.ops.aten.add.Tensor 函数将 arg2_1 和 4 相加，结果存储在 add 中；清空 arg2_1 引用
    add = torch.ops.aten.add.Tensor(arg2_1, 4);  arg2_1 = None
    # 使用 torch.ops.aten.cos.default 对 add 执行余弦计算，结果存储在 cos 中；清空 arg1_1 引用
    cos = torch.ops.aten.cos.default(add);  arg1_1 = None
    # 使用 torch.ops.aten.sum.default 对 cos 进行求和操作，结果存储在 sum_1 中；清空 cos 引用
    sum_1 = torch.ops.aten.sum.default(cos);  cos = None
    # 使用 torch.ops.aten.sum.default 对 arg0_1 进行求和操作，结果存储在 sum_2 中；清空 arg0_1 引用
    sum_2 = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
    # 使用 torch.ops.aten.add.Tensor 函数将 sum_1 和 sum_2 相加，结果存储在 add_1 中；清空 sum_1 和 sum_2 引用
    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    # 返回包含 add 和 add_1 的元组作为结果
    return (add, add_1)
    def test_aot_export_synthetic_bases_banned(self):
        # 定义测试函数 fn，接受参数 p, x, y，对 x 执行乘以2的操作，返回元组 (x + y,)
        def fn(p, x, y):
            x.mul_(2)
            return (x + y,)

        # 创建 TestMod 实例 mod，使用函数 fn 初始化
        mod = TestMod(fn)
        # 生成一个形状为 (2,) 的随机张量 inp
        inp = torch.randn(2)
        # 将 inp 变形为单行张量 inp2
        inp2 = inp.view(-1)
        # 使用断言检查运行时异常 RuntimeError，异常信息包含 "Encountered aliased inputs that are mutated"
        with self.assertRaisesRegex(
            RuntimeError, "Encountered aliased inputs that are mutated"
        ):
            # 对函数 fn 进行 AOT 导出，传入参数 [mod.p, inp, inp2]，不追踪联合操作
            aot_export_joint_simple(fn, [mod.p, inp, inp2], trace_joint=False)
            # 对函数 fn 进行 AOT 导出，传入参数 [mod.p, inp, inp2]，追踪联合操作
            aot_export_joint_simple(fn, [mod.p, inp, inp2], trace_joint=True)
            # 对模块 mod 进行 AOT 导出，传入参数 [inp, inp2]，不追踪联合操作
            aot_export_module(mod, [inp, inp2], trace_joint=False)

    def test_aot_export_input_dupes_banned(self):
        # 定义测试函数 fn，接受参数 p, x, y，对 x 执行乘以2的操作，返回元组 (x + y,)
        def fn(p, x, y):
            x.mul_(2)
            return (x + y,)

        # 创建 TestMod 实例 mod，使用函数 fn 初始化
        mod = TestMod(fn)
        # 生成一个形状为 (2,) 的随机张量 inp
        inp = torch.randn(2)
        # 使用断言检查运行时异常 RuntimeError，异常信息包含 "Encountered duplicated inputs that are mutated in the graph"
        with self.assertRaisesRegex(
            RuntimeError, "Encountered duplicated inputs that are mutated in the graph"
        ):
            # 对函数 fn 进行 AOT 导出，传入参数 [mod.p, inp, inp]，不追踪联合操作
            aot_export_joint_simple(fn, [mod.p, inp, inp], trace_joint=False)
            # 对函数 fn 进行 AOT 导出，传入参数 [mod.p, inp, inp]，追踪联合操作
            aot_export_joint_simple(fn, [mod.p, inp, inp], trace_joint=True)
            # 对模块 mod 进行 AOT 导出，传入参数 [inp, inp]，不追踪联合操作
            aot_export_module(mod, [inp, inp], trace_joint=False)

    def test_aot_export_multiple_outputs_require_grad_banned(self):
        # 定义测试函数 fn，接受参数 p, x，计算 p * x 和 p * x 的和作为返回值
        def fn(p, x):
            out = p * x
            return out, out.sum()

        # 创建 TestMod 实例 mod，使用函数 fn 初始化
        mod = TestMod(fn)
        # 生成一个形状为 (2,) 的随机张量 inp
        inp = torch.randn(2)
        # 使用断言检查运行时异常 RuntimeError，异常信息包含 "Found an output of the forward that requires gradients, that was not"
        with self.assertRaisesRegex(
            RuntimeError,
            "Found an output of the forward that requires gradients, that was not",
        ):
            # 对模块 mod 进行 AOT 导出，传入参数 [inp]，追踪联合操作，输出索引为 1 的损失
            aot_export_module(mod, [inp], trace_joint=True, output_loss_index=1)

    @unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
    @unittest.skipIf(
        not torch._dynamo.is_dynamo_supported(), "Cond needs dynamo to run"
    )
    def test_aot_export_with_torch_cond(self):
        # 定义 M 类，继承自 torch.nn.Module
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # 定义 true_fn 函数，接受参数 x，计算 x + 4，加上5，返回 x 的余弦值
                def true_fn(x):
                    y = x + 4
                    y.add_(5)
                    return x.cos()

                # 定义 false_fn 函数，接受参数 x，计算 x + 5，加上6，返回 x 的正弦值
                def false_fn(x):
                    y = x + 5
                    y.add_(6)
                    return x.sin()

                # 使用 torch.cond 根据 x.shape[0] > 4 条件选择执行 true_fn 或 false_fn，传入参数 [x]
                a = torch.cond(x.shape[0] > 4, true_fn, false_fn, [x])
                # 返回元组 (a + 3, a + 4)
                return (a + 3, a + 4)

        # 生成一个形状为 (3, 4) 的随机张量 inp
        inp = torch.randn(3, 4)
        # 对模块 M 实例进行 AOT 导出，传入参数 (inp,)，不追踪联合操作
        gm, _ = aot_export_module(M(), (inp,), trace_joint=False)
        # 使用断言检查 gm.code.strip() 的输出是否与预期字符串匹配
        self.assertExpectedInline(
            gm.code.strip(),
            """\
# 定义一个方法 `forward`，接受一个参数 `arg0_1`
def forward(self, arg0_1):
    # 将 `self.true_graph_0` 赋值给 `true_graph_0`
    true_graph_0 = self.true_graph_0
    # 将 `self.false_graph_0` 赋值给 `false_graph_0`
    false_graph_0 = self.false_graph_0
    # 调用高阶操作 `torch.ops.higher_order.cond`，传入条件 `False`、true 分支 `true_graph_0`、false 分支 `false_graph_0` 和参数列表 `[arg0_1]`，并获取返回值
    conditional = torch.ops.higher_order.cond(False, true_graph_0, false_graph_0, [arg0_1]);  true_graph_0 = false_graph_0 = arg0_1 = None
    # 从 `conditional` 中获取第一个元素，赋值给 `getitem`
    getitem = conditional[0];  conditional = None
    # 调用 `torch.ops.aten.add.Tensor`，使用 `getitem` 和常数 `3` 进行加法操作，结果赋值给 `add`
    add = torch.ops.aten.add.Tensor(getitem, 3)
    # 再次调用 `torch.ops.aten.add.Tensor`，使用 `getitem` 和常数 `4` 进行加法操作，结果赋值给 `add_1`；释放 `getitem` 的引用
    add_1 = torch.ops.aten.add.Tensor(getitem, 4);  getitem = None
    # 返回包含 `add` 和 `add_1` 的元组
    return (add, add_1)""",  # noqa: B950
        )

# 通过 `self.assertExpectedInline` 方法验证 `gm.true_graph_0.code.strip()` 输出是否符合预期
self.assertExpectedInline(
    gm.true_graph_0.code.strip(),
    """\
def forward(self, arg0_1):
    # 调用 `torch.ops.aten.add.Tensor`，使用 `arg0_1` 和常数 `4` 进行加法操作，结果赋值给 `add`
    add = torch.ops.aten.add.Tensor(arg0_1, 4)
    # 再次调用 `torch.ops.aten.add.Tensor`，使用 `add` 和常数 `5` 进行加法操作，结果赋值给 `add_1`；释放 `add` 的引用
    add_1 = torch.ops.aten.add.Tensor(add, 5);  add = None
    # 调用 `torch.ops.aten.cos.default`，使用 `arg0_1` 计算余弦函数，结果赋值给 `cos`；释放 `arg0_1` 的引用
    cos = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None
    # 返回包含 `cos` 的元组
    return (cos,)""",
)

# 通过 `self.assertExpectedInline` 方法验证 `gm.false_graph_0.code.strip()` 输出是否符合预期
self.assertExpectedInline(
    gm.false_graph_0.code.strip(),
    """\
def forward(self, arg0_1):
    # 调用 `torch.ops.aten.add.Tensor`，使用 `arg0_1` 和常数 `5` 进行加法操作，结果赋值给 `add`
    add = torch.ops.aten.add.Tensor(arg0_1, 5)
    # 再次调用 `torch.ops.aten.add.Tensor`，使用 `add` 和常数 `6` 进行加法操作，结果赋值给 `add_1`；释放 `add` 的引用
    add_1 = torch.ops.aten.add.Tensor(add, 6);  add = None
    # 调用 `torch.ops.aten.sin.default`，使用 `arg0_1` 计算正弦函数，结果赋值给 `sin`；释放 `arg0_1` 的引用
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    # 返回包含 `sin` 的元组
    return (sin,)""",
)
    def test_recompute_partitioning(self):
        def fn(a, b):
            return torch.sin(torch.sin(a)) + b
        # 定义一个测试函数 fn，计算 torch.sin(torch.sin(a)) + b

        # Reference calculation
        ref_a = torch.rand(10, 10, requires_grad=True)
        ref_b = torch.rand(10, 10, requires_grad=True)
        # 创建随机张量 ref_a 和 ref_b，要求梯度
        ref = fn(ref_a, ref_b)
        # 对 ref_a 和 ref_b 执行 fn 函数，得到结果 ref，并且求 ref 的和的梯度

        ref.sum().backward()
        # 对 ref 求和并反向传播梯度

        # Compiled function calculation
        res_a = ref_a.clone().detach().requires_grad_(True)
        res_b = ref_b.clone().detach().requires_grad_(True)
        # 克隆并分离 ref_a 和 ref_b，要求梯度

        def compile_fn(x, _):
            return x
        # 定义一个编译函数 compile_fn，返回其输入 x

        compiled_fn = compiled_function(
            fn, compile_fn, compile_fn, min_cut_rematerialization_partition
        )
        # 使用编译函数 compiled_function，编译 fn 函数，并应用指定的分区策略
        res = compiled_fn(res_a, res_b)
        # 对 res_a 和 res_b 执行编译后的函数 compiled_fn，得到结果 res

        res.sum().backward()
        # 对 res 求和并反向传播梯度

        assert torch.allclose(ref, res, atol=1e-3, rtol=1e-3)
        # 断言 ref 和 res 在给定的误差范围内相等
        assert torch.allclose(ref_a.grad, res_a.grad, atol=1e-3, rtol=1e-3)
        # 断言 ref_a 的梯度和 res_a 的梯度在给定的误差范围内相等
        assert torch.allclose(ref_b.grad, res_b.grad, atol=1e-3, rtol=1e-3)
        # 断言 ref_b 的梯度和 res_b 的梯度在给定的误差范围内相等

    def test_meta_tensor_inplace_op(self):
        # Following module results in inplace ops while tracing. The test checks
        # that the meta tensor information is stored for inplace ops.
        # 创建一个 MockModule 类，模拟包含 inplace 操作的模块
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(3072, 768, requires_grad=True)
                )
                self.bias = torch.nn.Parameter(torch.randn(3072, requires_grad=True))

            def forward(self, add_4):
                linear_4 = torch.nn.functional.linear(
                    add_4, self.weight, bias=self.bias
                )
                gelu = torch.nn.functional.gelu(linear_4)
                return gelu

        def check_meta_tensor(fx_g, _):
            for node in fx_g.graph.nodes:
                if node.op != "output":
                    assert "tensor_meta" in node.meta
            return fx_g
        # 定义一个检查函数 check_meta_tensor，确保图中的每个节点都包含 "tensor_meta" 元数据

        inp0 = torch.randn(16, 128, 768, requires_grad=True)
        # 创建一个随机张量 inp0，要求梯度
        inputs = [
            inp0,
        ]
        # 创建输入列表包含 inp0

        mod = MockModule().to(device="cpu")
        # 创建一个 MockModule 实例 mod，并将其移到 CPU 设备
        aot_mod = aot_module(mod, fw_compiler=check_meta_tensor)
        # 使用 aot_module 函数将 mod 编译成 aot_mod，同时应用检查函数 check_meta_tensor
        aot_mod(*inputs)
        # 对输入 inputs 执行 aot_mod

    def test_default_partitioner_getitem(self):
        mod = nn.LayerNorm([10])
        # 创建一个 LayerNorm 模块实例 mod，规范化维度为 10

        def f(x, mod_weight, mod_bias):
            return torch.nn.functional.layer_norm(
                x, [10], mod_weight, mod_bias, eps=1e-6
            )
        # 定义一个函数 f，使用 LayerNorm 对 x 进行规范化

        fw_graph, bw_graph = get_fw_bw_graph(
            f,
            [torch.randn(3, 10, requires_grad=True), mod.weight, mod.bias],
            partitioner=default_partition,
        )
        # 使用 get_fw_bw_graph 函数获取前向和后向图，使用默认分区器进行分区

        self.assertEqual(get_num_ins_outs(fw_graph), (3, 6))
        # 断言前向图的输入输出数目为 (3, 6)
        self.assertEqual(get_num_ins_outs(bw_graph), (6, 3))
        # 断言后向图的输入输出数目为 (6, 3)

    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    # 如果不使用 networkx，跳过当前测试
    # 定义一个测试方法，用于测试最小切割分区保存形状
    def test_min_cut_partitioner_save_shape(self):
        # 定义一个函数 f，接受一个参数 x，并对其进行操作
        def f(x):
            # 对输入 x 求和，dim=1 表示沿着第二个维度求和
            s = x.sum(dim=1)
            return s
        
        # 准备输入数据，包含一个 10x10 的全一张量，要求梯度计算
        inp = [torch.ones([10, 10], requires_grad=True)]
        # 获取前向图和后向图，dynamic=True 表示动态追踪图
        fw_graph, bw_graph = get_fw_bw_graph(f, inp, dynamic=True)
        # 获取前向图中的输入输出数目，期望为 (1, 3)
        self.assertEqual(get_num_ins_outs(fw_graph), (1, 3))
        # 获取后向图中的输入输出数目，期望为 (3, 1)
        self.assertEqual(get_num_ins_outs(bw_graph), (3, 1))
        # 确认前向输出的第一个节点名称为 "sum_1"
        self.assertEqual(str(fw_output[0]), "sum_1")
        # 确认前向输出的第二个节点名称为 "sym_size_int"
        self.assertEqual(str(fw_output[1]), "sym_size_int")
        # 确认前向输出的第三个节点名称为 "sym_size_int_1"

        # 准备输入数据，包含三个张量，均为随机值，要求梯度计算
        inp = [
            torch.randn(10, requires_grad=True),
            torch.randn((3, 10), requires_grad=True),
            torch.randn((2, 10), requires_grad=True),
        ]
        
        # 定义一个新的函数 f，接受三个参数 a, b, c
        def f(a, b, c):
            # 尝试测试如果在图中保存一个大小的元组会发生什么；
            # 由于追踪的方式，我们实际上永远不会这样做，但这仍然是各种大小操作的一个良好测试用例
            # 获取张量 b 的符号大小
            sb = torch.ops.aten.sym_size(b)
            # 获取张量 c 的大小
            sc = c.size()
            # 计算 x 为 sb[0] + sc[0]
            x = sb[0] + sc[0]
            # 构造新的大小元组 a_sz 为 (x, a.size(0))
            a_sz = (x, a.size(0))
            # 返回将 a 按照 a_sz 扩展后与 b, c 连接的张量
            return torch.cat([a.expand(a_sz), b, c])
        
        # 获取新定义函数 f 的前向图和后向图
        fw_graph, bw_graph = get_fw_bw_graph(f, inp, dynamic=True)
        # 获取前向图中的输入输出数目，期望为 (3, 4)
        self.assertEqual(get_num_ins_outs(fw_graph), (3, 4))
        # 获取后向图中的输入输出数目，期望为 (4, 3)
        self.assertEqual(get_num_ins_outs(bw_graph), (4, 3))
        # 获取前向图中的输入输出节点，命名为 outs
        _, outs = get_ins_outs(fw_graph)
        # 确保 outs 中除第一个节点外的所有节点都是符号节点
        self.assertTrue(all(is_sym_node(n) for n in outs[1:]))
    def test_default_partitioner_output_tensor_shape_tensor(self):
        # 定义输入张量列表，每个张量形状随机，需要梯度
        inp = [
            torch.randn(10, requires_grad=True),
            torch.randn((3, 10), requires_grad=True),
            torch.randn((2, 10), requires_grad=True),
            torch.randn((10, 1), requires_grad=True),
        ]

        def f(a, b, c, d):
            # 尝试在函数返回的输出中强制混合使用符号整数
            sb = b.size()  # 获取张量 b 的大小
            sc = c.size()  # 获取张量 c 的大小
            x = sb[0] + sc[0]  # 计算 sb 和 sc 第一个维度大小的和
            a_sz = (x, a.size(0))  # 定义新的大小元组 a_sz
            cat = torch.cat([a.expand(a_sz), b, c])  # 拼接张量 a, b, c
            mm = torch.mm(cat, d)  # 执行矩阵乘法，计算 mm
            mm2 = torch.mm(
                mm, a.view(mm.size(1), a.size(0))
            )  # 对 mm 执行矩阵乘法，这会保存 4 个新的整数用于反向传播。为什么呢？
            # 我需要做什么才能使其保存一个张量用于反向传播？
            return cat, sb, c, mm2  # 返回 cat, sb, c, mm2 四个张量

        fw_graph_cell = [None]  # 初始化前向图单元
        bw_graph_cell = [None]  # 初始化反向图单元
        compiled_outs = aot_function(
            f,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
            partition_fn=default_partition,  # 使用默认的分区函数
            decompositions=default_decompositions,  # 使用默认的分解方式
            dynamic=True,  # 设置为动态模式
        )(*inp)  # 执行编译后的函数，并传入输入张量
        fw_graph = fw_graph_cell[0]  # 获取编译后的前向图
        (compiled_outs[0].sum() + compiled_outs[2].sum()).backward()  # 对两个张量求和并执行反向传播
        bw_graph = bw_graph_cell[0]  # 获取编译后的反向图

        # 在前向图中，输出为 13 个，因为：
        # - 5 个原始输出（sb 是一个元组，扩展为 2 个符号整数）
        # - 8 个保存的输出用于反向传播：5 个张量，3 个符号整数
        self.assertEqual(get_num_ins_outs(fw_graph), (4, 13))
        # 在后向图中，输入为 10 个（梯度输出），因为：
        # - 前向图有 13 个输出
        # - 其中 1 个是输入的视图，会在图外重新生成，不参与反向传播
        # - 2 个用户输出是符号整数（b.size()），在反向传播中不生成切线
        self.assertEqual(get_num_ins_outs(bw_graph), (10, 4))
        _, fw_graph_out_nodes = get_ins_outs(fw_graph)
        self.assertEqual(
            # 前向输出包括 b.size()，它扩展为 2 个符号整数，
            #
            # TODO(whc)- 这里保存的张量/符号整数是否正确？
            # 我只是基于默认分区的测试通过来做的
            # 在 5 个原始前向输出中，第 4 个（c）是一个输入，
            # 它不会显示在编译后的前向图中
            [False, True, True, False, False] + [False] * 4 + [True] * 4,
            [is_sym_node(n) for n in fw_graph_out_nodes],
        )

        real_outs = f(*inp)  # 调用原始函数获取真实输出
        self.assertEqual(compiled_outs, real_outs)  # 比较编译后的输出和真实输出是否一致
        self.assertTrue(isinstance(real_outs[1], torch.Size))  # 确保真实输出的第二个元素是 torch.Size 类型

        # TODO(whc) 我们应该学会返回 torch.Size
        self.assertFalse(isinstance(compiled_outs[1], torch.Size))  # 确保编译后的输出的第二个元素不是 torch.Size 类型

    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    # 定义测试方法，验证分区器输出张量形状与张量之间的关系
    def test_min_cut_partitioner_output_tensor_shape_tensor(self):
        # 创建输入张量列表
        inp = [
            torch.randn(10, requires_grad=True),       # 随机张量，需要计算梯度
            torch.randn((3, 10), requires_grad=True),  # 随机张量，形状为 (3, 10)，需要计算梯度
            torch.randn((2, 10), requires_grad=True),  # 随机张量，形状为 (2, 10)，需要计算梯度
            torch.randn((10, 1), requires_grad=True),  # 随机张量，形状为 (10, 1)，需要计算梯度
        ]

        # 定义内部函数 f，接受四个参数 a, b, c, d
        def f(a, b, c, d):
            # 尝试在函数返回中强制混合输出和符号整数
            sb = b.size()  # 获取张量 b 的大小
            sc = c.size()  # 获取张量 c 的大小
            x = sb[0] + sc[0]  # 计算两个张量第一维度大小之和
            a_sz = (x, a.size(0))  # 创建元组 a_sz，包含 x 和 a 的第一维度大小
            cat = torch.cat([a.expand(a_sz), b, c])  # 拼接张量 a、b、c，根据 a_sz 扩展张量 a
            mm = torch.mm(cat, d)  # 执行矩阵乘法，cat 乘以 d
            mm2 = torch.mm(
                mm, a.view(mm.size(1), a.size(0))
            )  # 对 mm 执行矩阵乘法，a 转置后的形状作为第一个参数，a 的形状第一维作为第二个参数
            # 这将为反向传播保存 4 个新整数。为什么？
            # 我需要做什么才能使其保存一个张量用于反向传播？
            return cat, sb, c, mm2  # 返回拼接张量、张量 b 的大小、张量 c 和 mm2

        fw_graph_cell = [None]  # 前向图单元列表
        bw_graph_cell = [None]  # 后向图单元列表
        # 调用 aot_function 函数，编译函数 f，使用 min_cut_rematerialization_partition 分区函数和默认分解
        compiled_outs = aot_function(
            f,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),  # 部分应用提取图函数到前向图单元列表
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),  # 部分应用提取图函数到后向图单元列表
            partition_fn=min_cut_rematerialization_partition,  # 使用最小割重制分区
            decompositions=default_decompositions,  # 使用默认分解
            dynamic=True,  # 动态模式
        )(*inp)  # 展开输入张量列表作为参数

        fw_graph = fw_graph_cell[0]  # 获取前向图
        # 执行前向图计算结果的求和并反向传播
        (compiled_outs[0].sum() + compiled_outs[2].sum()).backward()
        bw_graph = bw_graph_cell[0]  # 获取后向图

        # 验证前向图的输入输出数目是否符合预期
        self.assertEqual(get_num_ins_outs(fw_graph), (4, 12))
        # 验证后向图的输入输出数目是否符合预期
        self.assertEqual(get_num_ins_outs(bw_graph), (9, 4))
        _, fw_graph_out_nodes = get_ins_outs(fw_graph)  # 获取前向图的输入输出节点

        # 验证前向图输出节点是否符合预期，包括 b.size()（扩展为 2 个符号整数）、4 个张量（用于 mm 的转置）、最后 3 个符号整数
        self.assertEqual(
            [False, True, True, False, False] + [False] * 4 + [True] * 3,
            [is_sym_node(n) for n in fw_graph_out_nodes],
        )

        real_outs = f(*inp)  # 执行函数 f 的真实输出
        self.assertEqual(compiled_outs, real_outs)  # 验证编译输出与真实输出是否相等
        self.assertTrue(isinstance(real_outs[1], torch.Size))  # 验证真实输出的第二个元素是否为 torch.Size 类型

        # TODO(whc) we should learn to return torch.Sizes
        self.assertFalse(isinstance(compiled_outs[1], torch.Size))  # 验证编译输出的第二个元素是否不为 torch.Size 类型

    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    def test_min_cut_partitioner(self):
        # 定义函数 f，对输入张量执行三次余弦函数
        def f(x):
            return x.cos().cos().cos()

        # 调用 get_fw_bw_graph 函数获取前向图和后向图，对 f 函数应用随机张量作为参数
        fw_graph, bw_graph = get_fw_bw_graph(f, [torch.randn(3, requires_grad=True)])
        # 验证前向图的输入输出数目是否符合预期
        self.assertEqual(get_num_ins_outs(fw_graph), (1, 2))
        # 验证后向图的输入输出数目是否符合预期
        self.assertEqual(get_num_ins_outs(bw_graph), (2, 1))

        # 定义函数 f，接受四个参数 a, b, c, d，计算它们的和后执行三次余弦函数
        def f(a, b, c, d):
            x = a + b + c + d  # 计算四个张量的和
            return x.cos().cos()  # 执行两次余弦函数

        # 调用 get_fw_bw_graph 函数获取前向图和后向图，对 f 函数应用四个随机张量作为参数
        fw_graph, bw_graph = get_fw_bw_graph(
            f, [torch.randn(3, requires_grad=True) for _ in range(4)]
        )
        # 验证前向图的输入输出数目是否符合预期
        self.assertEqual(get_num_ins_outs(fw_graph), (4, 2))
        # 验证后向图的输入输出数目是否符合预期
        self.assertEqual(get_num_ins_outs(bw_graph), (2, 4))
    # 定义一个测试方法，用于测试在反向传播中出现转置后跟视图的情况
    def test_contiguous(self):
        # 定义一个函数 f，对输入张量 x 进行视图变换（reshape）并转置
        def f(x):
            return x.view(2, 3).t()

        # 创建一个随机张量作为输入，需要计算梯度
        inp = torch.randn(6, requires_grad=True)
        # 使用 aot_function 对函数 f 进行 Ahead-of-Time 编译
        out = aot_function(f, nop)(inp)
        # 计算张量 out 对输入 inp 的梯度，使用随机梯度作为参数
        torch.autograd.grad(out, inp, torch.randn(3, 2))

    # 定义一个测试方法，用于验证随机性保持不变
    def test_preserve_random(self):
        # 定义一个函数 fn，对输入张量 x 进行 dropout 操作后加上原始张量
        def fn(x):
            return torch.nn.functional.dropout(x, 0.5) + x

        # 创建一个随机张量作为输入
        x = torch.randn(4)

        # 设置随机种子为 0，记录参考输出 ref
        torch.manual_seed(0)
        ref = fn(x)

        # 再次设置随机种子为 0，使用 aot_function 对函数 fn 进行 Ahead-of-Time 编译，记录结果 res
        torch.manual_seed(0)
        aot_fn = aot_function(fn, nop)
        res = aot_fn(x)

        # 断言：检查 ref 和 res 是否近似相等
        assert torch.allclose(ref, res)

    # 定义一个测试方法，验证生成的函数能够产生推理图
    # https://github.com/pytorch/pytorch/issues/110666
    def test_generate_gives_inference_graph(self):
        # 定义一个生成函数 generate，对输入张量 x 进行平方操作，并使用 torch.no_grad() 上下文管理器确保不计算梯度
        def generate(x):
            with torch.no_grad():
                return torch.mul(x, x)

        # 创建一个列表 inference_graph_cell 用于存储推理图
        inference_graph_cell = [None]
        # 创建一个推理编译器 inference_compiler，使用 partial 函数配置从图形中提取推理图的部分
        inference_compiler = make_boxed_compiler(
            partial(extract_graph, graph_cell=inference_graph_cell)
        )
        # 使用 aot_function 对生成函数 generate 进行 Ahead-of-Time 编译，使用推理编译器 inference_compiler
        aot_fn = aot_function(generate, nop, inference_compiler=inference_compiler)
        # 创建一个随机张量作为输入，需要计算梯度
        x = torch.randn(4, requires_grad=True)
        # 调用 aot_fn 对输入 x 进行推理
        res = aot_fn(x)
        # 断言：检查推理图是否已经生成
        self.assertTrue(inference_graph_cell[0] is not None)

    # 使用 unittest.skipIf 条件装饰器，如果 CUDA 不可用则跳过测试
    # 如果不使用 torchvision，则跳过测试
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    def test_autocast(self):
        # 创建一个 ResNet18 模型并移动到 CUDA 设备上
        mod = torchvision.models.resnet18().cuda()
        mod.train()

        # 创建一个随机张量作为输入，并移动到 CUDA 设备上
        x = torch.randn(16, 3, 32, 32, device="cuda")
        # 使用 memory_efficient_fusion 对模型进行内存高效融合处理
        aot_mod = memory_efficient_fusion(mod)

        # 在 CUDA 上启用 autocast 模式
        with torch.cuda.amp.autocast(True):
            # 对输入 x 应用 aot_mod 模型
            res = aot_mod(x)
        # 对结果进行求和并进行反向传播
        res.sum().backward()
class TestAOTDispatch(AOTTestCase):
    # Tests to add cases for (non-exhaustive list, mostly for my notes):
    # - subclass / mode introduced in the middle of the compiled fn
    # - various input mutation / intermediate base tests
    # - input mutation that changes a tensor into a subclass
    # - metadata mutation? (TBD)
    # - guard tests (fw guards *and* bw guards)
    # - subclass test involving _indices_of_inps_to_detach

    def test_aot_dispatch_simple(self):
        # a is a subclass, b is not
        # 定义一个函数 f，接受两个参数 a 和 b
        def f(a, b):
            # 对 a 进行乘法运算，结果存储在 aa 中
            aa = torch.mul(a, 6)
            # 对 b 进行除法运算，结果存储在 bb 中
            bb = torch.div(b, 2)
            # 返回 aa 和 bb 相加的结果
            return aa + bb

        # 创建参考数据 a_ref 和 b_ref
        a1_ref = torch.ones(3, 3, requires_grad=True)
        a2_ref = torch.ones(3, 3, requires_grad=True)
        a_ref = TwoTensor(a1_ref, a2_ref)
        b_ref = torch.ones(3, 3, requires_grad=True)

        # 创建测试数据 a_test 和 b_test，使用 clone() 复制并 detach() 分离梯度
        a1_test = a1_ref.clone().detach().requires_grad_(True)
        a2_test = a2_ref.clone().detach().requires_grad_(True)
        a_test = TwoTensor(a1_test, a2_test)
        b_test = b_ref.clone().detach().requires_grad_(True)

        # 初始化用于存储前向图和后向图的列表
        fw_graph_cell = [None]
        bw_graph_cell = [None]

        # 编译函数 f 成为 aot_function
        compiled_f = aot_function(
            f,
            # 使用 partial 将 extract_graph 函数与 fw_graph_cell 绑定
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            # 使用 partial 将 extract_graph 函数与 bw_graph_cell 绑定
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
            # 使用 min_cut_rematerialization_partition 作为分区函数
            partition_fn=min_cut_rematerialization_partition,
        )

        # 计算参考输出和测试输出
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)

        # 断言测试结果和参考结果的 TwoTensor 内部张量是否相等
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)

        # 对参考输出和测试输出的 sum() 结果进行反向传播
        out_ref.sum().backward()
        out_test.sum().backward()

        # 断言反向传播后的梯度结果是否相等
        self.assertEqual(a_ref.grad.a, a_test.grad.a)
        self.assertEqual(a_ref.grad.b, a_test.grad.b)
        self.assertEqual(b_ref.grad.a, b_test.grad.a)
        self.assertEqual(b_ref.grad.b, b_test.grad.b)

        # 重要的图形块：
        # - mul() 和 div() 出现两次，因为我们在 TwoTensor 上调用了它们
        # - add() 出现一次，因为我们在普通张量上调用了它
        # - 用户 forward() 函数返回一个输出（add 的结果），
        #   而图本身返回两个输出（add, add_1）
        # - add 和 add_1 对应于将被包装为单个 TwoTensor 输出的两个内部稠密张量
        # 输出部分，代码在下方示例中
        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    mul = torch.ops.aten.mul.Tensor(primals_1, 6);  primals_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(primals_2, 6);  primals_2 = None
    div = torch.ops.aten.div.Tensor(primals_3, 2);  primals_3 = None
    add = torch.ops.aten.add.Tensor(mul, div);  mul = None
    add_1 = torch.ops.aten.add.Tensor(mul_1, div);  mul_1 = div = None
"""
        )
    return [add, add_1]""",
        )

        # 重要的图形部分：
        # - 总共有4个密集输出。
        #   这对应于每个用户输入（a, b）都会得到一个梯度，它是TwoTensor子类的事实，
        #   因此(mul_2, mul_3)将被包装到a.grad中，
        #   而(div_1, div_2)将被包装到b.grad中。
        # - 总共有4个密集输出，
        self.assertExpectedInline(
            bw_graph_cell[0].code.strip(),
            """\

The annotations for the given code block have been provided as requested. If you have any more code that needs explanations, feel free to ask!
# 定义一个类方法 `forward`，接受两个张量作为输入，并返回计算结果列表
def forward(self, tangents_1, tangents_2):
    # 计算 tangents_1 的一半
    div_1 = torch.ops.aten.div.Tensor(tangents_1, 2)
    # 计算 tangents_2 的一半
    div_2 = torch.ops.aten.div.Tensor(tangents_2, 2)
    # 将 tangents_1 乘以 6，并将 tangents_1 置为 None
    mul_2 = torch.ops.aten.mul.Tensor(tangents_1, 6);  tangents_1 = None
    # 将 tangents_2 乘以 6，并将 tangents_2 置为 None
    mul_3 = torch.ops.aten.mul.Tensor(tangents_2, 6);  tangents_2 = None
    # 返回包含计算结果的列表，顺序为 mul_2, mul_3, div_1, div_2
    return [mul_2, mul_3, div_1, div_2]
    def test_aot_dispatch_incorrect_backward(self):
        # 定义一个测试函数，用于测试自动优化编译器的反向传播
        # a 是一个子类，b 不是
        def f(a, b):
            # 对 a 执行乘法操作
            aa = torch.mul(a, 2)
            # 对 b 执行加法操作
            bb = torch.add(b, 3)
            # 计算两个结果的除法
            out_subclass = torch.div(aa, bb)
            # 对 b 执行加法操作
            out_reg = torch.add(b, b)
            # 创建混合张量时，假设第二个 grad_out 不是子类。
            # 但在下面的测试用例中，我们的假设是错误的。
            # 这将需要重新跟踪和重新编译反向传播。
            return out_subclass, out_reg

        # 创建两个需要梯度的张量 a1_ref 和 a2_ref，初始化为全 1
        a1_ref = torch.ones(3, 3, requires_grad=True)
        a2_ref = torch.ones(3, 3, requires_grad=True)
        # 将这两个张量组合成一个 TwoTensor 对象 a_ref
        a_ref = TwoTensor(a1_ref, a2_ref)
        # 创建一个需要梯度的张量 b_ref，初始化为全 1
        b_ref = torch.ones(3, 3, requires_grad=True)

        # 克隆和分离 a1_ref 和 a2_ref，要求保留梯度信息
        a1_test = a1_ref.clone().detach().requires_grad_(True)
        a2_test = a2_ref.clone().detach().requires_grad_(True)
        # 将克隆后的张量组合成一个 TwoTensor 对象 a_test
        a_test = TwoTensor(a1_test, a2_test)
        # 克隆 b_ref，要求保留梯度信息
        b_test = b_ref.clone().detach().requires_grad_(True)

        # 使用自动优化编译器编译函数 f
        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )

        # 对参考数据进行函数计算
        out_ref = f(a_ref, b_ref)
        # 对测试数据进行编译后函数计算
        out_test = compiled_f(a_test, b_test)

        # 断言两组输出的第一个元素的属性 a 和 b 相等
        self.assertEqual(out_ref[0].a, out_test[0].a)
        self.assertEqual(out_ref[0].b, out_test[0].b)
        # 断言两组输出的第二个元素相等
        self.assertEqual(out_ref[1], out_test[1])

        # 我们在编译图时假设 type(grad_out[1]) == torch.Tensor，
        # 但实际上不是这样：在下面的测试中，它是一个子类。
        # 这将最终需要重新分区和重新编译
        with self.assertRaisesRegex(
            AssertionError,
            "incorrectly attempted to compile the backward with incorrect subclass metadata",
        ):
            # 对 out_test[0] + out_test[1] 求和，并进行反向传播
            (out_test[0] + out_test[1]).sum().backward()
    def test_aot_dispatch_output_alias(self):
        # 定义一个测试方法，用于测试 Ahead-Of-Time 编译的输出别名问题

        # 定义一个函数 f，接受两个参数 a 和 b，返回 b 的视图和 a 与 b 的乘积
        def f(a, b):
            return b.view(b.shape), a * b

        # 创建一个要测试的 TwoTensor 对象 b_ref，其两个成员张量 b1_ref 和 b2_ref 均为全1张量，并需要梯度
        b1_ref = torch.ones(3, 3, requires_grad=True)
        b2_ref = torch.ones(3, 3, requires_grad=True)
        b_ref = TwoTensor(b1_ref, b2_ref)

        # 创建一个测试用的 TwoTensor 对象 b_test，其两个成员张量 b1_test 和 b2_test 是 b1_ref 和 b2_ref 的克隆，并需要梯度
        b1_test = b1_ref.clone().detach().requires_grad_(True)
        b2_test = b2_ref.clone().detach().requires_grad_(True)
        b_test = TwoTensor(b1_test, b2_test)

        # 创建一个参考用的张量 a_ref，全1张量，并需要梯度
        a_ref = torch.ones(3, 3, requires_grad=True)

        # 创建一个测试用的张量 a_test，是 a_ref 的克隆，并需要梯度
        a_test = a_ref.clone().detach().requires_grad_(True)

        # 调用 Ahead-Of-Time 编译函数 aot_function，编译函数 f，使用 nop 作为前向和后向编译器，使用 min_cut_rematerialization_partition 作为分区函数
        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )

        # 调用函数 f，计算参考结果 out_ref1 和 out_ref2
        out_ref1, out_ref2 = f(a_ref, b_ref)

        # 调用编译后的函数 compiled_f，计算测试结果 out_test1 和 out_test2
        out_test1, out_test2 = compiled_f(a_test, b_test)

        # 使用断言检查 out_ref1 和 out_test1 是否相等
        self.assertEqual(out_ref1, out_test1)

        # 使用断言检查 out_ref2 的成员 a 和 b，与 out_test2 的成员 a 和 b 是否相等
        self.assertEqual(out_ref2.a, out_test2.a)
        self.assertEqual(out_ref2.b, out_test2.b)

        # 对 out_ref1 和 out_ref2 的和进行求和，并进行反向传播
        (out_ref1 + out_ref2).sum().backward()

        # 对 out_test1 和 out_test2 的和进行求和，并进行反向传播
        (out_test1 + out_test2).sum().backward()

        # 使用断言检查梯度是否正确传播，a_ref.grad.a 和 a_test.grad.a 应该相等
        self.assertEqual(a_ref.grad.a, a_test.grad.a)

        # 使用断言检查梯度是否正确传播，a_ref.grad.b 和 a_test.grad.b 应该相等
        self.assertEqual(a_ref.grad.b, a_test.grad.b)

        # 使用断言检查梯度是否正确传播，b_ref.grad.a 和 b_test.grad.a 应该相等
        self.assertEqual(b_ref.grad.a, b_test.grad.a)

        # 使用断言检查梯度是否正确传播，b_ref.grad.b 和 b_test.grad.b 应该相等
        self.assertEqual(b_ref.grad.b, b_test.grad.b)
    # 定义测试方法，用于测试自动编译和执行输入突变的函数
    def test_aot_dispatch_input_mutation(self):
        # 定义一个函数 f，接受两个参数 a 和 b，并对它们进行操作后返回结果
        def f(a, b):
            # 将 a 的每个元素乘以 2
            a.mul_(2)
            # 将 b 的每个元素乘以 3
            b.mul_(3)
            # 返回 a 和 b 元素相加的结果
            return a + b

        # 创建一个 3x3 的张量 b1_ref，要求计算梯度
        b1_ref = torch.ones(3, 3, requires_grad=True)
        # 创建一个 3x3 的张量 b2_ref，要求计算梯度
        b2_ref = torch.ones(3, 3, requires_grad=True)
        # 使用 b1_ref 和 b2_ref 创建 TwoTensor 对象 b_ref_base
        b_ref_base = TwoTensor(b1_ref, b2_ref)
        # 创建一个 3x3 的张量 a_ref_base，要求计算梯度
        a_ref_base = torch.ones(3, 3, requires_grad=True)
        # 将 a_ref_base 的所有元素加 1，并赋给 a_ref
        a_ref = a_ref_base + 1
        # 将 b_ref_base 的所有元素加 1，并赋给 b_ref
        b_ref = b_ref_base + 1

        # 克隆 b1_ref，并分离计算图，并要求计算梯度
        b1_test = b1_ref.clone().detach().requires_grad_(True)
        # 克隆 b2_ref，并分离计算图，并要求计算梯度
        b2_test = b2_ref.clone().detach().requires_grad_(True)
        # 使用 b1_test 和 b2_test 创建 TwoTensor 对象 b_test_base
        b_test_base = TwoTensor(b1_test, b2_test)
        # 克隆 a_ref_base，并分离计算图，并要求计算梯度
        a_test_base = a_ref_base.clone().detach().requires_grad_(True)
        # 将 a_test_base 的所有元素加 1，并赋给 a_test
        a_test = a_test_base + 1
        # 将 b_test_base 的所有元素加 1，并赋给 b_test
        b_test = b_test_base + 1

        # 使用 Ahead-Of-Time (AOT) 编译函数 f，并配置前向和反向编译器以及分区函数
        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )

        # 计算参考输出 out_ref，传入 a_ref 和 b_ref
        out_ref = f(a_ref, b_ref)
        # 计算测试输出 out_test，传入 a_test 和 b_test，使用编译后的函数
        out_test = compiled_f(a_test, b_test)

        # 断言参考输出的成员 a 和 b 等于测试输出的对应成员 a 和 b
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)

        # 确认输入参数的突变已生效
        self.assertEqual(a_test, a_ref)
        self.assertEqual(b_test.a, b_ref.a)
        self.assertEqual(b_test.b, b_ref.b)

        # 注意：我们需要在梯度计算中使用 b。否则我们将需要重新编译反向传播。
        # 计算参考梯度，对 b_ref 和 out_ref 进行点乘，然后进行反向传播
        (b_ref * out_ref).sum().backward()
        # 计算测试梯度，对 b_test 和 out_test 进行点乘，然后进行反向传播
        (b_test * out_test).sum().backward()

        # 断言梯度的输入是 TwoTensor 类型
        self.assertEqual(a_ref_base.grad.a, a_test_base.grad.a)
        self.assertEqual(a_ref_base.grad.b, a_test_base.grad.b)
        self.assertEqual(b_ref_base.grad.a, b_test_base.grad.a)
        self.assertEqual(b_ref_base.grad.b, b_test_base.grad.b)

    # 注意：子类的元数据突变当前已损坏且已禁用
    # 参见 https://github.com/pytorch/pytorch/issues/114975
    @unittest.expectedFailure
    # 定义一个内部函数 f，接受两个参数 a 和 b，并进行操作
    def test_aot_dispatch_input_metadata_mutation(self):
        # 在参数 a 上执行原地转置操作
        def f(a, b):
            a.t_()
            # 在参数 b 上增加一个维度
            b.unsqueeze_(0)
            # 返回 a 和 b 的和
            return a + b

        # 创建一个形状为 (3, 3)，需要梯度的浮点数张量 b1_ref
        b1_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        # 创建一个形状为 (3, 3)，需要梯度的浮点数张量 b2_ref
        b2_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        # 使用 b1_ref 和 b2_ref 创建 TwoTensor 对象 b_ref_base
        b_ref_base = TwoTensor(b1_ref, b2_ref)
        # 创建一个形状为 (3, 3)，不需要梯度的浮点数张量 a_ref_base
        a_ref_base = (
            torch.arange(9, dtype=torch.float32)
            .reshape(3, 3)
            .detach()  # 分离张量，使其不再跟踪历史记录
            .requires_grad_(True)  # 标记为需要梯度
        )
        # 对 a_ref_base 和 b_ref_base 分别加一，得到 a_ref 和 b_ref
        b_ref = b_ref_base + 1
        a_ref = a_ref_base + 1

        # 克隆 b1_ref 和 b2_ref，并标记需要梯度，得到 b1_test 和 b2_test
        b1_test = b1_ref.clone().detach().requires_grad_(True)
        b2_test = b2_ref.clone().detach().requires_grad_(True)
        # 使用 b1_test 和 b2_test 创建 TwoTensor 对象 b_test_base
        b_test_base = TwoTensor(b1_test, b2_test)
        # 克隆 a_ref_base，并标记需要梯度，得到 a_test_base
        a_test_base = a_ref_base.clone().detach().requires_grad_(True)
        # 对 a_test_base 和 b_test_base 分别加一，得到 a_test 和 b_test
        b_test = b_test_base + 1
        a_test = a_test_base + 1

        # 使用 AOT 编译函数 f，使用 nop 编译器，分区函数为 min_cut_rematerialization_partition
        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )
        # 分别用 a_ref 和 b_ref 调用函数 f 和编译后的函数 compiled_f，得到 out_ref 和 out_test
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)
        
        # 断言 out_ref 和 out_test 的属性 a 和 b 相等
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)

        # 确认输入的变异操作生效
        self.assertEqual(a_test, a_ref)
        self.assertEqual(b_test.a, b_ref.a)
        self.assertEqual(b_test.b, b_ref.b)

        # 注意：我们需要在梯度计算中使用 b。否则我们需要重新编译反向传播。
        # 计算 out_ref 和 out_test 与 b_ref 和 b_test 的乘积的和，并进行反向传播
        (b_ref * out_ref).sum().backward()
        (b_test * out_test).sum().backward()
        
        # 断言 a_ref_base 和 a_test_base 的梯度属性 a 和 b 相等
        self.assertEqual(a_ref_base.grad.a, a_test_base.grad.a)
        self.assertEqual(a_ref_base.grad.b, a_test_base.grad.b)
        # 断言 b_ref_base 和 b_test_base 的梯度属性 a 和 b 相等
        self.assertEqual(b_ref_base.grad.a, b_test_base.grad.a)
        self.assertEqual(b_ref_base.grad.b, b_test_base.grad.b)
    # 定义一个内部函数 f，接受两个参数 a 和 b，对其进行原地操作并返回运算结果
    def test_aot_dispatch_input_data_and_metadata_mutation(self):
        
        def f(a, b):
            # 在张量 a 上执行转置操作（原地修改）
            a.t_()
            # 在张量 b 上增加一个维度（原地修改）
            b.unsqueeze_(0)
            # 在张量 a 上每个元素乘以 2（原地修改）
            a.mul_(2)
            # 在张量 b 上每个元素乘以 3（原地修改）
            b.mul_(3)
            # 返回张量 a 和 b 的和
            return a + b
        
        # 创建一个张量 b1_ref，包含值从 0 到 8 的浮点数，需要计算梯度，形状为 3x3
        b1_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        # 创建一个张量 b2_ref，包含值从 0 到 8 的浮点数，需要计算梯度，形状为 3x3
        b2_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        # 创建一个 TwoTensor 对象 b_ref_base，由 b1_ref 和 b2_ref 构成
        b_ref_base = TwoTensor(b1_ref, b2_ref)
        # 创建一个张量 a_ref_base，包含值从 0 到 8 的浮点数，不需要计算梯度，形状为 3x3
        a_ref_base = (
            torch.arange(9, dtype=torch.float32)
            .reshape(3, 3)
            .detach()  # 分离出张量，使其不再跟踪计算历史
            .requires_grad_(True)  # 设置为需要计算梯度
        )
        # 在张量 b_ref_base 上每个元素加 1
        b_ref = b_ref_base + 1
        # 在张量 a_ref_base 上每个元素加 1
        a_ref = a_ref_base + 1
        
        # 克隆 b1_ref 并设置为需要计算梯度
        b1_test = b1_ref.clone().detach().requires_grad_(True)
        # 克隆 b2_ref 并设置为需要计算梯度
        b2_test = b2_ref.clone().detach().requires_grad_(True)
        # 创建一个 TwoTensor 对象 b_test_base，由克隆后的 b1_test 和 b2_test 构成
        b_test_base = TwoTensor(b1_test, b2_test)
        # 克隆 a_ref_base 并设置为需要计算梯度
        a_test_base = a_ref_base.clone().detach().requires_grad_(True)
        # 在张量 b_test_base 上每个元素加 1
        b_test = b_test_base + 1
        # 在张量 a_test_base 上每个元素加 1
        a_test = a_test_base + 1
        
        # 使用 ahead-of-time compilation 编译函数 f
        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )
        
        # 分别调用原始函数 f 和编译后的函数 compiled_f，传入 a_test 和 b_test
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)
        
        # 断言原始函数和编译后函数的输出 a 和 b 相等
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)
        
        # 确认输入的变异操作有效
        # 断言 a_test 与 a_ref 相等
        self.assertEqual(a_test, a_ref)
        # 断言 b_test 中的 a 与 b_ref 中的 a 相等
        self.assertEqual(b_test.a, b_ref.a)
        # 断言 b_test 中的 b 与 b_ref 中的 b 相等
        self.assertEqual(b_test.b, b_ref.b)
        
        # 注意：我们需要在梯度计算中使用 b。否则，我们需要重新编译反向传播。
        # 对 b_ref 和 out_ref 的乘积进行求和，并执行反向传播
        (b_ref * out_ref).sum().backward()
        # 对 b_test 和 out_test 的乘积进行求和，并执行反向传播
        (b_test * out_test).sum().backward()
        
        # 断言计算得到的梯度值在 a_ref_base 和 a_test_base 的 a 属性上相等
        self.assertEqual(a_ref_base.grad.a, a_test_base.grad.a)
        # 断言计算得到的梯度值在 a_ref_base 和 a_test_base 的 b 属性上相等
        self.assertEqual(a_ref_base.grad.b, a_test_base.grad.b)
        # 断言计算得到的梯度值在 b_ref_base 和 b_test_base 的 a 属性上相等
        self.assertEqual(b_ref_base.grad.a, b_test_base.grad.a)
        # 断言计算得到的梯度值在 b_ref_base 和 b_test_base 的 b 属性上相等
        self.assertEqual(b_ref_base.grad.b, b_test_base.grad.b)
    # 定义一个内部函数 f，接受两个参数 a 和 b，并对它们进行修改后返回一个元组
    def f(a, b):
        # 修改 a：每个元素乘以 2
        a.mul_(2)
        # 修改 b：每个元素乘以 3
        b.mul_(3)
        # 返回修改后的 b 的视图和 a 与 b 元素级相加的结果
        return b.view(b.shape), a + b

    # 创建一个 requires_grad=True 的浮点张量 b1_ref，形状为 (3, 3)，数值从 0 到 8
    b1_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
    # 创建一个 requires_grad=True 的浮点张量 b2_ref，形状为 (3, 3)，数值从 0 到 8
    b2_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
    # 创建一个 TwoTensor 对象 b_ref_base，包含 b1_ref 和 b2_ref
    b_ref_base = TwoTensor(b1_ref, b2_ref)
    # 创建一个浮点张量 a_ref_base，形状为 (3, 3)，数值从 0 到 8，并将其分离(detach)并标记为需要梯度计算
    a_ref_base = (
        torch.arange(9, dtype=torch.float32)
        .reshape(3, 3)
        .detach()
        .requires_grad_(True)
    )
    # b_ref 是 b_ref_base 的每个元素加 1 后的结果
    b_ref = b_ref_base + 1
    # a_ref 是 a_ref_base 的每个元素加 1 后的结果
    a_ref = a_ref_base + 1

    # 克隆 b1_ref 并标记为需要梯度计算的张量 b1_test
    b1_test = b1_ref.clone().detach().requires_grad_(True)
    # 克隆 b2_ref 并标记为需要梯度计算的张量 b2_test
    b2_test = b2_ref.clone().detach().requires_grad_(True)
    # 创建一个 TwoTensor 对象 b_test_base，包含 b1_test 和 b2_test
    b_test_base = TwoTensor(b1_test, b2_test)
    # 克隆 a_ref_base 并标记为需要梯度计算的张量 a_test_base
    a_test_base = a_ref_base.clone().detach().requires_grad_(True)
    # b_test 是 b_test_base 的每个元素加 1 后的结果
    b_test = b_test_base + 1
    # a_test 是 a_test_base 的每个元素加 1 后的结果
    a_test = a_test_base + 1

    # 编译函数 f 成为 aot_function，使用 nop 编译器，分区函数为 min_cut_rematerialization_partition
    compiled_f = aot_function(
        f,
        fw_compiler=nop,
        bw_compiler=nop,
        partition_fn=min_cut_rematerialization_partition,
    )
    # 使用 a_ref 和 b_ref 调用原始函数 f，返回两个张量 out_ref1 和 out_ref2
    out_ref1, out_ref2 = f(a_ref, b_ref)
    # 使用 a_test 和 b_test 调用编译后的函数 compiled_f，返回两个张量 out_test1 和 out_test2
    out_test1, out_test2 = compiled_f(a_test, b_test)

    # 断言原始函数和编译后函数的输出结果的 a 属性相等
    self.assertEqual(out_ref1.a, out_test1.a)
    # 断言原始函数和编译后函数的输出结果的 b 属性相等
    self.assertEqual(out_ref1.b, out_test1.b)
    # 断言原始函数和编译后函数的输出结果的 a 属性相等
    self.assertEqual(out_ref2.a, out_test2.a)
    # 断言原始函数和编译后函数的输出结果的 b 属性相等
    self.assertEqual(out_ref2.b, out_test2.b)

    # 确认输入的变异工作正常，断言 a_test 与 a_ref 相等
    self.assertEqual(a_test, a_ref)
    # 断言 b_test 的 a 属性与 b_ref 的 a 属性相等
    self.assertEqual(b_test.a, b_ref.a)
    # 断言 b_test 的 b 属性与 b_ref 的 b 属性相等
    self.assertEqual(b_test.b, b_ref.b)

    # 对 (out_ref1 * out_ref2) 的结果进行求和并执行反向传播
    (out_ref1 * out_ref2).sum().backward()
    # 对 (out_test1 * out_test2) 的结果进行求和并执行反向传播
    (out_test1 * out_test2).sum().backward()
    # 断言 a_ref_base 的梯度的 a 属性与 a_test_base 的梯度的 a 属性相等
    self.assertEqual(a_ref_base.grad.a, a_test_base.grad.a)
    # 断言 a_ref_base 的梯度的 b 属性与 a_test_base 的梯度的 b 属性相等
    self.assertEqual(a_ref_base.grad.b, a_test_base.grad.b)
class TestAOTModuleSimplified(AOTTestCase):
    def test_aot_module_simplified(self):
        # 定义一个 MockModule 类，继承自 torch.nn.Module，用于模拟神经网络模块
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在模块中定义一个线性层，输入维度为20，输出维度为30
                self.linear = torch.nn.Linear(20, 30)

            def forward(self, x, y):
                # 模块的前向传播函数，返回线性层处理后的结果加上输入 y
                return (self.linear(x) + y,)

        # 创建 MockModule 的实例 mod
        mod = MockModule()
        # 对模型进行零梯度初始化
        mod.zero_grad()

        # 生成一个大小为 [128, 20] 的随机张量 x，并设置 requires_grad=True
        x = torch.randn(128, 20, requires_grad=True)
        # 生成一个大小为 [128, 30] 的随机张量 y，并设置 requires_grad=True
        y = torch.randn(128, 30, requires_grad=True)
        # 将输入张量 x 和 y 放入列表 inputs 中
        inputs = [x, y]
        # 对 inputs 中的每个张量进行分离、克隆，并设置 requires_grad=True，生成 cloned_inputs
        cloned_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]

        # 对模型进行前向传播计算，得到 ref
        ref = mod(*inputs)
        # 对 ref[0] 的所有元素求和，并反向传播梯度
        ref[0].sum().backward()

        # 调用 aot_module_simplified 函数对模型进行简化 AOT 编译
        compiled_f = aot_module_simplified(mod, cloned_inputs, nop)
        # 对模型再次进行零梯度初始化
        mod.zero_grad()
        # 使用编译后的函数进行计算，得到 res
        res = compiled_f(*cloned_inputs)
        # 对 res[0] 的所有元素求和，并反向传播梯度
        res[0].sum().backward()

        # 断言 ref[0] 与 res[0] 的所有元素近似相等
        assert torch.allclose(ref[0], res[0])
        # 断言 inputs[0] 的梯度与 cloned_inputs[0] 的梯度近似相等
        assert torch.allclose(inputs[0].grad, cloned_inputs[0].grad)
        # 断言 inputs[1] 的梯度与 cloned_inputs[1] 的梯度近似相等
        assert torch.allclose(inputs[1].grad, cloned_inputs[1].grad)

    def test_aot_module_simplified_dynamic(self):
        # 定义一个 MockModule 类，继承自 torch.nn.Module，用于模拟神经网络模块
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在模块中定义一个线性层，输入维度为20，输出维度为30
                self.linear = torch.nn.Linear(20, 30)

            def forward(self, x, y):
                # 模块的前向传播函数，返回线性层处理后的结果加上输入 y
                return (self.linear(x) + y,)

        # 创建 MockModule 的实例 mod
        mod = MockModule()

        # 创建一个 ShapeEnv 对象
        shape_env = ShapeEnv()
        # 创建一个 FakeTensorMode 对象，传入 shape_env
        fake_mode = FakeTensorMode(shape_env=shape_env)

        # 生成一个大小为 [128, 20] 的随机张量 x，并设置 requires_grad=True
        x = torch.randn(128, 20, requires_grad=True)
        # 生成一个大小为 [128, 30] 的随机张量 y，并设置 requires_grad=True
        y = torch.randn(128, 30, requires_grad=True)

        # 将输入张量 x 和 y 放入列表 inputs 中
        inputs = [x, y]
        # 使用 fake_mode 对输入张量进行转换，生成 fake_inputs
        fake_inputs = [fake_mode.from_tensor(x) for x in inputs]
        # 调用 aot_module_simplified 函数对模型进行简化 AOT 编译
        compiled_f = aot_module_simplified(mod, fake_inputs, nop)

        # 对模型进行前向传播计算，得到 ref
        ref = mod(*inputs)
        # 对 ref[0] 的所有元素求和，并反向传播梯度
        ref[0].sum().backward()

        # 对 inputs 中的每个张量进行分离、克隆，并设置 requires_grad=True，生成 cloned_inputs
        cloned_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
        # 使用编译后的函数进行计算，得到 res
        res = compiled_f(*cloned_inputs)
        # 对 res[0] 的所有元素求和，并反向传播梯度
        res[0].sum().backward()

        # 使用 self.assertExpectedInline 断言 ShapeEnv 输出的格式保持一致
        self.assertExpectedInline(
            shape_env.format_guards(),
            """\
 - Eq(s1, 20)
 - Eq(s2, 30)""",
        )

        # 断言 ref[0] 与 res[0] 的所有元素近似相等
        assert torch.allclose(ref[0], res[0])
        # 断言 inputs[0] 的梯度与 cloned_inputs[0] 的梯度近似相等
        assert torch.allclose(inputs[0].grad, cloned_inputs[0].grad)
        # 断言 inputs[1] 的梯度与 cloned_inputs[1] 的梯度近似相等
        assert torch.allclose(inputs[1].grad, cloned_inputs[1].grad)

    # https://github.com/pytorch/pytorch/issues/105327
    def test_lift_fresh_copy_in_graph(self):
        # 定义一个简单的 PyTorch 模块类 MyMod
        class MyMod(torch.nn.Module):
            def forward(self, x):
                # 创建一个张量 _tensor_constant0，值为 [1]
                _tensor_constant0 = torch.tensor([1])
                # 调用 Torch 操作 aten.lift_fresh_copy.default 提升 _tensor_constant0 的副本
                lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(
                    _tensor_constant0
                )
                # 计算 x 和提升副本的乘积，并返回结果元组
                y = x.mul(lift_fresh_copy)
                return (y,)

        # 创建 MyMod 类的实例 mod
        mod = MyMod()
        # 创建 ShapeEnv 实例 shape_env
        shape_env = ShapeEnv()
        # 创建 FakeTensorMode 实例 fake_mode，使用 shape_env 初始化
        fake_mode = FakeTensorMode(shape_env=shape_env)
        # 创建一个全为 1 的张量 x，并设置 requires_grad=True
        x = torch.ones(4, requires_grad=True)
        # 将 x 放入列表 inputs 中
        inputs = [x]
        # 使用 fake_mode 将 inputs 中的张量转换为虚拟张量，并放入 fake_inputs 列表
        fake_inputs = [fake_mode.from_tensor(x) for x in inputs]
        # 使用 aot_module_simplified 函数编译 mod，使用 fake_inputs 和 nop 作为参数
        compiled_f = aot_module_simplified(mod, fake_inputs, nop)

        # 计算 mod 对 x 的前向传播结果
        out_ref = mod(x)
        # 计算编译后的函数 compiled_f 对 x 的结果
        out_test = compiled_f(x)
        # 断言编译前后的结果是否相等，去除梯度信息后比较
        self.assertEqual(out_ref[0].detach(), out_test[0].detach())

    def test_inference_python_dispatcher(self):
        # 从 unet 中提取的示例代码
        # 定义一个 MockModule 类，模拟 PyTorch 模块
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个 Upsample 层实例，配置为双线性插值，scale_factor=2
                self.upsample = torch.nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=True
                )

            def forward(self, x):
                # 对输入 x 进行上采样操作，并返回结果元组
                return (self.upsample(x),)

        # 创建 MockModule 类的实例 mod
        mod = MockModule()
        # 创建 ShapeEnv 实例 shape_env
        shape_env = ShapeEnv()
        # 创建 FakeTensorMode 实例 fake_mode，使用 shape_env 初始化
        fake_mode = FakeTensorMode(shape_env=shape_env)
        # 创建一个形状为 (2, 512, 40, 59) 的随机张量 x，不需要梯度信息
        x = torch.randn(2, 512, 40, 59)  # NB: must not require grad
        # 将 x 放入列表 inputs 中
        inputs = [x]
        # 使用 fake_mode 将 inputs 中的张量转换为虚拟张量，并放入 fake_inputs 列表
        fake_inputs = [fake_mode.from_tensor(x) for x in inputs]
        # 使用 aot_module_simplified 函数编译 mod，使用 fake_inputs 和 nop 作为参数
        compiled_f = aot_module_simplified(mod, fake_inputs, nop)
    # 定义一个测试函数，用于验证简化的 Ahead-of-Time（AOT）模块是否保留堆栈跟踪信息
    def test_aot_module_simplified_preserves_stack_trace(self):
        # 定义一个 MockModule 类，继承自 torch.nn.Module，用于模拟神经网络模块
        class MockModule(torch.nn.Module):
            # 初始化函数，创建一个线性层
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(20, 30)

            # 前向传播函数，接收输入 x 和 y，执行线性层计算和激活函数 relu 后返回结果
            def forward(self, x, y):
                z = self.linear(x)
                z = z + y
                z = z.relu()
                return (z,)

        # 创建一个 Torch FX 的 Tracer 对象
        tracer = torch.fx.Tracer()
        # 设置 Tracer 记录堆栈跟踪信息为 True
        tracer.record_stack_traces = True
        # 使用 Tracer 对 MockModule 进行追踪，生成计算图
        graph = tracer.trace(MockModule())
        # 使用 Tracer 根节点和生成的计算图创建 GraphModule 对象
        mod = torch.fx.GraphModule(tracer.root, graph)

        # 遍历模块中的每一个节点
        for node in mod.graph.nodes:
            # 如果节点的操作为 "output"，则跳过
            if node.op == "output":
                continue
            # 断言节点的堆栈跟踪信息不为空
            self.assertTrue(node.stack_trace is not None)
            # 断言堆栈跟踪信息中包含当前测试文件的文件名
            assert "test_aotdispatch.py" in node.stack_trace

        # 定义一个断言编译器函数，用于验证编译后的图模块的节点是否保留堆栈跟踪信息
        def assert_compiler(gm: torch.fx.GraphModule, _):
            # 遍历模块中的每一个节点
            for node in gm.graph.nodes:
                # 如果节点的操作为 "output" 或 "placeholder"，则跳过
                if node.op == "output" or node.op == "placeholder":
                    continue
                # 断言节点的堆栈跟踪信息不为空
                self.assertTrue(node.stack_trace is not None)
                # 断言堆栈跟踪信息中包含当前测试文件的文件名
                assert "test_aotdispatch.py" in node.stack_trace
            return gm.forward  # 返回一个 Python 可调用对象

        # 创建输入张量 x 和 y，形状为 (128, 20) 和 (128, 30)，并且需要梯度信息
        x = torch.randn(128, 20, requires_grad=True)
        y = torch.randn(128, 30, requires_grad=True)
        inputs = [x, y]

        # 调用 aot_module_simplified 函数进行模块的简化 Ahead-of-Time 编译
        compiled_f = aot_module_simplified(
            mod,  # 使用的模块对象
            inputs,  # 输入参数
            fw_compiler=assert_compiler,  # 前向编译器函数
            bw_compiler=assert_compiler  # 反向编译器函数
        )
        # 执行编译后的函数，传入输入参数，并获取结果
        res = compiled_f(*inputs)
        # 对结果的第一个元素执行求和后进行反向传播
        res[0].sum().backward()

    # 定义一个测试函数，用于验证简化的 Ahead-of-Time（AOT）模块在变异时是否保留堆栈跟踪信息
    def test_aot_module_simplified_preserves_stack_trace_from_mutation(self):
        # 定义一个 MockModule 类，继承自 torch.nn.Module，用于模拟神经网络模块
        class MockModule(torch.nn.Module):
            # 初始化函数，无需创建额外的模块
            def __init__(self):
                super().__init__()

            # 前向传播函数，接收输入 x，对 x[0] 执行乘以 2 的操作，并返回结果
            def forward(self, x):
                x_view = x[0]
                x_view.mul_(2)
                return (x + x,)  # 返回输入的两倍

        # 创建一个 Torch FX 的 Tracer 对象
        tracer = torch.fx.Tracer()
        # 设置 Tracer 记录堆栈跟踪信息为 True
        tracer.record_stack_traces = True
        # 使用 Tracer 对 MockModule 进行追踪，生成计算图
        graph = tracer.trace(MockModule())
        # 使用 Tracer 根节点和生成的计算图创建 GraphModule 对象
        mod = torch.fx.GraphModule(tracer.root, graph)

        # 遍历模块中的每一个节点
        for node in mod.graph.nodes:
            # 如果节点的操作为 "output"，则跳过
            if node.op == "output":
                continue
            # 断言节点的堆栈跟踪信息不为空
            self.assertTrue(node.stack_trace is not None)
            # 断言堆栈跟踪信息中包含当前测试文件的文件名
            assert "test_aotdispatch.py" in node.stack_trace

        # 定义一个断言编译器函数，用于验证编译后的图模块的节点是否保留堆栈跟踪信息
        def assert_compiler(gm: torch.fx.GraphModule, _):
            # 断言 torch.ops.aten.copy_.default 在所有节点的目标中
            assert torch.ops.aten.copy_.default in [x.target for x in gm.graph.nodes]
            # 遍历模块中的每一个节点
            for node in gm.graph.nodes:
                # 如果节点的目标为 torch.ops.aten.copy_.default
                if node.target == torch.ops.aten.copy_.default:
                    # 断言节点的元数据中包含 "stack_trace" 键
                    assert "stack_trace" in node.meta
                    # 断言节点的堆栈跟踪信息中包含特定的代码行 "x_view.mul_(2)"
                    assert "x_view.mul_(2)" in node.meta["stack_trace"]
            return gm.forward  # 返回一个 Python 可调用对象

        # 创建输入张量 x，形状为 (128, 20)
        x = torch.randn(128, 20)
        inputs = [x]

        # 调用 aot_module_simplified 函数进行模块的简化 Ahead-of-Time 编译
        aot_module_simplified(
            mod,  # 使用的模块对象
            inputs,  # 输入参数
            fw_compiler=assert_compiler,  # 前向编译器函数
            bw_compiler=assert_compiler,  # 反向编译器函数
            keep_inference_input_mutations=True,  # 保持推断输入的变异
        )
    def test_aot_module_simplified_fake_tensor_gm_raises(self):
        # 创建一个 FakeTensorMode 实例，用于操作伪造的张量
        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
        # 创建一个真实的张量，并标记为需要梯度
        real_x = torch.randn(4, requires_grad=True)
        # 将真实张量转换为伪造张量
        fake_x = fake_mode.from_tensor(real_x)
        # 创建另一个真实的张量
        real_z = torch.randn(4)
        # 将其转换为伪造张量
        fake_z = fake_mode.from_tensor(real_z)

        # 定义一个 MockModule 类，用于模拟神经网络模块
        class MockModule(torch.nn.Module):
            def forward(self, x):
                # 当访问一个自由变量的伪造张量时，它会被 MakeFx 视为常量，
                # 并导致该张量被追踪到图中，这是一个错误的情况。确保在这种情况下
                # 我们进行充分的报告。
                return (x + fake_z,)

        # 使用 assertRaisesRegex 断言上下文管理器捕获 AssertionError 异常，
        # 并检查错误消息中是否包含 "Unexpected fake"
        with self.assertRaisesRegex(AssertionError, "Unexpected fake"):
            # 调用 aot_module_simplified 函数，传入 MockModule 的实例、伪造张量 fake_x，
            # 和 nop 参数
            aot_module_simplified(MockModule(), (fake_x,), nop)
# 这里的条目存在问题需要修复。
# 每一个都是一个 bug（或需要调查的问题）的标记。

aot_autograd_failures = {
    # 数据相关的控制流
    xfail("cov"),  # 预期失败测试：cov 函数
    xfail("nn.functional.gaussian_nll_loss"),  # 预期失败测试：高斯负对数似然损失函数
    xfail("tensor_split"),  # 预期失败测试：张量分割函数
    xfail("corrcoef"),  # 预期失败测试：相关系数计算函数
    xfail("quantile"),  # 预期失败测试：分位数计算函数
    xfail("nanquantile"),  # 预期失败测试：带有 NaN 的分位数计算函数
    xfail("narrow"),  # 预期失败测试：张量缩窄函数
    xfail("istft"),  # 预期失败测试：逆短时傅立叶变换函数
    xfail("linalg.eig"),  # 预期失败测试：特征值分解函数
    skip("as_strided_scatter"),  # 跳过测试：as_strided_scatter 函数
    skip("as_strided", "partial_views"),  # 跳过测试：as_strided 函数的 partial_views 参数
    # 给定输入尺寸：(s0xs1x2)。计算的输出尺寸：...
    skip("max_pool2d_with_indices_backward"),  # 跳过测试：max_pool2d_with_indices_backward 函数
    skip("nn.functional.nll_loss", ""),  # 跳过测试：nn.functional.nll_loss 函数（UBSAN 失败）
    # 其他
    xfail("to_sparse"),  # 预期失败测试：to_sparse 函数
    xfail("corrcoef"),  # 预期失败测试：相关系数计算函数
    xfail("cov"),  # 预期失败测试：cov 函数
    xfail("chalf"),  # 预期失败测试：chalf 函数（RuntimeError: "sum_cpu" not implemented for 'ComplexHalf'）
    xfail("sparse.sampled_addmm"),  # 预期失败测试：稀疏矩阵乘法函数
    xfail("sparse.mm", "reduce"),  # 预期失败测试：稀疏矩阵乘法函数的 reduce 模式
    skip("nn.functional.binary_cross_entropy_with_logits"),  # 跳过测试：带 logits 的二分类交叉熵函数（有时失败？）
    skip("nn.functional.margin_ranking_loss"),  # 跳过测试：排名损失函数（有时不稳定）
    skip("linalg.lu_solve"),  # 跳过测试：LU 分解求解函数（有时不稳定）
    decorate("matmul", decorator=unittest.skipIf(IS_ARM64, "flaky")),  # 装饰测试：matmul 函数，根据平台 ARM64 来跳过（不稳定）
    decorate("__rmatmul__", decorator=unittest.skipIf(IS_ARM64, "flaky")),  # 装饰测试：__rmatmul__ 函数，根据平台 ARM64 来跳过（不稳定）
    # 覆盖 atol=1e-4, rtol=1e-5 也可以使用
    decorate(
        "svd_lowrank",
        decorator=toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-05)}),
    ),  # 装饰测试：svd_lowrank 函数，设定公差（根据数据类型）
    decorate(
        "linalg.householder_product",
        decorator=unittest.skipIf(IS_MACOS and IS_X86, "flaky"),
    ),  # 装饰测试：linalg.householder_product 函数，根据平台 MacOS 和 X86 来跳过（不稳定）
    decorate(
        "linalg.pinv",
        "singular",
        decorator=toleranceOverride({torch.float32: tol(atol=1e-05, rtol=1e-05)}),
    ),  # 装饰测试：linalg.pinv 函数，特定条件下的公差设定（根据数据类型）
    decorate(
        "nn.functional.interpolate",
        "bicubic",
        decorator=toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-05)}),
    ),  # 装饰测试：nn.functional.interpolate 函数，bicubic 模式下的公差设定（根据数据类型）
    # 在此配置下，conv2d 有时非确定性？
    decorate("nn.functional.conv2d", decorator=unittest.skipIf(IS_ARM64, "flaky")),  # 装饰测试：nn.functional.conv2d 函数，根据平台 ARM64 来跳过（不稳定）
}

symbolic_aot_autograd_failures = {
    xfail("combinations", ""),  # 预期失败测试：combinations 函数
    xfail(
        "index_fill", ""
    ),  # 预期失败测试：index_fill 函数（无法对具有符号尺寸/步幅的张量调用 sizes()）
    xfail("kthvalue", ""),  # 预期失败测试：kthvalue 函数（无法对具有符号尺寸/步幅的张量调用 sizes()）
    xfail(
        "linalg.lstsq", ""
    ),  # 预期失败测试：linalg.lstsq 函数（找不到符号化的元函数/分解）
    xfail(
        "linalg.lstsq", "grad_oriented"
    ),  # 预期失败测试：linalg.lstsq 函数，grad_oriented 模式（找不到符号化的元函数/分解）
    xfail(
        "linalg.lu_solve", ""
    ),  # 预期失败测试：linalg.lu_solve 函数（找不到符号化的元函数/分解）
    skip(
        "nn.functional.batch_norm", ""
    ),  # 跳过测试：nn.functional.batch_norm 函数（代理不跟踪的错误）
    xfail(
        "nn.functional.binary_cross_entropy", ""
    ),  # 预期失败测试：nn.functional.binary_cross_entropy 函数（找不到符号化的元函数）
    xfail(
        "nn.functional.cross_entropy", ""
    ),  # tensor的符号大小/步幅无法调用sizes()
    xfail(
        "nn.functional.ctc_loss", ""
    ),  # aten._ctc_loss.Tensor - 找不到符号元函数/装饰...
    xfail(
        "nn.functional.fractional_max_pool3d", ""
    ),  # rand()接收到无效的参数组合 - g...
    xfail(
        "nn.functional.group_norm", ""
    ),  # tensor的符号大小/步幅无法调用sizes()
    xfail(
        "nn.functional.nll_loss", ""
    ),  # tensor的符号大小/步幅无法调用sizes()
    xfail(
        "_segment_reduce", "lengths"
    ),  # aten.segment_reduce.default - 找不到符号元函数...
    xfail(
        "_segment_reduce", "offsets"
    ),  # aten.segment_reduce.default - 找不到符号元函数...
    xfail("trace", ""),  # tensor的符号大小/步幅无法调用sizes()
    xfail(
        "_upsample_bilinear2d_aa"
    ),  # RuntimeError: isIntList() INTERNAL ASSERT FAILED  预期IntList但得到GenericList
    decorate(
        "linalg.householder_product",
        decorator=unittest.skipIf(IS_MACOS and IS_X86, "flaky"),
    ),  # 如果在MacOS和x86下，则跳过"flaky"的单元测试装饰器
    xfail("fft.fft", ""),
    xfail("fft.hfft2", ""),
    xfail("fft.hfft", ""),
    xfail("fft.hfftn", ""),
    xfail("fft.ifft", ""),
    xfail("fft.ihfft2", ""),
    xfail("fft.ihfft", ""),
    xfail("fft.ihfftn", ""),
    xfail("fft.irfft2", ""),
    xfail("fft.irfft", ""),
    xfail("fft.irfftn", ""),
    xfail("fft.rfft2", ""),
    xfail("fft.rfft", ""),
    xfail("fft.rfftn", ""),
    xfail("stft", ""),  # tensor的符号大小/步幅无法调用sizes()
# 定义一个辅助测试函数，用于测试自动微分操作
def _test_aot_autograd_helper(self, device, dtype, op, dynamic=False):
    # 如果操作不支持自动微分，则跳过测试
    if not op.supports_autograd:
        self.skipTest("Op does not support autograd")

    # 这里列出一些不希望使用随机输入的操作
    cant_check_data_specialization = set(
        {
            "nn.functional.max_unpool1d",
            "nn.functional.max_unpool2d",
            "nn.functional.max_unpool3d",
        }
    )
    # 根据操作名称决定是否尝试检查数据特化
    try_check_data_specialization = op.name not in cant_check_data_specialization

    # 生成操作的样本输入，支持梯度计算
    sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
    for sample_input in sample_inputs_itr:
        # 准备传递给自动微分检查的参数和关键字参数
        t_args = [sample_input.input] + list(sample_input.args)
        t_kwargs = sample_input.kwargs
        try:
            # 执行自动微分检查
            aot_autograd_check(
                op.op,
                t_args,
                t_kwargs,
                dynamic,
                self.assertRaisesRegex,
                self.assertEqual,
                check_gradients=True,
                try_check_data_specialization=try_check_data_specialization,
            )
        except DynamicOutputShapeException:
            # 如果操作具有动态输出形状，跳过测试
            self.skipTest("Dynamic output shape operation in trace")
        except GuardOnDataDependentSymNode:
            # 特例处理，例如 '__getitem__' 操作不希望因此而失败测试
            if op.name == "__getitem__":
                self.skipTest("Dynamic output shape operation in trace")
            else:
                raise


# 定义一个辅助测试模块的自动微分函数
def _test_aot_autograd_module_helper(
    self, device, dtype, training, module_info, *, dynamic=False
):
    # 提取模块类和模块输入函数
    module_cls = module_info.module_cls
    module_inputs = module_info.module_inputs_func(
        module_info, device=device, dtype=dtype, requires_grad=True, training=training
    )
    # 遍历模块输入列表中的每个模块输入对象
    for module_input in module_inputs:
        # 如果当前模块输入的前向输入为 None，则跳过本次循环
        if module_input.forward_input is None:
            continue
        
        # 从模块输入对象中获取构造函数的参数和关键字参数
        args, kwargs = (
            module_input.constructor_input.args,
            module_input.constructor_input.kwargs,
        )
        # 使用获取的参数和关键字参数创建模块对象 m
        m = module_cls(*args, **kwargs)
        
        # 将模块对象 m 移动到指定的设备 device，并转换为指定的数据类型 dtype
        m.to(device).to(dtype)
        
        # 设置模块对象 m 是否处于训练模式
        m.train(training)

        # 对于懒加载模块，需要先传入一个输入以初始化参数
        args, kwargs = (
            module_input.forward_input.args,
            module_input.forward_input.kwargs,
        )
        # 将参数和关键字参数展开为扁平列表和结构信息
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))

        # 如果参数中存在 PackedSequence 类型的对象，则跳过本次循环
        if any(tuple(isinstance(flat_arg, PackedSequence) for flat_arg in flat_args)):
            continue

        # 如果模块类是 torch.nn.modules.lazy.LazyModuleMixin 的子类
        if issubclass(module_info.module_cls, torch.nn.modules.lazy.LazyModuleMixin):
            # 使用 torch.no_grad() 上下文管理器，调用模块对象 m 的前向方法
            with torch.no_grad():
                m(*args, **kwargs)

        # 定义一个特定值作为标记
        sentinel_val = -42
        # 判断每个参数是否为 torch.Tensor 类型，如果是则用 sentinel_val 标记，否则保持原样
        is_tensor_spec = [
            sentinel_val if isinstance(arg, torch.Tensor) else arg for arg in flat_args
        ]
        # 过滤出参数中的所有 torch.Tensor 对象
        args = [arg for arg in flat_args if isinstance(arg, torch.Tensor)]

        # 定义一个函数 f，用于执行函数调用
        def f(params_buffers_args):
            named_params, named_buffers, args = params_buffers_args
            cur_flat_args = list(is_tensor_spec)
            args = iter(args)
            for idx, v in enumerate(cur_flat_args):
                if v == sentinel_val:
                    cur_flat_args[idx] = next(args)
            c_args, c_kwargs = pytree.tree_unflatten(cur_flat_args, args_spec)
            params_and_buffers = {**named_params, **named_buffers}
            return torch.func.functional_call(m, params_and_buffers, c_args, c_kwargs)

        # 获取模块对象 m 中命名参数和命名缓冲区的字典
        named_params = dict(m.named_parameters(remove_duplicate=False))
        named_buffers = dict(m.named_buffers(remove_duplicate=False))
        # 计算命名参数和命名缓冲区的总数
        num_params_buffers = len(named_params) + len(named_buffers)
        # 使用 Ahead-Of-Time (AOT) 编译技术对函数 f 进行编译，生成编译后的函数对象 compiled_f
        compiled_f = aot_function(
            f, nop, num_params_buffers=num_params_buffers, dynamic=dynamic
        )
        # 构造包含命名参数、命名缓冲区和参数列表的参数组合 params_buffers_args
        params_buffers_args = [named_params, named_buffers, args]
        # 调用测试函数，验证 AOT 自动求导的前向和反向传播
        _test_aot_autograd_forwards_backwards_helper(
            f,
            compiled_f,
            params_buffers_args,
            self.assertRaisesRegex,
            self.assertEqual,
            True,
        )
# 定义一个测试类 TestEagerFusionOpInfo，继承自 AOTTestCase
class TestEagerFusionOpInfo(AOTTestCase):
    
    # 使用装饰器 ops，将 op_db 和 hop_db 中的操作作为参数传入，仅允许 torch.float 数据类型
    @ops(op_db + hop_db, allowed_dtypes=(torch.float,))
    # 使用装饰器 skipOps，指定跳过的测试函数，包括 test_aot_autograd_exhaustive，并提供 aot_autograd_failures 参数
    @skipOps(
        "TestEagerFusionOpInfo", "test_aot_autograd_exhaustive", aot_autograd_failures
    )
    # 定义测试函数 test_aot_autograd_exhaustive，接受 device、dtype 和 op 作为参数
    def test_aot_autograd_exhaustive(self, device, dtype, op):
        # 调用 _test_aot_autograd_helper 函数，传入 self、device、dtype 和 op 参数
        _test_aot_autograd_helper(self, device, dtype, op)

    # 使用装饰器 ops，将 op_db 和 hop_db 中的操作作为参数传入，仅允许 torch.float 数据类型
    @ops(op_db + hop_db, allowed_dtypes=(torch.float,))
    # 使用装饰器 patch，将 "functorch.compile.config.debug_assert" 的值设置为 True
    @patch("functorch.compile.config.debug_assert", True)
    # 使用装饰器 skipOps，指定跳过的测试函数，包括 test_aot_autograd_symbolic_exhaustive，并提供 aot_autograd_failures 和 symbolic_aot_autograd_failures 参数
    @skipOps(
        "TestEagerFusionOpInfo",
        "test_aot_autograd_symbolic_exhaustive",
        aot_autograd_failures | symbolic_aot_autograd_failures,
    )
    # 定义测试函数 test_aot_autograd_symbolic_exhaustive，接受 device、dtype 和 op 作为参数，并设置 dynamic 参数为 True
    def test_aot_autograd_symbolic_exhaustive(self, device, dtype, op):
        # 调用 _test_aot_autograd_helper 函数，传入 self、device、dtype、op 和 dynamic=True 参数
        _test_aot_autograd_helper(self, device, dtype, op, dynamic=True)
class TestEagerFusionModuleInfo(AOTTestCase):
    # TestEagerFusionModuleInfo 类，继承自 AOTTestCase，用于测试 eager 模式下的融合模块信息

    @modules(module_db, allowed_dtypes=(torch.float,))
    @decorateForModules(unittest.expectedFailure, aot_autograd_module_failures)
    # 为 test_aot_autograd_module_exhaustive 方法添加模块装饰器，指定模块数据库和允许的数据类型，并标记为预期失败

    def test_aot_autograd_module_exhaustive(self, device, dtype, training, module_info):
        # 测试方法：test_aot_autograd_module_exhaustive，接受设备、数据类型、训练状态和模块信息作为参数
        _test_aot_autograd_module_helper(self, device, dtype, training, module_info)
        # 调用辅助函数 _test_aot_autograd_module_helper 执行测试

    @modules(module_db, allowed_dtypes=(torch.float,))
    @decorateForModules(
        unittest.expectedFailure,
        aot_autograd_module_failures | symbolic_aot_autograd_module_failures,
    )
    # 为 test_aot_autograd_symbolic_module_exhaustive 方法添加模块装饰器，指定模块数据库和允许的数据类型，
    # 并标记为预期失败，包括符号化的自动求导模块失败情况

    def test_aot_autograd_symbolic_module_exhaustive(
        self, device, dtype, training, module_info
    ):
        # 测试方法：test_aot_autograd_symbolic_module_exhaustive，接受设备、数据类型、训练状态和模块信息作为参数
        _test_aot_autograd_module_helper(
            self, device, dtype, training, module_info, dynamic=True
        )
        # 调用辅助函数 _test_aot_autograd_module_helper 执行测试，设置 dynamic 参数为 True

# 实例化参数化测试类 TestAOTAutograd，仅适用于 CPU
instantiate_parametrized_tests(TestAOTAutograd)
only_for = "cpu"
# 实例化 TestPythonKey 测试类，为全局变量赋值，仅适用于 CPU
instantiate_device_type_tests(
    TestPythonKey,
    globals(),
    only_for=only_for,
)
# 实例化 TestEagerFusionOpInfo 测试类，为全局变量赋值，仅适用于 CPU
instantiate_device_type_tests(TestEagerFusionOpInfo, globals(), only_for=only_for)
# 实例化 TestEagerFusionModuleInfo 测试类，为全局变量赋值，仅适用于 CPU
instantiate_device_type_tests(TestEagerFusionModuleInfo, globals(), only_for=only_for)

# 标记为预期失败的测试集，包括指定的测试用例名称列表
@xfail_inherited_tests(
    [
        "test_set__and_data_mutation_bad",
        "test_subclass_metadata_mutation_req_grad_True",
        "test_subclass_metadata_mutation_req_grad_False",
    ]
)
# 如果已经使用 Dynamo，则跳过 TorchDynamo 测试
@skipIfTorchDynamo("This test suite already uses dynamo")
class TestAOTAutogradWithDynamo(TestAOTAutograd):
    """
    These are the same as TestAOTAutograd tests, but we run dynamo first to get a graph module.
    """

    def assertExpectedInline(self, *args, **kwargs):
        # These will have different outputs because dynamo returns a different graph module
        # But we don't really care about that assertion when testing with dynamo,
        # only that the outputs match, etc.
        pass

    # 创建编译器以传递给 Dynamo
    def make_compiler(self, graph_cell):
        return make_boxed_compiler(partial(extract_graph, graph_cell=graph_cell))

    # 运行自动求导
    def run_autograd(
        self,
        f: Callable,
        fw_graph_cell: List[Optional[Callable]],
        decompositions: Optional[Dict],
        keep_input_mutations: bool,
        dynamic: bool,
        # 动态测试标志
        """
        Runs dynamo and aot_autograd with the specified settings
        """

        # 定义 dynamo_compiler 函数，接受 gm、inputs 和其他关键字参数
        def dynamo_compiler(gm, inputs, **kwargs):
            # 使用简化的 aot_module_simplified 函数编译 gm 和 inputs
            result = aot_module_simplified(
                gm,
                inputs,
                # 使用 self.make_compiler 创建前向图编译器
                fw_compiler=self.make_compiler(fw_graph_cell),
                # 使用 self.make_compiler 创建反向图编译器
                bw_compiler=self.make_compiler([None]),
                # 指定的分解方法
                decompositions=decompositions,
                # 是否保留推断输入变异
                keep_inference_input_mutations=keep_input_mutations,
                # 动态参数根据输入是否有虚拟张量进行计算
                # （此注释应与代码对齐，但它被错误地放置在了前一个注释的末尾）
            )
            return result

        # 定义 torch_compile_wrapper 函数，接受任意位置和关键字参数
        def torch_compile_wrapper(*args, **kwargs):
            # 重置 torch._dynamo 的状态
            torch._dynamo.reset()
            # 使用 torch.compile 编译函数 f，指定后端为 dynamo_compiler
            fn = torch.compile(f, backend=dynamo_compiler)
            try:
                # 尝试调用编译后的函数 fn，传入任意位置和关键字参数
                result = fn(*args, **kwargs)
            except torch._dynamo.exc.BackendCompilerFailed as e:
                # 为了让 assertRaises 正常工作，重新引发异常 e 的内部异常
                raise e.inner_exception from e
            return result

        # 返回 torch_compile_wrapper 函数作为结果
        return torch_compile_wrapper
class MockFXGraphCache:
    """
    In memory version of FXGraphCache so we can isolate testing for FXGraphCache
    """

    def __init__(self):
        # 初始化缓存字典
        self.cache = {}

    def save(self, key, gm):
        # 将给定的键和图模型保存到缓存中
        self.cache[key] = gm

    def load(self, gm, inputs):
        # 计算 FX 图的哈希值作为键
        key = compiled_fx_graph_hash(gm, inputs, {}, {})
        if key in self.cache:
            # 如果缓存中已存在，则将图模型封装并返回
            gm = make_boxed_func(gm)
            gm._fx_graph_cache_key = key
            return gm
        else:
            # 否则保存到缓存中并封装返回
            self.save(key, gm)
            gm = make_boxed_func(gm)
            gm._fx_graph_cache_key = key
            return gm

    def _lookup_graph(self, key, inputs, local, remote_cache):
        # 根据键查找缓存中的图模型，并封装返回
        gm = self.cache.get(key)
        if gm is not None:
            gm = make_boxed_func(gm)
        return gm


# The following tests fail in strict caching mode (i.e. they bypass or
# cache miss instead of cache hitting). They will be fixed in the PRs above this.
FAILING_CACHE_TESTS = (
    # BypassAOTAutogradCache: unsupported nodes
    "test_backward_mutation_data",
    "test_backward_mutation_metadata",
    "test_custom_autograd",
    "test_inner_grad",
    "test_input_mutation_set__nop",
    "test_nonidempotent_amp",  # einsum
    # Pickle error: OutputAliasInfo/functional tensor
    "test_input_aliased_with_mutation_output_alias",
    "test_input_data_and_metadata_mutation",
    "test_input_mutation_aliases_and_output_alias",
    "test_input_mutation_alias_everything",
    "test_input_mutation_and_output_view",
    "test_input_mutation_output_view_multiple",
    "test_input_output_aliase_custom_autograd_function",
    "test_input_output_view_metadata_mutate_multiple",
    "test_input_output_view_mutate_multiple",
    "test_input_output_view_simple",
    "test_output_aliases_intermediate_and_returned",
    "test_output_aliases_intermediate_and_returned_different_grad",
    "test_output_aliases_intermediate_and_returned_flipped",
    "test_output_aliases_intermediate_multiple",
    "test_output_aliases_intermediate_multiple_mixed",
    "test_output_aliases_intermediate_returned_multiple_times",
    "test_output_aliases_multiple_inputs_get_correct_one",
    "test_output_all_alias_types",
    "test_some_outputs_dont_require_grad_view",
    "test_view_and_inplace_view",
    "test_view_detach",
    "test_some_output_requires_grad_input_doesnt",
)


@xfail_inherited_tests(FAILING_CACHE_TESTS)
class TestAOTAutogradWithCache(TestAOTAutogradWithDynamo):
    """
    In memory version of FXGraphCache so we can isolate testing for FXGraphCache
    """

    def make_compiler(self, fw_graph_cell):
        # 使用预期失败的缓存测试来标记父类测试用例
        mock_inductor_cache = self.inductor_cache

        def compiler(gm, inputs):
            nonlocal mock_inductor_cache, fw_graph_cell
            # 加载图模型，并将其设置到给定的单元格中
            result = mock_inductor_cache.load(gm, inputs)
            fw_graph_cell[0] = gm
            return result

        return compiler
    # 重写父类方法 `run_autograd`，执行自动求导过程
    def run_autograd(
        self,
        f: Callable,
        fw_graph_cell: List[Optional[Callable]],
        decompositions: Optional[Dict],
        keep_input_mutations: bool,
        dynamic: bool,
    ):
        # 调用父类的 `run_autograd` 方法，传递参数并返回结果
        return super().run_autograd(
            f,
            fw_graph_cell,
            decompositions,
            keep_input_mutations,
            dynamic,
        )

    # 使用装饰器对 `verify_aot_autograd` 方法进行配置修改
    @torch._functorch.config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    @torch._inductor.config.patch("fx_graph_cache", True)
    # 验证 AOT 自动求导的方法
    def verify_aot_autograd(
        self,
        f,
        inp_: Union[Callable, List[Any]],
        *,
        test_mutation: bool = False,
        keep_inp_mutations: bool = False,
        decompositions: Optional[Dict] = None,
        dynamic: bool = False,
        # 仅当 `inp_` 是 Callable 类型时生效
        # TODO: 可能合并所有测试以将 `inp` 统一为 Callable 类型
        make_inputs_subclasses: bool = False,
    ):
        # 设定当前对象的 `inductor_cache` 属性为 `MockFXGraphCache` 的实例
        self.inductor_cache = MockFXGraphCache()
        # 清空 AOTAutogradCache
        AOTAutogradCache.clear()
        # 使用 `patch` 修改 `torch._inductor.codecache.FxGraphCache._lookup_graph` 方法的行为
        with patch(
            "torch._inductor.codecache.FxGraphCache._lookup_graph",
            new=self.inductor_cache._lookup_graph,
        ):
            # 调用父类的 `verify_aot_autograd` 方法，传递参数并返回结果
            return super().verify_aot_autograd(
                f,
                inp_,
                test_mutation=test_mutation,
                keep_inp_mutations=keep_inp_mutations,
                decompositions=decompositions,
                dynamic=dynamic,
                make_inputs_subclasses=make_inputs_subclasses,
            )

    # 测试输入变化为假时的别名问题
    def test_input_mutation_false_aliasing(self):
        # 此测试被禁用，因为在严格缓存模式下会失败
        # 但不能使用 xfail，因为会导致 ASAN 的未定义行为
        self.skipTest("Skipping because it fails in strict cache mode")
# 如果当前脚本作为主程序运行，则调用 run_tests() 函数来执行测试
if __name__ == "__main__":
    run_tests()
```
# `.\pytorch\test\test_proxy_tensor.py`

```
# Owner(s): ["module: ProxyTensor"]

# 从 torch.testing._internal.common_utils 导入 TestCase 和 run_tests 模块
from torch.testing._internal.common_utils import TestCase, run_tests
# 导入 torch 库
import torch
# 导入 torch._dynamo 模块
import torch._dynamo
# 导入 unittest 模块
import unittest
# 导入警告模块
import warnings
# 导入 operator 操作符模块
import operator
# 导入 Iterable 抽象基类
from collections.abc import Iterable
# 从 torch.nn.utils 导入 stateless 模块
from torch.nn.utils import stateless
# 从 torch.testing._internal.common_device_type 导入 instantiate_device_type_tests 函数
from torch.testing._internal.common_device_type import instantiate_device_type_tests
# 从 torch.testing._internal.common_methods_invocations 导入 op_db, skip, xfail, skipOps 函数
from torch.testing._internal.common_methods_invocations import op_db, skip, xfail, skipOps
# 从 torch._subclasses.fake_tensor 导入 DynamicOutputShapeException, DataDependentOutputException, FakeTensorMode 类
from torch._subclasses.fake_tensor import DynamicOutputShapeException, DataDependentOutputException, FakeTensorMode
# 从 torch._subclasses.functional_tensor 导入 FunctionalTensor, FunctionalTensorMode 类
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode
# 从 torch._decomp 导入 decomposition_table
from torch._decomp import decomposition_table
# 从 torch.fx.experimental.symbolic_shapes 导入 eval_guards, bind_symbols, fx_placeholder_vals, fx_placeholder_targets,
# guard_int, GuardOnDataDependentSymNode 函数/类
from torch.fx.experimental.symbolic_shapes import (
    eval_guards, bind_symbols, fx_placeholder_vals, fx_placeholder_targets,
    guard_int, GuardOnDataDependentSymNode
)
# 从 torch.testing._internal.custom_op_db 导入 custom_op_db 函数
from torch.testing._internal.custom_op_db import custom_op_db
# 从 torch.testing._internal.hop_db 导入 hop_db 函数
from torch.testing._internal.hop_db import hop_db
# 从 torch.testing._internal.common_device_type 导入 ops
from torch.testing._internal.common_device_type import ops
# 从 torch.testing._internal.optests 导入 optests 模块
import torch.testing._internal.optests as optests
# 从 torch._C 导入 _disabled_torch_function_impl
from torch._C import _disabled_torch_function_impl
# 从 torch.fx.experimental.proxy_tensor 导入 make_fx, DecompositionInterpreter, get_isolated_graphmodule 函数
from torch.fx.experimental.proxy_tensor import make_fx, DecompositionInterpreter, get_isolated_graphmodule
# 从 torch.utils._pytree 导入 tree_map 函数
from torch.utils._pytree import tree_map
# 从 torch.fx.passes.runtime_assert 导入 insert_deferred_runtime_asserts 函数
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts
# 从 torch 导入 nn 模块
from torch import nn
# 导入 torch._functorch.config 模块
import torch._functorch.config
# 导入 re 模块
import re

# 导入 functools 模块
import functools
# 导入 itertools 模块
import itertools

# 使用 torch.ops.aten 别名为 aten
aten = torch.ops.aten

# 判断是否有 CUDA 可用，并赋值给 HAS_CUDA
HAS_CUDA = torch.cuda.is_available()


def strip_end(s, suffix):
    # 如果传入的字符串 s 以 suffix 结尾，则返回去掉 suffix 后的字符串
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    else:
        return s


def show_guards(gm):
    # 获取所有占位符的名称，去掉末尾的 "_1" 后返回一个列表
    names = [strip_end(n, "_1") for n in fx_placeholder_targets(gm)]
    # 调用 gm 对象的 shape_env.produce_guards 方法，返回符号卫兵的字符串表示形式
    return "\n".join(
        gm.shape_env.produce_guards(fx_placeholder_vals(gm), names, _simplified=True, input_contexts=None)
    )


def process_failures():
    """
    Takes file containing failures like

    FAILED test/test_proxy_tensor.py::TestProxyTensorOpInfoCPU::test_make_fx_symbolic_exhaustive___getitem___cpu_float32 - RuntimeError: aten.size.default - couldn't find symbolic meta function/decomposition  # noqa: B950

    and processes them into a list of opinfo xfails
    """
    # 打开名为 'pytest_failures' 的文件以读取失败信息
    f = open('pytest_failures')
    # 逐行读取并去除每行两侧的空白符，形成失败列表
    failures = f.readlines()
    failures = [i.strip() for i in failures]

    # 定义处理失败字符串的函数，使用正则表达式从字符串中匹配符号追踪失败信息
    def process_failure_string(s, matcher):
        out = re.search(matcher, s)
        return out.groups()

    # 定义符号追踪匹配的正则表达式模式
    SYMBOLIC_TRACE_MATCH = r'exhaustive_(.*)_cpu.*: (.*)'
    # 对失败列表中的每个字符串应用处理函数，将符号追踪失败信息提取出来
    failures = [process_failure_string(s, SYMBOLIC_TRACE_MATCH) for s in failures]

    # 定义创建标准化名称的函数，用于创建操作的规范化名称
    def create_normalized_name(op):
        if op.variant_test_name == '':
            s = op.name
        else:
            s = f"{op.name}.{op.variant_test_name}"
        return s.replace('.', '_')

    # 使用 op_db 中的操作对象创建名称到操作信息的映射字典
    remap_opinfo = {create_normalized_name(op): (op.name, op.variant_test_name) for op in op_db}

    # 打印符号张量失败信息的起始标记
    print("symbolic_tensor_failures = {")
    # 遍历 failures 列表中的每一个元素，每个元素是一个元组 (failure, reason)
    for failure, reason in failures:
        # 打印格式化的字符串，输出格式为 "    xfail{remap_opinfo[failure]},  # {reason}"
        print(f"    xfail{remap_opinfo[failure]},  # {reason}")
    # 打印右括号，结束打印块
    print("}")
# 设置一个标志，用于指示是否成功导入了torchvision模块，默认为False
USE_TORCHVISION = False
# 尝试导入torchvision模块，如果成功则将标志设为True
try:
    import torchvision
    USE_TORCHVISION = True
# 如果导入失败，则发出警告并提供安装建议
except ImportError:
    warnings.warn("Couldn't import torchvision. Some of our tests use it, try "
                  "to install it with commands from pytorch.org, post-fixed with "
                  "`--no-deps` to avoid overwriting the pytorch installation",
                  UserWarning)

# 定义一个函数，根据输入x创建新的输入对象
def _create_new_input(x):
    # 如果x不是torch.Tensor对象，则直接返回x
    if not isinstance(x, torch.Tensor):
        return x
    # 如果x的数据类型不是torch.float，则返回x + 1
    if x.dtype != torch.float:
        return x + 1
    # 如果x是叶子张量，则返回一个与x形状相同的随机张量，要求梯度与x保持一致
    if x.is_leaf:
        return torch.rand_like(x, requires_grad=x.requires_grad)
    # 否则返回一个与x形状相同的随机张量
    else:
        return torch.rand_like(x)

"""
延迟在未包装张量上执行cos操作，直到其被使用。模拟使用的CommTensor
"""
# 定义一个新的张量类UnwrapTensor，继承自torch.Tensor
class UnwrapTensor(torch.Tensor):
    # 静态方法，创建一个新的UnwrapTensor对象
    @staticmethod
    def __new__(cls, tensor: torch.Tensor):
        # 调用父类方法创建一个包装子类，传入张量的大小、数据类型、设备、布局、梯度需求等信息
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            dtype=tensor.dtype,
            device=tensor.device,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
        )
        # 将原始张量存储在新对象的_tensor属性中
        r._tensor = tensor
        return r

    # 返回对象的字符串表示，格式为"UnwrapTensor(原始张量)"
    def __repr__(self):
        # TODO: 考虑对所有本地张量进行全局聚合，以便进行更好的调试
        return f"UnwrapTensor({self._tensor})"

    # 禁用torch_function的实现
    __torch_function__ = _disabled_torch_function_impl

    # 类方法，用于处理torch调度功能
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # 定义一个解包函数unwrap，如果输入是UnwrapTensor对象，则返回其存储的原始张量执行cos操作的结果
        def unwrap(e):
            ret = e
            if isinstance(e, UnwrapTensor):
                ret = e._tensor.cos()
            return ret
        
        # 对输入的args和kwargs应用解包函数unwrap
        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)
        # 调用原始的func函数，并传入解包后的args和kwargs参数
        return func(*args, **kwargs)

# 测试类TestGenericProxyTensor，继承自TestCase
class TestGenericProxyTensor(TestCase):
    # 警告：如果您的输入包含索引张量，请勿使用此函数
    def _test(self, f, inps):
        # 使用make_fx函数以指定的跟踪模式创建fx_f函数对象，并传入输入inps进行调用
        fx_f = make_fx(f, tracing_mode=self.tracing_mode)(*inps)
        # 对输入inps中的每个元素应用_create_new_input函数，生成新的输入new_inps
        new_inps = tree_map(_create_new_input, inps)
        # 分别用新输入new_inps和原始输入inps调用fx_f和f函数，比较其结果是否相等
        r1 = fx_f(*new_inps)
        r2 = f(*new_inps)
        self.assertEqual(r1, r2)

    # 测试函数test_pre_dispatch_mode_stack
    def test_pre_dispatch_mode_stack(self):
        # 定义函数f，接受一个参数a，创建张量b并返回a与b的矩阵乘积
        def f(a):
            b = torch.ones(4, 4)
            return torch.matmul(a, b)
        # 期望在跟踪中看到matmul操作，不会将其分解为mm操作。
        # 同时torch.ones()不会出现在跟踪中。
        # 这是预期行为：ones()永远不会分派到Autograd分派键，因此我们的模式永远不会看到它，
        # 它直接到达BackendSelect键。
        inp = torch.ones(4, 4)
        # 测试使用make_fx(pre_dispatch=True)清除缓存是否正确
        from torch._dispatch.python import enable_python_dispatcher
        with enable_python_dispatcher():
            out1 = f(inp)
        # 使用make_fx函数以预调度模式创建fx_g函数对象，并传入输入inp进行调用
        fx_g = make_fx(f, pre_dispatch=True)(inp)
        # 断言fx_g的代码与期望的内联代码一致
        self.assertExpectedInline(fx_g.code.strip(), """\
def forward(self, a_1):
    ones = torch.ops.aten.ones.default([4, 4], device = device(type='cpu'), pin_memory = False)
"""
    # 使用 torch.ops.aten.matmul.default 进行矩阵乘法运算，并将结果赋给 matmul
    matmul = torch.ops.aten.matmul.default(a_1, ones);  a_1 = ones = None
    # 返回 matmul 作为函数的结果
    return matmul



    # 定义测试函数 test_pre_dispatch_linear
    def test_pre_dispatch_linear(self):
        # 定义函数 f，调用 torch.nn.functional.linear 对输入进行线性变换
        def f(a, b, c):
            return torch.nn.functional.linear(a, b, c)
        # 创建输入张量 a, b, c，每个元素均为 1
        a = torch.ones(4, 4)
        b = torch.ones(4, 4)
        c = torch.ones(4)
        # 使用 make_fx 创建带有预调度功能的 fx_g 函数
        fx_g = make_fx(f, pre_dispatch=True)(a, b, c)
        # 分别计算 f 和 fx_g 的输出
        out1 = f(a, b, c)
        out2 = fx_g(a, b, c)
        # 断言两个输出张量相等
        self.assertEqual(out1, out2)



    # 定义测试函数 test_pre_dispatch_no_grad
    def test_pre_dispatch_no_grad(self):
        # 定义函数 f，对输入张量 a 进行一系列数学操作
        def f(a):
            b = a.sin()
            # 关闭梯度追踪
            torch.set_grad_enabled(False)
            c = b.cos()
            # 开启梯度追踪
            torch.set_grad_enabled(True)
            return b + c.sin()
        # 创建需要梯度的输入张量 a1
        a1 = torch.randn(4, requires_grad=True)
        # 创建 a1 的克隆张量 a2，并且确保它也需要梯度
        a2 = a1.clone().detach().requires_grad_(True)
        # 创建 a1 的克隆张量 a_tmp，并且确保它也需要梯度
        a_tmp = a1.clone().detach().requires_grad_(True)
        # 使用 make_fx 创建带有预调度功能的 fx_g 函数
        fx_g = make_fx(f, pre_dispatch=True)(a_tmp)
        # 计算 f 和 fx_g 的输出
        out1 = f(a1)
        out2 = fx_g(a2)
        # 断言两个输出张量相等
        self.assertEqual(out1, out2)
        # 对 out1 和 out2 的 sum 求梯度
        out1.sum().backward()
        out2.sum().backward()
        # 断言 a1 和 a2 的梯度相等
        self.assertEqual(a1.grad, a2.grad)



    # 定义测试函数 test_make_fx_simple
    def test_make_fx_simple(self):
        # 定义简单的函数 f，对输入张量 x 计算 sin 函数
        def f(x):
            return torch.sin(x)
        # 调用辅助函数 _test 来测试函数 f
        self._test(f, (torch.randn(3),))



    # 定义测试函数 test_scalar_device，测试在指定设备上的张量运算
    def test_scalar_device(self, device='cpu'):
        # 定义函数 f，对输入张量 a 和标量 b 进行加法运算
        def f(a, b):
            return a + b
        # 调用辅助函数 _test 来测试函数 f，在指定设备上进行计算
        self._test(f, [torch.randn(3, device=device), torch.tensor(5)])



    # 测试函数 test_empty_like_doesnt_burn_in_defaults，验证 torch.empty_like 在保留默认设置时的行为
    def test_empty_like_doesnt_burn_in_defaults(self):
        # 定义函数 f，对输入张量 x 返回一个与 x 相同形状的空张量
        def f(x):
            return torch.empty_like(x)
        # 使用 make_fx 创建 f 的功能等价版本 out
        out = make_fx(f)(torch.randn(3))
        # 断言 out 的生成的代码与预期的一致
        self.assertExpectedInline(out.code.strip(), """\
def test_proxy_tensor_mode_with_decomp_table_preserves_proxy(self):
    # 定义函数 f，接受参数 x，创建一个与 x 大小相同的零张量 y，并将 x 的内容复制到 y 中，最后返回 y
    def f(x):
        y = x.new_zeros(x.size())
        y.copy_(x)
        return y

    # 定义 _new_zeros_decomp 函数，接受输入 inp、大小 size、数据类型 dtype、布局 layout、设备 device 和 pin_memory 参数，
    # 返回一个指定大小、指定数据类型和设备的零张量
    def _new_zeros_decomp(inp, size, dtype=None, layout=None, device=None, pin_memory=None):
        return torch.zeros(size, dtype=inp.dtype, device=inp.device)

    # 将 torch.ops.aten.new_zeros.default 映射到 _new_zeros_decomp 函数，构成函数工厂映射表 factory_func_decomp
    factory_func_decomp = {torch.ops.aten.new_zeros.default: _new_zeros_decomp}

    # 当 new_zeros() 函数分解为 torch.zero() 函数时，预期 ProxyTensorMode 仍然保持（可重入）启用状态，
    # 以便 `torch.zero()` 调用返回一个代理张量
    # out = make_fx(f, decomposition_table=factory_func_decomp)(torch.ones(2))
    # self.assertExpectedInline(out.code, """\



def forward(self, x_1):
    # 使用 torch.ops.aten.zeros.default 创建一个包含两个元素的浮点型张量 zeros，在 CPU 设备上执行
    zeros = torch.ops.aten.zeros.default([2], dtype=torch.float32, device=device(type='cpu'), pin_memory=False)
    # 使用 torch.ops.aten.copy_.default 将 zeros 复制到 x_1，然后将 zeros 和 x_1 置为 None
    copy_ = torch.ops.aten.copy_.default(zeros, x_1);  zeros = x_1 = None
    # 返回复制后的张量
    return copy_
    """

def test_make_fx_reentrant_dispatch(self):
    # 定义函数 f，接受参数 x，返回 torch.ops.aten.norm.Scalar(x, 2.0) 的结果
    def f(x):
        return torch.ops.aten.norm.Scalar(x, 2.0)

    # 定义 norm_decomp 函数，接受参数 x 和 p（默认为 2.0），如果 p 不等于 2.0，则抛出 RuntimeError
    # 否则返回 torch.sqrt(torch.sum(torch.square(x))) 的结果
    def norm_decomp(x, p=2.0):
        if p != 2.0:
            raise RuntimeError("can't handle with p != 2")
        return torch.sqrt(torch.sum(torch.square(x)))

    # 构建映射表 decomp，将 torch.ops.aten.norm.Scalar 映射到 norm_decomp 函数
    decomp = {torch.ops.aten.norm.Scalar: norm_decomp}

    # 使用 make_fx 函数对 f 进行函数合成，使用 decomp 映射表，根据 self.tracing_mode 进行跟踪模式的选择
    traced = make_fx(f, decomposition_table=decomp, tracing_mode=self.tracing_mode)(torch.rand(3))

    # 遍历 traced 中的所有图节点
    for n in traced.graph.nodes:
        # 断言图节点的目标中不包含字符串 "square"
        self.assertTrue("square" not in str(n.target))
        # 断言图节点的目标中不包含字符串 "norm"
        self.assertTrue("norm" not in str(n.target))

@unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
def test_resnet18_backward_trace(self):
    # 创建 torchvision.models.resnet18() 模型，并赋值给 mod
    mod = torchvision.models.resnet18()

    # 定义函数 f，接受输入 x、params 和 buffers
    def f(x, params, buffers):
        # 遍历 params 中的所有值，将其梯度设为 None
        for p in params.values():
            p.grad = None
        # 计算损失，使用 torch.func.functional_call 调用 mod 模型的函数式 API，并将 params 和 buffers 传递进去
        # 损失值为 (x,) 的和
        loss = torch.func.functional_call(mod, {**params, **buffers}, (x,)).sum()
        # 执行反向传播
        loss.backward()
        # 返回所有参数的梯度列表
        return [p.grad for p in params.values()]

    # 创建输入张量 inp，形状为 (3, 3, 250, 250)
    inp = torch.randn(3, 3, 250, 250)
    # 调用 self._test 方法，传入函数 f 和参数列表 [inp, dict(mod.named_parameters()), dict(mod.named_buffers())]
    self._test(f, [inp, dict(mod.named_parameters()), dict(mod.named_buffers())])

def test_varargs(self):
    # 定义函数 f，接受任意数量的参数 args，并返回所有参数的总和
    def f(*args):
        return sum(args)

    # 调用 self._test 方法，传入函数 f 和参数列表 [torch.randn(2), torch.randn(2)]
    self._test(f, [torch.randn(2), torch.randn(2)])
    # 定义测试函数 test_proxy_tensor，用于测试梯度计算和反向传播
    def test_proxy_tensor(self):
        # 定义计算梯度的函数 f_grad，返回输入张量的余弦函数嵌套求和的梯度
        def f_grad(x):
            val = x.cos().cos().sum()
            return torch.autograd.grad(val, x)

        # 定义执行反向传播的函数 f_backward，对输入张量执行余弦函数嵌套求和并反向传播梯度
        def f_backward(x):
            val = x.cos().cos().sum()
            val.backward()
            return x.grad

        # 对定义的两个函数 f_grad 和 f_backward 进行测试
        for f in [f_grad, f_backward]:
            self._test(f, [torch.randn(3, requires_grad=True)])

    # 定义测试函数 test_pickle_issue89626，测试对象序列化与反序列化的问题
    def test_pickle_issue89626(self):
        # 导入 pickle 库
        import pickle
        # 创建一个随机张量 x
        x = torch.randn(2)
        # 使用指定的跟踪模式调用 make_fx 函数，对 x 执行函数调用并返回结果
        make_fx(lambda x: x * 2, tracing_mode=self.tracing_mode)(x)
        # 对张量 x 进行 pickle 序列化
        pickle.dumps(x)

    # 定义测试函数 test_inplace_metadata，测试张量操作中的元数据和形状变化
    def test_inplace_metadata(self):
        # 定义函数 f，复制输入张量并在其最后一维添加维度，确保维度变化正确
        def f(x):
            x = x.clone()
            x.unsqueeze_(-1)
            assert x.shape[-1] == 1
            return x

        # 对函数 f 进行测试
        self._test(f, [torch.randn(5)])

    # 定义测试函数 test_mode_tracing_factory_function，测试跟踪工厂函数
    def test_mode_tracing_factory_function(self):
        # 定义函数 f，对输入张量执行加法并添加随机噪声
        def f(x):
            return x + torch.randn(x.shape)

        # 使用指定的跟踪模式调用 make_fx 函数，对函数 f 进行跟踪并返回结果
        traced = make_fx(f, tracing_mode=self.tracing_mode)(torch.randn(3))
        # 断言在跟踪的图中存在至少一个节点的目标为 aten.randn.default
        self.assertTrue(
            any(
                node.target == aten.randn.default
                for node in traced.graph.nodes
            )
        )

    # 定义测试函数 test_pre_dispatch_functionalization，测试功能化张量模式的预分派
    def test_pre_dispatch_functionalization(self):
        # 定义函数 f，创建一个功能化张量模式对象，执行一系列操作并返回结果
        def f(x):
            a = FunctionalTensorMode(pre_dispatch=True)
            with a:
                x_unwrapped = FunctionalTensor.to_functional(x)
                y = torch.matmul(x_unwrapped, x_unwrapped)
                y = y + x_unwrapped
                y.mul_(5)
                y_unwrapped = torch._from_functional_tensor(y.elem)
                return y_unwrapped

        # 导入启用 Python 分派器的函数
        from torch._dispatch.python import enable_python_dispatcher

        # 使用启用 Python 分派器的上下文环境
        with enable_python_dispatcher():
            # 创建一个随机输入张量
            inp = torch.randn(4, 4)
            # 使用指定的预分派标志调用 make_fx 函数，对函数 f 进行跟踪并返回结果
            gm = make_fx(f, pre_dispatch=True)(inp)

        # 断言生成的 gm.code 的内联版本与预期结果匹配
        # TODO actually not decompose
        self.assertExpectedInline(gm.code.strip(), """
def forward(self, x_1):
    # 执行矩阵乘法操作，并存储结果
    matmul = torch.ops.aten.matmul.default(x_1, x_1)
    # 执行张量加法操作，将矩阵乘法结果与输入张量相加，并释放 matmul 和 x_1
    add = torch.ops.aten.add.Tensor(matmul, x_1);  matmul = x_1 = None
    # 执行张量乘法操作，将加法结果乘以常数 5，并释放 add
    mul = torch.ops.aten.mul.Tensor(add, 5);  add = None
    # 返回乘法结果
    return mul
    def test_constant_proxy_tensor_mut(self):
        def f():
            # 创建一个包含浮点数值 1 的张量
            val = torch.tensor(float(1))
            # 将张量的值增加 2（原地操作）
            val.add_(2)
            # 返回一个大小为 (100, 100) 的张量，每个元素都是 val 的值
            return torch.full((100, 100), val)

        # 使用函数 make_fx 创建一个 FX 函数，并执行它以获得结果 g
        g = make_fx(f, tracing_mode=self.tracing_mode)()
        # 断言 g 的输出与函数 f 的输出相等
        self.assertEqual(g(), f())
        # 再次断言，以确保在 g 图中未发生共享状态的突变
        self.assertEqual(g(), f())

    def test_constant_unbind(self):
        def f():
            # 创建一个包含值为 2 的张量
            val = torch.tensor([2])
            # 解绑张量 val 的唯一维度，并返回结果作为标量
            r, = torch.unbind(val, 0)
            return r.item()

        g = make_fx(f, tracing_mode=self.tracing_mode)()
        # 断言 g 的输出与函数 f 的输出相等
        self.assertEqual(g(), f())

    def test_constant_blowup(self):
        def f():
            # 创建一个包含值为 2 的张量
            val = torch.tensor([2])
            # 将张量 val 在维度0上重复1000次，导致内存耗尽的异常
            blowup = val.repeat(1000)
            return bool(blowup.sum().item() == 2)

        def test_f():
            # 创建 FX 函数，并在运行时期望引发 RuntimeError 异常，其消息包含 "data-dependent"
            make_fx(f, tracing_mode=self.tracing_mode)()

        # 断言 test_f 函数运行时引发了预期的 RuntimeError 异常
        self.assertRaisesRegex(RuntimeError, "data-dependent", test_f)

    def test_constant_random(self):
        def f():
            # 创建一个包含值为 2.0 的张量
            val = torch.tensor([2.0])
            # 对张量 val 执行正态分布随机初始化
            val.normal_()
            # 返回一个布尔值，表示张量的值是否等于 2.1
            return bool(val.item() == 2.1)

        def test_f():
            # 创建 FX 函数，并在运行时期望引发 RuntimeError 异常，其消息包含 "data-dependent"
            make_fx(f, tracing_mode=self.tracing_mode)()

        # 断言 test_f 函数运行时引发了预期的 RuntimeError 异常
        self.assertRaisesRegex(RuntimeError, "data-dependent", test_f)

    def test_decomposition_interpreter(self):
        def fn(x):
            # 使用 torch.nn.functional.silu 函数对输入 x 执行 SiLU 激活函数
            return torch.nn.functional.silu(x)

        x = torch.rand((4, 4))
        # 使用 make_fx 创建 FX 模块，并应用函数 fn 到输入 x
        fx_module = make_fx(fn, tracing_mode=self.tracing_mode, decomposition_table=None)(x)

        found_silu = False
        # 检查 FX 模块的计算图中是否包含 torch.ops.aten.silu 或 torch.ops.aten.silu.default 操作
        for n in fx_module.graph.nodes:
            if n.target == torch.ops.aten.silu or n.target == torch.ops.aten.silu.default:
                found_silu = True

        # 断言在计算图中找到了 torch.nn.functional.silu 操作
        self.assertTrue(found_silu)

        new_graph = torch.fx.Graph()
        # 创建一个 SiLU 操作的分解表
        silu_decomp_table = {torch.ops.aten.silu.default: decomposition_table[torch.ops.aten.silu.default]}
        # 使用 DecompositionInterpreter 将 FX 模块分解为新图 new_graph
        DecompositionInterpreter(
            fx_module,
            new_graph=new_graph,
            decomposition_table=silu_decomp_table,
        ).run(x)

        # 创建分解后的 FX 模块
        decomposed_module = torch.fx.GraphModule(fx_module, new_graph)

        # 检查分解后的计算图中不再包含 torch.ops.aten.silu 或 torch.ops.aten.silu.default 操作
        for n in decomposed_module.graph.nodes:
            self.assertTrue(n.target != torch.ops.aten.silu)
            self.assertTrue(n.target != torch.ops.aten.silu.default)

        # 断言未分解前后的模块在相同输入下产生相同的输出
        self.assertEqual(fx_module(x), decomposed_module(x))
    # 定义一个测试函数，用于测试 make_fx 函数生成的模型的前向和反向传播
    def test_make_fx_model_fwd_bwd(self):
        # 定义一个简单的神经网络模型类 Foo
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            # 前向传播函数，对输入数据进行线性变换后再应用 ReLU 激活函数
            def forward(self, x):
                return self.linear(x).relu()

        # 创建 Foo 类的实例 model
        model = Foo()

        # 定义一个函数 f，接受输入 x 和模型参数 params，执行函数调用并计算梯度
        def f(x, params):
            # 调用 torch.func.functional_call 函数，对模型和参数进行处理并求和
            out = torch.func.functional_call(model, params, x).sum()
            # 执行反向传播
            out.backward()
            # 返回模型参数的值列表
            return list(params.values())

        # 生成一个随机输入张量 input，需要计算梯度
        input = torch.randn(3, 5, requires_grad=True)
        # 获取模型的所有参数并以字典形式存储在 params 中
        params = dict(model.named_parameters())
        # 使用 make_fx 函数对 f 进行处理，获得一个新的函数 fx_f，采用指定的追踪模式
        fx_f = make_fx(f, tracing_mode=self.tracing_mode)(input, params)

        # 断言：使用 fx_f 和 f 计算的第一个返回值应当相等，由于顺序可能变化，使用 allclose 进行比较
        self.assertTrue(
            torch.allclose(fx_f(input, params)[0], f(input, params)[0])
            or
            torch.allclose(fx_f(input, params)[0], f(input, params)[1])
        )
        # 断言：使用 fx_f 和 f 计算的第二个返回值应当相等，由于顺序可能变化，使用 allclose 进行比较
        self.assertTrue(
            torch.allclose(fx_f(input, params)[1], f(input, params)[0])
            or
            torch.allclose(fx_f(input, params)[1], f(input, params)[1])
        )

    # 定义一个测试函数，用于测试 make_fx 函数对具有重复参数的模型的处理
    def test_make_fx_model_double_param(self):
        # 定义一个简单的神经网络模型类 Emformer
        class Emformer(torch.nn.Module):
            def __init__(
                self,
                input_dim: int = 256,
            ) -> None:
                super().__init__()
                self.layer_norm = torch.nn.LayerNorm(input_dim)

            # 前向传播函数，使用 layer_norm 对输入进行归一化处理
            def forward(mod_self, x):  # noqa: B902
                # 断言：确保 layer_norm 的权重是 torch.Tensor 类型
                self.assertTrue(isinstance(mod_self.layer_norm.weight, torch.Tensor))
                y = mod_self.layer_norm(x)
                # 断言：确保 layer_norm 的权重仍然是 torch.Tensor 类型
                self.assertTrue(isinstance(mod_self.layer_norm.weight, torch.Tensor))
                z = mod_self.layer_norm(y)
                return z

        # 使用 make_fx 函数处理 Emformer 类的实例，生成一个图模型 gm
        gm = make_fx(Emformer())(torch.randn(16, 1, 256))
        # 获取 gm 图中所有调用函数节点的目标集合
        ops = {n.target for n in gm.graph.nodes if n.op == 'call_function'}
        # 断言：调用函数节点的目标数量应当为 2
        self.assertEqual(len(ops), 2)
    def test_trace_subclasses(self):
        # 定义函数 f1，接受一个参数 x，将其解包为 UnwrapTensor 类型，然后对 x 执行乘法运算并返回结果
        def f1(x):
            x = UnwrapTensor(x)
            y = x * 2
            return y

        # 定义函数 f2，接受一个参数 x，将其解包为 UnwrapTensor 类型，然后将 x 与其自身的乘积相乘并返回结果
        def f2(x):
            wrapped = UnwrapTensor(x)
            y = x * wrapped
            return y

        # 构造输入列表 inp，包含一个形状为 (5,) 的随机张量
        inp = [torch.randn(5)]
        # 调用 self._test 方法，分别对 f1 和 f2 进行测试
        self._test(f1, inp)
        self._test(f2, inp)

    def test_partial_decomp(self):
        # 定义函数 f，接受三个参数 a, b, c，分别执行 torch.addmm 操作，并返回结果的和
        def f(a, b, c):
            x = torch.addmm(a, b, c)
            y = torch.addmm(a, b, c, beta=2, alpha=1)
            return x + y
        
        # 构造输入列表 inps，包含三个形状为 (5, 5) 的随机张量
        inps = [torch.randn(5, 5), torch.randn(5, 5), torch.randn(5, 5)]
        # 使用 make_fx 函数对 f 进行功能提取，并将 inps 中的张量作为输入
        fx_g = make_fx(f)(*inps)

        # 定义函数 addmm，对 torch.addmm 进行部分分解，根据 beta 和 alpha 参数的不同返回不同结果
        def addmm(a, b, c, beta=1, alpha=1):
            if beta == 1 and alpha == 1:
                return NotImplemented
            return beta * a + alpha * (b @ c)

        # 使用指定的 decomposition_table 对 f 进行功能提取，将 inps 中的张量作为输入
        decomposed_fx = make_fx(f, decomposition_table={aten.addmm.default: addmm})(*inps)

        # 断言 fx_g 和 decomposed_fx 的输出结果相等
        self.assertEqual(fx_g(*inps), decomposed_fx(*inps))
        # 断言 fx_g 中使用了两次 torch.addmm 操作
        self.assertEqual(len([n for n in fx_g.graph.nodes if n.target == aten.addmm.default]), 2)
        # 断言 decomposed_fx 中使用了一次 torch.addmm 操作
        self.assertEqual(len([n for n in decomposed_fx.graph.nodes if n.target == aten.addmm.default]), 1)
    def test_decomp_of_capture(self):
        val = torch.randn(5)  # 创建一个包含5个随机数的张量 val

        def f(x):
            return x.t() + val.t()  # 对输入张量 x 和 val 进行转置并相加的操作

        def nop(x):
            return x.cos()  # 返回输入张量 x 的余弦值

        traced = make_fx(f, decomposition_table={torch.ops.aten.t.default: nop})(torch.randn(5))
        self.assertEqual(len([n for n in traced.graph.nodes if n.target == torch.ops.aten.t.default]), 0)
        # 断言转换后的图中不再包含 torch.ops.aten.t.default 操作

    @unittest.skipIf(not HAS_CUDA, 'CUDA-only test')  # 如果没有 CUDA 支持则跳过测试
    def test_amp_cache(self):
        layer = torch.nn.Conv2d(3, 3, 3).cuda()  # 创建一个在 CUDA 上运行的 3x3 2D 卷积层

        def f(x, w):
            return torch.nn.functional.conv2d(x, w, stride=layer.stride)  # 使用给定的 stride 对输入 x 进行卷积操作

        inp = torch.randn(4, 3, 10, 10, device='cuda')  # 在 CUDA 设备上创建一个形状为 (4, 3, 10, 10) 的随机张量
        with torch.autocast('cuda'):
            out_graph = make_fx(f)(inp, layer.weight).graph  # 使用 autocast 模式下的 make_fx 对 f 进行转换并记录其图结构
            out_graph2 = make_fx(f)(inp, layer.weight).graph  # 再次使用 make_fx 对 f 进行转换并记录其图结构

        self.assertEqual(len(out_graph.nodes), len(out_graph2.nodes))  # 断言两次转换后的图结构节点数相同
        for a, b in zip(out_graph.nodes, out_graph2.nodes):
            self.assertEqual(a.op, b.op)  # 断言两次转换后的图结构节点的操作类型相同

    def test_strides(self):
        def f(x):
            self.assertTrue(x.is_contiguous())  # 断言输入张量 x 是连续的
            self.assertFalse(x.is_contiguous(memory_format=torch.channels_last))  # 断言输入张量 x 不是 channels_last 内存格式
            x = x.permute(0, 3, 1, 2)  # 对输入张量 x 进行维度置换操作
            self.assertFalse(x.is_contiguous())  # 断言置换后的张量 x 不再是连续的
            self.assertTrue(x.is_contiguous(memory_format=torch.channels_last))  # 断言置换后的张量 x 是 channels_last 内存格式的
            return x

        make_fx(f)(torch.randn(2, 3, 4, 5))  # 对形状为 (2, 3, 4, 5) 的随机张量应用 make_fx 转换

        def f(x):
            self.assertTrue(x.is_contiguous())  # 断言输入张量 x 是连续的
            y = x[:, 1]  # 提取输入张量 x 的第二列
            self.assertFalse(y.is_contiguous())  # 断言提取的张量 y 不是连续的
            y = x[:, ::2]  # 提取输入张量 x 的奇数列
            self.assertFalse(y.is_contiguous())  # 断言提取的张量 y 不是连续的
            return x.cos()  # 返回输入张量 x 的余弦值

        make_fx(f)(torch.randn(2, 3, 4, 5))  # 对形状为 (2, 3, 4, 5) 的随机张量应用 make_fx 转换

    def test_pr_86917(self):
        # Tests the issue brought up here https://github.com/pytorch/pytorch/pull/86917#issuecomment-1283155344
        def f(a, b):
            return torch.ops.aten.nll_loss_forward(a, b, None, 1, 10)
        # 调用 torch.ops.aten.nll_loss_forward 函数计算输入张量 a, b 的负对数似然损失

        self._test(f, [torch.randn(1, 10), torch.zeros(1, dtype=torch.long)])
        # 调用自定义的 _test 方法对函数 f 进行测试，传入参数为随机张量和全零张量
class TestGenericProxyTensorReal(TestGenericProxyTensor):
    # 继承自 TestGenericProxyTensor，设置跟踪模式为 "real"
    tracing_mode = "real"


class TestGenericProxyTensorFake(TestGenericProxyTensor):
    # 继承自 TestGenericProxyTensor，设置跟踪模式为 "fake"
    tracing_mode = "fake"


class TestGenericProxyTensorSymbolic(TestGenericProxyTensor):
    # 继承自 TestGenericProxyTensor，设置跟踪模式为 "symbolic"
    tracing_mode = "symbolic"


del TestGenericProxyTensor
# 删除 TestGenericProxyTensor 类定义


class TestRealProxyTensor(TestCase):
    def test_error_on_data_dependent_ops(self):
        def f():
            # 创建两个随机张量 x 和 y
            x = torch.randn([])
            y = torch.randn([])
            # 断言 x * y 与 y * x 的值在相对误差小的情况下相等
            assert torch.allclose(x * y, y * x)
            # 将张量 x 和 y 转换为浮点数
            z = float(x)
            z2 = float(y)

        # 调用 make_fx 函数对函数 f 进行功能测试，关闭数据相关操作错误检查
        make_fx(f, _error_on_data_dependent_ops=False)()
        # 调用 make_fx 函数对函数 f 进行功能测试，开启预分发，关闭数据相关操作错误检查
        make_fx(f, pre_dispatch=True, _error_on_data_dependent_ops=False)()

class TestFakeProxyTensor(TestCase):
    def test_issue82547(self):
        # 创建一个 nn.Parameter 类型的张量 x
        x = nn.Parameter(torch.randn(3, 3))

        def f():
            # 调用 torch.ops.aten.t.default 对张量 x 进行操作
            return torch.ops.aten.t.default(x)
        # 断言调用 make_fx 函数对函数 f 进行功能测试时抛出指定异常
        self.assertRaisesRegex(Exception, "Please convert all Tensors", lambda: make_fx(f, tracing_mode="fake")())

        class A(torch.Tensor):
            pass

        # 创建类 A 的一个实例 x，传入张量数据
        x = A(torch.randn(3, 3))
        # 断言调用 make_fx 函数对函数 f 进行功能测试时抛出指定异常
        self.assertRaisesRegex(TypeError, "Multiple dispatch failed", lambda: make_fx(f, tracing_mode="fake")())

    def test_use_fake_and_tensor(self):
        def f(x, y):
            # 创建一个张量 z，包含两个浮点数
            z = torch.tensor([2.0, 3.0])
            return x + y + z

        # 调用 make_fx 函数对函数 f 进行功能测试，使用 "fake" 跟踪模式
        g = make_fx(f, tracing_mode="fake")(torch.randn(2), torch.randn(2))
        x, y = torch.randn(2), torch.randn(2)
        # 断言 g(x, y) 的结果与 f(x, y) 的结果相等
        self.assertEqual(g(x, y), f(x, y))

    def test_free_fake(self):
        def f(x):
            # 返回 x 与全局变量 y 的加法结果
            return torch.add(x, y)

        # 使用 FakeTensorMode 上下文管理器
        with FakeTensorMode() as fake_mode:
            # 创建一个张量 y，包含两个随机数
            y = torch.randn(2)
            # 调用 make_fx 函数对函数 f 进行功能测试，使用 "real" 跟踪模式
            make_fx(f, tracing_mode="real")(torch.randn(2))
    def test_fused_adam(self):
        # 在测试中使用 PyTorch 的自定义优化器 fused_adam
        # 解决 GitHub 上的问题 https://github.com/pytorch/pytorch/issues/99356
        # 创建一组随机参数、梯度、期望平均值、期望平方平均值、最大期望平方平均值和状态步数
        params = [torch.randn(10, 10) for _ in range(10)]
        grads = [torch.randn(10, 10) for _ in range(10)]
        exp_avgs = [torch.randn(10, 10) for _ in range(10)]
        exp_avg_sqs = [torch.randn(10, 10) for _ in range(10)]
        max_exp_avg_sqs = [torch.randn(10, 10) for _ in range(10)]
        state_steps = [torch.tensor(0) for _ in range(10)]

        def fused_adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps):
            # 调用 PyTorch 的 _fused_adam 操作
            # 执行优化步骤，返回更新后的参数、中间变量和优化状态
            (new_params, _, _, _, _) = aten._fused_adam.default(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                lr=0.1,
                beta1=0.9,
                beta2=0.999,
                weight_decay=0.01,
                eps=1e-8,
                amsgrad=False,
                maximize=False,
            )

            # 将更新后的参数复制回原参数列表
            for p, new_p in zip(params, new_params):
                p.copy_(new_p)

            return params

        # 使用 fake 模式创建函数的 TorchScript 图形表示
        gm = make_fx(fused_adam, tracing_mode='fake')(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
        )

        # 确保生成的 TorchScript 图形节点包含所需的元数据
        ensure_ops_have_val = [aten._fused_adam.default, operator.getitem]
        for n in gm.graph.nodes:
            if n.op == "call_function" and n.target in ensure_ops_have_val:
                # 断言每个调用节点的元数据中包含 'val' 键
                self.assertIn('val', n.meta)

    def test_alias(self):
        def f(x):
            # 调用 PyTorch 的 aten.alias 操作
            return torch.ops.aten.alias(x)

        # 在 fake 模式下创建函数 f 的 TorchScript 表示，并获取其代码的字符串表示
        r = str(make_fx(f, tracing_mode="fake")(torch.randn(2)).code).strip()
        # NB: 这里不应该有 detach 调用
        # 断言函数 f 的 TorchScript 表示与预期的内联代码匹配
        self.assertExpectedInline(r, """\
# 定义一个方法 forward，接收参数 self 和 x_1
def forward(self, x_1):
    # 调用 torch.ops.aten.alias.default 方法，传入参数 x_1，并将返回值赋给 alias
    alias = torch.ops.aten.alias.default(x_1);  x_1 = None
    # 返回 alias
    return alias

# 定义一个测试方法 test_meta
def test_meta(self):
    # 定义内部函数 f，接收参数 x
    def f(x):
        # 计算 x 的余弦
        a = x.cos()
        # 计算 a 沿第 0 维的方差和均值，并将结果赋给 b
        b = torch.var_mean(a, dim=0)
        # 将 b 扩展为原来的两倍，并将结果赋给 c
        c = b * 2
        # 返回 c
        return c

    # 使用 make_fx 方法，以 "fake" 模式跟踪函数 f 的运行结果，并传入参数 torch.randn(5, 5)，将结果赋给 out
    out = make_fx(f, tracing_mode="fake")(torch.randn(5, 5))
    # 遍历 out 图中的节点
    for n in out.graph.nodes:
        # 如果节点的操作为 'output'，则继续下一次循环
        if n.op == 'output':
            continue
        # 断言节点的元数据中包含 'val' 键
        self.assertTrue('val' in n.meta)

# 定义一个函数 _get_node，接收参数 fx_g 和 cond
def _get_node(fx_g, cond):
    # 遍历 fx_g 图中的节点
    for n in fx_g.graph.nodes:
        # 如果 cond(n) 返回 True，则返回当前节点 n
        if cond(n):
            return n
    # 如果未找到符合条件的节点，则抛出 AssertionError
    raise AssertionError

# 定义一个函数 _get_free_symbols，接收参数 shape_env
def _get_free_symbols(shape_env):
    # 获取 shape_env 中所有变量名的元组
    vars = tuple(shape_env.var_to_val.keys())
    # 计算不在 shape_env.replacements 中的变量数目，并返回该数目
    return len([var for var in vars if var not in shape_env.replacements])

# 定义一个函数 _trace，接收参数 f 和 args
def _trace(f, *args):
    # 使用 torch.randn 生成与 args 中每个参数形状对应的随机张量，并将其存储在列表 inps 中
    inps = [torch.randn(arg) for arg in args]
    # 使用 make_fx 方法，以 "symbolic" 模式跟踪函数 f 的运行结果，并传入参数 inps，将结果赋给 traced_f
    return make_fx(f, tracing_mode="symbolic")(*inps)

# 定义一个测试类 TestSymbolicTracing
class TestSymbolicTracing(TestCase):
    # 定义一个方法 _test_dynamic，接收参数 fn、trace_inputs、test_inputs 和 assert_eq（默认为 True）
    def _test_dynamic(self, fn, trace_inputs, test_inputs, assert_eq=True):
        """
        Tests fn traced with trace_inputs against test_inputs
        Also returns shape env
        """
        # 使用 trace_inputs 生成与其形状对应的随机张量，并将其存储在列表 trace_inputs 中
        trace_inputs = [torch.randn(shape) for shape in trace_inputs]
        # 使用 make_fx 方法，以 "symbolic" 模式跟踪函数 fn 的运行结果，并传入参数 trace_inputs，将结果赋给 traced_f
        traced_f = make_fx(fn, tracing_mode="symbolic")(*trace_inputs)
        # 遍历 test_inputs 中的每个输入
        for input in test_inputs:
            # 使用 input 生成与其形状对应的随机张量，并将其存储在列表 input 中
            input = [torch.randn(shape) for shape in input]
            # 调用 traced_f，传入 input 作为参数，并将结果分别赋给 rx 和 ry
            rx, ry = traced_f(*input), fn(*input)
            # 如果 assert_eq 为 True，则断言 rx 等于 ry
            if assert_eq:
                self.assertEqual(rx, ry)
        # 返回 traced_f
        return traced_f

    # 定义一个测试方法 test_debug_interpreter
    def test_debug_interpreter(self):
        # 导入 torch.library 和 Library
        import torch.library
        from torch.library import Library

        # 创建名为 foo 的 Library 对象，类型为 "DEF"
        foo = Library("foo", "DEF")  # noqa: TOR901
        # 为 foo 定义一个名为 "foo" 的方法，参数为 Tensor self，返回值为 Tensor
        foo.define("foo(Tensor self) -> Tensor")

        # 定义一个使用 CPU 的 foo 实现函数 foo_cpu，参数为 x
        @torch.library.impl(foo, "foo", "CPU")
        def foo_cpu(x):
            return x.clone().T

        # 定义一个使用 Meta 的 foo 实现函数 foo_meta，参数为 x
        @torch.library.impl(foo, "foo", "Meta")
        def foo_meta(x):
            return x.clone()

        # 定义一个函数 f，参数为 x，调用 torch.ops.foo.foo.default 方法，传入参数 x，并返回结果
        def f(x):
            return torch.ops.foo.foo.default(x)

        # 使用 make_fx 方法，以 "symbolic" 模式跟踪函数 f 的运行结果，并传入参数 torch.randn(2, 2)，将结果赋给 gm
        gm = make_fx(f, tracing_mode="symbolic")(torch.randn(2, 2))
        # 导入 DebugInterpreter 类
        from torch._functorch.compilers import DebugInterpreter

        # 创建 DebugInterpreter 对象 interp，传入 gm 作为参数
        interp = DebugInterpreter(gm)

        # 断言调用 interp.run 方法时抛出 AssertionError，并且异常信息包含 "3 != 1"
        self.assertRaisesRegex(
            AssertionError, r"3 != 1",
            lambda: interp.run(torch.randn(3, 3).T),
        )

        # 断言调用 interp.run 方法时抛出 AssertionError，并且异常信息包含 "(3, 1) != (1, 3)"
        self.assertRaisesRegex(
            AssertionError, r"\(3, 1\) != \(1, 3\)",
            lambda: interp.run(torch.randn(3, 3))
        )

    # 定义一个测试方法 test_int_input
    def test_int_input(self):
        # 定义一个函数 f，参数为 x 和 y，调用 x 的 view 方法，传入参数 y，并返回结果
        def f(x, y):
            return x.view(y)

        # 调用 make_fx 方法，以 "symbolic" 模式跟踪函数 f 的运行结果，并传入参数 torch.empty(3, 4) 和 12，将结果的代码部分转换为字符串并赋给 r
        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(3, 4), 12).code).strip()
        # 断言 r 的值与预期的字符串相等
        self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    view = torch.ops.aten.view.default(x_1, [y_1]);  x_1 = y_1 = None
    return view""")
    # 定义一个嵌套函数f，接受两个参数x和y，用于调整x的大小以匹配y的大小
    def test_resize_from_zero(self):
        # 调用make_fx函数，生成一个函数对象，该函数对象通过符号化跟踪模式生成代码
        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(0), torch.empty(2)).code).strip()
        # 使用self.assertExpectedInline方法断言结果r与预期输出字符串的一致性
        self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    # 调用 Torch 的操作符 aten.sym_size.int，获取张量 y_1 在第 0 维的大小，赋给 sym_size_int，并清空 y_1 引用
    sym_size_int = torch.ops.aten.sym_size.int(y_1, 0);  y_1 = None
    # 调用 Torch 的操作符 aten.resize_.default，将张量 x_1 调整为大小为 [sym_size_int] 的张量，并清空 x_1 和 sym_size_int 引用
    resize_ = torch.ops.aten.resize_.default(x_1, [sym_size_int]);  x_1 = sym_size_int = None
    # 返回空值 None
    return None

def test_broadcast_shapes(self):
    # 定义函数 f，返回两个张量大小的广播形状
    def f(x, y):
        return torch.functional.broadcast_shapes(x.size(), y.size()[0])
    # 使用 make_fx 函数生成符号模式的函数 f，并获取其代码字符串，并清除无用空格
    r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(3, 1), torch.empty(5)).code).strip()
    # 断言生成的代码字符串符合预期格式
    self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    # 调用 Torch 的操作符 aten.sym_size.int，获取张量 x_1 在第 0 维的大小，赋给 sym_size_int，并清空 x_1 引用
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0);  x_1 = None
    # 调用 Torch 的操作符 aten.sym_size.int，获取张量 y_1 在第 0 维的大小，赋给 sym_size_int_1，并清空 y_1 引用
    sym_size_int_1 = torch.ops.aten.sym_size.int(y_1, 0);  y_1 = None
    # 返回 sym_size_int 和 sym_size_int_1 的元组
    return (sym_size_int, sym_size_int_1)""")

def test_deduped_shape(self):
    # 定义函数 f，返回两个张量大小的广播形状和一个形状为 [x.shape[0]] 的空张量
    def f(s0, s1, x, y):
        return torch.functional.broadcast_shapes(x.size(), y.size()[0]), torch.empty(x.shape[0])
    # 创建大小为 (3, 1) 和 (5,) 的张量 x 和 y
    x = torch.empty(3, 1)
    y = torch.empty(5)
    # 导入 torch.fx.experimental.symbolic_shapes 中的 ShapeEnv 类
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    # 创建 ShapeEnv 的实例 shape_env
    shape_env = ShapeEnv()
    # 使用 FakeTensorMode 进行张量模拟
    with FakeTensorMode(shape_env=shape_env, static_shapes=False) as fake_mode:
        # 使用 fake_mode.from_tensor 将 x 和 y 转换为模拟张量
        x = fake_mode.from_tensor(x)
        y = fake_mode.from_tensor(y)
        # 使用 make_fx 函数生成真实模式的函数 f，并获取其代码字符串，并清除无用空格
        r = str(make_fx(f, tracing_mode="real")(x.shape[0], y.shape[0], x, y).code).strip()
        # 断言生成的代码字符串符合预期格式
        self.assertExpectedInline(r, """\
def forward(self, s0_1, s1_1, x_1, y_1):
    # 调用 Torch 的操作符 aten.empty.memory_format，创建形状为 [s0_1] 的空张量 empty
    empty = torch.ops.aten.empty.memory_format([s0_1], device = device(type='cpu'), pin_memory = False)
    # 返回形状为 (s0_1, s1_1) 和 empty 的元组
    return ((s0_1, s1_1), empty)""")

def test_non_deduped_shape(self):
    # 定义函数 f，返回两个张量大小的广播形状和一个形状为 [x.shape[0]] 的空张量
    def f(x, y):
        return torch.functional.broadcast_shapes(x.size(), y.size()[0]), torch.empty(x.shape[0])
    # 创建大小为 (3, 1) 和 (5,) 的张量 x 和 y
    x = torch.empty(3, 1)
    y = torch.empty(5)
    # 导入 torch.fx.experimental.symbolic_shapes 中的 ShapeEnv 类
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    # 创建 ShapeEnv 的实例 shape_env
    shape_env = ShapeEnv()
    # 使用 FakeTensorMode 进行张量模拟
    with FakeTensorMode(shape_env=shape_env, static_shapes=False) as fake_mode:
        # 使用 fake_mode.from_tensor 将 x 和 y 转换为模拟张量
        x = fake_mode.from_tensor(x)
        y = fake_mode.from_tensor(y)
        # 使用 make_fx 函数生成真实模式的函数 f，并获取其代码字符串，并清除无用空格
        r = str(make_fx(f, tracing_mode="real")(x, y).code).strip()
        # 断言生成的代码字符串符合预期格式
        self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    # 调用 Torch 的操作符 aten.sym_size.int，获取张量 x_1 在第 0 维的大小，赋给 sym_size_int，并清空 x_1 引用
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0);  x_1 = None
    # 调用 Torch 的操作符 aten.empty.memory_format，创建形状为 [sym_size_int] 的空张量 empty
    empty = torch.ops.aten.empty.memory_format([sym_size_int], device = device(type='cpu'), pin_memory = False)
    # 调用 Torch 的操作符 aten.sym_size.int，获取张量 y_1 在第 0 维的大小，赋给 sym_size_int_1，并清空 y_1 引用
    sym_size_int_1 = torch.ops.aten.sym_size.int(y_1, 0);  y_1 = None
    # 返回形状为 (sym_size_int, sym_size_int_1) 和 empty 的元组
    return ((sym_size_int, sym_size_int_1), empty)""")
    def test_unary(self):
        # 定义一个函数 f，接受参数 x
        def f(x):
            # 断言 x 的第一个维度小于 20
            assert x.shape[0] < 20
            # 返回 x 的余弦值
            return x.cos()
        
        # 初始化测试输入列表
        test_inputs = []
        # 向测试输入列表添加一个元组列表 [(2, 5)]
        test_inputs.append([(2, 5)])
        # 向测试输入列表添加一个元组列表 [(6, 8)]
        test_inputs.append([(6, 8)])
        # 调用 _test_dynamic 方法，传入函数 f 和输入参数 [(3, 4)], test_inputs
        gm = self._test_dynamic(f, [(3, 4)], test_inputs)
        # 断言 eval_guards(gm, torch.randn(4, 5)) 为 True
        self.assertTrue(eval_guards(gm, torch.randn(4, 5)))
        # 断言 bind_symbols(gm, torch.randn(4, 5)) 的表达式与期望值 "{s0: 4, s1: 5}" 相等
        self.assertEqual(repr(bind_symbols(gm, torch.randn(4, 5))), "{s0: 4, s1: 5}")
        # 断言 eval_guards(gm, torch.randn(25, 5)) 为 False
        self.assertFalse(eval_guards(gm, torch.randn(25, 5)))
        # 断言 show_guards(gm) 的内联结果符合期望值 """L['x'].size()[0] <= 19"""
        self.assertExpectedInline(show_guards(gm), """L['x'].size()[0] <= 19""")

    def test_repeat_interleave(self):
        # 定义函数 f，接受参数 src_tokens 和 beam_size_src
        def f(src_tokens, beam_size_src):
            # 使用 src_tokens 的值在维度 0 上重复 beam_size_src.size(0) 次
            return src_tokens.repeat_interleave(beam_size_src.size(0), 0)

        # 初始化常量 prompt_size, vocab_size, batch_size
        prompt_size = 64
        vocab_size = 64
        batch_size = 4
        # 生成一个随机整数张量 src_tokens，形状为 (batch_size, prompt_size)
        src_tokens = torch.randint(1, vocab_size, (batch_size, prompt_size))
        # 使用 make_fx 函数，以符号跟踪模式调用函数 f，传入 src_tokens 和 torch.randn(5)
        gm = make_fx(f, tracing_mode="symbolic")(src_tokens, torch.randn(5))
        # 断言 gm.shape_env.guards 的长度为 0
        self.assertEqual(len(gm.shape_env.guards), 0)

    def test_non_symint_size_spec(self):
        # 这不是一个代理张量测试，但这是获得具有符号大小的假张量的最方便方式
        # 定义函数 f，接受参数 x
        def f(x):
            # 调用 torch._C._non_sym_sizes(x)，表示 x 具有非符号大小
            torch._C._non_sym_sizes(x)
            # 返回 x + 1
            return x + 1

        # 创建一个形状为 (2, 3) 的随机张量 x
        x = torch.randn(2, 3)
        # 使用 make_fx 函数，以符号跟踪模式调用函数 f，传入 x
        make_fx(f, tracing_mode="symbolic")(x)

    # https://github.com/pytorch/pytorch/issues/108195
    def test_symbolic_repeat_interleave(self):
        # 定义函数 f，接受参数 y 和 x
        def f(y, x):
            # 在维度 1 上使用 y 的值重复 x 次
            return y.repeat_interleave(x, dim=1)

        # 创建一个二维张量 y
        y = torch.tensor([[1, 2], [3, 4]])
        # 创建一个一维张量 x
        x = torch.tensor([2, 3])
        # 转换 make_fx(f, tracing_mode="symbolic")(y, x) 的代码为字符串 r
        r = str(make_fx(f, tracing_mode="symbolic")(y, x).code).strip()
        # 断言 r 符合期望的内联结果 """\"
        self.assertExpectedInline(r, """\
# 定义一个类方法 `forward`，接受三个参数 `self`, `y_1`, `x_1`
def forward(self, y_1, x_1):
    # 调用 `torch.ops.aten.repeat_interleave.Tensor` 操作，对 `x_1` 进行重复插值
    repeat_interleave = torch.ops.aten.repeat_interleave.Tensor(x_1);  x_1 = None
    # 使用 `repeat_interleave` 对 `y_1` 进行索引选择，维度为 1
    index_select = torch.ops.aten.index_select.default(y_1, 1, repeat_interleave);  y_1 = repeat_interleave = None
    # 返回 `index_select` 结果
    return index_select


# 定义一个测试方法 `test_mod_gcd_unbacked`，内部定义了函数 `f`
def test_mod_gcd_unbacked(self):
    # 定义内部函数 `f`，接受 `_a`, `_b`, `_stride` 三个参数，并将其转换为标量值
    def f(_a, _b, _stride):
        a = _a.item()
        b = _b.item()
        stride = _stride.item()
        # 调用 `torch._check_is_size` 检查输入参数 `_a`, `_b`, `_stride` 是否符合尺寸标准
        torch._check_is_size(a)
        torch._check_is_size(b)
        torch._check_is_size(stride)
        # 生成大小为 `a * stride` 的随机张量 `ta` 和大小为 `b * stride` 的随机张量 `tb`
        ta = torch.randn(a * stride)
        tb = torch.randn(b * stride)
        # 连接 `ta` 和 `tb`，形成新的张量 `r`
        r = torch.cat([ta, tb])
        # 将张量 `r` 重塑为形状 `(a + b, stride)`
        return r.view(a + b, stride)

    # 初始化三个输入张量 `_a`, `_b`, `_stride`，并调用 `make_fx` 函数进行符号化跟踪
    _a = torch.tensor(30)
    _b = torch.tensor(20)
    _stride = torch.tensor(10)
    r = str(make_fx(f, tracing_mode="symbolic")(_a, _b, _stride).code).strip()
    # 断言跟踪结果 `r` 与期望的内联代码匹配
    self.assertExpectedInline(r, """\
def forward(self, _a_1, _b_1, _stride_1):
    # 调用 `torch.ops.aten._local_scalar_dense.default` 操作，对 `_a_1`, `_b_1`, `_stride_1` 进行本地标量稠密化处理
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(_a_1);  _a_1 = None
    _local_scalar_dense_1 = torch.ops.aten._local_scalar_dense.default(_b_1);  _b_1 = None
    _local_scalar_dense_2 = torch.ops.aten._local_scalar_dense.default(_stride_1);  _stride_1 = None
    # 计算 `_local_scalar_dense` 和 `_local_scalar_dense_2` 的乘积，生成 `mul`
    mul = _local_scalar_dense * _local_scalar_dense_2
    # 使用 `torch.ops.aten.randn.default` 操作生成大小为 `mul` 的随机张量 `randn`
    randn = torch.ops.aten.randn.default([mul], device = device(type='cpu'), pin_memory = False);  mul = None
    # 计算 `_local_scalar_dense_1` 和 `_local_scalar_dense_2` 的乘积，生成 `mul_1`
    mul_1 = _local_scalar_dense_1 * _local_scalar_dense_2
    # 使用 `torch.ops.aten.randn.default` 操作生成大小为 `mul_1` 的随机张量 `randn_1`
    randn_1 = torch.ops.aten.randn.default([mul_1], device = device(type='cpu'), pin_memory = False);  mul_1 = None
    # 连接 `randn` 和 `randn_1`，生成 `cat` 张量
    cat = torch.ops.aten.cat.default([randn, randn_1]);  randn = randn_1 = None
    # 计算 `_local_scalar_dense` 和 `_local_scalar_dense_1` 的和，生成 `add`
    add = _local_scalar_dense + _local_scalar_dense_1;  _local_scalar_dense = _local_scalar_dense_1 = None
    # 将 `cat` 张量重塑为形状 `(add, _local_scalar_dense_2)`，生成 `view`
    view = torch.ops.aten.view.default(cat, [add, _local_scalar_dense_2]);  cat = add = _local_scalar_dense_2 = None
    # 返回 `view` 结果
    return view""")


# 定义一个测试方法 `test_cumsum_unbacked`，内部定义了函数 `f`
def test_cumsum_unbacked(self):
    # 定义内部函数 `f`，接受一个参数 `x`，并将其转换为标量值
    def f(x):
        y = x.item()
        # 使用 `torch.randn` 生成大小为 `(3, y, 3)` 的随机张量 `z`
        z = torch.randn((3, y, 3))
        # 对张量 `z` 沿着维度 0 进行累加求和操作，生成 `cumsum`
        return z.cumsum(0)

    # 调用 `make_fx` 函数对内部函数 `f` 进行符号化跟踪，输入参数为大小为 1 的张量 `[5]`
    r = str(make_fx(f, tracing_mode="symbolic")(torch.tensor([5])).code).strip()
    # 断言跟踪结果 `r` 与期望的内联代码匹配
    self.assertExpectedInline(
        r, """\
def forward(self, x_1):
    # 调用 `torch.ops.aten._local_scalar_dense.default` 操作，对 `x_1` 进行本地标量稠密化处理
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    # 使用 `torch.ops.aten.randn.default` 操作生成大小为 `[3, _local_scalar_dense, 3]` 的随机张量 `randn`
    randn = torch.ops.aten.randn.default([3, _local_scalar_dense, 3], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    # 对 `randn` 张量沿着维度 0 进行累加求和操作，生成 `cumsum`
    cumsum = torch.ops.aten.cumsum.default(randn, 0);  randn = None
    # 返回 `cumsum` 结果"""  # noqa: B950
    )


def test_repeat_interleave_unbacked_output_size(self):
    # 定义测试方法 `test_repeat_interleave_unbacked_output_size`，内部定义了函数 `f`
    def f(x, y):
        # 计算张量 `x` 所有元素的和，生成标量 `s`
        s = x.sum().item()
        # 使用 `y.repeat_interleave` 对 `y` 进行重复插值，维度为 0，输出大小为 `s`
        return y.repeat_interleave(x, dim=0, output_size=s)

    # 调用 `make_fx` 函数对内部函数 `f` 进行符号化跟踪，输入参数为大小为 2 的张量 `[2, 3]` 和随机张量 `torch.randn(2)`
    r = str(make_fx(f, tracing_mode="symbolic")(torch.tensor([2, 3]), torch.randn(2)).code).strip()
    # 断言跟踪结果 `r` 与期望的内联代码匹配
    self.assertExpectedInline(
        r, """\
def forward(self, x_1, y_1):
    # 计算 `x_1` 张量所有元素的和，生成标量 `sum_1`
    sum_1 = torch.ops.aten.sum.default(x_1)
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(sum_1);  sum_1 = None
    # 调用 PyTorch 的 aten 操作，使用 sum_1 参数调用 _local_scalar_dense 的默认方法，并将结果赋给 _local_scalar_dense。清空 sum_1 变量。

    repeat_interleave = torch.ops.aten.repeat_interleave.Tensor(x_1, output_size = _local_scalar_dense);  x_1 = _local_scalar_dense = None
    # 调用 PyTorch 的 aten 操作，使用 x_1 参数和 _local_scalar_dense 变量作为输出大小，调用 repeat_interleave.Tensor 方法。最后清空 x_1 和 _local_scalar_dense 变量。

    index_select = torch.ops.aten.index_select.default(y_1, 0, repeat_interleave);  y_1 = repeat_interleave = None
    # 调用 PyTorch 的 aten 操作，使用 y_1 参数、索引 0 和 repeat_interleave 变量调用 index_select.default 方法。清空 y_1 和 repeat_interleave 变量。

    return index_select"""  # noqa: B950
    # 返回 index_select 结果，结束该函数。注意 noqa: B950 是用于忽略特定的 Flake8 错误，通常是指定过长的行或复杂的表达式。
def forward(self, x_1):
    # 使用 torch.ops.aten._local_scalar_dense.default 方法处理输入 x_1，返回结果给 _local_scalar_dense
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    # 使用 torch.ops.aten.arange.start 方法创建一个从 0 到 _local_scalar_dense 的张量 arange，设备为 CPU，不使用固定内存
    arange = torch.ops.aten.arange.start(0, _local_scalar_dense, device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    # 返回 arange 张量作为结果
    return arange"""  # noqa: B950
        )
    def test_tensor_symfloat(self):
        # 定义内部函数f，接受一个参数a，返回一个张量r，其值为a的行数平方，数据类型为浮点型
        def f(a):
            r = torch.tensor(a.size(0) ** 2.0)
            # 断言张量r的数据类型为torch中的浮点型
            assert r.dtype is torch.float
            return r

        # 使用make_fx函数创建一个symbolic模式的GraphModule对象gm，对输入的张量进行符号化处理
        gm = make_fx(f, tracing_mode="symbolic")(torch.randn(2))
        # 将GraphModule对象gm的代码表示为字符串r，去除首尾空白字符
        r = str(gm.code).strip()
        # 注意：此处进行了特化处理，目的是确保数据类型推断是正确的
        # 使用self.assertExpectedInline方法断言字符串r符合预期输出
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    # 拷贝 self._tensor_constant0 到 _tensor_constant0
    _tensor_constant0 = self._tensor_constant0
    # 调用 torch.ops.aten.lift_fresh_copy.default 方法来创建一个新的对象 lift_fresh_copy
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    # 返回 lift_fresh_copy 对象作为函数结果
    return lift_fresh_copy
    # 创建一个 3x3 的单位矩阵，放置在指定的设备上（这里是 CPU），并且不将数据固定在内存中
    eye = torch.ops.aten.eye.default(3, device=device(type='cpu'), pin_memory=False)
    # 复制一个常量张量的引用
    _tensor_constant0 = self._tensor_constant0
    # 创建一个常量张量的副本，并且释放原始张量的引用
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0); _tensor_constant0 = None
    # 从指定张量中选择索引为 0 的元素
    select = torch.ops.aten.select.int(eye, 0, 0)
    # 从先前选择的张量中再次选择索引为 0 的元素，并释放先前的选择张量的引用
    select_1 = torch.ops.aten.select.int(select, 0, 0); select = None
    # 将 lift_fresh_copy 复制到 select_1，然后释放这两个张量的引用
    copy_ = torch.ops.aten.copy_.default(select_1, lift_fresh_copy); select_1 = lift_fresh_copy = None
    # 计算索引张量的指定维度的大小
    sym_size_int = torch.ops.aten.sym_size.int(index, 0)
    # 将单位矩阵 eye 沿着指定维度进行扩展
    expand = torch.ops.aten.expand.default(eye, [sym_size_int, 3, 3])
    # 将扩展后的张量 view 成指定形状，并释放扩展张量的引用
    view = torch.ops.aten.view.default(expand, [sym_size_int, 3, 3]); expand = None
    # 计算 crop_camera_1 张量在第1维和第2维上的大小
    sym_size_int_1 = torch.ops.aten.sym_size.int(crop_camera_1, 1)
    sym_size_int_2 = torch.ops.aten.sym_size.int(crop_camera_1, 2)
    # 将 index 张量沿着指定维度进行扩展
    expand_1 = torch.ops.aten.expand.default(index, [sym_size_int, sym_size_int_1, sym_size_int_2]); index = None
    # 将扩展后的张量 view 成指定形状，并释放扩展张量和相关变量的引用
    view_1 = torch.ops.aten.view.default(expand_1, [sym_size_int, sym_size_int_1, sym_size_int_2]); expand_1 = sym_size_int_1 = sym_size_int_2 = None
    # 执行两个张量之间的批量矩阵乘法
    bmm = torch.ops.aten.bmm.default(view, view_1); view = view_1 = None
    # 将结果张量 view 成指定形状，并释放中间张量的引用
    view_2 = torch.ops.aten.view.default(bmm, [sym_size_int, 3, 3]); bmm = None
    # 计算 sym_size_int 乘以 3 的结果
    mul = sym_size_int * 3
    # 将 view_2 张量 view 成指定形状，并释放中间张量的引用
    view_3 = torch.ops.aten.view.default(view_2, [mul, 3]); view_2 = mul = None
    # 执行两个张量之间的矩阵乘法
    mm = torch.ops.aten.mm.default(view_3, eye); view_3 = eye = None
    # 将结果张量 view 成指定形状，并释放中间张量和相关变量的引用
    view_4 = torch.ops.aten.view.default(mm, [sym_size_int, 3, 3]); mm = sym_size_int = None
    # 在 crop_camera_1 张量中的 mask_1 索引位置上放置 view_4 张量的内容，并释放相关变量的引用
    index_put_ = torch.ops.aten.index_put_.default(crop_camera_1, [mask_1], view_4); crop_camera_1 = mask_1 = view_4 = None
    # 返回 None
    return None
    # 定义一个测试方法，用于测试布尔索引的功能
    def test_boolean_index(self):
        # 定义内部函数 f，接受三个参数 images、handedness 和 valid
        def f(images, handedness, valid):
            # 使用有效索引过滤图像数组
            images = images[valid]
            # 使用有效索引过滤 handedness 数组
            handedness = handedness[valid]
            # 创建一个右手为真的布尔掩码
            right_hand_mask = handedness == 1
            # 对右手的图像进行水平翻转操作
            images[right_hand_mask] = images[right_hand_mask].flip(-1)

        # 使用 make_fx 函数创建一个函数表示，跟踪模式为符号化
        r = str(make_fx(f, tracing_mode="symbolic")(
            # 创建一个大小为 (512, 1, 96, 96) 的随机整数张量 images
            torch.randint(0, 256, (512, 1, 96, 96)),
            # 创建一个大小为 (512,) 的随机整数张量 handedness，值为 0 或 1
            torch.randint(0, 1, (512,)),
            # 创建一个大小为 (512,) 的随机布尔张量 valid，值为 True 或 False
            torch.randint(0, 2, (512,), dtype=torch.bool)
        ).code).strip()
        # 使用 assertExpectedInline 方法断言结果 r 符合预期
        self.assertExpectedInline(r, """\
def forward(self, images_1, handedness_1, valid_1):
    # 使用 torch.ops.aten.index.Tensor 函数，根据 valid_1 索引 images_1 张量的数据，存储在 index 中；images_1 置为 None
    index = torch.ops.aten.index.Tensor(images_1, [valid_1]);  images_1 = None
    # 使用 torch.ops.aten.index.Tensor 函数，根据 valid_1 索引 handedness_1 张量的数据，存储在 index_1 中；handedness_1 和 valid_1 置为 None
    index_1 = torch.ops.aten.index.Tensor(handedness_1, [valid_1]);  handedness_1 = valid_1 = None
    # 使用 torch.ops.aten.eq.Scalar 函数，比较 index_1 是否等于标量 1，结果存储在 eq 中；index_1 置为 None
    eq = torch.ops.aten.eq.Scalar(index_1, 1);  index_1 = None
    # 使用 torch.ops.aten.index.Tensor 函数，根据 eq 中的值索引 index 张量，结果存储在 index_2 中；index 置为 None
    index_2 = torch.ops.aten.index.Tensor(index, [eq])
    # 使用 torch.ops.aten.flip.default 函数，对 index_2 进行沿最后一个维度的翻转操作，结果存储在 flip 中；index_2 置为 None
    flip = torch.ops.aten.flip.default(index_2, [-1]);  index_2 = None
    # 使用 torch.ops.aten.index_put_.default 函数，将 flip 的值插入到 index 中，索引由 eq 决定；index、eq 和 flip 置为 None
    index_put_ = torch.ops.aten.index_put_.default(index, [eq], flip);  index = eq = flip = None
    # 返回 None
    return None
    # 定义一个函数 f，接受两个参数 x 和 y
    def f(x, y):
        # 创建一个全零张量 z，大小为 x 的值
        z = torch.zeros(x.item())
        # 检查 z 的大小是否与 y 的大小相等
        torch._check(z.size(0) == y.size(0))  # refines i0 = s0
        # 如果 z 的大小为 4，则返回 y 的每个元素乘以 2
        if z.size(0) == 4:
            return y * 2
        # 否则返回 y 的每个元素加上 2
        else:
            return y + 2

    # 使用 symbolic 模式对函数 f 进行追踪，并传入参数 torch.tensor(10) 和 torch.randn(10)
    r = str(make_fx(f, tracing_mode="symbolic")(torch.tensor(10), torch.randn(10)).code).strip()
    # 断言 r 的值符合预期的内联代码
    self.assertExpectedInline(r, """\
# 定义一个名为 forward 的方法，接受两个参数 x_1 和 y_1
def forward(self, x_1, y_1):
    # 调用 torch.ops.aten._local_scalar_dense.default 方法，处理 x_1 参数并赋值给 _local_scalar_dense，然后将 x_1 置为 None
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    # 调用 torch.ops.aten.zeros.default 方法创建一个全零张量 zeros，其形状由 _local_scalar_dense 决定，设备为 'cpu'，不使用 pin memory
    zeros = torch.ops.aten.zeros.default([_local_scalar_dense], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    # 调用 torch.ops.aten.add.Tensor 方法，将 y_1 与标量值 2 相加，并赋值给 add，然后将 y_1 置为 None
    add = torch.ops.aten.add.Tensor(y_1, 2);  y_1 = None
    # 返回 add 变量作为方法的结果
    return add
        def f(lengths, values):
            # 将长度列表转换为整数列表，因为目前不直接支持 tolist 方法
            sizes = [lengths[i].item() for i in range(lengths.size(0))]
            for s in sizes:
                # 使用 torch._constrain_as_size 方法确保 s 是有效的张量尺寸
                # TODO(avik): 是否应该使用 torch._check_is_size 生成断言？
                torch._constrain_as_size(s)
            # 使用给定的尺寸将 values 张量分割成多个子张量
            return torch.split(values, sizes)

        # 生成函数 f 的 FX 代码并以字符串形式返回
        r = str(make_fx(f, tracing_mode="symbolic")(
            torch.tensor([2, 3, 4]),  # 创建包含长度信息的张量
            torch.randn(9)  # 创建一个包含 9 个随机数的张量
        ).code).strip()
        # 断言生成的 FX 代码与预期的内联字符串匹配
        self.assertExpectedInline(r, """\
def forward(self, lengths_1, values_1):
    # 从 lengths_1 中选择第一个元素
    select = torch.ops.aten.select.int(lengths_1, 0, 0)
    # 使用所选元素创建稠密标量 _local_scalar_dense
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(select);  select = None
    # 从 lengths_1 中选择第二个元素
    select_1 = torch.ops.aten.select.int(lengths_1, 0, 1)
    # 使用所选元素创建稠密标量 _local_scalar_dense_1
    _local_scalar_dense_1 = torch.ops.aten._local_scalar_dense.default(select_1);  select_1 = None
    # 从 lengths_1 中选择第三个元素，并清空 lengths_1
    select_2 = torch.ops.aten.select.int(lengths_1, 0, 2);  lengths_1 = None
    # 使用所选元素创建稠密标量 _local_scalar_dense_2
    _local_scalar_dense_2 = torch.ops.aten._local_scalar_dense.default(select_2);  select_2 = None
    # 使用三个稠密标量限制大小的符号约束
    sym_constrain_range_for_size = torch.ops.aten.sym_constrain_range_for_size.default(_local_scalar_dense)
    sym_constrain_range_for_size_1 = torch.ops.aten.sym_constrain_range_for_size.default(_local_scalar_dense_1)
    sym_constrain_range_for_size_2 = torch.ops.aten.sym_constrain_range_for_size.default(_local_scalar_dense_2)
    # 使用指定大小进行分割 values_1
    split_with_sizes = torch.ops.aten.split_with_sizes.default(values_1, [_local_scalar_dense, _local_scalar_dense_1, _local_scalar_dense_2]);  values_1 = _local_scalar_dense = _local_scalar_dense_1 = _local_scalar_dense_2 = None
    # 获取分割后的第一个、第二个和第三个部分
    getitem = split_with_sizes[0]
    getitem_1 = split_with_sizes[1]
    getitem_2 = split_with_sizes[2];  split_with_sizes = None
    # 返回分割后的结果元组
    return (getitem, getitem_1, getitem_2)
    # 定义一个测试方法，用于测试带有自定义追踪器且保留 nn_module_stack 的函数生成情况
    def test_make_fx_with_custom_tracer_preserving_nn_module_stack(self):

        # 定义一个继承自 torch.nn.Module 的类 Bar
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 重写 forward 方法，实现前向传播逻辑
            def forward(self, x):
                return x + 1

        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 实例化 Bar 类
                self.bar = Bar()

            # 重写 forward 方法，实现前向传播逻辑
            def forward(self, x):
                return x + self.bar(x)

        # 使用 make_fx 函数对 Foo 类实例进行函数化，并传入随机生成的张量作为输入
        gm = make_fx(Foo())(torch.randn(4, 4))

        # 遍历生成的图中的节点
        for node in gm.graph.nodes:
            # 断言节点的 meta 数据中不包含 "nn_module_stack"
            self.assertTrue("nn_module_stack" not in node.meta)

        # 实例化 Foo 类
        foo = Foo()

        # 定义一个函数 functional_call，将 foo 实例化后传入 stateless._reparametrize_module 中进行处理
        def functional_call(*args, **kwargs):
            with stateless._reparametrize_module(foo, {}):
                return foo(*args, **kwargs)

        # 将原始模块保存在 functional_call._orig_mod 属性中
        functional_call._orig_mod = foo

        # 使用 make_fx 函数对 functional_call 函数进行函数化，记录模块堆栈
        gm_with_stack = make_fx(functional_call, record_module_stack=True)(torch.randn(4, 4))

        # 标志变量，用于指示是否找到了符合条件的节点
        found = False

        # 遍历生成的带有堆栈信息的图中的节点
        for node in gm_with_stack.graph.nodes:
            if "nn_module_stack" in node.meta:
                # 判断节点 meta 数据中的 nn_module_stack 是否包含预期的堆栈信息
                if len(node.meta["nn_module_stack"]) == 1:
                    self.assertTrue("custom_tracer_preserving_nn_module_stack.<locals>.Foo" in str(node.meta["nn_module_stack"]))
                    found = True
                elif len(node.meta["nn_module_stack"]) == 2:
                    self.assertTrue("preserving_nn_module_stack.<locals>.Bar" in str(node.meta["nn_module_stack"]))
                    found = True
                else:
                    # 如果堆栈信息超过了预期的层数（最多 2 层），断言失败
                    self.assertTrue(False)

        # 最终断言 found 变量为 True，表示找到了符合条件的节点
        self.assertTrue(found)

        # 使用 make_fx 函数对 functional_call 函数进行函数化，但不记录模块堆栈
        gm_without_stack = make_fx(functional_call)(torch.randn(4, 4))

        # 遍历生成的不带堆栈信息的图中的节点
        for node in gm_without_stack.graph.nodes:
            # 断言节点的 meta 数据中不包含 "nn_module_stack"
            self.assertTrue("nn_module_stack" not in node.meta)

    # 定义一个测试方法，用于测试符号推理到张量的转换情况
    def test_symint_to_tensor(self):
        # 定义一个函数 f，接受一个参数 a，并返回 a 除以其形状大小的结果
        def f(a):
            return a / a.shape[0]

        # 使用 make_fx 函数对 f 函数进行函数化，选择符号跟踪模式
        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(4)).code).strip()

        # 断言 r 的值符合预期的内联代码
        self.assertExpectedInline(r, """\
# 定义一个名为 forward 的方法，接受一个参数 a_1
def forward(self, a_1):
    # 使用 torch.ops.aten.sym_size.int 方法获取张量 a_1 的第一个维度大小，返回一个整数
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0)
    # 使用 torch.ops.aten.div.Tensor 方法将张量 a_1 与 sym_size_int 相除，得到结果 div；
    # 清除 a_1 和 sym_size_int 的引用，释放内存
    div = torch.ops.aten.div.Tensor(a_1, sym_size_int);  a_1 = sym_size_int = None
    # 返回计算结果 div
    return div
    # 测试函数，验证元数据功能是否正常
    def test_metadata(self):
        # 内部函数定义，接受两个参数 a 和 b
        def f(a, b):
            # 调用 a 的 new_empty 方法创建一个新的张量 d，形状为 a.shape[0] + b.shape[0]
            d = a.new_empty(a.shape[0] + b.shape[0])
            return d
        
        # 使用 make_fx 函数将 f 函数转换为符号化的计算图，并传入两个随机张量进行计算
        fx_g = make_fx(f, tracing_mode="symbolic")(torch.randn(5), torch.randn(4))
        
        # 获取计算图中满足条件的节点 meta_c，条件为节点的目标操作为 aten.new_empty.default
        meta_c = _get_node(fx_g, lambda x: x.target == aten.new_empty.default)
        
        # 获取计算图中满足条件的节点 meta_d，条件为节点的目标操作为 operator.add
        meta_d = _get_node(fx_g, lambda x: x.target == operator.add)
        
        # 断言 meta_c 和 meta_d 的 'val' 属性的形状第一个维度的表达式相同
        self.assertTrue(meta_c.meta['val'].shape[0].node.expr == meta_d.meta['val'].node.expr)

    # 测试函数，验证元数据功能在新数据上的正确性
    def test_metadata_fresh(self):
        # 定义函数 f，接受一个参数 x，断言 x 的第一个维度为 3，返回 x 的余弦值
        def f(x):
            assert x.shape[0] == 3
            return x.cos()
        
        # 使用 make_fx 函数将 f 函数转换为符号化的计算图，并传入一个形状为 (3,) 的随机张量
        fx_g = make_fx(f, tracing_mode="symbolic")(torch.randn(3))
        
        # 获取计算图中满足条件的节点 meta_cos，条件为节点的目标操作为 aten.cos.default
        meta_cos = _get_node(fx_g, lambda x: x.target == aten.cos.default)
        
        # 获取计算图中满足条件的节点 meta_inp，条件为节点的操作类型为 'placeholder'
        meta_inp = _get_node(fx_g, lambda x: x.op == 'placeholder')
        
        # 断言 meta_cos 的 'val' 属性的第一个维度为 3
        self.assertTrue(meta_cos.meta['val'].shape[0] == 3)
        
        # 检查输入表达式是否已更新，尽管约束发生在之后
        self.assertTrue(meta_inp.meta['val'].shape[0] == 3)

    # 测试函数，验证符号化数值和符号化浮点数在元数据中的处理
    def test_elementwise_meta_with_sym_numbers(self):
        # 定义函数 f，接受三个参数 x、offset 和 as_sym_float（默认为 False）
        def f(x, offset, as_sym_float=False):
            # 获取张量 x 的第一个维度大小 x0
            x0 = x.size()[0]
            
            # 如果 as_sym_float 为 True，则将 x0 转换为符号化浮点数
            if as_sym_float:
                x0 = torch.sym_float(x0)
            
            # 返回 x0 + offset 的结果
            return torch.add(x0, offset)
        
        # 使用 make_fx 函数将 f 函数转换为符号化的计算图，并传入一个大小为 (2, 3) 的随机张量和 offset 为 2.0 的浮点数
        fx_g = make_fx(f, tracing_mode="symbolic")(torch.rand(2, 3), 2.0, False)
        
        # 获取计算图中满足条件的节点 meta_add，条件为节点的目标操作为 aten.add.Tensor
        meta_add = _get_node(fx_g, lambda x: x.target == aten.add.Tensor)
        
        # 断言 meta_add 的 'val' 属性的形状为空
        self.assertEqual(meta_add.meta['val'].shape, ())
        
        # 断言 meta_add 的 'val' 属性的数据类型为 torch.float32
        self.assertEqual(meta_add.meta['val'].dtype, torch.float32)

        # 使用 make_fx 函数将 f 函数转换为符号化的计算图，并传入一个大小为 (2, 3) 的随机张量和 offset 为 2 的整数
        fx_g = make_fx(f, tracing_mode="symbolic")(torch.rand(2, 3), 2, False)
        
        # 获取计算图中满足条件的节点 meta_add，条件为节点的目标操作为 aten.add.Tensor
        meta_add = _get_node(fx_g, lambda x: x.target == aten.add.Tensor)
        
        # 断言 meta_add 的 'val' 属性的形状为空
        self.assertEqual(meta_add.meta['val'].shape, ())
        
        # 断言 meta_add 的 'val' 属性的数据类型为 torch.int64
        self.assertEqual(meta_add.meta['val'].dtype, torch.int64)

        # 使用 make_fx 函数将 f 函数转换为符号化的计算图，并传入一个大小为 (2, 3) 的随机张量、offset 为 2 的整数和 as_sym_float 为 True
        fx_g = make_fx(f, tracing_mode="symbolic")(torch.rand(2, 3), 2, True)
        
        # 获取计算图中满足条件的节点 meta_add，条件为节点的目标操作为 aten.add.Tensor
        meta_add = _get_node(fx_g, lambda x: x.target == aten.add.Tensor)
        
        # 断言 meta_add 的 'val' 属性的形状为空
        self.assertEqual(meta_add.meta['val'].shape, ())
        
        # 断言 meta_add 的 'val' 属性的数据类型为 torch.float32
        self.assertEqual(meta_add.meta['val'].dtype, torch.float32)

    # 测试函数，验证返回符号化整数的情况
    def test_return_symint(self):
        # 定义函数 f，接受一个参数 x，返回 x 的第一个维度大小、x 的余弦值和 x 第一个维度大小除以 5 的结果
        def f(x):
            return x.shape[0], x.cos(), x.shape[0] / 5
        
        # 调用 _test_dynamic 方法，验证函数 f 在输入 [(5,)] 下的输出为 [[(4,)], [(12,)]]
        self._test_dynamic(f, [(5,)], [[(4,)], [(12,)]])

        # 定义函数 f，接受一个参数 x，返回 x 的形状
        def f(x):
            return x.shape
        
        # 调用 _test_dynamic 方法，验证函数 f 在输入 [(5, 3)] 下的输出为 [[(4, 6)]]
        self._test_dynamic(f, [(5, 3)], [[(4, 6)]])

    # 测试函数，验证方法调用中的符号化整数处理
    def test_rmethod(self):
        # 定义函数 f，接受一个参数 x，返回 x 的大小加上 x 本身的结果
        def f(x):
            return x.size(0) + x
        
        # 调用 _test_dynamic 方法，验证函数 f 在输入 [(5,)] 下的输出为 [[(4,)], [(12,)]] 
        self._test_dynamic(f, [(5,)], [[(4,)], [(12,)]])
    # 定义测试函数 test_mega_guard，用于测试函数 f 的行为
    def test_mega_guard(self):
        # 定义内部函数 f，验证输入张量 a 和 b 的形状条件，并返回 a 的余弦值
        def f(a, b):
            assert a.shape[0] == b.shape[0] * 2  # 断言：a 的行数是 b 行数的两倍
            return a.cos()  # 返回张量 a 的余弦值

        # 生成经过符号跟踪的 fx_g 张量
        fx_g = make_fx(f, tracing_mode="symbolic")(torch.randn(16), torch.randn(8))

        # 导入本地源 LocalSource 类
        from torch._dynamo.source import LocalSource

        # 断言 fx_g 的形状环境生成的保护条件，验证参数 fx_placeholder_vals(fx_g) 和 [LocalSource("a"), LocalSource("b")]，不忽略静态条件
        self.assertExpectedInline(
            str(fx_g.shape_env.produce_guards(fx_placeholder_vals(fx_g), [LocalSource("a"), LocalSource("b")], ignore_static=False)),  # noqa: B950
            """["L['a'].size()[0] == 2*L['b'].size()[0]", "L['a'].stride()[0] == 1", "L['a'].storage_offset() == 0", "L['b'].stride()[0] == 1", "L['b'].storage_offset() == 0", "2 <= L['b'].size()[0]"]"""  # noqa: B950
        )

        # 断言 fx_g 的形状环境生成的保护条件，验证参数 fx_placeholder_vals(fx_g) 和 [LocalSource("a"), LocalSource("b")]，忽略静态条件
        self.assertExpectedInline(
            str(fx_g.shape_env.produce_guards(fx_placeholder_vals(fx_g), [LocalSource("a"), LocalSource("b")], ignore_static=True)),  # noqa: B950
            """["L['a'].size()[0] == 2*L['b'].size()[0]", "2 <= L['b'].size()[0]"]"""  # noqa: B950
        )

    # 定义测试函数 test_guard_upperbound_range_refinement，用于测试函数 f 的行为
    def test_guard_upperbound_range_refinement(self):
        # 定义函数 f，验证输入张量 a 的形状条件，并返回 a 的余弦值
        def f(a):
            assert a.shape[0] > 5 and a.shape[0] > 12  # 断言：a 的行数大于 5 和 12
            return a.cos()  # 返回张量 a 的余弦值

        # 生成经过符号跟踪的 tensor 张量
        tensor = make_fx(f, tracing_mode="symbolic")(torch.randn(15))

        # 断言 tensor 的保护条件，显示张量 a 的行数大于等于 13
        self.assertExpectedInline(show_guards(tensor), """13 <= L['a'].size()[0]""")

    # 定义测试函数 test_guard_lowerbound_range_refinement，用于测试函数 f 的行为
    def test_guard_lowerbound_range_refinement(self):
        # 定义函数 f，验证输入张量 a 的形状条件，并返回 a 的余弦值
        def f(a):
            assert a.shape[0] < 20 and a.shape[0] < 30  # 断言：a 的行数小于 20 和 30
            return a.cos()  # 返回张量 a 的余弦值

        # 生成经过符号跟踪的 tensor 张量
        tensor = make_fx(f, tracing_mode="symbolic")(torch.randn(15))

        # 断言 tensor 的保护条件，显示张量 a 的行数小于等于 19
        self.assertExpectedInline(show_guards(tensor), """L['a'].size()[0] <= 19""")

    # 定义测试函数 test_guard_upperbound_range_refinement_multivariate，用于测试函数 f 的行为
    def test_guard_upperbound_range_refinement_multivariate(self):
        # 定义函数 f，验证输入张量 a 和 b 的形状条件，并返回 a 的余弦值
        def f(a):
            assert a.shape[0] > 5 and a.shape[0] > 12  # 断言：a 的行数大于 5 和 12
            assert a.shape[1] > 5 and a.shape[1] > a.shape[0]  # 断言：a 的列数大于 5 和大于行数
            return a.cos()  # 返回张量 a 的余弦值

        # 生成经过符号跟踪的 tensor 张量，形状为 (15, 20)
        tensor = make_fx(f, tracing_mode="symbolic")(torch.randn((15, 20)))

        # 断言 tensor 的保护条件，显示张量 a 的行数大于等于 13
        self.assertExpectedInline(show_guards(tensor), """\
L['a'].size()[1] > L['a'].size()[0]
L['a'].size()[0] >= 13
L['a'].size()[1] >= 14
    # 定义内部函数 f，用于检查张量 t 的形状是否为 (10, ...)
    def f(t):
        assert t.shape[0] == 10  # 断言张量的第一个维度是否为 10
        return t

    # 使用 make_fx 函数创建一个特化了 f 函数的张量操作，tracing_mode 参数设置为 "symbolic"
    tensor = make_fx(f, tracing_mode="symbolic")(torch.randn(10))
    # 使用 show_guards 函数展示张量的保护条件，预期结果为空字符串
    self.assertExpectedInline(show_guards(tensor), """""")
make_fx_failures = {
    # 使用 xfail 标记 'allclose' 为预期失败
    xfail('allclose'),
    # 使用 xfail 标记 'equal' 为预期失败
    xfail('equal'),
    # 使用 skip 标记 'new_empty' 为跳过测试
    skip('new_empty'),
    # 使用 skip 标记 'empty_like' 为跳过测试
    skip('empty_like'),
    # 使用 skip 标记 'empty' 为跳过测试
    skip('empty'),
    # 使用 skip 标记 'empty_permuted' 为跳过测试
    skip('empty_permuted'),
    # 使用 skip 标记 'linalg.lstsq' 的 'grad_oriented' 参数为跳过测试
    skip('linalg.lstsq', 'grad_oriented'),
    # 使用 skip 标记 'nn.functional.max_unpool1d' 在 'cpu' 设备上为跳过测试
    skip('nn.functional.max_unpool1d', '', device_type='cpu'),
    # 使用 skip 标记 'nn.functional.max_unpool2d' 在 'cpu' 设备上为跳过测试
    skip('nn.functional.max_unpool2d', '', device_type='cpu'),
    # 使用 skip 标记 'nn.functional.max_unpool3d' 在 'cpu' 设备上为跳过测试
    skip('nn.functional.max_unpool3d', '', device_type='cpu'),
    # 使用 skip 标记 'linalg.lstsq' 为跳过测试，可能是精度问题
    skip('linalg.lstsq'),  # flaky, probably just a precision issue

    # 数据依赖的控制流，使用 skip 标记 'item' 为跳过测试
    skip('item'),
    # 使用 xfail 标记 'cov' 为预期失败
    xfail('cov'),
    # 使用 xfail 标记 'nn.functional.gaussian_nll_loss' 为预期失败
    xfail('nn.functional.gaussian_nll_loss'),
    # 使用 xfail 标记 'tensor_split' 为预期失败
    xfail('tensor_split'),
    # 使用 xfail 标记 'corrcoef' 为预期失败
    xfail('corrcoef'),
    # 使用 xfail 标记 'quantile' 为预期失败
    xfail('quantile'),
    # 使用 xfail 标记 'nanquantile' 为预期失败
    xfail('nanquantile'),

    # 似乎是创建了一个稀疏张量，但不会被 tensor.is_sparse 捕获
    xfail('sparse.sampled_addmm'),
    # 使用 xfail 标记 'sparse.mm' 的 'reduce' 参数为预期失败
    xfail('sparse.mm', 'reduce'),

    # 代理张量当前不正确地支持稀疏
    skip('to_sparse'),
    # 使用 skip 标记 'block_diag' 为跳过测试，可能导致段错误
    skip('block_diag'),

    # AssertionError: Tensor-likes are not close!
    # 使用 skip 标记 'empty_strided' 在 'cpu' 设备上为跳过测试
    skip('empty_strided', '', device_type='cpu'),
}

only_real_tensor_failures = {
    # 使用 xfail 标记 'narrow' 为预期失败，仅适用于真实张量的情况
    xfail('narrow'),
}

only_fake_tensor_failures = {
    # 使用 xfail 标记 'narrow' 为预期失败，仅适用于虚假张量的情况
    xfail('narrow'),
}

fake_tensor_failures = {
    # ASAN 错误，由于除以 0 引起
    skip('nn.functional.nll_loss'),
}

symbolic_tensor_failures = {
    # 使用 xfail 标记 'combinations' 为预期失败
    xfail('combinations', ''),
    # 使用 xfail 标记 'geqrf' 为预期失败，原因是找不到符号元函数或分解
    xfail('geqrf', ''),
    # 使用 xfail 标记 'histogram' 为预期失败，原因是找不到符号元函数或分解
    xfail('histogram', ''),
    # 使用 xfail 标记 'histogramdd' 为预期失败，原因是找不到符号元函数或分解
    xfail('histogramdd', ''),
    # 使用 xfail 标记 'kthvalue' 为预期失败，原因是找不到符号元函数或分解
    xfail('kthvalue', ''),
    # 使用 xfail 标记 'nanquantile' 为预期失败，原因是在 'Meta' 后端无法运行 'aten::equal' 函数
    xfail('nanquantile', ''),
    # 使用 xfail 标记 'nn.functional.binary_cross_entropy' 为预期失败，原因是找不到符号元函数或分解
    xfail('nn.functional.binary_cross_entropy', ''),
    # 使用 xfail 标记 'nn.functional.cross_entropy' 为预期失败，原因是找不到符号元函数或分解
    xfail('nn.functional.cross_entropy', ''),
    # 使用 xfail 标记 'nn.functional.ctc_loss' 为预期失败，原因是找不到符号元函数或分解
    xfail('nn.functional.ctc_loss'),
    # 使用 xfail 标记 'quantile' 为预期失败，原因是在 'Meta' 后端无法运行 'aten::equal' 函数
    xfail('quantile', ''),
    # 使用 xfail 标记 'unique_consecutive' 为预期失败，原因是找不到符号元函数或分解
    xfail('unique_consecutive', ''),

    # 预期失败，预期内核大小参数为 'List[int]'
    xfail('max_pool2d_with_indices_backward', ''),
    
    # 许多复杂操作不正确的步幅、元数据
    xfail('fft.fft', ''),
    xfail('fft.hfft2', ''),
    xfail('fft.hfft', ''),
    xfail('fft.hfftn', ''),
    xfail('fft.ifft', ''),
    xfail('fft.ihfft2', ''),
    xfail('fft.ihfft', ''),
    xfail('fft.ihfftn', ''),
    xfail('fft.ihfft2', ''),
    xfail('fft.irfft2', ''),
    xfail('fft.irfft', ''),
}
    # 调用 xfail 函数，标记 'fft.irfftn' 功能为预期失败（xfail），不传递额外参数
    xfail('fft.irfftn', ''),
    # 调用 xfail 函数，标记 'fft.rfft2' 功能为预期失败（xfail），不传递额外参数
    xfail('fft.rfft2', ''),
    # 调用 xfail 函数，标记 'fft.rfft' 功能为预期失败（xfail），不传递额外参数
    xfail('fft.rfft', ''),
    # 调用 xfail 函数，标记 'fft.rfftn' 功能为预期失败（xfail），不传递额外参数
    xfail('fft.rfftn', ''),
    # 调用 xfail 函数，标记 'stft' 功能为预期失败（xfail），不传递额外参数
    xfail('stft', '')
}

symbolic_tensor_segfaults = {
    skip('nn.functional.batch_norm')  # Segfault??
}

symbolic_tensor_failures.update(symbolic_tensor_segfaults)

inplace_symbolic_tensor_failures = {
    # 定义带有失败标记的测试用例，说明为什么会失败
    xfail('float_power', ''),  # base given to float_power_ has dtype Float but the operation's result requires dtype Double
}

out_symbolic_tensor_failures = {
    # Cast error details: Unable to cast (...) to Tensor
    #
    # This happens because the test is set up to call the out variant using the `out` kwarg:
    #   torch._some_op(arg1, arg2, out=(out1, out2, out3))
    #
    # However, this only works on torch ops, not aten ops. For `_batch_norm_with_update`,
    # this fails because the op has no python bindings, so it doesn't support the `out` kwarg
    # way of calling its out variant.
    #
    # 定义带有失败标记的测试用例，详细说明失败原因和情况
    xfail('_batch_norm_with_update', ''),
    xfail('_native_batch_norm_legit', ''),
    xfail('angle', ''),
    xfail('argmax', ''),
    xfail('argmin', ''),
    xfail('fft.fft2', ''),
    xfail('fft.fftn', ''),
    xfail('fft.ifft2', ''),
    xfail('fft.ifftn', ''),
    xfail('gather', ''),
    xfail('linalg.pinv', ''),
    xfail('linalg.pinv', 'hermitian'),
    xfail('lu', ''),
    xfail('scatter_add', ''),
    xfail('scatter', ''),
    xfail('take_along_dim', ''),
    xfail('triangular_solve', ''),
    xfail('view_copy', ''),

    # SymIntArrayRef expected to contain only concrete
    #
    # 定义带有失败标记的测试用例，说明预期只接受具体参数
    xfail('ones', ''),
    xfail('randn', ''),
    xfail('zeros', ''),

    # RuntimeError: Cannot call numel() on tensor with symbolic sizes/strides
    #
    # 定义带有失败标记的测试用例，说明由于符号化尺寸/步长导致无法调用 numel() 方法
    xfail('index_reduce', 'prod'),
    xfail('index_reduce', 'mean'),
    xfail('index_reduce', 'amax'),
    xfail('index_reduce', 'amin'),
}

out_symbolic_tensor_segfaults = {
    skip('nanmean', ''),
}

out_symbolic_tensor_failures.update(out_symbolic_tensor_segfaults)

# Copies inputs to inplace operations to avoid inplace modifications
#   to leaves requiring gradient
#
# 定义函数 _get_safe_inplace，用于生成安全的 inplace 操作函数
def _get_safe_inplace(inplace_variant):
    @functools.wraps(inplace_variant)
    def _fn(t, *args, **kwargs):
        return inplace_variant(t.clone(), *args, **kwargs)

    return _fn

def _test_make_fx_helper(self, device, dtype, op, tracing_mode, inplace=False, out=False):
    fn = _get_safe_inplace(op.get_inplace()) if inplace else op.op
    sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)

    # Limit ourselves to first 100 inputs so symbolic tracing tests don't take too long
    #
    # 限制为前 100 个输入，以避免符号化追踪测试时间过长
    count = 100
    if out:
        count = 5
    # 遍历从sample_inputs_itr迭代器中取出的count个样本输入
    for sample_input in itertools.islice(sample_inputs_itr, count):
        # 如果inplace为True且sample_input标记为broadcasts_input，则跳过本次循环
        if inplace and sample_input.broadcasts_input:
            continue
        
        # 构造参数列表，包括sample_input.input和其余sample_input.args的内容
        args = [sample_input.input] + list(sample_input.args)
        # 直接使用sample_input.kwargs作为关键字参数kwargs
        kwargs = sample_input.kwargs
        
        # 如果存在输出(out为真)，则调用fn函数计算期望的输出，并将其作为关键字参数'out'传入kwargs中
        if out:
            expected = fn(*args, **kwargs)
            kwargs['out'] = expected
        
        # 尝试使用optests.make_fx_check进行函数效果检查
        try:
            # 调用make_fx_check函数来检查fn函数的行为，包括参数args、kwargs，使用tracing_mode进行跟踪模式
            optests.make_fx_check(fn, args, kwargs, tracing_mode, self.assertEqual,
                                  randomize_data=True)
        # 如果捕获到DynamicOutputShapeException异常，则跳过当前测试
        except DynamicOutputShapeException:
            self.skipTest("Dynamic output shape operation in trace")
# 定义一个装饰器函数，用于跳过测试函数，如果测试函数的名称与给定的模式匹配。
def skipIfNameMatches(pattern):
    """
    Decorator to skip a test if its name matches the given pattern.
    """
    # 实际的装饰器函数，接受一个测试函数作为参数
    def decorator(test_func):
        # 包装函数，接受任意数量的位置参数和关键字参数
        def wrapper(*args, **kwargs):
            # 如果测试函数的名称与模式匹配
            if re.match(pattern, test_func.__name__):
                # 抛出 unittest.SkipTest 异常，跳过测试，并给出相应的提示信息
                raise unittest.SkipTest(f"Test '{test_func.__name__}' skipped because its name matches the pattern '{pattern}'")
            # 否则，继续执行原始的测试函数
            return test_func(*args, **kwargs)
        return wrapper
    return decorator

# 从 hop_db 中过滤掉名称为 "auto_functionalize" 的操作，并将结果存储在 filtered_hop_db 中
filtered_hop_db = [op for op in hop_db if op.name != "auto_functionalize"]

# 如果 torch 不支持 dynamo，则跳过整个测试类 TestProxyTensorOpInfo
@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "Cond requires dynamo")
class TestProxyTensorOpInfo(TestCase):

    # 使用 op_db、filtered_hop_db 和 custom_op_db 进行测试，只允许 torch.float 数据类型
    @ops(op_db + filtered_hop_db + custom_op_db, allowed_dtypes=(torch.float,))
    # 跳过名称为 'TestProxyTensorOpInfo' 的测试函数 'test_make_fx_exhaustive'，如果包含在 make_fx_failures 或 only_real_tensor_failures 中
    @skipOps('TestProxyTensorOpInfo', 'test_make_fx_exhaustive', make_fx_failures.union(only_real_tensor_failures))
    def test_make_fx_exhaustive(self, device, dtype, op):
        _test_make_fx_helper(self, device, dtype, op, "real")

    # 使用 op_db、filtered_hop_db 和 custom_op_db 进行测试，只允许 torch.float 数据类型
    @ops(op_db + filtered_hop_db + custom_op_db, allowed_dtypes=(torch.float,))
    # 跳过名称为 'TestProxyTensorOpInfo' 的测试函数 'test_make_fx_fake_exhaustive'，如果包含在 make_fx_failures、fake_tensor_failures 或 only_fake_tensor_failures 中
    @skipOps('TestProxyTensorOpInfo', 'test_make_fx_fake_exhaustive',
             make_fx_failures.union(fake_tensor_failures, only_fake_tensor_failures))
    def test_make_fx_fake_exhaustive(self, device, dtype, op):
        _test_make_fx_helper(self, device, dtype, op, "fake")

    # 使用 op_db、filtered_hop_db 和 custom_op_db 进行测试，只允许 torch.float 数据类型
    @ops(op_db + filtered_hop_db + custom_op_db, allowed_dtypes=(torch.float,))
    # 跳过名称为 'TestProxyTensorOpInfo' 的测试函数 'test_make_fx_symbolic_exhaustive'，如果包含在 make_fx_failures、fake_tensor_failures 或 symbolic_tensor_failures 中
    @skipOps('TestProxyTensorOpInfo', 'test_make_fx_symbolic_exhaustive',
             make_fx_failures | fake_tensor_failures | symbolic_tensor_failures)
    def test_make_fx_symbolic_exhaustive(self, device, dtype, op):
        _test_make_fx_helper(self, device, dtype, op, "symbolic")

    # 使用 op_db 和 custom_op_db 进行测试，只允许 torch.float 数据类型
    @ops(op_db + custom_op_db, allowed_dtypes=(torch.float,))
    # 跳过名称为 'TestProxyTensorOpInfo' 的测试函数 'test_make_fx_symbolic_exhaustive_inplace'，如果包含在 make_fx_failures、fake_tensor_failures、symbolic_tensor_failures 或 inplace_symbolic_tensor_failures 中
    @skipOps('TestProxyTensorOpInfo', 'test_make_fx_symbolic_exhaustive_inplace',
             make_fx_failures | fake_tensor_failures | symbolic_tensor_failures | inplace_symbolic_tensor_failures)
    def test_make_fx_symbolic_exhaustive_inplace(self, device, dtype, op):
        # 如果操作不支持原地运算，则跳过此测试
        if not op.get_inplace():
            self.skipTest("No inplace variable for this op")
        _test_make_fx_helper(self, device, dtype, op, "symbolic", inplace=True)

    # 使用 op_db 和 custom_op_db 进行测试，只允许 torch.float 数据类型
    @ops(op_db + custom_op_db, allowed_dtypes=(torch.float,))
    # 跳过名称为 'TestProxyTensorOpInfo' 的测试函数 'test_make_fx_symbolic_exhaustive_out'，如果包含在 make_fx_failures、fake_tensor_failures、symbolic_tensor_failures 或 out_symbolic_tensor_failures 中
    @skipOps('TestProxyTensorOpInfo', 'test_make_fx_symbolic_exhaustive_out',
             make_fx_failures | fake_tensor_failures | symbolic_tensor_failures | out_symbolic_tensor_failures)
    def test_make_fx_symbolic_exhaustive_out(self, device, dtype, op):
        # 如果操作不支持 out 参数，则跳过此测试
        if not op.supports_out:
            self.skipTest("Op doesn't support out")
        _test_make_fx_helper(self, device, dtype, op, "symbolic", out=True)


# 只为 "cpu" 设备类型实例化测试类 TestProxyTensorOpInfo 的对象
only_for = ("cpu")
instantiate_device_type_tests(TestProxyTensorOpInfo, globals(), only_for=only_for)

# 如果作为主程序运行，则执行所有测试
if __name__ == '__main__':
    run_tests()
```
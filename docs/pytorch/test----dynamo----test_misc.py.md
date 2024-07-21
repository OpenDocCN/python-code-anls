# `.\pytorch\test\dynamo\test_misc.py`

```
# Owner(s): ["module: dynamo"]

# 引入 Python 标准库和第三方库
import abc  # 抽象基类模块
import collections  # 容器数据类型模块
import copy  # 复制对象模块
import dataclasses  # 数据类模块
import dis  # 解析 Python 字节码模块
import enum  # 枚举模块
import functools  # 函数工具模块
import gc  # 垃圾回收模块
import itertools  # 迭代工具模块
import logging  # 日志模块
import math  # 数学函数模块
import operator  # 运算符函数模块
import os  # 操作系统模块
import random  # 随机数模块
import sys  # 系统相关模块
import tempfile  # 临时文件模块
import threading  # 多线程模块
import traceback  # 调用堆栈模块
import typing  # 类型提示模块
import unittest  # 单元测试模块
import unittest.mock as mock  # 单元测试模块的模拟对象
import warnings  # 警告模块
import weakref  # 弱引用模块
from unittest.mock import patch  # 单元测试模块的模拟对象的补丁

# 引入第三方库
import numpy as np  # 数组处理库

# 引入 PyTorch 相关模块
import torch  # PyTorch 核心库
import torch._dynamo.testing  # PyTorch 内部测试工具

import torch._inductor.test_case  # PyTorch 内部的测试用例
import torch.onnx.operators  # ONNX 操作符模块

import torch.utils._pytree as pytree  # PyTorch 树形数据结构工具
import torch.utils.cpp_extension  # PyTorch C++ 扩展工具
from torch import Tensor  # PyTorch 张量
from torch._C import FileCheck  # PyTorch C++ 扩展工具中的文件检查
from torch._dynamo import allow_in_graph  # PyTorch 动态图中允许的操作
from torch._dynamo.eval_frame import _debug_get_cache_entry_list  # PyTorch 动态图中获取缓存条目列表的调试工具
from torch._dynamo.exc import Unsupported  # PyTorch 动态图中的异常
from torch._dynamo.source import ConstantSource, GetItemSource, LocalSource  # PyTorch 动态图中的源码对象
from torch._dynamo.testing import (  # PyTorch 动态图中的测试工具
    CompileCounter,  # 编译计数器
    CompileCounterWithBackend,  # 带后端的编译计数器
    expectedFailureDynamic,  # 预期的动态失败
    same,  # 相同性测试
    skipIfNotPy311,  # 如果不是 Python 3.11，则跳过测试
    unsupported,  # 不支持的操作
)
from torch._dynamo.utils import (  # PyTorch 动态图中的实用工具
    CompileProfiler,  # 编译分析工具
    counters,  # 计数器集合
    ifdynstaticdefault,  # 动态静态默认情况下的条件语句
)
from torch._inductor.utils import run_and_get_code  # PyTorch 模型导出工具

from torch.ao.quantization import MinMaxObserver  # PyTorch 量化模块中的最小-最大观察器
from torch.ao.quantization.fake_quantize import FakeQuantize  # PyTorch 量化模块中的假量化器
from torch.ao.quantization.qconfig import QConfig  # PyTorch 量化模块中的量化配置
from torch.ao.quantization.quantize_fx import prepare_qat_fx  # PyTorch 量化模块中的准备量化工具
from torch.fx.experimental.recording import NotEqualError, replay_shape_env_events  # PyTorch FX 模块中的记录和重播工具
from torch.fx.experimental.symbolic_shapes import (  # PyTorch FX 模块中的符号形状工具
    _constrain_range_for_size,  # 限制大小范围
    constrain_range,  # 限制范围
    constrain_unify,  # 统一限制
    ConstraintViolationError,  # 约束违规错误
    expect_true,  # 期望为真
    guard_size_oblivious,  # 大小不可知的防护
    ShapeEnv,  # 形状环境
)
from torch.nn import functional as F  # PyTorch 神经网络模块中的功能模块
from torch.testing import make_tensor  # PyTorch 测试工具中的创建张量工具
from torch.testing._internal.common_cuda import (  # PyTorch 内部 CUDA 测试工具
    PLATFORM_SUPPORTS_FLASH_ATTENTION,  # 平台是否支持闪存注意力
    SM80OrLater,  # 是否支持 SM80 或更高版本
    TEST_CUDA,  # 是否测试 CUDA
    TEST_MULTIGPU,  # 是否测试多 GPU
)
from torch.testing._internal.common_methods_invocations import (  # PyTorch 内部的常用方法调用
    sample_inputs_take_along_dim,  # 沿维度取样输入
)
from torch.testing._internal.common_utils import (  # PyTorch 内部的常用实用工具
    freeze_rng_state,  # 冻结随机数生成器状态
    IS_FBCODE,  # 是否为 Facebook 代码
    set_default_dtype,  # 设置默认数据类型
    skipIfNNModuleInlined,  # 如果 NN 模块已内联，则跳过测试
    wrapDeterministicFlagAPITest,  # 包装确定性标志 API 测试
)
from torch.testing._internal.jit_utils import JitTestCase  # PyTorch JIT 工具中的测试用例
from torch.testing._internal.logging_utils import logs_to_string  # PyTorch 日志工具中的日志转换工具

mytuple = collections.namedtuple("mytuple", ["a", "b", "ab"])  # 创建一个命名元组类型 mytuple
T = typing.TypeVar("T")  # 创建一个类型变量 T

# 仅在翻译验证开启时运行的装饰器，用于专门化测试
def onlyIfTranslationValidation(fn: typing.Callable) -> typing.Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        import torch.fx.experimental.validator  # 引入 PyTorch FX 模块中的翻译验证器

        # 如果翻译验证开启，则执行函数
        if torch.fx.experimental.validator.translation_validation_enabled():
            return fn(*args, **kwargs)
        # 否则，抛出跳过测试的异常
        raise unittest.SkipTest(f"only works when TV is True.")

    return wrapper


# 清理操作的函数，用于从操作名中分离命名空间和名称，并检查是否存在该操作
def cleanup_op(opname):
    ns, name = opname.split("::")  # 分离操作名中的命名空间和名称
    if not hasattr(torch.ops, ns):  # 如果 torch.ops 中不存在该命名空间
        return  # 则直接返回
    actual_ns = getattr(torch.ops, ns)  # 否则获取实际的命名空间对象
    # 检查 actual_ns 对象是否没有名为 name 的属性
    if not hasattr(actual_ns, name):
        # 如果没有该属性，则直接返回，不做任何操作
        return
    # 如果 actual_ns 对象存在名为 name 的属性，则删除该属性
    delattr(actual_ns, name)
# 定义一个自定义的 PyTorch 模块，继承自 nn.Module 类
class MyPickledModule(torch.nn.Module):
    # 初始化方法，接收参数 z
    def __init__(self, z):
        super().__init__()
        # 将参数 z 存储为模块的属性
        self.z = z

    # 前向传播方法，接收输入 x 和 y，返回计算结果
    def forward(self, x, y):
        return x * x * x + y + self.z


# 用于测试条件和映射量化的默认对称伪量化设置
default_symmetric_fake_quant = FakeQuantize.with_args(
    observer=MinMaxObserver, qscheme=torch.per_tensor_symmetric, dtype=torch.quint8
)
# 用于权重的默认对称伪量化设置
default_weight_symmetric_fake_quant = FakeQuantize.with_args(
    observer=MinMaxObserver, qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
)
# 定义一个8位量化的均匀量化配置
uniform_qconfig_8bit = QConfig(
    activation=default_symmetric_fake_quant,
    weight=default_weight_symmetric_fake_quant.with_args,
)
# 创建量化配置字典，指定线性层使用上述的均匀量化配置
qconfig_dict = {"object_type": [(torch.nn.Linear, uniform_qconfig_8bit)]}


# 定义一个闭包函数，用于创建带有固定值的加法函数
def closure_adder(val):
    def inner(x):
        return torch.sin(x + val)

    return inner


# 自定义类，用于动态设置属性
class UserDefineSetAttr:
    # 类属性，标识是否已经设置
    setup = False

    # 设置属性的方法，添加了一个断言以确保在动态编译时或已经设置时使用
    def __setattr__(self, key, value):
        assert torch.compiler.is_dynamo_compiling() or UserDefineSetAttr.setup
        super().__setattr__(f"pfx_{key}", value)

    # 获取属性的方法，同样添加了断言以确保在动态编译时或已经设置时使用
    def __getattr__(self, key, c=1):
        assert torch.compiler.is_dynamo_compiling() or UserDefineSetAttr.setup
        # 如果 c 不为 0，则返回相应属性值；否则返回 None
        if c:
            return self.__dict__[f"pfx_{key}"]
        else:
            return None


# 定义一个测试类，继承自 Torch 内部的 TestCase
class MiscTests(torch._inductor.test_case.TestCase):
    # 测试获取缓存条目的方法
    def test_get_cache_entry(self):
        # 定义一个简单的函数 f(x)，返回 x + 1
        def f(x):
            return x + 1

        # 编译函数 f 并传入一个 5x5x5 的张量进行测试
        torch.compile(f)(torch.randn(5, 5, 5))
        # 调用调试函数获取函数 f 的缓存条目列表
        entries = _debug_get_cache_entry_list(f)
        # 断言缓存条目列表长度大于 0
        self.assertTrue(len(entries) > 0)

        # 定义另一个简单函数 g(x)，返回 x + 2
        def g(x):
            return x + 2

        # 获取函数 g 的缓存条目列表
        entries = _debug_get_cache_entry_list(g)
        # 断言缓存条目列表长度为 0
        self.assertTrue(len(entries) == 0)

        # 尝试调用 _debug_get_cache_entry_list 方法传入一个非函数对象 1，预期抛出 TypeError 异常
        try:
            _debug_get_cache_entry_list(1)
        except TypeError as e:
            self.assertIn("expected a code object!", str(e))

        # 测试获取跳过的代码对象的缓存条目
        def h(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 1

        # 编译函数 h 并传入一个 3x3 的张量进行测试
        torch.compile(h)(torch.randn(3, 3))
        # 调用调试函数获取函数 torch._dynamo.graph_break 的缓存条目列表
        entries = _debug_get_cache_entry_list(torch._dynamo.graph_break)
        # 断言缓存条目列表长度为 0
        self.assertEqual(len(entries), 0)
    # 定义一个测试函数，用于测试 boolarg 函数的不同参数情况
    def test_boolarg(self):
        # 内部定义一个函数 boolarg，根据 flag 的真假返回不同的值
        def boolarg(aa, bb, flag):
            # 如果 flag 为真，返回 aa - bb
            if flag:
                return aa - bb
            # 如果 flag 为假，返回 bb - aa
            else:
                return bb - aa

        # 生成两个 10x10 的随机张量 a 和 b
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        # 分别计算 boolarg 的正确返回结果
        correct1 = boolarg(a, b, True)
        correct2 = boolarg(a, b, False)
        correct3 = boolarg(a, b, None)
        
        # 创建一个 CompileCounter 对象
        counter = CompileCounter()
        # 通过装饰器优化 boolarg 函数，得到优化后的版本 opt_boolarg
        opt_boolarg = torch._dynamo.optimize_assert(counter)(boolarg)
        
        # 使用优化后的 boolarg 函数计算不同参数情况下的返回值
        val1 = opt_boolarg(a, b, True)
        val2 = opt_boolarg(a, b, False)
        val3 = opt_boolarg(a, b, None)
        val4 = opt_boolarg(a, b, True)
        
        # 断言优化后的返回值与预期的正确返回值相同
        self.assertTrue(same(val1, correct1))
        self.assertTrue(same(val2, correct2))
        self.assertTrue(same(val3, correct3))
        self.assertTrue(same(val4, correct1))
        # 断言调用 counter 计数的帧数为 3
        self.assertEqual(counter.frame_count, 3)

    # 定义一个测试函数，用于测试带有无效参数的内建函数
    def test_invalid_args_builtin(self):
        # 使用 torch.compile 装饰器，指定后端为 "eager"，定义一个函数 fn
        @torch.compile(backend="eager")
        def fn(x):
            # 对输入张量 x 进行 sin 操作
            x = x.sin()
            # 如果 x 是 torch.Tensor 类型，并且参数 invalid 为 True，则再次进行 sin 操作
            if isinstance(x, torch.Tensor, invalid=True):
                x = x.sin()
            return x
        
        # 断言调用函数 fn 时会抛出 TypeError 异常
        with self.assertRaises(TypeError):
            fn(torch.randn(16))

    # 根据条件跳过测试，如果 NN 模块在内部 CI 环境中内联失败
    @skipIfNNModuleInlined("fails internal CI")
    # 定义一个测试函数，用于测试使用自定义操作的 C++ 扩展
    def test_cpp_extension_recommends_custom_ops(self):
        # 定义一个包含 C++ 源码的字符串，实现了一个名为 foobar 的函数，接受一个张量并返回其克隆
        cpp_source = """
        #include <torch/extension.h>
        at::Tensor foobar(const at::Tensor& x) {
            return x.clone();
        }
        """
        # 使用 torch.utils.cpp_extension.load_inline 加载内联 C++ 源码，创建名为 module 的扩展模块对象
        module = torch.utils.cpp_extension.load_inline(
            name="mylib",
            cpp_sources=cpp_source,
            functions="foobar",
            verbose=True,
        )

        # 创建一个形状为 (2, 2) 的张量 x，并要求计算梯度
        x = torch.ones(2, 2, requires_grad=True)
        # 清空计数器
        counters.clear()

        # 使用 torch.compile 注册一个编译函数 f，其后端为 "eager"
        @torch.compile(backend="eager")
        def f(x):
            # 调用 module 的 foobar 函数处理张量 x
            return module.foobar(x)

        # 使用 assertWarnsOnceRegex 断言捕获到一次 UserWarning，警告内容包含特定 URL
        with self.assertWarnsOnceRegex(
            UserWarning,
            ".*https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html.*",
        ):
            # 调用 f 函数处理张量 x
            f(x)
        # 断言计数器中的 "graph_break" 键对应的值的长度为 1
        self.assertEqual(len(counters["graph_break"]), 1)
        # 获取第一个 "graph_break" 键的具体值
        first_graph_break = list(counters["graph_break"].keys())[0]
        # 使用 assertExpectedInline 断言 first_graph_break 的内容符合预期
        self.assertExpectedInline(
            first_graph_break,
            """Graph break due to unsupported builtin mylib.PyCapsule.foobar. This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind). If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround. If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use torch.compiler.allow_in_graph.""",
        )

        # 定义一个包含 C++ 源码的字符串，实现了一个名为 baz 的函数，接受一个张量并返回其克隆
        cpp_source = """
        #include <torch/extension.h>
        at::Tensor baz(const at::Tensor& x) {
            return x.clone();
        }
        """
        # 使用 torch.utils.cpp_extension.load_inline 加载内联 C++ 源码，创建名为 module2 的扩展模块对象
        module2 = torch.utils.cpp_extension.load_inline(
            name="mylib2",
            cpp_sources=cpp_source,
            functions="baz",
            verbose=True,
        )

        # 重置 torch._dynamo

        # 测试每个警告只会发生一次
        @torch.compile(backend="eager")
        def f(x):
            module2.baz(x)
            module.foobar(x)
            module.foobar(x)
            module2.baz(x)
            module.foobar(x)
            module2.baz(x)
            return x.clone()

        # 使用 warnings.catch_warnings 捕获警告记录
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            # 调用 f 函数处理张量 x
            f(x)
            f(x)
        # 断言警告记录的长度为 2
        self.assertEqual(len(ws), 2)
    # 定义单元测试方法 test_callpacked，测试函数 call_packed 的行为
    def test_callpacked(self):
        # 定义函数 call_packed，接受一个参数 args，解包为 a, b, c，计算并返回 a - b * c 的结果
        def call_packed(args):
            a, b, c = args
            return a - b * c
        
        # 创建一个 CompileCounter 实例
        counter = CompileCounter()
        # 生成两个 10x10 的随机张量 a, b, c
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        c = torch.randn(10, 10)
        # 计算正确结果
        correct = call_packed([a, b, c])
        # 对 call_packed 进行优化，返回优化后的版本 opt_call_packed
        opt_call_packed = torch._dynamo.optimize_assert(counter)(call_packed)
        # 使用优化后的函数计算结果 val1, val2, val3, val4
        val1 = opt_call_packed([a, b, c])
        val2 = opt_call_packed((a, b, c))
        val3 = opt_call_packed([a, b, c])
        val4 = opt_call_packed((a, b, c))
        # 断言优化后的结果与正确结果相同
        self.assertTrue(same(val1, correct))
        self.assertTrue(same(val2, correct))
        self.assertTrue(same(val3, correct))
        self.assertTrue(same(val4, correct))
        # 断言 frame_count 属性为 2，表明优化计数器记录了两个帧
        self.assertEqual(counter.frame_count, 2)

    # 定义单元测试方法 test_raises，测试函数 fn 在特定条件下是否引发异常
    def test_raises(self):
        # 定义函数 fn，接受参数 a, b, c, cls，计算并抛出带有结果的异常
        def fn(a, b, c, cls):
            x = a + b - c * 10
            raise cls(str(x))
        
        # 创建一个 CompileCounter 实例
        counter = CompileCounter()
        # 生成两个 10x10 的随机张量 a, b, c
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        c = torch.randn(10, 10)
        # 对 fn 进行优化，返回优化后的版本 opt_fn
        opt_fn = torch._dynamo.optimize(counter)(fn)
        # 使用 lambda 表达式调用 opt_fn，断言其引发 AssertionError 异常
        self.assertRaises(AssertionError, lambda: opt_fn(a, b, c, AssertionError))
        # 断言 frame_count 属性为 1，表明优化计数器记录了一个帧
        self.assertEqual(counter.frame_count, 1)
        # 断言 op_count 属性为 3，表明优化计数器记录了三个操作
        self.assertEqual(counter.op_count, 3)

    # 定义单元测试方法 test_module_not_callable，测试在非法调用时函数 fn 是否引发特定类型的异常
    def test_module_not_callable(self):
        # 定义函数 fn，接受参数 x，调用 torch.fft(x) 并返回结果
        def fn(x):
            return torch.fft(x)
        
        # 创建一个 CompileCounter 实例
        counter = CompileCounter()
        # 生成一个 10x10 的随机张量 a
        a = torch.randn(10, 10)
        # 对 fn 进行优化，返回优化后的版本 opt_fn
        opt_fn = torch._dynamo.optimize(counter)(fn)
        # 使用 lambda 表达式调用 opt_fn，断言其引发 TypeError 异常并包含指定字符串
        self.assertRaisesRegex(
            TypeError, "'module' object is not callable", lambda: opt_fn(a)
        )

    # 定义单元测试方法 test_inplace，测试函数 inplace1 的行为
    def test_inplace(self):
        # 定义函数 inplace1，接受参数 a, b，创建一个空张量 o，将 a 复制到 o 后减去 b，返回 o
        def inplace1(a, b):
            o = torch.empty((10, 10))
            o.copy_(a)
            o -= b
            return o
        
        # 调用 torch._dynamo.testing.standard_test 测试 inplace1 函数，期望操作数为 3
        torch._dynamo.testing.standard_test(self, inplace1, 2, expected_ops=3)

    # 定义单元测试方法 test_inplace_desugaring，测试函数 inplace_on_literals 的行为
    def test_inplace_desugaring(self):
        # 定义函数 inplace_on_literals，接受参数 y，操作两个字面值 x0 和 x1 并返回它们
        def inplace_on_literals(y):
            x0 = 1
            x0 += y
            x1 = 1
            x1 -= y
            return x0, x1
        
        # 调用 torch._dynamo.testing.standard_test 测试 inplace_on_literals 函数，期望操作数为 2
        torch._dynamo.testing.standard_test(
            self, inplace_on_literals, 1, expected_ops=2
        )

    # 定义单元测试方法 test_unpack4，测试函数 unpack4 的行为
    def test_unpack4(self):
        # 定义函数 unpack4，接受参数 a, b，截取部分张量并将结果存储在 o 中，返回 o
        def unpack4(a, b):
            a = a[:5, :]
            b = b[:5, :]
            x, y = a.size()
            o = torch.empty((x, y))
            o.copy_(a / b)
            return o
        
        # 调用 torch._dynamo.testing.standard_test 测试 unpack4 函数，期望静态操作数为 5，动态操作数为 7
        torch._dynamo.testing.standard_test(
            self,
            unpack4,
            2,
            expected_ops=5,
            expected_ops_dynamic=ifdynstaticdefault(5, 7),
        )

    # 定义单元测试方法 test_unpack5，测试函数 unpack5 的行为
    def test_unpack5(self):
        # 定义函数 unpack5，接受参数 a, b，截取部分张量并将结果存储在 o 中，返回 o
        def unpack5(a, b):
            a = a[:5, :]
            b = b[:5, :]
            x, y = a.shape
            o = torch.empty((x, y))
            o.copy_(a / b)
            return o
        
        # 调用 torch._dynamo.testing.standard_test 测试 unpack5 函数，期望静态操作数为 5，动态操作数为 7
        torch._dynamo.testing.standard_test(
            self,
            unpack5,
            2,
            expected_ops=5,
            expected_ops_dynamic=ifdynstaticdefault(5, 7),
        )
    # 定义一个测试方法，测试矩阵乘法运算的第一种实现方式
    def test_matmul1(self):
        # 定义一个矩阵乘法操作函数
        def matmul_op1(a, b):
            return a @ b

        # TODO(jansel): FX doesn't support this, should add upstream support
        # 使用 torch._dynamo.testing.standard_test 方法执行标准测试，
        # 期望操作数为1
        torch._dynamo.testing.standard_test(self, matmul_op1, 2, expected_ops=1)

    # 定义一个测试方法，测试整数形状的二进制操作
    def test_int_shape_binops(self):
        # 定义一个函数 fn，接受参数 x
        def fn(x):
            # 测试通过将整数参数放在首位来翻转操作
            y = 15 - x.shape[0]  # 计算 x 的形状长度与 15 的差
            y = 4 + y  # 加 4
            y = 5 * y  # 乘以 5
            y = 2 % y  # 取余数
            y = 3**y  # 指数运算
            y = 10 // y  # 整除运算
            y = pow(2, y)  # 幂运算
            y = 10 / y  # 除以 y
            return x + y  # 返回 x 加上 y

        # 使用 torch._dynamo.testing.standard_test 方法执行标准测试，
        # 期望操作数为1，动态操作数为 ifdynstaticdefault(1, 11)
        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=ifdynstaticdefault(1, 11)
        )

    # 标记方法，使用 torch._dynamo.config.patch 配置，仅允许 pt2 兼容的操作
    @torch._dynamo.config.patch(only_allow_pt2_compliant_ops=True)
    def test_pt2_compliant_ops_are_allowed(self):
        # 创建一个名为 "mylib" 的库对象
        lib = torch.library.Library("mylib", "FRAGMENT")
        try:
            # 定义一个名为 "mylib::bar" 的库函数，接受参数 (Tensor x)，返回 Tensor
            torch.library.define(
                "mylib::bar",
                "(Tensor x) -> Tensor",
                lib=lib,
                tags=(torch.Tag.pt2_compliant_tag,),
            )
            # 实现 "mylib::bar" 函数，使用 CompositeImplicitAutograd 方式包装 torch.sin 函数
            torch.library.impl(
                "mylib::bar", "CompositeImplicitAutograd", torch.sin, lib=lib
            )
            # 断言默认的 "mylib::bar" 函数包含 pt2_compliant_tag 标签
            assert torch.Tag.pt2_compliant_tag in torch.ops.mylib.bar.default.tags

            # 定义函数 f(x)，调用 torch.ops.mylib.bar(x)
            def f(x):
                return torch.ops.mylib.bar(x)

            # 获取 torch.ops.mylib.bar 的重载版本
            overload = torch.ops.mylib.bar.default

            # 定义函数 g(x)，调用 overload(x)
            def g(x):
                return overload(x)

            # 生成一个包含三个随机数的张量 x
            x = torch.randn(3)

            # 创建一个编译计数器对象 counts
            counts = torch._dynamo.testing.CompileCounter()
            # 优化函数 f，使用 torch._dynamo.optimize 进行优化，启用 nopython 模式
            optimized_f = torch._dynamo.optimize(counts, nopython=True)(f)
            _ = optimized_f(x)

            # 优化函数 g，使用相同的编译计数器 counts 和 nopython 模式
            optimized_g = torch._dynamo.optimize(counts, nopython=True)(f)
            _ = optimized_g(x)
        finally:
            # 清理 "mylib::bar" 的操作
            cleanup_op("mylib::bar")
            # 删除库对象 lib
            del lib

    # 标记方法，使用 torch._dynamo.config.patch 配置，仅允许 pt2 兼容的操作
    @torch._dynamo.config.patch(only_allow_pt2_compliant_ops=True)
    # 定义一个测试函数，用于测试非 PT2 兼容操作图中的断点
    def test_non_pt2_compliant_ops_graph_break(self):
        # 创建一个名为 "mylib" 的 Torch 库对象，类型为 "FRAGMENT"
        lib = torch.library.Library("mylib", "FRAGMENT")
        try:
            # 定义一个名为 "mylib::bar2" 的 Torch 操作，接受输入参数是 Tensor 类型，返回值也是 Tensor 类型，使用上述创建的库
            torch.library.define("mylib::bar2", "(Tensor x) -> Tensor", lib=lib)
            # 将 "mylib::bar2" 操作实现为 "CompositeImplicitAutograd" 类型，实现函数为 torch.sin，使用上述创建的库
            torch.library.impl(
                "mylib::bar2", "CompositeImplicitAutograd", torch.sin, lib=lib
            )
            # 断言 "torch.Tag.pt2_compliant_tag" 不在 "torch.ops.mylib.bar2.default.tags" 中
            assert torch.Tag.pt2_compliant_tag not in torch.ops.mylib.bar2.default.tags

            # 定义函数 f(x)，调用 "torch.ops.mylib.bar2" 操作
            def f(x):
                return torch.ops.mylib.bar2(x)

            # 获取 "torch.ops.mylib.bar2.default" 的重载
            overload = torch.ops.mylib.bar2.default

            # 定义函数 g(x)，调用上述获取的重载
            def g(x):
                return overload(x)

            # 创建一个大小为 3 的随机张量 x
            x = torch.randn(3)

            # 创建一个编译计数器实例 counts
            counts = torch._dynamo.testing.CompileCounter()
            # 使用断言检测是否抛出了 torch._dynamo.exc.Unsupported 异常，并包含 "not PT2 compliant" 的错误信息
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "not PT2 compliant"
            ):
                # 对函数 f 进行优化处理，要求不使用 Python 代码（nopython=True），然后执行优化后的函数，计算结果 y
                optimized_f = torch._dynamo.optimize(counts, nopython=True)(f)
                y = optimized_f(x)

            # 再次使用断言检测是否抛出了 torch._dynamo.exc.Unsupported 异常，并包含 "not PT2 compliant" 的错误信息
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "not PT2 compliant"
            ):
                # 对函数 g 进行优化处理，要求不使用 Python 代码（nopython=True），然后执行优化后的函数，计算结果 y
                optimized_g = torch._dynamo.optimize(counts, nopython=True)(f)
                y = optimized_g(x)
        finally:
            # 清理操作，删除 "mylib::bar2" 操作
            cleanup_op("mylib::bar2")
            # 删除 Torch 库对象 lib
            del lib

    # 使用 Torch 动态图配置修补装饰器，仅允许 PT2 兼容操作
    @torch._dynamo.config.patch(only_allow_pt2_compliant_ops=True)
    # 测试函数，验证是否符合PT2的重载要求
    def test_pt2_compliant_overload(self):
        # 创建一个名为"mylib"的库，并设置为"FRAGMENT"类型
        lib = torch.library.Library("mylib", "FRAGMENT")
        try:
            # 定义名为"mylib::bar3.tensor"的操作，接受一个Tensor类型的参数并返回Tensor
            torch.library.define(
                "mylib::bar3.tensor",
                "(Tensor x) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,  # 添加PT2兼容标签
                lib=lib,
            )
            # 定义名为"mylib::bar3.int"的操作，接受一个Tensor类型和一个整数类型参数，并返回Tensor
            torch.library.define(
                "mylib::bar3.int", "(Tensor x, int dim) -> Tensor", lib=lib
            )

            # 实现"mylib::bar3.tensor"操作，使用CompositeImplicitAutograd方式实现torch.sin函数
            torch.library.impl(
                "mylib::bar3.tensor",
                "CompositeImplicitAutograd",
                torch.sin,
                lib=lib,
            )
            # 实现"mylib::bar3.int"操作，使用CompositeImplicitAutograd方式实现torch.sum函数
            torch.library.impl(
                "mylib::bar3.int", "CompositeImplicitAutograd", torch.sum, lib=lib
            )

            # 定义函数f，调用torch.ops.mylib.bar3操作
            def f(x):
                return torch.ops.mylib.bar3(x)

            # 定义函数g，调用torch.ops.mylib.bar3操作，传入一个额外的整数参数
            def g(x):
                return torch.ops.mylib.bar3(x, 1)

            # 定义函数h，调用torch.ops.mylib.bar3操作，传入三个相同的参数
            def h(x):
                return torch.ops.mylib.bar3(x, x, x)

            # 生成一个形状为(3,)的随机Tensor
            x = torch.randn(3)

            # 创建一个编译计数器对象
            counts = torch._dynamo.testing.CompileCounter()
            # 对函数f进行优化，开启无Python模式，并返回优化后的版本
            optimized_f = torch._dynamo.optimize(counts, nopython=True)(f)
            # 对函数g进行优化，开启无Python模式，并返回优化后的版本
            optimized_g = torch._dynamo.optimize(counts, nopython=True)(g)
            # 对函数h进行优化，开启无Python模式，并返回优化后的版本
            optimized_h = torch._dynamo.optimize(counts, nopython=True)(h)

            # 测试优化后的函数optimized_f是否能正常运行，不应该引发错误（符合PT2规范）
            optimized_f(x)

            # 测试优化后的函数optimized_g是否能正常运行，预期引发torch._dynamo.exc.Unsupported异常并包含指定信息
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "not PT2 compliant"
            ):
                y = optimized_g(x)

            # 测试优化后的函数optimized_h是否能正常运行，预期引发torch._dynamo.exc.Unsupported异常并包含指定信息
            with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "failed to"):
                y = optimized_h(x)

        finally:
            # 清理名为"mylib::bar3"的操作
            cleanup_op("mylib::bar3")
            # 删除lib对象
            del lib

    # 测试函数，验证是否支持默认情况下的自动功能化
    def test_auto_functionalize_can_with_default(self):
        # 创建一个名为"mylib"的库，并设置为"FRAGMENT"类型
        lib = torch.library.Library("mylib", "FRAGMENT")
        # 定义名为"mylib::foo"的操作，接受Tensor、整数、可选Tensor和可选整数参数，并无返回值
        torch.library.define(
            "mylib::foo",
            "(Tensor a, int b, Tensor(d!)? c=None, Tensor? d=None, int e=-1) -> ()",
            tags=torch.Tag.pt2_compliant_tag,  # 添加PT2兼容标签
            lib=lib,
        )

        # 实现"mylib::foo"操作，使用CPU模式，并调用a + b操作，无返回值
        @torch.library.impl("mylib::foo", "cpu", lib=lib)
        def foo_impl(a, b, c=None, d=None, e=-1):
            a + b
            return

        # 定义函数f，调用torch.ops.mylib.foo操作
        def f(a, mode):
            return torch.ops.mylib.foo(
                a,
                0,
            )

        # 创建一个形状为[10, 10, 10]，数据类型为torch.int64的Tensor
        a = torch.tensor([10, 10, 10], dtype=torch.int64)

        # 编译函数f，并调用编译后的版本，不应引发异常
        torch.compile(f)(a, 0)

        # 清理名为"mylib::foo"的操作
        cleanup_op("mylib::foo")
        # 删除lib对象
        del lib
    def test_auto_functionalize_can_with_none_return(self):
        # 使用 torch.library._scoped_library 创建名为 "mylib" 的库对象，作用域为 "FRAGMENT"
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            # 在库中定义函数 "foo"，接受两个参数 Tensor x 和 Tensor(a!) out，返回 None
            lib.define("foo(Tensor x, Tensor(a!) out) -> None")

            # 定义 foo_impl 函数，将输入 x 复制到输出 out 中
            def foo_impl(x, out):
                out.copy_(x)

            # 将 foo_impl 函数注册为名为 "foo" 的操作的实现，使用 "CompositeExplicitAutograd" 后端
            lib.impl("foo", foo_impl, "CompositeExplicitAutograd")

            # 生成随机张量 x 和全零张量 out
            x = torch.randn(3)
            out = torch.zeros(3)

            # 使用 torch.compile 装饰器编译 f 函数
            @torch.compile
            def f(x, out):
                torch.ops.mylib.foo(x, out)

            # 调用 f 函数
            f(x, out)

    def test_user_defined_setattr1(self):
        # 使用 torch.compile 装饰器编译 fn 函数，设置 backend="eager" 和 fullgraph=True
        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            # 将 obj 的属性 y 设置为 obj 的属性 x 加 1
            obj.y = obj.x + 1

        # 创建 UserDefineSetAttr 对象 obj
        obj = UserDefineSetAttr()

        # 使用 patch.object 修改 UserDefineSetAttr 类的 setup 属性为 True
        with patch.object(UserDefineSetAttr, "setup", True):
            # 设置 obj 的属性 x 为一个随机张量
            obj.x = torch.randn(8)

        # 调用 fn 函数
        fn(obj)

        # 使用 patch.object 修改 UserDefineSetAttr 类的 setup 属性为 True
        with patch.object(UserDefineSetAttr, "setup", True):
            # 断言 obj 的属性 y 等于 obj 的属性 x 加 1
            self.assertEqual(obj.y, obj.x + 1)

        # 断言 obj 的 __dict__ 中的键包含 "pfx_x" 和 "pfx_y"
        self.assertEqual(obj.__dict__.keys(), {"pfx_x", "pfx_y"})

    def test_user_defined_setattr2(self):
        # 使用 torch.compile 装饰器编译 fn 函数，设置 backend="eager" 和 fullgraph=True
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # 创建 UserDefineSetAttr 对象 obj
            obj = UserDefineSetAttr()

            # 设置 obj 的属性 x 为输入的 x
            obj.x = x

            # 设置 obj 的属性 y 为 obj 的属性 x 加 1
            obj.y = obj.x + 1

            return obj

        # 生成一个随机张量 x
        x = torch.randn(8)

        # 调用 fn 函数
        obj = fn(x)

        # 使用 patch.object 修改 UserDefineSetAttr 类的 setup 属性为 True
        with patch.object(UserDefineSetAttr, "setup", True):
            # 断言 obj 的属性 x 与输入的 x 相等
            self.assertIs(obj.x, x)

            # 断言 obj 的属性 y 等于输入的 x 加 1
            self.assertEqual(obj.y, x + 1)

        # 断言 obj 的 __dict__ 中的键包含 "pfx_x" 和 "pfx_y"
        self.assertEqual(obj.__dict__.keys(), {"pfx_x", "pfx_y"})

    def test_closure_recompiles(self):
        # 创建 CompileCounter 对象 cnt
        cnt = CompileCounter()

        # 定义 fn 函数，接受两个参数 x 和 other_fn
        def fn(x, other_fn):
            # 返回调用 other_fn 函数并对结果减 1 的值
            return other_fn(x + 1) - 1

        # 使用 torch.compile 编译 fn 函数，设置 backend=cnt 和 fullgraph=True
        opt = torch.compile(fn, backend=cnt, fullgraph=True)

        # 生成一个随机张量 x
        x = torch.randn(8)

        # 遍历四个不同的闭包函数 f，并比较其结果与 fn 函数的结果
        for f in (
            closure_adder(5),
            closure_adder(5),
            closure_adder(torch.randn(8)),
            closure_adder(torch.randn(8)),
        ):
            self.assertEqual(opt(x, f), fn(x, f))

        # 断言 CompileCounter 对象 cnt 的 frame_count 属性为 2
        self.assertEqual(cnt.frame_count, 2)

    def test_generate_trivial_abstract_impl(self):
        try:
            # 创建名为 "mylib" 的库对象 lib
            lib = torch.library.Library("mylib", "FRAGMENT")

            # 在库 lib 中定义函数 "mylib::foo"，定义标记为 torch.Tag.pt2_compliant_tag
            lib.define(
                "mylib::foo",
                "(Tensor x, Tensor[] y, Tensor(a!)? z, SymInt w) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            # 使用 torch.library.impl 装饰器注册 foo_impl 函数为 "mylib::foo" 的实现
            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y, z, w):
                # 对输入的张量进行加法操作
                x + y[0] + w
                return

            # 定义 f 函数，调用 "mylib::foo" 操作
            def f(x, y, z, w):
                return torch.ops.mylib.foo(x, y, z, 2)

            # 生成随机张量 x、y、z 和 w
            x = torch.randn(3)
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            w = torch.randn(3)
            args = (x, y, z, w)

            # 使用 torch.compile 编译 f 函数，设置 backend="eager" 和 fullgraph=True
            output = torch.compile(f, backend="eager", fullgraph=True)(*args)

            # 断言 output 的结果为 None
            self.assertEqual(output, None)
        finally:
            # 清理 "mylib::foo" 操作
            cleanup_op("mylib::foo")
            # 删除库对象 lib
            del lib
    # 定义一个测试方法，用于测试自动函数化功能
    def test_can_auto_functionalize(self):
        # 导入所需模块和函数
        from torch._higher_order_ops.auto_functionalize import can_auto_functionalize
        
        # 期望返回 True 的函数签名列表
        expected_true = [
            "(Tensor(a!) x) -> ()",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> ()",
            "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> ()",
            "(Tensor(a!) x, Tensor y, Tensor(b!)[] z, SymInt w) -> ()",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> Tensor",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> (Tensor, Tensor)",
        ]
        # 期望返回 False 的函数签名列表
        expected_false = [
            "(Tensor x) -> ()",
            "(Tensor(a) x) -> Tensor(a)",
            "(Tensor(a!) x) -> Tensor(a!)",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> Tensor(a)",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> (Tensor, Tensor(a))",
            "(Tensor(a) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> (Tensor, Tensor(a))",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> (Tensor, Tensor[])",
        ]
        
        # 对期望返回 True 的函数签名进行测试
        for schema in expected_true:
            try:
                # 创建一个名为 "mylib" 的库对象
                lib = torch.library.Library("mylib", "FRAGMENT")
                # 定义名为 "mylib::a" 的操作，并使用给定的 schema
                torch.library.define("mylib::a", schema, lib=lib)
                # 测试是否可以自动函数化该操作，默认情况下应返回 True
                self.assertTrue(
                    can_auto_functionalize(torch.ops.mylib.a.default), msg=schema
                )
                # 再次确认该操作不能自动函数化，应返回 False
                self.assertFalse(can_auto_functionalize(torch.ops.mylib.a))
            finally:
                # 清理操作定义
                cleanup_op("mylib::a")
                # 删除库对象
                del lib
        
        # 对期望返回 False 的函数签名进行测试
        for schema in expected_false:
            try:
                # 创建一个名为 "mylib" 的库对象
                lib = torch.library.Library("mylib", "FRAGMENT")
                # 定义名为 "mylib::a" 的操作，并使用给定的 schema
                torch.library.define("mylib::a", schema, lib=lib)
                # 测试是否可以自动函数化该操作，默认情况下应返回 False
                self.assertFalse(
                    can_auto_functionalize(torch.ops.mylib.a.default), msg=schema
                )
                # 再次确认该操作不能自动函数化，应返回 False
                self.assertFalse(can_auto_functionalize(torch.ops.mylib.a))
            finally:
                # 清理操作定义
                cleanup_op("mylib::a")
                # 删除库对象
                del lib
    def test_auto_functionalize(self):
        try:
            # 创建一个名为 "mylib" 的 Torch 库对象，类型为 "FRAGMENT"
            lib = torch.library.Library("mylib", "FRAGMENT")
            # 定义一个名为 "mylib::foo" 的 Torch 库函数签名
            # 这个函数接受一些张量参数和一个符号整数参数，无返回值
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> ()",
                tags=torch.Tag.pt2_compliant_tag,  # 添加标签以便与 PyTorch 2 兼容
                lib=lib,  # 将函数定义与之前创建的库对象关联
            )

            # 实现 "mylib::foo" 函数的具体功能，用于 CPU
            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable  # 禁用 Torch 的动态运行时
            def foo_impl(x, y, z, w, n):
                # 实现函数功能：x += y[0] + w，z += y[1] + n
                x.add_(y[0] + w)
                z.add_(y[1] + n)

            # 定义一个包装函数 f，调用 Torch 操作符调用库函数 "mylib::foo"
            def f(x, y, z, n):
                torch.ops.mylib.foo(x, y, z, 2, n)

            # 初始化张量参数
            x = torch.randn(3)
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            n = torch.randn(3)
            orig_args = (x, y, z, n)

            # 对原始参数进行复制操作，以备份
            compiled_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)

            # 创建日志流和上下文对象 ctx
            log_stream, ctx = logs_to_string(
                "torch._inductor.compile_fx", "post_grad_graphs"
            )

            # 进入 ctx 上下文环境
            with ctx():
                # 编译函数 f 到静态图中
                torch.compile(f, backend="inductor", fullgraph=True)(*compiled_args)

            # 从日志流中提取编译后的图形信息并格式化输出
            post_grad_graphs = "\n".join(
                log_stream.getvalue().strip().split("\n")[3:]
            ).strip()

            # 在假设默认情况下使用静态形状的情况下，检查生成的静态图
            if torch._dynamo.config.assume_static_by_default:
                # 使用 self.assertExpectedInline 检查生成的静态图是否符合预期
                self.assertExpectedInline(
                    post_grad_graphs,
                    """\
    def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu", arg4_1: "f32[3][1]cpu"):
        # 调用自定义的 Torch 操作 'mylib::foo.default' 进行前向传播计算
        foo_default = torch.ops.mylib.foo.default(arg4_1, [arg2_1, arg3_1], arg1_1, 2, arg0_1);  arg4_1 = arg2_1 = arg3_1 = arg1_1 = arg0_1 = None
        return ()""",
                )

            # 将原始参数转换为 eager 模式的 Torch 张量
            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            # 调用函数 f，并传入 eager 模式的参数
            f(*eager_args)
            # 断言编译后的参数与 eager 模式的参数相等
            self.assertEqual(compiled_args, eager_args)
        finally:
            # 清理名为 'mylib::foo' 的 Torch 库定义的操作
            cleanup_op("mylib::foo")
            # 删除 Torch 库实例
            del lib

    def test_auto_functionalize_with_returns(self):
        try:
            # 创建名为 'mylib' 的 Torch 库实例，类型为 'FRAGMENT'
            lib = torch.library.Library("mylib", "FRAGMENT")
            # 定义 Torch 操作 'mylib::foo'，指定签名和标签
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> (Tensor, Tensor)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            # 实现 'mylib::foo' 操作的具体逻辑，对输入进行计算并返回结果
            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y, z, w, n):
                x.add_(y[0] + w)
                z.add_(y[1] + n)
                return y[0] + w, y[1] + n

            # 定义 'mylib::foo' 操作的抽象逻辑，返回计算结果
            @torch.library.impl_abstract("mylib::foo", lib=lib)
            def foo_abstract(x, y, z, w, n):
                return y[0] + w, y[1] + n

            # 定义函数 f，调用 'mylib::foo' 操作进行计算
            def f(x, y, z, n):
                return torch.ops.mylib.foo(x, y, z, 2, n)

            # 创建测试数据
            x = torch.randn(3)
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            n = torch.randn(3)
            orig_args = (x, y, z, n)

            # 将原始参数转换为 eager 模式的 Torch 张量，并复制一份
            compiled_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            # 将日志转换为字符串，并获取上下文对象
            log_stream, ctx = logs_to_string(
                "torch._inductor.compile_fx", "post_grad_graphs"
            )
            # 在上下文中进行编译，生成编译后的输出
            with ctx():
                compiled_out = torch.compile(f, backend="inductor", fullgraph=True)(
                    *compiled_args
                )

            # 如果默认情况下假定为静态，则获取编译后的图形日志，并断言结果符合预期
            if torch._dynamo.config.assume_static_by_default:
                post_grad_graphs = "\n".join(
                    log_stream.getvalue().strip().split("\n")[3:]
                ).strip()
                self.assertExpectedInline(
                    post_grad_graphs,
                    """\
    def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu", arg4_1: "f32[3][1]cpu"):
        # 调用自定义的 Torch 操作 'mylib::foo.default' 进行前向传播计算
        foo_default = torch.ops.mylib.foo.default(arg4_1, [arg2_1, arg3_1], arg1_1, 2, arg0_1);  arg4_1 = arg2_1 = arg3_1 = arg1_1 = arg0_1 = None
        # 从 'foo_default' 结果中获取第一个元素
        getitem_4: "f32[3][1]cpu" = foo_default[0]
        # 从 'foo_default' 结果中获取第二个元素；清除 'foo_default' 变量
        getitem_5: "f32[3][1]cpu" = foo_default[1];  foo_default = None
        # 返回前向传播的结果，包括两个获取的元素
        return (getitem_4, getitem_5)
    def test_auto_functionalize_optional(self):
        # 尝试定义一个名为 "mylib" 的 Torch 库，并指定其类型为 "FRAGMENT"
        try:
            lib = torch.library.Library("mylib", "FRAGMENT")
            # 定义函数签名为 "mylib::foo"，接受特定参数并返回空
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!)? x, Tensor[] y, Tensor(b!)? z, SymInt w, Tensor n) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            # 实现名为 "mylib::foo" 的函数，用于处理输入参数
            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y, z, w, n):
                # 如果 x 不为空，则将其修改为 y[0] + w
                if x is not None:
                    x.add_(y[0] + w)
                # 如果 z 不为空，则将其修改为 y[1] + n
                if z is not None:
                    z.add_(y[1] + n)

            # 定义函数 f，调用 Torch 操作 "mylib::foo"
            def f(x, y, z, n):
                torch.ops.mylib.foo(x, y, z, 2, n)

            # 初始化变量 x, y, z, n
            x = None
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            n = torch.randn(3)
            orig_args = (x, y, z, n)

            # 使用 pytree 库对 orig_args 中的 Tensor 对象进行克隆处理
            compiled_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            # 将日志输出流和上下文对象赋值给 log_stream 和 ctx
            log_stream, ctx = logs_to_string(
                "torch._inductor.compile_fx", "post_grad_graphs"
            )
            # 在上下文中调用 torch.compile 函数，使用 "inductor" 后端和完整图形编译模式
            with ctx():
                torch.compile(f, backend="inductor", fullgraph=True)(*compiled_args)

            # 如果 assume_static_by_default 为真，则从日志中提取 post_grad_graphs
            if torch._dynamo.config.assume_static_by_default:
                post_grad_graphs = "\n".join(
                    log_stream.getvalue().strip().split("\n")[3:]
                ).strip()
                # 断言 post_grad_graphs 与期望输出相符
                self.assertExpectedInline(
                    post_grad_graphs,
                    """\
    # 定义一个方法，接受四个参数，每个参数都是一个形状为 (3, 1) 的 f32 类型的 CPU 张量
    def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu"):
        # 调用自定义的 Torch 操作 "torch.ops.mylib.foo.default"，传递参数并接收返回值到 foo_default
        foo_default = torch.ops.mylib.foo.default(None, [arg2_1, arg3_1], arg1_1, 2, arg0_1);  arg2_1 = arg3_1 = arg1_1 = arg0_1 = None
        # 返回一个空的元组
        return ()""",
                )

            # 创建 eager_args，使用 pytree.tree_map_only 将原始参数 orig_args 中的每个元素转换为 torch.Tensor，并使用 torch.clone 复制
            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            # 调用函数 f，传递 eager_args 中的参数
            f(*eager_args)
            # 断言 compiled_args 等于 eager_args
            self.assertEqual(compiled_args, eager_args)
        finally:
            # 清理名为 "mylib::foo" 的操作
            cleanup_op("mylib::foo")
            # 删除 lib 对象

    # 定义测试函数 test_auto_functionalize_tensorlist
    def test_auto_functionalize_tensorlist(self):
        # 进入 torch 库的作用域，使用名为 "mylib" 的库和 "FRAGMENT" 片段
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            # 定义一个 Torch 库函数 "mylib::foo"，指定其参数和标签
            torch.library.define(
                "mylib::foo",
                "(Tensor all_gather_output, SymInt[] all_gather_input_split_sizes, int dim, Tensor(a!)[] out) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            # 定义 foo_impl 函数作为 "mylib::foo" 的实现，用于复制 all_gather_output 到 out 中的每个张量
            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(all_gather_output, all_gather_input_split_sizes, dim, out):
                for o in out:
                    o.copy_(all_gather_output)

            # 定义函数 f，调用 "mylib::foo" 操作，传递参数
            def f(all_gather_output, all_gather_input_split_sizes, dim, out):
                torch.ops.mylib.foo(
                    all_gather_output, all_gather_input_split_sizes, dim, out
                )

            # 创建测试参数
            a = torch.ones(4)
            b = [2, 3]
            c = 0
            d = [torch.empty(4) for _ in range(2)]
            orig_args = (a, b, c, d)

            # 编译函数 f，使用编译器后端 "inductor"，完整图形为 True
            compiled_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            torch.compile(f, backend="inductor", fullgraph=True)(*compiled_args)

            # 创建 eager_args，同样处理原始参数
            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            # 调用函数 f，传递 eager_args 中的参数
            f(*eager_args)
            # 断言 compiled_args 等于 eager_args
            self.assertEqual(compiled_args, eager_args)

    # 定义测试函数 test_shape_int_inplace_binops
    def test_shape_int_inplace_binops(self):
        # 定义函数 fn，接受一个参数 x
        def fn(x):
            # 获取 x 的第一个维度大小，并赋值给变量 p
            p = x.shape[0]
            # p 增加 2
            p += 2
            # p 减去 2
            p -= 2
            # p 平方
            p **= 2
            # p 除以 2
            p /= 2
            # p 乘以 2
            p *= 2
            # p 取整除以 2
            p //= 2
            # p 取模 2
            p %= 2
            # 返回 x 加上 p 的结果
            return x + p

        # 使用 torch._dynamo.testing.standard_test 运行标准测试
        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=ifdynstaticdefault(1, 10)
        )

    # 定义测试函数 test_int_shape_inplace_binops
    def test_int_shape_inplace_binops(self):
        # 定义函数 fn，接受一个参数 x
        def fn(x):
            # 获取 x 的第一个维度大小，并赋值给变量 p
            p = x.shape[0]
            # 使用常量 2 进行加法操作
            y = 2
            y += p
            # 使用常量 2 进行减法操作
            y = 2
            y -= p
            # 使用常量 2 进行幂运算操作
            y = 2
            y **= p
            # 使用常量 2 进行除法操作
            y = 2
            y /= p
            # 使用常量 2 进行乘法操作
            y = 2
            y *= p
            # 使用常量 2 进行整除操作
            y = 2
            y //= p
            # 使用常量 2 进行取模操作
            y = 2
            y %= p
            # 返回 x 加上 y 的结果
            return x + y

        # 使用 torch._dynamo.testing.standard_test 运行标准测试
        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=ifdynstaticdefault(1, 4)
        )
    def test_int_int_comparisons(self):
        def fn(x):
            # 检查常量与常量比较，此处不成立
            if 2 != 2:
                out = 1
            # 检查常量小于常量，此处不成立
            elif 2 < 1:
                out = 1
            # 检查常量大于常量，此处不成立
            elif 1 > 2:
                out = 1
            # 检查常量大于或等于常量，此处不成立
            elif 1 >= 2:
                out = 1
            # 检查常量小于或等于常量，此处不成立
            elif 2 <= 1:
                out = 1
            # 检查常量等于常量，此处成立，将 out 设置为 2
            elif 2 == 2:
                out = 2
            else:
                out = 1
            # 返回 x 与 out 的和
            return x + out

        # 使用标准测试函数对 fn 进行测试，期望操作数为 1
        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_shape_int_comparisons(self):
        def fn(x):
            a = x.shape[0]
            # 检查数组形状的常量与常量比较，此处不成立
            if a != 10:
                out = 1
            # 检查数组形状的常量小于常量，此处成立，将 out 设置为 1
            elif a < 2:
                out = 1
            # 检查数组形状的常量大于常量，此处不成立
            elif a > 12:
                out = 1
            # 检查数组形状的常量大于或等于常量，此处成立，将 out 设置为 1
            elif a >= 12:
                out = 1
            # 检查数组形状的常量小于或等于常量，此处不成立
            elif a <= 2:
                out = 1
            # 检查数组形状的常量等于常量，此处成立，将 out 设置为 2
            elif a == 10:
                out = 2
            else:
                out = 1
            # 返回 x 与 out 的和
            return x + out

        # TODO: Test the guards maybe?
        # 使用标准测试函数对 fn 进行测试，期望操作数为 1
        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_int_shape_comparisons(self):
        def fn(x):
            a = x.shape[0]
            # 检查常量与数组形状的常量比较，此处不成立
            if 10 != a:
                out = 1
            # 检查常量小于数组形状的常量，此处不成立
            elif 12 < a:
                out = 1
            # 检查常量大于数组形状的常量，此处成立，将 out 设置为 1
            elif 2 > a:
                out = 1
            # 检查常量大于或等于数组形状的常量，此处成立，将 out 设置为 1
            elif 2 >= a:
                out = 1
            # 检查常量小于或等于数组形状的常量，此处不成立
            elif 12 <= a:
                out = 1
            # 检查常量等于数组形状的常量，此处成立，将 out 设置为 2
            elif 10 == a:
                out = 2
            else:
                out = 1
            # 返回 x 与 out 的和
            return x + out

        # TODO: Test the guards maybe?
        # 使用标准测试函数对 fn 进行测试，期望操作数为 1
        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_param_shape_binops(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个包含 15 个元素的参数张量
                self.param = torch.nn.Parameter(torch.randn(15))

            def forward(self, x):
                # 将参数张量的形状的第一个维度赋值给变量 p
                p = self.param.shape[0]
                # 执行多种二元操作
                y = p - x.shape[0]
                y = p + y
                y = p * y
                y = p % y
                y = p**y
                y = p // y
                y = pow(p, y)
                y = p / y
                # 返回 x 与 y 的和
                return x + y

        # 创建一个编译计数器对象
        counts = torch._dynamo.testing.CompileCounter()
        # 实例化 MyModule 类
        mod = MyModule()
        # 使用编译优化函数对 mod 进行优化
        optimized_mod = torch._dynamo.optimize(counts, nopython=True)(mod)

        # 生成一个形状为 (3,) 的随机张量 x
        x = torch.randn(3)
        # 计算未优化和优化模型的输出
        ref = mod(x)
        res = optimized_mod(x)

        # 断言未优化和优化模型输出相同
        self.assertTrue(same(ref, res))
        # 断言帧计数为 1
        self.assertEqual(counts.frame_count, 1)

        # 根据配置断言操作数符合预期值
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(counts.op_count, """1""")
        else:
            self.assertExpectedInline(counts.op_count, """11""")
    def test_user_defined_binop(self):
        # 定义一个名为 test_user_defined_binop 的测试方法
        class MyClass:
            # MyClass 类的构造函数，接受一个参数 value
            def __init__(self, value):
                self.value = value

            # 自定义右加运算符，返回 self.value + other
            def __radd__(self, other):
                return self.value + other

        # 定义一个名为 fn 的函数，接受两个参数 x 和 c
        def fn(x, c):
            # 计算 x 的第一个维度的长度加上 c 的值，赋给变量 y
            y = x.shape[0] + c
            # 返回 x 加上 y 的结果
            return x + y

        # 创建一个 CompileCounter 的实例 counts
        counts = torch._dynamo.testing.CompileCounter()
        # 对函数 fn 进行优化，得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(counts)(fn)

        # 创建一个形状为 (3,) 的随机张量 x
        x = torch.randn(3)
        # 创建一个 MyClass 的实例 c，值为 4
        c = MyClass(4)
        # 计算原始函数 fn 对 x 和 c 的调用结果，赋给 ref
        ref = fn(x, c)
        # 计算优化后函数 opt_fn 对 x 和 c 的调用结果，赋给 res
        res = opt_fn(x, c)

        # 断言 ref 和 res 相同
        self.assertTrue(same(ref, res))
        # 断言 counts.frame_count 等于 1
        self.assertEqual(counts.frame_count, 1)
        # 根据配置信息断言 counts.op_count 符合预期值
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(counts.op_count, """1""")
        else:
            self.assertExpectedInline(counts.op_count, """4""")

    def test_user_defined_iter(self):
        # 定义一个名为 test_user_defined_iter 的测试方法
        class Mod:
            # Mod 类的构造函数
            def __init__(self):
                # 初始化属性 a，包含两个形状为 (2, 2) 的随机张量
                self.a = [torch.randn(2, 2), torch.randn(2, 2)]

            # 定义 __iter__ 方法，返回属性 a 的迭代器
            def __iter__(self):
                return iter(self.a)

        # 定义一个名为 f 的函数，接受一个参数 mod
        def f(mod):
            # 初始化空列表 ret
            ret = []
            # 遍历 mod 中的每个元素 x
            for x in mod:
                # 将 x + 1 的结果追加到 ret 中
                ret.append(x + 1)
            # 返回 ret 列表
            return ret

        # 创建 Mod 类的实例 mod
        mod = Mod()
        # 创建一个 CompileCounter 的实例 counts
        counts = torch._dynamo.testing.CompileCounter()
        # 对函数 f 进行优化，得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(counts, nopython=True)(f)
        # 计算原始函数 f 对 mod 的调用结果，赋给 ref
        ref = f(mod)
        # 计算优化后函数 opt_fn 对 mod 的调用结果，赋给 res
        res = opt_fn(mod)
        res = opt_fn(mod)
        res = opt_fn(mod)
        res = opt_fn(mod)
        # 断言 ref 和 res 相同
        self.assertTrue(same(ref, res))
        # 断言 counts.frame_count 等于 1
        self.assertEqual(counts.frame_count, 1)

        # 向 mod.a 中添加一个形状为 (2, 2) 的随机张量
        mod.a.append(torch.randn(2, 2))
        # 在 for 循环中使用了 `for x in mod` 的内联化，其中 iter(m.a) 会对 mod.a 的长度进行保护
        # 修改 mod.a 的长度会导致重新编译
        # 计算原始函数 f 对 mod 的调用结果，赋给 ref2
        ref2 = f(mod)
        # 计算优化后函数 opt_fn 对 mod 的调用结果，赋给 res2
        res2 = opt_fn(mod)
        res2 = opt_fn(mod)
        res2 = opt_fn(mod)
        res2 = opt_fn(mod)
        # 断言 ref2 和 res2 相同
        self.assertTrue(same(ref2, res2))
        # 断言 counts.frame_count 等于 2
        self.assertEqual(counts.frame_count, 2)

    def test_compare_shapes_eq(self):
        # 定义一个名为 test_compare_shapes_eq 的测试方法
        # 定义一个名为 compare_shapes 的函数，接受三个参数 a、b、to_list
        def compare_shapes(a, b, to_list):
            # 如果 to_list 为 True，则将 a 在最后一维上unsqueeze后转为列表，否则保持原始形状
            x = list(a.unsqueeze(-1).shape) if to_list else a.shape
            y = list(b.unsqueeze(-1).shape) if to_list else b.shape
            # 如果 x 和 y 相等，则返回 a + 1，否则返回 a + 2
            if x == y:
                return a + 1
            else:
                return a + 2

        # 使用 torch._dynamo.testing.standard_test 对比形状，测试 to_list=True 的情况
        torch._dynamo.testing.standard_test(
            self, lambda a, b: compare_shapes(a, b, to_list=True), 2
        )
        # 使用 torch._dynamo.testing.standard_test 对比形状，测试 to_list=False 的情况
        torch._dynamo.testing.standard_test(
            self, lambda a, b: compare_shapes(a, b, to_list=False), 2
        )

    def test_compare_shapes_tuple_eq(self):
        # 定义一个名为 test_compare_shapes_tuple_eq 的测试方法
        # 定义一个名为 compare_shapes 的函数，接受两个参数 a、b
        def compare_shapes(a, b):
            # 将 a 和 b 在最后一维上unsqueeze后转为元组
            x = tuple(a.unsqueeze(-1).shape)
            y = tuple(b.unsqueeze(-1).shape)
            # 如果 x 和 y 相等，则返回 a + 1，否则返回 a + 2
            if x == y:
                return a + 1
            else:
                return a + 2

        # 使用 torch._dynamo.testing.standard_test 对比形状，测试两个张量的情况
        torch._dynamo.testing.standard_test(self, lambda a, b: compare_shapes(a, b), 2)
    # 定义一个测试方法，用于比较两个张量形状是否不相等
    def test_compare_shapes_tuple_neq(self):
        # 定义内部函数 compare_shapes，用于比较两个张量的形状
        def compare_shapes(a, b):
            # 获取张量 a 和 b 扩展后的形状，并转换为元组
            x = tuple(a.unsqueeze(-1).shape)
            y = tuple(b.unsqueeze(-1).shape)
            # 如果形状不相等，返回 a + 1，否则返回 a + 2
            if x != y:
                return a + 1
            else:
                return a + 2

        # 使用 torch._dynamo.testing.standard_test 运行标准化测试
        torch._dynamo.testing.standard_test(self, lambda a, b: compare_shapes(a, b), 2)

    # 定义一个测试方法，用于比较两个张量的形状是否不相等
    def test_compare_shapes_neq(self):
        # 定义内部函数 compare_shapes，根据 to_list 参数决定是否转换为列表形式
        def compare_shapes(a, b, to_list):
            # 根据 to_list 参数决定是否将张量 a 和 b 扩展后的形状转换为列表形式
            x = list(a.unsqueeze(-1).shape) if to_list else a.shape
            y = list(b.unsqueeze(-1).shape) if to_list else b.shape
            # 如果形状不相等，返回 a + 1，否则返回 a + 2
            if x != y:
                return a + 1
            else:
                return a + 2

        # 测试两种不同的参数传递方式：to_list=True 和 to_list=False
        torch._dynamo.testing.standard_test(
            self, lambda a, b: compare_shapes(a, b, to_list=True), 2
        )
        torch._dynamo.testing.standard_test(
            self, lambda a, b: compare_shapes(a, b, to_list=False), 2
        )

    # 定义一个测试方法，用于比较张量的形状是否符合预期常数值
    def test_compare_shapes_with_constant(self):
        # 定义内部函数 compare_shapes，用于比较张量的形状是否符合预期常数值
        def compare_shapes(a):
            # 获取张量 a 的形状
            x = a.shape
            # 如果形状的第一个维度不等于 3，返回 a * 4，否则返回 a * 3
            if x[0] != 3:
                return a * 4
            return a * 3

        # 定义 guard_failure 变量为 None
        guard_failure = None

        # 定义内部函数 guard_failures，用于设置 guard_failure 变量的值
        def guard_failures(failure):
            nonlocal guard_failure
            guard_failure = failure

        # 使用 torch._dynamo.optimize 进行优化，并传入 guard_failures 函数作为 guard_fail_fn 参数
        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(compare_shapes)
        
        # 分别对两个张量执行 opt_fn 函数
        opt_fn(torch.randn([3, 4]))
        opt_fn(torch.randn([4, 3]))
        
        # 断言 guard_failure.reason 包含特定错误信息
        self.assertIn(
            """tensor 'L['a']' size mismatch at index 0. expected 3, actual 4""",
            guard_failure.reason,
        )

    # 定义一个测试方法，用于测试内置函数 abs 的使用
    def test_builtin_abs(self):
        # 定义内部函数 fn，用于计算两个数的绝对值之和
        def fn(x, y):
            return abs(x) + abs(y)

        # 生成一个随机张量 sample
        sample = torch.randn(10, 10)
        
        # 使用 torch._dynamo.optimize 进行优化，并传入 fn 函数
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)

        # 遍历样本列表，对每对样本进行比较
        for sample in [
            (torch.randn(10, 10), torch.randn(10, 10)),
            (-10, make_tensor(10, dtype=torch.int64, device="cpu")),
            (-0.1, torch.randn(10)),
        ]:
            # 计算期望值和实际值
            expect = fn(*sample)
            actual = opt_fn(*sample)
            
            # 断言期望值和实际值相等
            self.assertEqual(expect, actual)

    # 定义一个测试方法，用于测试内置函数 isinstance 的使用
    def test_builtin_isinstance(self):
        # 定义内部函数 fn，用于测试不同类型的 isinstance 函数调用
        def fn(x):
            t = torch.arange(1, 3)
            a = isinstance(x, torch.Tensor)
            b = isinstance(t, torch.Tensor)
            c = isinstance(x, int)
            d = isinstance(3, int)
            e = isinstance([1, 2, 3], list)
            f = isinstance({"foo": 1, "bar": 2}, dict)
            res = [a, b, c, d, e, f]
            # 由于其他未实现的指令，当前无法运行
            # res += [isinstance(torch.nn.LazyLinear(2, 3), torch.nn.Linear)]
            return res

        # 使用 torch._dynamo.testing.standard_test 运行标准化测试
        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    # 使用 unittest.skipIf 标记跳过条件，条件为 Python 版本小于等于 3.8，且需要安装 astunparse 才能运行
    @unittest.skipIf(sys.version_info[:2] <= (3, 8), "Requires astunparse")
    def test_cse_dict_guards(self):
        # 定义内部函数 fn，接收参数 x，返回一个包含三个零张量的 ret
        def fn(x):
            ret = torch.zeros(3)
            # 遍历参数 x 的值，并累加到 ret
            for v in x.values():
                ret = ret + v
            return ret

        # 从 torch._dynamo.guards 模块导入 build_guard_function 和 CLOSURE_VARS
        from torch._dynamo.guards import build_guard_function, CLOSURE_VARS

        # 创建字典 x，包含三个键值对，键为整数，值为形状为 (3,) 的随机张量
        x = {3: torch.randn(3), 2: torch.randn(3), 4: torch.randn(3)}
        # 调用 torch._dynamo.export 函数，导出函数 fn 的导出值和守护代码 guards
        _, guards = torch._dynamo.export(fn, x)

        # 构建代码列表 code_lists，由 guards 中的 code_list 属性组成的列表扁平化处理
        code_lists = [c for g in guards for c in g.code_list or []]
        # 调用 build_guard_function 函数，传入 code_lists 和空列表，返回元组中的第二个元素给 pycode
        _, pycode = build_guard_function(code_lists, [])
        # 断言字符串 pycode 中 "keys" 出现的次数为 1
        self.assertEqual(pycode.count("keys"), 1)

    def test_sys_modules(self):
        # 定义内部函数 fn，接收参数 x 和 y
        def fn(x, y):
            # 获取名称为 "aaaaaaaa" 的模块，断言其为 None
            mod_a = sys.modules.get("aaaaaaaa")
            assert mod_a is None
            # 断言名称为 "bbbbbbbb" 的模块不在 sys.modules 中
            assert "bbbbbbbb" not in sys.modules

            # 断言名称为 "operator" 的模块在 sys.modules 中
            assert "operator" in sys.modules
            # 获取名称为 "operator" 的模块赋值给 operator
            operator = sys.modules["operator"]
            # 获取名称为 "builtins" 的模块，赋值给 builtins
            builtins = sys.modules.get("builtins")
            # 获取名称为 "cccccccc" 的模块，若不存在则使用 operator，赋值给 operator2
            operator2 = sys.modules.get("cccccccc", operator)

            # 返回 operator 模块的 add 方法应用于 x 和 y 的结果，
            # 以及 operator2 模块的 neg 方法应用于 builtins 模块的 abs 方法应用于 x 的结果
            return operator.add(x, y), operator2.neg(builtins.abs(x))

        # 调用 torch._dynamo.testing.standard_test 函数，传入 self、fn 和 2 作为参数，
        # 并断言其返回的操作数 expected_ops 为 3
        torch._dynamo.testing.standard_test(self, fn, 2, expected_ops=3)

        # 创建形状为 (10, 10) 的随机张量 x
        x = torch.randn(10, 10)
        # 调用 torch._dynamo.export 函数，导出函数 fn 的导出值和守护代码 guards
        _, guards = torch._dynamo.export(fn, x, x)

        # 初始化 guard_code 列表为空列表
        guard_code = []
        # 遍历 guards 中的每个 guard
        for guard in guards:
            # 如果 guard 的 code_list 不为空
            if guard.code_list:
                # 将 guard 的 code_list 添加到 guard_code 中
                guard_code += guard.code_list

        # 过滤掉 guard_code 中包含 "id" 或 "lookup_backend" 的行，将结果排序后返回
        guard_code = filter(
            lambda line: "id" not in line and "lookup_backend" not in line,
            sorted(guard_code),
        )
        # 将过滤后的 guard_code 转换为字符串，每行以换行符连接
        guard_code_str = "\n".join(guard_code)

        # 将 guard_code_str 拆分成行，并逐行进行以下操作：
        for line in """\
# 检查张量 'L['x']' 的第一个维度大小是否大于等于2
2 <= L['x'].size()[0]

# 检查张量 'L['x']' 是否与 'L['y']' 是同一个对象（引用相同）
L['x'] is L['y']

# 检查张量 'L['x']' 的维度是否为2
L['x'].ndimension() == 2

# 检查张量 'L['x']' 是否不需要梯度计算
L['x'].requires_grad == False

# 检查张量 'L['x']' 的第二个维度大小是否等于其第一个维度大小
L['x'].size()[1] == L['x'].size()[0]

# 检查张量 'L['x']' 的数据存储偏移是否为0
L['x'].storage_offset() == 0

# 检查模块 'builtins' 是否在全局变量 'G['sys'].modules' 中
___dict_contains('builtins', G['sys'].modules)

# 检查模块 'operator' 是否在全局变量 'G['sys'].modules' 中
___dict_contains('operator', G['sys'].modules)

# 检查模块 'operator' 是否在全局变量 'G['sys'].modules' 中
___dict_contains('operator', G['sys'].modules)

# 检查张量 'L['x']' 是否不具有属性 '_dynamo_dynamic_indices'
hasattr(L['x'], '_dynamo_dynamic_indices') == False

# 检查模块 'aaaaaaaa' 是否不在全局变量 'G['sys'].modules' 中
not ___dict_contains('aaaaaaaa', G['sys'].modules)

# 检查模块 'bbbbbbbb' 是否不在全局变量 'G['sys'].modules' 中
not ___dict_contains('bbbbbbbb', G['sys'].modules)

# 检查模块 'cccccccc' 是否不在全局变量 'G['sys'].modules' 中
not ___dict_contains('cccccccc', G['sys'].modules)

# 检查张量 'L['x']' 的设备是否为 'cpu'
str(L['x'].device) == 'cpu'

# 检查张量 'L['x']' 的数据类型是否为 'torch.float32'
str(L['x'].dtype) == 'torch.float32'

# 检查变量 'utils_device.CURRENT_DEVICE' 是否为 None
utils_device.CURRENT_DEVICE == None
    def test_min_max_over_iterable(self):
        # 定义一个返回测试函数的函数，参数是一个函数func
        def get_test_fn(func):
            # 内部函数，接受参数a, b和函数func，默认使用外部函数的func
            def _fn(a, b, func=func):
                # 试验各种类型的输入：列表，迭代器，元组，可变参数。
                lst = [a.shape[0] + 1, 8, a.shape[0]]
                # 分别用func处理列表，迭代器，元组和可变参数，并返回其结果
                x = func(lst)
                y = func(iter(lst))
                z = func(tuple(lst))
                w = func(*lst)
                # 返回a加上所有处理结果的和
                return a + (x + y + z + w)

            return _fn

        # 调用torch._dynamo.testing.standard_test进行标准化测试，使用func=min
        torch._dynamo.testing.standard_test(
            self,
            get_test_fn(func=min),
            2,
            expected_ops=1,
            expected_ops_dynamic=ifdynstaticdefault(1, 14),
        )
        # 调用torch._dynamo.testing.standard_test进行标准化测试，使用func=max
        torch._dynamo.testing.standard_test(
            self,
            get_test_fn(func=max),
            2,
            expected_ops=1,
            expected_ops_dynamic=ifdynstaticdefault(1, 17),
        )

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_torch_check(self):
        # 创建一个编译计数器
        cnts = torch._dynamo.testing.CompileCounter()

        # 定义一个编译函数f，使用cnts作为后端，fullgraph=True表示使用完整图
        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            # 将x转换为标量值
            y = x.item()
            # 检查y是否大于等于0
            torch._check(y >= 0)
            # 返回一个从0到y的张量
            return torch.arange(0, y)

        # 分别调用f函数，传入torch.tensor([3])和torch.tensor([4])
        f(torch.tensor([3]))
        f(torch.tensor([4]))
        # 断言编译帧数量为1
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_torch_check_symbolic_shape_rel(self):
        # 创建一个编译计数器
        cnts = torch._dynamo.testing.CompileCounter()

        # 定义一个编译函数f，使用cnts作为后端，fullgraph=True表示使用完整图
        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            # 将x转换为标量值
            y = x.item()
            # 检查x的形状的第一个维度是否等于1
            torch._check(x.shape[0] == 1)
            # 检查x的形状的第一个维度是否不等于2
            torch._check(x.shape[0] != 2)
            # 检查x的形状的第一个维度是否大于等于0
            torch._check(x.shape[0] >= 0)
            # 检查x的形状的第一个维度是否大于0
            torch._check(x.shape[0] > 0)
            # 检查x的形状的第一个维度是否小于4
            torch._check(x.shape[0] < 4)
            # 检查x的形状的第一个维度是否小于等于3
            torch._check(x.shape[0] <= 3)
            # 返回一个从0到y的张量
            return torch.arange(0, y)

        # 分别调用f函数，传入torch.tensor([3])和torch.tensor([4])
        f(torch.tensor([3]))
        f(torch.tensor([4]))
        # 断言编译帧数量为1
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    # 禁用翻译验证，因为它会改变异常类型
    @torch.fx.experimental._config.patch(translation_validation=False)
    def test_torch_check_is_size(self):
        # 创建一个编译计数器
        cnts = torch._dynamo.testing.CompileCounter()

        # 定义一个编译函数f，使用cnts作为后端，fullgraph=True表示使用完整图
        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            # 将x转换为标量值
            y = x.item()
            # 检查y是否是合法的张量大小
            torch._check_is_size(y)
            # 如果y等于0，则抛出UserError异常
            if y == 0:
                assert False
            else:
                # 否则返回一个从0到y的张量
                return torch.arange(0, y)

        # 使用lambda表达式调用f函数，传入torch.tensor([3])，断言抛出UserError异常
        self.assertRaises(torch._dynamo.exc.UserError, lambda: f(torch.tensor([3])))

    def test_assert(self):
        # 定义一个使用torch.compile装饰器的函数fn1，用于检查x的形状是否不等于自身的形状
        @torch.compile
        def fn1(x):
            assert x.shape != x.shape

        # 使用assertRaises断言捕获AssertionError异常
        with self.assertRaises(AssertionError):
            # 创建一个形状为10的随机张量a
            a = torch.randn(10)
            # 调用fn1函数，传入a作为参数
            fn1(a)

        # 定义一个普通函数fn2，用于检查x的形状是否等于自身的形状，并返回绝对值
        def fn2(x):
            assert x.shape == x.shape
            return x.abs()

        # 使用torch._dynamo.testing.standard_test进行标准化测试，使用fn=fn2，nargs=1，expected_ops=1
        torch._dynamo.testing.standard_test(self, fn=fn2, nargs=1, expected_ops=1)
    def test_config_obj(self):
        # 定义配置类 Cfg，包含初始值和计数属性
        class Cfg:
            def __init__(self):
                self.val = 0.5  # 初始值为 0.5
                self.count = 3  # 初始计数为 3

        # 定义一个函数 fn，根据配置对象 cfg 对输入 x 进行加法操作
        def fn(x, cfg):
            for i in range(cfg.count):
                x = x + cfg.val  # 对 x 进行 cfg.val 的累加
            return x

        # 创建两个不同的配置对象 cfg1 和 cfg2
        cfg1 = Cfg()
        cfg1.val = 1.0  # 修改 cfg1 的 val 属性为 1.0
        cfg2 = Cfg()
        
        v = torch.zeros(1)  # 创建一个值为 0 的 torch 张量 v
        cnts = torch._dynamo.testing.CompileCounter()  # 创建一个编译计数器 cnts
        opt_fn = torch._dynamo.optimize(cnts)(fn)  # 对函数 fn 进行优化

        # 使用优化后的函数 opt_fn 多次调用，验证不同配置的效果
        v = opt_fn(v, cfg1)  # 第一次调用，预期结果为 3
        v = opt_fn(v, cfg2)  # 第二次调用，预期结果为 4.5
        cfg2.count = 1  # 修改 cfg2 的 count 属性为 1
        v = opt_fn(v, cfg2)  # 第三次调用，预期结果为 5
        cfg2.val = 2.0  # 修改 cfg2 的 val 属性为 2.0
        v = opt_fn(v, cfg2)  # 第四次调用，预期结果为 7

        # 断言最终结果 v 的第一个元素为 7
        self.assertEqual(v[0], 7)
        # 断言操作计数 cnts.op_count 为 8
        self.assertEqual(cnts.op_count, 8)

    def test_config_getattr_default(self):
        # 定义配置类 Cfg，包含初始值和计数属性
        class Cfg:
            def __init__(self):
                self.val = 0.5  # 初始值为 0.5
                self.count = 10  # 初始计数为 10

        # 定义一个函数 fn，根据配置对象 cfg 对输入 x 进行加法操作
        def fn(x, cfg):
            if getattr(cfg, "just_add_7", False):  # 如果 cfg 中存在 just_add_7 属性且为真，则直接返回 x+7
                return x + 7
            for i in range(cfg.count):
                x = x + cfg.val  # 对 x 进行 cfg.val 的累加
            return x

        cfg1 = Cfg()  # 创建配置对象 cfg1
        v = torch.zeros(1)  # 创建一个值为 0 的 torch 张量 v
        cnts = torch._dynamo.testing.CompileCounter()  # 创建一个编译计数器 cnts
        opt_fn = torch._dynamo.optimize(cnts)(fn)  # 对函数 fn 进行优化

        # 使用优化后的函数 opt_fn 多次调用，验证不同配置的效果
        self.assertEqual(opt_fn(v, cfg1)[0], 5)  # 第一次调用，预期结果为 5
        self.assertEqual(opt_fn(v, cfg1)[0], 5)  # 第二次调用，预期结果为 5
        cfg1.just_add_7 = True  # 设置 cfg1 的 just_add_7 属性为 True
        self.assertEqual(opt_fn(v, cfg1)[0], 7)  # 第三次调用，预期结果为 7
        self.assertEqual(opt_fn(v, cfg1)[0], 7)  # 第四次调用，预期结果为 7
        cfg1.just_add_7 = False  # 设置 cfg1 的 just_add_7 属性为 False
        self.assertEqual(opt_fn(v, cfg1)[0], 5)  # 第五次调用，预期结果为 5
        self.assertEqual(opt_fn(v, cfg1)[0], 5)  # 第六次调用，预期结果为 5

        # 断言帧计数 cnts.frame_count 为 3
        self.assertEqual(cnts.frame_count, 3)

    def test_size_input(self):
        # 定义一个函数 fn，接受输入 x 和大小 s，进行加法操作
        def fn(x, s):
            a, b = s
            return x + (a - b)

        v = torch.zeros(10, 20)  # 创建一个大小为 (10, 20) 的 torch 张量 v
        cnts = torch._dynamo.testing.CompileCounter()  # 创建一个编译计数器 cnts
        opt_fn = torch._dynamo.optimize(cnts)(fn)  # 对函数 fn 进行优化

        # 使用优化后的函数 opt_fn 多次调用，验证不同输入类型的效果
        self.assertEqual(opt_fn(v, v.size())[0, 0], -10)  # 第一次调用，预期结果为 -10
        self.assertEqual(opt_fn(v, (10, 20))[0, 0], -10)  # 第二次调用，预期结果为 -10
        self.assertEqual(opt_fn(v, [10, 20])[0, 0], -10)  # 第三次调用，预期结果为 -10

        # 断言帧计数 cnts.frame_count 为 3
        self.assertEqual(cnts.frame_count, 3)

    def test_cell_output1(self):
        out = None

        # 定义一个函数 fn，接受输入 a 和 b，修改 nonlocal 变量 out 的值
        def fn(a, b):
            nonlocal out
            out = a + b * 10

        v = torch.Tensor([100])  # 创建一个值为 100 的 torch 张量 v
        cnts = torch._dynamo.testing.CompileCounter()  # 创建一个编译计数器 cnts
        opt_fn = torch._dynamo.optimize(cnts)(fn)  # 对函数 fn 进行优化

        self.assertIsNone(opt_fn(v, v))  # 调用优化后的函数 opt_fn，预期返回 None
        self.assertEqual(out[0], 1100)  # 验证 out 变量的值为 1100
        self.assertEqual(cnts.op_count, 2)  # 断言操作计数 cnts.op_count 为 2

    def test_cell_output2(self):
        out = None

        # 定义一个函数 fn，接受输入 a 和 b，修改 nonlocal 变量 out 的值，同时调用 unsupported 函数
        def fn(a, b):
            nonlocal out
            c = unsupported(a, b)  # 调用未支持的函数 unsupported
            out = a + b * 10 + c

        v = torch.Tensor([100])  # 创建一个值为 100 的 torch 张量 v
        cnts = torch._dynamo.testing.CompileCounter()  # 创建一个编译计数器 cnts
        opt_fn = torch._dynamo.optimize(cnts)(fn)  # 对函数 fn 进行优化

        self.assertIsNone(opt_fn(v, v))  # 调用优化后的函数 opt_fn，预期返回 None
        self.assertEqual(out[0], 1200)  # 验证 out 变量的值为 1200
        self.assertEqual(cnts.op_count, 3)  # 断言操作计数 cnts.op_count 为 3
    # 定义测试函数 test_return_nested_function，使用 self 作为参数（通常用于测试框架中的测试方法）
    def test_return_nested_function(self):
        # 初始化变量 out 为 None
        out = None

        # 定义内部函数 fn，接受参数 a 和 b
        def fn(a, b):
            # 使用 nonlocal 声明 out 变量，表示将在父级作用域中使用该变量
            nonlocal out
            # 计算 c，将 a 和 b 相加
            c = a + b
            # 计算 d，将 a 加上浮点数 1.0
            d = a + 1.0

            # 定义嵌套函数 fn2，接受参数 f（整数，默认为 7）和 g（浮点数，默认为 9.0）
            def fn2(f: int = 7, g: float = 9.0):
                # 使用 nonlocal 声明 out 变量，表示将在父级作用域中使用该变量
                nonlocal out
                # 将 out 设置为 a 加上 b 乘以 10 的结果
                out = a + b * 10
                # 返回计算结果 c 乘以 f 减去 d 乘以 g
                return c * f - d * g
            
            # 返回嵌套函数 fn2
            return fn2

        # 创建张量 v1，包含值为 100 的张量
        v1 = torch.Tensor([100])
        # 创建张量 v2，包含值为 200 的张量
        v2 = torch.Tensor([200])
        # 创建 CompileCounter 实例 cnts，用于统计编译的次数
        cnts = torch._dynamo.testing.CompileCounter()
        # 通过优化函数 torch._dynamo.optimize 对 fn 进行优化，返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 通过优化后的函数 opt_fn 对 v1 和 v2 进行调用，并再次进行优化，返回优化后的函数 opt_fn_ret
        opt_fn_ret = torch._dynamo.optimize(cnts)(opt_fn(v1, v2))
        # 使用 self.assertEqual 进行断言，验证 opt_fn_ret(1.5)[0] 的返回值是否等于 -459
        self.assertEqual(opt_fn_ret(1.5)[0], -459)
        # 使用 self.assertEqual 进行断言，验证 out[0] 的返回值是否等于 2100
        self.assertEqual(out[0], 2100)
        # 使用 self.assertEqual 进行断言，验证 cnts.frame_count 的值是否等于 2
        self.assertEqual(cnts.frame_count, 2)
        # 使用 self.assertEqual 进行断言，验证 cnts.op_count 的值是否等于 7
        self.assertEqual(cnts.op_count, 7)

    # 定义测试函数 test_tensor_dict1
    def test_tensor_dict1(self):
        # 定义函数 fn，接受参数 inputs，返回 inputs["a"] 减去 inputs["b"] 乘以 1.5 的结果
        def fn(inputs):
            return inputs["a"] - inputs["b"] * 1.5

        # 创建张量 v1，包含值为 100 的张量
        v1 = torch.Tensor([100])
        # 创建张量 v2，包含值为 200 的张量
        v2 = torch.Tensor([200])
        # 创建 CompileCounter 实例 cnts，用于统计编译的次数
        cnts = torch._dynamo.testing.CompileCounter()
        # 通过优化函数 torch._dynamo.optimize 对 fn 进行优化，设置 nopython=True，返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 使用 self.assertEqual 进行断言，验证 opt_fn({"a": v1, "b": v2})[0] 的返回值是否等于 -200
        self.assertEqual(opt_fn({"a": v1, "b": v2})[0], -200)
        # 使用 self.assertEqual 进行断言，验证 cnts.frame_count 的值是否等于 1
        self.assertEqual(cnts.frame_count, 1)
        # 使用 self.assertEqual 进行断言，验证 cnts.op_count 的值是否等于 2
        self.assertEqual(cnts.op_count, 2)

    # 定义测试函数 test_tensor_dict3
    def test_tensor_dict3(self):
        # 定义函数 fn，接受参数 inputs_a 和 inputs_b
        def fn(inputs_a, inputs_b):
            # 初始化 total 为包含值为 0 的张量
            total = torch.zeros(1)
            # 获取 inputs_a 和 inputs_b 的所有键，并将它们合并成一个集合
            input_keys = inputs_a.keys() | inputs_b.keys()
            # 遍历合并后的键集合
            for k in input_keys:
                # 如果键 k 存在于 inputs_a 中，则将 total 增加 inputs_a[k] 的值
                if k in inputs_a:
                    total += inputs_a[k]
                # 如果键 k 存在于 inputs_b 中，则将 total 增加 inputs_b[k] 的值
                if k in inputs_b:
                    total += inputs_b[k]
            # 返回 total
            return total

        # 创建张量 v1，包含值为 100 的张量
        v1 = torch.Tensor([100])
        # 创建张量 v2，包含值为 200 的张量
        v2 = torch.Tensor([200])
        # 创建 CompileCounter 实例 cnts，用于统计编译的次数
        cnts = torch._dynamo.testing.CompileCounter()
        # 通过优化函数 torch._dynamo.optimize 对 fn 进行优化，设置 nopython=True，返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 使用 self.assertEqual 进行断言，验证 opt_fn({"a": v1, "b": v2}, {"b": v1, "c": v2}) 的返回值
        # 是否与未优化的函数 fn({"a": v1, "b": v2}, {"b": v1, "c": v2}) 的返回值相等
        self.assertEqual(
            opt_fn({"a": v1, "b": v2}, {"b": v1, "c": v2}),
            fn({"a": v1, "b": v2}, {"b": v1, "c": v2}),
        )
        # 使用 self.assertEqual 进行断言，验证 cnts.frame_count 的值是否等于 1
        self.assertEqual(cnts.frame_count, 1)
        # 使用 self.assertEqual 进行断言，验证 cnts.op_count 的值是否等于 5
        self.assertEqual(cnts.op_count, 5)
    def test_tensor_dict2(self):
        # 定义第一个函数 fn1，计算输入字典中所有值的总和
        def fn1(inputs):
            total = torch.zeros(1)
            for k, v in inputs.items():
                total += v
            return total

        # 定义第二个函数 fn2，计算输入字典中所有值的总和（使用 .values()）
        def fn2(inputs):
            total = torch.zeros(1)
            for v in inputs.values():
                total += v
            return total

        # 定义第三个函数 fn3，计算输入字典中所有值的总和（使用 .keys()）
        def fn3(inputs):
            total = torch.zeros(1)
            for k in inputs.keys():
                total += inputs[k]
            return total

        # 创建两个张量 v1 和 v2，分别初始化为 [100] 和 [200]
        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        
        # 创建一个编译计数器 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        
        # 对三个函数分别进行优化，返回优化后的函数对象
        opt_fn1 = torch._dynamo.optimize(cnts, nopython=True)(fn1)
        opt_fn2 = torch._dynamo.optimize(cnts, nopython=True)(fn2)
        opt_fn3 = torch._dynamo.optimize(cnts, nopython=True)(fn3)
        
        # 断言优化后的 fn1、fn2、fn3 对输入 {"a": v1, "b": v2} 的输出为 [300]
        self.assertEqual(opt_fn1({"a": v1, "b": v2})[0], 300)
        self.assertEqual(opt_fn2({"a": v1, "b": v2})[0], 300)
        self.assertEqual(opt_fn3({"a": v1, "b": v2})[0], 300)
        
        # 断言编译计数器的帧计数为 3，操作计数为 9
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 9)

    def test_dictcomp(self):
        # 定义函数 fn1，使用字典推导式将输入字典中每个值加1
        def fn1(inputs):
            return {k: v + 1 for k, v in inputs.items()}

        # 创建两个张量 v1 和 v2，分别初始化为 [100] 和 [200]
        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        
        # 创建一个编译计数器 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        
        # 对函数 fn1 进行优化，返回优化后的函数对象
        opt_fn1 = torch._dynamo.optimize(cnts)(fn1)
        
        # 断言优化后的 fn1 对输入 {"a": v1, "b": v2} 的输出中 "a" 对应的值为 101，"b" 对应的值为 201
        self.assertEqual(opt_fn1({"a": v1, "b": v2})["a"], 101)
        self.assertEqual(opt_fn1({"a": v1, "b": v2})["b"], 201)
        
        # 断言编译计数器的帧计数为 1，操作计数为 2
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_listcomp(self):
        # 定义函数 fn2，使用列表推导式计算输入字典中所有值加1的总和
        def fn2(inputs):
            return torch.sum(torch.cat([v + 1 for k, v in inputs.items()], 0))

        # 创建两个张量 v1 和 v2，分别初始化为 [100] 和 [200]
        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        
        # 创建一个编译计数器 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        
        # 对函数 fn2 进行优化，返回优化后的函数对象
        opt_fn2 = torch._dynamo.optimize(cnts)(fn2)
        
        # 断言优化后的 fn2 对输入 {"a": v1, "b": v2} 的输出为 302
        self.assertEqual(opt_fn2({"a": v1, "b": v2}), 302)
        
        # 断言编译计数器的帧计数为 1，操作计数为 4
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    def test_is_floating_point(self):
        # 定义函数 fn，计算 a + 1.0，如果 b 是浮点数则再加上 b，最后再加上 2.0
        def fn(a, b):
            x = a + 1.0
            if torch.is_floating_point(b):
                x = x + b
            return x + 2.0

        # 调用测试工具函数，测试 fn 函数的标准测试，期望操作数为 3
        return torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_is_floating_point2(self):
        # 定义函数 fn，计算 a + 1.0，如果 b 是张量中的浮点数则再加上 b，最后再加上 2.0
        def fn(a, b):
            x = a + 1.0
            if b.is_floating_point():
                x = x + b
            return x + 2.0

        # 调用测试工具函数，测试 fn 函数的标准测试，期望操作数为 3
        return torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_is_tensor(self):
        # 定义函数 fn，计算 a + 1.0，如果 b 是张量则再加上 b，最后再加上 2.0
        def fn(a, b):
            x = a + 1.0
            if torch.is_tensor(b):
                x = x + b
            return x + 2.0

        # 调用测试工具函数，测试 fn 函数的标准测试，期望操作数为 3
        return torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)
    def test_is_tensor2(self):
        # 定义一个内部函数 fn，检查输入 x 是否为 tensor，如果是则返回 x + 1，否则返回一个形状为 [2, 3] 的全 1 张量
        def fn(x):
            if torch.is_tensor(x):
                return x + 1
            else:
                return torch.ones([2, 3])

        # 创建一个字典 x1，包含名为 "input" 的键，对应一个形状为 [2, 3] 的随机张量
        x1 = {"input": torch.rand(2, 3)}
        # 创建一个形状为 [2, 3] 的随机张量 x2
        x2 = torch.rand(2, 3)
        # 调用 fn 函数处理 x1，将结果保存为 ref1
        ref1 = fn(x1)
        # 调用 fn 函数处理 x2，将结果保存为 ref2
        ref2 = fn(x2)
        # 使用 torch._dynamo.optimize("eager") 对 fn 进行优化，结果保存为 opt_fn
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 分别使用优化后的函数 opt_fn 处理 x1 和 x2，结果保存为 res1 和 res2
        res1 = opt_fn(x1)
        res2 = opt_fn(x2)
        # 断言 ref1 和 res1 相等
        self.assertEqual(ref1, res1)
        # 断言 ref2 和 res2 相等
        self.assertEqual(ref2, res2)

    def test_numel(self):
        # 定义一个函数 fn，接受参数 a，返回包含两个元素的元组，第一个元素是 a + a.numel() + torch.numel(a)，第二个元素是 a + a.nelement()
        def fn(a):
            return (a + a.numel() + torch.numel(a), a + a.nelement())

        # 调用 torch._dynamo.testing.standard_test 运行测试，使用 fn 函数，传入一个参数，期望的操作数为 3
        return torch._dynamo.testing.standard_test(
            self,
            fn=fn,
            nargs=1,
            expected_ops=3,
            expected_ops_dynamic=ifdynstaticdefault(3, 6),
        )

    def test_pair(self):
        # 定义一个函数 fn，接受参数 a，返回一个张量，张量内容为 torch.zeros(a.size()) + a + torch.ones(torch.nn.modules.utils._ntuple(3)(3)).sum()
        def fn(a):
            return (
                torch.zeros(torch.nn.modules.utils._pair(a.size()))
                + a
                + torch.ones(torch.nn.modules.utils._ntuple(3)(3)).sum()
            )

        # 调用 torch._dynamo.testing.standard_test 运行测试，使用 fn 函数，传入一个参数，期望的操作数为 5
        return torch._dynamo.testing.standard_test(
            self,
            fn=fn,
            nargs=1,
            expected_ops=5,
            expected_ops_dynamic=ifdynstaticdefault(5, 8),
        )

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_tensor_item_capture(self):
        # 定义一个函数 fn，接受两个参数 a 和 b，计算 (a + b).sum().item() 的结果
        def fn(a, b):
            return (a + b).sum().item()

        # 创建两个形状为 (10, 10) 的随机张量 v1 和 v2
        v1 = torch.randn((10, 10))
        v2 = torch.randn((10, 10))
        # 计算正确结果，即调用 fn 函数计算 v1 和 v2 的和的总和
        correct = fn(v1, v2)
        # 创建一个 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize(cnts) 对 fn 函数进行优化，结果保存为 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 断言优化后的函数 opt_fn 处理 v1 和 v2 的结果与正确结果相等
        self.assertEqual(opt_fn(v1, v2), correct)
        # 断言 frame_count 等于 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言 op_count 等于 4
        self.assertEqual(cnts.op_count, 4)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", False)
    def test_tensor_item_no_capture(self):
        # 定义一个函数 fn，接受两个参数 a 和 b，计算 (a + b).sum().item() 的结果
        def fn(a, b):
            return (a + b).sum().item()

        # 创建两个形状为 (10, 10) 的随机张量 v1 和 v2
        v1 = torch.randn((10, 10))
        v2 = torch.randn((10, 10))
        # 计算正确结果，即调用 fn 函数计算 v1 和 v2 的和的总和
        correct = fn(v1, v2)
        # 创建一个 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize(cnts) 对 fn 函数进行优化，结果保存为 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 断言优化后的函数 opt_fn 处理 v1 和 v2 的结果与正确结果相等
        self.assertEqual(opt_fn(v1, v2), correct)
        # 断言 frame_count 等于 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言 op_count 等于 2
        self.assertEqual(cnts.op_count, 2)

    def test_namedtuple1(self):
        # 定义一个函数 fn，接受两个参数 a 和 b，创建一个 mytuple 对象，返回 mytuple(a, b, a + b) 和 mytuple(tmp.a, tmp[1], tmp.ab + b) 的结果
        def fn(a, b):
            tmp = mytuple(a, b, a + b)
            return mytuple(tmp.a, tmp[1], tmp.ab + b)

        # 创建两个张量 v1 和 v2，分别包含值 [10] 和 [20]
        v1 = torch.Tensor([10])
        v2 = torch.Tensor([20])
        # 创建一个 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize(cnts) 对 fn 函数进行优化，结果保存为 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 断言优化后的函数 opt_fn 处理 v1 和 v2 的结果中 ab 属性值等于 50
        self.assertEqual(opt_fn(v1, v2).ab, 50)
        # 断言 frame_count 等于 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言 op_count 等于 2
        self.assertEqual(cnts.op_count, 2)
    def test_namedtuple2(self):
        # 定义一个处理元组的函数，解包元组并执行操作
        def fn(packed):
            # 解包元组中的元素a, b, c
            a, b, c = packed
            # 如果元组具有属性"b"，则将b的值加1
            if hasattr(packed, "b"):
                b = packed.b + 1
            # 取元组中第三个元素的值赋给变量c
            c = packed[2]
            # 返回a、b、c三个值的和
            return a + b + c

        # 创建三个Tensor对象
        v1 = torch.Tensor([1])
        v2 = torch.Tensor([2])
        v3 = torch.Tensor([3])
        # 创建一个CompileCounter对象cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化函数fn，并将结果赋给opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 断言优化后的函数对mytuple(v1, v2, v3)的输出为7
        self.assertEqual(opt_fn(mytuple(v1, v2, v3))[0], 7)
        # 断言帧计数为1
        self.assertEqual(cnts.frame_count, 1)
        # 断言操作计数为3
        self.assertEqual(cnts.op_count, 3)

    def test_namedtuple3(self):
        # 定义一个函数fn，根据packed的类型返回不同的值
        def fn(x, packed):
            # 如果packed是mytuple类型，则返回x+1
            if isinstance(packed, mytuple):
                return x + 1
            # 否则返回x-1
            else:
                return x - 1

        # 创建一个2行3列的随机Tensor对象x
        x = torch.rand([2, 3])
        # 创建一个mytuple对象packed，包含三个元素1, 2, 3
        packed = mytuple(1, 2, 3)
        # 计算fn(x, packed)的结果并赋给ref
        ref = fn(x, packed)
        # 优化函数fn，并将结果赋给opt_fn
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 计算优化后函数的结果并赋给res
        res = opt_fn(x, packed)
        # 断言ref与res相同
        self.assertTrue(same(ref, res))

    def test_range_input(self):
        # 定义一个函数fn，对a与rng中的元素求和
        def fn(a, rng):
            # 将a赋给变量x
            x = a
            # 遍历rng中的元素，将其依次加到x上
            for i in rng:
                x = x + i
            # 返回最终结果x
            return x

        # 定义一个fn1函数，调用fn，并将range(3)赋给rng
        def fn1(a):
            return fn(a, rng=range(3))

        # 调用torch._dynamo.testing.standard_test函数，测试fn1的输出
        return torch._dynamo.testing.standard_test(
            self, fn=fn1, nargs=1, expected_ops=3
        )

    def test_range_with_shape(self):
        # 定义一个函数fn，对a的第一个维度进行遍历，每次加1
        def fn(a):
            # 遍历a的第一个维度（从1到a.shape[0]，不包括a.shape[0]）
            for i in range(1, a.shape[0]):
                a += 1
            # 返回修改后的a
            return a

        # 调用torch._dynamo.testing.standard_test函数，测试fn的输出
        return torch._dynamo.testing.standard_test(
            self,
            fn=fn,
            nargs=1,
            expected_ops=9,
        )

    def test_build_tuple_unpack(self):
        # 定义一个fn1函数，创建一个元组args，然后调用fn1函数
        def fn1(a, b, c):
            # 返回fn1函数的结果
            return a - b / c

        # 定义一个fn2函数，创建两个临时元组tmp1和tmp2，将它们合并为args，然后调用fn1函数
        def fn2(a, b, c):
            tmp1 = (a,)
            tmp2 = (b, c)
            args = (*tmp1, *tmp2)
            return fn1(*args)

        # 定义一个fn3函数，动态传入参数，调用fn1函数
        def fn3(a, *args):
            return fn1(a, *args)

        # 调用torch._dynamo.testing.standard_test函数，测试fn2的输出
        torch._dynamo.testing.standard_test(self, fn=fn2, nargs=3, expected_ops=2)
        # 调用torch._dynamo.testing.standard_test函数，测试fn3的输出
        torch._dynamo.testing.standard_test(self, fn=fn3, nargs=3, expected_ops=2)

    def test_list_mul(self):
        # 定义一个fn函数，根据count的值创建一个列表head_mask并返回
        def fn(count):
            head_mask = count * [None] * count
            return head_mask

        # 创建一个CompileCounter对象cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化函数fn，并将结果赋给opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 断言优化后的函数对输入2的输出为[None, None, None, None]
        self.assertEqual(opt_fn(2), [None] * 4)
        # 检查帧计数是否符合预期
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnts.frame_count, """0""")
            self.assertExpectedInline(cnts.op_count, """0""")
        else:
            self.assertExpectedInline(cnts.frame_count, """1""")
            self.assertExpectedInline(cnts.op_count, """2""")
    def test_list_slice_mul(self):
        # 定义一个内部函数 fn，接受一个参数 count
        def fn(count):
            # 创建列表 a 包含元素 [1, 2, 3]
            a = [1, 2, 3]
            # 计算 head_mask，它是 count 乘以列表 a 的切片 [2, 3] 的结果再乘以 count
            head_mask = count * a[1:] * count
            return head_mask

        # 创建 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对函数 fn 进行优化
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 断言优化后的函数对于输入 2 的结果为 [2, 3, 2, 3, 2, 3, 2, 3]
        self.assertEqual(opt_fn(2), [2, 3] * 4)
        # 根据 assume_static_by_default 配置进行断言
        if torch._dynamo.config.assume_static_by_default:
            # 断言 frame_count 的预期内联结果为 "0"
            self.assertExpectedInline(cnts.frame_count, """0""")
            # 断言 op_count 的预期内联结果为 "0"
            self.assertExpectedInline(cnts.op_count, """0""")
        else:
            # 断言 frame_count 的预期内联结果为 "1"
            self.assertExpectedInline(cnts.frame_count, """1""")
            # 断言 op_count 的预期内联结果为 "2"
            self.assertExpectedInline(cnts.op_count, """2""")

    def test_tuple_mul(self):
        # 定义一个内部函数 fn，接受一个参数 count
        def fn(count):
            # 计算 head_mask，它是 count 乘以元组 (2, 3) 的结果再乘以 count
            head_mask = count * (2, 3) * count
            return head_mask

        # 创建 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对函数 fn 进行优化
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 断言优化后的函数对于输入 2 的结果为元组 (2, 3, 2, 3, 2, 3, 2, 3)
        self.assertEqual(opt_fn(2), (2, 3) * 4)
        # 根据 assume_static_by_default 配置进行断言
        if torch._dynamo.config.assume_static_by_default:
            # 断言 frame_count 的预期内联结果为 "0"
            self.assertExpectedInline(cnts.frame_count, """0""")
            # 断言 op_count 的预期内联结果为 "0"
            self.assertExpectedInline(cnts.op_count, """0""")
        else:
            # 断言 frame_count 的预期内联结果为 "1"
            self.assertExpectedInline(cnts.frame_count, """1""")
            # 断言 op_count 的预期内联结果为 "2"
            self.assertExpectedInline(cnts.op_count, """2""")

    def test_tuple_mul_with_shape(self):
        # 定义一个函数 fn，接受一个参数 a
        def fn(a):
            # 获取数组 a 的第一个维度大小
            x = a.shape[0]
            # 计算 y，它是 2 乘以元组 (x, 3) 的结果再乘以 2
            y = 2 * (x, 3) * 2
            # 返回数组 a 与 y 第五个元素的和
            return a + y[4]

        # 使用 standard_test 函数测试 fn 函数
        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=ifdynstaticdefault(1, 3)
        )

    def test_tuple_iadd_with_shape(self):
        # 定义一个函数 fn，接受一个参数 a
        def fn(a):
            # 创建元组 output 包含两个元素：a 加上数组 a 的第一个维度大小，a 减去数组 a 的第一个维度大小
            output = (a + a.shape[0], a - a.shape[0])
            # 对 output 进行元组相加操作
            output += (a - a.shape[0], a + a.shape[0])
            # 将常数元组 (2, 3) 添加到 output 中
            output += (2, 3)
            return output

        # 使用 standard_test 函数测试 fn 函数
        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=4, expected_ops_dynamic=ifdynstaticdefault(4, 12)
        )

    def test_list_iadd_with_shape(self):
        # 定义一个函数 fn，接受一个参数 a
        def fn(a):
            # 创建列表 output 包含两个元素：a 加上数组 a 的第一个维度大小，a 减去数组 a 的第一个维度大小
            output = [a + a.shape[0], a - a.shape[0]]
            # 对 output 进行列表相加操作
            output += [a - a.shape[0], a + a.shape[0]]
            # 将元组 (a 加上数组 a 的第一个维度大小，a 减去数组 a 的第一个维度大小) 添加到 output 中
            output += (a + a.shape[0], a - a.shape[0])
            return output

        # 使用 standard_test 函数测试 fn 函数
        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=6, expected_ops_dynamic=ifdynstaticdefault(6, 18)
        )
    # 定义测试函数，用于测试列表的就地加法操作的副作用
    def test_list_iadd_side_effect(self):
        # 定义内部函数，接收列表 a 和对象 b 作为参数，将 b 添加到 a 中
        def fn(a, b):
            a += [b]
            # 调用 torch._dynamo.graph_break() 方法
            torch._dynamo.graph_break()
            # 返回修改后的列表 a
            return a

        # 初始化列表 a
        a = [1, 2, 3]
        # 创建一个全为 1 的 2x2 Tensor 对象 b
        b = torch.ones(2, 2)

        # 优化函数 fn，并将其结果保存到 opt_fn
        opt_fn = torch._dynamo.optimize("eager")(fn)

        # 计算预期结果
        exp = fn(a, b)

        # 重新初始化列表 a 和 Tensor 对象 b
        a = [1, 2, 3]
        b = torch.ones(2, 2)
        # 调用优化后的函数 opt_fn
        act = opt_fn(a, b)

        # 断言预期结果与实际结果相等
        self.assertEqual(exp, act)

    # 测试自定义类 MyConfig 的 getattr 方法
    def test_user_getattr1(self):
        # 定义继承自 dict 的 MyConfig 类
        class MyConfig(dict):
            # 实现 __getattr__ 方法，返回字典中对应属性的值
            def __getattr__(self, name):
                return self[name]

        # 定义函数 fn，接收配置对象 cfg、以及两个 Tensor 对象 x 和 y 作为参数
        def fn(cfg, x, y):
            # 返回 x、y 的和，加上 cfg 中的 offset 属性值
            return x + y + cfg.offset

        # 初始化随机 10 维度的 Tensor 对象 x
        x = torch.randn(10)
        # 创建 MyConfig 实例 cfg，设置 offset 属性为 5
        cfg = MyConfig(offset=5)
        # 创建 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化函数 fn，并将其结果保存到 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 断言优化后的函数调用结果与预期值相同
        self.assertTrue(same(opt_fn(cfg, x, x), 2 * x + 5))
        # 断言帧计数为 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言操作计数为 2
        self.assertEqual(cnts.op_count, 2)

    # 测试另一个自定义类 MyConfig 的 getattr 方法
    def test_user_getattr2(self):
        # 定义 MyConfig 类
        class MyConfig:
            defined_on_class = 1

            def __init__(self):
                self.defined_on_object = 2

            # 实现 __getattr__ 方法，返回固定值 3
            def __getattr__(self, name):
                return 3

        # 定义函数 fn，接收配置对象 cfg 和 Tensor 对象 x 作为参数
        def fn(cfg, x):
            # 返回 x 加上 cfg 中定义在类上的属性值，减去定义在对象上的属性值，加上不存在的属性值 not_defined（返回默认值 3）
            return x + cfg.defined_on_class - cfg.defined_on_object + cfg.not_defined

        # 初始化随机 10 维度的 Tensor 对象 x
        x = torch.randn(10)
        # 创建 MyConfig 实例 cfg
        cfg = MyConfig()
        # 创建 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化函数 fn，并将其结果保存到 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 断言优化后的函数调用结果与预期值相同
        self.assertTrue(same(opt_fn(cfg, x), x + 1 - 2 + 3))
        # 断言帧计数为 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言操作计数为 3
        self.assertEqual(cnts.op_count, 3)

    # 测试获取和设置描述符的函数
    def test_getset_descriptor(self):
        # 定义函数 fn，接收描述符 g 和 Tensor 对象 x 作为参数，返回 g 对 x 的获取结果
        def fn(g, x):
            return g.__get__(x)

        # 创建 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 编译函数 fn 并使用 eager 后端进行优化
        opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)
        # 获取 torch.Tensor.shape 描述符 g
        g = torch.Tensor.shape

        # 调用优化后的函数 opt_fn，计算结果 res
        res = opt_fn(g, torch.ones(2, 2))
        # 计算预期结果 exp_res
        exp_res = fn(g, torch.ones(2, 2))
        # 断言结果 res 与预期结果 exp_res 相等
        self.assertEqual(res, exp_res)

    # 测试获取属性函数
    def test_get_attr_function(self):
        # 定义函数 fn，接收函数 g 和 Tensor 对象 x 作为参数，返回 g 对 x 的调用结果
        def fn(g, x):
            return g(x)

        # 创建 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化函数 fn，并将其结果保存到 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 获取 torch.Tensor.shape.__get__ 方法 g
        g = torch.Tensor.shape.__get__

        # 调用优化后的函数 opt_fn，计算结果 res
        res = opt_fn(g, torch.ones(2, 2))
        # 计算预期结果 exp_res
        exp_res = fn(g, torch.ones(2, 2))
        # 断言结果 res 与预期结果 exp_res 相等
        self.assertEqual(res, exp_res)
    def test_user_getattribute(self):
        class MyObject:
            def __init__(self):
                self.custom_dict = {"a": torch.rand((2, 2))}
                self.my_number = 42

            def __getattribute__(self, name):
                # 调用父类的 __getattribute__ 方法获取自定义字典 custom_dict
                custom_dict = super().__getattribute__("custom_dict")
                # 如果属性名在 custom_dict 中，则返回对应的值
                if name in custom_dict:
                    return custom_dict[name]
                # 否则调用父类的 __getattribute__ 方法返回属性值
                return super().__getattribute__(name)

            def run(self, x):
                # 返回计算结果，使用 self.my_number 和 self.a（从 custom_dict 获取）乘以 x
                return self.my_number * x + self.a * x

        def fn(obj, x):
            # 调用对象的 run 方法
            return obj.run(x)

        obj = MyObject()
        x = torch.rand((2, 2))
        # 创建 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化，得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 断言优化后的函数和原始函数的结果相同
        self.assertTrue(same(opt_fn(obj, x), fn(obj, x)))

    def test_nn_module_getattr(self):
        class MyMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.custom_dict = {"queue": [torch.rand((2, 2)) for _ in range(3)]}
                self.other_attr = torch.rand((2, 2))

            def __getattr__(self, name):
                # 直接访问 self.custom_dict 中的自定义属性
                custom_dict = self.custom_dict
                if name in custom_dict:
                    return custom_dict[name]
                # 否则调用父类的 __getattr__ 方法返回属性值
                return super().__getattr__(name)

            def forward(self, x):
                # 返回计算结果，使用 self.other_attr 和 self.queue[-1] 计算 x 的线性组合
                return x @ self.other_attr + self.queue[-1]

        x = torch.rand((2, 2))
        mod = MyMod()
        # 创建 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 mod 模块进行优化，得到优化后的模块 opt_mod
        opt_mod = torch._dynamo.optimize(cnts)(mod)
        # 断言优化后的模块对输入 x 的计算结果和原始模块的计算结果相同
        self.assertTrue(same(opt_mod(x), mod(x)))
        # 断言帧数为 1
        self.assertTrue(cnts.frame_count, 1)
        # 断言操作数为 2
        self.assertTrue(cnts.op_count, 2)

    def test_nn_module_getattribute(self):
        class MyMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.my_number = 42

            def __getattribute__(self, name):
                # 如果属性名为 "special_attr"，则返回特定的张量
                if name == "special_attr":
                    return torch.tensor([[1, 2], [3, 4]])
                # 否则调用父类的 __getattribute__ 方法返回属性值
                return super().__getattribute__(name)

            def forward(self, x):
                # 返回计算结果，使用 self.my_number 和 self.special_attr 乘以 x
                return self.my_number * x + self.special_attr * x

        def fn(mod, x):
            # 调用模块的 forward 方法
            return mod(x)

        mod = MyMod()
        x = torch.rand((2, 2))
        # 创建 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化，得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 断言优化后的函数和原始函数的结果相同
        self.assertTrue(same(opt_fn(mod, x), fn(mod, x)))

    def test_constant_getattr(self):
        # https://github.com/pytorch/pytorch/issues/97480
        def fn():
            # 获取 None 对象的属性 "arg"，如果不存在则返回 3
            return getattr(None, "arg", 3)

        # 创建 CompileCounter 对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化，得到优化后的函数 optimized_fn
        optimized_fn = torch._dynamo.optimize(cnt)(fn)
        # 调用优化后的函数，得到结果 res
        res = optimized_fn()
        # 断言优化后的结果和预期结果相同
        self.assertTrue(same(res, 3))
    def test_user_property(self):
        # 定义一个内部类 MyConfig，用于测试属性的功能
        class MyConfig:
            @property
            def prop5(self):
                return 5

        # 定义一个函数 fn，接受一个配置对象 cfg，以及两个参数 x 和 y，返回它们的和再加上 cfg 的属性 prop5 的值
        def fn(cfg, x, y):
            return x + y + cfg.prop5

        # 生成一个包含 10 个随机数的张量 x
        x = torch.randn(10)
        # 创建 MyConfig 的一个实例 cfg
        cfg = MyConfig()
        # 创建一个 CompileCounter 的实例 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn 进行优化，返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 断言优化后的函数 opt_fn 对于给定的 cfg、x、x 的调用结果应与 2 * x + 5 相等
        self.assertTrue(same(opt_fn(cfg, x, x), 2 * x + 5))
        # 断言帧计数应为 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言操作计数应为 2
        self.assertEqual(cnts.op_count, 2)

    def test_dataclass_fields(self):
        # 定义一个数据类 MyDataClass，包含五个字段 a、b、c、d、e，其中 a 是必需的，其余字段默认为 None
        @dataclasses.dataclass
        class MyDataClass:
            a: torch.Tensor
            b: torch.Tensor = None
            c: torch.Tensor = None
            d: torch.Tensor = None
            e: torch.Tensor = None

        # 定义函数 fn，接受一个数据类对象 obj，检查其字段的默认值和非 None 的字段总和
        def fn(obj):
            # 获取数据类的字段列表 class_fields
            class_fields = dataclasses.fields(obj)
            # 断言字段列表不为空
            assert len(class_fields)
            # 断言除第一个字段外，其余字段的默认值均为 None
            assert all(field.default is None for field in class_fields[1:])
            # 断言除第一个字段外，其余字段都不为 None
            other_fields_are_none = all(
                getattr(obj, field.name) is None for field in class_fields[1:]
            )
            assert not other_fields_are_none

            # 如果 obj 没有字段 "a"，返回 -1
            if not hasattr(obj, "a"):
                return -1
            # 如果 obj 有字段 "z"，返回 -2
            if hasattr(obj, "z"):
                return -2

            # 初始化 total 为数据类对象 obj 的第一个字段的值
            total = getattr(obj, class_fields[0].name)
            # 遍历除第一个字段外的其他字段，将非 None 的字段值加到 total 上
            for field in class_fields[1:]:
                v = getattr(obj, field.name)
                if v is not None:
                    total += v

            return total

        # 创建两个 MyDataClass 的实例 obj1 和 obj2，分别初始化它们的字段值
        obj1 = MyDataClass(torch.randn(10), torch.randn(10), torch.randn(10))
        obj2 = MyDataClass(torch.randn(10), e=torch.randn(10))
        # 计算 fn 函数对 obj1 和 obj2 的正确输出结果
        correct1 = fn(obj1)
        correct2 = fn(obj2)

        # 创建 CompileCounter 的实例 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化，返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 断言优化后的函数 opt_fn 对于给定的 obj1 的调用结果应与 correct1 相等
        self.assertTrue(same(opt_fn(obj1), correct1))
        # 断言帧计数应为 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言操作计数应为 2
        self.assertEqual(cnts.op_count, 2)

        # 重置动态环境
        torch._dynamo.reset()
        # 重新创建 CompileCounter 的实例 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数再次进行优化，返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 断言优化后的函数 opt_fn 对于给定的 obj2 的调用结果应与 correct2 相等
        self.assertTrue(same(opt_fn(obj2), correct2))
        # 断言帧计数应为 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言操作计数应为 1
        self.assertEqual(cnts.op_count, 1)

        # 测试异常情况：给 obj2 添加一个新属性 "z"，预期结果为 -2
        obj2.z = True
        self.assertEqual(opt_fn(obj2), -2)
    # 定义一个测试函数，用于测试数据类的本地属性是否正常工作
    def test_dataclass_local_hasattr(self):
        # 创建一个编译计数器实例
        cnt = CompileCounter()
        # 生成一个包含随机数的张量
        x = torch.randn(10)

        # 定义一个数据类 MyDataClass，包含两个属性 a 和 b，均为 torch.Tensor 类型
        @dataclasses.dataclass
        class MyDataClass:
            a: torch.Tensor
            b: torch.Tensor

        # 使用装饰器将函数 fn 编译，并指定后端为 cnt，完整图形为 True
        @torch.compile(backend=cnt, fullgraph=True)
        def fn():
            # 创建 MyDataClass 的实例 obj，其属性 a 和 b 分别为 x + 1 和 x - 1
            obj = MyDataClass(x + 1, x - 1)
            # 检查 obj 是否具有属性 "a"，如果没有则返回 -1
            if not hasattr(obj, "a"):
                return -1
            # 检查 obj 是否具有属性 "z"，如果有则返回 -2
            if hasattr(obj, "z"):
                return -2
            # 返回 obj 对象本身
            return obj

        # 执行函数 fn，获取其返回结果
        result = fn()

        # 断言 result 的类型为 MyDataClass
        self.assertIsInstance(result, MyDataClass)
        # 断言 result 的属性 a 等于 x + 1
        self.assertEqual(result.a, x + 1)
        # 断言 result 的属性 b 等于 x - 1
        self.assertEqual(result.b, x - 1)
        # 断言 cnt 的帧计数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言 cnt 的操作计数为 2
        self.assertEqual(cnt.op_count, 2)
    # 定义一个测试函数，测试 numpy 的 take_along_axis 函数
    def test_numpy_take_along_axis(self):
        
        # 定义一个内部函数 fn，使用 numpy 的 take_along_axis 函数
        def fn(x, i, a):
            return np.take_along_axis(x, i, a)

        # 定义一个辅助函数 sample_to_args，将样本 s 转换为参数元组
        def sample_to_args(s):
            args = (s.input, *sample.args)
            return tuple(a.numpy() if isinstance(a, torch.Tensor) else a for a in args)

        # 生成样本列表 samples，使用 sample_inputs_take_along_dim 函数生成
        samples = list(
            sample_inputs_take_along_dim(
                None, "cpu", torch.float32, requires_grad=False
            )
        )

        # 创建一个编译计数器 cnts
        cnts = torch._dynamo.testing.CompileCounter()

        # 对函数 fn 进行优化，得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        
        # 初始化计数器 i 为 1
        i = 1
        
        # 遍历样本列表 samples
        for sample in samples:
            # 将样本转换为参数 args
            args = sample_to_args(sample)
            
            # 如果参数列表长度小于 3，处理第二个参数作为 1 维数组的情况
            if len(args) < 3:
                args = (args[0], np.ravel(args[1]), None)
            
            # 断言原始函数和优化函数的输出结果相等
            self.assertEqual(fn(*args), opt_fn(*args))
            
            # 断言帧计数器的值为 i
            self.assertEqual(cnts.frame_count, i)
            
            # 计数器 i 自增
            i += 1

    # 定义一个测试函数，测试 numpy 与 torch 运算符的组合
    def test_numpy_torch_operators(self):
        
        # 定义一个内部函数 fn，接受操作符 op 和两个张量 t1, t2，返回 op(t1, t2)
        def fn(op, t1, t2):
            return op(t1, t2)

        # 导入内置变量 BuiltinVariable
        from torch._dynamo.variables.builtin import BuiltinVariable

        # 获取内置变量 BuiltinVariable 的函数图操作符列表
        operators = BuiltinVariable._fx_graph_functions()

        # 遍历所有操作符 op 和 t1_np, t2_np 的组合
        for op, t1_np, t2_np in itertools.product(
            operators, (True, False), (True, False)
        ):
            # 如果操作符是 operator.eq 或 operator.ne，跳过处理
            if op in [operator.eq, operator.ne]:
                continue
            
            # 如果操作符是 operator.getitem，跳过处理
            if op is operator.getitem:
                continue
            
            # 如果操作符是 operator.imatmul 并且 t1_np 或 t2_np 为真，跳过处理
            if op is operator.imatmul and (t1_np or t2_np):
                continue
            
            # 创建随机张量 t1
            t1 = torch.rand(5)
            
            # 如果 t1_np 为真，将 t1 转换为 numpy 数组
            if t1_np:
                t1 = t1.numpy()
            
            # 创建随机张量 t2
            t2 = torch.rand(5)
            
            # 如果 t2_np 为真，将 t2 转换为 numpy 数组
            if t2_np:
                t2 = t2.numpy()
            
            try:
                # 尝试执行操作符 op(t1, t2)，捕获可能的异常
                result = op(t1, t2)
            except (RuntimeError, TypeError, IndexError):
                continue
            
            # 创建编译计数器 cnts
            cnts = torch._dynamo.testing.CompileCounter()
            
            # 对函数 fn 进行优化，得到优化后的函数 opt_fn
            opt_fn = torch._dynamo.optimize(cnts)(fn)
            
            # 断言原始函数和优化函数的输出结果相等
            self.assertEqual(result, opt_fn(op, t1, t2), msg=f"{op=} {t1_np=} {t2_np=}")
            
            # 断言帧计数器的值为 1
            self.assertEqual(cnts.frame_count, 1, msg=f"{op=} {t1_np=} {t2_np=}")
            
            # 重置 Torch 内部状态
            torch._dynamo.reset()

    # 定义一个测试函数，测试 numpy 数组与 Torch 张量的结合使用
    def test_numpy_ndarray_graph_break(self):
        
        # 定义一个内部函数 fn，接受一个 Torch 张量 x，返回处理后的结果
        def fn(x):
            # 将 Torch 张量 x 转换为 numpy 数组 a
            a = x.numpy()
            
            # 获取数组 a 的实部 b
            b = a.real
            
            # 断开 Torch 动态图
            torch._dynamo.graph_break()
            
            # 使用 numpy 的 multiply 函数计算 b 乘以 2.0 的结果 c
            c = np.multiply(b, 2.0)
            
            # 返回结果 c
            return c

        # 创建编译计数器 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        
        # 对函数 fn 进行优化，得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        
        # 进行 10 次循环测试
        for _ in range(10):
            # 创建随机 Torch 张量 x
            x = torch.randn(3)
            
            # 计算原始函数 fn(x) 的结果 ref
            ref = fn(x)
            
            # 计算优化函数 opt_fn(x) 的结果 res
            res = opt_fn(x)
            
            # 断言原始函数和优化函数的输出结果相等
            self.assertEqual(ref, res)
        
        # 断言帧计数器的值为 2
        self.assertEqual(cnts.frame_count, 2)
    # 测试函数，用于测试当有多个输出时，numpy ndarray 与 Torch 张量之间的交互是否会中断
    def test_numpy_ndarray_graph_break_with_multiple_outputs(self):
        # 定义一个函数 fn，接受两个输入 x 和 y
        def fn(x, y):
            # 将输入 x 转换为 numpy 数组 a
            a = x.numpy()
            # 将输入 y 转换为 numpy 数组 b
            b = y.numpy()
            # 调用 Torch 内部函数，中断编译图
            torch._dynamo.graph_break()
            # 返回 a 和 b 各自加 1 的结果作为输出
            return np.add(a, 1), np.add(b, 1)

        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 对函数 fn 应用优化器，得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 循环进行测试，重复 10 次
        for _ in range(10):
            # 生成一个形状为 [1, 3] 的随机张量 x
            x = torch.randn([1, 3])
            # 生成一个形状为 [1, 3] 的随机张量 y
            y = torch.randn([1, 3])
            # 调用原始函数 fn 和优化后的函数 opt_fn，获取参考结果 ref 和优化结果 res
            ref = fn(x, y)
            res = opt_fn(x, y)
            # 断言参考结果和优化结果是否相等
            self.assertEqual(ref, res)
        # 断言编译帧计数是否为 2
        self.assertEqual(cnts.frame_count, 2)

    # 测试函数，验证 numpy 数组与 Torch 张量之间的互动情况
    def test_tensor_interacts_with_numpy_ndarray(self):
        # 定义一个函数 fn，接受两个输入 x 和 y
        def fn(x, y):
            # 将输入 x 转换为 numpy 数组 a
            a = x.numpy()
            # 将输入 y 转换为 numpy 数组 b
            b = y.numpy()
            # 创建一个与 a 形状相同的全 1 数组 c
            c = np.ones_like(a)
            # 创建一个与 b 形状相同的全 1 数组 d
            d = np.ones_like(b)
            # 调用 Torch 内部函数，中断编译图
            torch._dynamo.graph_break()
            # 返回 a 加上 c 的结果和 b 加上 d 的结果作为输出
            return np.add(a, c), np.add(b, d)

        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 对函数 fn 应用优化器，得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 循环进行测试，重复 10 次
        for _ in range(10):
            # 生成一个形状为 [1, 3] 的随机张量 x
            x = torch.randn([1, 3])
            # 生成一个形状为 [1, 3] 的随机张量 y
            y = torch.randn([1, 3])
            # 调用原始函数 fn 和优化后的函数 opt_fn，获取参考结果 ref 和优化结果 res
            ref = fn(x, y)
            res = opt_fn(x, y)
            # 断言参考结果和优化结果是否相等
            self.assertEqual(ref, res)
        # 断言编译帧计数是否为 2
        self.assertEqual(cnts.frame_count, 2)

    # 测试函数，验证 numpy 数组与内置函数的交互是否正常工作
    def test_numpy_ndarray_works_with_builtin_function(self):
        # 定义一个函数 fn，接受一个输入 x
        def fn(x):
            # 计算 x 的总和并除以其长度，返回结果作为输出
            v = x.sum() / len(x)
            return v

        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 对函数 fn 应用优化器，开启无 Python 模式，得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 循环进行测试，重复 10 次
        for _ in range(10):
            # 生成一个形状为 [2, 3] 的随机 numpy 数组 x
            x = np.random.randn(2, 3)
            # 调用原始函数 fn 和优化后的函数 opt_fn，获取参考结果 ref 和优化结果 res
            ref = fn(x)
            res = opt_fn(x)
            # 断言参考结果和优化结果是否相等
            self.assertEqual(ref, res)
        # 断言编译帧计数是否为 1
        self.assertEqual(cnts.frame_count, 1)
    def test_numpy_array_of_arrays(self):
        # 定义一个函数 fn，接受两个参数 x 和 y，并返回一个包含 x 和 y 的 NumPy 数组
        def fn(x, y):
            return np.array([x, y])

        # 创建一个编译计数器对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize 对 fn 进行优化，设置 nopython=True
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)

        # 初始化 x 和 y 为 np.float64 类型的数值
        x, y = np.float64(1), np.float64(2)
        # 调用优化后的函数 opt_fn，传入 x 和 y，并将结果保存到 res
        res = opt_fn(x, y)
        # 断言 res 的值为 [1, 2]，并且数据类型为 np.ndarray
        self.assertEqual(res, np.array([1, 2], dtype=float))
        self.assertEqual(type(res), np.ndarray)
        # 断言编译计数器的 frame_count 为 1
        self.assertEqual(cnts.frame_count, 1)

        # 将 x 和 y 初始化为 np.arange(2) 和 np.arange(2) + 2
        x, y = np.arange(2), np.arange(2) + 2
        # 再次调用优化后的函数 opt_fn，传入 x 和 y，并将结果保存到 res
        res = opt_fn(x, y)
        # 断言 res 的值为 [[0, 1], [2, 3]]
        self.assertEqual(res, np.array([[0, 1], [2, 3]]))
        # 断言 res 的数据类型为 np.ndarray
        self.assertEqual(type(res), np.ndarray)
        # 断言编译计数器的 frame_count 现在为 2
        self.assertEqual(cnts.frame_count, 2)

    def test_numpy_readonly(self):
        # 使用装饰器 @torch.compile(fullgraph=True) 定义函数 fn，接受参数 x 并返回 x
        @torch.compile(fullgraph=True)
        def fn(x):
            return x

        # 将 x 初始化为 np.arange(3) 扩展到形状为 (2, 3) 的数组
        x = np.broadcast_to(np.arange(3), (2, 3))
        # 断言 x 不可写入
        self.assertFalse(x.flags.writeable)

        # 设置警告过滤器，捕获所有警告并抛出错误
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # 调用 fn，传入 x，并将结果保存到 y
            y = fn(x)
        # 断言 y 可写入，这与 NumPy 不同
        self.assertTrue(y.flags.writeable)  # XXX: differs from numpy

    def test_numpy_tolist(self):
        # 定义一个函数 fn，接受参数 x，并返回 x 的 tolist() 方法的结果
        def fn(x):
            return x.tolist()

        # 创建一个编译计数器对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize 对 fn 进行优化，设置 nopython=True
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)

        # 初始化 x 为 np.arange(5)
        x = np.arange(5)
        # 调用优化后的函数 opt_fn，传入 x，并将结果保存到 r
        r = opt_fn(x)

        # 断言 r 的值为 [0, 1, 2, 3, 4]
        self.assertEqual(r, [0, 1, 2, 3, 4])
        # 断言 r 的数据类型为 list
        self.assertEqual(type(r), list)
        # 断言编译计数器的 frame_count 为 1
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_size_attr(self):
        # 定义一个函数 fn，接受参数 x，并返回 x.size + x 的结果
        def fn(x):
            return x.size + x

        # 创建一个编译计数器对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize 对 fn 进行优化，设置 nopython=True
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)

        # 初始化 x 为 np.arange(5)
        x = np.arange(5)
        # 调用优化后的函数 opt_fn，传入 x，并将结果保存到 r
        r = opt_fn(x)

        # 断言 r 的值等于 fn(x) 的值
        self.assertEqual(r, fn(x))
        # 断言 r 的数据类型为 np.ndarray
        self.assertEqual(type(r), np.ndarray)
        # 断言编译计数器的 frame_count 为 1
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_no_raise(self):
        # 定义一个函数 _inf_nan_preprocess，接受两个参数 t 和 t_np
        def _inf_nan_preprocess(t, t_np):
            # 使用 np.nan_to_num 处理 t_np 中的 NaN 和 Inf，将结果保存到 t_np
            t_np = np.nan_to_num(t_np)
            # 返回 t 和处理后的 t_np
            return t, t_np

        # 定义函数 fn，没有参数，用于测试不同形状的随机张量和数组
        def fn():
            # 定义测试案例列表 test_cases
            test_cases = (
                (3, 3),
                (4, 4),
                (5, 5),
            )

            # 遍历测试案例
            for shape in test_cases:
                # 创建形状为 shape 的随机张量 t，数据类型为 torch.complex64
                t = torch.randn(shape, dtype=torch.complex64)
                # 创建形状为 shape 的随机数组 t_np，数据类型为 np.complex64
                t_np = np.random.randn(*shape).astype(np.complex64)

                # 调用 _inf_nan_preprocess 处理 t 和 t_np，将结果保存到 _, t_np
                _, t_np = _inf_nan_preprocess(t, t_np)
                # 打印 t 和 t_np，这只是为了触发编译过程的副作用
                print(t, t_np)  # Just a side effect so that compilation kicks in

        # 创建一个带有后端计数器的编译计数器对象 cnt
        cnt = CompileCounterWithBackend("inductor")
        # 使用 torch._dynamo.optimize 对 fn 进行优化，并调用优化后的函数 fn
        fn = torch._dynamo.optimize(cnt)(fn)
        fn()
        # 断言编译计数器的 frame_count 的值，根据条件选择不同的预期值
        self.assertEqual(cnt.frame_count, ifdynstaticdefault(2, 1))
    def test_mandelbrot_numpy(self):
        def mandelbrot_numpy(max_iter):
            # 定义复平面的边界
            xn = 450
            yn = 375
            xmin = -2.25
            xmax = 0.75
            ymin = -1.25
            ymax = 1.25

            # 创建复数网格
            x_values = np.linspace(xmin, xmax, xn, dtype=np.float64)
            y_values = np.linspace(ymin, ymax, yn, dtype=np.float64)
            rx, iy = np.meshgrid(x_values, y_values, indexing="xy")

            x = rx.copy()
            y = iy.copy()
            mask = np.zeros_like(x)
            for i in range(max_iter):
                x_prev = x
                y_prev = y
                # 计算 Mandelbrot 集合的下一个迭代
                x = x_prev**2 - y_prev**2 + rx
                y = 2 * x_prev * y_prev + iy
                inside = np.sqrt(x**2 + y**2) <= 2
                mask += inside
            return mask

        cnts = torch._dynamo.testing.CompileCounter()
        # 优化函数，用于编译和优化 mandelbrot_numpy 函数
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(mandelbrot_numpy)
        n_iter = torch._dynamo.config.cache_size_limit - 2
        for i in range(n_iter):
            x = i + 3
            ref = mandelbrot_numpy(x)
            res = opt_fn(x)
            # 断言优化前后的结果一致
            self.assertEqual(ref, res)
        # 断言编译器的帧计数与预期的迭代次数一致
        self.assertEqual(cnts.frame_count, n_iter)

    def test_numpy_as_global(self):
        global x
        x = np.arange(10)

        @torch.compile(fullgraph=True)
        def fn(y):
            return y + x + x

        r = fn(np.arange(10))
        # 断言返回结果类型为 ndarray
        self.assertEqual(type(r), np.ndarray)
        # 断言返回结果与预期结果一致
        self.assertEqual(r, x * 3)
        del x

    def test_numpy_gt(self):
        x = np.arange(10)

        @torch.compile
        def fn(y):
            return y >= 3

        r = fn(x)
        # 断言返回结果类型为 ndarray
        self.assertEqual(type(r), np.ndarray)
        # 断言返回结果与预期结果一致
        self.assertEqual(r, x >= 3)

    def test_numpy_min(self):
        x = np.arange(10)

        @torch.compile
        def fn(y):
            return min(y, 3), min(y, y - 1)

        r1, r2 = fn(x)
        # 断言返回结果类型为 ndarray
        self.assertEqual(type(r1), np.ndarray)
        self.assertEqual(type(r2), np.ndarray)
        # 断言返回结果与预期结果一致
        self.assertEqual(r1, np.minimum(x, 3))
        self.assertEqual(r2, np.minimum(x, x - 1))
    # 定义一个测试方法，验证将 NumPy ndarray 传递给 Torch 函数时，图形正确中断
    def test_graph_break_correctly_when_passing_numpy_ndarray_to_torch_function(self):
        # 定义一个函数 fn，接受一个整数 x 和一个 Torch 张量 y 作为参数
        def fn(x: int, y: torch.Tensor):
            # 创建一个包含一个形状为 [2, x] 的全 1 NumPy ndarray 的列表
            ndarray_list = [np.ones([2, x])]
            # 将列表中的 ndarray 堆叠起来，创建一个新的 ndarray
            ndarray = np.stack(ndarray_list, axis=0)
            # 将 ndarray 转换为 Torch 张量，并指定数据类型为 long
            tensor = torch.tensor(ndarray, dtype=torch.long)
            # 在第 0 维度上增加一个维度
            tensor.unsqueeze_(0)
            # 返回 tensor 加上 y 的结果
            return tensor + y

        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化函数 fn 并返回一个新的优化函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 对于 x 在 1 到 9 的范围内
        for x in range(1, 10):
            # 创建一个形状为 [1, 2, x] 的随机 Torch 张量 y
            y = torch.randn([1, 2, x])
            # 计算原始函数 fn 和优化函数 opt_fn 在相同输入下的结果
            ref = fn(x, y)
            res = opt_fn(x, y)
            # 断言优化函数的输出与原始函数相同
            self.assertEqual(ref, res)
        # 检查编译计数器中记录的帧数是否符合预期的动态/静态条件
        self.assertEqual(cnts.frame_count, ifdynstaticdefault(3, 2))

    # 定义一个测试方法，验证 NumPy 数组与内置类型的混合操作
    def test_numpy_with_builtin_type(self):
        # 创建一个长度为 5 的随机 NumPy 数组 x
        x = np.random.rand(5)

        # 定义一个函数 fn，接受一个输入参数 x
        def fn(x):
            # 将输入数组 x 扩展操作：乘以 5，转换为 bool 类型，再转换为 float 类型，最后转换为 int 类型，再加上 8
            return (x * 5).astype(bool).astype(float).astype(int) + 8

        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化函数 fn 并返回一个新的优化函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)

        # 调用优化函数 opt_fn，并断言返回结果的数据类型为 int
        r = opt_fn(x)
        self.assertEqual(r.dtype, int)
        # 检查编译计数器中记录的帧数是否符合预期
        self.assertEqual(cnts.frame_count, 1)

    # 定义一个测试方法，验证 Torch 张量与内置类型的混合操作
    def test_with_builtin_type(self):
        # 创建一个长度为 5 的随机 Torch 张量 x
        x = torch.randn(5)

        # 定义一个函数 fn，接受一个输入参数 x
        def fn(x):
            # 将输入张量 x 扩展操作：乘以 5，转换为 bool 类型，再转换为 float 类型，最后转换为 int 类型，再加上 8
            return (x * 5).to(bool).to(float).to(int) + 8

        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化函数 fn 并返回一个新的优化函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)

        # 调用优化函数 opt_fn，并断言返回结果的数据类型为 torch.int64
        r = opt_fn(x)
        self.assertEqual(r.dtype, torch.int64)
        # 检查编译计数器中记录的帧数是否符合预期
        self.assertEqual(cnts.frame_count, 1)

    # 定义一个测试方法，验证 NumPy 数组的唯一化操作
    def test_numpy_unique_f16(self):
        # 定义一个函数 fn，不接受任何输入参数
        def fn():
            # 创建一个包含 [1, 1, 2, 2, 3] 的 NumPy 数组，并指定数据类型为 np.float16
            x = np.asarray([1, 1, 2, 2, 3], dtype=np.float16)
            # 返回数组 x 的唯一值
            return np.unique(x)

        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化函数 fn 并返回一个新的优化函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)

        # 调用优化函数 opt_fn，并断言返回结果的数据类型为 np.float16
        r = opt_fn()
        self.assertEqual(r.dtype, np.float16)
        # 检查编译计数器中记录的帧数是否符合预期
        self.assertEqual(cnts.frame_count, 1)

    # 定义一个测试方法，验证 NumPy 数组在 eager 模式下的 fallback 行为
    def test_numpy_fallback_on_eager(self):
        # 定义一个函数 fn，不接受任何输入参数
        def fn():
            # 返回一个包含字符串 "L" 和 "U" 的 NumPy 数组
            return np.asarray(["L", "U"])

        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化函数 fn 并返回一个新的优化函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)

        # 调用优化函数 opt_fn，并断言编译计数器中记录的帧数为 0，表示图形中断
        r = opt_fn()
        self.assertEqual(cnts.frame_count, 0)
        # 断言优化函数的返回结果与原始函数的返回结果相同
        self.assertEqual(r, np.asarray(["L", "U"]))

        # 定义另一个函数 fn2，不接受任何输入参数
        def fn2():
            # 从 ["L", "U"] 中随机选择一个元素返回
            return np.random.choice(["L", "U"])

        # 创建一个新的编译计数器对象
        cnts2 = torch._dynamo.testing.CompileCounter()
        # 优化函数 fn2 并返回一个新的优化函数 opt_fn2
        opt_fn2 = torch._dynamo.optimize(cnts2)(fn2)

        # 调用原始函数 fn2，并断言编译计数器 cnts 的帧数仍然为 0
        r2 = fn2()
        self.assertEqual(cnts.frame_count, 0)
        # 断言返回结果在 "L" 和 "U" 中
        assert r2 in ("L", "U")

    # 定义一个测试方法，验证对 NumPy 数组进行跟踪的帧数
    def test_trace_ndarray_frame(self):
        # 定义一个函数 fn，接受一个输入参数 x
        def fn(x):
            # 将输入数组 x 的每个元素平方后乘以 2，并打印一条消息
            x = x**2
            print("graph break.")
            return 2 * x

        # 创建一个编译计数器对象
        counter = CompileCounter()
        # 优化函数 fn 并返回一个新的优化函数 compiled_fn
        compiled_fn = torch._dynamo.optimize(counter)(fn)

        # 创建一个长度为 8 的 NumPy 数组 x
        x = np.arange(8)
        # 调用原始函数 fn 和优化函数 compiled_fn，并断言它们的输出结果相同
        self.assertEqual(fn(x), compiled_fn(x))
        # 检查编译计数器中记录的帧数是否符合预期，应为 2
        self.assertEqual(counter.frame_count, 2)
    # 定义测试函数，验证不包含张量或 ndarray 的情况下的行为
    def test_trace_ndarray_frame_2(self):
        # 内部函数 fn 接受参数 x，并打印信息 "graph break."
        def fn(x):
            print("graph break.")
            # 返回一个 ndarray，其值为 0 到 2*x-1 的偶数
            return 2 * np.arange(x)

        # 创建 CompileCounter 实例
        counter = CompileCounter()
        # 使用 torch._dynamo.optimize 优化 fn 函数
        compiled_fn = torch._dynamo.optimize(counter)(fn)

        # 测试 fn 和 compiled_fn 在相同输入 x 下的输出是否相等
        x = 8
        self.assertEqual(fn(x), compiled_fn(x))
        # 验证 CompileCounter 记录的 frame_count 是否为 1
        self.assertEqual(counter.frame_count, 1)

    # 测试处理不具有对应 PyTorch 等效项的 dtype 时的行为
    def test_numpy_non_torch_dtype(self):
        # 内部函数 fn 判断 x 是否为 torch.Tensor 类型
        def fn(x):
            return isinstance(x, torch.Tensor)

        # 创建 CompileCounter 实例
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize 优化 fn 函数
        opt_fn = torch._dynamo.optimize(cnts)(fn)

        # 遍历不同的 uint16 类型数据，验证 opt_fn 的返回值和 frame_count 记录
        for x in [np.array([42], dtype=np.uint16), np.uint16(42), np.dtype("uint16")]:
            r = opt_fn(x)

            self.assertEqual(r, False)
            # 验证 CompileCounter 记录的 frame_count 是否为 0，表示出现了图形断裂
            self.assertEqual(cnts.frame_count, 0)  # graph break

    # 测试对 ndarray 进行迭代是否产生 ndarray 而不是裸张量
    def test_numpy_iter(self):
        # 内部函数 fn 返回 x 中每个元素的列表
        def fn(x):
            return [bm for bm in x]

        # 创建 CompileCounter 实例
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize 优化 fn 函数
        opt_fn = torch._dynamo.optimize(cnts)(fn)

        # 创建包含 0 到 2 的 3 行 1 列的 ndarray
        proba_map = np.arange(3)[:, None]
        # 执行 opt_fn 函数
        res = opt_fn(proba_map)

        # 验证 res 中每个元素的类型是否为 np.ndarray
        self.assertEqual([type(r) for r in res], [np.ndarray, np.ndarray, np.ndarray])
        # 验证 CompileCounter 记录的 frame_count 是否为 1
        self.assertEqual(cnts.frame_count, 1)

    # 配置缓存大小限制大于 dtypes 列表的大小
    @torch._dynamo.config.patch(cache_size_limit=12)
    def test_dtypes_no_graphbreaks(self):
        # 定义包含各种 dtype 的列表 dtypes
        dtypes = [
            # 浮点数
            float,
            np.float64,
            "float64",
            np.float32,
            "float32",
            # 整数
            int,
            "int",
            np.intp,
            np.int32,
            np.uint8
        ]

        # 内部函数 fn 根据给定的 dtype 返回一个 ndarray
        def fn(dt):
            return np.arange(5, dtype=dt)

        # 遍历 dtypes 列表，验证 fn 和 opt_fn 的行为及 CompileCounter 的记录
        for dtyp in dtypes:
            cnts = torch._dynamo.testing.CompileCounter()
            opt_fn = torch._dynamo.optimize(cnts)(fn)

            val = fn(dtyp)
            opt_val = opt_fn(dtyp)

            # 验证 CompileCounter 记录的 frame_count 是否为 1，表示没有图形断裂
            self.assertEqual(cnts.frame_count, 1)  # no graph break

    # 设置配置值使得 PRNG 与 numpy 的一致
    # 注意：这可能涉及到图形断裂
    @torch._dynamo.config.patch(use_numpy_random_stream=True)
    def test_numpy_random_config_to_numpy(self):
        # 使用 @torch.compile 修饰的函数 fn 返回一个服从均匀分布的 13 元素的 ndarray
        @torch.compile
        def fn():
            return np.random.uniform(size=13)

        # 验证 fn() 返回的 ndarray 的形状是否为 (13,)
        self.assertEqual(fn().shape, (13,))
    def test_inplace_view_on_graph_input(self):
        # 测试对图输入进行就地视图的方法时，可能会导致图断裂
        func_args_map = {
            # 改变大小并就地乘以2
            lambda x: x.resize_(6).mul_(2): torch.ones(4),
            # 转置并就地乘以2
            lambda x: x.t_().mul_(2): torch.rand(2, 3),
            # 交换维度并就地乘以2
            lambda x: x.transpose_(0, 1).mul_(2): torch.rand(2, 3),
            # 消除大小为1的维度并就地乘以2
            lambda x: x.squeeze_().mul_(2): torch.rand(1, 2, 3),
            # 在维度0处添加大小为1的维度并就地乘以2
            lambda x: x.unsqueeze_(0).mul_(2): torch.rand(2, 3),
            # 调整大小为与另一个张量相同并就地乘以2
            lambda x: x.resize_as_(torch.rand(200, 300)): torch.rand(2, 3),
            # 交换轴并就地乘以2
            lambda x: x.swapaxes_(0, 1).mul_(2): torch.rand(2, 3),
            # 交换维度并就地乘以2
            lambda x: x.swapdims_(0, 1).mul_(2): torch.rand(2, 3),
            # 重命名维度并就地乘以2
            lambda x: x.rename_("N", "C").mul_(2): torch.zeros(2, 3),
            # 使用给定的步幅和形状创建视图并就地乘以2
            lambda x: x.as_strided_((3, 2), (2, 1)).mul_(2): torch.zeros(2, 3),
            # 分离出视图并就地乘以2
            lambda x: x.detach_().mul_(2): torch.zeros(2, 3),
        }
        for func, args in func_args_map.items():
            args_clone = args.clone()
            cnts = torch._dynamo.testing.CompileCounter()
            opt_f = torch._dynamo.optimize(cnts)(func)
            self.assertTrue(same(func(args).shape, opt_f(args_clone).shape))
            self.assertEqual(cnts.frame_count, 1)
            # 断言操作次数为1（即乘法操作）
            self.assertEqual(cnts.op_count, 1)
    def test_dict_mutation_side_effect(self):
        # 定义一个操作字典的函数，将键"c"设置为键"a"的值加上键"b"的值，并删除键"b"
        def fn(d):
            d["c"] = d["a"] + d.pop("b")
            return d

        # 初始化字典args1和args2，args2是args1的副本
        args1 = {"a": torch.randn(10), "b": torch.randn(10)}
        args2 = dict(args1)
        
        # 断言fn(args1)返回的字典与args1是同一个对象
        assert fn(args1) is args1
        
        # 创建一个计数器对象cnts
        cnts = torch._dynamo.testing.CompileCounter()
        
        # 对函数fn进行优化，返回优化后的函数opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        
        # 断言优化后的函数opt_fn对args2的操作返回结果仍为args2本身
        self.assertIs(opt_fn(args2), args2)
        
        # 断言args1和args2的内容相同
        self.assertTrue(same(args1, args2))
        
        # 断言计数器对象cnts的frame_count为1
        self.assertEqual(cnts.frame_count, 1)
        
        # 断言计数器对象cnts的op_count为1
        self.assertEqual(cnts.op_count, 1)

    def test_dict_order_keys(self):
        # 定义一个函数fn，对字典d中所有值求和
        def fn(d):
            c = 0
            for v in d.values():
                c += v
            return c

        # 初始化空字典args1，并向其中添加两个键值对
        args1 = {}
        args1["a"] = torch.rand(10)
        args1["b"] = torch.rand(10)
        
        # 创建一个计数器对象cnts
        cnts = torch._dynamo.testing.CompileCounter()
        
        # 对函数fn进行优化，返回优化后的函数opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        
        # 断言fn(args1)与opt_fn(args1)的返回结果相等
        self.assertEqual(fn(args1), opt_fn(args1))
        
        # 断言计数器对象cnts的frame_count为1
        self.assertEqual(cnts.frame_count, 1)
        
        # 断言计数器对象cnts的op_count为2
        self.assertEqual(cnts.op_count, 2)

        # 改变args1中键的顺序会导致重新编译
        args2 = {}
        args2["b"] = args1["b"]
        args2["a"] = args1["a"]
        
        # 断言fn(args2)与opt_fn(args2)的返回结果相等
        self.assertEqual(fn(args2), opt_fn(args2))
        
        # 断言计数器对象cnts的frame_count为2
        self.assertEqual(cnts.frame_count, 2)
        
        # 额外的调用不会导致重新编译
        self.assertEqual(cnts.frame_count, 2)

    def test_dict_namedtuple(self):
        # 定义一个函数fn，返回字典d中键为3的值乘以2
        def fn(d):
            return d[3] * 2

        # 初始化args1，包含一个命名元组作为键和一个张量作为值
        args1 = {collections.namedtuple: None, 3: torch.randn(3)}
        
        # 创建一个计数器对象cnts
        cnts = torch._dynamo.testing.CompileCounter()
        
        # 对函数fn进行优化，返回优化后的函数opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        
        # 断言fn(args1)与opt_fn(args1)的返回结果相等
        self.assertEqual(fn(args1), opt_fn(args1))
        
        # 断言计数器对象cnts的frame_count为1
        self.assertEqual(cnts.frame_count, 1)
        
        # 测试失败的命名元组保护
        args2 = {2: None, 3: torch.randn(3)}
        
        # 断言fn(args2)与opt_fn(args2)的返回结果相等
        self.assertEqual(fn(args2), opt_fn(args2))
        
        # 断言计数器对象cnts的frame_count为2
        self.assertEqual(cnts.frame_count, 2)

    def test_dict_order_keys_tensors(self):
        # 定义一个函数fn，返回字典d中键x对应的值加上3
        def fn(d, x):
            return d[x] + 3

        # 初始化空字典args1，定义三个张量x、y、z
        args1 = {}
        x = torch.randn(10)
        y = torch.randn(10)
        z = torch.randn(10)
        
        # 向args1中添加两个键值对，一个键是张量x，另一个键是整数3，对应的值分别是张量y和张量z
        args1[x] = y
        args1[3] = z

        # 创建一个计数器对象cnts
        cnts = torch._dynamo.testing.CompileCounter()
        
        # 对函数fn进行优化，返回优化后的函数opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        
        # 断言fn(args1, x)与opt_fn(args1, x)的返回结果相等
        self.assertEqual(fn(args1, x), opt_fn(args1, x))
        
        # 断言计数器对象cnts的frame_count为1
        self.assertEqual(cnts.frame_count, 1)

        # 再次调用不会重新编译（相同的id和键顺序）
        opt_fn(args1, x)
        self.assertEqual(cnts.frame_count, 1)
        
        # 初始化空字典args2
        args2 = {}
        
        # 向args2中添加两个键值对，一个键是整数3，对应的值是张量z；另一个键是张量x，对应的值是张量y
        args2[3] = z
        args2[x] = y
        
        # 不同的键顺序会导致重新编译
        self.assertEqual(fn(args2, x), opt_fn(args2, x))
        
        # 断言计数器对象cnts的frame_count为2
        self.assertEqual(cnts.frame_count, 2)
    def test_dict_order_keys_modules(self):
        # 定义一个测试函数，接受字典和键作为参数
        def fn(d, x):
            return d[x](torch.ones(2, 2))

        # 初始化一个空字典 args1
        args1 = {}
        # 创建三个线性层对象
        x = torch.nn.Linear(2, 2)
        y = torch.nn.Linear(2, 2)
        z = torch.nn.Linear(2, 2)
        # 将 x 对应的值设置为 y
        args1[x] = y
        # 将整数 3 对应的值设置为 z
        args1[3] = z

        # 创建一个编译计数器对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 cnts 优化函数 fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 断言 fn(args1, x) 和 opt_fn(args1, x) 的结果相等
        self.assertEqual(fn(args1, x), opt_fn(args1, x))
        # 断言帧计数为 1
        self.assertEqual(cnts.frame_count, 1)

        # 再次调用不会重新编译（相同的 id 和键顺序）
        opt_fn(args1, x)
        # 断言帧计数仍为 1
        self.assertEqual(cnts.frame_count, 1)
        
        # 初始化一个空字典 args2
        args2 = {}
        # 将整数 3 对应的值设置为 z
        args2[3] = z
        # 将 x 对应的值设置为 y
        args2[x] = y

        # 不同的顺序会导致重新编译
        self.assertEqual(fn(args2, x), opt_fn(args2, x))
        # 断言帧计数增加到 2
        self.assertEqual(cnts.frame_count, 2)

    def test_dunder_new_function_inlining(self):
        # 清除计数器
        counters.clear()

        # 定义一个继承自 torch.nn.Module 的模型 ModelA
        class ModelA(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.tanh(x + 1)

        # 定义一个使用 __new__ 方法创建 ModelA 实例的模型 ModelB
        class ModelB(torch.nn.Module):
            def __new__(cls):
                return ModelA()

        # 定义一个包含线性层的模型 Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 2)

            def forward(self, x):
                other = ModelB()
                return self.layer(x) + other(x)

        # 生成随机输入 x
        x = torch.rand(2, 2)
        # 创建 Model 的实例 m
        m = Model()

        # 使用 eager 模式编译模型 m，得到 opt_m
        opt_m = torch.compile(backend="eager")(m)
        # 计算参考结果 ref
        ref = m(x)
        # 计算优化后模型 opt_m 的结果 res
        res = opt_m(x)
        # 断言 ref 和 res 结果相同
        self.assertTrue(same(ref, res))
        # 断言 "graph_break" 计数器的长度为 1
        self.assertEqual(len(counters["graph_break"]), 1)
        # 断言 "super() nn.Module.__init__" 不在 "graph_break" 计数器中
        self.assertFalse("super() nn.Module.__init__" in counters["graph_break"])

    def test_class_duner_mro(self):
        # 定义 ModuleA 类，继承自 torch.nn.Module
        class ModuleA(torch.nn.Module):
            pass

        # 定义 ModuleB 类，继承自 ModuleA
        class ModuleB(ModuleA):
            pass

        # 定义函数 fn，接受输入 x 和模块 mod
        def fn(x, mod):
            # 如果 ModuleA 在 mod 的类型 MRO 中，则返回 x + 1
            if ModuleA in type(mod).__mro__:
                return x + 1
            else:
                return x - 1

        # 生成随机输入 x
        x = torch.rand(2, 3)
        # 创建 ModuleB 的实例 mod
        mod = ModuleB()
        # 使用 eager 模式和完整图形编译 fn 函数，得到 opt_fn
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        # 计算参考结果 ref
        ref = fn(x, mod)
        # 计算优化后函数 opt_fn 的结果 res
        res = opt_fn(x, mod)
        # 断言 ref 和 res 结果相同
        self.assertTrue(same(ref, res))
    def test_nested_wraps(self):
        # 定义嵌套函数 foo，接受两个参数 x 和 y
        def foo(x, y):
            # 定义内部函数 add，接受两个参数 x 和 y，返回它们的和
            def add(x, y):
                return x + y

            # 使用 functools.wraps 对 add 进行包装，定义 wrapped_call 函数
            @functools.wraps(add)
            def wrapped_call(x, y):
                return add(x, y)

            # 返回 wrapped_call 的调用结果
            return wrapped_call(x, y)

        # 生成一个 3x3 的随机张量 x 和 y
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        # 使用 torch.compile 对 foo 进行编译，fullgraph=True 表示编译整个图，backend="eager" 表示使用 eager 模式
        o = torch.compile(foo, fullgraph=True, backend="eager")(x, y)
        # 断言 o 等于 x + y
        self.assertEqual(o, x + y)

        # 定义嵌套函数 foo，接受两个参数 x 和 y
        def foo(x, y):
            # 定义内部函数 nested_call，接受两个参数 x 和 y
            def nested_call(x, y):
                # 定义内部函数 mul，接受两个参数 x 和 y，返回它们的乘积
                def mul(x, y):
                    return x * y

                # 使用 functools.wraps 对 mul 进行包装，定义 double_nested_call 函数
                @functools.wraps(mul)
                def double_nested_call(x, y):
                    return mul(x, y)

                # 返回 double_nested_call 的调用结果
                return double_nested_call(x, y)

            # 返回 nested_call 的调用结果
            return nested_call(x, y)

        # 使用 torch.compile 对 foo 进行编译，fullgraph=True 表示编译整个图，backend="eager" 表示使用 eager 模式
        o = torch.compile(foo, fullgraph=True, backend="eager")(x, y)
        # 断言 o 等于 x * y
        self.assertEqual(o, x * y)

    def test_module_deepcopy(self):
        # 创建两个相同结构的神经网络模型 m1 和 m2
        m1 = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )
        m2 = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )

        # 定义函数 fn，接受模型 m 和输入向量 x
        def fn(m, x):
            # 使用 copy.deepcopy 对模型 m 进行深拷贝
            m_copy = copy.deepcopy(m)
            # 返回拷贝模型 m_copy 对输入 x 的调用结果
            return m_copy(x)

        # 生成一个长度为 10 的随机向量 v
        v = torch.randn(10)
        # 分别使用 fn 函数对 m1 和 m2 进行调用，并保存结果
        correct1 = fn(m1, v)
        correct2 = fn(m2, v)

        # 创建一个 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize 对 fn 函数进行优化
        opt_fn = torch._dynamo.optimize(cnts)(fn)

        # 对 m1 和 m2 进行多次优化后的调用，断言结果正确
        for _ in range(10):
            self.assertTrue(same(opt_fn(m1, v), correct1))
        for _ in range(10):
            self.assertTrue(same(opt_fn(m2, v), correct2))

        # 断言优化过程中帧数和操作数的正确性
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    def test_type_copy(self):
        # 定义函数 fn，接受一个序列 seq
        def fn(seq):
            # 将序列 seq 拆解为两个元素 a 和 b
            a, b = seq
            # 返回一个与 seq 类型相同的新序列，包含对 a+1、b+2 和 a+b 的计算结果
            return type(seq)([a + 1, b + 2, a + b])

        # 创建两个长度为 10 的随机向量 args1 和 args2
        args1 = [torch.randn(10), torch.randn(10)]
        args2 = (torch.randn(10), torch.randn(10))
        # 分别使用 fn 函数对 args1 和 args2 进行调用，并保存结果
        correct1 = fn(args1)
        correct2 = fn(args2)

        # 创建一个 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize 对 fn 函数进行优化
        opt_fn = torch._dynamo.optimize(cnts)(fn)

        # 断言优化后的调用结果与正确结果相同
        self.assertTrue(same(opt_fn(args1), correct1))
        self.assertTrue(same(opt_fn(args2), correct2))
        # 断言优化后返回的对象类型分别为 list 和 tuple
        self.assertIsInstance(opt_fn(args1), list)
        self.assertIsInstance(opt_fn(args2), tuple)
        # 断言优化过程中帧数和操作数的正确性
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 6)
    def test_setattr_mutation1(self):
        # 定义一个简单的类 MyObj，用来测试属性设置的变化
        class MyObj:  # noqa: B903
            def __init__(self, a, b):
                self.a = a
                self.b = b

        # 定义一个函数 fn，接受一个对象 obj，并进行一系列属性设置和计算
        def fn(obj):
            # 设置对象的属性 c 为 a * b + 1
            obj.c = obj.a * obj.b + 1
            # 设置对象的属性 b 为 a * c + 2
            obj.b = obj.a * obj.c + 2
            # 设置对象的属性 a 为 b * c + 3
            obj.a = obj.b * obj.c + 3
            # 再次设置对象的属性 c 为 a * b + 4
            obj.c = obj.a * obj.b + 4
            # 再次设置对象的属性 b 为 a * c + 5
            obj.b = obj.a * obj.c + 5
            # 最后设置对象的属性 a 为 b * c + 6
            obj.a = obj.b * obj.c + 6
            # 返回修改后的对象
            return obj

        # 生成两个随机张量对象 x1 和 x2
        x1 = torch.randn(10)
        x2 = torch.randn(10)
        # 分别使用 x1 和 x2 初始化 MyObj 的实例 obj1 和 obj2
        obj1 = MyObj(x1, x2)
        obj2 = MyObj(x1, x2)
        # 对 obj2 应用函数 fn，使其属性发生变化
        fn(obj2)
        # 创建一个编译计数器对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化函数 fn，并返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 断言优化后的函数对 obj1 的返回结果与 obj1 本身相同
        self.assertIs(opt_fn(obj1), obj1)
        # 断言 obj1 和 obj2 的属性 a 相同
        self.assertTrue(same(obj1.a, obj2.a))
        # 断言 obj1 和 obj2 的属性 b 相同
        self.assertTrue(same(obj1.b, obj2.b))
        # 断言 obj1 和 obj2 的属性 c 相同
        self.assertTrue(same(obj1.c, obj2.c))
        # 断言帧数计数器 cnts 的帧数为 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言操作计数器 cnts 的操作数为 12
        self.assertEqual(cnts.op_count, 12)

    def test_setattr_mutation2(self):
        # 定义一个简单的类 MyObj，用来测试属性设置的变化
        class MyObj:
            def __init__(self, x):
                self.a = x + 1
                self.b = x + 2

        # 定义一个函数 fn，接受一个参数 x，并返回一个 MyObj 的实例
        def fn(x):
            # 对参数 x 进行除法运算
            x = x / 3.0
            # 使用 x 初始化 MyObj 的实例 obj
            obj = MyObj(x)
            # 设置对象的属性 c 为 a * b + 1
            obj.c = obj.a * obj.b + 1
            # 设置对象的属性 b 为 a * c + 2
            obj.b = obj.a * obj.c + 2
            # 设置对象的属性 a 为 b * c + 3
            obj.a = obj.b * obj.c + 3
            # 返回修改后的对象
            return obj

        # 生成一个随机张量对象 x1
        x1 = torch.randn(10)
        # 调用函数 fn，生成一个 MyObj 的实例 obj2
        obj2 = fn(x1)

        # 创建一个编译计数器对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化函数 fn，并返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 调用优化后的函数 opt_fn，传入参数 x1，生成一个 MyObj 的实例 obj1
        obj1 = opt_fn(x1)
        # 断言 obj1 和 obj2 的属性 a 相同
        self.assertTrue(same(obj1.a, obj2.a))
        # 断言 obj1 和 obj2 的属性 b 相同
        self.assertTrue(same(obj1.b, obj2.b))
        # 断言 obj1 和 obj2 的属性 c 相同
        self.assertTrue(same(obj1.c, obj2.c))
        # 断言帧数计数器 cnts 的帧数为 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言操作计数器 cnts 的操作数为 9
        self.assertEqual(cnts.op_count, 9)

    def test_setattr_mutation3(self):
        # TODO(jansel): dead code eliminate the object creation
        # 定义一个简单的类 MyObj，用来测试属性设置的变化
        class MyObj:
            def __init__(self, x):
                super().__init__()
                self.a = x + 1
                self.b = x + 2

        # 定义一个函数 fn，接受一个参数 x，并返回 MyObj 的实例的三个属性值
        def fn(x):
            # 对参数 x 进行除法运算
            x = x / 3.0
            # 使用 x 初始化 MyObj 的实例 obj
            obj = MyObj(x)
            # 设置对象的属性 c 为 a * b + 1
            obj.c = obj.a * obj.b + 1
            # 设置对象的属性 b 为 a * c + 2
            obj.b = obj.a * obj.c + 2
            # 设置对象的属性 a 为 b * c + 3
            obj.a = obj.b * obj.c + 3
            # 返回 obj 的三个属性值
            return obj.a, obj.b, obj.c

        # 生成一个随机张量对象 x1
        x1 = torch.randn(10)
        # 调用函数 fn，生成一个包含三个属性值的元组 obj2
        obj2 = fn(x1)

        # 创建一个编译计数器对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化函数 fn，并返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 调用优化后的函数 opt_fn，传入参数 x1，生成一个包含三个属性值的元组 obj1
        obj1 = opt_fn(x1)
        # 断言 obj1 和 obj2 相等（包含三个属性值的元组）
        self.assertTrue(same(obj1, obj2))
        # 断言帧数计数器 cnts 的帧数为 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言操作计数器 cnts 的操作数为 9
        self.assertEqual(cnts.op_count, 9)
    def test_object_setattr(self):
        @dataclasses.dataclass
        class A:
            x: torch.Tensor

        # 定义函数 fn1，接受参数 x，返回 None
        def fn1(x) -> None:
            # 创建 A 类的实例 a，初始化 x 属性为参数 x
            a = A(x)
            # 使用 object.__setattr__ 设置 a 的 x 属性为 x + 2
            object.__setattr__(a, "x", x + 2)
            return a  # 返回实例 a

        # 生成一个大小为 10 的随机张量 x1
        x1 = torch.randn(10)
        # 调用 fn1 函数，传入 x1 的克隆，得到 obj11
        obj11 = fn1(x1.clone())

        # 创建一个 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize 优化 fn1 函数，传入参数 nopython=True
        opt_fn1 = torch._dynamo.optimize(cnts, nopython=True)(fn1)
        # 使用优化后的函数 opt_fn1，再次传入 x1 的克隆，得到 obj12
        obj12 = opt_fn1(x1.clone())
        # 断言 obj11.x 等于 x1 + 2
        self.assertTrue(same(obj11.x, x1 + 2))
        # 断言 obj12.x 等于 x1 + 2
        self.assertTrue(same(obj12.x, x1 + 2))
        # 断言 obj11.x 等于 obj12.x
        self.assertTrue(same(obj11.x, obj12.x))
        # 断言 cnts.frame_count 等于 1
        self.assertEqual(cnts.frame_count, 1)

        @dataclasses.dataclass(frozen=True)
        class B:
            x: torch.Tensor

        # 定义函数 fn2，接受参数 x，返回一个 B 类的实例
        def fn2(x) -> None:
            # 创建 B 类的实例 b，初始化 x 属性为参数 x
            b = B(x)
            return b  # 返回实例 b

        # 生成一个大小为 10 的随机张量 x2
        x2 = torch.randn(10)
        # 调用 fn2 函数，传入 x2 的克隆，得到 obj21
        obj21 = fn2(x2.clone())

        # 创建一个新的 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize 优化 fn2 函数，传入参数 nopython=True
        opt_fn2 = torch._dynamo.optimize(cnts, nopython=True)(fn2)
        # 使用优化后的函数 opt_fn2，再次传入 x2 的克隆，得到 obj22
        obj22 = opt_fn2(x2.clone())
        # 断言 obj21.x 等于 x2
        self.assertTrue(same(obj21.x, x2))
        # 断言 obj22.x 等于 x2
        self.assertTrue(same(obj22.x, x2))
        # 断言 obj21.x 等于 obj22.x
        self.assertTrue(same(obj21.x, obj22.x))
        # 断言 cnts.frame_count 等于 0
        self.assertEqual(cnts.frame_count, 0)

        @dataclasses.dataclass(frozen=True)
        class C:
            x: torch.Tensor

        # 定义函数 fn3，接受参数 x，返回 None
        def fn3(x) -> None:
            # 创建 C 类的实例 c，初始化 x 属性为参数 x
            c = C(x)
            # 使用 object.__setattr__ 设置 c 的 x 属性为 x + 2
            object.__setattr__(c, "x", x + 2)
            return c  # 返回实例 c

        # 生成一个大小为 10 的随机张量 x3
        x3 = torch.randn(10)
        # 调用 fn3 函数，传入 x3 的克隆，得到 obj31
        obj31 = fn3(x3.clone())

        # 创建一个新的 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize 优化 fn3 函数，传入参数 nopython=True
        opt_fn3 = torch._dynamo.optimize(cnts, nopython=True)(fn3)
        # 使用优化后的函数 opt_fn3，再次传入 x3 的克隆，得到 obj32
        obj32 = opt_fn3(x3.clone())
        # 断言 obj31.x 等于 x3 + 2
        self.assertTrue(same(obj31.x, x3 + 2))
        # 断言 obj32.x 等于 x3 + 2
        self.assertTrue(same(obj32.x, x3 + 2))
        # 断言 obj31.x 等于 obj32.x
        self.assertTrue(same(obj31.x, obj32.x))
        # 断言 cnts.frame_count 等于 1
        self.assertEqual(cnts.frame_count, 1)

        @dataclasses.dataclass(frozen=True)
        class D:
            x: torch.Tensor

            # 定义 D 类的 __post_init__ 方法
            def __post_init__(self):
                # 使用 object.__setattr__ 设置 self 的 y 属性为 self.x + 2
                object.__setattr__(self, "y", self.x + 2)

        # 定义函数 fn4，接受参数 x，返回一个 D 类的实例
        def fn4(x) -> None:
            # 创建 D 类的实例 d，初始化 x 属性为参数 x
            d = D(x)
            return d  # 返回实例 d

        # 生成一个大小为 10 的随机张量 x4
        x4 = torch.randn(10)
        # 调用 fn4 函数，传入 x4 的克隆，得到 obj41
        obj41 = fn4(x4.clone())

        # 创建一个新的 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize 优化 fn4 函数，传入参数 nopython=True
        opt_fn4 = torch._dynamo.optimize(cnts, nopython=True)(fn4)
        # 使用优化后的函数 opt_fn4，再次传入 x4 的克隆，得到 obj42
        obj42 = opt_fn4(x4.clone())
        # 断言 obj41.x 等于 x4
        self.assertTrue(same(obj41.x, x4))
        # 断言 obj42.x 等于 x4
        self.assertTrue(same(obj42.x, x4))
        # 断言 obj41.x 等于 obj42.x
        self.assertTrue(same(obj41.x, obj42.x))
        # 断言 obj41.y 等于 x4 + 2
        self.assertTrue(same(obj41.y, x4 + 2))
        # 断言 obj42.y 等于 x4 + 2
        self.assertTrue(same(obj42.y, x4 + 2))
        # 断言 obj41.y 等于 obj42.y
        self.assertTrue(same(obj41.y, obj42.y))
        # 断言 cnts.frame_count 等于 1
        self.assertEqual(cnts.frame_count, 1)

    # 定义测试函数 test_user_defined_class_name
    def test_user_defined_class_name(self):
        # 定义一个空的 MyClassFoo 类
        class MyClassFoo:
            pass

        # 定义函数 fn1，接受参数 a, b, c
        def fn1(a, b, c):
            # 创建 MyClassFoo 类的实例 tmp
            tmp = MyClassFoo()
            # 如果 tmp 的类名为 "MyClassFoo"，则返回 a - b / c
            if tmp.__class__.__name__ == "MyClassFoo":
                return a - b / c

        # 调用 torch._dynamo.testing.standard_test 进行标准测试
        torch._dynamo.testing.standard_test(self, fn=fn1, nargs=3)
    def test_user_defined_class_python_type(self):
        # 定义一个简单的用户自定义类 MyClass1
        class MyClass1:
            pass

        # 定义一个空的元类 ExampleMeta
        class ExampleMeta(type):
            pass

        # 定义一个使用 ExampleMeta 元类的用户自定义类 MyClass2
        class MyClass2(metaclass=ExampleMeta):
            pass

        # 定义一个函数 fn，根据传入对象的类型返回不同的计算结果
        def fn(x, c):
            if isinstance(c, MyClass1):
                return x + 1
            elif isinstance(c, MyClass2):
                return x + 2
            else:
                return x + 3

        # 创建一个 torch 张量对象 x
        x = torch.rand(3)
        # 通过 torch._dynamo.optimize("eager") 优化 fn 函数
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 遍历 MyClass1 和 MyClass2 类型，检验优化后的结果是否与原始结果相同
        for c in [MyClass1, MyClass2]:
            ref = fn(x, c)
            res = opt_fn(x, c)
            # 使用 self.assertTrue(same(ref, res)) 断言优化后的结果与原始结果相同
            self.assertTrue(same(ref, res))

    def test_super_calling_with_metaclass(self):
        # 定义一个简单的元类 ExampleMeta
        class ExampleMeta(type):
            pass

        # 定义一个使用 ExampleMeta 元类的用户自定义类 MyClass1
        class MyClass1(metaclass=ExampleMeta):
            # 类属性 coeff 设为 4，用于测试中的常量保护
            coeff = 4  # Force the constant guard to test source in guards

            # 类方法 add，返回传入参数加一的结果
            @classmethod
            def add(cls, x):
                return x + 1

        # 定义一个继承自 MyClass1 的用户自定义类 MyClass2
        class MyClass2(MyClass1):
            # 类方法 add，调用父类的 add 方法，并加上父类的 coeff 属性值
            @classmethod
            def add(cls, x):
                torch._dynamo.graph_break()  # 在图中断开，用于测试
                return x + super().add(x) + super().coeff

        # 定义一个函数 fn，接受一个对象 obj，返回 x 加上 obj 的 add 方法计算结果
        def fn(x, obj):
            return x + obj.add(x)

        # 创建一个 torch 张量对象 x
        x = torch.rand(3)
        # 创建 MyClass2 的一个实例 obj
        obj = MyClass2()
        # 通过 torch._dynamo.optimize("eager") 优化 fn 函数
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 计算原始结果和优化后的结果
        ref = fn(x, obj)
        res = opt_fn(x, obj)
        # 使用 self.assertTrue(same(ref, res)) 断言优化后的结果与原始结果相同
        self.assertTrue(same(ref, res))

    def test_usr_cls_staticmethod(self):
        # 定义一个简单的用户自定义类 Foo
        class Foo:
            # 静态方法 bar，返回 a 和 b 的和
            @staticmethod
            def bar(a, b):
                return a + b

        # 定义一个函数 fn，返回 Foo 类的 bar 方法对 a 和 b 求和后减一的结果
        def fn(a, b):
            return Foo.bar(a, b) - 1

        # 使用 torch._dynamo.testing.standard_test 运行测试函数 fn，期望 nargs 为 2
        torch._dynamo.testing.standard_test(self, fn=fn, nargs=2)

    def test_usr_cls_classmethod(self):
        # 定义一个简单的用户自定义类 Foo
        class Foo:
            # 类方法 bar，返回 a 和 b 的和
            @classmethod
            def bar(cls, a, b):
                return a + b

        # 定义一个函数 fn，返回 Foo 类的 bar 方法对 a 和 b 求和后减一的结果
        def fn(a, b):
            return Foo.bar(a, b) - 1

        # 使用 torch._dynamo.testing.standard_test 运行测试函数 fn，期望 nargs 为 2
        torch._dynamo.testing.standard_test(self, fn=fn, nargs=2)

    def test_dunder_methods(self):
        # 定义一个简单的用户自定义类 Foo，具有加法、乘法、除法、减法运算符重载方法
        class Foo:
            # 初始化方法，接受一个值 val 并将其赋给实例属性 self.val
            def __init__(self, val):
                super().__init__()

                self.val = val

            # 加法运算符重载方法，返回两个 Foo 对象相加后的结果
            def __add__(self, other):
                return Foo(self.val + other.val)

            # 乘法运算符重载方法，返回两个 Foo 对象相乘后的结果
            def __mul__(self, other):
                return Foo(self.val * other.val)

            # 除法运算符重载方法，返回两个 Foo 对象相除后的结果
            def __truediv__(self, other):
                return Foo(self.val / other.val)

            # 减法运算符重载方法，返回两个 Foo 对象相减后的结果
            def __sub__(self, other):
                return Foo(self.val - other.val)

        # 定义一个函数 fn，接受三个参数 a、b、c，返回多次 Foo 对象之间的运算结果
        def fn(a, b, c):
            return Foo(a) + Foo(b) * Foo(c) / Foo(a) - Foo(b)

        # 使用 torch._dynamo.testing.standard_test 运行测试函数 fn，期望 nargs 为 3，操作数为 4
        torch._dynamo.testing.standard_test(self, fn=fn, nargs=3, expected_ops=4)
    def test_function_annotation(self):
        # 定义一个内部类 Variable，暂时未使用
        class Variable:
            pass

        # 定义函数 fn，参数 x 是一个张量
        def fn(x):
            # 对输入张量 x 进行除法操作
            x = x / 3.0

            # 定义内部函数 inner，参数 y 是一个 Variable 对象列表，返回 x + 1
            def inner(y: typing.List[Variable]):
                return x + 1

            return inner

        # 生成一个大小为 10 的随机张量 x1
        x1 = torch.randn(10)
        # 调用 fn 函数，并传入一个空列表作为参数，返回 inner 函数的调用结果 obj2
        obj2 = fn(x1)([])

        # 创建一个编译计数器对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数应用 optimize_assert 优化，并赋值给 opt_fn
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        # 对 opt_fn 函数应用 optimize_assert 优化，并传入 x1 作为参数，赋值给 opt_fn_inner
        opt_fn_inner = torch._dynamo.optimize_assert(cnts)(opt_fn(x1))
        # 调用 opt_fn_inner 函数，并传入一个空列表作为参数，赋值给 obj1
        obj1 = opt_fn_inner([])
        # 断言 obj1 与 obj2 相同
        self.assertTrue(same(obj1, obj2))
        # 断言帧计数为 2
        self.assertEqual(cnts.frame_count, 2)
        # 断言操作计数为 2
        self.assertEqual(cnts.op_count, 2)

    def test_nested_closure(self):
        # 生成一个大小为 10 的随机张量 v0
        v0 = torch.randn(10)

        # 定义函数 fn1
        def fn1():
            # 生成一个大小为 10 的随机张量 v1
            v1 = torch.randn(10)

            # 定义函数 fn2，接受任意位置参数和关键字参数
            def fn2(*args, **kwargs):
                # 断言 args 的长度为 1
                assert len(args) == 1
                # 断言 kwargs 的长度为 1
                assert len(kwargs) == 1
                # 生成一个大小为 10 的随机张量 v2，加上 args[0] 和 kwargs["b"]
                v2 = torch.randn(10) + args[0] + kwargs["b"]

                # 定义函数 fn3，接受 v3 作为参数，默认为一个大小为 10 的随机张量
                def fn3(v3=torch.randn(10)):
                    # 定义函数 fn4，返回 v0 + v1 + v2 + v3 + 1
                    def fn4():
                        return v0 + v1 + v2 + v3 + 1

                    return fn4

                return fn3

            # 调用 fn2 函数，并传入 1 和 b=2 作为参数，再调用其返回的结果
            return fn2(1, b=2)()

        # 创建一个编译计数器对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn1 函数应用 optimize_assert 优化，并赋值给 opt_fn1
        opt_fn1 = torch._dynamo.optimize_assert(cnts)(fn1)
        # 调用 opt_fn1 函数，并调用其返回的结果，赋值给 tmp1
        tmp1 = torch._dynamo.optimize_assert(cnts)(opt_fn1())
        # 再次调用 opt_fn1 函数，并调用其返回的结果，赋值给 tmp2
        tmp2 = torch._dynamo.optimize_assert(cnts)(opt_fn1())
        # 断言 tmp1 的形状为 (10,)
        self.assertTrue(tmp1().shape, (10,))
        # 断言 tmp1 与其自身相同
        self.assertTrue(same(tmp1(), tmp1()))
        # 断言 tmp1 与 tmp2 不相同
        self.assertFalse(same(tmp1(), tmp2()))
        # 断言帧计数为 2
        self.assertEqual(cnts.frame_count, 2)
        # 断言操作计数为 9
        self.assertEqual(cnts.op_count, 9)

    def test_nested_closure_mutation(self):
        # 定义函数 fn1
        def fn1():
            # 生成一个大小为 10 的随机张量 v1
            v1 = torch.randn(10)

            # 定义函数 fn2
            def fn2():
                # 生成一个大小为 10 的随机张量 v2
                v2 = torch.randn(10)

                # 定义函数 fn3
                def fn3():
                    # 声明 v1 和 v2 为非本地变量
                    nonlocal v1, v2
                    # 修改 v1 和 v2 的值
                    v1 += 1
                    v2 += 2
                    # 返回 v1 + v2
                    return v1 + v2

                return fn3

            # 调用 fn2 函数，赋值给 rv
            rv = fn2()
            # 连续两次调用 rv 函数
            rv()
            rv()
            return rv

        # 设置随机种子为 9000
        torch.manual_seed(9000)
        # 调用 fn1 函数，返回 counter1
        counter1 = fn1()
        # 创建一个结果列表 result1，包含三次调用 counter1 函数的结果
        result1 = [counter1(), counter1(), counter1()]

        # 再次设置随机种子为 9000
        torch.manual_seed(9000)
        # 创建一个编译计数器对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn1 函数应用 optimize_assert 优化，并调用其返回的结果，赋值给 opt_fn1
        opt_fn1 = torch._dynamo.optimize_assert(cnts)(fn1)
        # 调用 opt_fn1 函数，返回 counter2
        counter2 = torch._dynamo.optimize_assert(cnts)(opt_fn1())
        # 创建一个结果列表 result2，包含三次调用 counter2 函数的结果
        result2 = [counter2(), counter2(), counter2()]
        # 将第四次调用 counter1 函数的结果添加到 result1 列表中
        result1.append(counter1())
        # 将第四次调用 counter2 函数的结果添加到 result2 列表中
        result2.append(counter2())

        # 断言 result1 与 result2 相同
        self.assertTrue(same(result1, result2))
        # 断言帧计数为 2
        self.assertEqual(cnts.frame_count, 2)
        # 断言操作计数为 11
        self.assertEqual(cnts.op_count, 11)
    # 定义一个测试函数，测试闭包在内联时的行为
    def test_write_to_closures_in_inlining(self):
        # 初始化一个空列表用于存储输出结果
        out = []
        # 遍历布尔值列表，测试两种情况
        for use_dynamo in [False, True]:

            # 定义一个生成计数器闭包的函数
            def make_counter():
                # 生成一个包含10个随机数的张量
                x = torch.randn(10)

                # 定义计数器闭包
                def counter():
                    nonlocal x
                    # 每次调用计数器闭包时增加 x 的值并返回
                    x = x + 1
                    return x

                return counter

            # 设置随机数种子为0，生成一个计数器闭包
            torch.manual_seed(0)
            counter = make_counter()
            # 根据 use_dynamo 决定使用不同的方法来调用计数器闭包并将结果添加到 out 列表中
            if not use_dynamo:
                out.append(counter() + counter())
            else:
                # 创建一个编译计数器对象
                cnts = torch._dynamo.testing.CompileCounter()

                # 使用动态编译优化器装饰函数 fn，并指定使用 no-python 模式
                @torch._dynamo.optimize(cnts, nopython=True)
                def fn(counter):
                    return counter() + counter()

                out.append(fn(counter))
                # 断言编译器帧数为1
                self.assertEqual(cnts.frame_count, 1)
                # 断言编译操作数为3
                self.assertEqual(cnts.op_count, 3)
                # 断言计算结果与未优化前的结果不相同
                self.assertFalse(same(counter() + counter(), out[-1]))

        # 断言两种方法的输出结果相同
        self.assertTrue(same(out[0], out[1]))

    # 定义测试闭包在超出作用域的情况下的行为
    def test_closure_out_of_scope_cell(self):
        # 创建一个随机数作为闭包外部变量
        cell1 = torch.rand(1).item()
        # 创建一个随机矩阵作为闭包外部变量
        cell2 = torch.rand(3, 3)

        # 定义间接调用闭包的函数
        def indirect():
            return direct()

        # 定义直接返回内部函数调用结果的函数
        def direct():
            # 定义内部函数，返回闭包外部变量的加一和加三结果
            def inner():
                return cell1 + 1, cell2 + 3

            return inner()

        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用动态编译优化器装饰函数 indirect
        opt_fn = torch._dynamo.optimize(cnts)(indirect)
        # 调用优化后的函数并断言结果与预期接近
        result1, result2 = opt_fn()
        self.assertAlmostEqual(cell1 + 1, result1)
        self.assertTrue(torch.allclose(cell2 + 3, result2))
        # 断言编译器帧数为1
        self.assertEqual(cnts.frame_count, 1)
        # 断言编译操作数为1
        self.assertEqual(cnts.op_count, 1)

    # 定义测试闭包在超出作用域且包含变量突变时的行为
    def test_closure_out_of_scope_cell_with_mutation(self):
        # 创建一个随机数作为闭包外部变量，并保存原始值
        cell1 = torch.rand(1).item()
        orig1 = cell1
        # 创建一个随机矩阵作为闭包外部变量，并保存原始值
        cell2 = torch.rand(3, 3)
        orig2 = cell2.clone()

        # 定义间接调用闭包的函数
        def indirect():
            return direct()

        # 定义直接返回内部函数调用结果的函数
        def direct():
            # 定义内部函数，使用 nonlocal 关键字更新 cell1 和 cell2 的值，并返回变化后的值
            def inner():
                nonlocal cell1, cell2
                x = cell2 + 1
                cell1 += 1
                cell2 += 10
                x = x + cell2
                return cell1, cell2, x

            return inner()

        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用动态编译优化器装饰函数 indirect，并指定使用 no-python 模式
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(indirect)
        # 循环调用优化后的函数，并断言结果与预期接近
        for i in range(1, 4):
            result1, result2, _ = opt_fn()
            self.assertAlmostEqual(orig1 + 1 * i, result1)
            self.assertTrue(torch.allclose(orig2 + 10 * i, result2))
            # 断言编译器帧数为1
            self.assertEqual(cnts.frame_count, 1)
            # 断言编译操作数为3
            self.assertEqual(cnts.op_count, 3)
            # 清空计数器对象
            cnts.clear()
    def test_closure_with_mutation_and_graph_break(self):
        # 测试具有变异和图断点的闭包函数
        def fn():
            # 创建一个包含单个零张量的 Torch 张量
            x = torch.zeros(1)

            # 定义一个内部函数，用于修改外部函数的局部变量 x
            def subfunc():
                x[0] = backup

            # 如果 x[0] 大于等于 -1e5，则执行 pass
            if x[0] >= -1e5:
                pass

            # 设置备份值为 1
            backup = 1
            # 调用内部函数 subfunc()，修改 x 的值
            subfunc()
            # 返回修改后的张量 x
            return x

        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化编译
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 获取原始函数 fn 的预期输出
        expected = fn()
        # 获取优化后函数 opt_fn 的实际输出
        actual = opt_fn()
        # 断言优化前后输出是否相同
        self.assertTrue(same(expected, actual))
        # 断言编译计数器的帧数是否为 2
        self.assertEqual(cnts.frame_count, 2)

    def test_closure_out_of_scope_cell_with_cond(self):
        # 测试闭包函数中使用超出作用域的闭包变量，并在条件语句中使用
        # 引入 cond 函数来自 functorch.experimental.control_flow 模块
        from functorch.experimental.control_flow import cond

        # 定义一个简单的函数 g(x)，返回 x
        def g(x):
            return x

        # 定义一个继承自 torch.nn.Module 的类 ModuleCondDeep
        class ModuleCondDeep(torch.nn.Module):
            def forward(self, pred, x):
                return self._indirection(pred, x)

            def _indirection(self, pred, x):
                return self.indirection(pred, x)

            def indirection(self, pred, x):
                # 定义一个深层函数 deep(x)
                def deep(x):
                    # y = g(x)
                    y = x
                    # 在条件语句中根据 x[0][0] 的值选择 true_fn 或 false_fn
                    return cond(
                        x[0][0] > 0,
                        true_fn,
                        false_fn,
                        [y],
                    )

                # 定义一个浅层函数 shallow(x)，返回 x * 2
                def shallow(x):
                    return x * 2

                # 返回根据 pred 条件选择执行 shallow 还是 deep 函数的结果
                return cond(pred, shallow, deep, [x])

        # 创建 ModuleCondDeep 类的实例 mod
        mod = ModuleCondDeep()
        # 对 mod 进行 eager 模式的优化编译
        opt_mod = torch._dynamo.optimize("eager")(mod)
        # 创建输入张量 inp，形状为 (3, 3)
        inp = torch.randn(3, 3)
        # 计算未优化模型的预期输出 exp1
        exp1 = mod(torch.tensor(False), inp)
        # 计算优化模型的实际输出 actual1
        actual1 = opt_mod(torch.tensor(False), inp)
        # 计算另一条件下未优化模型的预期输出 exp2
        exp2 = mod(torch.tensor(True), inp)
        # 计算另一条件下优化模型的实际输出 actual2
        actual2 = opt_mod(torch.tensor(True), inp)
        # 断言两次调用的输出是否近似相等
        self.assertTrue(torch.allclose(exp1, actual1))
        self.assertTrue(torch.allclose(exp2, actual2))

    def test_top_package_import(self):
        # 测试导入顶级包并使用其中的模块
        def fn(x):
            # 导入 torch.fx 模块
            import torch.fx

            # 断言 x 不是 torch.fx.Proxy 类型的实例
            assert not isinstance(x, torch.fx.Proxy)
            # 返回 x 的正弦值
            return torch.sin(x)

        # 创建输入张量 x，形状为 (4, 5)
        x = torch.randn(4, 5)
        # 计算未优化模型的预期输出 ref
        ref = fn(x)
        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化编译，并加入断言
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        # 计算优化模型的实际输出 res
        res = opt_fn(x)
        # 断言优化前后输出是否相同
        self.assertTrue(same(ref, res))

    def test_typing_typevar(self):
        # 测试使用类型变量进行类型注解
        def fn(x):
            # 定义一个接受 torch.Tensor 类型参数并返回 torch.Tensor 类型结果的函数 sumt
            def sumt(y: torch.Tensor) -> torch.Tensor:
                return torch.sum(y)

            # 定义一个接受 Callable[[T], T] 类型参数和类型为 T 的 y 参数，并返回类型为 T 的结果的函数 foo
            def foo(c: typing.Callable[[T], T], y: T) -> T:
                return c(y)

            # 调用 foo 函数，传入 sumt 函数和输入参数 x
            return foo(sumt, x)

        # 创建输入张量 x，形状为 (3,)
        x = torch.randn(3)
        # 计算未优化模型的预期输出 ref
        ref = fn(x)
        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化编译，并加入断言
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        # 计算优化模型的实际输出 res
        res = opt_fn(x)
        # 断言优化前后输出是否相同
        self.assertTrue(same(ref, res))
        # 断言编译计数器的帧数是否为 1
        self.assertEqual(cnts.frame_count, 1)
    # 定义一个测试函数，用于测试 typing 中的 Union 和 Optional 类型注解
    def test_typing_union_and_optional(self):
        # 定义内部函数 fn，接受一个参数 x
        def fn(x):
            # 使用 torch.jit.annotate 注解一个字典，键为字符串，值为可选的 torch.Tensor
            a = torch.jit.annotate(typing.Dict[str, typing.Optional[torch.Tensor]], {})
            # 使用 torch.jit.annotate 注解一个字典，键为字符串，值为 torch.Tensor 或 None
            b = torch.jit.annotate(
                typing.Dict[str, typing.Union[torch.Tensor, None]], {}
            )
            # 返回注解后的字典 a 和 b，以及参数 x 加 1 的结果
            return a, b, x + 1

        # 创建一个包含 3 个随机数的 torch.Tensor
        x = torch.randn(3)
        # 调用 fn 函数，返回结果保存在 ref 中
        ref = fn(x)
        # 对 fn 函数应用 torch._dynamo.optimize 优化器，生成新的函数 opt_fn
        opt_fn = torch._dynamo.optimize("eager", nopython=False)(fn)
        # 使用参数 x 调用优化后的函数 opt_fn，保存结果在 res 中
        res = opt_fn(x)
        # 使用 assertTrue 断言 ref 和 res 相同
        self.assertTrue(same(ref, res))

    # 定义测试函数，用于测试在模块上应用优化
    def test_optimize_on_module(self):
        # 定义 MockModule 类，继承自 torch.nn.Module
        class MockModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 添加一个 ReLU 层
                self.relu = torch.nn.ReLU()

            # 自定义成员方法，用于检查 Dynamo 返回的 mod 对象是否能重定向到这个方法
            def custom_member(self):
                pass

            # 前向传播方法
            def forward(self, x):
                return self.relu(x)

        # 创建一个 CompileCounter 对象 cnts1
        cnts1 = torch._dynamo.testing.CompileCounter()
        # 实例化 MockModule 类，创建一个模块对象 mod
        mod = MockModule()
        # 对 mod 应用 torch._dynamo.optimize 优化器，生成优化后的模块对象 optimized_mod
        optimized_mod = torch._dynamo.optimize(cnts1, nopython=True)(mod)

        # 创建一个包含 10 个随机数的 torch.Tensor
        a = torch.randn(10)
        # 使用原始模块 mod 对象进行前向传播，结果保存在 ref 中
        ref = mod(a)
        # 使用优化后的模块 optimized_mod 对象进行前向传播，结果保存在 res 中
        res = optimized_mod(a)

        # 调用优化后的模块对象的 custom_member 方法
        optimized_mod.custom_member()

        # 使用 assertTrue 断言 ref 和 res 相同
        self.assertTrue(same(ref, res))

    # 定义测试函数，测试嵌套应用 optimize 装饰器
    def test_nested_optimize_decorator(self):
        # 创建两个 CompileCounter 对象 cnts2 和 cnts3
        cnts2 = torch._dynamo.testing.CompileCounter()
        cnts3 = torch._dynamo.testing.CompileCounter()

        # 定义使用 torch._dynamo.run 装饰器的函数 fn1，接受参数 x，返回 sin(x) * 10
        @torch._dynamo.run()
        def fn1(x):
            return torch.sin(x) * 10

        # 定义使用 cnts2 和 nopython=True 参数的优化函数 fn2，接受参数 x，返回 fn1(x) + 1
        @torch._dynamo.optimize(cnts2, nopython=True)
        def fn2(x):
            return fn1(x) + 1

        # 定义使用 cnts3 和 nopython=True 参数的优化函数 fn3，接受参数 x，返回 torch.relu(fn2(x))
        @torch._dynamo.optimize(cnts3, nopython=True)
        def fn3(x):
            return torch.relu(fn2(x))

        # 使用一个 4x5 大小的随机数张量调用 fn3 函数
        fn3(torch.randn(4, 5))
        # 使用 assertEqual 断言 cnts2 的 frame_count 为 0
        self.assertEqual(cnts2.frame_count, 0)
        # 使用 assertEqual 断言 cnts3 的 frame_count 为 1
        self.assertEqual(cnts3.frame_count, 1)
        # 使用 assertEqual 断言 cnts3 的 op_count 为 4
        self.assertEqual(cnts3.op_count, 4)

    # 定义测试函数，测试在装饰的函数上运行 run 方法
    def test_nested_optimize_run(self):
        # 创建一个 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()

        # 定义使用 cnts 和 nopython=True 参数的优化函数 fn，接受参数 x，返回 torch.relu(torch.cos(x) + torch.sin(x))
        @torch._dynamo.optimize(cnts, nopython=True)
        def fn(x):
            return torch.relu(torch.cos(x) + torch.sin(x))

        # 使用一个大小为 4 的随机数张量调用 fn 函数
        fn(torch.randn(4))
        # 使用 assertEqual 断言 cnts 的 frame_count 为 1
        self.assertEqual(cnts.frame_count, 1)

        # 再次使用一个 4x4 大小的随机数张量调用 fn 函数
        fn(torch.randn(4, 4))
        # 使用 assertEqual 断言 cnts 的 frame_count 为 2
        self.assertEqual(cnts.frame_count, 2)

        # 测试在装饰的函数上运行 run 方法
        fn = torch._dynamo.run(fn)
        fn(torch.randn(4, 4, 4))
        # 使用 assertEqual 断言 cnts 的 frame_count 仍然为 2
        self.assertEqual(cnts.frame_count, 2)
    # 定义一个测试函数，用于测试嵌套优化的行为
    def test_nested_optimize(self):
        # 创建两个编译计数器对象
        cnts1 = torch._dynamo.testing.CompileCounter()
        cnts2 = torch._dynamo.testing.CompileCounter()

        # 定义一个简单的函数 fn，对输入进行一系列操作
        def fn(x):
            return torch.relu(torch.cos(x) + torch.sin(x))

        # 对 fn 进行第一层优化，并将结果赋给 fn1
        fn1 = torch._dynamo.optimize(cnts1, nopython=True)(fn)
        # 对 fn1 进行第二层优化，并将结果赋给 fn2
        fn2 = torch._dynamo.optimize(cnts2, nopython=True)(fn1)

        # 调用 fn2 以触发优化过程
        fn2(torch.randn(4))
        # 断言第二个编译计数器的帧数为 1
        self.assertEqual(cnts2.frame_count, 1)
        # 断言第一个编译计数器的帧数为 0，因为第一层优化应该被忽略
        self.assertEqual(cnts1.frame_count, 0)

        # 由于 fn 的代码对象已经编译完成，调用 fn1 应该直接调用编译后的可调用函数
        torch._dynamo.run()(fn1)(torch.randn(4))
        # 再次断言第一个编译计数器的帧数为 0
        self.assertEqual(cnts1.frame_count, 0)

        # 测试通过颠倒调用顺序来验证相同的行为
        torch._dynamo.reset()
        # 重新创建编译计数器对象
        cnts1 = torch._dynamo.testing.CompileCounter()
        cnts2 = torch._dynamo.testing.CompileCounter()
        # 对 fn 进行第一层优化，并将结果赋给 fn1
        fn1 = torch._dynamo.optimize(cnts1, nopython=True)(fn)
        # 对 fn1 进行第二层优化，并将结果赋给 fn2
        fn2 = torch._dynamo.optimize(cnts2, nopython=True)(fn1)
        # 调用 fn1 以触发第一层优化
        fn1(torch.randn(4))
        # 断言第一个编译计数器的帧数为 1
        self.assertEqual(cnts1.frame_count, 1)
        # 使用 torch._dynamo.run() 调用 fn2 以触发第二层优化
        torch._dynamo.run()(fn2)(torch.randn(4))
        # 再次断言第二个编译计数器的帧数为 0
        self.assertEqual(cnts2.frame_count, 0)

    # 测试 torch.Size 类的用法
    def test_torch_size(self):
        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()

        # 定义一个函数 fn，操作一个张量 x
        def fn(x):
            # 创建一个固定大小的输出张量
            output_size = torch.Size([10, 10])
            # 将输入张量 x 重新视图化为指定的 output_size
            x = x.view(*output_size)
            return (x,)

        # 创建一个需要梯度的随机张量 x
        x = torch.randn(100, requires_grad=True)
        # 克隆 x
        x_clone = x.clone()
        # 调用 fn 获取参考结果
        ref = fn(x)

        # 对 fn 进行优化，并将结果赋给 opt_fn
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 使用优化后的函数 opt_fn 处理 x_clone
        res = opt_fn(x_clone)

        # 断言优化前后的结果应该一致
        self.assertTrue(same(ref, res))

    # 测试 torch.Size 类的 numel 方法
    def test_torch_size_numel(self):
        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()

        # 定义一个函数 fn，返回 torch.Size([10, 8]) 的元素数量
        def fn():
            return torch.Size([10, 8]).numel()

        # 对 fn 进行优化，并将结果赋给 opt_fn
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 调用 opt_fn 获取计算结果
        num = torch.Size([10, 8]).numel()
        # 断言优化后的函数调用结果与预期的 num 相等
        self.assertEqual(opt_fn(), num)

    # 测试 torch.Size 类的动态 numel 方法
    def test_torch_size_numel_dynamic(self):
        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()

        # 定义一个函数 fn，返回输入张量 x 的尺寸的元素数量
        def fn(x):
            return x.size().numel()

        # 对 fn 进行优化，并将结果赋给 opt_fn
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 创建一个张量 x
        x = torch.rand(10, 1, 8, 1)
        # 计算预期的结果
        expect = fn(x)
        # 断言优化后的函数调用结果与预期的 expect 相等
        self.assertEqual(opt_fn(x), expect)

    # 测试张量的形状类型检查
    def test_shape_type(self):
        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()

        # 定义一个函数 fn，检查张量 x 的形状类型是否为 torch.Size
        def fn(x):
            return x + (type(x.shape) == torch.Size)

        # 对 fn 进行优化，并将结果赋给 opt_fn
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 创建一个空张量 x
        x = torch.zeros(())
        # 断言优化后的函数调用结果与未优化时的结果相同
        self.assertEqual(opt_fn(x), fn(x))

    # 测试张量的尺寸查询
    def test_size_dim(self):
        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()

        # 定义一个函数 fn，返回张量 x 在指定维度 dim 上的尺寸
        def fn(x, dim):
            return x.size(dim=dim)

        # 对 fn 进行优化，并将结果赋给 opt_fn
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 创建一个空张量 x，维度为 [4, 9, 8]
        x = torch.empty([4, 9, 8])
        # 断言优化后的函数调用结果与预期的尺寸相等
        self.assertEqual(opt_fn(x, 1), 9)
        self.assertEqual(opt_fn(x, -2), 9)
    # 定义测试函数，用于测试 torch.Tensor 的 stride_dim 方法的优化情况
    def test_stride_dim(self):
        # 创建一个编译计数器
        cnts = torch._dynamo.testing.CompileCounter()

        # 定义一个函数 fn，返回张量 x 指定维度 dim 的步幅
        def fn(x, dim):
            return x.stride(dim=dim)

        # 对 fn 应用优化器，使用 torch._dynamo.optimize 进行优化
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)

        # 创建一个形状为 [4, 9, 8] 的张量 x
        x = torch.empty([4, 9, 8])

        # 断言优化后的函数 opt_fn 对于不同维度的调用返回正确的步幅值
        self.assertEqual(opt_fn(x, 0), 72)
        self.assertEqual(opt_fn(x, -2), 8)

    # 测试 torch.manual_seed 的行为
    def test_torch_seed(self):
        # 从 torch._dynamo.utils 中导入计数器
        from torch._dynamo.utils import counters

        # 创建编译计数器
        cnts = torch._dynamo.testing.CompileCounter()

        # 清空计数器中的内容
        counters.clear()

        # 定义函数 fn，使用当前时间的系统最大整数作为随机种子，设置 torch 的手动随机种子
        def fn(x):
            attention_seed = int(torch.seed() % sys.maxsize)
            torch.manual_seed(attention_seed)
            return (x,)

        # 创建一个形状为 [10] 的正态分布张量 x
        x = torch.randn(10, requires_grad=True)

        # 计算参考值 ref，即调用 fn 函数返回值
        ref = fn(x)

        # 由于 torch.manual_seed 中的图断开，需要在此处使用 Python 代码
        opt_fn = torch._dynamo.optimize(cnts, nopython=False)(fn)
        res = opt_fn(x)

        # 断言优化后的结果与参考值相同
        self.assertTrue(same(ref, res))

        # 检查编译计数器的操作数和帧数是否正确
        self.assertEqual(cnts.op_count, 1)
        self.assertEqual(cnts.frame_count, 1)

        # 检查图在 manual_seed 处断开的计数器
        self.assertEqual(len(counters["graph_break"]), 1)

    # 测试 torch.overrides.is_tensor_like 函数的行为
    def test_is_tensor_like(self):
        # 创建编译计数器
        cnts = torch._dynamo.testing.CompileCounter()

        # 定义函数 f，判断输入 x 是否类似张量，如果是则返回 x 的两倍，否则返回一个全为 1 的张量与 x 的和
        def f(x):
            if torch.overrides.is_tensor_like(x):
                return (x * 2,)
            return (torch.ones(10) + x,)

        # 创建一个形状为 [10] 的正态分布张量 x
        x = torch.randn(10)

        # 计算参考值 ref0 和 ref1，分别对应于输入 x 和标量 4 的调用结果
        ref0 = f(x)
        ref1 = f(4)

        # 对函数 f 应用优化器 opt_f
        opt_f = torch._dynamo.optimize(cnts, nopython=True)(f)

        # 计算优化后的结果 res0 和 res1，分别对应于输入 x 和标量 4 的调用结果
        res0 = opt_f(x)
        res1 = opt_f(4)

        # 断言优化后的结果与参考值相同
        self.assertTrue(same(ref0, res0))
        self.assertTrue(same(ref1, res1))

    # 测试自定义类 MyTensor 的 __torch_function__ 方法
    def test_is_tensor_like2(self):
        # 定义类 MyTensor，模拟具有 __torch_function__ 方法的张量类
        class MyTensor:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}

                # 如果调用 torch.max 函数，则返回张量值为 123 的张量
                if func is torch.max:
                    return torch.tensor(123)
                return func(*args, **kwargs)

        # 定义函数 fn，如果输入 x 类似张量则返回 x 的最大值，否则返回一个全为 0 的张量
        def fn(x):
            if torch.overrides.is_tensor_like(x):
                return torch.max(x)
            else:
                return torch.zeros(1)

        # 创建一个 MyTensor 的实例 x
        x = MyTensor()

        # 计算参考值 ref0 和 ref1，分别对应于输入 x 和标量 4 的调用结果
        ref0 = fn(x)
        ref1 = fn(4)

        # 对函数 fn 应用优化器 opt_fn
        opt_fn = torch._dynamo.optimize("eager")(fn)

        # 计算优化后的结果 res0 和 res1，分别对应于输入 x 和标量 4 的调用结果
        res0 = opt_fn(x)
        res1 = opt_fn(4)

        # 断言优化后的结果与参考值相同
        self.assertTrue(same(ref0, res0))
        self.assertTrue(same(ref1, res1))

    # 测试函数 fn，从张量 x 中提取索引为 y.data 的元素
    def test_tensor_data(self):
        # 定义函数 fn，从张量 x 中提取索引为 y.data 的元素
        def fn(x, y):
            return x[y.data]

        # 创建形状为 [8] 的随机张量 x 和形状为 [8] 的全为 1 的整数张量 y
        x = torch.rand(8)
        y = torch.ones(8).to(torch.int)

        # 计算参考值 ref，即调用 fn 函数返回值
        ref = fn(x, y)

        # 对函数 fn 应用优化器 opt_fn
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)

        # 计算优化后的结果 res，即调用 opt_fn 函数返回值
        res = opt_fn(x, y)

        # 断言优化后的结果与参考值相同
        self.assertTrue(same(ref, res))
    def test_tensor_layout(self):
        # 定义一个函数 fn，接受一个张量 x，并返回一个形状与 x 相同的全零张量，使用相同的数据类型、布局和设备
        def fn(x):
            return torch.zeros(
                [x.size()[0], x.size()[1]],
                dtype=x.dtype,
                layout=x.layout,
                device=x.device,
            )

        # 创建一个形状为 (2, 3) 的随机张量 x
        x = torch.rand(2, 3)
        # 使用 fn 函数生成参考结果 ref
        ref = fn(x)
        # 对 fn 函数进行优化，使其支持即时编译，并执行优化后的函数得到结果 res
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = opt_fn(x)
        # 断言优化前后的结果应该相同
        self.assertTrue(same(ref, res))

    def test_version_ci(self):
        # 临时测试确保 torch 的 ci 版本设置正确，检查是否有 _subclasses 属性
        self.assertTrue(hasattr(torch, "_subclasses"))

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    def test_rand(self):
        cnts = torch._dynamo.testing.CompileCounter()
        device = "cuda"

        # 定义一个函数 fn，返回在给定设备上的随机张量
        def fn():
            return torch.randn(10, device=device)

        # 设置随机种子
        torch.manual_seed(10)
        # 生成参考运行结果 ref_run1
        ref_run1 = fn()

        torch.manual_seed(10)
        # 重新生成相同随机种子下的参考运行结果 ref_run2
        ref_run2 = fn()
        # 断言两次生成的结果应该相同
        self.assertTrue(same(ref_run1, ref_run2))

        torch.manual_seed(10)
        # 对 fn 函数进行优化，使其支持即时编译，并执行优化后的函数得到结果 res
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn()
        # 断言优化后的结果与参考结果 ref_run1 相同
        self.assertTrue(same(res, ref_run1))

    def test_slice_input(self):
        cnts = torch._dynamo.testing.CompileCounter()

        # 定义一个函数 getitem，接受一个数组 a 和索引 idx，根据 idx 返回对应的结果
        def getitem(a, idx):
            if isinstance(idx, slice):
                return (
                    torch.zeros(1),
                    a[idx]
                    + [
                        100,
                    ],
                )
            else:
                return (torch.zeros(1), a[idx])

        layers = list(range(10))
        # 生成 getitem 函数不同调用方式的参考结果 ref0, ref1, ref2
        ref0 = getitem(layers, slice(0, 2, 1))
        ref1 = getitem(layers, 2)
        ref2 = getitem(layers, slice(3, 8, 2))
        # 对 getitem 函数进行优化，使其支持即时编译，并执行优化后的函数得到结果 res0, res1, res2
        opt_getitem = torch._dynamo.optimize(cnts, nopython=True)(getitem)
        res0 = opt_getitem(layers, slice(0, 2, 1))
        res1 = opt_getitem(layers, 2)
        res2 = opt_getitem(layers, slice(3, 8, 2))

        # 断言优化前后的结果应该相同
        self.assertTrue(ref0 == res0)
        self.assertTrue(ref1 == res1)
        self.assertTrue(ref2 == res2)

    def test_grad(self):
        cnts = torch._dynamo.testing.CompileCounter()

        # 定义一个函数 fn，接受两个张量 a 和 b，执行一系列操作后返回一个张量
        def fn(a, b):
            out = a * b
            out.sum().backward()
            real_out = torch.sigmoid(a.grad + b)
            return real_out

        # 生成两个具有梯度的随机张量列表 inps
        inps = [torch.randn(4, requires_grad=True) for _ in range(2)]
        for inp in inps:
            inp.grad = None
        # 生成参考结果 ref
        ref = fn(*inps)

        for inp in inps:
            inp.grad = None
        # 对 fn 函数进行优化，使其支持即时编译，并执行优化后的函数得到结果 res
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(*inps)

        # 断言优化前后的结果应该相同
        self.assertTrue(same(ref, res))

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_source_non_input_grad_access(self):
        # This test function verifies access to gradients of model parameters and intermediary tensors.
        # It involves two scenarios: one in eager mode and another in compiled mode using Dynamo.
        cnts = torch._dynamo.testing.CompileCounter()

        class TrivialModel(torch.nn.Module):
            def __init__(self):
                super(TrivialModel, self).__init__()
                self.linear = torch.nn.Linear(2, 1)

            def forward(self, x):
                return self.linear(x)

        def fn(a, b):
            outs = []
            # Collect gradients of each parameter in the model
            for param in model.parameters():
                outs.append(torch.ones(param.grad.size()))
            # Access and modify the gradient of the last parameter
            return outs, param.grad + 1

        model = TrivialModel()
        # Eager mode setup
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        out = model(a)
        out_sum = out.sum()
        out_sum.backward()  # Compute gradients
        ref = fn(a, b)  # Reference result

        # Compiled mode setup
        model = TrivialModel()
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        out = model(a)
        out_sum = out.sum()
        out_sum.backward()  # Compute gradients

        # Optimize function using Dynamo's optimization
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(a, b)  # Result of optimized function

        # Assertions to verify correctness
        self.assertTrue(same(ref, res))  # Compare reference and result
        self.assertEqual(cnts.frame_count, 1)  # Ensure one frame was counted
        self.assertEqual(cnts.op_count, 3)  # Ensure three operations were counted

    def test_intermediary_tensor_grad_access(self):
        # This test function evaluates gradient access from model parameters
        # and an intermediary tensor in both eager and compiled modes.
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(a, b):
            intermediary = torch.ones(2, 2)
            c = a + intermediary
            outs = []
            outs.append(intermediary.grad)  # Access gradient of intermediary tensor
            return outs

        # Eager mode setup
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        ref = fn(a, b)  # Reference result

        # Compiled mode setup
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(a, b)  # Result of optimized function

        # Assertions to verify correctness
        self.assertTrue(same(ref, res))  # Compare reference and result
        self.assertEqual(cnts.frame_count, 1)  # Ensure one frame was counted
        self.assertEqual(cnts.op_count, 2)  # Ensure two operations were counted

    def test_clone_sparse_input(self):
        # This test function clones sparse input tensors across different layouts,
        # ensuring the cloned tensors are identical to the originals.
        for layout in [
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        ]:
            for sparse_input in self.generate_simple_inputs(
                layout,
                device="cpu",
                dtype=torch.float64,
                index_dtype=torch.int64,
            ):
                # Clone sparse input using Dynamo's utility method
                sparse_copy = torch._dynamo.utils.clone_input(sparse_input)
                # Ensure the cloned sparse tensor matches the original
                self.assertEqual(sparse_input, sparse_copy)
    def test_tensor_is_contiguous(self):
        def fn(x):
            # 创建一个大小为 (1, 16, 1, 1) 的随机张量作为输入
            input = torch.randn((1, 16, 1, 1))
            # 创建一个大小为 (8, 16, 3, 3) 的随机张量作为卷积核权重
            weight = torch.randn((8, 16, 3, 3))
            # 将权重张量按照指定的内存格式转换
            weight = weight.to(memory_format=x)
            # 进行二维卷积操作
            output = torch.conv2d(input, weight, None, (2, 1), (1, 1), (1, 1), 1)
            # 检查输出张量是否按照指定的内存格式是连续的
            return output.is_contiguous(memory_format=x)

        # 对函数 fn 进行优化处理
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 遍历两种内存格式进行测试，并断言优化后的结果与原函数结果一致
        for x in [torch.contiguous_format, torch.channels_last]:
            self.assertEqual(fn(x), opt_fn(x))

    def test_python_slice(self):
        def f1(input):
            # 初始化累加器为 0
            y = 0
            # 遍历 input 的切片从索引 2 开始到末尾
            for i, x in enumerate(input[2:], 1):
                # 累加每个切片元素的值到 y
                y = y + x
            return y

        def f2(input):
            # 初始化累加器为 0
            y = 0
            # 遍历 input 的形状的第三个维度（从索引 2 开始到末尾）
            for i, x in enumerate(input.shape[2:], 1):
                # 累加每个维度值到 y
                y = y + x
            return y

        # 创建编译计数器实例
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 f1 和 f2 函数分别进行优化处理
        opt_f1 = torch._dynamo.optimize(cnts)(f1)
        opt_f2 = torch._dynamo.optimize(cnts)(f2)
        # 对优化后的函数进行调用并获取结果
        res1 = opt_f1([1, 2, 3, 5])
        res2 = opt_f2(torch.rand([2, 3, 4, 5]))

        # 断言优化后的结果与预期结果相等
        self.assertEqual(res1, 8)
        self.assertEqual(res2, 9)

    def test_enum_as_dict_key(self):
        class MyEnum(enum.Enum):
            FOO = 10
            BAR = 20

        def fn(x):
            # 计算 x + 2
            y = x + 2
            # 创建包含枚举和其他类型键的字典
            z = {
                MyEnum.FOO: torch.tensor(1),
                MyEnum.BAR: 10,
                "MyEnum.BAR": torch.tensor(8),
                5: torch.rand(3),
            }
            # 在计算图中断开当前图节点
            torch._dynamo.graph_break()
            # 计算字典中 MyEnum.FOO 和 "MyEnum.BAR" 对应的值的和
            a = z[MyEnum.FOO] + z["MyEnum.BAR"]
            # 计算 y 的两倍
            b = y * 2
            return a, b

        # 创建编译计数器实例
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化处理
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 多次使用随机张量进行测试，验证优化后的结果与原函数结果一致
        for _ in range(10):
            x = torch.rand(3)
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
        # 断言编译帧数为 2
        self.assertEqual(cnts.frame_count, 2)

    def test_enum_as_dict_key_with_overloaded_str(self):
        class MyEnum(enum.Enum):
            FOO = 10
            BAR = 20

            # 重载 __str__ 方法返回枚举值本身
            def __str__(self):
                return self.value

        def fn(x):
            # 计算 x + 2
            y = x + 2
            # 创建包含枚举和其他类型键的字典
            z = {
                MyEnum.FOO: torch.tensor(1),
                MyEnum.BAR: 10,
                "MyEnum.BAR": torch.tensor(8),
                5: torch.rand(3),
            }
            # 在计算图中断开当前图节点
            torch._dynamo.graph_break()
            # 计算字典中 MyEnum.FOO 和 "MyEnum.BAR" 对应的值的和
            a = z[MyEnum.FOO] + z["MyEnum.BAR"]
            # 计算 y 的两倍
            b = y * 2
            return a, b

        # 创建编译计数器实例
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化处理
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 多次使用随机张量进行测试，验证优化后的结果与原函数结果一致
        for _ in range(10):
            x = torch.rand(3)
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
        # 断言编译帧数为 2
        self.assertEqual(cnts.frame_count, 2)
    def test_const_dict_variable_python_type(self):
        from torch._dynamo.variables import ConstantVariable, ConstDictVariable  # 导入必要的模块和类

        make_key = ConstantVariable.create  # 创建常量变量的工厂函数别名

        d1 = {  # 创建一个普通字典 d1
            make_key("a"): ConstantVariable.create(10),  # 键为常量 "a"，值为常量变量 10
            make_key("b"): ConstantVariable.create(20),  # 键为常量 "b"，值为常量变量 20
        }
        d2 = collections.OrderedDict(  # 创建一个有序字典 d2
            [
                (make_key("x"), ConstantVariable.create(12)),  # 键为常量 "x"，值为常量变量 12
                (make_key("y"), ConstantVariable.create(22)),  # 键为常量 "y"，值为常量变量 22
            ]
        )
        self.assertEqual(ConstDictVariable(d1).python_type(), dict)  # 断言 d1 的类型是普通字典
        self.assertEqual(
            ConstDictVariable(d2, collections.OrderedDict).python_type(),
            collections.OrderedDict,  # 断言 d2 的类型是有序字典
        )

    def test_builtin_subclasses_as_method_on_class_type(self):
        class Foo:  # 定义类 Foo
            def __init__(self, name):  # 构造函数，初始化成员变量 name_
                self.ame_ = name  # 错误的成员变量赋值

            def get_name(self):  # 获取名称的方法
                return "Foo " + self.name_

        class Bar(Foo):  # 定义类 Bar，继承自 Foo
            def __init__(self, name):  # 构造函数，初始化成员变量 name_
                self.name_ = name  # 正确的成员变量赋值

            def get_name(self):  # 获取名称的方法，返回 "Bar " + self.name_
                return "Bar " + self.name_

        class Baz(Foo):  # 定义类 Baz，继承自 Foo
            def __init__(self, name):  # 构造函数，初始化成员变量 name_
                self.name_ = name  # 正确的成员变量赋值

            def get_name(self):  # 获取名称的方法，返回 "Baz " + self.name_
                return "Baz " + self.name_

        subs_of_foo_reg = Foo.__subclasses__()  # 获取类 Foo 的直接子类列表

        counter = CompileCounter()  # 创建一个编译计数器对象

        @torch._dynamo.optimize_assert(counter)  # 用编译计数器装饰的优化断言函数
        def fn():
            return Foo.__subclasses__()  # 返回类 Foo 的直接子类列表

        subs_of_foo_optim = fn()  # 调用优化断言函数，获取类 Foo 的直接子类列表

        self.assertEqual(len(subs_of_foo_reg), 2)  # 断言正常的子类个数为 2
        self.assertEqual(subs_of_foo_reg, subs_of_foo_optim)  # 断言正常的子类列表和优化后的子类列表相等

    def test_builtin_subclasses_as_method_on_var(self):
        class Foo:  # 定义类 Foo
            def __init__(self, name):  # 构造函数，初始化成员变量 name_
                self.name_ = name  # 正确的成员变量赋值

            def get_name(self):  # 获取名称的方法，返回 "Foo " + self.name_
                return "Foo " + self.name_

        class Bar(Foo):  # 定义类 Bar，继承自 Foo
            def __init__(self, name):  # 构造函数，初始化成员变量 name_
                self.name_ = name  # 正确的成员变量赋值

            def get_name(self):  # 获取名称的方法，返回 "Bar " + self.name_
                return "Bar " + self.name_

        class Baz(Bar):  # 定义类 Baz，继承自 Bar
            def __init__(self, name):  # 构造函数，初始化成员变量 name_
                self.name_ = name  # 正确的成员变量赋值

            def get_name(self):  # 获取名称的方法，返回 "Baz " + self.name_
                return "Baz " + self.name_

        subs_of_foo_reg = Foo.__subclasses__()  # 获取类 Foo 的直接子类列表
        sub_of_foo_subclass_var_reg = subs_of_foo_reg[0].__subclasses__()  # 获取 Foo 的第一个子类的子类列表

        sub_of_foo_subclass_var_optim = list()  # 创建一个空列表
        counter = CompileCounter()  # 创建一个编译计数器对象

        @torch._dynamo.optimize_assert(counter)  # 用编译计数器装饰的优化断言函数
        def fn():
            return Foo.__subclasses__()  # 返回类 Foo 的直接子类列表

        @torch._dynamo.optimize_assert(counter)  # 用编译计数器装饰的优化断言函数
        def fn_single(subs_of_foo_optim):
            return subs_of_foo_optim[0].__subclasses__()  # 返回类 Foo 第一个子类的子类列表

        subs_of_foo_optim = fn()  # 调用优化断言函数，获取类 Foo 的直接子类列表
        sub_of_foo_subclass_var_optim = fn_single(subs_of_foo_optim)  # 调用优化断言函数，获取 Foo 的第一个子类的子类列表

        self.assertEqual(len(sub_of_foo_subclass_var_optim), 1)  # 断言 Foo 的第一个子类的子类列表长度为 1
        self.assertEqual(sub_of_foo_subclass_var_optim, sub_of_foo_subclass_var_reg)  # 断言 Foo 的第一个子类的子类列表和正常的子类列表相等
    def test_builtin_str_on_user_defined_function(self):
        # 定义一个内部函数 another_fn，仅占位，无实际功能
        def another_fn():
            pass

        # 定义函数 fn，检查字符串 "another_fn" 是否在另一个函数的字符串表示中
        def fn():
            return "another_fn" in str(another_fn)

        # 使用 torch._dynamo.optimize 进行装饰，设置 nopython=True，优化函数 fn
        opt_fn = torch._dynamo.optimize(nopython=True)(fn)
        # 断言优化后的函数 opt_fn 返回 True
        self.assertTrue(opt_fn())

    def test_enum_no_graphbreaks(self):
        # 定义一个枚举类 Foo，包含 FOO 和 BAR 两个成员
        class Foo(enum.Enum):
            FOO = 0
            BAR = 1

        # 定义函数 fn，根据枚举类型 Foo 的不同值执行不同的操作
        def fn(x, foo):
            if foo is Foo.FOO:
                x = torch.add(x, 1.0)
            x = torch.mul(x, 1.0)
            return x

        # 生成一个随机张量 x
        x = torch.randn(1)
        # 创建一个用于计数编译次数的 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize 进行装饰，设置 nopython=True，优化函数 fn
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 调用优化后的函数 opt_fn，传入 x 和 Foo.FOO
        opt_fn(x, Foo.FOO)
        # 断言编译计数为 2
        self.assertEqual(cnts.op_count, 2)

        # 重置 torch._dynamo
        torch._dynamo.reset()
        # 重新创建 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 重新优化函数 fn
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 调用优化后的函数 opt_fn，传入 x 和 Foo.BAR
        opt_fn(x, Foo.BAR)
        # 断言编译计数为 1
        self.assertEqual(cnts.op_count, 1)

    def test_repeat_interleave_graphbreaks(self):
        # 定义函数 fn_no_breaks，执行 torch.repeat_interleave 操作
        def fn_no_breaks(x):
            # 在 self_int 上没有中断
            x += 1
            x = torch.repeat_interleave(x, 2, 3)
            x += 1
            return x

        # 定义函数 fn_has_breaks，使用 Tensor 类型参数执行 torch.repeat_interleave 操作
        def fn_has_breaks(x):
            # 在 self_Tensor 上有中断
            x += 1
            x = torch.repeat_interleave(x, torch.tensor(2), 3)
            x += 1
            return x

        # 生成一个随机张量 x，形状为 [4, 16, 1, 64]
        x = torch.randn([4, 16, 1, 64])

        # 创建一个用于计数帧数的 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize 进行装饰，优化函数 fn_no_breaks
        opt_fn = torch._dynamo.optimize(cnts)(fn_no_breaks)
        # 调用优化后的函数 opt_fn，传入 x
        opt_fn(x)
        # 断言帧数计数为 1
        self.assertEqual(cnts.frame_count, 1)

        # 重置 torch._dynamo
        torch._dynamo.reset()
        # 重新创建 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 重新优化函数 fn_has_breaks
        opt_fn = torch._dynamo.optimize(cnts)(fn_has_breaks)
        # 调用优化后的函数 opt_fn，传入 x
        opt_fn(x)
        # 断言帧数计数为 2
        self.assertEqual(cnts.frame_count, 2)

    def test_id_guarded_object(self):
        # 定义一个用户定义对象 UDO
        class UDO:
            # 使用 torch.compile(backend="eager") 对 call 方法进行装饰
            @torch.compile(backend="eager")
            def call(self, x, ref_id):
                # 获取当前对象的 id
                self_id = id(self)
                # 检查当前对象的 id 是否等于给定的 ref_id
                if self_id == ref_id:
                    x = torch.mul(x, 1.0)
                else:
                    x = torch.mul(x, 0)
                return x

        # 生成一个全为 1 的张量 x
        x = torch.ones(2)
        # 创建 UDO 类的一个实例 obj1
        obj1 = UDO()
        # 获取实例 obj1 的 id
        obj1_id = id(obj1)
        # 断言调用 obj1 的 call 方法返回全为 1 的张量
        self.assertEqual(obj1.call(x, obj1_id), torch.ones(2))

        # 创建 UDO 类的另一个实例 obj2
        obj2 = UDO()
        # 如果未安装 ID_MATCH: ___check_obj_id(L['self'], xxx)，此处断言失败
        self.assertEqual(obj2.call(x, obj1_id), torch.zeros(2))
    def test_id_guarded_module(self):
        class M(torch.nn.Module):
            def forward(self, x, ref_id):
                self_id = id(self)
                # 获取当前对象的唯一标识符
                if self_id == ref_id:
                    x = torch.mul(x, 1.0)
                else:
                    x = torch.mul(x, 0)
                return x

        cnts = torch._dynamo.testing.CompileCounter()

        # 确保在不同的 self 对象上执行 id(self) 时重新编译
        x = torch.ones(2)
        m1 = M()
        m1_id = id(m1)
        opt_m1 = torch._dynamo.optimize(cnts, nopython=True)(m1)
        self.assertEqual(opt_m1(x, m1_id), torch.ones(2))
        self.assertEqual(opt_m1(x, m1_id), torch.ones(2))

        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

        m2 = M()
        opt_m2 = torch._dynamo.optimize(cnts, nopython=True)(m2)
        # 如果没有安装 ID_MATCH: ___check_obj_id(L['self'], xxx) ，则测试失败。
        self.assertEqual(opt_m2(x, m1_id), torch.zeros(2))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 2)

    def test_id_of_nn_module(self):
        class M(torch.nn.Module):
            def forward(self, x, ref_id):
                self_id = id(self)
                # 获取当前对象的唯一标识符
                if self_id == ref_id:
                    x = torch.mul(x, 1.0)
                x = torch.add(x, 1.0)
                return x

        m = M().eval()
        data = torch.randn(1)
        cnts = torch._dynamo.testing.CompileCounter()
        correct_ref_id = id(m)
        opt_m = torch._dynamo.optimize(cnts, nopython=True)(m)
        opt_m(data, correct_ref_id)
        # 额外的操作是记录的相等性测试（虽然一旦跟踪被展平，这就是无用的！）
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnts.op_count, """2""")
        else:
            self.assertExpectedInline(cnts.op_count, """2""")

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        incorrect_ref_id = id(m) + 1
        opt_m = torch._dynamo.optimize(cnts, nopython=True)(m)
        opt_m(data, incorrect_ref_id)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnts.op_count, """1""")
        else:
            self.assertExpectedInline(cnts.op_count, """1""")

    def test_inline_func_jump_on_tensor_condition(self):
        def f1(input):
            if input == 0:
                return input + 1
            else:
                return input + 2

        def f2(input):
            return f1(input)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_f2 = torch._dynamo.optimize(cnts)(f2)
        res1 = opt_f2(torch.tensor([1.0]))
        res2 = opt_f2(torch.tensor([0.0]))

        self.assertEqual(res1, 3)
        self.assertEqual(res2, 1)
    # 测试函数：验证 torch.add 是否在 frozenset funcs 中，并统计操作次数是否为 2
    def test_frozenset_torch_func_contains(self):
        # 创建一个不可变集合 funcs，包含 torch.add 函数
        funcs = frozenset([torch.add])

        # 定义函数 fn，接受 x 和 func 两个参数
        def fn(x, func):
            # 检查 func 是否在 funcs 中
            if func in funcs:
                # 如果在，则调用 torch.add 函数将 x 增加 1.0
                x = torch.add(x, 1.0)
            # 无论如何都调用 torch.mul 函数将 x 乘以 1.0
            x = torch.mul(x, 1.0)
            return x

        # 生成一个随机张量 x
        x = torch.randn(1)
        # 实例化一个 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn 进行优化，得到 opt_fn
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 调用 opt_fn，传入 x 和 torch.add
        opt_fn(x, torch.add)
        # 断言操作计数 cnts.op_count 是否为 2
        self.assertEqual(cnts.op_count, 2)

        # 重置 torch._dynamo
        torch._dynamo.reset()
        # 重新实例化 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn 进行优化，再次得到 opt_fn
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 调用 opt_fn，传入 x 和 torch.mul
        opt_fn(x, torch.mul)
        # 断言操作计数 cnts.op_count 是否为 1
        self.assertEqual(cnts.op_count, 1)

    # 测试函数：验证在函数调用过程中列表的修改是否能正确地内联
    def test_inline_list_mutation(self):
        # 定义函数 f1，向列表 x 中添加一个张量
        def f1(x):
            x.append(torch.ones(8))
            return x

        # 定义函数 f2，创建一个包含一个张量的列表，并调用 f1 修改它
        def f2():
            x = [torch.ones(6)]
            f1(x)
            return x

        # 调用 f2 得到结果 res1
        res1 = f2()
        # 实例化 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 f2 进行优化，得到 opt_f2
        opt_f2 = torch._dynamo.optimize(cnts)(f2)
        # 调用 opt_f2 得到结果 res2
        res2 = opt_f2()
        # 断言 res1 和 res2 是否相等
        self.assertTrue(same(res1, res2))

    # 测试函数：验证在函数调用过程中字典的修改是否能正确地内联
    def test_inline_dict_mutation(self):
        # 定义函数 f1，修改字典 d 中的元素
        def f1(d):
            d["c"] = d["a"] + d.pop("b")
            return d

        # 定义函数 f2，创建一个包含两个张量的字典 d，并调用 f1 修改它
        def f2():
            d = {"a": torch.ones(5), "b": torch.ones(5)}
            f1(d)
            return d

        # 调用 f2 得到结果 res1
        res1 = f2()
        # 实例化 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 f2 进行优化，得到 opt_f2
        opt_f2 = torch._dynamo.optimize(cnts)(f2)
        # 调用 opt_f2 得到结果 res2
        res2 = opt_f2()
        # 断言 res1 和 res2 是否相等
        self.assertTrue(same(res1, res2))

    # 测试函数：验证在函数调用过程中本地字典的清空是否能正确地内联
    def test_inline_local_dict_clear(self):
        # 定义函数 f，清空字典 d 并返回
        def f(d):
            d.clear()
            return d

        # 定义输入字典 inp
        inp = {"a": torch.randn(2, 2), "b": torch.randn(2, 2)}
        # 调用 torch.compile 对 f 进行编译，传入 inp 字典，使用 eager backend 并启用 fullgraph 模式，得到输出 out
        out = torch.compile(f, backend="eager", fullgraph=True)(inp)
        # 断言 out 的长度是否为 0
        self.assertEqual(len(out), 0)
        # 断言输入字典 inp 的长度是否为 0
        self.assertEqual(len(inp), 0)

    # 测试函数：验证在模块属性字典中的清空是否能正确地内联
    def test_inline_module_attr_dict_clear(self):
        # 定义 MyMod 类，继承自 torch.nn.Module
        class MyMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化属性 a 为包含两个张量的字典
                self.a = {"a": torch.randn(2, 2), "b": torch.randn(2, 2)}

            # 定义 forward 方法，清空属性 a 并返回
            def forward(self):
                self.a.clear()
                return self.a

        # 实例化 MyMod 对象 m
        m = MyMod()
        # 调用 torch.compile 对 m 进行编译，使用 eager backend 并启用 fullgraph 模式，得到输出 out
        out = torch.compile(m, backend="eager", fullgraph=True)()
        # 断言 out 的长度是否为 0
        self.assertEqual(len(out), 0)
        # 断言对象 m 的属性 a 的长度是否为 0
        self.assertEqual(len(m.a), 0)

    # 测试函数：验证在用户定义的类属性字典中的清空是否能正确地内联
    def test_inline_user_defined_dict_attr_clear(self):
        # 定义 MyMod 类
        class MyMod:
            def __init__(self):
                # 初始化属性 a 为包含两个张量的字典
                self.a = {"a": torch.randn(2, 2), "b": torch.randn(2, 2)}

        # 定义函数 f，接受对象 obj 和输入 inp 作为参数，返回清空后的 obj.a 和 ret（inp 与 obj.a 初始长度之和）
        def f(obj, inp):
            ret = len(obj.a) + inp
            obj.a.clear()
            return obj.a, ret

        # 实例化 MyMod 对象 m
        m = MyMod()
        # 记录 m.a 初始长度
        before_len = len(m.a)
        # 创建张量输入 t_inp
        t_inp = torch.ones(1)
        # 调用 torch.compile 对 f 进行编译，传入 m 和 t_inp，使用 eager backend 并启用 fullgraph 模式，得到输出 d 和 ret
        d, ret = torch.compile(f, backend="eager", fullgraph=True)(m, t_inp)
        # 断言对象 m 的属性 a 的长度是否为 0
        self.assertEqual(len(m.a), 0)
        # 断言输出字典 d 的长度是否为 0
        self.assertEqual(len(d), 0)
        # 断言 ret 是否等于 t_inp 加上初始长度 before_len
        self.assertEqual(ret, t_inp + before_len)
    def test_recursive_inline_list_mutation(self):
        # 定义函数 f1，向列表 x 和 y 中添加张量 [1.1] 和 [1.2]，并返回更新后的 x 和 y
        def f1(x, y):
            x.append(torch.tensor([1.1]))
            y.append(torch.tensor([1.2]))
            return x, y

        # 定义函数 f2，向列表 x 和 y 中添加张量 [2.1] 和 [2.2]，然后调用 f1 更新 x 和 y，并返回更新后的 x 和 y
        def f2(x, y):
            x.append(torch.tensor([2.1]))
            y.append(torch.tensor([2.2]))
            f1(x, y)  # 调用函数 f1 更新 x 和 y
            return x, y

        # 定义函数 f3，向列表 x 中添加张量 [3.1]，创建列表 y 包含张量 [3.2]，然后调用 f2 更新 x 和 y，并返回更新后的 x 和 y
        def f3(x):
            x.append(torch.tensor([3.1]))
            y = [torch.tensor([3.2])]
            f2(x, y)  # 调用函数 f2 更新 x 和 y
            return x, y

        # 定义函数 f4，创建列表 x 包含张量 [4.1]，然后调用 f3 更新 x 和 y，并返回更新后的 x 和 y
        def f4():
            x = [torch.tensor([4.1])]
            return f3(x)

        # 调用函数 f4，并比较其结果与优化后的结果是否相同
        res1 = f4()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_f4 = torch._dynamo.optimize(cnts)(f4)  # 对函数 f4 进行优化
        res2 = opt_f4()  # 执行优化后的函数 f4
        self.assertTrue(same(res1, res2))  # 断言优化前后结果是否相同

    def test_sample_input(self):
        from torch.testing._internal.common_methods_invocations import SampleInput

        # 定义函数 fn，如果 sample.input 是张量则返回其乘以 2，否则返回形状为 () 的零张量
        def fn(sample):
            if isinstance(sample.input, torch.Tensor):
                return sample.input * 2
            return torch.zeros(())

        sample = SampleInput(torch.ones(2))  # 创建 SampleInput 实例
        ref = fn(sample)  # 调用 fn 函数得到参考结果 ref

        opt_fn = torch._dynamo.optimize("eager")(fn)  # 对函数 fn 进行 eager 模式的优化
        res = opt_fn(sample)  # 执行优化后的 fn 函数得到结果 res

        self.assertTrue(same(ref, res))  # 断言优化前后结果是否相同

    def test_release_input_memory(self):
        x = torch.rand([4])  # 创建张量 x
        x_ref = weakref.ref(x)  # 创建 x 的弱引用对象 x_ref

        cnts = torch._dynamo.testing.CompileCounter()

        # 定义函数 foo，对输入张量 x 执行 x + x 操作
        @torch._dynamo.optimize(cnts)
        def foo(x):
            return x + x

        out = foo(x)  # 调用函数 foo 得到结果 out
        self.assertTrue(same(out, x + x))  # 断言函数 foo 的结果与预期是否相同
        del x  # 删除对张量 x 的引用
        self.assertIs(x_ref(), None)  # 断言张量 x 是否被正确释放

    def test_release_module_memory(self):
        mod = torch.nn.Linear(10, 10)  # 创建神经网络模块 mod
        x = torch.rand([10, 10])  # 创建张量 x
        mod_weight_ref = weakref.ref(mod.weight)  # 创建模块权重的弱引用对象 mod_weight_ref
        mod_ref = weakref.ref(mod)  # 创建模块 mod 的弱引用对象 mod_ref

        # 定义 NoLeakBackend 类，用于优化函数时防止内存泄漏
        class NoLeakBackend:
            def __call__(self, gm: torch.fx.GraphModule, example_inputs):
                gm.mod = None  # 从生成的 GraphModule 中移除对模块的引用

                def foo(*args, **kwargs):
                    return (1,)  # 返回常数 1

                return foo

        no_leak_backend = NoLeakBackend()

        # 定义函数 foo，接受模块 mod 和张量 x 作为输入，并返回模块对输入张量的输出
        @torch._dynamo.optimize(no_leak_backend)
        def foo(mod, x):
            return mod(x)

        foo(mod, x)  # 调用函数 foo
        del mod  # 删除对模块 mod 的引用
        del x  # 删除对张量 x 的引用
        self.assertIsNone(mod_ref())  # 断言模块 mod 是否被正确释放
        self.assertIsNone(mod_weight_ref())  # 断言模块权重是否被正确释放

    def test_release_scope_memory(self):
        # 定义函数 inner，接受参数 y 并直接返回
        def inner(y):
            y

        inner = torch._dynamo.optimize("eager")(inner)  # 对函数 inner 进行 eager 模式的优化

        p_ref = None

        x = torch.randn((10, 10))  # 创建张量 x
        inner(x)  # 调用优化后的 inner 函数处理张量 x

        p_ref = weakref.ref(x)  # 创建张量 x 的弱引用对象 p_ref
        self.assertTrue(p_ref() is not None)  # 断言张量 x 的引用存在
        del x  # 删除对张量 x 的引用
        self.assertTrue(p_ref() is None)  # 断言张量 x 是否被正确释放
    def test_update_locals_and_stack_uses_shared_cache(self):
        # 定义一个函数 fn，接受参数 x
        def fn(x):
            # 定义一个排列 perm
            perm = [0, 3, 5]
            # 将排列 perm 转换为列表，并在开头插入 0 到 min(perm) 的范围内的值
            perm = list(range(min(perm))) + perm
            # 扩展 perm，包括所有 x.dim() 中不在 perm 中的索引
            perm.extend(i for i in range(x.dim()) if i not in perm)
            # 返回排列 perm
            return perm

        # 创建一个形状为 [2, 2, 2, 2, 2, 2] 的随机张量 x
        x = torch.rand([2, 2, 2, 2, 2, 2])
        # 调用 fn 函数，得到结果 res1
        res1 = fn(x)
        # 创建一个 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化，得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 使用优化后的函数 opt_fn 处理张量 x，得到结果 res2
        res2 = opt_fn(x)
        # 断言 res1 和 res2 相同
        self.assertTrue(same(res1, res2))

    def test_dict_reconstruct_keeps_original_order(self):
        # 定义一个函数 fn，无参数
        def fn():
            # 创建一个有序字典 modules，包含键值对 ("act", torch.nn.ReLU())
            modules = collections.OrderedDict([("act", torch.nn.ReLU())])
            # 使用 modules 创建一个 ModuleDict 对象 module_dict
            module_dict = torch.nn.ModuleDict(modules)

            # 创建一个字典 next_modules，包含键值对 {"fc4": torch.nn.Linear(5, 6), "act3": torch.nn.Sigmoid()}
            next_modules = {"fc4": torch.nn.Linear(5, 6), "act3": torch.nn.Sigmoid()}
            # 更新 modules，将 next_modules 的所有项添加到 modules 中
            modules.update(next_modules.items())
            # 更新 module_dict，将 next_modules 的所有项添加到 module_dict 中
            module_dict.update(next_modules)
            # 返回 modules 和 module_dict
            return modules, module_dict

        # 创建一个 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化，得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 使用优化后的函数 opt_fn 执行，得到 modules 和 module_dict
        modules, module_dict = opt_fn()

        # 断言 module_dict 的长度与 modules 的长度相同
        self.assertEqual(len(module_dict), len(modules))
        # 遍历 modules 和 module_dict，断言 modules 中每个键对应的值与 module_dict 中对应子模块相同
        for k1, m2 in zip(modules, module_dict.children()):
            self.assertTrue(modules[k1] is m2)

    def test_side_effects_codegen_update_mutated(self):
        # 定义函数 f1，接受参数 x
        def f1(x):
            # 创建一个列表 alist，包含 x 和 x+1
            alist = [x]
            alist.append(x + 1)
            # 对 alist[0] 执行求和并获取其标量值
            alist[0].sum().item()  # graph break
            # 弹出 alist 的最后一个元素，存入变量 res
            res = alist.pop()
            # 对 res 执行求和并获取其标量值
            res.sum().item()  # graph break
            # 返回 res
            return res

        # 定义函数 f2，接受参数 a 和 b
        def f2(a, b):
            # 创建一个字典 d，包含键值对 {"a": a+1, "b": b+2}
            d = {"a": a + 1, "b": b + 2}
            # 弹出字典 d 中键为 "b" 的值，存入变量 x
            x = d.pop("b")
            # 对 x 执行求和并获取其标量值
            x.sum().item()  # graph break
            # 计算 y = d["a"] + x
            y = d["a"] + x
            # 对 y 执行求和并获取其标量值
            y.sum().item()  # graph break
            # 将 y 添加到字典 d 中，键为 "c"
            d["c"] = y
            # 返回字典 d
            return d

        # 创建张量 x，形状为 [2, 3]
        x = torch.rand([2, 3])
        # 创建张量 a 和 b，形状为 [5, 6]
        a = torch.rand([5, 6])
        b = torch.rand([5, 6])
        # 调用 f1 和 f2 函数，得到结果 res11 和 res21
        res11 = f1(x)
        res21 = f2(a, b)
        # 创建一个 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 f1 和 f2 函数进行优化，得到优化后的函数 opt_f1 和 opt_f2
        opt_f1 = torch._dynamo.optimize(cnts)(f1)
        opt_f2 = torch._dynamo.optimize(cnts)(f2)
        # 使用优化后的函数 opt_f1 和 opt_f2 处理输入，得到结果 res12 和 res22
        res12 = opt_f1(x)
        res22 = opt_f2(a, b)
        # 断言 res11 与 res12 相同，res21 与 res22 相同
        self.assertTrue(same(res11, res12))
        self.assertTrue(same(res21, res22))

    def test_list_append_return_none(self):
        # 定义函数 fn，接受参数 x
        def fn(x):
            # 创建一个空列表 alist
            alist = []
            # 向 alist 中追加 x+1，并将返回值存入 blist（实际上 blist 为 None）
            blist = alist.append(x + 1)
            # 返回 alist 和 blist
            return alist, blist

        # 创建张量 x，值为 [2.3]
        x = torch.tensor([2.3])
        # 调用 fn 函数，得到结果 res
        res = fn(x)
        # 创建一个 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化，得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 使用优化后的函数 opt_fn 处理张量 x，得到结果 res2
        res2 = opt_fn(x)
        # 断言 res 与 res2 相等
        self.assertEqual(res, res2)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    # 测试函数，验证通过列表中的张量构造张量的行为
    def test_tensor_ctor_list_of_tensor(self):
        # 定义一个函数，接受一个张量 x，返回一个包含 x 的张量，数据类型为 torch.int64
        def fn(x):
            return torch.tensor([x], dtype=torch.int64)

        # 创建一个张量 x，值为 20
        x = torch.tensor(20)
        # 调用 fn 函数，将 x 作为参数，得到结果 res
        res = fn(x)
        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化，得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 使用优化后的函数 opt_fn 处理输入 x，得到结果 res2
        res2 = opt_fn(x)
        # 断言结果 res 和 res2 相等
        self.assertEqual(res, res2)
        # 断言编译帧计数为 1
        self.assertEqual(cnts.frame_count, 1)

    # 测试函数，验证不同张量类型的行为
    def test_tensor_types(self):
        # 定义一个函数 fn，接受两个参数 dtype 和 tensor_type，创建一个指定类型和数据类型的空张量 x，并断言 x 是 tensor_type 类型
        def fn(dtype, tensor_type):
            x = torch.empty(4, dtype=dtype)
            assert isinstance(x, tensor_type)

        # 对 fn 函数进行优化，得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 分别调用优化后的函数 opt_fn，验证不同类型的张量行为
        opt_fn(torch.float32, torch.FloatTensor)
        opt_fn(torch.float64, torch.DoubleTensor)
        opt_fn(torch.float16, torch.HalfTensor)
        opt_fn(torch.bfloat16, torch.BFloat16Tensor)
        opt_fn(torch.uint8, torch.ByteTensor)
        opt_fn(torch.int8, torch.CharTensor)
        opt_fn(torch.int64, torch.LongTensor)
        opt_fn(torch.int, torch.IntTensor)
        opt_fn(torch.int16, torch.ShortTensor)
        opt_fn(torch.bool, torch.BoolTensor)

    # 测试函数，验证处理 NaN 的函数行为
    def test_nan(self):
        # 定义一个函数 f，接受两个参数 x 和 n，返回 x * 2 + n 的结果
        def f(x, n):
            return x * 2 + n

        # 创建一个包含随机数的张量 x
        x = torch.randn(4)
        # 创建一个 NaN 值
        n = float("nan")

        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 f 函数进行优化，得到优化后的函数 opt_f
        opt_f = torch._dynamo.optimize(cnts)(f)
        # 使用优化后的函数 opt_f 处理输入 x 和 n
        opt_f(x, n)
        opt_f(x, n)
        # 断言编译帧计数为 1
        self.assertEqual(cnts.frame_count, 1)

    # 使用 patch 对象，测试函数，验证 torch.Tensor.item() 方法的行为
    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_item(self):
        # 定义一个继承自 torch.nn.Module 的模块 MyMod
        class MyMod(torch.nn.Module):
            # 定义模块的前向传播函数 forward，接受一个输入 x
            def forward(self, x):
                # 计算张量 x 的最大值，并将其转换为整数类型，返回其标量值
                z = torch.max(x)
                return z.int().item()

        # 创建一个张量 x，包含指定的数据
        x = torch.tensor([[10.6763, 11.7445, -2.2369]])
        # 创建 MyMod 类的实例 model
        model = MyMod()
        # 对 model 进行优化，得到优化后的模型
        y = torch._dynamo.optimize("eager", nopython=True)(model)(x)

        # 断言优化后的模型输出为 11
        self.assertEqual(y, 11)

    # 使用 patch 对象，测试函数，验证在改变输入形状后 torch.Tensor.item() 方法的行为
    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_item_changes(self):
        # 定义一个继承自 torch.nn.Module 的模块 MyMod
        class MyMod(torch.nn.Module):
            # 定义模块的前向传播函数 forward，接受一个输入 x
            def forward(self, x):
                # 计算张量 x 的最大值，并将其转换为整数类型，返回其标量值
                z = torch.max(x)
                return z.int().item()

        # 创建一个张量 x，包含指定的数据
        x = torch.tensor([[10.6763, 11.7445, -2.2369]])
        # 创建 MyMod 类的实例 model
        model = MyMod()
        # 对 model 进行优化，得到优化后的模型
        opt_model = torch._dynamo.optimize("eager", nopython=True)(model)
        # 使用优化后的模型处理输入 x，得到结果 y
        y = opt_model(x)
        # 使用优化后的模型处理新的输入形状的张量，得到结果 z
        z = opt_model(torch.tensor([[y - 5, y + 10, y + 50]]))

        # 断言优化后的模型输出为 11
        self.assertEqual(y, 11)
        # 断言优化后的模型输出为 61
        self.assertEqual(z, 61)

    # 使用 patch 对象，测试函数，验证在改变输入形状后 torch.Tensor.item() 方法的行为
    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_item_changes_new_shape(self):
        # 定义一个继承自 torch.nn.Module 的模块 MyMod
        class MyMod(torch.nn.Module):
            # 定义模块的前向传播函数 forward，接受一个输入 x
            def forward(self, x):
                # 计算张量 x 的最大值，并将其转换为整数类型，返回其标量值
                z = torch.max(x)
                return z.int().item()

        # 创建一个张量 x，包含指定的数据
        x = torch.tensor([[10.6763, 11.7445, -2.2369]])
        # 创建 MyMod 类的实例 model
        model = MyMod()
        # 对 model 进行优化，得到优化后的模型
        opt_model = torch._dynamo.optimize("eager", nopython=True)(model)
        # 使用优化后的模型处理输入 x，得到结果 y
        y = opt_model(x)
        # 使用优化后的模型处理新的输入形状的张量，得到结果 z
        z = opt_model(torch.tensor([[y - 5, y + 50], [y + 5, y - 50]]))

        # 断言优化后的模型输出为 11
        self.assertEqual(y, 11)
        # 断言优化后的模型输出为 61
        self.assertEqual(z, 61)
    # 标记该测试用例为跳过状态，并提供链接到 GitHub issue 的说明
    @unittest.skip("https://github.com/pytorch/pytorch/issues/99726")
    def test_cross_entropy_loss_fancy_ctor1(self):
        # 生成一个大小为5的随机张量
        rand_5 = torch.randn(5)
        # 生成一个大小为3x5的随机张量
        rand_3_5 = torch.randn(3, 5)
        # 生成一个大小为3的长整型张量，并随机填充范围在0到4之间的整数
        target = torch.empty(3, dtype=torch.long).random_(5)

        # 使用权重rand_5、不进行减少、标签平滑系数为0.5创建交叉熵损失对象
        loss = torch.nn.CrossEntropyLoss(
            weight=rand_5, reduce=False, label_smoothing=0.5
        )
        # 优化损失函数以支持"急切"模式和无Python编译优化
        opt_loss = torch._dynamo.optimize("eager", nopython=True)(loss)
        # 输入数据为rand_3_5，目标标签为target，计算优化后的损失函数输出
        dynamo_output = opt_loss(input, target)

        # 使用权重rand_5、不进行减少、标签平滑系数为0.5创建另一个交叉熵损失对象
        loss = torch.nn.CrossEntropyLoss(
            weight=rand_5, reduce=False, label_smoothing=0.5
        )
        # 输入数据为rand_3_5，目标标签为target，计算原始损失函数的输出
        input = rand_3_5
        output = loss(input, target)

        # 断言优化后的输出与原始输出在数值上接近
        self.assertTrue(torch.allclose(dynamo_output, output))

    def test_cross_entropy_loss_fancy_ctor2(self):
        # 生成一个大小为3x5的随机张量
        rand_3_5 = torch.randn(3, 5)
        # 生成一个大小为3的长整型张量，并随机填充范围在0到4之间的整数
        target = torch.empty(3, dtype=torch.long).random_(5)

        # 使用不进行减少、标签平滑系数为0.5创建交叉熵损失对象
        loss = torch.nn.CrossEntropyLoss(reduce=False, label_smoothing=0.5)
        # 优化损失函数以支持"急切"模式和无Python编译优化
        opt_loss = torch._dynamo.optimize("eager", nopython=True)(loss)
        # 输入数据为rand_3_5，目标标签为target，计算优化后的损失函数输出
        dynamo_output = opt_loss(input, target)

        # 使用不进行减少、标签平滑系数为0.5创建另一个交叉熵损失对象
        loss = torch.nn.CrossEntropyLoss(reduce=False, label_smoothing=0.5)
        # 输入数据为rand_3_5，目标标签为target，计算原始损失函数的输出
        input = rand_3_5
        output = loss(input, target)

        # 断言优化后的输出与原始输出在数值上接近
        self.assertTrue(torch.allclose(dynamo_output, output))

    def test_cross_entropy_loss_simple_ctor(self):
        # 初始化输出为None
        output = None
        # 生成一个大小为3x5的随机张量
        rand_3_5 = torch.randn(3, 5)
        # 生成一个大小为3的长整型张量，并随机填充范围在0到4之间的整数
        target = torch.empty(3, dtype=torch.long).random_(5)

        # 使用默认参数创建交叉熵损失对象
        loss = torch.nn.CrossEntropyLoss()
        # 优化损失函数以支持"急切"模式和无Python编译优化
        opt_loss = torch._dynamo.optimize("eager", nopython=True)(loss)
        # 输入数据为rand_3_5，目标标签为target，计算优化后的损失函数输出
        dynamo_output = opt_loss(input, target)

        # 使用默认参数创建另一个交叉熵损失对象
        loss = torch.nn.CrossEntropyLoss()
        # 输入数据为rand_3_5，目标标签为target，计算原始损失函数的输出
        input = rand_3_5
        output = loss(input, target)

        # 断言优化后的输出与原始输出在数值上接近
        self.assertTrue(torch.allclose(dynamo_output, output))

    def test_nn_functional_reduction(self):
        # 定义一个函数fn，接受损失和减少方式作为参数
        def fn(loss, reduction):
            # 获取减少方式的枚举值
            reduction_enum = F._Reduction.get_enum(reduction)
            # 根据减少方式枚举值进行处理
            if reduction_enum == 0:
                return loss
            elif reduction_enum == 1:
                return loss.mean()
            elif reduction_enum == 2:
                return loss.sum()

        # 生成一个大小为3x5的随机张量
        x = torch.rand([3, 5])
        # 减少方式设为"mean"
        y = "mean"
        # 使用fn函数计算参考值ref
        ref = fn(x, y)
        # 优化fn函数以支持"急切"模式和无Python编译优化
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 使用优化后的fn函数计算结果res
        res = opt_fn(x, y)
        # 断言优化后的结果与参考值在数值上接近
        self.assertTrue(torch.allclose(ref, res))

    def test_large_reduction_list(self):
        # 数据类型为32位浮点数
        dtype = torch.float32
        # 设备为CPU
        device = "cpu"

        # 定义一个检查函数，用于验证张量的总和计算
        def check_sum_all(tensor: torch.Tensor) -> None:
            # 将张量重塑为一维数组并转换为Python列表
            pylist = tensor.reshape(-1).tolist()
            # 断言张量的总和与Python列表的总和相等
            self.assertTrue(same(tensor.sum(), torch.tensor(sum(pylist))))

        # 检查一个大小为200000的随机张量的总和
        check_sum_all(torch.randn(200000, dtype=dtype, device=device))
    def test_named_parameters(self):
        n_embd = 768
        block_size = 128
        vocab_size = 65
        embd_pdrop = 0.1

        class MyModel2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义词嵌入层，将vocab_size个词嵌入到n_embd维度的向量空间中
                self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
                # 定义位置嵌入层，使用torch.nn.Parameter将一个全零张量转化为可训练的参数
                self.pos_emb = torch.nn.Parameter(torch.zeros(1, block_size, n_embd))
                # 定义dropout层，以embd_pdrop的概率丢弃输入张量中的元素
                self.drop = torch.nn.Dropout(embd_pdrop)

            def forward(self, x):
                return x

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义词嵌入层，将vocab_size个词嵌入到n_embd维度的向量空间中
                self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
                # 定义位置嵌入层，使用torch.nn.Parameter将一个全零张量转化为可训练的参数
                self.pos_emb = torch.nn.Parameter(torch.zeros(1, block_size, n_embd))
                # 定义dropout层，以embd_pdrop的概率丢弃输入张量中的元素
                self.drop = torch.nn.Dropout(embd_pdrop)
                # 定义子模型MyModel2，包含在当前模型中
                self.submod2 = MyModel2()

            def forward(self, x):
                return x

        # Regular
        params = []
        # 创建模型实例
        mod = MyModel()
        # 获取模型中所有参数的名称及其对应的值
        actual_params = list(mod.named_parameters())

        @torch._dynamo.optimize("eager", nopython=True)
        def fn():
            return list(mod.named_parameters())

        # 调用fn函数，获取模型中所有参数的名称及其对应的值
        params = fn()

        # 断言实际参数列表和fn函数返回的参数列表长度相等
        self.assertEqual(len(actual_params), len(params))
        # 遍历参数列表，逐一比较参数名称和值是否相等
        for idx in range(len(params)):
            k_a, v_a = actual_params[idx]
            k, v = params[idx]
            self.assertEqual(k_a, k)
            # 使用torch.allclose函数检查参数值是否在误差允许范围内相等
            self.assertTrue(torch.allclose(v_a, v))

        # Prefix
        params = []
        # 创建模型实例
        mod = MyModel()
        # 获取模型中所有参数的名称及其对应的值，且参数名称使用前缀"foo"
        actual_params = list(mod.named_parameters(prefix="foo"))

        @torch._dynamo.optimize("eager", nopython=True)
        def fn1():
            return list(mod.named_parameters(prefix="foo"))

        # 调用fn1函数，获取模型中所有参数的名称及其对应的值，且参数名称使用前缀"foo"
        params = fn1()

        # 断言实际参数列表和fn1函数返回的参数列表长度相等
        self.assertEqual(len(actual_params), len(params))
        # 遍历参数列表，逐一比较参数名称和值是否相等
        for idx in range(len(params)):
            k_a, v_a = actual_params[idx]
            k, v = params[idx]
            self.assertEqual(k_a, k)
            # 使用torch.allclose函数检查参数值是否在误差允许范围内相等
            self.assertTrue(torch.allclose(v_a, v))
    # 定义一个测试函数，用于测试模型的迭代器
    def test_module_complex_iter(self):
        # 设置嵌入向量大小为768
        n_embd = 768
        # 设置块大小为128
        block_size = 128
        # 设置词汇表大小为65
        vocab_size = 65
        # 设置嵌入层的丢弃率为0.1
        embd_pdrop = 0.1

        # 定义一个虚拟的 GPT 模型类
        class FakeGPT(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建 token embedding 层，将词汇映射到768维的向量
                self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
                # 创建位置 embedding 层，形状为[1, block_size, n_embd]，初始化为全零
                self.pos_emb = torch.nn.Parameter(torch.zeros(1, block_size, n_embd))
                # 创建丢弃层，以embd_pdrop的概率丢弃输入
                self.drop = torch.nn.Dropout(embd_pdrop)
                # 创建 Layer Normalization 层，用于归一化输入
                self.ln_f = torch.nn.LayerNorm(n_embd)
                # 创建线性层，将768维向量映射到65维，用于输出
                self.head = torch.nn.Linear(n_embd, vocab_size, bias=False)

                # 设置模型的块大小
                self.block_size = block_size
                # 初始化模型的名称列表
                self.names = []

            # 前向传播方法
            def forward(self, idx, targets=None):
                # 获取输入的批次大小和序列长度
                b, t = idx.size()
                # 检查序列长度是否超过模型块大小，如果是则报错
                assert (
                    t <= self.block_size
                ), "Cannot forward, model block size is exhausted."

                # 前向传播 GPT 模型
                token_embeddings = self.tok_emb(
                    idx
                )  # 每个索引映射到一个（可学习的）向量
                position_embeddings = self.pos_emb[
                    :, :t, :
                ]  # 每个位置映射到一个（可学习的）向量
                x = self.drop(token_embeddings + position_embeddings)
                x = self.blocks(x)  # 调用未定义的blocks方法，可能存在错误
                x = self.ln_f(x)
                logits = self.head(x)

                # 如果提供了目标值，计算损失
                loss = None
                if targets is not None:
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), targets.view(-1)
                    )

                return logits, loss

            # 定义一个自定义方法foo，用于遍历模型的命名模块和参数
            def foo(self, memo=None, prefix="", remove_duplicate=False):
                # 遍历命名模块及其参数
                for mn, m in self.named_modules(
                    memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
                ):
                    for pn, p in self.named_parameters():
                        fpn = f"{mn}.{pn}" if mn else pn
                        # 将完整的参数名加入到模型的名称列表中
                        self.names.append(fpn)

        # 测试简单递归调用
        model_a = FakeGPT()
        model_a.foo()
        a_names = model_a.names

        # 创建新的模型实例并优化，然后调用foo方法
        model_b = FakeGPT()
        opt_model_b = torch._dynamo.optimize("eager", nopython=True)(model_b)
        opt_model_b.foo()

        # 断言两个模型实例的名称列表相等
        self.assertEqual(a_names, model_b.names)

        # 测试带有前缀的递归调用
        model_a = FakeGPT()
        model_a.foo(prefix="abc")
        a_names = model_a.names

        # 创建新的模型实例并优化，然后带有相同的前缀调用foo方法
        model_b = FakeGPT()
        opt_model_b = torch._dynamo.optimize("eager", nopython=True)(model_b)
        opt_model_b.foo(prefix="abc")

        # 断言两个模型实例的名称列表相等
        self.assertEqual(a_names, model_b.names)
    # 定义一个测试函数，用于验证输入的 m 是否为 numpy 数组类型，返回 x+1 或 x-1 的结果
    def test_numpy_variable_isinstance(self):
        # 定义内部函数 fn，判断 m 是否为 numpy 数组，返回 x+1 或 x-1
        def fn(x, m):
            if isinstance(m, np.ndarray):
                return x + 1
            else:
                return x - 1
        
        # 创建一个 PyTorch 张量 x，包含单个元素 2.3
        x = torch.tensor([2.3])
        # 创建一个 numpy 数组 m，包含元素 [1, 2, 3]
        m = np.array([1, 2, 3])
        # 计算参考值 ref，调用 fn 函数
        ref = fn(x, m)
        # 使用 CompileCounter 对 fn 函数进行优化，返回优化后的函数 opt_fn
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 计算结果 res，调用优化后的函数 opt_fn
        res = opt_fn(x, m)
        # 断言优化前后结果相等
        self.assertEqual(ref, res)

        # 测试另一条路径，此时 m 为 x，期望结果与之前相同
        ref = fn(x, x)
        res = opt_fn(x, x)
        # 断言优化前后结果相等
        self.assertEqual(ref, res)

    # 测试张量求点积和梯度计算过程中不中断图的情况
    def test_tensor_dot_grad_no_graph_break(self):
        # 定义函数 fn，计算表达式 y = 3 * a**3 - b**2，并计算其梯度
        def fn(a, b):
            y = 3 * a**3 - b**2
            y.backward(gradient=torch.tensor([1.0, 1.0]))
            # 清零 b 的梯度
            b.grad.zero_()
            return a.grad, b.grad
        
        # 创建两个张量 a 和 b，要求计算梯度
        a = torch.tensor([2.0, 3.0], requires_grad=True)
        b = torch.tensor([6.0, 4.0], requires_grad=True)
        # 使用 CompileCounter 对 fn 函数进行优化，返回优化后的函数 opt_fn
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 计算优化后的结果，并获取 b 的梯度
        _, b_grad = opt_fn(a, b)
        # 断言 b 的梯度为零向量
        self.assertTrue(same(b_grad, torch.tensor([0.0, 0.0])))
        # 断言编译计数器的帧数为 2
        self.assertEqual(cnts.frame_count, 2)

    # 测试 torch.nn.Parameter 是否为 torch.Tensor 类型的实例
    def test_torch_nn_parameter_isinstance(self):
        # 定义函数 fn，创建一个张量 a 作为 torch.nn.Parameter，判断其是否为 torch.Tensor 类型
        def fn(x):
            a = torch.nn.Parameter(torch.rand(2, 3))
            if isinstance(a, torch.Tensor):
                return x + 1
            else:
                return x - 1
        
        # 创建一个张量 x，包含单个元素 2.5
        x = torch.tensor([2.5])
        # 计算参考值 ref，调用 fn 函数
        ref = fn(x)
        # 使用 "eager" 模式对 fn 函数进行优化，返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 计算结果 res，调用优化后的函数 opt_fn
        res = opt_fn(x)
        # 断言优化前后结果相等
        self.assertEqual(ref, res)

    # 在执行函数 foo 之后进行优化，并验证期望的输出结果、帧数和缓存后端数目
    def _optimize_then_check_exp(
        self, foo, args, cnt, exp_out, exp_frame_count, exp_n_cached_backend
    ):
        # 使用指定的后端对 foo 函数进行优化，并返回优化结果 opt_out
        opt_out = torch._dynamo.optimize(backend=cnt)(foo)(*args)
        # 断言优化后的输出与期望输出一致
        self.assertEqual(exp_out, opt_out)
        # 断言编译计数器的帧数为期望帧数
        self.assertEqual(cnt.frame_count, exp_frame_count)
    # 定义一个测试方法，用于测试后端匹配保护
    def test_backend_match_guard(self):
        # 创建一个形状为 [3, 4] 的随机张量 x
        x = torch.randn([3, 4])

        # 定义一个函数 foo，对输入张量 x 执行正弦和余弦运算，并返回结果
        def foo(x):
            return x.sin() + x.cos()

        # 定义一个函数 foo_graph_break，对输入张量 x 分别计算正弦和余弦，
        # 在中间调用了 torch._dynamo.graph_break()，然后返回两者相加的结果
        def foo_graph_break(x):
            a = x.sin()
            torch._dynamo.graph_break()
            b = x.cos()
            return a + b

        # 创建一个测试用的 EagerAndRecordGraphs 后端实例
        eager_record_backend = torch._dynamo.testing.EagerAndRecordGraphs()
        
        # 将后端实例和字符串 "eager" 放入 backends 列表中
        backends = [eager_record_backend, "eager"]

        # 定义一个测试函数 test_recompile，用于测试编译重构
        def test_recompile(foo, *, exp_frame_count):
            # 使用 foo 函数计算在张量 x 上的结果 eager_result
            eager_result = foo(x)
            # 枚举 backends 列表中的每个后端
            for i, backend in enumerate(backends):
                # 创建一个 CompileCounterWithBackend 对象 cnt，用于计数编译次数
                cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
                # 多次运行 _optimize_then_check_exp 方法，以确保 dynamo 不会重新编译。
                # 具体来说，期望 frame_count 不增加，缓存的后端数为 i + 2，因为有优化后端 + None
                self._optimize_then_check_exp(
                    foo, (x,), cnt, eager_result, exp_frame_count, i + 2
                )
                self._optimize_then_check_exp(
                    foo, (x,), cnt, eager_result, exp_frame_count, i + 2
                )
                self._optimize_then_check_exp(
                    foo, (x,), cnt, eager_result, exp_frame_count, i + 2
                )

        # 调用 test_recompile 方法，传入 foo 函数和期望的 frame_count
        test_recompile(foo, exp_frame_count=1)
        # 重置 dynamo
        torch._dynamo.reset()
        # 再次调用 test_recompile 方法，这次传入 foo_graph_break 函数和期望的 frame_count
        test_recompile(foo_graph_break, exp_frame_count=2)
    # 测试多线程情况下后端匹配保护功能
    def test_backend_match_guard_multi_threads(self):
        # 创建一个形状为 [3, 4] 的随机张量 x
        x = torch.randn([3, 4])

        # 定义一个函数 foo，对输入张量执行 sin 和 cos 操作后返回结果
        def foo(x):
            return x.sin() + x.cos()

        # 定义一个函数，编译并检查表达式
        def compile_then_check_exp(foo, args, cnt, eager_result, exp_frame_count):
            # 循环执行三次优化后的 foo 函数调用，并断言结果与 eager_result 相等
            for i in range(3):
                opt_out = torch._dynamo.optimize(backend=cnt)(foo)(*args)
                self.assertEqual(opt_out, eager_result)
            # 断言帧计数与期望值 exp_frame_count 相等
            self.assertEqual(cnt.frame_count, exp_frame_count)
            # 将当前线程标记为成功
            thread_success[threading.current_thread()] = True

        # 创建 EagerAndRecordGraphs 对象用于记录 eager 计算结果和图形
        eager_record_backend = torch._dynamo.testing.EagerAndRecordGraphs()
        backends = [eager_record_backend, "eager"]

        # 测试 dynamo 重新编译，但每个线程仅缓存一个后端
        eager_result = foo(x)
        # 期望的帧计数为 1
        exp_frame_count = 1
        threads = []
        thread_success = {}
        
        # 对每个后端创建一个线程，执行 compile_then_check_exp 函数
        for i, backend in enumerate(backends):
            cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
            thread = threading.Thread(
                target=compile_then_check_exp,
                args=(
                    foo,
                    (x,),
                    cnt,
                    eager_result,
                    exp_frame_count,
                ),
            )
            threads.append(thread)
            thread.start()

        # 等待所有线程执行完毕
        for thread in threads:
            thread.join()

        # 断言所有线程都成功执行
        self.assertEqual(len(thread_success), len(threads))

    # 测试带有形状信息的 dynamo 最小操作符
    def test_dynamo_min_operator_with_shape(self):
        # 使用 eager 模式优化的 f 函数定义
        @torch._dynamo.optimize("eager", nopython=True)
        def f(x, a):
            return min(x.shape[0], a)

        # 测试 f 函数，期望结果为 3
        result = f(torch.ones(6), 3)
        self.assertEqual(result, 3)

    # 测试将 ONNX 形状视为张量的情况
    def test_onnx_shape_as_tensor(self):
        # 使用 eager 模式优化的 f 函数定义
        @torch._dynamo.optimize("eager", nopython=True)
        def f(x):
            return 1 + torch._shape_as_tensor(x)[0]

        # 导出 f 函数的计算图
        gm, _ = torch._dynamo.export(f)(torch.ones(6))

        # 创建输入张量 input_one_dim 和 input_two_dims
        input_one_dim = torch.ones(6)
        input_two_dims = torch.ones(7, 4)

        # 断言 f 函数对输入 input_one_dim 和 input_two_dims 的计算结果
        self.assertEqual(f(input_one_dim), 7)
        self.assertEqual(f(input_two_dims), 8)
        self.assertEqual(f(input_two_dims), 8)

        # 使用 eager 模式优化的 f_onnx 函数定义
        @torch._dynamo.optimize("eager", nopython=True)
        def f_onnx(x):
            return 1 + torch.onnx.operators.shape_as_tensor(x)[0]

        # 断言 f_onnx 函数对输入 input_one_dim 和 input_two_dims 的计算结果
        self.assertEqual(f_onnx(input_one_dim), 7)
        self.assertEqual(f_onnx(input_two_dims), 8)
        self.assertEqual(f_onnx(input_two_dims), 8)
    # 定义一个测试方法，用于测试条件控制流函数 cond 的功能
    def test_cond(self):
        # 导入 functorch 库中的 cond 函数
        from functorch.experimental.control_flow import cond
        
        # 定义一个返回输入张量的正弦函数
        def true_fn(x):
            return x.sin()
        
        # 定义一个返回输入张量的余弦函数
        def false_fn(x):
            return x.cos()
        
        # 定义一个函数 f，根据条件 pred 调用 true_fn 或 false_fn，并传入参数 x
        def f(pred, x):
            return cond(pred, true_fn, false_fn, [x])
        
        # 使用 torch._dynamo.optimize("eager") 对 f 进行优化
        opt_fn = torch._dynamo.optimize("eager")(f)
        
        # 调用优化后的函数 opt_fn 进行测试，检查返回值是否与预期相同
        a = opt_fn(torch.tensor(False), torch.tensor([0.25, 0.25]))
        self.assertTrue(same(torch.cos(torch.tensor([0.25, 0.25])), a))
        
        b = opt_fn(torch.tensor(True), torch.tensor([0.25, 0.25]))
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), b))

    # 定义一个测试方法，用于测试带量化的条件控制流函数 cond 的功能
    def test_cond_with_quantization(self):
        # 导入 functorch 库中的 cond 函数
        from functorch.experimental.control_flow import cond
        
        # 定义一个 PyTorch 模块 MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个示例输入
                example_inputs = (torch.randn(5, 5),)
                # 定义一个线性模型
                self.model = torch.nn.Linear(5, 5)
                # 准备量化模型
                self.quantized_model = prepare_qat_fx(
                    self.model, qconfig_dict, example_inputs=example_inputs
                )
            
            # 定义前向传播方法
            def forward(self, pred, x):
                # 定义一个返回输入张量正弦函数和量化模型输出之和的函数
                def true_fn(x):
                    return x.sin() + self.quantized_model(x)
                
                # 定义一个返回输入张量余弦函数和普通模型输出之和的函数
                def false_fn(x):
                    return x.cos() + self.model(x)
                
                # 调用 cond 函数根据条件 pred 调用 true_fn 或 false_fn，并传入参数 x
                return cond(pred, true_fn, false_fn, [x])
        
        # 创建 MyModule 实例
        module = MyModule()
        # 使用 torch._dynamo.optimize("eager", nopython=True) 对模块进行优化
        opt_m = torch._dynamo.optimize("eager", nopython=True)(module)
        
        # 创建一个随机张量作为输入
        x = torch.rand((5, 5))
        
        # 测试 pred 为 True 时模块输出是否与优化后模块输出相同
        pred = torch.tensor(True)
        self.assertTrue(same(module(pred, x), opt_m(pred, x)))
        
        # 测试 pred 为 False 时模块输出是否与优化后模块输出相同
        pred = torch.tensor(False)
        self.assertTrue(same(module(pred, x), opt_m(pred, x)))

    # 定义一个测试方法，用于测试带量化的映射函数 map 的功能
    def test_map_with_quantization(self):
        # 导入 functorch 库中的 map 函数
        from functorch.experimental.control_flow import map
        
        # 定义一个 PyTorch 模块 MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个示例输入
                example_inputs = (torch.randn(5, 5),)
                # 定义一个线性模型
                self.model = torch.nn.Linear(5, 5)
                # 准备量化模型
                self.quantized_model = prepare_qat_fx(
                    self.model, qconfig_dict, example_inputs=example_inputs
                )
            
            # 定义前向传播方法
            def forward(self, x):
                # 定义一个返回输入张量正弦函数和量化模型输出之和的函数
                def body(x):
                    return x.sin() + self.quantized_model(x)
                
                # 调用 map 函数对输入张量 x 中的每个元素应用 body 函数
                return map(body, x)
        
        # 创建 MyModule 实例
        module = MyModule()
        # 使用 torch._dynamo.optimize("eager", nopython=True) 对模块进行优化
        opt_m = torch._dynamo.optimize("eager", nopython=True)(module)
        
        # 创建一个随机张量作为输入
        x = torch.rand((5, 5))
        
        # 测试模块输出是否与优化后模块输出相同
        self.assertTrue(same(module(x), opt_m(x)))
    def test_cond_nested(self):
        # 导入控制流模块中的条件函数
        from functorch.experimental.control_flow import cond

        # 定义在条件为真时执行的嵌套函数
        def true_fn_nested(x):
            return x * 10

        # 定义在条件为假时执行的嵌套函数
        def false_fn_nested(x):
            return x * -1

        # 定义在条件为真时执行的函数
        def true_fn(pred2, x):
            return x.sin()

        # 定义在条件为假时执行的函数
        def false_fn(pred2, x):
            # 根据嵌套的条件调用函数来处理 x
            return x + cond(pred2, true_fn_nested, false_fn_nested, [x])

        # 定义主要函数 f，根据条件 pred 和 pred2 执行不同的函数
        def f(pred, pred2, x):
            return cond(pred, true_fn, false_fn, [pred2, x])

        # 创建一个编译计数器实例
        cc = torch._dynamo.testing.CompileCounter()
        # 使用编译计数器优化函数 f
        opt_fn = torch._dynamo.optimize(cc)(f)

        # 测试条件为真、pred2 为真时的结果
        true_true_sin = opt_fn(
            torch.tensor(True), torch.tensor(True), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), true_true_sin))

        # 测试条件为真、pred2 为假时的结果
        true_false_sin = opt_fn(
            torch.tensor(True), torch.tensor(False), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), true_false_sin))

        # 测试条件为假、pred2 为真时的结果
        false_true_sum_mult = opt_fn(
            torch.tensor(False), torch.tensor(True), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(
            same(torch.tensor([2.75, 2.75]), false_true_sum_mult)
        )  # * 10 then add x

        # 测试条件为假、pred2 为假时的结果
        false_false_sum_neg = opt_fn(
            torch.tensor(False), torch.tensor(False), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(
            same(torch.tensor([0.0, 0.0]), false_false_sum_neg)
        )  # * -1 then add x

        # 断言编译计数器的帧数为 2
        self.assertTrue(cc.frame_count, 2)
    def test_cond_export(self):
        from functorch.experimental.control_flow import cond  # 导入条件函数cond

        def true_fn_nested(x):
            return x * 10  # 返回输入x乘以10的结果

        def false_fn_nested(x):
            return x * -1  # 返回输入x乘以-1的结果

        def true_fn(pred2, x):
            return x.sin()  # 返回输入x的正弦值

        def false_fn(pred2, x):
            return x + cond(pred2, true_fn_nested, false_fn_nested, [x])  # 返回根据pred2条件选择不同函数结果的计算

        def f(pred, pred2, x):
            return cond(pred, true_fn, false_fn, [pred2, x])  # 根据pred条件选择不同函数结果的计算

        graph, guard = torch._dynamo.export(f)(  # 调用torch._dynamo.export导出的图和保护器
            torch.tensor(False), torch.tensor(True), torch.tensor([0.25, 0.25])  # 提供给函数f的输入参数
        )
        true_true_sin = graph(  # 使用图计算函数结果
            torch.tensor(True), torch.tensor(True), torch.tensor([0.25, 0.25])  # 提供给函数f的输入参数
        )
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), true_true_sin))  # 断言函数计算结果与预期一致

        true_false_sin = graph(  # 使用图计算函数结果
            torch.tensor(True), torch.tensor(False), torch.tensor([0.25, 0.25])  # 提供给函数f的输入参数
        )
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), true_false_sin))  # 断言函数计算结果与预期一致

        false_true_sum_mult = graph(  # 使用图计算函数结果
            torch.tensor(False), torch.tensor(True), torch.tensor([0.25, 0.25])  # 提供给函数f的输入参数
        )
        self.assertTrue(
            same(torch.tensor([2.75, 2.75]), false_true_sum_mult)
        )  # 断言函数计算结果与预期一致，这里是乘以10然后加上x

        false_false_sum_neg = graph(  # 使用图计算函数结果
            torch.tensor(False), torch.tensor(False), torch.tensor([0.25, 0.25])  # 提供给函数f的输入参数
        )
        self.assertTrue(
            same(torch.tensor([0.0, 0.0]), false_false_sum_neg)
        )  # 断言函数计算结果与预期一致，这里是乘以-1然后加上x

    def test_cond_export_single_arg(self):
        from functorch.experimental.control_flow import cond  # 导入条件函数cond

        def true_fn(x):
            return x  # 返回输入x本身

        def false_fn(x):
            return x.sin()  # 返回输入x的正弦值

        def f(pred, x):
            return cond(pred, true_fn, false_fn, [x])  # 根据pred条件选择不同函数结果的计算

        graph, guard = torch._dynamo.export(f)(  # 调用torch._dynamo.export导出的图和保护器
            torch.tensor(False), torch.tensor([0.25, 0.25])  # 提供给函数f的输入参数
        )
        true_mirror = graph(torch.tensor(True), torch.tensor([0.25, 0.25]))  # 使用图计算函数结果
        self.assertTrue(same(torch.tensor([0.25, 0.25]), true_mirror))  # 断言函数计算结果与预期一致
        true_mirror_2 = graph(torch.tensor(True), torch.tensor([0.33, 0.33, 0.33]))  # 使用图计算函数结果
        self.assertTrue(same(torch.tensor([0.33, 0.33, 0.33]), true_mirror_2))  # 断言函数计算结果与预期一致

        false_sin = graph(torch.tensor(False), torch.tensor([0.5, 0.5]))  # 使用图计算函数结果
        self.assertTrue(same(torch.sin(torch.tensor([0.5, 0.5])), false_sin))  # 断言函数计算结果与预期一致

    def test_enum_guards(self):
        class MyEnum(enum.Enum):  # 定义枚举类型MyEnum
            FOO = 10  # 枚举成员FOO的值为10
            BAR = 20  # 枚举成员BAR的值为20

        def fn(x, y):
            if y == MyEnum.FOO:  # 如果y等于枚举类型MyEnum的成员FOO
                return x + 1  # 返回x加1的结果
            else:
                return x - 1  # 返回x减1的结果

        x = torch.rand(3)  # 创建一个形状为(3,)的随机张量x
        y = MyEnum.BAR  # 将y设为枚举类型MyEnum的成员BAR
        ref = fn(x, y)  # 调用函数fn计算结果并将其保存为ref
        opt_fn = torch.compile(backend="eager")(fn)  # 编译函数fn并返回优化后的函数opt_fn
        res = opt_fn(x, y)  # 使用优化后的函数opt_fn计算结果
        self.assertTrue(same(ref, res))  # 断言函数计算结果与预期一致
    # 定义测试函数，检验图中断日志功能
    def test_duplicate_graph_break_log(self):
        # 开启图中断日志
        torch._logging.set_logs(graph_breaks=True)

        # 使用装饰器优化函数 f1
        @torch._dynamo.optimize("eager")
        def f1(a, b):
            # 调用函数 f2
            f2(a, b)

        # 定义函数 f2
        def f2(a, b):
            # 计算参数 a 和 b 的和
            c = a + b
            # 打印中断信息
            print("break")
            # 返回 a + b + c 的结果
            return a + b + c

        # 使用装饰器优化函数 g1
        @torch._dynamo.optimize("eager")
        def g1(a, b):
            # 调用函数 g2
            g2(a, b)

        # 定义函数 g2
        def g2(a, b):
            # 计算参数 a 和 b 的和
            c = a + b
            # 打印中断信息
            print("break")
            # 返回 a + b + c 的结果
            return a + b + c

        # 定义函数，用于统计图中断消息的数量
        def count_graph_break_msgs(msgs):
            # 返回包含 "Graph break" 的消息数量
            return sum(msg.find("Graph break") != -1 for msg in msgs)

        # 使用 assertLogs 上下文，检查 torch._dynamo 日志的 DEBUG 级别消息
        with self.assertLogs(
            logger="torch._dynamo", level=logging.DEBUG
        ) as log, torch._dynamo.config.patch(verbose=True):
            # 调用优化后的函数 f1，并检查日志中的图中断消息数量是否大于 1
            f1(torch.randn(10), torch.randn(10))
            self.assertGreater(count_graph_break_msgs(log.output), 1)

        # 使用 assertLogs 上下文，再次检查 torch._dynamo 日志的 DEBUG 级别消息
        with self.assertLogs(
            logger="torch._dynamo", level=logging.DEBUG
        ) as log, torch._dynamo.config.patch(verbose=False):
            # 调用优化后的函数 g1，并检查日志中的图中断消息数量是否等于 1
            g1(torch.randn(10), torch.randn(10))
            self.assertEqual(count_graph_break_msgs(log.output), 1)

        # 重置日志状态
        torch._logging.set_logs()

    # 定义测试函数，检验原地参数更新功能
    def test_inplace_param_update(self):
        # 定义函数 fn，接受参数 param 和 y
        def fn(param, y):
            # 保存之前的梯度状态
            prev_grad = torch.is_grad_enabled()
            try:
                # 关闭梯度计算
                torch.set_grad_enabled(False)
                # 打开梯度计算
                torch.set_grad_enabled(True)
                # 再次关闭梯度计算
                torch.set_grad_enabled(False)
                # 原地更新参数 param
                param.add_(y)
            finally:
                # 恢复之前的梯度状态
                torch.set_grad_enabled(prev_grad)

        # 创建随机张量 y
        y = torch.randn(4)
        # 创建参数张量 x
        x = torch.nn.Parameter(torch.randn(4))
        # 调用函数 fn，进行原地参数更新
        fn(x, y)

        # 创建编译计数器 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用装饰器优化函数 fn，并设置为无 Python 模式
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 调用优化后的函数 opt_fn，进行原地参数更新
        opt_fn(x, y)
        # 断言编译帧数为 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言操作计数为 3
        self.assertEqual(cnts.op_count, 3)

    # 如果不支持闪烁注意力机制，则跳过测试
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Can't run fused SDPA on this platform",
    )
    def test_parsing_sdpa(self):
        # 定义一个测试函数，用于测试 scaled_dot_product_attention 函数的多个调用情况
        class MyModule(torch.nn.Module):
            def forward(self, query, key, value):
                # 调用 scaled_dot_product_attention 函数，执行注意力计算
                out = F.scaled_dot_product_attention(query, key, value, None, 0, True)
                # 再次调用 scaled_dot_product_attention 函数，这次设置了 scale 参数为 8
                out = F.scaled_dot_product_attention(
                    query, key, value, None, 0, True, scale=8
                )
                # 第三次调用 scaled_dot_product_attention 函数，传入了具名参数
                out = F.scaled_dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=None,
                    dropout_p=0,
                    is_causal=True,
                )
                # 第四次调用 scaled_dot_product_attention 函数，传入了具名参数，格式略有不同
                out = F.scaled_dot_product_attention(
                    query,
                    key=key,
                    value=value,
                    attn_mask=None,
                    dropout_p=0,
                    is_causal=True,
                )
                # 第五次调用 scaled_dot_product_attention 函数，只传入了部分具名参数
                out = F.scaled_dot_product_attention(
                    query, key, value, None, dropout_p=0, is_causal=True
                )
                # 最后一次调用 scaled_dot_product_attention 函数，设置了 scale 参数为 8
                out = F.scaled_dot_product_attention(query, key, value, None, scale=8)
                return out

        # 设置测试所需的设备和数据类型等参数
        device = "cuda"
        dtype = torch.float16
        seq_len_q = 1
        seq_len_k = 1
        head_dim = 8
        # 创建输入数据张量
        query = torch.ones(
            1, 8, seq_len_q, head_dim, device=device, dtype=dtype, requires_grad=True
        )
        key = torch.ones(
            1, 8, seq_len_k, head_dim, device=device, dtype=dtype, requires_grad=True
        )
        value = torch.ones(
            1, 8, seq_len_k, head_dim, device=device, dtype=dtype, requires_grad=True
        )
        # 创建 MyModule 实例
        module = MyModule()
        # 优化 MyModule 实例
        opt_mod = torch._dynamo.optimize("inductor")(module)
        # 执行优化后的模块
        opt_mod(query, key, value)

    def test_generate_tensor_from_list_of_numpy_primitive_type(self):
        # 定义一个测试函数，测试从包含 numpy 原始类型数据的列表生成 torch.LongTensor
        def fn():
            # 创建 numpy 数组
            x = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
            # 创建包含 numpy 数组元素的列表
            y = [x[0], x[2], x[4]]
            # 返回转换后的 torch.LongTensor
            return torch.LongTensor(y)

        # 调用函数获取参考结果
        ref = fn()
        # 编译函数并执行，获取结果
        res = torch.compile(fullgraph=True)(fn)()
        # 断言参考结果和编译结果相等
        self.assertEqual(ref, res)

    def test_object_classmethod(self):
        # 定义一个测试类和类方法
        class C:
            @classmethod
            def fn(cls, x):
                # 类方法对输入参数执行加法运算
                return x + x

        # 使用 torch._dynamo.optimize 进行优化的函数
        @torch._dynamo.optimize("eager", nopython=True)
        def f():
            # 调用类的实例方法 fn，并传入参数 torch.ones(2, 3)
            return C().fn(torch.ones(2, 3))

        # 断言优化后的函数结果与预期的 torch.tensor([2.0]) 相近
        self.assertTrue(torch.allclose(f(), torch.tensor([2.0])))

    def test_object_staticmethod(self):
        # 定义一个测试类和静态方法
        class C:
            @staticmethod
            def fn(x):
                # 静态方法对输入参数执行加法运算
                return x + x

        # 使用 torch._dynamo.optimize 进行优化的函数
        @torch._dynamo.optimize("eager", nopython=True)
        def f():
            # 调用类的静态方法 fn，并传入参数 torch.ones(2, 3)
            return C().fn(torch.ones(2, 3))

        # 断言优化后的函数结果与预期的 torch.tensor([2.0]) 相近
        self.assertTrue(torch.allclose(f(), torch.tensor([2.0])))
    def test_user_function_variable_supports_enum_argument(self):
        # 定义一个枚举类 Foo，包含 FOO 和 BAR 两个枚举值
        class Foo(enum.Enum):
            FOO = 0
            BAR = 1

        # 定义函数 gn，接受两个参数 x 和 y（默认为 Foo.FOO 枚举值）
        def gn(x, y=Foo.FOO):
            # 如果 y 等于 Foo.FOO，则返回 x
            if y is Foo.FOO:
                return x
            else:
                # 否则返回 x + 1
                return x + 1

        # 定义函数 fn，接受一个参数 x，并调用 gn 函数
        def fn(x):
            return gn(x)

        # 生成一个 2x3 的随机张量 x
        x = torch.randn(2, 3)
        # 调用 fn 函数，保存结果到 ref
        ref = fn(x)
        # 使用 torch._dynamo.optimize 对 fn 函数进行优化，并保存为 opt_fn
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 使用优化后的函数 opt_fn 处理张量 x，保存结果到 res
        res = opt_fn(x)
        # 断言 ref 和 res 的所有元素近似相等
        self.assertTrue(torch.allclose(ref, res))

    def test_user_function_variable_supports_type_abcmeta_argument(self):
        # 定义抽象基类 Foo，其下有一个抽象方法 read
        class Foo(metaclass=abc.ABCMeta):
            @abc.abstractclassmethod
            def read(self):  # noqa: B027
                pass

        # 定义子类 Bar，实现了抽象方法 read
        class Bar(Foo):
            def read(self):
                return "Hello World!"

        # 定义空类 Baz
        class Baz:
            pass

        # 定义函数 gn，接受一个参数 x 和一个元组类型的 tys，默认值为 (Bar, Baz)
        def gn(x, tys=(Bar, Baz)):
            # 如果 Bar 在 tys 中，则返回 x - 1
            if Bar in tys:
                return x - 1
            else:
                # 否则返回 x + 1
                return x + 1

        # 定义函数 fn，接受一个参数 x，并调用 gn 函数
        def fn(x):
            return gn(x)

        # 生成一个 2x3 的随机张量 x
        x = torch.randn(2, 3)
        # 调用 fn 函数，保存结果到 ref
        ref = fn(x)
        # 使用 torch._dynamo.optimize 对 fn 函数进行优化，并保存为 opt_fn
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 使用优化后的函数 opt_fn 处理张量 x，保存结果到 res
        res = opt_fn(x)
        # 断言 ref 和 res 的所有元素近似相等
        self.assertTrue(torch.allclose(ref, res))

    def test_user_function_variable_supports_function_argument(self):
        # 定义函数 add1，接受一个参数 x，返回 x + 1
        def add1(x):
            return x + 1

        # 定义函数 gn，接受一个参数 x 和三个函数类型的参数 f1、f2、f3，
        # 默认分别为 add1、torch.sin、operator.neg
        def gn(x, f1=add1, f2=torch.sin, f3=operator.neg):
            # 返回对 x 进行 f1、f2、f3 函数嵌套处理后的结果
            return f3(f2(f1(x)))

        # 定义函数 fn，接受一个参数 x，并调用 gn 函数
        def fn(x):
            return gn(x)

        # 生成一个 2x3 的随机张量 x
        x = torch.randn(2, 3)
        # 调用 fn 函数，保存结果到 ref
        ref = fn(x)
        # 使用 torch._dynamo.optimize 对 fn 函数进行优化，并保存为 opt_fn
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 使用优化后的函数 opt_fn 处理张量 x，保存结果到 res
        res = opt_fn(x)
        # 断言 ref 和 res 的所有元素近似相等
        self.assertTrue(torch.allclose(ref, res))

    def test_typing_variable_isinstance(self):
        # 定义函数 fn，接受两个参数 x 和 m
        def fn(x, m):
            # 如果 m 是 typing.Mapping 类型，则返回 x + 1
            if isinstance(m, typing.Mapping):
                return x + 1
            else:
                # 否则返回 x - 1
                return x - 1

        # 生成一个 2x3 的随机张量 x
        x = torch.randn(2, 3)
        # 生成一个包含键 "x" 的字典 m，值为一个 3 元素的随机张量
        m = {"x": torch.randn(3)}
        # 调用 fn 函数，保存结果到 ref
        ref = fn(x, m)
        # 使用 torch._dynamo.optimize 对 fn 函数进行优化，并保存为 opt_fn
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 使用优化后的函数 opt_fn 处理张量 x 和字典 m，保存结果到 res
        res = opt_fn(x, m)
        # 断言 ref 和 res 的所有元素近似相等
        self.assertTrue(torch.allclose(ref, res))

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_repro_graph_breaks_in__get_item_by_idx(self):
        # 定义类 Mod，继承自 torch.nn.Module
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义模块 self.mod，包含两个线性层 torch.nn.Linear
                self.mod = torch.nn.Sequential(
                    torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)
                )

            # 定义前向传播方法，返回 self.mod 的第一个线性层对输入 x 的计算结果
            def forward(self, x):
                return self.mod[0](x)

        # 创建 Mod 类的实例 m
        m = Mod()
        # 调用 torch._dynamo.export 导出模型 m 的计算图和其他信息
        graph, _ = torch._dynamo.export(m)(torch.randn(3, 3))
    # 定义一个测试方法，用于测试 nn.Sequential 的调用
    def test_nn_sequential_invocation(self):
        # 冻结随机数生成器状态，确保测试结果可重复
        with freeze_rng_state():

            # 定义一个测试用的神经网络模型
            class TestModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    # 使用 nn.Sequential 定义几个线性层组成的模型
                    self.linears = torch.nn.Sequential(
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                    )

                # 前向传播方法，接受输入 x，返回前三个线性层的输出
                def forward(self, x):
                    all_but_last = self.linears[:-1]
                    return all_but_last(x)

            # 创建 TestModel 的实例
            m = TestModel()
            # 创建随机输入张量
            x = torch.rand((2, 2))
            # 使用实例 m 进行前向传播
            real = m(x)
            # 导出模型并执行计算图
            graph, _ = torch._dynamo.export(m)(x)
            dynamo_result = graph(x)
            # 断言实际输出与动态图结果相同
            self.assertTrue(same(real, dynamo_result))

    # 使用装饰器配置 nn 模块的保护，在特定上下文中进行测试
    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_nn_sequential_invocation_reposition_indices(self):
        # 冻结随机数生成器状态，确保测试结果可重复
        with freeze_rng_state():

            # 定义一个测试用的神经网络模型
            class TestModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    # 使用 nn.Sequential 定义几个线性层组成的模型
                    self.linears = torch.nn.Sequential(
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                    )

                # 前向传播方法，接受输入 x，返回第二个到第四个线性层的输出
                def forward(self, x):
                    all_but_last = self.linears[1:3]
                    return all_but_last(x)

            # 创建 TestModel 的实例
            m = TestModel()
            # 创建随机输入张量
            x = torch.rand((2, 2))
            # 使用实例 m 进行前向传播
            real = m(x)
            # 导出模型并执行计算图
            graph, _ = torch._dynamo.export(m)(x)
            dynamo_result = graph(x)
            # 断言实际输出与动态图结果相同
            self.assertTrue(same(real, dynamo_result))

    # 测试在嵌套 FX 跟踪时是否会抛出错误
    def test_error_on_nested_fx_trace(self):
        # 创建随机输入张量
        input = torch.rand(2, 3)

        # 定义一个简单的函数 f，对输入 x 执行加法操作
        def f(x):
            x + x

        # 计算真实结果
        real = f(input)

        # 对函数 f 进行优化，以提高性能
        optimized = torch._dynamo.optimize("eager")(f)
        # 断言优化后的结果与真实结果相同
        self.assertTrue(same(optimized(input), real))

        # 断言在使用 FX 进行符号跟踪时会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Detected that you are using FX"):
            gm = torch.fx.symbolic_trace(optimized)

    # 在不应抛出嵌套 FX 跟踪错误时进行测试
    @patch.object(torch._dynamo.config, "error_on_nested_fx_trace", False)
    def test_no_error_on_nested_fx_trace(self):
        # 创建随机输入张量
        input = torch.rand(2, 3)

        # 定义一个简单的函数 f，对输入 x 执行加法操作
        def f(x):
            x + x

        # 计算真实结果
        real = f(input)

        # 对函数 f 进行优化，以提高性能
        optimized = torch._dynamo.optimize("eager")(f)
        # 断言优化后的结果与真实结果相同
        self.assertTrue(same(optimized(input), real))

        # 在不应抛出异常的情况下，使用 FX 进行符号跟踪
        gm = torch.fx.symbolic_trace(optimized)
        self.assertTrue(same(gm(input), real))

    # 测试在静态作用域下函数行为
    def test_not_dynamic_scope(self):
        # 定义一个函数 f，其中包含一个嵌套函数 g
        def f(y):
            x = 1

            def g():
                x = 2
                return lambda: x

            return y + g()()

        # 创建一个零张量作为输入
        input = torch.zeros(1)
        # 计算真实结果
        real = f(input)
        # 对函数 f 进行优化，以提高性能
        optimized = torch._dynamo.optimize("eager")(f)
        # 计算优化后的结果
        opt = optimized(input)
        # 断言优化后的结果与真实结果相同
        self.assertTrue(same(opt, real))
    # 定义一个测试函数，用于测试推断模式下的函数功能
    def test_inference_mode(self):
        # 使用装饰器将 func 函数设置为推断模式
        @torch.inference_mode()
        def func(x, y):
            # 执行张量操作：x 加 1.0，然后加 y
            return x.add(1.0) + y

        # 创建两个需要梯度的张量 x 和 y，内容为全 1
        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=True)
        # 调用 func 函数，得到参考结果 ref
        ref = func(x, y)
        # 使用 eager 模式优化 func 函数
        opt_func = torch._dynamo.optimize("eager")(func)

        # 创建新的张量 x1，内容同样为全 1
        x1 = torch.ones(4, requires_grad=True)
        # 调用优化后的 func 函数，得到结果 res
        res = opt_func(x1, y)
        # 断言优化前后的结果应该相同
        self.assertTrue(same(ref, res))
        # 断言原始张量 x 和 x1 应该相同
        self.assertTrue(same(x, x1))

    # 定义一个测试函数，测试带条件语句的神经网络模块
    def test_if_cond_nn_mod1(self):
        # 定义一个 MockModule 类，继承自 torch.nn.Module
        class MockModule(torch.nn.Module):
            # 初始化方法
            def __init__(self, output_relu=True):
                super().__init__()
                # 根据 output_relu 参数决定是否添加 ReLU 层
                self.relu = torch.nn.ReLU() if output_relu else None

            # 前向传播方法
            def forward(self, x):
                # 对输入张量 x 执行 sin 函数
                x = torch.sin(x)
                # 如果存在 self.relu，则执行 ReLU 激活函数
                if self.relu:
                    x = self.relu(x)
                return x

        # 创建一个 MockModule 实例 model
        model = MockModule()
        # 使用 eager 模式优化 model
        opt_model = torch._dynamo.optimize("eager", nopython=True)(model)

        # 创建一个随机张量 x
        x = torch.rand(4)
        # 分别使用原始模型和优化模型计算结果
        ref = model(x)
        res = opt_model(x)
        # 断言两者结果应该相同
        self.assertTrue(same(ref, res))

        # 创建一个 MockModule 实例 model，禁用输出层的 ReLU
        model = MockModule(output_relu=False)
        # 使用 eager 模式优化 model
        opt_model = torch._dynamo.optimize("eager", nopython=True)(model)

        # 创建一个随机张量 x
        x = torch.rand(4)
        # 分别使用原始模型和优化模型计算结果
        ref = model(x)
        res = opt_model(x)
        # 断言两者结果应该相同
        self.assertTrue(same(ref, res))

    # 定义一个测试函数，测试带条件语句的神经网络模块
    def test_if_cond_nn_mod2(self):
        # 定义一个 MockModule 类，继承自 torch.nn.Module
        class MockModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 定义一个空的 Sequential 层
                self.layer = torch.nn.Sequential()

            # 前向传播方法
            def forward(self, x):
                # 如果 self.layer 不为空，则返回 x + 1；否则返回 x - 1
                if self.layer:
                    return x + 1
                else:
                    return x - 1

        # 创建一个 MockModule 实例 model
        model = MockModule()
        # 创建一个随机张量 x
        x = torch.rand(4)
        # 计算原始模型的结果 ref
        ref = model(x)
        # 使用 eager 模式编译优化模型
        opt_model = torch.compile(backend="eager")(model)
        # 计算优化模型的结果 res
        res = opt_model(x)
        # 断言两者结果应该相同
        self.assertTrue(same(ref, res))

    # 定义一个测试函数，测试带条件语句的函数
    def test_if_cond_nn_mod3(self):
        # 定义一个简单的函数 fn，带有条件语句
        def fn(x):
            # 如果 torch.nn.ModuleList() 存在，则返回 x + 1；否则返回 x - 1
            if torch.nn.ModuleList():
                return x + 1
            else:
                return x - 1

        # 创建一个随机张量 x
        x = torch.rand(4)
        # 计算原始函数的结果 ref
        ref = fn(x)
        # 使用 eager 模式编译优化函数
        opt_fn = torch.compile(backend="eager")(fn)
        # 计算优化函数的结果 res
        res = opt_fn(x)
        # 断言两者结果应该相同
        self.assertTrue(same(ref, res))
    def test_if_cond_user_defined_object(self):
        # 定义类 A，没有定义 __bool__ 方法
        class A:  # noqa: B903
            def __init__(self, x):
                self.x = x

        # 定义类 B，定义了 __bool__ 方法，并返回布尔类型
        class B:
            def __init__(self, x):
                self.x = x

            def __bool__(self):
                return self.x > 0

        # 定义类 C，错误地将 __bool__ 属性设置为 False 而不是方法
        class C:
            def __init__(self, x):
                self.x = x
                self.__bool__ = False

        # 定义函数 fn，根据 obj 的布尔值执行不同的操作
        def fn(x, obj):
            if not obj:
                return x + 1
            else:
                return x - 1

        # 创建一个 torch 的随机张量 x
        x = torch.rand(4)
        # 创建一个 CompileCounter 实例 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 通过装饰器优化 fn 函数
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 创建不同类型的对象实例 obj1, obj2, obj3, obj4
        obj1 = A(0.5)
        obj2 = B(0.5)
        obj3 = B(-0.5)
        obj4 = C(0.5)
        # 对于每个对象进行测试
        for obj in [obj1, obj2, obj3, obj4, obj3, obj2]:
            # 计算原始函数的结果 ref
            ref = fn(x, obj)
            # 计算优化后函数的结果 res
            res = opt_fn(x, obj)
            # 断言结果相同
            self.assertTrue(same(ref, res))
        # 断言函数调用的帧数为 4
        self.assertEqual(cnts.frame_count, 4)

    def test_if_cond_user_defined_object2(self):
        # 定义类 MyObj，定义了 __bool__ 方法，但返回的是非布尔类型
        class MyObj:
            def __init__(self, x):
                self.x = x

            def __bool__(self):
                self.x = 1.2
                return self.x

        # 定义函数 fn，根据 obj 的布尔值执行不同的操作
        def fn(a, obj):
            if not obj:
                return a + obj.x
            else:
                return a - obj.x

        # 创建一个 torch 的随机张量 x
        x = torch.rand(4)
        # 创建 MyObj 的实例 obj
        obj = MyObj(0.5)
        # 使用 "eager" 模式编译优化 fn 函数
        opt_fn = torch._dynamo.optimize("eager")(fn)
        try:
            # 尝试优化后函数的执行
            opt_fn(x, obj)
            # 如果执行成功，断言为假
            self.assertFalse(True)
        except TypeError as e:
            # 如果出现 TypeError，断言异常信息中包含指定字符串
            self.assertIn("__bool__ should return bool, returned float", str(e))

    def test_if_cond_user_defined_object3(self):
        # 定义类 A，没有定义 __bool__ 方法，但定义了 __len__ 方法
        class A:  # noqa: B903
            def __init__(self, x):
                self.x = x

            def __len__(self):
                return len(self.x)

        # 定义类 B，定义了 __bool__ 和 __len__ 方法
        class B:
            def __init__(self, x):
                self.x = x

            def __bool__(self):
                return False

            def __len__(self):
                return len(self.x)

        # 定义函数 fn，根据 obj 的布尔值执行不同的操作
        def fn(x, obj):
            if not obj:
                return x + 1
            else:
                return x - 1

        # 创建一个 torch 的随机张量 x
        x = torch.rand(4)
        # 使用 "eager" 模式编译优化 fn 函数
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        # 创建不同类型的对象实例 obj1, obj2, obj3, obj4
        obj1 = A([1, 2, 3])
        obj2 = A([])
        obj3 = B([1, 2, 3])
        obj4 = B([])
        # 对于每个对象进行测试
        for obj in [obj1, obj2, obj3, obj4]:
            # 计算原始函数的结果 ref
            ref = fn(x, obj)
            # 计算优化后函数的结果 res
            res = opt_fn(x, obj)
            # 断言结果相同
            self.assertTrue(same(ref, res))
    def test_class_has_instancecheck_method(self):
        # 定义一个类 A
        class A:
            pass

        # 定义一个元类 ExampleMeta，其中包含 __instancecheck__ 方法，始终返回 True
        class ExampleMeta(type):
            def __instancecheck__(cls, instance):
                return True

        # 定义一个类 B，使用 ExampleMeta 作为元类
        class B(metaclass=ExampleMeta):
            pass

        # 定义一个函数 fn，接受参数 x 和 obj
        def fn(x, obj):
            # 检查 obj 是否为类 B 的实例
            if isinstance(obj, B):
                return x + 1
            else:
                return x - 1

        # 创建一个包含随机数的 Tensor
        x = torch.rand(4)
        # 创建一个 A 类的实例
        obj = A()
        # 调用 fn 函数，获取结果 ref
        ref = fn(x, obj)
        # 对 fn 函数进行优化，生成优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 使用优化后的函数 opt_fn 处理数据 x 和 obj，获取结果 res
        res = opt_fn(x, obj)
        # 断言优化前后结果一致
        self.assertTrue(same(ref, res))

    def test_torch_cuda_is_available(self):
        # 定义一个函数 fn，接受参数 x
        def fn(x):
            # 检查 CUDA 是否可用，若可用则返回 x + 1，否则返回 x - 1
            if torch.cuda.is_available():
                return x + 1
            else:
                return x - 1

        # 创建一个包含随机数的 Tensor
        x = torch.rand(4)
        # 调用 fn 函数，获取结果 ref
        ref = fn(x)
        # 对 fn 函数进行优化，生成优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 使用优化后的函数 opt_fn 处理数据 x，获取结果 res
        res = opt_fn(x)
        # 断言优化前后结果一致
        self.assertTrue(same(ref, res))

    def test_variable_tracker_recursively_contains(self):
        # 定义一个函数 fn，接受参数 x
        # 创建一个二维列表 data，初始化为 None，包含 3 行 3 列
        def fn(x):
            data = [[None] * 3] * 3
            # 循环遍历 range(3)
            for i in range(3):
                # 若 i 为 0，则将 data[0][i] 设为 x，否则设为前一个元素加 1
                if i == 0:
                    data[0][i] = x
                else:
                    data[0][i] = data[0][i - 1] + 1
            # 返回 data[0][-1]，即最后一个元素
            return data[0][-1]

        # 创建一个包含随机数的 Tensor
        x = torch.rand(4)
        # 调用 fn 函数，获取结果 ref
        ref = fn(x)
        # 对 fn 函数进行优化，生成优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 使用优化后的函数 opt_fn 处理数据 x，获取结果 res
        res = opt_fn(x)
        # 断言优化前后结果一致
        self.assertTrue(same(ref, res))

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    @unittest.skipIf(not torch.backends.cudnn.is_available(), "requires cudnn")
    def test_torch_cudnn_is_acceptable(self):
        # 定义一个函数 fn，接受参数 x
        def fn(x):
            # 检查 cudnn 是否接受张量 x，若是则返回 x + 1，否则返回 x
            if torch.backends.cudnn.is_acceptable(tensor=x):
                return x + 1
            return x

        # 创建一个包含随机数的 Tensor，并将其移到 CUDA 设备上
        x = torch.rand(4).cuda()
        # 调用 fn 函数，获取结果 ref
        ref = fn(x)
        # 对 fn 函数进行优化，生成优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 使用优化后的函数 opt_fn 处理数据 x，获取结果 res
        res = opt_fn(x)
        # 断言优化前后结果一致
        self.assertTrue(same(ref, res))
    def test_torch_cudnn_is_acceptable_bad_inputs(self):
        # 定义内部函数 fn1，检查给定参数是否有效，若有效则返回 x+1，否则返回 x
        def fn1(x):
            if torch.backends.cudnn.is_acceptable("invalid"):
                return x + 1
            return x

        # 定义内部函数 fn2，检查给定参数是否有效，若有效则返回 x+1，否则返回 x
        def fn2(x):
            if torch.backends.cudnn.is_acceptable(x, 3.14):
                return x + 1
            return x

        # 使用断言检查 opt_fn1 函数是否抛出预期的异常消息
        with self.assertRaisesRegex(
            AssertionError, "Expect input to cudnn.is_acceptable to be a tensor"
        ):
            # 创建 cuda 上的随机张量 x1
            x1 = torch.rand(4).cuda()
            # 优化 fn1 函数
            opt_fn1 = torch._dynamo.optimize("eager", nopython=True)(fn1)
            # 调用优化后的 fn1 函数
            res1 = opt_fn1(x1)

        # 使用断言检查 opt_fn2 函数是否抛出预期的异常消息
        with self.assertRaisesRegex(
            AssertionError, "Expect 1 input to cudnn.is_acceptable"
        ):
            # 创建 cuda 上的随机张量 x2
            x2 = torch.rand(4).cuda()
            # 优化 fn2 函数
            opt_fn2 = torch._dynamo.optimize("eager", nopython=True)(fn2)
            # 调用优化后的 fn2 函数
            res = opt_fn2(x2)

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    def test_get_device(self):
        # 定义函数 fn，对输入张量 x 和 y 分别加一，并返回它们的设备编号
        def fn(x, y):
            x = x + 1
            y = y + 1
            return x.get_device(), y.get_device()

        # 创建一个在 cuda 上的随机张量 x 和一个在 cpu 上的随机张量 y
        x = torch.rand(4, device="cuda")
        y = torch.rand(4, device="cpu")
        # 计算参考结果
        ref = fn(x, y)
        # 优化 fn 函数
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 调用优化后的 fn 函数
        res = opt_fn(x, y)
        # 断言优化后的结果与参考结果相同
        self.assertTrue(same(ref, res))

    def test_disable_flag(self):
        # 创建编译计数器实例 cnt
        cnt = torch._dynamo.testing.CompileCounter()

        # 使用 patch.dict 设置环境变量 TORCH_COMPILE_DISABLE 为 "1"
        with patch.dict(os.environ, {"TORCH_COMPILE_DISABLE": "1"}):
            # 定义函数 fn，对输入张量 x 和 y 分别加一
            def fn(x, y):
                x = x + 1
                y = y + 1

            # 优化 fn 函数
            opt_fn = torch._dynamo.optimize(cnt)

        # 断言编译帧数为 0
        self.assertEqual(cnt.frame_count, 0)

    def test_is_compiling(self):
        # 定义函数 f1，检查是否正在使用动态编译器编译，返回全一张量或全零张量
        def f1():
            if torch._dynamo.is_compiling():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        # 定义函数 f2，检查是否正在编译，返回全一张量或全零张量
        def f2():
            if torch._utils.is_compiling():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        # 定义函数 f3，检查是否使用编译器编译，返回全一张量或全零张量
        def f3():
            if torch.compiler.is_compiling():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        # 定义函数 f4，检查是否使用动态编译器编译，返回全一张量或全零张量
        def f4():
            if torch.compiler.is_dynamo_compiling():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        # 遍历函数列表，对每个函数进行优化并断言其返回结果为全零张量
        for f in [f1, f2, f3, f4]:
            opt_f = torch._dynamo.optimize("eager")(f)

            self.assertEqual(f(), torch.zeros(2, 2))
            self.assertEqual(opt_f(), torch.ones(2, 2))
    # 定义一个测试函数，用于测试设置和恢复随机数生成器状态的功能
    def test_torch_generator_set_state(self):
        # 定义一个内部函数fn，用于生成随机数并操作默认随机数生成器的状态
        def fn():
            # 获取默认随机数生成器的当前状态
            default_state = torch.default_generator.get_state()
            # 生成一个随机数张量
            x = torch.rand([2, 3])
            # 如果默认状态的数据类型不是"float32"，则将x的值乘以2
            if default_state.dtype != "float32":
                x = x * 2
            # 手动触发动态图的断点
            torch._dynamo.graph_break()
            # 恢复默认随机数生成器的状态为初始状态
            torch.default_generator.set_state(default_state)
            # 再次生成一个随机数张量
            y = torch.rand([2, 3])
            return x, y

        # 对fn应用"eager"优化，使其立即执行
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 调用优化后的函数，获取返回值x和y
        x, y = opt_fn()
        # 断言优化后的结果x等于y乘以2
        self.assertEqual(x, y * 2)

    # 定义一个测试函数，测试懒加载属性的功能
    def test_torch_distributions_lazy_property(self):
        # 定义一个函数fn，接受一个输入x，返回x作为概率分布的熵
        def fn(x):
            return torch.distributions.Categorical(probs=x).entropy()

        # 对fn应用"eager"优化，使其立即执行
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 生成一个随机数张量x
        x = torch.rand([4, 4])
        # 断言优化后的fn(x)结果与未优化的fn(x)结果相等
        self.assertEqual(opt_fn(x), fn(x))

    # 定义一个测试函数，测试异常情况下的函数行为
    def test_guard_failure_fn(self):
        # 定义一个函数fn，接受三个输入x、y、k，对它们进行数学运算并返回结果
        def fn(x, y, k):
            x = x + 1
            y = y + 1
            return x * y * k

        # 创建两个输入张量x和y
        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([1.0, 1.0])

        guard_failure = None

        # 定义一个内部函数guard_failures，用于捕获失败时的信息
        def guard_failures(failure):
            nonlocal guard_failure
            guard_failure = failure

        # 对fn应用"eager"优化，同时设置保护失败时的处理函数为guard_failures
        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(fn)

        # 创建两个新的输入张量x2和y2
        x2 = torch.tensor([0.5, 0.5, 1.0])
        y2 = torch.tensor([0.5, 0.5, 0.5])

        # 调用优化后的函数，分别传入x、y和x2、y2
        opt_fn(x, y, 3)
        opt_fn(x2, y2, 5)

        # 根据特定的配置条件，检查是否出现了保护失败
        if (
            not torch._dynamo.config.specialize_int
            and not torch._dynamo.config.assume_static_by_default
        ):
            # 在特定配置下，验证未发生保护失败
            self.assertTrue(guard_failure is None)
        else:
            # 在其他配置下，验证确实发生了保护失败
            self.assertTrue(guard_failure is not None)

    # 定义一个测试函数，测试在特定形状控制下的函数行为
    def test_guard_failure_fn_shape_control(self):
        # 定义一个函数fn，接受两个输入x和y，根据它们的形状进行不同的数学运算并返回结果
        def fn(x, y):
            if x.shape[0] < 4:
                if y.shape[0] < 3:
                    return x * y
                else:
                    return x + y
            else:
                return -1

        # 创建两个随机数张量x和y，形状为[2, 2]
        x = torch.randn([2, 2])
        y = torch.randn([2, 2])

        guard_failure = None

        # 定义一个内部函数guard_failures，用于捕获失败时的信息
        def guard_failures(failure):
            nonlocal guard_failure
            guard_failure = failure

        # 对fn应用"eager"优化，同时设置保护失败时的处理函数为guard_failures
        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(fn)

        # 创建两个新的随机数张量x2和y2，形状为[5, 5]
        x2 = torch.randn([5, 5])
        y2 = torch.randn([5, 5])

        # 调用优化后的函数，分别传入x、y和x2、y2
        opt_fn(x, y)
        opt_fn(x2, y2)

        # 验证是否发生了保护失败
        self.assertTrue(guard_failure is not None)
        # 获取第一个保护失败信息的第一行，并根据配置条件验证其内容
        first_guard_failure = guard_failure[0].partition("\n")[0]
        if torch._dynamo.config.assume_static_by_default:
            # 在默认静态假设下，验证特定形状的不匹配信息是否在错误信息中
            self.assertIn(
                """tensor 'L['x']' size mismatch at index 0. expected 2, actual 5""",
                first_guard_failure,
            )
        else:
            # 在其他条件下，验证形状小于3的条件是否在错误信息中
            self.assertIn("""L['x'].size()[0] < 3""", first_guard_failure)
    # 定义一个测试函数，用于测试在特定条件下优化函数的失败情况
    def test_guard_failure_fn2(self):
        # 定义一个简单的数学函数 fn，对输入进行数学运算并返回结果
        def fn(x, y):
            x = x + 1  # 对输入张量 x 进行加法操作
            y = y + 1  # 对输入张量 y 进行加法操作
            return x * y  # 返回 x 和 y 的乘积结果

        # 创建两个张量 x 和 y，用于测试输入
        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([1.0, 1.0])

        # 初始化 guard_failure 为 None，用于捕获 guard_failures 函数中的失败情况
        guard_failure = None

        # 定义一个 guard_failures 函数，用于捕获优化函数失败的情况
        def guard_failures(failure):
            nonlocal guard_failure  # 声明在闭包中使用外部的 guard_failure 变量
            guard_failure = failure  # 将失败情况赋值给 guard_failure

        # 通过 torch._dynamo.optimize 创建一个优化后的函数 opt_fn，使用 guard_failures 函数捕获失败情况
        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(fn)

        # 创建另外两个张量 x2 和 y2，用于测试输入的维度不匹配情况
        x2 = torch.tensor([0.5, 0.5, 1.0])
        y2 = torch.tensor([0.5, 0.5, 0.5])

        # 分别对 opt_fn 使用 x, y 和 x2, y2 进行计算
        opt_fn(x, y)
        opt_fn(x2, y2)

        # 根据配置检查 guard_failure 是否为预期的失败情况
        if torch._dynamo.config.assume_static_by_default:
            self.assertIn(
                """tensor 'L['x']' size mismatch at index 0. expected 2, actual 3""",
                guard_failure[0],  # 检查 guard_failure 中是否包含预期的错误消息
            )
        else:
            self.assertTrue(guard_failure is None)  # 确保 guard_failure 为空

    # 定义另一个测试函数，用于测试张量迭代时的失败情况
    def test_guard_failure_fn_tensor_iter(self):
        # 定义一个函数 fn，对输入张量中的每个元素进行修改并返回结果
        def fn(x):
            for y in x:
                y.add_(1.0)  # 对输入张量 x 中的每个元素 y 加 1
            return y  # 返回最后一个处理过的元素 y

        # 初始化 guard_failure 为 None，用于捕获 guard_failures 函数中的失败情况
        guard_failure = None

        # 定义一个 guard_failures 函数，用于捕获优化函数失败的情况
        def guard_failures(failure):
            nonlocal guard_failure  # 声明在闭包中使用外部的 guard_failure 变量
            guard_failure = failure  # 将失败情况赋值给 guard_failure

        # 通过 torch._dynamo.optimize 创建一个优化后的函数 opt_fn，使用 guard_failures 函数捕获失败情况
        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(fn)

        # 创建一个随机张量 args1 用于测试输入
        args1 = torch.randn(10, 10)
        out = fn(args1)  # 使用原始函数 fn 处理 args1
        opt_out = opt_fn(args1)  # 使用优化后的函数 opt_fn 处理 args1
        self.assertTrue(same(out, opt_out))  # 断言两者输出结果相同

        # 创建一个维度不匹配的随机张量 args2 用于测试输入
        args2 = torch.randn(9, 10)
        out = fn(args2)  # 使用原始函数 fn 处理 args2
        opt_out = opt_fn(args2)  # 使用优化后的函数 opt_fn 处理 args2
        self.assertTrue(same(out, opt_out))  # 断言两者输出结果相同

        # 断言 guard_failure 不为空，并检查其中是否包含预期的错误消息
        self.assertTrue(guard_failure is not None)
        self.assertIn(
            """len(L['x']) == 10""",
            guard_failure[0],  # 检查 guard_failure 中是否包含预期的错误消息
        )

    # 定义一个测试函数，用于测试图状态的恢复
    def test_restore_graphstate(self):
        # 定义一个嵌套函数 nested_fn，根据条件返回不同的数学运算结果
        def nested_fn(s):
            if x[0] < 10:
                return s * s
            return s

        # 定义一个函数 fn，对输入张量 x 和 y 进行数学运算并返回结果
        def fn(x, y):
            x = x + 1  # 对输入张量 x 进行加法操作
            y = nested_fn(y)  # 调用嵌套函数 nested_fn 处理输入张量 y
            y = y + 10  # 对处理后的 y 进行加法操作
            return x * y  # 返回 x 和 y 的乘积结果

        # 初始化一个空列表 all_guards，用于存储所有的 guard
        all_guards = []

        # 定义一个 guard_export_print 函数，将接收到的 guards 添加到 all_guards 列表中
        def guard_export_print(guards):
            nonlocal all_guards  # 声明在闭包中使用外部的 all_guards 变量
            all_guards.extend(guards)  # 将 guards 扩展到 all_guards 中

        # 通过 torch._dynamo.optimize 创建一个优化后的函数 opt_fn，使用 guard_export_print 函数捕获所有的 guard
        opt_fn = torch._dynamo.optimize("eager", guard_export_fn=guard_export_print)(fn)

        # 创建两个张量 x 和 y，用于测试输入
        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([1.0, 1.0])

        opt_fn(x, y)  # 使用优化后的函数 opt_fn 处理输入张量 x 和 y

        # 遍历 all_guards 列表中的每一个 guard，并断言其名字不等于指定的字符串
        for guard in all_guards:
            # 此 guard 是在函数 nested_fn 中创建的
            self.assertTrue(guard.name != "nested_fn.__closure__[0].cell_contents")
    def test_call_parent_non_class_methods_from_child(self):
        class A:
            a = 4  # 类A的属性a

            def add(self, x):
                return x + 10  # 返回参数x加10的结果

            def mul(self, x):
                return x * 0.1  # 返回参数x乘以0.1的结果

        class B(A):
            coeff = 4  # 类B的类属性coeff为4

            def add(self, x):
                return x + 20  # 返回参数x加20的结果

            @classmethod
            def cube(cls, x):
                return cls.coeff * x * x * x  # 返回类属性coeff与参数x的立方乘积

            def mul(self, x):
                return super().mul(x) * x * 0.2  # 调用父类A的mul方法，返回其结果乘以参数x乘以0.2的结果

        class C(B):
            def add(self, x):
                b = super().cube(x)  # 调用父类B的cube方法，返回其结果赋值给变量b
                c = A.add(self, x)  # 调用类A的add方法，返回其结果赋值给变量c
                d = B.mul(self, x)  # 调用父类B的mul方法，返回其结果赋值给变量d
                e = super(B, self).add(x)  # 调用父类B的add方法，返回其结果赋值给变量e
                f = super().a * x  # 调用父类A的属性a，与参数x相乘赋值给变量f
                return b + c + d + e + f  # 返回变量b、c、d、e、f的和作为结果

        x = torch.rand(4)  # 生成一个4维的随机张量x
        fn = C().add  # 获取类C的实例的add方法赋值给变量fn
        ref = fn(x)  # 调用fn方法，传入参数x，返回结果赋值给变量ref
        cnt = torch._dynamo.testing.CompileCounter()  # 创建编译计数器对象cnt
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(fn)  # 使用优化器对fn方法进行优化赋值给opt_fn
        res = opt_fn(x)  # 调用优化后的opt_fn方法，传入参数x，返回结果赋值给变量res
        self.assertTrue(same(ref, res))  # 断言ref与res是否相同
        self.assertEqual(cnt.frame_count, 1)  # 断言编译计数器的帧计数是否为1

        # 检查重新编译
        A.a = 5  # 修改类A的属性a为5
        ref = fn(x)  # 再次调用fn方法，传入参数x，返回结果赋值给变量ref
        res = opt_fn(x)  # 再次调用优化后的opt_fn方法，传入参数x，返回结果赋值给变量res
        self.assertTrue(same(ref, res))  # 断言ref与res是否相同
        # 确保super保护检查按预期工作
        res = opt_fn(x)  # 再次调用优化后的opt_fn方法，传入参数x，返回结果赋值给变量res
        self.assertEqual(cnt.frame_count, 2)  # 断言编译计数器的帧计数是否为2
    def test_torch_package_working_with_trace(self):
        # 导入需要的测试函数
        # from torch._dynamo.test_case import run_tests

        # 创建输入数据列表
        inputs = [torch.randn([2, 2]), torch.randn([2, 2])]

        # 优化模型并序列化保存
        optimized_model = torch._dynamo.optimize(backend="eager")(
            MyPickledModule(torch.randn([2, 2]))
        )

        # 导入 torch 的 package 模块
        from torch import package

        # 定义模型保存路径及相关信息
        path = "/tmp/MyPickledModule.pt"
        package_name = "MyPickledModule"
        resource_name = "MyPickledModule.pkl"

        # 创建模型对象
        model = MyPickledModule(torch.randn([2, 2]))

        # 使用 PackageExporter 导出模型及其依赖资源
        with package.PackageExporter(path) as exp:
            exp.extern("**")
            exp.save_pickle(package_name, resource_name, model)

        # 使用 PackageImporter 导入模型
        imp = package.PackageImporter(path)
        loaded_model = imp.load_pickle(package_name, resource_name)

        # 优化加载的模型并进行推理
        optimized_loaded_model = torch._dynamo.optimize("eager")(loaded_model)(*inputs)

    def test_shape_and_tuple_equality(self):
        # 定义一个函数，根据条件返回计算结果
        def fn(x, y, t):
            z = x * y
            if x.size() == t:
                return z.cos()
            return z.sin()

        # 使用 eager 模式优化函数 fn
        torch._dynamo.optimize("eager", nopython=True)(fn)(
            torch.randn([4, 4]), torch.randn([4, 4]), (4, 4)
        )

    def test_int_list(self):
        # 如果 assume_static_by_default 为 True，则使用特定的整数列表
        # 否则使用未指定的整数列表
        def fn(x, y):
            return torch.sin(x + y[1] % 2)

        # 创建随机张量 x
        x = torch.randn(6)
        
        # 创建 CompileCounter 对象用于统计编译次数
        cnt = torch._dynamo.testing.CompileCounter()

        # 优化函数 fn 并进行测试
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        for i in range(10, 25, 3):
            y = [i, i + 1, i + 2]
            ref = fn(x, y)
            res = opt_fn(x, y)
            self.assertTrue(same(ref, res))

        # 根据不同的配置断言编译次数的期望值
        if torch._dynamo.config.assume_static_by_default:
            if torch._dynamo.config.automatic_dynamic_shapes:
                self.assertExpectedInline(cnt.frame_count, """2""")
            else:
                self.assertExpectedInline(cnt.frame_count, """5""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")
    # 定义测试方法，用于测试已修补的内建函数
    def test_patched_builtin_functions(self):
        import builtins

        # 缓存原始的内建函数标识
        torch._dynamo.trace_rules._builtin_function_ids()

        # 定义一个简单的类 MyClass
        class MyClass:
            pass

        # 缓存内建的 isinstance 函数
        builtin_isinstance = builtins.isinstance

        # 定义一个修补后的 isinstance 函数，用于替换内建函数
        def patched_isinstance(obj, classinfo) -> bool:
            if builtin_isinstance(obj, MyClass):
                return False
            else:
                return builtin_isinstance(obj, classinfo)

        # 定义一个测试函数 fn
        def fn(x, y):
            # 如果 y 是 MyClass 类型，则返回 x + 1
            if isinstance(y, MyClass):
                return x + 1
            else:
                return x - 1

        # 创建一个 torch 的 Tensor 对象 x
        x = torch.ones(2, 3)
        # 创建一个 MyClass 类的实例 y
        y = MyClass()

        try:
            # 记录原始的 isinstance 函数，并将其替换为修补后的版本
            ref = fn(x, y)
            builtins.isinstance = patched_isinstance
            # 使用 torch.compile 进行函数的优化编译
            opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
            # 执行优化后的函数并记录结果
            res = opt_fn(x, y)
            # 断言优化前后的结果是否一致
            self.assertTrue(same(ref, x + 1))
            self.assertTrue(same(res, x - 1))
        finally:
            # 恢复原始的 isinstance 函数
            builtins.isinstance = builtin_isinstance

        # 由于 builtins 现在已经恢复到未修补状态，检查是否需要重新编译
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        res = opt_fn(x, y)
        self.assertTrue(same(res, x + 1))

    # 测试 tensor.attribute -> torch.something() 的特定情况
    def test_real_imag_tensor_attribute(self):
        # 定义一个测试函数 fn
        def fn(x, y):
            # 获取复数张量 x 的实部和虚部
            a = x.real
            b = x.imag
            # 返回 a + y 乘以 b 的结果
            return torch.mul(torch.add(a, y), b)

        # 创建实数部分为随机值的张量 x_real
        x_real = torch.rand((4, 4))
        # 创建虚数部分为随机值的张量 x_imag
        x_imag = torch.rand((4, 4))
        # 创建复数张量 x
        x = torch.complex(x_real, x_imag)
        # 创建另一个随机值张量 y
        y = torch.rand((4, 4))

        # 记录原始的函数执行结果
        ref = fn(x, y)
        # 使用 torch._dynamo.optimize 进行函数的优化
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 执行优化后的函数并记录结果
        res = opt_fn(x, y)
        # 断言优化前后的结果是否一致
        self.assertTrue(same(ref, res))

    # 测试 cast 函数的使用
    def test_cast(self):
        # 导入 cast 函数
        from typing import cast

        # 定义一个测试函数 fn
        def fn(x):
            # 对输入的张量 x 执行加法操作，并将结果强制转换为 torch.Tensor 类型
            return cast(torch.Tensor, torch.add(x, 1.0))

        # 使用 torch.compile 进行函数的优化编译
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        # 记录原始的函数执行结果
        ref = fn(torch.ones(2, 2))
        # 执行优化后的函数并记录结果
        res = opt_fn(torch.ones(2, 2))

        # 断言优化前后的结果是否一致
        self.assertTrue(same(ref, res))

    # 测试 tensor.T 属性的使用
    def test_T_tensor_attribute(self):
        # 定义一个测试函数 fn
        def fn(x, y):
            # 获取张量 x 的转置，并将其与 y 相加
            a = x.T
            return torch.add(a, y)

        # 创建一个随机值张量 x
        x = torch.rand((4, 4))
        # 创建另一个随机值张量 y
        y = torch.rand((4, 4))

        # 记录原始的函数执行结果
        ref = fn(x, y)
        # 使用 torch._dynamo.optimize 进行函数的优化
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 执行优化后的函数并记录结果
        res = opt_fn(x, y)
        # 断言优化前后的结果是否一致
        self.assertTrue(same(ref, res))

    # 测试递归使用 tensor 属性的情况
    def test_recursive_tensor_attribute(self):
        # 定义一个测试函数 fn
        def fn(x, y):
            # 获取复数张量 x 的实部的转置，并与虚部相加后再乘以虚部
            a = x.real.T
            b = x.imag
            return torch.mul(torch.add(a, y), b)

        # 创建实数部分为随机值的张量 x_real
        x_real = torch.rand((4, 4))
        # 创建虚数部分为随机值的张量 x_imag
        x_imag = torch.rand((4, 4))
        # 创建复数张量 x
        x = torch.complex(x_real, x_imag)
        # 创建另一个随机值张量 y
        y = torch.rand((4, 4))

        # 记录原始的函数执行结果
        ref = fn(x, y)
        # 使用 torch._dynamo.optimize 进行函数的优化
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 执行优化后的函数并记录结果
        res = opt_fn(x, y)
        # 断言优化前后的结果是否一致
        self.assertTrue(same(ref, res))
    def test_assigning_function_to_object_attribute(self):
        # 定义一个用户自定义函数作为对象属性，不会转换为绑定方法
        def my_add(*args):
            a, b = args
            return a + b

        # 定义一个带有属性的类
        class MyClass:
            def __init__(self, func):
                self.add = func  # 将传入的函数作为对象的 add 属性

        obj = MyClass(my_add)  # 创建 MyClass 的实例，并将 my_add 函数作为参数传入

        def fn(x):
            return obj.add(x, 2)  # 调用 MyClass 实例的 add 方法

        x = torch.rand(2, 3)  # 创建一个随机张量 x
        ref = fn(x)  # 调用 fn 函数，计算 obj.add(x, 2) 的结果并赋给 ref
        opt_fn = torch.compile(backend="eager")(fn)  # 使用 torch.compile 进行优化编译
        res = opt_fn(x)  # 调用优化后的函数 opt_fn，计算 obj.add(x, 2) 的结果并赋给 res
        self.assertTrue(same(ref, res))  # 断言 ref 和 res 相等

    def test_assigning_function_to_class_attribute(self):
        # 定义一个用户自定义函数作为类的属性，会转换为绑定方法
        def my_add(*args):
            obj, a, b = args
            return obj.x + a + b  # 返回 obj.x + a + b 的结果

        # 定义一个带有属性的类
        class MyClass:
            add = my_add  # 将 my_add 函数直接作为类的属性

            def __init__(self, x):
                self.x = x  # 初始化实例的属性 x

        obj = MyClass(0.5)  # 创建 MyClass 的实例

        def fn(x):
            return obj.add(x, 2)  # 调用 MyClass 类的 add 方法

        x = torch.rand(2, 3)  # 创建一个随机张量 x
        ref = fn(x)  # 调用 fn 函数，计算 obj.add(x, 2) 的结果并赋给 ref
        opt_fn = torch.compile(backend="eager")(fn)  # 使用 torch.compile 进行优化编译
        res = opt_fn(x)  # 调用优化后的函数 opt_fn，计算 obj.add(x, 2) 的结果并赋给 res
        self.assertTrue(same(ref, res))  # 断言 ref 和 res 相等

    def test_tagging_tensors_simple(self):
        # 定义一个简单的函数 foo
        def foo(x, y):
            return x * y, x, y

        a = torch.randn([3, 3])  # 创建一个形状为 [3, 3] 的随机张量 a
        a.tag = "a"  # 给张量 a 添加标签 "a"
        a.frog = "ribbity ribbit"  # 给张量 a 添加属性 "frog"，值为 "ribbity ribbit"
        b = torch.randn([3, 3])  # 创建一个形状为 [3, 3] 的随机张量 b
        b.tag = "b"  # 给张量 b 添加标签 "b"
        b.frog = "ribbit"  # 给张量 b 添加属性 "frog"，值为 "ribbit"

        exported = torch._dynamo.export(foo)(a, b)  # 导出函数 foo 的计算图
        out_graph = exported[0]  # 获取导出的计算图的第一个输出

        nodes = list(out_graph.graph.nodes)  # 获取计算图中所有节点的列表
        placeholders = [node for node in nodes if node.op == "placeholder"]  # 获取所有占位符节点
        all_tags = []
        all_frogs = []
        for placeholder in placeholders:
            if "tensor_dict" in placeholder.meta:
                all_tags.append(placeholder.meta["tensor_dict"]["tag"])  # 提取占位符中的标签信息
                all_frogs.append(placeholder.meta["tensor_dict"]["frog"])  # 提取占位符中的 frog 属性信息

        self.assertEqual(all_tags, ["a", "b"])  # 断言所有标签信息与预期相符
        self.assertEqual(all_frogs, ["ribbity ribbit", "ribbit"])  # 断言所有 frog 属性信息与预期相符
    # 定义一个测试函数，测试在混合使用和未使用结构的情况下的张量标记
    def test_tagging_tensors_mix_used_unused_structure(self):
        # 定义一个内部函数，用于处理注意力机制前的状态操作
        def pre_attention_state_ops(input, mems, state):
            # 从状态列表中获取当前位置的键和值
            lc_key = state[0]
            lc_val = state[1]
            # 创建一个空列表 bar
            bar = []
            # 循环遍历范围在 0 到 4 的整数
            for i in range(0, 4):
                # 创建一个空列表 bar2
                bar2 = []
                # 循环遍历范围在 0 到 3 的整数
                for j in range(0, 3):
                    # 向 bar2 中添加 lc_key、lc_val 和一个新的张量
                    bar2.append(
                        lc_key + lc_val + torch.tensor([0.1, 0.25, 0.4, 0.5, 0.1])
                    )
                # 向 bar 中添加 bar2
                bar.append(bar2)
            # 返回 bar
            return bar

        # 创建一个张量 mems，并设置其标记为 "MEMS"
        mems = torch.tensor([[[1.8364, 0.2724, -1.4917, -0.4367, 0.8640]]])
        mems.tag = "MEMS"
        # 创建一个状态列表 state，包含两个张量，分别设置其标记为 "STATE_0" 和 "HMMM"
        state = [
            torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
            torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
        ]
        state[0].tag = "STATE_0"
        state[1].tag = "HMMM"
        # 创建一个张量 i，并设置其标记为 "FOO"
        i = torch.tensor(
            [
                [0.0313, -0.1487, -0.3846, -0.5321],
                [-1.7073, 1.3331, -0.0890, -1.4935],
                [-0.8314, -0.1862, -0.5935, 1.5232],
            ]
        )
        i.tag = "FOO"

        # 使用 torch._dynamo.export 方法导出 pre_attention_state_ops 函数
        exported = torch._dynamo.export(pre_attention_state_ops)(i, mems, state)
        # 获取导出结果中的图形对象
        out_graph = exported[0]

        # 获取导出图中的所有节点
        nodes = list(out_graph.graph.nodes)
        # 筛选出所有操作为 "placeholder" 的节点
        placeholders = [node for node in nodes if node.op == "placeholder"]
        # 创建一个空列表 all_tags，用于存储所有节点的标记信息
        all_tags = []
        # 遍历所有占位符节点
        for placeholder in placeholders:
            # 如果占位符节点的元数据中包含 "tensor_dict"
            if "tensor_dict" in placeholder.meta:
                # 获取该节点的标记信息，并添加到 all_tags 列表中
                all_tags.append(placeholder.meta["tensor_dict"]["tag"])

        # 使用 self.assertEqual 方法断言 all_tags 应为 ["STATE_0", "HMMM"]
        self.assertEqual(all_tags, ["STATE_0", "HMMM"])

    # 定义一个测试函数，测试获取自定义张量属性
    def test_get_custom_tensor_attribute(self):
        # 定义一个函数 fn，接受一个参数 x，返回 x.custom_attr 乘以 x 的结果
        def fn(x):
            return x.custom_attr * x

        # 创建一个随机张量 x，并设置其自定义属性 custom_attr 为 3.14
        x = torch.rand((2, 2))
        x.custom_attr = 3.14
        # 计算参考结果 ref
        ref = fn(x)
        # 使用 torch._dynamo.optimize("eager") 方法优化 fn 函数
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 计算优化后的结果 res
        res = opt_fn(x)
        # 使用 self.assertTrue 方法断言 ref 和 res 相同
        self.assertTrue(same(ref, res))

    # 定义一个测试函数，测试设置自定义张量属性
    def test_set_custom_tensor_attribute(self):
        # 定义一个函数 fn，接受一个参数 x，在其中设置 x 的自定义属性 custom_attr 为 3.14，然后返回其结果
        def fn(x):
            x.custom_attr = 3.14
            return x.custom_attr * x

        # 创建一个随机张量 x
        x = torch.rand((2, 2))
        # 计算参考结果 ref
        ref = fn(x)
        # 使用 torch._dynamo.optimize("eager") 方法优化 fn 函数
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 计算优化后的结果 res
        res = opt_fn(x)
        # 使用 self.assertTrue 方法断言 ref 和 res 相同
        self.assertTrue(same(ref, res))

    # 定义一个测试函数，测试在动态机制中未处理的异常情况
    def test_unhandled_exception_in_dynamo(self):
        # 定义一个函数 f，接受一个参数 a，在其中使 a 自增 1，并引发 RuntimeError 异常 "smoge"，然后返回 a
        def f(a):
            a += 1
            raise RuntimeError("smoge")
            return a

        # 使用 torch._dynamo.optimize("eager") 方法优化 f 函数
        opt_fn = torch._dynamo.optimize("eager")(f)
        try:
            # 尝试调用优化后的函数 opt_fn，并传入 torch.ones(2) 作为参数
            opt_fn(torch.ones(2))
        except RuntimeError as e:
            # 使用 self.assertIn 方法断言异常信息中包含 "smoge"
            self.assertIn("smoge", traceback.format_exc())
    def test_unhandled_exception_in_dynamo2(self):
        # 在 Python 3.11 中如果影子帧释放不当则会导致段错误
        from torch.testing import make_tensor
        
        def fn():
            # 测试稠密版本和稀疏版本的错误是否相同
            def test1(*, is_sparse):
                # 形状必须兼容以进行矩阵乘法
                a = make_tensor((2, 3), dtype=torch.float32, device="cpu")
                if is_sparse:
                    a_sparse = a.to_sparse_csr()
                    return torch.addmm(a, a_sparse, a)
                else:
                    return torch.addmm(a, a, a)
            
            try:
                test1(is_sparse=False)
            except RuntimeError as msg:
                try:
                    test1(is_sparse=True)
                except RuntimeError as msg2:
                    raise RuntimeError("smoge")  # 抛出自定义的 RuntimeError 异常
            
        # 对 fn 函数应用 torch._dynamo.optimize("eager") 优化
        opt_fn = torch._dynamo.optimize("eager")(fn)
        try:
            opt_fn()
        except RuntimeError:
            self.assertIn("smoge", traceback.format_exc())

    def test_variable_access_in_exception(self):
        def fn():
            x = torch.ones(1)
            try:
                raise RuntimeError("bad")  # 抛出 RuntimeError 异常
            except RuntimeError:
                x += 1  # 捕获异常后对 x 进行操作
            return x
        
        # 对 fn 函数应用 torch._dynamo.optimize("eager", nopython=True) 优化
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        self.assertEqual(opt_fn(), torch.tensor([2.0]))  # 断言返回值为 [2.0]

    def test_nested_sequential_with(self):
        def fn(x):
            with torch.set_grad_enabled(True):  # 启用梯度追踪
                with torch.set_grad_enabled(False):  # 禁用梯度追踪
                    x = x + 1  # 对 x 进行操作
                with torch.set_grad_enabled(True):  # 再次启用梯度追踪
                    x = x + 1  # 对 x 进行操作
                return x
        
        # 对 fn 函数应用 torch._dynamo.optimize("eager") 优化
        opt_fn = torch._dynamo.optimize("eager")(fn)
        self.assertEqual(opt_fn(torch.ones(1)), torch.tensor([3.0]))  # 断言返回值为 [3.0]

    def test_nested_sequential_try(self):
        def fn(x):
            try:
                try:
                    x = x + 1  # 对 x 进行操作
                except:
                    pass
                try:
                    try:
                        x = x + 1  # 对 x 进行操作
                    except:
                        pass
                except:
                    pass
            except:
                pass
            return x
        
        # 对 fn 函数应用 torch._dynamo.optimize("eager") 优化
        opt_fn = torch._dynamo.optimize("eager")(fn)
        self.assertEqual(opt_fn(torch.ones(1)), torch.tensor([3.0]))  # 断言返回值为 [3.0]

    def test_nested_sequential_try_with(self):
        def fn(x):
            with torch.set_grad_enabled(True):  # 启用梯度追踪
                try:
                    x = x + 1  # 对 x 进行操作
                except:
                    pass
                try:
                    with torch.set_grad_enabled(False):  # 禁用梯度追踪
                        x = x + 1  # 对 x 进行操作
                except:
                    pass
            return x
        
        # 对 fn 函数应用 torch._dynamo.optimize("eager") 优化
        opt_fn = torch._dynamo.optimize("eager")(fn)
        self.assertEqual(opt_fn(torch.ones(1)), torch.tensor([3.0]))  # 断言返回值为 [3.0]
    # 定义一个测试函数，用于测试嵌套的 try-with 语句与图形中断
    def test_nested_sequential_try_with_graph_break(self):
        
        # 定义一个内部函数 fn，接受参数 x 和 n
        def fn(x, n):
            # 启用梯度计算
            with torch.set_grad_enabled(True):
                # 禁用梯度计算
                with torch.set_grad_enabled(False):
                    # x 值加一
                    x = x + 1
                    # 调用 torch._dynamo.graph_break() 进行图形中断操作
                    torch._dynamo.graph_break()
                
                # 尝试执行以下代码块
                try:
                    # 再次禁用梯度计算
                    with torch.set_grad_enabled(False):
                        # x 值再加一
                        x = x + 1
                        # 如果 n 等于 0，则执行图形中断操作
                        if n == 0:
                            torch._dynamo.graph_break()
                # 捕获任何异常
                except:
                    pass
                
                # 最后再次禁用梯度计算
                with torch.set_grad_enabled(False):
                    # x 值再加一
                    x = x + 1
                    # 调用 torch._dynamo.graph_break() 进行图形中断操作
                    torch._dynamo.graph_break()
                
                # x 值再加一
                x = x + 1
            
            # 返回计算后的 x 值
            return x
        
        # 创建 CompileCounter 对象
        counter = CompileCounter()
        # 对函数 fn 进行优化，得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(counter)(fn)
        # 断言优化后的函数对输入 torch.ones(1) 和 0 的输出结果为 torch.tensor([5.0])
        self.assertEqual(opt_fn(torch.ones(1), 0), torch.tensor([5.0]))
        # 断言 frame_count 属性为 1
        self.assertEqual(counter.frame_count, 1)
        
        # 重置 torch._dynamo
        torch._dynamo.reset()
        # 创建新的 CompileCounter 对象
        counter = CompileCounter()
        # 再次对函数 fn 进行优化，得到新的优化函数 opt_fn
        opt_fn = torch._dynamo.optimize(counter)(fn)
        # 断言优化后的函数对输入 torch.ones(1) 和 1 的输出结果为 torch.tensor([5.0])
        self.assertEqual(opt_fn(torch.ones(1), 1), torch.tensor([5.0]))
        # 断言 frame_count 属性为 3
        self.assertEqual(counter.frame_count, 3)

    # 定义一个测试函数，测试 OrderedDict 的别名重构功能
    def test_ordered_dict_alias_reconstruct(self):
        # 导入 collections 模块中的 OrderedDict 类
        od = collections.OrderedDict
        
        # 定义内部函数 fn
        def fn():
            # 创建一个普通字典 d1
            d1 = dict()
            # 向字典 d1 中添加键值对 "a": 1
            d1["a"] = 1
            # 使用 OrderedDict 的别名 od 创建一个有序字典 d2，复制字典 d1 的内容
            d2 = od(d1)
            # 向字典 d2 中添加键值对 "b": 2
            d2["b"] = 2
            # 调用 torch._dynamo.graph_break() 进行图形中断操作
            torch._dynamo.graph_break()
            
            # 如果 d2 是 OrderedDict 类型，则返回 d2["a"] + d2["b"] 的和
            if isinstance(d2, od):
                return d2["a"] + d2["b"]
            else:
                # 否则返回 0
                return 0
        
        # 使用 dis.dis 函数显示 fn 函数的字节码指令
        dis.dis(fn)
        # 断言使用 "eager" 优化模式后 fn 函数的执行结果为 3
        self.assertEqual(torch._dynamo.optimize("eager")(fn)(), 3)
        
    # 注意：这个测试在 Python 支持多行错误时可以移除。
    # 参见 https://github.com/python/cpython/issues/106922
    @skipIfNotPy311
    def test_get_instruction_source_311(self):
        def f():
            # flake8: noqa  # 忽略 flake8 的警告
            # fmt: off  # 关闭代码格式化
            # test binary ops  # 测试二进制运算
            a = ( b   )   +   c  # 定义并计算 a 的值
            a = (a + b) // (c - d)  # 定义并计算 a 的值
            a = b    \  # 定义并计算 a 的值，多行连接
         +\
               c  # test  # 定义并计算 a 的值
            a = (  # 定义并计算 a 的值，多行连接
                (b  # test +  # 定义并计算 a 的值
                    )  \
                # +  # 注释
            << (  # 左移运算符
                c  # test  # 定义并计算 a 的值
                \
            )  # test  # 注释

            # test slice  # 测试切片操作
            a = bbb   [  ccc    ]  # 定义并计算 a 的值
            b = bbbbb \  # 定义并计算 b 的值，多行连接
                [  ccc # test  # 定义并计算 b 的值
                 + ddd  \  # 定义并计算 b 的值，多行连接
                ] # test  # 注释
            a = bbb[ccc][ddd][eee]  # 定义并计算 a 的值

            # test nested and multiline function calls  # 测试嵌套和多行函数调用
            a = g(g(g(b)))  # 定义并计算 a 的值
            a = g(h(  # 定义并计算 a 的值
                g(b),  # 定义并计算 a 的值
                c  # 定义并计算 a 的值
            ))

            # test chained function calls  # 测试链式函数调用
            a = (g(x).y)(  # 定义并计算 a 的值
                z  # 定义并计算 a 的值
            )(1)(2)  # 定义并计算 a 的值

            # test unicode (match traceback behavior)  # 测试 Unicode 和追踪行为匹配
            a = ("🔥🔥🔥" +  # 定义并计算 a 的值
                + "🔥🔥") + b  # 定义并计算 a 的值

        from torch._dynamo.utils import get_instruction_source_311

        if sys.version_info >= (3, 12):
            # Offsets changed in 3.12, e.g. due to removal of PRECALL inst  # 偏移在3.12中改变，例如由于删除 PRECALL 指令
            offsets = (3, 11, 15, 19, 23, 29, 35, 44, 53, 65)
        else:
            offsets = (3, 11, 15, 19, 23, 29, 35, 46, 58, 74)
        insts = list(dis.get_instructions(f))  # 获取函数 f 的字节码指令列表
        # uncomment to determine offsets  # 解除注释以确定偏移量
        # print(*enumerate(insts), sep="\n")
        all_sources = "\n".join(
            get_instruction_source_311(f.__code__, insts[offset]) for offset in offsets  # 获取每个偏移量处指令的源码
        )
        self.assertExpectedInline(
            all_sources,
            """\
            a = ( b   )   +   c
                ~~~~~~~~~~^~~~~

            a = (a + b) // (c - d)
                ~~~~~~~~^^~~~~~~~~

            a = b    \\
                ~~~~~~
         +\\
         ^~
               c  # test
               ~

                (b  # test +
                ~~~~~~~~~~~~
                    )  \\
                    ~~~~
                # +
                ~~~
            << (
            ^^~~


                c  # test
                ~~~~~~~~~
                \\
                ~
            )  # test
            ~

            a = bbb   [  ccc    ]
                ~~~~~~^^^^^^^^^^^

            b = bbbbb \\
                ~~~~~~~
                [  ccc # test
                ^^^^^^^^^^^^^


                 + ddd  \\
                 ^^^^^^^^


                ] # test
                ^

            a = bbb[ccc][ddd][eee]
                ~~~~~~~~^^^^^

            a = g(g(g(b)))
                  ~^^^^^^

            a = g(h(
                  ~^
                g(b),
                ^^^^^
                c
                ^
            ))
            ^

            a = (g(x).y)(
                ~~~~~~~~~
                z
                ~
            )(1)(2)
            ~^^^
    def test_raise_guard_full_constraint(self):
        # 创建一个形状为 [3, 3, 3] 的随机张量 y
        y = torch.randn([3, 3, 3])

        # 定义一个动态函数 my_dyn_fn，根据输入张量 x 的形状是否为 [3] 分别返回 x.sin() 或 x.cos()
        def my_dyn_fn(x):
            if x.shape[0] == 3:
                return x.sin()
            return x.cos()

        # 标记张量 y 的第一个维度为动态
        torch._dynamo.mark_dynamic(y, 0)
        # 使用 ConstraintViolationError 异常断言，期望优化后的 my_dyn_fn(y) 抛出异常
        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.optimize("eager")(my_dyn_fn)(y)

    def test_raise_guard_indirect_full_constraint(self):
        # 创建一个形状为 [3, 3, 3] 的随机张量 y
        y = torch.randn([3, 3, 3])

        # 定义一个动态函数 dyn_fn，根据输入张量 x 的形状第一个维度大于3返回 x.cos()，小于3返回 x*2，否则返回 x.sin()
        def dyn_fn(x):
            if x.shape[0] > 3:
                return x.cos()
            if x.shape[0] < 3:
                return x * 2
            return x.sin()

        # 标记张量 y 的第一个维度为动态
        torch._dynamo.mark_dynamic(y, 0)
        # 使用 ConstraintViolationError 异常断言，期望优化后的 dyn_fn(y) 抛出异常
        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.optimize("eager")(dyn_fn)(y)

    # Translation validation changes the exception type, don't run with it
    @torch.fx.experimental._config.patch(translation_validation=False)
    def test_mark_dynamic_with_ranges(self):
        # 创建一个形状为 [8, 3, 3] 的随机张量 y
        y = torch.randn([8, 3, 3])

        # 定义一个动态函数 my_dyn_fn，根据输入张量 x 的形状是否为 [3] 分别返回 x.sin() 或 x.cos()
        def my_dyn_fn(x):
            if x.shape[0] == 3:
                return x.sin()
            return x.cos()

        # 标记张量 y 的第一个维度为动态，并设置其在范围 [2, 5] 内
        torch._dynamo.mark_dynamic(y, 0, min=2, max=5)
        # 使用 ConstraintViolationError 异常断言，期望优化后的 my_dyn_fn(y) 抛出异常
        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.optimize("eager")(my_dyn_fn)(y)

    def test_mark_static(self):
        # 创建一个计数器对象 counter
        counter = CompileCounter()

        # 定义一个静态函数 my_dyn_fn，始终返回输入张量 x 的余弦函数值
        def my_dyn_fn(x):
            return x.cos()

        # 创建形状为 [3] 的随机张量 y
        y = torch.randn([3])
        # 标记张量 y 的第一个维度为静态，并使用计数器优化 my_dyn_fn(y)
        torch._dynamo.mark_static(y, 0)
        torch._dynamo.optimize(counter)(my_dyn_fn)(y)

        # 创建形状为 [4] 的随机张量 z
        z = torch.randn([4])
        # 使用计数器优化 my_dyn_fn(z)
        torch._dynamo.optimize(counter)(my_dyn_fn)(z)

        # 断言计数器的帧数为 2
        self.assertEqual(counter.frame_count, 2)

    def test_no_raise_guard_partial_constraint(self):
        # 创建一个形状为 [3, 3, 3] 的随机张量 y
        y = torch.randn([3, 3, 3])

        # 定义一个动态函数 my_dyn_fn，根据输入张量 x 的形状第一个维度大于3返回 x.sin()，否则返回 x.cos()
        def my_dyn_fn(x):
            if x.shape[0] > 3:
                return x.sin()
            return x.cos()

        # 使用计算图动态优化 my_dyn_fn(y)
        torch._dynamo.optimize("eager")(my_dyn_fn)(y)
        # 标记张量 y 的第一个维度为动态
        torch._dynamo.mark_dynamic(y, 0)
        # 重置动态张量优化设置
        torch._dynamo.reset()
        # 再次使用计算图动态优化 my_dyn_fn(y)
        torch._dynamo.optimize("eager")(my_dyn_fn)(y)

    def test_no_raise_guard_partial_constraint_across_break(self):
        # 创建一个形状为 [3, 3, 3] 的随机张量 y
        y = torch.randn([3, 3, 3])

        # 定义一个动态函数 my_dyn_fn，根据输入张量 x 和 y 计算 z = x * y 后，如果 z 形状的第一个维度大于2返回 z.cos()，否则返回 x.cos()
        def my_dyn_fn(x, y):
            z = x * y

            # 在计算图中创建断点
            torch._dynamo.graph_break()
            if z.shape[0] > 2:
                return z.cos()

            return x.cos()

        # 使用计算图动态优化 my_dyn_fn(y, y)
        torch._dynamo.optimize("eager")(my_dyn_fn)(y, y)
        # 标记张量 y 的第一个维度为动态
        torch._dynamo.mark_dynamic(y, 0)
        # 重置动态张量优化设置
        torch._dynamo.reset()
        # 再次使用计算图动态优化 my_dyn_fn(y, y)
        torch._dynamo.optimize("eager")(my_dyn_fn)(y, y)
    # 标记该测试用例预期为失败，因为在图形断点期间未正确传播异常
    @unittest.expectedFailure
    def test_raise_guard_partial_constraint_across_break(self):
        # 创建一个形状为 [3, 3, 3] 的随机张量 y
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x, y):
            # 计算张量 z，其值为 x 与 y 的逐元素乘积
            z = x * y

            # 调用 torch._dynamo.graph_break() 来模拟图形断点
            torch._dynamo.graph_break()

            # 如果 z 的第一个维度的长度为 3，则返回 z 的余弦值
            if z.shape[0] == 3:
                return z.cos()

            # 否则返回 x 的余弦值
            return x.cos()

        # 对 my_dyn_fn 进行优化，采用 eager 模式，然后调用并传入 y 作为参数
        torch._dynamo.optimize("eager")(my_dyn_fn)(y, y)

        # 在 y 的第一个维度上标记为动态
        torch._dynamo.mark_dynamic(y, 0)

        # 重置动态计算图
        torch._dynamo.reset()

        # 使用 assertRaisesRegex 断言异常被触发
        with self.assertRaisesRegex(
            Exception,
        ):
            # 再次优化 my_dyn_fn，并传入 y 作为参数，预期抛出异常
            torch._dynamo.optimize("eager")(my_dyn_fn)(y, y)

    # 测试在没有图形断点的情况下，部分约束是否会触发异常
    def test_raise_guard_partial_constraint_no_graph_break(self):
        # 创建一个形状为 [3, 3, 3] 的随机张量 y
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x, y):
            # 计算张量 z，其值为 x 与 y 的逐元素乘积
            z = x * y

            # 如果 z 的第一个维度的长度为 3，则返回 z 的余弦值
            if z.shape[0] == 3:
                return z.cos()

            # 否则返回 x 的余弦值
            return x.cos()

        # 在 y 的第一个维度上标记为动态
        torch._dynamo.mark_dynamic(y, 0)

        # 使用 assertRaises 断言 ConstraintViolationError 异常被触发
        with self.assertRaises(ConstraintViolationError):
            # 优化 my_dyn_fn，并传入 y 作为参数，预期抛出约束异常
            torch._dynamo.optimize("eager")(my_dyn_fn)(y, y)

    # 测试无法追踪 mark_dynamic 函数的情况
    def test_cannot_trace_mark_dynamic(self):
        # 创建一个形状为 [3, 3, 3] 的随机张量 y
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            # 尝试在动态计算图中标记张量 x 的第一个维度，预期触发异常
            torch._dynamo.mark_dynamic(x, 0)
            return x * x

        # 使用 assertRaisesRegex 断言 AssertionError 异常被触发，且包含特定消息
        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            # 优化 my_dyn_fn，并传入 y 作为参数，预期抛出异常
            torch._dynamo.optimize("eager")(my_dyn_fn)(y)

    # 测试安全未到达的情况下，无法追踪 mark_dynamic 函数
    def test_cannot_trace_mark_dynamic_safe_unreached(self):
        # 创建一个形状为 [3, 3, 3] 的随机张量 y
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            # 如果 x 的第一个维度的长度为 3，则直接返回 x
            if x.shape[0] == 3:
                return x

            # 打印一条信息并尝试标记张量 x 的第一个维度
            print("Running", torch._dynamo.mark_dynamic(x, 0))
            return x * x

        # 优化 my_dyn_fn，并传入 y 作为参数，预期不会抛出异常
        torch._dynamo.optimize("eager")(my_dyn_fn)(y)

    # 测试 AOT 自动求导异常情况
    def test_anomaly_aot_autograd(self):
        # 定义一个失败函数，抛出 AssertionError 异常
        def fail():
            raise AssertionError("fail")

        # 允许函数 h 在图形计算中使用
        @allow_in_graph
        def h(a):
            # 计算张量 a 的和
            r = a.sum()
            # 注册钩子函数，当反向传播时触发异常
            r.register_hook(lambda x: fail())
            return r

        # 使用 AOT 编译器，将函数 h 编译成静态图形
        @torch.compile(backend="aot_eager")
        def f(a):
            return h(a)

        # 使用 assertRaises 断言 BackendCompilerFailed 异常被触发
        with warnings.catch_warnings(record=True) as w, self.assertRaises(
            torch._dynamo.exc.BackendCompilerFailed
        ):
            # 调用编译后的函数 f，并传入形状为 [2, 2] 的随机张量，预期抛出异常
            f(torch.randn(2, 2, requires_grad=True))

        # 在警告消息中查找特定字符串，以确保相关警告被捕获
        self.assertIn("forward call that caused the error", str(w[-1].message))
    # 定义测试函数 test_py_guards_mark_dynamic
    def test_py_guards_mark_dynamic(self):
        
        # 定义内部动态函数 my_dyn_fn，根据输入张量 a 的形状决定返回 a 的余弦或正弦
        def my_dyn_fn(a):
            if a.shape[0] > 2:
                return a.cos()  # 如果张量 a 的第一个维度大于2，则返回 a 的余弦
            return a.sin()  # 否则返回 a 的正弦

        # 创建编译计数器对象
        counter = CompileCounter()

        # 使用动态特性运行优化过程

        # 创建形状为 [3, 3, 3] 的随机张量 x0
        x0 = torch.randn([3, 3, 3])
        # 将 x0 的第0维标记为动态
        torch._dynamo.mark_dynamic(x0, 0)
        # 对 my_dyn_fn 进行优化并计数编译帧数
        torch._dynamo.optimize(counter)(my_dyn_fn)(x0)
        self.assertEqual(counter.frame_count, 1)  # 断言编译帧数为1

        # 不使用动态特性，不会重新编译
        # 创建形状为 [3, 3, 3] 的随机张量 x
        x = torch.randn([3, 3, 3])
        # 对 my_dyn_fn 进行优化并计数编译帧数
        torch._dynamo.optimize(counter)(my_dyn_fn)(x)
        self.assertEqual(counter.frame_count, 1)  # 断言编译帧数仍为1，未增加

        # 标记新的维度 1 为动态
        # 创建形状为 [3, 3, 3] 的随机张量 x1
        x1 = torch.randn([3, 3, 3])
        # 将 x1 的第1维标记为动态
        torch._dynamo.mark_dynamic(x1, 1)
        # 对 my_dyn_fn 进行优化并计数编译帧数
        torch._dynamo.optimize(counter)(my_dyn_fn)(x1)
        # 标记新的维度导致重新编译，因此编译帧数增加到2
        self.assertEqual(counter.frame_count, 2)

        # 重置动态特性
        torch._dynamo.reset()
        # 重置计数器
        counter = CompileCounter()

        # 使用动态特性运行优化过程

        # 对 my_dyn_fn 进行优化并计数编译帧数，此时 x1 的第1维仍然为动态
        torch._dynamo.optimize(counter)(my_dyn_fn)(x1)
        self.assertEqual(counter.frame_count, 1)  # 断言编译帧数为1

        # 使用动态特性运行优化过程

        # 对 my_dyn_fn 进行优化并计数编译帧数，此时 x0 的第0维仍然为动态
        torch._dynamo.optimize(counter)(my_dyn_fn)(x0)
        self.assertEqual(counter.frame_count, 2)  # 断言编译帧数增加到2

        # 使用动态特性运行优化过程

        # 创建形状为 [3, 3, 3] 的随机张量 x012
        x012 = torch.randn([3, 3, 3])
        # 将 x012 的第0、1、2维标记为动态
        torch._dynamo.mark_dynamic(x012, 0)
        torch._dynamo.mark_dynamic(x012, 1)
        torch._dynamo.mark_dynamic(x012, 2)
        # 对 my_dyn_fn 进行优化并计数编译帧数
        torch._dynamo.optimize(counter)(my_dyn_fn)(x012)
        self.assertEqual(counter.frame_count, 3)  # 断言编译帧数增加到3
    # 定义测试函数，用于验证全局状态改变时重新编译的行为
    def test_recompile_on_global_state_change(self):
        # 初始化最后状态为空列表
        last_state = []
        # 计数器初始化为0
        cnt = 0

        # 定义自定义编译器函数
        def my_compiler(gm, _):
            nonlocal cnt
            # 计数器加一
            cnt += 1
            # 读取当前状态
            state = read_state()

            # 内部函数，设置最后状态为当前状态，然后调用原始函数
            def inner(*args):
                last_state[:] = state
                return gm(*args)

            return inner

        # 定义读取状态的函数
        def read_state():
            return [
                torch.is_grad_enabled(),
                torch.are_deterministic_algorithms_enabled(),
                torch._C._get_cublas_allow_tf32(),
            ]

        # 定义写入状态的函数
        def write_state(state):
            torch.set_grad_enabled(state[0]),
            torch.use_deterministic_algorithms(state[1])
            torch._C._set_cublas_allow_tf32(state[2]),

        # 使用自定义编译器装饰函数fn
        @torch.compile(backend=my_compiler)
        def fn(x):
            return x + 1

        # 获取初始状态
        initial_state = read_state()
        # 生成随机数张量
        y = torch.randn(10)
        try:
            # 进行三轮测试
            for round in range(3):
                # 遍历初始状态长度
                for i in range(len(initial_state)):
                    # 创建新状态，将当前状态的一个元素设置为True，其余为False
                    new_state = [False] * len(initial_state)
                    new_state[i] = True
                    # 写入新状态
                    write_state(new_state)
                    # 断言读取状态与新状态相同
                    assert read_state() == new_state
                    # 清空最后状态列表
                    last_state.clear()
                    # 调用fn函数
                    fn(y)
                    # 断言最后状态等于新状态
                    assert last_state == new_state
                    # 如果是第一轮测试
                    if round == 0:
                        # 断言计数器等于i+1
                        assert cnt == i + 1
                    else:
                        # 否则断言计数器等于初始状态长度
                        assert cnt == len(initial_state)
        finally:
            # 最终恢复初始状态
            write_state(initial_state)

    # 定义测试函数，用于验证梯度状态变化时的行为
    def test_grad_state_mutated(self):
        # 记录先前梯度状态
        prior = torch.is_grad_enabled()
        # 初始化值为None
        value = None
        # 实例化编译计数器对象
        cnt = CompileCounter()

        # 在图中允许检查状态的函数
        @torch._dynamo.allow_in_graph
        def check_state():
            nonlocal value
            # 设置值为当前梯度状态
            value = torch.is_grad_enabled()

        # 使用编译器计数器装饰函数fn，同时完整图形化编译
        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            # 调用检查状态函数
            check_state()
            # 设置梯度状态为False
            torch.set_grad_enabled(False)
            return x + 1

        try:
            # 设置梯度状态为True
            torch.set_grad_enabled(True)
            # 调用fn函数
            fn(torch.randn(10))
            # 断言值为True
            assert value is True
            # 断言梯度状态为False
            assert torch.is_grad_enabled() is False

            # 重置值为None
            value = None
            # 再次设置梯度状态为True
            torch.set_grad_enabled(True)
            # 再次调用fn函数
            fn(torch.randn(10))
            # 断言值为True
            assert value is True
            # 断言梯度状态为False
            assert torch.is_grad_enabled() is False

            # 断言编译计数器帧数为1
            assert cnt.frame_count == 1
        finally:
            # 最终将梯度状态恢复为先前状态
            torch.set_grad_enabled(prior)
    def test_deterministic_algorithms_mutated(self):
        # 保存当前的确定性算法状态和警告状态
        prior = torch.are_deterministic_algorithms_enabled()
        prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        # 初始化变量
        value = None
        warn_only = None
        # 编译计数器对象
        cnt = CompileCounter()

        @torch._dynamo.allow_in_graph
        def check_state():
            nonlocal value
            nonlocal warn_only
            # 获取当前的确定性算法状态和警告状态
            value = torch.are_deterministic_algorithms_enabled()
            warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            # 检查并更新状态
            check_state()
            # 禁用确定性算法，不警告
            torch.use_deterministic_algorithms(False, warn_only=False)
            return x + 1

        def run_fn():
            # 启用确定性算法，警告模式
            torch.use_deterministic_algorithms(True, warn_only=True)
            # 运行函数
            fn(torch.randn(10))
            # 断言确保状态正确
            assert value is True
            assert warn_only is True
            assert torch.are_deterministic_algorithms_enabled() is False
            assert torch.is_deterministic_algorithms_warn_only_enabled() is False

        try:
            # 第一次运行测试
            run_fn()
            # 重置状态变量
            value, warn_only = None, None
            # 第二次运行测试
            run_fn()
            # 断言确保编译计数器仅增加一次
            assert cnt.frame_count == 1
        finally:
            # 恢复初始的确定性算法状态和警告状态
            torch.use_deterministic_algorithms(prior, warn_only=prior_warn_only)

    def test_torch_compile_ctx_on_forward_and_training_step(self):
        class MyModel(torch.nn.Module):
            def forward(self):
                ...

            def training_step(self):
                self()

        model = MyModel()
        # 编译模型
        compiled_model = torch.compile(model)

        # 使用编译的上下文包装模型的 forward 和 training_step 方法
        model.forward = compiled_model.dynamo_ctx(model.forward)
        model.training_step = compiled_model.dynamo_ctx(model.training_step)

        # 运行编译后的 training_step 方法
        model.training_step()

    def test_torch_guards_stack_frame_register_inlining(self):
        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([0.75, 0.75, 0.75, 0.75])
        z = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

        def uwu_inline_me(x, y, z):
            # 执行向量操作并返回结果
            r = torch.cat((x, x)) + y
            r2 = torch.cat((y, y)) + z
            return r, r2

        def fn(x, y, z):
            # 调用内联函数 uwu_inline_me 并执行乘法操作
            r, r2 = uwu_inline_me(x, y, z)
            return torch.mul(r, r), torch.mul(r2, r2)

        seen_frames = []
        import contextlib

        @contextlib.contextmanager
        def global_context_capture_fn(frame_summary):
            if frame_summary is not None:
                seen_frames.append(frame_summary)
            yield

        with mock.patch(
            "torch._guards.TracingContext.current_frame",
            side_effect=global_context_capture_fn,
        ):
            # 优化并执行函数 fn，捕获调用栈帧信息
            torch._dynamo.optimize("eager")(fn)(x, y, z)

        # 断言确保只捕获到一个调用栈帧
        self.assertEqual(len(seen_frames), 1)
        # 断言栈帧的名称和行为正确
        self.assertEqual(seen_frames[0].name, "fn")
        self.assertEqual(seen_frames[0].line, "r, r2 = uwu_inline_me(x, y, z)")
    def test_torch_guards_stack_frame_register_inlining_deep(self):
        # 创建张量 x，包含两个元素 0.5
        x = torch.tensor([0.5, 0.5])
        # 创建张量 y，包含四个元素 0.75
        y = torch.tensor([0.75, 0.75, 0.75, 0.75])
        # 创建张量 z，包含八个元素 0.25
        z = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

        # 定义深层内联函数 uwu_inline_me_deep，将 x 重复连接后与 y 相加
        def uwu_inline_me_deep(x, y):
            return torch.cat((x, x)) + y

        # 定义内联函数 uwu_inline_me，分别计算 uwu_inline_me_deep(x, y) 和 uwu_inline_me_deep(y, z) 的结果并返回
        def uwu_inline_me(x, y, z):
            r = uwu_inline_me_deep(x, y)
            r2 = uwu_inline_me_deep(y, z)
            return r, r2

        # 定义函数 fn，调用 uwu_inline_me 并对其结果进行平方操作后返回
        def fn(x, y, z):
            r, r2 = uwu_inline_me(x, y, z)
            return torch.mul(r, r), torch.mul(r2, r2)

        # 初始化一个空列表 seen_frames，用于记录帧信息
        seen_frames = []
        # 导入 contextlib 模块，用于管理上下文
        import contextlib

        # 定义上下文管理器 global_context_capture_fn，用于捕获帧摘要信息并存储在 seen_frames 中
        @contextlib.contextmanager
        def global_context_capture_fn(frame_summary):
            if frame_summary is not None:
                seen_frames.append(frame_summary)
            yield

        # 使用 mock.patch 来替换 torch._guards.TracingContext.current_frame 方法，使用 global_context_capture_fn 作为其副作用函数
        with mock.patch(
            "torch._guards.TracingContext.current_frame",
            side_effect=global_context_capture_fn,
        ):
            # 调用 torch._dynamo.optimize("eager")(fn)(x, y, z)，执行优化后的函数 fn
            torch._dynamo.optimize("eager")(fn)(x, y, z)

        # 断言 seen_frames 列表长度为 3
        self.assertEqual(len(seen_frames), 3)
        # 断言 seen_frames 中第一个帧的名称为 "fn"
        self.assertEqual(seen_frames[0].name, "fn")
        # 断言 seen_frames 中第二个帧的名称为 "uwu_inline_me"
        self.assertEqual(seen_frames[1].name, "uwu_inline_me")
        # 断言 seen_frames 中第三个帧的行为 "r2 = uwu_inline_me_deep(y, z)"
        self.assertEqual(seen_frames[2].line, "r2 = uwu_inline_me_deep(y, z)")

    def test_error_on_recompile(self):
        # 使用 torch._dynamo.optimize("eager") 修饰函数 fn，将其优化为 "eager" 模式
        @torch._dynamo.optimize("eager")
        def fn(a, b):
            return a + b

        # 使用 unittest.mock.patch 来设置 torch._dynamo.config.error_on_recompile 为 True
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            # 在调用 fn(torch.rand(2, 3), torch.rand(2, 3)) 时，期望引发 torch._dynamo.exc.RecompileError 异常
            with self.assertRaises(torch._dynamo.exc.RecompileError):
                fn(torch.rand(2, 3), torch.rand(2, 3))
                # 在调用 fn(torch.rand(2, 3), (1, 2, 3)) 时，期望引发 torch._dynamo.exc.RecompileError 异常
                fn(torch.rand(2, 3), (1, 2, 3))

    @expectedFailureDynamic
    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    # 定义一个测试方法，用于测试编译性能分析器
    def test_compile_profiler(self):
        # 定义一个简单的 PyTorch 模型类，重写了 forward 方法
        class Model(torch.nn.Module):
            def forward(self, input):
                return input + input

        # 创建一个 Model 实例
        model = Model()
        # 创建一个编译性能分析器实例
        prof = CompileProfiler()
        # 使用编译函数将模型编译，使用 prof 作为后端
        compiled = torch.compile(model, backend=prof)
        
        # 定义一个基本的检查器，用于检查编译后的模型运行结果
        base_checker = (
            lambda: FileCheck()
            .check("Torchdynamo Profiler Report")  # 检查报告标题
            .check("Graph Breaks")  # 检查图的断裂情况
            .check("No graph breaks detected.")  # 检查是否没有图断裂
            .check("Recompilation")  # 检查是否重新编译
        )
        
        # 创建一个输入张量
        input = torch.rand((2, 3, 4))
        # 运行编译后的模型
        _ = compiled(input)
        # 运行基本检查器，检查不重新编译的情况，并输出性能分析报告
        base_checker().check("No recompilation detected.").run(prof.report())

        # 创建一个新形状的输入张量
        new_shape_input = torch.rand((3, 3, 4))
        # 使用新形状的输入运行编译后的模型
        _ = compiled(new_shape_input)

        # 如果默认假设静态形状，运行基本检查器，并检查重新编译的原因
        if torch._dynamo.config.assume_static_by_default:
            base_checker().check("Recompile Reasons").check("'forward'").check(
                "cache_size_limit to 1"
            ).run(prof.report())
        else:
            # 否则运行基本检查器，检查不重新编译的情况，并输出性能分析报告
            base_checker().check("No recompilation detected.").run(prof.report())

        # 更新输入张量的形状
        new_shape_input = torch.rand((4, 3, 4))
        # 运行编译后的模型
        _ = compiled(new_shape_input)

        # 运行基本检查器，检查重新编译的原因，并输出性能分析报告
        base_checker().check("Recompile Reasons").check("'forward'").check(
            "tensor 'L['input']' size mismatch at index 0. expected 2, actual 3"
        ).check(
            "tensor 'L['input']' size mismatch at index 0. expected 3, actual 4"
        ).run(
            prof.report()
        )

    # 定义一个测试方法，测试 guards 模块中的 strip_function_call 函数
    def test_guards_strip_function_call(self):
        # 导入 strip_function_call 函数
        from torch._dynamo.guards import strip_function_call

        # 测试用例列表，每个元素包含一个待测试的字符串和预期的对象
        test_case = [
            ("___odict_getitem(a, 1)", "a"),
            ("a.layers[slice(2)][0]._xyz", "a"),
            ("getattr(a.layers[slice(2)][0]._abc, '0')", "a"),
            ("getattr(getattr(a.x[3], '0'), '3')", "a"),
            ("a.layers[slice(None, -1, None)][0]._xyz", "a"),
            ("a.layers[func('offset', -1, None)][0]._xyz", "a"),
        ]
        
        # 对每个测试用例执行 strip_function_call 函数，并进行断言
        for name, expect_obj in test_case:
            self.assertEqual(strip_function_call(name), expect_obj)

    # 定义一个测试方法，测试 int_neg 函数
    def test_int_neg(self):
        # 定义一个内部函数 int_neg，接受两个参数 a 和 b
        def int_neg(a, b):
            # 计算两个输入张量的形状大小乘积的负值
            x = a.shape[0]
            y = b.shape[0]
            return -x * -y * a * b
        
        # 使用 torch._dynamo.testing.standard_test 运行标准测试
        torch._dynamo.testing.standard_test(self, int_neg, 2)
    def test_hash_getitem_slice(self):
        # 创建 GetItemSource 对象 s，用于处理 LocalSource 中的 "foo"，使用 slice(None, -1, None) 进行切片
        s = GetItemSource(LocalSource("foo"), slice(None, -1, None))
        # 创建 GetItemSource 对象 s2，与 s 相同
        s2 = GetItemSource(LocalSource("foo"), slice(None, -1, None))
        # 创建 GetItemSource 对象 s3，使用不同的切片参数 slice(None, -1, 2)
        s3 = GetItemSource(LocalSource("foo"), slice(None, -1, 2))
        some_set = set()

        # 断言 s 不在 some_set 中
        self.assertTrue(s not in some_set)
        # 断言 s2 不在 some_set 中
        self.assertTrue(s2 not in some_set)
        # 断言 s3 不在 some_set 中
        self.assertTrue(s3 not in some_set)

        # 将 s 添加到 some_set 中
        some_set.add(s)

        # 断言 s 在 some_set 中
        self.assertTrue(s in some_set)
        # 断言 s2 也在 some_set 中（s 和 s2 应该哈希相同）
        self.assertTrue(s2 in some_set)
        # 断言 s3 不在 some_set 中（s3 应该哈希不同）
        self.assertTrue(s3 not in some_set)

        # 断言 s 等于 s2
        self.assertTrue(s == s2)
        # 断言 s 不等于 s3
        self.assertTrue(s != s3)

    def test_inline_dict_function(self):
        # 定义内联函数 _result_type_dict，根据 dtype 返回一个字典 {bool: torch.float32}
        def _result_type_dict(dtype):
            return {bool: torch.float32}[dtype]

        # 使用 @torch.compile 修饰的函数 f，返回 torch.ones(3, dtype=_result_type_dict(bool))
        @torch.compile
        def f():
            return torch.ones(3, dtype=_result_type_dict(bool))

        # 断言调用 f() 返回 torch.ones(3, dtype=torch.float32)
        self.assertEqual(f(), torch.ones(3, dtype=torch.float32))

    def test_inline_dict_function_passed_as_arg(self):
        # 使用 @torch.compile 修饰的函数 fn，根据 d[x] 的值选择执行 y.cos() 或者 y.sin()
        @torch.compile
        def fn(d, x, y):
            if d[x] is torch.float32:
                return y.cos()
            else:
                return y.sin()

        # 定义字典 dd，包含 {bool: torch.float32, int: torch.int64}
        dd = {bool: torch.float32, int: torch.int64}
        # 断言 fn(dd, bool, torch.ones(4)) 返回 torch.ones(4).cos()
        self.assertEqual(fn(dd, bool, torch.ones(4)), torch.ones(4).cos())
        # 断言 fn(dd, int, torch.ones(4)) 返回 torch.ones(4).sin()
        self.assertEqual(fn(dd, int, torch.ones(4)), torch.ones(4).sin())

    def test_add_sizes(self):
        # 定义函数 func，获取输入张量 x 的大小，返回 y + y
        def func(x):
            y = x.size()
            return y + y

        # 使用 torch._dynamo.optimize("eager") 优化 func，然后执行 torch.ones(10, 10, 3) 的计算
        eager_out = func(torch.ones(10, 10, 3))
        # 断言优化后的结果 compile_out 是 torch.Size 类型
        compile_out = torch._dynamo.optimize("eager")(func)(torch.ones(10, 10, 3))
        self.assertTrue(isinstance(compile_out, torch.Size))
        # 断言 eager_out 等于 compile_out
        self.assertEqual(eager_out, compile_out)

    @unittest.skipIf(not TEST_MULTIGPU, "need multiple GPU")
    def test_cuda_set_device(self):
        # 定义函数 fn，设置设备为 "cuda"，然后返回 a + 1
        def fn():
            a = torch.ones(2, device="cuda")
            torch.cuda.set_device(1)
            return a + 1

        # 使用 torch.cuda.device(0) 设置当前设备为 GPU 0
        with torch.cuda.device(0):
            counter = CompileCounter()
            # 优化 fn 函数，记录优化帧数到 counter
            opt_fn = torch._dynamo.optimize(counter)(fn)
            res = opt_fn()
            # 断言 res 的设备类型为 "cuda"，索引为 0
            self.assertEqual(res.device.type, "cuda")
            self.assertEqual(res.device.index, 0)
            # 断言优化帧数为 2
            self.assertEqual(counter.frame_count, 2)

    def test_nested_function_resuming_with_correct_globals(self):
        # 导入 outer_func 函数
        try:
            from .utils import outer_func
        except ImportError:
            from utils import outer_func

        # 定义函数 gn，返回 x + y
        def gn(x, y):
            return x + y

        # 定义函数 fn，调用 outer_func(gn) 处理 x 和 y
        def fn(x, y):
            return outer_func(gn)(x, y)

        # 定义输入张量 x 和 y
        x = torch.rand([3])
        y = torch.rand([3])
        # 使用 torch.compile(backend="eager") 优化 fn 函数
        opt_fn = torch.compile(backend="eager")(fn)
        # 计算未优化和优化后的结果
        ref = fn(x, y)
        res = opt_fn(x, y)
        # 断言 ref 和 res 结果相同
        self.assertTrue(same(ref, res))

    @dataclasses.dataclass
    class CSETestCase:
        expr: str
        preface: typing.List[str] = dataclasses.field(default_factory=list)
        expected: typing.Optional[str] = None
        expected_py38: typing.Optional[str] = None
        # 定义一个测试用例类，包含表达式、前置语句列表、预期输出及 Python 3.8 预期输出

    def _is_py38(self) -> bool:
        return sys.version_info[:2] <= (3, 8)
        # 检查当前 Python 版本是否为 3.8 及以下

    def _has_ast_unparse(self) -> bool:
        from torch._dynamo.guards import HAS_UNPARSE_FUNCTIONS
        return HAS_UNPARSE_FUNCTIONS
        # 检查是否存在 astunparse 函数或者 Python 3.9+

    def test_guards_cse_pass_single(self):
        if not self._has_ast_unparse():
            if IS_FBCODE:
                raise RuntimeError("Needs astunparse or Python-3.9+")
            raise unittest.SkipTest("Needs astunparse or Python-3.9+")
        # 如果没有 astunparse 函数且不是 Python 3.9+，则抛出运行时错误或跳过测试

        from torch._dynamo.guards import PyExprCSEPass
        # 导入 PyExprCSEPass 类

        testcase = self.CSETestCase
        # 定义一个测试用例的别名

        testcases = [
            # 测试用例列表，每个测试用例包含表达式、前置语句和预期输出
            # 第一个测试用例，表达式 "x[0].a"，无需前置语句，没有预期输出变化
            testcase(expr="x[0].a"),
            testcase(expr="x[1].a"),
            testcase(expr="x[2].a"),

            # 第二个测试用例，表达式 "a.b.c[0].d.e"，有前置语句，预期输出为 "_var1[0].d.e"
            testcase(
                expr="a.b.c[0].d.e",
                preface=["_var0 = a.b", "_var1 = _var0.c"],
                expected="_var1[0].d.e",
            ),
            testcase(expr="a.b.c[1].d.e", expected="_var1[1].d.e"),
            testcase(expr="a.b.c[2].d.e", expected="_var1[2].d.e"),

            # 第三个测试用例，表达式 "f(m.n[0], '0').x.y.z"，有前置语句，预期输出为 "f(_var3, '0').x.y.z"
            testcase(
                expr="f(m.n[0], '0').x.y.z",
                preface=["_var2 = m.n", "_var3 = _var2[0]"],
                expected="f(_var3, '0').x.y.z",
            ),
            testcase(expr="f(m.n[0], '1').x.y.z", expected="f(_var3, '1').x.y.z"),
            testcase(expr="f(m.n[0], '2').x.y.z", expected="f(_var3, '2').x.y.z"),

            # 第四个测试用例，表达式 "self.g(a, b).k"，有前置语句，预期输出为 "_var6"
            testcase(
                expr="self.g(a, b).k",
                preface=["_var4 = self.g", "_var5 = _var4(a, b)", "_var6 = _var5.k"],
                expected="_var6",
            ),
            testcase(expr="self.g(a, b).k", expected="_var6"),
            testcase(expr="self.g(a, b).k", expected="_var6"),
        ]

        csepass = PyExprCSEPass()
        csepass.count([t.expr for t in testcases])
        # 创建 PyExprCSEPass 实例，并统计测试用例列表中的表达式数量

        for t in testcases:
            preface, expr = csepass.replace(t.expr)
            # 对每个测试用例中的表达式进行替换操作，返回替换后的前置语句和表达式
            self.assertEqual(preface, t.preface)
            # 断言替换后的前置语句与预期相符
            expected = t.expected if t.expected is not None else t.expr
            self.assertEqual(expr, expected)
            # 断言替换后的表达式与预期输出相符
    # 如果没有安装 astunparse 或者 Python 版本低于 3.9，则跳过测试
    def test_guards_cse_pass_multiple(self):
        if not self._has_ast_unparse():
            raise unittest.SkipTest("Needs astunparse or Python-3.9+")
        
        # 导入 PyExprCSEPass 类
        from torch._dynamo.guards import PyExprCSEPass

        # 定义测试用例
        testcase = self.CSETestCase
        testcases = [
            testcase(
                expr="x[0].a < x[1].a * (3 - x[2].a)",
                expected="x[0].a < x[1].a * (3 - x[2].a)",
                expected_py38="(x[0].a < (x[1].a * (3 - x[2].a)))",
            ),
            testcase(
                expr="a.b.c[0].d.e + a.b.c[1].d.e * a.b.c[2].d.e > 0",
                preface=["_var0 = a.b", "_var1 = _var0.c"],
                expected="_var1[0].d.e + _var1[1].d.e * _var1[2].d.e > 0",
                expected_py38="((_var1[0].d.e + (_var1[1].d.e * _var1[2].d.e)) > 0)",
            ),
            testcase(
                expr="f(m.n[0], '0').x.y.z * f(m.n[0], '1').x.y.z * f(m.n[0], '2').x.y.z < 512",
                preface=["_var2 = m.n", "_var3 = _var2[0]"],
                expected="f(_var3, '0').x.y.z * f(_var3, '1').x.y.z * f(_var3, '2').x.y.z < 512",
                expected_py38="(((f(_var3, '0').x.y.z * f(_var3, '1').x.y.z) * f(_var3, '2').x.y.z) < 512)",
            ),
            testcase(
                expr="self.g(a, b).k + (1 - self.g(a, b).k) <= m[0].a + self.g(a, b).k",
                preface=["_var4 = self.g", "_var5 = _var4(a, b)", "_var6 = _var5.k"],
                expected="_var6 + (1 - _var6) <= m[0].a + _var6",
                expected_py38="((_var6 + (1 - _var6)) <= (m[0].a + _var6))",
            ),
        ]

        # 创建 PyExprCSEPass 实例
        csepass = PyExprCSEPass()

        # 计算表达式的公共子表达式
        csepass.count([t.expr for t in testcases])

        # 遍历测试用例列表
        for t in testcases:
            # 替换表达式中的公共子表达式
            preface, expr = csepass.replace(t.expr)
            
            # 断言替换后的表达式和预期结果相符
            self.assertEqual(preface, t.preface)
            expected = t.expected_py38 if self._is_py38() else t.expected
            expected = expected if expected is not None else t.expr
            self.assertEqual(expr, expected)

    # 测试带有公共子表达式消除的 guard 函数生成器
    def test_guard_function_builder_with_cse(self):
        # 导入 build_guard_function 函数
        from torch._dynamo.guards import build_guard_function

        # 定义表达式列表
        exprs = [
            "x[0].a < x[1].a * (3 - x[2].a)",
            "a.b.c[0].d.e + a.b.c[1].d.e * a.b.c[2].d.e > 0",
            "f(m.n[0], '0').x.y.z * f(m.n[0], '1').x.y.z * f(m.n[0], '2').x.y.z < 512",
            "self.g(a, b).k + (1 - self.g(a, b).k) <= m[0].a + self.g(a, b).k",
        ]

        # 调用 build_guard_function 函数
        _, pycode = build_guard_function(exprs, "")

        expected = """\
def ___make_guard_fn():
    # 定义内部函数 guard，接受参数 L
    def guard(L):
        # 检查条件，如果不满足，则返回 False
        if not (x[0].a < x[1].a * (3 - x[2].a)):
            return False
        # 获取变量 _var0，并访问其属性 a.b，将结果赋给 _var0
        _var0 = a.b
        # 获取 _var0 的属性 c，将结果赋给 _var1
        _var1 = _var0.c
        # 检查条件，如果不满足，则返回 False
        if not (_var1[0].d.e + _var1[1].d.e * _var1[2].d.e > 0):
            return False
        # 获取变量 _var2，并访问其属性 m.n，将结果赋给 _var2
        _var2 = m.n
        # 获取 _var2 的第一个元素，将结果赋给 _var3
        _var3 = _var2[0]
        # 检查条件，如果不满足，则返回 False
        if not (f(_var3, '0').x.y.z * f(_var3, '1').x.y.z * f(_var3, '2').x.y.z < 512):
            return False
        # 获取 self 的属性 g，并调用它以参数 a 和 b，将结果赋给 _var4
        _var4 = self.g
        # 调用 _var4 返回的对象的属性 k，将结果赋给 _var5
        _var5 = _var4(a, b)
        # 获取 _var5 的属性 k，将结果赋给 _var6
        _var6 = _var5.k
        # 检查条件，如果不满足，则返回 False
        if not (_var6 + (1 - _var6) <= m[0].a + _var6):
            return False
        # 如果所有条件都满足，则返回 True
        return True
    # 返回内部函数 guard 的引用
    return guard
    def test_scalar_tensor_is_equivalent_to_symint_argument(self):
        # 定义一个名为 GumbelTopKSampler 的类，继承自 torch.nn.Module
        class GumbelTopKSampler(torch.nn.Module):
            # 类的初始化函数，接受 T 和 k 两个参数
            def __init__(self, T, k):
                super().__init__()
                # 使用 T 初始化一个不可梯度更新的参数 self.T，数据类型为 torch.float32
                self.T = torch.nn.Parameter(
                    torch.tensor(T, dtype=torch.float32), requires_grad=False
                )
                # 使用 k 初始化一个不可梯度更新的参数 self.k，数据类型为 torch.int32
                self.k = torch.nn.Parameter(
                    torch.tensor(k, dtype=torch.int32), requires_grad=False
                )

            # 定义一个方法 sample_discrete，接受 logits 作为输入
            def sample_discrete(self, logits):
                # 计算 logits 中前 k 大的值，并取最小的值作为阈值 threshold
                threshold = torch.topk(logits, self.k, sorted=True)[0][..., -1]
                # 将 logits 中大于等于阈值的部分置为1，其余为0，并转换为浮点数
                samples = torch.ge(logits.squeeze(1), threshold).float()
                return samples

            # 定义前向传播方法 forward，接受 logits 作为输入
            def forward(self, logits):
                # 调用 sample_discrete 方法获取离散样本
                dsamples = self.sample_discrete(logits)
                return dsamples

        # 创建一个 4x4x4x4 的随机张量 x
        x = torch.rand([4, 4, 4, 4])
        # 使用 T=4 和 k=4 初始化 GumbelTopKSampler 类的实例 m
        m = GumbelTopKSampler(T=4, k=4)
        # 使用 m 对 x 进行前向传播，得到原始输出 orig_out
        orig_out = m(x)
        # 使用 torch.compile 函数将模型 m 编译成 eager 模式下的优化模型 opt_m
        opt_m = torch.compile(backend="eager")(m)
        # 使用优化模型 opt_m 对 x 进行前向传播，得到优化输出 opt_out
        opt_out = opt_m(x)
        # 断言原始输出 orig_out 与优化输出 opt_out 相同
        self.assertTrue(same(orig_out, opt_out))

    def test_scalar_tensor_is_equivalent_to_symint_list_argument(self):
        # 定义一个名为 Jitter 的类，继承自 torch.nn.Module
        class Jitter(torch.nn.Module):
            # 类的初始化函数，接受 jitter_val 参数
            def __init__(self, jitter_val):
                super().__init__()
                # 将 jitter_val 存储在 self.jitter_val 中
                self.jitter_val = jitter_val

            # 定义一个方法 roll_tensor，接受 input 作为输入
            def roll_tensor(self, input):
                # 计算水平和垂直偏移量
                h_shift = self.jitter_val - 1
                w_shift = self.jitter_val + 1
                # 对输入张量 input 进行水平和垂直方向的滚动操作
                return torch.roll(
                    torch.roll(input, shifts=h_shift, dims=2), shifts=w_shift, dims=3
                )

            # 定义前向传播方法 forward，接受 input 作为输入
            def forward(self, input):
                # 调用 roll_tensor 方法处理输入 input
                return self.roll_tensor(input)

        # 创建一个 4x4x4x4 的随机张量 x
        x = torch.rand([4, 4, 4, 4])
        # 使用 jitter_val=4 初始化 Jitter 类的实例 m
        m = Jitter(jitter_val=4)
        # 使用 m 对 x 进行前向传播，得到原始输出 orig_out
        orig_out = m(x)
        # 使用 torch.compile 函数将模型 m 编译成 eager 模式下的优化模型 opt_m
        opt_m = torch.compile(backend="eager")(m)
        # 使用优化模型 opt_m 对 x 进行前向传播，得到优化输出 opt_out
        opt_out = opt_m(x)
        # 断言原始输出 orig_out 与优化输出 opt_out 相同
        self.assertTrue(same(orig_out, opt_out))

    def test_scalar_tensor_is_equivalent_to_int_list_argument(self):
        # 定义一个名为 MyModel 的类，继承自 torch.nn.Module
        class MyModel(torch.nn.Module):
            # 定义前向传播方法 forward，接受 input 作为输入
            def forward(self, input):
                # 创建一个固定的排列顺序 permute，然后对输入 input 进行置换操作
                permute = torch.tensor([0, 2, 1])
                x = input.permute(*permute)
                return x

        # 创建一个形状为 2x3x4 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 创建 MyModel 类的实例 m
        m = MyModel()
        # 使用 m 对 x 进行前向传播，得到原始输出 orig_out
        orig_out = m(x)
        # 使用 torch.compile 函数将模型 m 编译成 eager 模式下的优化模型 opt_m
        opt_m = torch.compile(backend="eager")(m)
        # 使用优化模型 opt_m 对 x 进行前向传播，得到优化输出 opt_out
        opt_out = opt_m(x)
        # 断言原始输出 orig_out 与优化输出 opt_out 相同
        self.assertTrue(same(orig_out, opt_out))

    def test_torch_variable_hasattr(self):
        # 定义一个名为 fn 的函数，接受 x 作为输入
        def fn(x):
            # 检查 torch.nn 模块中是否存在名为 "Module" 的属性
            if hasattr(torch.nn, "Module"):
                return x * x  # 如果存在，则返回 x 的平方
            return x + 1  # 否则返回 x 加 1

        # 使用 torch.compile 函数将函数 fn 编译成 eager 模式下的优化函数 compiled_fn
        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        # 创建一个形状为 4x4 的随机张量 x
        x = torch.rand([4, 4])
        # 调用原始函数 fn，计算 fn(x) 的输出 fn_out
        fn_out = fn(x)
        # 调用优化函数 compiled_fn，计算 compiled_fn(x) 的输出 compiled_out
        compiled_out = compiled_fn(x)
        # 断言原始输出 fn_out 与优化输出 compiled_out 相同
        self.assertTrue(same(fn_out, compiled_out))
    # 定义一个测试函数，用于测试具有属性 "foo" 的对象
    def test_list_hasattr1(self):
        # 定义一个内部函数 fn，接受参数 x
        def fn(x):
            # 如果对象 x 有属性 "foo"
            if hasattr(x, "foo"):
                # 返回 x 中第一个元素加 1 的结果
                return x[0] + 1
            # 否则返回 x 中第一个元素减 1 的结果
            return x[0] - 1

        # 使用 torch 的编译器，将 fn 编译为 Torch 的计算图形式
        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        # 创建一个包含随机数张量的列表 x
        x = [torch.randn(3)]
        # 调用 fn 函数，传入 x，获取其输出结果
        fn_out = fn(x)
        # 调用经编译后的 compiled_fn 函数，传入 x，获取其输出结果
        compiled_out = compiled_fn(x)
        # 断言 fn_out 和 compiled_out 相同
        self.assertTrue(same(fn_out, compiled_out))

    # 定义一个测试函数，用于测试对象是否具有 "__len__" 属性
    def test_list_hasattr2(self):
        # 定义一个不接受参数的内部函数 fn
        def fn():
            # 创建包含三个零张量的列表 x
            x = [torch.zeros(3)]
            # 如果 x 对象具有 "__len__" 属性
            if hasattr(x, "__len__"):
                # 返回 x 中第一个元素加 1 的结果
                return x[0] + 1
            # 否则返回 x 中第一个元素减 1 的结果
            return x[0] - 1

        # 使用 torch 的编译器，将 fn 编译为 Torch 的计算图形式
        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        # 调用 fn 函数，获取其输出结果
        fn_out = fn()
        # 调用经编译后的 compiled_fn 函数，获取其输出结果
        compiled_out = compiled_fn()
        # 断言 fn_out 和 compiled_out 相同
        self.assertTrue(same(fn_out, compiled_out))

    # 定义一个测试函数，用于测试具有属性 "foo" 的对象
    def test_tuple_hasattr(self):
        # 定义一个接受参数 x 的内部函数 fn
        def fn(x):
            # 如果对象 x 有属性 "foo"
            if hasattr(x, "foo"):
                # 返回 x 中第一个元素加 1 的结果
                return x[0] + 1
            # 否则返回 x 中第二个元素减 1 的结果
            return x[1] - 1

        # 使用 torch 的编译器，将 fn 编译为 Torch 的计算图形式
        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        # 创建包含两个随机数张量的元组 x
        x = (torch.randn(3), torch.randn(3))
        # 调用 fn 函数，传入 x，获取其输出结果
        fn_out = fn(x)
        # 调用经编译后的 compiled_fn 函数，传入 x，获取其输出结果
        compiled_out = compiled_fn(x)
        # 断言 fn_out 和 compiled_out 相同
        self.assertTrue(same(fn_out, compiled_out))

    # 定义一个测试函数，用于测试 lambda 函数是否具有 "__name__" 属性
    def test_fn_hasattr__name__1(self):
        # 定义一个不接受参数的内部函数 fn
        def fn():
            # 创建一个 lambda 函数 foo，使其返回参数 x 加 1
            foo = lambda x: x + 1
            # 返回 lambda 函数 foo 是否具有 "__name__" 属性的结果
            return hasattr(foo, "__name__")

        # 使用 torch 的编译器，将 fn 编译为 Torch 的计算图形式
        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        # 调用 fn 函数，获取其输出结果
        fn_out = fn()
        # 调用经编译后的 compiled_fn 函数，获取其输出结果
        compiled_out = compiled_fn()
        # 断言 fn_out 和 compiled_out 相同，并且 fn_out 为 True
        self.assertEqual(fn_out, compiled_out)
        self.assertTrue(fn_out)

    # 定义一个测试函数，用于测试普通函数是否具有 "__name__" 属性
    def test_fn_hasattr__name__2(self):
        # 定义一个接受参数 x 的函数 bar，返回参数 x 的正弦值
        def bar(x):
            return torch.sin(x)

        # 定义一个不接受参数的内部函数 fn
        def fn():
            # 返回函数 bar 是否具有 "__name__" 属性的结果
            return hasattr(bar, "__name__")

        # 使用 torch 的编译器，将 fn 编译为 Torch 的计算图形式
        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        # 调用 fn 函数，获取其输出结果
        fn_out = fn()
        # 调用经编译后的 compiled_fn 函数，获取其输出结果
        compiled_out = compiled_fn()
        # 断言 fn_out 和 compiled_out 相同，并且 fn_out 为 True
        self.assertEqual(fn_out, compiled_out)
        self.assertTrue(fn_out)

    # 定义一个测试函数，用于测试偏函数是否具有 "__name__" 属性
    def test_fn_hasattr__name__3(self):
        # 定义一个接受参数 x 和 y 的函数 bar，返回参数 x 的正弦值加上参数 y 的余弦值
        def bar(x, y):
            return torch.sin(x) + torch.cos(y)

        # 使用 functools 的 partial 函数创建一个偏函数 baz，固定参数 y 为 4
        baz = functools.partial(bar, y=4)

        # 定义一个不接受参数的内部函数 fn
        def fn():
            # 返回偏函数 baz 是否具有 "__name__" 属性的结果
            return hasattr(baz, "__name__")

        # 使用 torch 的编译器，将 fn 编译为 Torch 的计算图形式
        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        # 调用 fn 函数，获取其输出结果
        fn_out = fn()
        # 调用经编译后的 compiled_fn 函数，获取其输出结果
        compiled_out = compiled_fn()
        # 断言 fn_out 和 compiled_out 相同，并且 fn_out 为 False
        self.assertEqual(fn_out, compiled_out)
        self.assertFalse(fn_out)

    # 定义一个测试函数，用于测试将 torch 对象作为键的字典
    def test_torch_objects_as_keys(self):
        # 创建一个字典 remap，将 torch.float16 映射为 torch.float32
        remap = {torch.float16: torch.float32}

        # 定义一个不接受参数的内部函数 fn
        def fn():
            # 返回一个具有指定数据类型的随机数张量
            return torch.randn(3, dtype=remap[torch.float16])

        # 使用 torch 的优化器，对 fn 进行优化
        opt = torch._dynamo.optimize("eager")(fn)
        # 执行优化后的函数
        opt()

    # 定义一个测试函数，用于测试跟踪 Python 树结构
    def test_tracing_py_tree(self):
        # 定义一个接受参数 xs 的内部函数 fn
        def fn(xs):
            # 将输入的 Python 树结构 xs 展平，返回展平后的列表 flat_xs 和结构 spec
            flat_xs, spec = pytree.tree_flatten(xs)
            # 创建一个列表 res，其中每个元素是 flat_xs 中对应元素的克隆
            res = [x.clone() for x in flat_xs]
            # 根据结构 spec 将列表 res 还原为原始的 Python 树结构
            return pytree.tree_unflatten(res, spec)

        # 创建一个包含三个张量的列表 xs
        xs = [torch.tensor(i) for i in range(3)]

        # 创建一个编译计数器对象
        counter = CompileCounter()
        # 使用 torch 的动态编译器，对 fn 进行优化，并设置
    def test_tracing_nested_py_tree(self):
        # 导入 torch.utils._pytree 库，用于处理树形数据结构
        import torch.utils._pytree as pytree

        # 定义函数 fn，对输入的树形数据 xs 进行操作
        def fn(xs):
            # 将树形数据 xs 展平，并获取展平后的列表 flat_xs 和树的结构 spec
            flat_xs, spec = pytree.tree_flatten(xs)
            # 对展平后的每个张量 x 进行克隆操作，得到结果列表 res
            res = [x.clone() for x in flat_xs]
            # 根据结构 spec 将结果列表 res 还原成原始的树形数据结构
            return pytree.tree_unflatten(res, spec)

        # 创建包含张量的列表 xs，每个张量包含从 0 到 2 的整数
        xs = [torch.tensor(i) for i in range(3)]
        # 创建包含多个相同的 xs 列表的列表 xsl
        xsl = [xs, xs, xs, xs]

        # 创建计数器对象 CompileCounter 的实例
        counter = CompileCounter()
        # 使用 torch._dynamo.optimize 进行优化编译 fn 函数，设置 nopython=True
        comp_out = torch._dynamo.optimize(counter, nopython=True)(fn)(xsl)
        # 直接调用 fn 函数处理 xsl，得到真实输出 real_out
        real_out = fn(xsl)
        # 断言优化编译后的输出 comp_out 与真实输出 real_out 相等
        self.assertEqual(comp_out, real_out)
        # 断言计数器的 frame_count 属性为 1
        self.assertEqual(counter.frame_count, 1)
        # 断言计数器的 op_count 属性为 12
        self.assertEqual(counter.op_count, 12)

    def test_tracing_nested_py_tree_tuples(self):
        # 导入 torch.utils._pytree 库，用于处理树形数据结构
        import torch.utils._pytree as pytree

        # 定义函数 fn，对输入的树形数据 xs 进行操作
        def fn(xs):
            # 将树形数据 xs 展平，并获取展平后的列表 flat_xs 和树的结构 spec
            flat_xs, spec = pytree.tree_flatten(xs)
            # 对展平后的每个张量 x 进行克隆操作，得到结果列表 res
            res = [x.clone() for x in flat_xs]
            # 根据结构 spec 将结果列表 res 还原成原始的树形数据结构
            return pytree.tree_unflatten(res, spec)

        # 创建包含张量的列表 xs，每个张量包含从 0 到 2 的整数
        xs = [torch.tensor(i) for i in range(3)]
        # 创建包含多个相同的 xs 列表的元组 xsl
        xsl = (xs, xs, xs, xs)

        # 创建计数器对象 CompileCounter 的实例
        counter = CompileCounter()
        # 使用 torch._dynamo.optimize 进行优化编译 fn 函数，设置 nopython=True
        comp_out = torch._dynamo.optimize(counter, nopython=True)(fn)(xsl)
        # 直接调用 fn 函数处理 xsl，得到真实输出 real_out
        real_out = fn(xsl)
        # 断言优化编译后的输出 comp_out 与真实输出 real_out 相等
        self.assertEqual(comp_out, real_out)
        # 断言计数器的 frame_count 属性为 1
        self.assertEqual(counter.frame_count, 1)
        # 断言计数器的 op_count 属性为 12

    def test_tracing_nested_py_tree_dicts(self):
        # 导入 torch.utils._pytree 库，用于处理树形数据结构
        import torch.utils._pytree as pytree

        # 定义函数 fn，对输入的树形数据 xs 进行操作
        def fn(xs):
            # 将树形数据 xs 展平，并获取展平后的列表 flat_xs 和树的结构 spec
            flat_xs, spec = pytree.tree_flatten(xs)
            # 对展平后的每个张量 x 进行克隆操作，得到结果列表 res
            res = [x.clone() for x in flat_xs]
            # 根据结构 spec 将结果列表 res 还原成原始的树形数据结构
            return pytree.tree_unflatten(res, spec)

        # 创建包含张量的列表 xs，每个张量包含从 0 到 2 的整数
        xs = [torch.tensor(i) for i in range(3)]
        # 创建包含多个相同的 xs 列表的字典 xsl
        xsl = {
            "a": xs,
            "b": xs,
            "c": xs,
        }

        # 创建计数器对象 CompileCounter 的实例
        counter = CompileCounter()
        # 使用 torch._dynamo.optimize 进行优化编译 fn 函数，设置 nopython=True
        comp_out = torch._dynamo.optimize(counter, nopython=True)(fn)(xsl)
        # 直接调用 fn 函数处理 xsl，得到真实输出 real_out
        real_out = fn(xsl)
        # 断言优化编译后的输出 comp_out 与真实输出 real_out 相等
        self.assertEqual(comp_out, real_out)
        # 断言计数器的 frame_count 属性为 1
        self.assertEqual(counter.frame_count, 1)
        # 断言计数器的 op_count 属性为 9

    def test_dynamic_one_hot(self):
        # 定义函数 fn，对输入的张量 x 进行操作
        def fn(x):
            # 将张量 x 中的每个元素加 1
            x = x + 1
            # 使用 torch.nn.functional.one_hot 对 x 进行独热编码操作
            # 注意：这里的注释指出由于输出形状依赖于数据，可能会破坏图形
            x = torch.nn.functional.one_hot(x)
            # 将独热编码后的张量 x 中的每个元素再加 1
            x = x + 1
            return x

        # 创建一个张量 inp，其中包含从 0 到 19 的整数，并对 4 取模
        inp = torch.arange(20) % 4
        # 创建计数器对象 CompileCounter 的实例
        counter = CompileCounter()
        # 调用 fn 函数处理输入 inp，得到真实输出 real_out
        real_out = fn(inp)
        # 使用 torch.compile 对 fn 函数进行编译，使用计数器统计编译信息
        comp_out = torch.compile(fn, backend=counter)(inp)
        # 断言编译后的输出 comp_out 与真实输出 real_out 相等
        self.assertEqual(comp_out, real_out)
        # 断言计数器的 frame_count 属性为 2
        self.assertEqual(counter.frame_count, 2)
        # 断言计数器的 op_count 属性为 2
    def test_tracing_nested_py_tree_mixed_all(self):
        # 导入 torch.utils._pytree 模块
        import torch.utils._pytree as pytree

        # 定义函数 fn，接收一个参数 xs
        def fn(xs):
            # 将 xs 扁平化并返回扁平化后的列表 flat_xs 和结构 spec
            flat_xs, spec = pytree.tree_flatten(xs)
            # 对 flat_xs 中的每个元素进行克隆操作，存储在列表 res 中
            res = [x.clone() for x in flat_xs]
            # 根据 spec 将 res 恢复成原始结构并返回
            return pytree.tree_unflatten(res, spec)

        # 创建包含三个 torch.tensor 对象的列表 xs
        xs = [torch.tensor(i) for i in range(3)]
        # 创建包含 xs 两次的元组 xsa
        xsa = (xs, xs)
        # 创建包含 xsa 和 xs 的字典 xsb
        xsb = {"aa": xsa, "ab": xs}
        # 创建包含多层嵌套结构的字典 xsl
        xsl = {
            "a": xs,
            "b": xsa,
            "c": xsb,
        }

        # 创建 CompileCounter 实例
        counter = CompileCounter()
        # 对 fn 函数进行动态优化并执行，使用 xsl 作为参数
        comp_out = torch._dynamo.optimize(counter, nopython=True)(fn)(xsl)
        # 直接执行 fn 函数，使用 xsl 作为参数
        real_out = fn(xsl)
        # 断言动态优化后的输出与直接执行的输出相等
        self.assertEqual(comp_out, real_out)
        # 断言 frame_count 属性为 1
        self.assertEqual(counter.frame_count, 1)
        # 断言 op_count 属性为 18
        self.assertEqual(counter.op_count, 18)

    def test_any_all_symnode(self):
        # 创建 CompileCounter 实例
        cnt = CompileCounter()

        # 定义装饰器函数 fn，接收一个参数 x
        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def fn(x):
            # 判断 x.size(0) 是否大于等于 10，并赋值给 t
            t = x.size(0) >= 10
            # 判断 x.size(0) 是否大于等于 100，并赋值给 f
            f = x.size(0) >= 100
            # 如果传入的列表为空，或者列表中至少有一个元素为真，或者列表中至少有两个元素为真，则执行 x - 1 操作
            if any([]) or any([f]) or any([f, f]):
                return x - 1
            # 如果传入的列表中所有元素都为真，或者列表中所有元素都为真且至少有一个元素为真，或者其他组合情况，则执行 x - 2 操作
            if all([f]) or all([t, f]) or all([f, t]) or all([f, f]):
                return x - 2
            # 如果传入的列表为空，或者列表中所有元素都为假，或者其他组合情况，则执行 x - 3 操作
            if not (all([]) and all([t]) and all([t, t])):
                return x - 3
            # 如果传入的列表中至少有一个元素为真，或者列表中至少有一个元素为真且至少有一个元素为假，或者其他组合情况，则执行 x - 4 操作
            if not (any([t]) and any([t, f]) and any([f, t])):
                return x - 4
            # 否则执行 x + 1 操作
            return x + 1

        # 创建长度为 16 的随机张量 y1 和长度为 18 的随机张量 y2
        y1 = torch.randn(16)
        y2 = torch.randn(18)
        # 断言对 y1 和 y2 分别执行 fn 函数后的输出与预期值相等
        self.assertEqual(fn(y1), y1 + 1)
        self.assertEqual(fn(y2), y2 + 1)
        # 断言 frame_count 属性为 1
        self.assertEqual(cnt.frame_count, 1)
        # 创建长度为 5 的随机张量 y3
        y3 = torch.randn(5)
        # 断言对 y3 执行 fn 函数后的输出与预期值相等
        self.assertEqual(fn(y3), y3 - 3)
        # 断言 frame_count 属性为 2
        self.assertEqual(cnt.frame_count, 2)

    def test_tracing_py_tree_tensor_subclass(self):
        # 导入 torch.utils._pytree 模块和 TwoTensor 类
        import torch.utils._pytree as pytree
        from torch.testing._internal.two_tensor import TwoTensor
        # 导入 checkpoint 函数
        from torch.utils.checkpoint import checkpoint

        # 定义函数 fn，接收一个参数 xs
        def fn(xs):
            # 将 xs 嵌套在一个列表中并返回
            nested_xs = [[xs]]
            # 将 xs 扁平化并返回扁平化后的列表 flat_xs 和结构 spec
            flat_xs, spec = pytree.tree_flatten(xs)
            # 返回 flat_xs 中的第一个元素的克隆版本
            return flat_xs[0].clone()

        # 使用 checkpoint 函数包装 fn 函数，使其支持 "sourceless" tensor 子类
        def checkpoint_fn(xs):
            return checkpoint(fn, xs, use_reentrant=True)

        # 创建一个 TwoTensor 实例 xs，包含两个形状为 (2, 2) 的张量
        xs = TwoTensor(torch.ones(2, 2), torch.ones(2, 2))

        # 创建 CompileCounter 实例
        counter = CompileCounter()
        # 对 checkpoint_fn 函数进行动态优化并执行，使用 xs 作为参数
        torch._dynamo.optimize(counter, nopython=True)(checkpoint_fn)(xs)
        # 断言 frame_count 属性为 1
        self.assertEqual(counter.frame_count, 1)
        # 断言 op_count 属性为 2
        self.assertEqual(counter.op_count, 2)
    # 定义一个测试方法，用于测试“torch.utils._pytree”模块中的功能
    def test_tracing_tree_map_only(self):
        # 导入需要的模块或类
        import torch.utils._pytree as pytree

        # 定义一个函数fn，接受一个参数xs，并对其进行处理
        def fn(xs):
            # 定义一个内部函数mapper，用于对输入的x进行克隆操作
            def mapper(x):
                return x.clone()

            # 使用pytree模块中的tree_map_only函数，对xs中所有的torch.Tensor类型的元素应用mapper函数
            y = pytree.tree_map_only(torch.Tensor, mapper, xs)
            # 返回处理后的结果
            return y

        # 准备测试数据xs，包括3个torch.tensor和一个字符串"hi"组成的列表
        xs = [torch.tensor(i) for i in range(3)] + ["hi"]
        # 构建两个元组，每个元组包含相同的xs列表
        xsa = (xs, xs)
        # 构建一个字典，包含两个键值对，其中键是字符串，值是之前构建的xsa元组和xs列表
        xsb = {"aa": xsa, "ab": xs}

        # 创建一个编译计数器对象
        counter = CompileCounter()
        # 使用torch._dynamo.optimize进行优化，并传入编译计数器和参数fn，启用nopython模式进行编译
        comp_out = torch._dynamo.optimize(counter, nopython=True)(fn)(xsb)
        # 直接调用fn函数对xsb进行处理，得到真实的输出结果
        real_out = fn(xsb)

        # 断言编译后的输出与真实输出相等
        self.assertEqual(comp_out, real_out)
        # 断言frame_count计数为1
        self.assertEqual(counter.frame_count, 1)
        # 断言op_count计数为9
        self.assertEqual(counter.op_count, 9)

    # 标记为torch._dynamo.config.patch修饰的测试方法，用于测试未支持的符号整数计算
    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_unbacked_symint(self):
        # 定义一个函数f，接受lengths和values两个参数
        @torch.compile(backend="eager")
        def f(lengths, values):
            # 将lengths转换为列表sizes
            sizes = lengths.tolist()
            # 遍历sizes中的每个元素s，进行大小检查
            for s in sizes:
                torch._check_is_size(s)
                torch._check(s >= 2)
                torch._check(s <= 100)
            # 使用torch.split按照sizes对values进行分割
            return torch.split(values, sizes)

        # 调用f函数，传入测试数据
        f(torch.tensor([2, 3, 4]), torch.randn(9))

    # 标记为torch._dynamo.config.patch修饰的测试方法，用于测试自动生成功能化操作
    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_unbacked_auto_functionalize_op(self):
        # 定义一个自定义操作mk_image，接受decoder参数，并返回一个张量Tensor
        @torch.library.custom_op(
            "mylib::mk_image", mutates_args=("decoder",), device_types=["cpu"]
        )
        def mk_image(decoder: Tensor) -> Tensor:
            # 返回一个形状为[2, 3, 4, 5]的随机张量
            return torch.randn(2, 3, 4, 5)

        # 使用torch.library.register_fake注册一个假的操作"mylib::mk_image"
        @torch.library.register_fake("mylib::mk_image")
        def _(decoder: Tensor) -> Tensor:
            # 创建一个动态大小的图像张量，其大小由上下文中的new_dynamic_size()函数生成
            image_size = [torch.library.get_ctx().new_dynamic_size() for _ in range(4)]
            return torch.empty(image_size)

        # 使用torch.compile进行编译，fullgraph=True表示编译整个图形
        def f(x):
            return torch.ops.mylib.mk_image.default(x)

        # 准备测试数据x，一个形状为[100]的零张量
        x = torch.zeros(100, dtype=torch.int64)
        # 调用f函数，传入测试数据x
        f(x)
    # 定义测试函数 test_out_variant_custom_op
    def test_out_variant_custom_op(self):
        # 使用 torch 库的 _scoped_library 方法创建名为 "mylib" 的库，并指定库的作用域为 "FRAGMENT"
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            # 在库中定义一个名为 "split_with_sizes_copy" 的自定义操作
            lib.define(
                "split_with_sizes_copy(Tensor all_gather_output, SymInt[] all_gather_input_split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()"
            )

            # 在 "mylib" 库中实现名为 "split_with_sizes_copy" 的操作，支持 "Meta" 和 "CPU" 后端
            @torch.library.impl(lib, "split_with_sizes_copy", "Meta")
            @torch.library.impl(lib, "split_with_sizes_copy", "CPU")
            def split_with_sizes_copy(
                all_gather_output: torch.Tensor,
                all_gather_input_split_sizes: typing.List[int],
                dim: int,
                out: typing.List[torch.Tensor],
            ) -> None:
                # 调用 torch 库中的 split_with_sizes_copy 方法执行操作
                torch.split_with_sizes_copy(
                    all_gather_output, all_gather_input_split_sizes, dim=dim, out=out
                )

            # 编译函数 f1，使用 "eager" 后端，并且需要完整的计算图
            @torch.compile(backend="eager", fullgraph=True)
            def f1(all_gather_output, all_gather_input_split_sizes, dim, out):
                # 调用 "mylib" 库中的 split_with_sizes_copy 操作
                return torch.ops.mylib.split_with_sizes_copy(
                    all_gather_output, all_gather_input_split_sizes, dim, out=out
                )

            # 创建测试所需的输入数据
            all_gather_output = torch.randn(2, 272)
            all_gather_input_split_sizes = [128, 8, 128, 8]
            dim = 1
            out = [
                torch.empty(2, 128),
                torch.empty(2, 8),
                torch.empty(2, 128),
                torch.empty(2, 8),
            ]
            # 执行函数 f1 进行测试
            f1(all_gather_output, all_gather_input_split_sizes, dim, out)

        # 使用 torch 库的 _scoped_library 方法创建名为 "mylib" 的库，并指定库的作用域为 "FRAGMENT"
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            # 在库中定义一个名为 "chunk_cat" 的自定义操作
            lib.define(
                "chunk_cat(Tensor[] tensors, int dim, int num_chunks, *, Tensor(a!) out) -> ()"
            )

            # 在 "mylib" 库中实现名为 "chunk_cat" 的操作，支持 "Meta" 和 "CPU" 后端
            @torch.library.impl(lib, "chunk_cat", "Meta")
            @torch.library.impl(lib, "chunk_cat", "CPU")
            def chunk_cat(
                tensors: typing.List[torch.Tensor],
                dim: int,
                num_chunks: int,
                out: torch.Tensor,
            ) -> None:
                # 调用 torch 库中的 _chunk_cat 方法执行操作
                torch._chunk_cat(tensors, dim, num_chunks, out=out)

            # 编译函数 f2，使用 "eager" 后端，并且需要完整的计算图
            @torch.compile(backend="eager", fullgraph=True)
            def f2(tensors, dim, num_chunks, out):
                # 调用 "mylib" 库中的 chunk_cat 操作
                return torch.ops.mylib.chunk_cat(tensors, dim, num_chunks, out=out)

            # 创建测试所需的输入数据
            x = torch.zeros(100, dtype=torch.int64)
            tensors = [
                torch.randn(16, 16),
                torch.randn(16),
                torch.randn(16, 16),
                torch.randn(16),
            ]
            dim = 0
            num_chunks = 2
            out = torch.empty(2, 272)
            # 执行函数 f2 进行测试
            f2(tensors, dim, num_chunks, out)
    def test_runtime_assert_replacement(self):
        # 使用 @torch.compile 装饰器将函数编译为 eager 模式
        @torch.compile(backend="aot_eager")
        def fn(x, y):
            # 将 y 转换为标量值
            z = y.item()
            # 检查 z 是否等于 3，若不等则引发 RuntimeError
            torch._check(z == 3)
            # 返回 x 加上 z 的结果
            return x + z

        # 调用 fn 函数，验证输入 [x, y] 和 [x, y] + [4] 时是否引发 RuntimeError
        fn(torch.randn(4), torch.tensor([3]))
        self.assertRaises(RuntimeError, lambda: fn(torch.randn(4), torch.tensor([4])))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_cat_unbacked(self):
        # 使用 @torch.compile 装饰器将函数编译为 eager 模式
        @torch.compile(backend="eager")
        def fn(x, y):
            # 将 y 转换为标量值
            z = y.item()
            # 返回 x 与由 x 和大小为 z 的全 1 张量拼接而成的结果
            return torch.cat([x, torch.ones(z)])

        # 调用 fn 函数，验证输入 [2, 3] 和 [2, 3] + [1] 时是否引发 RuntimeError
        fn(torch.randn(2, 3), torch.tensor([0]))
        self.assertRaises(
            RuntimeError, lambda: fn(torch.randn(2, 3), torch.tensor([1]))
        )

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_aot_autograd_propagate_unbacked_symints_shape(self):
        # 使用 @torch.compile 装饰器将函数编译为 aot_eager 模式
        @torch.compile(backend="aot_eager")
        def f(x):
            # 返回输入张量 x 中非零元素的索引
            return torch.nonzero(x)

        # 调用 f 函数，验证对输入 [1, 0, 3, 2, 0] 的处理结果
        f(torch.tensor([1, 0, 3, 2, 0]))

    def test_simple_set_usage(self):
        # 定义函数 foo，创建包含 x 和 y 的集合 setty，并返回集合中一个元素的值乘以另一个元素的值
        def foo(x, y):
            setty = {x, y}
            return setty.pop() * setty.pop()

        # 创建 CompileCounter 实例 counter
        counter = CompileCounter()
        # 优化函数 foo，并使用 nopython=True 参数
        foo = torch._dynamo.optimize(counter, nopython=True)(foo)
        # 创建输入张量 x 和 y
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        # 调用 foo 函数，并验证 frame_count 是否等于 1
        foo(x, y)
        self.assertEqual(counter.frame_count, 1)

    def test_add_to_set(self):
        # 定义函数 foo，创建空集合 setty，并向其中添加输入张量 x 和 y 的元素
        def foo(x, y):
            setty = set()
            setty.add(x[0])
            setty.add(x[1])
            setty.add(x[2])
            setty.add(y)
            return y * len(setty)

        # 创建输入张量 x 和 y
        x = torch.randn(10, 10)
        y = torch.randn(2, 2)
        # 记录 foo 函数在给定输入 [x, x, x, x, y] 和 y 时的结果
        eager_result = foo([x, x, x, x, y], y)

        # 创建 CompileCounter 实例 counter
        counter = CompileCounter()
        # 优化函数 foo，并使用 nopython=True 参数
        foo = torch._dynamo.optimize(counter, nopython=True)(foo)
        # 计算优化后的 foo 函数在相同输入上的结果
        result = foo([x, x, x, x, y], y)
        # 验证 frame_count 是否等于 1，验证结果是否与 eager 模式下一致
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(result, eager_result)

    def test_iter_set(self):
        # 定义函数 foo，创建空集合 setty，并逐一将输入张量 x 中的元素添加到集合中
        def foo(x, y):
            setty = set()
            for t in x:
                setty.add(t)
            return y * len(setty)

        # 创建输入张量 x 和 y
        x = torch.randn(10, 10)
        y = torch.randn(2, 2)
        # 记录 foo 函数在给定输入 [x, x, x, x, y] 和 y 时的结果
        eager_result = foo([x, x, x, x, y], y)

        # 创建 CompileCounter 实例 counter
        counter = CompileCounter()
        # 优化函数 foo，并使用 nopython=True 参数
        foo = torch._dynamo.optimize(counter, nopython=True)(foo)
        # 计算优化后的 foo 函数在相同输入上的结果
        result = foo([x, x, x, x, y], y)
        # 验证 frame_count 是否等于 1，验证结果是否与 eager 模式下一致
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(result, eager_result)
    # 定义测试方法，用于测试在图断开时设置输入
    def test_input_set_graph_break(self):
        # 定义内部函数 foo，接受一个参数 x，并返回弹出栈顶两个元素相乘的结果
        def foo(x):
            return x.pop() * x.pop()

        # 生成大小为 10x10 的随机张量 x 和 y
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)

        # 创建一个编译计数器对象
        counter = CompileCounter()

        # 将输入的集合定义为包含 x 和 y 各四次的集合
        inp = {x, x, x, x, y, y}

        # 使用 torch._dynamo.optimize 进行装饰，启用 nopython 模式优化 foo 函数
        foo = torch._dynamo.optimize(counter, nopython=True)(foo)

        # 使用 assertRaisesRegex 断言捕获 torch._dynamo.exc.Unsupported 异常，并验证异常消息
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "^call_method UserDefinedObjectVariable\\(set\\).*",
        ):
            foo(inp)

        # 关闭 nopython 模式优化 foo 函数
        foo = torch._dynamo.optimize(counter, nopython=False)(foo)

        # 调用 foo 函数
        foo(inp)

        # 验证编译计数器的帧数为 1
        self.assertEqual(counter.frame_count, 1)

    # 定义测试方法，用于测试在图断开时重新构建集合
    def test_reconstruct_set_across_graph_break(self):
        # 定义内部函数 foo，接受两个参数 x 和 y，生成一个包含 x 元素的集合 setty，计算 y 与 setty 大小的乘积
        def foo(x, y):
            setty = set()
            for t in x:
                setty.add(t)
            print("Break!")
            return y * len(setty)

        # 生成大小为 10x10 的随机张量 x 和 y
        x = torch.randn(10, 10)
        y = torch.randn(2, 2)

        # 创建一个编译计数器对象
        counter = CompileCounter()

        # 使用 torch._dynamo.optimize 进行装饰，优化 foo 函数
        foo = torch._dynamo.optimize(counter)(foo)

        # 调用 foo 函数，传入参数 [x, x, x, x, y] 和 y
        result = foo([x, x, x, x, y], y)

    # 定义测试方法，用于测试集合别名重新编译
    def test_set_aliasing_recompiles(self):
        # 生成大小为 10 的随机张量 g1、g2、g3、g4
        g1 = torch.randn(10)
        g2 = torch.randn(10)
        g3 = torch.randn(10)
        g4 = torch.randn(10)

        # 定义内部函数 foo，接受三个参数 a、b、c，创建一个包含 g1、a、b、c 元素的集合 myset，并返回 a 与 myset 大小的和
        def foo(a, b, c):
            myset = {g1, a, b, c}
            return a + len(myset)

        # 创建一个编译计数器对象
        counter = CompileCounter()

        # 使用 torch._dynamo.optimize 进行装饰，优化 foo 函数
        foo = torch._dynamo.optimize(counter)(foo)

        # 第一次调用 foo 函数，传入参数 g2、g3、g4，断言编译计数器的帧数为 1
        foo(g2, g3, g4)
        self.assertEqual(counter.frame_count, 1)

        # 再次调用 foo 函数，传入参数 g3、g2、g4，断言编译计数器的帧数仍为 1
        foo(g3, g2, g4)
        self.assertEqual(counter.frame_count, 1)

        # 再次调用 foo 函数，传入参数 g2、g2、g2，断言编译计数器的帧数增加到 2
        foo(g2, g2, g2)
        self.assertEqual(counter.frame_count, 2)

        # 再次调用 foo 函数，传入参数 g3、g3、g3，断言编译计数器的帧数仍为 2
        foo(g3, g3, g3)
        self.assertEqual(counter.frame_count, 2)

        # 再次调用 foo 函数，传入参数 g1、g1、g1，断言编译计数器的帧数增加到 3
        foo(g1, g1, g1)
        self.assertEqual(counter.frame_count, 3)

        # 重置动态图机制
        torch._dynamo.reset()

        # 再次调用 foo 函数，传入参数 g1、g1、g1，断言编译计数器的帧数增加到 4
        foo(g1, g1, g1)
        self.assertEqual(counter.frame_count, 4)

        # 再次调用 foo 函数，传入参数 g3、g3、g3，断言编译计数器的帧数增加到 5
        foo(g3, g3, g3)
        self.assertEqual(counter.frame_count, 5)

        # 再次调用 foo 函数，传入参数 g2、g2、g2，断言编译计数器的帧数仍为 5
        foo(g2, g2, g2)
        self.assertEqual(counter.frame_count, 5)

        # 再次调用 foo 函数，传入参数 g2、g3、g4，断言编译计数器的帧数增加到 6
        foo(g2, g3, g4)
        self.assertEqual(counter.frame_count, 6)

        # 再次调用 foo 函数，传入参数 g3、g2、g4，断言编译计数器的帧数仍为 6
        foo(g3, g2, g4)
        self.assertEqual(counter.frame_count, 6)
    # 定义一个测试函数，验证字符串格式化返回结果的正确性，使用 Torch 的编译装饰器
    def test_str_format_return1(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(img):
            # 计算图像的正弦值
            x = torch.sin(img)
            # 格式化字符串以描述图像的形状和批次大小
            y = f"shape {img.shape[-2:]} batch size {img.shape[0]}"
            # 返回图像加上正弦值和格式化字符串
            return img + x, y

        # 创建一个随机张量作为输入
        img1 = torch.randn(1, 1, 8, 8)
        # 调用函数获取结果和消息
        res, msg = fn(img1)
        # 断言消息的正确性
        self.assertEqual(msg, "shape torch.Size([8, 8]) batch size 1")
        # 断言结果的正确性
        self.assertEqual(res, img1 + torch.sin(img1))

    # 定义另一个测试函数，验证字符串格式化返回结果的正确性，使用 Torch 的编译装饰器
    def test_str_format_return2(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(img):
            # 计算图像的正弦值
            x = torch.sin(img)
            # 使用 format 方法格式化字符串，描述图像的形状和批次大小
            y = "shape {} batch size {y:.2f}".format(img.shape[-2:], y=img.shape[0])
            # 返回图像加上正弦值和格式化字符串
            return img + x, y

        # 创建一个随机张量作为输入
        img1 = torch.randn(1, 1, 8, 8)
        # 调用函数获取结果和消息
        res, msg = fn(img1)
        # 断言消息的正确性
        self.assertEqual(msg, "shape torch.Size([8, 8]) batch size 1.00")
        # 断言结果的正确性
        self.assertEqual(res, img1 + torch.sin(img1))

    # 使用 Torch 的配置装饰器，测试验证输出未支持的情况
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_validate_outputs_unbacked(self):
        # 定义一个自定义 Torch 自动求导函数
        class SillyCat(torch.autograd.Function):
            @staticmethod
            # 前向传播函数
            def forward(ctx, x0, x1, i):
                # 保存张量以备后用
                ctx.save_for_backward(i)
                # 连接两个输入张量
                return torch.cat([x0, x1])

            @staticmethod
            # 反向传播函数
            def backward(ctx, grad_out):
                # 从保存的张量中恢复数据
                (i,) = ctx.saved_tensors
                i0, i1 = i.tolist()
                # 分别计算两个输入张量的梯度
                g_x0, g_x1 = grad_out.split([i0, i1])
                return g_x0, g_x1, None

        # 使用 Torch 的编译装饰器，并指定后端为 aot_eager，完整图模式
        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x, i):
            i0, i1 = i.tolist()
            # 将输入张量分割成两部分，并应用自定义的函数
            x0, x1 = x.split([i0, i1])
            return SillyCat.apply(x0, x1, i)

        # 调用函数，传入一个需要自动求导的随机张量和一个指定的索引张量
        f(torch.randn(9, requires_grad=True), torch.tensor([3, 6]))

    # 定义一个测试函数，验证字符串格式化断言的正确性，使用 Torch 的编译装饰器
    def test_str_format_assert1(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(img):
            # 计算图像的正弦值
            x = torch.sin(img)
            # 获取形状的最后两个维度
            val = x.shape[-2:]
            # 断言形状长度为 2，否则触发异常
            torch._assert(len(val) == 2, f"shape {img.shape}")
            # 返回图像加上正弦值
            return img + x

        # 创建一个随机张量作为输入
        img1 = torch.randn(1, 1, 8, 8)
        # 调用函数获取结果
        res = fn(img1)
        # 断言结果的正确性
        self.assertEqual(res, img1 + torch.sin(img1))

    # 定义另一个测试函数，验证字符串格式化断言的正确性，使用自定义的计数器类作为后端
    def test_str_format_assert2(self):
        # 创建一个编译计数器对象
        cnt = CompileCounter()

        # 使用自定义的编译装饰器，传入计数器对象作为后端
        @torch.compile(backend=cnt)
        def fn(img):
            # 计算图像的正弦值
            x = torch.sin(img)
            # 断言图像的最后两个维度分别为 8 和 16，否则触发异常
            torch._assert(
                img.shape[-2] == 8 and img.shape[-1] == 16, f"shape {img.shape}"
            )
            # 返回图像加上正弦值
            return img + x

        # 创建一个随机张量作为输入
        img1 = torch.randn(1, 3, 8, 16)
        # 调用函数获取结果
        res = fn(img1)
        # 断言结果的正确性
        self.assertEqual(res, img1 + torch.sin(img1))
        # 断言计数器的帧计数为 1
        self.assertEqual(cnt.frame_count, 1)

        # 触发重新编译和图形破坏
        img2 = torch.randn(1, 3, 8, 15)
        # 断言触发断言异常
        self.assertRaises(AssertionError, lambda: fn(img2))
    # 定义测试函数，测试处理标量情况下的列表转换
    def test_tolist_scalar(self):
        # 定义处理函数 fn，将输入张量 x 转换为新列表，每个元素乘以 4
        def fn(x):
            new_list = []
            # 遍历 x 转换为 Python 列表，每个元素乘以 4 后加入 new_list
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        # 创建输入张量 x，包含单个元素 3
        x = torch.tensor([3])
        # 预期结果，调用 fn 后的返回值
        eager = fn(x)
        # 创建 CompileCounter 对象，用于统计编译帧数
        counter = CompileCounter()
        # 对 fn 进行优化并编译，nopython=True 表示无 Python 解释器依赖
        compiled = torch._dynamo.optimize(counter, nopython=True)(fn)(x)
        # 断言预期结果与编译结果相等
        self.assertEqual(eager, compiled)
        # 断言编译帧数为 1
        self.assertEqual(counter.frame_count, 1)

    # 定义测试函数，测试处理一维张量情况下的列表转换
    def test_tolist_1d(self):
        # 定义处理函数 fn，将输入张量 x 转换为新列表，每个元素乘以 4
        def fn(x):
            new_list = []
            # 遍历 x 转换为 Python 列表，每个元素乘以 4 后加入 new_list
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        # 创建输入张量 x，包含两个元素 [2, 1]
        x = torch.tensor([2, 1])
        # 预期结果，调用 fn 后的返回值
        eager = fn(x)
        # 创建 CompileCounter 对象，用于统计编译帧数
        counter = CompileCounter()
        # 对 fn 进行优化并编译，nopython=True 表示无 Python 解释器依赖
        compiled = torch._dynamo.optimize(counter, nopython=True)(fn)(x)
        # 断言预期结果与编译结果相等
        self.assertEqual(eager, compiled)
        # 断言编译帧数为 1
        self.assertEqual(counter.frame_count, 1)

    # 定义测试函数，测试处理多维张量情况下的列表转换
    def test_tolist_kd(self):
        # 定义处理函数 fn，将输入张量 x 转换为新列表，每个元素乘以 4
        def fn(x):
            new_list = []
            # 遍历 x 转换为 Python 列表，每个元素乘以 4 后加入 new_list
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        # 创建输入张量 x，包含两个维度的元素 [[2, 1], [2, 1], [2, 1]]
        x = torch.tensor([[[2, 1], [2, 1], [2, 1]], [[2, 1], [2, 1], [2, 1]]])
        # 预期结果，调用 fn 后的返回值
        eager = fn(x)
        # 创建 CompileCounter 对象，用于统计编译帧数
        counter = CompileCounter()
        # 对 fn 进行优化并编译，nopython=True 表示无 Python 解释器依赖
        compiled = torch._dynamo.optimize(counter, nopython=True)(fn)(x)
        # 断言预期结果与编译结果相等
        self.assertEqual(eager, compiled)
        # 断言编译帧数为 1
        self.assertEqual(counter.frame_count, 1)

    # 定义测试函数，测试处理零维张量情况下的列表转换
    @patch.object(torch._dynamo.config, "specialize_int", True)
    def test_tolist_0d(self):
        # 定义处理函数 fn，将输入张量 x 转换为新列表，元素乘以 4 后加入 new_list
        def fn(x):
            new_list = []
            i = x.tolist()
            new_list.append(i * 4)
            return new_list

        # 创建输入张量 x，值为 42
        x = torch.tensor(42)
        # 预期结果，调用 fn 后的返回值
        eager = fn(x)
        # 创建 CompileCounter 对象，用于统计编译帧数
        counter = CompileCounter()
        # 对 fn 进行优化并编译，nopython=True 表示无 Python 解释器依赖
        compiled = torch._dynamo.optimize(counter, nopython=True)(fn)(x)
        # 断言预期结果与编译结果相等
        self.assertEqual(eager, compiled)
        # 断言编译帧数为 1
        self.assertEqual(counter.frame_count, 1)

    # 定义测试函数，测试处理动态形状的多维张量情况下的列表转换
    @patch.object(torch._dynamo.config, "assume_static_by_default", False)
    @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
    def test_tolist_kd_dynamic(self):
        # 定义处理函数 fn，将输入张量 x 转换为新列表，元素乘以 4 后加入 new_list
        def fn(x):
            new_list = []
            i = x.tolist()
            new_list.append(i * 4)
            return new_list

        # 创建输入张量 x，形状为 [5, 5]，值为在 [3, 5) 范围内的随机整数
        x = torch.randint(3, 5, [5, 5])
        # 预期结果，调用 fn 后的返回值
        eager = fn(x)
        # 创建 CompileCounter 对象，用于统计编译帧数
        counter = CompileCounter()
        # 对 fn 进行优化并编译，nopython=True 表示无 Python 解释器依赖
        compiled_fn = torch._dynamo.optimize(counter, nopython=True)(fn)
        compiled = compiled_fn(x)
        # 断言预期结果与编译结果相等
        self.assertEqual(eager, compiled)
        # 断言编译帧数为 1
        self.assertEqual(counter.frame_count, 1)

        # 改变输入张量 x 的值，不会触发重新编译
        x = torch.randint(7, 9, [5, 5])
        compiled_fn(x)
        # 断言编译帧数仍为 1
        self.assertEqual(counter.frame_count, 1)

        # 改变输入张量 x 的形状，强制触发重新编译
        x = torch.randint(3, 5, [3, 3])
        compiled_fn(x)
        # 断言编译帧数增加到 2
        self.assertEqual(counter.frame_count, 2)
    def test_tolist_float(self):
        # 定义一个函数 fn，接受一个参数 x
        def fn(x):
            # 初始化一个空列表 new_list
            new_list = []
            # 遍历 x 转换为列表后的每个元素 i
            for i in x.tolist():
                # 将 i 的每个元素乘以 4 后添加到 new_list 中
                new_list.append(i * 4)
            # 返回处理后的列表 new_list
            return new_list

        # 创建一个三维张量 x
        x = torch.tensor(
            [[[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]], [[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]]]
        )
        # 在非优化环境下调用 fn 函数，获得 eager 结果
        eager = fn(x)
        # 初始化一个编译计数器对象 counter
        counter = CompileCounter()
        # 使用 Torch 动态优化机制优化 fn 函数，并应用于 x
        compiled = torch._dynamo.optimize(counter)(fn)(x)
        # 断言优化后的结果与非优化结果相等
        self.assertEqual(eager, compiled)
        # 断言编译计数器中的帧数为 0，表示没有被编译
        self.assertEqual(counter.frame_count, 0)

    def test_inline_closure_not_loaded_by_parent(self):
        # 定义一个外部函数 outer，返回 a + 1
        def outer(a):
            return a + 1

        # 定义一个间接调用函数 indirect，调用 direct 函数
        def indirect(x):
            return direct(x)

        # 定义一个直接调用函数 direct，其中包含两层嵌套函数 deep 和 deep2
        def direct(x):
            # 定义深层嵌套函数 deep2，调用外部函数 outer
            def deep2(c):
                return outer(c)

            # 定义深层嵌套函数 deep，调用深层嵌套函数 deep2
            def deep(c):
                return deep2(c)

            # 返回 deep 函数的结果，传入参数 x
            return deep(x)

        # 创建一个长度为 3 的随机张量 x
        x = torch.randn(3)
        # 在非优化环境下调用 indirect 函数，获得 eager 结果
        eager = indirect(x)
        # 初始化一个编译计数器对象 counter
        counter = CompileCounter()
        # 使用 Torch 动态优化机制优化 indirect 函数，并应用于 x
        compiled = torch._dynamo.optimize(counter)(indirect)(x)
        # 断言优化后的结果与非优化结果相等
        self.assertEqual(eager, compiled)
        # 断言编译计数器中的帧数为 1，表示有一次被编译
        self.assertEqual(counter.frame_count, 1)

    def test_deque_input(self):
        # 创建两个随机张量 a 和 b
        a = torch.randn([2, 3])
        b = torch.randn([2, 3])
        # 创建一个 deque 对象 d1，并在索引 0 处插入字符串 "foo"
        d1 = collections.deque([a, b])
        d1.insert(0, "foo")

        # 创建另一个 deque 对象 d2，并在索引 0 处插入字符串 "foo"
        d2 = collections.deque([a, b])
        d2.insert(0, "foo")

        # 定义一个函数 fn，接受一个 deque 参数 q
        def fn(q):
            # 弹出 q 的最后一个元素赋给变量 a
            a = q.pop()
            # 弹出 q 的下一个最后一个元素赋给变量 b
            b = q.pop()
            # 返回 a 和 b 的元素对应位置相乘的结果
            return a * b

        # 在非优化环境下调用 fn 函数，获得 eager 结果
        eager = fn(d1)
        # 初始化一个编译计数器对象 counter
        counter = CompileCounter()
        # 使用 Torch 动态优化机制优化 fn 函数，并应用于 d2
        compiled = torch._dynamo.optimize(counter)(fn)(d2)
        # 断言优化后的结果与非优化结果相等
        self.assertEqual(eager, compiled)
        # 断言编译计数器中的帧数为 1，表示有一次被编译
        self.assertEqual(counter.frame_count, 1)

    def test_deque_append_left(self):
        # 创建一个 deque 对象 d1，并在索引 0 处插入字符串 "foo"
        d1 = collections.deque([10, 10])
        d1.insert(0, "foo")

        # 创建另一个 deque 对象 d2，并在索引 0 处插入字符串 "foo"
        d2 = collections.deque([10, 10])
        d2.insert(0, "foo")

        # 定义一个函数 fn，接受三个参数 q, a, b
        def fn(q, a, b):
            # 在 q 的头部插入元素 a
            q.appendleft(a)
            # 在 q 的头部插入元素 b
            q.appendleft(b)
            # 弹出 q 的两个头部元素相乘并返回结果
            return q.popleft() * q.popleft()

        # 创建两个随机张量 a 和 b
        a = torch.randn([3, 3])
        b = torch.randn([3, 3])
        # 在非优化环境下调用 fn 函数，获得 eager 结果
        eager = fn(d1, a, b)
        # 初始化一个编译计数器对象 counter
        counter = CompileCounter()
        # 使用 Torch 动态优化机制优化 fn 函数，并应用于 d2, a, b
        compiled = torch._dynamo.optimize(counter)(fn)(d2, a, b)
        # 断言优化后的结果与非优化结果相等
        self.assertEqual(eager, compiled)
        # 断言编译计数器中的帧数为 1，表示有一次被编译
        self.assertEqual(counter.frame_count, 1)
        # 断言编译后的结果为 torch.Tensor 类型
        self.assertTrue(isinstance(compiled, torch.Tensor))

    def test_yield_from(self):
        # 定义一个函数 yield_from_fn，接受一个列表 t_list 和一个数值 k
        def yield_from_fn(t_list, k):
            # 定义一个生成器函数 yield_from_gen，接受一个列表 l
            def yield_from_gen(l):
                # 将列表 l 中每个元素乘以 k，并放入列表 l2
                l2 = [t * k for t in l]
                # 使用 yield from 语法将 l2 的每个元素逐个返回
                yield from l2

            # 返回生成器函数 yield_from_gen 返回的结果列表
            return [t * k for t in yield_from_gen(t_list)]

        # 创建一个包含三个随机张量的列表 t_list
        t_list = [torch.randn([2, 3]) for _ in range(3)]
        # 在非优化环境下调用 yield_from_fn 函数，获得 eager 结果
        eager = yield_from_fn(t_list, 2)
        # 初始化一个编译计数器对象 counter
        counter = CompileCounter()
        # 使用 Torch 动态优化机制优化 yield_from_fn 函数，并应用于 t_list, 2
        compiled = torch._dynamo.optimize(counter)(yield_from_fn)(t_list, 2)
        # 断言优化后的结果与非优化结果相等
        self.assertEqual(eager, compiled)
        # 断言编译计数器中的帧数为 1，表示有一次被编译
        self.assertEqual(counter.frame_count, 1)
    def test_yield_from_in_a_loop(self):
        def gen2():
            yield 1

        def gen1():
            # 定义生成器函数 gen1
            for value in range(5):
                # 使用 yield from 调用 gen2 生成器
                yield from gen2()

        def fn(x):
            # 初始化累加器 c
            c = 0
            # 遍历 gen1 生成器产生的值，并累加到 c
            for i in gen1():
                c = c + i
            # 返回 x 加上累加结果 c
            return x + c

        # 使用 torch.compile 编译 fn 函数，使用 "eager" 后端
        opt_fn = torch.compile(fn, backend="eager")
        # 初始化一个全零张量 x
        x = torch.zeros(4)
        # 断言 fn(x) 和编译后的 opt_fn(x) 结果相等
        self.assertEqual(fn(x), opt_fn(x))

    def test_yield_gen_and_from(self):
        def populate_and_multiply_sequence(n, multiplier):
            # 内联生成器函数 tensor_generator
            def tensor_generator():
                for i in range(n):
                    # 生成 torch.tensor([i]) 的生成器
                    yield torch.tensor([i])

            # 使用 'yield from' 遍历 tensor_generator 生成器，并乘以 multiplier
            t_list = [tensor * multiplier for tensor in tensor_generator()]

            # 定义 yield from 生成器 yield_from_gen
            def yield_from_gen():
                yield from t_list

            # 返回 yield_from_gen 生成的结果列表
            return [t for t in yield_from_gen()]

        # 初始化 multiplier 为 torch.tensor([10])
        multiplier = torch.tensor([10])
        # 调用 populate_and_multiply_sequence 函数，返回 eager 结果
        eager = populate_and_multiply_sequence(5, multiplier)
        # 初始化 CompileCounter 对象
        counter = CompileCounter()
        # 使用 torch._dynamo.optimize(counter) 优化 populate_and_multiply_sequence 函数
        compiled = torch._dynamo.optimize(counter)(populate_and_multiply_sequence)(
            5, multiplier
        )
        # 断言 eager 和 compiled 结果相等
        self.assertEqual(eager, compiled)
        # 断言 counter.frame_count 等于 1
        self.assertEqual(counter.frame_count, 1)

    def test_yield_from_user_stop_iteration(self):
        class MyIter:
            def __init__(self, seq):
                self.seq = seq
                self.index = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.index += 1
                if self.index <= len(self.seq):
                    return self.seq[self.index - 1]
                # 抛出 StopIteration 异常并传递索引值
                raise StopIteration(self.index)

        # 定义 yield from 生成器 yield_from_iter_fn
        def yield_from_iter_fn(seq):
            # 定义生成器 gen，使用 yield from 调用 MyIter(seq) 生成器
            def gen(seq):
                yield from MyIter(seq)

            # 返回 gen 生成的结果列表
            return [i for i in gen(seq)]

        # 生成长度为 3 的随机张量列表 seq
        seq = [torch.randn([2, 3]) for _ in range(3)]
        # 调用 yield_from_iter_fn 函数，返回 eager 结果
        eager = yield_from_iter_fn(seq)
        # 初始化 CompileCounter 对象
        counter = CompileCounter()
        # 使用 torch._dynamo.optimize(counter) 优化 yield_from_iter_fn 函数
        compiled = torch._dynamo.optimize(counter)(yield_from_iter_fn)(seq)
        # 断言 eager 和 compiled 结果相等
        self.assertEqual(eager, compiled)
        # 断言 counter.frame_count 等于 0
        self.assertEqual(counter.frame_count, 0)

    def test_yield_send_to_subgenerator_graph_break(self):
        # 定义子生成器函数 subgenerator
        def subgenerator(tensor):
            # 接收 multiplier 值
            multiplier = yield
            # 生成 tensor * multiplier
            yield tensor * multiplier

        # 定义主生成器函数 main_generator
        def main_generator(t_list):
            # 遍历输入的张量列表 t_list
            for tensor in t_list:
                # 初始化子生成器 subgen
                subgen = subgenerator(tensor)
                # 启动子生成器
                next(subgen)
                # 使用 send 方法发送 torch.tensor([10]) 给子生成器 subgen
                yield from subgen.send(torch.tensor([10]))

        # 生成长度为 5 的张量列表 t_list
        t_list = [torch.tensor([i]) for i in range(5)]
        # 将 main_generator 生成的结果转换为列表 eager
        eager = list(main_generator(t_list))

        # 初始化 CompileCounter 对象
        counter = CompileCounter()
        # 使用 torch._dynamo.optimize(counter) 优化 main_generator 函数
        compiled_fn = torch._dynamo.optimize(counter)(main_generator)
        # 将编译后的结果转换为列表 compiled
        compiled = list(compiled_fn(t_list))

        # 断言 eager 和 compiled 结果相等
        self.assertEqual(eager, compiled)
        # 断言 counter.frame_count 等于 0
        self.assertEqual(counter.frame_count, 0)
    # 测试函数，验证自定义神经网络模块的使用情况
    def test_derpy_nn_module_usage(self):
        # 定义神经网络前向传播函数 ff1
        def ff1(x):
            self = mod1
            # 返回 torch.sigmoid 函数应用于 mod2 输出加上 param1 的结果
            return torch.sigmoid(self.mod2(x) + self.param1)

        # 定义另一个神经网络前向传播函数 ff2
        def ff2(x):
            self = mod2
            # 返回 torch.cos 函数应用于 torch.sin(x) * param2 + 10 的结果
            return torch.cos(torch.sin(x) * self.param2 + 10)

        # 创建两个空的神经网络模块实例 mod1 和 mod2
        mod1 = torch.nn.Module()
        mod2 = torch.nn.Module()
        # 将 mod2 注册为 mod1 的子模块
        mod1.register_module("mod2", mod2)
        # 为 mod1 添加名为 param1 的参数，值为一个形状为 (10,) 的随机张量
        mod1.register_parameter("param1", torch.nn.Parameter(torch.randn(10)))
        # 将 mod1 的 forward 方法设置为 ff1 函数
        mod1.forward = ff1
        # 为 mod2 添加名为 param2 的参数，值为一个形状为 (10,) 的随机张量
        mod2.register_parameter("param2", torch.nn.Parameter(torch.randn(10)))
        # 将 mod2 的 forward 方法设置为 ff2 函数
        mod2.forward = ff2
        # 将 mod1 设置为评估模式
        mod1.eval()

        # 创建输入张量 x，形状为 (10,)
        x = torch.randn(10)
        # 计算期望的输出值
        expected = mod1(x)
        # 创建 CompileCounter 实例
        counter = CompileCounter()
        # 使用 Torch 的编译方法编译 mod1，并计算实际输出值
        actual = torch.compile(mod1, backend=counter, fullgraph=True)(x)
        # 断言实际输出值与期望输出值相等
        self.assertEqual(actual, expected)
        # 断言操作计数器中的操作次数为 6
        self.assertEqual(counter.op_count, 6)

    # 测试默认参数设备和数据类型的情况
    def test_default_args_device_dtype(self):
        # 定义一个 Foo 类
        class Foo:
            def __init__(
                self,
                dtype: torch.dtype = torch.float16,
                device: torch.device = torch.device("cpu"),
            ) -> None:
                # 初始化一个张量 value，数据类型和设备由参数指定
                self.value = torch.tensor(10, dtype=dtype, device=device)

        # 定义一个返回 Foo 实例中 value 属性加一的函数 fn
        def fn():
            return Foo().value + 1

        # 使用 Torch 的优化方法对 fn 进行优化，并返回优化后的函数 opt_func
        opt_func = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 计算 fn 的预期输出结果
        ref = fn()
        # 计算优化后函数 opt_func 的输出结果
        res = opt_func()
        # 断言优化前后结果应该一致
        self.assertEqual(ref, res)

    # 测试 Torch 设备 Python 类型的情况
    def test_torch_device_python_type(self):
        # 遍历设备、设备类型和索引的组合列表
        for device, device_type, index in [
            ("cpu", "cpu", None),
            ("cuda:0", "cuda", 0),
        ]:
            # 如果设备是 "cuda:0" 且测试不包括 CUDA，则跳过本次循环
            if device == "cuda:0" and not TEST_CUDA:
                continue

            # 定义一个接收 target 参数的函数 fn
            def fn(target):
                # 获取目标设备的设备类型
                target_device = target.device
                # 创建一个形状为 (2, 3) 的零张量 a，设备与 target_device 相同
                a = torch.zeros(2, 3, device=target_device)
                # 在跟踪时进行常量断言
                assert isinstance(target_device, torch.device)
                assert target_device.type == device_type
                assert target_device.index == index
                # 创建两个形状为 (2, 3) 的零张量 b 和 c，设备与 target_device 相同
                b = torch.zeros(2, 3, device=target_device)
                c = torch.zeros(2, 3, device=target_device)
                # 返回张量 a、b 和 c 的和
                return a + b + c

            # 导入 Torch 的 ConstantVariable 类
            from torch._dynamo.variables import ConstantVariable

            # 将设备名称转换为 Torch 的设备对象
            device = torch.device(device)
            # 创建预期的 ConstantVariable 实例
            expected_variable = ConstantVariable(device)
            # 断言 ConstantVariable 实例的 Python 类型与设备对象的类型相同
            self.assertEqual(expected_variable.python_type(), type(device))

            # 使用 Torch 的优化方法对 fn 进行优化，并返回优化后的函数 opt_func
            opt_func = torch._dynamo.optimize("eager", nopython=True)(fn)
            # 创建形状为 (2,) 的张量 a，设备由 device 指定
            a = torch.tensor([2, 3], device=device)
            # 计算优化后函数 opt_func 的输出结果
            res = opt_func(a)
            # 断言 res 的类型为 torch.Tensor
            self.assertIsInstance(res, torch.Tensor)
    def test_torch_dtype_python_type(self):
        def fn(target):
            target_dtype = target.dtype
            # 使用目标张量的数据类型创建全零张量 a
            a = torch.zeros(2, 3, dtype=target_dtype)
            # 在跟踪时进行常量断言
            assert isinstance(target_dtype, torch.dtype)
            # 使用相同数据类型创建全零张量 b 和 c
            b = torch.zeros(2, 3, dtype=target_dtype)
            c = torch.zeros(2, 3, dtype=target_dtype)
            # 返回 a、b、c 张量的和
            return a + b + c

        # 从 torch._dynamo.variables 模块导入 ConstantVariable 类
        from torch._dynamo.variables import ConstantVariable

        # 设置 dtype 为 torch.float16
        dtype = torch.float16
        # 创建预期的 ConstantVariable 类对象
        expected_variable = ConstantVariable(dtype)
        # 断言预期的 ConstantVariable 对象的 Python 类型与 dtype 的类型相同
        self.assertEqual(expected_variable.python_type(), type(dtype))

        # 使用 torch._dynamo.optimize 对 fn 函数进行优化，并返回优化后的函数
        opt_func = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 创建一个张量 a，数据类型为 dtype
        a = torch.tensor([2, 3], dtype=dtype)
        # 使用优化后的函数处理张量 a
        res = opt_func(a)
        # 断言 res 是 torch.Tensor 类型的实例
        self.assertIsInstance(res, torch.Tensor)

    def test_itertools_repeat(self):
        # 清空计数器
        counters.clear()

        def fn(x):
            # 创建一个迭代器 r，用于重复值 100.0，重复 5 次
            r = itertools.repeat(100.0, 5)
            # 将 r 中的值加到 x 上
            for i in r:
                x += i
            return x

        # 创建一个形状为 [2, 5] 的随机张量 x
        x = torch.randn([2, 5])
        # 直接调用 fn 函数进行计算
        eager = fn(x)

        # 使用 torch._dynamo.optimize 对 fn 函数进行优化，并返回优化后的函数
        compiled_fn = torch._dynamo.optimize(backend="eager", nopython=True)(fn)
        # 使用优化后的函数处理张量 x
        compiled = compiled_fn(x)

        # 断言 eager 和 compiled 的结果列表相同
        self.assertEqual(list(eager), list(compiled))
        # 断言计数器中的 "graph_break" 数组长度为 0
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_infinite_repeat(self):
        # 清空计数器
        counters.clear()

        def fn(x):
            # 创建一个无限重复值 100.0 的迭代器 r
            r = itertools.repeat(100.0)
            idx = 0
            # 将 r 中的值加到 x 上，直到 idx 大于 10 时退出循环
            for i in r:
                x += i
                idx += 1
                if idx > 10:
                    break
            return x

        # 创建一个形状为 [2, 5] 的随机张量 x
        x = torch.randn([2, 5])
        # 直接调用 fn 函数进行计算
        eager = fn(x)

        # 使用 torch._dynamo.optimize 对 fn 函数进行优化，并返回优化后的函数
        compiled_fn = torch._dynamo.optimize(backend="eager", nopython=True)(fn)
        # 使用优化后的函数处理张量 x
        compiled = compiled_fn(x)

        # 断言 eager 和 compiled 的结果列表相同
        self.assertEqual(list(eager), list(compiled))
        # 断言计数器中的 "graph_break" 数组长度为 0
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_infinite_repeat_mutation(self):
        # 清空计数器
        counters.clear()

        def fn(x):
            # 创建一个无限重复 x 的迭代器 r
            r = itertools.repeat(x)
            idx = 0
            # 将 r 中的值加到 x 上，并将迭代器中的值加 1，直到 idx 大于 10 时退出循环
            for i in r:
                x += i
                i += 1
                idx += 1
                if idx > 10:
                    break
            return x

        # 创建一个形状为 [2, 5] 的随机张量 x
        x = torch.randn([2, 5])
        # 直接调用 fn 函数进行计算
        eager = fn(x)

        # 使用 torch._dynamo.optimize 对 fn 函数进行优化，并返回优化后的函数
        compiled_fn = torch._dynamo.optimize(backend="eager", nopython=True)(fn)
        # 使用优化后的函数处理张量 x
        compiled = compiled_fn(x)

        # 断言 eager 和 compiled 的结果列表相同
        self.assertEqual(list(eager), list(compiled))
        # 断言计数器中的 "graph_break" 数组长度为 0
        self.assertEqual(len(counters["graph_break"]), 0)
    # 测试 itertools.count 函数的使用，对不同参数组合进行测试
    def test_itertools_infinite_count(self):
        # 清空计数器
        counters.clear()

        # 定义一个函数 fn，接受参数 x
        def fn(x):
            # 使用 itertools.count 生成一个无限迭代器 r，根据不同参数组合
            r = itertools.count(*args)
            idx = 0
            # 遍历迭代器 r
            for i in r:
                x += i  # 累加迭代器的值到 x
                idx += 1
                if idx > 10:
                    break  # 当达到索引大于 10 时退出循环
            return x

        x = torch.randn([2, 5])  # 生成一个随机张量 x
        eager = fn(x)  # 使用 fn 函数计算 eager 结果

        # 使用 Torch 的优化方法进行编译 fn 函数
        compiled_fn = torch._dynamo.optimize(backend="eager", nopython=True)(fn)
        compiled = compiled_fn(x)  # 编译后的函数计算结果

        # 断言计算结果一致
        self.assertEqual(list(eager), list(compiled))
        # 断言 "graph_break" 计数器为空
        self.assertEqual(len(counters["graph_break"]), 0)

    # 测试 itertools.cycle 函数的使用，对不同迭代器进行循环测试
    def test_itertools_infinite_cycle(self):
        counters.clear()

        # 定义一个函数 fn，接受参数 x
        def fn(x):
            # 遍历不同的迭代器（空迭代器、包含整数和浮点数的迭代器、重复 -1 的迭代器、从 10 开始计数的迭代器）
            for iterator in (
                iter([]),
                iter([10, 11.0]),
                itertools.repeat(-1, 3),
                itertools.count(10),
            ):
                r = itertools.cycle(iterator)  # 使用 itertools.cycle 创建循环迭代器 r
                idx = 0
                x += 1
                # 遍历循环迭代器 r
                for i in r:
                    x += i  # 累加迭代器的值到 x
                    idx += 1
                    if idx > 10:
                        break  # 当达到索引大于 10 时退出循环
            return x

        x = torch.randn([2, 5])  # 生成一个随机张量 x
        eager = fn(x)  # 使用 fn 函数计算 eager 结果

        # 使用 Torch 的优化方法进行编译 fn 函数
        compiled_fn = torch._dynamo.optimize(backend="eager", nopython=True)(fn)
        compiled = compiled_fn(x)  # 编译后的函数计算结果

        # 断言计算结果一致
        self.assertEqual(list(eager), list(compiled))
        # 断言 "graph_break" 计数器为空
        self.assertEqual(len(counters["graph_break"]), 0)

    # 测试 itertools.accumulate 函数的使用，对张量的维度进行累积求积
    def test_itertools_accumulate_symint_default_sum(self):
        # https://github.com/pytorch/pytorch/issues/110287
        counters.clear()

        # 定义一个函数 fn，接受参数 x
        def fn(x):
            r = itertools.accumulate([x.size(0), x.size(1)])  # 使用 itertools.accumulate 对张量的维度进行累积
            # 遍历累积器 r
            for i in r:
                x *= i  # 将张量 x 与累积值相乘
            return x

        x = torch.randn(2, 3)  # 生成一个随机张量 x
        eager = fn(x)  # 使用 fn 函数计算 eager 结果

        # 使用 Torch 的优化方法进行编译 fn 函数
        compiled_fn = torch._dynamo.optimize(backend="eager", nopython=True)(fn)
        compiled = compiled_fn(x)  # 编译后的函数计算结果

        # 断言计算结果一致
        self.assertEqual(list(eager), list(compiled))
        # 断言 "graph_break" 计数器为空
        self.assertEqual(len(counters["graph_break"]), 0)

    # 测试 itertools.accumulate 函数的使用，对输入的多个张量进行累积
    def test_itertools_accumulate_tensors_default_sum(self):
        counters.clear()

        # 定义一个函数 fn，接受参数 a, b, c, d, x
        def fn(a, b, c, d, x):
            l = [a, b, c, d, x]  # 将输入的张量组成列表 l
            # 遍历列表 l，对每个张量乘以 x
            for i, t in enumerate(l):
                l[i] = t * x
            return itertools.accumulate(l)  # 对列表 l 进行累积

        t_list = [torch.tensor([i + 1]) for i in range(4)]  # 生成包含四个张量的列表 t_list
        x = torch.tensor([[1, 2], [3, 4]])  # 生成一个二维张量 x
        eager = fn(*t_list, x)  # 使用 fn 函数计算 eager 结果

        # 使用 Torch 的优化方法进行编译 fn 函数
        compiled_fn = torch._dynamo.optimize(backend="eager", nopython=True)(fn)
        compiled = compiled_fn(*t_list, x)  # 编译后的函数计算结果

        # 断言计算结果一致
        self.assertEqual(list(eager), list(compiled))
        # 断言 "graph_break" 计数器为空
        self.assertEqual(len(counters["graph_break"]), 0)
    # 定义一个测试方法，测试 itertools 的 accumulate 函数与张量操作的组合
    def test_itertools_accumulate_tensors_builtins(self):
        # 遍历三个内置运算符：乘法、减法、乘方
        for builtin_op in [operator.mul, operator.sub, operator.pow]:
            # 清空计数器
            counters.clear()

            # 定义一个函数 fn，接受五个参数，其中四个是张量列表，最后一个是张量 x
            def fn(a, b, c, d, x):
                # 将参数组成列表 l
                l = [a, b, c, d, x]
                # 将列表中的每个张量与 x 相乘
                for i, t in enumerate(l):
                    l[i] = t * x
                # 使用 itertools 的 accumulate 函数，应用内置运算符进行累积计算
                return itertools.accumulate(l, builtin_op)

            # 创建四个张量的列表
            t_list = [torch.tensor([i + 1]) for i in range(4)]
            # 创建一个张量 x
            x = torch.tensor([[1, 2], [3, 4]])
            # 调用函数 fn，使用 eager 模式进行计算
            eager = fn(*t_list, x)

            # 使用 torch._dynamo.optimize 进行函数编译，得到编译后的函数 compiled_fn
            compiled_fn = torch._dynamo.optimize(backend="eager", nopython=True)(fn)
            # 调用编译后的函数 compiled_fn，传入相同的参数，得到编译后的结果
            compiled = compiled_fn(*t_list, x)

            # 断言两种计算方式得到的结果列表相等
            self.assertEqual(list(eager), list(compiled))
            # 断言图断点计数器中的断点数量为 0
            self.assertEqual(len(counters["graph_break"]), 0)

    # 定义一个测试方法，测试 itertools 的 accumulate 函数与关键字参数的组合
    def test_itertools_accumulate_tensors_kwargs(self):
        # 导入计数器模块
        from torch._dynamo.utils import counters

        # 遍历三组关键字参数字典
        for kwargs in [
            {"func": operator.mul},
            {"initial": 100},
            {"func": operator.sub, "initial": -1},
        ]:
            # 清空计数器
            counters.clear()

            # 定义一个函数 fn，接受五个参数，其中四个是张量列表，最后一个是张量 x
            def fn(a, b, c, d, x):
                # 将参数组成列表 l
                l = [a, b, c, d, x]
                # 将列表中的每个张量与 x 相乘
                for i, t in enumerate(l):
                    l[i] = t * x
                # 使用 itertools 的 accumulate 函数，应用关键字参数进行累积计算
                return itertools.accumulate(l, **kwargs)

            # 创建四个张量的列表
            t_list = [torch.tensor([i + 1]) for i in range(4)]
            # 创建一个张量 x
            x = torch.tensor([[1, 2], [3, 4]])

            # 使用 torch._dynamo.optimize 进行函数编译，得到编译后的函数 compiled_fn
            compiled_fn = torch._dynamo.optimize(backend="eager", nopython=True)(fn)
            # 调用编译后的函数 compiled_fn，传入相同的参数，得到编译后的结果
            compiled = compiled_fn(*t_list, x)
            # 调用函数 fn，使用 eager 模式进行计算
            eager = fn(*t_list, x)

            # 断言两种计算方式得到的结果列表相等
            self.assertEqual(list(eager), list(compiled))
            # 断言图断点计数器中的断点数量为 0
            self.assertEqual(len(counters["graph_break"]), 0)

    # 定义一个测试方法，测试版本解析功能
    def test_packaging_version_parse(self):
        # 导入版本解析模块
        from packaging import version

        # 使用装饰器定义一个编译函数，以 eager 模式和完整图形进行编译
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # 创建一个张量 x，值为零
            x = torch.zeros(1)
            # 如果当前 Torch 版本大于等于 "2.0.0"
            if version.parse(torch.__version__) >= version.parse("2.0.0"):
                # 返回 x 加上 1
                return x + 1
            # 否则返回 x
            return x

        # 断言调用 fn 返回的结果值为 1
        self.assertEqual(fn().item(), 1)
    # 测试函数，用于测试使用自定义函数进行 itertools.accumulate 操作的情况
    def test_itertools_accumulate_tensors_user_defined(self):
        # 定义第一个用户定义函数 udo_fn_0，返回固定值 -1
        def udo_fn_0(a, b):
            return -1

        # 随机生成一个整数 rando，范围在 0 到 1 之间
        rando = random.randint(0, 1)

        # 定义第二个用户定义函数 udo_fn_1，根据随机数 rando 计算返回值
        def udo_fn_1(a, b):
            return a * rando + b * rando

        # 定义列表 seen，用于存储操作过程中的数据
        seen = []

        # 定义第三个用户定义函数 udo_fn_2，将 a 和 b 添加到 seen 中，返回 a 乘以 seen 长度的结果
        def udo_fn_2(a, b):
            seen.append(a)
            seen.append(b)
            return a * len(seen)

        # 遍历三个用户定义函数，执行清除计数器和 Torch 动态图重置的操作
        for udo_fn in [udo_fn_0, udo_fn_1, udo_fn_2]:
            counters.clear()  # 清除计数器
            torch._dynamo.reset()  # 重置 Torch 动态图

            # 定义主测试函数 fn，对输入的参数进行处理并使用 itertools.accumulate 进行累积操作
            def fn(a, b, c, d, x):
                l = [a, b, c, d, x]
                for i, t in enumerate(l):
                    l[i] = t * x
                return itertools.accumulate(l, udo_fn)

            # 创建一组测试数据 t_list 和 x
            t_list = [torch.tensor([i]) for i in range(4)]
            x = torch.tensor([[1, 2], [3, 4]])

            # 在 eager 模式下执行 fn 函数，并比较其结果
            eager = fn(*t_list, x)

            # 使用 Torch 动态编译优化 fn 函数，并在 eager 模式下执行编译后的函数
            compiled_fn = torch._dynamo.optimize(backend="eager", nopython=True)(fn)
            compiled = compiled_fn(*t_list, x)

            # 断言两种执行方式得到的结果相等
            self.assertEqual(list(eager), list(compiled))
            # 断言 graph_break 计数器长度为 0，表示没有中断动态图的操作
            self.assertEqual(len(counters["graph_break"]), 0)

    # 测试纯 Python 实现的 accumulate 函数
    def test_pure_python_accumulate(self):
        # 定义 accumulate 函数，实现对可迭代对象进行累积操作
        def accumulate(iterable, func=lambda x, y: x + y):
            it = iter(iterable)
            try:
                # 初始化累加器，使用可迭代对象的第一个值
                accumulator = next(it)
            except StopIteration:
                # 如果可迭代对象为空，则返回空生成器
                return
            yield accumulator

            # 对剩余的元素进行累积操作
            for element in it:
                accumulator = func(accumulator, element)
                yield accumulator

        # 定义测试函数 fn，对输入的可迭代对象应用 accumulate 函数
        def fn(it):
            return accumulate(it)

        # 创建一组测试数据 t_list
        t_list = [torch.tensor([i]) for i in range(4)]

        # 在 eager 模式下执行 fn 函数
        eager = fn(t_list)

        # 使用 Torch 动态编译优化 fn 函数，并在 eager 模式下执行编译后的函数
        counter = CompileCounter()
        compiled_fn = torch._dynamo.optimize(counter)(fn)
        compiled = compiled_fn(t_list)

        # 断言两种执行方式得到的结果相等
        self.assertEqual(list(eager), list(compiled))
        # 断言 frame_count 计数器为 1，表示只有一个编译帧
        self.assertEqual(counter.frame_count, 1)

    # 测试纯 Python 实现的 itertools.groupby 函数，默认使用身份函数作为 key 函数
    def test_itertools_groupby_pure_python_default_identify_func(self):
        counters.clear()  # 清除计数器

        # 定义函数 fn，对输入列表进行 itertools.groupby 操作
        def fn(l):
            return [(k, list(g)) for k, g in itertools.groupby(l)]

        # 创建输入列表 l
        l = [1, 2, 2, 3, 4, 4, 4, 1, 2]

        # 在 eager 模式下执行 fn 函数，并比较其结果
        eager = fn(l)

        # 使用 Torch 动态编译优化 fn 函数，并在 eager 模式下执行编译后的函数
        compiled_fn = torch._dynamo.optimize(backend="eager", nopython=True)(fn)
        compiled = compiled_fn(l)

        # 断言两种执行方式得到的结果相等
        self.assertEqual(eager, compiled)
        # 断言 graph_break 计数器长度为 0，表示没有中断动态图的操作
        self.assertEqual(len(counters["graph_break"]), 0)

    # 测试纯 Python 实现的 itertools.groupby 函数，使用自定义的 key 函数
    def test_itertools_groupby_pure_python_key_func(self):
        counters.clear()  # 清除计数器

        # 定义函数 fn，对输入列表按照 operator.neg 函数进行 itertools.groupby 操作
        def fn(l):
            return [(k, list(g)) for k, g in itertools.groupby(l, key=operator.neg)]

        # 创建输入列表 l
        l = [1, 2, -2, 3, 4, 4, -4, 0, -2]

        # 在 eager 模式下执行 fn 函数，并比较其结果
        eager = fn(l)

        # 使用 Torch 动态编译优化 fn 函数，并在 eager 模式下执行编译后的函数
        compiled_fn = torch._dynamo.optimize(backend="eager", nopython=True)(fn)
        compiled = compiled_fn(l)

        # 断言两种执行方式得到的结果相等
        self.assertEqual(eager, compiled)
        # 断言 graph_break 计数器长度为 0，表示没有中断动态图的操作
        self.assertEqual(len(counters["graph_break"]), 0)
    def test_list_iterator_contains(self):
        # 定义一个内部函数 fn，接受参数 x
        def fn(x):
            # 创建一个迭代器 it，包含字符串列表 ["my_weight", "not_my_weight"]
            it = iter(["my_weight", "not_my_weight"])
            # 获取迭代器的下一个元素，此时迭代器指向 "not_my_weight"
            next(it)
            # 检查字符串 "my_weight" 是否在迭代器 it 中
            if "my_weight" in it:
                # 如果存在，则返回 x + 2
                return x + 2
            # 否则返回 x + 1
            return x + 1

        # 创建一个包含三个元素的零张量 x
        x = torch.zeros(3)
        # 对函数 fn 进行编译优化，使用 Torch 的特定优化器
        compiled_fn = torch._dynamo.optimize(backend="eager", nopython=True)(fn)

        # 断言原始函数 fn 和优化后的函数 compiled_fn 在输入 x 下的输出相等
        self.assertEqual(fn(x), compiled_fn(x))

    def test_storage_return(self):
        # 定义一个装饰器函数 fn，接受参数 x
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # 计算 x+1 的正弦值
            y = torch.sin(x + 1)
            # 获取 x 的未类型化存储，并将其大小调整为0
            storage = x.untyped_storage()
            storage.resize_(0)
            # 计算 y 的余弦值
            y = torch.cos(y)
            # 返回 y 和调整后的存储对象 storage
            return y, storage

        # 创建一个包含10个随机数的张量 x
        x = torch.randn(10)
        # 计算预期值，对 x+1 先求正弦再求余弦
        expected = torch.cos(torch.sin(x + 1))
        # 调用函数 fn，获取返回值 y 和存储对象 s
        y, s = fn(x)
        # 断言计算结果 y 与预期值 expected 相等
        self.assertEqual(y, expected)
        # 断言 x 的未类型化存储的大小为0
        self.assertEqual(x.untyped_storage().size(), 0)
        # 断言返回的存储对象 s 与 x 的未类型化存储对象相同
        self.assertIs(s, x.untyped_storage())

    def test_flat_name_to_original_fqn(self):
        # 定义一个名为 FooBarModule 的类，继承自 torch.nn.Module
        class FooBarModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个名为 "0" 的参数，值为形状为 (3, 4) 的随机张量
                self.register_parameter("0", torch.nn.Parameter(torch.randn(3, 4)))
                # 注册一个名为 "test_buf" 的缓冲区，值为形状为 (3, 4) 的随机张量
                self.register_buffer("test_buf", torch.randn(3, 4))
                # 注册一个名为 "test_param" 的参数，值为形状为 (3, 4) 的随机张量
                self.register_parameter(
                    "test_param", torch.nn.Parameter(torch.randn(3, 4))
                )

            # 定义前向传播函数，接受参数 x
            def forward(self, x):
                # 返回 ((x + self.test_buf) * getattr(self, "0")) / self.test_param 的计算结果
                return ((x + self.test_buf) * getattr(self, "0")) / self.test_param

        # 定义一个名为 TestModule 的类，继承自 torch.nn.Module
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个 FooBarModule 类的实例 foo_bar
                self.foo_bar = FooBarModule()
                # 注册一个名为 "test_param" 的参数，值为形状为 (3, 4) 的随机张量
                self.register_parameter(
                    "test_param", torch.nn.Parameter(torch.randn(3, 4))
                )
                # 注册一个名为 "test_buf" 的缓冲区，值为形状为 (3, 4) 的随机张量
                self.register_buffer("test_buf", torch.randn(3, 4))

            # 定义前向传播函数，接受参数 x
            def forward(self, x):
                # 返回 (self.foo_bar(x) + self.test_param) * self.test_buf 的计算结果
                return (self.foo_bar(x) + self.test_param) * self.test_buf

        # 导出 TestModule 类的计算图和元数据，gm 为计算图，_ 为其他元数据
        gm, _ = torch._dynamo.export(TestModule(), torch.randn(3, 4))
        # 断言导出的元数据中包含键为 "dynamo_flat_name_to_original_fqn" 的条目
        self.assertIn("dynamo_flat_name_to_original_fqn", gm.meta)
        # 定义预期的完全限定名称映射 expected_fqn
        expected_fqn = {
            "L__self___test_param": "test_param",
            "L__self___test_buf": "test_buf",
            "getattr_L__self___foo_bar___0__": "foo_bar.0",
            "L__self___foo_bar_test_param": "foo_bar.test_param",
            "L__self___foo_bar_test_buf": "foo_bar.test_buf",
        }
        # 断言导出的完全限定名称映射与预期的完全限定名称映射 expected_fqn 相等
        self.assertEqual(expected_fqn, gm.meta["dynamo_flat_name_to_original_fqn"])
    def test_shape_env_no_recording(self):
        # 创建一个 ShapeEnv 实例，不记录事件
        main = ShapeEnv(should_record_events=False)

        # 主 ShapeEnv 实例不应记录任何事件
        self.assertEqual(len(main.events), 0)

        # 调用 create_symbolic_sizes_strides_storage_offset 方法
        # 在主 ShapeEnv 上应用，生成一个包含 symbolic sizes 和 strides 的结果 r
        r = main.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )

        # 创建一个守卫条件：size[0] == 3 (调用 evaluate_expr)
        #   - +1 守卫条目
        #   - +1 替换条目
        size = r[0]
        bool(size[0] == 3)

        # 主 ShapeEnv 实例仍然不应记录任何事件
        self.assertEqual(len(main.events), 0)

        # 如果启用了翻译验证
        if torch.fx.experimental.validator.translation_validation_enabled():
            from torch.fx.experimental.symbolic_shapes import (
                CURRENT_NODE_KEY,
                SHAPEENV_EVENT_KEY,
            )

            # 检查在符号形状 FX 图中的节点上不存储任何记录元数据
            for n in main.graph.nodes:
                self.assertFalse(SHAPEENV_EVENT_KEY in n.meta)
                self.assertFalse(CURRENT_NODE_KEY in n.meta)

    def _replay_and_check(self, shape_env: ShapeEnv):
        # 如果 shape_env 实例应记录事件
        if shape_env.should_record_events:
            # 回放 shape_env 事件，并检查结果是否与原始 shape_env 相等
            replayed = replay_shape_env_events(shape_env.events)
            shape_env.check_equal(replayed)

    def test_shape_env_equal_empty(self):
        # 创建两个空的 ShapeEnv 实例 main 和 other
        main, other = ShapeEnv(), ShapeEnv()
        # 检查两个实例是否相等
        main.check_equal(other)
        # 对主实例执行事件重放和检查
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_constructor(self):
        # 创建两个 ShapeEnv 实例 main 和 other，其中 main 允许标量输出为 False
        main, other = ShapeEnv(allow_scalar_outputs=False), ShapeEnv()
        # 断言主实例检查与其他实例的相等性会引发 NotEqualError 异常
        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
    @onlyIfTranslationValidation
    # 定义测试函数，验证两个 ShapeEnv 对象的不相等情况
    def test_shape_env_equal_create_symbolic_sizes_strides_storage_offset(self):
        # 创建两个空的 ShapeEnv 对象 main 和 other
        main, other = ShapeEnv(), ShapeEnv()
        # 在 main 对象中创建具有符号大小、步幅、存储偏移的属性，并命名为 "x"
        main.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )
        # 断言调用 check_equal 方法时会引发 NotEqualError 异常
        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> name_to_node: values don't match.
  >  Left: {x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
  > Right: {}
==> source_to_symbol: values don't match.
  >  Left: {x.size()[0]: x.size()[0], x.size()[1]: x.size()[1], x.storage_offset(): x.storage_offset(), x.stride()[0]: x.stride()[0], x.stride()[1]: x.stride()[1]}
  > Right: {}
==> val_to_var: values don't match.
  >  Left: {0: 0, 1: 1, 2: s1, 3: s0}
  > Right: {0: 0, 1: 1}
==> var_to_range: values don't match.
  >  Left: {s0: VR[2, int_oo], s1: VR[2, int_oo]}
  > Right: {}
==> var_to_sources: values don't match.
  >  Left: {s0: [TensorPropertySource(base=ConstantSource(source_name='x'), prop=<TensorProperty.SIZE: 0>, idx=0)], s1: [TensorPropertySource(base=ConstantSource(source_name='x'), prop=<TensorProperty.SIZE: 0>, idx=1)]}
  > Right: {}
==> var_to_val: values don't match.
  >  Left: {s0: 3, s1: 2}
  > Right: {}
""",
        )
        # 调用内部方法 _replay_and_check 对 main 进行进一步检查
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    # 定义测试函数，验证两个 ShapeEnv 对象的不相等情况
    def test_shape_env_equal_unbacked(self):
        # 创建两个空的 ShapeEnv 对象 main 和 other
        main, other = ShapeEnv(), ShapeEnv()
        # 在 main 对象中创建未支持的符号整数
        main.create_unbacked_symint()
        # 在 main 对象中创建未支持的符号浮点数
        main.create_unbacked_symfloat()
        # 在 main 对象中创建未支持的符号布尔值
        main.create_unbacked_symbool()
        # 断言调用 check_equal 方法时会引发 NotEqualError 异常
        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> name_to_node: values don't match.
  >  Left: {u0, u1, zuf0}
  > Right: {}
==> unbacked_symfloat_counter: values don't match.
  >  Left: 1
  > Right: 0
==> unbacked_symint_counter: values don't match.
  >  Left: 2
  > Right: 0
==> var_to_range: values don't match.
  >  Left: {u0: VR[-int_oo, int_oo], u1: VR[0, 1], zuf0: VR[-oo, oo]}
  > Right: {}
""",
        )
        # 调用内部方法 _replay_and_check 对 main 进行进一步检查
        self._replay_and_check(main)
    def test_shape_env_equal_evaluate_expr_divisible(self):
        # 创建两个 ShapeEnv 实例对象，分别为 main 和 other
        main, other = ShapeEnv(), ShapeEnv()

        # 在 main 和 other 上分别调用 create_symbolic_sizes_strides_storage_offset 方法
        # 用随机生成的张量和常量作为参数
        r = main.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )
        other.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )

        # 创建一个守卫条件: size[0] % 3 == 0 （仅在 main ShapeEnv 中）
        #   - 添加一个守卫条目
        #   - 添加一个可被整除的条目
        size = r[0]
        bool(size[0] % 3 == 0)

        # 断言预期的内联异常是 NotEqualError
        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
# 测试函数，用于验证 ShapeEnv 对象的字段是否相等，并断言是否引发 NotEqualError 异常
@onlyIfTranslationValidation
def test_shape_env_equal_evaluate_expr_replacement(self):
    # 创建两个空的 ShapeEnv 对象 main 和 other
    main, other = ShapeEnv(), ShapeEnv()

    # 在 main 和 other 上调用 create_symbolic_sizes_strides_storage_offset 方法，
    # 并传入随机生成的 3x2 的张量和常量源 "x"
    r = main.create_symbolic_sizes_strides_storage_offset(
        torch.randn(3, 2), ConstantSource("x")
    )
    other.create_symbolic_sizes_strides_storage_offset(
        torch.randn(3, 2), ConstantSource("x")
    )

    # 创建一个保护条件：size[0] == 3（仅在 main ShapeEnv 中）
    #   - +1 保护条件条目
    #   - +1 替换条目
    size = r[0]
    bool(size[0] == 3)

    # 断言调用 main.check_equal(other) 是否引发 NotEqualError 异常，并验证异常信息是否符合预期
    self.assertExpectedRaisesInline(
        NotEqualError,
        lambda: main.check_equal(other),
        """\
ShapeEnv not equal: field values don't match:

==> guards: values don't match.
  >  Left: [Eq(s0, 3)]
  > Right: []
==> name_to_node: values don't match.
  >  Left: {_assert, eq, x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
  > Right: {x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
==> replacements: values don't match.
  >  Left: {s0: 3}
  > Right: {}
==> var_to_range: values don't match.
  >  Left: {s0: VR[3, 3], s1: VR[2, int_oo]}
  > Right: {s0: VR[2, int_oo], s1: VR[2, int_oo]}
""",
    )

    # 调用辅助方法 _replay_and_check 验证 main 对象
    self._replay_and_check(main)
    # 调用创建 ShapeEnv 实例
    main, other = ShapeEnv(), ShapeEnv()

    # 在 main 和 other 上调用 create_unbacked_symint 方法
    r = main.create_unbacked_symint()
    other.create_unbacked_symint()

    # 创建一个运行时断言：r % 3 == 0 （仅在主 ShapeEnv 中）
    #   - +1 deferred_runtime_asserts 条目
    #   - 改变：num_deferred_runtime_asserts
    expect_true(r % 3 == 0)

    # 断言预期的异常是 NotEqualError
    self.assertExpectedRaisesInline(
        NotEqualError,
        lambda: main.check_equal(other),
        """\
ShapeEnv not equal: field values don't match:

==> deferred_runtime_asserts: values don't match.
  >  Left: {u0: [Eq(PythonMod(u0, 3), 0)]}
  > Right: {}
==> name_to_node: values don't match.
  >  Left: {_assert, eq, mod, u0}
  > Right: {u0}
==> num_deferred_runtime_asserts: values don't match.
  >  Left: 1
  > Right: 0
""",
    )

    # 对 main 进行回放和检查
    self._replay_and_check(main)
    # 定义一个测试函数，用于验证字典的子类不能在图模式中初始化
    def test_dict_subclass_cannot_be_initialized_in_graph(self):
        # 遍历两个超类：OrderedDict 和 dict
        for super_class in (
            collections.OrderedDict,
            dict,
        ):

            # 定义一个自定义的字典类，继承自 super_class
            class CustomDict(super_class):
                # 初始化方法，调用超类的初始化方法
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)

            # 定义一个测试函数 fn，接受参数 x
            def fn(x):
                # 创建一个 CustomDict 的实例 c
                c = CustomDict()
                # 向字典 c 中添加一个键值对
                c["key"] = x
                # 断言 "key" 在 c 中
                assert "key" in c
                # 返回字典中 "key" 对应的值加 1
                return c["key"] + 1

            # 编译函数 fn，使用 torch 的 eager 模式，并且生成完整的计算图
            fn_opt = torch.compile(fn, backend="eager", fullgraph=True)
            # 使用断言验证编译后的函数在运行时会抛出特定异常
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "call_function UserDefinedClassVariable"
            ):
                # 打印编译后的函数对输入 torch.zeros(1) 的结果
                print(fn_opt(torch.zeros(1)))

    # 使用装饰器 wrapDeterministicFlagAPITest 包装的测试函数
    @wrapDeterministicFlagAPITest
    def test_backward_deterministic_mode_mismatch_warning(self):
        # 定义一个用于求和的函数 func，使用 torch 的 compile 装饰器
        @torch.compile
        def func(a, b):
            return a + b

        # 遍历 forward_deterministic 和 backward_deterministic 的组合
        for forward_deterministic, backward_deterministic in itertools.product(
            [True, False], [True, False]
        ):
            # 设置前向和反向算法的确定性模式
            torch.use_deterministic_algorithms(forward_deterministic)
            # 创建一个随机张量 a，要求其梯度
            a = torch.randn(10, requires_grad=True)
            # 调用 func 函数，并传入参数 a 和 1，得到结果 res
            res = func(a, 1)
            # 创建一个梯度张量 grad，形状与 res 相同，值为 1
            grad = torch.ones_like(res)
            # 设置反向算法的确定性模式
            torch.use_deterministic_algorithms(backward_deterministic)

            # 如果前向算法不是确定性的且反向算法是确定性的，则断言会抛出运行时异常
            if not forward_deterministic and backward_deterministic:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "^This compiled backward function is being run with torch\.use_deterministic_algorithms",
                ):
                    # 对结果 res 执行反向传播，并传入梯度 grad
                    res.backward(grad)

            else:
                # 对结果 res 执行反向传播，并传入梯度 grad
                res.backward(grad)

    # 测试 torch dynamo 代码生成的幂函数
    def test_torch_dynamo_codegen_pow(self):
        # 定义一个幂函数 pow，计算输入 x 的平方
        def pow(x):
            return x**2

        # 创建一个 numpy 数组 x，包含值 0 到 7
        x = np.arange(8)
        # 编译函数 pow，生成优化版本 pow_opt
        pow_opt = torch.compile(pow)

        # 运行优化后的 pow_opt 函数，并获取其结果 actual 和生成的源代码 source_code
        actual, source_code = run_and_get_code(pow_opt, x)
        # 计算预期结果 expect，即对输入 x 应用 pow 函数
        expect = pow(x)

        # 使用断言验证 actual 和 expect 相等
        self.assertEqual(expect, actual)

        # 使用断言验证 source_code 中不包含 "aten.pow"，即没有回退到普通的 pytorch 实现
        self.assertTrue(
            all("aten.pow" not in code for code in source_code),
            msg="Encountered an unexpected fallback to 'aten pow' in dynamo compiled code",
        )
    # 测试函数：test_graph_break_compilation_metrics
    def test_graph_break_compilation_metrics(self):
        # 内部函数定义，接受参数 x
        def fn(x):
            # 对 x 执行余弦函数操作
            x.cos()
            # 触发 Torch 内部的图断点事件
            torch._dynamo.graph_break()
            # 对 x 执行正弦函数操作
            x.sin()
            # 再次触发 Torch 内部的图断点事件
            torch._dynamo.graph_break()
            # 返回 x 的余弦值
            return x.cos()

        # 清空编译指标的统计数据
        torch._dynamo.utils.clear_compilation_metrics()
        # 创建一个随机的 4x4 的张量 x
        x = torch.rand((4, 4))
        # 编译函数 fn，使用 "eager" 后端
        f = torch.compile(fn, backend="eager")
        # 执行编译后的函数 f，传入参数 x
        f(x)
        # 获取编译指标数据
        metrics = torch._dynamo.utils.get_compilation_metrics()

        # 检查第一个事件的重启原因
        (restart_reason,) = metrics[0].restart_reasons
        self.assertTrue(
            "skip function graph_break" in restart_reason,
            "Should have logged graph break reason",
        )
        # 检查第一个事件的动态时间小于或等于整个帧的编译时间
        self.assertTrue(
            metrics[0].dynamo_time_before_restart_s
            <= metrics[0].entire_frame_compile_time_s
        )

        # 检查第二个事件的重启原因
        (restart_reason,) = metrics[1].restart_reasons
        self.assertTrue(
            "skip function graph_break" in restart_reason,
            "Should have logged graph break reason",
        )
        # 检查第二个事件的动态时间小于或等于整个帧的编译时间
        self.assertTrue(
            metrics[1].dynamo_time_before_restart_s
            <= metrics[1].entire_frame_compile_time_s
        )

        # 检查最后一个编译事件没有重启原因
        self.assertTrue(
            len(metrics[2].restart_reasons) == 0, "Last compile has no graph break"
        )
        # 检查最后一个编译事件的动态时间为 0
        self.assertTrue(metrics[2].dynamo_time_before_restart_s == 0)

    # 测试函数：test_graph_break_compilation_metrics_on_failure
    def test_graph_break_compilation_metrics_on_failure(self):
        # 内部函数定义，接受参数 x
        def fn(x):
            # 返回 x 的正弦值
            return x.sin()

        # 定义一个会失败的后端函数
        def broken_backend(gm, example_inputs):
            raise RuntimeError("broken backend")

        # 创建一个随机的 4x4 的张量 x
        x = torch.rand((4, 4))
        # 编译函数 fn，使用自定义的失败后端 broken_backend
        f = torch.compile(fn, backend=broken_backend)
        
        # 使用 unittest.mock.patch 方法，设置 suppress_errors 为 True，以忽略错误
        with unittest.mock.patch("torch._dynamo.config.suppress_errors", True):
            # 清空编译指标的统计数据
            torch._dynamo.utils.clear_compilation_metrics()
            # 执行编译后的函数 f，传入参数 x
            f(x)
            # 获取编译指标数据
            metrics = torch._dynamo.utils.get_compilation_metrics()
            # 遍历所有指标数据
            for metric in metrics:
                # 检查动态时间大于 0
                self.assertTrue(metric.dynamo_time_before_restart_s > 0)
                # 检查失败原因中是否包含 "RuntimeError: broken backend"
                self.assertTrue(
                    "RuntimeError: broken backend" in metric.fail_reason,
                    "Should have logged fail reason",
                )
    # 定义一个测试函数，用于测试编译指标的大小限制
    def test_compilation_metrics_size_limit(self):
        # 定义函数 fn1，对输入进行 ReLU 激活操作
        def fn1(x):
            return x.relu()

        # 定义函数 fn2，对输入进行余弦操作
        def fn2(x):
            return x.cos()

        # 定义函数 fn3，对输入进行正弦操作
        def fn3(x):
            return x.sin()

        # 定义函数 fn4，对输入进行指数操作
        def fn4(x):
            return x.exp()

        # 导入上下文管理模块 contextlib

        # 定义上下文管理器 metrics_limit_ctx，用于限制编译指标数量
        @contextlib.contextmanager
        def metrics_limit_ctx():
            try:
                # 设置编译指标限制为 3
                torch._dynamo.utils.set_compilation_metrics_limit(3)
                yield
            finally:
                # 恢复默认的编译指标限制
                torch._dynamo.utils.set_compilation_metrics_limit(
                    torch._dynamo.utils.DEFAULT_COMPILATION_METRICS_LIMIT
                )

        # 创建一个 4x4 的随机张量 x
        x = torch.rand((4, 4))
        # 重置 Torch 的动态编译环境
        torch._dynamo.reset()
        # 对 fn1 到 fn4 分别进行编译，使用 "eager" 后端
        torch.compile(fn1, backend="eager")(x)
        torch.compile(fn2, backend="eager")(x)
        torch.compile(fn3, backend="eager")(x)
        torch.compile(fn4, backend="eager")(x)

        # 使用 metrics_limit_ctx 上下文管理器，测试编译指标的限制
        with metrics_limit_ctx():
            # 清除编译指标
            torch._dynamo.utils.clear_compilation_metrics()
            # 重置 Torch 的动态编译环境
            torch._dynamo.reset()
            # 断言当前编译指标列表的长度为 0
            self.assertEqual(0, len(torch._dynamo.utils.get_compilation_metrics()))
            # 对 fn1 进行编译，预期编译指标列表长度为 1
            torch.compile(fn1, backend="eager")(x)
            self.assertEqual(1, len(torch._dynamo.utils.get_compilation_metrics()))
            # 对 fn2 进行编译，预期编译指标列表长度为 2
            torch.compile(fn2, backend="eager")(x)
            self.assertEqual(2, len(torch._dynamo.utils.get_compilation_metrics()))
            # 对 fn3 进行编译，预期编译指标列表长度为 3
            torch.compile(fn3, backend="eager")(x)
            self.assertEqual(3, len(torch._dynamo.utils.get_compilation_metrics()))
            # 对 fn4 进行编译，预期编译指标列表长度仍为 3，因为已达到限制
            torch.compile(fn4, backend="eager")(x)
            self.assertEqual(3, len(torch._dynamo.utils.get_compilation_metrics()))

    # 定义另一个测试函数 test_funcname_cache
    def test_funcname_cache(self):
        # 定义源代码字符串 src
        src = """\
import torch
# 导入 PyTorch 库

if True:
    test = 3
    # 定义变量 test，并赋值为 3

class AAA:
    class DUMMY:
        class DUMMY2:
            pass
        # 定义嵌套类 DUMMY2

    def dummy(self):
        def dummy2():
            pass
        # 定义函数 dummy2 在函数 dummy 内部

    class BBB:
        @staticmethod
        def CCC():
            class DDD:
                if True:
                    @staticmethod
                    def EEE():
                        x = [torch.ones(3, 3) for _ in range(5)]
                        return x
            return DDD
        # 定义静态方法 CCC，内部定义嵌套类 DDD，以及静态方法 EEE 返回一个包含 5 个 3x3 全为1的张量列表

def fn():
    return 3
    # 定义函数 fn 返回整数 3

"""
        with tempfile.NamedTemporaryFile(mode="w") as f:
            f.write(src)
            f.flush()
            from torch._dynamo.funcname_cache import get_funcname

            names = [get_funcname(f.name, i + 1) for i in range(src.count("\n") + 1)]

        self.assertExpectedInline(
            "\n".join(names),
            """\

            # 使用临时文件写入 src 内容，并获取函数名列表 names
            # 断言函数名列表与预期的输出结果一致

AAA
AAA.DUMMY
AAA.DUMMY.DUMMY2
AAA.DUMMY.DUMMY2
AAA.DUMMY.DUMMY2
AAA.dummy
AAA.dummy.dummy2
AAA.dummy.dummy2
AAA.BBB
AAA.BBB
AAA.BBB.CCC
AAA.BBB.CCC.DDD
AAA.BBB.CCC.DDD
AAA.BBB.CCC.DDD
AAA.BBB.CCC.DDD.EEE
AAA.BBB.CCC.DDD.EEE
AAA.BBB.CCC.DDD.EEE
AAA.BBB.CCC
fn
fn
""",
        )

    def test_return_dict_with_graph_break_and_update(self):
        def create():
            torch._dynamo.graph_break()
            return {0: torch.tensor(3)}
            # 定义函数 create 用于创建包含张量的字典

        def fn():
            return {**create()}
            # 定义函数 fn，返回调用 create 函数的结果

        opt_fn = torch.compile(backend="eager")(fn)
        result = opt_fn()
        # 编译函数 fn 并调用执行，保存结果至 result

        self.assertIn(0, result)
        self.assertTrue(same(result[0], torch.tensor(3)))
        # 断言结果中包含键 0，且值与 torch.tensor(3) 相等

    def test_dynamo_reset_clears_cache(self):
        """Test that dynamo bytecode cache is freed
        when dynamo reset is called
        """

        def fn(x):
            return torch.sin(x)
            # 定义函数 fn，返回输入张量 x 的正弦值

        opt_fn = torch.compile(backend="eager")(fn)
        opt_fn(torch.randn(3, 3))
        # 编译函数 fn 并传入大小为 3x3 的随机张量进行调用

        c1 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c1), 1)
        # 获取函数 fn 的缓存条目列表 c1，并断言其长度为 1

        torch._dynamo.reset()
        c2 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c2), 0)
        # 重置 Torch 动态编译器缓存，并再次获取函数 fn 的缓存条目列表 c2，断言其长度为 0

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_guard_size_oblivious(self):
        # This code, in fact, does NOT work in eager
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = torch.zeros(x.item())
            if guard_size_oblivious(y.size(0) == 0):
                assert False
            return y
            # 定义函数 fn，根据输入张量 x 的大小创建零张量 y，并在条件成立时断言错误

        self.assertEqual(fn(torch.tensor([0])), torch.zeros(0))
        # 断言调用 fn 函数传入零张量后返回 torch.zeros(0)

    def test_guard_size_oblivious_backed(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            y = x.size(0)
            # This doesn't actually do anything
            if guard_size_oblivious(y == 0):
                return torch.randn(1)
            else:
                return torch.randn(2)
            # 定义函数 f，根据输入张量 x 的大小返回不同形状的随机张量

        # Should not fail in either case
        self.assertEqual(f(torch.randn(0)).shape, (1,))
        self.assertEqual(f(torch.randn(2)).shape, (2,))
        # 断言调用 f 函数传入大小为 0 和 2 的随机张量后返回的张量形状分别为 (1,) 和 (2,)
    def _test_compile_model_free(self, model_inp_ctr, weakref_watch):
        """
        Args:
        model_inp_ctr
            - 一个构造函数，返回一个新的模型和输入到该模型的数据
        weakref_watch
            - 一个函数，返回模型的某一层用于 weakref 监视，以便在模型超出作用域后检查该层是否被释放
        """
        cleared = False

        def finalize():
            nonlocal cleared
            cleared = True

        def run():
            # 调用 model_inp_ctr 函数获取模型和输入数据
            mod, inp = model_inp_ctr()
            # 使用 weakref.finalize 设置 finalize 函数来监视 weakref_watch(mod) 返回的对象
            weakref.finalize(weakref_watch(mod), finalize)
            # 编译模型 mod，并执行使用输入数据 inp
            torch.compile(mod, backend="eager")(inp)

        run()
        gc.collect()
        self.assertTrue(cleared)

    def test_custom_module_free(self):
        """测试当模型超出作用域时，模型是否被释放"""

        class Mod(torch.nn.Module):
            def __init__(self):
                super(Mod, self).__init__()
                self.fc = torch.nn.Linear(100, 100)

            def forward(self, out):
                return self.fc(out)

        self._test_compile_model_free(
            lambda: (Mod(), torch.randn(100, 100)),
            lambda mod: mod.fc,
        )

    def test_sequential_module_free(self):
        self._test_compile_model_free(
            lambda: (
                torch.nn.Sequential(
                    torch.nn.Linear(100, 100),
                    torch.nn.ReLU(),
                ),
                torch.randn(100, 100),
            ),
            lambda mod: mod[0],
        )

    def test_linear_module_free(self):
        self._test_compile_model_free(
            lambda: (torch.nn.Linear(100, 100), torch.randn(100, 100)),
            lambda mod: mod,
        )

    def test_outside_linear_module_free(self):
        # 与 test_linear_module_free 相比，线性层不是直接编译的代码对象。

        # 由于难以处理变量 fc，此测试不使用 _test_compile_model_free。

        cleared = False

        def finalize():
            nonlocal cleared
            cleared = True

        def run():
            # 创建线性层对象 fc
            fc = torch.nn.Linear(100, 100)

            class Mod(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc_ref = fc

                def forward(self, x):
                    return self.fc_ref(x)

            # 创建模型对象 mod
            mod = Mod()
            # 创建输入数据 inp
            inp = torch.randn(100, 100)
            # 使用 weakref.finalize 设置 finalize 函数来监视 fc 对象
            weakref.finalize(fc, finalize)
            # 编译模型 mod，并执行使用输入数据 inp
            torch.compile(mod, backend="eager")(inp)

        run()
        gc.collect()
        self.assertTrue(cleared)
    def test_parameter_free(self):
        # 定义一个内部函数 model_inp_ctr，用于创建模型和参数
        def model_inp_ctr():
            # 创建一个随机参数张量，并将其封装成 torch.nn.Parameter 对象
            param = torch.nn.Parameter(torch.randn(100, 100))

            # 定义一个模型类 Mod，继承自 torch.nn.Module
            class Mod(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 将参数 param 加入到模型中
                    self.param = param

                # 定义模型的前向传播方法
                def forward(self, x):
                    return self.param * x[0]

            # 返回创建的模型对象和参数
            return Mod(), (torch.randn(100, 100), param)

        # 调用测试方法 _test_compile_model_free，传入 model_inp_ctr 函数和一个 lambda 表达式，lambda 返回 mod.param
        self._test_compile_model_free(model_inp_ctr, lambda mod: mod.param)

    def test_conditional_list_comp_in_context(self):
        # 定义一个函数 fn，接受一个输入 inp
        def fn(inp):
            try:
                # 尝试对输入中不为 None 的每个元素计算正弦值，返回列表
                return [torch.sin(x) for x in inp if x is not None]
            except Exception:
                pass

        # 创建一个输入列表 inp，包含三个随机张量和一个 None 值
        inp = [torch.randn(3, 3) for _ in range(3)] + [None]
        
        # 使用 torch.compile 函数编译 fn 函数，指定 backend 为 "eager"
        opt_fn = torch.compile(fn, backend="eager")
        
        # 调用编译后的函数 opt_fn，传入 inp 作为参数
        opt_fn(inp)

    def test_312_binary_slice_with_graph_break1(self):
        # 创建两个线性层对象 l1 和 l2
        l1 = torch.nn.Linear(5, 5)
        l2 = torch.nn.Linear(5, 5)

        # 定义一个函数 fn，接受输入 x
        def fn(x):
            # 创建一个顺序容器 n，包含 l1 和 l2 两个线性层
            n = torch.nn.Sequential(l1, l2)
            
            # 对容器 n 进行切片操作，并对输入 x 进行前向传播
            out = n[1:](x)
            
            # 返回前向传播的结果 out
            return out

        # 使用 torch.compile 函数编译 fn 函数，指定 backend 为 "eager"
        opt_fn = torch.compile(fn, backend="eager")
        
        # 调用编译后的函数 opt_fn，传入随机张量作为参数
        opt_fn(torch.randn(5, 5))

    def test_312_binary_slice_with_graph_break2(self):
        # 定义一个类 Foo
        class Foo:
            # 定义 __setitem__ 方法，用于赋值操作
            def __setitem__(self, key, val):
                pass

            # 定义 __getitem__ 方法，用于获取操作
            def __getitem__(self, key):
                # 调用 torch._dynamo.graph_break() 方法，用于中断图计算
                torch._dynamo.graph_break()
                return 1

        # 创建 Foo 类的实例 foo
        foo = Foo()

        # 定义一个函数 fn，接受输入 x
        def fn(x):
            # 在对象 foo 上执行切片赋值操作
            foo[:] = x
            
            # 使用二进制切片操作 foo[:]，并将结果与输入 x 相加
            x = x + foo[:]
            
            # 如果 x 是 None，则将其加 1；否则也将其加 1
            if x is None:
                x = x + 1
            else:
                x = x + 1
            
            # 返回结果 x
            return x

        # 使用 torch.compile 函数编译 fn 函数，指定 backend 为 "eager"
        opt_fn = torch.compile(fn, backend="eager")
        
        # 调用编译后的函数 opt_fn，传入随机张量作为参数
        opt_fn(torch.randn(5, 5))

    def test_super_after_graph_break(self):
        # 定义一个继承自 torch.nn.Sequential 的类 Foo
        class Foo(torch.nn.Sequential):
            # 定义构造方法 __init__，在其中调用 torch._dynamo.graph_break() 方法
            def __init__(self, layers):
                torch._dynamo.graph_break()
                super().__init__(*layers)

        # 定义一个函数 fn，接受输入 x
        def fn(x):
            # 创建包含三个线性层的列表 layers
            layers = [torch.nn.Linear(3, 3) for _ in range(3)]
            
            # 创建 Foo 类的实例 mod，传入 layers 作为参数
            mod = Foo(layers)
            
            # 对模型 mod 执行前向传播，传入输入 x
            return mod(x)

        # 使用 torch.compile 函数编译 fn 函数，指定 backend 为 "eager"
        opt_fn = torch.compile(fn, backend="eager")
        
        # 调用编译后的函数 opt_fn，传入随机张量作为参数
        opt_fn(torch.randn(3, 3))

    def test_load_fast_and_clear_graph_break(self):
        # 定义一个函数 fn，没有参数
        def fn():
            # 创建一个张量列表，包含不同行数的随机张量，然后使用 torch.cat 进行拼接
            out = torch.cat([torch.randn(r, 5) for r in range(3)])
            
            # 调用 torch._dynamo.graph_break() 方法，中断图计算
            torch._dynamo.graph_break()
            
            # 再次创建一个张量列表，包含不同行数的随机张量，使用 torch.cat 进行拼接
            out = torch.cat([torch.randn(r, 5) for r in range(3)])
            
            # 返回拼接后的张量 out
            return out

        # 使用 torch._dynamo.optimize("eager")(fn)() 执行优化后的函数 fn，获取结果的形状并断言
        self.assertEqual(torch._dynamo.optimize("eager")(fn)().shape, (3, 5))
    def test_raises_importerror1(self):
        # 使用 torch.compile 注释装饰器编译函数 fn，使用 eager 后端
        @torch.compile(backend="eager")
        def fn(x):
            try:
                # 尝试导入一个肯定不存在的模块
                import some_module_that_surely_does_not_exist

                # 如果成功导入，直接返回
                return
            except ImportError:
                pass
            # 如果导入失败，执行下面的操作
            return x.sin()  # 返回 x 的正弦值

        x = torch.randn(8)
        # 断言调用 fn(x) 结果与 x.sin() 相等
        self.assertEqual(fn(x), x.sin())

    def test_raises_importerror2(self):
        # 使用 torch.compile 注释装饰器编译函数 fn，使用 eager 后端
        @torch.compile(backend="eager")
        def fn(x):
            # 尝试导入一个肯定不存在的模块
            import some_module_that_surely_does_not_exist

            # 如果成功导入，返回 x + 1
            return x + 1

        x = torch.randn(8)
        # 断言调用 fn(x) 会抛出 ImportError 异常
        with self.assertRaises(ImportError):
            fn(x)

    def test_dynamo_cache_move_to_front(self):
        def fn(x, const):
            # 返回 x 加上常量 const
            return x + const

        # 使用 torch.compile 编译函数 fn，使用 eager 后端，设置 dynamic=False 强制 Dynamo 重新编译
        opt_fn = torch.compile(fn, backend="eager", dynamic=False)

        inp = torch.randn(3, 3)

        # 注意：假设每个缓存条目都由唯一的 Mod 实例保护
        opt_fn(inp, 1)
        opt_fn(inp, 2)
        opt_fn(inp, 3)

        # 获取函数 fn 缓存条目的列表
        c1 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c1), 3)

        # 将缓存条目移到最前面
        opt_fn(inp, 2)
        c2 = _debug_get_cache_entry_list(fn.__code__)
        self.assertIs(c1[1], c2[0])

    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=False)
    def test_dynamo_cache_invalidate(self):
        # 定义一个简单的 torch.nn.Module 子类
        class Mod(torch.nn.Module):
            def __init__(self):
                super(Mod, self).__init__()
                self.fc = torch.nn.Linear(3, 3)

            def forward(self, out):
                return self.fc(out)

        def fn(x, mod):
            # 调用模块 mod 的 forward 方法
            return mod(x)

        # 使用 torch.compile 编译函数 fn，使用 eager 后端
        opt_fn = torch.compile(fn, backend="eager")

        m1 = Mod()
        m2 = Mod()
        m3 = Mod()
        inp = torch.randn(3, 3)

        # 注意：假设每个缓存条目都由唯一的 Mod 实例保护
        opt_fn(inp, m1)
        opt_fn(inp, m2)
        opt_fn(inp, m3)

        # 获取函数 fn 缓存条目的列表
        c1 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c1), 3)

        # 将缓存条目移到最前面
        opt_fn(inp, m2)
        c2 = _debug_get_cache_entry_list(fn.__code__)
        self.assertIs(c1[1], c2[0])

        # 删除缓存中心的条目
        del m3
        c3 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c3), 2)
        self.assertIs(c3[0], c2[0])
        self.assertIs(c3[1], c2[2])

        # 删除缓存末尾的条目
        del m1
        c4 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c4), 1)
        self.assertIs(c4[0], c3[0])

        # 删除最后一个缓存条目
        del m2
        c5 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c5), 0)
    # 定义一个测试方法，用于测试梯度为 None 的情况
    def test_grad_none(self):
        # 定义一个内部函数 fn，接受两个参数 x 和 y
        def fn(x, y):
            # 设置 x 的梯度为 y 的绝对值
            x.grad = torch.abs(y)
            # 将 y 加到 x 的梯度上
            x.grad.add_(y)
            # 返回 y 的绝对值
            return torch.abs(y)

        # 创建一个 tensor y，其值为从 0 到 3 的序列，形状为 (2, 2)，转换为 float 类型
        y = torch.arange(4).reshape(2, 2).to(torch.float)
        # 创建一个形状为 (2, 2) 的随机张量 x
        x = torch.randn(2, 2)
        # 将 x 的梯度设置为 None
        x.grad = None

        # 调用 fn 函数计算 z
        z = fn(x, y)
        # 克隆 z 并断开计算图的连接，得到参考的 y
        ref_y = torch.clone(z).detach()
        # 克隆 x 的梯度并断开计算图的连接，得到参考的 x.grad
        ref_x_grad = torch.clone(x.grad).detach()

        # 重新设置 y 和 x，重复上述过程以验证优化后的函数
        y = torch.arange(4).reshape(2, 2).to(torch.float)
        x = torch.randn(2, 2)
        x.grad = None

        # 使用 torch.compile 方法编译 fn 函数，指定后端为 "eager"
        opt_fn = torch.compile(fn, backend="eager")
        # 通过优化后的函数计算 z
        z = opt_fn(x, y)
        # 断言优化后的结果与参考的 y 相等
        self.assertEqual(z, ref_y)
        # 断言优化后的 x.grad 与参考的 x.grad 相等
        self.assertEqual(x.grad, ref_x_grad)

    # 定义一个测试方法，用于测试梯度非 None 的情况
    def test_grad_non_none(self):
        # 定义一个内部函数 fn，接受两个参数 x 和 y
        def fn(x, y):
            # 将 y 加到 x 的梯度上
            x.grad.add_(y)
            # 返回 y 的绝对值
            return torch.abs(y)

        # 创建一个全为 1 的形状为 (2, 2) 的张量 y
        y = torch.ones(2, 2)
        # 创建一个形状为 (2, 2) 的随机张量 x
        x = torch.randn(2, 2)
        # 将 x 的梯度设置为从 0 到 3 的序列，转换为 float 类型，形状为 (2, 2)
        x.grad = torch.arange(4).reshape(2, 2).to(torch.float)

        # 调用 fn 函数计算 z
        z = fn(x, y)
        # 克隆 z 并断开计算图的连接，得到参考的 y
        ref_y = torch.clone(z).detach()
        # 克隆 x 的梯度并断开计算图的连接，得到参考的 x.grad
        ref_x_grad = torch.clone(x.grad).detach()

        # 重新设置 y 和 x，重复上述过程以验证优化后的函数
        y = torch.ones(2, 2)
        x = torch.randn(2, 2)
        x.grad = torch.arange(4).reshape(2, 2).to(torch.float)

        # 使用 torch._dynamo.testing.CompileCounterWithBackend 方法计数编译调用次数，指定后端为 "eager"
        cnt = torch._dynamo.testing.CompileCounterWithBackend("eager")
        # 使用 torch.compile 方法编译 fn 函数，指定后端为 cnt
        opt_fn = torch.compile(fn, backend=cnt)
        # 通过优化后的函数计算 z
        z = opt_fn(x, y)

        # 确保生成的计算图只返回一个输出。我们希望梯度加法操作 add_ 能够作为图的一部分，
        # 以便归纳器可以在合适的位置移动 add_ 和 resulting copy_ 节点来释放内存。
        self.assertEqual(len(list(cnt.graphs[0].graph.nodes)[-1].all_input_nodes), 1)
        # 断言优化后的结果与参考的 y 相等
        self.assertEqual(z, ref_y)
        # 断言优化后的 x.grad 与参考的 x.grad 相等
        self.assertEqual(x.grad, ref_x_grad)

    # 定义一个测试方法，用于测试使用整数列表进行 torch.Tensor.new 方法的行为是否与 dynamo 一致
    def test_new_with_int_list(self):
        def fn(x):
            # 使用 x 的大小创建一个新的张量，并加上 5
            return x.new(*x.size()) + 5

        # 使用 torch.compile 方法编译 fn 函数，指定后端为 "eager"
        optfn = torch.compile(backend="eager")(fn)

        # 创建一个从 0 到 9 的张量，形状为 (2, 5)
        x = torch.arange(10).view(2, 5)

        # 计算预期结果
        expected = fn(x)
        # 通过优化后的函数计算实际结果
        actual = optfn(x)

        # 断言预期结果和实际结果的数据类型相等
        self.assertEqual(expected.dtype, actual.dtype)
        # 断言预期结果和实际结果的形状相等
        self.assertEqual(expected.shape, actual.shape)
        # 断言预期结果和实际结果的步长相等
        self.assertEqual(expected.stride(), actual.stride())
        # 断言预期结果和实际结果的存储偏移相等
        self.assertEqual(expected.storage_offset(), actual.storage_offset())

    # 使用 torch._dynamo.config.patch(guard_nn_modules=True) 装饰器定义一个测试方法，用于测试是否存在 nn 模块的保护
    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_hasattr_nn_module_guard(self):
        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个线性层 self.a，输入和输出都是 3
                self.a = torch.nn.Linear(3, 3)

            # 定义 forward 方法，接受输入 x
            def forward(self, x):
                # 如果 self 中有属性 "a"，则返回 self.a(x)
                if hasattr(self, "a"):
                    return self.a(x)
                else:
                    return x

        # 创建类 M 的实例 m
        m = M()
        # 创建一个形状为 (3, 3) 的随机张量 x
        x = torch.randn(3, 3)
        # 计算参考结果 ref，调用 m 的 forward 方法
        ref = m(x)

        # 使用 torch.compile 方法编译类 M 的实例 m 的 forward 方法，指定后端为 "eager"
        opt_m = torch.compile(backend="eager")(m)
        # 通过优化后的方法计算结果 res
        res = opt_m(x)

        # 断言优化后的结果与参考结果相等
        self.assertEqual(ref, res)
    def test_ordered_dict_move_to_end(self):
        # 创建一个普通字典
        d = {
            "foo": 1,
            "bar": 2,
        }

        # 将普通字典转换为有序字典
        d = collections.OrderedDict(d)
        
        # 将键 "foo" 移动到有序字典的尾部
        d.move_to_end("foo")

        # 定义一个使用 torch 编译的函数
        @torch.compile(backend="eager")
        def fn(x, d):
            # 返回 x 与字典中 "foo" 和 "bar" 对应值的乘积
            return x * d["foo"] * d["bar"]

        # 调用函数 fn，并传入参数 x 和有序字典 d
        fn(torch.randn(4), d)

        # 使用 unittest.mock.patch 来设置 torch._dynamo.config.error_on_recompile 为 True
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            # 再次调用函数 fn，并传入参数 x 和有序字典 d
            fn(torch.randn(4), d)

    def test_defaultdict(self):
        # 创建一个默认字典 defaultdict
        d = collections.defaultdict()
        
        # 向默认字典中添加键值对
        d["foo"] = 1
        d["bar"] = 2

        # 定义一个使用 torch 编译的函数
        @torch.compile(backend="eager")
        def fn(x, d):
            # 返回 x 与字典中 "foo" 和 "bar" 对应值的乘积
            return x * d["foo"] * d["bar"]

        # 调用函数 fn，并传入参数 x 和默认字典 d
        fn(torch.randn(4), d)

        # 使用 unittest.mock.patch 来设置 torch._dynamo.config.error_on_recompile 为 True
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            # 再次调用函数 fn，并传入参数 x 和默认字典 d
            fn(torch.randn(4), d)

    def test_custom_dict(self):
        # 定义一个继承自 dict 的自定义字典类 MyDict
        class MyDict(dict):
            pass

        # 创建一个普通字典
        d = {
            "foo": 1,
            "bar": 2,
        }

        # 将普通字典转换为自定义字典类 MyDict 的实例
        d = MyDict(d)

        # 定义一个使用 torch 编译的函数
        @torch.compile(backend="eager")
        def fn(x, d):
            # 返回 x 与字典中 "foo" 和 "bar" 对应值的乘积
            return x * d["foo"] * d["bar"]

        # 调用函数 fn，并传入参数 x 和自定义字典 d
        fn(torch.randn(4), d)

        # 使用 unittest.mock.patch 来设置 torch._dynamo.config.error_on_recompile 为 True
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            # 再次调用函数 fn，并传入参数 x 和自定义字典 d
            fn(torch.randn(4), d)

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True)
    def test_interpolate_propagate_real_tensors(self):
        # 定义一个使用 torch 编译的函数，fullgraph=True 表示编译整个图形
        @torch.compile(backend="eager", fullgraph=True)
        def f(mask, box):
            # u0, u1 = mask.tolist()
            # 创建一个随机张量 mask，并指定设备为 "cuda"
            mask = torch.randn(1, 1, 30, 30, device="cuda")
            h, w = box.tolist()
            # 使用双线性插值法调整 mask 的大小至 (h, w)，align_corners=False 表示不对齐角点
            return torch.nn.functional.interpolate(
                mask, (h, w), mode="bilinear", align_corners=False
            )

        # 调用函数 f，并传入参数 mask 和 box
        f(torch.tensor([30, 30], device="cuda"), torch.tensor([68, 32], device="cuda"))

    def test_custom_iter_dict(self):
        # 定义一个继承自 dict 的自定义字典类 ReversedDict，重写 __iter__ 方法
        class ReversedDict(dict):
            def __iter__(self):
                # 返回反向排序后的键列表的迭代器
                return reversed(list(self.keys()))

        # 创建一个普通字典
        d = {
            "foo": 1,
            "bar": 2,
        }

        # 将普通字典转换为自定义字典类 ReversedDict 的实例
        d = ReversedDict(d)

        # 定义一个使用 torch 编译的函数
        @torch.compile(backend="eager")
        def fn(x, d):
            # 返回 x 与字典中 "foo" 和 "bar" 对应值的乘积
            return x * d["foo"] * d["bar"]

        # 调用函数 fn，并传入参数 x 和自定义字典 d
        fn(torch.randn(4), d)

        # 使用 unittest.mock.patch 来设置 torch._dynamo.config.error_on_recompile 为 True
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            # 再次调用函数 fn，并传入参数 x 和自定义字典 d
            fn(torch.randn(4), d)

    def test_custom_keys_iter_dict(self):
        # 定义一个继承自 dict 的自定义字典类 ReversedDict，重写 keys 方法
        class ReversedDict(dict):
            def keys(self):
                # 返回键的列表，按指定顺序排序
                return ["bar", "foo"]

        # 创建一个普通字典
        d = {
            "foo": 1,
            "bar": 2,
        }

        # 将普通字典转换为自定义字典类 ReversedDict 的实例
        d = ReversedDict(d)

        # 定义一个使用 torch 编译的函数
        @torch.compile(backend="eager")
        def fn(x, d):
            # 返回 x 与字典中 "foo" 和 "bar" 对应值的乘积
            return x * d["foo"] * d["bar"]

        # 调用函数 fn，并传入参数 x 和自定义字典 d
        fn(torch.randn(4), d)

        # 使用 unittest.mock.patch 来设置 torch._dynamo.config.error_on_recompile 为 True
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            # 再次调用函数 fn，并传入参数 x 和自定义字典 d
            fn(torch.randn(4), d)
    def test_dict_guard_on_keys_order(self):
        # 创建一个字典 d，包含键值对 {2: 4, 3: 5}
        d = {
            2: 4,
            3: 5,
        }

        # 创建一个编译计数器 cnts 对象
        cnts = torch._dynamo.testing.CompileCounter()

        # 定义一个函数 fn，接受参数 x 和 d，并对 d 中的每个键值对进行处理
        def fn(x, d):
            # 遍历字典 d 的键值对
            for key, value in d.items():
                # 对 x 进行乘法和加法操作
                x = x * key + value
            return x

        # 使用 torch.compile 对 fn 进行编译，使用 cnts 作为后端
        opt_fn = torch.compile(fn, backend=cnts)
        # 调用 opt_fn，并传入参数 torch.randn(4) 和字典 d
        opt_fn(torch.randn(4), d)
        opt_fn(torch.randn(4), d)
        # 断言编译帧数为 1，表明没有重新编译
        self.assertEqual(cnts.frame_count, 1)

        # 将键为 2 的值移到字典末尾
        d[2] = d.pop(2)

        # 创建一个随机张量 x
        x = torch.randn(4)
        # 使用 opt_fn 处理 x 和更新后的字典 d，得到结果 res
        res = opt_fn(x, d)
        # 断言编译帧数为 2，表明发生了重新编译
        self.assertEqual(cnts.frame_count, 2)
        # 断言 opt_fn 的输出结果与未优化的 fn 函数处理结果一致
        self.assertEqual(res, fn(x, d))

    def test_dict_guard_on_keys_order2(self):
        # 创建一个字典 d，包含键值对 {2: 4, 3: 5}
        d = {
            2: 4,
            3: 5,
        }

        # 创建一个编译计数器 cnts 对象
        cnts = torch._dynamo.testing.CompileCounter()

        # 定义一个函数 fn，接受参数 x 和 d，并对 d 中的每个键进行处理
        def fn(x, d):
            # 遍历字典 d 的键
            for key in d:
                # 获取键对应的值
                value = d[key]
                # 对 x 进行乘法和加法操作
                x = x * key + value
            return x

        # 使用 torch.compile 对 fn 进行编译，使用 cnts 作为后端
        opt_fn = torch.compile(fn, backend=cnts)
        # 调用 opt_fn，并传入参数 torch.randn(4) 和字典 d
        opt_fn(torch.randn(4), d)
        opt_fn(torch.randn(4), d)
        # 断言编译帧数为 1，表明没有重新编译
        self.assertEqual(cnts.frame_count, 1)

        # 将键为 2 的值移到字典末尾
        d[2] = d.pop(2)

        # 创建一个随机张量 x
        x = torch.randn(4)
        # 使用 opt_fn 处理 x 和更新后的字典 d，得到结果 res
        res = opt_fn(x, d)
        # 断言编译帧数为 2，表明发生了重新编译
        self.assertEqual(cnts.frame_count, 2)
        # 断言 opt_fn 的输出结果与未优化的 fn 函数处理结果一致
        self.assertEqual(res, fn(x, d))

    def test_contains_dunder_dict(self):
        # 定义一个名为 UserDefined 的类
        class UserDefined:
            # 类初始化方法，设置属性 a 和 b
            def __init__(self):
                self.a = 3
                self.b = 5

            # 类方法 run，接受参数 x，根据对象的属性对 x 进行运算
            def run(self, x):
                # 如果对象的 __dict__ 中包含键 "a"
                if "a" in self.__dict__:
                    x = x * self.a
                # 如果对象的 __dict__ 中包含键 "b"
                if "b" in self.__dict__:
                    x = x * self.b
                # 设置对象的属性 c 为 7
                self.c = 7
                # 如果对象的 __dict__ 中包含键 "c"
                if "c" in self.__dict__:
                    x = x * self.c
                # 返回 x 乘以属性 "a" 的值乘以属性 "z" 的值（若不存在则默认为 2）
                return x * self.__dict__.get("a") * self.__dict__.get("z", 2)

        # 创建 UserDefined 类的对象 obj
        obj = UserDefined()

        # 定义一个函数 fn，接受参数 x，调用 obj 对象的 run 方法
        def fn(x):
            return obj.run(x)

        # 创建一个随机张量 x
        x = torch.randn(4)
        # 计算未优化的 fn 函数的参考结果 ref
        ref = fn(x)
        # 使用 torch.compile 对 fn 进行编译，使用 "eager" 后端和全图模式
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        # 使用优化后的 opt_fn 处理 x，得到结果 res
        res = opt_fn(x)
        # 断言优化前后的结果一致
        self.assertEqual(ref, res)

    def test_assert_size_stride(self):
        # 创建一个形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 使用断言检查 x 的大小和步幅是否符合预期
        with self.assertRaisesRegex(
            AssertionError,
            "expected size 2==5, stride 12==9 at dim=0; expected size 3==6, stride 4==9 at dim=1; expected size 4==7, stride 1==10 at dim=2",
        ):
            # 调用 torch._C._dynamo.guards.assert_size_stride 检查 x 的大小和步幅是否符合给定参数
            torch._C._dynamo.guards.assert_size_stride(x, (5, 6, 7), (9, 9, 10))
    def test_module_dunder_dict(self):
        # 定义一个名为 test_module_dunder_dict 的测试方法
        class MyModule(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的子类 MyModule
            def __init__(self):
                # 初始化方法
                super().__init__()
                # 调用父类的初始化方法
                self.foo = 1
                # 设置实例变量 foo 为 1
                self.bar = 2
                # 设置实例变量 bar 为 2
                self.baz = 3
                # 设置实例变量 baz 为 3

            def forward(self, x):
                # 定义前向传播方法 forward
                if "foo" in self.__dict__:
                    # 如果实例的 __dict__ 属性中包含键 'foo'
                    return x * self.bar
                    # 返回输入 x 乘以实例变量 bar 的结果
                return x * self.baz
                # 如果实例的 __dict__ 属性中不包含键 'foo'，返回输入 x 乘以实例变量 baz 的结果

        mod = MyModule()
        # 创建 MyModule 的一个实例 mod
        x = torch.randn(10)
        # 生成一个包含 10 个随机数的张量 x
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        # 使用 torch.compile 编译模块 mod，使用 eager 后端和完整图形选项
        self.assertEqual(mod(x), opt_mod(x))
        # 断言模块 mod 对输入 x 的输出与优化后的模块 opt_mod 对输入 x 的输出相等
# 定义一个名为 TestTracer 的测试类，继承自 JitTestCase
class TestTracer(JitTestCase):

    # 定义一个名为 test_jit_save 的测试方法
    def test_jit_save(self):

        # 定义一个名为 fn 的内部函数
        def fn():
            # 定义一个名为 Foo 的内部类，继承自 torch.nn.Module
            class Foo(torch.nn.Module):
                # 构造方法，初始化父类并设置实例变量 self.a 为 3
                def __init__(self):
                    super().__init__()
                    self.a = 3

                # 标记为 Torch 脚本导出方法，用于获取对象的状态
                @torch.jit.export
                def __getstate__(self):
                    return (3, self.training)

                # 标记为 Torch 脚本导出方法，用于设置对象的状态
                @torch.jit.export
                def __setstate__(self, state):
                    self.a = state[0]
                    self.training = state[1]

                # 前向传播方法，对输入 x 加上实例变量 self.a 并返回结果
                def forward(self, x):
                    return x + self.a

            # 创建 Foo 类的实例 f
            f = Foo()

            # 对 f 进行 Torch JIT 跟踪，传入一个形状为 (3, 4) 的随机张量作为输入
            return torch.jit.trace(f, (torch.rand(3, 4),))

        # 调用 fn 函数
        fn()

        # 使用 Torch 的私有模块 _dynamo 中的 optimize 函数进行优化，使用 "eager" 模式
        opt_fn = torch._dynamo.optimize("eager")(fn)

        # 调用经过优化后的函数 opt_fn
        opt_fn()


if __name__ == "__main__":
    # 如果该脚本作为主程序运行，则从 torch._dynamo.test_case 模块导入 run_tests 函数并执行
    from torch._dynamo.test_case import run_tests

    run_tests()
```
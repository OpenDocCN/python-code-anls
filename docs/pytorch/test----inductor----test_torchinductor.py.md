# `.\pytorch\test\inductor\test_torchinductor.py`

```py
# Owner(s): ["module: inductor"]

# 导入必要的标准库和第三方库
import contextlib                 # 提供对上下文管理的支持
import copy                       # 提供对象的浅复制和深复制操作
import dataclasses                # 提供用于创建不可变对象的装饰器和函数
import functools                  # 提供用于函数装饰器和高阶函数的工具
import gc                         # 提供垃圾回收机制的接口
import importlib                  # 提供导入和导出模块的函数
import itertools                  # 提供用于操作迭代器的函数
import math                       # 提供数学函数
import operator                   # 提供Python中的标准操作符函数
import os                         # 提供与操作系统交互的功能
import random                     # 提供生成随机数的功能
import re                         # 提供正则表达式操作
import subprocess                 # 提供执行外部命令的功能
import sys                        # 提供对Python解释器的访问
import threading                  # 提供多线程支持
import time                       # 提供时间操作的功能
import typing                     # 提供类型提示支持
import unittest                   # 提供单元测试框架
import unittest.mock              # 提供用于模拟对象的框架
import weakref                    # 提供弱引用对象的支持
from pathlib import Path          # 提供处理路径的功能
from typing import Tuple          # 提供类型提示支持
from unittest.mock import patch  # 提供对单元测试中的模拟打补丁的支持

import numpy as np               # 导入NumPy库，用于数值计算

import torch                     # 导入PyTorch深度学习库

import torch._dynamo.config as dynamo_config  # 导入Dynamo配置模块
import torch.nn as nn            # 导入PyTorch的神经网络模块
from torch._dispatch.python import enable_python_dispatcher  # 导入Python调度器支持
from torch._dynamo.debug_utils import aot_graph_input_parser  # 导入用于分析AOT图输入的工具函数
from torch._dynamo.testing import (  # 导入用于动态测试的各种实用工具和函数
    CompileCounterWithBackend,
    expectedFailureCodegenDynamic,
    rand_strided,
    same,
    skipIfPy312,
)
from torch._dynamo.utils import ifdynstaticdefault  # 导入用于动态静态默认处理的工具函数
from torch._inductor.codegen.common import DataTypePropagation, OptimizationContext  # 导入编码器公共类和优化上下文
from torch._inductor.fx_passes import pad_mm  # 导入Pad MM函数
from torch._inductor.test_case import TestCase as InductorTestCase  # 导入Inductor测试用例
from torch._inductor.utils import (  # 导入Inductor编码器的各种实用工具函数
    add_scheduler_init_hook,
    aoti_compile_with_persistent_cache,
    aoti_eager_cache_dir,
    load_aoti_eager_cache,
    run_and_get_code,
    run_and_get_cpp_code,
    run_and_get_triton_code,
)
from torch._inductor.virtualized import V  # 导入虚拟化V
from torch._prims_common import is_integer_dtype  # 导入判断整数数据类型的函数
from torch.fx.experimental.proxy_tensor import make_fx  # 导入制作FX代理张量的工具函数
from torch.library import _scoped_library  # 导入C++库
from torch.nn import functional as F  # 导入PyTorch的函数式API
from torch.testing import FileCheck, make_tensor  # 导入用于测试的文件检查和生成张量的工具
from torch.testing._internal.common_cuda import (  # 导入用于测试CUDA的常规功能
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
    SM80OrLater,
    TEST_CUDNN,
    with_tf32_off,
)
from torch.testing._internal.common_device_type import (  # 导入用于测试设备类型的工具函数
    _has_sufficient_memory,
    expectedFailureXPU,
)
from torch.testing._internal.common_dtype import all_types, get_all_dtypes  # 导入所有数据类型和获取所有数据类型的函数
from torch.testing._internal.common_utils import (  # 导入常用实用工具函数
    DeterministicGuard,
    instantiate_parametrized_tests,
    IS_CI,
    IS_FBCODE,
    IS_MACOS,
    IS_WINDOWS,
    IS_X86,
    parametrize,
    serialTest,
    skipIfNNModuleInlined,
    skipIfRocm,
    skipIfXpu,
    subtest,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
)
from torch.utils import _pytree as pytree  # 导入_pytree工具
from torch.utils._python_dispatch import TorchDispatchMode  # 导入Torch分发模式
from torch.utils._pytree import tree_flatten, tree_unflatten  # 导入树展平和树展开函数
from torch.utils.weak import WeakTensorKeyDictionary  # 导入弱引用张量字典

# 从环境变量中检查是否设置了性能测试标志
DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"

# 如果是在Windows且是在CI环境下
if IS_WINDOWS and IS_CI:
    # 输出错误信息，说明Windows上的CI环境缺少执行test_torchinductor所需的依赖项
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    # 如果脚本是直接运行的，退出程序
    if __name__ == "__main__":
        sys.exit(0)
    # 否则抛出unittest跳过测试的异常
    raise unittest.SkipTest("requires sympy/functorch/filelock")

# 动态导入functorch模块和filelock模块
importlib.import_module("functorch")
importlib.import_module("filelock")

# 导入torch._inductor模块中的配置和测试操作
from torch._inductor import config, test_operators

# 导入torch._inductor.compile_fx模块中的相关函数和类
from torch._inductor.compile_fx import (
    compile_fx,
    compile_fx_inner,
    complex_memory_overlap,
)
# 导入来自torch._inductor.utils模块的has_torchvision_roi_align函数
from torch._inductor.utils import has_torchvision_roi_align

# 导入来自torch.testing._internal.common_utils模块的slowTest函数
from torch.testing._internal.common_utils import slowTest
# 导入来自torch.testing._internal.inductor_utils模块的一系列变量和函数
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    HAS_MULTIGPU,
    requires_gpu,
    skipCPUIf,
    skipCUDAIf,
)

# 检查torch.backends.quantized.supported_engines中是否包含"fbgemm"，并将结果存储在HAS_AVX2变量中
HAS_AVX2 = "fbgemm" in torch.backends.quantized.supported_engines

# 获取torch.ops.aten命名空间
aten = torch.ops.aten

# 创建一个偏函数requires_multigpu，用于unittest.skipIf条件，如果没有多个GPU设备则跳过测试
requires_multigpu = functools.partial(
    unittest.skipIf, not HAS_MULTIGPU, f"requires multiple {GPU_TYPE} devices"
)

# 创建一个偏函数skip_if_x86_mac，用于unittest.skipIf条件，如果运行环境为x86 Mac则跳过测试
skip_if_x86_mac = functools.partial(
    unittest.skipIf, IS_MACOS and IS_X86, "Does not work on x86 Mac"
)

# 定义一个包含torch数据类型的列表vec_dtypes
vec_dtypes = [torch.float, torch.bfloat16, torch.float16]

# 创建一个名为libtest的torch.library.Library对象，用于测试目的，带有不规范的编码警告
libtest = torch.library.Library("test", "FRAGMENT")  # noqa: TOR901

# 创建一个空集合ids，用于存储唯一的标识符
ids = set()

# 定义常量f32，i64和i32，分别表示torch.float32，torch.int64和torch.int32
f32 = torch.float32
i64 = torch.int64
i32 = torch.int32


def _large_cumprod_input(shape, dim, dtype, device):
    # 构建一个cumprod输入，确保不会溢出或下溢
    if is_integer_dtype(dtype):
        # 大的乘积不适合整数类型，最好的方法是使用随机的+/-1值来测试结果的符号
        x = torch.randint(0, 1, shape, dtype=dtype, device=device)
        return x * 2 - 1

    # 获取计算使用的数据类型
    comp_dtype = torch._prims_common.get_computation_dtype(dtype)

    # 默认批次大小为256
    batch_size = 256
    if comp_dtype != dtype:
        # 如果计算数据类型不等于指定dtype，根据dtype的最大值计算批次大小
        batch_size = math.floor(math.log2(torch.finfo(dtype).max) / 3)

    # 创建随机值，具有均匀的幅度和均匀的指数
    num_batches = (shape[dim] + 2 * batch_size - 1) // (2 * batch_size)
    batch_shape = (
        shape[:dim]
        + (
            num_batches,
            batch_size,
        )
        + shape[dim + 1 :]
    )
    magnitude = 1 + torch.rand(batch_shape, dtype=comp_dtype, device=device)
    exponent = torch.randint(-1, 1, batch_shape, device=device).to(comp_dtype)
    batch = magnitude * exponent.exp2()

    # 将每个批次的值和它们的倒数交替，确保乘积不会离1太远
    t = torch.cat((batch, batch.reciprocal()), dim=dim + 1)
    t = t.flatten(dim, dim + 1)
    t = aten.slice(t, dim=dim, start=0, end=shape[dim])

    # 随机化符号
    sign = torch.randint(0, 1, shape, device=device) * 2 - 1
    return (t * sign).to(dtype)


def define_custom_op_for_test(id_, fn_cpu, fn_cuda, fn_xpu, fn_meta, tags=()):
    # 声明全局变量libtest和ids
    global libtest
    global ids
    # 如果id_不在ids集合中，则定义自定义操作并将其添加到libtest中
    if id_ not in ids:
        libtest.define(f"{id_}(Tensor self) -> Tensor", tags=tags)
        libtest.impl(id_, fn_cpu, "CPU")
        libtest.impl(id_, fn_cuda, "CUDA")
        libtest.impl(id_, fn_xpu, "XPU")
        libtest.impl(id_, fn_meta, "Meta")
        ids.add(id_)


def define_custom_op_2_for_test(id_, fn_cpu, fn_cuda, fn_xpu, fn_meta, tags=()):
    # 声明全局变量libtest和ids
    global libtest
    global ids
    # 检查 id_ 是否不在 ids 集合中
    if id_ not in ids:
        # 在 libtest 模块中定义一个新的函数签名，接受 Tensor 类型的 self 参数和 float 类型的 scale 参数，返回两个 Tensor 类型的结果，同时指定标签为 tags
        libtest.define(
            f"{id_}(Tensor self, float scale) -> (Tensor, Tensor)", tags=tags
        )
        # 使用 fn_cpu 函数实现 id_ 函数在 CPU 上的执行
        libtest.impl(id_, fn_cpu, "CPU")
        # 使用 fn_cuda 函数实现 id_ 函数在 CUDA 上的执行
        libtest.impl(id_, fn_cuda, "CUDA")
        # 使用 fn_xpu 函数实现 id_ 函数在 XPU 上的执行
        libtest.impl(id_, fn_xpu, "XPU")
        # 使用 fn_meta 函数实现 id_ 函数在 Meta 上的执行
        libtest.impl(id_, fn_meta, "Meta")
        # 将 id_ 添加到 ids 集合中，表示已经处理过该 id_
        ids.add(id_)
# 定义一个自定义操作函数，用于测试
def define_custom_op_3_for_test(id_, fn_cpu, fn_cuda, fn_xpu, fn_meta, tags=()):
    # 声明全局变量，用于存储测试库
    global libtest
    # 声明全局变量，用于存储已定义的操作函数的集合
    global ids
    # 如果当前操作函数 ID 不在已定义的 ID 集合中，则进行定义和注册
    if id_ not in ids:
        # 定义一个新的操作函数签名，并附加标签
        libtest.define(f"{id_}(Tensor[] x) -> Tensor", tags=tags)
        # 在 CPU 上实现该操作函数
        libtest.impl(id_, fn_cpu, "CPU")
        # 在 CUDA 上实现该操作函数
        libtest.impl(id_, fn_cuda, "CUDA")
        # 在 XPU 上实现该操作函数
        libtest.impl(id_, fn_xpu, "XPU")
        # 在 Meta 上实现该操作函数
        libtest.impl(id_, fn_meta, "Meta")
        # 将当前操作函数 ID 加入已定义的 ID 集合中
        ids.add(id_)


# 设置默认的浮点数类型为 torch.float32
f32 = torch.float32


# 定义一个函数，运行给定函数并返回其代码
def run_fw_bw_and_get_code(fn):
    # 定义一个新函数，运行给定函数并执行反向传播
    def run_with_backward():
        # 调用给定函数并获取结果
        result = fn()
        # 计算结果的和并执行反向传播
        result.sum().backward()
        # 返回结果
        return result

    # 返回内部定义的新函数
    return run_and_get_code(run_with_backward)


# 注册操作函数，并使用 AOTI 编译
def register_ops_with_aoti_compile(ns, op_set, dispatch_key, torch_compile_op_lib_impl):
    # 遍历操作函数集合
    for _op_name in op_set:
        # 构建完全限定的操作函数名称
        qualified_op_name = f"{ns}::{_op_name}"
        # 获取该操作函数的所有重载名称
        _, overload_names = torch._C._jit_get_operation(qualified_op_name)
        # 遍历每个重载名称
        for overload_name in overload_names:
            try:
                # 设置注册的操作名称为完全限定的操作函数名称
                reg_op_name = qualified_op_name
                # 获取操作函数的 schema 信息
                schema = torch._C._get_schema(qualified_op_name, overload_name)
                # 如果 schema 中包含重载名称，则使用完全限定的操作名称和重载名称
                if schema.overload_name:
                    reg_op_name = f"{qualified_op_name}.{schema.overload_name}"
                # 使用 AOTI 编译注册操作函数
                torch_compile_op_lib_impl._impl_with_aoti_compile(  # noqa: F821
                    reg_op_name, dispatch_key
                )
            except Exception as e:
                # 发生异常时继续下一个操作函数的注册
                continue


# 定义一个测试用例类，继承自 InductorTestCase
class TestCase(InductorTestCase):
    # 在测试类初始化时执行的方法
    @classmethod
    def setUpClass(cls):
        # 调用父类的初始化方法
        super().setUpClass()
        # 创建一个上下文管理器栈对象
        cls._stack = contextlib.ExitStack()
        # 进入上下文并配置一些测试参数
        cls._stack.enter_context(
            config.patch(
                {
                    "debug": True,
                    "debug_index_asserts": True,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,  # too slow
                    "implicit_fallbacks": False,
                    "generate_intermediate_hooks": True,
                }
            )
        )

    # 在测试类销毁时执行的方法
    @classmethod
    def tearDownClass(cls):
        # 关闭上下文管理器栈
        cls._stack.close()
        # 调用父类的销毁方法
        super().tearDownClass()

    # 在每个测试方法执行前执行的方法
    def setUp(self):
        # 重置 Dynamo 模块的状态
        torch._dynamo.reset()
        # 重置 Inductor 模块的指标数据
        torch._inductor.metrics.reset()
        # 调用父类的 setUp 方法
        super().setUp()
        # 记录测试开始时间
        self._start = time.perf_counter()

    # 在每个测试方法执行后执行的方法
    def tearDown(self):
        # 调用父类的 tearDown 方法
        super().tearDown()
        # 重置 Dynamo 模块的状态
        torch._dynamo.reset()
        # 如果环境变量中设置了 ERROR_ON_SLOW 为 "1"，则检查测试运行时间是否超过阈值
        if os.environ.get("ERROR_ON_SLOW") == "1":
            elapsed = time.perf_counter() - self._start
            assert elapsed < 120  # 断言测试运行时间小于 120 秒


# 定义一个简单的 torch.nn.Module 子类，实现将输入 x 转换为元组的功能
class ToTuple(torch.nn.Module):
    # 定义前向传播方法
    def forward(self, x):
        # 返回输入 x 的元组形式
        return (x,)


# 定义一个数据类，用于生成输入数据
@dataclasses.dataclass
class InputGen:
    # 定义属性 n，表示数据的维度
    n: int
    # 定义属性 device，表示数据存储的设备类型
    device: str

    # 生成一个随机的密集矩阵
    def dense(self):
        return torch.randn((self.n, self.n), device=self.device)

    # 生成一个转置后的密集矩阵
    def transposed(self):
        return self.dense().transpose(0, 1)

    # 生成一个具有步长的稀疏矩阵
    def strided(self):
        return torch.randn((self.n * 2, self.n * 3), device=self.device)[
            self.n :, self.n :: 2
        ]

    # 生成一个广播后的向量
    def broadcast1(self):
        return torch.randn((self.n,), device=self.device)
    # 返回一个形状为 (1, self.n, 1) 的张量，元素为随机数，设备为 self.device
    def broadcast2(self):
        return torch.randn((1, self.n, 1), device=self.device)
    
    # 返回一个形状为 (1,) 的张量，元素为随机数，设备为 self.device
    def broadcast3(self):
        return torch.randn((1,), device=self.device)
    
    # 返回一个形状为 (self.n, self.n) 的张量，元素为随机数，设备为 self.device，数据类型为 torch.double
    def double(self):
        return torch.randn((self.n, self.n), device=self.device, dtype=torch.double)
    
    # 返回一个形状为 (self.n,) 的整型张量，元素为从 0 到 self.n-1 的整数，设备为 self.device，数据类型为 torch.int32
    def int(self):
        return torch.arange(self.n, device=self.device, dtype=torch.int32)
# 计算梯度
def compute_grads(args, kwrags, results, grads):
    # 从参数中提取叶子张量
    def gather_leaf_tensors(args, kwargs):
        args = pytree.arg_tree_leaves(*args, **kwargs)
        leaf_tensors = [
            arg for arg in args if isinstance(arg, torch.Tensor) and arg.requires_grad
        ]
        return leaf_tensors

    # 将结果展平
    flat_results = pytree.tree_leaves(results)
    # 提取需要计算梯度的张量
    flat_diff_results = [
        r for r in flat_results if isinstance(r, torch.Tensor) and r.requires_grad
    ]
    assert len(flat_diff_results) > 0

    # 获取叶子张量
    leaf_tensors = gather_leaf_tensors(args, kwrags)
    assert len(leaf_tensors) > 0
    # 计算梯度
    return torch.autograd.grad(
        flat_diff_results,
        leaf_tensors,
        grads,
        allow_unused=True,
        retain_graph=True,
    )


# 克隆张量并保留步幅
def clone_preserve_strides(x, device=None):
    if not isinstance(x, torch.Tensor):
        return x
    buffer = torch.as_strided(
        x, (x.untyped_storage().size() // x.element_size(),), (1,), 0
    )
    if not device:
        buffer = buffer.clone()
    else:
        buffer = buffer.to(device, copy=True)
    out = torch.as_strided(buffer, x.size(), x.stride(), x.storage_offset())
    return out


# 检查模型
def check_model(
    self: TestCase,
    model,
    example_inputs,
    kwargs=None,
    *,
    atol=None,
    rtol=None,
    grad_atol=None,
    grad_rtol=None,
    check_lowp=True,
    exact_dtype=True,
    nopython=True,
    copy_to_gpu=True,
    reference_in_float=True,
    assert_equal=True,
    check_gradient=False,
    check_has_compiled=True,
    output_process_fn_grad=lambda x: x,
):
    kwargs = kwargs or {}
    torch._dynamo.reset()

    ref_inputs = [clone_preserve_strides(x) for x in example_inputs]
    ref_kwargs = kwargs
    has_lowp_args = False

    if reference_in_float and exact_dtype:
        # 存储预期的数据类型，以便检查实际结果是否给出了正确的类型
        torch.manual_seed(0)
        try:
            eager_result = model(*ref_inputs, **ref_kwargs)
        except RuntimeError:
            # 如果数据类型不受支持，Eager 模型可能会失败
            eager_result = None

        ref_inputs = [clone_preserve_strides(x) for x in example_inputs]
        expect_dtypes = [
            x.dtype if isinstance(x, torch.Tensor) else None
            for x in pytree.tree_leaves(eager_result)
        ]
        del eager_result

    ref_model = model
    # 如果 reference_in_float 为 True，则执行以下操作
    if reference_in_float:
        # 定义一个函数 upcast_fn，用于将输入张量类型转换为 float 类型，同时标记是否有低精度参数
        # 这里 check_lowp 被忽略，只是为了能够调用带有额外参数的 `common`
        def upcast_fn(x):
            nonlocal has_lowp_args
            # 如果 x 是 torch.Tensor，并且数据类型是 torch.float16 或 torch.bfloat16
            if isinstance(x, torch.Tensor) and (
                x.dtype == torch.float16 or x.dtype == torch.bfloat16
            ):
                has_lowp_args = True
                # 将 x 转换为 float 类型并返回
                return x.float()
            else:
                # 否则直接返回 x
                return x

        # 对示例输入列表 example_inputs 应用 upcast_fn 函数，生成 ref_inputs 列表
        ref_inputs = list(map(upcast_fn, example_inputs))
        # 对 kwargs 中的每个键值对应用 upcast_fn 函数，生成 ref_kwargs 字典
        ref_kwargs = {k: upcast_fn(v) for k, v in kwargs.items()}
        
        # 如果存在低精度参数并且模型 model 具有 "to" 方法，则深度复制模型并转换为 torch.float 类型
        if has_lowp_args and hasattr(model, "to"):
            ref_model = copy.deepcopy(model).to(torch.float)

    # 设置随机种子为 0
    torch.manual_seed(0)

    # 使用 ref_model 执行 ref_inputs 和 ref_kwargs，得到正确的输出 correct
    correct = ref_model(*ref_inputs, **ref_kwargs)

    # 重置 pytorch 的一些度量指标
    torch._inductor.metrics.reset()

    # 初始化 called 标志为 False
    called = False

    # 定义一个函数 compile_fx_wrapper，用于调用 compile_fx 并标记已调用
    def compile_fx_wrapper(model_, example_inputs_):
        nonlocal called
        called = True
        return compile_fx(model_, example_inputs_)

    # 定义一个函数 run，用于执行模型 model
    def run(*ex, **kwargs):
        return model(*ex, **kwargs)

    # 使用 torch._dynamo.optimize 对 compile_fx_wrapper 进行优化处理，并将结果重新赋给 run
    run = torch._dynamo.optimize(compile_fx_wrapper, nopython=nopython)(run)

    # 重新设置随机种子为 0
    torch.manual_seed(0)

    # 使用 example_inputs 和 kwargs 执行优化后的 run 函数，得到实际输出 actual
    actual = run(*example_inputs, **kwargs)

    # 如果 check_has_compiled 为 True，则断言 called 必须为 True，否则抛出异常
    if check_has_compiled:
        assert called, "Ran graph without calling compile_fx"

    # 断言 actual 的类型必须与 correct 的类型相同
    assert type(actual) == type(correct)

    # 如果 actual 是 tuple 或 list 类型，则断言其长度与 correct 相同，并且各元素的类型也相同
    if isinstance(actual, (tuple, list)):
        assert len(actual) == len(correct)
        assert all(
            type(actual_item) == type(correct_item)
            for actual_item, correct_item in zip(actual, correct)
        )

    # 使用 tree_flatten 将 correct 展平为 flat 和 spec 两部分
    correct_flat, correct_spec = tree_flatten(correct)
    # 使用 pytree.tree_leaves 将 actual 展平为 flat
    actual_flat = pytree.tree_leaves(actual)

    # 定义一个函数 reference_to_expect，用于将 actual_flat 转换为与 correct_flat 类型匹配的数据
    def reference_to_expect(actual_flat, correct_flat):
        return tuple(
            (
                y.to(x.dtype)
                if isinstance(y, torch.Tensor) and y.dtype.is_floating_point
                else y
            )
            for x, y in zip(actual_flat, correct_flat)
        )

    # 如果 reference_in_float 为 True 并且 exact_dtype 为 True，则执行以下操作
    if reference_in_float and exact_dtype:
        # 对每个 expect_dtypes 和 actual_flat 中的结果进行断言，确保其数据类型匹配
        for expect_dtype, actual_result in zip(expect_dtypes, actual_flat):
            if expect_dtype is not None:
                assert (
                    actual_result.dtype == expect_dtype
                ), f"dtype mismatch, expected {expect_dtype} but got {actual_result.dtype}"

    # 如果 reference_in_float 为 True，则使用 reference_to_expect 函数将 correct_flat 转换为期望的数据形式
    if reference_in_float:
        correct_flat = reference_to_expect(actual_flat, correct_flat)
        # 使用 tree_unflatten 将 correct_flat 和 correct_spec 重新组合成 correct
        correct = tree_unflatten(correct_flat, correct_spec)
    # 如果 assert_equal 为真，则进行断言检查
    if assert_equal:
        # 使用 self.assertEqual 方法比较 actual 和 correct 的值
        self.assertEqual(
            actual,
            correct,
            atol=atol,               # 允许的绝对误差
            rtol=rtol,               # 允许的相对误差
            equal_nan=True,          # 是否检查 NaN 值相等
            exact_dtype=exact_dtype, # 是否要求精确的数据类型匹配
        )
        # 对于可能有输入变异的情况，检查输入是否相同
        self.assertEqual(
            ref_inputs,
            example_inputs,
            atol=atol,               # 允许的绝对误差
            rtol=rtol,               # 允许的相对误差
            equal_nan=True,          # 是否检查 NaN 值相等
            # 我们的测试有时会使用高精度输入作为参考
            exact_dtype=False,       # 不要求精确的数据类型匹配
        )
    else:
        # 如果 assert_equal 为假，则遍历 correct_flat 和 actual_flat 的每对值
        for correct_val, actual_val in zip(correct_flat, actual_flat):
            # 如果 correct_val 是 torch.Tensor 类型
            if isinstance(correct_val, torch.Tensor):
                # 断言 correct_val 和 actual_val 的设备相同
                assert correct_val.device == actual_val.device
                # 断言 correct_val 和 actual_val 的大小相同
                assert correct_val.size() == actual_val.size()
                # 检查 significant strides 是否相等
                strides_equal, _ = torch._prims_common.check_significant_strides(
                    correct_val, actual_val
                )
                assert strides_equal  # 断言 significant strides 相等
                # 断言 correct_val 和 actual_val 的布局相同
                assert correct_val.layout == actual_val.layout
                # 如果需要精确的数据类型匹配
                if exact_dtype:
                    # 断言 correct_val 和 actual_val 的数据类型相同
                    assert correct_val.dtype == actual_val.dtype
    # 如果需要检查梯度
    if check_gradient:
        # 对实际输出和正确输出进行梯度处理函数
        actual = output_process_fn_grad(actual)
        correct = output_process_fn_grad(correct)
        # 将实际输出和正确输出展平为一维列表
        actual_flat = pytree.tree_leaves(actual)
        correct_flat = pytree.tree_leaves(correct)

        # 生成随机单位范数的梯度
        grads = [
            torch.rand(r.shape, device=r.device, dtype=r.dtype)
            for r in correct_flat
            if isinstance(r, torch.Tensor) and r.requires_grad
        ]
        # 归一化每个梯度向量
        for g in grads:
            g /= g.norm()

        # 计算正确输出的梯度
        correct_grad = compute_grads(ref_inputs, ref_kwargs, correct, grads)
        # 检查所有梯度是否都为 None
        all_none_grads = all(x is None for x in correct_grad)
        if all_none_grads:
            # 如果所有梯度都是 None，说明存在一些操作返回了 None 梯度，而不是零梯度。
            # 如果所有输入都应该得到 None 梯度，则 AOTAutograd 将强制所有输出的 forward 不需要梯度。
            # 对于这个问题，目前没有简单的修复方法（见上述注释），尽管一种选择是在核心中强制所有导数公式返回零张量而不是 None。
            # 获取实际输出的所有叶子节点
            flat_results = pytree.tree_leaves(actual)
            # 获取那些需要梯度的结果
            results_that_require_grad = [
                x
                for x in flat_results
                if isinstance(x, torch.Tensor) and x.requires_grad
            ]
            # 断言没有需要梯度的结果
            self.assertEqual(len(results_that_require_grad), 0)
        else:
            # 计算实际输出的梯度
            actual_grad = compute_grads(example_inputs, kwargs, actual, grads)

            # 根据 reference_in_float 决定使用哪个梯度作为期望梯度
            if reference_in_float:
                expect_grad = reference_to_expect(actual_grad, correct_grad)
            else:
                expect_grad = correct_grad

            # 断言实际梯度等于期望梯度
            self.assertEqual(
                actual_grad,
                expect_grad,
                atol=grad_atol or atol,
                rtol=grad_rtol or rtol,
                equal_nan=True,
                exact_dtype=exact_dtype,
            )

    # 重置 torch._dynamo
    torch._dynamo.reset()
# 使用装饰器将函数标记为在 torch._inductor.config.patch 中配置 triton.cudagraphs 为 False 时使用的函数
@torch._inductor.config.patch("triton.cudagraphs", False)
# 定义一个函数 check_model_gpu，接受多个参数，包括 model、example_inputs 和 kwargs
def check_model_gpu(
    self: TestCase,
    model,
    example_inputs,
    kwargs=None,
    *,
    atol=None,
    rtol=None,
    grad_atol=None,
    grad_rtol=None,
    check_lowp=True,
    exact_dtype=True,
    nopython=True,
    copy_to_gpu=True,
    reference_in_float=True,
    assert_equal=True,
    check_gradient=False,
    check_has_compiled=True,
    output_process_fn_grad=lambda x: x,
):
    # 将默认的 kwargs 设置为一个空字典，如果未提供的话
    kwargs = kwargs or {}
    # 如果 model 具有 to 方法，则将 model 移动到 GPU_TYPE 指定的设备上
    if hasattr(model, "to"):
        model = model.to(device=GPU_TYPE)

    # 如果 copy_to_gpu 为 True，则将 example_inputs 中的每个输入克隆并保留原始步长，并移动到 GPU_TYPE 指定的设备上
    if copy_to_gpu:
        example_inputs = tuple(
            clone_preserve_strides(x, device=GPU_TYPE) for x in example_inputs
        )

    # 调用 check_model 函数，检查模型在 GPU 上的运行情况
    check_model(
        self,
        model,
        example_inputs,
        kwargs,
        atol=atol,
        rtol=rtol,
        grad_atol=grad_atol,
        grad_rtol=grad_rtol,
        exact_dtype=exact_dtype,
        nopython=nopython,
        reference_in_float=reference_in_float,
        assert_equal=assert_equal,
        check_gradient=check_gradient,
        check_has_compiled=check_has_compiled,
        output_process_fn_grad=output_process_fn_grad,
    )

    # 如果 check_lowp 为 True，则进行低精度（half precision）检查
    if check_lowp:

        # 定义一个函数 downcast_fn，用于将输入张量 x 的数据类型降为 torch.half
        def downcast_fn(x):
            if not isinstance(x, torch.Tensor) or not x.dtype == torch.float:
                return x
            return torch.empty_strided(
                x.size(), x.stride(), device=GPU_TYPE, dtype=torch.half
            ).copy_(x)

        # 对 example_inputs 中的每个输入应用 downcast_fn 函数，实现数据类型降级
        example_inputs = list(map(downcast_fn, example_inputs))
        # 如果 model 具有 to 方法，则将 model 的数据类型设置为 torch.half
        if hasattr(model, "to"):
            model = model.to(torch.half)
        # 如果 rtol 不为 None，则将其设置为 max(2e-3, rtol)
        if rtol is not None:
            rtol = max(2e-3, rtol)
        # 再次调用 check_model 函数，以检查低精度条件下的模型运行情况
        check_model(
            self,
            model,
            example_inputs,
            kwargs,
            atol=atol,
            rtol=rtol,
            grad_atol=grad_atol,
            grad_rtol=grad_rtol,
            exact_dtype=exact_dtype,
            nopython=nopython,
            reference_in_float=reference_in_float,
            assert_equal=assert_equal,
            check_gradient=check_gradient,
            check_has_compiled=check_has_compiled,
            output_process_fn_grad=output_process_fn_grad,
        )


# 将 check_model_gpu 函数赋值给 check_model_cuda，作为其别名
check_model_cuda = check_model_gpu


# 定义一个私有函数 _run_and_assert_no_indirect_indexing，用于运行函数并断言是否存在间接索引
def _run_and_assert_no_indirect_indexing(
    test_case, func, *args, has_wrapping=None, has_assert=False, **kwargs
):
    # 调用 run_and_get_code 函数执行 func，并获取其代码运行结果和源码
    result, source_codes = run_and_get_code(func, *args, **kwargs)
    for code in source_codes:
        # 对于每个源代码片段中的每一行
        for line in code.split("\n"):
            stmt = None
            # 查找索引表达式
            if ".load(" in line:
                stmt = line.split(".load")[-1]  # 提取加载操作的目标
            elif "tl.store" in line:
                stmt = line.split(".store")[-1]  # 提取存储操作的目标
                stmt = ",".join(stmt.split(",")[:-2])  # 移除存储值和掩码
            elif ".store" in line:
                stmt = line.split(".store")[-1]  # 提取存储操作的目标
            elif "[" in line:
                stmt = line.split("[")[-1].split("]")[0]  # 提取索引表达式的内容
            if "tl.make_block_ptr(" in line:
                continue  # 跳过包含特定函数调用的行

            if stmt is None:
                continue  # 如果没有找到任何目标操作，继续下一行

            # 检查是否包含间接索引，这种情况下应该不存在 `tmp` 变量
            test_case.assertTrue(
                "tmp" not in stmt,
                msg=f"Found indirect indexing in statement '{stmt}' from code:\n{code}",
            )
        # 如果有包装测试条件，则检查是否符合预期
        if has_wrapping is not None:
            test_case.assertTrue(
                ("where" in code or "?" in code) is has_wrapping,
                msg=f"Wanted {has_wrapping=} but got\n{code}",
            )
    # 检查是否至少有一个源代码片段中包含断言相关的关键字
    test_case.assertTrue(
        any(
            ("device_assert" in code or "TORCH_CHECK" in code) is has_assert
            for code in source_codes
        )
    )
    return result
# 确保生成的内核数目与预期相等的断言函数
def assertGeneratedKernelCountEqual(self: TestCase, expected: int):
    # 当 multi_kernel 被启用时，为同一节点调度生成持久规约和非持久规约内核。
    # 这会混淆内核计数。因此不进行检查。
    if config.triton.multi_kernel:
        return
    # 如果配置了 cpp_wrapper，则预期值乘以 2
    if config.cpp_wrapper:
        expected *= 2
    # 断言生成的内核数目与预期值相等
    self.assertEqual(torch._inductor.metrics.generated_kernel_count, expected)


class SweepInputs2:
    # 第一组输入生成类型
    input_gen_types1 = [
        "dense",
        "transposed",
        "strided",
        "broadcast1",
        "broadcast2",
        "broadcast3",
        "double",
        "int",
    ]
    # 第二组输入生成类型与第一组相同
    input_gen_types2 = input_gen_types1
    # gen 默认为 None

    # 静态方法：定义一个简单的内核函数，返回元组 (a + b,)
    @staticmethod
    def kernel(a, b):
        return (a + b,)

    # 类方法：生成一个测试模板函数，用于检查模型
    @classmethod
    def gen_template(cls, name1, name2):
        def test(self):
            check_model(
                self,
                cls.kernel,
                (
                    getattr(cls.gen, name1)(),  # 获取 cls.gen 中 name1 对应的方法调用结果
                    getattr(cls.gen, name2)(),  # 获取 cls.gen 中 name2 对应的方法调用结果
                ),
            )

        # 动态设置测试函数的名称
        test.__name__ = f"test_{cls.gen.device}_{name1}_{name2}"
        setattr(cls, test.__name__, test)

    # 类方法：填充所有可能的测试用例模板
    @classmethod
    def populate(cls):
        for name1 in cls.input_gen_types1:
            for name2 in cls.input_gen_types2:
                cls.gen_template(name1, name2)


# 函数：判断设备是否使用了 C++ 后端
def is_cpp_backend(device):
    return getattr(device, "type", device) == "cpu" and config.cpu_backend == "cpp"


# 函数：判断设备是否使用了 Halide 后端
def is_halide_backend(device):
    if getattr(device, "type", device) == "cpu":
        return config.cpu_backend == "halide"
    return config.cuda_backend == "halide"


# 装饰器函数：如果是 Halide 后端，则跳过测试
def skip_if_halide(fn):
    @functools.wraps(fn)
    def wrapper(self):
        if is_halide_backend(self.device):
            raise unittest.SkipTest("halide not supported")
        return fn(self)

    return wrapper


# 装饰器函数：如果是 GPU 的 Halide 后端，则跳过测试
def skip_if_gpu_halide(fn):
    @functools.wraps(fn)
    def wrapper(self):
        if (
            is_halide_backend(self.device)
            and getattr(self.device, "type", self.device) == "cuda"
        ):
            raise unittest.SkipTest("halide not supported")
        return fn(self)

    return wrapper


# 类装饰器：实例化参数化测试
@instantiate_parametrized_tests
class CommonTemplate:
    # 测试布尔操作函数
    def test_bool(self):
        # 内部函数：对输入 a 和 b 执行多种逻辑操作
        def fn(a, b):
            return (
                a + b,                          # 加法
                a * b,                          # 乘法
                a & b,                          # 位与
                a | b,                          # 位或
                a ^ b,                          # 位异或
                torch.logical_and(a, b),        # 逻辑与
                torch.logical_or(a, b),         # 逻辑或
                torch.logical_not(a),           # 逻辑非
                torch.sign(b),                  # 符号函数
            )

        # 调用通用测试方法，传入逻辑函数 fn 和输入元组
        self.common(
            fn,
            (
                torch.tensor([True, False, True, False]),     # 第一个张量
                torch.tensor([False, False, True, True]),     # 第二个张量
            ),
        )

    # 装饰器函数：如果不满足条件 SM80OrLater，跳过 CUDA 测试，需要 sm80
    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    # 装饰器函数：如果是 Halide 后端，跳过测试
    @skip_if_halide
    # 定义一个测试方法，用于测试 eager 模式下的 clamp 操作支持
    def test_eager_aoti_support_out(self):
        # 设置命名空间
        ns = "aten"
        # 设置操作名
        op_name = "clamp"
        # 设置默认的调度键为 CPU
        dispatch_key = "CPU"
        # 设置默认的设备为 CPU
        device = "cpu"
        
        # 如果当前实例的设备是 CUDA，则修改调度键为 CUDA，设备为 cuda
        if self.device.lower() == "cuda":
            dispatch_key = "CUDA"
            device = "cuda"

        # 创建一个随机的输入张量，指定数据类型为 float，设备为指定的设备，并填充为 1.0
        inp_tensor = torch.randn(128, dtype=torch.float, device=device).fill_(1.0)
        # 根据输入张量计算最小张量
        min_tensor = inp_tensor - 0.05
        # 根据输入张量计算最大张量
        max_tensor = inp_tensor + 0.05
        
        # 使用 _scoped_library 上下文管理器，编译 aten 命名空间下的 IMPL 库
        with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
            # 创建一个随机的参考输出张量，数据类型为 float，设备为指定的设备，并填充为 -1
            ref_out_tensor = torch.randn(128, dtype=torch.float, device=device).fill_(-1)
            # 执行 torch.clamp 操作，将结果保存在参考张量中
            ref_tensor = torch.clamp(
                max=max_tensor, min=min_tensor, input=inp_tensor, out=ref_out_tensor
            )

            # 创建另一个随机的参考输出张量，数据类型为 float，设备为指定的设备，并填充为 -1
            ref_out_tensor1 = torch.randn(128, dtype=torch.float, device=device).fill_(-1)
            # 执行 torch.clamp 操作，将结果保存在另一个参考张量中
            ref_tensor1 = torch.clamp(
                max=max_tensor, out=ref_out_tensor1, min=min_tensor, input=inp_tensor
            )

            # 使用 register_ops_with_aoti_compile 注册在 AOTI 编译下的操作
            register_ops_with_aoti_compile(
                ns, [op_name], dispatch_key, torch_compile_op_lib_impl
            )

            # 创建一个随机的结果输出张量，数据类型为 float，设备为指定的设备，并填充为 -1
            res_out_tensor = torch.randn(128, dtype=torch.float, device=device).fill_(-1)
            # 执行 torch.clamp 操作，将结果保存在结果张量中
            res_tensor = torch.clamp(
                max=max_tensor, min=min_tensor, input=inp_tensor, out=res_out_tensor
            )

            # 断言参考张量和结果张量的值相等
            self.assertEqual(ref_tensor, res_tensor)
            # 断言参考输出张量和结果输出张量的值相等
            self.assertEqual(ref_out_tensor, res_out_tensor)

            # 创建另一个随机的结果输出张量，数据类型为 float，设备为指定的设备，并填充为 -1
            res_out_tensor1 = torch.randn(128, dtype=torch.float, device=device).fill_(-1)
            # 执行 torch.clamp 操作，将结果保存在另一个结果张量中
            res_tensor1 = torch.clamp(
                max=max_tensor, out=res_out_tensor1, min=min_tensor, input=inp_tensor
            )

            # 断言另一个参考张量和另一个结果张量的值相等
            self.assertEqual(ref_tensor1, res_tensor1)
            # 断言另一个参考输出张量和另一个结果输出张量的值相等
            self.assertEqual(ref_out_tensor1, res_out_tensor1)

    # 使用 skipCUDAIf 装饰器，如果条件不满足（不是 SM80 或更高），则跳过测试，需要 sm80 架构支持
    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    # 使用 skip_if_halide 装饰器，跳过 Halide 相关的测试
    @skip_if_halide  # aoti
    # 定义测试函数：测试 eager 模式下的 aoti 缓存命中情况
    def test_eager_aoti_cache_hit(self):
        # 设定命名空间 ns 和操作名 op_name
        ns = "aten"
        op_name = "abs"
        # 默认分派键为 CPU，设备为 CPU
        dispatch_key = "CPU"
        device = "cpu"
        # 如果当前设备是 CUDA，则更新分派键和设备为 CUDA
        if self.device.lower() == "cuda":
            dispatch_key = "CUDA"
            device = "cuda"

        # 生成随机输入张量，根据设备类型分配存储位置
        input_tensor = torch.randn(128, dtype=torch.float, device=device)
        # 使用 aoti_compile_with_persistent_cache 编译操作的内核库，并获取内核路径
        kernel_lib_path = aoti_compile_with_persistent_cache(
            ns,
            op_name,
            device,
            False,
            getattr(torch.ops.aten, op_name),
            (input_tensor,),
            {},
        )
        # 断言生成的内核库路径存在
        self.assertTrue(Path(kernel_lib_path).exists())

        # 导入 mock 模块，将 aoti_compile_with_persistent_cache 置为 None，确保不生成新的内核
        from unittest import mock

        # 使用 mock.patch 将 aoti_compile_with_persistent_cache 置为 None
        with mock.patch(
            "torch._inductor.utils.aoti_compile_with_persistent_cache", None
        ):
            # 使用 _scoped_library 上下文管理器，编译 aten 的操作库实现
            with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
                # 从 eager 模式获取参考结果
                ref_value = getattr(torch.ops.aten, op_name)(input_tensor)

                # 使用 register_ops_with_aoti_compile 注册编译后的操作到 aoti_compile 中
                register_ops_with_aoti_compile(
                    ns, [op_name], dispatch_key, torch_compile_op_lib_impl
                )

                # 调用预编译的内核并获取结果
                res_value = getattr(torch.ops.aten, op_name)(input_tensor)

                # 断言参考结果和实际结果相等
                self.assertEqual(ref_value, res_value)

    # 如果不支持 SM80 或更高版本的 CUDA，跳过测试
    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    # 如果满足条件则跳过测试，用于 Halide 编译
    @skip_if_halide  # aoti
    # 定义测试函数：测试 eager 模式下的 aoti 缓存持久化
    def test_eager_aoti_with_persistent_cache(self):
        # 定义函数 fn，实现对输入张量的绝对值操作
        def fn(a):
            return torch.abs(a)

        # 设定命名空间 ns 和操作名 op_name
        ns = "aten"
        op_name = "abs"

        # 默认设备为 CPU，如果当前设备是 CUDA，则设备更新为 CUDA
        device = "cpu"
        if self.device.lower() == "cuda":
            device = "cuda"

        # 生成随机输入张量，根据设备类型分配存储位置
        input_tensor = torch.randn(128, dtype=torch.float, device=device)
        # 使用 aoti_compile_with_persistent_cache 编译操作的内核库，并获取内核路径
        kernel_lib_path = aoti_compile_with_persistent_cache(
            ns,
            op_name,
            input_tensor.device.type,
            False,
            fn,
            args=(input_tensor,),
            kwargs={},
        )
        # 断言生成的内核库路径长度大于零
        self.assertTrue(len(kernel_lib_path) > 0)

        # 获取设备上 aoti 缓存目录
        device_kernel_cache = aoti_eager_cache_dir(ns, device)
        # 生成内核配置文件路径
        kernel_conf = device_kernel_cache / f"{op_name}.json"
        # 断言内核配置文件存在
        self.assertTrue(kernel_conf.exists())

        # 加载 aoti 缓存中的 JSON 数据，验证其存在性和类型
        json_data = load_aoti_eager_cache("aten", "abs", input_tensor.device.type)
        self.assertTrue(json_data is not None)
        self.assertTrue(isinstance(json_data, list))
        self.assertTrue(len(json_data) > 0)

        # 获取第一个操作信息，并验证其存在性和类型
        op_info = json_data[0]
        self.assertTrue(isinstance(op_info, dict))
        self.assertTrue("meta_info" in op_info)
        self.assertTrue("kernel_path" in op_info)

        # 获取所有 JSON 数据中的内核路径，并验证预编译的内核路径是否在其中
        kernel_libs_abs_path = []
        for item in json_data:
            kernel_path = device_kernel_cache / item["kernel_path"]
            kernel_libs_abs_path.append(kernel_path.as_posix())

        # 断言预编译的内核路径在生成的内核路径列表中
        self.assertTrue(kernel_lib_path in kernel_libs_abs_path)
    @skip_if_halide  # 如果有名为 "halide" 的条件标记，则跳过测试
    def test_eager_aoti_with_scalar(self):
        # 定义命名空间名称
        namespace_name = "aten"
        # 定义操作名称
        op_name = "add"
        # 定义操作重载名称
        op_overload_name = "Tensor"
        # 构建带有操作和重载名称的完整操作名称
        op_name_with_overload = f"{op_name}.{op_overload_name}"

        # 默认使用 CPU 分发键和设备
        dispatch_key = "CPU"
        device = torch.device("cpu")
        # 如果当前设备是 CUDA，则更新分发键和设备为 CUDA
        if self.device.lower() == "cuda":
            dispatch_key = "CUDA"
            device = torch.device("cuda")

        # 测试标量张量和标量之间的差异
        a = torch.scalar_tensor(1.0, device=device)
        b = torch.scalar_tensor(2.0, device=device)

        # 使用持久缓存编译 aoti
        kernel_lib_path = aoti_compile_with_persistent_cache(
            namespace_name,
            op_name_with_overload,
            a.device.type,
            False,
            torch.ops.aten.add,
            args=(a, b),
            kwargs={"alpha": 3.0},
        )
        self.assertTrue(Path(kernel_lib_path).exists())

        # 获取设备的 aoti 缓存目录
        device_kernel_cache = aoti_eager_cache_dir(namespace_name, device.type)
        # 构建操作的 kernel 配置文件路径
        kernel_conf = device_kernel_cache / f"{op_name_with_overload}.json"
        self.assertTrue(kernel_conf.exists())

        # 加载 aoti 的缓存数据
        json_data = load_aoti_eager_cache(
            namespace_name, op_name_with_overload, a.device.type
        )
        op_info = json_data[0]

        # 确保加载的信息是字典类型
        self.assertTrue(isinstance(op_info, dict))
        # 确保字典中有 "meta_info" 键
        self.assertTrue("meta_info" in op_info)
        # 确保 "meta_info" 列表长度为 3
        self.assertTrue(len(op_info["meta_info"]) == 3)

        # 确保第一个元素的 "sizes" 是空列表
        self.assertTrue(op_info["meta_info"][0]["sizes"] == [])
        # 确保第一个元素的 "strides" 是空列表
        self.assertTrue(op_info["meta_info"][0]["strides"] == [])
        # 确保第一个元素不包含 "scalar_value" 键
        self.assertTrue("scalar_value" not in op_info["meta_info"][0])

        # 确保第二个元素的 "sizes" 是空列表
        self.assertTrue(op_info["meta_info"][1]["sizes"] == [])
        # 确保第二个元素的 "strides" 是空列表
        self.assertTrue(op_info["meta_info"][1]["strides"] == [])
        # 确保第二个元素不包含 "scalar_value" 键
        self.assertTrue("scalar_value" not in op_info["meta_info"][1])

        # 确保第三个元素的 "sizes" 是空列表
        self.assertTrue(op_info["meta_info"][2]["sizes"] == [])
        # 确保第三个元素的 "strides" 是空列表
        self.assertTrue(op_info["meta_info"][2]["strides"] == [])
        # 确保第三个元素包含 "scalar_value" 键
        self.assertTrue("scalar_value" in op_info["meta_info"][2])

        # 在 "aten" 命名空间下编译操作库
        with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
            # 生成随机张量 a 和 b，设备为当前设备
            a = torch.randn(128, device=device)
            b = torch.randn(128, device=device)

            # 定义标量值列表
            scalar_values = [1.0, 2.0, 3.0]
            ref_values = []
            # 遍历标量值列表，对 a 和 b 执行加法，将结果保存到 ref_values 中
            for scalar_value in scalar_values:
                ref_values.append(torch.add(a, b, alpha=scalar_value))

            # 使用 aoti 编译注册操作
            register_ops_with_aoti_compile(
                namespace_name, [op_name], dispatch_key, torch_compile_op_lib_impl
            )

            res_values = []
            # 再次遍历标量值列表，对 a 和 b 执行加法，将结果保存到 res_values 中
            for scalar_value in scalar_values:
                res_values.append(torch.add(a, b, alpha=scalar_value))

            # 确保 ref_values 和 res_values 的长度相等
            self.assertEqual(len(ref_values), len(res_values))
            # 确保 ref_values 和 res_values 相等
            self.assertEqual(ref_values, res_values)
    # 定义测试方法，用于测试在 eager 模式下注册和调用 ATen 操作的功能
    def test_eager_aoti_override_registration(self):
        # 设置命名空间名称
        namespace_name = "aten"
        # 设置分派键
        dispatch_key = "CPU"
        # 创建 CPU 设备对象
        device = torch.device("cpu")
        # 如果当前设备是 CUDA，则更新分派键和设备对象
        if self.device.lower() == "cuda":
            dispatch_key = "CUDA"
            device = torch.device("cuda")

        # 设置一组一元操作名称列表
        unary_op_set = ["abs", "acos"]

        # 定义函数 fn，用于调用 torch 中的指定操作
        def fn(x, op_name=""):
            return getattr(torch, op_name)(x)

        # 调用 torch.compile 直接获取优化后的函数结果
        x = torch.randn(3, 4, device=device)

        # 初始化参考结果数组
        ref_array = []
        # 遍历一元操作集合
        for unary_op_name in unary_op_set:
            # 使用 functools.partial 创建带有操作名的函数
            opt_fn = torch.compile(functools.partial(fn, op_name=unary_op_name))
            # 调用优化后的函数并存储结果
            ref = opt_fn(x)
            ref_array.append(ref)

        # 使用 _scoped_library 上下文管理器注册 ATen 操作
        with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
            # 使用 register_ops_with_aoti_compile 注册操作到 ATen 编译库
            register_ops_with_aoti_compile(
                namespace_name, unary_op_set, dispatch_key, torch_compile_op_lib_impl
            )

            # 初始化结果数组
            res_array = []
            # 遍历一元操作集合
            for unary_op_name in unary_op_set:
                # 调用 torch 中的对应一元操作并存储结果
                res_array.append(getattr(torch, unary_op_name)(x))

            # 遍历参考结果数组和实际结果数组，使用 self.assertEqual 进行断言比较
            for ref, res in zip(ref_array, res_array):
                self.assertEqual(ref, res)

        # 创建输入张量 a 和最小值张量 min_tensor
        a = torch.randn(128, device=device)
        min_tensor = torch.randn(128, device=device)
        # 创建最大值张量 max_tensor，其为 min_tensor 增加 0.5
        max_tensor = min_tensor + 0.5

        # 使用 min_tensor 对 a 进行 clamp 操作，存储参考结果
        ref_with_min = torch.ops.aten.clamp(a, min_tensor)
        # 使用 min_tensor 和 max_tensor 对 a 进行 clamp 操作，存储参考结果
        ref_with_min_max = torch.ops.aten.clamp(a, min_tensor, max_tensor)

        # 使用 _scoped_library 上下文管理器注册 ATen 操作
        with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
            # 使用 register_ops_with_aoti_compile 注册 clamp 操作到 ATen 编译库
            register_ops_with_aoti_compile(
                namespace_name, ["clamp"], dispatch_key, torch_compile_op_lib_impl
            )
            # 使用 min_tensor 对 a 进行 clamp 操作，存储实际结果
            res_with_min = torch.ops.aten.clamp(a, min_tensor)
            # 使用 min_tensor 和 max_tensor 对 a 进行 clamp 操作，存储实际结果
            res_with_min_max = torch.ops.aten.clamp(a, min_tensor, max_tensor)
            # 使用 self.assertEqual 检查参考结果与实际结果的一致性
            self.assertEqual(ref_with_min, res_with_min)
            self.assertEqual(ref_with_min_max, res_with_min_max)

    # 定义测试方法，测试在不同数据类型下对张量执行加法操作的功能
    def test_add_const_int(self):
        # 定义函数 fn，对输入张量执行加法操作，并返回结果
        def fn(a):
            return (a + 1, torch.add(a, 1, alpha=2))

        # 遍历数据类型列表 [torch.float32, torch.int32, torch.int64]
        for dtype in [torch.float32, torch.int32, torch.int64]:
            # 调用 self.common 方法，传递 fn 函数和参数 (torch.arange(32, dtype=dtype),)
            self.common(fn, (torch.arange(32, dtype=dtype),))

    # 定义测试方法，测试在浮点数张量上执行加法操作的功能
    def test_add_const_float(self):
        # 定义函数 fn，对输入张量执行加法操作，并返回结果
        def fn(a):
            return (a + 1.5,)

        # 调用 self.common 方法，传递 fn 函数和参数 (torch.randn(32),)
        self.common(fn, (torch.randn(32),))

    # 定义测试方法，测试在执行就地加法操作时张量维度置换的功能
    def test_add_inplace_permuted(self):
        # 定义函数 fn，对输入张量 x 和 y 执行就地加法操作，并返回结果
        def fn(x, y):
            return x.add_(y)

        # 创建张量 x 和 y，其维度分别为 [2, 13, 12, 17] 和 [2, 13, 1, 17]
        x = torch.ones([2, 12, 13, 17]).transpose(1, 2)
        y = torch.randn([2, 13, 1, 17])

        # 调用 self.common 方法，传递 fn 函数和参数 (x, y)
        self.common(fn, (x, y))

    # 定义测试方法，测试在复数张量上执行加法操作的功能
    def test_add_complex(self):
        # 定义函数 fn，对输入张量 a 和 b 执行加法操作，并返回结果
        def fn(a, b, alpha):
            return torch.add(a, b, alpha=alpha)

        # 创建复数张量 x 和 y
        x = torch.tensor([1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1])
        y = torch.tensor([1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1])

        # 调用 self.common 方法，传递 fn 函数和参数 (x, y, 2)
        self.common(fn, (x, y, 2))
    def test_add_complex3(self):
        # 修复 https://github.com/pytorch/pytorch/issues/115071
        # 定义一个编译函数 fn，接受任意数量的参数
        @torch.compile
        def fn(*args):
            # 计算 args[0] 的负数
            a = torch.neg(args[0])
            # 计算 args[0] 与自身的和
            b = torch.add(args[0], args[0])
            return (a, b)

        # 创建一个形状为 (41,) 的复数张量 x，并复制给 y
        x = torch.randn(41, dtype=torch.complex64)
        y = x.clone()
        # 调用 fn 函数，不应该修改输入的内容
        fn(x)
        # 断言 x 与 y 相等
        self.assertEqual(x, y)

    def test_add_complex4(self):
        # 定义一个编译函数 fn，接受两个参数 a 和 b
        @torch.compile
        def fn(a, b):
            # 计算 a 与 b 的和赋值给 c
            c = a + b
            # 计算 a 与 b 的和赋值给 d
            d = a + b
            return c + d

        # 遍历复数类型 [torch.complex32, torch.complex64, torch.complex128]
        for dtype in [torch.complex32, torch.complex64, torch.complex128]:
            # 创建指定 dtype 和设备的张量 x 和 y
            x = torch.tensor(
                [1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1],
                dtype=dtype,
                device=self.device,
            )
            y = torch.tensor(
                [1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1],
                dtype=dtype,
                device=self.device,
            )
            # 运行并获取 fn 的代码和结果
            _, code = run_and_get_code(fn, x, y)
            # 断言代码中 "view_dtype" 或 "aten.view" 出现的次数为 3
            self.assertEqual(
                " ".join(code).count(
                    "view_dtype" if config.cpp_wrapper else "aten.view"
                ),
                3,
            )

    def test_add_complex5(self):
        # 定义一个函数 fn，接受三个参数 a、b 和 alpha
        def fn(a, b, alpha):
            # 返回 a 与 b 的加法，使用 alpha 作为系数
            return torch.add(a, b, alpha=alpha)

        # 创建一个复数张量 x 和 y
        x = torch.tensor([[1 + 1j, -1 + 1j], [-2 + 2j, 3 - 3j]])
        y = torch.tensor([[1 + 1j, -1 + 1j], [-2 + 2j, 3 - 3j]])

        # 使用 self.common 方法调用 fn 函数
        self.common(fn, (x, y, 2))

    def test_add_complex6(self):
        # 修复 https://github.com/pytorch/pytorch/issues/125745
        # 使用广播方式添加复数张量
        def fn(a, b, alpha):
            # 返回 a 与 b 的加法，使用 alpha 作为系数
            return torch.add(a, b, alpha=alpha)

        # 创建一个形状为 (1,4) 的复数张量 x 和 (1,1) 的复数张量 y
        x = torch.tensor([[1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j]])
        y = torch.tensor([[1 + 1j]])

        # 使用 self.common 方法调用 fn 函数
        self.common(fn, (x, y, 2))

    def test_concat_add_inplace(self):
        # 定义一个函数 fn，接受三个参数 x、y 和 z
        def fn(x, y, z):
            # 将 x 和 y 沿着 dim=1 进行拼接，并对结果加 z
            return torch.cat([x, y], dim=1).add_(z)

        # 创建形状为 [2, 12, 14, 14] 的随机张量 x、y 和 z
        x = torch.randn([2, 12, 14, 14])
        y = torch.randn([2, 12, 14, 14])
        z = torch.randn([2, 24, 14, 14])

        # 使用 self.common 方法调用 fn 函数
        self.common(fn, (x, y, z))

    def test_abs(self):
        # 定义一个函数 fn，接受一个参数 a
        def fn(a):
            # 返回 a 除以 |a|+1 的结果，组成元组返回
            return (a / (torch.abs(a) + 1),)

        # 使用 self.common 方法调用 fn 函数，传入一个形状为 (17,) 的随机张量
        self.common(fn, (torch.randn(17),))

    def test_angle(self):
        # 定义一个函数 fn，接受三个参数 a、b 和 c
        def fn(a, b, c):
            # 返回 a、b 和 c 的角度
            return torch.angle(a), torch.angle(b), torch.angle(c)

        # 创建不同类型的输入张量：复数、实数、整数
        complex_input = torch.tensor(
            [1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1, float("nan")]
        )
        real_input = torch.tensor([-1.0, 0.0, 1.0, float("nan")])
        integer_real_input = torch.tensor([-1, 0, 1])

        # 使用 self.common 方法调用 fn 函数，传入不同类型的输入
        self.common(fn, (complex_input, real_input, integer_real_input))

    def test_sgn(self):
        # 定义一个函数 fn，接受一个参数 a
        def fn(a):
            # 返回 a 和 a+1 的符号函数值，组成元组返回
            return torch.sgn(a), torch.sgn(a + 1) - 1

        # 使用 self.common 方法调用 fn 函数，传入从 -10 到 10 的 41 个均匀间隔的数值
        self.common(fn, [torch.linspace(-10, 10, 41)])

    @skipCUDAIf(not SM80OrLater, "uses bfloat16 which requires SM >= 80")
    def test_scatter_bf16(self):
        # 定义一个函数 fn，用于在输入张量 inp 上执行 scatter_add 操作，按照 index 指定的索引添加 src 的值
        def fn(inp, src, index):
            return inp.scatter_add(0, index, src)

        # 遍历不同的数据类型进行测试：torch.int64, torch.bool, torch.bfloat16
        for dtype in [torch.int64, torch.bool, torch.bfloat16]:
            # 调用 self.common 方法，传入 fn 函数和相应的参数进行测试
            self.common(
                fn,
                [
                    torch.zeros(3, 5, dtype=dtype),  # 创建一个 3x5 的零张量，指定数据类型为 dtype
                    torch.ones((2, 5), dtype=dtype),  # 创建一个 2x5 的全一张量，指定数据类型为 dtype
                    torch.tensor([[0, 1, 2, 0, 0]]),  # 创建一个指定值的张量，数据类型默认推断
                ],
            )

    def test_randn_generator(self):
        # 定义一个函数 fn，生成一个形状为 [20, 20] 的随机张量，设备为 a.device，使用指定的生成器 generator
        def fn(a, generator):
            return torch.randn([20, 20], generator=generator, device=a.device)

        # 调用 self.common 方法，传入 fn 函数和相应的参数进行测试，assert_equal=False 表示不进行相等性断言
        self.common(fn, (torch.linspace(-10, 10, 41), None))

        # 测试生成器 generator 在 dynamo 环境下不支持的情况
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "Generator"):
            # 调用 self.common 方法，传入 fn 函数和包含生成器的参数进行测试
            self.common(fn, (torch.linspace(-10, 10, 41), torch.Generator(self.device)))

    def test_sgn_extremal(self):
        # 定义一个函数 fn，返回输入张量 a 中每个元素的符号函数值组成的元组
        def fn(a):
            return (torch.sgn(a),)

        # 调用 self.common 方法，传入 fn 函数和相应的参数进行测试
        self.common(fn, [torch.tensor([np.nan, np.inf, -np.inf, 0])])

    def test_max_min(self):
        # 定义一个函数 fn，返回输入张量 a 和 b 中对应位置元素的最大值和最小值组成的元组
        def fn(a, b):
            return (torch.maximum(a, b), torch.minimum(a, b))

        # 调用 self.common 方法，传入 fn 函数和相应的参数进行测试
        self.common(fn, (torch.randn(8), torch.randn(8)))

        # 创建包含 NaN 值的张量，并调用 self.common 方法进行测试
        t1 = torch.randn(8)
        t1[0] = float("nan")
        t2 = torch.randn(8)
        t2[1] = float("nan")
        self.common(fn, (t1, t2))

    def test_neg_max_uint8(self):
        # https://github.com/pytorch/pytorch/issues/93380
        # 定义一个函数 fn，对输入张量 a 取负值后与 b 中对应位置元素的最大值进行比较
        def fn(a, b):
            c = torch.neg(a)  # 对张量 a 取负值
            return torch.maximum(b, c)  # 返回 b 和 c 的最大值张量

        # 创建随机生成的 uint8 类型的张量 a 和 b，并调用 self.common 方法进行测试
        a = torch.randint(256, (1,), dtype=torch.uint8)
        b = torch.randint(256, (8390,), dtype=torch.uint8)
        self.common(fn, (a, b))

    def test_compar(self):
        # 定义一个函数 fn，返回输入张量 x 中每个元素与指定值比较的结果，分别包括大于、大于等于、等于、小于等于、小于、不等于
        def fn(x):
            return x.gt(3.5), x.ge(3.5), x.eq(3.5), x.le(2.5), x.lt(3.5), x.ne(3.5)

        # 创建一个张量 a，包含一个值为 3 的元素，并调用 self.common 方法进行测试
        a = torch.tensor([3])
        self.common(fn, (a,))

    def test_horizonal_fusion1(self):
        # 定义一个函数 fn，对输入张量 a, b, c 进行水平融合操作，分别进行加法、减法和乘法运算
        def fn(a, b, c):
            return (a + b, a - c, b * c)

        # 调用 self.common 方法，传入 fn 函数和相应的参数进行测试
        self.common(
            fn, (torch.randn(8, 16, 16), torch.randn(8, 16, 16), torch.randn(1, 16, 1))
        )

    def test_horizonal_fusion2(self):
        # 定义一个函数 fn，对输入张量 a, b, c 进行水平融合操作，分别加上常数 1, 2, 3
        def fn(a, b, c):
            return a + 1, b + 2, c + 3

        # 调用 self.common 方法，传入 fn 函数和相应的参数进行测试
        self.common(fn, (torch.randn(8, 16, 8), torch.randn(8, 16), torch.randn(16, 8)))

    def test_vertical_fusion1(self):
        # 定义一个函数 fn，实现一个复杂的垂直融合操作
        def fn(sa, ct, p):
            # From torchbench.pyhpc_equation_of_state
            v17 = -3.087032500374211e-7
            v18 = -1.988366587925593e-8
            v19 = -1.061519070296458e-11
            v20 = 1.550932729220080e-10
            t15 = v19 * ct
            t19 = v17 + ct * (v18 + t15) + v20 * sa
            t20 = 1.0 / t19
            t128 = t19 * p
            return t20 + t128

        # 调用 self.common 方法，传入 fn 函数和相应的参数进行测试
        self.common(
            fn,
            (
                torch.randn(204, 204, 26),  # 创建一个随机张量，形状为 [204, 204, 26]
                torch.randn(204, 204, 26),  # 创建一个随机张量，形状为 [204, 204, 26]
                torch.randn(26),  # 创建一个形状为 [26] 的随机张量
            ),
        )
        assertGeneratedKernelCountEqual(self, 1)
    @config.patch({"fx_graph_cache": False})
    # 使用配置修补装饰器，禁用 fx_graph_cache 功能
    def test_forced_buffer_realize(self):
        # 测试 torch._test_inductor_realize 强制缓冲区实现的功能
        def fn(a):
            # 计算 a * 2，并强制实现该操作
            b = test_operators.realize(a * 2)
            return (b * 2,)

        # 调用共同测试函数，验证功能
        self.common(fn, (torch.randn(10),))
        # 断言预期结果与现有 IR 节点数的匹配
        self.assertEqual(torch._inductor.metrics.ir_nodes_pre_fusion, 2)

    @config.patch({"fx_graph_cache": False})
    # 使用配置修补装饰器，禁用 fx_graph_cache 功能
    def test_scheduler_vertical_fusion1(self):
        realize = test_operators.realize

        def fn(sa, ct, p):
            # 来自 torchbench.pyhpc_equation_of_state 的注释
            v17 = -3.087032500374211e-7
            v18 = -1.988366587925593e-8
            v19 = -1.061519070296458e-11
            v20 = 1.550932729220080e-10
            # 实现 v19 * ct，并返回结果
            t15 = realize(v19 * ct)
            # 计算复杂表达式 t19，并强制实现其中的运算
            t19 = realize(v17 + ct * (v18 + t15) + v20 * sa)
            # 计算 t19 的倒数，并强制实现该操作
            t20 = realize(1.0 / t19)
            # 计算 t19 * p，并强制实现该操作
            t128 = realize(t19 * p)
            return t20 + t128

        # 调用共同测试函数，验证功能
        self.common(
            fn,
            (
                torch.randn(204, 204, 26),
                torch.randn(204, 204, 26),
                torch.randn(26),
            ),
        )
        # 断言预期结果与现有 IR 节点数的匹配
        self.assertEqual(torch._inductor.metrics.ir_nodes_pre_fusion, 5)
        # 如果不是 C++ 后端，则断言生成的内核数量为 1，否则为 2
        assertGeneratedKernelCountEqual(
            self, 1 if not is_cpp_backend(self.device) else 2
        )

    def test_index_propagation(self):
        def copy(x):
            # 创建索引 i，范围是 x 的大小，设备为 x 的设备
            i = torch.arange(x.size(0), device=x.device)
            # 根据索引 i 复制 x 的内容
            return x[i]

        # 创建一个张量 x，形状为 (8,)，设备为 self.device
        x = torch.randn(8, device=self.device)
        # 对 copy 函数应用优化，并赋值给 copy_opt
        copy_opt = torch._dynamo.optimize("inductor")(copy)

        # 预期的复制结果
        expect = copy(x)
        # 运行优化后的函数，并断言无间接索引
        actual = _run_and_assert_no_indirect_indexing(self, copy_opt, x)
        # 断言预期结果与实际结果相等
        self.assertEqual(expect, actual)

    @dynamo_config.patch("capture_dynamic_output_shape_ops", True)
    # 使用动态配置修补装饰器，捕获动态输出形状操作，并设置为 True
    # 参考：https://github.com/halide/Halide/issues/8308
    @config.patch("halide.scheduler_cpu", "Mullapudi2016")
    # 使用配置修补装饰器，设置 Halide CPU 调度器为 Mullapudi2016
    @config.patch("halide.scheduler_cuda", "Li2018")
    # 使用配置修补装饰器，设置 Halide CUDA 调度器为 Li2018
    @config.patch(implicit_fallbacks=True)
    # 使用配置修补装饰器，启用隐式回退
    def test_index_propagation_nested_indirect_indexing(self):
        def nested(x, repeats):
            # 创建 rank 张量，形状是 repeats.numel()，设备为 x 的设备
            rank = torch.arange(repeats.numel(), device=x.device)
            # 重复 rank 张量中的元素，以 repeats 张量中的值作为重复次数
            index = rank.repeat_interleave(repeats, dim=0)
            # 根据 index 索引选取 x 的内容，维度为 0
            return torch.index_select(x, index=index, dim=0)

        # 示例输入数据
        example_inputs = (
            torch.randn((32, 64), device=self.device),
            repeats := torch.tensor([5, 10, 15], device=self.device),
        )
        # 标记 repeats 张量为动态符号整数（symint）
        torch._dynamo.mark_dynamic(repeats, 0)

        # 对 nested 函数应用优化，并赋值给 nested_opt
        nested_opt = torch._dynamo.optimize("inductor")(nested)

        # 预期的嵌套索引选择结果
        expect = nested(*example_inputs)
        # 运行优化后的函数，并获取实际结果
        actual = nested_opt(*example_inputs)
        # 断言预期结果与实际结果相等
        self.assertEqual(expect, actual)
    # 定义测试函数，用于测试反转操作的索引传播
    def test_index_propagation_flip(self):
        # 定义一个函数 flip，用于反转张量 x
        def flip(x):
            # 创建一个从 x.size(0)-1 到 0 的张量索引 i，设备与 x 相同
            i = torch.arange(x.size(0) - 1, -1, -1, device=x.device)
            return x[i]

        # 生成一个形状为 (8,) 的随机张量 x，设备为 self.device
        x = torch.randn(8, device=self.device)
        # 对 flip 函数进行优化，返回优化后的版本 flip_opt
        flip_opt = torch._dynamo.optimize("inductor")(flip)

        # 计算期望结果，即对 x 进行反转操作
        expect = flip(x)
        # 运行并断言优化后的 flip_opt 与期望结果相同，无间接索引
        actual = _run_and_assert_no_indirect_indexing(self, flip_opt, x)
        self.assertEqual(expect, actual)

    # 定义测试函数，用于测试重复插值操作的索引传播
    def test_index_propagation_floordiv(self):
        # 定义一个函数 repeat_interleave，用于重复插值张量 x
        def repeat_interleave(x, n):
            # 创建一个从 0 到 x.shape[0]*n-1 的张量索引 i，设备与 x 相同
            i = torch.arange(x.shape[0] * n, device=x.device)
            return x[i // n]

        # 生成一个形状为 (8, 16) 的随机张量 x，设备为 self.device
        x = torch.randn(8, 16, device=self.device)
        # 对 repeat_interleave 函数进行优化，返回优化后的版本 repeat_interleave_opt
        repeat_interleave_opt = torch._dynamo.optimize("inductor")(repeat_interleave)
        
        # 对于静态形状，可以证明界限，我们的动态形状推理不够好
        has_assert = ifdynstaticdefault(False, True)
        # 这应该折叠为直接索引
        actual = _run_and_assert_no_indirect_indexing(
            self, repeat_interleave_opt, x, 3, has_assert=has_assert
        )
        # 计算期望结果，即对 x 进行重复插值操作
        expect = torch.repeat_interleave(x, 3, dim=0)
        self.assertEqual(expect, actual)
        self.assertEqual(actual, repeat_interleave(x, 3))

    # 定义测试函数，用于测试重复操作的索引传播
    def test_index_propagation_remainder(self):
        # 定义一个函数 repeat，用于重复张量 x
        def repeat(x, n):
            # 创建一个从 0 到 x.shape[0]*n-1 的张量索引 i，设备与 x 相同
            i = torch.arange(x.shape[0] * n, device=x.device)
            return x[i % x.shape[0]]

        # 生成一个形状为 (8, 16) 的随机张量 x，设备为 self.device
        x = torch.randn(8, 16, device=self.device)
        # 对 repeat 函数进行优化，返回优化后的版本 repeat_opt
        repeat_opt = torch._dynamo.optimize("inductor")(repeat)

        # 对于静态形状，可以证明界限，我们的动态形状推理不够好
        has_assert = ifdynstaticdefault(False, True)
        # 这应该折叠为直接索引
        actual = _run_and_assert_no_indirect_indexing(
            self, repeat_opt, x, 3, has_wrapping=False, has_assert=has_assert
        )
        # 计算期望结果，即对 x 进行重复操作
        expect = x.repeat(3, 1)
        self.assertEqual(expect, actual)
        self.assertEqual(actual, repeat(x, 3))

    # 定义测试函数，用于测试反射填充操作的索引传播
    def test_index_propagation_abs(self):
        # 定义一个函数 reflection_pad_left，用于反射填充左侧张量 x
        def reflection_pad_left(x, n):
            # 创建一个从 0 到 x.shape[0]+n-1 的张量索引 i，设备与 x 相同
            i = torch.arange(x.shape[0] + n, device=x.device)
            return x[(i - n).abs()]

        # 生成一个形状为 (8,) 的随机张量 x，设备为 self.device
        x = torch.randn(8, device=self.device)
        # 对 reflection_pad_left 函数进行优化，返回优化后的版本 opt_fn
        opt_fn = torch._dynamo.optimize("inductor")(reflection_pad_left)

        # 对于静态形状，可以证明界限，我们的动态形状推理不够好
        has_assert = ifdynstaticdefault(False, True)
        # 这应该折叠为直接索引
        actual = _run_and_assert_no_indirect_indexing(
            self, opt_fn, x, 3, has_wrapping=False, has_assert=has_assert
        )
        # 计算期望结果，即对 x 进行反射填充操作
        expect = reflection_pad_left(x, 3)
        self.assertEqual(expect, actual)
    # 定义测试方法，验证索引传播与设备断言是否屏蔽
    def test_index_propagation_device_assert_masked(self):
        # 定义内部函数fn，接受参数a
        def fn(a):
            # 生成一个张量，从0到a的大小，设备与a相同
            idx = torch.arange(a.size(0), device=a.device)
            # 在idx张量的末尾填充1050个0，设备与idx相同
            padded_idx = torch.constant_pad_nd(idx, (1050, 0))
            # 将padded_idx张量中小于0的值替换为0，设备与padded_idx相同
            padded_idx = torch.where(padded_idx >= 0, padded_idx, padded_idx)
            # 返回a张量中根据padded_idx索引的值
            return a[padded_idx]

        # 调用共用方法common，传入fn函数和一个包含随机张量的元组
        self.common(fn, (torch.randn(1024),))

    # 使用config.patch装饰器，设置debug_index_asserts为False
    @config.patch(debug_index_asserts=False)
    # 定义测试方法，验证计算缓冲内联化
    def test_computed_buffer_inlining(self):
        # 定义内部函数flip，接受参数x
        def flip(x):
            # 生成一个从x.size(0)-1到0的逆序张量，设备与x相同
            idx = torch.arange(x.size(0) - 1, -1, -1, device=x.device)
            # 返回x张量根据idx索引的值和idx张量本身
            return x[idx], idx

        # 对flip函数进行优化，并获取优化后的版本flip_opt
        flip_opt = torch._dynamo.optimize("inductor")(flip)
        # 生成一个大小为8的随机张量，并设备为self.device
        x = torch.randn(8, device=self.device)

        # 预期值为flip(x)的结果
        expect = flip(x)
        # 运行并断言未间接索引的运行结果与期望值相等
        actual = _run_and_assert_no_indirect_indexing(self, flip_opt, x)
        # 断言期望值与实际值相等
        self.assertEqual(expect, actual)

    # 定义测试方法，验证不安全的掩码索引
    def test__unsafe_masked_index(self):
        # 定义内部函数fn，接受参数a, mask, idx
        def fn(a, mask, idx):
            # 调用aten._unsafe_masked_index函数，传入参数a, mask, idx, 1
            return aten._unsafe_masked_index(a, mask, idx, 1)

        # 调用共用方法common，传入fn函数和一个包含随机张量及掩码的元组
        self.common(
            fn,
            (
                torch.randn(8, device=self.device),
                torch.tensor([True, False, True], device=self.device),
                [torch.tensor([3, 9, -2], device=self.device)],
            ),
        )

    # 定义测试方法，验证不安全的掩码索引放置累积
    def test__unsafe_masked_index_put_accumulate(self):
        # 定义内部函数fn，接受参数a, mask, idx, values
        def fn(a, mask, idx, values):
            # 调用aten._unsafe_masked_index_put_accumulate函数，传入参数a, mask, idx, values
            return aten._unsafe_masked_index_put_accumulate(a, mask, idx, values)

        # 调用共用方法common，传入fn函数和一个包含随机张量及相关参数的元组
        self.common(
            fn,
            (
                torch.randn(8, device=self.device),
                torch.tensor([True, False, True], device=self.device),
                [torch.tensor([3, 9, -2], device=self.device)],
                torch.randn(3, device=self.device),
            ),
        )

    # 定义测试方法，验证求和操作1
    def test_sum1(self):
        # 定义内部函数fn，接受参数a, b
        def fn(a, b):
            # 返回(a + b)张量在最后一个维度上的和
            return ((a + b).sum(-1),)

        # 调用共用方法common，传入fn函数和一个包含两个随机张量的元组
        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    # 定义测试方法，验证求和操作2
    def test_sum2(self):
        # 定义内部函数fn，接受参数a, b
        def fn(a, b):
            # 返回(a + b)张量在指定维度列表上的和，以及(a + b)张量在最后一个维度上的和
            return ((a + b).sum([1, 2]), (a + b).sum(-1))

        # 调用共用方法common，传入fn函数和一个包含两个指定维度大小的随机张量的元组
        self.common(fn, (torch.randn(8, 9, 3, 21), torch.randn(8, 9, 3, 21)))

    # 定义测试方法，验证求和操作3
    def test_sum3(self):
        # 定义内部函数fn，接受参数a, b
        def fn(a, b):
            # 计算a + b的结果
            r1 = a + b
            # 计算r1在最后一个维度上的和
            r2 = r1.sum(-1)
            # 计算张量b在去除所有长度为1的维度后加10的结果
            r3 = torch.squeeze(b) + 10
            # 返回r1, r2, r3三个张量
            return (r1, r2, r3)

        # 调用共用方法common，传入fn函数和一个包含两个随机张量的元组，并设置允许的误差范围
        self.common(fn, (torch.randn(10, 10), torch.randn(1, 10)), atol=1e-5, rtol=2e-3)

    # 定义测试方法，验证求和操作4
    def test_sum4(self):
        # 定义内部函数fn，接受参数a
        def fn(a):
            # 将a张量中的每个元素加1，并赋值给b
            b = a + 1
            # 计算b张量在最后一个维度上的和，赋值给c
            c = b.sum(-1)
            # 将c张量中的每个元素加3，并赋值给d
            d = c + 3
            # 计算d张量在最后一个维度上的和，赋值给e
            e = d.sum(-1)
            # 将e张量中的每个元素加5，并赋值给f
            f = e + 5
            # 返回f, e, d, c, b五个张量
            return (f, e, d, c, b)

        # 调用共用方法common，传入fn函数和一个指定形状的随机张量的元组
        self.common(fn, (torch.randn(1, 16, 8, 8),))
    def test_sum5(self):
        # 定义内部函数 fn，对输入张量 a 进行一系列操作并返回元组
        def fn(a):
            # b 是 a + 1 的结果
            b = a + 1
            # c 是 b 沿着最后一个维度的和
            c = b.sum(-1)
            # d 是 c + 3 的结果
            d = c + 3
            # e 是 d 沿着最后一个维度的和
            e = d.sum(-1)
            # f 是 e + 5 的结果，返回为元组 (f,)
            f = e + 5
            return (f,)

        # 调用 self.common 方法，传入 fn 函数和一个随机生成的张量作为参数
        self.common(fn, (torch.randn(1, 17, 8, 9),))

    def test_reduction1(self):
        # 定义内部函数 fn，对输入张量 a 进行多种统计操作并返回元组
        def fn(a):
            # 返回张量 a 的总和、最大值、最小值、最大值索引、最小值索引组成的元组
            return (a.sum(), a.max(), a.min(), a.argmax(), a.argmin())

        # 调用 self.common 方法，传入 fn 函数和一个包含特定数值的张量作为参数
        self.common(fn, (torch.tensor([float("-inf"), 0.0, float("inf")]),))

    @skip_if_x86_mac()
    def test_reduction2(self):
        # 定义内部函数 fn，对输入张量 a 进行多种统计操作并返回元组
        def fn(a):
            # FIXME: a.argmax，返回张量 a 的总和、最大值、最小值、但不包括最大值索引
            return (a.sum(), a.max(), a.min(), a.argmin())

        # 调用 self.common 方法，传入 fn 函数和一个全是正无穷的张量作为参数
        self.common(fn, (torch.full((4,), float("inf")),))

    @skip_if_x86_mac()
    def test_reduction3(self):
        # 定义内部函数 fn，对输入张量 a 进行多种统计操作并返回元组
        def fn(a):
            # FIXME: a.argmin，返回张量 a 的总和、最大值、最小值、但不包括最小值索引
            return (a.sum(), a.max(), a.min(), a.argmax())

        # 调用 self.common 方法，传入 fn 函数和一个全是负无穷的张量作为参数
        self.common(fn, (torch.full((4,), float("-inf")),))

    def test_reduction4(self):
        # 如果设备是 CPU，则跳过测试，因为结果是非确定性的
        if self.device == "cpu":
            raise unittest.SkipTest("Non-deterministic CPU results")

        # 定义内部函数 fn，对输入张量 a 进行 argmax 和 argmin 操作并返回元组
        def fn(a):
            # 返回张量 a 沿着最后一个维度的 argmax 和 argmin 组成的元组
            return (a.argmax(-1), a.argmin(-1))

        # 定义输入张量列表
        inputs = (torch.ones(128), torch.ones(4, 4, 1))
        # 遍历输入列表，对每个输入调用 self.common 方法
        for i in inputs:
            # 若当前设备不是 Halide 后端，则检查低精度
            self.common(fn, (i,), check_lowp=not is_halide_backend(self.device))

    @config.patch(unroll_reductions_threshold=1)
    def test_reduction5(self):
        # 如果设备是 CPU，则跳过测试，因为结果是非确定性的
        if self.device == "cpu":
            raise unittest.SkipTest("Non-deterministic CPU results")

        # 定义内部函数 fn，对输入张量 a 进行多种统计操作并返回元组
        def fn(a):
            # 返回张量 a 的总和、最大值、最小值、但不包括最大值索引
            return (a.sum(), a.max(), a.min(), a.argmax())

        # 调用 self.common 方法，传入 fn 函数和一个全是负无穷的张量作为参数
        self.common(fn, (torch.full((4,), float("-inf")),))

    def test_prod(self):
        # 定义内部函数 fn，分别计算输入张量 a 沿着不同维度的乘积，并返回结果
        def fn(a):
            return a.prod(0), a.prod(1), a.prod()

        # 调用 self.common 方法，分别传入 fn 函数和两个随机生成的张量作为参数
        self.common(fn, (torch.rand((10, 10)),))
        self.common(fn, (torch.rand((1, 2050)),))

    def test_unroll_small_reduction(self):
        # 定义内部函数 fn，对输入张量 x 进行多种归约操作并返回结果元组
        def fn(x):
            # 分别计算 x 沿着最后一个维度的最小值和最小值索引，最大值和最大值索引，以及总和等
            val1, index1 = x.min(-1)
            val2, index2 = x.max(-1)
            return (
                val1,
                index1,
                val2,
                index2,
                x.sum(-1),
                (x > 1).any(-1),
                (x > 0).all(-1),
                x.argmin(-1),
                x.argmax(-1),
                x.amin(-1),
                x.amax(-1),
                x.aminmax(),
            )

        # 使用配置修补，设置 unroll_reductions_threshold=8
        with config.patch(unroll_reductions_threshold=8):
            # 对大小为 8 的输入张量调用 self.common 方法
            self.common(fn, (torch.randn(8, 3),))
        # 重置 torch._dynamo 状态
        torch._dynamo.reset()
        # 使用配置修补，设置 unroll_reductions_threshold=1
        with config.patch(unroll_reductions_threshold=1):
            # 对大小为 8 的输入张量再次调用 self.common 方法
            self.common(fn, (torch.randn(8, 3),))

    def test_multilayer_sum_low_prec(self):
        # 如果设备是 CPU，则跳过测试，因为需要 GPU
        if self.device == "cpu":
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        # 定义内部函数 fn，计算输入张量的平均值
        def fn(a):
            return torch.mean(a)

        # 调用 self.common 方法，传入 fn 函数和一个随机生成的 fp16 类型张量作为参数
        self.common(fn, ((torch.rand((10, 3, 352, 352), dtype=torch.float16),)))
    # 定义测试方法，用于测试多层次的素数大小
    def test_multilayer_prime_size(self):
        # 定义内部函数 fn，对输入张量 a 进行最大值和总和的计算
        def fn(a):
            return torch.max(a), torch.sum(a)

        # 创建一个长度为 3999971 的全零张量 sample，数据类型为 int64
        sample = torch.full((3999971,), 0, dtype=torch.int64)
        # 将 sample 的最后一个元素设为 1
        sample[-1] = 1
        # 调用 self.common 方法，传入 fn 函数和 sample 张量作为参数
        self.common(fn, (sample,))

    # 根据 GitHub issue，添加修补程序到测试方法 test_multilayer_var
    @config.patch("halide.scheduler_cuda", "Li2018")
    # 如果在 macOS 系统上运行，跳过测试
    @skipCPUIf(IS_MACOS, "fails on macos")
    def test_multilayer_var(self):
        # 定义内部函数 fn，对输入张量 a 计算方差
        def fn(a):
            return torch.var(a)

        # 调用 self.common 方法，分别传入两个不同形状的张量作为参数，并设置误差范围
        self.common(
            fn,
            ((torch.rand((10, 3, 352, 352), dtype=torch.float32),)),
            atol=1e-3,
            rtol=1e-3,
        )
        self.common(
            fn,
            ((torch.rand((14923), dtype=torch.float32),)),
            atol=1e-3,
            rtol=1e-3,
        )

    # 如果在 macOS 系统上运行，跳过测试
    @skipCPUIf(IS_MACOS, "fails on macos")
    # 跳过使用 Halide 的测试
    @skip_if_halide  # accuracy 4.7% off
    def test_multilayer_var_lowp(self):
        # 定义内部函数 fn，对输入张量 a 计算方差
        def fn(a):
            return torch.var(a)

        # 调用 self.common 方法，传入使用 float16 数据类型的两个张量作为参数
        self.common(fn, (torch.rand((16, 16, 352, 352), dtype=torch.float16),))
        self.common(fn, (torch.rand((14923), dtype=torch.float16),))

    # 定义测试方法，测试张量的累积和在指定维度上的计算
    def test_split_cumsum(self):
        # 定义内部函数 fn，对输入张量 a 在最后一个维度上进行累积和的计算
        def fn(a):
            return torch.cumsum(a, -1)

        # 遍历所有数据类型，生成具有特定形状和数据类型的输入张量 inp
        for dtype in get_all_dtypes(
            include_bfloat16=False,
            include_bool=True,
            include_complex=False,
            include_half=False,
        ):
            # 生成张量 inp，使用指定数据类型和设备，在最后一个维度上计算累积和
            inp = make_tensor(10, 3, 352, 352, low=0, dtype=dtype, device=self.device)
            # 调用 self.common 方法，传入 fn 函数和 inp 张量作为参数，并设置误差范围
            self.common(fn, (inp.view(-1),), rtol=1e-5, atol=1e-5, check_lowp=False)
            self.common(fn, (inp.view(10, -1),), rtol=1e-5, atol=1e-5, check_lowp=False)

    # 如果不是 SM80 或更高的 GPU，跳过测试
    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    # 如果在 ROCm 平台上，跳过测试
    @skipCUDAIf(TEST_WITH_ROCM, "Computation not done in float on ROCm")
    # 跳过使用 GPU Halide 的测试
    @skip_if_gpu_halide  # accuracy issue
    def test_split_cumsum_low_prec(self):
        # 如果设备是 CPU，则跳过测试
        if self.device == "cpu":
            raise unittest.SkipTest("ir.Scan nyi on CPU")

        # 定义内部函数 fn，对输入张量 a 的展平版本进行累积和的计算
        def fn(a):
            return torch.cumsum(a.view(-1), 0)

        # 调用 self.common 方法，传入使用 float16 数据类型的张量作为参数
        self.common(
            fn,
            (torch.rand((10, 3, 352, 352), dtype=torch.float16),),
            reference_in_float=True,
            check_lowp=False,
        )

    # 定义测试方法，测试两个张量在展平后的累积和
    def test_consecutive_split_cumsum(self):
        # 定义内部函数 fn，对输入张量 a 和 b 的展平版本进行累积和的计算
        def fn(a, b):
            a = a.view(-1)
            b = b.view(-1)
            return torch.cumsum(a, 0) + torch.cumsum(b, 0)

        # 生成两个具有相同形状和设备的输入张量 a 和 b
        a = make_tensor(10, 3, 352, 352, low=0, dtype=torch.float32, device=self.device)
        b = make_tensor(10, 3, 352, 352, low=0, dtype=torch.float64, device=self.device)
        # 调用 self.common 方法，传入 fn 函数、张量 a 和 b 作为参数，并设置误差范围
        self.common(fn, (a, b), rtol=1e-5, atol=1e-5, check_lowp=False)
    # 定义测试函数 test_split_cumprod，测试 torch.cumprod 在不同数据类型上的累积乘积计算
    def test_split_cumprod(self):
        # 定义内部函数 fn，对输入张量进行累积乘积计算
        def fn(a):
            return torch.cumprod(a, -1)

        # 遍历不同的数据类型进行测试
        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
            # 生成大尺寸输入用于累积乘积计算，设备为 self.device
            inp = _large_cumprod_input(
                (10, 10000), dim=1, dtype=dtype, device=self.device
            )
            # 调用 self.common 方法进行测试，验证结果的绝对误差和相对误差
            self.common(fn, (inp,), atol=1e-5, rtol=1e-4, check_lowp=False)

    # 如果不支持 CUDA 架构 SM80 或更高版本，则跳过测试
    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    # 如果在 ROCm 平台上，则跳过测试
    @skipCUDAIf(TEST_WITH_ROCM, "Computation not done in float on ROCm")
    # 如果存在 GPU Halide 的精度问题，则跳过测试
    @skip_if_gpu_halide  # accuracy issue
    # 定义测试函数 test_split_cumprod_low_prec，测试低精度下的 torch.cumprod 计算
    def test_split_cumprod_low_prec(self):
        # 如果设备是 CPU，则跳过测试
        if self.device == "cpu":
            raise unittest.SkipTest("ir.Scan nyi on CPU")

        # 定义内部函数 fn，对输入张量的视图进行累积乘积计算
        def fn(a):
            return torch.cumprod(a.view(-1), 0)

        # 遍历不同的低精度数据类型进行测试
        for dtype in [torch.float16, torch.bfloat16]:
            # 生成大尺寸输入用于累积乘积计算，设备为 self.device
            inp = _large_cumprod_input(
                (10, 10000), dim=1, dtype=dtype, device=self.device
            )
            # 调用 self.common 方法进行测试，同时要求参考结果为浮点数
            self.common(
                fn,
                (inp,),
                reference_in_float=True,
                check_lowp=False,
            )

    # 定义测试函数 test_consecutive_split_cumprod，测试两个张量的累积乘积之和
    def test_consecutive_split_cumprod(self):
        # 定义内部函数 fn，对两个输入张量进行累积乘积之和计算
        def fn(a, b):
            return torch.cumprod(a, 0) + torch.cumprod(b, 0)

        # 生成大尺寸输入用于累积乘积计算，设备为 self.device
        a = _large_cumprod_input(
            (10000,), dim=0, dtype=torch.float32, device=self.device
        )
        b = _large_cumprod_input(
            (10000,), dim=0, dtype=torch.float64, device=self.device
        )
        # 调用 self.common 方法进行测试，验证结果的绝对误差和相对误差
        self.common(fn, (a, b), atol=1e-5, rtol=1e-5, check_lowp=False)

    # 如果在 ROCm 平台上，则跳过测试，因为不支持 associative_scan
    @skipCUDAIf(TEST_WITH_ROCM, "associative_scan is not supported on ROCm")
    # 如果存在 Halide 的 scan 操作，则跳过测试
    @skip_if_halide  # scan ops
    # 定义测试函数 test_custom_scan_op，测试自定义的扫描操作
    def test_custom_scan_op(self):
        # 如果设备不是 CUDA，则跳过测试，因为 associative_scan 仅在 GPU 上支持
        if self.device != "cuda":
            raise unittest.SkipTest("associative_scan only supported on GPU")

        # 定义函数 sum_combine，用于计算两个数值的和
        def sum_combine(a, b):
            return a + b

        # 导入 associative_scan 函数用于测试
        from torch._higher_order_ops.associative_scan import associative_scan

        # 生成随机张量 a，设备为 self.device
        a = torch.randn(100, 100, device=self.device)
        # 预期结果为张量 a 沿着维度 0 的累加和
        expect = torch.cumsum(a, 0)
        # 实际计算使用 associative_scan 函数
        actual = associative_scan(sum_combine, a, 0)
        # 断言预期结果和实际结果是否相等
        self.assertEqual(expect, actual)

        # 定义函数 logcumsum_combine，用于对数累积和的组合操作
        def logcumsum_combine(a, b):
            min_v = torch.minimum(a, b)
            max_v = torch.maximum(a, b)
            mask = (min_v != max_v) | ~min_v.isinf()
            return torch.where(mask, max_v + (min_v - max_v).exp().log1p(), a)

        # 预期结果为张量 a 沿着维度 0 的对数累积和
        expect = torch.logcumsumexp(a, 0)
        # 实际计算使用 associative_scan 函数
        actual = associative_scan(logcumsum_combine, a, 0)
        # 断言预期结果和实际结果是否相等
        self.assertEqual(expect, actual)

    # 如果存在 Halide 的 scan 操作，则跳过测试
    @skip_if_halide  # scan ops
    def test_custom_scan_op_compiled(self):
        # 检查当前设备是否为 CUDA，如果不是，则跳过测试
        if self.device != "cuda":
            raise unittest.SkipTest("associative_scan only supported on GPU")

        # 导入关联扫描操作模块
        from torch._higher_order_ops.associative_scan import associative_scan

        # 定义一个用于求和的组合函数
        def sum_combine(a, b):
            return a + b

        # 定义一个函数，计算两个张量在指定维度上的差的绝对值的和，使用关联扫描操作
        def fn(a, b, dim):
            diff = (a - b).abs()
            sad = associative_scan(sum_combine, diff, dim)
            return sad.sum(dim)

        # 在指定设备上生成随机张量 a 和 b
        a = torch.randn(100, 100, device=self.device)
        b = torch.randn(100, 100, device=self.device)
        # 调用通用测试函数，测试 fn 函数
        self.common(fn, (a, b, 0))
        # 编译 fn 函数
        cfn = torch.compile(fn)
        # 运行并获取编译后的代码
        _, code = run_and_get_code(cfn, a, b, 0)

        # 检查所有内容是否融合成单个内核
        FileCheck().check_not("run(").check_regex(
            r"triton_.*\.run\(arg[01]_1, arg[12]_1, buf1,"
        ).check_not("run(").run(code[0])

    @skipCUDAIf(TEST_WITH_ROCM, "associative_scan is not supported on ROCm")
    @skip_if_halide  # scan ops
    def test_custom_scan_op_multi_input(self):
        # 检查当前设备是否为 CUDA，如果不是，则跳过测试
        if self.device != "cuda":
            raise unittest.SkipTest("associative_scan only supported on GPU")

        # 定义一个用于 argmax 的组合函数
        def argmax_combine(a, b):
            a_value, a_index = a
            b_value, b_index = b
            mask = (a_value > b_value) | ((a_value == b_value) & (a_index > b_index))
            return (
                torch.where(mask, a_value, b_value),
                torch.where(mask, a_index, b_index),
            )

        # 导入关联扫描操作模块
        from torch._higher_order_ops.associative_scan import associative_scan

        # 在指定设备上生成随机张量 a
        a = torch.randn(100, 100, device=self.device)
        # 计算预期的结果，使用 torch.cummax 函数
        expect = torch.cummax(a, 0)

        # 生成索引张量
        idx = torch.arange(100, device=self.device).view(100, 1).expand(100, 100)
        # 使用关联扫描操作，计算实际结果
        actual = associative_scan(argmax_combine, (a, idx), 0)
        # 断言预期结果和实际结果是否相等
        self.assertEqual(expect, actual)

    def test_embedding_bag_byte_unpack(self):
        # 检查当前设备是否为 CPU，如果不是，则跳过测试并给出相应的理由
        if self.device != "cpu":
            raise unittest.SkipTest(f"No {GPU_TYPE} implementation (it returns empty)")

        # 定义一个函数，调用量化嵌入包字节解包操作
        def fn(a):
            return torch.ops.quantized.embedding_bag_byte_unpack(a)

        # 设置常量 M 和 N
        M, N = 32, 64
        # 生成随机标度和偏移量张量，视图转换为 torch.uint8 类型
        scales = torch.randn(M, 1).view(torch.uint8)
        offsets = torch.randn(M, 1).view(torch.uint8)
        # 生成随机数据张量，数据类型为 torch.uint8
        data = torch.randint(0, 255, (M, N), dtype=torch.uint8)
        # 拼接数据、标度和偏移量，构成 packed 张量
        packed = torch.cat([data, scales, offsets], dim=-1)
        # 调用通用测试函数，测试 fn 函数
        self.common(fn, [packed])
    def test_expanded_reduction(self):
        def fn(x, y):
            z = x * y
            return z.sum((0, 1))

        atol = None  # 设定绝对误差初始值为 None
        rtol = None  # 设定相对误差初始值为 None

        # 默认情况下，在这种情况下，导体会生成非持久性缩减内核。但是当启用多内核时，导体将选择更快的持久性缩减内核和非持久性缩减内核中的较快者。
        # 在这种情况下，导体选择了持久性缩减内核。
        # 持久性缩减内核恰好需要更宽松的公差。
        if config.triton.multi_kernel:
            atol = 1e-5  # 如果启用了多内核，则设置绝对误差公差为 1e-5
            rtol = 1e-5  # 如果启用了多内核，则设置相对误差公差为 1e-5
        self.common(
            fn, (torch.randn(2, 197, 256), torch.randn(2, 1, 256)), atol=atol, rtol=rtol
        )

    def test_min_max_reduction(self):
        def fn(a, b):
            return (
                (a + b).max(),  # 计算 a + b 的最大值
                (a + b).min(),  # 计算 a + b 的最小值
                torch.amax(a + 1, keepdim=True),  # 计算 a + 1 的最大值，并保持维度
                torch.amin(b + 1, keepdim=True),  # 计算 b + 1 的最小值，并保持维度
            )

        dtypes = [torch.float, torch.float16]
        if not (self.device == "cuda" and not SM80OrLater):  # 如果设备不是 CUDA 或者不是 SM80 及之后版本
            dtypes += [torch.bfloat16]  # 则添加 torch.bfloat16 类型到 dtypes 中
        for dtype in dtypes:
            self.common(fn, (torch.randn(8, 8).to(dtype), torch.randn(8, 8).to(dtype)))

    @skip_if_halide  # nan 处理中的错误
    def test_min_max_reduction_nan(self):
        def fn(a):
            return (torch.max(a), torch.min(a))  # 计算张量 a 的最大值和最小值

        t1 = torch.randn(32)
        t1[16] = float("nan")  # 在位置 16 处设置为 NaN
        self.common(fn, (t1,))

    @skip_if_halide  # nan 处理中的错误
    def test_fmin_fmax(self):
        def fn(a, b):
            return (
                torch.fmin(a, b),  # 计算 a 和 b 逐元素的最小值
                torch.fmax(a, b),  # 计算 a 和 b 逐元素的最大值
                torch.fmax(a + 1, torch.tensor(0.0)),  # 计算 a + 1 和 0.0 逐元素的最大值
            )

        self.common(
            fn,
            (
                torch.tensor(
                    [-10.0, 10.0, float("nan"), float("nan"), float("nan"), 3, 4]
                ),  # 第一个张量输入
                torch.tensor(
                    [float("nan"), float("nan"), -10.0, 10.0, float("nan"), 4, 3]
                ),  # 第二个张量输入
            ),
        )

    def test_sum_int(self):
        def fn(x):
            return 2 * x.sum(-1) + x.sum()  # 计算张量 x 沿指定维度的和，并添加总和

        dtypes = torch.bool, torch.uint8, torch.int
        inps = [torch.randint(2, (64,), dtype=dtype) for dtype in dtypes]
        for i in inps:
            self.common(fn, (i,), check_lowp=False)

    def test_sum_dtype(self):
        def fn(x):
            return x * x.sum(-1, dtype=torch.double) + x.sum(dtype=torch.double)  # 使用指定数据类型计算张量 x 沿指定维度的和，并添加总和

        self.common(fn, (torch.ones(32, 32) * 70,))
    # 定义一个测试方法 test_cumsum，用于测试累积和操作
    def test_cumsum(self):
        # 定义一个函数 fn，计算输入张量在0轴和1轴上的累积和
        def fn(x):
            return x.cumsum(0), x.cumsum(1)

        # 测试不同大小的随机张量的累积和，使用 self.common 进行通用测试
        self.common(
            fn, (torch.rand(16, 32),), check_lowp=not is_halide_backend(self.device)
        )
        self.common(
            fn, (torch.rand(20, 30),), check_lowp=not is_halide_backend(self.device)
        )

        # 测试一个较大的随机张量的累积和，使用 self.common 进行通用测试
        self.common(
            fn, (torch.rand(100, 4000),), check_lowp=not is_halide_backend(self.device)
        )

    # 定义一个测试方法 test_cumsum_zero_dim，用于测试零维张量的累积和操作
    def test_cumsum_zero_dim(self):
        # 定义一个函数 fn，计算输入张量在0轴和-1轴上的累积和
        def fn(x):
            return x.cumsum(0), x.cumsum(-1)

        # 创建一个零维的随机张量 a
        a = torch.rand(())
        # 使用 self.common 进行通用测试
        self.common(fn, (a,))

    # 定义一个测试方法 test_cumsum_no_mask，用于测试不带掩码的累积和操作
    def test_cumsum_no_mask(self):
        # 定义一个函数 fn，计算输入张量在-1轴上的累积和
        def fn(x):
            return x.cumsum(-1)

        # 创建一个形状为 (1, 1024) 的随机张量 a
        a = torch.rand((1, 1024))
        # 使用 self.common 进行通用测试，根据条件选择是否检查低精度
        self.common(
            fn, (a,), check_lowp=not (TEST_WITH_ROCM or is_halide_backend(self.device))
        )

        # 创建一个形状为 (1, 8192) 的随机张量 b
        b = torch.rand((1, 8192))
        # 使用 self.common 进行通用测试，根据条件选择是否检查低精度
        self.common(
            fn, (b,), check_lowp=not (TEST_WITH_ROCM or is_halide_backend(self.device))
        )

    # 定义一个测试方法 test_cumprod_zero_dim，用于测试零维张量的累积乘积操作
    def test_cumprod_zero_dim(self):
        # 定义一个函数 fn，计算输入张量在0轴和-1轴上的累积乘积
        def fn(x):
            return x.cumprod(0), x.cumprod(-1)

        # 创建一个零维的随机张量 a
        a = torch.rand(())
        # 使用 self.common 进行通用测试
        self.common(fn, (a,))

    # 定义一个测试方法 test_logcumsumexp，用于测试对数累积和指数操作
    def test_logcumsumexp(self):
        # 定义一个函数 fn，计算输入张量在0轴和1轴上的对数累积和指数
        def fn(x):
            return x.logcumsumexp(0), x.logcumsumexp(1)

        # 测试不同大小的随机张量的对数累积和指数，使用 self.common 进行通用测试
        self.common(
            fn,
            (torch.rand(16, 32),),
            check_lowp=not (TEST_WITH_ROCM or is_halide_backend(self.device)),
        )
        self.common(
            fn,
            (torch.rand(20, 30),),
            check_lowp=not (TEST_WITH_ROCM or is_halide_backend(self.device)),
        )

        # 测试一个较大的随机张量的对数累积和指数，使用 self.common 进行通用测试
        self.common(
            fn,
            (torch.rand(100, 4000),),
            check_lowp=not (TEST_WITH_ROCM or is_halide_backend(self.device)),
        )

    # 定义一个测试方法 test_logcumsumexp_zero_dim，用于测试零维张量的对数累积和指数操作
    def test_logcumsumexp_zero_dim(self):
        # 定义一个函数 fn，计算输入张量在0轴和-1轴上的对数累积和指数
        def fn(x):
            return x.logcumsumexp(0), x.logcumsumexp(-1)

        # 创建一个零维的随机张量 a
        a = torch.rand(())
        # 使用 self.common 进行通用测试
        self.common(fn, (a,))

    # 定义一个测试方法 test_clamp，用于测试张量的 clamp 操作
    def test_clamp(self):
        # 定义一个函数 fn，对输入的两个张量 a 和 b 进行 clamp 操作
        def fn(a, b):
            return (a.clamp(-0.1, 0.1), b.clamp(0), torch.clamp(a + b, max=0))

        # 使用 self.common 进行通用测试，测试两个随机张量的 clamp 操作
        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    # 定义一个测试方法 test_clamp_type_promotion，用于测试张量的 clamp 操作，涉及类型提升
    def test_clamp_type_promotion(self):
        # 定义一个函数 fn，对输入的张量 a 进行 clamp 操作，其中 b 和 c 是预定义的张量
        def fn(a):
            b = torch.tensor(1.0, dtype=torch.double, device=self.device)
            c = torch.full((4,), 2, device=self.device)
            return a.clamp(min=b, max=c)

        # 使用 self.common 进行通用测试，测试一个随机整数张量的 clamp 操作
        self.common(fn, (torch.randint(4, (4,)),))

    # 定义一个测试方法 test_dist，用于测试张量之间的距离计算操作
    def test_dist(self):
        # 定义一个函数 fn，计算输入张量 a 和 b 之间的距离，包括 L2 范数和自定义 p 范数
        def fn(a, b):
            return (
                torch.dist(a, b),
                torch.dist(a, b, p=1.2),
            )

        # 使用 self.common 进行通用测试，测试两个随机张量之间的距离计算
        self.common(fn, (torch.randn(4, 4), torch.randn(4, 4)))

    # 使用 skipCUDAIf 和 skip_if_gpu_halide 装饰器进行条件性跳过测试，要求 SM80 或更高版本支持
    def test_dist_bf16(self):
        # 定义一个函数 fn，用于计算两个张量的 bfloat16 格式的欧几里德距离
        def fn(a, b):
            return torch.dist(a.to(torch.bfloat16), b.to(torch.bfloat16))

        # 调用共同的测试方法，测试 fn 函数的通用性，传入两个随机生成的 4x4 张量作为参数
        self.common(fn, (torch.randn(4, 4), torch.randn(4, 4)))

    def test_arange1(self):
        # 定义一个函数 fn，接受一个张量 x 作为参数，生成两个张量 rng1 和 rng2
        def fn(x):
            # 使用 torch.arange 生成一个 8x8 的浮点型张量 rng1，并根据 x 的设备属性进行设置
            rng1 = torch.arange(8 * 8, dtype=torch.float32, device=x.device).view(8, 8)
            # 使用 torch.arange 生成一个范围在 [10, 18) 的张量 rng2，并根据 x 的设备属性进行设置
            rng2 = torch.arange(10, 18, device=x.device)
            # 计算 tmp，为 x 与 rng1 的乘积
            tmp = x * rng1
            # 返回 tmp 和 tmp 加上 rng2 的元组
            return tmp, tmp + rng2

        # 调用共同的测试方法，测试 fn 函数的通用性，传入一个随机生成的 8x8 张量作为参数
        self.common(fn, (torch.randn(8, 8),))

    def test_arange2(self):
        # 定义一个函数 fn，接受一个张量 x 作为参数，生成一个张量，该张量是 x 加上一个设备相关的 arange 结果
        def fn(x):
            # 使用 torch.arange 生成一个长度为 8 的张量 rng1，并根据 x 的设备属性进行设置
            rng1 = torch.arange(8, device=x.device)
            # 返回 x 加上 rng1 的结果的元组
            return (x + rng1,)

        # 调用共同的测试方法，测试 fn 函数的通用性，传入一个随机生成的 8x8 张量作为参数，并禁用低位精度检查
        self.common(fn, (torch.randint(4, (8, 8)),), check_lowp=False)

    def test_arange3(self):
        # 定义一个函数 fn，接受一个张量 x 作为参数，返回 x 加上 torch.ops.aten.arange.start_step 的结果
        def fn(x):
            return x + torch.ops.aten.arange.start_step(
                0, 53, 4, dtype=torch.int64, device=x.device
            )

        # 调用共同的测试方法，测试 fn 函数的通用性，传入一个长度为 14 的随机生成的张量作为参数
        self.common(fn, (torch.randn(14),))

    def test_arange4(self):
        # 定义一个函数 fn，接受一个张量 x 作为参数，返回 x 减去一个设备相关的 arange 结果
        def fn(x):
            return x - torch.arange(512, -512, -1.0, device=x.device)

        # 调用共同的测试方法，测试 fn 函数的通用性，传入一个长度为 1024 的随机生成的张量作为参数
        self.common(fn, (torch.randn(1024),))

    def test_arange5(self):
        # 定义一个函数 fn，接受步长和设备作为参数，返回一个设备相关的 arange 结果
        def fn(step, device):
            return torch.arange(512, -512, step, device=device)

        # 使用 torch._dynamo.optimize() 优化 fn 函数，并进行断言检查结果是否一致
        compiled_fn = torch._dynamo.optimize()(fn)

        # 使用 assertEqual 检查不同步长情况下的结果是否一致
        for step in (-1, -1.0):
            expect = fn(step, self.device)
            actual = compiled_fn(step, self.device)
            self.assertEqual(expect, actual)
        self.assertEqual(expect, actual)

    def test_arange6(self):
        # 定义一个函数 fn，接受一个张量 x 作为参数，返回一个设备相关的 arange 结果
        def fn(x):
            return torch.arange(0.1, 8.0001, 1, dtype=x.dtype, device=x.device)

        # 使用 functools.partial 创建 make_arg 函数，用于生成特定设备的张量
        make_arg = functools.partial(
            make_tensor, device=self.device, requires_grad=False
        )
        # 调用共同的测试方法，测试 fn 函数的通用性，传入两个不同 dtype 的随机生成的张量作为参数
        self.common(fn, (make_arg(1, dtype=torch.float32),))
        self.common(fn, (make_arg(1, dtype=torch.int64),))

    def test_linspace1(self):
        # 定义一个函数 fn，接受一个张量 x 作为参数，返回 torch.linspace 的结果加上 x
        def fn(x):
            return torch.linspace(0.125, 0.875, 7, device=x.device) + x

        # 调用共同的测试方法，测试 fn 函数的通用性，传入一个随机生成的 1x7 张量作为参数
        self.common(fn, (torch.randn(1, 7),))

    def test_linspace2(self):
        # 定义一个函数 fn，接受一个张量 x 作为参数，返回 torch.linspace 的结果加上 x
        def fn(x):
            return torch.linspace(0, 2, 1, device=x.device) + x

        # 调用共同的测试方法，测试 fn 函数的通用性，传入一个随机生成的 1x1 张量作为参数
        self.common(fn, (torch.randn(1, 1),))

    def test_linspace3(self):
        # 定义一个函数 fn，接受一个张量 x 作为参数，返回一个空张量
        def fn(x):
            return torch.linspace(0, 2, 0, device=x.device)

        # 调用共同的测试方法，测试 fn 函数的通用性，传入一个空张量作为参数
        self.common(fn, (torch.Tensor([]),))

    def test_tensor1(self):
        # 定义一个函数 fn，接受一个张量 x 作为参数，返回 x 加上一个张量和一个标量的结果
        def fn(x):
            return torch.tensor([1], device=x.device) + x, torch.tensor(
                5, device=x.device
            )

        # 调用共同的测试方法，测试 fn 函数的通用性，传入一个随机生成的长度为 10 的张量作为参数
        self.common(fn, (torch.randn(10),))

    def test_tensor2(self):
        # 定义一个函数 fn，接受一个张量 x 作为参数，返回 x 加上一个设备相关的张量的结果
        def fn(x):
            return torch.tensor(list(range(2, 40, 2)), device=x.device) + x

        # 调用共同的测试方法，测试 fn 函数的通用性，传入一个随机生成的长度为 1 的张量作为参数
        self.common(fn, (torch.randn(1),))
    # 定义一个测试函数 test_tensor3，用于测试 fn 函数在给定参数上的输出
    def test_tensor3(self):
        # 定义函数 fn，接受一个参数 x，并返回包含四个张量的元组
        def fn(x):
            return (
                torch.tensor([], device=x.device),  # 返回一个空张量，设备与输入张量 x 相同
                torch.tensor([1, 2], device=x.device) + 1,  # 返回一个加一后的张量
                torch.tensor([1, 2, 3], device=x.device) + 2,  # 返回一个加二后的张量
                torch.tensor([1, 2, 3, 4], device=x.device) + x,  # 返回一个与 x 相加后的张量
            )

        # 调用 self.common 方法，测试 fn 函数在给定参数 [torch.randn(4)] 上的输出

    # 定义一个测试函数 test_views1，用于测试 fn1 和 fn2 函数在不同视图尺寸上的输出
    def test_views1(self):
        # 定义函数 fn1，接受两个参数 x 和 y，返回一个包含 x.view(size2) + y 的元组
        def fn1(x, y):
            return (x.view(size2) + y,)

        # 定义函数 fn2，接受两个参数 x 和 y，返回一个包含 (x + 1).view(size2) + y 的元组
        def fn2(x, y):
            return ((x + 1).view(size2) + y,)

        # 定义多个视图尺寸的列表 views
        views = [
            ([5 * 7], [5, 7]),
            ([2 * 3 * 4 * 5 * 6 * 7], [2, 3, 4, 5, 6, 7]),
            ([2 * 3, 4, 5, 6 * 7], [2, 3, 4, 5, 6, 7]),
            ([10 * 5, 20], [10, 5, 20]),
            ([1, 10, 1], [10]),
            ([10, 1, 10, 1, 10], [10, 100]),
            ([2, 2, 2, 2], [4, 4]),
        ]

        # 遍历 views 列表中的每一对尺寸 size1 和 size2
        for size1, size2 in views:
            # 调用 self.common 方法，测试 fn1 函数在给定参数 (torch.randn(size1), torch.randn(size2)) 上的输出
            self.common(fn1, (torch.randn(size1), torch.randn(size2)))
            # 调用 self.common 方法，测试 fn2 函数在给定参数 (torch.randn(size1), torch.randn(size2)) 上的输出
            self.common(fn2, (torch.randn(size1), torch.randn(size2)))

        # 再次遍历 views 列表，但将 size2 和 size1 交换位置
        for size2, size1 in views:
            # 调用 self.common 方法，测试 fn1 函数在给定参数 (torch.randn(size1), torch.randn(size2)) 上的输出
            self.common(fn1, (torch.randn(size1), torch.randn(size2)))
            # 调用 self.common 方法，测试 fn2 函数在给定参数 (torch.randn(size1), torch.randn(size2)) 上的输出
            self.common(fn2, (torch.randn(size1), torch.randn(size2)))

    # 定义一个测试函数 test_views2，用于测试 fn1 和 fn2 函数在不同视图尺寸上的输出
    def test_views2(self):
        # 定义函数 fn1，接受一个参数 x，并返回一个包含 x.view(size2) + 1 的元组
        def fn1(x):
            return (x.view(size2) + 1,)

        # 定义函数 fn2，接受一个参数 x，并返回一个包含 (x * 2).view(size2) + 1 的元组
        def fn2(x):
            return ((x * 2).view(size2) + 1,)

        # 遍历不同的视图尺寸列表，对应每个 size1 和 size2 的组合
        for size1, size2 in [
            ([2, 2, 2, 2], [4, -1]),
            ([10, 1, 10, 1, 10], [-1, 100]),
            ([10 * 5, 20], [10, -1, 20]),
        ]:
            # 调用 self.common 方法，测试 fn1 函数在给定参数 (torch.randn(size1),) 上的输出
            self.common(fn1, (torch.randn(size1),))
            # 调用 self.common 方法，测试 fn2 函数在给定参数 (torch.randn(size1),) 上的输出
            self.common(fn2, (torch.randn(size1),))

    # 定义一个测试函数 test_views3，用于测试 forward 函数在给定参数上的输出
    def test_views3(self):
        # 定义函数 forward，接受两个参数 arg1 和 arg2，进行索引操作和视图操作，并返回结果
        def forward(arg1, arg2):
            index = torch.ops.aten.index(arg1, [arg2])
            view_1 = torch.ops.aten.view(index, [1, 2232, 64])
            view_2 = torch.ops.aten.view(view_1, [1, 12, 62, 192])
            return view_2

        # 调用 self.common 方法，测试 forward 函数在给定参数上的输出

    # 定义一个测试函数 test_views4，用于测试 forward 函数在给定参数上的输出
    def test_views4(self):
        # 定义函数 forward，接受两个参数 arg1 和 arg2，进行索引选择和多次视图操作，并返回结果
        def forward(arg1, arg2):
            arg1 = arg1.index_select(0, arg2)
            arg1 = torch.ops.aten.view(arg1, [2, 3, 4, 5, 5])
            arg1 = torch.ops.aten.view(arg1, [2, 3, 2, 10, -1])
            return arg1

        # 调用 self.common 方法，测试 forward 函数在给定参数上的输出

    # 定义一个测试函数 test_views5，用于测试 forward 函数在给定参数上的输出
    def test_views5(self):
        # 定义函数 forward，接受一个参数 x，进行切片和视图操作，并返回结果
        def forward(x):
            y = x[:, 4:]  # 切片操作，保留第二个维度从索引 4 开始的所有元素
            return y.view(len(y), -1, 4)  # 对结果进行视图操作，将第一个维度设置为 y 的长度，第二个维度自动计算，第三个维度设置为 4

        # 调用 self.common 方法，测试 forward 函数在给定参数上的输出
    def test_views6(self):
        # 定义一个测试函数，对输入张量进行一系列操作后返回结果张量
        def forward(x):
            # 使用 torch.ops.aten.relu 对输入张量 x 执行 ReLU 激活函数
            x = torch.ops.aten.relu(x)
            # 使用 torch.ops.aten.slice 对张量 x 进行切片操作，保留所有维度
            s = torch.ops.aten.slice(x, 0, 0, 9223372036854775807)
            # 继续在切片结果上进行切片操作，保留所有维度
            s = torch.ops.aten.slice(s, 1, 0, 9223372036854775807)
            # 在切片结果上进行第三次切片操作，但只保留前三个维度
            s = torch.ops.aten.slice(s, 3, 0, 0)
            # 使用 torch.ops.aten.view 对张量 s 进行形状重塑，指定新形状为 [4, 2, -1]
            y = torch.ops.aten.view(s, [4, 2, -1])
            # 返回重塑后的张量 y
            return y

        # 调用通用测试方法 common，测试 forward 函数的输出
        self.common(
            forward,
            (torch.randn(4, 2, 4, 4),),  # 传入随机生成的 4x2x4x4 大小的张量作为输入
        )

    def test_views7(self):
        # 定义一个测试函数，对两个输入张量执行加一和类型转换后返回结果
        def forward(x, y):
            # 将张量 x 中的所有元素加一，并转换为 float32 类型
            x = (x + 1).to(torch.float32)
            # 将张量 y 中的所有元素加一，并转换为 int32 类型
            y = (y + 1).to(torch.int32)
            # 分别对 x 和 y 执行 view 操作，转换为指定的类型
            return x.view(torch.int32), y.view(torch.float32)

        # 调用通用测试方法 common，测试 forward 函数的输出
        self.common(
            forward,
            (
                torch.rand(2, 3, dtype=torch.float32),  # 传入形状为 2x3 的随机张量，数据类型为 float32
                torch.randint(10, (2, 3), dtype=torch.int32),  # 传入形状为 2x3 的随机整数张量，数据类型为 int32
            ),
        )

    def test_relu(self):
        # 定义一个测试函数，对两个输入张量分别执行 ReLU 操作并返回结果
        def fn(a, b):
            return (torch.relu(a), torch.relu(a + b) / 10)

        # 调用通用测试方法 common，测试 fn 函数的输出
        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_exp(self):
        # 定义一个测试函数，对两个输入张量分别执行指数函数操作并返回结果
        def fn(a, b):
            return (torch.exp(a), torch.exp(a + b))

        # 调用通用测试方法 common，测试 fn 函数的输出
        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_exp2(self):
        # 定义一个测试函数，对两个输入张量分别执行 2 的指数幂和差的绝对值的 2 的幂次方运算，并返回结果
        def fn(a, b):
            return (torch.exp2(a), torch.exp2(a + b), torch.pow(2, -torch.abs(a - b)))

        # 调用通用测试方法 common，测试 fn 函数的输出
        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_sigmoid(self):
        # 定义一个测试函数，对两个输入张量分别执行 Sigmoid 函数操作并返回结果
        def fn(a, b):
            return (torch.sigmoid(a), torch.sigmoid(a + b))

        # 调用通用测试方法 common，测试 fn 函数的输出
        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_round(self):
        # 定义一个测试函数，对两个输入张量分别执行四舍五入操作并返回结果
        def fn(a, b):
            return torch.round(a), torch.round(b + 1), torch.round(a, decimals=2)

        # 设置随机种子，确保测试结果的可重复性
        torch.manual_seed(0)

        # 调用通用测试方法 common，测试 fn 函数的输出
        self.common(fn, (torch.randn(8, 8) * 100, torch.randn(8, 8) * 10))

    def test_round_correctness(self):
        # 如果设备为 CUDA，则跳过测试
        if self.device == "cuda":
            raise unittest.SkipTest("need to debug tl.libdevice on A100/V100")

        # 定义一个测试函数，对输入张量执行四舍五入操作并返回结果
        def fn(a):
            return torch.round(a)

        # 调用通用测试方法 common，测试 fn 函数的输出
        self.common(
            fn,
            [torch.arange(-10, 10, 0.1, dtype=torch.float64)],  # 传入从 -10 到 10 的间隔为 0.1 的浮点数张量
            check_lowp=False,
        )

    def test_builtins_round(self):
        # 定义一个函数，对输入张量进行切片操作并返回结果
        def fn(x, i):
            return x[: round(i / 2 + 1)] + round(i / 2)

        # 使用 torch.compile 函数编译 fn 函数，启用全图模式和动态模式
        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        # 创建设备为 self.device 的零张量 x
        x = torch.zeros(5, dtype=torch.int, device=self.device)
        with torch.no_grad():
            # 对于 i 从 1 到 5 的范围
            for i in range(1, 6):
                # 断言编译后的 cfn 函数和原始 fn 函数的输出相等
                self.assertEqual(cfn(x, i), fn(x, i))
    # 测试用例：测试对浮点数进行四舍五入操作，指定正数位数
    def test_builtins_round_float_ndigits_pos(self):
        # 定义内部函数 fn，将 x 加上 i/2 * 123.4567 进行四舍五入，保留一位小数
        def fn(x, i):
            return x + round(i / 2 * 123.4567, 1)

        # 使用 Torch 的编译器编译 fn，允许动态图和完整图模式
        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        # 创建设备上的全零张量 x
        x = torch.zeros(2, device=self.device)
        # 设置 i 的值为 2
        i = 2

        # 在无需梯度更新的上下文中，验证 cfn 的输出与原始 fn 的输出是否相等
        with torch.no_grad():
            self.assertEqual(cfn(x, i), fn(x, i))

    # 测试用例：测试对浮点数进行四舍五入操作，指定零位小数
    def test_builtins_round_float_ndigits_zero(self):
        # 定义内部函数 fn，将 x 加上 i/2 * 123.4567 进行四舍五入，保留零位小数（即取整）
        def fn(x, i):
            return x + round(i / 2 * 123.4567, 0)

        # 使用 Torch 的编译器编译 fn，允许动态图和完整图模式
        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        # 创建设备上的全零张量 x
        x = torch.zeros(2, device=self.device)
        # 设置 i 的值为 2
        i = 2

        # 在无需梯度更新的上下文中，验证 cfn 的输出与原始 fn 的输出是否相等
        with torch.no_grad():
            self.assertEqual(cfn(x, i), fn(x, i))

    # 测试用例：测试对浮点数进行四舍五入操作，指定负数位数
    def test_builtins_round_float_ndigits_neg(self):
        # 定义内部函数 fn，将 x 加上 i/2 * 123.4567 进行四舍五入，保留负一位小数（即十位数）
        def fn(x, i):
            return x + round(i / 2 * 123.4567, -1)

        # 使用 Torch 的编译器编译 fn，允许动态图和完整图模式
        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        # 创建设备上的全零张量 x
        x = torch.zeros(2, device=self.device)
        # 设置 i 的值为 2
        i = 2

        # 在无需梯度更新的上下文中，验证 cfn 的输出与原始 fn 的输出是否相等
        with torch.no_grad():
            self.assertEqual(cfn(x, i), fn(x, i))

    # 测试用例：测试对整数进行四舍五入操作，指定正数位数
    def test_builtins_round_int_ndigits_pos(self):
        # 定义内部函数 fn，将 x 加上 i 进行四舍五入，保留一位小数
        def fn(x, i):
            return x + round(i, 1)

        # 使用 Torch 的编译器编译 fn，允许动态图和完整图模式
        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        # 创建设备上的全零张量 x
        x = torch.zeros(2, device=self.device)
        # 设置 i 的值为 123
        i = 123

        # 在无需梯度更新的上下文中，验证 cfn 的输出与原始 fn 的输出是否相等
        with torch.no_grad():
            self.assertEqual(cfn(x, i), fn(x, i))

    # 测试用例：测试对整数进行四舍五入操作，指定零位小数
    def test_builtins_round_int_ndigits_zero(self):
        # 定义内部函数 fn，将 x 加上 i 进行四舍五入，保留零位小数（即取整）
        def fn(x, i):
            return x + round(i, 0)

        # 使用 Torch 的编译器编译 fn，允许动态图和完整图模式
        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        # 创建设备上的全零张量 x
        x = torch.zeros(2, device=self.device)
        # 设置 i 的值为 123
        i = 123

        # 在无需梯度更新的上下文中，验证 cfn 的输出与原始 fn 的输出是否相等
        with torch.no_grad():
            self.assertEqual(cfn(x, i), fn(x, i))

    # 测试用例：测试对张量应用神经网络中的 SILU（sigmoid-weighted linear unit）函数
    def test_silu(self):
        # 定义内部函数 fn，应用 Torch 的 SILU 函数到张量 a 上
        def fn(a):
            return (torch.nn.functional.silu(a),)

        # 使用 self.common 方法进行测试
        self.common(fn, (torch.randn(8, 8),))

    @skip_if_halide  # 当 Halide 存在错误的 NaN 处理时跳过测试
    # 测试用例：测试将张量中的 NaN 替换为指定值，同时处理正负无穷大
    def test_nan_to_num(self):
        # 定义内部函数 fn，使用 torch.nan_to_num 处理输入张量 a 中的 NaN、正无穷大和负无穷大
        def fn(a):
            return (
                torch.nan_to_num(a),
                torch.nan_to_num(a, nan=3.0),
                torch.nan_to_num(a, nan=None),
                torch.nan_to_num(a, posinf=4.0),
                torch.nan_to_num(a, neginf=5.0),
                torch.nan_to_num(a, nan=3.0, posinf=4.0, neginf=5.0),
            )

        # 使用 self.common 方法进行测试，传入特定的张量和选项
        self.common(
            fn,
            (torch.tensor((float("nan"), float("inf"), float("-inf"), 1.0)),),
            check_lowp=False,  # 需要更复杂的测试以匹配浮点数和半精度浮点数的最大值
        )

    # 测试用例：测试对张量应用神经网络中的 one_hot 编码，然后加 1
    def test_one_hot(self):
        # 定义内部函数 fn，对输入张量 a 进行 one_hot 编码（8 类），然后加 1
        def fn(a):
            return torch.nn.functional.one_hot(a, 8) + 1

        # 使用 self.common 方法进行测试，传入特定的张量和选项
        self.common(
            fn,
            (torch.arange(100).view(4, 5, 5) % 8,),
            check_lowp=False,
        )
    # 定义测试函数 test_div1，用于测试除法操作
    def test_div1(self):
        # 定义内部函数 fn，接收两个参数 a 和 b，返回多种除法运算的结果
        def fn(a, b):
            return (
                # 调用 aten.div 函数，对 a 和 b 进行除法运算，不进行舍入
                aten.div(a, b, rounding_mode=None),
                # 调用 aten.div 函数，对 a 和 b 进行除法运算，向下取整
                aten.div(a, b, rounding_mode="floor"),
                # 调用 aten.div 函数，对 a 和 b 进行除法运算，向零取整
                aten.div(a, b, rounding_mode="trunc"),
                # 使用 Python 原生的除法运算符进行除法操作
                a / b,
                # 使用 Python 原生的整除运算符进行整除操作
                a // b,
            )
    
        # 调用外部类中的 common 方法，传入 fn 函数和两个随机生成的 8x8 的 Tensor 作为参数
        self.common(fn, (torch.randn(8, 8) * 100, torch.randn(8, 8) * 100))
    
    # 定义测试函数 test_div2，用于测试除法操作
    def test_div2(self):
        # 定义内部函数 fn，接收两个参数 a 和 b，返回多种除法运算的结果
        def fn(a, b):
            return (
                # 调用 aten.div 函数，对 a 和 b 进行除法运算，不进行舍入
                aten.div(a, b, rounding_mode=None),
                # 调用 aten.div 函数，对 a 和 b 进行除法运算，向下取整
                aten.div(a, b, rounding_mode="floor"),
                # 调用 aten.div 函数，对 a 和 b 进行除法运算，向零取整
                aten.div(a, b, rounding_mode="trunc"),
                # 使用 Python 原生的除法运算符进行除法操作
                a / b,
                # 使用 Python 原生的整除运算符进行整除操作
                a // b,
            )
    
        # 调用外部类中的 common 方法，传入 fn 函数和两个 Tensor 作为参数，
        # 第一个 Tensor 是由 torch.randint 生成的在 [-100, 100) 范围内的随机整数，形状为 8x8；
        # 第二个 Tensor 是由 torch.randn 生成的 8x8 的随机 Tensor，乘以 100 后的结果
        self.common(fn, (torch.randint(-100, 100, [8, 8]), 100 * torch.randn(8, 8)))
    
    # 定义测试函数 test_div3，用于测试除法操作
    def test_div3(self):
        # 定义内部函数 fn，接收两个参数 a 和 b，返回多种除法运算的结果
        def fn(a, b):
            return (
                # 调用 aten.div 函数，对 a 和 b 进行除法运算，不进行舍入
                aten.div(a, b, rounding_mode=None),
                # 调用 aten.div 函数，对 a 和 b 进行除法运算，向下取整
                aten.div(a, b, rounding_mode="floor"),
                # 调用 aten.div 函数，对 a 和 b 进行除法运算，向零取整
                aten.div(a, b, rounding_mode="trunc"),
                # 使用 Python 原生的除法运算符进行除法操作
                a / b,
                # 使用 Python 原生的整除运算符进行整除操作
                a // b,
            )
    
        # 创建一个 8x8 的随机整数 Tensor a，范围为 [1, 100)
        a = torch.randint(1, 100, [8, 8])
        # 调用外部类中的 common 方法，传入 fn 函数和 (a*2, a) 作为参数
        self.common(fn, (a * 2, a))
    
    # 定义测试函数 test_div4，用于测试除法操作
    def test_div4(self):
        # 定义内部函数 fn，接收两个参数 a 和 b，返回多种除法运算的结果
        def fn(a, b):
            return (
                # 调用 aten.div 函数，对 a 和 b 进行除法运算，不进行舍入
                aten.div(a, b, rounding_mode=None),
                # 调用 aten.div 函数，对 a 和 b 进行除法运算，向下取整
                aten.div(a, b, rounding_mode="floor"),
                # 调用 aten.div 函数，对 a 和 b 进行除法运算，向零取整
                aten.div(a, b, rounding_mode="trunc"),
                # 使用 Python 原生的除法运算符进行除法操作
                a / b,
                # 使用 Python 原生的整除运算符进行整除操作
                a // b,
            )
    
        # 调用外部类中的 common 方法，传入 fn 函数和两个 Tensor 作为参数，
        # 第一个 Tensor 是由 torch.randint 生成的在 [-100, 0) 范围内的随机整数，形状为 8x8；
        # 第二个 Tensor 是由 torch.randint 生成的在 [1, 10) 范围内的随机整数，形状为 8x8
        self.common(
            fn,
            (torch.randint(-100, 0, [8, 8]), torch.randint(1, 10, [8, 8])),
        )
    
    # 定义测试函数 test_div5，用于测试除法操作
    def test_div5(self):
        # 定义内部函数 fn，接收两个参数 a 和 b，返回多种除法运算的结果
        def fn(a, b):
            return (
                # 调用 aten.div 函数，对 a 和 b 进行除法运算，不进行舍入
                aten.div(a, b, rounding_mode=None),
                # 调用 aten.div 函数，对 a 和 b 进行除法运算，向下取整
                aten.div(a, b, rounding_mode="floor"),
                # 调用 aten.div 函数，对 a 和 b 进行除法运算，向零取整
                aten.div(a, b, rounding_mode="trunc"),
                # 使用 Python 原生的除法运算符进行除法操作
                a / b,
                # 使用 Python 原生的整除运算符进行整除操作
                a // b,
            )
    
        # 调用外部类中的 common 方法，传入 fn 函数和一个 Tensor 和一个标量作为参数，
        # Tensor 是由 torch.randint 生成的在 [-100, 0) 范围内的随机整数，形状为 8x8；
        # 标量值为 16
        self.common(fn, (torch.randint(-100, 0, [8, 8]), 16))
    
    # 定义测试函数 test_div6，用于测试除法操作
    def test_div6(self):
        # 定义内部函数 fn，接收两个参数 a 和 b，返回多种除法运算的结果
        def fn(a, b):
            return (
                # 调用 aten.div 函数，对 a 和 b 进行除法运算，不进行舍入
                aten.div(a, b, rounding_mode=None),
                # 调用 aten.div 函数，对 a 和 b 进行除法运算，向下取整
                aten.div(a, b, rounding_mode="floor"),
                # 调用 aten.div 函数，对 a 和 b 进行除法运算，向零取整
                aten.div(a, b, rounding_mode="trunc"),
                # 使用 Python 原生的除法运算符进行除法操作
                a / b,
                # 使用 Python 原生的整除运算符进行整除操作
                a // b,
            )
    
        # 调用外部类中的 common 方法，传入 fn 函数和一个 8x8 的全为 True 的布尔型 Tensor，
        # 和一个由 torch.randint 生成的在 [-100, -1) 范围内的随机
    # 定义一个名为 test_div8 的测试方法，用于测试 div 函数
    def test_div8(self):
        # 定义一个内部函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 返回一个包含多个表达式结果的元组
            return (
                # 调用 aten.div 函数，计算 a 除以 b 的结果，不进行舍入
                aten.div(a, b, rounding_mode=None),
                # 计算 a 乘以 0.5 后除以 b 的结果，不进行舍入
                aten.div(a * 0.5, b, rounding_mode=None),
                # 计算 a 除以 b*1.0 的结果，不进行舍入
                aten.div(a, b * 1.0, rounding_mode=None),
                # 计算 a 除以 b 的结果，向下舍入到最近的整数
                aten.div(a, b, rounding_mode="floor"),
                # 计算 a 除以 b 的结果，向零舍入到最近的整数
                aten.div(a, b, rounding_mode="trunc"),
                # 使用 Python 自带的除法运算符计算 a 除以 b 的浮点数结果
                a / b,
                # 使用 Python 自带的整数除法运算符计算 a 除以 b 的整数结果
                a // b,
            )

        # 调用 self.common 方法，传入 fn 函数和参数 (1024, 100)，执行测试
        self.common(fn, (1024, 100))

    # 定义一个名为 test_div9 的测试方法，用于测试 div 函数
    def test_div9(self):
        # 定义一个内部函数 fn，接受一个参数 x
        def fn(x):
            # 返回一个包含多个表达式结果的元组
            return (
                # 调用 torch.div 函数，计算常数 42 除以 x 的结果
                torch.div(42, x),
                # 调用 aten.true_divide 函数，计算常数 42 除以 x 的结果
                aten.true_divide(42, x),
                # 调用 aten.div.Tensor 方法，计算常数 42 除以 x 的结果
                aten.div.Tensor(42, x)
            )

        # 调用 self.common 方法，传入 fn 函数和参数 (torch.randn(8),)，执行测试
        self.common(fn, (torch.randn(8),))

    # 定义一个名为 test_div_zero_dim 的测试方法，用于测试 div 函数
    def test_div_zero_dim(self):
        # 定义一个内部函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 返回一个包含多个表达式结果的元组
            return (
                # 调用 aten.div 函数，计算 a 除以 b 的结果，不进行舍入
                aten.div(a, b, rounding_mode=None),
                # 调用 aten.div 函数，计算 a 除以 b 的结果，向下舍入到最近的整数
                aten.div(a, b, rounding_mode="floor"),
                # 调用 aten.div 函数，计算 a 除以 b 的结果，向零舍入到最近的整数
                aten.div(a, b, rounding_mode="trunc"),
                # 使用 Python 自带的除法运算符计算 a 除以 b 的浮点数结果
                a / b,
                # 使用 Python 自带的整数除法运算符计算 a 除以 b 的整数结果
                a // b,
            )

        # 遍历数据类型 torch.float32 和 torch.int64
        for dtype in (torch.float32, torch.int64):
            # 调用 self.common 方法，传入 fn 函数和两个张量作为参数，执行测试
            self.common(
                fn,
                (
                    # 创建一个形状为 (10,) 的张量，数据类型为 dtype，存储于指定设备上
                    make_tensor(10, device=self.device, dtype=dtype),
                    # 创建一个形状为 () 的张量，数据类型为 dtype，不包含零值，存储于指定设备上
                    make_tensor((), device=self.device, dtype=dtype, exclude_zero=True),
                ),
            )
            # 调用 self.common 方法，传入 fn 函数和两个张量作为参数，执行测试
            self.common(
                fn,
                (
                    # 创建一个形状为 () 的张量，数据类型为 dtype，存储于指定设备上
                    make_tensor((), device=self.device, dtype=dtype),
                    # 创建一个形状为 (10,) 的张量，数据类型为 dtype，不包含零值，存储于指定设备上
                    make_tensor(10, device=self.device, dtype=dtype, exclude_zero=True),
                ),
            )

    # 定义一个名为 test_div_prim 的测试方法，用于测试 div 函数
    def test_div_prim(self):
        # 定义一个内部函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 返回一个包含单个表达式结果的元组
            return (
                # 调用 torch.ops.prims.div 方法，计算 a 除以 b 的结果
                torch.ops.prims.div(a, b),
            )

        # 遍历数据类型 torch.float32 和 torch.int64
        for dtype in (torch.float32, torch.int64):
            # 调用 self.common 方法，传入 fn 函数和两个张量作为参数，执行测试
            self.common(
                fn,
                (
                    # 创建一个形状为 (100,) 的张量，数据类型为 dtype，存储于指定设备上
                    make_tensor(100, device=self.device, dtype=dtype),
                    # 创建一个形状为 (100,) 的张量，数据类型为 dtype，不包含零值，存储于指定设备上
                    make_tensor(
                        100, device=self.device, dtype=dtype, exclude_zero=True
                    ),
                ),
            )

    # 定义一个名为 test_floordiv 的测试方法，用于测试 floordiv 函数
    def test_floordiv(self):
        # 定义一个内部函数 fn_floor_input，接受两个参数 a 和 i
        def fn_floor_input(a, i):
            # 计算 (i * 1.234) 除以 8.234 的结果，向下舍入到最近的整数，并赋值给 n
            n = (i * 1.234) // 8.234
            # 返回 a 加上 n 的结果
            return a + n

        # 调用 self.common 方法，传入 fn_floor_input 函数和参数，执行测试
        self.common(
            fn_floor_input,
            # 创建一个形状为 (10,) 的张量，数据类型为 torch.float32，存储于指定设备上
            (make_tensor(10, device=self.device, dtype=torch.float32), 33),
        )

        # 定义一个内部函数 fn_int_input，接受两个参数 a 和 i
        def fn_int_input(a, i):
            # 计算 i 除以 8 的整数部分，并赋值给 n
            n = i // 8
            # 返回 a 加上 n 的结果
            return a + n

        # 调用 self.common 方法，传入 fn_int_input 函数和参数，执行测试
        self.common(
            fn_int_input, 
            # 创建一个形状为 (10,) 的张量，数据类型为 torch.float32，存储于指定设备上
            (make_tensor(10, device=self.device, dtype=torch.float32), 33)
        )
    def test_div_precision(self):
        # Reproducer for https://github.com/pytorch/pytorch/issues/101039

        # 定义一个内部函数 forward，接受两个参数 x 和 y
        def forward(x, y):
            # 计算 x 除以 y，并返回结果的 softmax，在最后一个维度上进行计算
            z = x.div(y)
            return F.softmax(z, dim=-1)

        # 生成一个大小为 (1, 10, 40) 的随机张量作为 query
        query = torch.randn(1, 10, 40)
        # 生成一个大小为 (1, 2, 40) 的随机张量作为 key
        key = torch.randn(1, 2, 40)
        # 使用 torch.matmul 计算 query 和 key 转置的乘积，结果为 (1, 10, 2) 的张量 x
        x = torch.matmul(query, key.transpose(-2, -1))
        # 调用 self.common 方法，将 forward 函数和参数 (x, 1e-6) 传递给它
        self.common(forward, (x, 1e-6))

        # 定义一个特定的大小为 (4, 4, 4, 4) 的张量 x，包含具体数值
        x = torch.tensor(
            [
                [
                    [
                        [-16.1649, 5.6846, -5.1022, -9.1134],
                        [-11.5552, -2.2615, -12.8913, 10.6538],
                        [-7.1666, -5.3333, 2.0776, -9.7984],
                        [7.4469, -2.3948, 2.7371, 0.9201],
                    ],
                    [
                        [-8.0361, -16.3771, 22.7741, 4.4685],
                        [20.8047, -0.7771, -2.4355, -2.2299],
                        [3.8343, -2.0914, -2.4077, 2.2740],
                        [-15.8663, -2.7015, -12.5241, -3.0040],
                    ],
                    [
                        [-2.5139, 14.4393, -3.7186, 1.2255],
                        [5.6742, 14.1842, -8.5976, 16.8366],
                        [-9.7358, -3.0279, 11.8164, -4.0787],
                        [-9.0621, 8.2580, 29.9486, -2.4107],
                    ],
                    [
                        [7.3622, 12.5640, -20.5592, 13.6237],
                        [-11.5640, 0.8832, 16.7275, -2.5009],
                        [-2.0953, -12.2276, -26.2633, 4.5268],
                        [15.3329, -11.7492, 6.5650, -9.2483],
                    ],
                ],
                [
                    [
                        [7.9980, -4.9369, 3.1508, 5.2994],
                        [3.8052, 3.9514, 8.4987, -10.5045],
                        [-2.6827, -4.0010, -4.0611, 6.4091],
                        [-19.0318, 6.4073, 2.8923, 8.0250],
                    ],
                    [
                        [7.1650, -3.4585, 5.7720, -5.0305],
                        [-0.9765, -3.0086, 11.7114, 8.0555],
                        [-3.1027, -3.5514, 9.6182, -8.8526],
                        [-9.2348, -6.0239, 6.2528, -6.7221],
                    ],
                    [
                        [11.5936, 22.4139, -0.4089, -4.9889],
                        [14.8217, -2.3426, -17.6189, 3.7427],
                        [1.9546, -13.0902, 8.6293, -7.2457],
                        [-7.6900, -4.5796, 9.6332, -10.2631],
                    ],
                    [
                        [0.8027, -1.0955, 14.8404, -0.2673],
                        [3.2143, -1.8640, -2.9678, 6.5165],
                        [-3.9865, 6.5230, 6.3019, -0.4247],
                        [8.3185, -13.5076, 27.0986, -1.6792],
                    ],
                ],
            ]
        )
        # 计算 x 和其自身的矩阵乘积，结果替换原来的 x
        x = torch.matmul(x, x)
        # 定义一个大小为 (4, 1, 1) 的张量 y，包含具体数值
        y = torch.tensor([[[0.6331]], [[1.6358]], [[-0.3459]], [[1.0196]]])
        # 调用 self.common 方法，将 forward 函数和参数 (x, y) 传递给它
        self.common(forward, (x, y))
    # 定义一个测试函数，用于测试除以零的情况
    def test_div_by_zero(self):
        # 定义内部函数fn，接受三个参数x、runtime_zero和runtime_neg_zero
        def fn(x, runtime_zero, runtime_neg_zero):
            # 创建一个与x相同形状的全零张量zero
            zero = torch.zeros_like(x)
            # 返回多个除以零的运算结果：
            # 1. x除以正零
            # 2. x除以负零
            # 3. zero除以正零
            # 4. x除以zero（会触发除以零异常）
            # 5. x除以-zero（会触发除以零异常）
            # 6. zero除以zero（会触发除以零异常）
            # 7. x除以runtime_zero
            # 8. x除以runtime_neg_zero
            # 9. runtime_zero除以runtime_neg_zero
            return (
                x / 0.0,
                x / -0.0,
                zero / 0.0,
                x / zero,
                x / -zero,
                zero / zero,
                x / runtime_zero,
                # 注意：-runtime_zero在triton中不起作用，因为-(0.0)在triton中有问题
                x / runtime_neg_zero,
                runtime_zero / runtime_neg_zero,
            )

        # 生成一个包含10个随机数的张量a
        a = torch.randn(10)
        # 生成一个形状为10的全零张量zero
        zero = torch.zeros(10)
        # 生成一个形状为10的负零张量neg_zero
        neg_zero = -zero
        # 调用共同的测试函数common，传入定义的fn函数和参数元组(a, zero, neg_zero)
        self.common(fn, (a, zero, neg_zero))

    # 定义一个测试函数，测试两个标量值的加减乘运算
    def test_both_scalars(self):
        # 定义内部函数fn，接受两个参数a和b
        def fn(a, b):
            # 返回多个标量值的加减乘结果：
            # 1. a加b
            # 2. b加a
            # 3. a减b
            # 4. b减a
            # 5. a乘b
            # 6. b乘a
            return (
                aten.add(a, b),
                aten.add(b, a),
                aten.sub(a, b),
                aten.sub(b, a),
                aten.mul(a, b),
                aten.mul(b, a),
            )

        # 调用共同的测试函数common，传入定义的fn函数和参数元组(4, 3.3)，禁用浮点参考
        self.common(fn, (4, 3.3), reference_in_float=False)

    # 定义一个测试函数，测试在指定维度上保持求和的操作
    def test_sum_keepdims(self):
        # 定义内部函数fn，接受两个参数a和b
        def fn(a, b):
            # 返回a加b后在最后一个维度上保持求和的结果
            return (torch.sum(a + b, -1, keepdim=True),)

        # 调用共同的测试函数common，传入定义的fn函数和参数元组(形状为(8,8)的随机张量, 形状为(8,8)的随机张量)
        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    # 跳过如果Halide支持时的测试函数，测试大张量的降维操作
    @skip_if_halide  # 只支持32位索引
    def test_large_tensor_reduction(self):
        # 如果当前设备的内存不足4.5GB（即4.5 GiB）
        if not _has_sufficient_memory(self.device, 4.5 * 1024**3):
            raise unittest.SkipTest("insufficient memory")

        # 如果当前设备是CPU，则跳过测试
        if self.device == "cpu":
            raise unittest.SkipTest("Fails on CPU")

        # 测试64位索引是否正确工作的内部函数fn，接受一个参数a
        def fn(a):
            # 返回张量a的最大值
            return torch.max(a)

        # 生成一个长度为2^32的全一张量t，数据类型为torch.int8，设备为当前self.device
        t = torch.ones(2**32, dtype=torch.int8, device=self.device)
        # 将张量t的最后一个元素设为2
        t[-1] = 2

        # 在这里self.common会OOM（内存不足），因为它会复制输入来检查是否有变化
        # 通过torch._dynamo.optimize()优化函数fn的编译版本
        compiled_fn = torch._dynamo.optimize()(fn)
        # 调用编译后的函数compiled_fn，传入张量t，获取计算结果actual
        actual = compiled_fn(t)
        # 期望的结果expect是一个与当前设备和数据类型匹配的张量，值为2
        expect = torch.tensor(2, dtype=torch.int8, device=self.device)
        # 使用self.assertEqual断言实际计算结果与期望结果相等
        self.assertEqual(actual, expect)

    # 跳过如果GPU Halide支持时的测试函数，测试大广播降维操作
    @skip_if_gpu_halide  # 只支持32位索引
    def test_large_broadcast_reduction(self):
        # 如果当前设备是CPU，则跳过测试
        if self.device == "cpu":
            raise unittest.SkipTest("Fails on CPU")

        # 测试64位索引是否正确工作的内部函数fn，接受两个参数a和b
        def fn(a, b):
            # 返回张量a加b后的最大值
            return torch.max(a + b)

        # 生成一个形状为(1, 2^16)的全一张量t1，数据类型为torch.int8，设备为当前self.device
        t1 = torch.ones(1, 2**16, dtype=torch.int8, device=self.device)
        # 生成一个形状为(2^16, 1)的全一张量t2，数据类型为torch.int8，设备为当前self.device
        t2 = torch.ones(2**16, 1, dtype=torch.int8, device=self.device)

        # 将张量t1和t2的最后一个元素设为2
        t1[-1, -1] = 2
        t2[-1, -1] = 2

        # 在这里self.common会OOM（内存不足），因为它会复制输入来检查是否有变化
        # 通过torch._dynamo.optimize()优化函数fn的编译版本
        compiled_fn = torch._dynamo.optimize()(fn)
        # 调用编译后的函数compiled_fn，传入张量t1和t2，获取计算结果actual
        actual = compiled_fn(t1, t2)
        # 期望的结果expect是一个与当前设备和数据类型匹配的张量，值为4
        expect = torch.tensor(4, dtype=torch.int8, device=self.device)
        # 使用self.assertEqual断言实际计算结果与期望结果相等
        self.assertEqual(actual, expect)
    def test_large_pointwise(self):
        # 检查是否有足够的内存来执行测试，需要至少 2GB + 1 个字节的内存
        if not _has_sufficient_memory(self.device, 2 * (2**31 + 1)):
            raise unittest.SkipTest("insufficient memory")

        # 定义一个简单的函数 fn，对输入的张量 a 进行加一操作
        def fn(a):
            return a + 1

        # 创建一个大小为 2GB + 1 的张量 t，数据类型为 int8，存储在指定设备上
        t = torch.ones(2**31 + 1, dtype=torch.int8, device=self.device)
        
        # 使用动态编译优化函数 fn
        compiled_fn = torch._dynamo.optimize()(fn)
        
        # 执行优化后的函数，并将结果保存在 actual 中
        actual = compiled_fn(t)

        # 由于 assertEqual 会扩展广播输入，因此无法使用
        del t
        
        # 如果设备是 GPU，则清空 GPU 缓存
        if torch.device(self.device).type == GPU_TYPE:
            getattr(torch, GPU_TYPE).empty_cache()

        # 断言 actual 中所有元素均为 2
        self.assertTrue((actual == 2).all())

    @skip_if_halide  # 只支持32位索引
    def test_large_offset_pointwise(self):
        # 测试当输入视图张量使用 32 位步长索引，但存储偏移超过 INT_MAX 时是否使用 64 位索引
        if not _has_sufficient_memory(self.device, (2**31 + 1) + (2**30 + 1)):
            raise unittest.SkipTest("insufficient memory")

        # 定义一个简单的函数 fn，对输入的张量 a 进行加四操作
        def fn(a):
            return a + 4

        # 创建一个大小为 2GB + 1 的张量 t，数据类型为 int8，存储在指定设备上
        t = torch.ones(2**31 + 1, dtype=torch.int8, device=self.device)
        
        # 将张量 t 的一部分赋值为 0，使用 64 位索引
        t[2**30 :] = 0
        
        # 使用动态编译优化函数 fn
        compiled_fn = torch._dynamo.optimize()(fn)
        
        # 执行优化后的函数，并将结果保存在 actual 中
        actual = compiled_fn(t[2**30 :])
        
        # 断言 actual 中所有元素均为 4
        self.assertTrue((actual == 4).all())

    @skip_if_halide  # 只支持32位索引
    def test_large_strided_reduction(self):
        # 测试当输入张量的元素个数小于 INT_MAX，但步长计算超过 INT_MAX 时是否使用 64 位索引
        if not _has_sufficient_memory(self.device, 2**31 + 2):
            raise unittest.SkipTest("insufficient memory")

        # 定义一个函数 fn，返回输入张量 a 的最大值
        def fn(a):
            return torch.max(a)

        # 创建一个大小为 2GB + 1 的张量 storage，数据类型为 int8，存储在指定设备上
        storage = torch.ones(2**31 + 1, dtype=torch.int8, device=self.device)
        
        # 创建 storage 的视图 view，步长为 32
        view = storage[::32]
        
        # 将 view 的最后一个元素赋值为 2
        view[-1] = 2

        # 使用动态编译优化函数 fn
        compiled_fn = torch._dynamo.optimize()(fn)
        
        # 执行优化后的函数，并将结果保存在 actual 中
        actual = compiled_fn(view)
        
        # 创建预期结果张量 expect，数据类型为 int8，存储在指定设备上
        expect = torch.tensor(2, dtype=torch.int8, device=self.device)
        
        # 断言 actual 等于 expect
        self.assertEqual(actual, expect)

    def test_softmax(self):
        # 定义一个函数 fn，对输入张量 a 和 b 分别进行 softmax 操作
        def fn(a, b):
            return (torch.softmax(a + b, -1), torch.softmax(a, 0), torch.softmax(b, 1))

        # 调用 self.common 方法，传递函数 fn 和两个随机生成的张量作为参数
        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_log_softmax(self):
        # 定义一个函数 fn，对输入张量 a 和 b 分别进行 log_softmax 操作
        def fn(a, b):
            return (F.log_softmax(a + b, -1), F.log_softmax(a, 0), F.log_softmax(b, 1))

        # 调用 self.common 方法，传递函数 fn 和两个随机生成的张量作为参数
        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_transpose(self):
        # 定义一个函数 fn，对输入张量 a 进行转置后加 b 操作，对 b 进行转置后乘以 2 后加 10 操作
        def fn(a, b):
            return (
                torch.t(a) + b,
                torch.transpose(b * 2, 0, 1) + 10,
            )

        # 调用 self.common 方法，传递函数 fn 和两个随机生成的张量作为参数
        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_permute1(self):
        # 定义一个函数 fn，对输入张量 a 进行指定维度的置换后加 2 操作
        def fn(a):
            return (
                torch.permute(a + 1, [2, 1, 4, 0, 3]) + 2,
                torch.permute(a, [2, 1, 4, 0, 3]) + 2,
            )

        # 调用 self.common 方法，传递函数 fn 和一个随机生成的张量作为参数
        self.common(fn, (torch.randn(2, 2, 2, 2, 2),))
    def test_permute2(self):
        def fn(a):
            # 在第一维上展开张量，步长为2，维度变化为(2, 2, ...)
            a = a.unfold(0, 2, 1)
            # 在第一维上增加一个维度
            a = torch.unsqueeze(a, 1)
            # 对张量进行维度重排，指定新的维度顺序[0, 2, 3, -3]
            a = torch.permute(a, [0, 2, 3, -3])
            return (a,)

        # 调用共同的测试函数，对给定的函数 fn 和参数进行测试
        self.common(fn, (torch.randn(4, 4),))

    def test_expand(self):
        def fn(a):
            # 对张量进行扩展操作，第一个操作：每个元素加1后扩展为(3, 4, 2, 3, 2)的张量，然后再加2
            return (
                (a + 1).expand(3, 4, 2, 3, 2) + 2,
                # 第二个操作：对原始张量进行扩展为(2, 1, 2, 3, 2)，然后再加2
                a.expand(2, 1, 2, 3, 2) + 2,
            ), a.expand(2, -1, 5, -1)  # 最后的操作：对原始张量进行扩展，其中-1表示维度维持不变

        # 调用共同的测试函数，对给定的函数 fn 和参数进行测试
        self.common(fn, (torch.randn(2, 1, 2),))

    def test_squeeze1(self):
        def fn(a):
            # 对张量进行挤压操作，去除所有大小为1的维度，然后每个元素加1再加2
            return ((a + 1).squeeze() + 2, a.squeeze() + 2)

        # 调用共同的测试函数，对给定的函数 fn 和参数进行测试
        self.common(fn, (torch.randn(1, 2, 1, 2, 2, 1, 1),))

    def test_squeeze2(self):
        def fn(a):
            # 对张量进行挤压操作，去除指定维度上大小为1的维度，然后每个元素加1再加2
            return ((a + 1).squeeze(-1).squeeze(2) + 2, a.squeeze(0) + 2)

        # 调用共同的测试函数，对给定的函数 fn 和参数进行测试
        self.common(fn, (torch.randn(1, 2, 1, 2, 2, 2, 1),))

    def test_squeeze_varargs(self):
        def fn(x):
            # 对张量进行挤压操作，去除指定维度上大小为1的维度，然后克隆新的张量
            return x.squeeze(1, 2).clone()

        a = torch.randn(1024, 1, 1)
        # 调用共同的测试函数，对给定的函数 fn 和参数进行测试
        self.common(fn, (a,))

    def test_simplify_loops(self):
        def fn(a, b):
            # 对两个张量进行简单的加法操作
            return a + b

        # 调用共同的测试函数，对给定的函数 fn 和参数进行测试
        self.common(
            fn,
            (
                torch.randn(2, 3, 4, 5, 6),
                torch.randn(4, 2, 3, 5, 6).permute(1, 2, 0, 3, 4),
            ),
        )

    def test_unsqueeze(self):
        def fn(a):
            # 对张量进行增加维度操作，分别在不同位置增加维度，并加2
            return (
                torch.unsqueeze(a + 1, -1) + 2,
                torch.unsqueeze(a, 2) + 2,
                torch.unsqueeze(a + 1, 0) + 2,
                torch.unsqueeze(a, -2) + 2,
            )

        # 调用共同的测试函数，对给定的函数 fn 和参数进行测试
        self.common(
            fn,
            (
                torch.randn(
                    2,
                    2,
                    2,
                    2,
                ),
            ),
        )

    def test_unsqueeze_inplace(self):
        def fn(a):
            tmp1 = a + 1
            # 使用原位操作增加张量维度，并指定位置为2
            aten.unsqueeze_(tmp1, 2)
            # 使用原位操作增加张量维度，并指定位置为0，然后再加2
            tmp2 = aten.unsqueeze_(a + 1, 0) + 2
            return (tmp1, tmp2)

        # 调用共同的测试函数，对给定的函数 fn 和参数进行测试
        self.common(
            fn,
            (
                torch.randn(
                    2,
                    2,
                    2,
                    2,
                ),
            ),
        )

    def test_addmm(self):
        def fn(a, b, c):
            # 执行矩阵乘法加法运算，对每个张量加相应的偏移值后再加4
            return (torch.addmm(a + 1, b + 2, c + 3) + 4,)

        # 调用共同的测试函数，对给定的函数 fn 和参数进行测试
        self.common(
            fn,
            (
                torch.randn(8, 8),
                torch.randn(8, 8),
                torch.randn(8, 8),
            ),
        )

    # https://github.com/pytorch/pytorch/issues/98979
    @skipCUDAIf(True, "cuda failed for float64 linear")
    @skipIfXpu(msg="Double and complex datatype matmul is not supported in oneDNN")
    def test_linear_float64(self):
        # 创建一个使用 float64 数据类型的线性层模型，并设置为评估模式
        mod = torch.nn.Sequential(torch.nn.Linear(8, 16).to(torch.float64)).eval()
        with torch.no_grad():
            # 调用共同的测试函数，对给定的模型 mod 和输入数据进行测试
            self.common(mod, (torch.randn(2, 8).to(torch.float64),))
    # 定义一个测试线性层的方法，使用了一个简单的神经网络模型
    def test_linear1(self):
        # 创建一个包含线性层、Sigmoid 激活函数和 ToTuple 转换的神经网络模型
        mod = torch.nn.Sequential(
            torch.nn.Linear(8, 16),  # 输入维度为 8，输出维度为 16 的线性层
            torch.nn.Sigmoid(),      # Sigmoid 激活函数
            ToTuple(),               # 将输出转换为元组的自定义转换器
        )
        # 调用共同的测试方法，验证模型的功能
        self.common(mod, (torch.randn(2, 8),))

    # 定义另一个测试线性层的方法，使用更深的神经网络模型
    def test_linear2(self):
        # 创建一个包含多个线性层和ReLU激活函数的神经网络模型
        mod = torch.nn.Sequential(
            torch.nn.Linear(8, 8),   # 输入维度为 8，输出维度为 8 的线性层
            torch.nn.ReLU(),         # ReLU 激活函数
            torch.nn.Linear(8, 8),   # 输入维度为 8，输出维度为 8 的线性层
            torch.nn.ReLU(),         # ReLU 激活函数
            torch.nn.Linear(8, 8),   # 输入维度为 8，输出维度为 8 的线性层
            torch.nn.ReLU(),         # ReLU 激活函数
            torch.nn.Linear(8, 8),   # 输入维度为 8，输出维度为 8 的线性层
            torch.nn.ReLU(),         # ReLU 激活函数
        )
        # 调用共同的测试方法，验证模型的功能
        self.common(
            mod,
            (torch.randn(2, 8),),    # 输入数据为 2 组，每组维度为 8
            atol=1e-3,               # 绝对误差容限为 1e-3
            rtol=0.01,               # 相对误差容限为 0.01
        )

    # 定义一个测试矩阵乘法的方法
    def test_bmm1(self):
        # 定义一个接受两个参数并返回两个矩阵乘积及其偏移的函数
        def fn(a, b):
            return (
                torch.bmm(a, b),                              # 计算 a 和 b 的批次矩阵乘积
                torch.bmm(a + 1, b + 2) + 3,                  # 计算 a+1 和 b+2 的批次矩阵乘积后加 3
            )
        # 调用共同的测试方法，验证函数的功能
        self.common(
            fn,
            (
                torch.randn(2, 8, 8),                         # 第一个参数为形状为 (2, 8, 8) 的张量
                torch.randn(2, 8, 8),                         # 第二个参数为形状为 (2, 8, 8) 的张量
            ),
            check_lowp=False,                                 # 禁用低精度检查
        )
        # 调用共同的测试方法，验证函数的功能
        self.common(
            fn,
            (
                torch.randn(1, 16, 8),                        # 第一个参数为形状为 (1, 16, 8) 的张量
                torch.randn(1, 8, 10),                        # 第二个参数为形状为 (1, 8, 10) 的张量
            ),
            check_lowp=False,                                 # 禁用低精度检查
        )

    # 定义另一个测试矩阵乘法的方法
    def test_bmm2(self):
        # 定义一个接受两个参数并返回它们的批次矩阵乘积的函数
        def fn(a, b):
            return torch.bmm(a.permute(0, 2, 1), b)           # 计算 a 的转置与 b 的批次矩阵乘积
        # 调用共同的测试方法，验证函数的功能
        self.common(
            fn,
            (
                torch.randn(1, 8, 8),                         # 第一个参数为形状为 (1, 8, 8) 的张量
                torch.randn(1, 8, 8),                         # 第二个参数为形状为 (1, 8, 8) 的张量
            ),
            check_lowp=False,                                 # 禁用低精度检查
        )

    # 标记为在 Python 3.12 上会导致段错误的测试
    @skipIfPy312
    # 使用 'triton' 作为混合矩阵乘选项的配置补丁
    @config.patch(mixed_mm_choice="triton")
    # 定义一个测试混合矩阵乘法的方法
    def test_mixed_mm(self):
        # 定义一个接受两个参数并返回它们的矩阵乘积的函数
        def fn(a, b):
            return torch.mm(a, b.to(a.dtype))                 # 计算 a 和 b 转换为 a 数据类型后的矩阵乘积
        # 调用共同的测试方法，验证函数的功能
        self.common(
            fn,
            (
                torch.randn(8, 8),                            # 第一个参数为形状为 (8, 8) 的张量
                torch.randint(-128, 127, (8, 8), dtype=torch.int8),  # 第二个参数为形状为 (8, 8) 的随机整数张量
            ),
            check_lowp=True,                                  # 启用低精度检查
        )

    # 标记为在 Python 3.12 上会导致段错误的测试
    @skipIfPy312
    # 使用 'triton' 作为混合矩阵乘选项的配置补丁
    @config.patch(mixed_mm_choice="triton")
    # 定义另一个测试混合矩阵乘法的方法
    def test_mixed_mm2(self):
        # 定义一个接受四个参数并返回加权矩阵乘积的函数
        def fn(a, b, scale, bias):
            return torch.mm(a, b.to(a.dtype)) * scale + bias  # 计算 a 和 b 转换为 a 数据类型后的加权矩阵乘积
        # 调用共同的测试方法，验证函数的功能
        self.common(
            fn,
            (
                torch.randn(8, 8),                            # 第一个参数为形状为 (8, 8) 的张量
                torch.randint(-128, 127, (8, 8), dtype=torch.int8),  # 第二个参数为形状为 (8, 8) 的随机整数张量
                torch.randn(8),                               # 第三个参数为形状为 (8,) 的随机张量
                torch.randn(8),                               # 第四个参数为形状为 (8,) 的随机张量
            ),
            check_lowp=True,                                  # 启用低精度检查
        )

    # 标记为在 Python 3.12 上会导致段错误的测试
    @skipIfPy312
    # 使用 'triton' 作为混合矩阵乘选项的配置补丁
    @config.patch(mixed_mm_choice="triton")
    # 定义另一个测试混合矩阵乘法的方法
    def test_mixed_mm3(self):
        # 定义一个接受两个参数并返回它们的矩阵乘积的函数
        def fn(a, b):
            return torch.mm(a, b.to(a.dtype))                 # 计算 a 和 b 转换为 a 数据类型后的矩阵乘积
        # 调用共同的测试方法，验证函数的功能
        # (256, 256) @ (256, 256)，
    def test_uint4x2_mixed_mm(self):
        # 定义内部函数 fn，用于计算两个张量的矩阵乘法，其中将张量 b 按位与操作并拼接后重塑形状，并转换数据类型后减去 8
        def fn(a, b):
            return torch.mm(
                a,
                torch.cat((b & 0xF, b >> 4), 1)  # 拼接操作，将 b 按位与 0xF 后的结果和右移 4 位后的结果拼接在一起
                .reshape(-1, b.shape[1])  # 重塑张量形状为 (-1, b.shape[1])
                .to(a.dtype)  # 转换为与张量 a 相同的数据类型
                .sub(8),  # 减去常数 8
            )

        # 调用通用测试函数 common，测试 fn 函数的功能，传入两个张量参数，第一个是随机生成的 8x8 张量，第二个是随机生成的 4x8 uint8 类型的张量
        self.common(
            fn,
            (
                torch.randn(8, 8),
                torch.randint(0, 255, (4, 8), dtype=torch.uint8),
            ),
            check_lowp=True,  # 启用低精度检查
        )

    @skipIfXpu
    def test_mm_mixed_dtype(self):
        # 定义函数 fn，用于计算两个张量的矩阵乘法
        def fn(a, b):
            return torch.mm(a, b)

        # 创建两个张量 t1 和 t2，分别是从 0 到 5 的 float 类型张量和从 0 到 8 的 int64 类型张量
        t1 = torch.arange(6, dtype=torch.float, device=self.device).view(2, 3)
        t2 = torch.arange(9, dtype=torch.int64, device=self.device).view(3, 3)

        # 预期抛出 RuntimeError，提示两个张量的数据类型不匹配
        msg = "expected .* and .* to have the same dtype, but got: .* != .*"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.compile(fn)(t1, t2)  # 使用 Torch JIT 编译 fn 函数并传入 t1 和 t2
        with self.assertRaisesRegex(RuntimeError, msg):
            fn(t1, t2)  # 直接调用 fn 函数传入 t1 和 t2

    @skipIfXpu
    def test_linear_mixed_dtype(self):
        # 定义一个继承自 nn.Module 的神经网络类 Net
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()  # 调用父类的初始化方法
                self.fc1 = nn.Linear(3, 3)  # 创建一个线性层，输入维度和输出维度均为 3

            def forward(self, x):
                x = self.fc1(x.permute(1, 2, 0))  # 对输入张量 x 进行维度置换后通过线性层 fc1
                return x

        # 创建 Net 类的实例 fn，将其移至指定设备
        fn = Net().to(self.device)
        # 创建一个 3x3x3 的张量 t，其元素值从 0 到 26，同时也移至指定设备
        t = torch.arange(27, device=self.device).view(3, 3, 3)

        # 预期抛出 RuntimeError，提示张量的数据类型不匹配
        msg = "expected .* and .* to have the same dtype, but got: .* != .*"
        with self.assertRaisesRegex(RuntimeError, msg):
            fn(t)  # 调用神经网络 fn 的前向传播，传入 t 张量
        with self.assertRaisesRegex(RuntimeError, msg):
            with torch.no_grad():
                torch.compile(fn)(t)  # 使用 Torch JIT 编译 fn 函数并传入 t 张量

        # TODO: Autograd internal assertion
        # 预期抛出 RuntimeError，提示自动求导过程中的内部断言失败
        msg = r".*isDifferentiableType\(variable.scalar_type\(\)\) INTERNAL ASSERT FAILED.*"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.compile(fn)(t)  # 使用 Torch JIT 编译 fn 函数并传入 t 张量

    def test_scalar_input(self):
        # 定义函数 fn，对输入张量 x 和标量 y 进行除法操作，取 floor 结果
        def fn(x, y):
            a = torch.div(x, y, rounding_mode="floor")  # 对 x 除以 y，取 floor
            return a

        # 调用通用测试函数 common，测试 fn 函数的功能，传入 x 是随机生成的 1x8 张量，y 是标量 5400
        self.common(fn, [torch.randint(5, (1, 8)), 5400])

    @torch._dynamo.config.patch(dynamic_shapes=True)
    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_scalar_output(self):
        # 定义函数 fn，接受两个参数 arg0_1 和 arg2_1，分别对其进行一系列操作后返回三个张量的元组
        def fn(arg0_1, arg2_1):
            arg1_1 = arg2_1.size(1)  # 获取 arg2_1 的第二维度大小
            view = torch.ops.aten.view.default(arg2_1, [-1, arg1_1])  # 使用 Torch 的 aten.view 操作，将 arg2_1 视图重塑为 [-1, arg1_1]
            embedding = torch.ops.aten.embedding.default(arg0_1, view)  # 使用 Torch 的 aten.embedding 操作进行嵌入计算
            full = torch.ops.aten.full.default([1, arg1_1], 1, dtype=torch.float32)  # 使用 Torch 的 aten.full 操作创建指定形状和数据类型的全填充张量
            return (full, arg1_1, embedding)  # 返回三个张量的元组

        # 使用 rand_strided 函数生成指定形状和数据类型的 arg0_1 和 arg2_1 张量
        arg0_1 = rand_strided((32128, 768), (768, 1), device="cpu", dtype=torch.float32)
        arg2_1 = rand_strided((1, 22), (22, 1), device="cpu", dtype=torch.int64)
        # 调用通用测试函数 common，测试 fn 函数的功能，传入 arg0_1 和 arg2_1 作为参数
        self.common(fn, [arg0_1, arg2_1])
    def test_shape_prop_torch_ones(self):
        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, attention_scores):
                # 创建一个形状为 (8, 1, 1, 512) 的全为1的张量，使用与 attention_scores 相同的设备
                extended_attention_mask = torch.ones(
                    8, 1, 1, 512, device=attention_scores.device
                )
                # 将 attention_scores 和 extended_attention_mask 相加
                attention_scores = attention_scores + extended_attention_mask

                return attention_scores

        # 创建 Model 类的实例 mod，并设置为评估模式
        mod = Model().eval()
        # 在无需梯度计算的上下文中执行以下操作
        with torch.no_grad():
            # 调用 self.common 方法，传入 mod 和一个随机生成的张量元组作为参数
            self.common(
                mod,
                (torch.randn(8, 12, 512, 512),),
            )

    @slowTest
    @expectedFailureCodegenDynamic
    @config.patch({"freezing": True})
    def test_conv_bn_fuse(self):
        # 如果当前设备为 GPU，跳过该测试，并输出相应提示信息
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("only support cpu conv bn test")

        # 定义不同维度的输入形状字典
        input_shapes = {1: (112,), 2: (112, 112), 3: (55, 55, 55)}
        # 定义不同维度的卷积模块字典
        conv_modules = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        # 定义不同维度的批标准化模块字典
        bn_modules = {
            1: torch.nn.BatchNorm1d,
            2: torch.nn.BatchNorm2d,
            3: torch.nn.BatchNorm3d,
        }
        # 生成不同参数组合的迭代器
        options = itertools.product(
            [1, 2, 3],
            [True, False],
            [1, 3],
            [1, 2],
            [1, 4],
        )

        # 遍历 options 中的每个参数组合
        for (
            dim,
            bias,
            kernel_size,
            dilation,
            groups,
        ) in options:
            # 计算输出通道数 oC 和输入通道数 iC
            oC = 32 * groups
            iC = 3 * groups
            # 构建输入张量的形状 x_shape
            x_shape = (1, iC) + input_shapes[dim]
            # 创建一个包含卷积和批标准化的序列模型 mod，并设置为评估模式
            mod = torch.nn.Sequential(
                conv_modules[dim](
                    iC,
                    oC,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                ),
                bn_modules[dim](oC),
            ).eval()
            # 测试内存格式的列表
            test_memory_format = [torch.contiguous_format]
            # 如果不是 GPU 环境且 dim 大于 1，则添加 channels_last 或 channels_last_3d 的测试内存格式
            if not HAS_GPU and dim > 1:
                channels_last = (
                    torch.channels_last if dim == 2 else torch.channels_last_3d
                )
                test_memory_format.append(channels_last)
            # 遍历测试内存格式列表中的每个内存格式
            for memory_format in test_memory_format:
                # 创建一个随机张量 v，并指定其内存格式
                v = torch.randn(x_shape, dtype=torch.float32).to(
                    memory_format=memory_format
                )
                # 在无需梯度计算的上下文中执行以下操作
                with torch.no_grad():
                    # 调用 self.common 方法，传入 mod 和张量 v 作为参数
                    self.common(
                        mod,
                        (v,),
                    )

    @skipIfRocm
    def test_conv_inference_heuristics(self):
        # 检查设备是否为 GPU_TYPE，如果不是则跳过测试
        if self.device != GPU_TYPE:
            raise unittest.SkipTest(f"{GPU_TYPE} only test")

        # 设置卷积层的输入通道数、输出通道数、核大小和分组数
        in_channels = 6
        out_channels = 6
        kernel_size = 3
        groups = 3

        # 创建分组卷积层对象，并将其移到指定设备上
        grouped_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, groups=groups
        ).to(self.device)

        # 创建输入张量，并将其移到指定设备上
        input_tensor = torch.randn(1, in_channels, 10, 10).to(self.device)

        # 执行前向传播
        @torch.compile()
        def foo(m, inp):
            return m(inp)

        with torch.no_grad():
            # 运行并获取编译后的代码
            _, code = run_and_get_code(foo, grouped_conv, input_tensor)
            # 检查编译后的代码是否符合预期格式
            # 在进行卷积之前不应进行通道置换
            FileCheck().check_not(".run(").check(".convolution(").run(code[0])

        # 设置输入通道数和输出通道数，用于推断过程中的通道置换
        in_channels = 8
        out_channels = 4
        kernel_size = 3

        # 创建普通卷积层对象，并将其移到指定设备上
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size).to(self.device)

        # 创建输入张量，并将其移到指定设备上
        input_tensor = torch.randn(1, in_channels, 10, 10).to(self.device)

        with torch.no_grad():
            # 运行并获取编译后的代码
            _, code = run_and_get_code(foo, conv_layer, input_tensor)
            # 检查编译后的代码是否符合预期格式
            if is_halide_backend(self.device):
                # 如果是 Halide 后端，检查是否存在通道置换
                FileCheck().check("halide_kernel_0(").check(".convolution(").run(
                    code[0]
                )
            else:
                # 否则检查编译后的代码是否符合预期格式
                FileCheck().check(".run(").check(".convolution(").run(code[0])

    def test_upsample_cat_conv(self):
        # 如果设备为 GPU_TYPE，则跳过测试
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("only support cpu upsample_cat_conv test")

        # 定义一个简单的神经网络模型
        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                # 设置最近邻插值上采样层和卷积层
                self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
                self.conv = torch.nn.Conv2d(
                    8,
                    5,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    dilation=1,
                    **kwargs,
                )

            def forward(self, x, y):
                # 对输入 x 进行上采样
                x = self.upsample(x)
                # 在通道维度上连接 x 和 y
                z = torch.cat([x, y], dim=1)
                # 对连接后的张量进行卷积操作
                z = self.conv(z)
                return z

        # 创建输入张量 v1 和 v2
        v1 = torch.randn([8, 2, 12, 26])
        v2 = torch.randn([8, 6, 24, 52])

        with torch.no_grad():
            # 使用 common 方法进行模型评估
            self.common(
                M().eval(),
                (v1, v2),
            )

    def test_aliased_buffer_reuse(self):
        # 定义一个简单的函数 fn，用于执行张量操作
        def fn(x, y):
            x = 2 * x
            y = 2 * y
            # 在通道维度上连接 x 和 y，并命名为 c
            c = torch.cat([x, y], dim=-1)
            # 对连接后的张量加上标量 1，并进行矩阵乘法
            d = 1 + c
            m = torch.mm(d, d)
            # 返回 m 的前两列与 x 的元素之和
            return m[:, :2] + x

        # 使用 common 方法进行函数 fn 的测试，关闭低精度检查
        self.common(fn, (torch.randn(4, 2), torch.randn(4, 2)), check_lowp=False)
    def test_slice_view_with_graph_break(self):
        # 定义测试函数 fn
        def fn():
            # 创建包含一个元素的张量 a，并指定设备为 self.device
            a = torch.tensor([1], device=self.device)
            # 对张量 a 进行切片操作，生成新的张量 a
            a = a[0:1]
            # 对张量 a 进行squeeze操作，去除维度为1的维度
            b = a.squeeze()
            # 修改张量 a 的第一个元素为 0
            a[0] = 0
            # 如果张量 a 的第一个元素小于 1e5，则什么也不做
            if a[0] < 1e5:
                pass
            # 再次修改张量 a 的第一个元素为 2
            a[0] = 2
            # 返回原始的张量 b
            return b

        # 调用 fn 函数得到期望的结果
        expect = fn()
        # 编译函数 fn
        opt_fn = torch.compile(fn)
        # 调用优化后的函数 opt_fn 得到实际结果
        actual = opt_fn()
        # 断言期望结果与实际结果相等
        self.assertEqual(expect, actual)

    def test_view_detach(self):
        # 定义函数 fn，接受参数 a，并对其第一个元素进行detach操作
        def fn(a):
            return a[0].detach()

        # 调用公共测试方法 common，传入函数 fn 和参数元组
        self.common(
            fn,
            (torch.randn([4, 4], requires_grad=True),),
        )

    def test_gather1(self):
        # 定义函数 fn，接受两个参数 a 和 b，并使用 torch.gather 在指定维度上进行索引操作
        def fn(a, b):
            return (
                torch.gather(a.expand([4, 5, 10, 6]), 3, b + 1),
                torch.gather(a.expand([4, 5, 10, 6]), -1, b + 1),
            )

        # 调用公共测试方法 common，传入函数 fn 和参数元组
        self.common(
            fn,
            (
                torch.randn([1, 1, 10, 6]),
                torch.randint(5, [4, 5, 10, 1], dtype=torch.int64),
            ),
        )

    def test_gather2(self):
        # 0维张量的测试
        def fn(a, b):
            return torch.gather(a, 0, b) + torch.gather(a, -1, b)

        # 创建张量 x 和 y
        x = torch.tensor(123)
        y = torch.tensor(0)
        # 断言 fn 函数的输出结果与预期相等
        self.assertEqual(fn(x, y), x + x)

    def test_gather3(self):
        # 定义函数 fn，接受两个参数 a 和 b，并使用 torch.gather 在指定维度上进行索引操作，启用稀疏梯度
        def fn(a, b):
            return torch.gather(a, 1, b, sparse_grad=True)

        # 调用公共测试方法 common，传入函数 fn 和参数元组
        self.common(
            fn,
            (
                torch.randn([4, 5, 10, 6], requires_grad=True),
                torch.randint(5, [4, 5, 10, 1], dtype=torch.int64),
            ),
        )

    def test_slice1(self):
        # 定义函数 fn，接受参数 a，进行不同形式的切片操作并返回结果元组
        def fn(a):
            return (
                a[:, :10, 0] + a[:, 10:, 0],
                (a + 1)[:, :10, 0] + (a + 1)[:, 10:, 0],
                a[:, -30:, 0],  # 负索引超出范围
                a[:, :-30, 0],  # 负索引超出范围
            )

        # 调用公共测试方法 common，传入函数 fn 和参数元组
        self.common(
            fn,
            (torch.randn([2, 20, 2]),),
        )

    def test_slice2(self):
        # 定义函数 fn，接受参数 a，进行不同形式的切片操作并返回结果元组
        def fn(a):
            return (
                a[:-1, ::2, -1] + a[-1:, 1::2, -2],
                (a + 1)[:-1, ::2, -1] + (a + 2)[-1:, 1::2, -2],
            )

        # 调用公共测试方法 common，传入函数 fn 和参数元组
        self.common(
            fn,
            (torch.randn([2, 20, 2]),),
        )

    # 它是一个视图，因此不生成内核
    @expectedFailureCodegenDynamic
    def test_slice3(self):
        # 定义函数 fn，接受参数 a 和 b，使用 torch.ops.aten.slice.Tensor 进行切片操作
        def fn(a, b):
            return torch.ops.aten.slice.Tensor(a, 0, 0, -b)

        # 创建张量 x
        x = torch.rand(48, 3, 512, 512)
        # 调用公共测试方法 common，传入函数 fn 和参数元组
        self.common(fn, (x, 2))

    @expectedFailureCodegenDynamic
    def test_slice4(self):
        # 需要将起始或结束位置夹紧的空切片
        def fn(a):
            return (
                aten.slice.Tensor(a, 0, 2, 0, 1),
                aten.slice.Tensor(a, 0, a.shape[0], a.shape[0] + 10, 1),
                aten.slice.Tensor(a, 0, -20, 0, 1),
                aten.slice.Tensor(a, 0, -20, -16, 1),
            )

        # 创建张量 x
        x = torch.rand(10)
        # 调用公共测试方法 common，传入函数 fn 和参数元组
        self.common(fn, (x,))
    # 定义一个测试函数，用于测试 torch.split 函数在给定不同参数情况下的行为
    def test_split_with_list(self):
        # 定义一个内部函数 fn，接受一个张量 a 和一个大小列表 sizes，返回一个列表，其中每个元素是对应切片加 1.0 后的结果
        def fn(a, sizes):
            return [t + 1.0 for t in torch.split(a * 2.0, sizes, -1)]

        # 调用 self.common 方法测试 fn 函数的行为，传入不同的张量和大小列表作为参数
        self.common(fn, (torch.randn(2, 2, 10), [3, 3, 4]))
        self.common(fn, (torch.randn(2, 2, 10), [4, 3, 3]))
        self.common(fn, (torch.randn(2, 2, 10), [1, 2, 3, 4]))

    # 定义一个测试函数，测试 torch.split 函数在给定整数作为参数时的行为
    def test_split_with_integer(self):
        # 使用 torch.compile 标记的动态编译函数 f，接受一个张量 x 和一个大小 sizes，返回对 x 进行按 sizes 切片的结果
        @torch.compile(dynamic=True)
        def f(x, sizes):
            return torch.split(x, sizes, -1)

        # 对长度为 10 的张量进行等分切片，将结果分为两部分，每部分大小为 (2, 5)
        r1, r2 = f(torch.randn(2, 10), 5)
        self.assertTrue(r1.size() == (2, 5))
        self.assertTrue(r2.size() == (2, 5))

        # 对长度为 12 的张量进行等分切片，将结果分为三部分，每部分大小为 (2, 4)
        r1, r2, r3 = f(torch.randn(2, 12), 4)
        self.assertTrue(r1.size() == (2, 4))
        self.assertTrue(r2.size() == (2, 4))
        self.assertTrue(r3.size() == (2, 4))

        # 对长度为 10 的张量进行不等分切片，将结果分为四部分，大小分别为 (2, 3), (2, 3), (2, 3), (2, 1)
        r1, r2, r3, r4 = f(torch.randn(2, 10), 3)
        self.assertTrue(r1.size() == (2, 3))
        self.assertTrue(r2.size() == (2, 3))
        self.assertTrue(r3.size() == (2, 3))
        self.assertTrue(r4.size() == (2, 1))

    # 定义一个测试函数，测试 torch.split 函数在无法成功切片时的行为
    def test_split_failed(self):
        # 使用 @torch._dynamo.optimize 标记的优化函数 fn，接受一个张量 a，尝试按给定大小 [2, 1, 1] 在维度 1 上切片
        @torch._dynamo.optimize("inductor")
        def fn(a):
            return torch.split(a, [2, 1, 1], dim=1)

        # 使用 self.assertRaisesRegex 检查调用 fn 函数时抛出的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, ""):
            fn(torch.randn(1, 5))

    # 定义一个测试函数，测试 torch._dynamo.optimize 标记的动态优化函数在带有 assert 语句的情况下的行为
    def test_inductor_assert(self):
        # 使用 @torch._dynamo.optimize 标记的动态优化函数 fn，接受一个张量 a，确保其形状至少为 (2, 4)，然后返回其余弦值
        @torch._dynamo.optimize("inductor", dynamic=True)
        def fn(a):
            assert a.shape[0] >= 2 and a.shape[1] >= 4
            return a.cos()

        # 创建一个输入张量 inp，形状为 (2, 4, 6)，并标记其在维度 0 和 1 上为动态的
        inp = torch.randn(2, 4, 6)
        torch._dynamo.mark_dynamic(inp, 0)
        torch._dynamo.mark_dynamic(inp, 1)
        # 使用 self.assertEqual 检查调用 fn 函数时返回值与 inp.cos() 的相等性
        self.assertEqual(fn(inp), inp.cos())

    # 定义一个测试函数，测试 torch.split 函数的基本行为
    def test_split(self):
        # 定义一个内部函数 fn，接受一个张量 a，使用 torch.split 将其按大小 3 在维度 -1 上切片为四部分，并返回结果的元组
        def fn(a):
            t = torch.split(a, 3, -1)
            return (t[0], t[1], t[2], t[3])

        # 使用 self.common 方法测试 fn 函数，传入不同的张量作为参数
        self.common(
            fn,
            (torch.randn([2, 2, 10]),),
        )

        # 定义一个内部函数 fn2，接受一个张量 a，并对 a 加 1 后调用 fn 函数，返回结果
        def fn2(a):
            return fn(a + 1)

        # 使用 self.common 方法测试 fn2 函数，传入不同的张量作为参数
        self.common(
            fn2,
            (torch.randn([2, 2, 10]),),
        )

    # 定义一个测试函数，测试 torch.aten._to_copy 和 torch.aten.to 函数在不同情况下的行为
    def test_to_dtype(self):
        # 定义一个内部函数 fn，接受两个张量 a 和 b，分别对 a 进行 dtype=6 的转换，并对 b 加 1 后转换为 torch.float64 和 torch.bool 类型，返回结果的元组
        def fn(a, b):
            return (
                aten._to_copy(a, dtype=6),
                aten._to_copy(b + 1, dtype=6),
                aten.to(b, torch.float64),
                aten.to(b, torch.bool),
            )

        # 使用 self.common 方法测试 fn 函数，传入不同的张量作为参数
        self.common(
            fn,
            (
                torch.randn([2, 2, 10]),
                torch.randn([2, 2, 10], dtype=torch.float64),
            ),
        )

    # 标记当前测试函数需要 GPU 支持
    @requires_gpu()
    def test_to_device(self):
        # 定义一个函数 fn，根据张量 a 的设备类型进行条件判断并复制到指定设备
        def fn(a):
            if a.device.type == "cpu":
                return aten._to_copy(
                    a, device=torch.device(GPU_TYPE), dtype=6, layout=0
                )
            else:
                return aten._to_copy(a, device=torch.device("cpu"), dtype=6, layout=0)

        # 调用公共测试函数 common，测试 fn 函数的功能
        self.common(
            fn,
            (torch.randn([2, 2, 10]),),  # 参数为一个随机张量
        )

    def test_to_memory_format(self):
        # 定义一个函数 fn，将张量 a 转换到指定的内存格式 memory_format
        def fn(a, memory_format):
            return a.to(memory_format=memory_format)

        # 调用公共测试函数 common，测试 fn 函数的功能
        self.common(
            fn,
            (torch.randn([2, 2, 10, 10]), torch.channels_last),  # 参数为一个随机张量和内存格式为 channels_last
        )
        # 再次调用公共测试函数 common，测试 fn 函数的功能，这次使用带有内存格式的张量和 contiguous_format 内存格式
        self.common(
            fn,
            (
                torch.randn([2, 2, 10, 10]).to(memory_format=torch.channels_last),
                torch.contiguous_format,
            ),
        )

    @requires_gpu()
    def test_to_device_constant(self):
        # 定义一个函数 fn，根据张量 a 的设备类型，将一个常量张量移到相反的设备上
        def fn(a):
            d1 = a.device.type
            if d1 == "cpu":
                d2 = GPU_TYPE
            else:
                d2 = "cpu"

            const1 = torch.as_tensor(list(range(64)), device=d2)
            return (
                torch.arange(10, device=d2).to(d1) + a,  # 将张量 a 移到设备 d2 上并执行加法
                const1.to(d1),  # 将常量张量 const1 移到设备 d1 上
                (const1 + 1).to(d1),  # 将常量张量 const1 加 1 并移动到设备 d1 上
            )

        # 调用公共测试函数 common，测试 fn 函数的功能
        self.common(
            fn,
            (torch.randn([10]),),  # 参数为一个随机张量
        )

    @requires_gpu()
    def test_multi_device(self):
        # 定义一个函数 fn，对输入张量 x 进行多次设备操作
        def fn(x):
            x = x + 1
            x = x + 2
            x = x.to(device=GPU_TYPE)  # 将张量 x 移到 GPU_TYPE 设备上
            x = x + 3
            x = x + 4
            x = x.cpu()  # 将张量 x 移回 CPU 设备
            x = x + 5
            x = x + 6
            x = x.to(device=GPU_TYPE)  # 将张量 x 移到 GPU_TYPE 设备上
            x = x + 7
            x = x + 8
            x = x.cpu()  # 将张量 x 移回 CPU 设备
            x = x + 9
            x = x + 10
            return x

        # 调用公共测试函数 common，测试 fn 函数的功能
        self.common(
            fn,
            (torch.randn([2, 2, 10]),),  # 参数为一个随机张量
            check_lowp=False,  # CPU 不支持 fp16，需要显式的 .cpu() 调用
        )

    @skipIfRocm
    @requires_multigpu()
    def test_multi_gpu_device(self):
        # TODO: https://github.com/pytorch/pytorch/issues/92627
        x = torch.rand([4], device=GPU_TYPE)  # 在指定设备 GPU_TYPE 上创建一个随机张量

        # 定义一个函数 fn，对输入张量 x 和 y 执行除法运算，并将结果移到指定的设备上
        def fn(x, y):
            r = torch.ops.aten.div(x, y)
            r = r.to(f"{GPU_TYPE}:1")  # 将结果张量 r 移到指定的设备 GPU_TYPE:1 上
            return 2 * r

        # 调用公共测试函数 common，测试 fn 函数的功能
        self.common(fn, (torch.randn(4), torch.randn(4)), check_lowp=False)

    @requires_multigpu()
    def test_multi_gpu_recompile_on_index(self):
        # 设置矩阵乘法的浮点精度为高
        torch.set_float32_matmul_precision("high")

        # 定义矩阵乘法函数
        def gemm(x, y):
            return x @ y

        # 定义失败处理函数
        failed_guard = None

        def fail(guard):
            nonlocal failed_guard
            failed_guard = guard

        # 对 gemm 函数进行优化，使用动态编译器
        gemm_opt = torch._dynamo.optimize("inductor", guard_fail_fn=fail)(gemm)

        # 创建两个随机张量 x0 和 y0，并指定在第一个 GPU 上
        x0 = torch.randn(1024, 1024, device=f"{GPU_TYPE}:0")
        y0 = torch.randn(1024, 1024, device=f"{GPU_TYPE}:0")

        # 使用优化后的 gemm 函数进行计算
        gemm_opt(x0, y0)

        # 创建两个随机张量 x1 和 y1，并指定在第二个 GPU 上
        x1 = torch.randn(1024, 1024, device=f"{GPU_TYPE}:1")
        y1 = torch.randn(1024, 1024, device=f"{GPU_TYPE}:1")

        # 使用优化后的 gemm 函数进行计算
        gemm_opt(x1, y1)

        # 断言失败处理函数被调用过，说明出现了预期的错误
        self.assertTrue(failed_guard is not None)
        # 断言错误信息中包含特定字符串，表示设备索引不匹配
        self.assertTrue(
            "tensor 'L['x']' Tensor device index mismatch. Expected device index to be"
            in failed_guard.reason
        )

    def test_unbind(self):
        # 定义函数 fn，对输入张量进行解绑操作
        def fn(a):
            return torch.unbind(a), torch.unbind(a, -1)

        # 调用通用测试函数，验证解绑操作的正确性
        self.common(
            fn,
            (torch.randn([4, 4, 4]),),
        )

    def test_convolution1(self):
        # 创建序列化模型 m，包含卷积层、ReLU 激活函数和 ToTuple 操作
        m = torch.nn.Sequential(
            torch.nn.Conv2d(5, 6, [3, 3]),
            torch.nn.ReLU(),
            ToTuple(),
        )

        # 调用通用测试函数，验证卷积模型 m 的输出
        self.common(
            m,
            (torch.randn([2, 5, 16, 16]),),
            # 允许的绝对误差和相对误差
            atol=6e-5,
            rtol=0.001,
        )

    def test_convolution2(self):
        # 定义函数 fn，实现转置卷积操作
        def fn(x, w, b):
            return (aten.convolution(x, w, b, [4], [0], [1], True, [0], 1),)

        # 调用通用测试函数，验证转置卷积操作的正确性
        self.common(
            fn,
            (
                torch.randn([2, 32, 90]),
                torch.randn([32, 16, 8]),
                torch.randn([16]),
            ),
            check_lowp=False,
        )

    def test_convolution3(self):
        # 创建序列化模型 m，包含指定参数的卷积层、ReLU 激活函数和 ToTuple 操作
        m = torch.nn.Sequential(
            torch.nn.Conv2d(5, 6, [3, 3], stride=[1], padding=[0], dilation=[1]),
            torch.nn.ReLU(),
            ToTuple(),
        )

        # 调用通用测试函数，验证卷积模型 m 的输出
        self.common(
            m,
            (torch.randn([2, 5, 16, 16]),),
            # 允许的绝对误差和相对误差
            atol=6e-5,
            rtol=0.001,
        )

    # https://github.com/halide/Halide/issues/8256
    @config.patch("halide.scheduler_cuda", "Li2018")
    def test_convolution4(self):
        # 定义函数 fn，实现卷积操作并返回结果的和
        def fn(x, w):
            x = F.conv2d(x, w, groups=w.shape[0])
            return x.sum()

        # 调用通用测试函数，验证卷积操作的正确性
        self.common(
            fn,
            (
                torch.randn([2, 3, 16, 20]),
                torch.randn([3, 1, 5, 5]),
            ),
        )
    # 定义一个测试方法，用于测试带有通道最后格式的二维卷积
    def test_convolution5(self):
        # 定义一个内部函数fn，执行二维卷积操作，并返回结果的和
        def fn(x, w):
            x = F.conv2d(x, w, dilation=[x.size(0)])
            return x.sum()

        # 生成一个形状为[2, 1, 16, 20]的随机张量作为输入x
        x = torch.randn([2, 1, 16, 20])
        # 生成一个形状为[1, 1, 5, 5]的随机张量作为卷积核w
        w = torch.randn([1, 1, 5, 5])

        # 对张量x进行动态标记
        torch._dynamo.mark_dynamic(x, 0)

        # 调用测试类中的共用方法common，测试fn函数的输出
        self.common(fn, (x, w))

    # 定义一个测试方法，用于测试带有通道最后格式的二维卷积
    def test_conv2d_channels_last(self):
        # 如果当前设备是GPU，则跳过测试
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("only support cpu conv2d channels_last")

        # 构建一个包含卷积层和ToTuple层的Sequential模型m
        m = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 1, 1),
            ToTuple(),
        )

        # 测试仅权重为通道最后格式的情况
        self.common(
            m.to(memory_format=torch.channels_last),
            (torch.randn([2, 3, 16, 16]),),
            check_lowp=False,
        )

        # 测试仅激活值为通道最后格式的情况
        self.common(
            m,
            (torch.randn([2, 3, 16, 16]).to(memory_format=torch.channels_last),),
            check_lowp=False,
        )

        # 测试激活值和权重均为通道最后格式的情况
        self.common(
            m.to(memory_format=torch.channels_last),
            (torch.randn([2, 3, 16, 16]).to(memory_format=torch.channels_last),),
            check_lowp=False,
        )

    # 定义一个测试方法，用于测试带有通道最后格式的二维卷积反向传播
    def test_conv2d_backward_channels_last(self):
        # 定义一个内部函数fn，执行卷积反向传播操作
        def fn(grad_output, inp, weight):
            convolution_backward_8 = torch.ops.aten.convolution_backward.default(
                grad_output,
                inp,
                weight,
                [320],
                [1, 1],
                [0, 0],
                [1, 1],
                False,
                [0, 0],
                1,
                [True, True, True],
            )
            return convolution_backward_8

        # 测试仅权重为通道最后格式的情况
        self.common(
            fn,
            (
                torch.randn([2, 320, 8, 8]),
                torch.randn([2, 2048, 8, 8]),
                torch.randn([320, 2048, 1, 1]).to(memory_format=torch.channels_last),
            ),
            check_lowp=False,
        )

    # 定义一个测试方法，用于测试带有通道最后格式的三维卷积
    def test_conv3d_channels_last(self):
        # 如果当前设备是GPU，则跳过测试
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("only support cpu conv3d channels_last")

        # 构建一个包含三维卷积层和ToTuple层的Sequential模型m
        m = torch.nn.Sequential(
            torch.nn.Conv3d(3, 3, 1, 1),
            ToTuple(),
        )

        # 测试仅权重为通道最后格式的情况
        self.common(
            m.to(memory_format=torch.channels_last_3d),
            (torch.randn([2, 3, 16, 16, 16]),),
        )

        # 测试仅激活值为通道最后格式的情况
        self.common(
            m,
            (torch.randn([2, 3, 16, 16, 16]).to(memory_format=torch.channels_last_3d),),
        )

        # 测试激活值和权重均为通道最后格式的情况
        self.common(
            m.to(memory_format=torch.channels_last_3d),
            (torch.randn([2, 3, 16, 16, 16]).to(memory_format=torch.channels_last_3d),),
        )

    # 标记为GPU Halide环境下测试缓慢时跳过
    @skip_if_gpu_halide  # slow
    # 定义测试函数 test_adaptive_avg_pool2d1，测试自适应平均池化函数
    def test_adaptive_avg_pool2d1(self):
        # 定义内部函数 fn，对输入 x 执行自适应平均池化操作，返回结果
        def fn(x):
            return aten._adaptive_avg_pool2d(x, (6, 6)), aten._adaptive_avg_pool2d(
                x + 1, (2, 5)
            )

        # 调用通用测试方法 common，测试不同输入下的自适应平均池化
        self.common(
            fn,
            (torch.randn(2, 4, 16, 16),),  # 输入为一个随机张量
            check_lowp=False,  # 关闭低精度检查
        )

        # 降级为 avg_pool2d 情况的测试
        self.common(
            fn,
            (torch.randn(2, 4, 3, 3),),  # 输入为一个较小的随机张量
        )

        # 没有操作的情况
        self.common(
            fn,
            (torch.randn(2, 4, 6, 6),),  # 输入为一个较大的随机张量
        )

    # 定义测试函数 test_adaptive_avg_pool2d2，测试自适应平均池化函数，使用较大的核大小时降级
    def test_adaptive_avg_pool2d2(self):
        # 定义内部函数 fn，对输入 x 执行自适应平均池化操作，返回结果
        def fn(x):
            return aten._adaptive_avg_pool2d(x, (4, 4))

        # 重置生成的内核计数为 0
        torch._inductor.metrics.generated_kernel_count = 0
        # 调用通用测试方法 common，测试较大输入下的自适应平均池化
        self.common(
            fn,
            (torch.randn(2, 4, 21, 21),),  # 输入为一个较大的随机张量
            check_lowp=False,  # 关闭低精度检查
        )
        # 断言生成的内核计数为 0
        assertGeneratedKernelCountEqual(self, 0)

    # 使用 GPU Halide 时跳过的测试函数修饰器
    @skip_if_gpu_halide  # 慢
    # 定义测试函数 test_adaptive_max_pool2d1，测试自适应最大池化函数
    def test_adaptive_max_pool2d1(self):
        # 定义内部函数 fn，对输入 x 执行自适应最大池化操作，返回结果
        def fn(x):
            return aten.adaptive_max_pool2d(x, (6, 6))

        # 调用通用测试方法 common，测试不同输入下的自适应最大池化
        self.common(
            fn,
            (torch.randn(2, 4, 16, 16),),  # 输入为一个随机张量
            check_lowp=False,  # 关闭低精度检查
        )

        # 降级为 max_pool2d 情况的测试
        self.common(
            fn,
            (torch.randn(2, 4, 3, 3),),  # 输入为一个较小的随机张量
        )

        # 没有操作的情况
        self.common(
            fn,
            (torch.randn(2, 4, 6, 6),),  # 输入为一个较大的随机张量
        )

    # 使用 GPU Halide 时跳过的测试函数修饰器
    @skip_if_gpu_halide  # 慢
    # 定义测试函数 test_adaptive_max_pool2d2，测试自适应最大池化函数，使用较大的核大小时降级
    def test_adaptive_max_pool2d2(self):
        # 定义内部函数 fn，对输入 x 执行自适应最大池化操作，返回结果
        def fn(x):
            return aten.adaptive_max_pool2d(x, (4, 4))

        # 重置生成的内核计数为 0
        torch._inductor.metrics.generated_kernel_count = 0
        # 调用通用测试方法 common，测试较大输入下的自适应最大池化
        self.common(
            fn,
            (torch.randn(2, 4, 21, 21),),  # 输入为一个较大的随机张量
            check_lowp=False,  # 关闭低精度检查
        )
        # 断言生成的内核计数为 0
        assertGeneratedKernelCountEqual(self, 0)

    # 定义测试函数 test_fractional_max_pool2d1，测试分数最大池化函数
    def test_fractional_max_pool2d1(self):
        # 定义内部函数 fn，对输入 x 和样本执行分数最大池化操作，返回结果
        def fn(x, samples):
            return aten.fractional_max_pool2d(x, (3, 3), (2, 2), samples)

        # 调用通用测试方法 common，测试不同输入下的分数最大池化
        self.common(
            fn, (torch.randn(1, 4, 16, 16), torch.rand(1, 4, 2)), check_lowp=False
        )

    # 定义测试函数 test_fractional_max_pool2d2，测试分数最大池化函数，使用较大的核大小时降级
    def test_fractional_max_pool2d2(self):
        # 定义内部函数 fn，对输入 x 和样本执行分数最大池化操作，返回结果
        def fn(x, samples):
            return aten.fractional_max_pool2d(x, (6, 5), (3, 3), samples)

        # 重置生成的内核计数为 0
        torch._inductor.metrics.generated_kernel_count = 0
        # 调用通用测试方法 common，测试较大输入下的分数最大池化
        self.common(
            fn,
            (torch.randn(2, 4, 36, 36), torch.rand(2, 4, 2)),  # 输入为一个较大的随机张量和样本
            check_lowp=False,  # 关闭低精度检查
        )
        # 断言生成的内核计数为 0
        assertGeneratedKernelCountEqual(self, 0)

    # 定义测试函数 test_fractional_max_pool2d3，测试分数最大池化函数，使用大样本时的情况
    def test_fractional_max_pool2d3(self):
        # 定义内部函数 fn，对输入 x 和样本执行分数最大池化操作，返回结果
        def fn(x, samples):
            return aten.fractional_max_pool2d(x, (1, 1), (16, 16), samples)

        # 调用通用测试方法 common，测试不同输入下的分数最大池化
        self.common(
            fn, (torch.randn(2, 4, 16, 16), torch.rand(2, 4, 2)), check_lowp=False
        )

    # 使用配置修饰器，启用回退随机化
    @config.patch(fallback_random=True)
    # 使用 Halide 时跳过的测试函数修饰器
    @skip_if_halide  # 仅能展开对常量范围内的循环
    # 定义测试函数：fractional_max_pool2d4，用于测试 torch.nn.functional.fractional_max_pool2d_with_indices 函数
    def test_fractional_max_pool2d4(self):
        # 设置随机种子以保证结果可复现
        random.seed(1234)
        torch.manual_seed(1234)

        # 检查矩形核和输出尺寸的情况

        # 定义内部函数 fn，调用 fractional_max_pool2d_with_indices 函数进行池化操作
        def fn(x):
            return torch.nn.functional.fractional_max_pool2d_with_indices(
                x, (4, 3), (3, 2)
            )

        # 调用共用的测试函数 common 进行测试，传入输入数据和关闭低精度检查
        self.common(fn, (torch.randn(1, 4, 16, 16),), check_lowp=False)

    # 定义测试函数：test_multi_threading，测试多线程模型执行
    def test_multi_threading(self):
        # 创建一个线性模型并设置为评估模式
        model = torch.nn.Linear(2, 3).eval()
        # 创建输入数据
        inp = torch.randn(4, 2)

        # 定义循环次数
        num_run = 3

        # 定义运行权重共享模型的函数
        def run_weights_sharing_model(m, inp):
            # 禁用梯度计算
            with torch.no_grad():
                # 循环执行模型 num_run 次
                for i in range(num_run):
                    y = m(inp)

        # 定义实例数量
        numb_instance = 2
        # 创建线程列表
        threads = []
        # 编译模型
        compiled_m = torch.compile(model)
        # 循环创建并启动线程
        for i in range(1, numb_instance + 1):
            # 创建线程并添加到线程列表
            thread = threading.Thread(
                target=run_weights_sharing_model, args=(compiled_m, inp)
            )
            threads.append(thread)
            thread.start()
        # 等待所有线程执行完成
        for thread in threads:
            thread.join()

    # 定义测试函数：test_adaptive_avg_pool2d_low_prec，测试自适应平均池化在低精度模式下的行为
    @unittest.skipIf(config.is_fbcode(), "fbcode triton error, needs debugging")
    @skip_if_gpu_halide  # https://github.com/halide/Halide/issues/8311
    def test_adaptive_avg_pool2d_low_prec(self):
        # 定义模型类 Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义自适应平均池化层
                self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                # 执行平均池化操作
                x = self.avgpool(x)
                return x

        # 创建模型实例 mod 并将其移到指定设备上进行计算
        mod = Model().to(self.device)
        # 循环测试不同数据类型的情况：torch.half 和 torch.bfloat16
        for dtype in [torch.half, torch.bfloat16]:
            # 创建输入数据 x，并将其转换为指定数据类型
            x = torch.randn(4, 3, 7, 7, device=self.device).to(dtype=dtype)
            # 编译优化模型
            opt_mod = torch.compile(mod)
            # 执行优化模型得到结果
            res = opt_mod(x)
            # 计算期望结果
            expected = mod(x)
            # 使用断言检查结果的接近程度
            self.assertTrue(torch.allclose(res, expected))
    # 定义一个测试方法，用于验证在图中复制缓冲区的行为
    def test_buffer_copied_in_graph(self):
        # 定义一个继承自 torch.nn.Module 的模型类 MyModel
        class MyModel(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 注册一个名为 "buf" 的缓冲区，初始值为零
                self.register_buffer("buf", torch.zeros(1))
                # 定义两个参数 w1 和 w2
                self.w1 = torch.nn.Parameter(torch.zeros(1))
                self.w2 = torch.nn.Parameter(torch.zeros(1))

            # 前向传播方法
            def forward(self, x):
                # 缓冲区 buf 加一
                self.buf.add_(1)
                # 返回 w1 * x * w2 的和，加上 buf 的和
                return (self.w1 * x * self.w2).sum() + self.buf.sum()

        # 创建一个 MyModel 实例，并将其移到指定的设备上
        model_for_eager = MyModel().to(self.device)
        # 深度复制一个 model_for_eager 的实例
        model_for_compile = copy.deepcopy(model_for_eager)

        # 获取 model_for_eager 中所有缓冲区的版本计数器列表
        eager_version_counters = [
            buffer._version for _, buffer in model_for_eager.named_buffers()
        ]
        # 获取 model_for_compile 中所有缓冲区的版本计数器列表
        compile_version_counters = [
            buffer._version for _, buffer in model_for_compile.named_buffers()
        ]

        # 编译 model_for_compile 成为一个函数对象 compiled_f，使用后端 "inductor"
        compiled_f = torch.compile(model_for_compile, backend="inductor")

        # 创建输入张量 inp_ref 和 inp_test，都为形状为 (1,) 的张量，并要求梯度计算
        inp_ref = torch.ones(1, requires_grad=True, device=self.device)
        inp_test = torch.ones(1, requires_grad=True, device=self.device)

        # 使用 model_for_eager 对 inp_ref 和 compiled_f 对 inp_test 进行前向传播
        out_ref = model_for_eager(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())

        # 获取前向传播后，model_for_eager 中所有缓冲区的版本计数器列表
        eager_version_counters_after = [
            buffer._version for _, buffer in model_for_eager.named_buffers()
        ]
        # 获取前向传播后，model_for_compile 中所有缓冲区的版本计数器列表
        compile_version_counters_after = [
            buffer._version for _, buffer in model_for_compile.named_buffers()
        ]

        # 计算前向传播后，model_for_eager 中缓冲区版本计数器列表的变化量
        eager_delta = list(
            map(operator.sub, eager_version_counters_after, eager_version_counters)
        )
        # 计算前向传播后，model_for_compile 中缓冲区版本计数器列表的变化量
        compile_delta = list(
            map(operator.sub, compile_version_counters_after, compile_version_counters)
        )

        # 使用断言验证 model_for_eager 和 compiled_f 在缓冲区版本计数器变化量上的一致性
        self.assertEqual(eager_delta, compile_delta)

    # 给定链接的注释，这行代码指定了使用 Li2018 的 Halide CUDA 调度器
    @config.patch("halide.scheduler_cuda", "Li2018")
    # 定义一个名为 `test_buffer_copied_in_graph_with_different_shapes` 的测试方法
    def test_buffer_copied_in_graph_with_different_shapes(self):
        # 定义一个内部类 `MyModel`，继承自 `torch.nn.Module`
        class MyModel(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 注册一个缓冲区 `buf`，初始为 4x4 的全1张量
                self.register_buffer("buf", torch.ones(4, 4))
                # 定义一个参数 `w`，为一个 4x2 的张量
                self.w = torch.nn.Parameter(
                    torch.Tensor([[4, 5], [1, 2], [6, 7], [8, 9]])
                )

            # 前向传播方法
            def forward(self, x):
                # 缓冲区 `buf` 内的值增加1
                self.buf.add_(1)
                # 返回结果，是权重 `w` 与输入 `x` 的矩阵乘积的和，加上缓冲区 `buf` 的总和
                return (self.w @ x).sum() + self.buf.sum()

        # 创建一个 `MyModel` 类的实例 `model_for_eager`，并移动到指定的设备上
        model_for_eager = MyModel().to(self.device)
        # 使用深度复制创建 `model_for_compile`，保持与 `model_for_eager` 相同的初始状态
        model_for_compile = copy.deepcopy(model_for_eager)

        # 获取 `model_for_eager` 内所有缓冲区的版本计数器列表 `eager_version_counters`
        eager_version_counters = [
            buffer._version for _, buffer in model_for_eager.named_buffers()
        ]
        # 获取 `model_for_compile` 内所有缓冲区的版本计数器列表 `compile_version_counters`
        compile_version_counters = [
            buffer._version for _, buffer in model_for_compile.named_buffers()
        ]

        # 使用 `torch.compile` 编译 `model_for_compile`，使用 "inductor" 后端
        compiled_f = torch.compile(model_for_compile, backend="inductor")

        # 创建一个形状为 (2, 4) 的张量 `inp_ref`，需要梯度，放置在指定设备上
        inp_ref = torch.ones(2, 4, requires_grad=True, device=self.device)
        # 创建一个形状为 (2, 4) 的张量 `inp_test`，需要梯度，放置在指定设备上
        inp_test = torch.ones(2, 4, requires_grad=True, device=self.device)

        # 使用 `model_for_eager` 处理 `inp_ref` 的克隆，得到 `out_ref`
        out_ref = model_for_eager(inp_ref.clone())
        # 使用 `compiled_f` 处理 `inp_test` 的克隆，得到 `out_test`
        out_test = compiled_f(inp_test.clone())

        # 获取处理后的 `model_for_eager` 内所有缓冲区的版本计数器列表 `eager_version_counters_after`
        eager_version_counters_after = [
            buffer._version for _, buffer in model_for_eager.named_buffers()
        ]
        # 获取处理后的 `model_for_compile` 内所有缓冲区的版本计数器列表 `compile_version_counters_after`
        compile_version_counters_after = [
            buffer._version for _, buffer in model_for_compile.named_buffers()
        ]

        # 计算 `eager_version_counters_after` 与 `eager_version_counters` 的差异，存储在 `eager_delta`
        eager_delta = list(
            map(operator.sub, eager_version_counters_after, eager_version_counters)
        )
        # 计算 `compile_version_counters_after` 与 `compile_version_counters` 的差异，存储在 `compile_delta`
        compile_delta = list(
            map(operator.sub, compile_version_counters_after, compile_version_counters)
        )

        # 断言 `eager_delta` 与 `compile_delta` 相等
        self.assertEqual(eager_delta, compile_delta)

    # 跳过此测试，若 `torch.nn.Module` 已内联
    @skipIfNNModuleInlined("https://github.com/pytorch/pytorch/issues/128198")
    def test_buffer_batch_norm(self):
        # 定义一个名为 MyModel 的自定义神经网络模型类
        class MyModel(torch.nn.Module):
            # 初始化方法，继承父类方法
            def __init__(self):
                super().__init__()
                # 添加 BatchNorm1d 层，对输入的维度为 100 的数据进行标准化
                self.m = torch.nn.BatchNorm1d(100)

            # 前向传播方法，定义网络的数据流向
            def forward(self, x):
                return self.m(x)

        # 创建一个实例 model_for_eager，并将其移动到指定的设备上
        model_for_eager = MyModel().to(self.device)
        # 使用深拷贝复制 model_for_eager 到 model_for_compile
        model_for_compile = copy.deepcopy(model_for_eager)

        # 获取 model_for_eager 中所有 buffer 的版本号列表
        eager_version_counters = [
            buffer._version for _, buffer in model_for_eager.named_buffers()
        ]
        # 获取 model_for_compile 中所有 buffer 的版本号列表
        compile_version_counters = [
            buffer._version for _, buffer in model_for_compile.named_buffers()
        ]

        # 对 model_for_compile 进行编译，生成 compiled_f 函数
        compiled_f = torch.compile(model_for_compile, backend="inductor")

        # 创建一个形状为 (20, 100) 的全为 1 的张量，需要梯度，放在指定设备上
        inp_ref = torch.ones(20, 100, requires_grad=True, device=self.device)
        # 创建另一个与 inp_ref 相同的张量 inp_test
        inp_test = torch.ones(20, 100, requires_grad=True, device=self.device)

        # 对 model_for_eager 使用 inp_ref 的克隆进行前向传播，计算输出 out_ref
        out_ref = model_for_eager(inp_ref.clone())
        # 对 compiled_f 使用 inp_test 的克隆进行前向传播，计算输出 out_test
        out_test = compiled_f(inp_test.clone())

        # 获取更新后 model_for_eager 中所有 buffer 的版本号列表
        eager_version_counters_after = [
            buffer._version for _, buffer in model_for_eager.named_buffers()
        ]
        # 获取更新后 model_for_compile 中所有 buffer 的版本号列表
        compile_version_counters_after = [
            buffer._version for _, buffer in model_for_compile.named_buffers()
        ]

        # 计算 model_for_eager buffer 版本号变化量
        eager_delta = list(
            map(operator.sub, eager_version_counters_after, eager_version_counters)
        )
        # 计算 model_for_compile buffer 版本号变化量
        compile_delta = list(
            map(operator.sub, compile_version_counters_after, compile_version_counters)
        )

        # 使用断言验证 eager_delta 与 compile_delta 是否相等
        self.assertEqual(eager_delta, compile_delta)

    # 测试 AdaptiveAvgPool1d 层输出尺寸为 0 的情况
    def test_adaptive_avg_pool_with_output_size_0(self):
        # 创建一个 AdaptiveAvgPool1d 层对象 m1，输出尺寸为 0
        m1 = nn.AdaptiveAvgPool1d(0)
        # 调用 self.common 方法，传入 m1 和一个随机输入的元组
        self.common(m1, (torch.randn(1, 2),))

    # 测试 AdaptiveAvgPool2d 层输出尺寸为 0 的情况
    def test_adaptive_avg_pool_with_output_size_0(self):
        # 创建一个 AdaptiveAvgPool2d 层对象 m2，输出尺寸为 0
        m2 = nn.AdaptiveAvgPool2d(0)
        # 调用 self.common 方法，传入 m2 和一个随机输入的元组
        self.common(m2, (torch.randn(1, 2, 3),))

    # 测试 max_pool2d_with_indices 函数的使用，输入为 2 维张量
    def test_max_pool2d1(self):
        # 定义一个函数 fn，对输入 x 进行 2 维最大池化操作
        def fn(x):
            return aten.max_pool2d_with_indices(x, [3, 3], [2, 2])

        # 调用 self.common 方法，传入 fn 函数和一个随机输入的元组
        self.common(
            fn,
            (torch.randn(2, 4, 16, 16),),
        )

    # 使用 skip_if_gpu_halide 装饰器标记，测试 max_pool2d_with_indices 函数的使用
    def test_max_pool2d2(self):
        # 定义一个函数 fn，对输入 x 进行 2 维最大池化操作
        def fn(x):
            return aten.max_pool2d_with_indices(x, [3, 3], [2, 2])

        # 调用 self.common 方法，传入 fn 函数和一个随机输入的元组
        self.common(
            fn,
            (torch.randn([16, 64, 55, 55]),),
        )

    # 使用 skip_if_gpu_halide 装饰器标记，测试带 padding 的 max_pool2d_with_indices 函数的使用
    def test_max_pool2d3(self):
        # 定义一个函数 fn，对输入 x 进行 2 维带 padding 的最大池化操作
        def fn(x):
            # 添加注释：带填充
            return (
                aten.max_pool2d_with_indices(x, [3, 3], [2, 2], [1, 1]),
                aten.max_pool2d_with_indices(
                    x,
                    [
                        3,
                    ],
                    [
                        2,
                    ],
                    [
                        1,
                    ],
                ),
            )

        # 调用 self.common 方法，传入 fn 函数和一个随机输入的元组
        self.common(
            fn,
            (-torch.arange(1 * 8 * 8, dtype=torch.float32).view(1, 1, 8, 8),),
        )

    # 使用 skip_if_halide 装饰器标记，测试只能展开常数大小循环的情况
    # 定义测试函数 test_max_pool2d4，测试 max_pool2d_with_indices 函数的效果
    def test_max_pool2d4(self):
        # 定义内部函数 fn，使用 aten.max_pool2d_with_indices 进行最大池化操作，包括填充
        def fn(x):
            # 使用 aten.max_pool2d_with_indices 对输入 x 进行最大池化操作，设置核大小为 [3, 3]，步幅为 [2, 2]，填充为 [0, 0]，膨胀为 [1, 1]，启用填充
            return aten.max_pool2d_with_indices(x, [3, 3], [2, 2], [0, 0], [1, 1], True)

        # 调用共用测试方法 self.common 进行测试
        self.common(
            fn,
            (torch.randn([2, 8, 111, 111]),),  # 传入测试参数，此处为一个大小为 [2, 8, 111, 111] 的张量
        )

    # 装饰器，如果使用 GPU Halide，则跳过该测试函数
    @skip_if_gpu_halide  # slow
    def test_max_pool2d5(self):
        # 定义内部函数 fn，使用 aten.max_pool2d_with_indices 进行最大池化操作
        def fn(x):
            # 使用 aten.max_pool2d_with_indices 对输入 x 进行最大池化操作，设置核大小为 [3, 3]，不使用填充
            return aten.max_pool2d_with_indices(x, [3, 3], [])

        # 调用共用测试方法 self.common 进行测试
        self.common(
            fn,
            (torch.randn([16, 64, 55, 55]),),  # 传入测试参数，此处为一个大小为 [16, 64, 55, 55] 的张量
        )

    # 装饰器，如果使用 GPU Halide，则跳过该测试函数
    @skip_if_gpu_halide  # slow
    def test_max_pool2d6(self):
        # 内部函数 fn，使用 aten.max_pool2d_with_indices 进行最大池化操作
        # 核大小过大，使用回退机制
        def fn(x):
            # 使用 aten.max_pool2d_with_indices 对输入 x 进行最大池化操作，设置核大小为 [13, 13]，不使用填充
            return aten.max_pool2d_with_indices(x, [13, 13], [])

        # 重置生成的内核数量为 0
        torch._inductor.metrics.generated_kernel_count = 0
        # 调用共用测试方法 self.common 进行测试
        self.common(
            fn,
            (torch.randn([16, 64, 55, 55]),),  # 传入测试参数，此处为一个大小为 [16, 64, 55, 55] 的张量
        )
        # 断言生成的内核数量与预期相等
        assertGeneratedKernelCountEqual(self, 0)

    # 从 https://github.com/pytorch/pytorch/issues/94775 获取的测试用例
    def test_max_pool2d7(self):
        # 内部函数 fn，使用 torch.nn.functional.max_pool2d 进行最大池化操作
        # 使用 ceil_mode
        def fn(x):
            # 使用 torch.nn.functional.max_pool2d 对输入 x 进行最大池化操作，设置核大小为 1，步幅为 (2, 2)，不使用填充，启用 ceil_mode
            return torch.nn.functional.max_pool2d(
                x, 1, stride=(2, 2), padding=0, ceil_mode=True
            )

        # 调用共用测试方法 self.common 进行测试
        self.common(
            fn,
            (torch.randn([1, 1, 6, 7]),),  # 传入测试参数，此处为一个大小为 [1, 1, 6, 7] 的张量
        )

    # 从 https://github.com/pytorch/pytorch/issues/93384 获取的测试用例
    def test_max_pool2d8(self):
        # 内部函数 fn，使用 aten.max_pool2d_with_indices 进行最大池化操作
        # 膨胀不为 1，使用回退机制
        def fn(x):
            # 使用 aten.max_pool2d_with_indices 对输入 x 进行最大池化操作，设置核大小为 [3, 2]，步幅为 [2, 1]，填充为 [1, 1]，膨胀为 [1, 2]
            return aten.max_pool2d_with_indices(x, [3, 2], [2, 1], [1, 1], [1, 2])

        # 重置生成的内核数量为 0
        torch._inductor.metrics.generated_kernel_count = 0
        # 调用共用测试方法 self.common 进行测试
        self.common(
            fn,
            (torch.randn([2, 2, 3, 6]),),  # 传入测试参数，此处为一个大小为 [2, 2, 3, 6] 的张量
        )
        # 断言生成的内核数量与预期相等
        assertGeneratedKernelCountEqual(self, 0)

    # 定义测试函数 test_avg_pool2d1，测试 avg_pool2d 函数的效果
    def test_avg_pool2d1(self):
        # 内部函数 fn，使用 aten.avg_pool2d 进行平均池化操作
        def fn(x):
            # 使用 aten.avg_pool2d 对输入 x 进行平均池化操作，设置核大小为 [3, 3]，步幅为 [2, 2]
            return aten.avg_pool2d(x, [3, 3], [2, 2])

        # 调用共用测试方法 self.common 进行测试
        self.common(
            fn,
            (torch.randn(2, 4, 16, 16),),  # 传入测试参数，此处为一个大小为 [2, 4, 16, 16] 的张量
        )

    # 定义测试函数 test_avg_pool2d2，测试 avg_pool2d 函数的效果
    def test_avg_pool2d2(self):
        # 内部函数 fn，使用 aten.avg_pool2d 进行平均池化操作
        def fn(x):
            # 使用 aten.avg_pool2d 对输入 x 进行平均池化操作，设置核大小为 [3, 3]，步幅为 [2, 2]
            return aten.avg_pool2d(x, [3, 3], [2, 2])

        # 调用共用测试方法 self.common 进行测试
        self.common(
            fn,
            (torch.randn([16, 64, 55, 55]),),  # 传入测试参数，此处为一个大小为 [16, 64, 55, 55] 的张量
        )

    # 定义测试函数 test_avg_pool2d3，测试 avg_pool2d 函数的效果
    def test_avg_pool2d3(self):
        # 内部函数 fn，使用 aten.avg_pool2d 进行平均池化操作，同时进行多次不同设置的测试
        def fn(x):
            return (
                # 使用 aten.avg_pool2d 对输入 x 进行平均池化操作，设置核大小为 [3, 3]，步幅为 [2, 2]，填充为 [1, 1]
                aten.avg_pool2d(x, [3, 3], [2, 2], [1, 1]),
                # 使用 aten.avg_pool2d 对输入 x 进行平均池化操作，设置核大小为 [3]，步幅为 [2]，填充为 [1]
                aten.avg_pool2d(
                    x,
                    [
                        3,
                    ],
                    [
                        2,
                    ],
                    [
                        1,
                    ],
                ),
            )

        # 调用共用测试方法 self.common 进行测试，并进行额外的低精度检查，若使用的是 Halide 后端则检查低精度地址对齐问题
        self.common(
            fn,
            (-torch.arange(1 * 8 * 8, dtype=torch.float32).view(1, 1, 8, 8),),  #
    # 定义测试函数 test_avg_pool2d5，测试 avg_pool2d 函数对输入张量进行 2D 平均池化操作
    def test_avg_pool2d5(self):
        # 定义内部函数 fn，使用 aten.avg_pool2d 对输入 x 进行 3x3 大小的 2D 平均池化操作
        def fn(x):
            return aten.avg_pool2d(x, [3, 3], [2, 2], [1, 1], count_include_pad=False)

        # 调用 self.common 方法进行通用测试
        self.common(
            fn,
            (-torch.arange(1 * 8 * 8, dtype=torch.float32).view(1, 1, 8, 8),),  # 输入参数为一个张量
            check_lowp=not is_halide_backend(self.device),  # 检查是否是 Halide 后端，用于低精度处理
        )

    # 定义测试函数 test_avg_pool2d6，测试 avg_pool2d 函数对输入张量进行 2D 平均池化操作
    def test_avg_pool2d6(self):
        # 定义内部函数 fn，使用 aten.avg_pool2d 对输入 x 进行 3x3 大小的 2D 平均池化操作，并指定 divisor_override=3
        def fn(x):
            return aten.avg_pool2d(x, [3, 3], [2, 2], [1, 1], divisor_override=3)

        # 调用 self.common 方法进行通用测试
        self.common(
            fn,
            (-torch.arange(1 * 8 * 8, dtype=torch.float32).view(1, 1, 8, 8),),  # 输入参数为一个张量
            check_lowp=not is_halide_backend(self.device),  # 检查是否是 Halide 后端，用于低精度处理
        )

    # 定义测试函数 test_avg_pool2d7，测试 avg_pool2d 函数对输入张量进行 2D 平均池化操作
    def test_avg_pool2d7(self):
        # 使用大的核大小 [13, 13]，因此使用回退机制
        def fn(x):
            return aten.avg_pool2d(x, [13, 13], [1, 1], [0, 0])

        # 设置 torch._inductor.metrics.generated_kernel_count 为 0
        torch._inductor.metrics.generated_kernel_count = 0
        # 调用 self.common 方法进行通用测试
        self.common(
            fn,
            (-torch.arange(1 * 24 * 24, dtype=torch.float32).view(1, 1, 24, 24),),  # 输入参数为一个张量
        )
        # 使用 assertGeneratedKernelCountEqual 方法断言生成的内核数量是否等于 0

    # 定义测试函数 test_avg_pool2d8，测试 avg_pool2d 函数对输入张量进行 2D 平均池化操作
    def test_avg_pool2d8(self):
        # https://github.com/pytorch/pytorch/issues/100987
        def fn(x):
            return aten.avg_pool2d(
                x, kernel_size=3, stride=2, padding=1, ceil_mode=True
            )

        # 调用 self.common 方法进行通用测试
        self.common(
            fn,
            (torch.randn(1, 3, 6, 6),),  # 输入参数为一个随机生成的张量
            check_lowp=not is_halide_backend(self.device),  # 检查是否是 Halide 后端，用于低精度处理
        )

    # 标记跳过 GPU Halide 测试的装饰器
    @skip_if_gpu_halide  # slow
    # 定义测试函数 test_alexnet_prefix，测试模型前向传播的函数
    def test_alexnet_prefix(self):
        # 定义内部函数 forward，使用 torch.ops.aten.convolution、torch.ops.aten.relu 和 torch.ops.aten.max_pool2d_with_indices 进行模拟卷积神经网络前向传播
        def forward(arg6, arg7, arg16):
            convolution = torch.ops.aten.convolution(
                arg16, arg7, arg6, [4, 4], [2, 2], [1, 1], False, [0, 0], 1
            )
            relu = torch.ops.aten.relu(convolution)
            max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices(
                relu, [3, 3], [2, 2]
            )
            getitem = max_pool2d_with_indices[0]
            return (getitem,)

        # 调用 self.common 方法进行通用测试
        self.common(
            forward,
            (
                rand_strided((64,), (1,), torch.float32, "cpu"),  # 随机生成的输入张量
                rand_strided((64, 3, 11, 11), (363, 121, 11, 1), torch.float32, "cpu"),  # 随机生成的输入张量
                rand_strided(
                    (16, 3, 224, 224), (150528, 50176, 224, 1), torch.float32, "cpu"  # 随机生成的输入张量
                ),
            ),
            atol=3e-3,  # 绝对容差
            rtol=2,  # 相对容差
        )

    # 定义测试函数 test_elu，测试 elu 函数对输入张量进行 ELU 激活操作
    def test_elu(self):
        # 定义内部函数 fn，使用 aten.elu 对输入 x 进行 ELU 激活，并将结果加 2；同时对 x+1 进行不同参数的 ELU 激活
        def fn(x):
            return aten.elu(x, 1.6732632423543772, 1.0507009873554805) + 2, aten.elu(
                x + 1, 2, 3, 4
            )

        # 调用 self.common 方法进行通用测试
        self.common(
            fn,
            (torch.randn([16, 16]),),  # 输入参数为一个随机生成的张量
            rtol=1e-4,  # 相对容差
            atol=1e-4,  # 绝对容差
        )
    def test_tan(self):
        # 定义一个函数 fn，计算输入张量 x 的 tan 函数值并返回加 2 后的结果，同时计算 x+1 的 tan 函数值
        def fn(x):
            return aten.tan(x) + 2, aten.tan(x + 1)

        # 调用公共测试函数 common，测试 fn 函数的输出
        self.common(
            fn,
            (torch.randn([16, 16]),),  # 传入一个形状为 [16, 16] 的随机张量作为参数
        )

    def test_tanh(self):
        # 定义一个函数 fn，计算输入张量 x 的 tanh 函数值并返回加 2 后的结果，同时计算 x+1 的 tanh 函数值
        def fn(x):
            return aten.tanh(x) + 2, aten.tanh(x + 1)

        # 调用公共测试函数 common，测试 fn 函数的输出
        self.common(
            fn,
            (torch.randn([16, 16]),),  # 传入一个形状为 [16, 16] 的随机张量作为参数
        )

    @skip_if_halide  # 如果没有实现 lgamma 函数，则跳过此测试
    def test_lgamma(self):
        # 定义一个函数 fn，计算输入张量 x 的 lgamma 函数值并返回加 2 后的结果，同时计算 x+1 的 cos 函数值
        def fn(x):
            return aten.lgamma(x) + 2, aten.cos(x + 1)

        # 调用公共测试函数 common，测试 fn 函数的输出
        self.common(
            fn,
            (torch.randn([16, 16]),),  # 传入一个形状为 [16, 16] 的随机张量作为参数
        )

    def test_cos(self):
        # 定义一个函数 fn，计算输入张量 x 的 cos 函数值并返回加 2 后的结果，同时计算 x+1 的 cos 函数值
        def fn(x):
            return aten.cos(x) + 2, aten.cos(x + 1)

        # 调用公共测试函数 common，测试 fn 函数的输出
        self.common(
            fn,
            (torch.randn([16, 16]),),  # 传入一个形状为 [16, 16] 的随机张量作为参数
        )

    def test_sin(self):
        # 定义一个函数 fn，计算输入张量 x 的 sin 函数值并返回加 2 后的结果，同时计算 x+1 的 sin 函数值
        def fn(x):
            return aten.sin(x) + 2, aten.sin(x + 1)

        # 调用公共测试函数 common，测试 fn 函数的输出
        self.common(
            fn,
            (torch.randn([16, 16]),),  # 传入一个形状为 [16, 16] 的随机张量作为参数
        )

    def test_repeat(self):
        # 定义一个函数 fn，对输入张量 x 进行不同形式的重复操作，并返回重复后的结果元组
        def fn(x):
            return (
                x.repeat(0, 1, 1, 1),     # 在四个维度上进行重复，但每个维度的重复次数都为1
                x.repeat(2, 2, 3, 1),     # 在四个维度上分别重复 2, 2, 3, 1 次
                x.repeat(8, 1, 1, 1),     # 在第一个维度上重复 8 次，其它维度重复 1 次
                x.repeat(2, 1, 1, 1, 1, 1),  # 在六个维度上分别重复 2, 1, 1, 1, 1, 1 次
            )

        # 调用公共测试函数 common，测试 fn 函数的输出
        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]),),  # 传入一个形状为 [1, 2, 4, 8] 的随机张量作为参数
        )

    def test_repeat_as_strided(self):
        # 用于复现问题 #127474

        # 定义一个函数 fn，对输入张量 x 进行重复操作并返回重复后的结果
        def fn(x):
            view_size = (3, 2)
            full = x.repeat((3, 2))  # 在两个维度上分别重复 3, 2 次
            view = torch.as_strided(full, view_size, full.stride())  # 根据 full 张量创建视图 view
            result = view + view  # 将视图 view 的内容加上自身的内容并返回

            return result

        # 调用公共测试函数 common，测试 fn 函数的输出
        self.common(fn, (torch.randn(1, 1),))  # 传入一个形状为 [1, 1] 的随机张量作为参数

    def test_repeat_interleave(self):
        # 定义一个函数 fn，对输入张量 x 进行插值重复操作并返回重复后的结果元组
        def fn(x):
            return (
                x.repeat_interleave(2),         # 沿着默认维度插值重复每个元素 2 次
                x.repeat_interleave(3, dim=0),  # 沿着第 0 维度插值重复每个元素 3 次
                x.repeat_interleave(x.size(1), dim=1),  # 沿着第 1 维度插值重复每个元素 x.size(1) 次
            )

        # 调用公共测试函数 common，测试 fn 函数的输出
        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]),),  # 传入一个形状为 [1, 2, 4, 8] 的随机张量作为参数
        )

    @config.patch(implicit_fallbacks=True)
    def test_repeat_interleave_2(self):
        # 定义一个函数 fn，使用 aten.repeat_interleave.Tensor 对输入张量 x 进行重复插值并返回结果
        def fn(x):
            return torch.ops.aten.repeat_interleave.Tensor(x, output_size=12)

        # 调用公共测试函数 common，测试 fn 函数的输出
        self.common(
            fn,
            (torch.tensor([2, 4, 6]),),  # 传入一个包含数值 [2, 4, 6] 的张量作为参数
        )

    @config.patch(fallback_random=True)
    def test_randn_with_dtype_and_device(self):
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("only support cpu randn_with_dtype_and_device test")

        # 定义一个函数 fn，生成具有指定设备和数据类型的随机张量，并返回加 1 后的结果
        def fn(vectors):
            rotations_shape = (12, vectors.shape[-1], 1, 64)
            random_rotations = torch.randn(
                rotations_shape, device=vectors.device, dtype=vectors.dtype
            )  # 生成具有指定设备和数据类型的随机张量
            random_rotations += 1  # 将随机张量的每个元素加上 1

            return random_rotations

        # 调用公共测试函数 common，测试 fn 函数的输出
        self.common(
            fn,
            (torch.randn([4, 12, 2, 64]),),  # 传入一个形状为 [4, 12, 2, 64] 的随机张量作为参数
        )
    # 定义一个测试函数，用于测试嵌入层的功能
    def test_embedding(self):
        # 创建一个包含嵌入层、ReLU激活函数和ToTuple转换的神经网络模型
        m = torch.nn.Sequential(
            torch.nn.Embedding(10, 4, padding_idx=0),  # 创建一个大小为10x4的嵌入层，其中0为填充索引
            torch.nn.ReLU(),  # ReLU激活函数
            ToTuple(),  # 自定义的转换操作
        )

        # 调用共用测试函数，测试嵌入层模型的输出
        self.common(
            m,
            (torch.randint(10, [2, 8]),),  # 传入模型的输入数据，大小为2x8的随机整数张量
        )

    # 定义一个测试函数，用于测试求均值操作
    def test_mean(self):
        # 定义一个接受输入x并返回多个均值操作结果的函数
        def fn(x):
            return (
                x.mean(),  # 计算所有元素的均值
                x.mean(-1),  # 按最后一个维度计算均值
                torch.mean(x, -2, keepdim=True),  # 在倒数第二个维度计算均值，并保持维度
                x.mean([0, 1]),  # 在指定维度上计算均值
            )

        # 调用共用测试函数，测试均值函数的输出
        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]),),  # 传入测试函数的输入数据，大小为1x2x4x8的随机张量
        )

    # 定义一个测试函数，用于测试同时求方差和均值的操作
    def test_var_mean(self):
        # 定义一个接受输入x并返回方差和均值的元组的函数
        def fn(x):
            return (
                *torch.var_mean(x, -1),  # 在最后一个维度上同时计算方差和均值
                *torch.var_mean(x, [1, 3]),  # 在指定的多个维度上同时计算方差和均值
            )

        # 调用共用测试函数，测试方差和均值函数的输出
        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]),),  # 传入测试函数的输入数据，大小为1x2x4x8的随机张量
        )

    # 定义一个测试函数，用于测试带有修正值的方差计算
    def test_var_correction(self):
        # 定义一个接受输入x并返回带有不同修正值的方差的函数
        def fn(x):
            dim = -1
            return (
                torch.var(x, dim=dim, correction=1.3),  # 使用指定的修正值计算方差
                torch.var(x, dim=dim, correction=3),  # 使用指定的修正值计算方差
                torch.var(x, dim=dim, correction=10),  # 使用指定的修正值计算方差
            )

        # 调用共用测试函数，测试带有不同修正值的方差计算
        self.common(fn, (torch.randn([2, 8]),))  # 传入测试函数的输入数据，大小为2x8的随机张量
        # 在更小的维度上进行展开的情况下再次调用测试函数
        self.common(fn, (torch.randn([2, 4]),))  # 传入测试函数的输入数据，大小为2x4的随机张量

    # 标记装饰器，配置指定的测试环境选项
    @config.patch(pick_loop_orders=True)
    # 定义一个测试函数，用于测试转置操作的传播性
    def test_transposed_propagates(self):
        # 定义一个使用特定优化器进行加速的函数
        @torch._dynamo.optimize("inductor", nopython=True)
        def fn(x, y):
            return x + y

        # 创建两个张量a和b，对张量a进行维度置换
        a = torch.randn(1, 4, 4, 4, device=self.device).permute(0, 2, 3, 1)
        b = torch.randn(4, 4, 4, device=self.device).permute(1, 2, 0)
        # 调用特定优化函数处理a和b，并获取结果张量c
        c = fn(a, b)
        # 断言a张量和c张量的步长相等
        self.assertEqual(a.stride(), c.stride())
        # 断言c张量在第2个维度上的步长为1
        self.assertEqual(c.stride()[2], 1)

    # 标记装饰器，配置指定的测试环境选项
    @config.patch("halide.scheduler_cuda", "Li2018")
    # 定义一个测试函数，用于测试标准差计算
    def test_std(self):
        # 定义一个接受输入x并返回多个标准差操作结果的函数
        def fn(x):
            return (
                torch.var(x, True),  # 计算所有元素的方差
                torch.var(x, False),  # 计算所有元素的方差（无偏估计）
                torch.var(x, -1, True),  # 按最后一个维度计算方差，并保持维度
                torch.var(x, -1, False),  # 按最后一个维度计算方差（无偏估计）
                torch.std(x, False),  # 计算所有元素的标准差
                torch.std(x, [0, 1], True),  # 在指定维度上计算标准差，并保持维度
                torch.std(x, [0, 1], False),  # 在指定维度上计算标准差（无偏估计）
                torch.std(x, -2, True, keepdim=True),  # 按倒数第二个维度计算标准差，并保持维度
            )

        # 调用共用测试函数，测试标准差函数的输出
        self.common(
            fn,
            (torch.randn([2, 4, 4, 8]),),  # 传入测试函数的输入数据，大小为2x4x4x8的随机张量
        )

    # 定义一个测试函数，用于测试嵌入包操作
    def test_embedding_bag(self):
        # 定义一个接受权重、索引和偏置张量并返回嵌入包操作结果的函数
        def fn(w, i, o):
            return aten._embedding_bag(w, i, o, False, 0, False, None)

        # 调用共用测试函数，测试嵌入包函数的输出
        self.common(
            fn,
            (
                torch.randn([10, 4]),  # 权重张量，大小为10x4的随机张量
                torch.randint(10, [8]),  # 索引张量，大小为8的随机整数张量
                torch.tensor([0, 2, 6]),  # 偏置张量，包含值为0、2和6的张量
            ),
        )

    # 定义一个测试函数，用于测试二维批量归一化操作
    def test_batch_norm_2d(self):
        # 创建一个包含二维批量归一化和ReLU激活函数的神经网络模型
        m = torch.nn.Sequential(
            torch.nn.BatchNorm2d(10),  # 创建一个二维批量归一化层，通道数为10
            torch.nn.ReLU(),  # ReLU激活函数
        )
        m.eval()  # 将模型设置为评估模式
        # 调用共用测试函数，测试二维批量归一化模型的输出，不检查低精度匹配类型
        self.common(m
    # 测试 2D 批归一化函数
    def test_batch_norm_2d_2(self):
        # 如果设备是 CPU，则跳过测试，显示所需的 GPU 类型
        if self.device == "cpu":
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        # 定义一个简单的神经网络模型
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 第一层卷积层，输入通道数 64，输出通道数 128，卷积核大小为 3x3，步长为 2x2，填充为 1x1，无偏置
                self.self_0 = torch.nn.Conv2d(
                    64,
                    128,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                )
                # 第二层 2D 批归一化层，输入通道数 128，epsilon 设置为 0.0001，动量为 0.03，可调整参数，跟踪运行统计信息
                self.self_1 = torch.nn.BatchNorm2d(
                    128,
                    eps=0.0001,
                    momentum=0.03,
                    affine=True,
                    track_running_stats=True,
                )
                # 第三层 LeakyReLU 激活函数，负斜率设置为 0.1，在原地执行
                self.self_2 = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

            # 前向传播函数
            def forward(self, l_input_: torch.Tensor):
                # 执行第一层卷积操作
                self_0 = self.self_0(l_input_)
                # 执行第二层批归一化操作
                self_1 = self.self_1(self_0)
                # 执行第三层 LeakyReLU 激活操作
                self_2 = self.self_2(self_1)
                # 返回结果元组
                return (self_2,)

        # 生成输入张量，形状为 (4, 64, 192, 256)，数据类型为 torch.float32，设备类型为 GPU_TYPE
        inp = torch.randn((4, 64, 192, 256), dtype=torch.float32, device=GPU_TYPE)
        # 创建 Repro 模型实例，并将其移动到 GPU_TYPE 设备上
        mod = Repro().to(device=GPU_TYPE)
        # 使用模型进行前向传播，获取输出 o1
        o1 = mod(inp)
        # 使用 torch.compile 编译模型并进行前向传播，获取输出 o2
        o2 = torch.compile(mod)(inp)
        # 使用断言检查 o1 和 o2 是否近似相等，相对和绝对误差容差均为 1e-3
        self.assertEqual(o1, o2, rtol=1e-3, atol=1e-3)

    # 测试 LayerNorm 层
    @patch.object(config.trace, "enabled", True)
    def test_layer_norm(self):
        # 创建包含 LayerNorm 和 ReLU 的序列模型
        m = torch.nn.Sequential(
            torch.nn.LayerNorm(32),
            torch.nn.ReLU(),
        )
        # 将模型设置为评估模式
        m.eval()
        # 使用 torch.no_grad 上下文，禁用梯度计算，执行 self.common 函数
        with torch.no_grad():
            self.common(m, (torch.randn([16, 32]),), check_lowp=False)
        # 如果设备不是 CPU，则断言生成的内核数等于 1
        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 1)

    # 测试转置和加法操作
    def test_transpose_add(self):
        # 定义一个函数 fn，执行矩阵转置和加法操作
        def fn(a, b):
            return a.t() + b

        # 使用 self.common 函数，传入随机生成的张量作为参数，执行操作
        self.common(
            fn, (torch.randn([16, 32]), torch.randn([32, 16])), check_lowp=False
        )
        # 如果设备不是 CPU，则断言生成的内核数等于 1
        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 1)

    # 测试 softmax 函数，使用单个内核持久化方法
    @patch.object(config.triton, "persistent_reductions", True)
    def test_softmax_one_kernel_persist(self):
        # 定义一个函数 fn，执行 softmax 计算
        def fn(x):
            dim = 1
            x_max = torch.amax(x, dim, keepdim=True)
            unnormalized = torch.exp(x - x_max)
            result = unnormalized / torch.sum(unnormalized, dim, keepdim=True)
            return result

        # 使用 self.common 函数，传入随机生成的张量作为参数，执行操作
        self.common(fn, (torch.randn([16, 32]),), check_lowp=False)
        # 如果设备不是 CPU，则断言生成的内核数等于 1
        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 1)

    # 测试 softmax 函数，使用单个内核循环方法
    @patch.object(config.triton, "persistent_reductions", False)
    def test_softmax_one_kernel_loop(self):
        # 定义一个函数 fn，执行 softmax 计算
        def fn(x):
            x_max = torch.amax(x, 1, keepdim=True)
            unnormalized = torch.exp(x - x_max)
            result = unnormalized / torch.sum(unnormalized, 1, keepdim=True)
            return result

        # 使用 self.common 函数，传入随机生成的张量作为参数，执行操作
        self.common(fn, (torch.randn([16, 32]),), check_lowp=False)
        # 如果设备不是 CPU，则断言生成的内核数等于 1
        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 1)
    @skip_if_gpu_halide  # 如果在GPU上使用Halide加速，则跳过测试（因为CUDA上结果不正确）
    def test_complex_fallback(self):
        def fn(x):
            return x * x + 10  # 对输入张量进行复数运算，返回结果

        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]).to(dtype=torch.complex64),),  # 调用通用测试函数，传入复数张量作为参数
        )
        assertGeneratedKernelCountEqual(self, 0)  # 断言生成的内核数为0

        class ToComplex(nn.Module):
            def forward(self, x):
                return (x + x + 12).to(torch.complex64)  # 将输入张量转换为复数类型并返回

        self.common(ToComplex(), (torch.rand([1, 2, 4, 8]),), check_lowp=False)  # 调用通用测试函数，传入ToComplex实例作为模块，忽略低精度检查

        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 1)  # 断言生成的内核数为1（仅在非CPU设备上执行）

    def test_view_as_complex(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, view_2):
                clone = torch.ops.aten.clone.default(
                    view_2, memory_format=torch.contiguous_format
                )  # 克隆输入张量，并设置内存格式为连续
                view_2 = None  # 清空原输入张量
                view_as_complex = torch.ops.aten.view_as_complex.default(clone)  # 将克隆的张量视为复数形式
                clone = None  # 清空克隆的张量
                return (view_as_complex,)  # 返回视为复数形式的张量元组

        inp = torch.empty_strided((128, 64, 12, 32, 2), (1, 98304, 8192, 256, 128)).to(
            self.device
        )  # 创建指定形状和步幅的张量，并移动到指定设备
        mod = Repro()  # 创建Repro模块实例

        o1 = mod(inp)  # 使用模块处理输入张量
        o2 = torch.compile(mod)(inp)  # 使用编译后的模块处理输入张量

        self.assertEqual(o1, o2)  # 断言o1和o2相等

    def test_view_as_real(self):
        def fn(x):
            y = torch.view_as_real(x)  # 返回输入张量的实部和虚部视图
            return y + 1  # 返回实部和虚部视图加1后的结果

        x = torch.randn(4, dtype=torch.complex64)  # 创建复数张量x

        self.common(fn, (x,))  # 调用通用测试函数，传入复数张量x作为参数

    def test_polar(self):
        def fn(dist, angle):
            return torch.polar(dist, angle)  # 返回极坐标形式的张量

        inp = (
            torch.tensor([1, 2], dtype=torch.float64),  # 创建浮点数张量dist
            torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64),  # 创建浮点数张量angle
        )
        self.common(fn, (*inp,))  # 调用通用测试函数，传入浮点数张量inp作为参数

    @skip_if_gpu_halide  # 如果在GPU上使用Halide加速，则跳过测试（因为CUDA上地址对齐错误）
    def test_cauchy(self):
        def fn(x, y):
            return torch.sum(1 / (torch.unsqueeze(x, -1) - y))  # 返回Cauchy分布相关计算的结果

        self.common(
            fn,
            (
                torch.randn(32),  # 创建长度为32的随机张量x
                torch.randn(32),  # 创建长度为32的随机张量y
            ),
            atol=5 * 1e-4,  # 绝对误差允许范围
            rtol=5 * 1e-5,  # 相对误差允许范围
            check_lowp=False,  # 不进行低精度检查
        )
        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 1)  # 断言生成的内核数为1（仅在非CPU设备上执行）

    @skip_if_gpu_halide  # 如果在GPU上使用Halide加速，则跳过测试（因为CUDA上内存写入与读取不一致）
    def test_fusing_write_into_disjoint_read(self):
        def test_flip(a):
            return a.copy_(torch.flip(a, (0,)))  # 执行张量a的翻转操作

        self.common(test_flip, (torch.rand([20]),))  # 调用通用测试函数，传入长度为20的随机张量a作为参数

        assertGeneratedKernelCountEqual(self, 2)  # 断言生成的内核数为2

        # 仅在CUDA设备上，使用大张量时出现问题
        if self.device != "cpu":

            def f(a):
                a[:, 20:40] = a[:, 20:40] + 1  # 对张量a的指定切片区域进行加法操作
                a[:, 2:900025] = a[:, 1:900024] + 2  # 对张量a的指定切片区域进行加法操作

            a = torch.rand((1, 1000000), device=self.device)  # 创建指定设备上的大张量a
            self.common(f, (a,))  # 调用通用测试函数，传入大张量a作为参数
    # 定义一个测试函数 test_gather_scatter，用于测试节点和边特征的收集与分散操作
    def test_gather_scatter(self):
        # 定义内部函数 fn，接收节点特征和边索引作为参数
        def fn(node_feat, edge_index):
            # 根据边索引从节点特征中收集源节点特征和目标节点特征
            src_node_feat = node_feat[edge_index[0]]
            dst_node_feat = node_feat[edge_index[1]]
            # 计算边特征为源节点特征减去目标节点特征再加一
            edge_feat = src_node_feat - dst_node_feat + 1
            # 创建与节点特征相同大小的全零张量 new_node_feat
            new_node_feat = torch.zeros_like(node_feat)
            # 将边特征按照扩展的边索引添加到 new_node_feat 上
            new_node_feat.scatter_add_(
                0, edge_index[1].unsqueeze(-1).expand_as(edge_feat), edge_feat
            )
            # 返回更新后的节点特征 new_node_feat
            return new_node_feat

        # 设定节点数和特征数
        num_nodes = 16
        num_features = 32
        # 生成随机节点特征和随机边索引
        node_feat = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, size=(2, num_nodes * 5))
        # 调用 self.common 方法测试 fn 函数
        self.common(
            fn,
            (
                node_feat,
                edge_index,
            ),
            check_lowp=False,
        )
        # 如果设备不是 CPU，断言生成的内核数为 2
        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 2)

    # 使用 config.patch 装饰器设定最大融合大小为 1 的测试函数 test_no_mega_fusion_during_lowering
    @config.patch(max_fusion_size=1)
    def test_no_mega_fusion_during_lowering(self):
        # 定义函数 fn，接收任意数量的参数
        n = 50
        def fn(*args):
            # 初始化 x 为第一个参数 args[0]
            x = args[0]
            # 循环 n 次，每次将 x 与 args[i] 相加
            for i in range(n):
                x = torch.add(x, args[i])
            # 返回相加后的结果 x
            return x

        # 使用 self.common 方法测试 fn 函数，传入包含 n 个随机张量的列表作为参数
        self.common(
            fn,
            [torch.randn(64) for _ in range(n)],
            check_lowp=False,
        )
        # 打印生成的内核数目前缀，输出 torch._inductor.metrics.generated_kernel_count
        print("-->", torch._inductor.metrics.generated_kernel_count)
        # 如果设备不是 CPU，断言生成的内核数目大于 1
        if self.device != "cpu":
            self.assertTrue(torch._inductor.metrics.generated_kernel_count > 1)

    # 定义测试函数 test_move_arange
    def test_move_arange(self):
        # 定义函数 fn，接收一个参数 x
        def fn(x):
            # 创建一个从 0 到 x 长度的张量，并指定在 CPU 上，然后将其转移到 x 所在的设备上再加 x
            return torch.arange(len(x), device="cpu").to(x.device) + x

        # 使用 self.common 方法测试 fn 函数，传入一个包含一个随机张量的元组作为参数
        self.common(fn, (torch.randn([32]),), check_lowp=False)
        # 如果存在复制操作，则断言生成的内核数为 1
        assertGeneratedKernelCountEqual(self, 1)

    # 定义测试函数 test_leaky_relu
    def test_leaky_relu(self):
        # 定义函数 fn，接收一个参数 x
        def fn(x):
            # 使用 aten.leaky_relu 对 x 进行操作，施加 0.2 的斜率，并在结果上加 2，
            # 同时对 x + 1 也应用 leaky_relu 操作
            return aten.leaky_relu(x, 0.2) + 2, aten.leaky_relu(x + 1)

        # 使用 self.common 方法测试 fn 函数，传入一个包含一个随机张量的元组作为参数
        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    # 定义测试函数 test_gelu
    def test_gelu(self):
        # 定义函数 fn，接收一个参数 x
        def fn(x):
            # 使用 aten.gelu 对 x 进行操作，并在结果上加 2，
            # 同时对 x + 1 也应用 gelu 操作
            return aten.gelu(x) + 2, aten.gelu(x + 1)

        # 使用 self.common 方法测试 fn 函数，传入一个包含一个随机张量的元组作为参数
        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    # 定义测试函数 test_clone
    def test_clone(self):
        # 定义函数 fn，接收一个参数 x
        def fn(x):
            # 使用 aten.clone 对 x 进行操作，并在结果上加 2，
            # 同时对 x + 1 也应用 clone 操作
            return aten.clone(x) + 2, aten.clone(x + 1)

        # 使用 self.common 方法测试 fn 函数，传入一个包含一个随机张量的元组作为参数
        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    # 定义测试函数 test_masked_fill
    def test_masked_fill(self):
        # 定义函数 fn，接收两个参数 mask 和 value
        def fn(mask, value):
            # 使用 aten.masked_fill 对 value 应用 mask，将未被 mask 的位置置为 -10000.0 并加 2，
            # 同时对 value / 2.0 使用逻辑非 mask，并将未被 mask 的位置置为 667
            return aten.masked_fill(value, mask, -10000.0) + 2, aten.masked_fill(
                value / 2.0, torch.logical_not(mask), 667
            )

        # 使用 self.common 方法测试 fn 函数，传入一个包含一个布尔类型张量和一个随机张量的元组作为参数
        self.common(
            fn,
            (
                torch.randint(0, 1, [1, 16], dtype=torch.bool),
                torch.randn([16, 16]),
            ),
        )
    def test_masked_fill_promotion(self):
        # 定义一个内部函数 fn，接受 mask 和 value 两个参数，将 value 中的 mask 部分替换为 torch.tensor(3.5)
        def fn(mask, value):
            return aten.masked_fill(value, mask, torch.tensor(3.5))

        # 使用 torch._dynamo.optimize("inductor") 对 fn 进行优化
        opt_fn = torch._dynamo.optimize("inductor")(fn)
        
        # 遍历输入数据 inp
        for inp in (
            torch.randn(
                [16, 16],
                dtype=torch.float16 if self.device == GPU_TYPE else torch.float32,
                device=self.device,
            ),
            torch.randint(16, (16, 16), device=self.device),
        ):
            # 构建输入 inputs，包含一个随机生成的布尔类型的 mask 和 inp
            inputs = (
                torch.randint(0, 1, [1, 16], dtype=torch.bool, device=self.device),
                inp,
            )
            # 断言 fn 和 opt_fn 对相同输入的输出相等
            self.assertEqual(fn(*inputs), opt_fn(*inputs))

    def test_masked_scatter(self):
        # 定义一个内部函数 fn，接受 value、mask 和 source 三个参数，使用 source 对 value 中的 mask 部分进行替换
        def fn(value, mask, source):
            return torch.masked_scatter(value, mask, source)

        # 生成三个张量 value、mask 和 source，并调用 common 方法进行测试
        value = make_tensor(10, 10, dtype=torch.float32, device=self.device)
        mask = make_tensor(10, 10, dtype=torch.bool, device=self.device)
        source = make_tensor(
            mask.count_nonzero(), dtype=torch.float32, device=self.device
        )

        self.common(fn, (value, mask, source))

    def test_fill1(self):
        # 定义一个内部函数 fn，接受输入张量 x，生成一个与 x 同形状的全为 1 的 tmp 张量，并返回 tmp 和使用 tmp 填充 x 的结果
        def fn(x):
            tmp = torch.ones_like(x)
            return tmp, aten.fill.Scalar(tmp, 2)

        # 调用 common 方法对 fn 进行测试，传入一个随机生成的张量作为输入
        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_fill2(self):
        # 定义一个内部函数 fn，接受输入张量 x，生成一个与 x 同形状的全为 1 的 tmp 张量，并返回 tmp 和使用标量 3.0 填充 tmp 的结果
        def fn(x):
            tmp = torch.ones_like(x)
            return tmp, aten.fill.Tensor(tmp, torch.tensor(3.0))

        # 调用 common 方法对 fn 进行测试，传入一个随机生成的张量作为输入
        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_pow1(self):
        # 定义一个内部函数 fn，接受输入张量 x，返回 x 的不同指数幂的列表
        def fn(x):
            return [aten.pow(x, e) for e in range(-8, 9)]

        # 调用 common 方法对 fn 进行测试，传入一个随机生成的张量作为输入
        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_pow2(self):
        # 定义一个内部函数 fn，接受输入张量 x，返回使用 x 的不同指数幂计算的结果
        def fn(x):
            return aten.pow(1000, x), aten.pow(x, 1000)

        # 调用 common 方法对 fn 进行测试，传入一个随机生成的张量作为输入，并设置绝对误差 atol 和相对误差 rtol 的值
        self.common(
            fn,
            (
                torch.randn(
                    [16, 16],
                    dtype=torch.float32,
                ),
            ),
            atol=1e-5,
            rtol=3e-05,
        )

    @skip_if_gpu_halide  # https://github.com/halide/Halide/issues/8318
    @config.patch("halide.scheduler_cuda", "Li2018")
    def test_pow3(self):
        # 定义一个内部函数 fn，接受输入张量 x，计算 x + 0.123 后的平方根
        def fn(x):
            z = torch.tensor(0.123, device=self.device)
            w = z + x
            return torch.pow(w, 0.5)

        # 使用 torch._dynamo.optimize("inductor") 对 fn 进行优化
        opt = torch._dynamo.optimize("inductor")(fn)
        # 生成一个随机数作为输入，并使用 assertTrue 断言优化后的结果与原函数的结果相同
        input = torch.rand((), device=self.device)
        self.assertTrue(same(opt(input), fn(input)))
    # 定义测试函数 test_pow_int，用于测试 torch.pow 函数在整数类型数据上的表现
    def test_pow_int(self):
        # 定义内部函数 fn，接受两个参数 x 和 y，返回 torch.pow(x, 0x57) 和 torch.pow(x, y) 的结果
        def fn(x, y):
            return torch.pow(x, 0x57), torch.pow(x, y)

        # 遍历整数类型的 tensor 数据类型
        for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # 获取当前数据类型的最大值
            intmax = torch.iinfo(dtype).max
            # 创建 tensor 的部分函数 make_arg，指定 dtype、device 和 requires_grad 参数
            make_arg = functools.partial(
                make_tensor, dtype=dtype, device=self.device, requires_grad=False
            )
            # 调用通用测试方法 self.common，传入 fn 函数和不同参数组合
            self.common(
                fn,
                (
                    make_arg(16, 16),  # 创建 dtype 数据类型的 tensor，元素值为 16
                    make_arg(16, 16, high=intmax),  # 创建 dtype 数据类型的 tensor，元素值在 [0, intmax] 之间
                ),
            )

    # 定义测试函数 test_glu，用于测试 aten.glu 函数在不同参数下的表现
    def test_glu(self):
        # 定义内部函数 fn，接受参数 x，返回 aten.glu(x, -1)、aten.glu(x, 1) 和 aten.glu(x, 2) 的结果
        def fn(x):
            return aten.glu(x, -1), aten.glu(x, 1), aten.glu(x, 2)

        # 调用通用测试方法 self.common，传入 fn 函数和不同参数组合
        self.common(
            fn,
            (torch.randn([8, 16, 8, 8]),),  # 创建大小为 [8, 16, 8, 8] 的随机 tensor
        )

    # 使用 torch._dynamo.config.patch 进行配置，在测试非零元素细化时可能出现的情况
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_nonzero_unbacked_refinement(self):
        # 定义内部函数 fn，接受参数 x，获取 x 中非零元素，检查其数量是否为 4，并返回 z + 3 的结果
        def fn(x):
            z = x.nonzero()  # 获取 x 中非零元素的索引
            torch._check(z.size(0) == 4)  # 检查非零元素的数量是否为 4
            return z + 3

        # 调用通用测试方法 self.common，传入 fn 函数和不同参数组合
        self.common(
            fn,
            (torch.tensor([0, 1, 3, 4, 2, 0, 0]),),  # 创建包含不同元素的 tensor
        )

        # 在运行时捕获 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            torch.compile(fn)(torch.tensor([0, 0, 0, 0]))  # 对给定 tensor 进行编译测试

    # 使用 torch._dynamo.config.patch 进行配置，在测试整除简化时可能出现的情况
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_floordiv_simplify(self):
        # 定义内部函数 fn，接受参数 x 和 y，获取 y 的标量值 z，检查 z // 2 是否等于 3，并返回 x + x.new_zeros(z) 的结果
        def fn(x, y):
            z = y.item()  # 获取 y 的标量值
            torch._check(z // 2 == 3)  # 检查 z // 2 是否等于 3
            return x + x.new_zeros(z)  # 返回 x 和形状与 z 相同的零 tensor 的和

        # 调用通用测试方法 self.common，传入 fn 函数和不同参数组合
        self.common(
            fn,
            (
                torch.randn(6),  # 创建大小为 6 的随机 tensor
                torch.tensor([6]),  # 创建包含单个元素 6 的 tensor
            ),
        )

        # 调用通用测试方法 self.common，传入 fn 函数和不同参数组合
        self.common(
            fn,
            (
                torch.randn(7),  # 创建大小为 7 的随机 tensor
                torch.tensor([7]),  # 创建包含单个元素 7 的 tensor
            ),
        )

    # 使用 torch._dynamo.config.patch 进行配置，在测试整除简化时可能出现的错误情况
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_floordiv_simplify_errors(self):
        # 定义内部函数 fn，接受参数 x 和 y，获取 y 的标量值 z，检查 z // 2 是否等于 3，并返回 x + x.new_zeros(z) 的结果
        def fn(x, y):
            z = y.item()  # 获取 y 的标量值
            torch._check(z // 2 == 3)  # 检查 z // 2 是否等于 3
            return x + x.new_zeros(z)  # 返回 x 和形状与 z 相同的零 tensor 的和

        # 在运行时捕获 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            torch.compile(fn)(torch.randn(8), torch.tensor(8))  # 对给定 tensor 进行编译测试

    # 定义测试函数 test_cat，用于测试 torch.cat 函数在不同情况下的表现
    def test_cat(self):
        # 定义内部函数 fn，接受参数 a，对 a 进行操作并返回三种不同的 torch.cat 结果
        def fn(a):
            tmp = a * 2  # 将 a 中的元素乘以 2，存储在 tmp 中
            return (
                torch.cat((a, a[:, :4] + 1, a + 2), -1),  # 在最后一个维度上连接三个 tensor
                torch.cat((tmp, tmp), 0),  # 沿第一个维度连接两个相同的 tensor
                torch.cat((tmp, tmp.double()), 0),  # 将 tmp 和 tmp 转换为 double 类型后沿第一个维度连接
            )

        # 调用通用测试方法 self.common，传入 fn 函数和不同参数组合
        self.common(
            fn,
            (torch.randn([8, 16]),),  # 创建大小为 [8, 16] 的随机 tensor
        )
        # 调用通用测试方法 self.common，传入 fn 函数和不同参数组合，同时指定 memory_format 为 torch.channels_last
        self.common(
            fn,
            (torch.randn([1, 3, 3, 16]).to(memory_format=torch.channels_last),),
        )

    # 定义测试函数 test_cat_uint8，用于测试 torch.cat 函数在 uint8 类型数据上的表现
    def test_cat_uint8(self):
        # 定义内部函数 fn，接受参数 x，获取 x 的批次形状，将 x 增加一个元素后在最后一个维度上连接
        def fn(x):
            batch_shape = x.shape[:1]  # 获取 x 的批次形状
            out = torch.cat([x.new_zeros(1).expand(batch_shape + (1,)), x], dim=-1)  # 在最后一个维度上连接 x 和扩展后的零 tensor
            return out

        # 调用通用测试方法 self.common，传入 fn 函数和不同参数组合
        self.common(
            fn,
            (torch.randint(0, 256, size=(3, 255), dtype=torch.uint8),),  # 创建大小为 [3, 255] 的随机 uint8 类型 tensor
        )
    # 定义一个测试方法，用于测试 torch.cat 处理空张量的情况
    def test_cat_empty(self):
        # 定义一个函数 fn_2，接收任意数量的张量，并执行 torch.cat 操作
        def fn_2(*tensors):
            return torch.cat(tensors)

        # 调用通用测试方法 common，测试 fn_2 处理一个非空张量和一个空张量的情况
        self.common(
            fn_2,
            (
                torch.randn([1, 3, 3, 16]),  # 生成一个形状为 [1, 3, 3, 16] 的随机张量
                torch.ones([0]),            # 生成一个大小为 0 的全 1 张量
            ),
        )

        # 再次调用 common 测试方法，测试 fn_2 处理两个非空张量的情况
        self.common(
            fn_2,
            (
                torch.randn([1, 3, 3, 16]),  # 生成一个形状为 [1, 3, 3, 16] 的随机张量
                torch.ones([0]),            # 生成一个大小为 0 的全 1 张量
                torch.randn([1, 3, 3, 16]),  # 又生成一个形状为 [1, 3, 3, 16] 的随机张量
            ),
        )

        # 再次调用 common 测试方法，测试 fn_2 处理一个空张量和一个非空张量的情况
        self.common(
            fn_2,
            (
                torch.ones([0]),            # 生成一个大小为 0 的全 1 张量
                torch.randn([1, 3, 3, 16]),  # 生成一个形状为 [1, 3, 3, 16] 的随机张量
            ),
        )

    # 带有装饰器 @torch._dynamo.config.patch 的测试方法，用于测试 torch.cat 处理空张量的情况（遗留版本）
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_cat_unbacked_legacy_empty(self):
        # 定义一个函数 fn，接收两个张量 x 和 y，对 y 进行 item 操作，并执行 torch.cat 操作
        def fn(x, y):
            z = y.item()                      # 获取张量 y 的标量值
            return torch.cat([x, x.new_ones(z)])  # 将张量 x 与一个新的形状为 (z,) 的全 1 张量连接起来

        # 调用通用测试方法 common，测试 fn 处理一个形状为 [2, 3] 的随机张量和一个标量为 0 的张量的情况
        self.common(
            fn,
            (
                torch.randn([2, 3]),  # 生成一个形状为 [2, 3] 的随机张量
                torch.tensor([0]),    # 生成一个标量为 0 的张量
            ),
        )

    # 带有装饰器 @torch._dynamo.config.patch 的测试方法，用于测试 torch.cat 处理空张量的情况（1维情况）
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_cat_unbacked_empty_1d(self):
        # 定义一个函数 fn，接收两个张量 x 和 y，对 y 进行 item 操作，并执行 torch.cat 操作
        def fn(x, y):
            z = y.item()                      # 获取张量 y 的标量值
            return torch.cat([x, x.new_ones(z)])  # 将张量 x 与一个新的形状为 (z,) 的全 1 张量连接起来

        # 调用通用测试方法 common，测试 fn 处理一个形状为 [2] 的随机张量和一个标量为 0 的张量的情况
        self.common(
            fn,
            (
                torch.randn([2]),    # 生成一个形状为 [2] 的随机张量
                torch.tensor([0]),   # 生成一个标量为 0 的张量
            ),
        )

        # 再次调用 common 测试方法，测试 fn 处理一个形状为 [2] 的随机张量和一个标量为 3 的张量的情况
        self.common(
            fn,
            (
                torch.randn([2]),    # 生成一个形状为 [2] 的随机张量
                torch.tensor([3]),   # 生成一个标量为 3 的张量
            ),
        )

    # 带有装饰器 @torch._dynamo.config.patch 的测试方法，用于测试 torch.cat 处理空张量的情况（2维情况）
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_cat_unbacked_2d(self):
        # 定义一个函数 fn，接收两个张量 x 和 y，对 y 进行 item 操作，并执行 torch.cat 操作
        def fn(x, y):
            z = y.item()                               # 获取张量 y 的标量值
            return torch.cat([x, x.new_ones(z, x.shape[1])])  # 将张量 x 与一个新的形状为 (z, x.shape[1]) 的全 1 张量连接起来

        # 调用通用测试方法 common，测试 fn 处理一个形状为 [2, 3] 的随机张量和一个标量为 0 的张量的情况
        self.common(
            fn,
            (
                torch.randn([2, 3]),  # 生成一个形状为 [2, 3] 的随机张量
                torch.tensor([0]),    # 生成一个标量为 0 的张量
            ),
        )

        # 再次调用 common 测试方法，测试 fn 处理一个形状为 [2, 3] 的随机张量和一个标量为 4 的张量的情况
        self.common(
            fn,
            (
                torch.randn([2, 3]),  # 生成一个形状为 [2, 3] 的随机张量
                torch.tensor([4]),    # 生成一个标量为 4 的张量
            ),
        )

    # 定义一个测试方法，用于测试 torch.cat 在负数维度下的行为
    def test_cat_negative_dim(self):
        # 定义一个函数 fn，接收任意数量的张量，并执行 torch.cat 操作，在最后一个维度上连接
        def fn(*tensors):
            return torch.cat(tensors, dim=-1)

        # 调用通用测试方法 common，测试 fn 处理两个形状为 [2, 3] 和 [2, 4] 的随机张量的情况
        self.common(
            fn,
            (
                torch.randn([2, 3]),  # 生成一个形状为 [2, 3] 的随机张量
                torch.randn([2, 4]),  # 生成一个形状为 [2, 4] 的随机张量
            ),
        )

        # 再次调用 common 测试方法，测试 fn 处理一个形状为 [2, 3] 和一个大小为 0 的张量和一个形状为 [2, 4] 的随机张量的情况
        self.common(
            fn,
            (
                torch.randn([2, 3]),  # 生成一个形状为 [2, 3] 的随机张量
                torch.randn([0]),     # 生成一个大小为 0 的随机张量
                torch.randn([2, 4]),  # 生成一个形状为 [2, 4] 的随机张量
            ),
        )

        # 再次调用 common 测试方法，测试 fn 处理一个大小为 0 的张量和一个形状为 [2, 3] 和一个形状为 [2, 4] 的随机张量的情况
        self.common(
            fn,
            (
                torch.randn([0]),     # 生成一个大小为 0 的随机张量
                torch.randn([2, 3]),  # 生成一个形状为 [2, 3] 的随机张量
                torch.randn([2, 4]),  # 生成一个形状为 [2, 4] 的随
    # 定义一个测试函数，测试 torch.cat 方法的向上转型行为
    def test_cat_upcasting(self):
        # 定义一个内部函数 fn，接受两个参数 arg4_1 和 slice_7
        def fn(arg4_1, slice_7):
            # 使用 torch.cat 方法将 arg4_1 和 slice_7 按照第二维度（1）拼接
            cat_1 = aten.cat.default([arg4_1, slice_7], 1)
            # 返回拼接后的结果，以元组形式返回
            return (cat_1,)

        # 调用 self.common 方法，传入 fn 函数以及两个 tensor 参数
        self.common(
            fn,
            (
                torch.randn([8, 16], dtype=torch.float32),
                torch.randn([8, 20], dtype=torch.float16),
            ),
        )

    # 定义一个测试函数，测试 torch.cat 方法在外部核心函数中的使用
    def test_cat_extern_kernel(self):
        # 定义一个内部函数 fn，接受四个参数 x1, x2, x3, x4
        def fn(x1, x2, x3, x4):
            # 计算 x2 和 x3 的矩阵乘积，结果保存在 x 中
            x = torch.mm(x2, x3)
            # 在 x 的第一维上进行裁剪，从索引 0 开始，长度为 100，结果保存在 s 中
            s = torch.narrow(x, 1, 0, 100)
            # 计算裁剪后的 s 和 x4 的矩阵乘积，结果保存在 x 中
            x = torch.mm(s, x4)
            # 使用 torch.cat 方法将 x 和 x1 按照第二维度（1）拼接，结果保存在 c 中
            c = torch.cat((x, x1), 1)
            # 返回拼接后的结果，以元组形式返回
            return (c,)

        # 根据设备类型选择合适的数值容差参数
        if self.device == "xpu":
            atol = 3e-4
            rtol = 1e-4
        else:
            # 使用默认值
            atol = None
            rtol = None
        
        # 调用 self.common 方法，传入 fn 函数以及四个 tensor 参数，以及数值容差参数和其他参数
        self.common(
            fn,
            (
                torch.randn(256, 256),
                torch.randn(256, 1024),
                torch.randn(1024, 1600),
                torch.randn(100, 256),
            ),
            atol=atol,
            rtol=rtol,
            check_lowp=False,  # 由于较大的矩阵乘法可能存在精度问题，关闭精度检查
        )

    # 如果不支持 CUDA 计算能力 80 或更高版本，则跳过测试
    @skipCUDAIf(not SM80OrLater, "uses bfloat16 which requires SM >= 80")
    # 由于问题 #108388，显式关闭了常量折叠，现在为测试重新打开
    @torch._inductor.config.patch(joint_graph_constant_folding=True)
    # 定义测试方法，测试移除无操作的情况
    def test_remove_no_ops(self):
        
        # 定义内部函数，用于执行带有操作的矩阵乘法
        def matmul_with_op(x, y, fn):
            return fn(x @ y)
        
        # 使用 torch.compile 编译 matmul_with_op 函数
        foo_opt = torch.compile(matmul_with_op)
        
        # 定义四个 lambda 函数，每个函数都不进行任何操作
        fns = (
            lambda x: x
            + torch.zeros(
                [256, 256], dtype=torch.float32, device=x.device
            ),  # noqa: E731
            lambda x: x
            - torch.zeros(
                [256, 256], dtype=torch.float32, device=x.device
            ),  # noqa: E731
            lambda x: x
            * torch.ones(
                [256, 256], dtype=torch.float32, device=x.device
            ),  # noqa: E731
            lambda x: x
            / torch.ones(
                [256, 256], dtype=torch.float32, device=x.device
            ),  # noqa: E731
        )
        
        # 生成两个随机张量作为输入
        inps = [torch.rand([256, 256], device=self.device) for _ in range(2)]
        
        # 遍历每个 lambda 函数
        for fn in fns:
            # 运行并获取优化后的结果和源代码
            out, source_codes = run_and_get_code(foo_opt, inps[0], inps[1], fn)
            # 断言优化结果与未优化结果一致
            self.assertEqual(out, matmul_with_op(inps[0], inps[1], fn))
            
            # 如果运行设备为 CPU
            if self.device == "cpu":
                # 使用 FileCheck 检查生成的源代码不包含 "cpp_fused"
                FileCheck().check_not("cpp_fused").run(source_codes[0])
            else:
                # 使用 FileCheck 检查生成的源代码不包含 "triton.jit"
                FileCheck().check_not("triton.jit").run(source_codes[0])
        
        # 测试数据类型转换的情况
        inps = [
            torch.rand([256, 256], device=self.device, dtype=torch.bfloat16)
            for _ in range(2)
        ]
        for fn in fns:
            # 运行并获取优化后的结果和源代码
            out, source_codes = run_and_get_code(foo_opt, inps[0], inps[1], fn)
            # 断言优化结果与未优化结果一致
            self.assertEqual(out, matmul_with_op(inps[0], inps[1], fn))
        
        # 测试广播形状不匹配的情况
        fn = lambda x: x + torch.zeros(  # noqa: E731
            [256, 256, 256], dtype=torch.bfloat16, device=self.device
        )
        # 运行并获取优化后的结果和源代码
        out, source_codes = run_and_get_code(foo_opt, inps[0], inps[1], fn)
        # 断言优化结果与未优化结果一致
        self.assertEqual(out, matmul_with_op(inps[0], inps[1], fn))

    # 定义测试方法，测试移除无操作的情况（复制操作）
    def test_remove_noop_copy(self):
        
        # 定义函数 fn，执行 cos() 操作后进行 copy_ 操作，并返回 sin() 操作的结果
        def fn(x, y):
            x = x.cos()
            a = x.copy_(y)
            return a.sin()
        
        # 调用共用函数 common 进行测试
        self.common(fn, (torch.randn(8, 8), torch.randn(8)))
        
        # 定义函数 fn2，计算张量 a 的绝对值的最大值，然后将 b[0] 设置为这个最大值，并返回 b
        def fn2(a, b):
            abs_max = torch.abs(a).max()
            b[0] = abs_max.to(a.dtype)
            return b
        
        # 调用共用函数 common 进行测试
        self.common(
            fn2,
            (
                torch.randn(8, 8, dtype=torch.float16),
                torch.randn(8, dtype=torch.float32),
            ),
        )

    # 定义测试方法，测试移除无操作的情况（克隆操作）
    def test_remove_noop_clone(self):
        
        # 定义函数 fn，克隆输入张量 x 并重塑形状，然后对部分列进行交换并与原始张量 x 相加
        def fn(x):
            y = x.clone().reshape(-1, 4)
            y[:, [2, 0]] = y[:, [0, 2]]
            return y + x
        
        # 调用共用函数 common 进行测试
        self.common(fn, (torch.randn(2, 4),))
    def test_cat_of_loops_and_extern_kernel(self):
        # 定义一个继承自 torch.nn.Module 的模块类 M
        class M(torch.nn.Module):
            # 初始化方法，接收任意关键字参数
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                # 创建一个 Conv2d 层对象，接收 64 个输入通道，输出 5 个通道，卷积核大小为 1，使用传入的其他关键字参数
                self.conv = torch.nn.Conv2d(
                    64,
                    5,
                    1,
                    **kwargs,
                )
                # 创建一个 MaxPool2d 层对象，池化核大小为 2
                self.max_pool2d = torch.nn.MaxPool2d(2)

            # 前向传播方法，接收两个输入 x 和 y
            def forward(self, x, y):
                # 对输入 x 进行卷积操作
                x1 = self.conv(x)
                # 对输入 y 进行最大池化操作
                y1 = self.max_pool2d(y)
                # 返回 x1 和 y1 拼接后的结果，沿着通道维度拼接
                return torch.cat([x1, y1], 1)

        # 创建 M 类的实例 mod
        mod = M()
        # 对 mod 进行优化，使用 torch._dynamo.optimize("inductor") 进行优化
        opt_mod = torch._dynamo.optimize("inductor")(mod)
        # 内存格式设为 torch.channels_last
        memory_format = torch.channels_last
        # 创建两个输入张量，形状分别为 [1, 64, 16, 16] 和 [1, 64, 32, 32]，并使用 channels_last 内存格式
        inputs = (
            torch.randn([1, 64, 16, 16]).to(memory_format=memory_format),
            torch.randn([1, 64, 32, 32]).to(memory_format=memory_format),
        )
        # 对 mod 和 opt_mod 进行前向传播，得到 y 和 opt_y
        y = mod(*inputs)
        opt_y = opt_mod(*inputs)
        # 断言 y 和 opt_y 相等
        self.assertEqual(y, opt_y)
        # 断言 y 的步幅与 opt_y 的步幅相等
        self.assertEqual(y.stride(), opt_y.stride())

    def test_cat_inplace(self):
        # 定义一个函数 fn，接收一个张量 x 作为输入
        def fn(x):
            # 对输入张量 x 进行拼接操作，返回结果
            rt = torch.cat([x])
            # 对输入张量 x 的元素进行 inplace 操作 sin_()，返回结果为操作前的拷贝 rt
            v = x.sin_()
            return rt

        # 创建一个全为 1 的输入张量 inp
        inp = torch.ones(2)
        # 编译优化函数 fn，得到 opt_fn
        opt_fn = torch.compile(fn)
        # 对 inp 的克隆进行 fn 函数计算，得到 res
        res = opt_fn(inp.clone())
        # 使用 fn 函数计算 inp 的克隆，得到期望结果 expected
        expected = fn(inp.clone())
        # 断言 res 和 expected 相等
        self.assertEqual(res, expected)

    def test_stack(self):
        # 定义一个函数 fn，接收两个张量 a 和 b 作为输入
        def fn(a, b):
            # 按照指定维度 2 对输入张量 a 和 b 进行堆叠操作，扩展维度为 [12, 16]
            return torch.stack(
                [
                    a.expand(12, 16),
                    b.expand(12, 16),
                ],
                2,
            )

        # 调用 self.common 方法，传入函数 fn 和两个输入张量的元组
        self.common(fn, (torch.randn([1, 16]), torch.randn([12, 1])))

    def test_hardtanh(self):
        # 定义一个函数 fn，接收一个张量 x 作为输入，应用于 torch.nn.functional.hardtanh 函数
        def fn(x):
            return F.hardtanh(x), F.hardtanh(x + 1), F.hardtanh(x - 1)

        # 调用 self.common 方法，传入函数 fn 和一个张量元组作为输入
        self.common(
            fn,
            (torch.randn([64]),),
        )

    def test_hardsigmoid(self):
        # 定义一个函数 fn，接收一个张量 x 作为输入，应用于 torch.nn.functional.hardsigmoid 函数
        def fn(x):
            return F.hardsigmoid(x), F.hardsigmoid(x + 3), F.hardsigmoid(x - 3)

        # 调用 self.common 方法，传入函数 fn 和一个张量元组作为输入
        self.common(
            fn,
            (torch.randn([64]),),
        )

    def test_hardswish(self):
        # 定义一个函数 fn，接收一个张量 x 作为输入，应用于 torch.nn.functional.hardswish 函数
        def fn(x):
            return F.hardswish(x), F.hardswish(x + 3), F.hardswish(x - 3)

        # 调用 self.common 方法，传入函数 fn 和一个张量元组作为输入
        self.common(
            fn,
            (torch.randn([64]),),
        )

    def test_rsqrt(self):
        # 定义一个函数 fn，接收一个张量 x 作为输入，应用于 torch.rsqrt 函数
        def fn(x):
            return torch.rsqrt(x), torch.rsqrt(x + 1) - 2

        # 调用 self.common 方法，传入函数 fn 和一个张量元组作为输入
        self.common(
            fn,
            (torch.randn([64]),),
        )

    def test_expm1(self):
        # 定义一个函数 fn，接收一个张量 x 作为输入，应用于 torch.expm1 函数
        def fn(x):
            return torch.expm1(x), torch.expm1(x) * 2

        # 遍历多种数据类型进行测试
        for dtype in (torch.float16, torch.float, torch.double, torch.int, torch.int64):
            # 调用 self.common 方法，传入函数 fn 和一个张量元组作为输入
            self.common(
                fn,
                (torch.randn([64]).to(dtype=dtype),),
            )
            # 再次调用 self.common 方法，传入函数 fn 和另一个张量元组作为输入
            self.common(
                fn,
                (torch.arange(-1e-5, 1e-5, 1e-7).to(dtype=dtype),),
            )
    # 定义一个测试函数 test_log1p，用于测试 torch.log1p 函数
    def test_log1p(self):
        # 定义内部函数 fn，接受参数 x，并返回 torch.log1p(x) 和 torch.log1p(x) * 2
        def fn(x):
            return torch.log1p(x), torch.log1p(x) * 2

        # 遍历不同的数据类型进行测试，包括 torch.float16, torch.float, torch.double, torch.int, torch.int64
        for dtype in (torch.float16, torch.float, torch.double, torch.int, torch.int64):
            # 调用测试框架的通用方法 common，传入 fn 函数和对应的参数元组
            self.common(
                fn,
                (torch.randn([64]).to(dtype=dtype),),
            )
            # 再次调用 common 方法，传入 fn 函数和不同的参数元组
            self.common(
                fn,
                (torch.arange(-1e-5, 1e-5, 1e-7).to(dtype=dtype),),
            )

    # 定义一个测试函数 test_flip，用于测试 torch.flip 函数
    def test_flip(self):
        # 定义内部函数 fn，接受参数 x，并返回 torch.flip(x, (-1,)) 和 torch.flip(x, (0, 2)) - 2
        def fn(x):
            return torch.flip(x, (-1,)), torch.flip(x, (0, 2)) - 2

        # 调用测试框架的通用方法 common，传入 fn 函数和对应的参数元组
        self.common(
            fn,
            (torch.randn([1, 2, 6, 6]),),
        )

    # 定义一个测试函数 test_signbit，用于测试 torch.signbit 函数
    def test_signbit(self):
        # 定义内部函数 fn，接受参数 x，并返回 torch.signbit(x) 和 ~torch.signbit(-x) & 1
        def fn(x):
            return torch.signbit(x), ~torch.signbit(-x) & 1

        # 调用测试框架的通用方法 common，传入 fn 函数和对应的参数元组
        self.common(
            fn,
            (torch.randn([1, 2, 6, 6]),),
        )

    # 定义一个测试函数 test_sign_dtype，用于测试 torch.sign 和 torch.tanh 函数
    def test_sign_dtype(self):
        # 定义内部函数 fn，接受参数 x
        def fn(x):
            # 对 x 应用 torch.sign 函数，结果赋给 y
            y = torch.sign(x)
            # 返回 torch.tanh(y) 的结果
            return torch.tanh(y)

        # 调用测试框架的通用方法 common，传入 fn 函数和对应的参数元组
        self.common(fn, (torch.randn([1, 2, 6, 6]),))

    # 定义一个测试函数 test_fmod，用于测试 torch.fmod 函数
    def test_fmod(self):
        # 定义内部函数 fn，接受参数 a 和 b，并返回 torch.fmod(a, b) 和 torch.fmod(3.0 * a, b) - 2.0
        def fn(a, b):
            return torch.fmod(a, b), torch.fmod(3.0 * a, b) - 2.0

        # 定义 shape 变量为 [1, 2, 6, 6]
        shape = [1, 2, 6, 6]
        # 调用测试框架的通用方法 common，传入 fn 函数和对应的参数元组
        self.common(fn, (torch.randn(shape), torch.randn(shape)))

    # 定义一个测试函数 test_fmod_zero_dim，用于测试 torch.fmod 函数处理零维张量的情况
    def test_fmod_zero_dim(self):
        # 定义内部函数 fn，接受参数 a 和 b，并返回 torch.fmod(a, b) 的结果元组
        def fn(a, b):
            return (torch.fmod(a, b),)

        # 调用测试框架的通用方法 common，传入 fn 函数和对应的参数元组
        self.common(
            fn,
            (
                make_tensor(10, device=self.device, dtype=torch.float32),
                make_tensor((), device=self.device, dtype=torch.float32),
            ),
        )
        # 再次调用 common 方法，传入 fn 函数和不同的参数元组
        self.common(
            fn,
            (
                make_tensor((), device=self.device, dtype=torch.float32),
                make_tensor(10, device=self.device, dtype=torch.float32),
            ),
        )

    # 定义一个测试函数 test_log2，用于测试 torch.log2 函数
    def test_log2(self):
        # 定义内部函数 fn，接受参数 x，并返回 torch.log2(x) 和 torch.log2(x + 1) - 2
        def fn(x):
            return torch.log2(x), torch.log2(x + 1) - 2

        # 调用测试框架的通用方法 common，传入 fn 函数和对应的参数元组
        self.common(
            fn,
            (torch.randn([64]) + 10,),
        )

    # 定义一个测试函数 test_logsumexp，用于测试 torch.logsumexp 函数
    def test_logsumexp(self):
        # 定义内部函数 fn，接受参数 x，并返回 torch.logsumexp(x, -1) 和 torch.logsumexp(x, 0) - 2
        def fn(x):
            return torch.logsumexp(x, -1), torch.logsumexp(x, 0) - 2

        # 调用测试框架的通用方法 common，传入 fn 函数和对应的参数元组
        self.common(
            fn,
            (torch.randn([8, 8]) + 10,),
        )

    # 定义一个测试函数 test_log_fp64，用于测试 torch.log 和 torch.log2 函数对于 double 类型数据的处理
    def test_log_fp64(self):
        # 定义内部函数 fn，接受参数 x，并返回 torch.log(x) 和 torch.log2(x)
        def fn(x):
            return torch.log(x), torch.log2(x)

        # 调用测试框架的通用方法 common，传入 fn 函数和对应的参数元组
        self.common(
            fn,
            (torch.randn([1024], dtype=torch.float64) + 10,),
        )

    # 定义一个测试函数 test_bitwise，用于测试位操作函数 torch.bitwise_not, torch.bitwise_or, torch.bitwise_xor, torch.bitwise_and
    def test_bitwise(self):
        # 定义内部函数 fn，接受参数 x 和 y，并返回 torch.bitwise_not(x), torch.bitwise_or(x, y), torch.bitwise_xor(x, y), torch.bitwise_and(x, y)
        def fn(x, y):
            return (
                torch.bitwise_not(x),
                torch.bitwise_or(x, y),
                torch.bitwise_xor(x, y),
                torch.bitwise_and(x, y),
            )

        # 调用测试框架的通用方法 common，传入 fn 函数和对应的参数元组
        self.common(
            fn,
            (
                torch.randint(0, 2**30, [64], dtype=torch.int32),
                torch.randint(0, 2**30, [64], dtype=torch.int32),
            ),
        )
    def test_bitwise2(self):
        # 测试位运算函数对布尔类型的操作
        def fn(x, y):
            # 使用 torch.bitwise_not 函数对 x 取反
            # 使用 torch.bitwise_or 函数对 x 和 y 进行按位或操作
            # 使用 torch.bitwise_xor 函数对 x 和 y 进行按位异或操作
            # 使用 torch.bitwise_and 函数对 x 和 y 进行按位与操作
            return (
                torch.bitwise_not(x),
                torch.bitwise_or(x, y),
                torch.bitwise_xor(x, y),
                torch.bitwise_and(x, y),
            )

        # 调用通用函数 common 来测试 fn 的行为
        self.common(
            fn,
            (
                torch.randint(0, 2, (2, 20), dtype=torch.bool),  # 生成一个布尔类型的随机张量 x
                torch.randint(0, 2, (2, 20), dtype=torch.bool),  # 生成一个布尔类型的随机张量 y
            ),
        )

    def test_bitwise3(self):
        # 用于重现 GitHub 问题 https://github.com/pytorch/pytorch/issues/97968
        def fn(x, y):
            # 使用 torch.bitwise_and 函数计算 x 和 y 的按位与，然后与 y 比较取最大值
            # 使用 torch.bitwise_or 函数计算 x 和 y 的按位或，然后与 y 比较取最大值
            # 使用 torch.bitwise_xor 函数计算 x 和 y 的按位异或，然后与 y 比较取最小值
            return (
                torch.max(torch.bitwise_and(x, y), y),
                torch.clamp_max(torch.bitwise_or(x, y), y),
                torch.clamp_min(torch.bitwise_xor(x, y), y),
            )

        # 调用通用函数 common 来测试 fn 的行为
        self.common(
            fn,
            (
                torch.rand([5, 10, 1]).to(torch.int8),  # 生成一个随机张量 x，转换为 int8 类型
                torch.rand([10, 1]).to(torch.int8),      # 生成一个随机张量 y，转换为 int8 类型
            ),
        )

    def test_inf(self):
        def fn(a):
            # 返回 a 加正无穷大，a 加负无穷大，a 乘以负无穷大 的结果
            return a + float("inf"), a + float("-inf"), a * -float("inf")

        # 调用通用函数 common 来测试 fn 的行为，输入是一个包含 8 个随机数的张量 a
        self.common(fn, (torch.randn(8),))

    def test_remainder(self):
        def fn(a, b):
            # 返回 torch.remainder 函数对 a 和 b 求余数的结果
            # 返回 torch.remainder 函数对 a+1 和 b-1 求余数的结果
            # 返回 torch.remainder 函数对 a-1 和 b+1 求余数的结果
            return (
                torch.remainder(a, b),
                torch.remainder(a + 1, b - 1),
                torch.remainder(a - 1, b + 1),
            )

        # 调用通用函数 common 来测试 fn 的行为，输入是两个包含 64 个随机数的张量 a 和 b
        self.common(fn, (torch.randn(64), torch.randn(64)))

    def test_zeros(self):
        def fn(a):
            # 返回 a+1 的结果
            # 返回一个形状为 (1, 8, 64, 64) 的全零张量，数据类型为 torch.float32，设备和 a 相同
            # 返回一个形状为 (1, 8, 64, 64) 的全零张量，数据类型为 torch.float32，设备和 a 相同
            # 返回一个形状为 (2, 3) 的全零张量
            # 返回 a 加上全一张量，设备和 a 相同
            # 返回一个形状为 (2, 3)、填充值为 3.1416 的张量，设备和 a 相同
            return (
                a + 1,
                torch.zeros(
                    (1, 8, 64, 64),
                    dtype=torch.float32,
                    device=a.device,
                ),
                torch.zeros(
                    1,
                    8,
                    64,
                    64,
                    dtype=torch.float32,
                    device=a.device,
                ),
                torch.zeros(2, 3),
                a + torch.ones(8, device=a.device),
                torch.full((2, 3), 3.1416, device=a.device),
            )

        # 调用通用函数 common 来测试 fn 的行为，输入是一个包含 8 个随机数的张量 a
        self.common(fn, (torch.randn(8),))

    def test_new_ones(self):
        def fn(a):
            # 返回使用 aten.new_ones 函数生成的张量，形状和设备与 a 相同，数据类型为 6，布局为 0，不固定在内存中
            # 返回使用 aten.new_zeros 函数生成的张量，形状和设备与 a 相同，数据类型为 6，布局为 0，不固定在内存中
            return (
                aten.new_ones(
                    a, [], device=a.device, dtype=6, layout=0, pin_memory=False
                ),
                aten.new_zeros(
                    a, [], device=a.device, dtype=6, layout=0, pin_memory=False
                ),
            )

        # 调用通用函数 common 来测试 fn 的行为，输入是一个包含 8 个随机数的张量 a
        self.common(fn, (torch.randn(8),))

    def test_full_like(self):
        def fn(a):
            # 返回 torch.full_like(a, 7.777) - 1 的结果
            return torch.full_like(a, 7.777) - 1

        # 调用通用函数 common 来测试 fn 的行为，输入是一个包含 8 个随机数的张量 a
        self.common(fn, (torch.randn(8),))

    def test_full_truncation(self):
        def fn(a):
            # 返回 a 加上 torch.full_like(a, 7.777) 的结果
            for dtype in all_types():
                self.common(fn, (make_tensor(8, dtype=dtype, device=self.device),))
    # 定义一个测试方法，用于测试返回布尔值的函数
    def test_full_boolean(self):
        # 定义一个返回布尔值的函数，根据输入值是否大于等于1024生成对应的张量
        def fn(n):
            x = torch.full((1,), n >= 1024, device=self.device)
            return x, x + 1

        # 调用通用测试方法，测试 fn 函数
        self.common(fn, (1024,))
        self.common(fn, (1023,))

    # 定义一个测试方法，用于测试索引操作的函数
    def test_index1(self):
        # 定义一个接受三个参数并返回索引操作结果的函数
        def fn(a, b, c):
            return aten.index(a, [b, c])

        # 调用通用测试方法，测试 fn 函数，传入随机张量及对应的索引列表
        self.common(
            fn,
            (
                torch.randn(8, 8, 12),
                torch.tensor([0, 0, 2, 2], dtype=torch.int64),
                torch.tensor([3, 4, 4, 3], dtype=torch.int64),
            ),
        )
        self.common(
            fn,
            (
                torch.randn(8, 8, 12),
                torch.tensor([[0, 0, 2, 2]], dtype=torch.int64),
                torch.tensor([[3], [4], [4], [3]], dtype=torch.int64),
            ),
        )

    # 定义一个测试方法，用于测试多维度索引操作的函数
    def test_index2(self):
        # 定义一个接受两个参数并返回多种索引操作结果的函数
        def fn(a, b):
            return (
                aten.index(a, [b]),
                aten.index(a, [None, b]),
            )

        # 调用通用测试方法，测试 fn 函数，传入随机张量及对应的索引列表
        self.common(
            fn,
            (
                torch.randn(8, 8, 8),
                torch.tensor([[0, 0, 2, 2]], dtype=torch.int64),
            ),
        )

    # 定义一个测试方法，用于测试多维度索引和切片操作的函数
    def test_index3(self):
        # 定义一个接受三个参数并返回经过多维度索引和切片操作的结果的函数
        def fn(x, ia, ib):
            return (x[:, ia, None, ib, 0],)

        # 调用通用测试方法，测试 fn 函数，传入随机张量及对应的索引列表
        self.common(
            fn,
            (
                torch.randn(3, 4, 4, 4, 3),
                torch.tensor([0, 2, 1], dtype=torch.int64),
                torch.tensor([0, 2, 1], dtype=torch.int64),
            ),
        )

    # 定义一个测试方法，用于测试张量操作中的视图和转置操作的函数
    def test_output_strides(self):
        # 定义一个接受一个参数并返回经过视图和转置操作后的结果的函数
        def fn(x):
            y = x.permute(0, 2, 3, 1).contiguous()
            torch._dynamo.graph_break()
            return y.view(-1, 4)

        # 创建一个随机输入张量
        inp = torch.rand([4, 4, 4, 4], device=self.device)
        # 通过指定优化器对 fn 函数进行优化
        fn_opt = torch._dynamo.optimize("inductor")(fn)

        # 断言普通和优化后的 fn 函数的输出结果及其步长是否相等
        self.assertEqual(fn(inp), fn_opt(inp))
        self.assertEqual(fn(inp).stride(), fn_opt(inp).stride())

        # 定义一个函数，执行张量切片和转置操作，并进行优化
        # 检查优化后的结果是否与输入张量的存储是否一致
        def foo(x):
            return x[0:2:2].T[3:].squeeze(0)

        foo_opt = torch._dynamo.optimize("inductor")(foo)
        out = foo_opt(inp)
        self.assertEqual(inp.storage(), out.storage())

    # 定义一个测试方法，用于测试张量索引选择操作的函数
    def test_index_select(self):
        # 定义一个接受两个参数并返回多种索引选择操作结果的函数
        def fn(a, b):
            return (
                torch.index_select(a, 0, b),
                torch.index_select(a, 1, b),
                torch.index_select(torch.index_select(a, 2, b), 1, b),
            )

        # 遍历不同的索引数据类型进行测试
        for ind_dtype in (torch.int32, torch.int64):
            self.common(
                fn,
                (
                    torch.randn(8, 8, 8),
                    torch.tensor([0, 0, 2, 1], dtype=ind_dtype),
                ),
            )

    # 根据条件跳过 CUDA 环境下测试的装饰器
    @skipCUDAIf(not TEST_CUDNN, "CUDNN not available")
    # 根据条件跳过 XPU 环境下测试的装饰器
    @skipIfXpu
    # 根据条件跳过 ROCm 环境下测试的装饰器
    @skipIfRocm
    # 定义一个名为 test_cudnn_rnn 的测试方法
    def test_cudnn_rnn(self):
        # 如果设备是 CPU，跳过此测试，并给出相应的提示信息
        if self.device == "cpu":
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        # 定义一个内部函数 fn，接受多个参数
        def fn(
            a0,  # 参数 a0
            b0,  # 参数 b0
            b1,  # 参数 b1
            b2,  # 参数 b2
            b3,  # 参数 b3
            b4,  # 参数 b4
            b5,  # 参数 b5
            b6,  # 参数 b6
            b7,  # 参数 b7
            b8,  # 参数 b8
            b9,  # 参数 b9
            b10,  # 参数 b10
            b11,  # 参数 b11
            b12,  # 参数 b12
            b13,  # 参数 b13
            b14,  # 参数 b14
            b15,  # 参数 b15
            a3,  # 参数 a3
            a4,  # 参数 a4
            a5,  # 参数 a5
        ):
            # 将 b0 到 b15 参数组成列表 a1
            a1 = [
                b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15
            ]
            # 调用 aten._cudnn_rnn 方法，传入各参数进行计算
            return aten._cudnn_rnn(
                a0,  # 参数 a0
                a1,  # 参数 a1
                4,  # 数字参数 4
                a3,  # 参数 a3
                a4,  # 参数 a4
                a5,  # 参数 a5
                2,  # 数字参数 2
                2048,  # 数字参数 2048
                0,  # 数字参数 0
                2,  # 数字参数 2
                False,  # 布尔参数 False
                0.0,  # 数字参数 0.0
                False,  # 布尔参数 False
                True,  # 布尔参数 True
                [],  # 空列表参数
                None,  # 空参数
            )

        # 调用 self.common 方法进行测试，传入 fn 函数及其参数
        self.common(
            fn,
            (
                torch.randn([92, 8, 2048]),  # 参数 1: 92x8x2048 大小的随机张量
                torch.randn([8192, 2048]),  # 参数 2: 8192x2048 大小的随机张量
                torch.randn([8192, 2048]),  # 参数 3: 8192x2048 大小的随机张量
                torch.randn([8192]),  # 参数 4: 8192 大小的随机张量
                torch.randn([8192]),  # 参数 5: 8192 大小的随机张量
                torch.randn([8192, 2048]),  # 参数 6: 8192x2048 大小的随机张量
                torch.randn([8192, 2048]),  # 参数 7: 8192x2048 大小的随机张量
                torch.randn([8192]),  # 参数 8: 8192 大小的随机张量
                torch.randn([8192]),  # 参数 9: 8192 大小的随机张量
                torch.randn([8192, 4096]),  # 参数 10: 8192x4096 大小的随机张量
                torch.randn([8192, 2048]),  # 参数 11: 8192x2048 大小的随机张量
                torch.randn([8192]),  # 参数 12: 8192 大小的随机张量
                torch.randn([8192]),  # 参数 13: 8192 大小的随机张量
                torch.randn([8192, 4096]),  # 参数 14: 8192x4096 大小的随机张量
                torch.randn([8192, 2048]),  # 参数 15: 8192x2048 大小的随机张量
                torch.randn([8192]),  # 参数 16: 8192 大小的随机张量
                torch.randn([8192]),  # 参数 17: 8192 大小的随机张量
                torch.randn([167837696]),  # 参数 18: 167837696 大小的随机张量
                torch.randn([4, 8, 2048]),  # 参数 19: 4x8x2048 大小的随机张量
                torch.randn([4, 8, 2048]),  # 参数 20: 4x8x2048 大小的随机张量
            ),
            check_lowp=False,  # 布尔参数，指示是否检查低精度
        )

    # 定义一个名为 test_upsample_nearest1d 的测试方法
    def test_upsample_nearest1d(self):
        # 定义一个内部函数 fn，接受一个参数 a
        def fn(a):
            # 分别调用 aten.upsample_nearest1d 方法进行上采样，返回结果组成元组
            return (
                aten.upsample_nearest1d(a, [74], None),  # 上采样至长度 74
                aten.upsample_nearest1d(a, [70], None),  # 上采样至长度 70
                aten.upsample_nearest1d(a, [45], None),  # 上采样至长度 45
                aten.upsample_nearest1d(a, [36], None),  # 上采样至长度 36
                aten.upsample_nearest1d(a, None, [2.0]),  # 上采样倍率为 2.0
            )

        # 调用 self.common 方法进行测试，传入 fn 函数及其参数
        self.common(fn, (torch.randn([2, 4, 37]),))  # 参数为大小为 2x4x37 的随机张量
    # 定义测试函数 test_upsample_nearest2d，用于测试 torch 的 nearest neighbor 二维上采样功能
    def test_upsample_nearest2d(self):
        # 定义内部函数 fn，接受参数 a，返回对 a 进行不同尺寸的 nearest neighbor 二维上采样结果
        def fn(a):
            return (
                aten.upsample_nearest2d(a, [74, 76]),  # 使用尺寸 [74, 76] 进行二维上采样
                aten.upsample_nearest2d(a, [70, 75]),  # 使用尺寸 [70, 75] 进行二维上采样
                aten.upsample_nearest2d(a, [45, 74]),  # 使用尺寸 [45, 74] 进行二维上采样
                aten.upsample_nearest2d(a, [36, 39]),  # 使用尺寸 [36, 39] 进行二维上采样
                aten.upsample_nearest2d(a, None, [2.0, 2.0]),  # 使用倍率 [2.0, 2.0] 进行二维上采样
            )

        # 调用公共测试函数 common，测试 fn 函数在给定参数 (torch.randn([2, 4, 37, 38]),) 下的结果
        self.common(fn, (torch.randn([2, 4, 37, 38]),))

    # 定义测试函数 test_upsample_nearest3d，用于测试 torch 的 nearest neighbor 三维上采样功能
    def test_upsample_nearest3d(self):
        # 定义内部函数 fn，接受参数 a，返回对 a 进行不同尺寸的 nearest neighbor 三维上采样结果
        def fn(a):
            return (
                aten.upsample_nearest3d(a, [74, 76, 78], None),  # 使用尺寸 [74, 76, 78] 进行三维上采样
                aten.upsample_nearest3d(a, [70, 75, 80], None),  # 使用尺寸 [70, 75, 80] 进行三维上采样
                aten.upsample_nearest3d(a, [45, 74, 103], None),  # 使用尺寸 [45, 74, 103] 进行三维上采样
                aten.upsample_nearest3d(a, [36, 39, 40], None),  # 使用尺寸 [36, 39, 40] 进行三维上采样
                aten.upsample_nearest3d(a, None, [2.0, 2.0, 2.0]),  # 使用倍率 [2.0, 2.0, 2.0] 进行三维上采样
            )

        # 调用公共测试函数 common，测试 fn 函数在给定参数 (torch.randn([2, 4, 37, 38, 39]),) 下的结果
        self.common(fn, (torch.randn([2, 4, 37, 38, 39]),))

    # 定义测试函数 test_upsample_nearest2d_backward，用于测试 torch 的 nearest neighbor 二维上采样反向传播功能
    def test_upsample_nearest2d_backward(self):
        # 获取 torch.ops.aten 中的 upsample_nearest2d_backward 函数引用
        func = torch.ops.aten.upsample_nearest2d_backward

        # 定义内部函数 fn，接受参数 a，返回对 a 进行不同输入和输出尺寸的 nearest neighbor 二维上采样反向传播结果
        def fn(a):
            return (
                func(a, output_size=[6, 12], input_size=[3, 3, 3, 6]),  # 使用输出尺寸 [6, 12] 和输入尺寸 [3, 3, 3, 6] 进行反向传播
                func(a, output_size=[6, 12], input_size=[3, 3, 4, 5]),  # 使用输出尺寸 [6, 12] 和输入尺寸 [3, 3, 4, 5] 进行反向传播
                func(a, output_size=[6, 12], input_size=[3, 3, 2, 8]),  # 使用输出尺寸 [6, 12] 和输入尺寸 [3, 3, 2, 8] 进行反向传播
                func(a, output_size=[6, 12], input_size=[3, 3, 2, 8]),  # 使用输出尺寸 [6, 12] 和输入尺寸 [3, 3, 2, 8] 进行反向传播
                func(a, output_size=[6, 12], input_size=[3, 3, 4, 7]),  # 使用输出尺寸 [6, 12] 和输入尺寸 [3, 3, 4, 7] 进行反向传播
            )

        # 调用公共测试函数 common，测试 fn 函数在给定参数 (torch.randn([3, 3, 6, 12]),) 下的结果
        self.common(fn, (torch.randn([3, 3, 6, 12]),))

    # 根据运行环境决定是否跳过，定义测试函数 test_upsample_bilinear2d_a，用于测试 torch 的 bilinear 二维上采样功能
    @skip_if_x86_mac()
    def test_upsample_bilinear2d_a(self):
        # 定义内部函数 fn，接受参数 a，返回对 a 进行不同设置的 bilinear 二维上采样结果
        def fn(a):
            return (
                aten.upsample_bilinear2d(a, [45, 45], False, None),  # 使用尺寸 [45, 45] 进行 bilinear 二维上采样
                aten.upsample_bilinear2d(a, None, True, [2.0, 2.0]),  # 使用倍率 [2.0, 2.0] 进行 bilinear 二维上采样
            )

        # 调用公共测试函数 common，测试 fn 函数在给定参数 (torch.randn([2, 4, 37, 38]),) 下的结果，设置绝对误差和相对误差
        self.common(fn, (torch.randn([2, 4, 37, 38]),), atol=2.5e-5, rtol=1.3e-6)

    # 定义测试函数 test_upsample_bilinear2d_b，用于测试 torch 的 bilinear 二维上采样功能（另一种形式）
    def test_upsample_bilinear2d_b(self):
        # 定义内部函数 fn，接受参数 a，返回对 a 进行 bilinear 二维上采样结果
        def fn(a):
            return aten.upsample_bilinear2d(a, None, True, [2.0, 2.0])

        # 调用公共测试函数 common，测试 fn 函数在给定参数 [torch.randn([1, 2, 40, 59]),] 下的结果，设置绝对误差和相对误差
        self.common(
            fn,
            [
                torch.randn([1, 2, 40, 59]),
            ],
            atol=2.5e-5,
            rtol=1.3e-6,
        )

    # 根据运行环境决定是否跳过，定义测试函数 test_reflection_pad2d，用于测试 torch 的 reflection padding 二维功能
    @skip_if_gpu_halide  # 准确性问题
    def test_reflection_pad2d(self):
        # 定义内部函数 fn，接受参数 a 和 pad，返回对 a 进行不同设置的 reflection padding 结果
        def fn(a, pad):
            return (
                aten.reflection_pad2d(a, [1, 1, 1, 1]),  # 使用 padding [1, 1, 1, 1] 进行 reflection padding
                aten.reflection_pad2d(a, pad),  # 使用给定的 pad 参数进行 reflection padding
            )

        #
    def test_reflection_pad2d_backward(self):
        # 定义内部函数模板，用于测试反射填充的反向传播
        def template(size, padding):
            # 定义具体的测试函数，接收梯度和输入数据，返回反射填充的反向传播结果
            def fn(grad_output, x):
                return aten.reflection_pad2d_backward(grad_output, x, padding)

            # 生成指定大小和填充的随机输入张量
            x = torch.randint(0, 999, size=size, dtype=torch.float32)
            # 对输入张量进行反射填充
            result = aten.reflection_pad2d(x, padding)
            # 生成与结果张量形状相同的随机梯度张量
            grad_output = torch.randn_like(result)

            # 调用公共测试函数，检查低精度是否为哈利德后端
            self.common(
                fn, (grad_output, x), check_lowp=not is_halide_backend(self.device)
            )

        # 使用模板函数测试不同的输入大小和填充组合
        template([1, 1, 8, 8], [0, 0, 0, 0])
        template([1, 1, 8, 8], [1, 1, 1, 1])
        template([1, 1, 8, 8], [1, 2, 3, 4])
        template([1, 1, 8, 8], [0, -1, 2, 2])
        template([1, 1, 8, 8], [-1, 0, 2, 2])
        template([1, 1, 8, 8], [2, 2, 0, -1])
        template([1, 1, 8, 8], [2, 2, -1, 0])

    def test_grid_sampler_2d(self):
        # 定义测试函数，评估二维网格采样
        def fn(a, b):
            return (
                # 调用aten.grid_sampler_2d进行二维网格采样，使用边界填充并返回结果
                aten.grid_sampler_2d(a, b, 0, 0, True),
                # 调用aten.grid_sampler_2d进行二维网格采样，不使用边界填充并返回结果
                aten.grid_sampler_2d(a, b, 0, 1, False),
            )

        # 调用公共测试函数，验证二维网格采样的行为
        self.common(
            fn,
            (
                torch.randn([4, 3, 352, 352], dtype=torch.float32),
                torch.rand([4, 352, 352, 2], dtype=torch.float32) * 2 - 1,
            ),
            check_lowp=False,
            # 检查绝对误差和相对误差的阈值
            # Mismatched elements: 154697 / 1486848 (10.4%)
            # Greatest absolute difference: 0.0001976490020751953 at index (0, 0, 101, 243) (up to 1e-05 allowed)
            # Greatest relative difference: 7.332530120481928 at index (1, 1, 258, 301) (up to 1.3e-06 allowed)
            atol=0.0002,
            rtol=1.3e-06,
        )

    def test_upsample_bicubic2d(self):
        # 定义测试函数，评估二维双三次上采样
        def fn(a):
            return (
                # 调用aten.upsample_bicubic2d进行二维双三次上采样，输出尺寸为(128, 128)，使用边界填充
                aten.upsample_bicubic2d(a, (128, 128), True),
                # 调用aten.upsample_bicubic2d进行二维双三次上采样，输出尺寸为(128, 256)，不使用边界填充
                aten.upsample_bicubic2d(a, (128, 256), False),
            )

        # 调用公共测试函数，验证二维双三次上采样的行为
        # Mismatched elements: 10 / 196608 (0.0%)
        # Greatest absolute difference: 1.3869255781173706e-05 at index (2, 1, 88, 65) (up to 1e-05 allowed)
        # Greatest relative difference: 0.0033082996811011046 at index (3, 1, 88, 91) (up to 1.3e-06 allowed)
        self.common(
            fn,
            (torch.randn([4, 3, 64, 32], dtype=torch.float32),),
            atol=2e-5,
            rtol=1e-3,
        )

    def test_float_index_expression(self):
        # 测试索引传播不会生成错误的索引表达式调用，例如 ops.index_expr(0.5*x, dtype)，其中表达式不是整数
        def fn(x):
            return aten.upsample_bicubic2d(x, (256, 256), False)

        # 创建指定设备上的随机输入张量
        x = torch.randn(1, 1, 128, 128, dtype=torch.float32, device=self.device)
        # 运行并获取代码，返回结果和源代码
        _, source_codes = run_and_get_code(fn, x)

        # 定义用于匹配错误索引表达式的正则表达式模式
        pattern = r"0\.50*\*[ix][\d]"
        # 在源代码中搜索匹配模式的表达式
        for code in source_codes:
            self.assertIsNone(
                re.search(pattern, code), msg="Found bad index_expr in code:\n" + code
            )
    # 测试浮点索引表达式参与类型提升的情况
    def test_float_index_expression_type_promotion(self):
        # 定义一个函数，接受参数 x，并返回 x + 1.0 / x.size(0)
        def fn(x):
            return x + 1.0 / x.size(0)

        # 创建一个包含 0 到 9 的张量
        x = torch.arange(10)
        # 调用 self.common 方法，传入函数 fn 和参数元组 (x,)
        self.common(fn, (x,))

    # 测试排序功能
    def test_sort(self):
        # 定义一个函数 fn，接受参数 a 和 descending，返回 torch.sort(a)
        def fn(a, descending):
            return torch.sort(a)

        # 创建一个形状为 [1, 1, 8, 8] 的随机整数张量
        inp = torch.randint(0, 999, size=[1, 1, 8, 8], dtype=torch.float32)
        # 调用 self.common 方法，分别传入 fn 和参数元组 (inp, False) 和 (inp, True)
        self.common(fn, (inp, False))
        self.common(fn, (inp, True))

    # 测试稳定排序功能
    def test_sort_stable(self):
        # 定义一个函数 fn，接受参数 a 和 descending，返回 a.sort(dim=-1, stable=True, descending=descending)
        
        # 创建一个形状为 [10, 128] 的随机浮点数张量 inp
        inp = torch.rand(10, 128, dtype=torch.float32)
        # 将 inp 的部分区域设置为重复的值 1.0，以测试稳定排序
        inp[:, 10:20] = 1.0
        inp[:, 30:40] = 1.0
        # 调用 self.common 方法，分别传入 fn 和参数元组 (inp, False) 和 (inp, True)
        self.common(fn, (inp, False))
        self.common(fn, (inp, True))

        # 创建一个形状为 [10, 120] 的切片 inp，以测试非二的幂排序
        inp = inp[:, :120]
        # 调用 self.common 方法，分别传入 fn 和参数元组 (inp, False) 和 (inp, True)
        self.common(fn, (inp, False))
        self.common(fn, (inp, True))

    # 测试布尔类型排序功能
    def test_sort_bool(self):
        # 定义一个函数 fn，接受参数 a 和 descending，返回 torch.sort(a.to(torch.int8), stable=True, descending=descending)
        
        # 创建一个形状为 [10, 128] 的随机布尔型张量 inp
        inp = torch.randint(0, 2, size=[10, 128], dtype=torch.bool)
        # 调用 self.common 方法，分别传入 fn 和参数元组 (inp, False) 和 (inp, True)
        self.common(fn, (inp, False))
        self.common(fn, (inp, True))

    # 测试转置后的排序功能
    def test_sort_transpose(self):
        # 定义一个函数 fn，接受参数 a 和 descending，返回 torch.sort(a, stable=True, descending=descending)
        
        # 创建一个形状为 [128, 10] 的随机正态分布张量 inp，并进行转置
        inp = torch.randn(128, 10).transpose(0, 1)
        # 调用 self.common 方法，分别传入 fn 和参数元组 (inp, False) 和 (inp, True)
        self.common(fn, (inp, False))
        self.common(fn, (inp, True))

    # 测试取最大 k 个元素的功能
    def test_topk(self):
        # 定义一个函数 fn，接受参数 a，返回 torch.topk(a, 2, -1)
        
        # 创建一个形状为 [1, 1, 8, 8] 的随机整数张量，并传入 fn
        self.common(
            fn, (torch.randint(0, 999, size=[1, 1, 8, 8], dtype=torch.float32),)
        )

    # 测试长整型张量与设备相关操作的功能
    def test_long_tensor(self):
        # 定义一个函数 fn，接受参数 a，返回 (torch.LongTensor([294]).to(a.device) - a, torch.as_tensor([295]).to(a.device) + a)
        
        # 创建一个形状为 [8, 8] 的随机整数张量，并传入 fn
        self.common(fn, (torch.randint(0, 999, size=[8, 8]),))

    @skip_if_gpu_halide  # correctness issue
    # 测试一维常数填充的功能
    def test_constant_pad_1d(self):
        # 定义一个函数 fn，接受参数 a，返回 (aten.constant_pad_nd(a, [0, 1], 6.0), aten.constant_pad_nd(a, [2, 3], 99.0))
        
        # 创建一个形状为 [2, 16, 31] 的随机浮点数张量，并传入 fn
        self.common(fn, (torch.randint(0, 999, size=[2, 16, 31], dtype=torch.float32),))

    # 测试填充与数据类型相关的功能
    def test_constant_pad_fill_dtype(self):
        # 定义一个函数 fn，接受参数 a 和 b，返回 (aten.constant_pad_nd(a, (1, 1), 1.0) & b, aten.constant_pad_nd(a, (1, 1), 0.0) & b)
        
        # 创建一个形状为 [4] 的随机布尔型张量 a 和形状为 [6] 的全 1 布尔型张量 b，并传入 fn
        self.common(
            fn,
            (torch.randint(2, (4,), dtype=torch.bool), torch.ones(6, dtype=torch.bool)),
        )

    @skip_if_gpu_halide  # misaligned address
    def test_constant_pad_2d(self):
        # 定义测试函数 fn，对输入张量进行常量填充操作，返回填充后的结果
        def fn(a):
            return (
                # 使用 aten.constant_pad_nd 函数对输入张量 a 进行 2D 的常量填充操作，填充值为 6.0
                aten.constant_pad_nd(a, [1, 1, 1, 1], 6.0),
                # 使用 aten.constant_pad_nd 函数对输入张量 a 进行 2D 的常量填充操作，填充值为 99.0
                aten.constant_pad_nd(a, [1, 2, 3, 4], 99.0),
            )

        # 调用 self.common 方法执行测试，传入 fn 函数和一个随机生成的 8x8 浮点数张量作为参数
        self.common(
            fn, (torch.randint(0, 999, size=[1, 1, 8, 8], dtype=torch.float32),)
        )

    @skip_if_gpu_halide  # 如果 GPU 上的 Halide 未对齐地址，跳过此测试
    def test_constant_pad_3d(self):
        # 定义测试函数 fn，对输入张量进行常量填充操作，返回填充后的结果
        def fn(a):
            return (
                # 使用 aten.constant_pad_nd 函数对输入张量 a 进行 3D 的常量填充操作，填充值为 6.0
                aten.constant_pad_nd(a, [1, 2, 3, 4, 5, 6], 6.0),
                # 使用 aten.constant_pad_nd 函数对输入张量 a 进行 3D 的常量填充操作，填充值为 6.0
                aten.constant_pad_nd(a, [0, 0, 3, 4, 0, 0], 6.0),
            )

        # 调用 self.common 方法执行测试，传入 fn 函数和一个随机生成的 2x4x4x4 浮点数张量作为参数
        self.common(
            fn, (torch.randint(0, 999, size=[2, 4, 4, 4], dtype=torch.float32),)
        )

    def test_constant_pad_float64(self):
        # 用于重现 https://github.com/pytorch/pytorch/issues/93351 的问题
        def fn(input):
            # 对输入张量进行 2D 的常量填充操作，填充区域为 (1, 0)
            v1 = torch.nn.functional.pad(input, pad=(1, 0))
            # 返回一个布尔张量，判断 v1 中的元素是否大于 input 中的对应元素
            return torch.gt(v1, input)

        # 创建一个随机生成的 1x2x2x1 双精度浮点数张量
        x = torch.rand([1, 2, 2, 1], dtype=torch.float64)
        # 调用 self.common 方法执行测试，传入 fn 函数和 x 作为参数
        self.common(fn, (x,))

    def test_constant_pad_nd_inplace(self):
        # 定义测试函数 fn，对输入张量进行常量填充操作，并返回填充后的结果
        def fn(a):
            # 使用 aten.constant_pad_nd 函数对输入张量 a 进行多维度的常量填充操作，填充区域为 (0, 0)
            return aten.constant_pad_nd(a, [0, 0])

        # 创建一个随机生成的 2 元素张量，并指定设备为 self.device
        x = torch.randn([2], device=self.device)
        # 编译 fn 函数，生成一个编译后的函数 fn_compiled
        fn_compiled = torch.compile(fn)
        # 调用 fn_compiled 函数，对 x 进行填充操作并返回结果 y
        y = fn_compiled(x)
        # 断言 y 不等于 x
        self.assertTrue(y is not x)

    def test_l1_loss(self):
        # 定义测试函数 fn，计算输入张量 a 和 b 之间的 L1 损失和 MSE 损失，并返回结果
        def fn(a, b):
            return torch.nn.functional.l1_loss(a, b), torch.nn.functional.mse_loss(a, b)

        # 调用 self.common 方法执行测试，传入 fn 函数和两个随机生成的张量作为参数，并关闭低精度检查
        self.common(
            fn,
            (
                torch.randn([2, 3, 16, 16]),
                torch.randn([2, 3, 16, 16]),
            ),
            check_lowp=False,
        )

    def test_triu(self):
        # 定义测试函数 fn，对输入张量进行上三角矩阵操作，并返回结果
        def fn(a):
            return aten.triu(a, 1), aten.triu(a, 0), aten.triu(a, 2)

        # 调用 self.common 方法执行测试，传入 fn 函数和一个随机生成的 2x10x10 浮点数张量作为参数
        self.common(fn, (torch.randn([2, 10, 10]),))

    def test_no_op_reduction(self):
        # 定义测试函数 fn，对输入张量进行求和操作和最大值操作，并返回结果
        def fn(a):
            return a.sum(-1), torch.amax(a + 1, 1, keepdim=True)

        # 调用 self.common 方法执行测试，传入 fn 函数和一个随机生成的 8x1x1 浮点数张量作为参数
        self.common(fn, (torch.randn([8, 1, 1]),))

    def test_inplace_add(self):
        # 使用 torch._dynamo.optimize 修饰器定义优化器函数 fn，对输入张量进行原地加法操作，并返回结果
        @torch._dynamo.optimize("inductor")
        def fn(x, y):
            return x.add_(y)

        # 创建两个随机生成的 4x4 张量 inputs，设备为 self.device
        inputs = (
            rand_strided((4, 4), (4, 1), device=self.device),
            rand_strided((4, 4), (4, 1), device=self.device),
        )
        # 克隆 inputs[0]，作为后续比较的基准
        inp_clone = inputs[0].clone()
        # 调用 fn 函数，对 inputs 进行原地加法操作并返回结果 out
        out = fn(*inputs)
        # 断言 out 等于 inp_clone 加上 inputs[1] 的结果，并且 out 是 inputs[0] 的引用
        self.assertTrue(same(out, inp_clone + inputs[1]))
        self.assertTrue(out is inputs[0])

    # 以下两个测试用例用于检查如果 xnumel = 1，则 Triton 载入/存储将跳过 xmask 的逻辑
    @requires_gpu()
    def test_single_elem(self):
        # 定义测试函数 fn，对输入张量进行加法操作，并返回结果
        def fn(a):
            b = a + 1
            return (b,)

        # 调用 self.common 方法执行测试，传入 fn 函数和一个随机生成的大小为 1 的张量作为参数
        self.common(fn, (torch.randn(1),))

    @requires_gpu()
    def test_single_elem_indirect(self):
        # 定义测试函数 fn，对输入张量进行索引加法操作，并返回结果
        def fn(a, b):
            c = a[b] + 1
            return (c,)

        # 创建一个随机生成的大小为 1 的张量 a 和一个索引张量 b
        a = torch.randn(1)
        b = (torch.tensor([0], dtype=torch.int64),)

        # 调用 self.common 方法执行测试，传入 fn 函数和张量 a、b 作为参数
        self.common(fn, (a, b))
    # 此测试旨在检查逻辑问题，即如果 XBLOCK 能整除 xnumel，则从 trito 加载/存储中去除 xmask

    @requires_gpu()
    def test_xblock_divides_xnumel(self):
        def fn(a):
            b = a + 1
            return (b,)

        # 假设 XBLOCK 总是 1024 的除数
        # 因此当 xnumel 是 1024 的倍数时，xmask 将被去除
        self.common(fn, (torch.randn(1024),))
        self.common(fn, (torch.randn(1025),))

    def test_inplace_mixed_dtype_ops(self):
        @torch._dynamo.optimize("inductor")
        def fn(x, y):
            z = x + y.float()
            w = z.add_(y)
            return w.mul_(y)

        inputs = (
            rand_strided((4, 4), (4, 1), device=self.device, dtype=torch.float),
            rand_strided((4, 4), (4, 1), device=self.device, dtype=torch.double),
        )
        out = fn(*inputs)
        out_eager = (inputs[0] + inputs[1].float()).add_(inputs[1]).mul_(inputs[1])
        self.assertTrue(same(out, out_eager))

    @config.patch(
        {"triton.unique_kernel_names": True, "triton.descriptive_names": False}
    )
    def test_kernel_names(self):
        @torch._dynamo.optimize("inductor")
        def fn(x):
            return 2 * x

        inputs = (rand_strided((8,), (1,), device=self.device),)
        self.assertTrue(same(fn(*inputs), 2 * inputs[0]))

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_strided_inputs(self):
        @torch._dynamo.optimize("inductor")
        def fn(x, y):
            return x + y

        inputs = (
            rand_strided((8, 16), (32, 2), device=self.device),
            rand_strided((8, 16), (16, 1), device=self.device),
        )
        self.assertTrue(same(fn(*inputs), inputs[0] + inputs[1]))

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_input_mutation1(self):
        def fn(a):
            b = a + 1
            a.copy_(b)
            c = a + 2
            return a * b / c

        arg1 = torch.randn(64, device=self.device)
        arg2 = arg1.clone()
        arg3 = torch.randn(64, device=self.device)
        arg4 = arg3.clone()
        correct1 = fn(arg1)
        correct2 = fn(arg3)
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        actual1 = opt_fn(arg2)
        actual2 = opt_fn(arg4)

        self.assertTrue(same(actual1, correct1))
        self.assertTrue(same(actual2, correct2))
        self.assertTrue(same(arg1, arg2))
        self.assertTrue(same(arg3, arg4))
    def test_slice_mutation2(self):
        # 定义一个函数 fn，它接受参数 a，并对其进行操作
        def fn(a):
            # 将 a 的列索引 20 到 40 的部分加 1
            a[:, 20:40] = a[:, 20:40] + 1
            # 将 a 的列索引 2 到 11 的部分赋值为 a 的列索引 1 到 10 的部分加 2
            a[:, 2:11] = a[:, 1:10] + 2

        # 创建一个在当前设备上随机初始化的张量 arg1
        arg1 = torch.randn([1, 64], device=self.device)
        # 克隆 arg1 以备后用
        arg2 = arg1.clone()
        # 调用函数 fn，并传入 arg1，直接修改 arg1
        fn(arg1)
        # 通过动态优化器对函数 fn 进行优化编译
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        # 使用优化后的函数 opt_fn 并传入克隆的 arg2，直接修改 arg2
        opt_fn(arg2)
        # 断言 arg1 和 arg2 在数值上相等
        self.assertTrue(same(arg1, arg2))
    # 定义一个测试函数 test_slice_mutation3，用于测试切片操作的变异性
    def test_slice_mutation3(self):
        # 定义一个内部函数 fn，接受一个参数 a
        def fn(a):
            # 对 a 的左上角 2x2 区域进行填充操作，填充值为 10
            a[:2, :2].fill_(10)

        # 使用 torch._dynamo.optimize_assert(compile_fx) 对 fn 进行优化
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)

        # 创建一个设备为 self.device 的 8x8 随机张量 x1，并复制给 x2
        x1 = torch.randn(8, 8, device=self.device)
        x2 = x1.clone()
        
        # 分别对 x1 和 x2 执行 fn 函数和 opt_fn 函数
        fn(x1)
        opt_fn(x2)
        
        # 断言 x1 和 x2 的值相等
        self.assertEqual(x1, x2)

    # 定义一个测试函数 test_tensor_index_slice，用于测试张量索引和切片操作
    def test_tensor_index_slice(self):
        # 定义一个内部函数 fn，接受一个参数 a
        def fn(a):
            # 创建设备为 self.device 的张量 x 和 y
            x = torch.tensor([1, 2], device=self.device)
            y = torch.tensor([2, 3], device=self.device)
            
            # 创建设备为 self.device 的 1x2 张量 xx 和 3x1 张量 yy
            xx = torch.tensor([1, 2], device=self.device).view(1, 2)
            yy = torch.tensor([1, 2, 3], device=self.device).view(3, 1)
            
            # 返回一系列索引和切片操作结果
            return [
                a[x, y],
                a[:, x, y],
                a[:, x, y, :],
                a[x, :, y],
                a[:, x, :, y, :],
                a[xx, yy],
                a[:, xx, yy],
                a[xx, :, yy],
                a[xx, yy, :],
                a[:, xx, :, yy],
            ]

        # 创建一个设备为 self.device 的张量 a，形状为 3x4x5x6x7
        a = torch.arange(3 * 4 * 5 * 6 * 7, device=self.device).view(3, 4, 5, 6, 7)
        
        # 调用 fn 函数获取参考结果 refs
        refs = fn(a)
        
        # 使用 torch.compile(fn) 对 fn 进行编译，得到测试结果 tests
        tests = torch.compile(fn)(a)
        
        # 逐一比较 refs 和 tests 中的每一对张量，确保它们在数值上接近
        for ref, test in zip(refs, tests):
            torch.testing.assert_close(ref, test)

    # 使用 torch._dynamo.config.patch 设置缓存大小限制为 10 的装饰器，定义测试函数 test_tensor_index_put_slice
    @torch._dynamo.config.patch(cache_size_limit=10)
    def test_tensor_index_put_slice(self):
        # 定义一个函数 fn，接受两个参数 a 和 version
        def fn(a, version):
            # 创建设备为 self.device、dtype 为 torch.int32 的张量 x 和 y
            x = torch.tensor([1, 2], device=self.device, dtype=torch.int32)
            y = torch.tensor([2, 3], device=self.device, dtype=torch.int32)

            # 创建设备为 self.device 的 1x2 张量 xx 和 3x1 张量 yy
            xx = torch.tensor([1, 2], device=self.device).view(1, 2)
            yy = torch.tensor([1, 2, 3], device=self.device).view(3, 1)

            # 根据 version 的值选择不同的切片操作，并将相应区域置为与 a[x, y] 形状相同的零张量
            if version == 0:
                a[x, y] = torch.zeros_like(a[x, y])
            elif version == 1:
                a[:, x, y] = torch.zeros_like(a[:, x, y])
            elif version == 2:
                a[:, x, y, :] = torch.zeros_like(a[:, x, y, :])
            elif version == 3:
                a[x, :, y] = torch.zeros_like(a[x, :, y])
            elif version == 4:
                a[:, x, :, y, :] = torch.zeros_like(a[:, x, :, y, :])
            elif version == 5:
                a[xx, yy] = torch.zeros_like(a[xx, yy])
            elif version == 6:
                a[:, xx, yy] = torch.zeros_like(a[:, xx, yy])
            elif version == 7:
                a[xx, :, yy] = torch.zeros_like(a[xx, :, yy])
            elif version == 8:
                a[xx, yy, :] = torch.zeros_like(a[xx, yy, :])
            elif version == 9:
                a[:, xx, :, yy] = torch.zeros_like(a[:, xx, :, yy])

            return a

        # 创建一个设备为 self.device、dtype 为 torch.int32 的张量 a，形状为 3x4x5x6x7
        a = torch.arange(3 * 4 * 5 * 6 * 7, device=self.device, dtype=torch.int32).view(
            3, 4, 5, 6, 7
        )

        # 对 version 从 0 到 9 的每个值，分别进行 fn 函数的参考结果 ref 和测试结果 test 的比较
        for i in range(10):
            ref = fn(torch.clone(a), i)
            test = torch.compile(fn)(torch.clone(a), i)
            torch.testing.assert_close(ref, test)
    # 定义一个测试函数，用于测试间接加载广播功能
    def test_indirect_load_broadcast(self):
        # 定义一个内部函数 fn，接受三个指针参数，返回根据指定规则聚集后的张量
        def fn(in_ptr0, in_ptr1, in_ptr2):
            return torch.gather(in_ptr1, 0, in_ptr2) + in_ptr0

        # 创建一个形状为 (32, 21) 的随机张量，设备为 self.device，数据类型为 torch.int64
        arg190 = rand_strided((32, 21), (1, 32), device=self.device, dtype=torch.int64)
        # 将 arg190 张量填充为 0
        arg190.fill_(0)
        # 创建一个形状为 (9521, 512) 的随机张量，设备为 self.device，数据类型为 torch.float32
        arg111 = rand_strided(
            (9521, 512), (512, 1), device=self.device, dtype=torch.float32
        )
        # 调用 self.common 方法，传入 fn 函数和其参数元组
        self.common(
            fn,
            (
                torch.randn(32, 1),  # 形状为 (32, 1) 的随机张量
                arg111,  # arg111 随机张量
                arg190,  # arg190 随机张量
            ),
        )

    # 定义一个测试函数，用于测试 ROI Align 功能
    def test_roi_align(self):
        # 检查是否支持 torchvision 中的 ROI Align，如果不支持则抛出跳过测试的异常
        if not has_torchvision_roi_align():
            raise unittest.SkipTest("requires torchvision")

        # 定义一个内部函数 fn，接受两个参数 a 和 b，调用 torchvision 的 ROI Align 操作
        def fn(a, b):
            return torch.ops.torchvision.roi_align(a, b, 0.25, 7, 7, 2, False)

        # 调用 self.common 方法，传入 fn 函数和其参数元组
        self.common(fn, (torch.zeros([4, 256, 296, 304]), torch.zeros([2292, 5])))

    # 用于测试 NLL Loss 前向传播功能的函数装饰器，修改了 Halide 的 CUDA 调度器
    @config.patch("halide.scheduler_cuda", "Li2018")
    def test_nll_loss_forward(self):
        # 定义一个内部函数 fn，接受两个参数 a 和 b，调用 aten.nll_loss_forward 函数计算 NLL Loss
        def fn(a, b):
            return aten.nll_loss_forward(a, b, None, 1, -100)

        # 定义标签张量 labels 和输入张量 inps
        labels = (
            torch.zeros([5], dtype=torch.int64),  # 形状为 (5,) 的零张量，数据类型为 torch.int64
            torch.tensor([-100, -100, 3, -100, -100], dtype=torch.int64),  # 包含指定数据的张量
        )
        inps = (torch.randn(5, 5), torch.randn(5, 5))  # 两个形状为 (5, 5) 的随机张量
        # 使用 zip 遍历输入张量和标签，调用 self.common 方法进行测试
        for a, b in zip(inps, labels):
            self.common(
                fn,
                (a, b),
            )

    # 用于测试 NLL Loss 反向传播功能的函数
    def test_nll_loss_backward(self):
        # 定义一个内部函数 fn，接受四个参数 a、b、c 和一个张量，调用 aten.nll_loss_backward 函数
        def fn(a, b, c):
            return aten.nll_loss_backward(
                a, b, c, None, 1, -100, torch.tensor(1.0, device=self.device)
            )

        # 定义标签张量 labels 和输入张量 inps
        labels = (
            torch.zeros([5], dtype=torch.int64),  # 形状为 (5,) 的零张量，数据类型为 torch.int64
            torch.tensor([-100, -100, 3, -100, -100], dtype=torch.int64),  # 包含指定数据的张量
        )
        inps = (torch.randn(5, 5), torch.randn(5, 5))  # 两个形状为 (5, 5) 的随机张量
        grad_outs = (torch.randn(()), torch.randn(()))  # 两个随机梯度张量
        # 使用 zip 遍历梯度输出、输入张量和标签，调用 self.common 方法进行测试
        for a, b, c in zip(grad_outs, inps, labels):
            self.common(
                fn,
                (a, b, c),
            )

    # 用于测试 isinf 和 isnan 方法的函数
    def test_isinf(self):
        # 定义一个内部函数 fn，接受一个参数 x，返回 x 中的无穷和 NaN 值的布尔张量
        def fn(x):
            return x.isinf(), x.isnan()

        # 调用 self.common 方法，传入 fn 函数和一个包含特定浮点数的张量列表
        self.common(
            fn, [torch.tensor([1, float("inf"), 2, float("-inf"), float("nan")])]
        )
        # 调用 self.common 方法，传入 fn 函数和一个包含特定浮点数的张量列表，数据类型为 torch.float64
        self.common(
            fn,
            [
                torch.tensor(
                    [1, float("inf"), 2, float("-inf"), float("nan")],
                    dtype=torch.float64,
                )
            ],
        )

    # 用于测试 isinf 方法的函数，当使用 Halide 时跳过测试
    @skip_if_halide  # different nan behavior in ==
    def test_isinf2(self):
        # 定义一个内部函数 fn，接受一个参数 x，返回 x 是否等于特定张量的布尔张量
        def fn(x):
            # 创建一个包含特定浮点数的张量 y，设备为 self.device
            y = torch.tensor(
                [1, float("inf"), 2, float("-inf"), float("nan")], device=self.device
            )
            return x == y

        # 调用 self.common 方法，传入 fn 函数和一个包含特定浮点数的张量元组
        self.common(
            fn, (torch.tensor([1, float("inf"), 2, float("-inf"), float("nan")]),)
        )
    # 定义一个测试方法，用于测试任意操作
    def test_any(self):
        # 定义一个内部函数 fn，接受参数 x
        def fn(x):
            # 返回四个元组值：
            # 1. 沿着最后一个维度检查是否有任意 True 的结果
            # 2. 检查 x 是否包含无穷大的值中是否有任意 True 的结果
            # 3. 沿着第0维检查 x 中是否所有值都是无穷大的结果
            # 4. 检查 x 中是否所有值都不是无穷大的结果
            return (
                x.any(-1),
                x.isinf().any(),
                torch.all(x.isinf(), dim=0),
                torch.all(torch.logical_not(x.isinf())),
            )

        # 使用 self.common 方法测试 fn 函数，传入一个随机数的负值列表作为参数
        self.common(fn, [-torch.rand(64)])

        # 创建一个大小为16x8的随机数张量 tmp
        tmp = torch.randn(16, 8)
        # 将 tmp 的索引 (1, 1) 处的值设置为正无穷
        tmp[1, 1] = float("inf")
        # 使用 self.common 方法测试 fn 函数，传入 tmp 作为参数
        self.common(fn, [tmp])

    # 定义一个测试方法，用于测试多层任意操作
    def test_multilayer_any(self):
        # 定义一个内部函数 fn，接受参数 x
        def fn(x):
            # 返回一个元组值：
            # 1. 检查 x 是否包含无穷大的值中是否有任意 True 的结果
            # 2. 检查 x 中是否所有值都是有限的结果
            return (x.isinf().any(), x.isfinite().all())

        # 创建一个大小为 9x3x353x353 的随机数张量 sample
        sample = torch.rand(9, 3, 353, 353)
        # 使用 self.common 方法测试 fn 函数，传入 sample 作为参数
        self.common(fn, [sample])

        # 将 sample 拉平后的最后一个元素设置为正无穷
        sample.view(-1)[-1] = float("inf")
        # 使用 self.common 方法测试 fn 函数，传入 sample 作为参数
        self.common(fn, [sample])

    # 定义一个测试方法，用于测试原地激活函数
    def test_inplace_activations(self):
        # 定义一个内部函数 fn，接受参数 x
        def fn(x):
            # 分别对 x 执行原地的不同激活函数操作，并将结果存储在各个变量中
            a = aten.hardswish_(x + 1)
            b = aten.hardtanh_(x + 1)
            c = aten.leaky_relu_(x + 1)
            d = aten.silu_(x + 1)
            e = aten.log1p(x + 1)
            f = aten.masked_fill_(x + 1, torch.zeros_like(x, dtype=torch.bool), 99.0)
            h = aten.masked_fill_(x + 1, torch.ones_like(x, dtype=torch.bool), 99.0)
            # 返回一个包含所有操作结果的元组
            return (a, b, c, d, e, f, h)

        # 使用 self.common 方法测试 fn 函数，传入一个大小为64的随机数张量的10倍
        self.common(fn, [torch.randn(64) * 10])

    # 定义一个测试方法，用于测试 baddbmm 函数
    def test_baddbmm(self):
        # 定义一个内部函数 fn，接受参数 a, b, c, beta
        def fn(a, b, c, beta):
            # 调用 aten.baddbmm 函数计算结果并返回
            return aten.baddbmm(a, b, c, beta=beta)

        # 创建两个大小分别为 6x128x64 和 6x64x100 的随机数张量 b 和 c
        b = torch.randn(6, 128, 64)
        c = torch.randn(6, 64, 100)
        # 使用 itertools.product 生成参数组合列表
        options = itertools.product(
            [torch.randn(6, 1, 100), torch.randn(6, 1, 100).fill_(torch.nan)],
            [0.0, 1.0],
        )
        # 遍历参数组合列表，并使用 self.common 方法测试 fn 函数
        for a, beta in options:
            self.common(
                fn,
                [a, b, c, beta],
                # 设置绝对容差和相对容差的值
                # Mismatched elements: 1212 / 76800 (1.6%)
                # Greatest absolute difference: 0.001953125 at index (0, 0, 93) (up to 1e-05 allowed)
                # Greatest relative difference: 1.0 at index (3, 19, 4) (up to 0.001 allowed)
                atol=0.002,
                rtol=0.001,
            )

    # 使用 config.patch 装饰器定义一个测试方法，用于测试融合分块操作
    @config.patch({"triton.max_tiles": 2})
    def test_fuse_tiled(self):
        # 定义一个内部函数 fn，接受参数 a, b, c
        def fn(a, b, c):
            # 返回两个张量的加法结果以及第二个张量加1的结果
            return a + b, c + 1

        # 使用 self.common 方法测试 fn 函数，传入三个随机数张量作为参数
        self.common(
            fn, [torch.randn(128, 1), torch.randn(1, 128), torch.randn(128, 128)]
        )

    # 定义一个测试方法，用于测试 expand_as 函数
    def test_expand_as(self):
        # 定义一个内部函数 fn，接受参数 a, b
        def fn(a, b):
            # 返回两个张量使用 expand_as 方法扩展后的结果，
            # 以及分别加1后再使用 expand_as 方法扩展后的结果加1
            return aten.expand_as(a, b), aten.expand_as(a + 1, b + 1) + 1

        # 使用 self.common 方法测试 fn 函数，传入两个不同大小的随机数张量作为参数
        self.common(
            fn,
            [
                torch.randn(6, 1, 100),
                torch.randn(6, 128, 100),
            ],
        )

    # 定义一个测试方法，用于测试 index_put 函数
    def test_index_put1(self):
        # 定义一个内部函数 fn，接受参数 a, b, c
        def fn(a, b, c):
            # 返回两个张量通过 index_put 方法操作后的结果
            return (
                torch.index_put(a, [b], c),
                torch.index_put_(a + 1, [b + 1], c + 1) + 1,
            )

        # 使用 self.common 方法测试 fn 函数，传入三个不同大小的随机数张量作为参数
        self.common(
            fn,
            [
                torch.randn([800, 256, 7, 7]),
                torch.randperm(601),
                torch.randn([601, 256, 7, 7]),
            ],
        )
        # 再次使用 self.common 方法测试 fn 函数，传入三个不同大小的随机数张量作为参数
        self.common(
            fn, [torch.randn(1024, 4, 2), torch.arange(4), torch.randn(4, 1, 1)]
        )
    # 定义测试函数 test_index_put2，用于测试 torch.index_put 的功能
    def test_index_put2(self):
        # 定义内部函数 fn，用于执行 torch.index_put 操作
        def fn(a, b, c):
            # 执行 torch.index_put 操作，将张量 a 中索引为 b 的位置替换为张量 c 的值
            return torch.index_put(a, [b], c, True)

        # 调用通用测试函数 common 进行测试
        self.common(
            fn,
            [
                torch.randn([100, 256, 7, 7]),     # 生成随机张量作为参数 a
                torch.randint(0, 100, size=[600], dtype=torch.int64),  # 生成随机整数张量作为参数 b
                torch.randn([600, 256, 7, 7]),     # 生成随机张量作为参数 c
            ],
            # 传递额外参数 check_lowp=False，用于解决已知问题 https://github.com/openai/triton/issues/558
            check_lowp=False,
        )

    # 定义测试函数 test_index_put3，用于测试 torch.ops.aten.index_put_ 的功能
    def test_index_put3(self):
        # 定义内部函数 fn，用于执行 torch.ops.aten.index_put_ 操作
        def fn(a, b, c):
            # 执行 torch.ops.aten.index_put_ 操作，使用索引元组 (None, b, None) 将张量 c 插入张量 a 中
            torch.ops.aten.index_put_(a, (None, b, None), c)
            # 创建新张量 a1，为 a 的副本加 1
            a1 = a + 1
            # 执行 torch.ops.aten.index_put_ 操作，使用索引元组 (None, b + 1, None) 将张量 c + 1 插入张量 a1 中
            torch.ops.aten.index_put_(a1, (None, b + 1, None), c + 1)
            # 返回元组 (a, a1)
            return (a, a1)

        # 调用通用测试函数 common 进行测试
        self.common(
            fn,
            [
                torch.randn([1024, 4, 2]),   # 生成随机张量作为参数 a
                torch.arange(3),            # 生成序列张量作为参数 b
                torch.randn([1024, 1, 2]),   # 生成随机张量作为参数 c
            ],
        )

    # 定义测试函数 test_index_put4，用于测试 torch.index_put 的功能
    def test_index_put4(self):
        # 提示：a, b[0] are not broadcastable
        # https://github.com/pytorch/pytorch/issues/97104
        # 定义内部函数 fn，用于执行 torch.index_put 操作
        def fn(a, b, c):
            # 执行 torch.index_put 操作，将张量 a 中索引为 b 的位置替换为张量 c 的值
            return torch.index_put(a, [b], c)

        # 调用通用测试函数 common 进行测试
        self.common(
            fn,
            [
                torch.rand([8, 2]),         # 生成随机张量作为参数 a
                torch.rand([8]) > 0.5,      # 生成布尔张量作为参数 b
                torch.rand([]),             # 生成随机标量张量作为参数 c
            ],
        )

    # 定义测试函数 test_index_put_as_masked_fill，用于测试 torch.ops.aten.index_put_ 的功能
    def test_index_put_as_masked_fill(self):
        # 定义内部函数 fn，用于执行 torch.ops.aten.index_put_ 操作
        def fn(a, b, c, d):
            # 克隆张量 a，保留原始数据
            a = a.clone()
            # 执行 torch.ops.aten.index_put_ 操作，使用索引列表 [b] 将张量 c 插入张量 a 中，采用 d 作为参数
            torch.ops.aten.index_put_(a, [b], c, d)
            # 返回修改后的张量 a
            return a

        # 调用通用测试函数 common 进行测试
        self.common(
            fn,
            (
                torch.randn([1024, 4, 2]),   # 生成随机张量作为参数 a
                torch.randn([1024, 4, 2]) > 0,  # 生成布尔张量作为参数 b
                torch.randn([]),             # 生成随机标量张量作为参数 c
                False,                      # 传递布尔参数 d
            ),
        )

        # 调用通用测试函数 common 进行测试
        self.common(
            fn,
            (
                torch.randn([1024, 4, 2]),   # 生成随机张量作为参数 a
                torch.randn([1024, 4, 2]) > 0,  # 生成布尔张量作为参数 b
                torch.randn([]),             # 生成随机标量张量作为参数 c
                True,                       # 传递布尔参数 d
            ),
        )

    # 定义测试函数 test_index_put_fallback1，用于测试 torch.ops.aten.index_put_ 的功能
    def test_index_put_fallback1(self):
        # 定义内部函数 fn，用于执行 torch.ops.aten.index_put_ 操作
        def fn(a, b, c, d):
            # 克隆张量 a，保留原始数据
            a = a.clone()
            # 执行 torch.ops.aten.index_put_ 操作，使用索引列表 [b] 将张量 c 插入张量 a 中，采用 d 作为参数
            torch.ops.aten.index_put_(a, [b], c, d)
            # 返回修改后的张量 a
            return a

        # 调用通用测试函数 common 进行测试
        self.common(
            fn,
            (
                torch.randn([3]),           # 生成随机张量作为参数 a
                torch.as_tensor([True, True, False]),  # 生成布尔张量作为参数 b
                torch.randn([2]),           # 生成随机张量作为参数 c
                False,                      # 传递布尔参数 d
            ),
        )

        # 调用通用测试函数 common 进行测试
        self.common(
            fn,
            (
                torch.randn([3]),           # 生成随机张量作为参数 a
                torch.as_tensor([True, True, False]),  # 生成布尔张量作为参数 b
                torch.randn([2]),           # 生成随机张量作为参数 c
                True,                       # 传递布尔参数 d
            ),
        )
    # 定义测试函数 test_index_put_fallback2，接受五个参数 a, b, c, d, e
    def test_index_put_fallback2(self):
        # 定义内部函数 fn，对参数 a 进行克隆操作，然后调用 torch.ops.aten.index_put_ 对 a 进行索引更新
        def fn(a, b, c, d, e):
            a = a.clone()
            torch.ops.aten.index_put_(a, [None, b, c], d, e)
            return a

        # 调用共用测试函数 self.common，分别传入 fn 函数和多个参数
        self.common(
            fn,
            (
                torch.randn([1, 2, 3]),  # 参数 a，随机生成的张量形状为 [1, 2, 3]
                torch.as_tensor([0, 1]),  # 参数 b，张量 [0, 1]
                torch.as_tensor([True, True, False]),  # 参数 c，张量 [True, True, False]
                torch.randn([]),  # 参数 d，随机生成的标量张量
                False,  # 参数 e，布尔值 False
            ),
        )
        # 再次调用 self.common，传入不同的参数
        self.common(
            fn,
            (
                torch.randn([1, 2, 3]),  # 参数 a，随机生成的张量形状为 [1, 2, 3]
                torch.as_tensor([0, 1]),  # 参数 b，张量 [0, 1]
                torch.as_tensor([True, True, False]),  # 参数 c，张量 [True, True, False]
                torch.randn([]),  # 参数 d，随机生成的标量张量
                True,  # 参数 e，布尔值 True
            ),
        )

    # 定义测试函数 test_index_put_deterministic_fallback，使用 DeterministicGuard 确保代码执行的确定性
    def test_index_put_deterministic_fallback(self):
        with DeterministicGuard(True):
            # 定义函数 fn，调用 torch.index_put 对 a 进行索引更新
            def fn(a, b, c):
                return torch.index_put(a, [b], c, True)

            # 调用共用测试函数 self.common，传入 fn 函数和参数列表
            self.common(
                fn,
                [
                    torch.randn([100, 32]),  # 参数 a，随机生成的张量形状为 [100, 32]
                    torch.randint(0, 100, size=[600], dtype=torch.int64),  # 参数 b，随机生成的整数张量
                    torch.randn([600, 32]),  # 参数 c，随机生成的张量形状为 [600, 32]
                ],
                check_lowp=False,  # 关键字参数 check_lowp 设为 False
            )

    # 使用 @skip_if_gpu_halide 装饰器跳过 GPU Halide 相关测试
    @skip_if_gpu_halide  # https://github.com/halide/Halide/issues/8312
    # 定义测试函数 test_index_put_index
    def test_index_put_index(self):
        # 定义函数 fn，调用 torch.ops.aten.index_put.default 对 x 进行索引更新
        def fn(ind, x, src):
            y = torch.ops.aten.index_put.default(x, [ind], src)
            return torch.ops.aten.index.Tensor(y, [ind])

        # 准备参数列表 args
        args = [torch.tensor([1], dtype=torch.int64), torch.randn(8, 4), torch.randn(4)]
        # 调用共用测试函数 self.common，传入 fn 函数和参数列表 args
        self.common(fn, args)

    # 定义测试函数 test_index_put_reinplace
    def test_index_put_reinplace(self):
        # 定义函数 fn，使用 x.index_put_ 进行原地索引更新，并返回扩展后的张量
        def fn(x, idx):
            src = torch.ones(idx.size(0), device=x.device)
            x.index_put_((idx,), src)
            return x.expand((2, x.shape[0]))

        # 初始化参数 a 和 idx
        a = torch.randn(1024)  # 随机生成的张量形状为 [1024]
        idx = torch.arange(10)  # 张量 [0, 1, 2, ..., 9]
        torch._inductor.metrics.generated_kernel_count = 0
        # 调用共用测试函数 self.common，传入 fn 函数和参数元组 (a, idx)
        self.common(fn, (a, idx))
        # 断言生成的内核数目与预期相等
        assertGeneratedKernelCountEqual(self, 1)

    # 定义测试函数 test_index_put_failed_reinplace
    def test_index_put_failed_reinplace(self):
        # 定义函数 fn，使用 x.index_put 对 x 进行索引更新，返回更新后的张量 x 和新张量 y
        def fn(x, idx):
            src = torch.ones(idx.size(0), device=x.device)
            y = x.index_put((idx,), src)
            return x, y

        # 初始化参数 a 和 idx
        a = torch.randn(1024)  # 随机生成的张量形状为 [1024]
        idx = torch.arange(10)  # 张量 [0, 1, 2, ..., 9]
        torch._inductor.metrics.generated_kernel_count = 0
        # 调用共用测试函数 self.common，传入 fn 函数和参数元组 (a, idx)
        self.common(fn, (a, idx))
        # 断言生成的内核数目与预期相等
        assertGeneratedKernelCountEqual(self, 2)

    # 定义测试函数 test_adding_tensor_offsets
    def test_adding_tensor_offsets(self):
        # 使用 @torch.compile(fullgraph=True) 装饰器编译函数 fn，使其全图编译
        @torch.compile(fullgraph=True)
        # 定义函数 fn，返回张量 x 的子张量 [16:32]
        def fn(x):
            return x[16:32]

        # 禁用梯度追踪环境
        with torch.no_grad():
            # 生成随机张量 x，并进行断言比较
            x = torch.randn(1024, device=self.device)
            self.assertEqual(fn(x[0:]), x[16:][:16])
            self.assertEqual(fn(x[128:]), x[128 + 16 :][:16])

    # 来自 GPT2ForSequenceClassification 的注释
    # TODO(jansel): incorrect results with Anderson, report bug
    # 使用 @config.patch("halide.scheduler_cuda", "Li2018") 装饰器进行配置修补
    @config.patch("halide.scheduler_cuda", "Li2018")
    def test_index_tensor(self):
        # 定义一个嵌套函数 fn，接受两个参数 x 和 y
        def fn(x, y):
            # 使用 torch.ops.aten.ne.Scalar 函数计算 x 是否不等于 0 的元素
            ne = torch.ops.aten.ne.Scalar(x, 0)
            # 对 ne 张量沿着最后一个维度求和
            sum = torch.ops.aten.sum.dim_IntList(ne, [-1])
            # 对 sum 张量减去标量 1
            sub = torch.ops.aten.sub.Tensor(sum, 1)
            # 使用 torch.ops.prims.iota.default 创建一个从 0 开始，步长为 1 的整数张量
            # 数据类型为 torch.int64，在 x 的设备上，不需要梯度
            iota = torch.ops.prims.iota.default(
                1,
                start=0,
                step=1,
                dtype=torch.int64,
                device=x.device,
                requires_grad=False,
            )
            # 使用 torch.ops.aten.index.Tensor 函数对张量 y 进行索引操作
            # 第一个维度使用 iota 张量，第二个维度使用 sub 张量
            return torch.ops.aten.index.Tensor(y, [iota, sub])

        # 调用 self.common 方法，传入 fn 函数和参数列表 [torch.randn(1, 1024), torch.randn(1, 1024, 2)]
        self.common(fn, [torch.randn(1, 1024), torch.randn(1, 1024, 2)])

    @config.patch(fallback_random=True)
    def test_bernoulli1(self):
        # 定义一个嵌套函数 fn，接受一个参数 a
        def fn(a):
            # 创建一个与 a 张量相同大小的空张量 b
            b = torch.empty_like(a)
            # 调用 aten.bernoulli_ 函数，在 b 上生成伯努利分布的随机数，并返回修改后的 b 张量
            return aten.bernoulli_(b), b

        # 调用 self.common 方法，传入 fn 函数和参数列表 [torch.randn([100])]
        self.common(
            fn,
            [
                torch.randn([100]),
            ],
        )

    @skip_if_halide  # rng
    def test_bernoulli2(self):
        # 定义一个嵌套函数 fn，接受一个参数 a
        def fn(a):
            # 调用 aten.bernoulli 函数，在 a 上生成伯努利分布的随机数张量
            return aten.bernoulli(a)

        # 调用 self.common 方法，传入 fn 函数和参数列表 [torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0])]
        self.common(
            fn,
            [torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0])],
        )

    def test_narrow(self):
        # 定义一个嵌套函数 fn，接受一个参数 x
        def fn(x):
            # 使用 aten.narrow 函数对 x 进行窄化操作，从第 1 维开始，长度为 16
            narrow_result_1 = aten.narrow(x, 1, 10, 16)
            # 对 x 加 2，再使用 aten.narrow 函数进行窄化操作，从第 0 维开始，长度为 16
            # 然后再加上标量 1
            narrow_result_2 = aten.narrow(x + 2, 0, 10, 16) + 1
            # 使用 aten.narrow_copy 函数对 x 进行窄化操作，从第 1 维开始，长度为 16
            narrow_copy_result = aten.narrow_copy(x, 1, 10, 16)
            return narrow_result_1, narrow_result_2, narrow_copy_result

        # 调用 self.common 方法，传入 fn 函数和参数列表 [torch.randn(64, 64)]
        self.common(fn, [torch.randn(64, 64)])

    def test_new_cpp_build_logical(self):
        # 导入 validate_new_cpp_commands 函数并调用
        from torch._inductor.codecache import validate_new_cpp_commands

        validate_new_cpp_commands()

    def test_as_strided(self):
        # 定义一个嵌套函数 fn，接受一个参数 x
        def fn(x):
            # 使用 aten.as_strided 函数对 x 进行视图操作，指定大小和步长
            as_strided_result_1 = aten.as_strided(x, (8, 8, 64), (8 * 64, 64, 1), 0)
            # 对 x 加 1，再使用 aten.as_strided 函数对 x 进行视图操作，指定大小和步长
            # 然后再加上标量 2
            as_strided_result_2 = aten.as_strided(x + 1, (8, 8, 64), (8 * 64, 64, 1), 0) + 2
            return as_strided_result_1, as_strided_result_2

        # 定义一个嵌套函数 fn_channels_last，接受一个参数 x
        def fn_channels_last(x):
            # 使用 aten.as_strided 函数对 x 进行视图操作，指定大小和步长
            as_strided_result_1 = aten.as_strided(
                x, (8, 384, 2, 20, 12), (153600, 1, 61440, 384, 7680), 0
            )
            # 对 x 加 1，再使用 aten.as_strided 函数对 x 进行视图操作，指定大小和步长
            # 然后再加上标量 2
            as_strided_result_2 = aten.as_strided(
                x + 1, (8, 384, 2, 20, 12), (153600, 1, 61440, 384, 7680), 0
            ) + 2
            return as_strided_result_1, as_strided_result_2

        # 调用 self.common 方法，传入 fn 函数和参数列表 [torch.randn(64, 64)]
        self.common(fn, [torch.randn(64, 64)])
        # 调用 self.common 方法，传入 fn_channels_last 函数和参数列表，参数为一个具有通道顺序的张量
        self.common(
            fn_channels_last,
            [torch.randn(8, 384, 20, 20).to(memory_format=torch.channels_last)],
        )

    def test_like_channels_last(self):
        # 定义一个名为 foo 的函数，不接受参数
        def foo():
            # 生成一个随机张量 randn，设备为 self.device，数据类型为 torch.float32
            randn = torch.randn((4, 3, 8, 8), device=self.device, dtype=torch.float32)
            # 将 randn 转换为通道顺序为 channels_last 的连续张量 xc
            xc = randn.contiguous(memory_format=torch.channels_last)
            # 创建一个与 xc 张量大小相同的零张量 clone，保留内存格式
            clone = torch.zeros_like(xc, memory_format=torch.preserve_format)
            # 创建一个与 randn 张量大小相同的随机张量 rand_like
            rand_like = torch.rand_like(randn)
            return (xc, clone, rand_like)

        # 调用 foo 函数，将结果存储在 out 中
        out = foo()
        # 调用 torch.compile()(foo) 函数，将结果存储在 out_comp 中
        out_comp = torch.compile()(foo)()

        # 对 out 和 out_comp 中的每对张量 t 和 t_comp 进行比较，要求它们的步长相同
        for t, t_comp in zip(out, out_comp):
            self.assertEqual(t.stride(), t_comp.stride())
    def test_as_strided_scatter(self):
        # 定义一个测试函数，用于测试 aten.as_strided_scatter 函数
        def fn(a, b):
            # 调用 aten.as_strided_scatter 函数，传入参数：
            #   - a * 8 + 10 作为第一个参数
            #   - b * 2 - 4 作为第二个参数
            #   - size 为 (a.shape[0], a.shape[1] // 2)
            #   - stride 为 (a.shape[1], 2)
            #   - storage_offset 为 0
            return aten.as_strided_scatter(
                a * 8 + 10,
                b * 2 - 4,
                size=(a.shape[0], a.shape[1] // 2),
                stride=(a.shape[1], 2),
                storage_offset=0,
            )

        # 调用 self.common 函数进行测试
        self.common(fn, [torch.randn(10, 1024), torch.randn(10, 512)])

    def test_select_scatter(self):
        # 定义一个测试函数，用于测试 aten.select_scatter 函数
        def fn(x, a, b):
            # 调用 aten.select_scatter 函数两次，分别传入参数：
            #   - x, a, 1, 0
            #   - x, b, 0, 1
            return (
                aten.select_scatter(x, a, 1, 0),
                aten.select_scatter(x, b, 0, 1),
            )

        # 调用 self.common 函数进行测试，传入参数列表
        self.common(
            fn,
            [
                torch.randn(8, 197, 38),
                torch.randn(8, 38),
                torch.randn(197, 38),
            ],
        )

    @skip_if_gpu_halide  # accuracy issue
    def test_slice_scatter(self):
        # 定义一个测试函数，用于测试 aten.slice_scatter 函数
        def fn(x, a):
            # 调用 aten.slice_scatter 函数两次，分别传入参数：
            #   - x, a, 2, 10, -10
            #   - x, a[:, :, :40], 2, 10, -10, 2
            return (
                aten.slice_scatter(x, a, 2, 10, -10),
                aten.slice_scatter(x, a[:, :, :40], 2, 10, -10, 2),
            )

        # 调用 self.common 函数进行测试，传入参数列表
        self.common(
            fn,
            [
                torch.randn(4, 8, 100),
                torch.randn(4, 8, 80),
            ],
        )

    def test_slice_scatter2(self):
        # 定义一个测试函数，用于测试 aten.slice_scatter 函数
        def fn(a, b):
            # 调用 aten.slice_scatter 函数，传入参数：
            #   - a, b, 0, 0, 9223372036854775807
            return aten.slice_scatter(a, b, 0, 0, 9223372036854775807)

        # 调用 self.common 函数进行测试，传入参数列表
        self.common(
            fn,
            [
                torch.randn([8, 197, 384]),
                torch.randn([8, 197, 384]),
            ],
        )

    def test_slice_scatter3(self):
        # 定义一个测试函数，用于测试 aten.slice_scatter.default 函数
        def fn(a, b):
            # 调用 aten.slice_scatter.default 函数，传入参数：
            #   - a, b, 1, 1, 9223372036854775807, 2
            return aten.slice_scatter.default(a, b, 1, 1, 9223372036854775807, 2)

        # 调用 self.common 函数进行测试，传入参数列表
        self.common(
            fn,
            [
                torch.randn([1, 4]),
                torch.randn([1, 2]),
            ],
        )

    def test_slice_scatter4(self):
        # 定义一个测试函数，用于测试 aten.slice_scatter.default 函数
        def fn(a, b):
            # 调用 aten.slice_scatter.default 函数，传入参数：
            #   - a, b, 1, 2, 9223372036854775807, 3
            return aten.slice_scatter.default(a, b, 1, 2, 9223372036854775807, 3)

        # 调用 self.common 函数进行测试，传入参数列表
        self.common(
            fn,
            [
                torch.randn([1, 9]),
                torch.randn([1, 3]),
            ],
        )

    def test_slice_scatter5(self):
        # 定义一个测试函数，用于测试 aten.slice_scatter.default 函数
        # 包含一些需要对开始或结束进行限制的空切片
        def fn(a, b):
            return (
                aten.slice_scatter.default(a, b, 0, 2, 0, 1),
                aten.slice_scatter.default(a, b, 0, a.shape[0], a.shape[0] + 10, 1),
                aten.slice_scatter.default(a, b, 0, -20, 0, 1),
                aten.slice_scatter.default(a, b, 0, -20, -16, 1),
            )

        # 创建一个张量 a 和一个空张量 b，调用 self.common 函数进行测试
        a = torch.arange(10, dtype=torch.float)
        b = torch.empty(0)
        self.common(fn, [a, b])
    def test_slice_scatter_reinplace(self):
        # 定义一个名为 M 的内部类，继承自 nn.Module
        class M(nn.Module):
            def __init__(self, device):
                super().__init__()
                # 初始化一个线性层，输入维度 64，输出维度 64，无偏置
                self.linear1 = nn.Linear(64, 64, bias=False)
                # 创建一个形状为 (56, 384, 8, 64) 的零张量，并移动到指定的设备上
                self.cache_k = torch.zeros((56, 384, 8, 64), device=device)

            def forward(self, x, start_pos):
                # 获取输入张量 x 的形状信息
                bsz, seqlen, _, _ = x.shape
                # 对输入张量 x 应用线性层 self.linear1
                xk = self.linear1(x)
                # 使用 torch.no_grad() 上下文管理器，将 xk 更新到 self.cache_k 中的指定位置
                with torch.no_grad():
                    self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
                # 从 self.cache_k 中获取 keys，以便后续计算使用
                keys = self.cache_k[:bsz, : start_pos + seqlen]
                # 计算 scores，这里涉及矩阵乘法操作
                scores = torch.matmul(
                    xk.transpose(1, 2), keys.transpose(1, 2).transpose(2, 3)
                )
                # 返回计算结果 scores
                return scores

        # 创建一个 M 类的实例 kv_cache_module，使用指定的设备
        kv_cache_module = M(self.device)
        # 生成一个形状为 (1, 32, 8, 64) 的随机张量作为输入
        inp = torch.randn(1, 32, 8, 64)

        # 使用 torch.no_grad() 上下文管理器，调用 self.common 方法进行测试
        # 确保缓存更新是就地更新，而不是复制-散布-复制回去的方式
        torch._inductor.metrics.generated_kernel_count = 0
        with torch.no_grad():
            self.common(kv_cache_module, (inp, 1), check_lowp=False)
        # 使用断言检查生成的内核数量是否等于 1
        assertGeneratedKernelCountEqual(self, 1)

    @skip_if_gpu_halide  # 如果在 GPU 上会编译错误，则跳过该测试
    def test_scatter1(self):
        # 定义一个函数 fn，实现对张量 a 在维度 dim 上的散射操作
        def fn(a, dim, index, b):
            return aten.scatter(a, dim, index, b)

        # 调用 self.common 方法进行测试，传入相应的参数
        self.common(
            fn,
            [
                torch.zeros(2, 3),  # a: 形状为 (2, 3) 的零张量
                -1,  # dim: 散射操作的维度
                torch.tensor([[0]]),  # index: 形状为 (1, 1) 的索引张量
                torch.ones(2, 3),  # b: 形状为 (2, 3) 的全一张量
            ],
        )

    def test_scatter2(self):
        # 如果当前设备是 "cuda"，则跳过测试，因为在 sm86 上不稳定
        if self.device == "cuda":
            raise unittest.SkipTest("unstable on sm86")

        # 初始化 check_lowp 为 True
        check_lowp = True
        # 如果当前设备是 "xpu"，则将 check_lowp 设置为 False
        if self.device == "xpu":
            check_lowp = False

        # 定义一个函数 fn，实现在维度 dim 上进行 reduce 操作的散射
        def fn(a, dim, index, b):
            return aten.scatter.reduce(a, dim, index, b, reduce="add")

        # 调用 self.common 方法进行测试，传入相应的参数和选项
        self.common(
            fn,
            [
                torch.zeros(64, 512),  # a: 形状为 (64, 512) 的零张量
                0,  # dim: 散射操作的维度
                torch.zeros((64, 512), dtype=torch.int64),  # index: 形状为 (64, 512) 的整数索引张量
                torch.ones(64, 512),  # b: 形状为 (64, 512) 的全一张量
            ],
            check_lowp=check_lowp,  # 是否检查低精度选项
        )

    def test_scatter3(self):
        # 定义一个函数 fn，实现在维度 dim 上进行 reduce="add" 操作的散射
        def fn(a, dim, index, b):
            return aten.scatter(a, dim, index, b, reduce="add")

        # 初始化 check_lowp 为 True
        check_lowp = True
        # 如果当前设备是 "xpu"，则将 check_lowp 设置为 False
        if self.device == "xpu":
            check_lowp = False

        # 调用 self.common 方法进行测试，传入相应的参数、选项和容差值
        self.common(
            fn,
            [
                torch.randn(5, 29, 13),  # a: 形状为 (5, 29, 13) 的随机张量
                2,  # dim: 散射操作的维度
                torch.tensor([[[3, 5, 7, 9]]]),  # index: 形状为 (1, 1, 4) 的索引张量
                0.8,  # b: 源可以是一个标量
            ],
            atol=2e-4,  # 绝对误差容差
            rtol=1e-3,  # 相对误差容差
            check_lowp=check_lowp,  # 是否检查低精度选项
        )
    def test_scatter4(self):
        # 定义一个函数 fn，用于执行 torch.scatter 操作
        def fn(x, ind, src):
            return torch.scatter(x, 0, ind, src)

        # 初始化一个布尔值 check_lowp，用于检查是否需要低精度处理
        check_lowp = True
        # 如果设备为 "xpu"，则将 check_lowp 设为 False
        if self.device == "xpu":
            check_lowp = False

        # 循环测试 deterministic 的取值为 False 和 True
        for deterministic in [False, True]:
            # 使用 DeterministicGuard 来控制代码的确定性行为
            with DeterministicGuard(deterministic):
                # 调用 self.common 方法执行 fn 函数，传入三个参数
                # 分别是一个随机张量、一个随机整数张量、一个随机张量
                # 同时传入 check_lowp 参数
                self.common(
                    fn,
                    [
                        torch.randn(196, 992),
                        torch.randint(196, (1, 992)),
                        torch.randn(1, 992),
                    ],
                    check_lowp=check_lowp,
                )

    def test_scatter5(self):
        # 定义一个函数 fn，用于执行 a 的原地 scatter 操作，并返回处理后的两个张量
        def fn(a, dim, index, b, reduce):
            a = a.clone()  # 克隆张量 a
            a.scatter_(dim, index, b, reduce=reduce)  # 在维度 dim 上进行 scatter 操作
            a1 = a + 1.0  # 创建新的张量 a1，为 a 加上 1.0
            a1.scatter_(dim, index, b, reduce=reduce)  # 在维度 dim 上对 a1 进行 scatter 操作
            return (a, a1)  # 返回两个张量 a 和 a1

        # 初始化一个布尔值 check_lowp，用于检查是否需要低精度处理
        check_lowp = True
        # 如果设备为 "xpu"，则将 check_lowp 设为 False
        if self.device == "xpu":
            check_lowp = False

        # 循环测试 reduce 的取值为 "add" 和 "multiply"
        for reduce in ["add", "multiply"]:
            # 调用 self.common 方法执行 fn 函数，传入五个参数
            # 分别是一个全 1 的 4x5 张量、一个整数 0、一个索引张量、一个随机张量和 reduce 字符串
            # 同时传入 check_lowp 参数
            self.common(
                fn,
                [
                    torch.ones((4, 5)),
                    0,
                    torch.tensor([[1], [2], [3]], dtype=torch.int64),
                    torch.randn(4, 5),
                    reduce,
                ],
                check_lowp=check_lowp,
            )

    def test_scatter6(self):
        # 定义一个函数 fn，用于执行 aten.scatter 操作
        def fn(a, dim, index, b):
            return aten.scatter(a, dim, index, b)

        # 初始化一个布尔值 check_lowp，用于检查是否需要低精度处理
        check_lowp = True
        # 如果设备为 "xpu"，则将 check_lowp 设为 False
        if self.device == "xpu":
            check_lowp = False

        # 循环测试 deterministic 的取值为 False 和 True
        for deterministic in [False, True]:
            # 使用 DeterministicGuard 来控制代码的确定性行为
            with DeterministicGuard(deterministic):
                # 调用 self.common 方法执行 fn 函数，传入四个参数
                # 分别是一个随机张量、一个整数 2、一个三维索引张量、一个标量 0.8
                # 同时传入 check_lowp 参数
                self.common(
                    fn,
                    [
                        torch.randn(5, 8, 13),
                        2,
                        torch.tensor([[[3, 5, 7, 9]]]),
                        0.8,  # src can be a scalar
                    ],
                    check_lowp=check_lowp,
                )

    @unittest.skip("Flaky test, needs debugging")
    def test_scatter_add1(self):
        # 定义一个函数 fn，用于执行 aten.scatter_add 操作
        def fn(a, dim, index, b):
            return aten.scatter_add(a, dim, index, b)

        # 初始化一个布尔值 check_lowp，用于检查是否需要低精度处理
        check_lowp = True
        # 如果设备为 "xpu"，则将 check_lowp 设为 False
        if self.device == "xpu":
            check_lowp = False

        # 调用 self.common 方法执行 fn 函数，传入四个参数
        # 分别是一个随机 2x3 张量、一个整数 0、一个 1x1 的索引张量、一个随机 2x3 张量
        # 不传入 check_lowp 参数，因为它采用默认值 True

    def test_scatter_add2(self):
        # 定义一个函数 fn，用于执行 aten.scatter_add 操作
        def fn(a, dim, index, b):
            return aten.scatter_add(a, dim, index, b)

        # 初始化一个布尔值 check_lowp，用于检查是否需要低精度处理
        check_lowp = True
        # 如果设备为 "xpu"，则将 check_lowp 设为 False
        if self.device == "xpu":
            check_lowp = False

        # 调用 self.common 方法执行 fn 函数，传入四个参数
        # 分别是一个随机 2x3 张量、一个整数 0、一个 2x3 的索引张量、一个随机 2x3 张量
        # 不传入 check_lowp 参数，因为它采用默认值 True
    # 定义测试函数 test_scatter_add3
    def test_scatter_add3(self):
        # 定义内部函数 fn，用于执行 scatter_add 操作
        def fn(a, dim, index, b):
            return aten.scatter_add(a, dim, index, b)

        # 初始化标志 check_lowp，根据设备类型设置其值
        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        # 对于 deterministic 的两种情况进行测试
        for deterministic in [False, True]:
            # 使用 DeterministicGuard 环境，确保测试的确定性
            with DeterministicGuard(deterministic):
                # 调用公共测试方法 common，测试 scatter_add 操作
                self.common(
                    fn,
                    [
                        torch.randn(5, 29, 13),        # 参数 a，随机生成的张量
                        2,                            # 参数 dim，指定的维度
                        torch.tensor([[[3, 5, 7, 9]]]), # 参数 index，指定的索引张量
                        torch.randn(1, 1, 10),         # 参数 b，随机生成的张量
                    ],
                    check_lowp=check_lowp,           # 是否检查低精度标志
                )

    # 定义测试函数 test_scatter_reduce1
    def test_scatter_reduce1(self):
        # 定义内部函数 fn，用于执行 scatter_reduce 操作
        def fn(a, dim, index, b):
            return aten.scatter_reduce(a, dim, index, b, "sum")

        # 初始化标志 check_lowp，根据设备类型设置其值
        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        # 调用公共测试方法 common，测试 scatter_reduce 操作
        self.common(
            fn,
            [
                torch.randn(5, 29, 13),        # 参数 a，随机生成的张量
                2,                            # 参数 dim，指定的维度
                torch.tensor([[[3, 5, 7, 9]]]), # 参数 index，指定的索引张量
                torch.randn(1, 1, 10),         # 参数 b，随机生成的张量
            ],
            check_lowp=check_lowp,           # 是否检查低精度标志
        )

    # 定义测试函数 test_scatter_reduce2
    def test_scatter_reduce2(self):
        # 定义内部函数 fn，用于执行 scatter_reduce 操作，带有 reduce 和 include_self 参数
        def fn(a, dim, index, b, reduce):
            return aten.scatter_reduce(a, dim, index, b, reduce, include_self=False)

        # 初始化标志 check_lowp，根据设备类型设置其值
        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        # 对于两种 reduce 模式进行测试
        for reduce in ["sum", "amax"]:
            # 调用公共测试方法 common，测试 scatter_reduce 操作
            self.common(
                fn,
                [
                    torch.randn(2, 3),        # 参数 a，随机生成的张量
                    0,                        # 参数 dim，指定的维度
                    torch.zeros((2, 3), dtype=torch.int64), # 参数 index，指定的索引张量
                    torch.randn(2, 3),        # 参数 b，随机生成的张量
                    reduce,                   # reduce 模式，sum 或 amax
                ],
                check_lowp=check_lowp,       # 是否检查低精度标志
            )

    # 定义测试函数 test_scatter_reduce3
    def test_scatter_reduce3(self):
        # 定义内部函数 fn，用于执行 scatter_reduce_ 操作，带有 reduce 参数
        def fn(a, dim, index, b, reduce):
            # 复制张量 a
            a = a.clone()
            # 在复制的张量上执行 scatter_reduce_ 操作
            a.scatter_reduce_(dim, index, b, reduce=reduce)
            # 对处理后的张量 a 执行加法操作
            a1 = a + 1.0
            # 在处理后的张量上再次执行 scatter_reduce_ 操作
            a1.scatter_reduce_(dim, index, b, reduce=reduce)
            # 返回处理前后的两个张量
            return (a, a1)

        # 初始化标志 check_lowp，根据设备类型设置其值
        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        # 对于两种 reduce 模式进行测试
        for reduce in ["sum", "prod"]:
            # 调用公共测试方法 common，测试 scatter_reduce_ 操作
            self.common(
                fn,
                [
                    torch.ones((4, 5)),                  # 参数 a，全 1 的张量
                    0,                                  # 参数 dim，指定的维度
                    torch.tensor([[1], [2], [3]], dtype=torch.int64),  # 参数 index，指定的索引张量
                    torch.randn(4, 5),                  # 参数 b，随机生成的张量
                    reduce,                             # reduce 模式，sum 或 prod
                ],
                check_lowp=check_lowp,                 # 是否检查低精度标志
            )

    # 此注释是一个链接，指向 GitHub 上的一个问题
    @config.patch("halide.scheduler_cuda", "Li2018")
    def test_dense_mask_index(self):
        r"""
        There will be a little difference for reduce order between aten and inductor
        https://github.com/pytorch/pytorch/pull/122289
        Absolute difference: 0.00067138671875 (up to 1e-05 allowed)
        Relative difference: 3.1747371732500974e-06 (up to 1.3e-06 allowed)
        """
        # 初始化一个空的参数字典
        kwargs = {}
        # 如果测试设备是 CPU，则设置绝对误差和相对误差的阈值
        if self.device == "cpu":
            kwargs["atol"] = 1e-4
            kwargs["rtol"] = 1.3e-5

        # 定义一个函数 fn，接受两个参数 x 和 y
        def fn(x, y):
            # 调用 torch.ops.aten.select.int 方法，选择 y 的第 0 列的数据
            y = torch.ops.aten.select.int(y, 0, 2)
            # 计算 x 和 y 的乘积
            z = x * y
            # 返回乘积的总和
            return z.sum()

        # 调用 self.common 方法，传入 fn 函数和两个随机张量作为参数，同时传入 kwargs 作为关键字参数
        self.common(fn, [torch.randn(102400), torch.randn(3)], **kwargs)

    def test_empty1(self):
        # 定义一个返回形状为 (1, 128, 128) 的空张量的函数 fn
        def fn():
            return torch.empty((1, 128, 128))

        # 调用 self.common 方法，传入 fn 函数和空参数列表作为参数，同时关闭相等性断言
        self.common(fn, [], assert_equal=False)

    def test_empty2(self):
        # 定义一个调用 aten.empty 方法返回形状为 (1, 128, 128) 的空张量的函数 fn
        def fn():
            return aten.empty((1, 128, 128))

        # 调用 self.common 方法，传入 fn 函数和空参数列表作为参数，同时关闭相等性断言
        self.common(fn, [], assert_equal=False)

    def test_new_empty(self):
        # 定义一个调用 aten.new_empty 方法根据输入张量 a 返回形状为 [1, 128, 128] 的空张量的函数 fn
        def fn(a):
            return aten.new_empty(a, [1, 128, 128])

        # 调用 self.common 方法，传入 fn 函数和包含一个随机张量的参数列表作为参数，同时关闭相等性断言
        self.common(fn, [torch.randn(55)], assert_equal=False)

    def test_empty_strided(self):
        # 定义一个调用 aten.empty_strided 方法返回形状为 [1, 128, 128]、步幅为 [16384, 128, 1] 的张量的函数 fn
        def fn():
            return aten.empty_strided([1, 128, 128], [16384, 128, 1])

        # 调用 self.common 方法，传入 fn 函数和空参数列表作为参数，同时关闭相等性断言
        self.common(fn, [], assert_equal=False)

    def test_new_empty_strided(self):
        # 定义一个调用 aten.new_empty_strided 方法根据输入张量 a 返回形状为 [1, 128, 128]、步幅为 [16384, 128, 1] 的张量的函数 fn
        def fn(a):
            return aten.new_empty_strided(a, [1, 128, 128], [16384, 128, 1])

        # 调用 self.common 方法，传入 fn 函数和包含一个随机张量的参数列表作为参数，同时关闭相等性断言
        self.common(fn, [torch.randn(55)], assert_equal=False)

    def test_dropout_trivial_0(self):
        # 定义一个简单的函数 fn1，使用概率为 0.0 的 dropout，并将结果与原始张量相加
        def fn1(a):
            return torch.nn.functional.dropout(a, 0.0, True) + a

        # 调用 self.common 方法，传入 fn1 函数和一个随机张量作为参数
        self.common(fn1, [torch.randn(55)])

    def test_dropout_trivial_1(self):
        # 定义一个简单的函数 fn2，使用概率为 1.0 的 dropout，并将结果与原始张量相加
        def fn2(a):
            return torch.nn.functional.dropout(a, 1.0, True) + a

        # 调用 self.common 方法，传入 fn2 函数和一个随机张量作为参数
        self.common(fn2, [torch.randn(55)])

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_dropout(self):
        # 设置随机种子确保结果可复现
        random.seed(1234)
        torch.manual_seed(1234)

        # 使用 @torch._dynamo.optimize("inductor") 优化 fn1 函数
        @torch._dynamo.optimize("inductor")
        def fn1(a):
            # 应用 dropout 操作到输入张量 a 上
            return torch.nn.functional.dropout(a)

        # 创建形状为 (1000,) 的张量 x，并应用 fn1 函数
        x = torch.ones(1000, device=self.device, dtype=torch.float32)
        result1 = fn1(x)
        # 断言结果中非零元素的数量在 400 到 600 之间
        self.assertTrue(400 < result1.nonzero().shape[0] < 600)
        # 断言结果的均值在 0.9 到 1.1 之间
        self.assertTrue(0.9 < result1.mean().item() < 1.1)

        # 重新设置随机种子确保结果可复现
        random.seed(1234)
        torch.manual_seed(1234)

        # 使用 @torch._dynamo.optimize("inductor") 优化 fn2 函数
        @torch._dynamo.optimize("inductor")
        def fn2(a):
            # 应用概率为 0.5 的 dropout 操作到输入张量 a 上
            return torch.nn.functional.dropout(a, 0.5, True)

        # 应用 fn2 函数到张量 x 上
        result2 = fn2(x)
        # 断言结果中非零元素的数量在 400 到 600 之间
        self.assertTrue(400 < result2.nonzero().shape[0] < 600)
        # 断言结果的均值在 0.9 到 1.1 之间
        self.assertTrue(0.9 < result2.mean().item() < 1.1)

    @dynamo_config.patch(automatic_dynamic_shapes=True)
    # 定义一个名为 test_dropout_deterministic 的测试方法，用于测试 dropout 函数的确定性行为
    def test_dropout_deterministic(self):
        # 使用 torch._dynamo.optimize 装饰器，优化 fn 函数
        @torch._dynamo.optimize("inductor")
        def fn(a):
            # 调用 torch.nn.functional.dropout 函数对输入 a 进行操作，保留率为 0.55，训练模式为 True
            return torch.nn.functional.dropout(a, 0.55, True)

        # 对于每个布尔值 cg，执行以下测试
        for cg in [False, True]:
            # 使用 patch.object 修改 triton 模块的 cudagraphs 属性为 cg
            with patch.object(config.triton, "cudagraphs", cg):
                # 重置 torch._dynamo 状态
                torch._dynamo.reset()

                # 创建一个全为 1 的张量 x，使用设备 self.device，数据类型为 torch.float32
                x = torch.ones(1024, device=self.device, dtype=torch.float32)

                # 设置随机数种子为 1234
                torch.manual_seed(1234)
                # 调用 fn 函数并克隆结果到 a0, a1, a2
                a0 = fn(x).clone()
                a1 = fn(x).clone()
                a2 = fn(x).clone()

                # 再次设置随机数种子为 1234
                torch.manual_seed(1234)
                # 再次调用 fn 函数并克隆结果到 b0, b1, b2
                b0 = fn(x).clone()
                b1 = fn(x).clone()
                b2 = fn(x).clone()

                # 断言：相同的种子应该得到相同的值
                self.assertTrue(torch.allclose(a0, b0))
                self.assertTrue(torch.allclose(a1, b1))
                self.assertTrue(torch.allclose(a2, b2))

                # 断言：不同的调用应该得到不同的值
                self.assertFalse(torch.allclose(a0, a1))
                self.assertFalse(torch.allclose(a1, a2))

    # 定义一个名为 test_rand_like_deterministic 的测试方法，用于测试 rand_like 函数的确定性行为
    def test_rand_like_deterministic(self):
        # 使用 torch._dynamo.optimize 装饰器，优化 fn 函数
        @torch._dynamo.optimize("inductor")
        def fn(a):
            # 返回一个与输入 a 形状相同的随机张量，同时生成另一个与 a 形状相同的随机张量
            return torch.rand_like(a), torch.rand_like(a)

        # 创建一个全为 1 的张量 x，使用设备 self.device，数据类型为 torch.float32
        x = torch.ones(1024, device=self.device, dtype=torch.float32)

        # 设置随机数种子为 1234
        torch.manual_seed(1234)
        # 调用 fn 函数的第一个返回值，并克隆到 a0, a1, a2
        a0 = fn(x)[0].clone()
        a1 = fn(x)[0].clone()
        a2 = fn(x)[0].clone()

        # 再次设置随机数种子为 1234
        torch.manual_seed(1234)
        # 再次调用 fn 函数的第一个返回值，并克隆到 b0, b1, b2
        b0 = fn(x)[0].clone()
        b1 = fn(x)[0].clone()
        b2 = fn(x)[0].clone()

        # 断言：相同的种子应该得到相同的值
        self.assertTrue(torch.allclose(a0, b0))
        self.assertTrue(torch.allclose(a1, b1))
        self.assertTrue(torch.allclose(a2, b2))

        # 断言：不同的调用应该得到不同的值
        self.assertFalse(torch.allclose(a0, a1))
        self.assertFalse(torch.allclose(a1, a2))

        # 获取 fn 函数的返回值 c, d
        c, d = fn(x)

        # 断言：c 和 d 应该不全相等
        self.assertFalse(torch.allclose(c, d))
        # 断言：c 的所有元素应该大于等于 0
        self.assertTrue((c >= 0).all())
        # 断言：c 的所有元素应该小于 1
        self.assertTrue((c < 1).all())
        # 断言：d 的所有元素应该大于等于 0
        self.assertTrue((d >= 0).all())
        # 断言：d 的所有元素应该小于 1
        self.assertTrue((d < 1).all())

    # 使用 config.patch 装饰器，启用隐式回退
    @config.patch(implicit_fallbacks=True)
    # 定义一个测试函数，用于测试可变操作的基本回退情况
    def test_fallback_mutable_op_basic(self):
        # 使用 torch 库的 _scoped_library 方法创建名为 "mylib" 的库对象，并指定为 "FRAGMENT" 片段
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:

            # 定义一个操作函数 impl，接受参数 a, b, c, d，其中 e 默认为 2
            def impl(a, b, c, d, e=2):
                # 将 b[0] * c * e 加到 a 上（inplace 操作）
                a.add_(b[0] * c * e),
                # 如果 d 不为 None，则将 b[1] 加到 d 上（inplace 操作）
                if d is not None:
                    d.add_(b[1])

            # 在库对象 m 中定义一个名为 "inplace_" 的操作符，其参数类型和返回类型
            m.define(
                "inplace_(Tensor(a!) a, Tensor[] b, SymInt c, *, Tensor(b!)? d, SymInt e=2) -> ()"
            )
            # 将 impl 函数注册到 "inplace_" 操作中，使用 CompositeExplicitAutograd 方法
            m.impl("inplace_", impl, "CompositeExplicitAutograd")

            # 测试 Inductor 不会重新排序 copy_ 和 inplace_ 操作的顺序
            # 定义函数 f，接受参数 a, b1, b2, c, d
            def f(a, b1, b2, c, d):
                # 克隆 a 和 d（如果不为 None），作为操作的备份
                a_ = a.clone()
                d_ = d if d is None else d.clone()
                # 调用 mylib 中的 inplace_ 操作，将结果存入 a_
                torch.ops.mylib.inplace_(a_, (b1, b2), c, d=d_)
                # 将 a_ 的值复制回 a（保持 inplace_ 操作的一致性）
                a.copy_(a_)
                # 如果 d 不为 None，则将 d_ 的值复制回 d
                if d is not None:
                    d.copy_(d_)
                # 返回空元组
                return ()

            # 初始化测试所需的参数
            a = torch.tensor([0.0, 1.0, 2])
            b = [torch.tensor([2.0, 3.0, 5.0]), torch.tensor([1.0, 4.0, 6.0])]
            c = 4
            d = torch.tensor([2.0, 1, 0])
            args = (a, b[0], b[1], c, d)

            # 克隆参数用于后续比较
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            # 使用 make_fx 函数将函数 f 转换为 FX 模块
            mod = make_fx(f)(*cloned_args)
            # 编译 FX 模块
            compiled_f = compile_fx_inner(mod, cloned_args)

            # 再次克隆参数，并在编译后的函数上执行测试
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            compiled_f(list(cloned_args))
            # 直接调用函数 f 进行比较
            f(*args)
            # 断言克隆的参数与原始参数相等
            self.assertEqual(cloned_args, args)

    # 使用 config.patch 设置隐式回退为 True
    @config.patch(implicit_fallbacks=True)
    # 定义一个测试函数，用于测试具有返回值的可变操作的后备方案
    def test_fallback_mutable_op_with_return(self):
        # 使用 torch 库的 _scoped_library 方法创建一个库作用域 "mylib"，模式为 "FRAGMENT"
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:

            # 定义一个内部函数 impl，实现了一个操作，可能会就地修改输入张量
            def impl(a, b, c, d, e=2):
                # 就地更新张量 a，增加 b[0] * c * e 的值
                a.add_(b[0] * c * e),
                # 如果 d 不是 None，则就地更新 d，增加 b[1] 的值
                if d is not None:
                    d.add_(b[1])
                # 返回 b[0] 和 b[1] 的和
                return b[0] + b[1]

            # 定义名为 "inplace_" 的操作的签名
            m.define(
                "inplace_(Tensor(a!) a, Tensor[] b, SymInt c, *, Tensor(b!)? d, SymInt e=2) -> Tensor"
            )
            # 将 impl 函数注册为名为 "inplace_" 的操作的实现，使用 "CompositeExplicitAutograd" 模式
            m.impl("inplace_", impl, "CompositeExplicitAutograd")

            # 定义函数 f，对输入进行克隆和复制操作，以测试 Inductor 不会重排 copy_ 相对于 inplace_ 的顺序
            def f(a, b0, b1, c, d):
                # 克隆张量 a 和 d（如果 d 不为 None）
                a_ = a.clone()
                d_ = d if d is None else d.clone()
                # 调用 inplace_ 操作，修改 a_ 的值
                res = torch.ops.mylib.inplace_(a_, (b0, b1), c, d=d_)
                # 将克隆后的结果复制回原始张量 a
                a.copy_(a_)
                # 如果 d 不为 None，则将克隆后的结果复制回原始张量 d
                if d is not None:
                    d.copy_(d_)
                # 返回结果元组
                return (res,)

            # 初始化测试数据
            a = torch.tensor([0.0, 1.0, 2])
            b = [torch.tensor([2.0, 3.0, 5.0]), torch.tensor([1.0, 4.0, 6.0])]
            c = 4
            d = torch.tensor([2.0, 1, 0])
            args = (a, b[0], b[1], c, d)

            # 对输入张量进行克隆
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            # 使用 make_fx 函数创建函数模块
            mod = make_fx(f)(*cloned_args)
            # 再次对输入张量进行克隆
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            # 编译生成内部函数
            compiled_f = compile_fx_inner(mod, cloned_args)

            # 再次对输入张量进行克隆
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            # 调用编译后的函数，并传入克隆后的输入张量，获取编译后的输出
            compiled_out = compiled_f(list(cloned_args))
            # 调用原始函数 f，并传入原始输入张量，获取原始输出
            out = f(*args)
            # 断言克隆后的输入张量与原始输入张量相等
            self.assertEqual(cloned_args, args)
            # 断言编译后的输出与原始输出相等
            self.assertEqual(compiled_out, out)

    # 使用 config.patch 方法，设置隐式后备为 True
    @config.patch(implicit_fallbacks=True)
    def test_fallback_mutable_op_no_mutated_tensors(self):
        # 使用 torch 库的 _scoped_library 方法创建一个库作用域 "mylib"，模式为 "FRAGMENT"
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:

            # 定义一个内部函数 impl，用于执行操作，当输入张量不会就地修改时
            def impl(a, b):
                # 如果 b 不为 None，则就地更新 b，增加 1 的值
                if b is not None:
                    b.add_(1)

            # 定义名为 "inplace_" 的操作的签名
            m.define("inplace_(Tensor a, Tensor(b!)? b) -> ()")
            # 将 impl 函数注册为名为 "inplace_" 的操作的实现，使用 "CompositeExplicitAutograd" 模式
            m.impl("inplace_", impl, "CompositeExplicitAutograd")

            # 定义函数 f，调用 inplace_ 操作，不会修改输入张量
            def f(a):
                # 调用 inplace_ 操作，传入张量 a 和 None
                torch.ops.mylib.inplace_(a, None)
                # 返回空元组
                return ()

            # 初始化测试数据
            a = torch.tensor([0.0, 1.0, 2])
            args = (a,)
            # 对输入张量进行克隆
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            # 使用 make_fx 函数创建函数模块
            mod = make_fx(f)(*cloned_args)
            # 再次对输入张量进行克隆
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            # 编译生成内部函数
            compiled_f = compile_fx_inner(mod, cloned_args)

            # 再次对输入张量进行克隆
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            # 调用编译后的函数，并传入克隆后的输入张量
            compiled_f(list(cloned_args))
            # 调用原始函数 f，并传入原始输入张量
            f(*args)
            # 断言克隆后的输入张量与原始输入张量相等
            self.assertEqual(cloned_args, args)
    def test_fallback_mutable_op_list(self):
        # 使用 torch 库的 scoped_library 方法，加载名为 "mylib" 的库和 "FRAGMENT" 版本
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:

            # 定义一个操作函数 impl，将标量 a 添加到列表 b 中的每个张量上
            def impl(a, b):
                for bi in b:
                    bi.add_(a)

            # 在 Torch 脚本中注册名为 inplace_ 的自定义函数签名
            m.define("inplace_(Tensor a, Tensor(a!)[] b) -> ()")

            # 将实现函数 impl 注册到 inplace_ 操作中，指定为 "CompositeExplicitAutograd" 形式
            m.impl("inplace_", impl, "CompositeExplicitAutograd")

            # 定义一个函数 f，调用 mylib 库中的 inplace_ 操作，然后返回 None
            def f(a, b):
                torch.ops.mylib.inplace_(a, b)
                return None

            # 创建一个张量 a，包含 [0.0, 1.0, 2.0]
            a = torch.tensor([0.0, 1.0, 2])
            # 创建一个张量列表 b，包含两个张量 [tensor([2.0, 3.0, 5.0]), tensor([1.0, 4.0, 6.0])]
            b = [torch.tensor([2.0, 3.0, 5.0]), torch.tensor([1.0, 4.0, 6.0])]
            # 将参数 args 设置为 (a, b)
            args = (a, b)
            # 使用 pytree 将参数 args 中的所有张量克隆一份，保存到 cloned_args 中
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            # 使用 make_fx 创建一个 Function 版本的函数模块 mod，传入克隆后的参数 cloned_args
            mod = make_fx(f)(*cloned_args)
            # 再次克隆参数 args 中的所有张量，保存到 cloned_args 中
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)

            # 编译函数模块 mod，传入克隆后的参数 cloned_args，保存到 compiled_f 中
            compiled_f = compile_fx_inner(mod, cloned_args)

        # 自定义 Torch 自定义操作，注册名为 "mylib::sin_out"，指定它会修改 outs 参数
        @torch.library.custom_op("mylib::sin_out", mutates_args={"outs"})
        # 定义一个 sin_out 函数，接受 torch.Tensor 类型的输入 x 和输出列表 outs
        def sin_out(x: torch.Tensor, outs: typing.List[torch.Tensor]) -> None:
            # 将 x 转换为 NumPy 数组 x_np
            x_np = x.numpy()
            # 断言 outs 列表长度为 2
            assert len(outs) == 2
            # 将 outs[0] 转换为 NumPy 数组 out_np0
            out_np0 = outs[0].numpy()
            # 将 outs[1] 转换为 NumPy 数组 out_np1
            out_np1 = outs[1].numpy()
            # 使用 NumPy 的 sin 函数，计算 x_np 的正弦值，并将结果存入 out_np0 中
            np.sin(x_np, out=out_np0)
            # 使用 NumPy 的 sin 函数，计算 x_np 的正弦值，并将结果存入 out_np1 中
            np.sin(x_np, out=out_np1)

        # 将函数 g 标记为 Torch 的编译函数
        @torch.compile
        # 定义函数 g，接受输入 x
        def g(x):
            # 创建一个与 x 同样形状的空张量列表 outs，包含两个张量
            outs = [torch.empty_like(x) for _ in range(2)]
            # 调用 sin_out 函数，将 x 的正弦值写入 outs 中
            sin_out(x, outs)
            # 返回计算后的 outs 列表
            return outs

        # 创建一个形状为 (3,) 的随机张量 x
        x = torch.randn(3)
        # 创建一个与 x 同样形状的空张量列表 out，包含两个张量
        out = [torch.empty_like(x) for _ in range(2)]
        # 调用函数 g，传入 x，并将结果保存到变量 y 中
        y = g(x)
    def test_functionalize_rng_wrappers(self):
        # 测试函数：test_functionalize_rng_wrappers
        # 理想情况下，我们希望为这些操作使用 torch.compile。但当前计划是在分区器级别引入这些操作，
        # 从而无需通过 torch.compile 完全支持它们。为了确保在使用最小化工具时有足够的调试功能，
        # 我们必须确保它们在 make_fx 中工作正常。本测试使用 make_fx 进行测试。未来，我们可以转向 torch.compile。

        def fn():
            # 定义函数 fn
            # 调用 torch._prims.rng_prims.run_and_save_rng_state 运行并保存随机数生成器状态
            rng_state1, a1 = torch._prims.rng_prims.run_and_save_rng_state(
                torch.ops.aten.rand.default,
                [4, 4],
                dtype=torch.float32,
                device=self.device,
            )
            rng_state2, a2 = torch._prims.rng_prims.run_and_save_rng_state(
                torch.ops.aten.rand.default,
                [4, 4],
                dtype=torch.float32,
                device=self.device,
            )

            # 使用保存的 rng_state1 运行随机数生成器
            b1 = torch._prims.rng_prims.run_with_rng_state(
                rng_state1,
                torch.ops.aten.rand.default,
                [4, 4],
                dtype=torch.float32,
                device=self.device,
            )
            # 使用保存的 rng_state2 运行随机数生成器
            b2 = torch._prims.rng_prims.run_with_rng_state(
                rng_state2,
                torch.ops.aten.rand.default,
                [4, 4],
                dtype=torch.float32,
                device=self.device,
            )

            # 返回结果元组 (a1, a2, b1, b2)
            return (a1, a2, b1, b2)

        # 使用 make_fx 对函数 fn 进行模块化处理，并调用生成的模块化函数
        mod = make_fx(fn)()
        # 使用 compile_fx_inner 编译生成的模块化函数
        compiled_f = compile_fx_inner(mod, ())
        # 调用编译后的函数，获取结果 a1, a2, b1, b2
        a1, a2, b1, b2 = compiled_f(())
        # 断言检查 a1 是否等于 b1
        self.assertEqual(a1, b1)
        # 断言检查 a2 是否等于 b2
        self.assertEqual(a2, b2)

    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    @expectedFailureXPU
    @skip_if_gpu_halide  # rand
    # 定义单元测试函数 test_philox_rand，用于测试随机数生成操作
    def test_philox_rand(self):
        # 如果设备是 CPU，则跳过此测试
        if self.device == "cpu":
            raise unittest.SkipTest(
                f"functionalization of rng ops supported only on {GPU_TYPE}"
            )

        # 使用 Torch 内部优化功能装饰函数 fn
        @torch._dynamo.optimize("inductor")
        def fn(x):
            # 生成一个与 x 相同形状的随机张量并与 x 相乘
            a = torch.rand_like(x) * x
            # 再次生成一个与 x 相同形状的随机张量并与前一步结果相乘
            a = torch.rand_like(x) * a
            return a

        # 定义检查函数 check，验证随机种子对 fn 的影响
        def check(x):
            # 设置随机种子为 123
            torch.manual_seed(123)
            a = fn(x)

            # 设置不同的随机种子为 1234
            torch.manual_seed(1234)
            b = fn(x)

            # 再次设置随机种子为 123，期望得到与第一次相同的结果
            torch.manual_seed(123)
            c = fn(x)

            # 断言：相同的随机种子应该得到相同的结果
            self.assertTrue(torch.allclose(a, c))

            # 断言：不同的随机种子应该得到不同的结果
            self.assertFalse(torch.allclose(a, b))

        # 检查形状为 (1024,) 的全一张量，传入当前设备和数据类型为 float32
        check(torch.ones(1024, device=self.device, dtype=torch.float32))
        # 需要注释：是否应该将 "_get_rng_state_offset" 添加到通用设备接口中？
        self.assertEqual(getattr(torch, self.device)._get_rng_state_offset(), 2048)
        
        # 检查形状为 (3,) 的全一张量，传入当前设备和数据类型为 float32
        check(torch.ones(3, device=self.device, dtype=torch.float32))
        # 断言：检查 RNG 状态偏移是否为 8
        self.assertEqual(getattr(torch, self.device)._get_rng_state_offset(), 8)

    # 默认情况下已启用，仅用于确认
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # 定义测试函数 test_reuse_buffers_with_aliasing，测试在缓冲区复用时的别名问题
    def test_reuse_buffers_with_aliasing(self):
        # 定义函数 f，接收输入张量 x 并进行复杂的视图变换操作
        def f(x):
            z = x + 1
            z = torch.view_as_complex(z)
            a = torch.view_as_real(z)
            out = a + 1
            return out, torch.view_as_real(z + 1)

        # 使用通用测试函数 common 对函数 f 进行测试，传入全零张量 (4, 2)
        self.common(f, (torch.zeros((4, 2)),))

        # 运行并获取 Triton 代码，用于检查是否支持复杂视图操作
        code = run_and_get_triton_code(torch.compile(f), torch.zeros((4, 2)))
        # 断言：确保 Triton 代码中存在视图操作 "aten.view_as_real"
        self.assertTrue("aten.view_as_real" in code)

        # 定义函数 f2，进行更复杂的视图变换操作
        def f2(x):
            z = x + 1
            z = torch.view_as_complex(z)
            z = torch.view_as_real(z)
            z = torch.view_as_complex(z)
            a = torch.view_as_real(z)
            out = a + 1
            return out, torch.view_as_real(z + 1)

        # 使用通用测试函数 common 对函数 f2 进行测试，传入全零张量 (4, 2)
        self.common(f, (torch.zeros((4, 2)),))

    # 定义测试函数 test_randn_like_empty，测试在空张量上的随机数生成
    def test_randn_like_empty(self):
        # 定义模型类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            # 定义模型的前向传播函数 forward，接收输入张量 v1
            def forward(self, v1: torch.Tensor):
                # 计算 v1 每行的最小值，并作为 vx
                vx = v1.min(dim=1).values
                # 根据 vx 生成一个与其形状相同的随机张量 v2
                v2 = torch.randn_like(vx)
                return v2

        # 创建 Model 类的实例 model
        model = Model()
        # 创建形状为 (10, 3, 0) 的随机张量 x
        x = torch.rand(10, 3, 0)

        # 使用通用测试函数 common 对模型进行测试，传入 x
        self.common(model, (x,))
    # 定义测试函数 test_randint，用于测试 torch.randint 和相关功能
    def test_randint(self):
        # 使用 torch.compile 装饰器，设置 fullgraph=True，编译函数 fn 的完整计算图
        @torch.compile(fullgraph=True)
        def fn(x):
            # 返回三个张量：
            # 1. 在 [0, 10) 范围内的随机整数张量，形状为 [1024]，设备与 x 相同
            # 2. 在 [-4, 7) 范围内的随机整数张量，形状为 [1024]，数据类型为 torch.int32，设备与 x 相同
            # 3. 形状与 x 相同，在 [0, 2**50) 范围内的随机整数张量
            return (
                torch.randint(10, [1024], device=x.device),
                torch.randint(-4, 7, [1024], dtype=torch.int32, device=x.device),
                torch.randint_like(x, 2**50),
            )

        # 设置随机种子为 12345
        torch.manual_seed(12345)
        # 调用 fn 函数，生成随机张量 a0, b0, c0
        a0, b0, c0 = fn(torch.zeros([40, 40], device=self.device))
        # 断言张量形状正确
        self.assertEqual(a0.shape, [1024])
        self.assertEqual(b0.shape, [1024])
        self.assertEqual(c0.shape, [40, 40])
        
        # 重新设置随机种子为 12345
        torch.manual_seed(12345)
        # 再次调用 fn 函数，生成相同随机种子下的随机张量 a1, b1, c1
        a1, b1, c1 = fn(torch.zeros([40, 40], device=self.device))
        # 断言生成的随机张量与之前生成的相同
        self.assertEqual(a0, a1)
        self.assertEqual(b0, b1)
        self.assertEqual(c0, c1)

        # 断言张量 a0 的最小值为 0，最大值为 9
        self.assertEqual(a0.min(), 0)
        self.assertEqual(a0.max(), 9)

        # 断言张量 b0 的最小值为 -4，最大值为 6
        self.assertEqual(b0.min(), -4)
        self.assertEqual(b0.max(), 6)

        # 断言张量 c0 的最小值至少为 0，最大值在 2**40 到 2**50 之间
        self.assertGreaterEqual(c0.min(), 0)
        self.assertGreater(c0.max(), 2**40)
        self.assertLess(c0.max(), 2**50)

    # 使用 config.patch 装饰器，设置 fallback_random=True，修补测试函数 test_like_rands
    @config.patch(fallback_random=True)
    def test_like_rands(self):
        # 定义函数 fn，返回与 x 形状相同的随机张量和随机正态分布张量
        def fn(x):
            return torch.rand_like(x), torch.randn_like(x)

        # 使用 common 函数测试 fn 函数，传入参数为包含零张量的列表
        self.common(fn, [torch.zeros([20, 20])])

    # 定义测试函数 test_like_rands2
    def test_like_rands2(self):
        # 输出注释：rand_like 函数的 device 参数为 str 类型
        # 检查 self.device 是否为字符串类型
        d = self.device
        assert isinstance(d, str)

        # 使用 torch.compile 装饰器编译函数 fn
        @torch.compile
        def fn(x):
            # 返回 x 形状相同的随机张量，设备由参数 d 指定
            return torch.rand_like(x, device=d)

        # 创建形状为 [10] 的浮点数张量 x，设备为 self.device，数据类型为 torch.float32
        x = torch.ones(10, device=self.device, dtype=torch.float32)
        # 调用 fn 函数生成随机张量 a0
        a0 = fn(x).clone()
        # 再次调用 fn 函数生成随机张量 a1
        a1 = fn(x).clone()
        # 断言 a0 和 a1 不完全相等
        self.assertFalse(torch.allclose(a0, a1))

    # 使用 requires_gpu 装饰器修饰测试函数 test_like_rands3
    @requires_gpu()
    def test_like_rands3(self):
        # 输出注释：rand_like 函数的 device 参数与 x.device 不同
        def test_like_rands_on_different_device(device1, device2):
            # 使用 torch.compile 装饰器编译函数 fn
            @torch.compile
            def fn(x, device):
                # 返回 x 形状相同的随机张量，设备由参数 device 指定
                return torch.rand_like(x, device=device)

            # 创建形状为 [10] 的浮点数张量 x，设备为 device1，数据类型为 torch.float32
            x = torch.ones(10, device=device1, dtype=torch.float32)
            # 调用 fn 函数生成随机张量，设备为 device2
            return fn(x, device2).clone()

        # 使用 CPU 和 GPU_TYPE（假定的 GPU 设备类型）测试函数 test_like_rands_on_different_device
        a0 = test_like_rands_on_different_device("cpu", GPU_TYPE)
        a1 = test_like_rands_on_different_device(GPU_TYPE, "cpu")
        # 断言 a0 在 GPU 上，a1 在 CPU 上
        self.assertTrue(a0.device.type == GPU_TYPE)
        self.assertTrue(a1.device.type == "cpu")

    # 定义测试函数 test_max_pool2d_with_indices_backward
    def test_max_pool2d_with_indices_backward(self):
        # 输出注释：对应函数 fn 使用 aten.max_pool2d_with_indices_backward 执行最大池化反向传播
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [2, 2], [2, 2], [0, 0], [1, 1], False, c
            )

        # 创建形状为 [2, 4, 18, 14] 的随机张量 x
        x = torch.randn([2, 4, 18, 14])
        # 调用 aten.max_pool2d_with_indices 函数，对 x 执行最大池化操作
        result, indices = aten.max_pool2d_with_indices(
            x,
            [2, 2],
            [2, 2],
            [0, 0],
            [1, 1],
            False,
        )

        # 使用 common 函数测试 fn 函数，传入参数为 result, x, indices
        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )

    # 使用 skip_if_gpu_halide 装饰器跳过测试
    @skip_if_gpu_halide  # slow
    # 定义测试函数 test_max_pool2d_with_indices_backward2
    def test_max_pool2d_with_indices_backward2(self):
        # 定义内部函数 fn，用于调用 torch 函数 aten.max_pool2d_with_indices_backward
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [3, 3], [2, 2], [1, 1], [1, 1], True, c
            )

        # 生成一个随机张量 x，形状为 [2, 4, 40, 56]
        x = torch.randn([2, 4, 40, 56])
        # 调用 torch 函数 aten.max_pool2d_with_indices，返回结果和索引
        result, indices = aten.max_pool2d_with_indices(
            x,
            [3, 3],
            [2, 2],
            [1, 1],
            [1, 1],
            True,
        )
        # 调用 self.common 方法，传入 fn 函数和参数列表
        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )

    # From https://github.com/pytorch/torchdynamo/issues/1200
    # 定义测试函数 test_max_pool2d_with_indices_backward3
    def test_max_pool2d_with_indices_backward3(self):
        # 定义内部函数 fn，用于调用 torch 函数 aten.max_pool2d_with_indices_backward
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [1, 1], [2, 2], [0, 0], [1, 1], False, c
            )

        # 生成一个随机张量 x，形状为 [32, 256, 37, 38]
        x = torch.randn([32, 256, 37, 38])
        # 调用 torch 函数 aten.max_pool2d_with_indices，返回结果和索引
        result, indices = aten.max_pool2d_with_indices(
            x,
            [1, 1],
            [2, 2],
            0,
            1,
            False,
        )
        # 调用 self.common 方法，传入 fn 函数和参数列表
        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )

    # From https://github.com/pytorch/torchdynamo/issues/1352
    # 根据 issue 提到的链接定义测试函数 test_max_pool2d_with_indices_backward4
    @skip_if_halide  # hangs forever
    def test_max_pool2d_with_indices_backward4(self):
        # 定义内部函数 fn，用于调用 torch 函数 aten.max_pool2d_with_indices_backward
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [5, 5], [1, 1], [2, 2], [1, 1], False, c
            )

        # 设置 torch._inductor.metrics.generated_kernel_count 为 0
        torch._inductor.metrics.generated_kernel_count = 0
        # 生成一个随机张量 x，形状为 [2, 64, 3, 4]
        x = torch.randn([2, 64, 3, 4])
        # 调用 torch 函数 aten.max_pool2d_with_indices，返回结果和索引
        result, indices = aten.max_pool2d_with_indices(
            x,
            [5, 5],
            [1, 1],
            2,
            1,
            False,
        )
        # 调用 self.common 方法，传入 fn 函数和参数列表
        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )
        # 断言生成的 kernel 数量等于 1
        assertGeneratedKernelCountEqual(self, 1)

    # 定义测试函数 test_max_pool2d_with_indices_backward5
    @expectedFailureXPU
    def test_max_pool2d_with_indices_backward5(self):
        # 窗口尺寸太大，预期会回退处理
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [13, 13], [1, 1], [2, 2], [1, 1], False, c
            )

        # 设置 torch._inductor.metrics.generated_kernel_count 为 0
        torch._inductor.metrics.generated_kernel_count = 0
        # 生成一个随机张量 x，形状为 [2, 64, 20, 20]
        x = torch.randn([2, 64, 20, 20])
        # 调用 torch 函数 aten.max_pool2d_with_indices，返回结果和索引
        result, indices = aten.max_pool2d_with_indices(
            x,
            [13, 13],
            [1, 1],
            2,
            1,
            False,
        )
        # 调用 self.common 方法，传入 fn 函数和参数列表
        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )
        # 断言生成的 kernel 数量等于 0
        assertGeneratedKernelCountEqual(self, 0)

    # From https://github.com/pytorch/pytorch/issues/93384
    def test_max_pool2d_with_indices_backward6(self):
        # dilation is not 1. Should fallback
        # 定义一个函数fn，接收三个参数a, b, c，并调用aten.max_pool2d_with_indices_backward进行处理
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [3, 2], [2, 1], [1, 1], [1, 2], False, c
            )

        # 重置计数器，用于记录生成的内核数量
        torch._inductor.metrics.generated_kernel_count = 0
        # 生成一个大小为[2, 2, 3, 6]的随机张量x
        x = torch.randn([2, 2, 3, 6])
        # 调用aten.max_pool2d_with_indices计算x的最大池化结果和对应的索引
        result, indices = aten.max_pool2d_with_indices(
            x,
            [3, 2],
            [2, 1],
            [1, 1],
            [1, 2],
            False,
        )
        # 使用self.common方法，传入fn函数和参数列表，进行通用测试
        self.common(
            fn,
            [
                torch.randn_like(result),  # 使用result的形状生成一个随机张量
                x,  # 将x作为参数传递
                indices,  # 将indices作为参数传递
            ],
        )
        # 断言生成的内核数量与期望值相等
        assertGeneratedKernelCountEqual(self, 0)

    def test_issue102546(self):
        # 定义一个函数fn，计算输入张量x在维度0上的均值
        def fn(x):
            return x.mean(0)

        # 使用self.common方法，传入fn函数和包含一个形状随机张量的参数列表进行通用测试
        self.common(fn, [torch.rand(())])

    def test_avg_pool2d_backward(self):
        # 定义一个函数fn，调用aten.avg_pool2d_backward进行平均池化的反向传播计算
        def fn(a, b):
            return aten.avg_pool2d_backward(
                a,
                b,
                [2, 2],  # 池化窗口大小为2x2
                [2, 2],  # 步幅为2x2
                [0, 0],  # 填充为0
                True,    # 是否使用平均池化的结果
                False,   # 是否进行求导
                None,    # 非空的output_gradient，可以是None
            )

        # 使用self.common方法，传入fn函数和两个形状随机张量的参数列表进行通用测试
        self.common(
            fn,
            [
                torch.randn([2, 4, 7, 7]),    # 大小为[2, 4, 7, 7]的随机张量
                torch.randn([2, 4, 14, 14]),  # 大小为[2, 4, 14, 14]的随机张量
            ],
        )

    @skip_if_gpu_halide  # slow
    def test_avg_pool2d_backward2(self):
        # 定义一个函数fn，调用aten.avg_pool2d_backward进行平均池化的反向传播计算
        def fn(a, b):
            return aten.avg_pool2d_backward(
                a,
                b,
                [3, 3],  # 池化窗口大小为3x3
                [1, 1],  # 步幅为1x1
                [1, 1],  # 填充为1
                True,    # 是否使用平均池化的结果
                False,   # 是否进行求导
                None,    # 非空的output_gradient，可以是None
            )

        # 使用self.common方法，传入fn函数和两个形状随机张量的参数列表进行通用测试
        self.common(
            fn,
            [
                torch.randn([1, 1, 20, 15]),  # 大小为[1, 1, 20, 15]的随机张量
                torch.randn([1, 1, 20, 15]),  # 大小为[1, 1, 20, 15]的随机张量
            ],
        )

    def test_avg_pool2d_backward3(self):
        # 定义一个函数fn，调用aten.avg_pool2d_backward进行平均池化的反向传播计算
        def fn(a, b):
            return aten.avg_pool2d_backward(
                a,
                b,
                [1, 1],  # 池化窗口大小为1x1
                [2, 2],  # 步幅为2x2
                [0, 0],  # 填充为0
                False,   # 是否使用平均池化的结果
                False,   # 是否进行求导
                None,    # 非空的output_gradient，可以是None
            )

        # 重置计数器，用于记录生成的内核数量
        torch._inductor.metrics.generated_kernel_count = 0
        # 使用self.common方法，传入fn函数和两个形状随机张量的参数列表进行通用测试
        self.common(
            fn,
            [
                torch.randn([1, 2016, 11, 11]),  # 大小为[1, 2016, 11, 11]的随机张量
                torch.randn([1, 2016, 21, 21]),  # 大小为[1, 2016, 21, 21]的随机张量
            ],
        )
        # 断言生成的内核数量与期望值相等
        assertGeneratedKernelCountEqual(self, 1)
    def test_avg_pool2d_backward4(self):
        # 定义一个函数 fn，用于计算二维平均池化的反向传播
        def fn(a, b):
            return aten.avg_pool2d_backward(
                a,
                b,
                [13, 13],  # 池化窗口的大小为 13x13
                [1, 1],    # 池化的步长为 1x1
                [0, 0],    # 池化填充的大小为 0x0
                True,      # 使用平均池化
                False,     # 不使用低精度（lowp）模式
                None,      # 不使用额外的配置参数
            )

        # 重置生成的内核计数
        torch._inductor.metrics.generated_kernel_count = 0
        # 调用共用函数 common 运行 fn，传入两个随机生成的张量作为参数
        self.common(
            fn,
            [
                torch.randn([1, 16, 12, 12]),  # 第一个张量的形状为 [1, 16, 12, 12]
                torch.randn([1, 16, 24, 24]),  # 第二个张量的形状为 [1, 16, 24, 24]
            ],
            check_lowp=False,  # 禁用低精度检查
        )
        # 断言生成的内核数量等于 0
        assertGeneratedKernelCountEqual(self, 0)

    def test_avg_pool3d_backward(self):
        # 定义一个函数 fn，用于计算三维平均池化的反向传播
        def fn(a, b):
            return aten.avg_pool3d_backward(
                a,
                b,
                [2, 2, 2],  # 池化窗口的大小为 2x2x2
                [2, 2, 2],  # 池化的步长为 2x2x2
                [0, 0, 0],  # 池化填充的大小为 0x0x0
                True,       # 使用平均池化
                False,      # 不使用低精度（lowp）模式
                None,       # 不使用额外的配置参数
            )

        # 调用共用函数 common 运行 fn，传入两个随机生成的三维张量作为参数
        self.common(
            fn,
            [
                torch.randn([2, 4, 7, 7, 7]),    # 第一个张量的形状为 [2, 4, 7, 7, 7]
                torch.randn([2, 4, 14, 14, 14]),  # 第二个张量的形状为 [2, 4, 14, 14, 14]
            ],
        )

    @skip_if_halide  # 如果条件满足，跳过测试（Halide 编译需要超过 5 分钟）
    def test_avg_pool3d_backward2(self):
        # 定义一个函数 fn，用于计算三维平均池化的反向传播
        def fn(a, b):
            return aten.avg_pool3d_backward(
                a,
                b,
                [3, 3, 3],  # 池化窗口的大小为 3x3x3
                [1, 1, 1],  # 池化的步长为 1x1x1
                [1, 1, 1],  # 池化填充的大小为 1x1x1
                True,       # 使用平均池化
                False,      # 不使用低精度（lowp）模式
                None,       # 不使用额外的配置参数
            )

        # 调用共用函数 common 运行 fn，传入两个相同的随机生成的三维张量作为参数
        self.common(
            fn,
            [
                torch.randn([1, 1, 20, 20, 15]),  # 第一个张量的形状为 [1, 1, 20, 20, 15]
                torch.randn([1, 1, 20, 20, 15]),  # 第二个张量的形状为 [1, 1, 20, 20, 15]
            ],
        )

    def test_avg_pool3d_backward3(self):
        # 定义一个函数 fn，用于计算三维平均池化的反向传播
        def fn(a, b):
            return aten.avg_pool3d_backward(
                a,
                b,
                [1, 1, 1],  # 池化窗口的大小为 1x1x1
                [2, 2, 2],  # 池化的步长为 2x2x2
                [0, 0, 0],  # 池化填充的大小为 0x0x0
                False,      # 不使用平均池化
                False,      # 不使用低精度（lowp）模式
                None,       # 不使用额外的配置参数
            )

        # 重置生成的内核计数
        torch._inductor.metrics.generated_kernel_count = 0
        # 调用共用函数 common 运行 fn，传入两个随机生成的三维张量作为参数
        self.common(
            fn,
            [
                torch.randn([1, 2016, 11, 11, 11]),  # 第一个张量的形状为 [1, 2016, 11, 11, 11]
                torch.randn([1, 2016, 21, 21, 21]),  # 第二个张量的形状为 [1, 2016, 21, 21, 21]
            ],
        )
        # 断言生成的内核数量等于 1
        assertGeneratedKernelCountEqual(self, 1)

    def test_avg_pool3d_backward4(self):
        # 定义一个函数 fn，用于计算三维平均池化的反向传播
        def fn(a, b):
            return aten.avg_pool3d_backward(
                a,
                b,
                [13, 13, 13],  # 池化窗口的大小为 13x13x13
                [1, 1, 1],     # 池化的步长为 1x1x1
                [0, 0, 0],     # 池化填充的大小为 0x0x0
                True,          # 使用平均池化
                False,         # 不使用低精度（lowp）模式
                None,          # 不使用额外的配置参数
            )

        # 重置生成的内核计数
        torch._inductor.metrics.generated_kernel_count = 0
        # 调用共用函数 common 运行 fn，传入两个随机生成的三维张量作为参数
        self.common(
            fn,
            [
                torch.randn([1, 16, 12, 12, 12]),   # 第一个张量的形状为 [1, 16, 12, 12, 12]
                torch.randn([1, 16, 24, 24, 24]),   # 第二个张量的形状为 [1, 16, 24, 24, 24]
            ],
            check_lowp=False,  # 禁用低精度检查
        )
        # 断言生成的内核数量等于 0
        assertGeneratedKernelCountEqual(self, 0)

    @config.patch(search_autotune_cache=False)
    def test_mm_views(self):
        # 定义一个函数 fn，用于计算两个矩阵的矩阵乘法，先将输入张量视图变形为 32x32 的形状后再进行计算
        def fn(a, b):
            return torch.mm(a.view(32, 32), b.view(32, 32))

        # 调用通用测试函数 common，测试 fn 函数的行为
        self.common(
            fn,
            (
                torch.randn([32, 32]).transpose(0, 1),  # 随机生成 32x32 的张量，并转置
                torch.randn([1, 32, 32]).transpose(0, 1),  # 随机生成 1x32x32 的张量，并转置
            ),
            check_lowp=False,  # 不检查低精度
        )
        expected_kernel = 0
        # 从模板生成 mm 内核代码
        self.assertEqual(
            torch._inductor.metrics.generated_kernel_count, expected_kernel
        )

    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_dtype_sympy_expr(self):
        # 定义一个函数 fn，对输入张量 a 进行操作，将其从倒数第二维开始到第二维的数据取出并保证内存连续性
        @torch._dynamo.optimize_assert("inductor")
        def fn(a):
            y = a[..., :-1, :].contiguous()
            return y

        # 对 fn 函数进行测试，输入为形状为 [1, 2, 16, 4] 的随机张量并要求计算梯度
        result = fn(torch.randn([1, 2, 16, 4]).requires_grad_())
        result.sum().backward()

    @skip_if_halide  # 如果条件 skip_if_halide 成立，则跳过以下函数的测试
    def test_dropout2(self):
        n = 100000
        # 在设备上创建形状为 n 的全一张量，设置为需要梯度计算的浮点型张量
        weight = torch.ones(
            n, device=self.device, dtype=torch.float32, requires_grad=True
        )
        # 在设备上创建形状为 n 的全一张量，浮点型
        ones = torch.ones(n, device=self.device, dtype=torch.float32)

        # 定义一个运行函数 run，对输入张量 x 执行带权重的 dropout 操作，参数 train 决定是否训练模式
        @torch._dynamo.optimize_assert("inductor")
        def run(x, train=True):
            return F.dropout(x * weight, 0.33, train)

        # 定义一个检查函数 check，对两个输入张量 r 和 g 进行一系列检查
        def check(r, g):
            rmean = r.mean().item()
            gmean = g.mean().item()
            rcount = len(r.nonzero())
            gcount = len(g.nonzero())

            # 断言丢弃的元素应该匹配
            self.assertTrue(same(r.nonzero(), g.nonzero()))
            self.assertEqual(rcount, gcount)

            # 断言丢弃的元素应接近 0.33
            self.assertGreater(rcount, 0.64 * n)
            self.assertGreater(0.68 * n, rcount)

            self.assertAlmostEqual(rmean, gmean)
            self.assertAlmostEqual(rmean, 1.0, places=2)

        # 第一次运行 run 函数，输入 ones 张量，关闭训练模式
        r1 = run(ones, train=False)
        r1.sum().backward()
        g1 = weight.grad.clone()
        # 在评估模式下，结果应全部为一
        self.assertTrue(same(r1, torch.ones_like(r1)))
        self.assertTrue(same(g1, torch.ones_like(g1)))

        # 设置随机种子为 1234
        torch.manual_seed(1234)
        weight.grad.zero_()
        # 运行前向和反向过程，并获取代码
        r2, (fw_code, bw_code) = run_fw_bw_and_get_code(lambda: run(ones))
        if self.device == GPU_TYPE:
            self.assertEqual(fw_code.count("tl.rand"), 1)
            self.assertEqual(bw_code.count("tl.rand"), 0)
        g2 = weight.grad.clone()
        check(r2, g2)

        # 重置随机种子为 1234
        torch.manual_seed(1234)
        weight.grad.zero_()
        # 第二次运行 run 函数，输入 ones 张量
        r3 = run(ones)
        r3.sum().backward()
        g3 = weight.grad.clone()
        check(r3, g3)

        # 第二次运行的结果应与第一次相同
        self.assertTrue(same(r2, r3))
        self.assertTrue(same(g2, g3))

    @config.patch(search_autotune_cache=False)
    @skip_if_halide  # 如果条件 skip_if_halide 成立，则跳过以下代码块的测试
    def test_dropout3(self):
        # 创建一个包含线性层和随机丢弃的神经网络模型
        m = torch.nn.Sequential(
            torch.nn.Linear(32, 32, bias=False),
            torch.nn.Dropout(),
            torch.nn.Linear(32, 32, bias=False),
            torch.nn.Dropout(),
        ).to(self.device)

        # 定义一个函数运行神经网络模型
        @torch._dynamo.optimize_assert("inductor")
        def run(x):
            return m(x)

        # 初始化指标
        torch._inductor.metrics.generated_kernel_count = 0

        # 运行前向和后向传播，并获取生成的代码
        result, (fw_code, bw_code) = run_fw_bw_and_get_code(
            lambda: run(torch.randn([8, 32], device=self.device))
        )

        # 验证生成的前向代码中随机数的使用次数
        if self.device == GPU_TYPE:
            self.assertEqual(fw_code.count("tl.rand"), 2)
            self.assertEqual(bw_code.count("tl.rand"), 0)
        expected_kernel = 4

        # 验证生成的内核数量是否符合预期
        self.assertEqual(
            torch._inductor.metrics.generated_kernel_count, expected_kernel
        )

    @skip_if_halide  # rand
    def test_randint_kernel_count(self):
        # 定义一个函数，生成随机整数张量，并获取生成的源代码
        @torch._dynamo.optimize_assert("inductor")
        def fn1():
            random_tensor1 = torch.randint(10, [32], device=self.device)
            random_tensor2 = torch.randint(10, [32], device=self.device)
            random_tensor3 = torch.randint(10, [32], device=self.device)
            return random_tensor1, random_tensor2, random_tensor3

        _, source_codes = run_and_get_code(fn1)
        
        # 如果在 GPU 上，验证生成的源代码中异步编译的使用次数
        if self.device == GPU_TYPE:
            self.assertEqual(len(source_codes), 1)
            self.assertEqual(source_codes[0].count("async_compile.triton"), 2)

    def test_roll(self):
        # 定义一个函数，使用 aten.roll 进行张量滚动操作
        def fn(a):
            return (
                aten.roll(a, [-3, 10], [1, 2]),
                aten.roll(a, [5]),
            )

        # 使用通用函数进行测试
        self.common(
            fn,
            [
                torch.randn([2, 56, 56, 16]),
            ],
        )

    def test_argmax_min_int32(self):
        # https://github.com/pytorch/pytorch/issues/94055
        # 定义一个函数，计算张量 a 沿着第 3 维的最大值索引，和张量 b 的最小值
        def fn(a, b):
            c = a.argmax(3)
            return torch.min(b, c)

        # 初始化张量 a 和 b 进行通用测试
        a = torch.rand(3, 4, 2, 1).int()
        b = torch.rand(2, 2, 1, 4, 1).int()
        self.common(fn, (a, b))

    def test_argmax_argmin1(self):
        # 定义一个函数，计算输入张量 x 的最大值和最小值索引
        def fn(x):
            return (aten.argmax(x), aten.argmin(x))

        # 使用通用函数进行测试
        self.common(
            fn,
            [
                torch.randn([8, 256, 256]),
            ],
        )

    def test_argmax_argmin2(self):
        # 定义一个函数，分别计算输入张量 x 沿着不同维度的最大值和最小值索引
        def fn(x):
            return (
                aten.argmax(x, 0),
                aten.argmin(x, 0),
                aten.argmax(x, 1),
                aten.argmin(x, 1),
            )

        # 使用通用函数进行测试
        self.common(fn, (torch.randn([144, 144]),))
    # 定义一个测试方法，用于测试带有重复元素的 argmax 和 argmin 函数
    def test_argmax_argmin_with_duplicates(self):
        # 定义一个内部函数 fn，接受参数 x，返回各维度上的 argmax 和 argmin 结果
        def fn(x):
            return (
                aten.argmax(x, 0),  # 按列计算 x 的最大值索引
                aten.argmin(x, 0),  # 按列计算 x 的最小值索引
                aten.argmax(x, 1),  # 按行计算 x 的最大值索引
                aten.argmin(x, 1),  # 按行计算 x 的最小值索引
            )

        # 创建一个 6x6 的整数张量 t1，元素范围在 [0, 1)
        t1 = torch.randint(2, size=(6, 6))
        # 调用公共方法 common，传入 fn 和参数元组 (t1,)
        self.common(fn, (t1,))

        # 创建一个 32x32 的整数张量 t1，元素范围在 [0, 8)
        t1 = torch.randint(8, size=(32, 32))
        # 调用公共方法 common，传入 fn 和参数元组 (t1,)
        self.common(fn, (t1,))

        # 创建一个 1028x1028 的整数张量 t1，元素范围在 [0, 8)
        t1 = torch.randint(8, size=(1028, 1028))
        # 调用公共方法 common，传入 fn 和参数元组 (t1,)
        self.common(fn, (t1,))

    @skip_if_halide  # 跳过测试，若为 Halide，则涉及 NaN 的行为
    # 定义一个测试方法，用于测试带有 NaN 值的 argmax 和 argmin 函数
    def test_argmax_argmin_with_nan(self):
        # 定义一个内部函数 fn，接受参数 x，返回各维度上的 argmax 和 argmin 结果
        def fn(x):
            return (
                aten.argmax(x, 0),  # 按列计算 x 的最大值索引
                aten.argmin(x, 0),  # 按列计算 x 的最小值索引
                aten.argmax(x, 1),  # 按行计算 x 的最大值索引
                aten.argmin(x, 1),  # 按行计算 x 的最小值索引
            )

        # 如果设备为 CPU，则跳过测试并抛出跳过测试异常
        if self.device == "cpu":
            raise unittest.SkipTest("broken on CPU")

        # 创建一个 6x6 的浮点数张量 t1，元素从标准正态分布中随机生成，部分元素设为 NaN
        t1 = torch.randn((6, 6))
        t1[:, 1] = float("nan")  # 将第二列设为 NaN
        t1[:, 3] = float("nan")  # 将第四列设为 NaN
        # 调用公共方法 common，传入 fn 和参数元组 (t1,)
        self.common(fn, (t1,))

        # 创建一个 32x32 的浮点数张量 t1，元素从标准正态分布中随机生成，部分元素设为 NaN
        t1 = torch.randn((32, 32))
        t1[:, 4] = float("nan")  # 将第五列设为 NaN
        t1[:, 8] = float("nan")  # 将第九列设为 NaN
        # 调用公共方法 common，传入 fn 和参数元组 (t1,)
        self.common(fn, (t1,))

        # 创建一个 1028x1028 的浮点数张量 t1，元素从标准正态分布中随机生成，部分元素设为 NaN
        t1 = torch.randn((1028, 1028))
        t1[:, 40] = float("nan")  # 将第四十列设为 NaN
        t1[:, 100] = float("nan")  # 将第一百列设为 NaN
        # 调用公共方法 common，传入 fn 和参数元组 (t1,)
        self.common(fn, (t1,))
    # 定义一个测试函数，用于测试反向卷积操作
    def test_conv_backward(self):
        # 定义一个内部函数 fn，接受三种不同维度的输入，并执行反向卷积操作
        def fn(rank4_inps, rank3_inps, rank5_inps):
            # 执行反向卷积操作，返回梯度输出对输入的反向传播
            out1 = aten.convolution_backward(
                *rank4_inps,
                [C],
                [1, 1],
                [0, 0],
                [1, 1],
                False,
                [0, 0],
                1,
                [True, True, True],
            )
            # 再次执行反向卷积操作，返回梯度输出对输入的反向传播
            out2 = aten.convolution_backward(
                *rank4_inps,
                [C],
                [1, 1],
                [0, 0],
                [1, 1],
                False,
                [0, 0],
                1,
                [True, False, False],
            )
            # 继续执行反向卷积操作，返回梯度输出对输入的反向传播
            out3 = aten.convolution_backward(
                *rank3_inps,
                [C],
                [1],
                [0],
                [1],
                False,
                [0],
                1,
                [True, True, True],
            )
            # 最后执行反向卷积操作，返回梯度输出对输入的反向传播
            out4 = aten.convolution_backward(
                *rank5_inps,
                [C],
                [1, 1, 1],
                [0, 0, 0],
                [1, 1, 1],
                False,
                [0, 0, 0],
                1,
                [True, True, True],
            )
            # 返回四个反向卷积操作的结果
            return (out1, out2, out3, out4)

        # 设置常量值 B, C, H 分别为 3, 4, 5
        B = 3
        C = 4
        H = 5
        # 创建随机梯度输出、输入和权重张量
        grad_out = torch.randn(B, C, H - 2, H - 2, H - 2)
        inp = torch.randn(B, C, H, H, H)
        weight = torch.randn(C, C, 3, 3, 3)

        # 定义一个函数 shrink_rank，用于缩减张量的维度到指定的 rank
        def shrink_rank(x, rank):
            res = x
            while res.dim() > rank:
                res = torch.select(res, -1, 0)
            return res.contiguous()

        # 分别对梯度输出、输入和权重张量进行维度缩减，得到 rank4_inps, rank3_inps, rank5_inps
        rank4_inps = [shrink_rank(x, 4) for x in [grad_out, inp, weight]]
        rank3_inps = [shrink_rank(x, 4) for x in [grad_out, inp, weight]]
        rank5_inps = [shrink_rank(x, 5) for x in [grad_out, inp, weight]]

        # 启用 cudnn，并执行通用测试函数 common
        with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
            self.common(
                fn,
                [rank4_inps, rank3_inps, rank5_inps],
            )

    # 跳过该测试函数，显示 FIXME 注释，指出在最大/最小元素相等时的实现问题
    @unittest.skip(
        """
        FIXME: In the case of having equally max/min elements, our implementation returns
        the last index instead of the first one
        """
    )
    # 定义一个测试函数，测试 argmax 和 argmin 操作
    def test_argmax_argmin3(self):
        # 定义一个函数 fn，对输入张量执行 argmax 和 argmin 操作
        def fn(x):
            return (
                aten.argmax(x, 0),
                aten.argmin(x, 0),
                aten.argmax(x, -1),
                aten.argmin(x, -1),
            )

        # 执行通用测试函数 common，对随机生成的 10x10 整数张量进行测试
        self.common(
            fn,
            [torch.randint(0, 5, [10, 10])],
        )

    # 定义一个测试函数，测试 clamp_min 操作
    def test_vdd_clamp(self):
        # 定义一个函数 fn，对输入张量执行 clamp_min 操作
        def fn(x):
            return torch.clamp_min(x, 3)

        # 执行通用测试函数 common，对随机生成的大小为 16 的浮点数张量进行测试
        self.common(
            fn,
            [
                torch.randn([16], requires_grad=True) * 10,
            ],
        )
    @unittest.skipIf(
        os.environ.get("BUILD_ENVIRONMENT", "").startswith("parallelnative"),
        "TODO: debug this with asan",
    )


        # 如果运行环境是parallelnative，则跳过此测试用例，显示待修复的信息


    def test_tmp_not_defined_issue2(self):


        # 定义测试函数，用于检测临时未定义的问题2


        def forward(arg38_1, arg81_1, getitem_17, new_zeros_default_4):


            # 前向函数定义，接受四个参数：arg38_1, arg81_1, getitem_17, new_zeros_default_4


            div_tensor_7 = torch.ops.aten.div.Tensor(getitem_17, arg81_1)


            # 使用 torch.ops.aten.div.Tensor 计算 getitem_17 除以 arg81_1 的张量除法


            mul_tensor_24 = torch.ops.aten.mul.Tensor(div_tensor_7, arg38_1)


            # 使用 torch.ops.aten.mul.Tensor 计算 div_tensor_7 乘以 arg38_1 的张量乘法


            sum_default_7 = torch.ops.aten.sum.default(mul_tensor_24)


            # 使用 torch.ops.aten.sum.default 对 mul_tensor_24 进行张量求和


            return (new_zeros_default_4, sum_default_7)


            # 返回元组 (new_zeros_default_4, sum_default_7)


        dtype = torch.float32


        # 设置数据类型为 torch.float32


        args = [


            # 定义测试函数的输入参数列表


            ((1, 88, 40, 40), (140800, 1600, 40, 1), dtype),


            # 第一组参数元组，包含两个元组和数据类型


            ((), (), dtype),


            # 第二组参数元组，包含两个空元组和数据类型


            ((1, 88, 40, 40), (140800, 1600, 40, 1), dtype),


            # 第三组参数元组，包含两个元组和数据类型


            ((3,), (1,), dtype),


            # 第四组参数元组，每个包含一个元素的元组和数据类型


        args = [


            # 使用 rand_strided 生成的张量，对 args 进行重新赋值


            rand_strided(shape, stride, dtype).requires_grad_(True).add(1)


            # 生成带有指定形状、步幅和数据类型的张量，并设置其为需要梯度，然后加1


            for shape, stride, dtype in args


            # 对 args 中的每组参数进行循环


        self.common(forward, args, atol=1e-5, rtol=1e-5)


        # 调用 self.common 方法，传入前向函数 forward 和参数 args，并设置公差 atol 和相对公差 rtol


    @requires_gpu()


        # 要求 GPU 环境才能运行以下测试用例


    @skip_if_halide  # cascading accuracy issues due rsqrt fallback


        # 如果由于 rsqrt 回退导致级联的精度问题，则跳过以下测试用例


    @config.patch("halide.scheduler_cpu", "Mullapudi2016")


        # 使用 Mullapudi2016 的配置补丁来设置 halide.scheduler_cpu
    # 定义测试函数，测试地址对齐问题
    def test_misaligned_address_issue1(self):
        # 定义内部函数 forward，接受两个参数 sub_tensor_1 和 unsqueeze_default
        def forward(sub_tensor_1, unsqueeze_default):
            # 使用 torch 的 aten 操作中的 gather.default 方法
            gather_default = torch.ops.aten.gather.default(
                sub_tensor_1, 1, unsqueeze_default
            )
            return gather_default
        
        # 定义测试参数 args
        args = [
            ((1, 1000), (1000, 1), torch.float32),  # 第一个参数元组
            ((1, 1), (1, 1), torch.int64),  # 第二个参数元组
        ]
        # 使用 rand_strided 函数生成参数列表 args 的具体值
        args = [rand_strided(shape, stride, dtype) for shape, stride, dtype in args]
        # 调用通用测试方法 self.common 进行测试
        self.common(forward, args)

    # 定义测试函数，测试无效操作数问题
    def test_invalid_operand_issue1(self):
        # 定义内部函数 forward，接受六个参数
        def forward(arg0_1, arg1_1, arg3_1, squeeze, view_1, slice_1):
            # 使用 torch 的 aten 操作中的 slice_scatter.default 方法
            slice_scatter = torch.ops.aten.slice_scatter.default(
                slice_1, arg3_1, 1, 1, 9223372036854775807
            )
            # 使用 torch 的 aten 操作中的 slice_scatter.default 方法
            slice_scatter_1 = torch.ops.aten.slice_scatter.default(
                arg1_1, slice_scatter, 0, 0, 9223372036854775807
            )
            # 使用 torch 的 aten 操作中的 slice.Tensor 方法
            slice_2 = torch.ops.aten.slice.Tensor(
                slice_scatter_1, 0, 0, 9223372036854775807
            )
            # 使用 torch 的 aten 操作中的 select_scatter.default 方法
            select_scatter = torch.ops.aten.select_scatter.default(
                slice_2, squeeze, 1, 0
            )
            # 使用 torch 的 aten 操作中的 slice_scatter.default 方法
            slice_scatter_2 = torch.ops.aten.slice_scatter.default(
                slice_scatter_1, select_scatter, 0, 0, 9223372036854775807
            )
            # 使用 torch 的 aten 操作中的 view.default 方法
            view = torch.ops.aten.view.default(slice_scatter_2, [-1, 128])
            # 使用 torch 的 aten 操作中的 embedding.default 方法
            embedding = torch.ops.aten.embedding.default(arg0_1, view, 1)
            # 返回包含两个元素的列表 [embedding, view_1]
            return [embedding, view_1]

        # 定义测试参数 args
        args = [
            ((50005, 768), (768, 1), torch.float32),  # 第一个参数元组
            ((8, 128), (128, 1), torch.int64),  # 第二个参数元组
            ((8, 127), (127, 1), torch.int64),  # 第三个参数元组
            ((8,), (1,), torch.int64),  # 第四个参数元组
            ((1024,), (1,), torch.int64),  # 第五个参数元组
            ((8, 128), (128, 1), torch.int64),  # 第六个参数元组
        ]
        # 使用 rand_strided 函数生成参数列表 args 的具体值
        args = [rand_strided(shape, stride, dtype) for shape, stride, dtype in args]
        # 调用通用测试方法 self.common 进行测试
        self.common(forward, args)

    # 定义测试函数，测试 sizehint 问题
    def test_sizehint_issue1(self):
        # 定义内部函数 forward，接受一个参数 x
        def forward(x):
            # 使用 torch 的 nn.functional.unfold 方法进行展开操作
            return torch.nn.functional.unfold(
                x, kernel_size=[4, 4], dilation=1, padding=0, stride=[4, 4]
            )

        # 定义测试参数 args
        args = [((2, 24, 56, 56), (75264, 3136, 56, 1), torch.float32, False)]
        # 使用 rand_strided 函数生成参数列表 args 的具体值，并将返回的张量要求梯度
        args = [
            rand_strided(sh, st, dt).requires_grad_(rg) for (sh, st, dt, rg) in args
        ]
        # 调用通用测试方法 self.common 进行测试
        self.common(forward, args)
    # 测试零维度的张量减少操作
    def test_zero_dim_reductions(self):
        # 遍历两种情况：kd 为 True 和 False
        for kd in [True, False]:
            # 设置输入参数：一个零维度的张量、1 和 kd
            inps0 = (torch.zeros(2, 0, device=self.device, dtype=torch.float16), 1, kd)
            # 定义失败操作的列表
            failed_ops = [aten.argmin, aten.argmax, aten.max, aten.min]
            # 遍历失败操作列表
            for fo in failed_ops:
                # 使用断言检查是否抛出 IndexError，期望出现 "Expected reduction dim 1 to have non-zero size"
                with self.assertRaisesRegex(
                    IndexError, "Expected reduction dim 1 to have non-zero size"
                ):
                    # 构造模块并编译内部表示
                    mod = make_fx(fo)(*inps0)
                    _ = compile_fx_inner(mod, inps0)

            # 定义通过操作的列表，使用 lambda 函数生成
            pass_ops = [
                lambda *x: fn(*x) for fn in [aten.sum, aten.prod, aten.any, aten.all]
            ]
            # 遍历通过操作列表
            for po in pass_ops:
                # 使用 torch._dynamo.optimize("inductor") 对操作进行优化
                compiled = torch._dynamo.optimize("inductor")(po)
                # 计算预期结果和实际结果
                expected = po(*inps0)
                actual = compiled(*inps0)

                # 使用 assertTrue 检查实际结果和预期结果的近似性
                self.assertTrue(torch.allclose(actual, expected, atol=1e-3, rtol=1e-3))

    # 测试展开零维度的张量
    def test_unfold_zero_dimension_tensor(self):
        # 定义前向函数，对输入张量进行 unfold 操作
        def forward(x):
            return torch.unfold_copy(dimension=1, input=x, size=0, step=7)

        # 创建一个零维度的随机张量
        x = torch.rand([1, 0], dtype=torch.float32)

        # 调用 forward 函数并计算编译后的结果
        y = forward(x)
        compiled_y = torch.compile(forward, fullgraph=True)(x)

        # 使用 assertEqual 检查 y 和 compiled_y 是否相等
        self.assertEqual(y, compiled_y)

    # 测试零元素变异
    def test_zero_element_mutation(self):
        # 定义一个自定义的神经网络模型
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.LeakyReLU(negative_slope=5.2955089, inplace=True)

            def forward(self, inputs):
                return self.layer1(inputs)

        # 定义输入大小为 [0] 的张量
        ip_size = [0]
        # 创建一个随机的符合输入张量
        input_tensor = torch.randn(ip_size)

        # 实例化 CustomModel 类
        mymodel = CustomModel()
        # 使用 self.common 方法进行测试
        self.common(mymodel, (input_tensor,))

    # 测试 lerp 函数
    def test_lerp(self):
        # 非连续输入的 lerp 函数
        def fn0(i0, i1):
            # 转置输入张量 i0 的维度 -2 和 -3，并使用 lerp 函数
            x1 = i0.transpose(-2, -3)
            return torch.lerp(i1, x1, 70000)

        # 连续输入的 lerp 函数
        def fn1(i0, i1):
            return torch.lerp(i1, i0, 70000)

        # 使用 self.common 方法对 fn0 和 fn1 进行测试
        self.common(fn0, [torch.rand(10, 3, 10), torch.rand(3, 10, 10)])
        self.common(fn1, [torch.rand(3, 10, 10), torch.rand(3, 10, 10)])

    @skip_if_gpu_halide  # https://github.com/halide/Halide/issues/8318
    # 测试不确定输入
    def test_unspec_inputs(self):
        # 如果设备是 "cpu"，则跳过测试
        if self.device == "cpu":
            raise unittest.SkipTest("Testing mixed devices")

        # 定义函数 fn，对两个张量执行加、乘和除操作
        def fn(x, y):
            return x + y, x * y, x / y

        # 使用 torch._dynamo.optimize("inductor") 对 fn 函数进行优化
        opt = torch._dynamo.optimize("inductor")(fn)
        # 定义数据类型列表
        dtypes = [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        ]

        # 遍历数据类型列表
        for d in dtypes:
            # 创建具有给定数据类型和设备的张量作为输入
            inputs = (
                rand_strided((2, 3), (3, 1), dtype=torch.float32, device=GPU_TYPE),
                rand_strided((), (), dtype=d, device="cpu"),
            )
            # 使用 assertTrue 检查优化后的结果与原始函数结果是否相同
            self.assertTrue(same(opt(*inputs), fn(*inputs)))
            # 交换输入顺序，再次检查优化后的结果与原始函数结果是否相同
            inputs = (inputs[1], inputs[0])
            self.assertTrue(same(opt(*inputs), fn(*inputs)))
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    # 使用 @dynamo_config.patch 装饰器，设置自动动态形状为 True
    def test_list_clearing(self):
        # 检查设备是否为 CPU
        if self.device == "cpu":
            # 如果是 CPU，创建包含一个 nullcontext 的上下文列表
            contexts = [contextlib.nullcontext]
        else:
            # 如果不是 CPU，创建包含两个上下文的列表：
            # 1. nullcontext
            # 2. 针对 triton.cudagraphs 设置为 True 的 lambda 函数
            contexts = [
                contextlib.nullcontext,
                lambda: config.patch({"triton.cudagraphs": True}),
            ]
    
        # 遍历 contexts 列表中的每个上下文
        for context in contexts:
            # 使用当前上下文
            with context():
                # 创建输入张量列表 inps，包含两个随机张量，转移到指定设备
                inps = [
                    torch.rand([5, 5]).to(self.device),
                    torch.rand([5, 5]).to(self.device),
                ]
                # 创建输入张量的弱引用列表 inp_refs
                inp_refs = [weakref.ref(inp) for inp in inps]
    
                # 定义函数 fn，接受两个参数 x 和 y
                def fn(x, y):
                    # 计算张量 x 和 y 的和，并赋值给变量 a
                    a = x + y
                    # 返回计算结果的矩阵乘积作为单元素元组
                    return (a @ a,)
    
                # 使用 make_fx 函数将 fn 转换为 fx 格式
                fn_fx = make_fx(fn)(inps[0], inps[1])
                # 使用 compile_fx_inner 函数编译 fn_fx，传入输入张量列表 inps
                fn_compiled = compile_fx_inner(fn_fx, inps)
    
                # 保存当前 self 到 test_self 变量
                test_self = self
                # 初始化 matmul_seen 标志为 False
                matmul_seen = False
    
                # 定义 TestRefMode 类，继承自 TorchDispatchMode
                class TestRefMode(TorchDispatchMode):
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        kwargs = kwargs if kwargs else {}
    
                        nonlocal inps
                        nonlocal inp_refs
                        nonlocal test_self
                        nonlocal matmul_seen
    
                        # 通过矩阵乘积操作，应当释放输入张量
                        # TODO: 这可能是不必要的，引用循环？
                        gc.collect()
                        # 如果 func 是 aten.mm.out 函数
                        if func is aten.mm.out:
                            # 标记 matmul_seen 为 True
                            matmul_seen = True
                            # 断言 inps 的长度为 0
                            test_self.assertEqual(len(inps), 0)
                            # 断言 inp_refs 中的第一个和第二个元素均为 None
                            test_self.assertIsNone(inp_refs[0]())
                            test_self.assertIsNone(inp_refs[1]())
    
                        return func(*args, **kwargs)
    
                # 使用 TestRefMode 上下文执行 fn_compiled 函数，传入输入张量列表 inps
                with TestRefMode():
                    fn_compiled(inps)
    
                # 额外运行以确保在热身和记录时进行释放
                if self.device == GPU_TYPE:
                    # 如果设备是 GPU 类型，扩展 inps 列表，添加两个新的随机张量，并转移到指定设备
                    inps.extend(
                        [
                            torch.rand([5, 5]).to(self.device),
                            torch.rand([5, 5]).to(self.device),
                        ]
                    )
                    # 更新 inp_refs 列表，添加新输入张量的弱引用
                    inp_refs.extend([weakref.ref(inp) for inp in inps])
                    # 重新初始化 matmul_seen 标志为 False
                    matmul_seen = False
    
                    # 再次使用 TestRefMode 上下文执行 fn_compiled 函数，传入输入张量列表 inps
                    with TestRefMode():
                        fn_compiled(inps)
    
                # 检查 TorchDispatch 是否捕获到 cuda mm 调用
                # 即使没有 cudagraphs，也应该捕获到
                if self.device == "cpu":
                    # 如果设备是 CPU，断言 matmul_seen 为 True
                    self.assertTrue(matmul_seen)
                else:
                    # 如果设备不是 CPU，断言 inps 的长度为 0
                    self.assertEqual(len(inps), 0)
    
        # 定义 test_dtype_mismatch_issue 函数
        def test_dtype_mismatch_issue(self):
            # 定义函数 fn，接受一个参数 x
            def fn(x):
                # 对张量 x 进行零填充，填充尺寸为 [0, 1]，并应用 softmax 函数在指定维度上
                attn = torch.nn.functional.pad(x, [0, 1])
                return attn.softmax(dim=-1)
    
            # 创建随机张量 x，尺寸为 [128, 32, 63]
            x = torch.rand(128, 32, 63)
            # 调用 self 的 common 方法，传入函数 fn 和参数元组 (x,)
            self.common(fn, (x,))
    # 定义一个测试方法，用于测试 torch.diagonal_copy 函数
    def test_diagonal_copy(self):
        # 定义一个内部函数 fn，接受参数 x，并返回 torch.diagonal_copy(x) 的结果
        def fn(x):
            return torch.diagonal_copy(x)

        # 对于每个不同形状的随机张量 x 进行测试
        for x in (torch.randn(2, 3), torch.randn(2, 2), torch.randn(3, 2)):
            # 调用共同的测试方法 self.common，传入 fn 函数和其参数 x
            self.common(fn, (x,))

    # 测试 kwargs 参数的处理
    def test_kwargs(self):
        # 如果设备是 GPU_TYPE，则跳过此测试
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("histogramdd only supports cpu")

        # 定义一个函数 fn，接受参数 x 和 y，并返回 torch.histogramdd 的结果
        def fn(x, y):
            return torch.histogramdd(
                x,
                bins=[3, 3],
                weight=y,
            )

        # 调用共同的测试方法 self.common，传入 fn 函数和其参数列表
        self.common(
            fn,
            [torch.randn((4, 2)), torch.randn(4)],
        )

    # 测试形状填充对输入的影响
    # 由于形状填充会导致所有输入都被特化，因此代码生成测试会失败
    @expectedFailureCodegenDynamic
    @requires_gpu()
    @torch._inductor.config.patch("shape_padding", True)
    def test_shape_padding(self):
        # 定义浮点数类型列表 dtypes
        dtypes = [
            torch.float16,
            torch.float32,
        ]

        # 定义变量 b, m, n, k 并赋值
        b, m, n, k = 7, 11, 13, 15

        # 定义一个生成张量的函数 gen，接受形状参数和 dtype，默认为 torch.float32
        def gen(*shape, dtype=torch.float32):
            return torch.randn(*shape, device=GPU_TYPE, dtype=dtype) / k + 1.0

        # 对于每种数据类型 dtype 进行测试
        for dtype in dtypes:
            # 生成具有特定形状和 dtype 的张量 x, y, z
            x = gen(m, k, dtype=dtype)
            y = gen(k, n, dtype=dtype)
            z = gen(n, dtype=dtype)
            # 调用共同的测试方法 self.common，传入 lambda 函数和其参数
            self.common(lambda x, y: torch.mm(x, y), (x, y))
            self.common(lambda x, y: torch.matmul(x, y), (x, y))
            self.common(lambda x, y, z: torch.addmm(z, x, y), (x, y, z))

        # 对于每种数据类型 dtype 进行测试
        for dtype in dtypes:
            # 生成具有特定形状和 dtype 的张量 x, y, z
            x = gen(b, m, k, dtype=dtype)
            y = gen(b, k, n, dtype=dtype)
            z = gen(n, dtype=dtype)
            # 调用共同的测试方法 self.common，传入 lambda 函数和其参数
            self.common(lambda x, y: torch.bmm(x, y), (x, y))
            self.common(lambda x, y: torch.matmul(x, y), (x, y))
            self.common(lambda x, y, z: torch.baddbmm(z, x, y), (x, y, z))

    # 测试 Inductor 布局优化输入变异
    @requires_gpu()
    @torch._inductor.config.patch("layout_optimization", True)
    def test_inductor_layout_optimization_input_mutations(self):
        # 创建一个 nn.Conv2d 模块，使用 GPU_TYPE 设备，输入通道为 3，输出通道为 128
        mod = nn.Conv2d(3, 128, 1, stride=1, bias=False).to(GPU_TYPE)

        # 定义一个函数 f，接受参数 x，对 x 进行就地乘法操作，然后使用 mod 对象进行卷积，返回输出
        def f(x):
            x.mul_(2)
            out = mod(x)
            return out

        # 编译函数 f，生成 f_compiled
        f_compiled = torch.compile(f)
        # 创建一个随机张量 x_ref，并复制、分离为 x_test
        x_ref = torch.rand(2, 3, 128, 128, device=GPU_TYPE)
        x_test = x_ref.clone().detach()
        # 禁用梯度下降
        with torch.no_grad():
            # 计算原始和编译后的函数对 x_ref, x_test 的输出，并进行形状、步长的比较
            out_ref = f(x_ref)
            out_test = f_compiled(x_test)
            self.assertEqual(out_ref, out_test)
            self.assertEqual(out_ref.shape, out_test.shape)
            # 重要的是，因为 inductor._config.keep_output_stride 设置为 True，
            # 这里输出的步长应该匹配
            self.assertEqual(out_ref.stride(), out_test.stride())
            # 检查 x_ref, x_test 的值是否相等
            self.assertEqual(x_ref, x_test)
    # 定义测试函数，验证动态形状下整数输入的情况
    def test_int_input_dynamic_shapes(self):
        # 使用 torch.compile 注解标记函数 fn 支持动态计算
        @torch.compile(dynamic=True)
        def fn(x, i):
            # 计算 x 与 i 的乘积并返回结果
            y = x * i
            return y

        # 调用 common 函数测试 fn 函数，验证不同输入情况
        # 参数包括一个形状为 [3, 1, 1, 1, 1] 的随机张量和一个整数 9132
        self.common(fn, [torch.randn(3, 1, 1, 1, 1), 9132])

    # 定义测试函数，验证动态形状下的平方根操作
    def test_sqrt_dynamic_shapes(self):
        # 提示信息：TIMM convit_base 模型相关的问题参见 GitHub issue 97877
        # TODO: 支持 CUDA 路径的开发计划
        if self.device == GPU_TYPE:
            # 如果设备为 GPU 类型，则跳过当前测试
            raise unittest.SkipTest("sqrt dynamic shapes only supports cpu")

        # 定义一个简单的 PyTorch 模型类
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # 获取输入张量 x 的形状信息 B, N, C
                B, N, C = x.shape
                # 调用 get_rel_indices 方法返回相对索引
                return self.get_rel_indices(N)

            def get_rel_indices(self, num_patches: int) -> torch.Tensor:
                # 计算图片大小，假设图像为正方形
                img_size = int(num_patches**0.5)
                # 创建一个从 0 到 img_size-1 的索引张量
                ind = torch.arange(img_size)
                return ind

        # 调用 common 函数测试 Model 类的实例，验证输入为形状为 [8, 4, 4] 的随机张量
        self.common(
            Model(),
            [
                torch.randn(8, 4, 4),
            ],
        )

    # 定义测试函数，验证动态形状下的倒数平方根操作
    def test_rsqrt_dynamic_shapes(self):
        # 提示信息：来自 HF hf_BigBird 模型
        @torch.compile(dynamic=True)
        def fn(a, b):
            # 计算 a 的第二个维度的平方根倒数
            r = 1 / math.sqrt(a.size(1))
            # 计算张量 a 和 b 的批次矩阵乘积，除以 r 并返回结果
            return torch.bmm(a, b) / r

        # 调用 common 函数测试 fn 函数，验证输入为两个形状为 [2, 4, 4] 的随机张量
        self.common(
            fn,
            [
                torch.randn(2, 4, 4),
                torch.randn(2, 4, 4),
            ],
        )
    def test_index_dynamic_shapes(self):
        # 定义一个测试函数，用于处理动态形状索引问题
        # Repro from vision_maskrcnn
        # 从 vision_maskrcnn 复现的问题

        def fn(arg0_1):
            # 将输入张量在第0维度上进行unsqueeze操作
            unsqueeze = arg0_1.unsqueeze(0)
            # 获取第1维度的大小
            sym_size = arg0_1.size(1)
            # 对第1维度大小乘以1.8735363483428955并向上取整
            ceil = math.ceil(sym_size * 1.8735363483428955)
            # 调用torch的iota操作创建一个整数序列
            iota = torch.ops.prims.iota.default(
                ceil,
                start=0,
                step=1,
                dtype=torch.int64,
                device=arg0_1.device,
                requires_grad=False,
            )
            # 将整数序列转换为float32类型
            convert_element_type_1 = iota.to(torch.float32)
            # 获取第2维度的大小
            sym_size_1 = arg0_1.size(2)
            # 对第2维度大小乘以1.8735363483428955并向下取整
            floor_1 = math.floor(sym_size_1 * 1.8735363483428955)
            # 再次向上取整第2维度大小
            ceil_1 = math.ceil(floor_1)
            # 调用torch的iota操作创建另一个整数序列
            iota_1 = torch.ops.prims.iota.default(
                ceil_1,
                start=0,
                step=1,
                dtype=torch.int64,
                device=arg0_1.device,
                requires_grad=False,
            )
            # 将第一个整数序列转换为float32类型
            convert_element_type_3 = iota_1.to(torch.float32)
            # 计算sub_2，涉及一系列数学运算
            sub_2 = (convert_element_type_1 + 0.5) * (sym_size / ceil) - 0.5
            # 对sub_2应用clamp_min操作
            clamp_min = sub_2.clamp_min(0.0)
            # 计算sub_3，涉及一系列数学运算
            sub_3 = (convert_element_type_3 + 0.5) * (sym_size_1 / floor_1) - 0.5
            # 对sub_3应用clamp_min操作
            clamp_min_1 = sub_3.clamp_min(0.0)
            # 将clamp_min转换为torch.int64类型
            convert_element_type_4 = clamp_min.to(torch.int64)
            # 计算sub_4
            sub_4 = sym_size - 1
            # 对clamp_min应用ceil和clamp_max操作
            clamp_max = clamp_min.ceil().clamp_max(sub_4)
            # 将clamp_max转换为torch.int64类型
            convert_element_type_5 = clamp_max.to(torch.int64)
            # 将clamp_min_1转换为torch.int64类型
            convert_element_type_6 = clamp_min_1.to(torch.int64)
            # 对convert_element_type_4在第1维度上进行unsqueeze操作
            unsqueeze_2 = convert_element_type_4.unsqueeze(1)
            # 调用torch的aten.index.Tensor操作
            index = torch.ops.aten.index.Tensor(
                unsqueeze, [None, None, unsqueeze_2, convert_element_type_6]
            )
            # 再次调用torch的aten.index.Tensor操作
            index_1 = torch.ops.aten.index.Tensor(
                unsqueeze,
                [
                    None,
                    None,
                    convert_element_type_5.unsqueeze(1),
                    convert_element_type_6,
                ],
            )
            # 计算sub_6
            sub_6 = clamp_min.unsqueeze(1) - unsqueeze_2
            # 计算mul_10，涉及一系列数学运算
            mul_10 = (index * (1.0 - sub_6) + index_1 * (sub_6)) * (
                1.0 - (clamp_min_1 - convert_element_type_6)
            )
            # 调用torch的aten.select.int操作
            select = torch.ops.aten.select.int(mul_10, 0, 0)
            # 返回结果元组
            return (select,)

        # 生成一个形状为(15, 20, 3)的随机张量
        x = torch.randn(15, 20, 3)
        # 调用self.common方法进行测试
        self.common(
            fn,
            [x],
        )
    # 定义一个测试方法，用于测试在使用整数参数设置张量元素时的行为
    def test_setitem_with_int_parameter(self):
        # 创建一个形状为 (7,) 的全零张量，并指定设备为 self.device
        x = torch.zeros(7, device=self.device)

        # 定义一个内部函数 fn，接受参数 n 和 a，将 a 中索引为 n 的元素设置为 -1，并返回修改后的张量 a
        def fn(n, a):
            a[n] = -1
            return a

        # 创建一个 CompileCounterWithBackend 实例，命名为 cnts，使用 "inductor" 作为后端
        cnts = CompileCounterWithBackend("inductor")
        # 对 fn 进行优化，设置 nopython=True，得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)

        # 遍历范围为 [2, x.shape[0]) 的整数 n
        for n in range(2, x.shape[0]):
            # 调用优化后的函数 opt_fn，传入参数 n 和 x
            opt_fn(n, x)
            # 断言 x 的第 n 个元素是否等于 -1
            self.assertEqual(x[n], -1)

        # 如果 assume_static_by_default 设置为 True，则上述调用将触发 3 次函数编译
        # 分别是：
        #   1. 假设 'n' 是静态的（等于 2）
        #   2. 将 'n' 设置为动态，但有一个保护条件 'end <= x.shape[0]'（来自 torch._inductor.ir.SliceView.create）
        frame_count = 2 if torch._dynamo.config.assume_static_by_default else 1
        # 断言 cnts 的帧计数是否等于 frame_count
        self.assertEqual(cnts.frame_count, frame_count)

        # 负索引触发新的编译
        opt_fn(-x.shape[0], x)
        # 断言 x 的第一个元素是否等于 -1
        self.assertEqual(x[0], -1)
        # 断言 cnts 的帧计数是否为 frame_count 加 1
        self.assertEqual(cnts.frame_count, frame_count + 1)

    # 使用 config.patch 设置 profiler_mark_wrapper_call=True 进行修饰的测试方法
    def test_profiler_mark_wrapper_call(self):
        # 导入 profile 方法从 torch.profiler 模块中
        from torch.profiler import profile

        # 定义一个函数 fn，使用 torch._dynamo.optimize 优化，设置 nopython=True
        def fn(a, b):
            return a + b

        # 创建大小为 (100,) 的随机张量 a 和 b
        a = torch.rand((100,))
        b = torch.rand((100,))
        # 使用 profile() 进行性能分析，记录 fn(a, b) 的执行情况
        with profile() as prof:
            fn(a, b)
        # 断言在 prof.profiler.function_events 中是否有包含 "inductor_wrapper_call" 的事件名称
        assert any(
            "inductor_wrapper_call" in e.name for e in prof.profiler.function_events
        )

    # 测试在无关紧要的步幅（strides）情况下的行为
    def test_insignificant_strides(self):
        # 定义一个函数 f，接受一个张量 x 作为参数，计算 x + 1 并对结果进行形状变换
        def f(x):
            tmp = x + 1
            return tmp.view(-1, 1, 2)

        # 创建一个大小为 8 的张量 x，数据类型为 torch.float32，设备为 self.device
        x = torch.arange(8, device=self.device, dtype=torch.float32)
        # 调用 torch.compile 对函数 f 进行编译，得到 compiled_out
        compiled_out = torch.compile(f)(x)

        # 断言 out 的步幅与 compiled_out 的步幅是否相等
        self.assertEqual(out.stride(), compiled_out.stride())
        # 断言 out 与 compiled_out 是否相等
        self.assertEqual(out, compiled_out)

    # 如果在 x86 架构且不支持 AVX2，则跳过此测试
    def test_pixel_shuffle_channels_last(self):
        # 定义一个函数 fn，接受一个张量 x 作为参数，对 x 进行像素洗牌和 ReLU 激活操作，并返回结果
        def fn(x):
            x = torch.nn.functional.pixel_shuffle(x, 2)
            x = torch.nn.functional.relu(x)
            return x

        # 调用 self.common 方法，传入 fn 和一个元组作为参数，元组包含一个大小为 (1, 16, 64, 72) 的张量
        # 并设置其内存格式为 torch.channels_last
        self.common(
            fn,
            (torch.randn(1, 16, 64, 72).to(memory_format=torch.channels_last),),
        )
   `
    def test_where_broadcast(self):
        # 定义一个测试函数，用于验证 torch.where 的广播行为
        def fn(x, p1, p0):
            # 使用 torch.where 函数根据条件 x 选择 p1 或 p0 中的元素组成输出张量 o
            o = torch.where(x, p1, p0)
            return o

        # 定义一个测试类，用于复现 GitHub 上的问题
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个缓冲区变量 _tensor_constant0，内容为随机生成的标量
                self.register_buffer(
                    "_tensor_constant0", torch.randn([], dtype=torch.float32)
                )

            def forward(self, arg0_1, arg1_1):
                # 将 arg1_1 张量转换为布尔类型
                convert_element_type = torch.ops.prims.convert_element_type.default(
                    arg1_1, torch.bool
                )
                # 对布尔类型张量取按位取反操作
                bitwise_not = torch.ops.aten.bitwise_not.default(convert_element_type)
                # 获取 self._tensor_constant0 的一个新拷贝
                _tensor_constant0 = self._tensor_constant0
                lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(
                    _tensor_constant0
                )
                # 使用 torch.ops.aten.where 方法根据条件 bitwise_not 选择 lift_fresh_copy 或 arg0_1
                where = torch.ops.aten.where.self(bitwise_not, lift_fresh_copy, arg0_1)
                return (where, bitwise_not)

        # 调用 self.common 方法进行测试
        self.common(
            fn,
            (torch.tensor([[True]]), torch.rand(13, 7, 3), torch.rand(1, 1)),
        )

        # 准备测试参数 args
        args = [
            torch.randn(1, 4, 64, 64),
            torch.zeros(1, 1, 64, 64, dtype=torch.uint8),
        ]
        # 修改 args[1] 张量的部分值为 1
        args[1][:, :, :32, :32] = 1
        # 创建 args 的深拷贝 eager_args
        eager_args = [x.clone() for x in args]
        # 创建 Repro 类的实例 eager_mod
        eager_mod = Repro()
        # 使用 make_fx 函数对 eager_mod 进行编译
        mod = make_fx(eager_mod, tracing_mode="real")(*args)
        # 编译模型
        compiled = compile_fx_inner(mod, args)
        # 调用编译后的模型获取输出
        inductor_out = compiled(args)
        # 直接调用 eager_mod 获取输出
        eager_out = eager_mod(*eager_args)
        # 断言两种方式的输出是否相等
        self.assertEqual(inductor_out, eager_out)

    @skipIfRocm
    def test_require_stride_expanded(self):
        # 定义一个测试函数，用于测试需要扩展步长的情况
        def forward(arg6, arg7, arg16):
            # 使用 torch.ops.aten.convolution 方法进行卷积操作
            convolution = torch.ops.aten.convolution(
                arg16.unsqueeze(0), arg7, arg6, [4, 4], [2, 2], [1, 1], False, [0, 0], 1
            )
            return (convolution,)

        # 调用 self.common 方法进行测试
        self.common(
            forward,
            (
                None,
                rand_strided(
                    (64, 3, 11, 11),
                    (363, 121, 11, 1),
                    torch.float32,
                    device=self.device,
                ).to(memory_format=torch.channels_last),
                rand_strided(
                    (1, 3, 224, 224),
                    (150528, 50176, 224, 1),
                    torch.float32,
                    device=self.device,
                )
                .to(memory_format=torch.channels_last)
                .squeeze(0),
            ),
            atol=1e-3,
            rtol=0.001,
        )

        # 断言生成的内核数量是否等于 0
        assertGeneratedKernelCountEqual(self, 0)

    @requires_gpu()
    @parametrize("use_block_ptr", (False, True))
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Does not support SDPA or pre-SM80 hardware",
    )
    @skipIfRocm
    # 定义一个测试函数，测试使用 block 指针的情况
    def test_sdpa(self, use_block_ptr):
        # 定义内部函数 foo，接受五个参数
        def foo(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
            # 使用 torch.ops.aten.view.default 对 arg3_1 进行视图变换，生成 view
            view = torch.ops.aten.view.default(arg3_1, [23760, 128])
            # 清空 arg3_1 引用
            arg3_1 = None
            # 使用 torch.ops.aten.mm.default 计算 view 和 arg4_1 的矩阵乘积，生成 mm
            mm = torch.ops.aten.mm.default(view, arg4_1)
            # 清空 view 和 arg4_1 引用
            view = arg4_1 = None
            # 使用 torch.ops.aten.view.default 对 mm 进行视图变换，生成 view_1
            view_1 = torch.ops.aten.view.default(mm, [3, 99, 80, 8])
            # 清空 mm 引用
            mm = None
            # 使用 torch.ops.aten.view.default 对 view_1 进行视图变换，生成 view_2
            view_2 = torch.ops.aten.view.default(view_1, [3, 99, 80, 8])
            # 清空 view_1 引用
            view_1 = None
            # 使用 torch.ops.aten.permute.default 对 view_2 进行维度排列，生成 permute
            permute = torch.ops.aten.permute.default(view_2, [0, 3, 1, 2])
            # 清空 view_2 引用
            view_2 = None
            # 使用 torch.ops.aten.view.default 对 permute 进行视图变换，生成 view_3
            view_3 = torch.ops.aten.view.default(permute, [3, 8, 99, 80])
            # 清空 permute 引用
            permute = None

            # 使用 torch.ops.aten.clone.default 克隆 view_3，指定内存格式为连续的
            clone = torch.ops.aten.clone.default(
                view_3, memory_format=torch.contiguous_format
            )
            # 清空 view_3 引用
            view_3 = None

            # 使用 torch.ops.aten.expand.default 对 clone 进行维度扩展，生成 expand
            expand = torch.ops.aten.expand.default(clone, [3, 8, 99, 80])
            # 清空 clone 引用
            clone = None
            # 使用 torch.ops.aten._scaled_dot_product_efficient_attention.default
            # 计算 scaled dot product efficient attention，生成 _scaled_dot_product_efficient_attention
            _scaled_dot_product_efficient_attention = (
                torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    arg0_1, arg1_1, arg2_1, expand, False
                )
            )
            # 清空参数和 expand 引用
            arg0_1 = arg1_1 = arg2_1 = expand = None
            # 从 _scaled_dot_product_efficient_attention 结果中获取第一个元素，生成 getitem
            getitem = _scaled_dot_product_efficient_attention[0]
            # 清空 _scaled_dot_product_efficient_attention 引用
            _scaled_dot_product_efficient_attention = None
            # 返回结果的元组，包含 getitem
            return (getitem,)

        # 如果设备是 CPU，则跳过测试，提示需要 GPU 类型
        if self.device == "cpu":
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        # 设置设备和数据类型常量
        DEVICE = torch.device(f"{GPU_TYPE}:0")
        DTYPE = torch.float16
        B = 3
        H = 8
        Q = 99
        K = 80
        D = 32
        C_bias = 128

        # 准备输入数据
        query = torch.randn((B, H, Q, D), device=DEVICE, dtype=DTYPE)
        key = torch.randn((B, H, K, D), device=DEVICE, dtype=DTYPE)
        value = torch.randn((B, H, K, D), device=DEVICE, dtype=DTYPE)
        bias = torch.randn((B, Q, K, C_bias), device=DEVICE, dtype=DTYPE)
        weights = torch.randn((C_bias, H), device=DEVICE, dtype=DTYPE)
        inps = (query, key, value, bias, weights)

        # 使用 config.patch 设置上下文，根据 use_block_ptr 参数修改 triton.use_block_ptr 的值
        with config.patch("triton.use_block_ptr", use_block_ptr):
            # 调用 self.common 方法进行通用测试，传入 foo 和 inps，设置绝对误差 atol 和相对误差 rtol
            self.common(
                foo,
                inps,
                atol=0.02,
                rtol=1e4,
            )

            # 对优化后的 foo 使用 torch._dynamo.optimize("inductor")，生成 foo_opt
            foo_opt = torch._dynamo.optimize("inductor")(foo)
            # 运行并获取 triton 代码，传入 foo_opt 和 inps，生成 code
            code = run_and_get_triton_code(foo_opt, *inps)
            # 检查 code 中是否包含 "tl.make_block_ptr"，生成 have_block_ptr
            have_block_ptr = code.count("tl.make_block_ptr") > 0
            # 如果设备不是 Halide 后端，断言 have_block_ptr 是否与 use_block_ptr 相等
            if not is_halide_backend(self.device):
                self.assertEqual(have_block_ptr, use_block_ptr)
    # 定义一个测试方法，用于测试非对齐掩码场景下的注意力计算
    def test_sdpa_unaligned_mask(self):
        # 定义一个内部函数 foo，接受四个参数，均为四维浮点数张量的类型注释
        def foo(
            arg0_1: "f32[8, 8, 16, 16]",
            arg1_1: "f32[8, 8, 15, 16]",
            arg2_1: "f32[8, 8, 15, 16]",
            arg3_1: "f32[1, 1, 16, 15]",
        ):
            # 使用 arg3_1 构造一个常数填充的 N 维张量，填充方式是在后两个维度末尾填充 0
            constant_pad_nd: "f32[1, 1, 16, 16]" = (
                torch.ops.aten.constant_pad_nd.default(arg3_1, [0, 1], 0.0)
            )
            # 将 arg3_1 置为 None，释放其内存
            arg3_1 = None
            # 从 constant_pad_nd 中切片获取一个新的张量 slice_1，切片的起始位置为倒数第二个维度的第 0 个元素，结束位置为第 15 个元素
            slice_1: "f32[1, 1, 16, 15]" = torch.ops.aten.slice.Tensor(
                constant_pad_nd, -1, 0, 15
            )
            # 将 constant_pad_nd 置为 None，释放其内存
            constant_pad_nd = None
            # 使用 slice_1 张量进行扩展，扩展成维度为 [8, 8, 16, 15] 的新张量 expand
            expand: "f32[8, 8, 16, 15]" = torch.ops.aten.expand.default(
                slice_1, [8, 8, 16, 15]
            )
            # 将 slice_1 置为 None，释放其内存
            slice_1 = None
            # 使用 efficient_attention 算子计算加权点积注意力，返回值赋给 _scaled_dot_product_efficient_attention
            _scaled_dot_product_efficient_attention = (
                torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    arg0_1, arg1_1, arg2_1, expand, False
                )
            )
            # 将 arg0_1, arg1_1, arg2_1, expand 四个张量置为 None，释放它们的内存
            arg0_1 = arg1_1 = arg2_1 = expand = None
            # 从 _scaled_dot_product_efficient_attention 中获取第一个元素，赋给 getitem
            getitem: "f32[8, 8, 16, 16]" = _scaled_dot_product_efficient_attention[0]
            # 将 _scaled_dot_product_efficient_attention 置为 None，释放其内存
            _scaled_dot_product_efficient_attention = None
            # 返回一个包含 getitem 的元组
            return (getitem,)

        # 创建一个随机张量 query，表示查询
        query = torch.rand(8, 8, 16, 16, device=GPU_TYPE)
        # 创建一个随机张量 key，表示键
        key = torch.rand(8, 8, 15, 16, device=GPU_TYPE)
        # 创建一个随机张量 value，表示值
        value = torch.rand(8, 8, 15, 16, device=GPU_TYPE)
        # 创建一个随机张量 bias，表示偏置
        bias = torch.rand(1, 1, 16, 15, device=GPU_TYPE)
        # 调用 self.common 方法，传入 foo 函数作为参数，同时传入 (query, key, value, bias) 元组，还设定 atol 和 rtol 参数
        self.common(
            foo,
            (query, key, value, bias),
            atol=0.02,
            rtol=1e4,
        )

    # 标记需要 GPU 支持，并跳过不支持 mem_eff_attention 的平台
    @requires_gpu()
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Does not support mem_eff_attention",
    )
    # 应用 config.patch 进行装饰，设置 freezing=True
    @config.patch(freezing=True)
    def test_sdpa_unaligned_mask_freezing(self):
        # 定义一个内部的 PyTorch 模块类
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个随机张量作为模块的属性
                self.arg3_1 = torch.rand(1, 1, 16, 15, device=GPU_TYPE)

            # 定义模块的前向传播函数
            def forward(
                self,
                arg0_1: "f32[8, 8, 16, 16]",
                arg1_1: "f32[8, 8, 15, 16]",
                arg2_1: "f32[8, 8, 15, 16]",
            ):
                # 将模块属性的值赋给局部变量
                arg3_1 = self.arg3_1
                # 使用 torch.ops.aten.constant_pad_nd.default 函数进行常数填充
                constant_pad_nd: "f32[1, 1, 16, 16]" = (
                    torch.ops.aten.constant_pad_nd.default(arg3_1, [0, 1], 0.0)
                )
                # 清空局部变量 arg3_1 的值
                arg3_1 = None
                # 使用 torch.ops.aten.slice.Tensor 进行张量切片操作
                slice_1: "f32[1, 1, 16, 15]" = torch.ops.aten.slice.Tensor(
                    constant_pad_nd, -1, 0, 15
                )
                # 清空常数填充张量的值
                constant_pad_nd = None
                # 使用 torch.ops.aten.expand.default 函数进行张量扩展
                expand: "f32[8, 8, 16, 15]" = torch.ops.aten.expand.default(
                    slice_1, [8, 8, 16, 15]
                )
                # 清空切片张量的值
                slice_1 = None
                # 使用 torch.ops.aten._scaled_dot_product_efficient_attention.default 函数进行有效注意力计算
                _scaled_dot_product_efficient_attention = (
                    torch.ops.aten._scaled_dot_product_efficient_attention.default(
                        arg0_1, arg1_1, arg2_1, expand, False
                    )
                )
                # 清空前向传播函数的所有输入和中间变量的值
                arg0_1 = arg1_1 = arg2_1 = expand = None
                # 从注意力计算结果中获取特定项
                getitem: "f32[8, 8, 16, 16]" = _scaled_dot_product_efficient_attention[
                    0
                ]
                # 清空注意力计算结果的值
                _scaled_dot_product_efficient_attention = None
                # 返回获取的结果项
                return (getitem,)

        # 生成随机查询、键和值张量
        query = torch.rand(8, 8, 16, 16, device=GPU_TYPE)
        key = torch.rand(8, 8, 15, 16, device=GPU_TYPE)
        value = torch.rand(8, 8, 15, 16, device=GPU_TYPE)

        # 创建 Mod 类的实例
        mod = Mod()
        # 使用实例进行前向传播，获取结果
        out_eager = mod(query, key, value)

        # 使用 torch.no_grad 上下文，编译模块并获取编译后的输出结果
        with torch.no_grad():
            out_compiled = torch.compile(mod)(query, key, value)
            # 断言计算出的结果与即时计算的结果在给定的误差范围内相等
            self.assertEqual(out_eager, out_compiled, atol=0.02, rtol=1e4)

    def test_where_with_logical_op(self):
        # 定义一个函数，使用逻辑与操作进行 torch.where 的应用
        def fn_and(x, y):
            return torch.where(torch.logical_and(x, y), 1.0, 0.0)

        # 定义一个函数，使用逻辑或操作进行 torch.where 的应用
        def fn_or(x, y):
            return torch.where(torch.logical_or(x, y), 1.0, 0.0)

        # 使用通用函数 common 运行 fn_and 和 fn_or，传入随机生成的两个张量
        self.common(
            fn_and,
            (torch.randn(32), torch.randn(32)),
        )
        self.common(
            fn_or,
            (torch.randn(32), torch.randn(32)),
        )

    # 跳过如果在 ROCm 平台上执行时
    @skipIfRocm
    def test_conv_with_as_strided(self):
        # 定义一个继承自 nn.Module 的模型类 Model
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个 2D 卷积层，输入通道数为 256，输出通道数为 384，核大小为 (1, 1)，步长为 (1, 1)，无偏置
                self.kv = torch.nn.Conv2d(
                    256, 384, kernel_size=(1, 1), stride=(1, 1), bias=False
                )

            def forward(self, x):
                # 对输入 x 进行卷积操作
                convolution = self.kv(x)
                # 对卷积结果进行常数填充，填充值为 0，填充尺寸为 [2, 2, 2, 2]
                constant_pad_nd = torch.ops.aten.constant_pad_nd.default(
                    convolution, [2, 2, 2, 2], 0.0
                )
                # 使用 as_strided 函数创建一个新的张量，尺寸为 [8, 384, 2, 20, 12]，步长为 [153600, 400, 160, 1, 20]
                as_strided = torch.ops.aten.as_strided.default(
                    constant_pad_nd, [8, 384, 2, 20, 12], [153600, 400, 160, 1, 20]
                )
                # 使用 as_strided 函数再次创建一个新的张量，尺寸为 [8, 384, 2, 2, 12, 12]，步长为 [153600, 400, 160, 8, 20, 1]
                as_strided_1 = torch.ops.aten.as_strided.default(
                    as_strided, [8, 384, 2, 2, 12, 12], [153600, 400, 160, 8, 20, 1]
                )
                # 使用 clone 函数复制张量 as_strided_1，并指定内存格式为 torch.contiguous_format
                clone = torch.ops.aten.clone.default(
                    as_strided_1, memory_format=torch.contiguous_format
                )
                return clone

        # 使用 self.common 方法对 Model 类的实例进行测试，传入一个形状为 (8, 256, 16, 16) 的张量作为输入
        self.common(
            Model(),
            (torch.randn(8, 256, 16, 16),),
            check_lowp=not is_halide_backend(self.device),
        )

    def test_inplace_where_pointwise(self):
        # 定义一个操作函数 fn，接受两个参数 a 和 b，修改 a 的第一个元素为 2，然后返回 a 与 b 的乘积
        def fn(a, b):
            a[0] = 2
            return a * b

        # 使用 self.common 方法对 fn 函数进行测试，传入两个张量作为参数
        self.common(fn, (torch.rand(1), torch.rand(2)))

    def test_view_on_aliased(self):
        # 定义一个操作函数 fn1，接受两个参数 a 和 b，对 a 取最大值后取 values，然后将结果与 b 拼接并四舍五入，最后返回结果 c
        def fn1(a, b):
            a = a.max(0).values
            c = torch.cat((a, b))
            c = c.round()
            b >= a[0]  # noqa: B015
            return c

        # 定义一个常量张量 some_const
        some_const = torch.tensor(6324)

        # 定义一个操作函数 fn2，不接受参数，返回一个张量 a 和 a 在垂直方向上拼接后的结果
        def fn2():
            a = torch.tensor([[0.6324]])
            ret = torch.cat((a, a), dim=0)
            some_const >= a[0]  # noqa: B015
            return ret

        # 使用 self.common 方法对 fn1 和 fn2 函数进行测试，传入相应的参数
        self.common(fn1, (torch.tensor([[4.0]]), torch.tensor([5.0])))
        self.common(fn2, ())

    def test_argmax_to_float(self):
        # 定义一个操作函数 fn，创建一个 2x2 的零张量 a，对其进行列方向上的 argmax 操作，然后转换为 float 类型并计算平均值
        def fn():
            a = torch.zeros([2, 2])
            b = a.argmax(0)
            return b.float().mean()

        # 使用 self.common 方法对 fn 函数进行测试，不传入任何参数
        self.common(fn, ())

    def test_const_int32_to_float(self):
        # 定义一个操作函数 fn，创建一个形状为 [1, 2]，数据类型为 torch.int32 的零张量 a，
        # 将 a 与自身相加得到新的张量 b，然后将 b 转换为 torch.float32 类型，并乘以 0.8 后返回结果
        def fn():
            a = torch.zeros([1, 2], dtype=torch.int32)
            a = a + a
            b = a.to(dtype=torch.float32)
            return b * 0.8

        # 使用 self.common 方法对 fn 函数进行测试，不传入任何参数
        self.common(fn, ())
    # 定义一个单元测试函数，测试索引操作的正确性
    def test_getitem(self):
        # 定义输出特征列表
        out_features = ["p3", "p4", "p5", "p6", "p7"]
        # 定义输入特征
        in_feature = "p5"

        # 定义一个内部函数，根据输入的列表返回特定特征的值
        def fn(a):
            return a[out_features.index(in_feature)]

        # 创建三个张量作为输入数据
        x = [
            torch.rand([1, 256, 100, 152], device=self.device),
            torch.rand([1, 256, 50, 76], device=self.device),
            torch.rand([1, 256, 25, 38], device=self.device),
        ]
        
        # 优化 fn 函数的执行
        opt_fn = torch._dynamo.optimize("inductor")(fn)
        # 断言优化后的函数与原函数在相同输入下结果一致
        same(fn(x), opt_fn(x))

    # 定义一个单元测试函数，测试张量填充和视图重塑操作
    def test_pad_view(self):
        # 定义一个内部函数，对输入张量进行填充和视图重塑操作
        def fn(a):
            # 在第二维和第三维度上对张量进行零填充，第四维度上填充一个单位
            y = torch.nn.functional.pad(a, (0, 0, 0, 1))
            # 将填充后的张量视图重塑为指定形状
            y = y.view(*y.size()[:-2], y.size(-1), y.size(-2))
            return y

        # 创建一个形状为 [48, 3, 512, 512] 的随机张量作为输入
        x = torch.rand(48, 3, 512, 512)
        # 使用共享的测试辅助函数进行测试
        self.common(fn, (x,))

    # 定义一个单元测试函数，测试张量类型转换和填充操作
    def test_pad_cast(self):
        # 定义一个内部函数，对输入张量进行类型转换和填充操作
        def fn(x):
            # 将输入张量转换为 float32 类型后，在第二和第四维度上分别填充 3 和 0 个单位
            return torch.nn.functional.pad(x.to(torch.float32), (0, 3, 0, 0))

        # 针对不同的数据类型进行测试
        for dtype in [torch.int32, torch.int64]:
            # 使用共享的测试辅助函数进行测试，输入形状为 [1, 1, 13]
            self.common(fn, (torch.ones(1, 1, 13, dtype=dtype),))

    # 跳过测试如果没有 CPU 支持，要求需要 C++ 编译器
    @unittest.skipIf(not HAS_CPU, "requires C++ compiler")
    @skip_if_halide  # bf16
    # 调用 div 函数时，只支持 torch.SymInt 类型的参数，暂不支持其它参数
    # 为了支持此行为，需要允许存储 symint 数据的常量张量进行常量传播
    # 目前，当 dynamo 遇到这种行为时会显式地中断图构建过程
    @expectedFailureCodegenDynamic
    @skip_if_gpu_halide  # 精度错误
    # 定义测试函数，用于测试 AllenaiLongformerBase 模型的重现性
    def test_AllenaiLongformerBase_repro(self):
        # 定义内部函数 fn，处理查询、分数和窗口重叠
        def fn(query, scores, window_overlap):
            # 获取查询的批量大小、序列长度、头数及其它参数
            batch_size, seq_len, num_heads, _ = query.size()
            # 计算分块的数量，根据窗口重叠进行分割
            chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
            # 创建全零的对角线注意力分数张量
            diagonal_attention_scores = scores.new_zeros(
                (
                    batch_size * num_heads,
                    chunks_count + 1,
                    window_overlap,
                    window_overlap * 2 + 1,
                )
            )
            # 填充对角线注意力分数张量的部分内容
            diagonal_attention_scores[:, :-1, :, window_overlap:] = scores[
                :, :, :window_overlap, : window_overlap + 1
            ]
            # 调整对角线注意力分数张量的形状，并进行维度转置
            input_tensor = diagonal_attention_scores.view(
                batch_size, num_heads, seq_len, 2 * window_overlap + 1
            ).transpose(2, 1)
            # 获取输入张量的起始部分
            beginning_input = input_tensor[:, :window_overlap, :, : window_overlap + 1]
            # 将起始部分的内容填充为负无穷大
            input_tensor[:, :window_overlap, :, : window_overlap + 1] = torch.full_like(
                beginning_input, -float("inf")
            )
            # 返回处理后的输入张量
            return input_tensor

        # 定义测试参数
        args = [
            ((4, 1024, 12, 64), (768, 3072, 64, 1)),
            ((48, 3, 512, 513), (787968, 262656, 513, 1)),
        ]
        # 对每个参数元组应用 rand_strided 函数
        args = [rand_strided(sh, st) for (sh, st) in args]
        # 添加额外参数 256
        args.append(256)

        # 如果当前使用的是 C++ 后端
        if is_cpp_backend(self.device):
            # 优化 fn 函数并获取生成的 C++ 代码
            opt_fn = torch._dynamo.optimize("inductor")(fn)
            _, code = run_and_get_cpp_code(opt_fn, *args)
            # 打印生成的 C++ 代码
            print(code)
            # 使用 FileCheck 检查生成的 C++ 代码中 "static_cast<int32_t>(256)" 的出现次数
            FileCheck().check_count(
                "static_cast<int32_t>(256)",
                1,
                exactly=True,
            ).run(code)

        # 调用共同的测试方法，传入 fn 函数和参数 args
        self.common(fn, args)

    # 定义测试函数 test_cumsum_pattern_matcher_issue
    def test_cumsum_pattern_matcher_issue(self):
        # 定义内部函数 fn，处理输入的 token IDs，返回累积和的张量
        def fn(input_ids) -> torch.Tensor:
            # 获取输入张量的形状信息
            input_shape = input_ids.size()
            # 将输入张量视图重塑为二维张量
            input_ids = input_ids.view(-1, input_shape[-1])
            # 获取批量大小和序列长度
            batch_size, seq_length = input_shape
            # 初始化过去键值对长度为 0
            past_key_values_length = 0
            # 计算掩码的序列长度，考虑过去键值对的长度和当前序列长度
            mask_seq_length = past_key_values_length + seq_length
            # 创建全 1 的注意力掩码张量
            attention_mask = torch.ones(
                batch_size, mask_seq_length, device=input_ids.device
            )
            # 将注意力掩码张量转换为长整型
            attention_mask = attention_mask.long()
            # 返回注意力掩码张量的累积和，沿着第二个维度
            return torch.cumsum(attention_mask, dim=1)

        # 创建形状为 (2, 2) 的随机张量 x
        x = torch.randn(2, 2)
        # 调用共同的测试方法，传入 fn 函数、参数 x 和指定的误差容限
        self.common(fn, (x,), atol=0, rtol=0)

    # 定义静态方法 _check_resize_common
    @staticmethod
    def _check_resize_common(
        self, fn, x, size_or_y, memory_format, inplace, deterministic
    ):
        # 将输入张量 x 移动到设备 self.device 上
        x = x.to(self.device)
        # 克隆张量 x 作为参考参数
        x_ref_arg = x.clone()
        # 克隆张量 x 作为优化参数
        x_opt_arg = x.clone()
        # 获取张量 x 的元素总数
        x_numel = x.numel()
        # 重置 Torch 的代码缓存
        torch._dynamo.reset_code_caches()
        # 使用 Torch 动态优化器对编译后的函数进行优化
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        # 调用原始函数 fn 计算正确结果
        correct = fn(x_ref_arg, size_or_y, memory_format)
        # 调用优化后的函数 opt_fn 计算实际结果
        actual = opt_fn(x_opt_arg, size_or_y, memory_format)

        def get_numel(size_or_y):
            # 如果 size_or_y 是张量，则返回其元素总数
            if isinstance(size_or_y, torch.Tensor):
                return size_or_y.numel()
            else:
                # 假定 size_or_y 是形状，计算其元素总数
                return functools.reduce(lambda x, y: x * y, size_or_y, 1)

        if deterministic:
            # 如果是确定性模式，使用 correct 的元素总数进行检查
            nele_check = correct.numel()
        else:
            # 否则使用 x_numel 和 size_or_y 的元素总数的最小值进行检查
            nele_check = min(x_numel, get_numel(size_or_y))

        # 从 correct 和 actual 中获取 nele_check 个元素构成的视图
        correct_values = correct.as_strided((nele_check,), (1,))
        actual_values = actual.as_strided((nele_check,), (1,))
        # 使用 assertTrue 断言两个视图的内容相同，支持 NaN 相等性检查
        self.assertTrue(same(correct_values, actual_values, equal_nan=deterministic))
        # 获取 correct 和 actual 的步幅信息
        correct_strides = correct.stride()
        actual_strides = actual.stride()
        # 使用 assertEqual 断言两个张量的步幅相同
        self.assertEqual(correct_strides, actual_strides)

    @staticmethod
    def _cases_resize_common():
        # 定义一系列测试用例的大小对，用于测试大小重置操作
        sizes = [
            ((2,), (1, 3, 2, 3)),
            ((100,), (1, 3, 2, 3)),
            ((1, 3, 2, 3), (1, 3, 2, 3)),
            ((2,), (1, 3, 2, 3, 1)),
            ((100,), (1, 3, 2, 3, 1)),
            ((1, 3, 2, 3, 1), (1, 3, 2, 3, 1)),
            ((2, 0, 1), (2, 2)),
        ]
        # 遍历测试用例
        for x_size, y_size in sizes:
            # 初始化内存格式列表，根据 y_size 的维度情况添加不同的内存格式
            memory_formats = [torch.contiguous_format]
            if len(y_size) == 4:
                memory_formats.append(torch.channels_last)
            if len(y_size) == 5:
                memory_formats.append(torch.channels_last_3d)
            # 对每个内存格式和尺寸组合生成随机张量 x
            for memory_format in memory_formats:
                x = torch.randn(*x_size)
                # 使用 yield 语句返回 x、y_size 和内存格式的组合，用于测试
                yield x, y_size, memory_format
                # 对一些非连续张量进行检查
                if x.numel() == 100:
                    x_strided = x[::2].reshape(25, 2).transpose(0, 1)
                    yield x_strided, y_size, memory_format

    def test_resize(self):
        def fn(x, size, memory_format):
            # 返回使用 Torch 的 aten.resize 方法调整后的张量
            # 注意: Tensor.resize() != aten::resize()
            return torch.ops.aten.resize(x, size, memory_format=memory_format)

        # 对于确定性和非确定性两种情况分别进行测试
        for deterministic in [True, False]:
            # 使用 DeterministicGuard 进行环境管理，确保每次测试的一致性
            with DeterministicGuard(
                deterministic, fill_uninitialized_memory=deterministic
            ):
                # 遍历 CommonTemplate._cases_resize_common() 返回的测试用例
                for x, y_size, memory_format in CommonTemplate._cases_resize_common():
                    # 使用 CommonTemplate._check_resize_common 方法对 resize 操作进行检查
                    CommonTemplate._check_resize_common(
                        self,
                        fn,
                        x,
                        y_size,
                        memory_format,
                        inplace=False,
                        deterministic=deterministic,
                    )

    @staticmethod
    def _cases_resize_as_common():
        # 遍历 CommonTemplate._cases_resize_common() 返回的每个元组 (x, y_size, memory_format)
        for x, y_size, memory_format in CommonTemplate._cases_resize_common():
            # 生成器函数，返回三种不同配置的测试用例:
            # 1. y 是连续的，函数使用 memory_format 参数
            # 2. y 具有 memory_format 的连续性，并且函数使用 preserve 参数
            # 3. y 具有其他步幅（非连续或通道为最后一维），函数使用 preserve 参数
            yield x, torch.randn(*y_size), memory_format
            yield x, torch.randn(*y_size).contiguous(memory_format=memory_format), torch.preserve_format
            yield x, torch.randn(*y_size).permute(tuple(reversed(range(len(y_size))))), torch.preserve_format

    @skipIfXpu
    def test_resize_as(self):
        # 定义测试函数 fn(x, y, memory_format)，调用 torch.ops.aten.resize_as(x, y, memory_format=memory_format)
        def fn(x, y, memory_format):
            return torch.ops.aten.resize_as(x, y, memory_format=memory_format)

        # 对于 deterministic 取值为 True 和 False，进行测试
        for deterministic in [True, False]:
            # 根据 deterministic 状态进入或退出 DeterministicGuard
            with DeterministicGuard(deterministic, fill_uninitialized_memory=deterministic):
                # 遍历 CommonTemplate._cases_resize_as_common() 返回的每个元组 (x, y, memory_format)
                for x, y, memory_format in CommonTemplate._cases_resize_as_common():
                    # 调用 CommonTemplate._check_resize_common 进行测试验证
                    CommonTemplate._check_resize_common(
                        self,
                        fn,
                        x,
                        y,
                        memory_format,
                        inplace=False,
                        deterministic=deterministic,
                    )

    def test_inplace_resize_as(self):
        # 定义测试函数 fn(x, y)，使用 x.resize_as_(y) 实现
        def fn(x, y):
            x.resize_as_(y)
            return x

        # 创建随机张量 x 和 y
        x = torch.randn(2, 3)
        y = torch.randn(200, 300)
        x_clone = x.clone()
        # 优化 fn 函数，使用 torch._dynamo.optimize("inductor")
        opt_fn = torch._dynamo.optimize("inductor")(fn)
        # 断言优化后的函数 opt_fn 和原始函数 fn 在相同输入下产生相同结果
        same(fn(x, y), opt_fn(x_clone, y))

    def test_erfc(self):
        # 定义测试函数 fn(x)，调用 torch.erfc(x) 计算
        def fn(x):
            return torch.erfc(x)

        # 使用 self.common(fn, (torch.randn(8, 8),)) 运行通用测试函数
        self.common(fn, (torch.randn(8, 8),))

    @skip_if_halide  # erfinv not implemented
    def test_erfinv(self):
        # 定义测试函数 fn(x)，调用 torch.erfinv(x) 计算
        def fn(x):
            return torch.erfinv(x)

        # 创建区间为 (-1, 1) 内随机初始化的张量 x
        x = torch.empty(8, 8).uniform_(-1, 1)
        # 使用 self.common(fn, (x,)) 运行通用测试函数
        self.common(fn, (x,))

    def test_uint(self):
        # 定义测试函数 fn(z)，创建 uint8 数据类型的张量 x，并对其执行 torch.neg(x) 操作
        def fn(z):
            x = torch.tensor(5, device=z.device, dtype=torch.uint8)
            y = torch.neg(x)
            # 返回比较结果 x < y
            return x < y

        # 使用 self.common(fn, (torch.randn(26),)) 运行通用测试函数
        self.common(fn, (torch.randn(26),))
    def test_scaled_dot_product_attention(self):
        # 如果设备是 "cuda" 并且不支持 FLASH_ATTENTION，则跳过测试
        if self.device == "cuda" and not PLATFORM_SUPPORTS_FLASH_ATTENTION:
            raise unittest.SkipTest("Can't run flash attention on this platform")
        # 如果设备是 "cuda" 并且 TEST_WITH_ROCM 为真，则跳过测试
        if self.device == "cuda" and TEST_WITH_ROCM:
            raise unittest.SkipTest(
                "Flash attention support is incomplete on this platform"
            )

        # 定义一个函数 fn，使用 torch.nn.functional.scaled_dot_product_attention 计算注意力
        def fn(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2).contiguous(),  # 转置 q 张量并保证连续性
                k.transpose(1, 2),  # 转置 k 张量
                v.transpose(1, 2),  # 转置 v 张量
                scale=0.125,  # 缩放参数为 0.125
            )[:2]  # 返回前两个元素

        # 调用 self.common 方法来测试 fn 函数的输出
        self.common(
            fn,
            (
                torch.randn(4, 2, 4, 2),  # 随机生成大小为 (4, 2, 4, 2) 的张量 q
                torch.randn(4, 2, 4, 2),  # 随机生成大小为 (4, 2, 4, 2) 的张量 k
                torch.randn(4, 2, 4, 2),  # 随机生成大小为 (4, 2, 4, 2) 的张量 v
            ),
            atol=2e-4,  # GPU 上通过低精度检查所需的绝对误差容忍度
            rtol=1e-2,  # GPU 上通过低精度检查所需的相对误差容忍度
        )

    @skipIfRocm
    @expectedFailureXPU
    def test_scaled_dot_product_efficient_attention(self):
        # 如果设备是 "cpu"，则跳过测试，并显示需要 GPU_TYPE
        if self.device == "cpu":
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        # 定义一个函数 fn，使用 aten._scaled_dot_product_efficient_attention 计算注意力
        # 返回前两个输出，即注意力输出和 logsumexp（因为没有设置 dropout）
        def fn(q, k, v, attn_bias, compute_log_sumexp):
            return aten._scaled_dot_product_efficient_attention(
                q, k, v, attn_bias, compute_log_sumexp
            )[:2]

        # 调用 self.common 方法来测试 fn 函数的输出
        self.common(
            fn,
            (
                torch.randn(4, 4, 36, 36),  # 随机生成大小为 (4, 4, 36, 36) 的张量 q
                torch.randn(4, 4, 36, 36),  # 随机生成大小为 (4, 4, 36, 36) 的张量 k
                torch.randn(4, 4, 36, 36),  # 随机生成大小为 (4, 4, 36, 36) 的张量 v
                torch.randn(4, 4, 36, 36),  # 随机生成大小为 (4, 4, 36, 36) 的张量 attn_bias
                False,  # compute_log_sumexp 设为 False
            ),
            check_lowp=False,  # 不进行低精度检查
        )

    def test_fft_real_input(self):
        # 定义一个函数 fn，使用 torch.fft.fftn 计算实输入的快速傅里叶变换
        def fn(x):
            return torch.fft.fftn(x)

        # 调用 self.common 方法来测试 fn 函数的输出
        self.common(fn, (torch.randn((16, 16, 16)),), check_lowp=False)

    def test_fft_real_input_real_output(self):
        # 定义一个函数 fn，使用 torch.fft.fftn 计算实输入的快速傅里叶变换，并取实部
        def fn(x):
            return torch.fft.fftn(x).real

        # 调用 self.common 方法来测试 fn 函数的输出
        self.common(fn, (torch.randn((16, 16, 16)),), check_lowp=False)

    def test_bucketize(self):
        # 定义一个函数 fn，使用 torch.bucketize 实现数据桶化
        def fn(input, boundaries, out_int32, right):
            return torch.bucketize(input, boundaries, out_int32=out_int32, right=right)

        # 创建随机输入张量 input 和边界张量 boundaries
        input = torch.rand((64, 64)) * 2 - 1
        boundaries = torch.tensor([-0.9, -0.8, 0.1, 0.2, 0.5, 0.9])

        # 在 out_int32 和 right 参数的所有组合下，调用 self.common 方法来测试 fn 函数的输出
        for out_int32 in [True, False]:
            for right in [True, False]:
                out_int32 = True  # 强制 out_int32 为 True
                right = False  # 强制 right 为 False
                self.common(fn, (input, boundaries, out_int32, right), check_lowp=False)
    # 定义一个测试用例，验证默认参数下的 torch.bucketize 函数的行为
    def test_bucketize_default_kwargs(self):
        # 定义一个内部函数 fn，用于调用 torch.bucketize 对输入 input 进行分桶操作
        def fn(input, offsets):
            return torch.bucketize(input, offsets)

        # 创建一个包含浮点数的 PyTorch 张量作为输入
        input = torch.tensor(
            [-1.0, -0.9, -0.8, -0.5, 0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.9, 0.91]
        )
        # 创建一个包含偏移值的 PyTorch 张量作为分桶的边界
        offsets = torch.tensor([-0.9, -0.8, 0.1, 0.2, 0.5, 0.9])

        # 调用通用的测试函数 common，验证 fn 的行为，关闭低精度检查
        self.common(fn, (input, offsets), check_lowp=False)

    # 定义一个测试用例，验证 torch.bucketize 在处理整数输入时的行为
    def test_bucketize_int(self):
        # 定义一个内部函数 fn，用于调用 torch.bucketize 对输入 input 进行分桶操作
        def fn(input, offsets, out_int32, right):
            return torch.bucketize(input, offsets, out_int32=out_int32, right=right)

        # 创建一个随机整数填充的 PyTorch 张量作为输入
        input = torch.randint(0, 102, (64, 64))
        # 创建一个整数类型的 PyTorch 张量作为分桶的边界
        offsets = torch.arange(10, dtype=torch.int32) ** 2 + 1

        # 遍历不同的输出选项和右边界选项
        for out_int32 in [True, False]:
            for right in [True, False]:
                # 调用通用的测试函数 common，验证 fn 的行为，关闭低精度检查
                self.common(fn, (input, offsets, out_int32, right), check_lowp=False)

    # 在 triton.autotune_pointwise 为 True 时，添加自动调整点运算的测试用例
    @patch.object(config.triton, "autotune_pointwise", True)
    def test_bucketize_add_autotune(self):
        # 定义一个内部函数 fn，用于调用 torch.bucketize 对输入 input 进行分桶操作后，再加上 add_value
        def fn(input, offsets, add_value):
            return torch.bucketize(input, offsets) + add_value

        # 创建一个随机填充的四维 PyTorch 张量作为输入
        input = torch.rand((16, 16, 64, 64))
        # 创建一个浮点数类型的 PyTorch 张量作为分桶的边界
        boundaries = torch.tensor([-0.9, -0.8, 0.1, 0.2, 0.5, 0.9])
        # 创建一个随机整数填充并采用通道优先内存格式的 PyTorch 张量作为加值
        add_value = torch.randint(0, 1024, (16, 16, 64, 64)).to(
            memory_format=torch.channels_last
        )

        # 调用通用的测试函数 common，验证 fn 的行为，关闭低精度检查
        self.common(fn, (input, boundaries, add_value), check_lowp=False)

        # 断言生成的内核数量等于 1
        assertGeneratedKernelCountEqual(self, 1)

    # 定义一个测试用例，验证 torch.bucketize 在处理计算后的偏移值时的行为
    def test_bucketize_computed_offsets(self):
        # 定义一个内部函数 fn，用于调用 torch.bucketize 对输入 inp 进行分桶操作
        def fn(inp, offsets):
            return torch.bucketize(inp, offsets + 0.01)

        # 创建一个包含浮点数的 PyTorch 张量作为输入
        inp = torch.tensor(
            [-1.0, -0.9, -0.8, -0.5, 0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.9, 0.91]
        )
        # 创建一个包含计算后偏移值的 PyTorch 张量作为分桶的边界
        offsets = torch.tensor([-0.9, -0.8, 0.1, 0.2, 0.5, 0.9]) - 0.01

        # 调用通用的测试函数 common，验证 fn 的行为，关闭低精度检查
        self.common(fn, (inp, offsets), check_lowp=False)

    # 需要 GPU 的配置选项，测试在不假设对齐的情况下的行为
    @requires_gpu()
    @config.patch(assume_aligned_inputs=False)
    def test_config_option_dont_assume_alignment(self):
        # 定义一个函数 fn，对输入 x 进行 sin 和 cos 操作后返回
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x.sin() + x.cos()

        # 对不同的偏移值进行测试，确保在不同配置下不会出现问题
        for offset in (0, 1, 2, 3, 4):
            # 创建一个随机填充的 PyTorch 张量作为基础数据，设备为 self.device
            base = torch.randn(64 * 64 + 64, dtype=torch.float32, device=self.device)
            # 创建一个以 stride 方式生成的视图，作为输入数据
            inp = torch.as_strided(base, (64, 64), (64, 1), offset)
            # 重置动态编译器状态
            torch._dynamo.reset()
            # 编译函数 fn
            fn_c = torch.compile(fn)

            # 计算基准结果和编译函数结果
            ref = fn(inp)
            res = fn_c(inp)
            # 断言基准结果与编译函数结果相等
            self.assertEqual(ref, res)

            # 对不同的偏移值进行进一步测试，设置绝对误差和相对误差的公差为 1e-5
            for offset2 in (0, 1, 2, 3, 4):
                # 创建另一个随机填充的 PyTorch 张量作为基础数据，设备为 self.device
                base2 = torch.randn(
                    64 * 64 + 64, dtype=torch.float32, device=self.device
                )
                # 创建另一个以 stride 方式生成的视图，作为输入数据
                inp2 = torch.as_strided(base2, (64, 64), (64, 1), offset2)
                # 计算基准结果和编译函数结果
                ref2 = fn(inp2)
                res2 = fn_c(inp2)
                # 断言基准结果与编译函数结果相等，设置绝对误差和相对误差的公差为 1e-5
                self.assertEqual(ref2, res2, atol=1e-5, rtol=1e-5)
    @config.patch(assume_aligned_inputs=False)
    # 使用 config 模块的 patch 方法，设置 assume_aligned_inputs 参数为 False
    def test_config_option_dont_assume_alignment_recompiles(self):
        # 测试函数，验证不假设输入对齐时的重新编译行为

        # 失败的守卫列表，用于记录失败的守卫
        failed_guards = []

        def fail(guard):
            nonlocal failed_guards
            failed_guards.append(guard)

        def fn(x: torch.Tensor) -> torch.Tensor:
            # 定义一个函数 fn，对输入张量 x 执行 sin 和 cos 操作后返回
            return x.sin() + x.cos()

        # 创建一个形状为 (64*64 + 64) 的随机张量 base，在 self.device 上
        base = torch.randn(64 * 64 + 64, dtype=torch.float32, device=self.device)

        # 使用 torch.as_strided 方法创建三个不同的输入张量
        # inp1: 形状为 (32, 32)，步幅为 (32, 1)，偏移为 4
        inp1 = torch.as_strided(base, (32, 32), (32, 1), 4)
        # inp2: 形状为 (64, 64)，步幅为 (64, 1)，偏移为 4
        inp2 = torch.as_strided(base, (64, 64), (64, 1), 4)
        # inp3: 形状为 (64, 64)，步幅为 (64, 1)，偏移为 5
        inp3 = torch.as_strided(base, (64, 64), (64, 1), 5)

        # 重置 torch._dynamo 的状态
        torch._dynamo.reset()

        # 使用 torch._dynamo.optimize 方法对函数 fn 进行优化，设置 guard_fail_fn 为 fail
        fn_c = torch._dynamo.optimize("inductor", guard_fail_fn=fail)(fn)

        # 对 inp1 进行函数 fn 和 fn_c 的计算，并比较结果是否相等
        ref1 = fn(inp1)
        res1 = fn_c(inp1)
        self.assertEqual(ref1, res1)

        # 验证 failed_guards 列表长度为 0，表示没有守卫失败
        self.assertEqual(0, len(failed_guards))

        # 对 inp2 进行函数 fn 和 fn_c 的计算，并比较结果是否相等
        ref2 = fn(inp2)
        res2 = fn_c(inp2)
        self.assertEqual(ref2, res2)

        # 如果 dynamic shapes 没有开启，可能会有守卫失败，因为我们正在开启 dynamic shapes
        # 验证 failed_guards 列表长度最多为 1
        self.assertLessEqual(len(failed_guards), 1)
        failed_guard_count_iteration_2 = len(failed_guards)

        # 重置 failed_guards 列表
        failed_guards = []

        # 对 inp3 进行函数 fn 和 fn_c 的计算，并比较结果是否相等
        ref3 = fn(inp3)
        res3 = fn_c(inp3)
        self.assertEqual(ref3, res3)

        # 我们可能仍然会遇到动态形状失败，但偏移改变不应该受到守卫限制
        # 参见注释：[Input Alignment handling in Inductor]
        self.assertLessEqual(len(failed_guards), failed_guard_count_iteration_2)

    @requires_gpu()
    @config.patch(assume_aligned_inputs=False)
    # 使用 config 模块的 patch 方法，设置 assume_aligned_inputs 参数为 False
    def test_config_option_dont_assume_alignment_cudagraphs(self):
        # 测试函数，验证不假设输入对齐时 cudagraph 的行为

        def fn(x):
            # 定义一个函数 fn，对输入 x 执行 cos 和 sin 操作后返回
            return x.cos() * x.sin()

        # 使用 torch.compile 方法编译函数 fn，模式为 "reduce-overhead"，启用动态模式
        fn_c = torch.compile(fn, mode="reduce-overhead", dynamic=True)

        # 迭代不同的 size、stride 和 offset 组合
        for size, stride, offset in (
            ((32, 32), (32, 1), 4),
            ((48, 48), (48, 1), 4),
            ((64, 64), (64, 1), 5),
        ):
            # 使用相同的种子生成随机数
            torch.manual_seed(42)
            base = torch.randn(64 * 64 + 64, dtype=torch.float32, device=self.device)
            torch.manual_seed(42)
            base_ref = torch.randn(64 * 64 + 64, dtype=torch.float32, device=self.device)

            # 使用 torch.as_strided 方法创建输入张量 inp 和参考张量 inp_ref
            inp = torch.as_strided(base, size, stride, offset)
            inp_ref = torch.as_strided(base_ref, size, stride, offset)

            # 将 inp 和 inp_ref 设置为需要梯度计算
            inp.requires_grad_(True)
            inp_ref.requires_grad_(True)

            # 计算 fn_c(inp) 和 fn(inp_ref) 的结果，并比较是否相等
            res = fn_c(inp)
            ref = fn(inp_ref)
            self.assertEqual(ref, res)

            # 对结果求和并进行反向传播
            res.sum().backward()
            ref.sum().backward()

            # 验证 base 和 base_ref 的梯度是否相等
            self.assertEqual(base.grad, base_ref.grad)

    @config.patch(implicit_fallbacks=True)
    # 使用 config 模块的 patch 方法，设置 implicit_fallbacks 参数为 True
    @config.patch(implicit_fallbacks=True)
    # 使用配置修补程序，启用隐式回退机制
    def test_custom_op_1(self):
        # 导入 torch 库的子模块 library
        import torch.library

        # 定义在 CPU 上运行的函数 foo_cpu，返回输入的三倍
        def foo_cpu(x):
            return 3 * x

        # 定义在 CUDA 上运行的函数 foo_cuda，返回输入的三倍
        def foo_cuda(x):
            return 3 * x

        # 定义在 XPU 上运行的函数 foo_xpu，返回输入的三倍
        def foo_xpu(x):
            return 3 * x

        # 定义元函数 foo_meta，创建一个与输入相同形状的空张量
        def foo_meta(x):
            return torch.empty_like(x)

        # 为测试目的定义自定义操作 "foo"，分别针对不同设备注册上述函数
        define_custom_op_for_test("foo", foo_cpu, foo_cuda, foo_xpu, foo_meta)

        # 定义测试函数 fn，对输入张量执行 ReLU 激活函数，然后调用自定义操作 "foo"，
        # 最后计算余弦函数并返回结果
        def fn(x):
            a = torch.nn.functional.relu(x)
            b = torch.ops.test.foo(a)
            c = torch.cos(b)
            return c

        # 调用公共测试方法，传入 fn 函数及其参数，关闭低精度检查
        self.common(fn, (torch.randn((16, 32)),), check_lowp=False)

    @config.patch(implicit_fallbacks=True)
    # 使用配置修补程序，启用隐式回退机制
    def test_custom_op_2(self):
        # 导入 torch 库的子模块 library
        import torch.library

        # 定义在 CPU 上运行的函数 foo_cpu，返回输入的按指定比例缩放后的值及其余弦值
        def foo_cpu(x, scale: float):
            return scale * x, torch.cos(x)

        # 定义在 CUDA 上运行的函数 foo_cuda，返回输入的按指定比例缩放后的值及其余弦值
        def foo_cuda(x, scale: float):
            return scale * x, torch.cos(x)

        # 定义在 XPU 上运行的函数 foo_xpu，返回输入的按指定比例缩放后的值及其余弦值
        def foo_xpu(x, scale: float):
            return scale * x, torch.cos(x)

        # 定义元函数 foo_meta，创建与输入张量相同形状的空张量，并返回其余弦值
        def foo_meta(x, scale: float):
            return torch.empty_like(x), torch.empty_like(x)

        # 为测试目的定义自定义操作 "foo2"，分别针对不同设备注册上述函数
        define_custom_op_2_for_test("foo2", foo_cpu, foo_cuda, foo_xpu, foo_meta)

        # 定义测试函数 fn，对输入张量执行 ReLU 激活函数，然后调用自定义操作 "foo2"，
        # 并传入额外的缩放因子参数，返回结果
        def fn(x, scale: float):
            a = torch.nn.functional.relu(x)
            return torch.ops.test.foo2(a, scale)

        # 调用公共测试方法，传入 fn 函数及其参数，关闭低精度检查
        self.common(fn, (torch.randn((16, 32)), 2.0), check_lowp=False)

    @config.patch(implicit_fallbacks=True)
    # 使用配置修补程序，启用隐式回退机制
    def test_custom_op_3(self):
        # 导入 torch 库的子模块 library
        import torch.library

        # 定义在 CPU 上运行的函数 foo_cpu，对输入列表中的张量进行求和并返回
        def foo_cpu(x):
            result = torch.zeros_like(x[0])
            for t in x:
                result += t
            return result

        # 定义在 CUDA 上运行的函数 foo_cuda，对输入列表中的张量进行求和并返回
        def foo_cuda(x):
            result = torch.zeros_like(x[0])
            for t in x:
                result += t
            return result

        # 定义在 XPU 上运行的函数 foo_xpu，对输入列表中的张量进行求和并返回
        def foo_xpu(x):
            result = torch.zeros_like(x[0])
            for t in x:
                result += t
            return result

        # 定义元函数 foo_meta，创建与输入列表中的第一个张量相同形状的空张量，并返回
        def foo_meta(x):
            return torch.empty_like(x[0])

        # 为测试目的定义自定义操作 "foo3"，分别针对不同设备注册上述函数
        define_custom_op_3_for_test("foo3", foo_cpu, foo_cuda, foo_xpu, foo_meta)

        # 定义测试函数 fn，对输入列表中的张量执行自定义操作 "foo3"，返回结果
        def fn(x):
            return torch.ops.test.foo3(x)

        # 调用公共测试方法，传入 fn 函数及其参数，关闭低精度检查
        self.common(
            fn,
            ([torch.randn((16, 32)), torch.randn((16, 32)), torch.randn((16, 32))],),
            check_lowp=False,
        )

    @requires_gpu()
    # 要求 GPU 环境下运行该测试
    @torch._inductor.config.patch("layout_optimization", True)
    # 使用布局优化的配置修补程序
    @torch._inductor.config.patch("keep_output_stride", False)
    # 使用保持输出步幅的配置修补程序
    @config.patch(implicit_fallbacks=True)
    # 使用配置修补程序，启用隐式回退机制
    def test_custom_op_fixed_layout_sequential(self):
        # 导入torch库
        import torch.library

        # 创建一个包含128个输出通道的1x1卷积层，无偏置项，放置在GPU上
        mod = nn.Conv2d(3, 128, 1, stride=1, bias=False).to(device=GPU_TYPE)
        # 创建一个形状为(2, 3, 128, 128)的随机张量，放置在GPU上
        inp = torch.rand(2, 3, 128, 128, device=GPU_TYPE)
        # 获取模型输出的预期步长
        expected_stride = mod(inp).stride()

        # 定义用于CPU的自定义操作
        def bar_cpu(x):
            self.assertEqual(x.stride(), expected_stride)
            return x.clone()

        # 定义用于CUDA的自定义操作
        def bar_cuda(x):
            self.assertEqual(x.stride(), expected_stride)
            return x.clone()

        # 定义用于XPU的自定义操作
        def bar_xpu(x):
            self.assertEqual(x.stride(), expected_stride)
            return x.clone()

        # 定义用于meta的自定义操作
        def bar_meta(x):
            return torch.empty_like(x)

        # 为测试定义一个名为"bar"的自定义操作，需要固定步长顺序的标签
        define_custom_op_for_test(
            "bar",
            bar_cpu,
            bar_cuda,
            bar_xpu,
            bar_meta,
            tags=[torch._C.Tag.needs_fixed_stride_order],
        )

        # 定义一个函数fn，对输入进行模型处理，然后调用自定义操作"bar"
        def fn(x):
            z = mod(x)
            output = torch.ops.test.bar(z)
            return output

        # 在无梯度更新的上下文中运行
        with torch.no_grad():
            # 使用keep_output_stride为False，正常情况下，输出会与即时执行的布局不同
            # 但因为我们的自定义操作需要固定布局，自定义操作中的断言将通过
            self.common(fn, (inp,), check_lowp=False)

    @requires_gpu()
    @config.patch(implicit_fallbacks=True)
    def test_custom_op_fixed_layout_channels_last(self):
        # 定义一个Block类，继承自nn.Module
        class Block(nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

                # 定义一个包含Dropout层的顺序结构
                self.in_layers = nn.Sequential(
                    nn.Dropout(p=0.1),
                )

            # 辅助方法helper，对输入应用gelu激活函数和in_layers序列
            def helper(self, x):
                out = F.gelu(x)
                out = self.in_layers(out)
                return out

            # 前向传播方法，对输入应用helper方法和自定义操作"baz"
            def forward(self, x):
                out = self.helper(x)
                out = torch.ops.test.baz(out)
                return out

        # 创建一个Block实例model，并将其放置在GPU上，使用通道为最后一维的内存格式
        model = Block()
        model = model.to(GPU_TYPE).to(memory_format=torch.channels_last)
        # 创建一个形状为[1, 320, 128, 128]的随机张量input_t，放置在GPU上，使用通道为最后一维的内存格式
        input_t = torch.randn([1, 320, 128, 128], dtype=torch.float32, device=GPU_TYPE)
        input_t = input_t.to(memory_format=torch.channels_last)
        # 获取模型辅助方法输出的预期步长
        expected_strides = model.helper(input_t).stride()

        # 定义用于CPU的自定义操作
        def baz_cpu(x):
            self.assertEqual(expected_strides, x.stride())
            return x.clone()

        # 定义用于CUDA的自定义操作
        def baz_cuda(x):
            self.assertEqual(expected_strides, x.stride())
            return x.clone()

        # 定义用于XPU的自定义操作
        def baz_xpu(x):
            self.assertEqual(expected_strides, x.stride())
            return x.clone()

        # 定义用于meta的自定义操作
        def baz_meta(x):
            return torch.empty_like(x)

        # 为测试定义一个名为"baz"的自定义操作，需要固定步长顺序的标签
        define_custom_op_for_test(
            "baz",
            baz_cpu,
            baz_cuda,
            baz_xpu,
            baz_meta,
            tags=[torch._C.Tag.needs_fixed_stride_order],
        )

        # 在无梯度更新的上下文中运行
        with torch.no_grad():
            # 编译模型net，并使用input_t作为输入
            net = torch.compile(model)
            out = net(input_t)

    @skip_if_gpu_halide  # cuda error
    def test_buffer_use_after_remove(self):
        # https://github.com/pytorch/pytorch/issues/102857
        # 定义一个测试方法，用于验证在移除后使用缓冲区的情况

        def rotvec_to_rotmat(rotvec) -> torch.Tensor:
            """Simplified rotvec to rotmat code from RoMa
            (https://github.com/naver/roma/blob/06e4b0cdc1c802a60a012bb19c581d6600c63358/roma/mappings.py#L371)
            """
            # 将旋转向量转换为旋转矩阵的简化代码，参考自 RoMa 项目
            theta = torch.norm(rotvec, dim=-1)
            axis = rotvec / theta[..., None]
            kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            one_minus_cos_theta = 1 - cos_theta
            xs = kx * sin_theta
            ys = ky * sin_theta
            zs = kz * sin_theta
            xyc = kx * ky * one_minus_cos_theta
            xzc = kx * kz * one_minus_cos_theta
            yzc = ky * kz * one_minus_cos_theta
            xxc = kx**2 * one_minus_cos_theta
            yyc = ky**2 * one_minus_cos_theta
            zzc = kz**2 * one_minus_cos_theta
            # 根据罗德里格斯公式计算旋转矩阵
            R_rodrigues = torch.stack(
                [
                    1 - yyc - zzc,
                    xyc - zs,
                    xzc + ys,
                    xyc + zs,
                    1 - xxc - zzc,
                    -xs + yzc,
                    xzc - ys,
                    xs + yzc,
                    1 - xxc - yyc,
                ],
                dim=-1,
            ).reshape(-1, 3, 3)
            R = R_rodrigues
            return R

        def f(coord, rot, trans):
            # 根据旋转向量和平移向量执行仿射变换
            rot_mat = rotvec_to_rotmat(rot)
            coord = torch.einsum("...ij,...bj->...bi", rot_mat, coord) + trans
            return coord.sum()

        # 使用动态编译功能编译函数 f
        foo_c = torch.compile(f, dynamic=True)

        def run(fn):
            # 初始化坐标，旋转向量和平移向量
            coord = torch.ones((2, 3), device=self.device)
            rot = nn.Parameter(torch.ones((2, 3), device=self.device))
            trans = nn.Parameter(torch.ones((2, 3), device=self.device))

            # 运行函数 fn，并执行反向传播
            U = fn(coord, rot, trans)
            U.backward()

            return U, rot, trans

        # 分别运行原始函数 f 和编译后的函数 foo_c
        U_e, rot_e, trans_e = run(f)
        U, rot, trans = run(foo_c)

        # 断言两次运行的结果和梯度是否一致
        self.assertEqual(U, U_e)
        self.assertEqual(rot.grad, rot_e.grad)
        self.assertEqual(trans.grad, trans_e.grad)

    # If we serve from the cache, the init hook isn't called
    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    # 如果从缓存中提供服务，则不调用初始化钩子
    def test_inner_fn_str_and_stride(self):
        # 定义内部函数 f，对输入 x 进行多次操作并返回结果
        def f(x):
            # x 增加 1
            x = x + 1
            # 使用 test_operators.realize 函数处理 x
            x = test_operators.realize(x)
            # x 乘以 2
            x = x * 2
            # 再次使用 test_operators.realize 函数处理 x
            x = test_operators.realize(x)
            return x

        # 生成一个随机的 Tensor x，然后对其进行转置操作
        x = torch.rand(3, 2, device=self.device).t()
        # 调用函数 f 处理 x，将结果保存在 ref 中
        ref = f(x)
        # 初始化一个标志位 called，用于检测 hook_fn 是否被调用
        called = False

        # 定义一个 hook 函数 hook_fn，用于接收 scheduler 和 nodes 作为参数
        def hook_fn(scheduler, nodes):
            nonlocal called
            called = True

            # 如果设备不是 CPU，则进行以下断言和验证
            if self.device != "cpu":
                # 断言 nodes 的长度为 3
                self.assertEqual(len(nodes), 3)
                # 解构 nodes 得到三个元素
                _, mul_buf, _ = nodes
                # 断言所有 buf 的步长满足条件
                self.assertTrue(
                    all(
                        V.graph.sizevars.size_hints(buf.get_stride()) == (1, 2)
                        for buf in nodes
                    )
                )
                # 在修复之前，错误的索引表达式 'i1 + 3 * i0' 被缓存了。
                self.assertTrue(
                    "i0 + 2 * i1" in mul_buf.data.inner_fn_str()
                    or "i0 + i1 * s1" in mul_buf.data.inner_fn_str()
                )

        # 使用 add_scheduler_init_hook 函数注册 hook_fn
        with add_scheduler_init_hook(hook_fn):
            # 对函数 f 进行编译，并传入参数 x，将结果保存在 actual 中
            actual = torch.compile(f, fullgraph=True)(x)
        # 断言 ref 和 actual 相等
        self.assertEqual(ref, actual)
        # 断言 called 被设置为 True
        self.assertTrue(called)

    @skip_if_gpu_halide  # 如果是 GPU Halide 则跳过该测试
    def test_mutations_loop_fusion(self):
        # 定义函数 fn，接收 tensor, index, source 三个参数
        def fn(tensor, index, source):
            # 在 tensor 上执行 index_add 操作，然后除以 2，将结果保存在 out 中
            out = tensor.index_add(0, index, source, alpha=2.0) / 2
            return out

        # 设备类型为 CPU
        device = "cpu"
        # 生成一个随机的双精度 Tensor tensor，长度为 1，设备为 CPU
        tensor = torch.rand((1,), dtype=torch.double, device=device)
        # 生成一个长为 1 的长整型 Tensor index，设备为 CPU
        index = torch.tensor([0], dtype=torch.long, device=device)
        # 生成一个随机的双精度 Tensor source，长度为 1，设备为 CPU
        source = torch.rand((1,), dtype=torch.double, device=device)
        # 调用 self.common 函数，传入 fn 和对应的参数
        self.common(
            fn,
            (
                tensor,
                index,
                source,
            ),
        )

    @config.patch(
        "triton.autotune_pointwise", True
    )  # 需要引入配置来超过最大共享内存使用量
    @serialTest()
    def test_large_block_sizes(self):
        """
        Inductor 将尝试 triton 配置，如 x = 64 和 y = 1024，如果 dtype 是 fp32，
        将导致超出共享内存。

        目前 Inductor 将跳过这些不良配置，并从剩余的配置中选择最佳配置。
        """
        # 如果设备上的内存不足以容纳所需的内存量，则跳过测试
        if not _has_sufficient_memory(self.device, 3 * 2**24 * 65 * 4):
            raise unittest.SkipTest("insufficient memory")

        # 定义编译函数 fn，接收 x 和 y 两个参数
        @torch.compile
        def fn(x, y):
            # 对 x 进行转置操作，然后加上 y，返回结果
            return x.t() + y

        # 使用 shape (2**24, 65) 而不是 (2**24, 128)，可以避免 CI 中的内存溢出，
        # 同时保持相同的上舍入大小提示。
        # 生成一个在 self.device 上的随机 Tensor a，形状为 (2**24, 65)
        a = torch.randn(2**24, 65, device=self.device)
        # 生成一个在 self.device 上的随机 Tensor b，形状为 (65, 2**24)
        b = torch.randn(65, 2**24, device=self.device)
        # 调用 fn 函数，传入 a 和 b 作为参数
        fn(a, b)

    # 在 ROCm 上跳过，直到 https://github.com/ROCm/triton/issues/443 解决
    def test_fuse_large_params(self):
        # 测试函数：测试大量参数融合

        def pt2_optimizer_step(optimizer):
            # 定义局部函数：执行优化器步骤
            @torch.compile()
            def f():
                optimizer.step()

            f()

        params = [
            torch.rand(10, 10, dtype=torch.float32, device=self.device)
            for _ in range(194)
        ]
        for p in params:
            p.grad = torch.rand_like(p)

        o = torch.optim.AdamW(params)
        pt2_optimizer_step(o)

    # https://github.com/halide/Halide/issues/8256
    @config.patch("halide.scheduler_cuda", "Li2018")
    def test_adaptive_avg_pool1d_argmax(self):
        # 测试函数：测试自适应平均池化1D与最大值索引

        # https://github.com/pytorch/pytorch/issues/113013
        def fn(x):
            # 定义函数 fn：接受输入 x，进行自适应平均池化1D并计算最大值索引
            x = torch.adaptive_avg_pool1d(input=x, output_size=2)
            x = torch.argmax(input=x)
            return x

        x = torch.rand([4, 4, 3], dtype=torch.float64)
        self.common(fn, (x,))

    def test_float16_to_int16(self):
        # 测试函数：测试浮点16位转整数16位

        def fn(x):
            # 定义函数 fn：将输入 x 视图转换为整数16位类型并乘以2
            x_view = x.view(dtype=torch.int16)
            return x_view.mul(2)

        x = torch.ones(4, dtype=torch.float16, device=self.device)
        ref = fn(x)
        actual = torch.compile(fn)(x)
        self.assertEqual(ref, actual)

    @skipCUDAIf(not SM80OrLater, "uses bfloat16 which requires SM >= 80")
    @skip_if_gpu_halide  # https://github.com/halide/Halide/issues/8311
    def test_bfloat16_to_int16(self):
        # 测试函数：测试bfloat16转整数16位

        def fn(a, b):
            # 定义函数 fn：将输入 a 和 b 相加，将结果转换为整数16位类型并乘以2
            x = a + b
            x_view = x.view(dtype=torch.int16)
            return x_view.mul(2)

        a = torch.ones(4, dtype=torch.bfloat16, device=self.device)
        b = torch.ones(4, dtype=torch.bfloat16, device=self.device)
        ref = fn(a, b)
        actual = torch.compile(fn)(a, b)
        self.assertEqual(ref, actual)

    def test_float32_to_int32(self):
        # 测试函数：测试浮点32位转整数32位

        def fn(a, b):
            # 定义函数 fn：将输入 a 和 b 相加，将结果转换为整数32位类型并乘以2
            x = a + b
            x_view = x.view(dtype=torch.int32)
            return x_view.mul(2)

        a = torch.ones(4, dtype=torch.float32, device=self.device)
        b = torch.ones(4, dtype=torch.float32, device=self.device)
        ref = fn(a, b)
        actual = torch.compile(fn)(a, b)
        self.assertEqual(ref, actual)

    def test_randint_int64_mod(self):
        # 测试函数：测试随机整数64位取模

        # This used to not compile due to a wrong return type of randint64_cpu
        # See https://github.com/pytorch/pytorch/issues/117435
        def fn(n):
            return (
                torch.randint(
                    low=-5, high=5, size=(n,), dtype=torch.int64, device=self.device
                )
                % 10
            )

        res = torch.compile(fn)(20)
        self.assertTrue(torch.all((0 <= res) & (res < 10)).item())

    @torch._inductor.config.patch(force_shape_pad=True)
    @skip_if_gpu_halide  # correctness issue
    # 测试函数：检查是否需要在执行矩阵乘法时进行填充
    def test_should_pad_bench_for_bmm(self):
        B = 2  # 定义批次大小为2
        M = 1024  # 第一个矩阵的行数为1024
        N = 1024  # 第二个矩阵的列数为1024
        K = 1024 + 1  # 第一个矩阵的列数，需要进行填充

        # 创建随机张量作为第一个矩阵和第二个矩阵，并指定设备
        mat1 = torch.rand(B, M, K, device=self.device)
        mat2 = torch.rand(B, K, N, device=self.device)

        # 调用函数判断是否需要填充
        should_pad = pad_mm.should_pad_bench(None, mat1, mat2, torch.ops.aten.bmm)

        # 断言应该进行填充
        self.assertTrue(should_pad)

    @parametrize(
        "name, op",
        [
            subtest((name, getattr(torch.special, name)), name=name)
            for name in torch.special.__all__  # 遍历所有特殊函数的名称
            if name not in {"softmax", "log_softmax", "logsumexp"}  # 排除指定的函数名
        ],
    )
    # 预期的动态代码生成测试失败，因为在动态形状测试中没有动态的for循环
    @expectedFailureCodegenDynamic
    def test_view_uint8_through_differing_bitwidths(self):
        # 定义一个函数，将输入张量转换为指定的数据类型，再转换为uint8类型
        def fn(x, view_dtype):
            return x.view(view_dtype).view(torch.uint8)

        view_dtypes = [torch.int16, torch.int32, torch.int64]  # 定义不同的数据类型列表
        for dtype in view_dtypes:
            x = torch.randint(0, 2**4, [4096, 4096], dtype=torch.uint8)  # 创建随机uint8张量
            self.common(
                fn,
                (
                    x,
                    dtype,
                ),
            )

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_split_with_sizes_with_unbacked_symints(self):
        # 定义一个使用torch.compile装饰器的测试函数，该函数调用了split_with_sizes操作符
        @torch.compile()
        def f(sz, x):
            s0, s1 = sz.tolist()  # 将输入大小张量转换为列表
            r0, r1 = torch.ops.aten.split_with_sizes.default(x, [s0, s1])  # 调用split_with_sizes操作符
            return torch.ops.aten.sort.default(r1)  # 对结果进行排序操作

        N = 7312  # 定义张量的长度为7312
        S0 = 420  # 第一个分割大小为420
        S1 = N - S0  # 第二个分割大小为(N - S0)

        result = f(torch.tensor([S0, S1]), torch.randn(N))  # 调用函数f并传入参数
        self.assertTrue(len(result) == 2)  # 断言结果长度为2

        @torch.compile()
        def f2(x):
            y = torch.arange(x.item())  # 根据输入值创建一个张量
            return torch.ops.aten.split_with_sizes.default(y, [5, 5, 10])  # 调用split_with_sizes操作符

        result = f2(torch.tensor([20]))  # 调用函数f2并传入参数
        self.assertTrue(len(result) == 3)  # 断言结果长度为3

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_split_with_unbacked_symints(self):
        # https://github.com/pytorch/pytorch/issues/122937
        @torch.compile()
        def f(x):
            y = torch.arange(x.item())  # 根据输入值创建一个张量
            return torch.split(y, [5, 5, 10])  # 调用split操作符进行分割

        result = f(torch.tensor([20]))  # 调用函数f并传入参数
        self.assertTrue(len(result) == 3)  # 断言结果长度为3

    # 测试函数：检查是否存在复杂的内存重叠
    def test_complex_memory_overlap(self):
        t = rand_strided((8, 1500, 1), (1504, 1, 1), device=self.device)  # 创建一个张量t
        self.assertFalse(complex_memory_overlap(t))  # 断言t张量不存在复杂的内存重叠

    # 测试函数：生成fp8类型的随机张量
    def test_generate_rand_fp8(self):
        """
        由于缺少必要的内核，PyTorch无法生成具有正常分布的fp8张量。

        我们通过在rand_strided中首先生成fp16张量，然后进行类型转换来解决这个问题。
        """
        t = rand_strided((2, 3), (3, 1), device=self.device, dtype=torch.float8_e4m3fn)  # 使用rand_strided生成fp8张量
        self.assertTrue(t.dtype is torch.float8_e4m3fn)  # 断言张量的数据类型为torch.float8_e4m3fn
    def test_large_grid(self):
        # 测试大型网格问题
        # 解决了https://github.com/pytorch/pytorch/issues/123210中的问题

        def fn(primals_5):
            # 对输入数据进行重塑为指定形状
            view = torch.ops.aten.reshape.default(primals_5, [-1, 2, 4])
            # 将原始输入置空
            primals_5 = None
            # 对重塑后的数据进行轴置换操作
            permute = torch.ops.aten.permute.default(view, [0, 2, 1])
            # 对轴置换后的数据进行克隆操作，保持内存格式连续
            clone = torch.ops.aten.clone.default(
                permute, memory_format=torch.contiguous_format
            )
            return clone

        s0 = 16777472  # 定义第一个维度的大小
        s1 = 8  # 定义第二个维度的大小
        # 使用优化器对函数进行编译，以加速执行
        compiled_fn = torch._dynamo.optimize()(fn)
        # 对编译后的函数应用输入数据并获取结果
        actual = compiled_fn(torch.ones(s0, s1))
        # 断言所有结果均为1
        self.assertTrue((actual == 1).all())

    def test_pattern_matcher_multi_user(self):
        # 多用户模式下的模式匹配测试
        # 重现了https://github.com/pytorch/pytorch/issues/129685中的问题

        def forward(float_1, view_1):
            # 计算logits并计算交叉熵损失
            logits = float_1 / 64.0
            loss = torch.nn.functional.cross_entropy(logits, view_1, ignore_index=5)
            # 计算logits的对数和指数总和
            logsumexp = logits.logsumexp(dim=-1)
            return [loss, logsumexp]

        # 生成随机张量a，形状为(512, 4096)，并要求梯度计算
        a = torch.randn(512, 4096, requires_grad=True)
        # 生成大小为512的随机整数张量b，取值范围从0到4095
        b = torch.randint(size=(512,), low=0, high=4095)

        # 调用通用函数进行前向计算并验证结果
        self.common(forward, (a, b))
# 使用 dataclasses 装饰器定义一个名为 TestFailure 的数据类，用于表示测试失败的信息
@dataclasses.dataclass
class TestFailure:
    suffixes: Tuple[str]  # 包含后缀字符串的元组
    is_skip: bool = False  # 是否为跳过测试，默认为 False
    __test__: bool = False  # 是否为测试项，默认为 False


# 定义一个函数 copy_tests，用于复制测试方法到其他类中
def copy_tests(
    my_cls, other_cls, suffix, test_failures=None, xfail_prop=None
):  # noqa: B902
    # 遍历 my_cls 类的所有属性和方法
    for name, value in my_cls.__dict__.items():
        if name.startswith("test_"):
            # 由于 Python 中不能直接复制函数，因此使用闭包创建具有不同 id 的对象。
            # 否则，unittest.skip 会修改所有共享相同对象 id 的方法。
            # 通过使用默认参数，我们创建一个副本而非引用，以保证对 value 的访问。

            @functools.wraps(value)
            def new_test(self, value=value):
                return value(self)

            # 复制 value 的 __dict__，其中可能包含测试的元数据
            new_test.__dict__ = copy.deepcopy(value.__dict__)

            # 如果 xfail_prop 不为 None，并且 value 具有 xfail_prop 属性，则标记为预期失败
            if xfail_prop is not None and hasattr(value, xfail_prop):
                new_test = unittest.expectedFailure(new_test)

            # 获取指定名称的测试失败信息 tf
            tf = test_failures and test_failures.get(name)
            if tf is not None and suffix in tf.suffixes:
                # 如果 tf 存在且 suffix 在其 suffixes 中，根据 is_skip 判断是否跳过测试
                skip_func = (
                    unittest.skip("Skipped!")
                    if tf.is_skip
                    else unittest.expectedFailure
                )
                new_test = skip_func(new_test)

            # 将新测试方法 new_test 设置为 other_cls 的属性，属性名为原始名称加上后缀
            setattr(other_cls, f"{name}_{suffix}", new_test)


# 如果存在 HAS_CPU：

    class SweepInputsCpuTest(SweepInputs2, TestCase):
        gen = InputGen(10, "cpu")

    SweepInputsCpuTest.populate()

    class CpuTests(TestCase):
        common = check_model
        device = "cpu"

    # 将 CommonTemplate 类的测试方法复制到 CpuTests 类中，后缀为 "cpu"
    copy_tests(CommonTemplate, CpuTests, "cpu")

# 如果同时存在 HAS_GPU 并且不在测试模式下进行地址安全性分析（ASAN）：

    class SweepInputsGPUTest(SweepInputs2, TestCase):
        gen = InputGen(10, GPU_TYPE)

    SweepInputsGPUTest.populate()

    class GPUTests(TestCase):
        common = check_model_gpu
        device = GPU_TYPE

    # 将 CommonTemplate 类的测试方法复制到 GPUTests 类中，后缀为 GPU_TYPE
    copy_tests(CommonTemplate, GPUTests, GPU_TYPE)

    # 加载指定位置的数据，并设置一些额外的参数
    tmp0 = tl.load(in_ptr0 + (x1 + (512*x0) + (262144*r2)), rmask, eviction_policy='evict_last', other=0.0)
        # 在指定内存位置in_ptr1 + (x3 + (262144*r2))处加载数据，使用指定的读取掩码rmask，
        # 并指定默认值other为0.0
        tmp1 = tl.load(in_ptr1 + (x3 + (262144*r2)), rmask, other=0.0)

        # 在指定内存位置in_ptr0 + (x1 + (512*x0) + (262144*r2))处加载数据，使用指定的读取掩码rmask，
        # 并指定eviction_policy='evict_last'，同时指定默认值other为0.0
        tmp0 = tl.load(in_ptr0 + (x1 + (512*x0) + (262144*r2)), rmask, eviction_policy='evict_last', other=0.0)

        # 在指定内存位置in_ptr1 + (x3 + (262144*r2))处加载数据，使用指定的读取掩码rmask，
        # 并指定eviction_policy='evict_first'，同时指定默认值other为0.0
        tmp1 = tl.load(in_ptr1 + (x3 + (262144*r2)), rmask, eviction_policy='evict_first', other=0.0)
    # 定义一个测试类 NanCheckerTest，继承自 TestCase
    class NanCheckerTest(TestCase):
    
        # 装饰器，用于测试函数，将 nan_asserts 配置为 True
        @config.patch("nan_asserts", True)
        # 定义测试函数 test_nan_checker_pass
        def test_nan_checker_pass(self):
            
            # 定义一个函数 f(x)，使用 torch.softmax 对 x 进行操作
            def f(x):
                return torch.softmax(x, dim=-1)
    
            # 生成一个设备为 GPU_TYPE 的 2x1024 的随机张量 x
            x = torch.randn(2, 1024, device=GPU_TYPE)
            # 计算参考结果 ref，调用函数 f(x)
            ref = f(x)
            # 运行并获取编译后的代码及其执行结果，返回值为 actual 和 (code,)
            actual, (code,) = run_and_get_code(torch.compile(f), x)
            
            # 断言 ref 和 actual 在数值上接近
            self.assertTrue(torch.allclose(ref, actual))
            # 断言代码字符串中包含 "# make sure graph inputs are not nan/inf"
            self.assertTrue("# make sure graph inputs are not nan/inf" in code)
            
            # 使用正则表达式搜索代码字符串，确保不存在任何元素是 nan 的断言
            self.assertTrue(
                re.search(r"assert not .*\.isnan\(\)\.any\(\).item\(\)", code)
                is not None
            )
            
            # 使用正则表达式搜索代码字符串，确保不存在任何元素是 inf 的断言
            self.assertTrue(
                re.search(r"assert not .*\.isinf\(\)\.any\(\).item\(\)", code)
                is not None
            )
    
        # 装饰器，用于测试函数，将 nan_asserts 配置为 True
        @config.patch("nan_asserts", True)
        # 定义测试函数 test_nan_checker_fail
        def test_nan_checker_fail(self):
            
            # 定义一个函数 f(x)，使用 torch.softmax 对 x 进行操作
            def f(x):
                return torch.softmax(x, dim=-1)
    
            # 生成一个设备为 GPU_TYPE 的 2x1024 的随机张量 x
            x = torch.randn(2, 1024, device=GPU_TYPE)
            # 将 x 的第一个元素设置为 float("nan")
            x[0, 0] = float("nan")
            
            # 断言运行函数 f(x) 抛出 AssertionError 异常
            with self.assertRaises(AssertionError):
                torch.compile(f)(x)
# 如果存在 CPU 的话执行以下代码块
if HAS_CPU:

    # 定义一个名为 TestFull 的测试类，继承自 TestCase
    class TestFull(TestCase):
        
        # 定义一个测试方法 test_full_dtype
        def test_full_dtype(self):
            # Python 原生数据类型列表
            pytypes = (
                bool,
                int,
                float,
                # TODO: Triton's JITFunction._type_of has no support for complex
                # 复数类型，目前 Triton 的 JITFunction._type_of 不支持
                # complex,
            )

            # Torch 张量数据类型列表
            dtypes = (
                torch.bool,
                torch.int32,
                torch.int64,
                torch.float32,
                torch.float64,
                None,  # 表示空数据类型
                # torch.complex64,
                # torch.complex128,
            )

            # 定义一个函数 fn，根据给定的 Python 类型和 Torch 数据类型返回一个填充了指定值的张量
            def fn(pytype, dtype):
                if pytype is bool:
                    fill_value = True
                elif pytype is int:
                    fill_value = 42
                elif pytype is float:
                    fill_value = 42.0
                else:
                    raise AssertionError(f"Unexpected Python type: {pytype}")

                # 调用 torch.full 创建一个指定大小、填充值、数据类型和设备的张量
                return torch.full(
                    (4, 6), fill_value, dtype=dtype, device=torch.device("cpu")
                )

            # 使用 Torch 内部函数优化 fn 函数
            fn_opt = torch._dynamo.optimize("inductor")(fn)

            # 使用 itertools.product 遍历 pytypes 和 dtypes 的所有组合
            for pytype, dtype in itertools.product(pytypes, dtypes):
                # 启用 Python 调度器
                with enable_python_dispatcher():
                    # 禁用梯度追踪
                    with torch.no_grad():
                        # 调用优化后的 fn_opt 函数计算结果
                        ret_opt = fn_opt(pytype, dtype)

                # 使用 TestCase 中的断言方法，验证优化后的结果与未优化结果一致
                self.assertEqual(ret_opt, fn(pytype, dtype))


# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 从 torch._inductor.test_case 模块导入 run_tests 函数
    from torch._inductor.test_case import run_tests

    # 如果有 CPU 或 GPU 存在，运行测试，需要 filelock
    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
```
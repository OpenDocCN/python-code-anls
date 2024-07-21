# `.\pytorch\torch\testing\_internal\hop_db.py`

```
# 忽略类型检查错误，通常用于向类型检查器表明有意忽略的特定错误
# import torch 模块，用于进行张量操作和深度学习任务
import torch
# functools 模块，提供了一些有用的功能工具，如偏函数
import functools
# 从 torch.testing 中导入 make_tensor 函数，用于创建测试用张量
from torch.testing import make_tensor
# 导入 unittest 模块，用于编写和运行单元测试
import unittest
# 从 functorch.experimental.control_flow 导入 map 函数，用于并行映射
from functorch.experimental.control_flow import map
# 从 torch.testing._internal.opinfo.core 导入 OpInfo 和 SampleInput 类
from torch.testing._internal.opinfo.core import (
    OpInfo,
    SampleInput,
)
# 从 torch.testing._internal.common_dtype 导入 all_types_and 和 custom_types 函数
from torch.testing._internal.common_dtype import all_types_and, custom_types
# 从 torch.testing._internal.common_utils 导入 IS_WINDOWS 常量
from torch.testing._internal.common_utils import IS_WINDOWS
# 从 torch.testing._internal.opinfo.core 导入 DecorateInfo 类
from torch.testing._internal.opinfo.core import DecorateInfo
# 从 torch.nn.attention._flex_attention 导入 _flex_attention 函数
from torch.nn.attention._flex_attention import _flex_attention

# 定义一个生成器函数，生成用于映射操作的输入样本
def sample_inputs_map(opinfo, device, dtype, requires_grad, **kwargs):
    # 使用 functools.partial 创建 make_tensor 的部分函数，设定设备、数据类型和梯度属性
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 生成包含两个张量的 SampleInput 对象，每个张量形状为 (2, 2, 2)，数值范围在 [0.1, 2] 之间
    yield SampleInput([make_arg(2, 2, 2, low=0.1, high=2), make_arg(2, 2, 2, low=0.1, high=2)],
                      args=(make_arg(1, low=0.1, high=2), make_arg(1, low=0.1, high=2)))

# 定义一个内部函数 inner_f，接受三个参数并返回一个列表
def inner_f(x, y0, y1):
    # 对第一个张量进行余弦运算，并在结果上加 1，再乘以 y0
    return [x[0].cos().add_(1.) * y0, (x[1] + y1.sin()).cos_().view(x[1].size())]

# 定义一个简单的映射函数 simple_map，接受三个参数并返回 map 函数的结果
def simple_map(xs, y0, y1):
    # 定义一个内部函数 f，接受三个参数并调用 inner_f 函数
    def f(x, y0, y1):
        return inner_f(x, y0, y1)
    # 调用 map 函数，对 xs 中的每个元素调用 f 函数，并传入 y0 和 y1
    return map(f, xs, y0, y1)

# 定义一个嵌套映射函数 nested_map，接受三个参数并返回 map 函数的结果
def nested_map(xs, y0, y1):
    # 定义一个内部函数 f1，接受三个参数并定义内部函数 f2，调用 inner_f 函数
    def f1(xx, y0, y1):
        def f2(x, y0, y1):
            return inner_f(x, y0, y1)
        # 调用 map 函数，对 xx 中的每个元素调用 f2 函数，并传入 y0 和 y1
        return map(f2, xx, y0, y1)
    # 调用 map 函数，对 xs 中的每个元素调用 f1 函数，并传入 y0 和 y1
    return map(f1, xs, y0, y1)

# 定义一个三重嵌套映射函数 triple_nested_map，接受三个参数并返回 map 函数的结果
def triple_nested_map(xs, y0, y1):
    # 定义一个内部函数 f0，接受三个参数并定义内部函数 f1 和 f2，调用 inner_f 函数
    def f0(xs, y0, y1):
        def f1(xx, y0, y1):
            def f2(x, y0, y1):
                return inner_f(x, y0, y1)
            # 调用 map 函数，对 xx 中的每个元素调用 f2 函数，并传入 y0 和 y1
            return map(f2, xx, y0, y1)
        # 调用 map 函数，对 xs 中的每个元素调用 f1 函数，并传入 y0 和 y1
        return map(f1, xs, y0, y1)
    # 调用 map 函数，对 xs 中的每个元素调用 f0 函数，并传入 y0 和 y1
    return map(f0, xs, y0, y1)

# 定义一个列表，列出不包含 opinfo 测试白名单中的操作
hop_that_doesnt_have_opinfo_test_allowlist = [
    "custom_function_call",
    "autograd_function_apply",
    "run_and_save_rng_state",
    "run_with_rng_state",
    "out_dtype",
    "trace_wrapped",
    "map",  # T183144629
    "map_impl",
    "with_effects",
    "strict_mode",
    "_export_tracepoint",
    "call_torchbind",
]

# 定义一个自定义的 torch.library，命名为 "testlib::mutating_custom_op"，描述其签名和标签
torch.library.define(
    "testlib::mutating_custom_op",
    "(Tensor(a!) x, Tensor(b!) z) -> (Tensor, Tensor, Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)

# 在 CPU 环境下实现 "testlib::mutating_custom_op" 的具体实现，对输入张量进行特定操作
@torch.library.impl("testlib::mutating_custom_op", "cpu")
def foo_impl_cpu(x, z):
    x.add_(5)
    z.add_(5)
    return x, z, x + z

# 在 CUDA 环境下实现 "testlib::mutating_custom_op" 的具体实现，对输入张量进行特定操作
@torch.library.impl("testlib::mutating_custom_op", "cuda")
def foo_impl_cuda(x, z):
    x.add_(5)
    z.add_(5)
    return x, z, x + z

# 注册一个假的 torch.library，用于在抽象层面上表示 "testlib::mutating_custom_op" 操作
@torch.library.register_fake("testlib::mutating_custom_op")
def foo_impl_abstract(x, z):
    return x, z, x + z

# 定义一个生成器函数，生成用于条件判断的输入样本
def sample_inputs_cond(opinfo, device, dtype, requires_grad, **kwargs):
    # 使用 functools.partial 创建 make_tensor 的部分函数，设定设备、数据类型和梯度属性
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=False
    )
    # 生成一个包含单个张量的 SampleInput 对象，张量形状为 (2, 2, 2)，数值范围在 [0.1, 2] 之间
    yield SampleInput(make_arg(2, 2, 2, low=0.1, high=2))

# 定义一个简单的条件函数 simple_cond，接受一个张量 x 作为输入，并根据其形状进行条件判断
def simple_cond(x):
    return torch.cond(x.shape[0] > 2, lambda x: x.cos(), lambda x: x.sin(), [x])

# 定义一个生成器函数，用于自动化功能化的输入样本
def sample_inputs_auto_functionalize(opinfo, device, dtype, requires_grad, **kwargs):
    # 创建一个部分函数应用，使用 functools.partial 来部分应用 make_tensor 函数
    # 这个部分函数 make_arg 的参数设定为：device=device, dtype=dtype, requires_grad=False
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=False
    )
    # 使用 yield 来生成一个 SampleInput 对象，包含两个参数：
    # 第一个参数是调用 make_arg 函数生成的张量（2x2x2），数值范围在 0.1 到 2 之间
    # 第二个参数同样是调用 make_arg 函数生成的张量（2x2x2），数值范围在 0.1 到 2 之间
    yield SampleInput(make_arg(2, 2, 2, low=0.1, high=2), make_arg(2, 2, 2, low=0.1, high=2))
# 定义一个简单的函数，调用 torch 库中的自定义操作，返回操作结果
def simple_auto_functionalize(x, z):
    return torch.ops.testlib.mutating_custom_op(x, z)


# 生成一个灵活的输入样本生成器函数，使用 functools.partial 部分应用函数生成特定参数的张量
def sample_inputs_flex_attention(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )

    # 定义一个得分修改函数，用于计算得分的加法操作
    def score_mod(score, b, h, m, n):
        return score + h

    # 使用生成器 yield 语句产生一个 SampleInput 对象，包含多个调用 make_arg 函数生成的张量和 score_mod 函数
    yield SampleInput(
        make_arg(2, 2, 128, 8, low=0.1, high=2),
        make_arg(2, 2, 128, 8, low=0.1, high=2),
        make_arg(2, 2, 128, 8, low=0.1, high=2),
        score_mod,
    )

# 生成一个包含 while 循环的输入样本生成器函数
def sample_inputs_while_loop(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=False
    )
    # 使用生成器 yield 语句产生一个 SampleInput 对象，包含多个张量
    yield SampleInput(
        torch.tensor(3),
        make_arg(2, 3, 4, low=0.1, high=2),
    )

# 定义一个简单的 while 循环函数，调用 torch 库中的 while_loop 操作执行条件和主体函数
def simple_while_loop(iter_t, x):
    # 定义循环条件函数
    def cond_fn(iter_t, x):
        return iter_t > 0

    # 定义循环主体函数
    def body_fn(iter_t, x):
        return iter_t - 1, x.cos()

    # 调用 torch 库中的 while_loop 函数执行循环
    return torch._higher_order_ops.while_loop(cond_fn, body_fn, (iter_t, x))

# 定义一个操作信息列表，每个元素是 OpInfo 对象，描述一个操作及其测试信息
hop_db = [
    OpInfo(
        name="map",
        variant_test_name="simple",
        op=simple_map,
        sample_inputs_func=sample_inputs_map,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        check_inplace_batched_forward_grad=False,
    ),
    OpInfo(
        name="map",
        variant_test_name="nested",
        op=nested_map,
        sample_inputs_func=sample_inputs_map,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        check_inplace_batched_forward_grad=False,
    ),
    OpInfo(
        name="map",
        variant_test_name="triple_nested",
        op=triple_nested_map,
        sample_inputs_func=sample_inputs_map,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        check_inplace_batched_forward_grad=False,
    ),
    OpInfo(
        name="cond",
        variant_test_name="simple",
        op=simple_cond,
        sample_inputs_func=sample_inputs_cond,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        check_inplace_batched_forward_grad=False,
        supports_autograd=False,
    ),
]
    OpInfo(
        name="while_loop",  # 操作名称为 while_loop
        variant_test_name="simple",  # 测试变体名称为 simple
        op=simple_while_loop,  # 操作函数为 simple_while_loop
        sample_inputs_func=sample_inputs_while_loop,  # 获取 while_loop 操作的样本输入函数
        dtypes=all_types_and(torch.bool, torch.half),  # 支持所有数据类型和 torch.bool、torch.half
        supports_out=False,  # 不支持输出参数
        check_batched_grad=False,  # 不检查批处理梯度
        check_batched_gradgrad=False,  # 不检查批处理梯度的梯度
        check_batched_forward_grad=False,  # 不检查批处理前向传播的梯度
        check_inplace_batched_forward_grad=False,  # 不检查批处理前向传播的原地操作梯度
        supports_autograd=False,  # 不支持自动求导
    ),
    OpInfo(
        name="auto_functionalize",  # 操作名称为 auto_functionalize
        variant_test_name="simple",  # 测试变体名称为 simple
        op=simple_auto_functionalize,  # 操作函数为 simple_auto_functionalize
        sample_inputs_func=sample_inputs_auto_functionalize,  # 获取 auto_functionalize 操作的样本输入函数
        dtypes=all_types_and(torch.bool, torch.half),  # 支持所有数据类型和 torch.bool、torch.half
        supports_out=False,  # 不支持输出参数
        check_batched_grad=False,  # 不检查批处理梯度
        check_batched_gradgrad=False,  # 不检查批处理梯度的梯度
        check_batched_forward_grad=False,  # 不检查批处理前向传播的梯度
        check_inplace_batched_forward_grad=False,  # 不检查批处理前向传播的原地操作梯度
        supports_autograd=False,  # 不支持自动求导
    ),
    OpInfo(
        name="flex_attention",  # 操作名称为 flex_attention
        variant_test_name="simple",  # 测试变体名称为 simple
        op=_flex_attention,  # 操作函数为 _flex_attention
        sample_inputs_func=sample_inputs_flex_attention,  # 获取 flex_attention 操作的样本输入函数
        dtypes=custom_types(torch.float16, torch.float32),  # 支持自定义数据类型，包括 torch.float16 和 torch.float32
        supports_out=False,  # 不支持输出参数
        check_batched_grad=False,  # 不检查批处理梯度
        check_batched_gradgrad=False,  # 不检查批处理梯度的梯度
        check_batched_forward_grad=False,  # 不检查批处理前向传播的梯度
        check_inplace_batched_forward_grad=False,  # 不检查批处理前向传播的原地操作梯度
        skips=(  # 跳过以下测试用例
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_aot_export"),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_pre_dispatch_export"),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_serialize_export"),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_retrace_export"),
            DecorateInfo(  # 条件跳过测试用例，仅在非 Windows 环境下生效
                unittest.expectedFailure,
                "TestProxyTensorOpInfo",
                "test_make_fx_symbolic_exhaustive",
                active_if=not IS_WINDOWS,
            ),
            DecorateInfo(  # 条件跳过测试用例，仅在非 Windows 环境下生效
                unittest.expectedFailure,
                "TestEagerFusionOpInfo",
                "test_aot_autograd_symbolic_exhaustive",
                active_if=not IS_WINDOWS,
            ),
        ),
    ),
    OpInfo(
        name="flex_attention_backward",  # 操作的名称为 "flex_attention_backward"
        variant_test_name="simple",  # 测试变体的名称为 "simple"
        op=_flex_attention,  # 使用函数 _flex_attention 作为操作的实现
        sample_inputs_func=sample_inputs_flex_attention,  # 用于生成输入样本的函数为 sample_inputs_flex_attention
        dtypes=custom_types(torch.float16, torch.float32),  # 支持的数据类型为 torch.float16 和 torch.float32
        supports_out=False,  # 不支持输出结果
        check_batched_grad=False,  # 不检查批量梯度
        check_batched_gradgrad=False,  # 不检查批量二阶梯度
        check_batched_forward_grad=False,  # 不检查批量前向梯度
        check_inplace_batched_forward_grad=False,  # 不检查原地批量前向梯度
        skips=(  # 跳过以下测试装饰器所标记的测试用例
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_aot_export"),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_pre_dispatch_export"),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_serialize_export"),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_retrace_export"),
            DecorateInfo(
                unittest.expectedFailure,
                "TestProxyTensorOpInfo",
                "test_make_fx_symbolic_exhaustive",
                active_if=not IS_WINDOWS,  # 仅在非 Windows 系统下激活
            ),
            DecorateInfo(
                unittest.expectedFailure,
                "TestEagerFusionOpInfo",
                "test_aot_autograd_symbolic_exhaustive",
                active_if=not IS_WINDOWS,  # 仅在非 Windows 系统下激活
            ),
        ),
    )
]



# 在此处省略了上文的部分代码，需在上下文中查看以理解其含义和作用
```
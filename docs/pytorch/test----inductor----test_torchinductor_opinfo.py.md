# `.\pytorch\test\inductor\test_torchinductor_opinfo.py`

```
# Owner(s): ["module: inductor"]
# 导入必要的模块和库
import atexit                # 用于注册退出函数的模块
import contextlib            # 提供上下文管理工具的模块
import functools             # 提供创建偏函数的模块
import os                    # 提供与操作系统交互的功能
import sys                   # 提供与Python解释器交互的功能
import unittest              # 提供单元测试框架的模块
from collections import defaultdict  # 提供默认字典的模块
from enum import Enum        # 提供枚举类型的支持
from functools import partial  # 提供创建偏函数的功能
from unittest.mock import patch  # 提供模拟对象的功能

import torch                 # PyTorch深度学习框架

from torch._dispatch.python import enable_python_dispatcher  # 导入Python调度器的功能
from torch._inductor.test_case import run_tests, TestCase    # 导入测试用例相关功能
from torch._subclasses.fake_tensor import (  # 导入虚拟张量相关异常和模式定义
    DataDependentOutputException,
    DynamicOutputShapeException,
    FakeTensorMode,
)
from torch.testing._internal.common_cuda import SM80OrLater  # 导入CUDA相关的硬件支持检查
from torch.testing._internal.common_device_type import (  # 导入设备类型相关的测试支持
    instantiate_device_type_tests,
    onlyNativeDeviceTypes,
    OpDTypes,
    ops,
    skipCPUIf,
    skipCUDAIf,
)
from torch.testing._internal.common_methods_invocations import (  # 导入方法调用相关的测试支持
    op_db,
    skipOps,
)
from torch.testing._internal.common_utils import (  # 导入通用工具函数
    dtype_abbrs,
    IS_MACOS,
    IS_X86,
    skipCUDAMemoryLeakCheckIf,
    skipIfCrossRef,
    skipIfTorchDynamo,
    suppress_warnings,
    TEST_MKL,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import (  # 导入感应器工具函数
    GPU_TYPE,
    HAS_CPU,
    HAS_CUDA,
)
from torch.utils._python_dispatch import TorchDispatchMode  # 导入分发模式支持
from torch.utils._pytree import tree_map  # 导入树映射相关功能

try:
    try:
        from .test_torchinductor import check_model, check_model_gpu  # 尝试导入本地测试模块中的检查函数
    except ImportError:
        from test_torchinductor import check_model, check_model_gpu  # 如果导入失败，则从全局中导入
except (unittest.SkipTest, ImportError) as e:
    sys.stderr.write(f"{type(e)}: {e}\n")  # 如果导入失败，则在标准错误流中写入异常信息
    if __name__ == "__main__":
        sys.exit(0)  # 如果作为主程序执行，则退出程序
    raise  # 否则，向上抛出导入异常

bf16 = torch.bfloat16  # 定义torch.bfloat16的别名bf16，但未进行测试
f64 = torch.float64    # 定义torch.float64的别名f64
f32 = torch.float32    # 定义torch.float32的别名f32
f16 = torch.float16    # 定义torch.float16的别名f16
i8 = torch.int8        # 定义torch.int8的别名i8，但未进行测试
i16 = torch.int16      # 定义torch.int16的别名i16，但未进行测试
i32 = torch.int32      # 定义torch.int32的别名i32
i64 = torch.int64      # 定义torch.int64的别名i64
b8 = torch.bool        # 定义torch.bool的别名b8
u8 = torch.uint8       # 定义torch.uint8的别名u8，但未进行测试（除了上采样和插值操作）
u16 = torch.uint16     # 定义torch.uint16的别名u16，但未进行测试
u32 = torch.uint32     # 定义torch.uint32的别名u32，但未进行测试
u64 = torch.uint64     # 定义torch.uint64的别名u64，但未进行测试

# 定义_ops作为ops的偏函数，指定支持的数据类型和允许的数据类型列表
_ops = partial(
    ops,
    dtypes=OpDTypes.supported,
    allowed_dtypes=[f16, f32, f64, i32, i64, b8, u8, u16, u32, u64],
)

# 枚举类型，用于标识测试的预期结果，包括成功、预期失败和跳过
ExpectedTestResult = Enum("ExpectedTestResult", ("SUCCESS", "XFAILURE", "SKIP"))

# 根据环境变量设置是否收集预期结果和是否包含所有样本的标志
COLLECT_EXPECT = os.getenv("PYTORCH_COLLECT_EXPECT", "0") == "1"
ALL_SAMPLES = os.getenv("PYTORCH_ALL_SAMPLES", "0") == "1"

# 根据环境变量设置测试范围的起始和结束索引
START = os.getenv("PYTORCH_TEST_RANGE_START", None)
END = os.getenv("PYTORCH_TEST_RANGE_END", None)

# 如果设置了起始和结束索引，则进行相应的断言和转换为整数
if START is not None or END is not None:
    assert END is not None
    assert START is not None
    START = int(START)
    END = int(END)
    assert START < END
else:
    START = 0  # 否则，起始索引默认为0
    END = len(op_db)  # 结束索引默认为op_db的长度

# 初始化存储已见失败的测试和失败原因的字典
seen_failed = defaultdict(set)
failed_reasons = defaultdict(set)


def print_seen():
    # 内部函数，用于格式化数据类型的简称
    def fmt_dtypes(dtypes):
        r = ", ".join(sorted(dtype_abbrs[d] for d in dtypes))
        return "{" + r + "}"
    def sort_key(kv):
        # 定义排序函数，根据键值对中的第二个元素排序
        k, v = kv
        device_type, op = k
        if isinstance(op, tuple):
            return op  # 如果操作是元组，直接返回作为排序依据
        else:
            return op, ""  # 如果操作不是元组，返回操作和空字符串作为排序依据

    for (device_type, op), failed_dtypes in sorted(seen_failed.items(), key=sort_key):
        # 遍历按照 sort_key 函数排序后的 seen_failed 字典的键值对
        key = device_type, op
        reasons = ""
        if failed_reasons[key]:
            # 如果 failed_reasons 字典中存在与当前键对应的值

            def maybe_truncate(x, length=80):
                # 定义截断字符串的函数，将换行符替换为空格，并根据长度截断
                x = str(x).replace("\n", " ")

                idx = x.find("\\n")
                if idx >= 0:
                    x = f"{x[:idx]}..."
                if len(x) > length:
                    return f"{x[:length - 3]}..."  # 如果超过长度，截断并加上省略号
                return x

            reasons = sorted(set(map(maybe_truncate, failed_reasons[key])))
            reasons = "  # " + ", ".join(reasons)  # 将截断后的原因以逗号分隔，并添加注释提示

        if failed_dtypes:
            # 如果 failed_dtypes 非空

            def format_op(op):
                # 格式化操作符，如果是元组则返回格式化后的字符串，否则直接返回字符串
                if isinstance(op, tuple):
                    return f'("{op[0]}", "{op[1]}")'
                else:
                    return f'"{op}"'

            expected_failures[device_type].append(
                f"    {format_op(op)}: {fmt_dtypes(failed_dtypes)},{reasons}"
            )
            # 将格式化后的操作符、失败数据类型和原因（如果有的话）加入到 expected_failures 字典中对应设备类型的列表中

    for device_type in ("cpu", GPU_TYPE):
        expected_failures[device_type]
        nl = "\n"
        print(
            f"""
                # 打印设备类型为 "cpu" 和 GPU_TYPE 的预期失败条目
# 定义期望失败的单个样本的一个字典，用于记录不同设备类型的期望失败情况
inductor_expected_failures_single_sample["{device_type}"] = {
    # 插入预期失败的测试列表，每个设备类型可以有多个失败测试
    {nl.join(expected_failures[device_type])}
}
"""
)  # 将期望失败的测试列表插入到字典中，使用设备类型作为键，存储对应的测试结果

if COLLECT_EXPECT:
    atexit.register(print_seen)  # 如果启用了期望结果的收集，注册一个函数在退出时打印已见的期望结果

# 在跳过和标记为预期失败的字典中，使用字符串作为默认测试的键，使用包含两个字符串的元组作为变体的键

# 初始化一个默认字典，用于记录各设备类型下的跳过测试
inductor_skips = defaultdict(dict)

# 对CPU设备类型进行配置，记录特定测试的跳过情况
inductor_skips["cpu"] = {
    "linalg.ldl_factor": {f32, f64},  # 核心功能不稳定
    "nn.functional.cosine_embedding_loss": {b8},  # 核心功能不稳定
    ("index_reduce", "prod"): {f16},  # 核心功能不稳定
    ("index_reduce", "mean"): {f16},  # 核心功能不稳定
}

# 如果运行在MacOS且是x86架构，则进一步配置CPU设备下的跳过测试
if IS_MACOS and IS_X86:
    inductor_skips["cpu"]["rsqrt"] = {b8, i32}
    inductor_skips["cpu"]["nn.functional.multi_margin_loss"] = {
        b8,
        f16,
        f32,
        f64,
        i32,
        i64,
    }

# 配置CUDA设备类型下的跳过测试
inductor_skips["cuda"] = {
    # 预期Jiterator内核不适用于Inductor
    "jiterator_2inputs_2outputs": {b8, f16, f32, f64, i32, i64},
    "jiterator_4inputs_with_extra_args": {b8, f16, f32, f64, i32, i64},
    "jiterator_binary": {b8, f16, f32, f64, i32, i64},
    "jiterator_binary_return_by_ref": {b8, f16, f32, f64, i32, i64},
    "jiterator_unary": {b8, f16, f32, f64, i32, i64},
    # 核心功能不稳定
    "nn.functional.cosine_embedding_loss": {b8},
    "native_batch_norm": {f16, f32, f64},
    "_native_batch_norm_legit": {f16, f32, f64},
    "_batch_norm_with_update": {f16, f32, f64},
}

# 如果不是SM80或更新版本，则配置CUDA设备类型下的跳过测试
if not SM80OrLater:
    inductor_skips["cuda"]["bfloat16"] = {b8, f16, f32, f64, i32, i64}

# 如果使用ROCM进行测试，则配置CUDA设备类型下的跳过测试
if TEST_WITH_ROCM:
    # 张量不相似
    inductor_skips["cuda"]["logcumsumexp"] = {f32}
    inductor_skips["cuda"]["special.modified_bessel_i1"] = {f64}

# 初始化一个默认字典，用于记录不同设备类型下的预期失败的单个样本
inductor_expected_failures_single_sample = defaultdict(dict)

# 配置CPU设备类型下的预期失败的单个样本
inductor_expected_failures_single_sample["cpu"] = {
    "_softmax_backward_data": {
        f16
    },  # half_to_float仅对CUDA实现有效
    "_upsample_bilinear2d_aa": {f32, f64},
    "cholesky": {f32, f64},
    "complex": {f16},
    "resize_": {b8, f16, f32, f64, i32, i64},
    "resize_as_": {b8, f16, f32, f64, i32, i64},
    "histc": {f16},
    "multinomial": {f16, f32, f64},
    "nn.functional.avg_pool1d": {i64},
    "nn.functional.avg_pool2d": {i64},
    "nn.functional.avg_pool3d": {i64},
    "nn.functional.local_response_norm": {i64},
    "nn.functional.rrelu": {f32, f64},
    "nonzero_static": {b8, f16, f32, f64, i32, i64},
    ("normal", "in_place"): {f16, f32, f64},
    ("normal", "number_mean"): {f16, f32, f64},
    ("sparse.mm", "reduce"): {f32, f64},
    "sparse.sampled_addmm": {f32, f64},
    "to_sparse": {
        f32,
        f64,
    },  # NYI: 找不到DispatchKey.SparseCPU分发键下的aten.view.default内核
    "view_as_complex": {f16},
}

# 配置CUDA设备类型下的预期失败的单个样本
inductor_expected_failures_single_sample["cuda"] = {
    "_upsample_bilinear2d_aa": {f16, f32, f64},
    "cholesky": {f32, f64},
    "multinomial": {f16, f32, f64},
    ("normal", "in_place"): {f16, f32, f64},
    ("normal", "number_mean"): {f16, f32, f64},
}
    "sparse.sampled_addmm": {f32, f64},
    # 定义键为 "sparse.sampled_addmm" 的字典条目，值为 {f32, f64}
    "torch.ops.aten._flash_attention_forward": {f16},
    # 定义键为 "torch.ops.aten._flash_attention_forward" 的字典条目，值为 {f16}
    "torch.ops.aten._efficient_attention_forward": {f16, f32},
    # 定义键为 "torch.ops.aten._efficient_attention_forward" 的字典条目，值为 {f16, f32}
    "to_sparse": {
        f16,
        f32,
        f64,
    },  # NYI: could not find kernel for aten.view.default at dispatch key DispatchKey.SparseCUDA
    # 定义键为 "to_sparse" 的字典条目，值为 {f16, f32, f64}，此处有一个注释说明 NYI，即 "Not Yet Implemented"，表示某些功能尚未实现，具体为找不到与 DispatchKey.SparseCUDA 相关的 aten.view.default 内核
# intentionally not handled
# 非故意不处理的情况
intentionally_not_handled = {
    "resize_": {b8, f16, f32, f64, i32, i64},  # 声明了一个名为 "resize_" 的集合，包含多种数据类型
    "resize_as_": {b8, f16, f32, f64, i32, i64},  # 声明了一个名为 "resize_as_" 的集合，包含多种数据类型
}

# This is only fixed when this config is set
# We should eventually always turn it on
# 只有在设置了此配置时才会修复此问题
# 我们最终应该总是打开它
import torch._functorch.config as functorch_config

if not functorch_config.view_replay_for_aliased_outputs:
    # 如果未设置 functorch_config.view_replay_for_aliased_outputs，添加一个新的条目到 intentionally_not_handled 字典中
    intentionally_not_handled['("as_strided", "partial_views")'] = {
        b8,
        f16,
        f32,
        f64,
        i32,
        i64,
    }

# 将 intentionally_not_handled 的内容更新到 inductor_expected_failures_single_sample["cuda"] 中
inductor_expected_failures_single_sample["cuda"].update(intentionally_not_handled)


# 创建一个 defaultdict，用于存储梯度相关的预期失败情况
inductor_gradient_expected_failures_single_sample = defaultdict(dict)

# 初始化 inductor_gradient_expected_failures_single_sample 的 CUDA 部分为空字典
inductor_gradient_expected_failures_single_sample["cuda"] = {}

# 如果 TEST_MKL 为假，将空字典更新到 inductor_expected_failures_single_sample["cpu"] 中
if not TEST_MKL:
    inductor_expected_failures_single_sample["cpu"].update({})

# 创建一个 defaultdict，用于指示应该有异常的情况
inductor_should_fail_with_exception = defaultdict(dict)

# 初始化 inductor_should_fail_with_exception 的 CPU 和 CUDA 部分为空字典
inductor_should_fail_with_exception["cpu"] = {}
inductor_should_fail_with_exception["cuda"] = {}


# 定义一个函数 get_skips_and_xfails，从给定的字典中获取跳过和预期失败的操作列表
def get_skips_and_xfails(from_dict, xfails=True):
    retval = set()
    for device, d in from_dict.items():
        for op, dtypes in d.items():
            if type(op) is tuple:
                op, variant_name = op
            else:
                variant_name = ""
            retval.add((op, variant_name, device, tuple(dtypes), xfails))
    return retval


# 如果出现 "AssertionError: Couldn't find OpInfo for ..." 错误，可能是因为试图使用一个测试变体，
# 需要将例如 "max.reduction_no_dim" 替换为 ("max", "reduction_no_dim") 作为这些字典中的键
# 这里列出了一些跳过测试或预期失败的操作和变体
test_skips_or_fails = (
    get_skips_and_xfails(inductor_skips, xfails=False)  # 获取不跳过但不预期失败的操作列表
    | get_skips_and_xfails(inductor_expected_failures_single_sample, xfails=True)  # 获取预期失败的操作列表
    | get_skips_and_xfails(inductor_gradient_expected_failures_single_sample, xfails=True)  # 获取梯度相关的预期失败的操作列表
)


# 定义一个函数 wrapper_noop_set_seed，用于包装操作函数并设置种子
def wrapper_noop_set_seed(op, *args, **kwargs):
    return op(*args, **kwargs)

# 将 wrapper_noop_set_seed 函数赋给 torch.testing._internal.common_methods_invocations.wrapper_set_seed
torch.testing._internal.common_methods_invocations.wrapper_set_seed = (
    wrapper_noop_set_seed
)


# 定义一个字典 inductor_override_kwargs，包含一些操作和特定参数的设置
# 键可以是 op_name，或 (op_name, deivce_type)，或 (op_name, device_type, dtype)
inductor_override_kwargs = {
    "empty": {"assert_equal": False},  # 对于 empty 操作，不需要相等断言
    "empty_permuted": {"assert_equal": False},  # 对于 empty_permuted 操作，不需要相等断言
    "empty_like": {"assert_equal": False},  # 对于 empty_like 操作，不需要相等断言
    "new_empty": {"assert_equal": False},  # 对于 new_empty 操作，不需要相等断言
    "empty_strided": {"assert_equal": False},  # 对于 empty_strided 操作，不需要相等断言
    "new_empty_strided": {"assert_equal": False},  # 对于 new_empty_strided 操作，不需要相等断言
    "randn": {"assert_equal": False},  # 对于 randn 操作，不需要相等断言
    ("cross", "cuda", f16): {"reference_in_float": True},  # 对于 CUDA 中的 cross 操作和 f16 类型，需要在浮点数中进行参考
    ("linalg.cross", "cuda", f16): {"reference_in_float": True},  # 对于 CUDA 中的 linalg.cross 操作和 f16 类型，需要在浮点数中进行参考
    ("addr", "cuda", f16): {"reference_in_float": True},  # 对于 CUDA 中的 addr 操作和 f16 类型，需要在浮点数中进行参考
    ("baddbmm", "cuda", f16): {"atol": 2e-3, "rtol": 0.002},  # 对于 CUDA 中的 baddbmm 操作和 f16 类型，需要在精度上进行调整
    ("angle", "cuda", f64): {"reference_in_float": True},  # 对于 CUDA 中的 angle 操作和 f64 类型，需要在浮点数中进行参考
    ("asin", "cuda", f16): {"reference_in_float": True},  # 对于 CUDA 中的 asin 操作和 f16 类型，需要在浮点数中进行参考
    ("atanh", "cuda", f16): {"reference_in_float": True},  # 对于 CUDA 中的 atanh 操作和 f16 类型，需要在浮点数中进行参考
}
    ("cauchy", "cuda"): {"reference_in_float": True},  # 指定 "cauchy" 函数在 CUDA 上运行时的参数设置，使用浮点数参考值
    ("cummax", "cuda", f16): {"atol": 5e-4, "rtol": 0.002},  # 指定 "cummax" 函数在 CUDA 上运行时的参数设置，包括绝对误差和相对误差
    ("cumsum", "cuda", f16): {"reference_in_float": True},  # 指定 "cumsum" 函数在 CUDA 上运行时的参数设置，使用浮点数参考值
    ("cumprod", "cuda"): {"reference_in_float": True, "atol": 7e-5, "rtol": 0.002},  # 指定 "cumprod" 函数在 CUDA 上运行时的参数设置，包括绝对误差和相对误差，使用浮点数参考值
    ("logcumsumexp", "cuda"): {"grad_atol": 8e-4, "grad_rtol": 0.001},  # 指定 "logcumsumexp" 函数在 CUDA 上运行时的参数设置，包括梯度的绝对误差和相对误差
    ("exponential", "cuda"): {"reference_in_float": True},  # 指定 "exponential" 函数在 CUDA 上运行时的参数设置，使用浮点数参考值
    ("geometric", "cuda"): {"reference_in_float": True},  # 指定 "geometric" 函数在 CUDA 上运行时的参数设置，使用浮点数参考值
    ("kron", "cuda", f16): {"reference_in_float": True},  # 指定 "kron" 函数在 CUDA 上运行时的参数设置，使用浮点数参考值
    ("log_normal", "cuda"): {"reference_in_float": True},  # 指定 "log_normal" 函数在 CUDA 上运行时的参数设置，使用浮点数参考值
    ("masked.softmin", "cuda", f16): {"atol": 1e-4, "rtol": 0.01},  # 指定 "masked.softmin" 函数在 CUDA 上运行时的参数设置，包括绝对误差和相对误差
    ("nn.functional.batch_norm", "cuda", f16): {"reference_in_float": True},  # 指定 "nn.functional.batch_norm" 函数在 CUDA 上运行时的参数设置，使用浮点数参考值
    ("nn.functional.batch_norm.without_cudnn", "cuda", f16): {"reference_in_float": True},  # 指定 "nn.functional.batch_norm.without_cudnn" 函数在 CUDA 上运行时的参数设置，使用浮点数参考值
    ("nn.functional.cosine_similarity", "cuda", f16): {"reference_in_float": True},  # 指定 "nn.functional.cosine_similarity" 函数在 CUDA 上运行时的参数设置，使用浮点数参考值
    ("nn.functional.instance_norm", "cuda", f16): {"reference_in_float": True},  # 指定 "nn.functional.instance_norm" 函数在 CUDA 上运行时的参数设置，使用浮点数参考值
    ("nn.functional.local_response_norm", "cuda", f16): {"reference_in_float": True},  # 指定 "nn.functional.local_response_norm" 函数在 CUDA 上运行时的参数设置，使用浮点数参考值
    ("nn.functional.normalize", "cuda", f16): {"atol": 1e-3, "rtol": 0.05},  # 指定 "nn.functional.normalize" 函数在 CUDA 上运行时的参数设置，包括绝对误差和相对误差
    ("nn.functional.rms_norm", "cuda", f16): {"reference_in_float": True},  # 指定 "nn.functional.rms_norm" 函数在 CUDA 上运行时的参数设置，使用浮点数参考值
    ("nn.functional.soft_margin_loss", "cuda", f16): {"reference_in_float": True},  # 指定 "nn.functional.soft_margin_loss" 函数在 CUDA 上运行时的参数设置，使用浮点数参考值
    ("nn.functional.softmin", "cuda", f16): {"atol": 1e-4, "rtol": 0.01},  # 指定 "nn.functional.softmin" 函数在 CUDA 上运行时的参数设置，包括绝对误差和相对误差
    ("nn.functional.softsign", "cuda", f16): {"reference_in_float": True},  # 指定 "nn.functional.softsign" 函数在 CUDA 上运行时的参数设置，使用浮点数参考值
    ("nn.functional.tanhshrink", "cuda", f16): {"atol": 3e-4, "rtol": 0.001},  # 指定 "nn.functional.tanhshrink" 函数在 CUDA 上运行时的参数设置，包括绝对误差和相对误差
    ("outer", "cuda", f16): {"reference_in_float": True},  # 指定 "outer" 函数在 CUDA 上运行时的参数设置，使用浮点数参考值
    ("round.decimals_3", "cuda", f16): {"reference_in_float": True},  # 指定 "round.decimals_3" 函数在 CUDA 上运行时的参数设置，使用浮点数参考值
    ("nn.functional.triplet_margin_loss", "cuda", f16): {"atol": 1e-4, "rtol": 0.02},  # 指定 "nn.functional.triplet_margin_loss" 函数在 CUDA 上运行时的参数设置，包括绝对误差和相对误差
    ("nn.functional.triplet_margin_with_distance_loss", "cuda", f16): {"atol": 1e-4, "rtol": 0.02},  # 指定 "nn.functional.triplet_margin_with_distance_loss" 函数在 CUDA 上运行时的参数设置，包括绝对误差和相对误差
    ("sinc", "cuda", f16): {"atol": 0.008, "rtol": 0.002},  # 指定 "sinc" 函数在 CUDA 上运行时的参数设置，包括绝对误差和相对误差
    ("softmax", "cpu", f16): {"atol": 1e-4, "rtol": 0.02},  # 指定 "softmax" 函数在 CPU 上运行时的参数设置，包括绝对误差和相对误差
    ("softmax", "cuda", f16): {"atol": 1e-4, "rtol": 0.02},  # 指定 "softmax" 函数在 CUDA 上运行时的参数设置，包括绝对误差和相对误差
    ("_softmax_backward_data", "cuda", f16): {"atol": 0.008, "rtol": 0.002},  # 指定 "_softmax_backward_data" 函数在 CUDA 上运行时的参数设置，包括绝对误差和相对误差
    ("special.log_ndtr", "cuda", f64): {"atol": 1e-6, "rtol": 1e-5},  # 指定 "special.log_ndtr" 函数在 CUDA 上运行时的参数设置，包括绝对误差和相对误差
    ("polygamma.polygamma_n_0", "cpu", f32): {"atol": 1e-3, "rtol": 1e-4},  # 指定 "polygamma.polygamma_n_0" 函数在 CPU 上运行时的参数设置，包括绝对误差和相对误差
    ("polygamma.polygamma_n_1", "cpu", f32): {"atol": 1e-3, "rtol": 1e-4},  # 指定 "polygamma.polygamma_n_1" 函数在 CPU 上运行时的参数设置，包括绝对误差和相对误差
    ("polygamma.polygamma_n_2", "cpu", f32): {"atol": 1e-3, "rtol": 1e-4},  # 指定 "polygamma.polygamma
    # 下面的测试用例在严格比较时失败，但由于舍入误差，允许 atol=1
    ("nn.functional.interpolate.bilinear", "cpu", u8): {"atol": 1, "rtol": 0},
    ("nn.functional.upsample_bilinear", "cpu", u8): {"atol": 1, "rtol": 0},
    ("nn.functional.interpolate.bicubic", "cpu", u8): {"atol": 1, "rtol": 0},
    
    # 由于精度损失，需要较高的 atol
    ("nn.functional.interpolate.bilinear", "cuda", f64): {"atol": 5e-4, "rtol": 0},
    ("nn.functional.upsample_bilinear", "cuda", f64): {"atol": 5e-4, "rtol": 0},
    ("nn.functional.interpolate.bicubic", "cpu", f32): {"atol": 5e-3, "rtol": 0},
    ("nn.functional.interpolate.bicubic", "cuda", f64): {"atol": 1e-3, "rtol": 0},
    
    # 要求的 atol 过高，不合理
    ("index_reduce.mean", "cuda", f16): {"check_gradient": False},
    ("index_reduce.mean", "cuda", f32): {"check_gradient": False},
    ("index_reduce.mean", "cuda", f64): {"check_gradient": False},
    
    # 梯度包含非有限条目
    ("index_reduce.amin", "cuda", f64): {"check_gradient": False},
    ("index_reduce.amin", "cuda", f32): {"check_gradient": False},
    ("index_reduce.amin", "cuda", f16): {"check_gradient": False},
    ("index_reduce.amax", "cuda", f64): {"check_gradient": False},
    ("index_reduce.amax", "cuda", f32): {"check_gradient": False},
    ("index_reduce.amax", "cuda", f16): {"check_gradient": False},
    
    # tanh 函数的 atol 和 rtol 范围
    ("tanh", "cuda", f16): {"atol": 1e-4, "rtol": 1e-2},
# 定义一个包含示例操作的字典，每个操作关联一个包含支持数据类型的集合
inductor_one_sample = {
    "_segment_reduce.lengths": {f16},  # 操作1: "_segment_reduce.lengths"，支持数据类型为 {f16}
    "_segment_reduce.offsets": {f16},  # 操作2: "_segment_reduce.offsets"，支持数据类型为 {f16}
    "addmv": {f16},                    # 操作3: "addmv"，支持数据类型为 {f16}
    "as_strided.partial_views": {f16}, # 操作4: "as_strided.partial_views"，支持数据类型为 {f16}
    "corrcoef": {f16},                 # 操作5: "corrcoef"，支持数据类型为 {f16}
    "diff": {f16},                     # 操作6: "diff"，支持数据类型为 {f16}
    "einsum": {f16, i32},              # 操作7: "einsum"，支持数据类型为 {f16, i32}
    "gradient": {f16},                 # 操作8: "gradient"，支持数据类型为 {f16}
    "histogram": {f32, f64},           # 操作9: "histogram"，支持数据类型为 {f32, f64}
    "histogramdd": {f32, f64},         # 操作10: "histogramdd"，支持数据类型为 {f32, f64}
    "index_put": {f16, f32, f64},      # 操作11: "index_put"，支持数据类型为 {f16, f32, f64}
    "linalg.eig": {f32, f64},          # 操作12: "linalg.eig"，支持数据类型为 {f32, f64}
    "linspace": {f16, i32, i64},       # 操作13: "linspace"，支持数据类型为 {f16, i32, i64}
    "linspace.tensor_overload": {f16, f32, f64, i32, i64},  # 操作14: "linspace.tensor_overload"，支持数据类型为 {f16, f32, f64, i32, i64}
    "logspace": {f16},                 # 操作15: "logspace"，支持数据类型为 {f16}
    "logspace.tensor_overload": {f16, f32, f64, i32, i64},    # 操作16: "logspace.tensor_overload"，支持数据类型为 {f16, f32, f64, i32, i64}
    "masked_logsumexp": {i64},         # 操作17: "masked_logsumexp"，支持数据类型为 {i64}
    "max_pool2d_with_indices_backward": {f16, f32, f64},       # 操作18: "max_pool2d_with_indices_backward"，支持数据类型为 {f16, f32, f64}
    "new_empty_strided": {f16},        # 操作19: "new_empty_strided"，支持数据类型为 {f16}
    "nn.functional.adaptive_avg_pool3d": {f16},  # 操作20: "nn.functional.adaptive_avg_pool3d"，支持数据类型为 {f16}
    "nn.functional.adaptive_max_pool1d": {f16, f32},  # 操作21: "nn.functional.adaptive_max_pool1d"，支持数据类型为 {f16, f32}
    "nn.functional.adaptive_max_pool2d": {f16, f32},  # 操作22: "nn.functional.adaptive_max_pool2d"，支持数据类型为 {f16, f32}
    "nn.functional.bilinear": {f16},   # 操作23: "nn.functional.bilinear"，支持数据类型为 {f16}
    "nn.functional.conv_transpose1d": {f16},  # 操作24: "nn.functional.conv_transpose1d"，支持数据类型为 {f16}
    "nn.functional.conv_transpose2d": {f16},  # 操作25: "nn.functional.conv_transpose2d"，支持数据类型为 {f16}
    "nn.functional.conv_transpose3d": {f16},  # 操作26: "nn.functional.conv_transpose3d"，支持数据类型为 {f16}
    "nn.functional.cosine_similarity": {f16},  # 操作27: "nn.functional.cosine_similarity"，支持数据类型为 {f16}
    "nn.functional.cross_entropy": {f16, f32, f64},  # 操作28: "nn.functional.cross_entropy"，支持数据类型为 {f16, f32, f64}
    "nn.functional.gaussian_nll_loss": {f16},  # 操作29: "nn.functional.gaussian_nll_loss"，支持数据类型为 {f16}
    "nn.functional.grid_sample": {f32, f64},  # 操作30: "nn.functional.grid_sample"，支持数据类型为 {f32, f64}
    "nn.functional.interpolate.area": {f16},  # 操作31: "nn.functional.interpolate.area"，支持数据类型为 {f16}
    "nn.functional.nll_loss": {f16, f32, f64},  # 操作32: "nn.functional.nll_loss"，支持数据类型为 {f16, f32, f64}
    "normal": {f16, f32, f64},          # 操作33: "normal"，支持数据类型为 {f16, f32, f64}
    "put": {f16, f32, f64},             # 操作34: "put"，支持数据类型为 {f16, f32, f64}
    "take": {b8, f16, f32, f64, i32, i64},  # 操作35: "take"，支持数据类型为 {b8, f16, f32, f64, i32, i64}
    ("__rdiv__", "cuda"): {f16},        # 操作36: "__rdiv__" (在 "cuda" 下)，支持数据类型为 {f16}
    ("__rmod__", "cuda"): {f16, i64},   # 操作37: "__rmod__" (在 "cuda" 下)，支持数据类型为 {f16, i64}
    ("__rmul__", "cuda"): {f16},        # 操作38: "__rmul__" (在 "cuda" 下)，支持数据类型为 {f16}
    ("__rpow__", "cuda"): {f16},        # 操作39: "__rpow__" (在 "cuda" 下)，支持数据类型为 {f16}
    ("_unsafe_masked_index", "cuda"): {f16},  # 操作40: "_unsafe_masked_index" (在 "cuda" 下)，支持数据类型为 {f16}
    ("_unsafe_masked_index_put_accumulate", "cuda"): {f16},  # 操作41: "_unsafe_masked_index_put_accumulate" (在 "cuda" 下)，支持数据类型为 {f16}
    ("addcdiv", "cuda"): {f16},         # 操作42: "addcdiv" (在 "cuda" 下)，支持数据类型为 {f16}
    ("addcmul", "cuda"): {f16},         # 操作43: "addcmul" (在 "cuda" 下)，支持数据类型为 {f16}
    ("atan2", "cuda"): {f16},           # 操作44: "atan2" (在 "cuda" 下)，支持数据类型为 {f16}
    ("cumsum", "cuda"): {f16},          # 操作45: "cumsum" (在 "cuda" 下)，支持数据类型为 {f16}
    ("cumulative_trapezoid", "cuda"): {f16},  # 操作46: "cumulative_trapezoid" (在 "cuda" 下)，支持数据类型为 {f16}
    ("dist", "cuda"): {f16},            # 操作47: "dist" (在 "cuda" 下)，支持数据类型为 {f16}
    ("div.no_rounding_mode", "cuda"): {f16},  # 操作48: "div.no_rounding_mode" (在 "cuda" 下)，支持数据类型为 {f16}
    ("fmod", "cuda"): {f16},            # 操作49: "fmod" (在 "cuda" 下)，支持数据类型为 {f16}
    ("grid_sampler_2d", "cuda"): {f16},  # 操作50: "grid_sampler_2d" (在 "cuda" 下)，支持数据类型为 {f16}
    ("index_fill", "cuda"): {f16, f32, f64},
    # 定义了一组函数名和相应的计算精度集合，指定在 CUDA 上支持的数据类型
    ("nn.functional.cosine_embedding_loss", "cuda"): {f16},
    ("nn.functional.dropout2d", "cuda"): {f16, f32, f64},
    ("nn.functional.dropout3d", "cuda"): {f16, f32, f64},
    ("nn.functional.dropout", "cuda"): {f16, f32, f64},
    ("nn.functional.feature_alpha_dropout.with_train", "cuda"): {f16, f32, f64},
    ("nn.functional.fractional_max_pool2d", "cuda"): {f16, f32, f64},
    ("nn.functional.fractional_max_pool3d", "cuda"): {f16, f32, f64},
    ("nn.functional.grid_sample", "cuda"): {f16},
    ("nn.functional.group_norm", "cuda"): {f16},
    ("nn.functional.hinge_embedding_loss", "cuda"): {f16},
    # 以下函数启用所有测试时，此测试会随机失败
    # 参考 https://github.com/pytorch/pytorch/issues/129238
    ("nn.functional.huber_loss", "cuda"): {f16},
    ("nn.functional.interpolate.bicubic", "cuda"): {f16},
    ("nn.functional.interpolate.bilinear", "cuda"): {f16},
    ("nn.functional.interpolate.trilinear", "cuda"): {f16},
    ("nn.functional.kl_div", "cuda"): {f16},
    ("nn.functional.margin_ranking_loss", "cuda"): {f16},
    ("nn.functional.max_pool1d", "cuda"): {f16, f32, f64},
    ("nn.functional.max_pool3d", "cuda"): {f16},
    ("nn.functional.mse_loss", "cuda"): {f16},
    ("nn.functional.multi_margin_loss", "cuda"): {f16},
    ("nn.functional.multilabel_margin_loss", "cuda"): {f16},
    ("nn.functional.multilabel_soft_margin_loss", "cuda"): {f16},
    ("nn.functional.normalize", "cuda"): {f16},
    ("nn.functional.pad.replicate", "cuda"): {f16, f32, f64},
    ("nn.functional.pad.reflect", "cuda"): {f16},
    ("nn.functional.pairwise_distance", "cuda"): {f16},
    ("nn.functional.poisson_nll_loss", "cuda"): {f16},
    ("nn.functional.rms_norm", "cuda"): {f16},
    ("norm", "cuda"): {f16},
    ("pow", "cuda"): {f16},
    ("prod", "cuda"): {f16},
    ("scatter_reduce.amax", "cuda"): {f16, f32, f64},
    ("scatter_reduce.amin", "cuda"): {f16, f32, f64},
    ("scatter_reduce.mean", "cuda"): {f16, f32, f64},
    ("special.xlog1py", "cuda"): {f16},
    ("std", "cuda"): {f16},
    ("std_mean", "cuda"): {f16},
    ("svd_lowrank", "cuda"): {f32, f64},
    ("trapezoid", "cuda"): {f16},
    ("trapz", "cuda"): {f16},
    ("true_divide", "cuda"): {f16},
    ("var", "cuda"): {f16},
    ("var_mean", "cuda"): {f16},
    ("xlogy", "cuda"): {f16},
# 定义一个装饰器函数，用于收集测试中出现的异常情况
def collection_decorator(fn):
    # 内部函数，实际执行被装饰函数并处理异常
    @functools.wraps(fn)
    def inner(self, device, dtype, op):
        try:
            # 调用被装饰的测试函数
            fn(self, device, dtype, op)
        except Exception as e:
            # 如果预期收集异常，则记录异常信息
            if COLLECT_EXPECT:
                variant = op.variant_test_name
                op_key = op.name if not variant else (op.name, variant)
                device_type = torch.device(device).type
                # 将异常信息添加到已见过的失败列表中
                seen_failed[device_type, op_key].add(dtype)
            # 将捕获到的异常重新抛出
            raise e

    return inner
```
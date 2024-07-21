# `.\pytorch\torch\testing\_internal\inductor_utils.py`

```py
# 忽略类型检查错误，适用于使用 mypy 时
# 导入必要的库和模块
import torch
import re
import unittest
import functools
import os
from subprocess import CalledProcessError
import sys
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._inductor.codecache import CppCodeCache
from torch.utils._triton import has_triton
from torch.testing._internal.common_utils import (
    LazyVal,
    IS_FBCODE,
)
from torch.testing._internal.common_utils import (
    TestCase,
    IS_CI,
    IS_WINDOWS,
)

# 定义测试函数 test_cpu，用于检查是否可以加载 CppCodeCache
def test_cpu():
    try:
        CppCodeCache.load("")
        return not IS_FBCODE
    except (
        CalledProcessError,
        OSError,
        torch._inductor.exc.InvalidCxxCompiler,
        torch._inductor.exc.CppCompileError,
    ):
        return False

# 延迟计算 LazyVal，用于测试 CPU 是否可用
HAS_CPU = LazyVal(test_cpu)

# 检查是否有 CUDA 可用并且 Triton 可用
HAS_CUDA = torch.cuda.is_available() and has_triton()

# 检查是否有 XPU 可用并且 Triton 可用
HAS_XPU = torch.xpu.is_available() and has_triton()

# 判断是否有 GPU 可用（CUDA 或 XPU）
HAS_GPU = HAS_CUDA or HAS_XPU

# GPU 类型列表
GPUS = ["cuda", "xpu"]

# 检查是否有多个 GPU 可用
HAS_MULTIGPU = any(
    getattr(torch, gpu).is_available() and getattr(torch, gpu).device_count() >= 2
    for gpu in GPUS
)

# 筛选可用的 GPU 类型
tmp_gpus = [x for x in GPUS if getattr(torch, x).is_available()]
# 断言最多只有一个 GPU 可用
assert len(tmp_gpus) <= 1
# 设置 GPU 类型变量
GPU_TYPE = "cuda" if len(tmp_gpus) == 0 else tmp_gpus.pop()
# 删除临时 GPU 列表
del tmp_gpus

# 定义用于检查动态形状的私有函数 _check_has_dynamic_shape
def _check_has_dynamic_shape(
    self: TestCase,
    code,
):
    # 初始化标志变量
    for_loop_found = False
    has_dynamic = False
    # 按行分割代码
    lines = code.split("\n")
    # 遍历每一行代码
    for line in lines:
        # 检查是否包含 for 循环关键字
        if "for(" in line:
            for_loop_found = True
            # 如果在 for 循环内部找到动态形状变量（如 ks），则设置标志为 True 并跳出循环
            if re.search(r";.*ks.*;", line) is not None:
                has_dynamic = True
                break
    # 使用 TestCase 断言检查结果
    self.assertTrue(
        has_dynamic, msg=f"Failed to find dynamic for loop variable\n{code}"
    )
    self.assertTrue(for_loop_found, f"Failed to find for loop\n{code}")

# 定义用于跳过特定设备的装饰器函数 skipDeviceIf
def skipDeviceIf(cond, msg, *, device):
    if cond:
        # 如果条件为真，返回装饰后的函数，用于在特定设备上跳过测试
        def decorate_fn(fn):
            def inner(self, *args, **kwargs):
                if self.device == device:
                    raise unittest.SkipTest(msg)
                return fn(self, *args, **kwargs)
            return inner
    else:
        # 如果条件为假，返回原始函数
        def decorate_fn(fn):
            return fn

    return decorate_fn

# functools.partial 创建跳过 CUDA 测试的装饰器 skipCUDAIf
skipCUDAIf = functools.partial(skipDeviceIf, device="cuda")

# functools.partial 创建跳过 XPU 测试的装饰器 skipXPUIf
skipXPUIf = functools.partial(skipDeviceIf, device="xpu")

# functools.partial 创建跳过 CPU 测试的装饰器 skipCPUIf
skipCPUIf = functools.partial(skipDeviceIf, device="cpu")
```
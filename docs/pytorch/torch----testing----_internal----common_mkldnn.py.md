# `.\pytorch\torch\testing\_internal\common_mkldnn.py`

```py
# 忽略 mypy 的错误提示

# 导入必要的库
import contextlib
import functools
import inspect

import torch

# 检查硬件是否启用 BF32 数学模式。仅在以下条件下启用：
# - MKLDNN 可用
# - MKLDNN 支持 BF16
def bf32_is_not_fp32():
    # 检查 MKLDNN 是否可用
    if not torch.backends.mkldnn.is_available():
        return False
    # 检查 MKLDNN 是否支持 BF16
    if not torch.ops.mkldnn._is_mkldnn_bf16_supported():
        return False
    return True

# 创建上下文管理器，用于关闭 BF32 模式
@contextlib.contextmanager
def bf32_off():
    # 保存旧的矩阵乘法精度设置
    old_matmul_precision = torch.get_float32_matmul_precision()
    try:
        # 设置矩阵乘法精度为最高
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        # 恢复旧的矩阵乘法精度设置
        torch.set_float32_matmul_precision(old_matmul_precision)

# 创建上下文管理器，用于启用 BF32 模式
@contextlib.contextmanager
def bf32_on(self, bf32_precision=1e-5):
    # 保存旧的矩阵乘法精度设置和精度设置
    old_matmul_precision = torch.get_float32_matmul_precision()
    old_precision = self.precision
    try:
        # 设置矩阵乘法精度为中等
        torch.set_float32_matmul_precision("medium")
        self.precision = bf32_precision
        yield
    finally:
        # 恢复旧的矩阵乘法精度设置和精度设置
        torch.set_float32_matmul_precision(old_matmul_precision)
        self.precision = old_precision

# 包装器，用于运行测试两次，一次允许 BF32，一次禁用 BF32。在允许 BF32 时，使用指定的降低精度
def bf32_on_and_off(bf32_precision=1e-5):
    def with_bf32_disabled(self, function_call):
        with bf32_off():
            function_call()

    def with_bf32_enabled(self, function_call):
        with bf32_on(self, bf32_precision):
            function_call()

    def wrapper(f):
        params = inspect.signature(f).parameters
        arg_names = tuple(params.keys())

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            for k, v in zip(arg_names, args):
                kwargs[k] = v
            cond = bf32_is_not_fp32()
            if "device" in kwargs:
                cond = cond and (torch.device(kwargs["device"]).type == "cpu")
            if "dtype" in kwargs:
                cond = cond and (kwargs["dtype"] == torch.float)
            if cond:
                with_bf32_disabled(kwargs["self"], lambda: f(**kwargs))
                with_bf32_enabled(kwargs["self"], lambda: f(**kwargs))
            else:
                f(**kwargs)

        return wrapped

    return wrapper
```
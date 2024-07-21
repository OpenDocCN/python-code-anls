# `.\pytorch\torch\_prims\debug_prims.py`

```py
# mypy: allow-untyped-defs
# 导入上下文管理模块
import contextlib
# 导入可选类型
from typing import Optional

# 导入 PyTorch 模块
import torch
# 导入内容存储读取器
from torch.utils._content_store import ContentStoreReader

# 全局变量，用于存储内容存储读取器的实例，初始为 None
LOAD_TENSOR_READER: Optional[ContentStoreReader] = None


# 上下文管理器函数，用于加载张量读取器
@contextlib.contextmanager
def load_tensor_reader(loc):
    global LOAD_TENSOR_READER
    # 断言确保 LOAD_TENSOR_READER 为 None，以确保不会重复加载
    assert LOAD_TENSOR_READER is None
    # 创建 ContentStoreReader 对象，禁用缓存以避免张量别名问题
    LOAD_TENSOR_READER = ContentStoreReader(loc, cache=False)
    try:
        yield
    finally:
        # 离开上下文管理器后，将 LOAD_TENSOR_READER 重新设置为 None
        LOAD_TENSOR_READER = None


# 注册调试基元函数
def register_debug_prims():
    # 定义 torch library 函数 "debugprims::load_tensor"
    torch.library.define(
        "debugprims::load_tensor",
        "(str name, int[] size, int[] stride, *, ScalarType dtype, Device device) -> Tensor",
    )

    # 实现 "debugprims::load_tensor" 的后端选择函数
    @torch.library.impl("debugprims::load_tensor", "BackendSelect")
    def load_tensor_factory(name, size, stride, dtype, device):
        # 如果 LOAD_TENSOR_READER 为 None，则使用随机生成的张量
        if LOAD_TENSOR_READER is None:
            from torch._dynamo.testing import rand_strided

            return rand_strided(size, stride, dtype, device)
        else:
            from torch._dynamo.utils import clone_input

            # 使用 LOAD_TENSOR_READER 读取指定名称的张量
            r = LOAD_TENSOR_READER.read_tensor(name, device=device)
            # 断言确保读取的张量大小与给定的 size 一致
            assert list(r.size()) == size, f"{r.size()} != {size}"
            # 断言确保读取的张量步长与给定的 stride 一致
            assert list(r.stride()) == stride, f"{r.stride()} != {stride}"
            # 断言确保读取的张量设备与给定的 device 一致
            assert r.device == device, f"{r.device} != {device}"

            # 如果读取的张量 dtype 与给定的 dtype 不匹配，则进行类型转换
            if r.dtype != dtype:
                r = clone_input(r, dtype=dtype)
            return r
```
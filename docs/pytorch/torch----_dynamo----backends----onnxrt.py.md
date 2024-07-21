# `.\pytorch\torch\_dynamo\backends\onnxrt.py`

```py
# mypy: ignore-errors

# This backend is maintained by ONNX team. To direct issues
# to the right people, please tag related GitHub issues with `module: onnx`.
#
# Maintainers' Github IDs: wschin, xadupre

# 从torch.onnx._internal.onnxruntime中导入必要的函数和对象
from torch.onnx._internal.onnxruntime import (
    is_onnxrt_backend_supported,  # 导入检查是否支持ONNX Runtime后端的函数
    torch_compile_backend,  # 导入用于编译后端的函数
)

# 从当前目录中的registry模块中导入register_backend函数
from .registry import register_backend


# 检查是否支持ONNX Runtime后端
def has_onnxruntime():
    # FIXME: update test/dynamo/test_backends.py to call is_onnxrt_backend_supported()
    return is_onnxrt_backend_supported()


# 如果支持ONNX Runtime后端，则注册onnxrt后端，并指定编译函数为torch_compile_backend
if is_onnxrt_backend_supported():
    register_backend(name="onnxrt", compiler_fn=torch_compile_backend)
else:
    # 如果不支持ONNX Runtime后端，则定义一个函数information_displaying_backend来抛出ImportError异常
    def information_displaying_backend(*args, **kwargs):
        raise ImportError(
            "onnxrt is not registered as a backend. "
            "Please make sure all dependencies such as "
            "numpy, onnx, onnxscript, and onnxruntime-training are installed. "
            "Suggested procedure to fix dependency problem:\n"
            "  (1) pip or conda install numpy onnx onnxscript onnxruntime-training.\n"
            "  (2) Open a new python terminal.\n"
            "  (3) Call the API `torch.onnx.is_onnxrt_backend_supported()`:\n"
            "  (4)   If it returns `True`, then you can use `onnxrt` backend.\n"
            "  (5)   If it returns `False`, please execute the package importing section in "
            "torch/onnx/_internal/onnxruntime.py under pdb line-by-line to see which import fails."
        )

    # 注册onnxrt后端，并指定编译函数为information_displaying_backend
    register_backend(name="onnxrt", compiler_fn=information_displaying_backend)
```
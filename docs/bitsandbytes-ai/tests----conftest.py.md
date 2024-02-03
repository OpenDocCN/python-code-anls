# `bitsandbytes\tests\conftest.py`

```py
# 导入 pytest 和 torch 模块
import pytest
import torch

# 定义 pytest_runtest_call 函数，用于运行测试用例
def pytest_runtest_call(item):
    # 尝试运行测试用例
    try:
        item.runtest()
    # 捕获 NotImplementedError 异常
    except NotImplementedError as nie:
        # 如果异常信息中包含 "NO_CUBLASLT"，则跳过测试用例
        if "NO_CUBLASLT" in str(nie):
            pytest.skip("CUBLASLT not available")
        # 如果异常信息不包含 "NO_CUBLASLT"，则继续抛出异常
        raise
    # 捕获 AssertionError 异常
    except AssertionError as ae:
        # 如果异常信息为 "Torch not compiled with CUDA enabled"，则跳过测试用例
        if str(ae) == "Torch not compiled with CUDA enabled":
            pytest.skip("Torch not compiled with CUDA enabled")
        # 如果异常信息不为 "Torch not compiled with CUDA enabled"，则继续抛出异常
        raise

# 定义 requires_cuda 装饰器，用于检查是否支持 CUDA
@pytest.fixture(scope="session")
def requires_cuda() -> bool:
    # 检查是否支持 CUDA
    cuda_available = torch.cuda.is_available()
    # 如果不支持 CUDA，则跳过测试用例
    if not cuda_available:
        pytest.skip("CUDA is required")
    # 返回 CUDA 是否可用的布尔值
    return cuda_available
```
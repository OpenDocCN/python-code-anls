# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_check_build.py`

```
"""
Smoke Test the check_build module
"""

# 引入 pytest 模块，用于单元测试和异常处理
import pytest

# 从 sklearn.__check_build 模块中引入 raise_build_error 函数
from sklearn.__check_build import raise_build_error

# 定义测试函数 test_raise_build_error，用于测试 raise_build_error 函数的异常处理功能
def test_raise_build_error():
    # 使用 pytest.raises 上下文管理器捕获 ImportError 异常
    with pytest.raises(ImportError):
        # 调用 raise_build_error 函数，传入 ImportError 异常作为参数
        raise_build_error(ImportError())
```
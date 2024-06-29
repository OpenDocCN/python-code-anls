# `D:\src\scipysrc\pandas\pandas\tests\util\test_numba.py`

```
# 导入pytest库，用于编写和运行测试
import pytest

# 导入pandas的测试装饰器模块
import pandas.util._test_decorators as td

# 从pandas库中导入option_context函数，用于临时设置选项
from pandas import option_context

# 使用装饰器td.skip_if_installed("numba")，如果numba已安装，则跳过该测试函数
@td.skip_if_installed("numba")
# 定义名为test_numba_not_installed_option_context的测试函数
def test_numba_not_installed_option_context():
    # 使用pytest.raises检查是否引发ImportError异常，并检查异常消息中是否包含"Missing optional"
    with pytest.raises(ImportError, match="Missing optional"):
        # 使用option_context临时设置"compute.use_numba"选项为True
        with option_context("compute.use_numba", True):
            # 在此代码块中不执行任何操作，只是验证异常处理和选项设置
            pass
```
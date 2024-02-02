# `ZeroNet\src\Test\TestSite.py`

```py
# 导入shutil和os模块
import shutil
import os

# 导入pytest模块和Site模块中的SiteManager类
import pytest
from Site import SiteManager

# 定义测试数据路径常量
TEST_DATA_PATH = "src/Test/testdata"

# 使用pytest的usefixtures装饰器，重置设置
@pytest.mark.usefixtures("resetSettings")
class TestSite:
```
# `D:\src\scipysrc\pandas\pandas\tests\io\json\conftest.py`

```
import pytest  # 导入 pytest 模块，用于单元测试和测试驱动开发


@pytest.fixture(params=["split", "records", "index", "columns", "values"])
def orient(request):
    """
    Fixture for orients excluding the table format.
    """
    return request.param
    # 返回一个参数化的 fixture，用于提供不同的数据格式方向，不包括表格格式
```
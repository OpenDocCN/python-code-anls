# `.\DB-GPT-src\dbgpt\core\interface\tests\conftest.py`

```py
# 导入 pytest 模块，用于编写和运行测试用例
import pytest

# 从 dbgpt.core.interface.storage 模块中导入 InMemoryStorage 类
# InMemoryStorage 类用于提供内存存储功能
from dbgpt.core.interface.storage import InMemoryStorage

# 从 dbgpt.util.serialization.json_serialization 模块中导入 JsonSerializer 类
# JsonSerializer 类用于提供 JSON 序列化和反序列化功能
from dbgpt.util.serialization.json_serialization import JsonSerializer


# 定义一个 pytest 的 fixture（测试夹具）函数，用于返回一个 JsonSerializer 对象
@pytest.fixture
def serializer():
    return JsonSerializer()


# 定义一个 pytest 的 fixture 函数，用于返回一个带有 JsonSerializer 对象的 InMemoryStorage 对象
@pytest.fixture
def in_memory_storage(serializer):
    return InMemoryStorage(serializer)
```
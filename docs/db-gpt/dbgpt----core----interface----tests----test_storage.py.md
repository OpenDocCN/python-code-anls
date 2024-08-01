# `.\DB-GPT-src\dbgpt\core\interface\tests\test_storage.py`

```py
# 导入所需的类型和模块
from typing import Dict, Type, Union

import pytest  # 导入 pytest 测试框架

# 导入核心存储接口相关模块和类
from dbgpt.core.interface.storage import (
    InMemoryStorage,
    QuerySpec,
    ResourceIdentifier,
    StorageError,
    StorageItem,
)

# 导入 JSON 序列化工具类
from dbgpt.util.serialization.json_serialization import JsonSerializer


# 定义模拟资源标识符类，继承自 ResourceIdentifier
class MockResourceIdentifier(ResourceIdentifier):
    def __init__(self, identifier: str):
        self._identifier = identifier

    @property
    def str_identifier(self) -> str:
        return self._identifier

    def to_dict(self) -> Dict:
        return {"identifier": self._identifier}


# 定义模拟存储项类，继承自 StorageItem
class MockStorageItem(StorageItem):
    def merge(self, other: "StorageItem") -> None:
        # 合并存储项数据，要求 other 必须是 MockStorageItem 类型
        if not isinstance(other, MockStorageItem):
            raise ValueError("other must be a MockStorageItem")
        self.data = other.data

    def __init__(self, identifier: Union[str, MockResourceIdentifier], data):
        # 初始化存储项，根据 identifier 类型初始化标识符字符串
        self._identifier_str = (
            identifier if isinstance(identifier, str) else identifier.str_identifier
        )
        self.data = data

    def to_dict(self) -> Dict:
        # 将存储项转换为字典形式，包括标识符和数据
        return {"identifier": self._identifier_str, "data": self.data}

    @property
    def identifier(self) -> ResourceIdentifier:
        # 返回存储项的标识符对象
        return MockResourceIdentifier(self._identifier_str)


# 定义 pytest fixture，返回 JsonSerializer 实例
@pytest.fixture
def serializer():
    return JsonSerializer()


# 定义 pytest fixture，返回 InMemoryStorage 实例，依赖于 serializer fixture
@pytest.fixture
def in_memory_storage(serializer):
    return InMemoryStorage(serializer)


# 测试函数：测试保存和加载功能
def test_save_and_load(in_memory_storage):
    resource_id = MockResourceIdentifier("1")
    item = MockStorageItem(resource_id, "test_data")

    in_memory_storage.save(item)

    loaded_item = in_memory_storage.load(resource_id, MockStorageItem)
    assert loaded_item.data == "test_data"


# 测试函数：测试重复保存相同数据时是否会引发 StorageError 异常
def test_duplicate_save(in_memory_storage):
    item = MockStorageItem("1", "test_data")

    in_memory_storage.save(item)

    with pytest.raises(StorageError):
        in_memory_storage.save(item)


# 测试函数：测试删除功能
def test_delete(in_memory_storage):
    resource_id = MockResourceIdentifier("1")
    item = MockStorageItem(resource_id, "test_data")

    in_memory_storage.save(item)
    in_memory_storage.delete(resource_id)

    # 删除后存储中不应包含数据
    assert in_memory_storage.load(resource_id, MockStorageItem) is None


# 测试函数：测试查询功能
def test_query(in_memory_storage):
    resource_id1 = MockResourceIdentifier("1")
    item1 = MockStorageItem(resource_id1, "test_data1")

    resource_id2 = MockResourceIdentifier("2")
    item2 = MockStorageItem(resource_id2, "test_data2")

    in_memory_storage.save(item1)
    in_memory_storage.save(item2)

    # 设置查询条件
    query_spec = QuerySpec(conditions={"data": "test_data1"})
    results = in_memory_storage.query(query_spec, MockStorageItem)

    # 断言查询结果数量和数据内容
    assert len(results) == 1
    assert results[0].data == "test_data1"


# 测试函数：测试计数功能
def test_count(in_memory_storage):
    item1 = MockStorageItem("1", "test_data1")
    item2 = MockStorageItem("2", "test_data2")

    in_memory_storage.save(item1)
    # 此处未完成的测试示例，可以根据需要补充测试代码
    # 将 item2 对象保存到内存存储中
    in_memory_storage.save(item2)
    
    # 创建一个空的查询规范对象 query_spec，用于构建查询条件
    query_spec = QuerySpec(conditions={})
    
    # 使用 in_memory_storage 对象执行查询，统计 MockStorageItem 类型的记录数量，并将结果存储在 count 变量中
    count = in_memory_storage.count(query_spec, MockStorageItem)
    
    # 断言查询结果 count 应该等于 2，用于验证查询操作的正确性
    assert count == 2
# 定义一个用于测试分页查询的函数，接受一个内存存储对象作为参数
def test_paginate_query(in_memory_storage):
    # 循环生成并保存模拟存储项，每个项都有一个唯一的资源标识符和测试数据
    for i in range(10):
        resource_id = MockResourceIdentifier(str(i))
        item = MockStorageItem(resource_id, f"test_data{i}")
        in_memory_storage.save(item)

    # 设置每页显示的项数
    page_size = 3
    # 创建一个空的查询规范对象
    query_spec = QuerySpec(conditions={})
    # 调用内存存储对象的分页查询方法，获取第二页的结果
    page_result = in_memory_storage.paginate_query(
        2, page_size, MockStorageItem, query_spec
    )

    # 断言：验证返回的页面结果中项的数量等于每页的项数
    assert len(page_result.items) == page_size
    # 断言：验证总共的项数为 10
    assert page_result.total_count == 10
    # 断言：验证总页数为 4
    assert page_result.total_pages == 4
    # 断言：验证当前页数为第 2 页
    assert page_result.page == 2
```
# `.\DB-GPT-src\dbgpt\storage\cache\storage\tests\test_storage.py`

```py
# 从 dbgpt.util.memory_utils 模块中导入 _get_object_bytes 函数
# 从 ..base 模块中导入 StorageItem 类
from dbgpt.util.memory_utils import _get_object_bytes
from ..base import StorageItem

# 定义一个测试函数 test_build_from，用于测试 StorageItem 类的 build_from 方法
def test_build_from():
    # 定义测试用的 key_hash, key_data, value_data
    key_hash = b"key_hash"
    key_data = b"key_data"
    value_data = b"value_data"
    # 调用 StorageItem 类的 build_from 方法，创建一个 StorageItem 对象 item
    item = StorageItem.build_from(key_hash, key_data, value_data)

    # 断言 item 的属性与预期值相等
    assert item.key_hash == key_hash
    assert item.key_data == key_data
    assert item.value_data == value_data
    # 断言 item 的 length 属性符合预期计算结果
    assert item.length == 32 + _get_object_bytes(key_hash) + _get_object_bytes(
        key_data
    ) + _get_object_bytes(value_data)

# 定义一个测试函数 test_build_from_kv，用于测试 StorageItem 类的 build_from_kv 方法
def test_build_from_kv():
    # 定义一个 MockCacheKey 类，用于模拟键对象
    class MockCacheKey:
        # 定义 get_hash_bytes 方法，返回模拟的 key_hash
        def get_hash_bytes(self):
            return b"key_hash"

        # 定义 serialize 方法，返回模拟的 key_data
        def serialize(self):
            return b"key_data"

    # 定义一个 MockCacheValue 类，用于模拟值对象
    class MockCacheValue:
        # 定义 serialize 方法，返回模拟的 value_data
        def serialize(self):
            return b"value_data"

    # 创建 MockCacheKey 和 MockCacheValue 的实例 key 和 value
    key = MockCacheKey()
    value = MockCacheValue()
    # 调用 StorageItem 类的 build_from_kv 方法，创建一个 StorageItem 对象 item
    item = StorageItem.build_from_kv(key, value)

    # 断言 item 的属性与预期值相等
    assert item.key_hash == key.get_hash_bytes()
    assert item.key_data == key.serialize()
    assert item.value_data == value.serialize()

# 定义一个测试函数 test_serialize_deserialize，用于测试 StorageItem 类的 serialize 和 deserialize 方法
def test_serialize_deserialize():
    # 定义测试用的 key_hash, key_data, value_data
    key_hash = b"key_hash"
    key_data = b"key_data"
    value_data = b"value_data"
    # 调用 StorageItem 类的 build_from 方法，创建一个 StorageItem 对象 item
    item = StorageItem.build_from(key_hash, key_data, value_data)

    # 将 item 进行序列化
    serialized = item.serialize()
    # 将序列化后的数据进行反序列化，得到一个新的 StorageItem 对象 deserialized
    deserialized = StorageItem.deserialize(serialized)

    # 断言 deserialized 的属性与原始 item 对象的属性相等
    assert deserialized.key_hash == item.key_hash
    assert deserialized.key_data == item.key_data
    assert deserialized.value_data == item.value_data
    assert deserialized.length == item.length
```
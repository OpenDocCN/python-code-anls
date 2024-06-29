# `.\numpy\numpy\_core\tests\test_hashtable.py`

```
# 导入 pytest 模块，用于测试
import pytest

# 导入 random 模块，用于生成随机数和随机选择
import random

# 导入特定的测试函数 identityhash_tester，用于测试哈希表功能
from numpy._core._multiarray_tests import identityhash_tester

# 使用 pytest 的参数化装饰器，定义测试函数的参数组合
@pytest.mark.parametrize("key_length", [1, 3, 6])
@pytest.mark.parametrize("length", [1, 16, 2000])
def test_identity_hashtable(key_length, length):
    # 创建一个包含 20 个对象的对象池列表，用于生成随机的 keys_vals 对
    pool = [object() for i in range(20)]
    
    keys_vals = []
    # 生成 length 次循环，每次生成一个 key_length 长度的 keys 元组和对应的随机值，加入 keys_vals 列表中
    for i in range(length):
        keys = tuple(random.choices(pool, k=key_length))
        keys_vals.append((keys, random.choice(pool)))

    # 根据 keys_vals 列表生成字典 dictionary
    dictionary = dict(keys_vals)

    # 将 keys_vals 列表中的随机项再添加一次，用于后续测试
    keys_vals.append(random.choice(keys_vals))
    # 预期的结果 expected 是 keys_vals 列表中最后一项对应的值
    expected = dictionary[keys_vals[-1][0]]

    # 调用 identityhash_tester 函数进行测试，预期结果应与 expected 相同
    res = identityhash_tester(key_length, keys_vals, replace=True)
    assert res is expected

    # 如果 length 为 1，则直接返回，不再进行后续操作
    if length == 1:
        return

    # 尝试添加一个新的项，其键已经在 keys_vals 中存在，并且值是新的对象，如果 replace 参数为 False，则应引发 RuntimeError 异常，用于验证功能是否正常
    new_key = (keys_vals[1][0], object())
    keys_vals[0] = new_key
    with pytest.raises(RuntimeError):
        identityhash_tester(key_length, keys_vals)
```
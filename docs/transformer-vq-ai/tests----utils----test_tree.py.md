# `transformer_vq\tests\utils\test_tree.py`

```
# 从transformer_vq.utils.tree模块中导入flattened_traversal函数
from transformer_vq.utils.tree import flattened_traversal

# 定义测试函数test_flattened_traversal
def test_flattened_traversal():
    # 定义一个函数func，用于处理键值对，返回拼接后的字符串
    def func(k, v):
        return f"{'.'.join(k)}_{v}"

    # 创建一个字典pytree1
    pytree1 = {"a": 1, "b": 2, "c": 3}
    # 调用flattened_traversal函数，对pytree1进行处理
    actual = flattened_traversal(func)(pytree1)
    # 期望的处理结果
    expected = {"a": "a_1", "b": "b_2", "c": "c_3"}
    # 断言实际处理结果与期望结果相等
    assert actual == expected

    # 创建一个嵌套字典pytree2
    pytree2 = {"a": 1, "b": 2, "c": {"d": {"e": 3}}}
    # 调用flattened_traversal函数，对pytree2进行处理
    actual = flattened_traversal(func)(pytree2)
    # 期望的处理结果
    expected = {"a": "a_1", "b": "b_2", "c": {"d": {"e": "c.d.e_3"}}}
    # 断言实际处理结果与期望结果相等
    assert actual == expected
```
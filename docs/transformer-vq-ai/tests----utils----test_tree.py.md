# `transformer_vq\tests\utils\test_tree.py`

```py
# 从transformer_vq.utils.tree模块中导入flattened_traversal函数
from transformer_vq.utils.tree import flattened_traversal

# 定义测试函数test_flattened_traversal
def test_flattened_traversal():
    # 定义一个函数func，接受键和值作为参数，返回由键和值组成的字符串
    def func(k, v):
        return f"{'.'.join(k)}_{v}"

    # 创建一个字典pytree1
    pytree1 = {"a": 1, "b": 2, "c": 3}
    # 调用flattened_traversal(func)函数，对pytree1进行处理，返回结果存储在actual中
    actual = flattened_traversal(func)(pytree1)
    # 创建一个期望的结果字典expected
    expected = {"a": "a_1", "b": "b_2", "c": "c_3"}
    # 断言actual和expected是否相等
    assert actual == expected

    # 创建一个字典pytree2
    pytree2 = {"a": 1, "b": 2, "c": {"d": {"e": 3}}}
    # 调用flattened_traversal(func)函数，对pytree2进行处理，返回结果存储在actual中
    actual = flattened_traversal(func)(pytree2)
    # 创建一个期望的结果字典expected
    expected = {"a": "a_1", "b": "b_2", "c": {"d": {"e": "c.d.e_3"}}}
    # 断言actual和expected是否相等
    assert actual == expected
```
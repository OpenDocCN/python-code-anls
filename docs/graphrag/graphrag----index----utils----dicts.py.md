# `.\graphrag\graphrag\index\utils\dicts.py`

```py
# 版权所有 (c) 2024 微软公司。
# 根据 MIT 许可证授权

"""一个包含用于检查和验证字典类型的方法的实用模块。"""

# 定义一个函数，用于检查字典是否包含指定的键和对应的类型
def dict_has_keys_with_types(
    data: dict, expected_fields: list[tuple[str, type]]
) -> bool:
    """如果给定的字典包含指定键且类型正确，则返回 True。"""
    # 遍历期望的字段及其类型
    for field, field_type in expected_fields:
        # 检查字段是否存在于数据字典中
        if field not in data:
            return False

        # 获取字段对应的值
        value = data[field]
        # 检查值的类型是否符合期望的字段类型
        if not isinstance(value, field_type):
            return False
    # 如果所有字段检查通过，则返回 True
    return True
```
# `D:\src\scipysrc\scipy\scipy\_build_utils\system_info.py`

```
# 组合多个字典，遵循 Numpy 的 distutils 风格库配置字典的方式

def combine_dict(*dicts, **kw):
    """
    Combine Numpy distutils style library configuration dictionaries.

    Parameters
    ----------
    *dicts
        Dictionaries of keys. List-valued keys will be concatenated.
        Otherwise, duplicate keys with different values result to
        an error. The input arguments are not modified.
    **kw
        Keyword arguments are treated as an additional dictionary
        (the first one, i.e., prepended).

    Returns
    -------
    combined
        Dictionary with combined values.
    """
    
    # 创建一个空字典，用于存储合并后的结果
    new_dict = {}

    # 遍历所有输入的字典和关键字参数
    for d in (kw,) + dicts:
        # 遍历当前字典的键值对
        for key, value in d.items():
            # 如果新字典中已存在相同的键
            if new_dict.get(key, None) is not None:
                # 获取旧值
                old_value = new_dict[key]
                # 如果值是列表或元组类型
                if isinstance(value, (list, tuple)):
                    if isinstance(old_value, (list, tuple)):
                        # 将新旧值的列表合并
                        new_dict[key] = list(old_value) + list(value)
                        continue
                # 如果值相同则继续，否则引发冲突错误
                elif value == old_value:
                    continue

                # 如果出现冲突，抛出 ValueError 异常
                raise ValueError(f"Conflicting configuration dicts: {new_dict!r} {d!r}")
            else:
                # 将新键值对添加到结果字典中
                new_dict[key] = value

    # 返回合并后的字典
    return new_dict
```
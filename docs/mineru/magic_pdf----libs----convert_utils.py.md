# `.\MinerU\magic_pdf\libs\convert_utils.py`

```
# 将字典转换为列表
def dict_to_list(input_dict):
    # 初始化一个空列表，用于存放字典的值
    items_list = []
    # 遍历字典中的每个键值对
    for _, item in input_dict.items():
        # 将字典中的值添加到列表中
        items_list.append(item)
    # 返回包含字典值的列表
    return items_list
```
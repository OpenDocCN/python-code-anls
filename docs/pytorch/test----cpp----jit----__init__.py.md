# `.\pytorch\test\cpp\jit\__init__.py`

```
# 定义一个名为 add_to_dict 的函数，接收两个参数：dict_obj 和 key_value
def add_to_dict(dict_obj, key_value):
    # 将 key_value 的键与值添加到 dict_obj 字典中
    dict_obj[key_value[0]] = key_value[1]

# 创建一个空字典 my_dict
my_dict = {}

# 调用 add_to_dict 函数，将 ('apple', 1) 添加到 my_dict 中
add_to_dict(my_dict, ('apple', 1))

# 调用 add_to_dict 函数，将 ('orange', 4) 添加到 my_dict 中
add_to_dict(my_dict, ('orange', 4))

# 输出 my_dict 字典的内容
print(my_dict)
```
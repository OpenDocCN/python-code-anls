# `.\pytorch\torch\jit\_pickle.py`

```
# 设置一个类型检查的标志，允许在未标注类型的函数中使用
# 这些函数被引用自由 ScriptModule.save() 生成的 pickle 存档中

# 下面的函数 (`build_*`) 曾被 `pickler.cpp` 使用来为某些特殊类型的列表指定类型，
# 但现在所有列表都会通过下面的 `restore_type_tag` 函数附加和恢复类型。
# 这些旧函数为了向后兼容性而保留。

# 接收一个数据参数并直接返回它
def build_intlist(data):
    return data

# 接收一个数据参数并直接返回它
def build_tensorlist(data):
    return data

# 接收一个数据参数并直接返回它
def build_doublelist(data):
    return data

# 接收一个数据参数并直接返回它
def build_boollist(data):
    return data

# 如果数据是整数类型，直接返回该整数，表示这只是一个标识符，无法进行其他操作
def build_tensor_from_id(data):
    if isinstance(data, int):
        return data

# 返回原始值，这个函数在 Python 中是不需要为了 JIT 反序列化器而恢复完整静态类型信息的 type_ptr
def restore_type_tag(value, type_str):
    return value
```
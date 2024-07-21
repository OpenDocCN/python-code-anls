# `.\pytorch\benchmarks\serialization\nested_annotation_str.py`

```py
import torch
import torch.utils.benchmark as benchmark

# 全局变量，用于缓存不同层级嵌套字典的类型
MEMO = {}

# 定义一个函数，创建指定层级嵌套字典的类型
def create_nested_dict_type(layers):
    # 如果层级为0，返回字符串类型
    if layers == 0:
        return torch._C.StringType.get()
    # 如果层级不在缓存中，则递归创建更少层级嵌套的字典类型并缓存
    if layers not in MEMO:
        less_nested = create_nested_dict_type(layers - 1)
        # 创建包含键为字符串类型，值为元组类型（包含两个less_nested类型）的字典类型
        result = torch._C.DictType(
            torch._C.StringType.get(), torch._C.TupleType([less_nested, less_nested])
        )
        MEMO[layers] = result
    # 返回缓存中的结果
    return MEMO[layers]

# 指定不同层级的嵌套字典类型的层级数
nesting_levels = (1, 3, 5, 10)
# 分别创建四种不同层级嵌套的字典类型并存储在types列表中
types = (reasonable, medium, big, huge) = [
    create_nested_dict_type(x) for x in nesting_levels
]

# 创建基准测试的计时器列表，每个计时器测试对应类型的annotation_str属性的性能
timers = [
    benchmark.Timer(stmt="x.annotation_str", globals={"x": nested_type})
    for nested_type in types
]

# 遍历不同层级嵌套字典类型及其对应的计时器
for nesting_level, typ, timer in zip(nesting_levels, types, timers):
    # 输出当前层级嵌套的字典类型的层级数
    print("Nesting level:", nesting_level)
    # 输出当前字典类型的annotation_str属性的前70个字符
    print("output:", typ.annotation_str[:70])
    # 执行计时器的性能测试，并打印结果
    print(timer.blocked_autorange())
```
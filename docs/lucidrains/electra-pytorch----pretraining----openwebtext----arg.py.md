# `.\lucidrains\electra-pytorch\pretraining\openwebtext\arg.py`

```
# 导入必要的模块
import argparse
import dataclasses

# 定义公开的类
__all__ = ('Arg', 'Int', 'Float', 'Bool', 'Str', 'Choice', 'parse_to')

# 定义参数类
class Arg:
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

# 定义整数参数类
class Int(Arg):
    def __init__(self, **kwargs):
        super().__init__(type=int, **kwargs)

# 定义浮点数参数类
class Float(Arg):
    def __init__(self, **kwargs):
        super().__init__(type=float, **kwargs)

# 定义布尔参数类
class Bool(Arg):
    def __init__(self, **kwargs):
        super().__init__(type=bool, **kwargs)

# 定义字符串参数类
class Str(Arg):
    def __init__(self, **kwargs):
        super().__init__(type=str, **kwargs)

# 定义选择参数类
class _MetaChoice(type):
    def __getitem__(self, item):
        return self(choices=list(item), type=item)

# 定义选择参数类
class Choice(Arg, metaclass=_MetaChoice):
    def __init__(self, choices, **kwargs):
        super().__init__(choices=choices, **kwargs)

# 解析参数并填充到指定的容器类中
def parse_to(container_class, **kwargs):
    # 将字段名转换为命令行参数格式
    def mangle_name(name):
        return '--' + name.replace('_', '-')

    # 创建参数解析器
    parser = argparse.ArgumentParser(description=container_class.__doc__)
    # 遍历容器类的字段
    for field in dataclasses.fields(container_class):
        name = field.name
        default = field.default
        value_or_class = field.type
        # 如果字段类型是类，则使用默认值创建实例
        if isinstance(value_or_class, type):
            value = value_or_class(default=default)
        else:
            value = value_or_class
            value.kwargs['default'] = default
        # 添加参数到参数解析器
        parser.add_argument(
            mangle_name(name), **value.kwargs)

    # 解析参数并返回填充后的容器类实例
    arg_dict = parser.parse_args(**kwargs)
    return container_class(**vars(arg_dict))
```
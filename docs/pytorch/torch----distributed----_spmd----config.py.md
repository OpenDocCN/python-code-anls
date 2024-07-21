# `.\pytorch\torch\distributed\_spmd\config.py`

```
# 设置类型提示，允许未标注类型的函数定义
# import logging 模块，用于记录日志
# import sys 模块，提供对 Python 解释器的访问
# from types 模块导入 ModuleType 类型，用于创建模块对象
# from typing 模块导入 Set 类型，用于定义集合类型

# 日志级别设定为 DEBUG，表示打印所有跟踪信息
log_level: int = logging.DEBUG
# 是否启用详细模式，默认为 False
verbose = False

# 日志文件名，默认为 None，表示不将日志写入文件
log_file_name: None = None


# 自定义模块类型 _AccessLimitingConfig，用于限制设置模块属性的访问权限
class _AccessLimitingConfig(ModuleType):
    def __setattr__(self, name, value) -> None:
        # 如果设置的属性名不在允许的配置名集合中，则抛出 AttributeError
        if name not in _allowed_config_names:
            raise AttributeError(f"{__name__}.{name} does not exist")
        # 调用父类的 __setattr__ 方法设置属性
        return object.__setattr__(self, name, value)

# 初始化允许的配置名集合为当前全局变量的键集合
_allowed_config_names: Set[str] = {*globals().keys()}
# 将当前模块的 __class__ 属性设置为 _AccessLimitingConfig 类型，以限制属性设置权限
sys.modules[__name__].__class__ = _AccessLimitingConfig
```
# `.\pytorch\torch\_export\serde\union.py`

```
# mypy: allow-untyped-defs
# 导入 functools 库，用于支持 lru_cache 缓存
import functools
# 从 dataclasses 库中导入 fields 函数，用于获取类的字段信息
from dataclasses import fields
# 导入 Hashable 和 Set 类型提示
from typing import Hashable, Set


# 定义一个特殊的字符串类 _UnionTag，继承自 str 类
class _UnionTag(str):
    _cls: Hashable  # 类型标识符

    # 静态方法，用于创建 _UnionTag 实例
    @staticmethod
    def create(t, cls):
        # 创建一个 _UnionTag 实例，并设置 _cls 属性为 cls
        tag = _UnionTag(t)
        assert not hasattr(tag, "_cls")  # 断言确保 _cls 属性未定义
        tag._cls = cls
        return tag

    # 定义相等性比较方法，确保与字符串的比较，并验证是否为有效的标签
    def __eq__(self, cmp) -> bool:
        assert isinstance(cmp, str)  # 断言比较对象为字符串类型
        other = str(cmp)
        # 验证比较对象是否为有效的类标签
        assert other in _get_field_names(
            self._cls
        ), f"{other} is not a valid tag for {self._cls}. Available tags: {_get_field_names(self._cls)}"
        return str(self) == other

    # 定义哈希方法，使用字符串的哈希值
    def __hash__(self):
        return hash(str(self))


# 使用 functools.lru_cache 装饰器，为 _get_field_names 函数添加缓存功能
@functools.lru_cache(maxsize=None)
# 获取给定类的字段名集合的函数
def _get_field_names(cls) -> Set[str]:
    return {f.name for f in fields(cls)}


# 定义一个特殊的类 _Union
class _Union:
    _type: _UnionTag  # 类型标签

    # 类方法，用于创建 _Union 实例
    @classmethod
    def create(cls, **kwargs):
        assert len(kwargs) == 1  # 断言确保传入的参数只有一个
        # 使用类的字段信息创建实例，type: ignore[arg-type] 用于忽略类型检查
        obj = cls(**{**{f.name: None for f in fields(cls)}, **kwargs})  # type: ignore[arg-type]
        # 创建 _UnionTag 类型的标签并赋值给 obj._type 属性
        obj._type = _UnionTag.create(next(iter(kwargs.keys())), cls)
        return obj

    # 初始化后的方法，用于验证特定字段不在对象中
    def __post_init__(self):
        assert not any(f.name in ("type", "_type", "create", "value") for f in fields(self))  # type: ignore[arg-type, misc]

    # 属性方法，返回对象的类型标签
    @property
    def type(self) -> str:
        try:
            return self._type
        except AttributeError as e:
            # 如果 _type 属性未定义，则引发运行时错误
            raise RuntimeError(
                f"Please use {type(self).__name__}.create to instantiate the union type."
            ) from e

    # 属性方法，返回对象的值
    @property
    def value(self):
        return getattr(self, self.type)

    # 获取属性方法，用于获取对象的属性
    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        # 如果属性值为 None 且在对象的字段名集合中存在且不等于 type，则引发属性错误
        if attr is None and name in _get_field_names(type(self)) and name != self.type:  # type: ignore[arg-type]
            raise AttributeError(f"Field {name} is not set.")
        return attr

    # 字符串表示方法，返回对象的字符串表示形式
    def __str__(self):
        return self.__repr__()

    # repr 方法，返回对象的详细字符串表示形式
    def __repr__(self):
        return f"{type(self).__name__}({self.type}={getattr(self, self.type)})"
```
# `.\pytorch\torch\_dynamo\variables\lazy.py`

```
# mypy: ignore-errors
# 导入必要的模块
import collections
import functools
from typing import Optional

# 从本地模块导入 VariableTracker 类
from .base import VariableTracker

# LazyCache 类定义，用于延迟缓存真实的 VariableTracker 对象
class LazyCache:
    """Container to cache the real VariableTracker"""

    def __init__(self, value, source):
        # 确保 source 参数存在
        assert source
        # 初始化值和来源
        self.value = value
        self.source = source
        # 初始化真实 VariableTracker 对象为 None
        self.vt: Optional[VariableTracker] = None

    def realize(self):
        # 确保尚未实例化真实的 VariableTracker 对象
        assert self.vt is None
        # 导入必要的模块
        from ..symbolic_convert import InstructionTranslator
        from .builder import VariableBuilder

        # 获取当前指令转换器对象
        tx = InstructionTranslator.current_tx()
        # 使用 VariableBuilder 创建真实的 VariableTracker 对象
        self.vt = VariableBuilder(tx, self.source)(self.value)

        # 删除不再需要的 value 和 source 属性
        del self.value
        del self.source

# LazyVariableTracker 类定义，延迟创建真实 VariableTracker 对象
class LazyVariableTracker(VariableTracker):
    """
    A structure that defers the creation of the actual VariableTracker
    for a given underlying value until it is accessed.

    The `realize` function invokes VariableBuilder to produce the real object.
    Once a LazyVariableTracker has been realized, internal bookkeeping will
    prevent double realization.

    This object should be utilized for processing containers, or objects that
    reference other objects where we may not want to take on creating all the
    VariableTrackers right away.
    """

    # _nonvar_fields 属性定义，指定 LazyVariableTracker 的非变量字段
    _nonvar_fields = {"_cache", *VariableTracker._nonvar_fields}

    @staticmethod
    def create(value, source, **options):
        # 创建 LazyVariableTracker 的静态方法，返回 LazyVariableTracker 对象
        return LazyVariableTracker(LazyCache(value, source), source=source, **options)

    def __init__(self, _cache, **kwargs):
        # 确保 _cache 参数是 LazyCache 类的实例
        assert isinstance(_cache, LazyCache)
        # 调用父类 VariableTracker 的初始化方法
        super().__init__(**kwargs)
        # 初始化 _cache 属性
        self._cache = _cache

    def realize(self) -> VariableTracker:
        """Force construction of the real VariableTracker"""
        # 如果尚未实例化真实的 VariableTracker 对象，则调用 LazyCache 的 realize 方法实例化它
        if self._cache.vt is None:
            self._cache.realize()
        return self._cache.vt

    def unwrap(self):
        """Return the real VariableTracker if it already exists"""
        # 如果已经实例化了真实的 VariableTracker 对象，则返回它
        if self.is_realized():
            return self._cache.vt
        # 否则返回当前 LazyVariableTracker 对象
        return self

    def is_realized(self):
        # 检查是否已经实例化了真实的 VariableTracker 对象
        return self._cache.vt is not None

    def clone(self, **kwargs):
        # 确保 _cache 参数未更改
        assert kwargs.get("_cache", self._cache) is self._cache
        # 如果 source 参数已更改，则调用 realize 方法创建真实的 VariableTracker 对象
        if kwargs.get("source", self.source) is not self.source:
            self.realize()
        # 调用父类 VariableTracker 的 clone 方法
        return VariableTracker.clone(self.unwrap(), **kwargs)

    def __str__(self):
        # 如果已经实例化了真实的 VariableTracker 对象，则返回其字符串表示
        if self.is_realized():
            return self.unwrap().__str__()
        # 否则返回当前 LazyVariableTracker 对象的字符串表示
        return VariableTracker.__str__(self.unwrap())

    def __getattr__(self, item):
        # 委托获取属性的操作给真实的 VariableTracker 对象
        return getattr(self.realize(), item)

    # 大多数方法在这里自动生成，这里列出我们要排除的方法
    visit = VariableTracker.visit
    __repr__ = VariableTracker.__repr__

    @classmethod
    def realize_all(
        cls,
        value,
        cache=None,
    ):
        """
        Walk an object and realize all LazyVariableTrackers inside it.
        """
        # 如果缓存为空，则初始化一个空字典
        if cache is None:
            cache = dict()

        # 获取对象的唯一标识符
        idx = id(value)
        # 如果对象已经在缓存中，则直接返回缓存中的结果
        if idx in cache:
            return cache[idx][0]

        # 获取对象的类
        value_cls = type(value)
        # 如果对象是 LazyVariableTracker 的子类，则递归实现其内容
        if issubclass(value_cls, LazyVariableTracker):
            result = cls.realize_all(value.realize(), cache)
        # 如果对象是 VariableTracker 的子类，则在原地更新其值
        elif issubclass(value_cls, VariableTracker):
            result = value
            # 获取对象的字典属性
            value_dict = value.__dict__
            # 获取非变量字段
            nonvars = value._nonvar_fields
            # 遍历对象的字典属性
            for key in value_dict:
                # 如果键不是非变量字段，则递归实现其值
                if key not in nonvars:
                    value_dict[key] = cls.realize_all(value_dict[key], cache)
        # 如果对象是列表，则递归实现其每个元素
        elif value_cls is list:
            result = [cls.realize_all(v, cache) for v in value]
        # 如果对象是元组，则递归实现其每个元素并转换为元组
        elif value_cls is tuple:
            result = tuple(cls.realize_all(v, cache) for v in value)
        # 如果对象是字典或有序字典，则递归实现其每个键值对
        elif value_cls in (dict, collections.OrderedDict):
            result = {k: cls.realize_all(v, cache) for k, v in list(value.items())}
        # 对于其他类型的对象，直接返回其值
        else:
            result = value

        # 将对象保存在缓存中，确保其唯一标识符不会被重复使用
        cache[idx] = (result, value)
        return result
# 定义一个函数，用于创建并返回一个新的函数，该函数实现了惰性变量跟踪器中某个属性的实现和转发
def _create_realize_and_forward(name):
    # 使用 functools.wraps 包装函数，以便将被包装函数的元数据复制到新创建的函数中
    @functools.wraps(getattr(VariableTracker, name))
    def realize_and_forward(self, *args, **kwargs):
        # 调用 self.realize() 方法获取变量的实际值，然后调用其属性 name，并将参数传递给该属性
        return getattr(self.realize(), name)(*args, **kwargs)

    return realize_and_forward

# 定义一个函数，用于将 VariableTracker 中未在 LazyVariableTracker 中定义的可调用属性复制到后者中
def _populate():
    # 遍历 VariableTracker 类的所有属性名和属性值
    for name, value in VariableTracker.__dict__.items():
        # 如果属性名 name 不在 LazyVariableTracker 类的属性字典中
        if name not in LazyVariableTracker.__dict__:
            # 如果属性值 value 是可调用的
            if callable(value):
                # 将属性名 name 在 LazyVariableTracker 类中动态设置为 _create_realize_and_forward(name) 返回的函数
                setattr(LazyVariableTracker, name, _create_realize_and_forward(name))

# 执行 _populate 函数，将 VariableTracker 中未定义的可调用属性复制到 LazyVariableTracker 中
_populate()
```
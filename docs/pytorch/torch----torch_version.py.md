# `.\pytorch\torch\torch_version.py`

```
# 忽略类型检查错误，通常用于跳过类型检查时使用
# 从 typing 模块导入 Any 和 Iterable 类型
from typing import Any, Iterable

# 导入 torch._vendor.packaging.version 模块中的 InvalidVersion 和 Version 类
from torch._vendor.packaging.version import InvalidVersion, Version
# 导入 torch.version 模块中的 __version__ 别名为 internal_version
from torch.version import __version__ as internal_version

# 设置 __all__ 列表，用于模块的导出
__all__ = ["TorchVersion"]

# 定义 TorchVersion 类，继承自 str 类
class TorchVersion(str):
    """A string with magic powers to compare to both Version and iterables!
    Prior to 1.10.0 torch.__version__ was stored as a str and so many did
    comparisons against torch.__version__ as if it were a str. In order to not
    break them we have TorchVersion which masquerades as a str while also
    having the ability to compare against both packaging.version.Version as
    well as tuples of values, eg. (1, 2, 1)
    Examples:
        Comparing a TorchVersion object to a Version object
            TorchVersion('1.10.0a') > Version('1.10.0a')
        Comparing a TorchVersion object to a Tuple object
            TorchVersion('1.10.0a') > (1, 2)    # 1.2
            TorchVersion('1.10.0a') > (1, 2, 1) # 1.2.1
        Comparing a TorchVersion object against a string
            TorchVersion('1.10.0a') > '1.2'
            TorchVersion('1.10.0a') > '1.2.1'
    """

    # 定义 _convert_to_version 方法，用于将输入转换为 Version 对象
    def _convert_to_version(self, inp: Any) -> Any:
        # 如果输入对象是 Version 类型，则直接返回
        if isinstance(inp, Version):
            return inp
        # 如果输入对象是 str 类型，则转换为 Version 对象返回
        elif isinstance(inp, str):
            return Version(inp)
        # 如果输入对象是 Iterable 类型，则尝试转换为 Version 对象后返回
        elif isinstance(inp, Iterable):
            # 尝试通过连接迭代器中的项来构建版本号字符串，并转换为 Version 对象返回
            return Version(".".join(str(item) for item in inp))
        else:
            # 若输入对象类型不支持转换为 Version，则抛出 InvalidVersion 异常
            raise InvalidVersion(inp)

    # 定义 _cmp_wrapper 方法，用于比较版本号或字符串
    def _cmp_wrapper(self, cmp: Any, method: str) -> bool:
        try:
            # 尝试使用 getattr 调用 Version 对象的比较方法，进行版本号比较
            return getattr(Version(self), method)(self._convert_to_version(cmp))
        except BaseException as e:
            # 处理无效版本号异常，若无效则回退到普通字符串比较
            if not isinstance(e, InvalidVersion):
                raise
            # 在处理无效版本号时，回退到使用 str 类的比较方法
            return getattr(super(), method)(cmp)

# 遍历比较方法列表，为 TorchVersion 类动态添加比较方法
for cmp_method in ["__gt__", "__lt__", "__eq__", "__ge__", "__le__"]:
    setattr(
        TorchVersion,
        cmp_method,
        lambda x, y, method=cmp_method: x._cmp_wrapper(y, method),
    )

# 将内部版本号 internal_version 转换为 TorchVersion 类型，并赋值给 __version__
__version__ = TorchVersion(internal_version)
```
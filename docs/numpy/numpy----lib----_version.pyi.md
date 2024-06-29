# `D:\src\scipysrc\numpy\numpy\lib\_version.pyi`

```py
# 定义一个模块级别的变量 __all__，用于指定在使用 import * 时应该导入的符号列表
__all__: list[str]

# 定义一个名为 NumpyVersion 的类
class NumpyVersion:
    # 类的属性定义
    vstring: str               # 版本字符串
    version: str               # 版本号
    major: int                 # 主版本号
    minor: int                 # 次版本号
    bugfix: int                # 修复版本号
    pre_release: str           # 预发布版本信息
    is_devversion: bool        # 是否为开发版本

    # 类的初始化方法，接受一个版本字符串作为参数
    def __init__(self, vstring: str) -> None: ...

    # 小于号比较方法，用于版本比较，支持字符串或 NumpyVersion 对象作为参数
    def __lt__(self, other: str | NumpyVersion) -> bool: ...

    # 小于等于号比较方法，用于版本比较，支持字符串或 NumpyVersion 对象作为参数
    def __le__(self, other: str | NumpyVersion) -> bool: ...

    # 等于号比较方法，用于版本比较，支持字符串或 NumpyVersion 对象作为参数
    def __eq__(self, other: str | NumpyVersion) -> bool: ...  # type: ignore[override]

    # 不等于号比较方法，用于版本比较，支持字符串或 NumpyVersion 对象作为参数
    def __ne__(self, other: str | NumpyVersion) -> bool: ...  # type: ignore[override]

    # 大于号比较方法，用于版本比较，支持字符串或 NumpyVersion 对象作为参数
    def __gt__(self, other: str | NumpyVersion) -> bool: ...

    # 大于等于号比较方法，用于版本比较，支持字符串或 NumpyVersion 对象作为参数
    def __ge__(self, other: str | NumpyVersion) -> bool: ...
```
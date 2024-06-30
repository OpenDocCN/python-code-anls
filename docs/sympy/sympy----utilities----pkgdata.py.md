# `D:\src\scipysrc\sympy\sympy\utilities\pkgdata.py`

```
# 导入必要的模块和函数
import sys
import os
from io import StringIO

# 导入 sympy 库中的装饰器函数 deprecated
from sympy.utilities.decorator import deprecated

# 使用装饰器标记函数为已弃用，并提供相关信息
@deprecated(
    """
    The sympy.utilities.pkgdata module and its get_resource function are
    deprecated. Use the stdlib importlib.resources module instead.
    """,
    deprecated_since_version="1.12",
    active_deprecations_target="pkgdata",
)
# 定义函数 get_resource，用于获取指定资源
def get_resource(identifier, pkgname=__name__):
    # 获取指定模块的引用
    mod = sys.modules[pkgname]
    # 获取模块的文件路径
    fn = getattr(mod, '__file__', None)
    # 如果模块没有文件路径，则抛出异常
    if fn is None:
        raise OSError("%r has no __file__!")
    # 构建资源文件的完整路径
    path = os.path.join(os.path.dirname(fn), identifier)
    # 获取模块的 loader 对象
    loader = getattr(mod, '__loader__', None)
    # 如果 loader 存在，则尝试从资源路径中获取数据
    if loader is not None:
        try:
            data = loader.get_data(path)
        except (OSError, AttributeError):
            pass
        else:
            # 返回解码为 UTF-8 的数据流
            return StringIO(data.decode('utf-8'))
    # 如果无法使用 loader 获取数据，则以二进制方式打开文件并返回
    return open(os.path.normpath(path), 'rb')
```
# `.\pytorch\torch\testing\_internal\distributed\nn\api\__init__.py`

```py
# 导入必要的模块
import os
from typing import List

# 定义一个类用于表示简单的文件系统路径
class Path:
    # 初始化方法，接受一个路径字符串并设置实例变量
    def __init__(self, path: str):
        self._path = path

    # 属性方法，返回路径字符串
    @property
    def path(self) -> str:
        return self._path

    # 方法用于检查路径是否存在
    def exists(self) -> bool:
        return os.path.exists(self._path)

    # 方法用于获取路径的基本名称
    def name(self) -> str:
        return os.path.basename(self._path)

    # 方法用于获取路径的目录部分
    def parent(self) -> str:
        return os.path.dirname(self._path)

    # 方法用于返回路径的所有组成部分
    def parts(self) -> List[str]:
        return self._path.split(os.path.sep)

    # 方法用于拼接路径
    def join(self, *args: str) -> str:
        return os.path.join(self._path, *args)

# 创建一个路径对象，路径为当前工作目录
p = Path(os.getcwd())

# 输出路径的基本名称
print(p.name())

# 输出路径的目录部分
print(p.parent())
```
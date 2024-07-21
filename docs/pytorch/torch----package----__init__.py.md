# `.\pytorch\torch\package\__init__.py`

```
# 导入用于检查模块是否来自特定包的函数
from .analyze.is_from_package import is_from_package
# 导入表示文件结构的Directory类
from .file_structure_representation import Directory
# 导入用于处理Glob组的类
from .glob_group import GlobGroup
# 导入导入器及其相关异常和类
from .importer import (
    Importer,                 # 导入通用导入器基类
    ObjMismatchError,         # 导入对象类型不匹配的异常类
    ObjNotFoundError,         # 导入对象未找到的异常类
    OrderedImporter,          # 导入有序导入器类
    sys_importer,             # 导入系统级导入器实例
)
# 导入用于导出包的异常类和导出器类
from .package_exporter import (
    EmptyMatchError,          # 导入空匹配异常类
    PackageExporter,          # 导入包导出器类
    PackagingError            # 导入打包错误异常类
)
# 导入包导入器类
from .package_importer import PackageImporter
```
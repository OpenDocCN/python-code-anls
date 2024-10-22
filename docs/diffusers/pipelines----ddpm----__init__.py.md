# `.\diffusers\pipelines\ddpm\__init__.py`

```py
# 导入类型检查功能
from typing import TYPE_CHECKING

# 从上层模块导入工具函数和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 导入用于慢速导入的常量
    _LazyModule,            # 导入懒加载模块的类
)

# 定义模块导入结构，指定将要导入的子模块
_import_structure = {"pipeline_ddpm": ["DDPMPipeline"]}

# 如果在类型检查模式或需要慢速导入，则导入 DDPMPipeline
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .pipeline_ddpm import DDPMPipeline  # 从子模块导入 DDPMPipeline

# 否则，进行懒加载模块的处理
else:
    import sys  # 导入系统模块

    # 将当前模块替换为懒加载模块，延迟实际导入
    sys.modules[__name__] = _LazyModule(
        __name__,                    # 当前模块名
        globals()["__file__"],      # 当前文件路径
        _import_structure,           # 导入结构
        module_spec=__spec__,       # 模块规范
    )
```
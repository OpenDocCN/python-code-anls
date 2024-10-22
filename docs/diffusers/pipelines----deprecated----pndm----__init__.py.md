# `.\diffusers\pipelines\deprecated\pndm\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，主要用于类型检查
from typing import TYPE_CHECKING

# 从上级目录的 utils 模块导入 DIFFUSERS_SLOW_IMPORT 和 _LazyModule
from ....utils import DIFFUSERS_SLOW_IMPORT, _LazyModule

# 定义模块导入结构，指定需要导入的子模块和类
_import_structure = {"pipeline_pndm": ["PNDMPipeline"]}

# 检查是否处于类型检查阶段或需要慢速导入
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 从子模块 pipeline_pndm 导入 PNDMPipeline 类
    from .pipeline_pndm import PNDMPipeline
else:
    # 如果不是类型检查阶段，则导入 sys 模块
    import sys

    # 将当前模块替换为 _LazyModule 对象，以实现懒加载功能
    sys.modules[__name__] = _LazyModule(
        __name__,  # 当前模块名称
        globals()["__file__"],  # 当前模块文件路径
        _import_structure,  # 导入结构
        module_spec=__spec__,  # 模块的规格
    )
```
# `.\diffusers\pipelines\consistency_models\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，用于类型检查
from typing import TYPE_CHECKING

# 从上级模块导入相关工具
from ...utils import (
    # 导入常量 DIFFUSERS_SLOW_IMPORT
    DIFFUSERS_SLOW_IMPORT,
    # 导入延迟加载模块的工具类
    _LazyModule,
)

# 定义要导入的模块结构，包含 'pipeline_consistency_models' 模块及其内容
_import_structure = {
    "pipeline_consistency_models": ["ConsistencyModelPipeline"],
}

# 判断是否处于类型检查阶段或需要慢速导入
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 从 'pipeline_consistency_models' 模块导入 ConsistencyModelPipeline 类
    from .pipeline_consistency_models import ConsistencyModelPipeline

# 否则执行以下代码
else:
    import sys

    # 使用 _LazyModule 创建一个延迟加载模块，并将其赋值给当前模块名
    sys.modules[__name__] = _LazyModule(
        # 当前模块名
        __name__,
        # 当前模块文件路径
        globals()["__file__"],
        # 导入结构
        _import_structure,
        # 模块规范
        module_spec=__spec__,
    )
```
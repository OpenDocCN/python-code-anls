# `.\diffusers\pipelines\dance_diffusion\__init__.py`

```py
# 导入类型检查相关的模块
from typing import TYPE_CHECKING

# 从上层目录导入慢速导入标志和懒加载模块工具
from ...utils import DIFFUSERS_SLOW_IMPORT, _LazyModule

# 定义要导入的结构，包括 'pipeline_dance_diffusion' 模块和其中的 'DanceDiffusionPipeline' 类
_import_structure = {"pipeline_dance_diffusion": ["DanceDiffusionPipeline"]}

# 如果是类型检查或慢速导入标志为真，则直接导入 DanceDiffusionPipeline
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .pipeline_dance_diffusion import DanceDiffusionPipeline
else:
    # 否则，导入 sys 模块以处理懒加载
    import sys

    # 将当前模块替换为懒加载模块，并提供必要的参数
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
```
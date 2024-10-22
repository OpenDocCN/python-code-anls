# `.\diffusers\pipelines\ddim\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，用于类型检查
from typing import TYPE_CHECKING

# 从相对路径导入 DIFFUSERS_SLOW_IMPORT 和 _LazyModule
from ...utils import DIFFUSERS_SLOW_IMPORT, _LazyModule

# 定义模块导入结构，包含 "pipeline_ddim" 和其下的 "DDIMPipeline"
_import_structure = {"pipeline_ddim": ["DDIMPipeline"]}

# 如果在类型检查模式下或需要慢速导入，直接导入 DDIMPipeline
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .pipeline_ddim import DDIMPipeline
else:
    # 否则，导入 sys 模块
    import sys

    # 将当前模块替换为 _LazyModule 实例，以实现延迟加载
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
```
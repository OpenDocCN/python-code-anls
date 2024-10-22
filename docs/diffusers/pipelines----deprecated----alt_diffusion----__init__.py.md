# `.\diffusers\pipelines\deprecated\alt_diffusion\__init__.py`

```py
# 导入类型检查相关的常量
from typing import TYPE_CHECKING

# 从上级模块导入工具函数和常量
from ....utils import (
    DIFFUSERS_SLOW_IMPORT,  # 慢导入的标志
    OptionalDependencyNotAvailable,  # 可选依赖不可用的异常
    _LazyModule,  # 懒加载模块的类
    get_objects_from_module,  # 从模块获取对象的函数
    is_torch_available,  # 检查 PyTorch 是否可用的函数
    is_transformers_available,  # 检查 Transformers 是否可用的函数
)

# 用于存储虚拟对象的字典
_dummy_objects = {}
# 用于存储模块导入结构的字典
_import_structure = {}

try:
    # 检查是否可用 Transformers 和 PyTorch
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()  # 抛出异常
except OptionalDependencyNotAvailable:
    # 导入虚拟对象的模块
    from ....utils import dummy_torch_and_transformers_objects

    # 更新虚拟对象字典
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # 添加可用的模型导入结构
    _import_structure["modeling_roberta_series"] = ["RobertaSeriesModelWithTransformation"]
    _import_structure["pipeline_alt_diffusion"] = ["AltDiffusionPipeline"]
    _import_structure["pipeline_alt_diffusion_img2img"] = ["AltDiffusionImg2ImgPipeline"]

    _import_structure["pipeline_output"] = ["AltDiffusionPipelineOutput"]

# 检查类型或慢导入标志
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        # 检查可用性
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()  # 抛出异常
    except OptionalDependencyNotAvailable:
        # 从虚拟对象模块导入所有内容
        from ....utils.dummy_torch_and_transformers_objects import *

    else:
        # 从相关模块导入具体类
        from .modeling_roberta_series import RobertaSeriesModelWithTransformation
        from .pipeline_alt_diffusion import AltDiffusionPipeline
        from .pipeline_alt_diffusion_img2img import AltDiffusionImg2ImgPipeline
        from .pipeline_output import AltDiffusionPipelineOutput

else:
    # 导入 sys 模块
    import sys

    # 设置当前模块为懒加载模块
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
    # 将虚拟对象添加到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```
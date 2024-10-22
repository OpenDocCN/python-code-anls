# `.\diffusers\pipelines\stable_diffusion_3\__init__.py`

```py
# 从类型检查模块导入 TYPE_CHECKING，用于静态类型检查
from typing import TYPE_CHECKING

# 从上级目录的 utils 模块导入所需的工具和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 导入标志，指示是否慢速导入
    OptionalDependencyNotAvailable,  # 导入自定义异常，用于处理可选依赖未满足的情况
    _LazyModule,  # 导入延迟加载模块的工具
    get_objects_from_module,  # 导入从模块获取对象的工具
    is_flax_available,  # 导入检查 Flax 库是否可用的工具
    is_torch_available,  # 导入检查 PyTorch 库是否可用的工具
    is_transformers_available,  # 导入检查 Transformers 库是否可用的工具
)

# 初始化一个空字典，用于存放虚拟对象
_dummy_objects = {}
# 初始化一个空字典，用于存放额外的导入对象
_additional_imports = {}
# 定义模块的导入结构，指定模块中的输出内容
_import_structure = {"pipeline_output": ["StableDiffusion3PipelineOutput"]}

# 尝试检查 Transformers 和 Torch 是否可用
try:
    if not (is_transformers_available() and is_torch_available()):  # 如果两个库都不可用
        raise OptionalDependencyNotAvailable()  # 抛出可选依赖未满足的异常
except OptionalDependencyNotAvailable:  # 捕获异常
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403  # 从 utils 导入虚拟对象的模块

    # 更新虚拟对象字典，获取从虚拟模块中提取的对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:  # 如果没有异常发生
    # 将可用的模块添加到导入结构中
    _import_structure["pipeline_stable_diffusion_3"] = ["StableDiffusion3Pipeline"]
    _import_structure["pipeline_stable_diffusion_3_img2img"] = ["StableDiffusion3Img2ImgPipeline"]
    _import_structure["pipeline_stable_diffusion_3_inpaint"] = ["StableDiffusion3InpaintPipeline"]

# 检查是否在类型检查阶段或者是否慢速导入
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):  # 如果两个库都不可用
            raise OptionalDependencyNotAvailable()  # 抛出可选依赖未满足的异常
    except OptionalDependencyNotAvailable:  # 捕获异常
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403  # 从 utils 导入虚拟对象模块
    else:  # 如果没有异常发生
        # 从稳定扩散的模块导入相应的管道类
        from .pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
        from .pipeline_stable_diffusion_3_img2img import StableDiffusion3Img2ImgPipeline
        from .pipeline_stable_diffusion_3_inpaint import StableDiffusion3InpaintPipeline

else:  # 如果不是类型检查阶段或慢速导入
    import sys  # 导入系统模块

    # 使用延迟加载模块替换当前模块的 sys.modules 条目
    sys.modules[__name__] = _LazyModule(
        __name__,  # 模块名称
        globals()["__file__"],  # 当前文件路径
        _import_structure,  # 模块的导入结构
        module_spec=__spec__,  # 模块的规格
    )

    # 将虚拟对象字典中的对象添加到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
    # 将额外的导入对象添加到当前模块
    for name, value in _additional_imports.items():
        setattr(sys.modules[__name__], name, value)
```
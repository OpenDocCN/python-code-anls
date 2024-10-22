# `.\diffusers\pipelines\stable_diffusion_gligen\__init__.py`

```py
# 导入类型检查常量
from typing import TYPE_CHECKING

# 从上级模块导入必要的工具和依赖
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 慢导入标志
    OptionalDependencyNotAvailable,  # 可选依赖未找到异常
    _LazyModule,  # 懒加载模块的工具
    get_objects_from_module,  # 从模块中获取对象的工具
    is_torch_available,  # 检查 PyTorch 是否可用的函数
    is_transformers_available,  # 检查 Transformers 是否可用的函数
)

# 初始化一个空字典用于存放虚拟对象
_dummy_objects = {}
# 初始化一个空字典用于存放导入结构
_import_structure = {}

# 尝试检查依赖是否可用
try:
    # 如果 Transformers 和 PyTorch 不可用，抛出异常
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖未找到异常
except OptionalDependencyNotAvailable:
    # 从工具中导入虚拟对象以防止错误
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    # 更新虚拟对象字典，获取虚拟对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # 如果依赖可用，添加稳定扩散管道到导入结构
    _import_structure["pipeline_stable_diffusion_gligen"] = ["StableDiffusionGLIGENPipeline"]
    # 添加文本到图像的稳定扩散管道到导入结构
    _import_structure["pipeline_stable_diffusion_gligen_text_image"] = ["StableDiffusionGLIGENTextImagePipeline"]

# 检查类型标志或慢导入标志
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        # 如果 Transformers 和 PyTorch 不可用，抛出异常
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    # 捕获可选依赖未找到异常
    except OptionalDependencyNotAvailable:
        # 从工具中导入虚拟对象
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        # 从稳定扩散管道中导入 StableDiffusionGLIGENPipeline
        from .pipeline_stable_diffusion_gligen import StableDiffusionGLIGENPipeline
        # 从稳定扩散管道中导入 StableDiffusionGLIGENTextImagePipeline
        from .pipeline_stable_diffusion_gligen_text_image import StableDiffusionGLIGENTextImagePipeline

# 如果不是类型检查或慢导入
else:
    # 导入系统模块
    import sys

    # 将当前模块替换为懒加载模块
    sys.modules[__name__] = _LazyModule(
        __name__,  # 模块名称
        globals()["__file__"],  # 当前文件路径
        _import_structure,  # 导入结构
        module_spec=__spec__,  # 模块规格
    )

    # 为当前模块设置虚拟对象
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```
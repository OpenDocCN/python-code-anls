# `.\diffusers\pipelines\cogvideo\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，用于类型检查
from typing import TYPE_CHECKING

# 从相对路径的 utils 模块导入所需的工具函数和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 用于标识慢速导入的标志
    OptionalDependencyNotAvailable,  # 用于处理可选依赖项未安装的异常
    _LazyModule,  # 用于创建懒加载模块
    get_objects_from_module,  # 从模块中获取对象的函数
    is_torch_available,  # 检查 PyTorch 是否可用的函数
    is_transformers_available,  # 检查 Transformers 是否可用的函数
)

# 创建一个空字典，用于存放虚拟对象
_dummy_objects = {}
# 创建一个空字典，用于存放模块的导入结构
_import_structure = {}

# 尝试检查依赖项的可用性
try:
    # 如果 Transformers 或 PyTorch 不可用，抛出异常
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖项未可用的异常
except OptionalDependencyNotAvailable:
    # 从 utils 模块导入虚拟对象，避免导入失败
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    # 更新虚拟对象字典，填充虚拟对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
# 如果依赖项可用，更新导入结构
else:
    _import_structure["pipeline_cogvideox"] = ["CogVideoXPipeline"]  # 添加 CogVideoXPipeline
    _import_structure["pipeline_cogvideox_image2video"] = ["CogVideoXImageToVideoPipeline"]  # 添加图像转视频管道
    _import_structure["pipeline_cogvideox_video2video"] = ["CogVideoXVideoToVideoPipeline"]  # 添加视频转视频管道

# 根据类型检查或慢速导入的标志进行判断
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 尝试检查依赖项的可用性
    try:
        # 如果 Transformers 或 PyTorch 不可用，抛出异常
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    # 捕获可选依赖项未可用的异常
    except OptionalDependencyNotAvailable:
        # 从虚拟对象模块导入所有对象
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        # 导入实际的管道类
        from .pipeline_cogvideox import CogVideoXPipeline
        from .pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
        from .pipeline_cogvideox_video2video import CogVideoXVideoToVideoPipeline

# 否则处理懒加载模块
else:
    import sys

    # 用 _LazyModule 创建当前模块的懒加载实例
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,  # 传递导入结构
        module_spec=__spec__,  # 传递模块规格
    )

    # 将虚拟对象添加到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```
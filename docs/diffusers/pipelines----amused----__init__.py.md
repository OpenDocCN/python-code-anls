# `.\diffusers\pipelines\amused\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，用于类型检查
from typing import TYPE_CHECKING

# 从上级目录的 utils 模块导入多个工具和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,               # 导入用于慢导入的标志
    OptionalDependencyNotAvailable,       # 导入可选依赖项不可用异常
    _LazyModule,                         # 导入延迟模块加载器
    is_torch_available,                  # 导入检查 PyTorch 是否可用的函数
    is_transformers_available,           # 导入检查 Transformers 是否可用的函数
)

# 初始化一个空字典，用于存储虚拟对象
_dummy_objects = {}
# 初始化一个空字典，用于存储模块导入结构
_import_structure = {}

# 尝试块，用于检查依赖项的可用性
try:
    # 如果 Transformers 和 Torch 其中一个不可用，抛出异常
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖项不可用的异常
except OptionalDependencyNotAvailable:
    # 从 dummy_torch_and_transformers_objects 中导入虚拟管道对象
    from ...utils.dummy_torch_and_transformers_objects import (
        AmusedImg2ImgPipeline,            # 导入虚拟图像到图像管道
        AmusedInpaintPipeline,             # 导入虚拟图像修复管道
        AmusedPipeline,                     # 导入虚拟通用管道
    )

    # 更新虚拟对象字典
    _dummy_objects.update(
        {
            "AmusedPipeline": AmusedPipeline,                # 更新字典，映射管道名称到对象
            "AmusedImg2ImgPipeline": AmusedImg2ImgPipeline, # 更新字典，映射图像到图像管道名称到对象
            "AmusedInpaintPipeline": AmusedInpaintPipeline,   # 更新字典，映射图像修复管道名称到对象
        }
    )
# 如果依赖项可用，更新导入结构字典
else:
    _import_structure["pipeline_amused"] = ["AmusedPipeline"]               # 添加通用管道到导入结构
    _import_structure["pipeline_amused_img2img"] = ["AmusedImg2ImgPipeline"] # 添加图像到图像管道到导入结构
    _import_structure["pipeline_amused_inpaint"] = ["AmusedInpaintPipeline"]   # 添加图像修复管道到导入结构

# 检查类型是否在检查阶段，或慢导入标志是否为真
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 尝试块，用于再次检查依赖项的可用性
    try:
        # 如果 Transformers 和 Torch 其中一个不可用，抛出异常
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖项不可用的异常
    except OptionalDependencyNotAvailable:
        # 从 dummy_torch_and_transformers_objects 中导入虚拟管道对象
        from ...utils.dummy_torch_and_transformers_objects import (
            AmusedPipeline,                          # 导入虚拟通用管道
        )
    # 如果依赖项可用，导入实际的管道对象
    else:
        from .pipeline_amused import AmusedPipeline                 # 导入实际通用管道
        from .pipeline_amused_img2img import AmusedImg2ImgPipeline # 导入实际图像到图像管道
        from .pipeline_amused_inpaint import AmusedInpaintPipeline   # 导入实际图像修复管道

# 如果不在类型检查或慢导入阶段
else:
    import sys

    # 使用延迟模块加载器替换当前模块
    sys.modules[__name__] = _LazyModule(
        __name__,                        # 当前模块名称
        globals()["__file__"],          # 当前文件路径
        _import_structure,               # 导入结构
        module_spec=__spec__,           # 模块规格
    )

    # 遍历虚拟对象字典，将虚拟对象添加到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)  # 动态设置模块属性
```
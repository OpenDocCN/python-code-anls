# `.\diffusers\pipelines\unidiffuser\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，用于类型检查
from typing import TYPE_CHECKING

# 从上级模块导入所需的工具和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 慢导入标志
    OptionalDependencyNotAvailable,  # 可选依赖不可用异常
    _LazyModule,  # 延迟加载模块类
    is_torch_available,  # 检查是否可用 PyTorch
    is_transformers_available,  # 检查是否可用 Transformers
)

# 初始化空字典用于存储虚拟对象
_dummy_objects = {}
# 初始化空字典用于存储导入结构
_import_structure = {}

# 尝试块，用于处理可选依赖
try:
    # 如果 Transformers 和 PyTorch 不可用，则抛出异常
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
# 异常处理块
except OptionalDependencyNotAvailable:
    # 从虚拟对象模块导入必要的类
    from ...utils.dummy_torch_and_transformers_objects import (
        ImageTextPipelineOutput,  # 图像文本管道输出
        UniDiffuserPipeline,  # UniDiffuser 管道
    )

    # 更新虚拟对象字典
    _dummy_objects.update(
        {"ImageTextPipelineOutput": ImageTextPipelineOutput, "UniDiffuserPipeline": UniDiffuserPipeline}
    )
# 否则块
else:
    # 更新导入结构以包含文本解码模型
    _import_structure["modeling_text_decoder"] = ["UniDiffuserTextDecoder"]
    # 更新导入结构以包含 UVIT 模型
    _import_structure["modeling_uvit"] = ["UniDiffuserModel", "UTransformer2DModel"]
    # 更新导入结构以包含图像文本管道
    _import_structure["pipeline_unidiffuser"] = ["ImageTextPipelineOutput", "UniDiffuserPipeline"]

# 检查类型是否在检查中或是否启用慢导入
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 尝试块，用于处理可选依赖
    try:
        # 如果 Transformers 和 PyTorch 不可用，则抛出异常
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    # 异常处理块
    except OptionalDependencyNotAvailable:
        # 从虚拟对象模块导入必要的类
        from ...utils.dummy_torch_and_transformers_objects import (
            ImageTextPipelineOutput,  # 图像文本管道输出
            UniDiffuserPipeline,  # UniDiffuser 管道
        )
    # 否则块
    else:
        # 从文本解码模型模块导入
        from .modeling_text_decoder import UniDiffuserTextDecoder
        # 从 UVIT 模型模块导入
        from .modeling_uvit import UniDiffuserModel, UTransformer2DModel
        # 从管道模块导入图像文本管道输出和 UniDiffuser 管道
        from .pipeline_unidiffuser import ImageTextPipelineOutput, UniDiffuserPipeline

# 否则块
else:
    # 导入系统模块以进行模块操作
    import sys

    # 用于延迟加载模块的类创建模块实例
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    # 将虚拟对象字典中的对象设置到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```
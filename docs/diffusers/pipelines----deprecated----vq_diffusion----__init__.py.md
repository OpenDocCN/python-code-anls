# `.\diffusers\pipelines\deprecated\vq_diffusion\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，以便在类型检查时使用
from typing import TYPE_CHECKING

# 从上层模块的 utils 导入一些工具和常量
from ....utils import (
    DIFFUSERS_SLOW_IMPORT,  # 导入标志，指示是否慢速导入
    OptionalDependencyNotAvailable,  # 导入异常，用于处理可选依赖未满足的情况
    _LazyModule,  # 导入懒加载模块的类
    is_torch_available,  # 导入函数，检查 PyTorch 是否可用
    is_transformers_available,  # 导入函数，检查 transformers 是否可用
)

# 初始化一个空字典，用于存储虚拟对象
_dummy_objects = {}
# 初始化一个空字典，用于存储模块导入结构
_import_structure = {}

# 尝试检查依赖是否可用
try:
    # 如果 transformers 和 torch 都不可用，抛出异常
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 从 dummy_torch_and_transformers_objects 模块导入虚拟对象
    from ....utils.dummy_torch_and_transformers_objects import (
        LearnedClassifierFreeSamplingEmbeddings,  # 导入虚拟的学习分类器对象
        VQDiffusionPipeline,  # 导入虚拟的 VQ 扩散管道对象
    )

    # 更新 _dummy_objects 字典，添加导入的虚拟对象
    _dummy_objects.update(
        {
            "LearnedClassifierFreeSamplingEmbeddings": LearnedClassifierFreeSamplingEmbeddings,  # 添加学习分类器对象
            "VQDiffusionPipeline": VQDiffusionPipeline,  # 添加 VQ 扩散管道对象
        }
    )
# 如果没有抛出异常，执行以下代码
else:
    # 更新 _import_structure 字典，添加实际模块的路径
    _import_structure["pipeline_vq_diffusion"] = ["LearnedClassifierFreeSamplingEmbeddings", "VQDiffusionPipeline"]  # 指定管道模块的导入

# 如果类型检查或慢速导入标志为真
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 尝试检查依赖是否可用
    try:
        # 如果 transformers 和 torch 都不可用，抛出异常
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖不可用的异常
    except OptionalDependencyNotAvailable:
        # 从 dummy_torch_and_transformers_objects 模块导入虚拟对象
        from ....utils.dummy_torch_and_transformers_objects import (
            LearnedClassifierFreeSamplingEmbeddings,  # 导入虚拟的学习分类器对象
            VQDiffusionPipeline,  # 导入虚拟的 VQ 扩散管道对象
        )
    # 如果没有抛出异常，执行以下代码
    else:
        # 从 pipeline_vq_diffusion 模块导入实际对象
        from .pipeline_vq_diffusion import LearnedClassifierFreeSamplingEmbeddings, VQDiffusionPipeline  # 导入实际的学习分类器和 VQ 扩散管道对象

# 如果不在类型检查或慢速导入模式
else:
    import sys  # 导入 sys 模块，用于操作 Python 运行时环境

    # 使用懒加载模块创建一个新的模块
    sys.modules[__name__] = _LazyModule(
        __name__,  # 当前模块名称
        globals()["__file__"],  # 当前模块文件路径
        _import_structure,  # 导入结构字典
        module_spec=__spec__,  # 模块的规格
    )

    # 遍历 _dummy_objects 字典，将虚拟对象设置到当前模块中
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)  # 设置虚拟对象到模块属性中
```
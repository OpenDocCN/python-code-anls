# `.\diffusers\pipelines\ledits_pp\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，用于类型检查的标志
from typing import TYPE_CHECKING

# 从 utils 模块导入多个功能和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 导入用于标识慢导入的常量
    OptionalDependencyNotAvailable,  # 导入表示可选依赖不可用的异常
    _LazyModule,  # 导入懒加载模块的类
    get_objects_from_module,  # 导入从模块获取对象的函数
    is_torch_available,  # 导入检查 PyTorch 是否可用的函数
    is_transformers_available,  # 导入检查 Transformers 是否可用的函数
)

# 初始化一个空字典，用于存储虚拟对象
_dummy_objects = {}
# 初始化一个空字典，用于存储模块导入结构
_import_structure = {}

# 尝试检测可选依赖项是否可用
try:
    # 检查 Transformers 和 PyTorch 是否都可用，如果不可用则抛出异常
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 从 utils 模块导入虚拟对象的模块，忽略相关的警告
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    # 更新虚拟对象字典，将虚拟对象加入到字典中
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
# 如果没有异常，执行以下代码
else:
    # 将稳定扩散的管道添加到导入结构字典中
    _import_structure["pipeline_leditspp_stable_diffusion"] = ["LEditsPPPipelineStableDiffusion"]
    # 将 XL 版本的稳定扩散管道添加到导入结构字典中
    _import_structure["pipeline_leditspp_stable_diffusion_xl"] = ["LEditsPPPipelineStableDiffusionXL"]

    # 将管道输出的相关类添加到导入结构字典中
    _import_structure["pipeline_output"] = ["LEditsPPDiffusionPipelineOutput", "LEditsPPDiffusionPipelineOutput"]

# 如果是类型检查或者使用了慢导入标志
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 尝试检测可选依赖项是否可用
    try:
        # 检查 Transformers 和 PyTorch 是否都可用，如果不可用则抛出异常
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    # 捕获可选依赖不可用的异常
    except OptionalDependencyNotAvailable:
        # 从虚拟对象的模块导入所有内容
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        # 从稳定扩散的管道模块导入相关类
        from .pipeline_leditspp_stable_diffusion import (
            LEditsPPDiffusionPipelineOutput,  # 导入稳定扩散管道输出类
            LEditsPPInversionPipelineOutput,  # 导入逆向稳定扩散管道输出类
            LEditsPPPipelineStableDiffusion,  # 导入稳定扩散管道类
        )
        # 从 XL 版本的稳定扩散管道模块导入相关类
        from .pipeline_leditspp_stable_diffusion_xl import LEditsPPPipelineStableDiffusionXL

# 如果不是类型检查且没有使用慢导入标志
else:
    # 导入 sys 模块以便操作模块系统
    import sys

    # 使用懒加载模块创建当前模块的 lazy 实例，并将其赋值给当前模块
    sys.modules[__name__] = _LazyModule(
        __name__,  # 当前模块的名称
        globals()["__file__"],  # 当前模块的文件路径
        _import_structure,  # 模块导入结构
        module_spec=__spec__,  # 模块的规格
    )

    # 将虚拟对象字典中的所有对象添加到当前模块中
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```
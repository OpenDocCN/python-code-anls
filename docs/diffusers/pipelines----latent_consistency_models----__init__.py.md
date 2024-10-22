# `.\diffusers\pipelines\latent_consistency_models\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，以便在类型检查时使用
from typing import TYPE_CHECKING

# 从相对路径下的 utils 模块导入所需的工具和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 导入用于判断慢导入的常量
    OptionalDependencyNotAvailable,  # 导入用于处理缺少可选依赖的异常
    _LazyModule,  # 导入懒加载模块的工具
    get_objects_from_module,  # 导入从模块中获取对象的函数
    is_torch_available,  # 导入判断 PyTorch 是否可用的函数
    is_transformers_available,  # 导入判断 Transformers 是否可用的函数
)

# 初始化一个空字典，用于存储虚拟对象
_dummy_objects = {}
# 初始化一个空字典，用于定义导入结构
_import_structure = {}

# 尝试检查 Transformers 和 PyTorch 的可用性
try:
    if not (is_transformers_available() and is_torch_available()):  # 如果两者都不可用
        raise OptionalDependencyNotAvailable()  # 抛出异常，表示缺少可选依赖
except OptionalDependencyNotAvailable:  # 捕获可选依赖缺失的异常
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403  # 导入虚拟对象模块

    # 更新虚拟对象字典，填充缺失的对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # 如果可用，更新导入结构，添加模型管道
    _import_structure["pipeline_latent_consistency_img2img"] = ["LatentConsistencyModelImg2ImgPipeline"]
    _import_structure["pipeline_latent_consistency_text2img"] = ["LatentConsistencyModelPipeline"]

# 检查是否处于类型检查状态或慢导入模式
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):  # 如果两者都不可用
            raise OptionalDependencyNotAvailable()  # 抛出异常，表示缺少可选依赖

    except OptionalDependencyNotAvailable:  # 捕获可选依赖缺失的异常
        from ...utils.dummy_torch_and_transformers_objects import *  # 导入虚拟对象
    else:
        # 如果可用，导入相应的模型管道
        from .pipeline_latent_consistency_img2img import LatentConsistencyModelImg2ImgPipeline
        from .pipeline_latent_consistency_text2img import LatentConsistencyModelPipeline

else:  # 如果不是类型检查或慢导入模式
    import sys  # 导入系统模块

    # 用懒加载模块替代当前模块
    sys.modules[__name__] = _LazyModule(
        __name__,  # 模块名
        globals()["__file__"],  # 当前文件的路径
        _import_structure,  # 导入结构
        module_spec=__spec__,  # 模块的规范
    )

    # 将虚拟对象添加到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)  # 设置模块属性
```
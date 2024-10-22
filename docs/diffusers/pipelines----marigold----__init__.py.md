# `.\diffusers\pipelines\marigold\__init__.py`

```py
# 导入类型检查相关的模块
from typing import TYPE_CHECKING

# 从相对路径导入所需的工具函数和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 慢导入标志
    OptionalDependencyNotAvailable,  # 可选依赖未可用的异常类
    _LazyModule,  # 延迟模块加载类
    get_objects_from_module,  # 从模块获取对象的函数
    is_torch_available,  # 检查 PyTorch 是否可用的函数
    is_transformers_available,  # 检查 Transformers 是否可用的函数
)

# 初始化一个空字典，用于存储占位对象
_dummy_objects = {}
# 初始化一个空字典，用于存储导入结构
_import_structure = {}

# 尝试检测依赖关系
try:
    # 检查 Transformers 和 PyTorch 是否都可用
    if not (is_transformers_available() and is_torch_available()):
        # 如果不可用，则抛出异常
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖未可用的异常
except OptionalDependencyNotAvailable:
    # 从工具模块中导入占位对象（忽略 F403 警告）
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    # 更新占位对象字典
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # 如果依赖可用，定义模块的导入结构
    _import_structure["marigold_image_processing"] = ["MarigoldImageProcessor"]
    _import_structure["pipeline_marigold_depth"] = ["MarigoldDepthOutput", "MarigoldDepthPipeline"]
    _import_structure["pipeline_marigold_normals"] = ["MarigoldNormalsOutput", "MarigoldNormalsPipeline"]

# 检查类型检查或慢导入标志
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 尝试检测依赖关系
    try:
        # 检查 Transformers 和 PyTorch 是否都可用
        if not (is_transformers_available() and is_torch_available()):
            # 如果不可用，则抛出异常
            raise OptionalDependencyNotAvailable()

    # 捕获可选依赖未可用的异常
    except OptionalDependencyNotAvailable:
        # 导入占位对象以避免错误
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        # 导入实际模块中的类
        from .marigold_image_processing import MarigoldImageProcessor
        from .pipeline_marigold_depth import MarigoldDepthOutput, MarigoldDepthPipeline
        from .pipeline_marigold_normals import MarigoldNormalsOutput, MarigoldNormalsPipeline

else:
    # 导入系统模块
    import sys

    # 用延迟模块加载类替代当前模块
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],  # 当前文件的全局名称
        _import_structure,  # 导入结构
        module_spec=__spec__,  # 模块的规范
    )
    # 将占位对象添加到当前模块中
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```
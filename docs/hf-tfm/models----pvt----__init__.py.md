# `.\transformers\models\pvt\__init__.py`

```py
# 设置文件编码为 UTF-8

# 版权声明，包括作者和 HuggingFace 公司团队，保留所有权利
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证中的规定，否则不得使用此文件
# 可以在以下网址获得许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或书面同意的要求，本软件按"原样"提供，不提供任何形式的担保，明示或暗示。
# 有关特定权限和限制的信息，请参阅许可证。

# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入实用函数模块
from ...utils import (
    OptionalDependencyNotAvailable,  # 导入可选依赖未安装时的异常
    _LazyModule,  # 导入惰性模块
    is_torch_available,  # 导入检查是否安装了 Torch 的函数
    is_vision_available,  # 导入检查是否安装了 Vision 的函数
)

# 定义导入结构
_import_structure = {
    "configuration_pvt": ["PVT_PRETRAINED_CONFIG_ARCHIVE_MAP", "PvtConfig", "PvtOnnxConfig"],  # 配置 PVT 模型的导入结构
}

# 如果未安装 Vision 相关依赖，则抛出异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Vision 可用，则添加图像处理模块到导入结构
    _import_structure["image_processing_pvt"] = ["PvtImageProcessor"]

# 如果未安装 Torch 相关依赖，则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加模型定义和相关模块到导入结构
    _import_structure["modeling_pvt"] = [
        "PVT_PRETRAINED_MODEL_ARCHIVE_LIST",  # PVT 预训练模型存档列表
        "PvtForImageClassification",  # 用于图像分类任务的 PVT 模型
        "PvtModel",  # PVT 模型
        "PvtPreTrainedModel",  # PVT 预训练模型
    ]


# 如果处于类型检查模式，则进行额外导入
if TYPE_CHECKING:
    from .configuration_pvt import PVT_PRETRAINED_CONFIG_ARCHIVE_MAP, PvtConfig, PvtOnnxConfig  # 导入 PVT 配置

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_pvt import PvtImageProcessor  # 导入 PVT 图像处理模块

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_pvt import (
            PVT_PRETRAINED_MODEL_ARCHIVE_LIST,  # PVT 预训练模型存档列表
            PvtForImageClassification,  # 用于图像分类任务的 PVT 模型
            PvtModel,  # PVT 模型
            PvtPreTrainedModel,  # PVT 预训练模型
        )

# 如果不处于类型检查模式，则将当前模块指定为惰性模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```
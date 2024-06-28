# `.\models\deprecated\trajectory_transformer\__init__.py`

```py
# 引入类型检查工具
from typing import TYPE_CHECKING
# 引入自定义的异常：依赖未安装异常和懒加载模块
from ....utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_trajectory_transformer": [
        "TRAJECTORY_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
        "TrajectoryTransformerConfig",  # 轨迹变换器配置类
    ],
}

# 检查是否存在 Torch 库，如果不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加以下模块到导入结构中
    _import_structure["modeling_trajectory_transformer"] = [
        "TRAJECTORY_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
        "TrajectoryTransformerModel",  # 轨迹变换器模型类
        "TrajectoryTransformerPreTrainedModel",  # 轨迹变换器预训练模型类
        "load_tf_weights_in_trajectory_transformer",  # 在轨迹变换器中加载 TensorFlow 权重
    ]


# 如果正在进行类型检查
if TYPE_CHECKING:
    # 从配置模块中导入所需的符号
    from .configuration_trajectory_transformer import (
        TRAJECTORY_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练配置文件映射
        TrajectoryTransformerConfig,  # 轨迹变换器配置类
    )

    # 再次检查 Torch 库的可用性
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从建模模块中导入所需的符号
        from .modeling_trajectory_transformer import (
            TRAJECTORY_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,  # 预训练模型存档列表
            TrajectoryTransformerModel,  # 轨迹变换器模型类
            TrajectoryTransformerPreTrainedModel,  # 轨迹变换器预训练模型类
            load_tf_weights_in_trajectory_transformer,  # 在轨迹变换器中加载 TensorFlow 权重
        )

# 如果不是在类型检查模式下
else:
    import sys

    # 将当前模块设置为懒加载模块，以便在需要时才加载其内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```
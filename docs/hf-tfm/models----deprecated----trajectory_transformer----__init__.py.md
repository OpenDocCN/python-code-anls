# `.\models\deprecated\trajectory_transformer\__init__.py`

```
# 版权声明和许可协议信息

# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入自定义的工具模块和异常类
from ....utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_trajectory_transformer": [
        "TRAJECTORY_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TrajectoryTransformerConfig",
    ],
}

# 检查是否存在 torch 库
try:
    if not is_torch_available():
        # 如果不存在，抛出自定义的依赖未安装异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
# 如果存在，执行以下代码块
else:
    # 更新导入结构字典，增加模型相关模块
    _import_structure["modeling_trajectory_transformer"] = [
        "TRAJECTORY_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TrajectoryTransformerModel",
        "TrajectoryTransformerPreTrainedModel",
        "load_tf_weights_in_trajectory_transformer",
    ]

# 如果处于类型检查模式
if TYPE_CHECKING:
    # 导入配置和模型相关的类和方法
    from .configuration_trajectory_transformer import (
        TRAJECTORY_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TrajectoryTransformerConfig,
    )

    # 再次检查是否存在 torch 库
    try:
        if not is_torch_available():
            # 如果不存在，抛出自定义的依赖未安装异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果存在，导入模型相关的类和方法
    else:
        from .modeling_trajectory_transformer import (
            TRAJECTORY_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TrajectoryTransformerModel,
            TrajectoryTransformerPreTrainedModel,
            load_tf_weights_in_trajectory_transformer,
        )

# 如果不处于类型检查模式
else:
    # 导入 sys 模块
    import sys
    # 将当前模块设置为惰性模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```
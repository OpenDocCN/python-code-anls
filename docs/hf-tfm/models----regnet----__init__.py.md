# `.\transformers\models\regnet\__init__.py`

```py
# 导入必要的模块和函数
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {"configuration_regnet": ["REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "RegNetConfig"]}

# 检查是否导入了 torch，若未导入则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若导入了 torch，则添加 torch 特定模块到导入结构中
    _import_structure["modeling_regnet"] = [
        "REGNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RegNetForImageClassification",
        "RegNetModel",
        "RegNetPreTrainedModel",
    ]

# 检查是否导入了 tensorflow，若未导入则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若导入了 tensorflow，则添加 tensorflow 特定模块到导入结构中
    _import_structure["modeling_tf_regnet"] = [
        "TF_REGNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFRegNetForImageClassification",
        "TFRegNetModel",
        "TFRegNetPreTrainedModel",
    ]

# 检查是否导入了 flax，若未导入则引发异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若导入了 flax，则添加 flax 特定模块到导入结构中
    _import_structure["modeling_flax_regnet"] = [
        "FlaxRegNetForImageClassification",
        "FlaxRegNetModel",
        "FlaxRegNetPreTrainedModel",
    ]

# 若为类型检查模式，则进一步导入相应的模块
if TYPE_CHECKING:
    from .configuration_regnet import REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP, RegNetConfig

    # 导入 torch 特定模块（若可用）
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_regnet import (
            REGNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            RegNetForImageClassification,
            RegNetModel,
            RegNetPreTrainedModel,
        )

    # 导入 tensorflow 特定模块（若可用）
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_regnet import (
            TF_REGNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFRegNetForImageClassification,
            TFRegNetModel,
            TFRegNetPreTrainedModel,
        )

    # 导入 flax 特定模块（若可用）
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果条件不满足，则执行以下代码块
    else:
        # 从当前目录的modeling_flax_regnet模块中导入以下类：
        # FlaxRegNetForImageClassification，FlaxRegNetModel，FlaxRegNetPreTrainedModel
        from .modeling_flax_regnet import (
            FlaxRegNetForImageClassification,
            FlaxRegNetModel,
            FlaxRegNetPreTrainedModel,
        )
# 如果不满足上面的所有条件，则执行以下代码块
else:
    # 导入sys模块，用于操作Python运行时的环境变量和函数
    import sys
    # 将当前模块的名字(__name__)和当前文件的路径(__file__)传给_LazyModule类，创建一个惰性加载的模块对象
    # _LazyModule用于在首次访问模块代码时才会导入依赖的模块，可以在程序运行时减少初始化时间和资源的开销
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```
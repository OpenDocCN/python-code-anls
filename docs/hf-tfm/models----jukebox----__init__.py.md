# `.\models\jukebox\__init__.py`

```py
# 版权声明和许可信息
# 该模块受 Apache License, Version 2.0 许可，详情请访问 http://www.apache.org/licenses/LICENSE-2.0

# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入自定义异常和模块惰性加载工具函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_jukebox": [
        "JUKEBOX_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "JukeboxConfig",
        "JukeboxPriorConfig",
        "JukeboxVQVAEConfig",
    ],
    "tokenization_jukebox": ["JukeboxTokenizer"],
}

# 检查是否 Torch 可用，若不可用则抛出自定义的依赖不可用异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加额外的模块导入结构
    _import_structure["modeling_jukebox"] = [
        "JUKEBOX_PRETRAINED_MODEL_ARCHIVE_LIST",
        "JukeboxModel",
        "JukeboxPreTrainedModel",
        "JukeboxVQVAE",
        "JukeboxPrior",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 从相应模块导入特定的类或变量
    from .configuration_jukebox import (
        JUKEBOX_PRETRAINED_CONFIG_ARCHIVE_MAP,
        JukeboxConfig,
        JukeboxPriorConfig,
        JukeboxVQVAEConfig,
    )
    from .tokenization_jukebox import JukeboxTokenizer

    # 再次检查 Torch 是否可用，若不可用则忽略异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若 Torch 可用，则从 modeling_jukebox 模块导入特定类或变量
        from .modeling_jukebox import (
            JUKEBOX_PRETRAINED_MODEL_ARCHIVE_LIST,
            JukeboxModel,
            JukeboxPreTrainedModel,
            JukeboxPrior,
            JukeboxVQVAE,
        )

# 如果不是类型检查阶段，则执行以下操作
else:
    # 导入 sys 模块
    import sys

    # 将当前模块定义为一个惰性加载模块
    # 使用 _LazyModule 类，传入当前模块的名称、文件路径、导入结构以及模块规范
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```
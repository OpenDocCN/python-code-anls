# `.\models\encodec\__init__.py`

```py
# 版权声明及使用许可信息，声明该代码版权归HuggingFace团队所有
#
# 在Apache许可版本2.0下授权使用本文件；除非符合许可条款，否则不得使用本文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件基于“现状”提供，不附带任何明示或暗示的担保
# 查看许可证以获取特定语言的权限和限制
from typing import TYPE_CHECKING

# 从utils模块导入所需的内容
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_encodec": [
        "ENCODEC_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "EncodecConfig",
    ],
    "feature_extraction_encodec": ["EncodecFeatureExtractor"],
}

# 检查是否存在torch可用，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果torch可用，则将以下模块添加到导入结构中
    _import_structure["modeling_encodec"] = [
        "ENCODEC_PRETRAINED_MODEL_ARCHIVE_LIST",
        "EncodecModel",
        "EncodecPreTrainedModel",
    ]

# 如果类型检查为真，则导入以下模块
if TYPE_CHECKING:
    from .configuration_encodec import (
        ENCODEC_PRETRAINED_CONFIG_ARCHIVE_MAP,
        EncodecConfig,
    )
    from .feature_extraction_encodec import EncodecFeatureExtractor

    # 再次检查torch是否可用，不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果torch可用，则导入以下模块
        from .modeling_encodec import (
            ENCODEC_PRETRAINED_MODEL_ARCHIVE_LIST,
            EncodecModel,
            EncodecPreTrainedModel,
        )

else:
    # 如果不是类型检查，则导入sys模块，并将当前模块设置为_LazyModule的实例
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```
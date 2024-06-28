# `.\models\lxmert\__init__.py`

```py
# 引入必要的类型检查模块
from typing import TYPE_CHECKING

# 从相对路径导入必要的实用工具和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构字典，包含 LXMERT 相关配置和模型的导入信息
_import_structure = {
    "configuration_lxmert": ["LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "LxmertConfig"],
    "tokenization_lxmert": ["LxmertTokenizer"],
}

# 尝试检查是否存在 tokenizers，如果不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 tokenizers，则加入 tokenization_lxmert_fast 到导入结构中
    _import_structure["tokenization_lxmert_fast"] = ["LxmertTokenizerFast"]

# 尝试检查是否存在 torch，如果不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch，则加入 modeling_lxmert 到导入结构中
    _import_structure["modeling_lxmert"] = [
        "LxmertEncoder",
        "LxmertForPreTraining",
        "LxmertForQuestionAnswering",
        "LxmertModel",
        "LxmertPreTrainedModel",
        "LxmertVisualFeatureEncoder",
        "LxmertXLayer",
    ]

# 尝试检查是否存在 tensorflow，如果不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 tensorflow，则加入 modeling_tf_lxmert 到导入结构中
    _import_structure["modeling_tf_lxmert"] = [
        "TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFLxmertForPreTraining",
        "TFLxmertMainLayer",
        "TFLxmertModel",
        "TFLxmertPreTrainedModel",
        "TFLxmertVisualFeatureEncoder",
    ]

# 如果是类型检查模式，则进一步导入特定模块，用于类型检查
if TYPE_CHECKING:
    from .configuration_lxmert import LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP, LxmertConfig
    from .tokenization_lxmert import LxmertTokenizer

    # 尝试检查是否存在 tokenizers，如果不存在则忽略导入
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果存在 tokenizers，则导入 tokenization_lxmert_fast 模块
        from .tokenization_lxmert_fast import LxmertTokenizerFast

    # 尝试检查是否存在 torch，如果不存在则忽略导入
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果存在 torch，则导入 modeling_lxmert 模块
        from .modeling_lxmert import (
            LxmertEncoder,
            LxmertForPreTraining,
            LxmertForQuestionAnswering,
            LxmertModel,
            LxmertPreTrainedModel,
            LxmertVisualFeatureEncoder,
            LxmertXLayer,
        )

    # 尝试检查是否存在 tensorflow，如果不存在则忽略导入
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果不是第一个情况，即没有直接从本地导入需要的模块，
        # 而是从当前包（package）中导入所需模块和类
        from .modeling_tf_lxmert import (
            TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST,  # 导入预训练模型的列表常量
            TFLxmertForPreTraining,  # 导入用于预训练的 TF LXMERT 模型
            TFLxmertMainLayer,  # 导入 TF LXMERT 主要层
            TFLxmertModel,  # 导入 TF LXMERT 模型
            TFLxmertPreTrainedModel,  # 导入 TF LXMERT 预训练模型基类
            TFLxmertVisualFeatureEncoder,  # 导入 TF LXMERT 视觉特征编码器
        )
else:
    # 如果不是以上任何情况，即当前模块并非主模块，需要导入 sys 模块进行处理
    import sys

    # 将当前模块(__name__)对应的模块对象替换为一个懒加载模块对象(_LazyModule)
    # _LazyModule会延迟加载模块内容，避免直接导入大量模块内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```
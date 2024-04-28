# `.\transformers\models\marian\__init__.py`

```
# 引入类型检查模块
from typing import TYPE_CHECKING
# 引入可选依赖未安装异常
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义导入结构字典，包含了模块及其导入内容
_import_structure = {
    "configuration_marian": ["MARIAN_PRETRAINED_CONFIG_ARCHIVE_MAP", "MarianConfig", "MarianOnnxConfig"],
}

# 尝试检查是否sentencepiece可用，若不可用则引发异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则将MarianTokenizer导入导入结构字典中
    _import_structure["tokenization_marian"] = ["MarianTokenizer"]

# 尝试检查是否torch可用，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则将modeling_marian模块相关内容导入导入结构字典中
    _import_structure["modeling_marian"] = [
        "MARIAN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MarianForCausalLM",
        "MarianModel",
        "MarianMTModel",
        "MarianPreTrainedModel",
    ]

# 尝试检查是否tensorflow可用，若不可用则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则将modeling_tf_marian模块相关内容导入导入结构字典中
    _import_structure["modeling_tf_marian"] = ["TFMarianModel", "TFMarianMTModel", "TFMarianPreTrainedModel"]

# 尝试检查是否flax可用，若不可用则引发异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则将modeling_flax_marian模块相关内容导入导入结构字典中
    _import_structure["modeling_flax_marian"] = ["FlaxMarianModel", "FlaxMarianMTModel", "FlaxMarianPreTrainedModel"]

# 如果是类型检查环境
if TYPE_CHECKING:
    # 从configuration_marian模块导入相关内容
    from .configuration_marian import MARIAN_PRETRAINED_CONFIG_ARCHIVE_MAP, MarianConfig, MarianOnnxConfig

    # 尝试检查是否sentencepiece可用，若不可用则引发异常
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则从tokenization_marian模块导入MarianTokenizer
        from .tokenization_marian import MarianTokenizer

    # 尝试检查是否torch可用，若不可用则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则从modeling_marian模块导入相关内容
        from .modeling_marian import (
            MARIAN_PRETRAINED_MODEL_ARCHIVE_LIST,
            MarianForCausalLM,
            MarianModel,
            MarianMTModel,
            MarianPreTrainedModel,
        )

    # 尝试检查是否tensorflow可用，若不可用则引发异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果未导入 TFMarianModel、TFMarianMTModel 和 TFMarianPreTrainedModel，则从 .modeling_tf_marian 模块中导入它们
    else:
        from .modeling_tf_marian import TFMarianModel, TFMarianMTModel, TFMarianPreTrainedModel
    
    # 尝试判断是否可以使用 Flax 库
    try:
        if not is_flax_available():
            # 如果 Flax 不可用，则引发 OptionalDependencyNotAvailable 异常
            raise OptionalDependencyNotAvailable()
    # 如果引发了 OptionalDependencyNotAvailable 异常，则什么也不做
    except OptionalDependencyNotAvailable:
        pass
    # 如果 Flax 可用，则从 .modeling_flax_marian 模块中导入 FlaxMarianModel、FlaxMarianMTModel 和 FlaxMarianPreTrainedModel
    else:
        from .modeling_flax_marian import FlaxMarianModel, FlaxMarianMTModel, FlaxMarianPreTrainedModel
# 如果之前的条件都不成立，则导入 sys 模块
import sys
# 将当前模块名添加到 sys.modules 中，对应的值为一个 _LazyModule 对象，该对象延迟加载模块的属性和子模块
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```
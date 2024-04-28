# `.\models\electra\__init__.py`

```
# 版权声明及许可证声明
#
# 版权所有 2020 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非遵守许可证，否则不得使用此文件。
# 您可以在以下链接处获得许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证发布的软件
# 基于“按原样提供”方式分发，不提供任何形式的担保或条件。
# 有关特定语言版本的详细信息，请参阅特定许可证。
# 根据许可证规定的权限和限制执行。

# 导入必要的库和模块，检查可用性
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义每个模块和相应的类或函数
_import_structure = {
    "configuration_electra": ["ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP", "ElectraConfig", "ElectraOnnxConfig"],
    "tokenization_electra": ["ElectraTokenizer"],
}

# 检查 tokenizers 是否可用，若不可用则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_electra_fast"] = ["ElectraTokenizerFast"]

# 检查 torch 是否可用，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_electra"] = [
        "ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ElectraForCausalLM",
        "ElectraForMaskedLM",
        "ElectraForMultipleChoice",
        "ElectraForPreTraining",
        "ElectraForQuestionAnswering",
        "ElectraForSequenceClassification",
        "ElectraForTokenClassification",
        "ElectraModel",
        "ElectraPreTrainedModel",
        "load_tf_weights_in_electra",
    ]

# 检查 tensorflow 是否可用，若不可用则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_electra"] = [
        "TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFElectraForMaskedLM",
        "TFElectraForMultipleChoice",
        "TFElectraForPreTraining",
        "TFElectraForQuestionAnswering",
        "TFElectraForSequenceClassification",
        "TFElectraForTokenClassification",
        "TFElectraModel",
        "TFElectraPreTrainedModel",
    ]

# 检查 flax 是否可用，若不可用则引发异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_electra"] = [
        "FlaxElectraForCausalLM",
        "FlaxElectraForMaskedLM",
        "FlaxElectraForMultipleChoice",
        "FlaxElectraForPreTraining",
        "FlaxElectraForQuestionAnswering",
        "FlaxElectraForSequenceClassification",
        "FlaxElectraForTokenClassification",
        "FlaxElectraModel",
        "FlaxElectraPreTrainedModel",
    ]

# 如果是类型检查，则执行以下代码
if TYPE_CHECKING:
    # 从当前目录下的 configuration_electra 模块中导入 ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig, ElectraOnnxConfig
    # 以及从 tokenization_electra 模块中导入 ElectraTokenizer
    from .configuration_electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig, ElectraOnnxConfig
    from .tokenization_electra import ElectraTokenizer
    
    # 尝试检测是否安装了 tokenizers 库，若未安装则抛出 OptionalDependencyNotAvailable 异常并捕获
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 若未安装 tokenizers 库，则不做任何处理
        pass
    else:
        # 若安装了 tokenizers 库，则从 tokenization_electra_fast 模块中导入 ElectraTokenizerFast
        from .tokenization_electra_fast import ElectraTokenizerFast
    
    # 尝试检测是否安装了 PyTorch 库，若未安装则抛出 OptionalDependencyNotAvailable 异常并捕获
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 若未安装 PyTorch 库，则不做任何处理
        pass
    else:
        # 若安装了 PyTorch 库，则从 modeling_electra 模块中导入一系列 Electra 模型相关的类和函数
        from .modeling_electra import (
            ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
            ElectraForCausalLM,
            ElectraForMaskedLM,
            ElectraForMultipleChoice,
            ElectraForPreTraining,
            ElectraForQuestionAnswering,
            ElectraForSequenceClassification,
            ElectraForTokenClassification,
            ElectraModel,
            ElectraPreTrainedModel,
            load_tf_weights_in_electra,
        )
    
    # 尝试检测是否安装了 TensorFlow 库，若未安装则抛出 OptionalDependencyNotAvailable 异常并捕获
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 若未安装 TensorFlow 库，则不做任何处理
        pass
    else:
        # 若安装了 TensorFlow 库，则从 modeling_tf_electra 模块中导入一系列 TensorFlow 版本的 Electra 模型相关的类和函数
        from .modeling_tf_electra import (
            TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFElectraForMaskedLM,
            TFElectraForMultipleChoice,
            TFElectraForPreTraining,
            TFElectraForQuestionAnswering,
            TFElectraForSequenceClassification,
            TFElectraForTokenClassification,
            TFElectraModel,
            TFElectraPreTrainedModel,
        )
    
    # 尝试检测是否安装了 Flax 库，若未安装则抛出 OptionalDependencyNotAvailable 异常并捕获
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 若未安装 Flax 库，则不做任何处理
        pass
    else:
        # 若安装了 Flax 库，则从 modeling_flax_electra 模块中导入一系列 Flax 版本的 Electra 模型相关的类和函数
        from .modeling_flax_electra import (
            FlaxElectraForCausalLM,
            FlaxElectraForMaskedLM,
            FlaxElectraForMultipleChoice,
            FlaxElectraForPreTraining,
            FlaxElectraForQuestionAnswering,
            FlaxElectraForSequenceClassification,
            FlaxElectraForTokenClassification,
            FlaxElectraModel,
            FlaxElectraPreTrainedModel,
        )
    ```  
else:
    # 导入系统模块
    import sys

    # 将当前模块注册到 sys.modules 字典中，使之成为一个懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```
# `.\models\ernie\__init__.py`

```
# 版权和许可信息，以及免责声明
# 声明 2022年 HuggingFace 团队保留所有权利，并根据 Apache License 2.0 版本授权。
# 提醒该软件在“按原样”的基础上发布，没有任何担保和保证。

# 导入 `typing` 模块中与类型检查相关的内容
from typing import TYPE_CHECKING

# 从相对路径导入一些实用工具和函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tensorflow_text_available, is_torch_available

# 创建一个字典，用于描述需要导入的模块和对象
_import_structure = {
    # 配置 `configuration_ernie` 模块中的项目
    "configuration_ernie": ["ERNIE_PRETRAINED_CONFIG_ARCHIVE_MAP", "ErnieConfig", "ErnieOnnxConfig"],
}

# 尝试检查 PyTorch 是否可用
try:
    # 如果 PyTorch 不可用，抛出自定义异常
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果发生异常，继续执行，不做进一步操作
    pass
else:
    # 如果 PyTorch 可用，则将 ERNIE 模型相关内容加入 `_import_structure`
    _import_structure["modeling_ernie"] = [
        "ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ErnieForCausalLM",
        "ErnieForMaskedLM",
        "ErnieForMultipleChoice",
        "ErnieForNextSentencePrediction",
        "ErnieForPreTraining",
        "ErnieForQuestionAnswering",
        "ErnieForSequenceClassification",
        "ErnieForTokenClassification",
        "ErnieModel",
        "ErniePreTrainedModel",
    ]

# 如果在类型检查阶段
if TYPE_CHECKING:
    # 导入 `configuration_ernie` 模块中的项目
    from .configuration_ernie import ERNIE_PRETRAINED_CONFIG_ARCHIVE_MAP, ErnieConfig, ErnieOnnxConfig

    # 再次尝试检查 PyTorch 是否可用
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 PyTorch 不可用，继续执行
        pass
    else:
        # 如果 PyTorch 可用，导入 `modeling_ernie` 中的项目
        from .modeling_ernie import (
            ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST,
            ErnieForCausalLM,
            ErnieForMaskedLM,
            ErnieForMultipleChoice,
            ErnieForNextSentencePrediction,
            ErnieForPreTraining,
            ErnieForQuestionAnswering,
            ErnieForSequenceClassification,
            ErnieForTokenClassification,
            ErnieModel,
            ErniePreTrainedModel,
        )

# 如果不在类型检查阶段
else:
    # 导入 `sys` 模块，用于动态替换模块
    import sys

    # 将当前模块替换为 `_LazyModule`，以便延迟加载并减少内存占用
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
```
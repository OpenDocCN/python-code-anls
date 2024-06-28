# `.\models\patchtsmixer\__init__.py`

```py
代码：


#
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
## 注意：这个代码块是用来组织和提供开源代码许可的元数据和项目结构的，以及决定不同组件的可用性。
#
## 实现了一个标准的包装和加载元数据的逻辑结构，用于表明项目是如何组织的，并提供了用于组件切换的功能，即对于需要依赖torch的组件取决于torch是否可用。

# 引入必要的依赖结构
from typing import TYPE_CHECKING

# 创建一个字典用于存储预训练的配置文件的下载链接和相关类名。
_import_structure = {
    "configuration_patchtsmixer": [
        "PATCHTSMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 字典包含预训练配置文件的远程链接列表
        "PatchTSMixerConfig",  # 代表模型配置类名
    ]
}

try:
    # 尝试检查torch是否可用，
    if not is_torch_available():  # 如果torch可用性检查结果为False
        raise OptionalDependencyNotAvailable()  # 抛出自定义异常
except OptionalDependencyNotAvailable:
    pass  # 如果torch不可用则忽略后续代码并直接通过is_torch_available()
else:
    # 如果torch可用，从模型层引入不同模型类相关的预训练链接和类名，
    _import_structure["modeling_patchtsmixer"] = [
        "PATCHTSMIXER_PRETRAINED_MODEL_ARCHIVE_LIST",  # 模型预训练集合的远程链接列表
        "PatchTSMixerPreTrainedModel",  # 代表预训练模型类名
        "PatchTSMixerModel",  # 代表主模型类名
        "PatchTSMixerForPretraining",  # 代表用于预训练的模型类名
        "PatchTSMixerForPrediction",  # 代表预测任务的模型类名
        "PatchTSMixerForTimeSeriesClassification",  # 代表时序分类任务的模型类名
        "PatchTSMixerForRegression",  # 代表回归任务的模型类名
    ]

## 默认情况下提供IDE检察功能：
if TYPE_CHECKING:
    # 为装修或需要导入类型提示的模块，引入相关类的类型信息。
    from .configuration_patchtsmixer import (
        PATCHTSMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练模型配置信息类型
        PatchTSMixerConfig,  # 配置类的具体类型引用
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()  # 修改内部状态检查处，但对于IDE注释策略，提供了类型检查支持
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果在指定条件下（即已确认没有torch依赖），导入模型类的部分及方法类型定义。
        from .modeling_patchtsmixer import (
            PATCHTSMIXER_PRETRAINED_MODEL_ARCHIVE_LIST,  # 预训练模型集合类型
            PatchTSMixerForPrediction,  # 预定义预测类类型
            PatchTSMixerForPretraining,  # 预定义预训练类类型
            PatchTSMixerForRegression,  # 预定义回归任务类类型
            PatchTSMixerForTimeSeriesClassification,  # 预定义时间序列分类任务类类型
            PatchTSMixerModel,  # 主模型类类型
            PatchTSMixerPreTrainedModel,  # 预训练模型基类类型
        )

# 在实际使用中，如果代码块遵循模块和类/函数的组织结构，会接受相对导入（如上面的例子），正确导入依赖并延迟直接执行过程。
else:
    # 空导入模块，通过创建一个惰性加载模块，将容器和绝对/相对导入组织当作元数据和初始检查逻辑，并封盖现有模块状态，以节省资源。
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```
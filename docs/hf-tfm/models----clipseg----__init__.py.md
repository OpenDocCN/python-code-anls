# `.\models\clipseg\__init__.py`

```
# 版权声明及许可信息
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入自定义的异常类和模块延迟加载工具函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_clipseg": [
        "CLIPSEG_PRETRAINED_CONFIG_ARCHIVE_MAP",  # CLIPSEG 预训练配置文件映射
        "CLIPSegConfig",  # CLIPSeg 模型配置
        "CLIPSegTextConfig",  # CLIPSeg 文本模型配置
        "CLIPSegVisionConfig",  # CLIPSeg 视觉模型配置
    ],
    "processing_clipseg": ["CLIPSegProcessor"],  # CLIPSeg 处理器模块
}

# 检查是否导入了 torch 模块，若未导入则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若导入成功则添加以下模块到导入结构中
    _import_structure["modeling_clipseg"] = [
        "CLIPSEG_PRETRAINED_MODEL_ARCHIVE_LIST",  # CLIPSEG 预训练模型归档列表
        "CLIPSegModel",  # CLIPSeg 模型
        "CLIPSegPreTrainedModel",  # CLIPSeg 预训练模型基类
        "CLIPSegTextModel",  # CLIPSeg 文本模型
        "CLIPSegVisionModel",  # CLIPSeg 视觉模型
        "CLIPSegForImageSegmentation",  # 用于图像分割的 CLIPSeg 模型
    ]

# 如果当前环境支持类型检查，则从相关模块导入具体类和常量
if TYPE_CHECKING:
    from .configuration_clipseg import (
        CLIPSEG_PRETRAINED_CONFIG_ARCHIVE_MAP,  # CLIPSEG 预训练配置文件映射
        CLIPSegConfig,  # CLIPSeg 模型配置
        CLIPSegTextConfig,  # CLIPSeg 文本模型配置
        CLIPSegVisionConfig,  # CLIPSeg 视觉模型配置
    )
    from .processing_clipseg import CLIPSegProcessor  # CLIPSeg 处理器模块

    # 检查是否导入了 torch 模块，若未导入则跳过
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若导入成功则从相关模块导入具体类和常量
        from .modeling_clipseg import (
            CLIPSEG_PRETRAINED_MODEL_ARCHIVE_LIST,  # CLIPSEG 预训练模型归档列表
            CLIPSegForImageSegmentation,  # 用于图像分割的 CLIPSeg 模型
            CLIPSegModel,  # CLIPSeg 模型
            CLIPSegPreTrainedModel,  # CLIPSeg 预训练模型基类
            CLIPSegTextModel,  # CLIPSeg 文本模型
            CLIPSegVisionModel,  # CLIPSeg 视觉模型
        )

# 若不是类型检查模式，则使用懒加载模块加载当前模块结构
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```
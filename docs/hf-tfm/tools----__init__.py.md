# `.\transformers\tools\__init__.py`

```
#!/usr/bin/env python
# coding=utf-8

# 版权声明
# 版权所有 © 2023 年 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，
# 没有任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言的权限和限制。
from typing import TYPE_CHECKING

from ..utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {
    "agents": ["Agent", "AzureOpenAiAgent", "HfAgent", "LocalAgent", "OpenAiAgent"],
    "base": ["PipelineTool", "RemoteTool", "Tool", "launch_gradio_demo", "load_tool"],
}

# 检查是否存在 torch 库
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 库，则添加以下模块到导入结构中
    _import_structure["document_question_answering"] = ["DocumentQuestionAnsweringTool"]
    _import_structure["image_captioning"] = ["ImageCaptioningTool"]
    _import_structure["image_question_answering"] = ["ImageQuestionAnsweringTool"]
    _import_structure["image_segmentation"] = ["ImageSegmentationTool"]
    _import_structure["speech_to_text"] = ["SpeechToTextTool"]
    _import_structure["text_classification"] = ["TextClassificationTool"]
    _import_structure["text_question_answering"] = ["TextQuestionAnsweringTool"]
    _import_structure["text_summarization"] = ["TextSummarizationTool"]
    _import_structure["text_to_speech"] = ["TextToSpeechTool"]
    _import_structure["translation"] = ["TranslationTool"]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入特定模块
    from .agents import Agent, AzureOpenAiAgent, HfAgent, LocalAgent, OpenAiAgent
    from .base import PipelineTool, RemoteTool, Tool, launch_gradio_demo, load_tool

    # 再次检查是否存在 torch 库
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果存在 torch 库，则导入以下模块
        from .document_question_answering import DocumentQuestionAnsweringTool
        from .image_captioning import ImageCaptioningTool
        from .image_question_answering import ImageQuestionAnsweringTool
        from .image_segmentation import ImageSegmentationTool
        from .speech_to_text import SpeechToTextTool
        from .text_classification import TextClassificationTool
        from .text_question_answering import TextQuestionAnsweringTool
        from .text_summarization import TextSummarizationTool
        from .text_to_speech import TextToSpeechTool
        from .translation import TranslationTool
else:
    import sys

    # 将当前模块设置为 LazyModule
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```
# `.\tools\__init__.py`

```py
#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

# 导入必要的类型检查工具
from typing import TYPE_CHECKING

# 导入自定义的异常
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

# 尝试导入 Torch，若不可用则抛出自定义异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，扩展导入结构
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

# 如果进行类型检查，则进一步导入具体模块
if TYPE_CHECKING:
    from .agents import Agent, AzureOpenAiAgent, HfAgent, LocalAgent, OpenAiAgent
    from .base import PipelineTool, RemoteTool, Tool, launch_gradio_demo, load_tool

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Torch 可用，详细导入相关工具类
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

    # 如果不是类型检查，使用 LazyModule 进行懒加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```
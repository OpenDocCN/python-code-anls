# `.\comic-translate\modules\ocr\pororo\pororo\tasks\__init__.py`

```py
# flake8: noqa
"""
__init__.py for import child .py files

    isort:skip_file
"""

# 从pororo.tasks.utils.download_utils模块中导入download_or_load函数，用于下载或加载资源
from ...pororo.tasks.utils.download_utils import download_or_load
# 从pororo.tasks.utils.base模块中导入多个基类，用于实现不同任务的工厂和基类
from ...pororo.tasks.utils.base import (
    PororoBiencoderBase,           # 双编码器基类
    PororoFactoryBase,            # 工厂基类
    PororoGenerationBase,         # 生成任务基类
    PororoSimpleBase,             # 简单任务基类
    PororoTaskGenerationBase,     # 任务生成基类
)

# 从pororo.tasks.optical_character_recognition模块中导入PororoOcrFactory类，用于光学字符识别任务的工厂类
from ...pororo.tasks.optical_character_recognition import PororoOcrFactory
```
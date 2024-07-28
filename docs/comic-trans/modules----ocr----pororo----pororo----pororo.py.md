# `.\comic-translate\modules\ocr\pororo\pororo\pororo.py`

```py
"""
Pororo task-specific factory class

    isort:skip_file

"""

# 导入必要的模块和库
import logging  # 导入日志记录模块
from typing import Optional  # 引入类型提示中的 Optional 类型
from ..pororo.tasks.utils.base import PororoTaskBase  # 导入基础任务类

import torch  # 导入 PyTorch 库

from ..pororo.tasks import (
    PororoOcrFactory,  # 导入 OCR 任务的工厂类
)

# 支持的任务类型和对应的工厂类映射关系
SUPPORTED_TASKS = {
    "ocr": PororoOcrFactory,
}

# 语言别名映射表
LANG_ALIASES = {
    "english": "en",
    "eng": "en",
    "korean": "ko",
    "kor": "ko",
    "kr": "ko",
    "chinese": "zh",
    "chn": "zh",
    "cn": "zh",
    "japanese": "ja",
    "jap": "ja",
    "jp": "ja",
    "jejueo": "je",
    "jje": "je",
}

# 设置特定模块的日志级别为 WARN，以避免过多的日志输出
logging.getLogger("transformers").setLevel(logging.WARN)
logging.getLogger("fairseq").setLevel(logging.WARN)
logging.getLogger("sentence_transformers").setLevel(logging.WARN)
logging.getLogger("youtube_dl").setLevel(logging.WARN)
logging.getLogger("pydub").setLevel(logging.WARN)
logging.getLogger("librosa").setLevel(logging.WARN)


class Pororo:
    r"""
    This is a generic class that will return one of the task-specific model classes of the library
    when created with the `__new__()` method

    """

    def __new__(
        cls,
        task: str,
        lang: str = "en",
        model: Optional[str] = None,
        **kwargs,
    ) -> PororoTaskBase:
        # 检查任务是否在支持的任务列表中，否则引发 KeyError 异常
        if task not in SUPPORTED_TASKS:
            raise KeyError("Unknown task {}, available tasks are {}".format(
                task,
                list(SUPPORTED_TASKS.keys()),
            ))

        # 将语言转换为小写，并在需要时使用语言别名映射
        lang = lang.lower()
        lang = LANG_ALIASES[lang] if lang in LANG_ALIASES else lang

        # 根据是否有 CUDA 支持，确定设备类型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 实例化特定任务的模块，加载到指定的设备上
        task_module = SUPPORTED_TASKS[task](
            task,
            lang,
            model,
            **kwargs,
        ).load(device)

        return task_module

    @staticmethod
    def available_tasks() -> str:
        """
        Returns available tasks in Pororo project

        Returns:
            str: Supported task names

        """
        return "Available tasks are {}".format(list(SUPPORTED_TASKS.keys()))

    @staticmethod
    def available_models(task: str) -> str:
        """
        Returns available model names correponding to the user-input task

        Args:
            task (str): user-input task name

        Returns:
            str: Supported model names corresponding to the user-input task

        Raises:
            KeyError: When user-input task is not supported

        """
        # 如果任务不在支持的任务列表中，则引发 KeyError 异常
        if task not in SUPPORTED_TASKS:
            raise KeyError(
                "Unknown task {} ! Please check available models via `available_tasks()`"
                .format(task))

        # 获取特定任务可用的模型列表，并格式化输出
        langs = SUPPORTED_TASKS[task].get_available_models()
        output = f"Available models for {task} are "
        for lang in langs:
            output += f"([lang]: {lang}, [model]: {', '.join(langs[lang])}), "
        return output[:-2]
```
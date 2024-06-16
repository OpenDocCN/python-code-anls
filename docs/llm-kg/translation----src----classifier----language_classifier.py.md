# `.\translation\src\classifier\language_classifier.py`

```
import yaml  # 导入yaml模块，用于读取和解析YAML格式的配置文件
from transformers import pipeline  # 导入transformers库中的pipeline函数，用于自然语言处理任务

from base.config import Config  # 从base.config模块中导入Config类，用于加载配置信息


class LanguageDetector(Config):
    """XLMRoBERTa for language detection"""

    def __init__(self) -> None:
        super().__init__()  # 调用Config类的构造函数，初始化配置信息
        self.model = pipeline(
            "text-classification",  # 使用文本分类任务的pipeline
            model=self.config["classifier"]["model"],  # 指定模型名称，从配置中获取
            device=self.config["classifier"]["device"],  # 指定设备，从配置中获取
        )

    def detect_language(self, document: str) -> str:
        """
        Detects the language of a document
        Args:
            document (str): document for language detection
        Returns:
            str: language
        """

        lang = self.model([document])[0]["label"]  # 使用模型预测文档的语言标签

        return self.lang_map[lang]  # 返回预测语言对应的真实语言名称
```
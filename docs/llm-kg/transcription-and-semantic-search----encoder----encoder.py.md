# `.\transcription-and-semantic-search\encoder\encoder.py`

```
from langchain.embeddings import HuggingFaceEmbeddings  # 导入 HuggingFaceEmbeddings 类，用于文本的嵌入表示

from base.config import Config  # 导入 Config 类，用于配置管理


class Encoder(Config):
    """Encoder to create workds embeddings from text"""
    # Encoder 类继承自 Config 类，用于从文本创建词嵌入表示

    def __init__(self) -> None:
        super().__init__()  # 调用父类 Config 的初始化方法
        self.encoder = HuggingFaceEmbeddings(
            # 初始化 HuggingFaceEmbeddings 实例，使用配置中指定的模型路径、模型参数和编码参数
            model_name=self.config["encoder"]["model_path"],  # 使用配置文件中指定的模型路径
            model_kwargs=self.config["encoder"]["model_kwargs"],  # 使用配置文件中指定的模型参数
            encode_kwargs=self.config["encoder"]["encode_kwargs"],  # 使用配置文件中指定的编码参数
        )
```
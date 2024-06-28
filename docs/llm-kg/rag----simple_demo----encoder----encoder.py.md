# `.\rag\simple_demo\encoder\encoder.py`

```
from base.config import Config  # 导入 Config 类，用于配置管理
from langchain.embeddings import HuggingFaceEmbeddings  # 导入 HuggingFaceEmbeddings 类，用于处理文本的嵌入向量化


class Encoder(Config):
    """Encoder to create workds embeddings from text"""

    def __init__(self) -> None:
        super().__init__()  # 调用父类 Config 的初始化方法
        # 初始化一个 HuggingFaceEmbeddings 对象，使用配置文件中指定的模型路径、模型参数和编码参数
        self.encoder = HuggingFaceEmbeddings(
            model_name=self.config["encoder"]["model_path"],  # 模型路径从配置中获取
            model_kwargs=self.config["encoder"]["model_kwargs"],  # 模型参数从配置中获取
            encode_kwargs=self.config["encoder"]["encode_kwargs"],  # 编码参数从配置中获取
        )
```
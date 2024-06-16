# `.\translation\src\encoder\encoder.py`

```
# 导入HuggingFaceEmbeddings类，该类用于处理Hugging Face提供的嵌入模型
from langchain.embeddings import HuggingFaceEmbeddings

# 从base.config模块中导入Config类，用于配置参数和设置
from base.config import Config


# 定义一个名为Encoder的类，继承自Config类，用于创建文本的词嵌入
class Encoder(Config):
    """Encoder to create workds embeddings from text"""

    # 初始化方法，设置类的初始状态
    def __init__(self) -> None:
        # 调用父类Config的初始化方法
        super().__init__()
        # 创建一个HuggingFaceEmbeddings对象，用于根据配置文件中指定的模型路径和参数进行文本编码
        self.encoder = HuggingFaceEmbeddings(
            model_name=self.config["encoder"]["model_path"],  # 使用配置文件中指定的模型路径
            model_kwargs=self.config["encoder"]["model_kwargs"],  # 使用配置文件中指定的模型参数
            encode_kwargs=self.config["encoder"]["encode_kwargs"],  # 使用配置文件中指定的编码参数
        )
```
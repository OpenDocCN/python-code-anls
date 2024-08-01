# `.\DB-GPT-src\dbgpt\rag\embedding\__init__.py`

```py
# 用于导入与嵌入相关的类和函数的模块

from .embedding_factory import (  # noqa: F401
    DefaultEmbeddingFactory,    # 导入默认嵌入工厂类
    EmbeddingFactory,           # 导入通用嵌入工厂类
    WrappedEmbeddingFactory,    # 导入包装后的嵌入工厂类
)

from .embeddings import (  # noqa: F401
    Embeddings,                         # 导入通用嵌入类
    HuggingFaceBgeEmbeddings,           # 导入Hugging Face BGE嵌入类
    HuggingFaceEmbeddings,              # 导入Hugging Face通用嵌入类
    HuggingFaceInferenceAPIEmbeddings,  # 导入Hugging Face推断API嵌入类
    HuggingFaceInstructEmbeddings,      # 导入Hugging Face Instruct嵌入类
    JinaEmbeddings,                     # 导入Jina嵌入类
    OllamaEmbeddings,                   # 导入Ollama嵌入类
    OpenAPIEmbeddings,                  # 导入OpenAPI嵌入类
    TongYiEmbeddings,                   # 导入TongYi嵌入类
)

from .rerank import CrossEncoderRerankEmbeddings, OpenAPIRerankEmbeddings  # noqa: F401
# 导入用于重新排序的交叉编码器重新排序嵌入类和OpenAPI重新排序嵌入类

__ALL__ = [  # 导出的模块成员列表
    "Embeddings",                               # 通用嵌入类
    "HuggingFaceBgeEmbeddings",                 # Hugging Face BGE嵌入类
    "HuggingFaceEmbeddings",                    # Hugging Face通用嵌入类
    "HuggingFaceInferenceAPIEmbeddings",        # Hugging Face推断API嵌入类
    "HuggingFaceInstructEmbeddings",            # Hugging Face Instruct嵌入类
    "JinaEmbeddings",                           # Jina嵌入类
    "OpenAPIEmbeddings",                        # OpenAPI嵌入类
    "OllamaEmbeddings",                         # Ollama嵌入类
    "DefaultEmbeddingFactory",                  # 默认嵌入工厂类
    "EmbeddingFactory",                         # 通用嵌入工厂类
    "WrappedEmbeddingFactory",                  # 包装后的嵌入工厂类
    "TongYiEmbeddings",                         # TongYi嵌入类
    "CrossEncoderRerankEmbeddings",             # 交叉编码器重新排序嵌入类
    "OpenAPIRerankEmbeddings",                  # OpenAPI重新排序嵌入类
]
```
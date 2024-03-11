# `.\Langchain-Chatchat\server\reranker\reranker.py`

```
# 导入 os 模块
import os
# 导入 sys 模块
import sys

# 将当前文件的父目录的父目录的父目录添加到 sys.path 中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# 从 typing 模块中导入 Any, List, Optional 类型
from typing import Any, List, Optional
# 从 sentence_transformers 模块中导入 CrossEncoder 类
from sentence_transformers import CrossEncoder
# 从 typing 模块中再次导入 Optional, Sequence 类型
from typing import Optional, Sequence
# 从 langchain_core.documents 模块中导入 Document 类
from langchain_core.documents import Document
# 从 langchain.callbacks.manager 模块中导入 Callbacks 类
from langchain.callbacks.manager import Callbacks
# 从 langchain.retrievers.document_compressors.base 模块中导入 BaseDocumentCompressor 类
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
# 从 llama_index.bridge.pydantic 模块中导入 Field, PrivateAttr 类
from llama_index.bridge.pydantic import Field, PrivateAttr

# 定义 LangchainReranker 类，继承自 BaseDocumentCompressor 类
class LangchainReranker(BaseDocumentCompressor):
    """Document compressor that uses `Cohere Rerank API`."""
    # 模型名称或路径的字符串属性
    model_name_or_path: str = Field()
    # 私有属性 _model，类型为 Any
    _model: Any = PrivateAttr()
    # top_n 属性，表示返回的结果数量
    top_n: int = Field()
    # 设备属性，表示模型运行的设备
    device: str = Field()
    # 最大长度属性，表示输入文本的最大长度
    max_length: int = Field()
    # 批处理大小属性，表示每个批次的样本数量
    batch_size: int = Field()
    # 显示进度条属性
    # show_progress_bar: bool = None
    # 工作线程数量属性
    num_workers: int = Field()

    # 激活函数属性
    # activation_fct = None
    # 是否应用 softmax 属性
    # apply_softmax = False
    # 初始化函数，设置模型名称或路径、返回结果数量、设备类型、最大长度、批处理大小、工作线程数等参数
    def __init__(self,
                 model_name_or_path: str,
                 top_n: int = 3,
                 device: str = "cuda",
                 max_length: int = 1024,
                 batch_size: int = 32,
                 # show_progress_bar: bool = None,
                 num_workers: int = 0,
                 # activation_fct = None,
                 # apply_softmax = False,
                 ):
        # self.top_n=top_n
        # self.model_name_or_path=model_name_or_path
        # self.device=device
        # self.max_length=max_length
        # self.batch_size=batch_size
        # self.show_progress_bar=show_progress_bar
        # self.num_workers=num_workers
        # self.activation_fct=activation_fct
        # self.apply_softmax=apply_softmax

        # 初始化交叉编码器模型
        self._model = CrossEncoder(model_name=model_name_or_path, max_length=1024, device=device)
        # 调用父类的初始化函数，设置参数
        super().__init__(
            top_n=top_n,
            model_name_or_path=model_name_or_path,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            # show_progress_bar=show_progress_bar,
            num_workers=num_workers,
            # activation_fct=activation_fct,
            # apply_softmax=apply_softmax
        )

    # 压缩文档函数，接收文档序列、查询字符串和回调函数作为参数
    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Cohere's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # 检查文档列表是否为空，以避免空的 API 调用
            return []
        doc_list = list(documents)  # 将文档列表转换为列表
        _docs = [d.page_content for d in doc_list]  # 提取文档内容并存储在 _docs 列表中
        sentence_pairs = [[query, _doc] for _doc in _docs]  # 创建包含查询和文档内容的句子对列表
        results = self._model.predict(sentences=sentence_pairs,
                                      batch_size=self.batch_size,
                                      #  show_progress_bar=self.show_progress_bar,
                                      num_workers=self.num_workers,
                                      #  activation_fct=self.activation_fct,
                                      #  apply_softmax=self.apply_softmax,
                                      convert_to_tensor=True
                                      )  # 使用模型预测句子对的相关性得分

        top_k = self.top_n if self.top_n < len(results) else len(results)  # 确定要保留的前 k 个结果数量

        values, indices = results.topk(top_k)  # 获取前 k 个结果的值和索引
        final_results = []  # 存储最终结果的列表
        for value, index in zip(values, indices):
            doc = doc_list[index]  # 获取与索引对应的文档
            doc.metadata["relevance_score"] = value  # 将相关性分数存储在文档的元数据中
            final_results.append(doc)  # 将文档添加到最终结果列表中
        return final_results  # 返回最终结果列表
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 从配置文件中导入所需的配置项
    from configs import (LLM_MODELS,
                         VECTOR_SEARCH_TOP_K,
                         SCORE_THRESHOLD,
                         TEMPERATURE,
                         USE_RERANKER,
                         RERANKER_MODEL,
                         RERANKER_MAX_LENGTH,
                         MODEL_PATH)
    # 从工具模块中导入 embedding_device 函数
    from server.utils import embedding_device

    # 如果使用 reranker
    if USE_RERANKER:
        # 获取 reranker 模型的路径
        reranker_model_path = MODEL_PATH["reranker"].get(RERANKER_MODEL, "BAAI/bge-reranker-large")
        # 打印模型路径信息
        print("-----------------model path------------------")
        print(reranker_model_path)
        # 初始化 LangchainReranker 对象
        reranker_model = LangchainReranker(top_n=3,
                                           device=embedding_device(),
                                           max_length=RERANKER_MAX_LENGTH,
                                           model_name_or_path=reranker_model_path
                                           )
```
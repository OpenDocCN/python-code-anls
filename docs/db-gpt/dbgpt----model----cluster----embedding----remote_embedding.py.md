# `.\DB-GPT-src\dbgpt\model\cluster\embedding\remote_embedding.py`

```py
from typing import List
# 导入 List 类型

from dbgpt.core import Embeddings, RerankEmbeddings
# 从 dbgpt.core 模块中导入 Embeddings 和 RerankEmbeddings 类
from dbgpt.model.cluster.manager_base import WorkerManager
# 从 dbgpt.model.cluster.manager_base 模块中导入 WorkerManager 类

class RemoteEmbeddings(Embeddings):
    # 定义 RemoteEmbeddings 类，继承自 Embeddings 类
    def __init__(self, model_name: str, worker_manager: WorkerManager) -> None:
        # 初始化方法，接受模型名称和 WorkerManager 对象作为参数
        self.model_name = model_name
        # 设置实例变量 model_name 为传入的模型名称
        self.worker_manager = worker_manager
        # 设置实例变量 worker_manager 为传入的 WorkerManager 对象

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 定义 embed_documents 方法，接受文本列表作为参数，返回嵌入向量列表
        """Embed search docs."""
        # 方法的文档字符串
        params = {"model": self.model_name, "input": texts}
        # 构建参数字典
        return self.worker_manager.sync_embeddings(params)
        # 调用 worker_manager 的 sync_embeddings 方法并返回结果

    def embed_query(self, text: str) -> List[float]:
        # 定义 embed_query 方法，接受查询文本作为参数，返回嵌入向量
        """Embed query text."""
        # 方法的文档字符串
        return self.embed_documents([text])[0]
        # 调用 embed_documents 方法并返回第一个元素

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        # 定义异步 embed_documents 方法，接受文本列表作为参数，返回嵌入向量列表
        """Asynchronous Embed search docs."""
        # 方法的文档字符串
        params = {"model": self.model_name, "input": texts}
        # 构建参数字典
        return await self.worker_manager.embeddings(params)
        # 调用 worker_manager 的 embeddings 方法并返回结果

    async def aembed_query(self, text: str) -> List[float]:
        # 定义异步 embed_query 方法，接受查询文本作为参数，返回嵌入向量
        """Asynchronous Embed query text."""
        # 方法的文档字符串
        return await self.aembed_documents([text])[0]
        # 调用 aembed_documents 方法并返回第一个元素

class RemoteRerankEmbeddings(RerankEmbeddings):
    # 定义 RemoteRerankEmbeddings 类，继承自 RerankEmbeddings 类
    def __init__(self, model_name: str, worker_manager: WorkerManager) -> None:
        # 初始化方法，接受模型名称和 WorkerManager 对象作为参数
        self.model_name = model_name
        # 设置实例变量 model_name 为传入的模型名称
        self.worker_manager = worker_manager
        # 设置实例变量 worker_manager 为传入的 WorkerManager 对象

    def predict(self, query: str, candidates: List[str]) -> List[float]:
        # 定义 predict 方法，接受查询文本和候选文本列表作为参数，返回分数列表
        """Predict the scores of the candidates."""
        # 方法的文档字符串
        params = {
            "model": self.model_name,
            "input": candidates,
            "query": query,
        }
        # 构建参数字典
        return self.worker_manager.sync_embeddings(params)[0]
        # 调用 worker_manager 的 sync_embeddings 方法并返回第一个元素

    async def apredict(self, query: str, candidates: List[str]) -> List[float]:
        # 定义异步 apredict 方法，接受查询文本和候选文本列表作为参数，返回分数列表
        """Asynchronously predict the scores of the candidates."""
        # 方法的文档字符串
        params = {
            "model": self.model_name,
            "input": candidates,
            "query": query,
        }
        # 构建参数字典
        scores = await self.worker_manager.embeddings(params)
        # 调用 worker_manager 的 embeddings 方法并将结果赋值给 scores
        return scores[0]
        # 返回 scores 的第一个元素
```
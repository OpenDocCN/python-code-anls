# `.\DB-GPT-src\dbgpt\model\cluster\worker\embedding_worker.py`

```py
import logging  # 导入日志模块
from typing import Dict, List, Type, Union  # 导入类型提示相关模块

from dbgpt.core import Embeddings, ModelMetadata, RerankEmbeddings  # 导入模型核心类
from dbgpt.model.adapter.embeddings_loader import (  # 导入嵌入加载器及其相关函数
    EmbeddingLoader,
    _parse_embedding_params,
)
from dbgpt.model.adapter.loader import _get_model_real_path  # 导入模型真实路径获取函数
from dbgpt.model.cluster.worker_base import ModelWorker  # 导入模型工作基类
from dbgpt.model.parameter import (  # 导入模型参数相关类和配置
    EMBEDDING_NAME_TO_PARAMETER_CLASS_CONFIG,
    BaseEmbeddingModelParameters,
    EmbeddingModelParameters,
    WorkerType,
)
from dbgpt.util.model_utils import _clear_model_cache  # 导入模型工具函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class EmbeddingsModelWorker(ModelWorker):
    def __init__(self, rerank_model: bool = False) -> None:
        self._embeddings_impl: Union[Embeddings, RerankEmbeddings, None] = None  # 嵌入实现对象或为空
        self._model_params = None  # 模型参数对象
        self.model_name = None  # 模型名称
        self.model_path = None  # 模型路径
        self._rerank_model = rerank_model  # 是否是重新排序模型
        self._loader = EmbeddingLoader()  # 嵌入加载器对象

    def load_worker(self, model_name: str, model_path: str, **kwargs) -> None:
        if model_path.endswith("/"):
            model_path = model_path[:-1]  # 如果模型路径以斜杠结尾，则去除斜杠
        model_path = _get_model_real_path(model_name, model_path)  # 获取模型的真实路径

        self.model_name = model_name  # 设置模型名称
        self.model_path = model_path  # 设置模型路径

    def worker_type(self) -> WorkerType:
        return WorkerType.TEXT2VEC  # 返回工作类型为文本向量化

    def model_param_class(self) -> Type:
        return EMBEDDING_NAME_TO_PARAMETER_CLASS_CONFIG.get(
            self.model_name, EmbeddingModelParameters
        )  # 返回模型参数类对象根据模型名称或默认为嵌入模型参数

    def parse_parameters(
        self, command_args: List[str] = None
    ) -> BaseEmbeddingModelParameters:
        param_cls = self.model_param_class()  # 获取模型参数类
        return _parse_embedding_params(
            model_name=self.model_name,
            model_path=self.model_path,
            command_args=command_args,
            param_cls=param_cls,
        )  # 解析模型参数并返回基础嵌入模型参数对象

    def start(
        self,
        model_params: EmbeddingModelParameters = None,
        command_args: List[str] = None,
    ) -> None:
        """Start model worker"""
        if not model_params:
            model_params = self.parse_parameters(command_args)  # 如果没有提供模型参数，则解析命令行参数为模型参数对象
        if self._rerank_model:
            model_params.rerank = True  # 如果是重新排序模型，则设置模型参数中的rerank属性为True（忽略类型检查）
        self._model_params = model_params  # 设置模型参数对象
        if model_params.is_rerank_model():
            logger.info(f"Load rerank embeddings model: {self.model_name}")  # 记录日志加载重新排序嵌入模型
            self._embeddings_impl = self._loader.load_rerank_model(
                self.model_name, model_params
            )  # 加载重新排序嵌入模型
        else:
            logger.info(f"Load embeddings model: {self.model_name}")  # 记录日志加载嵌入模型
            self._embeddings_impl = self._loader.load(self.model_name, model_params)  # 加载普通嵌入模型

    def __del__(self):
        self.stop()  # 析构函数，调用stop方法停止模型工作

    def stop(self) -> None:
        if not self._embeddings_impl:
            return  # 如果嵌入模型实现对象为空，则直接返回
        del self._embeddings_impl  # 删除嵌入模型实现对象
        self._embeddings_impl = None  # 将嵌入模型实现对象置为None
        _clear_model_cache(self._model_params.device)  # 清除模型缓存
    # 生成流式结果，适用于聊天场景
    def generate_stream(self, params: Dict):
        """Generate stream result, chat scene"""
        # 抛出未实现错误，因为嵌入模型不支持流式生成
        raise NotImplementedError("Not supported generate_stream for embeddings model")

    # 生成非流式结果
    def generate(self, params: Dict):
        """Generate non stream result"""
        # 抛出未实现错误，因为嵌入模型不支持生成非流式结果
        raise NotImplementedError("Not supported generate for embeddings model")

    # 计算提示文本中的标记数量
    def count_token(self, prompt: str) -> int:
        # 抛出未实现错误，因为嵌入模型不支持计算标记数量
        raise NotImplementedError("Not supported count_token for embeddings model")

    # 获取模型的元数据
    def get_model_metadata(self, params: Dict) -> ModelMetadata:
        # 抛出未实现错误，因为嵌入模型不支持获取模型元数据
        raise NotImplementedError(
            "Not supported get_model_metadata for embeddings model"
        )

    # 对文本进行嵌入处理，返回嵌入向量列表
    def embeddings(self, params: Dict) -> List[List[float]]:
        model = params.get("model")
        logger.info(f"Receive embeddings request, model: {model}")
        textx: List[str] = params["input"]
        
        # 如果嵌入实现是 RerankEmbeddings 类型，则进行查询和预测分数计算
        if isinstance(self._embeddings_impl, RerankEmbeddings):
            query = params["query"]
            scores: List[float] = self._embeddings_impl.predict(query, textx)
            return [scores]
        else:
            # 否则，使用嵌入实现对象处理文档并返回嵌入向量
            return self._embeddings_impl.embed_documents(textx)
```
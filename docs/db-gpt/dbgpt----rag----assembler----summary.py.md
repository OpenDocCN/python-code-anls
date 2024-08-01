# `.\DB-GPT-src\dbgpt\rag\assembler\summary.py`

```py
"""Summary Assembler."""
# 导入所需的模块和类
import os
from typing import Any, List, Optional

from dbgpt.core import Chunk, LLMClient

# 导入基类和相关模块
from ..assembler.base import BaseAssembler
from ..chunk_manager import ChunkParameters
from ..extractor.base import Extractor
from ..knowledge.base import Knowledge
from ..retriever.base import BaseRetriever


class SummaryAssembler(BaseAssembler):
    """Summary Assembler.

    Example:
       .. code-block:: python

           pdf_path = "../../../DB-GPT/docs/docs/awel.md"
           OPEN_AI_KEY = "{your_api_key}"
           OPEN_AI_BASE = "{your_api_base}"
           llm_client = OpenAILLMClient(api_key=OPEN_AI_KEY, api_base=OPEN_AI_BASE)
           knowledge = KnowledgeFactory.from_file_path(pdf_path)
           chunk_parameters = ChunkParameters(chunk_strategy="CHUNK_BY_SIZE")
           assembler = SummaryAssembler.load_from_knowledge(
               knowledge=knowledge,
               chunk_parameters=chunk_parameters,
               llm_client=llm_client,
               model_name="gpt-3.5-turbo",
           )
           summary = await assembler.generate_summary()
    """

    def __init__(
        self,
        knowledge: Knowledge,
        chunk_parameters: Optional[ChunkParameters] = None,
        model_name: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        extractor: Optional[Extractor] = None,
        language: Optional[str] = "en",
        **kwargs: Any,
    ) -> None:
        """Initialize with Embedding Assembler arguments.

        Args:
            knowledge: (Knowledge) Knowledge datasource.
            chunk_parameters: (Optional[ChunkParameters]) ChunkParameters to use for chunking.
            model_name: (Optional[str]) LLM model to use.
            llm_client: (Optional[LLMClient]) LLMClient to use.
            extractor: (Optional[Extractor]) Extractor to use for summarization.
            language: (Optional[str]) The language of the prompt. Defaults to "en".
        """
        # 检查必要参数是否被提供
        if knowledge is None:
            raise ValueError("knowledge datasource must be provided.")

        # 如果未提供模型名称，则从环境变量中获取
        model_name = model_name or os.getenv("LLM_MODEL")

        # 如果未提供提取器，则使用默认的摘要提取器
        if not extractor:
            from ..extractor.summary import SummaryExtractor

            # 如果未提供LLM客户端，则抛出异常
            if not llm_client:
                raise ValueError("llm_client must be provided.")
            # 如果未提供模型名称，则抛出异常
            if not model_name:
                raise ValueError("model_name must be provided.")
            # 创建摘要提取器对象
            extractor = SummaryExtractor(
                llm_client=llm_client,
                model_name=model_name,
                language=language,
            )
        # 如果提取器仍未提供，则抛出异常
        if not extractor:
            raise ValueError("extractor must be provided.")

        # 初始化摘要汇编器，设置提取器对象
        self._extractor: Extractor = extractor
        super().__init__(
            knowledge=knowledge,
            chunk_parameters=chunk_parameters,
            extractor=self._extractor,
            **kwargs,
        )

    @classmethod
    def load_from_knowledge(
        cls,
        knowledge: Knowledge,
        chunk_parameters: Optional[ChunkParameters] = None,
        model_name: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        extractor: Optional[Extractor] = None,
        language: Optional[str] = "en",
        **kwargs: Any,
    ) -> "SummaryAssembler":
        """Load document embedding into vector store from path.

        Args:
            knowledge: (Knowledge) Knowledge datasource.
                知识数据源。
            chunk_parameters: (Optional[ChunkParameters]) ChunkManager to use for
                chunking.
                用于分块的 ChunkManager 对象。
            model_name: (Optional[str]) llm model to use.
                要使用的 llm 模型名称。
            llm_client: (Optional[LLMClient]) LLMClient to use.
                要使用的 LLMClient 对象。
            extractor: (Optional[Extractor]) Extractor to use for summarization.
                用于摘要提取的 Extractor 对象。
            language: (Optional[str]) The language of the prompt. Defaults to "en".
                提示语言，默认为英语 ("en")。
        Returns:
             SummaryAssembler
                返回一个 SummaryAssembler 对象。
        """
        return cls(
            knowledge=knowledge,
            chunk_parameters=chunk_parameters,
            model_name=model_name,
            llm_client=llm_client,
            extractor=extractor,
            language=language,
            **kwargs,
        )

    async def generate_summary(self) -> str:
        """Generate summary."""
        return await self._extractor.aextract(self._chunks)
        # 调用 _extractor 对象的 aextract 方法来生成摘要并返回。

    def persist(self, **kwargs: Any) -> List[str]:
        """Persist chunks into store."""
        raise NotImplementedError
        # 抛出 NotImplementedError 异常，表示此方法需要在子类中实现。

    def _extract_info(self, chunks) -> List[Chunk]:
        """Extract info from chunks."""
        return []
        # 返回空列表，表示从 chunks 中提取信息，但当前实现未实际提取任何信息。

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a retriever."""
        raise NotImplementedError
        # 抛出 NotImplementedError 异常，表示此方法需要在子类中实现。
```
# `.\DB-GPT-src\dbgpt\rag\extractor\summary.py`

```py
"""Summary Extractor, it can extract document summary."""

# 导入所需的模块和类
from typing import List, Optional
from dbgpt._private.llm_metadata import LLMMetadata
from dbgpt.core import Chunk, LLMClient, ModelMessageRoleType, ModelRequest
from dbgpt.rag.extractor.base import Extractor
from dbgpt.util import utils
from dbgpt.util.chat_util import run_async_tasks

# 中文和英文的总结模板
SUMMARY_PROMPT_TEMPLATE_ZH = """请根据提供的上下文信息的进行精简地总结:
{context}
答案尽量精确和简单,不要过长，长度控制在100字左右, 注意:请用<中文>来进行总结。
"""

SUMMARY_PROMPT_TEMPLATE_EN = """
Write a quick summary of the following context:
{context}
the summary should be as concise as possible and not overly lengthy.Please keep the
answer within approximately 200 characters.
"""

REFINE_SUMMARY_TEMPLATE_ZH = """我们已经提供了一个到某一点的现有总结:{context}
请根据你之前推理的内容进行总结,总结回答的时候最好按照1.2.3.进行. 注意:请用<中文>来进行总结。
"""

REFINE_SUMMARY_TEMPLATE_EN = """
We have provided an existing summary up to a certain point: {context}, We have the
opportunity to refine the existing summary (only if needed) with some more context
below. \nBased on the previous reasoning, please summarize the final conclusion in
accordance with points 1.2.and 3.
"""

class SummaryExtractor(Extractor):
    """Summary Extractor, it can extract document summary."""

    def __init__(
        self,
        llm_client: LLMClient,
        model_name: str,
        llm_metadata: Optional[LLMMetadata] = None,
        language: Optional[str] = "en",
        max_iteration_with_llm: int = 5,
        concurrency_limit_with_llm: int = 3,
    ):
        """Create SummaryExtractor.

        Args:
            llm_client: (Optional[LLMClient]): The LLM client. Defaults to None.
            model_name: str
            llm_metadata: LLMMetadata
            language: (Optional[str]): The language of the prompt. Defaults to "en".
            max_iteration_with_llm: (Optional[int]): The max iteration with llm.
                Defaults to 5.
            concurrency_limit_with_llm: (Optional[int]): The concurrency limit with llm.
                Defaults to 3.
        """
        # 初始化 SummaryExtractor 对象
        self._llm_client = llm_client
        self._model_name = model_name
        self.llm_metadata = llm_metadata
        self._language = language
        # 根据语言设置不同的总结模板
        self._prompt_template = (
            SUMMARY_PROMPT_TEMPLATE_EN
            if language == "en"
            else SUMMARY_PROMPT_TEMPLATE_ZH
        )
        # 根据语言设置不同的精炼总结模板
        self._refine_prompt_template = (
            REFINE_SUMMARY_TEMPLATE_EN
            if language == "en"
            else REFINE_SUMMARY_TEMPLATE_ZH
        )
        # 设置最大迭代次数和并发限制
        self._concurrency_limit_with_llm = concurrency_limit_with_llm
        self._max_iteration_with_llm = max_iteration_with_llm
    async def _aextract(self, chunks: List[Chunk]) -> str:
        """Return extracted metadata from chunks of async.

        Args:
            chunks (List[Chunk]): extract metadata from chunks

        Returns:
            str: The summary of the documents.
        """
        # 提取每个文档块的内容
        texts = [doc.content for doc in chunks]
        # 导入 PromptHelper 类
        from dbgpt.util.prompt_util import PromptHelper

        # 使用 PromptHelper 实例化对象
        prompt_helper = PromptHelper()
        # 调用 PromptHelper 中的 repack 方法，重新组织文本以适应模型的最大上下文窗口
        texts = prompt_helper.repack(
            prompt_template=self._prompt_template, text_chunks=texts
        )
        # 如果只有一个文本块，直接运行 llm 任务并返回结果
        if len(texts) == 1:
            summary_outs = await self._llm_run_tasks(
                chunk_texts=texts, prompt_template=self._refine_prompt_template
            )
            return summary_outs[0]
        else:
            # 否则，通过 mapreduce 方法提取总结
            map_reduce_texts = await self._mapreduce_extract_summary(docs=texts)
            summary_outs = await self._llm_run_tasks(
                chunk_texts=[map_reduce_texts],
                prompt_template=self._refine_prompt_template,
            )
            return summary_outs[0]

    def _extract(self, chunks: List[Chunk]) -> str:
        """Return summary of the documents.

        Args:
            chunks(List[Chunk]): list of chunks

        Returns:
            summary: str
        """
        # 获取或创建事件循环
        loop = utils.get_or_create_event_loop()
        # 运行并等待 _aextract 方法完成并返回结果
        return loop.run_until_complete(self._aextract(chunks=chunks))

    async def _mapreduce_extract_summary(
        self,
        docs: List[str],
    ) -> str:
        """Return the summary of the documents.

        Extract summary by mapreduce mode.

        map -> multi async call llm to generate summary
        reduce -> merge the summaries by map process
        Args:
            docs:List[str]
        Returns:
            summary: str
        """
        # 如果只有一个文档，直接返回该文档
        if len(docs) == 1:
            return docs[0]
        else:
            # 否则，调用 _llm_run_tasks 方法生成文档摘要，并使用 PromptHelper 对摘要重新组织
            summary_outs = await self._llm_run_tasks(
                chunk_texts=docs[0 : self._max_iteration_with_llm],
                prompt_template=self._prompt_template,
            )
            from dbgpt.util.prompt_util import PromptHelper

            prompt_helper = PromptHelper()
            summary_outs = prompt_helper.repack(
                prompt_template=self._prompt_template, text_chunks=summary_outs
            )
            # 递归调用 _mapreduce_extract_summary 方法处理重新组织的摘要
            return await self._mapreduce_extract_summary(docs=summary_outs)

    async def _llm_run_tasks(
        self, chunk_texts: List[str], prompt_template: str
    ) -> List[str]:
        """Run tasks asynchronously using the language model.

        Args:
            chunk_texts (List[str]): List of text chunks to process.
            prompt_template (str): Template for generating prompts.

        Returns:
            List[str]: List of outputs generated by the language model.
        """
        # 此方法未提供完整代码，应该执行调用 llm 任务并返回结果
        pass  # Placeholder, actual implementation is missing in provided code
    ) -> List[str]:
        """Run LLM tasks.

        Args:
            chunk_texts: List[str] - 输入参数，包含多个文本块的列表
            prompt_template: str - 输入参数，用于格式化生成每个文本块的模板字符串

        Returns:
            summary_outs: List[str] - 返回结果，包含生成摘要的字符串列表
        """
        tasks = []  # 初始化任务列表
        for chunk_text in chunk_texts:  # 遍历输入的每个文本块
            from dbgpt.core import ModelMessage  # 导入模型消息类

            prompt = prompt_template.format(context=chunk_text)  # 格式化模板字符串，生成特定文本块的提示语
            messages = [ModelMessage(role=ModelMessageRoleType.HUMAN, content=prompt)]  # 创建包含提示语的模型消息列表
            request = ModelRequest(model=self._model_name, messages=messages)  # 创建模型请求对象，指定模型和消息列表
            tasks.append(self._llm_client.generate(request))  # 将生成的模型请求添加到任务列表中，调用LLM客户端生成摘要数据  # type ignore

        summary_results = await run_async_tasks(  # 并发异步运行任务列表中的所有任务
            tasks=tasks, concurrency_limit=self._concurrency_limit_with_llm
        )
        summary_outs = [model_out.text for model_out in summary_results]  # 从每个模型生成的结果中提取摘要文本
        return list(  # 返回过滤后的摘要文本列表，去除包含特定错误信息的结果
            filter(
                lambda model_out: "LLMServer Generate Error" not in model_out,
                summary_outs,
            )
        )
```
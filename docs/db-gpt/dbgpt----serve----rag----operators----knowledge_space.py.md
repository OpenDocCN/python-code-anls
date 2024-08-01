# `.\DB-GPT-src\dbgpt\serve\rag\operators\knowledge_space.py`

```py
from functools import reduce  # 导入 functools 模块中的 reduce 函数，用于高阶函数操作
from typing import List, Optional  # 导入 typing 模块中的 List 和 Optional 类型提示

from dbgpt.app.knowledge.api import knowledge_space_service  # 导入知识空间服务的 API 接口
from dbgpt.app.knowledge.request.request import KnowledgeSpaceRequest  # 导入知识空间请求相关的类
from dbgpt.app.knowledge.service import CFG, KnowledgeService  # 导入 CFG 和 KnowledgeService 类
from dbgpt.configs.model_config import EMBEDDING_MODEL_CONFIG  # 导入嵌入模型的配置
from dbgpt.core import (  # 导入 core 模块中的多个类
    BaseMessage,  # 基础消息类
    ChatPromptTemplate,  # 聊天提示模板类
    HumanPromptTemplate,  # 人类提示模板类
    ModelMessage,  # 模型消息类
)
from dbgpt.core.awel import JoinOperator, MapOperator  # 导入 awel 模块中的 JoinOperator 和 MapOperator 类
from dbgpt.core.awel.flow import (  # 导入 awel.flow 模块中的多个类
    FunctionDynamicOptions,  # 函数动态选项类
    IOField,  # 输入输出字段类
    OperatorCategory,  # 操作符类别枚举
    OperatorType,  # 操作符类型枚举
    OptionValue,  # 选项值类
    Parameter,  # 参数类
    ViewMetadata,  # 视图元数据类
)
from dbgpt.core.awel.task.base import IN, OUT  # 导入 awel.task.base 模块中的 IN 和 OUT 常量
from dbgpt.core.interface.operators.prompt_operator import BasePromptBuilderOperator  # 导入提示操作符基类
from dbgpt.rag.embedding.embedding_factory import EmbeddingFactory  # 导入嵌入工厂类
from dbgpt.rag.retriever.embedding import EmbeddingRetriever  # 导入嵌入检索器类
from dbgpt.serve.rag.connector import VectorStoreConnector  # 导入向量存储连接器类
from dbgpt.storage.vector_store.base import VectorStoreConfig  # 导入向量存储配置类
from dbgpt.util.function_utils import rearrange_args_by_type  # 导入按类型重新排列参数的实用函数
from dbgpt.util.i18n_utils import _  # 导入国际化翻译函数 "_"

def _load_space_name() -> List[OptionValue]:
    """加载空间名称列表的函数"""
    return [
        OptionValue(label=space.name, name=space.name, value=space.name)
        for space in knowledge_space_service.get_knowledge_space(
            KnowledgeSpaceRequest()
        )
    ]

class SpaceRetrieverOperator(MapOperator[IN, OUT]):
    """知识空间检索操作符类"""

    metadata = ViewMetadata(
        label=_("Knowledge Space Operator"),  # 操作符的标签
        name="space_operator",  # 操作符的名称
        category=OperatorCategory.RAG,  # 操作符的类别为 RAG
        description=_("knowledge space retriever operator."),  # 操作符的描述信息
        inputs=[IOField.build_from(_("Query"), "query", str, _("user query"))],  # 输入字段列表
        outputs=[
            IOField.build_from(
                _("related chunk content"),  # 输出字段的标签
                "related chunk content",  # 输出字段的名称
                List,  # 输出字段的类型为列表
                description=_("related chunk content"),  # 输出字段的描述信息
            )
        ],
        parameters=[
            Parameter.build_from(
                _("Space Name"),  # 参数的标签
                "space_name",  # 参数的名称
                str,  # 参数的类型为字符串
                options=FunctionDynamicOptions(func=_load_space_name),  # 参数的选项为动态函数返回的选项列表
                optional=False,  # 参数为必填项
                default=None,  # 参数的默认值为 None
                description=_("space name."),  # 参数的描述信息
            )
        ],
        documentation_url="https://github.com/openai/openai-python",  # 文档链接
    )
    def __init__(self, space_name: str, recall_score: Optional[float] = 0.3, **kwargs):
        """
        Args:
            space_name (str): The name of the space for knowledge retrieval.
            recall_score (Optional[float], optional): The recall score threshold. Defaults to 0.3.
            **kwargs: Additional keyword arguments for base class initialization.
        """
        # 设置对象的空间名称
        self._space_name = space_name
        # 设置对象的召回分数阈值
        self._recall_score = recall_score
        # 初始化知识服务对象
        self._service = KnowledgeService()
        # 获取嵌入工厂组件并创建嵌入函数
        embedding_factory = CFG.SYSTEM_APP.get_component(
            "embedding_factory", EmbeddingFactory
        )
        embedding_fn = embedding_factory.create(
            model_name=EMBEDDING_MODEL_CONFIG[CFG.EMBEDDING_MODEL]
        )
        # 配置向量存储的名称和嵌入函数
        config = VectorStoreConfig(name=self._space_name, embedding_fn=embedding_fn)
        # 初始化向量存储连接器
        self._vector_store_connector = VectorStoreConnector(
            vector_store_type=CFG.VECTOR_STORE_TYPE,
            vector_store_config=config,
        )

        # 调用父类的初始化方法
        super().__init__(**kwargs)

    async def map(self, query: IN) -> OUT:
        """Map input value to output value.

        Args:
            query (IN): The input query value.

        Returns:
            OUT: The mapped output value.
        """
        # 获取指定空间的上下文信息
        space_context = self._service.get_space_context(self._space_name)
        # 设置检索的前k个结果大小
        top_k = (
            CFG.KNOWLEDGE_SEARCH_TOP_SIZE
            if space_context is None
            else int(space_context["embedding"]["topk"])
        )
        # 设置检索时的召回分数阈值
        recall_score = (
            CFG.KNOWLEDGE_SEARCH_RECALL_SCORE
            if space_context is None
            else float(space_context["embedding"]["recall_score"])
        )
        # 初始化嵌入检索器
        embedding_retriever = EmbeddingRetriever(
            top_k=top_k,
            vector_store_connector=self._vector_store_connector,
        )
        # 根据查询类型进行检索，并返回候选项
        if isinstance(query, str):
            candidates = await embedding_retriever.aretrieve_with_scores(
                query, recall_score
            )
        elif isinstance(query, list):
            candidates = [
                await embedding_retriever.aretrieve_with_scores(q, recall_score)
                for q in query
            ]
            # 将多个查询结果合并为一个列表
            candidates = reduce(lambda x, y: x + y, candidates)
        # 返回候选项的内容列表
        return [candidate.content for candidate in candidates]
# 定义一个名为 KnowledgeSpacePromptBuilderOperator 的类，它继承自 BasePromptBuilderOperator 和 JoinOperator[List[ModelMessage]]
class KnowledgeSpacePromptBuilderOperator(
    BasePromptBuilderOperator, JoinOperator[List[ModelMessage]]
):
    """The operator to build the prompt with static prompt.

    The prompt will pass to this operator.
    """
    
    # 类的元数据，包含标签、名称、描述、操作类型、类别和参数信息
    metadata = ViewMetadata(
        label=_("Knowledge Space Prompt Builder Operator"),
        name="knowledge_space_prompt_builder_operator",
        description=_("Build messages from prompt template and chat history."),
        operator_type=OperatorType.JOIN,
        category=OperatorCategory.CONVERSION,
        parameters=[
            # 参数1：聊天提示模板，类型为 ChatPromptTemplate，描述为聊天提示模板
            Parameter.build_from(
                _("Chat Prompt Template"),
                "prompt",
                ChatPromptTemplate,
                description=_("The chat prompt template."),
            ),
            # 参数2：历史记录键，类型为 str，可选，缺省值为 "chat_history"，描述为在提示字典中历史记录的键
            Parameter.build_from(
                _("History Key"),
                "history_key",
                str,
                optional=True,
                default="chat_history",
                description=_("The key of history in prompt dict."),
            ),
            # 参数3：字符串历史，类型为 bool，可选，缺省值为 False，描述为是否将历史记录转换为字符串
            Parameter.build_from(
                _("String History"),
                "str_history",
                bool,
                optional=True,
                default=False,
                description=_("Whether to convert the history to string."),
            ),
        ],
        # 输入字段定义
        inputs=[
            # 输入1：用户输入，类型为 str，非列表，描述为用户输入
            IOField.build_from(
                _("user input"),
                "user_input",
                str,
                is_list=False,
                description=_("user input"),
            ),
            # 输入2：与空间相关的上下文，类型为 List，非列表，描述为知识空间的上下文
            IOField.build_from(
                _("space related context"),
                "related_context",
                List,
                is_list=False,
                description=_("context of knowledge space."),
            ),
            # 输入3：历史记录，类型为 BaseMessage，列表，描述为历史记录
            IOField.build_from(
                _("History"),
                "history",
                BaseMessage,
                is_list=True,
                description=_("The history."),
            ),
        ],
        # 输出字段定义
        outputs=[
            # 输出：格式化的消息，类型为 ModelMessage，列表，描述为格式化后的消息
            IOField.build_from(
                _("Formatted Messages"),
                "formatted_messages",
                ModelMessage,
                is_list=True,
                description=_("The formatted messages."),
            )
        ],
    )

    # 初始化方法，接受 prompt（聊天提示模板）、history_key（历史记录键）、check_storage（检查存储）和 str_history（字符串历史）等参数
    def __init__(
        self,
        prompt: ChatPromptTemplate,
        history_key: str = "chat_history",
        check_storage: bool = True,
        str_history: bool = False,
        **kwargs,
    ):
        """
        Create a new history dynamic prompt builder operator.

        Args:
            prompt (ChatPromptTemplate): The chat prompt template.
            history_key (str, optional): The key of history in prompt dict. Defaults to "chat_history".
            check_storage (bool, optional): Whether to check the storage. Defaults to True.
            str_history (bool, optional): Whether to convert the history to string. Defaults to False.
        """

        self._prompt = prompt
        self._history_key = history_key
        self._str_history = str_history
        # 初始化基类 BasePromptBuilderOperator 和 JoinOperator
        BasePromptBuilderOperator.__init__(self, check_storage=check_storage)
        JoinOperator.__init__(self, combine_function=self.merge_context, **kwargs)

    @rearrange_args_by_type
    async def merge_context(
        self,
        user_input: str,
        related_context: List[str],
        history: Optional[List[BaseMessage]],
    ) -> List[ModelMessage]:
        """Merge the prompt and history."""
        prompt_dict = dict()
        prompt_dict["context"] = related_context

        # 将用户输入与相关上下文结合到提示字典中
        for prompt in self._prompt.messages:
            if isinstance(prompt, HumanPromptTemplate):
                prompt_dict[prompt.input_variables[0]] = user_input

        # 处理历史消息
        if history:
            if self._str_history:
                # 如果需要将历史消息转换为字符串，则转换
                prompt_dict[self._history_key] = BaseMessage.messages_to_string(history)
            else:
                # 否则直接使用历史消息列表
                prompt_dict[self._history_key] = history

        # 格式化提示信息并返回格式化后的提示消息列表
        return await self.format_prompt(self._prompt, prompt_dict)
```
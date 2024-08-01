# `.\DB-GPT-src\dbgpt\app\scene\operators\app_operator.py`

```py
# 导入所需的模块和类
import dataclasses
from typing import Any, Dict, List, Optional

# 导入自定义的系统应用类
from dbgpt import SystemApp
# 导入组件类型枚举
from dbgpt.component import ComponentType
# 导入核心功能模块
from dbgpt.core import (
    BaseMessage,
    ChatPromptTemplate,
    LLMClient,
    ModelRequest,
    ModelRequestContext,
)
# 导入核心任务处理相关类和函数
from dbgpt.core.awel import (
    DAG,
    BaseOperator,
    BranchJoinOperator,
    InputOperator,
    MapOperator,
    SimpleCallDataInputSource,
)
# 导入核心操作符类
from dbgpt.core.operators import (
    BufferedConversationMapperOperator,
    HistoryPromptBuilderOperator,
)
# 导入语言模型操作相关类
from dbgpt.model.operators import LLMOperator, StreamingLLMOperator
# 导入缓存操作相关类
from dbgpt.storage.cache.operators import (
    CachedModelOperator,
    CachedModelStreamOperator,
    CacheManager,
    ModelCacheBranchOperator,
    ModelSaveCacheOperator,
    ModelStreamSaveCacheOperator,
)

# 定义一个数据类，用于描述聊天合成器的输入结构
@dataclasses.dataclass
class ChatComposerInput:
    """The composer input."""
    messages: List[BaseMessage]  # 包含多个基本消息对象的列表
    prompt_dict: Dict[str, Any]  # 包含键值对的字典，用于描述合成器的提示信息


class AppChatComposerOperator(MapOperator[ChatComposerInput, ModelRequest]):
    """App chat composer operator.

    TODO: Support more history merge mode.
    """

    def __init__(
        self,
        model: str,  # 指定要使用的模型的名称
        temperature: float,  # 温度参数，控制生成文本的多样性
        max_new_tokens: int,  # 生成文本的最大标记数
        prompt: ChatPromptTemplate,  # 聊天提示模板对象，用于生成聊天内容的基础模板
        message_version: str = "v2",  # 消息版本号，默认为v2
        echo: bool = False,  # 是否启用回显功能，默认为False
        streaming: bool = True,  # 是否使用流式处理，默认为True
        history_key: str = "chat_history",  # 聊天历史记录的键名，默认为"chat_history"
        history_merge_mode: str = "window",  # 历史记录合并模式，默认为"window"
        keep_start_rounds: Optional[int] = None,  # 保留开始轮次的数量，可选
        keep_end_rounds: Optional[int] = None,  # 保留结束轮次的数量，可选
        str_history: bool = False,  # 是否将历史记录转换为字符串形式，默认为False
        request_context: ModelRequestContext = None,  # 模型请求的上下文对象，默认为None
        **kwargs,
    ):
        super().__init__(**kwargs)
        # 如果没有提供请求上下文，则根据流式处理参数创建一个新的上下文对象
        if not request_context:
            request_context = ModelRequestContext(stream=streaming)
        # 初始化私有属性，用于存储各种配置参数和对象引用
        self._prompt_template = prompt  # 存储聊天提示模板对象
        self._history_key = history_key  # 存储聊天历史记录的键名
        self._history_merge_mode = history_merge_mode  # 存储历史记录合并模式
        self._keep_start_rounds = keep_start_rounds  # 存储开始轮次的保留数量
        self._keep_end_rounds = keep_end_rounds  # 存储结束轮次的保留数量
        self._str_history = str_history  # 存储历史记录转换为字符串的标志
        self._model_name = model  # 存储使用的模型名称
        self._temperature = temperature  # 存储温度参数
        self._max_new_tokens = max_new_tokens  # 存储生成文本的最大标记数
        self._message_version = message_version  # 存储消息版本号
        self._echo = echo  # 存储是否启用回显功能
        self._streaming = streaming  # 存储是否使用流式处理
        self._request_context = request_context  # 存储模型请求的上下文对象
        self._sub_compose_dag = self._build_composer_dag()  # 构建并存储子任务流程图
    async def map(self, input_value: ChatComposerInput) -> ModelRequest:
        # 获取子 DAG 的最末端节点
        end_node: BaseOperator = self._sub_compose_dag.leaf_nodes[0]
        # 使用父 DAG 的上下文来调用子 DAG
        messages = await end_node.call(
            call_data=input_value, dag_ctx=self.current_dag_context
        )
        # 获取请求上下文的 span_id
        span_id = self._request_context.span_id
        # 构建模型请求对象
        model_request = ModelRequest.build_request(
            model=self._model_name,
            messages=messages,
            context=self._request_context,
            temperature=self._temperature,
            max_new_tokens=self._max_new_tokens,
            span_id=span_id,
            echo=self._echo,
        )
        # 返回模型请求对象
        return model_request

    def _build_composer_dag(self) -> DAG:
        # 创建一个名为 "dbgpt_awel_app_chat_history_prompt_composer" 的 DAG
        with DAG("dbgpt_awel_app_chat_history_prompt_composer") as composer_dag:
            # 定义输入任务
            input_task = InputOperator(input_source=SimpleCallDataInputSource())
            # 历史转换任务
            history_transform_task = BufferedConversationMapperOperator(
                keep_start_rounds=self._keep_start_rounds,
                keep_end_rounds=self._keep_end_rounds,
            )
            # 历史提示构建任务
            history_prompt_build_task = HistoryPromptBuilderOperator(
                prompt=self._prompt_template,
                history_key=self._history_key,
                check_storage=False,
                str_history=self._str_history,
            )
            # 构建 composer DAG
            (
                input_task
                >> MapOperator(lambda x: x.messages)
                >> history_transform_task
                >> history_prompt_build_task
            )
            (
                input_task
                >> MapOperator(lambda x: x.prompt_dict)
                >> history_prompt_build_task
            )

        # 返回构建完成的 composer DAG
        return composer_dag
# 定义一个函数，用于构建和返回一个模型处理工作流的操作器（DAG 操作器）
def build_cached_chat_operator(
    llm_client: LLMClient,
    is_streaming: bool,
    system_app: SystemApp,
    cache_manager: Optional[CacheManager] = None,
):
    """Builds and returns a model processing workflow (DAG) operator.

    This function constructs a Directed Acyclic Graph (DAG) for processing data using a model.
    It includes caching and branching logic to either fetch results from a cache or process
    data using the model. It supports both streaming and non-streaming modes.

    .. code-block:: python

        input_task >> cache_check_branch_task
        cache_check_branch_task >> llm_task >> save_cache_task >> join_task
        cache_check_branch_task >> cache_task >> join_task

    equivalent to::

                          -> llm_task -> save_cache_task ->
                         /                                    \
        input_task -> cache_check_branch_task                   ---> join_task
                        \                                     /
                         -> cache_task ------------------- ->

    Args:
        llm_client (LLMClient): The LLM client for processing data using the model.
        is_streaming (bool): Whether the model is a streaming model.
        system_app (SystemApp): The system app.
        cache_manager (CacheManager, optional): The cache manager for managing cache operations. Defaults to None.

    Returns:
        BaseOperator: The final operator in the constructed DAG, typically a join node.
    """
    # 定义模型和缓存节点的任务名称
    model_task_name = "llm_model_node"
    cache_task_name = "llm_model_cache_node"
    # 如果缓存管理器未提供，则从系统应用获取模型缓存管理器组件
    if not cache_manager:
        cache_manager: CacheManager = system_app.get_component(
            ComponentType.MODEL_CACHE_MANAGER, CacheManager
        )
    with DAG("dbgpt_awel_app_model_infer_with_cached") as dag:
        # 创建一个 DAG 对象，命名为 dbgpt_awel_app_model_infer_with_cached

        # 创建一个输入任务
        input_task = InputOperator(SimpleCallDataInputSource())

        # 根据是否为流式处理创建模型操作任务和缓存操作任务的分支
        if is_streaming:
            # 如果是流式处理，则使用 StreamingLLMOperator 和 CachedModelStreamOperator
            llm_task = StreamingLLMOperator(llm_client, task_name=model_task_name)
            cache_task = CachedModelStreamOperator(
                cache_manager, task_name=cache_task_name
            )
            save_cache_task = ModelStreamSaveCacheOperator(cache_manager)
        else:
            # 如果不是流式处理，则使用 LLMOperator 和 CachedModelOperator
            llm_task = LLMOperator(llm_client, task_name=model_task_name)
            cache_task = CachedModelOperator(cache_manager, task_name=cache_task_name)
            save_cache_task = ModelSaveCacheOperator(cache_manager)

        # 创建一个分支节点，决定是从缓存获取还是使用模型处理
        cache_check_branch_task = ModelCacheBranchOperator(
            cache_manager,
            model_task_name=model_task_name,
            cache_task_name=cache_task_name,
        )

        # 创建一个合并节点，用于合并模型和缓存节点的输出，保留第一个非空输出
        join_task = BranchJoinOperator()

        # 定义工作流结构，使用 >> 运算符连接任务
        input_task >> cache_check_branch_task
        cache_check_branch_task >> llm_task >> save_cache_task >> join_task
        cache_check_branch_task >> cache_task >> join_task

        # 返回合并节点作为最终输出
        return join_task
```
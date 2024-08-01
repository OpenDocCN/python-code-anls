# `.\DB-GPT-src\dbgpt\core\operators\flow\composer_operator.py`

```py
# 导入必要的模块和类，包括类型提示和特定的操作符
from typing import Any, Dict, Optional, cast  # 导入类型提示相关的模块

from dbgpt.core.awel import (  # 导入 AWEL 相关的核心模块
    DAG,  # DAG 数据结构
    BaseOperator,  # 基础操作符类
    InputOperator,  # 输入操作符类
    JoinOperator,  # 连接操作符类
    MapOperator,  # 映射操作符类
    SimpleCallDataInputSource,  # 简单调用数据输入源
)
from dbgpt.core.awel.flow import (  # 导入 AWEL 流程相关的模块
    IOField,  # IO 字段
    OperatorCategory,  # 操作符类别
    Parameter,  # 参数类
    ViewMetadata,  # 视图元数据
)
from dbgpt.core.awel.trigger.http_trigger import CommonLLMHttpRequestBody  # 导入 HTTP 触发器相关类
from dbgpt.core.interface.llm import ModelRequest  # 导入模型请求类
from dbgpt.core.interface.message import (  # 导入消息接口相关模块
    MessageStorageItem,  # 消息存储项
    StorageConversation,  # 存储会话
)
from dbgpt.core.interface.operators.llm_operator import (  # 导入 LLM 操作符相关模块
    MergedRequestBuilderOperator,  # 合并请求构建器操作符
    RequestBuilderOperator,  # 请求构建器操作符
)
from dbgpt.core.interface.operators.message_operator import (  # 导入消息操作符相关模块
    BufferedConversationMapperOperator,  # 缓存会话映射操作符
    PreChatHistoryLoadOperator,  # 预加载聊天历史操作符
)
from dbgpt.core.interface.operators.prompt_operator import HistoryPromptBuilderOperator  # 导入历史提示构建器操作符
from dbgpt.core.interface.prompt import ChatPromptTemplate  # 导入聊天提示模板类
from dbgpt.core.interface.storage import StorageInterface  # 导入存储接口类
from dbgpt.util.i18n_utils import _  # 导入国际化工具函数

# 定义一个名为 ConversationComposerOperator 的类，继承自 MapOperator，用于组合对话的操作
class ConversationComposerOperator(MapOperator[CommonLLMHttpRequestBody, ModelRequest]):
    """A Composer operator for conversation.

    Build for AWEL Flow.
    """
    # 创建一个视图元数据对象 ViewMetadata
    metadata = ViewMetadata(
        # 设置标签为 "Conversation Composer Operator"
        label=_("Conversation Composer Operator"),
        # 设置名称为 "conversation_composer_operator"
        name="conversation_composer_operator",
        # 设置类别为 OperatorCategory.CONVERSION
        category=OperatorCategory.CONVERSION,
        # 设置描述信息，描述此操作符的作用和输出
        description=_(
            "A composer operator for conversation.\nIncluding chat history "
            "handling, prompt composing, etc. Output is ModelRequest."
        ),
        # 设置参数列表，包含多个参数定义
        parameters=[
            # 构建参数对象，设置参数名为 "Prompt Template"，类型为 ChatPromptTemplate
            Parameter.build_from(
                _("Prompt Template"),
                "prompt_template",
                ChatPromptTemplate,
                # 参数描述：用于对话的提示模板
                description=_("The prompt template for the conversation."),
            ),
            # 构建参数对象，设置参数名为 "Human Message Key"，类型为 str
            Parameter.build_from(
                _("Human Message Key"),
                "human_message_key",
                str,
                optional=True,  # 可选参数
                default="user_input",  # 默认值为 "user_input"
                # 参数描述：在提示格式字典中用于人类消息的键
                description=_("The key for human message in the prompt format dict."),
            ),
            # 构建参数对象，设置参数名为 "History Key"，类型为 str
            Parameter.build_from(
                _("History Key"),
                "history_key",
                str,
                optional=True,  # 可选参数
                default="chat_history",  # 默认值为 "chat_history"
                # 参数描述：包含对话历史消息的键，传递给提示模板
                description=_(
                    "The chat history key, with chat history message pass to "
                    "prompt template."
                ),
            ),
            # 构建参数对象，设置参数名为 "Keep Start Rounds"，类型为 int
            Parameter.build_from(
                _("Keep Start Rounds"),
                "keep_start_rounds",
                int,
                optional=True,  # 可选参数
                default=None,  # 默认值为 None
                # 参数描述：保留在对话历史中的起始轮数
                description=_("The start rounds to keep in the chat history."),
            ),
            # 构建参数对象，设置参数名为 "Keep End Rounds"，类型为 int
            Parameter.build_from(
                _("Keep End Rounds"),
                "keep_end_rounds",
                int,
                optional=True,  # 可选参数
                default=10,  # 默认值为 10
                # 参数描述：保留在对话历史中的结束轮数
                description=_("The end rounds to keep in the chat history."),
            ),
            # 构建参数对象，设置参数名为 "Conversation Storage"，类型为 StorageInterface
            Parameter.build_from(
                _("Conversation Storage"),
                "storage",
                StorageInterface,
                optional=True,  # 可选参数
                default=None,  # 默认值为 None
                # 参数描述：对话存储（不包含消息细节）
                description=_("The conversation storage(Not include message detail)."),
            ),
            # 构建参数对象，设置参数名为 "Message Storage"，类型为 StorageInterface
            Parameter.build_from(
                _("Message Storage"),
                "message_storage",
                StorageInterface,
                optional=True,  # 可选参数
                default=None,  # 默认值为 None
                # 参数描述：消息存储
                description=_("The message storage."),
            ),
        ],
        # 设置输入字段列表，包含一个输入字段定义
        inputs=[
            # 构建输入字段对象，设置字段名为 "Common LLM Http Request Body"，类型为 CommonLLMHttpRequestBody
            IOField.build_from(
                _("Common LLM Http Request Body"),
                "common_llm_http_request_body",
                CommonLLMHttpRequestBody,
                # 输入字段描述：通用的LLM HTTP请求体
                description=_("The common LLM http request body."),
            )
        ],
        # 设置输出字段列表，包含一个输出字段定义
        outputs=[
            # 构建输出字段对象，设置字段名为 "Model Request"，类型为 ModelRequest
            IOField.build_from(
                _("Model Request"),
                "model_request",
                ModelRequest,
                # 输出字段描述：包含对话历史提示的模型请求
                description=_("The model request with chat history prompt."),
            )
        ],
    )
    # ConversationComposerOperator 类的构造函数，初始化实例
    def __init__(
        self,
        prompt_template: ChatPromptTemplate,
        human_message_key: str = "user_input",
        history_key: str = "chat_history",
        keep_start_rounds: Optional[int] = None,
        keep_end_rounds: Optional[int] = None,
        storage: Optional[StorageInterface[StorageConversation, Any]] = None,
        message_storage: Optional[StorageInterface[MessageStorageItem, Any]] = None,
        **kwargs
    ):
        """Create a new instance of ConversationComposerOperator."""
        # 调用父类的构造函数，初始化继承的属性
        super().__init__(**kwargs)
        # 设置会话组合器的模板
        self._prompt_template = prompt_template
        # 设置人类消息的键名，默认为'user_input'
        self._human_message_key = human_message_key
        # 设置历史记录的键名，默认为'chat_history'
        self._history_key = history_key
        # 设置保留的对话起始轮数，可选参数
        self._keep_start_rounds = keep_start_rounds
        # 设置保留的对话结束轮数，可选参数
        self._keep_end_rounds = keep_end_rounds
        # 设置存储接口，用于存储会话相关数据，可选参数
        self._storage = storage
        # 设置消息存储接口，用于存储消息相关数据，可选参数
        self._message_storage = message_storage
        # 构建会话组合器的子组合图
        self._sub_compose_dag = self._build_composer_dag()

    async def map(self, input_value: CommonLLMHttpRequestBody) -> ModelRequest:
        """Receive the common LLM http request body, and build the model request."""
        # 获取子组合图的最末端节点
        end_node: BaseOperator = cast(BaseOperator, self._sub_compose_dag.leaf_nodes[0])
        # 在父组合图的上下文中调用子组合图的末端节点，传入调用数据和当前组合图的上下文
        return await end_node.call(
            call_data=input_value, dag_ctx=self.current_dag_context
        )
    def _build_composer_dag(self):
        # 使用 DAG 函数创建名为 "dbgpt_awel_chat_history_prompt_composer" 的 DAG 对象
        with DAG("dbgpt_awel_chat_history_prompt_composer") as composer_dag:
            # 创建输入任务，使用 SimpleCallDataInputSource 作为输入源
            input_task = InputOperator(input_source=SimpleCallDataInputSource())
            # 加载并存储聊天历史记录，默认使用 InMemoryStorage
            chat_history_load_task = PreChatHistoryLoadOperator(
                storage=self._storage, message_storage=self._message_storage
            )
            # 历史转换任务，保留最近的 5 轮消息
            history_transform_task = BufferedConversationMapperOperator(
                keep_start_rounds=self._keep_start_rounds,
                keep_end_rounds=self._keep_end_rounds,
            )
            # 历史提示构建任务，使用指定的提示模板和历史记录键
            history_prompt_build_task = HistoryPromptBuilderOperator(
                prompt=self._prompt_template, history_key=self._history_key
            )
            # 构建提示格式字典的任务，使用人类消息键
            prompt_build_task = PromptFormatDictBuilderOperator(
                human_message_key=self._human_message_key
            )
            # 模型请求构建任务，使用 JoinOperator 进行合并
            model_request_build_task: JoinOperator[
                ModelRequest
            ] = MergedRequestBuilderOperator()

            # 构建 composer dag 的执行流程
            (
                input_task
                >> MapOperator(lambda x: x.context)  # 映射操作，提取上下文信息
                >> chat_history_load_task  # 加载聊天历史任务
                >> history_transform_task  # 历史转换任务
                >> history_prompt_build_task  # 历史提示构建任务
            )
            input_task >> prompt_build_task >> history_prompt_build_task  # 输入任务到提示构建任务到历史提示构建任务的流程

            input_task >> RequestBuilderOperator() >> model_request_build_task  # 输入任务到请求构建任务到模型请求构建任务
            history_prompt_build_task >> model_request_build_task  # 历史提示构建任务到模型请求构建任务的数据流关系
        return composer_dag

    async def after_dag_end(self, event_loop_task_id: int):
        """Execute after dag end."""
        # 在 DAG 结束后执行的异步方法，调用子 DAG 的 after_dag_end 方法
        await self._sub_compose_dag._after_dag_end(event_loop_task_id)
class PromptFormatDictBuilderOperator(
    MapOperator[CommonLLMHttpRequestBody, Dict[str, Any]]
):
    """Prompt format dict builder operator for AWEL flow.

    Receive the common LLM http request body, and build the prompt format dict.
    """

    metadata = ViewMetadata(
        label=_("Prompt Format Dict Builder Operator"),
        name="prompt_format_dict_builder_operator",
        category=OperatorCategory.CONVERSION,
        description=_(
            "A operator to build prompt format dict from common LLM http "
            "request body."
        ),
        parameters=[
            Parameter.build_from(
                _("Human Message Key"),
                "human_message_key",
                str,
                optional=True,
                default="user_input",
                description=_("The key for human message in the prompt format dict."),
            )
        ],
        inputs=[
            IOField.build_from(
                _("Common LLM Http Request Body"),
                "common_llm_http_request_body",
                CommonLLMHttpRequestBody,
                description=_("The common LLM http request body."),
            )
        ],
        outputs=[
            IOField.build_from(
                _("Prompt Format Dict"),
                "prompt_format_dict",
                dict,
                description=_("The prompt format dict."),
            )
        ],
    )

    def __init__(self, human_message_key: str = "user_input", **kwargs):
        """Create a new instance of PromptFormatDictBuilderOperator."""
        # 初始化操作符实例，设定人类消息键值，默认为'user_input'
        self._human_message_key = human_message_key
        super().__init__(**kwargs)

    async def map(self, input_value: CommonLLMHttpRequestBody) -> Dict[str, Any]:
        """Build prompt format dict from common LLM http request body."""
        # 提取额外数据，如果存在则使用，否则为空字典
        extra_data = input_value.extra if input_value.extra else {}
        # 返回构建的提示格式字典，包括人类消息键和消息内容，以及额外数据
        return {
            self._human_message_key: input_value.messages,
            **extra_data,
        }
```
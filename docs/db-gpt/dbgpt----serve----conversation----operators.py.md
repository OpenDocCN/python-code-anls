# `.\DB-GPT-src\dbgpt\serve\conversation\operators.py`

```py
import logging  # 导入日志模块
from typing import Any, Optional  # 导入类型提示模块

from dbgpt.core import (  # 导入核心功能模块
    BaseMessage,
    InMemoryStorage,
    MessageStorageItem,
    ModelRequest,
    StorageConversation,
    StorageInterface,
)
from dbgpt.core.awel.flow import IOField, OperatorCategory, ViewMetadata  # 导入流程相关模块
from dbgpt.core.operators import PreChatHistoryLoadOperator  # 导入预加载聊天历史操作模块
from dbgpt.util.i18n_utils import _  # 导入国际化工具

from .serve import Serve  # 导入本地的 Serve 模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class ServePreChatHistoryLoadOperator(PreChatHistoryLoadOperator):
    """DB-GPT 服务组件的预加载聊天历史操作符

    Args:
        storage (Optional[StorageInterface[StorageConversation, Any]], optional):
            会话存储，用于存储会话项。默认为 None。
        message_storage (Optional[StorageInterface[MessageStorageItem, Any]], optional):
            消息存储，用于存储一个会话的消息。默认为 None。

    如果 storage 或 message_storage 不为 None，则优先使用它们。
    否则，尝试从系统应用中获取当前的 Serve 组件，
    并使用 Serve 组件的 storage 或 message_storage。
    如果无法获取 storage，则使用 InMemoryStorage 作为 storage 或 message_storage。
    """

    def __init__(
        self,
        storage: Optional[StorageInterface[StorageConversation, Any]] = None,
        message_storage: Optional[StorageInterface[MessageStorageItem, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            storage, message_storage, use_in_memory_storage_if_not_set=False, **kwargs
        )

    @property
    def storage(self):
        if self._storage:
            return self._storage
        storage = Serve.call_on_current_serve(
            self.system_app, lambda serve: serve.conv_storage
        )
        if not storage:
            logger.warning(
                "无法从当前 Serve 组件获取会话存储，将使用 InMemoryStorage 作为会话存储。"
            )
            self._storage = InMemoryStorage()
            return self._storage
        return storage

    @property
    def message_storage(self):
        if self._message_storage:
            return self._message_storage
        storage = Serve.call_on_current_serve(
            self.system_app,
            lambda serve: serve.message_storage,
        )
        if not storage:
            logger.warning(
                "无法从当前 Serve 组件获取消息存储，将使用 InMemoryStorage 作为消息存储。"
            )
            self._message_storage = InMemoryStorage()
            return self._message_storage
        return storage


class DefaultServePreChatHistoryLoadOperator(ServePreChatHistoryLoadOperator):
    """DB-GPT 服务组件的默认预加载聊天历史操作符

    使用 Serve 组件的 storage 和 message storage。
    """
    # 创建视图元数据对象，用于描述默认的聊天历史加载操作符
    metadata = ViewMetadata(
        # 设置操作符的显示标签，国际化处理
        label=_("Default Chat History Load Operator"),
        # 设置操作符的名称，用于标识
        name="default_serve_pre_chat_history_load_operator",
        # 设置操作符所属的类别，这里是转换类别
        category=OperatorCategory.CONVERSION,
        # 设置操作符的描述信息，包括从 serve 组件存储中加载聊天历史
        description=_(
            "Load chat history from the storage of the serve component."
            "It is the default storage of DB-GPT"
        ),
        # 设置操作符的参数列表为空
        parameters=[],
        # 设置操作符的输入字段，这里是模型请求的输入
        inputs=[
            IOField.build_from(
                # 输入字段的显示标签，国际化处理
                _("Model Request"),
                # 输入字段的名称
                "model_request",
                # 输入字段的类型为 ModelRequest 类型
                type=ModelRequest,
                # 输入字段的描述信息，表示模型请求
                description=_("The model request."),
            )
        ],
        # 设置操作符的输出字段，这里是输出存储的消息列表
        outputs=[
            IOField.build_from(
                # 输出字段的显示标签，国际化处理
                label=_("Stored Messages"),
                # 输出字段的名称
                name="output_value",
                # 输出字段的类型为 BaseMessage 类型
                type=BaseMessage,
                # 输出字段的描述信息，表示存储的消息
                description=_("The messages stored in the storage."),
                # 输出字段是一个列表
                is_list=True,
            )
        ],
    )
    
    # 初始化方法，继承父类的初始化方法
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
```
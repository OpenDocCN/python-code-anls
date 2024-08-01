# `.\DB-GPT-src\dbgpt\core\interface\operators\composer_operator.py`

```py
    """The chat history prompt composer operator.

    We can wrap some atomic operators to a complex operator.
    """
    import dataclasses  # 导入用于定义数据类的模块
    from typing import Any, Dict, List, Optional, cast  # 导入类型提示相关的模块

    from dbgpt.core import (
        ChatPromptTemplate,  # 导入聊天提示模板类
        MessageStorageItem,  # 导入消息存储项类
        ModelMessage,  # 导入模型消息类
        ModelRequest,  # 导入模型请求类
        StorageConversation,  # 导入存储对话类
        StorageInterface,  # 导入存储接口类
    )
    from dbgpt.core.awel import (
        DAG,  # 导入有向无环图类
        BaseOperator,  # 导入基础操作器类
        InputOperator,  # 导入输入操作器类
        JoinOperator,  # 导入连接操作器类
        MapOperator,  # 导入映射操作器类
        SimpleCallDataInputSource,  # 导入简单调用数据输入源类
    )
    from dbgpt.core.interface.operators.prompt_operator import HistoryPromptBuilderOperator  # 导入历史提示构建器操作器类

    from .message_operator import (
        BufferedConversationMapperOperator,  # 导入缓冲对话映射器操作器类
        ChatHistoryLoadType,  # 导入聊天历史加载类型类
        PreChatHistoryLoadOperator,  # 导入预加载聊天历史操作器类
    )


    @dataclasses.dataclass
    class ChatComposerInput:
        """The composer input."""
        prompt_dict: Dict[str, Any]  # 输入参数：提示字典，键为字符串，值为任意类型
        model_dict: Dict[str, Any]  # 输入参数：模型字典，键为字符串，值为任意类型
        context: ChatHistoryLoadType  # 输入参数：聊天历史加载类型


    class ChatHistoryPromptComposerOperator(MapOperator[ChatComposerInput, ModelRequest]):
        """The chat history prompt composer operator.

        For simple use, you can use this operator to compose the chat history prompt.
        """

        def __init__(
            self,
            prompt_template: ChatPromptTemplate,  # 聊天提示模板对象，用于生成提示
            history_key: str = "chat_history",  # 历史键名，默认为"chat_history"
            keep_start_rounds: Optional[int] = None,  # 保留起始轮数，可选整数类型
            keep_end_rounds: Optional[int] = None,  # 保留结束轮数，可选整数类型
            storage: Optional[StorageInterface[StorageConversation, Any]] = None,  # 存储接口对象，用于存储对话
            message_storage: Optional[StorageInterface[MessageStorageItem, Any]] = None,  # 消息存储接口对象，用于存储消息
            **kwargs,
        ):
            """Create a new chat history prompt composer operator."""
            super().__init__(**kwargs)  # 调用父类的构造函数
            self._prompt_template = prompt_template  # 初始化聊天提示模板
            self._history_key = history_key  # 初始化历史键名
            self._keep_start_rounds = keep_start_rounds  # 初始化保留起始轮数
            self._keep_end_rounds = keep_end_rounds  # 初始化保留结束轮数
            self._storage = storage  # 初始化存储接口对象
            self._message_storage = message_storage  # 初始化消息存储接口对象
            self._sub_compose_dag = self._build_composer_dag()  # 构建并初始化子组合 DAG

        async def map(self, input_value: ChatComposerInput) -> ModelRequest:
            """Compose the chat history prompt."""
            end_node: BaseOperator = cast(BaseOperator, self._sub_compose_dag.leaf_nodes[0])  # 获取子 DAG 的末尾节点
            # 子 DAG，使用相同的 DAG 上下文在父 DAG 中
            return await end_node.call(
                call_data=input_value, dag_ctx=self.current_dag_context  # 调用末尾节点的 call 方法，传入输入值和当前 DAG 上下文
            )
    def _build_composer_dag(self) -> DAG:
        # 创建名为 "dbgpt_awel_chat_history_prompt_composer" 的 DAG 对象
        with DAG("dbgpt_awel_chat_history_prompt_composer") as composer_dag:
            # 创建输入任务，使用 SimpleCallDataInputSource 作为输入源
            input_task = InputOperator(input_source=SimpleCallDataInputSource())
            # 加载和存储聊天历史，默认使用 InMemoryStorage
            chat_history_load_task = PreChatHistoryLoadOperator(
                storage=self._storage, message_storage=self._message_storage
            )
            # 历史转换任务，保留最近的 5 轮消息
            history_transform_task = BufferedConversationMapperOperator(
                keep_start_rounds=self._keep_start_rounds,
                keep_end_rounds=self._keep_end_rounds,
            )
            # 构建历史提示任务，使用给定的提示模板和历史键
            history_prompt_build_task = HistoryPromptBuilderOperator(
                prompt=self._prompt_template, history_key=self._history_key
            )
            # 构建模型请求任务，将消息和模型字典结合为 ModelRequest 对象
            model_request_build_task: JoinOperator[ModelRequest] = JoinOperator(
                combine_function=self._build_model_request
            )

            # 构建 DAG 流程
            (
                input_task
                >> MapOperator(lambda x: x.context)
                >> chat_history_load_task
                >> history_transform_task
                >> history_prompt_build_task
            )
            (
                input_task
                >> MapOperator(lambda x: x.prompt_dict)
                >> history_prompt_build_task
            )

            # 将历史提示构建任务输出连接到模型请求构建任务
            history_prompt_build_task >> model_request_build_task
            (
                input_task
                >> MapOperator(lambda x: x.model_dict)
                >> model_request_build_task
            )

        # 返回构建好的 composer_dag DAG 对象
        return composer_dag

    def _build_model_request(
        self, messages: List[ModelMessage], model_dict: Dict[str, Any]
    ) -> ModelRequest:
        # 使用给定的消息和模型字典构建 ModelRequest 对象
        return ModelRequest.build_request(messages=messages, **model_dict)

    async def after_dag_end(self, event_loop_task_id: int):
        """在 DAG 结束后执行的异步方法。"""
        # 调用子 DAG 的 after_dag_end() 方法，传入事件循环任务 ID
        await self._sub_compose_dag._after_dag_end(event_loop_task_id)
```
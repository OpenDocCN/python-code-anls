# `.\DB-GPT-src\dbgpt\app\scene\base_chat.py`

```py
# 异步编程模块导入
import asyncio
# 处理日期和时间的模块导入
import datetime
# 日志记录模块导入
import logging
# 异常追踪模块导入
import traceback
# 抽象基类模块导入
from abc import ABC, abstractmethod
# 类型提示模块导入
from typing import Any, AsyncIterator, Dict

# 导入私有配置模块
from dbgpt._private.config import Config
# 导入额外禁用类型检查的模块
from dbgpt._private.pydantic import EXTRA_FORBID
# 导入应用场景适配器基类和聊天场景模块
from dbgpt.app.scene.base import AppScenePromptTemplateAdapter, ChatScene
# 导入应用操作员模块，包括聊天合成器操作员和构建缓存聊天操作员方法
from dbgpt.app.scene.operators.app_operator import (
    AppChatComposerOperator,
    ChatComposerInput,
    build_cached_chat_operator,
)
# 导入组件类型模块
from dbgpt.component import ComponentType
# 导入核心模块，包括LLM客户端、模型输出、模型请求、模型请求上下文
from dbgpt.core import LLMClient, ModelOutput, ModelRequest, ModelRequestContext
# 导入消息存储会话接口模块
from dbgpt.core.interface.message import StorageConversation
# 导入默认LLM客户端模块
from dbgpt.model import DefaultLLMClient
# 导入集群工作管理器工厂模块
from dbgpt.model.cluster import WorkerManagerFactory
# 导入会话服务模块作为ConversationServe
from dbgpt.serve.conversation.serve import Serve as ConversationServe
# 导入获取或创建事件循环的工具方法
from dbgpt.util import get_or_create_event_loop
# 导入执行器工厂和将阻塞函数转换为异步函数的工具方法
from dbgpt.util.executor_utils import ExecutorFactory, blocking_func_to_async
# 导入异步重试装饰器模块
from dbgpt.util.retry import async_retry
# 导入追踪器模块，包括根追踪器和追踪装饰器
from dbgpt.util.tracer import root_tracer, trace

# 导入自定义异常模块BaseAppException
from .exceptions import BaseAppException

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)
# 加载配置对象
CFG = Config()

# 定义一个私有函数_build_conversation，用于构建存储会话对象
def _build_conversation(
    chat_mode: ChatScene,    # 聊天场景模式参数
    chat_param: Dict[str, Any],    # 聊天参数字典
    model_name: str,    # 模型名称
    conv_serve: ConversationServe,    # ConversationServe对象
) -> StorageConversation:    # 返回类型为StorageConversation
    param_type = ""    # 初始化参数类型为空字符串
    param_value = ""    # 初始化参数值为空字符串
    if chat_param["select_param"]:    # 如果选择参数不为空
        if len(chat_mode.param_types()) > 0:    # 如果聊天场景模式的参数类型列表长度大于0
            param_type = chat_mode.param_types()[0]    # 将第一个参数类型赋给param_type
        param_value = chat_param["select_param"]    # 将选择参数赋给param_value
    return StorageConversation(    # 返回构建的StorageConversation对象
        chat_param["chat_session_id"],    # 聊天会话ID
        chat_mode=chat_mode.value(),    # 聊天场景模式值
        user_name=chat_param.get("user_name"),    # 用户名
        sys_code=chat_param.get("sys_code"),    # 系统代码
        model_name=model_name,    # 模型名称
        param_type=param_type,    # 参数类型
        param_value=param_value,    # 参数值
        conv_storage=conv_serve.conv_storage,    # 会话存储
        message_storage=conv_serve.message_storage,    # 消息存储
    )


# 定义一个抽象基类BaseChat
class BaseChat(ABC):
    """DB-GPT Chat Service Base Module
    Include:
    stream_call():scene + prompt -> stream response
    nostream_call():scene + prompt -> nostream response
    """

    chat_scene: str = None    # 聊天场景默认为空
    llm_model: Any = None    # LLM模型默认为空
    # 默认保留最后两轮对话记录作为上下文
    keep_start_rounds: int = 0    # 开始轮数默认为0
    keep_end_rounds: int = 0    # 结束轮数默认为0

    # 某些模型不支持系统角色，此配置用于控制是否将系统消息转换为人类消息
    auto_convert_message: bool = True    # 默认为True，自动转换消息为人类消息

    @trace("BaseChat.__init__")    # 追踪BaseChat类的初始化过程
    def __init__(self, chat_param: Dict):
        """Chat Module Initialization
        
        Args:
           - chat_param: Dict
            - chat_session_id: (str) chat session_id
            - current_user_input: (str) current user input
            - model_name:(str) llm model name
            - select_param:(str) select param
        """
        # 初始化聊天模块，设置基本参数
        self.chat_session_id = chat_param["chat_session_id"]
        self.chat_mode = chat_param["chat_mode"]
        self.current_user_input: str = chat_param["current_user_input"]
        self.llm_model = (
            chat_param["model_name"] if chat_param["model_name"] else CFG.LLM_MODEL
        )
        self.llm_echo = False
        
        # 初始化工作管理器，使用系统配置获取组件
        self.worker_manager = CFG.SYSTEM_APP.get_component(
            ComponentType.WORKER_MANAGER_FACTORY, WorkerManagerFactory
        ).create()
        
        # 设置模型缓存是否启用的标志
        self.model_cache_enable = chat_param.get("model_cache_enable", False)

        ### 加载提示模板
        self.prompt_template: AppScenePromptTemplateAdapter = (
            CFG.prompt_template_registry.get_prompt_template(
                self.chat_mode.value(),
                language=CFG.LANGUAGE,
                model_name=self.llm_model,
                proxyllm_backend=CFG.PROXYLLM_BACKEND,
            )
        )
        
        # 获取对话服务的实例
        self._conv_serve = ConversationServe.get_instance(CFG.SYSTEM_APP)
        
        # 创建当前对话消息的实例
        self.current_message: StorageConversation = _build_conversation(
            self.chat_mode, chat_param, self.llm_model, self._conv_serve
        )
        
        # 获取历史消息
        self.history_messages = self.current_message.get_history_message()
        
        # 设置当前使用的令牌数为0
        self.current_tokens_used: int = 0
        
        # 创建执行器以提交阻塞函数
        self._executor = CFG.SYSTEM_APP.get_component(
            ComponentType.EXECUTOR_DEFAULT, ExecutorFactory
        ).create()

        # 消息版本号，默认为"v2"
        self._message_version = chat_param.get("message_version", "v2")
        self._chat_param = chat_param
    def chat_type(self) -> str:
        raise NotImplementedError("Not supported for this chat type.")
        # 抛出未实现错误，因为该聊天类型不支持此方法

    @abstractmethod
    async def generate_input_values(self) -> Dict:
        """Generate input to LLM

        Please note that you must not perform any blocking operations in this function

        Returns:
            a dictionary to be formatted by prompt template
        """
        # 抽象方法：生成输入数据给LLM模型使用
        # 注意：在此函数中不要执行任何阻塞操作
        # 返回一个字典，将按照提示模板格式化

    @property
    def llm_client(self) -> LLMClient:
        """Return the LLM client."""
        # 属性方法：返回LLM客户端对象

        # 获取工作管理器，创建工作管理器工厂实例
        worker_manager = CFG.SYSTEM_APP.get_component(
            ComponentType.WORKER_MANAGER_FACTORY, WorkerManagerFactory
        ).create()
        # 返回带有工作管理器和自动转换消息选项的默认LLM客户端实例
        return DefaultLLMClient(
            worker_manager, auto_convert_message=self.auto_convert_message
        )

    async def call_llm_operator(self, request: ModelRequest) -> ModelOutput:
        """Call LLM operator asynchronously."""
        # 异步方法：调用LLM操作者

        # 构建缓存聊天操作者任务，传入LLM客户端、False表示非流式操作以及系统应用配置
        llm_task = build_cached_chat_operator(self.llm_client, False, CFG.SYSTEM_APP)
        # 调用LLM任务，并返回调用结果
        return await llm_task.call(call_data=request)

    async def call_streaming_operator(
        self, request: ModelRequest
    ) -> AsyncIterator[ModelOutput]:
        """Call streaming LLM operator asynchronously."""
        # 异步方法：调用流式LLM操作者

        # 构建缓存聊天操作者任务，传入LLM客户端、True表示流式操作以及系统应用配置
        llm_task = build_cached_chat_operator(self.llm_client, True, CFG.SYSTEM_APP)
        # 异步迭代LLM任务的流式调用结果，并逐个产生输出
        async for out in await llm_task.call_stream(call_data=request):
            yield out

    def do_action(self, prompt_response):
        """Perform action based on prompt response."""
        # 方法：根据提示响应执行动作
        return prompt_response

    def message_adjust(self):
        """Adjust message if necessary."""
        # 方法：如果需要的话，调整消息内容
        pass

    def has_history_messages(self) -> bool:
        """Whether there is a history messages

        Returns:
            bool: True if there is a history message, False otherwise
        """
        # 方法：检查是否有历史消息

        # 返回历史消息列表的长度是否大于0的布尔值
        return len(self.history_messages) > 0

    def get_llm_speak(self, prompt_define_response):
        """Extract 'speak' information from prompt define response."""
        # 方法：从提示定义响应中提取“speak”信息

        if hasattr(prompt_define_response, "thoughts"):
            if isinstance(prompt_define_response.thoughts, dict):
                if "speak" in prompt_define_response.thoughts:
                    speak_to_user = prompt_define_response.thoughts.get("speak")
                else:
                    speak_to_user = str(prompt_define_response.thoughts)
            else:
                if hasattr(prompt_define_response.thoughts, "speak"):
                    speak_to_user = prompt_define_response.thoughts.get("speak")
                elif hasattr(prompt_define_response.thoughts, "reasoning"):
                    speak_to_user = prompt_define_response.thoughts.get("reasoning")
                else:
                    speak_to_user = prompt_define_response.thoughts
        else:
            speak_to_user = prompt_define_response

        return speak_to_user
    # 异步方法：构建模型请求对象，并返回该对象
    async def _build_model_request(self) -> ModelRequest:
        # 调用异步方法生成输入值
        input_values = await self.generate_input_values()
        
        # 加载历史消息
        self.history_messages = self.current_message.get_history_message()
        
        # 开始新的对话轮次
        self.current_message.start_new_round()
        
        # 将当前用户输入添加到消息中
        self.current_message.add_user_message(self.current_user_input)
        
        # 设置当前消息的开始日期为当前时间的格式化字符串
        self.current_message.start_date = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        
        # 将当前消息的 tokens 数量置零
        self.current_message.tokens = 0
        
        # 根据是否需要历史消息，确定保留的开始轮次数和结束轮次数
        keep_start_rounds = (
            self.keep_start_rounds
            if self.prompt_template.need_historical_messages
            else 0
        )
        keep_end_rounds = (
            self.keep_end_rounds if self.prompt_template.need_historical_messages else 0
        )
        
        # 创建模型请求的上下文对象
        req_ctx = ModelRequestContext(
            stream=self.prompt_template.stream_out,
            user_name=self._chat_param.get("user_name"),
            sys_code=self._chat_param.get("sys_code"),
            chat_mode=self.chat_mode.value(),
            span_id=root_tracer.get_current_span_id(),
        )
        
        # 创建应用聊天组件操作器的节点
        node = AppChatComposerOperator(
            model=self.llm_model,
            temperature=float(self.prompt_template.temperature),
            max_new_tokens=int(self.prompt_template.max_new_tokens),
            prompt=self.prompt_template.prompt,
            message_version=self._message_version,
            echo=self.llm_echo,
            streaming=self.prompt_template.stream_out,
            keep_start_rounds=keep_start_rounds,
            keep_end_rounds=keep_end_rounds,
            str_history=self.prompt_template.str_history,
            request_context=req_ctx,
        )
        
        # 创建聊天组件输入对象
        node_input = ChatComposerInput(
            messages=self.history_messages, prompt_dict=input_values
        )
        
        # 调用节点对象的方法进行模型请求，并获取模型请求对象
        model_request: ModelRequest = await node.call(call_data=node_input)
        
        # 设置模型请求的上下文缓存启用状态
        model_request.context.cache_enable = self.model_cache_enable
        
        # 返回模型请求对象
        return model_request

    # 流插件调用方法，返回传入的文本
    def stream_plugin_call(self, text):
        return text

    # 流调用增强函数，返回传入的文本
    def stream_call_reinforce_fn(self, text):
        return text

    # 获取跨度元数据的私有方法，返回处理后的元数据字典
    def _get_span_metadata(self, payload: Dict) -> Dict:
        # 复制 payload 的内容到 metadata
        metadata = {k: v for k, v in payload.items()}
        
        # 删除 metadata 中的 "prompt" 键
        del metadata["prompt"]
        
        # 将 metadata 中的 "messages" 转换为字典列表
        metadata["messages"] = list(
            map(lambda m: m if isinstance(m, dict) else m.dict(), metadata["messages"])
        )
        
        # 返回处理后的 metadata 字典
        return metadata
    async def stream_call(self):
        # TODO Retry when server connection error
        # 构建模型请求的载荷数据
        payload = await self._build_model_request()

        # 记录请求载荷信息到日志
        logger.info(f"payload request: \n{payload}")

        # 初始化空字符串，用于接收AI模型的响应文本
        ai_response_text = ""

        # 创建一个新的跟踪 span，并设置元数据为 payload 的字典形式
        span = root_tracer.start_span(
            "BaseChat.stream_call", metadata=payload.to_dict()
        )

        # 将 span 的 span_id 赋给 payload 的 span_id 属性
        payload.span_id = span.span_id

        try:
            # 使用异步迭代器调用流式操作函数 call_streaming_operator
            async for output in self.call_streaming_operator(payload):
                # 通过 prompt_template 解析模型流响应，0 为插件类型
                msg = self.prompt_template.output_parser.parse_model_stream_resp_ex(
                    output, 0
                )

                # 调用 stream_plugin_call 处理消息
                view_msg = self.stream_plugin_call(msg)

                # 将 view_msg 中的换行符替换为 "\\n"
                view_msg = view_msg.replace("\n", "\\n")

                # 生成处理后的消息
                yield view_msg

            # 将 AI 生成的消息添加到当前消息中
            self.current_message.add_ai_message(msg)

            # 强化流调用中的消息
            view_msg = self.stream_call_reinforce_fn(view_msg)

            # 将强化后的视图消息添加到当前消息中
            self.current_message.add_view_message(view_msg)

            # 结束当前 span 的跟踪
            span.end()

        except Exception as e:
            # 打印异常堆栈信息
            print(traceback.format_exc())

            # 记录错误日志
            logger.error("model response parse failed！" + str(e))

            # 将错误消息添加到视图消息中，并添加到当前会话消息中
            self.current_message.add_view_message(
                f"""<span style=\"color:red\">ERROR!</span>{str(e)}\n  {ai_response_text} """
            )

            # 结束当前 span 的跟踪，并附加错误元数据
            span.end(metadata={"error": str(e)})

        # 异步调用阻塞函数，结束当前会话轮次
        await blocking_func_to_async(
            self._executor, self.current_message.end_current_round
        )

    async def nostream_call(self):
        # 构建模型请求的载荷数据
        payload = await self._build_model_request()

        # 创建一个新的跟踪 span，并设置元数据为 payload 的字典形式
        span = root_tracer.start_span(
            "BaseChat.nostream_call", metadata=payload.to_dict()
        )

        # 记录请求信息到日志
        logger.info(f"Request: \n{payload}")

        # 将 span 的 span_id 赋给 payload 的 span_id 属性
        payload.span_id = span.span_id

        try:
            # 使用带重试的非流式调用函数 _no_streaming_call_with_retry
            ai_response_text, view_message = await self._no_streaming_call_with_retry(
                payload
            )

            # 将 AI 生成的消息添加到当前消息中
            self.current_message.add_ai_message(ai_response_text)

            # 将视图消息添加到当前消息中
            self.current_message.add_view_message(view_message)

            # 调整消息显示
            self.message_adjust()

            # 结束当前 span 的跟踪
            span.end()

        except BaseAppException as e:
            # 将 BaseAppException 的视图消息添加到当前消息中
            self.current_message.add_view_message(e.view)

            # 结束当前 span 的跟踪，并附加错误元数据
            span.end(metadata={"error": str(e)})

        except Exception as e:
            # 构造带有错误信息的视图消息
            view_message = f"<span style='color:red'>ERROR!</span> {str(e)}"

            # 将错误视图消息添加到当前消息中
            self.current_message.add_view_message(view_message)

            # 结束当前 span 的跟踪，并附加错误元数据
            span.end(metadata={"error": str(e)})

        # 异步调用阻塞函数，结束当前会话轮次
        await blocking_func_to_async(
            self._executor, self.current_message.end_current_round
        )

        # 返回当前的 AI 响应数据
        return self.current_ai_response()

    @async_retry(
        retries=CFG.DBGPT_APP_SCENE_NON_STREAMING_RETRIES_BASE,
        parallel_executions=CFG.DBGPT_APP_SCENE_NON_STREAMING_PARALLELISM_BASE,
        catch_exceptions=(Exception, BaseAppException),
    )
    # 异步方法，调用LLM操作并带有重试机制，不使用流式传输
    async def _no_streaming_call_with_retry(self, payload):
        # 使用根跟踪器创建一个跟踪 span，记录生成操作
        with root_tracer.start_span("BaseChat.invoke_worker_manager.generate"):
            # 调用LLM操作，获取模型输出
            model_output = await self.call_llm_operator(payload)

        # 解析模型输出，生成AI响应文本
        ai_response_text = self.prompt_template.output_parser.parse_model_nostream_resp(
            model_output, self.prompt_template.sep
        )
        # 解析AI响应文本，生成提示定义响应
        prompt_define_response = (
            self.prompt_template.output_parser.parse_prompt_response(ai_response_text)
        )
        # 构建元数据，记录模型输出、AI响应文本及其解析结果
        metadata = {
            "model_output": model_output.to_dict(),
            "ai_response_text": ai_response_text,
            "prompt_define_response": self._parse_prompt_define_response(
                prompt_define_response
            ),
        }
        # 使用根跟踪器创建一个跟踪 span，在元数据中包含记录的信息
        with root_tracer.start_span("BaseChat.do_action", metadata=metadata):
            # 使用阻塞函数将do_action方法异步化，执行动作
            result = await blocking_func_to_async(
                self._executor, self.do_action, prompt_define_response
            )

        # 获取LLM的语音输出
        speak_to_user = self.get_llm_speak(prompt_define_response)

        # 使用阻塞函数将视图响应解析方法异步化，解析视图消息
        view_message = await blocking_func_to_async(
            self._executor,
            self.prompt_template.output_parser.parse_view_response,
            speak_to_user,
            result,
            prompt_define_response,
        )
        # 返回AI响应文本和视图消息，替换换行符
        return ai_response_text, view_message.replace("\n", "\\n")

    # 异步方法，获取LLM的响应
    async def get_llm_response(self):
        # 构建模型请求payload
        payload = await self._build_model_request()
        logger.info(f"Request: \n{payload}")
        ai_response_text = ""
        prompt_define_response = None
        try:
            # 调用LLM操作，获取模型输出
            model_output = await self.call_llm_operator(payload)
            # 解析模型输出，生成AI响应文本
            ai_response_text = (
                self.prompt_template.output_parser.parse_model_nostream_resp(
                    model_output, self.prompt_template.sep
                )
            )
            # 将AI响应添加到当前消息中
            self.current_message.add_ai_message(ai_response_text)
            # 解析AI响应文本，生成提示定义响应
            prompt_define_response = (
                self.prompt_template.output_parser.parse_prompt_response(
                    ai_response_text
                )
            )
        except Exception as e:
            # 记录异常信息并添加视图消息
            print(traceback.format_exc())
            logger.error("model response parse failed！" + str(e))
            self.current_message.add_view_message(
                f"""model response parse failed！{str(e)}\n  {ai_response_text} """
            )
        # 返回提示定义响应
        return prompt_define_response

    # 阻塞式流调用方法，警告将被删除，请使用stream_call以获得更高性能
    def _blocking_stream_call(self):
        logger.warn(
            "_blocking_stream_call is only temporarily used in webserver and will be deleted soon, please use stream_call to replace it for higher performance"
        )
        # 获取或创建事件循环
        loop = get_or_create_event_loop()
        # 获取stream_call方法的异步生成器
        async_gen = self.stream_call()
        while True:
            try:
                # 运行异步生成器获取下一个值
                value = loop.run_until_complete(async_gen.__anext__())
                yield value
            except StopAsyncIteration:
                break
    # 提供警告日志，说明此方法仅在 web 服务器中临时使用，将来会被删除，请使用 nostream_call 方法以获得更高的性能
    def _blocking_nostream_call(self):
        # 记录警告日志消息
        logger.warn(
            "_blocking_nostream_call is only temporarily used in webserver and will be deleted soon, please use nostream_call to replace it for higher performance"
        )
        # 获取或创建事件循环
        loop = get_or_create_event_loop()
        try:
            # 运行并等待 nostream_call 方法完成，然后返回结果
            return loop.run_until_complete(self.nostream_call())
        finally:
            # 关闭事件循环
            loop.close()

    def call(self):
        # 如果 prompt_template 的 stream_out 属性为真，则使用流式调用 _blocking_stream_call 方法
        if self.prompt_template.stream_out:
            yield self._blocking_stream_call()
        else:
            # 否则，调用 _blocking_nostream_call 方法并返回其结果
            return self._blocking_nostream_call()

    async def prepare(self):
        # 占位符方法，当前为空实现
        pass

    def current_ai_response(self) -> str:
        # 获取当前消息中最后一条消息
        for message in self.current_message.messages[-1:]:
            # 如果消息类型为 "view"，则返回消息内容
            if message.type == "view":
                return message.content
        # 如果没有符合条件的消息，则返回 None
        return None

    async def prompt_context_token_adapt(self, prompt) -> str:
        """prompt token adapt according to llm max context length"""
        # 获取模型的元数据
        model_metadata = await self.worker_manager.get_model_metadata(
            {"model": self.llm_model}
        )
        # 计算当前 prompt 中的 token 数量
        current_token_count = await self.worker_manager.count_token(
            {"model": self.llm_model, "prompt": prompt}
        )
        # 如果未安装 tiktoken，current_token_count 会返回 -1，输出警告信息
        if current_token_count == -1:
            logger.warning(
                "tiktoken not installed, please `pip install tiktoken` first"
            )
        # 初始化模板定义 token 数量为 0
        template_define_token_count = 0
        # 如果 prompt_template 中的 template_define 不为空，则计算其 token 数量
        if len(self.prompt_template.template_define) > 0:
            template_define_token_count = await self.worker_manager.count_token(
                {
                    "model": self.llm_model,
                    "prompt": self.prompt_template.template_define,
                }
            )
            # 将 template_define_token_count 加到 current_token_count 中
            current_token_count += template_define_token_count
        # 如果当前 token 数量加上 prompt_template 的最大新增 token 数量大于模型的上下文长度
        if (
            current_token_count + self.prompt_template.max_new_tokens
        ) > model_metadata.context_length:
            # 调整 prompt 的长度，使其不超过模型的上下文长度减去最大新增 token 数量和 template_define_token_count
            prompt = prompt[
                : (
                    model_metadata.context_length
                    - self.prompt_template.max_new_tokens
                    - template_define_token_count
                )
            ]
        # 返回调整后的 prompt
        return prompt

    def generate(self, p) -> str:
        """
        generate context for LLM input
        Args:
            p: 

        Returns:

        """
        # 占位符方法，当前为空实现
        pass
    # 解析并返回处理后的提示定义响应
    def _parse_prompt_define_response(self, prompt_define_response: Any) -> Any:
        # 如果 prompt_define_response 为空，则返回空字符串
        if not prompt_define_response:
            return ""
        # 如果 prompt_define_response 是字符串或者字典，则直接返回它
        if isinstance(prompt_define_response, str) or isinstance(
            prompt_define_response, dict
        ):
            return prompt_define_response
        # 如果 prompt_define_response 是元组类型
        if isinstance(prompt_define_response, tuple):
            # 检查是否具有 _asdict 方法，表明是 namedtuple
            if hasattr(prompt_define_response, "_asdict"):
                # 将 namedtuple 转换为字典类型返回
                return prompt_define_response._asdict()
            else:
                # 将元组转换为字典，键为元素的索引，值为元素的值
                return dict(
                    zip(range(len(prompt_define_response)), prompt_define_response)
                )
        else:
            # 其他情况下直接返回 prompt_define_response
            return prompt_define_response

    # 生成带编号的列表作为字符串返回
    def _generate_numbered_list(self) -> str:
        """this function is moved from excel_analyze/chat.py,and used by subclass.
        Returns:
        
        """
        # 定义一个包含不同响应类型及其描述的列表
        antv_charts = [
            {"response_line_chart": "used to display comparative trend analysis data"},
            {
                "response_pie_chart": "suitable for scenarios such as proportion and distribution statistics"
            },
            {
                "response_table": "suitable for display with many display columns or non-numeric columns"
            },
            {
                "response_scatter_plot": "Suitable for exploring relationships between variables, detecting outliers, etc."
            },
            {
                "response_bubble_chart": "Suitable for relationships between multiple variables, highlighting outliers or special situations, etc."
            },
            {
                "response_donut_chart": "Suitable for hierarchical structure representation, category proportion display and highlighting key categories, etc."
            },
            {
                "response_area_chart": "Suitable for visualization of time series data, comparison of multiple groups of data, analysis of data change trends, etc."
            },
            {
                "response_heatmap": "Suitable for visual analysis of time series data, large-scale data sets, distribution of classified data, etc."
            },
        ]

        # 将列表转换为格式化字符串，每行包含响应类型及其描述
        return "\n".join(
            f"{key}:{value}"
            for dict_item in antv_charts
            for key, value in dict_item.items()
        )
```
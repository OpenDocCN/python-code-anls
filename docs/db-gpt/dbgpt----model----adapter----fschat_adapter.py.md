# `.\DB-GPT-src\dbgpt\model\adapter\fschat_adapter.py`

```py
"""Adapter for fastchat

You can import fastchat only in this file, so that the user does not need to install fastchat if he does not use it.
"""
import logging  # 导入日志模块
import os  # 导入操作系统功能模块
import threading  # 导入线程模块
from functools import cache  # 导入缓存装饰器
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple  # 导入类型提示相关模块

from dbgpt.model.adapter.base import LLMModelAdapter  # 导入LLM模型适配器基类
from dbgpt.model.adapter.template import ConversationAdapter, PromptType  # 导入对话适配器及提示类型

try:
    from fastchat.conversation import (  # 尝试导入fastchat的对话相关模块
        Conversation,
        SeparatorStyle,
        register_conv_template,
    )
except ImportError as exc:
    raise ValueError(
        "Could not import python package: fschat "
        "Please install fastchat by command `pip install fschat` "
    ) from exc  # 如果导入失败，抛出异常提示用户安装fastchat包

if TYPE_CHECKING:
    from fastchat.model.model_adapter import BaseModelAdapter  # 类型检查时导入fastchat模型适配器基类
    from torch.nn import Module as TorchNNModule  # 类型检查时导入PyTorch的神经网络模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象

thread_local = threading.local()  # 创建线程本地存储对象
_IS_BENCHMARK = os.getenv("DB_GPT_MODEL_BENCHMARK", "False").lower() == "true"  # 从环境变量中获取性能基准标志

# If some model is not in the blacklist, but it still affects the loading of DB-GPT, you can add it to the blacklist.
__BLACK_LIST_MODEL_PROMPT = []  # 定义模型提示黑名单列表

class FschatConversationAdapter(ConversationAdapter):
    """The conversation adapter for fschat."""

    def __init__(self, conv: Conversation):
        self._conv = conv  # 初始化传入的对话对象

    @property
    def prompt_type(self) -> PromptType:
        return PromptType.FSCHAT  # 返回适配器的对话类型为FSCHAT

    @property
    def roles(self) -> Tuple[str]:
        return self._conv.roles  # 返回对话对象的角色元组

    @property
    def sep(self) -> Optional[str]:
        return self._conv.sep  # 返回对话对象的分隔符

    @property
    def stop_str(self) -> str:
        return self._conv.stop_str  # 返回对话对象的停止字符串

    @property
    def stop_token_ids(self) -> Optional[List[int]]:
        return self._conv.stop_token_ids  # 返回对话对象的停止token ID列表

    def get_prompt(self) -> str:
        """Get the prompt string."""
        return self._conv.get_prompt()  # 获取对话对象的提示字符串

    def set_system_message(self, system_message: str) -> None:
        """Set the system message."""
        self._conv.set_system_message(system_message)  # 设置对话对象的系统消息字符串

    def append_message(self, role: str, message: str) -> None:
        """Append a new message.

        Args:
            role (str): The role of the message.
            message (str): The message content.
        """
        self._conv.append_message(role, message)  # 向对话对象添加新消息

    def update_last_message(self, message: str) -> None:
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.

        Args:
            message (str): The message content.
        """
        self._conv.update_last_message(message)  # 更新对话对象的最后一条消息

    def copy(self) -> "ConversationAdapter":
        """Copy the conversation."""
        return FschatConversationAdapter(self._conv.copy())  # 复制当前对话适配器对象的对话对象


class FastChatLLMModelAdapterWrapper(LLMModelAdapter):
    """Wrapping fastchat adapter"""
    # 初始化方法，接受一个名为adapter的BaseModelAdapter对象作为参数，并将其保存在实例变量self._adapter中
    def __init__(self, adapter: "BaseModelAdapter") -> None:
        self._adapter = adapter

    # 创建并返回一个FastChatLLMModelAdapterWrapper对象，使用当前对象的self._adapter作为参数
    def new_adapter(self, **kwargs) -> "LLMModelAdapter":
        return FastChatLLMModelAdapterWrapper(self._adapter)

    # 返回当前适配器对象是否使用了快速分词器的布尔值
    def use_fast_tokenizer(self) -> bool:
        return self._adapter.use_fast_tokenizer

    # 调用适配器对象的load_model方法，加载模型并返回加载结果
    def load(self, model_path: str, from_pretrained_kwargs: dict):
        return self._adapter.load_model(model_path, from_pretrained_kwargs)

    # 根据_IS_BENCHMARK的值决定返回哪个生成流函数：
    # 如果_IS_BENCHMARK为True，则返回来自fastchat_benchmarks_inference模块的generate_stream函数
    # 如果_IS_BENCHMARK为False，则返回来自model_adapter模块的get_generate_stream_function函数
    def get_generate_stream_function(self, model: "TorchNNModule", model_path: str):
        if _IS_BENCHMARK:
            from dbgpt.util.benchmarks.llm.fastchat_benchmarks_inference import (
                generate_stream,
            )
            return generate_stream
        else:
            from fastchat.model.model_adapter import get_generate_stream_function
            return get_generate_stream_function(model, model_path)

    # 使用适配器对象的get_default_conv_template方法获取默认的会话模板，并将其转换为FschatConversationAdapter对象返回
    # 如果获取到的会话模板为None，则返回None
    def get_default_conv_template(
        self, model_name: str, model_path: str
    ) -> Optional[ConversationAdapter]:
        conv_template = self._adapter.get_default_conv_template(model_path)
        return FschatConversationAdapter(conv_template) if conv_template else None

    # 返回当前对象的字符串表示形式，格式为 类名(适配器类所在模块.适配器类名)
    def __str__(self) -> str:
        return "{}({}.{})".format(
            self.__class__.__name__,
            self._adapter.__class__.__module__,
            self._adapter.__class__.__name__,
        )
# 定义一个私有函数，用于获取适配器以供快速聊天使用
def _get_fastchat_model_adapter(
    model_name: str,                           # 模型名称，用于确定适配器
    model_path: str,                           # 模型路径，用于加载模型
    caller: Callable[[str], None] = None,      # 可选参数，回调函数，接受模型路径作为参数
    use_fastchat_monkey_patch: bool = False,   # 是否使用快速聊天的monkey patch功能，默认为False
):
    from fastchat.model import model_adapter   # 导入fastchat的模型适配器

    _bak_get_model_adapter = model_adapter.get_model_adapter  # 备份原始的获取模型适配器函数

    try:
        if use_fastchat_monkey_patch:
            model_adapter.get_model_adapter = _fastchat_get_adapter_monkey_patch  # 如果使用monkey patch，则替换为自定义的函数
        thread_local.model_name = model_name  # 将模型名称存储在线程本地存储中
        _remove_black_list_model_of_fastchat()  # 调用私有函数，从fastchat中移除黑名单的模型
        if caller:
            return caller(model_path)  # 如果有回调函数，则使用模型路径调用回调函数
    finally:
        del thread_local.model_name  # 清除线程本地存储中的模型名称
        model_adapter.get_model_adapter = _bak_get_model_adapter  # 恢复原始的获取模型适配器函数


# 定义一个monkey patch函数，用于动态获取适配器
def _fastchat_get_adapter_monkey_patch(model_path: str, model_name: str = None):
    if not model_name:
        if not hasattr(thread_local, "model_name"):
            raise RuntimeError("fastchat get adapter monkey path need model_name")  # 如果模型名称未提供且线程本地存储中也没有，则抛出运行时错误
        model_name = thread_local.model_name  # 从线程本地存储中获取模型名称

    from fastchat.model.model_adapter import model_adapters  # 导入fastchat的模型适配器列表
    import os

    for adapter in model_adapters:
        if adapter.match(model_name):  # 遍历适配器列表，寻找与给定模型名称匹配的适配器
            logger.info(
                f"Found llm model adapter with model name: {model_name}, {adapter}"
            )
            return adapter

    model_path_basename = (
        None if not model_path else os.path.basename(os.path.normpath(model_path))
    )
    for adapter in model_adapters:
        if model_path_basename and adapter.match(model_path_basename):  # 如果提供了模型路径，则按基本名称匹配适配器
            logger.info(
                f"Found llm model adapter with model path: {model_path} and base name: {model_path_basename}, {adapter}"
            )
            return adapter

    for adapter in model_adapters:
        if model_path and adapter.match(model_path):  # 如果提供了模型路径，则按完整路径匹配适配器
            logger.info(
                f"Found llm model adapter with model path: {model_path}, {adapter}"
            )
            return adapter

    raise ValueError(
        f"Invalid model adapter for model name {model_name} and model path {model_path}"
    )


# 使用装饰器定义一个缓存函数，用于移除fastchat中的黑名单模型
@cache
def _remove_black_list_model_of_fastchat():
    from fastchat.model.model_adapter import model_adapters  # 导入fastchat的模型适配器列表

    black_list_models = []
    for adapter in model_adapters:
        try:
            if (
                adapter.get_default_conv_template("/data/not_exist_model_path").name
                in __BLACK_LIST_MODEL_PROMPT  # 尝试获取默认对话模板的名称，如果在黑名单中则添加到黑名单模型列表
            ):
                black_list_models.append(adapter)
        except Exception:
            pass
    for adapter in black_list_models:
        model_adapters.remove(adapter)  # 从fastchat的模型适配器列表中移除黑名单模型


# 注册对话模板的配置，用于快速聊天功能
# 参考自 https://huggingface.co/BAAI/AquilaChat2-34B/blob/4608b75855334b93329a771aee03869dbf7d88cc/predict.py#L212
register_conv_template(
    Conversation(
        name="aquila-legacy",  # 定义对话的名称为 "aquila-legacy"
        system_message="A chat between a curious human and an artificial intelligence assistant. "
                       "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",  
        # 系统消息，描述对话场景和机器助理的回答特点
        roles=("### Human: ", "### Assistant: ", "System"),  
        # 角色列表，包括人类角色、助理角色和系统角色
        messages=(),  # 初始为空的消息列表
        offset=0,  # 偏移量设置为0，即从头开始处理消息
        sep_style=SeparatorStyle.NO_COLON_TWO,  
        # 分隔样式，指定无冒号分隔和双行分隔
        sep="\n",  # 主分隔符为换行符
        sep2="</s>",  # 次分隔符为 "</s>"
        stop_str=["</s>", "[UNK]"],  
        # 停止字符串列表，包括 "</s>" 和 "[UNK]"
    ),
    override=True,  # 覆盖模式为真，表示将覆盖现有的对话设置
# 注册一个会话模板，命名为'aquila'，用于定义人类和人工智能助理之间的对话
register_conv_template(
    Conversation(
        name="aquila",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
                       "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant", "System"),  # 定义对话中的角色顺序为人类、助理、系统
        messages=(),  # 初始时消息列表为空
        offset=0,  # 偏移量设为0
        sep_style=SeparatorStyle.ADD_COLON_TWO,  # 使用特定的分隔符样式 ADD_COLON_TWO
        sep="###",  # 主分隔符设为 '###'
        sep2="</s>",  # 辅助分隔符设为 '</s>'
        stop_str=["</s>", "[UNK]"],  # 停止字符串列表包含 '</s>' 和 '[UNK]'
    ),
    override=True,  # 覆盖已存在的同名会话模板
)

# 注册另一个会话模板，命名为'aquila-v1'，用于定义片段的开始和结束以及额外信息的展示
register_conv_template(
    Conversation(
        name="aquila-v1",
        roles=("<|startofpiece|>", "<|endofpiece|>", ""),  # 定义角色为片段的开始、结束和额外信息的展示
        messages=(),  # 初始时消息列表为空
        offset=0,  # 偏移量设为0
        sep_style=SeparatorStyle.NO_COLON_TWO,  # 使用特定的分隔符样式 NO_COLON_TWO
        sep="",  # 主分隔符设为空字符串
        sep2="</s>",  # 辅助分隔符设为 '</s>'
        stop_str=["</s>", "<|endoftext|>"],  # 停止字符串列表包含 '</s>' 和 '<|endoftext|>'
    ),
    override=True,  # 覆盖已存在的同名会话模板
)
```
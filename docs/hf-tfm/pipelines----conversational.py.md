# `.\transformers\pipelines\conversational.py`

```py
import uuid  # 导入 uuid 模块，用于生成唯一标识符
from typing import Any, Dict, List, Union  # 导入 typing 模块，用于类型提示

from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging  # 导入一些辅助函数和模块
from .base import PIPELINE_INIT_ARGS, Pipeline  # 从 base 模块导入一些变量和类

# 如果 TensorFlow 可用，则导入 TensorFlow 模块
if is_tf_available():
    import tensorflow as tf

# 如果 PyTorch 可用，则导入 PyTorch 模块
if is_torch_available():
    import torch

# 获取日志记录器
logger = logging.get_logger(__name__)


class Conversation:
    """
    Utility class containing a conversation and its history. This class is meant to be used as an input to the
    [`ConversationalPipeline`]. The conversation contains several utility functions to manage the addition of new user
    inputs and generated model responses.

    Arguments:
        messages (Union[str, List[Dict[str, str]]], *optional*):
            The initial messages to start the conversation, either a string, or a list of dicts containing "role" and
            "content" keys. If a string is passed, it is interpreted as a single message with the "user" role.
        conversation_id (`uuid.UUID`, *optional*):
            Unique identifier for the conversation. If not provided, a random UUID4 id will be assigned to the
            conversation.

    Usage:

    ```python
    conversation = Conversation("Going to the movies tonight - any suggestions?")
    conversation.add_message({"role": "assistant", "content": "The Big lebowski."})
    conversation.add_message({"role": "user", "content": "Is it good?"})
    ```py"""

    def __init__(
        self, messages: Union[str, List[Dict[str, str]]] = None, conversation_id: uuid.UUID = None, **deprecated_kwargs
    ):
        """
        Initialize Conversation object.

        Arguments:
            messages (Union[str, List[Dict[str, str]]], *optional*):
                The initial messages to start the conversation, either a string, or a list of dicts containing "role" and
                "content" keys. If a string is passed, it is interpreted as a single message with the "user" role.
            conversation_id (`uuid.UUID`, *optional*):
                Unique identifier for the conversation. If not provided, a random UUID4 id will be assigned to the
                conversation.
            **deprecated_kwargs:
                Additional deprecated keyword arguments.
        """
        ):
            # 如果对话 ID 不存在，则生成一个新的 UUID 作为对话 ID
            if not conversation_id:
                conversation_id = uuid.uuid4()

            # 如果 messages 为空，则从 deprecated_kwargs 中提取 text，如果存在则构造消息列表
            if messages is None:
                text = deprecated_kwargs.pop("text", None)
                if text is not None:
                    messages = [{"role": "user", "content": text}]
                else:
                    messages = []
            # 如果 messages 是字符串，则构造消息列表
            elif isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]

            # 处理过时的参数 past_user_inputs 和 generated_responses
            # 新代码应完全避免使用这两个参数
            self._num_processed_user_inputs = 0
            generated_responses = deprecated_kwargs.pop("generated_responses", None)
            past_user_inputs = deprecated_kwargs.pop("past_user_inputs", None)
            # 如果 generated_responses 不为 None 但 past_user_inputs 为 None，则抛出 ValueError
            if generated_responses is not None and past_user_inputs is None:
                raise ValueError("generated_responses cannot be passed without past_user_inputs!")
            # 如果 past_user_inputs 不为 None，则处理 legacy 参数
            if past_user_inputs is not None:
                legacy_messages = []
                if generated_responses is None:
                    generated_responses = []
                # 结构化处理 past_user_inputs 和 generated_responses
                # 不使用 zip() 是因为它们的长度可能不一致
                for i in range(max([len(past_user_inputs), len(generated_responses)])):
                    if i < len(past_user_inputs):
                        legacy_messages.append({"role": "user", "content": past_user_inputs[i]})
                    if i < len(generated_responses):
                        legacy_messages.append({"role": "assistant", "content": generated_responses[i]})
                # 合并 legacy_messages 和当前 messages
                messages = legacy_messages + messages

            # 设置对话的 UUID 和消息列表
            self.uuid = conversation_id
            self.messages = messages

    # 判断两个 Conversation 对象是否相等
    def __eq__(self, other):
        # 如果 other 不是 Conversation 类型，则返回 False
        if not isinstance(other, Conversation):
            return False
        # 判断两个对象的 UUID 和消息列表是否相等
        return self.uuid == other.uuid or self.messages == other.messages

    # 添加消息到对话消息列表中
    def add_message(self, message: Dict[str, str]):
        # 检查消息中是否只包含 'role' 和 'content' 两个键
        if not set(message.keys()) == {"role", "content"}:
            raise ValueError("Message should contain only 'role' and 'content' keys!")
        # 检查消息中的 'role' 是否为 'user', 'assistant' 或 'system'
        if message["role"] not in ("user", "assistant", "system"):
            raise ValueError("Only 'user', 'assistant' and 'system' roles are supported for now!")
        # 添加消息到对话消息列表中
        self.messages.append(message)
    def add_user_input(self, text: str, overwrite: bool = False):
        """
        Add a user input to the conversation for the next round. This is a legacy method that assumes that inputs must
        alternate user/assistant/user/assistant, and so will not add multiple user messages in succession. We recommend
        just using `add_message` with role "user" instead.
        """
        # 检查是否上一条消息是用户输入，并且未被处理
        if len(self) > 0 and self[-1]["role"] == "user":
            # 如果设置了覆盖模式，将警告记录下来，并将新的用户输入覆盖到上一条未处理的输入
            if overwrite:
                logger.warning(
                    f'User input added while unprocessed input was existing: "{self[-1]["content"]}" was overwritten '
                    f'with: "{text}".'
                )
                self[-1]["content"] = text
            # 如果未设置覆盖模式，将警告记录下来，并忽略新的用户输入
            else:
                logger.warning(
                    f'User input added while unprocessed input was existing: "{self[-1]["content"]}" new input '
                    f'ignored: "{text}". Set `overwrite` to True to overwrite unprocessed user input'
                )
        # 如果上一条消息不是用户输入或者已被处理，将新的用户输入添加到对话中
        else:
            self.messages.append({"role": "user", "content": text})

    def append_response(self, response: str):
        """
        This is a legacy method. We recommend just using `add_message` with an appropriate role instead.
        """
        # 添加一个机器人的回复到对话中
        self.messages.append({"role": "assistant", "content": response})

    def mark_processed(self):
        """
        This is a legacy method, as the Conversation no longer distinguishes between processed and unprocessed user
        input. We set a counter here to keep behaviour mostly backward-compatible, but in general you should just read
        the messages directly when writing new code.
        """
        # 标记已处理的用户输入数量，保持向后兼容的行为
        self._num_processed_user_inputs = len(self._user_messages)

    def __iter__(self):
        # 在对话中迭代每一条消息
        for message in self.messages:
            yield message

    def __getitem__(self, item):
        # 获取对话中指定位置的消息
        return self.messages[item]

    def __setitem__(self, key, value):
        # 设置对话中指定位置的消息
        self.messages[key] = value

    def __len__(self):
        # 获取对话消息的数量
        return len(self.messages)

    def __repr__(self):
        """
        Generates a string representation of the conversation.

        Returns:
            `str`:

        Example:
            Conversation id: 7d15686b-dc94-49f2-9c4b-c9eac6a1f114 user: Going to the movies tonight - any suggestions?
            bot: The Big Lebowski
        """
        # 生成对话的字符串表示形式，包括对话的ID和每一条消息的内容
        output = f"Conversation id: {self.uuid}\n"
        for message in self.messages:
            output += f"{message['role']}: {message['content']}\n"
        return output

    def iter_texts(self):
        # 逐一迭代对话中的文本消息
        # This is a legacy method for backwards compatibility. It is recommended to just directly access
        # conversation.messages instead.
        for message in self.messages:
            yield message["role"] == "user", message["content"]

    @property
    def _user_messages(self):
        # 获取用户消息，推荐直接访问 conversation.messages
        return [message["content"] for message in self.messages if message["role"] == "user"]

    @property
    def past_user_inputs(self):
        # 获取过去用户输入，推荐直接访问 conversation.messages。现代类不关心哪些消息被"处理"或未处理。
        if not self._user_messages:
            return []
        # 过去，最近的用户消息必须在加入到 past_user_messages 之前进行 mark_processed() 处理。类实际上有一个单消息缓冲区，表示尚未回复的消息。
        # 现在不再需要这样做，但为了向后兼容，在此属性中模仿行为。
        if self.messages[-1]["role"] != "user" or self._num_processed_user_inputs == len(self._user_messages):
            return self._user_messages[:-1]

        return self._user_messages

    @property
    def generated_responses(self):
        # 获取生成的回复，推荐直接访问 conversation.messages
        return [message["content"] for message in self.messages if message["role"] == "assistant"]

    @property
    def new_user_input(self):
        # 获取最新用户输入，推荐直接访问 conversation.messages
        return self._user_messages[-1]
# 根据给定的参数添加文档字符串，详细描述了初始化参数
@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        min_length_for_response (`int`, *optional*, defaults to 32):
            The minimum length (in number of tokens) for a response.
        minimum_tokens (`int`, *optional*, defaults to 10):
            The minimum length of tokens to leave for a response.
    """,
)
# 定义一个多轮对话的Pipeline，继承自Pipeline类
class ConversationalPipeline(Pipeline):
    """
    多轮对话的Pipeline。

    例子：

    ```python
    >>> from transformers import pipeline, Conversation
    # 任何具有聊天模板的模型都可以用于ConversationalPipeline。

    >>> chatbot = pipeline(model="facebook/blenderbot-400M-distill")
    >>> # 用字符串初始化的Conversation对象将其视为用户消息
    >>> conversation = Conversation("I'm looking for a movie - what's your favourite one?")
    >>> conversation = chatbot(conversation)
    >>> conversation.messages[-1]["content"]
    "I don't really have a favorite movie, but I do like action movies. What about you?"

    >>> conversation.add_message({"role": "user", "content": "That's interesting, why do you like action movies?"})
    >>> conversation = chatbot(conversation)
    >>> conversation.messages[-1]["content"]
    " I think it's just because they're so fast-paced and action-fantastic."
    ```py

    了解有关在[pipeline tutorial](../pipeline_tutorial)中使用Pipeline的基础知识

    当前可以使用“conversational”的任务标识符从[`pipeline`]中加载此对话Pipeline。

    此Pipeline可以与任何具有[聊天模板](https://huggingface.co/docs/transformers/chat_templating)的模型一起使用。
    """

    # 初始化方法
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 如果分词器的pad_token_id为None，则将分词器的pad_token设置为分词器的eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # 清理参数，以便正确传递到生成对话的各个阶段
    def _sanitize_parameters(
        self, min_length_for_response=None, minimum_tokens=None, clean_up_tokenization_spaces=None, **generate_kwargs
    ):
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {}

        # 如果存在最小响应长度参数，则将其添加到预处理参数中
        if min_length_for_response is not None:
            preprocess_params["min_length_for_response"] = min_length_for_response
        # 如果存在最小标记数参数，则将其添加到转发参数中
        if minimum_tokens is not None:
            forward_params["minimum_tokens"] = minimum_tokens

        # 如果在生成参数中存在max_length，则添加到转发参数中
        if "max_length" in generate_kwargs:
            forward_params["max_length"] = generate_kwargs["max_length"]

        # 如果存在clean_up_tokenization_spaces参数，则将其添加到后处理参数中
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

        # 如果存在generate_kwargs，则将其合并到前向参数中
        if generate_kwargs:
            forward_params.update(generate_kwargs)
        return preprocess_params, forward_params, postprocess_params
    # 定义一个 __call__ 方法，用于生成对话输入的响应
    def __call__(self, conversations: Union[List[Dict], Conversation, List[Conversation]], num_workers=0, **kwargs):
        # 设置给定的对话及参数用于生成对话响应
        r"""
        Generate responses for the conversation(s) given as inputs.

        Args:
            conversations (a [`Conversation`] or a list of [`Conversation`]):
                对话信息，可以是一个 [`Conversation`] 对象，也可以是带有 `role` 和 `content` 键的字典列表 -
                在这种情况下，将自动转换为 `Conversation` 对象。多个对话可以作为列表传递。
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                是否清除文本输出中的可能额外空格。
            generate_kwargs:
                传递给模型生成方法的其他关键字参数(查看对应于您框架的生成方法 [这里](./model#generative-models))。

        Returns:
            [`Conversation`] or a list of [`Conversation`]: 生成响应后进行了更新的对话(包含新的用户输入)。
        """
        # XXX: num_workers==0 is required to be backward compatible
        # 否则线程将需要对话副本，会影响 GPU 上的性能，但必须选择此选项以支持这个 BC 变更。
        # 如果对话是以字典列表格式传递过来，则转换为 Conversation 对象
        if isinstance(conversations, list) and isinstance(conversations[0], dict):
            conversations = Conversation(conversations)
        # 如果对话以列表的形式传递，则转换成Conversation对象
        elif isinstance(conversations, list) and isinstance(conversations[0], list):
            conversations = [Conversation(conv) for conv in conversations]
        # 调用父类方法生成响应
        outputs = super().__call__(conversations, num_workers=num_workers, **kwargs)
        # 如果输出是列表且长度为1，则返回第一个元素
        if isinstance(outputs, list) and len(outputs) == 1:
            return outputs[0]
        # 返回输出
        return outputs

    # 预处理对话，生成模型所需的输入数据
    def preprocess(self, conversation: Conversation, min_length_for_response=32) -> Dict[str, Any]:
        # 应用对话模板，添加生成提示
        input_ids = self.tokenizer.apply_chat_template(conversation, add_generation_prompt=True)

        # 根据框架不同，将数据格式化处理为模型所需的输入格式
        if self.framework == "pt":
            input_ids = torch.LongTensor([input_ids])
        elif self.framework == "tf":
            input_ids = tf.constant([input_ids])
        # 返回格式化后的输入和对话信息
        return {"input_ids": input_ids, "conversation": conversation}
    # 用于模型的前向传播，生成输出
    def _forward(self, model_inputs, minimum_tokens=10, **generate_kwargs):
        # 获取输入的token数量
        n = model_inputs["input_ids"].shape[1]
        # 获取对话内容并从输入中移除
        conversation = model_inputs.pop("conversation")
        # 如果生成参数中未指定max_length且max_new_tokens，就设置max_new_tokens为256
        if "max_length" not in generate_kwargs and "max_new_tokens" not in generate_kwargs:
            generate_kwargs["max_new_tokens"] = 256
        # 生成输出的id
        output_ids = self.model.generate(**model_inputs, **generate_kwargs)
        # 如果模型是编码器-解码器模型，设置起始位置为1
        if self.model.config.is_encoder_decoder:
            start_position = 1
        # 否则设置起始位置为n
        else:
            start_position = n
        # 返回输出id和会话内容
        return {"output_ids": output_ids[:, start_position:], "conversation": conversation}
    
    # 后处理函数，用于处理模型输出
    def postprocess(self, model_outputs, clean_up_tokenization_spaces=True):
        # 获取输出的id
        output_ids = model_outputs["output_ids"]
        # 将输出id解码成文本答案
        answer = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        # 获取会话内容
        conversation = model_outputs["conversation"]
        # 将助手的回答添加到会话中
        conversation.add_message({"role": "assistant", "content": answer})
        # 返回更新后的会话内容
        return conversation
```
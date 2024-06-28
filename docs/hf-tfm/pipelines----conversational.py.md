# `.\pipelines\conversational.py`

```
# 导入必要的模块和库
import uuid  # 导入用于生成唯一标识符的模块
import warnings  # 导入警告处理模块
from typing import Any, Dict, List, Union  # 导入类型提示相关的模块

# 导入相对路径的模块和函数
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
# 从本地模块导入基础类和函数
from .base import Pipeline, build_pipeline_init_args

# 如果 TensorFlow 可用，则导入 TensorFlow 模块
if is_tf_available():
    import tensorflow as tf

# 如果 PyTorch 可用，则导入 PyTorch 模块
if is_torch_available():
    import torch

# 导入日志记录器
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
    ```

    """

    def __init__(
        self, messages: Union[str, List[Dict[str, str]]] = None, conversation_id: uuid.UUID = None, **deprecated_kwargs
    ):
        # 初始化函数，用于创建 Conversation 对象的实例
    ):
        # 如果未提供会话 ID，则生成一个新的 UUID 作为会话 ID
        if not conversation_id:
            conversation_id = uuid.uuid4()

        # 如果未提供消息内容，则从过时的关键字参数中取出"text"，创建包含用户角色和文本内容的消息列表
        if messages is None:
            text = deprecated_kwargs.pop("text", None)
            if text is not None:
                messages = [{"role": "user", "content": text}]
            else:
                messages = []
        elif isinstance(messages, str):
            # 如果消息直接是字符串，则转换为包含用户角色和文本内容的消息列表
            messages = [{"role": "user", "content": messages}]

        # 处理遗留的参数 - 新代码应完全避免使用 past_user_inputs 和 generated_responses
        # 设置已处理的用户输入数量为零
        self._num_processed_user_inputs = 0
        generated_responses = deprecated_kwargs.pop("generated_responses", None)
        past_user_inputs = deprecated_kwargs.pop("past_user_inputs", None)
        
        # 如果传入了 generated_responses 但未传入 past_user_inputs，则抛出异常
        if generated_responses is not None and past_user_inputs is None:
            raise ValueError("generated_responses cannot be passed without past_user_inputs!")

        # 如果传入了 past_user_inputs，则组装遗留消息列表和当前消息列表
        if past_user_inputs is not None:
            legacy_messages = []
            if generated_responses is None:
                generated_responses = []
            
            # 通过循环构建消息列表，长度可能不同，因此采用 max() 长度作为循环条件
            for i in range(max([len(past_user_inputs), len(generated_responses)])):
                if i < len(past_user_inputs):
                    legacy_messages.append({"role": "user", "content": past_user_inputs[i]})
                if i < len(generated_responses):
                    legacy_messages.append({"role": "assistant", "content": generated_responses[i]})
            
            # 合并遗留消息列表和当前消息列表
            messages = legacy_messages + messages

        # 设置会话的 UUID 和消息内容
        self.uuid = conversation_id
        self.messages = messages

    # 判断两个 Conversation 对象是否相等的方法
    def __eq__(self, other):
        if not isinstance(other, Conversation):
            return False
        return self.uuid == other.uuid or self.messages == other.messages

    # 向会话中添加消息的方法
    def add_message(self, message: Dict[str, str]):
        # 检查消息中只包含 "role" 和 "content" 两个键
        if not set(message.keys()) == {"role", "content"}:
            raise ValueError("Message should contain only 'role' and 'content' keys!")

        # 检查消息角色是否为 'user', 'assistant' 或 'system'
        if message["role"] not in ("user", "assistant", "system"):
            raise ValueError("Only 'user', 'assistant' and 'system' roles are supported for now!")

        # 添加消息到会话的消息列表中
        self.messages.append(message)
    def add_user_input(self, text: str, overwrite: bool = False):
        """
        Add a user input to the conversation for the next round. This is a legacy method that assumes that inputs must
        alternate user/assistant/user/assistant, and so will not add multiple user messages in succession. We recommend
        just using `add_message` with role "user" instead.
        """
        # 检查对话中是否有消息存在，并且最后一条消息的角色是用户
        if len(self) > 0 and self[-1]["role"] == "user":
            # 如果设置了 overwrite 参数为 True，则覆盖最后一条未处理的用户输入
            if overwrite:
                logger.warning(
                    f'User input added while unprocessed input was existing: "{self[-1]["content"]}" was overwritten '
                    f'with: "{text}".'
                )
                # 覆盖最后一条用户输入的内容为新的文本内容
                self[-1]["content"] = text
            else:
                logger.warning(
                    f'User input added while unprocessed input was existing: "{self[-1]["content"]}" new input '
                    f'ignored: "{text}". Set `overwrite` to True to overwrite unprocessed user input'
                )
        else:
            # 否则，将新的用户输入消息添加到对话中
            self.messages.append({"role": "user", "content": text})

    def append_response(self, response: str):
        """
        This is a legacy method. We recommend just using `add_message` with an appropriate role instead.
        """
        # 将机器人的回复添加到对话中，角色为助手
        self.messages.append({"role": "assistant", "content": response})

    def mark_processed(self):
        """
        This is a legacy method, as the Conversation no longer distinguishes between processed and unprocessed user
        input. We set a counter here to keep behaviour mostly backward-compatible, but in general you should just read
        the messages directly when writing new code.
        """
        # 将未处理的用户消息数量设置为已处理的用户消息数量
        self._num_processed_user_inputs = len(self._user_messages)

    def __iter__(self):
        # 实现迭代器接口，允许对 Conversation 对象进行迭代
        for message in self.messages:
            yield message

    def __getitem__(self, item):
        # 允许使用索引访问 Conversation 对象的消息
        return self.messages[item]

    def __setitem__(self, key, value):
        # 允许使用索引设置 Conversation 对象的消息
        self.messages[key] = value

    def __len__(self):
        # 返回 Conversation 对象中消息的数量
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
        # 生成 Conversation 对象的字符串表示形式，包含对话的 ID 和每条消息的角色及内容
        output = f"Conversation id: {self.uuid}\n"
        for message in self.messages:
            output += f"{message['role']}: {message['content']}\n"
        return output

    def iter_texts(self):
        # 提供对消息文本的迭代访问，返回 (是否为用户消息, 消息内容) 的元组
        # 这是为了向后兼容而存在，推荐直接访问 conversation.messages
        for message in self.messages:
            yield message["role"] == "user", message["content"]

    @property
    # 返回所有用户消息内容的列表，这是为了向后兼容而保留的遗留属性。
    # 推荐直接访问 conversation.messages 来获取消息。
    def _user_messages(self):
        return [message["content"] for message in self.messages if message["role"] == "user"]

    @property
    # 返回过去用户输入的列表，这是为了向后兼容而保留的遗留属性。
    # 推荐直接访问 conversation.messages 来获取消息。现代类不关心哪些消息被处理或未处理。
    # 在过去，最近的用户消息必须经过 mark_processed() 处理才能包含在 past_user_messages 中。
    # 类实际上有一个单消息缓冲区，表示尚未回复的消息。现在不再需要这样做，但为了向后兼容，在此属性中模仿其行为。
    def past_user_inputs(self):
        if not self._user_messages:
            return []
        if self.messages[-1]["role"] != "user" or self._num_processed_user_inputs == len(self._user_messages):
            return self._user_messages[:-1]

        return self._user_messages

    @property
    # 返回所有生成的助理响应内容的列表，这是为了向后兼容而保留的遗留属性。
    # 推荐直接访问 conversation.messages 来获取消息。
    def generated_responses(self):
        return [message["content"] for message in self.messages if message["role"] == "assistant"]

    @property
    # 返回最新的用户输入消息内容，这是为了向后兼容而保留的遗留属性。
    # 推荐直接访问 conversation.messages 来获取消息。
    def new_user_input(self):
        return self._user_messages[-1]
@add_end_docstrings(
    build_pipeline_init_args(has_tokenizer=True),
    r"""
        min_length_for_response (`int`, *optional*, defaults to 32):
            The minimum length (in number of tokens) for a response.""",
)
class ConversationalPipeline(Pipeline):
    """
    Multi-turn conversational pipeline.

    Example:

    ```python
    >>> from transformers import pipeline, Conversation
    # Any model with a chat template can be used in a ConversationalPipeline.

    >>> chatbot = pipeline(model="facebook/blenderbot-400M-distill")
    >>> # Conversation objects initialized with a string will treat it as a user message
    >>> conversation = Conversation("I'm looking for a movie - what's your favourite one?")
    >>> conversation = chatbot(conversation)
    >>> conversation.messages[-1]["content"]
    "I don't really have a favorite movie, but I do like action movies. What about you?"

    >>> conversation.add_message({"role": "user", "content": "That's interesting, why do you like action movies?"})
    >>> conversation = chatbot(conversation)
    >>> conversation.messages[-1]["content"]
    " I think it's just because they're so fast-paced and action-fantastic."
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This conversational pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"conversational"`.

    This pipeline can be used with any model that has a [chat
    template](https://huggingface.co/docs/transformers/chat_templating) set.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "`ConversationalPipeline` is now deprecated, and the functionality has been moved to the standard `text-generation` pipeline, which now accepts lists of message dicts as well as strings. This class will be removed in v4.42.",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)
        # Check if tokenizer does not have a pad token ID, set pad_token to eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _sanitize_parameters(self, min_length_for_response=None, clean_up_tokenization_spaces=None, **generate_kwargs):
        """
        Prepares and sanitizes generation parameters for text generation.

        Args:
            min_length_for_response (int, optional): Minimum length of response in tokens.
            clean_up_tokenization_spaces (bool, optional): Whether to clean up tokenization spaces.
            **generate_kwargs: Additional keyword arguments for text generation.

        Returns:
            tuple: Three dictionaries containing pre-process, forward, and post-process parameters.
        """
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {}

        if min_length_for_response is not None:
            preprocess_params["min_length_for_response"] = min_length_for_response

        if "max_length" in generate_kwargs:
            forward_params["max_length"] = generate_kwargs["max_length"]
            # self.max_length = generate_kwargs.get("max_length", self.model.config.max_length)
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

        if generate_kwargs:
            forward_params.update(generate_kwargs)

        return preprocess_params, forward_params, postprocess_params
    def __call__(self, conversations: Union[List[Dict], Conversation, List[Conversation]], num_workers=0, **kwargs):
        """
        Generate responses for the conversation(s) given as inputs.

        Args:
            conversations (a [`Conversation`] or a list of [`Conversation`]):
                Conversation to generate responses for. Inputs can also be passed as a list of dictionaries with `role`
                and `content` keys - in this case, they will be converted to `Conversation` objects automatically.
                Multiple conversations in either format may be passed as a list.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
                Whether or not to clean up the potential extra spaces in the text output.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./model#generative-models)).

        Returns:
            [`Conversation`] or a list of [`Conversation`]: Conversation(s) with updated generated responses for those
            containing a new user input.
        """
        # XXX: num_workers==0 is required to be backward compatible
        # Otherwise the threads will require a Conversation copy.
        # This will definitely hinder performance on GPU, but has to be opted
        # in because of this BC change.
        # 检查是否输入的是列表且第一个元素是字典，将其转换为 Conversation 对象
        if isinstance(conversations, list) and isinstance(conversations[0], dict):
            conversations = Conversation(conversations)
        # 检查是否输入的是列表且第一个元素是列表，将每个子列表转换为 Conversation 对象
        elif isinstance(conversations, list) and isinstance(conversations[0], list):
            conversations = [Conversation(conv) for conv in conversations]
        # 调用父类的 __call__ 方法进行生成响应
        outputs = super().__call__(conversations, num_workers=num_workers, **kwargs)
        # 如果输出是列表且长度为1，则返回第一个元素，否则返回整个列表
        if isinstance(outputs, list) and len(outputs) == 1:
            return outputs[0]
        return outputs

    def preprocess(self, conversation: Conversation, min_length_for_response=32) -> Dict[str, Any]:
        """
        Preprocesses the conversation to generate model inputs.

        Args:
            conversation (`Conversation`): Conversation object containing role and content information.
            min_length_for_response (`int`, *optional*, defaults to `32`):
                Minimum length required for the model to generate a response.

        Returns:
            Dict[str, Any]: Dictionary containing input_ids (tokenized input) and the original conversation object.
        """
        # 应用聊天模板并为生成模型输入的 token IDs
        input_ids = self.tokenizer.apply_chat_template(conversation, add_generation_prompt=True)

        # 根据所用的框架，将 input_ids 转换为适当的张量类型
        if self.framework == "pt":
            input_ids = torch.LongTensor([input_ids])
        elif self.framework == "tf":
            input_ids = tf.constant([input_ids])

        return {"input_ids": input_ids, "conversation": conversation}

    def _forward(self, model_inputs, **generate_kwargs):
        """
        Perform forward pass through the model to generate output IDs.

        Args:
            model_inputs (Dict[str, Any]): Dictionary containing input_ids (token IDs) and conversation object.
            generate_kwargs: Additional keyword arguments passed to the generate method of the model.

        Returns:
            Dict[str, Any]: Dictionary containing output_ids (generated token IDs) and conversation object.
        """
        # 获取输入 token IDs 的长度
        n = model_inputs["input_ids"].shape[1]
        # 获取并移除 model_inputs 中的 conversation 对象
        conversation = model_inputs.pop("conversation")
        # 如果 generate_kwargs 中未指定 max_length 或 max_new_tokens，则设置默认值
        if "max_length" not in generate_kwargs and "max_new_tokens" not in generate_kwargs:
            generate_kwargs["max_new_tokens"] = 256
        # 使用 generate 方法生成输出 token IDs
        output_ids = self.model.generate(**model_inputs, **generate_kwargs)
        # 根据模型配置，确定 start_position 的起始位置
        if self.model.config.is_encoder_decoder:
            start_position = 1
        else:
            start_position = n
        # 返回生成的输出 token IDs 和 conversation 对象
        return {"output_ids": output_ids[:, start_position:], "conversation": conversation}
    # 定义一个方法用于处理模型输出后续的后处理逻辑
    def postprocess(self, model_outputs, clean_up_tokenization_spaces=True):
        # 从模型输出中获取生成的文本的标识符序列
        output_ids = model_outputs["output_ids"]
        # 使用分词器将标识符序列解码成文本，跳过特殊标记符号，并可选择清理分词空格
        answer = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        # 从模型输出中获取对话对象，向其添加生成的助理回复消息
        conversation = model_outputs["conversation"]
        conversation.add_message({"role": "assistant", "content": answer})
        # 返回更新后的对话对象
        return conversation
```
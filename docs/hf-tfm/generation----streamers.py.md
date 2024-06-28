# `.\generation\streamers.py`

```py
# 从队列模块导入队列类
from queue import Queue
# 导入类型检查工具，用于类型提示
from typing import TYPE_CHECKING, Optional

# 如果 TYPE_CHECKING 为真，则从 ..models.auto 模块导入 AutoTokenizer 类
if TYPE_CHECKING:
    from ..models.auto import AutoTokenizer

# 基础流生成器的基类，用于所有生成器流类的继承
class BaseStreamer:
    """
    Base class from which `.generate()` streamers should inherit.
    """

    def put(self, value):
        """Function that is called by `.generate()` to push new tokens"""
        # 抛出未实现错误，子类需要实现该方法
        raise NotImplementedError()

    def end(self):
        """Function that is called by `.generate()` to signal the end of generation"""
        # 抛出未实现错误，子类需要实现该方法
        raise NotImplementedError()


class TextStreamer(BaseStreamer):
    """
    Simple text streamer that prints the token(s) to stdout as soon as entire words are formed.

    <Tip warning={true}>

    The API for the streamer classes is still under development and may change in the future.

    </Tip>

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:

        ```
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

        >>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
        >>> streamer = TextStreamer(tok)

        >>> # Despite returning the usual output, the streamer will also print the generated text to stdout.
        >>> _ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
        An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
        ```
    """

    def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs):
        # 初始化方法，接收一个自动标记器实例和可选参数
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        # 用于流处理的变量
        self.token_cache = []  # 初始化空的标记缓存列表
        self.print_len = 0  # 初始化打印长度为 0
        self.next_tokens_are_prompt = True  # 初始化下一个标记为提示状态
    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        # 检查输入值的维度和批处理大小是否符合要求
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        # 如果设置跳过提示且下一个标记是提示，则跳过处理
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # 将新标记添加到缓存并进行解码
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # 如果文本以换行符结尾，则刷新缓存
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # 如果最后一个标记是CJK字符，则打印这些字符
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # 否则，打印直到最后一个空格字符（简单的启发式方法，避免打印不完整的单词）
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        # 调用处理最终文本的回调函数
        self.on_finalized_text(printable_text)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        # 如果缓存中还有剩余内容，则刷新缓存
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        # 设置下一个标记为提示
        self.next_tokens_are_prompt = True
        # 调用处理最终文本的回调函数，并标志流结束
        self.on_finalized_text(printable_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        # 将新文本输出到标准输出，如果流结束则打印换行符
        print(text, flush=True, end="" if not stream_end else None)
    # 检查给定的代码点（CP）是否是CJK字符的代码点
    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 这里定义的“中文字符”是指CJK统一表意字符（Unicode块）中的任何字符：
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # 需要注意，尽管名称中包含CJK统一表意字符，但并非所有日文和韩文字符都包含在内。
        # 现代韩文Hangul字母使用了不同的Unicode块，日文的平假名和片假名也是如此。
        # 这些字母用于书写以空格分隔的词语，因此不被特别对待，会像其他语言一样处理。
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)            # CJK统一表意字符（4E00-9FFF）
            or (cp >= 0x3400 and cp <= 0x4DBF)        # CJK统一表意字符扩展A（3400-4DBF）
            or (cp >= 0x20000 and cp <= 0x2A6DF)      # CJK统一表意字符扩展B（20000-2A6DF）
            or (cp >= 0x2A700 and cp <= 0x2B73F)      # CJK统一表意字符扩展C（2A700-2B73F）
            or (cp >= 0x2B740 and cp <= 0x2B81F)      # CJK统一表意字符扩展D（2B740-2B81F）
            or (cp >= 0x2B820 and cp <= 0x2CEAF)      # CJK兼容扩展（2B820-2CEAF）
            or (cp >= 0xF900 and cp <= 0xFAFF)        # CJK兼容象形文字（F900-FAFF）
            or (cp >= 0x2F800 and cp <= 0x2FA1F)      # CJK兼容表意文字补充（2F800-2FA1F）
        ):  # 如果CP位于任何上述范围内，则返回True，表示是中文字符
            return True

        # 如果不在以上范围内，则返回False，表示不是中文字符
        return False
class TextIteratorStreamer(TextStreamer):
    """
    Streamer that stores print-ready text in a queue, to be used by a downstream application as an iterator. This is
    useful for applications that benefit from accessing the generated text in a non-blocking way (e.g. in an interactive
    Gradio demo).

    <Tip warning={true}>

    The API for the streamer classes is still under development and may change in the future.

    </Tip>

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenizer used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        timeout (`float`, *optional*):
            The timeout for the text queue. If `None`, the queue will block indefinitely. Useful to handle exceptions
            in `.generate()`, when it is called in a separate thread.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:

        ```
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        >>> from threading import Thread

        >>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
        >>> streamer = TextIteratorStreamer(tok)

        >>> # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        >>> generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
        >>> thread = Thread(target=model.generate, kwargs=generation_kwargs)
        >>> thread.start()
        >>> generated_text = ""
        >>> for new_text in streamer:
        ...     generated_text += new_text
        >>> generated_text
        'An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,'
        ```
    """

    def __init__(
        self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, timeout: Optional[float] = None, **decode_kwargs
    ):
        # 调用父类的初始化方法，传递 tokenizer 和 decode_kwargs
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        # 创建一个队列来存储生成的文本
        self.text_queue = Queue()
        # 初始化停止信号为 None
        self.stop_signal = None
        # 设置超时时间
        self.timeout = timeout

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        # 将新生成的文本放入队列中，如果流结束，则也放入停止信号
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        # 返回迭代器自身
        return self

    def __next__(self):
        # 从队列中获取值，如果是停止信号则抛出 StopIteration 异常，否则返回值
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value
```
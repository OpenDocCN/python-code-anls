# `.\transformers\generation\streamers.py`

```py
# 导入需要的模块
from queue import Queue
from typing import TYPE_CHECKING, Optional

# 如果是类型检查，则导入AutoTokenizer
if TYPE_CHECKING:
    from ..models.auto import AutoTokenizer

# 定义基础的流式处理器基类，`.generate()`的流处理器应该继承于此类
class BaseStreamer:
    """
    Base class from which `.generate()` streamers should inherit.
    """

    # 用于由`.generate()`调用以推送新的标记的函数
    def put(self, value):
        """Function that is called by `.generate()` to push new tokens"""
        raise NotImplementedError()

    # 用于由`.generate()`调用以表示生成结束的函数
    def end(self):
        """Function that is called by `.generate()` to signal the end of generation"""
        raise NotImplementedError()

# 文本流处理器，当形成完整单词时将标记打印到标准输出
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

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

        >>> tok = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
        >>> streamer = TextStreamer(tok)

        >>> # Despite returning the usual output, the streamer will also print the generated text to stdout.
        >>> _ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
        An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
        ```py
    """

    # 初始化文本流处理器
    def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        # 在流处理过程中使用的变量
        self.token_cache = []  # 标记缓存
        self.print_len = 0  # 打印长度
        self.next_tokens_are_prompt = True  # 下一个标记是否是提示
    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        # 检查输入的 tokens 是否符合要求
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            # 如果需要跳过提示且下一个 tokens 是提示，则直接返回
            self.next_tokens_are_prompt = False
            return

        # 将新的 token 添加到缓存中，并解码整个内容
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # 如果文本以换行符结尾，则刷新缓存
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # 如果最后一个 token 是中文字符，则打印这些字符
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # 否则，打印直到最后一个空格字符（简单的启发式方法，避免打印不完整的单词，可能会随后的 token 改变而改变）
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        # 刷新缓存，如果存在的话
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        # 将新文本打印到 stdout，如果流结束，则也打印一个换行符
        print(text, flush=True, end="" if not stream_end else None)
    # 检查给定的 Unicode 码点是否为中文字符
    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 中文字符的定义包括 CJK 统一表意文字区域内的所有内容：
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # 注意，CJK 统一表意文字区域并不包括所有日文和韩文字符，
        # 尽管其名称中含有“CJK”。现代韩文 Hangul 字母位于不同的区域，
        # 日文平假名和片假名也是如此。这些字母用于书写空格分隔的单词，
        # 因此不被特殊对待，而是像其他所有语言一样处理。
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)  # 基本汉字
            or (cp >= 0x3400 and cp <= 0x4DBF)  # 汉字扩展 A
            or (cp >= 0x20000 and cp <= 0x2A6DF)  # 汉字扩展 B
            or (cp >= 0x2A700 and cp <= 0x2B73F)  # 汉字扩展 C
            or (cp >= 0x2B740 and cp <= 0x2B81F)  # 汉字扩展 D
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  # 汉字扩展 E
            or (cp >= 0xF900 and cp <= 0xFAFF)  # CJK 兼容象形文字
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  # 兼容汉字
        ):  #
            return True

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

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        >>> from threading import Thread

        >>> tok = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
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
        ```py
    """

    def __init__(
        self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, timeout: Optional[float] = None, **decode_kwargs
    ):
        # 调用父类构造函数初始化
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        # 创建一个文本队列
        self.text_queue = Queue()
        # 初始化停止信号为 None
        self.stop_signal = None
        # 设置超时时间
        self.timeout = timeout

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        # 将新文本放入队��中
        self.text_queue.put(text, timeout=self.timeout)
        # 如果流结束，也将停止信号放入队列中
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        # 从队列中获取值，设置超时时间
        value = self.text_queue.get(timeout=self.timeout)
        # 如果值为停止信号，则抛出 StopIteration 异常
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value
```
# `.\DB-GPT-src\dbgpt\util\speech\base.py`

```py
"""
Base class for all voice classes.
"""
# 导入 abc 模块，用于抽象基类
import abc
# 导入线程锁，用于多线程同步
from threading import Lock
# 导入自定义的单例模块
from dbgpt.util.singleton import AbstractSingleton

# 定义 VoiceBase 类，继承自 AbstractSingleton 单例类
class VoiceBase(AbstractSingleton):
    """
    Base class for all voice classes.
    """

    # 初始化方法，设置 VoiceBase 实例的初始属性
    def __init__(self):
        """
        Initialize the voice class.
        """
        self._url = None  # 初始化 URL 属性为 None
        self._headers = None  # 初始化 headers 属性为 None
        self._api_key = None  # 初始化 api_key 属性为 None
        self._voices = []  # 初始化 voices 属性为空列表
        self._mutex = Lock()  # 创建一个线程锁对象，用于多线程同步
        self._setup()  # 调用 _setup 方法进行进一步初始化

    # say 方法，用于播放指定文本
    def say(self, text: str, voice_index: int = 0) -> bool:
        """
        Say the given text.

        Args:
            text (str): The text to say.
            voice_index (int): The index of the voice to use.
        """
        with self._mutex:  # 使用线程锁确保操作的原子性
            return self._speech(text, voice_index)  # 调用 _speech 方法进行文本播放

    # 抽象方法 _setup，用于设置语音、API 密钥等
    @abc.abstractmethod
    def _setup(self) -> None:
        """
        Setup the voices, API key, etc.
        """
        pass

    # 抽象方法 _speech，用于播放指定文本
    @abc.abstractmethod
    def _speech(self, text: str, voice_index: int = 0) -> bool:
        """
        Play the given text.

        Args:
            text (str): The text to play.
        """
        pass
```
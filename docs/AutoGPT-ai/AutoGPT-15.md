# AutoGPT源码解析 15

# `autogpts/autogpt/autogpt/speech/macos_tts.py`

该代码是一个基于Autogpt库的MacOS TTS（文本转语音） Voice实现。主要作用是当调用`MacOS TTS Voice`类时，会处理文本转语音的相关逻辑。具体来说，该代码以下几个方法实现了与Autogpt库的交互：

1. `__init__`方法：初始化函数，用于设置MacOS TTS Voice的接口和依赖库，以及定义一个独特的类名`MacOSTTS`。
2. `_setup`方法：在`__init__`方法内部，执行与操作系统相关的设置操作，例如创建TTS灵媒会话目录。
3. `_speech`方法：这个方法用于播放选定的文本。具体，根据用户提供的语音索引（0或1），创建并运行相应的命令。如果用户没有指定语音索引，则默认使用系统自带的"say"命令。


```py
""" MacOS TTS Voice. """
from __future__ import annotations

import os

from autogpt.speech.base import VoiceBase


class MacOSTTS(VoiceBase):
    """MacOS TTS Voice."""

    def _setup(self) -> None:
        pass

    def _speech(self, text: str, voice_index: int = 0) -> bool:
        """Play the given text."""
        if voice_index == 0:
            os.system(f'say "{text}"')
        elif voice_index == 1:
            os.system(f'say -v "Ava (Premium)" "{text}"')
        else:
            os.system(f'say -v Samantha "{text}"')
        return True

```

# `autogpts/autogpt/autogpt/speech/say.py`

这段代码定义了一个名为“Text to speech module”的功能模块。它从两个主要的模块中继承了类，一个是Threading中的Semaphore，另一个是typing.Literal。这个功能模块通过以下方式起作用：

1. 导入未来保留的类：from __future__ import annotations。
2. 从typing.Literal中导入Annotations。
3. 从Threading中的Semaphore导入一个名为Semaphore的类。
4. 从该模块的命名中，导入了Text to speech module。
5. 创建一个名为config的类，该类继承了UserConfigurable和SystemConfiguration。
6. 创建一个名为VoiceBase的类，该类继承了Voice。
7. 创建一个名为ElevenLabsSpeech的类，该类继承了ElevenLabsSpeech。
8. 创建一个名为GTTSVoice的类，该类继承了GTTSVoice。
9. 创建一个名为MacOSTTS的类，该类继承了MacOSTTS。
10. 创建一个名为StreamElementsConfig的类，该类继承了StreamElementsSpeech。
11. 创建一个名为StreamElementsSpeech的类，该类继承了StreamElementsSpeech。
12. 覆盖ElevenLabsSpeech的训练方法，并在该方法中传入参数。
13. 创建一个名为SystemConfiguration的类，该类继承了SystemConfiguration。
14. 创建一个名为UserConfigurable的类，该类继承了SystemConfiguration。
15. 覆盖UserConfigurable的get\_properties方法，用于获取配置文件中的属性。
16. 创建一个名为VoiceChanger的类，该类继承了Voice。
17. 创建一个名为Processor的类，该类继承了Processor。
18. 创建一个名为Semaphore的类，该类继承了Threading.Semaphore。
19. 创建一个名为Transformer的类，该类继承了Threading.Transformer。
20. 创建一个名为Thread的类，该类继承了Thread。
21. 创建一个名为Trainer的类，该类继承了Thread.Trainer。
22. 创建一个名为Callback的类，该类继承了Thread.Callback。
23. 创建一个名为Text的文化，该类继承了Python中的Text。
24. 创建一个名为transform的类，该类继承了Text.transform。
25. 创建一个名为Mean，该类继承了Text.mean。
26. 创建一个名为Sentence，该类继承了Text.sentence。
27. 创建一个名为TextVariable，该类继承了Text.variable。
28. 创建一个名为Translation，该类继承了Text.translation。
29. 创建一个名为Duration，该类继承了Text.duration。
30. 创建一个名为Registration，该类继承了Text.registration。
31. 创建一个名为Speak，该类继承了Text.speak。
32. 创建一个名为Voice，该类继承了Text.voice。
33. 创建一个名为FFSMulti，该类继承了FFSMulti。
34. 创建一个名为StreamElements，该类继承了StreamElementsSpeech。
35. 创建一个名为StreamElementsSpeech的类，该类继承了StreamElementsSpeech。
36. 创建一个名为ElevenLabs，该类继承了ElevenLabsSpeech。
37. 创建一个名为MacOS，该类继承了MacOSTTS。
38. 创建一个名为Threading，该类继承了Thread。
39. 创建一个名为SemaphoreSemaphore，该类继承了


```py
""" Text to speech module """
from __future__ import annotations

import threading
from threading import Semaphore
from typing import Literal, Optional

from autogpt.core.configuration.schema import SystemConfiguration, UserConfigurable

from .base import VoiceBase
from .eleven_labs import ElevenLabsConfig, ElevenLabsSpeech
from .gtts import GTTSVoice
from .macos_tts import MacOSTTS
from .stream_elements_speech import StreamElementsConfig, StreamElementsSpeech

```

The code you provided is a Python class named `TextToSpeechProvider` that is responsible for rendering text into speech for an application. Here's a brief overview of what it does:

- It has a configuration object named `config` that is a combination of user-provided values (such as the text to speak, voice quality, and platform) and some default values.
- It has a class method named `__init__` that is used to initialize the provider with the given configuration object.
- It has a class method named `say` that takes a text string and a voice index and renders the text into speech using the `say` method of the `VoiceBase` class.
- It has a class method named `__repr__` that returns a string representation of the provider.
- It has a static method named `_get_voice_engine` that gets the voice engine to use for the given configuration by calling the `config.provider` property.
- It imports the following classes: `ElevenLabsSpeech`, `MacOSTTS`, `StreamElementsSpeech`, `GTTSVoice`, and它们的 encapsulated classes.


```py
_QUEUE_SEMAPHORE = Semaphore(
    1
)  # The amount of sounds to queue before blocking the main thread


class TTSConfig(SystemConfiguration):
    speak_mode: bool = False
    provider: Literal[
        "elevenlabs", "gtts", "macos", "streamelements"
    ] = UserConfigurable(default="gtts")
    elevenlabs: Optional[ElevenLabsConfig] = None
    streamelements: Optional[StreamElementsConfig] = None


class TextToSpeechProvider:
    def __init__(self, config: TTSConfig):
        self._config = config
        self._default_voice_engine, self._voice_engine = self._get_voice_engine(config)

    def say(self, text, voice_index: int = 0) -> None:
        def _speak() -> None:
            success = self._voice_engine.say(text, voice_index)
            if not success:
                self._default_voice_engine.say(text, voice_index)
            _QUEUE_SEMAPHORE.release()

        if self._config.speak_mode:
            _QUEUE_SEMAPHORE.acquire(True)
            thread = threading.Thread(target=_speak)
            thread.start()

    def __repr__(self):
        return f"{self.__class__.__name__}(provider={self._voice_engine.__class__.__name__})"

    @staticmethod
    def _get_voice_engine(config: TTSConfig) -> tuple[VoiceBase, VoiceBase]:
        """Get the voice engine to use for the given configuration"""
        tts_provider = config.provider
        if tts_provider == "elevenlabs":
            voice_engine = ElevenLabsSpeech(config.elevenlabs)
        elif tts_provider == "macos":
            voice_engine = MacOSTTS()
        elif tts_provider == "streamelements":
            voice_engine = StreamElementsSpeech(config.streamelements)
        else:
            voice_engine = GTTSVoice()

        return GTTSVoice(), voice_engine

```

# `autogpts/autogpt/autogpt/speech/stream_elements_speech.py`

这段代码使用了Python 3中的一些新特性，包括`__future__`注解、import statements和函数式编程。下面逐步解释每个部分的作用：

1. `from __future__ import annotations`：这是一个Python 3+的注解，用于声明未来的函数或类可以定义。
2. `import logging`：导入Python标准库中的日志类`logging`。
3. `import os`：导入操作系统`os`。
4. `import requests`：导入第三方库`requests`，用于发送HTTP请求。
5. `from playsound import playsound`：导入`playsound`库，用于在运行时播放声音。
6. `from autogpt.core.configuration import SystemConfiguration, UserConfigurable`：从`autogpt.core`包中导入`SystemConfiguration`和`UserConfigurable`类，用于自定义AI模型的配置。
7. `from autogpt.speech.base import VoiceBase`：从`autogpt.speech`包中导入`VoiceBase`类，用于在运行时播放语音。
8. `logger = logging.getLogger(__name__)`：创建一个名为`logger`的 logger 实例，用于在运行时记录信息。
9. `logging.basicConfig(level=logging.DEBUG)`：设置日志记录器的级别为`DEBUG`，以便记录更多的信息。
10. ` StreamElementsConfig(voice: str = UserConfigurable(default="Brian"))`：创建一个名为`StreamElementsConfig`的类，用于配置AI模型的声音。`voice`属性是一个字符串，用于指定AI模型的声音来源。`UserConfigurable`接口用于设置默认值，如果用户没有指定默认值，则使用系统默认值。

该代码的作用是创建一个AI模型，用于在运行时播放声音。这个模型可以使用用户指定的声音来源进行播放。AI模型的配置使用了一个`SystemConfiguration`类和一个`VoiceBase`类，这些类用于设置AI模型的参数和初始化声音播放器。最后，`logging`代码用于在运行时记录信息，以便调试和错误追踪。


```py
from __future__ import annotations

import logging
import os

import requests
from playsound import playsound

from autogpt.core.configuration import SystemConfiguration, UserConfigurable
from autogpt.speech.base import VoiceBase

logger = logging.getLogger(__name__)


class StreamElementsConfig(SystemConfiguration):
    voice: str = UserConfigurable(default="Brian")


```

这段代码定义了一个名为 StreamElementsSpeech 的类，它继承自 VoiceBase 类。这个类的目的是实现一个将文本转换为语音输出的语音模块，用于基于 AutoGPT 的自然语言处理应用。

具体来说，这个类的 _setup 方法用于设置语音参数，包括 API key、声音等。在 _speech 方法中，我们使用 requests 库发送请求，获取来自 StreamElements API 的语音合成服务。如果请求成功，我们将合成好的音频文件写入到 "speech.mp3" 文件中，并使用 os.system(" playsound " "speech.mp3") 命令播放。如果请求失败或者播放过程中出现错误，我们会记录错误信息并返回 False。


```py
class StreamElementsSpeech(VoiceBase):
    """Streamelements speech module for autogpt"""

    def _setup(self, config: StreamElementsConfig) -> None:
        """Setup the voices, API key, etc."""
        self.config = config

    def _speech(self, text: str, voice: str, _: int = 0) -> bool:
        voice = self.config.voice
        """Speak text using the streamelements API

        Args:
            text (str): The text to speak
            voice (str): The voice to use

        Returns:
            bool: True if the request was successful, False otherwise
        """
        tts_url = (
            f"https://api.streamelements.com/kappa/v2/speech?voice={voice}&text={text}"
        )
        response = requests.get(tts_url)

        if response.status_code == 200:
            with open("speech.mp3", "wb") as f:
                f.write(response.content)
            playsound("speech.mp3")
            os.remove("speech.mp3")
            return True
        else:
            logger.error(
                "Request failed with status code: %s, response content: %s",
                response.status_code,
                response.content,
            )
            return False

```

# `autogpts/autogpt/autogpt/speech/__init__.py`

这段代码是一个Python模块，它包含了语音识别和语音合成的功能。模块包含两个函数，分别是TextToSpeechProvider和TTSConfig。

1. TextToSpeechProvider函数是一个基于自动链言语合成API的Python接口，它可以从文本输入中提取语音输出。这个函数需要通过训练或实例化来获取其TTS配置，并将配置作为参数传递给say函数。

2. TTSConfig函数是一个配置类，它包含了TTS的各种选项和设置。这个函数需要实例化并返回一个TTSConfig对象，以便在需要时进行初始化。


```py
"""This module contains the speech recognition and speech synthesis functions."""
from autogpt.speech.say import TextToSpeechProvider, TTSConfig

__all__ = ["TextToSpeechProvider", "TTSConfig"]

```

# `autogpts/autogpt/autogpt/url_utils/validators.py`

这段代码定义了一个名为 validate_url 的函数，其参数为一个函数类型，该函数需要接受一个字符串参数，并通过一系列验证规则检查该 URL 是否有效。以下是该函数的实际实现：
```py
import functools
import re
from typing import Any, Callable, ParamSpec, TypeVar
from urllib.parse import urljoin, urlparse

P = ParamSpec("P")
T = TypeVar("T")


def validate_url(func: Callable[P, T]) -> Callable[P, T]:
   """The method decorator validate_url is used to validate urls for any command that requires
   a url as an argument"""

   @functools.wraps(func)
   def wrapper(url: str, *args, **kwargs) -> Any:
       """Check if the URL is valid using a basic check, urllib check, and local file check

       Args:
           url (str): The URL to check

       Returns:
           the result of the wrapped function

       Raises:
           ValueError if the url fails any of the validation tests
       """
       # Most basic check if the URL is valid:
       if not re.match(r"^https?://", url):
           raise ValueError("Invalid URL format")
       # If the url is valid, check if it's a network location
       elif not is_valid_url(url):
           raise ValueError("Missing Scheme or Network location")
       # Restrict access to local files
       if check_local_file_access(url):
           raise ValueError("Access to local files is restricted")
       # Check URL length
       elif len(url) > 2000:
           raise ValueError("URL is too long")

       return func(sanitize_url(url), *args, **kwargs)

   return wrapper
```
函数内部首先通过调用 `re.match` 函数来验证 URL 是否符合基本的 URL 格式。然后，使用 `is_valid_url` 函数来验证 URL 是否包含有效的网络连接。如果上述两个验证步骤失败，将引发 `ValueError` 并返回。如果 URL 有效，则会尝试使用 `check_local_file_access` 函数来检查 URL 是否可以访问本地文件。最后，使用 `func` 参数将其包装起来，以便调用 `validate_url` 函数时可以获得更多的上下文。


```py
import functools
import re
from typing import Any, Callable, ParamSpec, TypeVar
from urllib.parse import urljoin, urlparse

P = ParamSpec("P")
T = TypeVar("T")


def validate_url(func: Callable[P, T]) -> Callable[P, T]:
    """The method decorator validate_url is used to validate urls for any command that requires
    a url as an argument"""

    @functools.wraps(func)
    def wrapper(url: str, *args, **kwargs) -> Any:
        """Check if the URL is valid using a basic check, urllib check, and local file check

        Args:
            url (str): The URL to check

        Returns:
            the result of the wrapped function

        Raises:
            ValueError if the url fails any of the validation tests
        """
        # Most basic check if the URL is valid:
        if not re.match(r"^https?://", url):
            raise ValueError("Invalid URL format")
        if not is_valid_url(url):
            raise ValueError("Missing Scheme or Network location")
        # Restrict access to local files
        if check_local_file_access(url):
            raise ValueError("Access to local files is restricted")
        # Check URL length
        if len(url) > 2000:
            raise ValueError("URL is too long")

        return func(sanitize_url(url), *args, **kwargs)

    return wrapper


```

这段代码定义了一个名为 `is_valid_url` 的函数，用于检查给定的 URL 是否有效。函数接受一个字符串参数 `url`，返回一个布尔值 `True` 或 `False`。

函数内部首先使用 `urlparse` 函数解析给定的 URL，获取其协议(scheme)和域名(netloc)。然后，使用 Python 的所有 `not` 函数，即 `not` 取反，获取反过来的结果，判断是否包含网络协议和域名。

如果 `is_valid_url` 函数返回 `True`，说明给定的 URL 是有效的，否则返回 `False`。


```py
def is_valid_url(url: str) -> bool:
    """Check if the URL is valid

    Args:
        url (str): The URL to check

    Returns:
        bool: True if the URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


```

I'm sorry, but I'm not sure what you are asking for. You are providing two functions, `sanitize_url` and `check_local_file_access`, but they don't seem to have any side effects or dependencies. Could you please provide more context or clarify what you are asking for?


```py
def sanitize_url(url: str) -> str:
    """Sanitize the URL

    Args:
        url (str): The URL to sanitize

    Returns:
        str: The sanitized URL
    """
    parsed_url = urlparse(url)
    reconstructed_url = f"{parsed_url.path}{parsed_url.params}?{parsed_url.query}"
    return urljoin(url, reconstructed_url)


def check_local_file_access(url: str) -> bool:
    """Check if the URL is a local file

    Args:
        url (str): The URL to check

    Returns:
        bool: True if the URL is a local file, False otherwise
    """
    local_prefixes = [
        "file:///",
        "file://localhost/",
        "file://localhost",
        "http://localhost",
        "http://localhost/",
        "https://localhost",
        "https://localhost/",
        "http://2130706433",
        "http://2130706433/",
        "https://2130706433",
        "https://2130706433/",
        "http://127.0.0.1/",
        "http://127.0.0.1",
        "https://127.0.0.1/",
        "https://127.0.0.1",
        "https://0.0.0.0/",
        "https://0.0.0.0",
        "http://0.0.0.0/",
        "http://0.0.0.0",
        "http://0000",
        "http://0000/",
        "https://0000",
        "https://0000/",
    ]
    return any(url.startswith(prefix) for prefix in local_prefixes)

```

# `autogpts/autogpt/autogpt/url_utils/__init__.py`

很抱歉，我不能解释以下代码的作用，因为我无法加载任何源代码。如果你能提供代码，我将尽力解释其作用。


```py

```

# `autogpts/autogpt/scripts/check_requirements.py`

这段代码的作用是安装 Poetry 包管理器，并检查项目中缺少的依赖包。具体步骤如下：

1. 首先定义了一个 `main` 函数，该函数创建一个 Poetry 项目，并检查项目中已安装的依赖关系。
2. 从 `poetry_project` 对象中获取 `locker` 对象，这个对象用于管理项目锁。
3. 从 `dependency_group` 对象中获取包含 `main` 依赖的所有包，并检查这些包的版本是否符合约束条件。
4. 如果任何包的版本不符合约束条件，则添加到 `missing_packages` 列表中。
5. 如果 `missing_packages` 列表不为空，则输出这些包的名称，并使用 `sys.exit()` 函数退出。

总的来说，这段代码的主要作用是安装并检查项目中缺少的依赖包，以便在项目中使用 Poetry 包管理器。


```py
import contextlib
import os
import sys
from importlib.metadata import version

try:
    import poetry.factory  # noqa
except ModuleNotFoundError:
    os.system(f"{sys.executable} -m pip install 'poetry>=1.6.1,<2.0.0'")

from poetry.core.constraints.version.version import Version
from poetry.factory import Factory


def main():
    poetry_project = Factory().create_poetry()
    # repository = poetry_project.locker.locked_repository()
    # dependencies = repository.packages
    dependency_group = poetry_project.package.dependency_group("main")

    missing_packages = []
    for dep in dependency_group.dependencies:
        # Try to verify that the installed version is suitable
        with contextlib.suppress(ModuleNotFoundError):
            installed_version = version(dep.name)  # if this fails -> not installed
            if dep.constraint.allows(Version.parse(installed_version)):
                continue
        # If the above verification fails, mark the package as missing
        missing_packages.append(str(dep))

    if missing_packages:
        print("Missing packages:")
        print(", ".join(missing_packages))
        sys.exit(1)


```

这段代码是一个Python程序中的一个if语句，它的作用是判断当前程序是否作为主程序运行。如果当前程序作为主程序运行，那么程序将跳转到if语句的内部，否则程序将继续执行if语句内部的代码。

在Python中，__name__属性返回模块的名称，如果这个名称等于"__main__"，那么这个模块就是Python标准库中的一个模块，这个模块是一个命令行脚本，可以直接运行程序。

因此，如果这段代码是作为主程序运行，那么它将跳转到程序的__main__函数内部，否则程序将继续执行if语句内部的代码。__main__函数是一个特殊的方法，它可以在程序作为主程序运行时被调用，也可以在程序运行时通过其他方式调用。


```py
if __name__ == "__main__":
    main()

```

# `autogpts/autogpt/scripts/install_plugin_deps.py`

This script appears to be designed to install zip-based and directory-based plugins for a Python application. The script uses the `plugins_dir` environment variable to store the directory containing the plugins, and the `PLUGINS_DIR` environment variable to surpress the warning message if it is not defined.

The script first checks for any zip-based plugins in the directory, and if one is found, extracts its contents and installs the dependencies using the `pip` package manager. It then checks for and installs any directory-based plugins.

It is important to note that this script may have security vulnerabilities since it installs the `pip` package, which could potentially be used to execute arbitrary code on the system.


```py
import logging
import os
import subprocess
import sys
import zipfile
from glob import glob
from pathlib import Path

logger = logging.getLogger(__name__)


def install_plugin_dependencies():
    """
    Installs dependencies for all plugins in the plugins dir.

    Args:
        None

    Returns:
        None
    """
    plugins_dir = Path(os.getenv("PLUGINS_DIR", "plugins"))

    logger.debug(f"Checking for dependencies in zipped plugins...")

    # Install zip-based plugins
    for plugin_archive in plugins_dir.glob("*.zip"):
        logger.debug(f"Checking for requirements in '{plugin_archive}'...")
        with zipfile.ZipFile(str(plugin_archive), "r") as zfile:
            if not zfile.namelist():
                continue

            # Assume the first entry in the list will be (in) the lowest common dir
            first_entry = zfile.namelist()[0]
            basedir = first_entry.rsplit("/", 1)[0] if "/" in first_entry else ""
            logger.debug(f"Looking for requirements.txt in '{basedir}'")

            basereqs = os.path.join(basedir, "requirements.txt")
            try:
                extracted = zfile.extract(basereqs, path=plugins_dir)
            except KeyError as e:
                logger.debug(e.args[0])
                continue

            logger.debug(f"Installing dependencies from '{basereqs}'...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", extracted]
            )
            os.remove(extracted)
            os.rmdir(os.path.join(plugins_dir, basedir))

    logger.debug(f"Checking for dependencies in other plugin folders...")

    # Install directory-based plugins
    for requirements_file in glob(f"{plugins_dir}/*/requirements.txt"):
        logger.debug(f"Installing dependencies from '{requirements_file}'...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            stdout=subprocess.DEVNULL,
        )

    logger.debug("Finished installing plugin dependencies")


```

这段代码是一个条件判断语句，它会判断当前脚本是否是主程序（__main__）。如果是主程序，则会执行if语句内的语句。这里的if语句会判断当前脚本是否是空字符串（""），如果是，则执行安装插件依赖管理器（install_plugin_dependencies）的函数。如果不是主程序，则直接跳过if语句内的语句。简单来说，这段代码的作用是判断当前脚本是否为空主程序，如果是，则安装依赖管理器并执行其中的函数。


```py
if __name__ == "__main__":
    install_plugin_dependencies()

```

# `autogpts/autogpt/scripts/__init__.py`

我需要更具体的上下文来回答你的问题。可以请你提供更多背景和上下文信息吗？


```py

```

# `autogpts/autogpt/tests/conftest.py`

这段代码的作用是测试一个名为"autogpt.agents.agent"的组件，它是一个AI代理，用于与OpenAI模型进行交互。以下是具体解释：

1. 导入一些必要的模块和函数：os、pathlib、tempfile、pytest、yaml、MockerFixture、AIProfile、Config、ConfigBuilder、ChatModelProvider、OpenAIProvider、FileWorkspace、ApiManager、logs.configure_logging。

2. 定义了一些变量：pathlib.Path(用于指定临时文件的目录)、TemporaryDirectory(用于创建临时文件目录)、os.path.join(用于指定 ChatModelProvider 和 OpenAIProvider 的路径)、os.path.join(用于指定 ChatModelProvider 的文件路径)、os.path.join(用于指定 OpenAIProvider 的路径)、os.path.join(用于指定 ChatModelProvider 的配置文件路径)、ApiManager(用于指定 ChatModelProvider 和 OpenAIProvider 的客户端实例)、MockerFixture(用于管理测试套件的环境变量)。

3. 加载了需要的配置文件：ConfigBuilder.from_yaml_file(openai_config_path)、Config.from_object(openai_config_path)、Agent.from_config(config)、AgentConfiguration.from_config(config)、AgentSettings.from_config(config)。

4. 创建了一个 ChatModelProvider 的实例：FileWorkspace.create_workspace_from_directory(根目录)、ApiManager.from_config(config)、MockerFixture.from_config(config)。

5. 创建了一个 OpenAIProvider 的实例：FileWorkspace.create_workspace_from_directory(根目录)、ApiManager.from_config(config)、MockerFixture.from_config(config)。

6. 创建了一个根目录：os.mkdir(用于指定 ChatModelProvider 和 OpenAIProvider 的根目录)、os.path.join(用于指定 ChatModelProvider 的文件路径)、os.path.join(用于指定 OpenAIProvider 的路径)。

7. 设置了一些变量：AIProfile.from_config(config)、Config.from_object(openai_config_path)、Agent.from_config(config)、AgentConfiguration.from_config(config)、AgentSettings.from_config(config)。

8. 运行了 pytest:pytest.main(argv=['--config', 'api_manager'], stdio=config.stdio)。


```py
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml
from pytest_mock import MockerFixture

from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
from autogpt.app.main import _configure_openai_provider
from autogpt.config import AIProfile, Config, ConfigBuilder
from autogpt.core.resource.model_providers import ChatModelProvider, OpenAIProvider
from autogpt.file_workspace import FileWorkspace
from autogpt.llm.api_manager import ApiManager
from autogpt.logs.config import configure_logging
```

这段代码的作用是测试自主学习语言模型的一个子模块，即`CommandRegistry`。通过在代码中导入了来自`autogpt.memory.vector`模块的`get_memory`函数，以及导入了来自`autogpt.models.command_registry`模块的`CommandRegistry`函数，可以实现对两个模块的协同测试，以更好地理解它们的功能和行为。

具体来说，这段代码会创建一个临时项目根目录，用于在测试过程中临时存放数据和结果。在测试过程中，可以调用`CommandRegistry`中的函数来执行一些命令，并将结果存储到内存中，然后使用`get_memory`函数从内存中读取数据，并输出结果。通过这样的方式，可以更好地理解`CommandRegistry`模块的功能和行为。


```py
from autogpt.memory.vector import get_memory
from autogpt.models.command_registry import CommandRegistry

pytest_plugins = [
    "tests.integration.agent_factory",
    "tests.integration.memory.utils",
    "tests.vcr",
]


@pytest.fixture()
def tmp_project_root(tmp_path: Path) -> Path:
    return tmp_path


```

这段代码定义了三个 fixture，用于在测试过程中提供数据和目录。

1. `app_data_dir` fixture 定义了一个数据目录，用于存储应用程序的数据。这个目录将在测试过程中被创建，并且在测试完成后被删除。

2. `agent_data_dir` fixture 定义了一个数据目录，用于存储人工智能模型的数据。这个目录将在测试过程中被创建，并且在测试完成后被删除。

3. `workspace_root` fixture 定义了一个数据目录，用于存储应用程序的工作空间。这个目录将在测试过程中被创建，并且在测试完成后被删除。

`@pytest.fixture()` 是装饰器，用于将 `app_data_dir`、`agent_data_dir` 和 `workspace_root` 中的任何一个数据目录作为测试函数的参数。

`tmp_project_root` 是一个虚拟环境变量，用于在测试过程中创建一个临时项目目录。这个目录将包含 `app_data_dir` 中定义的数据目录，但不包含 `agent_data_dir` 和 `workspace_root` 中定义的数据目录。


```py
@pytest.fixture()
def app_data_dir(tmp_project_root: Path) -> Path:
    return tmp_project_root / "data"


@pytest.fixture()
def agent_data_dir(app_data_dir: Path) -> Path:
    return app_data_dir / "agents/AutoGPT"


@pytest.fixture()
def workspace_root(agent_data_dir: Path) -> Path:
    return agent_data_dir / "workspace"


```

这两行代码定义了一个测试 fixture，名为 `temp_plugins_config_file`。该 fixture 会在测试运行时创建一个临时 directory，并将一个名为 `plugins_config.yaml` 的配置文件保存在该目录中。

具体来说，这两行代码的作用如下：

1. `@pytest.fixture()`：这是一个用于定义测试函数的装饰器，它告诉 pytest 应该如何使用这个 fixture。

2. `workspace(workspace_root: Path) -> FileWorkspace:`：这行代码定义了 `workspace` fixture，它接受一个 `workspace_root` 参数，并返回一个 `FileWorkspace` 对象。这个 `FileWorkspace` 对象是一个基于文件的工作区，可以限制工作区中的文件只读或者在同一时间只允许访问一次。

3. `workspace.initialize()`：在 `FileWorkspace` 对象被创建后，需要调用 `initialize()` 方法来设置一些默认值，例如最大文件大小限制、时间戳限制等。

4. `return workspace`：这行代码返回经过 `initialize()` 方法设置好初始状态的 `FileWorkspace` 对象。

5. `@pytest.fixture(lambda`：这行代码告诉 pytest 如何使用 `temp_plugins_config_file` fixture。

6. `temp_plugins_config_file()`：这个函数名告诉 pytest 如何使用 `temp_plugins_config_file` fixture。它会在测试运行时创建一个临时目录，并将 `plugins_config.yaml` 配置文件保存在该目录中。

7. `with open(config_file, "w+") as f:`：这行代码打开了一个可读写文件，并将 `config_file` 中的内容写入其中。

8. `f.write(yaml.dump({}))`：这行代码将一个字典对象用 YAML 格式写入到 `config_file` 中。

9. `yield config_file`：这行代码返回经过 `write()` 方法写入内容的 `config_file` 对象，以便在测试中重复使用。


```py
@pytest.fixture()
def workspace(workspace_root: Path) -> FileWorkspace:
    workspace = FileWorkspace(workspace_root, restrict_to_root=True)
    workspace.initialize()
    return workspace


@pytest.fixture
def temp_plugins_config_file():
    """Create a plugins_config.yaml file in a temp directory so that it doesn't mess with existing ones"""
    config_directory = TemporaryDirectory()
    config_file = Path(config_directory.name) / "plugins_config.yaml"
    with open(config_file, "w+") as f:
        f.write(yaml.dump({}))

    yield config_file


```

这段代码定义了一个 fixture，名为 `config`，用于在 Pytest 测试运行时创建一个配置对象。该配置对象包含了应用程序的数据目录、应用程序的配置目录、以及应用程序的插件目录和插件配置文件等信息。

具体来说，该配置对象的构建过程如下：

1. 从 `project_root` 环境变量中读取应用程序的配置目录，如果不存在，则创建一个根目录。

2. 从 `openaig_api_key` 环境变量中读取 OpenAI API 密钥，如果没有该密钥，则将其设置为 `sk-dummy`。

3. 将应用程序的数据目录设置为指定的目录。

4. 将应用程序的插件目录设置为指定的目录，并将插件配置文件设置为指定的目录。

5. 设置应用程序为非交互式模式，即不输出任何输出信息。

6. 将 `plugins_dir` 环境变量设置为测试插件目录。

7. 将 `plugins_config_file` 环境变量设置为测试插件配置文件。

8. 设置 `noninteractive_mode` 环境变量为 `True`，这意味着该插件不会与用户交互，即不会提示用户安装或禁用插件。

9. 将 `plain_output` 环境变量设置为 `True`，这意味着该插件会将输出信息打印为纯文本，而不是包含 HTML 标记。

该 `config` fixture 可以在每一次测试运行时被使用，从而每次运行测试时都会创建一个新的配置对象。


```py
@pytest.fixture()
def config(
    temp_plugins_config_file: Path,
    tmp_project_root: Path,
    app_data_dir: Path,
    mocker: MockerFixture,
):
    config = ConfigBuilder.build_config_from_env(project_root=tmp_project_root)
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "sk-dummy"

    config.app_data_dir = app_data_dir

    config.plugins_dir = "tests/unit/data/test_plugins"
    config.plugins_config_file = temp_plugins_config_file

    config.noninteractive_mode = True
    config.plain_output = True

    # avoid circular dependency
    from autogpt.plugins.plugins_config import PluginsConfig

    config.plugins_config = PluginsConfig.load_config(
        plugins_config_file=config.plugins_config_file,
        plugins_denylist=config.plugins_denylist,
        plugins_allowlist=config.plugins_allowlist,
    )
    yield config


```

这段代码是一个Python测试框架中的装饰函数，用于对测试函数的setup阶段（即测试函数被调用前的阶段）进行设置。

具体来说，这段代码的作用是：

1. 使用@pytest.fixture(scope="session")装饰函数，定义一个名为setup_logger的函数，其作用是在整个测试函数测试执行前的初始化阶段执行。
2. 使用configure_logging()函数，设置日志输出配置，包括开启调试模式、输出为 plain_output 以及日志目录等。
3. 在setup_logger()函数中，使用 config.parent / "logs" 获取日志目录，创建一个名为 "logs" 的目录，并将 config.debug_mode 和 config.plain_output 设置为 True 和 True，以便在日志中记录测试的调试信息和 plain 输出。
4. 在setup_logger()函数中，使用 ApiManager._instances.append() 方法添加一个名为 "ApiManager" 的实例，以便在测试中使用。
5. 在setup_logger()函数中，使用 ApiManager() 创建一个名为 "api_manager" 的实例，以便在测试中使用。
6. 在api_manager() 函数中，如果 ApiManager 类中已经存在名为 "ApiManager" 的实例，则删除该实例，以避免在测试中重复创建。
7. 在api_manager() 函数中，创建一个新的 ApiManager 实例，并将其返回，以便在测试中使用。


```py
@pytest.fixture(scope="session")
def setup_logger(config: Config):
    configure_logging(
        debug_mode=config.debug_mode,
        plain_output=config.plain_output,
        log_dir=Path(__file__).parent / "logs",
    )


@pytest.fixture()
def api_manager() -> ApiManager:
    if ApiManager in ApiManager._instances:
        del ApiManager._instances[ApiManager]
    return ApiManager()


```

这段代码使用了Python中的pytest和numpy库，以及OpenAIChat和OpenAIProvider模块。

它定义了一个fixture装饰器，用于在测试函数中管理实验对象。这个fixture装饰器接收一个配置对象和一个ChatModelProvider对象作为参数。

在函数内部，使用_configure_openai_provider函数来设置OpenAI提供器的配置。然后，使用return语句返回一个ChatModelProvider对象，这个对象将用于创建实验对象。

接下来的代码定义了一个agent函数，这个函数接收一个AIProfile对象，一个配置对象和一个ChatModelProvider对象作为参数。

在函数内部，创建了一个AIProfile对象，设置了一些AI的目标。然后，创建了一个CommandRegistry对象，以及一个PromptConfig对象，这个对象的配置与Agent.default_settings.prompt_config相似，但是这个函数的配置是使用函数式API。

接着，设置了一个Agent的设置对象，包含了一些AI的设置，这个对象的配置使用了函数式API，还使用了函数式API中的一个plugins参数。

最后，使用AttachFS方法将一个指定的数据目录挂载到了Agent上，并返回这个Agent。


```py
@pytest.fixture
def llm_provider(config: Config) -> OpenAIProvider:
    return _configure_openai_provider(config)


@pytest.fixture
def agent(
    agent_data_dir: Path, config: Config, llm_provider: ChatModelProvider
) -> Agent:
    ai_profile = AIProfile(
        ai_name="Base",
        ai_role="A base AI",
        ai_goals=[],
    )

    command_registry = CommandRegistry()

    agent_prompt_config = Agent.default_settings.prompt_config.copy(deep=True)
    agent_prompt_config.use_functions_api = config.openai_functions

    agent_settings = AgentSettings(
        name=Agent.default_settings.name,
        description=Agent.default_settings.description,
        ai_profile=ai_profile,
        config=AgentConfiguration(
            fast_llm=config.fast_llm,
            smart_llm=config.smart_llm,
            allow_fs_access=not config.restrict_to_workspace,
            use_functions_api=config.openai_functions,
            plugins=config.plugins,
        ),
        prompt_config=agent_prompt_config,
        history=Agent.default_settings.history.copy(deep=True),
    )

    agent = Agent(
        settings=agent_settings,
        llm_provider=llm_provider,
        command_registry=command_registry,
        legacy_config=config,
    )
    agent.attach_fs(agent_data_dir)
    return agent

```

# `autogpts/autogpt/tests/context.py`

这段代码的作用是：

1. 导入 os 和 sys 模块。
2. 将 scripts 目录添加到 sys.path 列表中，以便在程序中使用 browse 模块。
3. 在 scripts 目录下是否存在时，如果存在，则使用 absolute path 获取其路径并将其插入到 sys.path 的开头。

sys.path 列表是一个变量，用于存储 Python 模块的搜索路径。在该列表中，每个元素都是绝对路径，表示 Python 模块可能被安装在的文件系统中。通过将 scripts 目录添加到 sys.path 中，可以使程序在运行时使用 browse 模块，即使该模块不存在于当前安装的 Python环境中。


```py
import os
import sys

# Add the scripts directory to the path so that we can import the browse module.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts"))
)

```

# `autogpts/autogpt/tests/utils.py`

这段代码是一个 Python 函数，它的作用是：

1. 导入两个 Python 模块：os 和 pytest。
2. 定义一个名为 skip_in_ci的函数，它接受一个名为 test_function 的参数。
3. 在 skip_in_ci 函数中使用 pytest.mark.skipif 装饰器，来决定是否跳过 CI 测试。
4. 在 skip_in_ci 函数中添加一个条件，判断操作系统是否为 GitHub Actions。
5. 定义一个名为 get_workspace_file_path 的函数，它接受两个参数：工作区（workspace）和文件名（file_name）。
6. 在 get_workspace_file_path 函数中，使用 workspace.get_path 方法获取文件路径。
7. 最后，在 main 函数中调用 skip_in_ci 和 get_workspace_file_path 函数。


```py
import os

import pytest


def skip_in_ci(test_function):
    return pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="This test doesn't work on GitHub Actions.",
    )(test_function)


def get_workspace_file_path(workspace, file_name):
    return str(workspace.get_path(file_name))

```

# `autogpts/autogpt/tests/__init__.py`

很抱歉，我需要更多的上下文来回答您的问题。如果能提供更多上下文，我将非常乐意帮助理解代码的作用。


```py

```

# `autogpts/autogpt/tests/integration/agent_factory.py`

这段代码使用了两个 autogpt 的依赖库，pytest 和 autogpt。

autogpt 是一个基于 transformer 的自然语言处理框架，可以用于生成文本、回答问题等等。

pytest 是一个测试框架，可以用来编写和运行各种类型的测试。

autogpt 的代码中，import 自定义的类，包括 Agent、AgentConfiguration 和 AgentSettings，以及从 autogpt 的 Config 类中导入 Config 类。这些类和类成员可以被用于定义和设置 agen 的参数和设置。

在代码的主体部分，定义了一个 memory_json_file 函数，接受一个 Config 对象作为参数，将其 memory_backend 设置为 json_file，即从 autogpt 的 Config 类中读取 json_file 的配置，并将其记忆缓冲区的后端更改为 json_file。

在测试函数中，使用了 get_memory 函数获取一个可用的记忆缓冲区，并将其清空。然后，使用记忆缓冲区，通过 for 循环向其中写入数据，并输出结果。最后，在 for 循环外，将记忆缓冲区的后端设置为原本的 memory_backend，即 Config 中的 was_memory_backend。


```py
import pytest

from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
from autogpt.config import AIProfile, Config
from autogpt.memory.vector import get_memory
from autogpt.models.command_registry import CommandRegistry


@pytest.fixture
def memory_json_file(config: Config):
    was_memory_backend = config.memory_backend

    config.memory_backend = "json_file"
    memory = get_memory(config)
    memory.clear()
    yield memory

    config.memory_backend = was_memory_backend


```

这段代码定义了一个dummy_agent作为pytest-fixture库的一个装饰器函数，用于在测试函数中创建一个虚拟代理实体的实例。

具体来说，这个函数接收三个参数：配置对象(config)、LLM提供者实例(llm_provider)和一个内存中的JSON文件路径(memory_json_file)，用于在函数运行时加载LLM模型和相关配置信息。

然后，函数内部创建了一个命令注册表(CommandRegistry)，以及一个AI Profile对象，其中AI名称称为"Dummy Agent",AI角色称为"Dummy Role",AI目标包括"Dummy Task"。

接着，函数创建了一个Agent对象，使用了default_settings.prompt_config和AgentConfiguration来自定义Agent的 prompt_config和历史。同时，也通过提供者实例(llm_provider)允许了函数使用其LLM服务。

最后，函数返回了一个Agent实例，并在函数内部使用了该实例来执行测试用例。


```py
@pytest.fixture
def dummy_agent(config: Config, llm_provider, memory_json_file):
    command_registry = CommandRegistry()

    ai_profile = AIProfile(
        ai_name="Dummy Agent",
        ai_role="Dummy Role",
        ai_goals=[
            "Dummy Task",
        ],
    )

    agent_prompt_config = Agent.default_settings.prompt_config.copy(deep=True)
    agent_prompt_config.use_functions_api = config.openai_functions
    agent_settings = AgentSettings(
        name=Agent.default_settings.name,
        description=Agent.default_settings.description,
        ai_profile=ai_profile,
        config=AgentConfiguration(
            fast_llm=config.fast_llm,
            smart_llm=config.smart_llm,
            use_functions_api=config.openai_functions,
            plugins=config.plugins,
        ),
        prompt_config=agent_prompt_config,
        history=Agent.default_settings.history.copy(deep=True),
    )

    agent = Agent(
        settings=agent_settings,
        llm_provider=llm_provider,
        command_registry=command_registry,
        legacy_config=config,
    )

    return agent

```

# `autogpts/autogpt/tests/integration/test_execute_code.py`

这段代码的作用是进行人工智能相关任务。具体来说，它包括以下几个步骤：

1. 导入需要用到的模块：random、string、tempfile、pathlib、pytest。
2. 从pathlib模块中导入Path对象，以便于在测试中使用。
3. 从autogpt库中导入Agents库中的Agent类，以及执行代码的函数sut。
4. 创建一个临时文件，用于在测试中保存代码。
5. 从Agents库中导入InvalidArgumentError和OperationNotAllowedError类，以处理在测试中可能出现的错误。
6. 创建一个测试框架对象pytest，并使用sut执行代码。
7. 在测试中使用Agents库中的Agent类，对代码进行训练和测试。




```py
import random
import string
import tempfile
from pathlib import Path

import pytest

import autogpt.commands.execute_code as sut  # system under testing
from autogpt.agents.agent import Agent
from autogpt.agents.utils.exceptions import (
    InvalidArgumentError,
    OperationNotAllowedError,
)


```

这段代码使用了Python的联邦测试框架，主要用于测试中生成随机代码。代码的作用是定义了两个函数，`random_code` 函数会生成一段随机的字符串，而 `python_test_file` 函数会生成一段随机代码并保存到磁盘中的一个临时文件中。

具体来说，代码中的 `random_code` 函数接收一段随机字符串，然后将其返回。这个函数可以被看作是一个生成随机代码的装饰器。在 `python_test_file` 函数中，使用了 ` Agent` 类来管理测试中使用的资源和数据，包括磁盘和临时文件。

在 `random_code` 函数中，使用了 `f"print('Hello {random_string}!')"` 格式字符串来输出 "Hello " 和 "random_string"。这个字符串中的 `{}` 部分会被替换为 `random_string`，从而生成一段随机的字符串。

在 `python_test_file` 函数中，使用了 `tempfile.NamedTemporaryFile` 类来创建一个临时文件，并使用 `str.encode` 函数将 `random_code` 中的字符串转换成字节序列，然后将字节序列写入文件中。

接着，使用了 `tempfile.ShortFile` 类来创建一个临时文件并行写入数据，最后使用 `tempfile.NamedTemporaryFile.write` 方法将文件保存为磁盘中的文件。

在 `random_code` 函数中，使用 `yield` 语句来生成一个随机文件，并将其返回。这个随机文件可以包含任何字节序列，因此我们可以使用它来生成任何随机的字符串。

在 `python_test_file` 函数中，使用了 `Path` 类来生成文件名，并将其保存到磁盘中的临时文件中。

最后，在 `random_code` 函数中，使用了 `print` 函数来输出 "Hello random_string!"。


```py
@pytest.fixture
def random_code(random_string) -> str:
    return f"print('Hello {random_string}!')"


@pytest.fixture
def python_test_file(agent: Agent, random_code: str):
    temp_file = tempfile.NamedTemporaryFile(dir=agent.workspace.root, suffix=".py")
    temp_file.write(str.encode(random_code))
    temp_file.flush()

    yield Path(temp_file.name)
    temp_file.close()


```



This code defines two fixture functions, `python_test_args_file` and `random_string`, that are used in the context of the `pytest` module.

`python_test_args_file` is a fixture function that takes an `Agent` object as an argument. It creates a temporary file in the workspace root directory of the `Agent` object, writes the string `"import sys\nprint(sys.argv[1], sys.argv[2])"` to the file, and then flushes the file. The yield statement returns a path to the temporary file, which is used by the test function or fixture to fixture the file.

`random_string` is another fixture function that returns a random string of lowercase ASCII characters of length 10. This string is useful for testing that the output of the function is as expected.

The `@pytest.fixture` decorator is used to automatically recognize the fixture functions when they are called during the test suite, and to allow for easier testing of their behavior.


```py
@pytest.fixture
def python_test_args_file(agent: Agent):
    temp_file = tempfile.NamedTemporaryFile(dir=agent.workspace.root, suffix=".py")
    temp_file.write(str.encode("import sys\nprint(sys.argv[1], sys.argv[2])"))
    temp_file.flush()

    yield Path(temp_file.name)
    temp_file.close()


@pytest.fixture
def random_string():
    return "".join(random.choice(string.ascii_lowercase) for _ in range(10))


```

这两函数用于测试如何执行一个给定的Python文件，并使用不同的参数。第一个函数接收一个Python测试文件路径、一个随机字符串和一个AI代理。第二个函数接收一个Python测试参数文件路径、一个随机字符串和一个AI代理。这两个函数都使用sut.execute_python_file函数来执行Python文件，并将结果存储在result变量中。然后使用assert语句检查结果是否等于预先定义的输出字符串。

具体来说，第一个函数的目的是测试执行给定参数的Python文件并输出结果，并检查结果是否包含与随机字符串相同的字符串。如果结果正确，则可以验证函数可以正常工作。第二个函数的目的是测试执行给定参数的Python文件，并输出结果。如果AI代理确实可以执行给定文件并输出正确结果，则可以验证函数可以正常工作。


```py
def test_execute_python_file(python_test_file: Path, random_string: str, agent: Agent):
    result: str = sut.execute_python_file(python_test_file, agent=agent)
    assert result.replace("\r", "") == f"Hello {random_string}!\n"


def test_execute_python_file_args(
    python_test_args_file: Path, random_string: str, agent: Agent
):
    random_args = [random_string] * 2
    random_args_string = " ".join(random_args)
    result = sut.execute_python_file(
        python_test_args_file, args=random_args, agent=agent
    )
    assert result == f"{random_args_string}\n"


```

这段代码是一个测试用例，用于测试sut.execute_python_code函数的正确性。

具体来说，这段代码包含以下三个测试用例：

1. test_execute_python_code测试用例，用于测试sut.execute_python_code函数的正确性。在这个测试用例中，通过传入不同的random_code参数和random_string参数，来测试sut.execute_python_code函数能否正确地执行Python代码并返回预期的结果。具体地，代码中通过sut.execute_python_code函数执行Python代码，并返回一个结果变量result。然后，通过assert语句来验证结果变量result是否等于"Hello <random_string>！"。

2. test_execute_python_file_invalid测试用例，用于测试sut.execute_python_file函数的正确性。在这个测试用例中，传入的参数是"not_python"，表示要执行的Python文件名不含有"/".然后代码通过sut.execute_python_file函数执行这个文件，并捕获一个InvalidArgumentError异常。具体地，代码会尝试加载这个文件，但会因为文件不存在而产生异常。

3. test_execute_python_file_not_found测试用例，用于测试sut.execute_python_file函数的正确性。在这个测试用例中，传入的参数是"notexist.py"，表示要执行的Python文件名是"notexist.py"。然后代码通过sut.execute_python_file函数执行这个文件，并捕获一个FileNotFoundError异常。具体地，代码会尝试打开这个文件，但由于文件不存在而产生异常。


```py
def test_execute_python_code(random_code: str, random_string: str, agent: Agent):
    result: str = sut.execute_python_code(random_code, agent=agent)
    assert result.replace("\r", "") == f"Hello {random_string}!\n"


def test_execute_python_file_invalid(agent: Agent):
    with pytest.raises(InvalidArgumentError):
        sut.execute_python_file("not_python", agent)


def test_execute_python_file_not_found(agent: Agent):
    with pytest.raises(
        FileNotFoundError,
        match=r"python: can't open file '([a-zA-Z]:)?[/\\\-\w]*notexist.py': \[Errno 2\] No such file or directory",
    ):
        sut.execute_python_file("notexist.py", agent)


```

以上代码使用了Python的pytest库进行单元测试，具体解释如下：

```py
# 定义测试函数
def test_execute_shell(random_string: str, agent: Agent):
   # 执行 shell 命令并获取输出结果
   result = sut.execute_shell(f"echo 'Hello {random_string}'", agent)
   # 检查输出结果中是否包含 "Hello {random_string}!"
   assert "Hello {random_string}!" in result


   # 执行 shell 命令并使用本地命令
   def execute_shell(command):
       return sut.execute_shell(command, agent)
   result = execute_shell(f"echo 'Hello {random_string}'")
   assert result == "Hello {random_string}"


   # 禁止使用本地命令
   agent.legacy_config.shell_denylist = ["echo"]

   def execute_shell(command):
       return sut.execute_shell(command, agent)
   with pytest.raises(OperationNotAllowedError, match="not allowed"):
       execute_shell(f"echo 'Hello {random_string}'")
```

以上代码的作用是测试 `sut.execute_shell` 函数的正确性。其中，第一个测试函数 `test_execute_shell` 测试了使用 `sut.execute_shell` 函数输出 "Hello {random_string}" 是否正确；第二个测试函数 `test_execute_shell_local_commands_not_allowed` 测试了使用 `sut.execute_shell` 函数并禁止使用本地命令是否正确；第三个测试函数 `test_execute_shell_denylist_should_deny` 测试了使用 `sut.execute_shell` 函数并禁止使用本地命令是否能够成功地阻止使用本地命令。


```py
def test_execute_shell(random_string: str, agent: Agent):
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert f"Hello {random_string}!" in result


def test_execute_shell_local_commands_not_allowed(random_string: str, agent: Agent):
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert f"Hello {random_string}!" in result


def test_execute_shell_denylist_should_deny(agent: Agent, random_string: str):
    agent.legacy_config.shell_denylist = ["echo"]

    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        sut.execute_shell(f"echo 'Hello {random_string}!'", agent)


```

这段代码是一个使用SUT（模拟用户操作工具）进行测试的函数，用于测试在给定agent对象和随机字符串的情况下，执行shell命令是否被允许。

具体来说，这两部分测试函数分别尝试在给定的agent对象上执行echo "Hello {random_string}!"命令，并检查是否产生允许的结果。如果允许，测试将跳过，否则将引发一个名为"OperationNotAllowedError"的异常，该异常将捕获给定的错误信息并包含"not allowed"的提示。


```py
def test_execute_shell_denylist_should_allow(agent: Agent, random_string: str):
    agent.legacy_config.shell_denylist = ["cat"]

    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert "Hello" in result and random_string in result


def test_execute_shell_allowlist_should_deny(agent: Agent, random_string: str):
    agent.legacy_config.shell_command_control = sut.ALLOWLIST_CONTROL
    agent.legacy_config.shell_allowlist = ["cat"]

    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        sut.execute_shell(f"echo 'Hello {random_string}!'", agent)


```

这段代码定义了一个函数 `test_execute_shell_allowlist_should_allow`，该函数使用了 `sut` 和 `agent` 两个参数。函数的作用是允许 agent 执行一个特定的 shell 命令，这个命令会输出一个随机的字符串。

具体来说，函数接收两个参数：`agent` 和 `random_string`。首先，函数将 `agent` 的 `legacy_config` 设置为一个允许运行 `sut.ALLOWLIST_CONTROL` 和 `sut.execute_shell` 的 Shell，同时将 `shell_allowlist` 设置为 `["echo"]`。这里的 `sut.execute_shell` 是一个通用的函数，允许您执行任意 Shell 命令，并返回一个包含结果字符串的元组。

接下来，函数调用了 `sut.execute_shell` 函数，并传递了一个特定的执行命令：`f"echo 'Hello {random_string}!'`。这个执行命令会尝试运行 `sut.execute_shell` 函数，并输出一个包含 `random_string` 字符串的元组。

最后，函数使用 `assert` 语句来验证 `sut.execute_shell` 函数的输出是否包含 `random_string`。如果输出结果中包含 `random_string`，那么函数就不会输出任何错误信息，否则就会输出一个错误消息。


```py
def test_execute_shell_allowlist_should_allow(agent: Agent, random_string: str):
    agent.legacy_config.shell_command_control = sut.ALLOWLIST_CONTROL
    agent.legacy_config.shell_allowlist = ["echo"]

    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert "Hello" in result and random_string in result

```

# `autogpts/autogpt/tests/integration/test_image_gen.py`

这段代码使用了Python的一些常用的库和模块：

1. `functools` 库：这是一个Python函数式编程的支持库，提供了很多实用的函数，例如 `reduce`、`all`、`zip_longest` 等。
2. `hashlib` 库：这是一个Python哈希库，提供了很多哈希算法，例如 SHA-1、SHA-256、MD5 等。
3. `pathlib` 库：这是一个Python路径库，提供了一些用于处理文件和目录的函数和类，例如 `Path`、`resolve`、`directory_言` 等。
4. `unittest.mock` 库：这是一个Python测试框架中的测试马戏团，可以模拟Python中一些常见库和模块的行为，例如 `raise.NEOptimizingException`、` side_effect` 等。
5. `PIL` 库：这是一个Python图像库，提供了很多图像处理和显示的函数和类，例如 `Image`、`ImageDraw`、`ImageFont` 等。
6. `autogpt.agents.agent`：这是AutogPT中的一个类，定义了agent的接口，包括一些生成图像的方法，例如 `generate_image`、`generate_image_with_sd_webui` 等。
7. `pytest`：这是Python的一个测试框架，可以用于编写各种类型的测试，例如单元测试、功能测试、集成测试等。

接下来，会具体解释一下代码中的各个部分：

1. `import functools`：引入了functools库，用于在函数中使用f-函数风格的语法。
2. `import hashlib`：引入了hashlib库，用于在需要哈希算法的时候进行引入。
3. `from pathlib import Path`：引入了pathlib库，提供了用于处理文件和目录的函数和类，例如 `Path`、`resolve`、`directory_言` 等。
4. `unittest.mock import patch`：引入了unittest.mock库，用于模拟Python中一些常见库和模块的行为，例如 `raise.NEOptimizingException`、` side_effect` 等。
5. `Image.fromarray`：这是一个函数，用于将一个字节数组转换成一个Image对象，其中`fromarray`表示从字节数组中创建Image，而`Image.frombytes`表示从文件或其他来源中创建Image。
6. `generate_image`：这是一个方法，用于从AutogPT中的Agent对象中生成图像，其参数包括`image_size`、`num_細節`、`text`、`render_count`、`饶X`等，可以生成指定尺寸的图像，并返回一个Image对象。
7. `generate_image_with_sd_webui`：这也是一个方法，用于从AutogPT中的Agent对象中生成图像，其参数包括`image_size`、`num_細節`、`text`、`render_count`、`饶X`等，可以生成指定尺寸的图像，并返回一个Image对象，同时提供了使用SD WebUI生成的图像。
8. `@pytest.fixture(params=[256, 512, 1024])`：这是一个 fixture，用于控制对`image_size`的测试参数，其中`params`用于指定可能的值，这里是256、512或1024。
9. `def image_size(request):`：这是一个参数为`request`的函数，用于生成图像的大小，这里使用了`params`中指定的值。
10. `return request.param`：返回了`params`中指定的值，作为参数传递给`generate_image`和`generate_image_with_sd_webui`中的生成图像的方法。


```py
import functools
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from autogpt.agents.agent import Agent
from autogpt.commands.image_gen import generate_image, generate_image_with_sd_webui


@pytest.fixture(params=[256, 512, 1024])
def image_size(request):
    """Parametrize image size."""
    return request.param


```

这段代码使用了pytest的mark.requires_openai_api_key和mark.vcr装饰来定义一个测试函数，名为`test_dalle`。在该函数中，定义了三个参数：`agent`表示一个Agent对象，`workspace`表示一个工作区对象，`image_size`表示图像的大小，`patched_api_requestor`表示一个被 patch过的API 请求者对象。

函数内部调用了`generate_and_validate`函数，该函数的作用是在给定的Agent、工作区和图像大小的条件下，使用DALL-E API生成图像并验证其正确性。其中，参数`image_provider`指定为`dalle`，表示使用DALL-E API生成图像。

该函数使用了pytest的mark.xfail装饰来抛出异常，失败的情况是在运行测试时产生的。失败理由是图像大小超过了该测试函数所允许的最大值，需要进一步处理这个问题。


```py
@pytest.mark.requires_openai_api_key
@pytest.mark.vcr
def test_dalle(agent: Agent, workspace, image_size, patched_api_requestor):
    """Test DALL-E image generation."""
    generate_and_validate(
        agent,
        workspace,
        image_provider="dalle",
        image_size=image_size,
    )


@pytest.mark.xfail(
    reason="The image is too big to be put in a cassette for a CI pipeline. We're looking into a solution."
)
```

这段代码使用了pytestmark和pytest-parametrize。它的作用是测试HuggingFace中的图像生成功能，特别是在给定的图像大小下，使用给定的图像模型。

具体来说，这段代码会创建一个HuggingFace的图像代理（Agent），然后使用给定的工作空间（Workspace）来处理图像。此外，还会设置图像大小（image_size）以控制生成的图像的大小。接下来，将图像代理的模型设置为给定的图像模型（image_model）。

在测试部分，这段代码会使用给定的图像模型生成图像，并验证生成的图像是否符合预期。如果一切正常，这段代码会通过pytestmark输出“requires_huggingface_api_key”标记，并将结果记录在测试报告中。


```py
@pytest.mark.requires_huggingface_api_key
@pytest.mark.parametrize(
    "image_model",
    ["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1"],
)
def test_huggingface(agent: Agent, workspace, image_size, image_model):
    """Test HuggingFace image generation."""
    generate_and_validate(
        agent,
        workspace,
        image_provider="huggingface",
        image_size=image_size,
        hugging_face_image_model=image_model,
    )


```

这段代码是一个用于测试 SD WebUI 图像生成的 Python 函数，使用了 Pytest 库的 `@pytest.mark.xfail` 注解来标记测试中可能失败的功能。

具体来说，这段代码包含了两个测试函数：`test_sd_webui` 和 `test_sd_webui_negative_prompt`。这两个函数的作用都是生成一个带有 SD WebUI 图像的测试图像，并验证其是否与使用负面提示生成的图像哈希值相同。

在函数内部，首先通过调用 `generate_image_with_sd_webui` 函数来生成带有 SD WebUI 图像的测试图像。然后通过调用 `gen_image` 函数，传入一个负向请求，其中 `negative_prompt` 参数为 "astronaut riding a horse"，生成了一个带有负面提示的图像。接着，通过 `Image.open` 函数打开生成的图像文件，并获取其哈希值。最后，比较生成的图像哈希值是否与负向请求得到的哈希值相同。

如果两个图像哈希值相同，说明 SD WebUI 图像生成失败，抛出 `pytest.core.py. failure.Failure` 异常。


```py
@pytest.mark.xfail(reason="SD WebUI call does not work.")
def test_sd_webui(agent: Agent, workspace, image_size):
    """Test SD WebUI image generation."""
    generate_and_validate(
        agent,
        workspace,
        image_provider="sd_webui",
        image_size=image_size,
    )


@pytest.mark.xfail(reason="SD WebUI call does not work.")
def test_sd_webui_negative_prompt(agent: Agent, workspace, image_size):
    gen_image = functools.partial(
        generate_image_with_sd_webui,
        prompt="astronaut riding a horse",
        agent=agent,
        size=image_size,
        extra={"seed": 123},
    )

    # Generate an image with a negative prompt
    image_path = lst(gen_image(negative_prompt="horse", filename="negative.jpg"))
    with Image.open(image_path) as img:
        neg_image_hash = hashlib.md5(img.tobytes()).hexdigest()

    # Generate an image without a negative prompt
    image_path = lst(gen_image(filename="positive.jpg"))
    with Image.open(image_path) as img:
        image_hash = hashlib.md5(img.tobytes()).hexdigest()

    assert image_hash != neg_image_hash


```

这段代码定义了一个名为 `lst` 的函数，它接受一个文本格式的参数 `txt`。这个函数的作用是提取 `generate_image()` 函数的输出文件路径。

接下来是另一个名为 `generate_and_validate` 的函数，它接受一个名为 `agent` 的代理人对象，一个名为 `workspace` 的虚拟环境，一个表示图像大小的参数 `image_size`，一个表示图像提供商的参数 `image_provider`，以及一个名为 `hugging_face_image_model` 的参数。这个函数的作用是生成图像并在验证其输出后返回。

`generate_and_validate` 函数内部包含多个参数，包括 `agent`、`workspace`、`image_size`、`image_provider` 和 `hugging_face_image_model`。它首先将 `generate_image()` 函数作为参数传递给 `lst()` 函数，并将其输出存储在 `image_path` 变量中。然后，它验证 `image_path` 是否存在，并使用 `Image.open()` 函数打开图像文件。如果 `image_path` 存在，它检查 `img` 对象的大小是否符合期望的图像尺寸。最后，它将图像返回给调用者。


```py
def lst(txt):
    """Extract the file path from the output of `generate_image()`"""
    return Path(txt.split(":", maxsplit=1)[1].strip())


def generate_and_validate(
    agent: Agent,
    workspace,
    image_size,
    image_provider,
    hugging_face_image_model=None,
    **kwargs,
):
    """Generate an image and validate the output."""
    agent.legacy_config.image_provider = image_provider
    agent.legacy_config.huggingface_image_model = hugging_face_image_model
    prompt = "astronaut riding a horse"

    image_path = lst(generate_image(prompt, agent, image_size, **kwargs))
    assert image_path.exists()
    with Image.open(image_path) as img:
        assert img.size == (image_size, image_size)


```

The `generate_image` function is responsible for generating the image based on the prompt passed in. It is called with the `agent` object and an `image_size` parameter.

If the user passes in a bad image, the function should return an error message indicating that the image could not be generated. If the user does not specify an image size and a model, the default model should be used.

If the user specifies a model, the `generate_image` function should use that model to generate the image. If the model is not available, the function should return an error message.

If there is a delay in the `return_text` parameter, the `generate_image` function should wait for that amount of time before trying again to generate the image. If the user does not specify a delay, the function should wait indefinitely.


```py
@pytest.mark.parametrize(
    "return_text",
    [
        '{"error":"Model [model] is currently loading","estimated_time": [delay]}',  # Delay
        '{"error":"Model [model] is currently loading"}',  # No delay
        '{"error:}',  # Bad JSON
        "",  # Bad Image
    ],
)
@pytest.mark.parametrize(
    "image_model",
    ["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1"],
)
@pytest.mark.parametrize("delay", [10, 0])
def test_huggingface_fail_request_with_delay(
    agent: Agent, workspace, image_size, image_model, return_text, delay
):
    return_text = return_text.replace("[model]", image_model).replace(
        "[delay]", str(delay)
    )

    with patch("requests.post") as mock_post:
        if return_text == "":
            # Test bad image
            mock_post.return_value.status_code = 200
            mock_post.return_value.ok = True
            mock_post.return_value.content = b"bad image"
        else:
            # Test delay and bad json
            mock_post.return_value.status_code = 500
            mock_post.return_value.ok = False
            mock_post.return_value.text = return_text

        agent.legacy_config.image_provider = "huggingface"
        agent.legacy_config.huggingface_image_model = image_model
        prompt = "astronaut riding a horse"

        with patch("time.sleep") as mock_sleep:
            # Verify request fails.
            result = generate_image(prompt, agent, image_size)
            assert result == "Error creating image."

            # Verify retry was called with delay if delay is in return_text
            if "estimated_time" in return_text:
                mock_sleep.assert_called_with(delay)
            else:
                mock_sleep.assert_not_called()


```

这段代码是一个函数 `test_huggingface_fail_request_with_delay`，它的作用是测试一个名为 `astronaut riding a horse` 的模型，该模型使用 `huggingface` 图像提供者，使用了 `CompVis/stable-diffusion-v1-4` 模型。

具体来说，这段代码执行以下操作：

1. 将 `Agent` 对象的 `huggingface_api_token` 属性设置为 `"1"`。

2. 使用 `time.sleep` 函数模拟一段时间(在这里是 500 毫秒)，以确保模拟请求到达服务器的延迟。

3. 创建一个名为 `astronaut riding a horse` 的模型，并将其 `huggingface_image_provider` 属性设置为 `"huggingface"`,`huggingface_image_model` 属性设置为 `"CompVis/stable-diffusion-v1-4"`。

4. 调用 `generate_image` 函数，这个函数会创建一个与输入图像描述相符的图像。在这里，模拟生成失败，因为 `huggingface` 服务无法正常工作。

5. 验证 `generate_image` 函数返回的结果是否为 `"Error creating image"`。

6. 验证 `time.sleep` 函数是否被正确调用，以及模拟器是否发出了 500 毫秒的延迟。

由于 `huggingface` 服务不可用，因此模拟将失败。


```py
def test_huggingface_fail_request_with_delay(mocker, agent: Agent):
    agent.legacy_config.huggingface_api_token = "1"

    # Mock requests.post
    mock_post = mocker.patch("requests.post")
    mock_post.return_value.status_code = 500
    mock_post.return_value.ok = False
    mock_post.return_value.text = '{"error":"Model CompVis/stable-diffusion-v1-4 is currently loading","estimated_time":0}'

    # Mock time.sleep
    mock_sleep = mocker.patch("time.sleep")

    agent.legacy_config.image_provider = "huggingface"
    agent.legacy_config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"

    result = generate_image("astronaut riding a horse", agent, 512)

    assert result == "Error creating image."

    # Verify retry was called with delay.
    mock_sleep.assert_called_with(0)


```

这段代码的作用是测试HuggingFace的API请求是否可以成功，以及是否会在请求失败时进行重试。主要角色包括：

1. 使用Mocker模拟器来测试HuggingFace的API请求。
2. 创建一个HuggingFace的Agent对象。
3. 将HuggingFace API的访问令牌更改为"1"。
4. 模拟发送请求并检查请求的状态码是否为500。
5. 模拟等待一段时间并检查是否收到响应。
6. 设置图像提供者，以便在发生错误时使用HuggingFace的图像。
7. 生成图像，并检查返回的值是否为"Error creating image"。
8. 验证retry()方法没有被调用。


```py
def test_huggingface_fail_request_no_delay(mocker, agent: Agent):
    agent.legacy_config.huggingface_api_token = "1"

    # Mock requests.post
    mock_post = mocker.patch("requests.post")
    mock_post.return_value.status_code = 500
    mock_post.return_value.ok = False
    mock_post.return_value.text = (
        '{"error":"Model CompVis/stable-diffusion-v1-4 is currently loading"}'
    )

    # Mock time.sleep
    mock_sleep = mocker.patch("time.sleep")

    agent.legacy_config.image_provider = "huggingface"
    agent.legacy_config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"

    result = generate_image("astronaut riding a horse", agent, 512)

    assert result == "Error creating image."

    # Verify retry was not called.
    mock_sleep.assert_not_called()


```

这段代码是一个函数，名为 `test_huggingface_fail_request_bad_json`，旨在测试 Hugging Face 的 API 是否能够正确地创建模型。

具体来说，这段代码执行以下操作：

1. 将一个名为 `Agent` 的对象的 `huggingface_api_token` 属性设置为字符串 `"1"`，这行代码的作用是模拟一个用户向 Hugging Face API 发送请求并获取一个 JSON 响应的过程，实际上并不会向实际 API 发送请求。
2. 使用 `time.sleep` 函数来模拟一段时间，这行代码的作用是让测试程序等待一段时间，以模拟现实世界中的延迟。
3. 将 `Agent` 对象的 `image_provider` 属性设置为 `"huggingface"`，这行代码的作用是指定生成的图像将使用 Hugging Face 提供的 API 进行生成。
4. 使用 `generate_image` 函数创建一个模拟的图像，该函数接受三个参数：要生成的图像的描述、`Agent` 对象以及图像生成的最大尺寸（以字节为单位）。
5. 调用 `generate_image` 函数并传入参数，得到一个图像对象 `result`。
6. 断言 `result` 的值为 `"Error creating image."`，这表明生成的图像遇到了错误。
7. 模拟 `time.sleep` 函数的执行，这行代码的作用是让测试程序在一段时间内继续等待，以模拟现实世界中的延迟。

综上所述，这段代码的作用是测试 `huggingface` API 是否能够正确地创建模型，并模拟了 API 请求和延迟的情况。


```py
def test_huggingface_fail_request_bad_json(mocker, agent: Agent):
    agent.legacy_config.huggingface_api_token = "1"

    # Mock requests.post
    mock_post = mocker.patch("requests.post")
    mock_post.return_value.status_code = 500
    mock_post.return_value.ok = False
    mock_post.return_value.text = '{"error:}'

    # Mock time.sleep
    mock_sleep = mocker.patch("time.sleep")

    agent.legacy_config.image_provider = "huggingface"
    agent.legacy_config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"

    result = generate_image("astronaut riding a horse", agent, 512)

    assert result == "Error creating image."

    # Verify retry was not called.
    mock_sleep.assert_not_called()


```

这段代码是一个测试用例，用于测试Hugging Face的API请求是否可以正确地创建一个图像。

具体来说，这段代码实现了以下操作：

1. 设置一个名为"agent"的Hugging Face Agent对象。
2. 将"agent.legacy_config.huggingface_api_token"设置为字符串"1"。
3. 使用Python的"requests"库的"post"方法模拟一个HTTP请求，并将模拟请求的URL替换为"https://api.huggingface.co/v2/"。
4. 将"agent.legacy_config.image_provider"设置为"huggingface"。
5. 将"agent.legacy_config.huggingface_image_model"设置为"CompVis/stable-diffusion-v1-4"。
6. 使用Python的"generate_image"函数，并传入一个示例图像的参数，该函数将使用Hugging Face的API创建一个新的图像。
7. 使用"assert"语句，在测试成功后验证结果是否为"Error creating image."，如果成功，则表示函数运行正确。


```py
def test_huggingface_fail_request_bad_image(mocker, agent: Agent):
    agent.legacy_config.huggingface_api_token = "1"

    # Mock requests.post
    mock_post = mocker.patch("requests.post")
    mock_post.return_value.status_code = 200

    agent.legacy_config.image_provider = "huggingface"
    agent.legacy_config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"

    result = generate_image("astronaut riding a horse", agent, 512)

    assert result == "Error creating image."


```

这段代码是一个函数 `test_huggingface_fail_missing_api_token`，用于测试是否能在缺少HuggingFace API token的情况下训练HuggingFace模型。具体来说，该函数的作用是将HuggingFace Image Model指定为CompVis/stable-diffusion-v1-4，并将API token设置为"huggingface"。

具体实现包括以下步骤：

1. 将`agent.legacy_config.image_provider`设置为`"huggingface"`，这意味着将来HuggingFace Image和模型将使用此Image Provider。
2. 使用`mocker.patch`函数来模拟HuggingFace的API接口。该函数的`side_effect`参数指定其可能引发的异常。在本例中，它使用`requests.post`函数发送POST请求，并捕获由HuggingFace引发的`ValueError`异常。
3. 使用`with pytest.raises(ValueError)`语句来验证生成图像时是否引发`ValueError`异常。如果发生，说明HuggingFace API卡住了，也就是无法获得API token。

该函数的作用是测试在缺少HuggingFace API token的情况下，是否可以训练HuggingFace模型并生成图像。


```py
def test_huggingface_fail_missing_api_token(mocker, agent: Agent):
    agent.legacy_config.image_provider = "huggingface"
    agent.legacy_config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"

    # Mock requests.post to raise ValueError
    mock_post = mocker.patch("requests.post", side_effect=ValueError)

    # Verify request raises an error.
    with pytest.raises(ValueError):
        generate_image("astronaut riding a horse", agent, 512)

```
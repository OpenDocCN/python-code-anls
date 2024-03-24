# `.\lucidrains\naturalspeech2-pytorch\naturalspeech2_pytorch\utils\phonemizers\espeak_wrapper.py`

```
""" from https://github.com/coqui-ai/TTS/"""
# 导入所需的模块
import logging
import re
import subprocess
from typing import Dict, List

from packaging.version import Version

from naturalspeech2_pytorch.utils.phonemizers.base import BasePhonemizer
from naturalspeech2_pytorch.utils.phonemizers.punctuation import Punctuation

# 检查系统中是否存在指定的可执行程序
def is_tool(name):
    from shutil import which

    return which(name) is not None

# 使用正则表达式模式匹配 espeak 版本号
espeak_version_pattern = re.compile(r"text-to-speech:\s(?P<version>\d+\.\d+(\.\d+)?)")


# 获取 espeak 版本号
def get_espeak_version():
    output = subprocess.getoutput("espeak --version")
    match = espeak_version_pattern.search(output)

    return match.group("version")

# 获取 espeak-ng 版本号
def get_espeakng_version():
    output = subprocess.getoutput("espeak-ng --version")
    return output.split()[3]

# 优先使用 espeak-ng，其次使用 espeak
if is_tool("espeak-ng"):
    _DEF_ESPEAK_LIB = "espeak-ng"
    _DEF_ESPEAK_VER = get_espeakng_version()
elif is_tool("espeak"):
    _DEF_ESPEAK_LIB = "espeak"
    _DEF_ESPEAK_VER = get_espeak_version()
else:
    _DEF_ESPEAK_LIB = None
    _DEF_ESPEAK_VER = None

# 运行 espeak 命令行工具
def _espeak_exe(espeak_lib: str, args: List, sync=False) -> List[str]:
    """Run espeak with the given arguments."""
    cmd = [
        espeak_lib,
        "-q",
        "-b",
        "1",  # UTF8 text encoding
    ]
    cmd.extend(args)
    logging.debug("espeakng: executing %s", repr(cmd))

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as p:
        res = iter(p.stdout.readline, b"")
        if not sync:
            p.stdout.close()
            if p.stderr:
                p.stderr.close()
            if p.stdin:
                p.stdin.close()
            return res
        res2 = []
        for line in res:
            res2.append(line)
        p.stdout.close()
        if p.stderr:
            p.stderr.close()
        if p.stdin:
            p.stdin.close()
        p.wait()
    return res2

# ESpeak 类，用于调用 espeak 或 espeak-ng 执行 G2P
class ESpeak(BasePhonemizer):
    """ESpeak wrapper calling `espeak` or `espeak-ng` from the command-line the perform G2P

    Args:
        language (str):
            Valid language code for the used backend.

        backend (str):
            Name of the backend library to use. `espeak` or `espeak-ng`. If None, set automatically
            prefering `espeak-ng` over `espeak`. Defaults to None.

        punctuations (str):
            Characters to be treated as punctuation. Defaults to Punctuation.default_puncs().

        keep_puncs (bool):
            If True, keep the punctuations after phonemization. Defaults to True.

    Example:
        >>> phonemizer = ESpeak("tr")
        >>> phonemizer.phonemize("Bu Türkçe, bir örnektir.", separator="|")
        'b|ʊ t|ˈø|r|k|tʃ|ɛ, b|ɪ|r œ|r|n|ˈɛ|c|t|ɪ|r.'

    """

    _ESPEAK_LIB = _DEF_ESPEAK_LIB
    _ESPEAK_VER = _DEF_ESPEAK_VER

    def __init__(self, language: str, backend=None, punctuations=Punctuation.default_puncs(), keep_puncs=True):
        if self._ESPEAK_LIB is None:
            raise Exception(" [!] No espeak backend found. Install espeak-ng or espeak to your system.")
        self.backend = self._ESPEAK_LIB

        # band-aid for backwards compatibility
        if language == "en":
            language = "en-us"
        if language == "zh-cn":
            language = "cmn"

        super().__init__(language, punctuations=punctuations, keep_puncs=keep_puncs)
        if backend is not None:
            self.backend = backend

    @property
    def backend(self):
        return self._ESPEAK_LIB

    @property
    def backend_version(self):
        return self._ESPEAK_VER

    @backend.setter
    # 设置后端引擎
    def backend(self, backend):
        # 检查后端引擎是否为有效值
        if backend not in ["espeak", "espeak-ng"]:
            raise Exception("Unknown backend: %s" % backend)
        # 设置 ESPEAK_LIB 为指定的后端引擎
        self._ESPEAK_LIB = backend
        # 根据后端引擎设置 ESPEAK_VER
        self._ESPEAK_VER = get_espeakng_version() if backend == "espeak-ng" else get_espeak_version()

    # 自动设置 espeak 库
    def auto_set_espeak_lib(self) -> None:
        # 检查是否存在 espeak-ng 工具
        if is_tool("espeak-ng"):
            self._ESPEAK_LIB = "espeak-ng"
            self._ESPEAK_VER = get_espeakng_version()
        # 检查是否存在 espeak 工具
        elif is_tool("espeak"):
            self._ESPEAK_LIB = "espeak"
            self._ESPEAK_VER = get_espeak_version()
        else:
            raise Exception("Cannot set backend automatically. espeak-ng or espeak not found")

    # 返回引擎名称
    @staticmethod
    def name():
        return "espeak"

    # 将输入文本转换为音素
    def phonemize_espeak(self, text: str, separator: str = "|", tie=False) -> str:
        """Convert input text to phonemes.

        Args:
            text (str):
                Text to be converted to phonemes.

            tie (bool, optional) : When True use a '͡' character between
                consecutive characters of a single phoneme. Else separate phoneme
                with '_'. This option requires espeak>=1.49. Default to False.
        """
        # 设置参数
        args = ["-v", f"{self._language}"]
        # 根据 tie 参数选择不同的音素分隔方式
        if tie:
            # 在音素之间使用 '͡'
            if self.backend == "espeak":
                args.append("--ipa=1")
            else:
                args.append("--ipa=3")
        else:
            # 使用 '_' 分隔音素
            if self.backend == "espeak":
                if Version(self.backend_version) >= Version("1.48.15"):
                    args.append("--ipa=1")
                else:
                    args.append("--ipa=3")
            else:
                args.append("--ipa=1")
        if tie:
            args.append("--tie=%s" % tie)

        args.append('"' + text + '"')
        # 计算音素
        phonemes = ""
        for line in _espeak_exe(self._ESPEAK_LIB, args, sync=True):
            logging.debug("line: %s", repr(line))
            ph_decoded = line.decode("utf8").strip()
            # 处理 espeak ��� espeak-ng 返回的文本
            ph_decoded = ph_decoded[:1].replace("_", "") + ph_decoded[1:]
            # 移除 espeak-ng 返回文本中的语言标记
            ph_decoded = re.sub(r"\(.+?\)", "", ph_decoded)
            phonemes += ph_decoded.strip()
        return phonemes.replace("_", separator)

    # 调用 phonemize_espeak 方法，设置 tie 参数为 False
    def _phonemize(self, text, separator=None):
        return self.phonemize_espeak(text, separator, tie=False)

    # 返回支持的语言字典
    @staticmethod
    def supported_languages() -> Dict:
        """Get a dictionary of supported languages.

        Returns:
            Dict: Dictionary of language codes.
        """
        if _DEF_ESPEAK_LIB is None:
            return {}
        args = ["--voices"]
        langs = {}
        count = 0
        for line in _espeak_exe(_DEF_ESPEAK_LIB, args, sync=True):
            line = line.decode("utf8").strip()
            if count > 0:
                cols = line.split()
                lang_code = cols[1]
                lang_name = cols[3]
                langs[lang_code] = lang_name
            logging.debug("line: %s", repr(line))
            count += 1
        return langs
    # 返回当前使用的后端的版本号
    def version(self) -> str:
        """Get the version of the used backend.

        Returns:
            str: Version of the used backend.
        """
        # 定义参数列表，包含获取版本信息的参数
        args = ["--version"]
        # 遍历执行 espeak_exe 函数返回的结果，同步执行
        for line in _espeak_exe(self.backend, args, sync=True):
            # 解码行内容为 UTF-8 格式，去除空格并按空格分割，获取版本号
            version = line.decode("utf8").strip().split()[2]
            # 记录调试信息
            logging.debug("line: %s", repr(line))
            # 返回版本号
            return version

    @classmethod
    # 检查 ESpeak 是否可用，可用返回 True，否则返回 False
    def is_available(cls):
        """Return true if ESpeak is available else false"""
        # 检查是否存在 espeak 或 espeak-ng 工具
        return is_tool("espeak") or is_tool("espeak-ng")
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建一个 ESpeak 对象，指定语言为英语
    e = ESpeak(language="en-us")
    # 打印支持的语言列表
    print(e.supported_languages())
    # 打印 ESpeak 的版本信息
    print(e.version())
    # 打印 ESpeak 对象的语言属性
    print(e.language)
    # 打印 ESpeak 对象的名称
    print(e.name())
    # 打印 ESpeak 对象是否可用
    print(e.is_available())

    # 创建一个 ESpeak 对象，指定语言为英语，不保留标点符号
    e = ESpeak(language="en-us", keep_puncs=False)
    # 打印使用 ESpeak 对象将文本转换为音素的结果，加上反引号
    print("`" + e.phonemize("hello how are you today?") + "`")

    # 创建一个 ESpeak 对象，指定语言为英语，保留标点符号
    e = ESpeak(language="en-us", keep_puncs=True)
    # 打印使用 ESpeak 对象将文本转换为音素的结果，加上反引号
    print("`" + e.phonemize("hello how are you today?") + "`")
```
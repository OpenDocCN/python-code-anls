# `.\DB-GPT-src\dbgpt\util\i18n_utils.py`

```py
"""Internationalization utilities."""

# 导入所需模块和函数
import gettext  # 提供国际化支持
import inspect  # 提供获取调用者模块名的功能
import os  # 提供操作系统相关的功能
from functools import cache  # 提供缓存函数的装饰器
from typing import Callable, Optional  # 引入类型提示

from dbgpt.configs.model_config import LOCALES_DIR, ROOT_PATH  # 导入本地配置文件中的路径信息

# 定义全局变量
_DOMAIN = "dbgpt"  # 翻译域名，用于标识翻译的范围

_DEFAULT_LANGUAGE = os.getenv("LANGUAGE", "en")  # 默认语言从环境变量中获取，如未设置，默认为英语"en"

_LANGUAGE_MAPPING = {
    "zh": "zh_CN",   # 语言映射，将"zh"映射为"zh_CN"
    "zh_CN": "zh_CN",  # 将"zh_CN"映射为"zh_CN"
}


def get_module_name(depth=2):
    """Get the module name of the caller."""
    # 获取调用者模块名
    frame = inspect.currentframe()
    try:
        for _ in range(depth):
            frame = frame.f_back
        module_path = inspect.getmodule(frame).__file__  # 获取模块文件路径
        if module_path.startswith(ROOT_PATH):
            module_path = module_path[len(ROOT_PATH) + 1 :]  # 去掉根路径部分
        module_path = module_path.split("/")[1]  # 获取模块名
        if module_path.endswith(".py"):
            module_path = module_path[:-3]  # 去掉.py后缀
    except Exception:
        module_path = ""
    finally:
        del frame
    return module_path


def set_default_language(language: str):
    """Set the default language globally."""
    global _DEFAULT_LANGUAGE
    _DEFAULT_LANGUAGE = language


@cache
def _get_translator(domain: str, language: str) -> Callable[[str], str]:
    """Retrieve a translator function for the given domain and language."""
    try:
        translation = gettext.translation(domain, LOCALES_DIR, languages=[language])
    except FileNotFoundError:
        translation = gettext.NullTranslations()  # 文件未找到时返回空翻译对象

    return translation.gettext  # 返回翻译函数


def get_translator(language: Optional[str] = None) -> Callable[[str], str]:
    """Return a translator function."""
    
    def translator(message: str) -> str:
        nonlocal language
        if not language:
            language = _DEFAULT_LANGUAGE  # 如果未指定语言，则使用默认语言
        language = _LANGUAGE_MAPPING.get(language, language)  # 根据映射表获取标准语言代码
        module_name = get_module_name(depth=2)  # 获取调用者模块名
        domain = (
            f"{_DOMAIN}_{module_name.replace('.', '_')}" if module_name else _DOMAIN
        )  # 构造翻译域名
        return _get_translator(domain, language)(message)  # 返回翻译后的消息

    return translator  # 返回翻译函数


def _install():
    """Install the translator function globally."""
    import builtins

    builtins.__dict__["_"] = get_translator()  # 将翻译函数作为全局函数'_'


_ = get_translator()  # 设置默认全局翻译函数
```
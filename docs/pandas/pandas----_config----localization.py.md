# `D:\src\scipysrc\pandas\pandas\_config\localization.py`

```
"""
Helpers for configuring locale settings.

Name `localization` is chosen to avoid overlap with builtin `locale` module.
"""

from __future__ import annotations  # 导入未来版本的类型注解支持

from contextlib import contextmanager  # 导入上下文管理器支持
import locale  # 导入locale模块
import platform  # 导入platform模块
import re  # 导入正则表达式模块
import subprocess  # 导入子进程管理模块
from typing import (  # 导入类型提示相关内容
    TYPE_CHECKING,
    cast,
)

from pandas._config.config import options  # 导入Pandas配置选项

if TYPE_CHECKING:
    from collections.abc import Generator  # 导入生成器类型

@contextmanager
def set_locale(
    new_locale: str | tuple[str, str], lc_var: int = locale.LC_ALL
) -> Generator[str | tuple[str, str], None, None]:
    """
    Context manager for temporarily setting a locale.

    Parameters
    ----------
    new_locale : str or tuple
        A string of the form <language_country>.<encoding>. For example to set
        the current locale to US English with a UTF8 encoding, you would pass
        "en_US.UTF-8".
    lc_var : int, default `locale.LC_ALL`
        The category of the locale being set.

    Notes
    -----
    This is useful when you want to run a particular block of code under a
    particular locale, without globally setting the locale. This probably isn't
    thread-safe.
    """
    # getlocale is not always compliant with setlocale, use setlocale. GH#46595
    current_locale = locale.setlocale(lc_var)  # 获取当前的区域设置

    try:
        locale.setlocale(lc_var, new_locale)  # 设置新的区域设置
        normalized_code, normalized_encoding = locale.getlocale()
        if normalized_code is not None and normalized_encoding is not None:
            yield f"{normalized_code}.{normalized_encoding}"  # 生成格式化的区域设置字符串
        else:
            yield new_locale  # 如果无法获取规范化的区域设置，返回原始传入的区域设置
    finally:
        locale.setlocale(lc_var, current_locale)  # 恢复到之前的区域设置


def can_set_locale(lc: str, lc_var: int = locale.LC_ALL) -> bool:
    """
    Check to see if we can set a locale, and subsequently get the locale,
    without raising an Exception.

    Parameters
    ----------
    lc : str
        The locale to attempt to set.
    lc_var : int, default `locale.LC_ALL`
        The category of the locale being set.

    Returns
    -------
    bool
        Whether the passed locale can be set
    """
    try:
        with set_locale(lc, lc_var=lc_var):  # 尝试设置给定的区域设置
            pass
    except (ValueError, locale.Error):
        # horrible name for a Exception subclass
        return False  # 如果设置失败，返回False
    else:
        return True  # 如果设置成功，返回True


def _valid_locales(locales: list[str] | str, normalize: bool) -> list[str]:
    """
    Return a list of normalized locales that do not throw an ``Exception``
    when set.

    Parameters
    ----------
    locales : str
        A string where each locale is separated by a newline.
    normalize : bool
        Whether to call ``locale.normalize`` on each locale.

    Returns
    -------
    valid_locales : list
        A list of valid locales.
    """
    return [
        loc
        for loc in (
            locale.normalize(loc.strip()) if normalize else loc.strip()  # 根据normalize标志进行规范化处理
            for loc in locales
        )
        if can_set_locale(loc)  # 检查是否可以设置该locale
    ]


def get_locales(
    # prefix 是一个字符串或者 None 类型的参数，默认为 None
    prefix: str | None = None,
    # normalize 是一个布尔型参数，默认为 True，表示是否进行标准化处理
    normalize: bool = True,
# 定义一个函数，用于获取系统上所有可用的语言环境列表
def get_available_locales(prefix: str = None, normalize: bool = False) -> list[str]:
    """
    Get all the locales that are available on the system.

    Parameters
    ----------
    prefix : str, optional
        If not ``None`` then return only those locales with the prefix
        provided. For example to get all English language locales (those that
        start with ``"en"``), pass ``prefix="en"``.
    normalize : bool, optional
        Call ``locale.normalize`` on the resulting list of available locales.
        If ``True``, only locales that can be set without throwing an
        ``Exception`` are returned.

    Returns
    -------
    locales : list of strings
        A list of locale strings that can be set with ``locale.setlocale()``.
        For example::

            locale.setlocale(locale.LC_ALL, locale_string)

    On error will return an empty list (no locale available, e.g. Windows)

    """
    # 检查当前平台是否为 Linux 或 Darwin（macOS）
    if platform.system() in ("Linux", "Darwin"):
        # 使用 subprocess 调用命令 "locale -a" 并获取输出结果
        raw_locales = subprocess.check_output(["locale", "-a"])
    else:
        # 对于其他平台（如 Windows），返回空列表，因为没有对应的 "locale -a" 命令
        # 注意：is_platform_windows 在这里会导致循环导入，因此直接返回空列表
        return []

    try:
        # raw_locales 是以 "\n" 分隔的可用语言环境列表，可能包含非可解码部分，因此先拆分再重新组合
        split_raw_locales = raw_locales.split(b"\n")
        out_locales = []
        for x in split_raw_locales:
            try:
                # 尝试将每个元素解码为字符串，并使用 options.display.encoding 指定的编码
                out_locales.append(str(x, encoding=cast(str, options.display.encoding)))
            except UnicodeError:
                # 如果解码失败，可能是某些特殊字符使用了 windows-1252 编码（如 Redhat 7 Linux）
                out_locales.append(str(x, encoding="windows-1252"))

    except TypeError:
        pass

    # 如果 prefix 参数为 None，则返回所有找到的可用语言环境列表（经过验证后的）
    if prefix is None:
        return _valid_locales(out_locales, normalize)

    # 使用正则表达式找出以 prefix 开头的语言环境
    pattern = re.compile(f"{prefix}.*")
    found = pattern.findall("\n".join(out_locales))
    return _valid_locales(found, normalize)
```
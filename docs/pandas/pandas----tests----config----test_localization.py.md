# `D:\src\scipysrc\pandas\pandas\tests\config\test_localization.py`

```
import codecs  # 导入codecs模块，用于字符编码和解码
import locale  # 导入locale模块，用于处理特定地区设置
import os  # 导入os模块，用于与操作系统进行交互

import pytest  # 导入pytest测试框架

from pandas._config.localization import (  # 从pandas._config.localization模块导入以下函数
    can_set_locale,  # 可设置地区设置的函数
    get_locales,  # 获取系统支持的所有地区设置的函数
    set_locale,  # 设置地区设置的函数
)

from pandas.compat import ISMUSL  # 从pandas.compat模块导入ISMUSL变量，用于判断是否为MUSL libc系统

import pandas as pd  # 导入pandas库并简称为pd

_all_locales = get_locales()  # 获取系统支持的所有地区设置，并赋值给_all_locales变量

# 如果没有可用的地区设置，则跳过所有测试
pytestmark = pytest.mark.skipif(not _all_locales, reason="Need locales")

# 如果系统支持的地区设置少于等于1个，则跳过相关测试
_skip_if_only_one_locale = pytest.mark.skipif(
    len(_all_locales) <= 1, reason="Need multiple locales for meaningful test"
)


def _get_current_locale(lc_var: int = locale.LC_ALL) -> str:
    # 获取当前地区设置，参数lc_var指定了要获取的地区设置类型
    # getlocale在某些情况下不完全符合setlocale，使用setlocale。见GH#46595
    return locale.setlocale(lc_var)


@pytest.mark.parametrize("lc_var", (locale.LC_ALL, locale.LC_CTYPE, locale.LC_TIME))
def test_can_set_current_locale(lc_var):
    # 测试能否设置当前地区设置
    before_locale = _get_current_locale(lc_var)
    assert can_set_locale(before_locale, lc_var=lc_var)
    after_locale = _get_current_locale(lc_var)
    assert before_locale == after_locale


@pytest.mark.parametrize("lc_var", (locale.LC_ALL, locale.LC_CTYPE, locale.LC_TIME))
def test_can_set_locale_valid_set(lc_var):
    # 测试能否设置默认地区设置
    before_locale = _get_current_locale(lc_var)
    assert can_set_locale("", lc_var=lc_var)
    after_locale = _get_current_locale(lc_var)
    assert before_locale == after_locale


@pytest.mark.parametrize(
    "lc_var",
    (
        locale.LC_ALL,
        locale.LC_CTYPE,
        pytest.param(
            locale.LC_TIME,
            marks=pytest.mark.skipif(
                ISMUSL, reason="MUSL allows setting invalid LC_TIME."
            ),
        ),
    ),
)
def test_can_set_locale_invalid_set(lc_var):
    # 测试不能设置无效的地区设置
    before_locale = _get_current_locale(lc_var)
    assert not can_set_locale("non-existent_locale", lc_var=lc_var)
    after_locale = _get_current_locale(lc_var)
    assert before_locale == after_locale


@pytest.mark.parametrize(
    "lang,enc",
    [
        ("it_CH", "UTF-8"),
        ("en_US", "ascii"),
        ("zh_CN", "GB2312"),
        ("it_IT", "ISO-8859-1"),
    ],
)
@pytest.mark.parametrize("lc_var", (locale.LC_ALL, locale.LC_CTYPE, locale.LC_TIME))
def test_can_set_locale_no_leak(lang, enc, lc_var):
    # 测试即使返回False，can_set_locale也不会泄漏。见GH#46595
    before_locale = _get_current_locale(lc_var)
    can_set_locale((lang, enc), locale.LC_ALL)
    after_locale = _get_current_locale(lc_var)
    assert before_locale == after_locale


def test_can_set_locale_invalid_get(monkeypatch):
    # 见GH#22129
    # 在某些情况下，可以设置一个无效的地区设置，
    # 但后续的getlocale()会引发ValueError异常。

    def mock_get_locale():
        raise ValueError

    with monkeypatch.context() as m:
        m.setattr(locale, "getlocale", mock_get_locale)
        assert not can_set_locale("")


def test_get_locales_at_least_one():
    # 见GH#9744
    assert len(_all_locales) > 0
# 如果只有一个区域设置时跳过测试，用于测试前缀获取区域设置功能
@_skip_if_only_one_locale
def test_get_locales_prefix():
    # 获取所有区域设置中的第一个区域设置
    first_locale = _all_locales[0]
    # 断言获取具有指定前缀的区域设置数量大于零
    assert len(get_locales(prefix=first_locale[:2])) > 0


# 如果只有一个区域设置时跳过测试，用于测试设置区域设置功能
@_skip_if_only_one_locale
@pytest.mark.parametrize(
    "lang,enc",
    [
        ("it_CH", "UTF-8"),
        ("en_US", "ascii"),
        ("zh_CN", "GB2312"),
        ("it_IT", "ISO-8859-1"),
    ],
)
def test_set_locale(lang, enc):
    # 获取当前区域设置
    before_locale = _get_current_locale()

    # 查找并获取指定编码的编解码器名称
    enc = codecs.lookup(enc).name
    # 创建新的区域设置元组
    new_locale = lang, enc

    # 如果无法设置新的区域设置，则抛出特定错误消息的异常
    if not can_set_locale(new_locale):
        msg = "unsupported locale setting"
        with pytest.raises(locale.Error, match=msg):
            # 在特定条件下设置新的区域设置，并执行测试代码块
            with set_locale(new_locale):
                pass
    else:
        # 在可以设置新的区域设置时，使用新的区域设置并检查其是否被正常化
        with set_locale(new_locale) as normalized_locale:
            new_lang, new_enc = normalized_locale.split(".")
            # 获取指定编码的编解码器名称
            new_enc = codecs.lookup(enc).name

            normalized_locale = new_lang, new_enc
            # 断言正常化后的区域设置与预期的新区域设置相同
            assert normalized_locale == new_locale

    # 退出“with”语句后，应恢复到之前的区域设置状态
    after_locale = _get_current_locale()
    assert before_locale == after_locale


# 测试检测到的编码是否正确
def test_encoding_detected():
    # 获取系统环境变量中的当前区域设置
    system_locale = os.environ.get("LC_ALL")
    # 获取系统区域设置的编码部分，如果不存在则默认为utf-8
    system_encoding = system_locale.split(".")[-1] if system_locale else "utf-8"

    # 断言Pandas显示选项的编码与系统区域设置的编码相同
    assert (
        codecs.lookup(pd.options.display.encoding).name
        == codecs.lookup(system_encoding).name
    )
```
# `.\numpy\numpy\_core\tests\_locales.py`

```
# 导入系统和本地化模块
import sys
import locale

# 导入 pytest 模块
import pytest

# 声明模块的公共接口
__ALL__ = ['CommaDecimalPointLocale']

# 定义函数，用于查找使用逗号作为小数点的本地化设置
def find_comma_decimal_point_locale():
    """See if platform has a decimal point as comma locale.

    Find a locale that uses a comma instead of a period as the
    decimal point.

    Returns
    -------
    old_locale: str
        Locale when the function was called.
    new_locale: {str, None)
        First French locale found, None if none found.

    """
    # 如果操作系统为 Windows
    if sys.platform == 'win32':
        locales = ['FRENCH']
    else:
        locales = ['fr_FR', 'fr_FR.UTF-8', 'fi_FI', 'fi_FI.UTF-8']

    # 获取当前 LC_NUMERIC 的本地化设置
    old_locale = locale.getlocale(locale.LC_NUMERIC)
    new_locale = None
    try:
        # 遍历可能的本地化设置
        for loc in locales:
            try:
                # 尝试设置 LC_NUMERIC 的本地化为当前循环的 loc
                locale.setlocale(locale.LC_NUMERIC, loc)
                new_locale = loc
                break
            except locale.Error:
                pass
    finally:
        # 恢复 LC_NUMERIC 的本地化设置为函数执行前的设置
        locale.setlocale(locale.LC_NUMERIC, locale=old_locale)
    
    # 返回原始的本地化设置和找到的新的本地化设置（如果有）
    return old_locale, new_locale


# 定义一个类，用于设置 LC_NUMERIC 为使用逗号作为小数点的本地化
class CommaDecimalPointLocale:
    """Sets LC_NUMERIC to a locale with comma as decimal point.

    Classes derived from this class have setup and teardown methods that run
    tests with locale.LC_NUMERIC set to a locale where commas (',') are used as
    the decimal point instead of periods ('.'). On exit the locale is restored
    to the initial locale. It also serves as context manager with the same
    effect. If no such locale is available, the test is skipped.

    .. versionadded:: 1.15.0

    """
    # 调用前面定义的函数查找逗号作为小数点的本地化设置
    (cur_locale, tst_locale) = find_comma_decimal_point_locale()

    # 设置测试方法的初始化操作
    def setup_method(self):
        # 如果未找到逗号作为小数点的本地化设置，则跳过测试
        if self.tst_locale is None:
            pytest.skip("No French locale available")
        # 设置 LC_NUMERIC 的本地化为找到的逗号作为小数点的本地化设置
        locale.setlocale(locale.LC_NUMERIC, locale=self.tst_locale)

    # 设置测试方法的清理操作
    def teardown_method(self):
        # 恢复 LC_NUMERIC 的本地化设置为初始化时的设置
        locale.setlocale(locale.LC_NUMERIC, locale=self.cur_locale)

    # 设置类实例作为上下文管理器的进入操作
    def __enter__(self):
        # 如果未找到逗号作为小数点的本地化设置，则跳过测试
        if self.tst_locale is None:
            pytest.skip("No French locale available")
        # 设置 LC_NUMERIC 的本地化为找到的逗号作为小数点的本地化设置
        locale.setlocale(locale.LC_NUMERIC, locale=self.tst_locale)

    # 设置类实例作为上下文管理器的退出操作
    def __exit__(self, type, value, traceback):
        # 恢复 LC_NUMERIC 的本地化设置为初始化时的设置
        locale.setlocale(locale.LC_NUMERIC, locale=self.cur_locale)
```
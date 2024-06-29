# `D:\src\scipysrc\pandas\pandas\io\formats\console.py`

```
"""
Internal module for console introspection
"""

from __future__ import annotations  # 导入未来的注解支持

from shutil import get_terminal_size  # 导入获取终端大小的函数


def get_console_size() -> tuple[int | None, int | None]:
    """
    Return console size as tuple = (width, height).

    Returns (None,None) in non-interactive session.
    """
    from pandas import get_option  # 导入获取选项的函数

    display_width = get_option("display.width")  # 获取显示宽度选项
    display_height = get_option("display.max_rows")  # 获取显示最大行数选项

    # Consider
    # interactive shell terminal, can detect term size
    # interactive non-shell terminal (ipnb/ipqtconsole), cannot detect term
    # size non-interactive script, should disregard term size

    # in addition
    # width,height have default values, but setting to 'None' signals
    # should use Auto-Detection, But only in interactive shell-terminal.
    # Simple. yeah.

    if in_interactive_session():  # 如果运行在交互式会话中
        if in_ipython_frontend():  # 如果运行在IPython前端
            # sane defaults for interactive non-shell terminal
            # match default for width,height in config_init
            from pandas._config.config import get_default_val  # 导入获取默认值的函数

            terminal_width = get_default_val("display.width")  # 获取显示宽度的默认值
            terminal_height = get_default_val("display.max_rows")  # 获取显示最大行数的默认值
        else:
            # pure terminal
            terminal_width, terminal_height = get_terminal_size()  # 获取终端的实际宽度和高度
    else:
        terminal_width, terminal_height = None, None  # 如果不是交互式会话，则宽度和高度设为None

    # Note if the User sets width/Height to None (auto-detection)
    # and we're in a script (non-inter), this will return (None,None)
    # caller needs to deal.
    return display_width or terminal_width, display_height or terminal_height  # 返回显示宽度或终端宽度，以及显示最大行数或终端高度


# ----------------------------------------------------------------------
# Detect our environment


def in_interactive_session() -> bool:
    """
    Check if we're running in an interactive shell.

    Returns
    -------
    bool
        True if running under python/ipython interactive shell.
    """
    from pandas import get_option  # 导入获取选项的函数

    def check_main() -> bool:
        try:
            import __main__ as main  # 尝试导入主模块
        except ModuleNotFoundError:
            return get_option("mode.sim_interactive")  # 如果出现模块未找到错误，返回模拟交互模式选项
        return not hasattr(main, "__file__") or get_option("mode.sim_interactive")  # 如果主模块没有文件属性或模拟交互模式选项为真，则返回真

    try:
        # error: Name '__IPYTHON__' is not defined
        return __IPYTHON__ or check_main()  # type: ignore[name-defined]  # 如果__IPYTHON__已定义或check_main()返回真，则返回真
    except NameError:
        return check_main()  # 如果出现名称错误，返回check_main()的结果


def in_ipython_frontend() -> bool:
    """
    Check if we're inside an IPython zmq frontend.

    Returns
    -------
    bool
    """
    try:
        # error: Name 'get_ipython' is not defined
        ip = get_ipython()  # type: ignore[name-defined]  # 获取IPython实例
        return "zmq" in str(type(ip)).lower()  # 如果IPython实例类型中包含'zmq'，返回真
    except NameError:
        pass  # 如果出现名称错误，直接跳过

    return False  # 默认返回假
```
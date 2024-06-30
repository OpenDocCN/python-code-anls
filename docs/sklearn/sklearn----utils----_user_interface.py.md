# `D:\src\scipysrc\scikit-learn\sklearn\utils\_user_interface.py`

```
import timeit  # 导入用于测量代码执行时间的模块
from contextlib import contextmanager  # 导入上下文管理器模块


def _message_with_time(source, message, time):
    """Create one line message for logging purposes.

    Parameters
    ----------
    source : str
        String indicating the source or the reference of the message.

    message : str
        Short message.

    time : int
        Time in seconds.
    """
    start_message = "[%s] " % source  # 构建消息的起始部分

    # adapted from joblib.logger.short_format_time without the Windows -.1s
    # adjustment
    if time > 60:
        time_str = "%4.1fmin" % (time / 60)  # 如果时间大于60秒，转换为分钟表示
    else:
        time_str = " %5.1fs" % time  # 否则使用秒表示
    end_message = " %s, total=%s" % (message, time_str)  # 构建消息的结束部分
    dots_len = 70 - len(start_message) - len(end_message)  # 计算填充点的长度
    return "%s%s%s" % (start_message, dots_len * ".", end_message)  # 返回完整的消息字符串


@contextmanager
def _print_elapsed_time(source, message=None):
    """Log elapsed time to stdout when the context is exited.

    Parameters
    ----------
    source : str
        String indicating the source or the reference of the message.

    message : str, default=None
        Short message. If None, nothing will be printed.

    Returns
    -------
    context_manager
        Prints elapsed time upon exit if verbose.
    """
    if message is None:
        yield  # 如果消息为空，则直接返回，不执行后续操作
    else:
        start = timeit.default_timer()  # 记录进入上下文管理器时的时间
        yield  # 执行上下文管理器中的代码块
        print(_message_with_time(source, message, timeit.default_timer() - start))  # 计算并打印执行时间
```
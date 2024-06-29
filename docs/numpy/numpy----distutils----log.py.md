# `.\numpy\numpy\distutils\log.py`

```py
# Colored log
# 引入 sys 模块，用于访问系统标准输入输出等功能
import sys
# 从 distutils.log 中导入所有内容，不触发 F403 警告
from distutils.log import *  # noqa: F403
# 从 distutils.log 中导入 Log 类别名为 old_Log
from distutils.log import Log as old_Log
# 从 distutils.log 中导入全局日志对象 _global_log
from distutils.log import _global_log

# 从 numpy.distutils.misc_util 中导入一些函数和变量
from numpy.distutils.misc_util import (red_text, default_text, cyan_text,
        green_text, is_sequence, is_string)


# 定义修复参数的函数 _fix_args，参数 args 可能是字符串或序列，默认 flag 为 1
def _fix_args(args, flag=1):
    # 如果 args 是字符串，则替换 '%' 为 '%%'
    if is_string(args):
        return args.replace('%', '%%')
    # 如果 flag 为真且 args 是序列，则递归修复序列中的每个元素
    if flag and is_sequence(args):
        return tuple([_fix_args(a, flag=0) for a in args])
    return args  # 返回修复后的参数


# 定义 Log 类，继承自 old_Log
class Log(old_Log):
    # 内部方法，用于记录日志消息
    def _log(self, level, msg, args):
        # 如果日志级别大于等于设定的阈值
        if level >= self.threshold:
            # 如果 args 不为空，则格式化消息中的参数
            if args:
                msg = msg % _fix_args(args)
            # 如果 0：（始终为假条件，所以不会执行以下代码块）
            if 0:
                if msg.startswith('copying ') and msg.find(' -> ') != -1:
                    return
                if msg.startswith('byte-compiling '):
                    return
            # 打印经过全局颜色映射处理后的消息
            print(_global_color_map[level](msg))
            sys.stdout.flush()  # 刷新标准输出流

    # 自定义的记录“好”消息的方法
    def good(self, msg, *args):
        """
        If we log WARN messages, log this message as a 'nice' anti-warn
        message.

        """
        # 如果 WARN 级别大于等于设定的阈值
        if WARN >= self.threshold:
            # 如果 args 不为空，则打印经过绿色文本处理后的消息
            if args:
                print(green_text(msg % _fix_args(args)))
            else:
                print(green_text(msg))
            sys.stdout.flush()  # 刷新标准输出流


# 将 _global_log 对象的类设置为 Log，覆盖原有全局日志对象
_global_log.__class__ = Log

# 定义 good 函数，直接调用 _global_log 的 good 方法
good = _global_log.good

# 定义设置日志级别阈值的函数 set_threshold
def set_threshold(level, force=False):
    # 获取当前的日志级别阈值
    prev_level = _global_log.threshold
    # 如果当前级别高于 DEBUG 或者 force 参数为真
    if prev_level > DEBUG or force:
        # 如果当前级别小于等于 DEBUG，则将全局日志对象的阈值设置为指定级别
        _global_log.threshold = level
        if level <= DEBUG:
            # 如果设置级别为 DEBUG，则记录信息到控制台
            info('set_threshold: setting threshold to DEBUG level,'
                    ' it can be changed only with force argument')
    else:
        # 如果当前级别不高于 DEBUG 且 force 参数为假，则记录信息到控制台
        info('set_threshold: not changing threshold from DEBUG level'
                ' %s to %s' % (prev_level, level))
    return prev_level  # 返回之前的日志级别阈值

# 定义获取当前日志级别阈值的函数 get_threshold
def get_threshold():
    return _global_log.threshold  # 返回全局日志对象的当前阈值

# 定义设置日志详细程度的函数 set_verbosity
def set_verbosity(v, force=False):
    # 获取当前的日志级别阈值
    prev_level = _global_log.threshold
    # 如果 v 小于 0，则设置阈值为 ERROR 级别
    if v < 0:
        set_threshold(ERROR, force)
    # 如果 v 等于 0，则设置阈值为 WARN 级别
    elif v == 0:
        set_threshold(WARN, force)
    # 如果 v 等于 1，则设置阈值为 INFO 级别
    elif v == 1:
        set_threshold(INFO, force)
    # 如果 v 大于等于 2，则设置阈值为 DEBUG 级别
    elif v >= 2:
        set_threshold(DEBUG, force)
    # 返回之前的日志级别对应的数值
    return {FATAL:-2,ERROR:-1,WARN:0,INFO:1,DEBUG:2}.get(prev_level, 1)

# 定义全局颜色映射字典，将不同级别的日志映射到对应的颜色处理函数
_global_color_map = {
    DEBUG: cyan_text,
    INFO: default_text,
    WARN: red_text,
    ERROR: red_text,
    FATAL: red_text
}

# 设置日志详细程度为 WARN 级别，强制更新
set_verbosity(0, force=True)

# 将 _error、_warn、_info、_debug 四个函数赋值给新的变量名
_error = error
_warn = warn
_info = info
_debug = debug

# 定义 error 函数，对原 _error 函数进行封装，添加前缀 "ERROR: "
def error(msg, *a, **kw):
    _error(f"ERROR: {msg}", *a, **kw)

# 定义 warn 函数，对原 _warn 函数进行封装，添加前缀 "WARN: "
def warn(msg, *a, **kw):
    _warn(f"WARN: {msg}", *a, **kw)

# 定义 info 函数，对原 _info 函数进行封装，添加前缀 "INFO: "
def info(msg, *a, **kw):
    _info(f"INFO: {msg}", *a, **kw)

# 定义 debug 函数，对原 _debug 函数进行封装，添加前缀 "DEBUG: "
def debug(msg, *a, **kw):
    _debug(f"DEBUG: {msg}", *a, **kw)
```
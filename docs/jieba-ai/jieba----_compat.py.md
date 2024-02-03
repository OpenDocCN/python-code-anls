# `jieba\jieba\_compat.py`

```py
# 设置文件编码为 utf-8
# 导入 logging 模块
# 导入 os 模块
# 导入 sys 模块
import logging
import os
import sys

# 创建一个输出到标准错误流的日志处理器
log_console = logging.StreamHandler(sys.stderr)
# 创建一个默认的 logger 对象
default_logger = logging.getLogger(__name__)
# 设置默认 logger 的日志级别为 DEBUG
default_logger.setLevel(logging.DEBUG)

# 定义一个函数，用于设置日志级别
def setLogLevel(log_level):
    default_logger.setLevel(log_level)

# 检查是否安装了 PaddlePaddle 的标志
check_paddle_install = {'is_paddle_installed': False}

# 尝试导入 pkg_resources 模块，如果导入失败则使用 lambda 函数获取模块资源
try:
    import pkg_resources

    get_module_res = lambda *res: pkg_resources.resource_stream(__name__,
                                                                os.path.join(*res))
except ImportError:
    get_module_res = lambda *res: open(os.path.normpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__), *res)), 'rb')

# 定义一个函数，用于启用 PaddlePaddle
def enable_paddle():
    try:
        import paddle
    except ImportError:
        default_logger.debug("Installing paddle-tiny, please wait a minute......")
        os.system("pip install paddlepaddle-tiny")
        try:
            import paddle
        except ImportError:
            default_logger.debug(
                "Import paddle error, please use command to install: pip install paddlepaddle-tiny==1.6.1."
                "Now, back to jieba basic cut......")
    if paddle.__version__ < '1.6.1':
        default_logger.debug("Find your own paddle version doesn't satisfy the minimum requirement (1.6.1), "
                             "please install paddle tiny by 'pip install --upgrade paddlepaddle-tiny', "
                             "or upgrade paddle full version by "
                             "'pip install --upgrade paddlepaddle (-gpu for GPU version)' ")
    else:
        try:
            import jieba.lac_small.predict as predict
            default_logger.debug("Paddle enabled successfully......")
            check_paddle_install['is_paddle_installed'] = True
        except ImportError:
            default_logger.debug("Import error, cannot find paddle.fluid and jieba.lac_small.predict module. "
                                 "Now, back to jieba basic cut......")

# 检查 Python 版本是否为 2.x
PY2 = sys.version_info[0] == 2
# 获取系统默认的文件系统编码
default_encoding = sys.getfilesystemencoding()

# 如果是 Python 2 版本
if PY2:
    # 定义文本类型为 unicode
    text_type = unicode
    # 定义字符串类型为 str 和 unicode
    string_types = (str, unicode)

    # 定义获取字典键的迭代器函数
    iterkeys = lambda d: d.iterkeys()
    # 定义获取字典值的迭代器函数
    itervalues = lambda d: d.itervalues()
    # 定义获取字典键值对的迭代器函数
    iteritems = lambda d: d.iteritems()

# 如果是 Python 3 版本
else:
    # 定义文本类型为 str
    text_type = str
    # 定义字符串类型为 str
    string_types = (str,)
    # 定义 xrange 函数为 range 函数

    xrange = range

    # 定义获取字典键的迭代器函数
    iterkeys = lambda d: iter(d.keys())
    # 定义获取字典值的迭代器函数
    itervalues = lambda d: iter(d.values())
    # 定义获取字典键值对的迭代器函数
    iteritems = lambda d: iter(d.items())

# 解码字符串为文本类型
def strdecode(sentence):
    # 如果输入不是文本类型
    if not isinstance(sentence, text_type):
        # 尝试使用 utf-8 解码
        try:
            sentence = sentence.decode('utf-8')
        # 如果解码失败，尝试使用 gbk 解码，忽略错误
        except UnicodeDecodeError:
            sentence = sentence.decode('gbk', 'ignore')
    return sentence

# 解析文件名
def resolve_filename(f):
    # 尝试获取文件名属性
    try:
        return f.name
    # 如果获取失败，返回文件的字符串表示形式
    except AttributeError:
        return repr(f)
```
# `ZeroNet\plugins\Sidebar\ConsolePlugin.py`

```py
# 导入 re 模块，用于正则表达式操作
import re
# 导入 logging 模块，用于记录日志
import logging

# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 Config 模块中导入 config 变量
from Config import config
# 从 Debug 模块中导入 Debug 类
from Debug import Debug
# 从 util 模块中导入 SafeRe 模块
from util import SafeRe
# 从 util.Flag 模块中导入 flag 变量
from util.Flag import flag

# 创建 WsLogStreamer 类，继承自 logging.StreamHandler 类
class WsLogStreamer(logging.StreamHandler):
    # 初始化方法，接受 stream_id, ui_websocket, filter 三个参数
    def __init__(self, stream_id, ui_websocket, filter):
        # 设置 stream_id 属性为传入的 stream_id
        self.stream_id = stream_id
        # 设置 ui_websocket 属性为传入的 ui_websocket
        self.ui_websocket = ui_websocket

        # 如果 filter 存在
        if filter:
            # 如果 filter 不是安全的正则表达式模式
            if not SafeRe.isSafePattern(filter):
                # 抛出异常
                raise Exception("Not a safe prex pattern")
            # 使用 filter 构建正则表达式对象，赋值给 filter_re 属性
            self.filter_re = re.compile(".*" + filter)
        else:
            # 如果 filter 不存在，将 filter_re 属性设置为 None
            self.filter_re = None
        # 调用父类的初始化方法
        return super(WsLogStreamer, self).__init__()

    # 发送日志记录的方法，接受 record 参数
    def emit(self, record):
        # 如果 ui_websocket 的 ws 属性已关闭
        if self.ui_websocket.ws.closed:
            # 停止发送日志
            self.stop()
            return
        # 格式化日志记录，赋值给 line 变量
        line = self.format(record)
        # 如果 filter_re 存在且不匹配 line
        if self.filter_re and not self.filter_re.match(line):
            # 返回 False
            return False

        # 调用 ui_websocket 的 cmd 方法，发送日志行
        self.ui_websocket.cmd("logLineAdd", {"stream_id": self.stream_id, "lines": [line]})

    # 停止发送日志的方法
    def stop(self):
        # 从根记录器中移除日志处理器
        logging.getLogger('').removeHandler(self)

# 使用 PluginManager 注册到 "UiWebsocket" 的 UiWebsocketPlugin 类
@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    # 初始化方法，接受任意数量的位置参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 初始化 log_streamers 属性为空字典
        self.log_streamers = {}
        # 调用父类的初始化方法
        return super(UiWebsocketPlugin, self).__init__(*args, **kwargs)

    # 标记为不允许多用户访问的装饰器
    @flag.no_multiuser
    # 标记为管理员权限的装饰器
    @flag.admin
    # 从控制台日志文件中读取内容并返回
    def actionConsoleLogRead(self, to, filter=None, read_size=32 * 1024, limit=500):
        # 获取日志文件路径
        log_file_path = "%s/debug.log" % config.log_dir
        # 打开日志文件
        log_file = open(log_file_path, encoding="utf-8")
        # 定位到文件末尾
        log_file.seek(0, 2)
        end_pos = log_file.tell()
        # 定位到最后 read_size 大小的位置
        log_file.seek(max(0, end_pos - read_size))
        # 如果不在文件开头，则跳过部分行
        if log_file.tell() != 0:
            log_file.readline()  # Partial line junk

        pos_start = log_file.tell()
        lines = []
        # 如果有过滤条件，则编译正则表达式
        if filter:
            assert SafeRe.isSafePattern(filter)
            filter_re = re.compile(".*" + filter)

        last_match = False
        # 遍历日志文件的每一行
        for line in log_file:
            # 如果不是以 "[" 开头且上一行匹配成功，则为多行日志条目
            if not line.startswith("[") and last_match:  # Multi-line log entry
                lines.append(line.replace(" ", "&nbsp;"))
                continue

            # 如果有过滤条件且不匹配，则跳过
            if filter and not filter_re.match(line):
                last_match = False
                continue
            last_match = True
            lines.append(line)

        num_found = len(lines)
        # 只保留最后的 limit 行日志
        lines = lines[-limit:]

        return {"lines": lines, "pos_end": log_file.tell(), "pos_start": pos_start, "num_found": num_found}

    # 添加日志流处理器
    def addLogStreamer(self, stream_id, filter=None):
        logger = WsLogStreamer(stream_id, self, filter)
        logger.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)-8s %(name)s %(message)s'))
        logger.setLevel(logging.getLevelName("DEBUG"))

        logging.getLogger('').addHandler(logger)
        return logger

    # 控制台日志流处理
    @flag.no_multiuser
    @flag.admin
    def actionConsoleLogStream(self, to, filter=None):
        stream_id = to
        # 添加日志流处理器
        self.log_streamers[stream_id] = self.addLogStreamer(stream_id, filter)
        self.response(to, {"stream_id": stream_id})

    # 控制台日志流处理
    @flag.no_multiuser
    @flag.admin
    # 定义一个方法，用于从控制台日志流中移除指定的日志流
    def actionConsoleLogStreamRemove(self, to, stream_id):
        # 尝试停止指定日志流的记录
        try:
            self.log_streamers[stream_id].stop()
            # 从记录日志流的字典中删除指定的日志流
            del self.log_streamers[stream_id]
            # 返回操作成功的消息
            return "ok"
        # 如果发生异常，则返回异常信息
        except Exception as err:
            return {"error": Debug.formatException(err)}
```
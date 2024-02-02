# `ZeroNet\src\Config.py`

```py
# 导入必要的模块
import argparse  # 用于解析命令行参数
import sys  # 用于与 Python 解释器交互
import os  # 提供了许多与操作系统交互的函数
import locale  # 用于处理与特定地域文化相关的数据
import re  # 用于支持正则表达式操作
import configparser  # 用于读取和写入配置文件
import logging  # 用于记录日志信息
import logging.handlers  # 提供了不同类型的日志处理器
import stat  # 提供了解释和操作文件状态的方法
import time  # 提供了各种与时间相关的函数

# 创建 Config 类
class Config(object):

    # 初始化方法
    def __init__(self, argv):
        # 初始化版本和修订号
        self.version = "0.7.2"
        self.rev = 4555
        self.argv = argv
        self.action = None
        self.test_parser = None
        self.pending_changes = {}
        self.need_restart = False
        # 允许更改的 API 键集合
        self.keys_api_change_allowed = set([
            "tor", "fileserver_port", "language", "tor_use_bridges", "trackers_proxy", "trackers",
            "trackers_file", "open_browser", "log_level", "fileserver_ip_type", "ip_external", "offline",
            "threads_fs_read", "threads_fs_write", "threads_crypt", "threads_db"
        ])
        # 需要重启的键集合
        self.keys_restart_need = set([
            "tor", "fileserver_port", "fileserver_ip_type", "threads_fs_read", "threads_fs_write", "threads_crypt", "threads_db"
        ])
        # 获取启动目录
        self.start_dir = self.getStartDir()

        # 配置文件路径
        self.config_file = self.start_dir + "/zeronet.conf"
        # 数据目录路径
        self.data_dir = self.start_dir + "/data"
        # 日志目录路径
        self.log_dir = self.start_dir + "/log"
        self.openssl_lib_file = None
        self.openssl_bin_file = None

        self.trackers_file = False
        # 创建解析器
        self.createParser()
        # 创建参数
        self.createArguments()

    # 创建解析器
    def createParser(self):
        # 创建参数解析器
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # 注册自定义类型
        self.parser.register('type', 'bool', self.strToBool)
        # 添加子命令解析器
        self.subparsers = self.parser.add_subparsers(title="Action to perform", dest="action")

    # 转换字符串为布尔值
    def strToBool(self, v):
        return v.lower() in ("yes", "true", "t", "1")
    # 获取当前文件的绝对路径，并将反斜杠替换为斜杠，去掉末尾的 "cd"
    this_file = os.path.abspath(__file__).replace("\\", "/").rstrip("cd")

    # 如果命令行参数中包含 "--start_dir"，则将 start_dir 设置为该参数的下一个值
    if "--start_dir" in self.argv:
        start_dir = self.argv[self.argv.index("--start_dir") + 1]
    # 如果当前文件路径以 "/Contents/Resources/core/src/Config.py" 结尾
    elif this_file.endswith("/Contents/Resources/core/src/Config.py"):
        # 作为 ZeroNet.app 运行
        if this_file.startswith("/Application") or this_file.startswith("/private") or this_file.startswith(os.path.expanduser("~/Library")):
            # 从不可写入的目录运行，将数据放到 Application Support 中
            start_dir = os.path.expanduser("~/Library/Application Support/ZeroNet")
        else:
            # 从可写入的目录运行，将数据放到 .app 旁边
            start_dir = re.sub("/[^/]+/Contents/Resources/core/src/Config.py", "", this_file)
    # 如果当前文件路径以 "/core/src/Config.py" 结尾
    elif this_file.endswith("/core/src/Config.py"):
        # 作为可执行文件运行或者源代码在 Application Support 目录中，将 var 文件放到 core 目录之外
        start_dir = this_file.replace("/core/src/Config.py", "")
    # 如果当前文件路径以 "usr/share/zeronet/src/Config.py" 结尾
    elif this_file.endswith("usr/share/zeronet/src/Config.py"):
        # 从不可写入的位置运行，例如 AppImage
        start_dir = os.path.expanduser("~/ZeroNet")
    else:
        # 其他情况下将 start_dir 设置为当前目录
        start_dir = "."

    # 返回 start_dir
    return start_dir

# 创建命令行参数
    # 加载跟踪器文件内容
    def loadTrackersFile(self):
        # 如果没有跟踪器文件，则返回空
        if not self.trackers_file:
            return None

        # 复制跟踪器文件列表
        self.trackers = self.arguments.trackers[:]

        # 遍历跟踪器文件列表
        for trackers_file in self.trackers_file:
            try:
                # 如果跟踪器文件路径是绝对路径
                if trackers_file.startswith("/"):  # Absolute
                    trackers_file_path = trackers_file
                # 如果跟踪器文件路径是相对于 data_dir 的路径
                elif trackers_file.startswith("{data_dir}"):  # Relative to data_dir
                    trackers_file_path = trackers_file.replace("{data_dir}", self.data_dir)
                # 如果跟踪器文件路径是相对于 zeronet.py 的路径
                else:  # Relative to zeronet.py
                    trackers_file_path = self.start_dir + "/" + trackers_file

                # 遍历跟踪器文件内容
                for line in open(trackers_file_path):
                    tracker = line.strip()
                    # 如果跟踪器包含协议并且不在跟踪器列表中，则添加到跟踪器列表中
                    if "://" in tracker and tracker not in self.trackers:
                        self.trackers.append(tracker)
            except Exception as err:
                # 打印加载跟踪器文件时的错误信息
                print("Error loading trackers file: %s" % err)

    # 查找当前操作指定的参数
    def getActionArguments(self):
        back = {}
        # 获取当前操作的参数列表
        arguments = self.parser._subparsers._group_actions[0].choices[self.action]._actions[1:]  # First is --version
        # 遍历参数列表，将参数名和对应的值添加到字典中
        for argument in arguments:
            back[argument.dest] = getattr(self, argument.dest)
        return back

    # 尝试从 argv 中找到操作
    def getAction(self, argv):
        # 获取有效的操作列表
        actions = [list(action.choices.keys()) for action in self.parser._actions if action.dest == "action"][0]  # Valid actions
        found_action = False
        # 遍历操作列表，查看是否在 argv 中找到操作
        for action in actions:
            if action in argv:
                found_action = action
                break
        return found_action

    # 将插件参数移动到参数列表的末尾
    # 将未知参数移动到参数列表的末尾
    def moveUnknownToEnd(self, argv, default_action):
        # 获取所有有效的操作选项
        valid_actions = sum([action.option_strings for action in self.parser._actions], [])
        valid_parameters = []  # 有效的参数列表
        plugin_parameters = []  # 插件参数列表
        plugin = False  # 是否为插件
        for arg in argv:  # 遍历参数列表
            if arg.startswith("--"):  # 如果参数以"--"开头
                if arg not in valid_actions:  # 如果参数不在有效操作选项中
                    plugin = True  # 将插件标记为True
                else:
                    plugin = False  # 否则将插件标记为False
            elif arg == default_action:  # 如果参数等于默认操作
                plugin = False  # 将插件标记为False

            if plugin:  # 如果是插件
                plugin_parameters.append(arg)  # 将参数添加到插件参数列表中
            else:
                valid_parameters.append(arg)  # 否则将参数添加到有效参数列表中
        return valid_parameters + plugin_parameters  # 返回有效参数列表和插件参数列表的组合

    # 获取解析器
    def getParser(self, argv):
        action = self.getAction(argv)  # 获取操作
        if not action:  # 如果没有操作
            return self.parser  # 返回主解析器
        else:
            return self.subparsers.choices[action]  # 返回子解析器中对应操作的解析器

    # 从配置文件和命令行解析参数
    # 解析函数，用于解析命令行参数
    def parse(self, silent=False, parse_config=True):
        # 复制命令行参数
        argv = self.argv[:]  # Copy command line arguments
        # 获取当前解析器
        current_parser = self.getParser(argv)
        # 如果 silent 为 True，则不显示消息或在未知参数时退出
        if silent:  # Don't display messages or quit on unknown parameter
            # 保存原始的打印消息和退出函数
            original_print_message = self.parser._print_message
            original_exit = self.parser.exit

            # 定义一个函数用于静默处理消息和退出
            def silencer(parser, function_name):
                parser.exited = True
                return None
            current_parser.exited = False
            # 重写当前解析器的打印消息函数
            current_parser._print_message = lambda *args, **kwargs: silencer(current_parser, "_print_message")
            # 重写当前解析器的退出函数
            current_parser.exit = lambda *args, **kwargs: silencer(current_parser, "exit")

        # 解析命令行参数
        self.parseCommandline(argv, silent)  # Parse argv
        # 设置属性
        self.setAttributes()
        # 如果需要解析配置文件
        if parse_config:
            # 从配置文件中添加参数
            argv = self.parseConfig(argv)  # Add arguments from config file

        # 再次解析命令行参数
        self.parseCommandline(argv, silent)  # Parse argv
        # 再次设置属性
        self.setAttributes()

        # 如果不是静默模式
        if not silent:
            # 如果文件服务器 IP 不是 "*" 并且不在本地 IP 列表中，则添加到本地 IP 列表中
            if self.fileserver_ip != "*" and self.fileserver_ip not in self.ip_local:
                self.ip_local.append(self.fileserver_ip)

        # 如果是静默模式，则恢复原始函数
        if silent:  # Restore original functions
            # 如果当前解析器已退出并且动作是 "main"，则不启动主要动作
            if current_parser.exited and self.action == "main":  # Argument parsing halted, don't start ZeroNet with main action
                self.action = None
            # 恢复原始的打印消息和退出函数
            current_parser._print_message = original_print_message
            current_parser.exit = original_exit

        # 加载跟踪器文件
        self.loadTrackersFile()

    # 解析命令行参数
    # 解析命令行参数
    def parseCommandline(self, argv, silent=False):
        # 查找命令行参数中是否指定了动作
        action = self.getAction(argv)
        # 如果没有指定动作，则添加默认动作
        if not action:
            argv.append("--end")
            argv.append("main")
            action = "main"
        # 将未知参数移到参数列表末尾
        argv = self.moveUnknownToEnd(argv, action)
        # 如果静默模式，则使用 parse_known_args 方法解析参数
        if silent:
            res = self.parser.parse_known_args(argv[1:])
            if res:
                self.arguments = res[0]
            else:
                self.arguments = {}
        # 否则使用 parse_args 方法解析参数
        else:
            self.arguments = self.parser.parse_args(argv[1:])

    # 解析配置文件
    def parseConfig(self, argv):
        # 从参数中找到配置文件路径
        if "--config_file" in argv:
            self.config_file = argv[argv.index("--config_file") + 1]
        # 加载配置文件
        if os.path.isfile(self.config_file):
            config = configparser.RawConfigParser(allow_no_value=True, strict=False)
            config.read(self.config_file)
            # 遍历配置文件中的各个部分和键值对
            for section in config.sections():
                for key, val in config.items(section):
                    # 如果值为"True"，则将其设为 None
                    if val == "True":
                        val = None
                    # 如果不是全局配置，则使用部分名作为前缀
                    if section != "global":
                        key = section + "_" + key
                    # 如果键为"open_browser"，则优先使用配置文件中的值
                    if key == "open_browser":
                        while "--%s" % key in argv:
                            pos = argv.index("--open_browser")
                            del argv[pos:pos + 2]
                    # 扩展参数列表
                    argv_extend = ["--%s" % key]
                    if val:
                        for line in val.strip().split("\n"):  # 允许多行值
                            argv_extend.append(line)
                        if "\n" in val:
                            argv_extend.append("--end")
                    argv = argv[:1] + argv_extend + argv[1:]
        return argv

    # 返回给定参数的命令行值
    # 从命令行参数中获取指定键的值
    def getCmdlineValue(self, key):
        # 如果键不在命令行参数中，则返回 None
        if key not in self.argv:
            return None
        # 获取键在参数列表中的索引
        argv_index = self.argv.index(key)
        # 如果索引是参数列表的最后一个位置，表示未指定测试参数，返回 None
        if argv_index == len(self.argv) - 1:  # last arg, test not specified
            return None
        # 返回键对应的值
        return self.argv[argv_index + 1]

    # 将参数暴露为类属性
    def setAttributes(self):
        # 从参数中设置类属性
        if self.arguments:
            args = vars(self.arguments)
            for key, val in args.items():
                # 如果值的类型是列表，创建一个副本
                if type(val) is list:
                    val = val[:]
                # 如果键是指定的几个特定键，且值不为空，则将反斜杠替换为斜杠
                if key in ("data_dir", "log_dir", "start_dir", "openssl_bin_file", "openssl_lib_file"):
                    if val:
                        val = val.replace("\\", "/")
                # 设置类属性
                setattr(self, key, val)

    # 加载插件
    def loadPlugins(self):
        # 导入插件管理器
        from Plugin import PluginManager

        # 使用插件管理器接受插件
        @PluginManager.acceptPlugins
        class ConfigPlugin(object):
            # 初始化方法，接受配置对象作为参数
            def __init__(self, config):
                self.argv = config.argv
                self.parser = config.parser
                self.subparsers = config.subparsers
                self.test_parser = config.test_parser
                self.getCmdlineValue = config.getCmdlineValue
                self.createArguments()

            # 创建参数的方法，暂时为空
            def createArguments(self):
                pass

        # 实例化配置插件
        ConfigPlugin(self)
    # 保存键值对到配置文件中
    def saveValue(self, key, value):
        # 如果配置文件不存在，则内容为空字符串
        if not os.path.isfile(self.config_file):
            content = ""
        else:
            # 读取配置文件内容
            content = open(self.config_file).read()
        # 将内容按行分割
        lines = content.splitlines()

        global_line_i = None
        key_line_i = None
        i = 0
        # 遍历每一行
        for line in lines:
            # 如果当前行是 [global]，记录下标
            if line.strip() == "[global]":
                global_line_i = i
            # 如果当前行以键开头或者等于键，记录下标
            if line.startswith(key + " =") or line == key:
                key_line_i = i
            i += 1

        # 如果存在键的行并且配置文件行数大于键的行下标加1
        if key_line_i and len(lines) > key_line_i + 1:
            # 删除之前的多行值
            while True:
                is_value_line = lines[key_line_i + 1].startswith(" ") or lines[key_line_i + 1].startswith("\t")
                if not is_value_line:
                    break
                del lines[key_line_i + 1]

        # 如果值为 None，删除该行
        if value is None:
            if key_line_i:
                del lines[key_line_i]

        else:  # 添加/更新
            # 如果值的类型是列表
            if type(value) is list:
                # 将值转换为字符串列表
                value_lines = [""] + [str(line).replace("\n", "").replace("\r", "") for line in value]
            else:
                # 将值转换为字符串列表
                value_lines = [str(value).replace("\n", "").replace("\r", "")]
            # 创建新的行
            new_line = "%s = %s" % (key, "\n ".join(value_lines))
            # 如果键的行存在，更新该行
            if key_line_i:
                lines[key_line_i] = new_line
            # 如果没有全局部分，将新行追加到文件末尾
            elif global_line_i is None:
                lines.append("[global]")
                lines.append(new_line)
            else:  # 如果有全局部分，将新行追加到全局部分后面
                lines.insert(global_line_i + 1, new_line)

        # 将修改后的内容写入配置文件
        open(self.config_file, "w").write("\n".join(lines))
    # 获取服务器信息，包括平台、文件服务器 IP 和端口、UI IP 和端口、版本号、语言、调试模式、插件列表、日志目录、数据目录、源代码目录等
    def getServerInfo(self):
        # 导入插件管理器和主模块
        from Plugin import PluginManager
        import main

        # 构建服务器信息字典
        info = {
            "platform": sys.platform,
            "fileserver_ip": self.fileserver_ip,
            "fileserver_port": self.fileserver_port,
            "ui_ip": self.ui_ip,
            "ui_port": self.ui_port,
            "version": self.version,
            "rev": self.rev,
            "language": self.language,
            "debug": self.debug,
            "plugins": PluginManager.plugin_manager.plugin_names,

            "log_dir": os.path.abspath(self.log_dir),
            "data_dir": os.path.abspath(self.data_dir),
            "src_dir": os.path.dirname(os.path.abspath(__file__))
        }

        # 尝试获取外部 IP 地址和 Tor 状态，如果失败则忽略异常
        try:
            info["ip_external"] = main.file_server.port_opened
            info["tor_enabled"] = main.file_server.tor_manager.enabled
            info["tor_status"] = main.file_server.tor_manager.status
        except Exception:
            pass

        # 返回服务器信息字典
        return info

    # 初始化控制台日志记录器
    def initConsoleLogger(self):
        # 根据操作类型设置日志格式
        if self.action == "main":
            format = '[%(asctime)s] %(name)s %(message)s'
        else:
            format = '%(name)s %(message)s'

        # 根据配置设置控制台日志级别
        if self.console_log_level == "default":
            if self.silent:
                level = logging.ERROR
            elif self.debug:
                level = logging.DEBUG
            else:
                level = logging.INFO
        else:
            level = logging.getLevelName(self.console_log_level)

        # 创建控制台日志处理器，设置格式和级别，并添加到根日志记录器
        console_logger = logging.StreamHandler()
        console_logger.setFormatter(logging.Formatter(format, "%H:%M:%S"))
        console_logger.setLevel(level)
        logging.getLogger('').addHandler(console_logger)
    # 初始化文件日志记录器
    def initFileLogger(self):
        # 如果动作是主要的，则日志文件路径为log_dir/debug.log
        if self.action == "main":
            log_file_path = "%s/debug.log" % self.log_dir
        # 否则，日志文件路径为log_dir/cmd.log
        else:
            log_file_path = "%s/cmd.log" % self.log_dir

        # 如果日志轮转关闭
        if self.log_rotate == "off":
            # 创建一个文件处理器，以覆盖写模式打开日志文件，编码为utf-8
            file_logger = logging.FileHandler(log_file_path, "w", "utf-8")
        # 否则
        else:
            # 定义时间间隔名称映射
            when_names = {"weekly": "w", "daily": "d", "hourly": "h"}
            # 创建一个定时轮转文件处理器，设置轮转周期、备份文件数、编码等参数
            file_logger = logging.handlers.TimedRotatingFileHandler(
                log_file_path, when=when_names[self.log_rotate], interval=1, backupCount=self.log_rotate_backup_count,
                encoding="utf8"
            )

            # 如果日志文件路径存在
            if os.path.isfile(log_file_path):
                file_logger.doRollover()  # 总是从空日志文件开始
        # 设置日志格式
        file_logger.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)-8s %(name)s %(message)s'))
        # 设置文件处理器的日志级别
        file_logger.setLevel(logging.getLevelName(self.log_level))
        # 设置根记录器的日志级别
        logging.getLogger('').setLevel(logging.getLevelName(self.log_level))
        # 将文件处理器添加到根记录器
        logging.getLogger('').addHandler(file_logger)
    # 初始化日志记录，设置控制台日志和文件日志的默认值
    def initLogging(self, console_logging=None, file_logging=None):
        # 如果未指定控制台日志级别，则根据实例属性判断是否开启控制台日志
        if console_logging == None:
            console_logging = self.console_log_level != "off"

        # 如果未指定文件日志级别，则根据实例属性判断是否开启文件日志
        if file_logging == None:
            file_logging = self.log_level != "off"

        # 创建必要的文件和目录
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
            try:
                os.chmod(self.log_dir, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            except Exception as err:
                print("Can't change permission of %s: %s" % (self.log_dir, err))

        # 隐藏警告信息不在控制台显示
        logging.WARNING = 15  # 如果不是调试模式，则不显示警告
        logging.addLevelName(15, "WARNING")

        # 移除根前缀
        logging.getLogger('').name = "-"  # 移除根前缀

        # 初始化错误日志处理器
        self.error_logger = ErrorLogHandler()
        self.error_logger.setLevel(logging.getLevelName("ERROR"))
        logging.getLogger('').addHandler(self.error_logger)

        # 如果开启控制台日志，则初始化控制台日志
        if console_logging:
            self.initConsoleLogger()
        # 如果开启文件日志，则初始化文件日志
        if file_logging:
            self.initFileLogger()
# 创建一个自定义的日志处理器类，继承自 logging.StreamHandler
class ErrorLogHandler(logging.StreamHandler):
    # 初始化方法，初始化一个空列表用于存储日志记录
    def __init__(self):
        self.lines = []
        # 调用父类的初始化方法
        return super().__init__()

    # 重写 emit 方法，将日志记录的时间、级别和格式化后的消息存储到列表中
    def emit(self, record):
        self.lines.append([time.time(), record.levelname, self.format(record)])

    # 定义一个未被调用的方法，暂时没有实际作用
    def onNewRecord(self, record):
        pass

# 使用 sys.argv 创建一个配置对象
config = Config(sys.argv)
```
# `ZeroNet\plugins\AnnounceShare\AnnounceSharePlugin.py`

```
# 导入时间模块
import time
# 导入操作系统模块
import os
# 导入日志模块
import logging
# 导入 JSON 模块
import json
# 导入退出注册模块
import atexit

# 导入协程模块
import gevent

# 从 Config 模块中导入 config 对象
from Config import config
# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 util 模块中导入 helper 函数
from util import helper

# 定义 TrackerStorage 类
class TrackerStorage(object):
    # 初始化方法
    def __init__(self):
        # 获取名为 "TrackerStorage" 的日志记录器
        self.log = logging.getLogger("TrackerStorage")
        # 设置文件路径为 data_dir 目录下的 trackers.json 文件
        self.file_path = "%s/trackers.json" % config.data_dir
        # 载入数据
        self.load()
        # 设置发现时间为 0.0
        self.time_discover = 0.0
        # 在程序退出时保存数据
        atexit.register(self.save)

    # 获取默认文件内容的方法
    def getDefaultFile(self):
        return {"shared": {}}

    # 当发现追踪器时的方法
    def onTrackerFound(self, tracker_address, type="shared", my=False):
        # 如果追踪器地址不是以 "zero://" 开头，则返回 False
        if not tracker_address.startswith("zero://"):
            return False

        # 获取追踪器列表
        trackers = self.getTrackers()
        added = False
        # 如果追踪器地址不在列表中，则添加到列表中
        if tracker_address not in trackers:
            trackers[tracker_address] = {
                "time_added": time.time(),
                "time_success": 0,
                "latency": 99.0,
                "num_error": 0,
                "my": False
            }
            self.log.debug("New tracker found: %s" % tracker_address)
            added = True

        # 设置追踪器发现时间和是否为自己的标志
        trackers[tracker_address]["time_found"] = time.time()
        trackers[tracker_address]["my"] = my
        return added

    # 当追踪器成功时的方法
    def onTrackerSuccess(self, tracker_address, latency):
        # 获取追踪器列表
        trackers = self.getTrackers()
        # 如果追踪器地址不在列表中，则返回 False
        if tracker_address not in trackers:
            return False

        # 设置追踪器的延迟、成功时间和错误次数
        trackers[tracker_address]["latency"] = latency
        trackers[tracker_address]["time_success"] = time.time()
        trackers[tracker_address]["num_error"] = 0
    # 当跟踪器出现错误时的处理函数，传入跟踪器地址参数
    def onTrackerError(self, tracker_address):
        # 获取所有跟踪器
        trackers = self.getTrackers()
        # 如果跟踪器地址不在跟踪器列表中，则返回 False
        if tracker_address not in trackers:
            return False

        # 更新跟踪器的错误时间和错误次数
        trackers[tracker_address]["time_error"] = time.time()
        trackers[tracker_address]["num_error"] += 1

        # 根据工作中的共享跟踪器数量设置错误限制
        if len(self.getWorkingTrackers()) >= config.working_shared_trackers_limit:
            error_limit = 5
        else:
            error_limit = 30
        error_limit

        # 如果跟踪器的错误次数超过错误限制，并且距离上次成功时间超过1小时，则删除该跟踪器
        if trackers[tracker_address]["num_error"] > error_limit and trackers[tracker_address]["time_success"] < time.time() - 60 * 60:
            self.log.debug("Tracker %s looks down, removing." % tracker_address)
            del trackers[tracker_address]

    # 获取跟踪器列表的函数，可传入类型参数，默认为共享类型
    def getTrackers(self, type="shared"):
        return self.file_content.setdefault(type, {})

    # 获取工作中的跟踪器列表的函数，可传入类型参数，默认为共享类型
    def getWorkingTrackers(self, type="shared"):
        # 使用字典推导式筛选出最近1小时内成功的跟踪器
        trackers = {
            key: tracker for key, tracker in self.getTrackers(type).items()
            if tracker["time_success"] > time.time() - 60 * 60
        }
        return trackers

    # 获取文件内容的函数
    def getFileContent(self):
        # 如果文件不存在，则创建空文件并返回默认文件内容
        if not os.path.isfile(self.file_path):
            open(self.file_path, "w").write("{}")
            return self.getDefaultFile()
        try:
            # 尝试加载文件内容为 JSON 格式
            return json.load(open(self.file_path))
        except Exception as err:
            # 如果加载失败，则记录错误并返回默认文件内容
            self.log.error("Error loading trackers list: %s" % err)
            return self.getDefaultFile()

    # 加载文件内容的函数
    def load(self):
        # 获取文件内容
        self.file_content = self.getFileContent()

        # 获取所有跟踪器
        trackers = self.getTrackers()
        self.log.debug("Loaded %s shared trackers" % len(trackers))
        # 遍历跟踪器列表，重置错误次数并删除非 "zero://" 开头的跟踪器
        for address, tracker in list(trackers.items()):
            tracker["num_error"] = 0
            if not address.startswith("zero://"):
                del trackers[address]

    # 保存文件内容的函数
    def save(self):
        # 记录保存操作开始时间
        s = time.time()
        # 使用原子写入方式将文件内容以 JSON 格式写入文件
        helper.atomicWrite(self.file_path, json.dumps(self.file_content, indent=2, sort_keys=True).encode("utf8"))
        self.log.debug("Saved in %.3fs" % (time.time() - s))
    # 发现可用的跟踪器
    def discoverTrackers(self, peers):
        # 如果已经获取的可用跟踪器数量超过配置中的限制，则返回 False
        if len(self.getWorkingTrackers()) > config.working_shared_trackers_limit:
            return False
        # 记录当前时间
        s = time.time()
        # 记录成功获取跟踪器的数量
        num_success = 0
        # 遍历每个对等节点
        for peer in peers:
            # 如果对等节点存在连接并且握手信息中的版本号小于 3560，则跳过
            if peer.connection and peer.connection.handshake.get("rev", 0) < 3560:
                continue  # Not supported

            # 向对等节点发送请求获取跟踪器信息
            res = peer.request("getTrackers")
            # 如果没有返回结果或者返回结果中包含错误信息，则跳过
            if not res or "error" in res:
                continue

            # 记录成功获取跟踪器的数量
            num_success += 1
            # 遍历返回结果中的跟踪器地址
            for tracker_address in res["trackers"]:
                # 如果跟踪器地址的类型是字节类型，则转换为 UTF-8 编码的字符串（向后兼容）
                if type(tracker_address) is bytes:  # Backward compatibilitys
                    tracker_address = tracker_address.decode("utf8")
                # 调用 onTrackerFound 方法添加跟踪器地址，并记录是否成功添加
                added = self.onTrackerFound(tracker_address)
                # 如果成功添加了跟踪器地址，则跳出循环，只添加一个来源的跟踪器
                if added:  # Only add one tracker from one source
                    break

        # 如果没有成功获取跟踪器，并且对等节点数量小于 20，则将发现跟踪器的时间设置为 0
        if not num_success and len(peers) < 20:
            self.time_discover = 0.0

        # 如果成功获取了跟踪器，则保存当前状态
        if num_success:
            self.save()

        # 记录日志，显示从对等节点中发现的跟踪器数量和总对等节点数量，以及耗时
        self.log.debug("Trackers discovered from %s/%s peers in %.3fs" % (num_success, len(peers), time.time() - s))
# 如果当前作用域中不存在名为 "tracker_storage" 的变量，则创建一个 TrackerStorage 对象并赋值给它
if "tracker_storage" not in locals():
    tracker_storage = TrackerStorage()

# 将 SiteAnnouncerPlugin 类注册到 PluginManager 的 "SiteAnnouncer" 插件中
@PluginManager.registerTo("SiteAnnouncer")
class SiteAnnouncerPlugin(object):
    # 获取跟踪器
    def getTrackers(self):
        # 如果跟踪器发现时间早于当前时间 5 分钟之前，则更新发现时间并异步执行发现跟踪器的操作
        if tracker_storage.time_discover < time.time() - 5 * 60:
            tracker_storage.time_discover = time.time()
            gevent.spawn(tracker_storage.discoverTrackers, self.site.getConnectedPeers())
        # 调用父类的 getTrackers 方法获取跟踪器列表
        trackers = super(SiteAnnouncerPlugin, self).getTrackers()
        # 获取共享跟踪器列表
        shared_trackers = list(tracker_storage.getTrackers("shared").keys())
        # 如果存在共享跟踪器，则返回所有跟踪器列表和共享跟踪器列表的合并
        if shared_trackers:
            return trackers + shared_trackers
        # 如果不存在共享跟踪器，则返回所有跟踪器列表
        else:
            return trackers

    # 通告跟踪器
    def announceTracker(self, tracker, *args, **kwargs):
        # 调用父类的 announceTracker 方法通告跟踪器，并获取返回结果
        res = super(SiteAnnouncerPlugin, self).announceTracker(tracker, *args, **kwargs)
        # 如果返回结果为真，则将延迟时间传递给 tracker_storage 的 onTrackerSuccess 方法
        if res:
            latency = res
            tracker_storage.onTrackerSuccess(tracker, latency)
        # 如果返回结果为假，则调用 tracker_storage 的 onTrackerError 方法
        elif res is False:
            tracker_storage.onTrackerError(tracker)

        return res

# 将 FileRequestPlugin 类注册到 PluginManager 的 "FileRequest" 插件中
@PluginManager.registerTo("FileRequest")
class FileRequestPlugin(object):
    # 执行获取跟踪器的操作
    def actionGetTrackers(self, params):
        # 获取共享跟踪器列表并作为响应返回
        shared_trackers = list(tracker_storage.getWorkingTrackers("shared").keys())
        self.response({"trackers": shared_trackers})

# 将 FileServerPlugin 类注册到 PluginManager 的 "FileServer" 插件中
@PluginManager.registerTo("FileServer")
class FileServerPlugin(object):
    # 端口检查
    def portCheck(self, *args, **kwargs):
        # 调用父类的 portCheck 方法进行端口检查，并获取返回结果
        res = super(FileServerPlugin, self).portCheck(*args, **kwargs)
        # 如果返回结果为真且配置不是始终使用 Tor，并且 "Bootstrapper" 在 PluginManager 中注册的插件名称中
        if res and not config.tor == "always" and "Bootstrapper" in PluginManager.plugin_manager.plugin_names:
            # 遍历外部 IP 列表，构建自己的跟踪器地址并传递给 tracker_storage 的 onTrackerFound 方法
            for ip in self.ip_external_list:
                my_tracker_address = "zero://%s:%s" % (ip, config.fileserver_port)
                tracker_storage.onTrackerFound(my_tracker_address, my=True)
        return res

# 将 ConfigPlugin 类注册到 PluginManager 的 "ConfigPlugin" 插件中
@PluginManager.registerTo("ConfigPlugin")
class ConfigPlugin(object):
    # 创建参数方法，用于为插件创建参数
    def createArguments(self):
        # 创建参数组，用于将参数组织在一起
        group = self.parser.add_argument_group("AnnounceShare plugin")
        # 添加参数到参数组，设置参数说明、默认值、类型和参数名
        group.add_argument('--working_shared_trackers_limit', help='Stop discovering new shared trackers after this number of shared trackers reached', default=5, type=int, metavar='limit')
        
        # 调用父类的创建参数方法
        return super(ConfigPlugin, self).createArguments()
```
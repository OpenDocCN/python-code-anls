# `ZeroNet\src\Site\SiteAnnouncer.py`

```py
# 导入所需的模块
import random
import time
import hashlib
import re
import collections

# 导入 gevent 模块
import gevent

# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 Config 模块中导入 config 对象
from Config import config
# 从 Debug 模块中导入 Debug 类
from Debug import Debug
# 从 util 模块中导入 helper 函数
from util import helper
# 从 greenlet 模块中导入 GreenletExit 异常
from greenlet import GreenletExit
# 从 util 模块中导入所有内容
import util

# 定义一个自定义异常类 AnnounceError
class AnnounceError(Exception):
    pass

# 创建全局变量 global_stats，使用 collections 模块的 defaultdict 创建默认字典
global_stats = collections.defaultdict(lambda: collections.defaultdict(int))

# 使用 PluginManager.acceptPlugins 装饰器注册 SiteAnnouncer 类
@PluginManager.acceptPlugins
class SiteAnnouncer(object):
    # SiteAnnouncer 类的初始化方法
    def __init__(self, site):
        # 初始化 SiteAnnouncer 实例的属性
        self.site = site
        self.stats = {}
        self.fileserver_port = config.fileserver_port
        self.peer_id = self.site.connection_server.peer_id
        self.last_tracker_id = random.randint(0, 10)
        self.time_last_announce = 0

    # 获取所有的 trackers
    def getTrackers(self):
        return config.trackers

    # 获取支持的 trackers
    def getSupportedTrackers(self):
        trackers = self.getTrackers()

        # 如果未启用 tor_manager，则移除 .onion 结尾的 trackers
        if not self.site.connection_server.tor_manager.enabled:
            trackers = [tracker for tracker in trackers if ".onion" not in tracker]

        # 移除地址未知的 trackers
        trackers = [tracker for tracker in trackers if self.getAddressParts(tracker)]

        # 如果不支持 ipv6，则移除 ipv6 类型的 trackers
        if "ipv6" not in self.site.connection_server.supported_ip_types:
            trackers = [tracker for tracker in trackers if helper.getIpType(self.getAddressParts(tracker)["ip"]) != "ipv6"]

        return trackers

    # 获取要进行 announce 的 trackers
    def getAnnouncingTrackers(self, mode):
        trackers = self.getSupportedTrackers()

        # 如果存在 trackers 并且 mode 为 "update" 或 "more"，则只在一个 tracker 上进行 announce，并增加查询的 tracker id
        if trackers and (mode == "update" or mode == "more"):
            self.last_tracker_id += 1
            self.last_tracker_id = self.last_tracker_id % len(trackers)
            trackers_announcing = [trackers[self.last_tracker_id]]
        else:
            trackers_announcing = trackers

        return trackers_announcing
    # 获取已打开的服务类型列表
    def getOpenedServiceTypes(self):
        back = []
        # 如果代理被禁用且不总是使用 Tor
        if config.trackers_proxy == "disable" and config.tor != "always":
            # 遍历连接服务器的已打开端口字典，将已打开的端口类型添加到返回列表中
            for ip_type, opened in list(self.site.connection_server.port_opened.items()):
                if opened:
                    back.append(ip_type)
        # 如果连接服务器的 Tor 管理器启动了洋葱服务
        if self.site.connection_server.tor_manager.start_onions:
            # 添加 "onion" 到返回列表中
            back.append("onion")
        # 返回已打开的服务类型列表
        return back

    # 获取追踪器处理程序
    @util.Noparallel(blocking=False)
    def getTrackerHandler(self, protocol):
        # 返回空值
        return None

    # 获取地址部分
    def getAddressParts(self, tracker):
        # 如果地址中不包含 "://" 或者不符合指定的格式
        if "://" not in tracker or not re.match("^[A-Za-z0-9:/\\.#-]+$", tracker):
            # 返回空值
            return None
        # 分割协议和地址
        protocol, address = tracker.split("://", 1)
        # 如果地址中包含端口号
        if ":" in address:
            ip, port = address.rsplit(":", 1)
        else:
            ip = address
            # 如果协议以 "https" 开头
            if protocol.startswith("https"):
                port = 443
            else:
                port = 80
        # 构建包含协议、地址、IP 和端口的字典
        back = {}
        back["protocol"] = protocol
        back["address"] = address
        back["ip"] = ip
        back["port"] = port
        # 返回包含地址部分信息的字典
        return back

    # 获取追踪器处理程序
    @util.Noparallel(blocking=False)
    # 声明一个方法，用于向对等网络宣布 PEX 信息
    def announcePex(self, query_num=2, need_num=5):
        # 获取已连接的对等节点
        peers = self.site.getConnectedPeers()
        # 如果没有已连接的对等节点，则等待3秒钟
        if len(peers) == 0:  # Wait 3s for connections
            time.sleep(3)
            peers = self.site.getConnectedPeers()

        # 如果仍然没有已连接的对等节点，则从最近的20个对等节点中选择一些进行连接
        if len(peers) == 0:  # Small number of connected peers for this site, connect to any
            peers = list(self.site.getRecentPeers(20))
            need_num = 10

        # 随机打乱对等节点的顺序
        random.shuffle(peers)
        done = 0
        total_added = 0
        # 遍历对等节点列表
        for peer in peers:
            # 向对等节点请求 PEX 信息，并获取添加的新对等节点数量
            num_added = peer.pex(need_num=need_num)
            # 如果成功添加了新对等节点
            if num_added is not False:
                done += 1
                total_added += num_added
                # 如果添加了新对等节点，则触发对等节点管理器的事件
                if num_added:
                    self.site.worker_manager.onPeers()
                    # 更新 WebSocket 的信息，添加新对等节点数量
                    self.site.updateWebsocket(peers_added=num_added)
            else:
                # 如果未成功添加新对等节点，则等待0.1秒
                time.sleep(0.1)
            # 如果已经完成了指定数量的查询，则退出循环
            if done == query_num:
                break
        # 记录 PEX 操作的结果，包括从多少个对等节点获取了多少个新的对等节点
        self.site.log.debug("Pex result: from %s peers got %s new peers." % (done, total_added))

    # 更新 WebSocket 的信息
    def updateWebsocket(self, **kwargs):
        # 如果有参数传入，则将参数转换为事件格式
        if kwargs:
            param = {"event": list(kwargs.items())[0]}
        else:
            param = None

        # 遍历站点的所有 WebSocket 连接，并触发事件
        for ws in self.site.websockets:
            ws.event("announcerChanged", self.site, param)
```
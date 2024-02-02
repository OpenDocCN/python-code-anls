# `ZeroNet\plugins\AnnounceLocal\AnnounceLocalPlugin.py`

```py
# 导入时间模块
import time

# 导入协程模块
import gevent

# 从插件管理器中导入插件管理器和配置
from Plugin import PluginManager
from Config import config
# 从当前目录下的BroadcastServer模块中导入BroadcastServer类
from . import BroadcastServer


# 将SiteAnnouncerPlugin类注册到插件管理器中
@PluginManager.registerTo("SiteAnnouncer")
class SiteAnnouncerPlugin(object):
    # 定义announce方法，用于通知
    def announce(self, force=False, *args, **kwargs):
        # 获取本地通知器
        local_announcer = self.site.connection_server.local_announcer

        # 初始化线程为None
        thread = None
        # 如果存在本地通知器并且（强制或者当前时间距离上次发现超过5分钟）
        if local_announcer and (force or time.time() - local_announcer.last_discover > 5 * 60):
            # 创建一个协程来执行本地通知器的发现方法
            thread = gevent.spawn(local_announcer.discover, force=force)
        # 调用父类的announce方法，并将参数传递下去
        back = super(SiteAnnouncerPlugin, self).announce(force=force, *args, **kwargs)

        # 如果存在线程
        if thread:
            # 等待线程执行完毕
            thread.join()

        # 返回父类的返回结果
        return back


# 定义LocalAnnouncer类，继承自BroadcastServer.BroadcastServer类
class LocalAnnouncer(BroadcastServer.BroadcastServer):
    # 初始化方法
    def __init__(self, server, listen_port):
        # 调用父类的初始化方法
        super(LocalAnnouncer, self).__init__("zeronet", listen_port=listen_port)
        # 设置服务器属性
        self.server = server

        # 设置发送者信息
        self.sender_info["peer_id"] = self.server.peer_id
        self.sender_info["port"] = self.server.port
        self.sender_info["broadcast_port"] = listen_port
        self.sender_info["rev"] = config.rev

        # 初始化已知对等方和上次发现时间
        self.known_peers = {}
        self.last_discover = 0

    # 定义发现方法
    def discover(self, force=False):
        # 打印调试信息
        self.log.debug("Sending discover request (force: %s)" % force)
        # 更新上次发现时间
        self.last_discover = time.time()
        # 如果强制，清除已知对等方缓存
        if force:
            self.known_peers = {}

        # 遍历已知对等方，超过20分钟则移除
        for peer_id, known_peer in list(self.known_peers.items()):
            if time.time() - known_peer["found"] > 20 * 60:
                del(self.known_peers[peer_id])
                self.log.debug("Timeout, removing from known_peers: %s" % peer_id)
        # 广播发现请求
        self.broadcast({"cmd": "discoverRequest", "params": {}}, port=self.listen_port)
    # 处理发现请求，返回发现响应
    def actionDiscoverRequest(self, sender, params):
        # 构建发现响应的数据结构
        back = {
            "cmd": "discoverResponse",
            "params": {
                "sites_changed": self.server.site_manager.sites_changed
            }
        }

        # 如果发送者的 peer_id 不在已知的对等节点列表中
        if sender["peer_id"] not in self.known_peers:
            # 将发送者添加到已知的对等节点列表中，并记录相关信息
            self.known_peers[sender["peer_id"]] = {"added": time.time(), "sites_changed": 0, "updated": 0, "found": time.time()}
            # 记录日志，表示从未知的对等节点接收到发现请求
            self.log.debug("Got discover request from unknown peer %s (%s), time to refresh known peers" % (sender["ip"], sender["peer_id"]))
            # 延迟一段时间后执行发现操作，确保响应先到达请求者
            gevent.spawn_later(1.0, self.discover)

        # 返回发现响应数据结构
        return back

    # 处理发现响应
    def actionDiscoverResponse(self, sender, params):
        # 如果发送者的 peer_id 在已知的对等节点列表中
        if sender["peer_id"] in self.known_peers:
            # 更新发送者的发现时间
            self.known_peers[sender["peer_id"]]["found"] = time.time()
        # 如果发送者的 sites_changed 不等于已知的对等节点列表中的 sites_changed
        if params["sites_changed"] != self.known_peers.get(sender["peer_id"], {}).get("sites_changed"):
            # 对等节点的站点列表发生变化，请求新站点列表
            return {"cmd": "siteListRequest"}
        else:
            # 对等节点的站点列表未发生变化
            # 遍历服务器的站点列表
            for site in self.server.sites.values():
                # 获取发送者在站点中的对等节点
                peer = site.peers.get("%s:%s" % (sender["ip"], sender["port"]))
                if peer:
                    # 更新对等节点的发现状态为 "local"
                    peer.found("local")

    # 处理站点列表请求
    def actionSiteListRequest(self, sender, params):
        # 初始化返回的站点列表响应
        back = []
        # 获取服务器的站点列表
        sites = list(self.server.sites.values())

        # 将站点列表分组，每组最多包含 100 个站点，以避免 UDP 大小限制
        site_groups = [sites[i:i + 100] for i in range(0, len(sites), 100)]
        # 遍历每个站点分组
        for site_group in site_groups:
            # 初始化响应数据结构
            res = {}
            res["sites_changed"] = self.server.site_manager.sites_changed
            res["sites"] = [site.address_hash for site in site_group]
            # 构建站点列表响应数据结构，并添加到返回列表中
            back.append({"cmd": "siteListResponse", "params": res})
        # 返回站点列表响应
        return back
    # 处理站点列表响应的方法，接收发送者和参数
    def actionSiteListResponse(self, sender, params):
        # 记录当前时间
        s = time.time()
        # 将参数中的站点列表转换为集合
        peer_sites = set(params["sites"])
        # 初始化找到的站点数量和已添加的站点列表
        num_found = 0
        added_sites = []
        # 遍历服务器中的站点
        for site in self.server.sites.values():
            # 如果站点的地址哈希在参数中的站点集合中
            if site.address_hash in peer_sites:
                # 将发送者的 IP 和端口添加到站点的对等列表中，来源为本地
                added = site.addPeer(sender["ip"], sender["port"], source="local")
                # 增加找到的站点数量
                num_found += 1
                # 如果成功添加了对等节点
                if added:
                    # 触发站点的 worker_manager.onPeers() 方法
                    site.worker_manager.onPeers()
                    # 更新站点的 WebSocket 连接，添加了一个对等节点
                    site.updateWebsocket(peers_added=1)
                    # 将该站点添加到已添加站点列表中
                    added_sites.append(site)

        # 保存站点变化值，避免不必要的站点列表下载
        if sender["peer_id"] not in self.known_peers:
            self.known_peers[sender["peer_id"]] = {"added": time.time()}

        # 更新已知对等节点的站点变化信息和更新时间
        self.known_peers[sender["peer_id"]]["sites_changed"] = params["sites_changed"]
        self.known_peers[sender["peer_id"]]["updated"] = time.time()
        self.known_peers[sender["peer_id"]]["found"] = time.time()

        # 记录调试日志，显示发现来自发送者的响应解析时间、找到的站点数量、已添加的站点列表和总站点数量
        self.log.debug(
            "Tracker result: Discover from %s response parsed in %.3fs, found: %s added: %s of %s" %
            (sender["ip"], time.time() - s, num_found, added_sites, len(peer_sites))
        )
# 将 FileServerPlugin 类注册到 PluginManager 的 FileServer 插件中
@PluginManager.registerTo("FileServer")
class FileServerPlugin(object):
    # 初始化方法
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super(FileServerPlugin, self).__init__(*args, **kwargs)
        # 如果配置中存在广播端口、不是始终使用 Tor、并且未禁用 UDP，则创建本地广播器
        if config.broadcast_port and config.tor != "always" and not config.disable_udp:
            self.local_announcer = LocalAnnouncer(self, config.broadcast_port)
        else:
            self.local_announcer = None

    # 启动方法
    def start(self, *args, **kwargs):
        # 如果存在本地广播器，则使用协程启动本地广播器
        if self.local_announcer:
            gevent.spawn(self.local_announcer.start)
        # 调用父类的启动方法
        return super(FileServerPlugin, self).start(*args, **kwargs)

    # 停止方法
    def stop(self):
        # 如果存在本地广播器，则停止本地广播器
        if self.local_announcer:
            self.local_announcer.stop()
        # 调用父类的停止方法，并返回结果
        res = super(FileServerPlugin, self).stop()
        return res


# 将 ConfigPlugin 类注册到 PluginManager 的 ConfigPlugin 插件中
@PluginManager.registerTo("ConfigPlugin")
class ConfigPlugin(object):
    # 创建参数方法
    def createArguments(self):
        # 在解析器中创建参数组"AnnounceLocal plugin"
        group = self.parser.add_argument_group("AnnounceLocal plugin")
        # 添加参数'--broadcast_port'，用于本地对等体发现的 UDP 广播端口，默认为 1544
        group.add_argument('--broadcast_port', help='UDP broadcasting port for local peer discovery', default=1544, type=int, metavar='port')

        # 调用父类的创建参数方法
        return super(ConfigPlugin, self).createArguments()
```
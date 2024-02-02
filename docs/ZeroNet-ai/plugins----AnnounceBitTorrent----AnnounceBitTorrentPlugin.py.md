# `ZeroNet\plugins\AnnounceBitTorrent\AnnounceBitTorrentPlugin.py`

```py
# 导入时间模块
import time
# 导入 urllib.request 模块
import urllib.request
# 导入 struct 模块
import struct
# 导入 socket 模块
import socket

# 导入 lib.bencode_open 模块并重命名为 bencode_open
import lib.bencode_open as bencode_open
# 从 lib.subtl.subtl 模块中导入 UdpTrackerClient 类
from lib.subtl.subtl import UdpTrackerClient
# 导入 socks 模块
import socks
# 导入 sockshandler 模块
import sockshandler
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

# 在加载插件后执行的函数，用于导入插件主机类
@PluginManager.afterLoad
def importHostClasses():
    # 声明 Peer 和 AnnounceError 为全局变量
    global Peer, AnnounceError
    # 从 Peer 模块中导入 Peer 类
    from Peer import Peer
    # 从 Site.SiteAnnouncer 模块中导入 AnnounceError 类
    from Site.SiteAnnouncer import AnnounceError

# 注册到 SiteAnnouncer 的插件类
@PluginManager.registerTo("SiteAnnouncer")
class SiteAnnouncerPlugin(object):
    # 获取支持的追踪器列表
    def getSupportedTrackers(self):
        # 调用父类的 getSupportedTrackers 方法获取追踪器列表
        trackers = super(SiteAnnouncerPlugin, self).getSupportedTrackers()
        # 如果禁用了 UDP 或者设置了追踪器代理为 "disable"，则过滤掉 UDP 追踪器
        if config.disable_udp or config.trackers_proxy != "disable":
            trackers = [tracker for tracker in trackers if not tracker.startswith("udp://")]
        # 返回过滤后的追踪器列表
        return trackers

    # 获取追踪器处理程序
    def getTrackerHandler(self, protocol):
        # 根据协议选择相应的追踪器处理程序
        if protocol == "udp":
            handler = self.announceTrackerUdp
        elif protocol == "http":
            handler = self.announceTrackerHttp
        elif protocol == "https":
            handler = self.announceTrackerHttps
        else:
            handler = super(SiteAnnouncerPlugin, self).getTrackerHandler(protocol)
        # 返回选择的追踪器处理程序
        return handler
    # 声明一个方法，用于向 UDP 跟踪器宣告状态
    def announceTrackerUdp(self, tracker_address, mode="start", num_want=10):
        # 记录当前时间
        s = time.time()
        # 如果配置中禁用了 UDP，则抛出异常
        if config.disable_udp:
            raise AnnounceError("Udp disabled by config")
        # 如果配置中设置了代理，则抛出异常
        if config.trackers_proxy != "disable":
            raise AnnounceError("Udp trackers not available with proxies")

        # 从跟踪器地址中获取 IP 和端口
        ip, port = tracker_address.split("/")[0].split(":")
        # 创建 UDP 跟踪器客户端对象
        tracker = UdpTrackerClient(ip, int(port))
        # 如果 IP 类型在已打开的服务类型中，则设置 peer_port 为文件服务器端口，否则为 0
        if helper.getIpType(ip) in self.getOpenedServiceTypes():
            tracker.peer_port = self.fileserver_port
        else:
            tracker.peer_port = 0
        # 连接到跟踪器
        tracker.connect()
        # 如果连接失败，则抛出异常
        if not tracker.poll_once():
            raise AnnounceError("Could not connect")
        # 向跟踪器宣告状态
        tracker.announce(info_hash=self.site.address_sha1, num_want=num_want, left=431102370)
        # 获取跟踪器的响应
        back = tracker.poll_once()
        # 如果没有响应，则抛出异常
        if not back:
            raise AnnounceError("No response after %.0fs" % (time.time() - s))
        # 如果响应是字典并且包含 "response" 键，则获取 peers
        elif type(back) is dict and "response" in back:
            peers = back["response"]["peers"]
        else:
            # 否则抛出异常
            raise AnnounceError("Invalid response: %r" % back)

        # 返回 peers
        return peers
    # 发起 HTTP 请求
    def httpRequest(self, url):
        # 设置请求头信息
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
            'Accept-Encoding': 'none',
            'Accept-Language': 'en-US,en;q=0.8',
            'Connection': 'keep-alive'
        }

        # 创建 HTTP 请求对象
        req = urllib.request.Request(url, headers=headers)

        # 根据配置选择是否使用代理
        if config.trackers_proxy == "tor":
            # 如果配置为使用 Tor 代理，则创建 Tor 代理处理器
            tor_manager = self.site.connection_server.tor_manager
            handler = sockshandler.SocksiPyHandler(socks.SOCKS5, tor_manager.proxy_ip, tor_manager.proxy_port)
            opener = urllib.request.build_opener(handler)
            # 使用代理发起请求，设置超时时间为50秒
            return opener.open(req, timeout=50)
        elif config.trackers_proxy == "disable":
            # 如果配置为禁用代理，则直接发起请求，设置超时时间为25秒
            return urllib.request.urlopen(req, timeout=25)
        else:
            # 如果配置为使用自定义代理，则根据配置创建代理处理器
            proxy_ip, proxy_port = config.trackers_proxy.split(":")
            handler = sockshandler.SocksiPyHandler(socks.SOCKS5, proxy_ip, int(proxy_port))
            opener = urllib.request.build_opener(handler)
            # 使用代理发起请求，设置超时时间为50秒
            return opener.open(req, timeout=50)

    # 发起 HTTPS 请求
    def announceTrackerHttps(self, *args, **kwargs):
        # 设置协议为 HTTPS
        kwargs["protocol"] = "https"
        # 调用 HTTP 请求方法，传入参数并返回结果
        return self.announceTrackerHttp(*args, **kwargs)
    # 声明一个方法，用于向指定的 tracker 地址发送 HTTP 请求，进行连接或者断开连接
    def announceTrackerHttp(self, tracker_address, mode="start", num_want=10, protocol="http"):
        # 从 tracker 地址中分离出 IP 地址和端口号
        tracker_ip, tracker_port = tracker_address.rsplit(":", 1)
        # 如果 tracker_ip 的类型在已打开的服务类型列表中
        if helper.getIpType(tracker_ip) in self.getOpenedServiceTypes():
            # 使用文件服务器端口
            port = self.fileserver_port
        else:
            # 否则使用端口 1
            port = 1
        # 设置请求参数
        params = {
            'info_hash': self.site.address_sha1,
            'peer_id': self.peer_id, 'port': port,
            'uploaded': 0, 'downloaded': 0, 'left': 431102370, 'compact': 1, 'numwant': num_want,
            'event': 'started'
        }
        # 构建完整的 URL
        url = protocol + "://" + tracker_address + "?" + urllib.parse.urlencode(params)

        # 记录当前时间
        s = time.time()
        response = None
        # 加载 URL
        if config.tor == "always" or config.trackers_proxy != "disable":
            timeout = 60
        else:
            timeout = 30
        # 设置超时时间
        with gevent.Timeout(timeout, False):  # Make sure of timeout
            # 发起 HTTP 请求
            req = self.httpRequest(url)
            # 读取响应
            response = req.read()
            # 关闭请求
            req.close()
            req = None

        # 如果没有响应，则抛出异常
        if not response:
            raise AnnounceError("No response after %.0fs" % (time.time() - s))

        # 解码 peers
        try:
            # 解析响应数据
            peer_data = bencode_open.loads(response)[b"peers"]
            response = None
            # 计算 peer 的数量
            peer_count = int(len(peer_data) / 6)
            peers = []
            # 遍历 peer 数据，解析出 IP 地址和端口号
            for peer_offset in range(peer_count):
                off = 6 * peer_offset
                peer = peer_data[off:off + 6]
                addr, port = struct.unpack('!LH', peer)
                peers.append({"addr": socket.inet_ntoa(struct.pack('!L', addr)), "port": port})
        except Exception as err:
            # 如果解析出错，则抛出异常
            raise AnnounceError("Invalid response: %r (%s)" % (response, Debug.formatException(err)))

        # 返回解析出的 peers 列表
        return peers
```
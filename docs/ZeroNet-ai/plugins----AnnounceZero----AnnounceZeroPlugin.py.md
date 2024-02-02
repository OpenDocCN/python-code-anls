# `ZeroNet\plugins\AnnounceZero\AnnounceZeroPlugin.py`

```py
# 导入时间模块
import time
# 导入迭代工具模块
import itertools

# 从插件管理器中导入插件类
from Plugin import PluginManager
# 从工具模块中导入辅助函数
from util import helper
# 从加密模块中导入 RSA 加密类
from Crypt import CryptRsa

# 是否允许重新加载插件的标志
allow_reload = False  
# 追踪地址: 最后一次向追踪器宣布所有站点的字典
time_full_announced = {}  
# 追踪地址: 对等体对象的连接池
connection_pool = {}  

# 插件加载后导入主机类
@PluginManager.afterLoad
def importHostClasses():
    global Peer, AnnounceError
    # 从对等体模块中导入对等体类
    from Peer import Peer
    # 从站点宣布模块中导入宣布错误类
    from Site.SiteAnnouncer import AnnounceError

# 处理从追踪器返回的结果
def processPeerRes(tracker_address, site, peers):
    added = 0

    # 处理 Onion 地址
    found_onion = 0
    for packed_address in peers["onion"]:
        found_onion += 1
        # 解析 Onion 地址
        peer_onion, peer_port = helper.unpackOnionAddress(packed_address)
        if site.addPeer(peer_onion, peer_port, source="tracker"):
            added += 1

    # 处理 Ip4 地址
    found_ipv4 = 0
    # 合并 Ip4、Ipv4、Ipv6 地址
    peers_normal = itertools.chain(peers.get("ip4", []), peers.get("ipv4", []), peers.get("ipv6", []))
    for packed_address in peers_normal:
        found_ipv4 += 1
        # 解析 Ip4 地址
        peer_ip, peer_port = helper.unpackAddress(packed_address)
        if site.addPeer(peer_ip, peer_port, source="tracker"):
            added += 1

    # 如果有新增的对等体，则触发对等体管理器的事件
    if added:
        site.worker_manager.onPeers()
        # 更新 WebSocket，通知新增的对等体数量
        site.updateWebsocket(peers_added=added)
    return added

# 注册到站点宣布器的插件类
@PluginManager.registerTo("SiteAnnouncer")
class SiteAnnouncerPlugin(object):
    # 获取追踪器处理程序
    def getTrackerHandler(self, protocol):
        if protocol == "zero":
            return self.announceTrackerZero
        else:
            return super(SiteAnnouncerPlugin, self).getTrackerHandler(protocol)
```
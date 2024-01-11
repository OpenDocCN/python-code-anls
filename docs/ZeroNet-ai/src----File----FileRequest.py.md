# `ZeroNet\src\File\FileRequest.py`

```
# 导入所需的模块
import os
import time
import json
import collections
import itertools

# 导入第三方模块
import gevent

# 从 Debug 模块中导入 Debug 类
from Debug import Debug
# 从 Config 模块中导入 config 对象
from Config import config
# 从 util 模块中导入 RateLimit、Msgpack 和 helper
from util import RateLimit
from util import Msgpack
from util import helper
# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 contextlib 模块中导入 closing 函数
from contextlib import closing

# 定义文件缓冲区大小
FILE_BUFF = 1024 * 512


# 定义请求错误的异常类
class RequestError(Exception):
    pass


# 处理传入的请求
@PluginManager.acceptPlugins
class FileRequest(object):
    # 定义类的槽
    __slots__ = ("server", "connection", "req_id", "sites", "log", "responded")

    # 初始化方法
    def __init__(self, server, connection):
        self.server = server
        self.connection = connection

        self.req_id = None
        self.sites = self.server.sites
        self.log = server.log
        self.responded = False  # Responded to the request

    # 发送消息
    def send(self, msg, streaming=False):
        if not self.connection.closed:
            self.connection.send(msg, streaming)

    # 发送原始文件
    def sendRawfile(self, file, read_bytes):
        if not self.connection.closed:
            self.connection.sendRawfile(file, read_bytes)

    # 响应消息
    def response(self, msg, streaming=False):
        if self.responded:
            if config.verbose:
                self.log.debug("Req id %s already responded" % self.req_id)
            return
        if not isinstance(msg, dict):  # 如果消息不是字典，则创建一个 {"body": msg}
            msg = {"body": msg}
        msg["cmd"] = "response"
        msg["to"] = self.req_id
        self.responded = True
        self.send(msg, streaming=streaming)

    # 路由文件请求
    # 更新站点文件请求
    def isReadable(self, site, inner_path, file, pos):
        return True

    # 发送文件内容请求
    def actionGetFile(self, params):
        return self.handleGetFile(params)

    # 发送文件流请求
    def actionStreamFile(self, params):
        return self.handleGetFile(params, streaming=True)

    # 对等交换请求
    # 定义一个名为 actionPex 的方法，接受参数 params
    def actionPex(self, params):
        # 从 sites 字典中获取指定参数 site 对应的站点对象
        site = self.sites.get(params["site"])
        # 如果站点对象不存在或者不在服务状态，则返回错误信息并记录错误次数，然后返回 False
        if not site or not site.isServing():  # Site unknown or not serving
            self.response({"error": "Unknown site"})
            self.connection.badAction(5)
            return False

        # 初始化一个空列表用于存储已获取的对等节点的键
        got_peer_keys = []
        # 初始化一个变量用于记录添加的对等节点数量
        added = 0

        # 将请求者的对等节点添加到站点中
        connected_peer = site.addPeer(self.connection.ip, self.connection.port, source="request")

        # 如果成功添加了请求者的对等节点
        if connected_peer:  # It was not registered before
            # 增加添加数量，并将当前连接分配给对等节点
            added += 1
            connected_peer.connect(self.connection)  # Assign current connection to peer

        # 将发送的对等节点添加到站点中
        for packed_address in itertools.chain(params.get("peers", []), params.get("peers_ipv6", [])):
            # 解包地址信息
            address = helper.unpackAddress(packed_address)
            # 将解包后的地址添加到站点中，并记录对等节点的键
            got_peer_keys.append("%s:%s" % address)
            if site.addPeer(*address, source="pex"):
                added += 1

        # 将发送的洋葱对等节点添加到站点中
        for packed_address in params.get("peers_onion", []):
            # 解包洋葱地址信息
            address = helper.unpackOnionAddress(packed_address)
            # 将解包后的洋葱地址添加到站点中，并记录对等节点的键
            got_peer_keys.append("%s:%s" % address)
            if site.addPeer(*address, source="pex"):
                added += 1

        # 发送回不在发送列表中且可连接的对等节点（端口不为 0）
        packed_peers = helper.packPeers(site.getConnectablePeers(params["need"], ignore=got_peer_keys, allow_private=False))

        # 如果有添加对等节点
        if added:
            # 触发站点的 worker_manager.onPeers() 方法
            site.worker_manager.onPeers()
            # 如果配置为详细模式，则记录添加的对等节点数量和发送回的对等节点信息
            if config.verbose:
                self.log.debug(
                    "Added %s peers to %s using pex, sending back %s" %
                    (added, site, {key: len(val) for key, val in packed_peers.items()})
                )

        # 构建返回的数据字典
        back = {
            "peers": packed_peers["ipv4"],
            "peers_ipv6": packed_peers["ipv6"],
            "peers_onion": packed_peers["onion"]
        }

        # 返回数据字典作为响应
        self.response(back)
    # 获取自上次修改以来的 content.json 文件
    def actionListModified(self, params):
        # 获取指定站点
        site = self.sites.get(params["site"])
        # 如果站点不存在或者不在服务中，则返回错误信息并标记为错误操作
        if not site or not site.isServing():  # Site unknown or not serving
            self.response({"error": "Unknown site"})
            self.connection.badAction(5)
            return False
        # 获取自指定时间以来修改过的文件列表
        modified_files = site.content_manager.listModified(params["since"])
    
        # 如果对等节点之前未添加，则将对等节点添加到站点
        connected_peer = site.addPeer(self.connection.ip, self.connection.port, source="request")
        if connected_peer:  # Just added
            connected_peer.connect(self.connection)  # Assign current connection to peer
        # 返回修改过的文件列表
        self.response({"modified_files": modified_files})
    
    # 获取指定站点的哈希字段
    def actionGetHashfield(self, params):
        # 获取指定站点
        site = self.sites.get(params["site"])
        # 如果站点不存在或者不在服务中，则返回错误信息并标记为错误操作
        if not site or not site.isServing():  # Site unknown or not serving
            self.response({"error": "Unknown site"})
            self.connection.badAction(5)
            return False
        # 如果对等节点之前未添加，则将对等节点添加到站点
        peer = site.addPeer(self.connection.ip, self.connection.port, return_peer=True, source="request")
        if not peer.connection:  # Just added
            peer.connect(self.connection)  # Assign current connection to peer
        # 更新对等节点发送哈希字段的时间
        peer.time_my_hashfield_sent = time.time()  # Don't send again if not changed
        # 返回站点的哈希字段的原始数据
        self.response({"hashfield_raw": site.content_manager.hashfield.tobytes()})
    
    # 在指定站点中查找哈希 ID
    def findHashIds(self, site, hash_ids, limit=100):
        # 创建默认字典
        back = collections.defaultdict(lambda: collections.defaultdict(list))
        # 在工作管理器中查找可选的哈希 ID
        found = site.worker_manager.findOptionalHashIds(hash_ids, limit=limit)
    
        # 遍历找到的哈希 ID 和对应的对等节点
        for hash_id, peers in found.items():
            for peer in peers:
                # 获取对等节点的 IP 类型
                ip_type = helper.getIpType(peer.ip)
                # 如果对应 IP 类型下的哈希 ID 列表长度小于 20，则将对等节点的地址添加到列表中
                if len(back[ip_type][hash_id]) < 20:
                    back[ip_type][hash_id].append(peer.packMyAddress())
        # 返回结果
        return back
    # 查找哈希 ID
    def actionFindHashIds(self, params):
        # 获取指定站点
        site = self.sites.get(params["site"])
        # 记录当前时间
        s = time.time()
        # 如果站点不存在或者不在服务中，则返回错误信息并记录不良操作
        if not site or not site.isServing():  # Site unknown or not serving
            self.response({"error": "Unknown site"})
            self.connection.badAction(5)
            return False

        # 生成事件键
        event_key = "%s_findHashIds_%s_%s" % (self.connection.ip, params["site"], len(params["hash_ids"]))
        # 如果 CPU 时间超过0.5秒或者不允许访问频率，则休眠0.1秒后执行 findHashIds 方法
        if self.connection.cpu_time > 0.5 or not RateLimit.isAllowed(event_key, 60 * 5):
            time.sleep(0.1)
            back = self.findHashIds(site, params["hash_ids"], limit=10)
        else:
            back = self.findHashIds(site, params["hash_ids"])
        # 记录事件键被调用
        RateLimit.called(event_key)

        # 初始化哈希列表
        my_hashes = []
        # 获取站点的哈希字段集合
        my_hashfield_set = set(site.content_manager.hashfield)
        # 遍历参数中的哈希 ID，如果存在于哈希字段集合中，则添加到 my_hashes 中
        for hash_id in params["hash_ids"]:
            if hash_id in my_hashfield_set:
                my_hashes.append(hash_id)

        # 如果配置为详细模式，则记录日志
        if config.verbose:
            self.log.debug(
                "Found: %s for %s hashids in %.3fs" %
                ({key: len(val) for key, val in back.items()}, len(params["hash_ids"]), time.time() - s)
            )
        # 返回找到的对等节点信息和自身的哈希列表
        self.response({"peers": back["ipv4"], "peers_onion": back["onion"], "peers_ipv6": back["ipv6"], "my": my_hashes})

    # 设置哈希字段
    def actionSetHashfield(self, params):
        # 获取指定站点
        site = self.sites.get(params["site"])
        # 如果站点不存在或者不在服务中，则返回错误信息并记录不良操作
        if not site or not site.isServing():  # Site unknown or not serving
            self.response({"error": "Unknown site"})
            self.connection.badAction(5)
            return False

        # 添加或获取对等节点
        peer = site.addPeer(self.connection.ip, self.connection.port, return_peer=True, connection=self.connection, source="request")
        # 如果对等节点没有连接，则连接到当前连接
        if not peer.connection:
            peer.connect(self.connection)
        # 从参数中获取哈希字段的原始字节并替换站点的哈希字段
        peer.hashfield.replaceFromBytes(params["hashfield_raw"])
        # 返回更新成功信息
        self.response({"ok": "Updated"})

    # 发送简单的 Pong! 回复
    # 响应 Ping 请求，返回 "Pong!"
    def actionPing(self, params):
        self.response(b"Pong!")

    # 检查其他对等端请求的端口
    def actionCheckport(self, params):
        # 如果连接的 IP 类型是 IPv6，则创建包含 IPv6 地址和端口的套接字地址
        if helper.getIpType(self.connection.ip) == "ipv6":
            sock_address = (self.connection.ip, params["port"], 0, 0)
        # 否则，创建包含 IPv4 地址和端口的套接字地址
        else:
            sock_address = (self.connection.ip, params["port"])

        # 使用带有 IP 地址的套接字创建套接字对象，并使用 with 语句确保在使用后关闭套接字
        with closing(helper.createSocket(self.connection.ip)) as sock:
            # 设置套接字超时时间为 5 秒
            sock.settimeout(5)
            # 尝试连接到指定的套接字地址，如果连接成功则返回状态为 "open"，否则返回状态为 "closed"
            if sock.connect_ex(sock_address) == 0:
                self.response({"status": "open", "ip_external": self.connection.ip})
            else:
                self.response({"status": "closed", "ip_external": self.connection.ip})

    # 未知命令
    def actionUnknown(self, cmd, params):
        # 响应未知命令错误，并记录错误信息
        self.response({"error": "Unknown command: %s" % cmd})
        self.connection.badAction(5)
```
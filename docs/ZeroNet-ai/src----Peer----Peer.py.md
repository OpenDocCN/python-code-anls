# `ZeroNet\src\Peer\Peer.py`

```py
# 导入日志模块
import logging
# 导入时间模块
import time
# 导入系统模块
import sys
# 导入迭代工具模块
import itertools
# 导入集合模块
import collections

# 导入协程模块
import gevent

# 导入输入输出模块
import io
# 从 Debug 模块中导入 Debug 类
from Debug import Debug
# 从 Config 模块中导入 config 对象
from Config import config
# 从 util 模块中导入 helper 函数
from util import helper
# 从当前目录下的 PeerHashfield 模块中导入 PeerHashfield 类
from .PeerHashfield import PeerHashfield
# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager

# 如果配置中使用临时文件，则导入临时文件模块
if config.use_tempfiles:
    import tempfile

# 通信远程对等节点
# 使用插件管理器接受插件
class Peer(object):
    # 定义类的__slots__属性
    __slots__ = (
        "ip", "port", "site", "key", "connection", "connection_server", "time_found", "time_response", "time_hashfield",
        "time_added", "has_hashfield", "is_tracker_connection", "time_my_hashfield_sent", "last_ping", "reputation",
        "last_content_json_update", "hashfield", "connection_error", "hash_failed", "download_bytes", "download_time"
    )

    # 初始化方法
    def __init__(self, ip, port, site=None, connection_server=None):
        # 初始化对象属性
        self.ip = ip
        self.port = port
        self.site = site
        self.key = "%s:%s" % (ip, port)

        self.connection = None
        self.connection_server = connection_server
        self.has_hashfield = False  # Lazy hashfield object not created yet
        self.time_hashfield = None  # Last time peer's hashfiled downloaded
        self.time_my_hashfield_sent = None  # Last time my hashfield sent to peer
        self.time_found = time.time()  # Time of last found in the torrent tracker
        self.time_response = None  # Time of last successful response from peer
        self.time_added = time.time()
        self.last_ping = None  # Last response time for ping
        self.is_tracker_connection = False  # Tracker connection instead of normal peer
        self.reputation = 0  # More likely to connect if larger
        self.last_content_json_update = 0.0  # Modify date of last received content.json

        self.connection_error = 0  # Series of connection error
        self.hash_failed = 0  # Number of bad files from peer
        self.download_bytes = 0  # Bytes downloaded
        self.download_time = 0  # Time spent to download
    # 当试图获取一个不存在的属性时，会调用该方法
    def __getattr__(self, key):
        # 如果试图获取的属性是"hashfield"，则设置has_hashfield为True，并返回PeerHashfield对象
        if key == "hashfield":
            self.has_hashfield = True
            self.hashfield = PeerHashfield()
            return self.hashfield
        # 如果试图获取的属性不是"hashfield"，则调用默认的getattr方法
        else:
            return getattr(self, key)

    # 记录日志
    def log(self, text):
        # 如果不是调试模式，直接返回，不记录日志
        if not config.verbose:
            return  # Only log if we are in debug mode
        # 如果有site属性，则使用site的日志记录器记录日志
        if self.site:
            self.site.log.debug("%s:%s %s" % (self.ip, self.port, text))
        # 如果没有site属性，则使用全局的logging模块记录日志
        else:
            logging.debug("%s:%s %s" % (self.ip, self.port, text))

    # 连接到主机
    # 建立连接方法，可以传入一个连接对象
    def connect(self, connection=None):
        # 如果声誉小于-10，则将声誉设为-10
        if self.reputation < -10:
            self.reputation = -10
        # 如果声誉大于10，则将声誉设为10
        if self.reputation > 10:
            self.reputation = 10

        # 如果已经有连接对象
        if self.connection:
            # 记录日志，关闭当前连接
            self.log("Getting connection (Closing %s)..." % self.connection)
            self.connection.close("Connection change")
        else:
            # 记录日志，获取连接（根据声誉）
            self.log("Getting connection (reputation: %s)..." % self.reputation)

        # 如果传入了连接对象
        if connection:  # Connection specified
            # 记录日志，分配连接对象
            self.log("Assigning connection %s" % connection)
            self.connection = connection
            self.connection.sites += 1
        else:  # 如果没有传入连接对象，则尝试从连接池中查找或创建新的连接
            self.connection = None

            try:
                # 如果有连接服务器，则使用该连接服务器
                if self.connection_server:
                    connection_server = self.connection_server
                # 如果没有连接服务器但有站点信息，则使用站点的连接服务器
                elif self.site:
                    connection_server = self.site.connection_server
                # 否则，导入主模块，使用主文件服务器
                else:
                    import main
                    connection_server = main.file_server
                # 获取连接对象，并增加声誉和站点数
                self.connection = connection_server.getConnection(self.ip, self.port, site=self.site, is_tracker_connection=self.is_tracker_connection)
                self.reputation += 1
                self.connection.sites += 1
            except Exception as err:
                # 如果获取连接出现异常，则处理连接错误，并记录日志
                self.onConnectionError("Getting connection error")
                self.log("Getting connection error: %s (connection_error: %s, hash_failed: %s)" %
                         (Debug.formatException(err), self.connection_error, self.hash_failed))
                self.connection = None
        # 返回连接对象
        return self.connection

    # Check if we have connection to peer
    # 寻找与当前节点的连接
    def findConnection(self):
        # 如果已经建立连接并且连接处于连接状态
        if self.connection and self.connection.connected:  # We have connection to peer
            return self.connection
        else:  # 否则尝试从其他站点连接中寻找
            # 从站点连接服务器获取连接，如果不存在则创建一个新连接
            self.connection = self.site.connection_server.getConnection(self.ip, self.port, create=False, site=self.site)
            if self.connection:
                # 增加站点连接数
                self.connection.sites += 1
        return self.connection

    # 返回节点的字符串表示形式
    def __str__(self):
        if self.site:
            return "Peer:%-12s of %s" % (self.ip, self.site.address_short)
        else:
            return "Peer:%-12s" % self.ip

    # 返回节点的字符串表示形式
    def __repr__(self):
        return "<%s>" % self.__str__()

    # 打包节点的地址信息
    def packMyAddress(self):
        if self.ip.endswith(".onion"):
            return helper.packOnionAddress(self.ip, self.port)
        else:
            return helper.packAddress(self.ip, self.port)

    # 从某个来源找到一个节点
    def found(self, source="other"):
        # 如果声誉小于5
        if self.reputation < 5:
            # 如果来源是追踪器
            if source == "tracker":
                # 如果节点地址以.onion结尾，增加声誉1；否则增加声誉2
                if self.ip.endswith(".onion"):
                    self.reputation += 1
                else:
                    self.reputation += 2
            # 如果来源是本地，增加声誉20
            elif source == "local":
                self.reputation += 20

        # 如果来源是追踪器或本地，将节点添加到站点最近节点列表的左侧
        if source in ("tracker", "local"):
            self.site.peers_recent.appendleft(self)
        # 记录节点被发现的时间
        self.time_found = time.time()

    # 向节点发送命令并返回响应值
    # 发送请求到对等节点，可以包含命令和参数，还可以选择将结果流式传输到指定位置
    def request(self, cmd, params={}, stream_to=None):
        # 如果连接不存在或者已关闭，则重新连接
        if not self.connection or self.connection.closed:
            self.connect()
            # 如果重新连接失败，则返回空
            if not self.connection:
                self.onConnectionError("Reconnect error")
                return None  # Connection failed

        # 记录发送的请求信息
        self.log("Send request: %s %s %s %s" % (params.get("site", ""), cmd, params.get("inner_path", ""), params.get("location", "")))

        # 重试3次
        for retry in range(1, 4):  # Retry 3 times
            try:
                # 如果连接不存在，则抛出异常
                if not self.connection:
                    raise Exception("No connection found")
                # 发送请求并获取结果
                res = self.connection.request(cmd, params, stream_to)
                # 如果结果为空，则抛出异常
                if not res:
                    raise Exception("Send error")
                # 如果结果中包含错误信息，则记录错误并触发连接错误处理
                if "error" in res:
                    self.log("%s error: %s" % (cmd, res["error"]))
                    self.onConnectionError("Response error")
                    break
                else:  # Successful request, reset connection error num
                    self.connection_error = 0
                self.time_response = time.time()
                # 如果结果不为空，则返回结果
                if res:
                    return res
                else:
                    raise Exception("Invalid response: %s" % res)
            except Exception as err:
                # 如果异常类型为"Notify"，表示工作进程被终止，记录错误信息并中断命令
                if type(err).__name__ == "Notify":  # Greenlet killed by worker
                    self.log("Peer worker got killed: %s, aborting cmd: %s" % (err.message, cmd))
                    break
                else:
                    # 触发连接错误处理，记录错误信息并重连
                    self.onConnectionError("Request error")
                    self.log(
                        "%s (connection_error: %s, hash_failed: %s, retry: %s)" %
                        (Debug.formatException(err), self.connection_error, self.hash_failed, retry)
                    )
                    time.sleep(1 * retry)
                    self.connect()
        return None  # Failed after 4 retry

    # 从对等节点获取文件内容
    # 发送一个ping请求
    # 发送 ping 请求给对等节点，返回响应时间
    def ping(self):
        response_time = None
        for retry in range(1, 3):  # 重试 3 次
            s = time.time()  # 记录当前时间
            with gevent.Timeout(10.0, False):  # 设置 10 秒超时，不抛出异常
                res = self.request("ping")  # 发送 ping 请求

                if res and "body" in res and res["body"] == b"Pong!":  # 如果收到响应并且响应内容为 "Pong!"
                    response_time = time.time() - s  # 计算响应时间
                    break  # 一切正常，退出循环
            # 超时或者收到错误响应
            self.onConnectionError("Ping timeout")  # 处理连接错误
            self.connect()  # 重新连接
            time.sleep(1)  # 等待 1 秒

        if response_time:  # 如果有响应时间
            self.log("Ping: %.3f" % response_time)  # 记录响应时间
        else:  # 如果没有响应时间
            self.log("Ping failed")  # 记录 ping 失败
        self.last_ping = response_time  # 更新最后一次 ping 的时间
        return response_time  # 返回响应时间

    # 从对等节点请求对等交换
    # 定义一个方法，用于获取指定站点的可连接对等方，默认返回5个
    def pex(self, site=None, need_num=5):
        # 如果没有指定站点，则使用默认站点请求对等方
        if not site:
            site = self.site  # If no site defined request peers for this site

        # 将可连接对等方打包成字典格式
        packed_peers = helper.packPeers(self.site.getConnectablePeers(5, allow_private=False))
        # 构建请求参数
        request = {"site": site.address, "peers": packed_peers["ipv4"], "need": need_num}
        # 如果存在 onion 对等方，则添加到请求参数中
        if packed_peers["onion"]:
            request["peers_onion"] = packed_peers["onion"]
        # 如果存在 IPv6 对等方，则添加到请求参数中
        if packed_peers["ipv6"]:
            request["peers_ipv6"] = packed_peers["ipv6"]
        # 发起请求
        res = self.request("pex", request)
        # 如果没有响应或者响应中包含错误信息，则返回 False
        if not res or "error" in res:
            return False
        added = 0

        # 移除不支持的对等方类型
        if "peers_ipv6" in res and self.connection and "ipv6" not in self.connection.server.supported_ip_types:
            del res["peers_ipv6"]

        if "peers_onion" in res and self.connection and "onion" not in self.connection.server.supported_ip_types:
            del res["peers_onion"]

        # 添加 IPv4 + IPv6 对等方
        for peer in itertools.chain(res.get("peers", []), res.get("peers_ipv6", [])):
            address = helper.unpackAddress(peer)
            # 如果成功添加对等方，则计数加一
            if site.addPeer(*address, source="pex"):
                added += 1

        # 添加 Onion 对等方
        for peer in res.get("peers_onion", []):
            address = helper.unpackOnionAddress(peer)
            # 如果成功添加对等方，则计数加一
            if site.addPeer(*address, source="pex"):
                added += 1

        # 如果成功添加了对等方，则记录日志
        if added:
            self.log("Added peers using pex: %s" % added)

        return added

    # 列出自指定日期以来修改过的文件
    # 返回格式为 {inner_path: modification date,...}
    def listModified(self, since):
        return self.request("listModified", {"since": since, "site": self.site.address})
    # 更新哈希字段，如果 force 为 False 并且距离上次更新时间不到 5 分钟，则不更新
    def updateHashfield(self, force=False):
        if self.time_hashfield and time.time() - self.time_hashfield < 5 * 60 and not force:
            return False

        # 更新哈希字段的时间戳
        self.time_hashfield = time.time()
        # 通过请求获取哈希字段
        res = self.request("getHashfield", {"site": self.site.address})
        # 如果没有获取到结果或者结果中包含错误信息或者哈希字段原始数据不在结果中，则返回 False
        if not res or "error" in res or "hashfield_raw" not in res:
            return False
        # 用获取到的哈希字段原始数据替换当前对象的哈希字段
        self.hashfield.replaceFromBytes(res["hashfield_raw"])

        # 返回更新后的哈希字段
        return self.hashfield

    # 查找哈希 ID 对应的对等节点
    # 返回格式为：{hash1: ["ip:port", "ip:port",...],...}
    def findHashIds(self, hash_ids):
        # 通过请求获取哈希 ID 对应的对等节点
        res = self.request("findHashIds", {"site": self.site.address, "hash_ids": hash_ids})
        # 如果没有获取到结果或者结果中包含错误信息或者结果不是字典类型，则返回 False
        if not res or "error" in res or type(res) is not dict:
            return False

        # 创建默认值为列表的字典
        back = collections.defaultdict(list)

        # 遍历不同类型的 IP 地址（ipv4、ipv6、onion）
        for ip_type in ["ipv4", "ipv6", "onion"]:
            if ip_type == "ipv4":
                key = "peers"
            else:
                key = "peers_%s" % ip_type
            # 遍历每个哈希 ID 对应的对等节点，最多取前 30 个
            for hash, peers in list(res.get(key, {}).items())[0:30]:
                # 根据 IP 类型选择解包函数
                if ip_type == "onion":
                    unpacker_func = helper.unpackOnionAddress
                else:
                    unpacker_func = helper.unpackAddress

                # 将解包后的对等节点添加到对应的哈希 ID 中
                back[hash] += list(map(unpacker_func, peers))

        # 将自己的哈希 ID 添加到对应的哈希 ID 中
        for hash in res.get("my", []):
            if self.connection:
                back[hash].append((self.connection.ip, self.connection.port))
            else:
                back[hash].append((self.ip, self.port))

        # 返回包含对等节点的字典
        return back

    # 将自己的哈希字段发送给对等节点
    # 如果发送成功，则返回 True
    # 发送自己的哈希字段
    def sendMyHashfield(self):
        # 如果存在连接且握手信息中的版本小于510，则返回False，表示不支持
        if self.connection and self.connection.handshake.get("rev", 0) < 510:
            return False  # Not supported
        # 如果已经发送过自己的哈希字段，并且站点内容管理器中的哈希字段的修改时间小于等于上次发送的时间，则返回False，表示对方已经有最新的哈希字段
        if self.time_my_hashfield_sent and self.site.content_manager.hashfield.time_changed <= self.time_my_hashfield_sent:
            return False  # Peer already has the latest hashfield

        # 发送请求，设置哈希字段
        res = self.request("setHashfield", {"site": self.site.address, "hashfield_raw": self.site.content_manager.hashfield.tobytes()})
        # 如果没有响应或者响应中包含错误信息，则返回False
        if not res or "error" in res:
            return False
        else:
            # 更新发送哈希字段的时间，并返回True
            self.time_my_hashfield_sent = time.time()
            return True

    # 发布内容
    def publish(self, address, inner_path, body, modified, diffs=[]):
        # 如果内容长度超过10KB，并且存在连接且握手信息中的版本大于等于4095，则将内容置为空，以节省带宽
        if len(body) > 10 * 1024 and self.connection and self.connection.handshake.get("rev", 0) >= 4095:
            # To save bw we don't push big content.json to peers
            body = b""

        # 发送请求，更新内容
        return self.request("update", {
            "site": address,
            "inner_path": inner_path,
            "body": body,
            "modified": modified,
            "diffs": diffs
        })

    # 停止并从站点中移除
    def remove(self, reason="Removing"):
        # 记录日志，移除对等方，关闭连接
        self.log("Removing peer...Connection error: %s, Hash failed: %s" % (self.connection_error, self.hash_failed))
        if self.site and self.key in self.site.peers:
            del(self.site.peers[self.key])

        if self.site and self in self.site.peers_recent:
            self.site.peers_recent.remove(self)

        if self.connection:
            self.connection.close(reason)

    # - 事件 -

    # 连接错误时
    def onConnectionError(self, reason="Unknown"):
        # 增加连接错误次数，根据站点中对等方数量确定限制次数，减少声誉值，如果连接错误次数超过限制，则移除对等方
        self.connection_error += 1
        if self.site and len(self.site.peers) > 200:
            limit = 3
        else:
            limit = 6
        self.reputation -= 1
        if self.connection_error >= limit:  # Dead peer
            self.remove("Peer connection: %s" % reason)

    # 完成与对等方的工作
    # 定义一个方法，用于处理工作完成的情况
    def onWorkerDone(self):
        # 什么也不做，直接跳过
        pass
```
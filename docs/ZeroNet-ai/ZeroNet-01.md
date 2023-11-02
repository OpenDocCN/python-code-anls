# ZeroNet源码解析 1

# `plugins/__init__.py`

我需要您提供需要解释的代码，才能帮助您解释它的作用。


```py

```

# `plugins/AnnounceBitTorrent/AnnounceBitTorrentPlugin.py`

这段代码的作用是实现一个网络爬虫，用于从指定 URL 下载网页内容并获取相关信息。

具体来说，该代码使用 Python 的 `importlib` 函数导入了一系列库，包括 `time`、`urllib.request`、`struct`、`socket`、`lib.bencode_open`、`lib.subtl.subtl`、`libsocks`、`libsockshandler`、`gevent` 和 `Plugin`、`Config` 和 `Debug`、`util` 和 `Helper` 等库，其中 `lib.bencode_open` 和 `libsocks` 用于网页编码和解码，`libsockshandler` 用于处理 HTTP 请求和响应，`gevent` 用于创建事件循环，`Plugin` 和 `Config` 用于管理插件和配置，`Debug` 和 `Helper` 用于处理日志和错误信息，`util` 库用于一些通用的工具函数和数据结构。

具体实现过程如下：

1. 使用 `urllib.request.urlopen` 函数打开指定 URL，并获取响应对象。
2. 使用 `struct` 库中的 `unpack` 函数将响应对象中的 `bytes` 对象解包成一个字符串。
3. 将字符串中的内容使用 `lib.bencode_open` 库中的 `bdecode` 函数进行编码，得到一个二进制字符串。
4. 将编码后的字符串使用 `libsocks` 库中的 `socks` 函数创建一个 HTTP 连接，并使用该连接进行 HTTP 请求。
5. 使用 `gevent` 库中的 `get` 函数获取事件循环中的所有 `stdout` 事件，即所有从标准输出流（通常是终端）读取的数据。
6. 在事件循环中，使用 `sockshandler` 库中的 `handle` 函数处理收到的 HTTP 请求，其中 `handle` 函数的第一个参数是请求对象，第二个参数是一个编码后的数据对象，这些数据对象可以通过 `libsocks` 库中的 `socks` 函数获取。
7. 在处理完请求后，使用 `gevent` 库中的 `time_sleep` 函数休眠一段时间，然后继续循环处理其他事件。
8. 在循环过程中，使用 `lib.subtl.subtl` 库中的 `UdpTrackerClient` 类跟踪下载的网页内容，并在下载完成后将这些内容发送给用户。
9. 最后，使用 `Plugin` 和 `Config` 库中的 `PluginManager` 类来管理该爬虫的配置和状态，使用 `Debug` 和 `Helper` 库中的函数来处理下载过程中的日志和错误信息。


```py
import time
import urllib.request
import struct
import socket

import lib.bencode_open as bencode_open
from lib.subtl.subtl import UdpTrackerClient
import socks
import sockshandler
import gevent

from Plugin import PluginManager
from Config import config
from Debug import Debug
from util import helper


```

这是一个在K优管家系统中处理接收服务器列表的方法。它接受一个字符串参数tracker_ip，用于获取tracker服务器的位置。然后，它通过调用getOpenedServiceTypes方法获取开启的服务类型，并检查传来的tracker_ip是否与开启的服务类型中的地址匹配。如果不匹配，它会设置服务器端口为1，然后使用getIpType方法获取服务器ip类型，并将结果返回。

以下是方法的实现步骤：

1. 根据传入的tracker_ip，调用getOpenedServiceTypes方法获取开启的服务类型。
2. 如果tracker_ip与开启的服务类型中的地址匹配，那么将获取到的服务类型返回。
3. 如果tracker_ip与开启的服务类型中的地址不匹配，那么设置服务器端口为1，并将结果返回。
4. 如果返回的结果是列表，则遍历该列表，对于每个开启的服务类型，获取其对应的地址和端口，并将它们添加到peers列表中。
5. 如果函数执行成功，则返回peers列表。如果执行失败，则返回None。


```py
# We can only import plugin host clases after the plugins are loaded
@PluginManager.afterLoad
def importHostClasses():
    global Peer, AnnounceError
    from Peer import Peer
    from Site.SiteAnnouncer import AnnounceError


@PluginManager.registerTo("SiteAnnouncer")
class SiteAnnouncerPlugin(object):
    def getSupportedTrackers(self):
        trackers = super(SiteAnnouncerPlugin, self).getSupportedTrackers()
        if config.disable_udp or config.trackers_proxy != "disable":
            trackers = [tracker for tracker in trackers if not tracker.startswith("udp://")]

        return trackers

    def getTrackerHandler(self, protocol):
        if protocol == "udp":
            handler = self.announceTrackerUdp
        elif protocol == "http":
            handler = self.announceTrackerHttp
        elif protocol == "https":
            handler = self.announceTrackerHttps
        else:
            handler = super(SiteAnnouncerPlugin, self).getTrackerHandler(protocol)
        return handler

    def announceTrackerUdp(self, tracker_address, mode="start", num_want=10):
        s = time.time()
        if config.disable_udp:
            raise AnnounceError("Udp disabled by config")
        if config.trackers_proxy != "disable":
            raise AnnounceError("Udp trackers not available with proxies")

        ip, port = tracker_address.split("/")[0].split(":")
        tracker = UdpTrackerClient(ip, int(port))
        if helper.getIpType(ip) in self.getOpenedServiceTypes():
            tracker.peer_port = self.fileserver_port
        else:
            tracker.peer_port = 0
        tracker.connect()
        if not tracker.poll_once():
            raise AnnounceError("Could not connect")
        tracker.announce(info_hash=self.site.address_sha1, num_want=num_want, left=431102370)
        back = tracker.poll_once()
        if not back:
            raise AnnounceError("No response after %.0fs" % (time.time() - s))
        elif type(back) is dict and "response" in back:
            peers = back["response"]["peers"]
        else:
            raise AnnounceError("Invalid response: %r" % back)

        return peers

    def httpRequest(self, url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
            'Accept-Encoding': 'none',
            'Accept-Language': 'en-US,en;q=0.8',
            'Connection': 'keep-alive'
        }

        req = urllib.request.Request(url, headers=headers)

        if config.trackers_proxy == "tor":
            tor_manager = self.site.connection_server.tor_manager
            handler = sockshandler.SocksiPyHandler(socks.SOCKS5, tor_manager.proxy_ip, tor_manager.proxy_port)
            opener = urllib.request.build_opener(handler)
            return opener.open(req, timeout=50)
        elif config.trackers_proxy == "disable":
            return urllib.request.urlopen(req, timeout=25)
        else:
            proxy_ip, proxy_port = config.trackers_proxy.split(":")
            handler = sockshandler.SocksiPyHandler(socks.SOCKS5, proxy_ip, int(proxy_port))
            opener = urllib.request.build_opener(handler)
            return opener.open(req, timeout=50)

    def announceTrackerHttps(self, *args, **kwargs):
        kwargs["protocol"] = "https"
        return self.announceTrackerHttp(*args, **kwargs)

    def announceTrackerHttp(self, tracker_address, mode="start", num_want=10, protocol="http"):
        tracker_ip, tracker_port = tracker_address.rsplit(":", 1)
        if helper.getIpType(tracker_ip) in self.getOpenedServiceTypes():
            port = self.fileserver_port
        else:
            port = 1
        params = {
            'info_hash': self.site.address_sha1,
            'peer_id': self.peer_id, 'port': port,
            'uploaded': 0, 'downloaded': 0, 'left': 431102370, 'compact': 1, 'numwant': num_want,
            'event': 'started'
        }

        url = protocol + "://" + tracker_address + "?" + urllib.parse.urlencode(params)

        s = time.time()
        response = None
        # Load url
        if config.tor == "always" or config.trackers_proxy != "disable":
            timeout = 60
        else:
            timeout = 30

        with gevent.Timeout(timeout, False):  # Make sure of timeout
            req = self.httpRequest(url)
            response = req.read()
            req.close()
            req = None

        if not response:
            raise AnnounceError("No response after %.0fs" % (time.time() - s))

        # Decode peers
        try:
            peer_data = bencode_open.loads(response)[b"peers"]
            response = None
            peer_count = int(len(peer_data) / 6)
            peers = []
            for peer_offset in range(peer_count):
                off = 6 * peer_offset
                peer = peer_data[off:off + 6]
                addr, port = struct.unpack('!LH', peer)
                peers.append({"addr": socket.inet_ntoa(struct.pack('!L', addr)), "port": port})
        except Exception as err:
            raise AnnounceError("Invalid response: %r (%s)" % (response, Debug.formatException(err)))

        return peers

```

# `plugins/AnnounceBitTorrent/__init__.py`

这段代码是在导入名为"AnnounceBitTorrentPlugin"的模块，可能用于在PyTorch中管理BitTorrent种子或启发式下载。具体来说，它可能用于下载大型免费软件的源代码，或者下载部分受到保护的软件，并使用BitTorrent协议来加速下载过程。但是，除了知道这个模块的名称外，我们无法确定它的具体作用。


```py
from . import AnnounceBitTorrentPlugin
```

# `plugins/AnnounceLocal/AnnounceLocalPlugin.py`

这段代码定义了一个名为 SiteAnnouncerPlugin 的类，用于在 Web 站点中发布 announcement。具体来说，它实现了以下几个方法：

1. announce(self): 这个方法用于发布 announcement，可以传递一些参数来控制 announcement 的发布。其中，参数 force 表示是否强制发布 announcement，如果为 False，则不会发布 announcement。这个方法首先尝试使用服务器上的本地 annotation 函数，如果服务器上没有本地 annotation 函数，就会发布一个包含 announcement 的通知。在发布 announcement 后，会尝试等待一段时间再重复发布，这个时间间隔是 5 分钟。

2. discover(self, force=False): 这个方法用于在服务器上查找其他 annotation 服务器。如果服务器上没有其他 annotation 服务器，就会发布一个包含 announcement 的通知。如果服务器上有多个 annotation 服务器，就会尝试等待一段时间再选择其中一个来发布 announcement。

3. super(SiteAnnouncerPlugin, self): 这个方法重写了父类 SiteAnnouncerPlugin 的 announce 方法，用于在发布 announcement 前调用父类的相同方法。这个方法在 PluginManager 中注册，以便在需要时动态地加载。

这段代码的作用是在 Web 站点中发布 announcement，可以发布服务器上的本地 annotation 函数，如果服务器上没有本地 annotation 函数，就会发布一个包含 announcement 的通知。如果服务器上有多个 annotation 服务器，就会尝试等待一段时间再选择其中一个来发布 announcement。


```py
import time

import gevent

from Plugin import PluginManager
from Config import config
from . import BroadcastServer


@PluginManager.registerTo("SiteAnnouncer")
class SiteAnnouncerPlugin(object):
    def announce(self, force=False, *args, **kwargs):
        local_announcer = self.site.connection_server.local_announcer

        thread = None
        if local_announcer and (force or time.time() - local_announcer.last_discover > 5 * 60):
            thread = gevent.spawn(local_announcer.discover, force=force)
        back = super(SiteAnnouncerPlugin, self).announce(force=force, *args, **kwargs)

        if thread:
            thread.join()

        return back


```

This is a Python class that implements the Site Listenner. It listens for incoming events from a Site Manager server and sends back a response to the server every 100 site files have been modified.

The Site Listenner class has the following methods:

* siteList: This method is responsible for the site list response. It will parse the Site Manager server's response by comparing the sites that have been modified by the server to the known peers, and sends the modified site list back to the server.
* actionSiteListResponse: This method is responsible for sending the Site Manager server's site list response to the server. It will parse the Site Manager server's response, and send the response back to the server. It will also update the known peers and add the modified site list to the added_sites list.
* siteGroups: This method is responsible for generating the site groups by comparing the sites that have been modified by the server to the known peers.
* actionAddSite: This method is responsible for adding a site to the known peers. It will parse the Site Manager server's request and send the site to the known peers.
* actionRemoveSite: This method is responsible for removing a site from the known peers. It will parse the Site Manager server's request and remove the site from the known peers.
* siteManager: This is an instance of the Site Manager class that communicates with the Site Manager server. It is responsible for sending and receiving site commands from the server.
* server: This is an instance of the Server class that implements the Site Manager server. It is responsible for managing the Site Manager clients and sends site commands to them.
* sites: This is an instance of the Site class that represents a site in the Site Manager server. It contains information about the site, including its address hash, and has methods for adding and removing site peers, updating its websocket, and workers.


```py
class LocalAnnouncer(BroadcastServer.BroadcastServer):
    def __init__(self, server, listen_port):
        super(LocalAnnouncer, self).__init__("zeronet", listen_port=listen_port)
        self.server = server

        self.sender_info["peer_id"] = self.server.peer_id
        self.sender_info["port"] = self.server.port
        self.sender_info["broadcast_port"] = listen_port
        self.sender_info["rev"] = config.rev

        self.known_peers = {}
        self.last_discover = 0

    def discover(self, force=False):
        self.log.debug("Sending discover request (force: %s)" % force)
        self.last_discover = time.time()
        if force:  # Probably new site added, clean cache
            self.known_peers = {}

        for peer_id, known_peer in list(self.known_peers.items()):
            if time.time() - known_peer["found"] > 20 * 60:
                del(self.known_peers[peer_id])
                self.log.debug("Timeout, removing from known_peers: %s" % peer_id)
        self.broadcast({"cmd": "discoverRequest", "params": {}}, port=self.listen_port)

    def actionDiscoverRequest(self, sender, params):
        back = {
            "cmd": "discoverResponse",
            "params": {
                "sites_changed": self.server.site_manager.sites_changed
            }
        }

        if sender["peer_id"] not in self.known_peers:
            self.known_peers[sender["peer_id"]] = {"added": time.time(), "sites_changed": 0, "updated": 0, "found": time.time()}
            self.log.debug("Got discover request from unknown peer %s (%s), time to refresh known peers" % (sender["ip"], sender["peer_id"]))
            gevent.spawn_later(1.0, self.discover)  # Let the response arrive first to the requester

        return back

    def actionDiscoverResponse(self, sender, params):
        if sender["peer_id"] in self.known_peers:
            self.known_peers[sender["peer_id"]]["found"] = time.time()
        if params["sites_changed"] != self.known_peers.get(sender["peer_id"], {}).get("sites_changed"):
            # Peer's site list changed, request the list of new sites
            return {"cmd": "siteListRequest"}
        else:
            # Peer's site list is the same
            for site in self.server.sites.values():
                peer = site.peers.get("%s:%s" % (sender["ip"], sender["port"]))
                if peer:
                    peer.found("local")

    def actionSiteListRequest(self, sender, params):
        back = []
        sites = list(self.server.sites.values())

        # Split adresses to group of 100 to avoid UDP size limit
        site_groups = [sites[i:i + 100] for i in range(0, len(sites), 100)]
        for site_group in site_groups:
            res = {}
            res["sites_changed"] = self.server.site_manager.sites_changed
            res["sites"] = [site.address_hash for site in site_group]
            back.append({"cmd": "siteListResponse", "params": res})
        return back

    def actionSiteListResponse(self, sender, params):
        s = time.time()
        peer_sites = set(params["sites"])
        num_found = 0
        added_sites = []
        for site in self.server.sites.values():
            if site.address_hash in peer_sites:
                added = site.addPeer(sender["ip"], sender["port"], source="local")
                num_found += 1
                if added:
                    site.worker_manager.onPeers()
                    site.updateWebsocket(peers_added=1)
                    added_sites.append(site)

        # Save sites changed value to avoid unnecessary site list download
        if sender["peer_id"] not in self.known_peers:
            self.known_peers[sender["peer_id"]] = {"added": time.time()}

        self.known_peers[sender["peer_id"]]["sites_changed"] = params["sites_changed"]
        self.known_peers[sender["peer_id"]]["updated"] = time.time()
        self.known_peers[sender["peer_id"]]["found"] = time.time()

        self.log.debug(
            "Tracker result: Discover from %s response parsed in %.3fs, found: %s added: %s of %s" %
            (sender["ip"], time.time() - s, num_found, added_sites, len(peer_sites))
        )


```

这段代码定义了一个名为 FileServerPlugin 的类，用于注册到名为 "FileServer" 的插件中。这个插件允许在主服务器上监听来自客户端的文件传输请求。

在 FileServerPlugin 的构造函数中，初始化了一些 super 类方法和参数，包括构造函数和一些配置参数。然后检查是否有可用的广播端口、tor 设置，以及是否禁用 UDP 设置。如果是，就创建一个名为 LocalAnnouncer 的子类作为本地公告者，并在 start 方法中启动它。如果本地公告者创建失败，就直接返回。

在 start 方法中，如果本地公告者已经创建好，就创建一个 Gevent 事件循环并将本地公告者的 start 方法绑定到这个事件循环上。如果本地公告者没有创建好，就直接创建一个空的本地公告者并绑定到 start 方法。

在 stop 方法中，先调用父类的 stop 方法，然后停止本地公告者，如果本地公告者已经创建好，就调用本地公告者的 stop 方法。

通过这个插件，当在主服务器上运行时，它可以监听来自客户端的文件传输请求，并在本地创建一个公告者来接收这些请求，然后将它们发送到指定的端口。


```py
@PluginManager.registerTo("FileServer")
class FileServerPlugin(object):
    def __init__(self, *args, **kwargs):
        super(FileServerPlugin, self).__init__(*args, **kwargs)
        if config.broadcast_port and config.tor != "always" and not config.disable_udp:
            self.local_announcer = LocalAnnouncer(self, config.broadcast_port)
        else:
            self.local_announcer = None

    def start(self, *args, **kwargs):
        if self.local_announcer:
            gevent.spawn(self.local_announcer.start)
        return super(FileServerPlugin, self).start(*args, **kwargs)

    def stop(self):
        if self.local_announcer:
            self.local_announcer.stop()
        res = super(FileServerPlugin, self).stop()
        return res


```

这段代码是一个 Python 类，名为 `ConfigPlugin`。它是一个自定义插件，注册到名为 "ConfigPlugin" 的类中。

插件在加载时会创建一个 `argparse.ArgumentParser` 对象，然后添加一个名为 "AnnounceLocal plugin" 的参数组。接着，这个参数组添加一个名为 `--broadcast_port` 的参数，它的作用是指定一个用于本地对端发现的有用端口号。这个参数的默认值是 1544，类型是整数类型，使用 `metavar='port'` 的形式描述。

最后，`ConfigPlugin` 类的 `createArguments` 方法返回 `super` 方法返回的参数列表，这样就可以将参数正确地传递给 `argparse.ArgumentParser` 的 `add_argument` 方法，用于将参数添加到命令行工具中。


```py
@PluginManager.registerTo("ConfigPlugin")
class ConfigPlugin(object):
    def createArguments(self):
        group = self.parser.add_argument_group("AnnounceLocal plugin")
        group.add_argument('--broadcast_port', help='UDP broadcasting port for local peer discovery', default=1544, type=int, metavar='port')

        return super(ConfigPlugin, self).createArguments()

```

# `plugins/AnnounceLocal/BroadcastServer.py`

This is a simple implementation of a对策：IP-MAC Address Transaction (TLS) using the Flask microservice. The TLS message is sent between two endpoints (URLs) with two IP addresses, which are '22.22.22.22' and '111.111.111.111', respectively. The endpoints are accessible through a load balancer (url 'https://经历').

The endpoint for sending the TLS message is protected by the [TLS certificate](https://secure.meko.io/attestations/TLS/)-policy, which is installed automatically by the `以后` day-0c391st October 2021, as mentioned in the Flask documentation. The `以后`-day certificate is used for securing the endpoint.

The two main classes in this implementation are `MyBase` and `MyTLS`. `MyBase` is a base class for the common functionality, while `MyTLS` extends `MyBase` and adds the TLS functionality.

The `MyTLS` class has several methods: `connect`, `sendMessage`, and `handleMessage`.

* `connect`: Connects to the load balancer and establishes a TLS handshake with the load balancer. It also retrieves the remote IP address (22.22.22.22).
* `sendMessage`: Sends a TLS message to the load balancer. It uses the `以后`-day certificate for securing the message and adds the load balancer's IP address as a parameter.
* `handleMessage`: Handles the received TLS message. It receives the TLS message from the load balancer and the sender's IP address, which is stored in the `sender_info` object. It then passes the message to the appropriate function (method) depending on the `cmd` field in the message. If the sender is not known or the service is not 'zeronet', the message is not processed and discarded.


```py
import socket
import logging
import time
from contextlib import closing

from Debug import Debug
from util import UpnpPunch
from util import Msgpack


class BroadcastServer(object):
    def __init__(self, service_name, listen_port=1544, listen_ip=''):
        self.log = logging.getLogger("BroadcastServer")
        self.listen_port = listen_port
        self.listen_ip = listen_ip

        self.running = False
        self.sock = None
        self.sender_info = {"service": service_name}

    def createBroadcastSocket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, 'SO_REUSEPORT'):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except Exception as err:
                self.log.warning("Error setting SO_REUSEPORT: %s" % err)

        binded = False
        for retry in range(3):
            try:
                sock.bind((self.listen_ip, self.listen_port))
                binded = True
                break
            except Exception as err:
                self.log.error(
                    "Socket bind to %s:%s error: %s, retry #%s" %
                    (self.listen_ip, self.listen_port, Debug.formatException(err), retry)
                )
                time.sleep(retry)

        if binded:
            return sock
        else:
            return False

    def start(self):  # Listens for discover requests
        self.sock = self.createBroadcastSocket()
        if not self.sock:
            self.log.error("Unable to listen on port %s" % self.listen_port)
            return

        self.log.debug("Started on port %s" % self.listen_port)

        self.running = True

        while self.running:
            try:
                data, addr = self.sock.recvfrom(8192)
            except Exception as err:
                if self.running:
                    self.log.error("Listener receive error: %s" % err)
                continue

            if not self.running:
                break

            try:
                message = Msgpack.unpack(data)
                response_addr, message = self.handleMessage(addr, message)
                if message:
                    self.send(response_addr, message)
            except Exception as err:
                self.log.error("Handlemessage error: %s" % Debug.formatException(err))
        self.log.debug("Stopped listening on port %s" % self.listen_port)

    def stop(self):
        self.log.debug("Stopping, socket: %s" % self.sock)
        self.running = False
        if self.sock:
            self.sock.close()

    def send(self, addr, message):
        if type(message) is not list:
            message = [message]

        for message_part in message:
            message_part["sender"] = self.sender_info

            self.log.debug("Send to %s: %s" % (addr, message_part["cmd"]))
            with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(Msgpack.pack(message_part), addr)

    def getMyIps(self):
        return UpnpPunch._get_local_ips()

    def broadcast(self, message, port=None):
        if not port:
            port = self.listen_port

        my_ips = self.getMyIps()
        addr = ("255.255.255.255", port)

        message["sender"] = self.sender_info
        self.log.debug("Broadcast using ips %s on port %s: %s" % (my_ips, port, message["cmd"]))

        for my_ip in my_ips:
            try:
                with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                    sock.bind((my_ip, 0))
                    sock.sendto(Msgpack.pack(message), addr)
            except Exception as err:
                self.log.warning("Error sending broadcast using ip %s: %s" % (my_ip, err))

    def handleMessage(self, addr, message):
        self.log.debug("Got from %s: %s" % (addr, message["cmd"]))
        cmd = message["cmd"]
        params = message.get("params", {})
        sender = message["sender"]
        sender["ip"] = addr[0]

        func_name = "action" + cmd[0].upper() + cmd[1:]
        func = getattr(self, func_name, None)

        if sender["service"] != "zeronet" or sender["peer_id"] == self.sender_info["peer_id"]:
            # Skip messages not for us or sent by us
            message = None
        elif func:
            message = func(sender, params)
        else:
            self.log.debug("Unknown cmd: %s" % cmd)
            message = None

        return (sender["ip"], sender["broadcast_port"]), message

```

# `plugins/AnnounceLocal/__init__.py`

这段代码是在导入名为"AnnounceLocalPlugin"的第三方库，可能用于在Python应用程序中发送本地通知。具体来说，它可能用于在应用程序启动时显示本地通知，或者在应用程序中发送本地通知时使用。但是，除了提供代码本身，我无法提供更多信息。


```py
from . import AnnounceLocalPlugin
```

# `plugins/AnnounceLocal/Test/conftest.py`

这段代码是一个Python程序，它的作用是定义一个名为“config”的配置对象，并设置其名为“broadcast_port”的属性为0。

具体来说，首先从名为“src.Test.conftest”的导入中导入了所有定义在其中的函数和类。然后，定义了一个名为“config”的配置对象，该对象的类型未指定。

接下来，通过调用名为“config.broadcast_port”的属性，将0赋值给“config.broadcast_port”属性。这样做的话，这个配置对象将保留对“broadcast_port”属性的引用，即使程序卸载，该属性的值在程序重新加载时仍然保留。


```py
from src.Test.conftest import *

from Config import config
config.broadcast_port = 0

```

# `plugins/AnnounceLocal/Test/TestAnnounce.py`

这段代码定义了一个测试函数，使用名称空间抽象（namespace abstraction）的方式，导入了多种库，并定义了一个 announcing object，该 object 在测试函数中作为参数传入。这个 announcing object 通过一个 file server 来发布消息，该 file server 拥有一个测试站点，通过一个 spy 来模拟文件服务器的行为，在测试函数中，还引入了 AnnounceLocalPlugin 和 FileServer 两个库，用来实现发布消息的功能和模拟文件服务器的行为。在测试函数中，使用 spy 来模拟 AnnounceLocalPlugin 的行为，通过 gevent 库来保证 spy 延迟执行，在模拟 AnnounceLocalPlugin 行为的同时，还模拟了 FileServer 的行为，通过引入的 naming spy，来模拟文件服务器发布消息的行为。通过在 announcing object 上调用 start() 方法来启动 announcing 进程，并在之后使用 time.sleep() 方法来模拟一个短暂的延迟，然后通过 assert 断言 file_server.local_announcer.running 来验证 announcing 进程是否正常运行。


```py
import time
import copy

import gevent
import pytest
import mock

from AnnounceLocal import AnnounceLocalPlugin
from File import FileServer
from Test import Spy

@pytest.fixture
def announcer(file_server, site):
    file_server.sites[site.address] = site
    announcer = AnnounceLocalPlugin.LocalAnnouncer(file_server, listen_port=1100)
    file_server.local_announcer = announcer
    announcer.listen_port = 1100
    announcer.sender_info["broadcast_port"] = 1100
    announcer.getMyIps = mock.MagicMock(return_value=["127.0.0.1"])
    announcer.discover = mock.MagicMock(return_value=False)  # Don't send discover requests automatically
    gevent.spawn(announcer.start)
    time.sleep(0.5)

    assert file_server.local_announcer.running
    return file_server.local_announcer

```

这段代码定义了一个名为 `announcer_remote` 的 pytest fixture，用于模拟远程 announcement 服务。该 fixture 接受两个参数：`request` 和 `site_temp`。

首先，它使用 `FileServer` 类创建一个远程文件服务器，并将其存储在 `site_temp` 对象中。然后，它创建一个 `AnnounceLocalPlugin.LocalAnnouncer` 实例，并将其与远程文件服务器关联。接着，它将远程文件服务器设置为 `listen_port` 并将其设置为 `1101`。

然后，它创建一个 `Announcer` 实例，并将其与 `local_announcer` 属性关联。它还将其 `sites` 属性设置为 `site_temp`，以便在启动时使用 `announcer.sites` 访问远程服务器。

接下来，它使用 `spawn` 函数启动 `announcer`。在 `spawn` 函数回调中，它等待 0.5 秒钟，以确保 `announcer` 正常运行。然后，它使用 `assert` 确保 `file_server_remote.local_announcer` 正在运行。

最后，它在 `request.addfinalizer` 函数中添加了 `cleanup` 函数，该函数在测试结束后调用，并关闭远程文件服务器。


```py
@pytest.fixture
def announcer_remote(request, site_temp):
    file_server_remote = FileServer("127.0.0.1", 1545)
    file_server_remote.sites[site_temp.address] = site_temp
    announcer = AnnounceLocalPlugin.LocalAnnouncer(file_server_remote, listen_port=1101)
    file_server_remote.local_announcer = announcer
    announcer.listen_port = 1101
    announcer.sender_info["broadcast_port"] = 1101
    announcer.getMyIps = mock.MagicMock(return_value=["127.0.0.1"])
    announcer.discover = mock.MagicMock(return_value=False)  # Don't send discover requests automatically
    gevent.spawn(announcer.start)
    time.sleep(0.5)

    assert file_server_remote.local_announcer.running

    def cleanup():
        file_server_remote.stop()
    request.addfinalizer(cleanup)


    return file_server_remote.local_announcer

```



This is a Python test case that checks the functionality of the `discoverPeer` and `siteListPeer` methods in an `Announcer` object. The `discoverPeer` method sends a `discoverRequest` message to a randomly selected remote server, and the `siteListPeer` method sends a `siteListRequest` message to a randomly selected remote server and returns the list of sites that have recently connected to the server.

The `testPeerDiscover` method checks that the `discoverPeer` method sends the expected message and waits for the expected response. It also checks that the remote server has a unique `peer_id` and that the list of `sites` is empty before sending the `discoverRequest` message.

The `testRecentPeerList` method checks that the `siteListPeer` method sends the expected messages and waits for the expected response. It also checks that the remote server has a unique `peer_id` and that the list of `sites_recent` is empty before sending the `siteListRequest` message. It also updates the `site_peers_recent` list to reflect the recent connections and checks that the updated list has only one element.

If the tests pass, it should be possible to use the `discoverPeer` and `siteListPeer` methods in an `Announcer` object without any errors.


```py
@pytest.mark.usefixtures("resetSettings")
@pytest.mark.usefixtures("resetTempSettings")
class TestAnnounce:
    def testSenderInfo(self, announcer):
        sender_info = announcer.sender_info
        assert sender_info["port"] > 0
        assert len(sender_info["peer_id"]) == 20
        assert sender_info["rev"] > 0

    def testIgnoreSelfMessages(self, announcer):
        # No response to messages that has same peer_id as server
        assert not announcer.handleMessage(("0.0.0.0", 123), {"cmd": "discoverRequest", "sender": announcer.sender_info, "params": {}})[1]

        # Response to messages with different peer id
        sender_info = copy.copy(announcer.sender_info)
        sender_info["peer_id"] += "-"
        addr, res = announcer.handleMessage(("0.0.0.0", 123), {"cmd": "discoverRequest", "sender": sender_info, "params": {}})
        assert res["params"]["sites_changed"] > 0

    def testDiscoverRequest(self, announcer, announcer_remote):
        assert len(announcer_remote.known_peers) == 0
        with Spy.Spy(announcer_remote, "handleMessage") as responses:
            announcer_remote.broadcast({"cmd": "discoverRequest", "params": {}}, port=announcer.listen_port)
            time.sleep(0.1)

        response_cmds = [response[1]["cmd"] for response in responses]
        assert response_cmds == ["discoverResponse", "siteListResponse"]
        assert len(responses[-1][1]["params"]["sites"]) == 1

        # It should only request siteList if sites_changed value is different from last response
        with Spy.Spy(announcer_remote, "handleMessage") as responses:
            announcer_remote.broadcast({"cmd": "discoverRequest", "params": {}}, port=announcer.listen_port)
            time.sleep(0.1)

        response_cmds = [response[1]["cmd"] for response in responses]
        assert response_cmds == ["discoverResponse"]

    def testPeerDiscover(self, announcer, announcer_remote, site):
        assert announcer.server.peer_id != announcer_remote.server.peer_id
        assert len(list(announcer.server.sites.values())[0].peers) == 0
        announcer.broadcast({"cmd": "discoverRequest"}, port=announcer_remote.listen_port)
        time.sleep(0.1)
        assert len(list(announcer.server.sites.values())[0].peers) == 1

    def testRecentPeerList(self, announcer, announcer_remote, site):
        assert len(site.peers_recent) == 0
        assert len(site.peers) == 0
        with Spy.Spy(announcer, "handleMessage") as responses:
            announcer.broadcast({"cmd": "discoverRequest", "params": {}}, port=announcer_remote.listen_port)
            time.sleep(0.1)
        assert [response[1]["cmd"] for response in responses] == ["discoverResponse", "siteListResponse"]
        assert len(site.peers_recent) == 1
        assert len(site.peers) == 1

        # It should update peer without siteListResponse
        last_time_found = list(site.peers.values())[0].time_found
        site.peers_recent.clear()
        with Spy.Spy(announcer, "handleMessage") as responses:
            announcer.broadcast({"cmd": "discoverRequest", "params": {}}, port=announcer_remote.listen_port)
            time.sleep(0.1)
        assert [response[1]["cmd"] for response in responses] == ["discoverResponse"]
        assert len(site.peers_recent) == 1
        assert list(site.peers.values())[0].time_found > last_time_found



```

# `plugins/AnnounceShare/AnnounceSharePlugin.py`



This is a class called `FileManager`, which appears to manage the synchronization of file trackers on a Figma server. It has methods to load, save, and discover trackers.

The `load` method reads the contents of the file specified by `file_path` and loads it into the `file_content` attribute. It then retrieves a list of trackers and iterates through them, updating the `num_error` attribute of each tracker. If there are any trackers with errors, the method returns.

The `save` method writes the contents of the `file_content` to the specified file path. It also updates the number of trackers that have been successfully loaded.

The `discoverTrackers` method takes a list of peers and attempts to load each tracker from the specified peers. If there are not enough trackers from the peers, the method returns. Otherwise, it saves the file content to disk and updates the number of successfully loaded trackers in the log.

Note that the class assumes that there is a `FileManager` instance that has been initialized before being called, and that the initialization of the `FileManager` instance is atomic. It also assumes that the `getWorkingTrackers` method has been implemented to retrieve trackers from the local file system or another specified storage.


```py
import time
import os
import logging
import json
import atexit

import gevent

from Config import config
from Plugin import PluginManager
from util import helper


class TrackerStorage(object):
    def __init__(self):
        self.log = logging.getLogger("TrackerStorage")
        self.file_path = "%s/trackers.json" % config.data_dir
        self.load()
        self.time_discover = 0.0
        atexit.register(self.save)

    def getDefaultFile(self):
        return {"shared": {}}

    def onTrackerFound(self, tracker_address, type="shared", my=False):
        if not tracker_address.startswith("zero://"):
            return False

        trackers = self.getTrackers()
        added = False
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

        trackers[tracker_address]["time_found"] = time.time()
        trackers[tracker_address]["my"] = my
        return added

    def onTrackerSuccess(self, tracker_address, latency):
        trackers = self.getTrackers()
        if tracker_address not in trackers:
            return False

        trackers[tracker_address]["latency"] = latency
        trackers[tracker_address]["time_success"] = time.time()
        trackers[tracker_address]["num_error"] = 0

    def onTrackerError(self, tracker_address):
        trackers = self.getTrackers()
        if tracker_address not in trackers:
            return False

        trackers[tracker_address]["time_error"] = time.time()
        trackers[tracker_address]["num_error"] += 1

        if len(self.getWorkingTrackers()) >= config.working_shared_trackers_limit:
            error_limit = 5
        else:
            error_limit = 30
        error_limit

        if trackers[tracker_address]["num_error"] > error_limit and trackers[tracker_address]["time_success"] < time.time() - 60 * 60:
            self.log.debug("Tracker %s looks down, removing." % tracker_address)
            del trackers[tracker_address]

    def getTrackers(self, type="shared"):
        return self.file_content.setdefault(type, {})

    def getWorkingTrackers(self, type="shared"):
        trackers = {
            key: tracker for key, tracker in self.getTrackers(type).items()
            if tracker["time_success"] > time.time() - 60 * 60
        }
        return trackers

    def getFileContent(self):
        if not os.path.isfile(self.file_path):
            open(self.file_path, "w").write("{}")
            return self.getDefaultFile()
        try:
            return json.load(open(self.file_path))
        except Exception as err:
            self.log.error("Error loading trackers list: %s" % err)
            return self.getDefaultFile()

    def load(self):
        self.file_content = self.getFileContent()

        trackers = self.getTrackers()
        self.log.debug("Loaded %s shared trackers" % len(trackers))
        for address, tracker in list(trackers.items()):
            tracker["num_error"] = 0
            if not address.startswith("zero://"):
                del trackers[address]

    def save(self):
        s = time.time()
        helper.atomicWrite(self.file_path, json.dumps(self.file_content, indent=2, sort_keys=True).encode("utf8"))
        self.log.debug("Saved in %.3fs" % (time.time() - s))

    def discoverTrackers(self, peers):
        if len(self.getWorkingTrackers()) > config.working_shared_trackers_limit:
            return False
        s = time.time()
        num_success = 0
        for peer in peers:
            if peer.connection and peer.connection.handshake.get("rev", 0) < 3560:
                continue  # Not supported

            res = peer.request("getTrackers")
            if not res or "error" in res:
                continue

            num_success += 1
            for tracker_address in res["trackers"]:
                if type(tracker_address) is bytes:  # Backward compatibilitys
                    tracker_address = tracker_address.decode("utf8")
                added = self.onTrackerFound(tracker_address)
                if added:  # Only add one tracker from one source
                    break

        if not num_success and len(peers) < 20:
            self.time_discover = 0.0

        if num_success:
            self.save()

        self.log.debug("Trackers discovered from %s/%s peers in %.3fs" % (num_success, len(peers), time.time() - s))


```

这段代码是一个Python类，名为“SiteAnnouncerPlugin”。它是一个插件，用于向用户的系统 announcement 列表中添加新的 tracker。

具体来说，这段代码做以下几件事情：

1. 检查 "tracker_storage" 是否在当前场景的 local变量中，如果不存在，则创建一个名为 "tracker_storage" 的实例，并将其命名为 "SiteAnnouncerPlugin"。
2. 在插件的 "getTrackers" 方法中，判断 tracker_storage.time_discover 是否在 5 分钟之前，如果是，则重置 time_discover，并使用 discoverer 方法发现 trackers。
3. 在插件的 "announceTracker" 方法中，发现并调用父类的 announceTracker 方法，该方法将在参数中传递 tracker 和系统给定的 arguments，并返回结果。
4. 在 "announceTracker" 方法的回调中，如果结果为 False，则调用 tracker_storage.onTrackerError 方法，将错误信息发送到 tracker。
5. 在 "getTrackers" 方法的回调中，如果 shared_trackers 变量存在，则将其添加到 trackers 列表中。

概括一下，这段代码实现了一个简单的 tracker 插件，用于添加新的 tracker 到用户的系统 announcement 列表中。


```py
if "tracker_storage" not in locals():
    tracker_storage = TrackerStorage()


@PluginManager.registerTo("SiteAnnouncer")
class SiteAnnouncerPlugin(object):
    def getTrackers(self):
        if tracker_storage.time_discover < time.time() - 5 * 60:
            tracker_storage.time_discover = time.time()
            gevent.spawn(tracker_storage.discoverTrackers, self.site.getConnectedPeers())
        trackers = super(SiteAnnouncerPlugin, self).getTrackers()
        shared_trackers = list(tracker_storage.getTrackers("shared").keys())
        if shared_trackers:
            return trackers + shared_trackers
        else:
            return trackers

    def announceTracker(self, tracker, *args, **kwargs):
        res = super(SiteAnnouncerPlugin, self).announceTracker(tracker, *args, **kwargs)
        if res:
            latency = res
            tracker_storage.onTrackerSuccess(tracker, latency)
        elif res is False:
            tracker_storage.onTrackerError(tracker)

        return res


```

这段代码定义了两个插件，一个是名为 "FileRequestPlugin" 的类，另一个是名为 "FileServerPlugin" 的类。这两个插件的作用是分别是在文件请求和文件服务器中处理客户端请求。

"FileRequestPlugin" 插件的 "actionGetTrackers" 方法会从 "shared_trackers" 列表中获取所有共同的跟踪器，然后将它们作为 JSON 响应返回。

"FileServerPlugin" 插件的 "portCheck" 方法会在文件服务器端检查客户端连接。如果连接成功，它会在配置中设置 "always" 选项，并且 "Bootstrapper" 选项也是正确的。此外，该插件还会遍历外部 IP 地址列表，并且对于每个 IP 地址，它将创建一个客户端跟踪器并将其连接到服务器上的 "zero://%s:%s" 地址。这将触发插件的 "onTrackerFound" 方法，并将其添加到服务器端的跟踪器存储中。


```py
@PluginManager.registerTo("FileRequest")
class FileRequestPlugin(object):
    def actionGetTrackers(self, params):
        shared_trackers = list(tracker_storage.getWorkingTrackers("shared").keys())
        self.response({"trackers": shared_trackers})


@PluginManager.registerTo("FileServer")
class FileServerPlugin(object):
    def portCheck(self, *args, **kwargs):
        res = super(FileServerPlugin, self).portCheck(*args, **kwargs)
        if res and not config.tor == "always" and "Bootstrapper" in PluginManager.plugin_manager.plugin_names:
            for ip in self.ip_external_list:
                my_tracker_address = "zero://%s:%s" % (ip, config.fileserver_port)
                tracker_storage.onTrackerFound(my_tracker_address, my=True)
        return res


```

这段代码定义了一个名为 "ConfigPlugin" 的类，它实现了 `IPlugin` 接口。这个类有一个 `createArguments` 方法，用于生成命令行参数的列表。

具体来说，这段代码注册了一个名为 "ConfigPlugin" 的插件，该插件属于 "AnnounceShare" 类别。插件在启动时会生成一组命令行参数，其中包含一个名为 `--working_shared_trackers_limit` 的选项。这个选项用于指定在发现到第几个新的共享跟踪器之后，将停止注册新的共享跟踪器。

每当 "ConfigPlugin" 类创建一个新的命令行参数时，它都会调用 `IPlugin` 的 `createArguments` 方法来生成命令行参数列表。在这个命令行参数列表中，"ConfigPlugin" 类成员可以访问生成的参数，并将它们传递给 `IPlugin` 的 `createOptionStrings` 方法，以便将选项和选项值存储在 `Option` 对象中。

最后，这段代码使用 `@PluginManager.registerTo("ConfigPlugin")` 注解注册了这个插件到 "AnnounceShare" 类别中。这样，在命令行中使用 `announce-share` 命令时，就可以使用生成的命令行参数来指定选项和选项值了。


```py
@PluginManager.registerTo("ConfigPlugin")
class ConfigPlugin(object):
    def createArguments(self):
        group = self.parser.add_argument_group("AnnounceShare plugin")
        group.add_argument('--working_shared_trackers_limit', help='Stop discovering new shared trackers after this number of shared trackers reached', default=5, type=int, metavar='limit')

        return super(ConfigPlugin, self).createArguments()

```

# `plugins/AnnounceShare/__init__.py`

这段代码是在导入名为"AnnounceSharePlugin"的类，可能是一个用于在某些网站或应用程序中进行公告或分享功能的插件。具体来说，这个插件可能用于在用户完成某些操作后，将他们的行动结果分享到社交媒体或新闻网站上。但具体功能和实现可能因上下文和使用场景不同而有所不同。


```py
from . import AnnounceSharePlugin

```

# `plugins/AnnounceShare/Test/conftest.py`

这段代码是一个Python程序，它的作用是定义一个名为“conftest”的包，以及从名为“Config”的包中导入一个名为“config”的配置对象。

具体来说，这段代码的作用是创建一个名为“conftest”的包，其中包含了可以使用在任何测试场景中的测试帮助函数和断言。同时，它还从名为“Config”的包中导入了一个名为“config”的配置对象，这个配置对象可以用于在整个程序中设置或读取配置。

由于这段代码并没有对任何文件或数据进行读写操作，因此它对程序的运行环境没有具体的作用。但是，在测试驱动开发（TDD）的过程中，这段代码可以提供一个通用的框架，使得测试人员在编写测试之前定义一些通用的功能和断言，从而提高测试代码的可读性和可维护性。


```py
from src.Test.conftest import *

from Config import config

```

# `plugins/AnnounceShare/Test/TestAnnounceShare.py`

这段代码是用来测试AnnounceShare组件的功能的。首先，通过使用pytest的mark.usefixtures("resetSettings")和mark.usefixtures("resetTempSettings")来保证在测试过程中不会保存任何默认设置。

在测试中，首先导入了pytest、AnnounceShare组件、Peer组件和Config组件。然后定义了一个TestAnnounceShare类，其中包含四个测试方法。

在测试方法中，使用file_server.ip来获取文件服务器的外部IP地址。接着，使用Peer组件的构造函数来创建一个与文件服务器连接的Peer对象，并使用其request方法来调用AnnounceShare组件中的onTrackerFound和onTrackerSuccess方法。

在onTrackerFound方法中，当接收到一个Tracker对象时，会将其添加到已经连接的Tracker列表中。然后，使用请求来的IP地址和端口来获取Tracker列表，并检查其中是否包含已知的Tracker对象。在这个测试中，我们期望 Tracker列表为空。

在onTrackerSuccess方法中，当Tracker连接成功并且版本号1.0时，会将该Tracker添加到已经连接的Tracker列表中。然后，使用请求来的IP地址和端口来获取Tracker列表，并检查其中是否包含已知的Tracker对象。在这个测试中，我们期望 Tracker列表为{"zero://%s:15441" % file_server.ip:1.0}。

通过调用AnnounceShare组件中的这些方法，我们可以测试AnnounceShare组件的功能，包括Tracker列表、连接Tracker和声明Tracker。


```py
import pytest

from AnnounceShare import AnnounceSharePlugin
from Peer import Peer
from Config import config


@pytest.mark.usefixtures("resetSettings")
@pytest.mark.usefixtures("resetTempSettings")
class TestAnnounceShare:
    def testAnnounceList(self, file_server):
        open("%s/trackers.json" % config.data_dir, "w").write("{}")
        tracker_storage = AnnounceSharePlugin.tracker_storage
        tracker_storage.load()
        peer = Peer(file_server.ip, 1544, connection_server=file_server)
        assert peer.request("getTrackers")["trackers"] == []

        tracker_storage.onTrackerFound("zero://%s:15441" % file_server.ip)
        assert peer.request("getTrackers")["trackers"] == []

        # It needs to have at least one successfull announce to be shared to other peers
        tracker_storage.onTrackerSuccess("zero://%s:15441" % file_server.ip, 1.0)
        assert peer.request("getTrackers")["trackers"] == ["zero://%s:15441" % file_server.ip]


```

# `plugins/AnnounceZero/AnnounceZeroPlugin.py`

这段代码是一个插件，它用于插件系统，允许插件在加载时导入其类。具体来说，它做了以下几件事情：

1. 导入了time、itertools、Crypto和CryptoRsa库，这些库可能用于插件的依赖或者实现插件的功能。

2. 定义了一个allow_reload变量，其值为False，表示当前插件不会允许源代码重新加载。

3. 定义了一个time_full_announced变量，用于存储所有已经宣布过的全局变量，这些变量将在插件加载时被初始化。

4. 定义了一个connection_pool变量，用于存储所有已经连接过的 peers，这些peers将用于跟踪已经建立的peer connection。

5. 在PluginManager的afterLoad函数中执行了以下操作：

- 检查allow_reload变量，如果为False，则表示当前插件不会允许源代码重新加载，这样可以防止插件在加载时出现问题。

- 导入了Peer和AnnounceError类，这些类将用于插件的实现。

- 定义了一个AnnounceError类，用于在插件加载时处理宣布错误。

- 在ConnectionPool的构造函数中，创建了一个空的ConnectionPool，用于存储所有已经建立的peer connection。

- 在PluginManager的执行函数中，创建了一个ConnectedPeer类，用于存储已经连接过的peer objects。

- 在execute函数中，使用了以下代码来建立peer connection:


   conn = ConnectionPool.get_connection()
   if conn:
       conn.add_event_listener(AnnounceError, "site_announcement_error")
       conn.add_event_listener(AnnounceError, "notify_announcement_error")
       conn.add_event_listener(AnnounceError, "announce_error")
       conn.add_event_listener(AnnounceError, "announce_success")
       conn.add_event_listener(AnnounceError, "announce_failure")


这些函数和类用于插件的实现，具体包括：

- `PluginManager`类用于管理插件加载和卸载的过程，包括加载类、执行函数等。
- `peer`类用于建立和维护与插件进行交互的peer connections。
- `AnnounceError`类用于在插件宣布错误时进行处理。
- `ConnectionPool`类用于管理所有的peer connection，包括建立和获取已经建立的关系。
- `execute`函数用于执行插件的操作，包括建立peer connection、处理announce error等。


```py
import time
import itertools

from Plugin import PluginManager
from util import helper
from Crypt import CryptRsa

allow_reload = False  # No source reload supported in this plugin
time_full_announced = {}  # Tracker address: Last announced all site to tracker
connection_pool = {}  # Tracker address: Peer object


# We can only import plugin host clases after the plugins are loaded
@PluginManager.afterLoad
def importHostClasses():
    global Peer, AnnounceError
    from Peer import Peer
    from Site.SiteAnnouncer import AnnounceError


```

该代码是一个 Python 函数，名为 `processPeerRes`，其作用是处理与跟踪器（即协调器）通信得到的邻居结果。以下是功能解释：

1. 函数接收三个参数：`tracker_address`（跟踪器的地址）、`site`（要连接到的跟踪器）和 `peers`（跟踪器的邻居，包含两个键：'onion' 和 'ip4' 或 'ipv6'）。
2. 从 `peers` 字典中获取 `ip4` 或 `ipv6` 链中的邻居。
3. 对于每个邻居，处理函数检查跟踪器是否已经将其添加到其上了。如果是，就增加已添加的数量。如果是，则将其添加到 `added` 变量中。
4. 对于每个添加到的邻居，调用 `site.worker_manager.onPeers()` 方法。
5. 调用 `site.updateWebsocket(peers_added=added)` 方法来通知跟踪器添加的邻居数量。
6. 如果添加的数量大于0，则将其输出。

函数中的 `found_onion` 变量用于跟踪所有已发现的可用于 `onion` 协议的地址数量。

`found_ipv4` 变量用于跟踪所有已发现的 IPv4 地址数量。

`peers_normal` 变量包含所有可能的 IPv4 地址，通过 `itertools.chain()` 方法将其连接成一个列表，然后再遍历 `peers` 字典以获取每个地址并处理。

`site.addPeer` 方法用于将给定地址添加到跟踪器中。注意，这个方法将允许添加 `ipv4` 或 `ipv6` 地址。

`tracker_address` 参数是给协调器发送的地址。

`site` 参数是一个用于存储要连接的跟踪器的变量。

`peers` 参数是一个字典，包含两个键：'onion' 和 'ip4' 或 'ipv6'。这是协调器要处理的结果，包含跟踪器连接的邻居的元数据。


```py
# Process result got back from tracker
def processPeerRes(tracker_address, site, peers):
    added = 0

    # Onion
    found_onion = 0
    for packed_address in peers["onion"]:
        found_onion += 1
        peer_onion, peer_port = helper.unpackOnionAddress(packed_address)
        if site.addPeer(peer_onion, peer_port, source="tracker"):
            added += 1

    # Ip4
    found_ipv4 = 0
    peers_normal = itertools.chain(peers.get("ip4", []), peers.get("ipv4", []), peers.get("ipv6", []))
    for packed_address in peers_normal:
        found_ipv4 += 1
        peer_ip, peer_port = helper.unpackAddress(packed_address)
        if site.addPeer(peer_ip, peer_port, source="tracker"):
            added += 1

    if added:
        site.worker_manager.onPeers()
        site.updateWebsocket(peers_added=added)
    return added


```

This is a Python implementation of the tracker_announce() function that is used to add tracker connections to a known tracker server. The function takes a tracker address, a list of site IDs, a list of peer resources, and a mode. It returns True if the annotation is successful, or False.

The function starts by defining some constants and variables. It then loops through the tracker address, site IDs, and the corresponding peer resources. For each peer resource, it extracts the site ID and the onion address from the peer resource.

Next, it checks if the tracker address is present in the res["peers"] dictionary. If it is, it gets the corresponding site from the sites list and adds the peer resources to the processPeerRes function.

Finally, it checks if it needs to sign the peer addresses and adds the onions to the site's log.

It also checks if the onion_sign\_this key is present in res, if it is it creates a dictionary which sign by signing prove the onion addresses.

It then returns the value of res["onion_sign_this"] and also return True if it is present in res, otherwise False.


```py
@PluginManager.registerTo("SiteAnnouncer")
class SiteAnnouncerPlugin(object):
    def getTrackerHandler(self, protocol):
        if protocol == "zero":
            return self.announceTrackerZero
        else:
            return super(SiteAnnouncerPlugin, self).getTrackerHandler(protocol)

    def announceTrackerZero(self, tracker_address, mode="start", num_want=10):
        global time_full_announced
        s = time.time()

        need_types = ["ip4"]   # ip4 for backward compatibility reasons
        need_types += self.site.connection_server.supported_ip_types
        if self.site.connection_server.tor_manager.enabled:
            need_types.append("onion")

        if mode == "start" or mode == "more":  # Single: Announce only this site
            sites = [self.site]
            full_announce = False
        else:  # Multi: Announce all currently serving site
            full_announce = True
            if time.time() - time_full_announced.get(tracker_address, 0) < 60 * 15:  # No reannounce all sites within short time
                return None
            time_full_announced[tracker_address] = time.time()
            from Site import SiteManager
            sites = [site for site in SiteManager.site_manager.sites.values() if site.isServing()]

        # Create request
        add_types = self.getOpenedServiceTypes()
        request = {
            "hashes": [], "onions": [], "port": self.fileserver_port, "need_types": need_types, "need_num": 20, "add": add_types
        }
        for site in sites:
            if "onion" in add_types:
                onion = self.site.connection_server.tor_manager.getOnion(site.address)
                request["onions"].append(onion)
            request["hashes"].append(site.address_hash)

        # Tracker can remove sites that we don't announce
        if full_announce:
            request["delete"] = True

        # Sent request to tracker
        tracker_peer = connection_pool.get(tracker_address)  # Re-use tracker connection if possible
        if not tracker_peer:
            tracker_ip, tracker_port = tracker_address.rsplit(":", 1)
            tracker_peer = Peer(str(tracker_ip), int(tracker_port), connection_server=self.site.connection_server)
            tracker_peer.is_tracker_connection = True
            connection_pool[tracker_address] = tracker_peer

        res = tracker_peer.request("announce", request)

        if not res or "peers" not in res:
            if full_announce:
                time_full_announced[tracker_address] = 0
            raise AnnounceError("Invalid response: %s" % res)

        # Add peers from response to site
        site_index = 0
        peers_added = 0
        for site_res in res["peers"]:
            site = sites[site_index]
            peers_added += processPeerRes(tracker_address, site, site_res)
            site_index += 1

        # Check if we need to sign prove the onion addresses
        if "onion_sign_this" in res:
            self.site.log.debug("Signing %s for %s to add %s onions" % (res["onion_sign_this"], tracker_address, len(sites)))
            request["onion_signs"] = {}
            request["onion_sign_this"] = res["onion_sign_this"]
            request["need_num"] = 0
            for site in sites:
                onion = self.site.connection_server.tor_manager.getOnion(site.address)
                publickey = self.site.connection_server.tor_manager.getPublickey(onion)
                if publickey not in request["onion_signs"]:
                    sign = CryptRsa.sign(res["onion_sign_this"].encode("utf8"), self.site.connection_server.tor_manager.getPrivatekey(onion))
                    request["onion_signs"][publickey] = sign
            res = tracker_peer.request("announce", request)
            if not res or "onion_sign_this" in res:
                if full_announce:
                    time_full_announced[tracker_address] = 0
                raise AnnounceError("Announce onion address to failed: %s" % res)

        if full_announce:
            tracker_peer.remove()  # Close connection, we don't need it in next 5 minute

        self.site.log.debug(
            "Tracker announce result: zero://%s (sites: %s, new peers: %s, add: %s, mode: %s) in %.3fs" %
            (tracker_address, site_index, peers_added, add_types, mode, time.time() - s)
        )

        return True

```

# `plugins/AnnounceZero/__init__.py`

这段代码的作用是引入了一个名为"AnnounceZeroPlugin"的类，可能用于在应用程序中添加一个声明，告知用户某些操作即将发生，并在这些操作完成时通知用户。具体来说，这个插件可能用于在用户进行某些操作前告知他们将会做什么，例如安装更新、打开新窗口、连接到远程服务器等。用户可以通过点击插件的图标或右键单击任务栏图标来访问声明，并根据需要决定是否取消这些操作。


```py
from . import AnnounceZeroPlugin
```

# `plugins/Benchmark/BenchmarkDb.py`

This is a Python test case for the indexed and not indexed test methods of a database. The `testDbQueryIndexed` method tests the indexed test method, while the `testDbQueryNotIndexed` method tests the not indexed test method.

The `getTestDb` method is an asynchronous method that retrieves the test database from the server. The `fillTestDb` method is a test method that fills the test database with sample data.

The `yield` statement is used to generate output for each test method. The output is formatted as a percentage of the test run time (e.g., " (Db warmup done in 1000ms) ").

The `assert` statements are used to check the results of the database operations. The first `assert` statement checks if the found total is equal to 100, and the second `assert` statement checks if the current test run is greater than or equal to the first 100 tests.


```py
import os
import json
import contextlib
import time

from Plugin import PluginManager
from Config import config


@PluginManager.registerTo("Actions")
class ActionsPlugin:
    def getBenchmarkTests(self, online=False):
        tests = super().getBenchmarkTests(online)
        tests.extend([
            {"func": self.testDbConnect, "num": 10, "time_standard": 0.27},
            {"func": self.testDbInsert, "num": 10, "time_standard": 0.91},
            {"func": self.testDbInsertMultiuser, "num": 1, "time_standard": 0.57},
            {"func": self.testDbQueryIndexed, "num": 1000, "time_standard": 0.84},
            {"func": self.testDbQueryNotIndexed, "num": 1000, "time_standard": 1.30}
        ])
        return tests


    @contextlib.contextmanager
    def getTestDb(self):
        from Db import Db
        path = "%s/benchmark.db" % config.data_dir
        if os.path.isfile(path):
            os.unlink(path)
        schema = {
            "db_name": "TestDb",
            "db_file": path,
            "maps": {
                ".*": {
                    "to_table": {
                        "test": "test"
                    }
                }
            },
            "tables": {
                "test": {
                    "cols": [
                        ["test_id", "INTEGER"],
                        ["title", "TEXT"],
                        ["json_id", "INTEGER REFERENCES json (json_id)"]
                    ],
                    "indexes": ["CREATE UNIQUE INDEX test_key ON test(test_id, json_id)"],
                    "schema_changed": 1426195822
                }
            }
        }

        db = Db.Db(schema, path)

        yield db

        db.close()
        if os.path.isfile(path):
            os.unlink(path)

    def testDbConnect(self, num_run=1):
        import sqlite3
        for i in range(num_run):
            with self.getTestDb() as db:
                db.checkTables()
            yield "."
        yield "(SQLite version: %s, API: %s)" % (sqlite3.sqlite_version, sqlite3.version)

    def testDbInsert(self, num_run=1):
        yield "x 1000 lines "
        for u in range(num_run):
            with self.getTestDb() as db:
                db.checkTables()
                data = {"test": []}
                for i in range(1000):  # 1000 line of data
                    data["test"].append({"test_id": i, "title": "Testdata for %s message %s" % (u, i)})
                json.dump(data, open("%s/test_%s.json" % (config.data_dir, u), "w"))
                db.updateJson("%s/test_%s.json" % (config.data_dir, u))
                os.unlink("%s/test_%s.json" % (config.data_dir, u))
                assert db.execute("SELECT COUNT(*) FROM test").fetchone()[0] == 1000
            yield "."

    def fillTestDb(self, db):
        db.checkTables()
        cur = db.getCursor()
        cur.logging = False
        for u in range(100, 200):  # 100 user
            data = {"test": []}
            for i in range(100):  # 1000 line of data
                data["test"].append({"test_id": i, "title": "Testdata for %s message %s" % (u, i)})
            json.dump(data, open("%s/test_%s.json" % (config.data_dir, u), "w"))
            db.updateJson("%s/test_%s.json" % (config.data_dir, u), cur=cur)
            os.unlink("%s/test_%s.json" % (config.data_dir, u))
            if u % 10 == 0:
                yield "."

    def testDbInsertMultiuser(self, num_run=1):
        yield "x 100 users x 100 lines "
        for u in range(num_run):
            with self.getTestDb() as db:
                for progress in self.fillTestDb(db):
                    yield progress
                num_rows = db.execute("SELECT COUNT(*) FROM test").fetchone()[0]
                assert num_rows == 10000, "%s != 10000" % num_rows

    def testDbQueryIndexed(self, num_run=1):
        s = time.time()
        with self.getTestDb() as db:
            for progress in self.fillTestDb(db):
                pass
            yield " (Db warmup done in %.3fs) " % (time.time() - s)
            found_total = 0
            for i in range(num_run):  # 1000x by test_id
                found = 0
                res = db.execute("SELECT * FROM test WHERE test_id = %s" % (i % 100))
                for row in res:
                    found_total += 1
                    found += 1
                del(res)
                yield "."
                assert found == 100, "%s != 100 (i: %s)" % (found, i)
            yield "Found: %s" % found_total

    def testDbQueryNotIndexed(self, num_run=1):
        s = time.time()
        with self.getTestDb() as db:
            for progress in self.fillTestDb(db):
                pass
            yield " (Db warmup done in %.3fs) " % (time.time() - s)
            found_total = 0
            for i in range(num_run):  # 1000x by test_id
                found = 0
                res = db.execute("SELECT * FROM test WHERE json_id = %s" % i)
                for row in res:
                    found_total += 1
                    found += 1
                yield "."
                del(res)
                if i == 0 or i > 100:
                    assert found == 0, "%s != 0 (i: %s)" % (found, i)
                else:
                    assert found == 100, "%s != 100 (i: %s)" % (found, i)
            yield "Found: %s" % found_total

```

# `plugins/Benchmark/BenchmarkPack.py`

This code appears to be a Python script that is used to execute UDP/TCP (or both) communications between two nodes (nodes "A" and "B") to establish a摇椅 (a kind of chair that can be rotated) communication protocol.

It starts by defining some constants and variables:

* `data`: a bytes object that will be sent and received as the data in the communication.
* `num_run`: an integer that will run the simulation.
* `data_packed`: a bytes object that will hold the data to be sent.
* `output_packet_size`: an integer that specifies the maximum size of an output packet that will be sent.
* `is_client`: a boolean that will determine whether node A is the client (发送方) or the server (接收方).
* `is_verbose`: a boolean that will determine whether the node should send verbose messages when a packet is sent or received.

It then defines the `MessagePacket` class that will be used to send and receive messages:

* `MessagePacket`:

class MessagePacket:
   def __init__(self, data, output_packet_size):
       self.data = data
       self.output_packet_size = output_packet_size

   def run(self):
       self.data_unpacked = self.data
       self.output_packet = self.data[:self.output_packet_size]

   def send(self, sender):
       self.sender = sender
       sender.send(self.output_packet)

   def receive(self, receiver):
       self.receiver = receiver
       receiver.receive(self.output_packet)


This class has methods to run the message, send it, and receive it.

It is then main function that defines the main program flow:

* initialize some variables like data, output_packet_size, and is_verbose
* If the is_verbose is True, it will send verbose messages.
* If the is_client is True, it will send data through the TCP port 8888 and receive data through the TCP port 8888 and port 8888.
* If the is_verbose is True, it will send data through the TCP port 8888 and receive data through the TCP port 8888.
* If the is_client is True, it will send data through the TCP port 8888 and receive data through the TCP port 8888.
* If the is_verbose is True, it will use the bytes object to send data.
* If the is_client is True, it will use the MessagePacket class to send data.
* If the is_verbose is True, it will use the send method of the MessagePacket class to send the data.
* If the is_verbose is True, it will use the receive method of the MessagePacket class to receive the data.
* If the is_client is True, it will use the receive method of the MessagePacket class to receive the data.
* If the is_verbose is True, it will use the run method of the MessagePacket class to run the simulation.
* If the is_verbose is True, it will use the



```py
import os
import io
from collections import OrderedDict

from Plugin import PluginManager
from Config import config
from util import Msgpack


@PluginManager.registerTo("Actions")
class ActionsPlugin:
    def createZipFile(self, path):
        import zipfile
        test_data = b"Test" * 1024
        file_name = b"\xc3\x81rv\xc3\xadzt\xc5\xb1r\xc5\x91%s.txt".decode("utf8")
        with zipfile.ZipFile(path, 'w') as archive:
            for y in range(100):
                zip_info = zipfile.ZipInfo(file_name % y, (1980, 1, 1, 0, 0, 0))
                zip_info.compress_type = zipfile.ZIP_DEFLATED
                zip_info.create_system = 3
                zip_info.flag_bits = 0
                zip_info.external_attr = 25165824
                archive.writestr(zip_info, test_data)

    def testPackZip(self, num_run=1):
        """
        Test zip file creating
        """
        yield "x 100 x 5KB "
        from Crypt import CryptHash
        zip_path = '%s/test.zip' % config.data_dir
        for i in range(num_run):
            self.createZipFile(zip_path)
            yield "."

        archive_size = os.path.getsize(zip_path) / 1024
        yield "(Generated file size: %.2fkB)" % archive_size

        hash = CryptHash.sha512sum(open(zip_path, "rb"))
        valid = "cb32fb43783a1c06a2170a6bc5bb228a032b67ff7a1fd7a5efb9b467b400f553"
        assert hash == valid, "Invalid hash: %s != %s<br>" % (hash, valid)
        os.unlink(zip_path)

    def testUnpackZip(self, num_run=1):
        """
        Test zip file reading
        """
        yield "x 100 x 5KB "
        import zipfile
        zip_path = '%s/test.zip' % config.data_dir
        test_data = b"Test" * 1024
        file_name = b"\xc3\x81rv\xc3\xadzt\xc5\xb1r\xc5\x91".decode("utf8")

        self.createZipFile(zip_path)
        for i in range(num_run):
            with zipfile.ZipFile(zip_path) as archive:
                for f in archive.filelist:
                    assert f.filename.startswith(file_name), "Invalid filename: %s != %s" % (f.filename, file_name)
                    data = archive.open(f.filename).read()
                    assert archive.open(f.filename).read() == test_data, "Invalid data: %s..." % data[0:30]
            yield "."

        os.unlink(zip_path)

    def createArchiveFile(self, path, archive_type="gz"):
        import tarfile
        import gzip

        # Monkey patch _init_write_gz to use fixed date in order to keep the hash independent from datetime
        def nodate_write_gzip_header(self):
            self._write_mtime = 0
            original_write_gzip_header(self)

        test_data_io = io.BytesIO(b"Test" * 1024)
        file_name = b"\xc3\x81rv\xc3\xadzt\xc5\xb1r\xc5\x91%s.txt".decode("utf8")

        original_write_gzip_header = gzip.GzipFile._write_gzip_header
        gzip.GzipFile._write_gzip_header = nodate_write_gzip_header
        with tarfile.open(path, 'w:%s' % archive_type) as archive:
            for y in range(100):
                test_data_io.seek(0)
                tar_info = tarfile.TarInfo(file_name % y)
                tar_info.size = 4 * 1024
                archive.addfile(tar_info, test_data_io)

    def testPackArchive(self, num_run=1, archive_type="gz"):
        """
        Test creating tar archive files
        """
        yield "x 100 x 5KB "
        from Crypt import CryptHash

        hash_valid_db = {
            "gz": "92caec5121a31709cbbc8c11b0939758e670b055bbbe84f9beb3e781dfde710f",
            "bz2": "b613f41e6ee947c8b9b589d3e8fa66f3e28f63be23f4faf015e2f01b5c0b032d",
            "xz": "ae43892581d770959c8d993daffab25fd74490b7cf9fafc7aaee746f69895bcb",
        }
        archive_path = '%s/test.tar.%s' % (config.data_dir, archive_type)
        for i in range(num_run):
            self.createArchiveFile(archive_path, archive_type=archive_type)
            yield "."

        archive_size = os.path.getsize(archive_path) / 1024
        yield "(Generated file size: %.2fkB)" % archive_size

        hash = CryptHash.sha512sum(open("%s/test.tar.%s" % (config.data_dir, archive_type), "rb"))
        valid = hash_valid_db[archive_type]
        assert hash == valid, "Invalid hash: %s != %s<br>" % (hash, valid)

        if os.path.isfile(archive_path):
            os.unlink(archive_path)

    def testUnpackArchive(self, num_run=1, archive_type="gz"):
        """
        Test reading tar archive files
        """
        yield "x 100 x 5KB "
        import tarfile

        test_data = b"Test" * 1024
        file_name = b"\xc3\x81rv\xc3\xadzt\xc5\xb1r\xc5\x91%s.txt".decode("utf8")
        archive_path = '%s/test.tar.%s' % (config.data_dir, archive_type)
        self.createArchiveFile(archive_path, archive_type=archive_type)
        for i in range(num_run):
            with tarfile.open(archive_path, 'r:%s' % archive_type) as archive:
                for y in range(100):
                    assert archive.extractfile(file_name % y).read() == test_data
            yield "."
        if os.path.isfile(archive_path):
            os.unlink(archive_path)

    def testPackMsgpack(self, num_run=1):
        """
        Test msgpack encoding
        """
        yield "x 100 x 5KB "
        binary = b'fqv\xf0\x1a"e\x10,\xbe\x9cT\x9e(\xa5]u\x072C\x8c\x15\xa2\xa8\x93Sw)\x19\x02\xdd\t\xfb\xf67\x88\xd9\xee\x86\xa1\xe4\xb6,\xc6\x14\xbb\xd7$z\x1d\xb2\xda\x85\xf5\xa0\x97^\x01*\xaf\xd3\xb0!\xb7\x9d\xea\x89\xbbh8\xa1"\xa7]e(@\xa2\xa5g\xb7[\xae\x8eE\xc2\x9fL\xb6s\x19\x19\r\xc8\x04S\xd0N\xe4]?/\x01\xea\xf6\xec\xd1\xb3\xc2\x91\x86\xd7\xf4K\xdf\xc2lV\xf4\xe8\x80\xfc\x8ep\xbb\x82\xb3\x86\x98F\x1c\xecS\xc8\x15\xcf\xdc\xf1\xed\xfc\xd8\x18r\xf9\x80\x0f\xfa\x8cO\x97(\x0b]\xf1\xdd\r\xe7\xbf\xed\x06\xbd\x1b?\xc5\xa0\xd7a\x82\xf3\xa8\xe6@\xf3\ri\xa1\xb10\xf6\xd4W\xbc\x86\x1a\xbb\xfd\x94!bS\xdb\xaeM\x92\x00#\x0b\xf7\xad\xe9\xc2\x8e\x86\xbfi![%\xd31]\xc6\xfc2\xc9\xda\xc6v\x82P\xcc\xa9\xea\xb9\xff\xf6\xc8\x17iD\xcf\xf3\xeeI\x04\xe9\xa1\x19\xbb\x01\x92\xf5nn4K\xf8\xbb\xc6\x17e>\xa7 \xbbv'
        data = OrderedDict(
            sorted({"int": 1024 * 1024 * 1024, "float": 12345.67890, "text": "hello" * 1024, "binary": binary}.items())
        )
        data_packed_valid = b'\x84\xa6binary\xc5\x01\x00fqv\xf0\x1a"e\x10,\xbe\x9cT\x9e(\xa5]u\x072C\x8c\x15\xa2\xa8\x93Sw)\x19\x02\xdd\t\xfb\xf67\x88\xd9\xee\x86\xa1\xe4\xb6,\xc6\x14\xbb\xd7$z\x1d\xb2\xda\x85\xf5\xa0\x97^\x01*\xaf\xd3\xb0!\xb7\x9d\xea\x89\xbbh8\xa1"\xa7]e(@\xa2\xa5g\xb7[\xae\x8eE\xc2\x9fL\xb6s\x19\x19\r\xc8\x04S\xd0N\xe4]?/\x01\xea\xf6\xec\xd1\xb3\xc2\x91\x86\xd7\xf4K\xdf\xc2lV\xf4\xe8\x80\xfc\x8ep\xbb\x82\xb3\x86\x98F\x1c\xecS\xc8\x15\xcf\xdc\xf1\xed\xfc\xd8\x18r\xf9\x80\x0f\xfa\x8cO\x97(\x0b]\xf1\xdd\r\xe7\xbf\xed\x06\xbd\x1b?\xc5\xa0\xd7a\x82\xf3\xa8\xe6@\xf3\ri\xa1\xb10\xf6\xd4W\xbc\x86\x1a\xbb\xfd\x94!bS\xdb\xaeM\x92\x00#\x0b\xf7\xad\xe9\xc2\x8e\x86\xbfi![%\xd31]\xc6\xfc2\xc9\xda\xc6v\x82P\xcc\xa9\xea\xb9\xff\xf6\xc8\x17iD\xcf\xf3\xeeI\x04\xe9\xa1\x19\xbb\x01\x92\xf5nn4K\xf8\xbb\xc6\x17e>\xa7 \xbbv\xa5float\xcb@\xc8\x1c\xd6\xe61\xf8\xa1\xa3int\xce@\x00\x00\x00\xa4text\xda\x14\x00'
        data_packed_valid += b'hello' * 1024
        for y in range(num_run):
            for i in range(100):
                data_packed = Msgpack.pack(data)
            yield "."
        assert data_packed == data_packed_valid, "%s<br>!=<br>%s" % (repr(data_packed), repr(data_packed_valid))

    def testUnpackMsgpack(self, num_run=1):
        """
        Test msgpack decoding
        """
        yield "x 5KB "
        binary = b'fqv\xf0\x1a"e\x10,\xbe\x9cT\x9e(\xa5]u\x072C\x8c\x15\xa2\xa8\x93Sw)\x19\x02\xdd\t\xfb\xf67\x88\xd9\xee\x86\xa1\xe4\xb6,\xc6\x14\xbb\xd7$z\x1d\xb2\xda\x85\xf5\xa0\x97^\x01*\xaf\xd3\xb0!\xb7\x9d\xea\x89\xbbh8\xa1"\xa7]e(@\xa2\xa5g\xb7[\xae\x8eE\xc2\x9fL\xb6s\x19\x19\r\xc8\x04S\xd0N\xe4]?/\x01\xea\xf6\xec\xd1\xb3\xc2\x91\x86\xd7\xf4K\xdf\xc2lV\xf4\xe8\x80\xfc\x8ep\xbb\x82\xb3\x86\x98F\x1c\xecS\xc8\x15\xcf\xdc\xf1\xed\xfc\xd8\x18r\xf9\x80\x0f\xfa\x8cO\x97(\x0b]\xf1\xdd\r\xe7\xbf\xed\x06\xbd\x1b?\xc5\xa0\xd7a\x82\xf3\xa8\xe6@\xf3\ri\xa1\xb10\xf6\xd4W\xbc\x86\x1a\xbb\xfd\x94!bS\xdb\xaeM\x92\x00#\x0b\xf7\xad\xe9\xc2\x8e\x86\xbfi![%\xd31]\xc6\xfc2\xc9\xda\xc6v\x82P\xcc\xa9\xea\xb9\xff\xf6\xc8\x17iD\xcf\xf3\xeeI\x04\xe9\xa1\x19\xbb\x01\x92\xf5nn4K\xf8\xbb\xc6\x17e>\xa7 \xbbv'
        data = OrderedDict(
            sorted({"int": 1024 * 1024 * 1024, "float": 12345.67890, "text": "hello" * 1024, "binary": binary}.items())
        )
        data_packed = b'\x84\xa6binary\xc5\x01\x00fqv\xf0\x1a"e\x10,\xbe\x9cT\x9e(\xa5]u\x072C\x8c\x15\xa2\xa8\x93Sw)\x19\x02\xdd\t\xfb\xf67\x88\xd9\xee\x86\xa1\xe4\xb6,\xc6\x14\xbb\xd7$z\x1d\xb2\xda\x85\xf5\xa0\x97^\x01*\xaf\xd3\xb0!\xb7\x9d\xea\x89\xbbh8\xa1"\xa7]e(@\xa2\xa5g\xb7[\xae\x8eE\xc2\x9fL\xb6s\x19\x19\r\xc8\x04S\xd0N\xe4]?/\x01\xea\xf6\xec\xd1\xb3\xc2\x91\x86\xd7\xf4K\xdf\xc2lV\xf4\xe8\x80\xfc\x8ep\xbb\x82\xb3\x86\x98F\x1c\xecS\xc8\x15\xcf\xdc\xf1\xed\xfc\xd8\x18r\xf9\x80\x0f\xfa\x8cO\x97(\x0b]\xf1\xdd\r\xe7\xbf\xed\x06\xbd\x1b?\xc5\xa0\xd7a\x82\xf3\xa8\xe6@\xf3\ri\xa1\xb10\xf6\xd4W\xbc\x86\x1a\xbb\xfd\x94!bS\xdb\xaeM\x92\x00#\x0b\xf7\xad\xe9\xc2\x8e\x86\xbfi![%\xd31]\xc6\xfc2\xc9\xda\xc6v\x82P\xcc\xa9\xea\xb9\xff\xf6\xc8\x17iD\xcf\xf3\xeeI\x04\xe9\xa1\x19\xbb\x01\x92\xf5nn4K\xf8\xbb\xc6\x17e>\xa7 \xbbv\xa5float\xcb@\xc8\x1c\xd6\xe61\xf8\xa1\xa3int\xce@\x00\x00\x00\xa4text\xda\x14\x00'
        data_packed += b'hello' * 1024
        for y in range(num_run):
            data_unpacked = Msgpack.unpack(data_packed, decode=False)
            yield "."
        assert data_unpacked == data, "%s<br>!=<br>%s" % (data_unpacked, data)

    def testUnpackMsgpackStreaming(self, num_run=1, fallback=False):
        """
        Test streaming msgpack decoding
        """
        yield "x 1000 x 5KB "
        binary = b'fqv\xf0\x1a"e\x10,\xbe\x9cT\x9e(\xa5]u\x072C\x8c\x15\xa2\xa8\x93Sw)\x19\x02\xdd\t\xfb\xf67\x88\xd9\xee\x86\xa1\xe4\xb6,\xc6\x14\xbb\xd7$z\x1d\xb2\xda\x85\xf5\xa0\x97^\x01*\xaf\xd3\xb0!\xb7\x9d\xea\x89\xbbh8\xa1"\xa7]e(@\xa2\xa5g\xb7[\xae\x8eE\xc2\x9fL\xb6s\x19\x19\r\xc8\x04S\xd0N\xe4]?/\x01\xea\xf6\xec\xd1\xb3\xc2\x91\x86\xd7\xf4K\xdf\xc2lV\xf4\xe8\x80\xfc\x8ep\xbb\x82\xb3\x86\x98F\x1c\xecS\xc8\x15\xcf\xdc\xf1\xed\xfc\xd8\x18r\xf9\x80\x0f\xfa\x8cO\x97(\x0b]\xf1\xdd\r\xe7\xbf\xed\x06\xbd\x1b?\xc5\xa0\xd7a\x82\xf3\xa8\xe6@\xf3\ri\xa1\xb10\xf6\xd4W\xbc\x86\x1a\xbb\xfd\x94!bS\xdb\xaeM\x92\x00#\x0b\xf7\xad\xe9\xc2\x8e\x86\xbfi![%\xd31]\xc6\xfc2\xc9\xda\xc6v\x82P\xcc\xa9\xea\xb9\xff\xf6\xc8\x17iD\xcf\xf3\xeeI\x04\xe9\xa1\x19\xbb\x01\x92\xf5nn4K\xf8\xbb\xc6\x17e>\xa7 \xbbv'
        data = OrderedDict(
            sorted({"int": 1024 * 1024 * 1024, "float": 12345.67890, "text": "hello" * 1024, "binary": binary}.items())
        )
        data_packed = b'\x84\xa6binary\xc5\x01\x00fqv\xf0\x1a"e\x10,\xbe\x9cT\x9e(\xa5]u\x072C\x8c\x15\xa2\xa8\x93Sw)\x19\x02\xdd\t\xfb\xf67\x88\xd9\xee\x86\xa1\xe4\xb6,\xc6\x14\xbb\xd7$z\x1d\xb2\xda\x85\xf5\xa0\x97^\x01*\xaf\xd3\xb0!\xb7\x9d\xea\x89\xbbh8\xa1"\xa7]e(@\xa2\xa5g\xb7[\xae\x8eE\xc2\x9fL\xb6s\x19\x19\r\xc8\x04S\xd0N\xe4]?/\x01\xea\xf6\xec\xd1\xb3\xc2\x91\x86\xd7\xf4K\xdf\xc2lV\xf4\xe8\x80\xfc\x8ep\xbb\x82\xb3\x86\x98F\x1c\xecS\xc8\x15\xcf\xdc\xf1\xed\xfc\xd8\x18r\xf9\x80\x0f\xfa\x8cO\x97(\x0b]\xf1\xdd\r\xe7\xbf\xed\x06\xbd\x1b?\xc5\xa0\xd7a\x82\xf3\xa8\xe6@\xf3\ri\xa1\xb10\xf6\xd4W\xbc\x86\x1a\xbb\xfd\x94!bS\xdb\xaeM\x92\x00#\x0b\xf7\xad\xe9\xc2\x8e\x86\xbfi![%\xd31]\xc6\xfc2\xc9\xda\xc6v\x82P\xcc\xa9\xea\xb9\xff\xf6\xc8\x17iD\xcf\xf3\xeeI\x04\xe9\xa1\x19\xbb\x01\x92\xf5nn4K\xf8\xbb\xc6\x17e>\xa7 \xbbv\xa5float\xcb@\xc8\x1c\xd6\xe61\xf8\xa1\xa3int\xce@\x00\x00\x00\xa4text\xda\x14\x00'
        data_packed += b'hello' * 1024
        for i in range(num_run):
            unpacker = Msgpack.getUnpacker(decode=False, fallback=fallback)
            for y in range(1000):
                unpacker.feed(data_packed)
                for data_unpacked in unpacker:
                    pass
            yield "."
        assert data == data_unpacked, "%s != %s" % (data_unpacked, data)

```
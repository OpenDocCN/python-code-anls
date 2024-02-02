# `ZeroNet\src\File\FileServer.py`

```py
# 导入日志模块
import logging
# 导入时间模块
import time
# 导入随机数模块
import random
# 导入套接字模块
import socket
# 导入系统模块
import sys

# 导入协程模块
import gevent
# 导入协程池模块
import gevent.pool
# 从协程服务器模块中导入流服务器
from gevent.server import StreamServer

# 导入自定义工具模块
import util
# 从工具模块中导入辅助函数
from util import helper
# 从配置模块中导入配置
from Config import config
# 从文件请求模块中导入文件请求
from .FileRequest import FileRequest
# 从对等端口检查模块中导入对等端口检查
from Peer import PeerPortchecker
# 从站点管理模块中导入站点管理器
from Site import SiteManager
# 从连接模块中导入连接服务器
from Connection import ConnectionServer
# 从插件模块中导入插件管理器
from Plugin import PluginManager
# 从调试模块中导入调试
from Debug import Debug

# 使用插件管理器接受插件
@PluginManager.acceptPlugins
# 文件服务器类继承自连接服务器类
class FileServer(ConnectionServer):

    # 获取指定范围内的随机端口
    def getRandomPort(self, ip, port_range_from, port_range_to):
        # 记录日志信息
        self.log.info("Getting random port in range %s-%s..." % (port_range_from, port_range_to))
        # 已尝试的端口列表
        tried = []
        # 重试绑定端口100次
        for bind_retry in range(100):
            # 生成随机端口
            port = random.randint(port_range_from, port_range_to)
            # 如果端口已经尝试过，则继续下一次循环
            if port in tried:
                continue
            # 将端口添加到已尝试列表中
            tried.append(port)
            # 创建套接字
            sock = helper.createSocket(ip)
            try:
                # 尝试绑定套接字到指定IP和端口
                sock.bind((ip, port))
                success = True
            except Exception as err:
                # 记录绑定端口出错的日志信息
                self.log.warning("Error binding to port %s: %s" % (port, err))
                success = False
            # 关闭套接字
            sock.close()
            # 如果绑定成功，则返回找到的未使用的随机端口
            if success:
                self.log.info("Found unused random port: %s" % port)
                return port
            else:
                # 如果绑定失败，则等待0.1秒后重试
                time.sleep(0.1)
        # 如果重试100次后仍未找到未使用的随机端口，则返回False
        return False
    # 检查是否支持 IPv6
    def isIpv6Supported(self):
        # 如果配置为始终使用 TOR，则返回 True
        if config.tor == "always":
            return True
        # 测试是否可以连接到 IPv6 地址
        ipv6_testip = "fcec:ae97:8902:d810:6c92:ec67:efb2:3ec5"
        try:
            # 创建 IPv6 套接字
            sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
            # 连接到 IPv6 测试地址
            sock.connect((ipv6_testip, 80))
            # 获取本地 IPv6 地址
            local_ipv6 = sock.getsockname()[0]
            # 如果本地 IPv6 地址为 "::1"，则表示不支持 IPv6
            if local_ipv6 == "::1":
                self.log.debug("IPv6 not supported, no local IPv6 address")
                return False
            else:
                self.log.debug("IPv6 supported on IP %s" % local_ipv6)
                return True
        except socket.error as err:
            # IPv6 不受支持
            self.log.warning("IPv6 not supported: %s" % err)
            return False
        except Exception as err:
            # IPv6 检查出错
            self.log.error("IPv6 check error: %s" % err)
            return False

    # 监听代理
    def listenProxy(self):
        try:
            # 永久监听代理
            self.stream_server_proxy.serve_forever()
        except Exception as err:
            # 处理代理监听错误
            if err.errno == 98:  # 地址已被使用错误
                self.log.debug("StreamServer proxy listen error: %s" % err)
            else:
                self.log.info("StreamServer proxy listen error: %s" % err)

    # 处理对文件服务器的请求
    def handleRequest(self, connection, message):
        # 如果配置为冗长模式，记录文件请求
        if config.verbose:
            if "params" in message:
                self.log.debug(
                    "FileRequest: %s %s %s %s" %
                    (str(connection), message["cmd"], message["params"].get("site"), message["params"].get("inner_path"))
                )
            else:
                self.log.debug("FileRequest: %s %s" % (str(connection), message["cmd"]))
        # 创建文件请求对象
        req = FileRequest(self, connection)
        # 路由文件请求
        req.route(message["cmd"], message.get("req_id"), message.get("params"))
        # 如果没有网络并且连接不是私有 IP，则表示有网络连接
        if not self.has_internet and not connection.is_private_ip:
            self.has_internet = True
            self.onInternetOnline()
    # 当网络连接上时调用，记录日志信息“Internet online”，并使用协程调用checkSites方法
    def onInternetOnline(self):
        self.log.info("Internet online")
        gevent.spawn(self.checkSites, check_files=False, force_port_check=True)

    # 重新加载FileRequest类，以防止在调试模式下重新启动
    def reload(self):
        global FileRequest
        import imp
        FileRequest = imp.load_source("FileRequest", "src/File/FileRequest.py").FileRequest

    # 检查站点文件的完整性
    def checkSite(self, site, check_files=False):
        # 如果站点正在提供服务
        if site.isServing():
            # 向跟踪器宣布站点
            site.announce(mode="startup")
            # 更新站点的content.json并下载已更改的文件
            site.update(check_files=check_files)
            # 发送站点的哈希字段
            site.sendMyHashfield()
            # 更新哈希字段
            site.updateHashfield()

    # 检查站点的完整性
    @util.Noparallel()
    # 检查网站，可选择是否检查文件，是否强制端口检查
    def checkSites(self, check_files=False, force_port_check=False):
        # 记录调试信息
        self.log.debug("Checking sites...")
        # 记录开始时间
        s = time.time()
        # 标记是否正在检查网站
        sites_checking = False
        # 如果端口未打开或者强制端口检查
        if not self.port_opened or force_port_check:  # Test and open port if not tested yet
            # 如果网站数量小于等于2，不等待端口打开
            if len(self.sites) <= 2:  # Don't wait port opening on first startup
                # 标记正在检查网站
                sites_checking = True
                # 遍历网站字典，使用协程并发检查网站
                for address, site in list(self.sites.items()):
                    gevent.spawn(self.checkSite, site, check_files)

            # 执行端口检查
            self.portCheck()

            # 如果 IPv4 端口未打开，启动 Tor
            if not self.port_opened["ipv4"]:
                self.tor_manager.startOnions()

        # 如果没有正在检查网站
        if not sites_checking:
            # 创建协程池
            check_pool = gevent.pool.Pool(5)
            # 按照网站修改时间倒序遍历网站列表，检查网站完整性
            for site in sorted(list(self.sites.values()), key=lambda site: site.settings.get("modified", 0), reverse=True):
                if not site.isServing():
                    continue
                # 在新线程中检查网站
                check_thread = check_pool.spawn(self.checkSite, site, check_files)  # Check in new thread
                # 等待2秒
                time.sleep(2)
                # 如果网站修改时间距离当前时间超过一天，等待5秒
                if site.settings.get("modified", 0) < time.time() - 60 * 60 * 24:  # Not so active site, wait some sec to finish
                    check_thread.join(timeout=5)
        # 记录结束时间并输出调试信息
        self.log.debug("Checksites done in %.3fs" % (time.time() - s))
    # 清理站点资源
    def cleanupSites(self):
        import gc  # 导入垃圾回收模块
        startup = True  # 标记是否为启动阶段
        time.sleep(5 * 60)  # 在启动时已经清理了站点
        peers_protected = set([])  # 创建一个空的受保护对等点集合
        while 1:  # 无限循环
            # 每20分钟检查一次站点健康状况
            self.log.debug(
                "Running site cleanup, connections: %s, internet: %s, protected peers: %s" %
                (len(self.connections), self.has_internet, len(peers_protected))
            )

            for address, site in list(self.sites.items()):  # 遍历站点字典
                if not site.isServing():  # 如果站点不提供服务，则跳过
                    continue

                if not startup:  # 如果不是启动阶段
                    site.cleanupPeers(peers_protected)  # 清理受保护的对等点

                time.sleep(1)  # 防止请求过快

            peers_protected = set([])  # 重置受保护对等点集合
            for address, site in list(self.sites.items()):  # 再次遍历站点字典
                if not site.isServing():  # 如果站点不提供服务，则跳过
                    continue

                if site.peers:  # 如果站点有对等点
                    with gevent.Timeout(10, exception=False):  # 设置超时时间为10秒
                        site.announcer.announcePex()  # 发布对等点扩展信息

                # 最后一次检查修改失败
                if site.content_updated is False:  # 如果站点内容未更新
                    site.update()  # 更新站点内容
                elif site.bad_files:  # 如果站点有坏文件
                    site.retryBadFiles()  # 重试下载坏文件

                if time.time() - site.settings.get("modified", 0) < 60 * 60 * 24 * 7:  # 如果站点在7天内有修改
                    connected_num = site.needConnections(check_site_on_reconnect=True)  # 获取需要的连接数

                    if connected_num < config.connected_limit:  # 如果站点的对等点数量较少，保护它们不被关闭
                        peers_protected.update([peer.key for peer in site.getConnectedPeers()])  # 更新受保护对等点集合

                time.sleep(1)  # 防止请求过快

            site = None  # 释放站点对象
            gc.collect()  # 隐式垃圾回收
            startup = False  # 标记启动阶段结束
            time.sleep(60 * 20)  # 休眠20分钟
    # 声明一个方法，用于向指定站点宣布更新
    def announceSite(self, site):
        # 调用站点对象的announce方法，指定更新模式为"update"，不使用pex
        site.announce(mode="update", pex=False)
        # 判断站点是否为自己拥有或者活跃站点（最近修改时间在24小时内）
        active_site = time.time() - site.settings.get("modified", 0) < 24 * 60 * 60
        if site.settings["own"] or active_site:
            # 对于自己拥有或者活跃站点，需要更频繁地检查连接以加快首次连接速度
            site.needConnections(check_site_on_reconnect=True)
        # 向站点发送自己的哈希字段
        site.sendMyHashfield(3)
        # 更新站点的哈希字段
        site.updateHashfield(3)

    # 每20分钟宣布一次站点
    def announceSites(self):
        time.sleep(5 * 60)  # 在启动时已经宣布了站点
        while 1:
            # 加载跟踪器文件
            config.loadTrackersFile()
            s = time.time()
            # 遍历所有站点，宣布活跃站点的更新
            for address, site in list(self.sites.items()):
                if not site.isServing():
                    continue
                # 使用协程并行处理宣布站点更新的操作，并设置超时时间为10秒
                gevent.spawn(self.announceSite, site).join(timeout=10)
                time.sleep(1)
            taken = time.time() - s

            # 在20分钟内均匀分布地查询所有跟踪器
            sleep = max(0, 60 * 20 / len(config.trackers) - taken)

            # 记录站点宣布跟踪器完成的时间和休眠时间
            self.log.debug("Site announce tracker done in %.3fs, sleeping for %.3fs..." % (taken, sleep))
            time.sleep(sleep)

    # 检测计算机是否从休眠状态唤醒
    # 唤醒观察者方法
    def wakeupWatcher(self):
        # 记录当前时间
        last_time = time.time()
        # 获取本机的 IP 地址
        last_my_ips = socket.gethostbyname_ex('')[2]
        while 1:
            # 等待30秒
            time.sleep(30)
            # 判断时间是否发生变化
            is_time_changed = time.time() - max(self.last_request, last_time) > 60 * 3
            if is_time_changed:
                # 如果时间间隔超过3分钟，则表示计算机处于睡眠模式
                self.log.info(
                    "Wakeup detected: time warp from %0.f to %0.f (%0.f sleep seconds), acting like startup..." %
                    (last_time, time.time(), time.time() - last_time)
                )

            # 获取当前的 IP 地址
            my_ips = socket.gethostbyname_ex('')[2]
            # 判断 IP 地址是否发生变化
            is_ip_changed = my_ips != last_my_ips
            if is_ip_changed:
                # 如果 IP 地址发生变化，则记录日志
                self.log.info("IP change detected from %s to %s" % (last_my_ips, my_ips))

            # 如果时间或者 IP 地址发生变化，则调用 checkSites 方法
            if is_time_changed or is_ip_changed:
                self.checkSites(check_files=False, force_port_check=True)

            # 更新 last_time 和 last_my_ips
            last_time = time.time()
            last_my_ips = my_ips

    # 绑定并启动服务站点
    def start(self, check_sites=True):
        if self.stopping:
            return False

        # 启动连接服务器
        ConnectionServer.start(self)

        try:
            # 启动流服务器
            self.stream_server.start()
        except Exception as err:
            self.log.error("Error listening on: %s:%s: %s" % (self.ip, self.port, err))

        # 获取站点列表
        self.sites = self.site_manager.list()
        if config.debug:
            # 在调试模式下，当文件发生变化时自动重新加载 FileRequest
            from Debug import DebugReloader
            DebugReloader.watcher.addCallback(self.reload)

        if check_sites:  # 打开端口，更新站点，检查文件完整性
            # 异步调用 checkSites 方法
            gevent.spawn(self.checkSites)

        # 异步调用 announceSites 和 cleanupSites 方法
        thread_announce_sites = gevent.spawn(self.announceSites)
        thread_cleanup_sites = gevent.spawn(self.cleanupSites)
        # 异步调用 wakeupWatcher 方法
        thread_wakeup_watcher = gevent.spawn(self.wakeupWatcher)

        # 监听连接
        ConnectionServer.listen(self)

        self.log.debug("Stopped.")
    # 停止服务器的方法
    def stop(self):
        # 如果服务器正在运行并且 UPnP 端口已经打开
        if self.running and self.portchecker.upnp_port_opened:
            # 记录调试信息，关闭指定端口
            self.log.debug('Closing port %d' % self.port)
            try:
                # 尝试通过 UPnP 关闭端口
                self.portchecker.portClose(self.port)
                # 记录信息，说明通过 UPnP 关闭了端口
                self.log.info('Closed port via upnp.')
            except Exception as err:
                # 记录信息，说明尝试使用 UPnP 关闭端口失败
                self.log.info("Failed at attempt to use upnp to close port: %s" % err)

        # 调用父类的停止方法
        return ConnectionServer.stop(self)
```
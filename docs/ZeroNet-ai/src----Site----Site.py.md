# `ZeroNet\src\Site\Site.py`

```
# 导入所需的模块
import os
import json
import logging
import re
import time
import random
import sys
import hashlib
import collections
import base64

# 导入并使用协程库
import gevent
import gevent.pool

# 导入自定义的模块
import util
from Config import config
from Peer import Peer
from Worker import WorkerManager
from Debug import Debug
from Content import ContentManager
from .SiteStorage import SiteStorage
from Crypt import CryptHash
from util import helper
from util import Diff
from util import GreenletManager
from Plugin import PluginManager
from File import FileServer
from .SiteAnnouncer import SiteAnnouncer
from . import SiteManager

# 使用 PluginManager.acceptPlugins 装饰器注册插件
@PluginManager.acceptPlugins
# 定义 Site 类
class Site(object):

    # 定义 __str__ 方法，返回站点的简短地址信息
    def __str__(self):
        return "Site %s" % self.address_short

    # 定义 __repr__ 方法，返回站点对象的字符串表示形式
    def __repr__(self):
        return "<%s>" % self.__str__()

    # 从 data/sites.json 文件中加载站点设置
    # Load site settings from data/sites.json
    # 加载设置信息，如果没有传入设置，则从配置文件中读取对应地址的设置信息
    def loadSettings(self, settings=None):
        # 如果没有传入设置信息，则从配置文件中读取对应地址的设置信息
        if not settings:
            settings = json.load(open("%s/sites.json" % config.data_dir)).get(self.address)
        # 如果存在设置信息
        if settings:
            # 将传入的设置信息赋值给对象的设置属性
            self.settings = settings
            # 如果设置信息中没有缓存字段，则添加缓存字段并赋空字典
            if "cache" not in settings:
                settings["cache"] = {}
            # 如果设置信息中没有 size_files_optional 字段，则添加该字段并赋值为 0
            if "size_files_optional" not in settings:
                settings["size_optional"] = 0
            # 如果设置信息中没有 optional_downloaded 字段，则添加该字段并赋值为 0
            if "optional_downloaded" not in settings:
                settings["optional_downloaded"] = 0
            # 如果设置信息中没有 downloaded 字段，则添加该字段并赋值为 added 字段的值
            if "downloaded" not in settings:
                settings["downloaded"] = settings.get("added")
            # 将缓存中的坏文件赋值给对象的坏文件属性，并清空缓存中的坏文件
            self.bad_files = settings["cache"].get("bad_files", {})
            settings["cache"]["bad_files"] = {}
            # 给坏文件添加最小的重试次数为 10 次
            for inner_path in self.bad_files:
                self.bad_files[inner_path] = min(self.bad_files[inner_path], 20)
        # 如果不存在设置信息
        else:
            # 设置默认的设置信息
            self.settings = {
                "own": False, "serving": True, "permissions": [], "cache": {"bad_files": {}}, "size_files_optional": 0,
                "added": int(time.time()), "downloaded": None, "optional_downloaded": 0, "size_optional": 0
            }  # Default
            # 如果配置文件中的下载可选项为 "auto"，则设置自动下载可选项为 True
            if config.download_optional == "auto":
                self.settings["autodownloadoptional"] = True

        # 如果地址为主页或更新站点，并且设置信息中没有管理员权限，则添加管理员权限
        if self.address in (config.homepage, config.updatesite) and "ADMIN" not in self.settings["permissions"]:
            self.settings["permissions"].append("ADMIN")

        return

    # 将站点设置信息保存到 data/sites.json 文件中
    def saveSettings(self):
        # 如果站点管理器中没有站点信息，则初始化站点信息
        if not SiteManager.site_manager.sites:
            SiteManager.site_manager.sites = {}
        # 如果站点管理器中没有当前地址的站点信息，则添加当前站点信息，并加载站点信息
        if not SiteManager.site_manager.sites.get(self.address):
            SiteManager.site_manager.sites[self.address] = self
            SiteManager.site_manager.load(False)
        # 保存站点信息到文件中
        SiteManager.site_manager.saveDelayed()
    # 检查服务是否正在运行
    def isServing(self):
        # 如果处于离线模式，则返回 False
        if config.offline:
            return False
        else:
            # 否则返回设置中的 serving 值
            return self.settings["serving"]

    # 获取设置缓存
    def getSettingsCache(self):
        back = {}
        back["bad_files"] = self.bad_files
        back["hashfield"] = base64.b64encode(self.content_manager.hashfield.tobytes()).decode("ascii")
        return back

    # 获取站点的最大大小限制（以 MB 为单位）
    def getSizeLimit(self):
        return self.settings.get("size_limit", int(config.size_limit))

    # 基于当前大小获取下一个大小限制
    def getNextSizeLimit(self):
        size_limits = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
        size = self.settings.get("size", 0)
        for size_limit in size_limits:
            if size * 1.2 < size_limit * 1024 * 1024:
                return size_limit
        return 999999

    # 检查站点是否最近添加过
    def isAddedRecently(self):
        return time.time() - self.settings.get("added", 0) < 60 * 60 * 24

    # 从 content.json 下载所有文件
    # 返回重试次数少于 3 次的坏文件
    def getReachableBadFiles(self):
        if not self.bad_files:
            return False
        return [bad_file for bad_file, retry in self.bad_files.items() if retry < 3]

    # 重试下载坏文件
    # 重新尝试下载失败的文件，可选择强制重新下载
    def retryBadFiles(self, force=False):
        # 检查失败的文件
        self.checkBadFiles()

        # 记录调试信息，重试下载失败文件的数量
        self.log.debug("Retry %s bad files" % len(self.bad_files))
        # 初始化内容文件和普通文件的列表
        content_inner_paths = []
        file_inner_paths = []

        # 遍历失败的文件及其重试次数
        for bad_file, tries in list(self.bad_files.items()):
            # 如果强制或者随机数小于一定值（根据重试次数动态调整），则进行重试
            if force or random.randint(0, min(40, tries)) < 4:  # Larger number tries = less likely to check every 15min
                # 如果文件名以"content.json"结尾，则加入内容文件列表，否则加入普通文件列表
                if bad_file.endswith("content.json"):
                    content_inner_paths.append(bad_file)
                else:
                    file_inner_paths.append(bad_file)

        # 如果存在内容文件，则使用池化下载内容文件
        if content_inner_paths:
            self.pooledDownloadContent(content_inner_paths, only_if_bad=True)

        # 如果存在普通文件，则使用池化下载普通文件
        if file_inner_paths:
            self.pooledDownloadFile(file_inner_paths, only_if_bad=True)

    # 检查失败的文件
    def checkBadFiles(self):
        # 遍历失败的文件
        for bad_file in list(self.bad_files.keys()):
            # 获取文件信息
            file_info = self.content_manager.getFileInfo(bad_file)
            # 如果文件名以"content.json"结尾
            if bad_file.endswith("content.json"):
                # 如果文件信息为假且文件名不是"content.json"，则从失败文件列表中删除该文件
                if file_info is False and bad_file != "content.json":
                    del self.bad_files[bad_file]
                    self.log.debug("No info for file: %s, removing from bad_files" % bad_file)
            else:
                # 如果文件信息为假或者文件大小为空
                if file_info is False or not file_info.get("size"):
                    # 则从失败文件列表中删除该文件
                    del self.bad_files[bad_file]
                    self.log.debug("No info or size for file: %s, removing from bad_files" % bad_file)

    # 下载站点的所有文件
    @util.Noparallel(blocking=False)
    # 定义一个下载方法，可以选择是否检查文件大小，是否忽略错误文件，是否重试下载失败的文件
    def download(self, check_size=False, blind_includes=False, retry_bad_files=True):
        # 如果没有连接服务器，则跳过下载并记录日志
        if not self.connection_server:
            self.log.debug("No connection server found, skipping download")
            return False

        # 记录下载开始时间和相关参数
        s = time.time()
        self.log.debug(
            "Start downloading, bad_files: %s, check_size: %s, blind_includes: %s, isAddedRecently: %s" %
            (self.bad_files, check_size, blind_includes, self.isAddedRecently())
        )

        # 如果最近添加过文件，则异步通知服务器
        if self.isAddedRecently():
            gevent.spawn(self.announce, mode="start", force=True)
        else:
            gevent.spawn(self.announce, mode="update")

        # 如果需要检查文件大小，则先下载 content.json 文件进行检查
        if check_size:  # Check the size first
            valid = self.downloadContent("content.json", download_files=False)  # Just download content.json files
            if not valid:
                return False  # Cant download content.jsons or size is not fits

        # 下载所有文件
        valid = self.downloadContent("content.json", check_modifications=blind_includes)

        # 如果需要重试下载失败的文件，则在下载完成后触发重试方法
        if retry_bad_files:
            self.onComplete.once(lambda: self.retryBadFiles(force=True))
        # 记录下载完成时间并返回下载结果
        self.log.debug("Download done in %.3fs" % (time.time() - s))

        return valid
    # 使用协程池并发下载内容，可以设置池大小和是否只下载坏文件
    def pooledDownloadContent(self, inner_paths, pool_size=100, only_if_bad=False):
        # 记录日志，显示下载内容池的长度和是否只下载坏文件
        self.log.debug("New downloadContent pool: len: %s, only if bad: %s" % (len(inner_paths), only_if_bad))
        # 增加已开始任务数
        self.worker_manager.started_task_num += len(inner_paths)
        # 创建协程池
        pool = gevent.pool.Pool(pool_size)
        # 记录跳过的任务数
        num_skipped = 0
        # 获取站点大小限制
        site_size_limit = self.getSizeLimit() * 1024 * 1024
        # 遍历内部路径
        for inner_path in inner_paths:
            # 如果不仅下载坏文件或者内部路径在坏文件列表中
            if not only_if_bad or inner_path in self.bad_files:
                # 在协程池中执行下载内容任务
                pool.spawn(self.downloadContent, inner_path)
            else:
                # 增加跳过的任务数
                num_skipped += 1
            # 减少已开始任务数
            self.worker_manager.started_task_num -= 1
            # 如果站点大小超过限制的95%
            if self.settings["size"] > site_size_limit * 0.95:
                # 记录警告日志，站点大小接近限制，中止下载内容池
                self.log.warning("Site size limit almost reached, aborting downloadContent pool")
                # 遍历中止的内部路径
                for aborted_inner_path in inner_paths:
                    # 如果中止的内部路径在坏文件列表中
                    if aborted_inner_path in self.bad_files:
                        # 从坏文件列表中删除中止的内部路径
                        del self.bad_files[aborted_inner_path]
                # 移除已解决的文件任务，不标记为好文件
                self.worker_manager.removeSolvedFileTasks(mark_as_good=False)
                # 中止循环
                break
        # 等待协程池中的任务完成
        pool.join()
        # 记录结束下载内容池的日志，显示池长度和跳过的任务数
        self.log.debug("Ended downloadContent pool len: %s, skipped: %s" % (len(inner_paths), num_skipped))

    # 使用协程池并发下载文件，可以设置池大小和是否只下载坏文件
    def pooledDownloadFile(self, inner_paths, pool_size=100, only_if_bad=False):
        # 记录日志，显示下载文件池的长度和是否只下载坏文件
        self.log.debug("New downloadFile pool: len: %s, only if bad: %s" % (len(inner_paths), only_if_bad))
        # 增加已开始任务数
        self.worker_manager.started_task_num += len(inner_paths)
        # 创建协程池
        pool = gevent.pool.Pool(pool_size)
        # 记录跳过的任务数
        num_skipped = 0
        # 遍历内部路径
        for inner_path in inner_paths:
            # 如果不仅下载坏文件或者内部路径在坏文件列表中
            if not only_if_bad or inner_path in self.bad_files:
                # 在协程池中执行需要文件任务，更新为真
                pool.spawn(self.needFile, inner_path, update=True)
            else:
                # 增加跳过的任务数
                num_skipped += 1
            # 减少已开始任务数
            self.worker_manager.started_task_num -= 1
        # 记录结束下载文件池的日志，显示池长度和跳过的任务数
        self.log.debug("Ended downloadFile pool len: %s, skipped: %s" % (len(inner_paths), num_skipped))
    # 更新工作程序，尝试查找支持listModifications命令的客户端
    # 检查来自对等方的已修改的content.json文件，并将已修改的文件添加到bad_files中
    # 返回：成功查询的对等方[对等方，对等方...]
    # 检查是否有修改，可以指定起始时间
    def checkModifications(self, since=None):
        # 记录开始时间
        s = time.time()
        peers_try = []  # 尝试连接的对等节点
        queried = []  # 成功从这些对等节点查询到的信息
        limit = 5

        # 等待对等节点
        if not self.peers:
            # 如果没有对等节点，则更新通告
            self.announce(mode="update")
            for wait in range(10):
                time.sleep(5 + wait)
                self.log.debug("CheckModifications: Waiting for peers...")
                if self.peers:
                    break

        # 获取已连接的对等节点
        peers_try = self.getConnectedPeers()
        peers_connected_num = len(peers_try)
        # 如果已连接的对等节点数量小于限制的两倍，则添加更多未连接的对等节点
        if peers_connected_num < limit * 2:
            peers_try += self.getRecentPeers(limit * 5)

        # 如果没有指定起始时间，则从最后修改时间的前一天开始下载
        if since is None:
            since = self.settings.get("modified", 60 * 60 * 24) - 60 * 60 * 24

        # 如果设置了详细模式，则记录调试信息
        if config.verbose:
            self.log.debug(
                "CheckModifications: Try to get listModifications from peers: %s, connected: %s, since: %s" %
                (peers_try, peers_connected_num, since)
            )

        # 创建3个协程来执行更新操作
        updaters = []
        for i in range(3):
            updaters.append(gevent.spawn(self.updater, peers_try, queried, since))

        # 等待所有协程执行完毕，最长等待时间为10秒
        gevent.joinall(updaters, timeout=10)

        # 如果没有查询到任何信息，则启动另外3个线程
        if not queried:
            # 添加已连接的对等节点
            peers_try[0:0] = [peer for peer in self.getConnectedPeers() if peer.connection.connected]
            for _ in range(10):
                # 如果没有任何更新器完成，则再等待10秒
                gevent.joinall(updaters, timeout=10)
                if queried:
                    break

        # 记录查询到的信息和耗时
        self.log.debug("CheckModifications: Queried listModifications from: %s in %.3fs since %s" % (queried, time.time() - s, since))
        time.sleep(0.1)
        # 返回查询到的信息
        return queried

    # 从对等节点更新 content.json 并下载已更改的文件
    # 返回: None
    # 使用装饰器禁止并行执行
    @util.Noparallel()
    # 更新函数，可以选择是否通知、是否检查文件、起始时间
    def update(self, announce=False, check_files=False, since=None):
        # 重新加载 content.json 文件，不加载包含文件
        self.content_manager.loadContent("content.json", load_includes=False)  # Reload content.json
        # 重置内容更新时间
        self.content_updated = None  # Reset content updated time

        # 如果需要检查文件
        if check_files:
            # 快速检查文件大小，标记坏文件
            self.storage.updateBadFiles(quick_check=True)  # Quick check and mark bad files based on file size

        # 如果不在服务状态，返回 False
        if not self.isServing():
            return False

        # 更新 WebSocket
        self.updateWebsocket(updating=True)

        # 移除不在 content.json 中的文件
        self.checkBadFiles()

        # 如果需要通知
        if announce:
            # 强制通知更新
            self.announce(mode="update", force=True)

        # 完全更新，可以重置坏文件
        if check_files and since == 0:
            self.bad_files = {}

        # 检查修改
        queried = self.checkModifications(since)

        # 加载 content.json 文件，不加载包含文件
        changed, deleted = self.content_manager.loadContent("content.json", load_includes=False)

        # 如果存在坏文件
        if self.bad_files:
            # 记录坏文件
            self.log.debug("Bad files: %s" % self.bad_files)
            # 异步重试坏文件
            gevent.spawn(self.retryBadFiles, force=True)

        # 如果查询结果为空
        if len(queried) == 0:
            # 查询修改失败
            self.content_updated = False
        else:
            # 记录内容更新时间
            self.content_updated = time.time()

        # 更新 WebSocket
        self.updateWebsocket(updated=True)

    # 重新下载所有 content.json
    # 更新站点
    def redownloadContents(self):
        # 异步下载所有 content.json
        content_threads = []
        for inner_path in list(self.content_manager.contents.keys()):
            content_threads.append(self.needFile(inner_path, update=True, blocking=False))

        # 记录等待下载的 content.json 数量
        self.log.debug("Waiting %s content.json to finish..." % len(content_threads))
        # 等待所有下载完成
        gevent.joinall(content_threads)

    # 发布工作者
    # 在对等方更新 content.json
    # 使用装饰器禁止并行执行
    @util.Noparallel()
    # 复制该站点
    # 使用装饰器禁止并行执行
    @util.Noparallel()
    # 使用装饰器限制并行执行数量
    @util.Pooled(100)
    # 使用池化的方式异步下载文件
    def pooledNeedFile(self, *args, **kwargs):
        return self.needFile(*args, **kwargs)
    # 检查文件下载是否允许
    def isFileDownloadAllowed(self, inner_path, file_info):
        # 验证所有站点的空间限制
        if self.settings["size"] > self.getSizeLimit() * 1024 * 1024:
            return False
        # 验证文件的空间限制
        if file_info.get("size", 0) > config.file_size_limit * 1024 * 1024:
            self.log.debug(
                "File size %s too large: %sMB > %sMB, skipping..." %
                (inner_path, file_info.get("size", 0) / 1024 / 1024, config.file_size_limit)
            )
            return False
        else:
            return True

    # 获取文件信息
    def needFileInfo(self, inner_path):
        file_info = self.content_manager.getFileInfo(inner_path)
        if not file_info:
            # 文件没有信息，先下载所有的 content.json
            self.log.debug("No info for %s, waiting for all content.json" % inner_path)
            success = self.downloadContent("content.json", download_files=False)
            if not success:
                return False
            file_info = self.content_manager.getFileInfo(inner_path)
        return file_info

    # 检查并下载文件（如果文件不存在）
    # 添加或更新站点的对等节点
    # return_peer: 即使对等节点已经存在，也始终返回对等节点
    def addPeer(self, ip, port, return_peer=False, connection=None, source="other"):
        if not ip or ip == "0.0.0.0":
            return False

        key = "%s:%s" % (ip, port)
        peer = self.peers.get(key)
        if peer:  # 已经有这个 IP
            peer.found(source)
            if return_peer:  # 始终返回对等节点
                return peer
            else:
                return False
        else:  # 新的对等节点
            if (ip, port) in self.peer_blacklist:
                return False  # 忽略黑名单（例如自己）
            peer = Peer(ip, port, self)
            self.peers[key] = peer
            peer.found(source)
            return peer
    # 如果正在提供服务，则通知广播器进行通知
    def announce(self, *args, **kwargs):
        if self.isServing():
            self.announcer.announce(*args, **kwargs)

    # 保持连接以获取更新
    def needConnections(self, num=None, check_site_on_reconnect=False):
        if num is None:
            if len(self.peers) < 50:
                num = 3
            else:
                num = 6
        # 需要的连接数取决于当前已连接的对等节点数、指定的数量和配置的连接限制
        need = min(len(self.peers), num, config.connected_limit)  # Need 5 peer, but max total peers

        connected = len(self.getConnectedPeers())

        connected_before = connected

        # 记录需要的连接数、当前已连接的对等节点数和总对等节点数
        self.log.debug("Need connections: %s, Current: %s, Total: %s" % (need, connected, len(self.peers)))

        if connected < need:  # 需要更多连接
            for peer in self.getRecentPeers(30):
                if not peer.connection or not peer.connection.connected:  # 没有对等节点连接或已断开连接
                    peer.pex()  # 启动对等节点交换
                    if peer.connection and peer.connection.connected:
                        connected += 1  # 成功连接
                if connected >= need:
                    break
            # 记录连接前和连接后的对等节点数，以及在重新连接时是否检查站点
            self.log.debug(
                "Connected before: %s, after: %s. Check site: %s." %
                (connected_before, connected, check_site_on_reconnect)
            )

        # 如果在重新连接时需要检查站点，并且之前没有连接，现在有连接，并且连接服务器有互联网，则异步更新
        if check_site_on_reconnect and connected_before == 0 and connected > 0 and self.connection_server.has_internet:
            gevent.spawn(self.update, check_files=False)

        return connected

    # 返回：最近经过验证可以连接的对等节点
    # 获取可连接的对等节点列表，默认需要5个，可以指定忽略的节点列表，是否允许私有IP连接
    def getConnectablePeers(self, need_num=5, ignore=[], allow_private=True):
        # 获取所有对等节点
        peers = list(self.peers.values())
        found = []
        # 遍历所有对等节点
        for peer in peers:
            # 如果对等节点的key以":0"结尾，则跳过，表示不可连接
            if peer.key.endswith(":0"):
                continue  # Not connectable
            # 如果对等节点没有连接，则跳过
            if not peer.connection:
                continue  # No connection
            # 如果对等节点的IP以".onion"结尾，并且Tor管理器未启用，则跳过，表示不支持.onion连接
            if peer.ip.endswith(".onion") and not self.connection_server.tor_manager.enabled:
                continue  # Onion not supported
            # 如果对等节点在忽略列表中，则跳过
            if peer.key in ignore:
                continue  # The requester has this peer
            # 如果距离上次接收消息的时间超过2小时，则清除连接并跳过
            if time.time() - peer.connection.last_recv_time > 60 * 60 * 2:  # Last message more than 2 hours ago
                peer.connection = None  # Cleanup: Dead connection
                continue
            # 如果不允许私有IP连接，并且对等节点的IP是私有IP，则跳过
            if not allow_private and helper.isPrivateIp(peer.ip):
                continue
            # 将符合条件的对等节点加入到结果列表中
            found.append(peer)
            # 如果已经找到需要的数量的对等节点，则跳出循环
            if len(found) >= need_num:
                break  # Found requested number of peers

        # 如果找到的对等节点数量不足需要的数量，则返回一些较好的对等节点
        if len(found) < need_num:  # Return not that good peers
            found += [
                peer for peer in peers
                if not peer.key.endswith(":0") and
                peer.key not in ignore and
                (allow_private or not helper.isPrivateIp(peer.ip))
            ][0:need_num - len(found)]

        # 返回最终找到的对等节点列表
        return found

    # 返回：最近发现的对等节点
    # 获取最近的对等节点
    def getRecentPeers(self, need_num):
        # 从最近的对等节点列表中去重，得到找到的对等节点列表
        found = list(set(self.peers_recent))
        # 记录日志，显示最近对等节点数量、总对等节点数量和需要的对等节点数量
        self.log.debug(
            "Recent peers %s of %s (need: %s)" %
            (len(found), len(self.peers), need_num)
        )

        # 如果找到的对等节点数量大于等于需要的数量，或者大于等于总对等节点数量，则返回找到的对等节点列表
        if len(found) >= need_num or len(found) >= len(self.peers):
            return sorted(
                found,
                key=lambda peer: peer.reputation,
                reverse=True
            )[0:need_num]

        # 添加随机对等节点
        need_more = need_num - len(found)
        # 如果未启用 Tor 管理器，则选择非 .onion 结尾的对等节点
        if not self.connection_server.tor_manager.enabled:
            peers = [peer for peer in self.peers.values() if not peer.ip.endswith(".onion")]
        else:
            peers = list(self.peers.values())

        # 从对等节点中选择一定数量的对等节点，并按声誉值降序排序，取前 need_more*50 个
        found_more = sorted(
            peers[0:need_more * 50],
            key=lambda peer: peer.reputation,
            reverse=True
        )[0:need_more * 2]

        # 将新找到的对等节点添加到已找到的对等节点列表中
        found += found_more

        # 返回需要数量的对等节点列表
        return found[0:need_num]

    # 获取已连接的对等节点
    def getConnectedPeers(self):
        back = []
        # 如果未启用连接服务器，则返回空列表
        if not self.connection_server:
            return []

        tor_manager = self.connection_server.tor_manager
        # 遍历连接服务器的连接列表
        for connection in self.connection_server.connections:
            # 如果连接未建立且连接时间超过20秒，则跳过该连接
            if not connection.connected and time.time() - connection.start_time > 20:  # Still not connected after 20s
                continue
            # 根据连接的 IP 和端口获取对应的对等节点
            peer = self.peers.get("%s:%s" % (connection.ip, connection.port))
            if peer:
                # 如果连接的 IP 是 .onion 结尾且连接的目标是 .onion 地址，并且 Tor 管理器已启用
                if connection.ip.endswith(".onion") and connection.target_onion and tor_manager.start_onions:
                    # 检查连接是否是使用为该站点创建的 .onion 地址
                    valid_target_onions = (tor_manager.getOnion(self.address), tor_manager.getOnion("global"))
                    if connection.target_onion not in valid_target_onions:
                        continue
                # 如果对等节点未连接，则连接该对等节点
                if not peer.connection:
                    peer.connect(connection)
                # 将对等节点添加到返回列表中
                back.append(peer)
        # 返回已连接的对等节点列表
        return back
    # 清理可能已经断开的对等节点，并在连接过多时关闭连接
    # 清理对等节点，可选参数peers_protected默认为空列表
    def cleanupPeers(self, peers_protected=[]):
        # 获取所有对等节点的列表
        peers = list(self.peers.values())
        # 如果对等节点数量超过20个
        if len(peers) > 20:
            # 清理旧的对等节点
            removed = 0
            # 如果对等节点数量超过1000个
            if len(peers) > 1000:
                ttl = 60 * 60 * 1  # 设置存活时间为1小时
            else:
                ttl = 60 * 60 * 4  # 设置存活时间为4小时

            # 遍历对等节点
            for peer in peers:
                # 如果对等节点有连接并且连接处于连接状态，则继续下一个对等节点
                if peer.connection and peer.connection.connected:
                    continue
                # 如果对等节点有连接但连接不处于连接状态，则将连接置为None，表示连接已断开
                if peer.connection and not peer.connection.connected:
                    peer.connection = None  # 死连接
                # 如果当前时间减去对等节点发现时间大于存活时间ttl，则移除对等节点
                if time.time() - peer.time_found > ttl:  # 在最近4小时内未在tracker或pex中找到
                    peer.remove("Time found expired")
                    removed += 1
                # 如果移除的对等节点数量超过总对等节点数量的10%，则停止移除
                if removed > len(peers) * 0.1:  # 一次性不要移除太多
                    break

            # 如果有移除的对等节点
            if removed:
                self.log.debug("Cleanup peers result: Removed %s, left: %s" % (removed, len(self.peers)))

        # 关闭超出限制的对等节点
        closed = 0
        # 获取已连接的对等节点列表，且连接状态为已连接
        connected_peers = [peer for peer in self.getConnectedPeers() if peer.connection.connected]  # 只有完全连接的对等节点
        # 需要关闭的对等节点数量
        need_to_close = len(connected_peers) - config.connected_limit

        # 如果关闭的对等节点数量小于需要关闭的数量
        if closed < need_to_close:
            # 尝试保持与更多站点的连接
            # 根据连接的站点数量对对等节点进行排序
            for peer in sorted(connected_peers, key=lambda peer: min(peer.connection.sites, 5)):
                # 如果对等节点没有连接，则继续下一个对等节点
                if not peer.connection:
                    continue
                # 如果对等节点在受保护的对等节点列表中，则继续下一个对等节点
                if peer.key in peers_protected:
                    continue
                # 如果对等节点的站点数量大于5，则关闭连接
                if peer.connection.sites > 5:
                    break
                peer.connection.close("Cleanup peers")
                peer.connection = None
                closed += 1
                # 如果关闭的对等节点数量达到需要关闭的数量，则停止关闭
                if closed >= need_to_close:
                    break

        # 如果需要关闭的对等节点数量大于0
        if need_to_close > 0:
            self.log.debug("Connected: %s, Need to close: %s, Closed: %s" % (len(connected_peers), need_to_close, closed))
    # 将哈希字段发送给对等节点
    def sendMyHashfield(self, limit=5):
        # 如果没有可选文件，则返回 False
        if not self.content_manager.hashfield:  # No optional files
            return False

        sent = 0
        connected_peers = self.getConnectedPeers()
        for peer in connected_peers:
            # 如果成功发送哈希字段，则计数加一
            if peer.sendMyHashfield():
                sent += 1
                # 如果发送次数达到限制，则跳出循环
                if sent >= limit:
                    break
        # 如果成功发送了哈希字段
        if sent:
            # 记录发送哈希字段的时间，并输出日志
            my_hashfield_changed = self.content_manager.hashfield.time_changed
            self.log.debug("Sent my hashfield (chaged %.3fs ago) to %s peers" % (time.time() - my_hashfield_changed, sent))
        return sent

    # 更新哈希字段
    def updateHashfield(self, limit=5):
        # 如果没有可选文件且哈希字段没有改变，则返回 False
        if not self.content_manager.hashfield and not self.content_manager.has_optional_files:
            return False

        s = time.time()
        queried = 0
        connected_peers = self.getConnectedPeers()
        for peer in connected_peers:
            # 如果对等节点的哈希字段时间存在，则跳过
            if peer.time_hashfield:
                continue
            # 如果成功更新哈希字段，则计数加一
            if peer.updateHashfield():
                queried += 1
            # 如果更新次数达到限制，则跳出循环
            if queried >= limit:
                break
        # 如果成功更新了哈希字段
        if queried:
            # 输出日志，记录查询哈希字段的时间
            self.log.debug("Queried hashfield from %s peers in %.3fs" % (queried, time.time() - s))
        return queried

    # 返回可选文件是否需要下载
    def isDownloadable(self, inner_path):
        return self.settings.get("autodownloadoptional")
    # 删除站点的方法
    def delete(self):
        # 记录日志，表示正在删除站点
        self.log.info("Deleting site...")
        # 记录当前时间
        s = time.time()
        # 设置 serving 属性为 False，表示不再提供服务
        self.settings["serving"] = False
        # 设置 deleting 属性为 True，表示正在删除
        self.settings["deleting"] = True
        # 保存设置
        self.saveSettings()
        # 停止所有绿色线程
        num_greenlets = self.greenlet_manager.stopGreenlets("Site %s deleted" % self.address)
        # 停止所有工作线程
        self.worker_manager.running = False
        num_workers = self.worker_manager.stopWorkers()
        # 从站点管理器中删除该站点
        SiteManager.site_manager.delete(self.address)
        # 从内容管理器中删除该站点的内容
        self.content_manager.contents.db.deleteSite(self)
        # 更新 WebSocket，表示站点已删除
        self.updateWebsocket(deleted=True)
        # 删除站点的文件
        self.storage.deleteFiles()
        # 记录删除站点的信息
        self.log.info(
            "Deleted site in %.3fs (greenlets: %s, workers: %s)" %
            (time.time() - s, num_greenlets, num_workers)
        )

    # - 事件 -

    # 添加事件监听器
    def addEventListeners(self):
        # 当 WorkerManager 添加新任务时触发的事件
        self.onFileStart = util.Event()
        # 当 WorkerManager 成功下载文件时触发的事件
        self.onFileDone = util.Event()
        # 当 WorkerManager 下载文件失败时触发的事件
        self.onFileFail = util.Event()
        # 当所有文件完成时触发的事件
        self.onComplete = util.Event()

        # 添加事件监听器的回调函数
        self.onFileStart.append(lambda inner_path: self.fileStarted())
        self.onFileDone.append(lambda inner_path: self.fileDone(inner_path))
        self.onFileFail.append(lambda inner_path: self.fileFailed(inner_path))

    # 向 WebSocket 客户端发送站点状态更新
    def updateWebsocket(self, **kwargs):
        if kwargs:
            param = {"event": list(kwargs.items())[0]}
        else:
            param = None
        for ws in self.websockets:
            ws.event("siteChanged", self, param)

    # 向 WebSocket 客户端发送消息
    def messageWebsocket(self, message, type="info", progress=None):
        for ws in self.websockets:
            if progress is None:
                ws.cmd("notification", [type, message])
            else:
                ws.cmd("progress", [type, message, progress])
    # 文件下载开始
    @util.Noparallel(blocking=False)
    def fileStarted(self):
        time.sleep(0.001)  # 等待其他文件添加
        self.updateWebsocket(file_started=True)
    
    # 文件下载成功
    def fileDone(self, inner_path):
        # 文件下载成功，从坏文件列表中移除
        if inner_path in self.bad_files:
            if config.verbose:
                self.log.debug("Bad file solved: %s" % inner_path)
            del(self.bad_files[inner_path])
    
        # 更新 content.json 的最后下载时间
        if inner_path == "content.json":
            if not self.settings.get("downloaded"):
                self.settings["downloaded"] = int(time.time())
            self.content_updated = time.time()
    
        self.updateWebsocket(file_done=inner_path)
    
    # 文件下载失败
    def fileFailed(self, inner_path):
        if inner_path == "content.json":
            self.content_updated = False
            self.log.debug("Can't update content.json")
        if inner_path in self.bad_files and self.connection_server.has_internet:
            self.bad_files[inner_path] = self.bad_files.get(inner_path, 0) + 1
    
        self.updateWebsocket(file_failed=inner_path)
    
        if self.bad_files.get(inner_path, 0) > 30:
            self.fileForgot(inner_path)
    
    def fileForgot(self, inner_path):
        self.log.debug("Giving up on %s" % inner_path)
        del self.bad_files[inner_path]  # 经过30次尝试后放弃
```
# `ZeroNet\src\Site\SiteManager.py`

```py
# 导入所需的模块
import json
import logging
import re
import os
import time
import atexit

# 导入并使用协程库
import gevent

# 导入自定义的工具模块
import util
from Plugin import PluginManager
from Content import ContentDb
from Config import config
from util import helper
from util import RateLimit
from util import Cached

# 声明 SiteManager 类，并标记为可接受插件
@PluginManager.acceptPlugins
class SiteManager(object):
    # 初始化方法
    def __init__(self):
        # 创建日志记录器
        self.log = logging.getLogger("SiteManager")
        # 输出调试信息
        self.log.debug("SiteManager created.")
        # 初始化站点字典和站点变更时间
        self.sites = {}
        self.sites_changed = int(time.time())
        self.loaded = False
        # 使用协程创建保存定时器
        gevent.spawn(self.saveTimer)
        # 在程序退出时注册保存方法
        atexit.register(lambda: self.save(recalculate_size=True))

    # 延迟保存方法，用于异步调用
    @util.Noparallel()
    def saveDelayed(self):
        # 调用速率限制器异步保存站点信息到 sites.json 文件
        RateLimit.callAsync("Save sites.json", allowed_again=5, func=self.save)
    # 保存数据到文件，如果不重新计算大小则跳过
    def save(self, recalculate_size=False):
        # 如果没有站点，则跳过保存并记录日志
        if not self.sites:
            self.log.debug("Save skipped: No sites found")
            return
        # 如果数据没有加载，则跳过保存并记录日志
        if not self.loaded:
            self.log.debug("Save skipped: Not loaded")
            return
        # 记录开始时间
        s = time.time()
        # 初始化数据字典
        data = {}
        # 生成数据文件
        s = time.time()
        # 遍历站点列表，更新站点大小并添加到数据字典中
        for address, site in list(self.list().items()):
            if recalculate_size:
                site.settings["size"], site.settings["size_optional"] = site.content_manager.getTotalSize()  # Update site size
            data[address] = site.settings
            data[address]["cache"] = site.getSettingsCache()
        # 计算生成数据的时间
        time_generate = time.time() - s

        # 写入数据到文件
        s = time.time()
        if data:
            helper.atomicWrite("%s/sites.json" % config.data_dir, helper.jsonDumps(data).encode("utf8"))
        else:
            self.log.debug("Save error: No data")
        # 计算写入数据的时间
        time_write = time.time() - s

        # 清空站点设置中的缓存
        for address, site in self.list().items():
            site.settings["cache"] = {}

        # 记录保存站点的时间和生成数据的时间以及写入数据的时间
        self.log.debug("Saved sites in %.2fs (generate: %.2fs, write: %.2fs)" % (time.time() - s, time_generate, time_write))

    # 定时保存数据
    def saveTimer(self):
        while 1:
            time.sleep(60 * 10)
            self.save(recalculate_size=True)

    # 检查是否是有效的地址
    def isAddress(self, address):
        return re.match("^[A-Za-z0-9]{26,35}$", address)

    # 检查是否是域名
    def isDomain(self, address):
        return False

    # 使用缓存检查是否是域名
    @Cached(timeout=10)
    def isDomainCached(self, address):
        return self.isDomain(address)

    # 解析域名
    def resolveDomain(self, domain):
        return False

    # 使用缓存解析域名
    @Cached(timeout=10)
    def resolveDomainCached(self, domain):
        return self.resolveDomain(domain)

    # 返回站点对象或者如果找不到则返回None
    # 根据地址获取站点信息
    def get(self, address):
        # 如果地址已经被缓存，则使用缓存中的解析结果
        if self.isDomainCached(address):
            address_resolved = self.resolveDomainCached(address)
            if address_resolved:
                address = address_resolved

        # 如果站点未加载，则记录日志并加载站点
        if not self.loaded:  # Not loaded yet
            self.log.debug("Loading site: %s)..." % address)
            self.load()
        # 获取站点信息
        site = self.sites.get(address)

        return site

    # 添加新站点
    def add(self, address, all_file=True, settings=None, **kwargs):
        from .Site import Site
        # 更新站点变更时间
        self.sites_changed = int(time.time())
        # 尝试使用不同大小写的地址查找站点
        for recover_address, recover_site in list(self.sites.items()):
            if recover_address.lower() == address.lower():
                return recover_site

        # 如果地址不符合规范，则返回 False
        if not self.isAddress(address):
            return False  # Not address: %s % address
        # 记录日志，添加新站点
        self.log.debug("Added new site: %s" % address)
        config.loadTrackersFile()
        site = Site(address, settings=settings)
        self.sites[address] = site
        # 如果站点设置中的 serving 为 False，则设置为 True
        if not site.settings["serving"]:  # Maybe it was deleted before
            site.settings["serving"] = True
        site.saveSettings()
        # 如果 all_file 为 True，则下载用户文件
        if all_file:  # Also download user files on first sync
            site.download(check_size=True, blind_includes=True)
        return site

    # 返回或创建站点并开始下载站点文件
    def need(self, address, *args, **kwargs):
        # 如果地址已经被缓存，则使用缓存中的解析结果
        if self.isDomainCached(address):
            address_resolved = self.resolveDomainCached(address)
            if address_resolved:
                address = address_resolved

        # 获取站点信息
        site = self.get(address)
        # 如果站点不存在，则添加新站点
        if not site:  # Site not exist yet
            site = self.add(address, *args, **kwargs)
        return site

    # 删除站点
    def delete(self, address):
        # 更新站点变更时间
        self.sites_changed = int(time.time())
        # 记录日志，删除站点
        self.log.debug("Deleted site: %s" % address)
        del(self.sites[address])
        # 从 sites.json 中删除
        self.save()

    # 惰性加载站点
    # 定义一个方法，用于列出站点信息
    def list(self):
        # 如果站点信息还未加载
        if not self.loaded:  # Not loaded yet
            # 记录调试信息：站点信息尚未加载
            self.log.debug("Sites not loaded yet...")
            # 调用加载方法，参数为启动时加载
            self.load(startup=True)
        # 返回站点信息
        return self.sites
# 创建站点管理器对象，使用单例模式
site_manager = SiteManager()  # Singletone

# 如果配置动作为 "main"，则不连接/将自己添加到对等节点列表
if config.action == "main":  # Don't connect / add myself to peerlist
    # 设置对等节点黑名单，包括本地 IPv4 和 IPv6 地址
    peer_blacklist = [("127.0.0.1", config.fileserver_port), ("::1", config.fileserver_port)]
# 否则，对等节点黑名单为空列表
else:
    peer_blacklist = []
```
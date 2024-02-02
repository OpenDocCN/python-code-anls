# `ZeroNet\plugins\Chart\ChartCollector.py`

```py
# 导入时间模块
import time
# 导入系统模块
import sys
# 导入集合模块
import collections
# 导入迭代工具模块
import itertools
# 导入日志模块
import logging

# 导入协程模块
import gevent
# 导入工具模块中的帮助函数
from util import helper
# 导入配置模块中的配置
from Config import config

# 创建图表收集器类
class ChartCollector(object):
    # 初始化方法
    def __init__(self, db):
        # 设置数据库属性
        self.db = db
        # 如果配置中的动作为"main"，则延迟3分钟后执行收集器方法
        if config.action == "main":
            gevent.spawn_later(60 * 3, self.collector)
        # 创建日志对象
        self.log = logging.getLogger("ChartCollector")
        # 创建默认字典，用于存储上次数值
        self.last_values = collections.defaultdict(dict)

    # 设置初始的上次数值
    def setInitialLastValues(self, sites):
        # 恢复站点字节接收的上次数值
        for site in sites:
            self.last_values["site:" + site.address]["site_bytes_recv"] = site.settings.get("bytes_recv", 0)
            self.last_values["site:" + site.address]["site_bytes_sent"] = site.settings.get("bytes_sent", 0)

    # 获取站点收集器
    def getSiteCollectors(self):
        site_collectors = {}

        # 大小
        site_collectors["site_size"] = lambda site: site.settings.get("size", 0)
        site_collectors["site_size_optional"] = lambda site: site.settings.get("size_optional", 0)
        site_collectors["site_optional_downloaded"] = lambda site: site.settings.get("optional_downloaded", 0)
        site_collectors["site_content"] = lambda site: len(site.content_manager.contents)

        # 数据传输
        site_collectors["site_bytes_recv|change"] = lambda site: site.settings.get("bytes_recv", 0)
        site_collectors["site_bytes_sent|change"] = lambda site: site.settings.get("bytes_sent", 0)

        # 对等节点
        site_collectors["site_peer"] = lambda site: len(site.peers)
        site_collectors["site_peer_onion"] = lambda site: len(
            [True for peer in site.peers.values() if peer.ip.endswith(".onion")]
        )
        site_collectors["site_peer_connected"] = lambda site: len([True for peer in site.peers.values() if peer.connection])

        return site_collectors
    # 获取唯一的对等节点
    def getUniquePeers(self):
        # 导入 main 模块
        import main
        # 获取文件服务器的站点列表
        sites = main.file_server.sites
        # 返回所有站点中的对等节点的集合
        return set(itertools.chain.from_iterable(
            [site.peers.keys() for site in sites.values()]
        ))

    # 收集数据
    def collectDatas(self, collectors, last_values, site=None):
        # 如果站点为空，则获取唯一的对等节点
        if site is None:
            peers = self.getUniquePeers()
        # 创建一个空字典用于存储数据
        datas = {}
        # 遍历收集器字典
        for key, collector in collectors.items():
            try:
                # 尝试获取收集器的值
                if site:
                    value = collector(site)
                elif key.startswith("peer"):
                    value = collector(peers)
                else:
                    value = collector()
            except ValueError:
                value = None
            except Exception as err:
                # 如果出现异常，则记录错误信息
                self.log.info("Collector %s error: %s" % (key, err))
                value = None

            # 如果键中包含"|change"，则存储相对于上次值的变化
            if "|change" in key:
                key = key.replace("|change", "")
                last_value = last_values.get(key, 0)
                last_values[key] = value
                value = value - last_value

            # 如果值为空，则将键对应的值设为 None，否则保留三位小数
            if value is None:
                datas[key] = None
            else:
                datas[key] = round(value, 3)
        # 返回收集到的数据
        return datas

    # 收集全局数据
    def collectGlobal(self, collectors, last_values):
        # 获取当前时间戳
        now = int(time.time())
        s = time.time()
        # 收集数据
        datas = self.collectDatas(collectors, last_values["global"])
        values = []
        # 将数据转换为元组并添加到列表中
        for key, value in datas.items():
            values.append((self.db.getTypeId(key), value, now))
        # 记录全局收集器的执行时间
        self.log.debug("Global collectors done in %.3fs" % (time.time() - s))

        s = time.time()
        # 获取数据库游标
        cur = self.db.getCursor()
        # 批量插入数据到数据库
        cur.executemany("INSERT INTO data (type_id, value, date_added) VALUES (?, ?, ?)", values)
        # 记录全局收集器插入数据的执行时间
        self.log.debug("Global collectors inserted in %.3fs" % (time.time() - s))
    # 收集站点数据
    def collectSites(self, sites, collectors, last_values):
        # 获取当前时间戳
        now = int(time.time())
        # 记录开始时间
        s = time.time()
        # 初始化数值列表
        values = []
        # 遍历站点字典
        for address, site in list(sites.items()):
            # 收集站点数据
            site_datas = self.collectDatas(collectors, last_values["site:%s" % address], site)
            # 将站点数据添加到数值列表中
            for key, value in site_datas.items():
                values.append((self.db.getTypeId(key), self.db.getSiteId(address), value, now))
            # 等待一段时间
            time.sleep(0.001)
        # 记录站点收集完成的时间
        self.log.debug("Site collections done in %.3fs" % (time.time() - s))

        # 记录开始时间
        s = time.time()
        # 获取数据库游标
        cur = self.db.getCursor()
        # 批量插入站点收集数据
        cur.executemany("INSERT INTO data (type_id, site_id, value, date_added) VALUES (?, ?, ?, ?)", values)
        # 记录站点收集数据插入完成的时间
        self.log.debug("Site collectors inserted in %.3fs" % (time.time() - s))

    # 收集器方法
    def collector(self):
        # 获取所有收集器
        collectors = self.getCollectors()
        # 获取站点收集器
        site_collectors = self.getSiteCollectors()
        # 导入主模块
        import main
        # 获取站点字典
        sites = main.file_server.sites
        # 初始化计数器
        i = 0
        # 循环执行
        while 1:
            # 收集全局数据
            self.collectGlobal(collectors, self.last_values)
            # 每小时收集一次站点数据
            if i % 12 == 0:
                self.collectSites(sites, site_collectors, self.last_values)
            # 休眠5分钟
            time.sleep(60 * 5)
            # 计数器自增
            i += 1
```
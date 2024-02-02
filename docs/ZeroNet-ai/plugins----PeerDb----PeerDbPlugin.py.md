# `ZeroNet\plugins\PeerDb\PeerDbPlugin.py`

```py
# 导入时间模块
import time
# 导入 SQLite3 模块
import sqlite3
# 导入随机数模块
import random
# 导入 atexit 模块
import atexit

# 导入 gevent 模块
import gevent
# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager

# 将 ContentDbPlugin 类注册到 PluginManager 的 ContentDb 插件中
@PluginManager.registerTo("ContentDb")
class ContentDbPlugin(object):
    # 初始化方法
    def __init__(self, *args, **kwargs):
        # 在程序退出时注册保存所有对等节点的方法
        atexit.register(self.saveAllPeers)
        # 调用父类的初始化方法
        super(ContentDbPlugin, self).__init__(*args, **kwargs)

    # 获取数据库模式的方法
    def getSchema(self):
        # 调用父类的获取数据库模式的方法
        schema = super(ContentDbPlugin, self).getSchema()

        # 定义 peer 表的列和索引
        schema["tables"]["peer"] = {
            "cols": [
                ["site_id", "INTEGER REFERENCES site (site_id) ON DELETE CASCADE"],
                ["address", "TEXT NOT NULL"],
                ["port", "INTEGER NOT NULL"],
                ["hashfield", "BLOB"],
                ["reputation", "INTEGER NOT NULL"],
                ["time_added", "INTEGER NOT NULL"],
                ["time_found", "INTEGER NOT NULL"]
            ],
            "indexes": [
                "CREATE UNIQUE INDEX peer_key ON peer (site_id, address, port)"
            ],
            "schema_changed": 2
        }

        # 返回数据库模式
        return schema
    # 加载指定站点的对等节点信息
    def loadPeers(self, site):
        # 记录开始时间
        s = time.time()
        # 获取站点对应的 ID
        site_id = self.site_ids.get(site.address)
        # 执行 SQL 查询，获取对应站点的所有对等节点信息
        res = self.execute("SELECT * FROM peer WHERE site_id = :site_id", {"site_id": site_id})
        # 初始化计数器
        num = 0
        num_hashfield = 0
        # 遍历查询结果
        for row in res:
            # 向站点添加对等节点
            peer = site.addPeer(str(row["address"]), row["port"])
            # 如果对等节点已存在，则跳过
            if not peer:  # Already exist
                continue
            # 如果对等节点具有哈希字段，则替换哈希字段
            if row["hashfield"]:
                peer.hashfield.replaceFromBytes(row["hashfield"])
                num_hashfield += 1
            # 设置对等节点的添加时间、发现时间和声誉
            peer.time_added = row["time_added"]
            peer.time_found = row["time_found"]
            peer.reputation = row["reputation"]
            # 如果对等节点的地址以 ".onion" 结尾，则调整其声誉
            if row["address"].endswith(".onion"):
                peer.reputation = peer.reputation / 2 - 1 # Onion peers less likely working
            # 更新对等节点计数
            num += 1
        # 如果存在具有哈希字段的对等节点，则设置站点的可选文件标志为 True
        if num_hashfield:
            site.content_manager.has_optional_files = True
        # 记录加载对等节点的时间
        site.log.debug("%s peers (%s with hashfield) loaded in %.3fs" % (num, num_hashfield, time.time() - s))

    # 迭代指定站点的对等节点信息
    def iteratePeers(self, site):
        # 获取站点对应的 ID
        site_id = self.site_ids.get(site.address)
        # 遍历站点的对等节点
        for key, peer in list(site.peers.items()):
            # 解析对等节点的地址和端口
            address, port = key.rsplit(":", 1)
            # 如果对等节点具有哈希字段，则转换为二进制格式
            if peer.has_hashfield:
                hashfield = sqlite3.Binary(peer.hashfield.tobytes())
            else:
                hashfield = ""
            # 生成对等节点信息的迭代器
            yield (site_id, address, port, hashfield, peer.reputation, int(peer.time_added), int(peer.time_found))
    # 保存对等节点信息到数据库，可选择是否异步执行
    def savePeers(self, site, spawn=False):
        if spawn:
            # 如果需要异步执行，则在一小时后再次调用 savePeers 方法
            site.greenlet_manager.spawnLater(60 * 60 + random.randint(0, 60), self.savePeers, site, spawn=True)
        # 如果没有对等节点信息，则记录日志并返回
        if not site.peers:
            site.log.debug("Peers not saved: No peers found")
            return
        # 记录当前时间
        s = time.time()
        # 获取站点 ID
        site_id = self.site_ids.get(site.address)
        # 获取数据库游标
        cur = self.getCursor()
        try:
            # 删除对应站点 ID 的 peer 表中的数据
            cur.execute("DELETE FROM peer WHERE site_id = :site_id", {"site_id": site_id})
            # 批量插入对等节点信息到数据库
            cur.executemany(
                "INSERT INTO peer (site_id, address, port, hashfield, reputation, time_added, time_found) VALUES (?, ?, ?, ?, ?, ?, ?)",
                self.iteratePeers(site)
            )
        except Exception as err:
            # 记录保存对等节点信息出错的日志
            site.log.error("Save peer error: %s" % err)
        # 记录保存对等节点信息所花费的时间
        site.log.debug("Peers saved in %.3fs" % (time.time() - s))

    # 初始化站点
    def initSite(self, site):
        # 调用父类的 initSite 方法
        super(ContentDbPlugin, self).initSite(site)
        # 在 0.5 秒后异步执行 loadPeers 方法
        site.greenlet_manager.spawnLater(0.5, self.loadPeers, site)
        # 每隔一小时异步执行一次 savePeers 方法
        site.greenlet_manager.spawnLater(60*60, self.savePeers, site, spawn=True)

    # 保存所有站点的对等节点信息
    def saveAllPeers(self):
        # 遍历所有站点
        for site in list(self.sites.values()):
            try:
                # 保存对等节点信息
                self.savePeers(site)
            except Exception as err:
                # 记录保存对等节点信息出错的日志
                site.log.error("Save peer error: %s" % err)
```
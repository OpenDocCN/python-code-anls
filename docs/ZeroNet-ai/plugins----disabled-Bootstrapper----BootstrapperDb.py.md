# `ZeroNet\plugins\disabled-Bootstrapper\BootstrapperDb.py`

```
# 导入时间模块
import time
# 导入正则表达式模块
import re
# 导入协程模块
import gevent
# 从Config模块中导入config变量
from Config import config
# 从Db模块中导入Db类
from Db import Db
# 从util模块中导入helper函数
from util import helper

# BootstrapperDb类继承自Db类
class BootstrapperDb(Db.Db):
    # 初始化方法
    def __init__(self):
        # 设置版本号为7
        self.version = 7
        # 初始化哈希表，用于缓存哈希值对应的id
        self.hash_ids = {}  # hash -> id cache
        # 调用父类的初始化方法，传入数据库名称和路径
        super(BootstrapperDb, self).__init__({"db_name": "Bootstrapper"}, "%s/bootstrapper.db" % config.data_dir)
        # 开启外键约束
        self.foreign_keys = True
        # 检查数据库表
        self.checkTables()
        # 更新哈希缓存
        self.updateHashCache()
        # 创建一个协程，执行cleanup方法
        gevent.spawn(self.cleanup)

    # 清理方法
    def cleanup(self):
        # 循环执行
        while 1:
            # 休眠4分钟
            time.sleep(4 * 60)
            # 计算超时时间
            timeout = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 60 * 40))
            # 删除peer表中满足条件的记录
            self.execute("DELETE FROM peer WHERE date_announced < ?", [timeout])

    # 更新哈希缓存方法
    def updateHashCache(self):
        # 从数据库中查询哈希表的所有记录
        res = self.execute("SELECT * FROM hash")
        # 将查询结果转换为哈希值到id的字典
        self.hash_ids = {row["hash"]: row["hash_id"] for row in res}
        # 记录日志，显示加载了多少个哈希id
        self.log.debug("Loaded %s hash_ids" % len(self.hash_ids))

    # 检查数据库表方法
    def checkTables(self):
        # 获取数据库的版本号
        version = int(self.execute("PRAGMA user_version").fetchone()[0])
        # 记录日志，显示数据库版本和所需版本
        self.log.debug("Db version: %s, needed: %s" % (version, self.version))
        # 如果数据库版本低于所需版本，则创建表
        if version < self.version:
            self.createTables()
        # 否则，执行VACUUM命令
        else:
            self.execute("VACUUM")
    # 创建数据库表
    def createTables(self):
        # 设置可写模式，删除所有表
        self.execute("PRAGMA writable_schema = 1")
        self.execute("DELETE FROM sqlite_master WHERE type IN ('table', 'index', 'trigger')")
        self.execute("PRAGMA writable_schema = 0")
        self.execute("VACUUM")
        self.execute("PRAGMA INTEGRITY_CHECK")
        # 创建新表
        self.execute("""
            CREATE TABLE peer (
                peer_id        INTEGER PRIMARY KEY ASC AUTOINCREMENT NOT NULL UNIQUE,
                type           TEXT,
                address        TEXT,
                port           INTEGER NOT NULL,
                date_added     DATETIME DEFAULT (CURRENT_TIMESTAMP),
                date_announced DATETIME DEFAULT (CURRENT_TIMESTAMP)
            );
        """)
        # 创建 peer 表的唯一索引
        self.execute("CREATE UNIQUE INDEX peer_key ON peer (address, port);")

        # 创建 peer_to_hash 表
        self.execute("""
            CREATE TABLE peer_to_hash (
                peer_to_hash_id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL,
                peer_id         INTEGER REFERENCES peer (peer_id) ON DELETE CASCADE,
                hash_id         INTEGER REFERENCES hash (hash_id)
            );
        """)
        # 创建 peer_to_hash 表的索引
        self.execute("CREATE INDEX peer_id ON peer_to_hash (peer_id);")
        self.execute("CREATE INDEX hash_id ON peer_to_hash (hash_id);")

        # 创建 hash 表
        self.execute("""
            CREATE TABLE hash (
                hash_id    INTEGER  PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL,
                hash       BLOB     UNIQUE NOT NULL,
                date_added DATETIME DEFAULT (CURRENT_TIMESTAMP)
            );
        """)
        # 设置数据库用户版本
        self.execute("PRAGMA user_version = %s" % self.version)

    # 获取哈希值的 ID
    def getHashId(self, hash):
        # 如果哈希值不在哈希 ID 字典中，则插入新的哈希值并返回其 ID
        if hash not in self.hash_ids:
            self.log.debug("New hash: %s" % repr(hash))
            res = self.execute("INSERT OR IGNORE INTO hash ?", {"hash": hash})
            self.hash_ids[hash] = res.lastrowid
        return self.hash_ids[hash]
    # 定义一个方法用于向对等节点发送通告
    def peerAnnounce(self, ip_type, address, port=None, hashes=[], onion_signed=False, delete_missing_hashes=False):
        # 用于存储已通告的哈希ID列表
        hashes_ids_announced = []
        # 遍历传入的哈希列表，获取哈希ID并添加到已通告的哈希ID列表中
        for hash in hashes:
            hashes_ids_announced.append(self.getHashId(hash))

        # 检查用户是否存在
        res = self.execute("SELECT peer_id FROM peer WHERE ? LIMIT 1", {"address": address, "port": port})
        user_row = res.fetchone()
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        # 如果用户存在
        if user_row:
            peer_id = user_row["peer_id"]
            # 更新用户的通告日期
            self.execute("UPDATE peer SET date_announced = ? WHERE peer_id = ?", (now, peer_id))
        # 如果用户不存在
        else:
            # 记录新用户的信息
            self.log.debug("New peer: %s signed: %s" % (address, onion_signed))
            # 如果是onion类型的IP并且未签名，则返回哈希列表的长度
            if ip_type == "onion" and not onion_signed:
                return len(hashes)
            # 否则将新用户信息插入数据库，并获取插入的peer_id
            res = self.execute("INSERT INTO peer ?", {"type": ip_type, "address": address, "port": port, "date_announced": now})
            peer_id = res.lastrowid

        # 检查用户的哈希列表
        res = self.execute("SELECT * FROM peer_to_hash WHERE ?", {"peer_id": peer_id})
        hash_ids_db = [row["hash_id"] for row in res]
        # 如果数据库中的哈希ID列表与已通告的哈希ID列表不一致
        if hash_ids_db != hashes_ids_announced:
            # 计算新增的哈希ID和被移除的哈希ID
            hash_ids_added = set(hashes_ids_announced) - set(hash_ids_db)
            hash_ids_removed = set(hash_ids_db) - set(hashes_ids_announced)
            # 如果不是onion类型的IP或者已签名
            if ip_type != "onion" or onion_signed:
                # 将新增的哈希ID插入到数据库中
                for hash_id in hash_ids_added:
                    self.execute("INSERT INTO peer_to_hash ?", {"peer_id": peer_id, "hash_id": hash_id})
                # 如果存在被移除的哈希ID并且需要删除缺失的哈希
                if hash_ids_removed and delete_missing_hashes:
                    self.execute("DELETE FROM peer_to_hash WHERE ?", {"peer_id": peer_id, "hash_id": list(hash_ids_removed)})

            # 返回新增的哈希ID数量加上被移除的哈希ID数量
            return len(hash_ids_added) + len(hash_ids_removed)
        # 如果数据库中的哈希ID列表与已通告的哈希ID列表一致
        else:
            # 返回0
            return 0
    # 返回与给定哈希相关的对等节点列表
    def peerList(self, hash, address=None, onions=[], port=None, limit=30, need_types=["ipv4", "onion"], order=True):
        # 初始化返回结果字典
        back = {"ipv4": [], "ipv6": [], "onion": []}
        # 如果限制为0，则直接返回空字典
        if limit == 0:
            return back
        # 获取哈希的哈希ID
        hashid = self.getHashId(hash)

        # 根据order参数确定SQL语句中的排序方式
        if order:
            order_sql = "ORDER BY date_announced DESC"
        else:
            order_sql = ""
        # 构建where条件的SQL语句
        where_sql = "hash_id = :hashid"
        # 如果指定了onions，则构建排除指定onions的条件
        if onions:
            onions_escaped = ["'%s'" % re.sub("[^a-z0-9,]", "", onion) for onion in onions if type(onion) is str]
            where_sql += " AND address NOT IN (%s)" % ",".join(onions_escaped)
        # 如果指定了address，则构建排除指定address和port的条件
        elif address:
            where_sql += " AND NOT (address = :address AND port = :port)"

        # 构建查询语句
        query = """
            SELECT type, address, port
            FROM peer_to_hash
            LEFT JOIN peer USING (peer_id)
            WHERE %s
            %s
            LIMIT :limit
        """ % (where_sql, order_sql)
        # 执行查询
        res = self.execute(query, {"hashid": hashid, "address": address, "port": port, "limit": limit})

        # 遍历查询结果，根据need_types筛选并打包地址信息，添加到返回结果字典中
        for row in res:
            if row["type"] in need_types:
                if row["type"] == "onion":
                    packed = helper.packOnionAddress(row["address"], row["port"])
                else:
                    packed = helper.packAddress(str(row["address"]), row["port"])
                back[row["type"]].append(packed)
        # 返回结果字典
        return back
```
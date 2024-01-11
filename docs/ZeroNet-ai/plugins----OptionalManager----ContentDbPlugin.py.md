# `ZeroNet\plugins\OptionalManager\ContentDbPlugin.py`

```
# 导入时间、集合、迭代器、正则表达式模块
import time
import collections
import itertools
import re

# 导入协程模块
import gevent

# 从自定义模块中导入帮助函数
from util import helper
# 从插件管理模块中导入插件管理器
from Plugin import PluginManager
# 从配置模块中导入配置
from Config import config
# 从调试模块中导入调试工具
from Debug import Debug

# 如果 content_db 不在当前作用域中，则将其初始化为 None，以便在模块重新加载时保持不变
if "content_db" not in locals().keys():  
    content_db = None

# 将 ContentDbPlugin 类注册到插件管理器中
@PluginManager.registerTo("ContentDb")
class ContentDbPlugin(object):
    # 初始化方法
    def __init__(self, *args, **kwargs):
        # 声明 content_db 为全局变量，以便在整个模块中使用
        global content_db
        # 将 content_db 初始化为当前实例
        content_db = self
        # 存储已经填充的站点地址的字典
        self.filled = {}
        # 是否需要填充文件的标志
        self.need_filling = False
        # 最后一次更新时间对等数的时间戳
        self.time_peer_numbers_updated = 0
        # 最近 50 个由 fileWrite 调用的站点地址/内部路径的字典
        self.my_optional_files = {}
        # 默认字典，用于存储可选文件
        self.optional_files = collections.defaultdict(dict)
        # 可选文件是否已加载的标志
        self.optional_files_loaded = False
        # 检查可选文件限制的定时器
        self.timer_check_optional = helper.timer(60 * 5, self.checkOptionalLimit)
        # 调用父类的初始化方法
        super(ContentDbPlugin, self).__init__(*args, **kwargs)
    # 获取数据库模式
    def getSchema(self):
        # 调用父类方法获取数据库模式
        schema = super(ContentDbPlugin, self).getSchema()

        # 需要创建 file_optional 表
        schema["tables"]["file_optional"] = {
            "cols": [
                ["file_id", "INTEGER PRIMARY KEY UNIQUE NOT NULL"],
                ["site_id", "INTEGER REFERENCES site (site_id) ON DELETE CASCADE"],
                ["inner_path", "TEXT"],
                ["hash_id", "INTEGER"],
                ["size", "INTEGER"],
                ["peer", "INTEGER DEFAULT 0"],
                ["uploaded", "INTEGER DEFAULT 0"],
                ["is_downloaded", "INTEGER DEFAULT 0"],
                ["is_pinned", "INTEGER DEFAULT 0"],
                ["time_added", "INTEGER DEFAULT 0"],
                ["time_downloaded", "INTEGER DEFAULT 0"],
                ["time_accessed", "INTEGER DEFAULT 0"]
            ],
            "indexes": [
                "CREATE UNIQUE INDEX file_optional_key ON file_optional (site_id, inner_path)",
                "CREATE INDEX is_downloaded ON file_optional (is_downloaded)"
            ],
            "schema_changed": 11
        }

        return schema

    # 初始化站点
    def initSite(self, site):
        # 调用父类方法初始化站点
        super(ContentDbPlugin, self).initSite(site)
        # 如果需要填充数据表
        if self.need_filling:
            self.fillTableFileOptional(site)

    # 检查数据表
    def checkTables(self):
        # 调用父类方法检查数据表
        changed_tables = super(ContentDbPlugin, self).checkTables()
        # 如果 file_optional 表在修改的表中
        if "file_optional" in changed_tables:
            self.need_filling = True
        return changed_tables

    # 加载可选文件结束
    # 预测文件是否为可选文件
    def isOptionalFile(self, site_id, inner_path):
        return self.optional_files[site_id].get(inner_path[-8:])

    # 用在站点中找到的可选文件填充 file_optional 表
    # 填充文件可选表格，接受一个站点对象作为参数
    def fillTableFileOptional(self, site):
        # 记录当前时间
        s = time.time()
        # 获取站点地址对应的站点ID
        site_id = self.site_ids.get(site.address)
        # 如果站点ID不存在，则返回 False
        if not site_id:
            return False
        # 获取数据库游标
        cur = self.getCursor()
        # 执行 SQL 查询，选择文件可选大小大于0且站点ID匹配的内容
        res = cur.execute("SELECT * FROM content WHERE size_files_optional > 0 AND site_id = %s" % site_id)
        # 初始化计数器
        num = 0
        # 遍历查询结果的每一行
        for row in res.fetchall():
            # 获取站点内容管理器中指定内部路径的内容
            content = site.content_manager.contents[row["inner_path"]]
            try:
                # 调用 setContentFilesOptional 方法，处理站点、内部路径、内容和游标
                num += self.setContentFilesOptional(site, row["inner_path"], content, cur=cur)
            except Exception as err:
                # 记录错误日志
                self.log.error("Error loading %s into file_optional: %s" % (row["inner_path"], err))
        # 关闭游标
        cur.close()

        # 设置我的文件为已固定
        from User import UserManager
        # 获取当前用户，如果不存在则创建一个新用户
        user = UserManager.user_manager.get()
        if not user:
            user = UserManager.user_manager.create()
        # 获取用户的授权地址
        auth_address = user.getAuthAddress(site.address)
        # 执行 SQL 更新，将文件可选表中站点ID和内部路径匹配的记录设置为已固定
        res = self.execute(
            "UPDATE file_optional SET is_pinned = 1 WHERE site_id = :site_id AND inner_path LIKE :inner_path",
            {"site_id": site_id, "inner_path": "%%/%s/%%" % auth_address}
        )

        # 记录调试日志，包括填充文件可选表格的站点地址、耗时、加载的数量和已固定的数量
        self.log.debug(
            "Filled file_optional table for %s in %.3fs (loaded: %s, is_pinned: %s)" %
            (site.address, time.time() - s, num, res.rowcount)
        )
        # 将站点地址标记为已填充
        self.filled[site.address] = True
    # 设置内容的可选文件，如果没有指定当前操作的对象，则默认为 self
    def setContentFilesOptional(self, site, content_inner_path, content, cur=None):
        # 如果没有指定当前操作的对象，则默认为 self
        if not cur:
            cur = self

        # 初始化计数器
        num = 0
        # 获取站点 ID
        site_id = self.site_ids[site.address]
        # 获取内容内部目录
        content_inner_dir = helper.getDirname(content_inner_path)
        # 遍历内容中的可选文件
        for relative_inner_path, file in content.get("files_optional", {}).items():
            # 获取文件的内部路径
            file_inner_path = content_inner_dir + relative_inner_path
            # 计算文件哈希值的 ID
            hash_id = int(file["sha512"][0:4], 16)
            # 判断文件是否已经下载
            if hash_id in site.content_manager.hashfield:
                is_downloaded = 1
            else:
                is_downloaded = 0
            # 判断文件是否被固定
            if site.address + "/" + content_inner_dir in self.my_optional_files:
                is_pinned = 1
            else:
                is_pinned = 0
            # 插入或更新可选文件信息
            cur.insertOrUpdate("file_optional", {
                "hash_id": hash_id,
                "size": int(file["size"])
            }, {
                "site_id": site_id,
                "inner_path": file_inner_path
            }, oninsert={
                "time_added": int(time.time()),
                "time_downloaded": int(time.time()) if is_downloaded else 0,
                "is_downloaded": is_downloaded,
                "peer": is_downloaded,
                "is_pinned": is_pinned
            })
            # 更新可选文件列表
            self.optional_files[site_id][file_inner_path[-8:]] = 1
            # 更新计数器
            num += 1

        # 返回处理的文件数量
        return num
    # 设置内容的方法，包括站点、内部路径、内容、大小（默认为0）
    def setContent(self, site, inner_path, content, size=0):
        # 调用父类的setContent方法，设置站点、内部路径、内容、大小
        super(ContentDbPlugin, self).setContent(site, inner_path, content, size=size)
        # 获取旧内容
        old_content = site.content_manager.contents.get(inner_path, {})
        # 如果不需要填充或者已经填充，并且内容中包含"files_optional"，或者旧内容中包含"files_optional"
        if (not self.need_filling or self.filled.get(site.address)) and ("files_optional" in content or "files_optional" in old_content):
            # 设置内容的可选文件
            self.setContentFilesOptional(site, inner_path, content)
            # 检查已删除的文件
            if old_content:
                # 获取旧内容中的文件名集合
                old_files = old_content.get("files_optional", {}).keys()
                # 获取新内容中的文件名集合
                new_files = content.get("files_optional", {}).keys()
                # 获取内容的内部目录
                content_inner_dir = helper.getDirname(inner_path)
                # 找出已删除的文件路径
                deleted = [content_inner_dir + key for key in old_files if key not in new_files]
                # 获取站点ID
                site_id = self.site_ids[site.address]
                # 从数据库中删除已删除的文件
                self.execute("DELETE FROM file_optional WHERE ?", {"site_id": site_id, "inner_path": deleted})

    # 删除内容的方法，包括站点和内部路径
    def deleteContent(self, site, inner_path):
        # 获取内容
        content = site.content_manager.contents.get(inner_path)
        # 如果内容存在并且包含"files_optional"
        if content and "files_optional" in content:
            # 获取站点ID
            site_id = self.site_ids[site.address]
            # 获取内容的内部目录
            content_inner_dir = helper.getDirname(inner_path)
            # 获取所有可选文件的内部路径
            optional_inner_paths = [
                content_inner_dir + relative_inner_path
                for relative_inner_path in content.get("files_optional", {}).keys()
            ]
            # 从数据库中删除可选文件
            self.execute("DELETE FROM file_optional WHERE ?", {"site_id": site_id, "inner_path": optional_inner_paths})
        # 调用父类的deleteContent方法，删除站点和内部路径对应的内容
        super(ContentDbPlugin, self).deleteContent(site, inner_path)
    # 更新对等节点的数量信息
    def updatePeerNumbers(self):
        # 记录开始时间
        s = time.time()
        # 初始化文件数量、更新的文件数量、站点数量
        num_file = 0
        num_updated = 0
        num_site = 0
        # 遍历所有站点
        for site in list(self.sites.values()):
            # 如果站点没有可选文件，则跳过
            if not site.content_manager.has_optional_files:
                continue
            # 如果站点不在服务状态，则跳过
            if not site.isServing():
                continue
            # 检查站点中是否有更新的哈希字段
            has_updated_hashfield = next((
                peer
                for peer in site.peers.values()
                if peer.has_hashfield and peer.hashfield.time_changed > self.time_peer_numbers_updated
            ), None)

            # 如果没有更新的哈希字段，并且站点哈希字段的修改时间早于对等节点数量更新时间，则跳过
            if not has_updated_hashfield and site.content_manager.hashfield.time_changed < self.time_peer_numbers_updated:
                continue

            # 获取站点中所有对等节点的哈希字段
            hashfield_peers = itertools.chain.from_iterable(
                peer.hashfield.storage
                for peer in site.peers.values()
                if peer.has_hashfield
            )
            # 统计对等节点的数量
            peer_nums = collections.Counter(
                itertools.chain(
                    hashfield_peers,
                    site.content_manager.hashfield
                )
            )

            # 获取站点的ID
            site_id = self.site_ids[site.address]
            # 如果站点ID不存在，则跳过
            if not site_id:
                continue

            # 从数据库中查询文件ID、哈希ID、对等节点信息
            res = self.execute("SELECT file_id, hash_id, peer FROM file_optional WHERE ?", {"site_id": site_id})
            updates = {}
            # 遍历查询结果
            for row in res:
                # 获取哈希ID对应的对等节点数量
                peer_num = peer_nums.get(row["hash_id"], 0)
                # 如果对等节点数量不等于数据库中记录的对等节点数量，则更新字典
                if peer_num != row["peer"]:
                    updates[row["file_id"]] = peer_num

            # 更新数据库中的对等节点数量信息
            for file_id, peer_num in updates.items():
                self.execute("UPDATE file_optional SET peer = ? WHERE file_id = ?", (peer_num, file_id))

            # 更新统计信息
            num_updated += len(updates)
            num_file += len(peer_nums)
            num_site += 1

        # 更新对等节点数量更新时间
        self.time_peer_numbers_updated = time.time()
        # 记录日志
        self.log.debug("%s/%s peer number for %s site updated in %.3fs" % (num_updated, num_file, num_site, time.time() - s))
    # 查询可删除的文件
    def queryDeletableFiles(self):
        # 首先返回至少有10个种子并且在上周未被访问的文件
        query = """
            SELECT * FROM file_optional
            WHERE peer > 10 AND %s
            ORDER BY time_accessed < %s DESC, uploaded / size
        """ % (self.getOptionalUsedWhere(), int(time.time() - 60 * 60 * 7))
        limit_start = 0
        while 1:
            num = 0
            res = self.execute("%s LIMIT %s, 50" % (query, limit_start))
            for row in res:
                yield row
                num += 1
            if num < 50:
                break
            limit_start += 50

        self.log.debug("queryDeletableFiles returning less-seeded files")

        # 然后返回种子较少但仍未在上周访问的文件
        query = """
            SELECT * FROM file_optional
            WHERE peer <= 10 AND %s
            ORDER BY peer DESC, time_accessed < %s DESC, uploaded / size
        """ % (self.getOptionalUsedWhere(), int(time.time() - 60 * 60 * 7))
        limit_start = 0
        while 1:
            num = 0
            res = self.execute("%s LIMIT %s, 50" % (query, limit_start))
            for row in res:
                yield row
                num += 1
            if num < 50:
                break
            limit_start += 50

        self.log.debug("queryDeletableFiles returning everyting")

        # 最后返回所有文件
        query = """
            SELECT * FROM file_optional
            WHERE peer <= 10 AND %s
            ORDER BY peer DESC, time_accessed, uploaded / size
        """ % self.getOptionalUsedWhere()
        limit_start = 0
        while 1:
            num = 0
            res = self.execute("%s LIMIT %s, 50" % (query, limit_start))
            for row in res:
                yield row
                num += 1
            if num < 50:
                break
            limit_start += 50
    # 获取可选限制的字节数
    def getOptionalLimitBytes(self):
        # 如果配置的可选限制以百分比结尾
        if config.optional_limit.endswith("%"):
            # 获取百分比数值
            limit_percent = float(re.sub("[^0-9.]", "", config.optional_limit))
            # 计算限制的字节数
            limit_bytes = helper.getFreeSpace() * (limit_percent / 100)
        else:
            # 将配置的限制转换为字节数
            limit_bytes = float(re.sub("[^0-9.]", "", config.optional_limit)) * 1024 * 1024 * 1024
        return limit_bytes

    # 获取可选文件的使用条件
    def getOptionalUsedWhere(self):
        # 将配置的最小文件大小转换为字节数
        maxsize = config.optional_limit_exclude_minsize * 1024 * 1024
        # 构建查询条件
        query = "is_downloaded = 1 AND is_pinned = 0 AND size < %s" % maxsize

        # 不删除拥有站点的可选文件
        my_site_ids = []
        for address, site in self.sites.items():
            if site.settings["own"]:
                my_site_ids.append(str(self.site_ids[address]))

        if my_site_ids:
            # 添加站点ID的条件
            query += " AND site_id NOT IN (%s)" % ", ".join(my_site_ids)
        return query

    # 获取可选文件的使用字节数
    def getOptionalUsedBytes(self):
        # 执行查询获取可选文件的总大小
        size = self.execute("SELECT SUM(size) FROM file_optional WHERE %s" % self.getOptionalUsedWhere()).fetchone()[0]
        if not size:
            size = 0
        return size

    # 获取需要删除的可选文件大小
    def getOptionalNeedDelete(self, size):
        # 如果配置的可选限制以百分比结尾
        if config.optional_limit.endswith("%"):
            # 获取百分比数值
            limit_percent = float(re.sub("[^0-9.]", "", config.optional_limit))
            # 计算需要删除的大小
            need_delete = size - ((helper.getFreeSpace() + size) * (limit_percent / 100))
        else:
            # 计算需要删除的大小
            need_delete = size - self.getOptionalLimitBytes()
        return need_delete
    # 检查可选限制，如果未提供限制值，则获取默认限制值
    def checkOptionalLimit(self, limit=None):
        # 如果未提供限制值，则获取默认限制值
        if not limit:
            limit = self.getOptionalLimitBytes()

        # 如果限制值小于0，则记录错误并返回False
        if limit < 0:
            self.log.debug("Invalid limit for optional files: %s" % limit)
            return False

        # 获取已使用的可选文件大小
        size = self.getOptionalUsedBytes()

        # 计算需要删除的文件大小
        need_delete = self.getOptionalNeedDelete(size)

        # 记录可选文件大小、限制值和需要删除的大小
        self.log.debug(
            "Optional size: %.1fMB/%.1fMB, Need delete: %.1fMB" %
            (float(size) / 1024 / 1024, float(limit) / 1024 / 1024, float(need_delete) / 1024 / 1024)
        )

        # 如果需要删除的大小小于等于0，则返回False
        if need_delete <= 0:
            return False

        # 更新对等节点数
        self.updatePeerNumbers()

        # 创建反转的站点ID字典
        site_ids_reverse = {val: key for key, val in self.site_ids.items()}
        deleted_file_ids = []

        # 遍历可删除文件的查询结果
        for row in self.queryDeletableFiles():
            # 获取站点地址
            site_address = site_ids_reverse.get(row["site_id"])
            site = self.sites.get(site_address)
            if not site:
                # 如果未找到站点，则记录错误并继续下一次循环
                self.log.error("No site found for id: %s" % row["site_id"])
                continue
            # 记录站点删除文件的日志
            site.log.debug("Deleting %s %.3f MB left" % (row["inner_path"], float(need_delete) / 1024 / 1024))
            deleted_file_ids.append(row["file_id"])
            try:
                # 调用站点内容管理器的optionalRemoved方法和存储的delete方法删除文件，并更新需要删除的大小
                site.content_manager.optionalRemoved(row["inner_path"], row["hash_id"], row["size"])
                site.storage.delete(row["inner_path"])
                need_delete -= row["size"]
            except Exception as err:
                # 如果出现异常，则记录错误日志
                site.log.error("Error deleting %s: %s" % (row["inner_path"], err))

            # 如果需要删除的大小小于等于0，则跳出循环
            if need_delete <= 0:
                break

        # 获取游标
        cur = self.getCursor()
        # 更新已删除文件的is_downloaded和is_pinned字段，并减少对等节点数
        for file_id in deleted_file_ids:
            cur.execute("UPDATE file_optional SET is_downloaded = 0, is_pinned = 0, peer = peer - 1 WHERE ?", {"file_id": file_id})
        cur.close()
# 将 SiteManagerPlugin 类注册到 PluginManager 的 SiteManager 插件中
@PluginManager.registerTo("SiteManager")
class SiteManagerPlugin(object):
    # 加载方法，接受任意参数和关键字参数
    def load(self, *args, **kwargs):
        # 调用父类的 load 方法，并将结果保存在 back 变量中
        back = super(SiteManagerPlugin, self).load(*args, **kwargs)
        # 如果存在站点并且可选文件未加载并且 content_db 连接存在
        if self.sites and not content_db.optional_files_loaded and content_db.conn:
            # 将 content_db.optional_files_loaded 设置为 True
            content_db.optional_files_loaded = True
            # 调用 content_db 的 loadFilesOptional 方法
            content_db.loadFilesOptional()
        # 返回 back 变量
        return back
```
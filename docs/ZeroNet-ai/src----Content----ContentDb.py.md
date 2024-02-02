# `ZeroNet\src\Content\ContentDb.py`

```py
# 导入 os 模块
import os

# 从 Db 模块中导入 Db 类和 DbTableError 异常类
from Db.Db import Db, DbTableError
# 从 Config 模块中导入 config 变量
from Config import config
# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 Debug 模块中导入 Debug 类
from Debug import Debug

# 使用 PluginManager.acceptPlugins 装饰器注册为插件
@PluginManager.acceptPlugins
# 定义 ContentDb 类，继承自 Db 类
class ContentDb(Db):
    # 初始化方法，接受 path 参数
    def __init__(self, path):
        # 调用父类的初始化方法，传入数据库名称和空的表字典
        Db.__init__(self, {"db_name": "ContentDb", "tables": {}}, path)
        # 设置外键约束为 True
        self.foreign_keys = True

    # 初始化数据库方法
    def init(self):
        try:
            # 获取数据库的模式
            self.schema = self.getSchema()
            try:
                # 检查数据库表
                self.checkTables()
            except DbTableError:
                pass
            # 记录调试信息，检查外键约束
            self.log.debug("Checking foreign keys...")
            # 执行 PRAGMA foreign_key_check 命令，获取外键错误信息
            foreign_key_error = self.execute("PRAGMA foreign_key_check").fetchone()
            # 如果存在外键错误，则抛出异常
            if foreign_key_error:
                raise Exception("Database foreign key error: %s" % foreign_key_error)
        except Exception as err:
            # 记录错误日志，重新构建数据库
            self.log.error("Error loading content.db: %s, rebuilding..." % Debug.formatException(err))
            # 关闭数据库连接
            self.close()
            # 删除数据库文件并重试
            os.unlink(self.db_path)
            # 重新初始化数据库
            Db.__init__(self, {"db_name": "ContentDb", "tables": {}}, self.db_path)
            # 设置外键约束为 True
            self.foreign_keys = True
            # 获取数据库的模式
            self.schema = self.getSchema()
            try:
                # 检查数据库表
                self.checkTables()
            except DbTableError:
                pass
        # 初始化站点 ID 字典和站点字典
        self.site_ids = {}
        self.sites = {}
    # 获取数据库的架构信息
    def getSchema(self):
        # 创建一个空的架构字典
        schema = {}
        # 设置数据库名称
        schema["db_name"] = "ContentDb"
        # 设置数据库版本
        schema["version"] = 3
        # 创建空的表字典
        schema["tables"] = {}

        # 如果表的版本不存在
        if not self.getTableVersion("site"):
            # 输出日志信息
            self.log.debug("Migrating from table version-less content.db")
            # 获取数据库版本
            version = int(self.execute("PRAGMA user_version").fetchone()[0])
            # 如果版本大于0
            if version > 0:
                # 检查表
                self.checkTables()
                # 插入数据到keyvalue表
                self.execute("INSERT INTO keyvalue ?", {"json_id": 0, "key": "table.site.version", "value": 1})
                self.execute("INSERT INTO keyvalue ?", {"json_id": 0, "key": "table.content.version", "value": 1})

        # 设置site表的架构信息
        schema["tables"]["site"] = {
            "cols": [
                ["site_id", "INTEGER  PRIMARY KEY ASC NOT NULL UNIQUE"],
                ["address", "TEXT NOT NULL"]
            ],
            "indexes": [
                "CREATE UNIQUE INDEX site_address ON site (address)"
            ],
            "schema_changed": 1
        }

        # 设置content表的架构信息
        schema["tables"]["content"] = {
            "cols": [
                ["content_id", "INTEGER PRIMARY KEY UNIQUE NOT NULL"],
                ["site_id", "INTEGER REFERENCES site (site_id) ON DELETE CASCADE"],
                ["inner_path", "TEXT"],
                ["size", "INTEGER"],
                ["size_files", "INTEGER"],
                ["size_files_optional", "INTEGER"],
                ["modified", "INTEGER"]
            ],
            "indexes": [
                "CREATE UNIQUE INDEX content_key ON content (site_id, inner_path)",
                "CREATE INDEX content_modified ON content (site_id, modified)"
            ],
            "schema_changed": 1
        }

        # 返回架构信息
        return schema

    # 初始化站点信息
    def initSite(self, site):
        # 将站点信息添加到sites字典中
        self.sites[site.address] = site
    # 检查给定站点是否存在于数据库中，如果不存在则插入
    def needSite(self, site):
        if site.address not in self.site_ids:
            self.execute("INSERT OR IGNORE INTO site ?", {"address": site.address})
            self.site_ids = {}
            for row in self.execute("SELECT * FROM site"):
                self.site_ids[row["address"]] = row["site_id"]
        return self.site_ids[site.address]

    # 从数据库中删除指定站点及其相关内容
    def deleteSite(self, site):
        site_id = self.site_ids.get(site.address, 0)
        if site_id:
            self.execute("DELETE FROM site WHERE site_id = :site_id", {"site_id": site_id})
            del self.site_ids[site.address]
            del self.sites[site.address]

    # 向数据库中插入或更新内容信息
    def setContent(self, site, inner_path, content, size=0):
        self.insertOrUpdate("content", {
            "size": size,
            "size_files": sum([val["size"] for key, val in content.get("files", {}).items()]),
            "size_files_optional": sum([val["size"] for key, val in content.get("files_optional", {}).items()]),
            "modified": int(content.get("modified", 0))
        }, {
            "site_id": self.site_ids.get(site.address, 0),
            "inner_path": inner_path
        })

    # 从数据库中删除指定站点的指定内容
    def deleteContent(self, site, inner_path):
        self.execute("DELETE FROM content WHERE ?", {"site_id": self.site_ids.get(site.address, 0), "inner_path": inner_path})

    # 从数据库中加载指定站点的内容信息，并返回字典形式
    def loadDbDict(self, site):
        res = self.execute(
            "SELECT GROUP_CONCAT(inner_path, '|') AS inner_paths FROM content WHERE ?",
            {"site_id": self.site_ids.get(site.address, 0)}
        )
        row = res.fetchone()
        if row and row["inner_paths"]:
            inner_paths = row["inner_paths"].split("|")
            return dict.fromkeys(inner_paths, False)
        else:
            return {}
    # 获取指定站点的总大小，可以选择忽略某些路径
    def getTotalSize(self, site, ignore=None):
        # 根据站点地址获取站点 ID
        params = {"site_id": self.site_ids.get(site.address, 0)}
        # 如果有需要忽略的路径，则添加到参数中
        if ignore:
            params["not__inner_path"] = ignore
        # 执行 SQL 查询，获取总大小和可选文件大小
        res = self.execute("SELECT SUM(size) + SUM(size_files) AS size, SUM(size_files_optional) AS size_optional FROM content WHERE ?", params)
        # 将查询结果转换为字典
        row = dict(res.fetchone())

        # 如果总大小为空，则设置为 0
        if not row["size"]:
            row["size"] = 0
        # 如果可选文件大小为空，则设置为 0
        if not row["size_optional"]:
            row["size_optional"] = 0

        # 返回总大小和可选文件大小
        return row["size"], row["size_optional"]

    # 列出指定站点在指定时间范围内修改过的文件
    def listModified(self, site, after=None, before=None):
        # 根据站点地址获取站点 ID
        params = {"site_id": self.site_ids.get(site.address, 0)}
        # 如果有指定开始时间，则添加到参数中
        if after:
            params["modified>"] = after
        # 如果有指定结束时间，则添加到参数中
        if before:
            params["modified<"] = before
        # 执行 SQL 查询，获取内部路径和修改时间
        res = self.execute("SELECT inner_path, modified FROM content WHERE ?", params)
        # 将查询结果转换为字典，以内部路径为键，修改时间为值
        return {row["inner_path"]: row["modified"] for row in res}
# 创建一个空的内容数据库字典
content_dbs = {}

# 获取内容数据库的函数，如果没有指定路径，则默认为配置文件中的数据目录下的content.db文件
def getContentDb(path=None):
    # 如果没有指定路径，则使用默认路径
    if not path:
        path = "%s/content.db" % config.data_dir
    # 如果路径不在内容数据库字典中，则创建一个新的内容数据库对象，并初始化
    if path not in content_dbs:
        content_dbs[path] = ContentDb(path)
        content_dbs[path].init()
    # 返回内容数据库对象
    return content_dbs[path]

# 预连接到默认的内容数据库
getContentDb()
```
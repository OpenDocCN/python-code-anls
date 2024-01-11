# `ZeroNet\plugins\Chart\ChartDb.py`

```
# 从Config模块中导入config变量
from Config import config
# 从Db模块中导入Db类
from Db.Db import Db
# 导入time模块

# 定义ChartDb类，继承自Db类
class ChartDb(Db):
    # 初始化方法
    def __init__(self):
        # 设置version属性为2
        self.version = 2
        # 调用父类Db的初始化方法，传入getSchema()方法返回的schema和数据文件路径
        super(ChartDb, self).__init__(self.getSchema(), "%s/chart.db" % config.data_dir)
        # 设置foreign_keys属性为True
        self.foreign_keys = True
        # 检查数据库表是否存在，不存在则创建
        self.checkTables()
        # 加载站点数据到sites属性
        self.sites = self.loadSites()
        # 加载类型数据到types属性
        self.types = self.loadTypes()

    # 定义getSchema方法
    def getSchema(self):
        # 创建空的schema字典
        schema = {}
        # 设置数据库名
        schema["db_name"] = "Chart"
        # 创建数据表字典
        schema["tables"] = {}
        # 设置data表的列和索引信息
        schema["tables"]["data"] = {
            "cols": [
                ["data_id", "INTEGER PRIMARY KEY ASC AUTOINCREMENT NOT NULL UNIQUE"],
                ["type_id", "INTEGER NOT NULL"],
                ["site_id", "INTEGER"],
                ["value", "INTEGER"],
                ["date_added", "DATETIME DEFAULT (CURRENT_TIMESTAMP)"]
            ],
            "indexes": [
                "CREATE INDEX site_id ON data (site_id)",
                "CREATE INDEX date_added ON data (date_added)"
            ],
            "schema_changed": 2
        }
        # 设置type表的列信息
        schema["tables"]["type"] = {
            "cols": [
                ["type_id", "INTEGER PRIMARY KEY NOT NULL UNIQUE"],
                ["name", "TEXT"]
            ],
            "schema_changed": 1
        }
        # 设置site表的列信息
        schema["tables"]["site"] = {
            "cols": [
                ["site_id", "INTEGER PRIMARY KEY NOT NULL UNIQUE"],
                ["address", "TEXT"]
            ],
            "schema_changed": 1
        }
        # 返回schema字典
        return schema

    # 定义getTypeId方法，传入name参数
    def getTypeId(self, name):
        # 如果name不在types属性中
        if name not in self.types:
            # 执行插入操作，将name插入到type表中
            res = self.execute("INSERT INTO type ?", {"name": name})
            # 将插入的type_id保存到types属性中
            self.types[name] = res.lastrowid

        # 返回name对应的type_id
        return self.types[name]

    # 定义getSiteId方法，传入address参数
    def getSiteId(self, address):
        # 如果address不在sites属性中
        if address not in self.sites:
            # 执行插入操作，将address插入到site表中
            res = self.execute("INSERT INTO site ?", {"address": address})
            # 将插入的site_id保存到sites属性中
            self.sites[address] = res.lastrowid

        # 返回address对应的site_id
        return self.sites[address]
    # 加载站点信息，返回地址到站点ID的字典
    def loadSites(self):
        sites = {}
        # 从数据库中查询站点信息，并将地址和站点ID存入字典
        for row in self.execute("SELECT * FROM site"):
            sites[row["address"]] = row["site_id"]
        # 返回站点信息字典
        return sites
    
    # 加载类型信息，返回类型名称到类型ID的字典
    def loadTypes(self):
        types = {}
        # 从数据库中查询类型信息，并将类型名称和类型ID存入字典
        for row in self.execute("SELECT * FROM type"):
            types[row["name"]] = row["type_id"]
        # 返回类型信息字典
        return types
    
    # 删除指定地址的站点信息
    def deleteSite(self, address):
        # 如果地址在站点信息字典中
        if address in self.sites:
            # 获取站点ID
            site_id = self.sites[address]
            # 从站点信息字典中删除该地址
            del self.sites[address]
            # 从数据库中删除对应的站点信息
            self.execute("DELETE FROM site WHERE ?", {"site_id": site_id})
            # 从数据库中删除对应站点的数据信息
            self.execute("DELETE FROM data WHERE ?", {"site_id": site_id})
```
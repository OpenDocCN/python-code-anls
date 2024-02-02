# `ZeroNet\plugins\Benchmark\BenchmarkDb.py`

```py
# 导入必要的模块
import os
import json
import contextlib
import time

# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 Config 模块中导入 config 变量
from Config import config

# 将 ActionsPlugin 类注册到 PluginManager 的 Actions 插件中
@PluginManager.registerTo("Actions")
class ActionsPlugin:
    # 获取基准测试
    def getBenchmarkTests(self, online=False):
        # 调用父类的 getBenchmarkTests 方法获取基准测试
        tests = super().getBenchmarkTests(online)
        # 扩展基准测试列表
        tests.extend([
            {"func": self.testDbConnect, "num": 10, "time_standard": 0.27},
            {"func": self.testDbInsert, "num": 10, "time_standard": 0.91},
            {"func": self.testDbInsertMultiuser, "num": 1, "time_standard": 0.57},
            {"func": self.testDbQueryIndexed, "num": 1000, "time_standard": 0.84},
            {"func": self.testDbQueryNotIndexed, "num": 1000, "time_standard": 1.30}
        ])
        return tests

    # 上下文管理器，用于获取测试数据库
    @contextlib.contextmanager
    def getTestDb(self):
        # 从 Db 模块中导入 Db 类
        from Db import Db
        # 构建数据库文件路径
        path = "%s/benchmark.db" % config.data_dir
        # 如果文件路径存在，则删除文件
        if os.path.isfile(path):
            os.unlink(path)
        # 定义数据库 schema
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
        # 创建数据库对象
        db = Db.Db(schema, path)

        yield db  # 返回数据库对象

        db.close()  # 关闭数据库连接
        if os.path.isfile(path):  # 如果文件路径存在，则删除文件
            os.unlink(path)
    # 测试数据库连接，可以指定运行次数，默认为1次
    def testDbConnect(self, num_run=1):
        # 导入sqlite3模块
        import sqlite3
        # 循环指定次数
        for i in range(num_run):
            # 获取测试数据库连接
            with self.getTestDb() as db:
                # 检查数据库表
                db.checkTables()
            # 生成一个"."
            yield "."
        # 返回SQLite版本和API版本信息
        yield "(SQLite version: %s, API: %s)" % (sqlite3.sqlite_version, sqlite3.version)

    # 测试数据库插入操作，可以指定运行次数，默认为1次
    def testDbInsert(self, num_run=1):
        # 生成"x 1000 lines "
        yield "x 1000 lines "
        # 循环指定次数
        for u in range(num_run):
            # 获取测试数据库连接
            with self.getTestDb() as db:
                # 检查数据库表
                db.checkTables()
                # 初始化数据字典
                data = {"test": []}
                # 循环1000次，生成数据并添加到数据字典中
                for i in range(1000):  # 1000 line of data
                    data["test"].append({"test_id": i, "title": "Testdata for %s message %s" % (u, i)})
                # 将数据字典以JSON格式写入文件
                json.dump(data, open("%s/test_%s.json" % (config.data_dir, u), "w"))
                # 更新数据库中的JSON数据
                db.updateJson("%s/test_%s.json" % (config.data_dir, u))
                # 删除生成的JSON文件
                os.unlink("%s/test_%s.json" % (config.data_dir, u))
                # 断言数据库中的数据行数为1000
                assert db.execute("SELECT COUNT(*) FROM test").fetchone()[0] == 1000
            # 生成一个"."
            yield "."

    # 向测试数据库填充数据
    def fillTestDb(self, db):
        # 检查数据库表
        db.checkTables()
        # 获取数据库游标
        cur = db.getCursor()
        # 关闭日志记录
        cur.logging = False
        # 循环100次，生成数据并添加到数据字典中
        for u in range(100, 200):  # 100 user
            data = {"test": []}
            for i in range(100):  # 1000 line of data
                data["test"].append({"test_id": i, "title": "Testdata for %s message %s" % (u, i)})
            # 将数据字典以JSON格式写入文件
            json.dump(data, open("%s/test_%s.json" % (config.data_dir, u), "w"))
            # 更新数据库中的JSON数据
            db.updateJson("%s/test_%s.json" % (config.data_dir, u), cur=cur)
            # 删除生成的JSON文件
            os.unlink("%s/test_%s.json" % (config.data_dir, u))
            # 如果u能被10整除，则生成一个"."
            if u % 10 == 0:
                yield "."
    # 测试数据库插入多个用户
    def testDbInsertMultiuser(self, num_run=1):
        # 生成测试描述
        yield "x 100 users x 100 lines "
        # 循环执行 num_run 次
        for u in range(num_run):
            # 获取测试数据库连接
            with self.getTestDb() as db:
                # 填充测试数据库
                for progress in self.fillTestDb(db):
                    yield progress
                # 查询测试表中的行数
                num_rows = db.execute("SELECT COUNT(*) FROM test").fetchone()[0]
                # 断言测试表中的行数为 10000
                assert num_rows == 10000, "%s != 10000" % num_rows

    # 测试数据库查询索引
    def testDbQueryIndexed(self, num_run=1):
        # 记录开始时间
        s = time.time()
        # 获取测试数据库连接
        with self.getTestDb() as db:
            # 填充测试数据库
            for progress in self.fillTestDb(db):
                pass
            # 生成数据库预热完成的描述
            yield " (Db warmup done in %.3fs) " % (time.time() - s)
            # 初始化总共找到的行数
            found_total = 0
            # 循环执行 num_run 次
            for i in range(num_run):  # 1000x by test_id
                # 初始化找到的行数
                found = 0
                # 执行带有索引的查询
                res = db.execute("SELECT * FROM test WHERE test_id = %s" % (i % 100))
                # 遍历查询结果
                for row in res:
                    found_total += 1
                    found += 1
                # 释放查询结果
                del(res)
                yield "."
                # 断言找到的行数为 100
                assert found == 100, "%s != 100 (i: %s)" % (found, i)
            # 生成找到的行数描述
            yield "Found: %s" % found_total

    # 测试数据库查询非索引
    def testDbQueryNotIndexed(self, num_run=1):
        # 记录开始时间
        s = time.time()
        # 获取测试数据库连接
        with self.getTestDb() as db:
            # 填充测试数据库
            for progress in self.fillTestDb(db):
                pass
            # 生成数据库预热完成的描述
            yield " (Db warmup done in %.3fs) " % (time.time() - s)
            # 初始化总共找到的行数
            found_total = 0
            # 循环执行 num_run 次
            for i in range(num_run):  # 1000x by test_id
                # 初始化找到的行数
                found = 0
                # 执行不带索引的查询
                res = db.execute("SELECT * FROM test WHERE json_id = %s" % i)
                # 遍历查询结果
                for row in res:
                    found_total += 1
                    found += 1
                yield "."
                # 释放查询结果
                del(res)
                # 根据条件断言找到的行数
                if i == 0 or i > 100:
                    assert found == 0, "%s != 0 (i: %s)" % (found, i)
                else:
                    assert found == 100, "%s != 100 (i: %s)" % (found, i)
            # 生成找到的行数描述
            yield "Found: %s" % found_total
```
# `ZeroNet\src\Test\TestDb.py`

```
import io  # 导入io模块


class TestDb:  # 定义TestDb类
    def testCheckTables(self, db):  # 定义testCheckTables方法，接受db参数
        tables = [row["name"] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]  # 查询数据库中的表名
        assert "keyvalue" in tables  # 断言keyvalue表存在，用于存储简单的键值对
        assert "json" in tables  # 断言json表存在，用于存储Json文件路径
        assert "test" in tables  # 断言test表存在，根据dbschema.json中的定义

        # 验证test表
        cols = [col["name"] for col in db.execute("PRAGMA table_info(test)")]  # 查询test表的列名
        assert "test_id" in cols  # 断言test_id列存在
        assert "title" in cols  # 断言title列存在

        # 添加新表
        assert "newtest" not in tables  # 断言newtest表不存在
        db.schema["tables"]["newtest"] = {  # 在db的schema中添加newtest表的定义
            "cols": [
                ["newtest_id", "INTEGER"],
                ["newtitle", "TEXT"],
            ],
            "indexes": ["CREATE UNIQUE INDEX newtest_id ON newtest(newtest_id)"],  # 定义索引
            "schema_changed": 1426195822  # 记录模式变更时间戳
        }
        db.checkTables()  # 检查表的变更
        tables = [row["name"] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]  # 查询数据库中的表名
        assert "test" in tables  # 断言test表存在
        assert "newtest" in tables  # 断言newtest表存在

    def testEscaping(self, db):  # 定义testEscaping方法，接受db参数
        # 测试插入
        for i in range(100):  # 循环100次
            db.execute("INSERT INTO test ?", {"test_id": i, "title": "Test '\" #%s" % i})  # 向test表插入数据

        assert db.execute(  # 断言查询结果
            "SELECT COUNT(*) AS num FROM test WHERE ?",  # 查询test表中符合条件的记录数
            {"title": "Test '\" #1"}  # 查询条件
        ).fetchone()["num"] == 1  # 断言查询结果为1

        assert db.execute(  # 断言查询结果
            "SELECT COUNT(*) AS num FROM test WHERE ?",  # 查询test表中符合条件的记录数
            {"title": ["Test '\" #%s" % i for i in range(0, 50)]}  # 查询条件
        ).fetchone()["num"] == 50  # 断言查询结果为50

        assert db.execute(  # 断言查询结果
            "SELECT COUNT(*) AS num FROM test WHERE ?",  # 查询test表中符合条件的记录数
            {"not__title": ["Test '\" #%s" % i for i in range(50, 3000)]}  # 查询条件
        ).fetchone()["num"] == 50  # 断言查询结果为50
    # 测试更新 JSON 数据库
    def testUpdateJson(self, db):
        # 创建一个字节流对象
        f = io.BytesIO()
        # 向字节流中写入 JSON 数据
        f.write("""
            {
                "test": [
                    {"test_id": 1, "title": "Test 1 title", "extra col": "Ignore it"}
                ]
            }
        """.encode())
        # 将指针移动到字节流的开头
        f.seek(0)
        # 断言更新 JSON 数据库成功
        assert db.updateJson(db.db_dir + "data.json", f) is True
        # 断言执行 SQL 查询返回的结果为 1
        assert db.execute("SELECT COUNT(*) AS num FROM test_importfilter").fetchone()["num"] == 1
        # 断言执行 SQL 查询返回的结果为 1
        assert db.execute("SELECT COUNT(*) AS num FROM test").fetchone()["num"] == 1

    # 测试不安全的模式
    def testUnsafePattern(self, db):
        # 更新数据库模式中的映射
        db.schema["maps"] = {"[A-Za-z.]*": db.schema["maps"]["data.json"]}  # Only repetition of . supported
        # 创建一个字符串流对象
        f = io.StringIO()
        # 向字符串流中写入 JSON 数据
        f.write("""
            {
                "test": [
                    {"test_id": 1, "title": "Test 1 title", "extra col": "Ignore it"}
                ]
            }
        """)
        # 将指针移动到字符串流的开头
        f.seek(0)
        # 断言更新 JSON 数据库失败
        assert db.updateJson(db.db_dir + "data.json", f) is False
        # 断言执行 SQL 查询返回的结果为 0
        assert db.execute("SELECT COUNT(*) AS num FROM test_importfilter").fetchone()["num"] == 0
        # 断言执行 SQL 查询返回的结果为 0
        assert db.execute("SELECT COUNT(*) AS num FROM test").fetchone()["num"] == 0
```
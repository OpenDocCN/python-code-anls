# `ZeroNet\src\Db\DbCursor.py`

```py
# 导入时间模块
import time
# 导入正则表达式模块
import re
# 从util模块中导入helper函数
from util import helper

# 定义一个特殊的sqlite游标类
class DbCursor:

    # 初始化方法，接受一个数据库对象作为参数
    def __init__(self, db):
        # 将传入的数据库对象赋值给实例变量db
        self.db = db
        # 初始化日志记录标志为False
        self.logging = False

    # 定义quoteValue方法，用于对数值进行转义处理
    def quoteValue(self, value):
        # 如果数值类型为整数，则直接转换为字符串返回
        if type(value) is int:
            return str(value)
        # 如果不是整数，则将值转换为字符串，并对单引号进行转义处理
        else:
            return "'%s'" % value.replace("'", "''")
    # 执行数据库查询，可以带参数
    def execute(self, query, params=None):
        # 去除查询语句两端的空格
        query = query.strip()
        # 当数据库正在执行提交或者正在处理时，等待0.1秒
        while self.db.progress_sleeping or self.db.commiting:
            time.sleep(0.1)

        # 记录最后一次查询时间
        self.db.last_query_time = time.time()

        # 解析查询语句和参数
        query, params = self.parseQuery(query, params)

        # 获取数据库连接的游标
        cursor = self.db.getConn().cursor()
        # 将游标添加到游标集合中
        self.db.cursors.add(cursor)
        # 如果数据库被锁定，则记录锁定时间
        if self.db.lock.locked():
            self.db.log.debug("Locked for %.3fs" % (time.time() - self.db.lock.time_lock))

        try:
            # 记录开始执行查询的时间
            s = time.time()
            # 获取数据库锁
            self.db.lock.acquire(True)
            # 如果查询语句是VACUUM，则提交事务
            if query.upper().strip("; ") == "VACUUM":
                self.db.commit("vacuum called")
            # 执行查询语句，如果有参数则传入参数
            if params:
                res = cursor.execute(query, params)
            else:
                res = cursor.execute(query)
        finally:
            # 释放数据库锁
            self.db.lock.release()

        # 计算查询所花费的时间
        taken_query = time.time() - s
        # 如果开启了日志记录或者查询时间超过1秒，则记录查询信息
        if self.logging or taken_query > 1:
            if params:  # 查询带有参数
                self.db.log.debug("Query: " + query + " " + str(params) + " (Done in %.4f)" % (time.time() - s))
            else:
                self.db.log.debug("Query: " + query + " (Done in %.4f)" % (time.time() - s))

        # 记录查询统计信息
        if self.db.collect_stats:
            if query not in self.db.query_stats:
                self.db.query_stats[query] = {"call": 0, "time": 0.0}
            self.db.query_stats[query]["call"] += 1
            self.db.query_stats[query]["time"] += time.time() - s

        # 获取查询语句的类型
        query_type = query.split(" ", 1)[0].upper()
        # 判断查询语句是否是更新类型的
        is_update_query = query_type in ["UPDATE", "DELETE", "INSERT", "CREATE"]
        # 如果数据库不需要提交事务且查询语句是更新类型的，则设置需要提交事务的标志
        if not self.db.need_commit and is_update_query:
            self.db.need_commit = True

        # 如果查询语句是更新类型的，则返回游标，否则返回查询结果
        if is_update_query:
            return cursor
        else:
            return res
    # 执行多个参数化查询，确保在数据库没有进程正在提交或者在提交中
    def executemany(self, query, params):
        while self.db.progress_sleeping or self.db.commiting:
            time.sleep(0.1)

        # 记录最后一次查询的时间
        self.db.last_query_time = time.time()

        # 记录开始执行的时间
        s = time.time()
        # 获取数据库连接的游标
        cursor = self.db.getConn().cursor()
        self.db.cursors.add(cursor)

        try:
            # 获取锁
            self.db.lock.acquire(True)
            # 执行多个参数化查询
            cursor.executemany(query, params)
        finally:
            # 释放锁
            self.db.lock.release()

        # 计算查询所花费的时间
        taken_query = time.time() - s
        # 如果开启了日志记录或者查询时间超过0.1秒，则记录日志
        if self.logging or taken_query > 0.1:
            self.db.log.debug("Execute many: %s (Done in %.4f)" % (query, taken_query))

        # 设置需要提交的标志
        self.db.need_commit = True

        return cursor

    # 创建或更新数据库行，但不增加行id
    def insertOrUpdate(self, table, query_sets, query_wheres, oninsert={}):
        # 构建设置字段的SQL语句
        sql_sets = ["%s = :%s" % (key, key) for key in query_sets.keys()]
        # 构建条件字段的SQL语句
        sql_wheres = ["%s = :%s" % (key, key) for key in query_wheres.keys()]

        # 合并设置字段和条件字段的参数
        params = query_sets
        params.update(query_wheres)
        # 执行更新操作
        res = self.execute(
            "UPDATE %s SET %s WHERE %s" % (table, ", ".join(sql_sets), " AND ".join(sql_wheres)),
            params
        )
        # 如果更新的行数为0，则执行插入操作
        if res.rowcount == 0:
            params.update(oninsert)  # 添加仅插入的字段
            self.execute("INSERT INTO %s ?" % table, params)

    # 创建新表
    # 返回：成功返回True
    def createTable(self, table, cols):
        # TODO: 检查当前结构
        # 删除已存在的表
        self.execute("DROP TABLE IF EXISTS %s" % table)
        col_definitions = []
        # 构建列定义
        for col_name, col_type in cols:
            col_definitions.append("%s %s" % (col_name, col_type))

        # 创建表
        self.execute("CREATE TABLE %s (%s)" % (table, ",".join(col_definitions)))
        return True

    # 在表上创建索引
    # 返回：成功返回True
    # 创建表的索引
    def createIndexes(self, table, indexes):
        # 遍历索引列表
        for index in indexes:
            # 如果索引不是以"CREATE"开头，则记录错误信息并继续下一个索引
            if not index.strip().upper().startswith("CREATE"):
                self.db.log.error("Index command should start with CREATE: %s" % index)
                continue
            # 执行索引命令
            self.execute(index)

    # 如果表不存在则创建表
    # 返回：如果更新了表则返回True
    def needTable(self, table, cols, indexes=None, version=1):
        # 获取当前表的版本号
        current_version = self.db.getTableVersion(table)
        # 如果当前版本小于指定版本，则需要更新表或者表不存在
        if int(current_version) < int(version):  
            # 记录表过时的信息，并重建表
            self.db.log.debug("Table %s outdated...version: %s need: %s, rebuilding..." % (table, current_version, version))
            self.createTable(table, cols)
            # 如果有索引，则创建索引
            if indexes:
                self.createIndexes(table, indexes)
            # 更新表的版本号
            self.execute(
                "INSERT OR REPLACE INTO keyvalue ?",
                {"json_id": 0, "key": "table.%s.version" % table, "value": version}
            )
            return True
        else:  # 表未改变
            return False

    # 获取或创建JSON文件的行
    # 返回：数据库行
    # 从给定文件路径中提取目录和文件名
    directory, file_name = re.match("^(.*?)/*([^/]*)$", file_path).groups()
    # 根据数据库模式版本进行不同的操作
    if self.db.schema["version"] == 1:
        # 一个路径字段
        # 执行 SQL 查询，根据文件路径查询数据库中的记录
        res = self.execute("SELECT * FROM json WHERE ? LIMIT 1", {"path": file_path})
        # 获取查询结果的第一行记录
        row = res.fetchone()
        # 如果没有查询到记录，创建新记录
        if not row:
            # 执行 SQL 插入语句，插入新的记录
            self.execute("INSERT INTO json ?", {"path": file_path})
            # 重新查询数据库，获取新插入的记录
            res = self.execute("SELECT * FROM json WHERE ? LIMIT 1", {"path": file_path})
            row = res.fetchone()
    elif self.db.schema["version"] == 2:
        # 分开目录和文件名（更容易进行连接）
        res = self.execute("SELECT * FROM json WHERE ? LIMIT 1", {"directory": directory, "file_name": file_name})
        row = res.fetchone()
        if not row:  # No row yet, create it
            self.execute("INSERT INTO json ?", {"directory": directory, "file_name": file_name})
            res = self.execute("SELECT * FROM json WHERE ? LIMIT 1", {"directory": directory, "file_name": file_name})
            row = res.fetchone()
    elif self.db.schema["version"] == 3:
        # 分开站点、目录和文件名（用于合并站点）
        site_address, directory = re.match("^([^/]*)/(.*)$", directory).groups()
        res = self.execute("SELECT * FROM json WHERE ? LIMIT 1", {"site": site_address, "directory": directory, "file_name": file_name})
        row = res.fetchone()
        if not row:  # No row yet, create it
            self.execute("INSERT INTO json ?", {"site": site_address, "directory": directory, "file_name": file_name})
            res = self.execute("SELECT * FROM json WHERE ? LIMIT 1", {"site": site_address, "directory": directory, "file_name": file_name})
            row = res.fetchone()
    else:
        # 抛出异常，不支持的数据库模式版本
        raise Exception("Dbschema version %s not supported" % self.db.schema.get("version"))
    # 返回查询到的记录
    return row
    # 关闭当前对象的方法，但是没有实际操作
    def close(self):
        pass
```
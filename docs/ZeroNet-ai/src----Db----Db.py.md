# `ZeroNet\src\Db\Db.py`

```
# 导入所需的模块
import sqlite3
import json
import time
import logging
import re
import os
import atexit
import threading
import sys
import weakref
import errno

# 导入第三方模块
import gevent

# 导入自定义模块
from Debug import Debug
from .DbCursor import DbCursor
from util import SafeRe
from util import helper
from util import ThreadPool
from Config import config

# 创建一个线程池对象，用于数据库操作
thread_pool_db = ThreadPool.ThreadPool(config.threads_db)

# 初始化数据库ID和已打开的数据库列表
next_db_id = 0
opened_dbs = []

# 定义一个函数，用于关闭空闲的数据库以节省内存
def dbCleanup():
    while 1:
        time.sleep(60 * 5)
        for db in opened_dbs[:]:
            idle = time.time() - db.last_query_time
            if idle > 60 * 5 and db.close_idle:
                db.close("Cleanup")

# 定义一个函数，用于检查数据库是否需要提交事务
def dbCommitCheck():
    while 1:
        time.sleep(5)
        for db in opened_dbs[:]:
            if not db.need_commit:
                continue

            success = db.commit("Interval")
            if success:
                db.need_commit = False
            time.sleep(0.1)

# 定义一个函数，用于关闭所有已打开的数据库
def dbCloseAll():
    for db in opened_dbs[:]:
        db.close("Close all")

# 使用协程启动数据库清理和提交检查函数
gevent.spawn(dbCleanup)
gevent.spawn(dbCommitCheck)

# 在程序退出时，注册关闭所有数据库的函数
atexit.register(dbCloseAll)

# 定义一个自定义异常类，用于数据库表操作出错时抛出异常
class DbTableError(Exception):
    def __init__(self, message, table):
        super().__init__(message)
        self.table = table

# 定义一个数据库类
class Db(object):
    # 初始化数据库连接对象
    def __init__(self, schema, db_path, close_idle=False):
        # 声明全局变量，用于生成数据库连接对象的唯一标识
        global next_db_id
        # 设置数据库文件路径
        self.db_path = db_path
        # 获取数据库文件所在目录
        self.db_dir = os.path.dirname(db_path) + "/"
        # 设置数据库模式
        self.schema = schema
        # 如果模式中没有版本信息，则默认设置为1
        self.schema["version"] = self.schema.get("version", 1)
        # 初始化数据库连接和游标
        self.conn = None
        self.cur = None
        # 使用弱引用集合来保存游标对象
        self.cursors = weakref.WeakSet()
        # 生成数据库连接对象的唯一标识
        self.id = next_db_id
        next_db_id += 1
        # 初始化一些状态标识
        self.progress_sleeping = False
        self.commiting = False
        # 设置日志对象
        self.log = logging.getLogger("Db#%s:%s" % (self.id, schema["db_name"]))
        # 初始化一些属性
        self.table_names = None
        self.collect_stats = False
        self.foreign_keys = False
        self.need_commit = False
        self.query_stats = {}
        self.db_keyvalues = {}
        # 初始化延迟队列和线程
        self.delayed_queue = []
        self.delayed_queue_thread = None
        # 设置是否在空闲时关闭连接
        self.close_idle = close_idle
        # 设置最后一次查询时间和睡眠时间
        self.last_query_time = time.time()
        self.last_sleep_time = time.time()
        # 记录自上次睡眠以来执行的查询次数
        self.num_execute_since_sleep = 0
        # 初始化线程锁
        self.lock = ThreadPool.Lock()
        self.connect_lock = ThreadPool.Lock()

    # 返回数据库连接对象的字符串表示
    def __repr__(self):
        return "<Db#%s:%s close_idle:%s>" % (id(self), self.db_path, self.close_idle)
    # 连接数据库
    def connect(self):
        # 获取连接锁
        self.connect_lock.acquire(True)
        try:
            # 如果已经有连接，则记录日志并返回
            if self.conn:
                self.log.debug("Already connected, connection ignored")
                return

            # 如果当前对象不在已打开的数据库列表中，则添加到列表中
            if self not in opened_dbs:
                opened_dbs.append(self)
            # 记录当前时间
            s = time.time()
            try:  # 目录尚不存在
                # 创建数据库目录
                os.makedirs(self.db_dir)
                self.log.debug("Created Db path: %s" % self.db_dir)
            except OSError as err:
                # 如果错误不是目录已存在，则抛出异常
                if err.errno != errno.EEXIST:
                    raise err
            # 如果数据库文件不存在，则记录日志
            if not os.path.isfile(self.db_path):
                self.log.debug("Db file not exist yet: %s" % self.db_path)
            # 连接数据库
            self.conn = sqlite3.connect(self.db_path, isolation_level="DEFERRED", check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.set_progress_handler(self.progress, 5000000)
            self.conn.execute('PRAGMA journal_mode=WAL')
            # 如果启用外键约束，则设置 PRAGMA
            if self.foreign_keys:
                self.conn.execute("PRAGMA foreign_keys = ON")
            # 获取数据库游标
            self.cur = self.getCursor()

            # 记录连接成功的日志
            self.log.debug(
                "Connected to %s in %.3fs (opened: %s, sqlite version: %s)..." %
                (self.db_path, time.time() - s, len(opened_dbs), sqlite3.version)
            )
            self.log.debug("Connect by thread: %s" % threading.current_thread().ident)
            self.log.debug("Connect called by %s" % Debug.formatStack())
        finally:
            # 释放连接锁
            self.connect_lock.release()

    # 获取数据库连接
    def getConn(self):
        # 如果没有连接，则调用 connect() 方法进行连接
        if not self.conn:
            self.connect()
        return self.conn

    # 进度回调函数
    def progress(self, *args, **kwargs):
        # 设置进度睡眠标志为 True
        self.progress_sleeping = True
        # 睡眠一段时间
        time.sleep(0.001)
        # 设置进度睡眠标志为 False
        self.progress_sleeping = False

    # 使用数据库游标执行查询
    def execute(self, query, params=None):
        # 如果没有连接，则调用 connect() 方法进行连接
        if not self.conn:
            self.connect()
        # 使用游标执行查询
        return self.cur.execute(query, params)

    # 使用线程池装饰器包装
    @thread_pool_db.wrap
    # 提交数据库事务，如果进度正在休眠，则忽略提交并记录日志
    def commit(self, reason="Unknown"):
        if self.progress_sleeping:
            self.log.debug("Commit ignored: Progress sleeping")
            return False

        # 如果没有数据库连接，则忽略提交并记录日志
        if not self.conn:
            self.log.debug("Commit ignored: No connection")
            return False

        # 如果已经在提交，则忽略提交并记录日志
        if self.commiting:
            self.log.debug("Commit ignored: Already commiting")
            return False

        try:
            s = time.time()
            self.commiting = True
            # 提交数据库事务
            self.conn.commit()
            self.log.debug("Commited in %.3fs (reason: %s)" % (time.time() - s, reason))
            return True
        except Exception as err:
            # 如果出现异常，根据异常类型记录不同级别的日志
            if "SQL statements in progress" in str(err):
                self.log.warning("Commit delayed: %s (reason: %s)" % (Debug.formatException(err), reason))
            else:
                self.log.error("Commit error: %s (reason: %s)" % (Debug.formatException(err), reason))
            return False
        finally:
            self.commiting = False

    # 插入或更新数据，如果没有数据库连接，则先建立连接
    def insertOrUpdate(self, *args, **kwargs):
        if not self.conn:
            self.connect()
        return self.cur.insertOrUpdate(*args, **kwargs)

    # 延迟执行数据库操作，将操作添加到延迟队列中
    def executeDelayed(self, *args, **kwargs):
        if not self.delayed_queue_thread:
            self.delayed_queue_thread = gevent.spawn_later(1, self.processDelayed)
        self.delayed_queue.append(("execute", (args, kwargs)))

    # 延迟执行插入或更新操作，将操作添加到延迟队列中
    def insertOrUpdateDelayed(self, *args, **kwargs):
        if not self.delayed_queue:
            gevent.spawn_later(1, self.processDelayed)
        self.delayed_queue.append(("insertOrUpdate", (args, kwargs)))
    # 处理延迟队列中的操作
    def processDelayed(self):
        # 如果延迟队列为空，则记录日志并返回
        if not self.delayed_queue:
            self.log.debug("processDelayed aborted")
            return
        # 如果数据库连接不存在，则建立连接
        if not self.conn:
            self.connect()

        # 记录当前时间
        s = time.time()
        # 获取数据库游标
        cur = self.getCursor()
        # 遍历延迟队列中的操作
        for command, params in self.delayed_queue:
            # 如果操作是插入或更新
            if command == "insertOrUpdate":
                cur.insertOrUpdate(*params[0], **params[1])
            else:
                cur.execute(*params[0], **params[1])

        # 如果延迟队列长度大于10，则记录日志
        if len(self.delayed_queue) > 10:
            self.log.debug("Processed %s delayed queue in %.3fs" % (len(self.delayed_queue), time.time() - s))
        # 清空延迟队列
        self.delayed_queue = []
        self.delayed_queue_thread = None

    # 关闭数据库连接
    def close(self, reason="Unknown"):
        # 如果数据库连接不存在，则返回False
        if not self.conn:
            return False
        # 获取连接锁
        self.connect_lock.acquire()
        # 记录当前时间
        s = time.time()
        # 如果存在延迟队列，则处理延迟队列
        if self.delayed_queue:
            self.processDelayed()
        # 从已打开的数据库列表中移除当前数据库
        if self in opened_dbs:
            opened_dbs.remove(self)
        # 设置不需要提交
        self.need_commit = False
        # 提交事务
        self.commit("Closing: %s" % reason)
        # 记录关闭操作的调用栈
        self.log.debug("Close called by %s" % Debug.formatStack())
        # 循环5次，每次间隔0.1秒，检查是否存在未关闭的游标
        for i in range(5):
            if len(self.cursors) == 0:
                break
            self.log.debug("Pending cursors: %s" % len(self.cursors))
            time.sleep(0.1 * i)
        # 如果仍然存在未关闭的游标，则记录日志并中断连接
        if len(self.cursors):
            self.log.debug("Killing cursors: %s" % len(self.cursors))
            self.conn.interrupt()

        # 关闭游标和连接
        if self.cur:
            self.cur.close()
        if self.conn:
            ThreadPool.main_loop.call(self.conn.close)
        self.conn = None
        self.cur = None
        # 记录关闭操作的日志
        self.log.debug("%s closed (reason: %s) in %.3fs, opened: %s" % (self.db_path, reason, time.time() - s, len(opened_dbs)))
        # 释放连接锁
        self.connect_lock.release()
        # 返回True
        return True

    # 获取数据库游标对象
    # 返回：游标对象
    # 获取数据库游标对象
    def getCursor(self):
        # 如果数据库连接不存在，则进行连接
        if not self.conn:
            self.connect()

        # 创建并返回数据库游标对象
        cur = DbCursor(self)
        return cur

    # 获取共享的数据库游标对象
    def getSharedCursor(self):
        # 如果数据库连接不存在，则进行连接
        if not self.conn:
            self.connect()
        # 返回当前数据库游标对象
        return self.cur

    # 获取表的版本号
    # 返回：表的版本号，如果不存在则返回 None
    def getTableVersion(self, table_name):
        # 如果数据库键值对不存在
        if not self.db_keyvalues:  # Get db keyvalues
            try:
                # 执行 SQL 查询获取 keyvalue 表中 json_id=0 的记录
                res = self.execute("SELECT * FROM keyvalue WHERE json_id=0")  # json_id = 0 is internal keyvalues
            except sqlite3.OperationalError as err:  # 表不存在
                # 记录日志并返回 False
                self.log.debug("Query table version error: %s" % err)
                return False

            # 遍历查询结果，将键值对存储到 db_keyvalues 字典中
            for row in res:
                self.db_keyvalues[row["key"]] = row["value"]

        # 返回指定表的版本号，如果不存在则返回 0
        return self.db_keyvalues.get("table.%s.version" % table_name, 0)

    # 检查数据库表
    # 返回：<list> 变更的表名
    # 更新 JSON 文件到数据库
    # 返回：如果匹配则返回 True
# 如果当前模块是主程序，则执行以下代码
if __name__ == "__main__":
    # 获取当前时间
    s = time.time()
    # 创建一个控制台日志处理器
    console_log = logging.StreamHandler()
    # 设置根记录器的日志级别为 DEBUG
    logging.getLogger('').setLevel(logging.DEBUG)
    # 将控制台日志处理器添加到根记录器
    logging.getLogger('').addHandler(console_log)
    # 设置控制台日志处理器的日志级别为 DEBUG
    console_log.setLevel(logging.DEBUG)
    # 从 JSON 文件加载数据库模式，并连接到指定的 SQLite 数据库文件
    dbjson = Db(json.load(open("zerotalk.schema.json")), "data/users/zerotalk.db")
    # 开启收集数据库操作统计信息的功能
    dbjson.collect_stats = True
    # 检查数据库表是否存在，不存在则创建
    dbjson.checkTables()
    # 获取数据库游标
    cur = dbjson.getCursor()
    # 关闭日志记录
    cur.logging = False
    # 更新指定 JSON 文件到数据库
    dbjson.updateJson("data/users/content.json", cur=cur)
    # 遍历用户目录下的所有子目录
    for user_dir in os.listdir("data/users"):
        # 如果是目录
        if os.path.isdir("data/users/%s" % user_dir):
            # 更新指定 JSON 文件到数据库
            dbjson.updateJson("data/users/%s/data.json" % user_dir, cur=cur)
            # 打印"."，表示处理进度
            # print ".",
    # 开启日志记录
    cur.logging = True
    # 打印处理耗时
    print("Done in %.3fs" % (time.time() - s))
    # 遍历并打印数据库查询统计信息
    for query, stats in sorted(dbjson.query_stats.items()):
        print("-", query, stats)
```
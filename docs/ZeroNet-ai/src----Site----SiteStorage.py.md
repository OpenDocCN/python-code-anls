# `ZeroNet\src\Site\SiteStorage.py`

```py
# 导入所需的模块
import os  # 操作系统相关的功能
import re  # 正则表达式
import shutil  # 文件操作
import json  # JSON 数据格式
import time  # 时间相关的功能
import errno  # 错误码
from collections import defaultdict  # 默认字典

import sqlite3  # SQLite 数据库
import gevent.event  # 协程事件

import util  # 自定义工具模块
from util import SafeRe  # 安全的正则表达式
from Db.Db import Db  # 数据库类
from Debug import Debug  # 调试模块
from Config import config  # 配置
from util import helper  # 辅助函数
from util import ThreadPool  # 线程池
from Plugin import PluginManager  # 插件管理
from Translate import translate as _  # 翻译

# 创建线程池对象，用于文件系统读取
thread_pool_fs_read = ThreadPool.ThreadPool(config.threads_fs_read, name="FS read")
# 创建线程池对象，用于文件系统写入
thread_pool_fs_write = ThreadPool.ThreadPool(config.threads_fs_write, name="FS write")
# 创建线程池对象，用于文件系统批处理
thread_pool_fs_batch = ThreadPool.ThreadPool(1, name="FS batch")

# SiteStorage 类，用于管理站点数据
@PluginManager.acceptPlugins
class SiteStorage(object):
    # 初始化方法
    def __init__(self, site, allow_create=True):
        self.site = site  # 站点对象
        self.directory = "%s/%s" % (config.data_dir, self.site.address)  # 站点数据目录
        self.allowed_dir = os.path.abspath(self.directory)  # 只允许在该目录下提供文件服务
        self.log = site.log  # 日志对象
        self.db = None  # 数据库对象
        self.db_checked = False  # 是否已检查数据库表
        self.event_db_busy = None  # 数据库重建时的协程事件
        self.has_db = self.isFile("dbschema.json")  # 是否存在数据库模式文件

        if not os.path.isdir(self.directory):
            if allow_create:
                os.mkdir(self.directory)  # 如果目录不存在且允许创建，则创建目录
            else:
                raise Exception("Directory not exists: %s" % self.directory)  # 否则抛出异常

    # 获取数据库文件路径
    def getDbFile(self):
        if self.db:
            return self.db.schema["db_file"]  # 如果已有数据库对象，则返回数据库文件路径
        else:
            if self.isFile("dbschema.json"):
                schema = self.loadJson("dbschema.json")  # 加载数据库模式文件
                return schema["db_file"]  # 返回数据库文件路径
            else:
                return False  # 否则返回 False

    # 打开数据库
    def openDb(self, close_idle=False):
        schema = self.getDbSchema()  # 获取数据库模式
        db_path = self.getPath(schema["db_file"])  # 获取数据库文件路径
        return Db(schema, db_path, close_idle=close_idle)  # 返回数据库对象
    # 关闭数据库连接，如果有原因则记录原因
    def closeDb(self, reason="Unknown (SiteStorage)"):
        # 如果存在数据库连接，则关闭数据库连接
        if self.db:
            self.db.close(reason)
        # 重置事件数据库繁忙状态
        self.event_db_busy = None
        # 重置数据库连接
        self.db = None

    # 获取数据库模式
    def getDbSchema(self):
        try:
            # 确保站点需要文件"dbschema.json"
            self.site.needFile("dbschema.json")
            # 加载"dbschema.json"文件内容
            schema = self.loadJson("dbschema.json")
        except Exception as err:
            # 如果"dbschema.json"不是有效的 JSON，则抛出异常
            raise Exception("dbschema.json is not a valid JSON: %s" % err)
        # 返回数据库模式
        return schema

    # 加载数据库
    def loadDb(self):
        # 记录调试信息，表示没有数据库，正在等待"dbschema.json"文件
        self.log.debug("No database, waiting for dbschema.json...")
        # 确保站点需要文件"dbschema.json"，优先级为3
        self.site.needFile("dbschema.json", priority=3)
        # 记录调试信息，表示已经获取到"dbschema.json"文件
        self.log.debug("Got dbschema.json")
        # 重新检查"dbschema.json"文件是否存在
        self.has_db = self.isFile("dbschema.json")  # Recheck if dbschema exist
        # 如果存在数据库
        if self.has_db:
            # 获取数据库模式
            schema = self.getDbSchema()
            # 获取数据库文件路径
            db_path = self.getPath(schema["db_file"])
            # 如果数据库文件不存在或者大小为0
            if not os.path.isfile(db_path) or os.path.getsize(db_path) == 0:
                try:
                    # 重建数据库，原因为"Missing database"
                    self.rebuildDb(reason="Missing database")
                except Exception as err:
                    # 记录错误日志
                    self.log.error(err)
                    pass

            # 如果存在数据库连接
            if self.db:
                # 关闭数据库连接，原因为"Gettig new db for SiteStorage"
                self.db.close("Gettig new db for SiteStorage")
            # 打开数据库连接，同时关闭空闲连接
            self.db = self.openDb(close_idle=True)
            try:
                # 检查数据库表是否有变化
                changed_tables = self.db.checkTables()
                # 如果有变化的表
                if changed_tables:
                    # 重建数据库，不删除数据库，原因为"Changed tables"
                    self.rebuildDb(delete_db=False, reason="Changed tables")  # TODO: only update the changed table datas
            except sqlite3.OperationalError:
                pass

    # 返回数据库类
    @util.Noparallel()
    def getDb(self):
        # 如果数据库事件繁忙，则记录调试信息，表示正在等待数据库
        if self.event_db_busy:  # Db not ready for queries
            self.log.debug("Wating for db...")
            # 等待数据库事件
            self.event_db_busy.get()  # Wait for event
        # 如果没有数据库连接，则加载数据库
        if not self.db:
            self.loadDb()
        # 返回数据库连接
        return self.db
    # 更新数据库文件
    def updateDbFile(self, inner_path, file=None, cur=None):
        # 获取文件路径
        path = self.getPath(inner_path)
        # 如果有当前游标，则使用当前数据库，否则获取数据库
        if cur:
            db = cur.db
        else:
            db = self.getDb()
        # 调用数据库对象的更新方法，返回更新结果
        return db.updateJson(path, file, cur)

    # 获取站点可能的数据库文件
    @thread_pool_fs_read.wrap
    def getDbFiles(self):
        found = 0
        # 遍历站点内容管理器中的内容
        for content_inner_path, content in self.site.content_manager.contents.items():
            # 如果是 content.json 文件本身
            if self.isFile(content_inner_path):
                # 返回文件路径
                yield content_inner_path, self.getPath(content_inner_path)
            else:
                # 记录缺失的文件
                self.log.debug("[MISSING] %s" % content_inner_path)
            # 数据文件在 content.json 中
            content_inner_path_dir = helper.getDirname(content_inner_path)  # Content.json 相对于站点的目录
            # 遍历内容中的文件和可选文件
            for file_relative_path in list(content.get("files", {}).keys()) + list(content.get("files_optional", {}).keys()):
                # 如果不是以 .json 或 .json.gz 结尾的文件，则跳过
                if not file_relative_path.endswith(".json") and not file_relative_path.endswith("json.gz"):
                    continue  # 我们只关心 json 文件
                file_inner_path = content_inner_path_dir + file_relative_path  # 相对于站点目录的文件路径
                file_inner_path = file_inner_path.strip("/")  # 去除开头的 /
                # 如果文件存在，则返回文件路径
                if self.isFile(file_inner_path):
                    yield file_inner_path, self.getPath(file_inner_path)
                else:
                    # 记录缺失的文件
                    self.log.debug("[MISSING] %s" % file_inner_path)
                found += 1
                # 每找到 100 个文件，休眠一小段时间，避免 UI 阻塞
                if found % 100 == 0:
                    time.sleep(0.001)  # 切换上下文，避免 UI 阻塞

    # 重建 SQL 缓存
    @util.Noparallel()
    @thread_pool_fs_batch.wrap
    # 在数据库错误时执行 SQL 查询或重建
    # 查询数据库，执行给定的查询语句，可传入参数
    def query(self, query, params=None):
        # 如果查询语句不是以SELECT开头，则抛出异常
        if not query.strip().upper().startswith("SELECT"):
            raise Exception("Only SELECT query supported")

        try:
            # 执行查询语句并返回结果
            res = self.getDb().execute(query, params)
        except sqlite3.DatabaseError as err:
            # 如果捕获到数据库错误
            if err.__class__.__name__ == "DatabaseError":
                # 记录日志并尝试重建数据库
                self.log.error("Database error: %s, query: %s, try to rebuilding it..." % (err, query))
                try:
                    self.rebuildDb(reason="Query error")
                except sqlite3.OperationalError:
                    pass
                # 重新执行查询语句
                res = self.db.cur.execute(query, params)
            else:
                # 如果不是数据库错误，则抛出异常
                raise err
        # 返回查询结果
        return res

    # 确保目录存在，如果不存在则创建
    def ensureDir(self, inner_path):
        try:
            # 创建目录
            os.makedirs(self.getPath(inner_path))
        except OSError as err:
            # 如果目录已存在，则返回False
            if err.errno == errno.EEXIST:
                return False
            else:
                # 如果是其它错误，则抛出异常
                raise err
        # 目录创建成功，返回True
        return True

    # 打开文件对象
    def open(self, inner_path, mode="rb", create_dirs=False, **kwargs):
        # 获取文件路径
        file_path = self.getPath(inner_path)
        # 如果需要创建目录
        if create_dirs:
            # 获取文件所在目录路径，并确保目录存在
            file_inner_dir = os.path.dirname(inner_path)
            self.ensureDir(file_inner_dir)
        # 返回打开的文件对象
        return open(file_path, mode, **kwargs)

    # 读取文件对象
    @thread_pool_fs_read.wrap
    def read(self, inner_path, mode="rb"):
        # 打开文件并读取内容
        return open(self.getPath(inner_path), mode).read()

    # 写入文件对象
    @thread_pool_fs_write.wrap
    # 写入线程，将内容写入指定路径的文件
    def writeThread(self, inner_path, content):
        # 获取文件路径
        file_path = self.getPath(inner_path)
        # 如果目录不存在，则创建目录
        self.ensureDir(os.path.dirname(inner_path))
        # 如果内容是类文件对象
        if hasattr(content, 'read'):  # File-like object
            # 使用二进制写入模式打开文件，并将内容写入文件
            with open(file_path, "wb") as file:
                shutil.copyfileobj(content, file)  # Write buff to disk
        else:  # 如果内容是简单字符串
            # 如果文件路径为 "content.json" 并且文件已存在，则使用原子写入方式写入内容
            if inner_path == "content.json" and os.path.isfile(file_path):
                helper.atomicWrite(file_path, content)
            else:
                # 使用二进制写入模式打开文件，并将内容写入文件
                with open(file_path, "wb") as file:
                    file.write(content)

    # 将内容写入文件
    def write(self, inner_path, content):
        # 调用写入线程方法
        self.writeThread(inner_path, content)
        # 触发更新事件
        self.onUpdated(inner_path)

    # 从文件系统中删除文件
    def delete(self, inner_path):
        # 获取文件路径
        file_path = self.getPath(inner_path)
        # 删除文件
        os.unlink(file_path)
        # 触发更新事件，文件被删除
        self.onUpdated(inner_path, file=False)

    # 从文件系统中删除目录
    def deleteDir(self, inner_path):
        # 获取目录路径
        dir_path = self.getPath(inner_path)
        # 删除目录
        os.rmdir(dir_path)

    # 重命名文件或目录
    def rename(self, inner_path_before, inner_path_after):
        # 最多重试3次
        for retry in range(3):
            rename_err = None
            # 解决 "由于另一个进程正在使用文件，因此无法访问该进程" 错误
            try:
                # 重命名文件或目录
                os.rename(self.getPath(inner_path_before), self.getPath(inner_path_after))
                break
            except Exception as err:
                rename_err = err
                # 记录重命名错误
                self.log.error("%s rename error: %s (retry #%s)" % (inner_path_before, err, retry))
                time.sleep(0.1 + retry)
        # 如果重命名错误存在，则抛出异常
        if rename_err:
            raise rename_err

    # 从目录中列出文件
    @thread_pool_fs_read.wrap
    # 遍历指定目录下的文件和子目录，可选择忽略某些文件或目录
    def walk(self, dir_inner_path, ignore=None):
        # 获取目录的绝对路径
        directory = self.getPath(dir_inner_path)
        # 遍历目录下的所有文件和子目录
        for root, dirs, files in os.walk(directory):
            # 将路径中的反斜杠替换为斜杠
            root = root.replace("\\", "/")
            # 计算相对于目录的相对路径
            root_relative_path = re.sub("^%s" % re.escape(directory), "", root).lstrip("/")
            # 遍历目录下的文件
            for file_name in files:
                # 如果不是根目录，则计算文件的相对路径
                if root_relative_path:  # Not root dir
                    file_relative_path = root_relative_path + "/" + file_name
                else:
                    file_relative_path = file_name

                # 如果文件匹配忽略规则，则跳过
                if ignore and SafeRe.match(ignore, file_relative_path):
                    continue

                # 返回文件的相对路径
                yield file_relative_path

            # 如果设置了忽略规则，则过滤掉匹配忽略规则的子目录
            if ignore:
                dirs_filtered = []
                for dir_name in dirs:
                    # 如果不是根目录，则计算子目录的相对路径
                    if root_relative_path:
                        dir_relative_path = root_relative_path + "/" + dir_name
                    else:
                        dir_relative_path = dir_name

                    # 如果子目录匹配忽略规则，则跳过
                    if ignore == ".*" or re.match(".*([|(]|^)%s([|)]|$)" % re.escape(dir_relative_path + "/.*"), ignore):
                        continue

                    dirs_filtered.append(dir_name)
                dirs[:] = dirs_filtered

    # 列出指定目录下的所有文件和子目录
    @thread_pool_fs_read.wrap
    def list(self, dir_inner_path):
        # 获取目录的绝对路径
        directory = self.getPath(dir_inner_path)
        # 返回目录下的所有文件和子目录
        return os.listdir(directory)

    # 站点内容已更新
    # 当文件被更新时触发的方法，更新 SQL 缓存
    def onUpdated(self, inner_path, file=None):
        # 判断是否应该加载到数据库中
        should_load_to_db = inner_path.endswith(".json") or inner_path.endswith(".json.gz")
        # 如果文件路径为"dbschema.json"，则更新数据库模式标志
        if inner_path == "dbschema.json":
            self.has_db = self.isFile("dbschema.json")
            # 如果存在数据库，则重新打开数据库以检查更改
            if self.has_db:
                self.closeDb("New dbschema")
                # 异步获取数据库
                gevent.spawn(self.getDb)
        # 如果数据库未禁用且文件应该加载到数据库中且存在数据库
        elif not config.disable_db and should_load_to_db and self.has_db:  # Load json file to db
            # 如果配置为详细模式，则记录加载到数据库的json文件信息
            if config.verbose:
                self.log.debug("Loading json file to db: %s (file: %s)" % (inner_path, file))
            try:
                # 更新数据库中的json文件
                self.updateDbFile(inner_path, file)
            except Exception as err:
                # 记录json文件加载错误
                self.log.error("Json %s load error: %s" % (inner_path, Debug.formatException(err)))
                # 关闭数据库
                self.closeDb("Json load error")

    # 加载和解析json文件
    @thread_pool_fs_read.wrap
    def loadJson(self, inner_path):
        # 使用指定编码打开文件，并加载json数据
        with self.open(inner_path, "r", encoding="utf8") as file:
            return json.load(file)

    # 写入格式化的json文件
    def writeJson(self, inner_path, data):
        # 将数据转换为json格式并写入磁盘
        self.write(inner_path, helper.jsonDumps(data).encode("utf8"))

    # 获取文件大小
    def getSize(self, inner_path):
        # 获取文件路径并返回文件大小，如果出现异常则返回0
        path = self.getPath(inner_path)
        try:
            return os.path.getsize(path)
        except Exception:
            return 0

    # 判断文件是否存在
    def isFile(self, inner_path):
        # 判断文件是否存在
        return os.path.isfile(self.getPath(inner_path))

    # 判断文件或目录是否存在
    def isExists(self, inner_path):
        # 判断文件或目录是否存在
        return os.path.exists(self.getPath(inner_path))

    # 判断目录是否存在
    def isDir(self, inner_path):
        # 判断路径是否为目录
        return os.path.isdir(self.getPath(inner_path))

    # 安全检查并返回站点文件的路径
    # 获取内部路径对应的文件系统路径
    def getPath(self, inner_path):
        # 将内部路径中的反斜杠替换为斜杠，以修复 Windows 路径分隔符问题
        inner_path = inner_path.replace("\\", "/")  # Windows separator fix
        # 如果内部路径为空，则返回目录路径
        if not inner_path:
            return self.directory

        # 如果内部路径包含"../"，则抛出异常
        if "../" in inner_path:
            raise Exception("File not allowed: %s" % inner_path)

        # 返回目录路径和内部路径的组合
        return "%s/%s" % (self.directory, inner_path)

    # 获取站点目录相对路径
    def getInnerPath(self, path):
        # 如果路径等于目录路径，则内部路径为空
        if path == self.directory:
            inner_path = ""
        else:
            # 如果路径以目录路径开头，则内部路径为路径去掉目录路径部分
            if path.startswith(self.directory):
                inner_path = path[len(self.directory) + 1:]
            else:
                # 否则抛出异常
                raise Exception("File not allowed: %s" % path)
        return inner_path

    # 使用 content.json 验证所有文件的 sha512sum
    # 检查并尝试修复站点文件的完整性
    def updateBadFiles(self, quick_check=True):
        s = time.time()
        # 验证文件完整性，快速检查，添加可选文件，如果站点不是自有的则添加已更改文件
        res = self.verifyFiles(
            quick_check,
            add_optional=True,
            add_changed=not self.site.settings.get("own")  # Don't overwrite changed files if site owned
        )
        bad_files = res["bad_files"]
        self.site.bad_files = {}
        # 如果存在坏文件，则将其添加到站点的坏文件字典中
        if bad_files:
            for bad_file in bad_files:
                self.site.bad_files[bad_file] = 1
        # 记录日志，显示检查文件所花费的时间，发现的坏文件数量，以及是否进行了快速检查
        self.log.debug("Checked files in %.2fs... Found bad files: %s, Quick:%s" % (time.time() - s, len(bad_files), quick_check))

    # 使用线程池批量删除站点的所有文件
    @thread_pool_fs_batch.wrap
```
# `ZeroNet\plugins\OptionalManager\OptionalManagerPlugin.py`

```
# 导入时间模块
import time
# 导入正则表达式模块
import re
# 导入集合模块
import collections

# 导入协程模块
import gevent

# 从自定义工具模块中导入帮助函数
from util import helper
# 从插件模块中导入插件管理器
from Plugin import PluginManager
# 从当前目录下的内容数据库插件中导入内容数据库插件
from . import ContentDbPlugin

# 插件加载后执行的函数，用于导入插件主机类
@PluginManager.afterLoad
def importPluginnedClasses():
    # 声明全局变量config
    global config
    # 从配置模块中导入config
    from Config import config

# 处理访问日志的函数
def processAccessLog():
    # 声明全局变量access_log
    global access_log
    # 如果access_log存在
    if access_log:
        # 获取内容数据库插件的内容数据库
        content_db = ContentDbPlugin.content_db
        # 如果内容数据库连接不存在，则返回False
        if not content_db.conn:
            return False

        # 记录当前时间
        s = time.time()
        # 保存之前的访问日志
        access_log_prev = access_log
        # 重置访问日志
        access_log = collections.defaultdict(dict)
        # 获取当前时间戳
        now = int(time.time())
        # 初始化计数器
        num = 0
        # 遍历之前的访问日志
        for site_id in access_log_prev:
            # 执行SQL语句，更新文件可选表中的访问时间
            content_db.execute(
                "UPDATE file_optional SET time_accessed = %s WHERE ?" % now,
                {"site_id": site_id, "inner_path": list(access_log_prev[site_id].keys())}
            )
            # 更新计数器
            num += len(access_log_prev[site_id])

        # 记录日志
        content_db.log.debug("Inserted %s web request stat in %.3fs" % (num, time.time() - s))

# 处理请求日志的函数
def processRequestLog():
    # 声明全局变量request_log
    global request_log
    # 如果request_log存在
    if request_log:
        # 获取内容数据库插件的内容数据库
        content_db = ContentDbPlugin.content_db
        # 如果内容数据库连接不存在，则返回False
        if not content_db.conn:
            return False

        # 记录当前时间
        s = time.time()
        # 保存之前的请求日志
        request_log_prev = request_log
        # 重置请求日志
        request_log = collections.defaultdict(lambda: collections.defaultdict(int))  # {site_id: {inner_path1: 1, inner_path2: 1...}}
        # 初始化计数器
        num = 0
        # 遍历之前的请求日志
        for site_id in request_log_prev:
            for inner_path, uploaded in request_log_prev[site_id].items():
                # 执行SQL语句，更新文件可选表中的上传量
                content_db.execute(
                    "UPDATE file_optional SET uploaded = uploaded + %s WHERE ?" % uploaded,
                    {"site_id": site_id, "inner_path": inner_path}
                )
                # 更新计数器
                num += 1
        # 记录日志
        content_db.log.debug("Inserted %s file request stat in %.3fs" % (num, time.time() - s))

# 如果access_log不在局部变量中，则保持模块重新加载之间的状态
if "access_log" not in locals().keys():
    # 创建一个默认字典，用于存储访问日志，格式为 {site_id: {inner_path1: 1, inner_path2: 1...}}
    access_log = collections.defaultdict(dict)  
    # 创建一个默认字典，用于存储请求日志，格式为 {site_id: {inner_path1: 1, inner_path2: 1...}}
    request_log = collections.defaultdict(lambda: collections.defaultdict(int))  
    # 每隔61秒调用 processAccessLog 函数
    helper.timer(61, processAccessLog)
    # 每隔60秒调用 processRequestLog 函数
    helper.timer(60, processRequestLog)
# 将 ContentManagerPlugin 类注册到 PluginManager 的 ContentManager 插件中
@PluginManager.registerTo("ContentManager")
class ContentManagerPlugin(object):
    # 初始化方法，接受任意位置参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 初始化缓存是否被固定的字典
        self.cache_is_pinned = {}
        # 调用父类的初始化方法
        super(ContentManagerPlugin, self).__init__(*args, **kwargs)

    # 标记可选文件已下载的方法
    def optionalDownloaded(self, inner_path, hash_id, size=None, own=False):
        # 如果 inner_path 中包含 "|"，则表示是大文件的一部分
        if "|" in inner_path:  # Big file piece
            # 将 inner_path 拆分成文件内部路径和文件范围
            file_inner_path, file_range = inner_path.split("|")
        else:
            # 否则，直接将 inner_path 作为文件内部路径
            file_inner_path = inner_path

        # 更新数据库中的可选文件信息，标记为已下载
        self.contents.db.executeDelayed(
            "UPDATE file_optional SET time_downloaded = :now, is_downloaded = 1, peer = peer + 1 WHERE site_id = :site_id AND inner_path = :inner_path AND is_downloaded = 0",
            {"now": int(time.time()), "site_id": self.contents.db.site_ids[self.site.address], "inner_path": file_inner_path}
        )

        # 调用父类的标记可选文件已下载的方法
        return super(ContentManagerPlugin, self).optionalDownloaded(inner_path, hash_id, size, own)

    # 标记可选文件已移除的方法
    def optionalRemoved(self, inner_path, hash_id, size=None):
        # 更新数据库中的可选文件信息，标记为未下载和未固定，并减少对应站点的对等数
        res = self.contents.db.execute(
            "UPDATE file_optional SET is_downloaded = 0, is_pinned = 0, peer = peer - 1 WHERE site_id = :site_id AND inner_path = :inner_path AND is_downloaded = 1",
            {"site_id": self.contents.db.site_ids[self.site.address], "inner_path": inner_path}
        )

        # 如果更新了数据库中的记录
        if res.rowcount > 0:
            # 调用父类的标记可选文件已移除的方法
            back = super(ContentManagerPlugin, self).optionalRemoved(inner_path, hash_id, size)
            # 如果存在相同 hash_id 的其他文件，则重新添加到 hashfield 中
            if self.isDownloaded(hash_id=hash_id, force_check_db=True):
                self.hashfield.appendHashId(hash_id)
        else:
            back = False
        # 重置缓存是否被固定的字典
        self.cache_is_pinned = {}
        return back
    # 重命名文件或目录，更新数据库中的路径信息
    def optionalRenamed(self, inner_path_old, inner_path_new):
        # 调用父类方法执行重命名操作
        back = super(ContentManagerPlugin, self).optionalRenamed(inner_path_old, inner_path_new)
        # 重置缓存中的 is_pinned 标记
        self.cache_is_pinned = {}
        # 更新数据库中的文件路径信息
        self.contents.db.execute(
            "UPDATE file_optional SET inner_path = :inner_path_new WHERE site_id = :site_id AND inner_path = :inner_path_old",
            {"site_id": self.contents.db.site_ids[self.site.address], "inner_path_old": inner_path_old, "inner_path_new": inner_path_new}
        )
        # 返回操作结果
        return back

    # 检查文件是否已下载
    def isDownloaded(self, inner_path=None, hash_id=None, force_check_db=False):
        # 如果提供了 hash_id 并且不强制检查数据库，并且 hash_id 不在 hashfield 中，则返回 False
        if hash_id and not force_check_db and hash_id not in self.hashfield:
            return False

        # 如果提供了 inner_path，则查询数据库中对应文件的下载状态
        if inner_path:
            res = self.contents.db.execute(
                "SELECT is_downloaded FROM file_optional WHERE site_id = :site_id AND inner_path = :inner_path LIMIT 1",
                {"site_id": self.contents.db.site_ids[self.site.address], "inner_path": inner_path}
            )
        # 如果提供了 hash_id，则查询数据库中对应文件的下载状态
        else:
            res = self.contents.db.execute(
                "SELECT is_downloaded FROM file_optional WHERE site_id = :site_id AND hash_id = :hash_id AND is_downloaded = 1 LIMIT 1",
                {"site_id": self.contents.db.site_ids[self.site.address], "hash_id": hash_id}
            )
        # 获取查询结果的第一行数据
        row = res.fetchone()
        # 如果查询结果存在且 is_downloaded 为 True，则返回 True，否则返回 False
        if row and row["is_downloaded"]:
            return True
        else:
            return False
    # 检查指定的 inner_path 是否已经被缓存为 pinned
    def isPinned(self, inner_path):
        # 如果 inner_path 已经在缓存中，则打印日志并返回缓存中的值
        if inner_path in self.cache_is_pinned:
            self.site.log.debug("Cached is pinned: %s" % inner_path)
            return self.cache_is_pinned[inner_path]
    
        # 从数据库中查询指定 inner_path 的 is_pinned 值
        res = self.contents.db.execute(
            "SELECT is_pinned FROM file_optional WHERE site_id = :site_id AND inner_path = :inner_path LIMIT 1",
            {"site_id": self.contents.db.site_ids[self.site.address], "inner_path": inner_path}
        )
        row = res.fetchone()
    
        # 如果查询结果存在且 is_pinned 为真，则设置 is_pinned 为 True
        if row and row[0]:
            is_pinned = True
        else:
            is_pinned = False
    
        # 将 inner_path 对应的 is_pinned 值缓存起来
        self.cache_is_pinned[inner_path] = is_pinned
        self.site.log.debug("Cache set is pinned: %s %s" % (inner_path, is_pinned))
    
        # 返回 is_pinned 值
        return is_pinned
    
    # 设置指定 inner_path 的 is_pinned 值
    def setPin(self, inner_path, is_pinned):
        content_db = self.contents.db
        site_id = content_db.site_ids[self.site.address]
        # 更新数据库中指定 inner_path 的 is_pinned 值
        content_db.execute("UPDATE file_optional SET is_pinned = %d WHERE ?" % is_pinned, {"site_id": site_id, "inner_path": inner_path})
        # 清空缓存
        self.cache_is_pinned = {}
    
    # 删除指定的 inner_path 对应的可选文件
    def optionalDelete(self, inner_path):
        # 如果 inner_path 已经被标记为 pinned，则打印日志并返回 False
        if self.isPinned(inner_path):
            self.site.log.debug("Skip deleting pinned optional file: %s" % inner_path)
            return False
        else:
            # 否则调用父类的 optionalDelete 方法进行删除
            return super(ContentManagerPlugin, self).optionalDelete(inner_path)
# 将 WorkerManagerPlugin 类注册到 PluginManager 的 WorkerManager 插件中
@PluginManager.registerTo("WorkerManager")
class WorkerManagerPlugin(object):
    # 完成任务的方法
    def doneTask(self, task):
        # 调用父类的 doneTask 方法
        super(WorkerManagerPlugin, self).doneTask(task)

        # 如果任务中包含可选哈希 ID 并且没有其他任务在执行，则立即执行延迟查询
        if task["optional_hash_id"] and not self.tasks:
            ContentDbPlugin.content_db.processDelayed()


# 将 UiRequestPlugin 类注册到 PluginManager 的 UiRequest 插件中
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # 解析路径的方法
    def parsePath(self, path):
        # 使用父类的 parsePath 方法解析路径
        path_parts = super(UiRequestPlugin, self).parsePath(path)
        if path_parts:
            # 获取请求地址对应的站点 ID
            site_id = ContentDbPlugin.content_db.site_ids.get(path_parts["request_address"])
            if site_id:
                # 如果站点 ID 存在，并且请求的文件是可选文件，则记录访问日志
                if ContentDbPlugin.content_db.isOptionalFile(site_id, path_parts["inner_path"]):
                    access_log[site_id][path_parts["inner_path"]] = 1
        return path_parts


# 将 FileRequestPlugin 类注册到 PluginManager 的 FileRequest 插件中
@PluginManager.registerTo("FileRequest")
class FileRequestPlugin(object):
    # 获取文件的方法
    def actionGetFile(self, params):
        # 调用父类的 actionGetFile 方法获取文件
        stats = super(FileRequestPlugin, self).actionGetFile(params)
        # 记录文件请求
        self.recordFileRequest(params["site"], params["inner_path"], stats)
        return stats

    # 流式传输文件的方法
    def actionStreamFile(self, params):
        # 调用父类的 actionStreamFile 方法进行文件流式传输
        stats = super(FileRequestPlugin, self).actionStreamFile(params)
        # 记录文件请求
        self.recordFileRequest(params["site"], params["inner_path"], stats)
        return stats

    # 记录文件请求的方法
    def recordFileRequest(self, site_address, inner_path, stats):
        if not stats:
            # 只跟踪文件的最后一次请求
            return False
        site_id = ContentDbPlugin.content_db.site_ids[site_address]
        if site_id and ContentDbPlugin.content_db.isOptionalFile(site_id, inner_path):
            request_log[site_id][inner_path] += stats["bytes_sent"]


# 将 SitePlugin 类注册到 PluginManager 的 Site 插件中
@PluginManager.registerTo("Site")
class SitePlugin(object):
    # 检查给定的内部路径是否可下载
    def isDownloadable(self, inner_path):
        # 调用父类的方法检查内部路径是否可下载
        is_downloadable = super(SitePlugin, self).isDownloadable(inner_path)
        # 如果可下载，则返回结果
        if is_downloadable:
            return is_downloadable

        # 遍历可选帮助的路径，如果内部路径以某个可选帮助路径开头，则可下载
        for path in self.settings.get("optional_help", {}).keys():
            if inner_path.startswith(path):
                return True

        # 如果以上条件都不满足，则不可下载
        return False

    # 处理文件被遗忘的情况
    def fileForgot(self, inner_path):
        # 如果内部路径包含 "|" 并且文件已被固定，则不处理文件被遗忘的情况
        if "|" in inner_path and self.content_manager.isPinned(re.sub(r"\|.*", "", inner_path)):
            self.log.debug("File %s is pinned, no fileForgot" % inner_path)
            return False
        else:
            # 否则调用父类的方法处理文件被遗忘的情况
            return super(SitePlugin, self).fileForgot(inner_path)

    # 处理文件完成的情况
    def fileDone(self, inner_path):
        # 如果内部路径包含 "|" 并且该文件被标记为坏文件超过5次，则处理空闲可选文件完成的情况
        if "|" in inner_path and self.bad_files.get(inner_path, 0) > 5:  # Idle optional file done
            inner_path_file = re.sub(r"\|.*", "", inner_path)
            num_changed = 0
            # 遍历坏文件字典，将以内部路径文件开头且标记次数大于1的文件标记次数设为1
            for key, val in self.bad_files.items():
                if key.startswith(inner_path_file) and val > 1:
                    self.bad_files[key] = 1
                    num_changed += 1
            # 记录日志，说明空闲可选文件块完成，改变了重试次数的文件块数量
            self.log.debug("Idle optional file piece done, changed retry number of %s pieces." % num_changed)
            # 如果有文件块的重试次数被改变，则异步调用重试坏文件的方法
            if num_changed:
                gevent.spawn(self.retryBadFiles)

        # 调用父类的方法处理文件完成的情况
        return super(SitePlugin, self).fileDone(inner_path)
# 将 ConfigPlugin 类注册到 PluginManager 的 ConfigPlugin 插件中
@PluginManager.registerTo("ConfigPlugin")
class ConfigPlugin(object):
    # 创建参数
    def createArguments(self):
        # 添加一个参数组到解析器中
        group = self.parser.add_argument_group("OptionalManager plugin")
        # 添加一个可选参数 --optional_limit，用于限制可选文件的总大小，默认为"10%"，可以是GB或剩余空间的百分比
        group.add_argument('--optional_limit', help='Limit total size of optional files', default="10%", metavar="GB or free space %")
        # 添加一个可选参数 --optional_limit_exclude_minsize，用于排除大于指定大小的文件，不计入可选文件大小限制，默认为20MB
        group.add_argument('--optional_limit_exclude_minsize', help='Exclude files larger than this limit from optional size limit calculation', default=20, metavar="MB", type=int)

        # 调用父类的 createArguments 方法
        return super(ConfigPlugin, self).createArguments()
```
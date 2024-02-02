# `ZeroNet\src\Worker\Worker.py`

```py
# 导入时间模块
import time

# 导入协程模块
import gevent
import gevent.lock

# 从 Debug 模块中导入 Debug 类
from Debug import Debug

# 从 Config 模块中导入 config 对象
from Config import config

# 从 Content.ContentManager 模块中导入 VerifyError 异常类
from Content.ContentManager import VerifyError

# 定义 WorkerDownloadError 异常类
class WorkerDownloadError(Exception):
    pass

# 定义 WorkerIOError 异常类
class WorkerIOError(Exception):
    pass

# 定义 WorkerStop 异常类
class WorkerStop(Exception):
    pass

# 定义 Worker 类
class Worker(object):

    # 初始化方法
    def __init__(self, manager, peer):
        self.manager = manager
        self.peer = peer
        self.task = None
        self.key = None
        self.running = False
        self.thread = None
        self.num_downloaded = 0
        self.num_failed = 0

    # 返回 Worker 对象的字符串表示
    def __str__(self):
        return "Worker %s %s" % (self.manager.site.address_short, self.key)

    # 返回 Worker 对象的字符串表示
    def __repr__(self):
        return "<%s>" % self.__str__()

    # 等待任务的方法
    def waitForTask(self, task, timeout):  # Wait for other workers to finish the task
        for sleep_i in range(1, timeout * 10):
            time.sleep(0.1)
            if task["done"] or task["workers_num"] == 0:
                if config.verbose:
                    self.manager.log.debug("%s: %s, picked task free after %ss sleep. (done: %s)" % (
                        self.key, task["inner_path"], 0.1 * sleep_i, task["done"]
                    ))
                break

            if sleep_i % 10 == 0:
                workers = self.manager.findWorkers(task)
                if not workers or not workers[0].peer.connection:
                    break
                worker_idle = time.time() - workers[0].peer.connection.last_recv_time
                if worker_idle > 1:
                    if config.verbose:
                        self.manager.log.debug("%s: %s, worker %s seems idle, picked up task after %ss sleep. (done: %s)" % (
                            self.key, task["inner_path"], workers[0].key, 0.1 * sleep_i, task["done"]
                        ))
                    break
        return True
    def pickTask(self):  # Find and select a new task for the worker
        # 从管理器获取任务
        task = self.manager.getTask(self.peer)
        if not task:  # No more task
            # 如果没有任务，则等待一段时间
            time.sleep(0.1)  # Wait a bit for new tasks
            # 再次尝试获取任务
            task = self.manager.getTask(self.peer)
            if not task:  # Still no task, stop it
                # 如果仍然没有任务，则停止并记录统计信息
                stats = "downloaded files: %s, failed: %s" % (self.num_downloaded, self.num_failed)
                self.manager.log.debug("%s: No task found, stopping (%s)" % (self.key, stats))
                return False

        if not task["time_started"]:
            task["time_started"] = time.time()  # Task started now

        if task["workers_num"] > 0:  # Wait a bit if someone already working on it
            if task["peers"]:  # It's an update
                timeout = 3
            else:
                timeout = 1

            if task["size"] > 100 * 1024 * 1024:
                timeout = timeout * 2

            if config.verbose:
                # 如果配置为详细模式，则记录等待信息
                self.manager.log.debug("%s: Someone already working on %s (pri: %s), sleeping %s sec..." % (
                    self.key, task["inner_path"], task["priority"], timeout
                ))

            self.waitForTask(task, timeout)
        return task

    def downloadTask(self, task):
        try:
            # 从对等方获取文件数据
            buff = self.peer.getFile(task["site"].address, task["inner_path"], task["size"])
        except Exception as err:
            # 如果出现异常，则记录错误信息并抛出下载错误
            self.manager.log.debug("%s: getFile error: %s" % (self.key, err))
            raise WorkerDownloadError(str(err))

        if not buff:
            # 如果没有获取到数据，则抛出下载错误
            raise WorkerDownloadError("No response")

        return buff

    def getTaskLock(self, task):
        if task["lock"] is None:
            # 如果任务的锁为空，则创建一个新的锁
            task["lock"] = gevent.lock.Semaphore()
        return task["lock"]
    # 将任务写入存储，使用给定的缓冲区
    def writeTask(self, task, buff):
        # 将缓冲区指针移动到起始位置
        buff.seek(0)
        try:
            # 尝试将任务的内部路径和缓冲区内容写入到任务所在站点的存储中
            task["site"].storage.write(task["inner_path"], buff)
        except Exception as err:
            # 如果出现异常，根据异常类型进行处理
            if type(err) == Debug.Notify:
                # 如果是 Debug.Notify 类型的异常，记录调试信息
                self.manager.log.debug("%s: Write aborted: %s (%s: %s)" % (self.key, task["inner_path"], type(err), err))
            else:
                # 否则记录错误信息
                self.manager.log.error("%s: Error writing: %s (%s: %s)" % (self.key, task["inner_path"], type(err), err))
            # 抛出自定义的 WorkerIOError 异常
            raise WorkerIOError(str(err))

    # 当任务验证失败时的处理函数
    def onTaskVerifyFail(self, task, error_message):
        # 增加失败计数
        self.num_failed += 1
        # 如果任务数小于 50 或者设置了 verbose 标志，则记录调试信息
        if self.manager.started_task_num < 50 or config.verbose:
            self.manager.log.debug(
                "%s: Verify failed: %s, error: %s, failed peers: %s, workers: %s" %
                (self.key, task["inner_path"], error_message, len(task["failed"]), task["workers_num"])
            )
        # 将当前节点添加到任务的失败列表中
        task["failed"].append(self.peer)
        # 增加当前节点的哈希失败计数
        self.peer.hash_failed += 1
        # 如果哈希失败计数大于等于任务数的最大值或者连接错误计数大于 10，则抛出 WorkerStop 异常
        if self.peer.hash_failed >= max(len(self.manager.tasks), 3) or self.peer.connection_error > 10:
            # 抛出异常并附带错误信息
            raise WorkerStop(
                "Too many errors (hash failed: %s, connection error: %s)" %
                (self.peer.hash_failed, self.peer.connection_error)
            )
    # 处理任务的方法，接受一个任务对象作为参数
    def handleTask(self, task):
        # 初始化下载错误和写入错误标志
        download_err = write_err = False

        # 初始化写入锁
        write_lock = None
        try:
            # 下载任务内容到缓冲区
            buff = self.downloadTask(task)

            # 如果任务已完成，则尝试查找新任务
            if task["done"] is True:
                return None

            # 如果工作线程不再需要或已被终止
            if self.running is False:
                self.manager.log.debug("%s: No longer needed, returning: %s" % (self.key, task["inner_path"]))
                # 抛出自定义的 WorkerStop 异常
                raise WorkerStop("Running got disabled")

            # 获取任务的写入锁，并加锁
            write_lock = self.getTaskLock(task)
            write_lock.acquire()
            # 验证文件内容是否与任务中的内容一致
            if task["site"].content_manager.verifyFile(task["inner_path"], buff) is None:
                is_same = True
            else:
                is_same = False
            is_valid = True
        except (WorkerDownloadError, VerifyError) as err:
            # 如果下载或验证出现错误，设置相应的标志
            download_err = err
            is_valid = False
            is_same = False

        # 如果内容有效且不一致
        if is_valid and not is_same:
            # 如果任务数小于50或任务优先级大于10或配置为详细模式
            if self.manager.started_task_num < 50 or task["priority"] > 10 or config.verbose:
                self.manager.log.debug("%s: Verify correct: %s" % (self.key, task["inner_path"]))
            try:
                # 写入任务内容
                self.writeTask(task, buff)
            except WorkerIOError as err:
                # 如果写入出现错误，设置相应的标志
                write_err = err

        # 如果任务未完成
        if not task["done"]:
            # 如果写入错误，标记任务失败并记录日志
            if write_err:
                self.manager.failTask(task, reason="Write error")
                self.num_failed += 1
                self.manager.log.error("%s: Error writing %s: %s" % (self.key, task["inner_path"], write_err))
            # 如果内容有效，标记任务完成并增加已下载数量
            elif is_valid:
                self.manager.doneTask(task)
                self.num_downloaded += 1

        # 如果写入锁存在且已加锁，则释放锁
        if write_lock is not None and write_lock.locked():
            write_lock.release()

        # 如果内容无效，触发任务验证失败处理方法，然后休眠1秒并返回 False
        if not is_valid:
            self.onTaskVerifyFail(task, download_err)
            time.sleep(1)
            return False

        # 返回 True
        return True
    # 下载器方法，用于处理文件下载任务
    def downloader(self):
        # 重置哈希错误计数器
        self.peer.hash_failed = 0
        # 当线程正在运行时执行循环
        while self.running:
            # 尝试获取空闲的文件下载任务
            task = self.pickTask()

            # 如果没有任务，则退出循环
            if not task:
                break

            # 如果任务已完成，则继续下一个任务
            if task["done"]:
                continue

            # 设置当前任务
            self.task = task

            # 将任务添加到任务管理器中
            self.manager.addTaskWorker(task, self)

            try:
                # 处理任务
                success = self.handleTask(task)
            except WorkerStop as err:
                # 如果出现异常，记录日志并移除任务
                self.manager.log.debug("%s: Worker stopped: %s" % (self.key, err))
                self.manager.removeTaskWorker(task, self)
                break

            # 移除任务
            self.manager.removeTaskWorker(task, self)

        # 通知对等方工作已完成
        self.peer.onWorkerDone()
        # 设置运行状态为 False
        self.running = False
        # 移除工作线程
        self.manager.removeWorker(self)

    # 启动工作线程
    def start(self):
        self.running = True
        self.thread = gevent.spawn(self.downloader)

    # 跳过当前任务
    def skip(self, reason="Unknown"):
        # 记录跳过任务的日志
        self.manager.log.debug("%s: Force skipping (reason: %s)" % (self.key, reason))
        # 如果线程存在，则终止线程
        if self.thread:
            self.thread.kill(exception=Debug.createNotifyType("Worker skipping (reason: %s)" % reason))
        # 重新启动工作线程
        self.start()

    # 强制停止工作线程
    def stop(self, reason="Unknown"):
        # 记录停止工作线程的日志
        self.manager.log.debug("%s: Force stopping (reason: %s)" % (self.key, reason))
        # 设置运行状态为 False
        self.running = False
        # 如果线程存在，则终止线程
        if self.thread:
            self.thread.kill(exception=Debug.createNotifyType("Worker stopped (reason: %s)" % reason))
        # 删除线程
        del self.thread
        # 移除工作线程
        self.manager.removeWorker(self)
```
# `ZeroNet\src\Worker\WorkerManager.py`

```
# 导入时间模块
import time
# 导入日志模块
import logging
# 导入collections模块
import collections
# 导入gevent模块
import gevent
# 从当前目录下的Worker模块中导入Worker类
from .Worker import Worker
# 从当前目录下的WorkerTaskManager模块中导入WorkerTaskManager类
from .WorkerTaskManager import WorkerTaskManager
# 从Config模块中导入config对象
from Config import config
# 从util模块中导入helper函数
from util import helper
# 从Plugin模块中导入PluginManager类
from Plugin import PluginManager
# 从Debug目录下的DebugLock模块中导入DebugLock类
from Debug.DebugLock import DebugLock
# 导入util模块
import util

# 使用PluginManager.acceptPlugins装饰器注册插件
@PluginManager.acceptPlugins
# 定义WorkerManager类
class WorkerManager(object):

    # 初始化方法，接收site参数
    def __init__(self, site):
        # 设置site属性为传入的site参数
        self.site = site
        # 初始化workers属性为空字典，用于存储Worker对象
        self.workers = {}  # Key: ip:port, Value: Worker.Worker
        # 初始化tasks属性为WorkerTaskManager对象
        self.tasks = WorkerTaskManager()
        # 初始化next_task_id属性为1
        self.next_task_id = 1
        # 初始化lock_add_task属性为DebugLock对象，名称为"Lock AddTask:站点地址"
        self.lock_add_task = DebugLock(name="Lock AddTask:%s" % self.site.address_short)
        # 初始化started_task_num属性为0
        self.started_task_num = 0  # Last added task num
        # 初始化asked_peers属性为空列表
        self.asked_peers = []
        # 初始化running属性为True
        self.running = True
        # 初始化time_task_added属性为0
        self.time_task_added = 0
        # 初始化log属性为名称为"WorkerManager:站点地址"的日志记录器
        self.log = logging.getLogger("WorkerManager:%s" % self.site.address_short)
        # 在site的greenlet_manager中创建一个新的协程，运行checkTasks方法
        self.site.greenlet_manager.spawn(self.checkTasks)

    # 返回对象的字符串表示
    def __str__(self):
        return "WorkerManager %s" % self.site.address_short

    # 返回对象的字符串表示
    def __repr__(self):
        return "<%s>" % self.__str__()

    # 检查过期任务，返回下一个空闲或工作量较少的任务
    def getTask(self, peer):
        # 遍历tasks属性中的任务
        for task in self.tasks:  # Find a task
            # 如果peer不在任务的peers列表中，则继续下一次循环
            if task["peers"] and peer not in task["peers"]:
                continue  # This peer not allowed to pick this task
            # 如果peer在任务的failed列表中，则继续下一次循环
            if peer in task["failed"]:
                continue  # Peer already tried to solve this, but failed
            # 如果任务的optional_hash_id存在且peers为None，则继续下一次循环
            if task["optional_hash_id"] and task["peers"] is None:
                continue  # No peers found yet for the optional task
            # 如果任务已完成，则继续下一次循环
            if task["done"]:
                continue
            # 返回当前任务
            return task
    # 从任务列表中移除已解决的文件任务
    def removeSolvedFileTasks(self, mark_as_good=True):
        # 遍历任务列表的副本
        for task in self.tasks[:]:
            # 如果任务的内部路径不在站点的坏文件列表中
            if task["inner_path"] not in self.site.bad_files:
                # 输出调试信息，标记任务为好或坏，并移除任务
                self.log.debug("No longer in bad_files, marking as %s: %s" % (mark_as_good, task["inner_path"]))
                task["done"] = True
                task["evt"].set(mark_as_good)
                self.tasks.remove(task)
        # 如果任务列表为空，则重置已开始任务数为0
        if not self.tasks:
            self.started_task_num = 0
        # 更新网页套接字
        self.site.updateWebsocket()

    # 当站点添加新的对等方时
    def onPeers(self):
        # 启动工作线程，原因是发现更多对等方
        self.startWorkers(reason="More peers found")

    # 获取最大工作线程数
    def getMaxWorkers(self):
        if len(self.tasks) > 50:
            return config.workers * 3
        else:
            return config.workers

    # 添加新的工作线程
    def addWorker(self, peer, multiplexing=False, force=False):
        key = peer.key
        # 如果工作线程数超过最大工作线程数且不强制添加，则返回 False
        if len(self.workers) > self.getMaxWorkers() and not force:
            return False
        # 如果启用多路复用，则即使已经有该对等方的工作线程，也要添加
        if multiplexing:
            key = "%s/%s" % (key, len(self.workers))
        # 如果工作线程列表中没有该对等方的工作线程
        if key not in self.workers:
            # 获取对等方的任务
            task = self.getTask(peer)
            if task:
                # 创建并启动工作线程
                worker = Worker(self, peer)
                self.workers[key] = worker
                worker.key = key
                worker.start()
                return worker
            else:
                return False
        else:
            # 如果已经有该对等方的工作线程或者超过了最大限制，则返回 False
            return False

    # 将对等方添加到任务中
    def taskAddPeer(self, task, peer):
        # 如果任务的对等方列表为空，则初始化为空列表
        if task["peers"] is None:
            task["peers"] = []
        # 如果对等方在任务的失败列表中，则返回 False
        if peer in task["failed"]:
            return False
        # 如果对等方不在任务的对等方列表中，则添加到列表中并返回 True
        if peer not in task["peers"]:
            task["peers"].append(peer)
        return True

    # 启动工作线程来处理任务
    # 启动工作线程，处理任务
    def startWorkers(self, peers=None, force_num=0, reason="Unknown"):
        # 如果没有任务，则返回 False
        if not self.tasks:
            return False  # No task for workers
        # 获取最大工作线程数
        max_workers = min(self.getMaxWorkers(), len(self.site.peers))
        # 如果当前工作线程数已经达到最大，并且没有指定要启动的工作线程，则返回 False
        if len(self.workers) >= max_workers and not peers:
            return False  # Workers number already maxed and no starting peers defined
        # 记录调试日志
        self.log.debug(
            "Starting workers (%s), tasks: %s, peers: %s, workers: %s" %
            (reason, len(self.tasks), len(peers or []), len(self.workers))
        )
        # 如果没有指定要启动的工作线程，则获取已连接的对等节点作为启动工作线程的对等节点
        if not peers:
            peers = self.site.getConnectedPeers()
            # 如果已连接的对等节点数量小于最大工作线程数，则添加最近的对等节点作为启动工作线程的对等节点
            if len(peers) < max_workers:
                peers += self.site.getRecentPeers(max_workers * 2)
        # 如果对等节点的类型是集合，则转换为列表
        if type(peers) is set:
            peers = list(peers)

        # 根据对等节点的延迟进行排序
        peers.sort(key=lambda peer: peer.connection.last_ping_delay if peer.connection and peer.connection.last_ping_delay and len(peer.connection.waiting_requests) == 0 and peer.connection.connected else 9999)

        # 为每个对等节点创建一个工作线程
        for peer in peers:  # One worker for every peer
            # 如果指定了对等节点，并且当前对等节点不在指定的对等节点列表中，则跳过当前对等节点
            if peers and peer not in peers:
                continue  # If peers defined and peer not valid

            # 如果指定了强制启动的工作线程数量，则强制添加工作线程
            if force_num:
                worker = self.addWorker(peer, force=True)
                force_num -= 1
            else:
                worker = self.addWorker(peer)

            # 如果成功添加了工作线程，则记录调试日志
            if worker:
                self.log.debug("Added worker: %s (rep: %s), workers: %s/%s" % (peer.key, peer.reputation, len(self.workers), max_workers))

    # 在本地哈希表中查找可选哈希的对等节点，并添加到任务对等节点中
    # 寻找可选任务的对等方，如果需要重置任务则重置任务状态
    def findOptionalTasks(self, optional_tasks, reset_task=False):
        # 创建一个默认值为列表的字典，用于存储找到的对等方
        found = collections.defaultdict(list)  # { found_hash: [peer1, peer2...], ...}

        # 遍历站点中的所有对等方
        for peer in list(self.site.peers.values()):
            # 如果对等方没有哈希字段，则跳过
            if not peer.has_hashfield:
                continue

            # 将对等方的哈希字段转换为集合，以便更快地进行查找
            hashfield_set = set(peer.hashfield)  # Finding in set is much faster
            # 遍历所有可选任务
            for task in optional_tasks:
                optional_hash_id = task["optional_hash_id"]
                # 如果可选哈希 ID 在对等方的哈希字段集合中
                if optional_hash_id in hashfield_set:
                    # 如果需要重置任务并且任务的失败列表不为空，则清空失败列表
                    if reset_task and len(task["failed"]) > 0:
                        task["failed"] = []
                    # 如果对等方在任务的失败列表中，则跳过
                    if peer in task["failed"]:
                        continue
                    # 将对等方添加到任务中，并将对等方添加到找到的对等方字典中
                    if self.taskAddPeer(task, peer):
                        found[optional_hash_id].append(peer)

        return found

    # 在本地哈希表中查找可选哈希 ID 的对等方
    def findOptionalHashIds(self, optional_hash_ids, limit=0):
        # 创建一个默认值为列表的字典，用于存储找到的对等方
        found = collections.defaultdict(list)  # { found_hash_id: [peer1, peer2...], ...}

        # 遍历站点中的所有对等方
        for peer in list(self.site.peers.values()):
            # 如果对等方没有哈希字段，则跳过
            if not peer.has_hashfield:
                continue

            # 将对等方的哈希字段转换为集合，以便更快地进行查找
            hashfield_set = set(peer.hashfield)  # Finding in set is much faster
            # 遍历所有可选哈希 ID
            for optional_hash_id in optional_hash_ids:
                # 如果可选哈希 ID 在对等方的哈希字段集合中
                if optional_hash_id in hashfield_set:
                    # 将对等方添加到找到的对等方字典中
                    found[optional_hash_id].append(peer)
                    # 如果限制大于 0 并且找到的对等方数量达到限制，则从可选哈希 ID 列表中移除该 ID
                    if limit and len(found[optional_hash_id]) >= limit:
                        optional_hash_ids.remove(optional_hash_id)

        return found

    # 从找到的结果中为任务添加对等方
    # 为任务添加可选的对等节点
    def addOptionalPeers(self, found_ips):
        # 创建一个默认值为列表的字典
        found = collections.defaultdict(list)
        # 遍历找到的 IP 地址字典
        for hash_id, peer_ips in found_ips.items():
            # 查找具有相同哈希 ID 的任务
            task = [task for task in self.tasks if task["optional_hash_id"] == hash_id]
            # 如果找到了任务，则取第一个
            if task:  # Found task, lets take the first
                task = task[0]
            else:
                continue
            # 遍历对等节点 IP 地址列表
            for peer_ip in peer_ips:
                # 向站点添加对等节点，并返回对等节点对象
                peer = self.site.addPeer(peer_ip[0], peer_ip[1], return_peer=True, source="optional")
                # 如果对等节点不存在，则继续下一次循环
                if not peer:
                    continue
                # 如果成功为任务添加对等节点，则将对等节点添加到找到的字典中
                if self.taskAddPeer(task, peer):
                    found[hash_id].append(peer)
                # 如果对等节点包含哈希 ID，则将对等节点的哈希字段追加哈希 ID
                if peer.hashfield.appendHashId(hash_id):  # Peer has this file
                    peer.time_hashfield = None  # Peer hashfield probably outdated

        return found

    # 开始为可选文件查找对等节点
    @util.Noparallel(blocking=False, ignore_args=True)
    # 停止所有工作线程
    def stopWorkers(self):
        num = 0
        # 遍历所有工作线程并停止
        for worker in list(self.workers.values()):
            worker.stop(reason="Stopping all workers")
            num += 1
        tasks = self.tasks[:]  # 复制任务列表
        # 将所有当前任务标记为失败
        for task in tasks:  # Mark all current task as failed
            self.failTask(task, reason="Stopping all workers")
        return num

    # 通过任务查找工作线程
    def findWorkers(self, task):
        workers = []
        # 遍历所有工作线程并找到与任务相关的工作线程
        for worker in list(self.workers.values()):
            if worker.task == task:
                workers.append(worker)
        return workers

    # 结束并移除一个工作线程
    # 从工作队列中移除指定的 worker
    def removeWorker(self, worker):
        # 将 worker 的 running 属性设置为 False
        worker.running = False
        # 如果 worker 的 key 在 workers 字典中，则删除该 worker
        if worker.key in self.workers:
            del(self.workers[worker.key])
            # 记录日志，显示移除 worker 后的 workers 数量和最大 workers 数量
            self.log.debug("Removed worker, workers: %s/%s" % (len(self.workers), self.getMaxWorkers()))
        # 如果 workers 数量小于等于最大 workers 数量的三分之一，并且 asked_peers 的数量小于 10
        if len(self.workers) <= self.getMaxWorkers() / 3 and len(self.asked_peers) < 10:
            # 查找 tasks 中第一个 optional_hash_id 不为空的任务
            optional_task = next((task for task in self.tasks if task["optional_hash_id"]), None)
            # 如果找到 optional_task
            if optional_task:
                # 如果 workers 数量为 0，则启动查找 optional 任务
                if len(self.workers) == 0:
                    self.startFindOptional(find_more=True)
                # 否则，启动查找 optional 任务
                else:
                    self.startFindOptional()
            # 如果 tasks 不为空，workers 为空，worker 有任务，并且任务失败次数小于 20
            elif self.tasks and not self.workers and worker.task and len(worker.task["failed"]) < 20:
                # 记录日志，显示启动新 workers，以及 tasks 的数量
                self.log.debug("Starting new workers... (tasks: %s)" % len(self.tasks))
                # 启动 workers，原因是移除 worker
                self.startWorkers(reason="Removed worker")

    # 根据 inner_path 返回任务的优先级
    def getPriorityBoost(self, inner_path):
        # 如果 inner_path 是 "content.json"，则返回 9999，表示 content.json 优先级最高
        if inner_path == "content.json":
            return 9999  # Content.json always priority
        # 如果 inner_path 是 "index.html"，则返回 9998，表示 index.html 也很重要
        if inner_path == "index.html":
            return 9998  # index.html also important
        # 如果 inner_path 包含 "-default"，则返回 -4，表示默认文件不重要
        if "-default" in inner_path:
            return -4  # Default files are cloning not important
        # 如果 inner_path 以 "all.css" 结尾，则返回 14，表示提高 css 文件的优先级
        elif inner_path.endswith("all.css"):
            return 14  # boost css files priority
        # 如果 inner_path 以 "all.js" 结尾，则返回 13，表示提高 js 文件的优先级
        elif inner_path.endswith("all.js"):
            return 13  # boost js files priority
        # 如果 inner_path 以 "dbschema.json" 结尾，则返回 12，表示提高数据库规范的优先级
        elif inner_path.endswith("dbschema.json"):
            return 12  # boost database specification
        # 如果 inner_path 以 "content.json" 结尾，则返回 1，表示略微提高包含 content.json 的文件的优先级
        elif inner_path.endswith("content.json"):
            return 1  # boost included content.json files priority a bit
        # 如果 inner_path 以 ".json" 结尾
        elif inner_path.endswith(".json"):
            # 如果 inner_path 的长度小于 50，则返回 11，表示提高非用户 json 文件的优先级
            if len(inner_path) < 50:
                return 11
            # 否则返回 2
            else:
                return 2
        # 其他情况返回 0
        return 0
    # 添加任务更新，更新任务的优先级和可能的对等节点
    def addTaskUpdate(self, task, peer, priority=0):
        # 如果传入的优先级大于任务的优先级，则更新任务的优先级
        if priority > task["priority"]:
            self.tasks.updateItem(task, "priority", priority)
        # 如果有对等节点并且任务已经有对等节点，则将新的对等节点添加到任务的可能对等节点列表中
        if peer and task["peers"]:  
            task["peers"].append(peer)
            self.log.debug("Added peer %s to %s" % (peer.key, task["inner_path"]))
            self.startWorkers([peer], reason="Added new task (update received by peer)")
        # 如果有对等节点并且对等节点在任务的失败列表中，则将对等节点从失败列表中移除
        elif peer and peer in task["failed"]:
            task["failed"].remove(peer)  
            self.log.debug("Removed peer %s from failed %s" % (peer.key, task["inner_path"]))
            self.startWorkers([peer], reason="Added new task (peer failed before)")

    # 创建新任务并返回异步结果
    def addTask(self, inner_path, peer=None, priority=0, file_info=None):
        # 触发站点下载开始事件
        self.site.onFileStart(inner_path)  
        # 查找是否已经有针对该文件的任务
        task = self.tasks.findTask(inner_path)
        # 如果已经有任务，则调用添加任务更新方法
        if task:  
            self.addTaskUpdate(task, peer, priority)
        # 如果没有任务，则创建新任务
        else:  
            task = self.addTaskCreate(inner_path, peer, priority, file_info)
        return task

    # 添加任务工作者
    def addTaskWorker(self, task, worker):
        try:
            # 更新任务的工作者数量
            self.tasks.updateItem(task, "workers_num", task["workers_num"] + 1)
        except ValueError:
            task["workers_num"] += 1

    # 移除任务工作者
    def removeTaskWorker(self, task, worker):
        try:
            # 更新任务的工作者数量
            self.tasks.updateItem(task, "workers_num", task["workers_num"] - 1)
        except ValueError:
            task["workers_num"] -= 1
        # 如果失败的对等节点数量大于等于工作者数量，则标记任务为失败
        if len(task["failed"]) >= len(self.workers):
            fail_reason = "Too many fails: %s (workers: %s)" % (len(task["failed"]), len(self.workers))
            self.failTask(task, reason=fail_reason)

    # 等待其他任务
    # 检查任务是否完成
    def checkComplete(self):
        # 等待0.1秒
        time.sleep(0.1)
        # 如果没有任务了
        if not self.tasks:
            # 记录日志：检查完成，没有任务
            self.log.debug("Check complete: No tasks")
            # 调用完成方法
            self.onComplete()

    # 完成方法
    def onComplete(self):
        # 重置已开始的任务数量
        self.started_task_num = 0
        # 清空已询问的对等方列表
        del self.asked_peers[:]
        # 调用站点的完成方法，不再触发站点完成任务
        self.site.onComplete()

    # 标记任务已完成
    def doneTask(self, task):
        # 标记任务为已完成
        task["done"] = True
        # 从任务队列中移除
        self.tasks.remove(task)
        # 如果任务有可选的哈希ID
        if task["optional_hash_id"]:
            # 记录日志：在xx秒内下载了可选文件，添加到哈希字段
            self.log.debug(
                "Downloaded optional file in %.3fs, adding to hashfield: %s" %
                (time.time() - task["time_started"], task["inner_path"])
            )
            # 调用站点内容管理器的可选下载完成方法
            self.site.content_manager.optionalDownloaded(task["inner_path"], task["optional_hash_id"], task["size"])
        # 调用站点的文件完成方法
        self.site.onFileDone(task["inner_path"])
        # 设置事件为True
        task["evt"].set(True)
        # 如果没有任务了
        if not self.tasks:
            # 使用协程管理器创建一个新的协程，执行检查完成方法
            self.site.greenlet_manager.spawn(self.checkComplete)

    # 标记任务失败
    def failTask(self, task, reason="Unknown"):
        try:
            # 从任务队列中移除
            self.tasks.remove(task)
        except ValueError as err:
            return False

        # 记录日志：任务xx失败（原因：xx）
        self.log.debug("Task %s failed (Reason: %s)" % (task["inner_path"], reason))
        # 标记任务为已完成
        task["done"] = True
        # 调用站点的文件失败方法
        self.site.onFileFail(task["inner_path"])
        # 设置事件为False
        task["evt"].set(False)
        # 如果没有任务了
        if not self.tasks:
            # 使用协程管理器创建一个新的协程，执行检查完成方法
            self.site.greenlet_manager.spawn(self.checkComplete)
```
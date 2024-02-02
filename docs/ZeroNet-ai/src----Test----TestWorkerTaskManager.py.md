# `ZeroNet\src\Test\TestWorkerTaskManager.py`

```py
# 导入 pytest 模块
import pytest

# 从 Worker 模块中导入 WorkerTaskManager 类
from Worker import WorkerTaskManager
# 从当前目录中导入 Spy 模块

# 定义 TestUiWebsocket 类
class TestUiWebsocket:
    # 检查任务列表是否按照指定顺序排序
    def checkSort(self, tasks):
        # 将任务列表转换为列表并按照任务 ID 排序
        tasks_list = list(tasks)
        tasks_list.sort(key=lambda task: task["id"])
        # 断言排序后的列表与原列表不相等
        assert tasks_list != list(tasks)
        # 根据任务的优先级和工作者数量对列表进行排序
        tasks_list.sort(key=lambda task: (0 - (task["priority"] - task["workers_num"] * 10), task["id"]))
        # 断言排序后的列表与原列表相等
        assert tasks_list == list(tasks)

    # 测试简单的任务添加
    def testAppendSimple(self):
        # 创建 WorkerTaskManager 实例
        tasks = WorkerTaskManager.WorkerTaskManager()
        # 添加三个任务到任务列表中
        tasks.append({"id": 1, "priority": 15, "workers_num": 1, "inner_path": "file1.json"})
        tasks.append({"id": 2, "priority": 1, "workers_num": 0, "inner_path": "file2.json"})
        tasks.append({"id": 3, "priority": 8, "workers_num": 0, "inner_path": "file3.json"})
        # 断言任务列表中的文件路径与预期相等
        assert [task["inner_path"] for task in tasks] == ["file3.json", "file1.json", "file2.json"]

        # 调用 checkSort 方法检查任务列表是否按照指定顺序排序
        self.checkSort(tasks)

    # 测试添加大量任务
    def testAppendMany(self):
        # 创建 WorkerTaskManager 实例
        tasks = WorkerTaskManager.WorkerTaskManager()
        # 循环添加 1000 个任务到任务列表中
        for i in range(1000):
            tasks.append({"id": i, "priority": i % 20, "workers_num": i % 3, "inner_path": "file%s.json" % i})
        # 断言任务列表中第一个任务的文件路径为 "file39.json"
        assert tasks[0]["inner_path"] == "file39.json"
        # 断言任务列表中最后一个任务的文件路径为 "file980.json"
        assert tasks[-1]["inner_path"] == "file980.json"

        # 调用 checkSort 方法检查任务列表是否按照指定顺序排序
        self.checkSort(tasks)
    # 测试删除单个任务
    def testRemove(self):
        # 创建 WorkerTaskManager 实例
        tasks = WorkerTaskManager.WorkerTaskManager()
        # 循环添加1000个任务到任务管理器中
        for i in range(1000):
            tasks.append({"id": i, "priority": i % 20, "workers_num": i % 3, "inner_path": "file%s.json" % i})

        # 设置任务id为333
        i = 333
        # 创建任务对象
        task = {"id": i, "priority": i % 20, "workers_num": i % 3, "inner_path": "file%s.json" % i}
        # 断言任务在任务管理器中
        assert task in tasks

        # 使用 Spy 对象监视 indexSlow 方法的调用次数
        with Spy.Spy(tasks, "indexSlow") as calls:
            # 移除任务
            tasks.remove(task)
            # 断言 indexSlow 方法未被调用
            assert len(calls) == 0

        # 断言任务不在任务管理器中
        assert task not in tasks

        # 移除不存在的任务
        with Spy.Spy(tasks, "indexSlow") as calls:
            # 断言抛出 ValueError 异常
            with pytest.raises(ValueError):
                tasks.remove(task)
            # 断言 indexSlow 方法未被调用
            assert len(calls) == 0

        # 检查任务排序
        self.checkSort(tasks)

    # 测试移除所有任务
    def testRemoveAll(self):
        # 创建 WorkerTaskManager 实例
        tasks = WorkerTaskManager.WorkerTaskManager()
        # 创建任务列表
        tasks_list = []
        # 循环添加1000个任务到任务管理器中，并添加到任务列表中
        for i in range(1000):
            task = {"id": i, "priority": i % 20, "workers_num": i % 3, "inner_path": "file%s.json" % i}
            tasks.append(task)
            tasks_list.append(task)

        # 循环移除任务列表中的任务
        for task in tasks_list:
            tasks.remove(task)

        # 断言任务内部路径长度为0
        assert len(tasks.inner_paths) == 0
        # 断言任务管理器长度为0
        assert len(tasks) == 0
    # 定义测试用例，测试任务修改功能
    def testModify(self):
        # 创建 WorkerTaskManager 对象
        tasks = WorkerTaskManager.WorkerTaskManager()
        # 循环添加1000个任务
        for i in range(1000):
            tasks.append({"id": i, "priority": i % 20, "workers_num": i % 3, "inner_path": "file%s.json" % i})

        # 获取第333个任务
        task = tasks[333]
        # 修改任务的优先级
        task["priority"] += 10

        # 使用 pytest 断言捕获 Assertion Error 异常
        with pytest.raises(AssertionError):
            self.checkSort(tasks)

        # 使用 Spy 对象监视 indexSlow 方法的调用
        with Spy.Spy(tasks, "indexSlow") as calls:
            # 更新任务
            tasks.updateItem(task)
            # 断言 indexSlow 方法被调用一次
            assert len(calls) == 1

        # 断言任务在任务列表中
        assert task in tasks

        # 检查任务排序
        self.checkSort(tasks)

        # 检查重新排序优化
        with Spy.Spy(tasks, "indexSlow") as calls:
            # 更新任务的优先级
            tasks.updateItem(task, "priority", task["priority"] + 10)
            # 断言 indexSlow 方法未被调用
            assert len(calls) == 0

        with Spy.Spy(tasks, "indexSlow") as calls:
            # 更新任务的优先级
            tasks.updateItem(task, "priority", task["workers_num"] - 1)
            # 断言 indexSlow 方法未被调用
            assert len(calls) == 0

        # 再次检查任务排序
        self.checkSort(tasks)

    # 定义测试用例，测试相同优先级的任务修改功能
    def testModifySamePriority(self):
        # 创建 WorkerTaskManager 对象
        tasks = WorkerTaskManager.WorkerTaskManager()
        # 循环添加1000个相同优先级的任务
        for i in range(1000):
            tasks.append({"id": i, "priority": 10, "workers_num": 5, "inner_path": "file%s.json" % i})

        # 获取第333个任务
        task = tasks[333]

        # 检查重新排序优化
        with Spy.Spy(tasks, "indexSlow") as calls:
            # 更新任务的优先级
            tasks.updateItem(task, "priority", task["workers_num"] - 1)
            # 断言 indexSlow 方法未被调用
            assert len(calls) == 0

    # 定义测试用例，测试任务是否存在于任务列表中
    def testIn(self):
        # 创建 WorkerTaskManager 对象
        tasks = WorkerTaskManager.WorkerTaskManager()

        # 创建任务
        i = 1
        task = {"id": i, "priority": i % 20, "workers_num": i % 3, "inner_path": "file%s.json" % i}

        # 使用断言判断任务是否不在任务列表中
        assert task not in tasks
    # 定义一个测试用例，测试任务管理器中查找任务的功能
    def testFindTask(self):
        # 创建一个工作任务管理器对象
        tasks = WorkerTaskManager.WorkerTaskManager()
        # 循环添加1000个任务到任务管理器中
        for i in range(1000):
            tasks.append({"id": i, "priority": i % 20, "workers_num": i % 3, "inner_path": "file%s.json" % i})
    
        # 断言查找文件"file999.json"是否存在
        assert tasks.findTask("file999.json")
        # 断言查找文件"file-unknown.json"是否不存在
        assert not tasks.findTask("file-unknown.json")
        # 移除文件"file999.json"对应的任务
        tasks.remove(tasks.findTask("file999.json"))
        # 断言查找文件"file999.json"是否不存在
        assert not tasks.findTask("file999.json")
```
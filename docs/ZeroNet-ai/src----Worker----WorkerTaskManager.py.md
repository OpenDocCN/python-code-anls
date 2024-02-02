# `ZeroNet\src\Worker\WorkerTaskManager.py`

```py
# 导入 bisect 模块，用于插入排序
import bisect
# 导入 MutableSequence 抽象基类，用于创建自定义排序列表
from collections.abc import MutableSequence

# 创建自定义排序列表类 CustomSortedList
class CustomSortedList(MutableSequence):
    # 初始化方法
    def __init__(self):
        super().__init__()
        # 初始化列表 items，每个元素为 (priority, added index, actual value)
        self.items = []
        # 初始化日志记录标志
        self.logging = False

    # 返回对象的字符串表示形式
    def __repr__(self):
        return "<{0} {1}>".format(self.__class__.__name__, self.items)

    # 返回列表的长度
    def __len__(self):
        return len(self.items)

    # 获取指定索引处的元素
    def __getitem__(self, index):
        if type(index) is int:
            return self.items[index][2]
        else:
            return [item[2] for item in self.items[index]]

    # 删除指定索引处的元素
    def __delitem__(self, index):
        del self.items[index]

    # 设置指定索引处的元素
    def __setitem__(self, index, value):
        self.items[index] = self.valueToItem(value)

    # 返回对象的字符串表示形式
    def __str__(self):
        return str(self[:])

    # 在指定索引处插入元素
    def insert(self, index, value):
        self.append(value)

    # 在列表末尾添加元素
    def append(self, value):
        # 使用 bisect 模块的 insort 方法按顺序插入元素
        bisect.insort(self.items, self.valueToItem(value))

    # 更新元素的值
    def updateItem(self, value, update_key=None, update_value=None):
        # 移除指定元素
        self.remove(value)
        # 如果指定了更新键和更新值，则更新元素
        if update_key is not None:
            value[update_key] = update_value
        # 添加更新后的元素
        self.append(value)

    # 禁止对列表进行排序
    def sort(self, *args, **kwargs):
        raise Exception("Sorted list can't be sorted")

    # 将值转换为列表项
    def valueToItem(self, value):
        return (self.getPriority(value), self.getId(value), value)

    # 获取值的优先级
    def getPriority(self, value):
        return value

    # 获取值的唯一标识
    def getId(self, value):
        return id(value)

    # 使用线性搜索方式查找元素的索引
    def indexSlow(self, value):
        for pos, item in enumerate(self.items):
            if item[2] == value:
                return pos
        return None
    # 定义一个方法，用于查找给定值在优先级队列中的索引位置
    def index(self, value):
        # 创建一个元组，包含值的优先级、ID和值本身
        item = (self.getPriority(value), self.getId(value), value)
        # 使用二分查找算法找到元素应该插入的位置
        bisect_pos = bisect.bisect(self.items, item) - 1
        # 如果找到的位置大于等于0且该位置的值与要查找的值相等，则直接返回该位置
        if bisect_pos >= 0 and self.items[bisect_pos][2] == value:
            return bisect_pos

        # 如果找到的位置小于0或者找到的值与要查找的值不相等，则切换到慢速迭代方式进行查找
        pos = self.indexSlow(value)

        # 如果开启了日志记录，则打印慢速迭代方式查找的信息
        if self.logging:
            print("Slow index for %s in pos %s bisect: %s" % (item[2], pos, bisect_pos))

        # 如果慢速迭代方式未找到值，则抛出值错误异常
        if pos is None:
            raise ValueError("%r not in list" % value)
        else:
            return pos

    # 定义一个方法，用于判断给定值是否存在于优先级队列中
    def __contains__(self, value):
        try:
            # 调用index方法查找给定值，如果成功找到则返回True
            self.index(value)
            return True
        except ValueError:
            # 如果index方法抛出值错误异常，则返回False
            return False
# 定义一个 WorkerTaskManager 类，继承自 CustomSortedList 类
class WorkerTaskManager(CustomSortedList):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个内部路径的字典
        self.inner_paths = {}

    # 获取任务的优先级
    def getPriority(self, value):
        # 返回优先级的计算结果
        return 0 - (value["priority"] - value["workers_num"] * 10)

    # 获取任务的 ID
    def getId(self, value):
        # 返回任务的 ID
        return value["id"]

    # 判断任务是否存在
    def __contains__(self, value):
        # 判断任务的内部路径是否在内部路径字典中
        return value["inner_path"] in self.inner_paths

    # 删除指定索引的任务
    def __delitem__(self, index):
        # 从内部路径缓存中删除任务
        del self.inner_paths[self.items[index][2]["inner_path"]]
        # 调用父类的删除方法
        super().__delitem__(index)

    # 通过内部路径快速查找任务

    # 添加任务
    def append(self, task):
        # 如果任务的内部路径已经存在于内部路径字典中，则抛出数值错误
        if task["inner_path"] in self.inner_paths:
            raise ValueError("File %s already has a task" % task["inner_path"])
        # 调用父类的添加方法
        super().append(task)
        # 为了通过文件名快速查找任务，创建内部路径缓存
        self.inner_paths[task["inner_path"]] = task

    # 移除任务
    def remove(self, task):
        # 如果任务不在列表中，则抛出数值错误
        if task not in self:
            raise ValueError("%r not in list" % task)
        else:
            # 调用父类的移除方法
            super().remove(task)

    # 通过内部路径查找任务
    def findTask(self, inner_path):
        # 返回内部路径对应的任务，如果不存在则返回 None
        return self.inner_paths.get(inner_path, None)
```
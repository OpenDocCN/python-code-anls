# `.\DB-GPT-src\dbgpt\util\custom_data_structure.py`

```py
# 导入需要的模块：OrderedDict（有序字典）和 deque（双端队列）
from collections import OrderedDict, deque

# 创建一个继承自 OrderedDict 的类 FixedSizeDict，用于存储固定大小的键值对
class FixedSizeDict(OrderedDict):
    # 初始化方法，接受最大容量参数 max_size
    def __init__(self, max_size):
        # 调用父类 OrderedDict 的初始化方法
        super().__init__()
        # 设置最大容量属性
        self.max_size = max_size

    # 重写 __setitem__ 方法，在设置新的键值对时进行处理
    def __setitem__(self, key, value):
        # 如果当前字典的长度已经达到最大容量
        if len(self) >= self.max_size:
            # 弹出最早添加的键值对（最老的键值对），即队列中的第一个元素
            self.popitem(last=False)
        # 调用父类 OrderedDict 的设置键值对方法
        super().__setitem__(key, value)


# 创建一个固定大小的列表类 FixedSizeList，使用 deque 来实现
class FixedSizeList:
    # 初始化方法，接受最大容量参数 max_size
    def __init__(self, max_size):
        # 设置最大容量属性
        self.max_size = max_size
        # 使用 deque 来存储列表，并限制其最大长度为 max_size
        self.list = deque(maxlen=max_size)

    # 添加新元素到列表尾部的方法
    def append(self, value):
        self.list.append(value)

    # 获取指定索引位置的元素的方法
    def __getitem__(self, index):
        return self.list[index]

    # 设置指定索引位置的元素的方法
    def __setitem__(self, index, value):
        self.list[index] = value

    # 返回当前列表的长度
    def __len__(self):
        return len(self.list)

    # 返回当前列表的字符串表示形式
    def __str__(self):
        return str(list(self.list))
```
# `.\pytorch\aten\src\ATen\function_wrapper.py`

```
# 定义一个类名为 Node 的节点类，用于构建链表数据结构
class Node:
    # 类的初始化方法，用于设置节点的初始值和指向下一个节点的指针
    def __init__(self, data=None):
        # 设置节点的数据
        self.data = data
        # 设置指向下一个节点的指针，默认为 None，表示末尾节点
        self.next = None

# 定义一个名为 LinkedList 的链表类，用于管理和操作链表
class LinkedList:
    # 类的初始化方法，用于创建一个空链表，设置链表的头节点和尾节点
    def __init__(self):
        # 初始时链表为空，头节点和尾节点都为 None
        self.head = None
        self.tail = None
    
    # 向链表尾部添加一个新节点
    def append(self, data):
        # 创建一个新的节点对象
        new_node = Node(data)
        # 如果链表为空，设置新节点为头节点和尾节点
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            # 否则将新节点连接到链表的尾部，更新尾节点为新节点
            self.tail.next = new_node
            self.tail = new_node
    
    # 在链表头部插入一个新节点
    def prepend(self, data):
        # 创建一个新的节点对象
        new_node = Node(data)
        # 如果链表为空，设置新节点为头节点和尾节点
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            # 否则将新节点插入到链表头部，更新头节点为新节点
            new_node.next = self.head
            self.head = new_node
    
    # 打印链表的所有节点数据，用于调试和查看链表结构
    def print_list(self):
        # 从头节点开始遍历链表
        current = self.head
        while current:
            # 打印当前节点的数据
            print(current.data)
            # 移动到下一个节点
            current = current.next
```
# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_isolve\tests\__init__.py`

```
# 定义一个类名为 Node 的节点类，用于构建树形数据结构
class Node:
    # 类初始化方法，接受一个参数 value 作为节点的值，并初始化节点的左右子节点为 None
    def __init__(self, value):
        self.value = value  # 设置节点的值
        self.left = None    # 初始化节点的左子节点为 None
        self.right = None   # 初始化节点的右子节点为 None

# 创建一个名为 root 的 Node 对象，其节点值为 5
root = Node(5)
# 创建一个名为 node1 的 Node 对象，其节点值为 3
node1 = Node(3)
# 将 node1 设置为 root 的左子节点
root.left = node1
# 创建一个名为 node2 的 Node 对象，其节点值为 7
node2 = Node(7)
# 将 node2 设置为 root 的右子节点
root.right = node2
```
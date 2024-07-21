# `.\pytorch\tools\linter\clang_tidy\__init__.py`

```
# 定义一个类`Node`，表示图的节点
class Node:
    # 类级别变量`count`，用于记录节点个数
    count = 0
    
    # 初始化方法，每次创建节点时增加`count`，设置节点ID和标签
    def __init__(self, label):
        Node.count += 1  # 增加节点计数
        self.id = Node.count  # 设置节点ID
        self.label = label  # 设置节点标签
    
    # 方法`get_id`，返回节点的ID
    def get_id(self):
        return self.id
    
    # 方法`get_label`，返回节点的标签
    def get_label(self):
        return self.label

# 创建一个名为`node1`的Node对象，标签为'A'
node1 = Node('A')

# 创建一个名为`node2`的Node对象，标签为'B'
node2 = Node('B')

# 输出节点1的ID和标签
print(node1.get_id(), node1.get_label())

# 输出节点2的ID和标签
print(node2.get_id(), node2.get_label())
```
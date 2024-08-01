# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\5593.5c276434e1162c59.js`

```py
# 导入必要的模块：os（操作系统相关功能）、sys（系统参数和功能）、json（处理 JSON 数据）
import os, sys, json

# 定义一个类`TreeNode`，表示树节点
class TreeNode:
    # 初始化方法，接收参数`value`作为节点值，并初始化空列表作为子节点
    def __init__(self, value):
        self.value = value
        self.children = []

    # 添加子节点的方法，将给定的节点添加到当前节点的子节点列表中
    def add_child(self, child_node):
        self.children.append(child_node)

# 创建一个根节点，值为`"root"`
root = TreeNode("root")

# 创建三个子节点，并添加到根节点中
child1 = TreeNode("child1")
child2 = TreeNode("child2")
child3 = TreeNode("child3")
root.add_child(child1)
root.add_child(child2)
root.add_child(child3)

# 将整个树结构转换为 JSON 格式的字符串，并打印输出
print(json.dumps(root, default=lambda o: o.__dict__, indent=2))
```
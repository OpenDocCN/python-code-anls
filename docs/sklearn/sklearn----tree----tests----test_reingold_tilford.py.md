# `D:\src\scipysrc\scikit-learn\sklearn\tree\tests\test_reingold_tilford.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试

from sklearn.tree._reingold_tilford import Tree, buchheim  # 从sklearn.tree._reingold_tilford模块导入Tree类和buchheim函数

# 创建简单的树结构
simple_tree = Tree("", 0, Tree("", 1), Tree("", 2))

# 创建更复杂的树结构
bigger_tree = Tree(
    "",
    0,
    Tree(
        "",
        1,
        Tree("", 3),
        Tree("", 4, Tree("", 7), Tree("", 8)),
    ),
    Tree("", 2, Tree("", 5), Tree("", 6)),
)

# 使用pytest的参数化测试，定义两组参数：简单树和其节点数，复杂树和其节点数
@pytest.mark.parametrize("tree, n_nodes", [(simple_tree, 3), (bigger_tree, 9)])
def test_buchheim(tree, n_nodes):
    # 定义内部函数walk_tree，用于遍历树节点并验证布局结果
    def walk_tree(draw_tree):
        res = [(draw_tree.x, draw_tree.y)]  # 将当前节点的坐标加入结果列表
        for child in draw_tree.children:
            # 对每个子节点，验证其y坐标比父节点高1
            assert child.y == draw_tree.y + 1
            res.extend(walk_tree(child))  # 递归遍历子节点
        if len(draw_tree.children):
            # 对于有子节点的父节点，验证其x坐标为子节点x坐标的中点
            assert (
                draw_tree.x == (draw_tree.children[0].x + draw_tree.children[1].x) / 2
            )
        return res  # 返回所有节点的坐标列表

    layout = buchheim(tree)  # 使用布赖姆算法计算树的布局
    coordinates = walk_tree(layout)  # 获取树的所有节点坐标
    assert len(coordinates) == n_nodes  # 验证节点数是否符合预期

    # 验证每个深度/层级的节点x坐标唯一性
    depth = 0
    while True:
        x_at_this_depth = [node[0] for node in coordinates if node[1] == depth]
        if not x_at_this_depth:
            # 已经遍历到所有叶子节点
            break
        assert len(np.unique(x_at_this_depth)) == len(x_at_this_depth)
        depth += 1
```
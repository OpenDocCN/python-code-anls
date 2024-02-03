# `numpy-ml\numpy_ml\tests\test_trees.py`

```
# 禁用 flake8 检查
import numpy as np

# 导入随机森林分类器和回归器
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# 导入决策树分类器和回归器
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# 导入准确率评估指标和均方误差评估指标
from sklearn.metrics import accuracy_score, mean_squared_error
# 导入生成回归数据和分类数据的函数
from sklearn.datasets import make_regression, make_blobs
# 导入数据集划分函数
from sklearn.model_selection import train_test_split

# 导入梯度提升决策树类
from numpy_ml.trees.gbdt import GradientBoostedDecisionTree
# 导入决策树类、节点类和叶子节点类
from numpy_ml.trees.dt import DecisionTree, Node, Leaf
# 导入随机森林类
from numpy_ml.trees.rf import RandomForest
# 导入随机生成张量的测试函数
from numpy_ml.utils.testing import random_tensor

# 克隆决策树
def clone_tree(dtree):
    # 获取决策树的左子节点、右子节点、特征、阈值和值
    children_left = dtree.tree_.children_left
    children_right = dtree.tree_.children_right
    feature = dtree.tree_.feature
    threshold = dtree.tree_.threshold
    values = dtree.tree_.value

    # 递归构建决策树
    def grow(node_id):
        l, r = children_left[node_id], children_right[node_id]
        if l == r:
            return Leaf(values[node_id].argmax())
        n = Node(None, None, (feature[node_id], threshold[node_id]))
        n.left = grow(l)
        n.right = grow(r)
        return n

    node_id = 0
    root = Node(None, None, (feature[node_id], threshold[node_id]))
    root.left = grow(children_left[node_id])
    root.right = grow(children_right[node_id])
    return root

# 比较两棵树的结构和值
def compare_trees(mine, gold):
    # 克隆金标准决策树
    clone = clone_tree(gold)
    mine = mine.root

    # 递归测试两棵树的结构和值是否相等
    def test(mine, clone):
        if isinstance(clone, Node) and isinstance(mine, Node):
            assert mine.feature == clone.feature, "Node {} not equal".format(depth)
            np.testing.assert_allclose(mine.threshold, clone.threshold)
            test(mine.left, clone.left, depth + 1)
            test(mine.right, clone.right, depth + 1)
        elif isinstance(clone, Leaf) and isinstance(mine, Leaf):
            np.testing.assert_allclose(mine.value, clone.value)
            return
        else:
            raise ValueError("Nodes at depth {} are not equal".format(depth))

    depth = 0
    ok = True
    # 当 ok 为真时进入循环
    while ok:
        # 如果 clone 和 mine 都是 Node 类型的实例
        if isinstance(clone, Node) and isinstance(mine, Node):
            # 断言 mine 的特征与 clone 的特征相等
            assert mine.feature == clone.feature
            # 使用 np.testing.assert_allclose 检查 mine 的阈值与 clone 的阈值是否相等
            np.testing.assert_allclose(mine.threshold, clone.threshold)
            # 递归调用 test 函数，比较 mine 的左子节点和 clone 的左子节点
            test(mine.left, clone.left, depth + 1)
            # 递归调用 test 函数，比较 mine 的右子节点和 clone 的右子节点
            test(mine.right, clone.right, depth + 1)
        # 如果 clone 和 mine 都是 Leaf 类型的实例
        elif isinstance(clone, Leaf) and isinstance(mine, Leaf):
            # 使用 np.testing.assert_allclose 检查 mine 的值与 clone 的值是否相等
            np.testing.assert_allclose(mine.value, clone.value)
            # 返回，结束当前递归
            return
        # 如果 clone 和 mine 不是相同类型的节点
        else:
            # 抛出 ValueError 异常，提示深度为 depth 的节点不相等
            raise ValueError("Nodes at depth {} are not equal".format(depth))
# 测试决策树模型，N默认为1
def test_DecisionTree(N=1):
    # 初始化i为1
    i = 1
    # 设置随机种子为12345
    np.random.seed(12345)

# 测试随机森林模型，N默认为1
def test_RandomForest(N=1):
    # 设置随机种子为12345
    np.random.seed(12345)
    # 初始化i为1
    i = 1

# 测试梯度提升决策树模型，N默认为1
def test_gbdt(N=1):
    # 设置随机种子为12345
    np.random.seed(12345)
    # 初始化i为1
    i = 1
```
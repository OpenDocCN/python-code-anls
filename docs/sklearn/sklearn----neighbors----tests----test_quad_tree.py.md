# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\tests\test_quad_tree.py`

```
# 导入pickle模块，用于序列化和反序列化Python对象
import pickle

# 导入numpy库，并将其命名为np，用于科学计算
import numpy as np

# 导入pytest库，用于编写和运行测试
import pytest

# 从sklearn.neighbors._quad_tree模块中导入_QuadTree类，用于测试
from sklearn.neighbors._quad_tree import _QuadTree

# 从sklearn.utils中导入check_random_state函数，用于生成随机数种子
from sklearn.utils import check_random_state


# 定义测试函数，用于测试QuadTree类的边界计算功能
def test_quadtree_boundary_computation():
    # 创建空列表Xs，用于存储不同的测试数据集
    Xs = []

    # 添加一个随机测试用例到Xs列表中
    Xs.append(np.array([[-1, 1], [-4, -1]], dtype=np.float32))

    # 添加一个所有元素为0的测试用例到Xs列表中
    Xs.append(np.array([[0, 0], [0, 0]], dtype=np.float32))

    # 添加一个所有元素为负数的测试用例到Xs列表中
    Xs.append(np.array([[-1, -2], [-4, 0]], dtype=np.float32))

    # 添加一个所有元素接近零的测试用例到Xs列表中
    Xs.append(np.array([[-1e-6, 1e-6], [-4e-6, -1e-6]], dtype=np.float32))

    # 遍历Xs列表中的每个测试数据集
    for X in Xs:
        # 创建一个QuadTree对象，设定维度为2，关闭详细输出
        tree = _QuadTree(n_dimensions=2, verbose=0)
        
        # 构建QuadTree，传入当前的测试数据集X
        tree.build_tree(X)
        
        # 检查QuadTree的一致性
        tree._check_coherence()


# 定义测试函数，用于测试QuadTree类中处理相似点的情况
def test_quadtree_similar_point():
    # 创建空列表Xs，用于存储不同的测试数据集
    Xs = []

    # 添加一个元素不同的测试用例到Xs列表中
    Xs.append(np.array([[1, 2], [3, 4]], dtype=np.float32))

    # 添加一个在X轴上相同的测试用例到Xs列表中
    Xs.append(np.array([[1.0, 2.0], [1.0, 3.0]], dtype=np.float32))

    # 添加一个在X轴上任意接近的测试用例到Xs列表中
    Xs.append(np.array([[1.00001, 2.0], [1.00002, 3.0]], dtype=np.float32))

    # 添加一个在Y轴上相同的测试用例到Xs列表中
    Xs.append(np.array([[1.0, 2.0], [3.0, 2.0]], dtype=np.float32))

    # 添加一个在Y轴上任意接近的测试用例到Xs列表中
    Xs.append(np.array([[1.0, 2.00001], [3.0, 2.00002]], dtype=np.float32))

    # 添加一个在X轴和Y轴上任意接近的测试用例到Xs列表中
    Xs.append(np.array([[1.00001, 2.00001], [1.00002, 2.00002]], dtype=np.float32))

    # 添加一个在X轴接近机器epsilon的测试用例到Xs列表中
    Xs.append(np.array([[1, 0.0003817754041], [2, 0.0003817753750]], dtype=np.float32))

    # 添加一个在Y轴接近机器epsilon的测试用例到Xs列表中
    Xs.append(np.array([[0.0003817754041, 1.0], [0.0003817753750, 2.0]], dtype=np.float32))

    # 遍历Xs列表中的每个测试数据集
    for X in Xs:
        # 创建一个QuadTree对象，设定维度为2，关闭详细输出
        tree = _QuadTree(n_dimensions=2, verbose=0)
        
        # 构建QuadTree，传入当前的测试数据集X
        tree.build_tree(X)
        
        # 检查QuadTree的一致性
        tree._check_coherence()


# 使用pytest的参数化装饰器，指定不同的参数组合进行测试
@pytest.mark.parametrize("n_dimensions", (2, 3))
@pytest.mark.parametrize("protocol", (0, 1, 2))
def test_quad_tree_pickle(n_dimensions, protocol):
    # 使用check_random_state函数创建一个随机数生成器rng，种子为0
    rng = check_random_state(0)

    # 生成一个形状为(10, n_dimensions)的随机数据集X
    X = rng.random_sample((10, n_dimensions))

    # 创建一个QuadTree对象，设定维度为n_dimensions，关闭详细输出
    tree = _QuadTree(n_dimensions=n_dimensions, verbose=0)
    
    # 构建QuadTree，传入随机数据集X
    tree.build_tree(X)

    # 使用pickle模块将QuadTree对象tree序列化，指定协议为protocol，并保存到变量s中
    s = pickle.dumps(tree, protocol=protocol)
    
    # 使用pickle模块从序列化的数据s中反序列化，重新构建QuadTree对象bt2
    bt2 = pickle.loads(s)

    # 遍历随机数据集X中的每个数据点x
    for x in X:
        # 获取原始QuadTree对象tree中点x所在的叶子节点cell_x_tree
        cell_x_tree = tree.get_cell(x)
        
        # 获取反序列化后QuadTree对象bt2中点x所在的叶子节点cell_x_bt2
        cell_x_bt2 = bt2.get_cell(x)
        
        # 断言：原始QuadTree和反序列化后的QuadTree中点x所在的叶子节点应相等
        assert cell_x_tree == cell_x_bt2


# 使用pytest的参数化装饰器，指定不同的维度n_dimensions进行测试
@pytest.mark.parametrize("n_dimensions", (2, 3))
def test_qt_insert_duplicate(n_dimensions):
    # 使用种子值0初始化随机数生成器
    rng = check_random_state(0)
    
    # 生成一个形状为(10, n_dimensions)的随机数组X
    X = rng.random_sample((10, n_dimensions))
    
    # 将数组X与其前5行合并成Xd，形成一个包含重复行的数组
    Xd = np.r_[X, X[:5]]
    
    # 创建一个QuadTree对象tree，设置维度为n_dimensions，关闭详细输出
    tree = _QuadTree(n_dimensions=n_dimensions, verbose=0)
    
    # 构建QuadTree，使用合并后的数组Xd作为输入数据
    tree.build_tree(Xd)
    
    # 获取构建好的QuadTree的累积尺寸信息
    cumulative_size = tree.cumulative_size
    
    # 获取QuadTree的叶子节点信息
    leafs = tree.leafs
    
    # 断言前5个节点是重复的，并且后面的节点是单个点的叶子节点
    for i, x in enumerate(X):
        # 获取点x所在的单元格ID
        cell_id = tree.get_cell(x)
        # 断言该单元格是叶子节点
        assert leafs[cell_id]
        # 断言该单元格的累积尺寸等于1加上是否小于5的布尔值（前5个节点为重复）
        assert cumulative_size[cell_id] == 1 + (i < 5)
# 定义一个测试函数，用于检验四叉树的 summarize 方法
def test_summarize():
    # 设置角度为 0.9
    angle = 0.9
    # 创建一个包含四个点的二维数组 X，数据类型为浮点数
    X = np.array(
        [[-10.0, -10.0], [9.0, 10.0], [10.0, 9.0], [10.0, 10.0]], dtype=np.float32
    )
    # 将查询点设为 X 中的第一个点
    query_pt = X[0, :]
    # 计算 X 的维度数
    n_dimensions = X.shape[1]
    # 设置偏移量为维度数加上2
    offset = n_dimensions + 2

    # 创建一个 _QuadTree 类的对象 qt，不输出详细信息
    qt = _QuadTree(n_dimensions, verbose=0)
    # 构建四叉树，基于数组 X
    qt.build_tree(X)

    # 调用四叉树的 _py_summarize 方法，获取查询点的索引和摘要信息
    idx, summary = qt._py_summarize(query_pt, X, angle)

    # 从摘要中获取节点距离信息
    node_dist = summary[n_dimensions]
    # 从摘要中获取节点大小信息
    node_size = summary[n_dimensions + 1]

    # 计算查询点到 X[1:] 质心的距离平方和
    barycenter = X[1:].mean(axis=0)
    ds2c = ((X[0] - barycenter) ** 2).sum()

    # 断言查询点的索引应等于偏移量
    assert idx == offset
    # 断言节点大小应为3，如果不是则抛出异常
    assert node_size == 3, "summary size = {}".format(node_size)
    # 断言节点距离应接近计算得到的距离平方和
    assert np.isclose(node_dist, ds2c)

    # 对于角度为0时，摘要应包含所有3个节点，每个节点大小为1，距离分别为 X[1:] 中每个点到查询点的距离
    idx, summary = qt._py_summarize(query_pt, X, 0.0)
    barycenter = X[1:].mean(axis=0)
    ds2c = ((X[0] - barycenter) ** 2).sum()

    # 断言查询点的索引应为3倍的偏移量
    assert idx == 3 * (offset)
    # 循环遍历每个节点的摘要信息
    for i in range(3):
        # 获取每个节点的距离信息
        node_dist = summary[i * offset + n_dimensions]
        # 获取每个节点的大小信息
        node_size = summary[i * offset + n_dimensions + 1]

        # 计算查询点到 X[i+1] 的距离平方和
        ds2c = ((X[0] - X[i + 1]) ** 2).sum()

        # 断言节点大小应为1，如果不是则抛出异常
        assert node_size == 1, "summary size = {}".format(node_size)
        # 断言节点距离应接近计算得到的距离平方和
        assert np.isclose(node_dist, ds2c)
```
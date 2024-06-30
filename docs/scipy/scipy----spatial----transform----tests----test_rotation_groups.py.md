# `D:\src\scipysrc\scipy\scipy\spatial\transform\tests\test_rotation_groups.py`

```
import pytest  # 导入 pytest 库

import numpy as np  # 导入 numpy 库，并使用 np 作为别名
from numpy.testing import assert_array_almost_equal  # 导入 numpy.testing 中的 assert_array_almost_equal 函数
from scipy.spatial.transform import Rotation  # 导入 scipy.spatial.transform 中的 Rotation 类
from scipy.optimize import linear_sum_assignment  # 导入 scipy.optimize 中的 linear_sum_assignment 函数
from scipy.spatial.distance import cdist  # 导入 scipy.spatial.distance 中的 cdist 函数
from scipy.constants import golden as phi  # 导入 scipy.constants 中的 golden 常量，并使用 phi 作为别名
from scipy.spatial import cKDTree  # 导入 scipy.spatial 中的 cKDTree 类


TOL = 1E-12  # 定义精度阈值 TOL

NS = range(1, 13)  # 创建一个包含数字 1 到 12 的范围对象 NS
NAMES = ["I", "O", "T"] + ["C%d" % n for n in NS] + ["D%d" % n for n in NS]  # 创建一个包含字符串的列表 NAMES
SIZES = [60, 24, 12] + list(NS) + [2 * n for n in NS]  # 创建一个包含整数的列表 SIZES


def _calculate_rmsd(P, Q):
    """计算点集 P 和 Q 的均方根距离（RMSD）。
    距离被取为所有可能匹配中的最小值。如果 P 和 Q 相同则距离为零，否则为非零。
    """
    distance_matrix = cdist(P, Q, metric='sqeuclidean')  # 计算 P 和 Q 之间的距离矩阵
    matching = linear_sum_assignment(distance_matrix)  # 使用线性求和分配算法找到最佳匹配
    return np.sqrt(distance_matrix[matching].sum())  # 返回最小匹配距离的均方根值


def _generate_pyramid(n, axis):
    """生成一个具有 n 个面和指定轴的金字塔形状。
    """
    thetas = np.linspace(0, 2 * np.pi, n + 1)[:-1]  # 在 [0, 2π] 区间生成 n 个等间距角度
    P = np.vstack([np.zeros(n), np.cos(thetas), np.sin(thetas)]).T  # 生成底部 n 边形的顶点坐标
    P = np.concatenate((P, [[1, 0, 0]]))  # 添加金字塔的顶点坐标
    return np.roll(P, axis, axis=1)  # 按指定轴滚动顶点坐标数组


def _generate_prism(n, axis):
    """生成一个具有 n 个面和指定轴的棱柱形状。
    """
    thetas = np.linspace(0, 2 * np.pi, n + 1)[:-1]  # 在 [0, 2π] 区间生成 n 个等间距角度
    bottom = np.vstack([-np.ones(n), np.cos(thetas), np.sin(thetas)]).T  # 生成底部 n 边形的底面顶点坐标
    top = np.vstack([+np.ones(n), np.cos(thetas), np.sin(thetas)]).T  # 生成底部 n 边形的顶面顶点坐标
    P = np.concatenate((bottom, top))  # 组合底部和顶部的顶点坐标
    return np.roll(P, axis, axis=1)  # 按指定轴滚动顶点坐标数组


def _generate_icosahedron():
    """生成一个二十面体的顶点坐标。
    """
    x = np.array([[0, -1, -phi],
                  [0, -1, +phi],
                  [0, +1, -phi],
                  [0, +1, +phi]])
    return np.concatenate([np.roll(x, i, axis=1) for i in range(3)])


def _generate_octahedron():
    """生成一个八面体的顶点坐标。
    """
    return np.array([[-1, 0, 0], [+1, 0, 0], [0, -1, 0],
                     [0, +1, 0], [0, 0, -1], [0, 0, +1]])


def _generate_tetrahedron():
    """生成一个四面体的顶点坐标。
    """
    return np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])


@pytest.mark.parametrize("name", [-1, None, True, np.array(['C3'])])
def test_group_type(name):
    """测试 Rotation.create_group 函数对于不合法的群类型名称的行为。
    应该抛出 ValueError 异常并包含特定错误信息。
    """
    with pytest.raises(ValueError,
                       match="must be a string"):
        Rotation.create_group(name)


@pytest.mark.parametrize("name", ["Q", " ", "CA", "C ", "DA", "D ", "I2", ""])
def test_group_name(name):
    """测试 Rotation.create_group 函数对于不合法的群名称的行为。
    应该抛出 ValueError 异常并包含特定错误信息。
    """
    with pytest.raises(ValueError,
                       match="must be one of 'I', 'O', 'T', 'Dn', 'Cn'"):
        Rotation.create_group(name)


@pytest.mark.parametrize("name", ["C0", "D0"])
def test_group_order_positive(name):
    """测试 Rotation.create_group 函数对于不合法的群阶数的行为。
    应该抛出 ValueError 异常并包含特定错误信息。
    """
    with pytest.raises(ValueError,
                       match="Group order must be positive"):
        Rotation.create_group(name)


@pytest.mark.parametrize("axis", ['A', 'b', 0, 1, 2, 4, False, None])
def test_axis_valid(axis):
    """测试 Rotation.create_group 函数对于不合法的轴参数的行为。
    应该抛出 ValueError 异常并包含特定错误信息。
    """
    with pytest.raises(ValueError,
                       match="`axis` must be one of"):
        Rotation.create_group("C1", axis)


def test_icosahedral():
    """测试二十面体群的创建和应用。
    通过验证应用二十面体群的结果是否保持不变来进行测试。
    """
    pass  # 此处没有实际测试代码，只是一个测试的占位符
    of the rotation group."""
    # 生成一个正二十面体模型 P
    P = _generate_icosahedron()
    # 对于每个从 "I" 组创建的旋转 g
    for g in Rotation.create_group("I"):
        # 将 g 转换为四元数，然后再转换为旋转对象
        g = Rotation.from_quat(g.as_quat())
        # 断言 P 和应用 g 后的 P 的均方根误差小于 TOL
        assert _calculate_rmsd(P, g.apply(P)) < TOL
# 测试八面体群是否正确修正八面体的旋转
def test_octahedral():
    """Test that the octahedral group correctly fixes the rotations of an
    octahedron."""
    # 生成八面体的顶点集合
    P = _generate_octahedron()
    # 遍历创建的八面体群的每个旋转操作
    for g in Rotation.create_group("O"):
        # 断言旋转后的顶点集合与原始顶点集合的均方根偏差小于TOL
        assert _calculate_rmsd(P, g.apply(P)) < TOL


# 测试四面体群是否正确修正四面体的旋转
def test_tetrahedral():
    """Test that the tetrahedral group correctly fixes the rotations of a
    tetrahedron."""
    # 生成四面体的顶点集合
    P = _generate_tetrahedron()
    # 遍历创建的四面体群的每个旋转操作
    for g in Rotation.create_group("T"):
        # 断言旋转后的顶点集合与原始顶点集合的均方根偏差小于TOL
        assert _calculate_rmsd(P, g.apply(P)) < TOL


# 使用参数化测试来测试二面体群是否正确修正棱柱的旋转
@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("axis", 'XYZ')
def test_dicyclic(n, axis):
    """Test that the dicyclic group correctly fixes the rotations of a
    prism."""
    # 生成具有n个面和指定轴向的棱柱的顶点集合
    P = _generate_prism(n, axis='XYZ'.index(axis))
    # 遍历创建的二面体群的每个旋转操作
    for g in Rotation.create_group("D%d" % n, axis=axis):
        # 断言旋转后的顶点集合与原始顶点集合的均方根偏差小于TOL
        assert _calculate_rmsd(P, g.apply(P)) < TOL


# 使用参数化测试来测试循环群是否正确修正金字塔的旋转
@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("axis", 'XYZ')
def test_cyclic(n, axis):
    """Test that the cyclic group correctly fixes the rotations of a
    pyramid."""
    # 生成具有n个面和指定轴向的金字塔的顶点集合
    P = _generate_pyramid(n, axis='XYZ'.index(axis))
    # 遍历创建的循环群的每个旋转操作
    for g in Rotation.create_group("C%d" % n, axis=axis):
        # 断言旋转后的顶点集合与原始顶点集合的均方根偏差小于TOL
        assert _calculate_rmsd(P, g.apply(P)) < TOL


# 使用参数化测试来测试不同群的大小是否符合预期
@pytest.mark.parametrize("name, size", zip(NAMES, SIZES))
def test_group_sizes(name, size):
    # 断言创建的旋转群的大小与预期大小相等
    assert len(Rotation.create_group(name)) == size


# 使用参数化测试来测试不同群是否存在重复的旋转操作
@pytest.mark.parametrize("name, size", zip(NAMES, SIZES))
def test_group_no_duplicates(name, size):
    # 创建旋转群并构建其对应的KD树
    g = Rotation.create_group(name)
    kdtree = cKDTree(g.as_quat())
    # 断言KD树中不存在距离小于1E-3的重复点对
    assert len(kdtree.query_pairs(1E-3)) == 0


# 使用参数化测试来测试不同群的对称性
@pytest.mark.parametrize("name, size", zip(NAMES, SIZES))
def test_group_symmetry(name, size):
    # 创建旋转群并将其四元数形式与其负值合并
    g = Rotation.create_group(name)
    q = np.concatenate((-g.as_quat(), g.as_quat()))
    # 计算四元数之间的距离矩阵，并计算每列的最大差值
    distance = np.sort(cdist(q, q))
    deltas = np.max(distance, axis=0) - np.min(distance, axis=0)
    # 断言所有差值均小于TOL
    assert (deltas < TOL).all()


# 使用参数化测试来测试旋转群的约化操作
@pytest.mark.parametrize("name", NAMES)
def test_reduction(name):
    """Test that the elements of the rotation group are correctly
    mapped onto the identity rotation."""
    # 创建旋转群并约化其中的所有旋转操作
    g = Rotation.create_group(name)
    f = g.reduce(g)
    # 断言约化后的旋转操作的大小近似于零向量
    assert_array_almost_equal(f.magnitude(), np.zeros(len(g)))


# 使用参数化测试来测试单个旋转操作的约化操作
@pytest.mark.parametrize("name", NAMES)
def test_single_reduction(name):
    # 创建旋转群并约化最后一个旋转操作
    g = Rotation.create_group(name)
    f = g[-1].reduce(g)
    # 断言约化后的旋转操作的大小近似于零，且其四元数形式为(4,)
    assert_array_almost_equal(f.magnitude(), 0)
    assert f.as_quat().shape == (4,)
```
# `D:\src\scipysrc\scipy\scipy\spatial\tests\test_hausdorff.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
from numpy.testing import (assert_allclose,  # 导入NumPy测试工具中的函数
                           assert_array_equal,
                           assert_equal)
import pytest  # 导入pytest，用于编写和运行测试
from scipy.spatial.distance import directed_hausdorff  # 导入SciPy库中的directed_hausdorff函数
from scipy.spatial import distance  # 导入SciPy库中的距离计算模块
from scipy._lib._util import check_random_state  # 导入SciPy库中的随机状态检查函数


class TestHausdorff:
    # 测试 directed Hausdorff 距离的各种属性

    def setup_method(self):
        np.random.seed(1234)  # 设置随机数种子以保证结果的可重复性
        random_angles = np.random.random(100) * np.pi * 2  # 生成100个随机角度
        random_columns = np.column_stack(
            (random_angles, random_angles, np.zeros(100)))  # 将随机角度堆叠为三列数组
        random_columns[..., 0] = np.cos(random_columns[..., 0])  # 计算cos值
        random_columns[..., 1] = np.sin(random_columns[..., 1])  # 计算sin值
        random_columns_2 = np.column_stack(
            (random_angles, random_angles, np.zeros(100)))  # 再次生成相同的随机角度数组
        random_columns_2[1:, 0] = np.cos(random_columns_2[1:, 0]) * 2.0  # 第二列的cos值扩大两倍
        random_columns_2[1:, 1] = np.sin(random_columns_2[1:, 1]) * 2.0  # 第二列的sin值扩大两倍
        # 将第一个点移动得更远，避免两个完美圆形
        random_columns_2[0, 0] = np.cos(random_columns_2[0, 0]) * 3.3  # 第一个点的cos值乘以3.3
        random_columns_2[0, 1] = np.sin(random_columns_2[0, 1]) * 3.3  # 第一个点的sin值乘以3.3
        self.path_1 = random_columns  # 将第一个路径设置为随机列数组
        self.path_2 = random_columns_2  # 将第二个路径设置为随机列数组
        self.path_1_4d = np.insert(self.path_1, 3, 5, axis=1)  # 在第一个路径中插入一个维度为4的值为5
        self.path_2_4d = np.insert(self.path_2, 3, 27, axis=1)  # 在第二个路径中插入一个维度为4的值为27

    def test_symmetry(self):
        # 确保 directed Hausdorff 距离是对称的

        forward = directed_hausdorff(self.path_1, self.path_2)[0]  # 计算从路径1到路径2的 directed Hausdorff 距离
        reverse = directed_hausdorff(self.path_2, self.path_1)[0]  # 计算从路径2到路径1的 directed Hausdorff 距离
        assert forward != reverse  # 断言前向距离不等于反向距离

    def test_brute_force_comparison_forward(self):
        # 确保 directed_hausdorff 算法在前向方向上与简单/蛮力方法得出相同结果

        actual = directed_hausdorff(self.path_1, self.path_2)[0]  # 计算路径1到路径2的 directed Hausdorff 距离
        # 蛮力方法计算距离矩阵并取最大值作为期望值
        expected = max(np.amin(distance.cdist(self.path_1, self.path_2),
                               axis=1))
        assert_allclose(actual, expected)  # 断言实际值与期望值在数值上接近

    def test_brute_force_comparison_reverse(self):
        # 确保 directed_hausdorff 算法在反向方向上与简单/蛮力方法得出相同结果

        actual = directed_hausdorff(self.path_2, self.path_1)[0]  # 计算路径2到路径1的 directed Hausdorff 距离
        # 蛮力方法计算距离矩阵并取最大值作为期望值
        expected = max(np.amin(distance.cdist(self.path_1, self.path_2),
                               axis=0))
        assert_allclose(actual, expected)  # 断言实际值与期望值在数值上接近

    def test_degenerate_case(self):
        # 如果输入数据数组完全匹配，则 directed Hausdorff 距离必须为零

        actual = directed_hausdorff(self.path_1, self.path_1)[0]  # 计算路径1到路径1的 directed Hausdorff 距离
        assert_allclose(actual, 0.0)  # 断言实际值接近于0.0
    # 确保二维数据在简单情况下被正确处理，相对于蛮力方法。
    actual = directed_hausdorff(self.path_1[..., :2],
                                self.path_2[..., :2])[0]
    # 计算两个路径中每个点之间的最小距离，然后取最大值作为 Hausdorff 距离的一部分
    expected = max(np.amin(distance.cdist(self.path_1[..., :2],
                                          self.path_2[..., :2]),
                           axis=1))
    # 断言实际计算的值与预期的值非常接近
    assert_allclose(actual, expected)

    # 确保四维数据在简单情况下被正确处理，相对于蛮力方法。
    actual = directed_hausdorff(self.path_2_4d, self.path_1_4d)[0]
    # 蛮力法计算列之间的最小距离，然后取最大值
    expected = max(np.amin(distance.cdist(self.path_1_4d, self.path_2_4d),
                           axis=0))
    # 断言实际计算的值与预期的值非常接近
    assert_allclose(actual, expected)

    # 确保返回正确的点索引，它们应该对应于Hausdorff对
    path_simple_1 = np.array([[-1,-12],[0,0], [1,1], [3,7], [1,2]])
    path_simple_2 = np.array([[0,0], [1,1], [4,100], [10,9]])
    actual = directed_hausdorff(path_simple_2, path_simple_1)[1:]
    expected = (2, 3)
    # 断言实际返回的索引与预期的索引相等
    assert_array_equal(actual, expected)

    # 确保全局随机状态不会因为使用了随机化的 directed Hausdorff 算法而修改
    rs = check_random_state(None)
    old_global_state = rs.get_state()
    directed_hausdorff(self.path_1, self.path_2)
    rs2 = check_random_state(None)
    new_global_state = rs2.get_state()
    # 断言新的全局状态与旧的全局状态相等
    assert_equal(new_global_state, old_global_state)

    # 检查当种子值为 None 或整数时，全局随机状态不会改变
    rs = check_random_state(None)
    old_global_state = rs.get_state()
    directed_hausdorff(self.path_1, self.path_2, seed)
    rs2 = check_random_state(None)
    new_global_state = rs2.get_state()
    # 断言新的全局状态与旧的全局状态相等
    assert_equal(new_global_state, old_global_state)

    # 确保在列数不同的情况下引发 ValueError
    rng = np.random.default_rng(189048172503940875434364128139223470523)
    A = rng.random((3, 2))
    B = rng.random((3, 5))
    msg = r"need to have the same number of columns"
    # 使用 pytest 断言捕获到 ValueError，并且错误消息匹配预期的消息
    with pytest.raises(ValueError, match=msg):
        directed_hausdorff(A, B)
    # 使用 pytest 的参数化装饰器标记来定义多组测试参数
    @pytest.mark.parametrize("A, B, seed, expected", [
        # 第一组测试参数 (A=[(0,0)], B=[(0,1), (0,0)], seed=0, expected=(0.0, 0, 1))
        ([(0,0)],
         [(0,1), (0,0)],
         0,
         (0.0, 0, 1)),
        # 第二组测试参数 (A=[(0,0)], B=[(0,1), (0,0)], seed=1, expected=(0.0, 0, 1))
        ([(0,0)],
         [(0,1), (0,0)],
         1,
         (0.0, 0, 1)),
        # 第三组测试参数 (A=[(-5, 3), (0,0)], B=[(0,1), (0,0), (-5, 3)], seed=77098, expected=(0.0, 1, 1))
        ([(-5, 3), (0,0)],
         [(0,1), (0,0), (-5, 3)],
         77098,
         # 最大最小距离将是最后找到的距离，但不能保证总是唯一解
         (0.0, 1, 1)),
    ])
    # 定义测试方法，参数由装饰器提供
    def test_subsets(self, A, B, seed, expected):
        # 验证修复 gh-11332 的功能
        actual = directed_hausdorff(u=A, v=B, seed=seed)
        # 检查距离是否接近期望值
        assert_allclose(actual[0], expected[0])
        # 检查索引是否匹配期望值
        assert actual[1:] == expected[1:]
@pytest.mark.xslow
def test_massive_arr_overflow():
    # 标记为 xslow 的测试函数，预计运行较慢
    # 在 64 位系统上，我们应该能够处理超过 32 位有符号整数索引大小的数组
    try:
        import psutil
    except ModuleNotFoundError:
        # 如果找不到 psutil 模块，跳过测试并给出相应提示
        pytest.skip("psutil required to check available memory")
    if psutil.virtual_memory().available < 80*2**30:
        # 如果可用内存少于 80GB，不运行此测试
        pytest.skip('insufficient memory available to run this test')
    size = int(3e9)
    # 创建一个大小为 size x 2 的零数组 arr1
    arr1 = np.zeros(shape=(size, 2))
    # 创建一个大小为 3 x 2 的零数组 arr2
    arr2 = np.zeros(shape=(3, 2))
    # 将 arr1 的最后一行设置为 [5, 5]
    arr1[size - 1] = [5, 5]
    # 计算 arr1 和 arr2 之间的 Hausdorff 距离
    actual = directed_hausdorff(u=arr1, v=arr2)
    # 断言第一个返回值与预期值的接近程度
    assert_allclose(actual[0], 7.0710678118654755)
    # 断言第二个返回值与预期的 size - 1 相等
    assert_allclose(actual[1], size - 1)
```
# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_murmurhash.py`

```
# 导入必要的库和模块
import numpy as np  # 导入 NumPy 库
from numpy.testing import assert_array_almost_equal, assert_array_equal  # 导入 NumPy 测试相关的函数

# 从 scikit-learn 的工具包中导入 murmurhash3_32 函数
from sklearn.utils.murmurhash import murmurhash3_32


# 测试 murmurhash3_32 函数对整数输入的返回值
def test_mmhash3_int():
    assert murmurhash3_32(3) == 847579505  # 检查默认情况下的哈希值
    assert murmurhash3_32(3, seed=0) == 847579505  # 使用 seed=0 的哈希值
    assert murmurhash3_32(3, seed=42) == -1823081949  # 使用 seed=42 的哈希值

    assert murmurhash3_32(3, positive=False) == 847579505  # 确保 positive=False 时的哈希值
    assert murmurhash3_32(3, seed=0, positive=False) == 847579505  # 使用 seed=0 和 positive=False 的哈希值
    assert murmurhash3_32(3, seed=42, positive=False) == -1823081949  # 使用 seed=42 和 positive=False 的哈希值

    assert murmurhash3_32(3, positive=True) == 847579505  # 确保 positive=True 时的哈希值
    assert murmurhash3_32(3, seed=0, positive=True) == 847579505  # 使用 seed=0 和 positive=True 的哈希值
    assert murmurhash3_32(3, seed=42, positive=True) == 2471885347  # 使用 seed=42 和 positive=True 的哈希值


# 测试 murmurhash3_32 函数对整数数组输入的返回值
def test_mmhash3_int_array():
    rng = np.random.RandomState(42)  # 创建一个随机数生成器
    keys = rng.randint(-5342534, 345345, size=3 * 2 * 1).astype(np.int32)  # 生成随机整数数组
    keys = keys.reshape((3, 2, 1))  # 调整数组形状

    for seed in [0, 42]:
        expected = np.array([murmurhash3_32(int(k), seed) for k in keys.flat])  # 计算预期的哈希值数组
        expected = expected.reshape(keys.shape)
        assert_array_equal(murmurhash3_32(keys, seed), expected)  # 检查数组输入时的哈希值是否与预期相符

    for seed in [0, 42]:
        expected = np.array([murmurhash3_32(k, seed, positive=True) for k in keys.flat])  # 计算预期的正数哈希值数组
        expected = expected.reshape(keys.shape)
        assert_array_equal(murmurhash3_32(keys, seed, positive=True), expected)  # 检查数组输入和 positive=True 时的哈希值是否与预期相符


# 测试 murmurhash3_32 函数对字节串输入的返回值
def test_mmhash3_bytes():
    assert murmurhash3_32(b"foo", 0) == -156908512  # 检查字节串 b"foo" 的哈希值
    assert murmurhash3_32(b"foo", 42) == -1322301282  # 使用 seed=42 的哈希值

    assert murmurhash3_32(b"foo", 0, positive=True) == 4138058784  # 使用 positive=True 的哈希值
    assert murmurhash3_32(b"foo", 42, positive=True) == 2972666014  # 使用 seed=42 和 positive=True 的哈希值


# 测试 murmurhash3_32 函数对 Unicode 字符串输入的返回值
def test_mmhash3_unicode():
    assert murmurhash3_32("foo", 0) == -156908512  # 检查 Unicode 字符串 "foo" 的哈希值
    assert murmurhash3_32("foo", 42) == -1322301282  # 使用 seed=42 的哈希值

    assert murmurhash3_32("foo", 0, positive=True) == 4138058784  # 使用 positive=True 的哈希值
    assert murmurhash3_32("foo", 42, positive=True) == 2972666014  # 使用 seed=42 和 positive=True 的哈希值


# 测试空字符串的哈希值是否有冲突
def test_no_collision_on_byte_range():
    previous_hashes = set()
    for i in range(100):
        h = murmurhash3_32(" " * i, 0)  # 计算空字符串乘以 i 的哈希值
        assert h not in previous_hashes, "Found collision on growing empty string"  # 确保在递增空字符串时没有哈希冲突
        previous_hashes.add(h)


# 测试哈希值在正数范围内的均匀分布性质
def test_uniform_distribution():
    n_bins, n_samples = 10, 100000  # 定义箱子数和样本数
    bins = np.zeros(n_bins, dtype=np.float64)  # 创建一个浮点数数组用于计算每个箱子的样本数

    for i in range(n_samples):
        bins[murmurhash3_32(i, positive=True) % n_bins] += 1  # 计算哈希值并将其分配到相应的箱子中

    means = bins / n_samples  # 计算每个箱子的平均样本数
    expected = np.full(n_bins, 1.0 / n_bins)  # 定义期望的均匀分布

    assert_array_almost_equal(means / expected, np.ones(n_bins), 2)  # 检查实际分布与期望分布的一致性
```
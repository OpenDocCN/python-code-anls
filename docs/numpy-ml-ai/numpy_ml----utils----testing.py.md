# `numpy-ml\numpy_ml\utils\testing.py`

```
# 导入必要的库
import numbers
import numpy as np

# 断言函数

# 检查数组 `X` 是否沿其主对角线对称
def is_symmetric(X):
    return np.allclose(X, X.T)

# 检查矩阵 `X` 是否对称且正定
def is_symmetric_positive_definite(X):
    if is_symmetric(X):
        try:
            # 如果矩阵对称，检查 Cholesky 分解是否存在（仅对对称/共轭正定矩阵定义）
            np.linalg.cholesky(X)
            return True
        except np.linalg.LinAlgError:
            return False
    return False

# 检查 `X` 是否包含沿列和为1的概率
def is_stochastic(X):
    msg = "Array should be stochastic along the columns"
    assert len(X[X < 0]) == len(X[X > 1]) == 0, msg
    assert np.allclose(np.sum(X, axis=1), np.ones(X.shape[0]), msg)
    return True

# 检查值 `a` 是否为数值型
def is_number(a):
    return isinstance(a, numbers.Number)

# 如果数组 `x` 是二进制数组且只有一个1，则返回True
def is_one_hot(x):
    msg = "Matrix should be one-hot binary"
    assert np.array_equal(x, x.astype(bool)), msg
    assert np.allclose(np.sum(x, axis=1), np.ones(x.shape[0]), msg)
    return True

# 如果数组 `x` 仅由二进制值组成，则返回True
def is_binary(x):
    msg = "Matrix must be binary"
    assert np.array_equal(x, x.astype(bool), msg)
    return True
# 创建一个形状为 (`n_examples`, `n_classes`) 的随机独热编码矩阵
def random_one_hot_matrix(n_examples, n_classes):
    # 创建一个单位矩阵，行数为类别数，列数为类别数，表示每个类别的独热编码
    X = np.eye(n_classes)
    # 从类别数中随机选择 n_examples 个类别的独热编码
    X = X[np.random.choice(n_classes, n_examples)]
    return X


# 创建一个形状为 (`n_examples`, `n_classes`) 的随机随机矩阵
def random_stochastic_matrix(n_examples, n_classes):
    # 创建一个形状为 (`n_examples`, `n_classes`) 的随机矩阵
    X = np.random.rand(n_examples, n_classes)
    # 对每一行进行归一化，使得每一行的和为1
    X /= X.sum(axis=1, keepdims=True)
    return X


# 创建一个形状为 `shape` 的随机实值张量。如果 `standardize` 为 True，则确保每列的均值为0，标准差为1
def random_tensor(shape, standardize=False):
    # 创建一个形状为 `shape` 的随机偏移量
    offset = np.random.randint(-300, 300, shape)
    # 创建一个形状为 `shape` 的随机矩阵，并加上偏移量
    X = np.random.rand(*shape) + offset

    if standardize:
        # 计算一个很小的数，用于避免除以0
        eps = np.finfo(float).eps
        # 对每列进行标准化，使得均值为0，标准差为1
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + eps)
    return X


# 创建一个形状为 `shape` 的随机二值张量。`sparsity` 是一个控制输出张量中0和1比例的值
def random_binary_tensor(shape, sparsity=0.5):
    # 创建一个形状为 `shape` 的随机矩阵，大于等于 (1 - sparsity) 的值设为1，小于 (1 - sparsity) 的值设为0
    return (np.random.rand(*shape) >= (1 - sparsity)).astype(float)


# 生成一个由 `n_words` 个单词组成的随机段落。如果 `vocab` 不为 None，则从该列表中随机抽取单词；否则，从包含 26 个拉丁单词的集合中均匀抽取单词
    # 如果输入的词汇表为空，则使用默认的词汇表
    if vocab is None:
        vocab = [
            "at",
            "stet",
            "accusam",
            "aliquyam",
            "clita",
            "lorem",
            "ipsum",
            "dolor",
            "dolore",
            "dolores",
            "sit",
            "amet",
            "consetetur",
            "sadipscing",
            "elitr",
            "sed",
            "diam",
            "nonumy",
            "eirmod",
            "duo",
            "ea",
            "eos",
            "erat",
            "est",
            "et",
            "gubergren",
        ]
    # 返回一个包含 n_words 个随机词汇的列表
    return [np.random.choice(vocab) for _ in range(n_words)]
# 自定义警告类，继承自 RuntimeWarning
class DependencyWarning(RuntimeWarning):
    pass
```
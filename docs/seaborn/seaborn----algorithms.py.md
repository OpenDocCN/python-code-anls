# `D:\src\scipysrc\seaborn\seaborn\algorithms.py`

```
"""Algorithms to support fitting routines in seaborn plotting functions."""
import numpy as np
import warnings


def bootstrap(*args, **kwargs):
    """Resample one or more arrays with replacement and store aggregate values.

    Positional arguments are a sequence of arrays to bootstrap along the first
    axis and pass to a summary function.

    Keyword arguments:
        n_boot : int, default=10000
            Number of iterations
        axis : int, default=None
            Will pass axis to ``func`` as a keyword argument.
        units : array, default=None
            Array of sampling unit IDs. When used the bootstrap resamples units
            and then observations within units instead of individual
            datapoints.
        func : string or callable, default="mean"
            Function to call on the args that are passed in. If string, uses as
            name of function in the numpy namespace. If nans are present in the
            data, will try to use nan-aware version of named function.
        seed : Generator | SeedSequence | RandomState | int | None
            Seed for the random number generator; useful if you want
            reproducible resamples.

    Returns
    -------
    boot_dist: array
        array of bootstrapped statistic values

    """
    # Ensure list of arrays are same length
    # 确保所有输入数组的长度相同
    if len(np.unique(list(map(len, args)))) > 1:
        raise ValueError("All input arrays must have the same length")
    n = len(args[0])

    # Default keyword arguments
    # 默认关键字参数设置
    n_boot = kwargs.get("n_boot", 10000)  # 获取或设置默认的 bootstrap 迭代次数
    func = kwargs.get("func", "mean")  # 获取或设置默认的统计函数，默认为平均值
    axis = kwargs.get("axis", None)  # 获取或设置默认的轴参数，将传递给统计函数
    units = kwargs.get("units", None)  # 获取或设置默认的单位数组，用于分组抽样
    random_seed = kwargs.get("random_seed", None)
    if random_seed is not None:
        msg = "`random_seed` has been renamed to `seed` and will be removed"
        warnings.warn(msg)
    seed = kwargs.get("seed", random_seed)  # 获取随机数生成器的种子值
    if axis is None:
        func_kwargs = dict()
    else:
        func_kwargs = dict(axis=axis)

    # Initialize the resampler
    # 初始化重采样器
    if isinstance(seed, np.random.RandomState):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    # Coerce to arrays
    # 将输入参数转换为数组形式
    args = list(map(np.asarray, args))
    if units is not None:
        units = np.asarray(units)

    if isinstance(func, str):

        # Allow named numpy functions
        # 允许使用指定名称的 numpy 函数
        f = getattr(np, func)

        # Try to use nan-aware version of function if necessary
        # 如果数据中存在 NaN 值，则尝试使用 NaN 安全版本的函数
        missing_data = np.isnan(np.sum(np.column_stack(args)))

        if missing_data and not func.startswith("nan"):
            nanf = getattr(np, f"nan{func}", None)
            if nanf is None:
                msg = f"Data contain nans but no nan-aware version of `{func}` found"
                warnings.warn(msg, UserWarning)
            else:
                f = nanf

    else:
        f = func

    # Handle numpy changes
    # 处理 numpy 的变化
    try:
        integers = rng.integers
    except AttributeError:
        integers = rng.randint

    # Do the bootstrap
    # 执行 bootstrap 操作
    # 如果给定 units 参数不为空，则执行结构化的自助法计算
    if units is not None:
        return _structured_bootstrap(args, n_boot, units, f,
                                     func_kwargs, integers)

    # 如果 units 参数为空，则执行普通的自助法计算

    # 初始化一个空列表，用于存储自助法重抽样得到的统计量
    boot_dist = []

    # 循环执行 n_boot 次自助法重抽样
    for i in range(int(n_boot)):
        # 生成一个重抽样索引，integers 是一个函数用来生成整数索引，dtype=np.intp 表示索引的数据类型为整数
        resampler = integers(0, n, n, dtype=np.intp)  # intp is indexing dtype

        # 根据 resampler 的索引，从每个参数 args 中抽取样本
        sample = [a.take(resampler, axis=0) for a in args]

        # 使用函数 f 计算抽取样本的统计量，将结果添加到 boot_dist 列表中
        boot_dist.append(f(*sample, **func_kwargs))

    # 将 boot_dist 转换为 numpy 数组并返回
    return np.array(boot_dist)
# 使用单位作为重采样的单位，而不是数据点。
def _structured_bootstrap(args, n_boot, units, func, func_kwargs, integers):
    # 计算唯一单位的列表
    unique_units = np.unique(units)
    # 计算唯一单位的数量
    n_units = len(unique_units)

    # 为每个参数列表中的每个单位创建一个子列表
    args = [[a[units == unit] for unit in unique_units] for a in args]

    # 初始化空的 Bootstrap 分布列表
    boot_dist = []
    # 进行 n_boot 次 Bootstrap 抽样
    for i in range(int(n_boot)):
        # 从 [0, n_units) 区间中抽样，作为单位的索引
        resampler = integers(0, n_units, n_units, dtype=np.intp)
        # 根据抽样的单位索引，从 args 中抽取数据形成样本
        sample = [[a[i] for i in resampler] for a in args]
        # 计算每个样本的长度
        lengths = map(len, sample[0])
        # 为每个样本根据其长度从 [0, n) 区间中抽样
        resampler = [integers(0, n, n, dtype=np.intp) for n in lengths]
        # 根据抽样索引从每个样本中提取子集，并进行拼接
        sample = [[c.take(r, axis=0) for c, r in zip(a, resampler)] for a in sample]
        # 对拼接后的样本应用函数 func，并将结果添加到 Bootstrap 分布中
        boot_dist.append(func(*sample, **func_kwargs))
    
    # 将 Bootstrap 分布转换为 NumPy 数组并返回
    return np.array(boot_dist)
```
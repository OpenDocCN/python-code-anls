# `.\numpy\benchmarks\benchmarks\common.py`

```py
# 导入必要的库
import numpy as np  # 导入NumPy库，用于科学计算
import random  # 导入random库，用于生成随机数
import os  # 导入os库，提供与操作系统交互的功能
from functools import lru_cache  # 从functools模块导入lru_cache装饰器，用于缓存函数的结果
from pathlib import Path  # 导入Path类，用于处理文件路径

# Various pre-crafted datasets/variables for testing
# !!! Must not be changed -- only appended !!!
# while testing numpy we better not rely on numpy to produce random
# sequences

random.seed(1)  # 设置随机数种子为1，确保随机数序列的可重复性
np.random.seed(1)  # 设置NumPy随机数种子为1，确保NumPy生成的随机数序列的可重复性

nx, ny = 1000, 1000  # 定义变量nx和ny，表示数据矩阵的维度为1000x1000
# reduced squares based on indexes_rand, primarily for testing more
# time-consuming functions (ufunc, linalg, etc)
nxs, nys = 100, 100  # 定义变量nxs和nys，表示较小的数据矩阵的维度为100x100

# a list of interesting types to test
TYPES1 = [
    'int16', 'float16',
    'int32', 'float32',
    'int64', 'float64',  'complex64',
    'complex128',
]  # 定义列表TYPES1，包含不同的数据类型，用于测试

DLPACK_TYPES = [
    'int16', 'float16',
    'int32', 'float32',
    'int64', 'float64',  'complex64',
    'complex128', 'bool',
]  # 定义列表DLPACK_TYPES，包含不同的数据类型和bool类型，用于测试

# Path for caching
CACHE_ROOT = Path(__file__).resolve().parent.parent / 'env' / 'numpy_benchdata'
# 定义路径变量CACHE_ROOT，指定缓存数据的根目录路径

# values which will be used to construct our sample data matrices
# replicate 10 times to speed up initial imports of this helper
# and generate some redundancy

@lru_cache(typed=True)
def get_values():
    """
    Generate and cache random values for constructing sample data matrices.

    Returns
    -------
    values: ndarray
        Random values for constructing matrices.
    """
    rnd = np.random.RandomState(1804169117)  # 创建一个NumPy随机状态对象
    values = np.tile(rnd.uniform(0, 100, size=nx*ny//10), 10)  # 生成重复的随机数值
    return values


@lru_cache(typed=True)
def get_square(dtype):
    """
    Generate and cache a square matrix of given data type.

    Parameters
    ----------
    dtype: str
        Data type for the matrix elements.

    Returns
    -------
    arr: ndarray
        Generated square matrix.
    """
    values = get_values()  # 调用get_values函数获取随机数值
    arr = values.astype(dtype=dtype).reshape((nx, ny))  # 根据给定数据类型生成矩阵
    if arr.dtype.kind == 'c':
        arr += arr.T*1j  # 如果数据类型是复数，调整使得虚部非退化
    return arr


@lru_cache(typed=True)
def get_squares():
    """
    Generate and cache square matrices for all types in TYPES1.

    Returns
    -------
    squares: dict
        Dictionary mapping data types to their respective square matrices.
    """
    return {t: get_square(t) for t in sorted(TYPES1)}  # 生成不同数据类型的方阵，并以字典形式返回


@lru_cache(typed=True)
def get_square_(dtype):
    """
    Generate and cache a smaller square matrix of given data type.

    Parameters
    ----------
    dtype: str
        Data type for the matrix elements.

    Returns
    -------
    arr: ndarray
        Generated smaller square matrix.
    """
    arr = get_square(dtype)  # 调用get_square函数获取指定数据类型的方阵
    return arr[:nxs, :nys]  # 返回指定大小的子矩阵


@lru_cache(typed=True)
def get_squares_():
    """
    Generate and cache smaller square matrices for all types in TYPES1.

    Returns
    -------
    squares: dict
        Dictionary mapping data types to their respective smaller square matrices.
    """
    return {t: get_square_(t) for t in sorted(TYPES1)}  # 生成不同数据类型的小方阵，并以字典形式返回


@lru_cache(typed=True)
def get_indexes():
    """
    Generate and cache a list of indexes for data manipulation.

    Returns
    -------
    indexes: ndarray
        Array of indexes.
    """
    indexes = list(range(nx))  # 生成包含所有索引的列表
    indexes.pop(5)  # 移除索引为5的元素
    indexes.pop(95)  # 移除索引为95的元素
    indexes = np.array(indexes)  # 转换为NumPy数组
    return indexes


@lru_cache(typed=True)
def get_indexes_rand():
    """
    Generate and cache random indexes for data manipulation.

    Returns
    -------
    indexes_rand: ndarray
        Array of random indexes.
    """
    rnd = random.Random(1)  # 创建一个随机数生成器对象
    indexes_rand = get_indexes().tolist()  # 获取索引列表的副本
    rnd.shuffle(indexes_rand)  # 随机打乱索引列表
    indexes_rand = np.array(indexes_rand)  # 转换为NumPy数组
    return indexes_rand


@lru_cache(typed=True)
def get_indexes_():
    """
    Generate and cache a list of smaller indexes for data manipulation.

    Returns
    -------
    indexes_: ndarray
        Array of smaller indexes.
    """
    indexes = get_indexes()  # 调用get_indexes函数获取索引数组
    indexes_ = indexes[indexes < nxs]  # 筛选出小于nxs的索引
    return indexes_


@lru_cache(typed=True)
def get_indexes_rand_():
    """
    Generate and cache random smaller indexes for data manipulation.

    Returns
    -------
    indexes_rand_: ndarray
        Array of random smaller indexes.
    """
    indexes_rand = get_indexes_rand()  # 调用get_indexes_rand函数获取随机索引数组
    indexes_rand_ = indexes_rand[indexes_rand < nxs]  # 筛选出小于nxs的随机索引
    return indexes_rand_


@lru_cache(typed=True)
def get_data(size, dtype, ip_num=0, zeros=False, finite=True, denormal=False):
    """
    Generate a cached random array that covers several scenarios affecting benchmarks.

    Parameters
    ----------
    size: int
        Array length.
    dtype: dtype or dtype specifier
        Data type for the array elements.
    ip_num: int, optional
        Placeholder for future use.
    zeros: bool, optional
        Whether to include zeros in the array.
    finite: bool, optional
        Whether to include finite numbers in the array.
    denormal: bool, optional
        Whether to include denormal numbers in the array.

    Returns
    -------
    data: ndarray
        Generated random array based on specified parameters.
    """
    # Function details omitted for brevity
    # 定义输入的整数，以避免内存超载，并为每个操作数提供唯一的数据
    ip_num: int
        Input number, to avoid memory overload
        and to provide unique data for each operand.

    # 是否在生成的数据中添加零
    zeros: bool
        Spreading zeros along with generated data.

    # 避免在生成的浮点数中出现特殊情况，如NaN和inf
    finite: bool
        Avoid spreading fp special cases nan/inf.

    # 是否在生成的数据中添加次标准数（denormal）
    denormal:
        Spreading subnormal numbers along with generated data.
    """
    # 将输入的dtype转换为numpy的dtype对象
    dtype = np.dtype(dtype)
    # 获取dtype的名称
    dname = dtype.name
    # 构建缓存文件的名称，包括dtype名称、size、ip_num、zeros的标志位
    cache_name = f'{dname}_{size}_{ip_num}_{int(zeros)}'
    # 如果dtype是复数或者复数类型，则追加finite和denormal的标志位
    if dtype.kind in 'fc':
        cache_name += f'{int(finite)}{int(denormal)}'
    # 将'.bin'作为文件扩展名追加到缓存文件名
    cache_name += '.bin'
    # 构建缓存文件的完整路径
    cache_path = CACHE_ROOT / cache_name
    # 如果缓存文件已存在，则从文件中读取数据并返回numpy数组
    if cache_path.exists():
        return np.fromfile(cache_path, dtype)

    # 如果缓存文件不存在，则生成新的数据数组
    array = np.ones(size, dtype)
    # 生成随机数列表
    rands = []
    
    # 根据dtype的种类生成不同范围的随机整数
    if dtype.kind == 'i':  # 如果是有符号整数
        dinfo = np.iinfo(dtype)
        scale = 8
        if zeros:
            scale += 1
        lsize = size // scale
        for low, high in (
            (-0x80, -1),
            (1, 0x7f),
            (-0x8000, -1),
            (1, 0x7fff),
            (-0x80000000, -1),
            (1, 0x7fffffff),
            (-0x8000000000000000, -1),
            (1, 0x7fffffffffffffff),
        ):
            rands += [np.random.randint(
                max(low, dinfo.min),
                min(high, dinfo.max),
                lsize, dtype
            )]
    elif dtype.kind == 'u':  # 如果是无符号整数
        dinfo = np.iinfo(dtype)
        scale = 4
        if zeros:
            scale += 1
        lsize = size // scale
        for high in (0xff, 0xffff, 0xffffffff, 0xffffffffffffffff):
            rands += [np.random.randint(1, min(high, dinfo.max), lsize, dtype)]
    elif dtype.kind in 'fc':  # 如果是浮点数或复数
        scale = 1
        if zeros:
            scale += 1
        if not finite:
            scale += 2
        if denormal:
            scale += 1
        dinfo = np.finfo(dtype)
        lsize = size // scale
        rands = [np.random.rand(lsize).astype(dtype)]
        if not finite:
            # 如果不限制有限范围，则生成NaN和inf
            rands += [
                np.empty(lsize, dtype=dtype), np.empty(lsize, dtype=dtype)
            ]
            rands[1].fill(float('nan'))
            rands[2].fill(float('inf'))
        if denormal:
            # 如果允许生成次标准数，则填充最小次标准数
            rands += [np.empty(lsize, dtype=dtype)]
            rands[-1].fill(dinfo.smallest_subnormal)

    # 如果rands非空，则将生成的随机数填充到array中
    if rands:
        if zeros:
            # 如果需要在数组中填充零，则生成一个零数组，并按步长填充到array中
            rands += [np.zeros(lsize, dtype)]
        stride = len(rands)
        for start, r in enumerate(rands):
            array[start:len(r)*stride:stride] = r

    # 如果缓存根目录不存在，则创建之
    if not CACHE_ROOT.exists():
        CACHE_ROOT.mkdir(parents=True)
    # 将生成的数据数组写入缓存文件
    array.tofile(cache_path)
    # 返回生成的数据数组
    return array
# 定义一个名为 Benchmark 的空类，用于作为基准类或者后续扩展的基础
class Benchmark:
    pass
```
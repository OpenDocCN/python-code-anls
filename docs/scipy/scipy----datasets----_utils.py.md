# `D:\src\scipysrc\scipy\scipy\datasets\_utils.py`

```
import os  # 导入操作系统功能模块
import shutil  # 导入文件和目录管理模块
from ._registry import method_files_map  # 导入本地模块中的方法文件映射

try:
    import platformdirs  # 尝试导入平台相关的目录管理模块
except ImportError:
    platformdirs = None  # 如果导入失败，将其设为 None（用于类型提示）


def _clear_cache(datasets, cache_dir=None, method_map=None):
    if method_map is None:
        # 如果方法映射为空，则使用 SciPy Datasets 的方法映射
        method_map = method_files_map
    if cache_dir is None:
        # 如果缓存目录为空
        if platformdirs is None:
            # 如果平台目录模块未导入
            raise ImportError("Missing optional dependency 'pooch' required "
                              "for scipy.datasets module. Please use pip or "
                              "conda to install 'pooch'.")
        # 使用 platformdirs 模块获取用户缓存目录路径
        cache_dir = platformdirs.user_cache_dir("scipy-data")

    if not os.path.exists(cache_dir):
        # 如果缓存目录不存在，则打印消息并返回
        print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
        return

    if datasets is None:
        # 如果 datasets 为 None，则清空整个缓存目录
        print(f"Cleaning the cache directory {cache_dir}!")
        shutil.rmtree(cache_dir)
    else:
        if not isinstance(datasets, (list, tuple)):
            # 如果 datasets 不是列表或元组，则转换为单元素列表
            datasets = [datasets, ]
        for dataset in datasets:
            assert callable(dataset)  # 断言 dataset 是可调用对象
            dataset_name = dataset.__name__  # 获取数据集方法的名称
            if dataset_name not in method_map:
                # 如果数据集方法不在方法映射中，则抛出 ValueError
                raise ValueError(f"Dataset method {dataset_name} doesn't "
                                 "exist. Please check if the passed dataset "
                                 "is a subset of the following dataset "
                                 f"methods: {list(method_map.keys())}")

            data_files = method_map[dataset_name]  # 获取数据集方法对应的数据文件列表
            data_filepaths = [os.path.join(cache_dir, file)
                              for file in data_files]  # 构建数据文件完整路径列表
            for data_filepath in data_filepaths:
                if os.path.exists(data_filepath):
                    # 如果数据文件存在，则打印清理消息并删除文件
                    print("Cleaning the file "
                          f"{os.path.split(data_filepath)[1]} "
                          f"for dataset {dataset_name}")
                    os.remove(data_filepath)
                else:
                    # 如果数据文件不存在，则打印消息表示没有东西需要清理
                    print(f"Path {data_filepath} doesn't exist. "
                          "Nothing to clear.")


def clear_cache(datasets=None):
    """
    Cleans the scipy datasets cache directory.

    If a scipy.datasets method or a list/tuple of the same is
    provided, then clear_cache removes all the data files
    associated to the passed dataset method callable(s).

    By default, it removes all the cached data files.

    Parameters
    ----------
    datasets : callable or list/tuple of callable or None
        If provided, specifies which dataset(s) to clear from cache.

    Examples
    --------
    >>> from scipy import datasets
    >>> ascent_array = datasets.ascent()
    >>> ascent_array.shape
    (512, 512)
    >>> datasets.clear_cache([datasets.ascent])
    Cleaning the file ascent.dat for dataset ascent
    """
    _clear_cache(datasets)  # 调用 _clear_cache 函数清理缓存
```
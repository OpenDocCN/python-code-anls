# `D:\src\scipysrc\scipy\scipy\datasets\_download_all.py`

```
"""
Platform independent script to download all the
`scipy.datasets` module data files.
This doesn't require a full scipy build.

Run: python _download_all.py <download_dir>
"""

# 导入 argparse 模块，用于解析命令行参数
import argparse
# 尝试导入 pooch 库，用于数据集下载
try:
    import pooch
except ImportError:
    pooch = None

# 如果没有包信息或者包信息为空字符串，则假定作为 Python 脚本运行，使用绝对导入
if __package__ is None or __package__ == '':
    import _registry  # type: ignore
# 如果作为 Python 模块运行，则使用相对导入
else:
    from . import _registry


def download_all(path=None):
    """
    Utility method to download all the dataset files
    for `scipy.datasets` module.

    Parameters
    ----------
    path : str, optional
        Directory path to download all the dataset files.
        If None, default to the system cache_dir detected by pooch.
    """
    # 如果没有安装 pooch 库，则抛出 ImportError 异常
    if pooch is None:
        raise ImportError("Missing optional dependency 'pooch' required "
                          "for scipy.datasets module. Please use pip or "
                          "conda to install 'pooch'.")
    # 如果未指定下载路径，则使用 pooch 检测的系统缓存目录
    if path is None:
        path = pooch.os_cache('scipy-data')
    # 遍历数据集名称和哈希值的注册表，并下载数据集文件
    for dataset_name, dataset_hash in _registry.registry.items():
        pooch.retrieve(url=_registry.registry_urls[dataset_name],
                       known_hash=dataset_hash,
                       fname=dataset_name, path=path)


def main():
    # 创建命令行参数解析器对象
    parser = argparse.ArgumentParser(description='Download SciPy data files.')
    # 添加命令行参数 path，表示下载文件的目录路径，可选参数
    parser.add_argument("path", nargs='?', type=str,
                        default=pooch.os_cache('scipy-data'),
                        help="Directory path to download all the data files.")
    # 解析命令行参数
    args = parser.parse_args()
    # 调用 download_all 函数，并传入参数 args.path 作为下载路径
    download_all(args.path)


if __name__ == "__main__":
    # 当脚本直接执行时，调用主函数 main()
    main()
```
# `D:\src\scipysrc\scikit-learn\sklearn\utils\_optional_dependencies.py`

```
# 如果matplotlib未安装，则引发ImportError并提供详细的错误消息
def check_matplotlib_support(caller_name):
    try:
        import matplotlib  # noqa
    except ImportError as e:
        # 抛出带有详细错误消息的ImportError，指示调用者需要安装matplotlib
        raise ImportError(
            "{} requires matplotlib. You can install matplotlib with "
            "`pip install matplotlib`".format(caller_name)
        ) from e


# 如果pandas未安装，则引发ImportError并提供详细的错误消息
def check_pandas_support(caller_name):
    try:
        import pandas  # noqa
        # 返回已导入的pandas包
        return pandas
    except ImportError as e:
        # 抛出带有详细错误消息的ImportError，指示调用者需要安装pandas
        raise ImportError("{} requires pandas.".format(caller_name)) from e
```
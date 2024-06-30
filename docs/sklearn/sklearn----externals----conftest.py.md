# `D:\src\scipysrc\scikit-learn\sklearn\externals\conftest.py`

```
# 定义函数 pytest_ignore_collect，用于指定在收集测试时不收集任何外部模块。
# 这种方式比使用 --ignore 更健壮，因为 --ignore 需要指定一个路径，而在使用 --pyargs 时，
# 传递外部模块的路径（位于 site-packages 中，路径非常长且依赖于安装）并不方便。
def pytest_ignore_collect(collection_path, config):
    # 总是返回 True，表示不收集任何测试项
    return True
```
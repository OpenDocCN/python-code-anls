# `D:\src\scipysrc\matplotlib\doc\sphinxext\skip_deprecated.py`

```
# 跳过已弃用的成员

# 定义一个函数，用于在自动文档生成过程中跳过特定的成员
def skip_deprecated(app, what, name, obj, skip, options):
    # 如果已经标记为需要跳过，则直接返回跳过标记
    if skip:
        return skip
    # 定义一个字典，包含需要跳过的成员信息，按模块划分
    skipped = {"matplotlib.colors": ["ColorConverter", "hex2color", "rgb2hex"]}
    # 获取对象的模块名，并查看是否在跳过列表中
    skip_list = skipped.get(getattr(obj, "__module__", None))
    # 如果模块在跳过列表中，则检查对象的名称是否在列表中，若在则返回跳过标记
    if skip_list is not None:
        return getattr(obj, "__name__", None) in skip_list


# 设置函数，用于配置自动生成文档的应用程序
def setup(app):
    # 将 skip_deprecated 函数连接到 'autodoc-skip-member' 事件上
    app.connect('autodoc-skip-member', skip_deprecated)

    # 定义元数据，指示插件的并行读取和写入安全性
    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    # 返回插件的元数据
    return metadata
```
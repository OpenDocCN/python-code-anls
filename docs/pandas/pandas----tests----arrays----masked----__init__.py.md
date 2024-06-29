# `D:\src\scipysrc\pandas\pandas\tests\arrays\masked\__init__.py`

```
# 定义一个名为 `parse_config` 的函数，接受一个文件名参数 `fname`
def parse_config(fname):
    # 打开文件 `fname`，以只读模式读取内容并存储在 `f` 中
    with open(fname, 'r') as f:
        # 使用列表推导式读取每一行文件内容，过滤掉空白行，并去除每行末尾的换行符
        config = [line.strip() for line in f if line.strip()]
    # 返回处理后的配置列表 `config`
    return config
```
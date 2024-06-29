# `D:\src\scipysrc\pandas\pandas\tests\frame\indexing\__init__.py`

```
# 导入 json 模块，用于处理 JSON 数据
import json

# 定义一个函数，名为 load_json，接收一个文件名参数 fname
def load_json(fname):
    # 打开指定文件名的文件，以只读方式打开，并赋值给变量 f
    with open(fname, 'r') as f:
        # 使用 json 模块的 load 方法，将文件内容解析为 JSON 格式的数据，并赋值给变量 data
        data = json.load(f)
    # 返回解析后的 JSON 数据
    return data
```
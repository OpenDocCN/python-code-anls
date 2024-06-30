# `D:\src\scipysrc\scikit-learn\sklearn\inspection\_plot\__init__.py`

```
# 导入 json 模块，用于处理 JSON 数据
import json

# 定义一个名为 parse_config 的函数，接收一个文件名参数 fname
def parse_config(fname):
    # 打开文件 fname，模式为只读
    with open(fname, 'r') as f:
        # 使用 json 模块加载并解析文件内容，将其转换为 Python 对象（字典或列表）
        config = json.load(f)
    # 返回解析后的配置数据对象
    return config
```
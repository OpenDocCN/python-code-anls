# `.\pytorch\torch\testing\_internal\distributed\rpc\__init__.py`

```py
# 导入json模块，用于处理JSON格式数据
import json

# 定义一个函数parse_config，接收一个文件名参数
def parse_config(filename):
    # 打开指定文件名的文件，模式为只读
    with open(filename, 'r') as f:
        # 使用json.load()函数读取文件内容并解析为JSON格式
        config = json.load(f)
        # 返回解析后的JSON对象
        return config
```
# `.\pytorch\torch\distributed\algorithms\_checkpoint\__init__.py`

```
# 导入标准库中的 json 模块
import json

# 定义一个名为 parse_config 的函数，接收一个名为 fname 的参数
def parse_config(fname):
    # 打开指定文件 fname，模式为只读，并将其内容读取为文本
    with open(fname, 'r') as f:
        # 使用 json 模块解析文件 f 中的 JSON 数据，将其转换为 Python 字典对象
        config = json.load(f)
    
    # 返回解析后的配置字典
    return config
```
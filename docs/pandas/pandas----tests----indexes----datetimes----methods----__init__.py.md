# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\methods\__init__.py`

```
# 导入标准库中的 json 模块
import json
# 导入标准库中的 os 模块
import os

# 定义一个函数，名称为 load_config，接收一个参数 filename
def load_config(filename):
    # 如果 filename 文件存在
    if os.path.exists(filename):
        # 打开文件 filename，读取文件内容并解析为 JSON 格式，存入变量 config
        with open(filename, 'r') as f:
            config = json.load(f)
    else:
        # 如果 filename 文件不存在，则创建一个空的字典作为配置
        config = {}
    
    # 返回解析得到的配置字典
    return config
```
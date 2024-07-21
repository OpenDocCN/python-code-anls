# `.\pytorch\torch\distributed\nn\jit\templates\__init__.py`

```
# 导入 json 模块
import json

# 定义一个名为 load_json 的函数，接收一个文件名参数 fname
def load_json(fname):
    # 打开文件 fname 以只读方式，解析其中的 JSON 数据并返回
    with open(fname, 'r') as f:
        return json.load(f)
```
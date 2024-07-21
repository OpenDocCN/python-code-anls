# `.\pytorch\test\quantization\jit\__init__.py`

```py
# 导入标准库中的 json 模块
import json

# 定义一个函数，接收一个文件名作为参数
def load_json(fname):
    # 打开指定文件名的文件，使用只读模式，并将文件对象赋值给变量 f
    with open(fname, 'r') as f:
        # 使用 json 模块加载并解析文件内容，并将解析后的数据结构返回
        data = json.load(f)
        # 返回解析后的数据结构
        return data
```
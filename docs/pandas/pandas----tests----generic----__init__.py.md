# `D:\src\scipysrc\pandas\pandas\tests\generic\__init__.py`

```
# 导入模块json，用于处理JSON格式数据
import json

# 定义函数read_json，接收一个文件名参数
def read_json(fname):
    # 打开文件，模式为只读，将文件内容加载为JSON对象
    with open(fname, 'r') as f:
        data = json.load(f)
    # 返回加载后的JSON对象
    return data
```
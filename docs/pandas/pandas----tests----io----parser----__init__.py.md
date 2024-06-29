# `D:\src\scipysrc\pandas\pandas\tests\io\parser\__init__.py`

```
# 导入Python标准库中的json模块
import json

# 定义一个函数，名称为load_json，接受一个文件名参数fname
def load_json(fname):
    # 打开指定文件名的文件，并且以只读模式读取文件内容
    with open(fname, 'r') as f:
        # 使用json模块加载文件内容，解析成Python对象
        data = json.load(f)
    # 返回从JSON文件中解析得到的Python对象
    return data
```
# `D:\src\scipysrc\sympy\sympy\printing\tests\__init__.py`

```
# 导入json模块，用于处理JSON格式数据
import json

# 定义一个函数，名称为parse_config，接收一个参数fname表示文件名
def parse_config(fname):
    # 打开文件fname，模式为只读
    with open(fname, 'r') as f:
        # 使用json.load()方法加载JSON文件中的内容，并赋值给变量config
        config = json.load(f)
    # 返回从JSON文件中加载的配置数据
    return config
```
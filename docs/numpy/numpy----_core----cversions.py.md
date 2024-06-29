# `.\numpy\numpy\_core\cversions.py`

```py
"""Simple script to compute the api hash of the current API.

The API has is defined by numpy_api_order and ufunc_api_order.

"""

# 从 os.path 模块中导入 dirname 函数
from os.path import dirname

# 从 code_generators.genapi 模块中导入 fullapi_hash 函数
from code_generators.genapi import fullapi_hash
# 从 code_generators.numpy_api 模块中导入 full_api 对象
from code_generators.numpy_api import full_api

# 如果当前脚本被直接执行
if __name__ == '__main__':
    # 获取当前文件所在目录的路径
    curdir = dirname(__file__)
    # 打印计算出的当前 API 的哈希值
    print(fullapi_hash(full_api))
```
# `.\graphrag\graphrag\index\verbs\graph\merge\defaults.py`

```py
# 版权声明和许可信息，指明代码版权和许可条款
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入自定义的类型提示 BasicMergeOperation
from .typing import BasicMergeOperation

# 定义默认的节点操作字典 DEFAULT_NODE_OPERATIONS
DEFAULT_NODE_OPERATIONS = {
    "*": {
        "operation": BasicMergeOperation.Replace,  # "*" 键对应的操作是替换
    }
}

# 定义默认的边操作字典 DEFAULT_EDGE_OPERATIONS
DEFAULT_EDGE_OPERATIONS = {
    "*": {
        "operation": BasicMergeOperation.Replace,  # "*" 键对应的操作是替换
    },
    "weight": "sum",  # "weight" 键对应的操作是求和
}

# 定义默认的字符串连接分隔符 DEFAULT_CONCAT_SEPARATOR
DEFAULT_CONCAT_SEPARATOR = ","
```
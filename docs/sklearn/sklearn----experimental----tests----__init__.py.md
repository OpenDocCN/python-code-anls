# `D:\src\scipysrc\scikit-learn\sklearn\experimental\tests\__init__.py`

```
# 导入必要的模块：os（操作系统接口）、json（处理 JSON 格式数据）、re（正则表达式操作）
import os
import json
import re

# 定义函数 parse_config，接收一个文件名参数 fname
def parse_config(fname):
    # 初始化配置字典
    config = {}
    # 打开文件 fname 以只读方式
    with open(fname, 'r') as f:
        # 逐行读取文件内容
        for line in f:
            # 使用正则表达式匹配并去除行末的换行符
            line = line.rstrip()
            # 使用正则表达式匹配键值对格式（key=value）
            match = re.match(r'\s*(\w+)\s*=\s*(.+)\s*', line)
            # 如果成功匹配到键值对
            if match:
                # 获取键和值
                key, value = match.groups()
                # 将键值对添加到配置字典中
                config[key] = value
    # 返回解析后的配置字典
    return config
```
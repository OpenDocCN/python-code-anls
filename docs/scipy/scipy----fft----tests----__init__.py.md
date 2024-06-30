# `D:\src\scipysrc\scipy\scipy\fft\tests\__init__.py`

```
# 导入必要的模块：os（操作系统接口）、json（处理 JSON 格式数据）
import os
import json

# 定义函数 write_json(data, filename)，接收数据和文件名作为参数，用于将数据写入 JSON 文件
def write_json(data, filename):
    # 打开文件（以写入模式），使用 utf-8 编码
    with open(filename, 'w', encoding='utf-8') as f:
        # 将数据转换为 JSON 格式并写入文件
        json.dump(data, f, ensure_ascii=False, indent=4)

# 定义函数 read_json(filename)，接收文件名作为参数，用于从 JSON 文件中读取数据
def read_json(filename):
    # 如果文件存在
    if os.path.exists(filename):
        # 打开文件（以读取模式），使用 utf-8 编码
        with open(filename, 'r', encoding='utf-8') as f:
            # 从文件中加载 JSON 数据并返回
            return json.load(f)
    else:
        # 如果文件不存在，则返回空字典
        return {}
```
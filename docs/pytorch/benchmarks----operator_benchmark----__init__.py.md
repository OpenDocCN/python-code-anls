# `.\pytorch\benchmarks\operator_benchmark\__init__.py`

```py
# 导入所需的模块：json用于处理JSON格式数据，os用于操作系统相关功能
import json
import os

# 定义一个函数，用于读取指定路径下的JSON文件并返回解析后的数据
def load_json_file(file_path):
    # 判断文件路径是否存在
    if os.path.exists(file_path):
        # 打开文件，使用utf-8编码读取文件内容，并解析为JSON格式数据
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取文件内容并解析为JSON格式
            data = json.load(f)
            # 返回解析后的JSON数据
            return data
    else:
        # 如果文件路径不存在，则打印错误信息并返回空字典
        print(f"Error: File '{file_path}' not found.")
        return {}
```
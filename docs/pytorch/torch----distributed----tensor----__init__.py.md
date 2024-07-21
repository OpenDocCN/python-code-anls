# `.\pytorch\torch\distributed\tensor\__init__.py`

```py
# 导入所需的模块：json用于处理JSON格式数据，os用于操作文件系统
import json
import os

# 定义一个函数，用于从JSON文件中加载数据并返回Python对象
def load_json_file(filename):
    # 打开指定的JSON文件，模式为只读
    with open(filename, 'r') as f:
        # 使用json模块加载JSON数据并将其解析为Python对象
        data = json.load(f)
    # 返回解析后的Python对象
    return data

# 定义一个函数，用于将数据保存到JSON文件中
def save_json_file(data, filename):
    # 打开指定的JSON文件，模式为写入（如果不存在则创建）
    with open(filename, 'w') as f:
        # 使用json模块将Python对象转换为JSON格式并写入文件
        json.dump(data, f, indent=4)

# 检查当前目录下是否存在指定文件
def check_file_exists(filename):
    # 使用os模块的path.exists函数检查文件是否存在
    if os.path.exists(filename):
        # 如果文件存在，返回True
        return True
    else:
        # 如果文件不存在，返回False
        return False
```
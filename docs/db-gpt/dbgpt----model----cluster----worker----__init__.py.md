# `.\DB-GPT-src\dbgpt\model\cluster\worker\__init__.py`

```py
# 导入所需的模块：os（操作系统相关功能）、json（处理 JSON 格式数据）
import os
import json

# 定义一个函数，用于读取指定路径下的所有 JSON 文件，并将它们的内容合并为一个字典
def merge_json_files(folder_path):
    # 初始化一个空字典，用于存储所有 JSON 文件的内容
    merged_data = {}
    # 遍历指定文件夹下的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        # 遍历当前文件夹下的所有文件
        for file in files:
            # 如果文件名以 '.json' 结尾，则处理该文件
            if file.endswith('.json'):
                # 拼接文件的完整路径
                file_path = os.path.join(root, file)
                # 打开 JSON 文件，读取其中的数据
                with open(file_path, 'r', encoding='utf-8') as f:
                    # 将 JSON 数据加载到一个 Python 字典中
                    data = json.load(f)
                    # 将当前 JSON 文件的内容更新到总的合并字典中
                    merged_data.update(data)
    # 返回合并后的字典数据
    return merged_data
```
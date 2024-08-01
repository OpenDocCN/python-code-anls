# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\4134.d80f15f013b13337.js`

```py
# 导入所需的模块：os（操作系统相关功能）和 re（正则表达式）
import os
import re

# 定义一个函数，用于获取指定目录下的所有文件名
def get_filenames(directory):
    # 使用 os 模块中的 listdir 函数获取指定目录下的所有文件和目录列表
    files = os.listdir(directory)
    # 使用列表推导式，对 files 中的每个元素进行过滤，保留文件名（排除目录名）
    filenames = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    # 返回过滤后的文件名列表
    return filenames

# 定义一个函数，用于根据文件名列表和正则表达式筛选出匹配的文件名
def filter_filenames(filenames, pattern):
    # 使用列表推导式，对 filenames 中的每个文件名应用正则表达式匹配
    filtered = [f for f in filenames if re.match(pattern, f)]
    # 返回匹配成功的文件名列表
    return filtered

# 测试代码
if __name__ == "__main__":
    # 指定目录
    directory = '/path/to/your/directory'
    # 定义正则表达式模式，匹配以 'file_' 开头并且以 '.txt' 结尾的文件名
    pattern = r'^file_.*\.txt$'
    # 获取指定目录下的所有文件名
    filenames = get_filenames(directory)
    # 根据正则表达式模式过滤文件名
    filtered_filenames = filter_filenames(filenames, pattern)
    # 打印过滤后的文件名列表
    print(filtered_filenames)
```
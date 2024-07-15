# `.\Chat-Haruhi-Suzumiya\yuki_builder\audio_feature_ext\tool.py`

```py
# 导入必要的模块：os用于操作文件系统，Counter用于计数
import os
from collections import Counter

# 读取文本文件内容到列表并返回列表
def read_tolist(file, encoding='utf-8'):
    # 使用with语句打开文件，确保文件操作完毕后自动关闭
    with open(file, 'r', encoding=encoding) as f:
        # 读取所有行，去除每行两侧的空白字符，并且过滤掉空行
        lines = f.readlines()
        lines = [item.strip() for item in lines if item.strip()]
    return lines

# 获取一级子目录的绝对路径列表
def get_first_subdir(directory):
    # 使用列表推导式遍历目录中的子目录，返回子目录的绝对路径
    subdirectories = [os.path.abspath(os.path.join(directory, name)) for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return subdirectories

# 获取一级子目录的名称列表，按字母顺序排序
def get_onedir(directory):
    # 使用os.walk函数获取目录下的子目录，返回的是一个生成器，取第一个元素即当前目录下的子目录列表
    subdirectories = next(os.walk(directory))[1]
    subdirectories.sort()  # 对子目录列表进行排序
    return subdirectories

# 获取目录中所有子目录的绝对路径列表
def get_subdir(folder_path):
    # 使用列表推导式遍历目录中的子目录，返回子目录的绝对路径
    subdirectories = [os.path.abspath(os.path.join(folder_path, name)) for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories

# 统计列表中出现频率最高的元素及其出现次数
def most_pre_ele(lst, num=1):
    counter = Counter(lst)  # 使用Counter类统计列表中各元素的出现次数
    pre_lis = counter.most_common(num)  # 返回出现频率最高的num个元素及其出现次数的列表
    pre_ele = counter.most_common(num)[0][0]  # 返回出现频率最高的元素
    return pre_lis, pre_ele

# 获取目录中所有文件的绝对路径列表
def get_filelist(directory):
    file_list = []
    # 使用os.walk函数遍历目录及其子目录中的所有文件，并将文件的绝对路径添加到file_list中
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.startswith('.') and os.path.isfile(file_path):
                file_list.append(file_path)  # 如果文件不是以.开头且是一个普通文件，则将其路径添加到列表中
    file_list.sort()  # 对文件路径列表进行排序
    return file_list

# 获取目录中特定格式文件的绝对路径列表
def get_filelisform(directory, format=None):
    file_list = []
    # 使用os.walk函数遍历目录及其子目录中的所有文件，并根据指定的格式（如果有）筛选文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.startswith('.') and os.path.isfile(file_path):
                if format:
                    if file.endswith(format):
                        file_list.append(file_path)  # 如果文件名以指定格式结尾，则将其路径添加到列表中
                else:
                    file_list.append(file_path)  # 如果没有指定格式，则将所有文件的路径添加到列表中
    file_list.sort()  # 对文件路径列表进行排序
    return file_list

# 从文件中读取一个字典对象
def read_bigone(file):
    with open(file, 'r', encoding='utf-8') as f:
        line = f.readline()  # 读取文件的第一行内容
        line = eval(line.strip())  # 将读取的字符串转换为字典对象
    return line

# 获取目录中所有文件的名称和绝对路径列表
def get_filename(directory, format=None):
    file_list = []
    # 使用os.walk函数遍历目录及其子目录中的所有文件，并根据指定的格式（如果有）筛选文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.startswith('.') and os.path.isfile(file_path):
                if format:
                    if file.endswith(format):
                        file_list.append([file, file_path])  # 如果文件名以指定格式结尾，则将其名称和路径组成列表添加到结果中
                else:
                    file_list.append([file, file_path])  # 如果没有指定格式，则将所有文件的名称和路径组成列表添加到结果中
    file_list.sort()  # 对结果列表进行排序
    return file_list

# 将字符串写入文件
def write_to_file(file, line, mode='w'):
    with open(file, mode=mode, encoding='utf-8') as f:
        f.write(line + '\n')  # 将字符串写入文件，如果文件已存在，则根据模式选择覆盖或追加写入

# 将列表中的每个元素写入文本文件
def save_lis2txt(file, lines):
    with open(file, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(str(line) + '\n')  # 将列表中的每个元素转换为字符串后写入文件，每个元素占据一行
```
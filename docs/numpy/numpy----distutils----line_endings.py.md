# `.\numpy\numpy\distutils\line_endings.py`

```py
# 导入必要的库：os（操作系统相关）、re（正则表达式）、sys（系统相关）
import os
import re
import sys

# 定义函数 dos2unix，用于将 DOS 格式的换行符转换为 UNIX 格式的换行符
def dos2unix(file):
    # 如果传入的文件名是一个目录，则打印出信息并返回
    if os.path.isdir(file):
        print(file, "Directory!")
        return

    # 使用二进制模式打开文件
    with open(file, "rb") as fp:
        data = fp.read()
    
    # 检查文件中是否包含 NULL 字符，若包含则打印出信息并返回
    if b'\0' in data:
        print(file, "Binary!")
        return

    # 将文件中的 CRLF（\r\n）替换为 LF（\n）
    newdata = re.sub(b"\r\n", b"\n", data)

    # 如果替换后的数据与原始数据不同，则表示文件内容已修改
    if newdata != data:
        print('dos2unix:', file)
        # 将修改后的数据写回文件
        with open(file, "wb") as f:
            f.write(newdata)
        return file
    else:
        # 如果文件内容未变化，则打印出文件名和 'ok'
        print(file, 'ok')

# 定义函数 dos2unix_one_dir，用于处理指定目录下的所有文件，并将修改过的文件名加入到 modified_files 列表中
def dos2unix_one_dir(modified_files, dir_name, file_names):
    for file in file_names:
        full_path = os.path.join(dir_name, file)
        # 调用 dos2unix 函数处理每个文件，并将返回值（修改过的文件名）加入到 modified_files 列表中
        file = dos2unix(full_path)
        if file is not None:
            modified_files.append(file)

# 定义函数 dos2unix_dir，用于处理指定目录下的所有文件夹及其文件，并返回所有修改过的文件名列表
def dos2unix_dir(dir_name):
    modified_files = []
    # 递归遍历指定目录，对每个文件夹调用 dos2unix_one_dir 函数处理
    os.path.walk(dir_name, dos2unix_one_dir, modified_files)
    return modified_files
#----------------------------------

# 定义函数 unix2dos，用于将 UNIX 格式的换行符转换为 DOS 格式的换行符
def unix2dos(file):
    # 如果传入的文件名是一个目录，则打印出信息并返回
    if os.path.isdir(file):
        print(file, "Directory!")
        return

    # 使用二进制模式打开文件
    with open(file, "rb") as fp:
        data = fp.read()

    # 检查文件中是否包含 NULL 字符，若包含则打印出信息并返回
    if b'\0' in data:
        print(file, "Binary!")
        return

    # 将文件中的 CRLF（\r\n）替换为 LF（\n）
    newdata = re.sub(b"\r\n", b"\n", data)
    # 将 LF（\n）替换为 CRLF（\r\n）
    newdata = re.sub(b"\n", b"\r\n", newdata)

    # 如果替换后的数据与原始数据不同，则表示文件内容已修改
    if newdata != data:
        print('unix2dos:', file)
        # 将修改后的数据写回文件
        with open(file, "wb") as f:
            f.write(newdata)
        return file
    else:
        # 如果文件内容未变化，则打印出文件名和 'ok'
        print(file, 'ok')

# 定义函数 unix2dos_one_dir，用于处理指定目录下的所有文件，并将修改过的文件名加入到 modified_files 列表中
def unix2dos_one_dir(modified_files, dir_name, file_names):
    for file in file_names:
        full_path = os.path.join(dir_name, file)
        # 调用 unix2dos 函数处理每个文件，并将返回值（修改过的文件名）加入到 modified_files 列表中
        unix2dos(full_path)
        if file is not None:
            modified_files.append(file)

# 定义函数 unix2dos_dir，用于处理指定目录下的所有文件夹及其文件，并返回所有修改过的文件名列表
def unix2dos_dir(dir_name):
    modified_files = []
    # 递归遍历指定目录，对每个文件夹调用 unix2dos_one_dir 函数处理
    os.path.walk(dir_name, unix2dos_one_dir, modified_files)
    return modified_files

# 当脚本被直接执行时，调用 dos2unix_dir 函数，处理传入的第一个参数作为目录名
if __name__ == "__main__":
    dos2unix_dir(sys.argv[1])
```
# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\5880.26da9676624c2c0f.js`

```py
# 导入所需的模块：os 模块用于操作文件系统，subprocess 模块用于执行外部命令
import os
import subprocess

# 定义函数 compress_folder，接收一个文件夹路径作为参数
def compress_folder(folder):
    # 使用 os.path.basename 获取文件夹路径的基本名称（即最后一级目录名）
    folder_name = os.path.basename(folder)
    # 使用 subprocess.Popen 开启一个进程，执行 zip 命令将文件夹压缩成 ZIP 文件
    # 参数 'shell=True' 表示通过 shell 执行命令，命令字符串中的变量会被替换为实际的值
    # 命令字符串中的 {folder_name} 会被实际的文件夹路径替换
    process = subprocess.Popen(f'zip -r {folder_name}.zip {folder}', shell=True)
    # 等待进程执行完成
    process.wait()

# 调用 compress_folder 函数，压缩指定文件夹
compress_folder('/path/to/your/folder')
```
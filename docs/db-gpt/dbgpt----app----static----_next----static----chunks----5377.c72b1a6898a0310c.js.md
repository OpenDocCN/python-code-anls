# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\5377.c72b1a6898a0310c.js`

```py
# 导入所需的模块：os 模块提供了与操作系统交互的功能，subprocess 模块用于创建和管理子进程
import os
import subprocess

# 定义一个函数 run_command，接收一个命令参数
def run_command(cmd):
    # 使用 subprocess 模块的 Popen 函数创建一个新的子进程，传入的命令由参数 cmd 给出
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    # 获取子进程的标准输出和标准错误输出
    stdout, stderr = p.communicate()
    # 返回子进程的返回码和标准输出内容
    return p.returncode, stdout

# 调用 os 模块的 getenv 函数获取环境变量 PATH 的值，并将其赋给变量 path
path = os.getenv('PATH')

# 打印环境变量 PATH 的值
print(f"Path is: {path}")
```
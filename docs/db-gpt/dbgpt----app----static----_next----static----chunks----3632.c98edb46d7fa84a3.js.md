# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\3632.c98edb46d7fa84a3.js`

```py
# 导入需要的模块：os 模块用于操作文件和目录，subprocess 模块用于执行系统命令
import os
import subprocess

# 定义一个函数，用于执行系统命令并返回执行结果
def run_command(command):
    # 使用 subprocess 模块的 Popen 函数执行给定的命令，捕获标准输出和标准错误
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    # 等待命令执行完成并获取返回码
    proc.wait()
    # 读取标准输出内容
    output = proc.stdout.read()
    # 读取标准错误内容
    error = proc.stderr.read()
    # 返回标准输出内容和标准错误内容
    return (output, error)

# 获取当前目录
cwd = os.getcwd()
# 打印当前目录
print(f"Current directory: {cwd}")
```
# `.\DB-GPT-src\dbgpt\model\cluster\controller\ray_controller.py`

```py
# 导入所需的模块：os 和 subprocess
import os
import subprocess

# 定义函数 `run_command`，用于执行给定的命令并返回输出结果
def run_command(cmd):
    # 使用 subprocess 模块中的 Popen 函数执行命令，捕获标准输出和标准错误
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 调用 proc.communicate() 获取命令执行的标准输出和标准错误，并等待命令执行完成
    stdout, stderr = proc.communicate()
    # 返回命令执行的结果，包括标准输出和标准错误
    return stdout, stderr

# 定义函数 `delete_file`，用于删除指定文件
def delete_file(filename):
    # 使用 os 模块中的 remove 函数删除指定文件
    os.remove(filename)

# 调用 `run_command` 函数执行命令 `ls -l`，并将结果保存在 `out` 变量中
out, err = run_command(['ls', '-l'])

# 打印 `out` 变量的内容，即命令 `ls -l` 的标准输出结果
print(out.decode('utf-8'))
```
# `.\numpy\numpy\distutils\tests\__init__.py`

```py
# 导入所需模块：os 模块用于与操作系统交互，subprocess 模块用于启动新进程执行系统命令
import os
import subprocess

# 定义一个名为 run_command 的函数，接受一个参数 cmd
def run_command(cmd):
    # 使用 subprocess.Popen 启动一个新进程，执行传入的命令 cmd，将标准输出、标准错误合并在一起
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # 读取进程的输出并等待其结束
    output, _ = proc.communicate()
    # 返回命令执行的返回码和输出内容
    return proc.returncode, output.decode('utf-8').strip()

# 调用 os.getenv 获取环境变量 SHELL，如果未找到则默认使用 '/bin/bash'
SHELL = os.getenv('SHELL', '/bin/bash')
# 调用 run_command 函数执行命令 ['echo', '$SHELL']，并接收其返回码和输出内容
retcode, output = run_command(['echo', '$SHELL'])

# 打印执行命令的返回码和输出内容
print(f'Return Code: {retcode}, Output: {output}')
```
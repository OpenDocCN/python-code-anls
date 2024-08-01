# `.\DB-GPT-src\dbgpt\agent\resource\tool\tests\__init__.py`

```py
# 导入必要的模块：os 用于操作系统相关功能，subprocess 用于执行外部命令
import os
import subprocess

# 获取当前工作目录，并赋值给变量 cwd
cwd = os.getcwd()

# 使用 subprocess 模块执行系统命令 'ls -l'，并将结果保存在变量 result 中
result = subprocess.run(['ls', '-l'], stdout=subprocess.PIPE)

# 将命令的标准输出结果解码成字符串并打印出来
print(result.stdout.decode('utf-8'))
```
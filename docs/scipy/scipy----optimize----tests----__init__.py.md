# `D:\src\scipysrc\scipy\scipy\optimize\tests\__init__.py`

```
# 导入必要的模块：os 模块用于与操作系统交互，subprocess 模块用于执行外部命令
import os
import subprocess

# 获取当前工作目录，并赋值给变量 cwd
cwd = os.getcwd()

# 使用 subprocess 模块执行 git 命令，通过管道（pipe）连接到 less 命令，将输出分页显示
p = subprocess.Popen(['git', 'log'], stdout=subprocess.PIPE)
subprocess.Popen(['less'], stdin=p.stdout)
p.wait()
```
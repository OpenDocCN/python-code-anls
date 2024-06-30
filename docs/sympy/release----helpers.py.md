# `D:\src\scipysrc\sympy\release\helpers.py`

```
#!/usr/bin/env python

# 导入子进程模块中的 check_call 函数
from subprocess import check_call

# 定义一个函数 run，用于执行命令并获取输出行
def run(*cmdline, cwd=None, env=None):
    """
    在子进程中运行命令并获取输出行
    """
    # 调用 check_call 函数执行命令，指定编码为 utf-8，并传入当前工作目录和环境变量
    return check_call(cmdline, encoding='utf-8', cwd=cwd, env=env)
```
# `.\numpy\benchmarks\asv_pip_nopep517.py`

```
"""
This file is used by asv_compare.conf.json.tpl.
"""
# 导入所需模块：subprocess 用于执行外部命令，sys 用于系统相关操作
import subprocess, sys

# 定义命令列表，使用当前 Python 解释器执行 pip wheel 命令，禁用 pep517 插件
cmd = [sys.executable, '-mpip', 'wheel', '--no-use-pep517']

# 尝试执行命令并捕获输出，将标准错误输出重定向到标准输出
try:
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
except Exception as e:
    # 如果命令执行失败，则将异常输出转换成字符串
    output = str(e.output)

# 检查输出中是否包含 "no such option" 字符串，用于判断是否是旧版本的 pip
if "no such option" in output:
    # 如果是旧版本的 pip，则打印提示信息，并移除命令列表中的 '--no-use-pep517' 选项
    print("old version of pip, escape '--no-use-pep517'")
    cmd.pop()

# 执行命令及其后续参数，使用 subprocess.run() 函数
subprocess.run(cmd + sys.argv[1:])
```
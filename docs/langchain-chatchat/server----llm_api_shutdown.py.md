# `.\Langchain-Chatchat\server\llm_api_shutdown.py`

```py
"""
调用示例：
python llm_api_shutdown.py --serve all
可选"all","controller","model_worker","openai_api_server"， all表示停止所有服务
"""

# 导入必要的模块
import sys
import os

# 将上级目录添加到系统路径中
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 导入子进程模块和参数解析模块
import subprocess
import argparse

# 创建参数解析器
parser = argparse.ArgumentParser()
# 添加参数选项
parser.add_argument("--serve", choices=["all", "controller", "model_worker", "openai_api_server"], default="all")

# 解析命令行参数
args = parser.parse_args()

# 基础的 shell 命令模板
base_shell = "ps -eo user,pid,cmd|grep fastchat.serve{}|grep -v grep|awk '{{print $2}}'|xargs kill -9"

# 根据参数值构建具体的 shell 命令
if args.serve == "all":
    shell_script = base_shell.format("")
else:
    serve = f".{args.serve}"
    shell_script = base_shell.format(serve)

# 执行 shell 命令
subprocess.run(shell_script, shell=True, check=True)
# 打印提示信息
print(f"llm api sever --{args.serve} has been shutdown!")
```
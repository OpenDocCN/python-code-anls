# `.\Langchain-Chatchat\server\webui_allinone_stale.py`

```py
"""
Usage
加载本地模型：
python webui_allinone.py

调用远程api服务：
python webui_allinone.py --use-remote-api

后台运行webui服务：
python webui_allinone.py --nohup

加载多个非默认模型：
python webui_allinone.py --model-path-address model1@host1@port1 model2@host2@port2 

多卡启动：
python webui_alline.py --model-path-address model@host@port --num-gpus 2 --gpus 0,1 --max-gpu-memory 10GiB

"""

# 导入所需的库
import streamlit as st
from webui_pages.utils import *
from streamlit_option_menu import option_menu
from webui_pages import *
import os
from server.llm_api_stale import string_args,launch_all,controller_args,worker_args,server_args,LOG_PATH
from server.api_allinone_stale import parser, api_args
import subprocess

# 添加命令行参数
parser.add_argument("--use-remote-api",action="store_true")
parser.add_argument("--nohup",action="store_true")
parser.add_argument("--server.port",type=int,default=8501)
parser.add_argument("--theme.base",type=str,default='"light"')
parser.add_argument("--theme.primaryColor",type=str,default='"#165dff"')
parser.add_argument("--theme.secondaryBackgroundColor",type=str,default='"#f5f5f5"')
parser.add_argument("--theme.textColor",type=str,default='"#000000"')
web_args = ["server.port","theme.base","theme.primaryColor","theme.secondaryBackgroundColor","theme.textColor"]

# 启动 API 服务
def launch_api(args,args_list=api_args,log_name=None):
    print("Launching api ...")
    print("启动API服务...")
    if not log_name:
        log_name = f"{LOG_PATH}api_{args.api_host}_{args.api_port}"
    print(f"logs on api are written in {log_name}")
    print(f"API日志位于{log_name}下，如启动异常请查看日志")
    args_str = string_args(args,args_list)
    api_sh = "python  server/{script} {args_str} >{log_name}.log 2>&1 &".format(
        script="api.py",args_str=args_str,log_name=log_name)
    subprocess.run(api_sh, shell=True, check=True)
    print("launch api done!")
    print("启动API服务完毕.")

# 启动 webui 服务
def launch_webui(args,args_list=web_args,log_name=None):
    print("Launching webui...")
    print("启动webui服务...")
    # 如果没有指定日志文件名，则使用默认的日志路径和名称
    if not log_name:
        log_name = f"{LOG_PATH}webui"

    # 将参数转换为字符串形式
    args_str = string_args(args, args_list)
    
    # 如果设置了nohup参数，则输出日志信息
    if args.nohup:
        print(f"logs on api are written in {log_name}")
        print(f"webui服务日志位于{log_name}下，如启动异常请查看日志")
        # 构建启动webui服务的shell命令，将输出重定向到日志文件
        webui_sh = "streamlit run webui.py {args_str} >{log_name}.log 2>&1 &".format(
            args_str=args_str, log_name=log_name)
    else:
        # 构建启动webui服务的shell命令
        webui_sh = "streamlit run webui.py {args_str}".format(
            args_str=args_str)
    
    # 使用subprocess模块运行shell命令，确保命令在shell中执行
    subprocess.run(webui_sh, shell=True, check=True)
    
    # 输出启动webui服务完成的信息
    print("launch webui done!")
    print("启动webui服务完毕.")
# 如果当前脚本被作为主程序执行
if __name__ == "__main__":
    # 打印提示信息，表示开始执行 webui_allineone.py，需要一段时间，请耐心等待
    print("Starting webui_allineone.py, it would take a while, please be patient....")
    # 打印提示信息，表示开始执行 webui_allinone.py，启动LLM服务需要约3-10分钟，请耐心等待，同时提供日志路径
    print(f"开始启动webui_allinone,启动LLM服务需要约3-10分钟，请耐心等待，如长时间未启动，请到{LOG_PATH}下查看日志...")
    # 解析命令行参数
    args = parser.parse_args()

    # 打印分隔线
    print("*"*80)
    # 如果不使用远程 API，则启动所有服务
    if not args.use_remote_api:
        launch_all(args=args,controller_args=controller_args,worker_args=worker_args,server_args=server_args)
    # 启动 API 服务
    launch_api(args=args,args_list=api_args)
    # 启动 WebUI 服务
    launch_webui(args=args,args_list=web_args)
    # 打印提示信息，表示 webui_allinone.py 启动完成
    print("Start webui_allinone.py done!")
    # 打印提示信息，表示感谢耐心等待，webui_allinone.py 启动完毕
    print("感谢耐心等待，启动webui_allinone完毕。")
```
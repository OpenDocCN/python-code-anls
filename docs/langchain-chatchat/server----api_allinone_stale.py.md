# `.\Langchain-Chatchat\server\api_allinone_stale.py`

```py
# 用法说明
# 调用默认模型：python server/api_allinone.py
# 加载多个非默认模型：python server/api_allinone.py --model-path-address model1@host1@port1 model2@host2@port2 
# 多卡启动：python server/api_allinone.py --model-path-address model@host@port --num-gpus 2 --gpus 0,1 --max-gpu-memory 10GiB

import sys
import os

# 将当前文件所在目录添加到系统路径中
sys.path.append(os.path.dirname(__file__))
# 将当前文件的上一级目录添加到系统路径中
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 导入模块
from llm_api_stale import launch_all, parser, controller_args, worker_args, server_args
from api import create_app
import uvicorn

# 添加命令行参数
parser.add_argument("--api-host", type=str, default="0.0.0.0")
parser.add_argument("--api-port", type=int, default=7861)
parser.add_argument("--ssl_keyfile", type=str)
parser.add_argument("--ssl_certfile", type=str)

# 定义需要的 API 参数
api_args = ["api-host", "api-port", "ssl_keyfile", "ssl_certfile"]

# 运行 API
def run_api(host, port, **kwargs):
    app = create_app()
    # 如果存在 SSL 密钥文件和证书文件，则使用 SSL 运行
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(app,
                    host=host,
                    port=port,
                    ssl_keyfile=kwargs.get("ssl_keyfile"),
                    ssl_certfile=kwargs.get("ssl_certfile"),
                    )
    else:
        uvicorn.run(app, host=host, port=port)

# 主程序入口
if __name__ == "__main__":
    print("Luanching api_allinone，it would take a while, please be patient...")
    print("正在启动api_allinone，LLM服务启动约3-10分钟，请耐心等待...")
    # 初始化消息
    args = parser.parse_args()
    args_dict = vars(args)
    # 启动所有模块
    launch_all(args=args, controller_args=controller_args, worker_args=worker_args, server_args=server_args)
    # 运行 API
    run_api(
        host=args.api_host,
        port=args.api_port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )
    print("Luanching api_allinone done.")
    print("api_allinone启动完毕.")
```
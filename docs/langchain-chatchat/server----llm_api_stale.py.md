# `.\Langchain-Chatchat\server\llm_api_stale.py`

```
"""
这段代码是一个脚本文件的说明文档，描述了如何调用该脚本以及一些参数的使用说明
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import subprocess
import re
import logging
import argparse

LOG_PATH = "./logs/"
LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

parser = argparse.ArgumentParser()
# ------multi worker-----------------
# 添加参数--model-path-address，用于指定模型路径、主机和端口，格式为model-path@host@port
parser.add_argument('--model-path-address',
                    default="THUDM/chatglm2-6b@localhost@20002",
                    nargs="+",
                    type=str,
                    help="model path, host, and port, formatted as model-path@host@port")
# ---------------controller-------------------------
# 添加控制器相关参数
parser.add_argument("--controller-host", type=str, default="localhost")
parser.add_argument("--controller-port", type=int, default=21001)
parser.add_argument(
    "--dispatch-method",
    type=str,
    choices=["lottery", "shortest_queue"],
    default="shortest_queue",
)
controller_args = ["controller-host", "controller-port", "dispatch-method"]

# ----------------------worker------------------------------------------
# 添加工作节点相关参数
parser.add_argument("--worker-host", type=str, default="localhost")
parser.add_argument("--worker-port", type=int, default=21002)
# parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
# parser.add_argument(
#     "--controller-address", type=str, default="http://localhost:21001"
# )
parser.add_argument(
    "--model-path",
    type=str,
    default="lmsys/vicuna-7b-v1.3",
    help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
)
parser.add_argument(
    "--revision",
    type=str,
    # 设置参数的默认值为"main"
    default="main",
    # 提供关于Hugging Face Hub模型修订标识符的帮助信息
    help="Hugging Face Hub model revision identifier",
# 添加一个名为 device 的参数，类型为字符串，可选值为["cpu", "cuda", "mps", "xpu"]，默认值为 "cuda"，用于指定设备类型
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "cuda", "mps", "xpu"],
    default="cuda",
    help="The device type",
)

# 添加一个名为 gpus 的参数，类型为字符串，默认值为 "0"，用于指定单个 GPU 或多个 GPU，格式如 "0,2"
parser.add_argument(
    "--gpus",
    type=str,
    default="0",
    help="A single GPU like 1 or multiple GPUs like 0,2",
)

# 添加一个名为 num-gpus 的参数，类型为整数，默认值为 1，用于指定 GPU 的数量
parser.add_argument("--num-gpus", type=int, default=1)

# 添加一个名为 max-gpu-memory 的参数，类型为字符串，默认值为 "20GiB"，用于指定每个 GPU 的最大内存
parser.add_argument(
    "--max-gpu-memory",
    type=str,
    default="20GiB",
    help="The maximum memory per gpu. Use a string like '13Gib'",
)

# 添加一个名为 load-8bit 的参数，类型为布尔值，用于指定是否使用 8 位量化
parser.add_argument(
    "--load-8bit", action="store_true", help="Use 8-bit quantization"
)

# 添加一个名为 cpu-offloading 的参数，类型为布尔值，用于指定是否将超出 GPU 内存的权重转移到 CPU
parser.add_argument(
    "--cpu-offloading",
    action="store_true",
    help="Only when using 8-bit quantization: Offload excess weights to the CPU that don't fit on the GPU",
)

# 添加一个名为 gptq-ckpt 的参数，类型为字符串，默认值为 None，用于加载量化模型的路径
parser.add_argument(
    "--gptq-ckpt",
    type=str,
    default=None,
    help="Load quantized model. The path to the local GPTQ checkpoint.",
)

# 添加一个名为 gptq-wbits 的参数，类型为整数，默认值为 16，可选值为[2, 3, 4, 8, 16]，用于指定量化所使用的位数
parser.add_argument(
    "--gptq-wbits",
    type=int,
    default=16,
    choices=[2, 3, 4, 8, 16],
    help="#bits to use for quantization",
)

# 添加一个名为 gptq-groupsize 的参数，类型为整数，默认值为 -1，用于指定量化所使用的组大小
parser.add_argument(
    "--gptq-groupsize",
    type=int,
    default=-1,
    help="Groupsize to use for quantization; default uses full row.",
)

# 添加一个名为 gptq-act-order 的参数，类型为布尔值，用于指定是否应用激活顺序 GPTQ 启发式
parser.add_argument(
    "--gptq-act-order",
    action="store_true",
    help="Whether to apply the activation order GPTQ heuristic",
)

# 添加一个名为 model-names 的参数，类型为函数，用于将逗号分隔的字符串转换为列表
parser.add_argument(
    "--model-names",
    type=lambda s: s.split(","),
    help="Optional display comma separated names",
)

# 添加一个名为 limit-worker-concurrency 的参数，类型为整数，默认值为 5，用于限制模型并发以防止内存溢出
parser.add_argument(
    "--limit-worker-concurrency",
    type=int,
    default=5,
    help="Limit the model concurrency to prevent OOM.",
)

# 添加一个名为 stream-interval 的参数，类型为整数，默认值为 2
parser.add_argument("--stream-interval", type=int, default=2)

# 添加一个名为 no-register 的参数，类型为布尔值
parser.add_argument("--no-register", action="store_true")

# 定义一个包含所有 worker 参数的列表
worker_args = [
    "worker-host", "worker-port",
    "model-path", "revision", "device", "gpus", "num-gpus",
    "max-gpu-memory", "load-8bit", "cpu-offloading",
    "gptq-ckpt", "gptq-wbits", "gptq-groupsize",
    "gptq-act-order", "model-names", "limit-worker-concurrency",
]
    # 定义了四个字符串："stream-interval", "no-register", "controller-address", "worker-address"
# 结束括号，代码中可能存在错误
# -----------------openai server---------------------------

# 添加命令行参数：服务器主机名
parser.add_argument("--server-host", type=str, default="localhost", help="host name")
# 添加命令行参数：服务器端口号
parser.add_argument("--server-port", type=int, default=8888, help="port number")
# 添加命令行参数：是否允许凭据
parser.add_argument(
    "--allow-credentials", action="store_true", help="allow credentials"
)
# 添加命令行参数：API密钥列表
parser.add_argument(
    "--api-keys",
    type=lambda s: s.split(","),
    help="Optional list of comma separated API keys",
)
# 服务器参数列表
server_args = ["server-host", "server-port", "allow-credentials", "api-keys",
               "controller-address"
               ]

# 启动脚本模板，用于启动服务器
base_launch_sh = "nohup python3 -m fastchat.serve.{0} {1} >{2}/{3}.log 2>&1 &"

# 检查脚本模板，用于检查服务器是否正常运行
base_check_sh = """while [ `grep -c "Uvicorn running on" {0}/{1}.log` -eq '0' ];do
                        sleep 5s;
                        echo "wait {2} running"
                done
                echo '{2} running' """

# 将args中的key转化为字符串
def string_args(args, args_list):
    args_str = ""
    # 遍历args对象的关键字参数及其对应的值
    for key, value in args._get_kwargs():
        # 将key中的下划线替换为连字符，以符合指定的args列表格式
        key = key.replace("_", "-")
        # 如果key不在args_list中，则跳过当前循环
        if key not in args_list:
            continue
        # 如果key中包含"port"或"host"，则去除前缀
        key = key.split("-")[-1] if re.search("port|host", key) else key
        # 如果value为空，则跳过当前循环
        if not value:
            pass
        # 如果value为布尔类型且为True，则拼接参数字符串
        elif isinstance(value, bool) and value == True:
            args_str += f" --{key} "
        # 如果value为列表、元组或集合，则将其转换为字符串并拼接参数字符串
        elif isinstance(value, list) or isinstance(value, tuple) or isinstance(value, set):
            value = " ".join(value)
            args_str += f" --{key} {value} "
        # 其他情况下，拼接参数字符串
        else:
            args_str += f" --{key} {value} "

    # 返回拼接好的参数字符串
    return args_str
def launch_worker(item, args, worker_args=worker_args):
    # 从item中提取日志名称，替换特殊字符为下划线
    log_name = item.split("/")[-1].split("\\")[-1].replace("-", "_").replace("@", "_").replace(".", "_")
    # 将item按@符号分割，分别赋值给model_path、worker_host、worker_port
    args.model_path, args.worker_host, args.worker_port = item.split("@")
    # 拼接worker_address
    args.worker_address = f"http://{args.worker_host}:{args.worker_port}"
    print("*" * 80)
    print(f"如长时间未启动，请到{LOG_PATH}{log_name}.log下查看日志")
    # 将args和worker_args转换为字符串参数
    worker_str_args = string_args(args, worker_args)
    print(worker_str_args)
    # 根据模板生成worker启动脚本
    worker_sh = base_launch_sh.format("model_worker", worker_str_args, LOG_PATH, f"worker_{log_name}")
    # 根据模板生成worker检查脚本
    worker_check_sh = base_check_sh.format(LOG_PATH, f"worker_{log_name}", "model_worker")
    # 执行worker启动脚本
    subprocess.run(worker_sh, shell=True, check=True)
    # 执行worker检查脚本
    subprocess.run(worker_check_sh, shell=True, check=True)


def launch_all(args,
               controller_args=controller_args,
               worker_args=worker_args,
               server_args=server_args
               ):
    print(f"Launching llm service,logs are located in {LOG_PATH}...")
    print(f"开始启动LLM服务,请到{LOG_PATH}下监控各模块日志...")
    # 将args和controller_args转换为字符串参数
    controller_str_args = string_args(args, controller_args)
    # 根据模板生成controller启动脚本
    controller_sh = base_launch_sh.format("controller", controller_str_args, LOG_PATH, "controller")
    # 根据模板生成controller检查脚本
    controller_check_sh = base_check_sh.format(LOG_PATH, "controller", "controller")
    # 执行controller启动脚本
    subprocess.run(controller_sh, shell=True, check=True)
    # 执行controller检查脚本
    subprocess.run(controller_check_sh, shell=True, check=True)
    print(f"worker启动时间视设备不同而不同，约需3-10分钟，请耐心等待...")
    # 如果model_path_address是字符串，则启动单个worker
    if isinstance(args.model_path_address, str):
        launch_worker(args.model_path_address, args=args, worker_args=worker_args)
    else:
        # 启动多个worker
        for idx, item in enumerate(args.model_path_address):
            print(f"开始加载第{idx}个模型:{item}")
            launch_worker(item, args=args, worker_args=worker_args)

    # 将args和server_args转换为字符串参数
    server_str_args = string_args(args, server_args)
    # 根据模板生成openai_api_server启动脚本
    server_sh = base_launch_sh.format("openai_api_server", server_str_args, LOG_PATH, "openai_api_server")
    # 根据 base_check_sh 格式化字符串，生成检查服务器脚本的命令
    server_check_sh = base_check_sh.format(LOG_PATH, "openai_api_server", "openai_api_server")
    # 运行检查服务器脚本的命令，以 shell 模式运行，确保执行成功
    subprocess.run(server_sh, shell=True, check=True)
    # 运行检查服务器脚本的命令，以 shell 模式运行，确保执行成功
    subprocess.run(server_check_sh, shell=True, check=True)
    # 打印提示信息，表示LLM服务启动完成
    print("Launching LLM service done!")
    # 打印提示信息，表示LLM服务启动完毕
    print("LLM服务启动完毕。")
# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 解析命令行参数并存储在args中
    args = parser.parse_args()
    # 将controller_host和controller_port组合成controller-address，并添加http://前缀
    args = argparse.Namespace(**vars(args),
                              **{"controller-address": f"http://{args.controller_host}:{str(args.controller_port)}"})

    # 如果指定了GPU
    if args.gpus:
        # 检查指定的GPU数量是否小于num_gpus
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        # 设置CUDA_VISIBLE_DEVICES环境变量为指定的GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # 启动所有任务
    launch_all(args=args)
```
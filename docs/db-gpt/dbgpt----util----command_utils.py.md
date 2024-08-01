# `.\DB-GPT-src\dbgpt\util\command_utils.py`

```py
# 导入必要的模块
import os  # 提供与操作系统交互的功能
import platform  # 获取平台相关信息
import subprocess  # 启动子进程执行外部命令
import sys  # 提供对Python解释器的访问和控制
from functools import lru_cache  # 提供LRU缓存装饰器，用于函数结果的缓存
from typing import Dict, List  # 提供静态类型标注支持

import psutil  # 提供系统进程和系统利用率信息的接口


def _get_abspath_of_current_command(command_path: str):
    # 如果命令路径不是以.py结尾，则直接返回该路径
    if not command_path.endswith(".py"):
        return command_path
    # 将命令路径拼接成绝对路径，这里将上一级和上上级目录的路径加入
    command_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "cli_scripts.py",
    )
    return command_path


def _run_current_with_daemon(name: str, log_file: str):
    # 获取除了--daemon和-d之外的所有参数
    args = [arg for arg in sys.argv if arg != "--daemon" and arg != "-d"]
    # 将第一个参数替换为当前命令的绝对路径
    args[0] = _get_abspath_of_current_command(args[0])

    # 构建后台运行的命令行
    daemon_cmd = [sys.executable] + args
    daemon_cmd = " ".join(daemon_cmd)
    daemon_cmd += f" > {log_file} 2>&1"

    print(f"daemon cmd: {daemon_cmd}")
    
    # 根据平台设置不同的启动方式
    if "windows" in platform.system().lower():
        # 在Windows系统下启动子进程，创建新的进程组
        process = subprocess.Popen(
            daemon_cmd,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True,
        )
    else:  # macOS, Linux, and other Unix-like systems
        # 在Unix-like系统下启动子进程，并将进程设置为新的进程组的领导者
        process = subprocess.Popen(
            daemon_cmd,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True,
        )

    print(f"Started {name} in background with pid: {process.pid}")


def _run_current_with_gunicorn(app: str, config_path: str, kwargs: Dict):
    try:
        import gunicorn  # 尝试导入gunicorn包
    except ImportError as e:
        # 如果导入失败，则抛出异常
        raise ValueError(
            "Could not import python package: gunicorn"
            "Daemon mode need install gunicorn, please install `pip install gunicorn`"
        ) from e

    from dbgpt.util.parameter_utils import EnvArgumentParser

    env_to_app = {}
    env_to_app.update(os.environ)
    # 将传入的kwargs参数转换为环境变量的键值对
    app_env = EnvArgumentParser._kwargs_to_env_key_value(kwargs)
    env_to_app.update(app_env)
    cmd = f"uvicorn {app} --host 0.0.0.0 --port 5670"
    if "windows" in platform.system().lower():
        # Windows系统不支持使用gunicorn启动服务，抛出异常
        raise Exception("Not support on windows")
    else:  # macOS, Linux, and other Unix-like systems
        # 在Unix-like系统下使用subprocess启动uvicorn服务
        process = subprocess.Popen(cmd, shell=True, env=env_to_app)
    print(f"Started {app} with gunicorn in background with pid: {process.pid}")


def _stop_service(
    key: str, fullname: str, service_keys: List[str] = None, port: int = None
):
    # 如果未提供service_keys，则使用默认值构建参数列表
    if not service_keys:
        service_keys = [sys.argv[0], "start", key]
    # 初始化未找到标志
    not_found = True
    # 遍历所有正在运行的进程
    for process in psutil.process_iter(attrs=["pid", "datasource", "cmdline"]):
        try:
            # 获取进程的命令行参数并组合成字符串
            cmdline = " ".join(process.info["cmdline"])

            # 检查命令行参数中是否包含所有的服务关键字
            if all(fragment in cmdline for fragment in service_keys):
                # 如果指定了端口号
                if port:
                    # 遍历进程的数据源连接
                    for conn in process.info["datasource"]:
                        # 如果连接状态为监听，并且端口号匹配
                        if (
                            conn.status == psutil.CONN_LISTEN
                            and conn.laddr.port == port
                        ):
                            # 终止该进程
                            psutil.Process(process.info["pid"]).terminate()
                            # 输出终止信息
                            print(
                                f"Terminated the {fullname} with PID: {process.info['pid']} listening on port: {port}"
                            )
                            # 找到了符合条件的进程，标记为非未找到
                            not_found = False
                else:
                    # 没有指定端口号时，直接终止进程
                    psutil.Process(process.info["pid"]).terminate()
                    # 输出终止信息
                    print(f"Terminated the {fullname} with PID: {process.info['pid']}")
                    # 找到了符合条件的进程，标记为非未找到
                    not_found = False
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # 如果捕获到不存在的进程或访问被拒绝异常，则继续处理下一个进程
            continue

    # 如果未找到符合条件的进程，则输出未找到信息
    if not_found:
        print(f"{fullname} process not found.")
# 返回所有包含给定服务关键字的进程的监听端口列表，优先选择8000和5000端口，其次按升序排列。
def _get_ports_by_cmdline_part(service_keys: List[str]) -> List[int]:
    ports = []  # 初始化空列表，用于存储找到的端口号

    # 遍历当前所有进程，获取包含指定参数的进程信息
    for process in psutil.process_iter(attrs=["pid", "name", "cmdline", "connections"]):
        try:
            # 将进程的命令行参数列表转换为单个字符串，以便更方便地检查
            cmdline = ""
            if process.info.get("cmdline"):
                cmdline = " ".join(process.info["cmdline"])

            # 检查命令行中是否包含所有的服务关键字
            if cmdline and all(fragment in cmdline for fragment in service_keys):
                connections = process.info.get("connections")
                # 如果存在连接信息并且端口列表为空，则遍历连接，找到监听状态的端口号
                if connections is not None and len(ports) == 0:
                    for connection in connections:
                        if connection.status == psutil.CONN_LISTEN:
                            ports.append(connection.laddr.port)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    # 对端口列表进行排序，优先放置8000和5670端口
    ports.sort(key=lambda x: (x != 8000, x != 5670, x))
    return ports


# 使用LRU缓存装饰器，检测控制器地址并返回
def _detect_controller_address() -> str:
    # 获取环境变量中的控制器地址
    controller_addr = os.getenv("CONTROLLER_ADDRESS")
    if controller_addr:
        return controller_addr

    # 定义可能的控制器命令行片段列表
    cmdline_fragments = [
        ["python", "start", "controller"],
        ["python", "controller"],
        ["python", "start", "webserver"],
        ["python", "dbgpt_server"],
    ]

    # 遍历每个命令行片段，调用_get_ports_by_cmdline_part函数获取相关端口，并返回第一个端口对应的地址
    for fragments in cmdline_fragments:
        ports = _get_ports_by_cmdline_part(fragments)
        if ports:
            return f"http://127.0.0.1:{ports[0]}"

    # 如果未找到匹配的端口，返回默认的控制器地址
    return f"http://127.0.0.1:8000"
```
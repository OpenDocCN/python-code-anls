# `arknights-mower\arknights_mower\utils\simulator.py`

```
# 导入子进程模块
import subprocess
# 导入枚举类型模块
from enum import Enum
# 从自定义日志模块中导入日志对象
from arknights_mower.utils.log import logger
# 导入时间模块
import time

# 定义模拟器类型的枚举
class Simulator_Type(Enum):
    Nox = "夜神"
    MuMu12 = "MuMu12"

# 重启模拟器函数，接收一个包含模拟器索引和名称的数据字典
def restart_simulator(data):
    # 获取模拟器索引和名称
    index = data["index"]
    simulator_type = data["name"]
    cmd = ""
    # 判断模拟器类型是否为夜神或MuMu12
    if simulator_type in [Simulator_Type.Nox.value, Simulator_Type.MuMu12.value]:
        # 如果模拟器类型为夜神
        if simulator_type == Simulator_Type.Nox.value:
            # 设置命令为启动夜神模拟器
            cmd = "Nox.exe"
            # 如果索引大于等于0，添加克隆命令
            if index >= 0:
                cmd += f' -clone:Nox_{data["index"]}'
            # 添加关闭命令
            cmd += " -quit"
        # 如果模拟器类型为MuMu12
        elif simulator_type == Simulator_Type.MuMu12.value:
            # 设置命令为关闭MuMu12模拟器
            cmd = "MuMuManager.exe api -v "
            # 如果索引大于等于0，添加索引参数
            if index >= 0:
                cmd += f'{data["index"]} '
            # 添加关闭玩家命令
            cmd += "shutdown_player"
        # 执行命令，并指定模拟器文件夹路径
        exec_cmd(cmd, data["simulator_folder"])
        # 记录日志，提示开始关闭模拟器，并等待2秒钟
        logger.info(f'开始关闭{simulator_type}模拟器，等待2秒钟')
        time.sleep(2)
        # 如果模拟器类型为夜神
        if simulator_type == Simulator_Type.Nox.value:
            # 移除关闭命令，变为启动命令
            cmd = cmd.replace(' -quit', '')
        # 如果模拟器类型为MuMu12
        elif simulator_type == Simulator_Type.MuMu12.value:
            # 移除关闭玩家命令，变为启动玩家命令
            cmd = cmd.replace(' shutdown_player', ' launch_player')
        # 再次执行命令，并指定模拟器文件夹路径
        exec_cmd(cmd, data["simulator_folder"])
        # 记录日志，提示开始启动模拟器，并等待25秒钟
        logger.info(f'开始启动{simulator_type}模拟器，等待25秒钟')
        time.sleep(25)
    # 如果模拟器类型不是夜神或MuMu12
    else:
        # 记录警告日志，提示尚未支持该模拟器类型的重启/自动启动
        logger.warning(f"尚未支持{simulator_type}重启/自动启动")

# 执行命令函数，接收命令和文件夹路径作为参数
def exec_cmd(cmd, folder_path):
    try:
        # 使用子进程执行命令，指定工作目录和输出流
        process = subprocess.Popen(cmd, shell=True, cwd=folder_path, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   universal_newlines=True)
        # 等待命令执行，超时时间为2秒
        process.communicate(timeout=2)
    # 如果命令执行超时
    except subprocess.TimeoutExpired:
        # 终止进程
        process.kill()
```
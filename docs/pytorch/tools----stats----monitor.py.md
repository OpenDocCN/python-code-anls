# `.\pytorch\tools\stats\monitor.py`

```py
#!/usr/bin/env python3
# 定义脚本使用的 Python 解释器

from __future__ import annotations
# 导入未来的注解支持，用于声明类型提示中的自引用类型

import datetime
# 导入处理日期时间的模块
import json
# 导入处理 JSON 数据的模块
import signal
# 导入处理信号的模块
import time
# 导入处理时间的模块
from typing import Any
# 导入 Any 类型，用于灵活的类型注解

import psutil  # type: ignore[import]
# 导入 psutil 库来获取系统进程和系统利用率信息

def get_processes_running_python_tests() -> list[Any]:
    # 获取当前正在运行的 Python 测试相关进程列表
    python_processes = []
    for process in psutil.process_iter():
        try:
            if "python" in process.name() and process.cmdline():
                # 如果进程名称中包含 "python" 并且有命令行参数，则将其加入列表
                python_processes.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # 捕获进程不存在或访问被拒绝的异常，继续处理下一个进程
            pass
    return python_processes

def get_per_process_cpu_info() -> list[dict[str, Any]]:
    # 获取每个进程的 CPU 使用情况信息
    processes = get_processes_running_python_tests()
    per_process_info = []
    for p in processes:
        info = {
            "pid": p.pid,
            "cmd": " ".join(p.cmdline()),
            "cpu_percent": p.cpu_percent(),
            "rss_memory": p.memory_info().rss,
        }

        # 尝试获取完整的内存信息，可能会抛出 AccessDenied 异常（例如在 macOS）
        try:
            memory_full_info = p.memory_full_info()

            info["uss_memory"] = memory_full_info.uss
            if "pss" in memory_full_info:
                # 仅在 Linux 系统中可用的内存信息
                info["pss_memory"] = memory_full_info.pss

        except psutil.AccessDenied as e:
            # 如果访问被拒绝，可以跳过这部分信息
            pass

        per_process_info.append(info)
    return per_process_info

def get_per_process_gpu_info(handle: Any) -> list[dict[str, Any]]:
    # 获取每个 GPU 运行进程的信息
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    per_process_info = []
    for p in processes:
        info = {"pid": p.pid, "gpu_memory": p.usedGpuMemory}
        per_process_info.append(info)
    return per_process_info

def rocm_get_per_process_gpu_info(handle: Any) -> list[dict[str, Any]]:
    # 使用 ROCm 获取每个 GPU 上的进程信息
    processes = amdsmi.amdsmi_get_gpu_process_list(handle)
    per_process_info = []
    for p in processes:
        try:
            proc_info = amdsmi.amdsmi_get_gpu_process_info(handle, p)
        except AttributeError:
            # 如果 AMDsmi 不支持该 API，则使用备选方法处理
            proc_info = p
        info = {
            "pid": proc_info["pid"],
            "gpu_memory": proc_info["memory_usage"]["vram_mem"],
        }
        per_process_info.append(info)
    return per_process_info

if __name__ == "__main__":
    handle = None
    try:
        import pynvml  # type: ignore[import]

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # 初始化并获取第一个 GPU 的句柄
        except pynvml.NVMLError:
            # 如果出现 NVML 错误，则忽略
            pass
    except ModuleNotFoundError:
        # 如果找不到 pynvml 模块，可能是因为没有 CUDA 支持
        pass
    try:
        import amdsmi  # type: ignore[import]  # 尝试导入amdsmi模块，忽略类型检查

        try:
            amdsmi.amdsmi_init()  # 初始化amdsmi
            amdsmi_handle = amdsmi.amdsmi_get_processor_handles()[0]  # 获取第一个处理器句柄
        except amdsmi.AmdSmiException:
            pass  # 捕获AmdSmiException异常，不做处理
    except ModuleNotFoundError:
        # 若未找到amdsmi模块，执行以下代码块
        # no amdsmi is available
        pass  # 什么都不做，继续执行后续代码

    kill_now = False  # 初始化一个标志位，用于控制循环退出

    def exit_gracefully(*args: Any) -> None:
        global kill_now  # 声明使用全局变量kill_now
        kill_now = True  # 设置kill_now为True，表示需要退出循环

    signal.signal(signal.SIGTERM, exit_gracefully)  # 设置信号处理函数，当收到SIGTERM信号时调用exit_gracefully

    while not kill_now:  # 循环，直到kill_now为True时退出
        try:
            stats = {
                "time": datetime.datetime.utcnow().isoformat("T") + "Z",  # 获取当前时间的ISO格式字符串
                "total_cpu_percent": psutil.cpu_percent(),  # 获取当前CPU的总使用率
                "per_process_cpu_info": get_per_process_cpu_info(),  # 获取每个进程的CPU信息
            }
            if handle is not None:  # 如果存在handle对象
                stats["per_process_gpu_info"] = get_per_process_gpu_info(handle)  # 获取每个进程的GPU信息
                # https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html
                gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)  # 获取GPU利用率
                stats["total_gpu_utilization"] = gpu_utilization.gpu  # 总GPU利用率
                stats["total_gpu_mem_utilization"] = gpu_utilization.memory  # GPU内存利用率
            if amdsmi_handle is not None:  # 如果存在amdsmi_handle对象
                stats["per_process_gpu_info"] = rocm_get_per_process_gpu_info(
                    amdsmi_handle
                )  # 获取每个进程的GPU信息
                stats["total_gpu_utilization"] = amdsmi.amdsmi_get_gpu_activity(
                    amdsmi_handle
                )["gfx_activity"]  # 获取AMD GPU的总体利用率
                stats["total_gpu_mem_utilization"] = amdsmi.amdsmi_get_gpu_activity(
                    amdsmi_handle
                )["umc_activity"]  # 获取AMD GPU的总体内存利用率
        except Exception as e:
            stats = {
                "time": datetime.datetime.utcnow().isoformat("T") + "Z",  # 获取当前时间的ISO格式字符串
                "error": str(e),  # 将异常信息转换成字符串
            }
        finally:
            print(json.dumps(stats))  # 将stats字典转换成JSON格式并打印输出
            time.sleep(1)  # 等待1秒钟，继续下一次循环
```
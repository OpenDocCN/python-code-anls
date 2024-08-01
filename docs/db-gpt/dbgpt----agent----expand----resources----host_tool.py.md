# `.\DB-GPT-src\dbgpt\agent\expand\resources\host_tool.py`

```py
# 主机工具资源模块

# 导入工具基类
from ...resource.tool.base import tool

# 定义获取当前主机CPU状态的工具函数，并添加描述
@tool(description="Get current host CPU status.")
def get_current_host_cpu_status() -> str:
    """Get current host CPU status."""
    # 导入平台信息模块
    import platform
    # 导入系统性能监控模块
    import psutil

    # 获取CPU架构信息
    cpu_architecture = platform.machine()
    # 获取物理CPU核心数
    cpu_count_physical = psutil.cpu_count(logical=False)
    # 获取逻辑CPU核心数
    cpu_count_logical = psutil.cpu_count(logical=True)
    # 获取CPU使用率，间隔为1秒
    cpu_usage = psutil.cpu_percent(interval=1)

    # 构造并返回CPU状态信息字符串
    return (
        f"CPU Architecture: {cpu_architecture}\n"
        f"Physical CPU Cores: {cpu_count_physical}\n"
        f"Logical CPU Cores: {cpu_count_logical}\n"
        f"CPU Usage: {cpu_usage}%"
    )


# 定义获取当前主机内存状态的工具函数，并添加描述
@tool(description="Get current host memory status.")
def get_current_host_memory_status() -> str:
    """Get current host memory status."""
    # 导入系统内存信息模块
    import psutil

    # 获取系统内存信息
    memory = psutil.virtual_memory()

    # 构造并返回内存状态信息字符串，以GB为单位
    return (
        f"Total:  {memory.total / (1024**3):.2f} GB\n"
        f"Available: {memory.available / (1024**3):.2f} GB\n"
        f"Used:  {memory.used / (1024**3):.2f} GB\n"
        f"Percent: {memory.percent}%"
    )


# 定义获取当前主机系统负载状态的工具函数，并添加描述
@tool(description="Get current host system load.")
def get_current_host_system_load() -> str:
    """Get current host system load."""
    # 导入操作系统相关模块
    import os

    # 获取系统负载平均值，分别为1分钟、5分钟和15分钟的负载
    load1, load5, load15 = os.getloadavg()

    # 构造并返回系统负载信息字符串
    return f"System load average: {load1:.2f}, {load5:.2f}, {load15:.2f}"
```
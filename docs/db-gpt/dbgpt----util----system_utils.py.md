# `.\DB-GPT-src\dbgpt\util\system_utils.py`

```py
# 导入所需的标准库和第三方模块
import os  # 提供与操作系统交互的功能
import platform  # 提供访问平台相关属性的功能
import re  # 提供正则表达式匹配操作
import subprocess  # 提供运行外部命令和获取输出的功能
from dataclasses import asdict, dataclass  # dataclass 模块提供创建数据类的功能
from enum import Enum  # 提供创建枚举类的功能
from functools import cache  # 提供缓存函数调用结果的功能
from typing import Dict, Tuple  # 提供类型提示支持


@dataclass
class SystemInfo:
    platform: str  # 操作系统平台名称
    distribution: str  # 操作系统发行版信息
    python_version: str  # Python 解释器版本
    cpu: str  # CPU 型号
    cpu_avx: str  # CPU 支持的 AVX 指令集类型
    memory: str  # 系统内存信息
    torch_version: str  # PyTorch 版本
    device: str  # 计算设备类型 (CPU/GPU/MPS)
    device_version: str  # 计算设备版本信息
    device_count: int  # 可用的计算设备数量
    device_other: str  # 其他的设备相关信息

    def to_dict(self) -> Dict:
        return asdict(self)  # 将数据类转换为字典形式


class AVXType(Enum):
    BASIC = "basic"  # 基本的 AVX 支持
    AVX = "AVX"  # AVX 指令集
    AVX2 = "AVX2"  # AVX2 指令集
    AVX512 = "AVX512"  # AVX512 指令集

    @staticmethod
    def of_type(avx: str):
        for item in AVXType:
            if item._value_ == avx:
                return item
        return None


class OSType(str, Enum):
    WINDOWS = "win"  # Windows 操作系统
    LINUX = "linux"  # Linux 操作系统
    DARWIN = "darwin"  # macOS 操作系统
    OTHER = "other"  # 其他操作系统


def get_cpu_avx_support() -> Tuple[OSType, AVXType, str]:
    system = platform.system()  # 获取当前操作系统名称
    os_type = OSType.OTHER  # 默认操作系统类型为其他
    cpu_avx = AVXType.BASIC  # 默认 CPU 支持的 AVX 类型为基本
    env_cpu_avx = AVXType.of_type(os.getenv("DBGPT_LLAMA_CPP_AVX"))  # 获取环境变量中的 AVX 类型
    distribution = "Unknown Distribution"  # 默认操作系统发行版信息为未知

    # 根据不同操作系统类型执行相应的操作
    if "windows" in system.lower():
        os_type = OSType.WINDOWS  # 设置操作系统类型为 Windows
        output = "avx2"  # 默认使用 AVX2 作为 CPU 架构
        distribution = "Windows " + platform.release()  # 获取 Windows 版本信息
        print("Current platform is windows, use avx2 as default cpu architecture")
    elif system == "Linux":
        os_type = OSType.LINUX  # 设置操作系统类型为 Linux
        result = subprocess.run(
            ["lscpu"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )  # 执行 lscpu 命令获取 CPU 信息
        output = result.stdout.decode()  # 解码命令输出
        distribution = get_linux_distribution()  # 获取 Linux 发行版信息
    elif system == "Darwin":
        os_type = OSType.DARWIN  # 设置操作系统类型为 macOS
        result = subprocess.run(
            ["sysctl", "-a"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )  # 执行 sysctl -a 命令获取系统信息
        distribution = "Mac OS " + platform.mac_ver()[0]  # 获取 macOS 版本信息
        output = result.stdout.decode()  # 解码命令输出
    else:
        os_type = OSType.OTHER  # 设置操作系统类型为其他
        print("Unsupported OS to get cpu avx, use default")  # 输出不支持的操作系统信息
        return os_type, env_cpu_avx if env_cpu_avx else cpu_avx, distribution

    # 根据命令输出判断 CPU 支持的 AVX 类型
    if "avx512" in output.lower():
        cpu_avx = AVXType.AVX512  # 设置 AVX512 指令集类型
    elif "avx2" in output.lower():
        cpu_avx = AVXType.AVX2  # 设置 AVX2 指令集类型
    elif "avx " in output.lower():
        # cpu_avx =  AVXType.AVX
        pass
    return os_type, env_cpu_avx if env_cpu_avx else cpu_avx, distribution  # 返回操作系统类型、CPU AVX 类型和发行版信息


def get_device() -> str:
    try:
        import torch

        return (
            "cuda"
            if torch.cuda.is_available()  # 如果 CUDA 可用，则返回 cuda
            else "mps"
            if torch.backends.mps.is_available()  # 如果 MPS 可用，则返回 mps
            else "cpu"  # 否则返回 cpu
        )
    except ModuleNotFoundError:
        return "cpu"  # 如果 torch 模块未找到，则返回 cpu


def get_device_info() -> Tuple[str, str, str, int, str]:
    torch_version, device, device_version, device_count, device_other = (
        None,
        "cpu",  # 初始化默认的计算设备类型为 cpu
        None,
        0,
        "",  # 初始化其他设备信息为空字符串
    )
    try:
        import torch  # 尝试导入 torch 库

        torch_version = torch.__version__  # 获取 torch 库的版本信息
        if torch.cuda.is_available():  # 检查是否支持 CUDA
            device = "cuda"  # 如果支持 CUDA，设备类型为 "cuda"
            device_version = torch.version.cuda  # 获取 CUDA 版本
            device_count = torch.cuda.device_count()  # 获取 CUDA 设备数量
        elif torch.backends.mps.is_available():  # 检查是否支持 MPS
            device = "mps"  # 如果支持 MPS，设备类型为 "mps"
    except ModuleNotFoundError:
        pass  # 如果出现 ModuleNotFoundError 异常，忽略继续执行

    if not device_version:  # 如果设备版本未定义
        device_version = (
            get_cuda_version_from_nvcc() or get_cuda_version_from_nvidia_smi()
        )  # 尝试从 nvcc 或者 nvidia-smi 获取 CUDA 版本

    if device == "cuda":  # 如果设备类型为 "cuda"
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",  # 调用 nvidia-smi 命令
                    "--query-gpu=name,driver_version,memory.total,memory.free,memory.used",  # 查询 GPU 相关信息
                    "--format=csv",  # 指定输出格式为 CSV
                ]
            )
            device_other = output.decode("utf-8")  # 解码命令输出为 UTF-8 格式字符串
        except:
            pass  # 如果执行 nvidia-smi 命令异常，则忽略

    return torch_version, device, device_version, device_count, device_other  # 返回 torch 版本、设备类型、设备版本、CUDA 设备数量、其他设备信息
# 从 nvcc 工具获取 CUDA 版本号
def get_cuda_version_from_nvcc():
    try:
        # 执行 nvcc --version 命令，获取输出结果
        output = subprocess.check_output(["nvcc", "--version"])
        # 从输出中找到包含 "release" 的行，通常这行包含版本信息
        version_line = [
            line for line in output.decode("utf-8").split("\n") if "release" in line
        ][0]
        # 提取版本号并返回
        return version_line.split("release")[-1].strip().split(",")[0]
    except:
        # 出错时返回 None
        return None


# 从 nvidia-smi 命令获取 CUDA 版本号
def get_cuda_version_from_nvidia_smi():
    try:
        # 执行 nvidia-smi 命令并获取输出
        output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        # 使用正则表达式匹配 CUDA 版本号
        match = re.search(r"CUDA Version:\s+(\d+\.\d+)", output)
        if match:
            # 返回匹配到的 CUDA 版本号
            return match.group(1)
        else:
            # 如果未匹配到则返回 None
            return None
    except:
        # 出错时返回 None
        return None


# 获取当前 Linux 发行版的信息
def get_linux_distribution():
    """Get distribution of Linux"""
    if os.path.isfile("/etc/os-release"):
        # 如果 /etc/os-release 文件存在，则打开并读取信息
        with open("/etc/os-release", "r") as f:
            info = {}
            for line in f:
                key, _, value = line.partition("=")
                info[key] = value.strip().strip('"')
            # 组合并返回 Linux 发行版及版本号信息
            return f"{info.get('NAME', 'Unknown')} {info.get('VERSION_ID', '')}".strip()
    # 如果文件不存在或出现其他错误，则返回未知 Linux 发行版
    return "Unknown Linux Distribution"


# 获取 CPU 的详细信息
def get_cpu_info():
    # 获取操作系统类型、AVX 支持情况及 Linux 发行版信息
    os_type, avx_type, distribution = get_cpu_avx_support()

    # 初始化 CPU 信息为未知
    cpu_info = "Unknown CPU"
    # 根据操作系统类型获取 CPU 信息
    if os_type == OSType.LINUX:
        try:
            # 执行 lscpu 命令获取 CPU 信息
            output = subprocess.check_output(["lscpu"]).decode("utf-8")
            # 使用正则表达式匹配 CPU 型号名称
            match = re.search(r".*Model name:\s*(.+)", output)
            if match:
                cpu_info = match.group(1).strip()
            # 匹配其他可能的 CPU 信息（根据具体情况添加）
            match = re.search(r".*型号名称：\s*(.+)", output)
            if match:
                cpu_info = match.group(1).strip()
        except:
            pass
    elif os_type == OSType.DARWIN:
        try:
            # 在 macOS 上执行 sysctl 命令获取 CPU 型号字符串
            output = subprocess.check_output(["sysctl", "machdep.cpu.brand_string"]).decode("utf-8")
            # 使用正则表达式匹配 CPU 型号字符串
            match = re.search(r"machdep.cpu.brand_string:\s*(.+)", output)
            if match:
                cpu_info = match.group(1).strip()
        except:
            pass
    elif os_type == OSType.WINDOWS:
        try:
            # 在 Windows 上执行 wmic cpu get Name 命令获取 CPU 名称信息
            output = subprocess.check_output("wmic cpu get Name", shell=True).decode("utf-8")
            lines = output.splitlines()
            # 解析命令输出并获取 CPU 名称信息
            cpu_info = lines[2].split(":")[-1].strip()
        except:
            pass

    # 返回操作系统类型、AVX 支持情况、CPU 信息及 Linux 发行版信息
    return os_type, avx_type, cpu_info, distribution


# 获取系统内存信息
def get_memory_info(os_type: OSType) -> str:
    memory = "Unknown Memory"
    try:
        # 尝试导入 psutil 模块以获取虚拟内存信息
        import psutil
        # 使用 psutil 获取虚拟内存总量并转换为 GB 单位
        memory = f"{psutil.virtual_memory().total // (1024 ** 3)} GB"
    except ImportError:
        pass
    if os_type == OSType.LINUX:
        try:
            # 在 Linux 上打开 /proc/meminfo 文件并读取信息
            with open("/proc/meminfo", "r") as f:
                mem_info = f.readlines()
            # 遍历文件内容并找到包含 MemTotal 的行，获取内存总量信息
            for line in mem_info:
                if "MemTotal" in line:
                    memory = line.split(":")[1].strip()
                    break
        except:
            pass
    # 返回内存信息
    return memory


# 缓存装饰器，用于缓存系统信息
@cache
# 获取系统信息的函数，返回 SystemInfo 类型的对象
def get_system_info() -> SystemInfo:
    """Get System information"""

    # 调用自定义函数获取系统的操作系统类型、AVX类型、CPU信息和发行版信息
    os_type, avx_type, cpu_info, distribution = get_cpu_info()

    # 获取当前 Python 解释器的版本信息
    python_version = platform.python_version()

    # 调用自定义函数获取系统的内存信息
    memory = get_memory_info(os_type)

    # 调用自定义函数获取系统中的 Torch 版本、设备类型、设备版本、设备数量和其他设备信息
    (
        torch_version,
        device,
        device_version,
        device_count,
        device_other,
    ) = get_device_info()

    # 构建并返回 SystemInfo 对象，包括操作系统平台、发行版、Python 版本、CPU信息、AVX支持信息、内存信息、Torch版本和设备信息
    return SystemInfo(
        platform=os_type._value_,
        distribution=distribution,
        python_version=python_version,
        cpu=cpu_info,
        cpu_avx=avx_type._value_,
        memory=memory,
        torch_version=torch_version,
        device=device,
        device_version=device_version,
        device_count=device_count,
        device_other=device_other,
    )
```
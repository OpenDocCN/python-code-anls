# `.\DB-GPT-src\setup.py`

```py
import functools
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from enum import Enum
from typing import Callable, List, Optional, Tuple
from urllib.parse import quote, urlparse

import setuptools
from setuptools import find_packages

# 从 README.md 文件中读取长描述内容
with open("README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

# 判断是否处于开发模式，环境变量 IS_DEV_MODE 控制，默认为 true
IS_DEV_MODE = os.getenv("IS_DEV_MODE", "true").lower() == "true"

# 设定 DB-GPT 的版本号，默认为 0.5.10，从环境变量 DB_GPT_VERSION 获取
DB_GPT_VERSION = os.getenv("DB_GPT_VERSION", "0.5.10")

# 是否禁用构建缓存，环境变量 BUILD_NO_CACHE 控制，默认为 true
BUILD_NO_CACHE = os.getenv("BUILD_NO_CACHE", "true").lower() == "true"

# 是否启用 LLAMA C++ GPU 加速，环境变量 LLAMA_CPP_GPU_ACCELERATION 控制，默认为 true
LLAMA_CPP_GPU_ACCELERATION = (
    os.getenv("LLAMA_CPP_GPU_ACCELERATION", "true").lower() == "true"
)

# 是否从源代码构建，环境变量 BUILD_FROM_SOURCE 控制，默认为 false
BUILD_FROM_SOURCE = os.getenv("BUILD_FROM_SOURCE", "false").lower() == "true"

# 指定快速聊天项目的源代码 URL，环境变量 BUILD_FROM_SOURCE_URL_FAST_CHAT 控制，默认为 GitHub 地址
BUILD_FROM_SOURCE_URL_FAST_CHAT = os.getenv(
    "BUILD_FROM_SOURCE_URL_FAST_CHAT", "git+https://github.com/lm-sys/FastChat.git"
)

# OpenAI 版本号，从环境变量 BUILD_VERSION_OPENAI 获取
BUILD_VERSION_OPENAI = os.getenv("BUILD_VERSION_OPENAI")

# 是否包含量化功能，环境变量 INCLUDE_QUANTIZATION 控制，默认为 true
INCLUDE_QUANTIZATION = os.getenv("INCLUDE_QUANTIZATION", "true").lower() == "true"

# 是否包含可观测性功能，环境变量 INCLUDE_OBSERVABILITY 控制，默认为 true
INCLUDE_OBSERVABILITY = os.getenv("INCLUDE_OBSERVABILITY", "true").lower() == "true"


def parse_requirements(file_name: str) -> List[str]:
    # 解析给定文件中的依赖项列表，返回清单
    with open(file_name) as f:
        return [
            require.strip()  # 去除每行首尾空白字符
            for require in f
            if require.strip() and not require.startswith("#")  # 忽略空行和以 '#' 开头的注释行
        ]


def find_python():
    # 查找当前系统中的 Python 可执行文件路径
    python_path = sys.executable
    print(python_path)  # 输出找到的 Python 路径
    if not python_path:
        print("Python command not found.")  # 如果未找到 Python 可执行文件，输出错误信息
        return None
    return python_path  # 返回找到的 Python 路径


def get_latest_version(package_name: str, index_url: str, default_version: str):
    # 获取指定包的最新版本号
    python_command = find_python()  # 查找 Python 可执行文件路径
    if not python_command:
        print("Python command not found.")  # 如果未找到 Python 可执行文件，输出错误信息
        return default_version  # 返回默认版本号

    # 构建查询包最新版本的命令列表
    command_index_versions = [
        python_command,
        "-m",
        "pip",
        "index",
        "versions",
        package_name,
        "--index-url",
        index_url,
    ]

    # 运行命令获取结果
    result_index_versions = subprocess.run(
        command_index_versions, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    if result_index_versions.returncode == 0:
        output = result_index_versions.stdout.decode()  # 解码命令输出结果
        lines = output.split("\n")  # 按行分割输出结果
        for line in lines:
            if "Available versions:" in line:
                available_versions = line.split(":")[1].strip()  # 获取可用版本信息
                latest_version = available_versions.split(",")[0].strip()  # 获取第一个版本号
                # 对于 torch 或 torchvision 包，检查与最新版本的兼容性
                if package_name == "torch" or "torchvision":
                    latest_version = latest_version.split("+")[0]
                return latest_version  # 返回最新版本号
    # 如果没有找到匹配的最新版本信息，则模拟安装指定包的最新版本
    else:
        # 构建模拟安装命令，准备安装指定包的最新版本
        command_simulate_install = [
            python_command,      # 使用指定的 Python 解释器命令
            "-m",                # 指定模块调用方式
            "pip",               # 调用 pip 包管理工具
            "install",           # 执行安装操作
            f"{package_name}==", # 指定要安装的包及其版本号的前缀
        ]

        # 运行模拟安装命令，并捕获标准错误流
        result_simulate_install = subprocess.run(
            command_simulate_install, stderr=subprocess.PIPE
        )
        # 输出运行结果对象
        print(result_simulate_install)
        # 解码标准错误输出，以便分析安装结果
        stderr_output = result_simulate_install.stderr.decode()
        # 输出标准错误输出内容
        print(stderr_output)
        
        # 在标准错误输出中查找可用版本信息的匹配项
        match = re.search(r"from versions: (.+?)\)", stderr_output)
        if match:
            # 提取所有可用版本号并转换为列表
            available_versions = match.group(1).split(", ")
            # 获取最新的版本号（列表中的最后一个元素），并去除首尾空白字符
            latest_version = available_versions[-1].strip()
            # 返回找到的最新版本号
            return latest_version

    # 如果没有找到可用版本信息，则返回默认版本号
    return default_version
def encode_url(package_url: str) -> str:
    # 解析给定的包 URL
    parsed_url = urlparse(package_url)
    # 对 URL 中的路径部分进行编码
    encoded_path = quote(parsed_url.path)
    # 使用编码后的路径构建安全的 URL
    safe_url = parsed_url._replace(path=encoded_path).geturl()
    # 返回安全的 URL 和解析后的路径部分
    return safe_url, parsed_url.path


def cache_package(package_url: str, package_name: str, is_windows: bool = False):
    # 对包的 URL 进行编码和解析，获取安全的 URL 和解析后的路径
    safe_url, parsed_url = encode_url(package_url)
    
    # 如果设置了 BUILD_NO_CACHE，则直接返回安全的 URL
    if BUILD_NO_CACHE:
        return safe_url

    # 导入必要的库
    from pip._internal.utils.appdirs import user_cache_dir

    # 从解析后的 URL 中获取文件名
    filename = os.path.basename(parsed_url)
    # 构建缓存目录路径
    cache_dir = os.path.join(user_cache_dir("pip"), "http", "wheels", package_name)
    os.makedirs(cache_dir, exist_ok=True)

    # 构建本地文件路径
    local_path = os.path.join(cache_dir, filename)
    
    # 如果本地文件不存在
    if not os.path.exists(local_path):
        # 构建临时文件路径
        temp_path = local_path + ".tmp"
        # 如果临时文件存在，则删除
        if os.path.exists(temp_path):
            os.remove(temp_path)
        try:
            # 下载文件到临时文件路径
            print(f"Download {safe_url} to {local_path}")
            urllib.request.urlretrieve(safe_url, temp_path)
            # 将临时文件移动到最终路径
            shutil.move(temp_path, local_path)
        finally:
            # 确保在任何情况下都删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # 根据操作系统类型返回文件路径的 URL
    return f"file:///{local_path}" if is_windows else f"file://{local_path}"


class SetupSpec:
    def __init__(self) -> None:
        # 初始化额外的设置为字典
        self.extras: dict = {}
        # 初始化安装依赖为列表
        self.install_requires: List[str] = []

    @property
    def unique_extras(self) -> dict[str, list[str]]:
        # 返回唯一的额外设置，去除重复值
        unique_extras = {}
        for k, v in self.extras.items():
            unique_extras[k] = list(set(v))
        return unique_extras


setup_spec = SetupSpec()


class AVXType(Enum):
    BASIC = "basic"
    AVX = "AVX"
    AVX2 = "AVX2"
    AVX512 = "AVX512"

    @staticmethod
    def of_type(avx: str):
        # 根据给定的 AVX 类型字符串返回相应的枚举项
        for item in AVXType:
            if item._value_ == avx:
                return item
        return None


class OSType(Enum):
    WINDOWS = "win"
    LINUX = "linux"
    DARWIN = "darwin"
    OTHER = "other"


@functools.cache
def get_cpu_avx_support() -> Tuple[OSType, AVXType]:
    # 获取当前系统类型
    system = platform.system()
    os_type = OSType.OTHER
    cpu_avx = AVXType.BASIC
    # 从环境变量中获取 CPU AVX 类型
    env_cpu_avx = AVXType.of_type(os.getenv("DBGPT_LLAMA_CPP_AVX"))

    # 根据系统类型进行处理
    if "windows" in system.lower():
        os_type = OSType.WINDOWS
        output = "avx2"
        print("Current platform is windows, use avx2 as default cpu architecture")
    elif system == "Linux":
        os_type = OSType.LINUX
        # 运行 lscpu 命令获取 CPU 信息
        result = subprocess.run(
            ["lscpu"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output = result.stdout.decode()
    elif system == "Darwin":
        os_type = OSType.DARWIN
        # 运行 sysctl 命令获取系统信息
        result = subprocess.run(
            ["sysctl", "-a"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output = result.stdout.decode()
    else:
        os_type = OSType.OTHER
        print("Unsupported OS to get cpu avx, use default")
        # 如果环境变量中定义了 CPU AVX 类型，则返回它；否则返回默认类型
        return os_type, env_cpu_avx if env_cpu_avx else cpu_avx

    # 根据输出确定 CPU AVX 类型
    if "avx512" in output.lower():
        cpu_avx = AVXType.AVX512
    elif "avx2" in output.lower():
        # 如果输出中包含 "avx2"，则将 CPU 的 AVX 类型设置为 AVX2
        cpu_avx = AVXType.AVX2
    elif "avx " in output.lower():
        # 如果输出中包含 "avx "，则表示 AVX 类型，但暂时不处理
        # cpu_avx =  AVXType.AVX
        pass
    # 返回操作系统类型和根据环境或输出确定的 AVX 类型
    return os_type, env_cpu_avx if env_cpu_avx else cpu_avx
# 获取当前系统的 AVX 支持信息和操作系统类型
os_type, _ = get_cpu_avx_support()
# 获取当前系统的 CUDA 版本
cuda_version = get_cuda_version()
# 获取 Python 版本并生成对应的标识
py_version = platform.python_version()
py_version = "cp" + "".join(py_version.split(".")[0:2])

# 如果操作系统是 Darwin（macOS）或者无法获取到 CUDA 版本，则返回 None
if os_type == OSType.DARWIN or not cuda_version:
    return None

# 如果 CUDA 版本在支持的版本列表中，则继续使用该版本；否则进行处理
if cuda_version in supported_cuda_versions:
    cuda_version = cuda_version
else:
    # 如果 CUDA 版本不在支持的版本列表中，则发出警告并选择合适的版本
    print(
        f"Warning: Your CUDA version {cuda_version} is not in our set supported_cuda_versions , we will use our set version."
    )
    # 如果 CUDA 版本小于 "12.1"，则选择支持的 CUDA 版本列表的第一个版本；否则选择最后一个版本
    if cuda_version < "12.1":
        cuda_version = supported_cuda_versions[0]
    else:
        cuda_version = supported_cuda_versions[-1]

# 根据 CUDA 版本生成对应的标识
cuda_version = "cu" + cuda_version.replace(".", "")
# 根据操作系统类型生成对应的包名后缀
os_pkg_name = "linux_x86_64" if os_type == OSType.LINUX else "win_amd64"
    # 如果提供了 base_url_func 函数，则调用它以获取基础 URL
    if base_url_func:
        # 使用 base_url_func 函数获取基础 URL，传递包版本、CUDA 版本、Python 版本作为参数
        base_url = base_url_func(pkg_version, cuda_version, py_version)
        # 如果获取到了 base_url 并且它以斜杠结尾，则去除末尾的斜杠
        if base_url and base_url.endswith("/"):
            base_url = base_url[:-1]
    
    # 如果提供了 pkg_file_func 函数，则调用它以获取完整的包文件名
    if pkg_file_func:
        # 使用 pkg_file_func 函数获取完整的包文件名，传递包名、包版本、CUDA 版本、Python 版本、操作系统类型作为参数
        full_pkg_file = pkg_file_func(
            pkg_name, pkg_version, cuda_version, py_version, os_type
        )
    else:
        # 否则，构造默认的包文件名格式
        full_pkg_file = f"{pkg_name}-{pkg_version}+{cuda_version}-{py_version}-{py_version}-{os_pkg_name}.whl"
    
    # 如果没有获取到 base_url，则直接返回完整的包文件名
    if not base_url:
        return full_pkg_file
    else:
        # 否则，返回拼接了 base_url 的完整包文件 URL
        return f"{base_url}/{full_pkg_file}"
# 定义一个函数用于配置所需的 Torch 包依赖
def torch_requires(
    torch_version: str = "2.2.1",
    torchvision_version: str = "0.17.1",
    torchaudio_version: str = "2.2.1",
):
    # 获取操作系统类型和 CPU AVX 支持信息
    os_type, _ = get_cpu_avx_support()

    # 构建包含指定版本的 Torch 包列表
    torch_pkgs = [
        f"torch=={torch_version}",
        f"torchvision=={torchvision_version}",
        f"torchaudio=={torchaudio_version}",
    ]

    # 初始化用于非 Darwin 操作系统的 torch_cuda_pkgs；对于 Darwin 或不需要特定 CUDA 处理的情况，它与 torch_pkgs 相同
    torch_cuda_pkgs = torch_pkgs[:]

    # 如果操作系统不是 DARWIN
    if os_type != OSType.DARWIN:
        # 支持的 CUDA 版本列表
        supported_versions = ["11.8", "12.1"]
        # 定义基础 URL 函数
        base_url_func = lambda v, x, y: f"https://download.pytorch.org/whl/{x}"
        
        # 构建 torch 的 wheels 下载 URL
        torch_url = _build_wheels(
            "torch",
            torch_version,
            base_url_func=base_url_func,
            supported_cuda_versions=supported_versions,
        )
        
        # 构建 torchvision 的 wheels 下载 URL
        torchvision_url = _build_wheels(
            "torchvision",
            torchvision_version,
            base_url_func=base_url_func,
            supported_cuda_versions=supported_versions,
        )

        # 如果存在 torch_url，则缓存下载的包，并根据操作系统是否为 Windows 设置 torch_cuda_pkgs[0]
        if torch_url:
            torch_url_cached = cache_package(
                torch_url, "torch", os_type == OSType.WINDOWS
            )
            torch_cuda_pkgs[0] = f"torch @ {torch_url_cached}"
        
        # 如果存在 torchvision_url，则缓存下载的包，并根据操作系统是否为 Windows 设置 torch_cuda_pkgs[1]
        if torchvision_url:
            torchvision_url_cached = cache_package(
                torchvision_url, "torchvision", os_type == OSType.WINDOWS
            )
            torch_cuda_pkgs[1] = f"torchvision @ {torchvision_url_cached}"

    # 假设 setup_spec 是一个字典，用于添加这些依赖项到 extras 字段
    # 添加 torch 包到 extras["torch"]
    setup_spec.extras["torch"] = torch_pkgs
    # 添加 torch 包到 extras["torch_cpu"]
    setup_spec.extras["torch_cpu"] = torch_pkgs
    # 添加 torch_cuda_pkgs 到 extras["torch_cuda"]
    setup_spec.extras["torch_cuda"] = torch_cuda_pkgs


# 定义函数用于设置 llama_cpp_python_cuda 的依赖
def llama_cpp_python_cuda_requires():
    # 获取 CUDA 版本
    cuda_version = get_cuda_version()
    # 支持的 CUDA 版本列表
    supported_cuda_versions = ["11.8", "12.1"]
    # 默认设备为 CPU
    device = "cpu"

    # 如果没有 CUDA 版本，则输出消息并返回
    if not cuda_version:
        print("CUDA not support, use cpu version")
        return
    
    # 如果 LLAMA_CPP_GPU_ACCELERATION 为假，则输出消息并返回
    if not LLAMA_CPP_GPU_ACCELERATION:
        print("Disable GPU acceleration")
        return

    # 如果 CUDA 版本小于等于 "11.8" 且不为 None
    if cuda_version <= "11.8" and not None:
        device = "cu" + supported_cuda_versions[0].replace(".", "")
    else:
        device = "cu" + supported_cuda_versions[-1].replace(".", "")
    
    # 获取操作系统类型和 CPU AVX 支持信息
    os_type, cpu_avx = get_cpu_avx_support()

    # 输出操作系统类型和 CPU AVX 支持信息
    print(f"OS: {os_type}, cpu avx: {cpu_avx}")

    # 支持的操作系统列表
    supported_os = [OSType.WINDOWS, OSType.LINUX]
    
    # 如果当前操作系统不在支持列表中，则输出消息并返回
    if os_type not in supported_os:
        print(
            f"llama_cpp_python_cuda just support in os: {[r._value_ for r in supported_os]}"
        )
        return

    # CPU 设备类型
    cpu_device = ""

    # 如果 CPU 支持 AVX2 或 AVX512，则设置 cpu_device 为 "avx"，否则为 "basic"
    if cpu_avx == AVXType.AVX2 or cpu_avx == AVXType.AVX512:
        cpu_device = "avx"
    else:
        cpu_device = "basic"
    
    # 添加 CPU 设备类型到 device 字符串末尾
    device += cpu_device
    
    # llama-cpp-python-cuBLAS-wheels 的基础 URL
    base_url = "https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui"
    
    # llama-cpp-python-cuBLAS 的版本号
    llama_cpp_version = "0.2.26"
    # 设置 Python 版本号
    py_version = "cp310"
    
    # 根据操作系统类型选择包名称
    os_pkg_name = "manylinux_2_31_x86_64" if os_type == OSType.LINUX else "win_amd64"
    
    # 构建额外索引的 URL，包括基础 URL、llama_cpp 的版本号、设备信息、Python 版本号、操作系统包名称
    extra_index_url = f"{base_url}/llama_cpp_python_cuda-{llama_cpp_version}+{device}-{py_version}-{py_version}-{os_pkg_name}.whl"
    
    # 对额外索引 URL 进行编码
    extra_index_url, _ = encode_url(extra_index_url)
    
    # 打印安装消息，显示额外索引的 URL
    print(f"Install llama_cpp_python_cuda from {extra_index_url}")
    
    # 将额外索引 URL 添加到 setup_spec.extras["llama_cpp"] 列表中
    setup_spec.extras["llama_cpp"].append(f"llama_cpp_python_cuda @ {extra_index_url}")
# 定义函数 core_requires，用于设置不同功能组的依赖列表
def core_requires():
    # 设置 core 功能组的依赖列表
    setup_spec.extras["core"] = [
        "aiohttp==3.8.4",                      # 异步 HTTP 客户端库
        "chardet==5.1.0",                      # 字符编码检测库
        "importlib-resources==5.12.0",         # 导入资源文件的模块
        "python-dotenv==1.0.0",                # 加载环境变量文件的模块
        "cachetools",                          # 缓存工具库
        "pydantic>=2.6.0",                     # 数据验证和设置库
        # 用于 AWEL 类型检查
        "typeguard",                           
        # Snowflake 不需要额外依赖
        "snowflake-id",                        
        "typing_inspect",                      # 类型检查工具
    ]
    
    # 设置 client 功能组的依赖列表，包括 core 功能组的所有依赖和额外的依赖
    setup_spec.extras["client"] = setup_spec.extras["core"] + [
        "httpx",                               # 高性能 HTTP 客户端库
        "fastapi>=0.100.0",                    # 快速 API 框架
        # 用于重试，chromadb 需要 tenacity<=8.3.0
        "tenacity<=8.3.0",                     
    ]
    
    # 设置 cli 功能组的依赖列表，包括 client 功能组的所有依赖和额外的依赖
    setup_spec.extras["cli"] = setup_spec.extras["client"] + [
        "prettytable",                         # 格式化表格输出库
        "click",                               # 命令行交互工具
        "psutil==5.9.4",                       # 系统进程和系统利用率库
        "colorama==0.4.6",                     # 终端文本着色库
        "tomlkit",                             # TOML 文件处理工具
        "rich",                                # 丰富的文本输出工具
    ]
    
    # 设置 agent 功能组的依赖列表，包括 cli 功能组的所有依赖和额外的依赖
    setup_spec.extras["agent"] = setup_spec.extras["cli"] + [
        "termcolor",                           # 控制台文本颜色工具
        # pandas 依赖，但计划移除
        "pandas==2.0.3",                       
        # numpy 版本需求小于 2.0.0
        "numpy>=1.21.0,<2.0.0",                
    ]
    
    # 设置 simple_framework 功能组的依赖列表，包括 agent 功能组的所有依赖和额外的依赖
    setup_spec.extras["simple_framework"] = setup_spec.extras["agent"] + [
        "jinja2",                              # 模板引擎
        "uvicorn",                             # ASGI 服务器
        "shortuuid",                           # UUID 生成器
        # SQLAlchemy 版本需求范围，不支持 2.0.29
        "SQLAlchemy>=2.0.25,<2.0.29",          
        # 用于缓存
        "msgpack",                             
        # pympler 长时间未更新，需寻找新的工具包
        "pympler",                             
        "duckdb",                              # 嵌入式 SQL 数据库
        "duckdb-engine",                       # DuckDB Python 绑定
        # 轻量级 Python 任务调度库
        "schedule",                            
        # 用于数据源子包
        "sqlparse==0.4.4",                     
    ]
    
    # 如果从源代码构建，则添加 fschat 依赖，否则添加默认的 fschat 依赖
    if BUILD_FROM_SOURCE:
        setup_spec.extras["simple_framework"].append(
            f"fschat @ {BUILD_FROM_SOURCE_URL_FAST_CHAT}"
        )
    else:
        setup_spec.extras["simple_framework"].append("fschat")
    
    # 设置 framework 功能组的依赖列表，包括 simple_framework 功能组的所有依赖和额外的依赖
    setup_spec.extras["framework"] = setup_spec.extras["simple_framework"] + [
        "coloredlogs",                         # 彩色日志输出工具
        "seaborn",                             # 数据可视化工具
        "auto-gpt-plugin-template",            # 自动 GPT 插件模板
        "gTTS==2.3.1",                         # Google 文字到语音库
        "pymysql",                             # MySQL 数据库接口
        "jsonschema",                          # JSON 数据验证工具
        # 计划将 transformers 移至默认依赖
        "transformers>=4.34.0",                
        "alembic==1.12.0",                     # 数据库迁移工具
        # 用于处理 Excel 文件
        "openpyxl==3.1.2",                     
        "chardet==5.1.0",                      # 字符编码检测库
        "xlrd==2.0.1",                         # 用于读取 Excel 文件
        "aiofiles",                            # 异步文件操作库
        # 用于 agent 功能组
        "GitPython",                           # Git 操作库
        # AWEL DAG 可视化，graphviz 是一个小型的图形库，也可以移到默认依赖
        "graphviz",                            
    ]
    # 将额外的依赖项添加到 setup_spec.extras 字典中的 "rag" 键
    setup_spec.extras["rag"] = setup_spec.extras["vstore"] + [
        "spacy>=3.7",         # 添加 spacy 依赖项，要求版本 >= 3.7
        "markdown",           # 添加 markdown 依赖项
        "bs4",                # 添加 bs4 (Beautiful Soup) 依赖项
        "python-pptx",        # 添加 python-pptx 依赖项
        "python-docx",        # 添加 python-docx 依赖项
        "pypdf",              # 添加 pypdf 依赖项
        "pdfplumber",         # 添加 pdfplumber 依赖项
        "python-multipart",   # 添加 python-multipart 依赖项
        "sentence-transformers",  # 添加 sentence-transformers 依赖项
    ]
# 定义一个函数 llama_cpp_requires，用于配置 llama_cpp 扩展的依赖
def llama_cpp_requires():
    """
    pip install "dbgpt[llama_cpp]"
    """
    # 将 "llama-cpp-python" 添加到 setup_spec 的 extras 中的 "llama_cpp" 键下
    setup_spec.extras["llama_cpp"] = ["llama-cpp-python"]
    # 调用 llama_cpp_python_cuda_requires 函数
    llama_cpp_python_cuda_requires()


# 定义一个返回可选字符串的函数 _build_autoawq_requires
def _build_autoawq_requires() -> Optional[str]:
    # 调用 get_cpu_avx_support 函数获取操作系统类型和 AVX 支持信息
    os_type, _ = get_cpu_avx_support()
    # 如果操作系统是 DARWIN，则返回 None
    if os_type == OSType.DARWIN:
        return None
    # 否则返回字符串 "auto-gptq"
    return "auto-gptq"


# 定义一个函数 quantization_requires，用于配置量化相关的依赖
def quantization_requires():
    # 调用 get_cpu_avx_support 函数获取操作系统类型和 AVX 支持信息
    os_type, _ = get_cpu_avx_support()
    # 初始化一个空列表 quantization_pkgs
    quantization_pkgs = []
    
    # 如果操作系统是 WINDOWS
    if os_type == OSType.WINDOWS:
        # 获取 bitsandbytes 的最新版本号
        latest_version = get_latest_version(
            "bitsandbytes",
            "https://jllllll.github.io/bitsandbytes-windows-webui",
            "0.41.1",
        )
        # 构建 bitsandbytes 的 WHL 文件下载链接
        whl_url = f"https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-{latest_version}-py3-none-win_amd64.whl"
        # 缓存下载的包，并返回本地路径
        local_pkg_path = cache_package(whl_url, "bitsandbytes", True)
        # 将 "bitsandbytes" 和其本地安装路径添加到 setup_spec 的 extras 中的 "bitsandbytes" 键下
        setup_spec.extras["bitsandbytes"] = [f"bitsandbytes @ {local_pkg_path}"]
    else:
        # 如果不是 WINDOWS 操作系统，将 "bitsandbytes" 添加到 setup_spec 的 extras 中的 "bitsandbytes" 键下
        setup_spec.extras["bitsandbytes"] = ["bitsandbytes"]

    # 如果操作系统不是 DARWIN
    if os_type != OSType.DARWIN:
        # 获取 CUDA 版本信息
        cuda_version = get_cuda_version()
        # 如果 CUDA 版本为空或者是 "12.1"
        if cuda_version is None or cuda_version == "12.1":
            # 添加 "autoawq", _build_autoawq_requires() 返回的值，以及 "optimum" 到 quantization_pkgs 列表中
            quantization_pkgs.extend(["autoawq", _build_autoawq_requires(), "optimum"])
        else:
            # 否则，需要添加适用于 CUDA 版本 11.8 的 autoawq 的安装方法（TODO: 待实现）
            quantization_pkgs.extend(["autoawq", _build_autoawq_requires(), "optimum"])

    # 将 "cpm_kernels"、quantization_pkgs 列表和 setup_spec 的 extras 中的 "bitsandbytes" 键下的内容，合并后添加到 "quantization" 键下
    setup_spec.extras["quantization"] = (
        ["cpm_kernels"] + quantization_pkgs + setup_spec.extras["bitsandbytes"]
    )


# 定义一个函数 all_vector_store_requires，用于配置向量存储相关的依赖
def all_vector_store_requires():
    """
    pip install "dbgpt[vstore]"
    """
    # 设置 setup_spec 的 extras 中的 "vstore" 键下的内容为包含 "chromadb>=0.4.22" 的列表
    setup_spec.extras["vstore"] = [
        "chromadb>=0.4.22",
    ]
    # 设置 setup_spec 的 extras 中的 "vstore_weaviate" 键下的内容为 "vstore" 和 ["weaviate-client"] 的组合
    setup_spec.extras["vstore_weaviate"] = setup_spec.extras["vstore"] + [
        "weaviate-client",
    ]
    # 设置 setup_spec 的 extras 中的 "vstore_milvus" 键下的内容为 "vstore" 和 ["pymilvus"] 的组合
    setup_spec.extras["vstore_milvus"] = setup_spec.extras["vstore"] + [
        "pymilvus",
    ]
    # 设置 setup_spec 的 extras 中的 "vstore_all" 键下的内容为 "vstore"、"vstore_weaviate" 和 "vstore_milvus" 的组合
    setup_spec.extras["vstore_all"] = (
        setup_spec.extras["vstore"]
        + setup_spec.extras["vstore_weaviate"]
        + setup_spec.extras["vstore_milvus"]
    )


# 定义一个函数 all_datasource_requires，用于配置数据源相关的依赖
def all_datasource_requires():
    """
    pip install "dbgpt[datasource]"
    """
    # 设置 setup_spec 的 extras 中的 "datasource" 键下的内容为包含 "pymysql" 的列表
    setup_spec.extras["datasource"] = [
        # "sqlparse==0.4.4",
        "pymysql",
    ]
    # 如果要在 Ubuntu 上安装 psycopg2 和 mysqlclient，需要先安装 libpq-dev 和 libmysqlclient-dev
    # 将 "datasource_all" 设置为 "datasource" 列表加上额外的数据源列表
    setup_spec.extras["datasource_all"] = setup_spec.extras["datasource"] + [
        "pyspark",           # 添加 pyspark 到数据源列表
        "pymssql",           # 添加 pymssql 到数据源列表
        # 在虚拟环境中安装 psycopg2-binary
        # pip install psycopg2-binary
        "psycopg2",          # 添加 psycopg2 到数据源列表
        # mysqlclient 2.2.x 在 Python 3.10+ 中有 pkg-config 问题
        "mysqlclient==2.1.0",  # 添加 mysqlclient 2.1.0 到数据源列表
        # pydoris 版本过旧，建议寻找新的替代包
        "pydoris>=1.0.2,<2.0.0",  # 添加 pydoris 版本约束到数据源列表
        "clickhouse-connect",  # 添加 clickhouse-connect 到数据源列表
        "pyhive",             # 添加 pyhive 到数据源列表
        "thrift",             # 添加 thrift 到数据源列表
        "thrift_sasl",        # 添加 thrift_sasl 到数据源列表
        "neo4j",              # 添加 neo4j 到数据源列表
        "vertica_python",     # 添加 vertica_python 到数据源列表
    ]
def openai_requires():
    """
    pip install "dbgpt[openai]"
    """
    # 添加 'openai' 到 setup_spec.extras 字典中，其中包含 'tiktoken' 作为依赖
    setup_spec.extras["openai"] = ["tiktoken"]

    # 如果存在 BUILD_VERSION_OPENAI 变量，则添加 openai SDK 的指定版本到 'openai' 依赖中
    if BUILD_VERSION_OPENAI:
        # 从环境变量中读取 openai SDK 的版本号
        setup_spec.extras["openai"].append(f"openai=={BUILD_VERSION_OPENAI}")
    else:
        # 否则默认添加 'openai' 到 'openai' 依赖中
        setup_spec.extras["openai"].append("openai")

    # 如果 INCLUDE_OBSERVABILITY 为 True，则将 setup_spec.extras["observability"] 添加到 'openai' 依赖中
    if INCLUDE_OBSERVABILITY:
        setup_spec.extras["openai"] += setup_spec.extras["observability"]

    # 将 setup_spec.extras["framework"] 添加到 'openai' 依赖中
    setup_spec.extras["openai"] += setup_spec.extras["framework"]
    # 将 setup_spec.extras["rag"] 添加到 'openai' 依赖中
    setup_spec.extras["openai"] += setup_spec.extras["rag"]


def gpt4all_requires():
    """
    pip install "dbgpt[gpt4all]"
    """
    # 添加 'gpt4all' 到 setup_spec.extras 字典中，包含 'gpt4all' 作为依赖
    setup_spec.extras["gpt4all"] = ["gpt4all"]


def vllm_requires():
    """
    pip install "dbgpt[vllm]"
    """
    # 添加 'vllm' 到 setup_spec.extras 字典中，包含 'vllm' 作为依赖
    setup_spec.extras["vllm"] = ["vllm"]


def cache_requires():
    """
    pip install "dbgpt[cache]"
    """
    # 添加 'cache' 到 setup_spec.extras 字典中，包含 'rocksdict' 作为依赖
    setup_spec.extras["cache"] = ["rocksdict"]


def observability_requires():
    """
    pip install "dbgpt[observability]"

    Send DB-GPT traces to OpenTelemetry compatible backends.
    """
    # 添加 'observability' 到 setup_spec.extras 字典中，包含 Opentelemetry 相关包作为依赖
    setup_spec.extras["observability"] = [
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-exporter-otlp",
    ]


def default_requires():
    """
    pip install "dbgpt[default]"
    """
    # 添加 'default' 到 setup_spec.extras 字典中，包含一系列默认依赖
    setup_spec.extras["default"] = [
        # "tokenizers==0.13.3",
        "tokenizers>=0.14",
        "accelerate>=0.20.3",
        "zhipuai",
        "dashscope",
        "chardet",
        "sentencepiece",
        "ollama",
    ]
    # 将 setup_spec.extras["framework"] 添加到 'default' 依赖中
    setup_spec.extras["default"] += setup_spec.extras["framework"]
    # 将 setup_spec.extras["rag"] 添加到 'default' 依赖中
    setup_spec.extras["default"] += setup_spec.extras["rag"]
    # 将 setup_spec.extras["datasource"] 添加到 'default' 依赖中
    setup_spec.extras["default"] += setup_spec.extras["datasource"]
    # 将 setup_spec.extras["torch"] 添加到 'default' 依赖中
    setup_spec.extras["default"] += setup_spec.extras["torch"]
    # 将 setup_spec.extras["cache"] 添加到 'default' 依赖中
    setup_spec.extras["default"] += setup_spec.extras["cache"]
    # 如果 INCLUDE_QUANTIZATION 为 True，则将 setup_spec.extras["quantization"] 添加到 'default' 依赖中
    if INCLUDE_QUANTIZATION:
        setup_spec.extras["default"] += setup_spec.extras["quantization"]
    # 如果 INCLUDE_OBSERVABILITY 为 True，则将 setup_spec.extras["observability"] 添加到 'default' 依赖中
    if INCLUDE_OBSERVABILITY:
        setup_spec.extras["default"] += setup_spec.extras["observability"]


def all_requires():
    # 创建一个空集合 requires
    requires = set()
    # 遍历 setup_spec.extras 字典中的所有项目
    for _, pkgs in setup_spec.extras.items():
        # 将每个依赖包名加入 requires 集合中
        for pkg in pkgs:
            requires.add(pkg)
    # 将 requires 集合转换为列表并赋值给 setup_spec.extras["all"]
    setup_spec.extras["all"] = list(requires)


def init_install_requires():
    # 将 setup_spec.extras["core"] 添加到 setup_spec.install_requires 中
    setup_spec.install_requires += setup_spec.extras["core"]
    # 打印安装需要的依赖列表，以逗号分隔
    print(f"Install requires: \n{','.join(setup_spec.install_requires)}")
    # 使用 `find_packages` 函数查找指定模块的所有子模块，并将结果存储在 `packages` 变量中
    packages = find_packages(
        # 排除在 `excluded_packages` 列表中指定的模块
        exclude=excluded_packages,
        # 包含以下列出的模块及其所有子模块
        include=[
            "dbgpt",
            "dbgpt._private",
            "dbgpt._private.*",
            "dbgpt.agent",
            "dbgpt.agent.*",
            "dbgpt.cli",
            "dbgpt.cli.*",
            "dbgpt.client",
            "dbgpt.client.*",
            "dbgpt.configs",
            "dbgpt.configs.*",
            "dbgpt.core",
            "dbgpt.core.*",
            "dbgpt.datasource",
            "dbgpt.datasource.*",
            "dbgpt.experimental",
            "dbgpt.experimental.*",
            "dbgpt.model",
            "dbgpt.model.proxy",
            "dbgpt.model.proxy.*",
            "dbgpt.model.operators",
            "dbgpt.model.operators.*",
            "dbgpt.model.utils",
            "dbgpt.model.utils.*",
            "dbgpt.model.adapter",
            "dbgpt.rag",
            "dbgpt.rag.*",
            "dbgpt.storage",
            "dbgpt.storage.*",
            "dbgpt.util",
            "dbgpt.util.*",
            "dbgpt.vis",
            "dbgpt.vis.*",
        ],
    )
class PrintExtrasCommand(setuptools.Command):
    # 定义打印 extras_require 的自定义命令类
    description = "print extras_require"
    # 描述命令的作用

    user_options = [
        ("output=", "o", "Path to output the extras_require JSON"),
        # 用户选项，允许设置输出的 extras_require JSON 文件路径
    ]

    def initialize_options(self):
        self.output = None
        # 初始化命令选项中的 output 属性为 None

    def finalize_options(self):
        if self.output is None:
            raise ValueError("output is not set")
        # 确定命令选项中的 output 属性是否已设置，若未设置则引发 ValueError 异常

    def run(self):
        with open(self.output, "w") as f:
            json.dump(setup_spec.unique_extras, f, indent=2)
        # 运行命令时，将 setup_spec.unique_extras 中的数据以格式化 JSON 的形式写入到指定路径的文件中


setuptools.setup(
    name="dbgpt",
    packages=packages,
    version=DB_GPT_VERSION,
    author="csunny",
    author_email="cfqcsunny@gmail.com",
    description="DB-GPT is an experimental open-source project that uses localized GPT "
    "large models to interact with your data and environment."
    " With this solution, you can be assured that there is no risk of data leakage, "
    "and your data is 100% private and secure.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=setup_spec.install_requires,
    url="https://github.com/eosphoros-ai/DB-GPT",
    license="https://opensource.org/license/mit/",
    python_requires=">=3.10",
    extras_require=setup_spec.unique_extras,
    cmdclass={
        "print_extras": PrintExtrasCommand,
    },
    entry_points={
        "console_scripts": [
            "dbgpt=dbgpt.cli.cli_scripts:main",
        ],
    },
)
# 设置项目的元数据和配置信息，包括名称、包、版本、作者信息、描述、依赖、许可证、命令类、入口点等
```
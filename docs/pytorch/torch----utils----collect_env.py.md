# `.\pytorch\torch\utils\collect_env.py`

```py
# mypy: allow-untyped-defs

# Unlike the rest of the PyTorch this file must be python2 compliant.
# This script outputs relevant system environment info
# Run it with `python collect_env.py` or `python -m torch.utils.collect_env`
import datetime  # 导入 datetime 模块，用于处理日期和时间
import locale  # 导入 locale 模块，用于处理地区特定设置
import re  # 导入 re 模块，用于正则表达式操作
import subprocess  # 导入 subprocess 模块，用于执行外部命令
import sys  # 导入 sys 模块，用于访问系统相关变量和函数
import os  # 导入 os 模块，用于与操作系统进行交互
from collections import namedtuple  # 导入 namedtuple 类型，用于创建命名元组

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False

# System Environment Information
SystemEnv = namedtuple('SystemEnv', [
    'torch_version',  # Torch 版本号
    'is_debug_build',  # 是否为调试构建
    'cuda_compiled_version',  # CUDA 编译版本
    'gcc_version',  # GCC 版本
    'clang_version',  # Clang 版本
    'cmake_version',  # CMake 版本
    'os',  # 操作系统
    'libc_version',  # libc 版本
    'python_version',  # Python 版本
    'python_platform',  # Python 平台
    'is_cuda_available',  # CUDA 是否可用
    'cuda_runtime_version',  # CUDA 运行时版本
    'cuda_module_loading',  # CUDA 模块加载状态
    'nvidia_driver_version',  # Nvidia 驱动版本
    'nvidia_gpu_models',  # Nvidia GPU 模型
    'cudnn_version',  # cuDNN 版本
    'pip_version',  # 'pip' 或 'pip3'
    'pip_packages',  # pip 包列表
    'conda_packages',  # conda 包列表
    'hip_compiled_version',  # HIP 编译版本
    'hip_runtime_version',  # HIP 运行时版本
    'miopen_runtime_version',  # MIOpen 运行时版本
    'caching_allocator_config',  # 缓存分配器配置
    'is_xnnpack_available',  # 是否可用 XNNPACK
    'cpu_info',  # CPU 信息
])

DEFAULT_CONDA_PATTERNS = {
    "torch",  # 默认的 conda 包模式
    "numpy",  # 默认的 conda 包模式
    "cudatoolkit",  # 默认的 conda 包模式
    "soumith",  # 默认的 conda 包模式
    "mkl",  # 默认的 conda 包模式
    "magma",  # 默认的 conda 包模式
    "triton",  # 默认的 conda 包模式
    "optree",  # 默认的 conda 包模式
}

DEFAULT_PIP_PATTERNS = {
    "torch",  # 默认的 pip 包模式
    "numpy",  # 默认的 pip 包模式
    "mypy",  # 默认的 pip 包模式
    "flake8",  # 默认的 pip 包模式
    "triton",  # 默认的 pip 包模式
    "optree",  # 默认的 pip 包模式
    "onnx",  # 默认的 pip 包模式
}


def run(command):
    """Return (return-code, stdout, stderr)."""
    shell = True if type(command) is str else False
    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=shell)
    raw_output, raw_err = p.communicate()
    rc = p.returncode
    if get_platform() == 'win32':
        enc = 'oem'
    else:
        enc = locale.getpreferredencoding()
    output = raw_output.decode(enc)  # 解码输出为字符串
    err = raw_err.decode(enc)  # 解码错误输出为字符串
    return rc, output.strip(), err.strip()


def run_and_read_all(run_lambda, command):
    """Run command using run_lambda; reads and returns entire output if rc is 0."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out


def run_and_parse_first_match(run_lambda, command, regex):
    """Run command using run_lambda, returns the first regex match if it exists."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def run_and_return_first_line(run_lambda, command):
    """Run command using run_lambda and returns first line if output is not empty."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out.split('\n')[0]


def get_conda_packages(run_lambda, patterns=None):
    if patterns is None:
        patterns = DEFAULT_CONDA_PATTERNS
    conda = os.environ.get('CONDA_EXE', 'conda')
    out = run_and_read_all(run_lambda, "{} list".format(conda))
    if out is None:
        return out
    # 将输出字符串按行分割，并过滤出不以 '#' 开头且包含指定模式名称的行，然后用换行符连接成一个新的字符串返回
    return "\n".join(
        line
        for line in out.splitlines()  # 遍历输出字符串的每一行
        if not line.startswith("#")   # 筛选出不以 '#' 开头的行
        and any(name in line for name in patterns)  # 筛选出包含 patterns 中任意模式名称的行
    )
# 返回通过运行 lambda 函数获取的 gcc 版本号
def get_gcc_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'gcc --version', r'gcc (.*)')

# 返回通过运行 lambda 函数获取的 clang 版本号
def get_clang_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'clang --version', r'clang version (.*)')

# 返回通过运行 lambda 函数获取的 cmake 版本号
def get_cmake_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'cmake --version', r'cmake (.*)')

# 根据平台决定如何获取 Nvidia 驱动版本号
def get_nvidia_driver_version(run_lambda):
    if get_platform() == 'darwin':
        cmd = 'kextstat | grep -i cuda'
        return run_and_parse_first_match(run_lambda, cmd,
                                         r'com[.]nvidia[.]CUDA [(](.*?)[)]')
    smi = get_nvidia_smi()
    return run_and_parse_first_match(run_lambda, smi, r'Driver Version: (.*?) ')

# 根据平台和 Torch 是否可用决定如何获取 GPU 信息
def get_gpu_info(run_lambda):
    if get_platform() == 'darwin' or (TORCH_AVAILABLE and hasattr(torch.version, 'hip') and torch.version.hip is not None):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            if torch.version.hip is not None:
                prop = torch.cuda.get_device_properties(0)
                if hasattr(prop, "gcnArchName"):
                    gcnArch = " ({})".format(prop.gcnArchName)
                else:
                    gcnArch = "NoGCNArchNameOnOldPyTorch"
            else:
                gcnArch = ""
            return torch.cuda.get_device_name(None) + gcnArch
        return None
    smi = get_nvidia_smi()
    uuid_regex = re.compile(r' \(UUID: .+?\)')
    rc, out, _ = run_lambda(smi + ' -L')
    if rc != 0:
        return None
    # 匿名化 GPU 信息，移除其 UUID
    return re.sub(uuid_regex, '', out)

# 返回通过运行 lambda 函数获取的 nvcc 版本号
def get_running_cuda_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'nvcc --version', r'release .+ V(.*)')

# 返回通过运行 lambda 函数获取的 cudnn 版本号
def get_cudnn_version(run_lambda):
    """返回 libcudnn.so 的列表；很难确定正在使用的版本。"""
    if get_platform() == 'win32':
        system_root = os.environ.get('SYSTEMROOT', 'C:\\Windows')
        cuda_path = os.environ.get('CUDA_PATH', "%CUDA_PATH%")
        where_cmd = os.path.join(system_root, 'System32', 'where')
        cudnn_cmd = '{} /R "{}\\bin" cudnn*.dll'.format(where_cmd, cuda_path)
    elif get_platform() == 'darwin':
        # CUDA 库和驱动可以在 /usr/local/cuda/ 中找到。参见链接
        cudnn_cmd = 'ls /usr/local/cuda/lib/libcudnn*'
    else:
        cudnn_cmd = 'ldconfig -p | grep libcudnn | rev | cut -d" " -f1 | rev'
    rc, out, _ = run_lambda(cudnn_cmd)
    # 如果输出为空或者返回码不为 0 或 1，则找不到或有权限错误
    if len(out) == 0 or (rc != 1 and rc != 0):
        l = os.environ.get('CUDNN_LIBRARY')
        if l is not None and os.path.isfile(l):
            return os.path.realpath(l)
        return None
    files_set = set()
    # 将字符串 `out` 按换行符分割，遍历每个文件名
    for fn in out.split('\n'):
        # 获取文件的真实路径，消除符号链接的影响
        fn = os.path.realpath(fn)
        # 如果路径对应的是一个文件
        if os.path.isfile(fn):
            # 将文件路径添加到集合 `files_set` 中
            files_set.add(fn)
    
    # 如果集合 `files_set` 为空，返回 None
    if not files_set:
        return None
    
    # 对文件路径集合进行字母顺序排序，以确保结果的确定性
    files = sorted(files_set)
    
    # 如果只有一个文件，直接返回该文件路径
    if len(files) == 1:
        return files[0]
    
    # 将排序后的文件路径用换行符连接成一个字符串 `result`
    result = '\n'.join(files)
    
    # 返回包含文件路径的字符串，表明可能是以下文件之一
    return 'Probably one of the following:\n{}'.format(result)
# 获取适用于当前操作系统的 nvidia-smi 可执行文件路径
def get_nvidia_smi():
    # 默认 nvidia-smi 可执行文件名
    smi = 'nvidia-smi'
    
    # 如果当前操作系统是 Windows
    if get_platform() == 'win32':
        # 获取系统根目录，默认为 C:\Windows
        system_root = os.environ.get('SYSTEMROOT', 'C:\\Windows')
        # 获取程序文件目录，默认为 C:\Program Files
        program_files_root = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
        # 构建旧版 nvidia-smi 可执行文件路径
        legacy_path = os.path.join(program_files_root, 'NVIDIA Corporation', 'NVSMI', smi)
        # 构建新版 nvidia-smi 可执行文件路径
        new_path = os.path.join(system_root, 'System32', smi)
        # 候选路径列表
        smis = [new_path, legacy_path]
        
        # 遍历候选路径
        for candidate_smi in smis:
            # 如果候选路径存在
            if os.path.exists(candidate_smi):
                # 更新 smi 变量为找到的可执行文件路径，并添加双引号
                smi = '"{}"'.format(candidate_smi)
                break
    
    # 返回最终确定的 nvidia-smi 可执行文件路径
    return smi
# 定义函数以获取操作系统的CPU信息
def get_cpu_info(run_lambda):
    # 初始化变量，用于存储命令执行的返回码、标准输出和错误输出
    rc, out, err = 0, '', ''
    
    # 根据操作系统类型选择执行的命令，获取CPU信息
    if get_platform() == 'linux':
        # 如果是Linux系统，运行'lscpu'命令获取CPU信息
        rc, out, err = run_lambda('lscpu')
    elif get_platform() == 'win32':
        # 如果是Windows系统，运行'wmic'命令获取CPU信息，获取的字段包括名称、制造商、家族、架构、处理器类型、设备ID、当前时钟速度、最大时钟速度、L2缓存大小、L2缓存速度、修订版本
        rc, out, err = run_lambda('wmic cpu get Name,Manufacturer,Family,Architecture,ProcessorType,DeviceID,CurrentClockSpeed,MaxClockSpeed,L2CacheSize,L2CacheSpeed,Revision /VALUE')
    elif get_platform() == 'darwin':
        # 如果是MacOS系统，运行'sysctl'命令获取CPU品牌字符串信息
        rc, out, err = run_lambda("sysctl -n machdep.cpu.brand_string")
    
    # 初始化CPU信息，默认为'None'
    cpu_info = 'None'
    
    # 如果命令执行返回码为0，说明成功获取了CPU信息，将标准输出作为CPU信息返回
    if rc == 0:
        cpu_info = out
    # 否则，将错误输出作为CPU信息返回
    else:
        cpu_info = err
    
    # 返回最终的CPU信息
    return cpu_info


# 定义函数以获取当前操作系统类型
def get_platform():
    # 使用sys.platform判断当前操作系统类型，返回相应的字符串表示
    if sys.platform.startswith('linux'):
        return 'linux'
    elif sys.platform.startswith('win32'):
        return 'win32'
    elif sys.platform.startswith('cygwin'):
        return 'cygwin'
    elif sys.platform.startswith('darwin'):
        return 'darwin'
    else:
        return sys.platform
    # 如果平台是 'win32' 或者 'cygwin'，则调用 get_windows_version 函数处理
    if platform == 'win32' or platform == 'cygwin':
        return get_windows_version(run_lambda)

    # 如果平台是 'darwin'（macOS）
    if platform == 'darwin':
        # 获取 macOS 版本信息
        version = get_mac_version(run_lambda)
        # 如果获取的版本信息为 None，则返回 None
        if version is None:
            return None
        # 返回格式化后的 macOS 版本信息和机器类型
        return 'macOS {} ({})'.format(version, machine())

    # 如果平台是 'linux'
    if platform == 'linux':
        # 尝试获取 Ubuntu/Debian 的版本描述
        desc = get_lsb_version(run_lambda)
        # 如果获取到版本描述，则返回描述和机器类型
        if desc is not None:
            return '{} ({})'.format(desc, machine())

        # 尝试读取 /etc/*-release 文件
        desc = check_release_file(run_lambda)
        # 如果获取到版本描述，则返回描述和机器类型
        if desc is not None:
            return '{} ({})'.format(desc, machine())

        # 如果以上两种方法均未成功获取版本信息，则返回平台名称和机器类型
        return '{} ({})'.format(platform, machine())

    # 如果是未知平台，则直接返回平台名称
    # 这里假设 platform 变量应包含了所有可能的平台名称，因此不会发生未知平台的情况
    return platform
# 获取当前 Python 平台信息
def get_python_platform():
    import platform
    return platform.platform()


# 获取当前 libc 库的版本信息（仅适用于 Linux 系统）
def get_libc_version():
    import platform
    # 检查当前操作系统是否为 Linux，若不是则返回 'N/A'
    if get_platform() != 'linux':
        return 'N/A'
    # 返回 libc 库的版本信息，格式化为字符串
    return '-'.join(platform.libc_ver())


# 获取当前环境下的 pip 安装的包列表
def get_pip_packages(run_lambda, patterns=None):
    """Return `pip list` output. Note: will also find conda-installed pytorch and numpy packages."""
    # 如果未指定 patterns，则使用默认的 pip 匹配模式
    if patterns is None:
        patterns = DEFAULT_PIP_PATTERNS

    # 运行带有 pip 的命令，获取输出结果
    def run_with_pip(pip):
        # 调用 run_lambda 函数执行命令，并读取其输出
        out = run_and_read_all(run_lambda, pip + ["list", "--format=freeze"])
        # 返回包含指定 patterns 的行
        return "\n".join(
            line
            for line in out.splitlines()
            if any(name in line for name in patterns)
        )

    # 根据 Python 版本选择 pip 版本
    pip_version = 'pip3' if sys.version[0] == '3' else 'pip'
    # 使用当前 Python 执行环境运行 pip 命令，获取输出
    out = run_with_pip([sys.executable, '-mpip'])

    return pip_version, out


# 获取 PYTORCH_CUDA_ALLOC_CONF 环境变量的配置信息
def get_cachingallocator_config():
    ca_config = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    return ca_config


# 获取 CUDA 模块加载配置信息
def get_cuda_module_loading_config():
    # 如果 PyTorch 可用且 CUDA 可用，则初始化 CUDA 并返回加载配置信息
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.init()
        config = os.environ.get('CUDA_MODULE_LOADING', '')
        return config
    else:
        return "N/A"


# 检查是否可用 XNNPACK（仅适用于 PyTorch 环境）
def is_xnnpack_available():
    if TORCH_AVAILABLE:
        import torch.backends.xnnpack
        # 返回 XNNPACK 是否启用的字符串表示
        return str(torch.backends.xnnpack.enabled)  # type: ignore[attr-defined]
    else:
        return "N/A"


# 收集环境信息以便调试
def get_env_info():
    """
    Collects environment information to aid in debugging.

    The returned environment information contains details on torch version, is debug build
    or not, cuda compiled version, gcc version, clang version, cmake version, operating
    system, libc version, python version, python platform, CUDA availability, CUDA
    runtime version, CUDA module loading config, GPU model and configuration, Nvidia
    driver version, cuDNN version, pip version and versions of relevant pip and
    conda packages, HIP runtime version, MIOpen runtime version,
    Caching allocator config, XNNPACK availability and CPU information.

    Returns:
        SystemEnv (namedtuple): A tuple containining various environment details
            and system information.
    """
    # 获取 pip 版本和相关包的列表输出
    run_lambda = run
    pip_version, pip_list_output = get_pip_packages(run_lambda)
    # 检查是否导入了 torch 库
    if TORCH_AVAILABLE:
        # 获取 torch 库的版本号
        version_str = torch.__version__
        # 获取 torch 库的调试模式
        debug_mode_str = str(torch.version.debug)
        # 检查是否支持 CUDA
        cuda_available_str = str(torch.cuda.is_available())
        # 获取 CUDA 版本号
        cuda_version_str = torch.version.cuda
        # 如果存在 HIP 版本信息
        if not hasattr(torch.version, 'hip') or torch.version.hip is None:  # cuda version
            hip_compiled_version = hip_runtime_version = miopen_runtime_version = 'N/A'
        else:  # HIP version
            # 获取 HIP 运行时版本和 MIOpen 版本
            def get_version_or_na(cfg, prefix):
                _lst = [s.rsplit(None, 1)[-1] for s in cfg if prefix in s]
                return _lst[0] if _lst else 'N/A'

            cfg = torch._C._show_config().split('\n')
            hip_runtime_version = get_version_or_na(cfg, 'HIP Runtime')
            miopen_runtime_version = get_version_or_na(cfg, 'MIOpen')
            cuda_version_str = 'N/A'
            hip_compiled_version = torch.version.hip
    else:
        # 如果未导入 torch 库，则设置默认值为 'N/A'
        version_str = debug_mode_str = cuda_available_str = cuda_version_str = 'N/A'
        hip_compiled_version = hip_runtime_version = miopen_runtime_version = 'N/A'

    # 获取系统版本信息
    sys_version = sys.version.replace("\n", " ")

    # 获取 conda 包信息
    conda_packages = get_conda_packages(run_lambda)

    # 返回系统环境信息
    return SystemEnv(
        torch_version=version_str,
        is_debug_build=debug_mode_str,
        python_version='{} ({}-bit runtime)'.format(sys_version, sys.maxsize.bit_length() + 1),
        python_platform=get_python_platform(),
        is_cuda_available=cuda_available_str,
        cuda_compiled_version=cuda_version_str,
        cuda_runtime_version=get_running_cuda_version(run_lambda),
        cuda_module_loading=get_cuda_module_loading_config(),
        nvidia_gpu_models=get_gpu_info(run_lambda),
        nvidia_driver_version=get_nvidia_driver_version(run_lambda),
        cudnn_version=get_cudnn_version(run_lambda),
        hip_compiled_version=hip_compiled_version,
        hip_runtime_version=hip_runtime_version,
        miopen_runtime_version=miopen_runtime_version,
        pip_version=pip_version,
        pip_packages=pip_list_output,
        conda_packages=conda_packages,
        os=get_os(run_lambda),
        libc_version=get_libc_version(),
        gcc_version=get_gcc_version(run_lambda),
        clang_version=get_clang_version(run_lambda),
        cmake_version=get_cmake_version(run_lambda),
        caching_allocator_config=get_cachingallocator_config(),
        is_xnnpack_available=is_xnnpack_available(),
        cpu_info=get_cpu_info(run_lambda),
    )
# 定义一个格式化的字符串模板，用于存储环境信息的各项数据
env_info_fmt = """
PyTorch version: {torch_version}
Is debug build: {is_debug_build}
CUDA used to build PyTorch: {cuda_compiled_version}
ROCM used to build PyTorch: {hip_compiled_version}

OS: {os}
GCC version: {gcc_version}
Clang version: {clang_version}
CMake version: {cmake_version}
Libc version: {libc_version}

Python version: {python_version}
Python platform: {python_platform}
Is CUDA available: {is_cuda_available}
CUDA runtime version: {cuda_runtime_version}
CUDA_MODULE_LOADING set to: {cuda_module_loading}
GPU models and configuration: {nvidia_gpu_models}
Nvidia driver version: {nvidia_driver_version}
cuDNN version: {cudnn_version}
HIP runtime version: {hip_runtime_version}
MIOpen runtime version: {miopen_runtime_version}
Is XNNPACK available: {is_xnnpack_available}

CPU:
{cpu_info}

Versions of relevant libraries:
{pip_packages}
{conda_packages}
""".strip()

# 定义一个函数，用于处理环境信息的格式化输出
def pretty_str(envinfo):
    # 内部函数：将字典中为 None 的值替换为指定字符串
    def replace_nones(dct, replacement='Could not collect'):
        for key in dct.keys():
            if dct[key] is not None:
                continue
            dct[key] = replacement
        return dct

    # 内部函数：将字典中的布尔值替换为指定字符串
    def replace_bools(dct, true='Yes', false='No'):
        for key in dct.keys():
            if dct[key] is True:
                dct[key] = true
            elif dct[key] is False:
                dct[key] = false
        return dct

    # 内部函数：在字符串的每一行前添加特定标签
    def prepend(text, tag='[prepend]'):
        lines = text.split('\n')
        updated_lines = [tag + line for line in lines]
        return '\n'.join(updated_lines)

    # 内部函数：如果字符串是多行的，则在其前面添加换行符
    def replace_if_empty(text, replacement='No relevant packages'):
        if text is not None and len(text) == 0:
            return replacement
        return text

    # 内部函数：如果字符串是多行的，则在其前面添加换行符
    def maybe_start_on_next_line(string):
        if string is not None and len(string.split('\n')) > 1:
            return '\n{}\n'.format(string)
        return string

    # 将 namedtuple 转换为可变字典
    mutable_dict = envinfo._asdict()

    # 如果 nvidia_gpu_models 是多行的，将其移到下一行开始
    mutable_dict['nvidia_gpu_models'] = \
        maybe_start_on_next_line(envinfo.nvidia_gpu_models)

    # 如果机器没有安装 CUDA，则将一些字段报告为 'No CUDA'
    dynamic_cuda_fields = [
        'cuda_runtime_version',
        'nvidia_gpu_models',
        'nvidia_driver_version',
    ]
    all_cuda_fields = dynamic_cuda_fields + ['cudnn_version']
    all_dynamic_cuda_fields_missing = all(
        mutable_dict[field] is None for field in dynamic_cuda_fields)
    if TORCH_AVAILABLE and not torch.cuda.is_available() and all_dynamic_cuda_fields_missing:
        for field in all_cuda_fields:
            mutable_dict[field] = 'No CUDA'
        if envinfo.cuda_compiled_version is None:
            mutable_dict['cuda_compiled_version'] = 'None'

    # 将布尔值替换为 Yes 或 No
    mutable_dict = replace_bools(mutable_dict)

    # 将所有 None 替换为 'Could not collect'
    mutable_dict = replace_nones(mutable_dict)
    # 如果 mutable_dict 中的 'pip_packages' 或 'conda_packages' 为空字符串，则用 'No relevant packages' 替换
    mutable_dict['pip_packages'] = replace_if_empty(mutable_dict['pip_packages'])
    mutable_dict['conda_packages'] = replace_if_empty(mutable_dict['conda_packages'])

    # 给 conda 和 pip 的包名添加前缀
    # 如果它们之前是 None，则会显示为例如 '[conda] Could not collect'
    if mutable_dict['pip_packages']:
        # 在 pip_packages 前添加包含 pip 版本的前缀
        mutable_dict['pip_packages'] = prepend(mutable_dict['pip_packages'],
                                               '[{}] '.format(envinfo.pip_version))
    if mutable_dict['conda_packages']:
        # 在 conda_packages 前添加 '[conda]' 的前缀
        mutable_dict['conda_packages'] = prepend(mutable_dict['conda_packages'],
                                                 '[conda] ')
    
    # 将 mutable_dict 中的 'cpu_info' 设置为 envinfo.cpu_info 的值
    mutable_dict['cpu_info'] = envinfo.cpu_info
    
    # 使用 mutable_dict 中的数据格式化 env_info_fmt，并返回格式化后的字符串
    return env_info_fmt.format(**mutable_dict)
def get_pretty_env_info():
    """
    返回一个包含环境信息的格式化字符串。

    该函数通过调用 `get_env_info` 函数获取环境信息，然后将其格式化为易读的字符串。
    获取的环境信息在 `get_env_info` 文档中列出。
    此函数在执行 `python collect_env.py` 时使用，用于报告错误时收集环境信息。

    Returns:
        str: 包含环境信息的格式化字符串。
    """
    return pretty_str(get_env_info())


def main():
    print("Collecting environment information...")
    output = get_pretty_env_info()
    print(output)

    if TORCH_AVAILABLE and hasattr(torch, 'utils') and hasattr(torch.utils, '_crash_handler'):
        minidump_dir = torch.utils._crash_handler.DEFAULT_MINIDUMP_DIR
        if sys.platform == "linux" and os.path.exists(minidump_dir):
            dumps = [os.path.join(minidump_dir, dump) for dump in os.listdir(minidump_dir)]
            latest = max(dumps, key=os.path.getctime)
            ctime = os.path.getctime(latest)
            creation_time = datetime.datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
            msg = "\n*** Detected a minidump at {} created on {}, ".format(latest, creation_time) + \
                  "if this is related to your bug please include it when you file a report ***"
            print(msg, file=sys.stderr)

if __name__ == '__main__':
    main()
```
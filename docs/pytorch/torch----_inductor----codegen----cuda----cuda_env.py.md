# `.\pytorch\torch\_inductor\codegen\cuda\cuda_env.py`

```py
# 导入 functools 模块，用于支持 LRU 缓存功能
# 导入 logging 模块，用于记录日志信息
# 导入 typing 模块中的 Optional 类型，表示返回值可以为 None
# 导入 torch 模块，用于与 CUDA 相关的操作和信息获取
# 从相对路径 ... 中导入 config 模块

# 获取当前模块的日志记录器对象
log = logging.getLogger(__name__)


# 获取 CUDA 架构信息的函数，返回一个可选的字符串
def get_cuda_arch() -> Optional[str]:
    try:
        # 从配置中获取 CUDA 架构信息
        cuda_arch = config.cuda.arch
        if cuda_arch is None:
            # 如果配置中未指定 CUDA 架构，则获取第一个可见设备的计算能力
            major, minor = torch.cuda.get_device_capability(0)
            return str(major * 10 + minor)
        return str(cuda_arch)
    except Exception as e:
        # 记录获取 CUDA 架构信息时的错误日志
        log.error("Error getting cuda arch: %s", e)
        return None


# 获取 CUDA 版本信息的函数，返回一个可选的字符串
def get_cuda_version() -> Optional[str]:
    try:
        # 从配置中获取 CUDA 版本信息
        cuda_version = config.cuda.version
        if cuda_version is None:
            # 如果配置中未指定 CUDA 版本，则获取当前 torch 版本的 CUDA 版本
            cuda_version = torch.version.cuda
        return cuda_version
    except Exception as e:
        # 记录获取 CUDA 版本信息时的错误日志
        log.error("Error getting cuda version: %s", e)
        return None


# 使用 LRU 缓存装饰器，缓存 nvcc_exist 函数的结果
@functools.lru_cache(None)
# 检查是否存在 nvcc 可执行文件的函数，接受一个可选的 nvcc 路径参数，默认为 "nvcc" 字符串，返回布尔值
def nvcc_exist(nvcc_path: str = "nvcc") -> bool:
    # 如果 nvcc_path 参数为 None，则直接返回 False
    if nvcc_path is None:
        return False
    import subprocess

    # 调用系统命令 "which nvcc_path"，将标准输出和标准错误重定向到 /dev/null
    res = subprocess.call(
        ["which", nvcc_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    # 判断命令执行结果是否为 0，如果是则表示 nvcc 存在，返回 True，否则返回 False
    return res == 0
```
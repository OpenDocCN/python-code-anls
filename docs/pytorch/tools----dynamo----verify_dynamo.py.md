# `.\pytorch\tools\dynamo\verify_dynamo.py`

```
# 导入必要的标准库
import os  # 提供与操作系统交互的功能
import re  # 提供正则表达式操作支持
import subprocess  # 允许在子进程中执行命令
import sys  # 提供与 Python 解释器交互的功能
import traceback  # 提供打印异常调用栈的功能
import warnings  # 提供警告处理的功能

# 最低要求的 CUDA 版本和 ROCm 版本
MIN_CUDA_VERSION = "11.6"
MIN_ROCM_VERSION = "5.4"
MIN_PYTHON_VERSION = (3, 8)

# 自定义的异常类，用于验证 Dynamo
class VerifyDynamoError(BaseException):
    pass

# 检查 Python 版本是否符合最低要求
def check_python():
    if sys.version_info < MIN_PYTHON_VERSION:
        raise VerifyDynamoError(
            f"Python version not supported: {sys.version_info} "
            f"- minimum requirement: {MIN_PYTHON_VERSION}"
        )
    return sys.version_info  # 返回当前 Python 解释器的版本信息

# 检查 Torch 库的版本
def check_torch():
    import torch  # 导入 Torch 库
    return torch.__version__  # 返回当前 Torch 库的版本信息

# 获取当前系统中的 CUDA 版本
def get_cuda_version():
    from torch.torch_version import TorchVersion  # 导入 TorchVersion 类
    from torch.utils import cpp_extension  # 导入 cpp_extension 模块

    CUDA_HOME = cpp_extension._find_cuda_home()  # 查找 CUDA 安装路径
    if not CUDA_HOME:
        raise VerifyDynamoError(cpp_extension.CUDA_NOT_FOUND_MESSAGE)

    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")  # 构建 nvcc 的完整路径
    cuda_version_str = (
        subprocess.check_output([nvcc, "--version"])
        .strip()
        .decode(*cpp_extension.SUBPROCESS_DECODE_ARGS)
    )  # 运行 `nvcc --version` 命令，获取 CUDA 版本信息并解析
    cuda_version = re.search(r"release (\d+[.]\d+)", cuda_version_str)  # 从输出中提取 CUDA 版本号
    if cuda_version is None:
        raise VerifyDynamoError("CUDA version not found in `nvcc --version` output")

    cuda_str_version = cuda_version.group(1)  # 获取匹配到的 CUDA 版本号
    return TorchVersion(cuda_str_version)  # 返回 TorchVersion 对象

# 获取当前系统中的 ROCm 版本
def get_rocm_version():
    from torch.torch_version import TorchVersion  # 导入 TorchVersion 类
    from torch.utils import cpp_extension  # 导入 cpp_extension 模块

    ROCM_HOME = cpp_extension._find_rocm_home()  # 查找 ROCm 安装路径
    if not ROCM_HOME:
        raise VerifyDynamoError(
            "ROCM was not found on the system, please set ROCM_HOME environment variable"
        )

    hipcc = os.path.join(ROCM_HOME, "bin", "hipcc")  # 构建 hipcc 的完整路径
    hip_version_str = (
        subprocess.check_output([hipcc, "--version"])
        .strip()
        .decode(*cpp_extension.SUBPROCESS_DECODE_ARGS)
    )  # 运行 `hipcc --version` 命令，获取 ROCm 版本信息并解析
    hip_version = re.search(r"HIP version: (\d+[.]\d+)", hip_version_str)  # 从输出中提取 HIP 版本号
    if hip_version is None:
        raise VerifyDynamoError("HIP version not found in `hipcc --version` output")

    hip_str_version = hip_version.group(1)  # 获取匹配到的 HIP 版本号
    return TorchVersion(hip_str_version)  # 返回 TorchVersion 对象

# 检查当前系统是否支持 CUDA
def check_cuda():
    import torch  # 导入 Torch 库
    from torch.torch_version import TorchVersion  # 导入 TorchVersion 类

    if not torch.cuda.is_available() or torch.version.hip is not None:
        return None  # 如果当前系统不支持 CUDA 或者正在使用 HIP，则返回 None

    torch_cuda_ver = TorchVersion(torch.version.cuda)  # 获取 Torch 报告的 CUDA 版本

    # 检查 Torch 报告的 CUDA 版本是否与系统中的 CUDA 版本匹配
    cuda_ver = get_cuda_version()  # 获取系统中的 CUDA 版本
    if cuda_ver != torch_cuda_ver:
        # raise VerifyDynamoError(
        warnings.warn(
            f"CUDA version mismatch, `torch` version: {torch_cuda_ver}, env version: {cuda_ver}"
        )

    if torch_cuda_ver < MIN_CUDA_VERSION:
        # raise VerifyDynamoError(
        warnings.warn(
            f"(`torch`) CUDA version not supported: {torch_cuda_ver} "
            f"- minimum requirement: {MIN_CUDA_VERSION}"
        )
    # 如果当前 CUDA 版本小于最低要求版本 MIN_CUDA_VERSION
    if cuda_ver < MIN_CUDA_VERSION:
        # 引发警告，提示 CUDA 版本不受支持
        # raise VerifyDynamoError(
        warnings.warn(
            # 格式化字符串，显示当前 CUDA 版本和最低要求版本
            f"(env) CUDA version not supported: {cuda_ver} "
            f"- minimum requirement: {MIN_CUDA_VERSION}"
        )

    # 如果使用的是 PyTorch 的 HIP 版本，则返回字符串 "None"，否则返回当前 CUDA 版本
    return cuda_ver if torch.version.hip is None else "None"
# 检查 ROCm 环境是否符合要求，包括检查 Torch 是否支持 ROCm
def check_rocm():
    import torch  # 导入 torch 库
    from torch.torch_version import TorchVersion  # 从 torch.torch_version 模块导入 TorchVersion 类

    # 如果 CUDA 不可用或者 Torch 的 HIP 版本为 None，则返回 None
    if not torch.cuda.is_available() or torch.version.hip is None:
        return None

    # 从完整字符串中提取主要的 ROCm 版本
    torch_rocm_ver = TorchVersion(".".join(list(torch.version.hip.split(".")[0:2])))

    # 检查 Torch ROCm 版本是否与系统 ROCm 版本匹配
    rocm_ver = get_rocm_version()
    if rocm_ver != torch_rocm_ver:
        warnings.warn(
            f"ROCm version mismatch, `torch` version: {torch_rocm_ver}, env version: {rocm_ver}"
        )

    # 如果 Torch ROCm 版本低于最低要求 MIN_ROCM_VERSION，则发出警告
    if torch_rocm_ver < MIN_ROCM_VERSION:
        warnings.warn(
            f"(`torch`) ROCm version not supported: {torch_rocm_ver} "
            f"- minimum requirement: {MIN_ROCM_VERSION}"
        )

    # 如果环境 ROCm 版本低于最低要求 MIN_ROCM_VERSION，则发出警告
    if rocm_ver < MIN_ROCM_VERSION:
        warnings.warn(
            f"(env) ROCm version not supported: {rocm_ver} "
            f"- minimum requirement: {MIN_ROCM_VERSION}"
        )

    # 返回环境的 ROCm 版本（如果 Torch 的 HIP 版本存在），否则返回字符串 "None"
    return rocm_ver if torch.version.hip else "None"


# 检查指定的后端、设备和错误消息的 Dynamo 环境
def check_dynamo(backend, device, err_msg) -> None:
    import torch  # 导入 torch 库

    # 如果设备是 "cuda" 且 CUDA 不可用，则打印消息并返回
    if device == "cuda" and not torch.cuda.is_available():
        print(f"CUDA not available -- skipping CUDA check on {backend} backend\n")
        return

    try:
        import torch._dynamo as dynamo  # 导入 torch._dynamo 模块

        # 如果设备是 "cuda"，检查是否支持 Triton，否则打印警告消息并返回
        if device == "cuda":
            from torch.utils._triton import has_triton  # 导入 has_triton 函数

            if not has_triton():
                print(
                    f"WARNING: CUDA available but triton cannot be used. "
                    f"Your GPU may not be supported. "
                    f"Skipping CUDA check on {backend} backend\n"
                )
                return

        dynamo.reset()  # 重置 dynamo 环境

        # 定义一个通过 dynamo 优化的函数 fn
        @dynamo.optimize(backend, nopython=True)
        def fn(x):
            return x + x

        # 定义一个简单的 Module 类，并通过 dynamo 优化
        class Module(torch.nn.Module):
            def forward(self, x):
                return x + x

        mod = Module()  # 创建 Module 实例
        opt_mod = dynamo.optimize(backend, nopython=True)(mod)  # 通过 dynamo 优化 mod

        # 对 fn 和 opt_mod 进行测试
        for f in (fn, opt_mod):
            x = torch.randn(10, 10).to(device)  # 创建 tensor 并移到指定设备
            x.requires_grad = True  # 设置 tensor 需要梯度
            y = f(x)  # 计算函数 f 的输出
            torch.testing.assert_close(y, x + x)  # 测试计算结果是否正确
            z = y.sum()  # 对 y 求和
            z.backward()  # 反向传播
            torch.testing.assert_close(x.grad, 2 * torch.ones_like(x))  # 测试梯度是否正确

    except Exception:
        # 捕获异常并将 traceback 信息写入 stderr，同时输出错误消息
        sys.stderr.write(traceback.format_exc() + "\n" + err_msg + "\n\n")
        sys.exit(1)


# 定义 _SANITY_CHECK_ARGS 常量，包含各种后端和设备组合的错误消息
_SANITY_CHECK_ARGS = (
    ("eager", "cpu", "CPU eager sanity check failed"),
    ("eager", "cuda", "CUDA eager sanity check failed"),
    ("aot_eager", "cpu", "CPU aot_eager sanity check failed"),
    ("aot_eager", "cuda", "CUDA aot_eager sanity check failed"),
    ("inductor", "cpu", "CPU inductor sanity check failed"),
    (
        "inductor",
        "cuda",
        "CUDA inductor sanity check failed\n"
        + "NOTE: Please check that you installed the correct hash/version of `triton`",
    ),
)


# 主函数入口
def main() -> None:
    python_ver = check_python()  # 检查 Python 版本
    torch_ver = check_torch()  # 检查 Torch 版本
    # 调用函数检查当前系统上的 CUDA 版本
    cuda_ver = check_cuda()
    # 调用函数检查当前系统上的 ROCm 版本（AMD GPU 的类似 CUDA 实现）
    rocm_ver = check_rocm()
    # 打印 Python 解释器的主要版本号、`torch` 库的版本号、CUDA 版本号和 ROCm 版本号
    print(
        f"Python version: {python_ver.major}.{python_ver.minor}.{python_ver.micro}\n"
        f"`torch` version: {torch_ver}\n"
        f"CUDA version: {cuda_ver}\n"
        f"ROCM version: {rocm_ver}\n"
    )
    # 遍历 `_SANITY_CHECK_ARGS` 列表中的参数元组
    for args in _SANITY_CHECK_ARGS:
        # 如果 Python 版本高于或等于 3.13，则发出警告并跳过当前参数组的检查
        if sys.version_info >= (3, 13):
            warnings.warn("Dynamo not yet supported in Python 3.13. Skipping check.")
            continue
        # 否则，调用 `check_dynamo` 函数并传入当前参数组的参数进行检查
        check_dynamo(*args)
    # 所有必要的检查都通过后打印消息
    print("All required checks passed")
# 如果当前脚本作为主程序运行（而不是作为模块被导入），则执行 main 函数
if __name__ == "__main__":
    main()
```
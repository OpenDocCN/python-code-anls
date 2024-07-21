# `.\pytorch\torch\testing\_internal\common_cuda.py`

```
# mypy: ignore-errors
# 忽略类型检查错误，这可能是因为此文件在导入时需要初始化 CUDA 上下文

r"""This file is allowed to initialize CUDA context when imported."""
# 文件注释，指出此文件在导入时允许初始化 CUDA 上下文

import functools
import torch
import torch.cuda
from torch.testing._internal.common_utils import LazyVal, TEST_NUMBA, TEST_WITH_ROCM, TEST_CUDA, IS_WINDOWS
import inspect
import contextlib
import os

# 检查当前是否已经初始化了 CUDA
CUDA_ALREADY_INITIALIZED_ON_IMPORT = torch.cuda.is_initialized()

# 检查是否支持多GPU测试
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2
# 如果支持 CUDA，则指定第一个 CUDA 设备
CUDA_DEVICE = torch.device("cuda:0") if TEST_CUDA else None
# 如果目标是 ROCm，则 TEST_CUDNN 相当于 TEST_MIOPEN
if TEST_WITH_ROCM:
    TEST_CUDNN = LazyVal(lambda: TEST_CUDA)
else:
    # 检查是否支持 cuDNN，前提是支持 CUDA 并且 cuDNN 可接受
    TEST_CUDNN = LazyVal(lambda: TEST_CUDA and torch.backends.cudnn.is_acceptable(torch.tensor(1., device=CUDA_DEVICE)))

# 获取 cuDNN 的版本信息
TEST_CUDNN_VERSION = LazyVal(lambda: torch.backends.cudnn.version() if TEST_CUDNN else 0)

# 检查当前 CUDA 设备是否支持指定的计算能力
SM53OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (5, 3))
SM60OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (6, 0))
SM70OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 0))
SM75OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 5))
SM80OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0))
SM90OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0))

# 检查当前设备是否为 Jetson
IS_JETSON = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() in [(7, 2), (8, 7)])

def evaluate_gfx_arch_exact(matching_arch):
    # 如果不支持 CUDA，直接返回 False
    if not torch.cuda.is_available():
        return False
    # 获取当前 CUDA 设备的架构名称
    gcn_arch_name = torch.cuda.get_device_properties('cuda').gcnArchName
    # 从环境变量中获取或使用默认的 GCN 架构名称
    arch = os.environ.get('PYTORCH_DEBUG_FLASH_ATTENTION_GCN_ARCH_OVERRIDE', gcn_arch_name)
    # 返回当前架构是否与指定的匹配
    return arch == matching_arch

# 检查是否为 GFX90A 架构
GFX90A_Exact = LazyVal(lambda: evaluate_gfx_arch_exact('gfx90a:sramecc+:xnack-'))
# 检查是否为 GFX942 架构
GFX942_Exact = LazyVal(lambda: evaluate_gfx_arch_exact('gfx942:sramecc+:xnack-'))

def evaluate_platform_supports_flash_attention():
    # 如果是 ROCm 环境，检查是否支持指定的 GCN 架构
    if TEST_WITH_ROCM:
        return evaluate_gfx_arch_exact('gfx90a:sramecc+:xnack-') or evaluate_gfx_arch_exact('gfx942:sramecc+:xnack-')
    # 如果是 CUDA 环境，检查是否不是 Windows 系统且支持 SM80 或更新版本的 GPU
    if TEST_CUDA:
        return not IS_WINDOWS and SM80OrLater
    return False

def evaluate_platform_supports_efficient_attention():
    # 如果是 ROCm 环境，检查是否支持指定的 GCN 架构
    if TEST_WITH_ROCM:
        return evaluate_gfx_arch_exact('gfx90a:sramecc+:xnack-') or evaluate_gfx_arch_exact('gfx942:sramecc+:xnack-')
    # 如果是 CUDA 环境，返回 True
    if TEST_CUDA:
        return True
    return False

# 检查平台是否支持闪存注意力机制
PLATFORM_SUPPORTS_FLASH_ATTENTION: bool = LazyVal(lambda: evaluate_platform_supports_flash_attention())
# 检查平台是否支持内存高效的注意力机制
PLATFORM_SUPPORTS_MEM_EFF_ATTENTION: bool = LazyVal(lambda: evaluate_platform_supports_efficient_attention())
# TODO(eqy): gate this against a cuDNN version
# 设置一个布尔类型的常量，表示当前平台是否支持使用 cuDNN 的注意力机制
PLATFORM_SUPPORTS_CUDNN_ATTENTION: bool = LazyVal(lambda: TEST_CUDA and not TEST_WITH_ROCM and
                                                  torch.backends.cuda.cudnn_sdp_enabled())

# 设置一个布尔类型的常量，表示当前平台是否支持融合的注意力机制
# 这个条件始终等同于 PLATFORM_SUPPORTS_MEM_EFF_ATTENTION，但为了逻辑上的清晰性保持分开
PLATFORM_SUPPORTS_FUSED_ATTENTION: bool = LazyVal(lambda: PLATFORM_SUPPORTS_FLASH_ATTENTION or PLATFORM_SUPPORTS_MEM_EFF_ATTENTION)

# 设置一个布尔类型的常量，表示当前平台是否支持融合的 SDPA（Scaled Dot-Product Attention）
PLATFORM_SUPPORTS_FUSED_SDPA: bool = TEST_CUDA and not TEST_WITH_ROCM

# 设置一个布尔类型的常量，表示当前平台是否支持 BF16（Bfloat16）
PLATFORM_SUPPORTS_BF16: bool = LazyVal(lambda: TEST_CUDA and SM80OrLater)

# 评估当前平台是否支持 FP8（Float16）
def evaluate_platform_supports_fp8():
    if torch.cuda.is_available():
        if torch.version.hip:
            return 'gfx94' in torch.cuda.get_device_properties(0).gcnArchName
        else:
            return SM90OrLater or torch.cuda.get_device_capability() == (8, 9)
    return False

# 设置一个布尔类型的常量，表示当前平台是否支持 FP8（Float16）
PLATFORM_SUPPORTS_FP8: bool = LazyVal(lambda: evaluate_platform_supports_fp8())

# 如果 TEST_NUMBA 为真，尝试导入 numba.cuda，并检查是否可用，否则设为 False
if TEST_NUMBA:
    try:
        import numba.cuda
        TEST_NUMBA_CUDA = numba.cuda.is_available()
    except Exception as e:
        TEST_NUMBA_CUDA = False
        TEST_NUMBA = False
else:
    TEST_NUMBA_CUDA = False

# 用于确保在 `initialize_cuda_context_rng` 中 CUDA 上下文和 RNG 已经初始化
__cuda_ctx_rng_initialized = False


# 在每个 GPU 上初始化 CUDA 上下文和 RNG，确保它们已经被初始化
def initialize_cuda_context_rng():
    global __cuda_ctx_rng_initialized
    assert TEST_CUDA, 'CUDA must be available when calling initialize_cuda_context_rng'
    if not __cuda_ctx_rng_initialized:
        # 初始化 CUDA 上下文和 RNG 以进行内存测试
        for i in range(torch.cuda.device_count()):
            torch.randn(1, device=f"cuda:{i}")
        __cuda_ctx_rng_initialized = True


# 检测硬件是否启用了 TF32 数学模式，仅在以下条件下启用：
# - CUDA 版本 >= 11
# - GPU 架构 >= Ampere
def tf32_is_not_fp32():
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:
        return False
    if int(torch.version.cuda.split('.')[0]) < 11:
        return False
    return True


# 上下文管理器，用于在代码块中关闭 TF32 数学模式
@contextlib.contextmanager
def tf32_off():
    old_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(enabled=None, benchmark=None, deterministic=None, allow_tf32=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32_matmul


# 上下文管理器，用于在代码块中启用 TF32 数学模式
@contextlib.contextmanager
def tf32_on(self, tf32_precision=1e-5):
    old_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    old_precision = self.precision
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        self.precision = tf32_precision
        with torch.backends.cudnn.flags(enabled=None, benchmark=None, deterministic=None, allow_tf32=True):
            yield
    # 最终执行块，无论是否发生异常，都会执行这里的代码
    finally:
        # 恢复之前保存的 allow_tf32_matmul 设置
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32_matmul
        # 恢复之前保存的 precision 设置
        self.precision = old_precision
# 定义一个装饰器函数 tf32_on_and_off，用于在测试中控制 TF32 模式的开启和关闭
def tf32_on_and_off(tf32_precision=1e-5):
    # 定义一个内部函数，用于在 TF32 关闭的情况下执行测试函数
    def with_tf32_disabled(self, function_call):
        # 使用 tf32_off 上下文管理器，执行传入的函数调用
        with tf32_off():
            function_call()

    # 定义一个内部函数，用于在 TF32 开启的情况下执行测试函数
    def with_tf32_enabled(self, function_call):
        # 使用 tf32_on 上下文管理器，传入当前对象和 TF32 精度参数，执行函数调用
        with tf32_on(self, tf32_precision):
            function_call()

    # 定义一个装饰器函数，接收测试函数作为参数
    def wrapper(f):
        # 获取测试函数的参数信息
        params = inspect.signature(f).parameters
        arg_names = tuple(params.keys())

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            # 将位置参数与关键字参数映射到 kwargs 中
            for k, v in zip(arg_names, args):
                kwargs[k] = v
            
            # 检查当前条件是否满足 TF32 模式的使用要求
            cond = tf32_is_not_fp32()
            if 'device' in kwargs:
                cond = cond and (torch.device(kwargs['device']).type == 'cuda')
            if 'dtype' in kwargs:
                cond = cond and (kwargs['dtype'] in {torch.float32, torch.complex64})
            
            # 根据条件选择执行测试函数的方式
            if cond:
                # 如果满足条件，先执行 TF32 关闭的函数调用
                with_tf32_disabled(kwargs['self'], lambda: f(**kwargs))
                # 然后执行 TF32 开启的函数调用
                with_tf32_enabled(kwargs['self'], lambda: f(**kwargs))
            else:
                # 如果不满足条件，直接执行函数调用
                f(**kwargs)

        return wrapped
    return wrapper


# 定义一个装饰器函数 with_tf32_off，用于在测试中临时关闭 TF32 模式
def with_tf32_off(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        # 使用 tf32_off 上下文管理器，执行传入的测试函数调用
        with tf32_off():
            return f(*args, **kwargs)

    return wrapped


# 定义一个函数 _get_magma_version，用于获取当前环境中的 Magma 库版本信息
def _get_magma_version():
    # 检查当前环境中是否包含 Magma 库信息
    if 'Magma' not in torch.__config__.show():
        # 如果不包含，则返回版本信息 (0, 0)
        return (0, 0)
    # 获取字符串 "Magma " 在 torch.__config__.show() 返回结果中的位置
    position = torch.__config__.show().find('Magma ')
    # 根据找到的位置，从该位置开始截取版本信息字符串，直到遇到换行符为止，并取第一行
    version_str = torch.__config__.show()[position + len('Magma '):].split('\n')[0]
    # 将版本信息字符串按 "." 分割成数字字符串列表，然后转换为整数元组并返回
    return tuple(int(x) for x in version_str.split("."))
# 获取当前安装的 Torch CUDA 版本，如果未安装 CUDA，则返回 (0, 0)
def _get_torch_cuda_version():
    if torch.version.cuda is None:
        return (0, 0)
    cuda_version = str(torch.version.cuda)
    return tuple(int(x) for x in cuda_version.split("."))

# 获取当前安装的 Torch ROCm 版本，如果未启用 ROCm，则返回 (0, 0)
def _get_torch_rocm_version():
    if not TEST_WITH_ROCM:
        return (0, 0)
    rocm_version = str(torch.version.hip)
    rocm_version = rocm_version.split("-")[0]    # 忽略 git 提交 SHA
    return tuple(int(x) for x in rocm_version.split("."))

# 检查当前环境是否支持 cusparse 通用库
def _check_cusparse_generic_available():
    return not TEST_WITH_ROCM

# 检查当前环境是否支持 hipsparse 通用库
def _check_hipsparse_generic_available():
    if not TEST_WITH_ROCM:
        return False

    rocm_version = str(torch.version.hip)
    rocm_version = rocm_version.split("-")[0]    # 忽略 git 提交 SHA
    rocm_version_tuple = tuple(int(x) for x in rocm_version.split("."))
    return not (rocm_version_tuple is None or rocm_version_tuple < (5, 1))

# 检测是否启用 cusparse 通用库
TEST_CUSPARSE_GENERIC = _check_cusparse_generic_available()

# 检测是否启用 hipsparse 通用库
TEST_HIPSPARSE_GENERIC = _check_hipsparse_generic_available()

# 创建具有不同优化器的模型和优化器，用于测试在不同设备上的表现
def _create_scaling_models_optimizers(device="cuda", optimizer_ctor=torch.optim.SGD, optimizer_kwargs=None):
    # 创建一个使用 scaling 的模型+优化器和一个不使用 scaling 的控制模型+优化器，用于比较
    mod_control = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).to(device=device)
    mod_scaling = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).to(device=device)
    with torch.no_grad():
        for c, s in zip(mod_control.parameters(), mod_scaling.parameters()):
            s.copy_(c)

    kwargs = {"lr": 1.0}
    if optimizer_kwargs is not None:
        kwargs.update(optimizer_kwargs)
    opt_control = optimizer_ctor(mod_control.parameters(), **kwargs)
    opt_scaling = optimizer_ctor(mod_scaling.parameters(), **kwargs)

    return mod_control, mod_scaling, opt_control, opt_scaling

# 创建一个在不同测试中使用的 scaling 情况
def _create_scaling_case(device="cuda", dtype=torch.float, optimizer_ctor=torch.optim.SGD, optimizer_kwargs=None):
    data = [(torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
            (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
            (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
            (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device))]

    loss_fn = torch.nn.MSELoss().to(device)

    skip_iter = 2

    return _create_scaling_models_optimizers(
        device=device, optimizer_ctor=optimizer_ctor, optimizer_kwargs=optimizer_kwargs,
    ) + (data, loss_fn, skip_iter)


# 导入此模块时，如果 CUDA 尚未初始化，应断言 CUDA 未初始化
if not CUDA_ALREADY_INITIALIZED_ON_IMPORT:
    assert not torch.cuda.is_initialized()
```
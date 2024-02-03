# `bitsandbytes\bitsandbytes\cextension.py`

```py
# 导入必要的库
import ctypes as ct
from warnings import warn
import torch
from bitsandbytes.cuda_setup.main import CUDASetup

# 获取 CUDA 设置的实例
setup = CUDASetup.get_instance()
# 如果 CUDA 设置未初始化，则运行 CUDA 设置
if setup.initialized != True:
    setup.run_cuda_setup()

# 获取 CUDA 库
lib = setup.lib
try:
    # 如果库为空且 CUDA 可用，则生成指令并打印日志堆栈
    if lib is None and torch.cuda.is_available():
        CUDASetup.get_instance().generate_instructions()
        CUDASetup.get_instance().print_log_stack()
        # 抛出运行时错误，提供更多信息和建议
        raise RuntimeError('''
        CUDA Setup failed despite GPU being available. Please run the following command to get more information:

        python -m bitsandbytes

        Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
        to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
        and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues''')
    # 检查库是否存在，如果未找到则会出错，COMPILED_WITH_CUDA=False
    _ = lib.cadam32bit_grad_fp32
    # 设置返回类型
    lib.get_context.restype = ct.c_void_p
    lib.get_cusparse.restype = ct.c_void_p
    lib.cget_managed_ptr.restype = ct.c_void_p
    COMPILED_WITH_CUDA = True
except AttributeError as ex:
    # 如果没有 GPU 支持，则发出警告
    warn("The installed version of bitsandbytes was compiled without GPU support. "
        "8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.")
    COMPILED_WITH_CUDA = False
    # 打印错误信息
    print(str(ex))

# 检查错误后打印设置详情，避免重复打印
#if 'BITSANDBYTES_NOWELCOME' not in os.environ or str(os.environ['BITSANDBYTES_NOWELCOME']) == '0':
    #setup.print_log_stack()
```
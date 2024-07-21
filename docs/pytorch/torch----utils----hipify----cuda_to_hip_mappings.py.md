# `.\pytorch\torch\utils\hipify\cuda_to_hip_mappings.py`

```
# 导入 collections 模块，用于创建有序字典
import collections

# 从当前目录下的 constants 模块导入多个常量
from .constants import (API_BLAS, API_C10, API_CAFFE2, API_DRIVER, API_FFT,
                        API_PYTORCH, API_RAND, API_ROCTX, API_RTC, API_RUNTIME,
                        API_SPECIAL, API_ROCMSMI, CONV_CACHE, CONV_CONTEXT, CONV_D3D9,
                        CONV_D3D10, CONV_D3D11, CONV_DEF, CONV_DEVICE,
                        CONV_DEVICE_FUNC, CONV_EGL, CONV_ERROR, CONV_EVENT,
                        CONV_EXEC, CONV_GL, CONV_GRAPHICS, CONV_INCLUDE,
                        CONV_INCLUDE_CUDA_MAIN_H, CONV_INIT, CONV_JIT,
                        CONV_MATH_FUNC, CONV_MEM, CONV_MODULE,
                        CONV_NUMERIC_LITERAL, CONV_OCCUPANCY, CONV_OTHER,
                        CONV_PEER, CONV_SPECIAL_FUNC, CONV_STREAM,
                        CONV_SURFACE, CONV_TEX, CONV_THREAD, CONV_TYPE,
                        CONV_VDPAU, CONV_VERSION, HIP_UNSUPPORTED)

"""
Mapping of CUDA functions, include files, constants, and types to ROCm/HIP equivalents.
This closely follows the implementation in hipify-clang
https://github.com/ROCm-Developer-Tools/HIP/blob/master/hipify-clang/src/CUDA2HipMap.cpp
and its structure.
There are different maps for fundamental names, include files, identifiers, sparse,
and PyTorch specific translations.
Each of the entries in these maps translates a CUDA string to a tuple containing the
ROCm/HIP string, a type and API annotation and - optionally - an annotation if it is not
supported in ROCm/HIP yet.
"""

# 定义一个有序字典，用于存储应在设备代码中替换的数学函数名称和其对应的 HIP 实现
MATH_TRANSPILATIONS = collections.OrderedDict(
    [
        ("std::max", ("::max")),        # 替换 std::max 为 ::max
        ("std::min", ("::min")),        # 替换 std::min 为 ::min
        ("std::ceil", ("::ceil")),      # 替换 std::ceil 为 ::ceil
        ("std::floor", ("::floor")),    # 替换 std::floor 为 ::floor
        ("std::exp", ("::exp")),        # 替换 std::exp 为 ::exp
        ("std::log", ("::log")),        # 替换 std::log 为 ::log
        ("std::pow", ("::pow")),        # 替换 std::pow 为 ::pow
        ("std::fabs", ("::fabs")),      # 替换 std::fabs 为 ::fabs
        ("std::fmod", ("::fmod")),      # 替换 std::fmod 为 ::fmod
        ("std::remainder", ("::remainder")),  # 替换 std::remainder 为 ::remainder
        ("std::frexp", ("::frexp")),    # 替换 std::frexp 为 ::frexp
    ]
)

# 定义一个空的有序字典，用于存储 CUDA 类型名称映射
CUDA_TYPE_NAME_MAP = collections.OrderedDict(
    []
)

# 定义一个空的有序字典，用于存储 CUDA 包含文件映射
CUDA_INCLUDE_MAP = collections.OrderedDict(
    []
)

# 定义一个空的有序字典，用于存储 CUDA 标识符映射
CUDA_IDENTIFIER_MAP = collections.OrderedDict(
    []
)

# 定义一个空的有序字典，用于存储 CUDA 特殊映射
CUDA_SPECIAL_MAP = collections.OrderedDict(
    []
)

# 定义一个空的有序字典，用于存储 PyTorch 特定映射
PYTORCH_SPECIFIC_MAPPINGS = collections.OrderedDict(
    []
)

# 定义一个空的有序字典，用于存储 Caffe2 特定映射
CAFFE2_SPECIFIC_MAPPINGS = collections.OrderedDict(
    []
)

# 此处需要非常小心处理。类似 CAFFE2_SPECIFIC_MAPPINGS 中的全局转换在 PyTorch 中目前不受支持，
# 因为一个 CUDA 的正则表达式也会匹配到像 CUDAGuard.h 这样的文件名，
# 但 HIPIFY 脚本目前并未移动文件，因此替换将是无效的。
# 我们明确列出每个可能在外部使用的 c10/cuda 目录中的标识符和文件，并以这种方式进行替换。
#
# 注意：如果希望一个转换仅适用于 c10/ 目录，请将其放在 API_CAFFE2 中。
C10_MAPPINGS = collections.OrderedDict(
    []
)
# 定义一个列表，包含了一系列 CUDA 到 HIP 映射的字典
# 注意：C10 映射比 Caffe2 映射更为具体，因此排在前面
CUDA_TO_HIP_MAPPINGS = [
    CUDA_IDENTIFIER_MAP,         # 包含 CUDA 标识符映射的字典
    CUDA_TYPE_NAME_MAP,          # 包含 CUDA 类型名称映射的字典
    CUDA_INCLUDE_MAP,            # 包含 CUDA 头文件包含路径映射的字典
    CUDA_SPECIAL_MAP,            # 包含 CUDA 特殊映射的字典
    C10_MAPPINGS,                # 包含 C10 映射的字典
    PYTORCH_SPECIFIC_MAPPINGS,   # 包含 PyTorch 特定映射的字典
    CAFFE2_SPECIFIC_MAPPINGS,    # 包含 Caffe2 特定映射的字典
]
```
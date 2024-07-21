# `.\pytorch\torch\_inductor\cpu_vec_isa.py`

```py
# mypy: allow-untyped-defs
import dataclasses  # 导入 dataclasses 模块，用于支持数据类（data class）
import functools  # 导入 functools 模块，用于高阶函数（higher-order functions）的实现
import os  # 导入 os 模块，提供了与操作系统交互的功能
import platform  # 导入 platform 模块，用于访问平台相关的操作系统特性

import re  # 导入 re 模块，提供了正则表达式操作的支持
import subprocess  # 导入 subprocess 模块，用于生成子进程，执行外部命令
import sys  # 导入 sys 模块，提供了对解释器的访问，以及与解释器交互的功能
from typing import Any, Callable, Dict, List  # 导入 typing 模块，用于类型提示

import torch  # 导入 torch 模块，PyTorch 深度学习框架的核心功能
from torch._inductor import config  # 导入 config 模块，可能用于配置相关功能

_IS_WINDOWS = sys.platform == "win32"  # 检测操作系统是否为 Windows

def _get_isa_dry_compile_fingerprint(isa_flags: str) -> str:
    # 获取 ISA 干编译的指纹信息，用于标识编译环境和版本信息
    # 参考 GitHub 问题链接：https://github.com/pytorch/pytorch/issues/100378
    # 干编译主要用于检测 ISA 的编译能力，记录编译器版本、ISA 选项和 PyTorch 版本信息，
    # 生成用于输出二进制文件路径的哈希值，以优化并跳过已编译的二进制文件
    from torch._inductor.cpp_builder import get_compiler_version_info, get_cpp_compiler

    compiler_info = get_compiler_version_info(get_cpp_compiler())  # 获取编译器版本信息
    torch_version = torch.__version__  # 获取当前 PyTorch 版本
    fingerprint = f"{compiler_info}={isa_flags}={torch_version}"
    return fingerprint  # 返回生成的指纹字符串信息

class VecISA:
    _bit_width: int  # 向量 ISA 的位宽
    _macro: List[str]  # 构建宏的列表
    _arch_flags: str  # 架构标志
    _dtype_nelements: Dict[torch.dtype, int]  # 数据类型对应的元素数量字典

    # Note [Checking for Vectorized Support in Inductor]
    # TorchInductor CPU vectorization reuses PyTorch vectorization utility functions
    # Hence, TorchInductor would depend on Sleef* to accelerate mathematical functions
    # like exp, pow, sin, cos and etc.
    # But PyTorch and TorchInductor might use different compilers to build code. If
    # PyTorch uses gcc-7/g++-7 to build the release package, the libtorch_cpu.so
    # will not expose the Sleef* AVX512 symbols since gcc-7/g++-7 cannot pass
    # avx512 check in CMake - FindAVX.cmake. But TorchInductor install the latest
    # gcc/g++ compiler by default while it could support the AVX512 compilation.
    # Therefore, there would be a conflict sleef version between PyTorch and
    # TorchInductor. Hence, we dry-compile the following code to check whether current
    # HW platform and PyTorch both could support AVX512 or AVX2. And suppose ARM
    # also needs the logic
    # In fbcode however, we are using the same compiler for pytorch and for inductor codegen,
    # making the runtime check unnecessary.
    _avx_code = """
#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_ZVECTOR) || defined(CPU_CAPABILITY_NEON)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif

alignas(64) float in_out_ptr0[16] = {0.0};

extern "C" void __avx_chk_kernel() {
    auto tmp0 = at::vec::Vectorized<float>(1);
    auto tmp1 = tmp0.exp();
    tmp1.store(in_out_ptr0);
}
"""  # noqa: B950

    _avx_py_load = """
import torch
from ctypes import cdll
cdll.LoadLibrary("__lib_path__")
"""

    def bit_width(self) -> int:
        return self._bit_width  # 返回向量 ISA 的位宽

    def nelements(self, dtype: torch.dtype = torch.float) -> int:
        return self._dtype_nelements[dtype]  # 返回指定数据类型的元素数量

    def build_macro(self) -> List[str]:
        return self._macro  # 返回用于构建的宏列表
    # 返回对象的架构标志字符串
    def build_arch_flags(self) -> str:
        return self._arch_flags

    # 返回对象的哈希值
    def __hash__(self) -> int:
        return hash(str(self))

    # 检查给定代码的构建状态
    def check_build(self, code: str) -> bool:
        # 导入必要的模块和函数
        from torch._inductor.codecache import get_lock_dir, LOCK_TIMEOUT, write
        from torch._inductor.cpp_builder import CppBuilder, CppTorchOptions
        from filelock import FileLock

        # 写入代码到临时文件并获取相关信息
        key, input_path = write(
            code,
            "cpp",
            extra=_get_isa_dry_compile_fingerprint(self._arch_flags),
        )

        # 获取构建锁并等待锁定
        lock_dir = get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
        with lock:
            output_dir = os.path.dirname(input_path)
            # 设置构建选项
            buid_options = CppTorchOptions(vec_isa=self, warning_all=False)
            # 创建 C++ 构建器对象
            x86_isa_help_builder = CppBuilder(
                key,
                [input_path],
                buid_options,
                output_dir,
            )
            try:
                # 获取目标文件路径
                output_path = x86_isa_help_builder.get_target_file_path()
                # 如果目标文件不存在，则执行构建
                if not os.path.isfile(output_path):
                    status, target_file = x86_isa_help_builder.build()

                # 检查构建结果，运行生成的目标文件
                subprocess.check_call(
                    [
                        sys.executable,
                        "-c",
                        VecISA._avx_py_load.replace("__lib_path__", output_path),
                    ],
                    stderr=subprocess.DEVNULL,
                    env={**os.environ, "PYTHONPATH": ":".join(sys.path)},
                )
            except Exception as e:
                # 如果出现异常则返回构建失败
                return False

            # 构建成功则返回 True
            return True

    @functools.lru_cache(None)  # noqa: B019
    # 返回对象是否为真的布尔值
    def __bool__(self) -> bool:
        if config.cpp.vec_isa_ok is not None:
            return config.cpp.vec_isa_ok

        # 在特定环境下直接返回 True
        if config.is_fbcode():
            return True

        # 否则检查对象的构建状态
        return self.check_build(VecISA._avx_code)
# 使用 dataclasses 模块创建一个名为 VecNEON 的数据类，它继承自 VecISA 类
@dataclasses.dataclass
class VecNEON(VecISA):
    # 设置向量的位宽为 256，用于在 aten/src/ATen/cpu/vec/vec256/vec256_float_neon.h 中的计算
    _bit_width = 256  
    # 定义一个列表，包含宏定义 "CPU_CAPABILITY_NEON"
    _macro = ["CPU_CAPABILITY_NEON"]
    # 如果操作系统是 darwin（macOS）且处理器是 arm，则添加宏定义 "AT_BUILD_ARM_VEC256_WITH_SLEEF"
    if sys.platform == "darwin" and platform.processor() == "arm":
        _macro.append("AT_BUILD_ARM_VEC256_WITH_SLEEF")
    # 设置架构标志为一个空字符串，但未使用
    _arch_flags = ""  # Unused
    # 设置数据类型到元素个数的映射字典，包括 torch.float、torch.bfloat16 和 torch.float16
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16, torch.float16: 16}

    # 返回字符串 "asimd"，用于检测 armv8-a 内核上的高级 SIMD 存在
    def __str__(self) -> str:
        return "asimd"  # detects the presence of advanced SIMD on armv8-a kernels

    # 设置特殊方法 __hash__，其参数为 VecISA 类型，委托给 VecISA 的 __hash__ 方法
    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


# 使用 dataclasses 模块创建一个名为 VecAVX512 的数据类，它继承自 VecISA 类
@dataclasses.dataclass
class VecAVX512(VecISA):
    # 设置向量的位宽为 512
    _bit_width = 512
    # 定义一个列表，包含宏定义 "CPU_CAPABILITY_AVX512"
    _macro = ["CPU_CAPABILITY_AVX512"]
    # 设置架构标志为一个字符串，根据 _IS_WINDOWS 的值选择不同的标志
    _arch_flags = (
        "-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma"
        if not _IS_WINDOWS
        else "/arch:AVX512"
    )  # TODO: use cflags
    # 设置数据类型到元素个数的映射字典，包括 torch.float、torch.bfloat16 和 torch.float16
    _dtype_nelements = {torch.float: 16, torch.bfloat16: 32, torch.float16: 32}

    # 返回字符串 "avx512"
    def __str__(self) -> str:
        return "avx512"

    # 设置特殊方法 __hash__，其参数为 VecISA 类型，委托给 VecISA 的 __hash__ 方法
    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


# 使用 dataclasses 模块创建一个名为 VecAMX 的数据类，它继承自 VecAVX512 类
@dataclasses.dataclass
class VecAMX(VecAVX512):
    # 扩展父类 VecAVX512 的 _arch_flags，添加额外的标志 "-mamx-tile -mamx-bf16 -mamx-int8"
    _arch_flags = VecAVX512._arch_flags + " -mamx-tile -mamx-bf16 -mamx-int8"

    # 返回字符串 "avx512 amx_tile"，在父类字符串基础上增加 " amx_tile"
    def __str__(self) -> str:
        return super().__str__() + " amx_tile"

    # 设置特殊方法 __hash__，其参数为 VecISA 类型，委托给 VecISA 的 __hash__ 方法
    __hash__: Callable[[VecISA], Any] = VecISA.__hash__

    # 定义一个 C++ 代码块 _amx_code，包含一些 AMX 指令和结构体定义
    _amx_code = """
#include <cstdint>
#include <immintrin.h>

struct amx_tilecfg {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16];
  uint8_t rows[16];
};

extern "C" void __amx_chk_kernel() {
  amx_tilecfg cfg = {0};
  _tile_loadconfig(&cfg);
  _tile_zero(0);
  _tile_dpbf16ps(0, 1, 2);
  _tile_dpbusd(0, 1, 2);
}
"""

    # 使用 functools.lru_cache(None) 装饰器定义 __bool__ 方法，用于判断向量是否可用
    # 如果父类 VecAVX512 的 __bool__ 方法返回 True，并且满足一些条件，则返回 True
    def __bool__(self) -> bool:
        if super().__bool__():
            if config.is_fbcode():
                return False
            if self.check_build(VecAMX._amx_code) and torch.cpu._init_amx():
                return True
        return False


# 使用 dataclasses 模块创建一个名为 VecAVX2 的数据类，它继承自 VecISA 类
@dataclasses.dataclass
class VecAVX2(VecISA):
    # 设置向量的位宽为 256
    _bit_width = 256
    # 定义一个列表，包含宏定义 "CPU_CAPABILITY_AVX2"
    _macro = ["CPU_CAPABILITY_AVX2"]
    # 设置架构标志为一个字符串，根据 _IS_WINDOWS 的值选择不同的标志
    _arch_flags = (
        "-mavx2 -mfma -mf16c" if not _IS_WINDOWS else "/arch:AVX2"
    )  # TODO: use cflags
    # 设置数据类型到元素个数的映射字典，包括 torch.float、torch.bfloat16 和 torch.float16
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16, torch.float16: 16}

    # 返回字符串 "avx2"
    def __str__(self) -> str:
        return "avx2"

    # 设置特殊方法 __hash__，其参数为 VecISA 类型，委托给 VecISA 的 __hash__ 方法
    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


# 使用 dataclasses 模块创建一个名为 VecZVECTOR 的数据类，它继承自 VecISA 类
@dataclasses.dataclass
class VecZVECTOR(VecISA):
    # 设置向量的位宽为 256
    _bit_width = 256
    # 定义一个列表，包含多个宏定义 "CPU_CAPABILITY_ZVECTOR", "CPU_CAPABILITY=ZVECTOR", "HAVE_ZVECTOR_CPU_DEFINITION"
    _macro = [
        "CPU_CAPABILITY_ZVECTOR",
        "CPU_CAPABILITY=ZVECTOR",
        "HAVE_ZVECTOR_CPU_DEFINITION",
    ]
    # 设置架构标志为一个字符串 "-mvx -mzvector"
    _arch_flags = "-mvx -mzvector"
    # 设置数据类型到元素个数的映射字典，包括 torch.float、torch.bfloat16 和 torch.float16
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16, torch.float16: 16}

    # 返回字符串 "zvector"
    def __str__(self) -> str:
        return "zvector"

    # 设置特殊方法 __hash__，其参数为 VecISA 类型，委托给 VecISA 的 __hash__ 方法
    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


# 定义一个名为 InvalidVecISA 的类，它继承自 VecISA 类
class InvalidVecISA(VecISA):
    # 设置向量的位宽为 0
    _bit_width = 0
    # 定义一个空列表，没有宏定义
    _macro = [""]
    # 设置架构标志为一个空字符串
    _arch_flags = ""
    # 设置数据类型到元素个数的映射字典为空字典
    _dtype_nelements = {}
    # 返回一个字符串，表示对象的字符串表示为"INVALID_VEC_ISA"
    def __str__(self) -> str:
        return "INVALID_VEC_ISA"

    # 返回布尔值 False，用于类型转换为布尔值时
    def __bool__(self) -> bool:  # type: ignore[override]
        return False

    # 指定 __hash__ 属性为 VecISA 类的 __hash__ 方法的可调用对象
    __hash__: Callable[[VecISA], Any] = VecISA.__hash__
# 定义函数 x86_isa_checker，返回支持的指令集列表
def x86_isa_checker() -> List[str]:
    # 初始化支持的指令集列表
    supported_isa: List[str] = []

    # 定义内部函数 _check_and_append_supported_isa，用于检查并添加支持的指令集到目标列表
    def _check_and_append_supported_isa(
        dest: List[str], isa_supported: bool, isa_name: str
    ) -> None:
        # 如果指令集支持，则将其添加到目标列表中
        if isa_supported:
            dest.append(isa_name)

    # 获取当前系统的架构信息
    Arch = platform.machine()

    """
    Arch value is x86_64 on Linux, and the value is AMD64 on Windows.
    """
    # 如果架构不是 x86_64 或者 AMD64，则直接返回已知的支持指令集列表
    if Arch != "x86_64" and Arch != "AMD64":
        return supported_isa

    # 检查当前 CPU 是否支持 AVX2、AVX512 和 AMX_TILE 指令集
    avx2 = torch.cpu._is_cpu_support_avx2()
    avx512 = torch.cpu._is_cpu_support_avx512()
    amx_tile = torch.cpu._is_cpu_support_amx_tile()

    # 分别检查并添加支持的指令集到目标列表
    _check_and_append_supported_isa(supported_isa, avx2, "avx2")
    _check_and_append_supported_isa(supported_isa, avx512, "avx512")
    _check_and_append_supported_isa(supported_isa, amx_tile, "amx_tile")

    # 返回支持的指令集列表
    return supported_isa


# 创建无效的向量指令集对象
invalid_vec_isa = InvalidVecISA()

# 定义支持的向量指令集列表
supported_vec_isa_list = [VecAMX(), VecAVX512(), VecAVX2(), VecNEON()]


# 使用 functools.lru_cache 缓存 cpuinfo 以避免 I/O 开销，同时只缓存关键的 ISA 信息
@functools.lru_cache(None)
# 定义函数 valid_vec_isa_list，返回支持的向量指令集列表
def valid_vec_isa_list() -> List[VecISA]:
    # 初始化 ISA 列表
    isa_list: List[VecISA] = []

    # 如果是 macOS 并且处理器是 arm，添加 NEON 向量指令集
    if sys.platform == "darwin" and platform.processor() == "arm":
        isa_list.append(VecNEON())

    # 如果不是 linux 或者 win32，直接返回 ISA 列表
    if sys.platform not in ["linux", "win32"]:
        return isa_list

    # 获取当前系统的架构信息
    arch = platform.machine()

    # 如果是 s390x 架构，从 /proc/cpuinfo 中读取 CPU 信息
    if arch == "s390x":
        with open("/proc/cpuinfo") as _cpu_info:
            while True:
                line = _cpu_info.readline()
                if not line:
                    break
                # 处理每一行 CPU 信息
                featuresmatch = re.match(r"^features\s*:\s*(.*)$", line)
                if featuresmatch:
                    for group in featuresmatch.groups():
                        if re.search(r"[\^ ]+vxe[\$ ]+", group):
                            isa_list.append(VecZVECTOR())
                            break
    # 如果是 aarch64 架构，添加 NEON 向量指令集
    elif arch == "aarch64":
        isa_list.append(VecNEON())
    # 如果是 x86_64 或者 AMD64 架构
    elif arch in ["x86_64", "AMD64"]:
        """
        arch value is x86_64 on Linux, and the value is AMD64 on Windows.
        """
        # 获取当前系统支持的 x86 指令集列表
        _cpu_supported_x86_isa = x86_isa_checker()

        # 遍历已知的支持向量指令集列表，将当前系统支持的添加到 ISA 列表中
        for isa in supported_vec_isa_list:
            if all(flag in _cpu_supported_x86_isa for flag in str(isa).split()) and isa:
                isa_list.append(isa)

    # 返回支持的向量指令集列表
    return isa_list


# 定义函数 pick_vec_isa，选择最优的向量指令集
def pick_vec_isa() -> VecISA:
    # 如果是 fbcode 环境，返回 AVX2 向量指令集
    if config.is_fbcode():
        return VecAVX2()

    # 获取当前系统支持的向量指令集列表
    _valid_vec_isa_list: List[VecISA] = valid_vec_isa_list()

    # 如果列表为空，返回无效的向量指令集对象
    if not _valid_vec_isa_list:
        return invalid_vec_isa

    # 如果 simdlen 为 None，自动选择第一个向量指令集
    if config.cpp.simdlen is None:
        assert _valid_vec_isa_list
        return _valid_vec_isa_list[0]

    # 遍历有效的向量指令集列表，选择与 simdlen 匹配的指令集
    for isa in _valid_vec_isa_list:
        if config.cpp.simdlen == isa.bit_width():
            return isa

    # 如果没有匹配的指令集，返回无效的向量指令集对象
    return invalid_vec_isa
```
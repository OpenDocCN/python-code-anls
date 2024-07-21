# `.\pytorch\torch\_utils_internal.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和库
import functools  # 提供了函数式编程支持的工具
import logging  # 提供了日志记录功能
import os  # 提供了与操作系统交互的功能
import sys  # 提供了与Python解释器交互的功能
import tempfile  # 提供了临时文件和目录的创建功能
from typing import Any, Dict, Optional  # 引入类型提示相关功能

import torch  # 引入PyTorch库
from torch._strobelight.compile_time_profiler import StrobelightCompileTimeProfiler  # 引入Strobelight编译时性能分析器


log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象

if os.environ.get("TORCH_COMPILE_STROBELIGHT", False):
    import shutil  # 如果环境变量TORCH_COMPILE_STROBELIGHT为真，导入shutil模块

    if not shutil.which("strobeclient"):
        # 如果在FB机器上没有找到strobeclient工具，则记录警告信息
        log.info(
            "TORCH_COMPILE_STROBELIGHT is true, but seems like you are not on a FB machine."
        )
    else:
        # 否则，启用Strobelight分析器，并记录日志信息
        log.info("Strobelight profiler is enabled via environment variable")
        StrobelightCompileTimeProfiler.enable()

# this arbitrary-looking assortment of functionality is provided here
# to have a central place for overrideable behavior. The motivating
# use is the FB build environment, where this source file is replaced
# by an equivalent.

if torch._running_with_deploy():
    # 如果运行在Torch部署环境中，设置torch_parent为空字符串
    # 因为在冻结的Torch中，__file__属性没有意义
    torch_parent = ""
else:
    # 否则，根据当前文件的目录名确定torch_parent路径
    if os.path.basename(os.path.dirname(__file__)) == "shared":
        torch_parent = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    else:
        torch_parent = os.path.dirname(os.path.dirname(__file__))


def get_file_path(*path_components: str) -> str:
    # 拼接路径组件，返回完整的文件路径
    return os.path.join(torch_parent, *path_components)


def get_file_path_2(*path_components: str) -> str:
    # 拼接路径组件，返回完整的文件路径
    return os.path.join(*path_components)


def get_writable_path(path: str) -> str:
    # 检查路径是否可写，如果可写直接返回该路径，否则创建临时目录返回
    if os.access(path, os.W_OK):
        return path
    return tempfile.mkdtemp(suffix=os.path.basename(path))


def prepare_multiprocessing_environment(path: str) -> None:
    # 准备多进程环境的空实现函数
    pass


def resolve_library_path(path: str) -> str:
    # 解析库路径，返回其真实路径
    return os.path.realpath(path)


def throw_abstract_impl_not_imported_error(opname, module, context):
    # 抛出未实现的操作错误，根据是否导入了模块提供不同的错误信息
    if module in sys.modules:
        raise NotImplementedError(
            f"{opname}: We could not find the fake impl for this operator. "
        )
    else:
        raise NotImplementedError(
            f"{opname}: We could not find the fake impl for this operator. "
            f"The operator specified that you may need to import the '{module}' "
            f"Python module to load the fake impl. {context}"
        )


# NB!  This treats "skip" kwarg specially!!
def compile_time_strobelight_meta(phase_name):
    # 返回装饰器函数，用于在编译时对指定函数进行Strobelight性能分析
    def compile_time_strobelight_meta_inner(function):
        @functools.wraps(function)
        def wrapper_function(*args, **kwargs):
            # 如果关键字参数中包含"skip"，则将其增加1
            if "skip" in kwargs:
                kwargs["skip"] = kwargs["skip"] + 1
            # 调用Strobelight分析器对函数进行性能分析
            return StrobelightCompileTimeProfiler.profile_compile_time(
                function, phase_name, *args, **kwargs
            )

        return wrapper_function

    return compile_time_strobelight_meta_inner


# Meta only, see
# 用于记录事件到 Scuba，通过 signposts API 发送日志事件。可以在 https://fburl.com/scuba/workflow_signpost/zh9wmpqs 上查看 API 示例。
# 参数 category 和 name 用于指定事件的子系统和具体描述，parameters 是一个字典，用于存储其他参数信息。
def signpost_event(category: str, name: str, parameters: Dict[str, Any]):
    log.info("%s %s: %r", category, name, parameters)


# 记录编译事件的日志信息，metrics 是要记录的具体信息。
def log_compilation_event(metrics):
    log.info("%s", metrics)


# 上传图形数据的函数，暂时未实现。
def upload_graph(graph):
    pass


# 从 justknobs 获取 PyTorch 分布式环境变量的设置，暂时未实现。
def set_pytorch_distributed_envs_from_justknobs():
    pass


# 记录导出使用情况的日志，kwargs 是可能的参数，但未实际使用。
def log_export_usage(**kwargs):
    pass


# 记录 TorchScript 使用情况的日志，api 是使用的 TorchScript API 名称。
def log_torchscript_usage(api: str):
    _ = api
    return


# 检查当前 Torch 是否可以导出的函数，始终返回 False。
def check_if_torch_exportable():
    return False


# 记录 Torch JIT 追踪可导出性的日志，api, type_of_export, export_outcome 和 result 是相关参数。
def log_torch_jit_trace_exportability(
    api: str,
    type_of_export: str,
    export_outcome: str,
    result: str,
):
    _, _, _, _ = api, type_of_export, export_outcome, result
    return


# 检查导出 API 的发布情况，始终返回 False。
def export_api_rollout_check() -> bool:
    return False


# 查询 justknobs 中指定名称的开关状态，用于控制功能是否开启。
def justknobs_check(name: str) -> bool:
    """
    This function can be used to killswitch functionality in FB prod,
    where you can toggle this value to False in JK without having to
    do a code push.  In OSS, we always have everything turned on all
    the time, because downstream users can simply choose to not update
    PyTorch.  (If more fine-grained enable/disable is needed, we could
    potentially have a map we lookup name in to toggle behavior.  But
    the point is that it's all tied to source code in OSS, since there's
    no live server to query.)

    This is the bare minimum functionality I needed to do some killswitches.
    We have a more detailed plan at
    https://docs.google.com/document/d/1Ukerh9_42SeGh89J-tGtecpHBPwGlkQ043pddkKb3PU/edit
    In particular, in some circumstances it may be necessary to read in
    a knob once at process start, and then use it consistently for the
    rest of the process.  Future functionality will codify these patterns
    into a better high level API.

    WARNING: Do NOT call this function at module import time, JK is not
    fork safe and you will break anyone who forks the process and then
    hits JK again.
    """
    return True


# 查询 justknobs 中指定名称的整数值，用于获取配置信息，但这里始终返回 0。
def justknobs_getval_int(name: str) -> int:
    """
    Read warning on justknobs_check
    """
    return 0


# 使用 functools.lru_cache 进行缓存的函数，用于获取最大的时钟速率。
# 如果不是 Torch 的版本是 hip 的话，则从 triton.testing 中导入 nvsmi 模块并返回最大时钟速率。
@functools.lru_cache(None)
def max_clock_rate():
    if not torch.version.hip:
        from triton.testing import nvsmi

        return nvsmi(["clocks.max.sm"])[0]
    # 如果不是特定的 ROCm GPU 架构，则手动设置最大时钟速度
    # 这是为了在 triton.testing 中或通过 pyamdsmi 启用等效的 nvmsi 功能，需要在 test_snode_runtime 单元测试中使用。
    # 获取当前 CUDA 设备的属性，并提取其架构名称（可能包含多个部分，用冒号分隔，只取第一个部分）
    gcn_arch = str(torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0])
    # 根据不同的 GPU 架构名称返回对应的最大时钟速度
    if "gfx94" in gcn_arch:
        return 1700
    elif "gfx90a" in gcn_arch:
        return 1700
    elif "gfx908" in gcn_arch:
        return 1502
    elif "gfx11" in gcn_arch:
        return 1700
    elif "gfx103" in gcn_arch:
        return 1967
    elif "gfx101" in gcn_arch:
        return 1144
    else:
        # 如果 GPU 架构不在已知列表中，则返回默认的最大时钟速度
        return 1100
# 测试主地址，指定为本地回环地址
TEST_MASTER_ADDR = "127.0.0.1"
# 测试主端口号，指定为29500
TEST_MASTER_PORT = 29500

# USE_GLOBAL_DEPS 控制是否在 __init__.py 中尝试加载 libtorch_global_deps，
# 参见注释 [Global dependencies]
USE_GLOBAL_DEPS = True

# USE_RTLD_GLOBAL_WITH_LIBTORCH 控制是否在 __init__.py 中尝试使用 RTLD_GLOBAL
# 加载 _C.so，详见注释 [Global dependencies]
USE_RTLD_GLOBAL_WITH_LIBTORCH = False

# 如果一个操作在 C++ 中定义，并且通过 torch.library.register_fake 扩展到 Python，
# 返回值指示是否需要从 C++ 中进行 m.set_python_module("mylib.ops") 调用，
# 将 C++ 操作与 Python 模块关联起来。
REQUIRES_SET_PYTHON_MODULE = False


def maybe_upload_prof_stats_to_manifold(profile_path: str) -> Optional[str]:
    # 打印信息，上传性能统计数据到 manifold（仅限 Facebook，否则不执行操作）
    print("Uploading profile stats (fb-only otherwise no-op)")
    # 返回空值，表示未执行上传操作
    return None
```
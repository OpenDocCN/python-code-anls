# `.\pytorch\torch\utils\benchmark\utils\cpp_jit.py`

```py
# 导入必要的库和模块
"""JIT C++ strings into executables."""
import atexit           # 用于注册退出时的清理操作
import os               # 提供与操作系统交互的功能
import re               # 提供正则表达式操作支持
import shutil           # 提供文件和目录的高级操作函数
import textwrap         # 提供文本的格式化和填充
import threading        # 提供多线程支持
from typing import Any, List, Optional   # 引入类型提示相关的模块

import torch           # 引入 PyTorch 库
from torch.utils.benchmark.utils._stubs import CallgrindModuleType, TimeitModuleType  # 引入测试相关的模块
from torch.utils.benchmark.utils.common import _make_temp_dir   # 引入临时目录创建函数
from torch.utils import cpp_extension   # 引入 C++ 扩展相关的模块


LOCK = threading.Lock()   # 创建一个线程锁对象
SOURCE_ROOT = os.path.split(os.path.abspath(__file__))[0]   # 获取当前文件的绝对路径并提取其目录部分

# 在导入时计算 uuid 一次，以确保不同进程有不同的构建根目录，但线程共享相同的构建根目录。
# `cpp_extension` 使用构建根目录作为缓存键的一部分，因此如果每次调用都有不同的 uuid
# （例如每次调用 _compile_template 时不同的构建根目录），将导致缓存命中率为 0% 并且会出现不必要的重新编译。
# 考虑以下情况：
# ```
# setup = "auto x = torch::ones({1024, 1024});"
# stmt = "torch::mm(x, x);"
# for num_threads in [1, 2, 4, 8]:
#   print(Timer(stmt, setup, num_threads=num_threads, language="c++").blocked_autorange())
# ```py`
# `setup` 和 `stmt` 不会改变，因此我们可以重用第一次循环的可执行文件。
_BUILD_ROOT: Optional[str] = None

def _get_build_root() -> str:
    global _BUILD_ROOT
    if _BUILD_ROOT is None:
        _BUILD_ROOT = _make_temp_dir(prefix="benchmark_utils_jit_build")   # 创建一个临时目录作为构建根目录
        atexit.register(shutil.rmtree, _BUILD_ROOT)   # 注册退出时清理构建根目录的操作
    return _BUILD_ROOT   # 返回构建根目录的路径


# BACK_TESTING_NOTE:
#   这段代码可能用于两种工作流程。一种是显而易见的情况，即有人简单地构建或安装 PyTorch 并使用 Timer。
#   另一种情况是，从当前 PyTorch 检出的整个 `torch/utils/benchmark` 文件夹复制到较旧版本的 PyTorch 源代码中。
#   这就是我们所说的“回测”。其理念是，我们可能希望使用当前的工具来研究早期版本的 PyTorch 的某些方面（例如回归问题）。
#
#   问题在于，Timer 依赖于 PyTorch 核心的几个方面，特别是在 `torch._C` 中绑定函数的 Valgrind 符号和
#   `torch.__config__._cxx_flags()` 方法。如果我们简单地复制代码，那么在较早版本的 PyTorch 中这些感兴趣的符号将不存在。
#   为了解决这个问题，我们必须添加回测补丁。这些补丁在正常使用时不会被激活，但将允许 Timer 在“正确”的 PyTorch 版本之外工作，
#   通过模拟稍后添加的功能来实现。
#
#   这些补丁是临时的，随着 Timer 与 PyTorch 的集成程度提高，维护和代码复杂性成本会增加。
#   一旦不再需要回测（也就是说，我们已经进行了足够的历史分析，并且这些补丁不再值得维护和代码复杂性成本），回测路径将被删除。
CXX_FLAGS: Optional[List[str]]
if hasattr(torch.__config__, "_cxx_flags"):
    # 尝试获取 Torch 库的 C++ 编译标志
    try:
        # 调用 Torch 库的私有方法 _cxx_flags() 获取编译标志字符串，并进行处理
        CXX_FLAGS = torch.__config__._cxx_flags().strip().split()
        
        # 检查获取的编译标志列表是否不为空，并且 "-g" 标志不在其中时，添加 "-g" 标志
        if CXX_FLAGS is not None and "-g" not in CXX_FLAGS:
            CXX_FLAGS.append("-g")
        
        # 移除编译标志列表中以 "-W" 开头的标志，以放宽编译器版本的约束，允许构建基准测试
        if CXX_FLAGS is not None:
            CXX_FLAGS = list(filter(lambda x: not x.startswith("-W"), CXX_FLAGS))

    # 如果运行时出现 RuntimeError 异常
    except RuntimeError:
        # 我们处于 FBCode 环境中
        CXX_FLAGS = None
else:
    # FIXME: Remove when back testing is no longer required.
    # 在后测试不再需要时移除此处代码。
    CXX_FLAGS = ["-O2", "-fPIC", "-g"]

EXTRA_INCLUDE_PATHS: List[str] = [os.path.join(SOURCE_ROOT, "valgrind_wrapper")]
# 定义额外的包含路径列表，包括源根目录下的valgrind_wrapper目录

CONDA_PREFIX = os.getenv("CONDA_PREFIX")
if CONDA_PREFIX is not None:
    # 如果存在conda环境，添加conda环境的include路径到额外的包含路径列表中
    EXTRA_INCLUDE_PATHS.append(os.path.join(CONDA_PREFIX, "include"))


COMPAT_CALLGRIND_BINDINGS: Optional[CallgrindModuleType] = None
def get_compat_bindings() -> CallgrindModuleType:
    with LOCK:
        global COMPAT_CALLGRIND_BINDINGS
        if COMPAT_CALLGRIND_BINDINGS is None:
            # 如果兼容的Callgrind绑定对象为空，使用cpp_extension.load加载并赋值给COMPAT_CALLGRIND_BINDINGS
            COMPAT_CALLGRIND_BINDINGS = cpp_extension.load(
                name="callgrind_bindings",
                sources=[os.path.join(
                    SOURCE_ROOT,
                    "valgrind_wrapper",
                    "compat_bindings.cpp"
                )],
                extra_cflags=CXX_FLAGS,
                extra_include_paths=EXTRA_INCLUDE_PATHS,
            )
    return COMPAT_CALLGRIND_BINDINGS


def _compile_template(
    *,
    stmt: str,
    setup: str,
    global_setup: str,
    src: str,
    is_standalone: bool
) -> Any:
    for before, after, indentation in (
        ("// GLOBAL_SETUP_TEMPLATE_LOCATION", global_setup, 0),
        ("// SETUP_TEMPLATE_LOCATION", setup, 4),
        ("// STMT_TEMPLATE_LOCATION", stmt, 8)
    ):
        # C++ doesn't care about indentation so this code isn't load
        # bearing the way it is with Python, but this makes the source
        # look nicer if a human has to look at it.
        # 替换源码中的模板位置标记，使得代码在人类查看时更加美观

        src = re.sub(
            before,
            textwrap.indent(after, " " * indentation)[indentation:],
            src
        )

    # We want to isolate different Timers. However `cpp_extension` will
    # cache builds which will significantly reduce the cost of repeated
    # invocations.
    # 我们希望隔离不同的计时器。然而，`cpp_extension`会缓存构建结果，显著降低重复调用的成本。
    with LOCK:
        name = f"timer_cpp_{abs(hash(src))}"
        build_dir = os.path.join(_get_build_root(), name)
        os.makedirs(build_dir, exist_ok=True)

        src_path = os.path.join(build_dir, "timer_src.cpp")
        with open(src_path, "w") as f:
            f.write(src)

    # `cpp_extension` has its own locking scheme, so we don't need our lock.
    # `cpp_extension`有自己的锁定机制，因此我们不需要使用我们的锁。
    return cpp_extension.load(
        name=name,
        sources=[src_path],
        build_directory=build_dir,
        extra_cflags=CXX_FLAGS,
        extra_include_paths=EXTRA_INCLUDE_PATHS,
        is_python_module=not is_standalone,
        is_standalone=is_standalone,
    )


def compile_timeit_template(*, stmt: str, setup: str, global_setup: str) -> TimeitModuleType:
    # 读取时间测量模板文件路径
    template_path: str = os.path.join(SOURCE_ROOT, "timeit_template.cpp")
    with open(template_path) as f:
        src: str = f.read()

    # 编译时间测量模板，并返回模块对象
    module = _compile_template(stmt=stmt, setup=setup, global_setup=global_setup, src=src, is_standalone=False)
    assert isinstance(module, TimeitModuleType)
    return module
# 编译 Callgrind 模板函数，生成并返回编译后的目标代码字符串
def compile_callgrind_template(*, stmt: str, setup: str, global_setup: str) -> str:
    # 模板文件路径，使用 os.path.join 构建完整路径
    template_path: str = os.path.join(SOURCE_ROOT, "valgrind_wrapper", "timer_callgrind_template.cpp")
    
    # 打开模板文件并读取其内容为字符串 src
    with open(template_path) as f:
        src: str = f.read()

    # 调用 _compile_template 函数编译模板
    # 参数包括 stmt（语句）、setup（设置）、global_setup（全局设置）、src（模板源码）、is_standalone（独立模式）
    target = _compile_template(stmt=stmt, setup=setup, global_setup=global_setup, src=src, is_standalone=True)
    
    # 断言确保 target 是一个字符串类型
    assert isinstance(target, str)
    
    # 返回编译后的目标代码字符串
    return target
```
# `.\pytorch\tools\testing\discover_tests.py`

```
from __future__ import annotations
# 导入用于支持类型注解的模块，这样可以在函数签名中使用类型提示

import glob
# 导入用于文件模式匹配的 glob 模块
import os
# 导入操作系统相关功能的 os 模块
import sys
# 导入系统相关信息的 sys 模块
from pathlib import Path
# 导入路径处理类 Path，用于处理文件路径

CPP_TEST_PREFIX = "cpp"
# 定义用于标识 C++ 测试文件前缀的常量字符串
CPP_TEST_PATH = "build/bin"
# 定义 C++ 测试文件目录的默认路径字符串常量
CPP_TESTS_DIR = os.path.abspath(os.getenv("CPP_TESTS_DIR", default=CPP_TEST_PATH))
# 获取环境变量 CPP_TESTS_DIR 的值作为 C++ 测试文件目录的路径，并转换为绝对路径
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# 获取当前脚本文件所在的项目根目录的路径

def parse_test_module(test: str) -> str:
    # 定义函数 parse_test_module，用于从测试文件名中提取模块名
    return test.split(".")[0]
    # 返回测试文件名中第一个点号前的部分作为模块名

def discover_tests(
    base_dir: Path = REPO_ROOT / "test",
    cpp_tests_dir: str | Path | None = None,
    blocklisted_patterns: list[str] | None = None,
    blocklisted_tests: list[str] | None = None,
    extra_tests: list[str] | None = None,
) -> list[str]:
    """
    Searches for all python files starting with test_ excluding one specified by patterns.
    If cpp_tests_dir is provided, also scan for all C++ tests under that directory. They
    are usually found in build/bin
    """
    # 定义 discover_tests 函数，用于搜索测试文件，并返回测试文件名列表

    def skip_test_p(name: str) -> bool:
        # 定义内部函数 skip_test_p，用于判断是否跳过特定的测试文件
        rc = False
        if blocklisted_patterns is not None:
            rc |= any(name.startswith(pattern) for pattern in blocklisted_patterns)
            # 如果存在需要屏蔽的文件名模式，则判断当前文件名是否以这些模式开头
        if blocklisted_tests is not None:
            rc |= name in blocklisted_tests
            # 如果存在需要屏蔽的具体测试文件名，则判断当前文件名是否在屏蔽列表中
        return rc
        # 返回是否需要跳过当前测试文件的布尔值

    # This supports symlinks, so we can link domain library tests to PyTorch test directory
    all_py_files = [
        Path(p) for p in glob.glob(f"{base_dir}/**/test_*.py", recursive=True)
    ]
    # 使用 glob 模块找到指定目录下所有子目录中以 test_ 开头的 Python 测试文件

    cpp_tests_dir = (
        f"{base_dir.parent}/{CPP_TEST_PATH}" if cpp_tests_dir is None else cpp_tests_dir
    )
    # 如果未指定 cpp_tests_dir，则使用默认路径拼接，指向 C++ 测试文件的目录

    # CPP test files are located under pytorch/build/bin. Unlike Python test, C++ tests
    # are just binaries and could have any name, i.e. basic or atest
    all_cpp_files = [
        Path(p) for p in glob.glob(f"{cpp_tests_dir}/**/*", recursive=True)
    ]
    # 使用 glob 模块找到指定的 C++ 测试文件目录中所有文件，包括子目录中的文件

    rc = [str(fname.relative_to(base_dir))[:-3] for fname in all_py_files]
    # 构建 Python 测试文件名的相对路径列表，并去掉文件扩展名 ".py"

    # Add the cpp prefix for C++ tests so that we can tell them apart
    rc.extend(
        [
            parse_test_module(f"{CPP_TEST_PREFIX}/{fname.relative_to(cpp_tests_dir)}")
            for fname in all_cpp_files
        ]
    )
    # 将 C++ 测试文件名转换为模块名格式，并添加到结果列表中，以便区分 Python 和 C++ 测试文件

    # Invert slashes on Windows
    if sys.platform == "win32":
        rc = [name.replace("\\", "/") for name in rc]
        # 如果运行环境是 Windows，则将路径中的反斜杠替换为正斜杠

    rc = [test for test in rc if not skip_test_p(test)]
    # 过滤掉需要跳过的测试文件名

    if extra_tests is not None:
        rc += extra_tests
        # 如果存在额外的测试文件名列表，则将其添加到结果列表中

    return sorted(rc)
    # 返回按字母顺序排序后的测试文件名列表

TESTS = discover_tests(
    cpp_tests_dir=CPP_TESTS_DIR,
    blocklisted_patterns=[
        "ao",
        "bottleneck_test",
        "custom_backend",
        "custom_operator",
        "fx",  # executed by test_fx.py
        "jit",  # executed by test_jit.py
        "mobile",
        "onnx_caffe2",
        "package",  # executed by test_package.py
        "quantization",  # executed by test_quantization.py
        "autograd",  # executed by test_autograd.py
    ],
)
# 调用 discover_tests 函数，获取所有符合条件的测试文件名列表，并赋值给 TESTS 变量
    # 定义被列入黑名单的测试集合，这些测试将被排除在外
    blocklisted_tests=[
        "test_bundled_images",  # 测试捆绑图像
        "test_cpp_extensions_aot",  # 测试 C++ 扩展预先编译
        "test_determination",  # 测试决策
        "test_jit_fuser",  # 测试 JIT 融合器
        "test_jit_simple",  # 测试简单 JIT
        "test_jit_string",  # 测试 JIT 字符串
        "test_kernel_launch_checks",  # 测试内核启动检查
        "test_nnapi",  # 测试 NNAPI
        "test_static_runtime",  # 测试静态运行时
        "test_throughput_benchmark",  # 测试吞吐量基准
        "distributed/bin/test_script",  # 分布式测试脚本
        "distributed/elastic/multiprocessing/bin/test_script",  # 弹性多进程测试脚本
        "distributed/launcher/bin/test_script",  # 分布式启动器测试脚本
        "distributed/launcher/bin/test_script_init_method",  # 分布式启动器测试脚本初始化方法
        "distributed/launcher/bin/test_script_is_torchelastic_launched",  # 分布式启动器测试脚本是否为 TorchElastic 启动
        "distributed/launcher/bin/test_script_local_rank",  # 分布式启动器测试脚本本地排名
        "distributed/test_c10d_spawn",  # 分布式测试 c10d_spawn
        "distributions/test_transforms",  # 分布式转换测试
        "distributions/test_utils",  # 分布式实用工具测试
        "test/inductor/test_aot_inductor_utils",  # 引导器 AOT 引导工具测试
        "onnx/test_pytorch_onnx_onnxruntime_cuda",  # ONNX PyTorch ONNX Runtime CUDA 测试
        "onnx/test_models",  # ONNX 模型测试
        # These are not C++ tests
        f"{CPP_TEST_PREFIX}/CMakeFiles",  # 不是 C++ 测试的 CMakeFiles
        f"{CPP_TEST_PREFIX}/CTestTestfile.cmake",  # 不是 C++ 测试的 CTestTestfile.cmake
        f"{CPP_TEST_PREFIX}/Makefile",  # 不是 C++ 测试的 Makefile
        f"{CPP_TEST_PREFIX}/cmake_install.cmake",  # 不是 C++ 测试的 cmake_install.cmake
        f"{CPP_TEST_PREFIX}/c10_intrusive_ptr_benchmark",  # 不是 C++ 测试的 c10_intrusive_ptr_benchmark
        f"{CPP_TEST_PREFIX}/example_allreduce",  # 不是 C++ 测试的 example_allreduce
        f"{CPP_TEST_PREFIX}/parallel_benchmark",  # 不是 C++ 测试的 parallel_benchmark
        f"{CPP_TEST_PREFIX}/protoc",  # 不是 C++ 测试的 protoc
        f"{CPP_TEST_PREFIX}/protoc-3.13.0.0",  # 不是 C++ 测试的 protoc-3.13.0.0
        f"{CPP_TEST_PREFIX}/torch_shm_manager",  # 不是 C++ 测试的 torch_shm_manager
        f"{CPP_TEST_PREFIX}/tutorial_tensorexpr",  # 不是 C++ 测试的 tutorial_tensorexpr
    ],
    # 定义额外的测试集合，这些测试将被附加到主要测试集之外
    extra_tests=[
        "test_cpp_extensions_aot_ninja",  # 测试 C++ 扩展预先编译使用 Ninja
        "test_cpp_extensions_aot_no_ninja",  # 测试 C++ 扩展预先编译不使用 Ninja
        "distributed/elastic/timer/api_test",  # 弹性定时器 API 测试
        "distributed/elastic/timer/local_timer_example",  # 弹性定时器本地定时器示例
        "distributed/elastic/timer/local_timer_test",  # 弹性定时器本地定时器测试
        "distributed/elastic/events/lib_test",  # 弹性事件库测试
        "distributed/elastic/metrics/api_test",  # 弹性度量 API 测试
        "distributed/elastic/utils/logging_test",  # 弹性实用工具日志测试
        "distributed/elastic/utils/util_test",  # 弹性实用工具实用工具测试
        "distributed/elastic/utils/distributed_test",  # 弹性实用工具分布式测试
        "distributed/elastic/multiprocessing/api_test",  # 弹性多进程 API 测试
        "doctests",  # 文档测试
    ],
# 如果当前脚本被直接运行（而不是作为模块导入），则执行以下代码
if __name__ == "__main__":
    # 打印变量 TESTS 的值
    print(TESTS)
```
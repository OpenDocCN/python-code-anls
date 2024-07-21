# `.\pytorch\tools\testing\modulefinder_determinator.py`

```py
# 从未来版本导入注释
from __future__ import annotations

# 导入模块用于查找模块依赖
import modulefinder
# 导入操作系统相关的功能
import os
# 导入系统相关的功能
import sys
# 导入警告模块
import warnings
# 导入处理路径相关的模块
from pathlib import Path
# 导入类型提示相关的功能
from typing import Any

# 定义项目根目录为当前文件的上三级目录
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# 定义一个列表，包含了一些执行时间较长的测试模块，用于判断是否值得先计算相关文件变更
TARGET_DET_LIST = [
    # 以下模块是经过手动选择的，每次运行时，可以基于这个列表和先前的测试统计生成另一个列表
    "test_binary_ufuncs",
    "test_cpp_extensions_aot_ninja",
    "test_cpp_extensions_aot_no_ninja",
    "test_cpp_extensions_jit",
    "test_cpp_extensions_open_device_registration",
    "test_cpp_extensions_stream_and_event",
    "test_cpp_extensions_mtia_backend",
    "test_cuda",
    "test_cuda_primary_ctx",
    "test_dataloader",
    "test_determination",
    "test_futures",
    "test_jit",
    "test_jit_legacy",
    "test_jit_profiling",
    "test_linalg",
    "test_multiprocessing",
    "test_nn",
    "test_numpy_interop",
    "test_optim",
    "test_overrides",
    "test_pruning_op",
    "test_quantization",
    "test_reductions",
    "test_serialization",
    "test_shape_ops",
    "test_sort_and_select",
    "test_tensorboard",
    "test_testing",
    "test_torch",
    "test_utils",
    "test_view_ops",
]

# 缓存依赖模块的字典，键为字符串，值为字符串集合
_DEP_MODULES_CACHE: dict[str, set[str]] = {}


# 判断是否应该运行指定测试
def should_run_test(
    target_det_list: list[str], test: str, touched_files: list[str], options: Any
) -> bool:
    # 解析测试模块名
    test = parse_test_module(test)
    # 如果测试模块不在目标确定列表中，直接运行（不需要先计算）
    if test not in target_det_list:
        # 如果设置了详细输出选项，打印到标准错误流
        if options.verbose:
            print_to_stderr(f"Running {test} without determination")
        return True
    
    # HACK: "no_ninja" 不是一个真实的模块，处理这种情况
    if test.endswith("_no_ninja"):
        test = test[: (-1 * len("_no_ninja"))]
    if test.endswith("_ninja"):
        test = test[: (-1 * len("_ninja"))]

    # 获取测试模块的依赖模块
    dep_modules = get_dep_modules(test)
    # 遍历所有被修改过的文件
    for touched_file in touched_files:
        # 获取文件类型（影响测试的程度）
        file_type = test_impact_of_file(touched_file)
        # 如果文件类型为 "NONE"，跳过当前文件的处理
        if file_type == "NONE":
            continue
        # 如果文件类型为 "CI"
        elif file_type == "CI":
            # 如果 CI 配置文件有任何更改，强制运行所有测试
            log_test_reason(file_type, touched_file, test, options)
            return True
        # 如果文件类型为 "UNKNOWN"
        elif file_type == "UNKNOWN":
            # 假定未分类的源文件可能会影响每一个测试
            log_test_reason(file_type, touched_file, test, options)
            return True
        # 如果文件类型为 "TORCH", "CAFFE2", "TEST" 中的一种
        elif file_type in ["TORCH", "CAFFE2", "TEST"]:
            # 获取文件名（不含扩展名）并根据路径分割为部分
            parts = os.path.splitext(touched_file)[0].split(os.sep)
            # 将分割后的部分用点连接形成模块名
            touched_module = ".".join(parts)
            # 对于位于 "test/" 路径下的模块，去除开头的 "test."
            if touched_module.startswith("test."):
                touched_module = touched_module.split("test.")[1]
            # 如果修改的模块在依赖模块中或者与测试路径匹配，则记录测试原因并返回真值
            if touched_module in dep_modules or touched_module == test.replace(
                "/", "."
            ):
                log_test_reason(file_type, touched_file, test, options)
                return True

    # 如果没有确定需要运行测试的文件，则不运行测试
    if options.verbose:
        # 如果选项中设置了详细输出，打印跳过测试的消息
        print_to_stderr(f"Determination is skipping {test}")

    # 返回假值，表示不运行测试
    return False
# 确定文件对测试运行的影响类别

def test_impact_of_file(filename: str) -> str:
    """Determine what class of impact this file has on test runs.
    
    可能的返回值:
        TORCH - Torch Python 代码
        CAFFE2 - Caffe2 Python 代码
        TEST - Torch 测试代码
        UNKNOWN - 可能影响所有测试
        NONE - 已知对测试结果没有影响
        CI - CI 配置文件
    """
    # 使用操作系统特定的分隔符拆分文件路径
    parts = filename.split(os.sep)
    
    # 检查文件路径的第一个部分是否属于 CI 环境
    if parts[0] in [".jenkins", ".circleci", ".ci"]:
        return "CI"
    
    # 检查文件路径的第一个部分是否属于文档、脚本或一些特定的文件，已知对测试结果没有影响
    if parts[0] in ["docs", "scripts", "CODEOWNERS", "README.md"]:
        return "NONE"
    elif parts[0] == "torch":
        # 如果文件路径的第一个部分是 "torch"，并且文件名以 ".py" 或 ".pyi" 结尾，则属于 Torch Python 代码
        if parts[-1].endswith(".py") or parts[-1].endswith(".pyi"):
            return "TORCH"
    elif parts[0] == "caffe2":
        # 如果文件路径的第一个部分是 "caffe2"，并且文件名以 ".py" 或 ".pyi" 结尾，则属于 Caffe2 Python 代码
        if parts[-1].endswith(".py") or parts[-1].endswith(".pyi"):
            return "CAFFE2"
    elif parts[0] == "test":
        # 如果文件路径的第一个部分是 "test"，并且文件名以 ".py" 或 ".pyi" 结尾，则属于 Torch 测试代码
        if parts[-1].endswith(".py") or parts[-1].endswith(".pyi"):
            return "TEST"
    
    # 如果无法确定文件对测试运行的具体影响类别，则返回 UNKNOWN
    return "UNKNOWN"


# 记录测试文件对测试运行影响的原因
def log_test_reason(file_type: str, filename: str, test: str, options: Any) -> None:
    if options.verbose:
        # 如果选项中设置了 verbose 参数，将信息打印到标准错误输出中
        print_to_stderr(
            f"Determination found {file_type} file {filename} -- running {test}"
        )


# 获取测试模块的依赖模块集合
def get_dep_modules(test: str) -> set[str]:
    # 如果结果已经缓存，则直接返回缓存的结果
    if test in _DEP_MODULES_CACHE:
        return _DEP_MODULES_CACHE[test]
    
    # 构造测试脚本的路径
    test_location = REPO_ROOT / "test" / f"{test}.py"
    
    # 创建一个 ModuleFinder 对象用于查找依赖模块
    finder = modulefinder.ModuleFinder(
        # 排除所有第三方模块，以加快计算速度
        excludes=[
            "scipy",
            "numpy",
            "numba",
            "multiprocessing",
            "sklearn",
            "setuptools",
            "hypothesis",
            "llvmlite",
            "joblib",
            "email",
            "importlib",
            "unittest",
            "urllib",
            "json",
            "collections",
            # 由于存在特定问题，排除以下模块
            "mpl_toolkits",
            "google",
            "onnx",
            "mypy",  # 会触发递归错误
        ],
    )
    
    # 忽略运行过程中的警告信息
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 运行测试脚本以查找其依赖的模块
        finder.run_script(str(test_location))
    
    # 获取找到的所有依赖模块的集合
    dep_modules = set(finder.modules.keys())
    # 将结果缓存起来
    _DEP_MODULES_CACHE[test] = dep_modules
    return dep_modules


# 解析测试模块，获取其主要模块名称
def parse_test_module(test: str) -> str:
    return test.split(".")[0]


# 打印信息到标准错误输出
def print_to_stderr(message: str) -> None:
    print(message, file=sys.stderr)
```
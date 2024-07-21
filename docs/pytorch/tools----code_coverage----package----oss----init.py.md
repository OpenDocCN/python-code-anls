# `.\pytorch\tools\code_coverage\package\oss\init.py`

```
# 从未来模块中导入注解，以支持类型注解的声明
from __future__ import annotations

# 导入命令行参数解析模块
import argparse
# 导入操作系统相关功能模块
import os
# 导入类型提示相关模块
from typing import cast

# 导入设置相关的自定义模块
from ..util.setting import (
    CompilerType,
    JSON_FOLDER_BASE_DIR,
    LOG_DIR,
    Option,
    Test,
    TestList,
    TestType,
)
# 导入通用的工具函数
from ..util.utils import (
    clean_up,
    create_folder,
    print_log,
    raise_no_test_found_exception,
    remove_file,
    remove_folder,
)
# 导入初始化相关的自定义函数
from ..util.utils_init import add_arguments_utils, create_folders, get_options
# 导入本地模块的特定函数
from .utils import (
    clean_up_gcda,
    detect_compiler_type,
    get_llvm_tool_path,
    get_oss_binary_folder,
    get_pytorch_folder,
)

# 定义一个集合，包含被屏蔽的 Python 测试文件名
BLOCKED_PYTHON_TESTS = {
    "run_test.py",
    "test_dataloader.py",
    "test_multiprocessing.py",
    "test_multiprocessing_spawn.py",
    "test_utils.py",
}

# 初始化函数，返回一个包含选项、测试列表和感兴趣文件夹的元组
def initialization() -> tuple[Option, TestList, list[str]]:
    # 创建必要的文件夹（如果不存在）
    create_folders()
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 向解析器添加通用工具函数定义的命令行参数
    parser = add_arguments_utils(parser)
    # 向解析器添加特定于 OSS 的命令行参数
    parser = add_arguments_oss(parser)
    # 解析命令行参数并获取相关选项和参数值
    (options, args_interested_folder, args_run_only, arg_clean) = parse_arguments(
        parser
    )
    # 如果指定了清理选项，则执行清理操作（删除.gcda文件和其他清理工作）
    if arg_clean:
        clean_up_gcda()
        clean_up()
    # 获取测试列表
    test_list = get_test_list(args_run_only)
    # 获取感兴趣的文件夹列表（如果没有指定，则为空列表）
    interested_folders = empty_list_if_none(args_interested_folder)
    # 打印初始化信息
    print_init_info()
    # 删除上一次运行的日志文件
    remove_file(os.path.join(LOG_DIR, "log.txt"))
    # 返回选项、测试列表和感兴趣文件夹列表的元组
    return (options, test_list, interested_folders)

# 向 OSS 特定的命令行参数解析器添加参数
def add_arguments_oss(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--run-only",
        help="only run certain test(s), for example: atest test_nn.py.",
        nargs="*",
        default=None,
    )
    return parser

# 解析命令行参数，返回选项和相关参数的元组
def parse_arguments(
    parser: argparse.ArgumentParser,
) -> tuple[Option, list[str] | None, list[str] | None, bool | None]:
    # 解析命令行参数
    args = parser.parse_args()
    # 获取选项
    options = get_options(args)
    # 返回选项及感兴趣文件夹、仅运行测试和清理选项的值
    return (options, args.interest_only, args.run_only, args.clean)

# 根据指定的运行测试文件列表和测试类型获取测试列表
def get_test_list_by_type(run_only: list[str] | None, test_type: TestType) -> TestList:
    # 初始化测试列表为空列表
    test_list: TestList = []
    # 获取 OSS 二进制文件夹的路径
    binary_folder = get_oss_binary_folder(test_type)
    # 遍历二进制文件夹下的所有文件
    g = os.walk(binary_folder)
    for _, _, file_list in g:
        for file_name in file_list:
            # 如果运行的测试文件列表不为空且当前文件不在列表中，则跳过
            if run_only is not None and file_name not in run_only:
                continue
            # 创建测试对象，并添加到测试列表中
            test: Test = Test(
                name=file_name,
                target_pattern=file_name,
                test_set="",
                test_type=test_type,
            )
            test_list.append(test)
    # 返回测试列表
    return test_list

# 获取测试列表，根据指定的运行测试文件列表
def get_test_list(run_only: list[str] | None) -> TestList:
    # 初始化测试列表为空列表
    test_list: TestList = []
    # 添加 C++ 测试列表（待续）
    # 将运行类型为 CPP 的测试列表扩展到总测试列表中
    test_list.extend(get_test_list_by_type(run_only, TestType.CPP))
    # 添加 Python 测试列表到总测试列表中
    py_run_only = get_python_run_only(run_only)
    test_list.extend(get_test_list_by_type(py_run_only, TestType.PY))

    # 如果找不到任何要运行的测试
    if not test_list:
        # 抛出异常，指示未找到测试
        raise_no_test_found_exception(
            get_oss_binary_folder(TestType.CPP), get_oss_binary_folder(TestType.PY)
        )
    # 返回总的测试列表
    return test_list
# 如果输入参数 arg_interested_folder 是 None，则返回一个空列表
def empty_list_if_none(arg_interested_folder: list[str] | None) -> list[str]:
    if arg_interested_folder is None:
        return []
    # 如果指定了此参数，直接返回它本身
    return arg_interested_folder


# 初始化 GCC 导出设置，先删除指定的 JSON 文件夹，然后创建一个新的空文件夹
def gcc_export_init() -> None:
    remove_folder(JSON_FOLDER_BASE_DIR)  # 删除 JSON 文件夹
    create_folder(JSON_FOLDER_BASE_DIR)  # 创建 JSON 文件夹


# 根据参数 args_run_only 确定需要运行的 Python 测试脚本列表
def get_python_run_only(args_run_only: list[str] | None) -> list[str]:
    # 如果用户指定了运行仅限选项
    if args_run_only:
        return args_run_only

    # 如果未指定，则根据检测到的编译器类型确定默认设置，GCC 和 Clang 有不同的默认设置
    if detect_compiler_type() == CompilerType.GCC:
        return ["run_test.py"]
    else:
        # 对于 Clang，某些测试会生成太大的中间文件，无法由 LLVM 合并，因此需要跳过这些测试
        run_only: list[str] = []
        binary_folder = get_oss_binary_folder(TestType.PY)
        g = os.walk(binary_folder)
        for _, _, file_list in g:
            for file_name in file_list:
                if file_name in BLOCKED_PYTHON_TESTS or not file_name.endswith(".py"):
                    continue
                run_only.append(file_name)
            # 只在 test/ 的第一级文件夹中运行测试
            break
        return run_only


# 打印初始化信息，包括 PyTorch 文件夹路径、CPP 测试二进制文件夹路径、Python 测试脚本文件夹路径、编译器类型以及 LLVM 工具路径（仅适用于 Clang）
def print_init_info() -> None:
    print_log("pytorch folder: ", get_pytorch_folder())  # 打印 PyTorch 文件夹路径
    print_log("cpp test binaries folder: ", get_oss_binary_folder(TestType.CPP))  # 打印 CPP 测试二进制文件夹路径
    print_log("python test scripts folder: ", get_oss_binary_folder(TestType.PY))  # 打印 Python 测试脚本文件夹路径
    print_log("compiler type: ", cast(CompilerType, detect_compiler_type()).value)  # 打印编译器类型
    print_log(
        "llvm tool folder (only for clang, if you are using gcov please ignore it): ",
        get_llvm_tool_path(),  # 打印 LLVM 工具路径（仅适用于 Clang）
    )
```
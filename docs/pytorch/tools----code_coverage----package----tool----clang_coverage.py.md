# `.\pytorch\tools\code_coverage\package\tool\clang_coverage.py`

```
# 导入未来版本的注释，用于支持类型注解
from __future__ import annotations

# 导入操作系统相关功能
import os
# 导入子进程管理模块
import subprocess
# 导入时间模块
import time

# 导入设置相关的常量和枚举
from ..util.setting import (
    JSON_FOLDER_BASE_DIR,
    MERGED_FOLDER_BASE_DIR,
    TestList,
    TestPlatform,
    TestType,
)
# 导入自定义的实用函数
from ..util.utils import (
    check_platform_type,
    convert_to_relative_path,
    create_folder,
    get_raw_profiles_folder,
    get_test_name_from_whole_path,
    print_log,
    print_time,
    related_to_test_list,
    replace_extension,
)
# 导入本地模块中的函数
from .utils import get_tool_path_by_platform, run_cpp_test


# 定义一个函数，用于创建相应的文件夹结构
def create_corresponding_folder(
    cur_path: str, prefix_cur_path: str, dir_list: list[str], new_base_folder: str
) -> None:
    for dir_name in dir_list:
        # 计算相对路径
        relative_path = convert_to_relative_path(
            cur_path, prefix_cur_path
        )  # 获取类似 'aten' 的文件夹名
        # 构建新文件夹路径
        new_folder_path = os.path.join(new_base_folder, relative_path, dir_name)
        # 调用函数创建文件夹
        create_folder(new_folder_path)


# 定义一个函数，用于运行目标测试
def run_target(
    binary_file: str, raw_file: str, test_type: TestType, platform_type: TestPlatform
) -> None:
    print_log("start run: ", binary_file)
    # 设置环境变量 -- 设置二进制运行的原始性能文件输出路径
    os.environ["LLVM_PROFILE_FILE"] = raw_file
    # 运行二进制文件
    if test_type == TestType.PY and platform_type == TestPlatform.OSS:
        from ..oss.utils import run_oss_python_test

        # 调用 OSS 平台的 Python 测试函数
        run_oss_python_test(binary_file)
    else:
        # 否则运行 C++ 测试
        run_cpp_test(binary_file)


# 定义一个函数，用于合并目标文件
def merge_target(raw_file: str, merged_file: str, platform_type: TestPlatform) -> None:
    print_log("start to merge target: ", raw_file)
    # 运行命令
    llvm_tool_path = get_tool_path_by_platform(platform_type)
    subprocess.check_call(
        [
            f"{llvm_tool_path}/llvm-profdata",
            "merge",
            "-sparse",
            raw_file,
            "-o",
            merged_file,
        ]
    )


# 定义一个函数，用于导出目标文件
def export_target(
    merged_file: str,
    json_file: str,
    binary_file: str,
    shared_library_list: list[str],
    platform_type: TestPlatform,
) -> None:
    if binary_file is None:
        # 如果没有找到对应的二进制文件，则抛出异常
        raise Exception(
            f"{merged_file} doesn't have corresponding binary!"
        )
    print_log("start to export: ", merged_file)
    # 运行导出操作
    cmd_shared_library = (
        ""
        if not shared_library_list
        else f" -object  {' -object '.join(shared_library_list)}"
    )
    # 如果没有二进制文件，无需添加（Python 测试）
    cmd_binary = "" if not binary_file else f" -object {binary_file} "
    llvm_tool_path = get_tool_path_by_platform(platform_type)

    # 构建导出命令
    cmd = f"{llvm_tool_path}/llvm-cov export {cmd_binary} {cmd_shared_library}  -instr-profile={merged_file} > {json_file}"
    # 执行命令
    os.system(cmd)


# 定义一个函数，用于执行合并操作
def merge(test_list: TestList, platform_type: TestPlatform) -> None:
    print("start merge")
    start_time = time.time()
    # 查找所有原始性能文件的路径（包括子文件夹）
    raw_folder_path = get_raw_profiles_folder()
    g = os.walk(raw_folder_path)
    # 遍历生成器 g 中的每个元素，每个元素包含路径 path、文件夹列表 dir_list 和文件列表 file_list
    for path, dir_list, file_list in g:
        # 如果路径中存在 raw/aten/ 文件夹，且在 MERGED_FOLDER_BASE_DIR 中对应的 profile/merged/aten/ 文件夹不存在，则创建之
        create_corresponding_folder(
            path, raw_folder_path, dir_list, MERGED_FOLDER_BASE_DIR
        )
        # 遍历文件列表 file_list，检查是否存在以 .profraw 结尾的文件名
        for file_name in file_list:
            # 如果文件名以 .profraw 结尾，并且不属于 test_list 中指定的相关测试文件
            if file_name.endswith(".profraw"):
                if not related_to_test_list(file_name, test_list):
                    continue
                # 打印开始合并文件的信息
                print(f"start merge {file_name}")
                # 构建原始文件的完整路径
                raw_file = os.path.join(path, file_name)
                # 构建合并后文件的文件名，将 .profraw 替换为 .merged
                merged_file_name = replace_extension(file_name, ".merged")
                # 构建合并后文件的完整路径
                merged_file = os.path.join(
                    MERGED_FOLDER_BASE_DIR,
                    convert_to_relative_path(path, raw_folder_path),
                    merged_file_name,
                )
                # 执行合并操作，将原始文件 raw_file 合并到 merged_file 中，考虑 platform_type
                merge_target(raw_file, merged_file, platform_type)
    # 打印合并操作总耗时信息
    print_time("merge take time: ", start_time, summary_time=True)
def export(test_list: TestList, platform_type: TestPlatform) -> None:
    # 打印开始导出信息
    print("start export")
    # 记录开始时间
    start_time = time.time()
    # 遍历 MERGED_FOLDER_BASE_DIR 目录及其子目录下的所有文件和文件夹
    g = os.walk(MERGED_FOLDER_BASE_DIR)
    for path, dir_list, file_list in g:
        # 如果在 JSON_FOLDER_BASE_DIR 中不存在对应的合并文件夹，则创建
        create_corresponding_folder(
            path, MERGED_FOLDER_BASE_DIR, dir_list, JSON_FOLDER_BASE_DIR
        )
        # 检查当前路径下是否有以 .merged 结尾的文件
        for file_name in file_list:
            if file_name.endswith(".merged"):
                # 如果文件与测试列表无关，则跳过当前文件
                if not related_to_test_list(file_name, test_list):
                    continue
                # 打印开始导出具体文件的信息
                print(f"start export {file_name}")
                # 合并文件的完整路径
                merged_file = os.path.join(path, file_name)
                # 目标 JSON 文件名
                json_file_name = replace_extension(file_name, ".json")
                # JSON 文件的完整路径
                json_file = os.path.join(
                    JSON_FOLDER_BASE_DIR,
                    convert_to_relative_path(path, MERGED_FOLDER_BASE_DIR),
                    json_file_name,
                )
                # 检查平台类型是否合法
                check_platform_type(platform_type)
                # 二进制文件和共享库列表初始化
                binary_file = ""
                shared_library_list = []
                # 如果平台类型为 FBCODE
                if platform_type == TestPlatform.FBCODE:
                    from caffe2.fb.code_coverage.tool.package.fbcode.utils import (
                        get_fbcode_binary_folder,
                    )
                    # 获取 FBCODE 平台下的二进制文件路径
                    binary_file = os.path.join(
                        get_fbcode_binary_folder(path),
                        get_test_name_from_whole_path(merged_file),
                    )
                # 如果平台类型为 OSS
                elif platform_type == TestPlatform.OSS:
                    from ..oss.utils import get_oss_binary_file, get_oss_shared_library
                    # 获取测试名称
                    test_name = get_test_name_from_whole_path(merged_file)
                    # 如果是 Python 测试，则不需要提供二进制文件，仅需共享库
                    binary_file = (
                        ""
                        if test_name.endswith(".py")
                        else get_oss_binary_file(test_name, TestType.CPP)
                    )
                    # 获取 OSS 平台下的共享库列表
                    shared_library_list = get_oss_shared_library()
                # 导出目标文件
                export_target(
                    merged_file,
                    json_file,
                    binary_file,
                    shared_library_list,
                    platform_type,
                )
    # 打印导出所花时间的汇总信息
    print_time("export take time: ", start_time, summary_time=True)
```
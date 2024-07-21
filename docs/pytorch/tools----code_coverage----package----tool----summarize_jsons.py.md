# `.\pytorch\tools\code_coverage\package\tool\summarize_jsons.py`

```py
# 从 __future__ 模块导入 annotations 特性，这是为了支持后续的类型提示功能
from __future__ import annotations

# 导入所需的标准库模块
import json
import os
import time
# 导入类型提示相关的模块
from typing import Any, TYPE_CHECKING

# 导入自定义模块和类型定义
from ..util.setting import (
    CompilerType,
    JSON_FOLDER_BASE_DIR,
    TestList,
    TestPlatform,
    TestStatusType,
)
from ..util.utils import (
    detect_compiler_type,
    print_error,
    print_time,
    related_to_test_list,
)

# 导入特定的解析器模块
from .parser.gcov_coverage_parser import GcovCoverageParser
from .parser.llvm_coverage_parser import LlvmCoverageParser
# 导入打印报告相关的模块
from .print_report import (
    file_oriented_report,
    html_oriented_report,
    line_oriented_report,
)

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入类型定义，用于类型检查
    from .parser.coverage_record import CoverageRecord

# 初始化全局变量，用于记录覆盖的行和未覆盖的行
covered_lines: dict[str, set[int]] = {}
uncovered_lines: dict[str, set[int]] = {}

# 初始化测试类型的状态字典
tests_type: TestStatusType = {"success": set(), "partial": set(), "fail": set()}


def transform_file_name(
    file_path: str, interested_folders: list[str], platform: TestPlatform
) -> str:
    # 定义需要移除的文件名后缀模式集合
    remove_patterns: set[str] = {".DEFAULT.cpp", ".AVX.cpp", ".AVX2.cpp"}
    # 遍历每个模式，替换文件路径中的匹配模式为空
    for pattern in remove_patterns:
        file_path = file_path.replace(pattern, "")
    
    # 如果用户指定了感兴趣的文件夹
    if interested_folders:
        # 遍历每个感兴趣的文件夹
        for folder in interested_folders:
            # 如果文件路径中包含当前文件夹名
            if folder in file_path:
                return file_path[file_path.find(folder) :]
    
    # 如果是 OSS 平台，移除 PyTorch 基础文件夹路径
    if platform == TestPlatform.OSS:
        # 导入 OSS 特定的工具函数
        from package.oss.utils import get_pytorch_folder  # type: ignore[import]

        # 获取 PyTorch 文件夹路径
        pytorch_folder = get_pytorch_folder()
        # 断言文件路径以 PyTorch 文件夹路径开头
        assert file_path.startswith(pytorch_folder)
        # 截取除去 PyTorch 文件夹路径后的文件路径
        file_path = file_path[len(pytorch_folder) + 1 :]
    
    # 返回处理后的文件路径
    return file_path


def is_intrested_file(
    file_path: str, interested_folders: list[str], platform: TestPlatform
) -> bool:
    # 定义需要忽略的文件路径模式列表
    ignored_patterns = ["cuda", "aten/gen_aten", "aten/aten_", "build/"]
    # 如果文件路径中包含任何忽略的模式，则返回 False
    if any(pattern in file_path for pattern in ignored_patterns):
        return False

    # 如果是 OSS 平台，忽略不属于 PyTorch 的文件
    if platform == TestPlatform.OSS:
        # 导入 OSS 特定的工具函数
        from package.oss.utils import get_pytorch_folder

        # 如果文件路径不是以 PyTorch 文件夹路径开头，则返回 False
        if not file_path.startswith(get_pytorch_folder()):
            return False
    
    # 如果用户指定了感兴趣的文件夹
    if interested_folders:
        # 遍历每个感兴趣的文件夹
        for folder in interested_folders:
            # 构建感兴趣文件夹的路径，如果不以 '/' 结尾，则添加 '/'
            interested_folder_path = folder if folder.endswith("/") else f"{folder}/"
            # 如果文件路径中包含当前感兴趣文件夹路径，则返回 True
            if interested_folder_path in file_path:
                return True
        # 如果文件路径不匹配任何感兴趣的文件夹，则返回 False
        return False
    else:
        # 如果未指定感兴趣的文件夹，则返回 True
        return True


def get_json_obj(json_file: str) -> tuple[Any, int]:
    """
    有时在文件开头，llvm/gcov 会报错"fail to find coverage data"，
    这时需要跳过这些行
      -- success read: 0      - 该 JSON 文件包含完整的覆盖信息
      -- partial success: 1   - 该 JSON 文件以一些错误提示开头，但仍包含覆盖信息
      -- fail to read: 2      - 该 JSON 文件不包含任何覆盖信息
    """
    # 这是一个文档字符串，用于解释函数的作用和返回值含义
    # 初始化读取状态为 -1
    read_status = -1
    # 使用上下文管理器打开 JSON 文件
    with open(json_file) as f:
        # 逐行读取文件内容
        lines = f.readlines()
        # 遍历每一行内容
        for line in lines:
            try:
                # 尝试解析 JSON 字符串
                json_obj = json.loads(line)
            except json.JSONDecodeError:
                # 如果解析失败，设置读取状态为 1，并继续下一行
                read_status = 1
                continue
            else:
                # 解析成功的情况下
                if read_status == -1:
                    # 如果之前没有遇到过 JSON 解析错误，设置读取状态为 0
                    read_status = 0
                # 返回解析得到的 JSON 对象及读取状态
                return (json_obj, read_status)
    # 若未在循环中返回，则返回 None 及状态码 2，表示文件中没有有效的 JSON 数据
    return None, 2
# 解析给定的 JSON 文件并返回覆盖率记录列表
def parse_json(json_file: str, platform: TestPlatform) -> list[CoverageRecord]:
    # 打印开始解析的信息和 JSON 文件名
    print("start parse:", json_file)
    # 调用函数获取 JSON 对象和读取状态
    json_obj, read_status = get_json_obj(json_file)
    # 根据读取状态将文件名添加到对应的测试类型集合中
    if read_status == 0:
        tests_type["success"].add(json_file)
    elif read_status == 1:
        tests_type["partial"].add(json_file)
    else:
        tests_type["fail"].add(json_file)
        # 抛出运行时错误，指示加载 JSON 文件失败
        raise RuntimeError(
            "Fail to do code coverage! Fail to load json file: ", json_file
        )

    # 检测编译器类型
    cov_type = detect_compiler_type(platform)

    # 初始化覆盖率记录列表
    coverage_records: list[CoverageRecord] = []
    # 根据不同的编译器类型选择不同的解析器来解析 JSON 对象
    if cov_type == CompilerType.CLANG:
        coverage_records = LlvmCoverageParser(json_obj).parse("fbcode")
        # 如果需要，可以打印解析得到的覆盖率记录
        # print(coverage_records)
    elif cov_type == CompilerType.GCC:
        coverage_records = GcovCoverageParser(json_obj).parse()

    # 返回解析得到的覆盖率记录列表
    return coverage_records


# 解析给定的测试列表中的所有 JSON 文件
def parse_jsons(
    test_list: TestList, interested_folders: list[str], platform: TestPlatform
) -> None:
    # 遍历指定目录下的所有文件和子目录
    g = os.walk(JSON_FOLDER_BASE_DIR)

    for path, _, file_list in g:
        for file_name in file_list:
            # 只处理以 .json 结尾的文件
            if file_name.endswith(".json"):
                # 检测当前平台的编译器类型
                cov_type = detect_compiler_type(platform)
                # 如果编译器是 clang，并且 JSON 文件与测试列表无关，则跳过
                if cov_type == CompilerType.CLANG and not related_to_test_list(
                    file_name, test_list
                ):
                    continue
                # 构建完整的 JSON 文件路径
                json_file = os.path.join(path, file_name)
                try:
                    # 解析 JSON 文件，获取覆盖率记录
                    coverage_records = parse_json(json_file, platform)
                except RuntimeError:
                    # 如果解析失败，打印错误信息并继续下一个文件的处理
                    print_error("Fail to load json file: ", json_file)
                    continue
                # 更新覆盖率信息，合并到感兴趣的文件夹中
                update_coverage(coverage_records, interested_folders, platform)


# 更新覆盖率信息，将新记录合并到感兴趣的文件夹中
def update_coverage(
    coverage_records: list[CoverageRecord],
    interested_folders: list[str],
    platform: TestPlatform,
) -> None:
    # 对于每条覆盖记录中的每个项目进行处理
    for item in coverage_records:
        # 将记录转换为字典格式
        record = item.to_dict()
        # 获取文件路径
        file_path = record["filepath"]
        # 检查文件路径是否属于感兴趣的文件夹，并符合特定平台要求
        if not is_intrested_file(file_path, interested_folders, platform):
            continue
        # 获取已覆盖和未覆盖的行范围
        covered_range = record["covered_lines"]
        uncovered_range = record["uncovered_lines"]
        # 转换文件路径名：将远程路径格式转换为本地路径格式
        file_path = transform_file_name(file_path, interested_folders, platform)

        # 如果文件路径在已覆盖行字典中不存在，将其加入字典
        if file_path not in covered_lines:
            covered_lines[file_path] = set()
        # 如果文件路径在未覆盖行字典中不存在，将其加入字典
        if file_path not in uncovered_lines:
            uncovered_lines[file_path] = set()
        
        # 更新该文件的已覆盖和未覆盖行集合
        if covered_range is not None:
            covered_lines[file_path].update(covered_range)
        if uncovered_range is not None:
            uncovered_lines[file_path].update(uncovered_range)
# 更新覆盖线和未覆盖线集合之间的差集
def update_set() -> None:
    # 遍历覆盖线集合中的每个文件名
    for file_name in covered_lines:
        # 对未覆盖线集合中对应文件名的集合执行差集更新操作
        uncovered_lines[file_name].difference_update(covered_lines[file_name])

# 汇总 JSON 数据
def summarize_jsons(
    test_list: TestList,
    interested_folders: list[str],
    coverage_only: list[str],
    platform: TestPlatform,
) -> None:
    # 记录函数开始时间
    start_time = time.time()
    
    # 根据平台类型检测编译器类型，若为 GCC 则生成 HTML 导向的报告
    if detect_compiler_type(platform) == CompilerType.GCC:
        html_oriented_report()
    else:
        # 解析 JSON 数据，基于传入参数解析特定文件夹下的 JSON 文件
        parse_jsons(test_list, interested_folders, platform)
        # 更新未覆盖线集合中的数据
        update_set()
        # 生成基于行的测试覆盖报告
        line_oriented_report(
            test_list,
            tests_type,  # 变量 tests_type 应该在函数外定义
            interested_folders,
            coverage_only,
            covered_lines,
            uncovered_lines,
        )
        # 生成基于文件的测试覆盖报告
        file_oriented_report(
            test_list,
            tests_type,  # 变量 tests_type 应该在函数外定义
            interested_folders,
            coverage_only,
            covered_lines,
            uncovered_lines,
        )
    
    # 打印函数执行时间
    print_time("summary jsons take time: ", start_time)
```
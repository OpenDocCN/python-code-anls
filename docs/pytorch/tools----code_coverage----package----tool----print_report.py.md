# `.\pytorch\tools\code_coverage\package\tool\print_report.py`

```
# 从未来版本导入类型提示的功能
from __future__ import annotations

# 导入操作系统相关功能
import os
# 导入子进程管理模块
import subprocess
# 导入类型提示相关功能
from typing import IO, Tuple

# 导入自定义模块
from ..oss.utils import get_pytorch_folder
from ..util.setting import SUMMARY_FOLDER_DIR, TestList, TestStatusType

# 定义元组类型，表示覆盖率信息项
CoverageItem = Tuple[str, float, int, int]

# 根据覆盖率百分比排序的函数
def key_by_percentage(x: CoverageItem) -> float:
    return x[1]

# 根据名称排序的函数
def key_by_name(x: CoverageItem) -> str:
    return x[0]

# 判断文件是否属于感兴趣的文件类型
def is_intrested_file(file_path: str, interested_folders: list[str]) -> bool:
    # 如果文件路径中包含"cuda"，则不感兴趣
    if "cuda" in file_path:
        return False
    # 如果文件路径中包含特定模式如"aten/gen_aten"或"aten/aten_"，则不感兴趣
    if "aten/gen_aten" in file_path or "aten/aten_" in file_path:
        return False
    # 遍历感兴趣的文件夹列表，如果文件路径中包含其中任何一个文件夹，则感兴趣
    for folder in interested_folders:
        if folder in file_path:
            return True
    # 否则不感兴趣
    return False

# 判断该测试是否属于特定类型的测试集合
def is_this_type_of_tests(target_name: str, test_set_by_type: set[str]) -> bool:
    # 遍历测试集合中的每一个测试名称，如果目标名称匹配，则属于该类型的测试
    for test in test_set_by_type:
        if target_name in test:
            return True
    # 否则不属于该类型的测试
    return False

# 打印特定类型的测试到汇总文件中
def print_test_by_type(
    tests: TestList, test_set_by_type: set[str], type_name: str, summary_file: IO[str]
) -> None:
    print("Tests " + type_name + " to collect coverage:", file=summary_file)
    # 遍历测试列表，打印属于指定类型的测试
    for test in tests:
        if is_this_type_of_tests(test.name, test_set_by_type):
            print(test.target_pattern, file=summary_file)
    print(file=summary_file)

# 打印测试条件到汇总文件中
def print_test_condition(
    tests: TestList,
    tests_type: TestStatusType,
    interested_folders: list[str],
    coverage_only: list[str],
    summary_file: IO[str],
    summary_type: str,
) -> None:
    # 打印各种类型的测试到汇总文件中
    print_test_by_type(tests, tests_type["success"], "fully success", summary_file)
    print_test_by_type(tests, tests_type["partial"], "partially success", summary_file)
    print_test_by_type(tests, tests_type["fail"], "failed", summary_file)
    # 打印感兴趣的文件夹列表到汇总文件中
    print(
        "\n\nCoverage Collected Over Interested Folders:\n",
        interested_folders,
        file=summary_file,
    )
    # 打印只应用于覆盖率编译标志的文件列表到汇总文件中
    print(
        "\n\nCoverage Compilation Flags Only Apply To: \n",
        coverage_only,
        file=summary_file,
    )
    # 打印汇总类型分隔线到汇总文件中
    print(
        "\n\n---------------------------------- "
        + summary_type
        + " ----------------------------------",
        file=summary_file,
    )

# 基于行的报告函数，将覆盖的和未覆盖的行按文件进行报告
def line_oriented_report(
    tests: TestList,
    tests_type: TestStatusType,
    interested_folders: list[str],
    coverage_only: list[str],
    covered_lines: dict[str, set[int]],
    uncovered_lines: dict[str, set[int]],
) -> None:
    # 使用指定的路径和文件名打开文件，模式为读写，文件不存在则创建
    with open(os.path.join(SUMMARY_FOLDER_DIR, "line_summary"), "w+") as report_file:
        # 调用打印测试条件的函数，将结果写入报告文件
        print_test_condition(
            tests,              # 测试用例列表
            tests_type,         # 测试类型
            interested_folders, # 感兴趣的文件夹列表
            coverage_only,      # 是否只考虑覆盖率
            report_file,        # 报告文件对象
            "LINE SUMMARY",     # 标题
        )
        # 遍历覆盖行字典中的每个文件名
        for file_name in covered_lines:
            # 获取当前文件名下的覆盖行和未覆盖行列表
            covered = covered_lines[file_name]     # 覆盖的行号列表
            uncovered = uncovered_lines[file_name] # 未覆盖的行号列表
            # 将文件名、覆盖行和未覆盖行信息写入报告文件
            print(
                f"{file_name}\n  covered lines: {sorted(covered)}\n  unconvered lines:{sorted(uncovered)}",
                file=report_file,   # 输出目标为报告文件
            )
# 打印文件汇总信息，返回覆盖率百分比
def print_file_summary(
    covered_summary: int, total_summary: int, summary_file: IO[str]
) -> float:
    # 尝试计算覆盖率百分比，处理零除错误
    try:
        coverage_percentage = 100.0 * covered_summary / total_summary
    except ZeroDivisionError:
        coverage_percentage = 0
    # 打印汇总信息到指定文件
    print(
        f"SUMMARY\ncovered: {covered_summary}\nuncovered: {total_summary}\npercentage: {coverage_percentage:.2f}%\n\n",
        file=summary_file,
    )
    # 如果覆盖率为0，输出提示信息
    if coverage_percentage == 0:
        print("Coverage is 0, Please check if json profiles are valid")
    # 返回计算的覆盖率百分比
    return coverage_percentage


# 打印文件导向的报告
def print_file_oriented_report(
    tests_type: TestStatusType,
    coverage: list[CoverageItem],
    covered_summary: int,
    total_summary: int,
    summary_file: IO[str],
    tests: TestList,
    interested_folders: list[str],
    coverage_only: list[str],
) -> None:
    # 获取文件汇总的覆盖率百分比
    coverage_percentage = print_file_summary(
        covered_summary, total_summary, summary_file
    )
    # 打印测试条件（关注的文件夹 / 成功或失败的测试）
    print_test_condition(
        tests,
        tests_type,
        interested_folders,
        coverage_only,
        summary_file,
        "FILE SUMMARY",
    )
    # 打印每个文件的信息
    for item in coverage:
        print(
            item[0].ljust(75),
            (str(item[1]) + "%").rjust(10),
            str(item[2]).rjust(10),
            str(item[3]).rjust(10),
            file=summary_file,
        )
    # 打印汇总的覆盖率百分比
    print(f"summary percentage:{coverage_percentage:.2f}%")


# 文件导向的报告
def file_oriented_report(
    tests: TestList,
    tests_type: TestStatusType,
    interested_folders: list[str],
    coverage_only: list[str],
    covered_lines: dict[str, set[int]],
    uncovered_lines: dict[str, set[int]],
) -> None:
    # 打开文件用于写入汇总信息
    with open(os.path.join(SUMMARY_FOLDER_DIR, "file_summary"), "w+") as summary_file:
        covered_summary = 0
        total_summary = 0
        coverage = []
        # 遍历每个文件的覆盖行数信息
        for file_name in covered_lines:
            # 获取当前文件的覆盖行数和总行数
            covered_count = len(covered_lines[file_name])
            total_count = covered_count + len(uncovered_lines[file_name])
            try:
                # 计算覆盖率
                percentage = round(covered_count / total_count * 100, 2)
            except ZeroDivisionError:
                percentage = 0
            # 将文件信息存储在列表中以便排序
            coverage.append((file_name, percentage, covered_count, total_count))
            # 更新汇总信息
            covered_summary = covered_summary + covered_count
            total_summary = total_summary + total_count
        # 按照文件名和覆盖率进行排序
        coverage.sort(key=key_by_name)
        coverage.sort(key=key_by_percentage)
        # 打印文件导向的报告
        print_file_oriented_report(
            tests_type,
            coverage,
            covered_summary,
            total_summary,
            summary_file,
            tests,
            interested_folders,
            coverage_only,
        )
# 返回一个列表，包含需要忽略的文件路径模式列表，用于代码覆盖报告
def get_html_ignored_pattern() -> list[str]:
    return ["/usr/*", "*anaconda3/*", "*third_party/*"]


# 生成针对 HTML 页面的代码覆盖报告
def html_oriented_report() -> None:
    # 获取 PyTorch 源码文件夹路径，并拼接生成文件夹路径
    build_folder = os.path.join(get_pytorch_folder(), "build")
    # 定义覆盖信息文件路径为总结文件夹中的 coverage.info
    coverage_info_file = os.path.join(SUMMARY_FOLDER_DIR, "coverage.info")
    # 使用 lcov 工具捕获代码覆盖率信息到 coverage.info 文件中
    subprocess.check_call(
        [
            "lcov",
            "--capture",
            "--directory",
            build_folder,
            "--output-file",
            coverage_info_file,
        ]
    )
    # 移除与特定模式匹配的文件信息，更新覆盖信息文件
    cmd_array = (
        ["lcov", "--remove", coverage_info_file]
        + get_html_ignored_pattern()
        + ["--output-file", coverage_info_file]
    )
    subprocess.check_call(
        cmd_array
    )
    # 使用 genhtml 工具生成漂亮的 HTML 报告页面
    subprocess.check_call(
        [
            "genhtml",
            coverage_info_file,
            "--output-directory",
            os.path.join(SUMMARY_FOLDER_DIR, "html_report"),
        ]
    )
```
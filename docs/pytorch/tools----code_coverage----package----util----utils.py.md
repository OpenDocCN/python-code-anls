# `.\pytorch\tools\code_coverage\package\util\utils.py`

```py
# 从未来版本导入注解支持，用于类型提示
from __future__ import annotations

# 导入标准库模块
import os
import shutil
import sys
import time
from typing import Any, NoReturn

# 从自定义模块中导入指定内容
from .setting import (
    CompilerType,
    LOG_DIR,
    PROFILE_DIR,
    TestList,
    TestPlatform,
    TestType,
)


# 将秒数转换为时:分:秒格式的字符串表示
def convert_time(seconds: float) -> str:
    seconds = int(round(seconds))  # 将浮点数秒数四舍五入取整
    seconds = seconds % (24 * 3600)  # 计算出一天中剩余的秒数
    hour = seconds // 3600  # 计算小时数
    seconds %= 3600
    minutes = seconds // 60  # 计算分钟数
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)  # 返回格式化后的时间字符串


# 打印消息和从开始时间到当前时间的持续时间，将结果记录到日志文件中
def print_time(message: str, start_time: float, summary_time: bool = False) -> None:
    with open(os.path.join(LOG_DIR, "log.txt"), "a+") as log_file:
        end_time = time.time()  # 获取当前时间作为结束时间
        print(message, convert_time(end_time - start_time), file=log_file)  # 打印消息和时间到日志文件
        if summary_time:
            print("\n", file=log_file)  # 如果需要总结时间，添加换行到日志文件


# 打印日志消息到日志文件中
def print_log(*args: Any) -> None:
    with open(os.path.join(LOG_DIR, "log.txt"), "a+") as log_file:
        print(f"[LOG] {' '.join(args)}", file=log_file)  # 格式化日志消息并写入日志文件


# 打印错误消息到日志文件中
def print_error(*args: Any) -> None:
    with open(os.path.join(LOG_DIR, "log.txt"), "a+") as log_file:
        print(f"[ERROR] {' '.join(args)}", file=log_file)  # 格式化错误消息并写入日志文件


# 如果路径存在，则删除文件
def remove_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


# 递归删除文件夹及其内容
def remove_folder(path: str) -> None:
    shutil.rmtree(path)


# 创建文件夹，如果文件夹已存在则不做任何操作
def create_folder(*paths: Any) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


# 清理由覆盖工具生成的所有文件
def clean_up() -> None:
    # 删除配置文件夹
    remove_folder(PROFILE_DIR)
    sys.exit("Clean Up Successfully!")  # 终止程序并显示清理成功消息


# 将整个路径转换为相对路径，基于指定的基础路径
def convert_to_relative_path(whole_path: str, base_path: str) -> str:
    # 示例：("profile/raw", "profile") -> "raw"
    if base_path not in whole_path:
        raise RuntimeError(base_path + " is not in " + whole_path)
    return whole_path[len(base_path) + 1:]  # 返回相对路径部分


# 替换文件名的扩展名
def replace_extension(filename: str, ext: str) -> str:
    return filename[: filename.rfind(".")] + ext


# 检查文件是否与测试列表中的任何一个相关联
def related_to_test_list(file_name: str, test_list: TestList) -> bool:
    for test in test_list:
        if test.name in file_name:
            return True
    return False


# 获取原始配置文件夹的路径
def get_raw_profiles_folder() -> str:
    return os.environ.get("RAW_PROFILES_FOLDER", os.path.join(PROFILE_DIR, "raw"))


# 检测编译器类型，根据测试平台返回相应的编译器类型
def detect_compiler_type(platform: TestPlatform) -> CompilerType:
    if platform == TestPlatform.OSS:
        from package.oss.utils import (  # type: ignore[assignment, import, misc]
            detect_compiler_type,
        )
        cov_type = detect_compiler_type()  # 调用特定平台下的编译器检测函数
    else:
        from caffe2.fb.code_coverage.tool.package.fbcode.utils import (  # type: ignore[import]
            detect_compiler_type,
        )
        cov_type = detect_compiler_type()  # 调用默认平台下的编译器检测函数

    check_compiler_type(cov_type)  # 检查编译器类型
    return cov_type  # 返回检测到的编译器类型


# 从完整路径中提取测试名称
def get_test_name_from_whole_path(path: str) -> str:
    # 待实现
    # 找到路径中最后一个斜杠的位置
    start = path.rfind("/")
    # 找到路径中最后一个点号的位置，通常用于确定文件名的扩展名的结束位置
    end = path.rfind(".")
    # 确保路径中至少有一个斜杠和一个点号，否则断言失败
    assert start >= 0 and end >= 0
    # 返回从最后一个斜杠后到最后一个点号前的子字符串，即文件名部分（不包括路径和扩展名）
    return path[start + 1 : end]
# 检查编译器类型是否为 GCC 或 CLANG
def check_compiler_type(cov_type: CompilerType | None) -> None:
    # 如果编译器类型不为空，并且是 GCC 或 CLANG 中的一种，则通过
    if cov_type is not None and cov_type in [CompilerType.GCC, CompilerType.CLANG]:
        return
    # 否则，抛出异常并提供错误信息和建议
    raise Exception(
        f"Can't parse compiler type: {cov_type}.",
        " Please set environment variable COMPILER_TYPE as CLANG or GCC",
    )


# 检查测试平台类型是否为 OSS 或 FBCODE
def check_platform_type(platform_type: TestPlatform) -> None:
    # 如果测试平台类型是 OSS 或 FBCODE 中的一种，则通过
    if platform_type in [TestPlatform.OSS, TestPlatform.FBCODE]:
        return
    # 否则，抛出异常并提供错误信息和建议
    raise Exception(
        f"Can't parse platform type: {platform_type}.",
        " Please set environment variable COMPILER_TYPE as OSS or FBCODE",
    )


# 检查测试类型是否为 CPP 或 PY，并提供关于 buck target 的检查建议
def check_test_type(test_type: str, target: str) -> None:
    # 如果测试类型为 CPP 或 PY 中的一种，则通过
    if test_type in [TestType.CPP.value, TestType.PY.value]:
        return
    # 否则，抛出异常并提供错误信息和建议
    raise Exception(
        f"Can't parse test type: {test_type}.",
        f" Please check the type of buck target: {target}",
    )


# 抛出找不到测试的运行时异常，指定 CPP 和 Python 测试的文件夹路径
def raise_no_test_found_exception(
    cpp_binary_folder: str, python_binary_folder: str
) -> NoReturn:
    raise RuntimeError(
        f"No cpp and python tests found in folder **{cpp_binary_folder} and **{python_binary_folder}**"
    )
```
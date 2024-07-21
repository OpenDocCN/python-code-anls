# `.\pytorch\tools\code_coverage\package\tool\gcc_coverage.py`

```py
from __future__ import annotations

import os                   # 导入操作系统相关模块
import subprocess           # 导入子进程管理模块
import time                 # 导入时间模块

# gcc is only used in oss
from ..oss.utils import get_gcda_files, run_oss_python_test   # 导入oss工具函数：获取.gcda文件列表和运行oss Python测试函数
from ..util.setting import JSON_FOLDER_BASE_DIR, TestType     # 导入设置：JSON文件夹基本目录和测试类型枚举
from ..util.utils import print_log, print_time                # 导入打印日志和打印时间的工具函数
from .utils import run_cpp_test                               # 导入C++测试运行函数


def update_gzip_dict(gzip_dict: dict[str, int], file_name: str) -> str:
    # 将文件名转换为小写
    file_name = file_name.lower()
    # 更新gzip文件字典，记录文件名出现次数
    gzip_dict[file_name] = gzip_dict.get(file_name, 0) + 1
    # 获取文件名出现的次数
    num = gzip_dict[file_name]
    # 返回格式化后的文件名，格式为"次数_文件名"
    return str(num) + "_" + file_name


def run_target(binary_file: str, test_type: TestType) -> None:
    # 打印测试开始日志
    print_log("start run", test_type.value, "test: ", binary_file)
    # 记录开始时间
    start_time = time.time()
    # 断言测试类型为CPP或PY
    assert test_type in {TestType.CPP, TestType.PY}
    # 根据测试类型选择运行对应的测试
    if test_type == TestType.CPP:
        run_cpp_test(binary_file)
    else:
        run_oss_python_test(binary_file)

    # 打印测试完成时间
    print_time(" time: ", start_time)


def export() -> None:
    # 记录开始时间
    start_time = time.time()
    # 收集.gcda文件列表
    gcda_files = get_gcda_files()
    # 初始化gzip文件字典，用于记录文件名出现次数
    gzip_dict: dict[str, int] = {}
    # 遍历每个gcda文件项
    for gcda_item in gcda_files:
        # 生成对应的json.gz文件
        subprocess.check_call(["gcov", "-i", gcda_item])
        # 构建新的文件路径，将生成的gz文件移动到指定目录下
        gz_file_name = os.path.basename(gcda_item) + ".gcov.json.gz"
        new_file_path = os.path.join(
            JSON_FOLDER_BASE_DIR, update_gzip_dict(gzip_dict, gz_file_name)
        )
        os.rename(gz_file_name, new_file_path)
        # 解压缩json.gz文件为json文件
        subprocess.check_output(["gzip", "-d", new_file_path])
    # 打印导出过程所花费的时间
    print_time("export take time: ", start_time, summary_time=True)
```
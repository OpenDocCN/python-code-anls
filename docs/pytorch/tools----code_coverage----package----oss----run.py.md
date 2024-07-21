# `.\pytorch\tools\code_coverage\package\oss\run.py`

```py
import os
import time

from ..tool import clang_coverage, gcc_coverage
from ..util.setting import TestList, TestPlatform
from ..util.utils import get_raw_profiles_folder, print_time
from .utils import get_oss_binary_file


def clang_run(tests: TestList) -> None:
    # 记录开始时间
    start_time = time.time()
    # 遍历测试列表中的每个测试对象
    for test in tests:
        # 构造原始文件路径
        raw_file = os.path.join(get_raw_profiles_folder(), test.name + ".profraw")
        # 获取二进制文件路径
        binary_file = get_oss_binary_file(test.name, test.test_type)
        # 调用 clang_coverage 模块运行目标程序
        clang_coverage.run_target(
            binary_file, raw_file, test.test_type, TestPlatform.OSS
        )
    # 打印运行时间信息
    print_time("running binaries takes time: ", start_time, summary_time=True)


def gcc_run(tests: TestList) -> None:
    # 记录开始时间
    start_time = time.time()
    # 遍历测试列表中的每个测试对象
    for test in tests:
        # 获取二进制文件路径
        binary_file = get_oss_binary_file(test.name, test.test_type)
        # 调用 gcc_coverage 模块运行目标程序
        gcc_coverage.run_target(binary_file, test.test_type)
    # 打印运行时间信息
    print_time("run binaries takes time: ", start_time, summary_time=True)
```
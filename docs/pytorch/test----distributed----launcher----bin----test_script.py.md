# `.\pytorch\test\distributed\launcher\bin\test_script.py`

```
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse  # 导入 argparse 模块，用于处理命令行参数
import os  # 导入 os 模块，用于与操作系统交互
from pathlib import Path  # 导入 Path 类，用于处理文件路径

def parse_args():
    # 创建命令行参数解析器对象
    parser = argparse.ArgumentParser(description="test script")

    # 添加一个名为 --fail 的命令行参数，用于指定是否抛出 RuntimeError
    parser.add_argument(
        "--fail",
        default=False,
        action="store_true",
        help="forces the script to throw a RuntimeError",
    )

    # 添加一个名为 --touch-file-dir 的命令行参数，用于指定一个目录路径
    # 用于创建一个名为全局排名的文件
    parser.add_argument(
        "--touch-file-dir",
        "--touch_file_dir",
        type=str,
        help="dir to touch a file with global rank as the filename",
    )
    return parser.parse_args()  # 解析命令行参数并返回结果

def main():
    args = parse_args()  # 解析命令行参数
    env_vars = [
        "LOCAL_RANK",
        "RANK",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_NAME",
        "LOCAL_WORLD_SIZE",
        "WORLD_SIZE",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID",
        "OMP_NUM_THREADS",
        "TEST_SENTINEL_PARENT",
        "TORCHELASTIC_ERROR_FILE",
    ]

    print("Distributed env vars set by agent:")
    # 遍历环境变量列表，输出每个环境变量及其对应的值
    for env_var in env_vars:
        value = os.environ[env_var]
        print(f"{env_var} = {value}")

    if args.fail:
        raise RuntimeError("raising exception since --fail flag was set")  # 如果设置了 --fail 参数，则抛出 RuntimeError 异常
    else:
        # 否则，构造要创建的文件路径，文件名为环境变量 RANK 的值，放在指定的目录下
        file = os.path.join(args.touch_file_dir, os.environ["RANK"])
        Path(file).touch()  # 创建文件
        print(f"Success, created {file}")  # 输出成功消息

if __name__ == "__main__":
    main()  # 执行主函数
```
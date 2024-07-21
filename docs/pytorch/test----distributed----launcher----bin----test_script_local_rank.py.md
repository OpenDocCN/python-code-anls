# `.\pytorch\test\distributed\launcher\bin\test_script_local_rank.py`

```py
# 指定脚本的解释器为 Python 3
#!/usr/bin/env python3

# 脚本的所有权声明，指定责任人为 "oncall: r2p"
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# 版权声明，声明代码采用 BSD 风格许可证授权，详见根目录下的 LICENSE 文件

# 导入 argparse 和 os 模块
import argparse
import os

# 定义解析命令行参数的函数
def parse_args():
    # 创建 ArgumentParser 对象，描述为 "test script"
    parser = argparse.ArgumentParser(description="test script")

    # 添加命令行参数 "--local-rank"，别名 "--local_rank"，类型为整数，必需参数，帮助信息说明多节点分布式训练的节点排名
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        required=True,
        help="The rank of the node for multi-node distributed training",
    )

    # 解析命令行参数并返回解析结果
    return parser.parse_args()

# 主函数入口
def main():
    # 打印开始执行信息
    print("Start execution")
    # 解析命令行参数并获取结果
    args = parse_args()
    # 从环境变量 "LOCAL_RANK" 中获取预期的节点排名，并转换为整数
    expected_rank = int(os.environ["LOCAL_RANK"])
    # 获取实际传入的节点排名参数
    actual_rank = args.local_rank
    # 如果预期节点排名与实际传入的节点排名不一致，则抛出运行时错误
    if expected_rank != actual_rank:
        raise RuntimeError(
            "Parameters passed: --local-rank that has different value "
            f"from env var: expected: {expected_rank}, got: {actual_rank}"
        )
    # 打印结束执行信息
    print("End execution")

# 如果当前脚本作为主程序运行，则调用 main 函数
if __name__ == "__main__":
    main()
```
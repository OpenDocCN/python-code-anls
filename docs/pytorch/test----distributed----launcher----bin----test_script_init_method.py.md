# `.\pytorch\test\distributed\launcher\bin\test_script_init_method.py`

```
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 导入必要的库
import argparse  # 导入解析命令行参数的模块
import os  # 导入操作系统功能的模块

import torch  # 导入PyTorch深度学习库
import torch.distributed as dist  # 导入PyTorch分布式训练模块
import torch.nn.functional as F  # 导入PyTorch函数库


def parse_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="test script")

    # 添加命令行参数选项
    parser.add_argument(
        "--init-method",
        "--init_method",
        type=str,
        required=True,
        help="init_method to pass to `dist.init_process_group()` (e.g. env://)",
    )
    parser.add_argument(
        "--world-size",
        "--world_size",
        type=int,
        default=os.getenv("WORLD_SIZE", -1),
        help="world_size to pass to `dist.init_process_group()`",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=os.getenv("RANK", -1),
        help="rank to pass to `dist.init_process_group()`",
    )

    # 解析命令行参数并返回
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    # 初始化进程组，使用gloo后端
    dist.init_process_group(
        backend="gloo",
        init_method=args.init_method,
        world_size=args.world_size,
        rank=args.rank,
    )

    # 获取当前进程的rank和进程组的总数（world_size）
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 创建一个大小为world_size的one-hot张量，根据当前进程的rank填充
    t = F.one_hot(torch.tensor(rank), num_classes=world_size)

    # 对所有进程的张量t进行全局归约操作
    dist.all_reduce(t)

    # 计算归约后张量t中元素的总和，应该等于world_size
    derived_world_size = torch.sum(t).item()

    # 检查计算得到的world_size是否与预期的world_size相等，如果不相等则抛出异常
    if derived_world_size != world_size:
        raise RuntimeError(
            f"Wrong world size derived. Expected: {world_size}, Got: {derived_world_size}"
        )

    # 打印完成消息
    print("Done")


if __name__ == "__main__":
    main()
```
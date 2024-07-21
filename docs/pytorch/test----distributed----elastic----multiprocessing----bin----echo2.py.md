# `.\pytorch\test\distributed\elastic\multiprocessing\bin\echo2.py`

```py
#!/usr/bin/env python3

# 导入必要的模块 argparse 和 os

import argparse  # 导入命令行参数解析模块
import os        # 导入操作系统接口模块

# 如果当前脚本是直接执行的主程序，则执行以下代码块
if __name__ == "__main__":
    # 创建一个 ArgumentParser 对象，用于解析命令行参数并生成帮助信息
    parser = argparse.ArgumentParser(description="test binary, raises a RuntimeError")
    
    # 添加命令行参数 --raises，类型为布尔值，默认为 False
    parser.add_argument("--raises", type=bool, default=False)
    
    # 添加位置参数 msg，类型为字符串
    parser.add_argument("msg", type=str)
    
    # 解析命令行参数，将解析结果存储在 args 对象中
    args = parser.parse_args()

    # 从环境变量 RANK 中获取 RANK 的值，并转换为整数
    rank = int(os.environ["RANK"])

    # 如果命令行参数 --raises 被设置为 True，则抛出 RuntimeError 异常
    if args.raises:
        raise RuntimeError(f"raised from {rank}")
    else:
        # 否则，打印命令行参数 msg 和当前的 rank 值
        print(f"{args.msg} from {rank}")
```
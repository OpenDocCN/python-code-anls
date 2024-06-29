# `D:\src\scipysrc\pandas\scripts\run_vulture.py`

```
"""Look for unreachable code."""

# 导入 argparse 库，用于命令行参数解析
import argparse
# 导入 sys 库，用于处理系统相关操作
import sys
# 从 vulture 库中导入 Vulture 类，用于检测未使用的代码
from vulture import Vulture

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser()
    # 添加位置参数 files，表示要分析的文件列表，可选多个
    parser.add_argument("files", nargs="*")
    # 解析命令行参数，存储到 args 变量中
    args = parser.parse_args()

    # 创建 Vulture 对象
    v = Vulture()
    # 对给定的文件列表进行代码扫描
    v.scavenge(args.files)
    # 初始化返回值为 0
    ret = 0
    # 遍历未使用代码的检测结果
    for item in v.get_unused_code(min_confidence=100):
        # 如果发现未使用的不可达代码
        if item.typ == "unreachable_code":
            # 打印不可达代码的报告信息
            print(item.get_report())
            # 将返回值设置为 1，表示发现了不可达代码
            ret = 1

    # 退出程序，返回 ret 作为退出码
    sys.exit(ret)
```
# `.\pytorch\tools\stats\export_test_times.py`

```
import sys  # 导入系统模块sys，用于管理 Python 解释器的运行时环境
from pathlib import Path  # 导入路径操作相关模块Path，用于处理文件系统路径

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # 获取当前脚本文件的父目录的父目录的父目录作为项目根目录
sys.path.append(str(REPO_ROOT))  # 将项目根目录路径添加到系统路径中，使得可以在此路径下导入自定义模块

from tools.stats.import_test_stats import get_test_class_times, get_test_times
# 从自定义模块tools.stats.import_test_stats中导入函数get_test_class_times和get_test_times

def main() -> None:
    print("Exporting test times from test-infra")  # 打印消息，指示正在导出测试时间信息
    get_test_times()  # 调用导入的get_test_times函数，执行获取测试时间的操作
    get_test_class_times()  # 调用导入的get_test_class_times函数，执行获取测试类时间的操作

if __name__ == "__main__":
    main()  # 如果当前脚本作为主程序运行，则调用main函数开始执行程序逻辑
```
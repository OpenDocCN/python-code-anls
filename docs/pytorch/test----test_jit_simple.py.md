# `.\pytorch\test\test_jit_simple.py`

```py
# Owner(s): ["oncall: jit"]
# 导入系统模块 sys，用于处理命令行参数
import sys
# 向 sys.argv 列表中添加一个参数 "--jit-executor=simple"
sys.argv.append("--jit-executor=simple")
# 导入 test_jit 模块中的所有内容，禁止 F403 错误提示
from test_jit import *  # noqa: F403

# 如果当前脚本作为主程序执行
if __name__ == '__main__':
    # 运行 test_jit 模块中的测试函数
    run_tests()
```
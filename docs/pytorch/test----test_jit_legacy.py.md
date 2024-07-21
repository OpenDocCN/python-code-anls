# `.\pytorch\test\test_jit_legacy.py`

```
# Owner(s): ["oncall: jit"]
# 导入 sys 模块，用于访问系统相关功能
import sys
# 向 sys.argv 列表追加一个参数字符串 "--jit-executor=legacy"
sys.argv.append("--jit-executor=legacy")
# 导入 test_jit 模块中的所有内容，禁止 Flake8 检查 F403
from test_jit import *  # noqa: F403

# 如果当前脚本作为主程序执行
if __name__ == '__main__':
    # 调用 test_jit 模块中的 run_tests 函数，执行测试
    run_tests()
```
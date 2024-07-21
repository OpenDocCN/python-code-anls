# `.\pytorch\test\test_jit_fuser_legacy.py`

```
# 导入 sys 模块，用于处理系统相关功能
import sys

# 向命令行参数列表中添加选项 "--jit-executor=legacy"
sys.argv.append("--jit-executor=legacy")

# 从 test_jit_fuser 模块中导入所有内容，禁止 Flake8 检查时引发的 F403 警告
from test_jit_fuser import *  # noqa: F403

# 如果当前脚本作为主程序运行，则执行 run_tests 函数
if __name__ == '__main__':
    run_tests()
```
# `.\pytorch\torch\utils\model_dump\__main__.py`

```
#!/usr/bin/env python3
# 使用 /usr/bin/env python3 命令找到 Python 解释器并执行脚本

import sys
# 导入 sys 模块，用于访问系统相关功能

from . import main
# 从当前包中导入 main 模块（或者相对路径中的 main.py 文件）

sys.exit(main(sys.argv))
# 调用 main 模块的主函数 main，并传入当前脚本的命令行参数 sys.argv
# 使用 sys.exit 来终止当前脚本的执行，返回 main 函数的返回值作为退出状态码
```
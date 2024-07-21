# `.\pytorch\tools\testing\target_determination\gen_artifact.py`

```py
# 引入未来版本的注解特性，用于函数签名的类型提示
from __future__ import annotations

# 导入处理 JSON 的模块
import json
# 导入操作系统相关的功能
import os
# 导入处理路径的模块
from pathlib import Path
# 导入类型提示相关的工具
from typing import Any

# 定义全局变量，指向代码文件的上四级目录的根路径
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# 定义生成持续集成构建工件的函数，参数为包含和排除列表
def gen_ci_artifact(included: list[Any], excluded: list[Any]) -> None:
    # 创建一个随机生成的文件名，格式为 'td_exclusions-{随机10个字节的十六进制字符串}.json'
    file_name = f"td_exclusions-{os.urandom(10).hex()}.json"
    # 打开文件以写入 JSON 数据，路径为根路径下的 'test/test-reports' 目录下的生成的文件名
    with open(REPO_ROOT / "test" / "test-reports" / file_name, "w") as f:
        # 将包含和排除列表以字典形式写入 JSON 文件
        json.dump({"included": included, "excluded": excluded}, f)
```
# `.\pytorch\tools\setup_helpers\__init__.py`

```py
# 引入未来的注释语法支持
from __future__ import annotations

# 引入操作系统相关的功能模块
import os
# 引入系统相关的功能模块
import sys

# 定义一个函数，用于查找指定文件在系统路径中的位置
def which(thefile: str) -> str | None:
    # 获取当前操作系统环境中的 PATH 变量，若不存在则使用默认路径
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    
    # 遍历 PATH 中的每一个路径
    for d in path:
        # 将文件名与当前路径拼接成完整的文件路径
        fname = os.path.join(d, thefile)
        
        # 将拼接后的文件名添加到列表中
        fnames = [fname]
        
        # 如果当前操作系统是 Windows
        if sys.platform == "win32":
            # 获取环境变量 PATHEXT 中的可执行文件扩展名列表
            exts = os.environ.get("PATHEXT", "").split(os.pathsep)
            
            # 将每个可执行文件扩展名与文件名结合，形成完整的文件路径列表
            fnames += [fname + ext for ext in exts]
        
        # 遍历所有可能的文件路径
        for name in fnames:
            # 检查文件是否存在且可执行，且不是一个目录
            if os.access(name, os.F_OK | os.X_OK) and not os.path.isdir(name):
                # 如果找到符合条件的文件路径，则返回该路径
                return name
    
    # 如果在所有路径中都未找到符合条件的文件，则返回 None
    return None
```
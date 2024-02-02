# `arknights-mower\arknights_mower\__init__.py`

```py
# 导入 platform 模块
import platform
# 导入 sys 模块
import sys
# 从 pathlib 模块中导入 Path 类
from pathlib import Path

# 使用 sys.frozen 检查是否通过 pyinstaller 冻结的可执行文件运行，并使用 sys._MEIPASS 获取临时路径
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # 如果是通过 pyinstaller 冻结的可执行文件运行，则设置 __pyinstall__ 为 True
    __pyinstall__ = True
    # 为什么他们在这里创建一个 __init__ 文件夹...我不知道。
    # 使用 Path 类创建 __rootdir__ 路径，指向 sys._MEIPASS 下的 'arknights_mower' 下的 '__init__' 文件夹
    __rootdir__ = Path(sys._MEIPASS).joinpath('arknights_mower').joinpath('__init__').resolve()
else:
    # 如果不是通过 pyinstaller 冻结的可执行文件运行，则设置 __pyinstall__ 为 False
    __pyinstall__ = False
    # 使用 Path 类创建 __rootdir__ 路径，指向当前文件的父目录
    __rootdir__ = Path(__file__).parent.resolve()

# 命令行模式
# 如果 __pyinstall__ 为 True 且 sys.argv[1:] 为空，则设置 __cli__ 为 False；否则设置为 True
__cli__ = not (__pyinstall__ and not sys.argv[1:])

# 获取操作系统名称并转换为小写
__system__ = platform.system().lower()
# 设置 __version__ 为 'v3.4.3'
__version__ = 'v3.4.3'
```
# `.\Zelda-with-Python\Code\Player.py`

```

# 导入 pygame 模块
import pygame
# 从 Support 模块中导入 import_folder 函数
from Support import import_folder
# 从 Settings 模块中导入所有内容
from Settings import *
# 从 Entity 模块中导入 Entity 类
from Entity import Entity
# 导入 os 和 sys 模块
import os, sys

# 将当前工作目录更改为 Main.py 文件所在的目录
# 这是为了在 Main.py 中进行文件导入时的路径设置
os.chdir(os.path.dirname(os.path.abspath(__file__)))

```
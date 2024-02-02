# `MetaGPT\metagpt\utils\git_repository.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/11/20
@Author  : mashenquan
@File    : git_repository.py
@Desc: Git repository management. RFC 135 2.2.3.3.
"""
# 导入必要的模块
from __future__ import annotations  # 导入未来版本的注解特性

import shutil  # 导入 shutil 模块，用于文件操作
from enum import Enum  # 导入 Enum 枚举类型
from pathlib import Path  # 导入 Path 类，用于处理文件路径
from typing import Dict, List  # 导入类型提示

from git.repo import Repo  # 导入 Repo 类，用于操作 Git 仓库
from git.repo.fun import is_git_dir  # 导入 is_git_dir 函数，用于检查是否为 Git 仓库
from gitignore_parser import parse_gitignore  # 导入 parse_gitignore 函数，用于解析 .gitignore 文件

from metagpt.logs import logger  # 导入日志模块
from metagpt.utils.dependency_file import DependencyFile  # 导入依赖文件模块
from metagpt.utils.file_repository import FileRepository  # 导入文件仓库模块

# 定义 ChangeType 枚举类型，表示文件变更类型
class ChangeType(Enum):
    ADDED = "A"  # 文件被添加
    COPIED = "C"  # 文件被复制
    DELETED = "D"  # 文件被删除
    RENAMED = "R"  # 文件被重命名
    MODIFIED = "M"  # 文件被修改
    TYPE_CHANGED = "T"  # 文件类型被改变
    UNTRACTED = "U"  # 文件未被跟踪（未添加到版本控制）

```
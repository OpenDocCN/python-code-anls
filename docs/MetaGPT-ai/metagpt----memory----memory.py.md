# `MetaGPT\metagpt\memory\memory.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/20 12:15
@Author  : alexanderwu
@File    : memory.py
@Modified By: mashenquan, 2023-11-1. According to RFC 116: Updated the type of index key.
"""
# 导入所需的模块和类
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable, Set
# 导入 pydantic 模块中的 BaseModel、Field 和 SerializeAsAny 类
from pydantic import BaseModel, Field, SerializeAsAny
# 导入自定义模块中的常量和类
from metagpt.const import IGNORED_MESSAGE_ID
from metagpt.schema import Message
from metagpt.utils.common import (
    any_to_str,
    any_to_str_set,
    read_json_file,
    write_json_file,
)

```
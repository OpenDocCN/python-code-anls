# `MetaGPT\tests\metagpt\utils\test_common.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/29 16:19
@Author  : alexanderwu
@File    : test_common.py
@Modified by: mashenquan, 2023/11/21. Add unit tests.
"""
# 导入所需的模块
import importlib
import os
import platform
import uuid
from pathlib import Path
from typing import Any, Set

import aiofiles
import pytest
from pydantic import BaseModel

# 导入自定义模块
from metagpt.actions import RunCode
from metagpt.const import get_metagpt_root
from metagpt.roles.tutorial_assistant import TutorialAssistant
from metagpt.schema import Message
from metagpt.utils.common import (
    NoMoneyException,
    OutputParser,
    any_to_str,
    any_to_str_set,
    aread,
    awrite,
    check_cmd_exists,
    concat_namespace,
    import_class_inst,
    parse_recipient,
    print_members,
    read_file_block,
    read_json_file,
    require_python_version,
    split_namespace,
)

# 如果当前脚本为主程序，则执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```
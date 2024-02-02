# `MetaGPT\metagpt\environment.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 22:12
@Author  : alexanderwu
@File    : environment.py
@Modified By: mashenquan, 2023-11-1. According to Chapter 2.2.2 of RFC 116:
    1. Remove the functionality of `Environment` class as a public message buffer.
    2. Standardize the message forwarding behavior of the `Environment` class.
    3. Add the `is_idle` property.
@Modified By: mashenquan, 2023-11-4. According to the routing feature plan in Chapter 2.2.3.2 of RFC 113, the routing
    functionality is to be consolidated into the `Environment` class.
"""
# 导入必要的模块
import asyncio
from pathlib import Path
from typing import Iterable, Set

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator
# 导入自定义模块
from metagpt.config import CONFIG
from metagpt.logs import logger
from metagpt.roles.role import Role
from metagpt.schema import Message
from metagpt.utils.common import is_subscribed, read_json_file, write_json_file

```
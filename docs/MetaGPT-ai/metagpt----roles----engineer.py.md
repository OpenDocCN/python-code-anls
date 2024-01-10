# `MetaGPT\metagpt\roles\engineer.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : engineer.py
@Modified By: mashenquan, 2023-11-1. In accordance with Chapter 2.2.1 and 2.2.2 of RFC 116:
    1. Modify the data type of the `cause_by` value in the `Message` to a string, and utilize the new message
        distribution feature for message filtering.
    2. Consolidate message reception and processing logic within `_observe`.
    3. Fix bug: Add logic for handling asynchronous message processing when messages are not ready.
    4. Supplemented the external transmission of internal messages.
@Modified By: mashenquan, 2023-11-27.
    1. According to Section 2.2.3.1 of RFC 135, replace file data in the message with the file name.
    2. According to the design in Section 2.2.3.5.5 of RFC 135, add incremental iteration functionality.
@Modified By: mashenquan, 2023-12-5. Enhance the workflow to navigate to WriteCode or QaEngineer based on the results
    of SummarizeCode.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Set

from metagpt.actions import Action, WriteCode, WriteCodeReview, WriteTasks
from metagpt.actions.fix_bug import FixBug
from metagpt.actions.summarize_code import SummarizeCode
from metagpt.config import CONFIG
from metagpt.const import (
    CODE_SUMMARIES_FILE_REPO,
    CODE_SUMMARIES_PDF_FILE_REPO,
    SYSTEM_DESIGN_FILE_REPO,
    TASK_FILE_REPO,
)
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import (
    CodeSummarizeContext,
    CodingContext,
    Document,
    Documents,
    Message,
)
from metagpt.utils.common import any_to_name, any_to_str, any_to_str_set

IS_PASS_PROMPT = """
{context}

----
Does the above log indicate anything that needs to be done?
If there are any tasks to be completed, please answer 'NO' along with the to-do list in JSON format;
otherwise, answer 'YES' in JSON format.
"""


注释：
- 第一行指定了脚本的解释器路径，使得脚本可以直接执行
- 第二行指定了脚本的编码格式为utf-8
- 代码文件的基本信息，包括作者、修改时间等
- 通过@Modified By注释记录了对代码的修改，包括修改时间、修改人和具体修改内容
- 导入了需要使用的模块和类
- 定义了一个多行字符串常量IS_PASS_PROMPT，用于提示用户是否有需要完成的任务，并提供相应的格式要求。
```
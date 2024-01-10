# `MetaGPT\metagpt\roles\qa_engineer.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : qa_engineer.py
@Modified By: mashenquan, 2023-11-1. In accordance with Chapter 2.2.1 and 2.2.2 of RFC 116, modify the data
        type of the `cause_by` value in the `Message` to a string, and utilize the new message filtering feature.
@Modified By: mashenquan, 2023-11-27.
        1. Following the think-act principle, solidify the task parameters when creating the
        WriteTest/RunCode/DebugError object, rather than passing them in when calling the run function.
        2. According to Section 2.2.3.5.7 of RFC 135, change the method of transferring files from using the Message
        to using file references.
@Modified By: mashenquan, 2023-12-5. Enhance the workflow to navigate to WriteCode or QaEngineer based on the results
    of SummarizeCode.
"""

# 导入所需模块
from metagpt.actions import DebugError, RunCode, WriteTest
from metagpt.actions.summarize_code import SummarizeCode
from metagpt.config import CONFIG
from metagpt.const import (
    MESSAGE_ROUTE_TO_NONE,
    TEST_CODES_FILE_REPO,
    TEST_OUTPUTS_FILE_REPO,
)
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Document, Message, RunCodeContext, TestingContext
from metagpt.utils.common import any_to_str_set, parse_recipient
from metagpt.utils.file_repository import FileRepository

```
# `MetaGPT\metagpt\actions\debug_error.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:46
@Author  : alexanderwu
@File    : debug_error.py
@Modified By: mashenquan, 2023/11/27.
        1. Divide the context into three components: legacy code, unit test code, and console log.
        2. According to Section 2.2.3.1 of RFC 135, replace file data in the message with the file name.
"""
import re

from pydantic import Field

from metagpt.actions.action import Action
from metagpt.config import CONFIG
from metagpt.const import TEST_CODES_FILE_REPO, TEST_OUTPUTS_FILE_REPO
from metagpt.logs import logger
from metagpt.schema import RunCodeContext, RunCodeResult
from metagpt.utils.common import CodeParser
from metagpt.utils.file_repository import FileRepository

# 定义一个模板字符串，用于提示用户根据不同角色和错误信息重写代码
PROMPT_TEMPLATE = """
NOTICE
1. Role: You are a Development Engineer or QA engineer;
2. Task: You received this message from another Development Engineer or QA engineer who ran or tested your code. 
Based on the message, first, figure out your own role, i.e. Engineer or QaEngineer,
then rewrite the development code or the test code based on your role, the error, and the summary, such that all bugs are fixed and the code performs well.
Attention: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the test case or script and triple quotes.
The message is as follows:
# Legacy Code

{code}

---
# Unit Test Code

{test_code}

---
# Console logs

{logs}

---
Now you should start rewriting the code:
## file name of the code to rewrite: Write code with triple quote. Do your best to implement THIS IN ONLY ONE FILE.
"""

# 定义一个名为DebugError的类，继承自Action类
class DebugError(Action):
    name: str = "DebugError"
    context: RunCodeContext = Field(default_factory=RunCodeContext)

    # 异步运行方法，接收参数并返回字符串
    async def run(self, *args, **kwargs) -> str:
        # 从文件仓库获取输出文件内容
        output_doc = await FileRepository.get_file(
            filename=self.context.output_filename, relative_path=TEST_OUTPUTS_FILE_REPO
        )
        if not output_doc:
            return ""
        output_detail = RunCodeResult.loads(output_doc.content)
        pattern = r"Ran (\d+) tests in ([\d.]+)s\n\nOK"
        matches = re.search(pattern, output_detail.stderr)
        if matches:
            return ""

        # 记录日志并调试重写代码
        logger.info(f"Debug and rewrite {self.context.test_filename}")
        code_doc = await FileRepository.get_file(
            filename=self.context.code_filename, relative_path=CONFIG.src_workspace
        )
        if not code_doc:
            return ""
        test_doc = await FileRepository.get_file(
            filename=self.context.test_filename, relative_path=TEST_CODES_FILE_REPO
        )
        if not test_doc:
            return ""
        # 根据模板字符串格式化提示信息
        prompt = PROMPT_TEMPLATE.format(code=code_doc.content, test_code=test_doc.content, logs=output_detail.stderr)

        # 调用_aask方法，获取用户输入的代码
        rsp = await self._aask(prompt)
        code = CodeParser.parse_code(block="", text=rsp)

        return code

```
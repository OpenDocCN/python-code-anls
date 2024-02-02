# `MetaGPT\metagpt\actions\summarize_code.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : alexanderwu
@File    : summarize_code.py
@Modified By: mashenquan, 2023/12/5. Archive the summarization content of issue discovery for use in WriteCode.
"""
# 导入模块
from pathlib import Path
# 导入模块
from pydantic import Field
# 导入模块
from tenacity import retry, stop_after_attempt, wait_random_exponential
# 导入模块
from metagpt.actions.action import Action
# 导入模块
from metagpt.config import CONFIG
# 导入模块
from metagpt.const import SYSTEM_DESIGN_FILE_REPO, TASK_FILE_REPO
# 导入模块
from metagpt.logs import logger
# 导入模块
from metagpt.schema import CodeSummarizeContext
# 导入模块
from metagpt.utils.file_repository import FileRepository

# 定义常量
PROMPT_TEMPLATE = """
...
"""

# 定义常量
FORMAT_EXAMPLE = """
...
"""

# 定义类
class SummarizeCode(Action):
    # 定义属性
    name: str = "SummarizeCode"
    # 定义属性
    context: CodeSummarizeContext = Field(default_factory=CodeSummarizeContext)

    # 定义异步函数
    @retry(stop=stop_after_attempt(2), wait=wait_random_exponential(min=1, max=60))
    async def summarize_code(self, prompt):
        code_rsp = await self._aask(prompt)
        return code_rsp

    # 定义异步函数
    async def run(self):
        # 获取系统设计文档
        design_pathname = Path(self.context.design_filename)
        design_doc = await FileRepository.get_file(filename=design_pathname.name, relative_path=SYSTEM_DESIGN_FILE_REPO)
        # 获取任务文档
        task_pathname = Path(self.context.task_filename)
        task_doc = await FileRepository.get_file(filename=task_pathname.name, relative_path=TASK_FILE_REPO)
        # 获取源代码文件仓库
        src_file_repo = CONFIG.git_repo.new_file_repository(relative_path=CONFIG.src_workspace)
        code_blocks = []
        # 遍历代码文件名列表
        for filename in self.context.codes_filenames:
            # 获取代码文档
            code_doc = await src_file_repo.get(filename)
            # 构建代码块
            code_block = f"```python\n{code_doc.content}\n```\n-----"
            code_blocks.append(code_block)
        format_example = FORMAT_EXAMPLE
        # 构建提示信息
        prompt = PROMPT_TEMPLATE.format(
            system_design=design_doc.content,
            tasks=task_doc.content,
            code_blocks="\n".join(code_blocks),
            format_example=format_example,
        )
        logger.info("Summarize code..")
        # 总结代码
        rsp = await self.summarize_code(prompt)
        return rsp

```
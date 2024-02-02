# `MetaGPT\metagpt\actions\write_test.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 22:12
@Author  : alexanderwu
@File    : write_test.py
@Modified By: mashenquan, 2023-11-27. Following the think-act principle, solidify the task parameters when creating the
        WriteTest object, rather than passing them in when calling the run function.
"""

from typing import Optional

from metagpt.actions.action import Action  # 导入Action类
from metagpt.const import TEST_CODES_FILE_REPO  # 导入测试代码文件存储库的常量
from metagpt.logs import logger  # 导入日志记录器
from metagpt.schema import Document, TestingContext  # 导入文档和测试上下文的模式
from metagpt.utils.common import CodeParser  # 导入代码解析器

PROMPT_TEMPLATE = """
...  # 省略部分内容
"""

class WriteTest(Action):
    name: str = "WriteTest"  # 定义动作名称为WriteTest
    context: Optional[TestingContext] = None  # 定义上下文为可选的测试上下文对象

    async def write_code(self, prompt):
        code_rsp = await self._aask(prompt)  # 调用私有方法_aask，传入提示语句，获取代码响应

        try:
            code = CodeParser.parse_code(block="", text=code_rsp)  # 使用代码解析器解析代码响应
        except Exception:
            # 处理异常
            logger.error(f"Can't parse the code: {code_rsp}")  # 记录错误日志

            # 在发生异常的情况下返回code_rsp，假设llm只返回原始代码，不在```中包装它
            code = code_rsp  # 将code设置为code_rsp
        return code  # 返回解析后的代码

    async def run(self, *args, **kwargs) -> TestingContext:
        if not self.context.test_doc:  # 如果测试文档不存在
            self.context.test_doc = Document(  # 创建测试文档对象
                filename="test_" + self.context.code_doc.filename, root_path=TEST_CODES_FILE_REPO
            )
        fake_root = "/data"  # 设置虚拟根目录
        prompt = PROMPT_TEMPLATE.format(  # 根据模板生成提示语句
            code_to_test=self.context.code_doc.content,  # 传入待测试的代码
            test_file_name=self.context.test_doc.filename,  # 传入测试文档的文件名
            source_file_path=fake_root + "/" + self.context.code_doc.root_relative_path,  # 传入源文件路径
            workspace=fake_root,  # 传入工作空间路径
        )
        self.context.test_doc.content = await self.write_code(prompt)  # 调用write_code方法，传入提示语句，获取测试文档内容
        return self.context  # 返回测试上下文对象

```
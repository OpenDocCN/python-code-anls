# `MetaGPT\metagpt\actions\project_management.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 19:12
@Author  : alexanderwu
@File    : project_management.py
@Modified By: mashenquan, 2023/11/27.
        1. Divide the context into three components: legacy code, unit test code, and console log.
        2. Move the document storage operations related to WritePRD from the save operation of WriteDesign.
        3. According to the design in Section 2.2.3.5.4 of RFC 135, add incremental iteration functionality.
"""

import json
from typing import Optional

from metagpt.actions import ActionOutput
from metagpt.actions.action import Action
from metagpt.actions.project_management_an import PM_NODE
from metagpt.config import CONFIG
from metagpt.const import (
    PACKAGE_REQUIREMENTS_FILENAME,
    SYSTEM_DESIGN_FILE_REPO,
    TASK_FILE_REPO,
    TASK_PDF_FILE_REPO,
)
from metagpt.logs import logger
from metagpt.schema import Document, Documents
from metagpt.utils.file_repository import FileRepository

# 定义一个模板，用于生成新的需求文档
NEW_REQ_TEMPLATE = """
### Legacy Content
{old_tasks}

### New Requirements
{context}
"""

# 定义一个名为WriteTasks的类，继承自Action类
class WriteTasks(Action):
    name: str = "CreateTasks"
    context: Optional[str] = None

    # 异步运行方法
    async def run(self, with_messages, schema=CONFIG.prompt_schema):
        # 创建系统设计文件仓库
        system_design_file_repo = CONFIG.git_repo.new_file_repository(SYSTEM_DESIGN_FILE_REPO)
        # 获取已更改的系统设计文件
        changed_system_designs = system_design_file_repo.changed_files

        # 创建任务文件仓库
        tasks_file_repo = CONFIG.git_repo.new_file_repository(TASK_FILE_REPO)
        # 获取已更改的任务文件
        changed_tasks = tasks_file_repo.changed_files
        change_files = Documents()

        # 根据git head diff重写已更改的系统设计文件
        for filename in changed_system_designs:
            task_doc = await self._update_tasks(
                filename=filename, system_design_file_repo=system_design_file_repo, tasks_file_repo=tasks_file_repo
            )
            change_files.docs[filename] = task_doc

        # 根据git head diff重写已更改的任务文件
        for filename in changed_tasks:
            if filename in change_files.docs:
                continue
            task_doc = await self._update_tasks(
                filename=filename, system_design_file_repo=system_design_file_repo, tasks_file_repo=tasks_file_repo
            )
            change_files.docs[filename] = task_doc

        if not change_files.docs:
            logger.info("Nothing has changed.")
        # 在发送publish_message之前等待所有任务文件都被处理，为后续步骤中的全局优化留出空间
        return ActionOutput(content=change_files.model_dump_json(), instruct_content=change_files)

    # 更新任务方法
    async def _update_tasks(self, filename, system_design_file_repo, tasks_file_repo):
        system_design_doc = await system_design_file_repo.get(filename)
        task_doc = await tasks_file_repo.get(filename)
        if task_doc:
            task_doc = await self._merge(system_design_doc=system_design_doc, task_doc=task_doc)
        else:
            rsp = await self._run_new_tasks(context=system_design_doc.content)
            task_doc = Document(
                root_path=TASK_FILE_REPO, filename=filename, content=rsp.instruct_content.model_dump_json()
            )
        await tasks_file_repo.save(
            filename=filename, content=task_doc.content, dependencies={system_design_doc.root_relative_path}
        )
        await self._update_requirements(task_doc)
        await self._save_pdf(task_doc=task_doc)
        return task_doc

    # 运行新任务方法
    async def _run_new_tasks(self, context, schema=CONFIG.prompt_schema):
        node = await PM_NODE.fill(context, self.llm, schema)
        return node

    # 合并方法
    async def _merge(self, system_design_doc, task_doc, schema=CONFIG.prompt_schema) -> Document:
        context = NEW_REQ_TEMPLATE.format(context=system_design_doc.content, old_tasks=task_doc.content)
        node = await PM_NODE.fill(context, self.llm, schema)
        task_doc.content = node.instruct_content.model_dump_json()
        return task_doc

    # 更新需求方法
    @staticmethod
    async def _update_requirements(doc):
        m = json.loads(doc.content)
        packages = set(m.get("Required Python third-party packages", set()))
        file_repo = CONFIG.git_repo.new_file_repository()
        requirement_doc = await file_repo.get(filename=PACKAGE_REQUIREMENTS_FILENAME)
        if not requirement_doc:
            requirement_doc = Document(filename=PACKAGE_REQUIREMENTS_FILENAME, root_path=".", content="")
        lines = requirement_doc.content.splitlines()
        for pkg in lines:
            if pkg == "":
                continue
            packages.add(pkg)
        await file_repo.save(PACKAGE_REQUIREMENTS_FILENAME, content="\n".join(packages))

    # 保存PDF方法
    @staticmethod
    async def _save_pdf(task_doc):
        await FileRepository.save_as(doc=task_doc, with_suffix=".md", relative_path=TASK_PDF_FILE_REPO)

```
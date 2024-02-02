# `MetaGPT\metagpt\actions\prepare_documents.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/11/20
@Author  : mashenquan
@File    : prepare_documents.py
@Desc: PrepareDocuments Action: initialize project folder and add new requirements to docs/requirements.txt.
        RFC 135 2.2.3.5.1.
"""
# 导入所需的模块
import shutil
from pathlib import Path
from typing import Optional

# 导入自定义模块
from metagpt.actions import Action, ActionOutput
from metagpt.config import CONFIG
from metagpt.const import DOCS_FILE_REPO, REQUIREMENT_FILENAME
from metagpt.schema import Document
from metagpt.utils.file_repository import FileRepository
from metagpt.utils.git_repository import GitRepository

# 定义一个名为PrepareDocuments的类，继承自Action类
class PrepareDocuments(Action):
    """PrepareDocuments Action: initialize project folder and add new requirements to docs/requirements.txt."""

    # 类属性
    name: str = "PrepareDocuments"
    context: Optional[str] = None

    # 初始化Git环境的私有方法
    def _init_repo(self):
        """Initialize the Git environment."""
        # 如果项目路径不存在，则根据配置创建新的项目路径
        if not CONFIG.project_path:
            name = CONFIG.project_name or FileRepository.new_filename()
            path = Path(CONFIG.workspace_path) / name
        else:
            path = Path(CONFIG.project_path)
        # 如果路径存在且不是增量模式，则删除路径
        if path.exists() and not CONFIG.inc:
            shutil.rmtree(path)
        # 设置项目路径和Git仓库
        CONFIG.project_path = path
        CONFIG.git_repo = GitRepository(local_path=path, auto_init=True)

    # 异步运行方法
    async def run(self, with_messages, **kwargs):
        """Create and initialize the workspace folder, initialize the Git environment."""
        # 调用初始化Git环境的方法
        self._init_repo()

        # 将主参数idea中新增的需求写入`docs/requirement.txt`
        doc = Document(root_path=DOCS_FILE_REPO, filename=REQUIREMENT_FILENAME, content=with_messages[0].content)
        await FileRepository.save_file(filename=REQUIREMENT_FILENAME, content=doc.content, relative_path=DOCS_FILE_REPO)

        # 发送消息通知WritePRD操作，指示其使用`docs/requirement.txt`和`docs/prds/`处理需求
        return ActionOutput(content=doc.content, instruct_content=doc)

```
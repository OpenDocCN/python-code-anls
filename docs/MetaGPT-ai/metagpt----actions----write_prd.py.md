# `MetaGPT\metagpt\actions\write_prd.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:45
@Author  : alexanderwu
@File    : write_prd.py
@Modified By: mashenquan, 2023/11/27.
            1. According to Section 2.2.3.1 of RFC 135, replace file data in the message with the file name.
            2. According to the design in Section 2.2.3.5.2 of RFC 135, add incremental iteration functionality.
            3. Move the document storage operations related to WritePRD from the save operation of WriteDesign.
@Modified By: mashenquan, 2023/12/5. Move the generation logic of the project name to WritePRD.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from metagpt.actions import Action, ActionOutput
from metagpt.actions.action_node import ActionNode
from metagpt.actions.fix_bug import FixBug
from metagpt.actions.write_prd_an import (
    PROJECT_NAME,
    WP_IS_RELATIVE_NODE,
    WP_ISSUE_TYPE_NODE,
    WRITE_PRD_NODE,
)
from metagpt.config import CONFIG
from metagpt.const import (
    BUGFIX_FILENAME,
    COMPETITIVE_ANALYSIS_FILE_REPO,
    DOCS_FILE_REPO,
    PRD_PDF_FILE_REPO,
    PRDS_FILE_REPO,
    REQUIREMENT_FILENAME,
)
from metagpt.logs import logger
from metagpt.schema import BugFixContext, Document, Documents, Message
from metagpt.utils.common import CodeParser
from metagpt.utils.file_repository import FileRepository
from metagpt.utils.mermaid import mermaid_to_file

# 定义上下文模板
CONTEXT_TEMPLATE = """
### Project Name
{project_name}

### Original Requirements
{requirements}

### Search Information
-
"""

# 定义新需求模板
NEW_REQ_TEMPLATE = """
### Legacy Content
{old_prd}

### New Requirements
{requirements}
"""

```
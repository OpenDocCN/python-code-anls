# `MetaGPT\metagpt\actions\design_api.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 19:26
@Author  : alexanderwu
@File    : design_api.py
@Modified By: mashenquan, 2023/11/27.
            1. According to Section 2.2.3.1 of RFC 135, replace file data in the message with the file name.
            2. According to the design in Section 2.2.3.5.3 of RFC 135, add incremental iteration functionality.
@Modified By: mashenquan, 2023/12/5. Move the generation logic of the project name to WritePRD.
"""
# 导入所需的模块
import json
from pathlib import Path
from typing import Optional

# 导入自定义模块
from metagpt.actions import Action, ActionOutput
from metagpt.actions.design_api_an import DESIGN_API_NODE
from metagpt.config import CONFIG
from metagpt.const import (
    DATA_API_DESIGN_FILE_REPO,
    PRDS_FILE_REPO,
    SEQ_FLOW_FILE_REPO,
    SYSTEM_DESIGN_FILE_REPO,
    SYSTEM_DESIGN_PDF_FILE_REPO,
)
from metagpt.logs import logger
from metagpt.schema import Document, Documents, Message
from metagpt.utils.file_repository import FileRepository
from metagpt.utils.mermaid import mermaid_to_file

# 定义模板字符串
NEW_REQ_TEMPLATE = """
### Legacy Content
{old_design}

### New Requirements
{context}
"""

```
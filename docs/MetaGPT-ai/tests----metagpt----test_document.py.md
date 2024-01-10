# `MetaGPT\tests\metagpt\test_document.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/2 21:00
@Author  : alexanderwu
@File    : test_document.py
"""
# 导入所需的模块
from metagpt.config import CONFIG
from metagpt.document import Repo
from metagpt.logs import logger

# 定义函数，设置现有的仓库
def set_existing_repo(path):
    # 从指定路径创建仓库对象
    repo1 = Repo.from_path(path)
    # 设置仓库中的文件内容
    repo1.set("doc/wtf_file.md", "wtf content")
    repo1.set("code/wtf_file.py", "def hello():\n    print('hello')")
    # 记录仓库信息
    logger.info(repo1)  # check doc

# 定义函数，加载现有的仓库
def load_existing_repo(path):
    # 从指定路径创建仓库对象
    repo = Repo.from_path(path)
    # 记录仓库信息
    logger.info(repo)
    # 记录仓库的详细信息
    logger.info(repo.eda())

    # 断言仓库对象存在
    assert repo
    # 断言获取的文件内容符合预期
    assert repo.get("doc/wtf_file.md").content == "wtf content"
    assert repo.get("code/wtf_file.py").content == "def hello():\n    print('hello')"

# 定义测试函数，测试仓库的设置和加载
def test_repo_set_load():
    # 设置仓库路径
    repo_path = CONFIG.workspace_path / "test_repo"
    # 设置现有的仓库
    set_existing_repo(repo_path)
    # 加载现有的仓库
    load_existing_repo(repo_path)

```
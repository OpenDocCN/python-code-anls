# `MetaGPT\tests\metagpt\actions\mock_markdown.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/18 23:51
@Author  : alexanderwu
@File    : mock_markdown.py
"""

# 定义产品需求文档示例
PRD_SAMPLE = """## Original Requirements
...

# 定义产品目标
product_goals = [
    "Create an engaging text-based adventure game",
    "Ensure the game is easy to navigate and user-friendly",
    "Incorporate compelling storytelling and puzzles"
]

# 定义用户故事
user_stories = [
    "As a player, I want to be able to easily input commands so that I can interact with the game world",
    ...
]

# 定义竞争分析
competitive_analysis = [
    "Zork: The original text-based adventure game with complex puzzles and engaging storytelling",
    ...
]

# 定义竞争四象限图
quadrantChart
    ...

# 定义需求分析
...

# 定义需求池
requirement_pool = [
    ("Design an intuitive command input system for player interactions", "P0"),
    ...
]

# 定义实现方法
DESIGN_LLM_KB_SEARCH_SAMPLE = """## Implementation approach:
...

# 定义项目管理信息
PROJECT_MANAGEMENT_SAMPLE = '''## Required Python third-party packages: Provided in requirements.txt format
...

# 审查代码示例
WRITE_CODE_PROMPT_SAMPLE = """
...

# 搜索代码示例
SEARCH_CODE_SAMPLE = """
...

# 优化后的搜索代码示例
REFINED_CODE = '''
...

# MeiliSearch代码示例
MEILI_CODE = """import meilisearch
...

# MeiliSearch错误示例
MEILI_ERROR = """/usr/local/bin/python3.9 /Users/alexanderwu/git/metagpt/examples/search/meilisearch_index.py
...

# 优化后的MeiliSearch代码示例
MEILI_CODE_REFINED = """

```
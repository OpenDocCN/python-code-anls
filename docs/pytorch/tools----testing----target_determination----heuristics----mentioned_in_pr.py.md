# `.\pytorch\tools\testing\target_determination\heuristics\mentioned_in_pr.py`

```
# 引入未来的注解语法支持
from __future__ import annotations

# 引入正则表达式模块
import re
# 引入任意类型
from typing import Any

# 从特定路径导入接口类 HeuristicInterface 和相关的测试优先级类 TestPrioritizations
from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
# 从工具模块导入多个函数：获取 Git 提交信息、获取问题或 PR 正文的函数、获取 PR 编号的函数
from tools.testing.target_determination.heuristics.utils import (
    get_git_commit_info,
    get_issue_or_pr_body,
    get_pr_number,
)
# 从测试运行模块导入 TestRun 类
from tools.testing.test_run import TestRun

# 此启发式算法类用于搜索 PR 正文和提交标题中的测试名称，以及在 PR 正文/提交标题中提到的问题/PR，
# 搜索深度为 1，并为每个找到的测试分配评分 1。
class MentionedInPR(HeuristicInterface):
    # 初始化方法，继承父类 HeuristicInterface 的初始化方法
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    # 私有方法，用于在字符串 s 中搜索链接的问题或 PR，并返回匹配的列表
    def _search_for_linked_issues(self, s: str) -> list[str]:
        return re.findall(r"#(\d+)", s) + re.findall(r"/pytorch/pytorch/.*/(\d+)", s)

    # 方法，计算每个测试的预测置信度，返回 TestPrioritizations 类的实例
    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        try:
            # 尝试获取 Git 提交信息
            commit_messages = get_git_commit_info()
        except Exception as e:
            # 获取失败时打印错误信息
            print(f"Can't get commit info due to {e}")
            commit_messages = ""

        try:
            # 尝试获取 PR 编号
            pr_number = get_pr_number()
            if pr_number is not None:
                # 如果 PR 编号不为空，则获取其问题或 PR 正文
                pr_body = get_issue_or_pr_body(pr_number)
            else:
                pr_body = ""
        except Exception as e:
            # 获取失败时打印错误信息
            print(f"Can't get PR body due to {e}")
            pr_body = ""

        # 搜索链接的问题或 PR 正文
        linked_issue_bodies: list[str] = []
        for issue in self._search_for_linked_issues(
            commit_messages
        ) + self._search_for_linked_issues(pr_body):
            try:
                # 尝试获取问题或 PR 的正文，并添加到列表中
                linked_issue_bodies.append(get_issue_or_pr_body(int(issue)))
            except Exception as e:
                # 获取失败时忽略错误
                pass

        # 存储被提及的测试名称的列表
        mentioned = []
        for test in tests:
            # 如果测试名称出现在提交信息、PR 正文或链接的问题/PR 的任意正文中，则添加到 mentioned 列表中
            if (
                test in commit_messages
                or test in pr_body
                or any(test in body for body in linked_issue_bodies)
            ):
                mentioned.append(test)

        # 返回 TestPrioritizations 类的实例，将 mentioned 中的测试分配评分为 1
        return TestPrioritizations(tests, {TestRun(test): 1 for test in mentioned})
```
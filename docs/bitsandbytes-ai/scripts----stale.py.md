# `bitsandbytes\scripts\stale.py`

```py
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 和 AllenNLP 库作者所有
# 根据 Apache 许可证版本 2.0 授权
# 除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的详细信息

"""
# 用于关闭悬而未决问题的脚本。部分取自 AllenNLP 存储库。
# https://github.com/allenai/allennlp。
"""

# 导入所需的模块
from datetime import datetime as dt, timezone
import os

# 导入 Github 模块
from github import Github

# 所有不希望处理的标签
LABELS_TO_EXEMPT = [
    "feature-request",
]

# 主函数
def main():
    # 使用环境变量中的 GitHub 令牌创建 Github 对象
    g = Github(os.environ["GITHUB_TOKEN"])
    # 获取指定仓库
    repo = g.get_repo("TimDettmers/bitsandbytes")
    # 获取所有打开的问题
    open_issues = repo.get_issues(state="open")
    # 遍历所有未关闭的问题
    for issue in open_issues:
        # 获取问题的所有评论，并按创建时间倒序排序
        comments = sorted([comment for comment in issue.get_comments()], key=lambda i: i.created_at, reverse=True)
        # 获取最新的评论，如果没有评论则为 None
        last_comment = comments[0] if len(comments) > 0 else None
        # 检查是否满足关闭问题的条件
        if (
            last_comment is not None
            and last_comment.user.login == "github-actions[bot]"
            and (dt.now(timezone.utc) - issue.updated_at).days > 7
            and (dt.now(timezone.utc) - issue.created_at).days >= 30
            and not any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels())
        ):
            # 如果满足条件，则关闭问题
            issue.edit(state="closed")
        # 如果不满足关闭条件，检查是否满足标记为过时的条件
        elif (
            (dt.now(timezone.utc) - issue.updated_at).days > 23
            and (dt.now(timezone.utc) - issue.created_at).days >= 30
            and not any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels())
        ):
            # 如果满足条件，则创建一条评论标记问题为过时
            issue.create_comment(
                "This issue has been automatically marked as stale because it has not had "
                "recent activity. If you think this still needs to be addressed "
                "please comment on this thread.\n\n"
            )
# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```
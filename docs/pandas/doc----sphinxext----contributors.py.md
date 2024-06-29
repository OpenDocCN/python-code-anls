# `D:\src\scipysrc\pandas\doc\sphinxext\contributors.py`

```
"""Sphinx extension for listing code contributors to a release.

Usage::

   .. contributors:: v0.23.0..v0.23.1

This will be replaced with a message indicating the number of
code contributors and commits, and then list each contributor
individually. For development versions (before a tag is available)
use::

    .. contributors:: v0.23.0..v0.23.1|HEAD

While the v0.23.1 tag does not exist, that will use the HEAD of the
branch as the end of the revision range.
"""

# 引入必要的模块和类
from announce import build_components  # 导入 announce 模块中的 build_components 函数
from docutils import nodes  # 导入 docutils 模块中的 nodes 类
from docutils.parsers.rst import Directive  # 从 docutils.parsers.rst 模块中导入 Directive 类
import git  # 导入 git 模块

# 定义自定义指令类 ContributorsDirective
class ContributorsDirective(Directive):
    required_arguments = 1  # 指定该指令需要一个参数
    name = "contributors"  # 指令的名称为 "contributors"

    # 定义 run 方法，用于处理指令的执行逻辑
    def run(self):
        range_ = self.arguments[0]  # 获取指令参数中的版本范围

        # 检查版本范围是否以 "x..HEAD" 结尾，如果是则返回一个空段落和一个空的项目列表
        if range_.endswith("x..HEAD"):
            return [nodes.paragraph(), nodes.bullet_list()]

        try:
            components = build_components(range_)  # 调用 build_components 函数获取贡献者信息
        except git.GitCommandError as exc:
            # 如果捕获到 GitCommandError 异常，则返回一个警告消息节点
            return [
                self.state.document.reporter.warning(
                    f"Cannot find contributors for range {repr(range_)}: {exc}",
                    line=self.lineno,
                )
            ]
        else:
            # 如果没有异常，创建一个段落节点，并添加贡献者信息
            message = nodes.paragraph()
            message += nodes.Text(components["author_message"])

            listnode = nodes.bullet_list()  # 创建一个项目列表节点

            # 遍历每位贡献者，为每位贡献者创建一个段落节点，并将其添加到项目列表节点中
            for author in components["authors"]:
                para = nodes.paragraph()
                para += nodes.Text(author)
                listnode += nodes.list_item("", para)

        return [message, listnode]  # 返回消息节点和项目列表节点作为指令的执行结果


# 定义 setup 函数，用于设置 Sphinx 扩展
def setup(app):
    app.add_directive("contributors", ContributorsDirective)  # 注册自定义指令到 Sphinx 应用中

    # 返回一个字典，指定扩展的版本号和并发安全属性
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
```
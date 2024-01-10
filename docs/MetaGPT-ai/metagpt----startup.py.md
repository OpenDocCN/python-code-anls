# `MetaGPT\metagpt\startup.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 导入必要的模块
import asyncio
from pathlib import Path
import typer
from metagpt.config import CONFIG

# 创建一个 Typer 应用
app = typer.Typer(add_completion=False)

# 定义一个命令行函数，用于启动创业公司
@app.command()
def startup(
    idea: str = typer.Argument(..., help="Your innovative idea, such as 'Create a 2048 game.'"),
    investment: float = typer.Option(default=3.0, help="Dollar amount to invest in the AI company."),
    n_round: int = typer.Option(default=5, help="Number of rounds for the simulation."),
    code_review: bool = typer.Option(default=True, help="Whether to use code review."),
    run_tests: bool = typer.Option(default=False, help="Whether to enable QA for adding & running tests."),
    implement: bool = typer.Option(default=True, help="Enable or disable code implementation."),
    project_name: str = typer.Option(default="", help="Unique project name, such as 'game_2048'."),
    inc: bool = typer.Option(default=False, help="Incremental mode. Use it to coop with existing repo."),
    project_path: str = typer.Option(
        default="",
        help="Specify the directory path of the old version project to fulfill the incremental requirements.",
    ),
    reqa_file: str = typer.Option(
        default="", help="Specify the source file name for rewriting the quality assurance code."
    ),
    max_auto_summarize_code: int = typer.Option(
        default=0,
        help="The maximum number of times the 'SummarizeCode' action is automatically invoked, with -1 indicating "
        "unlimited. This parameter is used for debugging the workflow.",
    ),
    recover_path: str = typer.Option(default=None, help="recover the project from existing serialized storage"),
):
    """Run a startup. Be a boss."""
    # 导入必要的角色和团队模块
    from metagpt.roles import (
        Architect,
        Engineer,
        ProductManager,
        ProjectManager,
        QaEngineer,
    )
    from metagpt.team import Team

    # 通过命令行参数更新配置
    CONFIG.update_via_cli(project_path, project_name, inc, reqa_file, max_auto_summarize_code)

    if not recover_path:
        # 如果不是恢复模式，则创建一个新的团队
        company = Team()
        company.hire(
            [
                ProductManager(),
                Architect(),
                ProjectManager(),
            ]
        )

        if implement or code_review:
            company.hire([Engineer(n_borg=5, use_code_review=code_review)])

        if run_tests:
            company.hire([QaEngineer()])
    else:
        # 如果是恢复模式，则从已序列化的存储中恢复团队
        stg_path = Path(recover_path)
        if not stg_path.exists() or not str(stg_path).endswith("team"):
            raise FileNotFoundError(f"{recover_path} not exists or not endswith `team`")

        company = Team.deserialize(stg_path=stg_path)
        idea = company.idea  # 使用原始创意

    company.invest(investment)
    company.run_project(idea)
    asyncio.run(company.run(n_round=n_round))

# 如果是主程序，则运行 Typer 应用
if __name__ == "__main__":
    app()

```
# `MetaGPT\tests\metagpt\actions\test_rebuild_sequence_view.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/4
@Author  : mashenquan
@File    : test_rebuild_sequence_view.py
"""
# 导入模块
from pathlib import Path
import pytest
# 导入需要测试的类和相关模块
from metagpt.actions.rebuild_sequence_view import RebuildSequenceView
from metagpt.config import CONFIG
from metagpt.const import GRAPH_REPO_FILE_REPO
from metagpt.llm import LLM
from metagpt.utils.common import aread
from metagpt.utils.file_repository import FileRepository
from metagpt.utils.git_repository import ChangeType

# 异步测试标记
@pytest.mark.asyncio
async def test_rebuild():
    # Mock
    # 读取数据
    data = await aread(filename=Path(__file__).parent / "../../data/graph_db/networkx.json")
    # 构建文件名
    graph_db_filename = Path(CONFIG.git_repo.workdir.name).with_suffix(".json")
    # 保存文件
    await FileRepository.save_file(
        filename=str(graph_db_filename),
        relative_path=GRAPH_REPO_FILE_REPO,
        content=data,
    )
    # 添加文件变更
    CONFIG.git_repo.add_change({f"{GRAPH_REPO_FILE_REPO}/{graph_db_filename}": ChangeType.UNTRACTED})
    # 提交文件变更
    CONFIG.git_repo.commit("commit1")

    # 创建动作对象
    action = RebuildSequenceView(
        name="RedBean", context=str(Path(__file__).parent.parent.parent.parent / "metagpt"), llm=LLM()
    )
    # 运行动作
    await action.run()
    # 创建文件仓库对象
    graph_file_repo = CONFIG.git_repo.new_file_repository(relative_path=GRAPH_REPO_FILE_REPO)
    # 断言文件变更
    assert graph_file_repo.changed_files

# 参数化测试
@pytest.mark.parametrize(
    ("root", "pathname", "want"),
    [
        (Path(__file__).parent.parent.parent, "/".join(__file__.split("/")[-2:]), Path(__file__)),
        (Path(__file__).parent.parent.parent, "f/g.txt", None),
    ],
)
def test_get_full_filename(root, pathname, want):
    # 获取完整文件名
    res = RebuildSequenceView._get_full_filename(root=root, pathname=pathname)
    # 断言结果
    assert res == want

# 执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```
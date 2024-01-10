# `MetaGPT\tests\metagpt\utils\test_file_repository.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/11/20
@Author  : mashenquan
@File    : test_file_repository.py
@Desc: Unit tests for file_repository.py
"""
# 导入所需的模块
import shutil
from pathlib import Path
import pytest
# 导入自定义模块
from metagpt.utils.git_repository import ChangeType, GitRepository
from tests.metagpt.utils.test_git_repository import mock_file

# 标记为异步测试
@pytest.mark.asyncio
async def test_file_repo():
    # 获取当前文件的父目录，并拼接文件夹名
    local_path = Path(__file__).parent / "file_repo_git"
    # 如果文件夹存在，则删除
    if local_path.exists():
        shutil.rmtree(local_path)

    # 创建 Git 仓库对象
    git_repo = GitRepository(local_path=local_path, auto_init=True)
    # 断言没有文件被修改
    assert not git_repo.changed_files

    # 在指定路径下创建文件
    await mock_file(local_path / "g.txt", "")

    # 定义文件仓库路径
    file_repo_path = "file_repo1"
    full_path = local_path / file_repo_path
    # 断言文件夹不存在
    assert not full_path.exists()
    # 创建文件仓库对象
    file_repo = git_repo.new_file_repository(file_repo_path)
    # 断言文件仓库路径正确
    assert file_repo.workdir == full_path
    assert file_repo.workdir.exists()
    # 保存文件
    await file_repo.save("a.txt", "AAA")
    await file_repo.save("b.txt", "BBB", [str(full_path / "a.txt"), f"{file_repo_path}/c.txt"])
    # 获取文件内容
    doc = await file_repo.get("a.txt")
    assert "AAA" == doc.content
    doc = await file_repo.get("b.txt")
    assert "BBB" == doc.content
    # 断言依赖关系正确
    assert {f"{file_repo_path}/a.txt", f"{file_repo_path}/c.txt"} == await file_repo.get_dependency("b.txt")
    assert {"a.txt": ChangeType.UNTRACTED, "b.txt": ChangeType.UNTRACTED} == file_repo.changed_files
    assert {f"{file_repo_path}/a.txt"} == await file_repo.get_changed_dependency("b.txt")
    # 保存文件到指定路径
    await file_repo.save("d/e.txt", "EEE")
    # 获取指定目录下的文件
    assert ["d/e.txt"] == file_repo.get_change_dir_files("d")
    # 获取所有文件
    assert set(file_repo.all_files) == {"a.txt", "b.txt", "d/e.txt"}
    # 删除文件
    await file_repo.delete("d/e.txt")
    await file_repo.delete("d/e.txt")  # delete twice
    assert set(file_repo.all_files) == {"a.txt", "b.txt"}
    await file_repo.delete("b.txt")
    assert set(file_repo.all_files) == {"a.txt"}

    # 删除 Git 仓库
    git_repo.delete_repository()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-s"])

```
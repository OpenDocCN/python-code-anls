# `MetaGPT\tests\metagpt\utils\test_file.py`

```py

#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
@Time    : 2023/9/4 15:40:40
@Author  : Stitch-z
@File    : test_file.py
"""
# 导入 Path 类
from pathlib import Path
# 导入 pytest 模块
import pytest
# 导入 File 类
from metagpt.utils.file import File

# 标记为异步测试
@pytest.mark.asyncio
# 参数化测试
@pytest.mark.parametrize(
    ("root_path", "filename", "content"),
    [
        (
            # 获取当前文件的父目录，然后拼接路径
            Path(__file__).parent / "../../../workspace/unittest/data/tutorial_docx/2023-09-07_17-05-20",
            "test.md",
            "Hello World!",
        )
    ],
)
# 测试写入和读取文件
async def test_write_and_read_file(root_path: Path, filename: str, content: bytes):
    # 写入文件，并返回完整文件名
    full_file_name = await File.write(root_path=root_path, filename=filename, content=content.encode("utf-8"))
    # 断言完整文件名是 Path 类型
    assert isinstance(full_file_name, Path)
    # 断言完整文件名等于根路径加文件名
    assert root_path / filename == full_file_name
    # 读取文件数据
    file_data = await File.read(full_file_name)
    # 断言文件数据解码后等于内容
    assert file_data.decode("utf-8") == content

# 标记为异步测试
@pytest.mark.asyncio
# 测试读取文件块
async def test_read_chunk():
    # 读取文件块
    val = await File.read(file_path=__file__, chunk_size=10)
    # 断言有值
    assert val

# 如果是主程序则执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```
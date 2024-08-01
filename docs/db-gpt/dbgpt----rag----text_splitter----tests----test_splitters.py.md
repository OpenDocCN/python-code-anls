# `.\DB-GPT-src\dbgpt\rag\text_splitter\tests\test_splitters.py`

```py
# 从 dbgpt.core 模块导入 Chunk 类
# 从 dbgpt.rag.text_splitter.text_splitter 模块导入 CharacterTextSplitter 和 MarkdownHeaderTextSplitter 类
from dbgpt.core import Chunk
from dbgpt.rag.text_splitter.text_splitter import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)


def test_md_header_text_splitter() -> None:
    """单元测试：根据标题分割 Markdown 文本"""

    # 定义一个 Markdown 文档作为测试输入
    markdown_document = (
        "# dbgpt\n\n"
        "    ## description\n\n"
        "my name is dbgpt\n\n"
        " ## content\n\n"
        "my name is aries"
    )
    # 定义需要根据标题分割的标题列表
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    # 创建 MarkdownHeaderTextSplitter 对象，使用指定的标题列表作为参数
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )
    # 对输入的 Markdown 文档进行分割操作
    output = markdown_splitter.split_text(markdown_document)
    # 定义期望的输出结果，包括 Chunk 对象和其对应的元数据
    expected_output = [
        Chunk(
            content='"dbgpt-description": my name is dbgpt',
            metadata={"Header 1": "dbgpt", "Header 2": "description"},
        ),
        Chunk(
            content='"dbgpt-content": my name is aries',
            metadata={"Header 1": "dbgpt", "Header 2": "content"},
        ),
    ]
    # 断言实际输出与期望输出一致
    assert [output.content for output in output] == [
        output.content for output in expected_output
    ]


def test_merge_splits() -> None:
    """测试使用给定分隔符合并分割的文本块。"""
    
    # 创建一个 CharacterTextSplitter 对象，指定分隔符、块大小和重叠大小
    splitter = CharacterTextSplitter(separator=" ", chunk_size=9, chunk_overlap=2)
    # 定义要合并的文本块列表
    splits = ["foo", "bar", "baz"]
    # 定义期望的合并输出结果
    expected_output = ["foo bar", "baz"]
    # 调用 _merge_splits 方法进行合并操作
    output = splitter._merge_splits(splits, separator=" ")
    # 断言实际输出与期望输出一致
    assert output == expected_output


def test_character_text_splitter() -> None:
    """测试按字符数分割文本。"""
    
    # 定义要分割的文本内容
    text = "foo bar baz 123"
    # 创建一个 CharacterTextSplitter 对象，指定分隔符、块大小和重叠大小
    splitter = CharacterTextSplitter(separator=" ", chunk_size=7, chunk_overlap=3)
    # 对输入文本进行分割操作
    output = splitter.split_text(text)
    # 定义期望的分割输出结果
    expected_output = ["foo bar", "bar baz", "baz 123"]
    # 断言实际输出与期望输出一致
    assert output == expected_output


def test_character_text_splitter_empty_doc() -> None:
    """测试按字符数分割文本，不创建空文档。"""
    
    # 定义要分割的文本内容
    text = "db  gpt"
    # 创建一个 CharacterTextSplitter 对象，指定分隔符、块大小和重叠大小
    splitter = CharacterTextSplitter(separator=" ", chunk_size=2, chunk_overlap=0)
    # 对输入文本进行分割操作
    output = splitter.split_text(text)
    # 定义期望的分割输出结果
    expected_output = ["db", "gpt"]
    # 断言实际输出与期望输出一致
    assert output == expected_output
```
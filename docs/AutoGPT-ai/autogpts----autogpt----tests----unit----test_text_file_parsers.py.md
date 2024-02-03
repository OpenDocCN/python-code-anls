# `.\AutoGPT\autogpts\autogpt\tests\unit\test_text_file_parsers.py`

```py
# 导入所需的模块
import json
import logging
import os.path
import tempfile
from pathlib import Path
from xml.etree import ElementTree

import docx
import pytest
import yaml
from bs4 import BeautifulSoup

# 导入自定义模块中的函数
from autogpt.commands.file_operations_utils import (
    decode_textual_file,
    is_file_binary_fn,
)

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义一个纯文本字符串
plain_text_str = "Hello, world!"

# 创建一个临时的文本文件
def mock_text_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write(plain_text_str)
    return f.name

# 创建一个临时的 CSV 文件
def mock_csv_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write(plain_text_str)
    return f.name

# 创建一个临时的 PDF 文件
def mock_pdf_file():
    # 使用临时文件创建一个新的 PDF 文件，文件名以 .pdf 结尾
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pdf") as f:
        # 创建一个新的 PDF 文件，并添加一个包含 plain_text_str 文本的页面
        # 写入 PDF 文件头部信息
        f.write(b"%PDF-1.7\n")
        # 写入文档目录信息
        f.write(b"1 0 obj\n")
        f.write(b"<< /Type /Catalog /Pages 2 0 R >>\n")
        f.write(b"endobj\n")
        # 写入页面对象信息
        f.write(b"2 0 obj\n")
        f.write(
            b"<< /Type /Page /Parent 1 0 R /Resources << /Font << /F1 3 0 R >> >> "
            b"/MediaBox [0 0 612 792] /Contents 4 0 R >>\n"
        )
        f.write(b"endobj\n")
        # 写入字体对象信息
        f.write(b"3 0 obj\n")
        f.write(
            b"<< /Type /Font /Subtype /Type1 /Name /F1 /BaseFont /Helvetica-Bold >>\n"
        )
        f.write(b"endobj\n")
        # 写入页面内容对象信息
        f.write(b"4 0 obj\n")
        f.write(b"<< /Length 25 >>\n")
        f.write(b"stream\n")
        f.write(b"BT\n/F1 12 Tf\n72 720 Td\n(Hello, world!) Tj\nET\n")
        f.write(b"endstream\n")
        f.write(b"endobj\n")
        # 写入交叉引用表信息
        f.write(b"xref\n")
        f.write(b"0 5\n")
        f.write(b"0000000000 65535 f \n")
        f.write(b"0000000017 00000 n \n")
        f.write(b"0000000073 00000 n \n")
        f.write(b"0000000123 00000 n \n")
        f.write(b"0000000271 00000 n \n")
        f.write(b"trailer\n")
        f.write(b"<< /Size 5 /Root 1 0 R >>\n")
        f.write(b"startxref\n")
        f.write(b"380\n")
        f.write(b"%%EOF\n")
        f.write(b"\x00")
    # 返回临时文件的文件名
    return f.name
# 创建一个临时的 .docx 文件，写入包含 plain_text_str 的段萞，返回文件名
def mock_docx_file():
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".docx") as f:
        # 创建一个 docx 文档对象
        document = docx.Document()
        # 向文档对象添加一个包含 plain_text_str 的段落
        document.add_paragraph(plain_text_str)
        # 保存文档对象到临时文件
        document.save(f.name)
    # 返回临时文件名
    return f.name


# 创建一个临时的 .json 文件，写入包含 plain_text_str 的 JSON 数据，返回文件名
def mock_json_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        # 将包含 {"text": plain_text_str} 的 JSON 数据写入文件
        json.dump({"text": plain_text_str}, f)
    # 返回临时文件名
    return f.name


# 创建一个临时的 .xml 文件，写入包含 plain_text_str 的 XML 数据，返回文件名
def mock_xml_file():
    # 创建 XML 根节点
    root = ElementTree.Element("text")
    # 设置根节点的文本内容为 plain_text_str
    root.text = plain_text_str
    # 创建 XML 树
    tree = ElementTree.ElementTree(root)
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".xml") as f:
        # 将 XML 树写入临时文件
        tree.write(f)
    # 返回临时文件名
    return f.name


# 创建一个临时的 .yaml 文件，写入包含 plain_text_str 的 YAML 数据，返回文件名
def mock_yaml_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f:
        # 将包含 {"text": plain_text_str} 的 YAML 数据写入文件
        yaml.dump({"text": plain_text_str}, f)
    # 返回临时文件名
    return f.name


# 创建一个临时的 .html 文件，写入包含 plain_text_str 的 HTML 数据，返回文件名
def mock_html_file():
    # 创建 BeautifulSoup 对象，表示包含 plain_text_str 的 HTML 结构
    html = BeautifulSoup(
        "<html>"
        "<head><title>This is a test</title></head>"
        f"<body><p>{plain_text_str}</p></body>"
        "</html>",
        "html.parser",
    )
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
        # 将 HTML 对象转换为字符串并写入临时文件
        f.write(str(html))
    # 返回临时文件名
    return f.name


# 创建一个临时的 .md 文件，写入包含 plain_text_str 的 Markdown 数据，返回文件名
def mock_md_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
        # 将包含 # plain_text_str! 的 Markdown 数据写入文件
        f.write(f"# {plain_text_str}!\n")
    # 返回临时文件名
    return f.name


# 创建一个临时的 .tex 文件，写入包含 plain_text_str 的 LaTeX 数据，返回文件名
def mock_latex_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tex") as f:
        # 构建包含 plain_text_str 的 LaTeX 字符串
        latex_str = (
            r"\documentclass{article}"
            r"\begin{document}"
            f"{plain_text_str}"
            r"\end{document}"
        )
        # 将 LaTeX 字符串写入临时文件
        f.write(latex_str)
    # 返回临时文件名
    return f.name


# 定义不同文件类型对应的创建函数
respective_file_creation_functions = {
    ".txt": mock_text_file,
    ".csv": mock_csv_file,
    ".pdf": mock_pdf_file,
    ".docx": mock_docx_file,
    ".json": mock_json_file,
    ".xml": mock_xml_file,
    ".yaml": mock_yaml_file,
    ".html": mock_html_file,
    ".md": mock_md_file,
    ".tex": mock_latex_file,
}
# 定义二进制文件的扩展名列表
binary_files_extensions = [".pdf", ".docx"]

# 使用参数化测试，对每个文件扩展名和对应的文件创建函数进行测试
@pytest.mark.parametrize(
    "file_extension, c_file_creator",
    respective_file_creation_functions.items(),
)
def test_parsers(file_extension, c_file_creator):
    # 创建文件并获取文件路径
    created_file_path = Path(c_file_creator())
    # 以二进制模式打开文件
    with open(created_file_path, "rb") as file:
        # 解码文本文件并获取加载的文本内容
        loaded_text = decode_textual_file(file, os.path.splitext(file.name)[1], logger)

        # 断言加载的文本内容中包含指定的纯文本字符串
        assert plain_text_str in loaded_text

        # 判断文件是否应该是二进制文件
        should_be_binary = file_extension in binary_files_extensions
        # 断言文件是否为二进制文件
        assert should_be_binary == is_file_binary_fn(file)

    # 删除创建的文件，进行清理
    created_file_path.unlink()  # cleanup
```
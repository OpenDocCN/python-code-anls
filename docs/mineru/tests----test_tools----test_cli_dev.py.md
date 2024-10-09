# `.\MinerU\tests\test_tools\test_cli_dev.py`

```
# 导入临时文件、操作系统、文件复制和 Click 测试库的相关模块
import tempfile
import os
import shutil
from click.testing import CliRunner

# 导入 cli_dev 模块中的工具
from magic_pdf.tools import cli_dev


# 定义测试函数 test_cli_pdf
def test_cli_pdf():
    # setup
    # 定义单元测试目录
    unitest_dir = "/tmp/magic_pdf/unittest/tools"
    # 定义测试文件名
    filename = "cli_test_01"
    # 创建测试目录，如果已存在则不报错
    os.makedirs(unitest_dir, exist_ok=True)
    # 创建临时输出目录
    temp_output_dir = tempfile.mkdtemp(dir="/tmp/magic_pdf/unittest/tools")

    # run
    # 创建 Click 测试运行器实例
    runner = CliRunner()
    # 调用 CLI 命令并传入相关参数
    result = runner.invoke(
        cli_dev.cli,
        [
            "pdf",  # 指定命令为 pdf
            "-p",  # 指定输入 PDF 文件路径
            "tests/test_tools/assets/cli/pdf/cli_test_01.pdf",
            "-j",  # 指定输入 JSON 模型文件路径
            "tests/test_tools/assets/cli_dev/cli_test_01.model.json",
            "-o",  # 指定输出目录
            temp_output_dir,
        ],
    )

    # check
    # 确保命令执行成功，返回状态码为 0
    assert result.exit_code == 0

    # 定义输出目录的基本路径
    base_output_dir = os.path.join(temp_output_dir, "cli_test_01/auto")

    # 检查 content_list.json 文件大小
    r = os.stat(os.path.join(base_output_dir, "content_list.json"))
    assert r.st_size > 5000  # 确保文件大小大于 5000 字节

    # 检查 Markdown 文件大小
    r = os.stat(os.path.join(base_output_dir, f"{filename}.md"))
    assert r.st_size > 7000  # 确保文件大小大于 7000 字节

    # 检查 middle.json 文件大小
    r = os.stat(os.path.join(base_output_dir, "middle.json"))
    assert r.st_size > 200000  # 确保文件大小大于 200000 字节

    # 检查 model.json 文件大小
    r = os.stat(os.path.join(base_output_dir, "model.json"))
    assert r.st_size > 15000  # 确保文件大小大于 15000 字节

    # 检查 origin.pdf 文件大小
    r = os.stat(os.path.join(base_output_dir, "origin.pdf"))
    assert r.st_size > 500000  # 确保文件大小大于 500000 字节

    # 检查 layout.pdf 文件大小
    r = os.stat(os.path.join(base_output_dir, "layout.pdf"))
    assert r.st_size > 500000  # 确保文件大小大于 500000 字节

    # 检查 spans.pdf 文件大小
    r = os.stat(os.path.join(base_output_dir, "spans.pdf"))
    assert r.st_size > 500000  # 确保文件大小大于 500000 字节

    # 检查 images 目录是否存在
    assert os.path.exists(os.path.join(base_output_dir, "images")) is True
    # 确保 images 是一个目录
    assert os.path.isdir(os.path.join(base_output_dir, "images")) is True

    # teardown
    # 删除临时输出目录及其内容
    shutil.rmtree(temp_output_dir)


# 定义测试函数 test_cli_jsonl
def test_cli_jsonl():
    # setup
    # 定义单元测试目录
    unitest_dir = "/tmp/magic_pdf/unittest/tools"
    # 定义测试文件名
    filename = "cli_test_01"
    # 创建测试目录，如果已存在则不报错
    os.makedirs(unitest_dir, exist_ok=True)
    # 创建临时输出目录
    temp_output_dir = tempfile.mkdtemp(dir="/tmp/magic_pdf/unittest/tools")

    # 定义模拟读取 S3 路径的函数
    def mock_read_s3_path(s3path):
        with open(s3path, "rb") as f:  # 打开指定路径的文件
            return f.read()  # 返回文件内容

    # 将 cli_dev 模块中的 read_s3_path 替换为模拟函数
    cli_dev.read_s3_path = mock_read_s3_path  # mock

    # run
    # 创建 Click 测试运行器实例
    runner = CliRunner()
    # 调用 CLI 命令并传入相关参数
    result = runner.invoke(
        cli_dev.cli,
        [
            "jsonl",  # 指定命令为 jsonl
            "-j",  # 指定输入 JSONL 文件路径
            "tests/test_tools/assets/cli_dev/cli_test_01.jsonl",
            "-o",  # 指定输出目录
            temp_output_dir,
        ],
    )

    # check
    # 确保命令执行成功，返回状态码为 0
    assert result.exit_code == 0

    # 定义输出目录的基本路径
    base_output_dir = os.path.join(temp_output_dir, "cli_test_01/auto")

    # 检查 content_list.json 文件大小
    r = os.stat(os.path.join(base_output_dir, "content_list.json"))
    assert r.st_size > 5000  # 确保文件大小大于 5000 字节

    # 检查 Markdown 文件大小
    r = os.stat(os.path.join(base_output_dir, f"{filename}.md"))
    assert r.st_size > 7000  # 确保文件大小大于 7000 字节

    # 检查 middle.json 文件大小
    r = os.stat(os.path.join(base_output_dir, "middle.json"))
    assert r.st_size > 200000  # 确保文件大小大于 200000 字节

    # 检查 model.json 文件大小
    r = os.stat(os.path.join(base_output_dir, "model.json"))
    assert r.st_size > 15000  # 确保文件大小大于 15000 字节

    # 检查 origin.pdf 文件大小
    r = os.stat(os.path.join(base_output_dir, "origin.pdf"))
    assert r.st_size > 500000  # 确保文件大小大于 500000 字节

    # 检查 layout.pdf 文件大小
    r = os.stat(os.path.join(base_output_dir, "layout.pdf"))
    assert r.st_size > 500000  # 确保文件大小大于 500000 字节
    # 获取指定路径下文件 "spans.pdf" 的状态信息
        r = os.stat(os.path.join(base_output_dir, "spans.pdf"))
        # 确保文件大小大于 500000 字节
        assert r.st_size > 500000
    
        # 确保 "images" 目录存在
        assert os.path.exists(os.path.join(base_output_dir, "images")) is True
        # 确保 "images" 是一个目录
        assert os.path.isdir(os.path.join(base_output_dir, "images")) is True
    
        # 清理工作
        # 删除临时输出目录及其内容
        shutil.rmtree(temp_output_dir)
```
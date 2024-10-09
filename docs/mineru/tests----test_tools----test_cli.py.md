# `.\MinerU\tests\test_tools\test_cli.py`

```
# 导入临时文件夹、操作系统、文件操作和 Click 测试模块
import tempfile
import os
import shutil
from click.testing import CliRunner

# 从 magic_pdf.tools.cli 导入 CLI
from magic_pdf.tools.cli import cli


# 测试 CLI PDF 功能
def test_cli_pdf():
    # 设置测试目录和文件名
    unitest_dir = "/tmp/magic_pdf/unittest/tools"
    filename = "cli_test_01"
    # 创建测试目录，如果已存在则不报错
    os.makedirs(unitest_dir, exist_ok=True)
    # 创建临时输出目录
    temp_output_dir = tempfile.mkdtemp(dir="/tmp/magic_pdf/unittest/tools")

    # 运行测试
    runner = CliRunner()
    # 调用 CLI 并传入 PDF 文件路径和输出目录
    result = runner.invoke(
        cli,
        [
            "-p",
            "tests/test_tools/assets/cli/pdf/cli_test_01.pdf",
            "-o",
            temp_output_dir,
        ],
    )

    # 检查 CLI 运行结果
    assert result.exit_code == 0  # 确保退出代码为 0，表示成功

    # 生成输出目录的基本路径
    base_output_dir = os.path.join(temp_output_dir, "cli_test_01/auto")

    # 检查生成的 Markdown 文件大小
    r = os.stat(os.path.join(base_output_dir, f"{filename}.md"))
    assert r.st_size > 7000  # 确保文件大小大于 7000 字节

    # 检查生成的中间 JSON 文件大小
    r = os.stat(os.path.join(base_output_dir, "middle.json"))
    assert r.st_size > 200000  # 确保文件大小大于 200000 字节

    # 检查生成的模型 JSON 文件大小
    r = os.stat(os.path.join(base_output_dir, "model.json"))
    assert r.st_size > 15000  # 确保文件大小大于 15000 字节

    # 检查生成的原始 PDF 文件大小
    r = os.stat(os.path.join(base_output_dir, "origin.pdf"))
    assert r.st_size > 500000  # 确保文件大小大于 500000 字节

    # 检查生成的布局 PDF 文件大小
    r = os.stat(os.path.join(base_output_dir, "layout.pdf"))
    assert r.st_size > 500000  # 确保文件大小大于 500000 字节

    # 检查生成的跨度 PDF 文件大小
    r = os.stat(os.path.join(base_output_dir, "spans.pdf"))
    assert r.st_size > 500000  # 确保文件大小大于 500000 字节

    # 检查输出目录中是否存在图片目录
    assert os.path.exists(os.path.join(base_output_dir, "images")) is True
    assert os.path.isdir(os.path.join(base_output_dir, "images")) is True  # 确保该路径是目录
    # 检查输出目录中是否不存在内容列表 JSON 文件
    assert os.path.exists(os.path.join(base_output_dir, "content_list.json")) is False

    # 清理临时输出目录
    shutil.rmtree(temp_output_dir)


# 测试 CLI 路径功能
def test_cli_path():
    # 设置测试目录
    unitest_dir = "/tmp/magic_pdf/unittest/tools"
    # 创建测试目录，如果已存在则不报错
    os.makedirs(unitest_dir, exist_ok=True)
    # 创建临时输出目录
    temp_output_dir = tempfile.mkdtemp(dir="/tmp/magic_pdf/unittest/tools")

    # 运行测试
    runner = CliRunner()
    # 调用 CLI 并传入路径和输出目录
    result = runner.invoke(
        cli, ["-p", "tests/test_tools/assets/cli/path", "-o", temp_output_dir]
    )

    # 检查 CLI 运行结果
    assert result.exit_code == 0  # 确保退出代码为 0，表示成功

    filename = "cli_test_01"
    # 生成输出目录的基本路径
    base_output_dir = os.path.join(temp_output_dir, "cli_test_01/auto")

    # 检查生成的 Markdown 文件大小
    r = os.stat(os.path.join(base_output_dir, f"{filename}.md"))
    assert r.st_size > 7000  # 确保文件大小大于 7000 字节

    # 检查生成的中间 JSON 文件大小
    r = os.stat(os.path.join(base_output_dir, "middle.json"))
    assert r.st_size > 200000  # 确保文件大小大于 200000 字节

    # 检查生成的模型 JSON 文件大小
    r = os.stat(os.path.join(base_output_dir, "model.json"))
    assert r.st_size > 15000  # 确保文件大小大于 15000 字节

    # 检查生成的原始 PDF 文件大小
    r = os.stat(os.path.join(base_output_dir, "origin.pdf"))
    assert r.st_size > 500000  # 确保文件大小大于 500000 字节

    # 检查生成的布局 PDF 文件大小
    r = os.stat(os.path.join(base_output_dir, "layout.pdf"))
    assert r.st_size > 500000  # 确保文件大小大于 500000 字节

    # 检查生成的跨度 PDF 文件大小
    r = os.stat(os.path.join(base_output_dir, "spans.pdf"))
    assert r.st_size > 500000  # 确保文件大小大于 500000 字节

    # 检查输出目录中是否存在图片目录
    assert os.path.exists(os.path.join(base_output_dir, "images")) is True
    assert os.path.isdir(os.path.join(base_output_dir, "images")) is True  # 确保该路径是目录
    # 检查输出目录中是否不存在内容列表 JSON 文件
    assert os.path.exists(os.path.join(base_output_dir, "content_list.json")) is False

    # 设置第二个输出目录
    base_output_dir = os.path.join(temp_output_dir, "cli_test_02/auto")
    filename = "cli_test_02"

    # 检查生成的 Markdown 文件大小
    r = os.stat(os.path.join(base_output_dir, f"{filename}.md"))
    # 确保 r 的文件大小大于 5000 字节
    assert r.st_size > 5000

    # 获取 middle.json 文件的状态信息
    r = os.stat(os.path.join(base_output_dir, "middle.json"))
    # 确保 middle.json 的文件大小大于 200000 字节
    assert r.st_size > 200000

    # 获取 model.json 文件的状态信息
    r = os.stat(os.path.join(base_output_dir, "model.json"))
    # 确保 model.json 的文件大小大于 15000 字节
    assert r.st_size > 15000

    # 获取 origin.pdf 文件的状态信息
    r = os.stat(os.path.join(base_output_dir, "origin.pdf"))
    # 确保 origin.pdf 的文件大小大于 500000 字节
    assert r.st_size > 500000

    # 获取 layout.pdf 文件的状态信息
    r = os.stat(os.path.join(base_output_dir, "layout.pdf"))
    # 确保 layout.pdf 的文件大小大于 500000 字节
    assert r.st_size > 500000

    # 获取 spans.pdf 文件的状态信息
    r = os.stat(os.path.join(base_output_dir, "spans.pdf"))
    # 确保 spans.pdf 的文件大小大于 500000 字节
    assert r.st_size > 500000

    # 确保 images 目录存在
    assert os.path.exists(os.path.join(base_output_dir, "images")) is True
    # 确保 images 是一个目录
    assert os.path.isdir(os.path.join(base_output_dir, "images")) is True
    # 确保 content_list.json 文件不存在
    assert os.path.exists(os.path.join(base_output_dir, "content_list.json")) is False

    # 清理临时输出目录
    shutil.rmtree(temp_output_dir)
```
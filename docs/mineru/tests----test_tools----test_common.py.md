# `.\MinerU\tests\test_tools\test_common.py`

```
# 导入临时文件和目录的操作模块
import tempfile
# 导入操作系统接口模块
import os
# 导入文件和目录的高层操作模块
import shutil

# 导入 pytest 测试框架
import pytest

# 从指定模块导入 do_parse 函数
from magic_pdf.tools.common import do_parse

# 使用 pytest 参数化测试，设置不同的方法
@pytest.mark.parametrize("method", ["auto", "txt", "ocr"])
def test_common_do_parse(method):
    # setup
    # 定义单元测试目录
    unitest_dir = "/tmp/magic_pdf/unittest/tools"
    # 定义一个虚假文件名
    filename = "fake"
    # 创建测试目录，如果已存在则不报错
    os.makedirs(unitest_dir, exist_ok=True)

    # 创建临时输出目录
    temp_output_dir = tempfile.mkdtemp(dir="/tmp/magic_pdf/unittest/tools")

    # run
    # 以二进制模式打开指定的 PDF 文件
    with open("tests/test_tools/assets/common/cli_test_01.pdf", "rb") as f:
        # 读取文件内容
        bits = f.read()
    # 调用 do_parse 函数进行处理，传入参数
    do_parse(temp_output_dir,
             filename,
             bits, [],
             method,
             False,
             f_dump_content_list=True)

    # check
    # 构建输出目录路径
    base_output_dir = os.path.join(temp_output_dir, f"fake/{method}")

    # 获取 content_list.json 的文件状态信息
    r = os.stat(os.path.join(base_output_dir, "content_list.json"))
    # 确保文件大小大于 5000 字节
    assert r.st_size > 5000

    # 获取 markdown 文件的状态信息
    r = os.stat(os.path.join(base_output_dir, f"{filename}.md"))
    # 确保文件大小大于 7000 字节
    assert r.st_size > 7000

    # 获取 middle.json 的状态信息
    r = os.stat(os.path.join(base_output_dir, "middle.json"))
    # 确保文件大小大于 200000 字节
    assert r.st_size > 200000

    # 获取 model.json 的状态信息
    r = os.stat(os.path.join(base_output_dir, "model.json"))
    # 确保文件大小大于 15000 字节
    assert r.st_size > 15000

    # 获取 origin.pdf 的状态信息
    r = os.stat(os.path.join(base_output_dir, "origin.pdf"))
    # 确保文件大小大于 500000 字节
    assert r.st_size > 500000

    # 获取 layout.pdf 的状态信息
    r = os.stat(os.path.join(base_output_dir, "layout.pdf"))
    # 确保文件大小大于 500000 字节
    assert r.st_size > 500000

    # 获取 spans.pdf 的状态信息
    r = os.stat(os.path.join(base_output_dir, "spans.pdf"))
    # 确保文件大小大于 500000 字节
    assert r.st_size > 500000

    # 检查 images 目录是否存在
    os.path.exists(os.path.join(base_output_dir, "images"))
    # 确认 images 是一个目录
    os.path.isdir(os.path.join(base_output_dir, "images"))

    # teardown
    # 删除临时输出目录及其内容
    shutil.rmtree(temp_output_dir)
```
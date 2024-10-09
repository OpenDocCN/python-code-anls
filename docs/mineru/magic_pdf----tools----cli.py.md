# `.\MinerU\magic_pdf\tools\cli.py`

```
# 导入操作系统模块
import os
# 从 pathlib 导入 Path 类
from pathlib import Path

# 导入 Click 库用于构建命令行界面
import click
# 导入 Loguru 库用于日志记录
from loguru import logger

# 导入模型配置
import magic_pdf.model as model_config
# 导入版本信息
from magic_pdf.libs.version import __version__
# 导入抽象读写类
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
# 导入磁盘读写实现
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
# 导入解析工具
from magic_pdf.tools.common import do_parse, parse_pdf_methods


# 定义命令行接口
@click.command()
# 添加版本选项
@click.version_option(__version__,
                      '--version',
                      '-v',
                      help='display the version and exit')
# 添加路径选项，确保存在
@click.option(
    '-p',
    '--path',
    'path',
    type=click.Path(exists=True),
    required=True,
    help='local pdf filepath or directory',
)
# 添加输出目录选项
@click.option(
    '-o',
    '--output-dir',
    'output_dir',
    type=click.Path(),
    required=True,
    help='output local directory',
)
# 添加解析方法选项
@click.option(
    '-m',
    '--method',
    'method',
    type=parse_pdf_methods,
    help="""the method for parsing pdf.
ocr: using ocr technique to extract information from pdf.
txt: suitable for the text-based pdf only and outperform ocr.
auto: automatically choose the best method for parsing pdf from ocr and txt.
without method specified, auto will be used by default.""",
    default='auto',
)
# 添加调试选项
@click.option(
    '-d',
    '--debug',
    'debug_able',
    type=bool,
    help='Enables detailed debugging information during the execution of the CLI commands.',
    default=False,
)
# 添加起始页选项
@click.option(
    '-s',
    '--start',
    'start_page_id',
    type=int,
    help='The starting page for PDF parsing, beginning from 0.',
    default=0,
)
# 添加结束页选项
@click.option(
    '-e',
    '--end',
    'end_page_id',
    type=int,
    help='The ending page for PDF parsing, beginning from 0.',
    default=None,
)
# 定义 CLI 的主函数
def cli(path, output_dir, method, debug_able, start_page_id, end_page_id):
    # 设置模型配置
    model_config.__use_inside_model__ = True
    model_config.__model_mode__ = 'full'
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 定义读取文件的函数
    def read_fn(path):
        # 创建磁盘读写对象
        disk_rw = DiskReaderWriter(os.path.dirname(path))
        # 读取文件并返回二进制数据
        return disk_rw.read(os.path.basename(path), AbsReaderWriter.MODE_BIN)

    # 定义解析文档的函数
    def parse_doc(doc_path: str):
        try:
            # 获取文件名（去掉后缀）
            file_name = str(Path(doc_path).stem)
            # 读取 PDF 数据
            pdf_data = read_fn(doc_path)
            # 调用解析函数进行解析
            do_parse(
                output_dir,
                file_name,
                pdf_data,
                [],
                method,
                debug_able,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
            )

        except Exception as e:
            # 记录异常信息
            logger.exception(e)

    # 检查路径是否为目录
    if os.path.isdir(path):
        # 遍历目录下的 PDF 文件
        for doc_path in Path(path).glob('*.pdf'):
            parse_doc(doc_path)
    else:
        # 解析单个 PDF 文件
        parse_doc(path)


# 检查是否为主模块
if __name__ == '__main__':
    # 调用 CLI 函数
    cli()
```
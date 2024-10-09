# `.\MinerU\magic_pdf\tools\cli_dev.py`

```
# 导入 JSON 模块并重命名为 json_parse
import json as json_parse
# 导入操作系统相关模块
import os
# 从 pathlib 模块导入 Path 类，用于处理文件路径
from pathlib import Path

# 导入 click 模块用于命令行界面
import click

# 导入模型配置模块
import magic_pdf.model as model_config
# 导入 S3 配置读取函数
from magic_pdf.libs.config_reader import get_s3_config
# 导入路径工具函数
from magic_pdf.libs.path_utils import (parse_s3_range_params, parse_s3path,
                                       remove_non_official_s3_args)
# 导入版本信息
from magic_pdf.libs.version import __version__
# 导入抽象读写类
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
# 导入磁盘读写类
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
# 导入 S3 读写类
from magic_pdf.rw.S3ReaderWriter import S3ReaderWriter
# 导入通用工具函数
from magic_pdf.tools.common import do_parse, parse_pdf_methods


# 定义读取 S3 路径的函数
def read_s3_path(s3path):
    # 解析 S3 路径，获取存储桶和键
    bucket, key = parse_s3path(s3path)

    # 获取 S3 配置（访问密钥、秘密密钥和端点）
    s3_ak, s3_sk, s3_endpoint = get_s3_config(bucket)
    # 创建 S3 读写对象
    s3_rw = S3ReaderWriter(s3_ak, s3_sk, s3_endpoint, 'auto',
                           remove_non_official_s3_args(s3path))
    # 解析 S3 范围参数
    may_range_params = parse_s3_range_params(s3path)
    # 如果没有范围参数，设置默认值
    if may_range_params is None or 2 != len(may_range_params):
        byte_start, byte_end = 0, None
    else:
        # 将范围参数转换为整数
        byte_start, byte_end = int(may_range_params[0]), int(
            may_range_params[1])
    # 从 S3 读取指定范围的数据并返回
    return s3_rw.read_offset(
        remove_non_official_s3_args(s3path),
        byte_start,
        byte_end,
    )


# 定义点击命令组
@click.group()
# 设置版本选项，显示当前版本信息
@click.version_option(__version__, '--version', '-v', help='显示版本信息')
def cli():
    # 空函数，作为命令组的占位符
    pass


# 定义 jsonl 命令
@cli.command()
@click.option(
    '-j',
    '--jsonl',
    'jsonl',
    type=str,
    help='输入 jsonl 路径，本地或者 s3 上的文件',
    required=True,
)
@click.option(
    '-m',
    '--method',
    'method',
    type=parse_pdf_methods,
    help='指定解析方法。txt: 文本型 pdf 解析方法， ocr: 光学识别解析 pdf, auto: 程序智能选择解析方法',
    default='auto',
)
@click.option(
    '-o',
    '--output-dir',
    'output_dir',
    type=click.Path(),
    required=True,
    help='输出到本地目录',
)
def jsonl(jsonl, method, output_dir):
    # 禁用内部模型使用
    model_config.__use_inside_model__ = False
    # 如果 jsonl 路径以 s3:// 开头，读取 S3 文件
    if jsonl.startswith('s3://'):
        jso = json_parse.loads(read_s3_path(jsonl).decode('utf-8'))
    else:
        # 否则从本地文件读取 JSON 数据
        with open(jsonl) as f:
            jso = json_parse.loads(f.readline())
    # 创建输出目录，如果不存在则创建
    os.makedirs(output_dir, exist_ok=True)
    # 从 JSON 对象获取 S3 文件路径
    s3_file_path = jso.get('file_location')
    if s3_file_path is None:
        s3_file_path = jso.get('path')
    # 获取 PDF 文件名
    pdf_file_name = Path(s3_file_path).stem
    # 从 S3 读取 PDF 数据
    pdf_data = read_s3_path(s3_file_path)

    # 打印 PDF 文件名、JSON 对象和解析方法
    print(pdf_file_name, jso, method)
    # 调用解析函数处理 PDF 数据并输出结果
    do_parse(
        output_dir,
        pdf_file_name,
        pdf_data,
        jso['doc_layout_result'],
        method,
        False,
        f_dump_content_list=True,
        f_draw_model_bbox=True,
    )


# 定义 PDF 命令
@cli.command()
@click.option(
    '-p',
    '--pdf',
    'pdf',
    type=click.Path(exists=True),
    required=True,
    help='本地 PDF 文件',
)
@click.option(
    '-j',
    '--json',
    'json_data',
    type=click.Path(exists=True),
    required=True,
    help='本地模型推理出的 json 数据',
)
@click.option('-o',
              '--output-dir',
              'output_dir',
              type=click.Path(),
              required=True,
              help='本地输出目录')
# 使用 Click 库定义命令行选项，指定解析方法
@click.option(
    '-m',
    '--method',
    'method',
    type=parse_pdf_methods,  # 指定选项类型为解析 PDF 方法
    help='指定解析方法。txt: 文本型 pdf 解析方法， ocr: 光学识别解析 pdf, auto: 程序智能选择解析方法',  # 提供帮助信息
    default='auto',  # 设置默认解析方法为 auto
)
# 定义 pdf 函数，接收 pdf 文件路径、json 数据、输出目录和解析方法作为参数
def pdf(pdf, json_data, output_dir, method):
    model_config.__use_inside_model__ = False  # 禁用内部模型使用配置
    # 获取 pdf 文件的绝对路径
    full_pdf_path = os.path.realpath(pdf)
    # 创建输出目录，如果已存在则不报错
    os.makedirs(output_dir, exist_ok=True)

    # 定义读取文件的内部函数
    def read_fn(path):
        # 创建磁盘读写对象，使用文件所在目录
        disk_rw = DiskReaderWriter(os.path.dirname(path))
        # 读取指定文件并返回其内容，模式为二进制
        return disk_rw.read(os.path.basename(path), AbsReaderWriter.MODE_BIN)

    # 读取 json 数据并解析成模型 JSON 列表
    model_json_list = json_parse.loads(read_fn(json_data).decode('utf-8'))

    # 获取 pdf 文件名（不包含扩展名）
    file_name = str(Path(full_pdf_path).stem)
    # 读取 pdf 文件数据
    pdf_data = read_fn(full_pdf_path)
    # 调用解析函数，进行 pdf 数据解析并输出结果
    do_parse(
        output_dir,  # 输出目录
        file_name,  # 文件名
        pdf_data,  # pdf 数据
        model_json_list,  # 模型 JSON 列表
        method,  # 解析方法
        False,  # 不启用某个选项
        f_dump_content_list=True,  # 启用内容列表转储
        f_draw_model_bbox=True,  # 启用模型边界框绘制
    )

# 如果当前模块为主程序，则调用 CLI 函数
if __name__ == '__main__':
    cli()  # 启动命令行界面
```
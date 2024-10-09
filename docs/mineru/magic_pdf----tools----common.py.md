# `.\MinerU\magic_pdf\tools\common.py`

```
# 导入所需的标准库和第三方库
import copy  # 用于深拷贝对象
import json as json_parse  # 用于 JSON 数据解析
import os  # 用于操作文件和目录

import click  # 用于命令行界面工具
from loguru import logger  # 用于日志记录

# 导入模型配置和相关库
import magic_pdf.model as model_config
from magic_pdf.libs.draw_bbox import (draw_layout_bbox, draw_span_bbox,
                                      drow_model_bbox)  # 导入绘制边界框的函数
from magic_pdf.libs.MakeContentConfig import DropMode, MakeMode  # 导入内容配置模式
from magic_pdf.pipe.OCRPipe import OCRPipe  # 导入 OCR 处理管道
from magic_pdf.pipe.TXTPipe import TXTPipe  # 导入文本处理管道
from magic_pdf.pipe.UNIPipe import UNIPipe  # 导入统一处理管道
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter  # 导入抽象读写器
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter  # 导入磁盘读写器


# 准备环境，创建必要的目录结构
def prepare_env(output_dir, pdf_file_name, method):
    # 根据输出目录、PDF 文件名和方法创建本地父目录
    local_parent_dir = os.path.join(output_dir, pdf_file_name, method)

    # 创建用于存放图像的本地目录
    local_image_dir = os.path.join(str(local_parent_dir), 'images')
    # 本地元数据目录设置为本地父目录
    local_md_dir = local_parent_dir
    # 创建图像目录，如果目录已存在则不报错
    os.makedirs(local_image_dir, exist_ok=True)
    # 创建元数据目录，如果目录已存在则不报错
    os.makedirs(local_md_dir, exist_ok=True)
    # 返回图像目录和元数据目录
    return local_image_dir, local_md_dir


# 解析 PDF 文件的主函数
def do_parse(
    output_dir,  # 输出目录
    pdf_file_name,  # PDF 文件名
    pdf_bytes,  # PDF 文件的字节内容
    model_list,  # 模型列表
    parse_method,  # 解析方法
    debug_able,  # 调试开关
    f_draw_span_bbox=True,  # 是否绘制跨度边界框
    f_draw_layout_bbox=True,  # 是否绘制布局边界框
    f_dump_md=True,  # 是否导出元数据
    f_dump_middle_json=True,  # 是否导出中间 JSON
    f_dump_model_json=True,  # 是否导出模型 JSON
    f_dump_orig_pdf=True,  # 是否导出原始 PDF
    f_dump_content_list=False,  # 是否导出内容列表
    f_make_md_mode=MakeMode.MM_MD,  # 元数据生成模式
    f_draw_model_bbox=False,  # 是否绘制模型边界框
    start_page_id=0,  # 起始页面 ID
    end_page_id=None,  # 结束页面 ID
):
    # 如果启用调试模式，记录警告并调整导出选项
    if debug_able:
        logger.warning('debug mode is on')
        f_dump_content_list = True  # 启用内容列表导出
        f_draw_model_bbox = True  # 启用模型边界框绘制

    # 深拷贝模型列表以保留原始数据
    orig_model_list = copy.deepcopy(model_list)
    # 调用准备环境函数获取图像和元数据目录
    local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name,
                                                parse_method)

    # 创建图像和元数据的读写器实例
    image_writer, md_writer = DiskReaderWriter(
        local_image_dir), DiskReaderWriter(local_md_dir)
    # 获取图像目录的基础名称
    image_dir = str(os.path.basename(local_image_dir))

    # 根据解析方法选择相应的处理管道
    if parse_method == 'auto':
        # 定义有用的 JSON 键
        jso_useful_key = {'_pdf_type': '', 'model_list': model_list}
        # 创建统一处理管道
        pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer, is_debug=True,
                       start_page_id=start_page_id, end_page_id=end_page_id)
    elif parse_method == 'txt':
        # 创建文本处理管道
        pipe = TXTPipe(pdf_bytes, model_list, image_writer, is_debug=True,
                       start_page_id=start_page_id, end_page_id=end_page_id)
    elif parse_method == 'ocr':
        # 创建 OCR 处理管道
        pipe = OCRPipe(pdf_bytes, model_list, image_writer, is_debug=True,
                       start_page_id=start_page_id, end_page_id=end_page_id)
    else:
        # 如果解析方法未知，记录错误并退出程序
        logger.error('unknown parse method')
        exit(1)

    # 执行分类处理
    pipe.pipe_classify()

    # 如果模型列表为空，检查是否使用内部模型
    if len(model_list) == 0:
        if model_config.__use_inside_model__:
            # 如果使用内部模型，执行分析处理
            pipe.pipe_analyze()
            # 深拷贝处理后模型列表
            orig_model_list = copy.deepcopy(pipe.model_list)
        else:
            # 如果未提供模型列表，记录错误并退出程序
            logger.error('need model list input')
            exit(2)

    # 执行解析处理
    pipe.pipe_parse()
    # 从管道中获取 PDF 信息
    pdf_info = pipe.pdf_mid_data['pdf_info']
    # 如果启用布局边界框绘制，则执行绘制
    if f_draw_layout_bbox:
        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, pdf_file_name)
    # 如果 f_draw_span_bbox 为真，则绘制跨度边界框
    if f_draw_span_bbox:
        # 调用函数绘制跨度边界框
        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, pdf_file_name)
    # 如果 f_draw_model_bbox 为真，则绘制模型边界框
    if f_draw_model_bbox:
        # 深拷贝原始模型列表并绘制模型边界框
        drow_model_bbox(copy.deepcopy(orig_model_list), pdf_bytes, local_md_dir, pdf_file_name)

    # 生成 Markdown 内容，使用指定的丢弃模式和 Markdown 制作模式
    md_content = pipe.pipe_mk_markdown(image_dir,
                                       drop_mode=DropMode.NONE,
                                       md_make_mode=f_make_md_mode)
    # 如果 f_dump_md 为真，则将 Markdown 内容写入文件
    if f_dump_md:
        md_writer.write(
            # 写入内容、路径和模式
            content=md_content,
            path=f'{pdf_file_name}.md',
            mode=AbsReaderWriter.MODE_TXT,
        )

    # 如果 f_dump_middle_json 为真，则将中间 JSON 数据写入文件
    if f_dump_middle_json:
        md_writer.write(
            content=json_parse.dumps(pipe.pdf_mid_data,
                                     ensure_ascii=False,
                                     indent=4),
            # 指定 JSON 文件路径
            path=f'{pdf_file_name}_middle.json',
            mode=AbsReaderWriter.MODE_TXT,
        )

    # 如果 f_dump_model_json 为真，则将原始模型列表写入 JSON 文件
    if f_dump_model_json:
        md_writer.write(
            content=json_parse.dumps(orig_model_list,
                                     ensure_ascii=False,
                                     indent=4),
            # 指定模型 JSON 文件路径
            path=f'{pdf_file_name}_model.json',
            mode=AbsReaderWriter.MODE_TXT,
        )

    # 如果 f_dump_orig_pdf 为真，则将原始 PDF 数据写入文件
    if f_dump_orig_pdf:
        md_writer.write(
            content=pdf_bytes,
            # 指定原始 PDF 文件路径
            path=f'{pdf_file_name}_origin.pdf',
            mode=AbsReaderWriter.MODE_BIN,
        )

    # 生成统一格式的内容列表
    content_list = pipe.pipe_mk_uni_format(image_dir, drop_mode=DropMode.NONE)
    # 如果 f_dump_content_list 为真，则将内容列表写入 JSON 文件
    if f_dump_content_list:
        md_writer.write(
            content=json_parse.dumps(content_list,
                                     ensure_ascii=False,
                                     indent=4),
            # 指定内容列表 JSON 文件路径
            path=f'{pdf_file_name}_content_list.json',
            mode=AbsReaderWriter.MODE_TXT,
        )

    # 记录本地输出目录信息
    logger.info(f'local output dir is {local_md_dir}')
# 定义一个可选项列表，用于选择解析 PDF 的方法
parse_pdf_methods = click.Choice(['ocr', 'txt', 'auto'])
```
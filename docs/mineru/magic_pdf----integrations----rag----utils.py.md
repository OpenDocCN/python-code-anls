# `.\MinerU\magic_pdf\integrations\rag\utils.py`

```
# 导入 json 模块，用于处理 JSON 数据
import json
# 导入 os 模块，用于处理文件和目录
import os
# 从 pathlib 导入 Path 类，用于方便处理路径
from pathlib import Path

# 从 loguru 导入 logger，用于日志记录
from loguru import logger

# 导入模型配置
import magic_pdf.model as model_config
# 从 ocr_mkcontent 导入合并段落与文本的函数
from magic_pdf.dict2md.ocr_mkcontent import merge_para_with_text
# 从 rag.type 导入各种类型，用于内容处理
from magic_pdf.integrations.rag.type import (CategoryType, ContentObject,
                                             ElementRelation, ElementRelType,
                                             LayoutElements,
                                             LayoutElementsExtra, PageInfo)
# 从 ocr_content_type 导入块类型和内容类型
from magic_pdf.libs.ocr_content_type import BlockType, ContentType
# 从 rw 模块导入抽象读写器和磁盘读写器
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
# 从工具模块导入解析和环境准备函数
from magic_pdf.tools.common import do_parse, prepare_env


# 定义将中间 JSON 数据转换为布局元素的函数
def convert_middle_json_to_layout_elements(
    json_data: dict,  # 输入的 JSON 数据
    output_dir: str,  # 输出目录
) -> list[LayoutElements]:  # 返回一个 LayoutElements 类型的列表
    uniq_anno_id = 0  # 初始化唯一注释 ID

    res: list[LayoutElements] = []  # 初始化结果列表
    return res  # 返回空列表


# 定义推理函数，处理指定路径和输出目录
def inference(path, output_dir, method):
    model_config.__use_inside_model__ = True  # 设置模型使用标志
    model_config.__model_mode__ = 'full'  # 设置模型模式为完整模式
    if output_dir == '':  # 如果输出目录为空
        if os.path.isdir(path):  # 如果路径是目录
            output_dir = os.path.join(path, 'output')  # 设置输出目录为 path 下的 'output'
        else:  # 如果路径不是目录
            output_dir = os.path.join(os.path.dirname(path), 'output')  # 设置输出目录为路径所在目录下的 'output'

    # 准备本地图像目录和本地 Markdown 目录
    local_image_dir, local_md_dir = prepare_env(output_dir,
                                                str(Path(path).stem), method)

    # 定义读取文件的函数
    def read_fn(path):
        disk_rw = DiskReaderWriter(os.path.dirname(path))  # 创建磁盘读写器实例
        return disk_rw.read(os.path.basename(path), AbsReaderWriter.MODE_BIN)  # 读取文件并返回二进制数据

    # 定义解析文档的函数
    def parse_doc(doc_path: str):
        try:
            file_name = str(Path(doc_path).stem)  # 获取文档文件名（不包含扩展名）
            pdf_data = read_fn(doc_path)  # 读取 PDF 文件的数据
            # 解析 PDF 数据并输出结果
            do_parse(
                output_dir,
                file_name,
                pdf_data,
                [],
                method,
                False,
                f_draw_span_bbox=False,
                f_draw_layout_bbox=False,
                f_dump_md=False,
                f_dump_middle_json=True,
                f_dump_model_json=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False,
                f_draw_model_bbox=False,
            )

            middle_json_fn = os.path.join(local_md_dir,
                                          f'{file_name}_middle.json')  # 构建中间 JSON 文件名
            with open(middle_json_fn) as fd:  # 打开中间 JSON 文件
                jso = json.load(fd)  # 读取并解析 JSON 数据
            os.remove(middle_json_fn)  # 删除中间 JSON 文件
            return convert_middle_json_to_layout_elements(jso, local_image_dir)  # 转换 JSON 数据为布局元素并返回

        except Exception as e:  # 捕获异常
            logger.exception(e)  # 记录异常信息

    return parse_doc(path)  # 调用解析文档函数并返回结果


# 当脚本作为主程序运行时执行
if __name__ == '__main__':
    import pprint  # 导入 pprint 模块，用于美化打印

    base_dir = '/opt/data/pdf/resources/samples/'  # 设置基础目录
    if 0:  # 这个条件永远为假，不会执行
        with open(base_dir + 'json_outputs/middle.json') as f:  # 打开中间 JSON 文件
            d = json.load(f)  # 读取并解析 JSON 数据
        result = convert_middle_json_to_layout_elements(d, '/tmp')  # 转换 JSON 数据为布局元素
        pprint.pp(result)  # 美化打印结果
    # 如果条件为 0，则执行下面的代码块
        if 0:
            # 打开指定路径的 JSON 文件，文件对象命名为 f
            with open(base_dir + 'json_outputs/middle.3.json') as f:
                # 从文件中加载 JSON 数据并赋值给 d
                d = json.load(f)
            # 将 JSON 数据转换为布局元素，并保存结果到 result
            result = convert_middle_json_to_layout_elements(d, '/tmp')
            # 美化打印结果
            pprint.pp(result)
    
        # 如果条件为 1，则执行下面的代码块
        if 1:
            # 调用 inference 函数处理指定的 PDF 文件，并将输出结果保存到 res
            res = inference(
                base_dir + 'samples/pdf/one_page_with_table_image.pdf',  # PDF 文件路径
                '/tmp/output',  # 输出目录
                'ocr',  # 使用的处理方法
            )
            # 美化打印处理结果
            pprint.pp(res)
```
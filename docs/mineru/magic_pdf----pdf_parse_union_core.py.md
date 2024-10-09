# `.\MinerU\magic_pdf\pdf_parse_union_core.py`

```
# 导入时间模块
import time

# 导入日志记录工具
from loguru import logger

# 从库中导入所需的功能
from magic_pdf.libs.commons import fitz, get_delta_time
from magic_pdf.layout.layout_sort import get_bboxes_layout, LAYOUT_UNPROC, get_columns_cnt_of_layout
from magic_pdf.libs.convert_utils import dict_to_list
from magic_pdf.libs.drop_reason import DropReason
from magic_pdf.libs.hash_utils import compute_md5
from magic_pdf.libs.local_math import float_equal
from magic_pdf.libs.ocr_content_type import ContentType
from magic_pdf.model.magic_model import MagicModel
from magic_pdf.para.para_split_v2 import para_split
from magic_pdf.pre_proc.citationmarker_remove import remove_citation_marker
from magic_pdf.pre_proc.construct_page_dict import ocr_construct_page_component_v2
from magic_pdf.pre_proc.cut_image import ocr_cut_image_and_table
from magic_pdf.pre_proc.equations_replace import remove_chars_in_text_blocks, replace_equations_in_textblock, \
    combine_chars_to_pymudict
from magic_pdf.pre_proc.ocr_detect_all_bboxes import ocr_prepare_bboxes_for_layout_split
from magic_pdf.pre_proc.ocr_dict_merge import sort_blocks_by_layout, fill_spans_in_blocks, fix_block_spans, \
    fix_discarded_block
from magic_pdf.pre_proc.ocr_span_list_modify import remove_overlaps_min_spans, get_qa_need_list_v2, \
    remove_overlaps_low_confidence_spans
from magic_pdf.pre_proc.resolve_bbox_conflict import check_useful_block_horizontal_overlap


# 移除较小的水平重叠块
def remove_horizontal_overlap_block_which_smaller(all_bboxes):
    useful_blocks = []  # 创建存储有用块的列表
    for bbox in all_bboxes:  # 遍历所有边界框
        useful_blocks.append({  # 将边界框的前四个元素添加到有用块中
            "bbox": bbox[:4]
        })
    # 检查水平重叠块
    is_useful_block_horz_overlap, smaller_bbox, bigger_bbox = check_useful_block_horizontal_overlap(useful_blocks)
    if is_useful_block_horz_overlap:  # 如果存在有用的水平重叠块
        logger.warning(  # 记录警告信息
            f"skip this page, reason: {DropReason.USEFUL_BLOCK_HOR_OVERLAP}, smaller bbox is {smaller_bbox}, bigger bbox is {bigger_bbox}")
        for bbox in all_bboxes.copy():  # 遍历所有边界框的副本
            if smaller_bbox == bbox[:4]:  # 如果边界框等于较小的边界框
                all_bboxes.remove(bbox)  # 从列表中移除该边界框

    return is_useful_block_horz_overlap, all_bboxes  # 返回结果


# 替换特定字符函数
def __replace_STX_ETX(text_str:str):
    """ 替换\u0002和\u0003，因为这些字符在使用pymupdf提取时会变得乱码，实际上它们原本是引号。
缺点：此问题仅在英文文本中观察到，目前尚未在中文文本中发现。

    Args:
        text_str (str): 原始文本

    Returns:
        _type_: 替换后的文本
    """
    if text_str:  # 如果文本字符串不为空
        s = text_str.replace('\u0002', "'")  # 替换\u0002为单引号
        s = s.replace("\u0003", "'")  # 替换\u0003为单引号
        return s  # 返回替换后的文本
    return text_str  # 如果文本为空，返回原文本


# 提取文本块的函数
def txt_spans_extract(pdf_page, inline_equations, interline_equations):
    # 获取PDF页面的文本块
    text_raw_blocks = pdf_page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
    # 获取字符级别的文本块
    char_level_text_blocks = pdf_page.get_text("rawdict", flags=fitz.TEXTFLAGS_TEXT)[
        "blocks"
    ]
    # 将字符级文本块合并到一个字典中
    text_blocks = combine_chars_to_pymudict(text_raw_blocks, char_level_text_blocks)
    # 用给定的行内和行间方程替换文本块中的方程
        text_blocks = replace_equations_in_textblock(
            text_blocks, inline_equations, interline_equations
        )
        # 移除文本块中的引用标记
        text_blocks = remove_citation_marker(text_blocks)
        # 从文本块中移除特定字符
        text_blocks = remove_chars_in_text_blocks(text_blocks)
        # 初始化一个空列表以存储提取的跨度信息
        spans = []
        # 遍历每个文本块
        for v in text_blocks:
            # 遍历每个文本块中的行
            for line in v["lines"]:
                # 遍历每行中的跨度
                for span in line["spans"]:
                    # 获取跨度的边界框
                    bbox = span["bbox"]
                    # 如果边界框的宽度或高度为0，跳过该跨度
                    if float_equal(bbox[0], bbox[2]) or float_equal(bbox[1], bbox[3]):
                        continue
                    # 如果跨度的类型不是行内方程或行间方程
                    if span.get('type') not in (ContentType.InlineEquation, ContentType.InterlineEquation):
                        # 将跨度信息添加到列表中
                        spans.append(
                            {
                                # 将边界框转换为列表
                                "bbox": list(span["bbox"]),
                                # 替换文本中的 STX 和 ETX 字符
                                "content": __replace_STX_ETX(span["text"]),
                                # 设置类型为文本
                                "type": ContentType.Text,
                                # 设定默认分数为1.0
                                "score": 1.0,
                            }
                        )
        # 返回所有提取的跨度信息
        return spans
# 替换 OCR spans 中的文本类型 spans，返回不包含文本类型的 spans 和 pymu spans 的组合
def replace_text_span(pymu_spans, ocr_spans):
    # 过滤掉 OCR spans 中类型为文本的项，并与 pymu spans 合并
    return list(filter(lambda x: x["type"] != ContentType.Text, ocr_spans)) + pymu_spans


# 解析 PDF 页面核心逻辑
def parse_page_core(pdf_docs, magic_model, page_id, pdf_bytes_md5, imageWriter, parse_mode):
    # 标记是否需要丢弃该页面
    need_drop = False
    # 存储丢弃的原因
    drop_reason = []

    '''从magic_model对象中获取后面会用到的区块信息'''
    # 获取当前页面的图片块
    img_blocks = magic_model.get_imgs(page_id)
    # 获取当前页面的表格块
    table_blocks = magic_model.get_tables(page_id)
    # 获取被丢弃的块
    discarded_blocks = magic_model.get_discarded(page_id)
    # 获取文本块
    text_blocks = magic_model.get_text_blocks(page_id)
    # 获取标题块
    title_blocks = magic_model.get_title_blocks(page_id)
    # 获取行内和行间公式及其块
    inline_equations, interline_equations, interline_equation_blocks = magic_model.get_equations(page_id)

    # 获取页面的宽度和高度
    page_w, page_h = magic_model.get_page_size(page_id)

    # 获取当前页面的所有 spans
    spans = magic_model.get_all_spans(page_id)

    '''根据parse_mode，构造spans'''
    # 如果解析模式是文本
    if parse_mode == "txt":
        """ocr 中文本类的 span 用 pymu spans 替换！"""
        # 提取当前页面的文本类 spans
        pymu_spans = txt_spans_extract(
            pdf_docs[page_id], inline_equations, interline_equations
        )
        # 替换 OCR spans 中的文本类 spans
        spans = replace_text_span(pymu_spans, spans)
    # 如果解析模式是 OCR，什么都不做
    elif parse_mode == "ocr":
        pass
    # 如果解析模式不正确，抛出异常
    else:
        raise Exception("parse_mode must be txt or ocr")

    '''删除重叠spans中置信度较低的那些'''
    # 根据置信度移除重叠的 spans
    spans, dropped_spans_by_confidence = remove_overlaps_low_confidence_spans(spans)
    '''删除重叠spans中较小的那些'''
    # 移除较小的重叠 spans
    spans, dropped_spans_by_span_overlap = remove_overlaps_min_spans(spans)
    '''对image和table截图'''
    # 对图片和表格进行截图，返回处理后的 spans
    spans = ocr_cut_image_and_table(spans, pdf_docs[page_id], page_id, pdf_bytes_md5, imageWriter)

    '''将所有区块的bbox整理到一起'''
    # interline_equation_blocks参数不够准，后面切换到interline_equations上
    interline_equation_blocks = []
    # 如果有行间公式块，整理所有边界框
    if len(interline_equation_blocks) > 0:
        all_bboxes, all_discarded_blocks, drop_reasons = ocr_prepare_bboxes_for_layout_split(
            img_blocks, table_blocks, discarded_blocks, text_blocks, title_blocks,
            interline_equation_blocks, page_w, page_h)
    # 否则使用行间公式整理
    else:
        all_bboxes, all_discarded_blocks, drop_reasons = ocr_prepare_bboxes_for_layout_split(
            img_blocks, table_blocks, discarded_blocks, text_blocks, title_blocks,
            interline_equations, page_w, page_h)

    # 如果有丢弃原因，则需要丢弃该页面
    if len(drop_reasons) > 0:
        need_drop = True
        drop_reason.append(DropReason.OVERLAP_BLOCKS_CAN_NOT_SEPARATION)

    '''先处理不需要排版的discarded_blocks'''
    # 填充不需要排版的块中的 spans
    discarded_block_with_spans, spans = fill_spans_in_blocks(all_discarded_blocks, spans, 0.4)
    # 修复被丢弃的块
    fix_discarded_blocks = fix_discarded_block(discarded_block_with_spans)

    '''如果当前页面没有bbox则跳过'''
    # 如果没有找到有用的边界框，则跳过该页面
    if len(all_bboxes) == 0:
        logger.warning(f"skip this page, not found useful bbox, page_id: {page_id}")
        return ocr_construct_page_component_v2([], [], page_id, page_w, page_h, [],
                                               [], [], interline_equations, fix_discarded_blocks,
                                               need_drop, drop_reason)

    """在切分之前，先检查一下bbox是否有左右重叠的情况，如果有，那么就认为这个pdf暂时没有能力处理好，这种左右重叠的情况大概率是由于pdf里的行间公式、表格没有被正确识别出来造成的 """
    # 循环检查左右重叠的情况，如果存在就删除掉较小的那个bbox，直到不存在左右重叠的情况
    while True:  
        # 调用函数检查并移除较小的水平重叠块，并返回是否有用块和所有的边界框
        is_useful_block_horz_overlap, all_bboxes = remove_horizontal_overlap_block_which_smaller(all_bboxes)
        # 如果存在有用的块，则标记需要丢弃，并记录丢弃原因
        if is_useful_block_horz_overlap:
            need_drop = True
            drop_reason.append(DropReason.USEFUL_BLOCK_HOR_OVERLAP)
        else:
            # 如果没有有用的块，退出循环
            break

    # 根据区块信息计算页面布局
    page_boundry = [0, 0, page_w, page_h]  # 定义页面边界
    # 获取布局边界框和布局树
    layout_bboxes, layout_tree = get_bboxes_layout(all_bboxes, page_boundry, page_id)

    # 检查文本块和边界框的数量，如果文本块存在但边界框为空，则记录警告并标记为需要丢弃
    if len(text_blocks) > 0 and len(all_bboxes) > 0 and len(layout_bboxes) == 0:
        logger.warning(
            f"skip this page, page_id: {page_id}, reason: {DropReason.CAN_NOT_DETECT_PAGE_LAYOUT}")
        need_drop = True
        drop_reason.append(DropReason.CAN_NOT_DETECT_PAGE_LAYOUT)

    # 以下去掉复杂的布局和超过2列的布局
    # 检查布局边界框中是否存在复杂布局标记
    if any([lay["layout_label"] == LAYOUT_UNPROC for lay in layout_bboxes]):  # 复杂的布局
        logger.warning(
            f"skip this page, page_id: {page_id}, reason: {DropReason.COMPLICATED_LAYOUT}")
        need_drop = True
        drop_reason.append(DropReason.COMPLICATED_LAYOUT)

    # 获取布局的列宽
    layout_column_width = get_columns_cnt_of_layout(layout_tree)
    # 如果列宽超过2列，则记录警告并标记为需要丢弃
    if layout_column_width > 2:  # 去掉超过2列的布局pdf
        logger.warning(
            f"skip this page, page_id: {page_id}, reason: {DropReason.TOO_MANY_LAYOUT_COLUMNS}")
        need_drop = True
        drop_reason.append(DropReason.TOO_MANY_LAYOUT_COLUMNS)

    # 根据layout顺序，对当前页面所有需要留下的block进行排序
    sorted_blocks = sort_blocks_by_layout(all_bboxes, layout_bboxes)

    # 将span填入排好序的blocks中
    block_with_spans, spans = fill_spans_in_blocks(sorted_blocks, spans, 0.3)

    # 对block进行fix操作
    fix_blocks = fix_block_spans(block_with_spans, img_blocks, table_blocks)

    # 获取QA需要外置的list
    images, tables, interline_equations = get_qa_need_list_v2(fix_blocks)

    # 构造pdf_info_dict
    page_info = ocr_construct_page_component_v2(fix_blocks, layout_bboxes, page_id, page_w, page_h, layout_tree,
                                                images, tables, interline_equations, fix_discarded_blocks,
                                                need_drop, drop_reason)
    # 返回页面信息字典
    return page_info
# 定义解析 PDF 的函数，接收 PDF 字节流及其他参数
def pdf_parse_union(pdf_bytes,
                    model_list,
                    imageWriter,
                    parse_mode,
                    start_page_id=0,
                    end_page_id=None,
                    debug_mode=False,
                    ):
    # 计算 PDF 字节流的 MD5 哈希值
    pdf_bytes_md5 = compute_md5(pdf_bytes)
    # 打开 PDF 字节流，创建 PDF 文档对象
    pdf_docs = fitz.open("pdf", pdf_bytes)

    '''初始化空的pdf_info_dict'''
    # 创建一个空字典，用于存储解析后的 PDF 信息
    pdf_info_dict = {}

    '''用model_list和docs对象初始化magic_model'''
    # 使用模型列表和 PDF 文档对象初始化魔法模型
    magic_model = MagicModel(model_list, pdf_docs)

    '''根据输入的起始范围解析pdf'''
    # 确定结束页码，若未指定则默认为最后一页
    end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else len(pdf_docs) - 1

    # 检查结束页码是否超出文档范围，若超出则发出警告并调整
    if end_page_id > len(pdf_docs) - 1:
        logger.warning("end_page_id is out of range, use pdf_docs length")
        end_page_id = len(pdf_docs) - 1

    '''初始化启动时间'''
    # 记录开始解析的时间，用于计算耗时
    start_time = time.time()

    # 遍历 PDF 文档中的每一页
    for page_id, page in enumerate(pdf_docs):
        '''debug时输出每页解析的耗时'''
        # 若启用调试模式，输出当前页面解析的耗时信息
        if debug_mode:
            time_now = time.time()
            logger.info(
                f"page_id: {page_id}, last_page_cost_time: {get_delta_time(start_time)}"
            )
            # 更新开始时间为当前时间
            start_time = time_now

        '''解析pdf中的每一页'''
        # 根据起始和结束页码解析指定范围的页
        if start_page_id <= page_id <= end_page_id:
            page_info = parse_page_core(pdf_docs, magic_model, page_id, pdf_bytes_md5, imageWriter, parse_mode)
        else:
            # 获取当前页面的宽高信息
            page_w = page.rect.width
            page_h = page.rect.height
            # 构造一个跳过当前页面的 OCR 组件
            page_info = ocr_construct_page_component_v2([], [], page_id, page_w, page_h, [],
                                                [], [], [], [],
                                                True, "skip page")
        # 将当前页面的信息存入字典
        pdf_info_dict[f"page_{page_id}"] = page_info

    """分段"""
    # 对解析得到的 PDF 信息进行段落分割
    para_split(pdf_info_dict, debug_mode=debug_mode)

    """dict转list"""
    # 将字典格式的 PDF 信息转换为列表格式
    pdf_info_list = dict_to_list(pdf_info_dict)
    # 创建一个新的字典，包含转换后的 PDF 信息列表
    new_pdf_info_dict = {
        "pdf_info": pdf_info_list,
    }

    # 返回包含 PDF 信息的字典
    return new_pdf_info_dict


# 主程序入口
if __name__ == '__main__':
    # 不执行任何操作，保持程序的可扩展性
    pass
```
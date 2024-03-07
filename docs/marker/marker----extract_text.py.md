# `.\marker\marker\extract_text.py`

```py
# 导入所需的模块
import os
from typing import Tuple, List, Optional

# 导入拼写检查器 SpellChecker
from spellchecker import SpellChecker

# 导入正确旋转的边界框函数
from marker.bbox import correct_rotation
# 导入整页 OCR 函数
from marker.ocr.page import ocr_entire_page
# 导入检测不良 OCR 的工具函数和字体标志分解器
from marker.ocr.utils import detect_bad_ocr, font_flags_decomposer
# 导入设置模块中的设置
from marker.settings import settings
# 导入 Span, Line, Block, Page 数据结构
from marker.schema import Span, Line, Block, Page
# 导入线程池执行器
from concurrent.futures import ThreadPoolExecutor

# 设置环境变量 TESSDATA_PREFIX 为设置模块中的 TESSDATA_PREFIX
os.environ["TESSDATA_PREFIX"] = settings.TESSDATA_PREFIX

# 根据垂直分组对旋转文本进行排序
def sort_rotated_text(page_blocks, tolerance=1.25):
    vertical_groups = {}
    for block in page_blocks:
        group_key = round(block.bbox[1] / tolerance) * tolerance
        if group_key not in vertical_groups:
            vertical_groups[group_key] = []
        vertical_groups[group_key].append(block)

    # 对每个组进行水平排序，并将组展平为一个列表
    sorted_page_blocks = []
    for _, group in sorted(vertical_groups.items()):
        sorted_group = sorted(group, key=lambda x: x.bbox[0])
        sorted_page_blocks.extend(sorted_group)

    return sorted_page_blocks

# 获取单个页面的块信息
def get_single_page_blocks(doc, pnum: int, tess_lang: str, spellchecker: Optional[SpellChecker] = None, ocr=False) -> Tuple[List[Block], int]:
    # 获取文档中指定页码的页面
    page = doc[pnum]
    # 获取页面的旋转角度
    rotation = page.rotation

    # 如果需要进行 OCR
    if ocr:
        # 对整个页面进行 OCR，使用指定的语言和拼写检查器
        blocks = ocr_entire_page(page, tess_lang, spellchecker)
    else:
        # 否则，获取页面的文本块信息，按照设置中的标志进行排序
        blocks = page.get_text("dict", sort=True, flags=settings.TEXT_FLAGS)["blocks"]

    # 初始化页面块列表和跨度 ID
    page_blocks = []
    span_id = 0
    # 遍历每个块的索引和块内容
    for block_idx, block in enumerate(blocks):
        # 初始化存储每个块中行的列表
        block_lines = []
        # 遍历每个块中的行
        for l in block["lines"]:
            # 初始化存储每个行中span的列表
            spans = []
            # 遍历每个span
            for i, s in enumerate(l["spans"]):
                # 获取span的文本内容和边界框
                block_text = s["text"]
                bbox = s["bbox"]
                # 创建Span对象，包括文本内容、边界框、span id、字体和颜色等信息
                span_obj = Span(
                    text=block_text,
                    bbox=correct_rotation(bbox, page),
                    span_id=f"{pnum}_{span_id}",
                    font=f"{s['font']}_{font_flags_decomposer(s['flags'])}", # 在字体后面添加字体标志
                    color=s["color"],
                    ascender=s["ascender"],
                    descender=s["descender"],
                )
                spans.append(span_obj)  # 将span对象添加到spans列表中
                span_id += 1
            # 创建Line对象，包括spans列表和边界框
            line_obj = Line(
                spans=spans,
                bbox=correct_rotation(l["bbox"], page),
            )
            # 只选择有效的行，即边界框面积大于0的行
            if line_obj.area > 0:
                block_lines.append(line_obj)  # 将有效的行添加到block_lines列表中
        # 创建Block对象，包括lines列表和边界框
        block_obj = Block(
            lines=block_lines,
            bbox=correct_rotation(block["bbox"], page),
            pnum=pnum
        )
        # 只选择包含多行的块
        if len(block_lines) > 0:
            page_blocks.append(block_obj)  # 将包含多行的块添加到page_blocks列表中

    # 如果页面被旋转，重新对文本进行排序
    if rotation > 0:
        page_blocks = sort_rotated_text(page_blocks)
    return page_blocks  # 返回处理后的页面块列表
# 将单个页面转换为文本块，进行 OCR 处理
def convert_single_page(doc, pnum, tess_lang: str, spell_lang: Optional[str], no_text: bool, disable_ocr: bool = False, min_ocr_page: int = 2):
    # 初始化变量用于记录 OCR 页面数量、成功次数和失败次数
    ocr_pages = 0
    ocr_success = 0
    ocr_failed = 0
    spellchecker = None
    # 获取当前页面的边界框
    page_bbox = doc[pnum].bound()
    # 如果指定了拼写检查语言，则创建拼写检查器对象
    if spell_lang:
        spellchecker = SpellChecker(language=spell_lang)

    # 获取单个页面的文本块
    blocks = get_single_page_blocks(doc, pnum, tess_lang, spellchecker)
    # 创建页面对象，包含文本块、页面编号和边界框
    page_obj = Page(blocks=blocks, pnum=pnum, bbox=page_bbox)

    # 判断是否需要对页面进行 OCR 处理
    conditions = [
        (
            no_text  # 全文本为空，需要进行完整 OCR 处理
            or
            (len(page_obj.prelim_text) > 0 and detect_bad_ocr(page_obj.prelim_text, spellchecker))  # OCR 处理不佳
        ),
        min_ocr_page < pnum < len(doc) - 1,
        not disable_ocr
    ]
    if all(conditions) or settings.OCR_ALL_PAGES:
        # 获取当前页面对象
        page = doc[pnum]
        # 获取包含 OCR 处理的文本块
        blocks = get_single_page_blocks(doc, pnum, tess_lang, spellchecker, ocr=True)
        # 创建包含 OCR 处理的页面对象，包含文本块、页面编号、边界框和旋转信息
        page_obj = Page(blocks=blocks, pnum=pnum, bbox=page_bbox, rotation=page.rotation)
        ocr_pages = 1
        if len(blocks) == 0:
            ocr_failed = 1
        else:
            ocr_success = 1
    # 返回页面对象和 OCR 处理结果统计信息
    return page_obj, {"ocr_pages": ocr_pages, "ocr_failed": ocr_failed, "ocr_success": ocr_success}


# 获取文本块列表
def get_text_blocks(doc, tess_lang: str, spell_lang: Optional[str], max_pages: Optional[int] = None, parallel: int = settings.OCR_PARALLEL_WORKERS):
    all_blocks = []
    # 获取文档的目录
    toc = doc.get_toc()
    ocr_pages = 0
    ocr_failed = 0
    ocr_success = 0
    # 这是一个线程，因为大部分工作在一个单独的进程中进行（tesseract）
    range_end = len(doc)
    # 判断是否全文本为空
    no_text = len(naive_get_text(doc).strip()) == 0
    # 如果指定了最大页面数，则限制范围
    if max_pages:
        range_end = min(max_pages, len(doc))
    # 使用线程池执行并行任务，最大工作线程数为 parallel
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        # 生成参数列表，包含文档、页数、Tesseract语言、拼写语言、是否无文本的元组
        args_list = [(doc, pnum, tess_lang, spell_lang, no_text) for pnum in range(range_end)]
        # 根据并行数选择使用 map 函数或线程池的 map 函数
        if parallel == 1:
            func = map
        else:
            func = pool.map
        # 执行函数并获取结果
        results = func(lambda a: convert_single_page(*a), args_list)
    
        # 遍历结果
        for result in results:
            # 获取页面对象和 OCR 统计信息
            page_obj, ocr_stats = result
            # 将页面对象添加到所有块列表中
            all_blocks.append(page_obj)
            # 更新 OCR 页面数、失败数和成功数
            ocr_pages += ocr_stats["ocr_pages"]
            ocr_failed += ocr_stats["ocr_failed"]
            ocr_success += ocr_stats["ocr_success"]
    
    # 返回所有块列表、目录和 OCR 统计信息
    return all_blocks, toc, {"ocr_pages": ocr_pages, "ocr_failed": ocr_failed, "ocr_success": ocr_success}
# 定义一个函数，用于从文档中提取文本内容
def naive_get_text(doc):
    # 初始化一个空字符串，用于存储提取的文本内容
    full_text = ""
    # 遍历文档中的每一页
    for page in doc:
        # 获取当前页的文本内容，并按照指定的参数进行排序和处理
        full_text += page.get_text("text", sort=True, flags=settings.TEXT_FLAGS)
        # 在每一页的文本内容后添加换行符
        full_text += "\n"
    # 返回整个文档的文本内容
    return full_text
```
# `.\marker\marker\convert.py`

```
# 导入所需的库
import fitz as pymupdf

# 导入自定义模块
from marker.cleaners.table import merge_table_blocks, create_new_tables
from marker.debug.data import dump_bbox_debug_data
from marker.extract_text import get_text_blocks
from marker.cleaners.headers import filter_header_footer, filter_common_titles
from marker.cleaners.equations import replace_equations
from marker.ordering import order_blocks
from marker.postprocessors.editor import edit_full_text
from marker.segmentation import detect_document_block_types
from marker.cleaners.code import identify_code_blocks, indent_blocks
from marker.cleaners.bullets import replace_bullets
from marker.markdown import merge_spans, merge_lines, get_full_text
from marker.schema import Page, BlockType
from typing import List, Dict, Tuple, Optional
import re
import magic
from marker.settings import settings

# 定义函数，根据文件路径获取文件类型
def find_filetype(fpath):
    # 获取文件的 MIME 类型
    mimetype = magic.from_file(fpath).lower()

    # 根据 MIME 类型判断文件类型
    if "pdf" in mimetype:
        return "pdf"
    elif "epub" in mimetype:
        return "epub"
    elif "mobi" in mimetype:
        return "mobi"
    elif mimetype in settings.SUPPORTED_FILETYPES:
        return settings.SUPPORTED_FILETYPES[mimetype]
    else:
        # 输出非标准文件类型信息
        print(f"Found nonstandard filetype {mimetype}")
        return "other"

# 定义函数，为文本块添加标注
def annotate_spans(blocks: List[Page], block_types: List[BlockType]):
    for i, page in enumerate(blocks):
        page_block_types = block_types[i]
        page.add_block_types(page_block_types)

# 定义函数，获取文本文件的长度
def get_length_of_text(fname: str) -> int:
    # 获取文件类型
    filetype = find_filetype(fname)
    # 如果文件类型为其他，则返回长度为0
    if filetype == "other":
        return 0

    # 使用 pymupdf 打开文件
    doc = pymupdf.open(fname, filetype=filetype)
    full_text = ""
    # 遍历每一页，获取文本内容并拼接
    for page in doc:
        full_text += page.get_text("text", sort=True, flags=settings.TEXT_FLAGS)

    return len(full_text)
def convert_single_pdf(
        fname: str,  # 定义函数，将单个 PDF 文件转换为文本
        model_lst: List,  # 模型列表
        max_pages=None,  # 最大页数，默认为 None
        metadata: Optional[Dict]=None,  # 元数据，默认为 None
        parallel_factor: int = 1  # 并行因子，默认为 1
) -> Tuple[str, Dict]:  # 返回类型为元组，包含字符串和字典

    lang = settings.DEFAULT_LANG  # 设置默认语言为系统默认语言
    if metadata:  # 如果有元数据
        lang = metadata.get("language", settings.DEFAULT_LANG)  # 获取元数据中的语言信息，如果不存在则使用系统默认语言

    # 使用 Tesseract 语言，如果可用
    tess_lang = settings.TESSERACT_LANGUAGES.get(lang, "eng")  # 获取 Tesseract 语言设置
    spell_lang = settings.SPELLCHECK_LANGUAGES.get(lang, None)  # 获取拼写检查语言设置
    if "eng" not in tess_lang:  # 如果英语不在 Tesseract 语言中
        tess_lang = f"eng+{tess_lang}"  # 添加英语到 Tesseract 语言中

    # 输出元数据
    out_meta = {"language": lang}  # 设置输出元数据的语言信息

    filetype = find_filetype(fname)  # 查找文件类型
    if filetype == "other":  # 如果文件类型为其他
        return "", out_meta  # 返回空字符串和输出元数据

    out_meta["filetype"] = filetype  # 设置输出元数据的文件类型

    doc = pymupdf.open(fname, filetype=filetype)  # 打开文件
    if filetype != "pdf":  # 如果文件类型不是 PDF
        conv = doc.convert_to_pdf()  # 将文件转换为 PDF 格式
        doc = pymupdf.open("pdf", conv)  # 打开 PDF 文件

    blocks, toc, ocr_stats = get_text_blocks(
        doc,
        tess_lang,
        spell_lang,
        max_pages=max_pages,
        parallel=int(parallel_factor * settings.OCR_PARALLEL_WORKERS)
    )  # 获取文本块、目录和 OCR 统计信息

    out_meta["toc"] = toc  # 设置输出元数据的目录信息
    out_meta["pages"] = len(blocks)  # 设置输出元数据的页数
    out_meta["ocr_stats"] = ocr_stats  # 设置输出元数据的 OCR 统计信息
    if len([b for p in blocks for b in p.blocks]) == 0:  # 如果没有提取到任何文本块
        print(f"Could not extract any text blocks for {fname}")  # 打印无法提取文本块的消息
        return "", out_meta  # 返回空字符串和输出元数据

    # 解包模型列表
    texify_model, layoutlm_model, order_model, edit_model = model_lst  # 解包模型列表

    block_types = detect_document_block_types(
        doc,
        blocks,
        layoutlm_model,
        batch_size=int(settings.LAYOUT_BATCH_SIZE * parallel_factor)
    )  # 检测文档的块类型

    # 查找页眉和页脚
    bad_span_ids = filter_header_footer(blocks)  # 过滤页眉和页脚
    out_meta["block_stats"] = {"header_footer": len(bad_span_ids)}  # 设置输出元数据的块统计信息

    annotate_spans(blocks, block_types)  # 标注文本块

    # 如果设置了标志，则转储调试数据
    dump_bbox_debug_data(doc, blocks)  # 转储边界框调试数据
    # 根据指定的参数对文档中的块进行排序
    blocks = order_blocks(
        doc,
        blocks,
        order_model,
        batch_size=int(settings.ORDERER_BATCH_SIZE * parallel_factor)
    )

    # 识别代码块数量并更新元数据
    code_block_count = identify_code_blocks(blocks)
    out_meta["block_stats"]["code"] = code_block_count
    # 缩进代码块
    indent_blocks(blocks)

    # 合并表格块
    merge_table_blocks(blocks)
    # 创建新的表格块并更新元数据
    table_count = create_new_tables(blocks)
    out_meta["block_stats"]["table"] = table_count

    # 遍历每个页面的块
    for page in blocks:
        for block in page.blocks:
            # 过滤掉坏的 span id
            block.filter_spans(bad_span_ids)
            # 过滤掉坏的 span 类型
            block.filter_bad_span_types()

    # 替换方程式并更新元数据
    filtered, eq_stats = replace_equations(
        doc,
        blocks,
        block_types,
        texify_model,
        batch_size=int(settings.TEXIFY_BATCH_SIZE * parallel_factor)
    )
    out_meta["block_stats"]["equations"] = eq_stats

    # 复制以避免更改原始数据
    merged_lines = merge_spans(filtered)
    text_blocks = merge_lines(merged_lines, filtered)
    text_blocks = filter_common_titles(text_blocks)
    full_text = get_full_text(text_blocks)

    # 处理被连接的空块
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = re.sub(r'(\n\s){3,}', '\n\n', full_text)

    # 用 - 替换项目符号字符
    full_text = replace_bullets(full_text)

    # 使用编辑器模型后处理文本
    full_text, edit_stats = edit_full_text(
        full_text,
        edit_model,
        batch_size=settings.EDITOR_BATCH_SIZE * parallel_factor
    )
    out_meta["postprocess_stats"] = {"edit": edit_stats}

    # 返回处理后的文本和元数据
    return full_text, out_meta
```
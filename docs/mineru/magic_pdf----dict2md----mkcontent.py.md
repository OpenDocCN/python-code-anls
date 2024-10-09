# `.\MinerU\magic_pdf\dict2md\mkcontent.py`

```
# 导入数学模块
import math
# 导入日志库
from loguru import logger

# 从自定义库中导入相关函数
from magic_pdf.libs.boxbase import find_bottom_nearest_text_bbox, find_top_nearest_text_bbox
from magic_pdf.libs.commons import join_path
from magic_pdf.libs.ocr_content_type import ContentType

# 定义内联方程的内容类型
TYPE_INLINE_EQUATION = ContentType.InlineEquation
# 定义行间方程的内容类型
TYPE_INTERLINE_EQUATION = ContentType.InterlineEquation
# 定义支持的文本类型列表
UNI_FORMAT_TEXT_TYPE = ['text', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']


# 定义一个被弃用的函数，接受一个字典作为参数
@DeprecationWarning
def mk_nlp_markdown_1(para_dict: dict):
    """
    对排序后的bboxes拼接内容
    """
    # 初始化内容列表
    content_lst = []
    # 遍历段落字典中的每个页面信息
    for _, page_info in para_dict.items():
        para_blocks = page_info.get("para_blocks")  # 获取段落块
        if not para_blocks:  # 如果没有段落块则跳过
            continue

        # 遍历段落块
        for block in para_blocks:
            item = block["paras"]  # 获取段落内容
            for _, p in item.items():
                para_text = p["para_text"]  # 获取段落文本
                is_title = p["is_para_title"]  # 判断是否为标题
                title_level = p['para_title_level']  # 获取标题级别
                md_title_prefix = "#"*title_level  # 根据级别生成 Markdown 标题前缀
                if is_title:  # 如果是标题
                    content_lst.append(f"{md_title_prefix} {para_text}")  # 添加标题到内容列表
                else:  # 否则为普通段落
                    content_lst.append(para_text)  # 添加段落文本到内容列表

    # 将内容列表拼接成最终文本
    content_text = "\n\n".join(content_lst)

    return content_text  # 返回拼接后的内容


# 找到目标字符串在段落中的索引
def __find_index(paragraph, target):
    index = paragraph.find(target)  # 查找目标字符串的索引
    if index != -1:  # 如果找到目标
        return index  # 返回索引
    else:  # 如果未找到目标
        return None  # 返回 None


# 在段落中插入目标字符串
def __insert_string(paragraph, target, postion):
    new_paragraph = paragraph[:postion] + target + paragraph[postion:]  # 在指定位置插入目标字符串
    return new_paragraph  # 返回新的段落


# 在内容中找到目标，插入图像内容到目标后面
def __insert_after(content, image_content, target):
    """
    在content中找到target，将image_content插入到target后面
    """
    index = content.find(target)  # 查找目标字符串的索引
    if index != -1:  # 如果找到目标
        content = content[:index+len(target)] + "\n\n" + image_content + "\n\n" + content[index+len(target):]  # 插入图像内容
    else:  # 如果未找到目标
        logger.error(f"Can't find the location of image {image_content} in the markdown file, search target is {target}")  # 记录错误日志
    return content  # 返回修改后的内容

# 在内容中找到目标，插入图像内容到目标前面
def __insert_before(content, image_content, target):
    """
    在content中找到target，将image_content插入到target前面
    """
    index = content.find(target)  # 查找目标字符串的索引
    if index != -1:  # 如果找到目标
        content = content[:index] + "\n\n" + image_content + "\n\n" + content[index:]  # 插入图像内容
    else:  # 如果未找到目标
        logger.error(f"Can't find the location of image {image_content} in the markdown file, search target is {target}")  # 记录错误日志
    return content  # 返回修改后的内容


# 定义一个被弃用的函数，接受一个字典作为参数
@DeprecationWarning
def mk_mm_markdown_1(para_dict: dict):
    """拼装多模态markdown"""
    content_lst = []  # 初始化内容列表
    """拼装成全部页面的文本"""
    content_text = "\n\n".join(content_lst)  # 将内容列表拼接成最终文本

    return content_text  # 返回拼接后的内容


# 在内容列表中找到指定文本，将图像路径作为新的节点插入到文本后面
def __insert_after_para(text, type, element, content_list):
    """
    在content_list中找到text，将image_path作为一个新的node插入到text后面
    """
    # 遍历内容列表，获取索引 i 和内容 c
        for i, c in enumerate(content_list):
            # 获取内容的类型
            content_type = c.get("type")
            # 检查内容类型是否在指定的文本格式中，并且内容文本中包含指定文本
            if content_type in UNI_FORMAT_TEXT_TYPE and text in c.get("text", ''):
                # 如果类型是 "image"
                if type == "image":
                    # 创建一个图像节点，包含图像路径和空白的替代文本、标题和说明
                    content_node = {
                        "type": "image",
                        "img_path": element.get("image_path"),
                        "img_alt": "",
                        "img_title": "",
                        "img_caption": "",
                    }
                # 如果类型是 "table"
                elif type == "table":
                    # 创建一个表格节点，包含图像路径、LaTeX 表格文本和其他空白字段
                    content_node = {
                        "type": "table",
                        "img_path": element.get("image_path"),
                        "table_latex": element.get("text"),
                        "table_title": "",
                        "table_caption": "",
                        "table_quality": element.get("quality"),
                    }
                # 在当前位置后插入新的内容节点
                content_list.insert(i+1, content_node)
                # 终止循环
                break
        # 如果没有找到合适的位置，记录错误日志
        else:
            logger.error(f"Can't find the location of image {element.get('image_path')} in the markdown file, search target is {text}")
# 定义一个函数，用于在content_list中找到指定文本并在其前面插入新节点
def __insert_before_para(text, type, element, content_list):
    """
    在content_list中找到text，将image_path作为一个新的node插入到text前面
    """
    # 遍历content_list中的每个元素及其索引
    for i, c in enumerate(content_list):
        # 获取当前元素的类型
        content_type = c.get("type")
        # 检查当前元素类型是否在UNI_FORMAT_TEXT_TYPE中，并且text是否在当前元素的文本中
        if content_type in  UNI_FORMAT_TEXT_TYPE and text in c.get("text", ''):
            # 如果type为"image"，构建图像节点
            if type == "image":
                content_node = {
                    "type": "image",  # 节点类型为图像
                    "img_path": element.get("image_path"),  # 获取图像路径
                    "img_alt": "",  # 图像的替代文本
                    "img_title": "",  # 图像的标题
                    "img_caption": "",  # 图像的说明
                }
            # 如果type为"table"，构建表格节点
            elif type == "table":
                content_node = {
                    "type": "table",  # 节点类型为表格
                    "img_path": element.get("image_path"),  # 获取与表格相关的图像路径
                    "table_latex": element.get("text"),  # 获取表格的LaTeX文本
                    "table_title": "",  # 表格的标题
                    "table_caption": "",  # 表格的说明
                    "table_quality": element.get("quality"),  # 表格的质量
                }
            # 在找到的位置插入新节点
            content_list.insert(i, content_node)
            # 找到位置后跳出循环
            break
    else:
        # 如果没有找到指定位置，记录错误日志
        logger.error(f"Can't find the location of image {element.get('image_path')} in the markdown file, search target is {text}")
         

# 定义一个函数，用于构造统一格式的内容
def mk_universal_format(pdf_info_list: list, img_buket_path):
    """
    构造统一格式 https://aicarrier.feishu.cn/wiki/FqmMwcH69iIdCWkkyjvcDwNUnTY
    """
    content_lst = []  # 初始化一个空列表，用于存储内容
    # end for
    return content_lst  # 返回构造的内容列表


# 定义一个函数，用于根据元素类型插入图像或表格
def insert_img_or_table(type, element, pymu_raw_blocks, content_lst):
    element_bbox = element['bbox']  # 获取元素的边界框
    # 先看在哪个block内
    # 遍历所有原始块
    for block in pymu_raw_blocks:
        # 获取当前块的边界框
        bbox = block['bbox']
        # 检查元素的边界框是否在当前块的范围内
        if bbox[0] - 1 <= element_bbox[0] < bbox[2] + 1 and bbox[1] - 1 <= element_bbox[1] < bbox[
            3] + 1:  # 确定在这个大的block内，然后进入逐行比较距离
            # 遍历当前块中的所有行
            for l in block['lines']:
                # 获取当前行的边界框
                line_box = l['bbox']
                # 检查元素的边界框是否在当前行的范围内
                if line_box[0] - 1 <= element_bbox[0] < line_box[2] + 1 and line_box[1] - 1 <= element_bbox[1] < line_box[
                    3] + 1:  # 在line内的，插入line前面
                    # 获取当前行中所有跨度的文本内容
                    line_txt = "".join([s['text'] for s in l['spans']])
                    # 在段落之前插入文本
                    __insert_before_para(line_txt, type, element, content_lst)
                    break
                break
            else:  # 在行与行之间
                # 找到图片的左上角与行的左上角最近的行
                min_distance = 100000
                min_line = None
                # 遍历当前块中的所有行
                for l in block['lines']:
                    # 获取当前行的边界框
                    line_box = l['bbox']
                    # 计算当前行到元素的距离
                    distance = math.sqrt((line_box[0] - element_bbox[0]) ** 2 + (line_box[1] - element_bbox[1]) ** 2)
                    # 如果距离小于最小距离，更新最小距离和最接近的行
                    if distance < min_distance:
                        min_distance = distance
                        min_line = l
                # 如果找到了最近的行
                if min_line:
                    # 获取最近行中所有跨度的文本内容
                    line_txt = "".join([s['text'] for s in min_line['spans']])
                    # 计算元素的高度
                    img_h = element_bbox[3] - element_bbox[1]
                    # 如果最小距离小于图片高度，说明文字在图片前面
                    if min_distance < img_h:  # 文字在图片前面
                        # 在段落之后插入文本
                        __insert_after_para(line_txt, type, element, content_lst)
                    else:
                        # 在段落之前插入文本
                        __insert_before_para(line_txt, type, element, content_lst)
                    break
                else:
                    # 如果找不到位置，记录错误日志
                    logger.error(f"Can't find the location of image {element.get('image_path')} in the markdown file #1")
    else:  # 应当在两个block之间
        # 找到上方最近的块，如果没有找到就找下方最近的块
        top_txt_block = find_top_nearest_text_bbox(pymu_raw_blocks, element_bbox)
        # 如果找到了上方的文本块
        if top_txt_block:
            # 获取上方块中最后一行的所有跨度文本内容
            line_txt = "".join([s['text'] for s in top_txt_block['lines'][-1]['spans']])
            # 在段落之后插入文本
            __insert_after_para(line_txt, type, element, content_lst)
        else:
            # 找到下方最近的文本块
            bottom_txt_block = find_bottom_nearest_text_bbox(pymu_raw_blocks, element_bbox)
            # 如果找到了下方的文本块
            if bottom_txt_block:
                # 获取下方块中第一行的所有跨度文本内容
                line_txt = "".join([s['text'] for s in bottom_txt_block['lines'][0]['spans']])
                # 在段落之前插入文本
                __insert_before_para(line_txt, type, element, content_lst)
            else:  # TODO ，图片可能独占一列，这种情况上下是没有图片的
                # 如果找不到位置，记录错误日志
                logger.error(f"Can't find the location of image {element.get('image_path')} in the markdown file #2")
# 基于同一格式的内容列表，构造markdown，含图片
def mk_mm_markdown(content_list):
    # 初始化空列表用于存储markdown内容
    content_md = []
    # 遍历内容列表中的每个元素
    for c in content_list:
        # 获取当前元素的类型
        content_type = c.get("type")
        # 如果类型为文本，添加文本内容
        if content_type == "text":
            content_md.append(c.get("text"))
        # 如果类型为公式，处理LaTeX格式
        elif content_type == "equation":
            content = c.get("latex")
            # 如果内容以$$开头和结尾，直接添加
            if content.startswith("$$") and content.endswith("$$"):
                content_md.append(content)
            # 否则，添加格式化的LaTeX
            else:
                content_md.append(f"\n$$\n{c.get('latex')}\n$$\n")
        # 如果类型在指定的格式文本类型中，添加标题
        elif content_type in UNI_FORMAT_TEXT_TYPE:
            content_md.append(f"{'#'*int(content_type[1])} {c.get('text')}")
        # 如果类型为图片，添加图片markdown格式
        elif content_type == "image":
            content_md.append(f"![]({c.get('img_path')})")
    # 将markdown内容列表连接成字符串并返回
    return "\n\n".join(content_md)

# 基于同一格式的内容列表，构造markdown，不含图片
def mk_nlp_markdown(content_list):
    # 初始化空列表用于存储markdown内容
    content_md = []
    # 遍历内容列表中的每个元素
    for c in content_list:
        # 获取当前元素的类型
        content_type = c.get("type")
        # 如果类型为文本，添加文本内容
        if content_type == "text":
            content_md.append(c.get("text"))
        # 如果类型为公式，添加LaTeX格式
        elif content_type == "equation":
            content_md.append(f"$$\n{c.get('latex')}\n$$")
        # 如果类型为表格，添加表格的LaTeX格式
        elif content_type == "table":
            content_md.append(f"$$$\n{c.get('table_latex')}\n$$$")
        # 如果类型在指定的格式文本类型中，添加标题
        elif content_type in UNI_FORMAT_TEXT_TYPE:
            content_md.append(f"{'#'*int(content_type[1])} {c.get('text')}")
    # 将markdown内容列表连接成字符串并返回
    return "\n\n".join(content_md)
```
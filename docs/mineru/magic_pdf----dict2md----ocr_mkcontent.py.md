# `.\MinerU\magic_pdf\dict2md\ocr_mkcontent.py`

```
import re  # 导入正则表达式模块，用于字符串匹配和搜索

import wordninja  # 导入 wordninja 模块，用于将长单词拆分成更短的部分
from loguru import logger  # 从 loguru 导入 logger，用于记录日志

from magic_pdf.libs.commons import join_path  # 从自定义库导入 join_path 函数，用于路径连接
from magic_pdf.libs.language import detect_lang  # 从自定义库导入 detect_lang 函数，用于语言检测
from magic_pdf.libs.MakeContentConfig import DropMode, MakeMode  # 导入 DropMode 和 MakeMode 配置项
from magic_pdf.libs.markdown_utils import ocr_escape_special_markdown_char  # 导入处理特殊 Markdown 字符的函数
from magic_pdf.libs.ocr_content_type import BlockType, ContentType  # 导入 OCR 内容类型的相关枚举

def __is_hyphen_at_line_end(line):  # 定义私有函数，检查行末是否有连字符
    """
    Check if a line ends with one or more letters followed by a hyphen.
    
    Args:
    line (str): The line of text to check.
    
    Returns:
    bool: True if the line ends with one or more letters followed by a hyphen, False otherwise.
    """
    # 使用正则表达式检查行是否以一个或多个字母后跟连字符结尾
    return bool(re.search(r'[A-Za-z]+-\s*$', line))

def split_long_words(text):  # 定义函数，拆分文本中的长单词
    segments = text.split(' ')  # 将文本按空格拆分为单词段
    for i in range(len(segments)):  # 遍历每个单词段
        words = re.findall(r'\w+|[^\w]', segments[i], re.UNICODE)  # 使用正则表达式查找单词和非单词字符
        for j in range(len(words)):  # 遍历每个单词
            if len(words[j]) > 10:  # 如果单词长度超过 10
                words[j] = ' '.join(wordninja.split(words[j]))  # 拆分长单词并用空格连接
        segments[i] = ''.join(words)  # 将拆分后的单词重新组合为单词段
    return ' '.join(segments)  # 将所有单词段用空格连接为最终文本

def ocr_mk_mm_markdown_with_para(pdf_info_list: list, img_buket_path):  # 定义函数，生成带段落的 Markdown 内容
    markdown = []  # 初始化 Markdown 列表
    for page_info in pdf_info_list:  # 遍历 PDF 信息列表中的每一页
        paras_of_layout = page_info.get('para_blocks')  # 获取当前页的段落块
        page_markdown = ocr_mk_markdown_with_para_core_v2(  # 调用核心函数生成当前页的 Markdown
            paras_of_layout, 'mm', img_buket_path)
        markdown.extend(page_markdown)  # 将当前页的 Markdown 内容添加到总列表中
    return '\n\n'.join(markdown)  # 将所有 Markdown 内容用双换行符连接并返回

def ocr_mk_nlp_markdown_with_para(pdf_info_dict: list):  # 定义函数，生成 NLP 格式的 Markdown 内容
    markdown = []  # 初始化 Markdown 列表
    for page_info in pdf_info_dict:  # 遍历 PDF 信息字典中的每一页
        paras_of_layout = page_info.get('para_blocks')  # 获取当前页的段落块
        page_markdown = ocr_mk_markdown_with_para_core_v2(  # 调用核心函数生成当前页的 Markdown
            paras_of_layout, 'nlp')  
        markdown.extend(page_markdown)  # 将当前页的 Markdown 内容添加到总列表中
    return '\n\n'.join(markdown)  # 将所有 Markdown 内容用双换行符连接并返回

def ocr_mk_mm_markdown_with_para_and_pagination(pdf_info_dict: list,  # 定义函数，生成带段落和分页的 Markdown 内容
                                                img_buket_path):  
    markdown_with_para_and_pagination = []  # 初始化带段落和分页的 Markdown 列表
    page_no = 0  # 初始化页码
    for page_info in pdf_info_dict:  # 遍历 PDF 信息字典中的每一页
        paras_of_layout = page_info.get('para_blocks')  # 获取当前页的段落块
        if not paras_of_layout:  # 如果当前页没有段落块，则跳过
            continue
        page_markdown = ocr_mk_markdown_with_para_core_v2(  # 调用核心函数生成当前页的 Markdown
            paras_of_layout, 'mm', img_buket_path)
        markdown_with_para_and_pagination.append({  # 将当前页的 Markdown 和页码以字典形式添加到列表
            'page_no':
            page_no,
            'md_content':
            '\n\n'.join(page_markdown)  # 将当前页的 Markdown 内容用双换行符连接
        })
        page_no += 1  # 页码递增
    return markdown_with_para_and_pagination  # 返回带段落和分页的 Markdown 列表

def ocr_mk_markdown_with_para_core(paras_of_layout, mode, img_buket_path=''):  # 定义核心函数，用于生成 Markdown 内容
    page_markdown = []  # 初始化页面 Markdown 列表
    # 遍历布局中的段落列表
        for paras in paras_of_layout:
            # 遍历每个段落中的元素
            for para in paras:
                # 初始化段落文本为空
                para_text = ''
                # 遍历段落中的每一行
                for line in para:
                    # 遍历行中的每个 span 元素
                    for span in line['spans']:
                        # 获取 span 的类型
                        span_type = span.get('type')
                        # 初始化内容和语言为空
                        content = ''
                        language = ''
                        # 如果 span 类型是文本
                        if span_type == ContentType.Text:
                            # 获取文本内容
                            content = span['content']
                            # 检测内容的语言
                            language = detect_lang(content)
                            # 对英文长词进行分词处理，中文分词不处理
                            if (language == 'en'):  # 只对英文长词进行分词处理，中文分词会丢失文本
                                content = ocr_escape_special_markdown_char(
                                    split_long_words(content))
                            else:
                                # 对内容进行特殊字符处理
                                content = ocr_escape_special_markdown_char(content)
                        # 如果 span 类型是行内公式
                        elif span_type == ContentType.InlineEquation:
                            # 格式化为行内公式的 Markdown 形式
                            content = f"${span['content']}$"
                        # 如果 span 类型是行间公式
                        elif span_type == ContentType.InterlineEquation:
                            # 格式化为行间公式的 Markdown 形式
                            content = f"\n$$\n{span['content']}\n$$\n"
                        # 如果 span 类型是图像或表格
                        elif span_type in [ContentType.Image, ContentType.Table]:
                            # 根据模式处理图像内容
                            if mode == 'mm':
                                content = f"\n![]({join_path(img_buket_path, span['image_path'])})\n"
                            elif mode == 'nlp':
                                # 如果模式是 nlp，则不处理
                                pass
                        # 如果内容不为空
                        if content != '':
                            # 在英文环境下，内容间加空格
                            if language == 'en':  # 英文语境下 content间需要空格分隔
                                para_text += content + ' '
                            else:  # 中文语境下，content间不需要空格分隔
                                para_text += content
                # 如果段落文本去除空格后仍为空，则跳过
                if para_text.strip() == '':
                    continue
                else:
                    # 将处理后的段落文本添加到页面 Markdown 列表
                    page_markdown.append(para_text.strip() + '  ')
        # 返回生成的页面 Markdown 列表
        return page_markdown
# 定义函数 ocr_mk_markdown_with_para_core_v2，接受布局参数、模式和可选的图片存储路径
def ocr_mk_markdown_with_para_core_v2(paras_of_layout,
                                      mode,
                                      img_buket_path=''):
    # 初始化一个空列表，用于存储页面的 Markdown 内容
    page_markdown = []
    # 遍历布局中的每个段落块
        for para_block in paras_of_layout:
            # 初始化段落文本为空字符串
            para_text = ''
            # 获取当前段落块的类型
            para_type = para_block['type']
            # 如果段落类型为文本
            if para_type == BlockType.Text:
                # 合并段落内容并赋值给段落文本
                para_text = merge_para_with_text(para_block)
            # 如果段落类型为标题
            elif para_type == BlockType.Title:
                # 在合并段落内容前添加 '#' 符号表示标题
                para_text = f'# {merge_para_with_text(para_block)}'
            # 如果段落类型为行间方程
            elif para_type == BlockType.InterlineEquation:
                # 合并段落内容并赋值给段落文本
                para_text = merge_para_with_text(para_block)
            # 如果段落类型为图像
            elif para_type == BlockType.Image:
                # 如果模式为 'nlp'，则跳过该段落块
                if mode == 'nlp':
                    continue
                # 如果模式为 'mm'
                elif mode == 'mm':
                    # 遍历图像段落块中的所有子块以拼接图像主体
                    for block in para_block['blocks']:  # 1st.拼image_body
                        # 如果子块类型为图像主体
                        if block['type'] == BlockType.ImageBody:
                            # 遍历行及其中的跨度
                            for line in block['lines']:
                                for span in line['spans']:
                                    # 如果跨度类型为图像
                                    if span['type'] == ContentType.Image:
                                        # 拼接图像路径并添加到段落文本
                                        para_text += f"\n![]({join_path(img_buket_path, span['image_path'])})  \n"
                    # 遍历图像段落块中的所有子块以拼接图像标题
                    for block in para_block['blocks']:  # 2nd.拼image_caption
                        # 如果子块类型为图像标题
                        if block['type'] == BlockType.ImageCaption:
                            # 合并段落内容并添加到段落文本
                            para_text += merge_para_with_text(block)
                    # 遍历图像段落块中的所有子块以拼接图像脚注
                    for block in para_block['blocks']:  # 2nd.拼image_caption
                        # 如果子块类型为图像脚注
                        if block['type'] == BlockType.ImageFootnote:
                            # 合并段落内容并添加到段落文本
                            para_text += merge_para_with_text(block)
            # 如果段落类型为表格
            elif para_type == BlockType.Table:
                # 如果模式为 'nlp'，则跳过该段落块
                if mode == 'nlp':
                    continue
                # 如果模式为 'mm'
                elif mode == 'mm':
                    # 遍历表格段落块中的所有子块以拼接表格标题
                    for block in para_block['blocks']:  # 1st.拼table_caption
                        # 如果子块类型为表格标题
                        if block['type'] == BlockType.TableCaption:
                            # 合并段落内容并添加到段落文本
                            para_text += merge_para_with_text(block)
                    # 遍历表格段落块中的所有子块以拼接表格主体
                    for block in para_block['blocks']:  # 2nd.拼table_body
                        # 如果子块类型为表格主体
                        if block['type'] == BlockType.TableBody:
                            # 遍历行及其中的跨度
                            for line in block['lines']:
                                for span in line['spans']:
                                    # 如果跨度类型为表格
                                    if span['type'] == ContentType.Table:
                                        # 如果跨度包含 LaTeX
                                        if span.get('latex', ''):
                                            # 将 LaTeX 添加到段落文本
                                            para_text += f"\n\n$\n {span['latex']}\n$\n\n"
                                        # 如果跨度包含 HTML
                                        elif span.get('html', ''):
                                            # 将 HTML 添加到段落文本
                                            para_text += f"\n\n{span['html']}\n\n"
                                        # 否则，添加图像
                                        else:
                                            para_text += f"\n![]({join_path(img_buket_path, span['image_path'])})  \n"
                    # 遍历表格段落块中的所有子块以拼接表格脚注
                    for block in para_block['blocks']:  # 3rd.拼table_footnote
                        # 如果子块类型为表格脚注
                        if block['type'] == BlockType.TableFootnote:
                            # 合并段落内容并添加到段落文本
                            para_text += merge_para_with_text(block)
    
            # 如果段落文本为空，则跳过
            if para_text.strip() == '':
                continue
            else:
                # 将非空段落文本添加到页面 Markdown 列表中
                page_markdown.append(para_text.strip() + '  ')
    
        # 返回构建好的页面 Markdown 列表
        return page_markdown
# 合并段落与文本，返回格式化的段落文本
def merge_para_with_text(para_block):

    # 检测给定文本的语言
    def detect_language(text):
        # 定义正则表达式，匹配英文单词
        en_pattern = r'[a-zA-Z]+'
        # 在文本中查找所有英文单词
        en_matches = re.findall(en_pattern, text)
        # 计算英文单词的总长度
        en_length = sum(len(match) for match in en_matches)
        # 如果文本长度大于0
        if len(text) > 0:
            # 判断英文占比是否大于等于50%
            if en_length / len(text) >= 0.5:
                return 'en'  # 返回英文标识
            else:
                return 'unknown'  # 返回未知语言标识
        else:
            return 'empty'  # 返回空文本标识

    # 初始化段落文本
    para_text = ''
    # 遍历段落中的每一行
    for line in para_block['lines']:
        # 初始化行文本和语言
        line_text = ''
        line_lang = ''
        # 遍历行中的每个文本跨度
        for span in line['spans']:
            # 获取跨度类型
            span_type = span['type']
            # 如果跨度类型为文本
            if span_type == ContentType.Text:
                # 将内容去除前后空格后添加到行文本
                line_text += span['content'].strip()
        # 如果行文本不为空
        if line_text != '':
            # 检测行语言
            line_lang = detect_language(line_text)
        # 再次遍历行中的每个文本跨度
        for span in line['spans']:
            # 获取跨度类型
            span_type = span['type']
            content = ''
            # 如果跨度类型为文本
            if span_type == ContentType.Text:
                content = span['content']
                # language = detect_lang(content)  # 注释掉的语言检测
                language = detect_language(content)  # 检测当前内容的语言
                # 如果检测到英文
                if language == 'en':  # 只对英文长词进行分词处理，中文分词会丢失文本
                    # 处理长英文单词并转义特殊字符
                    content = ocr_escape_special_markdown_char(
                        split_long_words(content))
                else:
                    # 对内容转义特殊字符
                    content = ocr_escape_special_markdown_char(content)
            # 如果跨度类型为行内方程
            elif span_type == ContentType.InlineEquation:
                # 格式化行内方程内容
                content = f" ${span['content']}$ "
            # 如果跨度类型为行间方程
            elif span_type == ContentType.InterlineEquation:
                # 格式化行间方程内容
                content = f"\n$$\n{span['content']}\n$$\n"

            # 如果内容不为空
            if content != '':
                langs = ['zh', 'ja', 'ko']  # 定义需要判断的语言列表
                # 如果行语言在指定语言列表中
                if line_lang in langs:  # 遇到一些一个字一个span的文档，这种单字语言判断不准，需要用整行文本判断
                    para_text += content  # 中文/日语/韩文语境下，content间不需要空格分隔
                elif line_lang == 'en':
                    # 如果前一行以连字符结束，不添加空格
                    if __is_hyphen_at_line_end(content):
                        para_text += content[:-1]  # 去掉末尾空格
                    else:
                        para_text += content + ' '  # 添加空格
                else:
                    para_text += content + ' '  # 西方文本语境下 content间需要空格分隔
    # 返回合并后的段落文本
    return para_text


# 将段落转换为标准格式
def para_to_standard_format(para, img_buket_path):
    para_content = {}
    # 如果段落长度为1
    if len(para) == 1:
        # 转换为标准格式
        para_content = line_to_standard_format(para[0], img_buket_path)
    # 检查 para 列表的长度，如果大于 1，则执行以下操作
        elif len(para) > 1:
            # 初始化一个空字符串用于存储段落文本
            para_text = ''
            # 初始化计数器，用于记录行内公式的数量
            inline_equation_num = 0
            # 遍历 para 中的每一行
            for line in para:
                # 遍历当前行的每个 span
                for span in line['spans']:
                    # 初始化语言变量
                    language = ''
                    # 获取当前 span 的类型
                    span_type = span.get('type')
                    # 初始化内容变量
                    content = ''
                    # 如果 span 类型是文本
                    if span_type == ContentType.Text:
                        # 获取文本内容
                        content = span['content']
                        # 检测文本的语言
                        language = detect_lang(content)
                        # 如果检测到的语言是英文
                        if language == 'en':  # 只对英文长词进行分词处理，中文分词会丢失文本
                            # 对长英文词进行分词处理，并处理特殊 Markdown 字符
                            content = ocr_escape_special_markdown_char(
                                split_long_words(content))
                        else:
                            # 处理中文内容中的特殊 Markdown 字符
                            content = ocr_escape_special_markdown_char(content)
                    # 如果 span 类型是行内公式
                    elif span_type == ContentType.InlineEquation:
                        # 格式化行内公式内容
                        content = f"${span['content']}$"
                        # 计数增加
                        inline_equation_num += 1
                    # 如果语言是英文
                    if language == 'en':  # 英文语境下 content间需要空格分隔
                        # 在段落文本中添加内容，并添加空格
                        para_text += content + ' '
                    else:  # 中文语境下，content间不需要空格分隔
                        # 直接在段落文本中添加内容
                        para_text += content
            # 创建包含段落类型、文本和行内公式数量的字典
            para_content = {
                'type': 'text',
                'text': para_text,
                'inline_equation_num': inline_equation_num,
            }
        # 返回段落内容字典
        return para_content
# 将段落块转换为标准格式的函数，接受段落块、图片桶路径和页面索引作为参数
def para_to_standard_format_v2(para_block, img_buket_path, page_idx):
    # 获取段落块的类型
    para_type = para_block['type']
    # 如果段落类型为文本
    if para_type == BlockType.Text:
        # 创建包含类型、合并文本和页面索引的内容字典
        para_content = {
            'type': 'text',
            'text': merge_para_with_text(para_block),
            'page_idx': page_idx,
        }
    # 如果段落类型为标题
    elif para_type == BlockType.Title:
        # 创建包含类型、合并文本、文本级别和页面索引的内容字典
        para_content = {
            'type': 'text',
            'text': merge_para_with_text(para_block),
            'text_level': 1,
            'page_idx': page_idx,
        }
    # 如果段落类型为行间方程
    elif para_type == BlockType.InterlineEquation:
        # 创建包含类型、合并文本、文本格式和页面索引的内容字典
        para_content = {
            'type': 'equation',
            'text': merge_para_with_text(para_block),
            'text_format': 'latex',
            'page_idx': page_idx,
        }
    # 如果段落类型为图片
    elif para_type == BlockType.Image:
        # 创建包含类型和页面索引的内容字典
        para_content = {'type': 'image', 'page_idx': page_idx}
        # 遍历段落块中的所有块
        for block in para_block['blocks']:
            # 如果块类型为图片主体
            if block['type'] == BlockType.ImageBody:
                # 获取图片路径并加入内容字典
                para_content['img_path'] = join_path(
                    img_buket_path,
                    block['lines'][0]['spans'][0]['image_path'])
            # 如果块类型为图片标题
            if block['type'] == BlockType.ImageCaption:
                # 合并文本并加入内容字典
                para_content['img_caption'] = merge_para_with_text(block)
            # 如果块类型为图片脚注
            if block['type'] == BlockType.ImageFootnote:
                # 合并文本并加入内容字典
                para_content['img_footnote'] = merge_para_with_text(block)
    # 如果段落类型为表格
    elif para_type == BlockType.Table:
        # 创建包含类型和页面索引的内容字典
        para_content = {'type': 'table', 'page_idx': page_idx}
        # 遍历段落块中的所有块
        for block in para_block['blocks']:
            # 如果块类型为表格主体
            if block['type'] == BlockType.TableBody:
                # 如果存在 LaTeX 格式的内容
                if block["lines"][0]["spans"][0].get('latex', ''):
                    # 将 LaTeX 内容加入到表格主体中
                    para_content['table_body'] = f"\n\n$\n {block['lines'][0]['spans'][0]['latex']}\n$\n\n"
                # 如果存在 HTML 格式的内容
                elif block["lines"][0]["spans"][0].get('html', ''):
                    # 将 HTML 内容加入到表格主体中
                    para_content['table_body'] = f"\n\n{block['lines'][0]['spans'][0]['html']}\n\n"
                # 获取图片路径并加入内容字典
                para_content['img_path'] = join_path(img_buket_path, block["lines"][0]["spans"][0]['image_path'])
            # 如果块类型为表格标题
            if block['type'] == BlockType.TableCaption:
                # 合并文本并加入内容字典
                para_content['table_caption'] = merge_para_with_text(block)
            # 如果块类型为表格脚注
            if block['type'] == BlockType.TableFootnote:
                # 合并文本并加入内容字典
                para_content['table_footnote'] = merge_para_with_text(block)

    # 返回创建的段落内容字典
    return para_content


# 根据 PDF 信息字典和图片桶路径创建标准格式的内容列表
def make_standard_format_with_para(pdf_info_dict: list, img_buket_path: str):
    # 初始化内容列表
    content_list = []
    # 遍历 PDF 信息字典中的每一页信息
    for page_info in pdf_info_dict:
        # 获取该页面的段落块
        paras_of_layout = page_info.get('para_blocks')
        # 如果没有段落块则跳过
        if not paras_of_layout:
            continue
        # 遍历段落块
        for para_block in paras_of_layout:
            # 将段落块转换为标准格式
            para_content = para_to_standard_format_v2(para_block,
                                                      img_buket_path)
            # 将转换后的内容加入到内容列表中
            content_list.append(para_content)
    # 返回创建的内容列表
    return content_list


# 将行转换为标准格式的函数，接受行和图片桶路径作为参数
def line_to_standard_format(line, img_buket_path):
    # 初始化行文本为空字符串
    line_text = ''
    # 初始化行内方程计数为0
    inline_equation_num = 0
    # 遍历当前行中的所有跨度
    for span in line['spans']:
        # 检查跨度中是否有内容
        if not span.get('content'):
            # 如果没有内容，检查是否有图像路径
            if not span.get('image_path'):
                # 如果没有图像路径，跳过当前循环
                continue
            else:
                # 如果有图像路径，判断类型是否为图像
                if span['type'] == ContentType.Image:
                    # 创建包含图像类型和图像路径的内容字典
                    content = {
                        'type': 'image',
                        'img_path': join_path(img_buket_path,
                                              span['image_path']),
                    }
                    # 返回图像内容
                    return content
                # 判断类型是否为表格
                elif span['type'] == ContentType.Table:
                    # 创建包含表格类型和图像路径的内容字典
                    content = {
                        'type': 'table',
                        'img_path': join_path(img_buket_path,
                                              span['image_path']),
                    }
                    # 返回表格内容
                    return content
        else:
            # 检查跨度类型是否为行间公式
            if span['type'] == ContentType.InterlineEquation:
                # 获取行间公式内容
                interline_equation = span['content']
                # 创建包含公式类型和 LaTeX 格式的内容字典
                content = {
                    'type': 'equation',
                    'latex': f'$$\n{interline_equation}\n$$'
                }
                # 返回公式内容
                return content
            # 检查类型是否为内联公式
            elif span['type'] == ContentType.InlineEquation:
                # 获取内联公式内容
                inline_equation = span['content']
                # 将内联公式添加到行文本中
                line_text += f'${inline_equation}$'
                # 内联公式计数加一
                inline_equation_num += 1
            # 检查类型是否为文本
            elif span['type'] == ContentType.Text:
                # 转义特殊符号以处理文本内容
                text_content = ocr_escape_special_markdown_char(
                    span['content'])  # 转义特殊符号
                # 将处理后的文本内容添加到行文本中
                line_text += text_content
    # 创建包含文本类型、文本内容和内联公式计数的最终内容字典
    content = {
        'type': 'text',
        'text': line_text,
        'inline_equation_num': inline_equation_num,
    }
    # 返回最终内容
    return content
# 根据 PDF 信息字典生成标准格式内容列表
def ocr_mk_mm_standard_format(pdf_info_dict: list):
    """content_list type         string
    image/text/table/equation(行间的单独拿出来，行内的和text合并) latex        string
    latex文本字段。 text         string      纯文本格式的文本数据。 md           string
    markdown格式的文本数据。 img_path     string      s3://full/path/to/img.jpg."""
    # 初始化内容列表
    content_list = []
    # 遍历 PDF 信息字典中的每一页信息
    for page_info in pdf_info_dict:
        # 获取页面的预处理块
        blocks = page_info.get('preproc_blocks')
        # 如果没有预处理块，跳过当前页
        if not blocks:
            continue
        # 遍历每个块
        for block in blocks:
            # 遍历块中的每一行
            for line in block['lines']:
                # 将行转换为标准格式
                content = line_to_standard_format(line)
                # 将转换后的内容添加到内容列表中
                content_list.append(content)
    # 返回内容列表
    return content_list


# 根据 PDF 信息字典生成指定格式内容
def union_make(pdf_info_dict: list,
               make_mode: str,
               drop_mode: str,
               img_buket_path: str = ''):
    # 初始化输出内容列表
    output_content = []
    # 遍历 PDF 信息字典中的每一页信息
    for page_info in pdf_info_dict:
        # 检查当前页面是否需要丢弃
        if page_info.get('need_drop', False):
            # 获取丢弃原因
            drop_reason = page_info.get('drop_reason')
            # 根据丢弃模式进行处理
            if drop_mode == DropMode.NONE:
                pass
            elif drop_mode == DropMode.WHOLE_PDF:
                # 如果丢弃模式为丢弃整个 PDF，则抛出异常
                raise Exception((f'drop_mode is {DropMode.WHOLE_PDF} ,'
                                 f'drop_reason is {drop_reason}'))
            elif drop_mode == DropMode.SINGLE_PAGE:
                # 如果丢弃模式为丢弃单页，记录警告并继续
                logger.warning((f'drop_mode is {DropMode.SINGLE_PAGE} ,'
                                f'drop_reason is {drop_reason}'))
                continue
            else:
                # 丢弃模式无效，抛出异常
                raise Exception('drop_mode can not be null')

        # 获取页面的段落块
        paras_of_layout = page_info.get('para_blocks')
        # 获取页面索引
        page_idx = page_info.get('page_idx')
        # 如果没有段落块，跳过当前页
        if not paras_of_layout:
            continue
        # 根据制作模式生成相应的内容
        if make_mode == MakeMode.MM_MD:
            # 生成 Markdown 格式的页面内容
            page_markdown = ocr_mk_markdown_with_para_core_v2(
                paras_of_layout, 'mm', img_buket_path)
            # 将生成的内容添加到输出内容中
            output_content.extend(page_markdown)
        elif make_mode == MakeMode.NLP_MD:
            # 生成 NLP 模式下的 Markdown 内容
            page_markdown = ocr_mk_markdown_with_para_core_v2(
                paras_of_layout, 'nlp')
            # 将生成的内容添加到输出内容中
            output_content.extend(page_markdown)
        elif make_mode == MakeMode.STANDARD_FORMAT:
            # 遍历段落块
            for para_block in paras_of_layout:
                # 将段落转换为标准格式
                para_content = para_to_standard_format_v2(
                    para_block, img_buket_path, page_idx)
                # 将转换后的内容添加到输出内容中
                output_content.append(para_content)
    # 根据制作模式返回相应的内容格式
    if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
        # 返回以换行符连接的内容字符串
        return '\n\n'.join(output_content)
    elif make_mode == MakeMode.STANDARD_FORMAT:
        # 返回标准格式内容列表
        return output_content
```
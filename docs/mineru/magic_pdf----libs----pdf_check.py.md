# `.\MinerU\magic_pdf\libs\pdf_check.py`

```
# 导入必要的库
from io import BytesIO  # 用于字节流操作
import re  # 用于正则表达式匹配
import fitz  # 用于PDF文档处理
import numpy as np  # 用于数值计算和随机选择
from loguru import logger  # 用于日志记录
from pdfminer.high_level import extract_text  # 用于从PDF提取文本


def calculate_sample_count(total_page: int):
    """
    根据总页数和采样率计算采样页面的数量。
    """
    # 选择的页面数量不超过10页
    select_page_cnt = min(10, total_page)
    # 返回选择的页面数量
    return select_page_cnt


def extract_pages(src_pdf_bytes: bytes):
    # 打开PDF文档并将字节流转换为fitz文档对象
    pdf_docs = fitz.open("pdf", src_pdf_bytes)
    # 获取PDF文档的总页数
    total_page = len(pdf_docs)
    # 如果PDF没有页面，直接返回空文档
    if total_page == 0:
        logger.warning("PDF is empty, return empty document")  # 记录警告日志
        return fitz.Document()  # 返回空PDF文档
    # 计算要抽取的页面数量
    select_page_cnt = calculate_sample_count(total_page)
    # 随机选择不重复的页面编号
    page_num = np.random.choice(total_page, select_page_cnt, replace=False)
    # 创建一个新的PDF文档用于存放抽取的页面
    sample_docs = fitz.Document()
    try:
        # 遍历选择的页面编号并插入到新的PDF文档中
        for index in page_num:
            sample_docs.insert_pdf(pdf_docs, from_page=int(index), to_page=int(index))
    except Exception as e:
        logger.exception(e)  # 记录异常日志
    # 返回包含抽取页面的PDF文档
    return sample_docs


def detect_invalid_chars(src_pdf_bytes: bytes) -> bool:
    """"
    检测PDF中是否包含非法字符
    """
    '''pdfminer比较慢,需要先随机抽取10页左右的sample'''
    # 从PDF字节流中提取样本页面
    sample_docs = extract_pages(src_pdf_bytes)
    # 将样本PDF文档转换为字节流
    sample_pdf_bytes = sample_docs.tobytes()
    # 创建字节流对象用于提取文本
    sample_pdf_file_like_object = BytesIO(sample_pdf_bytes)
    # 提取样本中的文本
    text = extract_text(sample_pdf_file_like_object)
    # 移除文本中的换行符
    text = text.replace("\n", "")
    # logger.info(text)  # 记录提取的文本（被注释掉）
    '''乱码文本用pdfminer提取出来的文本特征是(cid:xxx)'''
    # 编译正则表达式用于匹配乱码字符
    cid_pattern = re.compile(r'\(cid:\d+\)')
    # 在提取的文本中查找所有匹配的乱码字符
    matches = cid_pattern.findall(text)
    # 统计匹配的数量
    cid_count = len(matches)
    # 统计匹配的总长度
    cid_len = sum(len(match) for match in matches)
    # 计算提取文本的总长度
    text_len = len(text)
    # 如果文本长度为0，cid字符比例为0
    if text_len == 0:
        cid_chars_radio = 0
    else:
        # 计算cid字符比例
        cid_chars_radio = cid_count / (cid_count + text_len - cid_len)
    # 记录cid字符数量、文本长度和比例
    logger.info(f"cid_count: {cid_count}, text_len: {text_len}, cid_chars_radio: {cid_chars_radio}")
    '''当一篇文章存在5%以上的文本是乱码时,认为该文档为乱码文档'''
    # 判断字符比例是否超过5%
    if cid_chars_radio > 0.05:
        return False  # 返回False，表示是乱码文档
    else:
        return True   # 返回True，表示正常文档
```
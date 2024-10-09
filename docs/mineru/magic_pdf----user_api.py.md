# `.\MinerU\magic_pdf\user_api.py`

```
# 用户输入相关说明，包括模型数组、PDF在S3的路径及截图保存位置
"""
用户输入：
    model数组，每个元素代表一个页面
    pdf在s3的路径
    截图保存的s3位置

然后：
    1）根据s3路径，调用spark集群的api,拿到ak,sk,endpoint，构造出s3PDFReader
    2）根据用户输入的s3地址，调用spark集群的api,拿到ak,sk,endpoint，构造出s3ImageWriter

其余部分至于构造s3cli, 获取ak,sk都在code-clean里写代码完成。不要反向依赖！！！

"""
# 导入正则表达式库
import re

# 导入日志记录库
from loguru import logger

# 从版本模块导入当前版本号
from magic_pdf.libs.version import __version__
# 从自定义模型分析模块导入文档分析函数
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
# 导入抽象读写器类
from magic_pdf.rw import AbsReaderWriter
# 从OCR解析模块导入PDF解析函数
from magic_pdf.pdf_parse_by_ocr import parse_pdf_by_ocr
# 从文本解析模块导入PDF解析函数
from magic_pdf.pdf_parse_by_txt import parse_pdf_by_txt

# 定义解析类型常量：文本
PARSE_TYPE_TXT = "txt"
# 定义解析类型常量：OCR
PARSE_TYPE_OCR = "ocr"


# 定义解析文本PDF的函数
def parse_txt_pdf(pdf_bytes: bytes, pdf_models: list, imageWriter: AbsReaderWriter, is_debug=False,
                  start_page_id=0, end_page_id=None,
                  *args, **kwargs):
    """
    解析文本类pdf
    """
    # 调用文本解析函数，获取PDF信息字典
    pdf_info_dict = parse_pdf_by_txt(
        pdf_bytes,
        pdf_models,
        imageWriter,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        debug_mode=is_debug,
    )

    # 将解析类型添加到字典中
    pdf_info_dict["_parse_type"] = PARSE_TYPE_TXT

    # 将版本名称添加到字典中
    pdf_info_dict["_version_name"] = __version__

    # 返回PDF信息字典
    return pdf_info_dict


# 定义解析OCR PDF的函数
def parse_ocr_pdf(pdf_bytes: bytes, pdf_models: list, imageWriter: AbsReaderWriter, is_debug=False,
                  start_page_id=0, end_page_id=None,
                  *args, **kwargs):
    """
    解析ocr类pdf
    """
    # 调用OCR解析函数，获取PDF信息字典
    pdf_info_dict = parse_pdf_by_ocr(
        pdf_bytes,
        pdf_models,
        imageWriter,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        debug_mode=is_debug,
    )

    # 将解析类型添加到字典中
    pdf_info_dict["_parse_type"] = PARSE_TYPE_OCR

    # 将版本名称添加到字典中
    pdf_info_dict["_version_name"] = __version__

    # 返回PDF信息字典
    return pdf_info_dict


# 定义解析混合PDF的函数
def parse_union_pdf(pdf_bytes: bytes, pdf_models: list, imageWriter: AbsReaderWriter, is_debug=False,
                    input_model_is_empty: bool = False,
                    start_page_id=0, end_page_id=None,
                    *args, **kwargs):
    """
    ocr和文本混合的pdf，全部解析出来
    """

    # 定义内部解析函数，使用不同解析方法
    def parse_pdf(method):
        try:
            # 调用传入的方法解析PDF，返回结果
            return method(
                pdf_bytes,
                pdf_models,
                imageWriter,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                debug_mode=is_debug,
            )
        except Exception as e:
            # 捕获异常并记录日志
            logger.exception(e)
            return None

    # 使用文本解析方法解析PDF，获取PDF信息字典
    pdf_info_dict = parse_pdf(parse_pdf_by_txt)
    # 检查 pdf_info_dict 是否为空或需要丢弃
        if pdf_info_dict is None or pdf_info_dict.get("_need_drop", False):
            # 记录警告信息，说明将切换到 OCR 解析
            logger.warning(f"parse_pdf_by_txt drop or error, switch to parse_pdf_by_ocr")
            # 如果输入模型为空，则进行文档分析，使用 OCR
            if input_model_is_empty:
                pdf_models = doc_analyze(pdf_bytes, ocr=True,
                                         start_page_id=start_page_id,
                                         end_page_id=end_page_id)
            # 使用 OCR 解析 PDF，并更新 pdf_info_dict
            pdf_info_dict = parse_pdf(parse_pdf_by_ocr)
            # 如果 OCR 解析失败，则抛出异常
            if pdf_info_dict is None:
                raise Exception("Both parse_pdf_by_txt and parse_pdf_by_ocr failed.")
            else:
                # 设置解析类型为 OCR
                pdf_info_dict["_parse_type"] = PARSE_TYPE_OCR
        else:
            # 设置解析类型为 TXT
            pdf_info_dict["_parse_type"] = PARSE_TYPE_TXT
    
        # 更新 pdf_info_dict 中的版本信息
        pdf_info_dict["_version_name"] = __version__
    
        # 返回解析后的 PDF 信息字典
        return pdf_info_dict
```
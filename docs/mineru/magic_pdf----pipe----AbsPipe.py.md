# `.\MinerU\magic_pdf\pipe\AbsPipe.py`

```
# 导入 ABC 和 abstractmethod，用于定义抽象基类和抽象方法
from abc import ABC, abstractmethod

# 从指定模块导入 union_make 函数
from magic_pdf.dict2md.ocr_mkcontent import union_make
# 从指定模块导入 classify 函数
from magic_pdf.filter.pdf_classify_by_type import classify
# 从指定模块导入 pdf_meta_scan 函数
from magic_pdf.filter.pdf_meta_scan import pdf_meta_scan
# 从指定模块导入 MakeMode 和 DropMode 配置类
from magic_pdf.libs.MakeContentConfig import MakeMode, DropMode
# 从指定模块导入抽象读取写入类
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
# 从指定模块导入 DropReason 类
from magic_pdf.libs.drop_reason import DropReason
# 从指定模块导入 JsonCompressor 类
from magic_pdf.libs.json_compressor import JsonCompressor


# 定义抽象类 AbsPipe，继承自 ABC
class AbsPipe(ABC):
    """
    txt和ocr处理的抽象类
    """
    # 定义常量 PIP_OCR 表示 OCR 处理
    PIP_OCR = "ocr"
    # 定义常量 PIP_TXT 表示 TXT 处理
    PIP_TXT = "txt"

    # 初始化方法，接收多个参数
    def __init__(self, pdf_bytes: bytes, model_list: list, image_writer: AbsReaderWriter, is_debug: bool = False,
                 start_page_id=0, end_page_id=None):
        # 存储 PDF 文件的字节数据
        self.pdf_bytes = pdf_bytes
        # 存储模型列表
        self.model_list = model_list
        # 存储图像写入器实例
        self.image_writer = image_writer
        # 初始化未压缩的中间数据
        self.pdf_mid_data = None  # 未压缩
        # 存储调试模式标志
        self.is_debug = is_debug
        # 存储开始页码
        self.start_page_id = start_page_id
        # 存储结束页码，默认为 None
        self.end_page_id = end_page_id
    
    # 获取压缩后的 PDF 中间数据
    def get_compress_pdf_mid_data(self):
        # 调用 JsonCompressor 的 compress_json 方法进行压缩
        return JsonCompressor.compress_json(self.pdf_mid_data)

    # 定义抽象方法 pipe_classify，子类需实现
    @abstractmethod
    def pipe_classify(self):
        """
        有状态的分类
        """
        # 抛出未实现错误
        raise NotImplementedError

    # 定义抽象方法 pipe_analyze，子类需实现
    @abstractmethod
    def pipe_analyze(self):
        """
        有状态的跑模型分析
        """
        # 抛出未实现错误
        raise NotImplementedError

    # 定义抽象方法 pipe_parse，子类需实现
    @abstractmethod
    def pipe_parse(self):
        """
        有状态的解析
        """
        # 抛出未实现错误
        raise NotImplementedError

    # 定义方法 pipe_mk_uni_format，生成统一格式内容
    def pipe_mk_uni_format(self, img_parent_path: str, drop_mode=DropMode.WHOLE_PDF):
        # 调用 mk_uni_format 方法生成内容列表
        content_list = AbsPipe.mk_uni_format(self.get_compress_pdf_mid_data(), img_parent_path, drop_mode)
        # 返回内容列表
        return content_list

    # 定义方法 pipe_mk_markdown，生成 Markdown 内容
    def pipe_mk_markdown(self, img_parent_path: str, drop_mode=DropMode.WHOLE_PDF, md_make_mode=MakeMode.MM_MD):
        # 调用 mk_markdown 方法生成 Markdown 内容
        md_content = AbsPipe.mk_markdown(self.get_compress_pdf_mid_data(), img_parent_path, drop_mode, md_make_mode)
        # 返回 Markdown 内容
        return md_content

    # 定义静态方法
    # 根据 PDF 字节数据判断其类型，返回相应字符串
        def classify(pdf_bytes: bytes) -> str:
            """
            根据pdf的元数据，判断是文本pdf，还是ocr pdf
            """
            # 扫描 PDF 元数据
            pdf_meta = pdf_meta_scan(pdf_bytes)
            # 检查是否需要丢弃该 PDF，如果是，则抛出异常
            if pdf_meta.get("_need_drop", False):  # 如果返回了需要丢弃的标志，则抛出异常
                raise Exception(f"pdf meta_scan need_drop,reason is {pdf_meta['_drop_reason']}")
            else:
                # 获取 PDF 是否加密和是否需要密码的信息
                is_encrypted = pdf_meta["is_encrypted"]
                is_needs_password = pdf_meta["is_needs_password"]
                # 如果 PDF 加密或需要密码，则抛出异常
                if is_encrypted or is_needs_password:  # 加密的，需要密码的，没有页面的，都不处理
                    raise Exception(f"pdf meta_scan need_drop,reason is {DropReason.ENCRYPTED}")
                else:
                    # 分类 PDF，根据其页面信息等提取特征
                    is_text_pdf, results = classify(
                        pdf_meta["total_page"],
                        pdf_meta["page_width_pts"],
                        pdf_meta["page_height_pts"],
                        pdf_meta["image_info_per_page"],
                        pdf_meta["text_len_per_page"],
                        pdf_meta["imgs_per_page"],
                        pdf_meta["text_layout_per_page"],
                        pdf_meta["invalid_chars"],
                    )
                    # 如果是文本 PDF，则返回对应标识
                    if is_text_pdf:
                        return AbsPipe.PIP_TXT
                    else:
                        # 否则返回 OCR PDF 的标识
                        return AbsPipe.PIP_OCR
    
        # 静态方法，根据 PDF 类型生成统一格式的内容列表
        @staticmethod
        def mk_uni_format(compressed_pdf_mid_data: str, img_buket_path: str, drop_mode=DropMode.WHOLE_PDF) -> list:
            """
            根据pdf类型，生成统一格式content_list
            """
            # 解压缩 JSON 数据，获取 PDF 中间数据
            pdf_mid_data = JsonCompressor.decompress_json(compressed_pdf_mid_data)
            # 获取 PDF 信息列表
            pdf_info_list = pdf_mid_data["pdf_info"]
            # 生成统一格式的内容列表
            content_list = union_make(pdf_info_list, MakeMode.STANDARD_FORMAT, drop_mode, img_buket_path)
            # 返回生成的内容列表
            return content_list
    
        # 静态方法，根据 PDF 类型生成 Markdown 格式的内容列表
        @staticmethod
        def mk_markdown(compressed_pdf_mid_data: str, img_buket_path: str, drop_mode=DropMode.WHOLE_PDF, md_make_mode=MakeMode.MM_MD) -> list:
            """
            根据pdf类型，markdown
            """
            # 解压缩 JSON 数据，获取 PDF 中间数据
            pdf_mid_data = JsonCompressor.decompress_json(compressed_pdf_mid_data)
            # 获取 PDF 信息列表
            pdf_info_list = pdf_mid_data["pdf_info"]
            # 生成 Markdown 格式的内容
            md_content = union_make(pdf_info_list, md_make_mode, drop_mode, img_buket_path)
            # 返回生成的 Markdown 内容
            return md_content
```
# `.\MinerU\magic_pdf\pipe\UNIPipe.py`

```
# 导入 json 模块，用于处理 JSON 数据
import json

# 从 loguru 导入 logger，用于日志记录
from loguru import logger

# 从 MakeContentConfig 导入 DropMode 和 MakeMode 枚举
from magic_pdf.libs.MakeContentConfig import DropMode, MakeMode

# 从自定义模型分析模块导入 doc_analyze 函数
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

# 从抽象读写器模块导入 AbsReaderWriter 类
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter

# 从磁盘读写器模块导入 DiskReaderWriter 类
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

# 从公共工具模块导入 join_path 函数
from magic_pdf.libs.commons import join_path

# 从抽象管道模块导入 AbsPipe 类
from magic_pdf.pipe.AbsPipe import AbsPipe

# 从用户 API 导入解析 PDF 的函数
from magic_pdf.user_api import parse_union_pdf, parse_ocr_pdf


# 定义 UNIPipe 类，继承自 AbsPipe
class UNIPipe(AbsPipe):

    # 初始化 UNIPipe 实例
    def __init__(self, pdf_bytes: bytes, jso_useful_key: dict, image_writer: AbsReaderWriter, is_debug: bool = False,
                 start_page_id=0, end_page_id=None):
        # 获取 PDF 类型
        self.pdf_type = jso_useful_key["_pdf_type"]
        # 调用父类构造函数
        super().__init__(pdf_bytes, jso_useful_key["model_list"], image_writer, is_debug, start_page_id, end_page_id)
        # 检查模型列表是否为空
        if len(self.model_list) == 0:
            self.input_model_is_empty = True
        else:
            self.input_model_is_empty = False

    # 处理分类逻辑
    def pipe_classify(self):
        # 分类 PDF 类型
        self.pdf_type = AbsPipe.classify(self.pdf_bytes)

    # 处理分析逻辑
    def pipe_analyze(self):
        # 如果 PDF 类型是文本
        if self.pdf_type == self.PIP_TXT:
            # 调用 doc_analyze 函数进行分析
            self.model_list = doc_analyze(self.pdf_bytes, ocr=False,
                                          start_page_id=self.start_page_id, end_page_id=self.end_page_id)
        # 如果 PDF 类型是 OCR
        elif self.pdf_type == self.PIP_OCR:
            # 调用 doc_analyze 函数进行分析
            self.model_list = doc_analyze(self.pdf_bytes, ocr=True,
                                          start_page_id=self.start_page_id, end_page_id=self.end_page_id)

    # 处理解析逻辑
    def pipe_parse(self):
        # 如果 PDF 类型是文本
        if self.pdf_type == self.PIP_TXT:
            # 解析 PDF 中的数据
            self.pdf_mid_data = parse_union_pdf(self.pdf_bytes, self.model_list, self.image_writer,
                                                is_debug=self.is_debug, input_model_is_empty=self.input_model_is_empty,
                                                start_page_id=self.start_page_id, end_page_id=self.end_page_id)
        # 如果 PDF 类型是 OCR
        elif self.pdf_type == self.PIP_OCR:
            # 解析 PDF 中的数据
            self.pdf_mid_data = parse_ocr_pdf(self.pdf_bytes, self.model_list, self.image_writer,
                                              is_debug=self.is_debug,
                                              start_page_id=self.start_page_id, end_page_id=self.end_page_id)

    # 创建统一格式的管道
    def pipe_mk_uni_format(self, img_parent_path: str, drop_mode=DropMode.WHOLE_PDF):
        # 调用父类的方法生成统一格式
        result = super().pipe_mk_uni_format(img_parent_path, drop_mode)
        # 记录生成完成的日志
        logger.info("uni_pipe mk content list finished")
        # 返回结果
        return result

    # 创建 Markdown 格式的管道
    def pipe_mk_markdown(self, img_parent_path: str, drop_mode=DropMode.WHOLE_PDF, md_make_mode=MakeMode.MM_MD):
        # 调用父类的方法生成 Markdown 格式
        result = super().pipe_mk_markdown(img_parent_path, drop_mode, md_make_mode)
        # 记录生成完成的日志
        logger.info(f"uni_pipe mk {md_make_mode} finished")
        # 返回结果
        return result


# 程序入口
if __name__ == '__main__':
    # 测试
    # 创建 DiskReaderWriter 实例，指定文件路径
    drw = DiskReaderWriter(r"D:/project/20231108code-clean")

    # PDF 文件路径
    pdf_file_path = r"linshixuqiu\19983-00.pdf"
    # 模型文件路径
    model_file_path = r"linshixuqiu\19983-00.json"
    # 从指定路径读取 PDF 文件内容，以二进制模式
    pdf_bytes = drw.read(pdf_file_path, AbsReaderWriter.MODE_BIN)
    # 从指定路径读取模型文件内容，以文本模式
    model_json_txt = drw.read(model_file_path, AbsReaderWriter.MODE_TXT)
    # 将读取的 JSON 格式的模型文本解析为 Python 对象（列表）
    model_list = json.loads(model_json_txt)
    # 定义结果写入的文件路径
    write_path = r"D:\project\20231108code-clean\linshixuqiu\19983-00"
    # 定义图像存储的子目录
    img_bucket_path = "imgs"
    # 创建一个 DiskReaderWriter 对象，用于向指定路径写入图像
    img_writer = DiskReaderWriter(join_path(write_path, img_bucket_path))

    # 注释掉的代码：根据 PDF 内容分类，得到 PDF 类型
    # pdf_type = UNIPipe.classify(pdf_bytes)
    # 定义一个字典，存储有用的键值对，包括 PDF 类型和模型列表
    # jso_useful_key = {
    #     "_pdf_type": pdf_type,
    #     "model_list": model_list
    # }

    # 定义一个字典，初始化 PDF 类型为空字符串，并包含模型列表
    jso_useful_key = {
        "_pdf_type": "",
        "model_list": model_list
    }
    # 创建一个 UNIPipe 实例，传入 PDF 内容、有效键字典和图像写入对象
    pipe = UNIPipe(pdf_bytes, jso_useful_key, img_writer)
    # 调用管道方法进行分类处理
    pipe.pipe_classify()
    # 调用管道方法进行解析处理
    pipe.pipe_parse()
    # 生成 Markdown 内容，基于图像存储路径
    md_content = pipe.pipe_mk_markdown(img_bucket_path)
    # 生成统一格式的内容列表，基于图像存储路径
    content_list = pipe.pipe_mk_uni_format(img_bucket_path)

    # 创建一个 DiskReaderWriter 对象，用于向指定路径写入 Markdown 文件
    md_writer = DiskReaderWriter(write_path)
    # 将 Markdown 内容写入指定文件
    md_writer.write(md_content, "19983-00.md", AbsReaderWriter.MODE_TXT)
    # 将解析得到的中间数据以 JSON 格式写入指定文件
    md_writer.write(json.dumps(pipe.pdf_mid_data, ensure_ascii=False, indent=4), "19983-00.json",
                    AbsReaderWriter.MODE_TXT)
    # 将统一格式的内容列表写入指定文本文件
    md_writer.write(str(content_list), "19983-00.txt", AbsReaderWriter.MODE_TXT)
```
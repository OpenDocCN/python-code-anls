# `.\MinerU\magic_pdf\integrations\rag\api.py`

```
# 导入操作系统模块
import os
# 从 pathlib 导入 Path 类
from pathlib import Path

# 从 loguru 导入 logger 用于日志记录
from loguru import logger

# 从 magic_pdf.integrations.rag.type 导入 ElementRelation, LayoutElements 和 Node
from magic_pdf.integrations.rag.type import (ElementRelation, LayoutElements,
                                             Node)
# 从 magic_pdf.integrations.rag.utils 导入 inference 函数
from magic_pdf.integrations.rag.utils import inference


# 定义 RagPageReader 类，用于读取页面数据
class RagPageReader:

    # 初始化方法，接收 LayoutElements 类型的 pagedata
    def __init__(self, pagedata: LayoutElements):
        # 创建一个 Node 对象列表，包含 pagedata 中的布局细节
        self.o = [
            Node(
                category_type=v.category_type,  # 设置类别类型
                text=v.text,                     # 设置文本内容
                image_path=v.image_path,         # 设置图片路径
                anno_id=v.anno_id,               # 设置注释 ID
                latex=v.latex,                   # 设置 LaTeX 内容
                html=v.html,                     # 设置 HTML 内容
            ) for v in pagedata.layout_dets  # 遍历布局细节以构建 Node 列表
        ]

        # 保存传入的 pagedata
        self.pagedata = pagedata

    # 定义迭代方法，使对象可迭代
    def __iter__(self):
        return iter(self.o)

    # 获取元素关系映射的方法
    def get_rel_map(self) -> list[ElementRelation]:
        return self.pagedata.extra.element_relation  # 返回额外的元素关系


# 定义 RagDocumentReader 类，用于读取文档数据
class RagDocumentReader:

    # 初始化方法，接收 LayoutElements 类型的 ragdata 列表
    def __init__(self, ragdata: list[LayoutElements]):
        # 创建一个 RagPageReader 对象列表，包含 ragdata 中的每个元素
        self.o = [RagPageReader(v) for v in ragdata]

    # 定义迭代方法，使对象可迭代
    def __iter__(self):
        return iter(self.o)


# 定义 DataReader 类，用于读取数据文件
class DataReader:

    # 初始化方法，接收路径或目录、方法和输出目录
    def __init__(self, path_or_directory: str, method: str, output_dir: str):
        # 保存传入的路径或目录
        self.path_or_directory = path_or_directory
        # 保存传入的方法
        self.method = method
        # 保存传入的输出目录
        self.output_dir = output_dir
        # 初始化 PDF 文件列表
        self.pdfs = []
        # 如果路径是目录，遍历目录下所有 PDF 文件
        if os.path.isdir(path_or_directory):
            for doc_path in Path(path_or_directory).glob('*.pdf'):
                # 将找到的 PDF 文件路径添加到列表中
                self.pdfs.append(doc_path)
        else:
            # 断言路径以 '.pdf' 结尾
            assert path_or_directory.endswith('.pdf')
            # 将单个 PDF 文件路径添加到列表中
            self.pdfs.append(Path(path_or_directory))

    # 返回文档数量的方法
    def get_documents_count(self) -> int:
        """Returns the number of documents in the directory."""
        return len(self.pdfs)  # 返回 PDF 文件列表的长度

    # 获取特定文档结果的方法
    def get_document_result(self, idx: int) -> RagDocumentReader | None:
        """
        Args:
            idx (int): the index of documents under the
                directory path_or_directory

        Returns:
            RagDocumentReader | None: RagDocumentReader is an iterable object,
            more details @RagDocumentReader
        """
        # 如果索引无效，记录错误并返回 None
        if idx >= self.get_documents_count() or idx < 0:
            logger.error(f'invalid idx: {idx}')
            return None
        # 调用 inference 函数处理指定索引的 PDF 文件
        res = inference(str(self.pdfs[idx]), self.output_dir, self.method)
        # 如果处理结果为 None，记录警告并返回 None
        if res is None:
            logger.warning(f'failed to inference pdf {self.pdfs[idx]}')
            return None
        # 返回处理结果的 RagDocumentReader 对象
        return RagDocumentReader(res)

    # 获取文档文件名的方法
    def get_document_filename(self, idx: int) -> Path:
        """get the filename of the document."""
        return self.pdfs[idx]  # 返回指定索引的 PDF 文件路径
```
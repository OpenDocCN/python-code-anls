# `.\Langchain-Chatchat\document_loaders\FilteredCSVloader.py`

```
## 指定制定列的csv文件加载器

# 导入所需的模块
from langchain.document_loaders import CSVLoader
import csv
from io import TextIOWrapper
from typing import Dict, List, Optional
from langchain.docstore.document import Document
from langchain.document_loaders.helpers import detect_file_encodings

# 定义一个继承自CSVLoader的类FilteredCSVLoader
class FilteredCSVLoader(CSVLoader):
    # 初始化方法，接收多个参数
    def __init__(
            self,
            file_path: str,  # 文件路径
            columns_to_read: List[str],  # 要读取的列
            source_column: Optional[str] = None,  # 源列，默认为None
            metadata_columns: List[str] = [],  # 元数据列，默认为空列表
            csv_args: Optional[Dict] = None,  # CSV参数，默认为None
            encoding: Optional[str] = None,  # 编码方式，默认为None
            autodetect_encoding: bool = False,  # 是否自动检测编码，默认为False
    ):
        # 调用父类CSVLoader的初始化方法，传入相应参数
        super().__init__(
            file_path=file_path,
            source_column=source_column,
            metadata_columns=metadata_columns,
            csv_args=csv_args,
            encoding=encoding,
            autodetect_encoding=autodetect_encoding,
        )
        # 设置当前类的columns_to_read属性为传入的columns_to_read参数

        self.columns_to_read = columns_to_read
    # 加载数据到文档对象中
    def load(self) -> List[Document]:
        """Load data into document objects."""

        # 初始化文档列表
        docs = []
        try:
            # 打开文件并读取数据，使用指定的编码格式
            with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
                # 调用私有方法读取文件内容并存入文档列表
                docs = self.__read_file(csvfile)
        except UnicodeDecodeError as e:
            # 如果遇到编码错误并且允许自动检测编码
            if self.autodetect_encoding:
                # 检测文件编码
                detected_encodings = detect_file_encodings(self.file_path)
                # 遍历检测到的编码
                for encoding in detected_encodings:
                    try:
                        # 使用检测到的编码重新打开文件并读取数据
                        with open(
                            self.file_path, newline="", encoding=encoding.encoding
                        ) as csvfile:
                            # 调用私有方法读取文件内容并存入文档列表
                            docs = self.__read_file(csvfile)
                            break
                    except UnicodeDecodeError:
                        continue
            else:
                # 如果不允许自动检测编码，则抛出运行时错误
                raise RuntimeError(f"Error loading {self.file_path}") from e
        except Exception as e:
            # 捕获其他异常并抛出运行时错误
            raise RuntimeError(f"Error loading {self.file_path}") from e

        # 返回文档列表
        return docs
    # 读取 CSV 文件中的内容并返回文档列表
    def __read_file(self, csvfile: TextIOWrapper) -> List[Document]:
        # 初始化一个空的文档列表
        docs = []
        # 使用 csv.DictReader 读取 CSV 文件内容，**self.csv_args 传递额外参数
        csv_reader = csv.DictReader(csvfile, **self.csv_args)  # type: ignore
        # 遍历 CSV 文件的每一行
        for i, row in enumerate(csv_reader):
            # 检查是否存在要读取的列
            if self.columns_to_read[0] in row:
                # 获取要读取的内容
                content = row[self.columns_to_read[0]]
                # 如果存在来源列，则提取来源信息，否则使用文件路径作为来源
                source = (
                    row.get(self.source_column, None)
                    if self.source_column is not None
                    else self.file_path
                )
                # 构建元数据字典，包括来源和行号
                metadata = {"source": source, "row": i}

                # 遍历其他元数据列，将其添加到元数据字典中
                for col in self.metadata_columns:
                    if col in row:
                        metadata[col] = row[col]

                # 创建文档对象，包括页面内容和元数据
                doc = Document(page_content=content, metadata=metadata)
                # 将文档添加到文档列表中
                docs.append(doc)
            else:
                # 如果要读取的列不存在，则抛出数值错误
                raise ValueError(f"Column '{self.columns_to_read[0]}' not found in CSV file.")

        # 返回文档列表
        return docs
```
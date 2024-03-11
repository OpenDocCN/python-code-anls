# `.\Langchain-Chatchat\server\knowledge_base\utils.py`

```
# 导入必要的模块和变量
import os
from configs import (
    KB_ROOT_PATH,
    CHUNK_SIZE,
    OVERLAP_SIZE,
    ZH_TITLE_ENHANCE,
    logger,
    log_verbose,
    text_splitter_dict,
    LLM_MODELS,
    TEXT_SPLITTER_NAME,
)
import importlib
from text_splitter import zh_title_enhance as func_zh_title_enhance
import langchain.document_loaders
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from pathlib import Path
from server.utils import run_in_thread_pool, get_model_worker_config
import json
from typing import List, Union,Dict, Tuple, Generator
import chardet

# 定义函数，用于验证知识库名称是否合法
def validate_kb_name(knowledge_base_id: str) -> bool:
    # 检查知识库名称中是否包含预期外的字符或路径攻击关键字
    if "../" in knowledge_base_id:
        return False
    return True

# 获取知识库路径
def get_kb_path(knowledge_base_name: str):
    return os.path.join(KB_ROOT_PATH, knowledge_base_name)

# 获取文档路径
def get_doc_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "content")

# 获取向量存储路径
def get_vs_path(knowledge_base_name: str, vector_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "vector_store", vector_name)

# 获取文件路径
def get_file_path(knowledge_base_name: str, doc_name: str):
    return os.path.join(get_doc_path(knowledge_base_name), doc_name)

# 列出文件夹中的所有知识库
def list_kbs_from_folder():
    return [f for f in os.listdir(KB_ROOT_PATH)
            if os.path.isdir(os.path.join(KB_ROOT_PATH, f))]

# 列出文件夹中的所有文件
def list_files_from_folder(kb_name: str):
    doc_path = get_doc_path(kb_name)
    result = []

    # 定义函数，用于检查是否跳过特定路径
    def is_skiped_path(path: str):
        tail = os.path.basename(path).lower()
        for x in ["temp", "tmp", ".", "~$"]:
            if tail.startswith(x):
                return True
        return False
    # 处理给定的目录项
    def process_entry(entry):
        # 如果路径被跳过，则直接返回
        if is_skiped_path(entry.path):
            return

        # 如果是符号链接
        if entry.is_symlink():
            # 获取符号链接的目标路径
            target_path = os.path.realpath(entry.path)
            # 遍历目标路径下的所有目录项
            with os.scandir(target_path) as target_it:
                for target_entry in target_it:
                    # 递归处理目标路径下的每个目录项
                    process_entry(target_entry)
        # 如果是文件
        elif entry.is_file():
            # 将文件路径转换为 posix 格式，并添加到结果列表中
            file_path = (Path(os.path.relpath(entry.path, doc_path)).as_posix())
            result.append(file_path)
        # 如果是目录
        elif entry.is_dir():
            # 遍历目录下的所有目录项
            with os.scandir(entry.path) as it:
                for sub_entry in it:
                    # 递归处理每个子目录项
                    process_entry(sub_entry)

    # 使用 os.scandir 遍历给定目录下的所有目录项
    with os.scandir(doc_path) as it:
        for entry in it:
            # 处理每个目录项
            process_entry(entry)

    # 返回结果列表
    return result
# 定义一个字典，将不同类型的文档加载器与其支持的文件扩展名对应起来
LOADER_DICT = {"UnstructuredHTMLLoader": ['.html', '.htm'],
               "MHTMLLoader": ['.mhtml'],
               "UnstructuredMarkdownLoader": ['.md'],
               "JSONLoader": [".json"],
               "JSONLinesLoader": [".jsonl"],
               "CSVLoader": [".csv"],
               # "FilteredCSVLoader": [".csv"], 如果使用自定义分割csv
               "RapidOCRPDFLoader": [".pdf"],
               "RapidOCRDocLoader": ['.docx', '.doc'],
               "RapidOCRPPTLoader": ['.ppt', '.pptx', ],
               "RapidOCRLoader": ['.png', '.jpg', '.jpeg', '.bmp'],
               "UnstructuredFileLoader": ['.eml', '.msg', '.rst',
                                          '.rtf', '.txt', '.xml',
                                          '.epub', '.odt','.tsv'],
               "UnstructuredEmailLoader": ['.eml', '.msg'],
               "UnstructuredEPubLoader": ['.epub'],
               "UnstructuredExcelLoader": ['.xlsx', '.xls', '.xlsd'],
               "NotebookLoader": ['.ipynb'],
               "UnstructuredODTLoader": ['.odt'],
               "PythonLoader": ['.py'],
               "UnstructuredRSTLoader": ['.rst'],
               "UnstructuredRTFLoader": ['.rtf'],
               "SRTLoader": ['.srt'],
               "TomlLoader": ['.toml'],
               "UnstructuredTSVLoader": ['.tsv'],
               "UnstructuredWordDocumentLoader": ['.docx', '.doc'],
               "UnstructuredXMLLoader": ['.xml'],
               "UnstructuredPowerPointLoader": ['.ppt', '.pptx'],
               "EverNoteLoader": ['.enex'],
               }
# 将所有支持的文件扩展名放入一个列表中
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]

# 定义一个新的函数_new_json_dumps，用于覆盖json.dumps方法，禁用ensure_ascii选项
def _new_json_dumps(obj, **kwargs):
    kwargs["ensure_ascii"] = False
    return _origin_json_dumps(obj, **kwargs)

# 如果当前的json.dumps方法不等于新定义的_new_json_dumps方法，则进行替换
if json.dumps is not _new_json_dumps:
    # 保存原始的json.dumps方法
    _origin_json_dumps = json.dumps
    # 将json.dumps方法替换为新定义的_new_json_dumps方法
    json.dumps = _new_json_dumps

# 定义一个JSONLinesLoader类，继承自langchain.document_loaders.JSONLoader类
class JSONLinesLoader(langchain.document_loaders.JSONLoader):
    '''
    # 行式 Json 加载器，要求文件扩展名为 .jsonl
    '''
    # 初始化方法，继承父类的初始化方法
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 设置属性 _json_lines 为 True，表示使用行式 Json 加载器
        self._json_lines = True
# 将JSONLinesLoader赋值给langchain.document_loaders.JSONLinesLoader
langchain.document_loaders.JSONLinesLoader = JSONLinesLoader

# 根据文件扩展名获取对应的加载器类
def get_LoaderClass(file_extension):
    for LoaderClass, extensions in LOADER_DICT.items():
        if file_extension in extensions:
            return LoaderClass

# 根据loader_name和文件路径或内容返回文档加载器
def get_loader(loader_name: str, file_path: str, loader_kwargs: Dict = None):
    '''
    根据loader_name和文件路径或内容返回文档加载器。
    '''
    loader_kwargs = loader_kwargs or {}
    try:
        if loader_name in ["RapidOCRPDFLoader", "RapidOCRLoader", "FilteredCSVLoader",
                           "RapidOCRDocLoader", "RapidOCRPPTLoader"]:
            document_loaders_module = importlib.import_module('document_loaders')
        else:
            document_loaders_module = importlib.import_module('langchain.document_loaders')
        DocumentLoader = getattr(document_loaders_module, loader_name)
    except Exception as e:
        msg = f"为文件{file_path}查找加载器{loader_name}时出错：{e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        document_loaders_module = importlib.import_module('langchain.document_loaders')
        DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")

    # 根据加载器名称设置加载器的特定参数
    if loader_name == "UnstructuredFileLoader":
        loader_kwargs.setdefault("autodetect_encoding", True)
    elif loader_name == "CSVLoader":
        if not loader_kwargs.get("encoding"):
            # 如果未指定 encoding，自动识别文件编码类型，避免langchain loader 加载文件报编码错误
            with open(file_path, 'rb') as struct_file:
                encode_detect = chardet.detect(struct_file.read())
            if encode_detect is None:
                encode_detect = {"encoding": "utf-8"}
            loader_kwargs["encoding"] = encode_detect["encoding"]

    elif loader_name == "JSONLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)
    # 如果加载器名称为"JSONLinesLoader"，则设置默认参数"jq_schema"为"."，"text_content"为False
    elif loader_name == "JSONLinesLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)

    # 使用给定的文件路径和加载器参数创建文档加载器对象
    loader = DocumentLoader(file_path, **loader_kwargs)
    # 返回文档加载器对象
    return loader
# 定义一个函数，用于创建特定的文本分词器
def make_text_splitter(
        splitter_name: str = TEXT_SPLITTER_NAME,  # 分词器名称，默认为全局变量TEXT_SPLITTER_NAME
        chunk_size: int = CHUNK_SIZE,  # 分块大小，默认为全局变量CHUNK_SIZE
        chunk_overlap: int = OVERLAP_SIZE,  # 分块重叠大小，默认为全局变量OVERLAP_SIZE
        llm_model: str = LLM_MODELS[0],  # 语言模型，默认为LLM_MODELS列表的第一个模型
):
    """
    根据参数获取特定的分词器
    """
    splitter_name = splitter_name or "SpacyTextSplitter"  # 如果分词器名称为空，则使用默认值"SpacyTextSplitter"
    except Exception as e:  # 捕获异常并将异常对象赋值给变量e
        print(e)  # 打印异常信息
        text_splitter_module = importlib.import_module('langchain.text_splitter')  # 动态导入langchain.text_splitter模块
        TextSplitter = getattr(text_splitter_module, "RecursiveCharacterTextSplitter")  # 获取TextSplitter类
        text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # 创建TextSplitter对象
        
    # 如果使用SpacyTextSplitter，则可以使用GPU来执行分割，类似于Issue #1287
    # text_splitter._tokenizer.max_length = 37016792
    # text_splitter._tokenizer.prefer_gpu()
    return text_splitter  # 返回文本分词器对象


class KnowledgeFile:
    def __init__(
            self,
            filename: str,  # 文件名
            knowledge_base_name: str,  # 知识库名称
            loader_kwargs: Dict = {},  # 加载器参数，默认为空字典
    ):
        '''
        对应知识库目录中的文件，必须是磁盘上存在的才能进行向量化等操作。
        '''
        self.kb_name = knowledge_base_name  # 知识库名称赋值给实例变量kb_name
        self.filename = str(Path(filename).as_posix())  # 将文件名转换为POSIX路径字符串并赋值给实例变量filename
        self.ext = os.path.splitext(filename)[-1].lower()  # 获取文件扩展名并转换为小写赋值给实例变量ext
        if self.ext not in SUPPORTED_EXTS:  # 如果文件扩展名不在支持的扩展名列表中
            raise ValueError(f"暂未支持的文件格式 {self.filename}")  # 抛出值错误异常，提示不支持的文件格式
        self.loader_kwargs = loader_kwargs  # 加载器参数赋值给实例变量loader_kwargs
        self.filepath = get_file_path(knowledge_base_name, filename)  # 获取文件路径并赋值给实例变量filepath
        self.docs = None  # 初始化实例变量docs为None
        self.splited_docs = None  # 初始化实例变量splited_docs为None
        self.document_loader_name = get_LoaderClass(self.ext)  # 获取加载器类名并赋值给实例变量document_loader_name
        self.text_splitter_name = TEXT_SPLITTER_NAME  # 全局变量TEXT_SPLITTER_NAME赋值给实例变量text_splitter_name
    # 将文件转换为文档对象
    def file2docs(self, refresh: bool = False):
        # 如果文档对象为空或需要刷新
        if self.docs is None or refresh:
            # 记录使用的文档加载器和文件路径
            logger.info(f"{self.document_loader_name} used for {self.filepath}")
            # 获取文档加载器
            loader = get_loader(loader_name=self.document_loader_name,
                                file_path=self.filepath,
                                loader_kwargs=self.loader_kwargs)
            # 加载文档
            self.docs = loader.load()
        # 返回文档对象
        return self.docs

    # 将文档转换为文本
    def docs2texts(
            self,
            docs: List[Document] = None,
            zh_title_enhance: bool = ZH_TITLE_ENHANCE,
            refresh: bool = False,
            chunk_size: int = CHUNK_SIZE,
            chunk_overlap: int = OVERLAP_SIZE,
            text_splitter: TextSplitter = None,
    ):
        # 如果未提供文档对象，则调用file2docs方法获取文档
        docs = docs or self.file2docs(refresh=refresh)
        # 如果文档为空，则返回空列表
        if not docs:
            return []
        # 如果文件扩展名不是csv
        if self.ext not in [".csv"]:
            # 如果文本分割器为空，则根据参数创建文本分割器
            if text_splitter is None:
                text_splitter = make_text_splitter(splitter_name=self.text_splitter_name, chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
            # 如果文本分割器名称为MarkdownHeaderTextSplitter，则对第一个文档内容进行分割
            if self.text_splitter_name == "MarkdownHeaderTextSplitter":
                docs = text_splitter.split_text(docs[0].page_content)
            else:
                # 否则对所有文档进行分割
                docs = text_splitter.split_documents(docs)

        # 如果文档为空，则返回空列表
        if not docs:
            return []

        # 打印文档切分示例
        print(f"文档切分示例：{docs[0]}")
        # 如果需要增强中文标题
        if zh_title_enhance:
            # 对文档进行中文标题增强处理
            docs = func_zh_title_enhance(docs)
        # 将处理后的文档赋值给splited_docs属性
        self.splited_docs = docs
        # 返回处理后的文档
        return self.splited_docs

    # 将文件转换为文本
    def file2text(
            self,
            zh_title_enhance: bool = ZH_TITLE_ENHANCE,
            refresh: bool = False,
            chunk_size: int = CHUNK_SIZE,
            chunk_overlap: int = OVERLAP_SIZE,
            text_splitter: TextSplitter = None,
    # 如果分割后的文档为空或需要刷新，则重新生成文档并分割文本
    if self.splited_docs is None or refresh:
        # 从文件中读取文档
        docs = self.file2docs()
        # 将文档转换为文本，并进行分割
        self.splited_docs = self.docs2texts(docs=docs,
                                            zh_title_enhance=zh_title_enhance,
                                            refresh=refresh,
                                            chunk_size=chunk_size,
                                            chunk_overlap=chunk_overlap,
                                            text_splitter=text_splitter)
    # 返回分割后的文档
    return self.splited_docs

# 检查文件是否存在
def file_exist(self):
    return os.path.isfile(self.filepath)

# 获取文件的修改时间
def get_mtime(self):
    return os.path.getmtime(self.filepath)

# 获取文件的大小
def get_size(self):
    return os.path.getsize(self.filepath)
# 定义一个函数，用于将磁盘文件转化成langchain Document，并利用多线程处理
def files2docs_in_thread(
        files: List[Union[KnowledgeFile, Tuple[str, str], Dict]],  # 接受文件列表作为参数，可以是KnowledgeFile对象、元组或字典
        chunk_size: int = CHUNK_SIZE,  # 定义块大小，默认为CHUNK_SIZE
        chunk_overlap: int = OVERLAP_SIZE,  # 定义块重叠大小，默认为OVERLAP_SIZE
        zh_title_enhance: bool = ZH_TITLE_ENHANCE,  # 是否增强中文标题，默认为ZH_TITLE_ENHANCE
) -> Generator:  # 返回一个生成器对象
    '''
    利用多线程批量将磁盘文件转化成langchain Document.
    如果传入参数是Tuple，形式为(filename, kb_name)
    生成器返回值为 status, (kb_name, file_name, docs | error)
    '''

    # 定义一个内部函数，用于将文件转化成Document对象
    def file2docs(*, file: KnowledgeFile, **kwargs) -> Tuple[bool, Tuple[str, str, List[Document]]]:
        try:
            return True, (file.kb_name, file.filename, file.file2text(**kwargs))  # 尝试将文件转化成文本并返回元组
        except Exception as e:
            msg = f"从文件 {file.kb_name}/{file.filename} 加载文档时出错：{e}"  # 捕获异常并记录错误信息
            logger.error(f'{e.__class__.__name__}: {msg}',  # 记录错误信息到日志
                         exc_info=e if log_verbose else None)
            return False, (file.kb_name, file.filename, msg)  # 返回错误信息

    kwargs_list = []  # 初始化参数列表
    for i, file in enumerate(files):  # 遍历文件列表
        kwargs = {}  # 初始化参数字典
        try:
            if isinstance(file, tuple) and len(file) >= 2:  # 如果文件是元组且长度大于等于2
                filename = file[0]  # 获取文件名
                kb_name = file[1]  # 获取知识库名称
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)  # 创建KnowledgeFile对象
            elif isinstance(file, dict):  # 如果文件是字典
                filename = file.pop("filename")  # 弹出文件名
                kb_name = file.pop("kb_name")  # 弹出知识库名称
                kwargs.update(file)  # 更新参数字典
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)  # 创建KnowledgeFile对象
            kwargs["file"] = file  # 添加文件到参数字典
            kwargs["chunk_size"] = chunk_size  # 添加块大小到参数字典
            kwargs["chunk_overlap"] = chunk_overlap  # 添加块重叠大小到参数字典
            kwargs["zh_title_enhance"] = zh_title_enhance  # 添加是否增强中文标题到参数字典
            kwargs_list.append(kwargs)  # 将参数字典添加到参数列表
        except Exception as e:
            yield False, (kb_name, filename, str(e))  # 捕获异常并返回错误信息

    # 使用线程池运行file2docs函数，并返回结果
    for result in run_in_thread_pool(func=file2docs, params=kwargs_list):
        yield result


if __name__ == "__main__":
    from pprint import pprint
    # 创建一个 KnowledgeFile 对象，指定文件路径和知识库名称
    kb_file = KnowledgeFile(
        filename="/home/congyin/Code/Project_Langchain_0814/Langchain-Chatchat/knowledge_base/csv1/content/gm.csv",
        knowledge_base_name="samples")
    # 设置文本分割器的名称为 "RecursiveCharacterTextSplitter"，但是这行代码被注释掉了
    # kb_file.text_splitter_name = "RecursiveCharacterTextSplitter"
    # 将文件转换为文档对象列表
    docs = kb_file.file2docs()
    # 打印文档列表中最后一个文档的内容
    # pprint(docs[-1])
```
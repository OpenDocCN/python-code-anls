# `.\Langchain-Chatchat\server\knowledge_base\kb_doc_api.py`

```py
# 导入必要的库
import os
import urllib
from fastapi import File, Form, Body, Query, UploadFile
# 从 configs 模块中导入配置信息
from configs import (DEFAULT_VS_TYPE, EMBEDDING_MODEL,
                     VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD,
                     CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE,
                     logger, log_verbose, )
# 从 server.utils 模块中导入 BaseResponse 和 ListResponse 类，以及 run_in_thread_pool 函数
from server.utils import BaseResponse, ListResponse, run_in_thread_pool
# 从 server.knowledge_base.utils 模块中导入一些函数和类
from server.knowledge_base.utils import (validate_kb_name, list_files_from_folder, get_file_path,
                                         files2docs_in_thread, KnowledgeFile)
# 从 fastapi.responses 模块中导入 FileResponse 类
from fastapi.responses import FileResponse
# 从 sse_starlette 模块中导入 EventSourceResponse 类
from sse_starlette import EventSourceResponse
# 从 pydantic 模块中导入 Json 类
from pydantic import Json
# 导入 json 库
import json
# 从 server.knowledge_base.kb_service.base 模块中导入 KBServiceFactory 类
from server.knowledge_base.kb_service.base import KBServiceFactory
# 从 server.db.repository.knowledge_file_repository 模块中导入 get_file_detail 函数
from server.db.repository.knowledge_file_repository import get_file_detail
# 从 langchain.docstore.document 模块中导入 Document 类
from langchain.docstore.document import Document
# 从 server.knowledge_base.model.kb_document_model 模块中导入 DocumentWithVSId 类
from server.knowledge_base.model.kb_document_model import DocumentWithVSId
# 导入 typing 模块中的 List 和 Dict 类型
from typing import List, Dict

# 定义一个函数 search_docs，接收一些参数并返回 DocumentWithVSId 类型的列表
def search_docs(
        query: str = Body("", description="用户输入", examples=["你好"]),
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
        score_threshold: float = Body(SCORE_THRESHOLD,
                                      description="知识库匹配相关度阈值，取值范围在0-1之间，"
                                                  "SCORE越小，相关度越高，"
                                                  "取到1相当于不筛选，建议设置在0.5左右",
                                      ge=0, le=1),
        file_name: str = Body("", description="文件名称，支持 sql 通配符"),
        metadata: dict = Body({}, description="根据 metadata 进行过滤，仅支持一级键"),
) -> List[DocumentWithVSId]:
    # 根据知识库名称获取对应的 KBService 实例
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    # 初始化一个空列表 data
    data = []
    # 如果知识库对象不为空
    if kb is not None:
        # 如果查询条件不为空
        if query:
            # 在知识库中搜索文档，返回前 top_k 个结果，且分数大于 score_threshold
            docs = kb.search_docs(query, top_k, score_threshold)
            # 将搜索结果转换为 DocumentWithVSId 对象列表，包括文档内容、分数和 ID
            data = [DocumentWithVSId(**x[0].dict(), score=x[1], id=x[0].metadata.get("id")) for x in docs]
        # 如果文件名或元数据不为空
        elif file_name or metadata:
            # 列出知识库中的文档，根据文件名或元数据筛选
            data = kb.list_docs(file_name=file_name, metadata=metadata)
            # 遍历文档列表
            for d in data:
                # 如果文档的元数据中包含向量信息
                if "vector" in d.metadata:
                    # 删除文档的向量信息
                    del d.metadata["vector"]
    # 返回处理后的文档数据
    return data
# 按照文档 ID 更新文档内容
def update_docs_by_id(
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        docs: Dict[str, Document] = Body(..., description="要更新的文档内容，形如：{id: Document, ...}")
) -> BaseResponse:
    # 根据知识库名称获取对应的知识库服务
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    # 如果知识库不存在，则返回错误响应
    if kb is None:
        return BaseResponse(code=500, msg=f"指定的知识库 {knowledge_base_name} 不存在")
    # 调用知识库服务的更新文档方法，根据返回结果返回相应响应
    if kb.update_doc_by_ids(docs=docs):
        return BaseResponse(msg=f"文档更新成功")
    else:
        return BaseResponse(msg=f"文档更新失败")


# 列出知识库中的所有文件
def list_files(
        knowledge_base_name: str
) -> ListResponse:
    # 验证知识库名称是否合法，防止攻击
    if not validate_kb_name(knowledge_base_name):
        return ListResponse(code=403, msg="Don't attack me", data=[])
    
    # 对知识库名称进行 URL 解码
    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    # 根据知识库名称获取对应的知识库服务
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    # 如果知识库不存在，则返回错误响应
    if kb is None:
        return ListResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}", data=[])
    else:
        # 调用知识库服务的列出文件方法，返回文件名列表
        all_doc_names = kb.list_files()
        return ListResponse(data=all_doc_names)


# 通过多线程将上传的文件保存到对应知识库目录内
def _save_files_in_thread(files: List[UploadFile],
                          knowledge_base_name: str,
                          override: bool):
    """
    通过多线程将上传的文件保存到对应知识库目录内。
    生成器返回保存结果：{"code":200, "msg": "xxx", "data": {"knowledge_base_name":"xxx", "file_name": "xxx"}}
    """
    # 定义保存单个文件的函数，接受上传文件、知识库名称和覆盖标志作为参数，返回一个字典
    def save_file(file: UploadFile, knowledge_base_name: str, override: bool) -> dict:
        '''
        保存单个文件。
        '''
        try:
            # 获取上传文件的文件名
            filename = file.filename
            # 获取文件路径
            file_path = get_file_path(knowledge_base_name=knowledge_base_name, doc_name=filename)
            # 构建数据字典
            data = {"knowledge_base_name": knowledge_base_name, "file_name": filename}

            # 读取上传文件的内容
            file_content = file.file.read()
            # 检查文件是否已存在且不覆盖，并且文件大小与上传文件大小相同
            if (os.path.isfile(file_path)
                    and not override
                    and os.path.getsize(file_path) == len(file_content)
            ):
                file_status = f"文件 {filename} 已存在。"
                logger.warn(file_status)
                return dict(code=404, msg=file_status, data=data)

            # 如果文件路径不存在，则创建目录
            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            # 将文件内容写入文件
            with open(file_path, "wb") as f:
                f.write(file_content)
            return dict(code=200, msg=f"成功上传文件 {filename}", data=data)
        except Exception as e:
            # 处理异常情况
            msg = f"{filename} 文件上传失败，报错信息为: {e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            return dict(code=500, msg=msg, data=data)

    # 构建参数列表，每个参数包含文件、知识库名称和覆盖标志
    params = [{"file": file, "knowledge_base_name": knowledge_base_name, "override": override} for file in files]
    # 在线程池中运行保存文件函数，并逐个返回结果
    for result in run_in_thread_pool(save_file, params=params):
        yield result
# 定义一个上传文档的函数，接受多个文件、知识库名称、覆盖选项、向量化选项、文本分段大小、文本重叠大小、中文标题加强选项、自定义文档、不刷新向量库缓存选项，并返回基本响应对象
def upload_docs(
        files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
        knowledge_base_name: str = Form(..., description="知识库名称", examples=["samples"]),
        override: bool = Form(False, description="覆盖已有文件"),
        to_vector_store: bool = Form(True, description="上传文件后是否进行向量化"),
        chunk_size: int = Form(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Form(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
        docs: Json = Form({}, description="自定义的docs，需要转为json字符串",
                          examples=[{"test.txt": [Document(page_content="custom doc")]}]),
        not_refresh_vs_cache: bool = Form(False, description="暂不保存向量库（用于FAISS）"),
) -> BaseResponse:
    """
    API接口：上传文件，并/或向量化
    """
    # 如果知识库名称不合法，返回禁止访问的响应
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    # 根据知识库名称获取知识库服务对象
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    # 如果知识库对象不存在，返回未找到知识库的响应
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    # 初始化一个空字典用于存储上传失败的文件
    failed_files = {}
    # 获取自定义文档的文件名列表
    file_names = list(docs.keys())

    # 先将上传的文件保存到磁盘
    # 遍历保存文件的结果列表
    for result in _save_files_in_thread(files, knowledge_base_name=knowledge_base_name, override=override):
        # 获取文件名
        filename = result["data"]["file_name"]
        # 如果保存文件的结果代码不是 200，则将文件名和消息添加到失败文件字典中
        if result["code"] != 200:
            failed_files[filename] = result["msg"]

        # 如果文件名不在文件名列表中，则将文件名添加到文件名列表中
        if filename not in file_names:
            file_names.append(filename)

    # 如果需要进行向量化处理
    if to_vector_store:
        # 更新文档向量
        result = update_docs(
            knowledge_base_name=knowledge_base_name,
            file_names=file_names,
            override_custom_docs=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
            docs=docs,
            not_refresh_vs_cache=True,
        )
        # 更新失败文件字典
        failed_files.update(result.data["failed_files"])
        # 如果不是不刷新向量存储缓存，则保存向量存储
        if not not_refresh_vs_cache:
            kb.save_vector_store()

    # 返回成功的响应对象
    return BaseResponse(code=200, msg="文件上传与向量化完成", data={"failed_files": failed_files})
# 删除知识库中的文档
def delete_docs(
        # 知识库名称，默认为"sample"
        knowledge_base_name: str = Body(..., examples=["samples"]),
        # 待删除的文件名列表
        file_names: List[str] = Body(..., examples=[["file_name.md", "test.txt"]]),
        # 是否删除文档内容，默认为False
        delete_content: bool = Body(False),
        # 是否暂不保存向量库（用于FAISS），默认为False
        not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
) -> BaseResponse:
    # 验证知识库名称是否合法
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    # 对知识库名称进行URL解码
    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    # 根据知识库名称获取知识库服务
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    # 如果知识库不存在，则返回错误响应
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    # 初始化失败文件字典
    failed_files = {}
    # 遍历待删除的文件名列表
    for file_name in file_names:
        # 如果文件不存在于知识库中，记录到失败文件字典中
        if not kb.exist_doc(file_name):
            failed_files[file_name] = f"未找到文件 {file_name}"

        try:
            # 创建知识库文件对象
            kb_file = KnowledgeFile(filename=file_name,
                                    knowledge_base_name=knowledge_base_name)
            # 删除文档
            kb.delete_doc(kb_file, delete_content, not_refresh_vs_cache=True)
        except Exception as e:
            # 记录删除失败的文件信息
            msg = f"{file_name} 文件删除失败，错误信息：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            failed_files[file_name] = msg

    # 如果不暂不保存向量库，则保存向量库
    if not not_refresh_vs_cache:
        kb.save_vector_store()

    # 返回成功响应
    return BaseResponse(code=200, msg=f"文件删除完成", data={"failed_files": failed_files})


# 更新知识库信息
def update_info(
        # 知识库名称
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        # 知识库介绍
        kb_info: str = Body(..., description="知识库介绍", examples=["这是一个知识库"]),
):
    # 验证知识库名称是否合法
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    # 根据知识库名称获取知识库服务
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    # 如果知识库不存在，则返回错误响应
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
    # 更新知识库信息
    kb.update_info(kb_info)
    # 返回一个包含特定状态码、消息和数据的基本响应对象
    return BaseResponse(code=200, msg=f"知识库介绍修改完成", data={"kb_info"})
# 更新知识库文档
def update_docs(
        # 知识库名称，默认为"samples"
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        # 文件名称列表，支持多文件
        file_names: List[str] = Body(..., description="文件名称，支持多文件", examples=[["file_name1", "text.txt"]]),
        # 知识库中单段文本最大长度
        chunk_size: int = Body(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        # 知识库中相邻文本重合长度
        chunk_overlap: int = Body(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        # 是否开启中文标题加强
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
        # 是否覆盖之前自定义的docs
        override_custom_docs: bool = Body(False, description="是否覆盖之前自定义的docs"),
        # 自定义的docs，需要转为json字符串
        docs: Json = Body({}, description="自定义的docs，需要转为json字符串",
                          examples=[{"test.txt": [Document(page_content="custom doc")]}]),
        # 暂不保存向量库（用于FAISS）
        not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
) -> BaseResponse:
    """
    更新知识库文档
    """
    # 验证知识库名称是否合法
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    # 根据知识库名称获取知识库服务
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    # 存储加载失败的文件和知识库文件列表
    failed_files = {}
    kb_files = []

    # 生成需要加载docs的文件列表
    for file_name in file_names:
        # 获取文件详情
        file_detail = get_file_detail(kb_name=knowledge_base_name, filename=file_name)
        # 如果该文件之前使用了自定义docs，则根据参数决定略过或覆盖
        if file_detail.get("custom_docs") and not override_custom_docs:
            continue
        # 如果文件名不在docs中，则尝试加载文档
        if file_name not in docs:
            try:
                kb_files.append(KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name))
            except Exception as e:
                msg = f"加载文档 {file_name} 时出错：{e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                failed_files[file_name] = msg

    # 从文件生成docs，并进行向量化。
    # 这里利用了KnowledgeFile的缓存功能，在多线程中加载Document，然后传给KnowledgeFile
    # 遍历处理每个文件，将文件内容转换为文档对象并添加到知识库中
    for status, result in files2docs_in_thread(kb_files,
                                               chunk_size=chunk_size,
                                               chunk_overlap=chunk_overlap,
                                               zh_title_enhance=zh_title_enhance):
        # 如果处理成功
        if status:
            kb_name, file_name, new_docs = result
            # 创建知识文件对象
            kb_file = KnowledgeFile(filename=file_name,
                                    knowledge_base_name=knowledge_base_name)
            # 设置文件的拆分文档
            kb_file.splited_docs = new_docs
            # 更新知识库中的文档
            kb.update_doc(kb_file, not_refresh_vs_cache=True)
        # 如果处理失败
        else:
            kb_name, file_name, error = result
            # 记录处理失败的文件和错误信息
            failed_files[file_name] = error

    # 将自定义的docs进行向量化
    # 遍历处理每个自定义文档，将其转换为文档对象并添加到知识库中
    for file_name, v in docs.items():
        try:
            # 将自定义文档转换为文档对象
            v = [x if isinstance(x, Document) else Document(**x) for x in v]
            # 创建知识文件对象
            kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name)
            # 更新知识库中的文档
            kb.update_doc(kb_file, docs=v, not_refresh_vs_cache=True)
        except Exception as e:
            # 处理异常情况，记录错误信息
            msg = f"为 {file_name} 添加自定义docs时出错：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            # 记录处理失败的文件和错误信息
            failed_files[file_name] = msg

    # 如果不是指定不刷新向量存储，则保存向量存储
    if not not_refresh_vs_cache:
        kb.save_vector_store()

    # 返回处理结果
    return BaseResponse(code=200, msg=f"更新文档完成", data={"failed_files": failed_files})
# 下载知识库文档
def download_doc(
        # 知识库名称参数，必填，描述为知识库名称，示例为["samples"]
        knowledge_base_name: str = Query(..., description="知识库名称", examples=["samples"]),
        # 文件名称参数，必填，描述为文件名称，示例为["test.txt"]
        file_name: str = Query(..., description="文件名称", examples=["test.txt"]),
        # 预览参数，可选，默认为False，描述为是：浏览器内预览；否：下载
        preview: bool = Query(False, description="是：浏览器内预览；否：下载"),
):
    """
    下载知识库文档
    """
    # 如果知识库名称不合法，返回禁止访问的响应
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    # 根据知识库名称获取知识库服务
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    # 如果知识库服务不存在，返回未找到知识库的响应
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    # 根据预览参数确定内容展示方式
    if preview:
        content_disposition_type = "inline"
    else:
        content_disposition_type = None

    try:
        # 创建知识库文件对象
        kb_file = KnowledgeFile(filename=file_name,
                                knowledge_base_name=knowledge_base_name)

        # 如果知识库文件存在，返回文件响应
        if os.path.exists(kb_file.filepath):
            return FileResponse(
                path=kb_file.filepath,
                filename=kb_file.filename,
                media_type="multipart/form-data",
                content_disposition_type=content_disposition_type,
            )
    except Exception as e:
        # 处理异常情况，记录错误日志并返回服务器错误响应
        msg = f"{kb_file.filename} 读取文件失败，错误信息是：{e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    # 返回读取文件失败的服务器错误响应
    return BaseResponse(code=500, msg=f"{kb_file.filename} 读取文件失败")


# 重新创建向量库
def recreate_vector_store(
        # 知识库名称参数，必填，示例为["samples"]
        knowledge_base_name: str = Body(..., examples=["samples"]),
        # 允许空知识库参数，可选，默认为True
        allow_empty_kb: bool = Body(True),
        # 向量库类型参数，可选，默认为DEFAULT_VS_TYPE
        vs_type: str = Body(DEFAULT_VS_TYPE),
        # 嵌入模型参数，可选，默认为EMBEDDING_MODEL
        embed_model: str = Body(EMBEDDING_MODEL),
        # 分块大小参数，可选，默认为CHUNK_SIZE，描述为知识库中单段文本最大长度
        chunk_size: int = Body(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        # 分块重叠参数，可选，默认为OVERLAP_SIZE，描述为知识库中相邻文本重合长度
        chunk_overlap: int = Body(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        # 中文标题加强参数，可选，默认为ZH_TITLE_ENHANCE，描述为是否开启中文标题加强
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
        # 不刷新向量库缓存参数，可选，默认为False，描述为暂不保存向量库（用于FAISS）
        not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
):
    """
    # 重新创建向量存储器从内容中。
    # 当用户可以直接将文件复制到内容文件夹而不是通过网络上传时，这很有用。
    # 默认情况下，get_service_by_name仅返回info.db中具有文档文件的知识库。
    # 将allow_empty_kb设置为True使其适用于不在info.db中或没有文档的空知识库。
    # 定义一个输出函数
    def output():
        # 获取知识库服务对象
        kb = KBServiceFactory.get_service(knowledge_base_name, vs_type, embed_model)
        # 如果知识库不存在且不允许空知识库，则返回未找到知识库的错误信息
        if not kb.exists() and not allow_empty_kb:
            yield {"code": 404, "msg": f"未找到知识库 ‘{knowledge_base_name}’"}
        else:
            # 如果知识库存在，则清空向量空间
            if kb.exists():
                kb.clear_vs()
            # 创建知识库
            kb.create_kb()
            # 获取文件夹中的文件列表
            files = list_files_from_folder(knowledge_base_name)
            # 构建知识库文件列表
            kb_files = [(file, knowledge_base_name) for file in files]
            i = 0
            # 遍历文件列表，将文件转换为文档并添加到知识库中
            for status, result in files2docs_in_thread(kb_files,
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap,
                                                       zh_title_enhance=zh_title_enhance):
                if status:
                    kb_name, file_name, docs = result
                    kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=kb_name)
                    kb_file.splited_docs = docs
                    # 返回成功信息
                    yield json.dumps({
                        "code": 200,
                        "msg": f"({i + 1} / {len(files)}): {file_name}",
                        "total": len(files),
                        "finished": i + 1,
                        "doc": file_name,
                    }, ensure_ascii=False)
                    # 将文档添加到知识库
                    kb.add_doc(kb_file, not_refresh_vs_cache=True)
                else:
                    kb_name, file_name, error = result
                    msg = f"添加文件‘{file_name}’到知识库‘{knowledge_base_name}’时出错：{error}。已跳过。"
                    logger.error(msg)
                    # 返回错误信息
                    yield json.dumps({
                        "code": 500,
                        "msg": msg,
                    })
                i += 1
            # 如果不是不刷新向量空间缓存，则保存向量空间
            if not not_refresh_vs_cache:
                kb.save_vector_store()

    # 返回事件源响应
    return EventSourceResponse(output())
```
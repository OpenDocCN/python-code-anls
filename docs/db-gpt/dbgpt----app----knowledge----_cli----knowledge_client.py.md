# `.\DB-GPT-src\dbgpt\app\knowledge\_cli\knowledge_client.py`

```py
# 导入所需的模块
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import requests  # 导入处理 HTTP 请求的模块
from prettytable import PrettyTable  # 导入用于创建美观表格的模块

# 导入自定义模块和类
from dbgpt.app.knowledge.request.request import (
    ChunkQueryRequest,
    DocumentQueryRequest,
    DocumentSyncRequest,
    KnowledgeDocumentRequest,
    KnowledgeQueryRequest,
    KnowledgeSpaceRequest,
)
from dbgpt.app.openapi.api_view_model import Result
from dbgpt.rag.knowledge.base import KnowledgeType

# 设置 HTTP 请求的头部信息
HTTP_HEADERS = {"Content-Type": "application/json"}

# 设置日志记录器的名称
logger = logging.getLogger("dbgpt_cli")

# 定义一个 API 客户端类
class ApiClient:
    def __init__(self, api_address: str) -> None:
        self.api_address = api_address

    # 处理 HTTP 响应的内部方法
    def _handle_response(self, response):
        if 200 <= response.status_code <= 300:
            # 将响应内容解析为 Result 对象
            result = Result(**response.json())
            if not result.success:
                # 如果请求不成功，则抛出异常，显示错误消息
                raise Exception(result.err_msg)
            return result.data
        else:
            # 如果 HTTP 请求返回错误状态码，抛出异常，显示错误信息
            raise Exception(
                f"Http request error, code: {response.status_code}, message: {response.text}"
            )

    # 发送 POST 请求的内部方法
    def _post(self, url: str, data=None):
        # 如果数据不是字典，则将其转换为字典
        if not isinstance(data, dict):
            data = data.__dict__
        # 拼接完整的请求 URL
        url = urljoin(self.api_address, url)
        # 记录调试信息，显示请求的 URL 和数据内容
        logger.debug(f"Send request to {url}, data: {data}")
        # 发送 POST 请求，传递 JSON 格式的数据和请求头信息
        response = requests.post(url, data=json.dumps(data), headers=HTTP_HEADERS)
        # 处理并返回 HTTP 响应
        return self._handle_response(response)

# 继承自 ApiClient 的知识库 API 客户端类
class KnowledgeApiClient(ApiClient):
    def __init__(self, api_address: str) -> None:
        super().__init__(api_address)

    # 向知识库添加空间的方法
    def space_add(self, request: KnowledgeSpaceRequest):
        try:
            return self._post("/knowledge/space/add", data=request)
        except Exception as e:
            # 如果空间名称已存在，则记录警告日志
            if "have already named" in str(e):
                logger.warn(f"you have already named {request.name}")
            else:
                # 否则，继续抛出异常
                raise e

    # 删除知识库中的空间的方法
    def space_delete(self, request: KnowledgeSpaceRequest):
        return self._post("/knowledge/space/delete", data=request)

    # 列出知识库中所有空间的方法
    def space_list(self, request: KnowledgeSpaceRequest):
        return self._post("/knowledge/space/list", data=request)

    # 向指定空间添加文档的方法
    def document_add(self, space_name: str, request: KnowledgeDocumentRequest):
        url = f"/knowledge/{space_name}/document/add"
        return self._post(url, data=request)

    # 删除指定空间中的文档的方法
    def document_delete(self, space_name: str, request: KnowledgeDocumentRequest):
        url = f"/knowledge/{space_name}/document/delete"
        return self._post(url, data=request)

    # 列出指定空间中文档的方法
    def document_list(self, space_name: str, query_request: DocumentQueryRequest):
        url = f"/knowledge/{space_name}/document/list"
        return self._post(url, data=query_request)
    # 定义一个方法，用于上传文档到指定的空间，并使用multipart/form-data格式
    def document_upload(self, space_name, doc_name, doc_type, doc_file_path):
        """Upload with multipart/form-data"""
        # 构建上传文档的URL，包括API地址和空间名称
        url = f"{self.api_address}/knowledge/{space_name}/document/upload"
        # 打开指定路径下的文档文件，以二进制方式读取
        with open(doc_file_path, "rb") as f:
            # 将文档文件封装为files字典的一部分
            files = {"doc_file": f}
            # 准备上传时需要的数据，包括文档名称和文档类型
            data = {"doc_name": doc_name, "doc_type": doc_type}
            # 发起POST请求，将数据和文件一起提交
            response = requests.post(url, data=data, files=files)
        # 处理服务器响应并返回处理结果
        return self._handle_response(response)

    # 定义一个方法，用于同步文档到指定的空间
    def document_sync(self, space_name: str, request: DocumentSyncRequest):
        # 构建同步文档的URL，包括空间名称
        url = f"/knowledge/{space_name}/document/sync"
        # 发起POST请求，将请求对象作为数据提交
        return self._post(url, data=request)

    # 定义一个方法，用于查询指定空间下的分块列表
    def chunk_list(self, space_name: str, query_request: ChunkQueryRequest):
        # 构建查询分块列表的URL，包括空间名称
        url = f"/knowledge/{space_name}/chunk/list"
        # 发起POST请求，将查询请求对象作为数据提交
        return self._post(url, data=query_request)

    # 定义一个方法，用于查询与给定向量名称相关的知识
    def similar_query(self, vector_name: str, query_request: KnowledgeQueryRequest):
        # 构建查询相似知识的URL，包括向量名称
        url = f"/knowledge/{vector_name}/query"
        # 发起POST请求，将查询请求对象作为数据提交
        return self._post(url, data=query_request)
    client = KnowledgeApiClient(api_address)
    # 创建一个 KnowledgeApiClient 对象，连接到指定的 API 地址

    space = KnowledgeSpaceRequest()
    # 创建一个 KnowledgeSpaceRequest 对象，用于描述知识空间

    space.name = space_name
    # 设置知识空间的名称为给定的 space_name

    space.desc = "DB-GPT cli"
    # 设置知识空间的描述为固定的字符串 "DB-GPT cli"

    space.vector_type = vector_store_type
    # 设置知识空间的向量类型为给定的 vector_store_type

    space.owner = "DB-GPT"
    # 设置知识空间的所有者为固定的字符串 "DB-GPT"

    # Create space
    logger.info(f"Create space: {space}")
    # 记录日志，表示开始创建知识空间，并输出空间的详细信息
    client.space_add(space)
    # 调用 API 客户端的 space_add 方法，向服务器添加新的知识空间
    logger.info("Create space successfully")
    # 记录日志，表示成功创建知识空间

    space_list = client.space_list(KnowledgeSpaceRequest(name=space.name))
    # 调用 API 客户端的 space_list 方法，获取指定名称的知识空间列表

    if len(space_list) != 1:
        raise Exception(f"List space {space.name} error")
    # 如果返回的知识空间列表长度不为 1，则抛出异常，表示获取空间信息出错

    space = KnowledgeSpaceRequest(**space_list[0])
    # 根据返回的知识空间列表的第一个元素，更新当前的 space 对象

    doc_ids = []

    def upload(filename: str):
        # 定义一个内部函数 upload，用于上传文档至知识空间
        try:
            logger.info(f"Begin upload document: {filename} to {space.name}")
            # 记录日志，表示开始上传指定文件到当前知识空间的操作

            doc_id = None
            try:
                doc_id = client.document_upload(
                    space.name, filename, KnowledgeType.DOCUMENT.value, filename
                )
                # 尝试调用 API 客户端的 document_upload 方法，将文件上传到知识空间
            except Exception as ex:
                if overwrite and "have already named" in str(ex):
                    logger.warn(
                        f"Document {filename} already exist in space {space.name}, overwrite it"
                    )
                    # 如果文件已存在且 overwrite 标志为 True，则记录警告日志并覆盖现有文档
                    client.document_delete(
                        space.name, KnowledgeDocumentRequest(doc_name=filename)
                    )
                    doc_id = client.document_upload(
                        space.name, filename, KnowledgeType.DOCUMENT.value, filename
                    )
                    # 删除现有文档后再次尝试上传文件
                else:
                    raise ex
                    # 如果不是因为文档已存在的错误，则抛出原始异常

            sync_req = DocumentSyncRequest(doc_ids=[doc_id])
            # 创建一个 DocumentSyncRequest 请求对象，包含要同步的文档 ID 列表
            if pre_separator:
                sync_req.pre_separator = pre_separator
                # 如果预分隔符定义了，则设置同步请求的预分隔符字段
            if separator:
                sync_req.separators = [separator]
                # 如果分隔符定义了，则设置同步请求的分隔符列表字段
            if chunk_size:
                sync_req.chunk_size = chunk_size
                # 如果块大小定义了，则设置同步请求的块大小字段
            if chunk_overlap:
                sync_req.chunk_overlap = chunk_overlap
                # 如果块重叠定义了，则设置同步请求的块重叠字段

            client.document_sync(space.name, sync_req)
            # 调用 API 客户端的 document_sync 方法，同步文档到知识空间

            return doc_id
            # 返回上传的文档 ID
        except Exception as e:
            if skip_wrong_doc:
                logger.warn(f"Upload {filename} to {space.name} failed: {str(e)}")
                # 如果允许跳过错误的文档上传，则记录警告日志
            else:
                raise e
                # 否则抛出异常，终止上传操作

    if not os.path.exists(local_doc_path):
        raise Exception(f"{local_doc_path} not exists")
    # 如果本地文档路径不存在，则抛出异常，表示路径错误或文件不存在
    # 使用线程池执行并发任务，最大并发数由 max_workers 指定
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # 初始化任务列表和文件名列表
        tasks = []
        file_names = []
    
        # 如果本地文档路径是一个目录
        if os.path.isdir(local_doc_path):
            # 遍历目录下的所有文件及其路径
            for root, _, files in os.walk(local_doc_path, topdown=False):
                for file in files:
                    file_names.append(os.path.join(root, file))
        else:
            # 如果本地文档路径是一个单独的文件，则直接添加到文件名列表中
            file_names.append(local_doc_path)
    
        # 提交每个文件上传任务到线程池
        [tasks.append(pool.submit(upload, filename)) for filename in file_names]
    
        # 等待所有任务完成并获取结果
        doc_ids = [r.result() for r in as_completed(tasks)]
    
        # 过滤掉空的文档 ID
        doc_ids = list(filter(lambda x: x, doc_ids))
    
        # 如果没有有效的文档 ID 返回警告并退出
        if not doc_ids:
            logger.warn("Warning: no document to sync")
            return
# _KnowledgeVisualizer 类定义，用于处理知识可视化相关操作
class _KnowledgeVisualizer:
    # 初始化方法，接受 API 地址和输出格式作为参数
    def __init__(self, api_address: str, out_format: str):
        # 创建 KnowledgeApiClient 实例，并保存到 self.client 属性中
        self.client = KnowledgeApiClient(api_address)
        # 保存输出格式到 self.out_format 属性中
        self.out_format = out_format
        # 初始化输出参数字典 self.out_kwargs
        self.out_kwargs = {}
        # 如果输出格式为 "json"，设置输出参数 ensure_ascii 为 False
        if out_format == "json":
            self.out_kwargs["ensure_ascii"] = False

    # 打印表格的方法，接受 PrettyTable 实例 table 作为参数
    def print_table(self, table):
        # 调用 PrettyTable 实例的方法，根据 self.out_format 和 self.out_kwargs 打印格式化的字符串
        print(table.get_formatted_string(out_format=self.out_format, **self.out_kwargs))

    # 列出所有的知识空间的方法
    def list_spaces(self):
        # 获取所有知识空间的信息列表
        spaces = self.client.space_list(KnowledgeSpaceRequest())
        # 创建一个标题为 "All knowledge spaces" 的 PrettyTable 实例 table
        table = PrettyTable(
            ["Space ID", "Space Name", "Vector Type", "Owner", "Description"],
            title="All knowledge spaces",
        )
        # 遍历每个空间信息，添加到表格中
        for sp in spaces:
            # 获取空间的上下文信息
            context = sp.get("context")
            # 向表格添加一行，包括空间的 ID、名称、向量类型、所有者和描述信息
            table.add_row(
                [
                    sp.get("id"),
                    sp.get("name"),
                    sp.get("vector_type"),
                    sp.get("owner"),
                    sp.get("desc"),
                ]
            )
        # 打印表格
        self.print_table(table)

    # 列出指定知识空间的所有文档的方法
    def list_documents(self, space_name: str, page: int, page_size: int):
        # 获取指定知识空间中的文档列表信息
        space_data = self.client.document_list(
            space_name, DocumentQueryRequest(page=page, page_size=page_size)
        )

        # 创建一个标题为 "Space {space_name} description" 的 PrettyTable 实例 space_table
        space_table = PrettyTable(
            [
                "Space Name",
                "Total Documents",
                "Current Page",
                "Current Size",
                "Page Size",
            ],
            title=f"Space {space_name} description",
        )
        # 向 space_table 表格添加一行，显示空间名称、总文档数、当前页码、当前数据条数和每页显示条数
        space_table.add_row(
            [space_name, space_data["total"], page, len(space_data["data"]), page_size]
        )

        # 创建一个标题为 "Documents of space {space_name}" 的 PrettyTable 实例 table
        table = PrettyTable(
            [
                "Space Name",
                "Document ID",
                "Document Name",
                "Type",
                "Chunks",
                "Last Sync",
                "Status",
                "Result",
            ],
            title=f"Documents of space {space_name}",
        )
        # 遍历每个文档信息，向 table 表格添加一行，包括空间名称、文档 ID、文档名称、类型、块数、上次同步时间、状态和结果信息
        for doc in space_data["data"]:
            table.add_row(
                [
                    space_name,
                    doc.get("id"),
                    doc.get("doc_name"),
                    doc.get("doc_type"),
                    doc.get("chunk_size"),
                    doc.get("last_sync"),
                    doc.get("status"),
                    doc.get("result"),
                ]
            )
        # 如果输出格式为 "text"，先打印 space_table 表格，然后换行打印空行
        if self.out_format == "text":
            self.print_table(space_table)
            print("")
        # 打印 table 表格
        self.print_table(table)

    # 列出指定文档块的方法
    def list_chunks(
        self,
        space_name: str,
        doc_id: int,
        page: int,
        page_size: int,
        show_content: bool,

        # 获取指定文档块的信息列表
        chunks_data = self.client.chunk_list(
            space_name, doc_id, ChunkQueryRequest(page=page, page_size=page_size)
        )

        # 创建一个标题为 "Chunks of document {doc_id}" 的 PrettyTable 实例 table
        table = PrettyTable(
            [
                "Space Name",
                "Document ID",
                "Chunk ID",
                "Content" if show_content else "Length",
                "Last Modified",
            ],
            title=f"Chunks of document {doc_id}",
        )
        # 遍历每个文档块信息，向 table 表格添加一行，包括空间名称、文档 ID、块 ID、内容或长度（根据 show_content 参数）、最后修改时间
        for chunk in chunks_data["data"]:
            table.add_row(
                [
                    space_name,
                    doc_id,
                    chunk.get("id"),
                    chunk.get("content") if show_content else len(chunk.get("content")),
                    chunk.get("last_modified"),
                ]
            )
        # 打印 table 表格
        self.print_table(table)
        ):
        # 使用 self.client 调用 chunk_list 方法，获取指定空间、文档ID、页码、页大小的文档数据
        doc_data = self.client.chunk_list(
            space_name,
            ChunkQueryRequest(document_id=doc_id, page=page, page_size=page_size),
        )

        # 创建一个 PrettyTable 对象，用于展示文档详细信息
        doc_table = PrettyTable(
            [
                "Space Name",
                "Document ID",
                "Total Chunks",
                "Current Page",
                "Current Size",
                "Page Size",
            ],
            title=f"Document {doc_id} in {space_name} description",
        )
        # 向 doc_table 中添加一行数据，显示空间名、文档ID、总块数、当前页码、当前数据大小、页大小
        doc_table.add_row(
            [
                space_name,
                doc_id,
                doc_data["total"],
                page,
                len(doc_data["data"]),
                page_size,
            ]
        )

        # 创建一个 PrettyTable 对象，用于展示文档各块的详细信息
        table = PrettyTable(
            ["Space Name", "Document ID", "Document Name", "Content", "Meta Data"],
            title=f"chunks of document id {doc_id} in space {space_name}",
        )
        # 遍历 doc_data 中的每个块，向 table 中添加一行数据，显示空间名、文档ID、文档名、内容（如果显示内容选项为真）、元数据
        for chunk in doc_data["data"]:
            table.add_row(
                [
                    space_name,
                    doc_id,
                    chunk.get("doc_name"),
                    chunk.get("content") if show_content else "[Hidden]",
                    chunk.get("meta_info"),
                ]
            )
        # 如果输出格式为文本，则打印 doc_table 和一个空行
        if self.out_format == "text":
            self.print_table(doc_table)
            print("")
        # 打印 table 表格
        self.print_table(table)
# 定义一个函数用于列出知识库相关信息
def knowledge_list(
    api_address: str,
    space_name: str,
    page: int,
    page_size: int,
    doc_id: int,
    show_content: bool,
    out_format: str,
):
    # 创建一个知识可视化对象，用于输出指定格式的知识库信息
    visualizer = _KnowledgeVisualizer(api_address, out_format)
    
    # 如果没有指定空间名，则列出所有空间的信息
    if not space_name:
        visualizer.list_spaces()
    
    # 如果没有指定文档ID，则列出指定空间内的文档列表
    elif not doc_id:
        visualizer.list_documents(space_name, page, page_size)
    
    # 否则，列出指定空间和文档ID对应的文档信息
    else:
        visualizer.list_chunks(space_name, doc_id, page, page_size, show_content)


# 定义一个函数用于删除知识库中的空间或文档
def knowledge_delete(
    api_address: str, space_name: str, doc_name: str, confirm: bool = False
):
    # 创建一个知识库客户端对象
    client = KnowledgeApiClient(api_address)
    
    # 创建一个知识空间请求对象并设置空间名
    space = KnowledgeSpaceRequest()
    space.name = space_name
    
    # 获取指定名称的知识空间列表
    space_list = client.space_list(KnowledgeSpaceRequest(name=space.name))
    
    # 如果空间列表为空，则抛出异常
    if not space_list:
        raise Exception(f"No knowledge space name {space_name}")

    # 如果没有指定文档名
    if not doc_name:
        # 如果没有确认删除操作
        if not confirm:
            # 用户确认操作
            user_input = (
                input(
                    f"Are you sure you want to delete the whole knowledge space {space_name}? Type 'yes' to confirm: "
                )
                .strip()
                .lower()
            )
            # 如果用户不确认删除，则记录警告信息并返回
            if user_input != "yes":
                logger.warn("Delete operation cancelled.")
                return
        
        # 执行删除整个知识空间操作
        client.space_delete(space)
        logger.info("Delete the whole knowledge space successfully!")
    
    # 否则，如果指定了文档名
    else:
        # 如果没有确认删除操作
        if not confirm:
            # 用户确认操作
            user_input = (
                input(
                    f"Are you sure you want to delete the document {doc_name} in knowledge space {space_name}? Type 'yes' to confirm: "
                )
                .strip()
                .lower()
            )
            # 如果用户不确认删除，则记录警告信息并返回
            if user_input != "yes":
                logger.warn("Delete operation cancelled.")
                return
        
        # 执行删除指定文档操作
        client.document_delete(space_name, KnowledgeDocumentRequest(doc_name=doc_name))
        logger.info(
            f"Delete the document {doc_name} in knowledge space {space_name} successfully!"
        )
```
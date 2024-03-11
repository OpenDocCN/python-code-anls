# `.\Langchain-Chatchat\server\chat\file_chat.py`

```py
# 导入必要的模块和类
from fastapi import Body, File, Form, UploadFile
from sse_starlette.sse import EventSourceResponse
from configs import (LLM_MODELS, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE,
                     CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE)
from server.utils import (wrap_done, get_ChatOpenAI,
                        BaseResponse, get_prompt_template, get_temp_dir, run_in_thread_pool)
from server.knowledge_base.kb_cache.faiss_cache import memo_faiss_pool
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter
from server.knowledge_base.utils import KnowledgeFile
import json
import os
from pathlib import Path

# 定义一个函数，用于在多线程中解析上传的文件并保存到指定目录
def _parse_files_in_thread(
    files: List[UploadFile],  # 上传的文件列表
    dir: str,  # 保存文件的目录
    zh_title_enhance: bool,  # 是否增强中文标题
    chunk_size: int,  # 分块大小
    chunk_overlap: int,  # 分块重叠大小
):
    """
    通过多线程将上传的文件保存到对应目录内。
    生成器返回保存结果：[success or error, filename, msg, docs]
    """
    # 定义一个函数，用于解析上传的文件并保存
    def parse_file(file: UploadFile) -> dict:
        '''
        保存单个文件。
        '''
        try:
            # 获取上传文件的文件名
            filename = file.filename
            # 拼接文件路径
            file_path = os.path.join(dir, filename)
            # 读取上传文件的内容
            file_content = file.file.read()

            # 如果文件路径不存在，则创建对应的目录
            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            # 将文件内容写入文件
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # 创建一个 KnowledgeFile 对象
            kb_file = KnowledgeFile(filename=filename, knowledge_base_name="temp")
            kb_file.filepath = file_path
            # 将文件转换为文本
            docs = kb_file.file2text(zh_title_enhance=zh_title_enhance,
                                     chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap)
            # 返回成功上传的信息和文档内容
            return True, filename, f"成功上传文件 {filename}", docs
        except Exception as e:
            # 返回上传失败的信息和错误信息
            msg = f"{filename} 文件上传失败，报错信息为: {e}"
            return False, filename, msg, []

    # 构建参数列表，每个参数为一个文件
    params = [{"file": file} for file in files]
    # 使用线程池并发执行 parse_file 函数
    for result in run_in_thread_pool(parse_file, params=params):
        # 生成每个文件处理的结果
        yield result
# 上传临时文档并进行向量化处理
def upload_temp_docs(
    files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
    prev_id: str = Form(None, description="前知识库ID"),
    chunk_size: int = Form(CHUNK_SIZE, description="知识库中单段文本最大长度"),
    chunk_overlap: int = Form(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
    zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
) -> BaseResponse:
    '''
    将文件保存到临时目录，并进行向量化。
    返回临时目录名称作为ID，同时也是临时向量库的ID。
    '''
    # 如果存在前知识库ID，则从内存中移除
    if prev_id is not None:
        memo_faiss_pool.pop(prev_id)

    # 初始化失败文件列表和文档列表
    failed_files = []
    documents = []
    # 获取临时目录路径和ID
    path, id = get_temp_dir(prev_id)
    # 在线程中解析文件并处理
    for success, file, msg, docs in _parse_files_in_thread(files=files,
                                                        dir=path,
                                                        zh_title_enhance=zh_title_enhance,
                                                        chunk_size=chunk_size,
                                                        chunk_overlap=chunk_overlap):
        if success:
            documents += docs
        else:
            failed_files.append({file: msg})

    # 使用临时向量库ID加载向量存储，并添加文档
    with memo_faiss_pool.load_vector_store(id).acquire() as vs:
        vs.add_documents(documents)
    # 返回包含ID和失败文件列表的响应
    return BaseResponse(data={"id": id, "failed_files": failed_files})
# 定义一个异步函数，用于处理文件聊天请求
async def file_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                    # 用户输入的查询内容
                    knowledge_id: str = Body(..., description="临时知识库ID"),
                    # 临时知识库的ID
                    top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                    # 匹配向量的数量
                    score_threshold: float = Body(SCORE_THRESHOLD, description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右", ge=0, le=2),
                    # 知识库匹配相关度阈值
                    history: List[History] = Body([],
                                                description="历史对话",
                                                examples=[[
                                                    {"role": "user",
                                                    "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                    {"role": "assistant",
                                                    "content": "虎头虎脑"}]]
                                                ),
                    # 历史对话记录
                    stream: bool = Body(False, description="流式输出"),
                    # 是否流式输出
                    model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                    # LLM 模型名称
                    temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                    # LLM 采样温度
                    max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
                    # 限制LLM生成Token数量
                    prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                ):
    # 如果知识库ID不在内存中
    if knowledge_id not in memo_faiss_pool.keys():
        # 返回未找到知识库的响应
        return BaseResponse(code=404, msg=f"未找到临时知识库 {knowledge_id}，请先上传文件")

    # 将历史对话数据转换为History对象
    history = [History.from_data(h) for h in history]

    # 返回事件源响应，使用知识库聊天迭代器
    return EventSourceResponse(knowledge_base_chat_iterator())
```
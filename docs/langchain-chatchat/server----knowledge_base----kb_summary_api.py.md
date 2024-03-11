# `.\Langchain-Chatchat\server\knowledge_base\kb_summary_api.py`

```
# 导入必要的模块和函数
from fastapi import Body
from configs import (DEFAULT_VS_TYPE, EMBEDDING_MODEL,
                     OVERLAP_SIZE,
                     logger, log_verbose, )
from server.knowledge_base.utils import (list_files_from_folder)
from sse_starlette import EventSourceResponse
import json
from server.knowledge_base.kb_service.base import KBServiceFactory
from typing import List, Optional
from server.knowledge_base.kb_summary.base import KBSummaryService
from server.knowledge_base.kb_summary.summary_chunk import SummaryAdapter
from server.utils import wrap_done, get_ChatOpenAI, BaseResponse
from configs import LLM_MODELS, TEMPERATURE
from server.knowledge_base.model.kb_document_model import DocumentWithVSId

# 定义一个函数，用于重建单个知识库文件摘要
def recreate_summary_vector_store(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        allow_empty_kb: bool = Body(True),
        vs_type: str = Body(DEFAULT_VS_TYPE),
        embed_model: str = Body(EMBEDDING_MODEL),
        file_description: str = Body(''),
        model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
        temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
        max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
):
    """
    重建单个知识库文件摘要
    :param max_tokens: 限制LLM生成Token数量，默认None代表模型最大值
    :param model_name: LLM 模型名称
    :param temperature: LLM 采样温度
    :param file_description: 文件描述
    :param knowledge_base_name: 知识库名称
    :param allow_empty_kb: 是否允许空知识库
    :param vs_type: 默认向量存储类型
    :param embed_model: 嵌入模型
    :return: EventSourceResponse对象
    """

    # 返回一个EventSourceResponse对象，调用output()方法
    return EventSourceResponse(output())
def summary_file_to_vector_store(
        knowledge_base_name: str = Body(..., examples=["samples"]),  # 定义知识库名称参数，必填
        file_name: str = Body(..., examples=["test.pdf"]),  # 定义文件名参数，必填
        allow_empty_kb: bool = Body(True),  # 定义是否允许空知识库参数，默认为True
        vs_type: str = Body(DEFAULT_VS_TYPE),  # 定义向量存储类型参数，默认为DEFAULT_VS_TYPE
        embed_model: str = Body(EMBEDDING_MODEL),  # 定义嵌入模型参数，默认为EMBEDDING_MODEL
        file_description: str = Body(''),  # 定义文件描述参数，默认为空字符串
        model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),  # 定义模型名称参数，默认为LLM_MODELS列表的第一个值，描述为LLM 模型名称
        temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),  # 定义采样温度参数，默认为TEMPERATURE，描述为LLM 采样温度，范围在0到1之间
        max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),  # 定义生成Token数量限制参数，默认为None，描述为限制LLM生成Token数量，默认None代表模型最大值
):
    """
    单个知识库根据文件名称摘要
    :param model_name: 模型名称参数
    :param max_tokens: 生成Token数量限制参数
    :param temperature: 采样温度参数
    :param file_description: 文件描述参数
    :param file_name: 文件名参数
    :param knowledge_base_name: 知识库名称参数
    :param allow_empty_kb: 是否允许空知识库参数
    :param vs_type: 向量存储类型参数
    :param embed_model: 嵌入模型参数
    :return: 无返回值
    """
    # 定义输出函数
    def output():
        # 获取知识库服务
        kb = KBServiceFactory.get_service(knowledge_base_name, vs_type, embed_model)
        # 如果知识库不存在且不允许空知识库，则返回错误信息
        if not kb.exists() and not allow_empty_kb:
            yield {"code": 404, "msg": f"未找到知识库 ‘{knowledge_base_name}’"}
        else:
            # 创建知识库摘要服务
            kb_summary = KBSummaryService(knowledge_base_name, embed_model)
            kb_summary.create_kb_summary()

            # 获取 ChatOpenAI 对象
            llm = get_ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            reduce_llm = get_ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # 创建文本摘要适配器
            summary = SummaryAdapter.form_summary(llm=llm,
                                                  reduce_llm=reduce_llm,
                                                  overlap_size=OVERLAP_SIZE)

            # 获取知识库中的文档信息
            doc_infos = kb.list_docs(file_name=file_name)
            # 对文档进行摘要
            docs = summary.summarize(file_description=file_description,
                                     docs=doc_infos)

            # 将摘要添加到知识库摘要中
            status_kb_summary = kb_summary.add_kb_summary(summary_combine_docs=docs)
            if status_kb_summary:
                logger.info(f" {file_name} 总结完成")
                # 返回成功信息
                yield json.dumps({
                    "code": 200,
                    "msg": f"{file_name} 总结完成",
                    "doc": file_name,
                }, ensure_ascii=False)
            else:
                # 返回错误信息
                msg = f"知识库'{knowledge_base_name}'总结文件‘{file_name}’时出错。已跳过。"
                logger.error(msg)
                yield json.dumps({
                    "code": 500,
                    "msg": msg,
                })

    # 返回事件源响应
    return EventSourceResponse(output())
# 将文档ID摘要存储到向量存储中
def summary_doc_ids_to_vector_store(
        # 知识库名称，默认为"samples"
        knowledge_base_name: str = Body(..., examples=["samples"]),
        # 文档ID列表，默认为空列表
        doc_ids: List = Body([], examples=[["uuid"]]),
        # 向量存储类型，默认为默认值
        vs_type: str = Body(DEFAULT_VS_TYPE),
        # 嵌入模型，默认为嵌入模型
        embed_model: str = Body(EMBEDDING_MODEL),
        # 文件描述，默认为空字符串
        file_description: str = Body(''),
        # LLM 模型名称，默认为第一个LLM模型
        model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
        # LLM 采样温度，默认为TEMPERATURE
        temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
        # 限制LLM生成Token数量，默认为None代表模型最大值
        max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
) -> BaseResponse:
    """
    单个知识库根据doc_ids摘要
    :param knowledge_base_name: 知识库名称
    :param doc_ids: 文档ID列表
    :param model_name: LLM 模型名称
    :param max_tokens: 限制LLM生成Token数量
    :param temperature: LLM 采样温度
    :param file_description: 文件描述
    :param vs_type: 向量存储类型
    :param embed_model: 嵌入模型
    :return: BaseResponse对象
    """
    # 获取知识库服务
    kb = KBServiceFactory.get_service(knowledge_base_name, vs_type, embed_model)
    # 如果知识库不存在，则返回404状态码和相应消息
    if not kb.exists():
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}", data={})
    else:
        # 获取 ChatOpenAI 模型，用于生成对话
        llm = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # 获取 ChatOpenAI 模型，用于生成对话的简化版本
        reduce_llm = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # 创建文本摘要适配器
        summary = SummaryAdapter.form_summary(llm=llm,
                                              reduce_llm=reduce_llm,
                                              overlap_size=OVERLAP_SIZE)

        # 通过知识库获取指定文档的信息
        doc_infos = kb.get_doc_by_ids(ids=doc_ids)
        # 将文档信息转换为包含文档ID的 DocumentWithVSId 对象
        doc_info_with_ids = [DocumentWithVSId(**doc.dict(), id=with_id) for with_id, doc in zip(doc_ids, doc_infos)]

        # 使用文本摘要适配器对文件描述和文档信息进行总结
        docs = summary.summarize(file_description=file_description,
                                 docs=doc_info_with_ids)

        # 将总结后的文档信息转换为字典形式
        resp_summarize = [{**doc.dict()} for doc in docs]

        # 返回包含总结结果的 BaseResponse 对象
        return BaseResponse(code=200, msg="总结完成", data={"summarize": resp_summarize})
```
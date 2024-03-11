# `.\Langchain-Chatchat\server\api.py`

```py
# 导入必要的库
import nltk
import sys
import os

# 将上级目录添加到系统路径中
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 导入配置文件中的版本号和路径信息
from configs import VERSION
from configs.model_config import NLTK_DATA_PATH
from configs.server_config import OPEN_CROSS_DOMAIN
import argparse
import uvicorn
from fastapi import Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from server.chat.chat import chat
from server.chat.search_engine_chat import search_engine_chat
from server.chat.completion import completion
from server.chat.feedback import chat_feedback
from server.embeddings_api import embed_texts_endpoint
from server.llm_api import (list_running_models, list_config_models,
                            change_llm_model, stop_llm_model,
                            get_model_config, list_search_engines)
from server.utils import (BaseResponse, ListResponse, FastAPI, MakeFastAPIOffline,
                          get_server_configs, get_prompt_template)
from typing import List, Literal

# 设置nltk数据路径
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

# 定义一个异步函数，返回重定向到/docs的响应
async def document():
    return RedirectResponse(url="/docs")

# 创建应用程序
def create_app(run_mode: str = None):
    # 创建 FastAPI 应用程序
    app = FastAPI(
        title="Langchain-Chatchat API Server",
        version=VERSION
    )
    # 将应用程序设置为离线模式
    MakeFastAPIOffline(app)
    # 添加 CORS 中间件以允许所有来源
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    # 挂载应用程序路由
    mount_app_routes(app, run_mode=run_mode)
    return app

# 挂载应用程序路由
def mount_app_routes(app: FastAPI, run_mode: str = None):
    # 设置根路径的响应为文档重定向
    app.get("/",
            response_model=BaseResponse,
            summary="swagger 文档")(document)

    # Tag: Chat
    # 注册与llm模型对话的接口
    app.post("/chat/chat",
             tags=["Chat"],
             summary="与llm模型对话(通过LLMChain)",
             )(chat)

    # 注册与搜索引擎对话的接口
    app.post("/chat/search_engine_chat",
             tags=["Chat"],
             summary="与搜索引擎对话",
             )(search_engine_chat)

    # 注册返回llm模型对话评分的接口
    app.post("/chat/feedback",
             tags=["Chat"],
             summary="返回llm模型对话评分",
             )(chat_feedback)

    # 注册知识库相关接口
    mount_knowledge_routes(app)
    
    # 注册摘要相关接口
    mount_filename_summary_routes(app)

    # 注册列出当前已加载的模型的接口
    app.post("/llm_model/list_running_models",
             tags=["LLM Model Management"],
             summary="列出当前已加载的模型",
             )(list_running_models)

    # 注册列出configs已配置的模型的接口
    app.post("/llm_model/list_config_models",
             tags=["LLM Model Management"],
             summary="列出configs已配置的模型",
             )(list_config_models)

    # 注册获取模型配置（合并后）的接口
    app.post("/llm_model/get_model_config",
             tags=["LLM Model Management"],
             summary="获取模型配置（合并后）",
             )(get_model_config)

    # 注册停止指定的LLM模型（Model Worker)的接口
    app.post("/llm_model/stop",
             tags=["LLM Model Management"],
             summary="停止指定的LLM模型（Model Worker)",
             )(stop_llm_model)

    # 注册切换指定的LLM模型（Model Worker)的接口
    app.post("/llm_model/change",
             tags=["LLM Model Management"],
             summary="切换指定的LLM模型（Model Worker)",
             )(change_llm_model)

    # 注册获取服务器原始配置信息的接口
    app.post("/server/configs",
             tags=["Server State"],
             summary="获取服务器原始配置信息",
             )(get_server_configs)

    # 注册获取服务器支持的搜索引擎的接口
    app.post("/server/list_search_engines",
             tags=["Server State"],
             summary="获取服务器支持的搜索引擎",
             )(list_search_engines)

    # 注册获取服务区配置的 prompt 模板的接口
    @app.post("/server/get_prompt_template",
             tags=["Server State"],
             summary="获取服务区配置的 prompt 模板")
    # 定义一个函数，用于获取服务器提示模板
    def get_server_prompt_template(
        type: Literal["llm_chat", "knowledge_base_chat", "search_engine_chat", "agent_chat"]=Body("llm_chat", description="模板类型，可选值：llm_chat，knowledge_base_chat，search_engine_chat，agent_chat"),
        name: str = Body("default", description="模板名称"),
    ) -> str:
        # 调用函数获取提示模板，传入模板类型和名称作为参数
        return get_prompt_template(type=type, name=name)

    # 定义一个接口，用于处理补全请求
    app.post("/other/completion",
             tags=["Other"],
             summary="要求llm模型补全(通过LLMChain)",
             )(completion)

    # 定义一个接口，用于将文本向量化，支持本地模型和在线模型
    app.post("/other/embed_texts",
            tags=["Other"],
            summary="将文本向量化，支持本地模型和在线模型",
            )(embed_texts_endpoint)
# 挂载知识库相关路由到 FastAPI 应用
def mount_knowledge_routes(app: FastAPI):
    # 导入知识库对话相关模块
    from server.chat.knowledge_base_chat import knowledge_base_chat
    # 导入文件对话相关模块
    from server.chat.file_chat import upload_temp_docs, file_chat
    # 导入 agent 对话相关模块
    from server.chat.agent_chat import agent_chat
    # 导入知识库 API 相关模块
    from server.knowledge_base.kb_api import list_kbs, create_kb, delete_kb
    # 导入知识库文档 API 相关模块
    from server.knowledge_base.kb_doc_api import (list_files, upload_docs, delete_docs,
                                                update_docs, download_doc, recreate_vector_store,
                                                search_docs, DocumentWithVSId, update_info,
                                                update_docs_by_id,)

    # 创建 POST 路由 "/chat/knowledge_base_chat"，用于知识库对话
    app.post("/chat/knowledge_base_chat",
             tags=["Chat"],
             summary="与知识库对话")(knowledge_base_chat)

    # 创建 POST 路由 "/chat/file_chat"，用于文件对话
    app.post("/chat/file_chat",
             tags=["Knowledge Base Management"],
             summary="文件对话"
             )(file_chat)

    # 创建 POST 路由 "/chat/agent_chat"，用于与 agent 对话
    app.post("/chat/agent_chat",
             tags=["Chat"],
             summary="与agent对话")(agent_chat)

    # 创建 GET 路由 "/knowledge_base/list_knowledge_bases"，用于获取知识库列表
    app.get("/knowledge_base/list_knowledge_bases",
            tags=["Knowledge Base Management"],
            response_model=ListResponse,
            summary="获取知识库列表")(list_kbs)

    # 创建 POST 路由 "/knowledge_base/create_knowledge_base"，用于创建知识库
    app.post("/knowledge_base/create_knowledge_base",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="创建知识库"
             )(create_kb)

    # 创建 POST 路由 "/knowledge_base/delete_knowledge_base"，用于删除知识库
    app.post("/knowledge_base/delete_knowledge_base",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="删除知识库"
             )(delete_kb)

    # 创建 GET 路由 "/knowledge_base/list_files"，用于获取知识库内的文件列表
    app.get("/knowledge_base/list_files",
            tags=["Knowledge Base Management"],
            response_model=ListResponse,
            summary="获取知识库内的文件列表"
            )(list_files)
    # 定义 POST 请求路由，用于搜索知识库文档
    app.post("/knowledge_base/search_docs",
             tags=["Knowledge Base Management"],
             response_model=List[DocumentWithVSId],
             summary="搜索知识库"
             )(search_docs)

    # 定义 POST 请求路由，用于直接更新知识库文档
    app.post("/knowledge_base/update_docs_by_id",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="直接更新知识库文档"
             )(update_docs_by_id)

    # 定义 POST 请求路由，用于上传文件到知识库，并/或进行向量化
    app.post("/knowledge_base/upload_docs",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="上传文件到知识库，并/或进行向量化"
             )(upload_docs)

    # 定义 POST 请求路由，用于删除知识库内指定文件
    app.post("/knowledge_base/delete_docs",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="删除知识库内指定文件"
             )(delete_docs)

    # 定义 POST 请求路由，用于更新知识库介绍
    app.post("/knowledge_base/update_info",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="更新知识库介绍"
             )(update_info)

    # 定义 POST 请求路由，用于更新现有文件到知识库
    app.post("/knowledge_base/update_docs",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="更新现有文件到知识库"
             )(update_docs)

    # 定义 GET 请求路由，用于下载对应的知识文件
    app.get("/knowledge_base/download_doc",
            tags=["Knowledge Base Management"],
            summary="下载对应的知识文件")(download_doc)

    # 定义 POST 请求路由，用于根据content中文档重建向量库，流式输出处理进度
    app.post("/knowledge_base/recreate_vector_store",
             tags=["Knowledge Base Management"],
             summary="根据content中文档重建向量库，流式输出处理进度。"
             )(recreate_vector_store)

    # 定义 POST 请求路由，用于上传文件到临时目录，用于文件对话
    app.post("/knowledge_base/upload_temp_docs",
             tags=["Knowledge Base Management"],
             summary="上传文件到临时目录，用于文件对话。"
             )(upload_temp_docs)
# 定义函数 mount_filename_summary_routes，用于挂载文件名摘要路由到 FastAPI 应用
def mount_filename_summary_routes(app: FastAPI):
    # 导入相关模块和函数
    from server.knowledge_base.kb_summary_api import (summary_file_to_vector_store, recreate_summary_vector_store,
                                                      summary_doc_ids_to_vector_store)

    # 挂载路由 "/knowledge_base/kb_summary_api/summary_file_to_vector_store"，指定标签和摘要信息
    app.post("/knowledge_base/kb_summary_api/summary_file_to_vector_store",
             tags=["Knowledge kb_summary_api Management"],
             summary="单个知识库根据文件名称摘要"
             )(summary_file_to_vector_store)
    
    # 挂载路由 "/knowledge_base/kb_summary_api/summary_doc_ids_to_vector_store"，指定标签、摘要信息和响应模型
    app.post("/knowledge_base/kb_summary_api/summary_doc_ids_to_vector_store",
             tags=["Knowledge kb_summary_api Management"],
             summary="单个知识库根据doc_ids摘要",
             response_model=BaseResponse,
             )(summary_doc_ids_to_vector_store)
    
    # 挂载路由 "/knowledge_base/kb_summary_api/recreate_summary_vector_store"，指定标签和摘要信息
    app.post("/knowledge_base/kb_summary_api/recreate_summary_vector_store",
             tags=["Knowledge kb_summary_api Management"],
             summary="重建单个知识库文件摘要"
             )(recreate_summary_vector_store)

# 定义函数 run_api，用于运行 FastAPI 应用
def run_api(host, port, **kwargs):
    # 如果存在 SSL 密钥文件和证书文件，则使用 SSL 运行应用
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(app,
                    host=host,
                    port=port,
                    ssl_keyfile=kwargs.get("ssl_keyfile"),
                    ssl_certfile=kwargs.get("ssl_certfile"),
                    )
    # 否则，使用普通方式运行应用
    else:
        uvicorn.run(app, host=host, port=port)

# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(prog='langchain-ChatGLM',
                                     description='About langchain-ChatGLM, local knowledge based ChatGLM with langchain'
                                                 ' ｜ 基于本地知识库的 ChatGLM 问答')
    # 添加参数选项
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    # 解析参数
    args = parser.parse_args()
    # 将参数转换为字典形式
    args_dict = vars(args)

    # 创建 FastAPI 应用
    app = create_app()
    # 运行 API 服务，传入主机名、端口号、SSL 密钥文件和SSL 证书文件作为参数
    run_api(host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            )
```
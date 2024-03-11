# `.\Langchain-Chatchat\server\llm_api.py`

```
# 导入所需模块和函数
from fastapi import Body
from configs import logger, log_verbose, LLM_MODELS, HTTPX_DEFAULT_TIMEOUT
from server.utils import (BaseResponse, fschat_controller_address, list_config_llm_models,
                          get_httpx_client, get_model_worker_config)
from typing import List

# 列出正在运行的模型及其配置项
def list_running_models(
    controller_address: str = Body(None, description="Fastchat controller服务器地址", examples=[fschat_controller_address()]),
    placeholder: str = Body(None, description="该参数未使用，占位用"),
) -> BaseResponse:
    '''
    从fastchat controller获取已加载模型列表及其配置项
    '''
    try:
        # 如果未提供controller_address，则使用默认地址
        controller_address = controller_address or fschat_controller_address()
        # 使用HTTPX客户端发送POST请求到controller_address的/list_models端点
        with get_httpx_client() as client:
            r = client.post(controller_address + "/list_models")
            # 从响应中获取模型列表
            models = r.json()["models"]
            # 获取每个模型的配置项数据
            data = {m: get_model_config(m).data for m in models}
            return BaseResponse(data=data)
    except Exception as e:
        # 记录错误日志并返回错误响应
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        return BaseResponse(
            code=500,
            data={},
            msg=f"failed to get available models from controller: {controller_address}。错误信息是： {e}")

# 列出配置的模型列表
def list_config_models(
    types: List[str] = Body(["local", "online"], description="模型配置项类别，如local, online, worker"),
    placeholder: str = Body(None, description="占位用，无实际效果")
) -> BaseResponse:
    '''
    从本地获取configs中配置的模型列表
    '''
    data = {}
    # 遍历不同类型的模型配置项，获取每个模型的配置数据
    for type, models in list_config_llm_models().items():
        if type in types:
            data[type] = {m: get_model_config(m).data for m in models}
    return BaseResponse(data=data)

# 获取模型配置项
def get_model_config(
    model_name: str = Body(description="配置中LLM模型的名称"),
    placeholder: str = Body(None, description="占位用，无实际效果")
) -> BaseResponse:
    '''
    获取LLM模型配置项（合并后的）
    '''
    config = {}
    # 删除ONLINE_MODEL配置中的敏感信息
    # 遍历获取指定模型的工作配置项，返回键值对
    for k, v in get_model_worker_config(model_name=model_name).items():
        # 检查键名是否不是"worker_class"，或者包含"key"，或者包含"secret"，或者以"id"结尾
        if not (k == "worker_class"
            or "key" in k.lower()
            or "secret" in k.lower()
            or k.lower().endswith("id")):
            # 将符合条件的配置项添加到config字典中
            config[k] = v

    # 返回包含筛选后配置项的响应对象
    return BaseResponse(data=config)
# 停止LLM模型的函数，接受要停止的模型名称和Fastchat controller服务器地址作为参数，返回BaseResponse对象
def stop_llm_model(
    model_name: str = Body(..., description="要停止的LLM模型名称", examples=[LLM_MODELS[0]]),
    controller_address: str = Body(None, description="Fastchat controller服务器地址", examples=[fschat_controller_address()])
) -> BaseResponse:
    '''
    向fastchat controller请求停止某个LLM模型。
    注意：由于Fastchat的实现方式，实际上是把LLM模型所在的model_worker停掉。
    '''
    # 尝试执行以下代码块，捕获可能发生的异常
    try:
        # 如果controller_address为None，则使用默认的fschat_controller_address()
        controller_address = controller_address or fschat_controller_address()
        # 使用get_httpx_client()获取HTTPX客户端
        with get_httpx_client() as client:
            # 向controller_address发送POST请求，请求停止指定模型的worker
            r = client.post(
                controller_address + "/release_worker",
                json={"model_name": model_name},
            )
            # 返回响应的JSON数据
            return r.json()
    # 捕获所有异常，并记录错误日志
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        # 返回BaseResponse对象，表示出现错误
        return BaseResponse(
            code=500,
            msg=f"failed to stop LLM model {model_name} from controller: {controller_address}。错误信息是： {e}")


# 切换LLM模型的函数，接受当前运行模型、要切换的新模型和Fastchat controller服务器地址作为参数
def change_llm_model(
    model_name: str = Body(..., description="当前运行模型", examples=[LLM_MODELS[0]]),
    new_model_name: str = Body(..., description="要切换的新模型", examples=[LLM_MODELS[0]]),
    controller_address: str = Body(None, description="Fastchat controller服务器地址", examples=[fschat_controller_address()])
):
    '''
    向fastchat controller请求切换LLM模型。
    '''
    # 尝试执行以下代码块
    try:
        # 如果controller_address为None，则使用默认的fschat_controller_address()
        controller_address = controller_address or fschat_controller_address()
        # 使用get_httpx_client()获取HTTPX客户端
        with get_httpx_client() as client:
            # 向controller_address发送POST请求，请求切换模型
            r = client.post(
                controller_address + "/release_worker",
                json={"model_name": model_name, "new_model_name": new_model_name},
                timeout=HTTPX_DEFAULT_TIMEOUT, # 等待新worker_model
            )
            # 返回响应的JSON数据
            return r.json()
    # 捕获所有异常
    except Exception as e:
        # 记录错误日志
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        # 返回BaseResponse对象，表示出现错误
        return BaseResponse(
            code=500,
            msg=f"failed to stop LLM model {model_name} from controller: {controller_address}。错误信息是： {e}")
    # 捕获所有异常，并记录错误信息
    except Exception as e:
        # 使用 logger 记录异常信息，包括异常类名和具体信息
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        # 返回一个包含错误信息的 BaseResponse 对象
        return BaseResponse(
            code=500,
            msg=f"failed to switch LLM model from controller: {controller_address}。错误信息是： {e}")
# 定义一个函数，用于列出搜索引擎列表，并返回给客户端
def list_search_engines() -> BaseResponse:
    # 从搜索引擎聊天模块中导入搜索引擎列表
    from server.chat.search_engine_chat import SEARCH_ENGINES
    # 将搜索引擎列表转换为列表形式，并作为数据返回给客户端
    return BaseResponse(data=list(SEARCH_ENGINES))
```
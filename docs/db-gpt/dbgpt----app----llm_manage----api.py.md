# `.\DB-GPT-src\dbgpt\app\llm_manage\api.py`

```py
# 导入 FastAPI 框架中的 APIRouter 类
from fastapi import APIRouter

# 导入配置文件 Config 和其他必要的模块和类
from dbgpt._private.config import Config
from dbgpt.app.llm_manage.request.request import ModelResponse
from dbgpt.app.openapi.api_view_model import Result
from dbgpt.component import ComponentType
from dbgpt.model.cluster import WorkerManagerFactory, WorkerStartupRequest

# 创建配置对象 CFG
CFG = Config()

# 创建一个 APIRouter 实例，用于定义和管理 API 路由
router = APIRouter()

# 定义一个 GET 请求处理函数，处理路径 '/v1/worker/model/params'
@router.get("/v1/worker/model/params")
async def model_params():
    print(f"/worker/model/params")  # 打印日志
    try:
        # 导入 WorkerManagerFactory 类，获取 WorkerManagerFactory 实例
        from dbgpt.model.cluster import WorkerManagerFactory

        # 从 CFG.SYSTEM_APP 中获取 WORKER_MANAGER_FACTORY 类型的组件实例
        worker_manager = CFG.SYSTEM_APP.get_component(
            ComponentType.WORKER_MANAGER_FACTORY, WorkerManagerFactory
        ).create()

        params = []  # 初始化一个空列表，用于存储模型参数信息
        workers = await worker_manager.supported_models()  # 获取支持的模型列表
        for worker in workers:
            for model in worker.models:
                model_dict = model.__dict__  # 获取模型对象的属性字典
                model_dict["host"] = worker.host  # 将模型所属的 worker 的 host 添加到模型字典中
                model_dict["port"] = worker.port  # 将模型所属的 worker 的 port 添加到模型字典中
                params.append(model_dict)  # 将模型字典添加到 params 列表中

        return Result.succ(params)  # 返回成功的结果对象，包含模型参数列表

        # 如果 worker_instance 不存在，则返回查找 worker manager 失败的结果
        if not worker_instance:
            return Result.failed(code="E000X", msg=f"can not find worker manager")
    except Exception as e:
        return Result.failed(code="E000X", msg=f"model stop failed {e}")  # 捕获异常并返回失败的结果对象


# 定义一个 GET 请求处理函数，处理路径 '/v1/worker/model/list'
@router.get("/v1/worker/model/list")
async def model_list():
    print(f"/worker/model/list")  # 打印日志
    try:
        # 导入 BaseModelController 类，用于管理基础模型控制器
        from dbgpt.model.cluster.controller.controller import BaseModelController

        # 从 CFG.SYSTEM_APP 中获取 MODEL_CONTROLLER 类型的组件实例
        controller = CFG.SYSTEM_APP.get_component(
            ComponentType.MODEL_CONTROLLER, BaseModelController
        )

        responses = []  # 初始化一个空列表，用于存储模型响应对象

        # 获取所有健康实例的 managers 列表
        managers = await controller.get_all_instances(
            model_name="WorkerManager@service", healthy_only=True
        )

        # 使用 map 函数创建 manager_map 字典，将 manager 的 host 作为键，manager 实例作为值
        manager_map = dict(map(lambda manager: (manager.host, manager), managers))

        # 获取所有模型实例的列表
        models = await controller.get_all_instances()

        for model in models:
            worker_name, worker_type = model.model_name.split("@")  # 分离 worker 名称和类型
            if worker_type == "llm" or worker_type == "text2vec":  # 如果 worker 类型为 'llm' 或 'text2vec'
                # 创建 ModelResponse 对象，封装模型信息
                response = ModelResponse(
                    model_name=worker_name,
                    model_type=worker_type,
                    host=model.host,
                    port=model.port,
                    healthy=model.healthy,
                    check_healthy=model.check_healthy,
                    last_heartbeat=model.last_heartbeat,
                    prompt_template=model.prompt_template,
                )

                # 设置 response 的 manager_host 属性，若 manager_map 中存在该 host 则为 host，否则为 None
                response.manager_host = (
                    model.host if manager_map.get(model.host) else None
                )

                # 设置 response 的 manager_port 属性，若 manager_map 中存在该 host 则为 port，否则为 None
                response.manager_port = (
                    manager_map[model.host].port
                    if manager_map.get(model.host)
                    else None
                )

                responses.append(response)  # 将 response 添加到 responses 列表中

        return Result.succ(responses)  # 返回成功的结果对象，包含模型响应列表

    except Exception as e:
        return Result.failed(code="E000X", msg=f"model list error {e}")  # 捕获异常并返回失败的结果对象


# 定义一个 POST 请求处理函数，处理路径 '/v1/worker/model/stop'
@router.post("/v1/worker/model/stop")
# 执行模型停止的异步函数，接收一个 WorkerStartupRequest 对象作为参数
async def model_stop(request: WorkerStartupRequest):
    # 输出调试信息，指示正在执行模型停止操作的路径
    print(f"/v1/worker/model/stop:")
    try:
        # 尝试导入 BaseModelController 类来处理模型停止操作
        from dbgpt.model.cluster.controller.controller import BaseModelController
        
        # 从系统配置中获取工作管理器工厂实例，并创建一个工作管理器对象
        worker_manager = CFG.SYSTEM_APP.get_component(
            ComponentType.WORKER_MANAGER_FACTORY, WorkerManagerFactory
        ).create()
        
        # 如果未能获取到有效的工作管理器对象，则返回一个失败的结果
        if not worker_manager:
            return Result.failed(code="E000X", msg=f"can not find worker manager")
        
        # 重置请求的参数为空字典
        request.params = {}
        
        # 调用工作管理器的模型关闭方法，并返回成功的结果
        return Result.succ(await worker_manager.model_shutdown(request))
    
    # 捕获任何异常，并返回一个失败的结果，包含错误代码和异常信息
    except Exception as e:
        return Result.failed(code="E000X", msg=f"model stop failed {e}")


# 处理模型启动 POST 请求的异步函数，接收一个 WorkerStartupRequest 对象作为参数
@router.post("/v1/worker/model/start")
async def model_start(request: WorkerStartupRequest):
    # 输出调试信息，指示正在执行模型启动操作的路径
    print(f"/v1/worker/model/start:")
    try:
        # 从系统配置中获取工作管理器工厂实例，并创建一个工作管理器对象
        worker_manager = CFG.SYSTEM_APP.get_component(
            ComponentType.WORKER_MANAGER_FACTORY, WorkerManagerFactory
        ).create()
        
        # 如果未能获取到有效的工作管理器对象，则返回一个失败的结果
        if not worker_manager:
            return Result.failed(code="E000X", msg=f"can not find worker manager")
        
        # 调用工作管理器的模型启动方法，并返回成功的结果
        return Result.succ(await worker_manager.model_startup(request))
    
    # 捕获任何异常，并返回一个失败的结果，包含错误代码和异常信息
    except Exception as e:
        return Result.failed(code="E000X", msg=f"model start failed {e}")
```
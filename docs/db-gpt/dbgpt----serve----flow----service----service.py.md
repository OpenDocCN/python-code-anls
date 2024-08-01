# `.\DB-GPT-src\dbgpt\serve\flow\service\service.py`

```py
import json
import logging
from typing import AsyncIterator, List, Optional, cast

import schedule
from fastapi import HTTPException

from dbgpt._private.pydantic import model_to_json
from dbgpt.component import SystemApp
from dbgpt.core.awel import DAG, BaseOperator, CommonLLMHttpRequestBody
from dbgpt.core.awel.dag.dag_manager import DAGManager
from dbgpt.core.awel.flow.flow_factory import (
    FlowCategory,
    FlowFactory,
    State,
    fill_flow_panel,
)
from dbgpt.core.awel.trigger.http_trigger import CommonLLMHttpTrigger
from dbgpt.core.awel.util.chat_util import (
    is_chat_flow_type,
    safe_chat_stream_with_dag_task,
    safe_chat_with_dag_task,
)
from dbgpt.core.interface.llm import ModelOutput
from dbgpt.core.schema.api import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
)
from dbgpt.serve.core import BaseService
from dbgpt.storage.metadata import BaseDao
from dbgpt.storage.metadata._base_dao import QUERY_SPEC
from dbgpt.util.dbgpts.loader import DBGPTsLoader
from dbgpt.util.pagination_utils import PaginationResult

from ..api.schemas import ServeRequest, ServerResponse
from ..config import SERVE_CONFIG_KEY_PREFIX, SERVE_SERVICE_COMPONENT_NAME, ServeConfig
from ..models.models import ServeDao, ServeEntity

logger = logging.getLogger(__name__)


class Service(BaseService[ServeEntity, ServeRequest, ServerResponse]):
    """The service class for Flow"""

    name = SERVE_SERVICE_COMPONENT_NAME

    def __init__(self, system_app: SystemApp, dao: Optional[ServeDao] = None):
        # 初始化服务实例
        self._system_app = None  # 系统应用实例
        self._serve_config: ServeConfig = None  # 服务配置对象
        self._dao: ServeDao = dao  # 数据访问对象
        self._flow_factory: FlowFactory = FlowFactory()  # 流工厂对象
        self._dbgpts_loader: Optional[DBGPTsLoader] = None  # 调试点加载器对象

        super().__init__(system_app)

    def init_app(self, system_app: SystemApp) -> None:
        """Initialize the service

        Args:
            system_app (SystemApp): The system app
        """
        super().init_app(system_app)

        self._serve_config = ServeConfig.from_app_config(
            system_app.config, SERVE_CONFIG_KEY_PREFIX
        )  # 从应用配置中加载服务配置
        self._dao = self._dao or ServeDao(self._serve_config)  # 初始化数据访问对象
        self._system_app = system_app  # 设置系统应用实例
        self._dbgpts_loader = system_app.get_component(
            DBGPTsLoader.name,
            DBGPTsLoader,
            or_register_component=DBGPTsLoader,
            load_dbgpts_interval=self._serve_config.load_dbgpts_interval,
        )  # 获取或注册调试点加载器

    def before_start(self):
        """Execute before the application starts"""
        super().before_start()
        self._pre_load_dag_from_db()  # 从数据库预加载 DAG
        self._pre_load_dag_from_dbgpts()  # 从调试点预加载 DAG

    def after_start(self):
        """Execute after the application starts"""
        self.load_dag_from_db()  # 从数据库加载 DAG
        self.load_dag_from_dbgpts(is_first_load=True)  # 首次从调试点加载 DAG
        schedule.every(self._serve_config.load_dbgpts_interval).seconds.do(
            self.load_dag_from_dbgpts
        )  # 定时从调试点加载 DAG

    @property
    # 返回内部的 DAO（数据访问对象）
    def dao(self) -> BaseDao[ServeEntity, ServeRequest, ServerResponse]:
        """Returns the internal DAO."""
        return self._dao

    # 返回内部的 DBGPTsLoader 对象
    @property
    def dbgpts_loader(self) -> DBGPTsLoader:
        """Returns the internal DBGPTsLoader."""
        if self._dbgpts_loader is None:
            raise ValueError("DBGPTsLoader is not initialized")
        return self._dbgpts_loader

    # 返回内部的 ServeConfig 对象
    @property
    def config(self) -> ServeConfig:
        """Returns the internal ServeConfig."""
        return self._serve_config

    # 创建一个新的 Flow 实体
    def create(self, request: ServeRequest) -> ServerResponse:
        """Create a new Flow entity

        Args:
            request (ServeRequest): The request

        Returns:
            ServerResponse: The response
        """

    # 创建一个新的 Flow 实体并保存 DAG
    def create_and_save_dag(
        self, request: ServeRequest, save_failed_flow: bool = False
    ) -> ServerResponse:
        """Create a new Flow entity and save the DAG

        Args:
            request (ServeRequest): The request
            save_failed_flow (bool): Whether to save the failed flow

        Returns:
            ServerResponse: The response
        """
        try:
            # 根据请求构建 DAG
            if request.define_type == "json":
                dag = self._flow_factory.build(request)
            else:
                dag = request.flow_dag
            request.dag_id = dag.dag_id
            # 将 DAG 保存到存储中
            request.flow_category = self._parse_flow_category(dag)
        except Exception as e:
            if save_failed_flow:
                # 如果需要保存失败的流程，设置请求状态和错误信息
                request.state = State.LOAD_FAILED
                request.error_message = str(e)
                request.dag_id = ""
                return self.dao.create(request)
            else:
                # 抛出异常，指示 DAG 创建失败
                raise ValueError(
                    f"Create DAG {request.name} error, define_type: {request.define_type}, error: {str(e)}"
                ) from e
        # 创建流程并返回结果
        res = self.dao.create(request)

        state = request.state
        try:
            if state == State.DEPLOYED:
                # 注册 DAG
                self.dag_manager.register_dag(dag, request.uid)
                # 更新状态为 RUNNING
                request.state = State.RUNNING
                request.error_message = ""
                self.dao.update({"uid": request.uid}, request)
            else:
                logger.info(f"Flow state is {state}, skip register DAG")
        except Exception as e:
            logger.warning(f"Register DAG({dag.dag_id}) error: {str(e)}")
            if save_failed_flow:
                # 如果需要保存失败的流程，设置请求状态和错误信息
                request.state = State.LOAD_FAILED
                request.error_message = f"Register DAG error: {str(e)}"
                request.dag_id = ""
                self.dao.update({"uid": request.uid}, request)
            else:
                # 回滚操作
                self.delete(request.uid)
            # 抛出异常
            raise e
        # 返回创建结果
        return res
    def _pre_load_dag_from_db(self):
        """从数据库预加载DAG"""
        # 从数据库获取所有实体列表
        entities = self.dao.get_list({})
        # 遍历每个实体
        for entity in entities:
            try:
                # 预加载实体所需的要求
                self._flow_factory.pre_load_requirements(entity)
            except Exception as e:
                # 记录警告日志，指出从数据库加载DAG过程中的异常
                logger.warning(
                    f"Pre load requirements for DAG({entity.name}, {entity.dag_id}) "
                    f"from db error: {str(e)}"
                )

    def load_dag_from_db(self):
        """从数据库加载DAG"""
        # 从数据库获取所有实体列表
        entities = self.dao.get_list({})
        # 遍历每个实体
        for entity in entities:
            try:
                # 如果实体定义类型不是"json"，则跳过
                if entity.define_type != "json":
                    continue
                # 使用实体构建DAG对象
                dag = self._flow_factory.build(entity)
                # 如果实体状态是DEPLOYED或RUNNING，或者版本为"0.1.0"且状态为INITIALIZING
                if entity.state in [State.DEPLOYED, State.RUNNING] or (
                    entity.version == "0.1.0" and entity.state == State.INITIALIZING
                ):
                    # 注册DAG到DAG管理器
                    self.dag_manager.register_dag(dag, entity.uid)
                    # 将实体状态更新为RUNNING
                    entity.state = State.RUNNING
                    entity.error_message = ""
                    # 更新实体信息到数据库
                    self.dao.update({"uid": entity.uid}, entity)
            except Exception as e:
                # 记录警告日志，指出从数据库加载DAG过程中的异常
                logger.warning(
                    f"Load DAG({entity.name}, {entity.dag_id}) from db error: {str(e)}"
                )

    def _pre_load_dag_from_dbgpts(self):
        """从dbgpts预加载DAG"""
        # 获取所有流程列表
        flows = self.dbgpts_loader.get_flows()
        # 遍历每个流程
        for flow in flows:
            try:
                # 如果流程定义类型是"json"，则预加载流程所需的要求
                if flow.define_type == "json":
                    self._flow_factory.pre_load_requirements(flow)
            except Exception as e:
                # 记录警告日志，指出从dbgpts加载DAG过程中的异常
                logger.warning(
                    f"Pre load requirements for DAG({flow.name}) from "
                    f"dbgpts error: {str(e)}"
                )

    def load_dag_from_dbgpts(self, is_first_load: bool = False):
        """从dbgpts加载DAG"""
        # 获取所有流程列表
        flows = self.dbgpts_loader.get_flows()
        # 遍历每个流程
        for flow in flows:
            try:
                # 如果流程定义类型是"python"且流程DAG为空，则跳过
                if flow.define_type == "python" and flow.flow_dag is None:
                    continue
                # 将流程状态设置为DEPLOYED
                flow.state = State.DEPLOYED
                # 检查流程是否已存在
                exist_inst = self.get({"name": flow.name})
                # 如果流程不存在，则创建并保存流程
                if not exist_inst:
                    self.create_and_save_dag(flow, save_failed_flow=True)
                # 如果是第一次加载或已存在流程状态不是RUNNING，则更新流程信息
                elif is_first_load or exist_inst.state != State.RUNNING:
                    # TODO 检查版本，必须大于现有版本
                    flow.uid = exist_inst.uid
                    self.update_flow(flow, check_editable=False, save_failed_flow=True)
            except Exception as e:
                import traceback

                message = traceback.format_exc()
                # 记录警告日志，指出从dbgpts加载DAG过程中的异常和详细信息
                logger.warning(
                    f"Load DAG {flow.name} from dbgpts error: {str(e)}, detail: {message}"
                )
    def update_flow(
        self,
        request: ServeRequest,
        check_editable: bool = True,
        save_failed_flow: bool = False,
    ) -> ServerResponse:
        """Update a Flow entity

        Args:
            request (ServeRequest): The request object containing information about the flow
            check_editable (bool): Flag indicating whether to check if the flow is editable
            save_failed_flow (bool): Flag indicating whether to save the flow if update fails
        Returns:
            ServerResponse: Response object indicating the result of the update operation
        """
        # Extract the new state from the request
        new_state = request.state
        try:
            # Try to build the directed acyclic graph (DAG) from the request
            if request.define_type == "json":
                # Build the DAG using a JSON definition
                dag = self._flow_factory.build(request)
            else:
                # Use the existing flow DAG from the request
                dag = request.flow_dag
            # Parse and set the flow category based on the DAG
            request.flow_category = self._parse_flow_category(dag)
        except Exception as e:
            if save_failed_flow:
                # If specified, mark the flow state as LOAD_FAILED on failure
                request.state = State.LOAD_FAILED
                request.error_message = str(e)
                request.dag_id = ""
                # Attempt to update the DAO with the failed request
                return self.dao.update({"uid": request.uid}, request)
            else:
                # If not saving failed flows, raise the exception
                raise e

        # Prepare a query request to retrieve the flow instance by its UID
        query_request = {"uid": request.uid}
        inst = self.get(query_request)
        if not inst:
            # If flow instance is not found, raise HTTP 404 exception
            raise HTTPException(status_code=404, detail=f"Flow {request.uid} not found")
        if check_editable and not inst.editable:
            # If checking for editability and flow is not editable, raise HTTP 403 exception
            raise HTTPException(
                status_code=403, detail=f"Flow {request.uid} is not editable"
            )

        # Retrieve the old state of the flow instance
        old_state = inst.state
        if not State.can_change_state(old_state, new_state):
            # Check if the state transition is valid; if not, raise HTTP 400 exception
            raise HTTPException(
                status_code=400,
                detail=f"Flow {request.uid} state can't change from {old_state} to "
                f"{new_state}",
            )

        # Initialize variable to store old data, initially None
        old_data: Optional[ServerResponse] = None
        try:
            # Attempt to update the DAO with the update request
            update_obj = self.dao.update(query_request, update_request=request)
            # Delete the old data corresponding to the flow UID
            old_data = self.delete(request.uid)
            # Restore the old state to the old data
            old_data.state = old_state
            if not old_data:
                # If old data is not found, raise HTTP 404 exception
                raise HTTPException(
                    status_code=404, detail=f"Flow detail {request.uid} not found"
                )
            # Update the flow DAG of the updated object with the request flow DAG
            update_obj.flow_dag = request.flow_dag
            # Create and save the DAG with the updated object
            return self.create_and_save_dag(update_obj)
        except Exception as e:
            if old_data and old_data.state == State.RUNNING:
                # If the old flow was running, attempt to recover it
                # Set the state of the old flow to DEPLOYED
                old_data.state = State.DEPLOYED
                # Create and save the DAG with the recovered old data
                self.create_and_save_dag(old_data)
            # Raise the original exception encountered during the update process
            raise e
    def get(self, request: QUERY_SPEC) -> Optional[ServerResponse]:
        """Get a Flow entity

        Args:
            request (ServeRequest): The request object containing query parameters

        Returns:
            ServerResponse: The response object representing the retrieved flow entity
        """
        # TODO: implement your own logic here
        # Build the query request from the request
        query_request = request
        # Retrieve the flow entity from data access object based on query request
        flow = self.dao.get_one(query_request)
        if flow:
            # Fill additional UI-specific details for the retrieved flow entity
            fill_flow_panel(flow)
            # Retrieve metadata associated with the flow's DAG (Directed Acyclic Graph)
            metadata = self.dag_manager.get_dag_metadata(
                flow.dag_id, alias_name=flow.uid
            )
            if metadata:
                # Convert metadata object to dictionary and assign it to flow metadata
                flow.metadata = metadata.to_dict()
        return flow

    def delete(self, uid: str) -> Optional[ServerResponse]:
        """Delete a Flow entity

        Args:
            uid (str): The unique identifier of the flow entity to be deleted

        Returns:
            ServerResponse: The response object representing data after deletion
        """

        # TODO: implement your own logic here
        # Build the query request from the request
        query_request = {"uid": uid}
        # Retrieve the instance of flow entity to be deleted
        inst = self.get(query_request)
        if inst is None:
            # Raise HTTP exception if flow entity with given uid is not found
            raise HTTPException(status_code=404, detail=f"Flow {uid} not found")
        if inst.state == State.RUNNING and not inst.dag_id:
            # Raise HTTP exception if running flow entity's DAG id is not found
            raise HTTPException(
                status_code=404, detail=f"Running flow {uid}'s dag id not found"
            )
        try:
            if inst.dag_id:
                # Unregister the DAG associated with the flow entity if it exists
                self.dag_manager.unregister_dag(inst.dag_id)
        except Exception as e:
            # Log a warning if there's an error while unregistering the DAG
            logger.warning(f"Unregister DAG({inst.dag_id}) error: {str(e)}")
        # Delete the flow entity from data access object
        self.dao.delete(query_request)
        return inst

    def get_list(self, request: ServeRequest) -> List[ServerResponse]:
        """Get a list of Flow entities

        Args:
            request (ServeRequest): The request object containing query parameters

        Returns:
            List[ServerResponse]: A list of response objects representing retrieved flow entities
        """
        # TODO: implement your own logic here
        # Build the query request from the request
        query_request = request
        # Retrieve a list of flow entities based on query request
        return self.dao.get_list(query_request)

    def get_list_by_page(
        self, request: QUERY_SPEC, page: int, page_size: int
    ) -> PaginationResult[ServerResponse]:
        """Get a list of Flow entities by page

        Args:
            request (ServeRequest): The request object containing query parameters
            page (int): The page number
            page_size (int): The size of each page

        Returns:
            PaginationResult[ServerResponse]: A paginated result of response objects representing retrieved flow entities
        """
        # Retrieve a paginated list of flow entities based on query request, page number, and page size
        page_result = self.dao.get_list_page(request, page, page_size)
        for item in page_result.items:
            # Retrieve metadata associated with each flow entity's DAG
            metadata = self.dag_manager.get_dag_metadata(
                item.dag_id, alias_name=item.uid
            )
            if metadata:
                # Convert metadata object to dictionary and assign it to flow entity's metadata
                item.metadata = metadata.to_dict()
        return page_result

    async def chat_stream_flow_str(
        self, flow_uid: str, request: CommonLLMHttpRequestBody
    ) -> str:
        """Stream chat for a specific flow entity

        Args:
            flow_uid (str): The unique identifier of the flow entity
            request (CommonLLMHttpRequestBody): The request object for the chat stream

        Returns:
            str: A string representing the chat stream for the specified flow entity
        """
    ) -> AsyncIterator[str]:
        """Stream chat with the AWEL flow.

        Args:
            flow_uid (str): The flow uid
            request (CommonLLMHttpRequestBody): The request
        """
        # 设置请求为非增量模式
        request.incremental = False
        # 异步迭代安全聊天流程输出
        async for output in self.safe_chat_stream_flow(flow_uid, request):
            text = output.text
            # 如果存在文本输出，将换行符替换为转义字符
            if text:
                text = text.replace("\n", "\\n")
            # 如果输出有错误码，生成包含错误信息的数据流消息并中断循环
            if output.error_code != 0:
                yield f"data:[SERVER_ERROR]{text}\n\n"
                break
            else:
                # 否则生成包含文本数据的数据流消息
                yield f"data:{text}\n\n"

    async def chat_stream_openai(
        self, flow_uid: str, request: CommonLLMHttpRequestBody
    ) -> AsyncIterator[str]:
        conv_uid = request.conv_uid
        # 创建聊天完成响应的选择数据
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None,
        )
        # 创建聊天完成流响应的片段数据
        chunk = ChatCompletionStreamResponse(
            id=conv_uid, choices=[choice_data], model=request.model
        )
        # 将响应数据转换为 JSON 格式
        json_data = model_to_json(chunk, exclude_unset=True, ensure_ascii=False)

        # 生成包含 JSON 数据的数据流消息
        yield f"data: {json_data}\n\n"

        # 设置请求为增量模式
        request.incremental = True
        # 异步迭代安全聊天流程输出
        async for output in self.safe_chat_stream_flow(flow_uid, request):
            # 如果输出不成功，生成包含输出字典的数据流消息，然后结束流
            if not output.success:
                yield f"data: {json.dumps(output.to_dict(), ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            # 创建带有助手角色和内容的聊天完成响应选择数据
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content=output.text),
            )
            # 创建聊天完成流响应的片段数据
            chunk = ChatCompletionStreamResponse(
                id=conv_uid,
                choices=[choice_data],
                model=request.model,
            )
            # 将响应数据转换为 JSON 格式
            json_data = model_to_json(chunk, exclude_unset=True, ensure_ascii=False)
            # 生成包含 JSON 数据的数据流消息
            yield f"data: {json_data}\n\n"
        # 生成表示完成的数据流消息
        yield "data: [DONE]\n\n"

    async def safe_chat_flow(
        self, flow_uid: str, request: CommonLLMHttpRequestBody
    ) -> ModelOutput:
        """Chat with the AWEL flow.

        Args:
            flow_uid (str): The flow uid
            request (CommonLLMHttpRequestBody): The request

        Returns:
            ModelOutput: The output
        """
        # 获取请求的增量模式设置
        incremental = request.incremental
        try:
            # 获取可调用任务并使用安全的 DAG 任务聊天
            task = await self._get_callable_task(flow_uid)
            return await safe_chat_with_dag_task(task, request)
        except HTTPException as e:
            # 处理 HTTP 异常并返回包含详细错误信息的模型输出
            return ModelOutput(error_code=1, text=e.detail, incremental=incremental)
        except Exception as e:
            # 处理其他异常并返回包含异常信息的模型输出
            return ModelOutput(error_code=1, text=str(e), incremental=incremental)

    async def safe_chat_stream_flow(
        self, flow_uid: str, request: CommonLLMHttpRequestBody
    ) -> AsyncIterator[str]:
        """Stream chat safely with the AWEL flow.

        Args:
            flow_uid (str): The flow uid
            request (CommonLLMHttpRequestBody): The request
        """
        # 这个方法的注释在上面的代码段已经给出
    ) -> AsyncIterator[ModelOutput]:
        """Stream chat with the AWEL flow.

        Args:
            flow_uid (str): The flow uid
            request (CommonLLMHttpRequestBody): The request

        Returns:
            AsyncIterator[ModelOutput]: The output
        """
        incremental = request.incremental  # 从请求对象中获取增量标志
        try:
            task = await self._get_callable_task(flow_uid)  # 调用私有方法获取可调用任务对象
            async for output in safe_chat_stream_with_dag_task(  # 使用安全聊天流和DAG任务异步迭代输出
                task, request, incremental
            ):
                yield output  # 生成聊天输出结果
        except HTTPException as e:
            yield ModelOutput(error_code=1, text=e.detail, incremental=incremental)  # 处理HTTP异常并生成错误模型输出
        except Exception as e:
            yield ModelOutput(error_code=1, text=str(e), incremental=incremental)  # 处理其他异常并生成错误模型输出

    async def _get_callable_task(
        self,
        flow_uid: str,
    ) -> BaseOperator:
        """Return the callable task.

        Returns:
            BaseOperator: The callable task

        Raises:
            HTTPException: If the flow is not found
            ValueError: If the flow is not a chat flow or the leaf node is not found.
        """
        flow = self.get({"uid": flow_uid})  # 根据流的UID从某处获取流对象
        if not flow:
            raise HTTPException(status_code=404, detail=f"Flow {flow_uid} not found")  # 如果流对象不存在，则抛出HTTP异常
        dag_id = flow.dag_id  # 获取流对象的DAG ID
        if not dag_id or dag_id not in self.dag_manager.dag_map:
            raise HTTPException(
                status_code=404, detail=f"Flow {flow_uid}'s dag id not found"
            )  # 如果DAG ID不存在或未在DAG管理器中找到，则抛出HTTP异常
        dag = self.dag_manager.dag_map[dag_id]  # 获取DAG对象
        leaf_nodes = dag.leaf_nodes  # 获取DAG的叶子节点列表
        if len(leaf_nodes) != 1:
            raise ValueError("Chat Flow just support one leaf node in dag")  # 如果叶子节点数量不为1，则抛出值错误异常
        return cast(BaseOperator, leaf_nodes[0])  # 返回叶子节点作为可调用任务对象
    def _parse_flow_category(self, dag: DAG) -> FlowCategory:
        """解析流程分类

        Args:
            dag (DAG): Directed Acyclic Graph representing the workflow

        Returns:
            FlowCategory: 返回流程的分类

        """
        from dbgpt.core.awel.flow.base import _get_type_cls

        # 获取触发节点和叶子节点
        triggers = dag.trigger_nodes
        leaf_nodes = dag.leaf_nodes

        # 检查条件，确定流程分类
        if (
            not triggers  # 如果没有触发节点
            or not leaf_nodes  # 如果没有叶子节点
            or len(leaf_nodes) > 1  # 如果叶子节点数量大于1
            or not isinstance(leaf_nodes[0], BaseOperator)  # 如果第一个叶子节点不是 BaseOperator 的实例
        ):
            return FlowCategory.COMMON  # 返回普通流程分类

        common_http_trigger = False
        # 检查触发器列表，确定是否存在 CommonLLMHttpTrigger
        for trigger in triggers:
            if isinstance(trigger, CommonLLMHttpTrigger):
                common_http_trigger = True
                break

        leaf_node = cast(BaseOperator, leaf_nodes[0])  # 将第一个叶子节点强制转换为 BaseOperator 类型

        # 检查叶子节点的元数据和输出
        if not leaf_node.metadata or not leaf_node.metadata.outputs:
            return FlowCategory.COMMON  # 返回普通流程分类

        output = leaf_node.metadata.outputs[0]  # 获取第一个输出

        try:
            real_class = _get_type_cls(output.type_cls)  # 获取输出类型的实际类
            # 如果存在 CommonLLMHttpTrigger 并且输出类型为聊天流类型，则返回聊天流分类
            if common_http_trigger and is_chat_flow_type(real_class, is_class=True):
                return FlowCategory.CHAT_FLOW
        except Exception:
            return FlowCategory.COMMON  # 处理异常情况，返回普通流程分类
```
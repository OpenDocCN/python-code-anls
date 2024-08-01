# `.\DB-GPT-src\dbgpt\core\awel\flow\flow_factory.py`

```py
"""Build AWEL DAGs from serialized data."""

import logging  # 导入日志模块
import uuid  # 导入 UUID 模块
from contextlib import suppress  # 导入上下文管理模块中的 suppress 函数
from enum import Enum  # 导入枚举模块
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast  # 导入类型注解相关的工具

from typing_extensions import Annotated  # 导入 Annotated 类型

from dbgpt._private.pydantic import (  # 导入 pydantic 模块中的相关功能
    BaseModel,  # 导入 pydantic 的基础模型
    ConfigDict,  # 导入配置字典
    Field,  # 导入字段
    WithJsonSchema,  # 导入 JsonSchema 相关功能
    field_validator,  # 导入字段验证器
    model_to_dict,  # 导入模型转换为字典的函数
    model_validator,  # 导入模型验证器
)
from dbgpt.core.awel.dag.base import DAG, DAGNode  # 导入 DAG 相关类
from dbgpt.core.awel.dag.dag_manager import DAGMetadata  # 导入 DAG 元数据

from .base import (  # 导入当前包中的相关模块和函数
    OperatorType,  # 导入操作符类型
    ResourceMetadata,  # 导入资源元数据
    ResourceType,  # 导入资源类型
    ViewMetadata,  # 导入视图元数据
    _get_operator_class,  # 导入获取操作符类的函数
    _get_resource_class,  # 导入获取资源类的函数
)
from .compat import get_new_class_name  # 导入获取新类名的兼容函数
from .exceptions import (  # 导入当前包中的异常类
    FlowClassMetadataException,  # 导入流程类元数据异常
    FlowDAGMetadataException,  # 导入流程 DAG 元数据异常
    FlowException,  # 导入流程异常
    FlowMetadataException,  # 导入流程元数据异常
)

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

AWEL_FLOW_VERSION = "0.1.1"  # 定义 AWEL 流程版本号


class FlowPositionData(BaseModel):
    """Position of a node in a flow."""

    x: float = Field(
        ..., description="X coordinate of the node", examples=[1081.1, 1000.9]
    )
    y: float = Field(
        ..., description="Y coordinate of the node", examples=[-113.7, -122]
    )
    zoom: float = Field(0, description="Zoom level of the node")


class FlowNodeData(BaseModel):
    """Node data in a flow."""

    width: int = Field(
        ...,
        description="Width of the node",
        examples=[300, 250],
    )
    height: int = Field(..., description="Height of the node", examples=[378, 400])
    id: str = Field(
        ...,
        description="Id of the node",
        examples=[
            "operator_llm_operator___$$___llm___$$___v1_0",
            "resource_dbgpt.model.proxy.llms.chatgpt.OpenAILLMClient_0",
        ],
    )
    position: FlowPositionData = Field(..., description="Position of the node")
    type: Optional[str] = Field(
        default=None,
        description="Type of current UI node(Just for UI)",
        examples=["customNode"],
    )
    data: Union[ViewMetadata, ResourceMetadata] = Field(
        ..., description="Data of the node"
    )
    position_absolute: FlowPositionData = Field(
        ..., description="Absolute position of the node"
    )

    @field_validator("data", mode="before")
    @classmethod
    def parse_data(cls, value: Any):
        """Parse the data."""
        if isinstance(value, dict):
            flow_type = value.get("flow_type")
            if flow_type == "operator":
                return ViewMetadata(**value)  # 如果是操作符类型，返回视图元数据对象
            elif flow_type == "resource":
                return ResourceMetadata(**value)  # 如果是资源类型，返回资源元数据对象
        raise ValueError("Unable to infer the type for `data`")  # 如果无法推断出 `data` 的类型，则抛出值错误


class FlowEdgeData(BaseModel):
    """Edge data in a flow."""

    source: str = Field(
        ...,
        description="Source node data id",
        examples=["resource_dbgpt.model.proxy.llms.chatgpt.OpenAILLMClient_0"],
    )
    # 定义字段 source_order，表示源节点在其输出中的顺序
    source_order: int = Field(
        description="The order of the source node in the source node's output",
        examples=[0, 1],
    )

    # 定义字段 target，表示目标节点的数据 ID
    target: str = Field(
        ...,
        description="Target node data id",
        examples=[
            "operator_llm_operator___$$___llm___$$___v1_0",
        ],
    )

    # 定义字段 target_order，表示目标节点在源节点输出中的顺序
    target_order: int = Field(
        description="The order of the target node in the source node's output",
        examples=[0, 1],
    )

    # 定义字段 id，表示边的唯一标识符
    id: str = Field(..., description="Id of the edge", examples=["edge_0"])

    # 定义可选字段 source_handle，用于在用户界面中表示源节点的句柄
    source_handle: Optional[str] = Field(
        default=None,
        description="Source handle, used in UI",
    )

    # 定义可选字段 target_handle，用于在用户界面中表示目标节点的句柄
    target_handle: Optional[str] = Field(
        default=None,
        description="Target handle, used in UI",
    )

    # 定义可选字段 type，表示当前 UI 节点的类型（仅用于用户界面）
    type: Optional[str] = Field(
        default=None,
        description="Type of current UI node (Just for UI)",
        examples=["buttonedge"],
    )

    # 类方法装饰器，用于模型验证，在实际验证之前预先填充字段值
    @model_validator(mode="before")
    @classmethod
    def pre_fill(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Pre fill the metadata."""
        # 如果传入的值不是字典类型，则直接返回
        if not isinstance(values, dict):
            return values
        
        # 如果 source_order 不存在但 source_handle 存在且不为 None，则尝试从 source_handle 中提取并填充 source_order
        if (
            "source_order" not in values
            and "source_handle" in values
            and values["source_handle"] is not None
        ):
            with suppress(Exception):
                values["source_order"] = int(values["source_handle"].split("|")[-1])
        
        # 如果 target_order 不存在但 target_handle 存在且不为 None，则尝试从 target_handle 中提取并填充 target_order
        if (
            "target_order" not in values
            and "target_handle" in values
            and values["target_handle"] is not None
        ):
            with suppress(Exception):
                values["target_order"] = int(values["target_handle"].split("|")[-1])
        
        # 返回填充后的值字典
        return values
class FlowData(BaseModel):
    """Flow data."""

    # 定义流程中的节点列表，类型为 FlowNodeData 的列表，不能为空
    nodes: List[FlowNodeData] = Field(..., description="Nodes in the flow")
    # 定义流程中的边列表，类型为 FlowEdgeData 的列表，不能为空
    edges: List[FlowEdgeData] = Field(..., description="Edges in the flow")
    # 定义流程的视口，类型为 FlowPositionData，不能为空
    viewport: FlowPositionData = Field(..., description="Viewport of the flow")


class State(str, Enum):
    """State of a flow panel."""

    # 定义流程面板的状态枚举
    INITIALIZING = "initializing"
    DEVELOPING = "developing"
    TESTING = "testing"
    DEPLOYED = "deployed"
    RUNNING = "running"
    DISABLED = "disabled"
    LOAD_FAILED = "load_failed"

    @classmethod
    def value_of(cls, value: Optional[str]) -> "State":
        """Get the state by value."""
        # 如果值为空，返回初始化状态
        if not value:
            return cls.INITIALIZING
        # 遍历所有状态，根据值返回对应的状态
        for state in State:
            if state.value == value:
                return state
        # 如果值无效，抛出值错误异常
        raise ValueError(f"Invalid state value: {value}")

    @classmethod
    def can_change_state(cls, current_state: "State", new_state: "State") -> bool:
        """Change the state of the flow panel."""
        # 定义允许的状态转换字典
        allowed_transitions: Dict[State, List[State]] = {
            State.INITIALIZING: [
                State.DEVELOPING,
                State.INITIALIZING,
                State.LOAD_FAILED,
            ],
            State.DEVELOPING: [
                State.TESTING,
                State.DEPLOYED,
                State.DISABLED,
                State.DEVELOPING,
                State.LOAD_FAILED,
            ],
            State.TESTING: [
                State.TESTING,
                State.DEPLOYED,
                State.DEVELOPING,
                State.DISABLED,
                State.RUNNING,
                State.LOAD_FAILED,
            ],
            State.DEPLOYED: [
                State.DEPLOYED,
                State.DEVELOPING,
                State.TESTING,
                State.DISABLED,
                State.RUNNING,
                State.LOAD_FAILED,
            ],
            State.RUNNING: [
                State.RUNNING,
                State.DEPLOYED,
                State.TESTING,
                State.DISABLED,
            ],
            State.DISABLED: [State.DISABLED, State.DEPLOYED],
            State.LOAD_FAILED: [
                State.LOAD_FAILED,
                State.DEVELOPING,
                State.DEPLOYED,
                State.DISABLED,
            ],
        }
        # 如果新状态在允许的当前状态转换列表中，返回 True
        if new_state in allowed_transitions[current_state]:
            return True
        else:
            # 如果不允许的状态转换，记录错误日志并返回 False
            logger.error(
                f"Invalid state transition from {current_state} to {new_state}"
            )
            return False


class FlowCategory(str, Enum):
    """Flow category."""

    # 定义流程的类别枚举
    COMMON = "common"
    CHAT_FLOW = "chat_flow"
    CHAT_AGENT = "chat_agent"

    @classmethod
    # 定义一个类方法，用于根据值获取对应的流程类别对象
    def value_of(cls, value: Optional[str]) -> "FlowCategory":
        """Get the flow category by value."""
        # 如果值为空，返回默认的 COMMON 类别
        if not value:
            return cls.COMMON
        # 遍历所有的 FlowCategory 枚举对象
        for category in FlowCategory:
            # 如果枚举对象的值等于给定的值，返回该枚举对象
            if category.value == value:
                return category
        # 如果未找到匹配的枚举对象，抛出 ValueError 异常
        raise ValueError(f"Invalid flow category value: {value}")
_DAGModel = Annotated[
    DAG,  # _DAGModel 是一个被注解的类型，代表一个 DAG 对象
    WithJsonSchema(  # 使用 WithJsonSchema 注解，指定了一个 JSON 模式
        {
            "type": "object",  # JSON 模式的类型为对象
            "properties": {
                "task_name": {"type": "string", "description": "Dummy task name"}  # 属性 task_name 是一个字符串，描述为虚拟任务名称
            },
            "description": "DAG model, not used in the serialization.",  # 描述信息，指出这个 DAG 模型不在序列化中使用
        }
    ),
]


class FlowPanel(BaseModel):
    """Flow panel."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_encoders={DAG: lambda v: None}  # 模型配置允许任意类型，且指定 DAG 类型的 JSON 编码器
    )

    uid: str = Field(
        default_factory=lambda: str(uuid.uuid4()),  # 默认工厂函数生成唯一标识符 uid
        description="Flow panel uid",  # 描述字段为流面板的唯一标识符
        examples=[  # 示例值
            "5b25ac8a-ba8e-11ee-b96d-3b9bfdeebd1c",
            "6a4752ae-ba8e-11ee-afff-af8fd9bfe727",
        ],
    )
    label: str = Field(
        ..., description="Flow panel label", examples=["First AWEL Flow", "My LLM Flow"]  # 流面板的标签描述和示例值
    )
    name: str = Field(
        ..., description="Flow panel name", examples=["first_awel_flow", "my_llm_flow"]  # 流面板的名称描述和示例值
    )
    flow_category: Optional[FlowCategory] = Field(
        default=FlowCategory.COMMON,  # 默认流类别为 COMMON
        description="Flow category",  # 描述为流类别
        examples=[FlowCategory.COMMON, FlowCategory.CHAT_AGENT],  # 示例值包括 COMMON 和 CHAT_AGENT
    )
    flow_data: Optional[FlowData] = Field(None, description="Flow data")  # 可选的流数据字段，描述为流数据
    flow_dag: Optional[_DAGModel] = Field(None, description="Flow DAG", exclude=True)  # 可选的流 DAG 字段，描述为流 DAG，但排除在序列化之外
    description: Optional[str] = Field(
        None,
        description="Flow panel description",  # 描述为流面板描述
        examples=["My first AWEL flow"],  # 示例描述
    )
    state: State = Field(
        default=State.INITIALIZING,  # 默认状态为 INITIALIZING
        description="Current state of the flow panel"  # 描述为流面板的当前状态
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message of load the flow panel",  # 描述为加载流面板时的错误消息
        examples=["Unable to load the flow panel."],  # 示例错误消息
    )
    source: Optional[str] = Field(
        "DBGPT-WEB",  # 默认来源为 DBGPT-WEB
        description="Source of the flow panel",  # 描述为流面板的来源
        examples=["DB-GPT-WEB", "DBGPT-GITHUB"],  # 示例来源
    )
    source_url: Optional[str] = Field(
        None,
        description="Source url of the flow panel",  # 描述为流面板的来源 URL
    )
    version: Optional[str] = Field(
        AWEL_FLOW_VERSION,  # 版本字段的默认值为 AWEL_FLOW_VERSION
        description="Version of the flow panel",  # 描述为流面板的版本
        examples=["0.1.0", "0.2.0"],  # 示例版本号
    )
    define_type: Optional[str] = Field(
        "json",  # 默认定义类型为 json
        description="Define type of the flow panel",  # 描述为流面板的定义类型
        examples=["json", "python"],  # 示例定义类型
    )
    editable: bool = Field(
        True,  # 可编辑属性默认为 True
        description="Whether the flow panel is editable",  # 描述为流面板是否可编辑
        examples=[True, False],  # 示例值为 True 或 False
    )
    user_name: Optional[str] = Field(None, description="User name")  # 可选的用户名称字段，描述为用户名称
    sys_code: Optional[str] = Field(None, description="System code")  # 可选的系统代码字段，描述为系统代码
    dag_id: Optional[str] = Field(None, description="DAG id, Created by AWEL")  # 可选的 DAG ID 字段，描述为 DAG ID，由 AWEL 创建

    gmt_created: Optional[str] = Field(
        None,
        description="The flow panel created time.",  # 描述为流面板的创建时间
        examples=["2021-08-01 12:00:00", "2021-08-01 12:00:01", "2021-08-01 12:00:02"],  # 示例创建时间
    )
    gmt_modified: Optional[str] = Field(
# 定义一个 FlowFactory 类，用于构建流程（DAG）的工厂。
class FlowFactory:
    """Flow factory."""

    def __init__(self, dag_prefix: str = "flow_dag"):
        """初始化流程工厂。

        Args:
            dag_prefix (str, optional): 流程图的前缀，默认为 "flow_dag"。
        """
        self._dag_prefix = dag_prefix

    def build_dag(
        self,
        flow_panel: FlowPanel,
        key_to_tasks: Dict[str, DAGNode],
        key_to_downstream: Dict[str, List[Tuple[str, int, int]]],
        key_to_upstream: Dict[str, List[Tuple[str, int, int]]],
        dag_id: Optional[str] = None,
    ) -> DAG:
        """构建流程图（DAG）。

        Args:
            flow_panel (FlowPanel): 流程面板对象，用于构建流程图。
            key_to_tasks (Dict[str, DAGNode]): 字典，将任务关键字映射到 DAGNode 对象。
            key_to_downstream (Dict[str, List[Tuple[str, int, int]]]): 字典，将任务关键字映射到下游任务列表。
            key_to_upstream (Dict[str, List[Tuple[str, int, int]]]): 字典，将任务关键字映射到上游任务列表。
            dag_id (Optional[str], optional): 流程图的唯一标识符，如果未提供，则根据流程面板自动生成。默认为 None。

        Returns:
            DAG: 构建完成的流程图对象。
        """
        # 格式化流程面板的名称，将空格替换为下划线
        formatted_name = flow_panel.name.replace(" ", "_")
        # 如果没有提供流程图 ID，则生成一个基于前缀、格式化名称和流程面板 UID 的新 ID
        if not dag_id:
            dag_id = f"{self._dag_prefix}_{formatted_name}_{flow_panel.uid}"
        
        # 使用 DAG 对象创建上下文环境
        with DAG(dag_id) as dag:
            # 遍历所有任务关键字及其对应的任务节点
            for key, task in key_to_tasks.items():
                # 如果任务节点没有设置节点 ID，则分配一个新的节点 ID
                if not task._node_id:
                    task.set_node_id(dag._new_node_id())
                
                # 获取当前任务节点的下游任务列表和上游任务列表
                downstream = key_to_downstream.get(key, [])
                upstream = key_to_upstream.get(key, [])
                
                # 将 DAG 对象与任务节点关联起来
                task._dag = dag
                
                # 如果既没有下游任务也没有上游任务，则当前任务为单一任务，直接添加到流程图中
                if not downstream and not upstream:
                    dag._append_node(task)
                    continue

                # 遍历上游任务列表，按照顺序连接上游任务和当前任务
                for upstream_key, _, _ in upstream:
                    # 根据上游任务关键字获取上游任务对象
                    upstream_task = key_to_tasks.get(upstream_key)
                    if not upstream_task:
                        raise ValueError(
                            f"无法找到关键字为 {upstream_key} 的上游任务。"
                        )
                    # 如果上游任务没有设置节点 ID，则分配一个新的节点 ID
                    if not upstream_task._node_id:
                        upstream_task.set_node_id(dag._new_node_id())
                    
                    # 将上游任务连接到当前任务
                    upstream_task >> task
            
            # 返回构建完成的流程图对象
            return dag
    def pre_load_requirements(self, flow_panel: FlowPanel):
        """预加载流面板所需的模块。

        Args:
            flow_panel (FlowPanel): 流面板对象
        """
        from dbgpt.util.module_utils import import_from_string  # 导入从字符串动态导入模块的工具函数

        if not flow_panel.flow_data:
            return  # 如果流面板没有流数据，则直接返回

        flow_data = cast(FlowData, flow_panel.flow_data)  # 将流面板的流数据强制类型转换为FlowData类型
        for node in flow_data.nodes:  # 遍历流数据中的节点
            if node.data.is_operator:  # 如果节点的数据是操作符
                node_data = cast(ViewMetadata, node.data)  # 将节点的数据强制类型转换为ViewMetadata类型
            else:
                node_data = cast(ResourceMetadata, node.data)  # 否则将节点的数据强制类型转换为ResourceMetadata类型
            if not node_data.type_cls:  # 如果节点的数据中没有类型类别，则跳过
                continue
            try:
                metadata_cls = import_from_string(node_data.type_cls)  # 尝试动态导入指定的类型类别
                logger.debug(
                    f"成功导入 {node_data.type_cls}，metadata_cls 是: "
                    f"{metadata_cls}"
                )
            except ImportError as e:
                raise_error = True
                new_type_cls: Optional[str] = None
                try:
                    new_type_cls = get_new_class_name(node_data.type_cls)  # 获取替换后的新类型类别名称
                    if new_type_cls:
                        metadata_cls = import_from_string(new_type_cls)  # 尝试使用新类型类别名称进行动态导入
                        logger.info(
                            f"成功导入 {new_type_cls}，metadata_cls 是: "
                            f"{metadata_cls}"
                        )
                        raise_error = False
                except ImportError as ex:
                    raise FlowClassMetadataException(
                        f"导入 {node_data.type_cls}，使用新类型 {new_type_cls} 失败：{ex}"
                    )
                if raise_error:
                    raise FlowClassMetadataException(
                        f"导入 {node_data.type_cls} 失败：{e}"
                    )
def _topological_sort(
    key_to_upstream_node: Dict[str, List[FlowNodeData]]
) -> Dict[str, int]:
    """Topological sort.

    Returns the topological order of the nodes and checks if the graph has at least
    one cycle.

    Args:
        key_to_upstream_node (Dict[str, List[FlowNodeData]]): The upstream nodes

    Returns:
        Dict[str, int]: The topological order of the nodes

    Raises:
        ValueError: Graph has at least one cycle
    """
    from collections import deque

    # Initialize an empty dictionary to store the topological order of nodes
    key_to_order: Dict[str, int] = {}
    # Initialize the current order counter
    current_order = 0

    # Collect all unique keys (nodes) in the graph
    keys = set()
    for key, upstreams in key_to_upstream_node.items():
        keys.add(key)
        for upstream in upstreams:
            keys.add(upstream.id)

    # Initialize in-degree for each key (node) to 0
    in_degree = {key: 0 for key in keys}
    # Initialize an empty graph dictionary where each key (node) maps to an empty list
    graph: Dict[str, List[str]] = {key: [] for key in keys}

    # Build the graph: connect each key to its downstream nodes
    for key in key_to_upstream_node:
        for node in key_to_upstream_node[key]:
            graph[node.id].append(key)
            in_degree[key] += 1

    # Find all nodes with in-degree 0 and add them to the queue
    queue = deque([key for key, degree in in_degree.items() if degree == 0])
    while queue:
        # Pop a node from the queue
        current_key: str = queue.popleft()
        # Assign the current order to the popped node
        key_to_order[current_key] = current_order
        current_order += 1

        # Process each adjacent node
        for adjacent in graph[current_key]:
            # Decrease the in-degree of the adjacent node
            in_degree[adjacent] -= 1
            # If the in-degree becomes 0, add the adjacent node to the queue
            if in_degree[adjacent] == 0:
                queue.append(adjacent)

    # If the number of nodes processed is not equal to the total number of nodes,
    # there must be a cycle in the graph
    if current_order != len(keys):
        raise ValueError("Graph has at least one cycle")

    # Return the dictionary containing the topological order of nodes
    return key_to_order


def fill_flow_panel(flow_panel: FlowPanel):
    """Fill the flow panel with the latest metadata.

    Args:
        flow_panel (FlowPanel): The flow panel to fill.
    """
    # If flow_data in the flow panel is empty, return early
    if not flow_panel.flow_data:
        return
    # 遍历流数据面板中的每个节点
    for node in flow_panel.flow_data.nodes:
        try:
            # 初始化参数映射字典
            parameters_map = {}

            # 如果节点是操作符
            if node.data.is_operator:
                # 将节点数据视为视图元数据
                data = cast(ViewMetadata, node.data)
                # 获取操作符的唯一标识符
                key = data.get_operator_key()
                # 获取操作符对应的类
                operator_cls: Type[DAGNode] = _get_operator_class(key)
                # 获取操作符的元数据
                metadata = operator_cls.metadata
                # 如果元数据未设置，抛出异常
                if not metadata:
                    raise ValueError("Metadata is not set.")
                # 创建输入参数的名称到参数对象的映射
                input_parameters = {p.name: p for p in metadata.inputs}
                # 创建输出参数的名称到参数对象的映射
                output_parameters = {p.name: p for p in metadata.outputs}

                # 更新节点的输入参数
                for i in node.data.inputs:
                    if i.name in input_parameters:
                        new_param = input_parameters[i.name]
                        i.label = new_param.label
                        i.description = new_param.description

                # 更新节点的输出参数
                for i in node.data.outputs:
                    if i.name in output_parameters:
                        new_param = output_parameters[i.name]
                        i.label = new_param.label
                        i.description = new_param.description

            else:
                # 将节点数据视为资源元数据
                data = cast(ResourceMetadata, node.data)
                # 获取资源的原始标识符
                key = data.get_origin_id()
                # 获取资源对应的类的元数据
                metadata = _get_resource_class(key).metadata

            # 将元数据中的参数列表映射到参数名到参数对象的字典中
            for param in metadata.parameters:
                parameters_map[param.name] = param

            # 更新节点数据的最新元数据信息
            node.data.label = metadata.label
            node.data.description = metadata.description
            node.data.category = metadata.category
            node.data.tags = metadata.tags
            node.data.icon = metadata.icon
            node.data.documentation_url = metadata.documentation_url

            # 更新节点数据中的参数信息
            for param in node.data.parameters:
                if param.name in parameters_map:
                    new_param = parameters_map[param.name]
                    param.label = new_param.label
                    param.description = new_param.description
                    param.options = new_param.get_dict_options()  # type: ignore
                    param.default = new_param.default
                    param.placeholder = new_param.placeholder

        except (FlowException, ValueError) as e:
            # 捕获流异常或数值错误异常，记录警告日志
            logger.warning(f"Unable to fill the flow panel: {e}")
```
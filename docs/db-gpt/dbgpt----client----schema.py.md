# `.\DB-GPT-src\dbgpt\client\schema.py`

```py
"""this module contains the schemas for the dbgpt client."""

# 导入必要的模块和类
import json  # 导入 json 模块，用于处理 JSON 数据
from datetime import datetime  # 从 datetime 模块中导入 datetime 类
from enum import Enum  # 导入 Enum 类，用于定义枚举类型
from typing import Any, Dict, List, Optional, Union  # 导入用于类型提示的各种类型

# 导入 FastAPI 框架中的 File 和 UploadFile 类
from fastapi import File, UploadFile  

# 导入 dbgpt 内部的一些模块和类
from dbgpt._private.pydantic import BaseModel, ConfigDict, Field  # 导入 Pydantic 中的 BaseModel, ConfigDict, Field 类
from dbgpt.rag.chunk_manager import ChunkParameters  # 导入 dbgpt.rag.chunk_manager 中的 ChunkParameters 类


# 定义一个用于聊天完成请求的请求体模型
class ChatCompletionRequestBody(BaseModel):
    """ChatCompletion LLM http request body."""

    model: str = Field(
        ..., description="The model name", examples=["gpt-3.5-turbo", "proxyllm"]
    )  # 模型名称，字符串类型，必填字段，包含示例值
    messages: Union[str, List[str]] = Field(
        ..., description="User input messages", examples=["Hello", "How are you?"]
    )  # 用户输入的消息，可以是单个字符串或字符串列表，必填字段，包含示例值
    stream: bool = Field(default=True, description="Whether return stream")  # 是否返回流式数据，默认为 True

    temperature: Optional[float] = Field(
        default=None,
        description="What sampling temperature to use, between 0 and 2. Higher values "
        "like 0.8 will make the output more random, "
        "while lower values like 0.2 will "
        "make it more focused and deterministic.",
    )  # 采样温度，可选的浮点数，用于控制输出的随机性
    max_new_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens that can be generated in the chat "
        "completion.",
    )  # 在聊天完成中可以生成的最大标记数
    conv_uid: Optional[str] = Field(
        default=None, description="The conversation id of the model inference"
    )  # 模型推断的对话 ID，可选字符串
    span_id: Optional[str] = Field(
        default=None, description="The span id of the model inference"
    )  # 模型推断的 span ID，可选字符串
    chat_mode: Optional[str] = Field(
        default="chat_normal",
        description="The chat mode",
        examples=["chat_awel_flow", "chat_normal"],
    )  # 聊天模式，可选字符串，默认为 "chat_normal"，包含示例值
    chat_param: Optional[str] = Field(
        default=None,
        description="The chat param of chat mode",
    )  # 聊天模式的参数，可选字符串
    user_name: Optional[str] = Field(
        default=None, description="The user name of the model inference"
    )  # 模型推断的用户名称，可选字符串
    sys_code: Optional[str] = Field(
        default=None, description="The system code of the model inference"
    )  # 模型推断的系统代码，可选字符串
    incremental: bool = Field(
        default=True,
        description="Used to control whether the content is returned incrementally "
        "or in full each time. "
        "If this parameter is not provided, the default is full return.",
    )  # 是否增量返回内容，默认为 True
    enable_vis: bool = Field(
        default=True, description="response content whether to output vis label"
    )  # 是否输出可视化标签作为响应内容，默认为 True


# 定义聊天模式枚举类
class ChatMode(Enum):
    """Chat mode."""

    CHAT_NORMAL = "chat_normal"  # 普通聊天模式
    CHAT_APP = "chat_app"  # 应用聊天模式
    CHAT_AWEL_FLOW = "chat_flow"  # AWEL 流程聊天模式
    CHAT_KNOWLEDGE = "chat_knowledge"  # 知识聊天模式
    CHAT_DATA = "chat_data"  # 数据聊天模式


# 定义 AWEL 团队模型
class AWELTeamModel(BaseModel):
    """AWEL team model."""

    dag_id: str = Field(
        ...,
        description="The unique id of dag",
        examples=["flow_dag_testflow_66d8e9d6-f32e-4540-a5bd-ea0648145d0e"],
    )  # DAG 的唯一 ID，字符串类型，必填字段，包含示例值
    uid: str = Field(
        default=None,
        description="The unique id of flow",
        examples=["66d8e9d6-f32e-4540-a5bd-ea0648145d0e"],
    )  # 流程的唯一 ID，可选字符串，包含示例值
    name: Optional[str] = Field(
        default=None,
        description="The name of dag",
    )
    # DAG（Directed Acyclic Graph，有向无环图）的名称，可选参数，默认为None

    label: Optional[str] = Field(
        default=None,
        description="The label of dag",
    )
    # DAG的标签，可选参数，默认为None

    version: Optional[str] = Field(
        default=None,
        description="The version of dag",
    )
    # DAG的版本信息，可选参数，默认为None

    description: Optional[str] = Field(
        default=None,
        description="The description of dag",
    )
    # DAG的描述信息，可选参数，默认为None

    editable: bool = Field(
        default=False,
        description="is the dag is editable",
        examples=[True, False],
    )
    # 指示DAG是否可编辑的布尔值，默认为False

    state: Optional[str] = Field(
        default=None,
        description="The state of dag",
    )
    # DAG的状态信息，可选参数，默认为None

    user_name: Optional[str] = Field(
        default=None,
        description="The owner of current dag",
    )
    # 当前DAG的所有者用户名，可选参数，默认为None

    sys_code: Optional[str] = Field(
        default=None,
        description="The system code of current dag",
    )
    # 当前DAG的系统代码，可选参数，默认为None

    flow_category: Optional[str] = Field(
        default="common",
        description="The flow category of current dag",
    )
    # 当前DAG的流程类别，可选参数，默认为"common"
# 定义代理资源类型枚举，包括数据库、知识、网络、插件、文本文件、Excel 文件、图片文件和 AWEL 流
class AgentResourceType(Enum):
    """Agent resource type."""

    DB = "database"
    Knowledge = "knowledge"
    Internet = "internet"
    Plugin = "plugin"
    TextFile = "text_file"
    ExcelFile = "excel_file"
    ImageFile = "image_file"
    AWELFlow = "awel_flow"


# 定义代理资源模型，包含类型、名称、值、是否动态等字段
class AgentResourceModel(BaseModel):
    """Agent resource model."""

    type: str  # 资源类型
    name: str  # 资源名称
    value: str  # 资源值
    is_dynamic: bool = (
        False  # 当前资源是否预定义或动态传入？
    )

    @staticmethod
    def from_dict(d: Dict[str, Any]):
        """从字典构造 AgentResourceModel 对象."""
        if d is None:
            return None
        return AgentResourceModel(
            type=d.get("type"),
            name=d.get("name"),
            introduce=d.get("introduce"),  # 没有 introduce 字段，将返回 None
            value=d.get("value", None),
            is_dynamic=d.get("is_dynamic", False),
        )

    @staticmethod
    def from_json_list_str(d: Optional[str]):
        """从 JSON 列表字符串构造 AgentResourceModel 列表."""
        if d is None:
            return None
        try:
            json_array = json.loads(d)
        except Exception as e:
            raise ValueError(f"Illegal AgentResource json string！{d},{e}")
        return [AgentResourceModel.from_dict(item) for item in json_array]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典."""
        temp = self.dict()
        for field, value in temp.items():
            if isinstance(value, Enum):
                temp[field] = value.value  # 如果值是枚举类型，转换为枚举值
        return temp


# 定义应用详情模型，包含应用代码、应用名称、代理名称、节点 ID、资源列表、提示模板、LLM 策略及其值、创建时间和更新时间等字段
class AppDetailModel(BaseModel):
    """App detail model."""

    app_code: Optional[str] = Field(None, description="app code")  # 应用代码
    app_name: Optional[str] = Field(None, description="app name")  # 应用名称
    agent_name: Optional[str] = Field(None, description="agent name")  # 代理名称
    node_id: Optional[str] = Field(None, description="node id")  # 节点 ID
    resources: Optional[list[AgentResourceModel]] = Field(None, description="resources")  # 资源列表
    prompt_template: Optional[str] = Field(None, description="prompt template")  # 提示模板
    llm_strategy: Optional[str] = Field(None, description="llm strategy")  # LLM 策略
    llm_strategy_value: Optional[str] = Field(None, description="llm strategy value")  # LLM 策略值
    created_at: datetime = datetime.now()  # 创建时间
    updated_at: datetime = datetime.now()  # 更新时间


# 定义应用模型，包含应用代码、应用名称、应用描述、团队模式、语言、团队上下文、用户代码、系统代码、是否收藏、图标和创建时间等字段
class AppModel(BaseModel):
    """App model."""

    app_code: Optional[str] = Field(None, title="app code")  # 应用代码
    app_name: Optional[str] = Field(None, title="app name")  # 应用名称
    app_describe: Optional[str] = Field(None, title="app describe")  # 应用描述
    team_mode: Optional[str] = Field(None, title="team mode")  # 团队模式
    language: Optional[str] = Field("en", title="language")  # 语言，默认为英语
    team_context: Optional[Union[str, AWELTeamModel]] = Field(
        None, title="team context"
    )  # 团队上下文
    user_code: Optional[str] = Field(None, title="user code")  # 用户代码
    sys_code: Optional[str] = Field(None, title="sys code")  # 系统代码
    is_collected: Optional[str] = Field(None, title="is collected")  # 是否收藏
    icon: Optional[str] = Field(None, title="icon")  # 图标
    created_at: datetime = datetime.now()  # 创建时间
    # 获取当前时间并赋值给 updated_at 变量，使用 datetime 模块的 now() 方法
    updated_at: datetime = datetime.now()
    
    # 创建一个空列表并赋值给 details 变量，该列表用于存储 AppDetailModel 对象
    # 使用 Field() 函数进行字段定义，初始值为空列表，同时设置字段的标题为 "app details"
    details: List[AppDetailModel] = Field([], title="app details")
class SpaceModel(BaseModel):
    """Space model."""

    # 定义空间模型的属性，包括空间ID、名称、向量类型、描述、所有者和上下文
    id: str = Field(
        default=None,
        description="space id",
    )
    name: str = Field(
        default=None,
        description="knowledge space name",
    )
    vector_type: str = Field(
        default=None,
        description="vector type",
    )
    desc: str = Field(
        default=None,
        description="space description",
    )
    owner: str = Field(
        default=None,
        description="space owner",
    )
    context: Optional[str] = Field(
        default=None,
        description="space argument context",
    )


class DocumentModel(BaseModel):
    """Document model."""

    # 定义文档模型的属性，包括文档ID、文档名称、文档类型、内容、文档文件和文档来源
    id: int = Field(None, description="The doc id")
    doc_name: str = Field(None, description="doc name")
    doc_type: str = Field(None, description="The doc type")  # 文档类型
    content: str = Field(None, description="content")  # 文档内容
    doc_file: UploadFile = Field(File(None), description="doc file")  # 文档文件
    doc_source: str = Field(None, description="doc source")  # 文档来源
    space_id: str = Field(None, description="space_id")  # 所属空间ID


class SyncModel(BaseModel):
    """Sync model."""

    model_config = ConfigDict(protected_namespaces=())

    # 定义同步模型的属性，包括文档ID、空间ID、模型名称和块参数
    doc_id: str = Field(None, description="The doc id")
    space_id: str = Field(None, description="The space id")  # 同步的空间ID
    model_name: Optional[str] = Field(None, description="model name")  # 模型名称
    chunk_parameters: ChunkParameters = Field(None, description="chunk parameters")  # 块参数


class DatasourceModel(BaseModel):
    """Datasource model."""

    # 定义数据源模型的属性，包括数据源ID、数据库类型、数据库名称、文件路径、数据库主机、数据库端口、数据库用户、密码和注释
    id: Optional[int] = Field(None, description="The datasource id")
    db_type: str = Field(..., description="Database type, e.g. sqlite, mysql, etc.")
    db_name: str = Field(..., description="Database name.")
    db_path: str = Field("", description="File path for file-based database.")
    db_host: str = Field("", description="Database host.")
    db_port: int = Field(0, description="Database port.")
    db_user: str = Field("", description="Database user.")
    db_pwd: str = Field("", description="Database password.")
    comment: str = Field("", description="Comment for the database.")
```
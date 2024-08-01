# `.\DB-GPT-src\dbgpt\app\scene\base.py`

```py
from enum import Enum  # 导入枚举类型模块
from typing import List, Optional  # 导入类型提示模块中的 List 和 Optional

from dbgpt._private.pydantic import BaseModel, ConfigDict, Field  # 导入 pydantic 模块中的 BaseModel, ConfigDict, Field
from dbgpt.core import BaseOutputParser, ChatPromptTemplate  # 导入 core 模块中的 BaseOutputParser, ChatPromptTemplate
from dbgpt.core._private.example_base import ExampleSelector  # 导入 example_base 模块中的 ExampleSelector

class Scene:  # 定义 Scene 类
    def __init__(  # Scene 类的初始化方法
        self,
        code,  # 场景代码
        name,  # 场景名称
        describe,  # 场景描述
        param_types: List = [],  # 参数类型列表，默认为空列表
        is_inner: bool = False,  # 是否为内部场景，默认为 False
        show_disable=False,  # 是否显示禁用，默认为 False
        prepare_scene_code: str = None,  # 准备场景代码，默认为 None
    ):
        self.code = code  # 设置场景代码属性
        self.name = name  # 设置场景名称属性
        self.describe = describe  # 设置场景描述属性
        self.param_types = param_types  # 设置参数类型列表属性
        self.is_inner = is_inner  # 设置是否为内部场景属性
        self.show_disable = show_disable  # 设置是否显示禁用属性
        self.prepare_scene_code = prepare_scene_code  # 设置准备场景代码属性

class ChatScene(Enum):  # 定义 ChatScene 枚举类
    ChatWithDbExecute = Scene(  # ChatWithDbExecute 枚举成员
        code="chat_with_db_execute",
        name="Chat Data",
        describe="Dialogue with your private data through natural language.",
        param_types=["DB Select"],  # 参数类型为 "DB Select" 的列表
    )
    ExcelLearning = Scene(  # ExcelLearning 枚举成员
        code="excel_learning",
        name="Excel Learning",
        describe="Analyze and summarize your excel files.",
        is_inner=True,  # 是内部场景
    )
    ChatExcel = Scene(  # ChatExcel 枚举成员
        code="chat_excel",
        name="Chat Excel",
        describe="Dialogue with your excel, use natural language.",
        param_types=["File Select"],  # 参数类型为 "File Select" 的列表
        prepare_scene_code="excel_learning",  # 准备场景代码为 "excel_learning"
    )

    ChatWithDbQA = Scene(  # ChatWithDbQA 枚举成员
        code="chat_with_db_qa",
        name="Chat DB",
        describe="Have a Professional Conversation with Metadata.",
        param_types=["DB Select"],  # 参数类型为 "DB Select" 的列表
    )
    ChatExecution = Scene(  # ChatExecution 枚举成员
        code="chat_execution",
        name="Use Plugin",
        describe="Use tools through dialogue to accomplish your goals.",
        param_types=["Plugin Select"],  # 参数类型为 "Plugin Select" 的列表
    )

    ChatAgent = Scene(  # ChatAgent 枚举成员
        code="chat_agent",
        name="Agent Chat",
        describe="Use tools through dialogue to accomplish your goals.",
        param_types=["Plugin Select"],  # 参数类型为 "Plugin Select" 的列表
    )

    ChatFlow = Scene(  # ChatFlow 枚举成员
        code="chat_flow",
        name="Flow Chat",
        describe="Have conversations with conversational AWEL flow.",
        param_types=["Flow Select"],  # 参数类型为 "Flow Select" 的列表
    )

    InnerChatDBSummary = Scene(  # InnerChatDBSummary 枚举成员
        "inner_chat_db_summary", "DB Summary", "Db Summary.", True  # 内部场景，场景代码 "inner_chat_db_summary"，名称 "DB Summary"，描述 "Db Summary."
    )

    ChatNormal = Scene(  # ChatNormal 枚举成员
        "chat_normal", "Chat Normal", "Native LLM large model AI dialogue."  # 场景代码 "chat_normal"，名称 "Chat Normal"，描述 "Native LLM large model AI dialogue."
    )
    ChatDashboard = Scene(  # ChatDashboard 枚举成员
        "chat_dashboard",
        "Dashboard",
        "Provide you with professional analysis reports through natural language.",
        ["DB Select"],  # 参数类型为 "DB Select" 的列表
    )
    ChatKnowledge = Scene(  # ChatKnowledge 枚举成员
        "chat_knowledge",
        "Chat Knowledge",
        "Dialogue through natural language and private documents and knowledge bases.",
        ["Knowledge Space Select"],  # 参数类型为 "Knowledge Space Select" 的列表
    )
    ExtractSummary = Scene(  # ExtractSummary 枚举成员
        "extract_summary",
        "Extract Summary",
        "Extract Summary",
        ["Extract Select"],  # 参数类型为 "Extract Select" 的列表
        True,  # 是内部场景
    )
    # 创建一个名为 ExtractRefineSummary 的 Scene 对象实例
    ExtractRefineSummary = Scene(
        "extract_refine_summary",
        "Extract Summary",
        "Extract Summary",
        ["Extract Select"],
        True,
    )

    # 创建一个名为 QueryRewrite 的 Scene 对象实例
    QueryRewrite = Scene(
        "query_rewrite", "query_rewrite", "query_rewrite", ["query_rewrite"], True
    )

    # 静态方法：根据给定的 mode 返回 ChatScene 中对应的实例
    @staticmethod
    def of_mode(mode):
        return [x for x in ChatScene if mode == x.value()][0]

    # 静态方法：检查给定的 mode 是否是 ChatScene 中任一实例的值
    @staticmethod
    def is_valid_mode(mode):
        return any(mode == item.value() for item in ChatScene)

    # 实例方法：返回当前 Scene 实例的值（代码）
    def value(self):
        return self._value_.code

    # 实例方法：返回当前 Scene 实例的名称
    def scene_name(self):
        return self._value_.name

    # 实例方法：返回当前 Scene 实例的描述
    def describe(self):
        return self._value_.describe

    # 实例方法：返回当前 Scene 实例的参数类型列表
    def param_types(self):
        return self._value_.param_types

    # 实例方法：返回当前 Scene 实例是否显示禁用状态
    def show_disable(self):
        return self._value_.show_disable

    # 实例方法：返回当前 Scene 实例是否为内部场景
    def is_inner(self):
        return self._value_.is_inner
class AppScenePromptTemplateAdapter(BaseModel):
    """The template of the scene.

    Include some fields that in :class:`dbgpt.core.PromptTemplate`
    """

    # 定义模型配置字典，允许任意类型
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 场景的提示文本模板
    prompt: ChatPromptTemplate = Field(..., description="The prompt of this scene")

    # 此模板的场景（可选）
    template_scene: Optional[str] = Field(
        default=None, description="The scene of this template"
    )

    # 是否严格模式（默认为严格）
    template_is_strict: Optional[bool] = Field(
        default=True, description="Whether strict"
    )

    # 输出解析器（可选）
    output_parser: Optional[BaseOutputParser] = Field(
        default=None, description="The output parser of this scene"
    )

    # 默认分隔符（###）用于此场景
    sep: Optional[str] = Field(
        default="###", description="The default separator of this scene"
    )

    # 是否输出流输出（默认为True）
    stream_out: Optional[bool] = Field(
        default=True, description="Whether to stream out"
    )

    # 示例选择器（可选）
    example_selector: Optional[ExampleSelector] = Field(
        default=None, description="Example selector"
    )

    # 是否需要历史消息（默认为False）
    need_historical_messages: Optional[bool] = Field(
        default=False, description="Whether to need historical messages"
    )

    # 默认温度（默认为0.6）
    temperature: Optional[float] = Field(
        default=0.6, description="The default temperature of this scene"
    )

    # 默认新令牌的最大数目（默认为1024）
    max_new_tokens: Optional[int] = Field(
        default=1024, description="The default max new tokens of this scene"
    )

    # 是否将历史转换为字符串（默认为False）
    str_history: Optional[bool] = Field(
        default=False, description="Whether transform history to str"
    )
```
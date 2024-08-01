# `.\DB-GPT-src\examples\awel\data_analyst_assistant.py`

```py
"""AWEL: Data analyst assistant.

    DB-GPT will automatically load and execute the current file after startup.

    Examples:

        .. code-block:: shell

            # Run this file in your terminal with dev mode.
            # First terminal
            export OPENAI_API_KEY=xxx
            export OPENAI_API_BASE=https://api.openai.com/v1
            python examples/awel/simple_chat_history_example.py


        Code fix command, return no streaming response

        .. code-block:: shell

            # Open a new terminal
            # Second terminal

            DBGPT_SERVER="http://127.0.0.1:5555"
            MODEL="gpt-3.5-turbo"
            # Fist round
            curl -X POST $DBGPT_SERVER/api/v1/awel/trigger/examples/data_analyst/copilot \
            -H "Content-Type: application/json" -d '{
                "command": "dbgpt_awel_data_analyst_code_fix",
                "model": "'"$MODEL"'",
                "stream": false,
                "context": {
                    "conv_uid": "uuid_conv_copilot_1234",
                    "chat_mode": "chat_with_code"
                },
                "messages": "SELECT * FRM orders WHERE order_amount > 500;"
            }'

"""

import logging
import os
from functools import cache
from typing import Any, Dict, List, Optional

from dbgpt._private.pydantic import BaseModel, Field
from dbgpt.core import (
    ChatPromptTemplate,
    HumanPromptTemplate,
    MessagesPlaceholder,
    ModelMessage,
    ModelRequest,
    ModelRequestContext,
    PromptManager,
    PromptTemplate,
    SystemPromptTemplate,
)
from dbgpt.core.awel import (
    DAG,
    BranchJoinOperator,
    HttpTrigger,
    JoinOperator,
    MapOperator,
)
from dbgpt.core.operators import (
    BufferedConversationMapperOperator,
    HistoryDynamicPromptBuilderOperator,
    LLMBranchOperator,
)
from dbgpt.model.operators import (
    LLMOperator,
    OpenAIStreamingOutputOperator,
    StreamingLLMOperator,
)
from dbgpt.serve.conversation.operators import ServePreChatHistoryLoadOperator

logger = logging.getLogger(__name__)

PROMPT_LANG_ZH = "zh"
PROMPT_LANG_EN = "en"

CODE_DEFAULT = "dbgpt_awel_data_analyst_code_default"
CODE_FIX = "dbgpt_awel_data_analyst_code_fix"
CODE_PERF = "dbgpt_awel_data_analyst_code_perf"
CODE_EXPLAIN = "dbgpt_awel_data_analyst_code_explain"
CODE_COMMENT = "dbgpt_awel_data_analyst_code_comment"
CODE_TRANSLATE = "dbgpt_awel_data_analyst_code_translate"

CODE_DEFAULT_TEMPLATE_ZH = """作为一名经验丰富的数据仓库开发者和数据分析师。
你可以根据最佳实践来优化代码, 也可以对代码进行修复, 解释, 添加注释, 以及将代码翻译成其他语言。"""
CODE_DEFAULT_TEMPLATE_EN = """As an experienced data warehouse developer and data analyst.
You can optimize the code according to best practices, or fix, explain, add comments to the code, 
and you can also translate the code into other languages.
"""

CODE_FIX_TEMPLATE_ZH = """作为一名经验丰富的数据仓库开发者和数据分析师，
这里有一段 {language} 代码。请按照最佳实践检查代码，找出并修复所有错误。请给出修复后的代码，并且提供对您所做的每一行更正的逐行解释，请使用和用户相同的语言进行回答。"""
CODE_FIX_TEMPLATE_EN = """As an experienced data warehouse developer and data analyst, 
here is a snippet of code of {language}. Please review the code following best practices to identify and fix all errors. 
Provide the corrected code and include a line-by-line explanation of all the fixes you've made, please use the same language as the user."""

CODE_PERF_TEMPLATE_ZH = """作为一名经验丰富的数据仓库开发者和数据分析师，这里有一段 {language} 代码。
请你按照最佳实践来优化这段代码。请在代码中加入注释点明所做的更改，并解释每项优化的原因，以便提高代码的维护性和性能，请使用和用户相同的语言进行回答。"""
CODE_PERF_TEMPLATE_EN = """As an experienced data warehouse developer and data analyst, 
you are provided with a snippet of code of {language}. Please optimize the code according to best practices. 
Include comments to highlight the changes made and explain the reasons for each optimization for better maintenance and performance, 
please use the same language as the user."""
CODE_EXPLAIN_TEMPLATE_ZH = """作为一名经验丰富的数据仓库开发者和数据分析师，
现在给你的是一份 {language} 代码。请你逐行解释代码的含义，请使用和用户相同的语言进行回答。"""

CODE_EXPLAIN_TEMPLATE_EN = """As an experienced data warehouse developer and data analyst, 
you are provided with a snippet of code of {language}. Please explain the meaning of the code line by line, 
please use the same language as the user."""

CODE_COMMENT_TEMPLATE_ZH = """作为一名经验丰富的数据仓库开发者和数据分析师，现在给你的是一份 {language} 代码。
请你为每一行代码添加注释，解释每个部分的作用，请使用和用户相同的语言进行回答。"""

CODE_COMMENT_TEMPLATE_EN = """As an experienced Data Warehouse Developer and Data Analyst. 
Below is a snippet of code written in {language}. 
Please provide line-by-line comments explaining what each section of the code does, please use the same language as the user."""

CODE_TRANSLATE_TEMPLATE_ZH = """作为一名经验丰富的数据仓库开发者和数据分析师，现在手头有一份用{source_language}语言编写的代码片段。
请你将这段代码准确无误地翻译成{target_language}语言，确保语法和功能在翻译后的代码中得到正确体现，请使用和用户相同的语言进行回答。"""
CODE_TRANSLATE_TEMPLATE_EN = """As an experienced data warehouse developer and data analyst, 
you're presented with a snippet of code written in {source_language}. 
Please translate this code into {target_language} ensuring that the syntax and functionalities are accurately reflected in the translated code, 
please use the same language as the user."""


class ReqContext(BaseModel):
    user_name: Optional[str] = Field(
        None, description="The user name of the model request."
    )

    sys_code: Optional[str] = Field(
        None, description="The system code of the model request."
    )
    conv_uid: Optional[str] = Field(
        None, description="The conversation uid of the model request."
    )
    chat_mode: Optional[str] = Field(
        "chat_with_code", description="The chat mode of the model request."
    )


class TriggerReqBody(BaseModel):
    messages: str = Field(..., description="User input messages")
    command: Optional[str] = Field(
        default=None, description="Command name, None if common chat"
    )
    model: Optional[str] = Field(default="gpt-3.5-turbo", description="Model name")
    # 定义一个可选的布尔型变量 stream，默认值为 False，表示是否返回流数据
    stream: Optional[bool] = Field(default=False, description="Whether return stream")
    # 定义一个可选的字符串变量 language，默认值为 "hive"，表示语言类型
    language: Optional[str] = Field(default="hive", description="Language")
    # 定义一个可选的字符串变量 target_language，默认值为 "hive"，表示目标语言，用于翻译
    target_language: Optional[str] = Field(
        default="hive", description="Target language, use in translate"
    )
    # 定义一个可选的 ReqContext 类型的变量 context，默认值为 None，表示模型请求的上下文信息
    context: Optional[ReqContext] = Field(
        default=None, description="The context of the model request."
    )
# 使用缓存装饰器装饰函数，用于加载或保存提示模板
@cache
def load_or_save_prompt_template(pm: PromptManager):
    # 中文扩展参数字典
    zh_ext_params = {
        "chat_scene": "chat_with_code",
        "sub_chat_scene": "data_analyst",
        "prompt_type": "common",
        "prompt_language": PROMPT_LANG_ZH,  # 使用中文提示语言常量
    }
    # 英文扩展参数字典
    en_ext_params = {
        "chat_scene": "chat_with_code",
        "sub_chat_scene": "data_analyst",
        "prompt_type": "common",
        "prompt_language": PROMPT_LANG_EN,  # 使用英文提示语言常量
    }

    # 查询或保存中文默认代码模板
    pm.query_or_save(
        PromptTemplate.from_template(CODE_DEFAULT_TEMPLATE_ZH),
        prompt_name=CODE_DEFAULT,
        **zh_ext_params,
    )
    # 查询或保存英文默认代码模板
    pm.query_or_save(
        PromptTemplate.from_template(CODE_DEFAULT_TEMPLATE_EN),
        prompt_name=CODE_DEFAULT,
        **en_ext_params,
    )
    # 查询或保存中文修复代码模板
    pm.query_or_save(
        PromptTemplate.from_template(CODE_FIX_TEMPLATE_ZH),
        prompt_name=CODE_FIX,
        **zh_ext_params,
    )
    # 查询或保存英文修复代码模板
    pm.query_or_save(
        PromptTemplate.from_template(CODE_FIX_TEMPLATE_EN),
        prompt_name=CODE_FIX,
        **en_ext_params,
    )
    # 查询或保存中文性能优化代码模板
    pm.query_or_save(
        PromptTemplate.from_template(CODE_PERF_TEMPLATE_ZH),
        prompt_name=CODE_PERF,
        **zh_ext_params,
    )
    # 查询或保存英文性能优化代码模板
    pm.query_or_save(
        PromptTemplate.from_template(CODE_PERF_TEMPLATE_EN),
        prompt_name=CODE_PERF,
        **en_ext_params,
    )
    # 查询或保存中文代码解释模板
    pm.query_or_save(
        PromptTemplate.from_template(CODE_EXPLAIN_TEMPLATE_ZH),
        prompt_name=CODE_EXPLAIN,
        **zh_ext_params,
    )
    # 查询或保存英文代码解释模板
    pm.query_or_save(
        PromptTemplate.from_template(CODE_EXPLAIN_TEMPLATE_EN),
        prompt_name=CODE_EXPLAIN,
        **en_ext_params,
    )
    # 查询或保存中文代码注释模板
    pm.query_or_save(
        PromptTemplate.from_template(CODE_COMMENT_TEMPLATE_ZH),
        prompt_name=CODE_COMMENT,
        **zh_ext_params,
    )
    # 查询或保存英文代码注释模板
    pm.query_or_save(
        PromptTemplate.from_template(CODE_COMMENT_TEMPLATE_EN),
        prompt_name=CODE_COMMENT,
        **en_ext_params,
    )
    # 查询或保存中文代码翻译模板
    pm.query_or_save(
        PromptTemplate.from_template(CODE_TRANSLATE_TEMPLATE_ZH),
        prompt_name=CODE_TRANSLATE,
        **zh_ext_params,
    )
    # 查询或保存英文代码翻译模板
    pm.query_or_save(
        PromptTemplate.from_template(CODE_TRANSLATE_TEMPLATE_EN),
        prompt_name=CODE_TRANSLATE,
        **en_ext_params,
    )


class PromptTemplateBuilderOperator(MapOperator[TriggerReqBody, ChatPromptTemplate]):
    """Build prompt template for chat with code."""

    def __init__(self, **kwargs):
        # 调用父类构造函数初始化操作符
        super().__init__(**kwargs)
        # 初始化默认的提示管理器对象
        self._default_prompt_manager = PromptManager()
    async def map(self, input_value: TriggerReqBody) -> ChatPromptTemplate:
        # 导入必要的模块和变量
        from dbgpt.serve.prompt.serve import SERVE_APP_NAME as PROMPT_SERVE_APP_NAME
        from dbgpt.serve.prompt.serve import Serve as PromptServe

        # 获取系统应用的指定组件（PromptServe）
        prompt_serve = self.system_app.get_component(
            PROMPT_SERVE_APP_NAME, PromptServe, default_component=None
        )

        # 如果成功获取到 PromptServe 实例，则使用其 prompt_manager
        if prompt_serve:
            pm = prompt_serve.prompt_manager
        else:
            # 否则使用默认的 prompt_manager
            pm = self._default_prompt_manager

        # 载入或保存 prompt 模板到 prompt_manager 中
        load_or_save_prompt_template(pm)

        # 获取用户当前语言设置，默认为英语
        user_language = self.system_app.config.get_current_lang(default="en")

        # 如果没有指定命令，则返回默认的聊天提示模板
        if not input_value.command:
            # 获取默认的聊天提示模板列表
            default_prompt_list = pm.prefer_query(
                CODE_DEFAULT, prefer_prompt_language=user_language
            )
            # 获取第一个默认模板的具体内容作为默认提示模板
            default_prompt_template = (
                default_prompt_list[0].to_prompt_template().template
            )
            # 构建聊天提示对象，不包括系统提示
            prompt = ChatPromptTemplate(
                messages=[
                    SystemPromptTemplate.from_template(default_prompt_template),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanPromptTemplate.from_template("{user_input}"),
                ]
            )
            return prompt

        # 如果有指定命令，则根据命令从 prompt_manager 中查询对应的提示模板
        prompt_list = pm.prefer_query(
            input_value.command, prefer_prompt_language=user_language
        )

        # 如果找不到对应的提示模板，则记录错误并抛出异常
        if not prompt_list:
            error_msg = f"Prompt not found for command {input_value.command}, user_language: {user_language}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 将找到的第一个提示模板转换为聊天提示模板格式并返回
        prompt_template = prompt_list[0].to_prompt_template()
        return ChatPromptTemplate(
            messages=[
                SystemPromptTemplate.from_template(prompt_template.template),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanPromptTemplate.from_template("{user_input}"),
            ]
        )
def parse_prompt_args(req: TriggerReqBody) -> Dict[str, Any]:
    # 构造初始的提示参数字典，使用请求中的用户输入作为 "user_input"
    prompt_args = {"user_input": req.messages}
    # 如果请求中没有指定命令，则直接返回初始的提示参数字典
    if not req.command:
        return prompt_args
    # 如果请求的命令是 CODE_TRANSLATE，则添加源语言和目标语言到提示参数字典中
    if req.command == CODE_TRANSLATE:
        prompt_args["source_language"] = req.language
        prompt_args["target_language"] = req.target_language
    else:
        # 否则，添加语言到提示参数字典中
        prompt_args["language"] = req.language
    # 返回构造好的提示参数字典
    return prompt_args


async def build_model_request(
    messages: List[ModelMessage], req_body: TriggerReqBody
) -> ModelRequest:
    # 使用 ModelRequest 类的 build_request 方法构建模型请求对象
    return ModelRequest.build_request(
        model=req_body.model,
        messages=messages,
        context=req_body.context,
        stream=req_body.stream,
    )


with DAG("dbgpt_awel_data_analyst_assistant") as dag:
    # 创建 HTTP 触发器实例，设置路径、请求体类型及请求方法
    trigger = HttpTrigger(
        "/examples/data_analyst/copilot",
        request_body=TriggerReqBody,
        methods="POST",
        streaming_predict_func=lambda x: x.stream,
    )

    prompt_template_load_task = PromptTemplateBuilderOperator()

    # 加载并存储聊天历史记录
    chat_history_load_task = ServePreChatHistoryLoadOperator()
    keep_start_rounds = int(os.getenv("DBGPT_AWEL_DATA_ANALYST_KEEP_START_ROUNDS", 0))
    keep_end_rounds = int(os.getenv("DBGPT_AWEL_DATA_ANALYST_KEEP_END_ROUNDS", 5))
    # 历史记录转换任务，保留指定的起始和结束轮次的消息
    history_transform_task = BufferedConversationMapperOperator(
        keep_start_rounds=keep_start_rounds, keep_end_rounds=keep_end_rounds
    )
    history_prompt_build_task = HistoryDynamicPromptBuilderOperator(
        history_key="chat_history"
    )

    model_request_build_task = JoinOperator(build_model_request)

    # 使用 BaseLLMOperator 生成响应
    llm_task = LLMOperator(task_name="llm_task")
    streaming_llm_task = StreamingLLMOperator(task_name="streaming_llm_task")
    branch_task = LLMBranchOperator(
        stream_task_name="streaming_llm_task", no_stream_task_name="llm_task"
    )
    model_parse_task = MapOperator(lambda out: out.to_dict())
    openai_format_stream_task = OpenAIStreamingOutputOperator()
    result_join_task = BranchJoinOperator()
    
    # 触发器连接到模板加载任务，再连接到历史动态提示构建任务
    trigger >> prompt_template_load_task >> history_prompt_build_task

    # 触发器通过 MapOperator 将请求转换为 ModelRequestContext 对象，然后加载聊天历史记录，
    # 经历史记录转换任务处理后，连接到历史动态提示构建任务
    (
        trigger
        >> MapOperator(
            lambda req: ModelRequestContext(
                conv_uid=req.context.conv_uid,
                stream=req.stream,
                user_name=req.context.user_name,
                sys_code=req.context.sys_code,
                chat_mode=req.context.chat_mode,
            )
        )
        >> chat_history_load_task
        >> history_transform_task
        >> history_prompt_build_task
    )

    # 触发器通过 MapOperator 调用 parse_prompt_args 函数处理请求参数，然后连接到历史动态提示构建任务
    trigger >> MapOperator(parse_prompt_args) >> history_prompt_build_task

    # 历史动态提示构建任务连接到模型请求构建任务
    history_prompt_build_task >> model_request_build_task

    # 触发器直接连接到模型请求构建任务
    trigger >> model_request_build_task

    # 模型请求构建任务连接到分支任务
    model_request_build_task >> branch_task

    # 分支任务中的无流式响应分支
    # 将任务分支到llm_task，然后到model_parse_task，最后到result_join_task
    (branch_task >> llm_task >> model_parse_task >> result_join_task)
    # 分支到streaming_llm_task，然后到openai_format_stream_task，最后到result_join_task
    # 处理流式响应的分支
    (branch_task >> streaming_llm_task >> openai_format_stream_task >> result_join_task)
# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 如果 DAG 的第一个叶子节点处于开发模式
    if dag.leaf_nodes[0].dev_mode:
        # 导入设置开发环境的函数
        from dbgpt.core.awel import setup_dev_environment
        # 调用设置开发环境的函数，传入 DAG 列表
        setup_dev_environment([dag])
    else:
        # 如果不处于开发模式，则不执行任何操作
        pass
```
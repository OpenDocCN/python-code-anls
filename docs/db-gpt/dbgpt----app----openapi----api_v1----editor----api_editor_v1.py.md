# `.\DB-GPT-src\dbgpt\app\openapi\api_v1\editor\api_editor_v1.py`

```py
# 导入必要的模块和库
import json
import logging
import time
from typing import Dict, List

from fastapi import APIRouter, Body, Depends  # 导入 FastAPI 的相关组件

from dbgpt._private.config import Config  # 导入 Config 类
from dbgpt.app.openapi.api_v1.editor.service import EditorService  # 导入 EditorService 类
from dbgpt.app.openapi.api_v1.editor.sql_editor import (  # 导入相关类
    ChartRunData,
    DataNode,
    SqlRunData,
)
from dbgpt.app.openapi.api_view_model import Result  # 导入 Result 类
from dbgpt.app.openapi.editor_view_model import (  # 导入编辑器视图模型相关类
    ChartDetail,
    ChartList,
    ChatChartEditContext,
    ChatDbRounds,
    ChatSqlEditContext,
    DbTable,
)
from dbgpt.app.scene import ChatFactory  # 导入 ChatFactory 类
from dbgpt.app.scene.chat_dashboard.data_loader import DashboardDataLoader  # 导入 DashboardDataLoader 类
from dbgpt.serve.conversation.serve import Serve as ConversationServe  # 导入 Serve 类，并重命名为 ConversationServe
from dbgpt.util.date_utils import convert_datetime_in_row  # 导入日期转换函数

from ._chat_history.chat_hisotry_factory import ChatHistory  # 导入本地聊天历史记录相关类

router = APIRouter()  # 创建 APIRouter 实例
CFG = Config()  # 创建 Config 实例
CHAT_FACTORY = ChatFactory()  # 创建 ChatFactory 实例

logger = logging.getLogger(__name__)  # 获取当前模块的 logger 对象


def get_conversation_serve() -> ConversationServe:
    # 返回配置系统应用的 ConversationServe 单例实例
    return ConversationServe.get_instance(CFG.SYSTEM_APP)


def get_edit_service() -> EditorService:
    # 返回配置系统应用的 EditorService 单例实例
    return EditorService.get_instance(CFG.SYSTEM_APP)


@router.get("/v1/editor/db/tables", response_model=Result[DataNode])
async def get_editor_tables(
    db_name: str, page_index: int, page_size: int, search_str: str = ""
):
    # 处理获取数据库表信息的 GET 请求，返回数据库表信息的数据结构
    logger.info(f"get_editor_tables:{db_name},{page_index},{page_size},{search_str}")
    db_conn = CFG.local_db_manager.get_connector(db_name)  # 获取指定数据库的连接器
    tables = db_conn.get_table_names()  # 获取数据库中的表名列表
    db_node: DataNode = DataNode(title=db_name, key=db_name, type="db")  # 创建数据库节点对象
    for table in tables:
        table_node: DataNode = DataNode(title=table, key=table, type="table")  # 创建表节点对象
        db_node.children.append(table_node)  # 将表节点添加到数据库节点的子节点列表中
        fields = db_conn.get_fields(table)  # 获取表的字段信息
        for field in fields:
            table_node.children.append(  # 将字段信息作为子节点添加到表节点的子节点列表中
                DataNode(
                    title=field[0],
                    key=field[0],
                    type=field[1],
                    default_value=field[2],
                    can_null=field[3] or "YES",
                    comment=str(field[-1]),
                )
            )

    return Result.succ(db_node)  # 返回成功的结果对象，包含数据库节点信息


@router.get("/v1/editor/sql/rounds", response_model=Result[List[ChatDbRounds]])
async def get_editor_sql_rounds(
    con_uid: str, editor_service: EditorService = Depends(get_edit_service)
):
    # 处理获取 SQL 执行轮次信息的 GET 请求，返回 SQL 执行轮次信息的数据结构
    logger.info(f"get_editor_sql_rounds:{con_uid}")
    return Result.succ(editor_service.get_editor_sql_rounds(con_uid))  # 返回成功的结果对象，包含 SQL 执行轮次信息


@router.get("/v1/editor/sql", response_model=Result[dict | list])
async def get_editor_sql(
    con_uid: str, round: int, editor_service: EditorService = Depends(get_edit_service)
):
    # 处理获取特定 SQL 执行信息的 GET 请求，返回特定 SQL 执行信息的数据结构或错误信息
    logger.info(f"get_editor_sql:{con_uid},{round}")
    context = editor_service.get_editor_sql_by_round(con_uid, round)  # 获取指定 SQL 执行轮次的详细信息
    if context:
        return Result.succ(context)  # 如果获取到执行信息，则返回成功的结果对象
    return Result.failed(msg="not have sql!")  # 如果未获取到执行信息，则返回包含错误信息的失败结果对象


@router.post("/v1/editor/sql/run", response_model=Result[SqlRunData])
# 异步函数，接收一个包含参数的字典作为请求体
async def editor_sql_run(run_param: dict = Body()):
    # 记录日志，打印编辑器SQL运行参数
    logger.info(f"editor_sql_run:{run_param}")
    # 从参数字典中获取数据库名和SQL语句
    db_name = run_param["db_name"]
    sql = run_param["sql"]
    # 如果数据库名和SQL语句均不存在，则返回参数错误的结果
    if not db_name and not sql:
        return Result.failed(msg="SQL run param error！")
    # 获取指定数据库的连接器
    conn = CFG.local_db_manager.get_connector(db_name)

    try:
        # 记录SQL执行开始时间
        start_time = time.time() * 1000
        # 执行SQL查询，获取列名和查询结果
        colunms, sql_result = conn.query_ex(sql)
        # 将查询结果转换为元组列表
        sql_result = [tuple(x) for x in sql_result]
        # 记录SQL执行结束时间
        end_time = time.time() * 1000
        # 构造SQL执行结果对象
        sql_run_data: SqlRunData = SqlRunData(
            result_info="",
            run_cost=int((end_time - start_time) / 1000),
            colunms=colunms,
            values=sql_result,
        )
        # 返回成功的结果对象
        return Result.succ(sql_run_data)
    except Exception as e:
        # 记录SQL执行过程中的异常信息
        logging.error("editor_sql_run exception!" + str(e))
        # 返回失败的结果对象，并记录异常信息
        return Result.succ(
            SqlRunData(result_info=str(e), run_cost=0, colunms=[], values=[])
        )


# 定义路由，处理SQL编辑器提交请求
@router.post("/v1/sql/editor/submit")
async def sql_editor_submit(
    sql_edit_context: ChatSqlEditContext = Body(),
    editor_service: EditorService = Depends(get_edit_service),
):
    # 记录日志，打印SQL编辑器提交上下文的属性字典
    logger.info(f"sql_editor_submit:{sql_edit_context.__dict__}")

    # 获取指定数据库的连接器
    conn = CFG.local_db_manager.get_connector(sql_edit_context.db_name)
    try:
        # 调用编辑服务的方法，执行SQL编辑器的提交和保存操作
        editor_service.sql_editor_submit_and_save(sql_edit_context, conn)
        # 返回成功的结果对象
        return Result.succ(None)
    except Exception as e:
        # 记录编辑SQL过程中的异常信息
        logger.error(f"edit sql exception!{str(e)}")
        # 返回失败的结果对象，并记录异常信息
        return Result.failed(msg=f"Edit sql exception!{str(e)}")


# 定义路由，处理获取编辑器图表列表请求
@router.get("/v1/editor/chart/list", response_model=Result[ChartList])
async def get_editor_chart_list(
    con_uid: str,
    editor_service: EditorService = Depends(get_edit_service),
):
    # 记录日志，打印获取编辑器图表列表的请求UID
    logger.info(
        f"get_editor_sql_rounds:{con_uid}",
    )
    # 调用编辑服务的方法，获取编辑器的图表列表
    chart_list = editor_service.get_editor_chart_list(con_uid)
    # 如果图表列表存在，则返回成功的结果对象
    if chart_list:
        return Result.succ(chart_list)
    # 如果图表列表不存在，则返回失败的结果对象，并提示没有图表
    return Result.failed(msg="Not have charts!")


# 定义路由，处理获取编辑器图表详情请求
@router.post("/v1/editor/chart/info", response_model=Result[ChartDetail])
async def get_editor_chart_info(
    param: dict = Body(), editor_service: EditorService = Depends(get_edit_service)
):
    # 记录日志，打印获取编辑器图表详情的参数字典
    logger.info(f"get_editor_chart_info:{param}")
    # 从参数字典中获取会话UID和图表标题
    conv_uid = param["con_uid"]
    chart_title = param["chart_title"]
    # 调用编辑服务的方法，获取编辑器的图表详情
    return editor_service.get_editor_chart_info(conv_uid, chart_title, CFG)


# 定义路由，处理运行编辑器图表请求
@router.post("/v1/editor/chart/run", response_model=Result[ChartRunData])
async def editor_chart_run(run_param: dict = Body()):
    # 记录日志，打印运行编辑器图表的参数字典
    logger.info(f"editor_chart_run:{run_param}")
    # 从参数字典中获取数据库名、SQL语句和图表类型
    db_name = run_param["db_name"]
    sql = run_param["sql"]
    chart_type = run_param["chart_type"]
    # 如果数据库名和SQL语句均不存在，则返回参数错误的结果
    if not db_name and not sql:
        return Result.failed("SQL run param error！")
    try:
        # 创建 DashboardDataLoader 的实例，用于加载仪表盘数据
        dashboard_data_loader: DashboardDataLoader = DashboardDataLoader()
        # 获取本地数据库管理器的连接器，并连接到指定数据库
        db_conn = CFG.local_db_manager.get_connector(db_name)
        # 使用数据库连接执行给定的 SQL 查询，并获取返回的列名和查询结果
        colunms, sql_result = db_conn.query_ex(sql)
        # 使用 DashboardDataLoader 实例处理查询结果，获取图表数据的字段名和值
        field_names, chart_values = dashboard_data_loader.get_chart_values_by_data(
            colunms, sql_result, sql
        )
        # 记录开始时间，用于计算 SQL 执行时间
        start_time = time.time() * 1000
        # 将查询结果中的日期时间字段转换为本地时区的时间格式
        sql_result = [convert_datetime_in_row(row) for row in sql_result]
        # 记录结束时间，计算 SQL 执行总耗时
        end_time = time.time() * 1000
        # 创建 SqlRunData 的实例，包含 SQL 执行的详细信息
        sql_run_data: SqlRunData = SqlRunData(
            result_info="",
            run_cost=int((end_time - start_time) / 1000),
            colunms=colunms,
            values=sql_result,
        )
        # 返回成功的结果对象，包含图表运行数据和图表类型
        return Result.succ(
            ChartRunData(
                sql_data=sql_run_data, chart_values=chart_values, chart_type=chart_type
            )
        )
    except Exception as e:
        # 如果发生异常，创建包含错误信息的 SqlRunData 实例，运行耗时为 0
        sql_result = SqlRunData(result_info=str(e), run_cost=0, colunms=[], values=[])
        # 返回成功的结果对象，包含仅错误信息的图表运行数据和图表类型
        return Result.succ(
            ChartRunData(sql_data=sql_result, chart_values=[], chart_type=chart_type)
        )
# 定义一个路由处理函数，用于处理POST请求，路径为"/v1/chart/editor/submit"，
# 并指定响应模型为Result[bool]，表示返回一个布尔类型的结果
async def chart_editor_submit(chart_edit_context: ChatChartEditContext = Body()):
    # 记录日志，输出提交的SQL编辑内容的属性字典
    logger.info(f"sql_editor_submit:{chart_edit_context.__dict__}")

    # 创建ChatHistory类的实例，用于管理聊天历史记录
    chat_history_fac = ChatHistory()

    # 获取与给定会话UID相关的聊天历史存储实例
    history_mem = chat_history_fac.get_store_instance(chart_edit_context.con_uid)

    # 获取历史消息的列表，每条消息表示为一个字典
    history_messages: List[Dict] = history_mem.get_messages()
    # 如果存在历史消息
    if history_messages:
        # 创建仪表板数据加载器对象
        dashboard_data_loader: DashboardDataLoader = DashboardDataLoader()
        # 获取本地数据库连接器对象
        db_conn = CFG.local_db_manager.get_connector(chart_edit_context.db_name)

        # 找到最新的编辑轮次，根据聊天顺序（chat_order）进行比较
        edit_round = max(history_messages, key=lambda x: x["chat_order"])
        
        # 如果存在编辑轮次
        if edit_round:
            try:
                # 遍历编辑轮次中的消息元素
                for element in edit_round["messages"]:
                    # 如果消息类型为视图
                    if element["type"] == "view":
                        # 解析视图数据内容为字典
                        view_data: dict = json.loads(element["data"]["content"])
                        # 获取视图中的图表列表
                        charts: List = view_data.get("charts")
                        # 查找特定名称的图表对象
                        find_chart = list(
                            filter(
                                lambda x: x["chart_name"]
                                == chart_edit_context.chart_title,
                                charts,
                            )
                        )[0]
                        # 如果有新的图表类型，则更新图表类型
                        if chart_edit_context.new_chart_type:
                            find_chart["chart_type"] = chart_edit_context.new_chart_type
                        # 如果有新的注释信息，则更新图表描述
                        if chart_edit_context.new_comment:
                            find_chart["chart_desc"] = chart_edit_context.new_comment

                        # 使用数据加载器获取图表的字段名和值
                        field_names, chart_values = dashboard_data_loader.get_chart_values_by_conn(
                            db_conn, chart_edit_context.new_sql
                        )
                        # 更新图表的 SQL 查询语句和值
                        find_chart["chart_sql"] = chart_edit_context.new_sql
                        find_chart["values"] = [value.dict() for value in chart_values]
                        find_chart["column_name"] = field_names

                        # 更新消息元素的内容为更新后的视图数据
                        element["data"]["content"] = json.dumps(
                            view_data, ensure_ascii=False
                        )
                    
                    # 如果消息类型为 AI
                    if element["type"] == "ai":
                        # 解析 AI 响应数据内容为字典
                        ai_resp: dict = json.loads(element["data"]["content"])
                        # 查找特定标题的 AI 响应项目
                        edit_item = list(
                            filter(
                                lambda x: x["title"] == chart_edit_context.chart_title,
                                ai_resp,
                            )
                        )[0]

                        # 更新 AI 响应项目的 SQL 查询语句、展示类型和备注信息
                        edit_item["sql"] = chart_edit_context.new_sql
                        edit_item["showcase"] = chart_edit_context.new_chart_type
                        edit_item["thoughts"] = chart_edit_context.new_comment

                        # 更新消息元素的内容为更新后的 AI 响应数据
                        element["data"]["content"] = json.dumps(
                            ai_resp, ensure_ascii=False
                        )

            # 捕获可能发生的异常并记录日志
            except Exception as e:
                logger.error(f"edit chart exception!{str(e)}", e)
                # 返回编辑图表异常的失败结果
                return Result.failed(msg=f"Edit chart exception!{str(e)}")

            # 更新历史消息中的编辑内容
            history_mem.update(history_messages)
            # 返回编辑成功的结果
            return Result.succ(None)

    # 如果没有历史消息，则返回编辑失败的结果
    return Result.failed(msg="Edit Failed!")
```
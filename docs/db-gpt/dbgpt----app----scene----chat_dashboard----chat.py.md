# `.\DB-GPT-src\dbgpt\app\scene\chat_dashboard\chat.py`

```py
import json  # 导入处理 JSON 格式数据的模块
import os  # 导入操作系统功能的模块
import uuid  # 导入生成 UUID 的模块
from typing import Dict, List  # 导入类型提示相关的模块

from dbgpt._private.config import Config  # 导入 Config 类，用于配置管理
from dbgpt.app.scene import BaseChat, ChatScene  # 导入基础聊天和聊天场景类
from dbgpt.app.scene.chat_dashboard.data_loader import DashboardDataLoader  # 导入数据加载类
from dbgpt.app.scene.chat_dashboard.data_preparation.report_schma import (  # 导入数据准备相关类
    ChartData,
    ReportData,
)
from dbgpt.util.executor_utils import blocking_func_to_async  # 导入用于将同步函数转为异步的工具函数
from dbgpt.util.tracer import trace  # 导入跟踪函数

CFG = Config()  # 创建 Config 实例

class ChatDashboard(BaseChat):
    chat_scene: str = ChatScene.ChatDashboard.value()  # 设置聊天场景类型为 ChatDashboard
    report_name: str  # 报告名称属性

    """Chat Dashboard to generate dashboard chart"""

    def __init__(self, chat_param: Dict):
        """Chat Dashboard Module Initialization
        
        Args:
           - chat_param: Dict
            - chat_session_id: (str) chat session_id
            - current_user_input: (str) current user input
            - model_name:(str) llm model name
            - select_param:(str) dbname
        """
        self.db_name = chat_param["select_param"]  # 从 chat_param 中获取数据库名称
        chat_param["chat_mode"] = ChatScene.ChatDashboard  # 设置 chat_mode 属性为 ChatDashboard
        super().__init__(chat_param=chat_param)  # 调用父类初始化方法

        if not self.db_name:  # 如果没有指定数据库名称
            raise ValueError(f"{ChatScene.ChatDashboard.value} mode should choose db!")

        self.db_name = self.db_name  # 设置实例的数据库名称属性
        self.report_name = chat_param.get("report_name", "report")  # 获取报告名称，如果不存在则默认为 "report"

        self.database = CFG.local_db_manager.get_connector(self.db_name)  # 获取数据库连接器实例

        self.top_k: int = 5  # 设置默认的 top_k 值为 5
        self.dashboard_template = self.__load_dashboard_template(self.report_name)  # 加载指定报告的仪表板模板

    def __load_dashboard_template(self, template_name):
        current_dir = os.getcwd()  # 获取当前工作目录
        print(current_dir)  # 打印当前工作目录路径

        current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录的绝对路径
        with open(f"{current_dir}/template/{template_name}/dashboard.json", "r") as f:
            data = f.read()  # 读取指定路径下的 JSON 文件内容
        return json.loads(data)  # 将 JSON 数据解析为 Python 字典并返回

    @trace()  # 使用跟踪装饰器
    async def generate_input_values(self) -> Dict:
        try:
            from dbgpt.rag.summary.db_summary_client import DBSummaryClient  # 导入数据库摘要客户端
        except ImportError:
            raise ValueError("Could not import DBSummaryClient. ")  # 如果导入失败则抛出异常

        client = DBSummaryClient(system_app=CFG.SYSTEM_APP)  # 创建数据库摘要客户端实例
        try:
            table_infos = await blocking_func_to_async(
                self._executor,
                client.get_db_summary,
                self.db_name,
                self.current_user_input,
                self.top_k,
            )  # 调用异步函数获取数据库摘要信息
            print("dashboard vector find tables:{}", table_infos)  # 打印找到的数据表信息
        except Exception as e:
            print("db summary find error!" + str(e))  # 打印数据库摘要查找错误信息

        input_values = {
            "input": self.current_user_input,  # 当前用户输入
            "dialect": self.database.dialect,  # 数据库方言
            "table_info": self.database.table_simple_info(),  # 数据库表简要信息
            "supported_chat_type": self.dashboard_template["supported_chart_type"]
            # "table_info": client.get_similar_tables(dbname=self.db_name, query=self.current_user_input, topk=self.top_k)
        }  # 构建输入值字典

        return input_values  # 返回输入值字典
    def do_action(self, prompt_response):
        """
        # 处理操作函数，接收一个提示响应作为参数

        chart_datas: List[ChartData] = []  # 初始化图表数据列表

        dashboard_data_loader = DashboardDataLoader()  # 创建仪表板数据加载器实例

        # 遍历提示响应中的每个图表项
        for chart_item in prompt_response:
            try:
                # 使用仪表板数据加载器获取图表数据的字段名和数值
                field_names, values = dashboard_data_loader.get_chart_values_by_conn(
                    self.database, chart_item.sql
                )

                # 将获取的图表数据组装成ChartData对象并添加到chart_datas列表中
                chart_datas.append(
                    ChartData(
                        chart_uid=str(uuid.uuid1()),  # 生成唯一的图表UID
                        chart_name=chart_item.title,  # 设置图表名称
                        chart_type=chart_item.showcase,  # 设置图表类型
                        chart_desc=chart_item.thoughts,  # 设置图表描述
                        chart_sql=chart_item.sql,  # 保存图表相关的SQL查询语句
                        column_name=field_names,  # 设置图表数据的字段名
                        values=values,  # 设置图表数据的数值
                    )
                )
            except Exception as e:
                # TODO 处理异常情况的修复流程
                print(str(e))  # 打印异常信息

        # 返回一个ReportData对象，包含对话会话ID、报告名称、图表数据等信息
        return ReportData(
            conv_uid=self.chat_session_id,
            template_name=self.report_name,
            template_introduce=None,  # 报告介绍暂未指定
            charts=chart_datas,  # 将收集的图表数据作为报告的一部分返回
        )
```
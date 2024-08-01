# `.\DB-GPT-src\dbgpt\core\awel\__init__.py`

```py
# 导入日志模块
import logging
# 导入类型提示相关模块
from typing import List, Optional

# 导入系统应用相关组件
from dbgpt.component import SystemApp

# 导入 DAG 相关模块
from .dag.base import DAG, DAGContext, DAGVar
# 导入操作符基类及工作流运行器
from .operators.base import BaseOperator, WorkflowRunner
# 导入常见操作符及分支操作相关模块
from .operators.common_operator import (
    BranchFunc,
    BranchJoinOperator,
    BranchOperator,
    BranchTaskType,
    InputOperator,
    JoinOperator,
    MapOperator,
    ReduceStreamOperator,
    TriggerOperator,
)
# 导入流操作符相关模块
from .operators.stream_operator import (
    StreamifyAbsOperator,
    TransformStreamAbsOperator,
    UnstreamifyAbsOperator,
)
# 导入本地运行器相关模块
from .runner.local_runner import DefaultWorkflowRunner
# 导入任务相关模块
from .task.base import (
    InputContext,
    InputSource,
    TaskContext,
    TaskOutput,
    TaskState,
    is_empty_data,
)
# 导入任务实现相关模块
from .task.task_impl import (
    BaseInputSource,
    DefaultInputContext,
    DefaultTaskContext,
    SimpleCallDataInputSource,
    SimpleInputSource,
    SimpleStreamTaskOutput,
    SimpleTaskOutput,
    _is_async_iterator,
)
# 导入触发器基类及 HTTP 触发器相关模块
from .trigger.base import Trigger
from .trigger.http_trigger import (
    CommonLLMHttpRequestBody,
    CommonLLMHttpResponseBody,
    HttpTrigger,
)
# 导入迭代器触发器相关模块
from .trigger.iterator_trigger import IteratorTrigger

# 尝试导入可选的 RequestHttpTrigger 模块
_request_http_trigger_available = False
try:
    from .trigger.ext_http_trigger import RequestHttpTrigger  # noqa: F401
    _request_http_trigger_available = True
except ImportError:
    pass

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)

# 导出的模块列表
__all__ = [
    "initialize_awel",
    "DAGContext",
    "DAG",
    "DAGVar",
    "BaseOperator",
    "JoinOperator",
    "ReduceStreamOperator",
    "TriggerOperator",
    "MapOperator",
    "BranchJoinOperator",
    "BranchOperator",
    "InputOperator",
    "BranchFunc",
    "BranchTaskType",
    "WorkflowRunner",
    "TaskState",
    "is_empty_data",
    "TaskOutput",
    "TaskContext",
    "InputContext",
    "InputSource",
    "DefaultWorkflowRunner",
    "SimpleInputSource",
    "BaseInputSource",
    "SimpleCallDataInputSource",
    "DefaultTaskContext",
    "DefaultInputContext",
    "SimpleTaskOutput",
    "SimpleStreamTaskOutput",
    "StreamifyAbsOperator",
    "UnstreamifyAbsOperator",
    "TransformStreamAbsOperator",
    "Trigger",
    "HttpTrigger",
    "IteratorTrigger",
    "CommonLLMHttpResponseBody",
    "CommonLLMHttpRequestBody",
    "setup_dev_environment",
    "_is_async_iterator",
]

# 如果 RequestHttpTrigger 可用，则添加到 __all__ 列表中
if _request_http_trigger_available:
    __all__.append("RequestHttpTrigger")

# 初始化 AWEL 系统
def initialize_awel(system_app: SystemApp, dag_dirs: List[str]):
    """Initialize AWEL."""
    # 导入 DAG 管理器
    from .dag.dag_manager import DAGManager
    # 导入初始化运行器函数
    from .operators.base import initialize_runner
    # 导入默认触发管理器类
    from .trigger.trigger_manager import DefaultTriggerManager
    
    # 设置当前系统应用程序到 DAG 变量
    DAGVar.set_current_system_app(system_app)
    
    # 注册默认触发管理器到系统应用程序
    system_app.register(DefaultTriggerManager)
    
    # 创建并初始化 DAG 管理器对象，使用给定的 DAG 目录列表
    dag_manager = DAGManager(system_app, dag_dirs)
    
    # 将 DAG 管理器实例注册到系统应用程序中
    system_app.register_instance(dag_manager)
    
    # 初始化默认工作流运行器并进行初始化
    initialize_runner(DefaultWorkflowRunner())
def setup_dev_environment(
    dags: List[DAG],
    host: str = "127.0.0.1",
    port: int = 5555,
    logging_level: Optional[str] = None,
    logger_filename: Optional[str] = None,
    show_dag_graph: Optional[bool] = True,
) -> None:
    """Run AWEL in development environment.

    Just using in development environment, not production environment.

    Args:
        dags (List[DAG]): The DAGs.
        host (Optional[str], optional): The host. Defaults to "127.0.0.1"
        port (Optional[int], optional): The port. Defaults to 5555.
        logging_level (Optional[str], optional): The logging level. Defaults to None.
        logger_filename (Optional[str], optional): The logger filename.
            Defaults to None.
        show_dag_graph (Optional[bool], optional): Whether show the DAG graph.
            Defaults to True. If True, the DAG graph will be saved to a file and open
            it automatically.
    """
    from dbgpt.component import SystemApp
    from dbgpt.util.utils import setup_logging

    from .trigger.trigger_manager import DefaultTriggerManager

    # 如果没有指定 logger_filename，则设置默认值为 "dbgpt_awel_dev.log"
    if not logger_filename:
        logger_filename = "dbgpt_awel_dev.log"
    # 设置日志记录
    setup_logging("dbgpt", logging_level=logging_level, logger_filename=logger_filename)

    # 检查是否有 HTTP 触发器
    start_http = _check_has_http_trigger(dags)
    if start_http:
        from dbgpt.util.fastapi import create_app

        # 创建 FastAPI 应用
        app = create_app()
    else:
        app = None
    # 创建系统应用
    system_app = SystemApp(app)
    # 设置当前系统应用
    DAGVar.set_current_system_app(system_app)
    # 创建默认触发器管理器
    trigger_manager = DefaultTriggerManager()
    # 注册触发器管理器实例到系统应用
    system_app.register_instance(trigger_manager)

    # 遍历所有 DAG
    for dag in dags:
        # 如果需要显示 DAG 图
        if show_dag_graph:
            try:
                # 可视化 DAG 并保存图像文件
                dag_graph_file = dag.visualize_dag()
                if dag_graph_file:
                    logger.info(f"Visualize DAG {str(dag)} to {dag_graph_file}")
            except Exception as e:
                logger.warning(
                    f"Visualize DAG {str(dag)} failed: {e}, if your system has no "
                    f"graphviz, you can install it by `pip install graphviz` or "  # noqa
                    f"`sudo apt install graphviz`"
                )
        # 遍历 DAG 的触发器节点
        for trigger in dag.trigger_nodes:
            # 注册触发器到触发器管理器
            trigger_manager.register_trigger(trigger, system_app)
    # 触发器注册完成后的操作
    trigger_manager.after_register()
    # 如果需要启动 HTTP 服务且触发器管理器保持运行状态且应用存在
    if start_http and trigger_manager.keep_running() and app:
        import uvicorn

        # 应该保持运行
        uvicorn.run(app, host=host, port=port)


def _check_has_http_trigger(dags: List[DAG]) -> bool:
    """Check whether has http trigger.

    Args:
        dags (List[DAG]): The dags.

    Returns:
        bool: Whether has http trigger.
    """
    # 遍历所有 DAG
    for dag in dags:
        # 遍历 DAG 的触发器节点
        for trigger in dag.trigger_nodes:
            # 如果触发器是 HttpTrigger 类型，则返回 True
            if isinstance(trigger, HttpTrigger):
                return True
    # 如果没有找到 HttpTrigger 类型的触发器，则返回 False
    return False
```
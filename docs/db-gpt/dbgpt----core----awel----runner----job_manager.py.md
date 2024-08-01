# `.\DB-GPT-src\dbgpt\core\awel\runner\job_manager.py`

```py
"""Job manager for DAG."""
import asyncio  # 引入异步IO库
import logging  # 引入日志库
import uuid  # 引入UUID库
from typing import Dict, List, Optional, cast  # 引入类型提示相关模块

from ..dag.base import DAGLifecycle  # 导入DAG生命周期基类
from ..operators.base import CALL_DATA, BaseOperator  # 导入调用数据和基础操作符

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class JobManager(DAGLifecycle):
    """Job manager for DAG.

    This class is used to manage the DAG lifecycle.
    """

    def __init__(
        self,
        root_nodes: List[BaseOperator],
        all_nodes: List[BaseOperator],
        end_node: BaseOperator,
        id2call_data: Dict[str, Optional[Dict]],
        node_name_to_ids: Dict[str, str],
    ) -> None:
        """Create a job manager.

        Args:
            root_nodes (List[BaseOperator]): The root nodes of the DAG.
            all_nodes (List[BaseOperator]): All nodes of the DAG.
            end_node (BaseOperator): The end node of the DAG.
            id2call_data (Dict[str, Optional[Dict]]): The call data of each node.
            node_name_to_ids (Dict[str, str]): The node name to node id mapping.
        """
        self._root_nodes = root_nodes  # 设置根节点列表
        self._all_nodes = all_nodes  # 设置所有节点列表
        self._end_node = end_node  # 设置结束节点
        self._id2node_data = id2call_data  # 设置节点ID到调用数据的映射
        self._node_name_to_ids = node_name_to_ids  # 设置节点名到节点ID的映射

    @staticmethod
    def build_from_end_node(
        end_node: BaseOperator, call_data: Optional[CALL_DATA] = None
    ) -> "JobManager":
        """Build a job manager from the end node.

        This will get all upstream nodes from the end node, and build a job manager.

        Args:
            end_node (BaseOperator): The end node of the DAG.
            call_data (Optional[CALL_DATA], optional): The call data of the end node.
                Defaults to None.
        """
        nodes = _build_from_end_node(end_node)  # 调用内部函数获取从结束节点开始的所有节点
        root_nodes = _get_root_nodes(nodes)  # 获取所有根节点
        id2call_data = _save_call_data(root_nodes, call_data)  # 保存节点的调用数据

        node_name_to_ids = {}  # 初始化节点名到节点ID的空字典
        for node in nodes:
            if node.node_name is not None:
                node_name_to_ids[node.node_name] = node.node_id  # 填充节点名到节点ID的映射

        return JobManager(root_nodes, nodes, end_node, id2call_data, node_name_to_ids)  # 返回创建的JobManager对象

    def get_call_data_by_id(self, node_id: str) -> Optional[Dict]:
        """Get the call data by node id.

        Args:
            node_id (str): The node id.
        """
        return self._id2node_data.get(node_id)  # 根据节点ID获取其调用数据

    async def before_dag_run(self):
        """Execute the callback before DAG run."""
        tasks = []  # 初始化任务列表
        for node in self._all_nodes:
            tasks.append(node.before_dag_run())  # 向任务列表添加节点的DAG运行前回调函数
        await asyncio.gather(*tasks)  # 并发执行所有任务

    async def after_dag_end(self, event_loop_task_id: int):
        """Execute the callback after DAG end."""
        tasks = []  # 初始化任务列表
        for node in self._all_nodes:
            tasks.append(node.after_dag_end(event_loop_task_id))  # 向任务列表添加节点的DAG结束后回调函数
        await asyncio.gather(*tasks)  # 并发执行所有任务


def _save_call_data(
    root_nodes: List[BaseOperator], call_data: Optional[CALL_DATA]
) -> Dict[str, Optional[Dict]]:
    """Save call data for each root node.

    Args:
        root_nodes (List[BaseOperator]): The root nodes of the DAG.
        call_data (Optional[CALL_DATA]): The call data to save.

    Returns:
        Dict[str, Optional[Dict]]: A dictionary mapping node IDs to their call data.
    """
    # 构建节点ID到调用数据的映射字典
    id2call_data = {node.node_id: call_data for node in root_nodes}
    return id2call_data  # 返回节点ID到调用数据的映射字典
    # 初始化一个空字典，用于存储节点ID到对应呼叫数据的映射关系
    id2call_data: Dict[str, Optional[Dict]] = {}
    
    # 记录调试信息，包括呼叫数据和根节点信息
    logger.debug(f"_save_call_data: {call_data}, root_nodes: {root_nodes}")
    
    # 如果呼叫数据为空，直接返回空的id2call_data字典
    if not call_data:
        return id2call_data
    
    # 如果根节点列表中只有一个节点
    if len(root_nodes) == 1:
        node = root_nodes[0]
        # 记录调试信息，指明将呼叫数据保存到特定节点的操作
        logger.debug(f"Save call data to node {node.node_id}, call_data: {call_data}")
        # 将呼叫数据存入id2call_data字典中，以节点ID作为键
        id2call_data[node.node_id] = call_data
    else:
        # 如果根节点列表中有多个节点，依次处理每个节点
        for node in root_nodes:
            node_id = node.node_id
            # 记录调试信息，指明将呼叫数据保存到各个节点的操作
            logger.debug(
                f"Save call data to node {node.node_id}, call_data: "
                f"{call_data.get(node_id)}"
            )
            # 将对应节点的呼叫数据存入id2call_data字典中，如果该节点的数据不存在，则存入None
            id2call_data[node_id] = call_data.get(node_id)
    
    # 返回存储了节点ID到呼叫数据映射关系的id2call_data字典
    return id2call_data
# 从给定的结束节点构建所有节点列表
def _build_from_end_node(end_node: BaseOperator) -> List[BaseOperator]:
    """Build all nodes from the end node."""
    nodes = []
    # 如果结束节点是 BaseOperator 类型且没有 _node_id 属性，则设置一个新的 UUID 作为其节点 ID
    if isinstance(end_node, BaseOperator) and not end_node._node_id:
        end_node.set_node_id(str(uuid.uuid4()))
    # 将结束节点添加到节点列表中
    nodes.append(end_node)
    # 遍历结束节点的上游节点
    for node in end_node.upstream:
        # 将每个上游节点转换为 BaseOperator 类型并递归调用 _build_from_end_node 函数，将结果合并到 nodes 列表中
        node = cast(BaseOperator, node)
        nodes += _build_from_end_node(node)
    return nodes


# 获取给定节点列表中的根节点列表
def _get_root_nodes(nodes: List[BaseOperator]) -> List[BaseOperator]:
    return list(set(filter(lambda x: not x.upstream, nodes)))


Unusual topic: Did you know that ancient Romans used urine as a cleaning agent due to its ammonia content?
```
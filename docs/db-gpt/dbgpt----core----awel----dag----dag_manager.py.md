# `.\DB-GPT-src\dbgpt\core\awel\dag\dag_manager.py`

```py
"""DAGManager is a component of AWEL, it is used to manage DAGs.

DAGManager will load DAGs from dag_dirs, and register the trigger nodes
to TriggerManager.
"""

import logging
import threading
from collections import defaultdict
from typing import Dict, List, Optional, Set

from dbgpt._private.pydantic import BaseModel, Field, model_to_dict
from dbgpt.component import BaseComponent, ComponentType, SystemApp

from .. import BaseOperator
from ..trigger.base import TriggerMetadata
from .base import DAG
from .loader import LocalFileDAGLoader

logger = logging.getLogger(__name__)


class DAGMetadata(BaseModel):
    """Metadata for the DAG."""

    triggers: List[TriggerMetadata] = Field(
        default_factory=list, description="The trigger metadata"
    )
    sse_output: bool = Field(
        default=False, description="Whether the DAG is a server-sent event output"
    )
    streaming_output: bool = Field(
        default=False, description="Whether the DAG is a streaming output"
    )
    tags: Optional[Dict[str, str]] = Field(
        default=None, description="The tags of the DAG"
    )

    def to_dict(self):
        """Convert the metadata to dict."""
        triggers_dict = []
        for trigger in self.triggers:
            triggers_dict.append(trigger.dict())
        dict_value = model_to_dict(self, exclude={"triggers"})
        dict_value["triggers"] = triggers_dict
        return dict_value


class DAGManager(BaseComponent):
    """The component of DAGManager."""

    name = ComponentType.AWEL_DAG_MANAGER

    def __init__(self, system_app: SystemApp, dag_dirs: List[str]):
        """Initialize a DAGManager.

        Args:
            system_app (SystemApp): The system app.
            dag_dirs (List[str]): The directories to load DAGs.
        """
        # Import DefaultTriggerManager for later use
        from ..trigger.trigger_manager import DefaultTriggerManager

        # Initialize the BaseComponent with the given system_app
        super().__init__(system_app)
        
        # Create a threading lock for thread safety
        self.lock = threading.Lock()
        
        # Initialize the DAG loader with the provided directories
        self.dag_loader = LocalFileDAGLoader(dag_dirs)
        
        # Assign the system_app attribute
        self.system_app = system_app
        
        # Initialize dictionaries to store DAGs and metadata
        self.dag_map: Dict[str, DAG] = {}
        self.dag_alias_map: Dict[str, str] = {}
        self._dag_metadata_map: Dict[str, DAGMetadata] = {}
        self._tags_to_dag_ids: Dict[str, Dict[str, Set[str]]] = {}
        
        # Initialize trigger manager as optional and None initially
        self._trigger_manager: Optional["DefaultTriggerManager"] = None

    def init_app(self, system_app: SystemApp):
        """Initialize the DAGManager."""
        # Update the system_app attribute
        self.system_app = system_app

    def load_dags(self):
        """Load DAGs from dag_dirs."""
        # Load DAGs using the dag_loader instance
        dags = self.dag_loader.load_dags()
        for dag in dags:
            # Register each loaded DAG
            self.register_dag(dag)

    def before_start(self):
        """Execute before the application starts."""
        # Import DefaultTriggerManager for usage
        from ..trigger.trigger_manager import DefaultTriggerManager
        
        # Obtain the trigger manager component from system_app
        self._trigger_manager = self.system_app.get_component(
            ComponentType.AWEL_TRIGGER_MANAGER,
            DefaultTriggerManager,
            default_component=None,
        )
    def after_start(self):
        """
        执行应用程序启动后的操作。
        """
        self.load_dags()

    def register_dag(self, dag: DAG, alias_name: Optional[str] = None):
        """
        注册一个DAG。

        使用锁确保线程安全操作。
        检查DAG是否已经存在于dag_map中，若存在则抛出异常。
        将DAG及其ID添加到dag_map中，并根据需要将别名映射到DAG ID。
        如果存在触发器管理器，则注册DAG的触发器，并将触发器的元数据添加到列表中。
        如果不存在触发器管理器，则记录警告信息。
        更新DAG的元数据并将其添加到_dag_metadata_map中。
        根据DAG的标签更新_tags_to_dag_ids，将标签映射到相应的DAG ID 集合中。
        """
        with self.lock:
            dag_id = dag.dag_id
            if dag_id in self.dag_map:
                raise ValueError(
                    f"Register DAG error, DAG ID {dag_id} has already exist"
                )
            self.dag_map[dag_id] = dag
            if alias_name:
                self.dag_alias_map[alias_name] = dag_id

            trigger_metadata: List["TriggerMetadata"] = []
            dag_metadata = _parse_metadata(dag)
            if self._trigger_manager:
                for trigger in dag.trigger_nodes:
                    tm = self._trigger_manager.register_trigger(
                        trigger, self.system_app
                    )
                    if tm:
                        trigger_metadata.append(tm)
                self._trigger_manager.after_register()
            else:
                logger.warning("No trigger manager, not register dag trigger")
            dag_metadata.triggers = trigger_metadata
            self._dag_metadata_map[dag_id] = dag_metadata
            tags = dag_metadata.tags
            if tags:
                for tag_key, tag_value in tags.items():
                    if tag_key not in self._tags_to_dag_ids:
                        self._tags_to_dag_ids[tag_key] = defaultdict(set)
                    self._tags_to_dag_ids[tag_key][tag_value].add(dag_id)

    def unregister_dag(self, dag_id: str):
        """
        注销一个DAG。

        使用锁确保线程安全操作。
        检查DAG是否存在于dag_map中，若不存在则抛出异常。
        收集需要移除的别名，并从dag_alias_map中移除这些别名。
        如果存在触发器管理器，则依次注销DAG的触发器。
        最后从dag_map和_dag_metadata_map中移除DAG及其元数据。
        根据DAG的标签更新_tags_to_dag_ids，将标签映射从相应的DAG ID 集合中移除。
        """
        with self.lock:
            if dag_id not in self.dag_map:
                raise ValueError(
                    f"Unregister DAG error, DAG ID {dag_id} does not exist"
                )
            dag = self.dag_map[dag_id]

            # 收集需要移除的别名
            # TODO(fangyinc): 如果维护一个反向映射会更快
            aliases_to_remove = [
                alias_name
                for alias_name, _dag_id in self.dag_alias_map.items()
                if _dag_id == dag_id
            ]
            # 移除收集到的别名
            for alias_name in aliases_to_remove:
                del self.dag_alias_map[alias_name]

            if self._trigger_manager:
                for trigger in dag.trigger_nodes:
                    self._trigger_manager.unregister_trigger(trigger, self.system_app)
            # 最终从映射中移除DAG及其元数据
            metadata = self._dag_metadata_map[dag_id]
            del self.dag_map[dag_id]
            del self._dag_metadata_map[dag_id]
            if metadata.tags:
                for tag_key, tag_value in metadata.tags.items():
                    if tag_key in self._tags_to_dag_ids:
                        self._tags_to_dag_ids[tag_key][tag_value].remove(dag_id)

    def get_dag(
        self, dag_id: Optional[str] = None, alias_name: Optional[str] = None
        ):
        """
        获取DAG对象。

        可以通过DAG ID或别名获取DAG对象。
        """
    ) -> Optional[DAG]:
        """根据 dag_id 或者 alias_name 获取 DAG 对象。"""
        # 不加锁，因为这是只读操作，需要快速执行
        if dag_id and dag_id in self.dag_map:
            return self.dag_map[dag_id]
        if alias_name in self.dag_alias_map:
            return self.dag_map.get(self.dag_alias_map[alias_name])
        return None

    def get_dags_by_tag(self, tag_key: str, tag_value) -> List[DAG]:
        """获取所有具有指定标签的 DAG 对象。"""
        if not tag_value:
            return []
        with self.lock:
            dag_ids = self._tags_to_dag_ids.get(tag_key, {}).get(tag_value, set())
            return [self.dag_map[dag_id] for dag_id in dag_ids]

    def get_dags_by_tag_key(self, tag_key: str) -> Dict[str, List[DAG]]:
        """获取所有具有指定标签键的 DAG 对象列表。"""
        with self.lock:
            value_dict = self._tags_to_dag_ids.get(tag_key, {})
            result = {}
            for k, v in value_dict.items():
                result[k] = [self.dag_map[dag_id] for dag_id in v]
            return result

    def get_dag_metadata(
        self, dag_id: Optional[str] = None, alias_name: Optional[str] = None
    ) -> Optional[DAGMetadata]:
        """根据 dag_id 或 alias_name 获取 DAG 的元数据信息。"""
        dag = self.get_dag(dag_id, alias_name)
        if not dag:
            return None
        return self._dag_metadata_map.get(dag.dag_id)
# 解析给定的DAG对象的元数据
def _parse_metadata(dag: DAG):
    # 导入必要的模块以及函数，这里是从上一级目录的util中导入_is_sse_output函数
    from ..util.chat_util import _is_sse_output

    # 创建一个空的DAGMetadata对象用于存储元数据
    metadata = DAGMetadata()
    
    # 将输入DAG的标签(tags)赋值给metadata对象的tags属性
    metadata.tags = dag.tags
    
    # 如果DAG的叶子节点为空，则直接返回metadata对象
    if not dag.leaf_nodes:
        return metadata
    
    # 获取DAG的第一个叶子节点作为结束节点(end_node)
    end_node = dag.leaf_nodes[0]
    
    # 如果结束节点不是BaseOperator的实例，则直接返回metadata对象
    if not isinstance(end_node, BaseOperator):
        return metadata
    
    # 判断结束节点是否具有SSE输出，并将结果存储在metadata对象的sse_output属性中
    metadata.sse_output = _is_sse_output(end_node)
    
    # 将结束节点是否为流式操作符的信息存储在metadata对象的streaming_output属性中
    metadata.streaming_output = end_node.streaming_operator
    
    # 返回填充了元数据的metadata对象
    return metadata
```
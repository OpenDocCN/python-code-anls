# `.\DB-GPT-src\dbgpt\storage\graph_store\tugraph_store.py`

```py
"""TuGraph vector store."""
import logging  # 导入日志模块
import os  # 导入操作系统接口模块
from typing import List, Optional, Tuple  # 导入类型提示模块

from dbgpt._private.pydantic import ConfigDict, Field  # 导入配置字典和字段类
from dbgpt.datasource.conn_tugraph import TuGraphConnector  # 导入TuGraph连接器类
from dbgpt.storage.graph_store.base import GraphStoreBase, GraphStoreConfig  # 导入图形存储基类和配置类
from dbgpt.storage.graph_store.graph import Direction, Edge, MemoryGraph, Vertex  # 导入方向、边、内存图和顶点类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class TuGraphStoreConfig(GraphStoreConfig):
    """TuGraph store config."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # 模型配置为允许任意类型

    host: str = Field(
        default="127.0.0.1",
        description="TuGraph host",
    )
    port: int = Field(
        default=7687,
        description="TuGraph port",
    )
    username: str = Field(
        default="admin",
        description="login username",
    )
    password: str = Field(
        default="123456",
        description="login password",
    )
    vertex_type: str = Field(
        default="entity",
        description="The type of graph vertex, `entity` by default.",
    )
    edge_type: str = Field(
        default="relation",
        description="The type of graph edge, `relation` by default.",
    )
    edge_name_key: str = Field(
        default="label",
        description="The label of edge name, `label` by default.",
    )


class TuGraphStore(GraphStoreBase):
    """TuGraph graph store."""

    def __init__(self, config: TuGraphStoreConfig) -> None:
        """Initialize the TuGraphStore with connection details."""
        # 从环境变量或配置中获取TuGraph连接详情
        self._host = os.getenv("TUGRAPH_HOST", "127.0.0.1") or config.host
        self._port = int(os.getenv("TUGRAPH_PORT", 7687)) or config.port
        self._username = os.getenv("TUGRAPH_USERNAME", "admin") or config.username
        self._password = os.getenv("TUGRAPH_PASSWORD", "73@TuGraph") or config.password
        self._node_label = (
            os.getenv("TUGRAPH_VERTEX_TYPE", "entity") or config.vertex_type
        )
        self._edge_label = (
            os.getenv("TUGRAPH_EDGE_TYPE", "relation") or config.edge_type
        )
        self.edge_name_key = (
            os.getenv("TUGRAPH_EDGE_NAME_KEY", "label") or config.edge_name_key
        )
        self._graph_name = config.name  # 设置图形名称
        self.conn = TuGraphConnector.from_uri_db(  # 使用连接器从URI创建数据库连接
            host=self._host,
            port=self._port,
            user=self._username,
            pwd=self._password,
            db_name=config.name,
        )
        self.conn.create_graph(graph_name=config.name)  # 创建图形

        self._create_schema()  # 调用私有方法创建模式

    def _check_label(self, elem_type: str):
        """Check if the element type label exists in the connected database."""
        result = self.conn.get_table_names()  # 获取连接数据库中的表名列表
        if elem_type == "vertex":
            return self._node_label in result["vertex_tables"]  # 检查顶点标签是否存在
        if elem_type == "edge":
            return self._edge_label in result["edge_tables"]  # 检查边标签是否存在
    def _create_schema(self):
        # 如果“vertex”标签不存在，则创建一个新的顶点标签
        if not self._check_label("vertex"):
            create_vertex_gql = (
                f"CALL db.createLabel("
                f"'vertex', '{self._node_label}', "
                f"'id', ['id',string,false])"
            )
            self.conn.run(create_vertex_gql)
        
        # 如果“edge”标签不存在，则创建一个新的边标签
        if not self._check_label("edge"):
            create_edge_gql = f"""CALL db.createLabel(
                'edge', '{self._edge_label}', '[["{self._node_label}",
                "{self._node_label}"]]', ["id",STRING,false])"""
            self.conn.run(create_edge_gql)

    def get_triplets(self, subj: str) -> List[Tuple[str, str]]:
        """Get triplets."""
        # 构建查询，获取指定主体的三元组
        query = (
            f"MATCH (n1:{self._node_label})-[r]->(n2:{self._node_label}) "
            f'WHERE n1.id = "{subj}" RETURN r.id as rel, n2.id as obj;'
        )
        data = self.conn.run(query)
        return [(record["rel"], record["obj"]) for record in data]

    def insert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""

        def escape_quotes(value: str) -> str:
            """Escape single and double quotes in a string for queries."""
            return value.replace("'", "\\'").replace('"', '\\"')

        # 对主体、关系和客体进行引号转义处理
        subj_escaped = escape_quotes(subj)
        rel_escaped = escape_quotes(rel)
        obj_escaped = escape_quotes(obj)

        # 构建并运行主体和客体的MERGE查询
        subj_query = f"MERGE (n1:{self._node_label} {{id:'{subj_escaped}'}})"
        obj_query = f"MERGE (n1:{self._node_label} {{id:'{obj_escaped}'}})"
        self.conn.run(query=subj_query)
        self.conn.run(query=obj_query)

        # 构建并运行关系的MERGE查询
        rel_query = (
            f"MERGE (n1:{self._node_label} {{id:'{subj_escaped}'}})"
            f"-[r:{self._edge_label} {{id:'{rel_escaped}'}}]->"
            f"(n2:{self._node_label} {{id:'{obj_escaped}'}})"
        )
        self.conn.run(query=rel_query)

    def drop(self):
        """Delete Graph."""
        # 删除整个图谱
        self.conn.delete_graph(self._graph_name)

    def delete_triplet(self, sub: str, rel: str, obj: str) -> None:
        """Delete triplet."""
        # 构建并运行删除三元组的查询
        del_query = (
            f"MATCH (n1:{self._node_label} {{id:'{sub}'}})"
            f"-[r:{self._edge_label} {{id:'{rel}'}}]->"
            f"(n2:{self._node_label} {{id:'{obj}'}}) DELETE n1,n2,r"
        )
        self.conn.run(query=del_query)

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the graph store."""
        # 获取图存储的架构信息
        query = "CALL dbms.graph.getGraphSchema()"
        data = self.conn.run(query=query)
        schema = data[0]["schema"]
        return schema

    def get_full_graph(self, limit: Optional[int] = None) -> MemoryGraph:
        """Get full graph."""
        # 获取完整的图，可以设置限制以控制结果大小
        if not limit:
            raise Exception("limit must be set")
        return self.query(f"MATCH (n)-[r]-(m) RETURN n,m,r LIMIT {limit}")
    def explore(
        self,
        subs: List[str],
        direct: Direction = Direction.BOTH,
        depth: Optional[int] = None,
        fan: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> MemoryGraph:
        """Explore the graph from given subjects up to a depth."""
        # 如果指定了 fan 参数，抛出异常，因为当前不支持该功能
        if fan is not None:
            raise ValueError("Fan functionality is not supported at this time.")
        else:
            # 根据 depth 参数设置查询字符串的深度限制
            depth_string = f"1..{depth}"
            if depth is None:
                depth_string = ".."

            # 根据 limit 参数设置查询字符串的结果数量限制
            limit_string = f"LIMIT {limit}"
            if limit is None:
                limit_string = ""

            # 构建 Cypher 查询语句，匹配符合条件的节点和边
            query = (
                f"MATCH p=(n:{self._node_label})"
                f"-[r:{self._edge_label}*{depth_string}]-(m:{self._node_label}) "
                f"WHERE n.id IN {subs} RETURN p {limit_string}"
            )
            # 执行查询并返回结果
            return self.query(query)
    def query(self, query: str, **args) -> MemoryGraph:
        """Execute a query on graph."""

        # 定义内部函数，用于格式化路径信息
        def _format_paths(paths):
            formatted_paths = []
            # 遍历路径列表
            for path in paths:
                formatted_path = []
                # 获取路径中的节点列表和关系列表
                nodes = list(path["p"].nodes)
                rels = list(path["p"].relationships)
                # 遍历节点列表，将节点的id添加到格式化路径中
                for i in range(len(nodes)):
                    formatted_path.append(nodes[i]._properties["id"])
                    # 如果还有关系，将关系的id也添加到格式化路径中
                    if i < len(rels):
                        formatted_path.append(rels[i]._properties["id"])
                formatted_paths.append(formatted_path)
            return formatted_paths

        # 定义内部函数，用于格式化查询返回的数据
        def _format_query_data(data):
            node_ids_set = set()
            rels_set = set()
            from neo4j import graph

            # 遍历查询返回的记录
            for record in data:
                for key in record.keys():
                    value = record[key]
                    # 如果值是节点对象，提取节点id并加入节点id集合
                    if isinstance(value, graph.Node):
                        node_id = value._properties["id"]
                        node_ids_set.add(node_id)
                    # 如果值是关系对象，提取关系相关信息并加入关系集合
                    elif isinstance(value, graph.Relationship):
                        rel_nodes = value.nodes
                        prop_id = value._properties["id"]
                        src_id = rel_nodes[0]._properties["id"]
                        dst_id = rel_nodes[1]._properties["id"]
                        rels_set.add((src_id, dst_id, prop_id))
                    # 如果值是路径对象，调用_format_paths函数处理路径并加入相应集合
                    elif isinstance(value, graph.Path):
                        formatted_paths = _format_paths(data)
                        for path in formatted_paths:
                            for i in range(0, len(path), 2):
                                node_ids_set.add(path[i])
                                if i + 2 < len(path):
                                    rels_set.add((path[i], path[i + 2], path[i + 1]))

            # 根据节点id集合创建节点对象列表
            nodes = [Vertex(node_id) for node_id in node_ids_set]
            # 根据关系集合创建边对象列表
            rels = [
                Edge(src_id, dst_id, label=prop_id)
                for (src_id, dst_id, prop_id) in rels_set
            ]
            return {"nodes": nodes, "edges": rels}

        # 执行查询并获取结果
        result = self.conn.run(query=query)
        # 格式化查询结果数据
        graph = _format_query_data(result)
        # 创建内存图对象
        mg = MemoryGraph()
        # 将格式化后的节点对象逐个插入内存图中
        for vertex in graph["nodes"]:
            mg.upsert_vertex(vertex)
        # 将格式化后的边对象逐个添加到内存图中
        for edge in graph["edges"]:
            mg.append_edge(edge)
        # 返回构建好的内存图对象
        return mg
```
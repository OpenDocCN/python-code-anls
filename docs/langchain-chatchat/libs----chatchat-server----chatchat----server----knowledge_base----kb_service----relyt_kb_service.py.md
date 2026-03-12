
# `Langchain-Chatchat\libs\chatchat-server\chatchat\server\knowledge_base\kb_service\relyt_kb_service.py` 详细设计文档

RelytKBService是基于PostgreSQL向量存储(PGVecto_rs)的知识库服务实现，提供知识库的创建、文档添加、删除、搜索等功能，支持向量相似度检索和全文检索，继承自KBService抽象基类。

## 整体流程

```mermaid
graph TD
    A[开始] --> B[do_init 初始化]
    B --> C[_load_relyt_vector 加载向量存储]
    C --> D[do_create_kb 创建知识库]
    D --> E{知识库已存在?}
E -- 否 --> F[创建索引]
E -- 是 --> G[结束初始化]
F --> G
G --> H[等待操作]
H --> I[do_add_doc 添加文档]
H --> J[do_delete_doc 删除文档]
H --> K[do_search 搜索文档]
H --> L[do_drop_kb 删除知识库]
I --> M[调用PGVecto_rs.add_documents]
J --> N[执行DELETE SQL]
K --> O[调用similarity_search_with_score]
L --> P[执行DROP TABLE SQL]
```

## 类结构

```
KBService (抽象基类)
└── RelytKBService (Relyt知识库服务实现)
```

## 全局变量及字段


### `kbs_config`
    
知识库配置对象，包含各类型知识库的连接配置

类型：`Any`
    


### `Document`
    
LangChain文档模式，用于表示文本及其元数据

类型：`Type[Document]`
    


### `PGVecto_rs`
    
PGVecto_rs向量数据库类，用于存储和检索向量

类型：`Type[PGVecto_rs]`
    


### `Session`
    
SQLAlchemy会话类，用于执行数据库事务

类型：`Type[Session]`
    


### `EmbeddingsFunAdapter`
    
嵌入函数适配器，将嵌入模型封装为LangChain可用的嵌入函数

类型：`Type[EmbeddingsFunAdapter]`
    


### `KBService`
    
知识库服务基类，提供知识库操作的标准接口

类型：`Type[KBService]`
    


### `SupportedVSType`
    
支持的向量存储类型枚举，包含各种向量数据库类型

类型：`Type[SupportedVSType]`
    


### `score_threshold_process`
    
分数阈值处理函数，用于过滤相似度低于阈值的文档

类型：`Callable`
    


### `KnowledgeFile`
    
知识文件模型，表示知识库中的文件及其元信息

类型：`Type[KnowledgeFile]`
    


### `RelytKBService.relyt`
    
Relyt向量数据库实例

类型：`PGVecto_rs`
    


### `RelytKBService.engine`
    
SQLAlchemy引擎，用于执行原始SQL语句

类型：`Engine`
    


### `RelytKBService.kb_name`
    
知识库名称

类型：`str`
    


### `RelytKBService.embed_model`
    
嵌入模型，用于将文本转换为向量

类型：`Any`
    
    

## 全局函数及方法



### `create_engine`

SQLAlchemy的`create_engine`函数是用于创建数据库引擎的核心函数，它根据提供的连接URI建立与数据库的连接，并返回一个`Engine`对象，该对象用于管理数据库连接池和执行SQL语句。

参数：

- `url`：`str`，数据库连接URI，格式为`dialect+driver://username:password@host:port/database`，例如`postgresql+psycopg2://user:password@localhost:5432/mydb`

返回值：`Engine`，返回一个SQLAlchemy引擎对象，用于后续的数据库操作，包括连接管理和SQL执行

#### 流程图

```mermaid
flowchart TD
    A[开始创建引擎] --> B[接收连接URI字符串]
    B --> C[解析URI获取数据库类型和连接参数]
    C --> D[初始化连接池]
    D --> E[配置日志和方言]
    E --> F[创建Engine对象]
    F --> G[返回Engine实例]
    
    subgraph "后续操作"
    H[通过Engine获取连接] --> I[执行SQL语句]
    I --> J[提交事务]
    J --> K[关闭连接]
    end
    
    G --> H
```

#### 带注释源码

```python
# 从sqlalchemy导入create_engine函数
# 这是SQLAlchemy 2.0之前的经典API
from sqlalchemy import create_engine, text

# 在RelytKBService类中的调用示例
class RelytKBService(KBService):
    def _load_relyt_vector(self):
        # ... embedding相关代码 ...
        
        # 创建SQLAlchemy引擎
        # 参数: kbs_config.get("relyt").get("connection_uri") - 数据库连接URI字符串
        # 返回值: Engine对象 - 用于管理数据库连接
        self.engine = create_engine(
            kbs_config.get("relyt").get("connection_uri")
        )
        
        # 引擎创建后可进行的操作:
        # 1. 获取连接: with self.engine.connect() as conn:
        # 2. 执行原始SQL: conn.execute(text("SELECT * FROM table"))
        # 3. 开始事务: conn.begin()
        # 4. 使用ORM: Session(self.engine)

# create_engine函数的标准用法
# 基本调用
engine = create_engine("sqlite:///example.db")

# 带连接参数的调用
engine = create_engine(
    "postgresql+psycopg2://user:password@localhost:5432/mydb",
    pool_size=5,           # 连接池大小
    max_overflow=10,       # 额外连接数
    echo=True,             # 是否打印SQL语句
    pool_pre_ping=True     # 连接前测试有效性
)

# 使用引擎执行SQL
with engine.connect() as connection:
    result = connection.execute(text("SELECT version()"))
    print(result.fetchone())
```



### `RelytKBService.get_doc_by_ids`

根据代码分析，该方法使用 SQLAlchemy 的 `text()` 函数构建原生 SELECT 查询语句，通过 Session 执行查询并从 Relyt 向量数据库中根据文档 ID 列表检索对应的 Document 对象。

参数：

- `ids`：`List[str]`，需要查询的文档 ID 列表

返回值：`List[Document]`，查询到的 Document 对象列表，每个 Document 包含 page_content（页面内容）和 metadata（元数据）

#### 流程图

```mermaid
flowchart TD
    A[开始 get_doc_by_ids] --> B[将 IDs 列表格式化为逗号分隔的字符串]
    B --> C[创建 Session 对象]
    C --> D[使用 text 构建 SELECT SQL 语句]
    D --> E[执行查询并获取所有结果]
    E --> F[遍历结果行, 构造 Document 对象]
    F --> G[返回 Document 列表]
```

#### 带注释源码

```python
def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
    """
    根据文档 ID 列表从 Relyt 向量数据库中检索对应的 Document 对象
    
    Args:
        ids: 文档唯一标识符列表
        
    Returns:
        包含页面内容和元数据的 Document 对象列表
    """
    # 将 ID 列表转换为逗号分隔的字符串格式，用于 SQL IN 子句
    ids_str = ", ".join([f"{id}" for id in ids])
    
    # 创建数据库会话，确保会话自动管理事务
    with Session(self.engine) as session:
        # 使用 SQLAlchemy text() 构建原生 SELECT 查询语句
        # 查询 collection_{kb_name} 表中指定 ID 的文档记录
        stmt = text(
            f"SELECT text, meta FROM collection_{self.kb_name} WHERE id in (:ids)"
        )
        
        # 执行查询并将结果转换为 Document 对象列表
        # row[0] 对应 text 字段，row[1] 对应 meta 字段
        results = [
            Document(page_content=row[0], metadata=row[1])
            for row in session.execute(stmt, {"ids": ids_str}).fetchall()
        ]
        return results
```

---

### `RelytKBService.del_doc_by_ids`

该方法使用 SQLAlchemy 的 `text()` 函数构建原生 DELETE 语句，通过 Session 执行删除操作，根据提供的文档 ID 列表从 Relyt 向量数据库中删除对应的文档记录。

参数：

- `ids`：`List[str]`，需要删除的文档 ID 列表

返回值：`bool`，删除操作是否成功完成（始终返回 True）

#### 流程图

```mermaid
flowchart TD
    A[开始 del_doc_by_ids] --> B[将 IDs 列表格式化为逗号分隔的字符串]
    B --> C[创建 Session 对象]
    C --> D[使用 text 构建 DELETE SQL 语句]
    D --> E[执行删除操作]
    E --> F[提交事务 commit]
    F --> G[返回 True 表示删除成功]
```

#### 带注释源码

```python
def del_doc_by_ids(self, ids: List[str]) -> bool:
    """
    根据文档 ID 列表从 Relyt 向量数据库中删除对应的文档记录
    
    Args:
        ids: 文档唯一标识符列表
        
    Returns:
        布尔值，表示删除操作是否成功
    """
    # 将 ID 列表转换为逗号分隔的字符串格式
    ids_str = ", ".join([f"{id}" for id in ids])
    
    # 创建数据库会话
    with Session(self.engine) as session:
        # 使用 SQLAlchemy text() 构建原生 DELETE 语句
        # 从 collection_{kb_name} 表中删除指定 ID 的记录
        stmt = text(f"DELETE FROM collection_{self.kb_name} WHERE id in (:ids)")
        
        # 执行删除操作并提交事务
        session.execute(stmt, {"ids": ids_str})
        session.commit()
    
    # 返回 True 表示删除操作成功完成
    return True
```

---

### `RelytKBService.do_create_kb`

该方法使用 SQLAlchemy 的 `text()` 函数构建原生 SQL 语句，首先检查指定的索引是否已存在于 pg_indexes 系统表中，如果不存在则创建基于向量的索引，以优化 Relyt 向量数据库的相似性搜索性能。

参数：

- 无

返回值：`None`

#### 流程图

```mermaid
flowchart TD
    A[开始 do_create_kb] --> B[构建索引名称: idx_{kb_name}_embedding]
    B --> C[获取数据库连接 conn]
    C --> D[开始事务 begin]
    D --> E[使用 text 构建索引查询 SQL]
    E --> F[执行查询检查索引是否存在]
    F --> G{result 是否为空?}
    G -->|是| H[使用 text 构建 CREATE INDEX SQL]
    H --> I[执行创建索引语句]
    I --> J[提交事务]
    G -->|否| K[跳过创建]
    K --> J
    J --> L[结束]
```

#### 带注释源码

```python
def do_create_kb(self):
    """
    创建 Relyt 向量数据库的索引
    如果索引不存在，则创建一个基于向量的索引以优化相似性搜索性能
    """
    # 构建索引名称，格式为 idx_{知识库名称}_embedding
    index_name = f"idx_{self.kb_name}_embedding"
    
    # 获取数据库连接并开始事务
    with self.engine.connect() as conn:
        with conn.begin():
            # 使用 SQLAlchemy text() 构建原生 SQL 查询语句
            # 检查 pg_indexes 系统表中是否已存在指定索引
            index_query = text(
                f"""
                    SELECT 1
                    FROM pg_indexes
                    WHERE indexname = '{index_name}';
                """
            )
            
            # 执行查询并获取结果（返回 1 或 None）
            result = conn.execute(index_query).scalar()
            
            # 如果索引不存在，则创建新的向量索引
            if not result:
                # 使用 SQLAlchemy text() 构建原生 CREATE INDEX 语句
                # 创建向量索引以支持 L2 距离的相似性搜索
                index_statement = text(
                    f"""
                        CREATE INDEX {index_name}
                        ON collection_{self.kb_name}
                        USING vectors (embedding vector_l2_ops)
                        WITH (options = $$
                        optimizing.optimizing_threads = 30
                        segment.max_growing_segment_size = 2000
                        segment.max_sealed_segment_size = 30000000
                        [indexing.hnsw]
                        m=30
                        ef_construction=500
                        $$);
                    """
                )
                # 执行索引创建语句
                conn.execute(index_statement)
```

---

### `RelytKBService.do_drop_kb`

该方法使用 SQLAlchemy 的 `text()` 函数构建原生 DROP TABLE 语句，通过数据库连接执行删除操作，将整个知识库对应的向量集合表从 Relyt 数据库中删除。

参数：

- 无

返回值：`None`

#### 流程图

```mermaid
flowchart TD
    A[开始 do_drop_kb] --> B[使用 text 构建 DROP TABLE SQL]
    B --> C[获取数据库连接 conn]
    C --> D[开始事务 begin]
    D --> E[执行 DROP TABLE 语句]
    E --> F[提交事务]
    F --> G[结束]
```

#### 带注释源码

```python
def do_drop_kb(self):
    """
    删除 Relyt 向量数据库中的整个知识库集合
    使用 DROP TABLE IF EXISTS 确保即使表不存在也不会报错
    """
    # 使用 SQLAlchemy text() 构建原生 DROP TABLE 语句
    # IF EXISTS 子句防止表不存在时抛出异常
    drop_statement = text(f"DROP TABLE IF EXISTS collection_{self.kb_name};")
    
    # 获取数据库连接并执行删除操作
    with self.engine.connect() as conn:
        with conn.begin():
            # 执行删除知识库集合表的操作
            conn.execute(drop_statement)
```

---

### `RelytKBService.do_delete_doc`

该方法使用 SQLAlchemy 的 `text()` 函数构建原生 DELETE 语句，通过 Session 执行删除操作，根据知识文件的相对路径从 Relyt 向量数据库中删除所有匹配的文档记录。

参数：

- `kb_file`：`KnowledgeFile`，包含文件路径信息的知识文件对象

返回值：`None`

#### 流程图

```mermaid
flowchart TD
    A[开始 do_delete_doc] --> B[获取文件的相对路径]
    B --> C[构建 DELETE SQL 语句]
    C --> D[创建 Session 对象]
    D --> E[使用 text 执行删除操作]
    E --> F[提交事务 commit]
    F --> G[结束]
```

#### 带注释源码

```python
def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
    """
    根据知识文件路径从 Relyt 向量数据库中删除所有匹配的文档记录
    
    Args:
        kb_file: 知识文件对象，包含待删除文件的路径信息
        **kwargs: 其他可选参数（未使用）
    """
    # 获取知识文件的相对路径，用于匹配数据库中的 source 元数据字段
    filepath = self.get_relative_source_path(kb_file.filepath)
    
    # 构建原生 DELETE SQL 语句
    # 使用 meta->>'source' 提取 JSONB 字段的值进行匹配
    stmt = f"DELETE FROM collection_{self.kb_name} WHERE meta->>'source'='{filepath}'; "
    
    # 创建数据库会话并执行删除操作
    with Session(self.engine) as session:
        # 使用 SQLAlchemy text() 构建并执行删除语句
        session.execute(text(stmt))
        # 提交事务以确保删除操作生效
        session.commit()
```



### `RelytKBService._load_relyt_vector`

该方法负责初始化Relyt向量存储实例和SQLAlchemy数据库引擎，通过获取嵌入模型适配器、创建向量存储连接和数据库连接，为知识库服务提供向量检索和数据库操作的能力。

参数： 无显式参数（该方法为类实例方法，通过 `self` 访问实例属性 `self.embed_model` 和 `self.kb_name`）

返回值：`None`（无返回值，该方法通过实例属性 `self.relyt` 和 `self.engine` 保存向量存储和数据库引擎实例）

#### 流程图

```mermaid
flowchart TD
    A[开始 _load_relyt_vector] --> B[创建 EmbeddingsFunAdapter 适配器]
    B --> C[调用 embed_query 获取样本向量]
    C --> D[获取 Relyt 配置的连接 URI]
    D --> E[创建 PGVecto_rs 向量存储实例]
    E --> F[创建 SQLAlchemy 数据库引擎]
    F --> G[结束]
    
    B -->|赋值| B1[embedding_func]
    C -->|用于| D
    D -->|db_url参数| E
    E -->|赋值到| E1[self.relyt]
    F -->|赋值到| F1[self.engine]
```

#### 带注释源码

```python
def _load_relyt_vector(self):
    """
    加载Relyt向量存储实例和SQLAlchemy引擎
    
    该方法完成以下初始化工作：
    1. 创建嵌入函数适配器
    2. 获取样本向量以确定维度
    3. 初始化PGVecto_rs向量存储
    4. 创建SQLAlchemy数据库引擎
    """
    # 第一步：创建嵌入函数适配器，将self.embed_model封装为LangChain兼容的嵌入接口
    embedding_func = EmbeddingsFunAdapter(self.embed_model)
    
    # 第二步：通过实际调用嵌入函数获取样本向量，用于确定向量维度
    # 注意：这里使用一个示例字符串来触发嵌入模型的实际调用
    sample_embedding = embedding_func.embed_query("Hello relyt!")
    
    # 第三步：创建PGVecto_rs向量存储实例
    # 参数说明：
    # - embedding: 嵌入函数适配器实例
    # - dimension: 嵌入向量的维度，由样本向量长度确定
    # - db_url: 从配置文件获取的Relyt数据库连接URI
    # - collection_name: 知识库名称，用于标识向量集合
    self.relyt = PGVecto_rs(
        embedding=embedding_func,
        dimension=len(sample_embedding),
        db_url=kbs_config.get("relyt").get("connection_uri"),
        collection_name=self.kb_name,
    )
    
    # 第四步：创建SQLAlchemy数据库引擎，用于执行原生SQL操作
    # 该引擎用于后续的文档ID查询、删除等数据库操作
    self.engine = create_engine(kbs_config.get("relyt").get("connection_uri"))
```



### `RelytKBService.get_doc_by_ids`

根据给定的文档ID列表从Relyt知识库中检索对应的Document对象列表。该方法通过SQL查询数据库中指定ID的文档记录，并将结果转换为LangChain的Document对象返回。

参数：

- `self`：隐式参数，RelytKBService实例本身
- `ids`：`List[str]`，文档ID列表，用于指定需要检索的文档唯一标识符

返回值：`List[Document]`，包含检索到的文档对象列表，每个Document对象包含页面内容（page_content）和元数据（metadata）

#### 流程图

```mermaid
flowchart TD
    A[开始 get_doc_by_ids] --> B[将 ids 列表转换为逗号分隔的字符串 ids_str]
    B --> C[创建数据库会话 Session]
    C --> D[构建 SQL 查询语句]
    D --> E[执行查询: SELECT text, meta FROM collection_{kb_name} WHERE id in (:ids)]
    E --> F[遍历查询结果]
    F --> G[为每行结果创建 Document 对象]
    G --> H[将所有 Document 对象收集到列表中]
    H --> I[返回结果列表]
    I --> J[结束]
```

#### 带注释源码

```python
def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
    """
    根据文档ID列表获取Document对象列表
    
    参数:
        ids: 文档ID列表
        
    返回:
        包含检索到的Document对象的列表
    """
    # 将ID列表转换为逗号分隔的字符串格式，用于SQL IN子句
    # 例如: ['id1', 'id2'] -> 'id1, id2'
    ids_str = ", ".join([f"{id}" for id in ids])
    
    # 使用SQLAlchemy Session管理数据库连接，确保连接正确关闭
    with Session(self.engine) as session:
        # 构建SQL查询语句，从指定知识库集合中查询文本和元数据
        # 注意: 使用参数化查询防止SQL注入，但此处表名无法参数化需确保kb_name安全
        stmt = text(
            f"SELECT text, meta FROM collection_{self.kb_name} WHERE id in (:ids)"
        )
        
        # 执行查询并遍历结果，为每行数据创建LangChain Document对象
        # page_content对应数据库中的text字段，metadata对应meta字段
        results = [
            Document(page_content=row[0], metadata=row[1])
            for row in session.execute(stmt, {"ids": ids_str}).fetchall()
        ]
        
        # 返回查询结果列表
        return results
```



### `RelytKBService.del_doc_by_ids`

根据文档ID列表从 Relyt 知识库中删除指定的文档记录。

参数：

- `ids`：`List[str]`，需要删除的文档ID列表

返回值：`bool`，删除操作是否成功（始终返回 `True`）

#### 流程图

```mermaid
flowchart TD
    A[开始 del_doc_by_ids] --> B[将IDs列表转换为逗号分隔的字符串]
    B --> C[创建数据库会话]
    C --> D[构建DELETE SQL语句]
    D --> E[执行DELETE语句]
    E --> F[提交事务]
    F --> G[返回True]
```

#### 带注释源码

```
def del_doc_by_ids(self, ids: List[str]) -> bool:
    """
    根据文档ID列表删除文档
    
    参数:
        ids: 文档ID列表
    
    返回:
        bool: 删除操作是否成功
    """
    # 将IDs列表转换为逗号分隔的字符串，供SQL语句使用
    ids_str = ", ".join([f"{id}" for id in ids])
    
    # 使用SQLAlchemy会话管理数据库连接
    with Session(self.engine) as session:
        # 构建DELETE SQL语句，从集合表中删除指定ID的记录
        # 注意：这里使用了参数化查询来防止SQL注入
        stmt = text(f"DELETE FROM collection_{self.kb_name} WHERE id in (:ids)")
        
        # 执行删除操作，将IDs字符串作为参数传入
        session.execute(stmt, {"ids": ids_str})
        
        # 提交事务，确保删除操作被永久保存到数据库
        session.commit()
    
    # 返回True表示删除操作执行成功
    return True
```



### `RelytKBService.do_init`

该方法是 `RelytKBService` 类的初始化方法，用于初始化知识库服务。它依次调用内部方法 `_load_relyt_vector()` 来加载 Relyt 向量数据库连接和嵌入模型，以及 `do_create_kb()` 来创建知识库的向量索引。

参数：

- 无显式参数（仅包含 `self` 实例参数）

返回值：`None`，该方法执行完成后不返回任何值

#### 流程图

```mermaid
flowchart TD
    A[do_init 方法开始] --> B[调用 _load_relyt_vector]
    B --> B1[创建 EmbeddingsFunAdapter 适配器]
    B1 --> B2[获取示例嵌入向量确定维度]
    B2 --> B3[初始化 PGVecto_rs 向量存储]
    B3 --> B4[创建 SQLAlchemy 引擎]
    B4 --> C[调用 do_create_kb]
    C --> C1[构建索引名称 idx_{kb_name}_embedding]
    C1 --> C2[查询 pg_indexes 验证索引是否存在]
    C2 --> C3{索引已存在?}
    C3 -->|是| D[不执行创建]
    C3 -->|否| E[执行 CREATE INDEX 语句]
    E --> F[设置 HNSW 索引参数: m=30, ef_construction=500]
    E --> G[设置优化参数: threads=30, segment sizes]
    G --> D
    D --> H[do_init 方法结束]
```

#### 带注释源码

```python
def do_init(self):
    """
    初始化知识库服务。
    
    该方法执行两个关键步骤：
    1. 加载 Relyt 向量数据库连接和嵌入模型
    2. 创建知识库的向量索引（如果索引不存在）
    """
    
    # 步骤1：加载 Relyt 向量数据库
    # - 创建嵌入函数适配器
    # - 初始化 PGVecto_rs 向量存储实例
    # - 创建 SQLAlchemy 引擎用于直接数据库操作
    self._load_relyt_vector()
    
    # 步骤2：创建知识库索引
    # - 检查索引是否已存在
    # - 如果不存在，创建基于 HNSW 算法的向量索引
    # - 配置索引优化参数和 HNSW 参数
    self.do_create_kb()
```



### `RelytKBService.do_create_kb`

该方法用于在 Relyt 向量数据库中创建知识库的向量索引，通过检查索引是否已存在，若不存在则使用指定的 HNSW 索引参数和优化配置创建新的索引，以支持高效的向量相似性搜索。

参数：

- 该方法无显式参数（除隐式 self）

返回值：`None`，该方法直接操作数据库，不返回任何值

#### 流程图

```mermaid
flowchart TD
    A[开始 do_create_kb] --> B[构建索引名称: idx_{kb_name}_embedding]
    B --> C[使用 engine 建立数据库连接]
    C --> D[开启事务]
    D --> E[执行查询: 检查 pg_indexes 中是否存在该索引]
    E --> F{索引是否存在?}
    F -->|是| G[不执行任何操作]
    F -->|否| H[构建 CREATE INDEX 语句]
    H --> I[设置优化参数: optimizing_threads=30, max_growing_segment_size=2000, max_sealed_segment_size=30000000]
    I --> J[设置 HNSW 索引参数: m=30, ef_construction=500]
    J --> K[执行 CREATE INDEX 语句]
    K --> G
    G --> L[结束]
```

#### 带注释源码

```python
def do_create_kb(self):
    """
    创建知识库的向量索引
    
    该方法执行以下操作：
    1. 检查指定名称的索引是否已存在于数据库中
    2. 如果不存在，则创建一个使用 HNSW 算法的向量索引
    3. 索引配置包含性能优化参数和 HNSW 参数
    """
    # 构建索引名称，格式为 idx_{知识库名称}_embedding
    index_name = f"idx_{self.kb_name}_embedding"
    
    # 使用 SQLAlchemy engine 建立数据库连接
    with self.engine.connect() as conn:
        # 开启事务，确保索引创建的原子性
        with conn.begin():
            # 查询系统表 pg_indexes 检查索引是否已存在
            index_query = text(
                f"""
                    SELECT 1
                    FROM pg_indexes
                    WHERE indexname = '{index_name}';
                """
            )
            # 执行查询并获取结果（scalar 返回第一行第一列，若无结果返回 None）
            result = conn.execute(index_query).scalar()
            
            # 如果索引不存在（result 为 None 或 0）
            if not result:
                # 构建创建索引的 SQL 语句
                # 使用 vectors 插件的 vector_l2_ops 操作符
                index_statement = text(
                    f"""
                        CREATE INDEX {index_name}
                        ON collection_{self.kb_name}
                        USING vectors (embedding vector_l2_ops)
                        WITH (options = $$
                        optimizing.optimizing_threads = 30
                        segment.max_growing_segment_size = 2000
                        segment.max_sealed_segment_size = 30000000
                        [indexing.hnsw]
                        m=30
                        ef_construction=500
                        $$);
                    """
                )
                # 执行创建索引的 SQL 语句
                # 注意：此处存在 SQL 注入风险，因为使用了 f-string 拼接知识库名称
                conn.execute(index_statement)
```



### `RelytKBService.vs_type`

该方法用于返回当前知识库服务所支持的向量存储类型标识，表明该服务使用 RELYT 作为向量存储后端。

参数： 无

返回值：`str`，返回 `SupportedVSType.RELYT` 枚举值对应的字符串表示，用于标识当前知识库服务使用 RELYT 类型的向量存储。

#### 流程图

```mermaid
flowchart TD
    A[开始] --> B{调用 vs_type 方法}
    B --> C[返回 SupportedVSType.RELYT]
    C --> D[结束]
    
    style B fill:#f9f,stroke:#333
    style C fill:#bbf,stroke:#333
```

#### 带注释源码

```python
def vs_type(self) -> str:
    """
    返回支持的向量存储类型
    
    该方法继承自 KBService 基类，用于标识当前知识库服务
    所使用的向量存储后端类型。在本实现中，返回 RELYT 类型。
    
    Returns:
        str: 向量存储类型标识，值为 SupportedVSType.RELYT
    """
    return SupportedVSType.RELYT
```

---

### 完整类信息补充

#### 类：`RelytKBService`

**描述**： RelytKBService 是基于 PGVecto_rs 的知识库服务实现类，封装了对 RELYT 向量数据库的 CRUD 操作，包括知识库创建、文档添加、文档删除、文档搜索等功能。

**继承关系**：
- 继承自 `KBService`（位于 `server.knowledge_base.kb_service.base`）

**类字段**：

| 字段名 | 类型 | 描述 |
|--------|------|------|
| `relyt` | `PGVecto_rs` | PGVecto_rs 向量存储实例，用于执行向量相似度搜索 |
| `engine` | `Engine` | SQLAlchemy 数据库引擎，用于执行原生 SQL 操作 |
| `kb_name` | `str` | 知识库名称（继承自 KBService） |
| `embed_model` | 任意 | 嵌入模型实例（继承自 KBService） |

**类方法**：

| 方法名 | 功能描述 |
|--------|----------|
| `_load_relyt_vector` | 初始化 PGVecto_rs 向量存储实例和 SQLAlchemy 引擎 |
| `get_doc_by_ids` | 根据文档 ID 列表批量查询文档 |
| `del_doc_by_ids` | 根据文档 ID 列表批量删除文档 |
| `do_init` | 初始化知识库服务，调用向量加载和知识库创建 |
| `do_create_kb` | 创建知识库表及向量索引 |
| `vs_type` | 返回支持的向量存储类型为 RELYT |
| `do_drop_kb` | 删除整个知识库表 |
| `do_search` | 执行相似度搜索并应用分数阈值过滤 |
| `do_add_doc` | 向知识库添加文档 |
| `do_delete_doc` | 根据文件路径删除文档 |
| `do_clear_vs` | 清空向量存储，等同于删除知识库 |

---

### 关键组件信息

| 组件名称 | 描述 |
|----------|------|
| `PGVecto_rs` | LangChain 社区版的 PGVecto_rs 向量数据库封装类，提供向量存储和相似度搜索能力 |
| `SupportedVSType` | 枚举类型，定义系统支持的向量存储类型（如 RELYT、Chroma、Milvus 等） |
| `EmbeddingsFunAdapter` | 嵌入函数适配器，将模型封装为 LangChain 兼容的嵌入接口 |
| `KnowledgeFile` | 知识文件对象，包含文件路径和元数据信息 |

---

### 潜在技术债务与优化空间

1. **SQL 注入风险**：`do_create_kb`、`get_doc_by_ids`、`del_doc_by_ids`、`do_delete_doc` 等方法中使用了字符串拼接构造 SQL 语句（如 `f"SELECT ... WHERE id in (:ids)"`），存在潜在的 SQL 注入风险。建议使用参数化查询或严格的输入校验。

2. **硬编码配置**：`do_create_kb` 中索引创建的参数（如 `optimizing_threads = 30`、`max_sealed_segment_size = 30000000` 等）是硬编码的，应考虑提取到配置文件或知识库级别设置中。

3. **调试打印语句**：`do_add_doc` 方法中使用了 `print(docs)` 和 `print(ids)` 进行调试输出，生产环境应移除或替换为日志记录。

4. **错误处理缺失**：多个方法（如 `do_search`、`do_add_doc` 等）缺少异常捕获机制，当数据库连接失败或查询超时时可能导致服务崩溃。

5. **资源未自动释放**：`Session` 对象虽使用 `with` 语句管理，但 `engine` 对象在类实例生命周期内持续存在，未提供显式的资源释放或连接池清理接口。

---

### 其它项目说明

#### 设计目标与约束

- **设计目标**：提供统一的知识库服务抽象接口，支持多种向量存储后端，RELYT 实现侧重于 PostgreSQL 向量扩展的集成。
- **约束**：依赖 PGVecto_rs 库和 PostgreSQL 数据库，必须保证数据库连接可用；嵌入模型维度必须与 PGVecto_rs 初始化时的 `dimension` 参数一致。

#### 错误处理与异常设计

- 当前实现未捕获数据库异常（如连接超时、索引不存在等），建议在关键操作（`do_create_kb`、`do_search`、`do_add_doc` 等）外层添加 `try-except` 块，捕获 `SQLAlchemyError` 及其子类异常。
- `vs_type` 方法本身无外部依赖，理论上不会抛出异常，但若 `SupportedVSType` 枚举定义缺失会导致导入错误。

#### 数据流与状态机

- **初始化流程**：`do_init` → `_load_relyt_vector` → `do_create_kb`
- **文档添加流程**：`do_add_doc` → `PGVecto_rs.add_documents`（向量入库） + 元数据同步到 `collection_{kb_name}` 表
- **文档删除流程**：`do_delete_doc` → SQL DELETE（仅删除元数据记录，向量由 PGVecto_rs 内部管理）
- **搜索流程**：`do_search` → `PGVecto_rs.similarity_search_with_score` → `score_threshold_process` 过滤

#### 外部依赖与接口契约

- **外部依赖**：
  - `PGVecto_rs`（LangChain 社区向量存储）
  - `kbs_config`（知识库配置，需包含 `relyt.connection_uri`）
  - `SupportedVSType`（枚举定义）
- **接口契约**：
  - `vs_type` 方法必须返回字符串类型，且返回值应与 `SupportedVSType` 枚举值一致
  - 继承自 `KBService` 的方法需遵循基类定义的签名和语义



### `RelytKBService.do_drop_kb`

该方法用于删除知识库对应的数据库表，通过构造并执行 `DROP TABLE IF EXISTS` SQL 语句，永久删除知识库中的所有向量数据集合。

参数：
- 无（仅包含隐式参数 `self`）

返回值：`None`，该方法执行后不返回任何值

#### 流程图

```mermaid
flowchart TD
    A[开始 do_drop_kb] --> B[构建 DROP TABLE 语句]
    B --> C[使用 engine 建立数据库连接]
    C --> D[开启事务]
    D --> E[执行 DROP TABLE 语句]
    E --> F[事务提交/回滚]
    F --> G[结束]
    
    style A fill:#e1f5fe
    style G fill:#e1f5fe
    style E fill:#fff3e0
```

#### 带注释源码

```python
def do_drop_kb(self):
    """
    删除知识库对应的数据库表
    
    该方法通过执行 DROP TABLE IF EXISTS 语句，
    删除与当前知识库关联的所有向量数据集合。
    """
    # 构建 DROP TABLE 语句，使用 IF EXISTS 确保即使表不存在也不会报错
    # collection_{self.kb_name} 是知识库对应的表名
    drop_statement = text(f"DROP TABLE IF EXISTS collection_{self.kb_name};")
    
    # 使用 SQLAlchemy engine 建立数据库连接
    with self.engine.connect() as conn:
        # 开启事务，确保操作的原子性
        with conn.begin():
            # 执行删除表的 SQL 语句
            conn.execute(drop_statement)
```



### `RelytKBService.do_search`

该方法执行向量相似度搜索，通过调用底层的 PGVecto_rs 向量数据库进行相似度查询，并根据分数阈值过滤结果，最终返回最相关的 Top-K 文档列表。

参数：

- `query`：`str`，用户输入的搜索查询文本
- `top_k`：`int`，返回最相似的文档数量
- `score_threshold`：`float`，用于过滤低相似度结果的分值阈值

返回值：`List[Document]`，经过分数阈值过滤后的相似文档列表

#### 流程图

```mermaid
flowchart TD
    A[开始 do_search] --> B[调用 self.relyt.similarity_search_with_score]
    B --> C[传入 query 和 top_k 参数]
    C --> D[执行向量相似度搜索]
    D --> E[返回原始搜索结果 docs]
    E --> F[调用 score_threshold_process 函数]
    F --> G[传入 score_threshold, top_k, docs]
    G --> H[根据阈值过滤低分文档]
    H --> I[返回过滤后的文档列表]
    I --> J[结束 do_search]
```

#### 带注释源码

```python
def do_search(self, query: str, top_k: int, score_threshold: float):
    """
    执行向量相似度搜索
    
    参数:
        query: str - 用户输入的搜索查询文本
        top_k: int - 返回最相似的文档数量
        score_threshold: float - 分值阈值，低于此阈值的结果将被过滤
    
    返回:
        List[Document] - 经过分数阈值过滤后的相似文档列表
    """
    # 调用 PGVecto_rs 向量数据库的相似度搜索方法
    # 返回包含相似文档及其分数的原始结果
    docs = self.relyt.similarity_search_with_score(query, top_k)
    
    # 使用 score_threshold_process 函数对结果进行后处理
    # 根据 score_threshold 过滤掉低相似度的文档
    # 同时确保返回的文档数量不超过 top_k
    return score_threshold_process(score_threshold, top_k, docs)
```



### `RelytKBService.do_add_doc`

该方法用于将文档添加到 Relyt 向量存储中，通过调用 PGVecto_rs 的 add_documents 方法将文档向量化和存储，并返回包含文档ID和元数据的信息列表。

参数：

- `docs`：`List[Document]`（langchain 的 Document 对象列表），要添加到向量存储的文档列表
- `**kwargs`：`Any`，可变关键字参数，用于接收额外的配置参数

返回值：`List[Dict]`，返回包含每个文档 ID 和元数据的字典列表，格式为 `[{"id": str, "metadata": dict}, ...]`

#### 流程图

```mermaid
flowchart TD
    A[开始 do_add_doc] --> B[打印输入文档 docs]
    B --> C[调用 self.relyt.add_documents 添加文档到向量存储]
    C --> D[获取返回的文档 IDs 列表]
    D --> E[打印 IDs]
    E --> F[使用 zip 将 IDs 与原始文档配对]
    F --> G[构建 doc_infos 列表]
    G --> H[返回 doc_infos]
    
    style A fill:#e1f5fe
    style H fill:#e8f5e8
```

#### 带注释源码

```python
def do_add_doc(self, docs: List[Document], **kwargs) -> List[Dict]:
    """
    将文档添加到 Relyt 向量存储中
    
    参数:
        docs: Document 对象列表，每个包含 page_content 和 metadata
        **kwargs: 额外的关键字参数
    
    返回:
        包含文档ID和元数据的字典列表
    """
    # 打印输入的文档列表，便于调试和日志记录
    print(docs)
    
    # 调用 PGVecto_rs 的 add_documents 方法进行向量化和存储
    # 该方法会:
    # 1. 对每个 Document 的 page_content 进行 embedding
    # 2. 将向量和元数据存储到 PostgreSQL 向量数据库中
    ids = self.relyt.add_documents(docs)
    
    # 打印返回的文档 ID 列表，便于调试
    print(ids)
    
    # 构建返回的文档信息列表
    # 使用 zip 将 IDs 与原始文档配对，提取每个文档的 ID 和 metadata
    doc_infos = [
        {"id": id, "metadata": doc.metadata} 
        for id, doc in zip(ids, docs)
    ]
    
    # 返回文档信息列表
    return doc_infos
```



### `RelytKBService.do_delete_doc`

该方法用于从Relyt知识库中删除指定文件路径的文档记录，通过构建SQL删除语句并使用数据库会话执行事务来完成文档的物理删除。

**参数：**

- `kb_file`：`KnowledgeFile`，待删除的文档文件对象，包含文件的路径信息
- `**kwargs`：可变关键字参数，用于接收额外的可选参数（当前未使用）

**返回值：** `None`，该方法没有返回值，执行完成后直接结束

#### 流程图

```mermaid
flowchart TD
    A[开始 do_delete_doc] --> B[获取文件相对路径]
    B --> C[调用 get_relative_source_path]
    C --> D[构建 DELETE SQL 语句]
    D --> E[创建数据库会话 Session]
    E --> F[执行 SQL 删除语句]
    F --> G[提交事务 commit]
    G --> H[结束]
```

#### 带注释源码

```python
def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
    """
    从Relyt知识库中删除指定文件的文档记录
    
    参数:
        kb_file: KnowledgeFile对象，包含待删除文件的路径信息
        **kwargs: 额外的可选参数
    """
    # 获取文件相对于知识库的相对路径
    # 例如: 将完整路径转换为知识库内部的相对路径标识
    filepath = self.get_relative_source_path(kb_file.filepath)
    
    # 构建SQL删除语句
    # 从collection_{kb_name}表中删除meta->>'source'字段等于filepath的记录
    # meta->>'source' 是JSONB字段提取操作，获取meta字段中source键的值
    stmt = f"DELETE FROM collection_{self.kb_name} WHERE meta->>'source'='{filepath}'; "
    
    # 使用SQLAlchemy Session管理数据库连接和事务
    # Session确保自动关闭连接和资源释放
    with Session(self.engine) as session:
        # 执行SQL语句（需转换为text对象以防止SQL注入警告，但此处存在字符串拼接风险）
        session.execute(text(stmt))
        # 提交事务使删除操作生效
        session.commit()
```



### `RelytKBService.do_clear_vs`

该方法用于清除向量存储，通过调用`do_drop_kb`方法删除知识库对应的数据库表来实现向量存储的清理。

参数：暂无参数

返回值：`None`，无返回值描述

#### 流程图

```mermaid
flowchart TD
    A[开始 do_clear_vs] --> B[调用 self.do_drop_kb]
    B --> C[执行 DROP TABLE IF EXISTS collection_{kb_name}]
    C --> D[结束]
```

#### 带注释源码

```python
def do_clear_vs(self):
    """
    清除向量存储。
    该方法通过调用 do_drop_kb 删除知识库对应的数据库表来清除向量存储。
    """
    self.do_drop_kb()  # 调用 do_drop_kb 方法删除知识库表
```

## 关键组件





### 向量索引与 HNSW 优化

在 `do_create_kb` 方法中，使用 HNSW（Hierarchical Navigable Small World）算法创建向量索引，配置了 m=30、ef_construction=500 等参数，并通过 `optimizing_threads = 30` 等配置优化索引性能，实现高效的近似最近邻搜索。

### 惰性加载机制

`_load_relyt_vector` 方法采用惰性加载模式，仅在首次需要向量功能时才初始化 PGVecto_rs 向量存储和 SQLAlchemy 引擎，避免服务启动时的资源浪费。

### 嵌入模型适配

通过 `EmbeddingsFunAdapter` 将嵌入模型包装为 langchain 兼容的嵌入函数，并使用 "Hello relyt!" 作为样本进行维度探测，确保向量维度与模型输出匹配。

### 知识库索引管理

`do_create_kb` 方法实现了索引的自动创建与存在性检查，通过查询 `pg_indexes` 系统表判断索引是否已存在，避免重复创建带来的性能开销。

### 文档元数据过滤删除

`do_delete_doc` 方法使用 PostgreSQL 的 JSONB 操作 `meta->>'source'` 实现基于文件路径的精确文档删除，通过相对路径匹配确保删除操作的准确性。

### 向量搜索与分数阈值处理

`do_search` 方法封装了 PGVecto_rs 的相似度搜索，并通过 `score_threshold_process` 函数对结果进行后处理，支持基于分数阈值的搜索结果过滤。

### 批量文档操作

`do_add_doc` 方法实现批量文档添加，返回包含文档 ID 和元数据的列表，支持与外部文档管理系统的集成。



## 问题及建议



### 已知问题

-   **SQL注入风险（高危）**：多处直接使用 f-string 拼接 SQL 语句，包括表名 `collection_{self.kb_name}`、索引名 `{index_name}`、文件路径 `{filepath}` 等，未进行任何校验或使用参数化查询，可能导致 SQL 注入攻击
-   **参数化查询使用不当**：`ids_str` 变量虽然声明了参数 `{"ids": ids_str}`，但 SQL 中使用 `:ids` 占位符时，字符串拼接的 `ids_str` 会被当作整体字符串而非列表参数，导致查询逻辑错误
-   **调试代码未清理**：`do_add_doc` 方法中存在 `print(docs)` 和 `print(ids)` 语句，生产环境应移除
-   **缺少异常处理与事务管理**：数据库操作（特别是 `do_create_kb`、`do_drop_kb`、`do_delete_doc`）缺少 try-except 捕获和事务回滚机制，连接异常时可能导致资源泄漏或数据不一致
-   **连接资源未显式释放**：`self.engine` 和 `self.relyt` 在类初始化后未提供显式的关闭/清理方法，可能导致数据库连接泄漏
-   **索引配置硬编码**：`do_create_kb` 中的索引参数（optimizing_threads=30、segment sizes、HNSW 参数 m=30 等）直接硬编码，缺乏配置化管理和调优能力

### 优化建议

-   **修复 SQL 注入**：对 `kb_name`、`index_name`、`filepath` 等用户可控的字符串进行严格的校验（仅允许字母、数字、下划线），或使用白名单机制；优先使用参数化查询
-   **修复参数化查询逻辑**：如需查询 IN 子句，应将 ids 拆分为多个参数占位符（如 `:id1, :id2`）或使用 PostgreSQL 的 `ANY` 数组函数
-   **移除调试代码**：删除 `do_add_doc` 中的所有 print 语句
-   **完善异常处理**：为所有数据库操作添加 try-except-finally 块，确保事务在异常时正确回滚；使用上下文管理器管理资源
-   **添加资源清理方法**：实现 `__del__` 或提供 `close()` 方法显式关闭 engine 和依赖连接
-   **配置化管理**：将索引参数提取到 `configs` 或环境变量中，支持运行时配置
-   **添加类型注解完善**：为 `do_add_doc` 的返回值添加明确的 Dict 类型定义，增强代码可读性

## 其它





### 设计目标与约束

本模块旨在为Relyt向量数据库提供知识库服务能力，支持知识库的创建、文档向量添加、相似性搜索、文档删除及知识库删除等核心功能。设计约束包括：依赖PGVecto_rs作为向量存储后端，仅支持通过SQLAlchemy进行数据库操作，要求embedding维度在初始化时确定，且当前实现主要面向单知识库场景。

### 错误处理与异常设计

代码中错误处理较为薄弱，存在多处潜在风险。`get_doc_by_ids`和`del_doc_by_ids`方法直接使用字符串拼接构建SQL语句，虽然通过参数化查询防止了注入，但未对空列表输入进行校验。数据库连接操作未显式处理连接失败场景，Session上下文管理器虽能自动释放资源，但未捕获SQL执行异常。建议在关键方法中添加try-except块，对数据库连接超时、查询超时、索引创建失败等情况进行统一捕获和日志记录。

### 数据流与状态机

知识库服务生命周期包含初始化状态（`do_init`）、就绪状态、文档操作状态（添加/删除/搜索）、以及销毁状态（`do_drop_kb`）。数据流向为：外部调用→KBService接口→RelytKBService实现→PGVecto_rs向量存储/PostgreSQL关系存储。文档添加时首先进入向量库，然后元数据同步至PostgreSQL表；搜索时先由向量库返回候选结果，再经score_threshold_process过滤。

### 外部依赖与接口契约

核心依赖包括：`kbs_config.get("relyt").get("connection_uri")`提供数据库连接字符串；`EmbeddingsFunAdapter`将嵌入模型封装为LangChain兼容接口；`PGVecto_rs`提供向量存储抽象；`KnowledgeFile`定义知识文件模型。接口契约方面，`do_add_doc`返回List[Dict]包含id和metadata；`do_search`返回经过score_threshold处理后的Document列表；`vs_type`返回SupportedVSType.RELYT标识向量类型。

### 性能考虑与优化空间

当前实现存在多项性能瓶颈：`do_create_kb`中索引创建仅检查索引是否存在，未考虑表是否存在；批量操作未使用连接池优化；`do_add_doc`中print语句会造成IO开销；相似性搜索默认返回所有结果后再进行阈值过滤，应考虑在数据库层面预过滤。优化方向包括：引入连接池、批量文档处理、异步索引创建、搜索结果在数据库层过滤。

### 安全性考虑

代码使用SQLAlchemy的Session和text()执行原始SQL，存在一定SQL注入风险（虽然使用了参数化查询）。数据库连接凭证从配置文件读取，应确保配置文件权限控制。`do_delete_doc`中使用字符串拼接构建filepath条件，存在安全隐患。建议使用参数化查询替代字符串拼接，并对用户输入进行严格校验。

### 配置管理

配置通过`kbs_config`全局配置对象获取，键名为"relyt"，包含connection_uri连接串。知识库名称`self.kb_name`作为集合名称直接拼接至SQL和索引名，需确保命名符合PostgreSQL标识符规范。embedding维度在首次初始化时通过sample_embedding动态获取，要求embed_model必须支持"Hello relyt!"这个查询的向量化。

### 版本兼容性

代码依赖langchain_community.vectorstores.pgvecto_rs的PGVecto_rs类，该库版本兼容性需关注。SQL语句中使用vectors扩展的特定语法（vector_l2_ops、HNSW参数），需要PostgreSQL已安装vectors扩展。当前代码仅支持L2距离度量，若需支持余弦相似度等其它度量方式需修改do_create_kb中的索引定义。

### 测试策略建议

建议补充单元测试覆盖：空输入边界条件、数据库连接失败场景、知识库重复创建处理、文档批量添加/删除、搜索结果准确性验证。集成测试应覆盖完整的知识库生命周期：创建→添加文档→搜索→删除文档→删除知识库。


    
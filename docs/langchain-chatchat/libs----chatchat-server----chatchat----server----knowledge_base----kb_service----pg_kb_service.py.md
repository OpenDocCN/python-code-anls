
# `Langchain-Chatchat\libs\chatchat-server\chatchat\server\knowledge_base\kb_service\pg_kb_service.py` 详细设计文档

该代码实现了一个基于PostgreSQL pgvector扩展的向量知识库服务类(PGKBService)，用于RAG系统中文档的存储、检索和管理，支持向量相似度搜索和知识库的全生命周期管理。

## 整体流程

```mermaid
graph TD
A[开始] --> B[PGKBService初始化]
B --> C[调用do_init加载pg_vector]
C --> D{操作类型}
D --> E[do_create_kb: 创建知识库]
D --> F[do_add_doc: 添加文档]
D --> G[do_search: 搜索文档]
D --> H[do_delete_doc: 删除文档]
D --> I[do_drop_kb: 删除知识库]
D --> J[do_clear_vs: 清空向量存储]
E --> K[返回执行结果]
F --> L[调用pg_vector.add_documents]
L --> K
G --> M[创建Retriever并搜索]
M --> K
H --> N[执行SQL删除相关记录]
N --> K
I --> O[删除pg_embedding和pg_collection表记录]
O --> P[删除文件系统目录]
P --> K
J --> Q[删除并重建collection]
Q --> K
```

## 类结构

```
KBService (抽象基类)
└── PGKBService (PostgreSQL向量库服务实现)
```

## 全局变量及字段


### `Settings`
    
全局设置对象，用于获取知识库配置

类型：`Settings`
    


### `KBService`
    
知识库服务基类，提供向量存储的抽象接口

类型：`class`
    


### `SupportedVSType`
    
支持的向量存储类型枚举

类型：`enum`
    


### `PGVector`
    
LangChain PGVector向量存储类，提供向量存储和检索功能

类型：`class`
    


### `DistanceStrategy`
    
距离策略枚举，用于计算向量相似度

类型：`enum`
    


### `get_Embeddings`
    
获取嵌入函数的工具函数

类型：`function`
    


### `get_Retriever`
    
获取检索器的工具函数

类型：`function`
    


### `KnowledgeFile`
    
知识文件类，表示知识库中的单个文件

类型：`class`
    


### `PGKBService.engine`
    
SQLAlchemy数据库引擎，用于PostgreSQL连接

类型：`Engine`
    


### `PGKBService.pg_vector`
    
pgvector向量存储实例

类型：`PGVector`
    


### `PGKBService.kb_name`
    
知识库名称（继承自KBService）

类型：`str`
    


### `PGKBService.embed_model`
    
嵌入模型名称（继承自KBService）

类型：`str`
    


### `PGKBService.kb_path`
    
知识库文件路径（继承自KBService）

类型：`str`
    
    

## 全局函数及方法



### `PGKBService`

这是一个基于 PostgreSQL 的知识库向量存储服务实现类，继承自 `KBService` 基类。通过 PGVector 扩展实现向量存储功能，支持文档的添加、删除、搜索和知识库的创建、删除等操作，利用 SQLAlchemy 进行数据库交互，并通过 LangChain 的向量检索器实现语义搜索。

参数：

- 无（构造函数参数在基类中定义）

返回值：无

#### 流程图

```mermaid
graph TD
    A[开始] --> B[初始化 engine: PostgreSQL 连接池]
    B --> C[加载 PGVector 向量存储]
    C --> D[准备就绪]
```

#### 带注释源码

```python
class PGKBService(KBService):
    # 类级别属性：创建 PostgreSQL 数据库引擎，配置连接池大小为 10
    engine: Engine = sqlalchemy.create_engine(
        Settings.kb_settings.kbs_config.get("pg").get("connection_uri"), pool_size=10
    )
```

---

### `PGKBService._load_pg_vector`

初始化并加载 PGVector 向量存储实例，配置嵌入模型、集合名称、距离策略和数据库连接。

参数：

- 无

返回值：无

#### 流程图

```mermaid
graph TD
    A[开始] --> B[获取嵌入模型]
    B --> C[构建 PGVector 实例]
    C --> D[设置 self.pg_vector]
```

#### 带注释源码

```python
def _load_pg_vector(self):
    # 加载 PGVector 向量存储，配置参数包括：
    # - embedding_function: 获取嵌入模型
    # - collection_name: 知识库名称
    # - distance_strategy: 使用欧几里得距离
    # - connection: 使用类级别数据库引擎
    # - connection_string: 数据库连接字符串
    self.pg_vector = PGVector(
        embedding_function=get_Embeddings(self.embed_model),
        collection_name=self.kb_name,
        distance_strategy=DistanceStrategy.EUCLIDEAN,
        connection=PGKBService.engine,
        connection_string=Settings.kb_settings.kbs_config.get("pg").get("connection_uri"),
    )
```

---

### `PGKBService.get_doc_by_ids`

根据文档 IDs 从向量数据库中检索对应的文档内容及元数据。

参数：

- `ids`：`List[str]`，文档唯一标识符列表

返回值：`List[Document]`，返回 LangChain Document 对象列表，每个包含 page_content 和 metadata

#### 流程图

```mermaid
graph TD
    A[开始] --> B[创建数据库会话]
    B --> C[构建 SQL 查询语句]
    C --> D[执行查询并获取结果]
    D --> E[遍历结果转换为 Document 对象]
    E --> F[返回文档列表]
```

#### 带注释源码

```python
def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
    # 使用 SQLAlchemy 会话上下文管理器
    with Session(PGKBService.engine) as session:
        # 构建原生 SQL 查询，从 langchain_pg_embedding 表获取文档内容
        # custom_id 字段支持数组查询，使用 ANY 函数匹配多个 ID
        stmt = text(
            "SELECT document, cmetadata FROM langchain_pg_embedding WHERE custom_id = ANY(:ids)"
        )
        # 执行查询并获取所有匹配结果
        results = [
            # 将每行数据转换为 LangChain Document 对象
            # page_content = 文档内容, metadata = 元数据
            Document(page_content=row[0], metadata=row[1])
            for row in session.execute(stmt, {"ids": ids}).fetchall()
        ]
        return results
```

---

### `PGKBService.del_doc_by_ids`

根据文档 IDs 删除向量数据库中的对应文档记录。

参数：

- `ids`：`List[str]`，需要删除的文档唯一标识符列表

返回值：`bool`，返回基类删除操作的结果

#### 流程图

```mermaid
graph TD
    A[开始] --> B[调用父类 del_doc_by_ids 方法]
```

#### 带注释源码

```python
def del_doc_by_ids(self, ids: List[str]) -> bool:
    # 调用父类 KBService 的删除方法，实现委托
    return super().del_doc_by_ids(ids)
```

---

### `PGKBService.do_init`

初始化知识库服务，加载 PGVector 向量存储实例。

参数：

- 无

返回值：无

#### 流程图

```mermaid
graph TD
    A[开始] --> B[调用 _load_pg_vector]
```

#### 带注释源码

```python
def do_init(self):
    # 初始化时加载 PGVector 向量存储
    self._load_pg_vector()
```

---

### `PGKBService.do_create_kb`

创建知识库，当前实现为空操作（知识库由 PGVector 自动管理）。

参数：

- 无

返回值：无

#### 流程图

```mermaid
graph TD
    A[开始] --> B[空操作]
```

#### 带注释源码

```python
def do_create_kb(self):
    # PGVector 在添加文档时会自动创建 collection
    # 此处无需额外操作
    pass
```

---

### `PGKBService.vs_type`

返回当前向量存储类型标识。

参数：

- 无

返回值：`str`，返回支持的向量存储类型枚举值

#### 流程图

```mermaid
graph TD
    A[开始] --> B[返回 SupportedVSType.PG]
```

#### 带注释源码

```python
def vs_type(self) -> str:
    # 返回枚举类型的字符串表示，表示使用 PostgreSQL PGVector
    return SupportedVSType.PG
```

---

### `PGKBService.do_drop_kb`

删除整个知识库，包括向量数据、集合记录及本地文件目录。

参数：

- 无

返回值：无

#### 流程图

```mermaid
graph TD
    A[开始] --> B[创建数据库会话]
    B --> C[删除 langchain_pg_embedding 表中关联记录]
    C --> D[删除 langchain_pg_collection 表中记录]
    E --> F[提交事务]
    F --> G[删除本地文件目录]
```

#### 带注释源码

```python
def do_drop_kb(self):
    # 使用会话上下文执行数据库操作
    with Session(PGKBService.engine) as session:
        # 执行原生 SQL：
        # 1. 删除 langchain_pg_embedding 表中关联到指定 collection 的所有向量记录
        # 2. 删除 langchain_pg_collection 表中对应的 collection 记录
        session.execute(
            text(
                f"""
                -- 删除 langchain_pg_embedding 表中关联到 langchain_pg_collection 表中 的记录
                DELETE FROM langchain_pg_embedding
                WHERE collection_id IN (
                  SELECT uuid FROM langchain_pg_collection WHERE name = '{self.kb_name}'
                );
                -- 删除 langchain_pg_collection 表中 记录
                DELETE FROM langchain_pg_collection WHERE name = '{self.kb_name}';
        """
            )
        )
        # 提交事务确保删除操作生效
        session.commit()
        # 删除本地文件系统中的知识库目录
        shutil.rmtree(self.kb_path)
```

---

### `PGKBService.do_search`

执行向量语义检索，返回与查询最相关的文档列表。

参数：

- `query`：`str`，用户查询字符串
- `top_k`：`int`，返回最相关的 K 个文档数量
- `score_threshold`：`float`，相似度分数阈值

返回值：`List[Document]`，返回符合阈值的相关文档列表

#### 流程图

```mermaid
graph TD
    A[开始] --> B[获取向量存储检索器]
    B --> C[配置 top_k 和 score_threshold]
    C --> D[执行相似度搜索]
    D --> E[返回文档列表]
```

#### 带注释源码

```python
def do_search(self, query: str, top_k: int, score_threshold: float):
    # 从向量存储创建检索器
    retriever = get_Retriever("vectorstore").from_vectorstore(
        self.pg_vector,      # PGVector 向量存储实例
        top_k=top_k,         # 返回前 K 个最相似文档
        score_threshold=score_threshold,  # 设置相似度阈值过滤
    )
    # 执行语义检索，获取相关文档
    docs = retriever.get_relevant_documents(query)
    return docs
```

---

### `PGKBService.do_add_doc`

将文档列表添加到向量存储中，并返回文档的 ID 和元数据信息。

参数：

- `docs`：`List[Document]`，待添加的 LangChain Document 列表
- `**kwargs`：可选关键字参数

返回值：`List[Dict]`，返回包含每个文档 ID 和元数据的字典列表

#### 流程图

```mermaid
graph TD
    A[开始] --> B[调用 pg_vector.add_documents 添加文档]
    B --> C[获取返回的 IDs 列表]
    C --> D[遍历 IDs 和文档构建 doc_infos]
    D --> E[返回 doc_infos 列表]
```

#### 带注释源码

```python
def do_add_doc(self, docs: List[Document], **kwargs) -> List[Dict]:
    # 向 PGVector 添加文档，获取生成的文档 ID 列表
    ids = self.pg_vector.add_documents(docs)
    # 构建返回信息列表，将 ID 与文档元数据配对
    # 格式: [{"id": "xxx", "metadata": {...}}, ...]
    doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
    return doc_infos
```

---

### `PGKBService.do_delete_doc`

根据知识文件删除向量数据库中对应的文档记录。

参数：

- `kb_file`：`KnowledgeFile`，知识库文件对象，包含文件路径和知识库名称
- `**kwargs`：可选关键字参数

返回值：无

#### 流程图

```mermaid
graph TD
    A[开始] --> B[构建查询获取 collection_id]
    B --> C[执行查询获取 UUID]
    C --> D[构建删除语句]
    D --> E[使用 JSONB 查询条件删除]
    E --> F[提交事务]
```

#### 带注释源码

```python
def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
    # 构建查询语句：获取指定知识库名称的 collection UUID
    select_query = text("SELECT uuid FROM langchain_pg_collection WHERE name = :name;")
    # 构建删除语句：使用 JSONB 操作符 @> 查询
    # cmetadata::jsonb 将元数据列转换为 JSONB 类型
    # @> 为包含查询操作符
    delete_query = text("""
        DELETE FROM langchain_pg_embedding
        WHERE cmetadata::jsonb @> :cmetadata
        AND collection_id = :collection_id;
    """)
    # 使用会话执行数据库操作
    with Session(PGKBService.engine) as session:
        # 获取 collection 的 UUID
        collection_id = session.execute(select_query, {"name": kb_file.kb_name}).fetchone()[0]
        # 执行删除操作，传入元数据条件和 collection_id
        session.execute(
            delete_query, 
            {
                # 格式化元数据 JSON 字符串，用于匹配 source 字段
                "cmetadata": '{"source": "%s"}' % self.get_relative_source_path(kb_file.filepath),
                "collection_id": collection_id
            }
        )
        # 提交事务确保删除生效
        session.commit()
```

---

### `PGKBService.do_clear_vs`

清空向量存储中的所有数据，先删除 collection 再重新创建。

参数：

- 无

返回值：无

#### 流程图

```mermaid
graph TD
    A[开始] --> B[调用 delete_collection 删除集合]
    B --> C[调用 create_collection 重新创建]
```

#### 带注释源码

```python
def do_clear_vs(self):
    # 删除当前的向量集合
    self.pg_vector.delete_collection()
    # 重新创建空的向量集合
    self.pg_vector.create_collection()
```

---

## 关键组件信息

| 组件名称 | 一句话描述 |
|---------|-----------|
| `engine` | PostgreSQL 数据库连接引擎，使用连接池管理 |
| `pg_vector` | PGVector 向量存储实例，用于向量检索 |
| `KBService` | 知识库服务抽象基类，定义标准接口 |
| `PGVector` | LangChain 的 PostgreSQL 向量存储实现 |

---

## 潜在的技术债务或优化空间

1. **SQL 注入风险**：`do_drop_kb` 方法中直接使用 f-string 拼接知识库名称，应使用参数化查询
2. **硬编码连接字符串**：多次调用 `Settings.kb_settings.kbs_config.get("pg").get("connection_uri")`，可提取为类常量
3. **错误处理缺失**：缺少对数据库连接失败、查询异常的处理
4. **资源泄漏风险**：`get_doc_by_ids` 和 `do_search` 中未显式关闭会话（虽使用上下文管理器但仍需验证）
5. **索引优化**：大量数据时需考虑在 `custom_id`、`cmetadata` 字段建立索引

---

## 其它项目

### 设计目标与约束

- 基于 PostgreSQL + PGVector 实现向量存储
- 遵循 KBService 抽象接口设计
- 使用 SQLAlchemy 进行数据库操作
- 支持欧几里得距离计算

### 错误处理与异常设计

- 数据库操作异常未捕获，需上层调用方处理
- 文件删除失败未处理 IO 异常
- collection 不存在时的行为依赖 PGVector 内部实现

### 数据流与状态机

- 知识库创建：空实现 → 添加文档时自动创建
- 文档检索：`do_search` → Retriever → VectorStore → Document
- 文档删除：元数据匹配 + collection_id 联合删除

### 外部依赖与接口契约

- 依赖 `langchain.vectorstores.pgvector.PGVector`
- 依赖 `sqlalchemy` 进行数据库操作
- 依赖 `chatchat.settings.Settings` 获取配置
- 依赖 `KBService` 基类定义的标准接口



### `PGKBService.do_drop_kb`

该方法用于删除整个知识库（Knowledge Base），它首先从 PostgreSQL 数据库中删除与该知识库相关的所有向量嵌入记录和集合记录，然后使用 `shutil.rmtree` 删除知识库对应的文件系统目录。

参数：

- `self`：`PGKBService` 类的实例，表示当前的知识库服务对象

返回值：`None`，该方法直接操作数据库和文件系统，不返回任何值

#### 流程图

```mermaid
flowchart TD
    A[开始 do_drop_kb] --> B[创建数据库会话]
    B --> C[执行 SQL 删除 langchain_pg_embedding 表中的记录]
    C --> D[执行 SQL 删除 langchain_pg_collection 表中的记录]
    D --> E[提交事务 commit]
    E --> F[使用 shutil.rmtree 删除文件系统目录]
    F --> G[结束]
    
    C -.-> C1[collection_id 匹配 kb_name 的记录]
    D -.-> D1[name 匹配 kb_name 的记录]
```

#### 带注释源码

```python
def do_drop_kb(self):
    """
    删除整个知识库及其相关数据
    
    该方法执行以下操作：
    1. 删除数据库中 langchain_pg_embedding 表的向量嵌入记录
    2. 删除数据库中 langchain_pg_collection 表的集合记录
    3. 删除文件系统中的知识库目录
    """
    # 使用 SQLAlchemy Session 上下文管理器创建数据库会话
    with Session(PGKBService.engine) as session:
        # 执行删除向量嵌入记录的 SQL
        # 删除 langchain_pg_embedding 表中关联到 langchain_pg_collection 表的记录
        session.execute(
            text(
                f"""
                -- 删除 langchain_pg_embedding 表中关联到 langchain_pg_collection 表中 的记录
                DELETE FROM langchain_pg_embedding
                WHERE collection_id IN (
                  SELECT uuid FROM langchain_pg_collection WHERE name = '{self.kb_name}'
                );
                -- 删除 langchain_pg_collection 表中 记录
                DELETE FROM langchain_pg_collection WHERE name = '{self.kb_name}';
        """
            )
        )
        # 提交事务，确保数据库操作生效
        session.commit()
        
        # 使用 shutil.rmtree 递归删除知识库对应的文件系统目录
        # self.kb_path 是知识库文件存储的根目录路径
        # rmtree 会删除目录及其所有子目录和文件
        shutil.rmtree(self.kb_path)
```



### `sqlalchemy.create_engine`

创建 SQLAlchemy 数据库引擎，用于建立与 PostgreSQL 数据库的连接，并通过连接池优化资源管理。

参数：

- `url`：`str`，数据库连接 URI，来源于 `Settings.kb_settings.kbs_config.get("pg").get("connection_uri")`
- `pool_size`：`int`，连接池大小，值为 10

返回值：`Engine`，SQLAlchemy 引擎对象，用于执行数据库操作

#### 流程图

```mermaid
flowchart TD
    A[开始] --> B[获取数据库连接URI]
    B --> C[配置连接池大小为10]
    D[调用create_engine] --> E[创建Engine对象]
    E --> F[返回Engine实例]
    F --> G[用于PGKBService的数据库操作]
    
    C --> D
```

#### 带注释源码

```python
# 导入 sqlalchemy 模块
import sqlalchemy
# 从 sqlalchemy.engine.base 导入 Engine 类型
from sqlalchemy.engine.base import Engine

# 使用 sqlalchemy.create_engine 创建数据库引擎
# 第一个参数：数据库连接 URI，从设置中获取 PostgreSQL 的连接字符串
# 第二个参数 pool_size=10：设置连接池大小为10个连接
engine: Engine = sqlalchemy.create_engine(
    Settings.kb_settings.kbs_config.get("pg").get("connection_uri"),  # PostgreSQL 连接字符串
    pool_size=10  # 连接池大小为10
)

# 解释：
# - create_engine 是 SQLAlchemy 的核心函数，用于创建数据库引擎
# - url 参数指定数据库的连接方式，格式如：postgresql://user:password@host:port/database
# - pool_size 参数控制连接池中保持的最大连接数
# - 返回的 Engine 对象是执行 SQL 的入口点，可用于创建会话和执行原始 SQL
```



### `sqlalchemy.text`

描述：`sqlalchemy.text` 是 SQLAlchemy 库中的核心函数，用于将原始 SQL 语句字符串封装成 SQL 表达式对象，以便在 SQLAlchemy 的 ORM 或核心引擎中安全、标准地执行。该函数支持参数绑定，可有效防止 SQL 注入，并提供数据库无关的 SQL 执行接口。在代码中有多处使用，用于构建查询和删除文档的 SQL 语句。

参数：

-  `text`：`str`，SQL 语句字符串，可以包含命名参数（如 `:ids`、`:name`）用于参数化查询

返回值：`TextClause`，返回 SQLAlchemy 的文本 SQL 表达式对象，可传递给 `session.execute()` 执行

#### 流程图

```mermaid
flowchart TD
    A[开始] --> B[调用 text 函数]
    B --> C[传入 SQL 字符串]
    C --> D{是否包含命名参数?}
    D -->|是| E[创建参数化 TextClause]
    D -->|否| F[创建普通 TextClause]
    E --> G[返回 TextClause 对象]
    F --> G
    G --> H[传递给 Session.execute]
    H --> I[数据库执行 SQL]
    I --> J[返回查询结果]
```

#### 带注释源码

```python
# 在 get_doc_by_ids 方法中使用的 text
stmt = text(
    "SELECT document, cmetadata FROM langchain_pg_embedding WHERE custom_id = ANY(:ids)"
)
# 说明：使用 text 构建查询语句，通过 ANY(:ids) 匹配多个 ID
# 参数 :ids 将由 session.execute 的第二个参数 {"ids": ids} 绑定

# 在 do_drop_kb 方法中使用的 text
session.execute(
    text(
        f"""
        -- 删除 langchain_pg_embedding 表中关联到 langchain_pg_collection 表中 的记录
        DELETE FROM langchain_pg_embedding
        WHERE collection_id IN (
          SELECT uuid FROM langchain_pg_collection WHERE name = '{self.kb_name}'
        );
        -- 删除 langchain_pg_collection 表中 记录
        DELETE FROM langchain_pg_collection WHERE name = '{self.kb_name}';
"""
    )
)
# 说明：使用 text + f-string 构建删除知识库的 SQL 语句
# 注意：此处直接拼接 self.kb_name，存在 SQL 注入风险，应改为参数化查询

# 在 do_delete_doc 方法中使用的两个 text
select_query = text("SELECT uuid FROM langchain_pg_collection WHERE name = :name;")
# 说明：参数化查询，通过 :name 占位符防止 SQL 注入

delete_query = text("""
    DELETE FROM langchain_pg_embedding
    WHERE cmetadata::jsonb @> :cmetadata
    AND collection_id = :collection_id;
""")
# 说明：删除文档的 SQL，使用 cmetadata::jsonb 的 JSON 查询功能
# 参数 :cmetadata 和 :collection_id 由 session.execute 绑定
```

### 具体使用示例

#### 1. `get_doc_by_ids` 方法中的 text 使用

```python
def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
    """
    通过文档 IDs 获取文档内容
    
    参数：
    - ids: 文档 ID 列表
    
    返回：
    - List[Document]: Document 对象列表
    """
    with Session(PGKBService.engine) as session:
        # 使用 text 构建查询语句
        # :ids 是命名参数，通过 session.execute 的第二个参数绑定
        stmt = text(
            "SELECT document, cmetadata FROM langchain_pg_embedding WHERE custom_id = ANY(:ids)"
        )
        # 执行查询，传入参数 {"ids": ids}
        results = [
            Document(page_content=row[0], metadata=row[1])
            for row in session.execute(stmt, {"ids": ids}).fetchall()
        ]
        return results
```

#### 2. `do_delete_doc` 方法中的 text 使用

```python
def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
    """
    删除知识库中的文档
    
    参数：
    - kb_file: KnowledgeFile 对象，包含知识库名称和文件路径
    """
    # 第一个 text：查询 collection 的 UUID
    select_query = text("SELECT uuid FROM langchain_pg_collection WHERE name = :name;")
    
    # 第二个 text：删除文档记录
    delete_query = text("""
        DELETE FROM langchain_pg_embedding
        WHERE cmetadata::jsonb @> :cmetadata
        AND collection_id = :collection_id;
    """)
    
    with Session(PGKBService.engine) as session:
        # 执行查询获取 collection_id
        collection_id = session.execute(select_query, {"name": kb_file.kb_name}).fetchone()[0]
        
        # 执行删除操作，绑定参数
        session.execute(
            delete_query, 
            {
                "cmetadata": '{"source": "%s"}' % self.get_relative_source_path(kb_file.filepath),
                "collection_id": collection_id
            }
        )
        session.commit()
```



### `PGKBService.get_doc_by_ids`

该方法通过 SQLAlchemy ORM 的 `Session` 上下文管理器建立数据库连接，根据提供的文档 ID 列表从 `langchain_pg_embedding` 表中查询对应的文档内容和元数据，并封装为 LangChain 的 `Document` 对象列表返回。

参数：

- `ids`：`List[str]`，需要查询的文档 ID 列表

返回值：`List[Document]`，包含文档内容和元数据的 LangChain Document 对象列表

#### 流程图

```mermaid
flowchart TD
    A[开始 get_doc_by_ids] --> B[创建 Session 上下文]
    B --> C[构建 SQL 查询语句]
    C --> D[执行查询并传递 ids 参数]
    D --> E[fetchall 获取所有结果]
    E --> F[遍历结果行]
    F --> G[为每行创建 Document 对象]
    G --> H{是否还有更多行?}
    H -->|是| F
    H -->|否| I[返回 Document 列表]
    I --> J[Session 上下文自动关闭]
```

#### 带注释源码

```python
def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
    """
    根据文档ID列表从PGVector存储中获取文档
    
    参数:
        ids: 文档唯一标识符列表
    
    返回:
        Document对象列表，每个包含page_content和metadata
    """
    # 使用 Session 上下文管理器，确保数据库连接自动管理
    # PGKBService.engine 是类级别的 SQLAlchemy Engine 实例
    with Session(PGKBService.engine) as session:
        # 构建原生 SQL 查询，从 langchain_pg_embedding 表获取 document 和 cmetadata
        # custom_id = ANY(:ids) 利用 PostgreSQL 的 ANY 函数进行数组匹配
        stmt = text(
            "SELECT document, cmetadata FROM langchain_pg_embedding WHERE custom_id = ANY(:ids)"
        )
        # 执行查询并获取所有匹配结果
        results = [
            Document(page_content=row[0], metadata=row[1])
            for row in session.execute(stmt, {"ids": ids}).fetchall()
        ]
        return results
```

---

### `PGKBService.do_drop_kb`

该方法通过 `Session` 上下文管理器执行数据库事务操作，先删除 `langchain_pg_embedding` 表中与指定知识库关联的向量数据，再删除 `langchain_pg_collection` 表中的集合记录，最后清理本地的知识库文件目录。

参数：无显式参数（继承自父类）

返回值：`None`（无返回值）

#### 流程图

```mermaid
flowchart TD
    A[开始 do_drop_kb] --> B[创建 Session 上下文]
    B --> C[构建删除 langchain_pg_embedding 的 SQL]
    C --> D[构建删除 langchain_pg_collection 的 SQL]
    D --> E[执行删除向量记录]
    E --> F[执行删除集合记录]
    F --> G[session.commit 提交事务]
    G --> H[shutil.rmtree 删除本地文件]
    H --> I[Session 上下文自动关闭]
```

#### 带注释源码

```python
def do_drop_kb(self):
    """
    删除整个知识库，包括数据库中的向量数据和本地文件
    """
    # 使用 Session 管理数据库事务，确保操作的原子性
    with Session(PGKBService.engine) as session:
        # 执行删除操作，包含两条 DELETE 语句
        # 1. 先删除 langchain_pg_embedding 表中关联到该知识库的向量记录
        # 2. 再删除 langchain_pg_collection 表中的集合记录
        session.execute(
            text(
                f"""
                -- 删除 langchain_pg_embedding 表中关联到 langchain_pg_collection 表中 的记录
                DELETE FROM langchain_pg_embedding
                WHERE collection_id IN (
                  SELECT uuid FROM langchain_pg_collection WHERE name = '{self.kb_name}'
                );
                -- 删除 langchain_pg_collection 表中 记录
                DELETE FROM langchain_pg_collection WHERE name = '{self.kb_name}';
        """
            )
        )
        # 提交事务，使删除操作生效
        session.commit()
        # 删除本地的知识库文件目录
        shutil.rmtree(self.kb_path)
```

---

### `PGKBService.do_delete_doc`

该方法通过 `Session` 上下文管理器执行数据库事务，首先查询目标知识库在 `langchain_pg_collection` 表中的 UUID，然后根据文件相对路径构建元数据条件，从 `langchain_pg_embedding` 表中删除匹配的向量记录。

参数：

- `kb_file`：`KnowledgeFile`，包含知识库名称和文件路径的知识文件对象

返回值：`None`（无返回值）

#### 流程图

```mermaid
flowchart TD
    A[开始 do_delete_doc] --> B[创建 Session 上下文]
    B --> C[构建查询 collection_uuid 的 SQL]
    C --> D[执行查询获取 collection_id]
    D --> E[构建删除向量记录的 SQL]
    E --> F[格式化 cmetadata JSON]
    F --> G[执行删除操作并传递参数]
    G --> H[session.commit 提交事务]
    H --> I[Session 上下文自动关闭]
```

#### 带注释源码

```python
def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
    """
    从知识库中删除指定文件的向量数据
    
    参数:
        kb_file: KnowledgeFile对象，包含kb_name和filepath信息
    """
    # 准备查询语句：获取指定知识库名称的 collection UUID
    select_query = text("SELECT uuid FROM langchain_pg_collection WHERE name = :name;")
    # 准备删除语句：使用 PostgreSQL 的 JSONB 操作符 @> 进行元数据匹配
    delete_query = text("""
        DELETE FROM langchain_pg_embedding
        WHERE cmetadata::jsonb @> :cmetadata
        AND collection_id = :collection_id;
    """)
    # 使用 Session 管理数据库事务
    with Session(PGKBService.engine) as session:
        # 查询获取该知识库的 collection_id (UUID)
        collection_id = session.execute(select_query, {"name": kb_file.kb_name}).fetchone()[0]
        # 执行删除操作，传入两个关键参数：
        # 1. cmetadata: JSON格式的source字段，用于匹配具体文件
        # 2. collection_id: 确保只删除该知识库下的记录
        session.execute(
            delete_query, 
            {
                "cmetadata": '{"source": "%s"}' % self.get_relative_source_path(kb_file.filepath),
                "collection_id": collection_id
            }
        )
        # 提交事务，使删除操作生效
        session.commit()
```



### PGVector.__init__ - 向量存储初始化

该函数用于初始化PGVector向量存储库，通过配置嵌入函数、集合名称、距离策略、数据库连接等参数，建立与PostgreSQL数据库的向量存储连接。

参数：

- `embedding_function`: `Callable`, 用于生成文本嵌入的函数，通过`get_Embeddings(self.embed_model)`获取
- `collection_name`: `str`, 向量集合的名称，对应`self.kb_name`（知识库名称）
- `distance_strategy`: `DistanceStrategy`, 距离计算策略，代码中使用`DistanceStrategy.EUCLIDEAN`（欧氏距离）
- `connection`: `Engine`, SQLAlchemy数据库引擎，使用`PGKBService.engine`
- `connection_string`: `str`, 数据库连接字符串，从配置中获取`Settings.kb_settings.kbs_config.get("pg").get("connection_uri")`

返回值：`PGVector`，返回初始化后的向量存储实例

#### 流程图

```mermaid
flowchart TD
    A[开始初始化] --> B[获取嵌入函数]
    B --> C[获取知识库名称作为集合名]
    C --> D[设置欧氏距离策略]
    D --> E[获取数据库引擎]
    E --> F[获取数据库连接字符串]
    F --> G[创建PGVector实例]
    G --> H[返回pg_vector对象]
```

#### 带注释源码

```python
def _load_pg_vector(self):
    """
    加载并初始化PGVector向量存储实例
    
    该方法完成以下工作：
    1. 获取嵌入函数用于将文本转为向量
    2. 设置集合名称（知识库名称）
    3. 配置距离策略为欧氏距离
    4. 使用已有的数据库引擎和连接字符串创建PGVector实例
    """
    self.pg_vector = PGVector(
        embedding_function=get_Embeddings(self.embed_model),  # 获取嵌入模型函数
        collection_name=self.kb_name,  # 使用知识库名称作为向量集合名
        distance_strategy=DistanceStrategy.EUCLIDEAN,  # 使用欧氏距离计算相似度
        connection=PGKBService.engine,  # 传入已有的数据库引擎
        connection_string=Settings.kb_settings.kbs_config.get("pg").get("connection_uri"),  # 数据库连接字符串
    )
```

#### 调用上下文

该方法在`do_init`方法中被调用：

```python
def do_init(self):
    """初始化知识库服务时调用"""
    self._load_pg_vector()
```

#### 依赖配置

- **数据库配置**：需要通过`Settings.kb_settings.kbs_config.get("pg").get("connection_uri")`获取有效的PostgreSQL连接URI
- **嵌入模型**：需要通过`get_Embeddings(self.embed_model)`获取有效的嵌入函数
- **langchain_pg_collection表**：PGVector会在首次添加文档时自动创建集合对应的数据库表结构



### PGKBService.do_add_doc

该方法是 PGKBService 类中用于将文档添加到向量存储的核心方法，内部调用 langchain 的 PGVector.add_documents 将 Document 对象列表持久化到 PostgreSQL 向量数据库中，并返回包含文档 ID 和元数据的字典列表。

参数：

- `self`：`PGKBService`，类的实例本身，包含知识库名称、嵌入模型等上下文信息
- `docs`：`List[Document]`（来自 langchain.schema），待添加的 Document 对象列表，每个 Document 包含 page_content（文本内容）和 metadata（元数据）
- `**kwargs`：可选关键字参数，用于传递额外的配置信息（如批次大小等），当前实现未使用

返回值：`List[Dict]`，返回文档信息列表，每个字典包含 `{ "id": 文档向量ID, "metadata": 原始Document的metadata }`

#### 流程图

```mermaid
flowchart TD
    A[开始 do_add_doc] --> B[调用 pg_vector.add_documents 添加文档到向量库]
    B --> C[获取返回的 ids 列表]
    D[遍历 ids 和 docs 的配对] --> E[构建 doc_info 字典]
    E --> F[将 doc_info 添加到 doc_infos 列表]
    F --> G[返回 doc_infos 列表]
    
    subgraph "底层 PGVector.add_documents"
        B1[接收 Document 列表] --> B2[使用 embedding_function 对文档内容进行向量化]
        B2 --> B3[将向量和元数据插入 langchain_pg_embedding 表]
        B3 --> B4[返回生成的文档 ID 列表]
    end
    
    B -.-> B1
    B2 --> B4
```

#### 带注释源码

```python
def do_add_doc(self, docs: List[Document], **kwargs) -> List[Dict]:
    """
    将文档列表添加到向量存储中
    
    参数:
        docs: Document对象列表，每个Document包含page_content和metadata
        **kwargs: 额外的可选参数（当前未使用）
    
    返回:
        包含文档ID和元数据的信息字典列表
    """
    # 调用 langchain 的 PGVector.add_documents 方法将文档添加到向量库
    # PGVector 内部会：
    # 1. 使用 embedding_function 对每个 Document 的 page_content 进行向量化
    # 2. 将向量、page_content、metadata 存储到 langchain_pg_embedding 表
    # 3. 返回数据库生成的自定义 ID 列表
    ids = self.pg_vector.add_documents(docs)
    
    # 将返回的 ids 与原始 docs 进行配对，构造返回结果
    # 每个元素格式为: {"id": "向量库中的文档ID", "metadata": "原始Document的metadata"}
    doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
    
    # 返回文档信息列表，供上层调用者记录或后续使用
    return doc_infos
```

#### 相关底层方法 PGVector.add_documents（langchain 库）

```python
# 以下为 langchain.vectorstores.pgvector.PGVector.add_documents 的典型实现逻辑
def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
    """
    添加文档到向量存储
    
    参数:
        documents: Document 对象列表
    
    返回:
        添加文档的自定义 ID 列表
    """
    # 1. 对文档内容进行嵌入向量化
    embeddings = [self.embedding_function(doc.page_content) for doc in documents]
    
    # 2. 批量插入到数据库
    with Session(self._conn) as session:
        for doc, embedding in zip(documents, embeddings):
            # 生成自定义 ID (UUID格式)
            custom_id = str(uuid.uuid4())
            
            # 插入 langchain_pg_embedding 表
            stmt = text("""
                INSERT INTO langchain_pg_embedding (custom_id, document, cmetadata, embedding)
                VALUES (:custom_id, :document, :cmetadata, :embedding)
                RETURNING custom_id
            """)
            session.execute(stmt, {
                "custom_id": custom_id,
                "document": doc.page_content,
                "cmetadata": json.dumps(doc.metadata),
                "embedding": embedding
            })
        session.commit()
    
    # 3. 返回生成的 ID 列表
    return [doc.metadata.get("id") if "id" in doc.metadata else custom_id for doc in documents]
```



### PGVector.delete_collection

该方法用于删除 PGVector 向量存储中的集合（collection），在 `PGKBService.do_clear_vs` 方法中被调用，用于清空向量存储以便重新创建集合。

#### 参数

此方法为无参数方法。

#### 流程图

```mermaid
sequenceDiagram
    participant Client
    participant PGKBService
    participant PGVector
    participant PostgreSQL
    
    Client->>PGKBService: do_clear_vs()
    PGKBService->>PGVector: delete_collection()
    PGVector->>PostgreSQL: DELETE FROM langchain_pg_collection
    PGVector->>PostgreSQL: DELETE FROM langchain_pg_embedding
    PostgreSQL-->>PGVector: 执行结果
    PGVector-->>PGKBService: 完成
    PGKBService->>PGVector: create_collection()
    PGKBService-->>Client: 完成
```

#### 带注释源码

```python
def do_clear_vs(self):
    """
    清空向量存储并重新创建集合
    流程：
    1. 调用 PGVector 的 delete_collection() 删除现有集合
    2. 调用 PGVector 的 create_collection() 重新创建空集合
    """
    # 调用 PGVector.delete_collection() - 删除向量存储中的集合
    # 该操作会删除 langchain_pg_collection 和 langchain_pg_embedding 表中的相关记录
    self.pg_vector.delete_collection()
    
    # 删除集合后立即重新创建空集合
    self.pg_vector.create_collection()
```



### `PGVector.create_collection`

该方法用于在 PostgreSQL 数据库中创建向量集合（collection），为存储文档嵌入向量做准备。通常在初始化向量存储或清空向量存储后调用，以确保集合存在。

参数：
- （无显式参数，使用默认行为）

返回值：`None`，该方法通常无返回值，直接在数据库中创建集合结构。

#### 流程图

```mermaid
flowchart TD
    A[开始 create_collection] --> B{检查连接是否有效}
    B -->|是| C{检查集合是否已存在}
    B -->|否| D[抛出连接异常]
    C -->|不存在| E[创建 langchain_pg_collection 表记录]
    C -->|已存在| F[跳过创建或抛出警告]
    E --> G[初始化集合元数据]
    G --> H[结束]
```

#### 带注释源码

```python
# 该方法是 langchain.vectorstores.pgvector.PGVector 类的成员方法
# 在当前代码中通过以下方式调用：

def do_clear_vs(self):
    """
    清空向量存储
    流程：删除现有集合 -> 重新创建集合
    """
    # 第一步：删除现有的向量集合
    self.pg_vector.delete_collection()
    
    # 第二步：重新创建空集合，为后续添加文档做准备
    # 此处调用的即为 create_collection 方法
    self.pg_vector.create_collection()


# PGVector.create_collection 方法的典型实现逻辑（基于 langchain 源码推断）：
def create_collection(self):
    """
    创建向量集合的底层实现逻辑：
    
    1. 检查与 PostgreSQL 数据库的连接是否有效
    2. 检查同名的 collection 是否已存在于 langchain_pg_collection 表中
    3. 如果不存在，则执行 CREATE 语句创建集合记录
    4. 初始化集合的元数据（如 collection_name, cmetadata 等）
    5. 返回 None，确认集合创建完成
    """
    # 连接数据库
    # 检查 collection_name 是否冲突
    # 创建表 langchain_pg_collection 的记录（包含 uuid, name, cmetadata 字段）
    # 如有必要，创建 langchain_pg_embedding 表的关联
    pass
```

#### 额外说明

| 项目 | 说明 |
|------|------|
| **调用场景** | 在 `PGKBService.do_clear_vs()` 方法中被调用，用于重置向量存储 |
| **外部依赖** | 依赖 `langchain.vectorstores.pgvector.PGVector` 类 |
| **数据库表** | 操作 `langchain_pg_collection` 和 `langchain_pg_embedding` 表 |
| **潜在优化** | 可考虑添加集合存在性检查的参数，避免重复创建时的异常 |



### `get_Retriever.from_vectorstore`

从给定的向量存储（PGVector）创建一个检索器实例，用于执行基于语义相似度的文档搜索。

参数：

- `vectorstore`：`PGVector`，向量存储实例，包含已嵌入的文档数据
- `top_k`：`int`，返回最相关的文档数量
- `score_threshold`：`float`，用于过滤低相关性文档的分数阈值（值越高相关性要求越严格）

返回值：`Retriever`，返回一个配置好的检索器对象，可通过调用 `get_relevant_documents(query)` 方法获取与查询语义相关的文档列表

#### 流程图

```mermaid
flowchart TD
    A[开始创建检索器] --> B[接收向量存储实例和搜索参数]
    B --> C{参数验证}
    C -->|参数有效| D[根据vectorstore创建Retriever实例]
    C -->|参数无效| E[抛出异常或返回默认值]
    D --> F[配置top_k和score_threshold]
    F --> G[返回配置好的Retriever对象]
    G --> H[调用get_relevant_documents执行搜索]
    H --> I[返回相关文档列表]
    
    style A fill:#f9f,color:#333
    style G fill:#9f9,color:#333
    style I fill:#9ff,color:#333
```

#### 带注释源码

```python
def do_search(self, query: str, top_k: int, score_threshold: float):
    """
    执行知识库文档搜索
    
    参数:
        query: str - 用户查询字符串，用于语义匹配
        top_k: int - 返回最相关的前N个文档
        score_threshold: float - 文档相关性分数阈值，低于此值的文档将被过滤
    
    返回:
        List[Document] - 返回与查询最相关的文档列表
    """
    # 调用get_Retriever工厂函数获取向量存储类型的检索器
    # 并通过from_vectorstore方法创建基于self.pg_vector的检索器实例
    retriever = get_Retriever("vectorstore").from_vectorstore(
        self.pg_vector,          # PGVector向量存储实例，包含已嵌入的文档
        top_k=top_k,             # 控制返回文档数量
        score_threshold=score_threshold,  # 控制文档相关性过滤阈值
    )
    
    # 使用创建的检索器获取与query语义相关的文档
    docs = retriever.get_relevant_documents(query)
    
    # 返回检索到的文档列表
    return docs
```



### `PGKBService._load_pg_vector`

该方法负责初始化并加载pgvector向量存储实例，将PGVector客户端绑定到当前知识库服务实例的`pg_vector`属性上，以便后续进行向量检索和文档添加等操作。

参数：

- `self`：实例本身，PGKBService类型，表示当前知识库服务实例

返回值：`None`，该方法无返回值，结果存储在实例属性`self.pg_vector`中

#### 流程图

```mermaid
flowchart TD
    A[开始 _load_pg_vector] --> B[获取嵌入模型配置]
    B --> C[调用 get_Embeddings 获取嵌入函数]
    C --> D[构建 PGVector 连接参数]
    D --> E[创建 PGVector 实例]
    E --> F[设置实例属性 self.pg_vector]
    F --> G[结束]
    
    style A fill:#e1f5fe
    style G fill:#e8f5e8
```

#### 带注释源码

```python
def _load_pg_vector(self):
    """
    加载并初始化 pgvector 向量存储实例
    
    该方法完成以下工作：
    1. 获取当前知识库使用的嵌入模型
    2. 配置向量距离计算策略
    3. 建立与 PostgreSQL 数据库的连接
    4. 创建 PGVector 向量存储客户端实例
    """
    # 使用当前知识库的嵌入模型名称获取嵌入函数
    # embedding_function: 用于将文本转换为向量表示的函数
    self.pg_vector = PGVector(
        embedding_function=get_Embeddings(self.embed_model),  # 获取嵌入模型对应的嵌入函数
        collection_name=self.kb_name,                          # 向量集合名称，默认为知识库名称
        distance_strategy=DistanceStrategy.EUCLIDEAN,          # 使用欧几里得距离计算向量相似度
        connection=PGKBService.engine,                         # SQLAlchemy 数据库引擎连接
        # 从设置中获取 PostgreSQL 连接字符串，用于新建集合时使用
        connection_string=Settings.kb_settings.kbs_config.get("pg").get("connection_uri"),
    )
```



### `PGKBService.get_doc_by_ids`

根据指定的文档ID列表，从PostgreSQL数据库中检索对应的文档内容和元数据，并封装为LangChain的Document对象列表返回。

参数：

- `ids`：`List[str]`，需要查询的文档ID列表

返回值：`List[Document]`，包含文档内容和元数据的Document对象列表

#### 流程图

```mermaid
flowchart TD
    A[开始] --> B[创建数据库会话]
    B --> C[构造SQL查询语句]
    C --> D[使用ANY操作符匹配IDs]
    D --> E[执行查询]
    E --> F{查询是否有结果?}
    F -->|是| G[遍历查询结果]
    F -->|否| H[返回空列表]
    G --> I[提取document和cmetadata字段]
    I --> J[创建Document对象]
    J --> K[添加到结果列表]
    K --> L{还有更多记录?}
    L -->|是| G
    L -->|否| M[返回Document列表]
    M --> N[结束]
```

#### 带注释源码

```python
def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
    """
    根据文档ID列表获取文档对象
    
    参数:
        ids: 文档ID列表
        
    返回:
        Document对象列表
    """
    # 使用上下文管理器创建数据库会话，确保会话自动关闭
    with Session(PGKBService.engine) as session:
        # 构造SQL查询语句，从langchain_pg_embedding表查询文档
        # custom_id字段使用ANY操作符匹配ids数组中的任意一个值
        stmt = text(
            "SELECT document, cmetadata FROM langchain_pg_embedding WHERE custom_id = ANY(:ids)"
        )
        # 执行查询并获取所有匹配的结果
        # 将每行结果转换为Document对象
        # row[0]对应document字段（文档内容）
        # row[1]对应cmetadata字段（文档元数据）
        results = [
            Document(page_content=row[0], metadata=row[1])
            for row in session.execute(stmt, {"ids": ids}).fetchall()
        ]
        return results
```



### `PGKBService.del_doc_by_ids`

该方法是一个委托方法，通过调用父类 `KBService` 的 `del_doc_by_ids` 实现，根据传入的文档ID列表删除知识库中对应的向量存储记录。

参数：

- `ids`：`List[str]`，需要删除的文档ID列表，每个ID唯一标识知识库中的一份文档

返回值：`bool`，返回 `True` 表示删除操作成功，返回 `False` 表示删除失败或未找到对应文档

#### 流程图

```mermaid
flowchart TD
    A[开始 del_doc_by_ids] --> B{检查 ids 是否有效}
    B -->|ids 为空或无效| C[返回 False]
    B -->|ids 有效| D[调用父类 KBService.del_doc_by_ids]
    D --> E[执行数据库删除操作]
    E --> F[返回删除结果 bool 值]
    F --> G[结束]
```

#### 带注释源码

```python
def del_doc_by_ids(self, ids: List[str]) -> bool:
    """
    根据文档ID列表删除知识库中的文档记录
    
    该方法是 PGKBService 类提供的文档删除接口，通过委托调用父类 KBService
    的实现来执行实际的删除逻辑。父类方法会处理向量存储（PGVector）中的
    文档删除操作。
    
    Args:
        ids: 文档ID列表，类型为字符串列表，每个ID对应知识库中的一条文档记录
        
    Returns:
        bool: 删除操作是否成功完成
    """
    # 调用父类 KBService 的 del_doc_by_ids 方法执行实际的删除逻辑
    # 父类方法会遍历 ids 列表，在向量存储中查找并删除对应的文档记录
    return super().del_doc_by_ids(ids)
```



### `PGKBService.do_init`

该方法用于初始化知识库服务，通过调用内部方法 `_load_pg_vector` 来加载并初始化 PGVector 向量存储实例，建立与 PostgreSQL 数据库的向量搜索能力。

参数：
- （无参数）

返回值：`None`，无返回值描述（该方法执行初始化操作，不返回任何值）

#### 流程图

```mermaid
flowchart TD
    A[开始 do_init] --> B{调用 self._load_pg_vector}
    B --> C[创建 PGVector 实例]
    C --> D[设置 embedding_function = get_Embeddings]
    C --> E[设置 collection_name = kb_name]
    C --> F[设置 distance_strategy = EUCLIDEAN]
    C --> G[设置 connection = PGKBService.engine]
    C --> H[设置 connection_string]
    D --> I[结束 do_init]
    E --> I
    F --> I
    G --> I
    H --> I
```

#### 带注释源码

```python
def do_init(self):
    """
    初始化知识库服务
    该方法调用内部方法 _load_pg_vector 来加载 PGVector 向量存储实例，
    完成知识库的初始化工作，使服务具备向量检索能力。
    """
    # 调用私有方法加载 PGVector 实例
    # 内部会创建 PGVector 对象并赋值给 self.pg_vector
    # 包含embedding模型配置、集合名称、距离策略、数据库连接等初始化
    self._load_pg_vector()
```



### `PGKBService.do_create_kb`

创建知识库的实现方法，目前为占位实现（pass），具体创建逻辑尚未完成。该方法用于在 PGVector 向量数据库中初始化一个新的知识库，但由于方法体为空，尚未实现实际的创建逻辑。

参数： 无

返回值：`None`，该方法不返回任何值

#### 流程图

```mermaid
flowchart TD
    A[开始 do_create_kb] --> B{方法调用}
    B --> C[执行 pass 空操作]
    C --> D[结束]
    
    style C fill:#f9f,stroke:#333,stroke-width:2px
```

#### 带注释源码

```python
def do_create_kb(self):
    """
    创建知识库
    该方法为占位实现，目前未完成具体的创建逻辑
    
    参数：
        无
    
    返回值：
        None - 不返回任何值
    """
    pass  # TODO: 实现知识库创建逻辑，需要在 PGVector 中创建 collection 和相关表结构
```

---

### 补充说明

#### 技术债务

1. **未实现的占位方法**：`do_create_kb` 方法当前为空实现（pass），是一个明显的技术债务。在实际使用中，创建一个新的知识库需要在 PGVector 中初始化 collection 和必要的数据库表结构。

2. **潜在问题**：由于该方法未实现，调用 `create_kb()` 时实际上不会创建任何知识库，可能导致后续操作失败或行为异常。

#### 优化建议

1. **实现创建逻辑**：应该在该方法中实现以下功能：
   - 在 `langchain_pg_collection` 表中创建新的 collection 记录
   - 如果需要，创建对应的向量存储表结构
   - 初始化 `self.pg_vector` 实例

2. **一致性考虑**：参考 `do_drop_kb` 方法的实现，`do_create_kb` 应该具有对称的操作逻辑。

#### 设计目标与约束

- **设计目标**：为 PGVector 向量数据库提供知识库管理能力
- **约束**：依赖 `langchain_pg_embedding` 和 `langchain_pg_collection` 表结构
- **当前状态**：该方法为待完成状态，是整体功能中的一个缺口



### `PGKBService.vs_type`

该方法用于返回当前知识库服务所使用的向量存储类型，固定返回 PG（PostgreSQL）类型，表示该知识库基于 PostgreSQL 的 pgvector 扩展进行向量存储。

参数：无需参数（`self` 为实例本身）

返回值：`str`，返回 `SupportedVSType.PG`，标识该服务使用 PostgreSQL 向量存储

#### 流程图

```mermaid
flowchart TD
    A[开始 vs_type] --> B{执行方法}
    B --> C[返回 SupportedVSType.PG]
    C --> D[结束]
```

#### 带注释源码

```python
def vs_type(self) -> str:
    """
    返回向量存储类型
    
    Returns:
        str: 向量存储类型，固定返回 SupportedVSType.PG
              表示该知识库服务使用 PostgreSQL + pgvector 进行向量存储
    """
    return SupportedVSType.PG
```



### PGKBService.do_drop_kb

删除知识库，清理数据库中相关的向量数据记录以及本地文件系统中的知识库目录。

参数：该方法无参数。

返回值：`None`，无返回值。

#### 流程图

```mermaid
flowchart TD
    A[开始执行 do_drop_kb] --> B[建立数据库会话 Session]
    B --> C[执行SQL: 删除langchain_pg_embedding表关联记录]
    C --> D[执行SQL: 删除langchain_pg_collection表记录]
    D --> E[提交事务 session.commit]
    E --> F[使用shutil.rmtree删除本地知识库目录]
    F --> G[结束]
```

#### 带注释源码

```python
def do_drop_kb(self):
    """
    删除知识库，包括数据库中的向量记录和本地文件系统中的知识库目录
    """
    # 使用SQLAlchemy Session管理数据库连接会话
    with Session(PGKBService.engine) as session:
        # 执行SQL语句：先删除langchain_pg_embedding表中关联到langchain_pg_collection的记录
        # 再删除langchain_pg_collection表中该知识库对应的记录
        session.execute(
            text(
                f"""
                -- 删除 langchain_pg_embedding 表中关联到 langchain_pg_collection 表中 的记录
                DELETE FROM langchain_pg_embedding
                WHERE collection_id IN (
                  SELECT uuid FROM langchain_pg_collection WHERE name = '{self.kb_name}'
                );
                -- 删除 langchain_pg_collection 表中 记录
                DELETE FROM langchain_pg_collection WHERE name = '{self.kb_name}';
        """
            )
        )
        # 提交事务，确保删除操作生效
        session.commit()
        # 使用shutil.rmtree递归删除本地知识库目录（包括所有文件）
        shutil.rmtree(self.kb_path)
```



### `PGKBService.do_search`

搜索相似文档，根据传入的查询字符串、top_k 数量和相似度阈值，从 PGVector 向量存储中检索最相似的文档。

参数：

- `query`：`str`，查询字符串，用于在知识库中搜索相似的文档
- `top_k`：`int`，返回最相似的文档数量
- `score_threshold`：`float`，相似度阈值，只有分数高于该阈值的文档才会被返回

返回值：`List[Document]`，返回与查询相似的文档列表，每个 Document 包含页面内容和元数据

#### 流程图

```mermaid
flowchart TD
    A[开始 do_search] --> B[调用 get_Retriever 创建检索器]
    B --> C[配置向量存储为 pg_vector]
    C --> D[设置 top_k 和 score_threshold 参数]
    D --> E[调用 retriever.get_relevant_documents 获取相似文档]
    E --> F[返回文档列表]
    
    subgraph 内部操作
    B -.-> B1[从 self.pg_vector 创建 Retriever]
    end
```

#### 带注释源码

```python
def do_search(self, query: str, top_k: int, score_threshold: float):
    """
    搜索相似文档
    
    参数:
        query: str - 查询字符串
        top_k: int - 返回前 k 个最相似的文档
        score_threshold: float - 相似度阈值
    
    返回:
        List[Document] - 相似文档列表
    """
    # 使用工厂方法从向量存储创建 Retriever（检索器）
    # get_Retriever("vectorstore") 获取向量存储类型的检索器
    # from_vectorstore 方法将 pg_vector 向量存储转换为可查询的检索器
    retriever = get_Retriever("vectorstore").from_vectorstore(
        self.pg_vector,          # PostgreSQL 向量存储实例
        top_k=top_k,             # 返回最相似的 top_k 个文档
        score_threshold=score_threshold,  # 相似度过滤阈值
    )
    
    # 调用检索器的 get_relevant_documents 方法执行相似度搜索
    # 该方法会:
    # 1. 将 query 转换为向量嵌入
    # 2. 在 PGVector 中执行向量相似度搜索
    # 3. 根据 score_threshold 过滤低相似度结果
    # 4. 返回包含 page_content 和 metadata 的 Document 对象列表
    docs = retriever.get_relevant_documents(query)
    
    # 返回搜索结果文档列表
    return docs
```



### `PGKBService.do_add_doc`

该方法用于将文档添加到PostgreSQL向量库中，通过pgvector扩展实现向量存储，并返回添加的文档ID和元数据信息。

参数：

- `docs`：`List[Document]` - 要添加的LangChain Document对象列表，每个Document包含页面内容和元数据
- `**kwargs`：可变关键字参数，用于接收其他可选参数（如元数据选项等）

返回值：`List[Dict]` - 返回包含文档ID和元数据信息的字典列表，每个字典包含"id"和"metadata"两个键

#### 流程图

```mermaid
flowchart TD
    A[开始 do_add_doc] --> B[调用 pg_vector.add_documents 添加文档到向量库]
    B --> C[获取返回的文档ID列表 ids]
    D[遍历 ids 和 docs] --> E[构建 doc_infos 列表]
    E --> F[将每个文档的id和metadata组合成字典]
    F --> G[返回 doc_infos 列表]
    
    B -.->|使用embedding_function| H[对文档内容进行向量化]
    H --> I[将向量存储到PGVector中]
    I --> C
```

#### 带注释源码

```python
def do_add_doc(self, docs: List[Document], **kwargs) -> List[Dict]:
    """
    将文档添加到PostgreSQL向量库中
    
    参数:
        docs: LangChain Document对象列表，每个Document包含page_content和metadata
        **kwargs: 可选参数，用于扩展功能
    
    返回:
        包含文档ID和元数据的字典列表
    """
    # 调用PGVector的add_documents方法将文档添加到向量库
    # 该方法会:
    # 1. 使用embedding_function对文档内容进行向量化
    # 2. 将向量和元数据存储到langchain_pg_embedding表中
    # 3. 返回自动生成的文档ID列表
    ids = self.pg_vector.add_documents(docs)
    
    # 使用列表推导式构建返回的文档信息列表
    # 遍历返回的ids和原始docs，通过zip配对
    # 每个元素包含该文档的唯一ID和对应的metadata
    doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
    
    # 返回文档信息列表，供调用方记录或后续使用
    return doc_infos
```



### `PGKBService.do_delete_doc`

删除指定文档，该方法通过知识库名称查询对应的collection_id，然后使用JSONB路径匹配删除langchain_pg_embedding表中的相关文档记录。

参数：

- `kb_file`：`KnowledgeFile`，要删除的知识文件对象，包含知识库名称和文件路径信息
- `**kwargs`：`dict`，其他可选参数，用于扩展方法功能

返回值：`None`，该方法执行删除操作后直接提交事务，无返回值

#### 流程图

```mermaid
flowchart TD
    A[开始 do_delete_doc] --> B[构建查询SQL]
    B --> C[创建数据库会话]
    C --> D[执行select_query查询collection_id]
    D --> E[从查询结果获取uuid作为collection_id]
    E --> F[构建删除参数]
    F --> G[执行delete_query删除文档]
    G --> H[提交事务 session.commit]
    H --> I[结束]
    
    style A fill:#f9f,stroke:#333
    style I fill:#9f9,stroke:#333
```

#### 带注释源码

```python
def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
    """
    删除指定文档
    
    参数:
        kb_file: KnowledgeFile对象,包含要删除文件的知识库名称和文件路径
        **kwargs: 额外参数,用于扩展
    """
    # 定义查询语句:根据知识库名称查询对应的collection UUID
    select_query = text("SELECT uuid FROM langchain_pg_collection WHERE name = :name;")
    
    # 定义删除语句:使用JSONB路径匹配删除embeddings记录
    # cmetadata::jsonb @> :cmetadata 表示JSONB字段包含指定对象
    delete_query = text("""
        DELETE FROM langchain_pg_embedding
        WHERE cmetadata::jsonb @> :cmetadata
        AND collection_id = :collection_id;
    """)
    
    # 使用上下文管理器创建数据库会话,自动管理连接
    with Session(PGKBService.engine) as session:
        # 执行查询获取collection_id
        collection_id = session.execute(select_query, {"name": kb_file.kb_name}).fetchone()[0]
        
        # 构建删除参数:通过相对路径匹配source字段
        # 注意:此处使用字符串格式化存在SQL注入风险
        session.execute(
            delete_query, 
            {
                "cmetadata": '{"source": "%s"}' % self.get_relative_source_path(kb_file.filepath),
                "collection_id": collection_id
            }
        )
        
        # 提交事务,使删除操作生效
        session.commit()
```



### `PGKBService.do_clear_vs`

该方法用于清空向量存储（Vector Store），通过先删除已有的向量集合，再重新创建一个新的空集合来实现数据的完全清除。

参数：
- （无额外参数，仅包含隐式 `self` 参数）

返回值：`None`，无返回值描述

#### 流程图

```mermaid
flowchart TD
    A[开始 do_clear_vs] --> B[调用 pg_vector.delete_collection]
    B --> C[调用 pg_vector.create_collection]
    C --> D[结束]
```

#### 带注释源码

```python
def do_clear_vs(self):
    """
    清空向量存储（Vector Store）
    
    该方法通过以下两步实现向量存储的清空：
    1. 删除当前已有的向量集合（delete_collection）
    2. 重新创建一个同名的新空集合（create_collection）
    
    注意：此操作会清除集合中的所有向量数据，但保留集合名称。
    """
    # 第一步：删除已有的向量集合
    # 调用 PGVector 的 delete_collection 方法移除整个集合及其所有向量数据
    self.pg_vector.delete_collection()
    
    # 第二步：创建新的空向量集合
    # 调用 PGVector 的 create_collection 方法重新创建同名集合
    # 该集合初始状态为空，可用于后续的文档添加操作
    self.pg_vector.create_collection()
```

## 关键组件





### PGVector 惰性加载

通过 `_load_pg_vector` 方法实现，仅在首次调用时初始化 `PGVector` 实例，避免启动时的资源消耗。初始化时设置欧几里得距离策略，并复用全局引擎连接。

### PostgreSQL 连接池管理

类级别属性 `engine: Engine` 使用 `sqlalchemy.create_engine` 创建，配置 `pool_size=10` 的连接池，实现连接复用和资源管理。

### 向量检索与阈值过滤

`do_search` 方法通过 `get_Retriever` 创建检索器，支持 `top_k` 和 `score_threshold` 参数，实现带分数阈值的向量相似度检索。

### 文档批量添加

`do_add_doc` 方法调用 `pg_vector.add_documents` 批量添加文档，返回包含 id 和 metadata 的文档信息列表。

### 文档ID查询

`get_doc_by_ids` 方法使用原生 SQL 查询 `langchain_pg_embedding` 表，通过 `ANY` 操作符批量获取指定 ID 的文档内容与元数据。

### 知识库删除

`do_drop_kb` 方法执行级联删除操作，先删除 `langchain_pg_embedding` 中的关联记录，再删除 `langchain_pg_collection` 记录，最后清理本地文件目录。

### 文档删除（按文件）

`do_delete_doc` 方法通过 JSONB 查询条件 `cmetadata::jsonb @>` 匹配源文件路径，精确删除指定知识文件的向量记录。

### 向量库重置

`do_clear_vs` 方法先删除现有集合再重建，实现向量库的清空与初始化。



## 问题及建议



### 已知问题

-   **SQL注入风险**：`do_drop_kb` 方法中直接使用 f-string 拼接 SQL 语句 `f"WHERE name = '{self.kb_name}'"`，存在严重的 SQL 注入漏洞
-   **字符串格式化导致JSON注入**：`do_delete_doc` 方法中使用 `%` 格式化字符串构建 JSON 查询条件，可能导致查询失败或注入问题
-   **异常处理缺失**：数据库操作（Session 上下文）未进行异常捕获和处理，数据库连接失败或查询异常会导致程序崩溃
-   **硬编码配置**：`pool_size=10` 被硬编码在类属性中，无法通过配置文件灵活调整
-   **资源管理不当**：`engine` 作为类级别共享资源，多个 `PGKBService` 实例共享同一个连接池，缺乏实例级别的隔离和生命周期管理
-   **路径遍历风险**：`shutil.rmtree(self.kb_path)` 直接删除目录，未验证路径是否在预期范围内，可能存在目录遍历攻击风险
-   **代码重复**：`Settings.kb_settings.kbs_config.get("pg").get("connection_uri")` 在多处重复调用，应提取为类或实例方法
-   **测试代码混入生产代码**：`if __name__ == "__main__"` 块包含测试代码，不应在生产部署的源码中出现
-   **类型注解不完整**：部分方法缺少返回类型注解，如 `del_doc_by_ids` 返回类型标注为 `bool`，但实际可能返回其他值

### 优化建议

-   使用参数化查询替代 f-string 拼接 SQL，例如使用 `text("DELETE ... WHERE name = :kb_name")` 并传递参数
-   使用 `json.dumps()` 或 `jsonb_build_object()` 构建 JSON 查询条件，避免字符串格式化
-   为所有数据库操作添加 try-except 异常处理，捕获 `SQLAlchemyError` 及其子类
-   将 `pool_size` 等配置项提取到 `Settings` 配置类中，支持运行时配置
-   考虑使用依赖注入或上下文管理器管理数据库连接，而非类级别共享 engine
-   在执行 `shutil.rmtree` 前验证 `self.kb_path` 是否在允许的知识库目录范围内，可使用 `os.path.realpath` 和 `startswith` 进行路径安全检查
-   提取 `connection_uri` 为类方法或实例属性，避免重复调用
-   删除 `if __name__ == "__main__"` 块，或将其移至独立的测试文件
-   完善方法返回类型注解，确保类型声明与实际返回值一致
-   考虑添加连接池监控和重试机制，提高系统鲁棒性

## 其它




### 设计目标与约束

本模块旨在提供基于PostgreSQL向量数据库(PGVector)的知识库服务实现，支持文档的向量存储、检索和删除等核心功能。设计约束包括：1) 必须继承KBService基类以保持知识库服务接口一致性；2) 使用langchain的PGVector作为向量存储引擎；3) 仅支持EUCLIDEAN距离策略；4) 依赖Settings配置中的connection_uri进行数据库连接。

### 错误处理与异常设计

1) 数据库连接错误：通过SQLAlchemy的连接池管理，pool_size=10，当连接池耗尽时将抛出TimeoutError；2) SQL执行错误：session.execute可能抛出SQLAlchemy异常，应捕获并返回False或空列表；3) 文件操作错误：shutil.rmtree在do_drop_kb中可能抛出FileNotFoundError；4) 文档ID不存在：get_doc_by_ids查询空结果时返回空列表而非异常；5) 建议添加更详细的异常日志记录和重试机制。

### 数据流与状态机

数据流：1) 添加文档流程：do_add_doc接收Document列表 -> 调用pg_vector.add_documents -> 返回ids和metadata组成的doc_infos；2) 删除文档流程：do_delete_doc先查询collection_id -> 执行DELETE语句 -> commit事务；3) 搜索流程：do_search创建Retriever -> 调用get_relevant_documents返回Document列表；4) 知识库生命周期：init -> create_kb(空实现) -> add_doc/delete_doc/search -> drop_kb(物理删除)。

### 外部依赖与接口契约

外部依赖：1) langchain.vectorstores.pgvector.PGVector - 向量存储核心类；2) sqlalchemy - 数据库ORM和连接管理；3) chatchat.settings.Settings - 配置管理；4) chatchat.server.knowledge_base.kb_service.base.KBService - 基类接口；5) chatchat.server.knowledge_base.utils.KnowledgeFile - 知识文件实体。接口契约：所有public方法需与KBService基类签名一致，返回类型需匹配基类定义（do_add_doc返回List[Dict]，do_search返回List[Document]）。

### 性能考虑与优化空间

1) 连接池：当前固定pool_size=10，建议根据实际负载动态调整；2) 批量操作：do_add_doc支持批量添加但未实现批量预分配ID；3) 索引优化：建议在collection_name和metadata字段上建立索引；4) 缓存机制：可考虑缓存embedding结果避免重复计算；5) 异步处理：当前为同步执行，高并发场景下建议使用async/await；6) 分页支持：do_search未实现分页，大数据集可能性能下降。

### 安全性考虑

1) SQL注入风险：do_drop_kb中直接使用f-string拼接kb_name，应使用参数化查询；2) 敏感信息：数据库连接字符串存储在配置中，需确保配置安全存储；3) 权限控制：数据库用户应遵循最小权限原则，仅授予必要表的操作权限；4) 文件路径：do_delete_doc中处理filepath需校验路径遍历攻击。

### 事务处理与一致性

1) 删除操作：do_delete_doc和do_drop_kb使用session.commit()确保事务一致性；2) 批量操作：do_add_doc的add_documents调用链中已包含事务管理；3) 建议改进：添加事务回滚机制，当部分操作失败时自动回滚；4) 建议改进：实现分布式事务支持以应对跨服务场景。

### 配置管理

1) 数据库连接：connection_uri从Settings.kb_settings.kbs_config.get("pg")获取；2) 连接池配置：当前硬编码pool_size=10，建议外置配置；3) collection_name：动态使用self.kb_name；4) 距离策略：固定为EUCLIDEAN，建议支持配置化。

### 并发控制

1) SQLAlchemy Session：每次操作创建新Session，线程安全；2) 连接池：pool_size=10限制最大并发连接数；3) 建议改进：添加读写分离支持；4) 建议改进：实现乐观锁或悲观锁防止并发更新冲突。

### 测试策略

1) 单元测试：针对每个方法(mock PGVector和Session)；2) 集成测试：测试真实数据库连接和操作；3) 边界测试：空列表输入、超大top_k、非法ID格式；4) 性能测试：大规模文档添加和检索的响应时间。

### 监控与日志

1) 建议添加：方法执行耗时日志；2) 建议添加：数据库查询慢查询日志；3) 建议添加：连接池使用率监控指标；4) 建议添加：文档操作成功/失败计数。

    
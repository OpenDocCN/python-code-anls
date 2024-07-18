# `.\graphrag\graphrag\index\verbs\entities\extraction\strategies\nltk.py`

```py
# 导入必要的库和模块
import networkx as nx  # 导入网络图库，用于构建和操作图结构
import nltk  # 导入自然语言工具包，用于文本处理和分析
from datashaper import VerbCallbacks  # 导入数据形状模块中的动词回调函数
from nltk.corpus import words  # 导入 NLTK 中的单词语料库

from graphrag.index.cache import PipelineCache  # 从 graphrag 模块中导入 PipelineCache 类
from .typing import Document, EntityExtractionResult, EntityTypes, StrategyConfig  # 从当前目录下的 typing 模块导入特定类型

# 确保单词语料库已加载，以避免在多线程环境下可能出现的问题
words.ensure_loaded()


async def run(  # 定义异步函数 run，用于执行实体提取任务
    docs: list[Document],  # 参数 docs：包含 Document 类型的列表，表示要处理的文档
    entity_types: EntityTypes,  # 参数 entity_types：表示待识别的实体类型集合
    reporter: VerbCallbacks,  # 参数 reporter：动词回调对象，用于报告处理进度
    pipeline_cache: PipelineCache,  # 参数 pipeline_cache：流水线缓存对象，用于数据缓存
    args: StrategyConfig,  # 参数 args：策略配置对象，用于控制算法行为
) -> EntityExtractionResult:
    """运行方法定义."""
    entity_map = {}  # 初始化空字典，用于存储实体名称和对应类型的映射关系
    graph = nx.Graph()  # 创建空的无向图对象，用于存储实体间的关系

    # 遍历每个文档对象
    for doc in docs:
        connected_entities = []  # 初始化空列表，用于存储文档中相连的实体名称

        # 使用 NLTK 进行命名实体识别和词性标注，并遍历识别出的实体块
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(doc.text))):
            if hasattr(chunk, "label"):  # 检查实体块是否有标签属性
                entity_type = chunk.label().lower()  # 获取实体块的标签，并转换为小写
                if entity_type in entity_types:  # 如果实体类型在待识别的实体类型集合中
                    name = (" ".join(c[0] for c in chunk)).upper()  # 将实体块中的词组合成大写形式的名称
                    connected_entities.append(name)  # 将实体名称添加到相连实体列表中

                    if name not in entity_map:  # 如果实体名称不在映射字典中
                        entity_map[name] = entity_type  # 将实体名称和类型映射关系添加到字典中
                        graph.add_node(  # 在图中添加节点
                            name, type=entity_type, description=name, source_id=doc.id
                        )

        # 将同一文档中的实体连接起来
        if len(connected_entities) > 1:  # 如果相连实体列表长度大于1
            for i in range(len(connected_entities)):
                for j in range(i + 1, len(connected_entities)):
                    description = f"{connected_entities[i]} -> {connected_entities[j]}"  # 构建连接描述
                    graph.add_edge(  # 在图中添加边，连接两个实体
                        connected_entities[i],
                        connected_entities[j],
                        description=description,
                        source_id=doc.id,
                    )

    # 构建实体提取结果对象，包括实体列表和图的 GraphML 表示
    return EntityExtractionResult(
        entities=[
            {"type": entity_type, "name": name}
            for name, entity_type in entity_map.items()
        ],
        graphml_graph="".join(nx.generate_graphml(graph)),  # 将图对象转换为 GraphML 格式字符串
    )
```
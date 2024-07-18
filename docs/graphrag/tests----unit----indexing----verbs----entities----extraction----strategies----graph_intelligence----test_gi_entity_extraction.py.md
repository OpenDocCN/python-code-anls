# `.\graphrag\tests\unit\indexing\verbs\entities\extraction\strategies\graph_intelligence\test_gi_entity_extraction.py`

```py
# 引入unittest模块，用于编写和运行测试代码
import unittest

# 引入networkx库，用于图论和网络分析
import networkx as nx

# 从指定路径导入相关模块和函数
from graphrag.index.verbs.entities.extraction.strategies.graph_intelligence.run_graph_intelligence import (
    Document,
    run_extract_entities,
)
# 从测试文件中导入mock_llm函数
from tests.unit.indexing.verbs.helpers.mock_llm import create_mock_llm

# 定义测试用例的类
class TestRunChain(unittest.IsolatedAsyncioTestCase):
    
    # 定义异步测试函数，测试在单个文档中提取实体时是否返回正确的实体信息
    async def test_run_extract_entities_single_document_correct_entities_returned(self):
        
        # 调用run_extract_entities函数，传入指定参数，返回提取的结果
        results = await run_extract_entities(
            docs=[Document("test_text", "1")],
            entity_types=["person"],
            reporter=None,
            args={
                "prechunked": True,
                "max_gleanings": 0,
                "summarize_descriptions": False,
            },
            llm=create_mock_llm(
                responses=[
                    """
                    ("entity"<|>TEST_ENTITY_1<|>COMPANY<|>TEST_ENTITY_1 is a test company)
                    ##
                    ("entity"<|>TEST_ENTITY_2<|>COMPANY<|>TEST_ENTITY_2 owns TEST_ENTITY_1 and also shares an address with TEST_ENTITY_1)
                    ##
                    ("entity"<|>TEST_ENTITY_3<|>PERSON<|>TEST_ENTITY_3 is director of TEST_ENTITY_1)
                    ##
                    ("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_2<|>TEST_ENTITY_1 and TEST_ENTITY_2 are related because TEST_ENTITY_1 is 100% owned by TEST_ENTITY_2 and the two companies also share the same address)<|>2)
                    ##
                    ("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_3<|>TEST_ENTITY_1 and TEST_ENTITY_3 are related because TEST_ENTITY_3 is director of TEST_ENTITY_1<|>1))
                    """.strip()
                ]
            ),
        )

        # 使用断言比较两个排序后的列表是否相等
        assert sorted(["TEST_ENTITY_1", "TEST_ENTITY_2", "TEST_ENTITY_3"]) == sorted([
            entity["name"] for entity in results.entities
        ])

    # 定义异步测试函数，测试在多个文档中提取实体时是否返回正确的实体信息
    async def test_run_extract_entities_multiple_documents_correct_entities_returned(
        self,
        results = await run_extract_entities(
            # 调用异步函数 run_extract_entities，用于从文档中提取实体信息
            docs=[Document("text_1", "1"), Document("text_2", "2")],
            # 提供两个文档作为输入，每个文档包含一个文本和一个编号
            entity_types=["person"],  # 指定要提取的实体类型为 "person"
            reporter=None,  # 报告器参数设置为 None
            args={
                # 提供其他参数的字典
                "prechunked": True,  # 使用预分块的文本输入
                "max_gleanings": 0,  # 最大获取信息数为 0
                "summarize_descriptions": False,  # 不进行描述总结
            },
            llm=create_mock_llm(
                # 使用模拟的语言模型对象 llm，传入以下模拟响应
                responses=[
                    """
                    ("entity"<|>TEST_ENTITY_1<|>COMPANY<|>TEST_ENTITY_1 is a test company)
                    ##
                    ("entity"<|>TEST_ENTITY_2<|>COMPANY<|>TEST_ENTITY_2 owns TEST_ENTITY_1 and also shares an address with TEST_ENTITY_1)
                    ##
                    ("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_2<|>TEST_ENTITY_1 and TEST_ENTITY_2 are related because TEST_ENTITY_1 is 100% owned by TEST_ENTITY_2 and the two companies also share the same address)<|>2)
                    ##
                    """.strip(),
                    """
                    ("entity"<|>TEST_ENTITY_1<|>COMPANY<|>TEST_ENTITY_1 is a test company)
                    ##
                    ("entity"<|>TEST_ENTITY_3<|>PERSON<|>TEST_ENTITY_3 is director of TEST_ENTITY_1)
                    ##
                    ("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_3<|>TEST_ENTITY_1 and TEST_ENTITY_3 are related because TEST_ENTITY_3 is director of TEST_ENTITY_1<|>1))
                    """.strip(),
                ]
            ),
        )

        # self.assertItemsEqual isn't available yet, or I am just silly
        # so we sort the lists and compare them
        # 断言：对从结果中提取的实体名称列表进行排序，与预期的实体名称列表进行比较
        assert sorted(["TEST_ENTITY_1", "TEST_ENTITY_2", "TEST_ENTITY_3"]) == sorted([
            entity["name"] for entity in results.entities
        ])
    # 异步测试函数，用于验证从多个文档中提取实体时是否返回正确的边缘信息
    async def test_run_extract_entities_multiple_documents_correct_edges_returned(self):
        results = await run_extract_entities(
            # 传入两个文档对象及其内容作为提取实体的输入
            docs=[Document("text_1", "1"), Document("text_2", "2")],
            # 指定希望提取的实体类型为 "person"
            entity_types=["person"],
            # 无报告器传入
            reporter=None,
            # 附加参数设置：启用预分块，最大获取信息数为 0，不要汇总描述信息
            args={
                "prechunked": True,
                "max_gleanings": 0,
                "summarize_descriptions": False,
            },
            # 创建模拟的语言模型，用于生成响应数据
            llm=create_mock_llm(
                # 模拟的语言模型响应数据列表
                responses=[
                    """
                    ("entity"<|>TEST_ENTITY_1<|>COMPANY<|>TEST_ENTITY_1 is a test company)
                    ##
                    ("entity"<|>TEST_ENTITY_2<|>COMPANY<|>TEST_ENTITY_2 owns TEST_ENTITY_1 and also shares an address with TEST_ENTITY_1)
                    ##
                    ("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_2<|>TEST_ENTITY_1 and TEST_ENTITY_2 are related because TEST_ENTITY_1 is 100% owned by TEST_ENTITY_2 and the two companies also share the same address)<|>2)
                    ##
                    """.strip(),
                    """
                    ("entity"<|>TEST_ENTITY_1<|>COMPANY<|>TEST_ENTITY_1 is a test company)
                    ##
                    ("entity"<|>TEST_ENTITY_3<|>PERSON<|>TEST_ENTITY_3 is director of TEST_ENTITY_1)
                    ##
                    ("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_3<|>TEST_ENTITY_1 and TEST_ENTITY_3 are related because TEST_ENTITY_3 is director of TEST_ENTITY_1<|>1))
                    """.strip(),
                ]
            ),
        )

        # 使用断言验证返回的结果中是否包含有效的图形表示
        assert results.graphml_graph is not None, "No graphml graph returned!"
        # 解析 graphml 格式的图形数据
        graph = nx.parse_graphml(results.graphml_graph)  # type: ignore

        # 将边缘数据转换为字符串列表以便进行可视化比较
        edges_str = sorted([f"{edge[0]} -> {edge[1]}" for edge in graph.edges])
        # 使用断言比较排序后的边缘列表是否与预期的相同
        assert edges_str == sorted([
            "TEST_ENTITY_1 -> TEST_ENTITY_2",
            "TEST_ENTITY_1 -> TEST_ENTITY_3",
        ])

    # 下一个测试函数开始了，用于验证从多个文档中提取实体时是否正确映射实体源ID
    async def test_run_extract_entities_multiple_documents_correct_entity_source_ids_mapped(
        self,
    ):
        results = await run_extract_entities(
            docs=[Document("text_1", "1"), Document("text_2", "2")],  # 准备两个文档，每个文档有一个唯一的标识符
            entity_types=["person"],  # 指定需要提取的实体类型为人物
            reporter=None,  # 报告器设置为 None，表示没有特定的报告器
            args={
                "prechunked": True,  # 使用预分块，可能是指提前对文本进行分块处理
                "max_gleanings": 0,  # 最大的信息提取数量为 0，可能是暂时不提取具体信息
                "summarize_descriptions": False,  # 不对描述进行汇总
            },
            llm=create_mock_llm(  # 创建模拟的语言理解模型
                responses=[
                    """
                    ("entity"<|>TEST_ENTITY_1<|>COMPANY<|>TEST_ENTITY_1 is a test company)
                    ##
                    ("entity"<|>TEST_ENTITY_2<|>COMPANY<|>TEST_ENTITY_2 owns TEST_ENTITY_1 and also shares an address with TEST_ENTITY_1)
                    ##
                    ("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_2<|>TEST_ENTITY_1 and TEST_ENTITY_2 are related because TEST_ENTITY_1 is 100% owned by TEST_ENTITY_2 and the two companies also share the same address)<|>2)
                    ##
                    """.strip(),
                    """
                    ("entity"<|>TEST_ENTITY_1<|>COMPANY<|>TEST_ENTITY_1 is a test company)
                    ##
                    ("entity"<|>TEST_ENTITY_3<|>PERSON<|>TEST_ENTITY_3 is director of TEST_ENTITY_1)
                    ##
                    ("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_3<|>TEST_ENTITY_1 and TEST_ENTITY_3 are related because TEST_ENTITY_3 is director of TEST_ENTITY_1<|>1))
                    """.strip(),
                ]
            ),
        )

        assert results.graphml_graph is not None, "No graphml graph returned!"  # 断言确保返回的结果中包含 GraphML 图形
        graph = nx.parse_graphml(results.graphml_graph)  # type: ignore

        # TODO: The edges might come back in any order, but we're assuming they're coming
        # back in the order that we passed in the docs, that might not be true
        assert (
            graph.nodes["TEST_ENTITY_3"].get("source_id") == "2"
        )  # 断言：TEST_ENTITY_3 应该只在 2 中出现
        assert (
            graph.nodes["TEST_ENTITY_2"].get("source_id") == "1"
        )  # 断言：TEST_ENTITY_2 应该只在 1 中出现
        assert sorted(
            graph.nodes["TEST_ENTITY_1"].get("source_id").split(",")
        ) == sorted(["1", "2"])  # 断言：TEST_ENTITY_1 应该同时在 1 和 2 中出现

    async def test_run_extract_entities_multiple_documents_correct_edge_source_ids_mapped(
        self,
        results = await run_extract_entities(
            # 调用异步函数 run_extract_entities，传入以下参数:
            docs=[Document("text_1", "1"), Document("text_2", "2")],  # 传入两个文档对象，每个包含文本和标识符
            entity_types=["person"],  # 期望提取的实体类型列表，这里只包含 "person"
            reporter=None,  # 报告器对象为空
            args={
```
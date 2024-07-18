# `.\graphrag\tests\unit\indexing\workflows\test_load.py`

```py
# 导入单元测试库 unittest
import unittest

# 导入 pytest，用于测试异常情况
import pytest

# 导入需要测试的模块和类
from graphrag.index.config import PipelineWorkflowReference
from graphrag.index.errors import UnknownWorkflowError
from graphrag.index.workflows.load import create_workflow, load_workflows

# 导入测试辅助函数
from .helpers import mock_verbs, mock_workflows

# 定义测试类 TestCreateWorkflow，继承自 unittest.TestCase
class TestCreateWorkflow(unittest.TestCase):
    
    # 测试用例：测试包含步骤的工作流创建不应失败
    def test_workflow_with_steps_should_not_fail(self):
        create_workflow(
            "workflow_with_steps",  # 工作流名称
            [  # 包含一个步骤的列表，每个步骤包括动作（verb）和参数（args）
                {
                    "verb": "mock_verb",  # 模拟的动作名称
                    "args": {  # 参数字典，这里仅有一个测试列名参数
                        "column": "test",
                    },
                }
            ],
            config=None,  # 配置参数为空
            additional_verbs=mock_verbs,  # 额外的动作列表
        )

    # 测试用例：测试不存在的工作流且没有步骤时应该抛出异常
    def test_non_existent_workflow_without_steps_should_crash(self):
        # 因为没有名为 "test" 的工作流，并且用户没有提供步骤，所以应该抛出异常
        with pytest.raises(UnknownWorkflowError):
            create_workflow("test", None, config=None, additional_verbs=mock_verbs)

    # 测试用例：测试已存在的工作流创建不应失败
    def test_existing_workflow_should_not_crash(self):
        create_workflow(
            "mock_workflow",  # 存在的工作流名称
            None,  # 没有指定步骤
            config=None,  # 配置参数为空
            additional_verbs=mock_verbs,  # 额外的动作列表
            additional_workflows=mock_workflows,  # 额外的工作流列表
        )


# 定义测试类 TestLoadWorkflows，继承自 unittest.TestCase
class TestLoadWorkflows(unittest.TestCase):
    
    # 测试用例：测试加载不存在的工作流应该抛出异常
    def test_non_existent_workflow_should_crash(self):
        with pytest.raises(UnknownWorkflowError):
            load_workflows(
                [
                    PipelineWorkflowReference(
                        name="some_workflow_that_does_not_exist",  # 不存在的工作流名称
                        config=None,  # 配置参数为空
                    )
                ],
                additional_workflows=mock_workflows,  # 额外的工作流列表
                additional_verbs=mock_verbs,  # 额外的动作列表
            )

    # 测试用例：测试加载单个存在的工作流不应失败
    def test_single_workflow_should_not_crash(self):
        load_workflows(
            [
                PipelineWorkflowReference(
                    name="mock_workflow",  # 存在的工作流名称
                    config=None,  # 配置参数为空
                )
            ],
            additional_workflows=mock_workflows,  # 额外的工作流列表
            additional_verbs=mock_verbs,  # 额外的动作列表
        )

    # 测试用例：测试加载多个存在的工作流不应失败
    def test_multiple_workflows_should_not_crash(self):
        load_workflows(
            [
                PipelineWorkflowReference(
                    name="mock_workflow",  # 存在的工作流名称
                    config=None,  # 配置参数为空
                ),
                PipelineWorkflowReference(
                    name="mock_workflow_2",  # 另一个存在的工作流名称
                    config=None,  # 配置参数为空
                ),
            ],
            additional_workflows=mock_workflows,  # 额外的工作流列表
            additional_verbs=mock_verbs,  # 额外的动作列表
        )
    # 定义一个测试方法，验证两个相互依赖的工作流在加载后能够按正确顺序提供
    def test_two_interdependent_workflows_should_provide_correct_order(self):
        # 载入工作流，返回排序后的工作流列表和依赖关系
        ordered_workflows, _deps = load_workflows(
            [
                # 第一个工作流参考，包含一个步骤，依赖第二个工作流
                PipelineWorkflowReference(
                    name="interdependent_workflow_1",
                    steps=[
                        {
                            "verb": "mock_verb",
                            "args": {
                                "column": "test",
                            },
                            "input": {
                                "source": "workflow:interdependent_workflow_2"
                            },  # 这个步骤依赖第二个工作流，因此在载入后应该排在第一个位置
                        }
                    ],
                ),
                # 第二个工作流参考，包含一个步骤
                PipelineWorkflowReference(
                    name="interdependent_workflow_2",
                    steps=[
                        {
                            "verb": "mock_verb",
                            "args": {
                                "column": "test",
                            },
                        }
                    ],
                ),
            ],
            # mock_workflows 列表中包含上述两个工作流
            additional_workflows=mock_workflows,
            additional_verbs=mock_verbs,
        )

        # 断言排序后的工作流列表长度应为2
        assert len(ordered_workflows) == 2
        # 断言排序后的第一个工作流的名称为 "interdependent_workflow_2"
        assert ordered_workflows[0].workflow.name == "interdependent_workflow_2"
        # 断言排序后的第二个工作流的名称为 "interdependent_workflow_1"
        assert ordered_workflows[1].workflow.name == "interdependent_workflow_1"
    def test_three_interdependent_workflows_should_provide_correct_order(self):
        # 调用 load_workflows 函数加载三个相互依赖的工作流，返回有序的工作流列表和依赖关系字典
        ordered_workflows, _deps = load_workflows(
            [
                PipelineWorkflowReference(
                    name="interdependent_workflow_3",
                    steps=[
                        {
                            "verb": "mock_verb",
                            "args": {
                                "column": "test",
                            },
                        }
                    ],
                ),
                PipelineWorkflowReference(
                    name="interdependent_workflow_1",
                    steps=[
                        {
                            "verb": "mock_verb",
                            "args": {
                                "column": "test",
                            },
                            "input": {"source": "workflow:interdependent_workflow_2"},
                        }
                    ],
                ),
                PipelineWorkflowReference(
                    name="interdependent_workflow_2",
                    steps=[
                        {
                            "verb": "mock_verb",
                            "args": {
                                "column": "test",
                            },
                            "input": {"source": "workflow:interdependent_workflow_3"},
                        }
                    ],
                ),
            ],
            # mock_workflows 列表包含以上定义的工作流
            additional_workflows=mock_workflows,
            additional_verbs=mock_verbs,
        )

        # 定义预期的工作流执行顺序
        order = [
            "interdependent_workflow_3",
            "interdependent_workflow_2",
            "interdependent_workflow_1",
        ]
        # 断言加载的工作流顺序与预期顺序一致
        assert [x.workflow.name for x in ordered_workflows] == order

    def test_two_workflows_dependent_on_another_single_workflow_should_provide_correct_order(
        self,
    ):
        ordered_workflows, _deps = load_workflows(
            [
                # 创建一个包含依赖关系的工作流列表，按指定顺序加载
                PipelineWorkflowReference(
                    name="interdependent_workflow_3",
                    steps=[
                        {
                            "verb": "mock_verb",
                            "args": {
                                "column": "test",
                            },
                        }
                    ],
                ),
                # 创建第一个依赖于第三个工作流的工作流引用
                PipelineWorkflowReference(
                    name="interdependent_workflow_1",
                    steps=[
                        {
                            "verb": "mock_verb",
                            "args": {
                                "column": "test",
                            },
                            "input": {"source": "workflow:interdependent_workflow_3"},
                        }
                    ],
                ),
                # 创建第二个依赖于第三个工作流的工作流引用
                PipelineWorkflowReference(
                    name="interdependent_workflow_2",
                    steps=[
                        {
                            "verb": "mock_verb",
                            "args": {
                                "column": "test",
                            },
                            "input": {"source": "workflow:interdependent_workflow_3"},
                        }
                    ],
                ),
            ],
            # 将模拟的工作流和动词添加到加载工作流函数中的附加参数中
            additional_workflows=mock_workflows,
            additional_verbs=mock_verbs,
        )

        # 断言：已正确加载并排序了工作流列表
        assert len(ordered_workflows) == 3
        assert ordered_workflows[0].workflow.name == "interdependent_workflow_3"

        # 断言：其余两个工作流的顺序无关紧要，但它们必须在列表中
        assert ordered_workflows[1].workflow.name in [
            "interdependent_workflow_1",
            "interdependent_workflow_2",
        ]
        assert ordered_workflows[2].workflow.name in [
            "interdependent_workflow_1",
            "interdependent_workflow_2",
        ]
```
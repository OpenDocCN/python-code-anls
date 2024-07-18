# `.\graphrag\graphrag\index\workflows\load.py`

```py
# 著作权声明，版权归 2024 年微软公司所有，遵循 MIT 许可证
# 导入所需模块和类型声明

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from datashaper import Workflow  # 导入 Workflow 类

from graphrag.index.errors import (
    NoWorkflowsDefinedError,
    UndefinedWorkflowError,
    UnknownWorkflowError,
)
from graphrag.index.utils import topological_sort

from .default_workflows import default_workflows as _default_workflows  # 导入默认工作流
from .typing import VerbDefinitions, WorkflowDefinitions, WorkflowToRun  # 导入类型定义

if TYPE_CHECKING:
    from graphrag.index.config import (
        PipelineWorkflowConfig,
        PipelineWorkflowReference,
        PipelineWorkflowStep,
    )

# 全局变量：用于记录匿名工作流的数量
anonymous_workflow_count = 0

# 定义函数日志记录器
log = logging.getLogger(__name__)


class LoadWorkflowResult(NamedTuple):
    """A workflow loading result object."""

    workflows: list[WorkflowToRun]
    """The loaded workflow names in the order they should be run."""

    dependencies: dict[str, list[str]]
    """A dictionary of workflow name to workflow dependencies."""


def load_workflows(
    workflows_to_load: list[PipelineWorkflowReference],
    additional_verbs: VerbDefinitions | None = None,
    additional_workflows: WorkflowDefinitions | None = None,
    memory_profile: bool = False,
) -> LoadWorkflowResult:
    """Load the given workflows.

    Args:
        - workflows_to_load - The workflows to load
        - additional_verbs - The list of custom verbs available to the workflows
        - additional_workflows - The list of custom workflows
        - memory_profile - Flag indicating whether to enable memory profiling
    Returns:
        - output[0] - The loaded workflow names in the order they should be run
        - output[1] - A dictionary of workflow name to workflow dependencies
    """
    # 初始化空的工作流图
    workflow_graph: dict[str, WorkflowToRun] = {}

    # 遍历待加载的工作流引用列表
    global anonymous_workflow_count
    for reference in workflows_to_load:
        name = reference.name
        is_anonymous = name is None or name.strip() == ""
        if is_anonymous:
            # 处理匿名工作流的情况，为其生成一个唯一名称
            name = f"Anonymous Workflow {anonymous_workflow_count}"
            anonymous_workflow_count += 1
        name = cast(str, name)

        config = reference.config
        # 创建工作流对象并加入到工作流图中
        workflow = create_workflow(
            name or "MISSING NAME!",
            reference.steps,
            config,
            additional_verbs,
            additional_workflows,
        )
        workflow_graph[name] = WorkflowToRun(workflow, config=config or {})

    # 后期填充任何缺失的工作流
    # 遍历工作流图中所有的工作流名称
    for name in list(workflow_graph.keys()):
        # 获取当前工作流的详细信息
        workflow = workflow_graph[name]
        # 提取当前工作流的依赖关系，并去除依赖名称前缀"workflow:"
        deps = [
            d.replace("workflow:", "")
            for d in workflow.workflow.dependencies
            if d.startswith("workflow:")
        ]
        # 遍历当前工作流的每个依赖
        for dependency in deps:
            # 如果依赖在工作流图中不存在，则创建一个新的工作流对象并加入到图中
            if dependency not in workflow_graph:
                # 创建依赖的参考对象，包含依赖名称和当前工作流的配置信息
                reference = {"name": dependency, **workflow.config}
                # 根据依赖创建工作流对象，并加入到工作流图中
                workflow_graph[dependency] = WorkflowToRun(
                    workflow=create_workflow(
                        dependency,
                        config=reference,
                        additional_verbs=additional_verbs,
                        additional_workflows=additional_workflows,
                        memory_profile=memory_profile,
                    ),
                    config=reference,
                )

    # 根据依赖关系的顺序运行工作流
    def filter_wf_dependencies(name: str) -> list[str]:
        # 提取当前工作流的外部依赖，去除依赖名称前缀"workflow:"
        externals = [
            e.replace("workflow:", "")
            for e in workflow_graph[name].workflow.dependencies
        ]
        # 返回当前工作流中存在于工作流图中的外部依赖列表
        return [e for e in externals if e in workflow_graph]

    # 构建任务图，包含每个工作流及其外部依赖列表
    task_graph = {name: filter_wf_dependencies(name) for name in workflow_graph}
    # 对任务图进行拓扑排序，获取工作流的执行顺序
    workflow_run_order = topological_sort(task_graph)
    # 根据执行顺序获取对应的工作流对象列表
    workflows = [workflow_graph[name] for name in workflow_run_order]
    # 记录工作流执行顺序
    log.info("Workflow Run Order: %s", workflow_run_order)
    # 返回加载工作流的结果对象，包含工作流列表和依赖关系图
    return LoadWorkflowResult(workflows=workflows, dependencies=task_graph)
# 从给定的配置创建一个工作流
def create_workflow(
    name: str,
    steps: list[PipelineWorkflowStep] | None = None,
    config: PipelineWorkflowConfig | None = None,
    additional_verbs: VerbDefinitions | None = None,
    additional_workflows: WorkflowDefinitions | None = None,
    memory_profile: bool = False,
) -> Workflow:
    """Create a workflow from the given config."""
    # 合并默认工作流和额外提供的工作流定义
    additional_workflows = {
        **_default_workflows,
        **(additional_workflows or {}),
    }
    # 如果未提供步骤，则根据名称、配置和额外工作流获取步骤列表
    steps = steps or _get_steps_for_workflow(name, config, additional_workflows)
    # 去除已禁用的步骤
    steps = _remove_disabled_steps(steps)
    # 返回一个 Workflow 对象，其中包括动词定义、工作流名称和步骤列表等信息
    return Workflow(
        verbs=additional_verbs or {},
        schema={
            "name": name,
            "steps": steps,
        },
        validate=False,
        memory_profile=memory_profile,
    )


# 根据给定的工作流配置获取步骤列表
def _get_steps_for_workflow(
    name: str | None,
    config: PipelineWorkflowConfig | None,
    workflows: dict[str, Callable] | None,
) -> list[PipelineWorkflowStep]:
    """Get the steps for the given workflow config."""
    # 如果配置不为 None 并且包含步骤定义，则直接返回配置中的步骤列表
    if config is not None and "steps" in config:
        return config["steps"]

    # 如果未定义工作流字典，则抛出异常 NoWorkflowsDefinedError
    if workflows is None:
        raise NoWorkflowsDefinedError

    # 如果未指定工作流名称，则抛出异常 UndefinedWorkflowError
    if name is None:
        raise UndefinedWorkflowError

    # 如果给定的工作流名称不在工作流字典中，则抛出异常 UnknownWorkflowError
    if name not in workflows:
        raise UnknownWorkflowError(name)

    # 否则，调用相应的工作流函数获取步骤列表，并返回
    return workflows[name](config or {})


# 去除步骤列表中已禁用的步骤
def _remove_disabled_steps(
    steps: list[PipelineWorkflowStep],
) -> list[PipelineWorkflowStep]:
    return [step for step in steps if step.get("enabled", True)]
```
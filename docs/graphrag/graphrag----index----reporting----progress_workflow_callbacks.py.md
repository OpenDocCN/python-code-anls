# `.\graphrag\graphrag\index\reporting\progress_workflow_callbacks.py`

```py
# 2024年版权所有 Microsoft Corporation.
# 根据 MIT 许可证授权

"""一个工作流回调管理器，向 ProgressReporter 发出更新。"""

from typing import Any

from datashaper import ExecutionNode, NoopWorkflowCallbacks, Progress, TableContainer

from graphrag.index.progress import ProgressReporter


class ProgressWorkflowCallbacks(NoopWorkflowCallbacks):
    """一个回调管理器，委托给 ProgressReporter 处理。"""

    _root_progress: ProgressReporter
    _progress_stack: list[ProgressReporter]

    def __init__(self, progress: ProgressReporter) -> None:
        """创建一个新的 ProgressWorkflowCallbacks 实例。"""
        self._progress = progress
        self._progress_stack = [progress]

    def _pop(self) -> None:
        """从进度栈中弹出当前进度。"""
        self._progress_stack.pop()

    def _push(self, name: str) -> None:
        """将指定名称的进度推入进度栈。"""
        self._progress_stack.append(self._latest.child(name))

    @property
    def _latest(self) -> ProgressReporter:
        """获取当前栈顶的最新进度。"""
        return self._progress_stack[-1]

    def on_workflow_start(self, name: str, instance: object) -> None:
        """在工作流开始时执行此回调。"""
        self._push(name)

    def on_workflow_end(self, name: str, instance: object) -> None:
        """在工作流结束时执行此回调。"""
        self._pop()

    def on_step_start(self, node: ExecutionNode, inputs: dict[str, Any]) -> None:
        """每次步骤开始时执行此回调。"""
        verb_id_str = f" ({node.node_id})" if node.has_explicit_id else ""
        self._push(f"Verb {node.verb.name}{verb_id_str}")
        self._latest(Progress(percent=0))

    def on_step_end(self, node: ExecutionNode, result: TableContainer | None) -> None:
        """每次步骤结束时执行此回调。"""
        self._pop()

    def on_step_progress(self, node: ExecutionNode, progress: Progress) -> None:
        """处理步骤进度发生变化时的回调。"""
        self._latest(progress)
```
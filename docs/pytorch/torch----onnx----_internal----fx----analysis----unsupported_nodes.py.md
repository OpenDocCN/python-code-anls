# `.\pytorch\torch\onnx\_internal\fx\analysis\unsupported_nodes.py`

```
# mypy: allow-untyped-defs
# 引入未类型定义函数的兼容性声明
from __future__ import annotations

# 引入数据类和字典类型
import dataclasses
from typing import Dict

# 从torch.onnx._internal.fx中引入_pass, diagnostics, registration模块
from torch.onnx._internal.fx import _pass, diagnostics, registration


@dataclasses.dataclass
# 定义一个数据类UnsupportedFxNodesAnalysisResult，继承自_pass.AnalysisResult类
class UnsupportedFxNodesAnalysisResult(_pass.AnalysisResult):
    # 不支持的FX节点到目标映射的字典，键为字符串，值为映射到空值的字典
    unsupported_op_to_target_mapping: Dict[str, Dict[str, None]]


class UnsupportedFxNodesAnalysis(_pass.Analysis):
    """An analysis that detects unsupported FX nodes in the graph."""

    def _lint(
        self,
        analysis_result: UnsupportedFxNodesAnalysisResult,
        diagnostic_level: diagnostics.infra.Level,
    ):
        """Lint the graph and emit diagnostics if unsupported FX nodes are found."""
        # 如果未发现不支持的FX节点映射，则直接返回
        if not analysis_result.unsupported_op_to_target_mapping:
            return

        # 规范化操作与目标映射，将不支持的操作及其目标列表转换为字典形式
        normalized_op_targets_map = {
            op: list(targets.keys())
            for op, targets in analysis_result.unsupported_op_to_target_mapping.items()
        }

        # 使用规则unsupported_fx_node_analysis创建诊断消息
        rule = diagnostics.rules.unsupported_fx_node_analysis
        diagnostic = diagnostics.Diagnostic(
            rule,
            level=diagnostic_level,
            message=rule.format_message(normalized_op_targets_map),
        )
        # 记录和抛出诊断消息，如果存在错误
        self.diagnostic_context.log_and_raise_if_error(diagnostic)

    def analyze(
        self, diagnostic_level: diagnostics.infra.Level
    ):
        # 分析函数，接收一个诊断级别作为参数
    ) -> UnsupportedFxNodesAnalysisResult:
        """Analyze the graph, emit diagnostics and return a result that contains unsupported FX nodes.

        Args:
            diagnostic_level: The diagnostic level to use when emitting diagnostics.

        Returns:
            An analysis result that contains unsupported FX nodes.

        Raises:
            RuntimeErrorWithDiagnostic: If diagnostics are emitted and the diagnostic
                level is `ERROR`.
        """
        # 初始化一个空字典，用于存储操作到目标映射关系
        op_to_target_mapping: Dict[str, Dict[str, None]] = {}
        # 遍历模块的图中的每个节点
        for node in self.module.graph.nodes:
            # 如果节点的操作是函数调用
            if node.op == "call_function":
                # 获取节点对应的内部操作名称
                internal_opname: registration.OpName = (
                    self.onnxfunction_dispatcher._get_aten_name(
                        node=node, diagnostic_context=self.diagnostic_context
                    )
                )
                # 查询是否注册了当前操作的重载版本
                overload_registration = (
                    self.onnxfunction_dispatcher.onnx_registry.is_registered_op(
                        namespace=internal_opname.namespace,
                        op_name=internal_opname.op_name,
                        overload=internal_opname.overload,
                    )
                )
                # 如果当前重载版本未注册，则回退到默认重载版本
                default_registration = (
                    self.onnxfunction_dispatcher.onnx_registry.is_registered_op(
                        namespace=internal_opname.namespace,
                        op_name=internal_opname.op_name,
                        overload=None,
                    )
                )
                # 如果既没有注册当前重载版本，也没有默认重载版本，则将节点的操作和目标添加到映射中
                if not overload_registration and not default_registration:
                    op_to_target_mapping.setdefault(node.op, {}).setdefault(
                        str(node.target), None
                    )

        # 创建一个分析结果对象，包含不受支持的 FX 节点映射
        analysis_result = UnsupportedFxNodesAnalysisResult(op_to_target_mapping)
        # 对分析结果进行静态分析，根据诊断级别发出诊断信息
        self._lint(analysis_result, diagnostic_level)
        # 返回分析结果对象
        return analysis_result
```
# `.\pytorch\torch\_export\passes\functionalize_side_effectful_ops_pass.py`

```
import copy  # 导入copy模块，用于深拷贝对象
from typing import Dict, Optional, Tuple, List  # 引入类型提示，用于静态类型检查

import torch  # 导入PyTorch库
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse, PassResult, Argument  # 导入PassBase相关类
from torch._export.pass_infra.node_metadata import NodeMetadata  # 导入节点元数据类
from torch._export.pass_infra.proxy_value import ProxyValue  # 导入代理值类
from torch._ops import OpOverload  # 导入操作重载类

aten = torch.ops.aten  # 获取torch操作

_NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS: Dict[OpOverload, OpOverload] = {
    aten.sym_constrain_range.default: aten._functional_sym_constrain_range,
    aten._assert_async.msg: aten._functional_assert_async.msg,
}
# 将非函数式具有副作用的操作映射到对应的函数式操作

class _FunctionalizeSideEffectfulOpsPass(_ExportPassBaseDeprecatedDoNotUse):
    """
    Functionalize ops with side effect in graph module by replacing the op with
    functional version of it. A new dependency token (`dep_token`) will be
    created and propagated through functional ops to output.
    For example:
    ```
    def f(x):
        sym_constrain_range(x.shape[0], min=1, max=3)
        return x.add(3)
    ```
    Will be transformed to:
    ```
    def f(x):
        dep_token0 = _make_dep_token()
        dep_token1 = _functional_sym_constrain_range(
            x.shape[0], min=1, max=3, dep_token=dep_token0
        )

        return x.add(3), dep_token1
    ```
    """
    def __init__(self) -> None:
        super().__init__()  # 调用父类构造函数
        self._dep_token: Optional[ProxyValue] = None  # 初始化依赖令牌为可选的代理值对象或None
        self._next_dep_token_index: Optional[int] = None  # 初始化下一个依赖令牌索引为可选的整数或None

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # 如果图中没有非函数式断言，早期返回
        if not any(
            n.target in _NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS
            for n in graph_module.graph.nodes
        ):
            return PassResult(graph_module=graph_module, modified=False)
        # 深拷贝图模块对象
        gm = copy.deepcopy(graph_module)
        self._dep_token = None  # 初始化依赖令牌为None
        self._next_dep_token_index = None  # 初始化下一个依赖令牌索引为None
        return super().call(gm)  # 调用父类的call方法并返回结果

    def call_operator(
        self,
        op: OpOverload,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        # 如果操作不在无副作用函数到有副作用函数的映射中，调用父类的操作符方法
        if op not in _NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS:
            return super().call_operator(op, args, kwargs, meta)

        # 如果依赖标记为空，则创建一个新的依赖标记节点
        if self._dep_token is None:
            self._dep_token = super().call_operator(
                aten._make_dep_token,
                args=(),
                kwargs={},
                meta=self._create_dummy_node_metadata(),
            )
            # 将依赖标记节点命名为"dep_token0"
            self._dep_token.node.name = "dep_token0"
            self._next_dep_token_index = 1

        # 调用父类的操作符方法，使用功能性操作函数处理操作符
        self._dep_token = super().call_operator(
            _NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS[op],
            args=args,
            kwargs={**kwargs, "dep_token": self._dep_token},
            meta=meta,
        )
        # 确保下一个依赖标记索引已定义
        assert self._next_dep_token_index is not None
        # 更新当前依赖标记节点的名称为"dep_token{index}"
        self._dep_token.node.name = f"dep_token{self._next_dep_token_index}"
        # 递增下一个依赖标记索引
        self._next_dep_token_index += 1

        # 返回当前依赖标记节点
        return self._dep_token

    # 输出方法，接受结果列表和节点元数据作为参数，返回代理值
    def output(self, results: List[Argument], meta: NodeMetadata) -> ProxyValue:
        # 确保依赖标记不为空
        assert self._dep_token is not None

        # 调用父类的输出方法，包括依赖标记在内的结果列表和节点元数据
        return super().output(results=(*results, self._dep_token), meta=meta)  # type: ignore[arg-type]
```
# `.\pytorch\torch\distributed\fsdp\_dynamo_utils.py`

```py
# mypy: allow-untyped-defs
# 导入所需模块
from typing import Set

# 导入 PyTorch 的神经网络模块
import torch.nn as nn


def _annotate_modules_for_dynamo(
    module: nn.Module,
    ignored_modules: Set[nn.Module],
    use_orig_params: bool,
):
    """
    Annotates the submodules in ``module`` 's tree, except those in
    ``ignored_modules``, indicating that the submodules are FSDP-managed and
    saving the ``use_orig_params`` setting passed to the FSDP constructor.
    """
    # 遍历模块的所有子模块，标注为 FSDP 管理，除了被忽略的模块
    for submodule in module.modules():
        if submodule not in ignored_modules:
            """
            [note: Dynamo treats FSDP wrapped modules as UnspecializedNNModule]

            Dynamo doesn't get to see this instance (FullyShardedDataParallel) during tracing, since
            it skips tracing all the torch.distributed.fsdp code.
                - Why? Running the FSDP code eagerly avoids lots of issues trying to trace complex hooks, and also
                gets us graph-breaks on FSDP module boundaries which we want anyway for comm ops.
                - However, we _also_ want dynamo to treat the wrapped module inside FSDP 'unspecially' (*),
                and we need a way to indicate to dynamo which modules are wrapped by FSDP.

            (*) UnspecializedNNModules in dynamo are traced-through without any assumptions, and with thorough
            guards.  NNModules otherwise are 'specialized', meaning there is less overhead due to assuming
            their code is well-behaved.

            One particular issue with specialized NNModules for FSDP is that the
            views created for orig_params are captured into the compiled graph on the first iteration, and while
            they are always going to point to the correct flatparameter and give correct results, their order
            of creation influences the order of backward execution, preventing overlap of comm and computation
            during backward.  We need to _use_ the new parameter views created on each forward iteration, in
            order for backward to interleave hooks with compute per layer.  UnspecializedNNModule lets us achieve
            this by capturing the module code more 'functionally' and passing parameters in as inputs each time.
            """
            # 标记子模块为 FSDP 管理的模块
            submodule._is_fsdp_managed_module = True  # type: ignore[assignment]

            # Dynamo 只支持使用 use_orig_params=True 的 FSDP
            # 这是一种不太正式的方式，但我想不到另一种方法来向 dynamo 添加这个断言
            # 因为 Dynamo 跳过所有 FSDP 代码帧，无法直接检查 FSDP 模块
            submodule._fsdp_use_orig_params = use_orig_params  # type: ignore[assignment]
```
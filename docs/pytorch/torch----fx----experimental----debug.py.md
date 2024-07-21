# `.\pytorch\torch\fx\experimental\debug.py`

```
# mypy: allow-untyped-defs
# 导入 torch.fx 库中的 fx 模块
import torch.fx as fx

# 定义函数 set_trace，用于在 gm 的生成的 Python 代码中设置断点，当运行 gm 时进入 pdb 调试器
def set_trace(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Sets a breakpoint in `gm`'s generated python code. It drops into pdb when
    `gm` gets run.

    Args:
        gm: graph module to insert breakpoint. It is then recompiled for it to
            take effect.

    Returns:
        the `gm` with breakpoint inserted.
    """
    # 定义内部函数 insert_pdb，用于在给定代码体 body 前插入 pdb.set_trace() 语句
    def insert_pdb(body):
        return ["import pdb; pdb.set_trace()\n", *body]

    # 在 gm 的图生成代码时，注册一个新的代码转换器
    with gm.graph.on_generate_code(
        make_transformer=lambda cur_transform: (
            # 新的代码转换器来注册
            lambda body: (
                insert_pdb(
                    cur_transform(body) if cur_transform
                    else body
                )
            )
        )
    ):
        # 重新编译 gm 以使插入的断点生效
        gm.recompile()

    # 返回带有插入断点的 gm
    return gm
```
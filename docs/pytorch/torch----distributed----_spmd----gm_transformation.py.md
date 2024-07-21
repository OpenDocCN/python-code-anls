# `.\pytorch\torch\distributed\_spmd\gm_transformation.py`

```py
from typing import Callable

from torch import fx
from torch.distributed._spmd.graph_optimization import (
    comm_fusion_with_concat,
    enable_graph_optimization_dump,
    remove_copy_from_optimizer,
    schedule_comm_wait,
)
from torch.distributed._spmd.graph_utils import dump_graphs_to_files
from torch.distributed._spmd.iter_graph_module import IterGraphModule

class GraphModuleTransformation:
    def __init__(
        self,
        *,
        enable_graph_optimization: bool = False,
        enable_inductor: bool = False,
        dump_graphs: bool = False,
    ) -> None:
        # 初始化函数，设定转换参数
        self.enable_graph_optimization = enable_graph_optimization
        self.enable_inductor = enable_inductor
        self.dump_graphs = dump_graphs

    def __call__(self, gm: fx.GraphModule) -> Callable:
        # 如果需要将图形输出到文件，则首先将原始图形模块打印为可读文本并存储
        if self.dump_graphs:
            graph_folder = dump_graphs_to_files(
                {"before_transformation_gm": gm.print_readable(False)}
            )
            # 启用图优化时，将图形文件夹路径传递给优化器
            enable_graph_optimization_dump(graph_folder)

        # 使用 IterGraphModule 包装图形模块
        iter_gm = IterGraphModule(gm, enable_inductor=self.enable_inductor)

        # 如果启用了图优化，执行以下优化步骤
        if self.enable_graph_optimization:
            # 将通信操作融合并拼接以提高效率
            comm_fusion_with_concat(iter_gm, 100)
            # 调度通信等待操作以优化执行顺序
            schedule_comm_wait(iter_gm)
            # 从优化器中移除复制操作以减少开销
            remove_copy_from_optimizer(iter_gm)

        # 完成图形模块设置的最后步骤，确保在移动图形之后调用
        iter_gm.finalize_setup()

        # 如果需要将迭代后的图形模块输出到文件
        if self.dump_graphs:
            dump_graphs_to_files(
                {
                    "iter_graph_setup_gm": iter_gm.setup_gm.print_readable(False),
                    "iter_graph_main_gm": iter_gm.main_gm.print_readable(False),
                    "iter_graph_cleanup_gm": iter_gm.cleanup_gm.print_readable(False),
                },
                graph_folder,  # 类型提示: 忽略可能未定义的变量警告
            )

        # 返回经过转换的迭代图形模块
        return iter_gm
```
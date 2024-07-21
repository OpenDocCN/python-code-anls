# `.\pytorch\test\distributed\pipelining\schedule_registry.py`

```
# 定义计算类型常量，分别表示前向计算、反向计算和权重更新
F = _ComputationType.FORWARD
B = _ComputationType.BACKWARD
W = _ComputationType.WEIGHT

# 定义一个多阶段流水线调度的基类，用于测试 torch.distributed.pipelining
class ScheduleVShaped(PipelineScheduleMulti):
    # 流水线阶段数为 4
    n_stages = 4
    # 定义每个组的阶段对应关系
    rank_stages = {
        0: [0, 3],
        1: [1, 2],
    }

    # 初始化方法，接收多个参数，包括阶段列表、微批次数、阶段索引到组排名的映射、可选的损失函数
    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        stage_index_to_group_rank: Dict[int, int],
        loss_fn: Optional[Callable] = None,
    ):
        # 调用父类的初始化方法
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            stage_index_to_group_rank=stage_index_to_group_rank,
        )

        # 定义流水线执行顺序的字典，包含两个组的执行顺序
        self.pipeline_order = {
            0: [
                _Action(F, 0, 0),  # 前向计算，阶段 0 的任务 0
                None,              # 空操作
                None,              # 空操作
                _Action(F, 0, 3),  # 前向计算，阶段 0 的任务 3
                _Action(B, 0, 3),  # 反向计算，阶段 0 的任务 3
                None,              # 空操作
                None,              # 空操作
                _Action(B, 0, 0),  # 反向计算，阶段 0 的任务 0
            ],
            1: [
                None,              # 空操作
                _Action(F, 0, 1),  # 前向计算，阶段 0 的任务 1
                _Action(F, 0, 2),  # 前向计算，阶段 0 的任务 2
                None,              # 空操作
                None,              # 空操作
                _Action(B, 0, 2),  # 反向计算，阶段 0 的任务 2
                _Action(B, 0, 1),  # 反向计算，阶段 0 的任务 1
                None,              # 空操作
            ],
        }

# 定义另一个多阶段流水线调度的子类
class ScheduleUnbalanced(PipelineScheduleMulti):
    # 流水线阶段数为 5
    n_stages = 5
    # 定义每个组的阶段对应关系
    rank_stages = {
        0: [0, 1, 4],
        1: [2, 3],
    }

    # 初始化方法，接收多个参数，包括阶段列表、微批次数、阶段索引到组排名的映射、可选的损失函数
    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        stage_index_to_group_rank: Dict[int, int],
        loss_fn: Optional[Callable] = None,
    ):
        # 调用父类的初始化方法
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            stage_index_to_group_rank=stage_index_to_group_rank,
        )

        # 定义流水线执行顺序的字典，包含两个组的执行顺序
        self.pipeline_order = {
            0: [
                _Action(F, 0, 0),  # 前向计算，阶段 0 的任务 0
                _Action(F, 0, 1),  # 前向计算，阶段 0 的任务 1
                None,              # 空操作
                None,              # 空操作
                _Action(F, 0, 4),  # 前向计算，阶段 0 的任务 4
                _Action(B, 0, 4),  # 反向计算，阶段 0 的任务 4
                None,              # 空操作
                None,              # 空操作
                _Action(B, 0, 1),  # 反向计算，阶段 0 的任务 1
                _Action(B, 0, 0),  # 反向计算，阶段 0 的任务 0
            ],
            1: [
                None,              # 空操作
                None,              # 空操作
                _Action(F, 0, 2),  # 前向计算，阶段 0 的任务 2
                _Action(F, 0, 3),  # 前向计算，阶段 0 的任务 3
                None,              # 空操作
                None,              # 空操作
                _Action(B, 0, 3),  # 反向计算，阶段 0 的任务 3
                _Action(B, 0, 2),  # 反向计算，阶段 0 的任务 2
                None,              # 空操作
                None,              # 空操作
            ],
        }
# 定义一个名为 ScheduleWithW 的类，继承自 PipelineScheduleMulti 类
class ScheduleWithW(PipelineScheduleMulti):
    # 定义类变量 n_stages，表示流水线的阶段数为 4
    n_stages = 4
    # 定义 rank_stages 字典，指定每个阶段对应的排名列表
    rank_stages = {
        0: [0, 2],
        1: [1, 3],
    }

    # 初始化方法，接受 stages（阶段列表）、n_microbatches（微批次数）、loss_fn（损失函数，可选）
    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
    ):
        # 调用父类的初始化方法，传入阶段列表、微批次数、损失函数
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
        )

        # 设置实例变量 use_full_backward 为 False，用于所有使用 "W" 的调度需要更新
        self.use_full_backward = False

        # 定义 pipeline_order 字典，描述流水线中每个阶段的操作顺序
        # 每个键对应一个阶段，值是操作动作 _Action 的列表
        self.pipeline_order = {
            0: [
                _Action(F, 0, 0),
                _Action(F, 1, 0),
                _Action(F, 0, 2),
                _Action(F, 1, 2),
                None,
                _Action(B, 0, 2),
                _Action(W, 0, 2),
                _Action(B, 0, 0),
                _Action(B, 1, 2),
                _Action(W, 0, 0),
                _Action(B, 1, 0),
                _Action(W, 1, 2),
                _Action(W, 1, 0),
            ],
            1: [
                None,
                _Action(F, 0, 1),
                _Action(F, 1, 1),
                _Action(F, 0, 3),
                _Action(B, 0, 3),
                _Action(F, 1, 3),
                _Action(B, 0, 1),
                _Action(B, 1, 3),
                _Action(W, 0, 3),
                _Action(B, 1, 1),
                _Action(W, 0, 1),
                _Action(W, 1, 3),
                _Action(W, 1, 1),
            ],
        }
```
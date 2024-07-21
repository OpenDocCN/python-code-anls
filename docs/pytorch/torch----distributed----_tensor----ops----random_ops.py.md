# `.\pytorch\torch\distributed\_tensor\ops\random_ops.py`

```
#`
# Copyright (c) Meta Platforms, Inc. and affiliates
# 导入 PyTorch 库
import torch
# 从 torch.distributed._tensor._op_schema 模块导入相关类
from torch.distributed._tensor._op_schema import (
    OpSchema,                     # 操作模式定义类
    OpStrategy,                   # 操作策略定义类
    PlacementStrategy,            # 放置策略定义类
    StrategyType,                 # 策略类型定义类
)
# 从 torch.distributed._tensor.ops.utils 模块导入辅助函数
from torch.distributed._tensor.ops.utils import is_tensor_partial, register_op_strategy
# 导入设备网格类
from torch.distributed.device_mesh import DeviceMesh

# 设置 aten 操作命名空间
aten = torch.ops.aten

# 使用装饰器注册一个操作策略
@register_op_strategy(
    [aten.normal_.default, aten.uniform_.default, aten.native_dropout.default]  # 注册的操作
)
# 定义一个函数，指定设备网格和操作模式作为参数，返回策略类型
def random_op_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    self_strategy = op_schema.args_schema[0]  # 获取操作参数的策略
    assert isinstance(self_strategy, OpStrategy)  # 确保参数是操作策略

    random_strategy = OpStrategy([])  # 创建一个空的操作策略对象
    for arg_strategy in self_strategy.strategies:  # 遍历参数策略
        arg_spec = arg_strategy.output_spec  # 获取参数的输出规格
        if is_tensor_partial(arg_spec):  # 判断输出规格是否为部分张量
            # TODO: 解决部分张量情况下，inplace 随机操作的行为
            raise RuntimeError(f"{op_schema.op} with Partial is not supported yet!")  # 抛出运行时异常
        random_strategy.strategies.append(PlacementStrategy(output_specs=arg_spec))  # 添加放置策略

    return random_strategy  # 返回创建的操作策略
```
# `.\lucidrains\q-transformer\q_transformer\mocks.py`

```
from random import randrange
# 从 random 模块中导入 randrange 函数

import torch
# 导入 torch 库
from torch.utils.data import Dataset
# 从 torch.utils.data 模块中导入 Dataset 类

from beartype.typing import Tuple, Optional
# 从 beartype.typing 模块中导入 Tuple 和 Optional 类型

from torchtyping import TensorType
# 从 torchtyping 模块中导入 TensorType 类型
from q_transformer.agent import BaseEnvironment
# 从 q_transformer.agent 模块中导入 BaseEnvironment 类

class MockEnvironment(BaseEnvironment):
    # 定义 MockEnvironment 类，继承自 BaseEnvironment 类
    def init(self) -> Tuple[
        Optional[str],
        TensorType[float]
    ]:
        # 初始化方法，返回一个元组，包含可选的字符串和浮点数张量
        return 'please clean the kitchen', torch.randn(self.state_shape, device = self.device)
        # 返回指令字符串和根据状态形状和设备生成的随机张量

    def forward(self, actions) -> Tuple[
        TensorType[(), float],
        TensorType[float],
        TensorType[(), bool]
    ]:
        # 前向传播方法，接受动作参数，返回一个元组，包含标量浮点数张量、浮点数张量和布尔值张量
        rewards = torch.randn((), device = self.device)
        # 生成一个随机标量浮点数张量
        next_states = torch.randn(self.state_shape, device = self.device)
        # 生成一个随机状态形状的浮点数张量
        done = torch.zeros((), device = self.device, dtype = torch.bool)
        # 生成一个全零张量，数据类型为布尔型

        return rewards, next_states, done
        # 返回奖励、下一个状态和完成标志

class MockReplayDataset(Dataset):
    # 定义 MockReplayDataset 类，继承自 Dataset 类
    def __init__(
        self,
        length = 10000,
        num_actions = 1,
        num_action_bins = 256,
        video_shape = (6, 224, 224)
    ):
        # 初始化方法，设置数据集长度、动作数量、动作区间数量和视频形状
        self.length = length
        self.num_actions = num_actions
        self.num_action_bins = num_action_bins
        self.video_shape = video_shape

    def __len__(self):
        # 返回数据集长度
        return self.length

    def __getitem__(self, _):
        # 获取数据集中的一项
        instruction = "please clean the kitchen"
        # 指令字符串
        state = torch.randn(3, *self.video_shape)
        # 随机生成一个状态张量

        if self.num_actions == 1:
            action = torch.tensor(randrange(self.num_action_bins + 1))
        else:
            action = torch.randint(0, self.num_action_bins + 1, (self.num_actions,))
        # 根据动作数量生成动作张量

        next_state = torch.randn(3, *self.video_shape)
        # 随机生成下一个状态张量
        reward = torch.tensor(randrange(2))
        # 随机生成奖励张量
        done = torch.tensor(randrange(2), dtype = torch.bool)
        # 随机生成完成标志张量

        return instruction, state, action, next_state, reward, done
        # 返回指令、状态、动作、下一个状态、奖励和完成标志

class MockReplayNStepDataset(Dataset):
    # 定义 MockReplayNStepDataset 类，继承自 Dataset 类
    def __init__(
        self,
        length = 10000,
        num_steps = 2,
        num_actions = 1,
        num_action_bins = 256,
        video_shape = (6, 224, 224)
    ):
        # 初始化方法，设置数据集长度、步数、动作数量、动作区间数量和视频形状
        self.num_steps = num_steps
        self.time_shape = (num_steps,)
        self.length = length
        self.num_actions = num_actions
        self.num_action_bins = num_action_bins
        self.video_shape = video_shape

    def __len__(self):
        # 返回数据集长度
        return self.length

    def __getitem__(self, _):
        # 获取数据集中的一项
        action_dims = (self.num_actions,) if self.num_actions > 1 else tuple()
        # 根据动作数量设置动作维度元组

        instruction = "please clean the kitchen"
        # 指令字符串
        state = torch.randn(*self.time_shape, 3, *self.video_shape)
        # 随机生成一个时间维度状态张量
        action = torch.randint(0, self.num_action_bins + 1, (*self.time_shape, *action_dims))
        # 根据动作数量生成动作张量
        next_state = torch.randn(3, *self.video_shape)
        # 随机生成下一个状态张量
        reward = torch.randint(0, 2, self.time_shape)
        # 随机生成奖励张量
        done = torch.zeros(self.time_shape, dtype = torch.bool)
        # 生成全零完成标志张量

        return instruction, state, action, next_state, reward, done
        # 返回指令、状态、动作、下一个状态、奖励和完成标志
```